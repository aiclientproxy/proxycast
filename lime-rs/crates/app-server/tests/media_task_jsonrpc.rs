use app_server::AppServer;
use app_server::EventLogWriter;
use app_server::LocalAppDataSource;
use app_server::MockBackend;
use app_server::ProjectionStore;
use app_server::RuntimeBackend;
use app_server::RuntimeCore;
use app_server::SidecarStore;
use app_server_protocol::protocol::v2::{METHOD_ITEM_COMPLETED, METHOD_TURN_COMPLETED};
use app_server_protocol::*;
use chrono::Utc;
use lime_core::database::dao::api_key_provider::{
    ApiKeyEntry, ApiKeyProvider, ApiKeyProviderDao, ApiProviderType, ProviderGroup,
};
use lime_core::database::schema::create_tables;
use lime_core::database::{lock_db, DbConnection};
use lime_core::models::model_registry::{
    ModelCapabilities, ModelModality, ModelRuntimeFeature, ModelTaskFamily,
    ProviderModelCapability, ProviderModelConfig,
};
use lime_services::api_key_provider_service::ApiKeyProviderService;
use rusqlite::Connection;
use serde_json::json;
use serde_json::Value;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::broadcast;
use tokio::time::{timeout, Duration};

struct MediaTaskAppServer {
    _temp: TempDir,
    event_log_writer: Arc<EventLogWriter>,
    sidecar_store: Arc<SidecarStore>,
    workspace_root: String,
    server: AppServer,
}

async fn read_fixture_http_request(stream: &mut tokio::net::TcpStream) -> (String, Value) {
    let mut bytes = Vec::new();
    let header_end = loop {
        let mut chunk = [0_u8; 1024];
        let read = stream.read(&mut chunk).await.expect("read fixture request");
        assert!(read > 0, "fixture request closed before headers completed");
        bytes.extend_from_slice(&chunk[..read]);
        if let Some(position) = bytes.windows(4).position(|item| item == b"\r\n\r\n") {
            break position + 4;
        }
    };
    let headers = String::from_utf8_lossy(&bytes[..header_end]).to_string();
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    while bytes.len() < header_end + content_length {
        let mut chunk = vec![0_u8; content_length.max(1024)];
        let read = stream
            .read(&mut chunk)
            .await
            .expect("read fixture request body");
        assert!(read > 0, "fixture request body ended early");
        bytes.extend_from_slice(&chunk[..read]);
    }
    let body = if content_length == 0 {
        Value::Null
    } else {
        serde_json::from_slice(&bytes[header_end..header_end + content_length])
            .expect("decode fixture request body")
    };
    (headers, body)
}

async fn write_fixture_json_response(stream: &mut tokio::net::TcpStream, body: Value) {
    let body = body.to_string();
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream
        .write_all(response.as_bytes())
        .await
        .expect("write fixture response");
}

async fn media_task_app_server() -> MediaTaskAppServer {
    build_media_task_app_server(|db| {
        insert_image_provider_with_key(db, "provider-image", "gpt-image-test");
    })
    .await
}

async fn build_media_task_app_server(
    configure_providers: impl FnOnce(&DbConnection),
) -> MediaTaskAppServer {
    let temp = TempDir::new().expect("create media task fixture temp dir");
    let data_root = temp.path().join("app-server-data");
    let workspace_root = temp.path().join("workspace").to_string_lossy().to_string();
    std::fs::create_dir_all(&workspace_root).expect("create workspace root");

    let conn = Connection::open_in_memory().expect("open in-memory product db");
    create_tables(&conn).expect("create product schema");
    let db = Arc::new(Mutex::new(conn));
    configure_providers(&db);
    let event_log_writer =
        Arc::new(EventLogWriter::new(temp.path().join("events")).expect("event log writer"));
    let sidecar_store =
        Arc::new(SidecarStore::new(temp.path().join("sidecars")).expect("sidecar store"));
    let app_data_source = LocalAppDataSource::initialize_with_roots(db, temp.path(), data_root)
        .await
        .expect("local app data source");
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_app_data_source(Arc::new(app_data_source))
        .with_event_log_writer(event_log_writer.clone())
        .with_sidecar_store(sidecar_store.clone());

    MediaTaskAppServer {
        _temp: temp,
        event_log_writer,
        sidecar_store,
        workspace_root,
        server: AppServer::with_runtime(runtime),
    }
}

#[tokio::test]
async fn video_task_create_executes_current_worker_from_public_jsonrpc() {
    let captured_request = Arc::new(Mutex::new(None::<String>));
    let captured_body = Arc::new(Mutex::new(None::<Value>));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind video provider");
    let address = listener.local_addr().expect("video provider address");
    let captured_request_for_server = Arc::clone(&captured_request);
    let captured_body_for_server = Arc::clone(&captured_body);
    let provider_server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept video request");
        let mut bytes = Vec::new();
        let mut header_end = None;
        loop {
            let mut chunk = [0_u8; 1024];
            let read = stream.read(&mut chunk).await.expect("read video request");
            if read == 0 {
                break;
            }
            bytes.extend_from_slice(&chunk[..read]);
            if let Some(position) = bytes.windows(4).position(|item| item == b"\r\n\r\n") {
                header_end = Some(position + 4);
                break;
            }
        }
        let header_end = header_end.expect("video request headers");
        let headers = String::from_utf8_lossy(&bytes[..header_end]).to_string();
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        while bytes.len() < header_end + content_length {
            let mut chunk = vec![0_u8; content_length.max(1024)];
            let read = stream
                .read(&mut chunk)
                .await
                .expect("read video request body");
            if read == 0 {
                break;
            }
            bytes.extend_from_slice(&chunk[..read]);
        }
        let body: Value = serde_json::from_slice(&bytes[header_end..header_end + content_length])
            .expect("decode video request body");
        *captured_request_for_server.lock().expect("capture request") = Some(headers);
        *captured_body_for_server.lock().expect("capture body") = Some(body);

        let response_body = json!({
            "data": [{
                "id": "video-result-1",
                "url": "https://cdn.example.test/video-result-1.mp4",
                "mime_type": "video/mp4",
                "duration": 6
            }]
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        stream
            .write_all(response.as_bytes())
            .await
            .expect("write video response");
    });

    let api_host = format!("http://{address}/v1/videos/generations");
    let app = build_media_task_app_server(|db| {
        insert_video_provider_with_key(db, "provider-video", "fal-ai/video-test", &api_host);
    })
    .await;
    initialize_server(&app.server, 1, "media-task-video-create-test").await;

    let created = request(
        &app.server,
        2,
        METHOD_MEDIA_TASK_ARTIFACT_VIDEO_CREATE,
        json!({
            "projectRootPath": app.workspace_root,
            "prompt": "生成一段青柠实验室视频",
            "providerId": "provider-video",
            "model": "fal-ai/video-test",
            "duration": 6,
            "aspectRatio": "16:9"
        }),
    )
    .await;
    let task_id = created
        .pointer("/result/task_id")
        .and_then(Value::as_str)
        .expect("created video task id")
        .to_string();
    assert_eq!(
        created.pointer("/result/record/payload/model_route_execution/executor/bindingKey"),
        Some(&json!("mediaTaskArtifact/video/create"))
    );

    let completed = timeout(Duration::from_secs(5), async {
        let mut request_id = 3;
        loop {
            let task = request(
                &app.server,
                request_id,
                METHOD_MEDIA_TASK_ARTIFACT_GET,
                json!({
                    "projectRootPath": app.workspace_root,
                    "taskRef": task_id
                }),
            )
            .await;
            request_id += 1;
            if matches!(
                task.pointer("/result/normalized_status")
                    .and_then(Value::as_str),
                Some("succeeded" | "failed" | "cancelled")
            ) {
                break task;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("video task terminal");

    assert_eq!(
        completed.pointer("/result/normalized_status"),
        Some(&json!("succeeded"))
    );
    assert_eq!(
        completed.pointer("/result/record/result/video/url"),
        Some(&json!("https://cdn.example.test/video-result-1.mp4"))
    );
    assert_eq!(
        captured_request
            .lock()
            .expect("captured request")
            .as_deref()
            .is_some_and(|headers| headers
                .lines()
                .any(|line| line.eq_ignore_ascii_case("authorization: Key test-key"))),
        true
    );
    assert_eq!(
        captured_body
            .lock()
            .expect("captured body")
            .as_ref()
            .and_then(|body| body.get("model")),
        Some(&json!("fal-ai/video-test"))
    );

    provider_server.abort();
}

#[tokio::test]
async fn xai_video_task_start_and_poll_run_from_public_jsonrpc() {
    let captured_requests = Arc::new(Mutex::new(Vec::<String>::new()));
    let captured_start_body = Arc::new(Mutex::new(None::<Value>));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind xAI video provider");
    let address = listener.local_addr().expect("xAI video provider address");
    let requests_for_server = Arc::clone(&captured_requests);
    let body_for_server = Arc::clone(&captured_start_body);
    let provider_server = tokio::spawn(async move {
        let (mut start_stream, _) = listener.accept().await.expect("accept xAI video start");
        let (start_headers, start_body) = read_fixture_http_request(&mut start_stream).await;
        requests_for_server
            .lock()
            .expect("capture xAI start")
            .push(start_headers);
        *body_for_server.lock().expect("capture xAI start body") = Some(start_body);
        write_fixture_json_response(
            &mut start_stream,
            json!({ "request_id": "xai-jsonrpc-request-1" }),
        )
        .await;

        let (mut poll_stream, _) = listener.accept().await.expect("accept xAI video poll");
        let (poll_headers, poll_body) = read_fixture_http_request(&mut poll_stream).await;
        assert!(poll_body.is_null());
        requests_for_server
            .lock()
            .expect("capture xAI poll")
            .push(poll_headers);
        write_fixture_json_response(
            &mut poll_stream,
            json!({
                "status": "done",
                "video": { "url": "https://cdn.example.test/xai-jsonrpc.mp4" }
            }),
        )
        .await;
    });

    let api_host = format!("http://{address}/v1");
    let app = build_media_task_app_server(|db| {
        insert_xai_video_provider_with_key(db, "xai-video", "grok-imagine-video", &api_host);
    })
    .await;
    initialize_server(&app.server, 1, "media-task-xai-video-create-test").await;

    let created = request(
        &app.server,
        2,
        METHOD_MEDIA_TASK_ARTIFACT_VIDEO_CREATE,
        json!({
            "projectRootPath": app.workspace_root,
            "prompt": "生成一段 Grok 青柠实验室视频",
            "providerId": "xai-video",
            "model": "grok-imagine-video",
            "duration": 6,
            "resolution": "720p",
            "aspectRatio": "16:9",
            "imageUrl": "https://example.test/lime.png"
        }),
    )
    .await;
    let task_id = created
        .pointer("/result/task_id")
        .and_then(Value::as_str)
        .expect("created xAI video task id")
        .to_string();
    assert_eq!(
        created.pointer("/result/record/payload/resolved_route/protocol"),
        Some(&json!("xai_video"))
    );

    let completed = timeout(Duration::from_secs(10), async {
        let mut request_id = 3;
        loop {
            let task = request(
                &app.server,
                request_id,
                METHOD_MEDIA_TASK_ARTIFACT_GET,
                json!({
                    "projectRootPath": app.workspace_root,
                    "taskRef": task_id
                }),
            )
            .await;
            request_id += 1;
            if matches!(
                task.pointer("/result/normalized_status")
                    .and_then(Value::as_str),
                Some("succeeded" | "failed" | "cancelled")
            ) {
                break task;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("xAI video task terminal");

    assert_eq!(
        completed.pointer("/result/normalized_status"),
        Some(&json!("succeeded")),
        "xAI video terminal response: {completed}"
    );
    assert_eq!(
        completed.pointer("/result/record/payload/provider_task/request_id"),
        Some(&json!("xai-jsonrpc-request-1"))
    );
    assert_eq!(
        completed.pointer("/result/record/result/video/url"),
        Some(&json!("https://cdn.example.test/xai-jsonrpc.mp4"))
    );

    let requests = captured_requests.lock().expect("captured xAI requests");
    assert_eq!(requests.len(), 2);
    assert!(requests[0].starts_with("POST /v1/videos/generations HTTP/1.1"));
    assert!(requests[1].starts_with("GET /v1/videos/xai-jsonrpc-request-1 HTTP/1.1"));
    assert!(requests.iter().all(|headers| headers
        .lines()
        .any(|line| line.eq_ignore_ascii_case("authorization: Bearer test-key"))));
    let body = captured_start_body
        .lock()
        .expect("captured xAI start body")
        .clone()
        .expect("xAI start body");
    assert_eq!(body["model"], "grok-imagine-video");
    assert_eq!(body["image"]["url"], "https://example.test/lime.png");

    provider_server.abort();
}

#[tokio::test]
async fn image_task_complete_uses_current_jsonrpc_method() {
    let app = media_task_app_server().await;
    initialize_server(&app.server, 1, "media-task-image-complete-test").await;

    let created = request(
        &app.server,
        2,
        METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE,
        json!({
            "projectRootPath": app.workspace_root,
            "prompt": "给春日咖啡活动生成一张配图",
            "size": "1024x1024",
            "count": 1,
            "providerId": "provider-image",
            "model": "gpt-image-test",
            "threadId": "thread-image-complete",
            "turnId": "turn-image-complete",
            "entrySource": "at_image_command"
        }),
    )
    .await;
    let task_id = created
        .pointer("/result/task_id")
        .and_then(Value::as_str)
        .expect("created task id")
        .to_string();
    assert_eq!(
        created.pointer("/result/normalized_status"),
        Some(&json!("pending"))
    );

    let completed = request(
        &app.server,
        3,
        METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE,
        json!({
            "projectRootPath": app.workspace_root,
            "taskRef": task_id,
            "providerId": "provider-image",
            "model": "gpt-image-test",
            "responseId": "response-image-complete",
            "images": [{
                "url": "file:///tmp/lime-image-complete.png",
                "revisedPrompt": "春日咖啡活动插画",
                "slotId": "hero",
                "slotIndex": 1,
                "slotPrompt": "主视觉配图"
            }]
        }),
    )
    .await;

    assert_eq!(
        completed.pointer("/result/normalized_status"),
        Some(&json!("succeeded"))
    );
    assert_eq!(
        completed.pointer("/result/record/result/images/0/url"),
        Some(&json!("file:///tmp/lime-image-complete.png"))
    );
    assert_eq!(
        completed.pointer("/result/record/result/received_count"),
        Some(&json!(1))
    );
    assert_eq!(
        completed.pointer("/result/record/payload/received_count"),
        Some(&json!(1))
    );
    assert_eq!(
        completed.pointer("/result/record/progress/percent"),
        Some(&json!(100))
    );
    assert_eq!(
        completed.pointer("/result/record/progress/preview_slots/0/status"),
        Some(&json!("complete"))
    );
    assert_eq!(
        completed.pointer("/result/record/attempts/0/worker_id"),
        Some(&json!("app-server-image-output-writer"))
    );
    assert_eq!(
        completed.pointer("/result/record/attempts/0/result_snapshot/images/0/url"),
        Some(&json!("file:///tmp/lime-image-complete.png"))
    );

    let restored = request(
        &app.server,
        4,
        METHOD_MEDIA_TASK_ARTIFACT_GET,
        json!({
            "projectRootPath": app.workspace_root,
            "taskRef": completed.pointer("/result/task_id").and_then(Value::as_str).expect("completed task id")
        }),
    )
    .await;
    assert_eq!(
        restored.pointer("/result/record/result/images/0/url"),
        Some(&json!("file:///tmp/lime-image-complete.png"))
    );
}

#[tokio::test]
async fn image_task_complete_writes_data_url_sidecar_via_jsonrpc() {
    let app = media_task_app_server().await;
    initialize_server(&app.server, 1, "media-task-image-complete-sidecar-test").await;

    let created = request(
        &app.server,
        2,
        METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE,
        json!({
            "projectRootPath": app.workspace_root,
            "prompt": "给春日咖啡活动生成一张可读 sidecar 的配图",
            "size": "1024x1024",
            "count": 1,
            "providerId": "provider-image",
            "model": "gpt-image-test",
            "sessionId": "session-image-complete-sidecar",
            "threadId": "thread-image-complete-sidecar",
            "turnId": "turn-image-complete-sidecar",
            "entrySource": "at_image_command"
        }),
    )
    .await;
    let task_id = created
        .pointer("/result/task_id")
        .and_then(Value::as_str)
        .expect("created task id")
        .to_string();

    let completed = request(
        &app.server,
        3,
        METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE,
        json!({
            "projectRootPath": app.workspace_root,
            "taskRef": task_id,
            "providerId": "provider-image",
            "model": "gpt-image-test",
            "responseId": "response-image-complete-sidecar",
            "images": [{
                "url": "data:image/png;base64,AAECAw==",
                "revisedPrompt": "春日咖啡活动插画",
                "slotId": "hero",
                "slotIndex": 1,
                "slotPrompt": "主视觉配图"
            }]
        }),
    )
    .await;

    let sidecar_ref = completed
        .pointer("/result/record/result/images/0/sidecarRef")
        .expect("sidecar ref");
    assert_eq!(sidecar_ref["kind"].as_str(), Some("media"));
    assert_eq!(sidecar_ref["mimeType"].as_str(), Some("image/png"));
    assert!(sidecar_ref["ref"]
        .as_str()
        .is_some_and(|value| value.starts_with("sidecar://media/")));
    let relative_path = sidecar_ref["relativePath"].as_str().expect("relative path");
    let sha256 = sidecar_ref["sha256"].as_str();
    let bytes = app
        .sidecar_store
        .read_bytes_verified(relative_path, sha256, 16)
        .expect("read sidecar bytes")
        .expect("sidecar bytes");
    assert_eq!(bytes.bytes, vec![0, 1, 2, 3]);
}

#[tokio::test]
async fn image_task_complete_rejects_wrong_task_type() {
    let app = media_task_app_server().await;
    initialize_server(&app.server, 1, "media-task-image-complete-wrong-type-test").await;

    let created = request(
        &app.server,
        2,
        METHOD_MEDIA_TASK_ARTIFACT_AUDIO_CREATE,
        json!({
            "projectRootPath": app.workspace_root,
            "sourceText": "给春日咖啡活动生成一段播报",
            "voice": "narrator"
        }),
    )
    .await;
    let task_id = created
        .pointer("/result/task_id")
        .and_then(Value::as_str)
        .expect("created audio task id");

    let rejected = request_error(
        &app.server,
        3,
        METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE,
        json!({
            "projectRootPath": app.workspace_root,
            "taskRef": task_id,
            "images": [{
                "url": "file:///tmp/wrong-type.png"
            }]
        }),
    )
    .await;
    assert!(rejected
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .contains("只能完成 image_generate 任务"));
}

#[tokio::test]
async fn image_task_complete_rejects_failed_or_cancelled_task() {
    let app = media_task_app_server().await;
    initialize_server(&app.server, 1, "media-task-image-complete-terminal-test").await;

    let created = request(
        &app.server,
        2,
        METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE,
        json!({
            "projectRootPath": app.workspace_root,
            "prompt": "生成一张会被取消的图片",
            "providerId": "provider-image",
            "model": "gpt-image-test"
        }),
    )
    .await;
    let task_id = created
        .pointer("/result/task_id")
        .and_then(Value::as_str)
        .expect("created image task id");

    request(
        &app.server,
        3,
        METHOD_MEDIA_TASK_ARTIFACT_CANCEL,
        json!({
            "projectRootPath": app.workspace_root,
            "taskRef": task_id
        }),
    )
    .await;

    let rejected = request_error(
        &app.server,
        4,
        METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE,
        json!({
            "projectRootPath": app.workspace_root,
            "taskRef": task_id,
            "images": [{
                "url": "file:///tmp/cancelled.png"
            }]
        }),
    )
    .await;
    assert!(rejected
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .contains("不能直接写回完成态"));
}

#[tokio::test]
async fn image_command_turn_start_creates_task_from_jsonrpc_metadata() {
    let app = image_command_app_server().await;
    initialize_server(&app.server, 1, "image-command-jsonrpc-test").await;

    let started = request_allowing_notifications(
        &app.server,
        2,
        METHOD_THREAD_START,
        json!({
            "cwd": app.workspace_root,
            "model": "gpt-image-test",
            "modelProvider": "provider-image"
        }),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("started thread id")
        .to_string();
    let session_id = started
        .pointer("/result/thread/sessionId")
        .and_then(Value::as_str)
        .expect("started session id")
        .to_string();

    let application_metadata = image_command_metadata(
        &app.workspace_root,
        "E2E 图片命令路由测试，请生成一张青柠插画",
        "@配图 E2E 图片命令路由测试，请生成一张青柠插画",
        "provider-image",
        "gpt-image-test",
    );
    let mut outbound = app.server.subscribe_outbound_messages();
    let messages = request_turn_with_notifications(
        &app.server,
        &mut outbound,
        3,
        &thread_id,
        json!({
            "threadId": thread_id,
            "input": [{
                "type": "text",
                "text": "@配图 E2E 图片命令路由测试，请生成一张青柠插画"
            }],
            "additionalContext": {
                "metadata": {
                    "kind": "application",
                    "value": application_metadata.to_string()
                }
            },
            "cwd": app.workspace_root,
            "model": "gpt-image-test"
        }),
    )
    .await;

    let event_types = notification_event_types(&messages);
    assert!(
        event_types.contains(&"runtime.status"),
        "image command should emit accepted status: {event_types:?}"
    );
    assert!(
        event_types.contains(&"image_task.presentation.generated"),
        "image command should surface provided presentation before creating task: {event_types:?}"
    );
    assert!(
        event_types.contains(&"image_task.created"),
        "image command should create task: {event_types:?}"
    );
    assert_dynamic_tool_completed(&messages, "completed", true);
    assert!(
        !event_types.contains(&"routing.decision.made"),
        "image command must not fall through to ordinary chat routing: {event_types:?}"
    );
    assert!(
        !event_types
            .iter()
            .any(|event_type| event_type.starts_with("workflow.")),
        "workflow audit events should stay out of user-visible session stream: {event_types:?}"
    );
    let workflow_events = app
        .event_log_writer
        .read_session_workflow_audit_events(&session_id)
        .expect("workflow audit events");
    let workflow_event_types = workflow_events
        .iter()
        .map(|record| record.event.event_type.as_str())
        .collect::<Vec<_>>();
    assert!(
        workflow_event_types.contains(&"workflow.step.completed"),
        "image command should audit completed workflow steps: {workflow_event_types:?}"
    );
    assert!(
        workflow_event_types.contains(&"workflow.run.completed"),
        "image command should audit workflow completion before turn terminal: {workflow_event_types:?}"
    );

    let tasks = request(
        &app.server,
        4,
        METHOD_MEDIA_TASK_ARTIFACT_LIST,
        json!({
            "projectRootPath": app.workspace_root,
            "taskType": "image_generate",
            "limit": 20
        }),
    )
    .await;
    assert_eq!(
        tasks.pointer("/result/tasks/0/record/payload/prompt"),
        Some(&json!("E2E 图片命令路由测试，请生成一张青柠插画"))
    );
    assert_eq!(
        tasks.pointer("/result/tasks/0/record/payload/provider_id"),
        Some(&json!("provider-image"))
    );
    assert_eq!(
        tasks.pointer("/result/tasks/0/record/payload/entry_source"),
        Some(&json!("at_image_command"))
    );
}

#[tokio::test]
async fn image_command_turn_start_rejects_missing_explicit_provider_before_task_write() {
    let app = image_command_app_server().await;
    initialize_server(&app.server, 1, "image-command-jsonrpc-stale-provider-test").await;

    let started = request_allowing_notifications(
        &app.server,
        2,
        METHOD_THREAD_START,
        json!({
            "cwd": app.workspace_root,
            "model": "gpt-image-test",
            "modelProvider": "provider-image"
        }),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("started thread id")
        .to_string();

    let raw_text = "@配图 stale provider 回归，请生成一张青柠插画";
    let prompt = "stale provider 回归，请生成一张青柠插画";
    let metadata = image_command_metadata(
        &app.workspace_root,
        prompt,
        raw_text,
        "deleted-provider",
        "gpt-image-test",
    );
    let mut outbound = app.server.subscribe_outbound_messages();
    let messages = request_turn_with_notifications(
        &app.server,
        &mut outbound,
        3,
        &thread_id,
        json!({
            "threadId": thread_id,
            "input": [{
                "type": "text",
                "text": raw_text
            }],
            "additionalContext": {
                "metadata": {
                    "kind": "application",
                    "value": metadata.to_string()
                }
            },
            "cwd": app.workspace_root,
            "model": "gpt-image-test"
        }),
    )
    .await;

    let event_types = notification_event_types(&messages);
    assert!(
        event_types.contains(&"image_task.create_failed"),
        "stale provider should fail during task creation: {event_types:?}"
    );
    assert_dynamic_tool_completed(&messages, "failed", false);
    assert!(
        !event_types.contains(&"image_task.created"),
        "stale provider must not create a task: {event_types:?}"
    );
    assert!(
        !event_types.contains(&"image_task.failed"),
        "preflight failure must not be deferred to worker failure: {event_types:?}"
    );

    let tasks = request(
        &app.server,
        4,
        METHOD_MEDIA_TASK_ARTIFACT_LIST,
        json!({
            "projectRootPath": app.workspace_root,
            "taskType": "image_generate",
            "limit": 20
        }),
    )
    .await;
    assert_eq!(
        tasks.pointer("/result/tasks"),
        Some(&json!([])),
        "stale explicit provider should fail before any image task is written"
    );
}

async fn initialize_server(server: &AppServer, id: u64, client_name: &str) {
    let initialize = request(
        server,
        id,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": client_name,
                "version": "1.0.0"
            }
        }),
    )
    .await;
    assert_eq!(
        initialize.pointer("/result/serverInfo/protocolVersion"),
        Some(&json!(PROTOCOL_VERSION)),
    );
    notify(server, METHOD_INITIALIZED, json!({})).await;
}

fn image_command_metadata(
    workspace_root: &str,
    prompt: &str,
    raw_text: &str,
    provider_id: &str,
    model: &str,
) -> Value {
    json!({
        "harness": {
            "image_command_intent": {
                "kind": "image_task",
                "image_task": {
                    "project_root_path": workspace_root,
                    "prompt": prompt,
                    "raw_text": raw_text,
                    "mode": "generate",
                    "count": 1,
                    "provider_id": provider_id,
                    "model": model,
                    "executor_mode": "images_api",
                    "entry_source": "at_image_command",
                    "presentation": {
                        "assistant_intro": "好啊，我来按青柠插画的清爽方向处理。",
                        "planning_summary": "用明亮绿色、简洁构图和轻盈质感组织画面。",
                        "completion_caption": "完成了，青柠插画的清爽层次已经生成。"
                    }
                }
            }
        }
    })
}

async fn image_command_app_server() -> MediaTaskAppServer {
    let temp = TempDir::new().expect("create image command fixture temp dir");
    let data_root = temp.path().join("app-server-data");
    let workspace_root = temp.path().join("workspace").to_string_lossy().to_string();
    std::fs::create_dir_all(&workspace_root).expect("create workspace root");

    let conn = Connection::open_in_memory().expect("open in-memory product db");
    create_tables(&conn).expect("create product schema");
    let db = Arc::new(Mutex::new(conn));
    insert_image_provider_with_key(&db, "provider-image", "gpt-image-test");
    let event_log_writer =
        Arc::new(EventLogWriter::new(temp.path().join("events")).expect("event log writer"));
    let sidecar_store =
        Arc::new(SidecarStore::new(temp.path().join("sidecars")).expect("sidecar store"));
    let app_data_source =
        LocalAppDataSource::initialize_with_roots(db.clone(), temp.path(), data_root)
            .await
            .expect("local app data source");
    let runtime = RuntimeCore::with_backend(Arc::new(RuntimeBackend::with_db(db)))
        .with_projection_store(Arc::new(
            ProjectionStore::initialize(temp.path().join("projection.sqlite"))
                .expect("image command projection store"),
        ))
        .with_app_data_source(Arc::new(app_data_source))
        .with_event_log_writer(event_log_writer.clone())
        .with_sidecar_store(sidecar_store.clone());

    MediaTaskAppServer {
        _temp: temp,
        event_log_writer,
        sidecar_store,
        workspace_root,
        server: AppServer::with_runtime(runtime),
    }
}

fn insert_image_provider_with_key(db: &DbConnection, provider_id: &str, model: &str) {
    let now = Utc::now();
    let provider = ApiKeyProvider {
        id: provider_id.to_string(),
        name: provider_id.to_string(),
        provider_type: ApiProviderType::Openai,
        api_host: "https://api.openai.com/v1".to_string(),
        is_system: false,
        group: ProviderGroup::Custom,
        enabled: true,
        sort_order: 1,
        api_version: None,
        project: None,
        location: None,
        region: None,
        models: vec![ProviderModelConfig {
            id: model.to_string(),
            display_name: None,
            capability: Some(ProviderModelCapability {
                task_families: vec![ModelTaskFamily::ImageGeneration],
                input_modalities: vec![ModelModality::Text],
                output_modalities: vec![ModelModality::Image],
                runtime_features: vec![ModelRuntimeFeature::ImagesApi],
                capabilities: ModelCapabilities::default(),
            }),
        }],
        prompt_cache_mode: None,
        created_at: now,
        updated_at: now,
    };
    let key = ApiKeyEntry {
        id: format!("{provider_id}-key"),
        provider_id: provider_id.to_string(),
        api_key_encrypted: ApiKeyProviderService::new().encrypt_api_key("test-key"),
        alias: None,
        enabled: true,
        usage_count: 0,
        error_count: 0,
        last_used_at: None,
        created_at: now,
    };
    let conn = lock_db(db).expect("lock product db");
    ApiKeyProviderDao::insert_provider(&conn, &provider).expect("insert image provider");
    ApiKeyProviderDao::insert_api_key(&conn, &key).expect("insert image provider api key");
}

fn insert_video_provider_with_key(
    db: &DbConnection,
    provider_id: &str,
    model: &str,
    api_host: &str,
) {
    let now = Utc::now();
    let provider = ApiKeyProvider {
        id: provider_id.to_string(),
        name: provider_id.to_string(),
        provider_type: ApiProviderType::Fal,
        api_host: api_host.to_string(),
        is_system: false,
        group: ProviderGroup::Custom,
        enabled: true,
        sort_order: 1,
        api_version: None,
        project: None,
        location: None,
        region: None,
        models: vec![ProviderModelConfig {
            id: model.to_string(),
            display_name: None,
            capability: Some(ProviderModelCapability {
                task_families: vec![ModelTaskFamily::VideoGeneration],
                input_modalities: vec![ModelModality::Text, ModelModality::Image],
                output_modalities: vec![ModelModality::Video],
                runtime_features: Vec::new(),
                capabilities: ModelCapabilities::default(),
            }),
        }],
        prompt_cache_mode: None,
        created_at: now,
        updated_at: now,
    };
    let key = ApiKeyEntry {
        id: format!("{provider_id}-key"),
        provider_id: provider_id.to_string(),
        api_key_encrypted: ApiKeyProviderService::new().encrypt_api_key("test-key"),
        alias: None,
        enabled: true,
        usage_count: 0,
        error_count: 0,
        last_used_at: None,
        created_at: now,
    };
    let conn = lock_db(db).expect("lock product db");
    ApiKeyProviderDao::insert_provider(&conn, &provider).expect("insert video provider");
    ApiKeyProviderDao::insert_api_key(&conn, &key).expect("insert video provider api key");
}

fn insert_xai_video_provider_with_key(
    db: &DbConnection,
    provider_id: &str,
    model: &str,
    api_host: &str,
) {
    let now = Utc::now();
    let provider = ApiKeyProvider {
        id: provider_id.to_string(),
        name: provider_id.to_string(),
        provider_type: ApiProviderType::Openai,
        api_host: api_host.to_string(),
        is_system: false,
        group: ProviderGroup::Custom,
        enabled: true,
        sort_order: 1,
        api_version: None,
        project: None,
        location: None,
        region: None,
        models: vec![ProviderModelConfig {
            id: model.to_string(),
            display_name: None,
            capability: Some(ProviderModelCapability {
                task_families: vec![ModelTaskFamily::VideoGeneration],
                input_modalities: vec![ModelModality::Text, ModelModality::Image],
                output_modalities: vec![ModelModality::Video],
                runtime_features: Vec::new(),
                capabilities: ModelCapabilities::default(),
            }),
        }],
        prompt_cache_mode: None,
        created_at: now,
        updated_at: now,
    };
    let key = ApiKeyEntry {
        id: format!("{provider_id}-key"),
        provider_id: provider_id.to_string(),
        api_key_encrypted: ApiKeyProviderService::new().encrypt_api_key("test-key"),
        alias: None,
        enabled: true,
        usage_count: 0,
        error_count: 0,
        last_used_at: None,
        created_at: now,
    };
    let conn = lock_db(db).expect("lock product db");
    ApiKeyProviderDao::insert_provider(&conn, &provider).expect("insert xAI video provider");
    ApiKeyProviderDao::insert_api_key(&conn, &key).expect("insert xAI video provider api key");
}

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let lines = server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            })
            .to_string(),
        )
        .await
        .expect("handle JSON-RPC request");
    assert_eq!(
        lines.len(),
        1,
        "{method} should return exactly one response"
    );
    let response: Value = serde_json::from_str(&lines[0]).expect("decode JSON-RPC response");
    if let Some(error) = response.get("error") {
        panic!("{method} failed: {error}");
    }
    assert_eq!(response.get("id"), Some(&json!(id)));
    response
}

async fn request_allowing_notifications(
    server: &AppServer,
    id: u64,
    method: &str,
    params: Value,
) -> Value {
    let lines = server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            })
            .to_string(),
        )
        .await
        .expect("handle JSON-RPC request");
    let response = lines
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("decode JSON-RPC message"))
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("{method} should include response id {id}"));
    if let Some(error) = response.get("error") {
        panic!("{method} failed: {error}");
    }
    response
}

async fn request_turn_with_notifications(
    server: &AppServer,
    outbound: &mut broadcast::Receiver<JsonRpcMessage>,
    id: u64,
    thread_id: &str,
    params: Value,
) -> Vec<Value> {
    let response = request(server, id, METHOD_TURN_START, params).await;
    let turn_id = response
        .pointer("/result/turn/id")
        .and_then(Value::as_str)
        .expect("turn/start response turn id")
        .to_string();

    timeout(Duration::from_secs(5), async {
        let mut messages = Vec::new();
        loop {
            let message = outbound.recv().await.expect("outbound turn notification");
            let JsonRpcMessage::Notification(notification) = message else {
                continue;
            };
            let completed = notification.method == METHOD_TURN_COMPLETED
                && notification
                    .params
                    .as_ref()
                    .and_then(|params| params.pointer("/threadId"))
                    .and_then(Value::as_str)
                    == Some(thread_id)
                && notification
                    .params
                    .as_ref()
                    .and_then(|params| params.pointer("/turn/id"))
                    .and_then(Value::as_str)
                    == Some(turn_id.as_str());
            messages.push(
                serde_json::to_value(JsonRpcMessage::Notification(notification))
                    .expect("encode outbound turn notification"),
            );
            if completed {
                return messages;
            }
        }
    })
    .await
    .expect("timed out waiting for turn/completed")
}

fn notification_event_types(messages: &[Value]) -> Vec<&str> {
    messages
        .iter()
        .filter(|message| message.get("method") == Some(&json!(METHOD_AGENT_SESSION_EVENT)))
        .filter_map(|message| {
            message
                .pointer("/params/event/type")
                .and_then(Value::as_str)
        })
        .collect()
}

fn assert_dynamic_tool_completed(messages: &[Value], status: &str, success: bool) {
    assert!(
        messages.iter().any(|message| {
            message.get("method") == Some(&json!(METHOD_ITEM_COMPLETED))
                && message.pointer("/params/item/type") == Some(&json!("dynamicToolCall"))
                && message.pointer("/params/item/tool")
                    == Some(&json!("lime_create_image_generation_task"))
                && message.pointer("/params/item/status") == Some(&json!(status))
                && message.pointer("/params/item/success") == Some(&json!(success))
        }),
        "image command should expose a {status} dynamic tool terminal: {messages:#?}"
    );
}

async fn request_error(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let lines = server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            })
            .to_string(),
        )
        .await
        .expect("handle JSON-RPC request");
    assert_eq!(
        lines.len(),
        1,
        "{method} should return exactly one response"
    );
    let response: Value = serde_json::from_str(&lines[0]).expect("decode JSON-RPC response");
    assert!(
        response.get("error").is_some(),
        "{method} should return an error response"
    );
    assert_eq!(response.get("id"), Some(&json!(id)));
    response
}

async fn notify(server: &AppServer, method: &str, params: Value) {
    let lines = server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            })
            .to_string(),
        )
        .await
        .expect("handle JSON-RPC notification");
    assert!(
        lines.is_empty(),
        "{method} notification should not return responses"
    );
}
