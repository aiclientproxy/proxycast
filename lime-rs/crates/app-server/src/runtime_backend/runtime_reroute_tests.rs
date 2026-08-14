use super::*;
use crate::runtime::RuntimeHostContext;
use crate::ExecutionBackend;
use app_server_protocol::{
    AgentSession, AgentSessionStatus, AgentTurn, AgentTurnStatus, RuntimeOptions, RuntimeRequest,
};
use lime_core::database::dao::api_key_provider::ApiProviderType;
use lime_core::database::schema::create_tables;
use rusqlite::Connection;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};

const PRIMARY_MODEL: &str = "gpt-4.1-mini";
const BACKUP_MODEL: &str = "openai/gpt-4.1-mini";

#[derive(Default)]
struct RecordingSink {
    events: Vec<RuntimeEvent>,
}

impl RuntimeEventSink for RecordingSink {
    fn emit(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        self.events.push(event);
        Ok(())
    }
}

struct RuntimeRerouteFixture {
    base_url: String,
    models: Arc<Mutex<Vec<String>>>,
    server_task: JoinHandle<()>,
}

struct CredentialRerouteFixture {
    base_url: String,
    failed_key: Arc<Mutex<Option<String>>>,
    retry_after: Arc<Mutex<Option<u64>>>,
    requests: Arc<Mutex<Vec<(String, String)>>>,
    server_task: JoinHandle<()>,
}

impl CredentialRerouteFixture {
    async fn start() -> Self {
        Self::start_for_successes(1).await
    }

    async fn start_for_successes(expected_successes: usize) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind credential reroute fixture");
        let address = listener.local_addr().expect("fixture address");
        let failed_key = Arc::new(Mutex::new(None::<String>));
        let server_failed_key = Arc::clone(&failed_key);
        let retry_after = Arc::new(Mutex::new(None::<u64>));
        let server_retry_after = Arc::clone(&retry_after);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let server_requests = Arc::clone(&requests);
        let server_task = tokio::spawn(async move {
            let mut successful_responses = 0;
            for _ in 0..8 {
                let (mut stream, _) = listener.accept().await.expect("accept provider request");
                let (headers, body) = read_request(&mut stream)
                    .await
                    .expect("read provider request");
                let authorization = header_value(&headers, "authorization").unwrap_or_default();
                let model = body
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                server_requests
                    .lock()
                    .expect("record credential request")
                    .push((authorization.clone(), model.clone()));
                let rejected = server_failed_key
                    .lock()
                    .expect("failed credential key")
                    .as_ref()
                    .is_some_and(|failed_key| authorization == format!("Bearer {failed_key}"));
                if rejected {
                    let retry_after = *server_retry_after
                        .lock()
                        .expect("failed credential retry-after");
                    let status = if retry_after.is_some() {
                        "429 Too Many Requests"
                    } else {
                        "401 Unauthorized"
                    };
                    let extra_headers = retry_after
                        .map(|seconds| format!("retry-after: {seconds}\r\n"))
                        .unwrap_or_default();
                    write_response_with_headers(
                        &mut stream,
                        status,
                        "application/json",
                        &extra_headers,
                        &json!({ "error": { "message": "credential rejected" } }).to_string(),
                    )
                    .await;
                    continue;
                }

                let body = format!(
                    "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                    json!({
                        "id": "chatcmpl-credential-reroute",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": model.clone(),
                        "choices": [{
                            "index": 0,
                            "delta": { "content": "credential reroute answer" },
                            "finish_reason": null
                        }]
                    }),
                    json!({
                        "id": "chatcmpl-credential-reroute",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 3,
                            "total_tokens": 5
                        }
                    })
                );
                write_response(&mut stream, "200 OK", "text/event-stream", &body).await;
                successful_responses += 1;
                if successful_responses >= expected_successes {
                    break;
                }
            }
        });
        Self {
            base_url: format!("http://{address}/v1"),
            failed_key,
            retry_after,
            requests,
            server_task,
        }
    }

    fn reject(&self, api_key: &str) {
        *self.failed_key.lock().expect("set failed credential key") = Some(api_key.to_string());
    }

    fn reject_with_retry_after(&self, api_key: &str, retry_after_seconds: u64) {
        self.reject(api_key);
        *self.retry_after.lock().expect("set retry-after") = Some(retry_after_seconds);
    }
}

impl Drop for CredentialRerouteFixture {
    fn drop(&mut self) {
        self.server_task.abort();
    }
}

impl RuntimeRerouteFixture {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind runtime reroute fixture");
        let address = listener.local_addr().expect("fixture address");
        let models = Arc::new(Mutex::new(Vec::new()));
        let server_models = Arc::clone(&models);
        let server_task = tokio::spawn(async move {
            for _ in 0..8 {
                let (mut stream, _) = listener.accept().await.expect("accept provider request");
                let body = read_request_body(&mut stream)
                    .await
                    .expect("read provider request");
                let model = body
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                server_models
                    .lock()
                    .expect("record provider model")
                    .push(model.clone());
                if model == PRIMARY_MODEL {
                    write_response(
                        &mut stream,
                        "503 Service Unavailable",
                        "application/json",
                        &json!({ "error": { "message": "primary unavailable" } }).to_string(),
                    )
                    .await;
                    continue;
                }

                let body = format!(
                    "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                    json!({
                        "id": "chatcmpl-runtime-reroute",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": BACKUP_MODEL,
                        "choices": [{
                            "index": 0,
                            "delta": { "content": "backup answer" },
                            "finish_reason": null
                        }]
                    }),
                    json!({
                        "id": "chatcmpl-runtime-reroute",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": BACKUP_MODEL,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 2,
                            "total_tokens": 4
                        }
                    })
                );
                write_response(&mut stream, "200 OK", "text/event-stream", &body).await;
                break;
            }
        });
        Self {
            base_url: format!("http://{address}/v1"),
            models,
            server_task,
        }
    }
}

impl Drop for RuntimeRerouteFixture {
    fn drop(&mut self) {
        self.server_task.abort();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exhausted_retryable_credential_failure_does_not_cross_model_route() {
    let fixture = RuntimeRerouteFixture::start().await;
    let db = test_db();
    let backend = RuntimeBackend::with_db(db.clone());
    backend
        .api_key_provider_service
        .initialize_system_providers(&db)
        .expect("initialize providers");
    let primary_provider = add_provider(
        &backend.api_key_provider_service,
        &db,
        "openai",
        "Primary",
        &fixture.base_url,
        PRIMARY_MODEL,
    );
    let backup_provider = add_provider(
        &backend.api_key_provider_service,
        &db,
        "openrouter",
        "Backup",
        &fixture.base_url,
        BACKUP_MODEL,
    );
    let request = execution_request(&primary_provider, &backup_provider);
    let mut sink = RecordingSink::default();

    let error = timeout(
        Duration::from_secs(15),
        ExecutionBackend::start_turn(&backend, request, &mut sink),
    )
    .await
    .expect("runtime reroute timeout")
    .expect_err("exhausted credential failure must not use another model route");

    assert!(matches!(error, RuntimeCoreError::Backend(_)));
    let models = fixture.models.lock().expect("provider request models");
    assert!(!models.is_empty());
    assert!(models.iter().all(|model| model == PRIMARY_MODEL));
    assert!(sink.events.iter().all(|event| {
        event.event_type != "routing.fallback.applied"
            || event.payload["selectedProvider"] != backup_provider
    }));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn authentication_failure_switches_key_without_changing_model_route() {
    let fixture = CredentialRerouteFixture::start().await;
    let db = test_db();
    let backend = RuntimeBackend::with_db(db.clone());
    backend
        .api_key_provider_service
        .initialize_system_providers(&db)
        .expect("initialize providers");
    let primary = backend
        .api_key_provider_service
        .update_provider(
            &db,
            "openai",
            Some("Credential Primary".to_string()),
            Some(ApiProviderType::Openai),
            Some(fixture.base_url.clone()),
            Some(true),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(vec![
                lime_core::models::model_registry::ProviderModelConfig::hint(PRIMARY_MODEL),
            ]),
        )
        .expect("configure credential provider");
    let key_a = backend
        .api_key_provider_service
        .add_api_key(&db, &primary.id, "credential-key-a", None, false)
        .expect("add credential key A");
    let key_b = backend
        .api_key_provider_service
        .add_api_key(&db, &primary.id, "credential-key-b", None, false)
        .expect("add credential key B");
    let probe = backend
        .api_key_provider_service
        .select_credential_for_provider(&db, &primary.id, Some(&primary.id), None)
        .await
        .expect("probe round-robin order")
        .expect("probe credential");
    let ref_a = lime_core::models::runtime_api_key_credential_uuid(&key_a.id);
    let ref_b = lime_core::models::runtime_api_key_credential_uuid(&key_b.id);
    let (failed_ref, failed_key, succeeding_ref, succeeding_key) = if probe.uuid == ref_a {
        (&ref_b, "credential-key-b", &ref_a, "credential-key-a")
    } else {
        (&ref_a, "credential-key-a", &ref_b, "credential-key-b")
    };
    fixture.reject(failed_key);

    let backup = add_provider(
        &backend.api_key_provider_service,
        &db,
        "openrouter",
        "Backup",
        &fixture.base_url,
        BACKUP_MODEL,
    );
    let request = execution_request(&primary.id, &backup);
    let mut sink = RecordingSink::default();

    timeout(
        Duration::from_secs(15),
        ExecutionBackend::start_turn(&backend, request, &mut sink),
    )
    .await
    .expect("credential reroute timeout")
    .expect("credential reroute turn");

    timeout(Duration::from_secs(2), async {
        while !fixture.server_task.is_finished() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("credential fixture completion");
    let requests = fixture.requests.lock().expect("credential requests");
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].0, format!("Bearer {failed_key}"));
    assert_eq!(requests[1].0, format!("Bearer {succeeding_key}"));
    assert!(requests.iter().all(|(_, model)| model == PRIMARY_MODEL));
    assert!(sink
        .events
        .iter()
        .any(|event| event.event_type == "turn.completed"));
    assert!(sink.events.iter().all(|event| {
        let payload = event.payload.to_string();
        !payload.contains(failed_ref) && !payload.contains(succeeding_ref)
    }));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retry_after_cooldown_skips_failed_credential_on_next_turn() {
    let fixture = CredentialRerouteFixture::start_for_successes(2).await;
    let db = test_db();
    let backend = RuntimeBackend::with_db(db.clone());
    backend
        .api_key_provider_service
        .initialize_system_providers(&db)
        .expect("initialize providers");
    let primary = backend
        .api_key_provider_service
        .update_provider(
            &db,
            "openai",
            Some("Cooldown Primary".to_string()),
            Some(ApiProviderType::Openai),
            Some(fixture.base_url.clone()),
            Some(true),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(vec![
                lime_core::models::model_registry::ProviderModelConfig::hint(PRIMARY_MODEL),
            ]),
        )
        .expect("configure cooldown provider");
    let key_a = backend
        .api_key_provider_service
        .add_api_key(&db, &primary.id, "cooldown-key-a", None, false)
        .expect("add cooldown key A");
    let _key_b = backend
        .api_key_provider_service
        .add_api_key(&db, &primary.id, "cooldown-key-b", None, false)
        .expect("add cooldown key B");
    let probe = backend
        .api_key_provider_service
        .select_credential_for_provider(&db, &primary.id, Some(&primary.id), None)
        .await
        .expect("probe round-robin order")
        .expect("probe credential");
    let ref_a = lime_core::models::runtime_api_key_credential_uuid(&key_a.id);
    let (failed_key, succeeding_key) = if probe.uuid == ref_a {
        ("cooldown-key-b", "cooldown-key-a")
    } else {
        ("cooldown-key-a", "cooldown-key-b")
    };
    fixture.reject_with_retry_after(failed_key, 60);
    let backup = add_provider(
        &backend.api_key_provider_service,
        &db,
        "openrouter",
        "Cooldown Backup",
        &fixture.base_url,
        BACKUP_MODEL,
    );

    let mut first_sink = RecordingSink::default();
    timeout(
        Duration::from_secs(15),
        ExecutionBackend::start_turn(
            &backend,
            execution_request(&primary.id, &backup),
            &mut first_sink,
        ),
    )
    .await
    .expect("first cooldown turn timeout")
    .expect("first cooldown turn");

    let mut second_request = execution_request(&primary.id, &backup);
    second_request.session.session_id = "session-runtime-cooldown-next".to_string();
    second_request.session.thread_id = "thread-runtime-cooldown-next".to_string();
    second_request.turn.turn_id = "turn-runtime-cooldown-next".to_string();
    second_request.turn.session_id = second_request.session.session_id.clone();
    second_request.turn.thread_id = second_request.session.thread_id.clone();
    let mut second_sink = RecordingSink::default();
    timeout(
        Duration::from_secs(15),
        ExecutionBackend::start_turn(&backend, second_request, &mut second_sink),
    )
    .await
    .expect("second cooldown turn timeout")
    .expect("second cooldown turn");

    timeout(Duration::from_secs(2), async {
        while !fixture.server_task.is_finished() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("cooldown fixture completion");
    let requests = fixture.requests.lock().expect("cooldown requests");
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].0, format!("Bearer {failed_key}"));
    assert_eq!(requests[1].0, format!("Bearer {succeeding_key}"));
    assert_eq!(requests[2].0, format!("Bearer {succeeding_key}"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exhausted_credentials_return_original_error_without_route_fallback() {
    let fixture = CredentialRerouteFixture::start().await;
    let db = test_db();
    let backend = RuntimeBackend::with_db(db.clone());
    backend
        .api_key_provider_service
        .initialize_system_providers(&db)
        .expect("initialize providers");
    let primary = add_provider(
        &backend.api_key_provider_service,
        &db,
        "openai",
        "OnlyCredential",
        &fixture.base_url,
        PRIMARY_MODEL,
    );
    fixture.reject("OnlyCredential-key");
    let backup = add_provider(
        &backend.api_key_provider_service,
        &db,
        "openrouter",
        "Backup",
        &fixture.base_url,
        BACKUP_MODEL,
    );
    let mut sink = RecordingSink::default();

    let result = timeout(
        Duration::from_secs(15),
        ExecutionBackend::start_turn(&backend, execution_request(&primary, &backup), &mut sink),
    )
    .await
    .expect("credential exhaustion timeout");
    let requests = fixture
        .requests
        .lock()
        .expect("credential requests")
        .clone();
    let event_types = sink
        .events
        .iter()
        .map(|event| event.event_type.clone())
        .collect::<Vec<_>>();
    let error = match result {
        Err(error) => error,
        Ok(()) => panic!(
            "credential exhaustion unexpectedly completed: requests={requests:?}, events={event_types:?}"
        ),
    };

    let RuntimeCoreError::Backend(message) = error else {
        panic!("credential exhaustion must preserve the provider failure: {error:?}");
    };
    assert_ne!(message, "resolved_credential_unavailable");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].1, PRIMARY_MODEL);
    assert!(sink.events.iter().all(|event| {
        event.event_type != "routing.fallback.applied"
            || event.payload["selectedModel"] != BACKUP_MODEL
    }));
}

fn add_provider(
    service: &ApiKeyProviderService,
    db: &DbConnection,
    provider_id: &str,
    name: &str,
    base_url: &str,
    model: &str,
) -> String {
    let provider = service
        .update_provider(
            db,
            provider_id,
            Some(name.to_string()),
            Some(ApiProviderType::Openai),
            Some(base_url.to_string()),
            Some(true),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(vec![
                lime_core::models::model_registry::ProviderModelConfig::hint(model),
            ]),
        )
        .expect("configure provider");
    service
        .add_api_key(db, &provider.id, &format!("{name}-key"), None, true)
        .expect("provider api key");
    provider.id
}

fn execution_request(primary_provider: &str, backup_provider: &str) -> ExecutionRequest {
    let session_id = "session-runtime-reroute".to_string();
    let thread_id = "thread-runtime-reroute".to_string();
    ExecutionRequest {
        host: RuntimeHostContext::default(),
        session: AgentSession {
            session_id: session_id.clone(),
            thread_id: thread_id.clone(),
            app_id: "agent".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: AgentSessionStatus::Running,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        },
        turn: AgentTurn {
            turn_id: "turn-runtime-reroute".to_string(),
            session_id,
            thread_id,
            status: AgentTurnStatus::Accepted,
            started_at: None,
            completed_at: None,
        },
        forked_from_thread_id: None,
        input: agent_runtime::reply_input::RuntimeReplyInput::text("hello"),
        runtime_options: Some(RuntimeOptions {
            stream: true,
            runtime_request: Some(RuntimeRequest {
                provider_preference: Some(primary_provider.to_string()),
                model_preference: Some(PRIMARY_MODEL.to_string()),
                metadata: Some(json!({
                    "harness": {
                        "coding_model_slots": {
                            "coding": {
                                "provider": primary_provider,
                                "model": PRIMARY_MODEL
                            },
                            "base": {
                                "provider": backup_provider,
                                "model": BACKUP_MODEL
                            }
                        }
                    }
                })),
                ..RuntimeRequest::default()
            }),
            ..RuntimeOptions::default()
        }),
        event_name: None,
        expected_output: None,
        structured_output: None,
        output_schema: None,
        queued_turn_id: None,
        queue_if_busy: false,
        skip_pre_submit_resume: false,
        agent_control_gateway: None,
        rollout_budget_reminder_source: None,
    }
}

fn test_db() -> DbConnection {
    let connection = Connection::open_in_memory().expect("open in-memory db");
    create_tables(&connection).expect("create schema");
    Arc::new(Mutex::new(connection))
}

async fn read_request_body(stream: &mut TcpStream) -> std::io::Result<Value> {
    read_request(stream).await.map(|(_, body)| body)
}

async fn read_request(stream: &mut TcpStream) -> std::io::Result<(String, Value)> {
    let mut buffer = Vec::new();
    let header_end = loop {
        let mut chunk = [0_u8; 1024];
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(header_end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break header_end;
        }
    };
    let headers = String::from_utf8_lossy(&buffer[..header_end]).into_owned();
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    let body_start = header_end + 4;
    while buffer.len().saturating_sub(body_start) < content_length {
        let mut chunk = vec![0_u8; content_length - (buffer.len() - body_start)];
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    let body = serde_json::from_slice(&buffer[body_start..body_start + content_length])
        .map_err(std::io::Error::other)?;
    Ok((headers, body))
}

fn header_value(headers: &str, expected_name: &str) -> Option<String> {
    headers.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case(expected_name)
            .then(|| value.trim().to_string())
    })
}

async fn write_response(stream: &mut TcpStream, status: &str, content_type: &str, body: &str) {
    write_response_with_headers(stream, status, content_type, "", body).await;
}

async fn write_response_with_headers(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    extra_headers: &str,
    body: &str,
) {
    let response = format!(
        "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n{extra_headers}\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .await
        .expect("write provider response");
    stream.shutdown().await.expect("close provider response");
}
