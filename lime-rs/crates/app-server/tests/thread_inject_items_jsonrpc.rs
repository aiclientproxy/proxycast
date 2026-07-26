use std::sync::Arc;

use app_server::{
    AppServer, EventLogWriter, MockBackend, ProjectionStore, RuntimeCore, StorageRoots,
};
use app_server_protocol::protocol::v2::{METHOD_THREAD_ARCHIVE, METHOD_THREAD_INJECT_ITEMS};
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_START,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use uuid::Uuid;

#[tokio::test]
async fn thread_inject_items_validates_and_persists_raw_response_items() {
    let temp = TempDir::new().expect("inject-items JSON-RPC temp dir");
    let projection = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
        .expect("inject-items projection store");
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_projection_store(Arc::new(projection));
    let server = AppServer::with_runtime(runtime);
    initialize(&server).await;

    let thread_id = start_thread(&server, 2).await;
    let response = request_raw(
        &server,
        3,
        METHOD_THREAD_INJECT_ITEMS,
        json!({"threadId": thread_id, "items": [assistant_item()]}),
    )
    .await;
    assert_eq!(response.get("result"), Some(&json!({})));

    for (id, items) in [
        (4, json!([])),
        (5, json!([{"type": "message", "role": "assistant"}])),
        (
            6,
            json!([{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "https://example.com/a.png"}]
            }]),
        ),
    ] {
        let response = request_raw(
            &server,
            id,
            METHOD_THREAD_INJECT_ITEMS,
            json!({"threadId": thread_id, "items": items}),
        )
        .await;
        assert_eq!(
            response.pointer("/error/code"),
            Some(&json!(error_codes::INVALID_REQUEST))
        );
    }

    let unknown_thread = request_raw(
        &server,
        7,
        METHOD_THREAD_INJECT_ITEMS,
        json!({"threadId": Uuid::now_v7().to_string(), "items": [assistant_item()]}),
    )
    .await;
    assert_eq!(
        unknown_thread.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );
}

#[tokio::test]
async fn thread_inject_items_resumes_cold_history_and_rejects_archived_thread() {
    let temp = TempDir::new().expect("inject-items restart JSON-RPC temp dir");
    let roots = StorageRoots::initialize(temp.path(), temp.path().join("app-server"))
        .expect("inject-items restart storage roots");
    let thread_id = {
        let projection = Arc::new(
            ProjectionStore::initialize(&roots.projection_db_path)
                .expect("inject-items initial projection store"),
        );
        let event_log = Arc::new(
            EventLogWriter::new(&roots.event_log_root)
                .expect("inject-items initial event log writer"),
        );
        let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
            .with_event_log_writer(event_log)
            .with_projection_store(projection);
        let server = AppServer::with_runtime(runtime);
        initialize(&server).await;
        let thread_id = start_thread(&server, 2).await;
        let response = request_raw(
            &server,
            3,
            METHOD_THREAD_INJECT_ITEMS,
            json!({"threadId": thread_id, "items": [assistant_item()]}),
        )
        .await;
        assert_eq!(response.get("result"), Some(&json!({})));
        thread_id
    };

    let projection = Arc::new(
        ProjectionStore::initialize(&roots.projection_db_path)
            .expect("inject-items restarted projection store"),
    );
    let event_log = Arc::new(
        EventLogWriter::new(&roots.event_log_root)
            .expect("inject-items restarted event log writer"),
    );
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_event_log_writer(event_log)
        .with_projection_store(projection);
    let server = AppServer::with_runtime(runtime);
    initialize(&server).await;

    let cold_response = request_raw(
        &server,
        2,
        METHOD_THREAD_INJECT_ITEMS,
        json!({"threadId": thread_id, "items": [assistant_item()]}),
    )
    .await;
    assert_eq!(cold_response.get("result"), Some(&json!({})));

    let archive = request_raw(
        &server,
        3,
        METHOD_THREAD_ARCHIVE,
        json!({"threadId": thread_id}),
    )
    .await;
    assert!(archive.get("error").is_none(), "archive failed: {archive}");

    let archived_response = request_raw(
        &server,
        4,
        METHOD_THREAD_INJECT_ITEMS,
        json!({"threadId": thread_id, "items": [assistant_item()]}),
    )
    .await;
    assert_eq!(
        archived_response.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );
    assert!(archived_response
        .pointer("/error/message")
        .and_then(Value::as_str)
        .is_some_and(|message| message.contains("cannot inject into archived thread")));
}

fn assistant_item() -> Value {
    json!({
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "injected context"}],
        "provider_extension": {"keep": true}
    })
}

async fn initialize(server: &AppServer) {
    request_raw(
        server,
        1,
        METHOD_INITIALIZE,
        json!({"clientInfo": {"name": "thread-inject-items-test", "version": "1"}}),
    )
    .await;
    server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {}}).to_string(),
        )
        .await
        .expect("initialized notification");
}

async fn start_thread(server: &AppServer, id: u64) -> String {
    request_raw(
        server,
        id,
        METHOD_THREAD_START,
        json!({"model": "fixture-model", "modelProvider": "fixture-provider"}),
    )
    .await["result"]["thread"]["id"]
        .as_str()
        .expect("thread id")
        .to_string()
}

async fn request_raw(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params
            })
            .to_string(),
        )
        .await
        .expect("handle JSON-RPC request")
        .into_iter()
        .map(|line| serde_json::from_str::<Value>(&line).expect("valid JSON-RPC response"))
        .find(|message| message.get("id") == Some(&json!(id)))
        .expect("matching JSON-RPC response")
}
