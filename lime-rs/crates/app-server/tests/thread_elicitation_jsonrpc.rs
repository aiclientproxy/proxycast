use std::sync::Arc;

use app_server::{AppServer, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_ARCHIVE, METHOD_THREAD_DECREMENT_ELICITATION, METHOD_THREAD_INCREMENT_ELICITATION,
};
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_START,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use uuid::Uuid;

#[tokio::test]
async fn elicitation_count_is_thread_local_and_cleared_when_unloaded() {
    let temp = TempDir::new().expect("elicitation JSON-RPC temp dir");
    let projection = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
        .expect("elicitation projection store");
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_projection_store(Arc::new(projection));
    let server = AppServer::with_runtime(runtime);
    initialize(&server).await;

    let thread_id = start_thread(&server, 2).await;
    let underflow = request_raw(
        &server,
        3,
        METHOD_THREAD_DECREMENT_ELICITATION,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(
        underflow.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );

    assert_eq!(
        request(
            &server,
            4,
            METHOD_THREAD_INCREMENT_ELICITATION,
            json!({"threadId": thread_id})
        )
        .await["result"],
        json!({"count": 1, "paused": true})
    );
    assert_eq!(
        request(
            &server,
            5,
            METHOD_THREAD_INCREMENT_ELICITATION,
            json!({"threadId": thread_id})
        )
        .await["result"],
        json!({"count": 2, "paused": true})
    );
    assert_eq!(
        request(
            &server,
            6,
            METHOD_THREAD_DECREMENT_ELICITATION,
            json!({"threadId": thread_id})
        )
        .await["result"],
        json!({"count": 1, "paused": true})
    );
    assert_eq!(
        request(
            &server,
            7,
            METHOD_THREAD_DECREMENT_ELICITATION,
            json!({"threadId": thread_id})
        )
        .await["result"],
        json!({"count": 0, "paused": false})
    );

    let unknown_thread = request_raw(
        &server,
        8,
        METHOD_THREAD_INCREMENT_ELICITATION,
        json!({"threadId": Uuid::now_v7().to_string()}),
    )
    .await;
    assert_eq!(
        unknown_thread.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );

    request(
        &server,
        9,
        METHOD_THREAD_INCREMENT_ELICITATION,
        json!({"threadId": thread_id}),
    )
    .await;
    request(
        &server,
        10,
        METHOD_THREAD_ARCHIVE,
        json!({"threadId": thread_id}),
    )
    .await;
    let unloaded = request_raw(
        &server,
        11,
        METHOD_THREAD_INCREMENT_ELICITATION,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(
        unloaded.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );
}

async fn initialize(server: &AppServer) {
    request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({"clientInfo": {"name": "thread-elicitation-test", "version": "1"}}),
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
    request(
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

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let response = request_raw(server, id, method, params).await;
    assert!(
        response.get("error").is_none(),
        "{method} failed: {response:#}"
    );
    response
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
