use std::sync::Arc;

use app_server::{
    ActionRespondRequest, AppServer, CancelExecutionRequest, ExecutionBackend, ExecutionRequest,
    ProjectionStore, RuntimeCore, RuntimeCoreError, RuntimeEventSink,
};
use app_server_protocol::{
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_READ, METHOD_THREAD_START,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use tempfile::TempDir;

struct NameBackend;

#[async_trait]
impl ExecutionBackend for NameBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

#[tokio::test]
async fn thread_name_set_persists_and_emits_typed_notification() {
    let temp = TempDir::new().expect("thread name temp dir");
    let projection = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
        .expect("thread name projection store");
    let server = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(NameBackend))
            .with_projection_store(Arc::new(projection)),
    );
    request(
        &server,
        1,
        METHOD_INITIALIZE,
        json!({"clientInfo": {"name": "thread-name-jsonrpc-test", "version": "1"}}),
    )
    .await;
    server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {}}).to_string(),
        )
        .await
        .expect("initialized notification");

    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({"model": "fixture-model", "modelProvider": "fixture-provider"}),
    )
    .await;
    let thread_id = started["result"]["thread"]["id"]
        .as_str()
        .expect("thread id")
        .to_string();

    let messages = server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "thread/name/set",
                "params": {"threadId": thread_id, "name": "  Durable name  "}
            })
            .to_string(),
        )
        .await
        .expect("thread name request");
    let values = messages
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("JSON-RPC message"))
        .collect::<Vec<_>>();
    assert!(
        values
            .iter()
            .any(|value| value.get("id") == Some(&json!(3))
                && value.get("result") == Some(&json!({})))
    );
    let notification = values
        .iter()
        .find(|value| value.get("method") == Some(&json!("thread/name/updated")))
        .expect("thread/name/updated notification");
    assert_eq!(
        notification["params"],
        json!({"threadId": thread_id, "threadName": "Durable name"})
    );

    let read = request(
        &server,
        4,
        METHOD_THREAD_READ,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(read["result"]["thread"]["name"], "Durable name");
}

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let messages = server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string(),
        )
        .await
        .expect("JSON-RPC request");
    let response = messages
        .iter()
        .filter_map(|message| serde_json::from_str::<Value>(message).ok())
        .find(|message| message.get("id") == Some(&json!(id)))
        .expect("JSON-RPC response");
    assert!(
        response.get("error").is_none(),
        "request failed: {response:#}"
    );
    response
}
