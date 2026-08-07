use std::sync::Arc;

use app_server::{AppServer, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_MCP_SERVER_RESOURCE_READ, METHOD_MCP_SERVER_TOOL_CALL,
};
use app_server_protocol::{METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_START};
use serde_json::{json, Value};
use tempfile::TempDir;

#[tokio::test]
async fn exact_mcp_methods_are_registered_and_fail_closed_on_invalid_identity() {
    let server = AppServer::with_runtime(RuntimeCore::with_backend(Arc::new(MockBackend)));
    initialize(&server).await;

    let resource_error = request_error(
        &server,
        2,
        METHOD_MCP_SERVER_RESOURCE_READ,
        json!({"server": "", "uri": "docs://readme"}),
    )
    .await;
    assert_eq!(resource_error["error"]["code"], -32600);
    assert!(resource_error["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("requires server and uri")));

    let tool_error = request_error(
        &server,
        3,
        METHOD_MCP_SERVER_TOOL_CALL,
        json!({"threadId": "", "server": "docs", "tool": "search"}),
    )
    .await;
    assert_eq!(tool_error["error"]["code"], -32600);
    assert!(tool_error["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("requires threadId")));
}

#[tokio::test]
async fn exact_mcp_tool_call_requires_an_existing_thread() {
    let temp = TempDir::new().expect("MCP exact projection temp");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("MCP exact projection store"),
    );
    let runtime =
        RuntimeCore::with_backend(Arc::new(MockBackend)).with_projection_store(projection_store);
    let server = AppServer::with_runtime(runtime);
    initialize(&server).await;
    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({"model": "gpt-5.4", "modelProvider": "openai"}),
    )
    .await;
    let thread_id = started["result"]["thread"]["id"]
        .as_str()
        .expect("thread id");

    let missing_thread = request_error(
        &server,
        3,
        METHOD_MCP_SERVER_TOOL_CALL,
        json!({
            "threadId": format!("{thread_id}-missing"),
            "server": "docs",
            "tool": "search",
            "arguments": {"query": "MCP"}
        }),
    )
    .await;
    assert_ne!(missing_thread["error"]["code"], -32601);
    assert!(missing_thread["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("not found")));

    let backend_error = request_error(
        &server,
        4,
        METHOD_MCP_SERVER_TOOL_CALL,
        json!({
            "threadId": thread_id,
            "server": "docs",
            "tool": "search",
            "arguments": {"query": "MCP"}
        }),
    )
    .await;
    assert_eq!(backend_error["error"]["code"], -32000);
    assert!(backend_error["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("runtime backend does not execute MCP tools")));
}

async fn initialize(server: &AppServer) {
    request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({"clientInfo":{"name":"mcp-exact-jsonrpc-test","version":"1"}}),
    )
    .await;
    let lines = server
        .handle_json_line(
            &json!({"jsonrpc":"2.0","method":METHOD_INITIALIZED,"params":{}}).to_string(),
        )
        .await
        .expect("initialized notification");
    assert!(lines.is_empty());
}

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let response = request_message(server, id, method, params).await;
    assert!(
        response.get("error").is_none(),
        "request failed: {response:#}"
    );
    response
}

async fn request_error(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let response = request_message(server, id, method, params).await;
    assert!(
        response.get("error").is_some(),
        "request succeeded: {response:#}"
    );
    response
}

async fn request_message(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    server
        .handle_json_line(
            &json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}).to_string(),
        )
        .await
        .expect("JSON-RPC request")
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("decode response"))
        .find(|value| value.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("missing response id {id}"))
}
