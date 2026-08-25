use std::sync::Arc;

use app_server::{AppServer, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_MCP_SERVER_EVENT_STREAM_START, METHOD_MCP_SERVER_EVENT_STREAM_STOP,
    METHOD_MCP_SERVER_RESOURCE_READ, METHOD_MCP_SERVER_TOOL_CALL, METHOD_PLUGIN_INSTALL,
};
use app_server_protocol::{METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_START};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::time::{timeout, Duration};

mod support;

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

    let origin_without_thread = request_error(
        &server,
        3,
        METHOD_MCP_SERVER_RESOURCE_READ,
        json!({
            "server": "codex_apps",
            "uri": "ui://calendar/event",
            "originCallId": "item-calendar"
        }),
    )
    .await;
    assert_eq!(origin_without_thread["error"]["code"], -32600);
    assert!(origin_without_thread["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("originCallId requires threadId")));

    let tool_error = request_error(
        &server,
        4,
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mcp_event_stream_uses_public_transport_and_cleans_up_on_stop() {
    let temp = TempDir::new().expect("MCP event stream temp");
    let plugin_source = write_event_stream_plugin(&temp);
    let app_data_db = Arc::new(std::sync::Mutex::new(
        rusqlite::Connection::open_in_memory().expect("MCP app data db"),
    ));
    lime_core::database::schema::create_tables(&app_data_db.lock().expect("MCP app data db lock"))
        .expect("MCP app data schema");
    let app_data_source = app_server::LocalAppDataSource::initialize_with_roots(
        app_data_db,
        temp.path(),
        temp.path().join("app-data"),
    )
    .await
    .expect("MCP app data source");
    let runtime = support::runtime_core_with_chat_provider(&temp, "provider-test", "model-test")
        .with_app_data_source(Arc::new(app_data_source));
    let server = AppServer::with_runtime(runtime);
    let (mut input_client, input_server) = tokio::io::duplex(64 * 1024);
    let (output_server, output_client) = tokio::io::duplex(64 * 1024);
    let runner = tokio::spawn(app_server::run_json_lines(
        server,
        input_server,
        output_server,
    ));
    let mut output_lines = BufReader::new(output_client).lines();

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": METHOD_INITIALIZE,
            "params": {
                "clientInfo": {"name": "mcp-event-stream-jsonrpc-test", "version": "1"},
                "capabilities": {"experimentalApi": true}
            }
        }),
    )
    .await;
    let initialized = read_response(&mut output_lines, 1).await;
    assert!(initialized.get("error").is_none(), "{initialized:#}");
    write_message(
        &mut input_client,
        json!({"jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {}}),
    )
    .await;

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": METHOD_PLUGIN_INSTALL,
            "params": {"sourcePath": plugin_source, "marketplaceId": "fixture", "source": "local"}
        }),
    )
    .await;
    let installed = read_response(&mut output_lines, 2).await;
    assert!(installed.get("error").is_none(), "{installed:#}");

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": METHOD_THREAD_START,
            "params": {"model": "model-test", "modelProvider": "provider-test"}
        }),
    )
    .await;
    let started = read_response(&mut output_lines, 3).await;
    let thread_id = started["result"]["thread"]["id"]
        .as_str()
        .expect("thread/start id")
        .to_string();
    let thread_started = read_notification(&mut output_lines, "thread/started").await;
    assert_eq!(thread_started["params"]["thread"]["id"], thread_id);

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": METHOD_MCP_SERVER_RESOURCE_READ,
            "params": {
                "threadId": thread_id,
                "server": "plugin__event-plugin__events",
                "uri": "ui://calendar/event",
                "connectorId": "calendar"
            }
        }),
    )
    .await;
    let resource = read_response(&mut output_lines, 4).await;
    let resource_meta: Value = serde_json::from_str(
        resource["result"]["contents"][0]["text"]
            .as_str()
            .expect("resource metadata text"),
    )
    .expect("decode resource metadata");
    assert_eq!(
        resource_meta.pointer("/x-codex-turn-metadata/mcp_request_meta/selected_connector_ids/0"),
        Some(&json!("calendar"))
    );
    assert_eq!(resource["result"]["originCallId"], Value::Null);

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": METHOD_MCP_SERVER_EVENT_STREAM_START,
            "params": {
                "threadId": thread_id,
                "server": "plugin__event-plugin__events",
                "subscriptionId": "subscription-1",
                "name": "issue.updated",
                "arguments": {"project": "codex"},
                "_meta": {"source": "desktop"}
            }
        }),
    )
    .await;
    let active = read_message(&mut output_lines).await;
    assert_eq!(
        active["id"],
        Value::Null,
        "active notification must precede response"
    );
    assert_eq!(active["method"], "mcpServer/event/stream/notification");
    assert_eq!(active["params"]["subscriptionId"], "subscription-1");
    assert_eq!(
        active["params"]["notification"]["method"],
        "notifications/events/active"
    );
    let after_active = [
        read_message(&mut output_lines).await,
        read_message(&mut output_lines).await,
    ];
    let event = after_active
        .iter()
        .find(|message| message["method"] == "mcpServer/event/stream/notification")
        .expect("event notification after active barrier");
    assert_eq!(event["method"], "mcpServer/event/stream/notification");
    assert_eq!(
        event["params"]["notification"]["method"],
        "notifications/events/event"
    );
    assert_eq!(
        event["params"]["notification"]["params"]["name"],
        "issue.updated"
    );

    let started_response = after_active
        .iter()
        .find(|message| message["id"] == json!(5))
        .expect("event stream start response after active barrier");
    assert!(
        started_response.get("error").is_none(),
        "{started_response:#}"
    );

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 6,
            "method": METHOD_MCP_SERVER_EVENT_STREAM_STOP,
            "params": {"subscriptionId": "subscription-1"}
        }),
    )
    .await;
    let stopped = read_response(&mut output_lines, 6).await;
    assert_eq!(stopped["result"], json!({}));

    drop(input_client);
    timeout(Duration::from_secs(2), runner)
        .await
        .expect("JSON lines runner must stop after transport close")
        .expect("JSON lines runner task")
        .expect("JSON lines runner result");
}

fn write_event_stream_plugin(temp: &TempDir) -> String {
    let root = temp.path().join("event-plugin");
    std::fs::create_dir_all(root.join(".codex-plugin")).expect("plugin manifest directory");
    std::fs::write(
        root.join("plugin.json"),
        r#"{"$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json","name":"event-plugin","version":"1.0.0","description":"MCP event fixture"}"#,
    )
    .expect("plugin manifest");
    std::fs::write(
        root.join(".codex-plugin/plugin.json"),
        r#"{"name":"event-plugin","version":"1.0.0"}"#,
    )
    .expect("Codex plugin manifest");
    std::fs::write(
        root.join("mcp.json"),
        r#"{
          "$schema":"https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
          "mcpServers":{"events":{"type":"stdio","command":"node","args":["./server.mjs"]}}
        }"#,
    )
    .expect("MCP manifest");
    std::fs::write(
        root.join("server.mjs"),
        r#"
import readline from "node:readline";
const rl = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
const send = (message) => process.stdout.write(`${JSON.stringify(message)}\n`);
rl.on("line", (line) => {
  if (!line.trim()) return;
  const message = JSON.parse(line);
  if (message.method === "initialize") {
    send({jsonrpc:"2.0", id:message.id, result:{protocolVersion:"2025-03-26", capabilities:{resources:{}}, serverInfo:{name:"event-fixture", version:"1"}}});
  } else if (message.method === "notifications/initialized") {
  } else if (message.method === "tools/list") {
    send({jsonrpc:"2.0", id:message.id, result:{tools:[]}});
  } else if (message.method === "resources/read") {
    send({jsonrpc:"2.0", id:message.id, result:{contents:[{
      uri:message.params?.uri,
      mimeType:"application/json",
      text:JSON.stringify(message.params?._meta ?? null)
    }]}});
  } else if (message.method === "events/stream") {
    const meta = {"io.modelcontextprotocol/subscriptionId": message.id, provider:"event-fixture"};
    send({jsonrpc:"2.0", method:"notifications/events/active", params:{_meta:meta, status:"active"}});
    send({jsonrpc:"2.0", method:"notifications/events/event", params:{_meta:meta, name:"issue.updated", data:{issue:42}}});
  } else if (message.method === "notifications/cancelled") {
  }
});
"#,
    )
    .expect("MCP fixture server");
    root.to_string_lossy().into_owned()
}

async fn write_message(input: &mut DuplexStream, message: Value) {
    input
        .write_all(format!("{message}\n").as_bytes())
        .await
        .expect("write JSON-RPC message");
}

async fn read_response(output: &mut Lines<BufReader<DuplexStream>>, id: u64) -> Value {
    loop {
        let message = read_message(output).await;
        if message["id"] == json!(id) {
            return message;
        }
    }
}

async fn read_notification(output: &mut Lines<BufReader<DuplexStream>>, method: &str) -> Value {
    loop {
        let message = read_message(output).await;
        if message["method"] == method {
            return message;
        }
    }
}

async fn read_message(output: &mut Lines<BufReader<DuplexStream>>) -> Value {
    loop {
        let line = timeout(Duration::from_secs(5), output.next_line())
            .await
            .expect("JSON-RPC response timeout")
            .expect("read JSON-RPC response")
            .expect("JSON-RPC output closed");
        let message: Value = serde_json::from_str(&line).expect("decode JSON-RPC message");
        return message;
    }
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
