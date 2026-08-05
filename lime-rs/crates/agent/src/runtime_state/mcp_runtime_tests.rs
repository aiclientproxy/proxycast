use crate::AgentRuntimeState;
use lime_mcp::{
    ElicitationRequestRouter, McpRuntimeServerSpec, McpServerConfig, McpServerTransport,
};
use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

fn stdio_server_spec(
    name: &str,
    command: String,
    args: Vec<String>,
    required: bool,
) -> McpRuntimeServerSpec {
    McpRuntimeServerSpec {
        name: name.to_string(),
        plugin_id: None,
        config: McpServerConfig {
            transport: McpServerTransport::Stdio {
                command,
                args,
                env: HashMap::new(),
                cwd: None,
            },
            enabled: true,
            required,
            ..McpServerConfig::default()
        },
    }
}

fn node_binary() -> Option<String> {
    let mut candidates = std::env::var("NODE")
        .ok()
        .into_iter()
        .chain(["node".to_string()]);
    candidates.find(|candidate| {
        Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success())
    })
}

#[tokio::test]
async fn session_owned_mcp_runtimes_are_thread_exact_and_close_independently() {
    let state = AgentRuntimeState::new();
    let router = ElicitationRequestRouter::default();
    let runtime_a = state
        .ensure_mcp_runtime_generation(
            "session-a".to_string(),
            "thread-a".to_string(),
            router.clone(),
            Vec::new(),
        )
        .await
        .expect("create session A runtime");
    let runtime_b = state
        .ensure_mcp_runtime_generation(
            "session-b".to_string(),
            "thread-b".to_string(),
            router,
            Vec::new(),
        )
        .await
        .expect("create session B runtime");

    assert!(!Arc::ptr_eq(&runtime_a, &runtime_b));
    assert_eq!(state.mcp_runtime_count().await, 2);
    assert!(state.mcp_runtime("session-a", "thread-b").await.is_err());

    state.close_mcp_runtime("session-a", "thread-a").await;
    assert_eq!(state.mcp_runtime_count().await, 1);
    assert!(state.mcp_runtime("session-a", "thread-a").await.is_err());
    assert!(Arc::ptr_eq(
        &runtime_b,
        &state
            .mcp_runtime("session-b", "thread-b")
            .await
            .expect("session B runtime remains"),
    ));
}

#[tokio::test]
async fn unchanged_server_specs_reuse_the_session_generation() {
    let state = AgentRuntimeState::new();
    let router = ElicitationRequestRouter::default();
    let first = state
        .ensure_mcp_runtime_generation(
            "session-a".to_string(),
            "thread-a".to_string(),
            router.clone(),
            Vec::new(),
        )
        .await
        .expect("first generation");
    let second = state
        .ensure_mcp_runtime_generation(
            "session-a".to_string(),
            "thread-a".to_string(),
            router,
            Vec::new(),
        )
        .await
        .expect("same generation");

    assert!(Arc::ptr_eq(&first, &second));
}

#[tokio::test]
async fn concurrent_unchanged_runtime_ensures_publish_one_generation() {
    let state = AgentRuntimeState::new();
    let first_state = state.clone();
    let first = tokio::spawn(async move {
        first_state
            .ensure_mcp_runtime_generation(
                "session-a".to_string(),
                "thread-a".to_string(),
                ElicitationRequestRouter::default(),
                Vec::new(),
            )
            .await
    });
    let second_state = state.clone();
    let second = tokio::spawn(async move {
        second_state
            .ensure_mcp_runtime_generation(
                "session-a".to_string(),
                "thread-a".to_string(),
                ElicitationRequestRouter::default(),
                Vec::new(),
            )
            .await
    });

    let first = first
        .await
        .expect("first ensure task")
        .expect("first generation");
    let second = second
        .await
        .expect("second ensure task")
        .expect("second generation");

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(state.mcp_runtime_count().await, 1);
}

#[tokio::test]
async fn failed_runtime_server_does_not_block_healthy_server_generation() {
    let Some(node) = node_binary() else {
        return;
    };
    let temp_dir = tempfile::tempdir().expect("create runtime MCP fixture directory");
    let server_path = temp_dir.path().join("healthy-mcp-server.mjs");
    std::fs::write(
        &server_path,
        r#"
import readline from "node:readline";

const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
const send = (message) => process.stdout.write(JSON.stringify(message) + "\n");

lines.on("line", (line) => {
  if (!line.trim()) return;
  const message = JSON.parse(line);
  if (message.method === "initialize") {
    send({
      jsonrpc: "2.0",
      id: message.id,
      result: {
        protocolVersion: "2025-03-26",
        capabilities: { tools: {} },
        serverInfo: { name: "healthy-runtime-fixture", version: "1.0.0" },
      },
    });
    return;
  }
  if (message.method === "notifications/initialized") return;
  if (message.method === "tools/list") {
    send({
      jsonrpc: "2.0",
      id: message.id,
      result: {
        tools: [{
          name: "healthy_tool",
          description: "healthy runtime MCP fixture",
          inputSchema: { type: "object", properties: {}, additionalProperties: false },
        }],
      },
    });
  }
});
"#,
    )
    .expect("write healthy runtime MCP fixture");

    let state = AgentRuntimeState::new();
    let runtime = state
        .ensure_mcp_runtime_generation(
            "session-a".to_string(),
            "thread-a".to_string(),
            ElicitationRequestRouter::default(),
            vec![
                stdio_server_spec(
                    "broken",
                    "/definitely/not/a/runtime-mcp-server".to_string(),
                    Vec::new(),
                    false,
                ),
                stdio_server_spec(
                    "healthy",
                    node,
                    vec![server_path.to_string_lossy().into_owned()],
                    false,
                ),
            ],
        )
        .await
        .expect("healthy MCP server must publish despite a failed sibling");

    assert_eq!(runtime.connections().names().await, vec!["mcp__healthy"]);
    assert!(Arc::ptr_eq(
        &runtime,
        &state
            .mcp_runtime("session-a", "thread-a")
            .await
            .expect("published runtime generation"),
    ));
    state.clear_mcp_runtimes().await;
}

#[tokio::test]
async fn session_owned_mcp_runtime_reads_app_resource_with_meta() {
    let Some(node) = node_binary() else {
        return;
    };
    let temp_dir = tempfile::tempdir().expect("create runtime MCP app fixture directory");
    let server_path = temp_dir.path().join("mcp-app-resource-server.mjs");
    std::fs::write(
        &server_path,
        r#"
import readline from "node:readline";

const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
const send = (message) => process.stdout.write(JSON.stringify(message) + "\n");

lines.on("line", (line) => {
  if (!line.trim()) return;
  const message = JSON.parse(line);
  if (message.method === "initialize") {
    send({
      jsonrpc: "2.0",
      id: message.id,
      result: {
        protocolVersion: "2025-03-26",
        capabilities: { resources: {}, tools: {} },
        serverInfo: { name: "mcp-app-resource-fixture", version: "1.0.0" },
      },
    });
    return;
  }
  if (message.method === "notifications/initialized") return;
  if (message.method === "tools/list") {
    send({ jsonrpc: "2.0", id: message.id, result: { tools: [] } });
    return;
  }
  if (message.method === "resources/read") {
    send({
      jsonrpc: "2.0",
      id: message.id,
      result: {
        contents: [{
          uri: message.params?.uri,
          mimeType: "text/html;profile=mcp-app",
          text: "<!doctype html><main>MCP App</main>",
          _meta: {
            ui: { csp: { connectDomains: ["https://api.example.com"] } },
          },
        }],
      },
    });
  }
});
"#,
    )
    .expect("write runtime MCP app fixture");

    let state = AgentRuntimeState::new();
    state
        .ensure_mcp_runtime_generation(
            "session-app".to_string(),
            "thread-app".to_string(),
            ElicitationRequestRouter::default(),
            vec![stdio_server_spec(
                "plugin__demo__server",
                node,
                vec![server_path.to_string_lossy().into_owned()],
                true,
            )],
        )
        .await
        .expect("publish MCP App runtime generation");

    let content = state
        .read_mcp_resource(
            "session-app",
            "thread-app",
            "plugin__demo__server",
            "ui://demo/report.html",
        )
        .await
        .expect("read MCP App resource from session runtime");
    assert_eq!(content.uri, "ui://demo/report.html");
    assert_eq!(
        content.mime_type.as_deref(),
        Some("text/html;profile=mcp-app")
    );
    assert_eq!(
        content
            .meta
            .as_ref()
            .and_then(|meta| meta.pointer("/ui/csp/connectDomains/0")),
        Some(&serde_json::json!("https://api.example.com"))
    );
    assert!(state
        .read_mcp_resource(
            "session-app",
            "thread-other",
            "plugin__demo__server",
            "ui://demo/report.html",
        )
        .await
        .is_err());
    state.clear_mcp_runtimes().await;
}

#[tokio::test]
async fn required_runtime_server_failure_keeps_the_previous_generation_published() {
    let state = AgentRuntimeState::new();
    let previous = state
        .ensure_mcp_runtime_generation(
            "session-a".to_string(),
            "thread-a".to_string(),
            ElicitationRequestRouter::default(),
            Vec::new(),
        )
        .await
        .expect("publish initial runtime generation");

    let result = state
        .ensure_mcp_runtime_generation(
            "session-a".to_string(),
            "thread-a".to_string(),
            ElicitationRequestRouter::default(),
            vec![stdio_server_spec(
                "required-broken",
                "/definitely/not/a/required-runtime-mcp-server".to_string(),
                Vec::new(),
                true,
            )],
        )
        .await;
    assert!(
        result.is_err(),
        "a required MCP failure must reject replacement generation"
    );
    let error = result.err().expect("required MCP failure");

    assert!(error.contains("required-broken"));
    assert!(Arc::ptr_eq(
        &previous,
        &state
            .mcp_runtime("session-a", "thread-a")
            .await
            .expect("previous generation remains published"),
    ));
    state.clear_mcp_runtimes().await;
}

#[tokio::test]
async fn replacement_keeps_previous_generation_visible_until_startup_completes() {
    let Some(node) = node_binary() else {
        return;
    };
    let state = AgentRuntimeState::new();
    let previous = state
        .ensure_mcp_runtime_generation(
            "session-a".to_string(),
            "thread-a".to_string(),
            ElicitationRequestRouter::default(),
            Vec::new(),
        )
        .await
        .expect("publish initial runtime generation");

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind startup gate");
    let gate_address = listener.local_addr().expect("startup gate address");
    let temp_dir = tempfile::tempdir().expect("create runtime MCP fixture directory");
    let server_path = temp_dir.path().join("gated-mcp-server.mjs");
    std::fs::write(
        &server_path,
        r#"
import net from "node:net";
import readline from "node:readline";

const [host, portText] = process.argv[2].split(":");
const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
const send = (message) => process.stdout.write(JSON.stringify(message) + "\n");

lines.on("line", (line) => {
  if (!line.trim()) return;
  const message = JSON.parse(line);
  if (message.method === "initialize") {
    const gate = net.createConnection({ host, port: Number(portText) });
    gate.once("connect", () => gate.write("started\n"));
    gate.on("data", (chunk) => {
      if (!chunk.toString().includes("release")) return;
      gate.end();
      send({
        jsonrpc: "2.0",
        id: message.id,
        result: {
          protocolVersion: "2025-03-26",
          capabilities: { tools: {} },
          serverInfo: { name: "gated-runtime-fixture", version: "1.0.0" },
        },
      });
    });
    return;
  }
  if (message.method === "notifications/initialized") return;
  if (message.method === "tools/list") {
    send({
      jsonrpc: "2.0",
      id: message.id,
      result: {
        tools: [{
          name: "gated_tool",
          description: "gated runtime MCP fixture",
          inputSchema: { type: "object", properties: {}, additionalProperties: false },
        }],
      },
    });
  }
});
"#,
    )
    .expect("write gated runtime MCP fixture");

    let replacement_state = state.clone();
    let replacement = tokio::spawn(async move {
        replacement_state
            .ensure_mcp_runtime_generation(
                "session-a".to_string(),
                "thread-a".to_string(),
                ElicitationRequestRouter::default(),
                vec![stdio_server_spec(
                    "gated",
                    node,
                    vec![
                        server_path.to_string_lossy().into_owned(),
                        gate_address.to_string(),
                    ],
                    true,
                )],
            )
            .await
    });

    let (gate, _) = tokio::time::timeout(Duration::from_secs(5), listener.accept())
        .await
        .expect("replacement startup must reach gate")
        .expect("accept replacement startup gate");
    let (reader, mut writer) = gate.into_split();
    let mut reader = BufReader::new(reader);
    let mut signal = String::new();
    reader
        .read_line(&mut signal)
        .await
        .expect("read replacement startup signal");
    assert_eq!(signal, "started\n");

    let during_startup = state
        .mcp_runtime("session-a", "thread-a")
        .await
        .expect("previous generation remains readable during replacement startup");
    assert!(Arc::ptr_eq(&previous, &during_startup));

    writer
        .write_all(b"release\n")
        .await
        .expect("release replacement startup");
    let replacement = tokio::time::timeout(Duration::from_secs(10), replacement)
        .await
        .expect("replacement startup must complete")
        .expect("replacement task")
        .expect("publish replacement generation");

    assert!(!Arc::ptr_eq(&previous, &replacement));
    assert_eq!(replacement.connections().names().await, vec!["mcp__gated"]);
    assert!(Arc::ptr_eq(
        &replacement,
        &state
            .mcp_runtime("session-a", "thread-a")
            .await
            .expect("replacement generation is published after startup"),
    ));
    state.clear_mcp_runtimes().await;
}
