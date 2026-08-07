use super::*;
use crate::AppServer;
use crate::MockBackend;
use crate::RightSurfaceAppDataSource;
use crate::RuntimeCore;
use crate::WorkspaceObjectCanvasSnapshotListParams;
use app_server_protocol::protocol::v2::METHOD_MCP_SERVER_RESOURCE_READ;
use app_server_protocol::WorkspaceRightSurfacePendingListParams;
use app_server_protocol::WorkspaceRightSurfaceRequestParams;
use app_server_protocol::METHOD_INITIALIZE;
use app_server_protocol::METHOD_INITIALIZED;
use app_server_protocol::METHOD_MCP_RESOURCE_LIST;
use app_server_protocol::METHOD_MCP_RESOURCE_SUBSCRIBE;
use app_server_protocol::METHOD_MCP_RESOURCE_UNSUBSCRIBE;
use app_server_protocol::METHOD_MCP_SERVER_CREATE;
use app_server_protocol::METHOD_MCP_SERVER_START;
use app_server_protocol::METHOD_MCP_SERVER_STATUS_LIST;
use app_server_protocol::METHOD_MCP_SERVER_STOP;
use app_server_protocol::METHOD_MCP_TOOL_LIST;
use app_server_protocol::METHOD_SKILL_CACHE_REFRESH;
use app_server_protocol::SERVER_NAME;
use app_server_transport::decode_message;
use lime_core::database::schema::create_tables;
use rusqlite::params;
use rusqlite::Connection;
use serde_json::json;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use tempfile::TempDir;

const WORKSPACE_ID: &str = "workspace-current";
const WORKSPACE_ROOT: &str = "/tmp/lime-current-workspace";
fn setup_data_source() -> LocalAppDataSource {
    let app_data_root = std::env::temp_dir().join("app-server-local-data-source-test");
    let agent_root = app_data_root.join("agent");
    let conn = Connection::open_in_memory().expect("open in-memory db");
    create_tables(&conn).expect("create schema");
    conn.execute(
        "INSERT INTO workspaces (
                id, name, workspace_type, root_path, is_default, settings_json,
                created_at, updated_at, icon, color, is_favorite, is_archived,
                tags_json, default_persona_id
             )
             VALUES (?1, '当前工作区', 'persistent', ?2, 1, '{}', 1, 1,
                     NULL, NULL, 0, 0, '[]', NULL)",
        params![WORKSPACE_ID, WORKSPACE_ROOT],
    )
    .expect("insert workspace");
    LocalAppDataSource {
        db: Arc::new(Mutex::new(conn)),
        app_data_root,
        product_db_path: crate::runtime::product_db_path_for_agent_root(&agent_root),
        plugin_data_root: agent_root.join("plugins"),
        session_files_root: agent_root.join("artifacts").join("sessions"),
        connect_registry_cache_path: agent_root
            .join("cache")
            .join("connect")
            .join("registry.json"),
        logs: Arc::new(tokio::sync::RwLock::new(
            lime_core::logger::LogStore::in_memory(),
        )),
        api_key_provider_service: ApiKeyProviderService::new(),
        model_registry_service: ModelRegistryService::new(Arc::new(Mutex::new(
            Connection::open_in_memory().expect("open model db"),
        ))),
        mcp_manager: Arc::new(TokioMutex::new(McpClientManager::new(None))),
        mcp_elicitation_router: lime_mcp::ElicitationRequestRouter::default(),
        telegram_gateway_state: TelegramGatewayState::default(),
        feishu_gateway_state: FeishuGatewayState::default(),
        discord_gateway_state: DiscordGatewayState::default(),
        wechat_gateway_state: WechatGatewayState::default(),
        gateway_tunnel_state: GatewayTunnelState::default(),
        wechat_login_state: WechatLoginState::default(),
        memory_backend: Arc::new(LocalMemoryBackend::new(
            std::env::temp_dir().join("app-server-local-data-source-test-memory"),
        )),
        sidecar_store: None,
    }
}

#[tokio::test]
async fn successful_skill_catalog_mutation_emits_typed_invalidation() {
    let server = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(MockBackend))
            .with_app_data_source(Arc::new(setup_data_source())),
    );
    let init = app_server_request(
        &server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": { "name": "skills-changed-test", "version": "test" },
            "capabilities": {}
        }),
    )
    .await;
    assert_eq!(
        init.pointer("/result/serverInfo/name"),
        Some(&json!(SERVER_NAME))
    );
    app_server_notification(&server, METHOD_INITIALIZED, json!({})).await;

    let line = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": METHOD_SKILL_CACHE_REFRESH,
        "params": {}
    })
    .to_string();
    let responses = server
        .handle_json_line(&line)
        .await
        .expect("handle Skill catalog mutation");

    assert_eq!(responses.len(), 2, "{responses:?}");
    let response: Value = serde_json::from_str(&responses[0]).expect("decode response");
    let notification = decode_message(&responses[1]).expect("decode notification");
    assert_eq!(response["result"], json!({ "success": true }));
    assert_eq!(
        notification,
        app_server_protocol::JsonRpcMessage::Notification(
            app_server_protocol::protocol::v2::ServerNotification::SkillsChanged(
                app_server_protocol::protocol::v2::SkillsChangedNotification {},
            )
            .into(),
        )
    );
}

#[tokio::test]
async fn diagnostics_use_injected_agent_log_root() {
    let temp = tempfile::tempdir().expect("tempdir");
    let agent_root = temp.path().join("app-server");
    let mut data_source = setup_data_source();
    data_source.logs = Arc::new(tokio::sync::RwLock::new(
        lime_core::logger::LogStore::new(&agent_root).expect("log store"),
    ));
    data_source
        .logs
        .write()
        .await
        .add("info", "injected log root");

    let diagnostics = data_source
        .read_log_storage_diagnostics()
        .await
        .expect("log storage diagnostics");
    let listed = data_source.list_logs().await.expect("list logs");
    let expected = agent_root
        .join("observability")
        .join("log")
        .join("lime.log");
    let expected = expected.to_string_lossy().to_string();

    assert_eq!(
        diagnostics.current_log_path.as_deref(),
        Some(expected.as_str())
    );
    assert_eq!(diagnostics.in_memory_log_count, 1);
    assert_eq!(listed.entries.len(), 1);
    assert_eq!(listed.entries[0].message, "injected log root");
    assert!(!agent_root.join("logs").exists());
}

#[tokio::test]
async fn right_surface_object_canvas_snapshot_persists_in_local_sqlite_data_source() {
    let data_source = Arc::new(setup_data_source());
    let core = RuntimeCore::default().with_app_data_source(data_source.clone());

    let response = core
        .request_workspace_right_surface(WorkspaceRightSurfaceRequestParams {
            workspace_id: Some(WORKSPACE_ID.to_string()),
            workspace_root: Some(WORKSPACE_ROOT.to_string()),
            session_id: Some("browser-session-local".to_string()),
            surface_kind: "objectCanvas".to_string(),
            origin: "runtime".to_string(),
            reason: Some("object_canvas_persist_requested".to_string()),
            priority: None,
            candidate_id: Some("browser-assist-local".to_string()),
            ttl_ms: None,
            metadata: Some(json!({
                "source": "objectCanvas",
                "schemaVersion": "object-canvas.persist.v1",
                "candidateId": "browser-assist-local",
                "objectCanvas": {
                    "board": {
                        "id": "object-canvas-board:browser-assist-local",
                        "revision": 1,
                        "primaryObjectId": "browser-session:browser-assist-local",
                        "objectCount": 1,
                        "edgeCount": 0
                    },
                    "event": {
                        "kind": "persistRequested",
                        "owner": "appServer",
                        "enabled": false,
                        "request": {
                            "boardId": "object-canvas-board:browser-assist-local",
                            "revision": 1,
                            "persistenceKey": "workspace:object-canvas:browser-assist-local",
                            "objectId": "browser-session:browser-assist-local",
                            "objectKind": "browserSession",
                            "facts": {
                                "candidateId": "browser-assist-local",
                                "sessionId": "browser-session-local"
                            }
                        }
                    }
                }
            })),
        })
        .await
        .expect("right surface persist request");

    let snapshots = data_source
        .list_workspace_object_canvas_snapshots(WorkspaceObjectCanvasSnapshotListParams {
            workspace_id: Some(WORKSPACE_ID.to_string()),
            workspace_root: Some(WORKSPACE_ROOT.to_string()),
            session_id: Some("browser-session-local".to_string()),
            board_id: Some("object-canvas-board:browser-assist-local".to_string()),
            persistence_key: Some("workspace:object-canvas:browser-assist-local".to_string()),
            limit: None,
        })
        .await
        .expect("list persisted object canvas snapshots");

    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].request_id, response.request_id);
    assert_eq!(
        snapshots[0]
            .snapshot_json
            .pointer("/metadata/objectCanvas/event/request/persistenceKey")
            .and_then(Value::as_str),
        Some("workspace:object-canvas:browser-assist-local")
    );

    let pending = core
        .list_workspace_right_surface_pending(WorkspaceRightSurfacePendingListParams {
            workspace_id: Some(WORKSPACE_ID.to_string()),
            workspace_root: Some(WORKSPACE_ROOT.to_string()),
            session_id: Some("browser-session-local".to_string()),
            surface_kind: Some("objectCanvas".to_string()),
            limit: None,
        })
        .await
        .expect("list persisted pending requests");
    assert_eq!(pending.pending.len(), 1);
    assert_eq!(pending.pending[0].request_id, response.request_id);
}

#[tokio::test]
async fn mcp_current_jsonrpc_starts_real_stdio_server_and_reads_resource() {
    let temp_dir = TempDir::new().expect("create mcp fixture temp dir");
    let server_path = write_mcp_stdio_fixture(temp_dir.path());
    let server = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(MockBackend))
            .with_app_data_source(Arc::new(setup_data_source())),
    );

    let init = app_server_request(
        &server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "mcp-current-fixture",
                "version": "test"
            },
            "capabilities": {
                "experimental": true
            }
        }),
    )
    .await;
    assert_eq!(
        init.pointer("/result/serverInfo/name"),
        Some(&json!(SERVER_NAME))
    );
    app_server_notification(&server, METHOD_INITIALIZED, json!({})).await;

    let create = app_server_request(
        &server,
        2,
        METHOD_MCP_SERVER_CREATE,
        json!({
            "server": {
                "id": "fixture",
                "name": "fixture",
                "description": "Current MCP JSON-RPC fixture",
                "server_config": {
                    "command": "node",
                    "args": [server_path.to_string_lossy()],
                    "cwd": temp_dir.path().to_string_lossy(),
                    "timeout": 3
                },
                "enabled_lime": true,
                "enabled_claude": false,
                "enabled_codex": false,
                "enabled_gemini": false,
                "created_at": 1
            }
        }),
    )
    .await;
    assert_eq!(
        create.pointer("/result/servers/0/name"),
        Some(&json!("fixture"))
    );

    let start = app_server_request(
        &server,
        3,
        METHOD_MCP_SERVER_START,
        json!({ "name": "fixture" }),
    )
    .await;
    assert!(start.get("result").is_some(), "{start:?}");

    let status = app_server_request(&server, 4, METHOD_MCP_SERVER_STATUS_LIST, json!({})).await;
    assert_eq!(
        status.pointer("/result/servers/0/is_running"),
        Some(&json!(true))
    );
    assert_eq!(
        status.pointer("/result/servers/0/server_info/supports_tools"),
        Some(&json!(true))
    );
    assert_eq!(
        status.pointer("/result/servers/0/server_info/supports_resources"),
        Some(&json!(true))
    );

    let tools = app_server_request(&server, 5, METHOD_MCP_TOOL_LIST, json!({})).await;
    assert_eq!(
        tools.pointer("/result/tools/0/name"),
        Some(&json!("mcp__fixture__echo"))
    );

    let resources = app_server_request(&server, 6, METHOD_MCP_RESOURCE_LIST, json!({})).await;
    assert_eq!(
        resources.pointer("/result/resources/0/uri"),
        Some(&json!("fixture://status")),
        "{resources:?}"
    );
    assert_eq!(
        resources.pointer("/result/resources/0/server_name"),
        Some(&json!("fixture")),
        "{resources:?}"
    );
    assert_eq!(
        resources.pointer("/result/resourceTemplates/0/uri_template"),
        Some(&json!("fixture://item/{id}")),
        "{resources:?}"
    );
    assert_eq!(
        resources.pointer("/result/resourceTemplates/0/server_name"),
        Some(&json!("fixture"))
    );

    let resource = app_server_request(
        &server,
        7,
        METHOD_MCP_SERVER_RESOURCE_READ,
        json!({
            "server": "fixture",
            "uri": "fixture://status"
        }),
    )
    .await;
    assert_eq!(
        resource.pointer("/result/contents/0/uri"),
        Some(&json!("fixture://status"))
    );
    assert_eq!(
        resource.pointer("/result/contents/0/text"),
        Some(&json!("fixture resource ok"))
    );

    let subscribe = app_server_request(
        &server,
        9,
        METHOD_MCP_RESOURCE_SUBSCRIBE,
        json!({
            "server": "fixture",
            "uri": "fixture://status"
        }),
    )
    .await;
    assert_eq!(subscribe.pointer("/result"), Some(&json!({})));

    let unsubscribe = app_server_request(
        &server,
        10,
        METHOD_MCP_RESOURCE_UNSUBSCRIBE,
        json!({
            "server": "fixture",
            "uri": "fixture://status"
        }),
    )
    .await;
    assert_eq!(unsubscribe.pointer("/result"), Some(&json!({})));

    let stop = app_server_request(
        &server,
        11,
        METHOD_MCP_SERVER_STOP,
        json!({ "name": "fixture" }),
    )
    .await;
    assert!(stop.get("result").is_some(), "{stop:?}");
}

fn write_mcp_stdio_fixture(root: &Path) -> PathBuf {
    let path = root.join("mcp-current-fixture.mjs");
    fs::write(
        &path,
        r#"
import readline from "node:readline";

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

function send(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function result(id, value) {
  send({ jsonrpc: "2.0", id, result: value });
}

function error(id, code, message) {
  send({ jsonrpc: "2.0", id, error: { code, message } });
}

rl.on("line", (line) => {
  if (!line.trim()) return;
  const message = JSON.parse(line);
  const { id, method, params } = message;

  if (method === "initialize") {
    result(id, {
      protocolVersion: "2025-03-26",
      capabilities: {
        tools: {},
        resources: {},
      },
      serverInfo: {
        name: "fixture-mcp",
        version: "1.0.0",
      },
    });
    return;
  }

  if (method === "notifications/initialized") return;

  if (method === "tools/list") {
    result(id, {
      tools: [
        {
          name: "echo",
          description: "Echo a message for current MCP tests",
          inputSchema: {
            type: "object",
            properties: {
              message: { type: "string" },
            },
          },
        },
      ],
    });
    return;
  }

  if (method === "tools/call") {
    result(id, {
      content: [
        {
          type: "text",
          text: `echo: ${params?.arguments?.message ?? ""}`,
        },
      ],
      isError: false,
    });
    return;
  }

  if (method === "resources/list") {
    result(id, {
      resources: [
        {
          uri: "fixture://status",
          name: "status",
          description: "Current MCP fixture status",
          mimeType: "text/plain",
        },
      ],
    });
    return;
  }

  if (method === "resources/templates/list") {
    result(id, {
      resourceTemplates: [
        {
          uriTemplate: "fixture://item/{id}",
          name: "fixture-item",
          title: "Fixture Item",
          description: "Current MCP fixture resource template",
          mimeType: "text/plain",
        },
      ],
    });
    return;
  }

  if (method === "resources/read") {
    result(id, {
      contents: [
        {
          uri: params?.uri ?? "fixture://status",
          mimeType: "text/plain",
          text: "fixture resource ok",
        },
      ],
    });
    return;
  }

  if (method === "resources/subscribe" || method === "resources/unsubscribe") {
    result(id, {});
    return;
  }

  error(id, -32601, `unsupported fixture method: ${method}`);
});
"#,
    )
    .expect("write mcp fixture");
    path
}

async fn app_server_request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let line = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    })
    .to_string();
    let responses = server
        .handle_json_line(&line)
        .await
        .expect("handle app-server request");
    assert_eq!(responses.len(), 1, "{responses:?}");
    let message = decode_message(&responses[0]).expect("decode app-server response");
    serde_json::to_value(message).expect("serialize app-server response")
}

async fn app_server_notification(server: &AppServer, method: &str, params: Value) {
    let line = json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
    })
    .to_string();
    let responses = server
        .handle_json_line(&line)
        .await
        .expect("handle app-server notification");
    assert!(responses.is_empty(), "{responses:?}");
}
