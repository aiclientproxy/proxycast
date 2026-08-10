use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use agent_protocol::hook::{
    HookEventName, HookExecutionMode, HookHandlerType, HookRunStatus, HookScope, HookSource,
};
use app_server::{
    AppServer, LocalAppDataSource, MockBackend, ProjectionStore, RuntimeCore, RuntimeEvent,
};
use app_server_protocol::protocol::v2::METHOD_HOOKS_LIST;
use app_server_protocol::{
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_ITEMS_LIST, METHOD_THREAD_READ,
    METHOD_THREAD_START, METHOD_TURN_START,
};
use lime_core::database::schema::create_tables;
use rusqlite::Connection;
use serde_json::{json, Value};
use tempfile::TempDir;

#[tokio::test]
async fn hook_lifecycle_stays_out_of_public_thread_history() {
    let temp = TempDir::new().expect("hook history temp dir");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("hook history projection store"),
    );
    let runtime =
        RuntimeCore::with_backend(Arc::new(MockBackend)).with_projection_store(projection_store);
    let server = AppServer::with_runtime(runtime.clone());
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
        .expect("thread id")
        .to_string();
    let session_id = started["result"]["thread"]["sessionId"]
        .as_str()
        .expect("session id")
        .to_string();
    let turn = request(
        &server,
        3,
        METHOD_TURN_START,
        json!({
            "threadId": thread_id,
            "input": [{"type": "text", "text": "run the hook"}]
        }),
    )
    .await;
    let turn_id = turn["result"]["turn"]["id"]
        .as_str()
        .expect("turn id")
        .to_string();

    let run_id = "hook-run-public-history";
    let running = hook_run(run_id, HookRunStatus::Running, 1_800_000_000_000, None);
    let completed = hook_run(
        run_id,
        HookRunStatus::Completed,
        1_800_000_000_000,
        Some(1_800_000_000_025),
    );
    runtime
        .event_appender()
        .append_external_runtime_events(
            &session_id,
            Some(&turn_id),
            vec![hook_event(&running), hook_event(&completed)],
        )
        .expect("persist hook lifecycle");

    let read = request(
        &server,
        4,
        METHOD_THREAD_READ,
        json!({"threadId": thread_id, "includeTurns": true}),
    )
    .await;
    let read_items = read["result"]["thread"]["turns"]
        .as_array()
        .expect("read turns")
        .iter()
        .flat_map(|turn| turn["items"].as_array().into_iter().flatten())
        .collect::<Vec<_>>();
    assert!(read_items.iter().all(|item| item["type"] != "hook"));

    let items = request(
        &server,
        5,
        METHOD_THREAD_ITEMS_LIST,
        json!({"threadId": thread_id, "turnId": turn_id}),
    )
    .await;
    let entries = items["result"]["data"].as_array().expect("item list");
    assert!(entries.iter().all(|entry| entry["item"]["type"] != "hook"));
}

#[tokio::test]
async fn hooks_list_returns_codex_metadata_over_public_jsonrpc() {
    let codex_home = TempDir::new().expect("codex home");
    let cwd = TempDir::new().expect("project cwd");
    fs::create_dir_all(cwd.path().join(".codex")).expect("project config directory");
    fs::write(
        cwd.path().join(".codex/config.toml"),
        r#"[hooks]

[[hooks.PreToolUse]]
matcher = "Bash"

[[hooks.PreToolUse.hooks]]
type = "command"
command = "python3 /tmp/project-hook.py"
timeout = 7
statusMessage = "checking project"
additionalContextLimit = 4096
"#,
    )
    .expect("project hook config");

    let previous_codex_home = std::env::var_os("CODEX_HOME");
    let _restore_codex_home = scopeguard::guard(previous_codex_home, |value| {
        if let Some(value) = value {
            std::env::set_var("CODEX_HOME", value);
        } else {
            std::env::remove_var("CODEX_HOME");
        }
    });
    std::env::set_var("CODEX_HOME", codex_home.path());

    let connection = Connection::open_in_memory().expect("product db");
    create_tables(&connection).expect("product schema");
    let app_data_source = LocalAppDataSource::initialize_with_roots(
        Arc::new(Mutex::new(connection)),
        cwd.path(),
        cwd.path().join("app-server"),
    )
    .await
    .expect("app data source");
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_app_data_source(Arc::new(app_data_source));
    let server = AppServer::with_runtime(runtime);

    request(
        &server,
        1,
        METHOD_INITIALIZE,
        json!({"clientInfo": {"name": "hooks-jsonrpc-test", "version": "1"}}),
    )
    .await;
    server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {}}).to_string(),
        )
        .await
        .expect("initialized notification");

    let response = request(&server, 2, METHOD_HOOKS_LIST, json!({"cwds": [cwd.path()]})).await;
    let entry = &response["result"]["data"][0];
    assert_eq!(entry["cwd"], cwd.path().to_string_lossy().as_ref());
    assert!(entry["warnings"].as_array().is_some_and(Vec::is_empty));
    assert!(entry["errors"].as_array().is_some_and(Vec::is_empty));
    let hook = &entry["hooks"][0];
    assert_eq!(hook["eventName"], "preToolUse");
    assert_eq!(hook["handlerType"], "command");
    assert_eq!(hook["executionMode"], "sync");
    assert_eq!(hook["matcher"], "Bash");
    assert_eq!(hook["command"], "python3 /tmp/project-hook.py");
    assert_eq!(hook["timeoutSec"], 7);
    assert_eq!(hook["statusMessage"], "checking project");
    assert_eq!(hook["additionalContextLimit"], 4096);
    assert_eq!(hook["source"], "project");
    assert_eq!(hook["trustStatus"], "untrusted");
    assert_eq!(hook["enabled"], true);
    assert_eq!(hook["isManaged"], false);
    assert!(hook["key"]
        .as_str()
        .is_some_and(|key| key.contains("pre_tool_use:0:0")));
    assert!(hook["currentHash"]
        .as_str()
        .is_some_and(|hash| hash.starts_with("sha256:")));
}

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let lines = server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string(),
        )
        .await
        .expect("JSON-RPC request");
    let response = lines
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("decode response"))
        .find(|value| value.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("{method} should return response id {id}: {lines:?}"));
    assert!(
        response.get("error").is_none(),
        "request failed: {response:#}"
    );
    response
}

async fn initialize(server: &AppServer) {
    request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "hooks-jsonrpc-test",
                "version": "1"
            }
        }),
    )
    .await;
    let lines = server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {}}).to_string(),
        )
        .await
        .expect("initialized notification");
    assert!(lines.is_empty());
}

fn hook_run(
    id: &str,
    status: HookRunStatus,
    started_at: i64,
    completed_at: Option<i64>,
) -> agent_protocol::hook::HookRunSummary {
    agent_protocol::hook::HookRunSummary {
        id: id.to_string(),
        event_name: HookEventName::PreToolUse,
        handler_type: HookHandlerType::Command,
        execution_mode: HookExecutionMode::Sync,
        scope: HookScope::Turn,
        source_path: PathBuf::from("/tmp/public-history-hook.sh"),
        source: HookSource::Project,
        display_order: 0,
        status,
        status_message: None,
        started_at,
        completed_at,
        duration_ms: completed_at.map(|value| value - started_at),
        entries: Vec::new(),
    }
}

fn hook_event(run: &agent_protocol::hook::HookRunSummary) -> RuntimeEvent {
    RuntimeEvent::new(
        match run.status {
            HookRunStatus::Running => "hook.started",
            _ => "hook.completed",
        },
        json!({"run": run}),
    )
}
