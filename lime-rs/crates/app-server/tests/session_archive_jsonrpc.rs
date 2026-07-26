use app_server::AppServer;
use app_server::EventLogWriter;
use app_server::LocalAppDataSource;
use app_server::MockBackend;
use app_server::ProjectionStore;
use app_server::RuntimeCore;
use app_server::StorageRoots;
use app_server_protocol::*;
use lime_core::database::schema::create_tables;
use rusqlite::Connection;
use serde_json::json;
use serde_json::Value;
use std::fs;
use std::sync::Arc;
use tempfile::TempDir;

const SESSION_ID: &str = "persisted-session";
const THREAD_ID: &str = "persisted-thread";
const WORKSPACE_ID: &str = "workspace-current";

struct ProjectionAppServer {
    _temp: TempDir,
    roots: StorageRoots,
    event_log_writer: Arc<EventLogWriter>,
    server: AppServer,
}

async fn projection_app_server(sessions: &[(&str, &str, &str, &str)]) -> ProjectionAppServer {
    let temp = TempDir::new().expect("create projection fixture temp dir");
    let roots = StorageRoots::initialize(temp.path(), temp.path().join("app-server"))
        .expect("storage roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("event log"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let app_data_source = Arc::new(local_app_data_source(&roots).await);
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_app_data_source(app_data_source)
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());
    for (session_id, thread_id, title, updated_at) in sessions {
        runtime
            .start_session(AgentSessionStartParams {
                session_id: Some((*session_id).to_string()),
                thread_id: Some((*thread_id).to_string()),
                app_id: "session-archive-jsonrpc-test".to_string(),
                workspace_id: Some(WORKSPACE_ID.to_string()),
                business_object_ref: None,
                locale: None,
            })
            .expect("seed canonical thread");
        seed_projected_session(
            &projection_store,
            &event_log_writer,
            session_id,
            thread_id,
            title,
            updated_at,
        );
    }
    ProjectionAppServer {
        _temp: temp,
        roots,
        event_log_writer,
        server: AppServer::with_runtime(runtime),
    }
}

async fn local_app_data_source(roots: &StorageRoots) -> LocalAppDataSource {
    let conn = Connection::open_in_memory().expect("open in-memory product db");
    create_tables(&conn).expect("create product schema");
    LocalAppDataSource::initialize_with_roots(
        Arc::new(std::sync::Mutex::new(conn)),
        roots.app_data_root.clone(),
        roots.data_root.clone(),
    )
    .await
    .expect("local app data source")
}

fn seed_projected_session(
    projection_store: &ProjectionStore,
    event_log_writer: &EventLogWriter,
    session_id: &str,
    thread_id: &str,
    title: &str,
    updated_at: &str,
) {
    let event = AgentEvent {
        event_id: format!("evt-{session_id}-accepted"),
        sequence: 1,
        session_id: session_id.to_string(),
        thread_id: Some(thread_id.to_string()),
        turn_id: Some(format!("{thread_id}-turn")),
        event_type: "turn.accepted".to_string(),
        timestamp: updated_at.to_string(),
        payload: json!({
            "session": {
                "title": title,
                "modelName": "gpt-5.4",
                "workspaceId": WORKSPACE_ID,
                "workingDir": "/tmp/workspace-current",
                "executionStrategy": "react"
            }
        }),
    };
    projection_store
        .apply_event(&event)
        .expect("seed projected session");
    event_log_writer.append(&event).expect("seed event log");
}

#[tokio::test]
async fn memory_store_reset_does_not_delete_persisted_session_history() {
    let app = projection_app_server(&[(
        SESSION_ID,
        THREAD_ID,
        "Persisted Session",
        "2026-06-07T00:00:00.000Z",
    )])
    .await;
    initialize_server(&app.server, 1, "memory-reset-history-jsonrpc-test").await;

    let note = request(
        &app.server,
        2,
        METHOD_MEMORY_STORE_ADD_NOTE,
        json!({
            "scope": "global",
            "title": "Reset isolation",
            "slug": "reset-isolation",
            "content": "This memory note should be removed by reset."
        }),
    )
    .await;
    let note_path = note
        .pointer("/result/path")
        .and_then(Value::as_str)
        .expect("memory note path");
    let memory_root = app.roots.memory_root.clone();
    fs::write(
        memory_root.join("memory_summary.md"),
        "summary that should be cleared",
    )
    .expect("write summary");
    let event_log_path = app
        .event_log_writer
        .read_session_events(SESSION_ID)
        .expect("seeded event log")
        .into_iter()
        .next()
        .expect("one event log record")
        .path;

    let reset = request(
        &app.server,
        3,
        METHOD_MEMORY_STORE_RESET,
        json!({
            "scope": "global"
        }),
    )
    .await;
    assert_eq!(reset.pointer("/result/preservedSoul"), Some(&json!(true)));
    assert!(!memory_root.join(note_path).exists());
    assert_eq!(
        fs::read_to_string(memory_root.join("memory_summary.md")).expect("reset summary"),
        ""
    );

    let recent = request(&app.server, 4, METHOD_THREAD_LIST, json!({})).await;
    assert_eq!(thread_ids(&recent), vec![THREAD_ID.to_string()]);

    let read = request(
        &app.server,
        5,
        METHOD_THREAD_READ,
        json!({
            "threadId": THREAD_ID
        }),
    )
    .await;
    assert_eq!(
        read.pointer("/result/thread/sessionId"),
        Some(&json!(SESSION_ID)),
    );
    assert!(event_log_path.is_file());
    assert_eq!(
        app.event_log_writer
            .read_session_events(SESSION_ID)
            .expect("event log after reset")
            .len(),
        1
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

fn thread_ids(response: &Value) -> Vec<String> {
    response
        .pointer("/result/data")
        .and_then(Value::as_array)
        .expect("result.data should be an array")
        .iter()
        .map(|thread| {
            thread
                .get("id")
                .and_then(Value::as_str)
                .expect("thread id should be a string")
                .to_string()
        })
        .collect()
}
