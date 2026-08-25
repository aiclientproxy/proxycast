use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use app_server::{
    run_json_lines, AppServer, EventLogWriter, MockBackend, ProjectionStore, RuntimeCore,
    SidecarStore, SidecarWriteRequest,
};
use app_server_protocol::protocol::v2::{METHOD_THREAD_DELETE, METHOD_THREAD_DELETED};
use app_server_protocol::{
    error_codes, AgentEvent, AgentSessionStartParams, METHOD_INITIALIZE, METHOD_INITIALIZED,
    METHOD_THREAD_READ, METHOD_THREAD_START,
};
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use tempfile::TempDir;
use thread_store::{
    AgentGraphStore, AgentIdentity, AgentIdentityStore, AgentMailboxDeliveryMode,
    AgentMailboxDeliveryStatus, AgentMailboxMessage, AgentMailboxMessageKind, AgentMailboxStore,
    AppendAgentMailboxMessageParams, ThreadSpawnEdgeStatus,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::time::timeout;

#[tokio::test]
async fn thread_delete_removes_every_persisted_owner_over_public_jsonl() {
    let temp = TempDir::new().expect("thread delete temp dir");
    let projection_path = temp.path().join("projection.sqlite");
    let state_path = temp.path().join("state.sqlite");
    let history_path = temp.path().join("thread-history.sqlite");
    let agent_root = temp.path().join("agent-root");
    let event_log = Arc::new(
        EventLogWriter::new(temp.path().join("event-log")).expect("thread delete event log"),
    );
    let sidecar_root = temp.path().join("sidecars");
    let sidecar = Arc::new(SidecarStore::new(&sidecar_root).expect("thread delete sidecar"));
    let store = Arc::new(
        ProjectionStore::initialize_with_storage_paths(
            &projection_path,
            &state_path,
            &history_path,
            &agent_root,
        )
        .expect("thread delete projection store"),
    );
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_projection_store(store.clone())
        .with_event_log_writer(event_log.clone())
        .with_sidecar_store(sidecar.clone());
    let runtime_handle = runtime.clone();
    let server = AppServer::with_runtime(runtime);

    let (mut input_client, input_server) = tokio::io::duplex(32 * 1024);
    let (output_server, output_client) = tokio::io::duplex(32 * 1024);
    let runner = tokio::spawn(run_json_lines(server, input_server, output_server));
    let mut output_lines = BufReader::new(output_client).lines();

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": METHOD_INITIALIZE,
            "params": {
                "clientInfo": {
                    "name": "thread-delete-jsonrpc-test",
                    "version": "1.0.0"
                }
            }
        }),
    )
    .await;
    assert_response_ok(&read_response(&mut output_lines, 1).await, "initialize");
    write_message(
        &mut input_client,
        json!({ "jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {} }),
    )
    .await;

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": METHOD_THREAD_START,
            "params": {
                "model": "fixture-model",
                "modelProvider": "fixture-provider",
                "cwd": temp.path()
            }
        }),
    )
    .await;
    let start = read_response(&mut output_lines, 2).await;
    assert_response_ok(&start, "thread/start");
    let thread_id = required_string(&start, "/result/thread/id", "thread id");
    let session_id = required_string(&start, "/result/thread/sessionId", "session id");
    let started = next_message(&mut output_lines).await;
    assert_eq!(started.get("method"), Some(&json!("thread/started")));
    let child_thread_id = format!("{thread_id}_child");
    let child_session_id = format!("{session_id}_child");
    let pending_thread_id = format!("{child_thread_id}_pending");
    let pending_session_id = format!("{child_session_id}_pending");
    runtime_handle
        .start_session(AgentSessionStartParams {
            session_id: Some(child_session_id.clone()),
            thread_id: Some(child_thread_id.clone()),
            app_id: "thread-delete-jsonrpc-test".to_string(),
            workspace_id: None,
            business_object_ref: None,
            locale: None,
        })
        .expect("start child thread for subtree delete");
    store
        .upsert_thread_spawn_edge(
            agent_protocol::ThreadId::new(thread_id.clone()),
            agent_protocol::ThreadId::new(child_thread_id.clone()),
            ThreadSpawnEdgeStatus::Open,
        )
        .await
        .expect("seed thread delete child edge");
    store
        .create_pending_thread_spawn_edge(
            agent_protocol::ThreadId::new(child_thread_id.clone()),
            agent_protocol::ThreadId::new(pending_thread_id.clone()),
            pending_session_id.clone(),
        )
        .await
        .expect("seed pending-only thread delete child edge");

    event_log
        .append(&AgentEvent {
            event_id: "evt-thread-delete-proof".to_string(),
            sequence: next_event_sequence(&event_log, &session_id),
            session_id: session_id.clone(),
            thread_id: Some(thread_id.clone()),
            turn_id: None,
            event_type: "thread.delete.test".to_string(),
            timestamp: "2026-07-21T00:00:00Z".to_string(),
            payload: json!({}),
        })
        .expect("seed thread delete event log");

    let sidecar_ref = sidecar
        .write_text(&SidecarWriteRequest {
            session_id: session_id.clone(),
            kind: "thread-delete-proof".to_string(),
            logical_id: "delete-proof".to_string(),
            relative_path: format!("sessions/{session_id}/delete-proof.txt"),
            content: "delete me".to_string(),
        })
        .expect("write thread delete sidecar");
    let sidecar_path = sidecar_root.join(&sidecar_ref.relative_path);
    let pending_sidecar_ref = sidecar
        .write_text(&SidecarWriteRequest {
            session_id: pending_session_id.clone(),
            kind: "thread-delete-pending-proof".to_string(),
            logical_id: "delete-pending-proof".to_string(),
            relative_path: format!("sessions/{pending_session_id}/delete-proof.txt"),
            content: "delete pending child".to_string(),
        })
        .expect("write pending child thread delete sidecar");
    let pending_sidecar_path = sidecar_root.join(&pending_sidecar_ref.relative_path);
    event_log
        .append(&AgentEvent {
            event_id: "evt-thread-delete-pending-proof".to_string(),
            sequence: next_event_sequence(&event_log, &pending_session_id),
            session_id: pending_session_id.clone(),
            thread_id: Some(pending_thread_id.clone()),
            turn_id: None,
            event_type: "thread.delete.pending.test".to_string(),
            timestamp: "2026-07-21T00:00:00Z".to_string(),
            payload: json!({}),
        })
        .expect("seed pending child thread delete event log");
    assert!(sidecar_path.exists());
    assert!(pending_sidecar_path.exists());
    assert_eq!(rollout_files(&agent_root).len(), 2);
    assert!(!event_log
        .read_session_events(&session_id)
        .expect("read thread delete event log")
        .is_empty());
    store
        .upsert_agent_identity(AgentIdentity {
            root_thread_id: agent_protocol::ThreadId::new(thread_id.clone()),
            thread_id: agent_protocol::ThreadId::new(thread_id.clone()),
            agent_path: "/root".to_string(),
            nickname: None,
            role: None,
            last_task_message: None,
        })
        .await
        .expect("seed thread delete identity");
    store
        .append_agent_mailbox_message(AppendAgentMailboxMessageParams {
            message: AgentMailboxMessage {
                message_id: "thread-delete-mail".to_string(),
                root_thread_id: agent_protocol::ThreadId::new(thread_id.clone()),
                sender_thread_id: agent_protocol::ThreadId::new(thread_id.clone()),
                recipient_thread_id: agent_protocol::ThreadId::new(thread_id.clone()),
                content: "delete me".to_string(),
                kind: AgentMailboxMessageKind::Message,
                source_turn_id: None,
                result_status: None,
                delivery_mode: AgentMailboxDeliveryMode::QueueOnly,
                delivery_status: AgentMailboxDeliveryStatus::Pending,
                created_at_ms: 1,
                delivered_at_ms: None,
            },
        })
        .await
        .expect("seed thread delete mailbox");

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": METHOD_THREAD_DELETE,
            "params": { "threadId": thread_id }
        }),
    )
    .await;
    let deleted = read_response(&mut output_lines, 3).await;
    assert_response_ok(&deleted, "thread/delete");
    assert_eq!(deleted.get("result"), Some(&json!({})));
    let pending_notification = next_message(&mut output_lines).await;
    assert_eq!(
        pending_notification.get("method"),
        Some(&json!(METHOD_THREAD_DELETED))
    );
    assert_eq!(
        pending_notification.pointer("/params/threadId"),
        Some(&json!(pending_thread_id))
    );
    let child_notification = next_message(&mut output_lines).await;
    assert_eq!(
        child_notification.get("method"),
        Some(&json!(METHOD_THREAD_DELETED))
    );
    assert_eq!(
        child_notification.pointer("/params/threadId"),
        Some(&json!(child_thread_id))
    );
    let notification = next_message(&mut output_lines).await;
    assert_eq!(
        notification.get("method"),
        Some(&json!(METHOD_THREAD_DELETED))
    );
    assert_eq!(
        notification.pointer("/params/threadId"),
        Some(&json!(thread_id))
    );

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": METHOD_THREAD_READ,
            "params": { "threadId": thread_id }
        }),
    )
    .await;
    let read_after_delete = read_response(&mut output_lines, 4).await;
    assert_eq!(
        read_after_delete.pointer("/error/code"),
        Some(&json!(error_codes::RUNTIME_ERROR))
    );
    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": METHOD_THREAD_READ,
            "params": { "threadId": child_thread_id }
        }),
    )
    .await;
    let child_read_after_delete = read_response(&mut output_lines, 5).await;
    assert_eq!(
        child_read_after_delete.pointer("/error/code"),
        Some(&json!(error_codes::RUNTIME_ERROR))
    );
    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 6,
            "method": METHOD_THREAD_READ,
            "params": { "threadId": pending_thread_id }
        }),
    )
    .await;
    let pending_read_after_delete = read_response(&mut output_lines, 6).await;
    assert_eq!(
        pending_read_after_delete.pointer("/error/code"),
        Some(&json!(error_codes::RUNTIME_ERROR))
    );

    assert!(rollout_files(&agent_root).is_empty());
    assert!(event_log
        .read_session_events(&session_id)
        .expect("read deleted event log")
        .is_empty());
    assert!(event_log
        .read_session_events(&pending_session_id)
        .expect("read deleted pending child event log")
        .is_empty());
    assert!(!sidecar_path.exists());
    assert!(!pending_sidecar_path.exists());
    assert_eq!(
        count_rows(&state_path, "canonical_threads", "thread_id", &thread_id),
        0
    );
    assert_eq!(
        count_rows(
            &state_path,
            "canonical_threads",
            "thread_id",
            &child_thread_id
        ),
        0
    );
    assert_eq!(
        count_rows(&history_path, "canonical_turns", "thread_id", &thread_id),
        0
    );
    assert_eq!(
        count_rows(&history_path, "canonical_items", "thread_id", &thread_id),
        0
    );
    assert_eq!(
        count_rows(
            &projection_path,
            "projected_sessions",
            "session_id",
            &session_id
        ),
        0
    );
    assert_eq!(
        count_rows(
            &projection_path,
            "projected_turns",
            "session_id",
            &session_id
        ),
        0
    );
    assert_eq!(
        count_rows(
            &projection_path,
            "projected_items",
            "session_id",
            &session_id
        ),
        0
    );
    assert_eq!(
        count_rows(
            &projection_path,
            "agent_identities",
            "thread_id",
            &thread_id
        ),
        0
    );
    assert_eq!(
        count_rows(
            &projection_path,
            "agent_mailbox_messages",
            "root_thread_id",
            &thread_id
        ),
        0
    );

    drop(input_client);
    timeout(Duration::from_secs(2), runner)
        .await
        .expect("JSONL runner should stop after input closes")
        .expect("JSONL runner task")
        .expect("JSONL runner result");

    let restarted = AppServer::with_runtime(
        RuntimeCore::default().with_projection_store(Arc::new(
            ProjectionStore::initialize_with_storage_paths(
                &projection_path,
                &state_path,
                &history_path,
                &agent_root,
            )
            .expect("reopen deleted thread stores"),
        )),
    );
    initialize_direct(&restarted).await;
    let read = restarted
        .handle_json_line(&format!(
            "{}\n",
            json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": METHOD_THREAD_READ,
                "params": { "threadId": thread_id }
            })
        ))
        .await
        .expect("thread/read after restart");
    let read = read
        .iter()
        .filter_map(|message| serde_json::from_str::<Value>(message.trim()).ok())
        .collect::<Vec<_>>();
    assert_eq!(
        read.iter()
            .find(|message| message.get("id") == Some(&json!(2)))
            .and_then(|message| message.pointer("/error/code").cloned()),
        Some(json!(error_codes::RUNTIME_ERROR)),
        "thread/read after restart must fail: {read:#?}"
    );
}

fn next_event_sequence(event_log: &EventLogWriter, session_id: &str) -> u64 {
    event_log
        .read_session_events(session_id)
        .expect("read event log before seeding delete proof")
        .last()
        .map(|record| record.event.sequence + 1)
        .unwrap_or(1)
}

async fn initialize_direct(server: &AppServer) {
    for line in [
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": METHOD_INITIALIZE,
            "params": { "clientInfo": { "name": "restart", "version": "1.0" } }
        }),
        json!({ "jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {} }),
    ] {
        server
            .handle_json_line(&format!("{line}\n"))
            .await
            .expect("initialize restarted server");
    }
}

fn count_rows(path: &Path, table: &str, column: &str, value: &str) -> i64 {
    let conn = Connection::open(path).expect("open SQLite evidence");
    conn.query_row(
        &format!("SELECT COUNT(*) FROM {table} WHERE {column} = ?1"),
        params![value],
        |row| row.get(0),
    )
    .expect("query SQLite deletion evidence")
}

fn rollout_files(root: &Path) -> Vec<PathBuf> {
    if !root.exists() {
        return Vec::new();
    }
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(path).expect("read rollout directory") {
            let path = entry.expect("read rollout entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().and_then(|value| value.to_str()) == Some("jsonl") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

async fn write_message(client: &mut DuplexStream, message: Value) {
    client
        .write_all(format!("{message}\n").as_bytes())
        .await
        .expect("write JSONL request");
}

async fn read_response(lines: &mut Lines<BufReader<DuplexStream>>, id: i64) -> Value {
    for _ in 0..64 {
        let message = next_message(lines).await;
        if message.get("id") == Some(&json!(id)) {
            return message;
        }
    }
    panic!("response {id} not found");
}

async fn next_message(lines: &mut Lines<BufReader<DuplexStream>>) -> Value {
    let line = timeout(Duration::from_secs(3), lines.next_line())
        .await
        .expect("timed out waiting for JSONL output")
        .expect("read JSONL output")
        .expect("JSONL output closed unexpectedly");
    serde_json::from_str(&line).expect("parse JSONL output")
}

fn assert_response_ok(response: &Value, context: &str) {
    assert!(
        response.get("error").is_none(),
        "{context} returned JSON-RPC error: {response:#?}"
    );
    assert!(
        response.get("result").is_some(),
        "{context} omitted result: {response:#?}"
    );
}

fn required_string(value: &Value, pointer: &str, context: &str) -> String {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| panic!("{context} missing at {pointer}: {value:#?}"))
        .to_string()
}
