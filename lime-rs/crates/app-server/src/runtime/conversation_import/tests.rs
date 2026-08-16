use super::*;
use app_server_protocol::{
    AgentInput, AgentSessionReadParams, AgentSessionTurnStartParams, AgentTurnStatus,
    ConversationImportSourceStatus,
};
use rusqlite::{params, Connection};
use std::fs;
use std::sync::{Arc, Mutex};

use crate::runtime::{
    ActionRespondRequest, CancelExecutionRequest, ExecutionBackend, ExecutionRequest, RuntimeEvent,
    RuntimeEventSink, RuntimeHostContext,
};

mod dry_run;
mod health;
mod idempotency;
mod path_resolution;
mod performance;
mod runtime_events;
mod security;

#[test]
fn scans_codex_state_db_with_filters_and_cursor() {
    let temp = tempfile::tempdir().expect("tempdir");
    let db_path = temp.path().join("state_test.sqlite");
    let conn = Connection::open(&db_path).expect("db");
    conn.execute_batch(
        r#"
CREATE TABLE threads (
    id TEXT PRIMARY KEY,
    rollout_path TEXT,
    created_at INTEGER,
    updated_at INTEGER,
    source TEXT,
    model_provider TEXT,
    cwd TEXT,
    title TEXT,
    sandbox_policy TEXT,
    approval_mode TEXT,
    archived INTEGER,
    archived_at INTEGER
);
            "#,
    )
    .expect("schema");
    insert_thread(
        &conn,
        "thread-a",
        "Alpha work",
        "/workspace/a",
        &write_named_rollout(temp.path(), "thread-a", "Alpha work").to_string_lossy(),
        1,
        2,
        false,
    );
    insert_thread(
        &conn,
        "thread-b",
        "Beta archived",
        "/workspace/a",
        &write_named_rollout(temp.path(), "thread-b", "Beta archived").to_string_lossy(),
        1,
        3,
        true,
    );
    insert_thread(
        &conn,
        "thread-c",
        "Alpha later",
        "/workspace/a",
        &write_named_rollout(temp.path(), "thread-c", "Alpha later").to_string_lossy(),
        1,
        4,
        false,
    );

    let response = codex::scan_source(ConversationImportSourceScanParams {
        source_root: Some(temp.path().to_string_lossy().into_owned()),
        project_path: Some("/workspace/a".to_string()),
        query: Some("alpha".to_string()),
        limit: Some(1),
        ..Default::default()
    })
    .expect("scan");

    assert_eq!(
        response.source.status,
        ConversationImportSourceStatus::Ready
    );
    assert_eq!(response.source.thread_count, 2);
    assert!(response.source.source_home_exists);
    assert!(response.source.state_db_readable);
    assert_eq!(response.source.rollout_file_count, 3);
    assert_eq!(response.threads.len(), 1);
    assert_eq!(response.threads[0].source_thread_id, "thread-c");
    assert_eq!(response.next_cursor.as_deref(), Some("1"));

    let second_page = codex::scan_source(ConversationImportSourceScanParams {
        source_root: Some(temp.path().to_string_lossy().into_owned()),
        project_path: Some("/workspace/a".to_string()),
        query: Some("alpha".to_string()),
        limit: Some(1),
        cursor: response.next_cursor,
        ..Default::default()
    })
    .expect("scan page 2");
    assert_eq!(second_page.threads[0].source_thread_id, "thread-a");
    assert!(second_page.next_cursor.is_none());
}

#[test]
fn scans_current_codex_state_db_metadata_and_millisecond_timestamps() {
    let temp = tempfile::tempdir().expect("tempdir");
    let db_path = temp.path().join("state_5.sqlite");
    let conn = Connection::open(&db_path).expect("db");
    conn.execute_batch(
        r#"
CREATE TABLE threads (
    id TEXT PRIMARY KEY,
    rollout_path TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    source TEXT NOT NULL,
    model_provider TEXT NOT NULL,
    cwd TEXT NOT NULL,
    title TEXT NOT NULL,
    sandbox_policy TEXT NOT NULL,
    approval_mode TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0,
    archived_at INTEGER,
    git_sha TEXT,
    git_branch TEXT,
    git_origin_url TEXT,
    cli_version TEXT NOT NULL DEFAULT '',
    first_user_message TEXT NOT NULL DEFAULT '',
    agent_nickname TEXT,
    agent_role TEXT,
    memory_mode TEXT NOT NULL DEFAULT 'enabled',
    model TEXT,
    reasoning_effort TEXT,
    agent_path TEXT,
    created_at_ms INTEGER,
    updated_at_ms INTEGER,
    thread_source TEXT,
    preview TEXT NOT NULL DEFAULT ''
);
            "#,
    )
    .expect("schema");
    insert_current_thread(
        &conn,
        "thread-current-old",
        "Old thread",
        "/workspace/current",
        &write_named_rollout(temp.path(), "thread-current-old", "old").to_string_lossy(),
        1_781_516_300,
        1_781_516_300_100,
        1_781_516_350,
        1_781_516_350_100,
    );
    insert_current_thread(
        &conn,
        "thread-current-new",
        "New thread",
        "/workspace/current",
        &write_named_rollout(temp.path(), "thread-current-new", "new").to_string_lossy(),
        1_781_516_362,
        1_781_516_362_283,
        1_781_518_329,
        1_781_518_329_154,
    );

    let response = codex::scan_source(ConversationImportSourceScanParams {
        source_root: Some(temp.path().to_string_lossy().into_owned()),
        project_path: Some("/workspace/current".to_string()),
        limit: Some(10),
        ..Default::default()
    })
    .expect("scan");

    assert_eq!(response.source.thread_count, 2);
    assert_eq!(response.threads[0].source_thread_id, "thread-current-new");
    assert_eq!(response.threads[0].source.as_deref(), Some("cli"));
    assert_eq!(
        response.threads[0].created_at.as_deref(),
        Some("2026-06-15T09:39:22.283Z")
    );
    assert_eq!(
        response.threads[0].updated_at.as_deref(),
        Some("2026-06-15T10:12:09.154Z")
    );
    let metadata = response.threads[0].metadata.as_ref().expect("metadata");
    assert_eq!(metadata["model"], "gpt-5.5");
    assert_eq!(metadata["reasoningEffort"], "xhigh");
    assert_eq!(metadata["threadSource"], "user");
    assert_eq!(metadata["cliVersion"], "0.139.0");
    assert_eq!(metadata["gitBranch"], "main");
    assert_eq!(metadata["firstUserMessage"], "真实 Codex 请求");
    assert_eq!(metadata["preview"], "真实 Codex 预览");
    assert_eq!(metadata["approvalPolicy"], "on-request");
    assert_eq!(metadata["sandboxPolicy"], "workspace-write");
}

#[tokio::test]
async fn imported_conversation_continues_with_imported_runtime_context() {
    let temp = tempfile::tempdir().expect("tempdir");
    let db_path = temp.path().join("state_5.sqlite");
    let rollout_path = temp.path().join("rollout-thread-continue.jsonl");
    fs::write(
        &rollout_path,
        codex_session_meta_line("thread-continue", "/workspace/continue", "continue import"),
    )
    .expect("write rollout");
    let conn = Connection::open(&db_path).expect("db");
    conn.execute_batch(
        r#"
CREATE TABLE threads (
    id TEXT PRIMARY KEY,
    rollout_path TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    source TEXT NOT NULL,
    model_provider TEXT NOT NULL,
    cwd TEXT NOT NULL,
    title TEXT NOT NULL,
    sandbox_policy TEXT NOT NULL,
    approval_mode TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0,
    archived_at INTEGER,
    cli_version TEXT,
    memory_mode TEXT,
    model TEXT,
    reasoning_effort TEXT,
    agent_path TEXT,
    thread_source TEXT
);
            "#,
    )
    .expect("schema");
    conn.execute(
        r#"
INSERT INTO threads (
    id, rollout_path, created_at, updated_at, source, model_provider, cwd, title,
    sandbox_policy, approval_mode, archived, archived_at, cli_version, memory_mode,
    model, reasoning_effort, agent_path, thread_source
) VALUES (
    'thread-continue', ?1, 1, 2, 'cli', 'openai', '/workspace/continue',
    'Continue import', 'workspace-write', 'on-request', 0, NULL, '0.139.0',
    'enabled', 'gpt-5.5', 'high', '/tmp/AGENTS.md', 'user'
)
            "#,
        params![rollout_path.to_string_lossy().as_ref()],
    )
    .expect("insert thread");
    let backend = Arc::new(RecordingBackend::default());
    let core = RuntimeCore::with_backend(backend.clone());

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_thread_id: Some("thread-continue".to_string()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    assert!(
        backend
            .requests
            .lock()
            .expect("requests mutex poisoned")
            .is_empty(),
        "importing persisted history must not replay provider or tool execution",
    );

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: response.session.session_id.clone(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read imported session");
    let detail = read.detail.expect("detail");
    let execution_runtime = &detail["execution_runtime"];
    assert_eq!(execution_runtime["provider_name"], "openai");
    assert_eq!(execution_runtime["model_name"], "gpt-5.5");
    assert_eq!(execution_runtime["working_dir"], "/workspace/continue");
    assert_eq!(execution_runtime["reasoning_effort"], "high");
    assert_eq!(execution_runtime["approval_policy"], "on-request");
    assert_eq!(execution_runtime["sandbox_policy"], "workspace-write");
    assert_eq!(execution_runtime["memory_mode"], "enabled");
    assert_eq!(execution_runtime["agent_path"], "/tmp/AGENTS.md");
    assert!(execution_runtime.get("imported_continuation").is_some());
    assert!(execution_runtime.get("imported_thread_settings").is_some());

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: response.session.session_id.clone(),
            turn_id: Some("turn-continue".to_string()),
            input: AgentInput {
                text: "continue".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("continue turn");

    let requests = backend.requests.lock().expect("requests mutex poisoned");
    assert_eq!(requests.len(), 1, "only the live continuation may execute");
    let request = requests.last().expect("recorded request");
    assert_eq!(request.provider_preference(), None);
    assert_eq!(request.model_preference(), None);
    let runtime_options = request.runtime_options.as_ref().expect("runtime options");
    let runtime_request = runtime_options
        .runtime_request
        .as_ref()
        .expect("runtime request");
    assert_eq!(runtime_request.provider_preference.as_deref(), None);
    assert_eq!(runtime_request.model_preference.as_deref(), None);
    assert_eq!(
        runtime_request.working_dir.as_deref(),
        Some("/workspace/continue")
    );
    assert_eq!(runtime_request.reasoning_effort.as_deref(), Some("high"));
    assert_eq!(
        runtime_request.approval_policy.as_deref(),
        Some("on-request")
    );
    assert_eq!(
        runtime_request.sandbox_policy.as_deref(),
        Some("workspace-write")
    );
    assert_eq!(
        runtime_options
            .runtime_metadata()
            .and_then(|value| value.get("memoryMode"))
            .and_then(serde_json::Value::as_str),
        Some("enabled")
    );
}

#[test]
fn previews_codex_rollout_from_source_path() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-thread-a.jsonl");
    let long_text = "a".repeat(codex::MAX_PREVIEW_TEXT_BYTES + 12);
    fs::write(
        &rollout_path,
        [
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:00.000Z",
                "type": "session_meta",
                "payload": {
                    "id": "thread-a",
                    "timestamp": "2026-06-16T00:00:00.000Z",
                    "cwd": "/workspace/lime",
                    "source": "cli",
                    "model_provider": "openai"
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:01.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": long_text},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,abc",
                            "detail": "high"
                        }
                    ]
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:02.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "assistant reply"}]
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:03.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "hello from event"
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:04.000Z",
                "type": "turn_context",
                "payload": {}
            })
            .to_string(),
        ]
        .join("\n"),
    )
    .expect("write rollout");

    let response = codex::preview_thread(ConversationImportThreadPreviewParams {
        source_root: Some(temp.path().to_string_lossy().into_owned()),
        source_path: Some(rollout_path.to_string_lossy().into_owned()),
        limit: Some(10),
        ..Default::default()
    })
    .expect("preview");

    assert_eq!(response.thread.source_thread_id, "thread-a");
    assert_eq!(response.thread.cwd.as_deref(), Some("/workspace/lime"));
    assert_eq!(response.thread.model_provider.as_deref(), Some("openai"));
    assert_eq!(response.summary.line_count, 5);
    assert_eq!(response.summary.message_count, 3);
    assert_eq!(response.summary.rollout_event_items, 1);
    assert_eq!(response.summary.unsupported_count, 1);
    assert_eq!(response.messages.len(), 3);
    assert!(response.messages[0].truncated);
    assert_eq!(response.messages[0].omitted_bytes, 12);
    assert_eq!(response.messages[0].attachments.len(), 1);
    assert_eq!(response.messages[0].attachments[0].kind, "image");
    assert_eq!(
        response.messages[0].attachments[0].uri.as_deref(),
        Some("data:image/png;base64,abc")
    );
    assert_eq!(
        response.messages[0].attachments[0]
            .metadata
            .as_ref()
            .and_then(|metadata| { metadata.get("detail").and_then(serde_json::Value::as_str) }),
        Some("high")
    );
    assert_eq!(response.messages[2].text, "hello from event");
    assert_eq!(response.events[0].kind, "user_message");
}

#[test]
fn previews_codex_event_user_images_as_lime_attachments() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-thread-images.jsonl");
    fs::write(
        &rollout_path,
        serde_json::json!({
            "timestamp": "2026-06-16T00:00:01.000Z",
            "type": "event_msg",
            "payload": {
                "type": "user_message",
                "message": "",
                "images": ["data:image/jpeg;base64,remote"],
                "image_details": ["low"],
                "local_images": ["/tmp/local.png"],
                "local_image_details": ["high"]
            }
        })
        .to_string(),
    )
    .expect("write rollout");

    let response = codex::preview_thread(ConversationImportThreadPreviewParams {
        source_root: Some(temp.path().to_string_lossy().into_owned()),
        source_path: Some(rollout_path.to_string_lossy().into_owned()),
        limit: Some(10),
        ..Default::default()
    })
    .expect("preview");

    assert_eq!(response.summary.message_count, 1);
    assert_eq!(response.messages[0].text, "[Image]");
    assert_eq!(response.messages[0].attachments.len(), 2);
    assert_eq!(
        response.messages[0].attachments[0].uri.as_deref(),
        Some("data:image/jpeg;base64,remote")
    );
    assert_eq!(
        response.messages[0].attachments[0]
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("mediaType"))
            .and_then(serde_json::Value::as_str),
        Some("image/jpeg")
    );
    assert_eq!(
        response.messages[0].attachments[1]
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("localPath"))
            .and_then(serde_json::Value::as_str),
        Some("/tmp/local.png")
    );
}

#[test]
fn previews_duplicate_codex_messages_merge_response_item_attachments() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-thread-merged-images.jsonl");
    fs::write(
        &rollout_path,
        [
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:01.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "## My request for Codex: describe image"
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:02.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe image"},
                        {"type": "input_image", "image_url": "data:image/png;base64,abc"}
                    ]
                }
            })
            .to_string(),
        ]
        .join("\n"),
    )
    .expect("write rollout");

    let response = codex::preview_thread(ConversationImportThreadPreviewParams {
        source_root: Some(temp.path().to_string_lossy().into_owned()),
        source_path: Some(rollout_path.to_string_lossy().into_owned()),
        limit: Some(10),
        ..Default::default()
    })
    .expect("preview");

    assert_eq!(response.summary.message_count, 1);
    assert_eq!(response.messages[0].text, "describe image");
    assert_eq!(
        response.messages[0].source_type.as_deref(),
        Some("event_msg")
    );
    assert_eq!(response.messages[0].attachments.len(), 1);
    assert_eq!(
        response.messages[0].attachments[0].uri.as_deref(),
        Some("data:image/png;base64,abc")
    );
}

#[test]
fn previews_codex_event_messages_as_primary_codex_history() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-thread-events.jsonl");
    fs::write(
        &rollout_path,
        [
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:01.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "## My request for Codex: actual request"
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:02.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "actual reply"
                }
            })
            .to_string(),
        ]
        .join("\n"),
    )
    .expect("write rollout");

    let response = codex::preview_thread(ConversationImportThreadPreviewParams {
        source_root: Some(temp.path().to_string_lossy().into_owned()),
        source_path: Some(rollout_path.to_string_lossy().into_owned()),
        limit: Some(10),
        ..Default::default()
    })
    .expect("preview");

    assert_eq!(response.thread.title.as_deref(), Some("actual request"));
    assert_eq!(response.summary.message_count, 2);
    assert_eq!(response.summary.rollout_event_items, 2);
    assert_eq!(response.messages[0].role, "user");
    assert_eq!(response.messages[0].text, "actual request");
    assert_eq!(
        response.messages[0].source_type.as_deref(),
        Some("event_msg")
    );
    assert_eq!(response.messages[1].role, "assistant");
    assert_eq!(response.messages[1].text, "actual reply");
}

#[test]
fn previews_codex_rollout_hides_contextual_response_items() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-thread-contextual.jsonl");
    fs::write(
        &rollout_path,
        [
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:00.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": "# AGENTS.md instructions for /workspace\n\n<INSTRUCTIONS>\n- internal rule\n</INSTRUCTIONS>"
                    }]
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:01.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "<skills_instructions>\n## Skills\n</skills_instructions>"}]
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:02.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "<environment_context>\n<cwd>/workspace</cwd>\n</environment_context>"}]
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:03.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "## My request for Codex: actual request"
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:04.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "actual reply"
                }
            })
            .to_string(),
        ]
        .join("\n"),
    )
    .expect("write rollout");

    let response = codex::preview_thread(ConversationImportThreadPreviewParams {
        source_root: Some(temp.path().to_string_lossy().into_owned()),
        source_path: Some(rollout_path.to_string_lossy().into_owned()),
        limit: Some(10),
        ..Default::default()
    })
    .expect("preview");

    assert_eq!(response.summary.message_count, 2);
    assert_eq!(response.messages.len(), 2);
    assert_eq!(response.messages[0].role, "user");
    assert_eq!(response.messages[0].text, "actual request");
    assert_eq!(response.messages[1].role, "assistant");
    assert_eq!(response.messages[1].text, "actual reply");
    let rendered = response
        .messages
        .iter()
        .map(|message| message.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!rendered.contains("AGENTS.md instructions"));
    assert!(!rendered.contains("skills_instructions"));
    assert!(!rendered.contains("environment_context"));
}

#[test]
fn committing_codex_rollout_requires_confirmation() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-thread-denied.jsonl");
    fs::write(
        &rollout_path,
        serde_json::json!({
            "timestamp": "2026-06-16T00:00:01.000Z",
            "type": "event_msg",
            "payload": {
                "type": "user_message",
                "message": "## My request for Codex: denied import"
            }
        })
        .to_string(),
    )
    .expect("write rollout");
    let core = RuntimeCore::default();

    let err = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: false,
            ..Default::default()
        },
    )
    .expect_err("unconfirmed import should fail");

    assert!(err.to_string().contains("explicit user confirmation"));
    assert!(core
        .state
        .lock()
        .expect("runtime core state mutex poisoned")
        .sessions
        .is_empty());
}

#[test]
fn committing_codex_rollout_does_not_import_contextual_items_as_messages() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-thread-contextual-import.jsonl");
    fs::write(
        &rollout_path,
        [
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:00.000Z",
                "type": "session_meta",
                "payload": {
                    "id": "thread-contextual-import",
                    "timestamp": "2026-06-16T00:00:00.000Z",
                    "cwd": "/workspace/lime",
                    "source": "cli",
                    "model_provider": "openai"
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:01.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": "# AGENTS.md instructions for /workspace\n\n<INSTRUCTIONS>\n- internal rule\n</INSTRUCTIONS>"
                    }]
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:02.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "<skill>\n<name>demo</name>\n</skill>"}]
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:03.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "## My request for Codex: actual request"
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:04.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "actual reply"
                }
            })
            .to_string(),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = RuntimeCore::default();

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    assert_eq!(response.imported_messages, 2);
    assert_eq!(response.imported_turns, 1);

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: response.session.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read imported session");
    let detail = read.detail.expect("detail");
    let messages = detail
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .expect("messages");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"][0]["text"], "actual request");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"][0]["text"], "actual reply");

    let rendered = serde_json::to_string(messages).expect("messages json");
    assert!(!rendered.contains("AGENTS.md instructions"));
    assert!(!rendered.contains("<skill>"));
}

#[test]
fn committing_codex_rollout_imports_completed_lime_session() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-thread-import.jsonl");
    let full_reply = "assistant reply ".repeat(400);
    fs::write(
        &rollout_path,
        [
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:00.000Z",
                "type": "session_meta",
                "payload": {
                    "id": "thread-import",
                    "timestamp": "2026-06-16T00:00:00.000Z",
                    "cwd": "/workspace/lime",
                    "source": "cli",
                    "model_provider": "openai"
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:01.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "## My request for Codex: import this",
                    "images": ["data:image/png;base64,abc"],
                    "image_details": ["low"]
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:02.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": full_reply
                }
            })
            .to_string(),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:03.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "import this"}]
                }
            })
            .to_string(),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = RuntimeCore::default();

    let response = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            workspace_id: Some("workspace-import".to_string()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    assert_eq!(response.thread.source_thread_id, "thread-import");
    assert_eq!(response.imported_messages, 2);
    assert_eq!(response.imported_turns, 1);
    assert!(response.can_continue);
    assert_eq!(
        response.session.workspace_id.as_deref(),
        Some("workspace-import")
    );

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: response.session.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read imported session");
    assert_eq!(read.turns.len(), 1);
    assert_eq!(read.turns[0].status, AgentTurnStatus::Completed);
    let detail = read.detail.expect("detail");
    let messages = detail
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .expect("messages");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"][0]["text"], "import this");
    assert_eq!(messages[0]["content"][1]["type"], "image");
    assert_eq!(
        messages[0]["content"][1]["uri"],
        "data:image/png;base64,abc"
    );
    assert_eq!(messages[0]["attachments"][0]["kind"], "image");
    assert_eq!(messages[0]["attachments"][0]["detail"], "low");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"][0]["text"], full_reply.trim());
}

#[test]
fn previews_codex_rollout_by_thread_id_from_state_db() {
    let temp = tempfile::tempdir().expect("tempdir");
    let db_path = temp.path().join("state_5.sqlite");
    let rollout_path = temp.path().join("rollout-thread-db.jsonl");
    fs::write(
        &rollout_path,
        serde_json::json!({
            "timestamp": "2026-06-16T00:00:01.000Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "from db"}]
            }
        })
        .to_string(),
    )
    .expect("write rollout");
    let conn = Connection::open(&db_path).expect("db");
    conn.execute_batch(
        r#"
CREATE TABLE threads (
    id TEXT PRIMARY KEY,
    rollout_path TEXT,
    created_at INTEGER,
    updated_at INTEGER,
    source TEXT,
    model_provider TEXT,
    cwd TEXT,
    title TEXT,
    sandbox_policy TEXT,
    approval_mode TEXT,
    archived INTEGER,
    archived_at INTEGER
);
            "#,
    )
    .expect("schema");
    insert_thread(
        &conn,
        "thread-db",
        "Thread from DB",
        "/workspace/db",
        &rollout_path.to_string_lossy(),
        1,
        2,
        false,
    );

    let rollout_path_string = rollout_path.to_string_lossy().into_owned();
    let response = codex::preview_thread(ConversationImportThreadPreviewParams {
        source_root: Some(temp.path().to_string_lossy().into_owned()),
        source_thread_id: Some("thread-db".to_string()),
        ..Default::default()
    })
    .expect("preview");

    assert_eq!(response.thread.source_thread_id, "thread-db");
    assert_eq!(
        response.thread.source_path.as_deref(),
        Some(rollout_path_string.as_str())
    );
    assert_eq!(response.messages[0].text, "from db");
}

fn insert_thread(
    conn: &Connection,
    id: &str,
    title: &str,
    cwd: &str,
    rollout_path: &str,
    created_at: i64,
    updated_at: i64,
    archived: bool,
) {
    conn.execute(
        r#"
INSERT INTO threads (
    id, rollout_path, created_at, updated_at, source, model_provider, cwd, title,
    sandbox_policy, approval_mode, archived, archived_at
) VALUES (?1, ?2, ?3, ?4, 'cli', 'openai', ?5, ?6, 'workspace-write', 'on-request', ?7, NULL)
            "#,
        params![
            id,
            rollout_path,
            created_at,
            updated_at,
            cwd,
            title,
            if archived { 1_i64 } else { 0_i64 }
        ],
    )
    .expect("insert thread");
}

fn create_legacy_threads_table(conn: &Connection) {
    conn.execute_batch(
        r#"
CREATE TABLE threads (
    id TEXT PRIMARY KEY,
    rollout_path TEXT,
    created_at INTEGER,
    updated_at INTEGER,
    source TEXT,
    model_provider TEXT,
    cwd TEXT,
    title TEXT,
    sandbox_policy TEXT,
    approval_mode TEXT,
    archived INTEGER,
    archived_at INTEGER
);
            "#,
    )
    .expect("schema");
}

fn codex_rollout_path(
    root: &std::path::Path,
    subdir: &str,
    date_path: &str,
    thread_id: &str,
) -> std::path::PathBuf {
    let path = root
        .join(subdir)
        .join(date_path)
        .join(format!("rollout-2026-06-15T18-12-45-{thread_id}.jsonl"));
    fs::create_dir_all(path.parent().expect("rollout parent")).expect("create rollout parent");
    path
}

fn codex_session_meta_line(thread_id: &str, cwd: &str, user_message: &str) -> String {
    [
        serde_json::json!({
            "timestamp": "2026-06-16T00:00:00.000Z",
            "type": "session_meta",
            "payload": {
                "id": thread_id,
                "timestamp": "2026-06-16T00:00:00.000Z",
                "cwd": cwd,
                "source": "cli",
                "model_provider": "openai"
            }
        })
        .to_string(),
        serde_json::json!({
            "timestamp": "2026-06-16T00:00:01.000Z",
            "type": "event_msg",
            "payload": {
                "type": "user_message",
                "message": format!("## My request for Codex: {user_message}")
            }
        })
        .to_string(),
    ]
    .join("\n")
}

fn write_named_rollout(
    root: &std::path::Path,
    thread_id: &str,
    user_message: &str,
) -> std::path::PathBuf {
    let path = codex_rollout_path(root, "sessions", "2026/06/15", thread_id);
    fs::write(
        &path,
        codex_session_meta_line(thread_id, "/workspace/fixture", user_message),
    )
    .expect("write named rollout");
    path
}

#[allow(clippy::too_many_arguments)]
fn insert_current_thread(
    conn: &Connection,
    id: &str,
    title: &str,
    cwd: &str,
    rollout_path: &str,
    created_at: i64,
    created_at_ms: i64,
    updated_at: i64,
    updated_at_ms: i64,
) {
    conn.execute(
        r#"
INSERT INTO threads (
    id, rollout_path, created_at, updated_at, source, model_provider, cwd, title,
    sandbox_policy, approval_mode, archived, archived_at, git_sha, git_branch, git_origin_url,
    cli_version, first_user_message, agent_nickname, agent_role, memory_mode, model,
    reasoning_effort, agent_path, created_at_ms, updated_at_ms, thread_source, preview
) VALUES (
    ?1, ?2, ?3, ?4, 'cli', 'custom', ?5, ?6, 'workspace-write', 'on-request',
    0, NULL, 'abc123', 'main', 'https://example.invalid/repo.git', '0.139.0',
    '真实 Codex 请求', 'Coder', 'implementer', 'enabled', 'gpt-5.5',
    'xhigh', '/tmp/agent.md', ?7, ?8, 'user', '真实 Codex 预览'
)
            "#,
        params![
            id,
            rollout_path,
            created_at,
            updated_at,
            cwd,
            title,
            created_at_ms,
            updated_at_ms
        ],
    )
    .expect("insert current thread");
}

#[derive(Default)]
struct RecordingBackend {
    requests: Mutex<Vec<ExecutionRequest>>,
}

#[async_trait::async_trait]
impl ExecutionBackend for RecordingBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.requests
            .lock()
            .expect("requests mutex poisoned")
            .push(request);
        sink.emit(RuntimeEvent::new("turn.completed", serde_json::json!({})))
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
