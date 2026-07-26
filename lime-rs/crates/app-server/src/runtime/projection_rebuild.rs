use agent_protocol::{
    Thread, ThreadActiveFlag, ThreadItem, ThreadItemPayload, ThreadStatus, Turn, TurnStatus,
};
use chrono::{DateTime, SecondsFormat, Utc};
use rusqlite::{params, Connection};
use serde_json::{json, Value};

use super::projection_payload_summary::bounded_payload_summary;

pub(super) fn projected_tables_are_empty(conn: &Connection) -> Result<bool, String> {
    conn.query_row(
        "SELECT
            (SELECT COUNT(*) FROM projected_sessions) +
            (SELECT COUNT(*) FROM projected_turns) +
            (SELECT COUNT(*) FROM projected_items) +
            (SELECT COUNT(*) FROM projection_watermarks)",
        [],
        |row| row.get::<_, i64>(0),
    )
    .map(|count| count == 0)
    .map_err(|error| format!("cannot inspect projected read model before rebuild: {error}"))
}

pub(super) fn rebuild_projected_thread_snapshot(
    conn: &Connection,
    thread: &Thread,
    last_sequence: u64,
) -> Result<(), String> {
    let created_at = timestamp(thread.created_at_ms, "thread created_at_ms")?;
    let updated_at = timestamp(thread.updated_at_ms, "thread updated_at_ms")?;
    let archived_at = thread.archived.then(|| updated_at.clone());
    let metadata_json = serde_json::to_string(&thread.metadata)
        .map_err(|error| format!("cannot encode projected thread metadata: {error}"))?;
    let last_event_sequence = sqlite_u64(last_sequence, "thread history sequence")?;
    let last_event_id = format!("rollout:{}:{last_sequence}", thread.thread_id);

    conn.execute(
        "INSERT INTO projected_sessions (
            session_id, thread_id, status, created_at, updated_at,
            archived_at, title, model, workspace_id, working_dir,
            execution_strategy, metadata_json, last_event_sequence, last_event_id
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        params![
            thread.session_id.as_str(),
            thread.thread_id.as_str(),
            projected_thread_status(thread),
            created_at,
            updated_at,
            archived_at,
            thread
                .name
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .or_else(|| (!thread.preview.trim().is_empty()).then_some(thread.preview.as_str())),
            metadata_string(&thread.metadata, &["modelName", "model", "model_name"]),
            metadata_string(&thread.metadata, &["workspaceId", "workspace_id"]),
            metadata_string(&thread.metadata, &["workingDir", "working_dir", "cwd"]),
            metadata_string(
                &thread.metadata,
                &["executionStrategy", "execution_strategy"],
            ),
            metadata_json,
            last_event_sequence,
            last_event_id,
        ],
    )
    .map_err(|error| format!("cannot rebuild projected session: {error}"))?;

    for (turn_index, turn) in thread.turns.iter().enumerate() {
        rebuild_projected_turn(conn, turn, turn_index)?;
        for item in &turn.items {
            rebuild_projected_item(conn, item)?;
        }
    }

    if last_sequence > 0 {
        conn.execute(
            "INSERT INTO projection_watermarks (
                session_id, last_sequence, last_event_id, updated_at
             ) VALUES (?1, ?2, ?3, ?4)",
            params![
                thread.session_id.as_str(),
                last_event_sequence,
                last_event_id,
                updated_at,
            ],
        )
        .map_err(|error| format!("cannot rebuild projection watermark: {error}"))?;
    }
    Ok(())
}

fn rebuild_projected_turn(conn: &Connection, turn: &Turn, turn_index: usize) -> Result<(), String> {
    let started_at = turn
        .started_at_ms
        .map(|value| timestamp(value, "turn started_at_ms"))
        .transpose()?;
    let completed_at = turn
        .completed_at_ms
        .map(|value| timestamp(value, "turn completed_at_ms"))
        .transpose()?;
    let ordering = i64::try_from(turn_index + 1)
        .map_err(|_| "projected turn ordering exceeds SQLite range".to_string())?;
    conn.execute(
        "INSERT INTO projected_turns (
            turn_id, session_id, thread_id, status, started_at,
            completed_at, last_event_sequence
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            turn.turn_id.as_str(),
            turn.session_id.as_str(),
            turn.thread_id.as_str(),
            projected_turn_status(turn.status),
            started_at,
            completed_at,
            ordering,
        ],
    )
    .map_err(|error| format!("cannot rebuild projected turn: {error}"))?;
    Ok(())
}

fn rebuild_projected_item(conn: &Connection, item: &ThreadItem) -> Result<(), String> {
    let created_at = timestamp(item.created_at_ms, "item created_at_ms")?;
    let sequence = sqlite_u64(item.ordinal, "item ordinal")?;
    let (item_type, payload) = projected_item_payload(item);
    conn.execute(
        "INSERT INTO projected_items (
            event_id, session_id, thread_id, turn_id, sequence,
            item_type, payload_summary_json, created_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            format!("canonical:{}", item.item_id),
            item.session_id.as_str(),
            item.thread_id.as_str(),
            item.turn_id.as_str(),
            sequence,
            item_type,
            bounded_payload_summary(&payload),
            created_at,
        ],
    )
    .map_err(|error| format!("cannot rebuild projected item: {error}"))?;
    Ok(())
}

fn projected_item_payload(item: &ThreadItem) -> (&'static str, Value) {
    let identity = json!({
        "itemId": item.item_id.as_str(),
        "kind": item.kind,
        "status": item.status,
    });
    match &item.payload {
        ThreadItemPayload::UserMessage { content, client_id } => (
            "message.created",
            json!({
                "input": content,
                "text": super::turn_start::user_input_text(content),
                "clientId": client_id,
                "canonical": identity,
            }),
        ),
        ThreadItemPayload::AgentMessage { text, .. } => (
            "message.batch",
            json!({ "text": text, "phase": "final", "canonical": identity }),
        ),
        ThreadItemPayload::Plan {
            text,
            revision_id,
            plan,
            explanation,
            ..
        } => (
            "plan.final",
            json!({
                "text": text,
                "revisionId": revision_id,
                "plan": plan,
                "explanation": explanation,
                "canonical": identity,
            }),
        ),
        ThreadItemPayload::Reasoning { summary, content } => (
            "reasoning.final",
            json!({
                "text": summary.iter().chain(content).cloned().collect::<Vec<_>>().join("\n"),
                "canonical": identity,
            }),
        ),
        ThreadItemPayload::Approval { decision, .. } => (
            if decision.is_some() || item.status.is_terminal() {
                "action.resolved"
            } else {
                "action.required"
            },
            json!({ "canonical": identity }),
        ),
        ThreadItemPayload::Command {
            command,
            cwd,
            output,
            exit_code,
        } => (
            if item.status.is_terminal() {
                "command.exited"
            } else {
                "command.started"
            },
            json!({
                "command": command,
                "cwd": cwd,
                "text": output,
                "exitCode": exit_code,
                "canonical": identity,
            }),
        ),
        ThreadItemPayload::File { changes, status } => (
            match status {
                agent_protocol::FileChangeStatus::Proposed => "patch.started",
                agent_protocol::FileChangeStatus::Applied => "patch.applied",
                agent_protocol::FileChangeStatus::Rejected => "patch.declined",
                agent_protocol::FileChangeStatus::Failed => "patch.failed",
            },
            json!({
                "changes": changes,
                "status": status,
                "canonical": identity,
            }),
        ),
        ThreadItemPayload::SubAgent {
            child_thread_id,
            activity,
            detail,
        } => (
            "subagent.activity",
            json!({
                "childThreadId": child_thread_id.as_str(),
                "activity": activity,
                "text": detail,
                "canonical": identity,
            }),
        ),
        ThreadItemPayload::ContextCompaction {
            summary, window_id, ..
        } => (
            "context.compaction.completed",
            json!({
                "text": summary,
                "windowId": window_id,
                "canonical": identity,
            }),
        ),
        _ => ("item.completed", json!({ "canonical": identity })),
    }
}

fn projected_thread_status(thread: &Thread) -> &'static str {
    match &thread.status {
        ThreadStatus::Active { active_flags }
            if active_flags.iter().any(|flag| {
                matches!(
                    flag,
                    ThreadActiveFlag::WaitingOnApproval | ThreadActiveFlag::WaitingOnUserInput
                )
            }) =>
        {
            "waitingAction"
        }
        ThreadStatus::Active { .. } => "running",
        ThreadStatus::SystemError => "failed",
        ThreadStatus::NotLoaded => "idle",
        ThreadStatus::Idle => thread
            .turns
            .last()
            .map(|turn| match turn.status {
                TurnStatus::InProgress => "running",
                TurnStatus::Completed => "completed",
                TurnStatus::Interrupted => "canceled",
                TurnStatus::Failed => "failed",
            })
            .unwrap_or("idle"),
    }
}

fn projected_turn_status(status: TurnStatus) -> &'static str {
    match status {
        TurnStatus::InProgress => "running",
        TurnStatus::Completed => "completed",
        TurnStatus::Interrupted => "canceled",
        TurnStatus::Failed => "failed",
    }
}

fn metadata_string(metadata: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        metadata
            .get(*key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
    })
}

fn timestamp(value: i64, field: &str) -> Result<String, String> {
    DateTime::<Utc>::from_timestamp_millis(value)
        .map(|value| value.to_rfc3339_opts(SecondsFormat::Millis, true))
        .ok_or_else(|| format!("{field} is outside the supported timestamp range"))
}

fn sqlite_u64(value: u64, field: &str) -> Result<i64, String> {
    i64::try_from(value).map_err(|_| format!("{field} exceeds SQLite range"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_status_preserves_waiting_and_terminal_states() {
        let mut thread = Thread {
            session_id: agent_protocol::SessionId::new("session"),
            thread_id: agent_protocol::ThreadId::new("thread"),
            status: ThreadStatus::Active {
                active_flags: vec![ThreadActiveFlag::WaitingOnUserInput],
            },
            created_at_ms: 0,
            updated_at_ms: 0,
            archived: false,
            recency_at_ms: None,
            parent_thread_id: None,
            agent_path: None,
            agent_nickname: None,
            agent_role: None,
            last_task_message: None,
            agent_state: None,
            forked_from_id: None,
            preview: String::new(),
            model_provider: String::new(),
            product: None,
            name: None,
            metadata: Value::Null,
            turns: Vec::new(),
            turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
        };
        assert_eq!(projected_thread_status(&thread), "waitingAction");
        thread.status = ThreadStatus::SystemError;
        assert_eq!(projected_thread_status(&thread), "failed");
        assert_eq!(projected_turn_status(TurnStatus::Interrupted), "canceled");
    }
}
