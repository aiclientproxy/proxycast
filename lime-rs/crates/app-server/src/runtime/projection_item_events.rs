use app_server_protocol::AgentEvent;
use rusqlite::Connection;
use serde_json::Value;
use std::collections::BTreeSet;

const PROJECTED_THREAD_ITEM_EVENT_TYPES: &[&str] = &[
    "message.delta",
    "message.delta_batch",
    "message.batch",
    "reasoning.delta",
    "reasoning.summary",
    "reasoning.completed",
    "reasoning.final",
    "item.started",
    "item.updated",
    "item.completed",
    "plan.delta",
    "plan.final",
    "tool.output.delta",
    "permission.denied",
    "sandbox.blocked",
    "command.started",
    "command.output",
    "command.exited",
    "test.started",
    "test.completed",
    "patch.started",
    "patch.applied",
    "patch.failed",
    "patch.declined",
    "action.required",
    "action.resolved",
    "action.cancelled",
    "action.canceled",
    "action.expired",
    "context.compaction.started",
    "context.compaction.completed",
    "subagent.activity",
    "artifact.snapshot",
    "file.changed",
    "routing.decision.made",
    "routing.fallback.applied",
    "routing.not_possible",
    "runtime.warning",
    "turn.failed",
    "runtime.error",
];

const PROJECTED_WORKSPACE_PATCH_EVENT_TYPES: &[&str] = &["artifact.snapshot"];

pub(super) fn query_projected_session_item_events(
    conn: &Connection,
    session_id: &str,
    messages: &[Value],
) -> Result<Vec<AgentEvent>, String> {
    Ok(merge_projected_item_events(
        query_projected_window_item_events(conn, session_id, messages)?,
        query_projected_workspace_patch_events(conn, session_id)?,
    ))
}

fn query_projected_window_item_events(
    conn: &Connection,
    session_id: &str,
    messages: &[Value],
) -> Result<Vec<AgentEvent>, String> {
    let turn_ids = projected_window_turn_ids(messages);
    if turn_ids.is_empty() {
        return Ok(Vec::new());
    }

    let turn_placeholders = placeholders(2, turn_ids.len());
    let type_placeholders =
        placeholders(2 + turn_ids.len(), PROJECTED_THREAD_ITEM_EVENT_TYPES.len());
    let sql = format!(
        "SELECT event_id, session_id, thread_id, turn_id, sequence,
                item_type, payload_summary_json, created_at
         FROM projected_items
         WHERE session_id = ?1
           AND turn_id IN ({turn_placeholders})
           AND item_type IN ({type_placeholders})
         ORDER BY sequence ASC, event_id ASC"
    );
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|error| format!("无法准备 projected_items 过程项查询: {error}"))?;
    let mut query_params = Vec::<rusqlite::types::Value>::with_capacity(
        1 + turn_ids.len() + PROJECTED_THREAD_ITEM_EVENT_TYPES.len(),
    );
    query_params.push(rusqlite::types::Value::from(session_id.to_string()));
    query_params.extend(
        turn_ids
            .iter()
            .map(|turn_id| rusqlite::types::Value::from(turn_id.clone())),
    );
    query_params.extend(
        PROJECTED_THREAD_ITEM_EVENT_TYPES
            .iter()
            .map(|item_type| rusqlite::types::Value::from((*item_type).to_string())),
    );
    let rows = stmt
        .query_map(
            rusqlite::params_from_iter(query_params),
            projected_event_row,
        )
        .map_err(|error| format!("无法查询 projected_items 过程项: {error}"))?;
    rows.map(|row| row.map(projected_event_from_row))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("无法读取 projected_items 过程项: {error}"))
        .map(|events| events.into_iter().flatten().collect())
}

fn query_projected_workspace_patch_events(
    conn: &Connection,
    session_id: &str,
) -> Result<Vec<AgentEvent>, String> {
    let type_placeholders = placeholders(2, PROJECTED_WORKSPACE_PATCH_EVENT_TYPES.len());
    let sql = format!(
        "SELECT event_id, session_id, thread_id, turn_id, sequence,
                item_type, payload_summary_json, created_at
         FROM projected_items
         WHERE session_id = ?1
           AND item_type IN ({type_placeholders})
         ORDER BY sequence ASC, event_id ASC"
    );
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|error| format!("无法准备 projected_items workspace patch 事件查询: {error}"))?;
    let mut query_params = Vec::<rusqlite::types::Value>::with_capacity(
        1 + PROJECTED_WORKSPACE_PATCH_EVENT_TYPES.len(),
    );
    query_params.push(rusqlite::types::Value::from(session_id.to_string()));
    query_params.extend(
        PROJECTED_WORKSPACE_PATCH_EVENT_TYPES
            .iter()
            .map(|item_type| rusqlite::types::Value::from((*item_type).to_string())),
    );
    let rows = stmt
        .query_map(
            rusqlite::params_from_iter(query_params),
            projected_event_row,
        )
        .map_err(|error| format!("无法查询 projected_items workspace patch 事件: {error}"))?;
    rows.map(|row| row.map(projected_event_from_row))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("无法读取 projected_items workspace patch 事件: {error}"))
        .map(|events| {
            events
                .into_iter()
                .flatten()
                .filter(is_workspace_patch_projection_event)
                .collect()
        })
}

fn merge_projected_item_events(
    mut window_events: Vec<AgentEvent>,
    workspace_patch_events: Vec<AgentEvent>,
) -> Vec<AgentEvent> {
    let mut event_ids = window_events
        .iter()
        .map(|event| event.event_id.clone())
        .collect::<BTreeSet<_>>();
    for event in workspace_patch_events {
        if event_ids.insert(event.event_id.clone()) {
            window_events.push(event);
        }
    }
    window_events.sort_by(|left, right| {
        left.sequence
            .cmp(&right.sequence)
            .then_with(|| left.event_id.cmp(&right.event_id))
    });
    window_events
}

fn is_workspace_patch_projection_event(event: &AgentEvent) -> bool {
    event.event_type == "artifact.snapshot" && payload_has_workspace_patch(&event.payload)
}

fn payload_has_workspace_patch(payload: &Value) -> bool {
    let artifact = payload.get("artifact");
    let metadata = payload.get("metadata");
    let artifact_metadata = artifact.and_then(|artifact| artifact.get("metadata"));
    [
        payload.get("articleWorkspace"),
        payload.get("article_workspace"),
        payload.get("workspacePatch"),
        payload.get("workspace_patch"),
        metadata.and_then(|value| value.get("articleWorkspace")),
        metadata.and_then(|value| value.get("article_workspace")),
        metadata.and_then(|value| value.get("workspacePatch")),
        metadata.and_then(|value| value.get("workspace_patch")),
        artifact.and_then(|value| value.get("articleWorkspace")),
        artifact.and_then(|value| value.get("article_workspace")),
        artifact.and_then(|value| value.get("workspacePatch")),
        artifact.and_then(|value| value.get("workspace_patch")),
        artifact_metadata.and_then(|value| value.get("articleWorkspace")),
        artifact_metadata.and_then(|value| value.get("article_workspace")),
        artifact_metadata.and_then(|value| value.get("workspacePatch")),
        artifact_metadata.and_then(|value| value.get("workspace_patch")),
    ]
    .into_iter()
    .flatten()
    .any(|candidate| candidate.get("objects").and_then(Value::as_array).is_some())
}

fn projected_window_turn_ids(messages: &[Value]) -> Vec<String> {
    let mut turn_ids = BTreeSet::new();
    for message in messages {
        for key in ["runtimeTurnId", "runtime_turn_id"] {
            if let Some(turn_id) = message
                .get(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                turn_ids.insert(turn_id.to_string());
            }
        }
    }
    turn_ids.into_iter().collect()
}

fn placeholders(start_index: usize, count: usize) -> String {
    (start_index..start_index + count)
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, Clone)]
struct ProjectedEventRow {
    event_id: String,
    session_id: String,
    thread_id: Option<String>,
    turn_id: Option<String>,
    sequence: i64,
    item_type: String,
    payload: Value,
    created_at: String,
}

fn projected_event_row(row: &rusqlite::Row<'_>) -> Result<ProjectedEventRow, rusqlite::Error> {
    let payload_summary_json: String = row.get(6)?;
    Ok(ProjectedEventRow {
        event_id: row.get(0)?,
        session_id: row.get(1)?,
        thread_id: row.get(2)?,
        turn_id: row.get(3)?,
        sequence: row.get::<_, i64>(4)?.max(0),
        item_type: row.get(5)?,
        payload: serde_json::from_str::<Value>(&payload_summary_json).unwrap_or(Value::Null),
        created_at: row.get(7)?,
    })
}

fn projected_event_from_row(row: ProjectedEventRow) -> Option<AgentEvent> {
    if !should_keep_projected_item_event(&row) {
        return None;
    }
    Some(AgentEvent {
        event_id: row.event_id,
        sequence: row.sequence as u64,
        session_id: row.session_id,
        thread_id: row.thread_id,
        turn_id: row.turn_id,
        event_type: row.item_type,
        timestamp: row.created_at,
        payload: row.payload,
    })
}

fn should_keep_projected_item_event(row: &ProjectedEventRow) -> bool {
    if !is_assistant_message_event_type(&row.item_type) {
        return true;
    }
    has_message_delta_phase(&row.payload)
        && !should_project_message_delta_as_final_text(&row.payload)
}

fn is_assistant_message_event_type(event_type: &str) -> bool {
    matches!(
        event_type,
        "message.delta" | "message.delta_batch" | "message.batch"
    )
}

fn has_message_delta_phase(payload: &Value) -> bool {
    payload_string(Some(payload), &["phase", "messagePhase", "message_phase"]).is_some()
}

fn should_project_message_delta_as_final_text(payload: &Value) -> bool {
    let Some(phase) = payload_string(Some(payload), &["phase", "messagePhase", "message_phase"])
    else {
        return true;
    };
    let normalized = phase.trim().to_ascii_lowercase();
    matches!(normalized.as_str(), "final" | "final_answer")
}

fn payload_string(value: Option<&Value>, keys: &[&str]) -> Option<String> {
    let value = value?;
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str).map(str::to_string))
}
