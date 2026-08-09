mod agent_message;
pub(super) mod change_set;
mod coding_items;
mod control_items;
mod helpers;
pub(super) mod materializer;
pub(in crate::runtime) mod media_result;
mod plan;

pub(in crate::runtime) use change_set::merge_turn_snapshot;
pub(in crate::runtime) use materializer::{materialize_events, IncrementalMaterializer};

#[cfg(test)]
mod typed_tests;

use super::raw_string_field;
use super::string_field;
use super::StoredSession;
use app_server_protocol::AgentEvent;
use coding_items::{upsert_command_item, upsert_patch_item};
use control_items::{
    expert_profile_switch_item, upsert_approval_item, upsert_context_compaction_item,
    upsert_subagent_activity_item,
};
use helpers::{base_item, compact_json, event_metadata, sort_thread_items};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub(super) fn thread_items_from_events(stored: &StoredSession) -> Vec<Value> {
    let mut items = Vec::new();
    let mut agent_message_item_by_id = HashMap::<String, usize>::new();
    let mut last_text_item_by_turn = std::collections::HashMap::<String, usize>::new();
    let mut command_items = HashMap::<String, Value>::new();
    let mut patch_items = HashMap::<String, Value>::new();
    let mut reasoning_items = HashMap::<String, Value>::new();
    let mut approval_items = HashMap::<String, Value>::new();
    let mut context_compaction_items = HashMap::<String, Value>::new();
    let mut subagent_items = HashMap::<String, Value>::new();
    let mut media_result_items = HashMap::<String, Value>::new();

    for event in &stored.events {
        match event.event_type.as_str() {
            "message.delta" | "message.delta_batch" | "message.batch" => {
                if let Some(item) = agent_message::item_from_delta(stored, event) {
                    if let Some(stable_item_id) = agent_message::payload_id(event) {
                        if let Some(existing_index) =
                            agent_message_item_by_id.get(&stable_item_id).copied()
                        {
                            agent_message::merge_item(&mut items[existing_index], &item);
                            continue;
                        }
                        agent_message_item_by_id.insert(stable_item_id, items.len());
                        items.push(item);
                        continue;
                    }
                    if agent_message::is_imported_event(event) {
                        items.push(item);
                        continue;
                    }
                    if let Some(turn_id) = event.turn_id.as_deref() {
                        if let Some(existing_index) = last_text_item_by_turn.get(turn_id).copied() {
                            agent_message::merge_item(&mut items[existing_index], &item);
                            continue;
                        }
                        last_text_item_by_turn.insert(turn_id.to_string(), items.len());
                    }
                    items.push(item);
                }
            }
            "reasoning.delta" | "reasoning.summary" | "reasoning.completed" | "reasoning.final" => {
                if let Some(item) = reasoning_item(stored, event) {
                    items.push(item);
                }
            }
            "item.started" | "item.updated" | "item.completed" => {
                if let Some(item) = review_item_from_item_event(stored, event) {
                    items.push(item);
                    continue;
                }
                if agent_message::upsert_from_item_event(
                    stored,
                    event,
                    &mut items,
                    &mut agent_message_item_by_id,
                ) {
                    continue;
                }
                upsert_reasoning_item(stored, event, &mut reasoning_items);
                media_result::upsert_from_event(stored, event, &mut media_result_items);
            }
            "plan.delta" | "plan.final" => {
                if let Some(item) = plan::plan_item(stored, event) {
                    items.push(item);
                }
            }
            "command.started" | "command.output" | "command.exited" => {
                upsert_command_item(stored, event, &mut command_items);
            }
            "patch.started" | "patch.applied" | "patch.failed" | "patch.declined" => {
                upsert_patch_item(stored, event, &mut patch_items);
            }
            "action.required" | "action.resolved" | "action.cancelled" | "action.canceled"
            | "action.expired" => {
                upsert_approval_item(stored, event, &mut approval_items);
            }
            "context.compaction.started" | "context.compaction.completed" => {
                upsert_context_compaction_item(stored, event, &mut context_compaction_items);
            }
            "expert.profile_switch.completed" => {
                if let Some(item) = expert_profile_switch_item(stored, event) {
                    items.push(item);
                }
            }
            "subagent.activity" => {
                upsert_subagent_activity_item(stored, event, &mut subagent_items);
            }
            _ => {}
        }
    }

    items.extend(command_items.into_values());
    items.extend(patch_items.into_values());
    items.extend(reasoning_items.into_values());
    items.extend(approval_items.into_values());
    items.extend(context_compaction_items.into_values());
    items.extend(subagent_items.into_values());
    items.extend(media_result_items.into_values());
    sort_thread_items(&mut items);
    items
}

fn review_item_from_item_event(stored: &StoredSession, event: &AgentEvent) -> Option<Value> {
    let item = event.payload.get("item")?.as_object()?;
    let payload = item.get("payload")?.as_object()?;
    let name = payload.get("name")?.as_str()?;
    let review = payload
        .get("data")
        .and_then(|data| data.get("review"))
        .and_then(Value::as_str)?
        .trim();
    if review.is_empty() || !matches!(name, "enteredReviewMode" | "exitedReviewMode") {
        return None;
    }
    let status = if event.event_type == "item.started" {
        "in_progress"
    } else {
        "completed"
    };
    Some(base_item(
        stored,
        event,
        "review_boundary",
        status,
        json!({
            "boundary": if name == "enteredReviewMode" { "entered" } else { "exited" },
            "review": review,
            "metadata": event.payload.get("metadata").cloned().unwrap_or(Value::Null),
        }),
    ))
}

fn upsert_reasoning_item(
    stored: &StoredSession,
    event: &AgentEvent,
    items: &mut HashMap<String, Value>,
) {
    let Some(next) = reasoning_item_from_item_event(stored, event) else {
        return;
    };
    let Some(item_id) = string_field(&next, &["id"]) else {
        return;
    };
    if let Some(existing) = items.get_mut(&item_id) {
        merge_reasoning_item(existing, &next);
        return;
    }
    if next.get("text").and_then(Value::as_str).is_none() {
        return;
    }
    items.insert(item_id, next);
}

fn merge_reasoning_item(existing: &mut Value, next: &Value) {
    let Some(existing_object) = existing.as_object_mut() else {
        return;
    };
    let existing_is_completed = existing_object
        .get("status")
        .and_then(Value::as_str)
        .is_some_and(|status| status == "completed");
    let next_status = string_field(next, &["status"]).unwrap_or_else(|| "in_progress".to_string());
    if !existing_is_completed || next_status == "completed" {
        existing_object.insert("status".to_string(), Value::String(next_status));
    }
    for key in ["text", "summary", "metadata"] {
        if let Some(value) = next.get(key).cloned() {
            existing_object.insert(key.to_string(), value);
        }
    }
    if let Some(started_at) = next.get("started_at").cloned() {
        existing_object
            .entry("started_at".to_string())
            .or_insert(started_at);
    }
    if let Some(updated_at) = next.get("updated_at").cloned() {
        existing_object.insert("updated_at".to_string(), updated_at);
    }
    if let Some(completed_at) = next.get("completed_at").cloned() {
        existing_object.insert("completed_at".to_string(), completed_at);
    }
}

fn reasoning_item(stored: &StoredSession, event: &AgentEvent) -> Option<Value> {
    let text = raw_string_field(
        &event.payload,
        &[
            "text",
            "delta",
            "summary",
            "content",
            "message",
            "outputText",
            "output_text",
        ],
    )?;
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let status = if matches!(
        event.event_type.as_str(),
        "reasoning.completed" | "reasoning.final"
    ) {
        "completed"
    } else {
        "in_progress"
    };
    Some(base_item(
        stored,
        event,
        "reasoning",
        status,
        json!({
            "text": text,
            "summary": summary_list(&event.payload),
            "metadata": reasoning_metadata(event, &event.payload),
        }),
    ))
}

fn reasoning_item_from_item_event(stored: &StoredSession, event: &AgentEvent) -> Option<Value> {
    let item = event.payload.get("item").unwrap_or(&event.payload);
    let payload = item.get("payload").unwrap_or(item);
    let item_type = string_field(payload, &["type", "kind"])
        .or_else(|| string_field(item, &["type", "kind"]))?;
    if item_type.trim().to_ascii_lowercase() != "reasoning" {
        return None;
    }
    let text = raw_string_field(
        payload,
        &[
            "text",
            "delta",
            "summary",
            "content",
            "message",
            "outputText",
            "output_text",
        ],
    )
    .map(|value| value.trim().to_string())
    .filter(|value| !value.is_empty());
    let status = string_field(item, &["status"])
        .or_else(|| string_field(payload, &["status"]))
        .map(|status| normalize_reasoning_item_status(&status))
        .unwrap_or_else(|| {
            if event.event_type == "item.completed" {
                "completed".to_string()
            } else {
                "in_progress".to_string()
            }
        });
    if text.is_none() && event.event_type != "item.completed" {
        return None;
    }
    let mut value = base_item(
        stored,
        event,
        "reasoning",
        &status,
        compact_json(json!({
            "text": text,
            "summary": summary_list(payload),
            "metadata": reasoning_metadata(event, payload),
        })),
    );
    if let Some(object) = value.as_object_mut() {
        if let Some(id) = string_field(item, &["id", "itemId", "item_id"])
            .or_else(|| string_field(payload, &["id", "itemId", "item_id"]))
        {
            object.insert("id".to_string(), Value::String(id));
        }
        if let Some(thread_id) =
            string_field(item, &["thread_id", "threadId"]).or_else(|| event.thread_id.clone())
        {
            object.insert("thread_id".to_string(), Value::String(thread_id));
        }
        if let Some(turn_id) =
            string_field(item, &["turn_id", "turnId"]).or_else(|| event.turn_id.clone())
        {
            object.insert("turn_id".to_string(), Value::String(turn_id));
        }
        if let Some(sequence) = item.get("sequence").and_then(Value::as_u64) {
            object.insert("sequence".to_string(), json!(sequence));
        }
        if let Some(started_at) = string_field(item, &["started_at", "startedAt"]) {
            object.insert("started_at".to_string(), Value::String(started_at));
        }
        if let Some(updated_at) = string_field(item, &["updated_at", "updatedAt"]) {
            object.insert("updated_at".to_string(), Value::String(updated_at));
        }
        if let Some(completed_at) = string_field(item, &["completed_at", "completedAt"]) {
            object.insert("completed_at".to_string(), Value::String(completed_at));
        }
    }
    Some(value)
}

fn normalize_reasoning_item_status(status: &str) -> String {
    match status.trim() {
        "running" | "pending" | "started" | "inProgress" | "in_progress" => {
            "in_progress".to_string()
        }
        "completed" | "succeeded" | "success" => "completed".to_string(),
        "failed" | "error" => "failed".to_string(),
        _ => "in_progress".to_string(),
    }
}

fn reasoning_metadata(event: &AgentEvent, payload: &Value) -> Value {
    let mut metadata = event_metadata(event);
    let Some(metadata_object) = metadata.as_object_mut() else {
        return metadata;
    };

    merge_reasoning_metadata_object(metadata_object, payload.get("metadata"));
    merge_provider_metadata_aliases(metadata_object, payload);
    merge_provider_metadata_aliases(metadata_object, &event.payload);

    compact_json(metadata)
}

fn merge_reasoning_metadata_object(target: &mut Map<String, Value>, value: Option<&Value>) {
    let Some(Value::Object(source)) = value else {
        return;
    };
    for (key, value) in source {
        if value.is_null() {
            continue;
        }
        target.insert(key.clone(), value.clone());
    }
    if let Some(value) = value {
        merge_provider_metadata_aliases(target, value);
    }
}

fn merge_provider_metadata_aliases(target: &mut Map<String, Value>, source: &Value) {
    let Some(source) = source.as_object() else {
        return;
    };
    if let Some(value) = source
        .get("provider_metadata")
        .or_else(|| source.get("providerMetadata"))
        .cloned()
        .filter(|value| !value.is_null())
    {
        target.insert("provider_metadata".to_string(), value);
    }
}

fn summary_list(payload: &Value) -> Vec<String> {
    let mut values = Vec::new();
    if let Some(summary) = raw_string_field(payload, &["summary"]) {
        values.push(summary);
    }
    for value in payload
        .get("summary")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        values.push(value.to_string());
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::{AgentSession, AgentSessionStatus};
    use std::collections::HashMap;

    fn stored_session(events: Vec<AgentEvent>) -> StoredSession {
        StoredSession {
            session: AgentSession {
                session_id: "session-1".to_string(),
                thread_id: "thread-1".to_string(),
                app_id: "agent-runtime".to_string(),
                workspace_id: None,
                business_object_ref: None,
                status: AgentSessionStatus::Running,
                created_at: "2026-06-23T00:00:00.000Z".to_string(),
                updated_at: "2026-06-23T00:00:00.000Z".to_string(),
            },
            turns: Vec::new(),
            turn_inputs: HashMap::new(),
            turn_runtime_options: HashMap::new(),
            events,
            output_blobs: HashMap::new(),
        }
    }

    fn agent_event(event_id: &str, sequence: u64, event_type: &str, payload: Value) -> AgentEvent {
        AgentEvent {
            event_id: event_id.to_string(),
            sequence,
            session_id: "session-1".to_string(),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            event_type: event_type.to_string(),
            timestamp: "2026-06-23T00:00:01.000Z".to_string(),
            payload,
        }
    }

    #[test]
    fn reasoning_item_payload_metadata_is_preserved_in_thread_items() {
        let stored = stored_session(vec![agent_event(
            "evt-reasoning-item",
            7,
            "item.started",
            json!({
                "item": {
                    "id": "reasoning-1",
                    "thread_id": "thread-1",
                    "turn_id": "turn-1",
                    "sequence": 7,
                    "status": "in_progress",
                    "type": "reasoning",
                    "text": "先判断任务类型",
                    "summary": ["先判断任务类型"],
                    "metadata": {
                        "provider_metadata": {
                            "signature": "sig-anthropic"
                        },
                        "native_reasoning_item_id": "rs_123"
                    }
                }
            }),
        )]);

        let items = thread_items_from_events(&stored);

        assert_eq!(items.len(), 1);
        let metadata = items[0]
            .get("metadata")
            .and_then(Value::as_object)
            .expect("reasoning metadata");
        assert_eq!(
            metadata.get("source_event_id"),
            Some(&json!("evt-reasoning-item"))
        );
        assert_eq!(
            metadata
                .get("provider_metadata")
                .and_then(|value| value.get("signature")),
            Some(&json!("sig-anthropic"))
        );
        assert_eq!(
            metadata.get("native_reasoning_item_id"),
            Some(&json!("rs_123"))
        );
    }

    #[test]
    fn reasoning_final_provider_metadata_is_projected_to_thread_item_metadata() {
        let stored = stored_session(vec![agent_event(
            "evt-reasoning-final",
            8,
            "reasoning.final",
            json!({
                "reasoningId": "runtime-thinking",
                "text": "完整思考摘要",
                "providerMetadata": {
                    "backend": "codex",
                    "summary_index": 1
                }
            }),
        )]);

        let items = thread_items_from_events(&stored);

        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[0]["status"], "completed");
        assert_eq!(items[0]["text"], "完整思考摘要");
        assert_eq!(
            items[0]
                .get("metadata")
                .and_then(|metadata| metadata.get("provider_metadata"))
                .and_then(|provider_metadata| provider_metadata.get("backend")),
            Some(&json!("codex"))
        );
        assert_eq!(
            items[0]
                .get("metadata")
                .and_then(|metadata| metadata.get("provider_metadata"))
                .and_then(|provider_metadata| provider_metadata.get("summary_index")),
            Some(&json!(1))
        );
    }

    #[test]
    fn agent_message_delta_preserves_item_id_and_phase() {
        let stored = stored_session(vec![
            agent_event(
                "evt-commentary-1",
                1,
                "message.delta",
                json!({
                    "itemId": "agent-message-commentary",
                    "text": "我先搜索",
                    "phase": "commentary",
                    "imported": true
                }),
            ),
            agent_event(
                "evt-commentary-2",
                2,
                "message.delta",
                json!({
                    "itemId": "agent-message-commentary",
                    "text": "并筛选来源。",
                    "phase": "commentary",
                    "imported": true
                }),
            ),
        ]);

        let items = thread_items_from_events(&stored);

        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["id"], "agent-message-commentary");
        assert_eq!(items[0]["type"], "agent_message");
        assert_eq!(items[0]["phase"], "commentary");
        assert_eq!(items[0]["text"], "我先搜索并筛选来源。");
    }

    #[test]
    fn agent_message_delta_waits_for_item_completed_terminal() {
        let stored = stored_session(vec![
            agent_event(
                "evt-agent-delta-1",
                1,
                "message.delta",
                json!({
                    "itemId": "agent-final-1",
                    "text": "Hel",
                    "phase": "final_answer"
                }),
            ),
            agent_event(
                "evt-agent-delta-2",
                2,
                "message.delta",
                json!({
                    "itemId": "agent-final-1",
                    "text": "lo",
                    "phase": "final_answer"
                }),
            ),
            agent_event(
                "evt-agent-terminal",
                3,
                "item.completed",
                json!({
                    "item": {
                        "id": "agent-final-1",
                        "type": "agent_message",
                        "status": "completed"
                    }
                }),
            ),
        ]);

        let items = thread_items_from_events(&stored);

        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["id"], "agent-final-1");
        assert_eq!(items[0]["type"], "agent_message");
        assert_eq!(items[0]["status"], "completed");
        assert_eq!(items[0]["text"], "Hello");
        assert!(items[0].get("completed_at").is_some());
    }

    #[test]
    fn turn_failed_does_not_complete_agent_message_item_without_item_terminal() {
        let stored = stored_session(vec![
            agent_event(
                "evt-agent-delta",
                1,
                "message.delta",
                json!({
                    "itemId": "agent-final-1",
                    "text": "partial",
                    "phase": "final_answer"
                }),
            ),
            agent_event(
                "evt-turn-failed",
                2,
                "turn.failed",
                json!({
                    "message": "provider stream timed out"
                }),
            ),
        ]);

        let items = thread_items_from_events(&stored);

        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["id"], "agent-final-1");
        assert_eq!(items[0]["type"], "agent_message");
        assert_eq!(items[0]["status"], "in_progress");
        assert_eq!(items[0]["text"], "partial");
        assert!(items[0].get("completed_at").is_none());
    }

    #[test]
    fn item_completed_agent_message_replaces_delta_text_when_terminal_has_full_text() {
        let stored = stored_session(vec![
            agent_event(
                "evt-agent-delta",
                1,
                "message.delta",
                json!({
                    "itemId": "agent-final-1",
                    "text": "draft",
                    "phase": "final_answer"
                }),
            ),
            agent_event(
                "evt-agent-terminal",
                2,
                "item.completed",
                json!({
                    "item": {
                        "id": "agent-final-1",
                        "type": "agent_message",
                        "text": "final answer",
                        "status": "completed"
                    }
                }),
            ),
        ]);

        let items = thread_items_from_events(&stored);

        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["id"], "agent-final-1");
        assert_eq!(items[0]["type"], "agent_message");
        assert_eq!(items[0]["status"], "completed");
        assert_eq!(items[0]["text"], "final answer");
    }

    #[test]
    fn item_updated_agent_message_cumulative_text_replaces_delta_prefix() {
        let stored = stored_session(vec![
            agent_event(
                "evt-agent-delta-1",
                1,
                "message.delta",
                json!({
                    "itemId": "agent-final-1",
                    "text": "写作",
                    "phase": "final_answer"
                }),
            ),
            agent_event(
                "evt-agent-update-1",
                2,
                "item.updated",
                json!({
                    "item": {
                        "id": "agent-final-1",
                        "type": "agent_message",
                        "text": "写作思路：",
                        "status": "in_progress"
                    }
                }),
            ),
            agent_event(
                "evt-agent-delta-2",
                3,
                "message.delta",
                json!({
                    "itemId": "agent-final-1",
                    "text": "先用",
                    "phase": "final_answer"
                }),
            ),
            agent_event(
                "evt-agent-update-2",
                4,
                "item.updated",
                json!({
                    "item": {
                        "id": "agent-final-1",
                        "type": "agent_message",
                        "text": "写作思路：先用两句话自然说明写作思路。",
                        "status": "in_progress"
                    }
                }),
            ),
        ]);

        let items = thread_items_from_events(&stored);

        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["id"], "agent-final-1");
        assert_eq!(items[0]["type"], "agent_message");
        assert_eq!(items[0]["status"], "in_progress");
        assert_eq!(items[0]["text"], "写作思路：先用两句话自然说明写作思路。");
    }
}
