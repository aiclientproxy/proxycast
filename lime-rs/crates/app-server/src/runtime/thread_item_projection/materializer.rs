//! Event-to-typed Thread/Turn/Item materialization.
//!
//! Event payloads are an adapter boundary only.  The result of this module is
//! always the canonical `agent_protocol` DTOs; callers do not receive a raw
//! `serde_json::Value` item projection.

mod fields;
mod lifecycle;
mod lowering;

use self::fields::{
    approval_payload_source, explicit_item_id, map_string, non_empty, payload_source, value_string,
    value_u64,
};
use self::lifecycle::{
    event_timestamp_ms, item_status, parse_timestamp_ms, queued_turn_id, rollback_target,
    turn_approval_state, turn_queue_state,
};
use self::lowering::{item_family, typed_payload, ItemFamily};
use super::change_set::{ChangeSetAccumulator, MaterializationError};
use super::helpers::event_metadata;
use agent_protocol::{
    ItemId, ItemStatus, SessionId, ThreadHistoryChangeSet, ThreadId, ThreadItem, ThreadItemPayload,
    Turn, TurnError, TurnId, TurnItemsView, TurnStatus,
};
use app_server_protocol::AgentEvent;
use serde_json::Value;
use std::collections::HashMap;

/// Materialize durable AgentEvents.  Events may contain gaps, but must be in
/// canonical sequence order; lower-sequence records are stale and ignored.
pub(in crate::runtime) fn materialize_events(
    events: &[AgentEvent],
    default_session_id: &str,
    default_thread_id: &str,
) -> Result<ThreadHistoryChangeSet, MaterializationError> {
    let mut materializer = Materializer::new(default_session_id, default_thread_id);
    for event in events {
        materializer.apply(event)?;
    }
    Ok(materializer.finish())
}

#[derive(Debug, Clone, Default)]
pub(in crate::runtime) struct MaterializedEventEntities {
    pub(in crate::runtime) turn: Option<Turn>,
    pub(in crate::runtime) item: Option<ThreadItem>,
}

pub(in crate::runtime) struct IncrementalMaterializer<'a> {
    materializer: Materializer<'a>,
}

impl<'a> IncrementalMaterializer<'a> {
    pub(in crate::runtime) fn from_events(
        events: &[AgentEvent],
        default_session_id: &'a str,
        default_thread_id: &'a str,
    ) -> Result<Self, MaterializationError> {
        let mut materializer = Materializer::new(default_session_id, default_thread_id);
        for event in events {
            materializer.apply(event)?;
        }
        Ok(Self { materializer })
    }

    pub(in crate::runtime) fn apply(
        &mut self,
        event: &AgentEvent,
    ) -> Result<MaterializedEventEntities, MaterializationError> {
        self.materializer.apply(event)
    }

    pub(in crate::runtime) fn items_for_turn(&self, turn_id: &str) -> Vec<ThreadItem> {
        self.materializer.accumulator.items_for_turn(turn_id)
    }
}

struct Materializer<'a> {
    default_session_id: &'a str,
    default_thread_id: &'a str,
    seen_event_ids: HashMap<String, u64>,
    seen_sequences: HashMap<u64, String>,
    latest_sequence: u64,
    accumulator: ChangeSetAccumulator,
}

impl<'a> Materializer<'a> {
    fn new(default_session_id: &'a str, default_thread_id: &'a str) -> Self {
        Self {
            default_session_id,
            default_thread_id,
            seen_event_ids: HashMap::new(),
            seen_sequences: HashMap::new(),
            latest_sequence: 0,
            accumulator: ChangeSetAccumulator::default(),
        }
    }

    fn apply(
        &mut self,
        event: &AgentEvent,
    ) -> Result<MaterializedEventEntities, MaterializationError> {
        if let Some(previous_sequence) = self.seen_event_ids.get(&event.event_id).copied() {
            if previous_sequence == event.sequence {
                return Ok(MaterializedEventEntities::default());
            }
            return Err(MaterializationError::EventIdentityCollision {
                event_id: event.event_id.clone(),
                previous_sequence,
                sequence: event.sequence,
            });
        }
        if let Some(previous_event_id) = self.seen_sequences.get(&event.sequence) {
            if previous_event_id != &event.event_id {
                return Err(MaterializationError::SequenceCollision {
                    sequence: event.sequence,
                    previous_event_id: previous_event_id.clone(),
                    event_id: event.event_id.clone(),
                });
            }
        }
        self.seen_event_ids
            .insert(event.event_id.clone(), event.sequence);
        self.seen_sequences
            .insert(event.sequence, event.event_id.clone());
        if event.sequence < self.latest_sequence {
            return Ok(MaterializedEventEntities::default());
        }
        self.latest_sequence = event.sequence;

        if let Some(target) = rollback_target(event) {
            self.accumulator.rollback(target);
        }

        if matches!(event.event_type.as_str(), "turn.removed" | "turn.deleted") {
            if let Some(turn_id) = event
                .turn_id
                .clone()
                .or_else(|| value_string(&event.payload, &["turnId", "turn_id"]))
            {
                self.accumulator.remove_turn(turn_id);
            }
            return Ok(MaterializedEventEntities::default());
        }
        if event.event_type == "queue.removed" {
            if let Some(turn_id) = queued_turn_id(event) {
                self.accumulator.remove_turn(turn_id);
            }
            return Ok(MaterializedEventEntities::default());
        }
        if event.event_type == "queue.promoted" {
            return Ok(MaterializedEventEntities::default());
        }
        if matches!(
            event.event_type.as_str(),
            "item.removed" | "item.deleted" | "message.removed" | "tool.removed"
        ) {
            if let Some(item_id) = explicit_item_id(payload_source(&event.payload)) {
                self.accumulator.remove_item(ItemId::new(item_id));
            }
        }

        let payload_turn_id = value_string(&event.payload, &["turnId", "turn_id"]);
        let queue_added_turn_id = (event.event_type == "queue.added")
            .then(|| queued_turn_id(event))
            .flatten();
        let turn_id = if event.event_type == "queue.added" {
            queue_added_turn_id.as_deref()
        } else {
            event.turn_id.as_deref().or(payload_turn_id.as_deref())
        };
        let turn = turn_id.and_then(|turn_id| {
            self.accumulator.push_turn(
                turn_snapshot(
                    event,
                    self.default_session_id,
                    self.default_thread_id,
                    turn_id,
                ),
                event.sequence,
            )
        });

        let item = item_from_event(event, self.default_session_id, self.default_thread_id)
            .and_then(|item| self.accumulator.push_item(item));
        Ok(MaterializedEventEntities { turn, item })
    }

    fn finish(self) -> ThreadHistoryChangeSet {
        self.accumulator.finish(self.latest_sequence)
    }
}

fn item_from_event(
    event: &AgentEvent,
    default_session_id: &str,
    default_thread_id: &str,
) -> Option<ThreadItem> {
    if let Some(item) = canonical_item_from_event(event, default_session_id, default_thread_id) {
        return Some(item);
    }
    if event
        .payload
        .get("item")
        .and_then(Value::as_object)
        .is_some_and(|item| item.contains_key("payload"))
    {
        return None;
    }

    let event_type = event.event_type.as_str();
    let payload_turn_id = value_string(&event.payload, &["turnId", "turn_id"]);
    let turn_id = event.turn_id.as_deref().or(payload_turn_id.as_deref())?;
    let session_id = non_empty(&event.session_id).unwrap_or(default_session_id);
    let payload_thread_id = value_string(&event.payload, &["threadId", "thread_id"]);
    let thread_id = event
        .thread_id
        .as_deref()
        .or(payload_thread_id.as_deref())
        .unwrap_or(default_thread_id);
    let family = item_family(event_type, &event.payload)?;
    let approval_source =
        matches!(family, ItemFamily::Approval).then(|| approval_payload_source(&event.payload));
    let source = approval_source
        .as_ref()
        .unwrap_or_else(|| payload_source(&event.payload));
    let raw_item_id = family.item_id(source, turn_id, &event.event_id)?;
    let preserves_imported_codex_plan_id = matches!(family, ItemFamily::Plan)
        && map_string(source, &["sourceClient", "source_client"]).as_deref() == Some("codex")
        && family.explicit_item_id(source).is_some();
    let item_id = if preserves_imported_codex_plan_id {
        ItemId::from_legacy(raw_item_id)
    } else {
        ItemId::new(raw_item_id)
    };
    let timestamp = event_timestamp_ms(event);
    let status = if uses_legacy_import_ordinal(event)
        && matches!(family, ItemFamily::AgentMessage)
        && event_type == "message.delta"
    {
        ItemStatus::Completed
    } else {
        item_status(event_type, &event.payload)
    };
    let payload = typed_payload(family, event_type, source, item_id.as_str(), timestamp)?;
    let completed_at_ms = status.is_terminal().then_some(timestamp);
    Some(ThreadItem {
        session_id: SessionId::new(session_id),
        thread_id: ThreadId::new(thread_id),
        turn_id: TurnId::new(turn_id),
        item_id,
        sequence: event.sequence,
        ordinal: canonical_item_ordinal(event),
        created_at_ms: timestamp,
        updated_at_ms: timestamp,
        completed_at_ms,
        kind: payload.kind(),
        status,
        payload,
        metadata: materialized_item_metadata(event, source, family),
    })
}

fn materialized_item_metadata(
    event: &AgentEvent,
    source: &serde_json::Map<String, Value>,
    family: ItemFamily,
) -> Value {
    let mut metadata = event_metadata(event)
        .as_object()
        .cloned()
        .unwrap_or_default();
    if let Some(source_metadata) = source.get("metadata").and_then(Value::as_object) {
        metadata.extend(source_metadata.clone());
    }
    if matches!(family, ItemFamily::Command) {
        if let Some(command_id) = map_string(source, &["commandId", "command_id"]) {
            metadata.insert("source_call_id".to_string(), Value::String(command_id));
        }
    }
    Value::Object(metadata)
}

fn canonical_item_from_event(
    event: &AgentEvent,
    default_session_id: &str,
    default_thread_id: &str,
) -> Option<ThreadItem> {
    if !matches!(
        event.event_type.as_str(),
        "item.started"
            | "item.updated"
            | "item.completed"
            | super::super::thread_fork::FORK_CANONICAL_ITEM_EVENT_TYPE
    ) {
        return None;
    }

    let source = event.payload.get("item")?.as_object()?;
    let mut item = serde_json::from_value::<ThreadItem>(Value::Object(source.clone())).ok()?;
    if !canonical_payload_is_safe(&item.payload) {
        return None;
    }
    let turn_id = event.turn_id.as_deref()?;
    let timestamp = parse_timestamp_ms(&event.timestamp).unwrap_or(event.sequence as i64);

    item.session_id = SessionId::new(non_empty(&event.session_id).unwrap_or(default_session_id));
    item.thread_id = ThreadId::new(
        event
            .thread_id
            .as_deref()
            .and_then(non_empty)
            .unwrap_or(default_thread_id),
    );
    item.turn_id = TurnId::new(turn_id);
    item.sequence = event.sequence;
    if event.event_type != super::super::thread_fork::FORK_CANONICAL_ITEM_EVENT_TYPE {
        item.ordinal = canonical_item_ordinal(event);
        item.created_at_ms = timestamp;
        item.updated_at_ms = timestamp;
        item.status = canonical_item_lifecycle_status(event.event_type.as_str(), item.status);
        item.completed_at_ms = item.status.is_terminal().then_some(timestamp);
    }
    Some(item)
}

fn canonical_payload_is_safe(payload: &ThreadItemPayload) -> bool {
    match payload {
        ThreadItemPayload::AgentMessage { content_parts, .. } => content_parts
            .iter()
            .all(agent_protocol::MessageContentPart::is_safe),
        _ => true,
    }
}

fn canonical_item_ordinal(event: &AgentEvent) -> u64 {
    event.sequence
}

fn uses_legacy_import_ordinal(event: &AgentEvent) -> bool {
    event.payload.get("imported").and_then(Value::as_bool) == Some(true)
        && event.payload.get("sourceClient").and_then(Value::as_str) == Some("codex")
        && event.payload.get("importVersion").is_none()
}

fn canonical_item_lifecycle_status(event_type: &str, nested_status: ItemStatus) -> ItemStatus {
    match event_type {
        "item.started" | "item.updated" => ItemStatus::InProgress,
        "item.completed" => match nested_status {
            ItemStatus::Failed | ItemStatus::Interrupted | ItemStatus::Cancelled => nested_status,
            _ => ItemStatus::Completed,
        },
        _ => nested_status,
    }
}

fn turn_snapshot(
    event: &AgentEvent,
    default_session_id: &str,
    default_thread_id: &str,
    turn_id: &str,
) -> Turn {
    if matches!(
        event.event_type.as_str(),
        super::super::thread_fork::FORK_CANONICAL_ITEM_EVENT_TYPE | "context.compaction.completed"
    ) {
        if let Some(mut turn) = event
            .payload
            .get("forkTurn")
            .cloned()
            .and_then(|value| serde_json::from_value::<Turn>(value).ok())
        {
            turn.session_id =
                SessionId::new(non_empty(&event.session_id).unwrap_or(default_session_id));
            turn.thread_id = ThreadId::new(
                event
                    .thread_id
                    .as_deref()
                    .and_then(non_empty)
                    .unwrap_or(default_thread_id),
            );
            turn.turn_id = TurnId::new(turn_id);
            turn.items.clear();
            return turn;
        }
    }
    let timestamp = event_timestamp_ms(event);
    let status = match event.event_type.as_str() {
        "turn.completed" => TurnStatus::Completed,
        "turn.failed" => TurnStatus::Failed,
        "turn.canceled" | "turn.cancelled" => TurnStatus::Interrupted,
        _ => TurnStatus::InProgress,
    };
    let error = if status == TurnStatus::Failed {
        value_string(&event.payload, &["error", "message", "reason"]).map(|message| TurnError {
            message,
            code: value_string(&event.payload, &["code", "errorCode", "error_code"]),
            details: value_string(
                &event.payload,
                &["details", "errorDetails", "error_details"],
            ),
        })
    } else {
        None
    };
    let started_at_ms = (event.event_type == "turn.started").then_some(timestamp);
    let completed_at_ms = status.is_terminal().then_some(timestamp);
    let payload_thread_id = value_string(&event.payload, &["threadId", "thread_id"]);
    Turn {
        session_id: SessionId::new(non_empty(&event.session_id).unwrap_or(default_session_id)),
        thread_id: ThreadId::new(
            event
                .thread_id
                .as_deref()
                .or(payload_thread_id.as_deref())
                .unwrap_or(default_thread_id),
        ),
        turn_id: TurnId::new(turn_id),
        status,
        admission: Default::default(),
        queue: turn_queue_state(event),
        approval: turn_approval_state(event),
        items: Vec::new(),
        items_view: TurnItemsView::NotLoaded,
        error,
        created_at_ms: timestamp,
        updated_at_ms: timestamp,
        started_at_ms,
        completed_at_ms,
        duration_ms: value_u64(&event.payload, &["durationMs", "duration_ms"]),
    }
}
