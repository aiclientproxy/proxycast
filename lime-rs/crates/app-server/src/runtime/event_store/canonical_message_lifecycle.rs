use super::super::{thread_item_projection, RuntimeEvent, StoredSession};
use app_server_protocol::AgentEvent;
use serde_json::{Map, Value};
use std::collections::HashMap;

const LIFECYCLE_SOURCE: &str = "runtime_message_reasoning.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum MessageItemFamily {
    User,
    Agent,
    Reasoning,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum LifecycleOwner {
    #[default]
    Unknown,
    ManagedActive,
    ManagedCompleted,
    ExternalActive,
    ExternalCompleted,
}

#[derive(Clone, Debug, Default)]
pub(in crate::runtime) struct CanonicalMessageLifecycleState {
    items: HashMap<(MessageItemFamily, String), LifecycleOwner>,
    active_items: HashMap<MessageItemFamily, String>,
}

impl CanonicalMessageLifecycleState {
    pub(in crate::runtime) fn from_events(events: &[AgentEvent], turn_id: &str) -> Self {
        let mut state = Self::default();
        for event in events
            .iter()
            .filter(|event| event.turn_id.as_deref() == Some(turn_id))
        {
            state.observe_stored_event(event);
        }
        state
    }

    fn owner(&self, family: MessageItemFamily, item_id: &str) -> LifecycleOwner {
        self.items
            .get(&(family, item_id.to_string()))
            .copied()
            .unwrap_or_default()
    }

    fn start(
        &mut self,
        family: MessageItemFamily,
        item_id: &str,
        owner: LifecycleOwner,
    ) -> Result<(), String> {
        if let Some(active_item_id) = self.active_items.get(&family) {
            return Err(format!(
                "{} Item {} started while {} is still active",
                family.label(),
                item_id,
                active_item_id
            ));
        }
        if self.owner(family, item_id) != LifecycleOwner::Unknown {
            return Err(format!(
                "{} Item {} started more than once",
                family.label(),
                item_id
            ));
        }
        self.items.insert((family, item_id.to_string()), owner);
        self.active_items.insert(family, item_id.to_string());
        Ok(())
    }

    fn complete(
        &mut self,
        family: MessageItemFamily,
        item_id: &str,
        expected: LifecycleOwner,
        completed: LifecycleOwner,
    ) -> Result<(), String> {
        let owner = self.owner(family, item_id);
        if owner != expected {
            return Err(format!(
                "{} Item {} completed from invalid lifecycle state {:?}",
                family.label(),
                item_id,
                owner
            ));
        }
        self.items.insert((family, item_id.to_string()), completed);
        if self.active_items.get(&family).map(String::as_str) == Some(item_id) {
            self.active_items.remove(&family);
        }
        Ok(())
    }

    fn begin_or_validate_presentation(
        &mut self,
        family: MessageItemFamily,
        item_id: &str,
        has_explicit_item_id: bool,
        event_type: &str,
    ) -> Result<bool, String> {
        if let Some(active_item_id) = self.active_items.get(&family) {
            if active_item_id != item_id {
                return Err(format!(
                    "{} event {} targets Item {} while {} is active",
                    family.label(),
                    event_type,
                    item_id,
                    active_item_id
                ));
            }
        }
        match self.owner(family, item_id) {
            LifecycleOwner::Unknown => {
                self.start(family, item_id, LifecycleOwner::ManagedActive)?;
                Ok(true)
            }
            LifecycleOwner::ManagedActive => Ok(false),
            LifecycleOwner::ExternalActive if has_explicit_item_id => Ok(false),
            LifecycleOwner::ExternalActive => Err(format!(
                "external {} Item {} requires an explicit itemId on {}",
                family.label(),
                item_id,
                event_type
            )),
            LifecycleOwner::ManagedCompleted | LifecycleOwner::ExternalCompleted => Err(format!(
                "late {} event {} cannot update completed Item {}",
                family.label(),
                event_type,
                item_id
            )),
        }
    }

    fn observe_stored_event(&mut self, event: &AgentEvent) {
        let Some(family) = lifecycle_item_family(event.event_type.as_str(), &event.payload) else {
            return;
        };
        let Some(item_id) = explicit_item_identity(&event.payload) else {
            return;
        };
        let managed = is_managed_lifecycle_payload(&event.payload);
        let completed = event.event_type == "item.completed"
            || event
                .payload
                .get("item")
                .and_then(|item| string_field(item, &["status"]))
                .is_some_and(is_terminal_item_status);
        let owner = if completed {
            if managed {
                LifecycleOwner::ManagedCompleted
            } else {
                LifecycleOwner::ExternalCompleted
            }
        } else if managed {
            LifecycleOwner::ManagedActive
        } else {
            LifecycleOwner::ExternalActive
        };
        self.items.insert((family, item_id.clone()), owner);
        if matches!(
            owner,
            LifecycleOwner::ManagedActive | LifecycleOwner::ExternalActive
        ) {
            self.active_items.insert(family, item_id);
        } else if self.active_items.get(&family).map(String::as_str) == Some(item_id.as_str()) {
            self.active_items.remove(&family);
        }
    }

    fn observe_runtime_lifecycle_event(&mut self, event: &RuntimeEvent) -> Result<bool, String> {
        let Some(family) = lifecycle_item_family(event.event_type.as_str(), &event.payload) else {
            return Ok(false);
        };
        let item_id = explicit_item_identity(&event.payload).ok_or_else(|| {
            format!(
                "external {} lifecycle event {} requires itemId",
                family.label(),
                event.event_type
            )
        })?;
        let managed = is_managed_lifecycle_payload(&event.payload);
        let active_owner = if managed {
            LifecycleOwner::ManagedActive
        } else {
            LifecycleOwner::ExternalActive
        };
        let completed_owner = if managed {
            LifecycleOwner::ManagedCompleted
        } else {
            LifecycleOwner::ExternalCompleted
        };
        match event.event_type.as_str() {
            "item.started" => self.start(family, &item_id, active_owner)?,
            "item.updated" => {
                if self.owner(family, &item_id) != active_owner {
                    return Err(format!(
                        "{} Item {} updated without a matching item.started",
                        family.label(),
                        item_id
                    ));
                }
                if event
                    .payload
                    .get("item")
                    .and_then(|item| string_field(item, &["status"]))
                    .is_some_and(is_terminal_item_status)
                {
                    self.complete(family, &item_id, active_owner, completed_owner)?;
                }
            }
            "item.completed" => {
                self.complete(family, &item_id, active_owner, completed_owner)?;
            }
            _ => {
                return Err(format!(
                    "unsupported {} lifecycle event {} for Item {}",
                    family.label(),
                    event.event_type,
                    item_id
                ));
            }
        }
        Ok(true)
    }
}

impl MessageItemFamily {
    fn label(self) -> &'static str {
        match self {
            Self::User => "UserMessage",
            Self::Agent => "AgentMessage",
            Self::Reasoning => "Reasoning",
        }
    }

    fn fallback_item_id(self, turn_id: &str) -> String {
        match self {
            Self::User => format!("user-{turn_id}"),
            Self::Agent => format!("agent-{turn_id}"),
            Self::Reasoning => format!("reasoning-{turn_id}"),
        }
    }
}

pub(super) fn with_canonical_message_reasoning_lifecycle(
    existing_events: &[AgentEvent],
    turn_id: Option<&str>,
    runtime_events: Vec<RuntimeEvent>,
    cached_state: Option<&mut CanonicalMessageLifecycleState>,
) -> Result<Vec<RuntimeEvent>, String> {
    let Some(turn_id) = turn_id else {
        return Ok(runtime_events);
    };
    let mut recovered_state;
    let state = match cached_state {
        Some(state) => state,
        None => {
            recovered_state = CanonicalMessageLifecycleState::from_events(existing_events, turn_id);
            &mut recovered_state
        }
    };

    let mut normalized = Vec::with_capacity(runtime_events.len() + 4);
    for event in runtime_events {
        if event.event_type.starts_with("item.") && state.observe_runtime_lifecycle_event(&event)? {
            normalized.push(event);
            continue;
        }
        match event.event_type.as_str() {
            "message.created" if is_user_message(&event.payload) => {
                let (item_id, explicit) =
                    runtime_item_identity(MessageItemFamily::User, &event.payload, turn_id);
                if state.begin_or_validate_presentation(
                    MessageItemFamily::User,
                    &item_id,
                    explicit,
                    event.event_type.as_str(),
                )? {
                    normalized.push(lifecycle_event(
                        "item.started",
                        MessageItemFamily::User,
                        &item_id,
                        "in_progress",
                        Some(&event.payload),
                    ));
                    let completed = lifecycle_event(
                        "item.completed",
                        MessageItemFamily::User,
                        &item_id,
                        "completed",
                        Some(&event.payload),
                    );
                    normalized.push(event);
                    normalized.push(completed);
                    state.complete(
                        MessageItemFamily::User,
                        &item_id,
                        LifecycleOwner::ManagedActive,
                        LifecycleOwner::ManagedCompleted,
                    )?;
                } else {
                    normalized.push(event);
                }
            }
            "message.delta" | "message.delta_batch" | "message.batch" => {
                let (item_id, explicit) =
                    runtime_item_identity(MessageItemFamily::Agent, &event.payload, turn_id);
                if state.begin_or_validate_presentation(
                    MessageItemFamily::Agent,
                    &item_id,
                    explicit,
                    event.event_type.as_str(),
                )? {
                    normalized.push(lifecycle_event(
                        "item.started",
                        MessageItemFamily::Agent,
                        &item_id,
                        "in_progress",
                        None,
                    ));
                }
                normalized.push(event);
            }
            "message.completed" => {
                let (item_id, explicit) =
                    runtime_item_identity(MessageItemFamily::Agent, &event.payload, turn_id);
                match state.owner(MessageItemFamily::Agent, &item_id) {
                    LifecycleOwner::ManagedActive => {
                        normalized.push(lifecycle_event(
                            "item.completed",
                            MessageItemFamily::Agent,
                            &item_id,
                            lifecycle_status(&event.payload, "completed"),
                            Some(&event.payload),
                        ));
                        state.complete(
                            MessageItemFamily::Agent,
                            &item_id,
                            LifecycleOwner::ManagedActive,
                            LifecycleOwner::ManagedCompleted,
                        )?;
                    }
                    LifecycleOwner::ExternalActive if explicit => normalized.push(event),
                    LifecycleOwner::ExternalActive => {
                        return Err(format!(
                            "external AgentMessage Item {} requires itemId on message.completed",
                            item_id
                        ));
                    }
                    LifecycleOwner::Unknown => {
                        return Err(format!(
                            "AgentMessage Item {} completed without message content or item.started",
                            item_id
                        ));
                    }
                    LifecycleOwner::ManagedCompleted | LifecycleOwner::ExternalCompleted => {
                        return Err(format!(
                            "AgentMessage Item {} completed more than once",
                            item_id
                        ));
                    }
                }
            }
            "reasoning.started" => {
                let (item_id, _) =
                    runtime_item_identity(MessageItemFamily::Reasoning, &event.payload, turn_id);
                state.start(
                    MessageItemFamily::Reasoning,
                    &item_id,
                    LifecycleOwner::ManagedActive,
                )?;
                normalized.push(lifecycle_event(
                    "item.started",
                    MessageItemFamily::Reasoning,
                    &item_id,
                    "in_progress",
                    Some(&event.payload),
                ));
            }
            "reasoning.delta" | "reasoning.summary" | "reasoning.completed" | "reasoning.final" => {
                let (item_id, explicit) =
                    runtime_item_identity(MessageItemFamily::Reasoning, &event.payload, turn_id);
                if state.begin_or_validate_presentation(
                    MessageItemFamily::Reasoning,
                    &item_id,
                    explicit,
                    event.event_type.as_str(),
                )? {
                    let start_context = lifecycle_start_context(&event.payload);
                    normalized.push(lifecycle_event(
                        "item.started",
                        MessageItemFamily::Reasoning,
                        &item_id,
                        "in_progress",
                        start_context.as_ref(),
                    ));
                }
                normalized.push(event);
            }
            "reasoning.ended" => {
                let (item_id, explicit) =
                    runtime_item_identity(MessageItemFamily::Reasoning, &event.payload, turn_id);
                match state.owner(MessageItemFamily::Reasoning, &item_id) {
                    LifecycleOwner::ManagedActive => {
                        normalized.push(lifecycle_event(
                            "item.completed",
                            MessageItemFamily::Reasoning,
                            &item_id,
                            lifecycle_status(&event.payload, "completed"),
                            Some(&event.payload),
                        ));
                        state.complete(
                            MessageItemFamily::Reasoning,
                            &item_id,
                            LifecycleOwner::ManagedActive,
                            LifecycleOwner::ManagedCompleted,
                        )?;
                    }
                    LifecycleOwner::ExternalActive if explicit => normalized.push(event),
                    LifecycleOwner::ExternalActive => {
                        return Err(format!(
                            "external Reasoning Item {} requires itemId on reasoning.ended",
                            item_id
                        ));
                    }
                    LifecycleOwner::Unknown => {
                        return Err(format!(
                            "Reasoning Item {} ended without item.started or reasoning content",
                            item_id
                        ));
                    }
                    LifecycleOwner::ManagedCompleted | LifecycleOwner::ExternalCompleted => {
                        return Err(format!("Reasoning Item {} ended more than once", item_id));
                    }
                }
            }
            _ => normalized.push(event),
        }
    }
    Ok(normalized)
}

pub(super) fn attach_canonical_item_entity(
    stored: &StoredSession,
    pending_events: &[AgentEvent],
    event: &mut AgentEvent,
) -> Result<(), String> {
    let is_item_lifecycle = event.event_type.starts_with("item.");
    let is_mcp_progress = is_mcp_tool_progress_event(event);
    if event.payload.get("item").is_some()
        || (!is_item_lifecycle && !is_mcp_progress)
        || (is_item_lifecycle && !is_managed_lifecycle_payload(&event.payload))
    {
        return Ok(());
    }

    let mut history = Vec::with_capacity(stored.events.len() + pending_events.len() + 1);
    history.extend(stored.events.iter().cloned());
    history.extend(pending_events.iter().cloned());
    history.push(event.clone());
    let changes = thread_item_projection::materialize_events(
        &history,
        &stored.session.session_id,
        &stored.session.thread_id,
    )
    .map_err(|error| {
        format!(
            "cannot materialize canonical item event {}: {error}",
            event.event_id
        )
    })?;
    let item = changes
        .changed_items
        .into_iter()
        .find(|item| item.sequence == event.sequence)
        .ok_or_else(|| format!("canonical item event {} produced no Item", event.event_id))?;
    let payload = event.payload.as_object_mut().ok_or_else(|| {
        format!(
            "canonical item event {} has non-object payload",
            event.event_id
        )
    })?;
    payload.insert(
        "item".to_string(),
        serde_json::to_value(item).map_err(|error| {
            format!(
                "cannot serialize canonical Item for {}: {error}",
                event.event_id
            )
        })?,
    );
    Ok(())
}

fn is_mcp_tool_progress_event(event: &AgentEvent) -> bool {
    if event.event_type != "tool.progress" {
        return false;
    }
    let Some(metadata) = event
        .payload
        .pointer("/progress/metadata")
        .and_then(Value::as_object)
    else {
        return false;
    };
    let Some(notification_kind) = ["notification_kind", "notificationKind"]
        .into_iter()
        .find_map(|key| metadata.get(key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|kind| {
            matches!(
                *kind,
                "mcp_progress"
                    | "mcp_resources_changed"
                    | "mcp_tools_changed"
                    | "mcp_prompts_changed"
            )
        })
    else {
        return false;
    };
    let has_server = string_field(
        &event.payload,
        &["serverName", "server_name", "mcpServer", "mcp_server"],
    )
    .or_else(|| {
        ["server_name", "serverName"]
            .into_iter()
            .find_map(|key| metadata.get(key).and_then(Value::as_str))
    })
    .is_some();
    let has_tool = string_field(&event.payload, &["toolName", "tool_name", "name"])
        .or_else(|| {
            ["tool_name", "toolName"]
                .into_iter()
                .find_map(|key| metadata.get(key).and_then(Value::as_str))
        })
        .is_some();
    !notification_kind.is_empty() && has_server && has_tool
}

fn lifecycle_event(
    event_type: &str,
    family: MessageItemFamily,
    item_id: &str,
    status: &str,
    source_payload: Option<&Value>,
) -> RuntimeEvent {
    let mut payload = source_payload
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    payload.insert(
        "itemType".to_string(),
        Value::String(
            match family {
                MessageItemFamily::User => "user_message",
                MessageItemFamily::Agent => "agent_message",
                MessageItemFamily::Reasoning => "reasoning",
            }
            .to_string(),
        ),
    );
    payload.insert("status".to_string(), Value::String(status.to_string()));
    payload.insert("itemId".to_string(), Value::String(item_id.to_string()));
    payload.insert(
        "canonicalLifecycle".to_string(),
        Value::String(LIFECYCLE_SOURCE.to_string()),
    );
    let metadata = payload
        .entry("metadata")
        .or_insert_with(|| Value::Object(Map::new()));
    if let Some(metadata) = metadata.as_object_mut() {
        metadata.insert(
            "source".to_string(),
            Value::String(LIFECYCLE_SOURCE.to_string()),
        );
    }
    RuntimeEvent::new(event_type, Value::Object(payload))
}

fn lifecycle_item_family(event_type: &str, payload: &Value) -> Option<MessageItemFamily> {
    if !event_type.starts_with("item.") {
        return None;
    }
    let item = payload.get("item").unwrap_or(payload);
    let kind = string_field(item, &["kind", "type", "itemType", "item_type"])?;
    match kind.to_ascii_lowercase().as_str() {
        "usermessage" | "user_message" | "user" => Some(MessageItemFamily::User),
        "agentmessage" | "agent_message" | "assistant" | "message" => {
            Some(MessageItemFamily::Agent)
        }
        "reasoning" | "reasoning_message" => Some(MessageItemFamily::Reasoning),
        _ => None,
    }
}

fn runtime_item_identity(
    family: MessageItemFamily,
    payload: &Value,
    turn_id: &str,
) -> (String, bool) {
    match explicit_item_identity(payload) {
        Some(item_id) => (item_id, true),
        None => (family.fallback_item_id(turn_id), false),
    }
}

fn explicit_item_identity(payload: &Value) -> Option<String> {
    let item = payload.get("item").unwrap_or(payload);
    string_field(
        item,
        &[
            "itemId",
            "item_id",
            "id",
            "messageId",
            "message_id",
            "reasoningId",
            "reasoning_id",
        ],
    )
    .map(str::to_string)
}

fn is_terminal_item_status(status: &str) -> bool {
    matches!(
        status.to_ascii_lowercase().as_str(),
        "completed" | "failed" | "interrupted" | "cancelled" | "canceled"
    )
}

fn is_managed_lifecycle_payload(payload: &Value) -> bool {
    payload.get("canonicalLifecycle").and_then(Value::as_str) == Some(LIFECYCLE_SOURCE)
}

fn is_user_message(payload: &Value) -> bool {
    string_field(payload, &["role", "author"]).is_some_and(|role| role.eq_ignore_ascii_case("user"))
        || payload.get("input").is_some()
}

fn lifecycle_status<'a>(payload: &'a Value, fallback: &'a str) -> &'a str {
    string_field(payload, &["status", "state"]).unwrap_or(fallback)
}

fn lifecycle_start_context(payload: &Value) -> Option<Value> {
    let payload = payload.as_object()?;
    let mut context = Map::new();
    for key in [
        "ordinal",
        "imported",
        "sourceClient",
        "source_client",
        "importVersion",
        "sourceEventSeq",
        "source_event_seq",
        "sourceProvenance",
        "phase",
        "metadata",
    ] {
        if let Some(value) = payload.get(key) {
            context.insert(key.to_string(), value.clone());
        }
    }
    (!context.is_empty()).then_some(Value::Object(context))
}

fn string_field<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
    let object = value.as_object()?;
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests;
