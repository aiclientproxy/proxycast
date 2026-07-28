//! App Server current event store 到 provider transcript 的投影。
//!
//! Codex 的 history 是有序 response item，而不是 UI 聚合后的消息文本。本模块直接从
//! RuntimeCore 已持久化的 user/message/tool 事件恢复 provider 所需的最小 item 顺序，
//! 避免任何旧 session 或 provider adapter 参与多轮采样。

mod canonical;

use super::StoredSession;
use agent_protocol::{AgentInput, ThreadItem};
use agent_runtime::reply_input::{RuntimeReplyInput, RuntimeReplyInputPart};
use app_server_protocol::AgentEvent;
use model_provider::current_client::{
    CurrentProviderContent, CurrentProviderMessage, CurrentProviderRole,
};
use serde_json::Value;
use std::collections::HashMap;

const PROVIDER_TOOL_OUTPUT_MAX_BYTES: usize = 10_000;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProviderRouteIdentity {
    provider: String,
    model: String,
}

#[derive(Debug, Clone, Default)]
pub struct ProviderTurnHistory {
    messages: Vec<CurrentProviderMessage>,
    previous_route: Option<ProviderRouteIdentity>,
}

impl ProviderTurnHistory {
    pub fn messages_for_route(&self, provider: &str, model: &str) -> Vec<CurrentProviderMessage> {
        let mut messages = self.messages.clone();
        let Some(previous) = self.previous_route.as_ref() else {
            return messages;
        };
        let current = ProviderRouteIdentity {
            provider: provider.to_string(),
            model: model.to_string(),
        };
        if previous != &current {
            messages.push(model_switch_message(previous, &current));
        }
        messages
    }

    #[cfg(test)]
    pub(crate) fn into_messages(self) -> Vec<CurrentProviderMessage> {
        self.messages
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

/// Current turn input is supplied separately to the provider. It remains durable and visible in
/// the canonical Item log, but must not be submitted twice in the same provider request.
#[cfg(test)]
pub(in crate::runtime) fn provider_history_excluding_current_turn_input(
    stored: &StoredSession,
    sidecar_store: Option<&super::SidecarStore>,
    turn_id: &str,
) -> Result<Vec<CurrentProviderMessage>, super::RuntimeCoreError> {
    let history =
        provider_turn_history_excluding_current_turn_input(stored, sidecar_store, turn_id)?;
    let Some(current) = current_turn_provider_route(stored, turn_id) else {
        return Ok(history.into_messages());
    };
    Ok(history.messages_for_route(&current.provider, &current.model))
}

pub(in crate::runtime) fn provider_turn_history_excluding_current_turn_input(
    stored: &StoredSession,
    sidecar_store: Option<&super::SidecarStore>,
    turn_id: &str,
) -> Result<ProviderTurnHistory, super::RuntimeCoreError> {
    let (replacement_history, events) = provider_history_source(stored);
    let events = events
        .into_iter()
        .filter(|event| {
            event.turn_id.as_deref() != Some(turn_id)
                || !super::turn_input_events::is_provider_input_event(event)
        })
        .collect::<Vec<_>>();
    let mut messages = replacement_history.unwrap_or_default();
    messages.extend(
        messages_from_events_with_provider_input(&events, |input| {
            RuntimeReplyInput::try_from_user_parts(input, |media| {
                super::input_media::resolve_runtime_input_media(
                    media,
                    sidecar_store,
                    &stored.session.session_id,
                )
            })
            .map_err(|error| error.to_string())
        })
        .map_err(super::RuntimeCoreError::Backend)?,
    );
    Ok(ProviderTurnHistory {
        messages,
        previous_route: latest_completed_provider_route(&stored.events, turn_id),
    })
}

fn model_switch_message(
    previous: &ProviderRouteIdentity,
    current: &ProviderRouteIdentity,
) -> CurrentProviderMessage {
    CurrentProviderMessage::developer(vec![CurrentProviderContent::Text(render_model_switch(
        previous, current,
    ))])
}

fn latest_completed_provider_route(
    events: &[AgentEvent],
    current_turn_id: &str,
) -> Option<ProviderRouteIdentity> {
    for terminal in events.iter().rev().filter(|event| {
        event.event_type == "turn.completed" && event.turn_id.as_deref() != Some(current_turn_id)
    }) {
        let Some(turn_id) = terminal.turn_id.as_deref() else {
            continue;
        };
        if let Some(route) = events
            .iter()
            .rev()
            .find(|event| {
                event.event_type == "routing.decision.made"
                    && event.turn_id.as_deref() == Some(turn_id)
                    && event.sequence <= terminal.sequence
            })
            .and_then(provider_route_from_routing_event)
        {
            return Some(route);
        }
    }
    None
}

fn provider_route_from_routing_event(event: &AgentEvent) -> Option<ProviderRouteIdentity> {
    let decision = event
        .payload
        .get("routingDecision")
        .unwrap_or(&event.payload);
    provider_route_from_fields(decision, "selectedProvider", "selectedModel")
}

#[cfg(test)]
fn current_turn_provider_route(
    stored: &StoredSession,
    turn_id: &str,
) -> Option<ProviderRouteIdentity> {
    let route = stored
        .turn_runtime_options
        .get(turn_id)?
        .runtime_metadata()?
        .get("agentControlRoute")?;
    if route.get("schemaVersion").and_then(Value::as_u64) != Some(2) {
        return None;
    }
    provider_route_from_fields(route, "providerPreference", "modelPreference")
}

fn provider_route_from_fields(
    value: &Value,
    provider_key: &str,
    model_key: &str,
) -> Option<ProviderRouteIdentity> {
    let field = |key| {
        value
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
    };
    Some(ProviderRouteIdentity {
        provider: field(provider_key)?,
        model: field(model_key)?,
    })
}

fn render_model_switch(
    previous: &ProviderRouteIdentity,
    current: &ProviderRouteIdentity,
) -> String {
    format!(
        "<model_switch>\n\
The user was previously using a different model. Please continue the conversation according to the following instructions:\n\n\
Continue to follow the current system and developer instructions.\n\
<previous_route>\n  <provider>{}</provider>\n  <model>{}</model>\n</previous_route>\n\
<current_route>\n  <provider>{}</provider>\n  <model>{}</model>\n</current_route>\n\
</model_switch>",
        escape_xml_text(&previous.provider),
        escape_xml_text(&previous.model),
        escape_xml_text(&current.provider),
        escape_xml_text(&current.model),
    )
}

fn escape_xml_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn provider_history_source(
    stored: &StoredSession,
) -> (Option<Vec<CurrentProviderMessage>>, Vec<AgentEvent>) {
    let Some(compaction) = stored
        .events
        .iter()
        .rev()
        .find(|event| event.event_type == "context.compaction.completed")
    else {
        return (None, stored.events.clone());
    };
    let replacement_history = replacement_history_from_compaction(compaction);
    let Some(tail_start_turn_id) = compaction
        .payload
        .get("tailStartTurnId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return (None, stored.events.clone());
    };

    let Some(start_index) = stored
        .events
        .iter()
        .position(|event| event.turn_id.as_deref() == Some(tail_start_turn_id))
    else {
        // A malformed or imported compaction marker must never discard history.
        return (None, stored.events.clone());
    };
    let Some(replacement_history) = replacement_history else {
        // Legacy compaction markers do not carry a durable replacement history. Rebuild the
        // complete event history instead of guessing which prefix can be dropped.
        return (None, stored.events.clone());
    };
    (
        Some(replacement_history),
        stored.events[start_index..].to_vec(),
    )
}

fn replacement_history_from_compaction(event: &AgentEvent) -> Option<Vec<CurrentProviderMessage>> {
    let field = |name: &str| {
        event
            .payload
            .get("artifact")
            .filter(|value| value.is_object())
            .and_then(|artifact| artifact.get(name))
            .or_else(|| event.payload.get(name))
    };
    if field("windowNumber").and_then(Value::as_u64).is_none()
        || field("firstWindowId").and_then(Value::as_str).is_none()
        || field("windowId").and_then(Value::as_str).is_none()
    {
        return None;
    }
    let replacement_history = field("replacementHistory")?.as_array()?;
    replacement_history
        .iter()
        .map(provider_message_from_replacement_item)
        .collect()
}

fn provider_message_from_replacement_item(item: &Value) -> Option<CurrentProviderMessage> {
    let role = match item.get("role").and_then(Value::as_str)? {
        "user" => CurrentProviderRole::User,
        "developer" => CurrentProviderRole::Developer,
        "assistant" => CurrentProviderRole::Assistant,
        "tool" => CurrentProviderRole::Tool,
        _ => return None,
    };
    let content = replacement_content(item.get("content")?)?;
    Some(CurrentProviderMessage { role, content })
}

fn replacement_content(value: &Value) -> Option<Vec<CurrentProviderContent>> {
    if let Some(text) = value.as_str().filter(|text| !text.is_empty()) {
        return Some(vec![CurrentProviderContent::Text(text.to_string())]);
    }
    let parts = value.as_array()?;
    let mut content = Vec::with_capacity(parts.len());
    for part in parts {
        let text = part
            .as_str()
            .or_else(|| part.get("text").and_then(Value::as_str))
            .filter(|text| !text.is_empty())?;
        content.push(CurrentProviderContent::Text(text.to_string()));
    }
    (!content.is_empty()).then_some(content)
}

#[cfg(test)]
fn messages_from_events(events: &[AgentEvent]) -> Vec<CurrentProviderMessage> {
    messages_from_events_with_provider_input(events, |input| {
        RuntimeReplyInput::try_from_user_parts(input, |media| {
            let (uri, detail) = match media {
                agent_runtime::reply_input::RuntimeReplyInputMedia::Image { uri, detail }
                | agent_runtime::reply_input::RuntimeReplyInputMedia::LocalImage {
                    path: uri,
                    detail,
                } => (uri, detail),
            };
            Ok::<_, String>(agent_runtime::reply_input::RuntimeReplyInputImage {
                uri: uri.clone(),
                media_type: "image/*".to_string(),
                provider_data: uri.starts_with("data:").then(|| uri.clone()),
                detail,
            })
        })
        .map_err(|error| error.to_string())
    })
    .expect("identity provider input projection cannot fail")
}

fn messages_from_events_with_provider_input<G>(
    events: &[AgentEvent],
    mut provider_input: G,
) -> Result<Vec<CurrentProviderMessage>, String>
where
    G: FnMut(Vec<AgentInput>) -> Result<RuntimeReplyInput, String>,
{
    let mut messages = Vec::new();
    let mut assistant_content = Vec::new();
    let mut assistant_text_by_item = HashMap::new();
    let mut tool_results = Vec::new();

    for event in events {
        match event.event_type.as_str() {
            super::thread_fork::FORK_CANONICAL_ITEM_EVENT_TYPE => {
                let item = serde_json::from_value::<ThreadItem>(
                    event.payload.get("item").cloned().ok_or_else(|| {
                        format!(
                            "fork canonical history event {} omitted item",
                            event.event_id
                        )
                    })?,
                )
                .map_err(|error| {
                    format!(
                        "fork canonical history event {} has invalid item: {error}",
                        event.event_id
                    )
                })?;
                canonical::append_fork_item(
                    &item,
                    &mut messages,
                    &mut assistant_content,
                    &mut assistant_text_by_item,
                    &mut tool_results,
                    &mut provider_input,
                )?;
            }
            super::thread_fork::FORK_INTERRUPTED_MARKER_EVENT_TYPE => {
                flush_assistant(
                    &mut messages,
                    &mut assistant_content,
                    &mut assistant_text_by_item,
                );
                flush_tool_results(&mut messages, &mut tool_results);
                let text = event
                    .payload
                    .get("text")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .ok_or_else(|| {
                        format!(
                            "fork interrupted marker {} omitted developer text",
                            event.event_id
                        )
                    })?;
                messages.push(CurrentProviderMessage::developer(vec![
                    CurrentProviderContent::Text(text.to_string()),
                ]));
            }
            super::thread_guardian::GUARDIAN_APPROVAL_EVENT_TYPE => {
                flush_assistant(
                    &mut messages,
                    &mut assistant_content,
                    &mut assistant_text_by_item,
                );
                flush_tool_results(&mut messages, &mut tool_results);
                let text = event
                    .payload
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|text| !text.trim().is_empty())
                    .ok_or_else(|| {
                        format!(
                            "Guardian approval event {} omitted developer text",
                            event.event_id
                        )
                    })?;
                messages.push(CurrentProviderMessage::developer(vec![
                    CurrentProviderContent::Text(text.to_string()),
                ]));
            }
            super::thread_inject_items::RESPONSE_ITEM_INJECTED_EVENT_TYPE => {
                flush_assistant(
                    &mut messages,
                    &mut assistant_content,
                    &mut assistant_text_by_item,
                );
                flush_tool_results(&mut messages, &mut tool_results);
                let item = event.payload.get("item").cloned().ok_or_else(|| {
                    format!(
                        "injected response item event {} omitted item",
                        event.event_id
                    )
                })?;
                let parsed = serde_json::from_value::<agent_protocol::ResponseItem>(item.clone())
                    .map_err(|error| {
                    format!(
                        "injected response item event {} is invalid: {error}",
                        event.event_id
                    )
                })?;
                if parsed.contains_remote_image_url() {
                    return Err(format!(
                        "injected response item event {} contains a remote image URL",
                        event.event_id
                    ));
                }
                messages.push(CurrentProviderMessage::raw_response_item(item));
            }
            "message.created" | "thread.goal.continuation" => {
                flush_assistant(
                    &mut messages,
                    &mut assistant_content,
                    &mut assistant_text_by_item,
                );
                flush_tool_results(&mut messages, &mut tool_results);
                if let Some(message) = user_message_from_event(event, &mut provider_input)? {
                    messages.push(message);
                }
            }
            "message.delta" | "message.delta_batch" | "message.batch" => {
                flush_tool_results(&mut messages, &mut tool_results);
                if let Some(text) = text_from_payload(&event.payload) {
                    assistant_text_by_item
                        .entry(message_item_key(event))
                        .or_insert_with(String::new)
                        .push_str(&text);
                    assistant_content.push(CurrentProviderContent::Text(text));
                }
            }
            "message.completed" => {
                flush_tool_results(&mut messages, &mut tool_results);
                if let Some(snapshot) = text_from_payload(&event.payload) {
                    append_completed_message_snapshot(
                        &mut assistant_content,
                        &mut assistant_text_by_item,
                        message_item_key(event),
                        snapshot,
                    );
                }
            }
            "reasoning.delta" => {
                flush_tool_results(&mut messages, &mut tool_results);
                if let Some(text) = text_from_payload(&event.payload) {
                    assistant_content.push(CurrentProviderContent::Reasoning(text));
                }
            }
            "item.started" => {
                if let Some(call) = canonical::tool_call_from_event(event) {
                    flush_tool_results(&mut messages, &mut tool_results);
                    assistant_content.push(CurrentProviderContent::ToolCall(call));
                }
            }
            "item.completed" => {
                if let Some(result) = canonical::tool_result_from_event(event) {
                    flush_assistant(
                        &mut messages,
                        &mut assistant_content,
                        &mut assistant_text_by_item,
                    );
                    tool_results.push(CurrentProviderContent::ToolResult(result));
                }
            }
            _ => {}
        }
    }

    flush_assistant(
        &mut messages,
        &mut assistant_content,
        &mut assistant_text_by_item,
    );
    flush_tool_results(&mut messages, &mut tool_results);
    Ok(messages)
}

fn flush_assistant(
    messages: &mut Vec<CurrentProviderMessage>,
    assistant_content: &mut Vec<CurrentProviderContent>,
    assistant_text_by_item: &mut HashMap<String, String>,
) {
    if assistant_content.is_empty() {
        assistant_text_by_item.clear();
        return;
    }
    messages.push(CurrentProviderMessage::assistant(std::mem::take(
        assistant_content,
    )));
    assistant_text_by_item.clear();
}

fn append_completed_message_snapshot(
    assistant_content: &mut Vec<CurrentProviderContent>,
    assistant_text_by_item: &mut HashMap<String, String>,
    item_key: String,
    snapshot: String,
) {
    let aggregate = assistant_content
        .iter()
        .filter_map(|content| match content {
            CurrentProviderContent::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    if snapshot == aggregate || aggregate.starts_with(&snapshot) {
        return;
    }
    if !aggregate.is_empty() && snapshot.starts_with(&aggregate) {
        assistant_content.push(CurrentProviderContent::Text(
            snapshot[aggregate.len()..].to_string(),
        ));
        assistant_text_by_item.insert(item_key, snapshot);
        return;
    }

    let accumulated = assistant_text_by_item.entry(item_key).or_default();
    let suffix = if accumulated.is_empty() {
        snapshot.as_str()
    } else if snapshot == *accumulated || accumulated.starts_with(&snapshot) {
        ""
    } else if snapshot.starts_with(accumulated.as_str()) {
        &snapshot[accumulated.len()..]
    } else {
        snapshot.as_str()
    };
    if !suffix.is_empty() {
        assistant_content.push(CurrentProviderContent::Text(suffix.to_string()));
    }
    if accumulated.is_empty() || snapshot.starts_with(accumulated.as_str()) {
        *accumulated = snapshot;
    } else if !accumulated.starts_with(&snapshot) {
        accumulated.push_str(&snapshot);
    }
}

fn message_item_key(event: &AgentEvent) -> String {
    let payload = provider_event_payload(&event.payload);
    ["itemId", "item_id", "messageId", "message_id", "id"]
        .iter()
        .find_map(|key| payload.get(*key).and_then(Value::as_str))
        .or_else(|| event.turn_id.as_deref())
        .unwrap_or(event.event_id.as_str())
        .to_string()
}

fn flush_tool_results(
    messages: &mut Vec<CurrentProviderMessage>,
    tool_results: &mut Vec<CurrentProviderContent>,
) {
    if tool_results.is_empty() {
        return;
    }
    messages.push(CurrentProviderMessage::tool(std::mem::take(tool_results)));
}

fn user_message_from_event<G>(
    event: &AgentEvent,
    provider_input: &mut G,
) -> Result<Option<CurrentProviderMessage>, String>
where
    G: FnMut(Vec<AgentInput>) -> Result<RuntimeReplyInput, String>,
{
    let input = event
        .payload
        .get("input")
        .cloned()
        .and_then(|value| serde_json::from_value::<Vec<AgentInput>>(value).ok())
        .or_else(|| {
            let text = event
                .payload
                .get("content")
                .and_then(|content| content.get("text").or_else(|| content.get("message")))
                .and_then(Value::as_str)
                .or_else(|| event.payload.get("text").and_then(Value::as_str))?
                .to_string();
            Some(vec![AgentInput::text(text)])
        });
    let Some(input) = input else {
        return Ok(None);
    };
    let input = provider_input(input)?;
    Ok(user_message_from_input(&input))
}

fn user_message_from_input(input: &RuntimeReplyInput) -> Option<CurrentProviderMessage> {
    let mut content = Vec::new();
    for part in &input.parts {
        match part {
            RuntimeReplyInputPart::Text { text, .. } if !text.trim().is_empty() => {
                content.push(CurrentProviderContent::Text(text.clone()));
            }
            RuntimeReplyInputPart::Image(image) => {
                content.push(CurrentProviderContent::Image {
                    uri: image.uri.clone(),
                    media_type: image.media_type.clone(),
                    provider_data: image.provider_data.clone(),
                    detail: image.detail,
                });
            }
            RuntimeReplyInputPart::Text { .. }
            | RuntimeReplyInputPart::Skill { .. }
            | RuntimeReplyInputPart::Mention { .. } => {}
        }
    }
    (!content.is_empty()).then(|| CurrentProviderMessage::user(content))
}

fn provider_event_payload(payload: &Value) -> &Value {
    payload.get("runtimeEvent").unwrap_or(payload)
}

fn text_from_payload(payload: &Value) -> Option<String> {
    text_from_message_value(provider_event_payload(payload))
}

fn text_from_message_value(value: &Value) -> Option<String> {
    if let Some(text) = value.as_str().filter(|text| !text.is_empty()) {
        return Some(text.to_string());
    }
    if let Some(text) = [
        "text",
        "delta",
        "content",
        "message",
        "outputText",
        "output_text",
    ]
    .iter()
    .find_map(|key| value.get(*key).and_then(Value::as_str))
    .filter(|text| !text.is_empty())
    {
        return Some(text.to_string());
    }
    if let Some(text) = value
        .get("content")
        .and_then(|content| text_from_message_value(content))
    {
        return Some(text);
    }
    for key in ["deltas", "messages", "items", "parts", "content"] {
        let Some(values) = value.get(key).and_then(Value::as_array) else {
            continue;
        };
        let text = values
            .iter()
            .filter_map(text_from_message_value)
            .collect::<String>();
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_protocol::{
        ImageDetail, ItemId, ItemKind, ItemStatus, SessionId, ThreadId, ThreadItemPayload,
        ToolArgument, ToolOutput, TurnId,
    };
    use serde_json::json;

    fn event(sequence: u64, event_type: &str, payload: Value) -> AgentEvent {
        AgentEvent {
            event_id: format!("evt-{sequence}"),
            sequence,
            session_id: "session-1".to_string(),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            event_type: event_type.to_string(),
            timestamp: "2026-07-12T00:00:00.000Z".to_string(),
            payload,
        }
    }

    fn stored_with_events(events: Vec<AgentEvent>) -> StoredSession {
        StoredSession {
            session: app_server_protocol::AgentSession {
                session_id: "session-1".to_string(),
                thread_id: "thread-1".to_string(),
                app_id: "agent-chat".to_string(),
                workspace_id: None,
                business_object_ref: None,
                status: app_server_protocol::AgentSessionStatus::Idle,
                created_at: "now".to_string(),
                updated_at: "now".to_string(),
            },
            turns: Vec::new(),
            turn_inputs: HashMap::new(),
            turn_runtime_options: HashMap::new(),
            events,
            output_blobs: HashMap::new(),
        }
    }

    fn turn_event(sequence: u64, turn_id: &str, event_type: &str, payload: Value) -> AgentEvent {
        let mut event = event(sequence, event_type, payload);
        event.turn_id = Some(turn_id.to_string());
        event
    }

    fn route_options(provider: &str, model: &str) -> app_server_protocol::RuntimeOptions {
        app_server_protocol::RuntimeOptions {
            runtime_request: Some(app_server_protocol::RuntimeRequest {
                metadata: Some(json!({
                    "agentControlRoute": {
                        "schemaVersion": 2,
                        "providerPreference": provider,
                        "modelPreference": model
                    }
                })),
                ..app_server_protocol::RuntimeRequest::default()
            }),
            ..app_server_protocol::RuntimeOptions::default()
        }
    }

    fn routing_decision(sequence: u64, turn_id: &str, provider: &str, model: &str) -> AgentEvent {
        turn_event(
            sequence,
            turn_id,
            "routing.decision.made",
            json!({
                "selectedProvider": provider,
                "selectedModel": model
            }),
        )
    }

    fn message_text(message: &CurrentProviderMessage) -> &str {
        let [CurrentProviderContent::Text(text)] = &message.content[..] else {
            panic!("expected one text content part");
        };
        text
    }

    #[test]
    fn replacement_history_is_used_before_bounded_tail() {
        let mut tail = event(
            3,
            "message.created",
            json!({"input": [{"type": "text", "text": "tail"}]}),
        );
        tail.turn_id = Some("turn-tail".to_string());
        let compaction = event(
            2,
            "context.compaction.completed",
            json!({
                "tailStartTurnId": "turn-tail",
                "artifact": {
                    "windowNumber": 1,
                    "firstWindowId": "ctx_window_session_1",
                    "windowId": "ctx_window_session_1",
                    "replacementHistory": [{
                        "role": "assistant",
                        "content": [{"type": "text", "text": "remember this"}]
                    }]
                }
            }),
        );
        let stored = stored_with_events(vec![
            event(
                1,
                "message.created",
                json!({"input": [{"type": "text", "text": "old"}]}),
            ),
            compaction,
            tail,
        ]);

        let (replacement, events) = provider_history_source(&stored);
        let replacement = replacement.expect("replacement history");
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &replacement[0].content[..],
            [CurrentProviderContent::Text(text)] if text == "remember this"
        ));
    }

    #[test]
    fn legacy_compaction_without_replacement_history_rebuilds_full_history() {
        let compaction = event(
            2,
            "context.compaction.completed",
            json!({"tailStartTurnId": "turn-1", "artifact": {"contextEpoch": 1}}),
        );
        let stored = stored_with_events(vec![
            event(
                1,
                "message.created",
                json!({"input": [{"type": "text", "text": "old"}]}),
            ),
            compaction,
        ]);

        let (replacement, events) = provider_history_source(&stored);
        assert!(replacement.is_none());
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn thread_goal_continuation_is_restored_for_future_turns_but_not_duplicated_in_its_turn() {
        let stored = stored_with_events(vec![event(
            1,
            super::super::turn_input_events::THREAD_GOAL_CONTINUATION_EVENT_TYPE,
            json!({
                "visibility": "agent_only",
                "source": "thread_goal",
                "input": [{"type": "text", "text": "continue the active goal"}]
            }),
        )]);

        let current = provider_history_excluding_current_turn_input(&stored, None, "turn-1")
            .expect("current continuation history");
        assert!(current.is_empty());

        let future = provider_history_excluding_current_turn_input(&stored, None, "turn-2")
            .expect("future continuation history");
        assert_eq!(future.len(), 1);
        assert_eq!(future[0].role, CurrentProviderRole::User);
        assert!(matches!(
            &future[0].content[..],
            [CurrentProviderContent::Text(text)] if text == "continue the active goal"
        ));
    }

    #[test]
    fn completed_route_switch_appends_codex_developer_context_after_durable_history() {
        let mut stored = stored_with_events(vec![
            turn_event(
                1,
                "turn-1",
                "message.created",
                json!({"input": [{"type": "text", "text": "old input"}]}),
            ),
            routing_decision(2, "turn-1", "provider<&", "model-a>"),
            turn_event(3, "turn-1", "turn.completed", json!({})),
        ]);
        stored.turn_runtime_options.insert(
            "turn-2".to_string(),
            route_options("provider-b>", "model<&b"),
        );

        let history = provider_history_excluding_current_turn_input(&stored, None, "turn-2")
            .expect("provider history");

        assert_eq!(history.len(), 2);
        assert_eq!(history[0].role, CurrentProviderRole::User);
        assert_eq!(history[1].role, CurrentProviderRole::Developer);
        let text = message_text(&history[1]);
        assert!(
            text.starts_with("<model_switch>\nThe user was previously using a different model.")
        );
        assert!(text.contains("<provider>provider&lt;&amp;</provider>"));
        assert!(text.contains("<model>model-a&gt;</model>"));
        assert!(text.contains("<provider>provider-b&gt;</provider>"));
        assert!(text.contains("<model>model&lt;&amp;b</model>"));
        assert!(text.ends_with("</model_switch>"));
    }

    #[test]
    fn same_route_does_not_inject_for_effort_or_service_tier_changes() {
        let mut stored = stored_with_events(vec![
            routing_decision(1, "turn-1", "provider-a", "model-a"),
            turn_event(2, "turn-1", "turn.completed", json!({})),
        ]);
        let mut options = route_options("provider-a", "model-a");
        let request = options.runtime_request_mut();
        request.reasoning_effort = Some("high".to_string());
        request.service_tier = Some("fast".to_string());
        stored
            .turn_runtime_options
            .insert("turn-2".to_string(), options);

        let history = provider_history_excluding_current_turn_input(&stored, None, "turn-2")
            .expect("provider history");

        assert!(history.is_empty());
    }

    #[test]
    fn failed_turn_route_never_becomes_previous_model_truth() {
        let mut stored = stored_with_events(vec![
            routing_decision(1, "turn-1", "provider-a", "model-a"),
            turn_event(2, "turn-1", "turn.completed", json!({})),
            routing_decision(3, "turn-2", "provider-b", "model-b"),
            turn_event(4, "turn-2", "turn.failed", json!({})),
        ]);
        stored
            .turn_runtime_options
            .insert("turn-3".to_string(), route_options("provider-a", "model-a"));

        let history = provider_history_excluding_current_turn_input(&stored, None, "turn-3")
            .expect("provider history");

        assert!(history.is_empty());
    }

    #[test]
    fn typed_history_retargets_model_switch_for_each_actual_route_attempt() {
        let stored = stored_with_events(vec![
            routing_decision(1, "turn-1", "provider-a", "model-a"),
            turn_event(2, "turn-1", "turn.completed", json!({})),
        ]);
        let history = provider_turn_history_excluding_current_turn_input(&stored, None, "turn-2")
            .expect("typed provider history");

        assert!(history
            .messages_for_route("provider-a", "model-a")
            .is_empty());
        let rerouted = history.messages_for_route("provider-b", "model-b");
        assert_eq!(rerouted.len(), 1);
        let text = message_text(&rerouted[0]);
        assert!(text.contains("<previous_route>"));
        assert!(text.contains("<provider>provider-a</provider>"));
        assert!(text.contains("<current_route>"));
        assert!(text.contains("<provider>provider-b</provider>"));
        assert!(text.contains("<model>model-b</model>"));
    }

    #[test]
    fn model_switch_context_is_not_repeated_after_switched_turn_completes() {
        let mut stored = stored_with_events(vec![
            routing_decision(1, "turn-1", "provider-a", "model-a"),
            turn_event(2, "turn-1", "turn.completed", json!({})),
        ]);
        stored
            .turn_runtime_options
            .insert("turn-2".to_string(), route_options("provider-b", "model-b"));
        assert_eq!(
            provider_history_excluding_current_turn_input(&stored, None, "turn-2")
                .expect("switch history")
                .len(),
            1
        );

        stored
            .events
            .push(routing_decision(3, "turn-2", "provider-b", "model-b"));
        stored
            .events
            .push(turn_event(4, "turn-2", "turn.completed", json!({})));
        stored
            .turn_runtime_options
            .insert("turn-3".to_string(), route_options("provider-b", "model-b"));

        let history = provider_history_excluding_current_turn_input(&stored, None, "turn-3")
            .expect("post-switch history");
        assert!(history.is_empty());
    }

    #[test]
    fn first_turn_without_completed_route_does_not_inject_model_switch_context() {
        let mut stored = stored_with_events(Vec::new());
        stored
            .turn_runtime_options
            .insert("turn-1".to_string(), route_options("provider-a", "model-a"));

        let history = provider_history_excluding_current_turn_input(&stored, None, "turn-1")
            .expect("first turn history");

        assert!(history.is_empty());
    }

    fn canonical_tool_event(
        sequence: u64,
        event_type: &str,
        status: ItemStatus,
        call_id: &str,
        name: &str,
        arguments: Value,
        output: Option<ToolOutput>,
    ) -> AgentEvent {
        let arguments = arguments
            .as_object()
            .expect("canonical tool arguments object")
            .iter()
            .map(|(name, value)| ToolArgument {
                name: name.clone(),
                value: value
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| value.to_string()),
            })
            .collect();
        let item = ThreadItem {
            session_id: SessionId::new("session-1"),
            thread_id: ThreadId::new("thread-1"),
            turn_id: TurnId::new("turn-1"),
            item_id: ItemId::new(call_id),
            sequence,
            ordinal: 1,
            created_at_ms: 1,
            updated_at_ms: 2,
            completed_at_ms: status.is_terminal().then_some(2),
            kind: ItemKind::Tool,
            status,
            payload: ThreadItemPayload::Tool {
                call_id: call_id.to_string(),
                name: name.to_string(),
                arguments,
                output,
            },
            metadata: Value::Null,
        };
        event(sequence, event_type, json!({ "item": item }))
    }

    #[test]
    fn canonical_history_preserves_tool_call_result_and_order() {
        let arguments = json!({ "path": "README.md" });
        let messages = messages_from_events(&[
            event(
                1,
                "message.created",
                json!({ "input": [{ "type": "text", "text": "read it" }] }),
            ),
            canonical_tool_event(
                2,
                "item.started",
                ItemStatus::InProgress,
                "call-read",
                "Read",
                arguments.clone(),
                None,
            ),
            canonical_tool_event(
                3,
                "item.completed",
                ItemStatus::Completed,
                "call-read",
                "Read",
                arguments,
                Some(ToolOutput {
                    text: Some("contents".to_string()),
                    ..ToolOutput::default()
                }),
            ),
            event(4, "message.delta", json!({ "text": "Done." })),
        ]);

        assert_eq!(messages.len(), 4);
        assert!(matches!(
            &messages[1].content[..],
            [CurrentProviderContent::ToolCall(call)]
                if call.id == "call-read"
                    && call.name == "Read"
                    && call.arguments == json!({ "path": "README.md" })
        ));
        assert!(matches!(
            &messages[2].content[..],
            [CurrentProviderContent::ToolResult(result)]
                if result.call_id == "call-read" && result.success && result.output == "contents"
        ));
        assert!(matches!(
            &messages[3].content[..],
            [CurrentProviderContent::Text(text)] if text == "Done."
        ));
    }

    #[test]
    fn canonical_history_preserves_failed_tool_result() {
        let messages = messages_from_events(&[canonical_tool_event(
            1,
            "item.completed",
            ItemStatus::Failed,
            "call-failed",
            "Read",
            json!({ "path": "missing.md" }),
            Some(ToolOutput {
                text: Some("read failed".to_string()),
                error: Some("not found".to_string()),
                ..ToolOutput::default()
            }),
        )]);

        assert!(matches!(
            &messages[0].content[..],
            [CurrentProviderContent::ToolResult(result)]
                if result.call_id == "call-failed"
                    && !result.success
                    && result.output == "read failed"
                    && result.error.as_deref() == Some("not found")
        ));
    }

    #[test]
    fn canonical_history_merges_completed_full_text_without_duplicate_delta() {
        let messages = messages_from_events(&[
            event(
                1,
                "message.delta",
                json!({"itemId": "agent-1", "text": "Hello "}),
            ),
            event(
                2,
                "message.completed",
                json!({"itemId": "agent-1", "text": "Hello world", "status": "completed"}),
            ),
        ]);

        assert_eq!(messages.len(), 1);
        let text = messages[0]
            .content
            .iter()
            .filter_map(|content| match content {
                CurrentProviderContent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn canonical_history_reads_message_delta_batch_parts() {
        let messages = messages_from_events(&[event(
            1,
            "message.delta_batch",
            json!({
                "itemId": "agent-1",
                "deltas": [
                    {"text": "batched "},
                    {"delta": "answer"}
                ]
            }),
        )]);

        assert!(matches!(
            &messages[0].content[..],
            [CurrentProviderContent::Text(text)] if text == "batched answer"
        ));
    }

    #[test]
    fn canonical_history_deduplicates_turn_wide_completed_snapshot() {
        let messages = messages_from_events(&[
            event(
                1,
                "message.delta",
                json!({"itemId": "commentary-1", "text": "inspect "}),
            ),
            event(
                2,
                "message.delta",
                json!({"itemId": "final-1", "text": "done"}),
            ),
            event(
                3,
                "message.completed",
                json!({"itemId": "final-1", "text": "inspect done", "status": "completed"}),
            ),
        ]);

        let text = messages[0]
            .content
            .iter()
            .filter_map(|content| match content {
                CurrentProviderContent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(text, "inspect done");
    }

    #[test]
    fn canonical_history_prefers_bounded_preview_over_persisted_output_ref() {
        let messages = messages_from_events(&[canonical_tool_event(
            1,
            "item.completed",
            ItemStatus::Completed,
            "call-large",
            "Read",
            json!({ "path": "large.txt" }),
            Some(ToolOutput {
                text: Some("preview".to_string()),
                truncated: true,
                output_ref: Some("output://large".to_string()),
                ..ToolOutput::default()
            }),
        )]);

        assert!(matches!(
            &messages[0].content[..],
            [CurrentProviderContent::ToolResult(result)]
                if result.call_id == "call-large" && result.output == "preview"
        ));
    }

    #[test]
    fn canonical_history_bounds_unoffloaded_inline_tool_output() {
        let messages = messages_from_events(&[canonical_tool_event(
            1,
            "item.completed",
            ItemStatus::Completed,
            "call-inline",
            "Read",
            json!({ "path": "large.txt" }),
            Some(ToolOutput {
                text: Some("x".repeat(PROVIDER_TOOL_OUTPUT_MAX_BYTES * 2)),
                ..ToolOutput::default()
            }),
        )]);

        let CurrentProviderContent::ToolResult(result) = &messages[0].content[0] else {
            panic!("expected tool result");
        };
        assert!(result.output.len() < PROVIDER_TOOL_OUTPUT_MAX_BYTES * 2);
        assert!(result.output.contains("Warning: truncated output"));
    }

    #[test]
    fn canonical_history_ignores_outer_event_output_ref() {
        let mut event = canonical_tool_event(
            1,
            "item.completed",
            ItemStatus::Completed,
            "call-large",
            "Read",
            json!({ "path": "large.txt" }),
            Some(ToolOutput {
                text: Some("preview".to_string()),
                ..ToolOutput::default()
            }),
        );
        event.payload["outputRef"] = json!("output://outer");

        let messages = messages_from_events(&[event]);

        assert!(matches!(
            &messages[0].content[..],
            [CurrentProviderContent::ToolResult(result)] if result.output == "preview"
        ));
    }

    #[test]
    fn raw_tool_events_do_not_enter_provider_history() {
        let messages = messages_from_events(&[
            event(
                1,
                "message.created",
                json!({ "input": [{ "type": "text", "text": "read it" }] }),
            ),
            event(
                2,
                "reasoning.delta",
                json!({ "runtimeEvent": { "text": "inspect" } }),
            ),
            event(
                3,
                "message.delta",
                json!({ "runtimeEvent": { "text": "I will read it." } }),
            ),
            event(
                4,
                "tool.started",
                json!({ "runtimeEvent": { "tool_id": "call-read", "tool_name": "Read", "arguments": "{\"path\":\"README.md\"}" } }),
            ),
            event(
                5,
                "tool.result",
                json!({ "runtimeEvent": { "tool_id": "call-read", "tool_name": "Read", "result": { "success": true, "output": "contents" } } }),
            ),
            event(
                6,
                "tool.failed",
                json!({ "toolId": "call-failed", "toolName": "Read", "error": "failed" }),
            ),
            event(
                7,
                "tool.completed",
                json!({ "toolId": "call-completed", "toolName": "Read", "output": "done" }),
            ),
            event(
                8,
                "message.delta",
                json!({ "runtimeEvent": { "text": "Done." } }),
            ),
        ]);

        assert_eq!(messages.len(), 2);
        assert!(matches!(
            &messages[0].content[..],
            [CurrentProviderContent::Text(text)] if text == "read it"
        ));
        assert!(matches!(
            &messages[1].content[..],
            [CurrentProviderContent::Reasoning(reasoning), CurrentProviderContent::Text(before), CurrentProviderContent::Text(after)]
                if reasoning == "inspect" && before == "I will read it." && after == "Done."
        ));
    }

    #[test]
    fn inline_image_input_is_preserved_for_current_provider() {
        let root = tempfile::tempdir().expect("sidecar root");
        let store =
            super::super::sidecar_store::SidecarStore::new(root.path()).expect("sidecar store");
        let input = vec![
            AgentInput::text("describe it"),
            AgentInput::Image {
                uri: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==".to_string(),
                detail: Some(ImageDetail::High),
            },
        ];
        let reply = RuntimeReplyInput::try_from_user_parts(input, |media| {
            super::super::input_media::resolve_runtime_input_media(media, Some(&store), "session-1")
        })
        .expect("provider input");
        let images = reply.images().collect::<Vec<_>>();
        assert_eq!(images.len(), 1);
        assert!(images[0].uri.starts_with("sidecar://media/"));
        assert_eq!(images[0].media_type, "image/png");
        assert_eq!(
            images[0].provider_data.as_deref(),
            Some("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==")
        );
        assert_eq!(images[0].detail, Some(ImageDetail::High));
    }

    #[test]
    fn persisted_image_history_is_hydrated_without_rewriting_canonical_reference() {
        const PNG_DATA_URL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";
        let root = tempfile::tempdir().expect("sidecar root");
        let store =
            super::super::sidecar_store::SidecarStore::new(root.path()).expect("sidecar store");
        let input = super::super::input_media::prepare_runtime_input(
            vec![
                AgentInput::text("describe it"),
                AgentInput::Image {
                    uri: PNG_DATA_URL.to_string(),
                    detail: None,
                },
            ],
            Some(&store),
            "session-1",
        )
        .expect("prepare durable image input")
        .durable;
        assert!(!serde_json::to_string(&input)
            .expect("serialize durable image input")
            .contains("base64,"));
        let messages = messages_from_events_with_provider_input(
            &[event(1, "message.created", json!({ "input": input }))],
            |input| {
                RuntimeReplyInput::try_from_user_parts(input, |media| {
                    super::super::input_media::resolve_runtime_input_media(
                        media,
                        Some(&store),
                        "session-1",
                    )
                })
                .map_err(|error| error.to_string())
            },
        )
        .expect("provider history");

        assert!(matches!(
            &messages[0].content[..],
            [CurrentProviderContent::Text(text), CurrentProviderContent::Image {
                uri,
                media_type,
                provider_data: Some(provider_data),
                detail: None,
            }] if text == "describe it"
                && uri.starts_with("sidecar://media/")
                && media_type == "image/png"
                && provider_data == PNG_DATA_URL
        ));
    }
}
