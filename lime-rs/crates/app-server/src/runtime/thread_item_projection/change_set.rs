//! Typed, batch-local history changes produced by the item materializer.
//!
//! This follows Codex's `ThreadHistoryChangeAccumulator` rule: repeated snapshots
//! for one `(thread, turn, item)` replace the latest value while preserving the
//! first-change order.  The accumulator deliberately contains no storage logic;
//! SQLite/read-model ownership belongs to the S2d/S2e slices.

use agent_protocol::{ItemId, ThreadHistoryChangeSet, ThreadItem, Turn};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::runtime) enum MaterializationError {
    EventIdentityCollision {
        event_id: String,
        previous_sequence: u64,
        sequence: u64,
    },
    SequenceCollision {
        sequence: u64,
        previous_event_id: String,
        event_id: String,
    },
}

impl std::fmt::Display for MaterializationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EventIdentityCollision {
                event_id,
                previous_sequence,
                sequence,
            } => write!(
                formatter,
                "event id {event_id} changed sequence from {previous_sequence} to {sequence}"
            ),
            Self::SequenceCollision {
                sequence,
                previous_event_id,
                event_id,
            } => write!(
                formatter,
                "sequence {sequence} has event ids {previous_event_id} and {event_id}"
            ),
        }
    }
}

impl std::error::Error for MaterializationError {}

#[derive(Debug, Default)]
pub(super) struct ChangeSetAccumulator {
    changed_items: Vec<Option<ThreadItem>>,
    item_indexes: HashMap<(String, String, ItemId), usize>,
    changed_turns: Vec<Option<Turn>>,
    changed_turn_sequences: Vec<Option<u64>>,
    turn_indexes: HashMap<String, usize>,
    removed_item_ids: Vec<ItemId>,
    removed_item_indexes: HashMap<ItemId, usize>,
    removed_turn_ids: Vec<String>,
    removed_turn_indexes: HashMap<String, usize>,
    rollback_to_sequence: Option<u64>,
}

impl ChangeSetAccumulator {
    pub(super) fn items_for_turn(&self, turn_id: &str) -> Vec<ThreadItem> {
        self.changed_items
            .iter()
            .filter_map(Option::as_ref)
            .filter(|item| item.turn_id.as_str() == turn_id)
            .cloned()
            .collect()
    }

    pub(super) fn push_item(&mut self, item: ThreadItem) -> Option<ThreadItem> {
        let key = (
            item.thread_id.to_string(),
            item.turn_id.to_string(),
            item.item_id.clone(),
        );
        if self.removed_item_indexes.contains_key(&item.item_id) {
            return None;
        }
        if let Some(index) = self.item_indexes.get(&key).copied() {
            let previous = self.changed_items[index]
                .take()
                .expect("item index always points at a live item");
            let snapshot = merge_item_snapshot(previous, item);
            self.changed_items[index] = Some(snapshot.clone());
            Some(snapshot)
        } else {
            self.item_indexes.insert(key, self.changed_items.len());
            self.changed_items.push(Some(item.clone()));
            Some(item)
        }
    }

    pub(super) fn push_turn(&mut self, turn: Turn, sequence: u64) -> Option<Turn> {
        let key = turn.turn_id.to_string();
        if self.removed_turn_indexes.contains_key(&key) {
            return None;
        }
        if let Some(index) = self.turn_indexes.get(&key).copied() {
            let previous = self.changed_turns[index]
                .take()
                .expect("turn index always points at a live turn");
            let snapshot = merge_turn_snapshot(previous, turn);
            self.changed_turns[index] = Some(snapshot.clone());
            self.changed_turn_sequences[index] = Some(sequence);
            Some(snapshot)
        } else {
            self.turn_indexes.insert(key, self.changed_turns.len());
            self.changed_turns.push(Some(turn.clone()));
            self.changed_turn_sequences.push(Some(sequence));
            Some(turn)
        }
    }

    pub(super) fn remove_item(&mut self, item_id: ItemId) {
        if !self.removed_item_indexes.contains_key(&item_id) {
            self.removed_item_indexes
                .insert(item_id.clone(), self.removed_item_ids.len());
            self.removed_item_ids.push(item_id.clone());
        }
        let indexes = self
            .item_indexes
            .iter()
            .filter(|(_, index)| {
                self.changed_items[**index]
                    .as_ref()
                    .is_some_and(|item| item.item_id == item_id)
            })
            .map(|(key, index)| (key.clone(), *index))
            .collect::<Vec<_>>();
        for (key, index) in indexes {
            self.item_indexes.remove(&key);
            self.changed_items[index] = None;
        }
    }

    pub(super) fn rollback(&mut self, sequence: u64) {
        self.rollback_to_sequence = Some(
            self.rollback_to_sequence
                .map_or(sequence, |previous| previous.min(sequence)),
        );
        let item_ids = self
            .changed_items
            .iter()
            .filter_map(|item| item.as_ref())
            .filter(|item| item.sequence > sequence)
            .map(|item| item.item_id.clone())
            .collect::<Vec<_>>();
        for item_id in item_ids {
            self.remove_item(item_id);
        }
        let turn_ids = self
            .changed_turns
            .iter()
            .zip(&self.changed_turn_sequences)
            .filter_map(|(turn, turn_sequence)| {
                let turn = turn.as_ref()?;
                (turn_sequence
                    .as_ref()
                    .is_some_and(|value| *value > sequence))
                .then(|| turn.turn_id.to_string())
            })
            .collect::<Vec<_>>();
        for turn_id in turn_ids {
            self.remove_turn(turn_id);
        }
    }

    pub(super) fn finish(self, sequence: u64) -> ThreadHistoryChangeSet {
        ThreadHistoryChangeSet {
            sequence,
            changed_turns: self.changed_turns.into_iter().flatten().collect(),
            changed_items: self.changed_items.into_iter().flatten().collect(),
            removed_item_ids: self.removed_item_ids,
            removed_turn_ids: self.removed_turn_ids.into_iter().map(Into::into).collect(),
            rollback_to_sequence: self.rollback_to_sequence,
        }
    }

    pub(super) fn remove_turn(&mut self, turn_id: String) {
        if !self.removed_turn_indexes.contains_key(&turn_id) {
            self.removed_turn_indexes
                .insert(turn_id.clone(), self.removed_turn_ids.len());
            self.removed_turn_ids.push(turn_id.clone());
        }
        if let Some(index) = self.turn_indexes.remove(&turn_id) {
            self.changed_turns[index] = None;
            self.changed_turn_sequences[index] = None;
        }
        let keys = self
            .item_indexes
            .keys()
            .filter(|(_, item_turn_id, _)| item_turn_id == &turn_id)
            .cloned()
            .collect::<Vec<_>>();
        for key in keys {
            if let Some(index) = self.item_indexes.remove(&key) {
                self.changed_items[index] = None;
            }
        }
    }
}

pub(in crate::runtime) fn merge_item_snapshot(
    previous: ThreadItem,
    mut next: ThreadItem,
) -> ThreadItem {
    let previous_terminal_interactions = previous.metadata.get("terminalInteractions").cloned();
    let source_event_type = next
        .metadata
        .get("source_event_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    next.created_at_ms = previous.created_at_ms;
    next.ordinal = previous.ordinal;
    if next.completed_at_ms.is_none() {
        next.completed_at_ms = previous.completed_at_ms;
    }
    if previous.status.is_terminal() && !next.status.is_terminal() {
        next.status = previous.status;
    }
    next.payload = merge_payload(previous.payload, next.payload, source_event_type.as_deref());
    if matches!(
        next.payload,
        agent_protocol::ThreadItemPayload::Command { .. }
    ) {
        if let Some(interactions) = previous_terminal_interactions {
            if let Some(metadata) = next.metadata.as_object_mut() {
                let next_interactions = metadata
                    .get("terminalInteractions")
                    .and_then(serde_json::Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let mut merged = interactions.as_array().cloned().unwrap_or_default();
                merged.extend(next_interactions);
                if merged.len() > 20 {
                    merged.drain(..merged.len() - 20);
                }
                metadata.insert(
                    "terminalInteractions".to_string(),
                    serde_json::Value::Array(merged),
                );
            }
        }
    }
    next
}

fn merge_payload(
    previous: agent_protocol::ThreadItemPayload,
    next: agent_protocol::ThreadItemPayload,
    source_event_type: Option<&str>,
) -> agent_protocol::ThreadItemPayload {
    use agent_protocol::ThreadItemPayload;

    match (previous, next) {
        (
            ThreadItemPayload::UserMessage {
                client_id: previous_client_id,
                ..
            },
            ThreadItemPayload::UserMessage { content, client_id },
        ) => ThreadItemPayload::UserMessage {
            content,
            client_id: client_id.or(previous_client_id),
        },
        (
            ThreadItemPayload::AgentMessage {
                text: previous,
                content_parts: previous_parts,
                ..
            },
            ThreadItemPayload::AgentMessage {
                text,
                phase,
                content_parts,
            },
        ) => ThreadItemPayload::AgentMessage {
            text: merge_stream_text(previous, text),
            phase,
            content_parts: merge_message_content_parts(previous_parts, content_parts),
        },
        (
            ThreadItemPayload::Plan {
                text: previous_text,
                revision_id: previous_revision_id,
                source: previous_source,
                plan: previous_plan,
                explanation: previous_explanation,
                tool_call_id: previous_tool_call_id,
                source_item_id: previous_source_item_id,
            },
            ThreadItemPayload::Plan {
                text,
                revision_id,
                source,
                plan,
                explanation,
                tool_call_id,
                source_item_id,
            },
        ) => ThreadItemPayload::Plan {
            text: if text.is_empty() { previous_text } else { text },
            revision_id: prefer_string(previous_revision_id, revision_id, ""),
            source: source.or(previous_source),
            plan: if plan.is_empty() { previous_plan } else { plan },
            explanation: explanation.or(previous_explanation),
            tool_call_id: tool_call_id.or(previous_tool_call_id),
            source_item_id: source_item_id.or(previous_source_item_id),
        },
        (
            ThreadItemPayload::Reasoning {
                mut summary,
                mut content,
            },
            ThreadItemPayload::Reasoning {
                summary: next_summary,
                content: next_content,
            },
        ) => match source_event_type {
            Some("reasoning.final" | "item.completed" | "item.updated") => {
                ThreadItemPayload::Reasoning {
                    summary: if next_summary.is_empty() {
                        summary
                    } else {
                        next_summary
                    },
                    content: if next_content.is_empty() {
                        content
                    } else {
                        next_content
                    },
                }
            }
            _ => {
                summary.extend(next_summary);
                content.extend(next_content);
                ThreadItemPayload::Reasoning { summary, content }
            }
        },
        (
            ThreadItemPayload::Tool {
                call_id: previous_call_id,
                name: previous_name,
                arguments: previous_arguments,
                output: previous_output,
            },
            ThreadItemPayload::Tool {
                call_id,
                name,
                arguments,
                output,
            },
        ) => ThreadItemPayload::Tool {
            call_id: if call_id.is_empty() {
                previous_call_id
            } else {
                call_id
            },
            name: if name == "tool" { previous_name } else { name },
            arguments: if arguments.is_empty() {
                previous_arguments
            } else {
                arguments
            },
            output: merge_tool_output(previous_output, output),
        },
        (
            ThreadItemPayload::McpToolCall {
                call_id: previous_call_id,
                server_name: previous_server_name,
                tool_name: previous_tool_name,
                mcp_app_resource_uri: previous_mcp_app_resource_uri,
                plugin_id: previous_plugin_id,
                arguments: previous_arguments,
                output: previous_output,
            },
            ThreadItemPayload::McpToolCall {
                call_id,
                server_name,
                tool_name,
                mcp_app_resource_uri,
                plugin_id,
                arguments,
                output,
            },
        ) => ThreadItemPayload::McpToolCall {
            call_id: prefer_string(previous_call_id, call_id, ""),
            server_name: prefer_string(previous_server_name, server_name, "unknown"),
            tool_name: prefer_string(previous_tool_name, tool_name, "tool"),
            mcp_app_resource_uri: mcp_app_resource_uri.or(previous_mcp_app_resource_uri),
            plugin_id: plugin_id.or(previous_plugin_id),
            arguments: if arguments.is_empty() {
                previous_arguments
            } else {
                arguments
            },
            output: merge_tool_output(previous_output, output),
        },
        (
            ThreadItemPayload::File {
                changes: previous_changes,
                status: _,
            },
            ThreadItemPayload::File { changes, status },
        ) => ThreadItemPayload::File {
            changes: if changes.is_empty() {
                previous_changes
            } else {
                changes
            },
            status,
        },
        (
            ThreadItemPayload::CollabAgentToolCall {
                call_id: previous_call_id,
                operation: previous_operation,
                target_thread_id: previous_target,
                message: previous_message,
                output: previous_output,
                agent_states: mut previous_states,
            },
            ThreadItemPayload::CollabAgentToolCall {
                call_id,
                operation,
                target_thread_id,
                message,
                output,
                agent_states,
            },
        ) => {
            previous_states.extend(agent_states);
            ThreadItemPayload::CollabAgentToolCall {
                call_id: prefer_string(previous_call_id, call_id, ""),
                operation: if target_thread_id.is_none()
                    && message.is_none()
                    && output.is_none()
                    && previous_states.is_empty()
                {
                    previous_operation
                } else {
                    operation
                },
                target_thread_id: target_thread_id.or(previous_target),
                message: message.or(previous_message),
                output: merge_tool_output(previous_output, output),
                agent_states: previous_states,
            }
        }
        (
            ThreadItemPayload::Approval {
                request_id: previous_request_id,
                action: previous_action,
                scope: previous_scope,
                available_decisions: previous_available_decisions,
                decision: previous_decision,
                requested_at_ms: previous_requested_at_ms,
                resolved_at_ms: previous_resolved_at_ms,
                reason_code: previous_reason_code,
                expires_at_ms: previous_expires_at_ms,
            },
            ThreadItemPayload::Approval {
                request_id,
                action,
                scope,
                available_decisions,
                decision,
                requested_at_ms,
                resolved_at_ms,
                reason_code,
                expires_at_ms,
            },
        ) => ThreadItemPayload::Approval {
            request_id: prefer_string(previous_request_id, request_id, ""),
            action: if action.description.is_empty() {
                previous_action
            } else {
                action
            },
            scope: if matches!(scope, agent_protocol::ApprovalScope::Once) {
                previous_scope
            } else {
                scope
            },
            available_decisions: if available_decisions.is_empty() {
                previous_available_decisions
            } else {
                available_decisions
            },
            decision: if resolved_at_ms.is_some() {
                decision
            } else {
                decision.or(previous_decision)
            },
            requested_at_ms: requested_at_ms.or(previous_requested_at_ms),
            resolved_at_ms: resolved_at_ms.or(previous_resolved_at_ms),
            reason_code: reason_code.or(previous_reason_code),
            expires_at_ms: expires_at_ms.or(previous_expires_at_ms),
        },
        (
            ThreadItemPayload::Command {
                command: previous_command,
                cwd: previous_cwd,
                output: previous_output,
                exit_code: previous_exit_code,
            },
            ThreadItemPayload::Command {
                command,
                cwd,
                output,
                exit_code,
            },
        ) => ThreadItemPayload::Command {
            command: prefer_string(previous_command, command, "command"),
            cwd: cwd.or(previous_cwd),
            output: output.or(previous_output),
            exit_code: exit_code.or(previous_exit_code),
        },
        (_, next) => next,
    }
}

fn merge_message_content_parts(
    mut previous: Vec<agent_protocol::MessageContentPart>,
    next: Vec<agent_protocol::MessageContentPart>,
) -> Vec<agent_protocol::MessageContentPart> {
    for part in next {
        match &part {
            agent_protocol::MessageContentPart::Text { text } => {
                if previous.iter().any(|part| {
                    matches!(
                        part,
                        agent_protocol::MessageContentPart::Text {
                            text: previous_text,
                        } if previous_text == text
                    )
                }) {
                    continue;
                }
                if let Some(previous_text) = previous.iter_mut().find_map(|part| match part {
                    agent_protocol::MessageContentPart::Text {
                        text: previous_text,
                    } if text.starts_with(previous_text.as_str())
                        || previous_text.starts_with(text.as_str()) =>
                    {
                        Some(previous_text)
                    }
                    _ => None,
                }) {
                    *previous_text = merge_stream_text(previous_text.clone(), text.clone());
                    continue;
                }
                if let Some(agent_protocol::MessageContentPart::Text {
                    text: previous_text,
                }) = previous.last_mut()
                {
                    *previous_text = merge_stream_text(previous_text.clone(), text.clone());
                    continue;
                }
            }
            agent_protocol::MessageContentPart::Media { reference, .. } => {
                if let Some(index) = previous.iter().position(|part| {
                    matches!(
                        part,
                        agent_protocol::MessageContentPart::Media {
                            reference: previous_reference,
                            ..
                        } if previous_reference.uri == reference.uri
                    )
                }) {
                    previous[index] = part;
                    continue;
                }
            }
        }
        previous.push(part);
    }
    previous
}

fn prefer_string(previous: String, next: String, placeholder: &str) -> String {
    if next.is_empty() || next == placeholder {
        previous
    } else {
        next
    }
}

fn merge_tool_output(
    previous: Option<agent_protocol::ToolOutput>,
    next: Option<agent_protocol::ToolOutput>,
) -> Option<agent_protocol::ToolOutput> {
    match (previous, next) {
        (Some(previous), Some(mut next)) => {
            next.text = next.text.or(previous.text);
            next.structured_content = next.structured_content.or(previous.structured_content);
            next.error = next.error.or(previous.error);
            next.duration_ms = next.duration_ms.or(previous.duration_ms);
            next.truncated |= previous.truncated;
            next.output_ref = next.output_ref.or(previous.output_ref);
            Some(next)
        }
        (previous, next) => next.or(previous),
    }
}

fn merge_stream_text(previous: String, next: String) -> String {
    if next.is_empty() || next == previous {
        previous
    } else if next.starts_with(&previous) {
        next
    } else {
        format!("{previous}{next}")
    }
}

pub(in crate::runtime) fn merge_turn_snapshot(previous: Turn, mut next: Turn) -> Turn {
    next.created_at_ms = previous.created_at_ms.min(next.created_at_ms);
    next.started_at_ms = next.started_at_ms.or(previous.started_at_ms);
    next.completed_at_ms = next.completed_at_ms.or(previous.completed_at_ms);
    next.duration_ms = next.duration_ms.or(previous.duration_ms);
    next.error = next.error.or(previous.error);
    if next.queue == Default::default() {
        next.queue = previous.queue;
    }
    if next.approval == Default::default() {
        next.approval = previous.approval;
    }
    if previous.status.is_terminal() && !next.status.is_terminal() {
        next.status = previous.status;
    }
    next
}
