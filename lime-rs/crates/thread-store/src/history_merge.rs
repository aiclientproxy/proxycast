use agent_protocol::{
    ItemId, MessageContentPart, ThreadHistoryChangeSet, ThreadItem, ThreadItemPayload, ToolOutput,
    Turn, TurnId,
};
use std::collections::HashMap;

pub(super) fn merge_item_snapshot(previous: ThreadItem, mut next: ThreadItem) -> ThreadItem {
    let source_event_type = next
        .metadata
        .get("source_event_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    next.created_at_ms = previous.created_at_ms;
    next.ordinal = previous.ordinal;
    next.completed_at_ms = next.completed_at_ms.or(previous.completed_at_ms);
    if previous.status.is_terminal() && !next.status.is_terminal() {
        next.status = previous.status;
    }
    next.payload = merge_payload(previous.payload, next.payload, source_event_type.as_deref());
    next
}

pub(super) fn merge_turn_snapshot(previous: Turn, mut next: Turn) -> Turn {
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
    if next.items.is_empty() {
        next.items = previous.items;
    }
    if previous.status.is_terminal() && !next.status.is_terminal() {
        next.status = previous.status;
    }
    next
}

fn merge_payload(
    previous: ThreadItemPayload,
    next: ThreadItemPayload,
    source_event_type: Option<&str>,
) -> ThreadItemPayload {
    use ThreadItemPayload::*;

    match (previous, next) {
        (
            UserMessage {
                client_id: old_id, ..
            },
            UserMessage { content, client_id },
        ) => UserMessage {
            content,
            client_id: client_id.or(old_id),
        },
        (
            AgentMessage {
                text: old,
                content_parts: old_parts,
                ..
            },
            AgentMessage {
                text,
                phase,
                content_parts,
            },
        ) => AgentMessage {
            text: merge_stream_text(old, text),
            phase,
            content_parts: merge_parts(old_parts, content_parts),
        },
        (
            Plan {
                text: old_text,
                revision_id: old_revision,
                source: old_source,
                plan: old_plan,
                explanation: old_explanation,
                tool_call_id: old_call,
                source_item_id: old_item,
            },
            Plan {
                text,
                revision_id,
                source,
                plan,
                explanation,
                tool_call_id,
                source_item_id,
            },
        ) => Plan {
            text: if text.is_empty() { old_text } else { text },
            revision_id: prefer_string(old_revision, revision_id, ""),
            source: source.or(old_source),
            plan: if plan.is_empty() { old_plan } else { plan },
            explanation: explanation.or(old_explanation),
            tool_call_id: tool_call_id.or(old_call),
            source_item_id: source_item_id.or(old_item),
        },
        (
            Reasoning {
                mut summary,
                mut content,
            },
            Reasoning {
                summary: next_summary,
                content: next_content,
            },
        ) => match source_event_type {
            Some("reasoning.final" | "item.completed" | "item.updated") => Reasoning {
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
            },
            _ => {
                summary.extend(next_summary);
                content.extend(next_content);
                Reasoning { summary, content }
            }
        },
        (
            Tool {
                call_id: old_call,
                name: old_name,
                arguments: old_args,
                output: old_output,
            },
            Tool {
                call_id,
                name,
                arguments,
                output,
            },
        ) => Tool {
            call_id: prefer_string(old_call, call_id, ""),
            name: prefer_string(old_name, name, "tool"),
            arguments: if arguments.is_empty() {
                old_args
            } else {
                arguments
            },
            output: merge_tool_output(old_output, output),
        },
        (
            McpToolCall {
                call_id: old_call,
                server_name: old_server,
                tool_name: old_tool,
                mcp_app_resource_uri: old_mcp_app_resource_uri,
                plugin_id: old_plugin_id,
                arguments: old_args,
                output: old_output,
            },
            McpToolCall {
                call_id,
                server_name,
                tool_name,
                mcp_app_resource_uri,
                plugin_id,
                arguments,
                output,
            },
        ) => McpToolCall {
            call_id: prefer_string(old_call, call_id, ""),
            server_name: prefer_string(old_server, server_name, "unknown"),
            tool_name: prefer_string(old_tool, tool_name, "tool"),
            mcp_app_resource_uri: mcp_app_resource_uri.or(old_mcp_app_resource_uri),
            plugin_id: plugin_id.or(old_plugin_id),
            arguments: if arguments.is_empty() {
                old_args
            } else {
                arguments
            },
            output: merge_tool_output(old_output, output),
        },
        (
            CollabAgentToolCall {
                call_id: old_call,
                operation: old_operation,
                target_thread_id: old_target,
                message: old_message,
                output: old_output,
                agent_states: mut old_states,
            },
            CollabAgentToolCall {
                call_id,
                operation,
                target_thread_id,
                message,
                output,
                agent_states,
            },
        ) => {
            old_states.extend(agent_states);
            CollabAgentToolCall {
                call_id: prefer_string(old_call, call_id, ""),
                operation: if target_thread_id.is_none()
                    && message.is_none()
                    && output.is_none()
                    && old_states.is_empty()
                {
                    old_operation
                } else {
                    operation
                },
                target_thread_id: target_thread_id.or(old_target),
                message: message.or(old_message),
                output: merge_tool_output(old_output, output),
                agent_states: old_states,
            }
        }
        (
            Approval {
                request_id: old_request,
                action: old_action,
                scope: old_scope,
                available_decisions: old_decisions,
                decision: old_decision,
                requested_at_ms: old_requested,
                resolved_at_ms: old_resolved,
                reason_code: old_reason,
                expires_at_ms: old_expires,
            },
            Approval {
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
        ) => Approval {
            request_id: prefer_string(old_request, request_id, ""),
            action: if action.description.is_empty() {
                old_action
            } else {
                action
            },
            scope: if matches!(scope, agent_protocol::ApprovalScope::Once) {
                old_scope
            } else {
                scope
            },
            available_decisions: if available_decisions.is_empty() {
                old_decisions
            } else {
                available_decisions
            },
            decision: if resolved_at_ms.is_some() {
                decision
            } else {
                decision.or(old_decision)
            },
            requested_at_ms: requested_at_ms.or(old_requested),
            resolved_at_ms: resolved_at_ms.or(old_resolved),
            reason_code: reason_code.or(old_reason),
            expires_at_ms: expires_at_ms.or(old_expires),
        },
        (
            Command {
                command: old_command,
                cwd: old_cwd,
                output: old_output,
                exit_code: old_exit,
            },
            Command {
                command,
                cwd,
                output,
                exit_code,
            },
        ) => Command {
            command: prefer_string(old_command, command, "command"),
            cwd: cwd.or(old_cwd),
            output: output.or(old_output),
            exit_code: exit_code.or(old_exit),
        },
        (_, next) => next,
    }
}

fn merge_parts(
    mut previous: Vec<MessageContentPart>,
    next: Vec<MessageContentPart>,
) -> Vec<MessageContentPart> {
    for part in next {
        match &part {
            MessageContentPart::Text { text } => {
                if previous.iter().any(
                    |part| matches!(part, MessageContentPart::Text { text: old } if old == text),
                ) {
                    continue;
                }
                if let Some(old) = previous.iter_mut().find_map(|part| match part {
                    MessageContentPart::Text { text: old }
                        if text.starts_with(old.as_str()) || old.starts_with(text.as_str()) =>
                    {
                        Some(old)
                    }
                    _ => None,
                }) {
                    *old = merge_stream_text(old.clone(), text.clone());
                    continue;
                }
                if let Some(MessageContentPart::Text { text: old }) = previous.last_mut() {
                    *old = merge_stream_text(old.clone(), text.clone());
                    continue;
                }
            }
            MessageContentPart::Media { reference, .. } => {
                if let Some(index) = previous.iter().position(|part| {
                    matches!(part, MessageContentPart::Media { reference: old, .. } if old.uri == reference.uri)
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

fn merge_tool_output(previous: Option<ToolOutput>, next: Option<ToolOutput>) -> Option<ToolOutput> {
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

/// Batch-local coalescer. Event lowering may emit several snapshots for one
/// identity; the durable builder owns the resulting merge, while this helper
/// only preserves first-change order within one batch.
#[derive(Default)]
pub(super) struct ChangeAccumulator {
    changed_items: Vec<ThreadItem>,
    item_indexes: HashMap<(TurnId, ItemId), usize>,
    changed_turns: Vec<Turn>,
    turn_indexes: HashMap<TurnId, usize>,
}

impl ChangeAccumulator {
    pub(super) fn push_item(&mut self, key: (TurnId, ItemId), item: ThreadItem) {
        if let Some(index) = self.item_indexes.get(&key).copied() {
            self.changed_items[index] = item;
        } else {
            self.item_indexes.insert(key, self.changed_items.len());
            self.changed_items.push(item);
        }
    }

    pub(super) fn push_turn(&mut self, turn: Turn) {
        if let Some(index) = self.turn_indexes.get(&turn.turn_id).copied() {
            self.changed_turns[index] = turn;
        } else {
            self.turn_indexes
                .insert(turn.turn_id.clone(), self.changed_turns.len());
            self.changed_turns.push(turn);
        }
    }

    pub(super) fn finish(self, sequence: u64) -> ThreadHistoryChangeSet {
        ThreadHistoryChangeSet {
            sequence,
            changed_turns: self.changed_turns,
            changed_items: self.changed_items,
            ..Default::default()
        }
    }
}
