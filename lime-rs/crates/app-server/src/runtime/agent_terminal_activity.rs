//! Durable direct-parent activity derived from canonical child terminal turns.

use super::*;
use agent_protocol::{ThreadId, ThreadItemPayload, ThreadTurnsView, Turn, TurnStatus};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thread_store::{
    AgentMailboxDeliveryMode, AgentMailboxDeliveryStatus, AgentMailboxMessage,
    AgentMailboxMessageKind, AgentMailboxResultStatus, AppendAgentMailboxMessageParams,
    ReadThreadParams, ThreadSpawnEdgeStatus,
};

const ERROR_NEXT_ACTION: &str = "This agent's turn failed. If you still need this agent, use the available collaboration tools to give it another task.";
const ERROR_TEXT_MAX_CHARS: usize = 4_000;

pub(in crate::runtime) struct RuntimeSessionTerminalActivity {
    pub recipient_session_id: String,
    pub input: agent_runtime::session_loop::RuntimeSessionInterAgentInput,
}

pub(in crate::runtime) fn publish_terminal_agent_activities(
    session_loops: &agent_runtime::session_loop::RuntimeSessionRegistry,
    activities: Vec<RuntimeSessionTerminalActivity>,
) {
    if activities.is_empty() {
        return;
    }
    let session_loops = session_loops.clone();
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        tracing::debug!(
            activity_count = activities.len(),
            "terminal mailbox activity remains durable until a session boundary observes it"
        );
        return;
    };
    runtime.spawn(async move {
        for activity in activities {
            if let Err(error) = session_loops
                .notify_inter_agent_communication(&activity.recipient_session_id, activity.input)
                .await
            {
                tracing::warn!(
                    session_id = activity.recipient_session_id,
                    error = %error,
                    "failed to publish durable terminal mailbox activity"
                );
            }
        }
    });
}

impl ProjectionStore {
    pub(in crate::runtime) fn terminal_agent_result_required_sync(
        &self,
        child_thread_id: &ThreadId,
        events: &[app_server_protocol::AgentEvent],
    ) -> Result<bool, String> {
        if terminal_result_turn_ids(events).is_empty() {
            return Ok(false);
        }
        self.read_thread_spawn_parent_sync(child_thread_id.clone())
            .map(|parent| parent.is_some_and(|parent| parent.status == ThreadSpawnEdgeStatus::Open))
            .map_err(|error| error.to_string())
    }

    pub(in crate::runtime) fn append_terminal_agent_results_sync(
        &self,
        child_thread_id: &ThreadId,
        events: &[app_server_protocol::AgentEvent],
    ) -> Result<Vec<RuntimeSessionTerminalActivity>, String> {
        let turn_ids = terminal_result_turn_ids(events);
        if turn_ids.is_empty() {
            return Ok(Vec::new());
        }
        let Some(parent) = self
            .read_thread_spawn_parent_sync(child_thread_id.clone())
            .map_err(|error| error.to_string())?
        else {
            return Ok(Vec::new());
        };
        if parent.status != ThreadSpawnEdgeStatus::Open {
            return Ok(Vec::new());
        }
        let child_identity = self
            .read_agent_identity_sync(child_thread_id.clone())
            .map_err(|error| error.to_string())?
            .ok_or_else(|| format!("spawned child {child_thread_id} has no durable identity"))?;
        let parent_identity = self
            .read_agent_identity_sync(parent.parent_thread_id.clone())
            .map_err(|error| error.to_string())?
            .ok_or_else(|| {
                format!(
                    "spawned child {child_thread_id} parent {} has no durable identity",
                    parent.parent_thread_id
                )
            })?;
        if parent_identity.root_thread_id != child_identity.root_thread_id {
            return Err(
                "agent terminal result parent and child root identity mismatch".to_string(),
            );
        }
        let parent_thread = self
            .read_thread_sync(ReadThreadParams {
                thread_id: parent.parent_thread_id.clone(),
                include_archived: true,
                turns_view: ThreadTurnsView::NotLoaded,
            })
            .map_err(|error| error.to_string())?
            .ok_or_else(|| {
                format!(
                    "spawned child {child_thread_id} parent {} has no canonical Thread",
                    parent.parent_thread_id
                )
            })?;
        let child_thread = self
            .read_thread_sync(ReadThreadParams {
                thread_id: child_thread_id.clone(),
                include_archived: true,
                turns_view: ThreadTurnsView::Full,
            })
            .map_err(|error| error.to_string())?
            .ok_or_else(|| format!("spawned child {child_thread_id} has no canonical Thread"))?;

        let mut activities = Vec::new();
        for turn_id in turn_ids {
            let turn = child_thread
                .turns
                .iter()
                .find(|turn| turn.turn_id.as_str() == turn_id)
                .ok_or_else(|| {
                    format!("terminal child turn {turn_id} is not canonical before result delivery")
                })?;
            let Some((result_status, payload)) = terminal_result_payload(turn) else {
                continue;
            };
            let created_at_ms = turn.completed_at_ms.ok_or_else(|| {
                format!("terminal child turn {turn_id} has no durable completion timestamp")
            })?;
            let content = format!(
                "Message Type: FINAL_ANSWER\nTask name: {}\nSender: {}\nPayload:\n{}",
                parent_identity.agent_path, child_identity.agent_path, payload
            );
            let message = self
                .append_agent_mailbox_message_sync(AppendAgentMailboxMessageParams {
                    message: AgentMailboxMessage {
                        message_id: terminal_result_message_id(
                            &child_identity.root_thread_id,
                            child_thread_id,
                            &turn.turn_id,
                            result_status,
                        ),
                        root_thread_id: child_identity.root_thread_id.clone(),
                        sender_thread_id: child_thread_id.clone(),
                        recipient_thread_id: parent.parent_thread_id.clone(),
                        content,
                        kind: AgentMailboxMessageKind::Result,
                        source_turn_id: Some(turn.turn_id.clone()),
                        result_status: Some(result_status),
                        delivery_mode: AgentMailboxDeliveryMode::QueueOnly,
                        delivery_status: AgentMailboxDeliveryStatus::Pending,
                        created_at_ms,
                        delivered_at_ms: None,
                    },
                })
                .map_err(|error| error.to_string())?;
            if message.delivery_status == AgentMailboxDeliveryStatus::Pending {
                activities.push(RuntimeSessionTerminalActivity {
                    recipient_session_id: parent_thread.session_id.to_string(),
                    input: super::inter_agent_input::from_mailbox_message(&message),
                });
            }
        }
        Ok(activities)
    }
}

fn terminal_result_turn_ids(events: &[app_server_protocol::AgentEvent]) -> BTreeSet<&str> {
    events
        .iter()
        .filter(|event| matches!(event.event_type.as_str(), "turn.completed" | "turn.failed"))
        .filter_map(|event| event.turn_id.as_deref())
        .collect()
}

pub(in crate::runtime) fn terminal_result_payload(
    turn: &Turn,
) -> Option<(AgentMailboxResultStatus, String)> {
    match turn.status {
        TurnStatus::Completed => Some((
            AgentMailboxResultStatus::Completed,
            turn.items
                .iter()
                .rev()
                .find_map(|item| match &item.payload {
                    ThreadItemPayload::AgentMessage { text, phase, .. }
                        if matches!(
                            phase,
                            Some(agent_protocol::response_item::MessagePhase::FinalAnswer)
                        ) =>
                    {
                        Some(text.clone())
                    }
                    _ => None,
                })
                .unwrap_or_default(),
        )),
        TurnStatus::Failed => {
            let error = turn
                .error
                .as_ref()
                .map(|error| truncate_error_text(&error.message))
                .unwrap_or_else(|| "unknown child agent failure".to_string());
            Some((
                AgentMailboxResultStatus::Failed,
                format!("Agent errored: {error}\n\n{ERROR_NEXT_ACTION}"),
            ))
        }
        TurnStatus::Interrupted | TurnStatus::InProgress => None,
    }
}

fn terminal_result_message_id(
    root_thread_id: &ThreadId,
    child_thread_id: &ThreadId,
    turn_id: &agent_protocol::TurnId,
    status: AgentMailboxResultStatus,
) -> String {
    let status = match status {
        AgentMailboxResultStatus::Completed => "completed",
        AgentMailboxResultStatus::Failed => "failed",
    };
    let digest = Sha256::digest(
        [
            root_thread_id.as_str(),
            child_thread_id.as_str(),
            turn_id.as_str(),
            status,
        ]
        .join("\u{1f}")
        .as_bytes(),
    );
    format!("agent-result-{}", hex::encode(digest))
}

fn truncate_error_text(value: &str) -> String {
    value.chars().take(ERROR_TEXT_MAX_CHARS).collect()
}
