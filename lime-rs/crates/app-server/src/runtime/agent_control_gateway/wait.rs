//! Durable mailbox and queued-steer wait semantics.

use super::*;

pub(super) struct AgentControlWaitResult {
    pub(super) output: serde_json::Value,
    pub(super) state_facts: Vec<AgentStateProjectionFact>,
}

#[derive(Default)]
struct AgentControlWaitActivity {
    output: Vec<serde_json::Value>,
    state_facts: Vec<AgentStateProjectionFact>,
}

impl AgentControlWaitResult {
    fn without_states(output: serde_json::Value) -> Self {
        Self {
            output,
            state_facts: Vec::new(),
        }
    }
}

impl AgentControlWaitActivity {
    fn is_empty(&self) -> bool {
        self.output.is_empty()
    }

    fn into_result(self) -> AgentControlWaitResult {
        AgentControlWaitResult {
            output: json!({
                "message": "Wait completed.",
                "timed_out": false,
                "activity": self.output,
            }),
            state_facts: self.state_facts,
        }
    }
}

impl RuntimeCore {
    pub(super) async fn execute_agent_control_wait(
        &self,
        caller: &ResolvedAgentControlCaller,
        timeout_ms: u64,
        cancel_token: Option<tokio_util::sync::CancellationToken>,
    ) -> Result<AgentControlWaitResult, RuntimeCoreError> {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        let activity = self
            .session_loops
            .subscribe_input_activity(&caller.session.session_id)
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
        let (mut activity_rx, mut pending_activity) = activity
            .map(|(receiver, pending)| (Some(receiver), pending))
            .unwrap_or((None, None));
        let mut next_recovery = tokio::time::Instant::now();
        let mut recovery_interval = std::time::Duration::from_millis(250);
        loop {
            if self.has_queued_agent_control_steer(&caller.session.session_id, &caller.turn_id)? {
                return Ok(AgentControlWaitResult::without_states(json!({
                    "message": "Wait interrupted by new input.",
                    "timed_out": false,
                })));
            }
            if let Some(activity) = pending_activity.take() {
                match activity {
                    agent_runtime::session_loop::RuntimeSessionInputActivity::Steer => {
                        return Ok(AgentControlWaitResult::without_states(json!({
                            "message": "Wait interrupted by new input.",
                            "timed_out": false,
                        })));
                    }
                    agent_runtime::session_loop::RuntimeSessionInputActivity::Mailbox => {
                        let activity = self.consume_agent_control_wait_activity(caller).await?;
                        if !activity.is_empty() {
                            return Ok(activity.into_result());
                        }
                    }
                }
            }
            let now = tokio::time::Instant::now();
            if now >= next_recovery {
                self.recover_direct_child_terminal_activity(&caller.session)
                    .await?;
                next_recovery = now + recovery_interval;
                recovery_interval = recovery_interval
                    .saturating_mul(2)
                    .min(std::time::Duration::from_secs(2));
            }
            if self.has_queued_agent_control_steer(&caller.session.session_id, &caller.turn_id)? {
                return Ok(AgentControlWaitResult::without_states(json!({
                    "message": "Wait interrupted by new input.",
                    "timed_out": false,
                })));
            }
            if self
                .has_pending_agent_mailbox_activity(&caller.session.session_id)
                .await?
            {
                let activity = self.consume_agent_control_wait_activity(caller).await?;
                if !activity.is_empty() {
                    return Ok(activity.into_result());
                }
            }
            if tokio::time::Instant::now() >= deadline {
                self.recover_direct_child_terminal_activity(&caller.session)
                    .await?;
                if self
                    .has_queued_agent_control_steer(&caller.session.session_id, &caller.turn_id)?
                {
                    return Ok(AgentControlWaitResult::without_states(json!({
                        "message": "Wait interrupted by new input.",
                        "timed_out": false,
                    })));
                }
                if self
                    .has_pending_agent_mailbox_activity(&caller.session.session_id)
                    .await?
                {
                    let activity = self.consume_agent_control_wait_activity(caller).await?;
                    if !activity.is_empty() {
                        return Ok(activity.into_result());
                    }
                }
                return Ok(AgentControlWaitResult::without_states(json!({
                    "message": "Wait timed out.",
                    "timed_out": true
                })));
            }
            let recovery_sleep = tokio::time::sleep_until(next_recovery);
            tokio::pin!(recovery_sleep);
            let deadline_sleep = tokio::time::sleep_until(deadline);
            tokio::pin!(deadline_sleep);
            tokio::select! {
                changed = async {
                    match activity_rx.as_mut() {
                        Some(receiver) => receiver.changed().await.ok(),
                        None => std::future::pending::<Option<()>>().await,
                    }
                } => {
                    if changed.is_none() {
                        activity_rx = None;
                        continue;
                    }
                    let activity = activity_rx
                        .as_mut()
                        .map(|receiver| *receiver.borrow_and_update());
                    match activity {
                        Some(agent_runtime::session_loop::RuntimeSessionInputActivity::Steer) => {
                            return Ok(AgentControlWaitResult::without_states(json!({
                                "message": "Wait interrupted by new input.",
                                "timed_out": false,
                            })));
                        }
                        Some(agent_runtime::session_loop::RuntimeSessionInputActivity::Mailbox) => {
                            let activity = self.consume_agent_control_wait_activity(caller).await?;
                            if !activity.is_empty() {
                                return Ok(activity.into_result());
                            }
                        }
                        None => {}
                    }
                }
                _ = &mut recovery_sleep => {
                    self.recover_direct_child_terminal_activity(&caller.session).await?;
                    next_recovery = tokio::time::Instant::now() + recovery_interval;
                    recovery_interval = recovery_interval
                        .saturating_mul(2)
                        .min(std::time::Duration::from_secs(2));
                }
                _ = &mut deadline_sleep => {
                    self.recover_direct_child_terminal_activity(&caller.session).await?;
                    if self.has_queued_agent_control_steer(&caller.session.session_id, &caller.turn_id)? {
                        return Ok(AgentControlWaitResult::without_states(json!({
                            "message": "Wait interrupted by new input.",
                            "timed_out": false,
                        })));
                    }
                    if self.has_pending_agent_mailbox_activity(&caller.session.session_id).await? {
                        let activity = self.consume_agent_control_wait_activity(caller).await?;
                        if !activity.is_empty() {
                            return Ok(activity.into_result());
                        }
                    }
                    return Ok(AgentControlWaitResult::without_states(json!({
                        "message": "Wait timed out.",
                        "timed_out": true
                    })));
                }
                _ = async {
                    match cancel_token.as_ref() {
                        Some(token) => token.cancelled().await,
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    return Ok(AgentControlWaitResult::without_states(json!({
                        "message": "Wait interrupted.",
                        "timed_out": false
                    })));
                }
            }
        }
    }

    async fn consume_agent_control_wait_activity(
        &self,
        caller: &ResolvedAgentControlCaller,
    ) -> Result<AgentControlWaitActivity, RuntimeCoreError> {
        let messages = self
            .consume_pending_agent_mailbox_for_wait(&caller.session.session_id, &caller.turn_id)
            .await?;
        if messages.is_empty() {
            return Ok(AgentControlWaitActivity::default());
        }
        let store = self.agent_control_store()?;
        let identities = store
            .list_agent_identities(caller.identity.root_thread_id.clone())
            .await
            .map_err(agent_control_store_error)?;
        let mut seen_senders = HashSet::new();
        let mut state_facts = Vec::new();
        for message in &messages {
            if !seen_senders.insert(message.sender_thread_id.clone()) {
                continue;
            }
            let fallback = mailbox_agent_state(message);
            let state = match store
                .read_thread(ReadThreadParams {
                    thread_id: message.sender_thread_id.clone(),
                    include_archived: true,
                    turns_view: agent_protocol::ThreadTurnsView::Full,
                })
                .await
            {
                Ok(Some(thread)) => thread.agent_state.unwrap_or(fallback),
                Ok(None) => agent_protocol::CollabAgentState {
                    status: agent_protocol::CollabAgentStatus::NotFound,
                    message: None,
                },
                Err(error) => {
                    tracing::warn!(
                        thread_id = %message.sender_thread_id,
                        error = %error,
                        "failed to read canonical agent state for wait projection"
                    );
                    fallback
                }
            };
            state_facts.push(AgentStateProjectionFact {
                target_thread_id: message.sender_thread_id.clone(),
                state,
            });
        }
        let output = messages
            .into_iter()
            .map(|message| {
                let sender = identities
                    .iter()
                    .find(|identity| identity.thread_id == message.sender_thread_id)
                    .map(|identity| identity.agent_path.clone())
                    .unwrap_or_else(|| message.sender_thread_id.to_string());
                let kind = match message.kind {
                    AgentMailboxMessageKind::Message => "message",
                    AgentMailboxMessageKind::Result => "result",
                };
                let result_status = message.result_status.map(|status| match status {
                    thread_store::AgentMailboxResultStatus::Completed => "completed",
                    thread_store::AgentMailboxResultStatus::Failed => "failed",
                });
                json!({
                    "message_id": message.message_id,
                    "sender": sender,
                    "content": message.content,
                    "kind": kind,
                    "source_turn_id": message.source_turn_id.map(|turn_id| turn_id.to_string()),
                    "result_status": result_status,
                })
            })
            .collect();
        Ok(AgentControlWaitActivity {
            output,
            state_facts,
        })
    }

    fn has_queued_agent_control_steer(
        &self,
        session_id: &str,
        current_turn_id: &str,
    ) -> Result<bool, RuntimeCoreError> {
        let state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let stored = state
            .sessions
            .get(session_id)
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
        Ok(stored
            .turns
            .iter()
            .any(|turn| turn.turn_id != current_turn_id && turn.status == AgentTurnStatus::Queued))
    }
}

fn mailbox_agent_state(message: &AgentMailboxMessage) -> agent_protocol::CollabAgentState {
    let status = match message.result_status {
        Some(thread_store::AgentMailboxResultStatus::Completed) => {
            agent_protocol::CollabAgentStatus::Completed
        }
        Some(thread_store::AgentMailboxResultStatus::Failed) => {
            agent_protocol::CollabAgentStatus::Errored
        }
        None => agent_protocol::CollabAgentStatus::Running,
    };
    agent_protocol::CollabAgentState {
        status,
        message: None,
    }
}
