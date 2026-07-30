//! Per-turn model-tool gateway for the durable agent control graph.

use super::agent_control::AgentControlSpawnRequest;
use super::*;
use agent_protocol::ThreadId;
use app_server_protocol::{
    AgentSessionStatus, AgentSessionTurnCancelParams, AgentTurnStatus, RuntimeOptions,
};
use async_trait::async_trait;
use serde_json::json;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;
use thread_store::{
    AgentGraphStore, AgentIdentity, AgentIdentityStore, AgentMailboxDeliveryMode,
    AgentMailboxDeliveryStatus, AgentMailboxMessage, AgentMailboxMessageKind, AgentMailboxStore,
    AppendAgentMailboxMessageParams, ReadThreadParams, ThreadSpawnEdgeStatus, ThreadStore,
};
use tool_runtime::agent_control::{
    AgentControlGateway, AgentControlGatewayError, AgentControlGatewayHandle,
    AgentControlGatewayRequest, AgentControlGatewayResult, AgentStateProjectionFact,
    SubAgentProjectionActivity, SubAgentProjectionFact,
};

use super::agent_control_gateway_support::{
    agent_control_path_matches, agent_control_status_from_turn, agent_control_turn_created_at_ms,
    required_agent_control_id, resolve_agent_control_path, stable_agent_control_message_id,
    validate_agent_control_task_name, ROOT_AGENT_PATH,
};

mod spawn_runtime_options;
pub(super) mod spawn_tool_options;
mod wait;

#[derive(Clone)]
struct RuntimeCoreAgentControlGateway {
    core: RuntimeCore,
    host: RuntimeHostContext,
    session_id: String,
    thread_id: String,
    turn_id: String,
    child_runtime_options: Option<RuntimeOptions>,
}

#[async_trait]
impl AgentControlGateway for RuntimeCoreAgentControlGateway {
    async fn execute(
        &self,
        request: AgentControlGatewayRequest,
    ) -> Result<AgentControlGatewayResult, AgentControlGatewayError> {
        let caller = &request.caller;
        if caller.session_id != self.session_id
            || caller.thread_id != self.thread_id
            || caller.turn_id != self.turn_id
        {
            return Err(AgentControlGatewayError::new(
                "agent control caller is outside its per-turn gateway scope",
            ));
        }
        self.core
            .execute_agent_control_gateway(request, &self.host, self.child_runtime_options.clone())
            .await
            .map_err(agent_control_gateway_error)
    }
}

impl RuntimeCore {
    pub(in crate::runtime) fn agent_control_gateway_for_turn(
        &self,
        session: &AgentSession,
        turn: &app_server_protocol::AgentTurn,
        host: RuntimeHostContext,
    ) -> AgentControlGatewayHandle {
        let child_runtime_options = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .get(&session.session_id)
            .and_then(|stored| stored.turn_runtime_options.get(&turn.turn_id))
            .cloned()
            .map(agent_control_child_runtime_options);
        AgentControlGatewayHandle::new(Arc::new(RuntimeCoreAgentControlGateway {
            core: self.clone(),
            host,
            session_id: session.session_id.clone(),
            thread_id: session.thread_id.clone(),
            turn_id: turn.turn_id.clone(),
            child_runtime_options,
        }))
    }

    async fn execute_agent_control_gateway(
        &self,
        request: AgentControlGatewayRequest,
        host: &RuntimeHostContext,
        child_runtime_options: Option<RuntimeOptions>,
    ) -> Result<AgentControlGatewayResult, RuntimeCoreError> {
        let caller = self.resolve_agent_control_caller(&request.caller).await?;
        let (output, projection_facts, state_facts) = match request.command {
            tool_runtime::agent_control::AgentControlCommand::SpawnAgent {
                task_name,
                message,
                fork_mode,
                model_overrides,
            } => {
                let (output, target_thread_id, agent_path) = self
                    .execute_agent_control_spawn(
                        &caller,
                        task_name,
                        message,
                        fork_mode,
                        model_overrides,
                        host,
                        child_runtime_options.clone(),
                    )
                    .await?;
                (
                    output,
                    vec![SubAgentProjectionFact {
                        target_thread_id,
                        activity: SubAgentProjectionActivity::Started,
                        detail: Some(agent_path),
                    }],
                    Vec::new(),
                )
            }
            tool_runtime::agent_control::AgentControlCommand::SendMessage { target, message } => {
                let (output, target_thread_id, agent_path) = self
                    .execute_agent_control_message(
                        &caller,
                        target,
                        message,
                        AgentMailboxDeliveryMode::QueueOnly,
                        host,
                        child_runtime_options.clone(),
                    )
                    .await?;
                (
                    output,
                    vec![SubAgentProjectionFact {
                        target_thread_id,
                        activity: SubAgentProjectionActivity::Interacted,
                        detail: Some(agent_path),
                    }],
                    Vec::new(),
                )
            }
            tool_runtime::agent_control::AgentControlCommand::FollowupTask { target, message } => {
                let (output, target_thread_id, agent_path) = self
                    .execute_agent_control_message(
                        &caller,
                        target,
                        message,
                        AgentMailboxDeliveryMode::TriggerTurn,
                        host,
                        child_runtime_options.clone(),
                    )
                    .await?;
                (
                    output,
                    vec![SubAgentProjectionFact {
                        target_thread_id,
                        activity: SubAgentProjectionActivity::Interacted,
                        detail: Some(agent_path),
                    }],
                    Vec::new(),
                )
            }
            tool_runtime::agent_control::AgentControlCommand::WaitAgent { timeout_ms } => {
                let result = self
                    .execute_agent_control_wait(&caller, timeout_ms, request.cancel_token)
                    .await?;
                (result.output, Vec::new(), result.state_facts)
            }
            tool_runtime::agent_control::AgentControlCommand::InterruptAgent { target } => {
                let (output, target_thread_id, agent_path) = self
                    .execute_agent_control_interrupt(&caller, target, host.clone())
                    .await?;
                (
                    output,
                    vec![SubAgentProjectionFact {
                        target_thread_id,
                        activity: SubAgentProjectionActivity::Interrupted,
                        detail: Some(agent_path),
                    }],
                    Vec::new(),
                )
            }
            tool_runtime::agent_control::AgentControlCommand::ListAgents { path_prefix } => (
                self.execute_agent_control_list(&caller, path_prefix)
                    .await?,
                Vec::new(),
                Vec::new(),
            ),
        };
        Ok(AgentControlGatewayResult {
            output,
            projection_facts,
            state_facts,
        })
    }

    async fn resolve_agent_control_caller(
        &self,
        caller: &tool_runtime::agent_control::AgentControlCaller,
    ) -> Result<ResolvedAgentControlCaller, RuntimeCoreError> {
        let session_id = required_agent_control_id(
            caller.session_id.clone(),
            "agent control caller session id is required",
        )?;
        let thread_id = required_agent_control_id(
            caller.thread_id.clone(),
            "agent control caller thread id is required",
        )?;
        let turn_id = required_agent_control_id(
            caller.turn_id.clone(),
            "agent control caller turn id is required",
        )?;
        let call_id = required_agent_control_id(
            caller.call_id.clone(),
            "agent control caller call id is required",
        )?;
        self.ensure_current_session_hydrated(&session_id).await?;
        let (session, turns) = self.session_snapshot(&session_id)?;
        if session.thread_id != thread_id {
            return Err(RuntimeCoreError::Backend(
                "agent control caller session/thread identity mismatch".to_string(),
            ));
        }
        let turn = turns
            .iter()
            .find(|turn| {
                turn.turn_id == turn_id
                    && matches!(
                        turn.status,
                        AgentTurnStatus::Accepted
                            | AgentTurnStatus::Running
                            | AgentTurnStatus::WaitingAction
                    )
            })
            .ok_or_else(|| RuntimeCoreError::TurnNotActive(turn_id.clone()))?;
        let store = self.agent_control_store()?;
        let identity = self
            .agent_control_identity_or_root(&store, &session)
            .await?;
        let visible_thread_ids = self
            .open_agent_control_thread_ids(&store, &identity.root_thread_id)
            .await?;
        if !visible_thread_ids.contains(&identity.thread_id) {
            return Err(RuntimeCoreError::Backend(
                "agent control caller is not an open member of its durable root-thread tree"
                    .to_string(),
            ));
        }
        Ok(ResolvedAgentControlCaller {
            session,
            identity,
            turn_id: caller.turn_id.trim().to_string(),
            call_id,
            created_at_ms: agent_control_turn_created_at_ms(turn)?,
        })
    }

    async fn execute_agent_control_spawn(
        &self,
        caller: &ResolvedAgentControlCaller,
        task_name: String,
        message: String,
        fork_mode: tool_runtime::agent_control::SpawnAgentForkMode,
        model_overrides: tool_runtime::agent_control::SpawnAgentModelOverrides,
        host: &RuntimeHostContext,
        child_runtime_options: Option<RuntimeOptions>,
    ) -> Result<(serde_json::Value, ThreadId, String), RuntimeCoreError> {
        let task_name = validate_agent_control_task_name(task_name)?;
        let message = required_agent_control_id(message, "agent control message is required")?;
        let path = format!("{}/{}", caller.identity.agent_path, task_name);
        let child_thread_id = uuid::Uuid::now_v7().to_string();
        let child_session_id = child_thread_id.clone();
        let child_runtime_options = spawn_runtime_options::prepare(
            self,
            &caller.session,
            host,
            &child_session_id,
            &child_thread_id,
            child_runtime_options,
            &model_overrides,
        )
        .await?;
        let response = self
            .stage_agent_control_spawn_with_runtime_options(
                AgentControlSpawnRequest {
                    parent_session_id: caller.session.session_id.clone(),
                    child_session_id: Some(child_session_id),
                    child_thread_id: Some(child_thread_id),
                    fork_mode,
                },
                child_runtime_options.clone(),
            )
            .await?;
        let store = self.agent_control_store()?;
        let identity = AgentIdentity {
            root_thread_id: caller.identity.root_thread_id.clone(),
            thread_id: ThreadId::new(response.session.thread_id.clone()),
            agent_path: path,
            nickname: None,
            role: None,
            last_task_message: Some(message.clone()),
        };
        if let Err(error) = store.upsert_agent_identity(identity.clone()).await {
            self.cleanup_unusable_agent_control_child(
                &caller.identity.root_thread_id,
                Some(&response.session.session_id),
                ThreadId::new(response.session.thread_id.clone()),
            )
                .await
                .map_err(|cleanup_error| {
                    RuntimeCoreError::Backend(format!(
                        "failed to persist agent identity after child spawn: {error}; failed to close and remove unlinked child: {cleanup_error}"
                    ))
                })?;
            return Err(RuntimeCoreError::Backend(format!(
                "failed to persist agent identity after child spawn: {error}"
            )));
        }
        let message_id = match self
            .append_agent_control_message(
                caller,
                &response.session.session_id,
                ThreadId::new(response.session.thread_id.clone()),
                message,
                AgentMailboxDeliveryMode::TriggerTurn,
                "spawn_agent",
            )
            .await
        {
            Ok(message_id) => message_id,
            Err(error) => {
                self.cleanup_unusable_agent_control_child(
                    &caller.identity.root_thread_id,
                    Some(&response.session.session_id),
                    ThreadId::new(response.session.thread_id.clone()),
                )
                    .await
                    .map_err(|cleanup_error| {
                        RuntimeCoreError::Backend(format!(
                            "failed to persist initial child mailbox message: {error}; failed to close and remove unlinked child: {cleanup_error}"
                        ))
                    })?;
                return Err(error);
            }
        };
        if let Err(error) = self.commit_agent_control_spawn(&response.session).await {
            self.cleanup_unusable_agent_control_child(
                &caller.identity.root_thread_id,
                Some(&response.session.session_id),
                ThreadId::new(response.session.thread_id.clone()),
            )
            .await
            .map_err(|cleanup_error| {
                RuntimeCoreError::Backend(format!(
                    "failed to commit child spawn: {error}; failed to remove pending child: {cleanup_error}"
                ))
            })?;
            return Err(error);
        }
        self.wake_pending_session_work(
            response.session.session_id.clone(),
            host.clone(),
            child_runtime_options,
        );
        let task_name = identity.agent_path.clone();
        Ok((
            json!({
                "task_name": task_name,
                "message_id": message_id,
            }),
            identity.thread_id,
            identity.agent_path,
        ))
    }

    async fn execute_agent_control_message(
        &self,
        caller: &ResolvedAgentControlCaller,
        target: String,
        message: String,
        delivery_mode: AgentMailboxDeliveryMode,
        host: &RuntimeHostContext,
        child_runtime_options: Option<RuntimeOptions>,
    ) -> Result<(serde_json::Value, ThreadId, String), RuntimeCoreError> {
        let target = self.resolve_agent_control_target(caller, &target).await?;
        if delivery_mode == AgentMailboxDeliveryMode::TriggerTurn
            && target.thread_id == caller.identity.root_thread_id
        {
            return Err(RuntimeCoreError::Backend(
                "followup_task cannot target the root agent".to_string(),
            ));
        }
        let message = required_agent_control_id(message, "agent control message is required")?;
        let operation = if delivery_mode == AgentMailboxDeliveryMode::QueueOnly {
            "send_message"
        } else {
            "followup_task"
        };
        let message_id = self
            .append_agent_control_message(
                caller,
                &target.session_id,
                target.thread_id.clone(),
                message,
                delivery_mode,
                operation,
            )
            .await?;
        if delivery_mode == AgentMailboxDeliveryMode::TriggerTurn {
            let runtime_options = self
                .agent_control_followup_runtime_options(&target.session_id, child_runtime_options);
            self.wake_pending_session_work(
                target.session_id.clone(),
                host.clone(),
                runtime_options,
            );
        }
        Ok((
            json!({ "message_id": message_id }),
            target.thread_id,
            target.agent_path,
        ))
    }

    async fn execute_agent_control_interrupt(
        &self,
        caller: &ResolvedAgentControlCaller,
        target: String,
        host: RuntimeHostContext,
    ) -> Result<(serde_json::Value, ThreadId, String), RuntimeCoreError> {
        let target = self.resolve_agent_control_target(caller, &target).await?;
        if target.thread_id == caller.identity.root_thread_id {
            return Err(RuntimeCoreError::Backend(
                "root is not a spawned agent".to_string(),
            ));
        }
        if target.thread_id == caller.identity.thread_id {
            return Err(RuntimeCoreError::Backend(
                "an agent cannot interrupt itself".to_string(),
            ));
        }
        self.ensure_current_session_hydrated(&target.session_id)
            .await?;
        let (_, turns) = self.session_snapshot(&target.session_id)?;
        let active = turns
            .iter()
            .find(|turn| {
                matches!(
                    turn.status,
                    AgentTurnStatus::Accepted
                        | AgentTurnStatus::Queued
                        | AgentTurnStatus::Running
                        | AgentTurnStatus::WaitingAction
                )
            })
            .cloned();
        let previous_status = active
            .as_ref()
            .map(|turn| agent_control_status_from_turn(turn.status))
            .unwrap_or("completed");
        if let Some(turn) = active {
            self.cancel_turn(
                AgentSessionTurnCancelParams {
                    session_id: target.session_id.clone(),
                    turn_id: turn.turn_id,
                },
                host,
            )
            .await?;
        }
        Ok((
            json!({ "previous_status": previous_status }),
            target.thread_id,
            target.agent_path,
        ))
    }

    async fn execute_agent_control_list(
        &self,
        caller: &ResolvedAgentControlCaller,
        path_prefix: Option<String>,
    ) -> Result<serde_json::Value, RuntimeCoreError> {
        let store = self.agent_control_store()?;
        let prefix = path_prefix
            .as_deref()
            .map(|value| resolve_agent_control_path(&caller.identity.agent_path, value))
            .transpose()?;
        let visible_thread_ids = self
            .open_agent_control_thread_ids(&store, &caller.identity.root_thread_id)
            .await?;
        let mut entries = Vec::new();
        for identity in store
            .list_agent_identities(caller.identity.root_thread_id.clone())
            .await
            .map_err(agent_control_store_error)?
        {
            if identity.root_thread_id != caller.identity.root_thread_id
                || !visible_thread_ids.contains(&identity.thread_id)
            {
                continue;
            }
            let thread = store
                .read_thread(thread_store::ReadThreadParams {
                    thread_id: identity.thread_id.clone(),
                    include_archived: true,
                    turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
                })
                .await
                .map_err(agent_control_store_error)?
                .ok_or_else(|| {
                    RuntimeCoreError::Backend(format!(
                        "agent identity {} has no canonical Thread",
                        identity.thread_id
                    ))
                })?;
            entries.push(AgentControlListEntry {
                session_id: thread.session_id.to_string(),
                identity,
            });
        }
        entries.retain(|entry| {
            prefix
                .as_ref()
                .is_none_or(|prefix| agent_control_path_matches(&entry.identity.agent_path, prefix))
        });
        entries.sort_by(|left, right| {
            left.identity
                .agent_path
                .cmp(&right.identity.agent_path)
                .then_with(|| {
                    left.identity
                        .thread_id
                        .as_str()
                        .cmp(right.identity.thread_id.as_str())
                })
        });
        let mut agents = Vec::with_capacity(entries.len());
        for entry in entries {
            let status = self
                .agent_control_list_status(&entry.session_id)
                .await
                .unwrap_or("not_found");
            agents.push(json!({
                "agent_name": entry.identity.agent_path,
                "agent_status": status,
                "last_task_message": entry.identity.last_task_message,
            }));
        }
        Ok(json!({ "agents": agents }))
    }

    fn agent_control_store(&self) -> Result<Arc<ProjectionStore>, RuntimeCoreError> {
        self.projection_store.clone().ok_or_else(|| {
            RuntimeCoreError::Backend(
                "agent control requires canonical ProjectionStore".to_string(),
            )
        })
    }

    async fn cleanup_unusable_agent_control_child(
        &self,
        root_thread_id: &ThreadId,
        child_session_id: Option<&str>,
        child_thread_id: ThreadId,
    ) -> Result<(), RuntimeCoreError> {
        let store = self.agent_control_store()?;
        let mut cleanup_errors = Vec::new();
        if let Err(error) = store
            .delete_agent_mailbox_messages(root_thread_id.clone(), child_thread_id.clone())
            .await
        {
            cleanup_errors.push(format!("mailbox: {error}"));
        }
        if let Err(error) = store.delete_agent_identity(child_thread_id.clone()).await {
            cleanup_errors.push(format!("identity: {error}"));
        }
        if let Some(child_session_id) = child_session_id {
            if let Err(error) = self.delete_agent_control_child_data(child_session_id) {
                cleanup_errors.push(error);
            }
        }
        if cleanup_errors.is_empty() {
            if let Err(error) = store.delete_thread_spawn_edge(child_thread_id).await {
                cleanup_errors.push(format!("graph edge: {error}"));
            }
        }
        if cleanup_errors.is_empty() {
            Ok(())
        } else {
            Err(RuntimeCoreError::Backend(cleanup_errors.join("; ")))
        }
    }

    pub async fn recover_agent_control_spawns(
        &self,
        host: RuntimeHostContext,
        runtime_options: Option<RuntimeOptions>,
    ) -> Result<(), RuntimeCoreError> {
        let store = self.projection_store.clone();
        if let Some(store) = store.as_ref() {
            let pending_spawns = store
                .list_pending_thread_spawn_intents_sync()
                .map_err(agent_control_store_error)?;
            for (parent_thread_id, child_thread_id, child_session_id) in pending_spawns {
                let root_thread_id = match store
                    .read_agent_identity(child_thread_id.clone())
                    .await
                    .map_err(agent_control_store_error)?
                {
                    Some(identity) => identity.root_thread_id,
                    None => store
                        .read_agent_identity(parent_thread_id.clone())
                        .await
                        .map_err(agent_control_store_error)?
                        .map(|identity| identity.root_thread_id)
                        .unwrap_or(parent_thread_id),
                };
                self.cleanup_unusable_agent_control_child(
                    &root_thread_id,
                    Some(&child_session_id),
                    child_thread_id,
                )
                .await?;
            }
        }

        let mut queued_session_ids = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            state
                .sessions
                .iter()
                .filter(|(_, stored)| {
                    stored
                        .turns
                        .iter()
                        .any(|turn| matches!(turn.status, AgentTurnStatus::Queued))
                })
                .map(|(session_id, _)| session_id.clone())
                .collect::<BTreeSet<_>>()
        };
        let mut recovery_error = None;
        if let Some(store) = store.as_ref() {
            match store.list_queued_session_ids() {
                Ok(session_ids) => queued_session_ids.extend(session_ids),
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "failed to discover durable queued sessions from the projection store"
                    );
                    recovery_error.get_or_insert(agent_control_store_error(error));
                }
            }
        }

        let mut recovered_pending_work_sessions = BTreeSet::new();
        let mut pending_route = None;
        for session_id in queued_session_ids {
            recovered_pending_work_sessions.insert(session_id.clone());
            let recovery = self
                .process_pending_session_work_with_options(
                    &session_id,
                    host.clone(),
                    runtime_options.clone(),
                    None,
                )
                .await;
            match recovery {
                Ok(_) => {}
                Err(error @ RuntimeCoreError::PendingRoute { .. }) => {
                    tracing::debug!(
                        session_id = %session_id,
                        error = %error,
                        "queued turn recovery is pending provider/model route"
                    );
                    pending_route.get_or_insert(error);
                }
                Err(error) => {
                    tracing::warn!(
                        session_id = %session_id,
                        error = %error,
                        "queued turn recovery failed; continuing with remaining pending work"
                    );
                    recovery_error.get_or_insert(error);
                }
            }
        }

        let Some(store) = store else {
            if let Some(error) = recovery_error {
                return Err(error);
            }
            return pending_route.map_or(Ok(()), Err);
        };
        let recipients = store
            .list_pending_agent_mailbox_trigger_recipients()
            .await
            .map_err(agent_control_store_error)?;
        for recipient in recipients {
            let Some(parent) = store
                .read_thread_spawn_parent(recipient.recipient_thread_id.clone())
                .await
                .map_err(agent_control_store_error)?
            else {
                continue;
            };
            if parent.status != ThreadSpawnEdgeStatus::Open {
                continue;
            }
            let identity = store
                .read_agent_identity(recipient.recipient_thread_id.clone())
                .await
                .map_err(agent_control_store_error)?;
            let thread = store
                .read_thread(thread_store::ReadThreadParams {
                    thread_id: recipient.recipient_thread_id.clone(),
                    include_archived: true,
                    turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
                })
                .await
                .map_err(agent_control_store_error)?;
            let (identity, thread) = match (identity, thread) {
                (Some(identity), Some(thread)) => (identity, thread),
                (_, thread) => {
                    let child_session_id = thread
                        .as_ref()
                        .map(|thread| thread.session_id.as_str().to_string());
                    self.cleanup_unusable_agent_control_child(
                        &recipient.root_thread_id,
                        child_session_id.as_deref(),
                        recipient.recipient_thread_id,
                    )
                    .await?;
                    continue;
                }
            };
            if identity.root_thread_id != recipient.root_thread_id {
                return Err(RuntimeCoreError::Backend(format!(
                    "pending mailbox recipient {} changed root identity",
                    identity.thread_id
                )));
            }
            if recovered_pending_work_sessions.contains(thread.session_id.as_str()) {
                continue;
            }
            let recovery = self
                .process_pending_session_work_with_options(
                    thread.session_id.as_str(),
                    host.clone(),
                    runtime_options.clone(),
                    None,
                )
                .await;
            match recovery {
                Ok(_) => {}
                Err(RuntimeCoreError::SessionNotFound(_)) => {
                    self.cleanup_unusable_agent_control_child(
                        &recipient.root_thread_id,
                        Some(thread.session_id.as_str()),
                        recipient.recipient_thread_id,
                    )
                    .await?;
                }
                Err(error @ RuntimeCoreError::PendingRoute { .. }) => {
                    tracing::debug!(
                        session_id = %thread.session_id,
                        error = %error,
                        "agent control recovery is pending provider/model route"
                    );
                    pending_route.get_or_insert(error);
                }
                Err(error) => {
                    tracing::warn!(
                        session_id = %thread.session_id,
                        error = %error,
                        "agent control recovery failed; continuing with remaining recipients"
                    );
                    recovery_error.get_or_insert(error);
                }
            }
        }
        if let Some(error) = recovery_error {
            return Err(error);
        }
        pending_route.map_or(Ok(()), Err)
    }

    async fn agent_control_identity_or_root(
        &self,
        store: &ProjectionStore,
        session: &AgentSession,
    ) -> Result<AgentIdentity, RuntimeCoreError> {
        if let Some(identity) = store
            .read_agent_identity(ThreadId::new(session.thread_id.clone()))
            .await
            .map_err(agent_control_store_error)?
        {
            return Ok(identity);
        }
        store
            .upsert_agent_identity(AgentIdentity {
                root_thread_id: ThreadId::new(session.thread_id.clone()),
                thread_id: ThreadId::new(session.thread_id.clone()),
                agent_path: ROOT_AGENT_PATH.to_string(),
                nickname: None,
                role: None,
                last_task_message: None,
            })
            .await
            .map_err(agent_control_store_error)
    }

    async fn resolve_agent_control_target(
        &self,
        caller: &ResolvedAgentControlCaller,
        target: &str,
    ) -> Result<ResolvedAgentControlTarget, RuntimeCoreError> {
        let target =
            required_agent_control_id(target.to_string(), "agent control target is required")?;
        let store = self.agent_control_store()?;
        let visible_thread_ids = self
            .open_agent_control_thread_ids(&store, &caller.identity.root_thread_id)
            .await?;
        let identities = store
            .list_agent_identities(caller.identity.root_thread_id.clone())
            .await
            .map_err(agent_control_store_error)?
            .into_iter()
            .filter(|identity| {
                identity.root_thread_id == caller.identity.root_thread_id
                    && visible_thread_ids.contains(&identity.thread_id)
            })
            .collect::<Vec<_>>();
        if target == caller.identity.root_thread_id.as_str() {
            return Ok(ResolvedAgentControlTarget {
                session_id: self
                    .root_agent_control_session_id(&store, &caller.identity.root_thread_id)
                    .await?,
                thread_id: caller.identity.root_thread_id.clone(),
                agent_path: ROOT_AGENT_PATH.to_string(),
            });
        }
        if let Some(identity) = identities
            .iter()
            .find(|identity| identity.thread_id.as_str() == target)
        {
            return self
                .agent_control_target_from_identity(&store, identity.clone())
                .await;
        }
        let resolved_path = resolve_agent_control_path(&caller.identity.agent_path, &target)?;
        let identity = identities
            .into_iter()
            .find(|identity| identity.agent_path == resolved_path)
            .ok_or_else(|| {
                RuntimeCoreError::Backend(format!(
                    "agent target {target} is not in the current durable root-thread tree"
                ))
            })?;
        self.agent_control_target_from_identity(&store, identity)
            .await
    }

    fn agent_control_followup_runtime_options(
        &self,
        target_session_id: &str,
        cold_target_options: Option<RuntimeOptions>,
    ) -> Option<RuntimeOptions> {
        let target_options = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .get(target_session_id)
            .and_then(|stored| {
                stored
                    .turns
                    .iter()
                    .rev()
                    .find_map(|turn| stored.turn_runtime_options.get(&turn.turn_id).cloned())
            })
            .map(agent_control_child_runtime_options);
        target_options.or(cold_target_options)
    }

    async fn open_agent_control_thread_ids(
        &self,
        store: &ProjectionStore,
        root_thread_id: &ThreadId,
    ) -> Result<HashSet<ThreadId>, RuntimeCoreError> {
        let mut thread_ids = store
            .list_thread_spawn_descendants(
                root_thread_id.clone(),
                Some(ThreadSpawnEdgeStatus::Open),
            )
            .await
            .map_err(agent_control_store_error)?
            .into_iter()
            .collect::<HashSet<_>>();
        thread_ids.insert(root_thread_id.clone());
        Ok(thread_ids)
    }

    async fn append_agent_control_message(
        &self,
        caller: &ResolvedAgentControlCaller,
        recipient_session_id: &str,
        recipient_thread_id: ThreadId,
        content: String,
        delivery_mode: AgentMailboxDeliveryMode,
        operation: &str,
    ) -> Result<String, RuntimeCoreError> {
        let message_id = stable_agent_control_message_id(
            &caller.identity.root_thread_id,
            &caller.identity.thread_id,
            &caller.turn_id,
            &caller.call_id,
            operation,
            &recipient_thread_id,
        );
        let store = self.agent_control_store()?;
        let stored_message = store
            .append_agent_mailbox_message(AppendAgentMailboxMessageParams {
                message: AgentMailboxMessage {
                    message_id: message_id.clone(),
                    root_thread_id: caller.identity.root_thread_id.clone(),
                    sender_thread_id: caller.identity.thread_id.clone(),
                    recipient_thread_id,
                    content,
                    kind: AgentMailboxMessageKind::Message,
                    source_turn_id: None,
                    result_status: None,
                    delivery_mode,
                    delivery_status: AgentMailboxDeliveryStatus::Pending,
                    created_at_ms: caller.created_at_ms,
                    delivered_at_ms: None,
                },
            })
            .await
            .map_err(agent_control_store_error)?;
        if let Err(error) = self
            .session_loops
            .notify_inter_agent_communication(
                recipient_session_id,
                super::inter_agent_input::from_mailbox_message(&stored_message),
            )
            .await
        {
            // The durable mailbox is authoritative. A wake-up failure must not turn a committed
            // message into a caller-visible send failure; recovery and the next boundary will
            // still observe the pending record.
            tracing::warn!(
                session_id = recipient_session_id,
                error = %error,
                "failed to publish durable mailbox activity"
            );
        }
        Ok(message_id)
    }

    async fn agent_control_target_from_identity(
        &self,
        store: &ProjectionStore,
        identity: AgentIdentity,
    ) -> Result<ResolvedAgentControlTarget, RuntimeCoreError> {
        let thread = store
            .read_thread(thread_store::ReadThreadParams {
                thread_id: identity.thread_id.clone(),
                include_archived: true,
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .await
            .map_err(agent_control_store_error)?
            .ok_or_else(|| {
                RuntimeCoreError::Backend(format!(
                    "agent identity {} has no canonical Thread",
                    identity.thread_id
                ))
            })?;
        Ok(ResolvedAgentControlTarget {
            session_id: thread.session_id.to_string(),
            thread_id: identity.thread_id,
            agent_path: identity.agent_path,
        })
    }

    async fn root_agent_control_session_id(
        &self,
        store: &ProjectionStore,
        root_thread_id: &ThreadId,
    ) -> Result<String, RuntimeCoreError> {
        let thread = store
            .read_thread(thread_store::ReadThreadParams {
                thread_id: root_thread_id.clone(),
                include_archived: true,
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .await
            .map_err(agent_control_store_error)?
            .ok_or_else(|| {
                RuntimeCoreError::Backend("agent control root thread is not canonical".to_string())
            })?;
        Ok(thread.session_id.to_string())
    }

    async fn agent_control_list_status(
        &self,
        session_id: &str,
    ) -> Result<&'static str, RuntimeCoreError> {
        self.ensure_current_session_hydrated(session_id).await?;
        let (session, turns) = self.session_snapshot(session_id)?;
        if let Some(turn) = turns.iter().find(|turn| {
            matches!(
                turn.status,
                AgentTurnStatus::Accepted
                    | AgentTurnStatus::Queued
                    | AgentTurnStatus::Running
                    | AgentTurnStatus::WaitingAction
            )
        }) {
            return Ok(agent_control_status_from_turn(turn.status));
        }
        Ok(match session.status {
            AgentSessionStatus::Idle => "pending_init",
            AgentSessionStatus::Running | AgentSessionStatus::WaitingAction => "running",
            AgentSessionStatus::Completed => "completed",
            AgentSessionStatus::Failed => "errored",
            AgentSessionStatus::Canceled => "interrupted",
        })
    }
}

fn agent_control_child_runtime_options(mut options: RuntimeOptions) -> RuntimeOptions {
    options.event_name = None;
    options.queued_turn_id = None;
    options.expected_output = None;
    options.structured_output = None;
    options.output_schema = None;
    if let Some(request) = options.runtime_request.as_mut() {
        if let Some(provider_config) = request.provider_config.as_mut() {
            provider_config.api_key = None;
            provider_config.base_url = None;
        }
    }
    options
}

#[derive(Clone)]
struct ResolvedAgentControlCaller {
    session: AgentSession,
    identity: AgentIdentity,
    turn_id: String,
    call_id: String,
    created_at_ms: i64,
}

struct ResolvedAgentControlTarget {
    session_id: String,
    thread_id: ThreadId,
    agent_path: String,
}

struct AgentControlListEntry {
    session_id: String,
    identity: AgentIdentity,
}

fn agent_control_gateway_error(error: RuntimeCoreError) -> AgentControlGatewayError {
    AgentControlGatewayError::new(error.to_string())
}

fn agent_control_store_error(error: impl std::fmt::Display) -> RuntimeCoreError {
    RuntimeCoreError::Backend(format!("durable agent control failed: {error}"))
}
