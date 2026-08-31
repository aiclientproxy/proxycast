//! Canonical parent/child graph control backed by RuntimeCore and ProjectionStore.
//!
//! The graph is deliberately independent of legacy session metadata and Team state. A child is
//! usable only after its canonical session/thread, identity, initial mailbox, and Open edge exist.

use super::agent_control_gateway_support::{
    required_agent_control_id, stable_agent_control_digest,
};
use super::context_compaction::ForkCompactionSeed;
mod route;

use super::*;
use agent_protocol::{
    ItemStatus, ThreadId, ThreadItemPayload, ThreadTurnsView, TurnQueueState, TurnStatus,
};
use app_server_protocol::{
    AgentSessionStartParams, AgentSessionStatus, AgentTurnStatus, BusinessObjectRef, RuntimeOptions,
};
use serde_json::json;
use thread_store::{
    AgentGraphStore, ReadThreadParams, ThreadMetadataPatch, ThreadStore, UpdateThreadMetadataParams,
};
use tool_runtime::agent_control::SpawnAgentForkMode;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AgentControlSpawnRequest {
    pub parent_session_id: String,
    pub child_session_id: Option<String>,
    pub child_thread_id: Option<String>,
    pub fork_mode: SpawnAgentForkMode,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AgentControlSpawnResponse {
    pub session: AgentSession,
    pub parent_thread_id: String,
}

use route::{agent_control_route_snapshot, AGENT_CONTROL_ROUTE_KEY};

pub(in crate::runtime) fn session_metadata_has_agent_control_route(
    metadata: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    metadata
        .get(AGENT_CONTROL_ROUTE_KEY)
        .is_some_and(route::agent_control_route_snapshot_is_complete)
}

impl RuntimeCore {
    pub(in crate::runtime) fn persist_agent_control_session_defaults(
        &self,
        session_id: &str,
        runtime_options: Option<&RuntimeOptions>,
    ) -> Result<AgentSession, RuntimeCoreError> {
        let default_metadata = agent_control_session_default_metadata(runtime_options);
        if default_metadata.is_empty() {
            return self
                .session_snapshot(session_id)
                .map(|(session, _)| session);
        };

        let mut session = {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get_mut(session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
            let session_id = stored.session.session_id.clone();
            let reference =
                stored
                    .session
                    .business_object_ref
                    .get_or_insert_with(|| BusinessObjectRef {
                        kind: "agent.session".to_string(),
                        id: session_id,
                        title: None,
                        uri: None,
                        metadata: None,
                    });
            let mut metadata = match reference.metadata.take() {
                Some(serde_json::Value::Object(metadata)) => metadata,
                Some(previous) => {
                    let mut metadata = serde_json::Map::new();
                    metadata.insert("previousMetadata".to_string(), previous);
                    metadata
                }
                None => serde_json::Map::new(),
            };
            metadata.extend(default_metadata);
            reference.metadata = Some(serde_json::Value::Object(metadata));
            stored.session.updated_at = timestamp();
            stored.session.clone()
        };
        if let Some(projection_store) = self.projection_store.as_deref() {
            projection_store
                .persist_session_metadata(&mut session)
                .map_err(|error| {
                    RuntimeCoreError::Backend(format!(
                        "failed to persist AgentControl child route metadata: {error}"
                    ))
                })?;
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get_mut(session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
            stored.session = session.clone();
        }
        Ok(session)
    }

    pub(in crate::runtime) fn is_pending_agent_control_thread(
        &self,
        thread_id: &str,
    ) -> Result<bool, RuntimeCoreError> {
        self.projection_store
            .as_ref()
            .map(|store| {
                store
                    .is_pending_thread_spawn_sync(&ThreadId::new(thread_id))
                    .map_err(|error| RuntimeCoreError::Backend(error.to_string()))
            })
            .transpose()
            .map(|pending| pending.unwrap_or(false))
    }

    /// Reserves a hidden durable intent and materializes the canonical child session/thread.
    ///
    /// A graph-enabled control path requires ProjectionStore. We fail before session creation
    /// when it is unavailable so an in-memory child cannot escape without durable topology.
    #[cfg(test)]
    pub(crate) async fn stage_agent_control_spawn(
        &self,
        request: AgentControlSpawnRequest,
    ) -> Result<AgentControlSpawnResponse, RuntimeCoreError> {
        self.stage_agent_control_spawn_with_runtime_options(request, None)
            .await
    }

    pub(crate) async fn stage_agent_control_spawn_with_runtime_options(
        &self,
        request: AgentControlSpawnRequest,
        runtime_options: Option<RuntimeOptions>,
    ) -> Result<AgentControlSpawnResponse, RuntimeCoreError> {
        let AgentControlSpawnRequest {
            parent_session_id,
            child_session_id,
            child_thread_id,
            fork_mode,
        } = request;
        let parent_session_id = required_agent_control_id(
            parent_session_id,
            "parent session id is required for agent control spawn",
        )?;
        let projection_store = self.projection_store.as_ref().ok_or_else(|| {
            RuntimeCoreError::Backend(
                "agent control requires canonical ProjectionStore".to_string(),
            )
        })?;
        if self.backend.requires_provider_selection()
            && !route::has_complete_agent_control_route_snapshot(runtime_options.as_ref())
        {
            return Err(RuntimeCoreError::pending_route_for_session(
                parent_session_id.clone(),
                runtime_options.as_ref(),
            ));
        }
        let parent = self.loaded_agent_control_parent(&parent_session_id)?;
        let child_session_id = child_session_id
            .map(|value| required_agent_control_id(value, "child session id is empty"))
            .transpose()?
            .unwrap_or_else(|| new_id("sess"));
        let child_thread_id = child_thread_id
            .map(|value| required_agent_control_id(value, "child thread id is empty"))
            .transpose()?
            .unwrap_or_else(|| new_id("thread"));
        let inherits_parent_history = fork_mode != SpawnAgentForkMode::None;
        let (forked_turns, compaction_seed) = if inherits_parent_history {
            let parent_thread = projection_store
                .read_thread_sync(ReadThreadParams {
                    thread_id: ThreadId::new(parent.thread_id.clone()),
                    include_archived: true,
                    turns_view: ThreadTurnsView::Full,
                })
                .map_err(|error| {
                    RuntimeCoreError::Backend(format!(
                        "failed to read canonical parent history for fork: {error}"
                    ))
                })?
                .ok_or_else(|| {
                    RuntimeCoreError::Backend(format!(
                        "canonical parent thread is missing: {}",
                        parent.thread_id
                    ))
                })?;
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored_parent = state
                .sessions
                .get(&parent_session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(parent_session_id.to_string()))?;
            selected_agent_control_fork_history(stored_parent, &parent_thread.turns, fork_mode)?
        } else {
            (Vec::new(), None)
        };

        projection_store
            .create_pending_thread_spawn_edge(
                ThreadId::new(parent.thread_id.clone()),
                ThreadId::new(child_thread_id.clone()),
                child_session_id.clone(),
            )
            .await
            .map_err(|error| {
                RuntimeCoreError::Backend(format!(
                    "failed to reserve canonical child thread spawn: {error}"
                ))
            })?;

        let child_business_object_ref = {
            let metadata = agent_control_session_default_metadata(runtime_options.as_ref());
            (!metadata.is_empty()).then(|| BusinessObjectRef {
                kind: "agent.session".to_string(),
                id: child_session_id.clone(),
                title: None,
                uri: None,
                metadata: Some(serde_json::Value::Object(metadata)),
            })
        };
        let mut response = match self.start_session(AgentSessionStartParams {
            session_id: Some(child_session_id.clone()),
            thread_id: Some(child_thread_id.clone()),
            app_id: parent.app_id,
            workspace_id: parent.workspace_id,
            business_object_ref: child_business_object_ref,
            locale: None,
        }) {
            Ok(response) => response,
            Err(error) => {
                return match projection_store
                    .delete_thread_spawn_edge(ThreadId::new(child_thread_id))
                    .await
                {
                    Ok(()) => Err(error),
                    Err(cleanup_error) => Err(RuntimeCoreError::Backend(format!(
                        "failed to create canonical child session: {error}; failed to remove pending spawn intent: {cleanup_error}"
                    ))),
                };
            }
        };
        if let Err(error) = self.persist_agent_control_session_defaults(
            &response.session.session_id,
            runtime_options.as_ref(),
        ) {
            return match self
                .rollback_staged_agent_control_spawn(&response.session)
                .await
            {
                Ok(()) => Err(error),
                Err(cleanup_error) => Err(RuntimeCoreError::Backend(format!(
                    "failed to persist child session provider defaults: {error}; failed to remove partial child session: {cleanup_error}"
                ))),
            };
        }
        response.session = self.session_snapshot(&response.session.session_id)?.0;
        if let Err(error) = self.append_runtime_events(
            &response.session.session_id,
            &response.session.thread_id,
            None,
            vec![RuntimeEvent::new(
                "session.created",
                json!({ "session": &response.session }),
            )],
        ) {
            return match self
                .rollback_staged_agent_control_spawn(&response.session)
                .await
            {
                Ok(()) => Err(error),
                Err(cleanup_error) => Err(RuntimeCoreError::Backend(format!(
                    "failed to persist canonical child session: {error}; failed to remove partial child session: {cleanup_error}"
                ))),
            };
        }
        if inherits_parent_history {
            if let Err(error) = projection_store
                .update_thread_metadata(UpdateThreadMetadataParams {
                    thread_id: ThreadId::new(response.session.thread_id.clone()),
                    patch: ThreadMetadataPatch {
                        forked_from_id: Some(ThreadId::new(parent.thread_id.clone())),
                        ..Default::default()
                    },
                    include_archived: false,
                })
                .await
            {
                return match self
                    .rollback_staged_agent_control_spawn(&response.session)
                    .await
                {
                    Ok(()) => Err(RuntimeCoreError::Backend(format!(
                        "failed to persist canonical fork lineage: {error}"
                    ))),
                    Err(cleanup_error) => Err(RuntimeCoreError::Backend(format!(
                        "failed to persist canonical fork lineage: {error}; failed to remove partial child session: {cleanup_error}"
                    ))),
                };
            }
        }
        if let Err(error) =
            self.fork_agent_control_history(&response.session, forked_turns, compaction_seed)
        {
            return match self
                .rollback_staged_agent_control_spawn(&response.session)
                .await
            {
                Ok(()) => Err(error),
                Err(cleanup_error) => Err(RuntimeCoreError::Backend(format!(
                    "failed to fork canonical parent history: {error}; failed to remove partial child session: {cleanup_error}"
                ))),
            };
        }
        Ok(AgentControlSpawnResponse {
            session: response.session,
            parent_thread_id: parent.thread_id,
        })
    }

    pub(crate) async fn commit_agent_control_spawn(
        &self,
        child: &AgentSession,
    ) -> Result<(), RuntimeCoreError> {
        let projection_store = self.projection_store.as_ref().ok_or_else(|| {
            RuntimeCoreError::Backend(
                "agent control requires canonical ProjectionStore".to_string(),
            )
        })?;
        let committed = projection_store
            .commit_pending_thread_spawn_edge(
                ThreadId::new(child.thread_id.clone()),
                child.session_id.clone(),
            )
            .await
            .map_err(|error| {
                RuntimeCoreError::Backend(format!(
                    "failed to commit canonical child thread spawn: {error}"
                ))
            })?;
        if committed {
            Ok(())
        } else {
            Err(RuntimeCoreError::Backend(format!(
                "pending child thread spawn is missing or changed: {}",
                child.thread_id
            )))
        }
    }

    #[cfg(test)]
    pub(crate) async fn create_open_agent_control_child_for_test(
        &self,
        request: AgentControlSpawnRequest,
    ) -> Result<AgentControlSpawnResponse, RuntimeCoreError> {
        let response = self.stage_agent_control_spawn(request).await?;
        self.commit_agent_control_spawn(&response.session).await?;
        Ok(response)
    }

    async fn rollback_staged_agent_control_spawn(
        &self,
        child: &AgentSession,
    ) -> Result<(), String> {
        self.delete_agent_control_child_data(&child.session_id)?;
        let projection_store = self.projection_store.as_ref().ok_or_else(|| {
            "canonical ProjectionStore is missing during spawn rollback".to_string()
        })?;
        projection_store
            .delete_thread_spawn_edge(ThreadId::new(child.thread_id.clone()))
            .await
            .map_err(|error| format!("graph edge: {error}"))
    }

    fn loaded_agent_control_parent(
        &self,
        session_id: &str,
    ) -> Result<AgentSession, RuntimeCoreError> {
        self.state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .get(session_id)
            .map(|stored| stored.session.clone())
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))
    }

    fn fork_agent_control_history(
        &self,
        child: &AgentSession,
        forked_turns: Vec<ForkedAgentControlTurn>,
        compaction_seed: Option<ForkCompactionSeed>,
    ) -> Result<(), RuntimeCoreError> {
        let child_turn_ids = forked_turns
            .iter()
            .map(|forked| {
                (
                    forked.source_turn_id.clone(),
                    forked_agent_control_turn_id(child, forked),
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        if let Some(seed) = compaction_seed {
            let child_tail_turn_id = child_turn_ids
                .get(&seed.source_tail_start_turn_id)
                .ok_or_else(|| {
                    RuntimeCoreError::Backend(format!(
                        "compaction tail turn {} is not present in the fork history",
                        seed.source_tail_start_turn_id
                    ))
                })?;
            let payload = rewrite_fork_compaction_payload(seed.payload, child, child_tail_turn_id);
            self.append_runtime_events(
                &child.session_id,
                &child.thread_id,
                None,
                vec![RuntimeEvent::new("context.compaction.completed", payload)],
            )?;
        }
        for forked in forked_turns {
            let turn_id = forked_agent_control_turn_id(child, &forked);
            {
                let mut state = self
                    .state
                    .lock()
                    .expect("runtime core state mutex poisoned");
                let stored = state
                    .sessions
                    .get_mut(&child.session_id)
                    .ok_or_else(|| RuntimeCoreError::SessionNotFound(child.session_id.clone()))?;
                stored.session.status = AgentSessionStatus::Running;
                stored.session.updated_at = timestamp();
                stored.turn_inputs.insert(turn_id.clone(), forked.input);
                stored.turns.push(AgentTurn {
                    turn_id: turn_id.clone(),
                    session_id: child.session_id.clone(),
                    thread_id: child.thread_id.clone(),
                    status: AgentTurnStatus::Running,
                    started_at: None,
                    completed_at: None,
                });
            }

            let mut events = Vec::with_capacity(forked.assistant_messages.len() * 2 + 2);
            events.push(RuntimeEvent::new(
                "turn.started",
                json!({
                    "forked": true,
                    "forkedFromThreadId": forked.source_thread_id,
                    "forkedFromTurnId": forked.source_turn_id,
                }),
            ));
            for message in forked.assistant_messages {
                let item_id = format!(
                    "fork-{}",
                    stable_agent_control_digest(&[
                        child.thread_id.as_str(),
                        message.source_item_id.as_str(),
                    ])
                );
                let payload = json!({
                    "itemId": item_id,
                    "phase": "final_answer",
                    "text": message.text,
                    "contentParts": message.content_parts,
                    "metadata": {
                        "forkedFromThreadId": message.source_thread_id,
                        "forkedFromTurnId": message.source_turn_id,
                        "forkedFromItemId": message.source_item_id,
                    },
                });
                events.push(RuntimeEvent::new("message.delta", payload.clone()));
                events.push(RuntimeEvent::new(
                    "message.completed",
                    with_completed_message_status(payload),
                ));
            }
            events.push(RuntimeEvent::new(
                "turn.completed",
                json!({ "forked": true }),
            ));
            self.append_runtime_events(
                &child.session_id,
                &child.thread_id,
                Some(&turn_id),
                events,
            )?;
        }
        Ok(())
    }

    pub(in crate::runtime) fn delete_agent_control_child_data(
        &self,
        session_id: &str,
    ) -> Result<(), String> {
        let child_session_id = session_id.to_string();
        let mut cleanup_errors = Vec::new();
        if let Some(projection_store) = self.projection_store.as_ref() {
            if let Err(error) = projection_store.delete_session_data(&child_session_id) {
                cleanup_errors.push(format!("canonical projection: {error}"));
            }
        }
        if let Some(event_log_writer) = self.event_log_writer.as_ref() {
            if let Err(error) = event_log_writer.clear_session(&child_session_id) {
                cleanup_errors.push(format!("event log: {error}"));
            }
        }
        if let Some(sidecar_store) = self.sidecar_store.as_ref() {
            if let Err(error) = sidecar_store.clear_session(&child_session_id) {
                cleanup_errors.push(format!("sidecar: {error}"));
            }
        }
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        if let Some(thread_id) = state
            .sessions
            .get(&child_session_id)
            .map(|stored| stored.session.thread_id.clone())
        {
            state.thread_elicitation_counts.remove(&thread_id);
        }
        state.sessions.remove(&child_session_id);
        if cleanup_errors.is_empty() {
            Ok(())
        } else {
            Err(cleanup_errors.join("; "))
        }
    }
}

fn agent_control_session_defaults(
    runtime_options: Option<&RuntimeOptions>,
) -> Option<(String, String, String)> {
    let request = runtime_options?.runtime_request.as_ref()?;
    let provider_config = request.provider_config.as_ref();
    let provider_selector = trimmed_option(request.provider_preference.as_deref())
        .or_else(|| {
            provider_config.and_then(|config| trimmed_option(config.provider_id.as_deref()))
        })
        .or_else(|| {
            provider_config.and_then(|config| trimmed_option(config.provider_name.as_deref()))
        })?;
    let provider_name = provider_config
        .and_then(|config| trimmed_option(config.provider_name.as_deref()))
        .unwrap_or_else(|| provider_selector.clone());
    let model_name = trimmed_option(request.model_preference.as_deref()).or_else(|| {
        provider_config.and_then(|config| trimmed_option(config.model_name.as_deref()))
    })?;
    Some((provider_selector, provider_name, model_name))
}

fn agent_control_session_default_metadata(
    runtime_options: Option<&RuntimeOptions>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut metadata = serde_json::Map::new();
    if let Some((provider_selector, provider_name, model_name)) =
        agent_control_session_defaults(runtime_options)
    {
        metadata.insert("providerSelector".to_string(), json!(provider_selector));
        metadata.insert("providerName".to_string(), json!(provider_name));
        metadata.insert("modelName".to_string(), json!(model_name));
    }
    let route_snapshot = agent_control_route_snapshot(runtime_options);
    let service_tier = runtime_options
        .and_then(RuntimeOptions::runtime_request)
        .and_then(|request| trimmed_option(request.service_tier.as_deref()))
        .or_else(|| {
            route_snapshot.as_ref().and_then(|snapshot| {
                snapshot
                    .get("serviceTier")
                    .and_then(serde_json::Value::as_str)
                    .and_then(|value| trimmed_option(Some(value)))
            })
        });
    if let Some(route_snapshot) = route_snapshot {
        metadata.insert(AGENT_CONTROL_ROUTE_KEY.to_string(), route_snapshot);
    }
    if let Some(service_tier) = service_tier {
        metadata.insert("serviceTier".to_string(), json!(service_tier));
    }
    metadata
}

pub(in crate::runtime) fn runtime_options_with_agent_control_session_defaults(
    runtime_options: Option<RuntimeOptions>,
    session: &AgentSession,
) -> Option<RuntimeOptions> {
    let defaults = agent_control_session_default_values(session);
    let route_snapshot = route::agent_control_route_snapshot_from_session(session);
    let session_service_tier = session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref())
        .and_then(|metadata| metadata.get("serviceTier"))
        .and_then(serde_json::Value::as_str)
        .and_then(|value| trimmed_option(Some(value)));
    let has_route_snapshot = route_snapshot.is_some();
    if defaults.is_none() && route_snapshot.is_none() && session_service_tier.is_none() {
        return runtime_options;
    }

    let mut options = runtime_options.unwrap_or_default();
    let request = options.runtime_request_mut();
    if let Some(route_snapshot) = route_snapshot {
        if let Some(stored_provider_config) =
            route::runtime_provider_config_from_route_snapshot(route_snapshot)
        {
            request.provider_config = Some(route::merge_runtime_provider_config(
                stored_provider_config,
                request.provider_config.take(),
            ));
        }
        route::restore_agent_control_route_metadata(request, route_snapshot);
        if let Some(provider) = route_snapshot
            .get("providerPreference")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
        {
            request.provider_preference = Some(provider);
        }
        if let Some(model) = route_snapshot
            .get("modelPreference")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
        {
            request.model_preference = Some(model);
        }
        request.service_tier = route_snapshot
            .get("serviceTier")
            .and_then(serde_json::Value::as_str)
            .and_then(|value| trimmed_option(Some(value)))
            .or(session_service_tier.clone());
    } else if request.service_tier.is_none() {
        request.service_tier = session_service_tier;
    }

    let Some((provider_selector, model_name)) = defaults else {
        return Some(options);
    };
    let provider_config_has_value = request.provider_config.as_ref().is_some_and(|config| {
        trimmed_option(config.provider_id.as_deref()).is_some()
            || trimmed_option(config.provider_name.as_deref()).is_some()
    });
    if !has_route_snapshot
        && trimmed_option(request.provider_preference.as_deref()).is_none()
        && !provider_config_has_value
    {
        request.provider_preference = Some(provider_selector);
    }

    let model_config_has_value = request
        .provider_config
        .as_ref()
        .and_then(|config| trimmed_option(config.model_name.as_deref()))
        .is_some();
    if !has_route_snapshot
        && trimmed_option(request.model_preference.as_deref()).is_none()
        && !model_config_has_value
    {
        request.model_preference = Some(model_name);
    }
    Some(options)
}

pub(in crate::runtime) fn has_agent_control_runtime_route(
    runtime_options: Option<&RuntimeOptions>,
) -> bool {
    agent_control_session_defaults(runtime_options).is_some()
}

fn agent_control_session_default_values(session: &AgentSession) -> Option<(String, String)> {
    let Some(metadata) = session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref())
    else {
        return None;
    };
    if let Some(route_snapshot) = metadata
        .get(route::AGENT_CONTROL_ROUTE_KEY)
        .and_then(|value| value.as_object())
    {
        let provider = route_snapshot
            .get("providerPreference")
            .and_then(serde_json::Value::as_str)
            .and_then(|value| trimmed_option(Some(value)));
        let model = route_snapshot
            .get("modelPreference")
            .and_then(serde_json::Value::as_str)
            .and_then(|value| trimmed_option(Some(value)));
        if let Some(defaults) = provider.zip(model) {
            return Some(defaults);
        }
    }
    let provider = [
        "providerSelector",
        "provider_selector",
        "providerName",
        "provider_name",
    ]
    .into_iter()
    .find_map(|key| trimmed_option(metadata.get(key).and_then(serde_json::Value::as_str)));
    let model = ["modelName", "model_name", "model"]
        .into_iter()
        .find_map(|key| trimmed_option(metadata.get(key).and_then(serde_json::Value::as_str)));
    provider.zip(model)
}

fn trimmed_option(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn forked_agent_control_turn_id(child: &AgentSession, forked: &ForkedAgentControlTurn) -> String {
    format!(
        "fork-{}",
        stable_agent_control_digest(&[
            child.thread_id.as_str(),
            forked.source_thread_id.as_str(),
            forked.source_turn_id.as_str(),
        ])
    )
}

fn rewrite_fork_compaction_payload(
    mut payload: serde_json::Value,
    child: &AgentSession,
    child_tail_turn_id: &str,
) -> serde_json::Value {
    if let Some(object) = payload.as_object_mut() {
        let source_compaction_id = object
            .get("compactionId")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string)
            .unwrap_or_else(|| "compaction".to_string());
        let child_compaction_id = format!("{}_{}", child.session_id, source_compaction_id);
        object.insert("compactionId".to_string(), json!(child_compaction_id));
        object.insert("sessionId".to_string(), json!(child.session_id));
        object.insert("threadId".to_string(), json!(child.thread_id));
        object.insert("tailStartTurnId".to_string(), json!(child_tail_turn_id));
        if let Some(artifact) = object
            .get_mut("artifact")
            .and_then(|value| value.as_object_mut())
        {
            artifact.insert(
                "compactionId".to_string(),
                json!(format!("{}_{}", child.session_id, source_compaction_id)),
            );
            artifact.insert("sessionId".to_string(), json!(child.session_id));
            artifact.insert("threadId".to_string(), json!(child.thread_id));
            artifact.insert("tailStartTurnId".to_string(), json!(child_tail_turn_id));
        }
    }
    payload
}

#[derive(Debug, Clone)]
struct ForkedAgentControlTurn {
    source_thread_id: String,
    source_turn_id: String,
    input: Vec<AgentInput>,
    assistant_messages: Vec<ForkedAgentControlMessage>,
}

#[derive(Debug, Clone)]
struct ForkedAgentControlMessage {
    source_thread_id: String,
    source_turn_id: String,
    source_item_id: String,
    text: String,
    content_parts: Vec<agent_protocol::MessageContentPart>,
}

fn selected_agent_control_fork_history(
    parent: &StoredSession,
    canonical_turns: &[agent_protocol::Turn],
    fork_mode: SpawnAgentForkMode,
) -> Result<(Vec<ForkedAgentControlTurn>, Option<ForkCompactionSeed>), RuntimeCoreError> {
    if fork_mode == SpawnAgentForkMode::None {
        return Ok((Vec::new(), None));
    }
    let compaction_seed = if fork_mode == SpawnAgentForkMode::FullHistory {
        super::context_compaction::latest_fork_compaction_seed(&parent.events)?
    } else {
        None
    };
    let mut source_turns = canonical_turns
        .iter()
        .filter(|turn| !matches!(turn.queue, TurnQueueState::Queued { .. }))
        .filter_map(|turn| {
            let mut canonical_inputs = turn.items.iter().filter_map(|item| {
                if item.status != ItemStatus::Completed {
                    return None;
                }
                match &item.payload {
                    ThreadItemPayload::UserMessage { .. } => Some(()),
                    _ => None,
                }
            });
            canonical_inputs.next()?;
            Some((turn, canonical_inputs.next().is_some()))
        })
        .collect::<Vec<_>>();
    if let SpawnAgentForkMode::LastNTurns(limit) = &fork_mode {
        source_turns = source_turns.split_off(source_turns.len().saturating_sub(*limit));
    }
    let turns = source_turns
        .into_iter()
        .map(|(turn, has_duplicate_input)| {
            if has_duplicate_input {
                return Err(RuntimeCoreError::Backend(format!(
                    "canonical fork source Turn {} contains multiple user inputs",
                    turn.turn_id
                )));
            }
            let input = parent
                .turn_inputs
                .get(turn.turn_id.as_str())
                .cloned()
                .ok_or_else(|| {
                    RuntimeCoreError::Backend(format!(
                        "canonical fork source Turn {} is missing durable input",
                        turn.turn_id
                    ))
                })?;
            ensure_agent_control_fork_input_is_lossless(turn, &input)?;
            validate_agent_control_fork_items(turn)?;
            let assistant_messages = if turn.status == TurnStatus::Completed {
                turn.items
                    .iter()
                    .filter(|item| item.status == ItemStatus::Completed)
                    .filter_map(|item| match &item.payload {
                        ThreadItemPayload::AgentMessage {
                            text,
                            phase,
                            content_parts,
                        } if matches!(
                            phase,
                            Some(agent_protocol::response_item::MessagePhase::FinalAnswer)
                        ) =>
                        {
                            Some(ForkedAgentControlMessage {
                                source_thread_id: turn.thread_id.as_str().to_string(),
                                source_turn_id: turn.turn_id.as_str().to_string(),
                                source_item_id: item.item_id.as_str().to_string(),
                                text: text.clone(),
                                content_parts: content_parts.clone(),
                            })
                        }
                        _ => None,
                    })
                    .collect()
            } else {
                Vec::new()
            };
            Ok(ForkedAgentControlTurn {
                source_thread_id: turn.thread_id.as_str().to_string(),
                source_turn_id: turn.turn_id.as_str().to_string(),
                input,
                assistant_messages,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if fork_mode == SpawnAgentForkMode::FullHistory {
        let compaction_marker = parent
            .events
            .iter()
            .find(|event| event.event_type == "context.compaction.completed");
        if let Some(marker) = compaction_marker.filter(|_| compaction_seed.is_none()) {
            let turn_id = marker.turn_id.as_deref().unwrap_or("unknown");
            return Err(RuntimeCoreError::Backend(format!(
                "canonical AgentControl full-history fork source Turn {turn_id} contains compaction without replacement history"
            )));
        }
        if let Some(seed) = compaction_seed.as_ref() {
            if !turns
                .iter()
                .any(|turn| turn.source_turn_id == seed.source_tail_start_turn_id)
            {
                return Err(RuntimeCoreError::Backend(format!(
                    "compaction tail turn {} is not present in the fork history",
                    seed.source_tail_start_turn_id
                )));
            }
        }
    }
    Ok((turns, compaction_seed))
}

fn ensure_agent_control_fork_input_is_lossless(
    turn: &agent_protocol::Turn,
    input: &[AgentInput],
) -> Result<(), RuntimeCoreError> {
    let text_only = matches!(
        input,
        [AgentInput::Text {
            text_elements,
            ..
        }] if text_elements.is_empty()
    );
    if text_only {
        return Ok(());
    }
    Err(RuntimeCoreError::Backend(format!(
        "canonical AgentControl fork source Turn {} contains user input that cannot be copied without loss",
        turn.turn_id
    )))
}

fn validate_agent_control_fork_items(turn: &agent_protocol::Turn) -> Result<(), RuntimeCoreError> {
    for item in &turn.items {
        match &item.payload {
            ThreadItemPayload::UserMessage { .. } if item.status == ItemStatus::Completed => {}
            ThreadItemPayload::UserMessage { .. } => {
                return Err(unsupported_agent_control_fork_item(turn, item));
            }
            ThreadItemPayload::Reasoning { .. }
            | ThreadItemPayload::Tool { .. }
            | ThreadItemPayload::DynamicToolCall { .. }
            | ThreadItemPayload::McpToolCall { .. }
            | ThreadItemPayload::CollabAgentToolCall { .. }
            | ThreadItemPayload::Plan { .. }
            | ThreadItemPayload::Approval { .. }
            | ThreadItemPayload::Command { .. }
            | ThreadItemPayload::File { .. }
            | ThreadItemPayload::Media { .. }
            | ThreadItemPayload::SubAgent { .. } => {}
            ThreadItemPayload::AgentMessage {
                phase,
                content_parts,
                ..
            } if matches!(
                phase,
                Some(agent_protocol::response_item::MessagePhase::FinalAnswer)
            ) =>
            {
                if item.status != ItemStatus::Completed
                    || content_parts.iter().any(|part| {
                        matches!(part, agent_protocol::MessageContentPart::Media { .. })
                    })
                {
                    return Err(unsupported_agent_control_fork_item(turn, item));
                }
            }
            ThreadItemPayload::AgentMessage { .. } => {}
            ThreadItemPayload::ContextCompaction { .. }
            | ThreadItemPayload::Unknown { .. }
            | ThreadItemPayload::Extension { .. } => {}
        }
    }
    Ok(())
}

fn unsupported_agent_control_fork_item(
    turn: &agent_protocol::Turn,
    item: &agent_protocol::ThreadItem,
) -> RuntimeCoreError {
    RuntimeCoreError::Backend(format!(
        "canonical AgentControl fork source Turn {} contains {:?} Item {} that cannot be copied without loss",
        turn.turn_id, item.kind, item.item_id
    ))
}

fn with_completed_message_status(mut payload: serde_json::Value) -> serde_json::Value {
    if let Some(payload) = payload.as_object_mut() {
        payload.insert("status".to_string(), json!("completed"));
    }
    payload
}

#[cfg(test)]
#[path = "tests/agent_control.rs"]
mod tests;
