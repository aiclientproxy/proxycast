use super::read_model;
use super::session_list_scope::normalize_cwd_values;
use super::session_list_scope::SessionListScope;
use super::session_title;
use super::status::resolve_agent_session_runtime_state;
use super::*;
use agent_protocol::AgentInput as UserInput;
use app_server_protocol::*;
use std::collections::{HashMap, HashSet};

const DEFAULT_SESSION_LIST_LIMIT: u32 = 100;

fn stored_session_to_overview(
    stored: &StoredSession,
    active_turn_id: Option<&str>,
) -> AgentSessionOverview {
    let session = &stored.session;
    let runtime_state =
        resolve_agent_session_runtime_state(session.status, 0, &stored.turns, active_turn_id);
    let explicit_title = session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.title.clone())
        .or_else(|| {
            session
                .business_object_ref
                .as_ref()
                .and_then(|reference| metadata_string(reference.metadata.as_ref(), "title"))
        });
    let first_user_message = first_user_message_from_stored_session(stored);
    AgentSessionOverview {
        session_id: session.session_id.clone(),
        thread_id: Some(session.thread_id.clone()),
        title: session_title::resolve_session_title(explicit_title, first_user_message),
        business_object_ref_metadata: session
            .business_object_ref
            .as_ref()
            .and_then(|reference| reference.metadata.clone()),
        model: session
            .business_object_ref
            .as_ref()
            .and_then(|reference| metadata_string(reference.metadata.as_ref(), "model"))
            .or_else(|| {
                session
                    .business_object_ref
                    .as_ref()
                    .and_then(|reference| metadata_string(reference.metadata.as_ref(), "modelName"))
            })
            .unwrap_or_default(),
        created_at: session.created_at.clone(),
        updated_at: session.updated_at.clone(),
        archived_at: None,
        workspace_id: session.workspace_id.clone(),
        working_dir: session
            .business_object_ref
            .as_ref()
            .and_then(|reference| metadata_string(reference.metadata.as_ref(), "workingDir"))
            .or_else(|| {
                session.business_object_ref.as_ref().and_then(|reference| {
                    metadata_string(reference.metadata.as_ref(), "working_dir")
                })
            }),
        execution_strategy: session
            .business_object_ref
            .as_ref()
            .and_then(|reference| metadata_string(reference.metadata.as_ref(), "executionStrategy"))
            .or_else(|| {
                session.business_object_ref.as_ref().and_then(|reference| {
                    metadata_string(reference.metadata.as_ref(), "execution_strategy")
                })
            }),
        messages_count: read_model::runtime_session_messages(stored).len(),
        thread_status: runtime_state.thread_status,
        latest_turn_status: runtime_state.latest_turn_status,
        active_turn_id: runtime_state.active_turn_id,
        queued_turn_count: runtime_state.queued_turn_count,
    }
}

fn first_user_message_from_stored_session(stored: &StoredSession) -> Option<String> {
    stored
        .turns
        .iter()
        .find_map(|turn| {
            stored
                .turn_inputs
                .get(&turn.turn_id)
                .and_then(|input| session_title::first_user_message_from_agent_input(input))
        })
        .or_else(|| {
            stored
                .events
                .iter()
                .filter(|event| event.event_type == turn_input_events::TURN_INPUT_EVENT_TYPE)
                .find_map(|event| {
                    session_title::first_user_message_from_runtime_payload(&event.payload)
                })
        })
}

pub(super) fn stored_session_hidden_from_user_recents(stored: &StoredSession) -> bool {
    stored
        .session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref())
        .is_some_and(metadata_hidden_from_user_recents)
}

fn metadata_hidden_from_user_recents(metadata: &serde_json::Value) -> bool {
    metadata_bool(metadata, "hiddenFromUserRecents")
        .or_else(|| metadata_bool(metadata, "hidden_from_user_recents"))
        .or_else(|| metadata_nested_bool(metadata, "harness", "hiddenFromUserRecents"))
        .or_else(|| metadata_nested_bool(metadata, "harness", "hidden_from_user_recents"))
        .unwrap_or(false)
}

fn metadata_bool(metadata: &serde_json::Value, key: &str) -> Option<bool> {
    metadata.get(key).and_then(serde_json::Value::as_bool)
}

fn metadata_nested_bool(metadata: &serde_json::Value, parent: &str, key: &str) -> Option<bool> {
    metadata
        .get(parent)
        .and_then(|value| value.get(key))
        .and_then(serde_json::Value::as_bool)
}

impl RuntimeCore {
    pub(in crate::runtime) async fn active_turn_id_for_session(
        &self,
        session_id: &str,
    ) -> Result<Option<String>, RuntimeCoreError> {
        self.session_loops
            .snapshot(session_id)
            .await
            .map(|snapshot| snapshot.and_then(|snapshot| snapshot.active_turn_id))
            .map_err(|error| {
                RuntimeCoreError::Backend(format!(
                    "read runtime session snapshot for session state: {error}"
                ))
            })
    }

    fn list_runtime_core_session_overviews(
        &self,
        params: &AgentSessionListParams,
        active_turn_ids: &HashMap<String, String>,
    ) -> Result<Vec<AgentSessionOverview>, RuntimeCoreError> {
        if params.archived_only.unwrap_or(false) {
            return Ok(Vec::new());
        }

        let scope = SessionListScope::from_params(params);
        let pending_thread_ids = self
            .projection_store
            .as_ref()
            .map(|store| {
                store
                    .list_pending_thread_spawn_ids_sync()
                    .map(|thread_ids| thread_ids.into_iter().collect::<HashSet<_>>())
                    .map_err(|error| RuntimeCoreError::Backend(error.to_string()))
            })
            .transpose()?
            .unwrap_or_default();
        let state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        Ok(state
            .sessions
            .values()
            .filter(|stored| {
                !pending_thread_ids.contains(&agent_protocol::ThreadId::new(
                    stored.session.thread_id.clone(),
                ))
            })
            .filter(|stored| !stored_session_hidden_from_user_recents(stored))
            .map(|stored| {
                stored_session_to_overview(
                    stored,
                    active_turn_ids
                        .get(&stored.session.session_id)
                        .map(String::as_str),
                )
            })
            .filter(|overview| {
                scope.matches_session(
                    overview.workspace_id.as_deref(),
                    overview.working_dir.as_deref(),
                )
            })
            .collect())
    }

    pub async fn list_agent_sessions(
        &self,
        params: AgentSessionListParams,
    ) -> Result<AgentSessionListResponse, RuntimeCoreError> {
        let params = self.normalize_agent_session_list_params(params).await?;
        let loaded_session_ids = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let mut active_turn_ids = HashMap::new();
        for session_id in loaded_session_ids {
            if let Some(active_turn_id) = self.active_turn_id_for_session(&session_id).await? {
                active_turn_ids.insert(session_id, active_turn_id);
            }
        }
        let mut sessions = Vec::new();
        let mut persisted_session_ids = HashSet::new();
        if let Some(projection_store) = self.projection_store.as_ref() {
            let projected = projection_store
                .list_session_overviews(&params)
                .map_err(RuntimeCoreError::Backend)?;
            for session in projected {
                if persisted_session_ids.insert(session.session_id.clone()) {
                    sessions.push(session);
                }
            }
        }
        for runtime_session in
            self.list_runtime_core_session_overviews(&params, &active_turn_ids)?
        {
            if let Some(projected) = sessions
                .iter_mut()
                .find(|projected| projected.session_id == runtime_session.session_id)
            {
                projected.thread_status = runtime_session.thread_status;
                projected.latest_turn_status = runtime_session.latest_turn_status;
                projected.active_turn_id = runtime_session.active_turn_id;
                projected.queued_turn_count = runtime_session.queued_turn_count;
            } else if persisted_session_ids.insert(runtime_session.session_id.clone()) {
                sessions.push(runtime_session);
            }
        }
        sessions.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        if let Some(limit) = params.limit.map(|value| value as usize) {
            sessions.truncate(limit);
        }
        Ok(AgentSessionListResponse { sessions })
    }

    async fn normalize_agent_session_list_params(
        &self,
        mut params: AgentSessionListParams,
    ) -> Result<AgentSessionListParams, RuntimeCoreError> {
        // Session overviews are a bounded navigation read. Callers that need a
        // larger archive must request an explicit limit instead of blocking the
        // App Server with the entire projection.
        if params.limit.is_none() {
            params.limit = Some(DEFAULT_SESSION_LIST_LIMIT);
        }
        if params.cwd.is_some() {
            params.workspace_id = None;
            return Ok(params);
        }

        let Some(workspace_id) = params
            .workspace_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return Ok(params);
        };

        let response = self
            .app_data_source
            .read_workspace(WorkspaceReadParams {
                id: workspace_id.to_string(),
            })
            .await?;
        let root_path = response.workspace.as_ref().and_then(workspace_root_path);
        let cwd_filters = normalize_cwd_values(root_path);
        if !cwd_filters.is_empty() {
            params.cwd = Some(AgentSessionCwdFilter::Many(cwd_filters));
        }
        Ok(params)
    }

    pub fn start_session(
        &self,
        params: AgentSessionStartParams,
    ) -> Result<AgentSessionStartResponse, RuntimeCoreError> {
        let now = timestamp();
        let session_id = optional_id_or_new(params.session_id, "sess");
        let thread_id = optional_id_or_new(params.thread_id, "thread");
        let session = AgentSession {
            session_id: session_id.clone(),
            thread_id,
            app_id: params.app_id,
            workspace_id: params.workspace_id,
            business_object_ref: params.business_object_ref,
            status: AgentSessionStatus::Idle,
            created_at: now.clone(),
            updated_at: now,
        };

        let stored = StoredSession {
            session: session.clone(),
            turns: Vec::new(),
            turn_inputs: HashMap::new(),
            turn_runtime_options: HashMap::new(),
            events: Vec::new(),
            output_blobs: HashMap::new(),
        };
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        if state.sessions.contains_key(&session_id) {
            return Err(RuntimeCoreError::SessionAlreadyExists(session_id));
        }
        if let Some(projection_store) = self.projection_store.as_deref() {
            projection_store
                .create_empty_canonical_thread(&stored)
                .map_err(RuntimeCoreError::Backend)?;
        }
        state.sessions.insert(session_id, stored);

        Ok(AgentSessionStartResponse { session })
    }

    pub fn read_session(
        &self,
        params: AgentSessionReadParams,
    ) -> Result<AgentSessionReadResponse, RuntimeCoreError> {
        let stored = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            state
                .sessions
                .get(&params.session_id)
                .cloned()
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(params.session_id.clone()))?
        };
        if self.is_pending_agent_control_thread(&stored.session.thread_id)? {
            return Err(RuntimeCoreError::SessionNotFound(params.session_id));
        }
        let workflow_audit_events =
            self.read_workflow_audit_events_for_session(&params.session_id)?;
        let active_turn_id = stored
            .turns
            .iter()
            .rev()
            .find(|turn| super::status::agent_turn_blocks_queue_resume(turn.status))
            .map(|turn| turn.turn_id.as_str());
        let detail = read_model::runtime_session_read_detail_with_options(
            &stored,
            read_model::ReadDetailOptions::from_params(&params),
            &workflow_audit_events,
            active_turn_id,
        );

        Ok(AgentSessionReadResponse {
            session: stored.session.clone(),
            turns: stored.turns.clone(),
            detail: Some(detail),
        })
    }

    pub async fn read_session_current(
        &self,
        params: AgentSessionReadParams,
    ) -> Result<AgentSessionReadResponse, RuntimeCoreError> {
        self.load_session_current(params)
            .await
            .map(|context| context.response)
    }

    pub(crate) async fn resolve_session_thread_id_current(
        &self,
        session_id: &str,
    ) -> Result<String, RuntimeCoreError> {
        if let Some(thread_id) = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .get(session_id)
            .map(|stored| stored.session.thread_id.clone())
        {
            if self.is_pending_agent_control_thread(&thread_id)? {
                return Err(RuntimeCoreError::SessionNotFound(session_id.to_string()));
            }
            return Ok(thread_id);
        }
        if let Some(projection_store) = self.projection_store.as_ref() {
            if let Some(projection) = projection_store
                .read_session_projection(
                    session_id,
                    super::projection_store::ProjectionReadWindow::default(),
                )
                .map_err(RuntimeCoreError::Backend)?
            {
                return Ok(projection.session.thread_id);
            }
        }
        Err(RuntimeCoreError::SessionNotFound(session_id.to_string()))
    }

    pub(in crate::runtime) async fn ensure_current_session_hydrated(
        &self,
        session_id: &str,
    ) -> Result<(), RuntimeCoreError> {
        if self.has_runtime_core_session(session_id) {
            return Ok(());
        }

        let context = self
            .load_session_current(AgentSessionReadParams {
                session_id: session_id.to_string(),
                history_limit: None,
                history_offset: None,
                history_before_message_id: None,
            })
            .await?;
        self.insert_hydrated_session(context.stored);
        Ok(())
    }

    fn has_runtime_core_session(&self, session_id: &str) -> bool {
        let state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        state.sessions.get(session_id).is_some_and(|stored| {
            !self
                .is_pending_agent_control_thread(&stored.session.thread_id)
                .unwrap_or(true)
        })
    }

    fn insert_hydrated_session(&self, stored: StoredSession) {
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let session_id = stored.session.session_id.clone();
        state.sessions.entry(session_id).or_insert(stored);
    }

    pub(in crate::runtime) fn stored_turn(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Result<Option<AgentTurn>, RuntimeCoreError> {
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
            .find(|turn| turn.turn_id == turn_id)
            .cloned())
    }

    pub(in crate::runtime) fn session_snapshot(
        &self,
        session_id: &str,
    ) -> Result<(AgentSession, Vec<AgentTurn>), RuntimeCoreError> {
        let state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let stored = state
            .sessions
            .get(session_id)
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
        Ok((stored.session.clone(), stored.turns.clone()))
    }

    pub(in crate::runtime) fn rollback_started_turn(
        &self,
        session_id: &str,
        turn_id: &str,
        previous_session: AgentSession,
    ) {
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        if let Some(stored) = state.sessions.get_mut(session_id) {
            stored.turns.retain(|turn| turn.turn_id != turn_id);
            stored.turn_inputs.remove(turn_id);
            stored.turn_runtime_options.remove(turn_id);
            stored.session = previous_session;
        }
    }

    pub(in crate::runtime) fn restore_queued_turn_if_missing(
        &self,
        session_id: &str,
        index: usize,
        turn: AgentTurn,
        input: Vec<UserInput>,
        runtime_options: Option<RuntimeOptions>,
    ) {
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        if let Some(stored) = state.sessions.get_mut(session_id) {
            if stored.turns.iter().any(|stored_turn| {
                stored_turn.turn_id == turn.turn_id
                    && !matches!(stored_turn.status, AgentTurnStatus::Queued)
            }) {
                return;
            }
            if !stored.turns.iter().any(|stored_turn| {
                stored_turn.turn_id == turn.turn_id
                    && matches!(stored_turn.status, AgentTurnStatus::Queued)
            }) {
                let insert_at = index.min(stored.turns.len());
                stored.turns.insert(insert_at, turn.clone());
            }
            let turn_id = turn.turn_id;
            stored.turn_inputs.insert(turn_id.clone(), input);
            match runtime_options {
                Some(runtime_options) => {
                    stored.turn_runtime_options.insert(turn_id, runtime_options);
                }
                None => {
                    stored.turn_runtime_options.remove(&turn_id);
                }
            }
        }
    }
}

fn workspace_root_path(workspace: &serde_json::Value) -> Option<&str> {
    workspace
        .get("rootPath")
        .or_else(|| workspace.get("root_path"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}
