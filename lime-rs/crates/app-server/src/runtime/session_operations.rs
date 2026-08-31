use super::{RuntimeCore, RuntimeCoreError, RuntimeCoreState};
use crate::permission_profile::{
    apply_permission_profile_to_metadata, permission_profile_policy,
    resolve_allowed_permission_profile, PermissionProfilePolicy, ResolvedPermissionProfile,
};
use agent_protocol::{CollaborationMode, CollaborationModeSettings, ModeKind};
use agent_runtime::session_loop::{
    RuntimeSessionHandler, RuntimeSessionOperation, RuntimeSessionOperationResult,
    RuntimeSessionOperationSubmission,
};
use app_server_protocol::protocol::v2::{
    ThreadMemoryMode, ThreadMemoryModeSetParams, ThreadMemoryModeSetResponse, ThreadSettings,
    ThreadSettingsUpdateParams,
};
use lime_core::config::ConfigManager;
use serde_json::{Map, Value};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tool_runtime::sandbox::SandboxBackendPlatform;

mod model_defaults;

#[derive(Clone)]
enum SessionMetadataMutation {
    ThreadSettings(ThreadSettingsUpdateParams),
    MemoryMode(ThreadMemoryMode),
}

enum SessionMetadataMutationResult {
    ThreadSettings(ThreadSettings),
    MemoryMode,
}

impl RuntimeCore {
    pub(crate) fn current_permission_profile_policy(
        &self,
        cwd: Option<&str>,
    ) -> Result<PermissionProfilePolicy, RuntimeCoreError> {
        let manager = ConfigManager::load(&self.app_config_path).map_err(|error| {
            RuntimeCoreError::Backend(format!(
                "failed to load permission profile policy config: {error}"
            ))
        })?;
        permission_profile_policy(manager.config(), cwd, SandboxBackendPlatform::current())
            .map_err(RuntimeCoreError::InvalidRequest)
    }

    pub(crate) fn resolve_allowed_permission_profile(
        &self,
        id: &str,
        cwd: Option<&str>,
    ) -> Result<ResolvedPermissionProfile, RuntimeCoreError> {
        let policy = self.current_permission_profile_policy(cwd)?;
        resolve_allowed_permission_profile(&policy, id).map_err(RuntimeCoreError::InvalidRequest)
    }

    pub(crate) async fn preflight_thread_start(
        &self,
        params: &app_server_protocol::AgentSessionStartParams,
    ) -> Result<(), RuntimeCoreError> {
        if !self.backend.requires_provider_selection() {
            return Ok(());
        }
        let session_id = normalized_identity(
            params.session_id.as_deref().unwrap_or_default(),
            "thread/start sessionId",
        )?;
        let thread_id = normalized_identity(
            params.thread_id.as_deref().unwrap_or_default(),
            "thread/start threadId",
        )?;
        let metadata = params
            .business_object_ref
            .as_ref()
            .and_then(|reference| reference.metadata.as_ref())
            .and_then(Value::as_object)
            .ok_or_else(|| {
                RuntimeCoreError::InvalidRequest(
                    "thread/start requires model route metadata".to_string(),
                )
            })?;
        let settings =
            thread_settings_from_metadata(metadata).map_err(RuntimeCoreError::InvalidRequest)?;
        let now = super::timestamp();
        let session = app_server_protocol::AgentSession {
            session_id,
            thread_id,
            app_id: params.app_id.clone(),
            workspace_id: params.workspace_id.clone(),
            business_object_ref: params.business_object_ref.clone(),
            status: app_server_protocol::AgentSessionStatus::Idle,
            created_at: now.clone(),
            updated_at: now,
        };
        let first_error = match self
            .backend
            .preflight_thread_settings(&session, &settings)
            .await
        {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        let Some(provider_id) = missing_model_catalog_provider(&first_error) else {
            return Err(first_error);
        };

        let refresh = self.refresh_model_provider_catalog(&provider_id).await;
        match refresh {
            Ok(response) if response.source == "Api" => {}
            Ok(response) => {
                tracing::warn!(
                    provider_id,
                    source = response.source,
                    error_kind = response.error_kind.as_deref().unwrap_or("unknown"),
                    "thread start model catalog refresh did not produce API metadata"
                );
                return Err(first_error);
            }
            Err(_) => {
                tracing::warn!(
                    provider_id,
                    "thread start model catalog refresh failed; preserving pending route"
                );
                return Err(first_error);
            }
        }

        self.backend
            .preflight_thread_settings(&session, &settings)
            .await
    }

    pub(in crate::runtime) fn session_memory_enabled(&self, session_id: &str) -> bool {
        let state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        state
            .sessions
            .get(session_id)
            .and_then(|stored| stored.session.business_object_ref.as_ref())
            .and_then(|reference| reference.metadata.as_ref())
            .and_then(|metadata| {
                metadata
                    .get("memoryMode")
                    .or_else(|| metadata.get("memory_mode"))
            })
            .and_then(Value::as_str)
            != Some("disabled")
    }

    pub async fn update_thread_settings(
        &self,
        mut params: ThreadSettingsUpdateParams,
    ) -> Result<ThreadSettings, RuntimeCoreError> {
        validate_thread_settings(&params)?;
        if let Some(profile_id) = params.permissions.as_deref() {
            self.resolve_allowed_permission_profile(profile_id, params.cwd.as_deref())?;
        }
        self.apply_target_model_defaults(&mut params).await?;
        let thread_id = params.thread_id.clone();
        let result = self
            .dispatch_session_metadata_mutation(
                &thread_id,
                SessionMetadataMutation::ThreadSettings(params),
            )
            .await?;
        match result {
            SessionMetadataMutationResult::ThreadSettings(settings) => Ok(settings),
            SessionMetadataMutationResult::MemoryMode => Err(RuntimeCoreError::Backend(
                "thread settings operation returned an invalid result".to_string(),
            )),
        }
    }

    pub(in crate::runtime) fn loaded_thread_settings(
        &self,
        thread_id: &str,
    ) -> Result<Option<(String, ThreadSettings, bool)>, RuntimeCoreError> {
        let state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let stored = state
            .sessions
            .values()
            .find(|stored| stored.session.thread_id == thread_id)
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(thread_id.to_string()))?;
        let Some(metadata) = stored
            .session
            .business_object_ref
            .as_ref()
            .and_then(|reference| reference.metadata.as_ref())
            .and_then(Value::as_object)
        else {
            return Ok(None);
        };
        if metadata_string(metadata, &["modelName", "model"]).is_none()
            || metadata_string(
                metadata,
                &["providerSelector", "providerName", "modelProvider"],
            )
            .is_none()
        {
            return Ok(None);
        }
        let settings =
            thread_settings_from_metadata(metadata).map_err(RuntimeCoreError::Backend)?;
        Ok(Some((
            stored.session.session_id.clone(),
            settings,
            super::agent_control::session_metadata_has_agent_control_route(metadata),
        )))
    }

    pub async fn set_thread_memory_mode(
        &self,
        params: ThreadMemoryModeSetParams,
    ) -> Result<ThreadMemoryModeSetResponse, RuntimeCoreError> {
        let thread_id = normalized_identity(&params.thread_id, "thread/memoryMode/set threadId")?;
        let result = self
            .dispatch_session_metadata_mutation(
                &thread_id,
                SessionMetadataMutation::MemoryMode(params.mode),
            )
            .await?;
        match result {
            SessionMetadataMutationResult::MemoryMode => Ok(ThreadMemoryModeSetResponse {}),
            SessionMetadataMutationResult::ThreadSettings(_) => Err(RuntimeCoreError::Backend(
                "memory mode operation returned an invalid result".to_string(),
            )),
        }
    }

    async fn dispatch_session_metadata_mutation(
        &self,
        thread_id: &str,
        mutation: SessionMetadataMutation,
    ) -> Result<SessionMetadataMutationResult, RuntimeCoreError> {
        let thread_id = normalized_identity(thread_id, "threadId")?;
        let thread = self
            .read_thread(agent_protocol::thread::ThreadReadParams {
                thread_id: agent_protocol::ThreadId::new(thread_id.clone()),
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .await?;
        if thread.thread.archived {
            return Err(RuntimeCoreError::InvalidRequest(format!(
                "thread is archived: {thread_id}"
            )));
        }
        let session_id = thread.thread.session_id.as_str().to_string();
        self.ensure_current_session_hydrated(&session_id).await?;

        let state = Arc::clone(&self.state);
        let backend = Arc::clone(&self.backend);
        let projection_store = self.projection_store.clone().ok_or_else(|| {
            RuntimeCoreError::Backend(
                "session metadata persistence requires the projection store".to_string(),
            )
        })?;
        let result = Arc::new(Mutex::new(None));
        let preflight_error = Arc::new(Mutex::new(None));
        let handler_result = Arc::clone(&result);
        let handler_preflight_error = Arc::clone(&preflight_error);
        let handler_session_id = session_id.clone();
        let handler_thread_id = thread_id.clone();
        let handler_mutation = mutation.clone();
        let handler = RuntimeSessionHandler::new(move |context| {
            let state = Arc::clone(&state);
            let projection_store = Arc::clone(&projection_store);
            let result = Arc::clone(&handler_result);
            let session_id = handler_session_id.clone();
            let thread_id = handler_thread_id.clone();
            let mutation = handler_mutation.clone();
            let backend = Arc::clone(&backend);
            let preflight_error = Arc::clone(&handler_preflight_error);
            Box::pin(async move {
                if context.session_id != session_id {
                    return Err("session actor identity changed during metadata update".to_string());
                }
                if let SessionMetadataMutation::ThreadSettings(params) = &mutation {
                    if thread_settings_require_route_preflight(params) {
                        let (candidate_session, candidate_settings) =
                            preview_thread_settings_mutation(
                                &state,
                                &session_id,
                                &thread_id,
                                params.clone(),
                            )?;
                        if let Err(error) = backend
                            .preflight_thread_settings(&candidate_session, &candidate_settings)
                            .await
                        {
                            let message = error.to_string();
                            *preflight_error.lock().map_err(|_| {
                                "thread settings preflight error lock poisoned".to_string()
                            })? = Some(error);
                            return Err(message);
                        }
                    }
                }
                let mutation_result = apply_session_metadata_mutation(
                    &state,
                    projection_store.as_ref(),
                    &session_id,
                    &thread_id,
                    mutation,
                )?;
                *result
                    .lock()
                    .map_err(|_| "session metadata result lock poisoned".to_string())? =
                    Some(mutation_result);
                Ok(())
            })
        });
        let operation = match mutation {
            SessionMetadataMutation::ThreadSettings(_) => {
                RuntimeSessionOperation::ThreadSettings { handler }
            }
            SessionMetadataMutation::MemoryMode(_) => {
                RuntimeSessionOperation::SetMemoryMode { handler }
            }
        };
        let session = self
            .session_loops
            .get_or_create(&session_id, &thread_id)
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
        let dispatch_result = match session
            .dispatch(RuntimeSessionOperationSubmission::new(operation))
            .await
        {
            Ok(result) => result,
            Err(error) => {
                if let Some(error) = preflight_error
                    .lock()
                    .map_err(|_| {
                        RuntimeCoreError::Backend(
                            "thread settings preflight error lock poisoned".to_string(),
                        )
                    })?
                    .take()
                {
                    return Err(error);
                }
                return Err(RuntimeCoreError::Backend(error.to_string()));
            }
        };
        if !matches!(
            dispatch_result,
            RuntimeSessionOperationResult::Accepted { .. }
        ) {
            return Err(RuntimeCoreError::Backend(
                "session metadata operation was not accepted".to_string(),
            ));
        }
        let mut result = result.lock().map_err(|_| {
            RuntimeCoreError::Backend("session metadata result lock poisoned".into())
        })?;
        result.take().ok_or_else(|| {
            RuntimeCoreError::Backend(
                "session metadata operation completed without a result".to_string(),
            )
        })
    }
}

fn preview_thread_settings_mutation(
    state: &Arc<Mutex<RuntimeCoreState>>,
    session_id: &str,
    thread_id: &str,
    params: ThreadSettingsUpdateParams,
) -> Result<(app_server_protocol::AgentSession, ThreadSettings), String> {
    let state = state
        .lock()
        .map_err(|_| "runtime core state lock poisoned".to_string())?;
    let stored = state
        .sessions
        .get(session_id)
        .ok_or_else(|| format!("session not found: {session_id}"))?;
    if stored.session.thread_id != thread_id {
        return Err(format!(
            "session/thread identity mismatch for thread {thread_id}"
        ));
    }

    let mut candidate_session = stored.session.clone();
    let metadata = candidate_session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.clone())
        .unwrap_or_else(|| Value::Object(Default::default()));
    let mut metadata = metadata
        .as_object()
        .cloned()
        .ok_or_else(|| "thread metadata must be a JSON object".to_string())?;
    apply_thread_settings_patch(&mut metadata, params)?;
    let settings = thread_settings_from_metadata(&metadata)?;
    let reference = candidate_session
        .business_object_ref
        .get_or_insert_with(|| app_server_protocol::BusinessObjectRef {
            kind: "agent.thread".to_string(),
            id: thread_id.to_string(),
            title: None,
            uri: None,
            metadata: None,
        });
    reference.metadata = Some(Value::Object(metadata));
    Ok((candidate_session, settings))
}

fn thread_settings_require_route_preflight(params: &ThreadSettingsUpdateParams) -> bool {
    params.model.is_some()
        || params.model_provider.is_some()
        || params.service_tier.is_some()
        || params.effort.is_some()
        || params.collaboration_mode.is_some()
}

fn apply_session_metadata_mutation(
    state: &Arc<Mutex<RuntimeCoreState>>,
    projection_store: &super::ProjectionStore,
    session_id: &str,
    thread_id: &str,
    mutation: SessionMetadataMutation,
) -> Result<SessionMetadataMutationResult, String> {
    let mut state = state
        .lock()
        .map_err(|_| "runtime core state lock poisoned".to_string())?;
    let stored = state
        .sessions
        .get_mut(session_id)
        .ok_or_else(|| format!("session not found: {session_id}"))?;
    if stored.session.thread_id != thread_id {
        return Err(format!(
            "session/thread identity mismatch for thread {thread_id}"
        ));
    }

    let mut updated_session = stored.session.clone();
    let metadata = updated_session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.clone())
        .unwrap_or_else(|| Value::Object(Default::default()));
    let mut metadata = metadata
        .as_object()
        .cloned()
        .ok_or_else(|| "thread metadata must be a JSON object".to_string())?;
    let result = match mutation {
        SessionMetadataMutation::ThreadSettings(params) => {
            apply_thread_settings_patch(&mut metadata, params)?;
            SessionMetadataMutationResult::ThreadSettings(thread_settings_from_metadata(&metadata)?)
        }
        SessionMetadataMutation::MemoryMode(mode) => {
            metadata.insert(
                "memoryMode".to_string(),
                Value::String(mode.as_str().to_string()),
            );
            SessionMetadataMutationResult::MemoryMode
        }
    };
    let reference = updated_session.business_object_ref.get_or_insert_with(|| {
        app_server_protocol::BusinessObjectRef {
            kind: "agent.thread".to_string(),
            id: thread_id.to_string(),
            title: None,
            uri: None,
            metadata: None,
        }
    });
    reference.metadata = Some(Value::Object(metadata));
    updated_session.updated_at = super::timestamp();

    projection_store.persist_session_metadata(&mut updated_session)?;
    stored.session = updated_session;
    Ok(result)
}

fn validate_thread_settings(params: &ThreadSettingsUpdateParams) -> Result<(), RuntimeCoreError> {
    normalized_identity(&params.thread_id, "thread/settings/update threadId")?;
    if !params.has_updates() {
        return Err(RuntimeCoreError::InvalidRequest(
            "thread/settings/update requires at least one setting".to_string(),
        ));
    }
    if params.sandbox_policy.is_some() && params.permissions.is_some() {
        return Err(RuntimeCoreError::InvalidRequest(
            "permissions cannot be combined with sandboxPolicy".to_string(),
        ));
    }
    if let Some(cwd) = params.cwd.as_deref() {
        let cwd = normalized_value(cwd, "cwd")?;
        if !Path::new(cwd).is_absolute() {
            return Err(RuntimeCoreError::InvalidRequest(
                "thread/settings/update cwd must be absolute".to_string(),
            ));
        }
    }
    validate_optional_string(params.model.as_deref(), "model")?;
    validate_optional_string(params.model_provider.as_deref(), "modelProvider")?;
    if params.model_provider.is_some() && params.model.is_none() {
        return Err(RuntimeCoreError::InvalidRequest(
            "thread/settings/update modelProvider requires model".to_string(),
        ));
    }
    validate_optional_string(params.effort.as_deref(), "effort")?;
    if let Some(Some(service_tier)) = params.service_tier.as_ref() {
        normalized_value(service_tier, "serviceTier")?;
    }
    for (name, value) in [
        ("approvalPolicy", params.approval_policy.as_ref()),
        ("approvalsReviewer", params.approvals_reviewer.as_ref()),
        ("sandboxPolicy", params.sandbox_policy.as_ref()),
        ("summary", params.summary.as_ref()),
        ("personality", params.personality.as_ref()),
        ("toolPreferences", params.tool_preferences.as_ref()),
    ] {
        if let Some(value) = value {
            validate_setting_value(value, name)?;
        }
    }
    if let Some(mode) = params.collaboration_mode.as_ref() {
        normalized_value(&mode.settings.model, "collaborationMode.settings.model")?;
        if params
            .model
            .as_ref()
            .is_some_and(|model| model != &mode.settings.model)
        {
            return Err(RuntimeCoreError::InvalidRequest(
                "thread/settings/update model must match collaborationMode.settings.model"
                    .to_string(),
            ));
        }
        validate_optional_string(
            mode.settings.reasoning_effort.as_deref(),
            "collaborationMode.settings.reasoning_effort",
        )?;
    }
    Ok(())
}

fn apply_thread_settings_patch(
    metadata: &mut Map<String, Value>,
    params: ThreadSettingsUpdateParams,
) -> Result<(), String> {
    let current_model = metadata_string(metadata, &["modelName", "model"]);
    let current_provider = metadata_string(
        metadata,
        &["providerSelector", "providerName", "modelProvider"],
    );
    let target_model = params
        .collaboration_mode
        .as_ref()
        .map(|mode| mode.settings.model.as_str())
        .or(params.model.as_deref())
        .or(current_model.as_deref());
    let target_provider = params
        .model_provider
        .as_deref()
        .or(current_provider.as_deref());
    let model_identity_changed =
        target_model != current_model.as_deref() || target_provider != current_provider.as_deref();
    let model_update = params.model.clone();
    let effort_update = params.effort.clone();
    let collaboration_model_update = params.collaboration_mode.is_some();
    let reset_effort_for_model =
        model_identity_changed && effort_update.is_none() && !collaboration_model_update;
    let reset_service_tier_for_model = model_identity_changed && params.service_tier.is_none();
    let clears_permission_profile = params.sandbox_policy.is_some();
    let permission_profile = params.permissions.clone();
    insert_string(metadata, "workingDir", params.cwd);
    insert_value(metadata, "approvalPolicy", params.approval_policy);
    insert_value(metadata, "approvalsReviewer", params.approvals_reviewer);
    insert_value(metadata, "sandboxPolicy", params.sandbox_policy);
    if clears_permission_profile {
        metadata.remove("permissions");
        metadata.remove("activePermissionProfile");
    }
    if let Some(profile_id) = permission_profile {
        apply_permission_profile_to_metadata(metadata, &profile_id)?;
    }
    insert_string(metadata, "modelName", params.model);
    if let Some(model_provider) = params.model_provider {
        metadata.insert(
            "providerSelector".to_string(),
            Value::String(model_provider.clone()),
        );
        metadata.insert("providerName".to_string(), Value::String(model_provider));
    }
    if model_identity_changed {
        metadata.remove("agentControlRoute");
    }
    if let Some(service_tier) = params.service_tier {
        match service_tier {
            Some(service_tier) => {
                metadata.insert("serviceTier".to_string(), Value::String(service_tier));
            }
            None => {
                metadata.remove("serviceTier");
                metadata.remove("service_tier");
            }
        }
    } else if reset_service_tier_for_model {
        metadata.remove("serviceTier");
        metadata.remove("service_tier");
    }
    insert_string(metadata, "reasoningEffort", params.effort);
    if reset_effort_for_model {
        metadata.remove("reasoningEffort");
        metadata.remove("effort");
    }
    insert_value(metadata, "reasoningSummary", params.summary);
    if let Some(mode) = params.collaboration_mode {
        metadata.insert(
            "modelName".to_string(),
            Value::String(mode.settings.model.clone()),
        );
        match mode.settings.reasoning_effort.as_ref() {
            Some(effort) => {
                metadata.insert("reasoningEffort".to_string(), Value::String(effort.clone()));
            }
            None => {
                metadata.remove("reasoningEffort");
                metadata.remove("effort");
            }
        }
        persist_collaboration_mode(metadata, mode)?;
    } else if model_update.is_some() || effort_update.is_some() {
        let model = metadata_string(metadata, &["modelName", "model"])
            .ok_or_else(|| "thread settings require a persisted model".to_string())?;
        let effort = metadata_string(metadata, &["reasoningEffort", "effort"]);
        let mut mode =
            persisted_collaboration_mode(metadata)?.unwrap_or_else(|| CollaborationMode {
                mode: ModeKind::Default,
                settings: CollaborationModeSettings {
                    model: model.clone(),
                    reasoning_effort: effort.clone(),
                    developer_instructions: None,
                },
            });
        if let Some(model) = model_update {
            mode.settings.model = model;
        }
        if reset_effort_for_model {
            mode.settings.reasoning_effort = None;
        } else if let Some(effort) = effort_update {
            mode.settings.reasoning_effort = Some(effort);
        }
        persist_collaboration_mode(metadata, mode)?;
    }
    insert_value(metadata, "personality", params.personality);
    insert_value(metadata, "toolPreferences", params.tool_preferences);
    Ok(())
}

fn thread_settings_from_metadata(metadata: &Map<String, Value>) -> Result<ThreadSettings, String> {
    let model = metadata_string(metadata, &["modelName", "model"])
        .ok_or_else(|| "thread settings require a persisted model".to_string())?;
    let model_provider = metadata_string(
        metadata,
        &["providerSelector", "providerName", "modelProvider"],
    )
    .ok_or_else(|| "thread settings require a persisted model provider".to_string())?;
    let effort = metadata_string(metadata, &["reasoningEffort", "effort"]);
    let collaboration_mode =
        persisted_collaboration_mode(metadata)?.unwrap_or_else(|| CollaborationMode {
            mode: ModeKind::Default,
            settings: CollaborationModeSettings {
                model: model.clone(),
                reasoning_effort: effort.clone(),
                developer_instructions: None,
            },
        });
    Ok(ThreadSettings {
        cwd: metadata_string(metadata, &["workingDir", "cwd"]).unwrap_or_default(),
        approval_policy: metadata_value(metadata, &["approvalPolicy"]),
        approvals_reviewer: metadata_value(metadata, &["approvalsReviewer"]),
        sandbox_policy: metadata_value(metadata, &["sandboxPolicy", "sandbox"]),
        active_permission_profile: metadata_alias(metadata, &["activePermissionProfile"]),
        model,
        model_provider,
        service_tier: metadata_string(metadata, &["serviceTier"]),
        effort,
        summary: metadata_alias(metadata, &["reasoningSummary", "summary"]),
        collaboration_mode,
        personality: metadata_alias(metadata, &["personality"]),
        tool_preferences: metadata_alias(metadata, &["toolPreferences"]),
    })
}

fn persisted_collaboration_mode(
    metadata: &Map<String, Value>,
) -> Result<Option<CollaborationMode>, String> {
    metadata
        .get("collaborationMode")
        .cloned()
        .map(serde_json::from_value::<CollaborationMode>)
        .transpose()
        .map_err(|error| format!("invalid persisted collaborationMode: {error}"))
}

fn persist_collaboration_mode(
    metadata: &mut Map<String, Value>,
    mode: CollaborationMode,
) -> Result<(), String> {
    metadata.insert(
        "collaborationMode".to_string(),
        serde_json::to_value(mode)
            .map_err(|error| format!("serialize collaboration mode: {error}"))?,
    );
    Ok(())
}

fn normalized_identity(value: &str, field: &str) -> Result<String, RuntimeCoreError> {
    normalized_value(value, field).map(str::to_string)
}

fn missing_model_catalog_provider(error: &RuntimeCoreError) -> Option<String> {
    match error {
        RuntimeCoreError::PendingRoute {
            provider: Some(provider),
            reason_code,
            ..
        } if reason_code == "model_registry_metadata_missing" => {
            let provider = provider.trim();
            (!provider.is_empty()).then(|| provider.to_string())
        }
        _ => None,
    }
}

fn normalized_value<'a>(value: &'a str, field: &str) -> Result<&'a str, RuntimeCoreError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(RuntimeCoreError::InvalidRequest(format!(
            "{field} must not be empty"
        )));
    }
    Ok(value)
}

fn validate_optional_string(value: Option<&str>, field: &str) -> Result<(), RuntimeCoreError> {
    if let Some(value) = value {
        normalized_value(value, field)?;
    }
    Ok(())
}

fn validate_setting_value(value: &Value, field: &str) -> Result<(), RuntimeCoreError> {
    if value.is_null() || value.as_str().is_some_and(|value| value.trim().is_empty()) {
        return Err(RuntimeCoreError::InvalidRequest(format!(
            "thread/settings/update {field} must not be empty"
        )));
    }
    Ok(())
}

fn insert_string(metadata: &mut Map<String, Value>, key: &str, value: Option<String>) {
    if let Some(value) = value {
        metadata.insert(key.to_string(), Value::String(value));
    }
}

fn insert_value(metadata: &mut Map<String, Value>, key: &str, value: Option<Value>) {
    if let Some(value) = value {
        metadata.insert(key.to_string(), value);
    }
}

fn metadata_alias(metadata: &Map<String, Value>, keys: &[&str]) -> Option<Value> {
    keys.iter().find_map(|key| metadata.get(*key)).cloned()
}

fn metadata_value(metadata: &Map<String, Value>, keys: &[&str]) -> Value {
    metadata_alias(metadata, keys).unwrap_or(Value::Null)
}

fn metadata_string(metadata: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| metadata.get(*key))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;

    struct RejectingModelSwitchBackend;

    #[async_trait]
    impl crate::ExecutionBackend for RejectingModelSwitchBackend {
        async fn preflight_thread_settings(
            &self,
            session: &app_server_protocol::AgentSession,
            settings: &ThreadSettings,
        ) -> Result<(), RuntimeCoreError> {
            Err(RuntimeCoreError::RouteRejected {
                session_id: session.session_id.clone(),
                provider: Some(settings.model_provider.clone()),
                model: Some(settings.model.clone()),
                category: app_server_protocol::RouteFailureCategory::ModelUnavailable,
                reason_code: "model_registry_metadata_missing".to_string(),
            })
        }

        async fn start_turn(
            &self,
            _request: crate::ExecutionRequest,
            _sink: &mut dyn crate::RuntimeEventSink,
        ) -> Result<(), RuntimeCoreError> {
            Ok(())
        }

        async fn cancel_turn(
            &self,
            _request: crate::CancelExecutionRequest,
            _sink: &mut dyn crate::RuntimeEventSink,
        ) -> Result<(), RuntimeCoreError> {
            Ok(())
        }

        async fn respond_action(
            &self,
            _request: crate::ActionRespondRequest,
            _sink: &mut dyn crate::RuntimeEventSink,
        ) -> Result<(), RuntimeCoreError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn rejected_route_settings_do_not_persist_candidate_settings() {
        let temp = tempfile::TempDir::new().expect("thread settings temp dir");
        let store = Arc::new(
            crate::ProjectionStore::initialize(temp.path().join("projection.sqlite"))
                .expect("thread settings projection store"),
        );
        let core = RuntimeCore::with_backend(Arc::new(RejectingModelSwitchBackend))
            .with_projection_store(store);
        core.start_session(app_server_protocol::AgentSessionStartParams {
            session_id: Some("session-settings-preflight".to_string()),
            thread_id: Some("thread-settings-preflight".to_string()),
            app_id: "agent-chat".to_string(),
            workspace_id: None,
            business_object_ref: Some(app_server_protocol::BusinessObjectRef {
                kind: "agent.thread".to_string(),
                id: "thread-settings-preflight".to_string(),
                title: None,
                uri: None,
                metadata: Some(json!({
                    "providerSelector": "provider-a",
                    "providerName": "provider-a",
                    "modelName": "model-a"
                })),
            }),
            locale: None,
        })
        .expect("start settings preflight session");

        let tier_error = core
            .update_thread_settings(ThreadSettingsUpdateParams {
                thread_id: "thread-settings-preflight".to_string(),
                service_tier: Some(Some("unsupported-tier".to_string())),
                ..ThreadSettingsUpdateParams::default()
            })
            .await
            .expect_err("service tier update must be preflighted");
        assert!(matches!(
            tier_error,
            RuntimeCoreError::RouteRejected {
                reason_code,
                ..
            } if reason_code == "model_registry_metadata_missing"
        ));
        {
            let state = core
                .state
                .lock()
                .expect("runtime state after tier rejection");
            let metadata = state.sessions["session-settings-preflight"]
                .session
                .business_object_ref
                .as_ref()
                .and_then(|reference| reference.metadata.as_ref())
                .expect("persisted metadata after tier rejection");
            assert!(metadata.get("serviceTier").is_none());
        }

        let error = core
            .update_thread_settings(ThreadSettingsUpdateParams {
                thread_id: "thread-settings-preflight".to_string(),
                model: Some("missing-model".to_string()),
                model_provider: Some("provider-b".to_string()),
                ..ThreadSettingsUpdateParams::default()
            })
            .await
            .expect_err("model switch must be rejected");

        assert!(matches!(
            error,
            RuntimeCoreError::RouteRejected {
                reason_code,
                ..
            } if reason_code == "model_registry_metadata_missing"
        ));
        let state = core.state.lock().expect("runtime state");
        let metadata = state.sessions["session-settings-preflight"]
            .session
            .business_object_ref
            .as_ref()
            .and_then(|reference| reference.metadata.as_ref())
            .expect("persisted metadata");
        assert_eq!(metadata["providerSelector"], "provider-a");
        assert_eq!(metadata["providerName"], "provider-a");
        assert_eq!(metadata["modelName"], "model-a");
    }

    #[test]
    fn plain_model_and_effort_updates_refresh_the_active_collaboration_mode() {
        let mut metadata = json!({
            "modelName": "model-a",
            "providerSelector": "provider-a",
            "reasoningEffort": "low",
            "collaborationMode": {
                "mode": "plan",
                "settings": {
                    "model": "model-a",
                    "reasoning_effort": "low",
                    "developer_instructions": "Keep the existing plan instructions."
                }
            },
            "agentControlRoute": {
                "schemaVersion": 2,
                "providerPreference": "provider-a",
                "modelPreference": "model-a"
            }
        })
        .as_object()
        .expect("metadata object")
        .clone();

        apply_thread_settings_patch(
            &mut metadata,
            ThreadSettingsUpdateParams {
                thread_id: "thread-1".to_string(),
                model: Some("model-b".to_string()),
                model_provider: Some("provider-b".to_string()),
                effort: Some("high".to_string()),
                ..ThreadSettingsUpdateParams::default()
            },
        )
        .expect("update metadata");

        let mode = persisted_collaboration_mode(&metadata)
            .expect("valid mode")
            .expect("persisted mode");
        assert_eq!(mode.mode, ModeKind::Plan);
        assert_eq!(mode.settings.model, "model-b");
        assert_eq!(mode.settings.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(metadata["providerSelector"], "provider-b");
        assert_eq!(metadata["providerName"], "provider-b");
        assert!(metadata.get("agentControlRoute").is_none());
        assert_eq!(
            mode.settings.developer_instructions.as_deref(),
            Some("Keep the existing plan instructions.")
        );
    }

    #[test]
    fn model_only_update_clears_effort_owned_by_the_previous_model() {
        let mut metadata = json!({
            "modelName": "reasoning-model",
            "providerSelector": "provider-a",
            "reasoningEffort": "high",
            "serviceTier": "priority",
            "collaborationMode": {
                "mode": "default",
                "settings": {
                    "model": "reasoning-model",
                    "reasoning_effort": "high"
                }
            }
        })
        .as_object()
        .expect("metadata object")
        .clone();

        apply_thread_settings_patch(
            &mut metadata,
            ThreadSettingsUpdateParams {
                thread_id: "thread-1".to_string(),
                model: Some("plain-model".to_string()),
                model_provider: Some("provider-b".to_string()),
                ..ThreadSettingsUpdateParams::default()
            },
        )
        .expect("update metadata");

        let settings = thread_settings_from_metadata(&metadata).expect("thread settings");
        assert_eq!(settings.model, "plain-model");
        assert_eq!(settings.model_provider, "provider-b");
        assert_eq!(settings.effort, None);
        assert_eq!(settings.service_tier, None);
        assert_eq!(settings.collaboration_mode.settings.reasoning_effort, None);
    }
}
