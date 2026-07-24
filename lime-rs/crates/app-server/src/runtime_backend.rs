mod action_response;
mod agent_skills_context;
mod agent_skills_telemetry;
mod coding_events;
mod event_mapper;
mod execution_backend;
mod image_command;
mod image_tools;
pub(crate) mod knowledge_builder_runtime;
mod live_execution_process;
mod mcp_bridges;
mod mcp_resource_tools;
mod memory_tools;
mod mention_selection;
mod model_capability;
mod model_registry_metadata;
mod model_route_contract;
pub(crate) mod model_route_credential;
mod model_route_resolver;
mod model_routing;
mod native_tools;
mod permission_preflight;
mod plan_events;
mod plugin_activation_context;
mod plugin_runtime_context;
mod plugin_worker_generation;
mod proposed_plan_parser;
mod provider_config;
mod reasoning_events;
mod skill_runtime_enable;
mod tool_events;
mod tool_inventory;
pub(crate) mod tool_process_external_metadata;
mod tool_process_kind_metadata;
pub(crate) mod tool_process_metadata;
mod tool_process_risk_metadata;
mod tool_process_runtime_metadata;
mod tool_search_tools;
mod workspace_patch_host_execution;
mod workspace_patch_host_tools;

use crate::execution_process::ExecutionProcessServer;
use crate::AppDataSource;
use crate::ExecutionRequest;
use crate::RuntimeCoreError;
use crate::RuntimeEvent;
use crate::RuntimeEventSink;
use agent_runtime::session_loop::RuntimeSessionInputHandle;
use app_server_protocol::{RouteFailure, RouteFailureCategory};
use lime_agent::{
    run_agent_turn_with_policy, AgentRuntimeState, AgentTurnExecutionRequest,
    AgentTurnProviderConfiguration,
};
use lime_core::database::DbConnection;
use lime_services::api_key_provider_service::ApiKeyProviderService;
use model_provider::current_client::CurrentProviderMessage;
use serde_json::{json, Value};
use std::sync::Arc;
use std::sync::RwLock;
use tokio_util::sync::CancellationToken;

mod request_context;

pub(crate) use provider_config::current_agent_runtime_config_metadata;
use provider_config::{initialize_runtime_database, model_effective_event_from_runtime};
use request_context::{
    apply_app_server_turn_policy, direct_provider_config_from_request,
    request_tool_policy_from_request, resolve_runtime_model_selection,
    runtime_request_from_request, selection_with_effective_reasoning, session_config_from_request,
    session_scope_from_request, should_use_compact_tool_surface,
};

#[cfg(test)]
use app_server_protocol::AgentSessionActionType;
#[cfg(test)]
use event_mapper::emit_runtime_agent_event_with_coding_mirror;
use event_mapper::{
    emit_agent_message_finish, emit_reasoning_finish,
    emit_runtime_agent_event_with_coding_mirror_and_plan_parser_with_soul_style,
};

#[derive(Default)]
pub struct RuntimeBackend {
    agent_state: AgentRuntimeState,
    api_key_provider_service: ApiKeyProviderService,
    db: Option<DbConnection>,
    app_data_source: Arc<RwLock<Option<Arc<dyn AppDataSource>>>>,
    live_execution_process: Option<ExecutionProcessServer>,
}

struct ResolvedTurnRoute {
    db: DbConnection,
    requested_selection: request_context::RuntimeModelSelection,
    selection: request_context::RuntimeModelSelection,
    direct_provider_config: Option<lime_agent::SessionProviderConfig>,
    resolution: model_route_resolver::ChatModelRouteResolution,
    effective_generation: u64,
}

fn agent_control_route_snapshot_for_resolved_route(
    backend: &RuntimeBackend,
    route: &ResolvedTurnRoute,
) -> Result<Value, RuntimeCoreError> {
    let route_protocol = serde_json::to_value(&route.resolution.resolved_route.protocol)
        .map_err(|error| RuntimeCoreError::Backend(format!("serialize route protocol: {error}")))?;
    let provider_name = route
        .direct_provider_config
        .as_ref()
        .map(|config| config.provider_name.clone())
        .or_else(|| {
            backend
                .api_key_provider_service
                .get_provider(&route.db, &route.selection.provider)
                .ok()
                .flatten()
                .map(|provider| provider.provider.name)
        })
        .unwrap_or_else(|| route.selection.provider.clone());
    let auth = &route.resolution.resolved_route.auth;
    let direct_provider_config = route.direct_provider_config.as_ref();

    Ok(json!({
        "schemaVersion": 2,
        "providerPreference": route.selection.provider,
        "modelPreference": route.selection.model,
        "providerConfig": {
            "providerId": route.selection.provider,
            "providerName": provider_name,
            "modelName": route.selection.model,
            "reasoningEffort": route.selection.reasoning_effort,
            "toolshim": direct_provider_config.map(|config| config.toolshim),
            "toolshimModel": direct_provider_config
                .and_then(|config| config.toolshim_model.as_deref()),
            "supportsWebsockets": direct_provider_config
                .map(|config| config.supports_websockets)
        },
        "routeProtocol": route_protocol,
        "authKind": auth.kind,
        "credentialRef": auth.credential_ref,
        "effectiveGeneration": route.effective_generation
    }))
}

fn read_route_generation(db: &DbConnection) -> Result<u64, RuntimeCoreError> {
    let connection = db
        .lock()
        .map_err(|_| RuntimeCoreError::Backend("runtime database lock poisoned".to_string()))?;
    lime_core::database::dao::route_state::RouteStateDao::read_generation(&connection)
        .map_err(|error| RuntimeCoreError::Backend(format!("read route generation: {error}")))
}

fn durable_credential_ref_for_generation<'a>(
    request: &'a ExecutionRequest,
    selection: &request_context::RuntimeModelSelection,
    generation: u64,
) -> Option<&'a str> {
    let route = request
        .runtime_request()?
        .metadata
        .as_ref()?
        .get("agentControlRoute")?;
    (route.get("schemaVersion").and_then(Value::as_u64) == Some(2)
        && route.get("effectiveGeneration").and_then(Value::as_u64) == Some(generation)
        && route.get("providerPreference").and_then(Value::as_str)
            == Some(selection.provider.as_str())
        && route.get("modelPreference").and_then(Value::as_str) == Some(selection.model.as_str()))
    .then(|| route.get("credentialRef").and_then(Value::as_str))
    .flatten()
    .map(str::trim)
    .filter(|value| !value.is_empty())
}

fn runtime_error_from_route_failure(
    session_id: &str,
    selection: &request_context::RuntimeModelSelection,
    failure: &RouteFailure,
) -> RuntimeCoreError {
    let pending_after_route_generation_change = match &failure.category {
        RouteFailureCategory::NoCandidate => {
            matches!(
                failure.reason_code.as_str(),
                "no_candidate" | "routing_no_candidate"
            )
        }
        RouteFailureCategory::CapabilityGap => false,
        RouteFailureCategory::ProviderNeedsSetup => {
            failure.reason_code == "provider_not_configured"
        }
        RouteFailureCategory::ProviderDisabled => failure.reason_code == "provider_disabled",
        RouteFailureCategory::MissingCredential => failure.reason_code == "missing_enabled_api_key",
        RouteFailureCategory::ModelUnavailable => matches!(
            failure.reason_code.as_str(),
            "model_registry_metadata_missing" | "provider_models_cache_missing_requested_model"
        ),
        RouteFailureCategory::UnsupportedProtocol
        | RouteFailureCategory::UnsupportedEndpoint
        | RouteFailureCategory::InternalError => false,
    };
    if pending_after_route_generation_change {
        return RuntimeCoreError::PendingRoute {
            session_id: session_id.to_string(),
            provider: failure
                .provider_id
                .clone()
                .or_else(|| Some(selection.provider.clone())),
            model: failure
                .model_id
                .clone()
                .or_else(|| Some(selection.model.clone())),
            reason_code: failure.reason_code.clone(),
        };
    }

    if matches!(
        &failure.category,
        RouteFailureCategory::CapabilityGap
            | RouteFailureCategory::UnsupportedProtocol
            | RouteFailureCategory::UnsupportedEndpoint
    ) {
        return RuntimeCoreError::RouteRejected {
            session_id: session_id.to_string(),
            provider: failure
                .provider_id
                .clone()
                .or_else(|| Some(selection.provider.clone())),
            model: failure
                .model_id
                .clone()
                .or_else(|| Some(selection.model.clone())),
            category: failure.category.clone(),
            reason_code: failure.reason_code.clone(),
        };
    }

    RuntimeCoreError::Backend(format!(
        "App Server runtime backend route resolution failed: category={:?}, reason={}, provider={:?}, model={:?}, capability_gap={:?}",
        failure.category,
        failure.reason_code,
        failure.provider_id,
        failure.model_id,
        failure.capability_gap,
    ))
}

impl RuntimeBackend {
    pub fn new() -> Self {
        Self::build(None, None)
    }

    pub fn with_db(db: DbConnection) -> Self {
        Self::build(Some(db), None)
    }

    pub(crate) fn with_execution_process_server(execution_process: ExecutionProcessServer) -> Self {
        Self::build(None, Some(execution_process))
    }

    pub(crate) fn with_db_and_execution_process_server(
        db: DbConnection,
        execution_process: ExecutionProcessServer,
    ) -> Self {
        Self::build(Some(db), Some(execution_process))
    }

    fn build(
        db: Option<DbConnection>,
        live_execution_process: Option<ExecutionProcessServer>,
    ) -> Self {
        Self {
            agent_state: AgentRuntimeState::new(),
            api_key_provider_service: ApiKeyProviderService::new(),
            db,
            app_data_source: Arc::new(RwLock::new(None)),
            live_execution_process,
        }
    }

    async fn install_live_execution_process_hook_if_available(
        &self,
    ) -> Result<(), RuntimeCoreError> {
        let Some(execution_process) = self.live_execution_process.clone() else {
            return Ok(());
        };
        self.agent_state
            .install_live_execution_process_gateway(Arc::new(execution_process))
            .await
            .map_err(backend_error)
    }

    async fn register_current_native_tools_if_available(&self) -> Result<(), RuntimeCoreError> {
        native_tools::register_current_native_tools_if_available(
            &self.agent_state,
            &self.app_data_source,
        )
        .await
    }

    async fn ensure_agent_initialized(&self, db: &DbConnection) -> Result<(), RuntimeCoreError> {
        self.agent_state
            .init_agent_with_db(db)
            .await
            .map_err(backend_error)
    }

    async fn resolve_turn_route(
        &self,
        request: &ExecutionRequest,
    ) -> Result<ResolvedTurnRoute, RuntimeCoreError> {
        let db = initialize_runtime_database(self.db.as_ref())?;
        let requested_selection = resolve_runtime_model_selection(request)?;
        let effective_requested_selection =
            selection_with_effective_reasoning(&requested_selection);
        let host_request = runtime_request_from_request(request);
        let direct_provider_config = direct_provider_config_from_request(
            host_request.as_ref(),
            &effective_requested_selection,
            effective_requested_selection.reasoning_effort.clone(),
        );
        let mut retry_credential_binding: Option<(String, String, String)> = None;
        for _ in 0..3 {
            let generation_before = read_route_generation(&db)?;
            let prepared = model_route_resolver::prepare_chat_model_route(
                &db,
                &self.api_key_provider_service,
                request,
                &effective_requested_selection,
                direct_provider_config.as_ref(),
            )
            .map_err(backend_error)?;
            let prepared_selection = selection_with_effective_reasoning(prepared.selection());
            let durable_credential_ref = if direct_provider_config.is_none() {
                durable_credential_ref_for_generation(
                    request,
                    &prepared_selection,
                    generation_before,
                )
            } else {
                None
            };
            let retry_credential_ref = retry_credential_binding
                .as_ref()
                .filter(|(provider, model, _)| {
                    provider == &prepared_selection.provider && model == &prepared_selection.model
                })
                .map(|(_, _, credential_ref)| credential_ref.as_str());
            let resolution = model_route_resolver::assemble_chat_model_route(
                &db,
                &self.api_key_provider_service,
                request,
                &effective_requested_selection,
                direct_provider_config.as_ref(),
                prepared,
                durable_credential_ref.or(retry_credential_ref),
            )
            .await
            .map_err(|error| {
                if error == "resolved_credential_unavailable" {
                    RuntimeCoreError::PendingRoute {
                        session_id: request.session.session_id.clone(),
                        provider: Some(prepared_selection.provider.clone()),
                        model: Some(prepared_selection.model.clone()),
                        reason_code: error,
                    }
                } else {
                    backend_error(error)
                }
            })?;
            let generation_after = read_route_generation(&db)?;
            if generation_before != generation_after {
                retry_credential_binding = resolution
                    .resolved_route
                    .auth
                    .credential_ref
                    .as_ref()
                    .map(|credential_ref| {
                        (
                            prepared_selection.provider.clone(),
                            prepared_selection.model.clone(),
                            credential_ref.clone(),
                        )
                    });
                continue;
            }
            let selection = selection_with_effective_reasoning(&resolution.selection);
            return Ok(ResolvedTurnRoute {
                db,
                requested_selection,
                selection,
                direct_provider_config,
                resolution,
                effective_generation: generation_after,
            });
        }

        Err(RuntimeCoreError::Backend(
            "model route generation changed repeatedly during route resolution".to_string(),
        ))
    }

    async fn prepare_turn_route(
        &self,
        request: &ExecutionRequest,
        first_sampling_turn: bool,
    ) -> Result<Option<app_server_protocol::RuntimeOptions>, RuntimeCoreError> {
        let session_scope = session_scope_from_request(request)?;
        if image_command::is_image_command_turn(request, &session_scope)? {
            return Ok(request.runtime_options.clone());
        }
        // Permission preflight must run before provider route selection. A browser-control
        // confirmation is a local safety boundary and must remain actionable even when no
        // provider/model has been selected yet.
        let host_request = runtime_request_from_request(request);
        if permission_preflight::browser_control_permission_event(
            request,
            host_request.as_ref(),
            &session_scope,
        )
        .is_some()
        {
            return Ok(request.runtime_options.clone());
        }
        let mut route_request = request.clone();
        let initial_host_request = runtime_request_from_request(&route_request);
        let initial_tool_policy = request_tool_policy_from_request(initial_host_request.as_ref());
        apply_app_server_turn_policy(
            &mut route_request,
            first_sampling_turn,
            &initial_tool_policy,
        );
        let route = self.resolve_turn_route(&route_request).await?;
        if let Some(route_failure) = route.resolution.resolved_route.failure.as_ref() {
            return Err(runtime_error_from_route_failure(
                &route_request.session.session_id,
                &route.selection,
                route_failure,
            ));
        }
        let snapshot = agent_control_route_snapshot_for_resolved_route(self, &route)?;
        let mut options = request.runtime_options.clone().unwrap_or_default();
        let runtime_request = options.runtime_request_mut();
        let metadata = runtime_request
            .metadata
            .get_or_insert_with(|| Value::Object(Default::default()));
        if !metadata.is_object() {
            *metadata = Value::Object(Default::default());
        }
        metadata
            .as_object_mut()
            .expect("runtime metadata object")
            .insert("agentControlRoute".to_string(), snapshot);
        Ok(Some(options))
    }

    async fn handle_turn_start(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.handle_turn_start_with_provider_history(request, Vec::new(), None, None, sink)
            .await
    }

    async fn handle_turn_start_with_provider_history(
        &self,
        mut request: ExecutionRequest,
        provider_history: Vec<CurrentProviderMessage>,
        pending_input: Option<RuntimeSessionInputHandle>,
        cancellation_token: Option<CancellationToken>,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let session_scope = session_scope_from_request(&request)?;
        if image_command::handle_image_command_turn_if_present(
            Some(self),
            &request,
            &session_scope,
            self.current_app_data_source()?,
            sink,
        )
        .await?
        {
            return Ok(());
        }
        let initial_host_request = runtime_request_from_request(&request);
        let initial_tool_policy = request_tool_policy_from_request(initial_host_request.as_ref());
        apply_app_server_turn_policy(
            &mut request,
            provider_history.is_empty(),
            &initial_tool_policy,
        );
        let host_request = runtime_request_from_request(&request);
        let request_tool_policy = request_tool_policy_from_request(host_request.as_ref());
        let compact_tool_surface = should_use_compact_tool_surface(&request);
        let _skill_runtime_enable_guard =
            skill_runtime_enable::apply_workspace_skill_runtime_enable(
                &request,
                &session_scope.session_id,
            );
        let agent_skill_events =
            agent_skills_telemetry::runtime_status_events_for_agent_skills(&request);
        for event in agent_skill_events {
            sink.emit(event)?;
        }
        if let Some(event) = permission_preflight::browser_control_permission_event(
            &request,
            host_request.as_ref(),
            &session_scope,
        ) {
            match event {
                permission_preflight::PermissionPreflightOutcome::Required(event) => {
                    sink.emit(event)?;
                    return Ok(());
                }
                permission_preflight::PermissionPreflightOutcome::Cached(event) => {
                    sink.emit(event)?;
                }
            }
        }
        let ResolvedTurnRoute {
            db,
            requested_selection,
            selection,
            direct_provider_config,
            resolution: route_resolution,
            ..
        } = self.resolve_turn_route(&request).await?;

        sink.emit(RuntimeEvent::new(
            "routing.decision.made",
            route_resolution.decision_payload.clone(),
        ))?;
        if let Some(payload) = route_resolution.fallback_payload.as_ref() {
            sink.emit(RuntimeEvent::new(
                "routing.fallback.applied",
                payload.clone(),
            ))?;
        }
        if let Some(route_failure) = route_resolution.resolved_route.failure.as_ref() {
            sink.emit(RuntimeEvent::new(
                "routing.not_possible",
                route_resolution
                    .not_possible_payload
                    .clone()
                    .unwrap_or_else(|| route_resolution.decision_payload.clone()),
            ))?;
            return Err(runtime_error_from_route_failure(
                &request.session.session_id,
                &selection,
                route_failure,
            ));
        }

        self.ensure_agent_initialized(&db).await?;
        self.install_live_execution_process_hook_if_available()
            .await?;
        if !compact_tool_surface {
            self.register_current_native_tools_if_available().await?;
            mcp_bridges::ensure_thread_mcp_runtime_if_available(
                &self.agent_state,
                &self.app_data_source,
                &session_scope.session_id,
                &session_scope.thread_id,
            )
            .await?;
        }
        let config_metadata = current_agent_runtime_config_metadata();
        let soul_style = tool_process_metadata::SoulStyleMetadata::from_config_metadata(
            config_metadata.as_ref(),
        );
        let mention_selection =
            mention_selection::resolve_mentions(&request, self.current_app_data_source()?).await;
        let mut session_config = session_config_from_request(
            &request,
            host_request.as_ref(),
            &session_scope,
            &selection,
            &request_tool_policy,
            config_metadata,
        );
        mention_selection.apply_to_session_config(&mut session_config);
        let model_context_window = lime_agent::model_request_policy_from_turn_context(
            session_config.turn_context.as_ref(),
        )
        .and_then(|policy| policy.context_policy)
        .and_then(|policy| policy.model_context_window);
        let mut emit_error = None;
        let mut coding_event_mirror = coding_events::CodingEventMirror::default();
        let mut proposed_plan_parser = proposed_plan_parser::ProposedPlanParser::default();
        let mut reasoning_event_state = reasoning_events::ReasoningEventState::default();
        let mut turn_usage = None;
        let execution_result = run_agent_turn_with_policy(
            &self.agent_state,
            AgentTurnExecutionRequest {
                session_id: &session_scope.session_id,
                input: request.input.clone(),
                initial_messages: provider_history,
                session_config,
                request_tool_policy: &request_tool_policy,
                provider_configuration: Some(AgentTurnProviderConfiguration {
                    db: &db,
                    session_id: &session_scope.session_id,
                    route_configuration: model_route_contract::provider_configuration_from_runtime(
                        &selection,
                        &route_resolution.resolved_route,
                        direct_provider_config,
                    ),
                    credential_ref: route_resolution
                        .resolved_route
                        .auth
                        .credential_ref
                        .as_deref(),
                }),
                agent_control_gateway: request.agent_control_gateway.clone(),
                pending_input,
                cancellation_token,
            },
            |event| {
                if let lime_agent::AgentEvent::Done { usage } = event {
                    turn_usage = usage.clone();
                }
                if emit_error.is_some() {
                    return;
                }
                if let Err(error) =
                    emit_runtime_agent_event_with_coding_mirror_and_plan_parser_with_soul_style(
                        event,
                        sink,
                        &mut coding_event_mirror,
                        &mut proposed_plan_parser,
                        &mut reasoning_event_state,
                        soul_style.as_ref(),
                    )
                {
                    emit_error = Some(error);
                }
            },
        )
        .await;
        let turn_execution = match execution_result {
            Ok(turn_execution) => turn_execution,
            Err(error) => {
                if let Some(error) = emit_error {
                    return Err(error);
                }
                emit_reasoning_finish(&mut reasoning_event_state, "failed", sink)?;
                emit_agent_message_finish(&mut proposed_plan_parser, "failed", sink)?;
                return Err(runtime_error_from_reply_attempt(error));
            }
        };
        let provider_config = turn_execution.provider_config.ok_or_else(|| {
            RuntimeCoreError::Backend(
                "App Server runtime backend expected provider configuration for main turn"
                    .to_string(),
            )
        })?;
        let execution = turn_execution.stream;
        if let Some(error) = emit_error {
            return Err(error);
        }
        sink.emit(model_effective_event_from_runtime(
            &requested_selection,
            &selection,
            &provider_config,
            route_resolution.service_model_slot(),
            &route_resolution.resolved_route.capability_snapshot,
        ))?;
        if execution.cancelled {
            emit_reasoning_finish(&mut reasoning_event_state, "canceled", sink)?;
            emit_agent_message_finish(&mut proposed_plan_parser, "interrupted", sink)?;
            sink.emit(RuntimeEvent::new(
                "turn.canceled",
                json!({
                    "backend": "runtime",
                    "model": provider_config.model_name,
                    "provider": provider_config
                        .provider_selector
                        .as_deref()
                        .unwrap_or(&selection.provider),
                    "searchMode": request_tool_policy.search_mode.as_str(),
                    "attempts": execution.attempts_summary,
                }),
            ))?;
            return Ok(());
        }

        emit_reasoning_finish(&mut reasoning_event_state, "completed", sink)?;
        emit_agent_message_finish(&mut proposed_plan_parser, "completed", sink)?;
        sink.emit(RuntimeEvent::new(
            "turn.completed",
            json!({
                "backend": "runtime",
                "model": provider_config.model_name,
                "provider": provider_config
                    .provider_selector
                    .as_deref()
                    .unwrap_or(&selection.provider),
                "searchMode": request_tool_policy.search_mode.as_str(),
                "attempts": execution.attempts_summary,
                "usage": turn_usage,
                "modelContextWindow": model_context_window,
            }),
        ))?;

        Ok(())
    }

    fn current_app_data_source(&self) -> Result<Option<Arc<dyn AppDataSource>>, RuntimeCoreError> {
        self.app_data_source
            .read()
            .map_err(|_| {
                RuntimeCoreError::Backend(
                    "runtime backend app data source lock poisoned".to_string(),
                )
            })
            .map(|guard| guard.clone())
    }
}

fn backend_error(error: impl std::fmt::Display) -> RuntimeCoreError {
    RuntimeCoreError::Backend(error.to_string())
}

fn runtime_error_from_reply_attempt(error: lime_agent::ReplyAttemptError) -> RuntimeCoreError {
    if error.is_usage_limit_exceeded() {
        RuntimeCoreError::UsageLimitExceeded(error.message)
    } else {
        RuntimeCoreError::Backend(error.message)
    }
}

#[cfg(test)]
mod initialization_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod websocket_fallback_tests;
