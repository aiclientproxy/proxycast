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
mod model_candidate_set;
mod model_capability;
mod model_registry_metadata;
mod model_route_contract;
pub(crate) mod model_route_credential;
mod model_route_resolver;
mod model_routing;
mod native_tools;
mod orchestrator_skills;
mod plan_events;
mod proposed_plan_parser;
mod provider_config;
mod reasoning_events;
mod route_support;
mod skill_runtime_enable;
mod tool_events;
mod tool_inventory;
pub(crate) mod tool_process_external_metadata;
mod tool_process_kind_metadata;
pub(crate) mod tool_process_metadata;
mod tool_process_risk_metadata;
mod tool_process_runtime_metadata;
mod tool_search_tools;

use crate::execution_process::ExecutionProcessServer;
use crate::AppDataSource;
use crate::ExecutionRequest;
use crate::RuntimeCoreError;
use crate::RuntimeEvent;
use crate::RuntimeEventSink;
use agent_runtime::session_loop::RuntimeSessionInputHandle;
use lime_agent::{
    run_agent_turn_with_policy, AgentRuntimeState, AgentTurnExecutionRequest,
    AgentTurnProviderConfiguration,
};
use lime_core::database::DbConnection;
use lime_services::api_key_provider_service::ApiKeyProviderService;
use runtime_core::ModelRouteExclusion;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::RwLock;
use tokio_util::sync::CancellationToken;

mod request_context;

pub(crate) use model_routing::configured_provider_readiness;
pub(crate) use provider_config::current_agent_runtime_config_metadata;
use provider_config::{initialize_runtime_database, model_effective_event_from_runtime};
#[cfg(test)]
use request_context::session_config_from_request;
use request_context::{
    apply_app_server_turn_policy, direct_provider_config_from_request,
    request_tool_policy_from_request, resolve_runtime_model_selection,
    runtime_request_from_request, service_tier_from_request,
    session_config_from_request_with_plugins_and_orchestrator, session_scope_from_request,
    should_use_compact_tool_surface,
};
use route_support::{
    agent_control_route_snapshot_for_resolved_route, durable_credential_ref_for_generation,
    read_route_generation, runtime_error_from_route_failure, runtime_route_exclusion,
};

#[cfg(test)]
use app_server_protocol::AgentSessionActionType;
#[cfg(test)]
use event_mapper::emit_runtime_agent_event_with_coding_mirror;
use event_mapper::{
    emit_agent_message_finish, emit_reasoning_finish,
    emit_runtime_agent_event_with_coding_mirror_and_plan_parser_with_soul_style,
    ModelRouteEvidence,
};

#[derive(Default)]
pub struct RuntimeBackend {
    agent_state: AgentRuntimeState,
    api_key_provider_service: ApiKeyProviderService,
    db: Option<DbConnection>,
    app_data_source: Arc<RwLock<Option<Arc<dyn AppDataSource>>>>,
    current_time_gateway:
        Arc<RwLock<Option<Arc<dyn tool_runtime::current_time::CurrentTimeGateway>>>>,
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
            current_time_gateway: Arc::new(RwLock::new(None)),
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
            &self.current_time_gateway,
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
        self.resolve_turn_route_excluding(request, &[]).await
    }

    async fn resolve_turn_route_excluding(
        &self,
        request: &ExecutionRequest,
        excluded_routes: &[ModelRouteExclusion],
    ) -> Result<ResolvedTurnRoute, RuntimeCoreError> {
        let db = initialize_runtime_database(self.db.as_ref())?;
        let requested_selection = resolve_runtime_model_selection(request)?;
        let host_request = runtime_request_from_request(request);
        let direct_provider_config = direct_provider_config_from_request(
            host_request.as_ref(),
            &requested_selection,
            requested_selection.reasoning_effort.clone(),
        );
        let mut retry_credential_binding: Option<(String, String, String)> = None;
        for _ in 0..3 {
            let generation_before = read_route_generation(&db)?;
            let prepared = model_route_resolver::prepare_chat_model_route(
                &db,
                &self.api_key_provider_service,
                request,
                &requested_selection,
                direct_provider_config.as_ref(),
                excluded_routes,
            )
            .map_err(backend_error)?;
            let prepared_selection = prepared.selection().clone();
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
                &requested_selection,
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
            let selection = resolution.selection.clone();
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

    async fn preflight_thread_settings_route(
        &self,
        session: &app_server_protocol::AgentSession,
        settings: &app_server_protocol::protocol::v2::ThreadSettings,
    ) -> Result<(), RuntimeCoreError> {
        execution_backend::preflight_thread_settings_route(self, session, settings).await
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
        let snapshot = agent_control_route_snapshot_for_resolved_route(
            self,
            &route,
            route_request
                .runtime_request()
                .and_then(|request| request.service_tier.as_deref()),
        )?;
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
        self.handle_turn_start_with_provider_history(
            request,
            crate::runtime::provider_history::ProviderTurnHistory::default(),
            None,
            None,
            sink,
        )
        .await
    }

    async fn handle_turn_start_with_provider_history(
        &self,
        mut request: ExecutionRequest,
        provider_history: crate::runtime::provider_history::ProviderTurnHistory,
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
        let plugin_snapshots = self.current_plugin_turn_snapshots().await?;
        let agent_skill_events = agent_skills_telemetry::runtime_status_events_for_agent_skills(
            &request,
            &plugin_snapshots,
        );
        for event in agent_skill_events {
            sink.emit(event)?;
        }
        let mut excluded_routes = Vec::new();
        let mut previous_reply_error = None;
        let mut resolved_turn_route = self
            .resolve_turn_route_excluding(&request, &excluded_routes)
            .await?;

        let config_metadata = current_agent_runtime_config_metadata();
        let soul_style = tool_process_metadata::SoulStyleMetadata::from_config_metadata(
            config_metadata.as_ref(),
        );
        let mention_selection = mention_selection::resolve_mentions(
            &request,
            self.current_app_data_source()?,
            &plugin_snapshots,
        )
        .await;
        let turn_plugin_snapshots = mention_selection.plugin_snapshots_for_turn(&plugin_snapshots);
        let mut emit_error = None;
        let mut coding_event_mirror = coding_events::CodingEventMirror::default();
        let mut proposed_plan_parser = proposed_plan_parser::ProposedPlanParser::default();
        let mut reasoning_event_state = reasoning_events::ReasoningEventState::default();
        let mut turn_usage = None;
        let mut model_reroute_emitted = false;
        let mut model_verification_emitted = false;
        let mut server_model_evidence_keys = HashSet::new();
        let mut runtime_initialized = false;
        let mut orchestrator_skill_discovery = None;
        let (
            turn_execution,
            provider_config,
            requested_selection,
            selection,
            route_resolution,
            model_context_window,
        ) = loop {
            let ResolvedTurnRoute {
                db,
                requested_selection,
                selection,
                direct_provider_config,
                resolution: route_resolution,
                ..
            } = resolved_turn_route;

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
                if let Some(error) = previous_reply_error {
                    emit_reasoning_finish(&mut reasoning_event_state, "failed", sink)?;
                    emit_agent_message_finish(&mut proposed_plan_parser, "failed", sink)?;
                    return Err(runtime_error_from_reply_attempt(error));
                }
                return Err(runtime_error_from_route_failure(
                    &request.session.session_id,
                    &selection,
                    route_failure,
                ));
            }

            if !runtime_initialized {
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
                orchestrator_skill_discovery = Some(
                    orchestrator_skills::discover_for_turn(
                        &self.agent_state,
                        &session_scope.session_id,
                        &session_scope.thread_id,
                        config_metadata.as_ref(),
                    )
                    .await,
                );
                if let Some(discovery) = orchestrator_skill_discovery.as_ref() {
                    for warning in &discovery.warnings {
                        tracing::warn!(
                            session_id = %session_scope.session_id,
                            thread_id = %session_scope.thread_id,
                            warning,
                            "Orchestrator Skill discovery warning"
                        );
                    }
                }
                runtime_initialized = true;
            }

            let turn_config_metadata = merge_route_model_metadata(
                config_metadata.clone(),
                route_resolution.decision_payload.get("modelRegistry"),
            );
            let mut session_config = session_config_from_request_with_plugins_and_orchestrator(
                &request,
                host_request.as_ref(),
                &session_scope,
                &selection,
                &request_tool_policy,
                turn_config_metadata,
                &turn_plugin_snapshots,
                orchestrator_skill_discovery
                    .as_ref()
                    .map(|discovery| discovery.skills.as_slice())
                    .unwrap_or_default(),
            );
            session_config.tool_mode = route_resolution.tool_mode;
            session_config.supports_custom_tools = route_resolution.supports_custom_tools;
            mention_selection.apply_to_session_config(&mut session_config);
            let model_context_window = lime_agent::model_request_policy_from_turn_context(
                session_config.turn_context.as_ref(),
            )
            .and_then(|policy| policy.context_policy)
            .and_then(|policy| policy.model_context_window);
            let model_route_evidence = ModelRouteEvidence {
                provider: selection.provider.clone(),
                requested_model: requested_selection.model.clone(),
                selected_model: selection.model.clone(),
                route_attempt: excluded_routes.len() + 1,
            };
            session_config.rollout_budget_reminder_source = session_config
                .rollout_budget_reminder_source
                .take()
                .map(|source| source.with_route_attempt(model_route_evidence.route_attempt));
            let event_failure_cancellation = cancellation_token.clone();
            let execution_result = run_agent_turn_with_policy(
                &self.agent_state,
                AgentTurnExecutionRequest {
                    session_id: &session_scope.session_id,
                    input: request.input.clone(),
                    initial_messages: provider_history
                        .messages_for_route(&selection.provider, &selection.model),
                    session_config,
                    request_tool_policy: &request_tool_policy,
                    provider_configuration: Some(AgentTurnProviderConfiguration {
                        db: &db,
                        session_id: &session_scope.session_id,
                        route_configuration:
                            model_route_contract::provider_configuration_from_runtime(
                                &selection,
                                &route_resolution.resolved_route,
                                route_resolution.decision_payload.get("modelRegistry"),
                                direct_provider_config.clone(),
                                service_tier_from_request(&request),
                            ),
                        credential_ref: route_resolution
                            .resolved_route
                            .auth
                            .credential_ref
                            .as_deref(),
                    }),
                    agent_control_gateway: request.agent_control_gateway.clone(),
                    pending_input: pending_input.clone(),
                    cancellation_token: cancellation_token.clone(),
                },
                |event| {
                    if let lime_agent::AgentEvent::Done { usage } = event {
                        turn_usage = usage.clone();
                    }
                    if matches!(event, lime_agent::AgentEvent::ModelVerification { .. }) {
                        if model_verification_emitted {
                            return;
                        }
                        model_verification_emitted = true;
                    }
                    if matches!(event, lime_agent::AgentEvent::ModelReroute { .. }) {
                        if model_reroute_emitted {
                            return;
                        }
                        model_reroute_emitted = true;
                    }
                    if let lime_agent::AgentEvent::ServerModel { model } = event {
                        let key = (
                            model_route_evidence.provider.clone(),
                            model_route_evidence.selected_model.clone(),
                            model_route_evidence.route_attempt,
                            model.to_ascii_lowercase(),
                        );
                        if !server_model_evidence_keys.insert(key) {
                            return;
                        }
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
                            Some(&model_route_evidence),
                        )
                    {
                        if let Some(cancellation_token) = event_failure_cancellation.as_ref() {
                            cancellation_token.cancel();
                        }
                        emit_error = Some(error);
                    }
                },
            )
            .await;
            match execution_result {
                Ok(turn_execution) => {
                    let provider_config = turn_execution.provider_config.clone().ok_or_else(|| {
                        RuntimeCoreError::Backend(
                            "App Server runtime backend expected provider configuration for main turn"
                                .to_string(),
                        )
                    })?;
                    if let Some(error) = emit_error.take() {
                        return Err(error);
                    }
                    break (
                        turn_execution,
                        provider_config,
                        requested_selection,
                        selection,
                        route_resolution,
                        model_context_window,
                    );
                }
                Err(error) => {
                    if let Some(error) = emit_error.take() {
                        return Err(error);
                    }
                    let Some(exclusion) = runtime_route_exclusion(
                        &selection,
                        direct_provider_config.is_some(),
                        route_resolution
                            .resolved_route
                            .auth
                            .credential_ref
                            .as_deref(),
                        &error,
                    ) else {
                        emit_reasoning_finish(&mut reasoning_event_state, "failed", sink)?;
                        emit_agent_message_finish(&mut proposed_plan_parser, "failed", sink)?;
                        return Err(runtime_error_from_reply_attempt(error));
                    };
                    if let (Some(credential_ref), Some(retry_after)) =
                        (exclusion.credential_ref(), error.retry_after())
                    {
                        self.api_key_provider_service
                            .cooldown_runtime_credential(credential_ref, retry_after)
                            .map_err(backend_error)?;
                    }
                    excluded_routes.push(exclusion);
                    previous_reply_error = Some(error);
                    resolved_turn_route = match self
                        .resolve_turn_route_excluding(&request, &excluded_routes)
                        .await
                    {
                        Ok(route) => route,
                        Err(_) => {
                            let error = previous_reply_error
                                .take()
                                .expect("runtime reroute preserves provider failure");
                            emit_reasoning_finish(&mut reasoning_event_state, "failed", sink)?;
                            emit_agent_message_finish(&mut proposed_plan_parser, "failed", sink)?;
                            return Err(runtime_error_from_reply_attempt(error));
                        }
                    };
                }
            }
        };
        let execution = turn_execution.stream;
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

    async fn current_plugin_turn_snapshots(
        &self,
    ) -> Result<Vec<crate::runtime::PluginTurnSnapshot>, RuntimeCoreError> {
        let Some(app_data_source) = self.current_app_data_source()? else {
            return Ok(Vec::new());
        };
        app_data_source.list_enabled_plugin_turn_snapshots().await
    }
}

fn merge_route_model_metadata(
    config_metadata: Option<serde_json::Value>,
    model_registry: Option<&serde_json::Value>,
) -> Option<serde_json::Value> {
    let Some(model_registry) = model_registry else {
        return config_metadata;
    };
    let mut metadata = match config_metadata {
        Some(serde_json::Value::Object(object)) => object,
        _ => serde_json::Map::new(),
    };
    metadata.insert("modelRegistry".to_string(), model_registry.clone());
    metadata.insert("model_registry".to_string(), model_registry.clone());
    Some(serde_json::Value::Object(metadata))
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
mod runtime_reroute_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod websocket_fallback_tests;
