use super::{
    action_response, current_agent_runtime_config_metadata, initialize_runtime_database,
    mcp_bridges, request_context::effective_runtime_options_for_turn, tool_inventory,
    RuntimeBackend,
};
use crate::runtime::ToolInventoryReadRequest;
use crate::{
    ActionRespondRequest, AppDataSource, CancelExecutionRequest, ExecutionBackend,
    ExecutionRequest, RuntimeCoreError, RuntimeEvent, RuntimeEventSink, RuntimeHostContext,
};
use agent_runtime::action_required::{ActionTerminalStatus, PendingActionRestoreOutcome};
use agent_runtime::session_loop::RuntimeSessionInputHandle;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub(super) async fn preflight_thread_settings_route(
    backend: &RuntimeBackend,
    session: &app_server_protocol::AgentSession,
    settings: &app_server_protocol::protocol::v2::ThreadSettings,
) -> Result<(), RuntimeCoreError> {
    let mut runtime_options = app_server_protocol::RuntimeOptions::default();
    let runtime_request = runtime_options.runtime_request_mut();
    runtime_request.provider_preference = Some(settings.model_provider.clone());
    runtime_request.model_preference = Some(settings.model.clone());
    runtime_request.reasoning_effort = settings.effort.clone();
    runtime_request.service_tier = settings.service_tier.clone();
    let request = ExecutionRequest {
        host: RuntimeHostContext::default(),
        session: session.clone(),
        turn: app_server_protocol::AgentTurn {
            turn_id: "thread-settings-preflight".to_string(),
            session_id: session.session_id.clone(),
            thread_id: session.thread_id.clone(),
            status: app_server_protocol::AgentTurnStatus::Accepted,
            started_at: None,
            completed_at: None,
        },
        forked_from_thread_id: None,
        input: agent_runtime::reply_input::RuntimeReplyInput::text(
            "thread settings route preflight",
        ),
        runtime_options: Some(runtime_options),
        expected_output: None,
        structured_output: None,
        output_schema: None,
        event_name: None,
        queued_turn_id: None,
        queue_if_busy: false,
        skip_pre_submit_resume: false,
        agent_control_gateway: None,
    };
    let route = backend.resolve_turn_route(&request).await?;
    if let Some(failure) = route.resolution.resolved_route.failure.as_ref() {
        return Err(super::runtime_error_from_route_failure(
            &session.session_id,
            &route.selection,
            failure,
        ));
    }
    if route.selection.provider != settings.model_provider
        || route.selection.model != settings.model
    {
        return Err(RuntimeCoreError::RouteRejected {
            session_id: session.session_id.clone(),
            provider: Some(settings.model_provider.clone()),
            model: Some(settings.model.clone()),
            category: app_server_protocol::RouteFailureCategory::NoCandidate,
            reason_code: "model_switch_fallback_not_allowed".to_string(),
        });
    }
    if settings.effort.is_some() && route.selection.reasoning_effort != settings.effort {
        return Err(RuntimeCoreError::RouteRejected {
            session_id: session.session_id.clone(),
            provider: Some(settings.model_provider.clone()),
            model: Some(settings.model.clone()),
            category: app_server_protocol::RouteFailureCategory::CapabilityGap,
            reason_code: "reasoning_effort_unsupported".to_string(),
        });
    }
    if let Some(service_tier) = settings.service_tier.as_deref() {
        if !route_supports_service_tier(&route.resolution.decision_payload, service_tier) {
            return Err(RuntimeCoreError::RouteRejected {
                session_id: session.session_id.clone(),
                provider: Some(settings.model_provider.clone()),
                model: Some(settings.model.clone()),
                category: app_server_protocol::RouteFailureCategory::CapabilityGap,
                reason_code: "service_tier_unsupported".to_string(),
            });
        }
    }
    Ok(())
}

fn route_supports_service_tier(decision_payload: &Value, requested: &str) -> bool {
    [
        "/modelRegistry/model/service_tiers",
        "/modelRegistry/model/serviceTiers",
    ]
    .into_iter()
    .find_map(|pointer| decision_payload.pointer(pointer).and_then(Value::as_array))
    .is_some_and(|tiers| {
        tiers.iter().any(|tier| {
            tier.as_str()
                .or_else(|| tier.get("id").and_then(Value::as_str))
                .is_some_and(|id| id == requested)
        })
    })
}

#[async_trait]
impl ExecutionBackend for RuntimeBackend {
    fn requires_provider_selection(&self) -> bool {
        true
    }

    fn has_live_session_responses(&self) -> bool {
        true
    }

    fn set_app_data_source(
        &self,
        app_data_source: Arc<dyn AppDataSource>,
    ) -> Result<(), RuntimeCoreError> {
        let mut guard = self.app_data_source.write().map_err(|_| {
            RuntimeCoreError::Backend("memory tool app data source lock poisoned".to_string())
        })?;
        *guard = Some(app_data_source);
        Ok(())
    }

    fn set_current_time_gateway(
        &self,
        gateway: Arc<dyn tool_runtime::current_time::CurrentTimeGateway>,
    ) -> Result<(), RuntimeCoreError> {
        let mut guard = self.current_time_gateway.write().map_err(|_| {
            RuntimeCoreError::Backend("current-time gateway lock poisoned".to_string())
        })?;
        *guard = Some(gateway);
        Ok(())
    }

    fn effective_turn_runtime_options(
        &self,
        request: &ExecutionRequest,
        first_sampling_turn: bool,
    ) -> Option<app_server_protocol::RuntimeOptions> {
        effective_runtime_options_for_turn(request, first_sampling_turn)
            .or_else(|| request.runtime_options.clone())
    }

    async fn preflight_turn(
        &self,
        request: &ExecutionRequest,
        first_sampling_turn: bool,
    ) -> Result<(), RuntimeCoreError> {
        self.prepare_turn_route(request, first_sampling_turn)
            .await
            .map(|_| ())
    }

    async fn preflight_thread_settings(
        &self,
        session: &app_server_protocol::AgentSession,
        settings: &app_server_protocol::protocol::v2::ThreadSettings,
    ) -> Result<(), RuntimeCoreError> {
        self.preflight_thread_settings_route(session, settings)
            .await
    }

    async fn prepare_turn_runtime_options(
        &self,
        request: &ExecutionRequest,
        first_sampling_turn: bool,
    ) -> Result<Option<app_server_protocol::RuntimeOptions>, RuntimeCoreError> {
        self.prepare_turn_route(request, first_sampling_turn).await
    }

    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.handle_turn_start(request, sink).await
    }

    async fn start_turn_with_provider_history(
        &self,
        request: ExecutionRequest,
        provider_history: crate::runtime::provider_history::ProviderTurnHistory,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.handle_turn_start_with_provider_history(request, provider_history, None, None, sink)
            .await
    }

    async fn start_turn_with_provider_history_and_session_input(
        &self,
        request: ExecutionRequest,
        provider_history: crate::runtime::provider_history::ProviderTurnHistory,
        pending_input: Option<RuntimeSessionInputHandle>,
        cancellation_token: Option<CancellationToken>,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.handle_turn_start_with_provider_history(
            request,
            provider_history,
            pending_input,
            cancellation_token,
            sink,
        )
        .await
    }

    async fn cancel_turn(
        &self,
        request: CancelExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.agent_state
            .cancel_session(&request.session.session_id)
            .await;
        sink.emit(RuntimeEvent::new(
            "turn.canceled",
            json!({ "backend": "runtime" }),
        ))
    }

    async fn close_session(
        &self,
        session_id: &str,
        thread_id: &str,
    ) -> Result<(), RuntimeCoreError> {
        self.agent_state.cancel_session(session_id).await;
        self.agent_state.close_provider_session(session_id).await;
        self.agent_state
            .close_mcp_runtime(session_id, thread_id)
            .await;
        Ok(())
    }

    async fn respond_action(
        &self,
        request: ActionRespondRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let db = initialize_runtime_database(self.db.as_ref())?;
        self.ensure_agent_initialized(&db).await?;
        action_response::validate_action_scope(&request)?;
        if !self
            .agent_state
            .contains_pending_action(&request.request_id)
            .await
        {
            let descriptor = request.pending_action_descriptor.clone().ok_or_else(|| {
                action_response_error("action_descriptor_invalid", &request.request_id)
            })?;
            let outcomes = self
                .agent_state
                .restore_pending_action_descriptors([descriptor])
                .await;
            match outcomes.as_slice() {
                [PendingActionRestoreOutcome::Restored]
                | [PendingActionRestoreOutcome::AlreadyPresent] => {}
                [PendingActionRestoreOutcome::Expired] => {
                    return Err(action_response_error("action_expired", &request.request_id));
                }
                [PendingActionRestoreOutcome::Terminal] => {
                    let code = match self
                        .agent_state
                        .terminal_action_status(&request.request_id)
                        .await
                    {
                        Some(ActionTerminalStatus::NotResumable) => "action_not_resumable",
                        Some(ActionTerminalStatus::ContinuationClosed) => {
                            "action_continuation_closed"
                        }
                        Some(ActionTerminalStatus::Expired) => "action_expired",
                        Some(ActionTerminalStatus::Canceled) => "action_canceled",
                        Some(ActionTerminalStatus::Resolved) => "action_already_resolved",
                        None => "action_terminal",
                    };
                    return Err(action_response_error(code, &request.request_id));
                }
                [PendingActionRestoreOutcome::Invalid] | _ => {
                    return Err(action_response_error(
                        "action_descriptor_invalid",
                        &request.request_id,
                    ));
                }
            }
        }
        match action_response::handle_action_response(&self.agent_state, &request).await? {
            action_response::ActionResponseOutcome::Resolved => {
                sink.emit(action_response::action_resolved_event(&request))
            }
            action_response::ActionResponseOutcome::Canceled => {
                sink.emit(action_response::action_canceled_event(&request))
            }
        }
    }

    async fn resolve_permission_action(
        &self,
        request: &crate::PermissionRespondRequest,
    ) -> Result<(), RuntimeCoreError> {
        let scope = lime_agent::AgentActionRequiredScope::from_parts(
            Some(request.session_id.clone()),
            Some(request.thread_id.clone()),
            Some(request.turn_id.clone()),
        )
        .ok_or_else(|| RuntimeCoreError::ActionResponse {
            code: "action_scope_missing".to_string(),
            request_id: request.request_id.clone(),
        })?;
        self.agent_state
            .resolve_permission_action(&request.session_id, &request.request_id, Some(scope))
            .await
            .map_err(|error| RuntimeCoreError::ActionResponse {
                code: error.code().to_string(),
                request_id: error.request_id().to_string(),
            })
    }

    async fn read_tool_inventory(
        &self,
        request: ToolInventoryReadRequest,
    ) -> Result<Value, RuntimeCoreError> {
        self.register_current_native_tools_if_available().await?;
        let app_data_source = self
            .app_data_source
            .read()
            .map_err(|_| {
                RuntimeCoreError::Backend(
                    "tool inventory app data source lock poisoned".to_string(),
                )
            })?
            .clone();
        tool_inventory::read_tool_inventory(
            &self.agent_state,
            request,
            current_agent_runtime_config_metadata(),
            app_data_source,
        )
        .await
    }

    async fn read_mcp_runtime_resource(
        &self,
        session_id: &str,
        thread_id: &str,
        server: &str,
        uri: &str,
    ) -> Result<app_server_protocol::protocol::v2::McpServerResourceReadResponse, RuntimeCoreError>
    {
        let db = initialize_runtime_database(self.db.as_ref())?;
        self.ensure_agent_initialized(&db).await?;
        mcp_bridges::ensure_thread_mcp_runtime_if_available(
            &self.agent_state,
            &self.app_data_source,
            session_id,
            thread_id,
        )
        .await?;
        let content = self
            .agent_state
            .read_mcp_resource(session_id, thread_id, server, uri)
            .await
            .map_err(RuntimeCoreError::Backend)?;
        let content = match (content.text, content.blob) {
            (Some(text), None) => Some(
                app_server_protocol::protocol::v2::McpServerResourceContent::Text {
                    uri: content.uri,
                    mime_type: content.mime_type,
                    text,
                    meta: content.meta,
                },
            ),
            (None, Some(blob)) => Some(
                app_server_protocol::protocol::v2::McpServerResourceContent::Blob {
                    uri: content.uri,
                    mime_type: content.mime_type,
                    blob,
                    meta: content.meta,
                },
            ),
            (None, None) => None,
            (Some(_), Some(_)) => {
                return Err(RuntimeCoreError::Backend(
                    "MCP resource response contained both text and blob".to_string(),
                ));
            }
        };
        Ok(
            app_server_protocol::protocol::v2::McpServerResourceReadResponse {
                contents: content.into_iter().collect(),
            },
        )
    }

    async fn call_mcp_runtime_tool(
        &self,
        session_id: &str,
        thread_id: &str,
        server: &str,
        tool: &str,
        arguments: Value,
    ) -> Result<lime_mcp::McpToolResult, RuntimeCoreError> {
        let db = initialize_runtime_database(self.db.as_ref())?;
        self.ensure_agent_initialized(&db).await?;
        mcp_bridges::ensure_thread_mcp_runtime_if_available(
            &self.agent_state,
            &self.app_data_source,
            session_id,
            thread_id,
        )
        .await?;
        self.agent_state
            .call_mcp_tool(session_id, thread_id, server, tool, arguments)
            .await
            .map_err(RuntimeCoreError::Backend)
    }
}

fn action_response_error(code: &str, request_id: &str) -> RuntimeCoreError {
    RuntimeCoreError::ActionResponse {
        code: code.to_string(),
        request_id: request_id.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::route_supports_service_tier;
    use serde_json::json;

    #[test]
    fn service_tier_preflight_uses_exact_catalog_ids() {
        let decision = json!({
            "modelRegistry": {
                "model": {
                    "service_tiers": [
                        {"id": "priority", "name": "Priority"},
                        "flex"
                    ]
                }
            }
        });

        assert!(route_supports_service_tier(&decision, "priority"));
        assert!(route_supports_service_tier(&decision, "flex"));
        assert!(!route_supports_service_tier(&decision, "default"));
        assert!(!route_supports_service_tier(&json!({}), "priority"));
    }
}
