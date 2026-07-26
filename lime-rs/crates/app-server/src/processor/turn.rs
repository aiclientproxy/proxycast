//! v2 turn command handlers.

use super::{
    dispatch_result, parse_params, to_jsonrpc_error,
    v2_notifications::{project_events, V2NotificationProjector},
    RequestProcessor, RpcDispatch,
};
use crate::processor::config_warning::ConfigWarningScope;
use agent_protocol::{AgentInput, ThreadId, ThreadTurnsView};
use app_server_protocol::protocol::v2::{
    AdditionalContextKind, Turn as V2Turn, TurnInterruptParams, TurnInterruptResponse,
    TurnStartParams, TurnStartResponse, TurnStatus as V2TurnStatus, TurnSteerParams,
    TurnSteerResponse, UserInput,
};
use app_server_protocol::{
    error_codes, AgentSessionTurnCancelParams, AgentSessionTurnStartResponse, AgentTurn,
    AgentTurnStatus, JsonRpcError, JsonRpcMessage, RuntimeOptions, RuntimeRequest,
    ThreadReadParams,
};
use serde_json::{Map, Value};

const DIRECT_INPUT_TO_PARENT_OWNED_THREAD_ERROR: &str =
    "direct app-server input is not allowed for parent-owned threads";

impl RequestProcessor {
    /// v2 `turn/start` boundary. The runtime still owns execution; this
    /// adapter only resolves the canonical thread identity and lowers the
    /// typed v2 request into the current RuntimeCore request.
    pub(super) async fn handle_turn_start_v2_impl(
        &self,
        params: Option<Value>,
        event_callback: Option<&mut (dyn FnMut(JsonRpcMessage) + Send)>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: TurnStartParams = parse_params(params)?;
        let session_id = self.resolve_loaded_v2_thread_session(&params.thread_id)?;
        self.ensure_direct_input_allowed(&params.thread_id).await?;
        let runtime_params = lower_turn_start_params(&params, session_id)?;
        let host = self.runtime_host_context();
        let config_warnings = self.config_warning_notifications(ConfigWarningScope::TurnStart);

        let _ = event_callback;
        let output = self
            .runtime
            .start_turn_admitted(runtime_params, host)
            .await
            .map_err(to_jsonrpc_error)?;
        let response = v2_start_response(output.response);
        Ok(dispatch_result(response)?.with_notifications(config_warnings))
    }

    /// v2 `turn/interrupt` boundary. The thread lookup is deliberately
    /// canonical so callers cannot smuggle an unrelated session id.
    pub(super) async fn handle_turn_interrupt_v2_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: TurnInterruptParams = parse_params(params)?;
        let session_id = self.resolve_loaded_v2_thread_session(&params.thread_id)?;
        let turn_id = non_empty_param(&params.turn_id, "turnId")?;
        self.runtime
            .ensure_turn_interruptible(&session_id, &turn_id)
            .map_err(|error| map_interrupt_runtime_error(error, &params))?;
        self.abort_server_requests_for_turn(params.thread_id.clone(), turn_id.clone())
            .await;
        let output = self
            .runtime
            .cancel_turn(
                AgentSessionTurnCancelParams {
                    session_id,
                    turn_id,
                },
                self.runtime_host_context(),
            )
            .await
            .map_err(|error| map_interrupt_runtime_error(error, &params))?;
        let mut notification_projector = V2NotificationProjector::default();
        let notifications = project_events(&mut notification_projector, output.events)?;
        Ok(dispatch_result(TurnInterruptResponse {})?.with_notifications(notifications))
    }

    pub(super) fn resolve_loaded_v2_thread_session(
        &self,
        thread_id: &str,
    ) -> Result<String, JsonRpcError> {
        let thread_id = non_empty_param(thread_id, "threadId")?;
        self.runtime
            .loaded_session_id_for_thread(&thread_id)
            .ok_or_else(|| invalid_request(format!("thread not found: {thread_id}")))
    }

    pub(super) async fn resolve_persisted_v2_thread_session(
        &self,
        thread_id: &str,
    ) -> Result<String, JsonRpcError> {
        let thread_id = non_empty_param(thread_id, "threadId")?;
        let response = self
            .runtime
            .read_thread(ThreadReadParams {
                thread_id: ThreadId::from(thread_id),
                turns_view: ThreadTurnsView::NotLoaded,
            })
            .await
            .map_err(to_jsonrpc_error)?;
        Ok(response.thread.session_id.to_string())
    }

    pub(super) async fn ensure_direct_input_allowed(
        &self,
        thread_id: &str,
    ) -> Result<(), JsonRpcError> {
        let allowed = self
            .runtime
            .can_accept_direct_input(thread_id)
            .await
            .map_err(to_jsonrpc_error)?;
        if !allowed {
            return Err(invalid_request(DIRECT_INPUT_TO_PARENT_OWNED_THREAD_ERROR));
        }
        Ok(())
    }
}

fn lower_turn_start_params(
    params: &TurnStartParams,
    session_id: String,
) -> Result<crate::runtime::TurnStartRequest, JsonRpcError> {
    Ok(crate::runtime::TurnStartRequest {
        session_id,
        turn_id: None,
        input: lower_user_input(params.input.clone())?,
        runtime_options: lower_runtime_options(params)?,
        queue_if_busy: false,
        skip_pre_submit_resume: false,
    })
}

fn lower_runtime_options(params: &TurnStartParams) -> Result<Option<RuntimeOptions>, JsonRpcError> {
    let mut options = RuntimeOptions::default();
    options.output_schema = params.output_schema.clone();
    let mut request = RuntimeRequest::default();
    request.collaboration_mode = params.collaboration_mode.clone();
    let mut metadata = Map::new();

    if let Some(mode) = params.collaboration_mode.as_ref() {
        let model = mode.settings.model.trim();
        if model.is_empty() {
            return Err(invalid_params(
                "collaborationMode.settings.model must not be empty",
            ));
        }
        request.model_preference = Some(model.to_string());
        request.reasoning_effort = mode.settings.reasoning_effort.clone();
        request.system_prompt = mode.settings.developer_instructions.clone();
    } else {
        if let Some(model) = params
            .model
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            request.model_preference = Some(model.to_string());
        }
        if let Some(effort) = params
            .effort
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            request.reasoning_effort = Some(effort.to_string());
        }
    }
    if let Some(cwd) = params
        .cwd
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        request.working_dir = Some(cwd.to_string());
    }
    if let Some(root) = params
        .runtime_workspace_roots
        .as_ref()
        .and_then(|roots| roots.first())
        .map(String::as_str)
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        request.workspace_root = Some(root.to_string());
    }
    if let Some(policy) = params.approval_policy.as_ref() {
        if let Some(policy) = policy.as_str().map(str::trim).filter(|v| !v.is_empty()) {
            request.approval_policy = Some(policy.to_string());
        } else {
            metadata.insert("approvalPolicy".to_string(), policy.clone());
        }
    }
    if let Some(policy) = params.sandbox_policy.as_ref() {
        if let Some(policy) = policy.as_str().map(str::trim).filter(|v| !v.is_empty()) {
            request.sandbox_policy = Some(policy.to_string());
        } else {
            metadata.insert("sandboxPolicy".to_string(), policy.clone());
        }
    }
    if let Some(value) = params.permissions.as_ref() {
        metadata.insert("permissions".to_string(), Value::String(value.clone()));
    }
    if let Some(value) = params
        .client_user_message_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        metadata.insert(
            "clientUserMessageId".to_string(),
            Value::String(value.to_string()),
        );
    }
    if let Some(value) = params.responsesapi_client_metadata.as_ref() {
        metadata.insert(
            "responsesapiClientMetadata".to_string(),
            serde_json::to_value(value).map_err(|error| {
                invalid_params(format!("invalid responsesapiClientMetadata: {error}"))
            })?,
        );
    }
    if let Some(value) = params.additional_context.as_ref() {
        lower_application_metadata(value, &mut metadata);
        metadata.insert(
            "additionalContext".to_string(),
            serde_json::to_value(value)
                .map_err(|error| invalid_params(format!("invalid additionalContext: {error}")))?,
        );
    }
    if let Some(value) = params.environments.as_ref() {
        metadata.insert(
            "environments".to_string(),
            serde_json::to_value(value)
                .map_err(|error| invalid_params(format!("invalid environments: {error}")))?,
        );
    }
    if let Some(value) = params.service_tier.as_ref() {
        metadata.insert(
            "serviceTier".to_string(),
            serde_json::to_value(value)
                .map_err(|error| invalid_params(format!("invalid serviceTier: {error}")))?,
        );
    }
    for (key, value) in [
        ("summary", params.summary.clone().map(Value::String)),
        ("personality", params.personality.clone()),
    ] {
        if let Some(value) = value {
            metadata.insert(key.to_string(), value);
        }
    }
    if !metadata.is_empty() {
        request.metadata = Some(Value::Object(metadata));
    }
    if request != RuntimeRequest::default() {
        options.runtime_request = Some(request);
    }
    if options.output_schema.is_some() || options.runtime_request.is_some() {
        Ok(Some(options))
    } else {
        Ok(None)
    }
}

fn lower_application_metadata(
    additional_context: &std::collections::HashMap<
        String,
        app_server_protocol::protocol::v2::AdditionalContextEntry,
    >,
    metadata: &mut Map<String, Value>,
) {
    let Some(entry) = additional_context.get("metadata") else {
        return;
    };
    if entry.kind != AdditionalContextKind::Application {
        return;
    }
    let Ok(Value::Object(application_metadata)) = serde_json::from_str(&entry.value) else {
        return;
    };
    for (key, value) in application_metadata {
        metadata.entry(key).or_insert(value);
    }
}

fn v2_start_response(response: AgentSessionTurnStartResponse) -> TurnStartResponse {
    TurnStartResponse {
        turn: v2_turn_from_agent_turn(response.turn),
    }
}

fn v2_turn_from_agent_turn(turn: AgentTurn) -> V2Turn {
    let status = match turn.status {
        AgentTurnStatus::Completed => V2TurnStatus::Completed,
        AgentTurnStatus::Canceled => V2TurnStatus::Interrupted,
        AgentTurnStatus::Failed => V2TurnStatus::Failed,
        AgentTurnStatus::Accepted
        | AgentTurnStatus::Queued
        | AgentTurnStatus::Running
        | AgentTurnStatus::WaitingAction => V2TurnStatus::InProgress,
    };
    let started_at_ms = turn.started_at.as_deref().and_then(timestamp_millis);
    let completed_at_ms = turn.completed_at.as_deref().and_then(timestamp_millis);
    let duration_ms = started_at_ms
        .zip(completed_at_ms)
        .map(|(started, completed)| completed.saturating_sub(started));
    V2Turn {
        id: turn.turn_id,
        items: Vec::new(),
        items_view: app_server_protocol::protocol::v2::TurnItemsView::NotLoaded,
        status,
        error: None,
        started_at: started_at_ms.map(|value| value.div_euclid(1_000)),
        completed_at: completed_at_ms.map(|value| value.div_euclid(1_000)),
        duration_ms,
    }
}

fn timestamp_millis(value: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|value| value.timestamp_millis())
}

fn non_empty_param(value: &str, field: &str) -> Result<String, JsonRpcError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(invalid_params(format!("turn request requires {field}")));
    }
    Ok(value.to_string())
}

impl RequestProcessor {
    pub(super) async fn handle_turn_steer_impl(
        &self,
        params: Option<Value>,
        event_callback: Option<&mut (dyn FnMut(JsonRpcMessage) + Send)>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: TurnSteerParams = parse_params(params)?;
        reject_unsupported_map(
            "responsesapiClientMetadata",
            params.responsesapi_client_metadata.as_ref(),
        )?;
        reject_unsupported_map("additionalContext", params.additional_context.as_ref())?;
        self.resolve_loaded_v2_thread_session(&params.thread_id)?;
        self.ensure_direct_input_allowed(&params.thread_id).await?;
        let input = lower_user_input(params.input.clone())?;
        let output = self
            .runtime
            .steer_turn(
                &params.thread_id,
                &params.expected_turn_id,
                input,
                params.client_user_message_id.clone(),
            )
            .await
            .map_err(|error| map_steer_runtime_error(error, &params))?;

        let response = TurnSteerResponse {
            turn_id: output.response,
        };
        let mut notification_projector = V2NotificationProjector::default();
        if let Some(event_callback) = event_callback {
            for event in output.events {
                for notification in notification_projector.project(event)? {
                    event_callback(JsonRpcMessage::Notification(notification));
                }
            }
            dispatch_result(response)
        } else {
            let notifications = project_events(&mut notification_projector, output.events)?;
            Ok(dispatch_result(response)?.with_notifications(notifications))
        }
    }
}

fn map_steer_runtime_error(
    error: crate::RuntimeCoreError,
    params: &TurnSteerParams,
) -> JsonRpcError {
    match error {
        crate::RuntimeCoreError::TurnNotActive(_) => invalid_request(format!(
            "expected active turn id `{}` is no longer active",
            params.expected_turn_id
        )),
        crate::RuntimeCoreError::SessionNotFound(_) => {
            invalid_request(format!("thread not found: {}", params.thread_id.trim()))
        }
        crate::RuntimeCoreError::InvalidRequest(message) => invalid_request(message),
        other => to_jsonrpc_error(other),
    }
}

fn map_interrupt_runtime_error(
    error: crate::RuntimeCoreError,
    params: &TurnInterruptParams,
) -> JsonRpcError {
    match error {
        crate::RuntimeCoreError::TurnNotActive(_) => invalid_request("no active turn to interrupt"),
        crate::RuntimeCoreError::SessionNotFound(_) => {
            invalid_request(format!("thread not found: {}", params.thread_id.trim()))
        }
        crate::RuntimeCoreError::InvalidRequest(message) => invalid_request(message),
        other => to_jsonrpc_error(other),
    }
}

fn lower_user_input(items: Vec<UserInput>) -> Result<Vec<AgentInput>, JsonRpcError> {
    if items.is_empty() {
        return Err(invalid_params("turn/steer input must not be empty"));
    }

    let input = items
        .into_iter()
        .map(UserInput::into_core)
        .collect::<Vec<_>>();
    for part in &input {
        part.validate()
            .map_err(|error| invalid_params(error.to_string()))?;
    }
    if input.iter().all(|part| {
        matches!(
            part,
            AgentInput::Text { text, .. } if text.trim().is_empty()
        )
    }) {
        return Err(invalid_params("turn/steer input must not be empty"));
    }
    Ok(input)
}

fn reject_unsupported_map<K, V>(
    field: &str,
    value: Option<&std::collections::HashMap<K, V>>,
) -> Result<(), JsonRpcError> {
    if value.is_some_and(|value| !value.is_empty()) {
        return Err(invalid_params(format!(
            "turn/steer {field} is not supported by the current runtime boundary"
        )));
    }
    Ok(())
}

fn invalid_params(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(error_codes::INVALID_PARAMS, message)
}

fn invalid_request(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(error_codes::INVALID_REQUEST, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_protocol::{CollaborationMode, CollaborationModeSettings, ImageDetail, ModeKind};
    use serde_json::json;

    #[test]
    fn lowers_v2_text_and_image_input_without_losing_ordered_text() {
        let input = lower_user_input(vec![
            UserInput::Text {
                text: "first".to_string(),
                text_elements: Vec::new(),
            },
            UserInput::Text {
                text: "second".to_string(),
                text_elements: Vec::new(),
            },
            UserInput::LocalImage {
                detail: Some(ImageDetail::High),
                path: "/tmp/image.png".to_string(),
            },
        ])
        .expect("lower input");

        assert_eq!(
            input,
            vec![
                AgentInput::text("first"),
                AgentInput::text("second"),
                AgentInput::LocalImage {
                    path: "/tmp/image.png".to_string(),
                    detail: Some(ImageDetail::High),
                },
            ]
        );
    }

    #[test]
    fn preserves_structured_v2_input_without_turning_it_into_text() {
        let input = lower_user_input(vec![UserInput::Skill {
            name: "review".to_string(),
            path: "/skills/review/SKILL.md".to_string(),
        }])
        .expect("skill input");

        assert_eq!(
            input,
            vec![AgentInput::Skill {
                name: "review".to_string(),
                path: "/skills/review/SKILL.md".to_string(),
            }]
        );
    }

    #[test]
    fn lowers_client_message_identity_into_runtime_metadata() {
        let params = TurnStartParams {
            thread_id: "thread-1".to_string(),
            client_user_message_id: Some(" client-1 ".to_string()),
            input: vec![UserInput::Text {
                text: "hello".to_string(),
                text_elements: Vec::new(),
            }],
            ..TurnStartParams::default()
        };

        let lowered =
            lower_turn_start_params(&params, "session-1".to_string()).expect("lower turn start");
        assert_eq!(
            lowered
                .runtime_options
                .as_ref()
                .and_then(RuntimeOptions::runtime_metadata)
                .and_then(|metadata| metadata.get("clientUserMessageId")),
            Some(&json!("client-1"))
        );
    }

    #[test]
    fn collaboration_mode_settings_override_plain_turn_fields() {
        let params = TurnStartParams {
            thread_id: "thread-1".to_string(),
            input: vec![UserInput::Text {
                text: "hello".to_string(),
                text_elements: Vec::new(),
            }],
            model: Some("stale-model".to_string()),
            effort: Some("low".to_string()),
            collaboration_mode: Some(CollaborationMode {
                mode: ModeKind::Plan,
                settings: CollaborationModeSettings {
                    model: "gpt-5.4".to_string(),
                    reasoning_effort: Some("high".to_string()),
                    developer_instructions: Some("Plan before editing.".to_string()),
                },
            }),
            ..TurnStartParams::default()
        };

        let options = lower_runtime_options(&params)
            .expect("typed collaboration mode")
            .expect("runtime options");
        let request = options.runtime_request.expect("runtime request");

        assert_eq!(request.model_preference.as_deref(), Some("gpt-5.4"));
        assert_eq!(request.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(
            request.system_prompt.as_deref(),
            Some("Plan before editing.")
        );
        assert_eq!(request.collaboration_mode, params.collaboration_mode);
    }

    #[test]
    fn collaboration_mode_rejects_an_empty_settings_model() {
        let params = TurnStartParams {
            thread_id: "thread-1".to_string(),
            input: vec![UserInput::Text {
                text: "hello".to_string(),
                text_elements: Vec::new(),
            }],
            collaboration_mode: Some(CollaborationMode {
                mode: ModeKind::Plan,
                settings: CollaborationModeSettings {
                    model: "   ".to_string(),
                    reasoning_effort: None,
                    developer_instructions: None,
                },
            }),
            ..TurnStartParams::default()
        };

        let error = lower_runtime_options(&params).expect_err("empty model must fail closed");
        assert_eq!(error.code, error_codes::INVALID_PARAMS);
        assert!(error.message.contains("settings.model must not be empty"));
    }

    #[test]
    fn deprecated_multi_agent_mode_does_not_enter_runtime_metadata() {
        let params = TurnStartParams {
            thread_id: "thread-1".to_string(),
            input: vec![UserInput::Text {
                text: "hello".to_string(),
                text_elements: Vec::new(),
            }],
            multi_agent_mode: Some(agent_protocol::MultiAgentMode::Proactive),
            ..TurnStartParams::default()
        };

        assert!(lower_runtime_options(&params)
            .expect("deprecated field is accepted")
            .is_none());
    }

    #[test]
    fn lowers_v2_application_metadata_into_runtime_options() {
        let trace = json!({
            "traceId": "trace-v2",
            "requestId": "request-v2",
            "submittedAt": 1_784_447_000_000_i64,
        });
        let harness = json!({
            "image_command_intent": {
                "kind": "image_task",
                "image_task": {
                    "prompt": "draw a lime"
                }
            }
        });
        let params = TurnStartParams {
            thread_id: "thread-1".to_string(),
            input: vec![UserInput::Text {
                text: "hello".to_string(),
                text_elements: Vec::new(),
            }],
            additional_context: Some(
                [(
                    "metadata".to_string(),
                    app_server_protocol::protocol::v2::AdditionalContextEntry {
                        kind: AdditionalContextKind::Application,
                        value: json!({
                            "agentUiPerformanceTrace": trace,
                            "harness": harness,
                        })
                        .to_string(),
                    },
                )]
                .into(),
            ),
            ..TurnStartParams::default()
        };

        let lowered = lower_runtime_options(&params)
            .expect("lower runtime options")
            .expect("runtime options");
        let metadata = lowered.runtime_metadata().expect("runtime metadata");
        assert_eq!(metadata.get("agentUiPerformanceTrace"), Some(&trace));
        assert_eq!(metadata.get("harness"), Some(&harness));
        assert!(metadata.get("additionalContext").is_some());
    }

    #[test]
    fn does_not_lower_untrusted_v2_trace_metadata() {
        let params = TurnStartParams {
            thread_id: "thread-1".to_string(),
            input: vec![UserInput::Text {
                text: "hello".to_string(),
                text_elements: Vec::new(),
            }],
            additional_context: Some(
                [(
                    "metadata".to_string(),
                    app_server_protocol::protocol::v2::AdditionalContextEntry {
                        kind: AdditionalContextKind::Untrusted,
                        value: json!({
                            "agentUiPerformanceTrace": { "traceId": "must-not-lower" }
                        })
                        .to_string(),
                    },
                )]
                .into(),
            ),
            ..TurnStartParams::default()
        };

        let lowered = lower_runtime_options(&params)
            .expect("lower runtime options")
            .expect("runtime options");
        let metadata = lowered.runtime_metadata().expect("runtime metadata");
        assert!(metadata.get("agentUiPerformanceTrace").is_none());
    }
}
