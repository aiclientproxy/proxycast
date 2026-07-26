use super::*;
use agent_runtime::reply_input::RuntimeReplyInput;
use app_server_protocol::{AgentTurn, AgentTurnStatus, RouteFailureCategory};
use tool_runtime::agent_control::SpawnAgentModelOverrides;

pub(super) async fn prepare(
    core: &RuntimeCore,
    parent_session: &AgentSession,
    host: &RuntimeHostContext,
    child_session_id: &str,
    child_thread_id: &str,
    inherited: Option<RuntimeOptions>,
    overrides: &SpawnAgentModelOverrides,
) -> Result<Option<RuntimeOptions>, RuntimeCoreError> {
    if overrides == &SpawnAgentModelOverrides::default() {
        return Ok(inherited);
    }

    let mut options = inherited.unwrap_or_default();
    let runtime_request = options.runtime_request_mut();
    if let Some(model) = overrides.model.as_ref() {
        runtime_request.model_preference = Some(model.clone());
    }
    if let Some(reasoning_effort) = overrides.reasoning_effort.as_ref() {
        runtime_request.reasoning_effort = Some(reasoning_effort.clone());
    }
    if let Some(service_tier) = overrides.service_tier.as_ref() {
        runtime_request.service_tier = Some(service_tier.clone());
    }

    let mut child_session = parent_session.clone();
    child_session.session_id = child_session_id.to_string();
    child_session.thread_id = child_thread_id.to_string();
    let turn_id = format!("spawn-preflight-{child_thread_id}");
    let mut request = ExecutionRequest {
        host: host.clone(),
        session: child_session,
        turn: AgentTurn {
            turn_id,
            session_id: child_session_id.to_string(),
            thread_id: child_thread_id.to_string(),
            status: AgentTurnStatus::Accepted,
            started_at: None,
            completed_at: None,
        },
        forked_from_thread_id: Some(parent_session.thread_id.clone()),
        input: RuntimeReplyInput::text("agent control spawn route preflight"),
        runtime_options: Some(options),
        expected_output: None,
        structured_output: None,
        output_schema: None,
        event_name: None,
        queued_turn_id: None,
        queue_if_busy: false,
        skip_pre_submit_resume: false,
        agent_control_gateway: None,
    };
    request.runtime_options = core.backend.effective_turn_runtime_options(&request, true);
    let prepared = core
        .backend
        .prepare_turn_runtime_options(&request, true)
        .await?;
    validate_resolved_overrides(child_session_id, prepared.as_ref(), overrides)?;
    Ok(prepared)
}

fn validate_resolved_overrides(
    child_session_id: &str,
    prepared: Option<&RuntimeOptions>,
    overrides: &SpawnAgentModelOverrides,
) -> Result<(), RuntimeCoreError> {
    let runtime_request = prepared.and_then(RuntimeOptions::runtime_request);
    let route = runtime_request
        .and_then(|request| request.metadata.as_ref())
        .and_then(|metadata| metadata.get("agentControlRoute"));
    let resolved_provider = route
        .and_then(|route| route.get("providerPreference"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let resolved_model = route
        .and_then(|route| route.get("modelPreference"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let resolved_effort = route
        .and_then(|route| route.pointer("/providerConfig/reasoningEffort"))
        .and_then(serde_json::Value::as_str);
    let resolved_service_tier = route
        .and_then(|route| route.get("serviceTier"))
        .and_then(serde_json::Value::as_str);

    if let Some(requested_model) = overrides.model.as_deref() {
        if resolved_model.as_deref() != Some(requested_model) {
            return Err(RuntimeCoreError::RouteRejected {
                session_id: child_session_id.to_string(),
                provider: resolved_provider,
                model: Some(requested_model.to_string()),
                category: RouteFailureCategory::NoCandidate,
                reason_code: "spawn_agent_model_fallback_not_allowed".to_string(),
            });
        }
    }
    if let Some(requested_effort) = overrides.reasoning_effort.as_deref() {
        if resolved_effort != Some(requested_effort) {
            return Err(RuntimeCoreError::RouteRejected {
                session_id: child_session_id.to_string(),
                provider: resolved_provider,
                model: resolved_model,
                category: RouteFailureCategory::CapabilityGap,
                reason_code: "spawn_agent_reasoning_effort_unsupported".to_string(),
            });
        }
    }
    if let Some(requested_service_tier) = overrides.service_tier.as_deref() {
        let supported_service_tiers = supported_service_tier_ids(route);
        if resolved_service_tier != Some(requested_service_tier)
            || !supported_service_tiers
                .iter()
                .any(|supported| supported == requested_service_tier)
        {
            return Err(RuntimeCoreError::RouteRejected {
                session_id: child_session_id.to_string(),
                provider: resolved_provider,
                model: resolved_model,
                category: RouteFailureCategory::CapabilityGap,
                reason_code: "spawn_agent_service_tier_unsupported".to_string(),
            });
        }
    }
    Ok(())
}

fn supported_service_tier_ids(route: Option<&serde_json::Value>) -> Vec<String> {
    route
        .filter(|route| {
            route
                .pointer("/modelRegistry/status")
                .and_then(serde_json::Value::as_str)
                == Some("matched")
        })
        .and_then(|route| route.pointer("/modelRegistry/model/service_tiers"))
        .and_then(serde_json::Value::as_array)
        .map(|tiers| {
            tiers
                .iter()
                .filter_map(|tier| tier.get("id"))
                .filter_map(serde_json::Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn prepared(model: &str, reasoning_effort: &str) -> RuntimeOptions {
        RuntimeOptions {
            runtime_request: Some(app_server_protocol::RuntimeRequest {
                metadata: Some(json!({
                    "agentControlRoute": {
                        "providerPreference": "openai",
                        "modelPreference": model,
                        "serviceTier": "priority",
                        "providerConfig": {
                            "reasoningEffort": reasoning_effort
                        },
                        "modelRegistry": {
                            "status": "matched",
                            "model": {
                                "service_tiers": [
                                    { "id": "priority", "name": "Priority", "description": "" }
                                ]
                            }
                        }
                    }
                })),
                ..app_server_protocol::RuntimeRequest::default()
            }),
            ..RuntimeOptions::default()
        }
    }

    #[test]
    fn accepts_exact_model_and_reasoning_resolution() {
        let overrides = SpawnAgentModelOverrides {
            model: Some("gpt-5.6-sol".to_string()),
            reasoning_effort: Some("high".to_string()),
            service_tier: Some("priority".to_string()),
        };

        validate_resolved_overrides(
            "child-session",
            Some(&prepared("gpt-5.6-sol", "high")),
            &overrides,
        )
        .expect("exact route");
    }

    #[test]
    fn rejects_model_fallback_and_reasoning_downgrade() {
        let model_error = validate_resolved_overrides(
            "child-session",
            Some(&prepared("fallback-model", "high")),
            &SpawnAgentModelOverrides {
                model: Some("gpt-5.6-sol".to_string()),
                reasoning_effort: None,
                service_tier: None,
            },
        )
        .expect_err("model fallback");
        assert!(matches!(
            model_error,
            RuntimeCoreError::RouteRejected { reason_code, .. }
                if reason_code == "spawn_agent_model_fallback_not_allowed"
        ));

        let reasoning_error = validate_resolved_overrides(
            "child-session",
            Some(&prepared("gpt-5.6-sol", "medium")),
            &SpawnAgentModelOverrides {
                model: Some("gpt-5.6-sol".to_string()),
                reasoning_effort: Some("high".to_string()),
                service_tier: None,
            },
        )
        .expect_err("reasoning downgrade");
        assert!(matches!(
            reasoning_error,
            RuntimeCoreError::RouteRejected { reason_code, .. }
                if reason_code == "spawn_agent_reasoning_effort_unsupported"
        ));
    }

    #[test]
    fn rejects_service_tier_missing_from_selected_model_catalog() {
        let mut route = prepared("gpt-5.6-sol", "high");
        let agent_control_route = route
            .runtime_request
            .as_mut()
            .and_then(|request| request.metadata.as_mut())
            .and_then(|metadata| metadata.get_mut("agentControlRoute"))
            .expect("agent control route");
        agent_control_route["serviceTier"] = json!("flex");
        agent_control_route
            .as_object_mut()
            .expect("route object")
            .remove("modelRegistry");

        let unsupported = validate_resolved_overrides(
            "child-session",
            Some(&route),
            &SpawnAgentModelOverrides {
                model: Some("gpt-5.6-sol".to_string()),
                reasoning_effort: Some("high".to_string()),
                service_tier: Some("flex".to_string()),
            },
        )
        .expect_err("unsupported service tier");

        assert!(matches!(
            unsupported,
            RuntimeCoreError::RouteRejected { reason_code, .. }
                if reason_code == "spawn_agent_service_tier_unsupported"
        ));
    }
}
