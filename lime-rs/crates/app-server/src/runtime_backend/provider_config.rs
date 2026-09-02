use super::model_capability;
use super::request_context::RuntimeModelSelection;
use crate::runtime::memory_prompt::memory_soul_prompt_context_from_config;
use crate::RuntimeCoreError;
use crate::RuntimeEvent;
use app_server_protocol::CapabilitySnapshot;
use lime_agent::SessionProviderConfig;
use lime_core::config::{
    load_config, OrchestratorConfig, ToolExecutionPolicyConfig, WorkspaceSandboxConfig,
};
use lime_core::database::{self, DbConnection};
use serde_json::{json, Value};
use std::sync::Arc;

pub(crate) fn current_agent_runtime_config_metadata() -> Option<Value> {
    let config = match load_config() {
        Ok(config) => config,
        Err(error) => {
            return Some(json!({
                "agent": {
                    "toolExecution": {
                        "loadError": error.to_string(),
                    }
                },
                "orchestrator": {
                    "loadError": error.to_string(),
                }
            }));
        }
    };
    current_agent_runtime_config_metadata_from_config(&config)
}

fn current_agent_runtime_config_metadata_from_config(
    config: &lime_core::config::Config,
) -> Option<Value> {
    let mut agent_config = serde_json::Map::new();
    if !WorkspaceSandboxConfig::is_default(&config.agent.workspace_sandbox) {
        agent_config.insert(
            "workspaceSandbox".to_string(),
            json!(config.agent.workspace_sandbox),
        );
    }
    if !ToolExecutionPolicyConfig::is_default(&config.agent.tool_execution) {
        agent_config.insert(
            "toolExecution".to_string(),
            json!(config.agent.tool_execution),
        );
    }
    let skills_config = (!config.skills.config.is_empty()).then(|| json!(config.skills));
    let orchestrator_config =
        (!OrchestratorConfig::is_default(&config.orchestrator)).then(|| json!(config.orchestrator));
    let soul_context = memory_soul_prompt_context_from_config(config.memory.soul.as_ref());
    if agent_config.is_empty()
        && skills_config.is_none()
        && orchestrator_config.is_none()
        && soul_context.is_none()
    {
        return None;
    }

    let mut metadata = serde_json::Map::new();
    if !agent_config.is_empty() {
        metadata.insert("agent".to_string(), Value::Object(agent_config));
    }
    if let Some(skills_config) = skills_config {
        metadata.insert("skills".to_string(), skills_config);
    }
    if let Some(orchestrator_config) = orchestrator_config {
        metadata.insert("orchestrator".to_string(), orchestrator_config);
    }
    if let Some(soul_context) = soul_context {
        metadata.insert(
            "memory".to_string(),
            json!({
                "soul": soul_context,
            }),
        );
    }

    Some(Value::Object(metadata))
}

#[cfg(test)]
mod tests {
    use super::current_agent_runtime_config_metadata_from_config;
    use lime_core::config::Config;
    use serde_json::json;

    #[test]
    fn default_config_does_not_emit_runtime_metadata() {
        assert!(current_agent_runtime_config_metadata_from_config(&Config::default()).is_none());
    }

    #[test]
    fn enabled_update_plan_is_emitted_under_tool_execution_config() {
        let mut config = Config::default();
        config.agent.tool_execution.update_plan_enabled = true;

        let metadata = current_agent_runtime_config_metadata_from_config(&config)
            .expect("enabled update_plan should produce config metadata");

        assert_eq!(
            metadata.pointer("/agent/toolExecution/update_plan_enabled"),
            Some(&json!(true))
        );
        assert!(metadata
            .pointer("/agent/toolExecution/tool_overrides")
            .is_none());
    }

    #[test]
    fn emitted_metadata_is_read_by_update_plan_runtime_gate() {
        let mut config = Config::default();
        config.agent.tool_execution.update_plan_enabled = true;
        let config_metadata = current_agent_runtime_config_metadata_from_config(&config)
            .expect("enabled update_plan should produce config metadata");
        let runtime_metadata = json!({"config": config_metadata});

        assert!(
            tool_runtime::update_plan::update_plan_enabled_from_metadata(Some(&runtime_metadata))
        );
    }
}

pub(super) fn model_effective_event_from_runtime(
    requested_selection: &RuntimeModelSelection,
    selection: &RuntimeModelSelection,
    provider_config: &SessionProviderConfig,
    service_model_slot: &str,
    capability_snapshot: &CapabilitySnapshot,
) -> RuntimeEvent {
    let provider_id = provider_config
        .provider_selector
        .as_deref()
        .unwrap_or(&selection.provider)
        .to_string();
    let model_ref =
        model_capability::ModelRef::new(provider_id.clone(), provider_config.model_name.clone());
    let capability =
        model_capability::resolve_model_capability(model_ref, Some(capability_snapshot));
    let requested_reasoning_effort = requested_selection.reasoning_effort.as_deref();
    let reasoning_policy =
        model_capability::resolve_reasoning_policy(&capability, requested_reasoning_effort);
    let effective_reasoning_effort = reasoning_policy.effective_level.as_deref();
    let mut payload = model_capability::model_effective_payload(&capability, &reasoning_policy);
    if let Some(payload_object) = payload.as_object_mut() {
        payload_object.insert("provider".to_string(), json!(provider_id));
        payload_object.insert(
            "modelName".to_string(),
            json!(provider_config.model_name.clone()),
        );
        payload_object.insert(
            "model_name".to_string(),
            json!(provider_config.model_name.clone()),
        );
        payload_object.insert("source".to_string(), json!(selection.source));
        payload_object.insert("serviceModelSlot".to_string(), json!(service_model_slot));
        payload_object.insert("service_model_slot".to_string(), json!(service_model_slot));
        if let Some(reasoning_effort) = requested_reasoning_effort {
            payload_object.insert(
                "requestedReasoningEffort".to_string(),
                json!(reasoning_effort),
            );
            payload_object.insert(
                "requested_reasoning_effort".to_string(),
                json!(reasoning_effort),
            );
        }
        if let Some(reasoning_effort) = effective_reasoning_effort {
            payload_object.insert(
                "effectiveReasoningEffort".to_string(),
                json!(reasoning_effort),
            );
            payload_object.insert(
                "effective_reasoning_effort".to_string(),
                json!(reasoning_effort),
            );
        }
    }
    RuntimeEvent::new("model.effective", payload)
}

pub(super) fn initialize_runtime_database(
    db: Option<&DbConnection>,
) -> Result<DbConnection, RuntimeCoreError> {
    let db = if let Some(db) = db {
        Arc::clone(db)
    } else {
        database::init_database().map_err(|error| {
            RuntimeCoreError::Backend(format!("failed to initialize database: {error}"))
        })?
    };
    Ok(db)
}
