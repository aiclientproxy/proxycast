use agent_protocol::CollaborationMode;
use app_server_protocol::{
    AgentEvent, AgentTurn, AgentTurnStatus, RuntimeOptions, RuntimeProviderConfig, RuntimeRequest,
    RuntimeSearchMode,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

const QUEUED_TURN_INTENT_FIELD: &str = "queuedTurnIntent";
const QUEUED_TURN_INTENT_SCHEMA_VERSION: u32 = 2;
const MAX_INTENT_STRING_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct QueuedTurnIntent {
    schema_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_options: Option<QueuedRuntimeOptions>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct QueuedRuntimeOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    capability_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    event_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_request: Option<QueuedRuntimeRequest>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct QueuedRuntimeRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider: Option<QueuedProviderRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider_preference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_preference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    collaboration_mode: Option<CollaborationMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    service_tier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    thinking_enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    approval_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sandbox_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    workspace_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    working_dir: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    workspace_root: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    project_root: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    web_search: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    search_mode: Option<RuntimeSearchMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    execution_strategy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    auto_continue: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    metadata: Option<QueuedRuntimeMetadata>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct QueuedProviderRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct QueuedRuntimeMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    client_user_message_id: Option<String>,
}

pub(super) fn snapshot_value(runtime_options: Option<&RuntimeOptions>) -> Result<Value, String> {
    let runtime_options = runtime_options
        .map(QueuedRuntimeOptions::from_runtime_options)
        .transpose()?;
    serde_json::to_value(QueuedTurnIntent {
        schema_version: QUEUED_TURN_INTENT_SCHEMA_VERSION,
        runtime_options,
    })
    .map_err(|error| format!("failed to serialize queued turn intent: {error}"))
}

pub(super) fn runtime_options_from_events(
    turns: &[AgentTurn],
    events: &[AgentEvent],
) -> Result<HashMap<String, RuntimeOptions>, String> {
    let queued_turn_ids = turns
        .iter()
        .filter(|turn| turn.status == AgentTurnStatus::Queued)
        .map(|turn| turn.turn_id.as_str())
        .collect::<HashSet<_>>();
    let mut options_by_turn = HashMap::new();

    for event in events
        .iter()
        .filter(|event| event.event_type == "queue.added")
    {
        let Some(turn_id) = event.turn_id.as_deref() else {
            continue;
        };
        if !queued_turn_ids.contains(turn_id) {
            continue;
        }
        let Some(snapshot) = event.payload.get(QUEUED_TURN_INTENT_FIELD) else {
            continue;
        };
        let intent = serde_json::from_value::<QueuedTurnIntent>(snapshot.clone())
            .map_err(|error| format!("invalid queued turn intent for turn {turn_id}: {error}"))?;
        if intent.schema_version != QUEUED_TURN_INTENT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported queued turn intent schema version {} for turn {turn_id}",
                intent.schema_version
            ));
        }
        if let Some(runtime_options) = intent.runtime_options {
            options_by_turn.insert(turn_id.to_string(), runtime_options.into_runtime_options());
        }
    }

    Ok(options_by_turn)
}

impl QueuedRuntimeOptions {
    fn from_runtime_options(options: &RuntimeOptions) -> Result<Self, String> {
        Ok(Self {
            capability_id: bounded_string(options.capability_id.as_deref(), "capabilityId")?,
            event_name: bounded_string(options.event_name.as_deref(), "eventName")?,
            runtime_request: options
                .runtime_request
                .as_ref()
                .map(QueuedRuntimeRequest::from_runtime_request)
                .transpose()?,
        })
    }

    fn into_runtime_options(self) -> RuntimeOptions {
        RuntimeOptions {
            capability_id: self.capability_id,
            event_name: self.event_name,
            runtime_request: self
                .runtime_request
                .map(QueuedRuntimeRequest::into_runtime_request),
            ..RuntimeOptions::default()
        }
    }
}

impl QueuedRuntimeRequest {
    fn from_runtime_request(request: &RuntimeRequest) -> Result<Self, String> {
        let provider = request
            .provider_config
            .as_ref()
            .map(QueuedProviderRef::from_provider_config)
            .transpose()?
            .filter(|provider| !provider.is_empty());
        let provider_preference = bounded_string(
            request.provider_preference.as_deref(),
            "runtimeRequest.providerPreference",
        )?;
        if request
            .provider_config
            .as_ref()
            .is_some_and(|config| config.api_key.is_some() || config.base_url.is_some())
            && !provider
                .as_ref()
                .is_some_and(QueuedProviderRef::has_provider_identity)
            && provider_preference.is_none()
        {
            return Err(
                "queued turn direct provider credentials require a durable provider identity"
                    .to_string(),
            );
        }
        Ok(Self {
            provider,
            provider_preference,
            model_preference: bounded_string(
                request.model_preference.as_deref(),
                "runtimeRequest.modelPreference",
            )?,
            collaboration_mode: request.collaboration_mode.clone(),
            reasoning_effort: bounded_string(
                request.reasoning_effort.as_deref(),
                "runtimeRequest.reasoningEffort",
            )?,
            service_tier: bounded_string(
                request.service_tier.as_deref(),
                "runtimeRequest.serviceTier",
            )?,
            thinking_enabled: request.thinking_enabled,
            approval_policy: bounded_string(
                request.approval_policy.as_deref(),
                "runtimeRequest.approvalPolicy",
            )?,
            sandbox_policy: bounded_string(
                request.sandbox_policy.as_deref(),
                "runtimeRequest.sandboxPolicy",
            )?,
            workspace_id: bounded_string(
                request.workspace_id.as_deref(),
                "runtimeRequest.workspaceId",
            )?,
            working_dir: bounded_string(
                request.working_dir.as_deref(),
                "runtimeRequest.workingDir",
            )?,
            workspace_root: bounded_string(
                request.workspace_root.as_deref(),
                "runtimeRequest.workspaceRoot",
            )?,
            project_root: bounded_string(
                request.project_root.as_deref(),
                "runtimeRequest.projectRoot",
            )?,
            web_search: request.web_search,
            search_mode: request.search_mode,
            execution_strategy: bounded_string(
                request.execution_strategy.as_deref(),
                "runtimeRequest.executionStrategy",
            )?,
            auto_continue: request.auto_continue,
            metadata: request
                .metadata
                .as_ref()
                .map(QueuedRuntimeMetadata::from_value)
                .transpose()?
                .filter(|metadata| !metadata.is_empty()),
        })
    }

    fn into_runtime_request(self) -> RuntimeRequest {
        RuntimeRequest {
            provider_config: self.provider.map(QueuedProviderRef::into_provider_config),
            provider_preference: self.provider_preference,
            model_preference: self.model_preference,
            collaboration_mode: self.collaboration_mode,
            reasoning_effort: self.reasoning_effort,
            service_tier: self.service_tier,
            thinking_enabled: self.thinking_enabled,
            approval_policy: self.approval_policy,
            sandbox_policy: self.sandbox_policy,
            workspace_id: self.workspace_id,
            working_dir: self.working_dir,
            workspace_root: self.workspace_root,
            project_root: self.project_root,
            web_search: self.web_search,
            search_mode: self.search_mode,
            execution_strategy: self.execution_strategy,
            auto_continue: self.auto_continue,
            metadata: self.metadata.map(QueuedRuntimeMetadata::into_value),
            ..RuntimeRequest::default()
        }
    }
}

impl QueuedProviderRef {
    fn from_provider_config(config: &RuntimeProviderConfig) -> Result<Self, String> {
        Ok(Self {
            provider_id: bounded_string(
                config.provider_id.as_deref(),
                "runtimeRequest.provider.providerId",
            )?,
            provider_name: bounded_string(
                config.provider_name.as_deref(),
                "runtimeRequest.provider.providerName",
            )?,
            model_name: bounded_string(
                config.model_name.as_deref(),
                "runtimeRequest.provider.modelName",
            )?,
        })
    }

    fn is_empty(&self) -> bool {
        self.provider_id.is_none() && self.provider_name.is_none() && self.model_name.is_none()
    }

    fn has_provider_identity(&self) -> bool {
        self.provider_id.is_some() || self.provider_name.is_some()
    }

    fn into_provider_config(self) -> RuntimeProviderConfig {
        RuntimeProviderConfig {
            provider_id: self.provider_id,
            provider_name: self.provider_name,
            model_name: self.model_name,
            ..RuntimeProviderConfig::default()
        }
    }
}

impl QueuedRuntimeMetadata {
    fn from_value(value: &Value) -> Result<Self, String> {
        Ok(Self {
            client_user_message_id: bounded_string(
                value
                    .get("clientUserMessageId")
                    .or_else(|| value.get("client_user_message_id"))
                    .and_then(Value::as_str),
                "runtimeRequest.metadata.clientUserMessageId",
            )?,
        })
    }

    fn is_empty(&self) -> bool {
        self.client_user_message_id.is_none()
    }

    fn into_value(self) -> Value {
        let mut metadata = Map::new();
        if let Some(value) = self.client_user_message_id {
            metadata.insert("clientUserMessageId".to_string(), Value::String(value));
        }
        Value::Object(metadata)
    }
}

fn bounded_string(value: Option<&str>, field: &str) -> Result<Option<String>, String> {
    if let Some(value) = value {
        if value.len() > MAX_INTENT_STRING_BYTES {
            return Err(format!(
                "queued turn intent field {field} exceeds {MAX_INTENT_STRING_BYTES} bytes"
            ));
        }
        return Ok(Some(value.to_string()));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::{RuntimeProviderConfig, RuntimeRequest};
    use serde_json::json;

    fn turn(turn_id: &str, status: AgentTurnStatus) -> AgentTurn {
        AgentTurn {
            turn_id: turn_id.to_string(),
            session_id: "session-1".to_string(),
            thread_id: "thread-1".to_string(),
            status,
            started_at: None,
            completed_at: None,
        }
    }

    fn queue_event(turn_id: Option<&str>, snapshot: Value) -> AgentEvent {
        let mut payload = json!({
            "queuedTurnId": "untrusted-payload-turn",
        });
        payload[QUEUED_TURN_INTENT_FIELD] = snapshot;
        AgentEvent {
            event_id: format!("event-{}", turn_id.unwrap_or("none")),
            sequence: 1,
            session_id: "session-1".to_string(),
            thread_id: Some("thread-1".to_string()),
            turn_id: turn_id.map(str::to_string),
            event_type: "queue.added".to_string(),
            timestamp: "2026-07-19T00:00:00Z".to_string(),
            payload,
        }
    }

    fn runtime_options() -> RuntimeOptions {
        RuntimeOptions {
            capability_id: Some("capability-1".to_string()),
            stream: true,
            event_name: Some("queued-event".to_string()),
            queued_turn_id: Some("ephemeral-queued-id".to_string()),
            runtime_request: Some(RuntimeRequest {
                provider_config: Some(RuntimeProviderConfig {
                    provider_id: Some("provider-1".to_string()),
                    provider_name: Some("Provider One".to_string()),
                    model_name: Some("model-1".to_string()),
                    api_key: Some("fixture-secret-key".to_string()),
                    base_url: Some(
                        "https://user:pass@provider.example/v1?api_key=secret#token".to_string(),
                    ),
                    tool_call_strategy: Some(
                        app_server_protocol::RuntimeToolCallStrategy::ToolShim,
                    ),
                    toolshim_model: Some("ephemeral-toolshim-model".to_string()),
                    model_capabilities: Some(json!({ "secretCapabilityOverride": true })),
                    supports_websockets: Some(true),
                    ..RuntimeProviderConfig::default()
                }),
                provider_preference: Some("provider-1".to_string()),
                model_preference: Some("model-1".to_string()),
                collaboration_mode: Some(agent_protocol::CollaborationMode {
                    mode: agent_protocol::ModeKind::Plan,
                    settings: agent_protocol::CollaborationModeSettings {
                        model: "model-1".to_string(),
                        reasoning_effort: Some("high".to_string()),
                        developer_instructions: None,
                    },
                }),
                reasoning_effort: Some("high".to_string()),
                thinking_enabled: Some(true),
                approval_policy: Some("on-request".to_string()),
                sandbox_policy: Some("workspace-write".to_string()),
                workspace_id: Some("workspace-1".to_string()),
                working_dir: Some("/workspace/project".to_string()),
                web_search: Some(true),
                execution_strategy: Some("current".to_string()),
                auto_continue: Some(true),
                system_prompt: Some("sensitive-system-prompt".to_string()),
                metadata: Some(json!({
                    "clientUserMessageId": "client-message-1",
                    "unknownAuthorization": "Bearer secret-metadata-token",
                    "memory_store_prompt_context": "sensitive-memory-context",
                    "lime_runtime": { "approval_session_cache": "secret-cache" }
                })),
                ..RuntimeRequest::default()
            }),
            expected_output: Some(json!({ "secretExpectedOutput": true })),
            output_schema: Some(json!({ "secretOutputSchema": true })),
            ..RuntimeOptions::default()
        }
    }

    #[test]
    fn snapshot_uses_typed_allowlist_and_preserves_recovery_facts() {
        let snapshot = snapshot_value(Some(&runtime_options())).expect("snapshot");
        let encoded = serde_json::to_string(&snapshot).expect("encode snapshot");
        for forbidden in [
            "fixture-secret-key",
            "user:pass",
            "api_key=secret",
            "sensitive-system-prompt",
            "ephemeral-toolshim-model",
            "secretCapabilityOverride",
            "secret-metadata-token",
            "sensitive-memory-context",
            "secret-cache",
            "secretExpectedOutput",
            "secretOutputSchema",
            "ephemeral-queued-id",
        ] {
            assert!(
                !encoded.contains(forbidden),
                "forbidden queued intent value leaked: {forbidden}"
            );
        }

        let turns = vec![turn("queued-turn", AgentTurnStatus::Queued)];
        let events = vec![queue_event(Some("queued-turn"), snapshot)];
        let mut hydrated = runtime_options_from_events(&turns, &events).expect("hydrate options");
        let options = hydrated.remove("queued-turn").expect("runtime options");
        assert_eq!(options.capability_id.as_deref(), Some("capability-1"));
        assert_eq!(options.event_name.as_deref(), Some("queued-event"));
        assert!(!options.stream);
        assert_eq!(options.queued_turn_id, None);
        assert_eq!(options.expected_output, None);
        assert_eq!(options.output_schema, None);
        let request = options.runtime_request.expect("runtime request");
        assert_eq!(request.provider_preference.as_deref(), Some("provider-1"));
        assert_eq!(request.model_preference.as_deref(), Some("model-1"));
        assert_eq!(request.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(request.approval_policy.as_deref(), Some("on-request"));
        assert_eq!(request.system_prompt, None);
        let collaboration_mode = request
            .collaboration_mode
            .as_ref()
            .expect("typed collaboration mode");
        assert_eq!(collaboration_mode.mode, agent_protocol::ModeKind::Plan);
        assert_eq!(collaboration_mode.settings.model, "model-1");
        assert_eq!(
            request.metadata,
            Some(json!({
                "clientUserMessageId": "client-message-1",
            }))
        );
        let provider = request.provider_config.expect("provider config");
        assert_eq!(provider.provider_id.as_deref(), Some("provider-1"));
        assert_eq!(provider.model_name.as_deref(), Some("model-1"));
        assert_eq!(provider.api_key, None);
        assert_eq!(provider.base_url, None);
        assert_eq!(provider.tool_call_strategy, None);
        assert_eq!(provider.model_capabilities, None);
    }

    #[test]
    fn hydrate_uses_event_turn_id_and_only_restores_currently_queued_turns() {
        let queued_snapshot = snapshot_value(Some(&runtime_options())).expect("snapshot");
        let completed_snapshot = snapshot_value(Some(&runtime_options())).expect("snapshot");
        let turns = vec![
            turn("queued-turn", AgentTurnStatus::Queued),
            turn("completed-turn", AgentTurnStatus::Completed),
        ];
        let events = vec![
            queue_event(Some("queued-turn"), queued_snapshot),
            queue_event(Some("completed-turn"), completed_snapshot),
            queue_event(None, json!({ "schemaVersion": 1 })),
        ];

        let hydrated = runtime_options_from_events(&turns, &events).expect("hydrate options");
        assert_eq!(hydrated.len(), 1);
        assert!(hydrated.contains_key("queued-turn"));
        assert!(!hydrated.contains_key("untrusted-payload-turn"));
        assert!(!hydrated.contains_key("completed-turn"));
    }

    #[test]
    fn malformed_current_queued_intent_fails_closed() {
        let turns = vec![turn("queued-turn", AgentTurnStatus::Queued)];
        let events = vec![queue_event(
            Some("queued-turn"),
            json!({ "schemaVersion": "invalid" }),
        )];

        assert!(runtime_options_from_events(&turns, &events).is_err());
    }

    #[test]
    fn unsupported_schema_version_fails_closed() {
        let turns = vec![turn("queued-turn", AgentTurnStatus::Queued)];
        let events = vec![queue_event(
            Some("queued-turn"),
            json!({ "schemaVersion": 1, "runtimeOptions": {} }),
        )];

        assert!(runtime_options_from_events(&turns, &events).is_err());
    }

    #[test]
    fn direct_credentials_without_repository_provider_identity_fail_closed() {
        let mut options = runtime_options();
        let request = options.runtime_request.as_mut().expect("runtime request");
        request.provider_preference = None;
        let provider = request.provider_config.as_mut().expect("provider config");
        provider.provider_id = None;
        provider.provider_name = None;

        assert!(snapshot_value(Some(&options)).is_err());
    }
}
