use super::{RuntimeCore, RuntimeCoreError};
use agent_protocol::CollaborationMode;
use app_server_protocol::{AgentSession, RuntimeOptions, RuntimeRequest};
use serde_json::{json, Map, Value};

const IMPORTED_CONVERSATION_KIND: &str = "conversation.import";

impl RuntimeCore {
    pub(in crate::runtime) fn session_runtime_defaults(
        &self,
        session_id: &str,
    ) -> Result<Option<RuntimeOptions>, RuntimeCoreError> {
        let state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let Some(stored) = state.sessions.get(session_id) else {
            return Err(RuntimeCoreError::SessionNotFound(session_id.to_string()));
        };
        default_runtime_options_for_session(&stored.session)
    }
}

fn default_runtime_options_for_session(
    session: &AgentSession,
) -> Result<Option<RuntimeOptions>, RuntimeCoreError> {
    let reference = session.business_object_ref.as_ref();
    let metadata = reference.and_then(|reference| reference.metadata.as_ref());
    let imported = reference.is_some_and(|reference| reference.kind == IMPORTED_CONVERSATION_KIND);
    let imported_continuation = if imported {
        metadata.and_then(|metadata| {
            metadata
                .get("importedContinuation")
                .or_else(|| metadata.get("imported_continuation"))
        })
    } else {
        None
    };
    let thread_settings = if imported {
        metadata.and_then(|metadata| {
            metadata
                .get("importedThreadSettings")
                .or_else(|| metadata.get("imported_thread_settings"))
        })
    } else {
        None
    };
    let sources = [imported_continuation, thread_settings, metadata];
    let provider_preference = metadata
        .and_then(|metadata| string_alias(metadata, &["providerSelector", "provider_selector"]))
        .or_else(|| {
            if imported {
                None
            } else {
                metadata
                    .and_then(|metadata| string_alias(metadata, &["providerName", "provider_name"]))
            }
        });
    let persisted_model_preference = provider_preference
        .as_ref()
        .and_then(|_| string_alias_from_sources(&sources, &["modelName", "model_name", "model"]));
    let collaboration_mode = sources
        .iter()
        .filter_map(|source| *source)
        .find_map(|source| source.get("collaborationMode"))
        .cloned()
        .map(serde_json::from_value::<CollaborationMode>)
        .transpose()
        .map_err(|error| {
            RuntimeCoreError::InvalidRequest(format!(
                "invalid persisted collaborationMode for session {}: {error}",
                session.session_id
            ))
        })?;
    let model_preference = collaboration_mode
        .as_ref()
        .map(|mode| mode.settings.model.clone())
        .or(persisted_model_preference);
    let working_dir = string_alias_from_sources(&sources, &["cwd", "workingDir", "working_dir"]);
    let workspace_root = metadata
        .and_then(|metadata| first_string(metadata, &["runtimeWorkspaceRoots"]))
        .or_else(|| {
            string_alias_from_sources(
                &sources,
                &[
                    "workspaceRoot",
                    "workspace_root",
                    "projectRoot",
                    "project_root",
                ],
            )
        });
    let imported_metadata = imported.then(|| {
        continuation_metadata(
            metadata.unwrap_or(&Value::Null),
            thread_settings,
            imported_continuation,
        )
    });
    let settings_metadata = metadata
        .map(thread_settings_runtime_metadata)
        .filter(|value| value.as_object().is_some_and(|value| !value.is_empty()));
    let runtime_metadata = merge_json_objects(imported_metadata, settings_metadata);
    let request = RuntimeRequest {
        provider_preference,
        model_preference,
        collaboration_mode: collaboration_mode.clone(),
        reasoning_effort: collaboration_mode
            .as_ref()
            .and_then(|mode| mode.settings.reasoning_effort.clone())
            .or_else(|| {
                string_alias_from_sources(
                    &sources,
                    &["effort", "reasoningEffort", "reasoning_effort"],
                )
            }),
        service_tier: string_alias_from_sources(&sources, &["serviceTier", "service_tier"]),
        approval_policy: string_alias_from_sources(
            &sources,
            &["approvalPolicy", "approval_policy"],
        ),
        sandbox_policy: string_alias_from_sources(
            &sources,
            &["sandboxPolicy", "sandbox_policy", "sandbox"],
        ),
        workspace_id: session.workspace_id.clone(),
        working_dir: working_dir.clone(),
        workspace_root: workspace_root.clone(),
        project_root: workspace_root.or(working_dir),
        execution_strategy: string_alias_from_sources(
            &sources,
            &["executionStrategy", "execution_strategy"],
        ),
        system_prompt: collaboration_mode.and_then(|mode| mode.settings.developer_instructions),
        metadata: runtime_metadata,
        ..RuntimeRequest::default()
    };
    if request == RuntimeRequest::default() {
        return Ok(None);
    }
    Ok(Some(RuntimeOptions {
        runtime_request: Some(request),
        ..RuntimeOptions::default()
    }))
}

pub(super) fn merge_with_request_options(
    defaults: RuntimeOptions,
    request: Option<RuntimeOptions>,
) -> RuntimeOptions {
    let Some(request) = request else {
        return defaults;
    };

    RuntimeOptions {
        capability_id: request.capability_id.or(defaults.capability_id),
        stream: request.stream || defaults.stream,
        event_name: request.event_name.or(defaults.event_name),
        queued_turn_id: request.queued_turn_id.or(defaults.queued_turn_id),
        runtime_request: merge_runtime_requests(defaults.runtime_request, request.runtime_request),
        expected_output: request.expected_output.or(defaults.expected_output),
        structured_output: request.structured_output.or(defaults.structured_output),
        output_schema: request.output_schema.or(defaults.output_schema),
    }
}

fn merge_runtime_requests(
    defaults: Option<RuntimeRequest>,
    request: Option<RuntimeRequest>,
) -> Option<RuntimeRequest> {
    let (defaults, request) = match (defaults, request) {
        (Some(defaults), Some(request)) => (defaults, request),
        (defaults, request) => return request.or(defaults),
    };

    Some(RuntimeRequest {
        provider_config: request.provider_config.or(defaults.provider_config),
        provider_preference: request.provider_preference.or(defaults.provider_preference),
        model_preference: request.model_preference.or(defaults.model_preference),
        collaboration_mode: request.collaboration_mode.or(defaults.collaboration_mode),
        reasoning_effort: request.reasoning_effort.or(defaults.reasoning_effort),
        service_tier: request.service_tier.or(defaults.service_tier),
        thinking_enabled: request.thinking_enabled.or(defaults.thinking_enabled),
        approval_policy: request.approval_policy.or(defaults.approval_policy),
        sandbox_policy: request.sandbox_policy.or(defaults.sandbox_policy),
        workspace_id: request.workspace_id.or(defaults.workspace_id),
        working_dir: request.working_dir.or(defaults.working_dir),
        workspace_root: request.workspace_root.or(defaults.workspace_root),
        project_root: request.project_root.or(defaults.project_root),
        web_search: request.web_search.or(defaults.web_search),
        search_mode: request.search_mode.or(defaults.search_mode),
        execution_strategy: request.execution_strategy.or(defaults.execution_strategy),
        auto_continue: request.auto_continue.or(defaults.auto_continue),
        system_prompt: request.system_prompt.or(defaults.system_prompt),
        metadata: merge_json_objects(defaults.metadata, request.metadata),
    })
}

fn continuation_metadata(
    metadata: &Value,
    thread_settings: Option<&Value>,
    imported_continuation: Option<&Value>,
) -> Value {
    compact_json(json!({
        "imported": true,
        "sourceClient": string_alias(metadata, &["sourceClient", "source_client"]),
        "sourceThreadId": string_alias(metadata, &["sourceThreadId", "source_thread_id"]),
        "sourceRoot": string_alias(metadata, &["sourceRoot", "source_root"]),
        "sourcePath": string_alias(metadata, &["sourcePath", "source_path"]),
        "statePath": string_alias(metadata, &["statePath", "state_path"]),
        "threadSource": string_alias(metadata, &["threadSource", "thread_source"]),
        "memoryMode": string_alias(metadata, &["memoryMode", "memory_mode"]),
        "agentPath": string_alias(metadata, &["agentPath", "agent_path"]),
        "cliVersion": string_alias(metadata, &["cliVersion", "cli_version"]),
        "gitSha": string_alias(metadata, &["gitSha", "git_sha"]),
        "gitBranch": string_alias(metadata, &["gitBranch", "git_branch"]),
        "gitOriginUrl": string_alias(metadata, &["gitOriginUrl", "git_origin_url"]),
        "importedThreadSettings": thread_settings.cloned(),
        "importedContinuation": imported_continuation.cloned(),
    }))
}

fn thread_settings_runtime_metadata(metadata: &Value) -> Value {
    compact_json(json!({
        "serviceTier": metadata.get("serviceTier").cloned(),
        "approvalsReviewer": metadata.get("approvalsReviewer").cloned(),
        "reasoningSummary": metadata
            .get("reasoningSummary")
            .or_else(|| metadata.get("summary"))
            .cloned(),
        "personality": metadata.get("personality").cloned(),
        "memoryMode": metadata
            .get("memoryMode")
            .or_else(|| metadata.get("memory_mode"))
            .cloned(),
    }))
}

fn string_alias(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(*key))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn string_alias_from_sources(sources: &[Option<&Value>], keys: &[&str]) -> Option<String> {
    sources
        .iter()
        .filter_map(|source| *source)
        .find_map(|source| string_alias(source, keys))
}

fn first_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(*key))
        .and_then(Value::as_array)
        .and_then(|values| values.iter().find_map(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn merge_json_objects(defaults: Option<Value>, request: Option<Value>) -> Option<Value> {
    match (defaults, request) {
        (Some(Value::Object(mut defaults)), Some(Value::Object(request))) => {
            merge_object(&mut defaults, request);
            Some(Value::Object(defaults))
        }
        (_, Some(request)) => Some(request),
        (defaults, None) => defaults,
    }
}

fn merge_object(target: &mut Map<String, Value>, request: Map<String, Value>) {
    for (key, value) in request {
        match (target.get_mut(&key), value) {
            (Some(Value::Object(target)), Value::Object(request)) => {
                merge_object(target, request);
            }
            (_, value) => {
                target.insert(key, value);
            }
        }
    }
}

fn compact_json(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut compacted = Map::new();
            for (key, value) in object {
                let value = compact_json(value);
                if !value.is_null() {
                    compacted.insert(key, value);
                }
            }
            Value::Object(compacted)
        }
        Value::Array(values) => Value::Array(values.into_iter().map(compact_json).collect()),
        value => value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::{AgentSessionStatus, BusinessObjectRef};

    fn session(kind: &str, metadata: Value) -> AgentSession {
        AgentSession {
            session_id: "session-1".to_string(),
            thread_id: "thread-1".to_string(),
            app_id: "agent-chat".to_string(),
            workspace_id: Some("workspace-1".to_string()),
            business_object_ref: Some(BusinessObjectRef {
                kind: kind.to_string(),
                id: "thread-1".to_string(),
                title: None,
                uri: None,
                metadata: Some(metadata),
            }),
            status: AgentSessionStatus::Idle,
            created_at: "2026-07-19T00:00:00Z".to_string(),
            updated_at: "2026-07-19T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn current_thread_metadata_becomes_typed_sticky_defaults() {
        let options = default_runtime_options_for_session(&session(
            "agent.thread",
            json!({
                "providerSelector": "openai",
                "providerName": "openai",
                "modelName": "gpt-5.4",
                "workingDir": "/workspace",
                "runtimeWorkspaceRoots": ["/workspace", "/shared"],
                "approvalPolicy": "on-request",
                "sandbox": "workspace-write"
            }),
        ))
        .expect("valid metadata")
        .expect("defaults");
        let request = options.runtime_request.expect("runtime request");

        assert_eq!(request.provider_preference.as_deref(), Some("openai"));
        assert_eq!(request.model_preference.as_deref(), Some("gpt-5.4"));
        assert_eq!(request.working_dir.as_deref(), Some("/workspace"));
        assert_eq!(request.workspace_root.as_deref(), Some("/workspace"));
        assert_eq!(request.approval_policy.as_deref(), Some("on-request"));
        assert_eq!(request.sandbox_policy.as_deref(), Some("workspace-write"));
        assert_eq!(request.workspace_id.as_deref(), Some("workspace-1"));
        assert!(request.metadata.is_none());
    }

    #[test]
    fn request_model_override_keeps_durable_provider() {
        let defaults = default_runtime_options_for_session(&session(
            "agent.thread",
            json!({
                "providerSelector": "openai",
                "modelName": "gpt-4.1"
            }),
        ))
        .expect("valid metadata")
        .expect("defaults");
        let request = RuntimeOptions {
            runtime_request: Some(RuntimeRequest {
                model_preference: Some("gpt-5.4".to_string()),
                ..RuntimeRequest::default()
            }),
            ..RuntimeOptions::default()
        };

        let merged = merge_with_request_options(defaults, Some(request));
        let runtime_request = merged.runtime_request.expect("runtime request");

        assert_eq!(
            runtime_request.provider_preference.as_deref(),
            Some("openai")
        );
        assert_eq!(runtime_request.model_preference.as_deref(), Some("gpt-5.4"));
    }

    #[test]
    fn imported_source_route_is_not_promoted_without_current_selector() {
        let options = default_runtime_options_for_session(&session(
            IMPORTED_CONVERSATION_KIND,
            json!({
                "providerName": "openai",
                "modelName": "gpt-5.4",
                "importedContinuation": {
                    "cwd": "/imported",
                    "approvalPolicy": "never"
                }
            }),
        ))
        .expect("valid metadata")
        .expect("continuation defaults");
        let request = options.runtime_request.expect("runtime request");

        assert!(request.provider_preference.is_none());
        assert!(request.model_preference.is_none());
        assert_eq!(request.working_dir.as_deref(), Some("/imported"));
        assert_eq!(request.approval_policy.as_deref(), Some("never"));
        assert_eq!(
            request
                .metadata
                .as_ref()
                .and_then(|value| value.get("imported")),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn typed_collaboration_mode_becomes_the_sticky_runtime_owner() {
        let options = default_runtime_options_for_session(&session(
            "agent.thread",
            json!({
                "providerSelector": "openai",
                "modelName": "stale-model",
                "reasoningEffort": "low",
                "collaborationMode": {
                    "mode": "plan",
                    "settings": {
                        "model": "gpt-5.4",
                        "reasoning_effort": "high",
                        "developer_instructions": "Plan before editing."
                    }
                }
            }),
        ))
        .expect("valid typed mode")
        .expect("defaults");
        let request = options.runtime_request.expect("runtime request");

        assert_eq!(request.model_preference.as_deref(), Some("gpt-5.4"));
        assert_eq!(request.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(
            request.system_prompt.as_deref(),
            Some("Plan before editing.")
        );
        assert_eq!(
            request.collaboration_mode.as_ref().map(|mode| mode.mode),
            Some(agent_protocol::ModeKind::Plan)
        );
        assert!(
            request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("collaborationMode"))
                .is_none(),
            "typed collaboration mode must not leak into generic runtime metadata"
        );
    }

    #[test]
    fn invalid_persisted_collaboration_mode_fails_closed() {
        let error = default_runtime_options_for_session(&session(
            "agent.thread",
            json!({
                "providerSelector": "openai",
                "modelName": "gpt-5.4",
                "collaborationMode": "plan"
            }),
        ))
        .expect_err("invalid typed mode must fail closed");

        assert!(matches!(
            error,
            RuntimeCoreError::InvalidRequest(message)
                if message.contains("invalid persisted collaborationMode")
        ));
    }
}
