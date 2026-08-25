use crate::trace_context::w3c_trace_context;
use crate::ExecutionRequest;
use agent_protocol::world_state::{
    RuntimeWorldEnvironment, RuntimeWorldEnvironmentSelection, RuntimeWorldEnvironmentStatus,
    RuntimeWorldMode, RuntimeWorldPermissions, RuntimeWorldState, WORLD_STATE_SOURCE,
    WORLD_STATE_TURN_METADATA_KEY,
};
use agent_protocol::{CollaborationMode, ModeKind, MultiAgentMode};
use lime_agent::{
    build_agent_turn_context, AgentTurnContext, AgentTurnContextConfigurationRequest,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;

use super::{
    host_approval_policy, host_metadata_value, host_sandbox_policy, host_thinking_enabled,
    non_empty, request_tool_policy_from_request, request_workspace_scope, RuntimeModelSelection,
    RuntimeRequest, RuntimeSessionScope,
};

const LIME_RUNTIME_METADATA_KEY: &str = "lime_runtime";
const LIME_RUNTIME_CONTEXT_POLICY_KEY: &str = "context_policy";
const DEFAULT_EFFECTIVE_CONTEXT_WINDOW_PERCENT: i64 = 95;
const AUTO_COMPACT_CONTEXT_WINDOW_RATIO_NUMERATOR: i64 = 9;
const AUTO_COMPACT_CONTEXT_WINDOW_RATIO_DENOMINATOR: i64 = 10;
const TRACE_METADATA_KEYS: &[&str] = &["agentUiPerformanceTrace", "agent_ui_performance_trace"];

pub(in crate::runtime_backend) fn turn_context_from_request(
    request: &ExecutionRequest,
    host_request: Option<&RuntimeRequest>,
    scope: &RuntimeSessionScope,
    selection: &RuntimeModelSelection,
    config_metadata: Option<Value>,
) -> Option<AgentTurnContext> {
    let workspace_scope = request_workspace_scope(request, host_request);
    let mut metadata = HashMap::new();
    metadata.insert(
        "app_server_runtime_backend".to_string(),
        json!({
            "sessionId": scope.session_id,
            "threadId": scope.thread_id,
            "turnId": scope.turn_id,
            "workspaceId": scope.workspace_id,
            "workingDir": workspace_scope
                .working_dir
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
            "projectRoot": workspace_scope
                .project_root
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
            "thinkingEnabled": host_request.and_then(host_thinking_enabled),
        }),
    );
    if let Some(host_metadata) = host_request.and_then(host_metadata_value) {
        metadata.insert("runtime_request".to_string(), host_metadata);
    }
    if request_tool_policy_from_request(host_request).allows_web_search() {
        metadata.insert("web_search_enabled".to_string(), json!(true));
        metadata.insert("webSearchEnabled".to_string(), json!(true));
    }
    if let Some(w3c_trace_context) = w3c_trace_context_metadata_from_request(request) {
        metadata.insert("w3c_trace_context".to_string(), w3c_trace_context);
    }
    if let Some(context_policy) = lime_runtime_context_policy_from_request(request) {
        merge_lime_runtime_metadata(&mut metadata, context_policy);
    }
    if let Some(turn_policy) = super::app_server_turn_policy_runtime(request) {
        merge_lime_runtime_metadata(&mut metadata, Value::Object(turn_policy.clone()));
    }
    if let Some(config_metadata) = config_metadata {
        metadata.insert("config".to_string(), config_metadata);
    }
    let world_state = world_state_from_request(
        request,
        host_request,
        scope,
        selection,
        workspace_scope.working_dir.as_deref(),
        workspace_scope.project_root.as_deref(),
    );
    let primary_environment_cwd = world_state.as_ref().and_then(|state| {
        state
            .environments
            .iter()
            .find(|environment| environment.primary)
            .map(|environment| environment.cwd.clone().into())
    });
    if let Some(world_state) = world_state {
        metadata.insert(
            WORLD_STATE_TURN_METADATA_KEY.to_string(),
            serde_json::to_value(world_state).expect("world state must serialize"),
        );
    }
    build_agent_turn_context(AgentTurnContextConfigurationRequest {
        cwd: primary_environment_cwd.or_else(|| workspace_scope.working_dir.clone()),
        model: Some(selection.model.clone()),
        effort: selection.reasoning_effort.clone(),
        approval_policy: host_request.and_then(host_approval_policy),
        sandbox_policy: host_request.and_then(host_sandbox_policy),
        collaboration_mode: collaboration_mode_from_request(request, host_request),
        user_visible_input_text: (!request.input.agent_only)
            .then(|| request.input.concat_text())
            .and_then(|text| non_empty(Some(&text))),
        output_schema: output_schema_from_request(request, host_request),
        metadata,
    })
}

fn world_state_from_request(
    request: &ExecutionRequest,
    host_request: Option<&RuntimeRequest>,
    scope: &RuntimeSessionScope,
    selection: &RuntimeModelSelection,
    working_dir: Option<&Path>,
    project_root: Option<&Path>,
) -> Option<RuntimeWorldState> {
    let environments = world_environments_from_request(request);
    let primary_cwd = environments
        .iter()
        .find(|environment| environment.primary)
        .map(|environment| environment.cwd.clone());
    let collaboration =
        collaboration_mode_from_request(request, host_request).map(|mode| RuntimeWorldMode {
            mode: match mode.mode {
                ModeKind::Plan => "plan",
                ModeKind::Default => "default",
            }
            .to_string(),
            source: Some("runtime_request".to_string()),
        });
    let permissions = Some(RuntimeWorldPermissions {
        approval_policy: host_request.and_then(host_approval_policy),
        sandbox_policy: host_request.and_then(host_sandbox_policy),
        web_search: Some(request_tool_policy_from_request(host_request).allows_web_search()),
    });
    let state = RuntimeWorldState {
        environment: Some(RuntimeWorldEnvironment {
            cwd: primary_cwd
                .or_else(|| working_dir.map(|path| path.to_string_lossy().into_owned())),
            project_root: project_root.map(|path| path.to_string_lossy().into_owned()),
            workspace_id: scope.workspace_id.clone(),
            thread_id: Some(scope.thread_id.clone()),
            turn_id: Some(scope.turn_id.clone()),
            provider: Some(selection.provider.clone()),
            model: Some(selection.model.clone()),
            reasoning_effort: selection.reasoning_effort.clone(),
        }),
        environments,
        permissions,
        collaboration,
        multi_agent: Some(MultiAgentMode::from_reasoning_effort(
            selection.reasoning_effort.as_deref(),
        )),
        instruction_sections: Vec::new(),
        source: Some(WORLD_STATE_SOURCE.to_string()),
    };
    (!state.is_empty()).then_some(state)
}

fn world_environments_from_request(
    request: &ExecutionRequest,
) -> Vec<RuntimeWorldEnvironmentSelection> {
    if let Some(mut snapshot) = request
        .runtime_metadata()
        .and_then(|metadata| metadata.get("environmentWorldState"))
        .cloned()
        .and_then(|value| {
            serde_json::from_value::<Vec<RuntimeWorldEnvironmentSelection>>(value).ok()
        })
    {
        snapshot.sort_by(|left, right| left.environment_id.cmp(&right.environment_id));
        return snapshot;
    }
    let Some(value) = request
        .runtime_metadata()
        .and_then(|metadata| metadata.get("environments"))
        .cloned()
    else {
        return Vec::new();
    };
    let Ok(selections) = serde_json::from_value::<
        Vec<app_server_protocol::protocol::v2::TurnEnvironmentParams>,
    >(value) else {
        return Vec::new();
    };
    let primary_environment_id = selections
        .first()
        .map(|selection| selection.environment_id.clone());
    let mut environments = selections
        .into_iter()
        .map(|selection| RuntimeWorldEnvironmentSelection {
            primary: primary_environment_id.as_deref() == Some(selection.environment_id.as_str()),
            status: (selection.environment_id == "local")
                .then_some(RuntimeWorldEnvironmentStatus::Ready),
            environment_id: selection.environment_id,
            cwd: selection.cwd,
            runtime_workspace_roots: selection.runtime_workspace_roots.unwrap_or_default(),
            shell: None,
        })
        .collect::<Vec<_>>();
    environments.sort_by(|left, right| left.environment_id.cmp(&right.environment_id));
    environments
}

fn w3c_trace_context_metadata_from_request(request: &ExecutionRequest) -> Option<Value> {
    request
        .runtime_metadata()
        .and_then(Value::as_object)
        .and_then(w3c_trace_context_metadata)
}

fn w3c_trace_context_metadata(metadata: &serde_json::Map<String, Value>) -> Option<Value> {
    let trace = TRACE_METADATA_KEYS
        .iter()
        .filter_map(|key| metadata.get(*key))
        .find_map(Value::as_object)?;
    let w3c = w3c_trace_context(trace)?;
    let mut payload = serde_json::Map::new();
    payload.insert("traceparent".to_string(), Value::String(w3c.traceparent));
    if let Some(tracestate) = w3c.tracestate {
        payload.insert("tracestate".to_string(), Value::String(tracestate));
    }
    Some(Value::Object(payload))
}

fn merge_lime_runtime_metadata(metadata: &mut HashMap<String, Value>, patch: Value) {
    let Some(patch_object) = patch.as_object() else {
        return;
    };
    if patch_object.is_empty() {
        return;
    }

    let runtime = metadata
        .entry(LIME_RUNTIME_METADATA_KEY.to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if !runtime.is_object() {
        *runtime = Value::Object(serde_json::Map::new());
    }
    let runtime_object = runtime.as_object_mut().expect("lime_runtime object");
    for (key, value) in patch_object {
        runtime_object.insert(key.clone(), value.clone());
    }
}

fn lime_runtime_context_policy_from_request(request: &ExecutionRequest) -> Option<Value> {
    request
        .runtime_metadata()
        .and_then(lime_runtime_context_policy_from_metadata)
}

fn lime_runtime_context_policy_from_metadata(metadata: &Value) -> Option<Value> {
    let policy = [
        "/harness/model_request_policy/context_policy",
        "/harness/modelRequestPolicy/contextPolicy",
        "/model_request_policy/context_policy",
        "/modelRequestPolicy/contextPolicy",
    ]
    .into_iter()
    .find_map(|pointer| metadata.pointer(pointer))?;

    let context_window = positive_i64_field(policy, &["context_window", "contextWindow"]);
    let max_context_window =
        positive_i64_field(policy, &["max_context_window", "maxContextWindow"]);
    let resolved_context_window = positive_i64_field(
        policy,
        &["resolved_context_window", "resolvedContextWindow"],
    )
    .or(context_window)
    .or(max_context_window);
    let effective_context_window_percent = positive_i64_field(
        policy,
        &[
            "effective_context_window_percent",
            "effectiveContextWindowPercent",
        ],
    )
    .filter(|percent| *percent <= 100)
    .unwrap_or(DEFAULT_EFFECTIVE_CONTEXT_WINDOW_PERCENT);
    let model_context_window =
        positive_i64_field(policy, &["model_context_window", "modelContextWindow"]).or_else(|| {
            resolved_context_window
                .map(|window| window.saturating_mul(effective_context_window_percent) / 100)
        });
    let auto_compact_token_limit = positive_i64_field(
        policy,
        &["auto_compact_token_limit", "autoCompactTokenLimit"],
    )
    .map(|limit| {
        resolved_context_window.map_or(limit, |window| {
            let max_limit = window.saturating_mul(AUTO_COMPACT_CONTEXT_WINDOW_RATIO_NUMERATOR)
                / AUTO_COMPACT_CONTEXT_WINDOW_RATIO_DENOMINATOR;
            limit.min(max_limit)
        })
    })
    .or_else(|| {
        resolved_context_window.map(|window| {
            window.saturating_mul(AUTO_COMPACT_CONTEXT_WINDOW_RATIO_NUMERATOR)
                / AUTO_COMPACT_CONTEXT_WINDOW_RATIO_DENOMINATOR
        })
    });

    if resolved_context_window.is_none()
        && model_context_window.is_none()
        && auto_compact_token_limit.is_none()
    {
        return None;
    }

    let mut context_policy = serde_json::Map::new();
    context_policy.insert("source".to_string(), json!("model_request_policy"));
    if let Some(value) = context_window {
        context_policy.insert("context_window".to_string(), json!(value));
    }
    if let Some(value) = max_context_window {
        context_policy.insert("max_context_window".to_string(), json!(value));
    }
    if let Some(value) = resolved_context_window {
        context_policy.insert("resolved_context_window".to_string(), json!(value));
    }
    context_policy.insert(
        "effective_context_window_percent".to_string(),
        json!(effective_context_window_percent),
    );
    if let Some(value) = model_context_window {
        context_policy.insert("model_context_window".to_string(), json!(value));
    }
    if let Some(value) = auto_compact_token_limit {
        context_policy.insert("auto_compact_token_limit".to_string(), json!(value));
    }

    let mut runtime = serde_json::Map::new();
    runtime.insert(
        LIME_RUNTIME_CONTEXT_POLICY_KEY.to_string(),
        Value::Object(context_policy),
    );
    if let Some(value) = model_context_window {
        runtime.insert("model_context_window".to_string(), json!(value));
    }
    if let Some(value) = auto_compact_token_limit {
        runtime.insert("auto_compact_token_limit".to_string(), json!(value));
    }

    Some(Value::Object(runtime))
}

fn output_schema_from_request(
    request: &ExecutionRequest,
    _host_request: Option<&RuntimeRequest>,
) -> Option<Value> {
    request
        .output_schema
        .clone()
        .or_else(|| {
            request
                .structured_output
                .as_ref()
                .and_then(|value| value.schema.clone())
        })
        .or_else(|| output_schema_from_expected_output(request.expected_output.as_ref()))
        .or_else(|| {
            request
                .runtime_options
                .as_ref()
                .and_then(|options| options.output_schema.clone())
        })
        .or_else(|| {
            request
                .runtime_options
                .as_ref()
                .and_then(|options| options.structured_output.as_ref())
                .and_then(|value| value.schema.clone())
        })
        .or_else(|| {
            request.runtime_options.as_ref().and_then(|options| {
                output_schema_from_expected_output(options.expected_output.as_ref())
            })
        })
}

fn collaboration_mode_from_request(
    request: &ExecutionRequest,
    host_request: Option<&RuntimeRequest>,
) -> Option<CollaborationMode> {
    host_request
        .and_then(|host| host.collaboration_mode.clone())
        .or_else(|| {
            request
                .runtime_options
                .as_ref()
                .and_then(|options| options.runtime_request())
                .and_then(|runtime_request| runtime_request.collaboration_mode.clone())
        })
}

fn positive_i64_field(value: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(|field| {
            field
                .as_i64()
                .or_else(|| field.as_u64().and_then(|value| i64::try_from(value).ok()))
                .filter(|value| *value > 0)
        })
}

fn output_schema_from_expected_output(value: Option<&Value>) -> Option<Value> {
    output_schema_from_expected_output_value(value?).cloned()
}

fn output_schema_from_expected_output_value(value: &Value) -> Option<&Value> {
    if let Some(schema) = value
        .get("outputFormat")
        .or_else(|| value.get("output_format"))
        .and_then(output_schema_from_output_format)
    {
        return Some(schema);
    }
    output_schema_from_output_format(value)
}

fn output_schema_from_output_format(value: &Value) -> Option<&Value> {
    value
        .get("schema")
        .or_else(|| value.get("outputSchema"))
        .or_else(|| value.get("output_schema"))
}
