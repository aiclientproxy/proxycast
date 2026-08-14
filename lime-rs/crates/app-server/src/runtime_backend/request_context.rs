use crate::ExecutionRequest;
use crate::RuntimeCoreError;
use app_server_protocol::{
    RuntimeOptions, RuntimeProviderConfig, RuntimeRequest, RuntimeSearchMode,
    RuntimeToolCallStrategy,
};
use lime_agent::{
    request_tool_policy_with_additional_required_tools, resolve_request_tool_policy_with_mode,
    RequestToolPolicy, RequestToolPolicyMode, SessionProviderConfig,
};
pub(super) use runtime_core::RuntimeModelSelection;
use serde_json::Value;
use std::path::PathBuf;

const LIME_RUNTIME_COMPACT_TOOLS_TOOL_SURFACE: &str = "compact_tools";
const LIME_RUNTIME_METADATA_KEY: &str = "lime_runtime";
const LIME_RUNTIME_MODEL_SLOT_KEY: &str = "model_slot";
const LIME_RUNTIME_TOOL_SURFACE_KEY: &str = "tool_surface";
const LIME_RUNTIME_AUTO_COMPACT_KEY: &str = "auto_compact";
const LIME_RUNTIME_SOURCE_KEY: &str = "source";
pub(super) const INPUT_MENTIONS_TURN_METADATA_KEY: &str = "input_mentions";
const APP_SERVER_TURN_POLICY_SOURCE: &str = "app_server_turn_policy";
const HARNESS_TURN_TOOL_SURFACE_POINTER: &str = "/harness/turn_policy/tool_surface";
const HARNESS_TURN_POLICY_SOURCE: &str = "harness_turn_policy";
const RESPONSIVE_CHAT_MODEL_SLOT: &str = "fast";

mod session_config;
mod turn_context;
mod workspace_scope;

#[cfg(test)]
pub(super) use session_config::session_config_from_request;
pub(super) use session_config::session_config_from_request_with_plugins_and_orchestrator;
pub(super) use turn_context::turn_context_from_request;
pub(super) use workspace_scope::request_workspace_scope;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnToolSurface {
    DirectAnswer,
    Full,
    CompactTools,
}

impl TurnToolSurface {
    fn uses_light_session_prompt(self) -> bool {
        matches!(self, Self::DirectAnswer | Self::CompactTools)
    }

    fn is_harness_direct_answer(self) -> bool {
        matches!(self, Self::DirectAnswer)
    }
}

pub(super) fn apply_app_server_turn_policy(
    request: &mut ExecutionRequest,
    first_sampling_turn: bool,
    request_tool_policy: &RequestToolPolicy,
) {
    let use_responsive_profile =
        should_use_responsive_chat_profile(request, first_sampling_turn, request_tool_policy);
    let options = request.runtime_options.get_or_insert_with(Default::default);
    let metadata = options
        .runtime_metadata_mut()
        .get_or_insert_with(|| Value::Object(Default::default()));
    if !metadata.is_object() {
        *metadata = Value::Object(Default::default());
    }
    let metadata_object = metadata.as_object_mut().expect("runtime metadata object");
    let runtime = metadata_object
        .entry(LIME_RUNTIME_METADATA_KEY.to_string())
        .or_insert_with(|| Value::Object(Default::default()));
    if !runtime.is_object() {
        *runtime = Value::Object(Default::default());
    }
    let runtime = runtime.as_object_mut().expect("lime_runtime object");
    runtime.remove(LIME_RUNTIME_MODEL_SLOT_KEY);
    runtime.remove(LIME_RUNTIME_TOOL_SURFACE_KEY);
    runtime.remove(LIME_RUNTIME_AUTO_COMPACT_KEY);
    if runtime.get(LIME_RUNTIME_SOURCE_KEY).and_then(Value::as_str)
        == Some(APP_SERVER_TURN_POLICY_SOURCE)
    {
        runtime.remove(LIME_RUNTIME_SOURCE_KEY);
    }

    if use_responsive_profile {
        runtime.insert(
            LIME_RUNTIME_MODEL_SLOT_KEY.to_string(),
            Value::String(RESPONSIVE_CHAT_MODEL_SLOT.to_string()),
        );
        runtime.insert(
            LIME_RUNTIME_TOOL_SURFACE_KEY.to_string(),
            Value::String(LIME_RUNTIME_COMPACT_TOOLS_TOOL_SURFACE.to_string()),
        );
        runtime.insert(
            LIME_RUNTIME_AUTO_COMPACT_KEY.to_string(),
            Value::Bool(false),
        );
        runtime.insert(
            LIME_RUNTIME_SOURCE_KEY.to_string(),
            Value::String(APP_SERVER_TURN_POLICY_SOURCE.to_string()),
        );
    }

    if runtime.is_empty() {
        metadata_object.remove(LIME_RUNTIME_METADATA_KEY);
    }
}

pub(super) fn selection_with_capability_reasoning(
    selection: &RuntimeModelSelection,
    capability_snapshot: &app_server_protocol::CapabilitySnapshot,
) -> RuntimeModelSelection {
    let capability = super::model_capability::resolve_model_capability(
        super::model_capability::ModelRef::new(selection.provider.clone(), selection.model.clone()),
        Some(capability_snapshot),
    );
    let policy = super::model_capability::resolve_reasoning_policy(
        &capability,
        selection.reasoning_effort.as_deref(),
    );
    RuntimeModelSelection {
        provider: selection.provider.clone(),
        model: selection.model.clone(),
        source: selection.source,
        reasoning_effort: policy.effective_level,
    }
}

pub(super) fn effective_runtime_options_for_turn(
    request: &ExecutionRequest,
    first_sampling_turn: bool,
) -> Option<RuntimeOptions> {
    let mut effective_request = request.clone();
    let initial_host_request = runtime_request_from_request(&effective_request);
    let initial_tool_policy = request_tool_policy_from_request(initial_host_request.as_ref());
    apply_app_server_turn_policy(
        &mut effective_request,
        first_sampling_turn,
        &initial_tool_policy,
    );

    let selection = resolve_runtime_model_selection(&effective_request).ok()?;
    let scope = session_scope_from_request(&effective_request).ok()?;
    let workspace_scope = request_workspace_scope(
        &effective_request,
        runtime_request_from_request(&effective_request).as_ref(),
    );
    let request_tool_policy =
        request_tool_policy_from_request(runtime_request_from_request(&effective_request).as_ref());
    let mut options = effective_request.runtime_options.unwrap_or_default();
    let runtime_request = options.runtime_request_mut();
    runtime_request.provider_preference = Some(selection.provider);
    runtime_request.model_preference = Some(selection.model);
    runtime_request.reasoning_effort = selection.reasoning_effort;
    runtime_request.workspace_id = scope.workspace_id;
    if let Some(working_dir) = workspace_scope.working_dir {
        runtime_request.working_dir = Some(working_dir.to_string_lossy().into_owned());
    }
    if let Some(project_root) = workspace_scope.project_root {
        let project_root = project_root.to_string_lossy().into_owned();
        runtime_request.workspace_root = Some(project_root.clone());
        runtime_request.project_root = Some(project_root);
    }
    runtime_request.web_search = Some(request_tool_policy.allows_web_search());
    runtime_request.search_mode = Some(match request_tool_policy.search_mode {
        RequestToolPolicyMode::Disabled => RuntimeSearchMode::Disabled,
        RequestToolPolicyMode::Auto => RuntimeSearchMode::Auto,
        RequestToolPolicyMode::Required => RuntimeSearchMode::Required,
    });
    Some(options)
}

pub(super) fn resolve_runtime_model_selection(
    request: &ExecutionRequest,
) -> Result<RuntimeModelSelection, RuntimeCoreError> {
    let metadata_values = runtime_model_metadata_values(request);
    let preferred_slot = app_server_turn_policy_value(request, LIME_RUNTIME_MODEL_SLOT_KEY);
    if let Some(preferred_slot) = preferred_slot {
        if let Some(selection) = runtime_core::selection_from_profile_model_slot(
            &metadata_values,
            reasoning_effort_from_request(request),
            Some(preferred_slot),
        ) {
            return Ok(selection);
        }
    }
    if let Some(selection) = selection_from_explicit_preferences(request) {
        return Ok(selection);
    }
    if let Some(selection) = selection_from_host_provider_config(request) {
        return Ok(selection);
    }
    if let Some(selection) = runtime_core::selection_from_profile_model_slot(
        &metadata_values,
        reasoning_effort_from_request(request),
        None,
    ) {
        return Ok(selection);
    }
    if let Some(selection) = selection_from_session_default(request) {
        return Ok(selection);
    }

    Err(RuntimeCoreError::pending_route_for_session(
        request.session.session_id.clone(),
        request.runtime_options.as_ref(),
    ))
}

pub(super) fn selection_from_explicit_preferences(
    request: &ExecutionRequest,
) -> Option<RuntimeModelSelection> {
    let runtime_request = request.runtime_request()?;
    let explicit_provider = non_empty(runtime_request.provider_preference.as_deref());
    let explicit_model = non_empty(runtime_request.model_preference.as_deref());
    if explicit_provider.is_none() && explicit_model.is_none() {
        return None;
    }
    let session_default = selection_from_session_default(request);
    let provider = explicit_provider.clone().or_else(|| {
        session_default
            .as_ref()
            .map(|selection| selection.provider.clone())
    })?;
    let model = explicit_model.clone().or_else(|| {
        session_default
            .as_ref()
            .map(|selection| selection.model.clone())
    })?;
    Some(RuntimeModelSelection {
        provider,
        model,
        source: if explicit_provider.is_some() && explicit_model.is_some() {
            "runtime_request"
        } else {
            "runtime_request_with_session_default"
        },
        reasoning_effort: reasoning_effort_from_request(request),
    })
}

pub(super) fn selection_from_host_provider_config(
    request: &ExecutionRequest,
) -> Option<RuntimeModelSelection> {
    let host_request = runtime_request_from_request(request)?;
    let provider_config = host_provider_config(&host_request);
    let provider = non_empty(
        host_provider_preference(&host_request)
            .as_deref()
            .or_else(|| provider_config.and_then(|config| config.provider_id.as_deref()))
            .or_else(|| provider_config.and_then(|config| config.provider_name.as_deref())),
    )?;
    let model = non_empty(
        host_model_preference(&host_request)
            .as_deref()
            .or_else(|| provider_config.and_then(|config| config.model_name.as_deref())),
    )?;
    Some(RuntimeModelSelection {
        provider,
        model,
        source: "runtime_request_provider_config",
        reasoning_effort: host_reasoning_effort(&host_request)
            .or_else(|| reasoning_effort_from_request(request)),
    })
}

pub(super) fn selection_from_session_default(
    request: &ExecutionRequest,
) -> Option<RuntimeModelSelection> {
    let business_object_ref = request.session.business_object_ref.as_ref()?;
    let metadata = business_object_ref.metadata.as_ref()?;
    let provider = session_default_provider(metadata)?;
    let model = session_default_model(metadata)?;
    if business_object_ref.kind == "conversation.import"
        && !session_default_has_current_provider_selector(metadata)
    {
        return None;
    }
    Some(RuntimeModelSelection {
        provider,
        model,
        source: "session_default",
        reasoning_effort: reasoning_effort_from_request(request),
    })
}

fn session_default_provider(metadata: &Value) -> Option<String> {
    json_pointer_string(
        metadata,
        &[
            "/providerSelector",
            "/provider_selector",
            "/executionRuntime/providerSelector",
            "/execution_runtime/provider_selector",
            "/extensionData/lime_provider_routing.v0/providerSelector",
            "/extensionData/lime_provider_routing.v0/provider_selector",
            "/providerName",
            "/provider_name",
            "/executionRuntime/providerName",
            "/execution_runtime/provider_name",
        ],
    )
}

fn session_default_has_current_provider_selector(metadata: &Value) -> bool {
    json_pointer_string(
        metadata,
        &[
            "/providerSelector",
            "/provider_selector",
            "/executionRuntime/providerSelector",
            "/execution_runtime/provider_selector",
            "/extensionData/lime_provider_routing.v0/providerSelector",
            "/extensionData/lime_provider_routing.v0/provider_selector",
        ],
    )
    .is_some()
}

fn session_default_model(metadata: &Value) -> Option<String> {
    json_pointer_string(
        metadata,
        &[
            "/modelName",
            "/model_name",
            "/model",
            "/executionRuntime/modelName",
            "/execution_runtime/model_name",
        ],
    )
}

pub(super) fn reasoning_effort_from_request(request: &ExecutionRequest) -> Option<String> {
    if let Some(reasoning_effort) =
        runtime_request_from_request(request).and_then(|host| host_reasoning_effort(&host))
    {
        return Some(reasoning_effort);
    }
    request
        .runtime_metadata()
        .and_then(metadata_reasoning_effort)
}

pub(super) fn service_tier_from_request(request: &ExecutionRequest) -> Option<String> {
    runtime_request_from_request(request).and_then(|host| non_empty(host.service_tier.as_deref()))
}

fn runtime_model_metadata_values(request: &ExecutionRequest) -> Vec<&Value> {
    request.runtime_metadata().into_iter().collect()
}

fn metadata_reasoning_effort(metadata: &Value) -> Option<String> {
    json_pointer_string(
        metadata,
        &[
            "/reasoning_effort",
            "/reasoningEffort",
            "/model_reasoning_effort",
            "/modelReasoningEffort",
            "/reasoning/effort",
            "/harness/reasoning_effort",
            "/harness/reasoningEffort",
            "/harness/model_reasoning_effort",
            "/harness/modelReasoningEffort",
            "/harness/reasoning/effort",
        ],
    )
    .or_else(|| metadata_model_request_policy_reasoning_effort(metadata))
}

fn metadata_model_request_policy_reasoning_effort(metadata: &Value) -> Option<String> {
    [
        "/harness/model_request_policy/reasoning_policy",
        "/harness/modelRequestPolicy/reasoningPolicy",
        "/model_request_policy/reasoning_policy",
        "/modelRequestPolicy/reasoningPolicy",
    ]
    .into_iter()
    .filter_map(|pointer| metadata.pointer(pointer))
    .find_map(|policy| {
        json_pointer_string(
            policy,
            &["/default_reasoning_level", "/defaultReasoningLevel"],
        )
    })
}

pub(super) fn session_scope_from_request(
    request: &ExecutionRequest,
) -> Result<RuntimeSessionScope, RuntimeCoreError> {
    let session_id = non_empty(Some(&request.session.session_id)).ok_or_else(|| {
        RuntimeCoreError::Backend(
            "App Server runtime backend session.sessionId is empty".to_string(),
        )
    })?;
    let session_thread_id = non_empty(Some(&request.session.thread_id)).ok_or_else(|| {
        RuntimeCoreError::Backend(
            "App Server runtime backend session.threadId is empty".to_string(),
        )
    })?;
    let turn_thread_id = non_empty(Some(&request.turn.thread_id)).ok_or_else(|| {
        RuntimeCoreError::Backend("App Server runtime backend turn.threadId is empty".to_string())
    })?;
    if turn_thread_id != session_thread_id {
        return Err(RuntimeCoreError::Backend(format!(
            "App Server runtime backend turn thread '{}' does not match session thread '{}'",
            turn_thread_id, session_thread_id
        )));
    }
    let thread_id = session_thread_id;
    let turn_id = non_empty(Some(&request.turn.turn_id)).ok_or_else(|| {
        RuntimeCoreError::Backend("App Server runtime backend turn.turnId is empty".to_string())
    })?;
    if let Some(turn_session_id) = non_empty(Some(&request.turn.session_id)) {
        if turn_session_id != session_id {
            return Err(RuntimeCoreError::Backend(format!(
                "App Server runtime backend turn session '{}' does not match session '{}'",
                turn_session_id, session_id
            )));
        }
    }
    Ok(RuntimeSessionScope {
        session_id,
        thread_id,
        turn_id,
        workspace_id: non_empty(request.session.workspace_id.as_deref()).or_else(|| {
            runtime_request_from_request(request)
                .and_then(|host| non_empty(host.workspace_id.as_deref()))
        }),
    })
}

pub(super) fn runtime_request_from_request(request: &ExecutionRequest) -> Option<RuntimeRequest> {
    request.runtime_request().cloned()
}

fn host_provider_config(host: &RuntimeRequest) -> Option<&RuntimeProviderConfig> {
    host.provider_config.as_ref()
}

pub(super) fn direct_provider_config_from_request(
    host_request: Option<&RuntimeRequest>,
    selection: &RuntimeModelSelection,
    reasoning_effort: Option<String>,
) -> Option<SessionProviderConfig> {
    let service_tier = host_request.and_then(|host| non_empty(host.service_tier.as_deref()));
    let request = host_request.and_then(host_provider_config)?;
    if request.api_key.is_none() && request.base_url.is_none() {
        return None;
    }

    let provider_name =
        non_empty(request.provider_name.as_deref()).or_else(|| Some(selection.provider.clone()))?;
    let provider_selector =
        non_empty(request.provider_id.as_deref()).or_else(|| Some(selection.provider.clone()));
    let model_name =
        non_empty(request.model_name.as_deref()).or_else(|| Some(selection.model.clone()))?;

    Some(SessionProviderConfig {
        provider_name,
        provider_selector,
        model_name,
        api_key: request.api_key.clone(),
        base_url: request.base_url.clone(),
        api_version: None,
        credential_uuid: None,
        reasoning_effort,
        service_tier,
        route_protocol: None,
        toolshim: matches!(
            request.tool_call_strategy,
            Some(RuntimeToolCallStrategy::ToolShim)
        ),
        toolshim_model: request.toolshim_model.clone(),
        model_capabilities: request.model_capabilities.clone(),
        supports_websockets: request.supports_websockets.unwrap_or(false),
    })
}

fn host_provider_preference(host: &RuntimeRequest) -> Option<String> {
    non_empty(host.provider_preference.as_deref())
}

fn host_model_preference(host: &RuntimeRequest) -> Option<String> {
    non_empty(host.model_preference.as_deref())
}

pub(super) fn host_reasoning_effort(host: &RuntimeRequest) -> Option<String> {
    non_empty(host.reasoning_effort.as_deref())
}

pub(super) fn host_thinking_enabled(host: &RuntimeRequest) -> Option<bool> {
    host.thinking_enabled
}

pub(super) fn host_approval_policy(host: &RuntimeRequest) -> Option<String> {
    non_empty(host.approval_policy.as_deref())
}

pub(super) fn host_sandbox_policy(host: &RuntimeRequest) -> Option<String> {
    non_empty(host.sandbox_policy.as_deref())
}

pub(super) fn host_system_prompt(host: &RuntimeRequest) -> Option<String> {
    non_empty(host.system_prompt.as_deref())
}

fn host_web_search(host: &RuntimeRequest) -> Option<bool> {
    host.web_search
}

fn host_request_search_mode(host: &RuntimeRequest) -> Option<RequestToolPolicyMode> {
    host.search_mode.map(|mode| match mode {
        RuntimeSearchMode::Disabled => RequestToolPolicyMode::Disabled,
        RuntimeSearchMode::Auto => RequestToolPolicyMode::Auto,
        RuntimeSearchMode::Required => RequestToolPolicyMode::Required,
    })
}

pub(super) fn request_tool_policy_from_request(
    host_request: Option<&RuntimeRequest>,
) -> RequestToolPolicy {
    let web_search = host_request.and_then(host_web_search);
    let search_mode = host_request.and_then(host_request_search_mode);
    let policy = resolve_request_tool_policy_with_mode(web_search, search_mode);
    if host_request.is_some_and(host_requests_research_web_fetch) {
        request_tool_policy_with_additional_required_tools(policy, &["WebFetch"])
    } else {
        policy
    }
}

pub(super) fn should_use_compact_tool_surface(request: &ExecutionRequest) -> bool {
    matches!(
        turn_tool_surface_for_request(request),
        TurnToolSurface::CompactTools
    )
}

fn turn_tool_surface_for_request(request: &ExecutionRequest) -> TurnToolSurface {
    if request
        .runtime_metadata()
        .and_then(|metadata| metadata.pointer(HARNESS_TURN_TOOL_SURFACE_POINTER))
        .and_then(Value::as_str)
        == Some(tool_runtime::turn_tool_surface::TURN_TOOL_SURFACE_DIRECT_ANSWER)
    {
        return TurnToolSurface::DirectAnswer;
    }

    match app_server_turn_policy_value(request, LIME_RUNTIME_TOOL_SURFACE_KEY) {
        Some(LIME_RUNTIME_COMPACT_TOOLS_TOOL_SURFACE) => TurnToolSurface::CompactTools,
        _ => TurnToolSurface::Full,
    }
}

pub(super) fn host_metadata_value(host: &RuntimeRequest) -> Option<Value> {
    host.metadata.clone()
}

fn host_requests_research_web_fetch(host: &RuntimeRequest) -> bool {
    host_metadata_value(host).is_some_and(|metadata| {
        metadata
            .pointer("/harness/research_skill_launch/research_request")
            .is_some()
    })
}

fn should_use_responsive_chat_profile(
    request: &ExecutionRequest,
    first_sampling_turn: bool,
    request_tool_policy: &RequestToolPolicy,
) -> bool {
    first_sampling_turn
        && request.session.app_id == "agent-chat"
        && request.session.workspace_id.is_none()
        && request
            .session
            .business_object_ref
            .as_ref()
            .is_none_or(|reference| reference.kind == "agent.thread")
        && !request.input.has_images()
        && request.runtime_options.as_ref().is_none_or(|options| {
            options.capability_id.is_none()
                && options.expected_output.is_none()
                && options.structured_output.is_none()
                && options.output_schema.is_none()
        })
        && request.expected_output.is_none()
        && request.structured_output.is_none()
        && request.output_schema.is_none()
        && !request_tool_policy.requires_web_search()
        && structured_control_mentions(request).is_empty()
        && !request_has_workspace_context(request)
        && !request
            .runtime_metadata()
            .is_some_and(metadata_requests_tool_surface)
}

pub(super) fn structured_control_mentions(request: &ExecutionRequest) -> Vec<Value> {
    let mut mentions = Vec::new();
    for part in &request.input.parts {
        let agent_runtime::reply_input::RuntimeReplyInputPart::Mention { name, path } = part else {
            continue;
        };
        let name = name.trim();
        let path = path.trim();
        let Some(kind) = control_mention_kind(path) else {
            continue;
        };
        if name.is_empty()
            || mentions.iter().any(|mention: &Value| {
                mention.get("kind").and_then(Value::as_str) == Some(kind)
                    && mention.get("path").and_then(Value::as_str) == Some(path)
            })
        {
            continue;
        }
        mentions.push(serde_json::json!({
            "kind": kind,
            "name": name,
            "path": path,
        }));
    }
    mentions
}

fn control_mention_kind(path: &str) -> Option<&'static str> {
    [
        ("app://", "app"),
        ("plugin://", "plugin"),
        ("mcp://", "mcp"),
    ]
    .into_iter()
    .find_map(|(prefix, kind)| path.starts_with(prefix).then_some(kind))
}

fn request_has_workspace_context(request: &ExecutionRequest) -> bool {
    let runtime_request = request.runtime_request();
    runtime_request.is_some_and(|runtime_request| {
        runtime_request
            .workspace_id
            .as_deref()
            .is_some_and(non_blank)
            || runtime_request
                .working_dir
                .as_deref()
                .is_some_and(non_blank)
            || runtime_request
                .workspace_root
                .as_deref()
                .is_some_and(non_blank)
            || runtime_request
                .project_root
                .as_deref()
                .is_some_and(non_blank)
            || runtime_request.auto_continue == Some(true)
    }) || request.runtime_metadata().is_some_and(|metadata| {
        [
            "/harness/cwd",
            "/harness/working_dir",
            "/harness/workspace_root",
            "/harness/project_root",
            "/cwd",
            "/working_dir",
            "/workspace_root",
            "/project_root",
        ]
        .iter()
        .any(|pointer| {
            metadata
                .pointer(pointer)
                .and_then(Value::as_str)
                .is_some_and(non_blank)
        })
    }) || request
        .session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref())
        .is_some_and(|metadata| {
            [
                "/workingDir",
                "/working_dir",
                "/projectRoot",
                "/project_root",
            ]
            .iter()
            .any(|pointer| {
                metadata
                    .pointer(pointer)
                    .and_then(Value::as_str)
                    .is_some_and(non_blank)
            })
        })
}

fn non_blank(value: &str) -> bool {
    !value.trim().is_empty()
}

fn app_server_turn_policy_value<'a>(request: &'a ExecutionRequest, key: &str) -> Option<&'a str> {
    let runtime = app_server_turn_policy_runtime(request)?;
    runtime.get(key).and_then(Value::as_str)
}

fn app_server_turn_policy_runtime(
    request: &ExecutionRequest,
) -> Option<&serde_json::Map<String, Value>> {
    let runtime = request
        .runtime_metadata()?
        .get(LIME_RUNTIME_METADATA_KEY)?
        .as_object()?;
    if runtime.get(LIME_RUNTIME_SOURCE_KEY).and_then(Value::as_str)
        != Some(APP_SERVER_TURN_POLICY_SOURCE)
    {
        return None;
    }
    Some(runtime)
}

fn metadata_requests_tool_surface(metadata: &Value) -> bool {
    [
        "/harness/workspace_skill_runtime_enable",
        "/harness/workspaceSkillRuntimeEnable",
        "/workspace_skill_runtime_enable",
        "/workspaceSkillRuntimeEnable",
        "/harness/service_scene_launch",
        "/harness/serviceSceneLaunch",
        "/service_scene_launch",
        "/serviceSceneLaunch",
        "/harness/expert/skill_refs",
        "/harness/expert/skillRefs",
        "/expert/skill_refs",
        "/expert/skillRefs",
        "/harness/agent_skills",
        "/harness/agentSkills",
        "/agent_skills",
        "/agentSkills",
    ]
    .iter()
    .any(|pointer| metadata.pointer(pointer).is_some())
}

fn json_pointer_string(value: &Value, pointers: &[&str]) -> Option<String> {
    pointers.iter().find_map(|pointer| {
        value
            .pointer(pointer)
            .and_then(Value::as_str)
            .and_then(|value| non_empty(Some(value)))
    })
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RuntimeSessionScope {
    pub(super) session_id: String,
    pub(super) thread_id: String,
    pub(super) turn_id: String,
    pub(super) workspace_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RuntimeWorkspaceScope {
    pub(super) working_dir: Option<PathBuf>,
    pub(super) project_root: Option<PathBuf>,
}
