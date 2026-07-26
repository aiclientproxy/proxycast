use crate::runtime::memory_prompt::{
    append_memory_context_to_system_prompt, append_soul_context_to_system_prompt,
};
use crate::ExecutionRequest;
use agent_runtime::reply_loop::DEFAULT_MAX_REPLY_TURNS;
use lime_agent::{
    build_agent_session_config, merge_system_prompt_with_request_tool_policy,
    merge_system_prompt_with_runtime_agents_for_project, AgentSessionConfig,
    AgentSessionConfigurationRequest, RequestToolPolicy,
};
use serde_json::Value;

use super::{
    host_system_prompt, request_workspace_scope, runtime_request_from_request,
    turn_context_from_request, RuntimeModelSelection, RuntimeRequest, RuntimeSessionScope,
};

const DEFAULT_SESSION_SYSTEM_PROMPT: &str = "你是 Lime 桌面端中的 AI 助手。交互口吻、问候、自我介绍、工具进展和失败恢复由当前会话的 `memory.soul` 上下文控制；如果没有 `memory.soul` 上下文，保持清晰、准确、可执行。";
const HARNESS_MAX_PROVIDER_STEPS_POINTER: &str = "/harness/provider_budget/max_provider_steps";
const HARNESS_PROVIDER_TOKEN_BUDGET_POINTER: &str = "/harness/provider_budget/token_budget";

pub(in crate::runtime_backend) fn session_config_from_request(
    request: &ExecutionRequest,
    host_request: Option<&RuntimeRequest>,
    scope: &RuntimeSessionScope,
    selection: &RuntimeModelSelection,
    request_tool_policy: &RequestToolPolicy,
    config_metadata: Option<Value>,
) -> AgentSessionConfig {
    let workspace_scope = request_workspace_scope(request, host_request);
    let metadata_values = super::super::skill_runtime_enable::request_metadata_values(request);
    let turn_tool_surface = super::turn_tool_surface_for_request(request);
    let runtime_metadata = request.runtime_metadata();
    let (system_prompt, skill_snapshot) = if turn_tool_surface.uses_light_session_prompt() {
        (
            append_soul_context_to_system_prompt(
                Some(request_system_prompt(request)),
                config_metadata.as_ref(),
                runtime_metadata,
            ),
            None,
        )
    } else {
        let system_prompt = merge_system_prompt_with_runtime_agents_for_project(
            Some(request_system_prompt(request)),
            workspace_scope.working_dir.as_deref(),
            workspace_scope.project_root.as_deref(),
        );
        let skills_context = super::super::agent_skills_context::agent_skills_context_for_turn(
            system_prompt,
            &request.input.concat_text(),
            &metadata_values,
            workspace_scope.working_dir.as_deref(),
            workspace_scope.project_root.as_deref(),
        );
        let system_prompt =
            super::super::plugin_activation_context::append_plugin_activation_context_to_system_prompt(
                skills_context.system_prompt,
                &metadata_values,
            );
        let system_prompt =
            super::super::plugin_runtime_context::append_plugin_runtime_context_to_system_prompt(
                system_prompt,
                &metadata_values,
            );
        let system_prompt = append_memory_context_to_system_prompt(system_prompt, runtime_metadata);
        let system_prompt = append_soul_context_to_system_prompt(
            system_prompt,
            config_metadata.as_ref(),
            runtime_metadata,
        );
        (
            merge_system_prompt_with_request_tool_policy(system_prompt, request_tool_policy),
            skills_context.snapshot,
        )
    };
    let mut turn_context =
        turn_context_from_request(request, host_request, scope, selection, config_metadata);
    let structured_mentions = super::structured_control_mentions(request);
    if !structured_mentions.is_empty() {
        turn_context
            .get_or_insert_with(Default::default)
            .metadata
            .insert(
                super::INPUT_MENTIONS_TURN_METADATA_KEY.to_string(),
                Value::Array(structured_mentions),
            );
    }
    if let Some(skill_snapshot) = skill_snapshot {
        let turn_context = turn_context.get_or_insert_with(Default::default);
        turn_context.metadata.insert(
            lime_skills::SKILL_SNAPSHOT_TURN_METADATA_KEY.to_string(),
            serde_json::to_value(skill_snapshot).expect("skill snapshot must serialize"),
        );
    }
    if turn_tool_surface.is_harness_direct_answer() {
        let turn_context = turn_context.get_or_insert_with(Default::default);
        let runtime = turn_context
            .metadata
            .entry("lime_runtime".to_string())
            .or_insert_with(|| Value::Object(Default::default()));
        if !runtime.is_object() {
            *runtime = Value::Object(Default::default());
        }
        let runtime = runtime.as_object_mut().expect("lime_runtime object");
        runtime.insert("auto_compact".to_string(), Value::Bool(false));
        runtime.insert(
            "tool_surface".to_string(),
            Value::String(
                tool_runtime::turn_tool_surface::TURN_TOOL_SURFACE_DIRECT_ANSWER.to_string(),
            ),
        );
        runtime.insert(
            "source".to_string(),
            Value::String(super::HARNESS_TURN_POLICY_SOURCE.to_string()),
        );
    }
    build_agent_session_config(AgentSessionConfigurationRequest {
        session_id: scope.session_id.clone(),
        thread_id: scope.thread_id.clone(),
        turn_id: scope.turn_id.clone(),
        forked_from_thread_id: request.forked_from_thread_id.clone(),
        max_turns: harness_max_provider_steps(runtime_metadata),
        provider_token_budget: harness_provider_token_budget(runtime_metadata),
        system_prompt,
        turn_context,
        include_context_trace: !turn_tool_surface.uses_light_session_prompt(),
    })
}

fn harness_max_provider_steps(runtime_metadata: Option<&Value>) -> Option<u32> {
    runtime_metadata
        .and_then(|metadata| metadata.pointer(HARNESS_MAX_PROVIDER_STEPS_POINTER))
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| (1..=DEFAULT_MAX_REPLY_TURNS).contains(value))
}

fn harness_provider_token_budget(runtime_metadata: Option<&Value>) -> Option<u64> {
    runtime_metadata
        .and_then(|metadata| metadata.pointer(HARNESS_PROVIDER_TOKEN_BUDGET_POINTER))
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
}

fn request_system_prompt(request: &ExecutionRequest) -> String {
    runtime_request_from_request(request)
        .and_then(|host| host_system_prompt(&host))
        .unwrap_or_else(|| DEFAULT_SESSION_SYSTEM_PROMPT.to_string())
}
