//! 调度任务运行上下文解析。
//!
//! `lime-scheduler` 只保留历史任务上下文投影测试与数据模型守卫。
//! 真实自动化执行必须走 App Server `scheduledTask/run/start` current 主链。

use crate::types::ScheduledTask;
use agent_protocol::turn_context::{TurnContextOverride, TurnOutputSchemaSource};
use agent_protocol::{CollaborationMode, CollaborationModeSettings, ModeKind};
use lime_agent::{
    merge_system_prompt_with_request_tool_policy,
    merge_system_prompt_with_runtime_agents_for_project, resolve_request_tool_policy_with_mode,
    RequestToolPolicy, RequestToolPolicyMode,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;

const METADATA_VALUE_MAX_BYTES: usize = 8 * 1024;

const WORKING_DIR_POINTERS: &[&str] = &[
    "/working_dir",
    "/workingDir",
    "/working_directory",
    "/workingDirectory",
    "/cwd",
    "/turn_config/working_dir",
    "/turnConfig/workingDir",
    "/turn_config/working_directory",
    "/turnConfig/workingDirectory",
    "/turn_config/cwd",
    "/turnConfig/cwd",
    "/harness/working_dir",
    "/harness/workingDir",
    "/harness/working_directory",
    "/harness/workingDirectory",
    "/harness/cwd",
    "/source_metadata/working_dir",
    "/source_metadata/workingDir",
    "/source_metadata/working_directory",
    "/source_metadata/workingDirectory",
    "/source_metadata/cwd",
    "/source_metadata/harness/working_dir",
    "/source_metadata/harness/workingDir",
    "/source_metadata/harness/working_directory",
    "/source_metadata/harness/workingDirectory",
    "/source_metadata/harness/cwd",
    "/sourceMetadata/working_dir",
    "/sourceMetadata/workingDir",
    "/sourceMetadata/working_directory",
    "/sourceMetadata/workingDirectory",
    "/sourceMetadata/cwd",
    "/sourceMetadata/harness/working_dir",
    "/sourceMetadata/harness/workingDir",
    "/sourceMetadata/harness/working_directory",
    "/sourceMetadata/harness/workingDirectory",
    "/sourceMetadata/harness/cwd",
];

const PROJECT_ROOT_POINTERS: &[&str] = &[
    "/workspace_root",
    "/workspaceRoot",
    "/project_root",
    "/projectRoot",
    "/turn_config/workspace_root",
    "/turn_config/workspaceRoot",
    "/turn_config/project_root",
    "/turn_config/projectRoot",
    "/turnConfig/workspaceRoot",
    "/turnConfig/projectRoot",
    "/harness/workspace_root",
    "/harness/workspaceRoot",
    "/harness/project_root",
    "/harness/projectRoot",
    "/source_metadata/workspace_root",
    "/source_metadata/workspaceRoot",
    "/source_metadata/project_root",
    "/source_metadata/projectRoot",
    "/sourceMetadata/workspace_root",
    "/sourceMetadata/workspaceRoot",
    "/sourceMetadata/project_root",
    "/sourceMetadata/projectRoot",
    "/source_metadata/harness/workspace_root",
    "/source_metadata/harness/workspaceRoot",
    "/source_metadata/harness/project_root",
    "/source_metadata/harness/projectRoot",
    "/sourceMetadata/harness/workspace_root",
    "/sourceMetadata/harness/workspaceRoot",
    "/sourceMetadata/harness/project_root",
    "/sourceMetadata/harness/projectRoot",
];

const REASONING_POINTERS: &[&str] = &[
    "/reasoning_effort",
    "/reasoningEffort",
    "/model_reasoning_effort",
    "/modelReasoningEffort",
    "/reasoning/effort",
    "/turn_config/reasoning_effort",
    "/turnConfig/reasoningEffort",
    "/turn_config/model_reasoning_effort",
    "/turnConfig/modelReasoningEffort",
    "/turn_config/reasoning/effort",
    "/turnConfig/reasoning/effort",
    "/harness/reasoning_effort",
    "/harness/reasoningEffort",
    "/harness/model_reasoning_effort",
    "/harness/modelReasoningEffort",
    "/harness/reasoning/effort",
    "/source_metadata/reasoning_effort",
    "/source_metadata/reasoningEffort",
    "/source_metadata/model_reasoning_effort",
    "/source_metadata/modelReasoningEffort",
    "/source_metadata/reasoning/effort",
    "/source_metadata/harness/reasoning_effort",
    "/source_metadata/harness/reasoningEffort",
    "/source_metadata/harness/model_reasoning_effort",
    "/source_metadata/harness/modelReasoningEffort",
    "/source_metadata/harness/reasoning/effort",
    "/sourceMetadata/reasoning_effort",
    "/sourceMetadata/reasoningEffort",
    "/sourceMetadata/modelReasoningEffort",
    "/sourceMetadata/reasoning/effort",
    "/sourceMetadata/harness/reasoning_effort",
    "/sourceMetadata/harness/reasoningEffort",
    "/sourceMetadata/harness/modelReasoningEffort",
    "/sourceMetadata/harness/reasoning/effort",
];

const SYSTEM_PROMPT_POINTERS: &[&str] = &[
    "/system_prompt",
    "/systemPrompt",
    "/turn_config/system_prompt",
    "/turnConfig/systemPrompt",
];

const SEARCH_MODE_POINTERS: &[&str] = &[
    "/search_mode",
    "/searchMode",
    "/turn_config/search_mode",
    "/turnConfig/searchMode",
    "/harness/search_mode",
    "/harness/searchMode",
    "/source_metadata/search_mode",
    "/source_metadata/searchMode",
    "/source_metadata/harness/search_mode",
    "/source_metadata/harness/searchMode",
    "/sourceMetadata/search_mode",
    "/sourceMetadata/searchMode",
    "/sourceMetadata/harness/search_mode",
    "/sourceMetadata/harness/searchMode",
];

const WEB_SEARCH_POINTERS: &[&str] = &[
    "/web_search",
    "/webSearch",
    "/turn_config/web_search",
    "/turnConfig/webSearch",
    "/harness/web_search",
    "/harness/webSearch",
    "/source_metadata/web_search",
    "/source_metadata/webSearch",
    "/source_metadata/harness/web_search",
    "/source_metadata/harness/webSearch",
    "/sourceMetadata/web_search",
    "/sourceMetadata/webSearch",
    "/sourceMetadata/harness/web_search",
    "/sourceMetadata/harness/webSearch",
];

const APPROVAL_POLICY_POINTERS: &[&str] = &[
    "/approval_policy",
    "/approvalPolicy",
    "/turn_config/approval_policy",
    "/turnConfig/approvalPolicy",
    "/harness/approval_policy",
    "/harness/approvalPolicy",
    "/source_metadata/harness/approval_policy",
    "/source_metadata/harness/approvalPolicy",
    "/sourceMetadata/harness/approval_policy",
    "/sourceMetadata/harness/approvalPolicy",
];

const SANDBOX_POLICY_POINTERS: &[&str] = &[
    "/sandbox_policy",
    "/sandboxPolicy",
    "/turn_config/sandbox_policy",
    "/turnConfig/sandboxPolicy",
    "/harness/sandbox_policy",
    "/harness/sandboxPolicy",
    "/source_metadata/harness/sandbox_policy",
    "/source_metadata/harness/sandboxPolicy",
    "/sourceMetadata/harness/sandbox_policy",
    "/sourceMetadata/harness/sandboxPolicy",
];

const THREAD_ID_POINTERS: &[&str] = &[
    "/thread_id",
    "/threadId",
    "/turn_config/thread_id",
    "/turnConfig/threadId",
    "/harness/thread_id",
    "/harness/threadId",
];

const TURN_ID_POINTERS: &[&str] = &[
    "/turn_id",
    "/turnId",
    "/turn_config/turn_id",
    "/turnConfig/turnId",
    "/harness/turn_id",
    "/harness/turnId",
];

const SCHEDULE_ID_POINTERS: &[&str] = &[
    "/schedule_id",
    "/scheduleId",
    "/turn_config/schedule_id",
    "/turnConfig/scheduleId",
    "/harness/schedule_id",
    "/harness/scheduleId",
    "/source_metadata/schedule_id",
    "/source_metadata/scheduleId",
    "/source_metadata/harness/schedule_id",
    "/source_metadata/harness/scheduleId",
    "/sourceMetadata/schedule_id",
    "/sourceMetadata/scheduleId",
    "/sourceMetadata/harness/schedule_id",
    "/sourceMetadata/harness/scheduleId",
];

const SYSTEM_PROMPT_OVERRIDE_POINTERS: &[&str] = &[
    "/system_prompt_override",
    "/systemPromptOverride",
    "/turn_config/system_prompt_override",
    "/turnConfig/systemPromptOverride",
];

const MAX_TURNS_POINTERS: &[&str] = &[
    "/max_turns",
    "/maxTurns",
    "/turn_config/max_turns",
    "/turnConfig/maxTurns",
];

const COLLABORATION_MODE_POINTERS: &[&str] = &[
    "/collaboration_mode",
    "/collaborationMode",
    "/turn_config/collaboration_mode",
    "/turnConfig/collaborationMode",
    "/harness/collaboration_mode",
    "/harness/collaborationMode",
];

const OUTPUT_SCHEMA_POINTERS: &[&str] = &[
    "/output_schema",
    "/outputSchema",
    "/turn_config/output_schema",
    "/turnConfig/outputSchema",
];

#[derive(Debug, Clone)]
pub(crate) struct AgentTaskRuntimeContext {
    pub(crate) session_id: String,
    pub(crate) schedule_id: String,
    pub(crate) thread_id: Option<String>,
    pub(crate) turn_id: Option<String>,
    pub(crate) working_dir: Option<PathBuf>,
    pub(crate) project_root: Option<PathBuf>,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) request_tool_policy: RequestToolPolicy,
    pub(crate) system_prompt_override: Option<bool>,
    pub(crate) max_turns: Option<u32>,
    system_prompt: Option<String>,
    turn_context: TurnContextOverride,
}

impl AgentTaskRuntimeContext {
    pub(crate) fn from_task(task: &ScheduledTask, user_visible_input_text: Option<&str>) -> Self {
        let session_id = resolve_session_id(task);
        let schedule_id = resolve_schedule_id(task);
        let thread_id = string_at(&task.params, THREAD_ID_POINTERS);
        let turn_id = string_at(&task.params, TURN_ID_POINTERS);
        let working_dir = resolve_working_dir(&task.params);
        let project_root = resolve_project_root(&task.params);
        let reasoning_effort = string_at(&task.params, REASONING_POINTERS);
        let request_tool_policy = resolve_request_tool_policy_with_mode(
            bool_at(&task.params, WEB_SEARCH_POINTERS),
            search_mode_at(&task.params),
        );
        let turn_context = build_turn_context(
            task,
            working_dir.clone().or_else(|| project_root.clone()),
            reasoning_effort.clone(),
            &request_tool_policy,
            user_visible_input_text,
        );

        Self {
            session_id,
            schedule_id,
            thread_id,
            turn_id,
            working_dir: working_dir.or_else(|| project_root.clone()),
            project_root,
            reasoning_effort,
            request_tool_policy,
            system_prompt_override: bool_at(&task.params, SYSTEM_PROMPT_OVERRIDE_POINTERS),
            max_turns: u32_at(&task.params, MAX_TURNS_POINTERS),
            system_prompt: string_at(&task.params, SYSTEM_PROMPT_POINTERS),
            turn_context,
        }
    }

    pub(crate) fn merged_system_prompt(&self) -> Option<String> {
        let system_prompt = merge_system_prompt_with_runtime_agents_for_project(
            self.system_prompt.clone(),
            self.working_dir.as_deref(),
            self.project_root.as_deref(),
        );
        merge_system_prompt_with_request_tool_policy(system_prompt, &self.request_tool_policy)
    }

    pub(crate) fn turn_context(&self) -> TurnContextOverride {
        self.turn_context.clone()
    }
}

fn resolve_session_id(task: &ScheduledTask) -> String {
    string_at(&task.params, &["/session_id", "/sessionId"])
        .unwrap_or_else(|| format!("scheduler-agent-chat-{}", task.id))
}

fn resolve_schedule_id(task: &ScheduledTask) -> String {
    string_at(&task.params, SCHEDULE_ID_POINTERS).unwrap_or_else(|| task.id.clone())
}

fn resolve_working_dir(params: &Value) -> Option<PathBuf> {
    string_at(params, WORKING_DIR_POINTERS)
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
}

fn resolve_project_root(params: &Value) -> Option<PathBuf> {
    string_at(params, PROJECT_ROOT_POINTERS)
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
}

fn search_mode_at(params: &Value) -> Option<RequestToolPolicyMode> {
    let raw = string_at(params, SEARCH_MODE_POINTERS)?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "required" => Some(RequestToolPolicyMode::Required),
        "auto" => Some(RequestToolPolicyMode::Auto),
        "disabled" => Some(RequestToolPolicyMode::Disabled),
        _ => None,
    }
}

fn bool_at(value: &Value, pointers: &[&str]) -> Option<bool> {
    pointers.iter().find_map(|pointer| {
        let value = value.pointer(pointer)?;
        match value {
            Value::Bool(flag) => Some(*flag),
            Value::String(raw) => match raw.trim().to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" | "on" | "enabled" => Some(true),
                "false" | "0" | "no" | "off" | "disabled" => Some(false),
                _ => None,
            },
            Value::Number(number) => number.as_i64().map(|v| v != 0),
            _ => None,
        }
    })
}

fn u32_at(value: &Value, pointers: &[&str]) -> Option<u32> {
    pointers.iter().find_map(|pointer| {
        let value = value.pointer(pointer)?;
        match value {
            Value::Number(number) => number.as_u64().and_then(|value| u32::try_from(value).ok()),
            Value::String(raw) => raw.trim().parse::<u32>().ok(),
            _ => None,
        }
    })
}

fn value_at(value: &Value, pointers: &[&str]) -> Option<Value> {
    pointers.iter().find_map(|pointer| {
        value.pointer(pointer).and_then(|value| {
            if value.is_null() {
                None
            } else {
                Some(value.clone())
            }
        })
    })
}

fn string_at(value: &Value, pointers: &[&str]) -> Option<String> {
    pointers.iter().find_map(|pointer| {
        value
            .pointer(pointer)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn build_turn_context(
    task: &ScheduledTask,
    working_dir: Option<PathBuf>,
    reasoning_effort: Option<String>,
    request_tool_policy: &RequestToolPolicy,
    user_visible_input_text: Option<&str>,
) -> TurnContextOverride {
    let mut metadata = HashMap::new();
    metadata.insert(
        "scheduler_task".to_string(),
        json!({
            "id": &task.id,
            "name": &task.name,
            "taskType": &task.task_type,
            "providerType": &task.provider_type,
            "model": &task.model,
            "compatExecutor": true,
        }),
    );
    metadata.insert(
        "request_tool_policy".to_string(),
        json!({
            "searchMode": request_tool_policy.search_mode.as_str(),
            "effectiveWebSearch": request_tool_policy.effective_web_search,
            "requiredTools": request_tool_policy.required_tools,
            "allowedTools": request_tool_policy.allowed_tools,
            "disallowedTools": request_tool_policy.disallowed_tools,
        }),
    );
    if let Some(source_metadata) = source_metadata(&task.params).and_then(bounded_metadata_value) {
        metadata.insert("source_metadata".to_string(), source_metadata);
    }
    let output_schema = value_at(&task.params, OUTPUT_SCHEMA_POINTERS).and_then(|value| {
        bounded_metadata_value(&value).and_then(|value| value.is_object().then_some(value))
    });
    let output_schema_source = output_schema.as_ref().map(|_| TurnOutputSchemaSource::Turn);

    let collaboration_mode =
        string_at(&task.params, COLLABORATION_MODE_POINTERS).and_then(|mode| {
            let mode = match mode.as_str() {
                "plan" => ModeKind::Plan,
                "default" => ModeKind::Default,
                _ => return None,
            };
            Some(CollaborationMode {
                mode,
                settings: CollaborationModeSettings {
                    model: task.model.clone(),
                    reasoning_effort: reasoning_effort.clone(),
                    developer_instructions: None,
                },
            })
        });

    TurnContextOverride {
        cwd: working_dir,
        model: Some(task.model.clone()),
        effort: reasoning_effort,
        approval_policy: string_at(&task.params, APPROVAL_POLICY_POINTERS),
        sandbox_policy: string_at(&task.params, SANDBOX_POLICY_POINTERS),
        collaboration_mode,
        user_visible_input_text: user_visible_input_text
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned),
        output_schema,
        output_schema_source,
        metadata,
        ..TurnContextOverride::default()
    }
}

fn source_metadata(params: &Value) -> Option<&Value> {
    params
        .get("source_metadata")
        .or_else(|| params.get("sourceMetadata"))
}

fn bounded_metadata_value(value: &Value) -> Option<Value> {
    let encoded = serde_json::to_vec(value).ok()?;
    if encoded.len() <= METADATA_VALUE_MAX_BYTES {
        Some(value.clone())
    } else {
        Some(json!({
            "truncated": true,
            "originalBytes": encoded.len(),
            "maxBytes": METADATA_VALUE_MAX_BYTES,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use lime_agent::REQUEST_TOOL_POLICY_MARKER;
    use lime_agent::RUNTIME_AGENTS_PROMPT_MARKER;
    use std::fs;
    use tempfile::TempDir;

    fn task_with_params(params: Value) -> ScheduledTask {
        ScheduledTask::new(
            "ctx-test".to_string(),
            "agent_chat".to_string(),
            params,
            "openai".to_string(),
            "gpt-4.1".to_string(),
            Utc::now(),
        )
    }

    #[test]
    fn resolves_absolute_working_dir_and_rejects_relative_path() {
        let workspace = TempDir::new().expect("workspace");
        let absolute = task_with_params(json!({
            "workingDir": workspace.path().to_string_lossy()
        }));
        let relative = task_with_params(json!({
            "working_dir": "relative/workspace"
        }));

        let absolute_context = AgentTaskRuntimeContext::from_task(&absolute, Some("hello"));
        let relative_context = AgentTaskRuntimeContext::from_task(&relative, Some("hello"));

        assert_eq!(
            absolute_context.working_dir.as_deref(),
            Some(workspace.path())
        );
        assert!(relative_context.working_dir.is_none());
    }

    #[test]
    fn merges_workspace_runtime_agents_and_request_tool_policy() {
        let workspace = TempDir::new().expect("workspace");
        let agents_path = workspace.path().join(".lime").join("AGENTS.md");
        fs::create_dir_all(agents_path.parent().expect("agents parent"))
            .expect("create agents parent");
        fs::write(&agents_path, "- 工作区调度指令").expect("write agents");
        let task = task_with_params(json!({
            "systemPrompt": "请求级提示",
            "working_dir": workspace.path().to_string_lossy(),
            "searchMode": "required"
        }));

        let context = AgentTaskRuntimeContext::from_task(&task, Some("需要检索"));
        let prompt = context.merged_system_prompt().expect("merged prompt");

        assert!(prompt.contains("请求级提示"));
        assert!(prompt.contains(RUNTIME_AGENTS_PROMPT_MARKER));
        assert!(prompt.contains("工作区调度指令"));
        assert!(prompt.contains(REQUEST_TOOL_POLICY_MARKER));
        assert_eq!(
            context.request_tool_policy.search_mode,
            RequestToolPolicyMode::Required
        );
    }

    #[test]
    fn search_mode_parser_accepts_current_values_only() {
        assert_eq!(
            search_mode_at(&json!({"searchMode": "auto"})),
            Some(RequestToolPolicyMode::Auto)
        );
        assert_eq!(
            search_mode_at(&json!({"searchMode": "required"})),
            Some(RequestToolPolicyMode::Required)
        );
        assert_eq!(
            search_mode_at(&json!({"searchMode": "disabled"})),
            Some(RequestToolPolicyMode::Disabled)
        );

        for legacy_value in [
            "allowed", "allow", "enabled", "true", "1", "require", "must",
        ] {
            assert_eq!(search_mode_at(&json!({"searchMode": legacy_value})), None);
        }
    }

    #[test]
    fn merges_runtime_agents_with_explicit_project_root_boundary() {
        let workspace = TempDir::new().expect("workspace");
        let parent = workspace.path().join("parent");
        let repo = parent.join("repo");
        let nested = repo.join("apps").join("writer");
        fs::create_dir_all(parent.join(".lime")).expect("create parent agents dir");
        fs::create_dir_all(repo.join(".lime")).expect("create root agents dir");
        fs::create_dir_all(nested.join(".lime")).expect("create nested agents dir");
        fs::write(parent.join(".lime").join("AGENTS.md"), "- 父级规则不应出现")
            .expect("write parent agents");
        fs::write(repo.join(".lime").join("AGENTS.md"), "- 调度根规则").expect("write root agents");
        fs::write(
            nested.join(".lime").join("AGENTS.override.md"),
            "- 调度覆盖规则",
        )
        .expect("write nested override agents");
        let task = task_with_params(json!({
            "systemPrompt": "调度请求提示",
            "projectRoot": repo.to_string_lossy(),
            "workingDir": nested.to_string_lossy(),
            "searchMode": "auto"
        }));

        let context = AgentTaskRuntimeContext::from_task(&task, Some("需要检索"));
        let prompt = context.merged_system_prompt().expect("merged prompt");

        assert_eq!(context.working_dir.as_deref(), Some(nested.as_path()));
        assert_eq!(context.project_root.as_deref(), Some(repo.as_path()));
        assert!(prompt.contains("调度请求提示"));
        assert!(prompt.contains("调度根规则"));
        assert!(prompt.contains("调度覆盖规则"));
        assert!(!prompt.contains("父级规则不应出现"));
        assert!(prompt.contains("# AGENTS.md instructions"));
        assert!(!prompt.contains(REQUEST_TOOL_POLICY_MARKER));
        assert_eq!(
            context.request_tool_policy.search_mode,
            RequestToolPolicyMode::Auto
        );
    }

    #[test]
    fn builds_full_session_and_turn_runtime_context() {
        let workspace = TempDir::new().expect("workspace");
        let task = task_with_params(json!({
            "scheduleId": "schedule-custom",
            "sessionId": "session-custom",
            "threadId": "thread-custom",
            "turnId": "turn-custom",
            "workingDir": workspace.path().to_string_lossy(),
            "reasoningEffort": "high",
            "approvalPolicy": "on-request",
            "sandboxPolicy": "workspace-write",
            "collaborationMode": "default",
            "systemPromptOverride": true,
            "maxTurns": 4,
            "outputSchema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"}
                }
            },
            "searchMode": "auto"
        }));

        let context = AgentTaskRuntimeContext::from_task(&task, Some("生成总结"));
        let turn_context = context.turn_context();

        assert_eq!(context.schedule_id, "schedule-custom");
        assert_eq!(context.session_id, "session-custom");
        assert_eq!(context.thread_id.as_deref(), Some("thread-custom"));
        assert_eq!(context.turn_id.as_deref(), Some("turn-custom"));
        assert_eq!(context.system_prompt_override, Some(true));
        assert_eq!(context.max_turns, Some(4));
        assert_eq!(turn_context.cwd.as_deref(), Some(workspace.path()));
        assert_eq!(turn_context.effort.as_deref(), Some("high"));
        assert_eq!(turn_context.approval_policy.as_deref(), Some("on-request"));
        assert_eq!(
            turn_context.sandbox_policy.as_deref(),
            Some("workspace-write")
        );
        assert_eq!(
            turn_context
                .collaboration_mode
                .as_ref()
                .map(|mode| mode.mode),
            Some(ModeKind::Default)
        );
        assert_eq!(
            turn_context
                .output_schema
                .as_ref()
                .and_then(|value| value.get("type"))
                .and_then(Value::as_str),
            Some("object")
        );
        assert_eq!(
            turn_context
                .metadata
                .get("request_tool_policy")
                .and_then(|value| value.get("searchMode"))
                .and_then(Value::as_str),
            Some("auto")
        );
    }

    #[test]
    fn forwards_reasoning_effort_to_provider_and_turn_context() {
        let task = task_with_params(json!({
            "turnConfig": {
                "modelReasoningEffort": "high"
            }
        }));

        let context = AgentTaskRuntimeContext::from_task(&task, Some("分析"));
        let turn_context = context.turn_context();

        assert_eq!(context.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(turn_context.effort.as_deref(), Some("high"));
        assert_eq!(turn_context.model.as_deref(), Some("gpt-4.1"));
        assert_eq!(
            turn_context.user_visible_input_text.as_deref(),
            Some("分析")
        );
    }

    #[test]
    fn reads_runtime_context_from_source_metadata() {
        let workspace = TempDir::new().expect("workspace");
        let project_root = TempDir::new().expect("project root");
        let task = task_with_params(json!({
            "sourceMetadata": {
                "projectRoot": project_root.path().to_string_lossy(),
                "harness": {
                    "workingDir": workspace.path().to_string_lossy(),
                    "reasoning": {
                        "effort": "medium"
                    },
                    "webSearch": true
                }
            }
        }));

        let context = AgentTaskRuntimeContext::from_task(&task, Some("hello"));

        assert_eq!(context.working_dir.as_deref(), Some(workspace.path()));
        assert_eq!(context.project_root.as_deref(), Some(project_root.path()));
        assert_eq!(context.reasoning_effort.as_deref(), Some("medium"));
        assert!(context.request_tool_policy.effective_web_search);
    }

    #[test]
    fn caps_large_source_metadata_in_turn_context() {
        let task = task_with_params(json!({
            "source_metadata": {
                "payload": "x".repeat(METADATA_VALUE_MAX_BYTES + 128)
            }
        }));

        let context = AgentTaskRuntimeContext::from_task(&task, Some("hello"));
        let source_metadata = context
            .turn_context()
            .metadata
            .get("source_metadata")
            .cloned()
            .expect("source metadata");

        assert_eq!(source_metadata["truncated"], true);
    }
}
