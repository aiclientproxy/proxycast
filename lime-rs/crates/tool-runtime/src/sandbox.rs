use crate::execution_policy::ToolExecutionSandboxProfile;
use serde::Serialize;
use serde_json::Map as JsonMap;
use serde_json::Value as JsonValue;

mod command;

pub use command::{prepare_sandbox_command, SandboxCommandError, SandboxCommandRequest};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestedSandboxPolicy {
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
}

impl RequestedSandboxPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::DangerFullAccess => "danger-full-access",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxBackendPlatform {
    Macos,
    Linux,
    Windows,
    Unsupported,
}

impl SandboxBackendPlatform {
    pub fn current() -> Self {
        match std::env::consts::OS {
            "macos" => Self::Macos,
            "linux" => Self::Linux,
            "windows" => Self::Windows,
            _ => Self::Unsupported,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Macos => "macos",
            Self::Linux => "linux",
            Self::Windows => "windows",
            Self::Unsupported => "unsupported",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxBackend {
    None,
    Seatbelt,
    LinuxSandbox,
    RestrictedToken,
}

impl SandboxBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Seatbelt => "seatbelt",
            Self::LinuxSandbox => "linux_sandbox",
            Self::RestrictedToken => "restricted_token",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxBackendStatus {
    NotRequired,
    Disabled,
    Ready,
    Unavailable,
    Bypassed,
}

impl SandboxBackendStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::NotRequired => "not_required",
            Self::Disabled => "disabled",
            Self::Ready => "ready",
            Self::Unavailable => "unavailable",
            Self::Bypassed => "bypassed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxBackendConfigSource {
    Default,
    Persisted,
    Organization,
    User,
    Runtime,
    Request,
}

impl SandboxBackendConfigSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Persisted => "persisted",
            Self::Organization => "organization",
            Self::User => "user",
            Self::Runtime => "runtime",
            Self::Request => "request",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceSandboxRuntimeConfig {
    pub enabled: bool,
    pub strict: bool,
    pub notify_on_fallback: bool,
    pub windows_mode: Option<WindowsSandboxExecutionMode>,
    pub source: SandboxBackendConfigSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowsSandboxExecutionMode {
    Elevated,
    Unelevated,
}

impl Default for WorkspaceSandboxRuntimeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strict: false,
            notify_on_fallback: true,
            windows_mode: None,
            source: SandboxBackendConfigSource::Default,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SandboxBackendPlanInput<'a> {
    pub sandbox_profile: ToolExecutionSandboxProfile,
    pub requested_policy: Option<&'a str>,
    pub request_metadata: Option<&'a JsonValue>,
    pub bypass_restrictions: bool,
    pub platform: SandboxBackendPlatform,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SandboxBackendPlan {
    pub platform: SandboxBackendPlatform,
    pub backend: SandboxBackend,
    pub status: SandboxBackendStatus,
    pub enforced: bool,
    pub required: bool,
    pub config: WorkspaceSandboxRuntimeConfig,
    pub reason_code: &'static str,
    pub reason: &'static str,
}

impl SandboxBackendPlan {
    pub fn strict_fallback_blocks_execution(self) -> bool {
        self.required && self.config.enabled && self.config.strict && !self.enforced
    }

    pub fn can_run_with_backend(self) -> bool {
        self.required && self.config.enabled && self.enforced
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SandboxEvaluationInput<'a> {
    pub sandbox_profile: ToolExecutionSandboxProfile,
    pub requested_policy: Option<&'a str>,
    pub params: &'a JsonValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxEvaluation {
    Allow,
    Block(SandboxBlock),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxBlock {
    pub policy: RequestedSandboxPolicy,
    pub reason_code: &'static str,
    pub reason: &'static str,
    pub diagnostic: &'static str,
}

pub fn evaluate_sandbox(input: SandboxEvaluationInput<'_>) -> SandboxEvaluation {
    if input.sandbox_profile != ToolExecutionSandboxProfile::WorkspaceCommand {
        return SandboxEvaluation::Allow;
    }
    if parse_requested_sandbox_policy(input.requested_policy)
        != Some(RequestedSandboxPolicy::ReadOnly)
    {
        return SandboxEvaluation::Allow;
    }
    if command_text(input.params)
        .as_deref()
        .is_some_and(shell_command_is_read_only)
    {
        return SandboxEvaluation::Allow;
    }

    SandboxEvaluation::Block(SandboxBlock {
        policy: RequestedSandboxPolicy::ReadOnly,
        reason_code: "read_only_sandbox_blocks_shell_command",
        reason: "read-only 沙箱禁止执行会修改工作区或状态的 shell 命令",
        diagnostic: "read-only sandbox blocks non-read-only shell command",
    })
}

pub fn plan_sandbox_backend(input: SandboxBackendPlanInput<'_>) -> SandboxBackendPlan {
    let config = resolve_workspace_sandbox_runtime_config(input.request_metadata);
    let required = sandbox_backend_required(input.sandbox_profile, input.requested_policy);

    if input.bypass_restrictions {
        return SandboxBackendPlan {
            platform: input.platform,
            backend: SandboxBackend::None,
            status: SandboxBackendStatus::Bypassed,
            enforced: false,
            required,
            config,
            reason_code: "sandbox_backend_bypassed_by_full_access",
            reason: "full-access 已绕过 workspace sandbox backend",
        };
    }
    if !required {
        return SandboxBackendPlan {
            platform: input.platform,
            backend: SandboxBackend::None,
            status: SandboxBackendStatus::NotRequired,
            enforced: false,
            required,
            config,
            reason_code: "sandbox_backend_not_required",
            reason: "当前工具策略不需要 workspace sandbox backend",
        };
    }
    if !config.enabled {
        return SandboxBackendPlan {
            platform: input.platform,
            backend: SandboxBackend::None,
            status: SandboxBackendStatus::Disabled,
            enforced: false,
            required,
            config,
            reason_code: "workspace_sandbox_disabled",
            reason: "workspace sandbox backend 未启用",
        };
    }

    let (backend, status, enforced, reason_code, reason) = match input.platform {
        SandboxBackendPlatform::Macos if macos_seatbelt_available() => (
            SandboxBackend::Seatbelt,
            SandboxBackendStatus::Ready,
            true,
            "sandbox_backend_ready",
            "macOS seatbelt backend 可用于当前 shell 工具执行",
        ),
        SandboxBackendPlatform::Macos => (
            SandboxBackend::Seatbelt,
            SandboxBackendStatus::Unavailable,
            false,
            "sandbox_backend_unavailable",
            "macOS seatbelt backend 不可用",
        ),
        SandboxBackendPlatform::Linux if linux_bubblewrap_available() => (
            SandboxBackend::LinuxSandbox,
            SandboxBackendStatus::Ready,
            true,
            "sandbox_backend_ready",
            "Linux bubblewrap backend 可用于当前 shell 工具执行",
        ),
        SandboxBackendPlatform::Linux => (
            SandboxBackend::LinuxSandbox,
            SandboxBackendStatus::Unavailable,
            false,
            "sandbox_backend_unavailable",
            "Linux bubblewrap backend 不可用",
        ),
        SandboxBackendPlatform::Windows => (
            SandboxBackend::RestrictedToken,
            SandboxBackendStatus::Ready,
            true,
            "sandbox_backend_ready",
            "Windows restricted token backend 可用于当前 shell 工具执行",
        ),
        SandboxBackendPlatform::Unsupported => (
            SandboxBackend::None,
            SandboxBackendStatus::Unavailable,
            false,
            "sandbox_backend_unsupported_platform",
            "当前平台没有可用 workspace sandbox backend",
        ),
    };

    SandboxBackendPlan {
        platform: input.platform,
        backend,
        status,
        enforced,
        required,
        config,
        reason_code,
        reason,
    }
}

fn macos_seatbelt_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        std::path::Path::new("/usr/bin/sandbox-exec").is_file()
    }

    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

fn linux_bubblewrap_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("bwrap")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

pub fn parse_requested_sandbox_policy(value: Option<&str>) -> Option<RequestedSandboxPolicy> {
    let normalized = value?.trim().replace(['_', ' '], "-").to_ascii_lowercase();
    match normalized.as_str() {
        "read-only" | "readonly" => Some(RequestedSandboxPolicy::ReadOnly),
        "workspace-write" | "workspacewrite" => Some(RequestedSandboxPolicy::WorkspaceWrite),
        "danger-full-access" | "dangerfullaccess" => Some(RequestedSandboxPolicy::DangerFullAccess),
        _ => None,
    }
}

pub fn requested_sandbox_policy_label(value: Option<&str>) -> &'static str {
    parse_requested_sandbox_policy(value)
        .map(RequestedSandboxPolicy::label)
        .unwrap_or("workspace-write")
}

pub fn command_text(params: &JsonValue) -> Option<String> {
    params.as_object().and_then(|object| {
        ["command", "cmd", "script"]
            .iter()
            .filter_map(|key| object.get(*key))
            .filter_map(JsonValue::as_str)
            .map(str::trim)
            .find(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn sandbox_backend_required(
    sandbox_profile: ToolExecutionSandboxProfile,
    requested_policy: Option<&str>,
) -> bool {
    sandbox_profile == ToolExecutionSandboxProfile::WorkspaceCommand
        && parse_requested_sandbox_policy(requested_policy)
            != Some(RequestedSandboxPolicy::DangerFullAccess)
}

fn resolve_workspace_sandbox_runtime_config(
    request_metadata: Option<&JsonValue>,
) -> WorkspaceSandboxRuntimeConfig {
    let mut config = WorkspaceSandboxRuntimeConfig::default();
    let Some(metadata) = request_metadata else {
        return config;
    };

    for (source, pointers) in workspace_sandbox_config_pointers() {
        for pointer in pointers {
            if let Some(object) = metadata.pointer(pointer).and_then(JsonValue::as_object) {
                apply_workspace_sandbox_config_layer(&mut config, object, source);
            }
        }
    }

    config
}

fn workspace_sandbox_config_pointers() -> [(SandboxBackendConfigSource, &'static [&'static str]); 6]
{
    [
        (
            SandboxBackendConfigSource::Persisted,
            &[
                "/config/agent/workspaceSandbox",
                "/config/agent/workspace_sandbox",
                "/agent/workspaceSandbox",
                "/agent/workspace_sandbox",
            ],
        ),
        (
            SandboxBackendConfigSource::Organization,
            &[
                "/organizationWorkspaceSandbox",
                "/organization_workspace_sandbox",
                "/harness/organizationWorkspaceSandbox",
                "/harness/organization_workspace_sandbox",
                "/policies/organization/workspaceSandbox",
                "/policies/organization/workspace_sandbox",
                "/policy/organization/workspaceSandbox",
                "/policy/organization/workspace_sandbox",
            ],
        ),
        (
            SandboxBackendConfigSource::User,
            &[
                "/userWorkspaceSandbox",
                "/user_workspace_sandbox",
                "/harness/userWorkspaceSandbox",
                "/harness/user_workspace_sandbox",
                "/policies/user/workspaceSandbox",
                "/policies/user/workspace_sandbox",
                "/policy/user/workspaceSandbox",
                "/policy/user/workspace_sandbox",
            ],
        ),
        (
            SandboxBackendConfigSource::Runtime,
            &[
                "/runtimeWorkspaceSandbox",
                "/runtime_workspace_sandbox",
                "/executionPolicy/workspaceSandbox",
                "/executionPolicy/workspace_sandbox",
                "/execution_policy/workspaceSandbox",
                "/execution_policy/workspace_sandbox",
                "/harness/workspaceSandbox",
                "/harness/workspace_sandbox",
                "/harness/executionPolicy/workspaceSandbox",
                "/harness/executionPolicy/workspace_sandbox",
                "/harness/execution_policy/workspaceSandbox",
                "/harness/execution_policy/workspace_sandbox",
                "/runtimeOptions/workspaceSandbox",
                "/runtimeOptions/workspace_sandbox",
                "/runtime_options/workspaceSandbox",
                "/runtime_options/workspace_sandbox",
                "/runtimeOptions/metadata/workspaceSandbox",
                "/runtimeOptions/metadata/workspace_sandbox",
                "/runtime_options/metadata/workspaceSandbox",
                "/runtime_options/metadata/workspace_sandbox",
                "/metadata/workspaceSandbox",
                "/metadata/workspace_sandbox",
            ],
        ),
        (
            SandboxBackendConfigSource::Request,
            &[
                "/requestWorkspaceSandbox",
                "/request_workspace_sandbox",
                "/harness/requestWorkspaceSandbox",
                "/harness/request_workspace_sandbox",
                "/requestExecutionPolicy/workspaceSandbox",
                "/requestExecutionPolicy/workspace_sandbox",
                "/request_execution_policy/workspaceSandbox",
                "/request_execution_policy/workspace_sandbox",
                "/harness/requestExecutionPolicy/workspaceSandbox",
                "/harness/requestExecutionPolicy/workspace_sandbox",
                "/harness/request_execution_policy/workspaceSandbox",
                "/harness/request_execution_policy/workspace_sandbox",
            ],
        ),
        (
            SandboxBackendConfigSource::Request,
            &["/workspaceSandbox", "/workspace_sandbox"],
        ),
    ]
}

fn apply_workspace_sandbox_config_layer(
    config: &mut WorkspaceSandboxRuntimeConfig,
    object: &JsonMap<String, JsonValue>,
    source: SandboxBackendConfigSource,
) {
    let enabled = bool_field(object, &["enabled"]);
    let strict = bool_field(object, &["strict"]);
    let notify_on_fallback = bool_field(object, &["notifyOnFallback", "notify_on_fallback"]);
    let windows_mode = object
        .get("mode")
        .or_else(|| object.get("windowsMode"))
        .or_else(|| object.get("windows_mode"))
        .and_then(JsonValue::as_str)
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "elevated" => Some(WindowsSandboxExecutionMode::Elevated),
            "unelevated" => Some(WindowsSandboxExecutionMode::Unelevated),
            _ => None,
        });

    if enabled.is_none()
        && strict.is_none()
        && notify_on_fallback.is_none()
        && windows_mode.is_none()
    {
        return;
    }

    if let Some(enabled) = enabled {
        config.enabled = enabled;
    }
    if let Some(strict) = strict {
        config.strict = strict;
    }
    if let Some(notify_on_fallback) = notify_on_fallback {
        config.notify_on_fallback = notify_on_fallback;
    }
    if windows_mode.is_some() {
        config.windows_mode = windows_mode;
    }
    config.source = source;
}

fn bool_field(object: &JsonMap<String, JsonValue>, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(JsonValue::as_bool)
}

pub fn shell_command_is_read_only(command: &str) -> bool {
    split_shell_segments(command)
        .into_iter()
        .filter(|segment| !segment.trim().is_empty())
        .all(shell_segment_is_read_only)
}

fn split_shell_segments(command: &str) -> Vec<&str> {
    command
        .split(['\n', ';', '|'])
        .flat_map(|segment| segment.split("&&"))
        .flat_map(|segment| segment.split("||"))
        .collect()
}

fn shell_segment_is_read_only(segment: &str) -> bool {
    let normalized = segment.trim();
    if normalized.is_empty() {
        return true;
    }
    if normalized.contains('>')
        || normalized.contains('<')
        || normalized.contains('`')
        || normalized.contains("$(")
    {
        return false;
    }

    let mut parts = normalized.split_whitespace();
    let Some(command) = parts.next() else {
        return true;
    };
    let command = command.trim_matches(|value| value == '\'' || value == '"');
    if command.contains('=') {
        return false;
    }
    let args = parts.collect::<Vec<_>>();

    match command {
        "cd" | "pwd" | "ls" | "rg" | "grep" | "cat" | "head" | "tail" | "wc" | "uniq" | "cut"
        | "tr" | "test" | "stat" | "file" | "du" | "df" | "uname" | "whoami" | "echo"
        | "printf" | "true" | ":" => true,
        "find" => !args
            .iter()
            .any(|arg| matches!(*arg, "-delete" | "-exec" | "-execdir" | "-ok" | "-okdir")),
        "sort" => !args.iter().any(|arg| *arg == "-o" || arg.starts_with("-o")),
        "git" => args.first().is_some_and(|subcommand| {
            matches!(
                *subcommand,
                "status"
                    | "diff"
                    | "show"
                    | "log"
                    | "grep"
                    | "ls-files"
                    | "rev-parse"
                    | "describe"
                    | "blame"
            )
        }),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_policy::ToolExecutionSandboxProfile;
    use serde_json::json;

    #[test]
    fn parse_requested_sandbox_policy_accepts_common_aliases() {
        assert_eq!(
            parse_requested_sandbox_policy(Some("read_only")),
            Some(RequestedSandboxPolicy::ReadOnly)
        );
        assert_eq!(
            parse_requested_sandbox_policy(Some("workspace write")),
            Some(RequestedSandboxPolicy::WorkspaceWrite)
        );
        assert_eq!(
            parse_requested_sandbox_policy(Some("danger-full-access")),
            Some(RequestedSandboxPolicy::DangerFullAccess)
        );
    }

    #[test]
    fn evaluate_sandbox_blocks_non_read_only_workspace_commands() {
        let params = json!({ "command": "cargo test" });
        let result = evaluate_sandbox(SandboxEvaluationInput {
            sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
            requested_policy: Some("read-only"),
            params: &params,
        });

        assert!(matches!(result, SandboxEvaluation::Block(_)));
    }

    #[test]
    fn evaluate_sandbox_allows_read_only_shell_segments() {
        let params = json!({ "command": "pwd && git status | head -20" });
        let result = evaluate_sandbox(SandboxEvaluationInput {
            sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
            requested_policy: Some("read-only"),
            params: &params,
        });

        assert_eq!(result, SandboxEvaluation::Allow);
    }

    #[test]
    fn command_text_uses_first_non_empty_command_alias() {
        let params = json!({ "command": " ", "cmd": " cargo test ", "script": "ignored" });
        assert_eq!(command_text(&params), Some("cargo test".to_string()));
    }

    #[test]
    fn sandbox_plan_preserves_windows_execution_mode() {
        let metadata = json!({
            "agent": {
                "workspaceSandbox": {
                    "enabled": true,
                    "strict": true,
                    "mode": "unelevated"
                }
            }
        });
        let plan = plan_sandbox_backend(SandboxBackendPlanInput {
            sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
            requested_policy: Some("workspace-write"),
            request_metadata: Some(&metadata),
            bypass_restrictions: false,
            platform: SandboxBackendPlatform::Windows,
        });
        assert_eq!(
            plan.config.windows_mode,
            Some(WindowsSandboxExecutionMode::Unelevated)
        );
    }
}
