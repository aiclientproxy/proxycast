//! Hook discovery 与 command handler 执行 owner。
//!
//! 职责划分：
//! - 本模块负责从配置发现 Hook、构造不可变 `RuntimeHookSnapshot`，并执行 `Command` handler。
//! - 裁决语义归 [`crate::hook_lifecycle`]；本模块只把进程结果翻译成 `RuntimeHookHandlerReport`。
//!
//! 配置格式只有一种：按 Codex 对齐的 `RuntimeHookEventName` 分组。不接受旧的扁平
//! `{"hooks": [...]}` 或已退役的事件名，未知事件一律 fail closed，不静默丢弃。

use crate::hook_lifecycle::{RuntimeHookEventContext, RuntimeHookHandlerReport};
use crate::turn_snapshot::{
    RuntimeHookEventName, RuntimeHookExecutionMode, RuntimeHookHandlerType, RuntimeHookSnapshot,
    RuntimeHookSource, RuntimeHookTrustStatus,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::process::Command;

const DEFAULT_TIMEOUT_SEC: u64 = 10;

fn default_timeout_sec() -> u64 {
    DEFAULT_TIMEOUT_SEC
}

/// 磁盘上的 Hook 配置：`{"hooks": {"pre_tool_use": [ ... ]}}`。
#[derive(Debug, Deserialize)]
struct HookConfigFile {
    hooks: BTreeMap<String, Vec<HookConfigEntry>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HookConfigEntry {
    command: String,
    #[serde(default)]
    matcher: Option<String>,
    #[serde(default = "default_timeout_sec")]
    timeout_sec: u64,
    /// Hook 失败是否阻断原操作。
    #[serde(default)]
    blocking: bool,
    #[serde(default = "default_enabled")]
    enabled: bool,
}

fn default_enabled() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookDiscoveryError {
    Unreadable { path: PathBuf, message: String },
    Malformed { path: PathBuf, message: String },
    UnknownEvent { path: PathBuf, event: String },
    EmptyCommand { path: PathBuf, event: String },
    InvalidTimeout { path: PathBuf, event: String },
}

/// 已发现的 Hook：不可变快照 + 执行所需的命令与阻断策略。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredHook {
    pub snapshot: RuntimeHookSnapshot,
    pub command: String,
    pub blocking: bool,
}

fn parse_event_name(value: &str) -> Option<RuntimeHookEventName> {
    // 与 `RuntimeHookEventName` 的 serde 表示保持一致，不额外接受别名。
    serde_json::from_value::<RuntimeHookEventName>(serde_json::Value::String(value.to_string()))
        .ok()
}

/// 从单个配置文件发现 Hook。`source` 决定信任与优先级，由调用方按配置层级传入。
pub fn discover_hooks_from_file(
    path: &Path,
    source: RuntimeHookSource,
    trust_status: RuntimeHookTrustStatus,
) -> Result<Vec<DiscoveredHook>, HookDiscoveryError> {
    let content =
        std::fs::read_to_string(path).map_err(|error| HookDiscoveryError::Unreadable {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let config: HookConfigFile =
        serde_json::from_str(&content).map_err(|error| HookDiscoveryError::Malformed {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;

    let mut discovered = Vec::new();
    let mut display_order = 0_i64;
    for (event_label, entries) in &config.hooks {
        let event_name =
            parse_event_name(event_label).ok_or_else(|| HookDiscoveryError::UnknownEvent {
                path: path.to_path_buf(),
                event: event_label.clone(),
            })?;
        for (index, entry) in entries.iter().enumerate() {
            if entry.command.trim().is_empty() {
                return Err(HookDiscoveryError::EmptyCommand {
                    path: path.to_path_buf(),
                    event: event_label.clone(),
                });
            }
            if entry.timeout_sec == 0 {
                return Err(HookDiscoveryError::InvalidTimeout {
                    path: path.to_path_buf(),
                    event: event_label.clone(),
                });
            }
            discovered.push(DiscoveredHook {
                snapshot: RuntimeHookSnapshot {
                    key: format!("{}:{event_label}:{index}", source_label(source)),
                    event_name,
                    handler_type: RuntimeHookHandlerType::Command,
                    execution_mode: RuntimeHookExecutionMode::Sync,
                    matcher: entry
                        .matcher
                        .as_deref()
                        .map(str::trim)
                        .filter(|matcher| !matcher.is_empty())
                        .map(str::to_string),
                    timeout_sec: entry.timeout_sec,
                    status_message: None,
                    source_path: path.to_path_buf(),
                    source,
                    display_order,
                    enabled: entry.enabled,
                    trust_status,
                },
                command: entry.command.clone(),
                blocking: entry.blocking,
            });
            display_order += 1;
        }
    }
    Ok(discovered)
}

fn source_label(source: RuntimeHookSource) -> &'static str {
    match source {
        RuntimeHookSource::System => "system",
        RuntimeHookSource::User => "user",
        RuntimeHookSource::Project => "project",
        RuntimeHookSource::Mdm => "mdm",
        RuntimeHookSource::SessionFlags => "session_flags",
        RuntimeHookSource::Plugin => "plugin",
        RuntimeHookSource::CloudRequirements => "cloud_requirements",
        RuntimeHookSource::CloudManagedConfig => "cloud_managed_config",
        RuntimeHookSource::LegacyManagedConfigFile => "legacy_managed_config_file",
        RuntimeHookSource::LegacyManagedConfigMdm => "legacy_managed_config_mdm",
        RuntimeHookSource::Unknown => "unknown",
    }
}

fn shell_command_flag(shell: &str) -> &'static str {
    let executable = Path::new(shell)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(shell)
        .to_ascii_lowercase();

    if executable == "cmd" || executable == "cmd.exe" {
        "/C"
    } else if executable.contains("powershell") || executable == "pwsh" || executable == "pwsh.exe"
    {
        "-Command"
    } else {
        "-c"
    }
}

/// 解析执行 Hook 的 shell，同时覆盖 macOS 与 Windows。
pub fn resolve_command_shell() -> String {
    let shell_from_env = std::env::var("SHELL").ok().and_then(|value| {
        let cleaned = value
            .split('\0')
            .next()
            .unwrap_or_default()
            .trim()
            .to_string();
        (!cleaned.is_empty()).then_some(cleaned)
    });

    #[cfg(target_os = "windows")]
    {
        if let Some(shell) = shell_from_env {
            let path = Path::new(&shell);
            if path.is_absolute() && path.exists() {
                return shell;
            }
        }
        if let Ok(comspec) = std::env::var("COMSPEC") {
            let cleaned = comspec
                .split('\0')
                .next()
                .unwrap_or_default()
                .trim()
                .to_string();
            if !cleaned.is_empty() && Path::new(&cleaned).exists() {
                return cleaned;
            }
        }
        "cmd.exe".to_string()
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Some(shell) = shell_from_env {
            if Path::new(&shell).exists() {
                return shell;
            }
        }
        if Path::new("/bin/sh").exists() {
            "/bin/sh".to_string()
        } else {
            "sh".to_string()
        }
    }
}

/// command handler 的进程输出，尚未翻译成裁决结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookCommandOutput {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

/// handler stdout 允许携带的结构化结果。未提供则按纯放行处理。
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HookCommandPayload {
    #[serde(alias = "additional_context")]
    additional_context: Option<String>,
    #[serde(alias = "rewritten_arguments")]
    rewritten_arguments: Option<String>,
    #[serde(alias = "block_reason")]
    block_reason: Option<String>,
    #[serde(alias = "abort_reason")]
    abort_reason: Option<String>,
}

/// 执行一个 `Command` Hook。超时和 spawn 失败都不会静默放行。
pub async fn run_command_hook(
    hook: &DiscoveredHook,
    context: &RuntimeHookEventContext,
) -> HookCommandOutput {
    let shell = resolve_command_shell();
    let shell_flag = shell_command_flag(&shell);
    let context_json = serde_json::json!({
        "toolName": context.tool_name,
        "toolArguments": context.tool_arguments,
        "content": context.content,
    })
    .to_string();

    let child = Command::new(&shell)
        .arg(shell_flag)
        .arg(&hook.command)
        .env("LIME_HOOK_EVENT", event_env_value(hook.snapshot.event_name))
        .env(
            "LIME_HOOK_TOOL_NAME",
            context.tool_name.as_deref().unwrap_or(""),
        )
        .env("LIME_HOOK_CONTEXT", &context_json)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let mut child = match child {
        Ok(child) => child,
        Err(error) => {
            return HookCommandOutput {
                success: false,
                stdout: String::new(),
                stderr: format!("hook spawn failed: {error}"),
                timed_out: false,
            };
        }
    };

    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        let _ = stdin.write_all(context_json.as_bytes()).await;
        drop(stdin);
    }

    match tokio::time::timeout(
        Duration::from_secs(hook.snapshot.timeout_sec),
        child.wait_with_output(),
    )
    .await
    {
        Ok(Ok(output)) => HookCommandOutput {
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            timed_out: false,
        },
        Ok(Err(error)) => HookCommandOutput {
            success: false,
            stdout: String::new(),
            stderr: format!("hook wait failed: {error}"),
            timed_out: false,
        },
        Err(_) => HookCommandOutput {
            success: false,
            stdout: String::new(),
            stderr: format!("hook timed out after {}s", hook.snapshot.timeout_sec),
            timed_out: true,
        },
    }
}

fn event_env_value(event_name: RuntimeHookEventName) -> String {
    serde_json::to_value(event_name)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_default()
}

/// 把进程输出翻译成裁决输入。
///
/// 失败的 blocking hook 阻断；失败的非 blocking hook 只放行且不注入上下文，
/// 不把 stderr 当作模型可见内容。
pub fn report_from_output(
    hook: &DiscoveredHook,
    output: &HookCommandOutput,
) -> RuntimeHookHandlerReport {
    if !output.success {
        return if hook.blocking {
            RuntimeHookHandlerReport::Block {
                reason: blocking_reason(output),
            }
        } else {
            RuntimeHookHandlerReport::Allow {
                additional_context: None,
            }
        };
    }

    let payload = serde_json::from_str::<HookCommandPayload>(output.stdout.trim())
        .unwrap_or_else(|_| HookCommandPayload::default());

    if let Some(reason) = non_empty(payload.abort_reason) {
        return RuntimeHookHandlerReport::Abort { reason };
    }
    if let Some(reason) = non_empty(payload.block_reason) {
        return RuntimeHookHandlerReport::Block { reason };
    }
    if let Some(arguments) = non_empty(payload.rewritten_arguments) {
        return RuntimeHookHandlerReport::Rewrite { arguments };
    }
    RuntimeHookHandlerReport::Allow {
        additional_context: non_empty(payload.additional_context),
    }
}

fn blocking_reason(output: &HookCommandOutput) -> String {
    if output.timed_out {
        return output.stderr.clone();
    }
    let stderr = output.stderr.trim();
    if stderr.is_empty() {
        "hook exited with a non-zero status".to_string()
    } else {
        stderr.to_string()
    }
}

fn non_empty(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_config(contents: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().expect("temp config");
        file.write_all(contents.as_bytes()).expect("write config");
        file.flush().expect("flush config");
        file
    }

    fn discovered(command: &str, blocking: bool, timeout_sec: u64) -> DiscoveredHook {
        DiscoveredHook {
            snapshot: RuntimeHookSnapshot {
                key: "project:pre_tool_use:0".to_string(),
                event_name: RuntimeHookEventName::PreToolUse,
                handler_type: RuntimeHookHandlerType::Command,
                execution_mode: RuntimeHookExecutionMode::Sync,
                matcher: None,
                timeout_sec,
                status_message: None,
                source_path: PathBuf::from("/tmp/hooks.json"),
                source: RuntimeHookSource::Project,
                display_order: 0,
                enabled: true,
                trust_status: RuntimeHookTrustStatus::Trusted,
            },
            command: command.to_string(),
            blocking,
        }
    }

    #[test]
    fn discovers_codex_aligned_events_with_stable_keys_and_order() {
        let file = write_config(
            r#"{"hooks":{"pre_tool_use":[{"command":"echo one","matcher":"^shell$"},
                {"command":"echo two","blocking":true,"timeout_sec":3}],
                "session_end":[{"command":"echo bye","enabled":false}]}}"#,
        );

        let hooks = discover_hooks_from_file(
            file.path(),
            RuntimeHookSource::Project,
            RuntimeHookTrustStatus::Trusted,
        )
        .expect("discovery");

        assert_eq!(hooks.len(), 3);
        // BTreeMap 顺序：pre_tool_use 在 session_end 之前，display_order 全局单调。
        assert_eq!(hooks[0].snapshot.key, "project:pre_tool_use:0");
        assert_eq!(hooks[0].snapshot.matcher.as_deref(), Some("^shell$"));
        assert_eq!(hooks[0].snapshot.timeout_sec, DEFAULT_TIMEOUT_SEC);
        assert!(!hooks[0].blocking);
        assert_eq!(hooks[1].snapshot.key, "project:pre_tool_use:1");
        assert_eq!(hooks[1].snapshot.timeout_sec, 3);
        assert!(hooks[1].blocking);
        assert_eq!(
            hooks[2].snapshot.event_name,
            RuntimeHookEventName::SessionEnd
        );
        assert!(!hooks[2].snapshot.enabled);
        assert_eq!(
            hooks
                .iter()
                .map(|h| h.snapshot.display_order)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(hooks
            .iter()
            .all(|hook| hook.snapshot.execution_mode == RuntimeHookExecutionMode::Sync));
    }

    #[test]
    fn retired_event_names_and_flat_format_fail_closed() {
        let retired = write_config(r#"{"hooks":{"BeforeToolCall":[{"command":"echo hi"}]}}"#);
        let flat = write_config(r#"{"hooks":[{"event":"BeforeToolCall","command":"echo hi"}]}"#);

        let retired_error = discover_hooks_from_file(
            retired.path(),
            RuntimeHookSource::Project,
            RuntimeHookTrustStatus::Trusted,
        )
        .expect_err("retired event must fail closed");
        assert!(matches!(
            retired_error,
            HookDiscoveryError::UnknownEvent { ref event, .. } if event == "BeforeToolCall"
        ));

        assert!(matches!(
            discover_hooks_from_file(
                flat.path(),
                RuntimeHookSource::Project,
                RuntimeHookTrustStatus::Trusted,
            ),
            Err(HookDiscoveryError::Malformed { .. })
        ));
    }

    #[test]
    fn empty_command_zero_timeout_and_unknown_field_fail_closed() {
        let empty = write_config(r#"{"hooks":{"pre_tool_use":[{"command":"   "}]}}"#);
        let zero = write_config(r#"{"hooks":{"pre_tool_use":[{"command":"x","timeout_sec":0}]}}"#);
        let unknown =
            write_config(r#"{"hooks":{"pre_tool_use":[{"command":"x","async_exec":true}]}}"#);

        for (file, expect_malformed) in [(empty, false), (zero, false), (unknown, true)] {
            let error = discover_hooks_from_file(
                file.path(),
                RuntimeHookSource::Project,
                RuntimeHookTrustStatus::Trusted,
            )
            .expect_err("must fail closed");
            if expect_malformed {
                assert!(matches!(error, HookDiscoveryError::Malformed { .. }));
            } else {
                assert!(matches!(
                    error,
                    HookDiscoveryError::EmptyCommand { .. }
                        | HookDiscoveryError::InvalidTimeout { .. }
                ));
            }
        }
    }

    #[test]
    fn unreadable_config_fails_closed() {
        assert!(matches!(
            discover_hooks_from_file(
                Path::new("/nonexistent/lime-hooks.json"),
                RuntimeHookSource::Project,
                RuntimeHookTrustStatus::Trusted,
            ),
            Err(HookDiscoveryError::Unreadable { .. })
        ));
    }

    #[test]
    fn failed_blocking_hook_blocks_and_failed_optional_hook_allows_without_context() {
        let output = HookCommandOutput {
            success: false,
            stdout: "{\"additionalContext\":\"ignored\"}".to_string(),
            stderr: "boom".to_string(),
            timed_out: false,
        };

        assert_eq!(
            report_from_output(&discovered("x", true, 10), &output),
            RuntimeHookHandlerReport::Block {
                reason: "boom".to_string()
            }
        );
        assert_eq!(
            report_from_output(&discovered("x", false, 10), &output),
            RuntimeHookHandlerReport::Allow {
                additional_context: None
            }
        );
    }

    #[test]
    fn timeout_reason_is_preserved_for_blocking_hooks() {
        let output = HookCommandOutput {
            success: false,
            stdout: String::new(),
            stderr: "hook timed out after 3s".to_string(),
            timed_out: true,
        };

        assert_eq!(
            report_from_output(&discovered("x", true, 3), &output),
            RuntimeHookHandlerReport::Block {
                reason: "hook timed out after 3s".to_string()
            }
        );
    }

    #[test]
    fn structured_stdout_selects_abort_over_block_over_rewrite() {
        let hook = discovered("x", false, 10);
        let success = |stdout: &str| HookCommandOutput {
            success: true,
            stdout: stdout.to_string(),
            stderr: String::new(),
            timed_out: false,
        };

        assert_eq!(
            report_from_output(
                &hook,
                &success(
                    r#"{"abortReason":"stop","blockReason":"deny","rewrittenArguments":"{}"}"#
                )
            ),
            RuntimeHookHandlerReport::Abort {
                reason: "stop".to_string()
            }
        );
        assert_eq!(
            report_from_output(
                &hook,
                &success(r#"{"blockReason":"deny","rewrittenArguments":"{}"}"#)
            ),
            RuntimeHookHandlerReport::Block {
                reason: "deny".to_string()
            }
        );
        assert_eq!(
            report_from_output(&hook, &success(r#"{"rewrittenArguments":"{\"safe\":1}"}"#)),
            RuntimeHookHandlerReport::Rewrite {
                arguments: "{\"safe\":1}".to_string()
            }
        );
    }

    #[test]
    fn non_json_success_output_is_a_plain_allow() {
        let hook = discovered("x", false, 10);

        assert_eq!(
            report_from_output(
                &hook,
                &HookCommandOutput {
                    success: true,
                    stdout: "not json".to_string(),
                    stderr: String::new(),
                    timed_out: false,
                }
            ),
            RuntimeHookHandlerReport::Allow {
                additional_context: None
            }
        );
    }

    #[test]
    fn blank_structured_fields_do_not_become_decisions() {
        let hook = discovered("x", false, 10);

        assert_eq!(
            report_from_output(
                &hook,
                &HookCommandOutput {
                    success: true,
                    stdout: r#"{"abortReason":"  ","blockReason":"","additionalContext":" note "}"#
                        .to_string(),
                    stderr: String::new(),
                    timed_out: false,
                }
            ),
            RuntimeHookHandlerReport::Allow {
                additional_context: Some("note".to_string())
            }
        );
    }

    #[tokio::test]
    async fn command_hook_runs_and_reports_structured_context() {
        let hook = discovered("printf '{\"additionalContext\":\"from hook\"}'", false, 10);
        let context = RuntimeHookEventContext {
            tool_name: Some("shell".to_string()),
            tool_arguments: Some("{}".to_string()),
            content: None,
        };

        let output = run_command_hook(&hook, &context).await;

        assert!(output.success, "stderr: {}", output.stderr);
        assert!(!output.timed_out);
        assert_eq!(
            report_from_output(&hook, &output),
            RuntimeHookHandlerReport::Allow {
                additional_context: Some("from hook".to_string())
            }
        );
    }

    #[tokio::test]
    async fn failing_blocking_command_hook_blocks_with_process_stderr() {
        let hook = discovered("printf 'nope' >&2; exit 3", true, 10);

        let output = run_command_hook(&hook, &RuntimeHookEventContext::default()).await;

        assert!(!output.success);
        assert_eq!(
            report_from_output(&hook, &output),
            RuntimeHookHandlerReport::Block {
                reason: "nope".to_string()
            }
        );
    }

    #[tokio::test]
    async fn timed_out_command_hook_is_reported_as_timeout() {
        let hook = discovered("sleep 5", true, 1);

        let output = run_command_hook(&hook, &RuntimeHookEventContext::default()).await;

        assert!(output.timed_out);
        assert!(!output.success);
        assert!(matches!(
            report_from_output(&hook, &output),
            RuntimeHookHandlerReport::Block { ref reason } if reason.contains("timed out")
        ));
    }
}
