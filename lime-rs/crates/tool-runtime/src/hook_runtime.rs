//! Command Hook execution and lifecycle reporting.

pub use crate::hook_discovery::DiscoveredHook;
use crate::hook_lifecycle::{
    RuntimeHookEventContext, RuntimeHookHandlerReport, RuntimeHookReportFuture, RuntimeHookReporter,
};
use crate::turn_snapshot::{RuntimeHookEventName, RuntimeHookSnapshot};
use agent_protocol::hook::{HookOutputEntry, HookOutputEntryKind, HookRunStatus, HookRunSummary};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookLifecyclePhase {
    Started,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookLifecycleEvent {
    pub phase: HookLifecyclePhase,
    pub turn_id: Option<String>,
    pub run: HookRunSummary,
}

pub trait HookLifecycleEmitter: Send + Sync {
    fn emit(&self, event: HookLifecycleEvent);
}

pub struct CommandHookReporter {
    hooks_by_key: HashMap<String, DiscoveredHook>,
    emitter: Arc<dyn HookLifecycleEmitter>,
}

impl CommandHookReporter {
    pub fn new(
        hooks: impl IntoIterator<Item = DiscoveredHook>,
        emitter: Arc<dyn HookLifecycleEmitter>,
    ) -> Self {
        Self {
            hooks_by_key: hooks
                .into_iter()
                .filter(|hook| hook.is_executable())
                .map(|hook| (hook.snapshot.key.clone(), hook))
                .collect(),
            emitter,
        }
    }
}

impl RuntimeHookReporter for CommandHookReporter {
    fn report<'a>(
        &'a self,
        snapshot: &'a RuntimeHookSnapshot,
        _event_name: RuntimeHookEventName,
        context: &'a RuntimeHookEventContext,
    ) -> RuntimeHookReportFuture<'a> {
        Box::pin(async move {
            let hook = self.hooks_by_key.get(&snapshot.key)?;
            let mut run = running_summary(snapshot, context.tool_call_id.as_deref());
            self.emitter.emit(HookLifecycleEvent {
                phase: HookLifecyclePhase::Started,
                turn_id: context.turn_id.clone(),
                run: run.clone(),
            });

            let started = Instant::now();
            let output = run_command_hook(hook, context).await;
            let report = report_from_output(hook, &output);
            let completed_at = chrono::Utc::now().timestamp();
            run.status = status_for_report(&report);
            run.completed_at = Some(completed_at);
            run.duration_ms = Some(started.elapsed().as_millis().min(i64::MAX as u128) as i64);
            run.entries = entries_for_report(&report);
            self.emitter.emit(HookLifecycleEvent {
                phase: HookLifecyclePhase::Completed,
                turn_id: context.turn_id.clone(),
                run,
            });
            Some(report)
        })
    }
}

fn running_summary(snapshot: &RuntimeHookSnapshot, tool_call_id: Option<&str>) -> HookRunSummary {
    let id = tool_call_id
        .map(|tool_call_id| format!("{}:{tool_call_id}", snapshot.run_id()))
        .unwrap_or_else(|| snapshot.run_id());
    HookRunSummary {
        id,
        event_name: snapshot.event_name,
        handler_type: snapshot.handler_type,
        execution_mode: snapshot.execution_mode,
        scope: snapshot.scope(),
        source_path: snapshot.source_path.clone(),
        source: snapshot.source,
        display_order: snapshot.display_order,
        status: HookRunStatus::Running,
        status_message: snapshot.status_message.clone(),
        started_at: chrono::Utc::now().timestamp(),
        completed_at: None,
        duration_ms: None,
        entries: Vec::new(),
    }
}

fn status_for_report(report: &RuntimeHookHandlerReport) -> HookRunStatus {
    match report {
        RuntimeHookHandlerReport::Allow { .. } | RuntimeHookHandlerReport::Rewrite { .. } => {
            HookRunStatus::Completed
        }
        RuntimeHookHandlerReport::Block { .. } => HookRunStatus::Blocked,
        RuntimeHookHandlerReport::Abort { .. } => HookRunStatus::Stopped,
        RuntimeHookHandlerReport::Failed { .. } => HookRunStatus::Failed,
    }
}

fn entries_for_report(report: &RuntimeHookHandlerReport) -> Vec<HookOutputEntry> {
    match report {
        RuntimeHookHandlerReport::Allow {
            additional_context: Some(text),
        } => vec![HookOutputEntry {
            kind: HookOutputEntryKind::Context,
            text: text.clone(),
        }],
        RuntimeHookHandlerReport::Block { reason } => vec![HookOutputEntry {
            kind: HookOutputEntryKind::Feedback,
            text: reason.clone(),
        }],
        RuntimeHookHandlerReport::Abort { reason } => vec![HookOutputEntry {
            kind: HookOutputEntryKind::Stop,
            text: reason.clone(),
        }],
        RuntimeHookHandlerReport::Failed { reason } => vec![HookOutputEntry {
            kind: HookOutputEntryKind::Error,
            text: reason.clone(),
        }],
        RuntimeHookHandlerReport::Allow {
            additional_context: None,
        }
        | RuntimeHookHandlerReport::Rewrite { .. } => Vec::new(),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookCommandOutput {
    pub success: bool,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HookCommandPayload {
    #[serde(rename = "continue")]
    continue_processing: Option<bool>,
    stop_reason: Option<String>,
    decision: Option<String>,
    reason: Option<String>,
    additional_context: Option<String>,
    hook_specific_output: Option<HookSpecificOutput>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HookSpecificOutput {
    hook_event_name: Option<String>,
    permission_decision: Option<String>,
    permission_decision_reason: Option<String>,
    updated_input: Option<serde_json::Value>,
    additional_context: Option<String>,
}

pub async fn run_command_hook(
    hook: &DiscoveredHook,
    context: &RuntimeHookEventContext,
) -> HookCommandOutput {
    let Some(command) = hook.snapshot.command.as_deref() else {
        return failed_output("command hook has no command");
    };
    if context.working_directory.as_os_str().is_empty() {
        return failed_output("command hook has no working directory");
    }
    let shell = resolve_command_shell();
    let shell_flag = shell_command_flag(&shell);
    let context_json = command_input(hook, context).to_string();
    let child = Command::new(&shell)
        .arg(shell_flag)
        .arg(command)
        .current_dir(&context.working_directory)
        .kill_on_drop(true)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(error) => return failed_output(format!("hook spawn failed: {error}")),
    };
    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        if let Err(error) = stdin.write_all(context_json.as_bytes()).await {
            return failed_output(format!("hook stdin write failed: {error}"));
        }
    }
    match tokio::time::timeout(
        Duration::from_secs(hook.snapshot.timeout_sec),
        child.wait_with_output(),
    )
    .await
    {
        Ok(Ok(output)) => HookCommandOutput {
            success: output.status.success(),
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            timed_out: false,
        },
        Ok(Err(error)) => failed_output(format!("hook wait failed: {error}")),
        Err(_) => HookCommandOutput {
            success: false,
            exit_code: None,
            stdout: String::new(),
            stderr: format!("hook timed out after {}s", hook.snapshot.timeout_sec),
            timed_out: true,
        },
    }
}

fn failed_output(message: impl Into<String>) -> HookCommandOutput {
    HookCommandOutput {
        success: false,
        exit_code: None,
        stdout: String::new(),
        stderr: message.into(),
        timed_out: false,
    }
}

fn command_input(hook: &DiscoveredHook, context: &RuntimeHookEventContext) -> serde_json::Value {
    let tool_input = context
        .tool_arguments
        .as_deref()
        .and_then(|arguments| serde_json::from_str(arguments).ok())
        .unwrap_or(serde_json::Value::Null);
    let tool_response = context
        .tool_output
        .as_deref()
        .and_then(|output| serde_json::from_str(output).ok())
        .unwrap_or(serde_json::Value::Null);
    serde_json::json!({
        "session_id": context.session_id,
        "turn_id": context.turn_id,
        "cwd": context.working_directory,
        "hook_event_name": event_config_label(hook.snapshot.event_name),
        "tool_name": context.tool_name,
        "tool_input": tool_input,
        "tool_response": tool_response,
        "tool_use_id": context.tool_call_id,
    })
}

fn event_config_label(event_name: RuntimeHookEventName) -> &'static str {
    match event_name {
        RuntimeHookEventName::PreToolUse => "PreToolUse",
        RuntimeHookEventName::PermissionRequest => "PermissionRequest",
        RuntimeHookEventName::PostToolUse => "PostToolUse",
        RuntimeHookEventName::PreCompact => "PreCompact",
        RuntimeHookEventName::PostCompact => "PostCompact",
        RuntimeHookEventName::SessionStart => "SessionStart",
        RuntimeHookEventName::SessionEnd => "SessionEnd",
        RuntimeHookEventName::UserPromptSubmit => "UserPromptSubmit",
        RuntimeHookEventName::SubagentStart => "SubagentStart",
        RuntimeHookEventName::SubagentStop => "SubagentStop",
        RuntimeHookEventName::Stop => "Stop",
    }
}

pub fn report_from_output(
    hook: &DiscoveredHook,
    output: &HookCommandOutput,
) -> RuntimeHookHandlerReport {
    if output.exit_code == Some(2) {
        let reason = non_empty(Some(output.stderr.clone())).unwrap_or_else(|| {
            format!(
                "{} hook exited with code 2 without feedback",
                event_config_label(hook.snapshot.event_name)
            )
        });
        return RuntimeHookHandlerReport::Block { reason };
    }
    if !output.success {
        return RuntimeHookHandlerReport::Failed {
            reason: failure_reason(output),
        };
    }
    let stdout = output.stdout.trim();
    if stdout.is_empty() {
        return RuntimeHookHandlerReport::Allow {
            additional_context: None,
        };
    }
    let payload = match serde_json::from_str::<HookCommandPayload>(stdout) {
        Ok(payload) => payload,
        Err(error) if stdout.starts_with('{') || stdout.starts_with('[') => {
            return RuntimeHookHandlerReport::Failed {
                reason: format!("hook returned invalid JSON output: {error}"),
            };
        }
        Err(_) => {
            return RuntimeHookHandlerReport::Allow {
                additional_context: None,
            };
        }
    };
    if payload.continue_processing == Some(false) {
        let reason =
            non_empty(payload.stop_reason).unwrap_or_else(|| "hook stopped execution".to_string());
        return RuntimeHookHandlerReport::Abort { reason };
    }
    if payload.decision.as_deref() == Some("block") {
        let reason =
            non_empty(payload.reason).unwrap_or_else(|| "hook blocked execution".to_string());
        return RuntimeHookHandlerReport::Block { reason };
    }
    if let Some(specific) = payload.hook_specific_output {
        if specific.hook_event_name.as_deref() != Some(event_config_label(hook.snapshot.event_name))
        {
            return RuntimeHookHandlerReport::Failed {
                reason: "hookSpecificOutput event does not match the invoked hook".to_string(),
            };
        }
        if specific.permission_decision.as_deref() == Some("deny") {
            let reason = non_empty(specific.permission_decision_reason)
                .unwrap_or_else(|| "hook denied execution".to_string());
            return RuntimeHookHandlerReport::Block { reason };
        }
        if let Some(arguments) = specific.updated_input {
            return RuntimeHookHandlerReport::Rewrite {
                arguments: arguments.to_string(),
            };
        }
        return RuntimeHookHandlerReport::Allow {
            additional_context: non_empty(specific.additional_context)
                .or_else(|| non_empty(payload.additional_context)),
        };
    }
    RuntimeHookHandlerReport::Allow {
        additional_context: non_empty(payload.additional_context),
    }
}

fn failure_reason(output: &HookCommandOutput) -> String {
    if output.timed_out {
        return output.stderr.clone();
    }
    let stderr = output.stderr.trim();
    if stderr.is_empty() {
        match output.exit_code {
            Some(exit_code) => format!("hook exited with code {exit_code}"),
            None => "hook exited without a status code".to_string(),
        }
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
    use crate::turn_snapshot::{
        RuntimeHookExecutionMode, RuntimeHookHandlerType, RuntimeHookSource, RuntimeHookTrustStatus,
    };
    use std::path::PathBuf;
    use std::sync::Mutex;

    fn discovered(command: &str, timeout_sec: u64) -> DiscoveredHook {
        DiscoveredHook {
            snapshot: RuntimeHookSnapshot {
                key: "project:pre_tool_use:0:0".to_string(),
                event_name: RuntimeHookEventName::PreToolUse,
                handler_type: RuntimeHookHandlerType::Command,
                execution_mode: RuntimeHookExecutionMode::Sync,
                matcher: None,
                command: Some(command.to_string()),
                timeout_sec,
                status_message: Some("checking".to_string()),
                additional_context_limit: None,
                source_path: PathBuf::from("/tmp/hooks.json"),
                source: RuntimeHookSource::Project,
                plugin_id: None,
                display_order: 0,
                enabled: true,
                is_managed: false,
                current_hash: "sha256:test".to_string(),
                trust_status: RuntimeHookTrustStatus::Trusted,
            },
            executable: true,
        }
    }

    fn context() -> RuntimeHookEventContext {
        RuntimeHookEventContext {
            session_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            tool_call_id: Some("call-1".to_string()),
            working_directory: std::env::temp_dir(),
            tool_name: Some("Bash".to_string()),
            tool_arguments: Some(r#"{"command":"pwd"}"#.to_string()),
            tool_output: None,
            content: None,
        }
    }

    fn successful(stdout: &str) -> HookCommandOutput {
        HookCommandOutput {
            success: true,
            exit_code: Some(0),
            stdout: stdout.to_string(),
            stderr: String::new(),
            timed_out: false,
        }
    }

    #[test]
    fn codex_pre_tool_output_can_block_rewrite_or_add_context() {
        let hook = discovered("true", 5);
        assert_eq!(
            report_from_output(
                &hook,
                &successful(
                    r#"{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"denied"}}"#,
                ),
            ),
            RuntimeHookHandlerReport::Block {
                reason: "denied".to_string()
            }
        );
        assert_eq!(
            report_from_output(
                &hook,
                &successful(
                    r#"{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","updatedInput":{"command":"echo safe"}}}"#,
                ),
            ),
            RuntimeHookHandlerReport::Rewrite {
                arguments: r#"{"command":"echo safe"}"#.to_string()
            }
        );
        assert_eq!(
            report_from_output(
                &hook,
                &successful(
                    r#"{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":"reviewed"}}"#,
                ),
            ),
            RuntimeHookHandlerReport::Allow {
                additional_context: Some("reviewed".to_string())
            }
        );
    }

    #[test]
    fn process_failures_fail_closed_and_exit_two_blocks() {
        let hook = discovered("false", 5);
        assert!(matches!(
            report_from_output(
                &hook,
                &HookCommandOutput {
                    success: false,
                    exit_code: Some(3),
                    stdout: String::new(),
                    stderr: String::new(),
                    timed_out: false,
                }
            ),
            RuntimeHookHandlerReport::Failed { .. }
        ));
        assert_eq!(
            report_from_output(
                &hook,
                &HookCommandOutput {
                    success: false,
                    exit_code: Some(2),
                    stdout: String::new(),
                    stderr: "policy denied".to_string(),
                    timed_out: false,
                }
            ),
            RuntimeHookHandlerReport::Block {
                reason: "policy denied".to_string()
            }
        );
    }

    #[derive(Default)]
    struct RecordingEmitter(Mutex<Vec<HookLifecycleEvent>>);

    impl HookLifecycleEmitter for RecordingEmitter {
        fn emit(&self, event: HookLifecycleEvent) {
            self.0.lock().expect("events").push(event);
        }
    }

    #[tokio::test]
    async fn command_reporter_executes_and_emits_paired_lifecycle() {
        let hook = discovered(
            "printf '{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"additionalContext\":\"from hook\"}}'",
            5,
        );
        let emitter = Arc::new(RecordingEmitter::default());
        let reporter = CommandHookReporter::new(vec![hook.clone()], emitter.clone());
        let context = context();

        let report = reporter
            .report(&hook.snapshot, RuntimeHookEventName::PreToolUse, &context)
            .await
            .expect("report");

        assert_eq!(
            report,
            RuntimeHookHandlerReport::Allow {
                additional_context: Some("from hook".to_string())
            }
        );
        let events = emitter.0.lock().expect("events");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].phase, HookLifecyclePhase::Started);
        assert_eq!(events[1].phase, HookLifecyclePhase::Completed);
        assert_eq!(events[0].run.id, events[1].run.id);
        assert_eq!(events[1].run.status, HookRunStatus::Completed);
        assert_eq!(events[1].run.entries[0].kind, HookOutputEntryKind::Context);
    }

    #[tokio::test]
    async fn timed_out_command_is_failed() {
        let hook = discovered("sleep 5", 1);
        let output = run_command_hook(&hook, &context()).await;
        assert!(output.timed_out);
        assert!(matches!(
            report_from_output(&hook, &output),
            RuntimeHookHandlerReport::Failed { ref reason } if reason.contains("timed out")
        ));
    }
}
