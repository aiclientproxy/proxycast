use crate::execution_process::live::RuntimeLiveExecutionGateway;
use crate::tool_definition::RuntimeToolDefinition;
use crate::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionResult, RuntimeToolPolicyErrorKind,
    RuntimeToolTurnContext,
};
use crate::tool_io::{
    estimate_tool_io_tokens, format_tool_output_for_model, ToolOutputTruncationPolicy,
};
use app_server_protocol::{
    ExecutionProcessDrainOutputParams, ExecutionProcessIdParams, ExecutionProcessStartParams,
    ExecutionProcessStatus, ExecutionProcessWriteStdinParams,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

pub const EXEC_COMMAND_TOOL_NAME: &str = "exec_command";
pub const WRITE_STDIN_TOOL_NAME: &str = "write_stdin";

const DEFAULT_EXEC_YIELD_TIME_MS: u64 = 10_000;
const MIN_EXEC_YIELD_TIME_MS: u64 = 250;
const MAX_EXEC_YIELD_TIME_MS: u64 = 30_000;
const DEFAULT_WRITE_YIELD_TIME_MS: u64 = 250;
const DEFAULT_POLL_YIELD_TIME_MS: u64 = 5_000;
const MAX_POLL_YIELD_TIME_MS: u64 = 300_000;
const DEFAULT_MAX_OUTPUT_TOKENS: usize = 10_000;
const MAX_OUTPUT_TOKENS: usize = 100_000;
const OUTPUT_DRAIN_LIMIT: u16 = 256;
const OUTPUT_DRAIN_MAX_BYTES: u64 = 1024 * 1024;
const PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Clone)]
pub struct RuntimeUnifiedExecToolRequest<'a> {
    pub tool_name: &'a str,
    pub params: &'a Value,
    pub thread_id: &'a str,
    pub working_directory: PathBuf,
    pub environment: HashMap<String, String>,
    pub tool_call_id: String,
    pub cancel_token: Option<CancellationToken>,
    pub turn_context: Option<&'a RuntimeToolTurnContext>,
    pub granted_permissions: Option<app_server_protocol::protocol::v2::GrantedPermissionProfile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecCommandInput {
    cmd: String,
    #[serde(default)]
    workdir: Option<String>,
    #[serde(default)]
    shell: Option<String>,
    #[serde(default = "default_login")]
    login: bool,
    #[serde(default)]
    tty: bool,
    #[serde(default)]
    yield_time_ms: Option<u64>,
    #[serde(default)]
    max_output_tokens: Option<usize>,
    #[serde(default)]
    sandbox_permissions: Option<String>,
    #[serde(default)]
    justification: Option<String>,
    #[serde(default)]
    prefix_rule: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteStdinInput {
    session_id: i32,
    #[serde(default)]
    chars: String,
    #[serde(default)]
    yield_time_ms: Option<u64>,
    #[serde(default)]
    max_output_tokens: Option<usize>,
}

#[derive(Debug)]
struct UnifiedExecSession {
    thread_id: String,
    process_id: String,
    call_id: String,
    command: String,
    cwd: String,
    after_sequence: Option<u64>,
    empty_poll_count: u64,
}

#[derive(Default)]
struct UnifiedExecSessionRegistry {
    sessions: HashMap<i32, UnifiedExecSessionRegistration>,
}

struct UnifiedExecSessionRegistration {
    process_id: String,
    session: Arc<tokio::sync::Mutex<UnifiedExecSession>>,
}

#[derive(Debug)]
struct UnifiedExecCallOutput {
    session_id: Option<i32>,
    call_id: String,
    command: String,
    cwd: String,
    output: String,
    exit_code: Option<i32>,
    observation: UnifiedExecObservationKind,
    empty_poll_count: u64,
    wall_time: Duration,
    max_output_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnifiedExecObservationKind {
    Output,
    Waiting,
    Terminal,
}

impl UnifiedExecObservationKind {
    fn label(self) -> &'static str {
        match self {
            Self::Output => "output",
            Self::Waiting => "waiting",
            Self::Terminal => "terminal",
        }
    }
}

pub fn unified_exec_tool_definitions() -> Vec<RuntimeToolDefinition> {
    vec![
        exec_command_tool_definition(),
        write_stdin_tool_definition(),
    ]
}

pub fn exec_command_tool_definition() -> RuntimeToolDefinition {
    RuntimeToolDefinition::new(
        EXEC_COMMAND_TOOL_NAME,
        "Runs a command, returning output or a session ID for ongoing interaction.",
        json!({
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "Shell command to execute."
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command. Defaults to the turn cwd."
                },
                "shell": {
                    "type": "string",
                    "description": "Shell binary to launch. Defaults to the user's default shell."
                },
                "login": {
                    "type": "boolean",
                    "description": "True runs the shell with login semantics; false disables them. Defaults to true."
                },
                "tty": {
                    "type": "boolean",
                    "description": "True requests a PTY; false or omitted uses plain pipes."
                },
                "yield_time_ms": {
                    "type": "number",
                    "description": "Wait before yielding output. Defaults to 10000 ms; effective range is 250-30000 ms."
                },
                "max_output_tokens": {
                    "type": "number",
                    "description": "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy."
                },
                "sandbox_permissions": {
                    "type": "string",
                    "enum": ["use_default", "require_escalated"],
                    "description": "Per-command sandbox override. Defaults to use_default."
                },
                "justification": {
                    "type": "string",
                    "description": "User-facing approval question for require_escalated; omit otherwise."
                },
                "prefix_rule": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Reusable approval prefix for cmd, only with require_escalated."
                }
            },
            "required": ["cmd"],
            "additionalProperties": false
        }),
    )
}

pub fn write_stdin_tool_definition() -> RuntimeToolDefinition {
    RuntimeToolDefinition::new(
        WRITE_STDIN_TOOL_NAME,
        "Writes characters to an existing exec session and returns recent output.",
        json!({
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "number",
                    "description": "Identifier of the running exec session."
                },
                "chars": {
                    "type": "string",
                    "description": "Bytes to write to stdin. Defaults to empty, which polls without writing."
                },
                "yield_time_ms": {
                    "type": "number",
                    "description": "Wait before yielding output. Non-empty writes default to 250 ms; empty polls default to 5000 ms."
                },
                "max_output_tokens": {
                    "type": "number",
                    "description": "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy."
                }
            },
            "required": ["session_id"],
            "additionalProperties": false
        }),
    )
}

pub fn is_unified_exec_tool_name(tool_name: &str) -> bool {
    matches!(
        tool_name.trim(),
        EXEC_COMMAND_TOOL_NAME | WRITE_STDIN_TOOL_NAME
    )
}

pub async fn execute_runtime_unified_exec_tool(
    gateway: Arc<dyn RuntimeLiveExecutionGateway>,
    request: RuntimeUnifiedExecToolRequest<'_>,
) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
    match request.tool_name.trim() {
        EXEC_COMMAND_TOOL_NAME => execute_command(gateway, request).await,
        WRITE_STDIN_TOOL_NAME => write_stdin(gateway, request).await,
        name => Err(unified_exec_error(format!(
            "unsupported unified exec tool: {name}"
        ))),
    }
}

async fn execute_command(
    gateway: Arc<dyn RuntimeLiveExecutionGateway>,
    request: RuntimeUnifiedExecToolRequest<'_>,
) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
    let input: ExecCommandInput =
        serde_json::from_value(request.params.clone()).map_err(|error| {
            unified_exec_error(format!("exec_command arguments are invalid: {error}"))
        })?;
    validate_exec_approval_fields(&input)?;
    let command = require_non_empty(input.cmd, "cmd")?;
    let cwd = resolve_working_directory(&request.working_directory, input.workdir.as_deref())?;
    let call_id = require_non_empty(request.tool_call_id, "tool call id")?;
    let session_id = next_session_id();
    let process_id = format!("unified-exec-{session_id}");
    let shell_command = build_shell_command(&command, input.shell.as_deref(), input.login);
    let cwd_text = cwd.to_string_lossy().to_string();
    let session = Arc::new(tokio::sync::Mutex::new(UnifiedExecSession {
        thread_id: request.thread_id.to_string(),
        process_id: process_id.clone(),
        call_id: call_id.clone(),
        command: command.clone(),
        cwd: cwd_text.clone(),
        after_sequence: None,
        empty_poll_count: 0,
    }));
    register_session(session_id, process_id.clone(), Arc::clone(&session))?;

    let start_result = gateway
        .start_process(
            request.thread_id,
            &command,
            ExecutionProcessStartParams {
                process_id,
                tool_id: call_id,
                tool_name: EXEC_COMMAND_TOOL_NAME.to_string(),
                command: shell_command,
                working_directory: cwd_text,
                tty: input.tty,
                approval_policy: Some("never".to_string()),
                sandbox_policy: effective_sandbox_policy(
                    input.sandbox_permissions.as_deref(),
                    request.turn_context,
                ),
                runtime_metadata: request
                    .turn_context
                    .map(|context| serde_json::to_value(context).unwrap_or(Value::Null)),
                cwd: None,
                env: request.environment,
            },
            request.granted_permissions,
        )
        .await;
    if let Err(error) = start_result {
        remove_session(session_id);
        return Err(unified_exec_error(error));
    }

    let output = collect_process_output(
        gateway,
        session_id,
        session,
        clamped_exec_yield_time(input.yield_time_ms),
        clamped_output_tokens(input.max_output_tokens),
        request.cancel_token,
    )
    .await?;
    Ok(project_output(output))
}

async fn write_stdin(
    gateway: Arc<dyn RuntimeLiveExecutionGateway>,
    request: RuntimeUnifiedExecToolRequest<'_>,
) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
    let input: WriteStdinInput =
        serde_json::from_value(request.params.clone()).map_err(|error| {
            unified_exec_error(format!("write_stdin arguments are invalid: {error}"))
        })?;
    let session = find_session(input.session_id)?;
    {
        let session = session.lock().await;
        if session.thread_id != request.thread_id {
            return Err(unified_exec_error(format!(
                "unified exec session {} does not belong to thread {}",
                input.session_id, request.thread_id
            )));
        }
        gateway
            .write_stdin(ExecutionProcessWriteStdinParams {
                process_id: session.process_id.clone(),
                data: input.chars.clone(),
            })
            .map_err(unified_exec_error)?;
    }
    let output = collect_process_output(
        gateway,
        input.session_id,
        session,
        clamped_write_yield_time(input.yield_time_ms, input.chars.is_empty()),
        clamped_output_tokens(input.max_output_tokens),
        request.cancel_token,
    )
    .await?;
    Ok(project_output(output))
}

async fn collect_process_output(
    gateway: Arc<dyn RuntimeLiveExecutionGateway>,
    session_id: i32,
    session: Arc<tokio::sync::Mutex<UnifiedExecSession>>,
    yield_time: Duration,
    max_output_tokens: usize,
    cancel_token: Option<CancellationToken>,
) -> Result<UnifiedExecCallOutput, RuntimeToolExecutionError> {
    let started_at = Instant::now();
    let deadline = started_at + yield_time;
    let mut output = String::new();

    loop {
        let (process_id, after_sequence) = {
            let session = session.lock().await;
            (session.process_id.clone(), session.after_sequence)
        };
        let drained = gateway
            .drain_output(ExecutionProcessDrainOutputParams {
                process_id: Some(process_id.clone()),
                after_sequence,
                limit: Some(OUTPUT_DRAIN_LIMIT),
                max_bytes: Some(OUTPUT_DRAIN_MAX_BYTES),
            })
            .map_err(unified_exec_error)?;
        for delta in drained.deltas {
            output.push_str(&delta.delta);
        }
        if let Some(next_sequence) = drained.next_sequence {
            session.lock().await.after_sequence = Some(next_sequence);
        }

        if cancel_token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            let _ = gateway.terminate(ExecutionProcessIdParams {
                process_id: process_id.clone(),
            });
        }

        let snapshot = gateway
            .status(ExecutionProcessIdParams { process_id })
            .map_err(unified_exec_error)?
            .snapshot;
        if process_status_is_terminal(snapshot.status) {
            let final_drain = gateway
                .drain_output(ExecutionProcessDrainOutputParams {
                    process_id: Some(snapshot.process_id.clone()),
                    after_sequence: session.lock().await.after_sequence,
                    limit: Some(OUTPUT_DRAIN_LIMIT),
                    max_bytes: Some(OUTPUT_DRAIN_MAX_BYTES),
                })
                .map_err(unified_exec_error)?;
            for delta in final_drain.deltas {
                output.push_str(&delta.delta);
            }
            let facts = session_facts(&session).await;
            remove_session(session_id);
            return Ok(UnifiedExecCallOutput {
                session_id: None,
                call_id: facts.call_id,
                command: facts.command,
                cwd: facts.cwd,
                output,
                exit_code: snapshot.exit_code,
                observation: UnifiedExecObservationKind::Terminal,
                empty_poll_count: 0,
                wall_time: started_at.elapsed(),
                max_output_tokens,
            });
        }

        if Instant::now() >= deadline {
            let mut facts = session.lock().await;
            let observation = if output.is_empty() {
                facts.empty_poll_count = facts.empty_poll_count.saturating_add(1);
                UnifiedExecObservationKind::Waiting
            } else {
                facts.empty_poll_count = 0;
                UnifiedExecObservationKind::Output
            };
            return Ok(UnifiedExecCallOutput {
                session_id: Some(session_id),
                call_id: facts.call_id.clone(),
                command: facts.command.clone(),
                cwd: facts.cwd.clone(),
                output,
                exit_code: None,
                observation,
                empty_poll_count: facts.empty_poll_count,
                wall_time: started_at.elapsed(),
                max_output_tokens,
            });
        }
        tokio::time::sleep(PROCESS_POLL_INTERVAL).await;
    }
}

async fn session_facts(session: &tokio::sync::Mutex<UnifiedExecSession>) -> UnifiedExecSession {
    let session = session.lock().await;
    UnifiedExecSession {
        thread_id: session.thread_id.clone(),
        process_id: session.process_id.clone(),
        call_id: session.call_id.clone(),
        command: session.command.clone(),
        cwd: session.cwd.clone(),
        after_sequence: session.after_sequence,
        empty_poll_count: session.empty_poll_count,
    }
}

fn project_output(output: UnifiedExecCallOutput) -> RuntimeToolExecutionResult {
    let original_token_count = estimate_tool_io_tokens(&output.output);
    let visible_output = format_tool_output_for_model(
        &output.output,
        ToolOutputTruncationPolicy::Tokens(output.max_output_tokens),
    );
    let mut structured = json!({
        "chunk_id": Uuid::new_v4().to_string(),
        "wall_time_seconds": output.wall_time.as_secs_f64(),
        "original_token_count": original_token_count,
        "output": visible_output,
        "observation": {
            "kind": output.observation.label(),
            "empty_poll_count": output.empty_poll_count,
            "process_active": output.session_id.is_some(),
        },
    });
    if let Some(session_id) = output.session_id {
        structured["session_id"] = json!(session_id);
    }
    if let Some(exit_code) = output.exit_code {
        structured["exit_code"] = json!(exit_code);
    }
    let success = output.exit_code.map(|code| code == 0).unwrap_or(true);
    let serialized = serde_json::to_string(&structured).unwrap_or_else(|_| "{}".to_string());
    let metadata = HashMap::from([
        ("exec_command_call_id".to_string(), json!(output.call_id)),
        ("command".to_string(), json!(output.command)),
        ("cwd".to_string(), json!(output.cwd)),
        ("session_id".to_string(), json!(output.session_id)),
        ("exit_code".to_string(), json!(output.exit_code)),
        (
            "wall_time_seconds".to_string(),
            json!(output.wall_time.as_secs_f64()),
        ),
        (
            "original_token_count".to_string(),
            json!(original_token_count),
        ),
        ("command_output".to_string(), json!(visible_output)),
        (
            "unified_exec_observation".to_string(),
            json!(output.observation.label()),
        ),
        (
            "unified_exec_empty_poll_count".to_string(),
            json!(output.empty_poll_count),
        ),
        ("execution_surface".to_string(), json!("unified_exec")),
    ]);
    RuntimeToolExecutionResult::new(
        success,
        serialized,
        (!success).then_some(output.output),
        metadata,
    )
    .with_structured_content(structured)
}

fn build_shell_command(command: &str, shell: Option<&str>, login: bool) -> Vec<String> {
    #[cfg(target_os = "windows")]
    {
        let shell = shell
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("powershell.exe");
        if shell.to_ascii_lowercase().contains("powershell")
            || shell.to_ascii_lowercase().contains("pwsh")
        {
            return vec![
                shell.to_string(),
                "-NoProfile".to_string(),
                "-NonInteractive".to_string(),
                "-Command".to_string(),
                command.to_string(),
            ];
        }
        return vec![
            shell.to_string(),
            "/D".to_string(),
            "/S".to_string(),
            "/C".to_string(),
            command.to_string(),
        ];
    }

    #[cfg(not(target_os = "windows"))]
    {
        let shell = shell
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| {
                std::env::var("SHELL")
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            })
            .unwrap_or_else(|| "sh".to_string());
        vec![
            shell,
            if login { "-lc" } else { "-c" }.to_string(),
            command.to_string(),
        ]
    }
}

fn resolve_working_directory(
    turn_cwd: &Path,
    requested: Option<&str>,
) -> Result<PathBuf, RuntimeToolExecutionError> {
    let requested = requested.map(str::trim).filter(|value| !value.is_empty());
    let candidate = match requested {
        Some(value) if Path::new(value).is_absolute() => PathBuf::from(value),
        Some(value) => turn_cwd.join(value),
        None => turn_cwd.to_path_buf(),
    };
    let canonical = std::fs::canonicalize(&candidate).map_err(|error| {
        unified_exec_error(format!(
            "exec_command workdir '{}' is unavailable: {error}",
            candidate.display()
        ))
    })?;
    if !canonical.is_dir() {
        return Err(unified_exec_error(format!(
            "exec_command workdir '{}' is not a directory",
            canonical.display()
        )));
    }
    Ok(canonical)
}

fn effective_sandbox_policy(
    sandbox_permissions: Option<&str>,
    turn_context: Option<&RuntimeToolTurnContext>,
) -> Option<String> {
    if sandbox_permissions == Some("require_escalated") {
        return Some("danger-full-access".to_string());
    }
    turn_context
        .and_then(|context| context.sandbox_policy.clone())
        .or_else(|| Some("workspace-write".to_string()))
}

fn validate_exec_approval_fields(
    input: &ExecCommandInput,
) -> Result<(), RuntimeToolExecutionError> {
    match input.sandbox_permissions.as_deref() {
        None | Some("use_default") => {
            if input.justification.is_some() || input.prefix_rule.is_some() {
                return Err(unified_exec_error(
                    "justification and prefix_rule require sandbox_permissions=require_escalated",
                ));
            }
        }
        Some("require_escalated") => {
            if input
                .justification
                .as_deref()
                .map(str::trim)
                .unwrap_or_default()
                .is_empty()
            {
                return Err(unified_exec_error(
                    "sandbox_permissions=require_escalated requires justification",
                ));
            }
        }
        Some(value) => {
            return Err(unified_exec_error(format!(
                "unsupported sandbox_permissions value: {value}"
            )));
        }
    }
    Ok(())
}

fn process_status_is_terminal(status: ExecutionProcessStatus) -> bool {
    matches!(
        status,
        ExecutionProcessStatus::Exited
            | ExecutionProcessStatus::Interrupted
            | ExecutionProcessStatus::Terminated
            | ExecutionProcessStatus::Failed
    )
}

fn default_login() -> bool {
    true
}

fn clamped_exec_yield_time(value: Option<u64>) -> Duration {
    Duration::from_millis(
        value
            .unwrap_or(DEFAULT_EXEC_YIELD_TIME_MS)
            .clamp(MIN_EXEC_YIELD_TIME_MS, MAX_EXEC_YIELD_TIME_MS),
    )
}

fn clamped_write_yield_time(value: Option<u64>, empty_poll: bool) -> Duration {
    let default = if empty_poll {
        DEFAULT_POLL_YIELD_TIME_MS
    } else {
        DEFAULT_WRITE_YIELD_TIME_MS
    };
    let max = if empty_poll {
        MAX_POLL_YIELD_TIME_MS
    } else {
        MAX_EXEC_YIELD_TIME_MS
    };
    Duration::from_millis(value.unwrap_or(default).clamp(MIN_EXEC_YIELD_TIME_MS, max))
}

fn clamped_output_tokens(value: Option<usize>) -> usize {
    value
        .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS)
        .clamp(1, MAX_OUTPUT_TOKENS)
}

fn require_non_empty(
    value: impl Into<String>,
    field: &str,
) -> Result<String, RuntimeToolExecutionError> {
    let value = value.into();
    let value = value.trim();
    if value.is_empty() {
        return Err(unified_exec_error(format!(
            "missing required parameter: {field}"
        )));
    }
    Ok(value.to_string())
}

fn session_registry() -> &'static Mutex<UnifiedExecSessionRegistry> {
    static REGISTRY: OnceLock<Mutex<UnifiedExecSessionRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(UnifiedExecSessionRegistry::default()))
}

fn next_session_id() -> i32 {
    static NEXT_SESSION_ID: AtomicI32 = AtomicI32::new(1_000);
    NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed)
}

fn register_session(
    session_id: i32,
    process_id: String,
    session: Arc<tokio::sync::Mutex<UnifiedExecSession>>,
) -> Result<(), RuntimeToolExecutionError> {
    let mut registry = session_registry()
        .lock()
        .map_err(|_| unified_exec_error("unified exec session registry is unavailable"))?;
    registry.sessions.insert(
        session_id,
        UnifiedExecSessionRegistration {
            process_id,
            session,
        },
    );
    Ok(())
}

fn find_session(
    session_id: i32,
) -> Result<Arc<tokio::sync::Mutex<UnifiedExecSession>>, RuntimeToolExecutionError> {
    session_registry()
        .lock()
        .map_err(|_| unified_exec_error("unified exec session registry is unavailable"))?
        .sessions
        .get(&session_id)
        .map(|registration| Arc::clone(&registration.session))
        .ok_or_else(|| unified_exec_error(format!("unified exec session not found: {session_id}")))
}

fn remove_session(session_id: i32) {
    if let Ok(mut registry) = session_registry().lock() {
        registry.sessions.remove(&session_id);
    }
}

pub fn forget_process_session(process_id: &str) {
    if let Ok(mut registry) = session_registry().lock() {
        registry
            .sessions
            .retain(|_, registration| registration.process_id != process_id);
    }
}

fn unified_exec_error(message: impl Into<String>) -> RuntimeToolExecutionError {
    RuntimeToolExecutionError::new(
        message.into(),
        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
            "unified_exec".to_string(),
        )),
    )
}

#[cfg(test)]
#[path = "unified_exec/tests.rs"]
mod tests;
