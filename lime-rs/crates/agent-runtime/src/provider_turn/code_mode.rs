use super::RuntimeToolStepSnapshot;
use futures::future::join_all;
use model_provider::current_client::{
    CurrentProviderCustomToolCall, CurrentProviderTool, CurrentProviderToolCall,
    CurrentProviderToolResult, FreeformToolFormat,
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio_util::sync::CancellationToken;
use tool_runtime::code_mode::{
    code_mode_exec_tool_description, code_mode_wait_tool_definition, parse_code_mode_exec_source,
    RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest, RuntimeCodeModeFuture,
    RuntimeCodeModeNestedToolCall, RuntimeCodeModeSessionDelegate, RuntimeCodeModeSessionHandle,
    RuntimeCodeModeTool, RuntimeCodeModeToolResult, RuntimeCodeModeWaitOutcome,
    RuntimeCodeModeWaitRequest, CODE_MODE_EXEC_FREEFORM_GRAMMAR, CODE_MODE_EXEC_TOOL_NAME,
    DEFAULT_CODE_MODE_MAX_OUTPUT_TOKENS, DEFAULT_CODE_MODE_WAIT_YIELD_TIME_MS,
};
use tool_runtime::tool_call::{ToolCall, ToolEnvironment};
use tool_runtime::tool_definition::RuntimeToolExposure;
use tool_runtime::tool_executor::{
    RuntimeToolExecutionContext, RuntimeToolExecutionContextInput, RuntimeToolExecutorHandle,
};
use tool_runtime::tool_lifecycle::{ToolLifecycleEmitter, ToolOutputDeltaEvent};
use tool_runtime::tool_result_projection::NormalizedToolOutput;

pub(super) enum PendingProviderToolCall {
    Function(CurrentProviderToolCall),
    Custom(CurrentProviderCustomToolCall),
    CodeModeWait(CurrentProviderToolCall),
}

#[derive(Clone, Default)]
pub(super) struct CodeModeNotificationSink {
    outputs: Arc<Mutex<Vec<CurrentProviderToolResult>>>,
}

impl CodeModeNotificationSink {
    fn push(&self, output: CurrentProviderToolResult) {
        self.outputs
            .lock()
            .expect("code mode notification sink")
            .push(output);
    }

    pub(super) fn drain(&self) -> Vec<CurrentProviderToolResult> {
        std::mem::take(&mut *self.outputs.lock().expect("code mode notification sink"))
    }
}

pub(super) struct CodeModeExecutionBatch {
    pub(super) results: Vec<CurrentProviderToolResult>,
    pub(super) notifications: Vec<CurrentProviderToolResult>,
}

#[derive(Debug, Deserialize)]
struct RuntimeCodeModeWaitArgs {
    cell_id: String,
    #[serde(default = "default_code_mode_wait_yield_time_ms")]
    yield_time_ms: u64,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    terminate: bool,
}

fn default_code_mode_wait_yield_time_ms() -> u64 {
    DEFAULT_CODE_MODE_WAIT_YIELD_TIME_MS
}

pub(super) fn advertised_tools(tools: &[RuntimeCodeModeTool]) -> Vec<CurrentProviderTool> {
    let wait = code_mode_wait_tool_definition();
    vec![
        CurrentProviderTool::custom(
            CODE_MODE_EXEC_TOOL_NAME,
            code_mode_exec_tool_description(tools),
            FreeformToolFormat {
                r#type: "grammar".to_string(),
                syntax: "lark".to_string(),
                definition: CODE_MODE_EXEC_FREEFORM_GRAMMAR.to_string(),
            },
        ),
        CurrentProviderTool::function(wait.name, wait.description, wait.input_schema),
    ]
}

pub(super) async fn execute_calls(
    tool_step_snapshot: &RuntimeToolStepSnapshot,
    turn_id: &str,
    session_id: &str,
    turn_context: Option<&agent_protocol::turn_context::TurnContextOverride>,
    working_directory: &std::path::Path,
    lifecycle_emitter: Arc<dyn ToolLifecycleEmitter>,
    notification_sink: CodeModeNotificationSink,
    calls: Vec<CurrentProviderCustomToolCall>,
    cancel_token: Option<CancellationToken>,
    allow_parallel: bool,
) -> CodeModeExecutionBatch {
    let Some(session) = tool_step_snapshot.code_mode_session().cloned() else {
        return CodeModeExecutionBatch {
            results: calls
                .into_iter()
                .map(|call| {
                    custom_provider_result(
                        call,
                        RuntimeCodeModeToolResult::failure(
                            RuntimeCodeModeCellId::new("unavailable"),
                            "custom tool call requires an executable CodeMode session",
                        ),
                    )
                })
                .collect(),
            notifications: Vec::new(),
        };
    };
    let enabled_tools = Arc::clone(&tool_step_snapshot.code_mode_tools);
    let execute = |call| {
        let nested_delegate: Arc<dyn RuntimeCodeModeSessionDelegate> =
            Arc::new(RuntimeCodeModeNestedToolDelegate {
                executor: tool_step_snapshot.executor.clone(),
                tools: Arc::clone(&enabled_tools),
                turn_id: turn_id.to_string(),
                session_id: session_id.to_string(),
                turn_context: turn_context.cloned(),
                working_directory: working_directory.to_path_buf(),
                lifecycle_emitter: lifecycle_emitter.clone(),
                notification_sink: notification_sink.clone(),
                closed: Arc::new(AtomicBool::new(false)),
            });
        execute_call(
            session.clone(),
            Arc::clone(&enabled_tools),
            nested_delegate,
            turn_id.to_string(),
            working_directory.to_path_buf(),
            lifecycle_emitter.clone(),
            cancel_token.clone(),
            call,
        )
    };

    let results = if allow_parallel && calls.len() > 1 {
        join_all(calls.into_iter().map(execute)).await
    } else {
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            results.push(execute(call).await);
        }
        results
    };
    CodeModeExecutionBatch {
        results,
        notifications: notification_sink.drain(),
    }
}

pub(super) async fn execute_wait_calls(
    tool_step_snapshot: &RuntimeToolStepSnapshot,
    turn_id: &str,
    working_directory: &std::path::Path,
    lifecycle_emitter: Arc<dyn ToolLifecycleEmitter>,
    calls: Vec<CurrentProviderToolCall>,
    cancel_token: Option<CancellationToken>,
    allow_parallel: bool,
) -> Vec<CurrentProviderToolResult> {
    let Some(session) = tool_step_snapshot.code_mode_session().cloned() else {
        return calls
            .into_iter()
            .map(|call| {
                wait_provider_result(
                    call,
                    RuntimeCodeModeToolResult::failure(
                        RuntimeCodeModeCellId::new("unavailable"),
                        "wait requires an executable CodeMode session",
                    ),
                )
            })
            .collect();
    };
    let execute = |call| {
        execute_wait_call(
            session.clone(),
            turn_id.to_string(),
            working_directory.to_path_buf(),
            lifecycle_emitter.clone(),
            cancel_token.clone(),
            call,
        )
    };

    if allow_parallel && calls.len() > 1 {
        join_all(calls.into_iter().map(execute)).await
    } else {
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            results.push(execute(call).await);
        }
        results
    }
}

async fn execute_wait_call(
    session: RuntimeCodeModeSessionHandle,
    turn_id: String,
    working_directory: std::path::PathBuf,
    lifecycle_emitter: Arc<dyn ToolLifecycleEmitter>,
    cancel_token: Option<CancellationToken>,
    call: CurrentProviderToolCall,
) -> CurrentProviderToolResult {
    let started_at = Instant::now();
    let tool_call = ToolCall::new(
        turn_id,
        call.id.clone(),
        call.name.clone(),
        call.arguments.clone(),
        vec![ToolEnvironment::new("local", working_directory)],
        lifecycle_emitter.clone(),
    );
    tool_call.emit_started().await;
    let result = match serde_json::from_value::<RuntimeCodeModeWaitArgs>(call.arguments.clone()) {
        Ok(args) => {
            let cell_id = RuntimeCodeModeCellId::new(args.cell_id);
            let max_tokens = args
                .max_tokens
                .unwrap_or(DEFAULT_CODE_MODE_MAX_OUTPUT_TOKENS);
            let outcome = if args.terminate {
                session.terminate(cell_id.clone()).await
            } else {
                let wait = session.wait(RuntimeCodeModeWaitRequest {
                    cell_id: cell_id.clone(),
                    yield_time_ms: args.yield_time_ms,
                });
                match cancel_token.as_ref() {
                    Some(cancel_token) => {
                        tokio::select! {
                            biased;
                            _ = cancel_token.cancelled() => session.terminate(cell_id.clone()).await,
                            outcome = wait => outcome,
                        }
                    }
                    None => wait.await,
                }
            };
            match outcome {
                Ok(outcome) => outcome
                    .into_response()
                    .into_tool_result_with_max_tokens(max_tokens),
                Err(error) => RuntimeCodeModeToolResult::failure(cell_id, error),
            }
        }
        Err(error) => RuntimeCodeModeToolResult::failure(
            RuntimeCodeModeCellId::new("invalid"),
            format!("failed to parse wait arguments: {error}"),
        ),
    };
    tool_call
        .emit_completed(normalized_code_mode_output(&result, started_at))
        .await;
    wait_provider_result(call, result)
}

async fn execute_call(
    session: RuntimeCodeModeSessionHandle,
    enabled_tools: Arc<Vec<RuntimeCodeModeTool>>,
    nested_delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    turn_id: String,
    working_directory: std::path::PathBuf,
    lifecycle_emitter: Arc<dyn ToolLifecycleEmitter>,
    cancel_token: Option<CancellationToken>,
    call: CurrentProviderCustomToolCall,
) -> CurrentProviderToolResult {
    let started_at = Instant::now();
    let fallback_cell_id = RuntimeCodeModeCellId::new(call.id.clone());
    let tool_call = ToolCall::new(
        turn_id,
        call.id.clone(),
        call.name.clone(),
        Value::String(call.input.clone()),
        vec![ToolEnvironment::new("local", working_directory)],
        lifecycle_emitter.clone(),
    );
    tool_call.emit_started().await;
    let parsed = match parse_code_mode_exec_source(&call.input) {
        Ok(parsed) => parsed,
        Err(error) => {
            let result = RuntimeCodeModeToolResult::failure(fallback_cell_id, error);
            tool_call
                .emit_completed(normalized_code_mode_output(&result, started_at))
                .await;
            return custom_provider_result(call, result);
        }
    };
    let max_output_tokens = parsed
        .max_output_tokens
        .unwrap_or(DEFAULT_CODE_MODE_MAX_OUTPUT_TOKENS);
    let started = match session
        .execute_with_delegate(
            RuntimeCodeModeExecuteRequest {
                tool_call_id: call.id.clone(),
                source: parsed.code,
                enabled_tools: enabled_tools.as_ref().clone(),
                yield_time_ms: parsed.yield_time_ms,
                max_output_tokens: parsed.max_output_tokens,
                cancellation_token: cancel_token.clone(),
            },
            Some(nested_delegate),
        )
        .await
    {
        Ok(started) => started,
        Err(error) => {
            let result = RuntimeCodeModeToolResult::failure(fallback_cell_id, error);
            tool_call
                .emit_completed(normalized_code_mode_output(&result, started_at))
                .await;
            return custom_provider_result(call, result);
        }
    };
    let cell_id = started.cell_id.clone();
    let response = match cancel_token.as_ref() {
        Some(cancel_token) => {
            tokio::select! {
                biased;
                _ = cancel_token.cancelled() => session
                    .terminate(cell_id.clone())
                    .await
                    .map(RuntimeCodeModeWaitOutcome::into_response),
                response = started.initial_response() => response,
            }
        }
        None => started.initial_response().await,
    };
    let result = match response {
        Ok(response) => response.into_tool_result_with_max_tokens(max_output_tokens),
        Err(error) => RuntimeCodeModeToolResult::failure(cell_id, error),
    };
    tool_call
        .emit_completed(normalized_code_mode_output(&result, started_at))
        .await;
    custom_provider_result(call, result)
}

fn normalized_code_mode_output(
    result: &RuntimeCodeModeToolResult,
    duration: Instant,
) -> NormalizedToolOutput {
    let mut metadata = HashMap::from([
        (
            "code_mode_cell_id".to_string(),
            Value::String(result.cell_id.as_str().to_string()),
        ),
        ("code_mode".to_string(), Value::Bool(true)),
    ]);
    metadata.insert(
        "code_mode_output_status".to_string(),
        Value::String(if result.success { "success" } else { "failure" }.to_string()),
    );
    metadata.insert(
        tool_runtime::tool_result_projection::TOOL_HANDLER_EXECUTED_METADATA_KEY.to_string(),
        Value::Bool(true),
    );
    NormalizedToolOutput {
        success: result.success,
        text: result.output.clone(),
        structured_content: None,
        error: result.error.clone(),
        duration_ms: u64::try_from(duration.elapsed().as_millis()).unwrap_or(u64::MAX),
        truncation: None,
        sidecar_reference: None,
        metadata,
        agent_control_projection_facts: Vec::new(),
        agent_control_state_facts: Vec::new(),
    }
}

struct RuntimeCodeModeNestedToolDelegate {
    executor: RuntimeToolExecutorHandle,
    tools: Arc<Vec<RuntimeCodeModeTool>>,
    turn_id: String,
    session_id: String,
    turn_context: Option<agent_protocol::turn_context::TurnContextOverride>,
    working_directory: std::path::PathBuf,
    lifecycle_emitter: Arc<dyn ToolLifecycleEmitter>,
    notification_sink: CodeModeNotificationSink,
    closed: Arc<AtomicBool>,
}

impl RuntimeCodeModeSessionDelegate for RuntimeCodeModeNestedToolDelegate {
    fn invoke_tool<'a>(
        &'a self,
        invocation: RuntimeCodeModeNestedToolCall,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, serde_json::Value> {
        Box::pin(async move {
            if self.closed.load(Ordering::Acquire) {
                return Err(format!(
                    "code mode cell {} is already closed",
                    invocation.cell_id
                ));
            }
            if matches!(
                invocation.tool_name.as_str(),
                CODE_MODE_EXEC_TOOL_NAME | tool_runtime::code_mode::CODE_MODE_WAIT_TOOL_NAME
            ) {
                return Err("CodeMode cannot invoke its control tools as nested tools".to_string());
            }
            let tool = self
                .tools
                .iter()
                .find(|tool| tool.global_name == invocation.tool_name)
                .ok_or_else(|| {
                    format!(
                        "nested tool '{}' was not enabled for this sampling step",
                        invocation.tool_name
                    )
                })?;
            let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
                working_directory: self.working_directory.clone(),
                session_id: self.session_id.clone(),
                cancel_token: Some(cancellation_token),
                workspace_sandbox: None,
            });
            let call = ToolCall::new(
                self.turn_id.clone(),
                format!("code-mode-{}", invocation.runtime_tool_call_id),
                tool.definition.name.clone(),
                invocation.input.unwrap_or(serde_json::Value::Null),
                vec![ToolEnvironment::new(
                    "local",
                    self.working_directory.clone(),
                )],
                self.lifecycle_emitter.clone(),
            );
            let runtime_tool = self
                .executor
                .clone()
                .bind(tool.definition.clone(), RuntimeToolExposure::Direct);
            let output = runtime_tool
                .execute_call(&call, &context, self.turn_context.as_ref())
                .await;
            if self.closed.load(Ordering::Acquire) {
                return Err(format!(
                    "code mode cell {} is already closed",
                    invocation.cell_id
                ));
            }
            if output.success {
                Ok(output
                    .structured_content
                    .unwrap_or_else(|| serde_json::Value::String(output.text)))
            } else {
                Err(output.error.unwrap_or(output.text))
            }
        })
    }

    fn notify<'a>(
        &'a self,
        tool_call_id: String,
        cell_id: RuntimeCodeModeCellId,
        text: String,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, ()> {
        Box::pin(async move {
            if cancellation_token.is_cancelled() || self.closed.load(Ordering::Acquire) {
                return Err("code mode notification cancelled".to_string());
            }
            let text = text.trim().to_string();
            if text.is_empty() {
                return Ok(());
            }
            let provider_output = CurrentProviderToolResult {
                call_id: tool_call_id.clone(),
                name: CODE_MODE_EXEC_TOOL_NAME.to_string(),
                success: true,
                output: text.clone(),
                error: None,
            };
            let metadata = HashMap::from([
                (
                    "code_mode_cell_id".to_string(),
                    Value::String(cell_id.as_str().to_string()),
                ),
                (
                    "notification_kind".to_string(),
                    Value::String("code_mode_notify".to_string()),
                ),
            ]);
            self.lifecycle_emitter
                .emit_output_delta(ToolOutputDeltaEvent {
                    turn_id: self.turn_id.clone(),
                    call_id: tool_call_id,
                    tool_name: CODE_MODE_EXEC_TOOL_NAME.to_string(),
                    delta: text,
                    output_kind: Some("code_mode_notify".to_string()),
                    metadata,
                })
                .await;
            if cancellation_token.is_cancelled() || self.closed.load(Ordering::Acquire) {
                return Err("code mode notification cancelled".to_string());
            }
            self.notification_sink.push(provider_output);
            Ok(())
        })
    }

    fn cell_closed(&self, _cell_id: &RuntimeCodeModeCellId) {
        self.closed.store(true, Ordering::Release);
    }
}

fn custom_provider_result(
    call: CurrentProviderCustomToolCall,
    result: RuntimeCodeModeToolResult,
) -> CurrentProviderToolResult {
    CurrentProviderToolResult {
        call_id: call.id,
        name: call.name,
        success: result.success,
        output: result.output,
        error: result.error,
    }
}

fn wait_provider_result(
    call: CurrentProviderToolCall,
    result: RuntimeCodeModeToolResult,
) -> CurrentProviderToolResult {
    CurrentProviderToolResult {
        call_id: call.id,
        name: call.name,
        success: result.success,
        output: result.output,
        error: result.error,
    }
}
