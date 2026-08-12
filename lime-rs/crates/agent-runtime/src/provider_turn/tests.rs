use super::*;
use crate::reply_input::RuntimeReplyInput;
use crate::session_loop::{RuntimeSessionClosureTask, RuntimeSessionInput, RuntimeSessionRegistry};
use agent_protocol::provider_trace::ProviderTraceStage;
use agent_protocol::world_state::{
    RuntimeWorldEnvironment, RuntimeWorldMode, RuntimeWorldPermissions, RuntimeWorldState,
    WORLD_STATE_TURN_METADATA_KEY,
};
use agent_protocol::MultiAgentMode;
use futures::future::BoxFuture;
use futures::{stream, StreamExt};
use model_provider::current_client::CurrentProviderRole;
use model_provider::current_client::FinishReason;
use model_provider::current_client::{CurrentProviderError, CurrentProviderStream};
use model_provider::provider_stream::RuntimeReplyProviderTraceMetadata;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use tokio::sync::oneshot;
use tool_runtime::code_mode::{
    RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest, RuntimeCodeModeFuture,
    RuntimeCodeModeNestedToolCall, RuntimeCodeModeResponse, RuntimeCodeModeSession,
    RuntimeCodeModeSessionDelegate, RuntimeCodeModeSessionHandle, RuntimeCodeModeStartedCell,
    RuntimeCodeModeWaitOutcome, RuntimeCodeModeWaitRequest,
};
use tool_runtime::tool_executor::{
    RuntimeToolExecutionFuture, RuntimeToolExecutionRequest, RuntimeToolExecutionResult,
    RuntimeToolExecutor,
};
use tool_runtime::tool_lifecycle::{
    ToolLifecycleEmissionFuture, ToolLifecycleEvent, ToolLifecyclePhase, ToolOutputDeltaEvent,
};

#[test]
fn harness_generation_projects_provider_request_controls() {
    let mut turn_context = agent_protocol::turn_context::TurnContextOverride::default();
    turn_context.metadata.insert(
        "runtime_request".to_string(),
        serde_json::json!({
            "harness": {
                "generation": {
                    "max_output_tokens": 128,
                    "enable_thinking": false
                }
            }
        }),
    );
    let config = crate::session_config::SessionConfigBuilder::new("session-1")
        .turn_context(turn_context)
        .build();

    let (generation, provider_options) = provider_request_controls(&config);

    assert_eq!(generation.max_tokens, Some(128));
    assert_eq!(provider_options.get("enable_thinking"), Some(&false.into()));
}

#[test]
fn app_server_thinking_control_projects_provider_request_option() {
    let mut turn_context = agent_protocol::turn_context::TurnContextOverride::default();
    turn_context.metadata.insert(
        "app_server_runtime_backend".to_string(),
        serde_json::json!({ "thinkingEnabled": false }),
    );
    let config = crate::session_config::SessionConfigBuilder::new("session-1")
        .turn_context(turn_context)
        .build();

    let (generation, provider_options) = provider_request_controls(&config);

    assert_eq!(generation.max_tokens, None);
    assert_eq!(provider_options.get("enable_thinking"), Some(&false.into()));
}

#[test]
fn provider_failure_trace_preserves_auth_rate_limit_and_server_categories() {
    let cases = [
        (
            Some(FailureClassification::Authentication),
            false,
            "auth",
            true,
        ),
        (
            Some(FailureClassification::RateLimit),
            true,
            "rate_limit",
            false,
        ),
        (
            Some(FailureClassification::ProviderInternal),
            true,
            "server",
            false,
        ),
    ];

    for (classification, retryable, category, non_retryable_rejection) in cases {
        assert_eq!(
            provider_trace_failure(classification, retryable),
            ProviderTraceFailure::new(category, retryable, non_retryable_rejection)
        );
    }
}

#[derive(Clone)]
struct ScriptedProvider {
    streams: Arc<Mutex<VecDeque<Vec<Result<CanonicalLlmEvent, CurrentProviderError>>>>>,
    requests: Arc<Mutex<Vec<CurrentProviderRequest>>>,
}

impl ScriptedProvider {
    fn new(streams: Vec<Vec<Result<CanonicalLlmEvent, CurrentProviderError>>>) -> Self {
        Self {
            streams: Arc::new(Mutex::new(VecDeque::from(streams))),
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl CurrentProvider for ScriptedProvider {
    fn stream<'a>(
        &'a self,
        request: CurrentProviderRequest,
    ) -> BoxFuture<'a, Result<CurrentProviderStream, CurrentProviderError>> {
        self.requests.lock().expect("record request").push(request);
        let stream = self
            .streams
            .lock()
            .expect("take stream")
            .pop_front()
            .unwrap_or_else(|| {
                vec![Ok(CanonicalLlmEvent::Finish {
                    reason: FinishReason::Stop,
                    usage: None,
                    response_id: None,
                })]
            });
        Box::pin(async move {
            let stream: CurrentProviderStream = Box::pin(stream::iter(stream));
            Ok(stream)
        })
    }
}

struct GatedEmptyThenRetryableErrorProvider {
    attempt: AtomicUsize,
    started: Mutex<Option<oneshot::Sender<()>>>,
    continue_after_steer: Mutex<Option<oneshot::Receiver<()>>>,
    requests: Mutex<Vec<CurrentProviderRequest>>,
}

impl CurrentProvider for GatedEmptyThenRetryableErrorProvider {
    fn stream<'a>(
        &'a self,
        request: CurrentProviderRequest,
    ) -> BoxFuture<'a, Result<CurrentProviderStream, CurrentProviderError>> {
        self.requests.lock().expect("record request").push(request);
        if self.attempt.fetch_add(1, Ordering::SeqCst) == 0 {
            if let Some(started) = self.started.lock().expect("started sender").take() {
                let _ = started.send(());
            }
            let continue_after_steer = self
                .continue_after_steer
                .lock()
                .expect("continue receiver")
                .take()
                .expect("first request continuation");
            return Box::pin(async move {
                let stream: CurrentProviderStream = Box::pin(stream::once(async move {
                    let _ = continue_after_steer.await;
                    Ok(CanonicalLlmEvent::Finish {
                        reason: FinishReason::Stop,
                        usage: None,
                        response_id: Some("empty-before-steer".to_string()),
                    })
                }));
                Ok(stream)
            });
        }

        Box::pin(async move {
            let stream: CurrentProviderStream =
                Box::pin(stream::iter([Ok(CanonicalLlmEvent::ProviderError {
                    message: "provider failed after steer".to_string(),
                    classification: Some(FailureClassification::Transport),
                    retryable: Some(true),
                })]));
            Ok(stream)
        })
    }
}

struct HangingRequestProvider {
    started: Mutex<Option<oneshot::Sender<()>>>,
}

impl CurrentProvider for HangingRequestProvider {
    fn stream<'a>(
        &'a self,
        _request: CurrentProviderRequest,
    ) -> BoxFuture<'a, Result<CurrentProviderStream, CurrentProviderError>> {
        Box::pin(async move {
            if let Some(sender) = self.started.lock().expect("provider started lock").take() {
                let _ = sender.send(());
            }
            std::future::pending::<Result<CurrentProviderStream, CurrentProviderError>>().await
        })
    }
}

struct HangingFirstEventProvider {
    stream_started: Mutex<Option<oneshot::Sender<()>>>,
}

impl CurrentProvider for HangingFirstEventProvider {
    fn stream<'a>(
        &'a self,
        _request: CurrentProviderRequest,
    ) -> BoxFuture<'a, Result<CurrentProviderStream, CurrentProviderError>> {
        Box::pin(async move {
            if let Some(sender) = self
                .stream_started
                .lock()
                .expect("stream started lock")
                .take()
            {
                let _ = sender.send(());
            }
            let stream: CurrentProviderStream = Box::pin(stream::pending());
            Ok(stream)
        })
    }
}

struct ToolCallThenHangingProvider {
    attempt: AtomicUsize,
    requests: Mutex<Vec<CurrentProviderRequest>>,
}

impl CurrentProvider for ToolCallThenHangingProvider {
    fn stream<'a>(
        &'a self,
        request: CurrentProviderRequest,
    ) -> BoxFuture<'a, Result<CurrentProviderStream, CurrentProviderError>> {
        self.requests.lock().expect("record request").push(request);
        if self.attempt.fetch_add(1, Ordering::SeqCst) == 0 {
            return Box::pin(async move {
                let stream: CurrentProviderStream = Box::pin(stream::iter([
                    Ok(CanonicalLlmEvent::ToolCall {
                        id: "call-timeout".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({
                            "file_path": " README.md ",
                            "start_line": "2"
                        }),
                        raw_arguments: Some(
                            r#"{"file_path":" README.md ","start_line":"2"}"#.to_string(),
                        ),
                        provider_executed: None,
                        provider_metadata: Default::default(),
                    }),
                    Ok(CanonicalLlmEvent::Finish {
                        reason: FinishReason::ToolCall,
                        usage: None,
                        response_id: Some("response-tool".to_string()),
                    }),
                ]));
                Ok(stream)
            });
        }

        Box::pin(async move {
            let stream: CurrentProviderStream = Box::pin(stream::pending());
            Ok(stream)
        })
    }
}

struct ReasoningHeartbeatProvider;

impl CurrentProvider for ReasoningHeartbeatProvider {
    fn stream<'a>(
        &'a self,
        _request: CurrentProviderRequest,
    ) -> BoxFuture<'a, Result<CurrentProviderStream, CurrentProviderError>> {
        Box::pin(async move {
            let stream: CurrentProviderStream =
                Box::pin(stream::unfold(0_u64, |sequence| async move {
                    if sequence > 0 {
                        tokio::task::yield_now().await;
                    }
                    Some((
                        Ok(CanonicalLlmEvent::ReasoningContentDelta {
                            id: "reasoning-0".to_string(),
                            text: format!("heartbeat-{sequence}"),
                            content_index: 0,
                        }),
                        sequence + 1,
                    ))
                }));
            Ok(stream)
        })
    }
}

struct TextHeartbeatProvider;

impl CurrentProvider for TextHeartbeatProvider {
    fn stream<'a>(
        &'a self,
        _request: CurrentProviderRequest,
    ) -> BoxFuture<'a, Result<CurrentProviderStream, CurrentProviderError>> {
        Box::pin(async move {
            let stream: CurrentProviderStream =
                Box::pin(stream::unfold(0_u64, |sequence| async move {
                    if sequence > 0 {
                        tokio::task::yield_now().await;
                    }
                    Some((
                        Ok(CanonicalLlmEvent::TextDelta {
                            id: "text-0".to_string(),
                            text: format!("heartbeat-{sequence}"),
                        }),
                        sequence + 1,
                    ))
                }));
            Ok(stream)
        })
    }
}

struct CancelOnFirstUsageProvider {
    cancel_token: CancellationToken,
}

impl CurrentProvider for CancelOnFirstUsageProvider {
    fn stream<'a>(
        &'a self,
        _request: CurrentProviderRequest,
    ) -> BoxFuture<'a, Result<CurrentProviderStream, CurrentProviderError>> {
        let cancel_token = self.cancel_token.clone();
        Box::pin(async move {
            let stream: CurrentProviderStream = Box::pin(stream::once(async move {
                cancel_token.cancel();
                Ok(CanonicalLlmEvent::Usage {
                    usage: Usage {
                        input_tokens: Some(17),
                        output_tokens: Some(5),
                        cache_write_input_tokens: Some(6),
                        ..Usage::default()
                    },
                })
            }));
            Ok(stream)
        })
    }
}

struct UsageThenProviderError;

impl CurrentProvider for UsageThenProviderError {
    fn stream<'a>(
        &'a self,
        _request: CurrentProviderRequest,
    ) -> BoxFuture<'a, Result<CurrentProviderStream, CurrentProviderError>> {
        Box::pin(async move {
            let stream: CurrentProviderStream = Box::pin(stream::iter(vec![
                Ok(CanonicalLlmEvent::Usage {
                    usage: Usage {
                        input_tokens: Some(19),
                        output_tokens: Some(7),
                        cache_write_input_tokens: Some(8),
                        ..Usage::default()
                    },
                }),
                Ok(CanonicalLlmEvent::ProviderError {
                    message: "provider stopped after usage".to_string(),
                    classification: None,
                    retryable: Some(false),
                }),
            ]));
            Ok(stream)
        })
    }
}

struct UsageThenHangingStream;

impl CurrentProvider for UsageThenHangingStream {
    fn stream<'a>(
        &'a self,
        _request: CurrentProviderRequest,
    ) -> BoxFuture<'a, Result<CurrentProviderStream, CurrentProviderError>> {
        Box::pin(async move {
            let stream: CurrentProviderStream = Box::pin(
                stream::iter(vec![Ok(CanonicalLlmEvent::Usage {
                    usage: Usage {
                        input_tokens: Some(23),
                        output_tokens: Some(11),
                        cache_write_input_tokens: Some(13),
                        ..Usage::default()
                    },
                })])
                .chain(stream::pending()),
            );
            Ok(stream)
        })
    }
}

struct UsageThenStreamError;

impl CurrentProvider for UsageThenStreamError {
    fn stream<'a>(
        &'a self,
        _request: CurrentProviderRequest,
    ) -> BoxFuture<'a, Result<CurrentProviderStream, CurrentProviderError>> {
        Box::pin(async move {
            let stream: CurrentProviderStream = Box::pin(stream::iter(vec![
                Ok(CanonicalLlmEvent::Usage {
                    usage: Usage {
                        input_tokens: Some(29),
                        output_tokens: Some(17),
                        cache_write_input_tokens: Some(21),
                        ..Usage::default()
                    },
                }),
                Err(CurrentProviderError::new(
                    "provider stream failed after usage",
                )),
            ]));
            Ok(stream)
        })
    }
}

// The production client owns HTTP. This fake only documents turn-loop behavior below.
struct EchoTool;

impl RuntimeToolExecutor for EchoTool {
    fn execute<'a>(
        &'a self,
        request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move {
            Ok(RuntimeToolExecutionResult::new(
                true,
                format!("executed {}", request.tool_name),
                None,
                Default::default(),
            ))
        })
    }
}

struct TaggedTool(&'static str);

impl RuntimeToolExecutor for TaggedTool {
    fn execute<'a>(
        &'a self,
        _request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move {
            Ok(RuntimeToolExecutionResult::new(
                true,
                self.0.to_string(),
                None,
                Default::default(),
            ))
        })
    }
}

#[derive(Default)]
struct CountingTool {
    calls: AtomicUsize,
}

impl RuntimeToolExecutor for CountingTool {
    fn execute<'a>(
        &'a self,
        request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(RuntimeToolExecutionResult::new(
                true,
                format!("executed {}", request.tool_name),
                None,
                Default::default(),
            ))
        })
    }
}

struct RecordingCodeModeSession {
    requests: Mutex<Vec<(String, String, usize, bool)>>,
    responses: Mutex<VecDeque<Result<RuntimeCodeModeResponse, String>>>,
    wait_requests: Mutex<Vec<(String, u64)>>,
    wait_responses: Mutex<VecDeque<Result<RuntimeCodeModeWaitOutcome, String>>>,
    terminations: Mutex<Vec<String>>,
}

impl RecordingCodeModeSession {
    fn new(responses: Vec<Result<RuntimeCodeModeResponse, String>>) -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            responses: Mutex::new(VecDeque::from(responses)),
            wait_requests: Mutex::new(Vec::new()),
            wait_responses: Mutex::new(VecDeque::new()),
            terminations: Mutex::new(Vec::new()),
        }
    }

    fn with_wait_responses(
        mut self,
        responses: Vec<Result<RuntimeCodeModeWaitOutcome, String>>,
    ) -> Self {
        self.wait_responses = Mutex::new(VecDeque::from(responses));
        self
    }

    fn requests(&self) -> Vec<(String, String, usize, bool)> {
        self.requests.lock().expect("code mode requests").clone()
    }

    fn terminations(&self) -> Vec<String> {
        self.terminations
            .lock()
            .expect("code mode terminations")
            .clone()
    }

    fn wait_requests(&self) -> Vec<(String, u64)> {
        self.wait_requests
            .lock()
            .expect("code mode wait requests")
            .clone()
    }
}

impl RuntimeCodeModeSession for RecordingCodeModeSession {
    fn execute<'a>(
        &'a self,
        request: RuntimeCodeModeExecuteRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeStartedCell> {
        Box::pin(async move {
            self.requests
                .lock()
                .expect("record code mode request")
                .push((
                    request.tool_call_id,
                    request.source,
                    request.enabled_tools.len(),
                    request.cancellation_token.is_some(),
                ));
            let response = self
                .responses
                .lock()
                .expect("take code mode response")
                .pop_front()
                .unwrap_or_else(|| Err("missing code mode response".to_string()))?;
            let cell_id = response.cell_id().clone();
            Ok(RuntimeCodeModeStartedCell::new(
                cell_id,
                Box::pin(async move { Ok(response) }),
            ))
        })
    }

    fn wait<'a>(
        &'a self,
        request: RuntimeCodeModeWaitRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            self.wait_requests
                .lock()
                .expect("record code mode wait")
                .push((request.cell_id.to_string(), request.yield_time_ms));
            self.wait_responses
                .lock()
                .expect("take code mode wait response")
                .pop_front()
                .unwrap_or_else(|| Err("missing code mode wait response".to_string()))
        })
    }

    fn terminate<'a>(
        &'a self,
        cell_id: RuntimeCodeModeCellId,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            self.terminations
                .lock()
                .expect("record code mode terminate")
                .push(cell_id.to_string());
            Ok(RuntimeCodeModeWaitOutcome::LiveCell(
                RuntimeCodeModeResponse::Terminated {
                    cell_id,
                    output: String::new(),
                },
            ))
        })
    }

    fn shutdown(&self) -> RuntimeCodeModeFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}

struct NestedDispatchCodeModeSession;

impl RuntimeCodeModeSession for NestedDispatchCodeModeSession {
    fn execute<'a>(
        &'a self,
        _request: RuntimeCodeModeExecuteRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeStartedCell> {
        Box::pin(async {
            let cell_id = RuntimeCodeModeCellId::new("cell-nested-provider-turn");
            Ok(RuntimeCodeModeStartedCell::new(
                cell_id.clone(),
                Box::pin(async move {
                    Ok(RuntimeCodeModeResponse::Result {
                        cell_id,
                        output: "unused".to_string(),
                        error_text: None,
                    })
                }),
            ))
        })
    }

    fn execute_with_delegate<'a>(
        &'a self,
        request: RuntimeCodeModeExecuteRequest,
        delegate: Option<Arc<dyn RuntimeCodeModeSessionDelegate>>,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeStartedCell> {
        Box::pin(async move {
            let cell_id = RuntimeCodeModeCellId::new("cell-nested-provider-turn");
            let delegate = delegate.expect("provider turn must bind nested delegate");
            let tool_call_id = request.tool_call_id;
            Ok(RuntimeCodeModeStartedCell::new(
                cell_id.clone(),
                Box::pin(async move {
                    delegate
                        .notify(
                            tool_call_id,
                            cell_id.clone(),
                            "nested tool starting".to_string(),
                            tokio_util::sync::CancellationToken::new(),
                        )
                        .await?;
                    let nested = delegate
                        .invoke_tool(
                            RuntimeCodeModeNestedToolCall {
                                cell_id: cell_id.clone(),
                                runtime_tool_call_id: "nested-read-1".to_string(),
                                tool_name: "read".to_string(),
                                input: Some(serde_json::json!({"path": "README.md"})),
                            },
                            tokio_util::sync::CancellationToken::new(),
                        )
                        .await?;
                    Ok(RuntimeCodeModeResponse::Result {
                        cell_id,
                        output: nested.to_string(),
                        error_text: None,
                    })
                }),
            ))
        })
    }

    fn wait<'a>(
        &'a self,
        request: RuntimeCodeModeWaitRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            Ok(RuntimeCodeModeWaitOutcome::MissingCell(
                RuntimeCodeModeResponse::Result {
                    cell_id: request.cell_id,
                    output: String::new(),
                    error_text: Some("cell not found".to_string()),
                },
            ))
        })
    }

    fn terminate<'a>(
        &'a self,
        cell_id: RuntimeCodeModeCellId,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            Ok(RuntimeCodeModeWaitOutcome::MissingCell(
                RuntimeCodeModeResponse::Result {
                    cell_id,
                    output: String::new(),
                    error_text: Some("cell not found".to_string()),
                },
            ))
        })
    }

    fn shutdown(&self) -> RuntimeCodeModeFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}

struct HangingCodeModeSession {
    started: Mutex<Option<oneshot::Sender<()>>>,
    terminations: Mutex<Vec<String>>,
}

impl RuntimeCodeModeSession for HangingCodeModeSession {
    fn execute<'a>(
        &'a self,
        _request: RuntimeCodeModeExecuteRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeStartedCell> {
        Box::pin(async move {
            if let Some(started) = self.started.lock().expect("code mode started").take() {
                let _ = started.send(());
            }
            Ok(RuntimeCodeModeStartedCell::new(
                RuntimeCodeModeCellId::new("cell-hanging"),
                Box::pin(std::future::pending()),
            ))
        })
    }

    fn wait<'a>(
        &'a self,
        _request: RuntimeCodeModeWaitRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async { Err("unexpected code mode wait".to_string()) })
    }

    fn terminate<'a>(
        &'a self,
        cell_id: RuntimeCodeModeCellId,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            self.terminations
                .lock()
                .expect("record hanging termination")
                .push(cell_id.to_string());
            Ok(RuntimeCodeModeWaitOutcome::LiveCell(
                RuntimeCodeModeResponse::Terminated {
                    cell_id,
                    output: String::new(),
                },
            ))
        })
    }

    fn shutdown(&self) -> RuntimeCodeModeFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}

struct LateCompletingTool {
    calls: AtomicUsize,
    started: Mutex<Option<oneshot::Sender<()>>>,
    release_after_cancel: Mutex<Option<oneshot::Receiver<()>>>,
    late_completed: Mutex<Option<oneshot::Sender<()>>>,
}

impl RuntimeToolExecutor for LateCompletingTool {
    fn execute<'a>(
        &'a self,
        request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let release_after_cancel = self
                .release_after_cancel
                .lock()
                .expect("late completion release")
                .take()
                .expect("late completion release receiver");
            let late_completed = self
                .late_completed
                .lock()
                .expect("late completion sender")
                .take()
                .expect("late completion sender");
            let (result_sender, result_receiver) = oneshot::channel();
            if let Some(started) = self.started.lock().expect("tool started sender").take() {
                let _ = started.send(());
            }
            let tool_name = request.tool_name.to_string();
            tokio::spawn(async move {
                let _ = release_after_cancel.await;
                let result = RuntimeToolExecutionResult::new(
                    true,
                    format!("late success from {tool_name}"),
                    None,
                    Default::default(),
                );
                let _ = result_sender.send(result);
                let _ = late_completed.send(());
            });

            result_receiver.await.map_err(|_| {
                RuntimeToolExecutionError::new("late tool result channel closed", None)
            })
        })
    }
}

struct SequencedToolStepSnapshotSource {
    snapshots: Mutex<VecDeque<RuntimeToolStepSnapshot>>,
}

impl RuntimeToolStepSnapshotSource for SequencedToolStepSnapshotSource {
    fn capture(&self) -> RuntimeToolStepSnapshotFuture<'_> {
        Box::pin(async move {
            self.snapshots
                .lock()
                .expect("take tool step snapshot")
                .pop_front()
                .ok_or_else(|| "missing tool step snapshot".to_string())
        })
    }
}

#[derive(Default)]
struct ParallelProbe {
    active: AtomicUsize,
    max_active: AtomicUsize,
}

impl RuntimeToolExecutor for ParallelProbe {
    fn execute<'a>(
        &'a self,
        _request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(RuntimeToolExecutionResult::new(
                true,
                "done".to_string(),
                None,
                Default::default(),
            ))
        })
    }
}

#[derive(Default)]
struct RecordingLifecycleEmitter {
    events: Mutex<Vec<ToolLifecycleEvent>>,
    output_deltas: Mutex<Vec<ToolOutputDeltaEvent>>,
}

impl RecordingLifecycleEmitter {
    fn events(&self) -> Vec<ToolLifecycleEvent> {
        self.events.lock().expect("lifecycle events").clone()
    }

    fn output_deltas(&self) -> Vec<ToolOutputDeltaEvent> {
        self.output_deltas.lock().expect("output deltas").clone()
    }
}

impl ToolLifecycleEmitter for RecordingLifecycleEmitter {
    fn emit<'a>(&'a self, event: ToolLifecycleEvent) -> ToolLifecycleEmissionFuture<'a> {
        Box::pin(async move {
            self.events
                .lock()
                .expect("record lifecycle event")
                .push(event);
        })
    }

    fn emit_output_delta<'a>(
        &'a self,
        event: ToolOutputDeltaEvent,
    ) -> ToolLifecycleEmissionFuture<'a> {
        Box::pin(async move {
            self.output_deltas
                .lock()
                .expect("record output delta")
                .push(event);
        })
    }
}

#[test]
fn provider_output_item_id_is_turn_and_attempt_scoped() {
    let first_turn = provider_output_item_id("turn-1", 1, ProviderOutputFamily::Text, "text-0");
    let second_turn = provider_output_item_id("turn-2", 1, ProviderOutputFamily::Text, "text-0");
    let second_attempt = provider_output_item_id("turn-1", 2, ProviderOutputFamily::Text, "text-0");

    assert_eq!(first_turn, "provider:turn-1:1:text:text-0");
    assert_ne!(first_turn, second_turn);
    assert_ne!(first_turn, second_attempt);
}

#[tokio::test]
async fn provider_request_uses_typed_cwd_world_state_without_app_server_snapshot() {
    let provider = Arc::new(ScriptedProvider::new(vec![vec![
        Ok(CanonicalLlmEvent::TextDelta {
            id: "text-0".to_string(),
            text: "done".to_string(),
        }),
        Ok(CanonicalLlmEvent::Finish {
            reason: FinishReason::Stop,
            usage: None,
            response_id: Some("response-1".to_string()),
        }),
    ]]));
    let requests = Arc::clone(&provider.requests);

    run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("inspect the workspace".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("/tmp/task<&>"),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("provider turn");

    let requests = requests.lock().expect("recorded requests");
    assert_eq!(requests.len(), 1);
    assert!(matches!(
        requests[0].messages.as_slice(),
        [
            CurrentProviderMessage {
                role: CurrentProviderRole::User,
                content: environment_content,
            },
            CurrentProviderMessage {
                role: CurrentProviderRole::User,
                content: user_content,
            }
        ] if matches!(environment_content.as_slice(), [CurrentProviderContent::Text(text)]
            if text == "<environment_context>\n  <cwd>/tmp/task&lt;&amp;&gt;</cwd>\n</environment_context>")
            && matches!(user_content.as_slice(), [CurrentProviderContent::Text(text)]
                if text == "inspect the workspace")
    ));
}

#[tokio::test]
async fn provider_request_projects_typed_world_state_once_before_current_user() {
    let provider = Arc::new(ScriptedProvider::new(vec![vec![
        Ok(CanonicalLlmEvent::TextDelta {
            id: "text-0".to_string(),
            text: "done".to_string(),
        }),
        Ok(CanonicalLlmEvent::Finish {
            reason: FinishReason::Stop,
            usage: None,
            response_id: Some("response-1".to_string()),
        }),
    ]]));
    let requests = Arc::clone(&provider.requests);
    let world_state = RuntimeWorldState {
        environment: Some(RuntimeWorldEnvironment {
            cwd: Some("/tmp/repo & app".to_string()),
            project_root: Some("/tmp/repo".to_string()),
            workspace_id: Some("workspace-1".to_string()),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            provider: Some("anthropic".to_string()),
            model: Some("claude <sonnet>".to_string()),
            reasoning_effort: Some("high".to_string()),
        }),
        permissions: Some(RuntimeWorldPermissions {
            approval_policy: Some("on-request".to_string()),
            sandbox_policy: Some("workspace-write".to_string()),
            web_search: Some(false),
        }),
        collaboration: Some(RuntimeWorldMode {
            mode: "default".to_string(),
            source: Some("request & config".to_string()),
        }),
        multi_agent: Some(MultiAgentMode::ExplicitRequestOnly),
        instruction_sections: Vec::new(),
        source: Some("app_server_world_state".to_string()),
    };
    let mut turn_context = agent_protocol::turn_context::TurnContextOverride::default();
    turn_context.metadata.insert(
        WORLD_STATE_TURN_METADATA_KEY.to_string(),
        serde_json::to_value(world_state).expect("serialize world state"),
    );

    run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .thread_id("thread-1")
                .turn_id("turn-1")
                .turn_context(turn_context)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("inspect the workspace".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("/tmp/ignored-fallback"),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("provider turn");

    let requests = requests.lock().expect("recorded requests");
    assert_eq!(requests.len(), 1);
    let messages = &requests[0].messages;
    assert_eq!(messages.len(), 2);
    let CurrentProviderContent::Text(environment_context) = &messages[0].content[0] else {
        panic!("world state must be provider-visible text");
    };
    assert_eq!(
        environment_context,
        "<environment_context>\n  <cwd>/tmp/repo &amp; app</cwd>\n  <project_root>/tmp/repo</project_root>\n  <workspace_id>workspace-1</workspace_id>\n  <thread_id>thread-1</thread_id>\n  <turn_id>turn-1</turn_id>\n  <model provider=\"anthropic\" name=\"claude &lt;sonnet&gt;\" reasoning_effort=\"high\" />\n  <permissions approval_policy=\"on-request\" sandbox_policy=\"workspace-write\" web_search=\"disabled\" />\n  <collaboration mode=\"default\" source=\"request &amp; config\" />\n  <multi_agent_mode>Any earlier instruction enabling proactive multi-agent delegation no longer applies. Do not spawn sub-agents unless the user or applicable AGENTS.md/skill instructions explicitly ask for sub-agents, delegation, or parallel agent work.</multi_agent_mode>\n</environment_context>"
    );
    assert_eq!(
        environment_context.matches("<environment_context>").count(),
        1
    );
    assert_eq!(environment_context.matches("<multi_agent_mode>").count(), 1);
    assert!(!environment_context.contains("<instructions"));
    assert!(matches!(
        messages[1].content.as_slice(),
        [CurrentProviderContent::Text(text)] if text == "inspect the workspace"
    ));
}

#[test]
fn invalid_world_state_metadata_is_not_hidden_by_cwd_fallback() {
    let mut turn_context = agent_protocol::turn_context::TurnContextOverride::default();
    turn_context.metadata.insert(
        WORLD_STATE_TURN_METADATA_KEY.to_string(),
        serde_json::json!({ "environment": "invalid" }),
    );
    let config = crate::session_config::SessionConfigBuilder::new("session-1")
        .turn_context(turn_context)
        .build();

    let error = resolve_world_state(&config, Path::new("/tmp/fallback"))
        .expect_err("invalid snapshot must fail closed");

    assert!(error.message.contains("Invalid world_state turn metadata"));
}

#[tokio::test]
async fn provider_request_preserves_canonical_fork_lineage() {
    let provider = Arc::new(ScriptedProvider::new(vec![vec![
        Ok(CanonicalLlmEvent::TextDelta {
            id: "text-0".to_string(),
            text: "done".to_string(),
        }),
        Ok(CanonicalLlmEvent::Finish {
            reason: FinishReason::Stop,
            usage: None,
            response_id: Some("response-1".to_string()),
        }),
    ]]));
    let requests = Arc::clone(&provider.requests);

    run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .thread_id("thread-1")
                .turn_id("turn-1")
                .forked_from_thread_id("thread-source")
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("continue".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("/tmp"),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("provider turn");

    let requests = requests.lock().expect("recorded requests");
    let metadata = requests[0].metadata.as_ref().expect("request metadata");
    assert_eq!(metadata.session_id, "session-1");
    assert_eq!(metadata.thread_id, "thread-1");
    assert_eq!(metadata.turn_id, "turn-1");
    assert_eq!(
        metadata.forked_from_thread_id.as_deref(),
        Some("thread-source")
    );
}

#[tokio::test]
async fn provider_metadata_is_deduplicated_across_sampling_steps() {
    let metadata = || {
        vec![
            Ok(CanonicalLlmEvent::ServerModel {
                model: "gpt-5-codex".to_string(),
            }),
            Ok(CanonicalLlmEvent::ModelReroute {
                from_model: "gpt-5-codex".to_string(),
                to_model: "gpt-5.1-codex".to_string(),
                reason: model_provider::current_client::ModelRerouteReason::HighRiskCyberActivity,
            }),
            Ok(CanonicalLlmEvent::ModelVerification {
                verifications: vec![ModelVerification::TrustedAccessForCyber],
            }),
            Ok(CanonicalLlmEvent::TurnModerationMetadata {
                metadata: serde_json::json!({ "presentation": "inline" }),
            }),
        ]
    };
    let mut first = metadata();
    first.extend([
        Ok(CanonicalLlmEvent::ToolCall {
            id: "call-1".to_string(),
            name: "Read".to_string(),
            input: serde_json::json!({ "path": "README.md" }),
            raw_arguments: None,
            provider_executed: None,
            provider_metadata: Default::default(),
        }),
        Ok(CanonicalLlmEvent::Finish {
            reason: FinishReason::ToolCall,
            usage: None,
            response_id: Some("response-1".to_string()),
        }),
    ]);
    let mut second = metadata();
    second.extend([
        Ok(CanonicalLlmEvent::TextDelta {
            id: "text-0".to_string(),
            text: "done".to_string(),
        }),
        Ok(CanonicalLlmEvent::Finish {
            reason: FinishReason::Stop,
            usage: None,
            response_id: Some("response-2".to_string()),
        }),
    ]);
    let provider = Arc::new(ScriptedProvider::new(vec![first, second]));
    let mut events = Vec::new();

    run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .max_turns(3)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("read it".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    vec![RuntimeToolDefinition::new(
                        "Read",
                        "read files",
                        serde_json::json!({ "type": "object" }),
                    )],
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |event| events.push(event),
    )
    .await
    .expect("provider turn");

    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, CurrentProviderTurnEvent::ServerModel { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, CurrentProviderTurnEvent::ModelReroute { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, CurrentProviderTurnEvent::ModelVerification { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                CurrentProviderTurnEvent::TurnModerationMetadata { .. }
            ))
            .count(),
        2
    );
}

#[tokio::test]
async fn reasoning_summary_and_content_share_item_but_only_content_enters_provider_history() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::ReasoningSummaryDelta {
                id: "reasoning-1".to_string(),
                text: "用户可见摘要".to_string(),
                summary_index: 0,
            }),
            Ok(CanonicalLlmEvent::ReasoningContentDelta {
                id: "reasoning-1".to_string(),
                text: "provider 原始推理".to_string(),
                content_index: 0,
            }),
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-1".to_string(),
                name: "Read".to_string(),
                input: serde_json::json!({ "path": "README.md" }),
                raw_arguments: None,
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: Some("response-1".to_string()),
            }),
        ],
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-0".to_string(),
                text: "done".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: Some("response-2".to_string()),
            }),
        ],
    ]));
    let requests = Arc::clone(&provider.requests);
    let mut events = Vec::new();

    run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .max_turns(3)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("read it".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    vec![RuntimeToolDefinition::new(
                        "Read",
                        "read files",
                        serde_json::json!({ "type": "object" }),
                    )],
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |event| events.push(event),
    )
    .await
    .expect("provider turn");

    let reasoning_events = events
        .iter()
        .filter_map(|event| match event {
            CurrentProviderTurnEvent::ReasoningSummaryDelta { item_id, .. } => {
                Some(("summary", item_id.as_str()))
            }
            CurrentProviderTurnEvent::ReasoningContentDelta { item_id, .. } => {
                Some(("content", item_id.as_str()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        reasoning_events,
        vec![
            ("summary", "provider:turn-1:1:reasoning:reasoning-1"),
            ("content", "provider:turn-1:1:reasoning:reasoning-1"),
        ]
    );

    let requests = requests.lock().expect("recorded requests");
    let assistant = requests[1]
        .messages
        .iter()
        .find(|message| message.role == CurrentProviderRole::Assistant)
        .expect("assistant provider history");
    assert!(assistant.content.iter().any(|content| matches!(
        content,
        CurrentProviderContent::Reasoning(text) if text == "provider 原始推理"
    )));
    assert!(!assistant.content.iter().any(|content| matches!(
        content,
        CurrentProviderContent::Text(text) | CurrentProviderContent::Reasoning(text)
            if text == "用户可见摘要"
    )));
}

#[tokio::test]
async fn each_sampling_attempt_emits_independent_provider_phase_trace() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-0".to_string(),
                text: "working".to_string(),
            }),
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-1".to_string(),
                name: "Read".to_string(),
                input: serde_json::json!({ "path": "README.md" }),
                raw_arguments: None,
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: Some(Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(4),
                    cache_read_input_tokens: Some(2),
                    ..Usage::default()
                }),
                response_id: Some("response-1".to_string()),
            }),
        ],
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-0".to_string(),
                text: "done".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: Some(Usage {
                    input_tokens: Some(20),
                    output_tokens: Some(6),
                    cache_read_input_tokens: Some(5),
                    ..Usage::default()
                }),
                response_id: Some("response-2".to_string()),
            }),
        ],
    ]));
    let mut events = Vec::new();

    run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: Some(RuntimeReplyProviderTraceMetadata {
                provider_name: "openai".to_string(),
                model_name: "gpt-5".to_string(),
                runtime_provider_backend: "current".to_string(),
                runtime_provider_selector: Some("primary".to_string()),
                runtime_provider_protocol: Some("responses".to_string()),
                runtime_provider_active_model: Some("gpt-5".to_string()),
            }),
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .max_turns(3)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("read it".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    vec![RuntimeToolDefinition::new(
                        "Read",
                        "read files",
                        serde_json::json!({ "type": "object" }),
                    )],
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |event| events.push(event),
    )
    .await
    .expect("turn execution");

    let text_lifecycle = events
        .iter()
        .filter_map(|event| match event {
            CurrentProviderTurnEvent::TextStart { item_id } => Some(("start", item_id.as_str())),
            CurrentProviderTurnEvent::TextDelta { item_id, .. } => {
                Some(("delta", item_id.as_str()))
            }
            CurrentProviderTurnEvent::TextEnd { item_id, .. } => Some(("end", item_id.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        text_lifecycle,
        vec![
            ("start", "provider:turn-1:1:text:text-0"),
            ("delta", "provider:turn-1:1:text:text-0"),
            ("end", "provider:turn-1:1:text:text-0"),
            ("start", "provider:turn-1:2:text:text-0"),
            ("delta", "provider:turn-1:2:text:text-0"),
            ("end", "provider:turn-1:2:text:text-0"),
        ]
    );
    let text_phases = events
        .iter()
        .filter_map(|event| match event {
            CurrentProviderTurnEvent::TextEnd { phase, .. } => Some(*phase),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        text_phases,
        vec![
            CurrentProviderTextPhase::Commentary,
            CurrentProviderTextPhase::FinalAnswer,
        ]
    );

    let traces = events
        .iter()
        .filter_map(|event| match event {
            CurrentProviderTurnEvent::ProviderTrace { event } => Some(event.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        traces
            .iter()
            .map(|event| (event.attempt, event.stage))
            .collect::<Vec<_>>(),
        vec![
            (1, ProviderTraceStage::RequestStarted),
            (1, ProviderTraceStage::FirstEventReceived),
            (1, ProviderTraceStage::FirstTextDeltaReceived),
            (2, ProviderTraceStage::RequestStarted),
            (2, ProviderTraceStage::FirstEventReceived),
            (2, ProviderTraceStage::FirstTextDeltaReceived),
        ]
    );
    assert!(traces.iter().all(|event| {
        event.provider == "openai"
            && event.model == "gpt-5"
            && event.runtime_provider_backend.as_deref() == Some("current")
            && event.runtime_provider_selector.as_deref() == Some("primary")
            && event.runtime_provider_protocol.as_deref() == Some("responses")
            && event.runtime_provider_active_model.as_deref() == Some("gpt-5")
    }));
    assert!(traces
        .iter()
        .filter(|event| event.stage == ProviderTraceStage::RequestStarted)
        .all(|event| event.tool_names == ["Read"]));
    assert!(traces
        .iter()
        .filter(|event| event.stage != ProviderTraceStage::RequestStarted)
        .all(|event| event.tool_names.is_empty()));
    let steps = events
        .iter()
        .filter_map(|event| match event {
            CurrentProviderTurnEvent::ProviderStep {
                attempt,
                completed,
                finish_reason,
                text_output_chars,
                reasoning_output_chars,
                tool_call_count,
                usage,
            } => Some((
                *attempt,
                *completed,
                finish_reason.as_deref(),
                *text_output_chars,
                *reasoning_output_chars,
                *tool_call_count,
                usage.clone(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        steps,
        vec![
            (
                1,
                true,
                Some("tool_call"),
                7,
                0,
                1,
                Some(CurrentProviderUsage {
                    input_tokens: 10,
                    output_tokens: 4,
                    cached_input_tokens: Some(2),
                    cache_creation_input_tokens: None,
                }),
            ),
            (
                2,
                true,
                Some("stop"),
                4,
                0,
                0,
                Some(CurrentProviderUsage {
                    input_tokens: 20,
                    output_tokens: 6,
                    cached_input_tokens: Some(5),
                    cache_creation_input_tokens: None,
                }),
            ),
        ]
    );
}

#[tokio::test]
async fn max_turns_stops_before_starting_an_extra_provider_request() {
    let tool_call_stream = |call_id: &str, response_id: &str| {
        vec![
            Ok(CanonicalLlmEvent::ToolCall {
                id: call_id.to_string(),
                name: "Read".to_string(),
                input: serde_json::json!({ "path": "README.md" }),
                raw_arguments: None,
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: Some(response_id.to_string()),
            }),
        ]
    };
    let provider = Arc::new(ScriptedProvider::new(vec![
        tool_call_stream("call-1", "response-1"),
        tool_call_stream("call-2", "response-2"),
    ]));
    let requests = Arc::clone(&provider.requests);
    let tool = Arc::new(CountingTool::default());
    let mut events = Vec::new();

    let execution = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: Some(RuntimeReplyProviderTraceMetadata {
                provider_name: "openai".to_string(),
                model_name: "gpt-5".to_string(),
                runtime_provider_backend: "current".to_string(),
                runtime_provider_selector: Some("primary".to_string()),
                runtime_provider_protocol: Some("responses".to_string()),
                runtime_provider_active_model: Some("gpt-5".to_string()),
            }),
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .max_turns(2)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("read twice".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    vec![RuntimeToolDefinition::new(
                        "Read",
                        "read files",
                        serde_json::json!({ "type": "object" }),
                    )],
                    RuntimeToolExecutorHandle::new(tool.clone()),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |event| events.push(event),
    )
    .await
    .expect("turn execution");

    assert_eq!(requests.lock().expect("provider requests").len(), 2);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 2);
    assert_eq!(execution.text_output, MAX_REPLY_TURNS_REACHED_MESSAGE);
    assert_eq!(
        events
            .iter()
            .filter_map(|event| match event {
                CurrentProviderTurnEvent::ProviderTrace { event }
                    if event.stage == ProviderTraceStage::RequestStarted =>
                {
                    Some(event.attempt)
                }
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(
        events
            .iter()
            .filter_map(|event| match event {
                CurrentProviderTurnEvent::ProviderStep { attempt, .. } => Some(*attempt),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
}

#[tokio::test]
async fn provider_token_budget_stops_before_tool_execution_and_next_sampling() {
    let provider = Arc::new(ScriptedProvider::new(vec![vec![
        Ok(CanonicalLlmEvent::ToolCall {
            id: "call-1".to_string(),
            name: "Read".to_string(),
            input: serde_json::json!({ "path": "README.md" }),
            raw_arguments: None,
            provider_executed: None,
            provider_metadata: Default::default(),
        }),
        Ok(CanonicalLlmEvent::Finish {
            reason: FinishReason::ToolCall,
            usage: Some(Usage {
                input_tokens: Some(100),
                output_tokens: Some(25),
                cache_read_input_tokens: Some(25),
                ..Usage::default()
            }),
            response_id: Some("response-1".to_string()),
        }),
    ]]));
    let requests = Arc::clone(&provider.requests);
    let tool = Arc::new(CountingTool::default());
    let mut events = Vec::new();

    let execution = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: Some(RuntimeReplyProviderTraceMetadata {
                provider_name: "openai".to_string(),
                model_name: "gpt-5".to_string(),
                runtime_provider_backend: "current".to_string(),
                runtime_provider_selector: Some("primary".to_string()),
                runtime_provider_protocol: Some("responses".to_string()),
                runtime_provider_active_model: Some("gpt-5".to_string()),
            }),
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .max_turns(3)
                .provider_token_budget(100)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("read it".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    vec![RuntimeToolDefinition::new(
                        "Read",
                        "read files",
                        serde_json::json!({ "type": "object" }),
                    )],
                    RuntimeToolExecutorHandle::new(tool.clone()),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |event| events.push(event),
    )
    .await
    .expect("budget exhaustion is a canceled execution");

    assert!(execution.cancelled);
    assert_eq!(requests.lock().expect("provider requests").len(), 1);
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
    assert!(execution.event_errors.iter().any(|error| {
        error == "Provider token budget exhausted after attempt 1: used=100 limit=100"
    }));
    assert_eq!(
        events
            .iter()
            .filter_map(|event| match event {
                CurrentProviderTurnEvent::ProviderTrace { event }
                    if event.stage == ProviderTraceStage::RequestStarted =>
                {
                    Some(event.attempt)
                }
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec![1]
    );
    assert_eq!(
        events
            .iter()
            .filter_map(|event| match event {
                CurrentProviderTurnEvent::ProviderStep { attempt, .. } => Some(*attempt),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec![1]
    );
}

#[tokio::test]
async fn turn_executes_tool_then_continues_with_tool_result_transcript() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::ToolInputDelta {
                id: "call-1".to_string(),
                name: "Read".to_string(),
                text: "{\"path\":\"README.md\"}".to_string(),
            }),
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-1".to_string(),
                name: "Read".to_string(),
                input: serde_json::json!({ "path": "README.md" }),
                raw_arguments: None,
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: Some("response-1".to_string()),
            }),
        ],
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-0".to_string(),
                text: "done".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: Some("response-2".to_string()),
            }),
        ],
    ]));
    let requests = Arc::clone(&provider.requests);
    let lifecycle_emitter = Arc::new(RecordingLifecycleEmitter::default());
    let mut events = Vec::new();
    let execution = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .max_turns(3)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("read it".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    vec![RuntimeToolDefinition::new(
                        "Read",
                        "read files",
                        serde_json::json!({ "type": "object" }),
                    )],
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: lifecycle_emitter.clone(),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |event| events.push(event),
    )
    .await
    .expect("turn execution");

    assert_eq!(execution.text_output, "done");
    assert_eq!(execution.attempts_summary, "attempts=2");
    let lifecycle_events = lifecycle_emitter.events();
    assert_eq!(lifecycle_events.len(), 2);
    assert_eq!(lifecycle_events[0].phase, ToolLifecyclePhase::Started);
    assert_eq!(lifecycle_events[0].turn_id, "turn-1");
    assert_eq!(lifecycle_events[0].call_id, "call-1");
    assert_eq!(lifecycle_events[0].tool_name, "Read");
    assert_eq!(lifecycle_events[0].environments.len(), 1);
    assert_eq!(lifecycle_events[0].environments[0].environment_id, "local");
    assert_eq!(lifecycle_events[0].environments[0].cwd, PathBuf::from("."));
    assert_eq!(lifecycle_events[1].phase, ToolLifecyclePhase::Completed);
    assert_eq!(
        lifecycle_events[1]
            .output
            .as_ref()
            .map(|output| output.text.as_str()),
        Some("executed Read")
    );
    assert!(events.iter().any(|event| matches!(
        event,
        CurrentProviderTurnEvent::ToolInputDelta { tool_id, .. } if tool_id == "call-1"
    )));

    let requests = requests.lock().expect("recorded requests");
    assert_eq!(requests.len(), 2);
    assert!(matches!(
        requests[1].messages.last(),
        Some(CurrentProviderMessage {
            role: CurrentProviderRole::Tool,
            content,
        }) if matches!(content.as_slice(), [CurrentProviderContent::ToolResult(result)]
            if result.call_id == "call-1" && result.output == "executed Read")
    ));
}

#[tokio::test]
async fn provider_executed_web_search_emits_item_without_local_execution() {
    let started_raw_item = serde_json::json!({
        "id": "ws_1",
        "type": "web_search_call",
        "status": "in_progress",
        "action": { "type": "search", "query": "Rust release" },
    });
    let completed_raw_item = serde_json::json!({
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": { "type": "search", "query": "Rust release" },
    });
    let provider = Arc::new(ScriptedProvider::new(vec![vec![
        Ok(CanonicalLlmEvent::ToolCall {
            id: "ws_1".to_string(),
            name: "web_search".to_string(),
            input: serde_json::json!({ "type": "search", "query": "Rust release" }),
            raw_arguments: None,
            provider_executed: Some(true),
            provider_metadata: ProviderMetadata::from([(
                "raw_response_item".to_string(),
                started_raw_item,
            )]),
        }),
        Ok(CanonicalLlmEvent::ToolResult {
            id: "ws_1".to_string(),
            name: "web_search".to_string(),
            result: ToolResultValue::Json {
                value: completed_raw_item,
            },
            provider_executed: Some(true),
        }),
        Ok(CanonicalLlmEvent::TextDelta {
            id: "text-0".to_string(),
            text: "Rust 1.90".to_string(),
        }),
        Ok(CanonicalLlmEvent::Finish {
            reason: FinishReason::Stop,
            usage: None,
            response_id: Some("response-search".to_string()),
        }),
    ]]));
    let requests = Arc::clone(&provider.requests);
    let local_tool = Arc::new(CountingTool::default());
    let lifecycle_emitter = Arc::new(RecordingLifecycleEmitter::default());

    let execution = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .max_turns(2)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("search the web".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    vec![RuntimeToolDefinition::new(
                        "WebSearch",
                        "search",
                        serde_json::json!({ "type": "object" }),
                    )],
                    RuntimeToolExecutorHandle::new(local_tool.clone()),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: lifecycle_emitter.clone(),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("provider-executed search turn");

    assert_eq!(execution.text_output, "Rust 1.90");
    assert_eq!(requests.lock().expect("provider requests").len(), 1);
    assert_eq!(local_tool.calls.load(Ordering::SeqCst), 0);
    let lifecycle_events = lifecycle_emitter.events();
    assert_eq!(lifecycle_events.len(), 2);
    assert_eq!(lifecycle_events[0].phase, ToolLifecyclePhase::Started);
    assert_eq!(lifecycle_events[0].call_id, "ws_1");
    assert_eq!(lifecycle_events[0].tool_name, "web_search");
    assert_eq!(
        lifecycle_events[0].environments[0].environment_id,
        "provider"
    );
    assert_eq!(lifecycle_events[1].phase, ToolLifecyclePhase::Completed);
    assert_eq!(
        lifecycle_events[1].provider_metadata["raw_response_item"]["status"],
        "completed"
    );
    assert_eq!(
        lifecycle_events[1]
            .output
            .as_ref()
            .and_then(|output| output.structured_content.as_ref())
            .and_then(|value| value.get("type"))
            .and_then(serde_json::Value::as_str),
        Some("web_search_call")
    );
}

#[test]
fn provider_executed_raw_response_history_keeps_terminal_item() {
    let mut content = vec![CurrentProviderContent::RawResponseItem(serde_json::json!({
        "id": "ig_1",
        "type": "image_generation_call",
        "status": "in_progress"
    }))];

    upsert_raw_response_item(
        &mut content,
        serde_json::json!({
            "id": "ig_1",
            "type": "image_generation_call",
            "status": "completed",
            "revised_prompt": "a blue square",
            "result": "Zm9v"
        }),
    );

    assert!(matches!(
        content.as_slice(),
        [CurrentProviderContent::RawResponseItem(item)]
            if item["status"] == "completed" && item["result"] == "Zm9v"
    ));
}

#[tokio::test]
async fn each_sampling_step_uses_a_fresh_definition_and_executor_snapshot() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-1".to_string(),
                name: "FirstTool".to_string(),
                input: serde_json::json!({}),
                raw_arguments: None,
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: Some("response-1".to_string()),
            }),
        ],
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-final".to_string(),
                text: "done".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: Some("response-2".to_string()),
            }),
        ],
    ]));
    let requests = Arc::clone(&provider.requests);
    let source =
        RuntimeToolStepSnapshotSourceHandle::new(Arc::new(SequencedToolStepSnapshotSource {
            snapshots: Mutex::new(VecDeque::from([
                RuntimeToolStepSnapshot::new(
                    vec![RuntimeToolDefinition::new(
                        "FirstTool",
                        "first step",
                        serde_json::json!({}),
                    )],
                    RuntimeToolExecutorHandle::new(Arc::new(TaggedTool("first-executor"))),
                ),
                RuntimeToolStepSnapshot::new(
                    vec![RuntimeToolDefinition::new(
                        "SecondTool",
                        "second step",
                        serde_json::json!({}),
                    )],
                    RuntimeToolExecutorHandle::new(Arc::new(TaggedTool("second-executor"))),
                ),
            ])),
        }));

    run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .max_turns(3)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("run it".to_string()),
            ])],
            tool_step_snapshot_source: source,
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("step snapshot turn");

    let requests = requests.lock().expect("recorded requests");
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].tools[0].name(), "FirstTool");
    assert_eq!(requests[1].tools[0].name(), "SecondTool");
    assert!(matches!(
        requests[1].messages.last(),
        Some(CurrentProviderMessage {
            role: CurrentProviderRole::Tool,
            content,
        }) if matches!(content.as_slice(), [CurrentProviderContent::ToolResult(result)]
            if result.output == "first-executor")
    ));
}

#[tokio::test]
async fn mcp_tool_lifecycle_uses_captured_environment_identity() {
    let lifecycle_emitter = Arc::new(RecordingLifecycleEmitter::default());
    let snapshot = RuntimeToolStepSnapshot::with_tool_metadata(
        vec![RuntimeToolDefinition::new(
            "docs__search",
            "search docs",
            serde_json::json!({}),
        )],
        RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
        Vec::<String>::new(),
        [("docs__search".to_string(), "remote-tools".to_string())],
    );

    let results = execute_calls(
        &snapshot,
        "turn-1",
        "session-1",
        None,
        &PathBuf::from("/host/workspace"),
        None,
        lifecycle_emitter.clone(),
        vec![CurrentProviderToolCall::new(
            "call-1",
            "docs__search",
            serde_json::json!({ "query": "snapshot" }),
        )
        .with_provider_metadata(ProviderMetadata::from([(
            "google".to_string(),
            serde_json::json!({ "thoughtSignature": "sig" }),
        )]))],
        false,
    )
    .await;

    assert_eq!(results.len(), 1);
    assert!(results[0].success);
    let lifecycle_events = lifecycle_emitter.events();
    assert_eq!(lifecycle_events.len(), 2);
    for event in lifecycle_events {
        assert_eq!(event.environments.len(), 1);
        assert_eq!(event.environments[0].environment_id, "remote-tools");
        assert_eq!(event.environments[0].cwd, PathBuf::from("/host/workspace"));
        assert_eq!(event.provider_metadata["google"]["thoughtSignature"], "sig");
    }
}

#[tokio::test]
async fn repaired_tool_call_uses_canonical_snapshot_identity_and_arguments() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-repaired".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({
                    "file_path": " README.md ",
                    "start_line": "2",
                    "end_line": "3"
                }),
                raw_arguments: Some(
                    r#"{"file_path":" README.md ","start_line":"2","end_line":"3"}"#.to_string(),
                ),
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: Some("response-1".to_string()),
            }),
        ],
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-final".to_string(),
                text: "done".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: Some("response-2".to_string()),
            }),
        ],
    ]));
    let requests = Arc::clone(&provider.requests);
    let step_executor = Arc::new(CountingTool::default());
    let lifecycle_emitter = Arc::new(RecordingLifecycleEmitter::default());

    run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .max_turns(3)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("read it".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    vec![RuntimeToolDefinition::new(
                        "Read",
                        "visible tool",
                        serde_json::json!({
                            "type": "object",
                            "properties": {
                                "path": { "type": "string" },
                                "start_line": { "type": "integer", "minimum": 1 },
                                "end_line": { "type": "integer", "minimum": 1 }
                            },
                            "required": ["path"]
                        }),
                    )],
                    RuntimeToolExecutorHandle::new(step_executor.clone()),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: lifecycle_emitter.clone(),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("repaired tool call should complete");

    assert_eq!(step_executor.calls.load(Ordering::SeqCst), 1);
    let lifecycle_events = lifecycle_emitter.events();
    assert_eq!(lifecycle_events.len(), 2);
    for event in &lifecycle_events {
        assert_eq!(event.tool_name, "Read");
        assert_eq!(event.arguments["path"], "README.md");
        assert_eq!(event.arguments["start_line"], 2);
        assert_eq!(event.arguments["end_line"], 3);
        assert_eq!(
            event.provider_metadata[TOOL_CALL_REPAIR_METADATA_KEY]["status"],
            "ready"
        );
    }

    let requests = requests.lock().expect("recorded requests");
    let repaired_call = requests[1]
        .messages
        .iter()
        .find_map(|message| {
            message.content.iter().find_map(|content| match content {
                CurrentProviderContent::ToolCall(call) => Some(call),
                _ => None,
            })
        })
        .expect("repaired assistant tool call");
    assert_eq!(repaired_call.name, "Read");
    assert_eq!(repaired_call.arguments["path"], "README.md");
    assert!(matches!(
        requests[1].messages.last(),
        Some(CurrentProviderMessage {
            role: CurrentProviderRole::Tool,
            content,
        }) if matches!(content.as_slice(), [CurrentProviderContent::ToolResult(result)]
            if result.name == "Read" && result.success)
    ));
}

#[tokio::test]
async fn repaired_tool_call_cancel_wins_over_late_handler_completion() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-cancel-repaired".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({
                    "file_path": " README.md ",
                    "start_line": "2"
                }),
                raw_arguments: Some(r#"{"file_path":" README.md ","start_line":"2"}"#.to_string()),
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: Some("response-tool".to_string()),
            }),
        ],
        vec![Ok(CanonicalLlmEvent::Finish {
            reason: FinishReason::Stop,
            usage: None,
            response_id: Some("response-after-tool".to_string()),
        })],
    ]));
    let requests = Arc::clone(&provider.requests);
    let (started_sender, started_receiver) = oneshot::channel();
    let (release_sender, release_receiver) = oneshot::channel();
    let (late_completed_sender, late_completed_receiver) = oneshot::channel();
    let tool = Arc::new(LateCompletingTool {
        calls: AtomicUsize::new(0),
        started: Mutex::new(Some(started_sender)),
        release_after_cancel: Mutex::new(Some(release_receiver)),
        late_completed: Mutex::new(Some(late_completed_sender)),
    });
    let lifecycle_emitter = Arc::new(RecordingLifecycleEmitter::default());
    let turn_lifecycle_emitter = lifecycle_emitter.clone();
    let turn_tool = tool.clone();
    let cancel_token = CancellationToken::new();
    let turn_cancel_token = cancel_token.clone();

    let turn = tokio::spawn(async move {
        run_current_provider_turn(
            CurrentProviderTurnInput {
                provider,
                provider_trace_metadata: None,
                session_config: crate::session_config::SessionConfigBuilder::new("session-cancel")
                    .turn_id("turn-cancel")
                    .max_turns(3)
                    .build(),
                initial_messages: vec![CurrentProviderMessage::user(vec![
                    CurrentProviderContent::Text("read it".to_string()),
                ])],
                tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                    RuntimeToolStepSnapshot::new(
                        vec![RuntimeToolDefinition::new(
                            "Read",
                            "visible tool",
                            serde_json::json!({
                                "type": "object",
                                "properties": {
                                    "path": { "type": "string" },
                                    "start_line": { "type": "integer", "minimum": 1 }
                                },
                                "required": ["path"]
                            }),
                        )],
                        RuntimeToolExecutorHandle::new(turn_tool),
                    ),
                ),
                hook_snapshot_source: None,
                model_request_policy: None,
                tool_lifecycle_emitter: turn_lifecycle_emitter,
                working_directory: PathBuf::from("."),
                cancel_token: Some(turn_cancel_token),
                pending_input: None,
            },
            |_| {},
        )
        .await
    });

    started_receiver
        .await
        .expect("repaired handler should start");
    cancel_token.cancel();
    let execution = tokio::time::timeout(Duration::from_secs(5), turn)
        .await
        .expect("cancel should release the repaired tool call")
        .expect("turn task should complete")
        .expect("cancel should be a normal terminal result");

    assert!(execution.cancelled);
    assert_eq!(execution.attempts_summary, "attempts=1");
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    assert_eq!(requests.lock().expect("provider requests").len(), 1);
    let lifecycle_events = lifecycle_emitter.events();
    assert_eq!(lifecycle_events.len(), 2);
    assert_eq!(lifecycle_events[0].phase, ToolLifecyclePhase::Started);
    assert_eq!(lifecycle_events[1].phase, ToolLifecyclePhase::Completed);
    for event in &lifecycle_events {
        assert_eq!(event.tool_name, "Read");
        assert_eq!(event.arguments["path"], "README.md");
        assert_eq!(event.arguments["start_line"], 2);
        assert_eq!(
            event.provider_metadata[TOOL_CALL_REPAIR_METADATA_KEY]["status"],
            "ready"
        );
    }
    assert_eq!(
        lifecycle_events[1]
            .output
            .as_ref()
            .and_then(|output| output.metadata.get("tool_outcome"))
            .and_then(serde_json::Value::as_str),
        Some("aborted")
    );

    release_sender
        .send(())
        .expect("release late handler completion");
    tokio::time::timeout(Duration::from_secs(5), late_completed_receiver)
        .await
        .expect("late handler should finish")
        .expect("late handler completion signal");
    assert_eq!(lifecycle_emitter.events(), lifecycle_events);
    assert_eq!(requests.lock().expect("provider requests").len(), 1);
}

#[tokio::test]
async fn repaired_tool_call_is_not_replayed_when_next_provider_step_times_out() {
    let provider = Arc::new(ToolCallThenHangingProvider {
        attempt: AtomicUsize::new(0),
        requests: Mutex::new(Vec::new()),
    });
    let retained_provider = provider.clone();
    let tool = Arc::new(CountingTool::default());
    let lifecycle_emitter = Arc::new(RecordingLifecycleEmitter::default());
    let mut turn_context = agent_protocol::turn_context::TurnContextOverride::default();
    turn_context.metadata.insert(
        "runtime_request".to_string(),
        serde_json::json!({
            "harness": {
                "generation": {
                    "first_visible_output_timeout_ms": 1_000,
                    "provider_step_timeout_ms": 20
                }
            }
        }),
    );

    let error = tokio::time::timeout(
        Duration::from_secs(5),
        run_current_provider_turn(
            CurrentProviderTurnInput {
                provider,
                provider_trace_metadata: None,
                session_config: crate::session_config::SessionConfigBuilder::new("session-timeout")
                    .turn_id("turn-timeout")
                    .turn_context(turn_context)
                    .max_turns(3)
                    .build(),
                initial_messages: vec![CurrentProviderMessage::user(vec![
                    CurrentProviderContent::Text("read it".to_string()),
                ])],
                tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                    RuntimeToolStepSnapshot::new(
                        vec![RuntimeToolDefinition::new(
                            "Read",
                            "visible tool",
                            serde_json::json!({
                                "type": "object",
                                "properties": {
                                    "path": { "type": "string" },
                                    "start_line": { "type": "integer", "minimum": 1 }
                                },
                                "required": ["path"]
                            }),
                        )],
                        RuntimeToolExecutorHandle::new(tool.clone()),
                    ),
                ),
                hook_snapshot_source: None,
                model_request_policy: None,
                tool_lifecycle_emitter: lifecycle_emitter.clone(),
                working_directory: PathBuf::from("."),
                cancel_token: None,
                pending_input: None,
            },
            |_| {},
        ),
    )
    .await
    .expect("provider timeout should beat the outer test timeout")
    .expect_err("second provider step should hit its absolute deadline");

    assert_eq!(
        error.message,
        "Provider step exceeded the absolute deadline of 20ms"
    );
    assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    let lifecycle_events = lifecycle_emitter.events();
    assert_eq!(lifecycle_events.len(), 2);
    assert_eq!(lifecycle_events[0].phase, ToolLifecyclePhase::Started);
    assert_eq!(lifecycle_events[1].phase, ToolLifecyclePhase::Completed);
    assert!(lifecycle_events[1]
        .output
        .as_ref()
        .is_some_and(|output| output.success));

    let requests = retained_provider
        .requests
        .lock()
        .expect("provider requests");
    assert_eq!(requests.len(), 2);
    let repaired_call = requests[1]
        .messages
        .iter()
        .flat_map(|message| message.content.iter())
        .find_map(|content| match content {
            CurrentProviderContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .expect("repaired assistant tool call");
    assert_eq!(repaired_call.name, "Read");
    assert_eq!(repaired_call.arguments["path"], "README.md");
    assert_eq!(repaired_call.arguments["start_line"], 2);
    assert!(matches!(
        requests[1].messages.last(),
        Some(CurrentProviderMessage {
            role: CurrentProviderRole::Tool,
            content,
        }) if matches!(content.as_slice(), [CurrentProviderContent::ToolResult(result)]
            if result.name == "Read" && result.success)
    ));
}

#[tokio::test]
async fn malformed_unknown_and_schema_mismatched_calls_fail_without_reaching_step_executor() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-malformed".to_string(),
                name: "Read".to_string(),
                input: serde_json::json!("{not-json"),
                raw_arguments: Some("{not-json".to_string()),
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-mcp".to_string(),
                name: "mcp__hidden__unknown".to_string(),
                input: serde_json::json!({}),
                raw_arguments: None,
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-schema".to_string(),
                name: "Read".to_string(),
                input: serde_json::json!({ "path": 42 }),
                raw_arguments: Some(r#"{"path":42}"#.to_string()),
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: Some("response-1".to_string()),
            }),
        ],
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-final".to_string(),
                text: "done".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: Some("response-2".to_string()),
            }),
        ],
    ]));
    let requests = Arc::clone(&provider.requests);
    let step_executor = Arc::new(CountingTool::default());
    let lifecycle_emitter = Arc::new(RecordingLifecycleEmitter::default());

    run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .max_turns(3)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("guess hidden tools".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    vec![RuntimeToolDefinition::new(
                        "Read",
                        "visible tool",
                        serde_json::json!({
                            "type": "object",
                            "properties": {
                                "path": { "type": "string" }
                            },
                            "required": ["path"],
                            "additionalProperties": false
                        }),
                    )],
                    RuntimeToolExecutorHandle::new(step_executor.clone()),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: lifecycle_emitter.clone(),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("invalid tool calls should become failed tool results");

    assert_eq!(step_executor.calls.load(Ordering::SeqCst), 0);
    let lifecycle_events = lifecycle_emitter.events();
    assert_eq!(lifecycle_events.len(), 6);
    assert!(lifecycle_events
        .iter()
        .all(|event| event.tool_name == INVALID_TOOL_CALL_NAME));
    let errors = lifecycle_events
        .iter()
        .filter(|event| event.phase == ToolLifecyclePhase::Completed)
        .filter_map(|event| event.output.as_ref())
        .filter_map(|output| output.error.as_deref())
        .collect::<Vec<_>>();
    assert!(errors
        .iter()
        .any(|error| error.contains("malformed JSON arguments")));
    assert!(errors.iter().any(|error| error.contains("not advertised")));
    assert!(errors
        .iter()
        .any(|error| error.contains("did not match the advertised input schema")));
    for completed in lifecycle_events
        .iter()
        .filter(|event| event.phase == ToolLifecyclePhase::Completed)
    {
        let output = completed.output.as_ref().expect("completed output");
        assert!(!output.success);
        assert_eq!(
            output
                .metadata
                .get(tool_runtime::tool_result_projection::TOOL_HANDLER_EXECUTED_METADATA_KEY),
            Some(&serde_json::Value::Bool(false))
        );
    }

    let requests = requests.lock().expect("recorded requests");
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].tools.len(), 1);
    assert_eq!(requests[0].tools[0].name(), "Read");
    let schema_invalid_call = requests[1]
        .messages
        .iter()
        .flat_map(|message| message.content.iter())
        .find_map(|content| match content {
            CurrentProviderContent::ToolCall(call) if call.id == "call-schema" => Some(call),
            _ => None,
        })
        .expect("schema-invalid assistant tool call");
    assert_eq!(schema_invalid_call.name, INVALID_TOOL_CALL_NAME);
    assert!(!schema_invalid_call.arguments.to_string().contains("42"));
    assert!(matches!(
        requests[1].messages.last(),
        Some(CurrentProviderMessage {
            role: CurrentProviderRole::Tool,
            content,
        }) if content.len() == 3 && content.iter().all(|part| matches!(
            part,
            CurrentProviderContent::ToolResult(result)
                if !result.success
                    && result.name == INVALID_TOOL_CALL_NAME
        ))
    ));
}

#[tokio::test]
async fn turn_executes_same_response_tool_batch_in_parallel_when_policy_allows() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-1".to_string(),
                name: "Read".to_string(),
                input: serde_json::json!({ "path": "README.md" }),
                raw_arguments: None,
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-2".to_string(),
                name: "Glob".to_string(),
                input: serde_json::json!({ "pattern": "*.rs" }),
                raw_arguments: None,
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: Some("response-1".to_string()),
            }),
        ],
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-final".to_string(),
                text: "done".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: Some("response-2".to_string()),
            }),
        ],
    ]));
    let probe = Arc::new(ParallelProbe::default());
    let policy = RuntimeReplyModelRequestPolicy {
        responses: None,
        tool_call: Some(
            model_provider::provider_stream::RuntimeReplyToolCallPolicy {
                supports_parallel_tool_calls: true,
                parallel_tool_calls: true,
            },
        ),
        reasoning_output: None,
    };

    run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .max_turns(3)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("inspect it".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    vec![
                        RuntimeToolDefinition::new("Read", "read files", serde_json::json!({})),
                        RuntimeToolDefinition::new("Glob", "find files", serde_json::json!({})),
                    ],
                    RuntimeToolExecutorHandle::new(probe.clone()),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: Some(policy),
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("parallel tool turn");

    assert_eq!(probe.max_active.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn tool_batch_serializes_calls_that_do_not_support_parallel_execution() {
    let probe = Arc::new(ParallelProbe::default());
    let snapshot = RuntimeToolStepSnapshot::with_tool_metadata(
        vec![
            RuntimeToolDefinition::new("Read", "read files", serde_json::json!({})),
            RuntimeToolDefinition::new("Glob", "find files", serde_json::json!({})),
        ],
        RuntimeToolExecutorHandle::new(probe.clone()),
        ["Glob".to_string()],
        Vec::<(String, String)>::new(),
    );

    let results = execute_calls(
        &snapshot,
        "turn-1",
        "session-1",
        None,
        &PathBuf::from("."),
        None,
        Arc::new(RecordingLifecycleEmitter::default()),
        vec![
            CurrentProviderToolCall::new(
                "call-1",
                "Read",
                serde_json::json!({ "path": "README.md" }),
            ),
            CurrentProviderToolCall::new(
                "call-2",
                "Glob",
                serde_json::json!({ "pattern": "*.rs" }),
            ),
        ],
        true,
    )
    .await;

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|result| result.success));
    assert_eq!(probe.max_active.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn turn_propagates_canonical_provider_error() {
    let provider = Arc::new(ScriptedProvider::new(vec![vec![Ok(
        CanonicalLlmEvent::ProviderError {
            message: "stream truncated".to_string(),
            classification: Some(FailureClassification::Transport),
            retryable: Some(true),
        },
    )]]));

    let error = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("hello".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect_err("provider error must fail the turn");

    assert_eq!(error.message, "stream truncated");
    assert!(!error.emitted_any);
    assert_eq!(
        error.classification(),
        Some(FailureClassification::Transport)
    );
    assert!(error.retryable());
    assert!(error.is_reroutable_provider_failure());
}

#[tokio::test]
async fn custom_tool_call_fails_without_executable_code_mode_session() {
    let provider = Arc::new(ScriptedProvider::new(vec![vec![Ok(
        CanonicalLlmEvent::CustomToolCall {
            id: "custom-call-1".to_string(),
            name: "exec".to_string(),
            input: "return 42;".to_string(),
            namespace: Some("codemode".to_string()),
            provider_metadata: Default::default(),
        },
    )]]));
    let tool = Arc::new(CountingTool::default());
    let lifecycle_emitter = Arc::new(RecordingLifecycleEmitter::default());

    let error = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-custom")
                .turn_id("turn-custom")
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("run the code".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(tool.clone()),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: lifecycle_emitter.clone(),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect_err("custom tool call must require an executable CodeMode session");

    assert_eq!(
        error.message,
        "custom tool call requires an executable CodeMode session"
    );
    assert!(!error.emitted_any);
    assert_eq!(
        error.classification(),
        Some(FailureClassification::InvalidRequest)
    );
    assert!(!error.retryable());
    assert!(!error.is_reroutable_provider_failure());
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
    assert!(lifecycle_emitter.events().is_empty());
}

#[tokio::test]
async fn custom_exec_uses_code_mode_session_and_resamples_with_typed_result() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::CustomToolCall {
                id: "custom-call-1".to_string(),
                name: "exec".to_string(),
                input: "text(40 + 2);".to_string(),
                namespace: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: Some("response-custom-1".to_string()),
            }),
        ],
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-0".to_string(),
                text: "done".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: Some("response-custom-2".to_string()),
            }),
        ],
    ]));
    let requests = Arc::clone(&provider.requests);
    let code_mode = Arc::new(RecordingCodeModeSession::new(vec![Ok(
        RuntimeCodeModeResponse::Result {
            cell_id: RuntimeCodeModeCellId::new("1"),
            output: "42".to_string(),
            error_text: None,
        },
    )]));
    let lifecycle_emitter = Arc::new(RecordingLifecycleEmitter::default());
    let cancel_token = CancellationToken::new();

    let execution = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-custom")
                .turn_id("turn-custom")
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("run the code".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(Arc::new(CountingTool::default())),
                )
                .with_code_mode_session(
                    RuntimeCodeModeSessionHandle::new(code_mode.clone()),
                    Vec::new(),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: lifecycle_emitter.clone(),
            working_directory: PathBuf::from("."),
            cancel_token: Some(cancel_token),
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("custom exec turn");

    assert_eq!(execution.text_output, "done");
    assert_eq!(
        code_mode.requests(),
        vec![(
            "custom-call-1".to_string(),
            "text(40 + 2);".to_string(),
            0,
            true,
        )]
    );
    let requests = requests.lock().expect("provider requests");
    assert_eq!(requests.len(), 2);
    assert!(matches!(
        requests[0].tools.as_slice(),
        [
            CurrentProviderTool::Custom { name: exec, .. },
            CurrentProviderTool::Function { name: wait, .. },
        ] if exec == "exec" && wait == "wait"
    ));
    assert!(matches!(
        requests[1].messages.as_slice(),
        [..,
            CurrentProviderMessage {
                role: CurrentProviderRole::Assistant,
                content: assistant_content,
            },
            CurrentProviderMessage {
                role: CurrentProviderRole::Tool,
                content: tool_content,
            }
        ] if matches!(assistant_content.as_slice(), [CurrentProviderContent::CustomToolCall(call)]
            if call.id == "custom-call-1" && call.name == "exec" && call.input == "text(40 + 2);")
            && matches!(tool_content.as_slice(), [CurrentProviderContent::CustomToolResult(result)]
                if result.call_id == "custom-call-1"
                    && result.name == "exec"
                    && result.success
                    && result.output == "Script completed\nOutput:\n42"
                    && result.error.is_none())
    ));
    let events = lifecycle_emitter.events();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].phase, ToolLifecyclePhase::Started);
    assert_eq!(events[0].call_id, "custom-call-1");
    assert_eq!(events[0].tool_name, "exec");
    assert_eq!(events[1].phase, ToolLifecyclePhase::Completed);
    assert_eq!(events[1].call_id, "custom-call-1");
    assert!(events[1]
        .output
        .as_ref()
        .is_some_and(|output| output.success));
}

#[tokio::test]
async fn custom_exec_nested_tool_reuses_the_frozen_executor_and_lifecycle() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::CustomToolCall {
                id: "custom-nested-1".to_string(),
                name: "exec".to_string(),
                input: "await tools.read({ path: 'README.md' });".to_string(),
                namespace: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: Some("response-nested-1".to_string()),
            }),
        ],
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-nested-final".to_string(),
                text: "nested done".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: Some("response-nested-2".to_string()),
            }),
        ],
    ]));
    let requests = Arc::clone(&provider.requests);
    let nested_executor = Arc::new(CountingTool::default());
    let lifecycle_emitter = Arc::new(RecordingLifecycleEmitter::default());
    let nested_definition =
        RuntimeToolDefinition::new("read", "read files", serde_json::json!({"type": "object"}));
    let nested_tool = RuntimeCodeModeTool {
        identity: RuntimeToolIdentity::plain("read"),
        definition: nested_definition.clone(),
        code_name: "read".to_string(),
        global_name: "read".to_string(),
    };

    let execution = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-nested")
                .turn_id("turn-nested")
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("use a nested tool".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(nested_executor.clone()),
                )
                .with_code_mode_session(
                    RuntimeCodeModeSessionHandle::new(Arc::new(NestedDispatchCodeModeSession)),
                    vec![nested_tool],
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: lifecycle_emitter.clone(),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("nested CodeMode turn");

    assert_eq!(execution.text_output, "nested done");
    assert_eq!(nested_executor.calls.load(Ordering::SeqCst), 1);
    let events = lifecycle_emitter.events();
    assert_eq!(events.len(), 4);
    assert_eq!(events[0].phase, ToolLifecyclePhase::Started);
    assert_eq!(events[0].call_id, "custom-nested-1");
    assert_eq!(events[0].tool_name, "exec");
    assert_eq!(events[1].phase, ToolLifecyclePhase::Started);
    assert_eq!(events[1].call_id, "code-mode-nested-read-1");
    assert_eq!(events[1].tool_name, "read");
    assert_eq!(events[2].phase, ToolLifecyclePhase::Completed);
    assert_eq!(events[2].call_id, "code-mode-nested-read-1");
    assert!(events[2]
        .output
        .as_ref()
        .is_some_and(|output| output.success));
    assert_eq!(events[3].phase, ToolLifecyclePhase::Completed);
    assert_eq!(events[3].call_id, "custom-nested-1");
    let output_deltas = lifecycle_emitter.output_deltas();
    assert_eq!(output_deltas.len(), 1);
    assert_eq!(output_deltas[0].turn_id, "turn-nested");
    assert_eq!(output_deltas[0].call_id, "custom-nested-1");
    assert_eq!(output_deltas[0].tool_name, "exec");
    assert_eq!(output_deltas[0].delta, "nested tool starting");
    assert_eq!(
        output_deltas[0]
            .metadata
            .get("code_mode_cell_id")
            .and_then(serde_json::Value::as_str),
        Some("cell-nested-provider-turn")
    );
    let requests = requests.lock().expect("provider requests");
    assert!(matches!(
        requests[1].messages.last(),
        Some(CurrentProviderMessage {
            role: CurrentProviderRole::Tool,
            content,
        }) if matches!(
            content.as_slice(),
            [
                CurrentProviderContent::CustomToolResult(notification),
                CurrentProviderContent::CustomToolResult(result),
            ] if notification.call_id == "custom-nested-1"
                && notification.success
                && notification.output == "nested tool starting"
                && result.call_id == "custom-nested-1"
                && result.success
                && result.output.contains("executed read")
        )
    ));
}

#[tokio::test]
async fn wait_function_uses_the_same_code_mode_session_and_resamples() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::ToolCall {
                id: "wait-call-1".to_string(),
                name: "wait".to_string(),
                input: serde_json::json!({
                    "cell_id": "cell-running",
                    "yield_time_ms": 250,
                    "max_tokens": 128,
                }),
                raw_arguments: None,
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: Some("response-wait-1".to_string()),
            }),
        ],
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-after-wait".to_string(),
                text: "waited".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: Some("response-wait-2".to_string()),
            }),
        ],
    ]));
    let requests = Arc::clone(&provider.requests);
    let code_mode = Arc::new(
        RecordingCodeModeSession::new(Vec::new()).with_wait_responses(vec![Ok(
            RuntimeCodeModeWaitOutcome::LiveCell(RuntimeCodeModeResponse::Result {
                cell_id: RuntimeCodeModeCellId::new("cell-running"),
                output: "finished".to_string(),
                error_text: None,
            }),
        )]),
    );

    let execution = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-wait")
                .turn_id("turn-wait")
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("wait for the cell".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(Arc::new(CountingTool::default())),
                )
                .with_code_mode_session(
                    RuntimeCodeModeSessionHandle::new(code_mode.clone()),
                    Vec::new(),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("wait turn");

    assert_eq!(execution.text_output, "waited");
    assert_eq!(
        code_mode.wait_requests(),
        vec![("cell-running".to_string(), 250)]
    );
    let requests = requests.lock().expect("provider requests");
    assert!(matches!(
        requests[1].messages.last(),
        Some(CurrentProviderMessage {
            role: CurrentProviderRole::Tool,
            content,
        }) if matches!(content.as_slice(), [CurrentProviderContent::ToolResult(result)]
            if result.call_id == "wait-call-1"
                && result.name == "wait"
                && result.success
                && result.output == "Script completed\nOutput:\nfinished")
    ));
}

#[tokio::test]
async fn custom_exec_session_failure_is_returned_to_model_for_recovery() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::CustomToolCall {
                id: "custom-call-failed".to_string(),
                name: "exec".to_string(),
                input: "throw new Error('boom');".to_string(),
                namespace: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: Some("response-custom-failed".to_string()),
            }),
        ],
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-recovered".to_string(),
                text: "recovered".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: Some("response-custom-recovered".to_string()),
            }),
        ],
    ]));
    let requests = Arc::clone(&provider.requests);
    let code_mode = Arc::new(RecordingCodeModeSession::new(vec![Err(
        "isolated runtime rejected source".to_string(),
    )]));

    let execution = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new(
                "session-custom-error",
            )
            .turn_id("turn-custom-error")
            .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("run invalid code".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(Arc::new(CountingTool::default())),
                )
                .with_code_mode_session(RuntimeCodeModeSessionHandle::new(code_mode), Vec::new()),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("model should recover from code mode failure");

    assert_eq!(execution.text_output, "recovered");
    let requests = requests.lock().expect("provider requests");
    assert!(matches!(
        requests[1].messages.last(),
        Some(CurrentProviderMessage {
            role: CurrentProviderRole::Tool,
            content,
        }) if matches!(content.as_slice(), [CurrentProviderContent::CustomToolResult(result)]
            if !result.success
                && result.output == "Script failed\nOutput:\n\nScript error:\nisolated runtime rejected source"
                && result.error.as_deref() == Some("isolated runtime rejected source"))
    ));
}

#[tokio::test]
async fn mixed_function_and_custom_results_preserve_provider_call_order() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::CustomToolCall {
                id: "custom-yielded".to_string(),
                name: "exec".to_string(),
                input: "yield_control();".to_string(),
                namespace: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::ToolCall {
                id: "function-read".to_string(),
                name: "Read".to_string(),
                input: serde_json::json!({ "path": "README.md" }),
                raw_arguments: None,
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::CustomToolCall {
                id: "custom-completed".to_string(),
                name: "exec".to_string(),
                input: "text('done');".to_string(),
                namespace: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: Some("response-mixed-tools".to_string()),
            }),
        ],
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-mixed-final".to_string(),
                text: "mixed done".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: Some("response-mixed-final".to_string()),
            }),
        ],
    ]));
    let requests = Arc::clone(&provider.requests);
    let code_mode = Arc::new(RecordingCodeModeSession::new(vec![
        Ok(RuntimeCodeModeResponse::Yielded {
            cell_id: RuntimeCodeModeCellId::new("cell-yielded"),
            output: "partial".to_string(),
        }),
        Ok(RuntimeCodeModeResponse::Result {
            cell_id: RuntimeCodeModeCellId::new("cell-completed"),
            output: "done".to_string(),
            error_text: None,
        }),
    ]));

    let execution = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-mixed-tools")
                .turn_id("turn-mixed-tools")
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("run mixed tools".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    vec![RuntimeToolDefinition::new(
                        "Read",
                        "read files",
                        serde_json::json!({ "type": "object" }),
                    )],
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                )
                .with_code_mode_session(
                    RuntimeCodeModeSessionHandle::new(code_mode.clone()),
                    Vec::new(),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("mixed tool turn");

    assert_eq!(execution.text_output, "mixed done");
    assert!(code_mode.terminations().is_empty());
    let requests = requests.lock().expect("provider requests");
    assert!(matches!(
        requests[1].messages.as_slice(),
        [..,
            CurrentProviderMessage {
                role: CurrentProviderRole::Assistant,
                content: assistant_content,
            },
            CurrentProviderMessage {
                role: CurrentProviderRole::Tool,
                content: tool_content,
            }
        ] if matches!(assistant_content.as_slice(), [
                CurrentProviderContent::CustomToolCall(first),
                CurrentProviderContent::ToolCall(second),
                CurrentProviderContent::CustomToolCall(third),
            ] if first.id == "custom-yielded"
                && second.id == "function-read"
                && third.id == "custom-completed")
            && matches!(tool_content.as_slice(), [
                CurrentProviderContent::CustomToolResult(first),
                CurrentProviderContent::ToolResult(second),
                CurrentProviderContent::CustomToolResult(third),
            ] if first.call_id == "custom-yielded"
                && first.output == "Script running with cell ID cell-yielded\nOutput:\npartial"
                && second.call_id == "function-read"
                && third.call_id == "custom-completed")
    ));
}

#[tokio::test]
async fn cancellation_terminates_a_started_code_mode_cell() {
    let provider = Arc::new(ScriptedProvider::new(vec![vec![
        Ok(CanonicalLlmEvent::CustomToolCall {
            id: "custom-hanging".to_string(),
            name: "exec".to_string(),
            input: "await new Promise(() => {});".to_string(),
            namespace: None,
            provider_metadata: Default::default(),
        }),
        Ok(CanonicalLlmEvent::Finish {
            reason: FinishReason::ToolCall,
            usage: None,
            response_id: Some("response-hanging".to_string()),
        }),
    ]]));
    let (started_tx, started_rx) = oneshot::channel();
    let code_mode = Arc::new(HangingCodeModeSession {
        started: Mutex::new(Some(started_tx)),
        terminations: Mutex::new(Vec::new()),
    });
    let cancel_token = CancellationToken::new();
    let turn_cancel_token = cancel_token.clone();
    let task_code_mode = code_mode.clone();

    let turn = tokio::spawn(async move {
        run_current_provider_turn(
            CurrentProviderTurnInput {
                provider,
                provider_trace_metadata: None,
                session_config: crate::session_config::SessionConfigBuilder::new(
                    "session-custom-cancel",
                )
                .turn_id("turn-custom-cancel")
                .build(),
                initial_messages: vec![CurrentProviderMessage::user(vec![
                    CurrentProviderContent::Text("run until canceled".to_string()),
                ])],
                tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                    RuntimeToolStepSnapshot::new(
                        Vec::new(),
                        RuntimeToolExecutorHandle::new(Arc::new(CountingTool::default())),
                    )
                    .with_code_mode_session(
                        RuntimeCodeModeSessionHandle::new(task_code_mode),
                        Vec::new(),
                    ),
                ),
                hook_snapshot_source: None,
                model_request_policy: None,
                tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
                working_directory: PathBuf::from("."),
                cancel_token: Some(turn_cancel_token),
                pending_input: None,
            },
            |_| {},
        )
        .await
    });

    started_rx.await.expect("code mode cell started");
    cancel_token.cancel();
    let execution = tokio::time::timeout(Duration::from_secs(1), turn)
        .await
        .expect("cancellation should terminate the code mode cell")
        .expect("turn task")
        .expect("cancellation is a normal terminal result");

    assert!(execution.cancelled);
    assert_eq!(
        code_mode
            .terminations
            .lock()
            .expect("hanging terminations")
            .as_slice(),
        ["cell-hanging"]
    );
}

#[tokio::test]
async fn non_exec_custom_tool_is_rejected_before_code_mode_dispatch() {
    let provider = Arc::new(ScriptedProvider::new(vec![vec![Ok(
        CanonicalLlmEvent::CustomToolCall {
            id: "custom-call-other".to_string(),
            name: "run_code".to_string(),
            input: "return 42;".to_string(),
            namespace: None,
            provider_metadata: Default::default(),
        },
    )]]));
    let code_mode = Arc::new(RecordingCodeModeSession::new(Vec::new()));

    let error = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new(
                "session-custom-other",
            )
            .turn_id("turn-custom-other")
            .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("run the code".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(Arc::new(CountingTool::default())),
                )
                .with_code_mode_session(
                    RuntimeCodeModeSessionHandle::new(code_mode.clone()),
                    Vec::new(),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect_err("only exact exec custom calls are executable");

    assert_eq!(error.message, "unsupported custom tool call: run_code");
    assert_eq!(
        error.classification(),
        Some(FailureClassification::InvalidRequest)
    );
    assert!(!error.retryable());
    assert!(code_mode.requests().is_empty());
}

#[tokio::test]
async fn provider_failure_after_consuming_steer_is_not_reroutable() {
    let (started_tx, started_rx) = oneshot::channel();
    let (continue_tx, continue_rx) = oneshot::channel();
    let provider = Arc::new(GatedEmptyThenRetryableErrorProvider {
        attempt: AtomicUsize::new(0),
        started: Mutex::new(Some(started_tx)),
        continue_after_steer: Mutex::new(Some(continue_rx)),
        requests: Mutex::new(Vec::new()),
    });
    let observed_error = Arc::new(Mutex::new(None::<RuntimeReplyAttemptError>));
    let registry = RuntimeSessionRegistry::default();
    let session = registry
        .get_or_create("session-steer-reroute", "thread-steer-reroute")
        .await
        .expect("bind steer reroute actor");
    let task_provider = Arc::clone(&provider);
    let task_error = Arc::clone(&observed_error);
    let task = RuntimeSessionClosureTask::new(
        "turn-steer-reroute",
        Vec::new(),
        move |context, _input, _task_cancel| {
            let provider = Arc::clone(&task_provider);
            let observed_error = Arc::clone(&task_error);
            Box::pin(async move {
                let error = run_current_provider_turn(
                    CurrentProviderTurnInput {
                        provider,
                        provider_trace_metadata: None,
                        session_config: crate::session_config::SessionConfigBuilder::new(
                            "session-steer-reroute",
                        )
                        .turn_id("turn-steer-reroute")
                        .build(),
                        initial_messages: vec![CurrentProviderMessage::user(vec![
                            CurrentProviderContent::Text("initial".to_string()),
                        ])],
                        tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                            RuntimeToolStepSnapshot::new(
                                Vec::new(),
                                RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                            ),
                        ),
                        hook_snapshot_source: None,
                        model_request_policy: None,
                        tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
                        working_directory: PathBuf::from("."),
                        cancel_token: None,
                        pending_input: Some(context.input_handle()),
                    },
                    |_| {},
                )
                .await
                .expect_err("second sampling step should fail");
                *observed_error.lock().expect("record provider error") = Some(error);
                Ok(())
            })
        },
    );
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("submit provider turn");

    started_rx.await.expect("first provider request");
    session
        .steer(vec![RuntimeSessionInput::User(RuntimeReplyInput::text(
            "follow-up",
        ))])
        .await
        .expect("steer active turn");
    continue_tx.send(()).expect("continue provider stream");
    assert_eq!(
        submission.completion.await.expect("task completion"),
        Ok(crate::session_loop::RuntimeSessionTaskOutcome::Completed)
    );

    let error = observed_error
        .lock()
        .expect("provider error")
        .clone()
        .expect("recorded provider error");
    assert!(!error.emitted_any);
    assert!(error.retryable());
    assert!(!error.is_reroutable_provider_failure());
    let requests = provider.requests.lock().expect("provider requests");
    assert_eq!(requests.len(), 2);
    assert!(requests[1].messages.iter().any(|message| {
        message.content.iter().any(
            |content| matches!(content, CurrentProviderContent::Text(text) if text == "follow-up"),
        )
    }));

    registry
        .shutdown("session-steer-reroute")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn provider_quota_error_preserves_usage_limit_kind() {
    let provider = Arc::new(ScriptedProvider::new(vec![vec![Ok(
        CanonicalLlmEvent::ProviderError {
            message: "provider quota exhausted".to_string(),
            classification: Some(FailureClassification::Quota),
            retryable: Some(false),
        },
    )]]));

    let error = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("hello".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect_err("provider quota must fail the turn");

    assert_eq!(error.message, "provider quota exhausted");
    assert!(error.is_usage_limit_exceeded());
}

#[tokio::test]
async fn turn_resamples_reasoning_only_response_with_same_sampling_snapshot() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::ReasoningContentDelta {
                id: "reasoning-1".to_string(),
                text: "I need to think about this first.".to_string(),
                content_index: 0,
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: Some(Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(4),
                    ..Usage::default()
                }),
                response_id: Some("response-1".to_string()),
            }),
        ],
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-2".to_string(),
                text: "done".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: Some(Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(1),
                    ..Usage::default()
                }),
                response_id: Some("response-2".to_string()),
            }),
        ],
    ]));
    let requests = Arc::clone(&provider.requests);
    let snapshot_source =
        RuntimeToolStepSnapshotSourceHandle::new(Arc::new(SequencedToolStepSnapshotSource {
            snapshots: Mutex::new(VecDeque::from([RuntimeToolStepSnapshot::new(
                Vec::new(),
                RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
            )])),
        }));

    let mut events = Vec::new();
    let execution = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .max_turns(1)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("hello".to_string()),
            ])],
            tool_step_snapshot_source: snapshot_source,
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |event| events.push(event),
    )
    .await
    .expect("reasoning-only completion should be resampled");

    assert_eq!(execution.text_output, "done");
    assert_eq!(execution.attempts_summary, "attempts=2");
    let requests = requests.lock().expect("provider requests");
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0], requests[1]);

    assert_eq!(
        events
            .iter()
            .filter_map(|event| match event {
                CurrentProviderTurnEvent::ReasoningStart { item_id } => {
                    Some(("start", item_id.as_str()))
                }
                CurrentProviderTurnEvent::ReasoningContentDelta { item_id, .. } => {
                    Some(("delta", item_id.as_str()))
                }
                CurrentProviderTurnEvent::ReasoningEnd { item_id } => {
                    Some(("end", item_id.as_str()))
                }
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec![
            ("start", "provider:turn-1:1:reasoning:reasoning-1"),
            ("delta", "provider:turn-1:1:reasoning:reasoning-1"),
            ("end", "provider:turn-1:1:reasoning:reasoning-1"),
        ]
    );
    assert_eq!(
        events
            .iter()
            .filter_map(|event| match event {
                CurrentProviderTurnEvent::ProviderStep { attempt, .. } => Some(*attempt),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
}

#[tokio::test]
async fn empty_response_retry_budget_is_bounded_and_does_not_spend_max_turns() {
    let empty = || {
        vec![Ok(CanonicalLlmEvent::Finish {
            reason: FinishReason::Stop,
            usage: None,
            response_id: None,
        })]
    };
    let provider = Arc::new(ScriptedProvider::new(vec![empty(), empty(), empty()]));
    let requests = Arc::clone(&provider.requests);

    let error = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .max_turns(1)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("hello".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect_err("empty response retries must be bounded");

    assert_eq!(requests.lock().expect("provider requests").len(), 3);
    assert_eq!(
        error.message,
        "Provider completed without user-visible output after 3 attempts (empty response retries exhausted: 2/2)"
    );
}

#[tokio::test]
async fn empty_final_after_tool_call_resamples_without_spending_max_turns() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-1".to_string(),
                name: "Read".to_string(),
                input: serde_json::json!({ "path": "README.md" }),
                raw_arguments: None,
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: Some("response-1".to_string()),
            }),
        ],
        vec![Ok(CanonicalLlmEvent::Finish {
            reason: FinishReason::Stop,
            usage: None,
            response_id: Some("response-empty".to_string()),
        })],
        vec![
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-final".to_string(),
                text: "done".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: Some("response-3".to_string()),
            }),
        ],
    ]));
    let requests = Arc::clone(&provider.requests);

    let execution = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .max_turns(2)
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("inspect it".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    vec![RuntimeToolDefinition::new(
                        "Read",
                        "read files",
                        serde_json::json!({}),
                    )],
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("empty final answer after a tool call should be resampled");

    assert_eq!(execution.text_output, "done");
    assert_eq!(execution.attempts_summary, "attempts=3");
    assert!(!execution
        .text_output
        .contains(MAX_REPLY_TURNS_REACHED_MESSAGE));

    let requests = requests.lock().expect("provider requests");
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[1], requests[2]);
    assert!(requests[2].messages.iter().any(|message| {
        message.role == CurrentProviderRole::Tool
            && message.content.iter().any(|part| {
                matches!(part, CurrentProviderContent::ToolResult(result)
                    if result.output == "executed Read")
            })
    }));
    assert!(requests[2]
        .messages
        .iter()
        .all(|message| !(message.role == CurrentProviderRole::Assistant
            && message.content.is_empty())));
}

#[tokio::test]
async fn content_filtered_empty_response_completes_without_resampling() {
    let provider = Arc::new(ScriptedProvider::new(vec![vec![Ok(
        CanonicalLlmEvent::Finish {
            reason: FinishReason::ContentFilter,
            usage: None,
            response_id: Some("response-refusal".to_string()),
        },
    )]]));
    let requests = Arc::clone(&provider.requests);

    let execution = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("blocked request".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect("content-filtered response is a legitimate terminal");

    assert!(execution.text_output.is_empty());
    assert_eq!(execution.attempts_summary, "attempts=1");
    assert_eq!(requests.lock().expect("provider requests").len(), 1);
}

#[tokio::test]
async fn empty_length_response_fails_without_resampling() {
    let provider = Arc::new(ScriptedProvider::new(vec![vec![Ok(
        CanonicalLlmEvent::Finish {
            reason: FinishReason::Length,
            usage: None,
            response_id: Some("response-truncated".to_string()),
        },
    )]]));
    let requests = Arc::clone(&provider.requests);

    let error = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("long response".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect_err("empty max-token response is deterministic");

    assert_eq!(
        error.message,
        "Provider reached its output limit without user-visible output"
    );
    assert_eq!(requests.lock().expect("provider requests").len(), 1);
}

#[tokio::test]
async fn cancelling_during_provider_request_releases_the_turn_without_waiting_for_http() {
    let (started_sender, started_receiver) = oneshot::channel();
    let cancel_token = CancellationToken::new();
    let turn_cancel_token = cancel_token.clone();
    let provider = Arc::new(HangingRequestProvider {
        started: Mutex::new(Some(started_sender)),
    });

    let turn = tokio::spawn(async move {
        run_current_provider_turn(
            CurrentProviderTurnInput {
                provider,
                provider_trace_metadata: None,
                session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                    .turn_id("turn-1")
                    .build(),
                initial_messages: vec![CurrentProviderMessage::user(vec![
                    CurrentProviderContent::Text("hello".to_string()),
                ])],
                tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                    RuntimeToolStepSnapshot::new(
                        Vec::new(),
                        RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                    ),
                ),
                hook_snapshot_source: None,
                model_request_policy: None,
                tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
                working_directory: PathBuf::from("."),
                cancel_token: Some(turn_cancel_token),
                pending_input: None,
            },
            |_| {},
        )
        .await
    });

    started_receiver
        .await
        .expect("provider request should start");
    cancel_token.cancel();

    let execution = tokio::time::timeout(std::time::Duration::from_millis(100), turn)
        .await
        .expect("cancel should not wait for the provider")
        .expect("turn task should complete")
        .expect("canceled provider request should be a normal terminal result");

    assert!(execution.cancelled);
}

#[tokio::test]
async fn cancelling_while_waiting_for_the_first_provider_event_releases_the_turn() {
    let (started_sender, started_receiver) = oneshot::channel();
    let cancel_token = CancellationToken::new();
    let turn_cancel_token = cancel_token.clone();
    let provider = Arc::new(HangingFirstEventProvider {
        stream_started: Mutex::new(Some(started_sender)),
    });

    let turn = tokio::spawn(async move {
        run_current_provider_turn(
            CurrentProviderTurnInput {
                provider,
                provider_trace_metadata: None,
                session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                    .turn_id("turn-1")
                    .build(),
                initial_messages: vec![CurrentProviderMessage::user(vec![
                    CurrentProviderContent::Text("hello".to_string()),
                ])],
                tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                    RuntimeToolStepSnapshot::new(
                        Vec::new(),
                        RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                    ),
                ),
                hook_snapshot_source: None,
                model_request_policy: None,
                tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
                working_directory: PathBuf::from("."),
                cancel_token: Some(turn_cancel_token),
                pending_input: None,
            },
            |_| {},
        )
        .await
    });

    started_receiver
        .await
        .expect("provider stream should start");
    cancel_token.cancel();

    let execution = tokio::time::timeout(std::time::Duration::from_millis(100), turn)
        .await
        .expect("cancel should not wait for the first provider event")
        .expect("turn task should complete")
        .expect("canceled provider stream should be a normal terminal result");

    assert!(execution.cancelled);
}

#[tokio::test]
async fn cancellation_preserves_usage_returned_by_the_same_provider_poll() {
    let cancel_token = CancellationToken::new();
    let provider = Arc::new(CancelOnFirstUsageProvider {
        cancel_token: cancel_token.clone(),
    });
    let mut events = Vec::new();

    let execution = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                .turn_id("turn-1")
                .build(),
            initial_messages: vec![CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("hello".to_string()),
            ])],
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: Some(cancel_token),
            pending_input: None,
        },
        |event| events.push(event),
    )
    .await
    .expect("canceled provider stream should be a normal terminal result");

    assert!(execution.cancelled);
    assert_eq!(
        events
            .iter()
            .filter_map(|event| match event {
                CurrentProviderTurnEvent::Usage { attempt, usage } => {
                    Some((*attempt, usage.input_tokens, usage.output_tokens))
                }
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec![(1, 17, 5)]
    );
    assert!(!events
        .iter()
        .any(|event| matches!(event, CurrentProviderTurnEvent::ProviderStep { .. })));
}

#[tokio::test]
async fn cancellation_flushes_provider_usage_to_the_session_runtime() {
    let registry = RuntimeSessionRegistry::default();
    let session = registry
        .get_or_create("session-cancel-usage", "thread-cancel-usage")
        .await
        .expect("bind cancel usage actor");
    let cancel_token = CancellationToken::new();
    let provider = Arc::new(CancelOnFirstUsageProvider {
        cancel_token: cancel_token.clone(),
    });
    let task = RuntimeSessionClosureTask::new(
        "turn-cancel-usage",
        Vec::new(),
        move |context, _input, _task_cancel| {
            let provider = Arc::clone(&provider);
            let cancel_token = cancel_token.clone();
            Box::pin(async move {
                let execution = run_current_provider_turn(
                    CurrentProviderTurnInput {
                        provider,
                        provider_trace_metadata: None,
                        session_config: crate::session_config::SessionConfigBuilder::new(
                            "session-cancel-usage",
                        )
                        .turn_id("turn-cancel-usage")
                        .build(),
                        initial_messages: vec![CurrentProviderMessage::user(vec![
                            CurrentProviderContent::Text("hello".to_string()),
                        ])],
                        tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                            RuntimeToolStepSnapshot::new(
                                Vec::new(),
                                RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                            ),
                        ),
                        hook_snapshot_source: None,
                        model_request_policy: None,
                        tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
                        working_directory: PathBuf::from("."),
                        cancel_token: Some(cancel_token),
                        pending_input: Some(context.input_handle()),
                    },
                    |_| {},
                )
                .await
                .expect("canceled provider turn should resolve");
                assert!(execution.cancelled);
                assert_eq!(context.token_usage().await.cache_write_input_tokens, 6);
                Ok(())
            })
        },
    );
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("submit canceled usage task");
    assert_eq!(
        submission.completion.await.expect("task completion"),
        Ok(crate::session_loop::RuntimeSessionTaskOutcome::Completed)
    );
    registry
        .shutdown("session-cancel-usage")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn provider_error_flushes_prior_usage_to_the_session_runtime() {
    let registry = RuntimeSessionRegistry::default();
    let session = registry
        .get_or_create("session-error-usage", "thread-error-usage")
        .await
        .expect("bind error usage actor");
    let task = RuntimeSessionClosureTask::new(
        "turn-error-usage",
        Vec::new(),
        move |context, _input, _task_cancel| {
            Box::pin(async move {
                let error = run_current_provider_turn(
                    CurrentProviderTurnInput {
                        provider: Arc::new(UsageThenProviderError),
                        provider_trace_metadata: None,
                        session_config: crate::session_config::SessionConfigBuilder::new(
                            "session-error-usage",
                        )
                        .turn_id("turn-error-usage")
                        .build(),
                        initial_messages: vec![CurrentProviderMessage::user(vec![
                            CurrentProviderContent::Text("hello".to_string()),
                        ])],
                        tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                            RuntimeToolStepSnapshot::new(
                                Vec::new(),
                                RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                            ),
                        ),
                        hook_snapshot_source: None,
                        model_request_policy: None,
                        tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
                        working_directory: PathBuf::from("."),
                        cancel_token: None,
                        pending_input: Some(context.input_handle()),
                    },
                    |_| {},
                )
                .await
                .expect_err("provider error should fail the turn");
                assert_eq!(error.message, "provider stopped after usage");
                let usage = context.token_usage().await;
                assert_eq!(usage.input_tokens, 19);
                assert_eq!(usage.output_tokens, 7);
                assert_eq!(usage.cache_write_input_tokens, 8);
                Ok(())
            })
        },
    );
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("submit error usage task");
    assert_eq!(
        submission.completion.await.expect("task completion"),
        Ok(crate::session_loop::RuntimeSessionTaskOutcome::Completed)
    );
    registry
        .shutdown("session-error-usage")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn stream_error_flushes_prior_usage_to_the_session_runtime() {
    let registry = RuntimeSessionRegistry::default();
    let session = registry
        .get_or_create("session-stream-error-usage", "thread-stream-error-usage")
        .await
        .expect("bind stream error usage actor");
    let task = RuntimeSessionClosureTask::new(
        "turn-stream-error-usage",
        Vec::new(),
        move |context, _input, _task_cancel| {
            Box::pin(async move {
                let error = run_current_provider_turn(
                    CurrentProviderTurnInput {
                        provider: Arc::new(UsageThenStreamError),
                        provider_trace_metadata: None,
                        session_config: crate::session_config::SessionConfigBuilder::new(
                            "session-stream-error-usage",
                        )
                        .turn_id("turn-stream-error-usage")
                        .build(),
                        initial_messages: vec![CurrentProviderMessage::user(vec![
                            CurrentProviderContent::Text("hello".to_string()),
                        ])],
                        tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                            RuntimeToolStepSnapshot::new(
                                Vec::new(),
                                RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                            ),
                        ),
                        hook_snapshot_source: None,
                        model_request_policy: None,
                        tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
                        working_directory: PathBuf::from("."),
                        cancel_token: None,
                        pending_input: Some(context.input_handle()),
                    },
                    |_| {},
                )
                .await
                .expect_err("stream error should fail the turn");
                assert_eq!(error.message, "provider stream failed after usage");
                let usage = context.token_usage().await;
                assert_eq!(usage.input_tokens, 29);
                assert_eq!(usage.output_tokens, 17);
                assert_eq!(usage.cache_write_input_tokens, 21);
                Ok(())
            })
        },
    );
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("submit stream error usage task");
    assert_eq!(
        submission.completion.await.expect("task completion"),
        Ok(crate::session_loop::RuntimeSessionTaskOutcome::Completed)
    );
    registry
        .shutdown("session-stream-error-usage")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn provider_step_timeout_flushes_prior_usage_to_the_session_runtime() {
    let registry = RuntimeSessionRegistry::default();
    let session = registry
        .get_or_create("session-timeout-usage", "thread-timeout-usage")
        .await
        .expect("bind timeout usage actor");
    let task = RuntimeSessionClosureTask::new(
        "turn-timeout-usage",
        Vec::new(),
        move |context, _input, _task_cancel| {
            Box::pin(async move {
                let mut turn_context = agent_protocol::turn_context::TurnContextOverride::default();
                turn_context.metadata.insert(
                    "runtime_request".to_string(),
                    serde_json::json!({
                        "harness": {
                            "generation": {
                                "first_visible_output_timeout_ms": 1_000,
                                "provider_step_timeout_ms": 20
                            }
                        }
                    }),
                );
                let error = run_current_provider_turn(
                    CurrentProviderTurnInput {
                        provider: Arc::new(UsageThenHangingStream),
                        provider_trace_metadata: None,
                        session_config: crate::session_config::SessionConfigBuilder::new(
                            "session-timeout-usage",
                        )
                        .turn_id("turn-timeout-usage")
                        .turn_context(turn_context)
                        .build(),
                        initial_messages: vec![CurrentProviderMessage::user(vec![
                            CurrentProviderContent::Text("hello".to_string()),
                        ])],
                        tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                            RuntimeToolStepSnapshot::new(
                                Vec::new(),
                                RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                            ),
                        ),
                        hook_snapshot_source: None,
                        model_request_policy: None,
                        tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
                        working_directory: PathBuf::from("."),
                        cancel_token: None,
                        pending_input: Some(context.input_handle()),
                    },
                    |_| {},
                )
                .await
                .expect_err("provider step deadline should fail the turn");
                assert_eq!(
                    error.message,
                    "Provider step exceeded the absolute deadline of 20ms"
                );
                let usage = context.token_usage().await;
                assert_eq!(usage.input_tokens, 23);
                assert_eq!(usage.output_tokens, 11);
                assert_eq!(usage.cache_write_input_tokens, 13);
                Ok(())
            })
        },
    );
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("submit timeout usage task");
    assert_eq!(
        submission.completion.await.expect("task completion"),
        Ok(crate::session_loop::RuntimeSessionTaskOutcome::Completed)
    );
    registry
        .shutdown("session-timeout-usage")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn turn_requires_canonical_turn_id_before_provider_sampling() {
    let provider = Arc::new(ScriptedProvider::new(Vec::new()));
    let requests = Arc::clone(&provider.requests);

    let error = run_current_provider_turn(
        CurrentProviderTurnInput {
            provider,
            provider_trace_metadata: None,
            session_config: crate::session_config::SessionConfigBuilder::new("session-1").build(),
            initial_messages: Vec::new(),
            tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                RuntimeToolStepSnapshot::new(
                    Vec::new(),
                    RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                ),
            ),
            hook_snapshot_source: None,
            model_request_policy: None,
            tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            pending_input: None,
        },
        |_| {},
    )
    .await
    .expect_err("missing canonical turn id must fail closed");

    assert_eq!(
        error.message,
        "Current provider turn requires a canonical turn_id"
    );
    assert!(requests.lock().expect("provider requests").is_empty());
}

#[tokio::test]
async fn reasoning_heartbeats_do_not_bypass_first_visible_output_deadline() {
    let mut turn_context = agent_protocol::turn_context::TurnContextOverride::default();
    turn_context.metadata.insert(
        "runtime_request".to_string(),
        serde_json::json!({
            "harness": {
                "generation": {
                    "first_visible_output_timeout_ms": 20
                }
            }
        }),
    );
    let mut events = Vec::new();

    let error = tokio::time::timeout(
        Duration::from_secs(1),
        run_current_provider_turn(
            CurrentProviderTurnInput {
                provider: Arc::new(ReasoningHeartbeatProvider),
                provider_trace_metadata: None,
                session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                    .turn_id("turn-1")
                    .turn_context(turn_context)
                    .build(),
                initial_messages: vec![CurrentProviderMessage::user(vec![
                    CurrentProviderContent::Text("hello".to_string()),
                ])],
                tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                    RuntimeToolStepSnapshot::new(
                        Vec::new(),
                        RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                    ),
                ),
                hook_snapshot_source: None,
                model_request_policy: None,
                tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
                working_directory: PathBuf::from("."),
                cancel_token: None,
                pending_input: None,
            },
            |event| events.push(event),
        ),
    )
    .await
    .expect("reasoning-only stream must stop before outer test timeout")
    .expect_err("reasoning-only stream must fail without user-visible output");

    assert_eq!(
        error.message,
        "Provider produced no user-visible output within 20ms"
    );
    assert!(error.emitted_any);
    assert!(events
        .iter()
        .any(|event| matches!(event, CurrentProviderTurnEvent::ReasoningEnd { .. })));
}

#[tokio::test]
async fn provider_step_deadline_stops_continuous_heartbeat_stream() {
    let mut turn_context = agent_protocol::turn_context::TurnContextOverride::default();
    turn_context.metadata.insert(
        "runtime_request".to_string(),
        serde_json::json!({
            "harness": {
                "generation": {
                    "first_visible_output_timeout_ms": 1_000,
                    "provider_step_timeout_ms": 20
                }
            }
        }),
    );
    let mut events = Vec::new();

    let error = tokio::time::timeout(
        Duration::from_secs(1),
        run_current_provider_turn(
            CurrentProviderTurnInput {
                provider: Arc::new(ReasoningHeartbeatProvider),
                provider_trace_metadata: None,
                session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                    .turn_id("turn-1")
                    .turn_context(turn_context)
                    .build(),
                initial_messages: vec![CurrentProviderMessage::user(vec![
                    CurrentProviderContent::Text("hello".to_string()),
                ])],
                tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                    RuntimeToolStepSnapshot::new(
                        Vec::new(),
                        RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                    ),
                ),
                hook_snapshot_source: None,
                model_request_policy: None,
                tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
                working_directory: PathBuf::from("."),
                cancel_token: None,
                pending_input: None,
            },
            |event| events.push(event),
        ),
    )
    .await
    .expect("continuous stream must stop before outer test timeout")
    .expect_err("continuous stream must fail on the absolute deadline");

    assert_eq!(
        error.message,
        "Provider step exceeded the absolute deadline of 20ms"
    );
    assert!(error.emitted_any);
    assert!(events
        .iter()
        .any(|event| matches!(event, CurrentProviderTurnEvent::ReasoningEnd { .. })));
}

#[tokio::test]
async fn provider_step_deadline_closes_continuous_visible_text_stream() {
    let mut turn_context = agent_protocol::turn_context::TurnContextOverride::default();
    turn_context.metadata.insert(
        "runtime_request".to_string(),
        serde_json::json!({
            "harness": {
                "generation": {
                    "first_visible_output_timeout_ms": 1_000,
                    "provider_step_timeout_ms": 100
                }
            }
        }),
    );
    let mut events = Vec::new();

    let error = tokio::time::timeout(
        Duration::from_secs(1),
        run_current_provider_turn(
            CurrentProviderTurnInput {
                provider: Arc::new(TextHeartbeatProvider),
                provider_trace_metadata: None,
                session_config: crate::session_config::SessionConfigBuilder::new("session-1")
                    .turn_id("turn-1")
                    .turn_context(turn_context)
                    .build(),
                initial_messages: vec![CurrentProviderMessage::user(vec![
                    CurrentProviderContent::Text("hello".to_string()),
                ])],
                tool_step_snapshot_source: RuntimeToolStepSnapshotSourceHandle::fixed(
                    RuntimeToolStepSnapshot::new(
                        Vec::new(),
                        RuntimeToolExecutorHandle::new(Arc::new(EchoTool)),
                    ),
                ),
                hook_snapshot_source: None,
                model_request_policy: None,
                tool_lifecycle_emitter: Arc::new(RecordingLifecycleEmitter::default()),
                working_directory: PathBuf::from("."),
                cancel_token: None,
                pending_input: None,
            },
            |event| events.push(event),
        ),
    )
    .await
    .expect("visible text stream must stop before outer test timeout")
    .expect_err("visible text stream must fail on the absolute deadline");

    assert_eq!(
        error.message,
        "Provider step exceeded the absolute deadline of 100ms"
    );
    assert!(error.emitted_any);

    let text_item_id = "provider:turn-1:1:text:text-0";
    assert!(events.iter().any(|event| matches!(
        event,
        CurrentProviderTurnEvent::TextStart { item_id } if item_id == text_item_id
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        CurrentProviderTurnEvent::TextDelta { item_id, text }
            if item_id == text_item_id && !text.is_empty()
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        CurrentProviderTurnEvent::TextEnd {
            item_id,
            phase: CurrentProviderTextPhase::FinalAnswer,
        } if item_id == text_item_id
    )));
}

#[test]
fn session_user_input_preserves_provider_part_order_without_injection_text() {
    use crate::reply_input::{
        ImageDetail, RuntimeReplyInput, RuntimeReplyInputImage, RuntimeReplyInputPart,
    };
    use crate::session_loop::RuntimeSessionInput;
    use model_provider::current_client::CurrentProviderContent;

    let message = runtime_session_input_message(RuntimeSessionInput::User(
        RuntimeReplyInput::from_parts(vec![
            RuntimeReplyInputPart::Text {
                text: "before".to_string(),
                text_elements: Vec::new(),
            },
            RuntimeReplyInputPart::Skill {
                name: "review".to_string(),
                path: "/skills/review/SKILL.md".to_string(),
            },
            RuntimeReplyInputPart::Image(RuntimeReplyInputImage {
                uri: "sidecar://image-1".to_string(),
                media_type: "image/png".to_string(),
                provider_data: Some("data:image/png;base64,abc".to_string()),
                detail: Some(ImageDetail::High),
            }),
            RuntimeReplyInputPart::Mention {
                name: "docs".to_string(),
                path: "app://docs".to_string(),
            },
            RuntimeReplyInputPart::Text {
                text: "after".to_string(),
                text_elements: Vec::new(),
            },
        ]),
    ))
    .expect("provider message");

    assert!(matches!(
        message.content.as_slice(),
        [
            CurrentProviderContent::Text(before),
            CurrentProviderContent::Image {
                uri,
                media_type,
                detail: Some(ImageDetail::High),
                ..
            },
            CurrentProviderContent::Text(after),
        ] if before == "before"
            && uri == "sidecar://image-1"
            && media_type == "image/png"
            && after == "after"
    ));
}

#[test]
fn inter_agent_input_preserves_typed_identity_and_delivery_semantics() {
    use crate::session_loop::{
        RuntimeSessionInterAgentDeliveryMode, RuntimeSessionInterAgentInput,
        RuntimeSessionInterAgentMessageKind, RuntimeSessionInterAgentResultStatus,
    };

    let text = runtime_inter_agent_text(&RuntimeSessionInterAgentInput {
        message_id: "message-1".to_string(),
        root_thread_id: "thread-root".to_string(),
        sender_thread_id: "thread-sender".to_string(),
        recipient_thread_id: "thread-recipient".to_string(),
        content: "done <ok>".to_string(),
        kind: RuntimeSessionInterAgentMessageKind::Result,
        source_turn_id: Some("turn-source".to_string()),
        result_status: Some(RuntimeSessionInterAgentResultStatus::Completed),
        delivery_mode: RuntimeSessionInterAgentDeliveryMode::TriggerTurn,
    });

    assert!(text.contains("<message_id>message-1</message_id>"));
    assert!(text.contains("<sender_thread_id>thread-sender</sender_thread_id>"));
    assert!(text.contains("<recipient_thread_id>thread-recipient</recipient_thread_id>"));
    assert!(text.contains("<kind>result</kind>"));
    assert!(text.contains("<result_status>completed</result_status>"));
    assert!(text.contains("<delivery_mode>trigger_turn</delivery_mode>"));
    assert!(text.contains("<content>done &lt;ok&gt;</content>"));
}
