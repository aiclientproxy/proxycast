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
use tool_runtime::tool_executor::{
    RuntimeToolExecutionFuture, RuntimeToolExecutionRequest, RuntimeToolExecutionResult,
    RuntimeToolExecutor,
};
use tool_runtime::tool_lifecycle::{
    ToolLifecycleEmissionFuture, ToolLifecycleEvent, ToolLifecyclePhase,
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
}

impl RecordingLifecycleEmitter {
    fn events(&self) -> Vec<ToolLifecycleEvent> {
        self.events.lock().expect("lifecycle events").clone()
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
        ]
    };
    let mut first = metadata();
    first.extend([
        Ok(CanonicalLlmEvent::ToolCall {
            id: "call-1".to_string(),
            name: "Read".to_string(),
            input: serde_json::json!({ "path": "README.md" }),
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
    assert_eq!(requests[0].tools[0].name, "FirstTool");
    assert_eq!(requests[1].tools[0].name, "SecondTool");
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
async fn unadvertised_native_and_mcp_calls_fail_without_reaching_step_executor() {
    let provider = Arc::new(ScriptedProvider::new(vec![
        vec![
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-native".to_string(),
                name: "apply_patch".to_string(),
                input: serde_json::json!({ "patch": "hidden" }),
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-mcp".to_string(),
                name: "mcp__hidden__unknown".to_string(),
                input: serde_json::json!({}),
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
                        serde_json::json!({}),
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
    .expect("unadvertised tool calls should become failed tool results");

    assert_eq!(step_executor.calls.load(Ordering::SeqCst), 0);
    let lifecycle_events = lifecycle_emitter.events();
    assert_eq!(lifecycle_events.len(), 4);
    for completed in lifecycle_events
        .iter()
        .filter(|event| event.phase == ToolLifecyclePhase::Completed)
    {
        let output = completed.output.as_ref().expect("completed output");
        assert!(!output.success);
        assert!(output
            .error
            .as_deref()
            .is_some_and(|error| error.contains("not advertised")));
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
    assert_eq!(requests[0].tools[0].name, "Read");
    assert!(matches!(
        requests[1].messages.last(),
        Some(CurrentProviderMessage {
            role: CurrentProviderRole::Tool,
            content,
        }) if content.len() == 2 && content.iter().all(|part| matches!(
            part,
            CurrentProviderContent::ToolResult(result)
                if !result.success
                    && result.error.as_deref().is_some_and(|error| error.contains("not advertised"))
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
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-2".to_string(),
                name: "Glob".to_string(),
                input: serde_json::json!({ "pattern": "*.rs" }),
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
    let session = registry.get_or_create("session-steer-reroute").await;
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
    let session = registry.get_or_create("session-cancel-usage").await;
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
    let session = registry.get_or_create("session-error-usage").await;
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
    let session = registry.get_or_create("session-stream-error-usage").await;
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
    let session = registry.get_or_create("session-timeout-usage").await;
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
