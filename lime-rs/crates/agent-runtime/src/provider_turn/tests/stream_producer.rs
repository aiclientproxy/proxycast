use super::*;
use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant as StdInstant;

struct BlockingPollStream;

impl Stream for BlockingPollStream {
    type Item = Result<CanonicalLlmEvent, CurrentProviderError>;

    fn poll_next(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        std::thread::sleep(Duration::from_millis(400));
        Poll::Pending
    }
}

struct BlockingPollProvider;

impl CurrentProvider for BlockingPollProvider {
    fn stream<'a>(
        &'a self,
        _request: CurrentProviderRequest,
    ) -> BoxFuture<'a, Result<CurrentProviderStream, CurrentProviderError>> {
        Box::pin(async move {
            let stream: CurrentProviderStream = Box::pin(BlockingPollStream);
            Ok(stream)
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn first_visible_output_deadline_is_independent_from_provider_stream_poll() {
    let mut turn_context = agent_protocol::turn_context::TurnContextOverride::default();
    turn_context.metadata.insert(
        "runtime_request".to_string(),
        serde_json::json!({
            "harness": {
                "generation": {
                    "first_visible_output_timeout_ms": 30,
                    "provider_step_timeout_ms": 1_000
                }
            }
        }),
    );
    let started_at = StdInstant::now();

    let error = tokio::time::timeout(
        Duration::from_secs(1),
        run_current_provider_turn(
            CurrentProviderTurnInput {
                provider: Arc::new(BlockingPollProvider),
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
            |_| {},
        ),
    )
    .await
    .expect("provider deadline should complete before the outer timeout")
    .expect_err("blocking provider stream must fail on first-visible-output deadline");

    assert_eq!(
        error.message,
        "Provider produced no user-visible output within 30ms"
    );
    assert!(
        started_at.elapsed() < Duration::from_millis(250),
        "deadline was blocked by provider stream poll for {:?}",
        started_at.elapsed()
    );
}
