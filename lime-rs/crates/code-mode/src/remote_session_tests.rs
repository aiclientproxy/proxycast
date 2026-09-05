use super::ProcessCodeModeSessionProvider;
use code_mode_protocol::{
    RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest, RuntimeCodeModeFuture,
    RuntimeCodeModeNestedToolCall, RuntimeCodeModeResponse, RuntimeCodeModeSession,
    RuntimeCodeModeSessionDelegate, RuntimeCodeModeSessionLimits, RuntimeCodeModeSessionProvider,
    RuntimeCodeModeWaitRequest,
};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::time::{sleep, timeout, Duration};
use tokio_util::sync::CancellationToken;

#[test]
fn missing_process_host_fails_closed_at_availability_boundary() {
    let provider = ProcessCodeModeSessionProvider::with_host_path(PathBuf::from(
        "/tmp/code-mode-host-that-does-not-exist",
    ));
    let error = provider.availability().expect_err("missing host");
    assert!(error.contains("not found"));
}

#[test]
fn cloned_providers_share_the_process_host_and_session_id_allocator() {
    let provider = ProcessCodeModeSessionProvider::with_host_path(PathBuf::from(
        "/tmp/code-mode-host-that-does-not-exist",
    ));
    let clone = provider.clone();

    assert_eq!(provider.next_session_id(), "session-1");
    assert_eq!(clone.next_session_id(), "session-2");
}

#[derive(Default)]
struct RecordingDelegate {
    closed_cells: Mutex<Vec<RuntimeCodeModeCellId>>,
}

impl RuntimeCodeModeSessionDelegate for RecordingDelegate {
    fn invoke_tool<'a>(
        &'a self,
        _invocation: RuntimeCodeModeNestedToolCall,
        _cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, Value> {
        Box::pin(async { Err("nested tools are not used in this test".to_string()) })
    }

    fn notify<'a>(
        &'a self,
        _tool_call_id: String,
        _cell_id: RuntimeCodeModeCellId,
        _text: String,
        _cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

    fn cell_closed(&self, cell_id: &RuntimeCodeModeCellId) {
        self.closed_cells
            .lock()
            .expect("closed cells poisoned")
            .push(cell_id.clone());
    }
}

fn execute_request(source: &str, yield_time_ms: u64) -> RuntimeCodeModeExecuteRequest {
    RuntimeCodeModeExecuteRequest {
        tool_call_id: "process-reconnect-test".to_string(),
        source: source.to_string(),
        enabled_tools: Vec::new(),
        yield_time_ms: Some(yield_time_ms),
        max_output_tokens: None,
        cancellation_token: None,
    }
}

#[tokio::test]
async fn process_host_disconnect_fails_pending_work_and_rebinds_generation() {
    let host_path = crate::default_code_mode_host_path();
    assert!(
        host_path.is_file(),
        "code-mode-host binary is required for this integration test: {}",
        host_path.display()
    );
    let provider = ProcessCodeModeSessionProvider::with_host_path(host_path);
    let delegate = Arc::new(RecordingDelegate::default());
    let session = super::ReconnectableSession::new(
        provider.clone(),
        delegate.clone(),
        RuntimeCodeModeSessionLimits::default(),
    );
    session.initialize().await.expect("create process session");

    let first_binding = session.inner.live_binding().expect("first process binding");
    let first_connection = first_binding.connection.clone();
    let first_started = session
        .execute(execute_request("await new Promise(() => {});", 25))
        .await
        .expect("start first cell");
    let first_cell_id = first_started.cell_id.clone();
    assert!(!first_cell_id.as_str().starts_with("g"));
    assert!(matches!(
        first_started
            .initial_response()
            .await
            .expect("first initial response"),
        RuntimeCodeModeResponse::Yielded { .. }
    ));

    let wait_connection = first_connection.clone();
    let wait_session_id = first_binding.session_id.clone();
    let wait_cell_id = first_cell_id.clone();
    let mut pending_wait = tokio::spawn(async move {
        wait_connection
            .wait(
                wait_session_id,
                RuntimeCodeModeWaitRequest {
                    cell_id: wait_cell_id,
                    yield_time_ms: 60_000,
                },
            )
            .await
    });
    timeout(Duration::from_secs(5), async {
        loop {
            if first_connection.pending_request_count_for_test() > 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("wait request should be registered before disconnect");
    first_connection.disconnect_for_test();

    let wait_error = timeout(Duration::from_secs(5), &mut pending_wait)
        .await
        .expect("pending wait should fail after disconnect")
        .expect("pending wait task should join")
        .expect_err("disconnected wait must fail closed");
    assert!(
        wait_error.contains("disconnected") || wait_error.contains("closed"),
        "unexpected disconnect error: {wait_error}"
    );
    timeout(Duration::from_secs(5), async {
        loop {
            if delegate
                .closed_cells
                .lock()
                .expect("closed cells poisoned")
                .as_slice()
                == [first_cell_id.clone()]
            {
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("active cell should close exactly once");

    let second_started = session
        .execute(execute_request("text(42);", 1_000))
        .await
        .expect("rebind process session");
    assert!(second_started.cell_id.as_str().starts_with("g2:"));
    assert!(matches!(
        second_started
            .initial_response()
            .await
            .expect("second initial response"),
        RuntimeCodeModeResponse::Result { .. }
    ));

    let stale_error = session
        .wait(RuntimeCodeModeWaitRequest {
            cell_id: first_cell_id,
            yield_time_ms: 1,
        })
        .await
        .expect_err("retired generation cell must be rejected");
    assert!(stale_error.contains("stale") || stale_error.contains("generation"));

    session.shutdown().await.expect("shutdown process session");
}
