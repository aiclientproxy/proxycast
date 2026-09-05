use super::{ConnectionState, PendingRequest};
use code_mode_protocol::host::{
    ClientToHost, DelegateRequest, HostResponse, HostToClient, WireResult, MAX_IN_FLIGHT_REQUESTS,
};
use code_mode_protocol::{
    RuntimeCodeModeCellId, RuntimeCodeModeFuture, RuntimeCodeModeNestedToolCall,
    RuntimeCodeModeSessionDelegate,
};
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, Notify};
use tokio_util::sync::CancellationToken;

#[test]
fn request_tracker_rejects_duplicate_ids_and_releases_on_failure() {
    let state = Arc::new(ConnectionState::new(CancellationToken::new()));
    let (first_tx, _first_rx) = oneshot::channel();
    state
        .register_pending(1, PendingRequest::Standard(first_tx))
        .expect("first request");
    let (second_tx, _second_rx) = oneshot::channel();
    let error = state
        .register_pending(1, PendingRequest::Standard(second_tx))
        .expect_err("duplicate request ID");
    assert!(error.contains("duplicate"));
    state.fail("driver stopped");
    assert_eq!(state.failure_message(), "driver stopped");
    assert!(!state.alive.load(std::sync::atomic::Ordering::Acquire));
    assert!(MAX_IN_FLIGHT_REQUESTS > 0);
}

#[derive(Default)]
struct RecordingDelegate {
    closed: Mutex<Vec<RuntimeCodeModeCellId>>,
}

impl RuntimeCodeModeSessionDelegate for RecordingDelegate {
    fn invoke_tool<'a>(
        &'a self,
        _invocation: RuntimeCodeModeNestedToolCall,
        _cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, Value> {
        Box::pin(async { Ok(Value::Null) })
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
        self.closed
            .lock()
            .expect("closed cells")
            .push(cell_id.clone());
    }
}

#[tokio::test]
async fn execute_admission_tracks_cells_and_failure_only_closes_live_cells() {
    let state = Arc::new(ConnectionState::new(CancellationToken::new()));
    let delegate = Arc::new(RecordingDelegate::default());
    state.insert_session("session-1".to_string(), delegate.clone());
    let (started_tx, started_rx) = oneshot::channel();
    let (initial_tx, _initial_rx) = oneshot::channel();
    state
        .register_pending(
            1,
            PendingRequest::Execute {
                session_id: "session-1".to_string(),
                started: started_tx,
                initial: initial_tx,
            },
        )
        .expect("execute request");
    let (outgoing, _outgoing_rx) = mpsc::channel::<ClientToHost>(1);
    let cell_id = RuntimeCodeModeCellId::new("1");
    state
        .handle_host_message(
            HostToClient::Response {
                id: 1,
                result: WireResult::Ok {
                    value: HostResponse::ExecutionStarted {
                        cell_id: cell_id.clone(),
                    },
                },
            },
            &outgoing,
        )
        .expect("execution started");
    assert!(matches!(
        started_rx.await.expect("started response"),
        Ok(HostResponse::ExecutionStarted { cell_id: started }) if started == cell_id
    ));

    state
        .handle_host_message(
            HostToClient::CellClosed {
                session_id: "session-1".to_string(),
                cell_id: cell_id.clone(),
            },
            &outgoing,
        )
        .expect("cell closed");
    state.fail("connection stopped");

    assert_eq!(
        delegate.closed.lock().expect("closed cells").as_slice(),
        [cell_id]
    );
}

#[tokio::test]
async fn initial_response_for_the_wrong_cell_fails_the_connection() {
    let state = Arc::new(ConnectionState::new(CancellationToken::new()));
    let delegate = Arc::new(RecordingDelegate::default());
    state.insert_session("session-1".to_string(), delegate.clone());
    state
        .register_cell("session-1", RuntimeCodeModeCellId::new("1"))
        .expect("cell");
    let (initial_tx, initial_rx) = oneshot::channel();
    state
        .register_pending(
            1,
            PendingRequest::ExecuteStarted {
                cell_id: RuntimeCodeModeCellId::new("1"),
                initial: initial_tx,
            },
        )
        .expect("pending initial response");
    let (outgoing, _outgoing_rx) = mpsc::channel::<ClientToHost>(1);
    let error = state
        .handle_host_message(
            HostToClient::InitialResponse {
                id: 1,
                result: WireResult::Ok {
                    value: code_mode_protocol::RuntimeCodeModeResponse::Result {
                        cell_id: RuntimeCodeModeCellId::new("2"),
                        content_items: Vec::new(),
                        error_text: None,
                        code_mode_host_duration: None,
                    },
                },
            },
            &outgoing,
        )
        .expect_err("wrong initial cell must fail");
    assert!(error.contains("after starting"));
    assert!(initial_rx
        .await
        .expect("initial response result")
        .expect_err("wrong cell")
        .contains("after starting"));
    state.fail(error);
    assert_eq!(
        delegate.closed.lock().expect("closed cells").as_slice(),
        [RuntimeCodeModeCellId::new("1")]
    );
}

#[test]
fn connection_failure_closes_each_remaining_cell_once() {
    let state = ConnectionState::new(CancellationToken::new());
    let delegate = Arc::new(RecordingDelegate::default());
    state.insert_session("session-1".to_string(), delegate.clone());
    state
        .register_cell("session-1", RuntimeCodeModeCellId::new("1"))
        .expect("first cell");
    state
        .register_cell("session-1", RuntimeCodeModeCellId::new("2"))
        .expect("second cell");

    state.fail("connection stopped");
    state.fail("duplicate failure");

    let mut closed = delegate.closed.lock().expect("closed cells").clone();
    closed.sort_by(|left, right| left.as_str().cmp(right.as_str()));
    assert_eq!(
        closed,
        [
            RuntimeCodeModeCellId::new("1"),
            RuntimeCodeModeCellId::new("2")
        ]
    );
}

#[test]
fn stale_cell_events_fail_closed_at_the_driver_boundary() {
    let state = Arc::new(ConnectionState::new(CancellationToken::new()));
    let delegate = Arc::new(RecordingDelegate::default());
    state.insert_session("session-1".to_string(), delegate);
    let (outgoing, _outgoing_rx) = mpsc::channel::<ClientToHost>(1);

    let error = state
        .handle_host_message(
            HostToClient::DelegateRequest {
                id: 1,
                session_id: "session-1".to_string(),
                request: DelegateRequest::Notify {
                    tool_call_id: "call-1".to_string(),
                    cell_id: RuntimeCodeModeCellId::new("stale-cell"),
                    text: "late".to_string(),
                },
            },
            &outgoing,
        )
        .expect_err("delegate for an inactive cell must fail closed");
    assert!(error.contains("unknown cell"));

    let error = state
        .handle_host_message(
            HostToClient::CellClosed {
                session_id: "session-1".to_string(),
                cell_id: RuntimeCodeModeCellId::new("stale-cell"),
            },
            &outgoing,
        )
        .expect_err("close for an inactive cell must fail closed");
    assert!(error.contains("unknown cell"));
}

#[derive(Default)]
struct BlockingDelegate {
    started: Notify,
    cancelled: AtomicBool,
}

impl RuntimeCodeModeSessionDelegate for BlockingDelegate {
    fn invoke_tool<'a>(
        &'a self,
        _invocation: RuntimeCodeModeNestedToolCall,
        _cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, Value> {
        Box::pin(async { Ok(Value::Null) })
    }

    fn notify<'a>(
        &'a self,
        _tool_call_id: String,
        _cell_id: RuntimeCodeModeCellId,
        _text: String,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, ()> {
        Box::pin(async move {
            self.started.notify_one();
            cancellation_token.cancelled().await;
            self.cancelled.store(true, Ordering::Release);
            Err("notification cancelled".to_string())
        })
    }

    fn cell_closed(&self, _cell_id: &RuntimeCodeModeCellId) {}
}

#[tokio::test]
async fn closing_a_cell_cancels_its_delegate_callbacks() {
    let state = Arc::new(ConnectionState::new(CancellationToken::new()));
    let delegate = Arc::new(BlockingDelegate::default());
    state.insert_session("session-1".to_string(), delegate.clone());
    let cell_id = RuntimeCodeModeCellId::new("1");
    state
        .register_cell("session-1", cell_id.clone())
        .expect("cell");
    let (outgoing, _outgoing_rx) = mpsc::channel::<ClientToHost>(2);
    let callback_started = delegate.started.notified();
    state
        .handle_host_message(
            HostToClient::DelegateRequest {
                id: 7,
                session_id: "session-1".to_string(),
                request: DelegateRequest::Notify {
                    tool_call_id: "call-1".to_string(),
                    cell_id: cell_id.clone(),
                    text: "progress".to_string(),
                },
            },
            &outgoing,
        )
        .expect("delegate request");
    callback_started.await;
    state
        .handle_host_message(
            HostToClient::CellClosed {
                session_id: "session-1".to_string(),
                cell_id,
            },
            &outgoing,
        )
        .expect("cell close");
    tokio::time::timeout(Duration::from_secs(1), async {
        while !delegate.cancelled.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("delegate callback was not cancelled");
}
