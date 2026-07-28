use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_protocol::{
    ItemId, ItemStatus, SessionId, ThreadId, ThreadItem, ThreadItemPayload, TurnId,
};
use app_server::{
    run_json_lines, ActionRespondRequest, AppServer, CancelExecutionRequest, ExecutionBackend,
    ExecutionRequest, ProjectionStore, ProviderTurnHistory, RuntimeCore, RuntimeCoreError,
    RuntimeEvent, RuntimeEventSink,
};
use app_server_protocol::{
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_READ, METHOD_THREAD_START,
    METHOD_TURN_INTERRUPT, METHOD_TURN_START,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::time::timeout;

const REQUEST_METHOD: &str = "item/commandExecution/requestApproval";
const RESOLVED_METHOD: &str = "serverRequest/resolved";
const TURN_COMPLETED_METHOD: &str = "turn/completed";
const ITEM_COMPLETED_METHOD: &str = "item/completed";
const APPROVAL_ID: &str = "approval-host-interrupt";
const TOOL_CALL_ID: &str = "tool-host-interrupt";

struct PendingApprovalBackend {
    respond_calls: AtomicUsize,
}

#[async_trait]
impl ExecutionBackend for PendingApprovalBackend {
    fn has_live_session_responses(&self) -> bool {
        true
    }

    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Err(RuntimeCoreError::Backend(
            "pending approval backend requires session input".to_string(),
        ))
    }

    async fn start_turn_with_provider_history_and_session_input(
        &self,
        request: ExecutionRequest,
        _provider_history: ProviderTurnHistory,
        pending_input: Option<agent_runtime::session_loop::RuntimeSessionInputHandle>,
        _cancellation_token: Option<tokio_util::sync::CancellationToken>,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let pending_input = pending_input.ok_or_else(|| {
            RuntimeCoreError::Backend("session response owner is required".to_string())
        })?;
        let pending_response = pending_input
            .register_response(
                agent_runtime::session_loop::RuntimeSessionResponseKind::Approval,
                APPROVAL_ID,
            )
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.message))?;
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(canonical_tool_started_event(
            &request.session.session_id,
            &request.session.thread_id,
            &request.turn.turn_id,
        ))?;
        sink.emit(RuntimeEvent::new(
            "action.required",
            json!({
                "requestId": APPROVAL_ID,
                "actionId": APPROVAL_ID,
                "actionType": "tool_confirmation",
                "actionKind": "tool_execution_policy",
                "availableDecisions": ["allow_once", "decline", "cancel"],
                "toolCallId": TOOL_CALL_ID,
                "toolName": "Bash",
                "prompt": "Allow the pending command?",
                "scope": {
                    "sessionId": request.session.session_id,
                    "threadId": request.session.thread_id,
                    "turnId": request.turn.turn_id,
                },
            }),
        ))?;
        pending_response
            .wait()
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.message))?;
        Err(RuntimeCoreError::Backend(
            "host interrupt must end the waiting task without an action response".to_string(),
        ))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.respond_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
async fn host_interrupt_aborts_pending_approval_before_terminal_and_rejects_late_response() {
    let temp = TempDir::new().expect("host interrupt transport temp dir");
    let backend = Arc::new(PendingApprovalBackend {
        respond_calls: AtomicUsize::new(0),
    });
    let runtime = RuntimeCore::with_backend(backend.clone()).with_projection_store(Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("host interrupt projection store"),
    ));
    let runtime_for_assertions = runtime.clone();
    let server = AppServer::with_runtime(runtime);

    let (mut input_client, input_server) = tokio::io::duplex(16 * 1024);
    let (output_server, output_client) = tokio::io::duplex(16 * 1024);
    let runner = tokio::spawn(run_json_lines(server, input_server, output_server));
    let mut output_lines = BufReader::new(output_client).lines();

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": METHOD_INITIALIZE,
            "params": {
                "clientInfo": {
                    "name": "host-interrupt-transport-jsonrpc-test",
                    "version": "1.0.0"
                }
            }
        }),
    )
    .await;
    assert_response_ok(&read_response(&mut output_lines, 1).await, "initialize");
    write_message(
        &mut input_client,
        json!({ "jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {} }),
    )
    .await;

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": METHOD_THREAD_START,
            "params": {
                "model": "fixture-model",
                "modelProvider": "fixture-provider",
                "cwd": temp.path()
            }
        }),
    )
    .await;
    let thread_start = read_response(&mut output_lines, 2).await;
    assert_response_ok(&thread_start, "thread/start");
    let thread_id = required_string(&thread_start, "/result/thread/id", "thread/start thread id");
    let session_id = required_string(
        &thread_start,
        "/result/thread/sessionId",
        "thread/start session id",
    );

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": METHOD_TURN_START,
            "params": {
                "threadId": thread_id,
                "input": [{"type": "text", "text": "wait for approval"}],
                "model": "fixture-model",
                "approvalPolicy": "on-request",
                "sandboxPolicy": "workspace-write"
            }
        }),
    )
    .await;

    let (turn_id, outer_request_id) =
        wait_for_pending_approval(&mut output_lines, &thread_id).await;

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": METHOD_TURN_INTERRUPT,
            "params": { "threadId": thread_id, "turnId": turn_id }
        }),
    )
    .await;

    let terminal = wait_for_host_interrupt_terminal(
        &mut output_lines,
        &thread_id,
        &turn_id,
        &outer_request_id,
    )
    .await;
    assert_eq!(terminal.resolved_before_item_completed, true);
    assert_eq!(terminal.resolved_before_turn_completed, true);
    assert_eq!(backend.respond_calls.load(Ordering::SeqCst), 0);

    let canonical_events = runtime_for_assertions
        .events_for_session(&session_id)
        .expect("canonical host interrupt events");
    let action_canceled_index = canonical_events
        .iter()
        .position(|event| event.event_type == "action.canceled")
        .expect("canonical action.canceled event");
    let item_canceled_index = canonical_events
        .iter()
        .position(|event| {
            event.event_type == "item.completed"
                && event.payload.pointer("/item/status") == Some(&json!("cancelled"))
        })
        .expect("canonical cancelled item.completed event");
    let turn_canceled_index = canonical_events
        .iter()
        .position(|event| event.event_type == "turn.canceled")
        .expect("canonical turn.canceled event");
    assert!(action_canceled_index < item_canceled_index);
    assert!(item_canceled_index < turn_canceled_index);

    // The input order makes the following read a completion barrier for the late response.
    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": outer_request_id,
            "result": { "decision": "accept" }
        }),
    )
    .await;
    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": METHOD_THREAD_READ,
            "params": { "threadId": thread_id, "includeTurns": true }
        }),
    )
    .await;
    let read = read_response(&mut output_lines, 5).await;
    assert_response_ok(&read, "thread/read after host interrupt");
    let turns = read
        .pointer("/result/thread/turns")
        .and_then(Value::as_array)
        .expect("thread/read turns array");
    let turn = turns
        .iter()
        .find(|turn| turn.get("id") == Some(&json!(turn_id)))
        .unwrap_or_else(|| panic!("interrupted turn missing from read model: {read:#?}"));
    assert!(
        matches!(turn.get("status"), Some(Value::String(status)) if status == "interrupted" || status == "canceled"),
        "host interrupt must project a terminal interrupted turn: {turn:#?}"
    );
    assert_eq!(backend.respond_calls.load(Ordering::SeqCst), 0);

    drop(input_client);
    timeout(Duration::from_secs(2), runner)
        .await
        .expect("JSONL runner should stop after input closes")
        .expect("JSONL runner task")
        .expect("JSONL runner result");
}

fn canonical_tool_started_event(session_id: &str, thread_id: &str, turn_id: &str) -> RuntimeEvent {
    let payload = ThreadItemPayload::Tool {
        call_id: TOOL_CALL_ID.to_string(),
        name: "Bash".to_string(),
        arguments: Vec::new(),
        output: None,
    };
    RuntimeEvent::new(
        "item.started",
        json!({
            "item": ThreadItem {
                session_id: SessionId::new(session_id),
                thread_id: ThreadId::new(thread_id),
                turn_id: TurnId::new(turn_id),
                item_id: ItemId::new(format!("item_{TOOL_CALL_ID}")),
                sequence: 1,
                ordinal: 1,
                created_at_ms: 1,
                updated_at_ms: 1,
                completed_at_ms: None,
                kind: payload.kind(),
                status: ItemStatus::InProgress,
                payload,
                metadata: json!({}),
            }
        }),
    )
}

struct InterruptTerminal {
    resolved_before_item_completed: bool,
    resolved_before_turn_completed: bool,
}

async fn wait_for_pending_approval(
    lines: &mut Lines<BufReader<DuplexStream>>,
    thread_id: &str,
) -> (String, Value) {
    let mut turn_id = None;
    let mut outer_request_id = None;
    for _ in 0..32 {
        let message = next_message(lines).await;
        if message.get("id") == Some(&json!(3)) {
            assert_response_ok(&message, "turn/start");
            turn_id = Some(required_string(
                &message,
                "/result/turn/id",
                "turn/start turn id",
            ));
        }
        if message.get("method") == Some(&json!(REQUEST_METHOD)) {
            assert_eq!(message.pointer("/params/threadId"), Some(&json!(thread_id)));
            let request_turn_id = required_string(&message, "/params/turnId", "approval turn id");
            if let Some(turn_id) = turn_id.as_ref() {
                assert_eq!(request_turn_id, *turn_id);
            }
            outer_request_id = Some(
                message
                    .get("id")
                    .cloned()
                    .unwrap_or_else(|| panic!("reverse request has no outer id: {message:#?}")),
            );
        }
        if let (Some(turn_id), Some(outer_request_id)) = (turn_id.clone(), outer_request_id.clone())
        {
            return (turn_id, outer_request_id);
        }
    }
    panic!("did not receive admitted turn and typed approval request");
}

async fn wait_for_host_interrupt_terminal(
    lines: &mut Lines<BufReader<DuplexStream>>,
    thread_id: &str,
    turn_id: &str,
    outer_request_id: &Value,
) -> InterruptTerminal {
    let mut resolved_index = None;
    let mut item_completed_index = None;
    let mut turn_completed_index = None;
    let mut interrupt_response_seen = false;
    for index in 0..48 {
        let message = next_message(lines).await;
        if message.get("id") == Some(&json!(4)) {
            assert_response_ok(&message, "turn/interrupt");
            interrupt_response_seen = true;
        }
        if message.get("method") == Some(&json!(RESOLVED_METHOD)) {
            assert_eq!(message.pointer("/params/threadId"), Some(&json!(thread_id)));
            assert_eq!(message.pointer("/params/requestId"), Some(outer_request_id));
            resolved_index = Some(index);
        }
        if message.get("method") == Some(&json!(ITEM_COMPLETED_METHOD))
            && message.pointer("/params/item/id") == Some(&json!(format!("item_{TOOL_CALL_ID}")))
        {
            assert_eq!(
                message.pointer("/params/item/status"),
                Some(&json!("failed")),
                "cancelled dynamic tool lowers to failed in the v2 wire enum: {message:#?}"
            );
            item_completed_index = Some(index);
        }
        if message.get("method") == Some(&json!(TURN_COMPLETED_METHOD))
            && message.pointer("/params/turn/id") == Some(&json!(turn_id))
        {
            turn_completed_index = Some(index);
        }
        if interrupt_response_seen
            && resolved_index.is_some()
            && item_completed_index.is_some()
            && turn_completed_index.is_some()
        {
            let resolved_index = resolved_index.expect("resolved index");
            return InterruptTerminal {
                resolved_before_item_completed: resolved_index
                    < item_completed_index.expect("item completed index"),
                resolved_before_turn_completed: resolved_index
                    < turn_completed_index.expect("turn completed index"),
            };
        }
    }
    panic!("host interrupt did not publish resolved, item completion, and turn terminal");
}

async fn write_message(client: &mut DuplexStream, message: Value) {
    client
        .write_all(format!("{}\n", message).as_bytes())
        .await
        .expect("write JSON-RPC message");
    client.flush().await.expect("flush JSON-RPC message");
}

async fn read_response(lines: &mut Lines<BufReader<DuplexStream>>, id: u64) -> Value {
    for _ in 0..64 {
        let message = next_message(lines).await;
        if message.get("id") == Some(&json!(id)) {
            return message;
        }
    }
    panic!("missing JSON-RPC response id {id}");
}

async fn next_message(lines: &mut Lines<BufReader<DuplexStream>>) -> Value {
    let line = timeout(Duration::from_secs(2), lines.next_line())
        .await
        .expect("timed out waiting for JSON-RPC output")
        .expect("read JSON-RPC output")
        .expect("JSONL output closed");
    serde_json::from_str(&line)
        .unwrap_or_else(|error| panic!("invalid JSON-RPC output {line:?}: {error}"))
}

fn assert_response_ok(response: &Value, context: &str) {
    assert!(
        response.get("error").is_none(),
        "{context} failed: {response:#?}"
    );
}

fn required_string(message: &Value, pointer: &str, context: &str) -> String {
    message
        .pointer(pointer)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{context} missing from {message:#?}"))
        .to_string()
}
