use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_protocol::{
    ItemId, ItemStatus, SessionId, ThreadId, ThreadItem, ThreadItemPayload, ToolArgument,
    ToolOutput, TurnId,
};
use app_server::{
    run_json_lines, ActionRespondRequest, AppServer, CancelExecutionRequest, ExecutionBackend,
    ExecutionRequest, ProjectionStore, ProviderTurnHistory, RuntimeCore, RuntimeCoreError,
    RuntimeEvent, RuntimeEventSink,
};
use app_server_protocol::{
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_READ, METHOD_THREAD_START,
    METHOD_TURN_START,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::time::timeout;

const REQUEST_METHOD: &str = "item/fileChange/requestApproval";
const RESOLVED_METHOD: &str = "serverRequest/resolved";
const TURN_COMPLETED_METHOD: &str = "turn/completed";
const APPROVAL_ID: &str = "approval-file-change-transport";
const PATCH_ID: &str = "patch-file-change-transport";
const TOOL_ITEM_ID: &str = "tool-file-change-transport";

struct PendingFileChangeBackend {
    response: Mutex<Option<Value>>,
}

#[async_trait]
impl ExecutionBackend for PendingFileChangeBackend {
    fn has_live_session_responses(&self) -> bool {
        true
    }

    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Err(RuntimeCoreError::Backend(
            "file change transport backend requires session input".to_string(),
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
        sink.emit(tool_event(
            &request.session.session_id,
            &request.session.thread_id,
            &request.turn.turn_id,
            ItemStatus::InProgress,
            None,
        ))?;
        sink.emit(RuntimeEvent::new(
            "patch.started",
            json!({
                "patchId": PATCH_ID,
                "status": "proposed",
                "changes": changes(),
            }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "action.required",
            json!({
                "requestId": APPROVAL_ID,
                "actionId": APPROVAL_ID,
                "actionType": "tool_confirmation",
                "availableDecisions": ["allow_once", "decline", "cancel"],
                "toolCallId": PATCH_ID,
                "toolName": "apply_patch",
                "prompt": "Apply the protected file batch?",
                "scope": {
                    "sessionId": request.session.session_id,
                    "threadId": request.session.thread_id,
                    "turnId": request.turn.turn_id,
                },
            }),
        ))?;

        let response = pending_response
            .wait()
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.message))?;
        *self
            .response
            .lock()
            .expect("file change response mutex poisoned") = Some(response);
        sink.emit(RuntimeEvent::new(
            "patch.declined",
            json!({
                "patchId": PATCH_ID,
                "status": "declined",
                "changes": changes(),
            }),
        ))?;
        sink.emit(tool_event(
            &request.session.session_id,
            &request.session.thread_id,
            &request.turn.turn_id,
            ItemStatus::Failed,
            Some(ToolOutput {
                text: None,
                structured_content: Some(json!({
                    "success": false,
                    "reasonCode": "tool_approval_declined",
                })),
                error: Some("patch declined by user".to_string()),
                duration_ms: None,
                truncated: false,
                output_ref: None,
            }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "turn.completed",
            json!({ "reason": "file_change_approval_declined" }),
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
        request: ActionRespondRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new(
            "action.resolved",
            json!({
                "requestId": request.request_id,
                "actionId": APPROVAL_ID,
                "actionType": "tool_confirmation",
                "decision": "decline",
                "confirmed": false,
                "toolCallId": PATCH_ID,
                "toolName": "apply_patch",
            }),
        ))
    }
}

#[tokio::test]
async fn file_change_decline_uses_one_typed_request_and_projects_the_terminal_read_model() {
    let temp = TempDir::new().expect("file change transport temp dir");
    let backend = Arc::new(PendingFileChangeBackend {
        response: Mutex::new(None),
    });
    let runtime = RuntimeCore::with_backend(backend.clone()).with_projection_store(Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("file change transport projection store"),
    ));
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
                    "name": "file-change-transport-jsonrpc-test",
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

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": METHOD_TURN_START,
            "params": {
                "threadId": thread_id,
                "input": [{"type": "text", "text": "decline the file change"}],
                "model": "fixture-model",
                "approvalPolicy": "on-request",
                "sandboxPolicy": "workspace-write"
            }
        }),
    )
    .await;

    let (turn_id, outer_request_id) =
        wait_for_file_change_request(&mut output_lines, &thread_id).await;
    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": outer_request_id,
            "result": { "decision": "decline" }
        }),
    )
    .await;

    let terminal =
        wait_for_decline_terminal(&mut output_lines, &thread_id, &turn_id, &outer_request_id).await;
    assert!(terminal.resolved_before_turn_completed);
    assert_eq!(terminal.typed_request_count, 1);
    assert_eq!(
        *backend
            .response
            .lock()
            .expect("file change response mutex poisoned"),
        Some(json!({
            "confirmed": false,
            "decision": "decline",
            "decisionScope": "once",
        }))
    );

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": METHOD_THREAD_READ,
            "params": { "threadId": thread_id, "includeTurns": true }
        }),
    )
    .await;
    let read = read_response(&mut output_lines, 4).await;
    assert_response_ok(&read, "thread/read after file change decline");
    let turn = read
        .pointer("/result/thread/turns")
        .and_then(Value::as_array)
        .and_then(|turns| {
            turns
                .iter()
                .find(|turn| turn.get("id") == Some(&json!(turn_id)))
        })
        .unwrap_or_else(|| panic!("declined turn missing from read model: {read:#?}"));
    assert_eq!(turn.pointer("/status"), Some(&json!("completed")));
    let file_change = turn
        .get("items")
        .and_then(Value::as_array)
        .and_then(|items| {
            items.iter().find(|item| {
                item.get("id") == Some(&json!(format!("item_{PATCH_ID}")))
                    && item.get("type") == Some(&json!("fileChange"))
            })
        })
        .unwrap_or_else(|| panic!("declined FileChange missing from read model: {turn:#?}"));
    assert_eq!(file_change.get("status"), Some(&json!("declined")));
    assert_eq!(file_change.get("changes"), Some(&projected_changes()));

    drop(input_client);
    timeout(Duration::from_secs(2), runner)
        .await
        .expect("JSONL runner should stop after input closes")
        .expect("JSONL runner task")
        .expect("JSONL runner result");
}

fn tool_event(
    session_id: &str,
    thread_id: &str,
    turn_id: &str,
    status: ItemStatus,
    output: Option<ToolOutput>,
) -> RuntimeEvent {
    let completed_at_ms = status.is_terminal().then_some(2);
    let payload = ThreadItemPayload::Tool {
        call_id: PATCH_ID.to_string(),
        name: "apply_patch".to_string(),
        arguments: vec![ToolArgument {
            name: "changes".to_string(),
            value: changes().to_string(),
        }],
        output,
    };
    RuntimeEvent::new(
        if status.is_terminal() {
            "item.completed"
        } else {
            "item.started"
        },
        json!({
            "item": ThreadItem {
                session_id: SessionId::new(session_id),
                thread_id: ThreadId::new(thread_id),
                turn_id: TurnId::new(turn_id),
                item_id: ItemId::new(TOOL_ITEM_ID),
                sequence: 1,
                ordinal: 1,
                created_at_ms: 1,
                updated_at_ms: completed_at_ms.unwrap_or(1),
                completed_at_ms,
                kind: payload.kind(),
                status,
                payload,
                metadata: json!({}),
            }
        }),
    )
}

fn changes() -> Value {
    json!([
        { "path": "src/added.ts", "kind": "add", "diff": "+added" },
        { "path": "src/deleted.ts", "kind": "delete", "diff": "-deleted" },
        { "path": "src/updated.ts", "kind": "update", "diff": "-old\n+new" },
        {
            "path": "src/move-source.ts",
            "kind": "update",
            "movePath": "src/move-destination.ts",
            "diff": "-source\n+destination",
        },
    ])
}

fn projected_changes() -> Value {
    json!([
        {
            "path": "src/added.ts",
            "kind": { "type": "add" },
            "diff": "+added",
        },
        {
            "path": "src/deleted.ts",
            "kind": { "type": "delete" },
            "diff": "-deleted",
        },
        {
            "path": "src/updated.ts",
            "kind": { "type": "update" },
            "diff": "-old\n+new",
        },
        {
            "path": "src/move-source.ts",
            "kind": {
                "type": "update",
                "move_path": "src/move-destination.ts",
            },
            "diff": "-source\n+destination",
        },
    ])
}

struct DeclineTerminal {
    resolved_before_turn_completed: bool,
    typed_request_count: usize,
}

async fn wait_for_file_change_request(
    lines: &mut Lines<BufReader<DuplexStream>>,
    thread_id: &str,
) -> (String, Value) {
    let mut turn_id = None;
    let mut outer_request_id = None;
    for _ in 0..48 {
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
            assert_eq!(message.pointer("/params/itemId"), Some(&json!(PATCH_ID)));
            let request_turn_id = required_string(&message, "/params/turnId", "request turn id");
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
    panic!("did not receive admitted turn and typed FileChange request");
}

async fn wait_for_decline_terminal(
    lines: &mut Lines<BufReader<DuplexStream>>,
    thread_id: &str,
    turn_id: &str,
    outer_request_id: &Value,
) -> DeclineTerminal {
    let mut resolved_index = None;
    let mut turn_completed_index = None;
    let mut typed_request_count = 1;
    for index in 0..64 {
        let message = next_message(lines).await;
        if message.get("method") == Some(&json!(REQUEST_METHOD)) {
            typed_request_count += 1;
        }
        if message.get("method") == Some(&json!(RESOLVED_METHOD)) {
            assert_eq!(message.pointer("/params/threadId"), Some(&json!(thread_id)));
            assert_eq!(message.pointer("/params/requestId"), Some(outer_request_id));
            resolved_index = Some(index);
        }
        if message.get("method") == Some(&json!(TURN_COMPLETED_METHOD))
            && message.pointer("/params/turn/id") == Some(&json!(turn_id))
        {
            assert_eq!(
                message.pointer("/params/turn/status"),
                Some(&json!("completed"))
            );
            turn_completed_index = Some(index);
        }
        if let (Some(resolved_index), Some(turn_completed_index)) =
            (resolved_index, turn_completed_index)
        {
            return DeclineTerminal {
                resolved_before_turn_completed: resolved_index < turn_completed_index,
                typed_request_count,
            };
        }
    }
    panic!("file change decline did not publish resolved and turn terminal");
}

async fn write_message(client: &mut DuplexStream, message: Value) {
    client
        .write_all(format!("{message}\n").as_bytes())
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
