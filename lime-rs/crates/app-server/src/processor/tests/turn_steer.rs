use super::super::*;
use super::tests_support::initialize_processor;
use crate::{
    ActionRespondRequest, CancelExecutionRequest, ExecutionBackend, ExecutionRequest,
    ProjectionStore, RuntimeCore, RuntimeCoreError, RuntimeEvent, RuntimeEventSink,
};
use app_server_protocol::protocol::v2::METHOD_TURN_STEER;
use app_server_protocol::{
    error_codes, AgentSessionReadParams, AgentSessionStartParams, JsonRpcMessage, JsonRpcRequest,
    RequestId, METHOD_THREAD_READ, METHOD_TURN_INTERRUPT, METHOD_TURN_START,
};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Notify;
use tokio::time::{timeout, Duration};

const INLINE_PNG_DATA_URL: &str =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";

struct BlockingTurnBackend {
    started: Arc<Notify>,
    release: Arc<Notify>,
}

#[async_trait::async_trait]
impl ExecutionBackend for BlockingTurnBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        self.started.notify_one();
        self.release.notified().await;
        sink.emit(RuntimeEvent::new("turn.completed", json!({})))
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
        Ok(())
    }
}

fn start_session(runtime: &RuntimeCore, session_id: &str, thread_id: &str) {
    runtime
        .start_session(AgentSessionStartParams {
            session_id: Some(session_id.to_string()),
            thread_id: Some(thread_id.to_string()),
            app_id: "agent-chat".to_string(),
            workspace_id: Some("workspace-current".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("start session");
}

async fn active_turn_id(processor: &RequestProcessor, session_id: &str) -> String {
    processor
        .runtime()
        .read_session_current(AgentSessionReadParams {
            session_id: session_id.to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read active session")
        .turns
        .into_iter()
        .find(|turn| {
            matches!(
                turn.status,
                app_server_protocol::AgentTurnStatus::Accepted
                    | app_server_protocol::AgentTurnStatus::Running
                    | app_server_protocol::AgentTurnStatus::WaitingAction
            )
        })
        .expect("active turn")
        .turn_id
}

#[tokio::test]
async fn turn_steer_without_active_turn_fails_without_creating_a_turn() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let runtime = RuntimeCore::default().with_projection_store(store);
    start_session(&runtime, "session-steer-empty", "thread-steer-empty");
    let processor = RequestProcessor::new(runtime);
    initialize_processor(&processor).await;

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(2),
            METHOD_TURN_STEER,
            Some(json!({
                "threadId": "thread-steer-empty",
                "expectedTurnId": "turn-missing",
                "input": [{"type": "text", "text": "follow up"}]
            })),
        ))
        .await
        .expect("steer request");
    let [JsonRpcMessage::Error(error)] = messages.as_slice() else {
        panic!("expected turn/steer error, got {messages:?}");
    };
    assert_eq!(error.error.code, error_codes::INVALID_REQUEST);
    assert!(error.error.message.contains("active") || error.error.message.contains("turn"));

    // The atomic runtime precondition must fail before any replacement turn is admitted.
    let read_messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(3),
            METHOD_THREAD_READ,
            Some(json!({"threadId": "thread-steer-empty"})),
        ))
        .await
        .expect("thread/read request");
    let [JsonRpcMessage::Response(response)] = read_messages.as_slice() else {
        panic!("expected thread/read response, got {read_messages:?}");
    };
    assert_eq!(response.result["thread"]["turns"], json!([]));
}

#[tokio::test]
async fn turn_steer_dispatches_v2_input_to_the_active_turn() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let runtime = RuntimeCore::with_backend(Arc::new(BlockingTurnBackend {
        started: started.clone(),
        release: release.clone(),
    }))
    .with_projection_store(store)
    .with_sidecar_store(Arc::new(
        crate::SidecarStore::new(temp.path().join("sidecar")).expect("sidecar store"),
    ));
    start_session(&runtime, "session-steer-active", "thread-steer-active");
    let processor = RequestProcessor::new(runtime);
    initialize_processor(&processor).await;

    let turn_processor = processor.clone();
    let turn = tokio::spawn(async move {
        turn_processor
            .handle_request(JsonRpcRequest::new(
                RequestId::Integer(2),
                METHOD_TURN_START,
                Some(json!({
                    "threadId": "thread-steer-active",
                    "input": [{"type": "text", "text": "initial"}]
                })),
            ))
            .await
    });
    timeout(Duration::from_secs(1), started.notified())
        .await
        .expect("backend should observe active turn");
    let active_turn_id = active_turn_id(&processor, "session-steer-active").await;

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(3),
            METHOD_TURN_STEER,
            Some(json!({
                "threadId": "thread-steer-active",
                "expectedTurnId": active_turn_id,
                "clientUserMessageId": "client-steer-1",
                "input": [
                    {"type": "text", "text": "clarify"},
                    {"type": "text", "text": "the result"},
                    {"type": "image", "url": INLINE_PNG_DATA_URL, "detail": "auto"}
                ]
            })),
        ))
        .await
        .expect("steer request");
    let response = messages
        .iter()
        .find_map(|message| match message {
            JsonRpcMessage::Response(response) => Some(response),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected steer response, got {messages:?}"));
    let notification = messages
        .iter()
        .find_map(|message| match message {
            JsonRpcMessage::Notification(notification)
                if notification
                    .params
                    .as_ref()
                    .and_then(|params| params["event"]["type"].as_str())
                    == Some("message.created") =>
            {
                Some(notification)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected message.created notification, got {messages:?}"));
    assert_eq!(response.result["turnId"], active_turn_id);
    assert_eq!(notification.method, "agentSession/event");
    let event = &notification.params.as_ref().expect("event params")["event"];
    assert_eq!(event["type"], "message.created");
    assert_eq!(event["payload"]["source"], "turn/steer");
    assert_eq!(event["payload"]["clientId"], "client-steer-1");
    assert_eq!(event["payload"]["content"]["text"], "clarify\nthe result");
    let durable_uri = event["payload"]["input"][2]["uri"]
        .as_str()
        .expect("steer canonical image URI");
    assert!(durable_uri.starts_with("sidecar://media/input-"));
    assert_eq!(event["payload"]["outputRefs"][0]["ref"], durable_uri);
    assert!(
        !serde_json::to_string(event)
            .expect("serialize steer event")
            .contains("base64,"),
        "turn/steer event leaked inline media"
    );

    let read_messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(4),
            METHOD_THREAD_READ,
            Some(json!({
                "threadId": "thread-steer-active",
                "includeTurns": true
            })),
        ))
        .await
        .expect("thread/read after steer");
    let read_response = read_messages
        .iter()
        .find_map(|message| match message {
            JsonRpcMessage::Response(response) => Some(response),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected thread/read response, got {read_messages:?}"));
    let read_json = serde_json::to_string(&read_response.result).expect("serialize thread/read");
    assert!(
        !read_json.contains("base64,"),
        "thread/read leaked inline media"
    );
    assert!(read_json.contains(durable_uri));

    let interrupt = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(5),
            METHOD_TURN_INTERRUPT,
            Some(json!({
                "threadId": "thread-steer-active",
                "turnId": active_turn_id
            })),
        ))
        .await
        .expect("interrupt request");
    let interrupt_response = interrupt
        .iter()
        .find_map(|message| match message {
            JsonRpcMessage::Response(response) => Some(response),
            _ => None,
        })
        .expect("turn/interrupt response");
    assert_eq!(interrupt_response.result, json!({}));

    release.notify_one();
    let start_messages = timeout(Duration::from_secs(1), turn)
        .await
        .expect("turn should finish")
        .expect("turn task")
        .expect("turn request");
    let start_response = start_messages
        .iter()
        .find_map(|message| match message {
            JsonRpcMessage::Response(response) => Some(response),
            _ => None,
        })
        .expect("turn/start response");
    assert_eq!(start_response.result["turn"]["id"], active_turn_id);
    assert_eq!(start_response.result["turn"]["status"], "inProgress");
}

#[tokio::test]
async fn turn_interrupt_rejects_terminal_turn_before_abort_hook() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let runtime = RuntimeCore::with_backend(Arc::new(BlockingTurnBackend {
        started: started.clone(),
        release: release.clone(),
    }))
    .with_projection_store(store);
    start_session(
        &runtime,
        "session-interrupt-terminal",
        "thread-interrupt-terminal",
    );
    let hook_calls = Arc::new(AtomicUsize::new(0));
    let hook_calls_for_hook = hook_calls.clone();
    let processor = RequestProcessor::new(runtime).with_turn_interrupt_hook(Arc::new(
        move |_thread_id, _turn_id| {
            let hook_calls = hook_calls_for_hook.clone();
            Box::pin(async move {
                hook_calls.fetch_add(1, Ordering::SeqCst);
            })
        },
    ));
    initialize_processor(&processor).await;

    let turn_processor = processor.clone();
    let turn = tokio::spawn(async move {
        turn_processor
            .handle_request(JsonRpcRequest::new(
                RequestId::Integer(2),
                METHOD_TURN_START,
                Some(json!({
                    "threadId": "thread-interrupt-terminal",
                    "input": [{"type": "text", "text": "finish first"}]
                })),
            ))
            .await
    });
    timeout(Duration::from_secs(1), started.notified())
        .await
        .expect("backend should observe active turn");
    let turn_id = active_turn_id(&processor, "session-interrupt-terminal").await;
    release.notify_one();
    timeout(Duration::from_secs(1), turn)
        .await
        .expect("turn should finish")
        .expect("turn task")
        .expect("turn request");
    timeout(Duration::from_secs(1), async {
        loop {
            let read = processor
                .runtime()
                .read_session_current(AgentSessionReadParams {
                    session_id: "session-interrupt-terminal".to_string(),
                    history_limit: None,
                    history_offset: None,
                    history_before_message_id: None,
                })
                .await
                .expect("read terminal session");
            if read.turns.iter().any(|candidate| {
                candidate.turn_id == turn_id
                    && candidate.status == app_server_protocol::AgentTurnStatus::Completed
            }) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("turn should be terminal before interrupt");

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(3),
            METHOD_TURN_INTERRUPT,
            Some(json!({
                "threadId": "thread-interrupt-terminal",
                "turnId": turn_id
            })),
        ))
        .await
        .expect("interrupt request");
    let [JsonRpcMessage::Error(error)] = messages.as_slice() else {
        panic!("expected turn/interrupt error, got {messages:?}");
    };
    assert_eq!(error.error.code, error_codes::INVALID_REQUEST);
    assert_eq!(hook_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn turn_steer_rejects_expected_turn_mismatch_without_creating_a_turn() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let runtime = RuntimeCore::with_backend(Arc::new(BlockingTurnBackend {
        started: started.clone(),
        release: release.clone(),
    }))
    .with_projection_store(store);
    start_session(&runtime, "session-steer-mismatch", "thread-steer-mismatch");
    let processor = RequestProcessor::new(runtime);
    initialize_processor(&processor).await;

    let turn_processor = processor.clone();
    let turn = tokio::spawn(async move {
        turn_processor
            .handle_request(JsonRpcRequest::new(
                RequestId::Integer(2),
                METHOD_TURN_START,
                Some(json!({
                    "threadId": "thread-steer-mismatch",
                    "input": [{"type": "text", "text": "initial"}]
                })),
            ))
            .await
    });
    timeout(Duration::from_secs(1), started.notified())
        .await
        .expect("backend should observe active turn");
    let active_turn_id = active_turn_id(&processor, "session-steer-mismatch").await;

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(3),
            METHOD_TURN_STEER,
            Some(json!({
                "threadId": "thread-steer-mismatch",
                "expectedTurnId": "turn-wrong",
                "input": [{"type": "text", "text": "must not become a new turn"}]
            })),
        ))
        .await
        .expect("steer mismatch request");
    let [JsonRpcMessage::Error(error)] = messages.as_slice() else {
        panic!("expected turn/steer error, got {messages:?}");
    };
    assert_eq!(error.error.code, error_codes::INVALID_REQUEST);
    assert!(error.error.message.contains("turn-wrong"));

    let read = processor
        .runtime()
        .read_session_current(AgentSessionReadParams {
            session_id: "session-steer-mismatch".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read active session");
    assert_eq!(read.turns.len(), 1);
    assert_eq!(read.turns[0].turn_id, active_turn_id);

    release.notify_one();
    timeout(Duration::from_secs(1), turn)
        .await
        .expect("turn should finish")
        .expect("turn task")
        .expect("turn request");
}
