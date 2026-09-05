use super::routing::publish_tool_call;
use super::session::{GrpcSession, Subscriber};
use code_mode_protocol::grpc as proto;
use code_mode_protocol::{RuntimeCodeModeCellId, RuntimeCodeModeNestedToolCall};
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use tokio::sync::{mpsc, Mutex};
use tokio_util::sync::CancellationToken;

fn session() -> GrpcSession {
    let (session_events, _events) = mpsc::channel(1);
    GrpcSession {
        id: uuid::Uuid::new_v4().to_string(),
        peer: crate::peer::PeerState::new(),
        closed: std::sync::atomic::AtomicBool::new(false),
        runtime: Mutex::new(None),
        subscribers: Mutex::new(Vec::new()),
        session_events: Mutex::new(Some(session_events)),
        event_shutdown: CancellationToken::new(),
        pending: Mutex::new(HashMap::new()),
        pending_notifications: Mutex::new(HashMap::new()),
        next_subscriber: AtomicU64::new(0),
        waits: Mutex::new(HashMap::new()),
        cancelled_waits: Mutex::new(std::collections::HashSet::new()),
        pending_executions: Mutex::new(std::collections::HashSet::new()),
        seen_executions: Mutex::new(Default::default()),
        execution_ids: Mutex::new(HashMap::new()),
        tool_call_sequences: Mutex::new(HashMap::new()),
        active_cells: Mutex::new(HashMap::new()),
    }
}

#[tokio::test]
async fn publishes_to_one_matching_subscription_with_exact_filtering() {
    let session = std::sync::Arc::new(session());
    let (wrong_sender, mut wrong_receiver) = mpsc::channel(1);
    let (matching_sender, mut matching_receiver) = mpsc::channel(1);
    session.subscribers.lock().await.extend([
        Subscriber {
            tool_names: vec![proto::ToolName {
                name: "other".to_string(),
                namespace: None,
            }],
            sender: wrong_sender,
        },
        Subscriber {
            tool_names: vec![proto::ToolName {
                name: "echo".to_string(),
                namespace: None,
            }],
            sender: matching_sender,
        },
    ]);
    let permit = std::sync::Arc::new(tokio::sync::Semaphore::new(1))
        .try_acquire_owned()
        .expect("cell permit");
    session
        .reserve_execution("execution")
        .await
        .expect("reserve execution");
    session
        .register_execution("cell", "execution", permit)
        .await
        .expect("register execution");
    let cancellation = CancellationToken::new();
    let invocation = RuntimeCodeModeNestedToolCall {
        cell_id: RuntimeCodeModeCellId::new("cell"),
        runtime_tool_call_id: "runtime-call".to_string(),
        tool_name: "echo".to_string(),
        kind: code_mode_protocol::CodeModeToolKind::Function,
        input: None,
    };
    let session_id = session.id.clone();
    let routing_session = session.clone();
    let task =
        tokio::spawn(
            async move { publish_tool_call(&routing_session, invocation, cancellation).await },
        );
    let call = matching_receiver
        .recv()
        .await
        .expect("matching call")
        .expect("call");
    assert_eq!(call.tool_name.expect("tool name").name, "echo");
    assert!(wrong_receiver.try_recv().is_err());
    assert_eq!(call.session_id, session_id);
    assert_eq!(call.execution_id, "execution");
    let invocation_id = call.invocation_id.clone();
    let sender = session
        .pending
        .lock()
        .await
        .remove(&invocation_id)
        .expect("pending call");
    sender
        .send(Ok(serde_json::json!({"ok": true})))
        .expect("complete call");
    assert_eq!(
        task.await.expect("routing task").expect("routing result"),
        serde_json::json!({"ok": true})
    );
}
