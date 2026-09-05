use super::session::{GrpcSession, Subscriber};
use super::validation::{require_identifier, require_uuid, tool_filters, MAX_IDENTIFIER_BYTES};
use super::waits::WaitControl;
use code_mode_protocol::grpc as proto;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex, Notify};
use tokio_util::sync::CancellationToken;

fn session() -> (
    GrpcSession,
    mpsc::Receiver<Result<proto::SessionEvent, tonic::Status>>,
) {
    let (session_events, events) = mpsc::channel(4);
    (
        GrpcSession {
            id: uuid::Uuid::new_v4().to_string(),
            peer: crate::peer::PeerState::new(),
            closed: std::sync::atomic::AtomicBool::new(false),
            runtime: Mutex::new(None),
            subscribers: Mutex::new(Vec::<Subscriber>::new()),
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
        },
        events,
    )
}

#[test]
fn validation_rejects_empty_oversized_and_non_uuid_identifiers() {
    assert!(require_identifier("", "id").is_err());
    assert!(require_identifier(&"x".repeat(MAX_IDENTIFIER_BYTES + 1), "id").is_err());
    assert!(require_uuid("not-a-uuid", "id").is_err());
    assert!(require_uuid(&uuid::Uuid::new_v4().to_string(), "id").is_ok());
}

#[test]
fn validation_limits_tool_filters_and_namespaces() {
    let valid = proto::ToolName {
        name: "echo".to_string(),
        namespace: Some("tools".to_string()),
    };
    assert!(tool_filters(std::slice::from_ref(&valid)).is_ok());
    assert!(tool_filters(&vec![valid.clone(); 65]).is_err());
    assert!(tool_filters(&[proto::ToolName {
        name: "echo".to_string(),
        namespace: Some("x".repeat(MAX_IDENTIFIER_BYTES + 1)),
    }])
    .is_err());
}

#[tokio::test]
async fn close_pending_fails_callbacks_and_cancels_waits() {
    let (session, mut events) = session();
    let (tool_sender, tool_receiver) = oneshot::channel();
    session
        .pending
        .lock()
        .await
        .insert("tool".to_string(), tool_sender);
    let (notification_sender, notification_receiver) = oneshot::channel();
    session
        .pending_notifications
        .lock()
        .await
        .insert("notice".to_string(), notification_sender);

    let cancellation = CancellationToken::new();
    let retired = Arc::new(Notify::new());
    session.waits.lock().await.insert(
        "wait".to_string(),
        WaitControl {
            cancellation: cancellation.clone(),
            retired,
        },
    );

    session.close_pending("closed").await;

    assert_eq!(
        tool_receiver.await.expect("tool result"),
        Err("closed".to_string())
    );
    assert_eq!(
        notification_receiver.await.expect("notification result"),
        Err("closed".to_string())
    );
    assert!(cancellation.is_cancelled());
    assert!(session.pending.lock().await.is_empty());
    assert!(session.waits.lock().await.is_empty());

    let mut event_ids = std::collections::HashSet::new();
    for _ in 0..2 {
        let event = events
            .recv()
            .await
            .expect("cancellation event")
            .expect("session event");
        match event.event {
            Some(proto::session_event::Event::ToolCallCancelled(cancelled)) => {
                event_ids.insert(("tool", cancelled.invocation_id));
            }
            Some(proto::session_event::Event::NotificationCancelled(cancelled)) => {
                event_ids.insert(("notification", cancelled.notification_id));
            }
            other => panic!("unexpected cancellation event: {other:?}"),
        }
    }
    assert!(event_ids.contains(&("tool", "tool".to_string())));
    assert!(event_ids.contains(&("notification", "notice".to_string())));

    session.close_pending("closed again").await;
    assert!(events.try_recv().is_err());
}
