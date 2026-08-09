use super::super::*;
use super::tests_support::initialize_processor;
use crate::{
    ActionRespondRequest, CancelExecutionRequest, ExecutionBackend, ExecutionRequest, MockBackend,
    ProjectionStore, RuntimeCore, RuntimeCoreError, RuntimeEvent, RuntimeEventSink,
};
use app_server_protocol::protocol::v2::{ReviewDelivery, ReviewTarget, METHOD_REVIEW_START};
use app_server_protocol::{
    error_codes, AgentSessionStartParams, AgentTurnStatus, JsonRpcMessage, JsonRpcRequest,
    RequestId,
};
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Notify;
use tokio::time::{timeout, Duration};

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

#[tokio::test]
async fn review_start_returns_canonical_turn_and_thread_id() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend)).with_projection_store(store);
    start_session(&runtime, "session-review", "thread-review");
    let processor = RequestProcessor::new(runtime);
    initialize_processor(&processor).await;

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(2),
            METHOD_REVIEW_START,
            Some(json!({
                "threadId": "thread-review",
                "delivery": "inline",
                "target": {
                    "type": "commit",
                    "sha": " abc123 ",
                    "title": " Tidy colors "
                }
            })),
        ))
        .await
        .expect("review/start request");
    let response = messages
        .iter()
        .find_map(|message| match message {
            JsonRpcMessage::Response(response) => Some(response),
            _ => None,
        })
        .expect("review/start response");

    assert_eq!(response.result["reviewThreadId"], "thread-review");
    assert_eq!(response.result["turn"]["status"], "inProgress");
    assert!(response.result["turn"]["id"].as_str().is_some());
}

#[tokio::test]
async fn review_start_rejects_empty_target_fields_and_detached_delivery() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend)).with_projection_store(store);
    start_session(&runtime, "session-review-errors", "thread-review-errors");
    let processor = RequestProcessor::new(runtime);
    initialize_processor(&processor).await;

    for (id, target, expected) in [
        (
            2,
            json!({"type": "baseBranch", "branch": " "}),
            "branch must not be empty",
        ),
        (
            3,
            json!({"type": "commit", "sha": " "}),
            "sha must not be empty",
        ),
        (
            4,
            json!({"type": "custom", "instructions": " "}),
            "instructions must not be empty",
        ),
    ] {
        let messages = processor
            .handle_request(JsonRpcRequest::new(
                RequestId::Integer(id),
                METHOD_REVIEW_START,
                Some(json!({
                    "threadId": "thread-review-errors",
                    "target": target
                })),
            ))
            .await
            .expect("review/start request");
        let [JsonRpcMessage::Error(error)] = messages.as_slice() else {
            panic!("expected review/start error, got {messages:?}");
        };
        assert_eq!(error.error.code, error_codes::INVALID_REQUEST);
        assert!(error.error.message.contains(expected));
    }

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(5),
            METHOD_REVIEW_START,
            Some(json!({
                "threadId": "thread-review-errors",
                "delivery": "detached",
                "target": {"type": "uncommittedChanges"}
            })),
        ))
        .await
        .expect("review/start detached request");
    let [JsonRpcMessage::Error(error)] = messages.as_slice() else {
        panic!("expected detached review error, got {messages:?}");
    };
    assert_eq!(error.error.code, error_codes::INVALID_REQUEST);
    assert!(error.error.message.contains("detached"));
}

struct BlockingReviewBackend {
    started: Arc<Notify>,
    release: Arc<Notify>,
}

#[async_trait]
impl ExecutionBackend for BlockingReviewBackend {
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

#[tokio::test]
async fn review_start_rejects_an_active_turn() {
    let temp = tempfile::tempdir().expect("tempdir");
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let runtime = RuntimeCore::with_backend(Arc::new(BlockingReviewBackend {
        started: started.clone(),
        release: release.clone(),
    }))
    .with_projection_store(Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    ));
    start_session(&runtime, "session-review-active", "thread-review-active");
    let processor = RequestProcessor::new(runtime);
    initialize_processor(&processor).await;

    let turn_processor = processor.clone();
    let turn = tokio::spawn(async move {
        turn_processor
            .handle_request(JsonRpcRequest::new(
                RequestId::Integer(2),
                app_server_protocol::METHOD_TURN_START,
                Some(json!({
                    "threadId": "thread-review-active",
                    "input": [{"type": "text", "text": "initial"}]
                })),
            ))
            .await
    });
    timeout(Duration::from_secs(1), started.notified())
        .await
        .expect("backend should observe active turn");

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(3),
            METHOD_REVIEW_START,
            Some(json!({
                "threadId": "thread-review-active",
                "target": {"type": "uncommittedChanges"}
            })),
        ))
        .await
        .expect("review/start active request");
    let [JsonRpcMessage::Error(error)] = messages.as_slice() else {
        panic!("expected active review error, got {messages:?}");
    };
    assert_eq!(error.error.code, error_codes::TURN_ALREADY_ACTIVE);
    assert!(error.error.message.contains("active"));

    release.notify_one();
    timeout(Duration::from_secs(1), turn)
        .await
        .expect("turn should finish")
        .expect("turn task")
        .expect("turn request");
}

#[tokio::test]
async fn runtime_review_persists_normalized_boundaries_around_the_turn() {
    let temp = tempfile::tempdir().expect("tempdir");
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let runtime = RuntimeCore::with_backend(Arc::new(BlockingReviewBackend {
        started: started.clone(),
        release: release.clone(),
    }))
    .with_projection_store(Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    ));
    start_session(
        &runtime,
        "session-review-lifecycle",
        "thread-review-lifecycle",
    );

    let output = runtime
        .start_review(
            "thread-review-lifecycle",
            ReviewTarget::Commit {
                sha: " abc123 ".to_string(),
                title: Some(" Tidy colors ".to_string()),
            },
            Some(ReviewDelivery::Inline),
        )
        .await
        .expect("start admitted review");
    assert_eq!(output.response.turn.status, AgentTurnStatus::Accepted);

    timeout(Duration::from_secs(1), started.notified())
        .await
        .expect("review backend should start");
    release.notify_one();

    let events = timeout(Duration::from_secs(1), async {
        loop {
            let events = runtime
                .events_for_session("session-review-lifecycle")
                .expect("review lifecycle events");
            if events
                .iter()
                .any(|event| event.event_type == "turn.completed")
            {
                break events;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("review should reach a terminal event");

    let lifecycle = events
        .iter()
        .filter(|event| {
            matches!(
                event.event_type.as_str(),
                "turn.accepted" | "turn.completed"
            ) || (matches!(event.event_type.as_str(), "item.started" | "item.completed")
                && event.payload["item"]["payload"]["type"] == "extension"
                && matches!(
                    event.payload["item"]["payload"]["name"].as_str(),
                    Some("enteredReviewMode") | Some("exitedReviewMode")
                ))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        lifecycle
            .iter()
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        vec![
            "item.started",
            "turn.accepted",
            "item.completed",
            "turn.completed"
        ]
    );
    assert_eq!(
        lifecycle[0].payload["item"]["payload"]["data"]["target"],
        json!({
            "type": "commit",
            "sha": "abc123",
            "title": "Tidy colors"
        })
    );
    assert_eq!(
        lifecycle[2].payload["item"]["payload"]["data"]["target"],
        lifecycle[0].payload["item"]["payload"]["data"]["target"]
    );
    let review_input = events
        .iter()
        .find(|event| event.event_type == "review.input")
        .expect("review input should remain durable but agent-only");
    assert_eq!(review_input.payload["visibility"], "agent_only");
    assert_eq!(review_input.payload["source"], "review");
    assert!(!events.iter().any(|event| {
        event.event_type == "message.created"
            && event.payload["role"] == "user"
            && event.payload["visibility"] == "user_visible"
    }));
}
