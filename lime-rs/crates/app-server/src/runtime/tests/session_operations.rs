use super::support::{
    canonical_tool_completed_event, canonical_tool_started_event, wait_for_runtime_event,
};
use super::*;
use app_server_protocol::protocol::v2::{ThreadCompactStartParams, ThreadCompactStartResponse};
use serde_json::json;
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::timeout;

struct ReplaceBlockingBackend {
    first_started: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    start_count: AtomicUsize,
}

struct LiveActionBackend {
    response: Mutex<Option<Value>>,
}

#[async_trait]
impl ExecutionBackend for LiveActionBackend {
    fn has_live_session_responses(&self) -> bool {
        true
    }

    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Err(RuntimeCoreError::Backend(
            "live action backend requires the session input boundary".to_string(),
        ))
    }

    async fn start_turn_with_provider_history_and_session_input(
        &self,
        request: ExecutionRequest,
        _provider_history: Vec<model_provider::current_client::CurrentProviderMessage>,
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
                "approval-live",
            )
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.message))?;
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(canonical_tool_started_event(
            &request.session.session_id,
            &request.session.thread_id,
            &request.turn.turn_id,
            "tool-live",
            "Bash",
        ))?;
        sink.emit(RuntimeEvent::new(
            "action.required",
            json!({
                "requestId": "approval-live",
                "actionId": "approval-live",
                "actionType": "tool_confirmation",
                "actionKind": "tool_execution_policy",
                "availableDecisions": ["allow_once", "allow_for_session", "decline", "cancel"],
                "toolCallId": "tool-live",
                "toolName": "Bash",
                "prompt": "Allow?",
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
        *self.response.lock().expect("live response mutex poisoned") = Some(response);
        sink.emit(canonical_tool_completed_event(
            &request.session.session_id,
            &request.session.thread_id,
            &request.turn.turn_id,
            "tool-live",
            "Bash",
            "allowed",
        ))?;
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
        request: ActionRespondRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let event_type = if request
            .decision
            .is_some_and(AgentSessionApprovalDecision::is_cancel)
        {
            "action.canceled"
        } else {
            "action.resolved"
        };
        sink.emit(RuntimeEvent::new(
            event_type,
            json!({
                "requestId": request.request_id,
                "actionId": request.request_id,
                "actionType": request.action_type,
                "confirmed": request.confirmed,
                "scope": request.action_scope,
            }),
        ))
    }
}

#[async_trait]
impl ExecutionBackend for ReplaceBlockingBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.start_count.fetch_add(1, Ordering::SeqCst);
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        let first_started = self
            .first_started
            .lock()
            .expect("first started mutex poisoned")
            .take();
        if let Some(first_started) = first_started {
            let _ = first_started.send(());
            std::future::pending::<()>().await;
        }
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
async fn compact_replaces_active_turn_before_building_the_new_context_window() {
    let (first_started_tx, first_started_rx) = tokio::sync::oneshot::channel();
    let backend = Arc::new(ReplaceBlockingBackend {
        first_started: Mutex::new(Some(first_started_tx)),
        start_count: AtomicUsize::new(0),
    });
    let projection_root = tempfile::tempdir().expect("projection root");
    let projection_store = Arc::new(
        ProjectionStore::initialize(projection_root.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::with_backend(backend.clone()).with_projection_store(projection_store);
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_compact_replace".to_string()),
        thread_id: Some("thread_compact_replace".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    let turn_core = core.clone();
    let turn_task = tokio::spawn(async move {
        turn_core
            .start_turn(
                AgentSessionTurnStartParams {
                    session_id: "sess_compact_replace".to_string(),
                    turn_id: Some("turn_compact_replace_1".to_string()),
                    input: AgentInput {
                        text: "keep running".to_string(),
                        attachments: Vec::new(),
                    },
                    runtime_options: None,
                    queue_if_busy: false,
                    skip_pre_submit_resume: false,
                },
                RuntimeHostContext::default(),
            )
            .await
    });
    timeout(Duration::from_secs(1), first_started_rx)
        .await
        .expect("first turn should reach backend")
        .expect("first turn observer should remain open");

    let mut runtime_events = core.take_event_receiver().expect("runtime event receiver");
    let compact = timeout(
        Duration::from_secs(1),
        core.compact_thread(ThreadCompactStartParams {
            thread_id: "thread_compact_replace".to_string(),
        }),
    )
    .await
    .expect("compact should replace the active turn")
    .expect("compact");
    assert_eq!(compact.response, ThreadCompactStartResponse {});
    wait_for_runtime_event(&mut runtime_events, "context.compaction.completed").await;
    let first = timeout(Duration::from_secs(1), turn_task)
        .await
        .expect("replaced turn should finish")
        .expect("turn task should not panic")
        .expect("replaced turn output");
    assert_eq!(first.response.turn.status, AgentTurnStatus::Canceled);

    let events = core
        .events_for_session("sess_compact_replace")
        .expect("session events");
    let canceled_sequence = events
        .iter()
        .find(|event| event.event_type == "turn.canceled")
        .expect("replaced turn terminal event")
        .sequence;
    let compact_started_sequence = events
        .iter()
        .find(|event| event.event_type == "context.compaction.started")
        .expect("compact started event")
        .sequence;
    assert!(canceled_sequence < compact_started_sequence);

    let next = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_compact_replace".to_string(),
                turn_id: Some("turn_compact_replace_2".to_string()),
                input: AgentInput {
                    text: "continue".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("next turn");
    assert_eq!(next.response.turn.status, AgentTurnStatus::Completed);
    assert_eq!(backend.start_count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn action_response_resumes_the_waiter_owned_by_the_active_session_task() {
    let backend = Arc::new(LiveActionBackend {
        response: Mutex::new(None),
    });
    let core = RuntimeCore::with_backend(backend.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_live_response".to_string()),
        thread_id: Some("thread_live_response".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    let observed = Arc::new(Mutex::new(Vec::new()));
    let observed_for_task = Arc::clone(&observed);
    let turn_core = core.clone();
    let turn_task = tokio::spawn(async move {
        let mut callback = move |event: AgentEvent| {
            observed_for_task
                .lock()
                .expect("observed events mutex poisoned")
                .push(event.event_type.clone());
            let _ = event_tx.send(event);
            Ok(())
        };
        turn_core
            .start_turn_with_event_callback(
                AgentSessionTurnStartParams {
                    session_id: "sess_live_response".to_string(),
                    turn_id: Some("turn_live_response".to_string()),
                    input: AgentInput {
                        text: "run protected tool".to_string(),
                        attachments: Vec::new(),
                    },
                    runtime_options: None,
                    queue_if_busy: false,
                    skip_pre_submit_resume: false,
                },
                RuntimeHostContext::default(),
                &mut callback,
            )
            .await
    });
    timeout(Duration::from_secs(1), async {
        loop {
            if event_rx
                .recv()
                .await
                .is_some_and(|event| event.event_type == "action.required")
            {
                break;
            }
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "action required event; observed={:?}, turn_finished={} ",
            observed.lock().expect("observed events mutex poisoned"),
            turn_task.is_finished()
        )
    });

    core.respond_action(
        AgentSessionActionRespondParams {
            session_id: "sess_live_response".to_string(),
            request_id: "approval-live".to_string(),
            action_type: AgentSessionActionType::ToolConfirmation,
            decision: Some(AgentSessionApprovalDecision::AllowOnce),
            confirmed: None,
            response: None,
            user_data: None,
            metadata: None,
            event_name: None,
            action_scope: Some(AgentSessionActionScope {
                session_id: Some("sess_live_response".to_string()),
                thread_id: Some("thread_live_response".to_string()),
                turn_id: Some("turn_live_response".to_string()),
            }),
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("respond action");

    let turn = timeout(Duration::from_secs(1), turn_task)
        .await
        .expect("turn completion timeout")
        .expect("turn task")
        .expect("turn result");
    assert_eq!(turn.response.turn.status, AgentTurnStatus::Completed);
    assert_eq!(
        *backend
            .response
            .lock()
            .expect("live response mutex poisoned"),
        Some(json!({ "confirmed": true }))
    );
}

#[tokio::test]
async fn approval_cancel_interrupts_without_delivering_a_decline_response() {
    let backend = Arc::new(LiveActionBackend {
        response: Mutex::new(None),
    });
    let core = RuntimeCore::with_backend(backend.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_live_cancel".to_string()),
        thread_id: Some("thread_live_cancel".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    let turn_core = core.clone();
    let turn_task = tokio::spawn(async move {
        let mut callback = move |event: AgentEvent| {
            let _ = event_tx.send(event);
            Ok(())
        };
        turn_core
            .start_turn_with_event_callback(
                AgentSessionTurnStartParams {
                    session_id: "sess_live_cancel".to_string(),
                    turn_id: Some("turn_live_cancel".to_string()),
                    input: AgentInput {
                        text: "run protected tool".to_string(),
                        attachments: Vec::new(),
                    },
                    runtime_options: None,
                    queue_if_busy: false,
                    skip_pre_submit_resume: false,
                },
                RuntimeHostContext::default(),
                &mut callback,
            )
            .await
    });
    timeout(Duration::from_secs(1), async {
        while event_rx
            .recv()
            .await
            .is_none_or(|event| event.event_type != "action.required")
        {}
    })
    .await
    .expect("action required event");

    let response = core
        .respond_action(
            AgentSessionActionRespondParams {
                session_id: "sess_live_cancel".to_string(),
                request_id: "approval-live".to_string(),
                action_type: AgentSessionActionType::ToolConfirmation,
                decision: Some(AgentSessionApprovalDecision::Cancel),
                confirmed: None,
                response: None,
                user_data: None,
                metadata: None,
                event_name: None,
                action_scope: Some(AgentSessionActionScope {
                    session_id: Some("sess_live_cancel".to_string()),
                    thread_id: Some("thread_live_cancel".to_string()),
                    turn_id: Some("turn_live_cancel".to_string()),
                }),
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("cancel action");

    let turn = timeout(Duration::from_secs(1), turn_task)
        .await
        .expect("turn cancellation timeout")
        .expect("turn task")
        .expect("turn output");
    assert_eq!(turn.response.turn.status, AgentTurnStatus::Canceled);
    assert!(backend
        .response
        .lock()
        .expect("live response mutex poisoned")
        .is_none());
    assert!(response
        .events
        .iter()
        .any(|event| event.event_type == "action.canceled"));
    assert!(response
        .events
        .iter()
        .all(|event| event.event_type != "action.resolved"));
    let events = core
        .events_for_session("sess_live_cancel")
        .expect("session events");
    assert!(events
        .iter()
        .any(|event| event.event_type == "turn.canceled"));
    let canceled_tool = events
        .iter()
        .find(|event| {
            event.event_type == "item.completed"
                && event.payload["item"]["itemId"] == "item_tool-live"
        })
        .expect("active generic tool should close before turn cancellation");
    assert_eq!(
        canceled_tool.payload["item"]["status"], "cancelled",
        "events={events:#?}"
    );
    assert!(events.iter().all(|event| {
        !matches!(
            event.event_type.as_str(),
            "patch.declined" | "patch.failed" | "patch.applied"
        )
    }));
}
