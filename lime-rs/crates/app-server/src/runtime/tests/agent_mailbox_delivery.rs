use super::*;
use crate::runtime::agent_mailbox_delivery::{
    canonical_mailbox_item_is_terminal, mailbox_item_id, mailbox_message_runtime_events,
    mailbox_turn_id,
};
use crate::runtime::inter_agent_input::from_mailbox_message;
use agent_protocol::{ThreadId, ThreadTurnsView, TurnStatus};
use app_server_protocol::AgentSessionStartParams;
use async_trait::async_trait;
use rusqlite::Connection;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use thread_store::{
    AgentIdentity, AgentIdentityStore, AgentMailboxDeliveryMode, AgentMailboxDeliveryStatus,
    AgentMailboxMessage, AgentMailboxMessageKind, AgentMailboxStore,
    AppendAgentMailboxMessageParams, ReadThreadParams, ThreadStore,
};

#[derive(Default)]
struct RecordingBackend {
    requests: Mutex<Vec<ExecutionRequest>>,
    histories: Mutex<Vec<Vec<model_provider::current_client::CurrentProviderMessage>>>,
}

struct BlockingBackend {
    started: tokio::sync::mpsc::UnboundedSender<String>,
    release: Arc<tokio::sync::Semaphore>,
}

struct PendingPreflightBackend {
    pending_once: AtomicBool,
    executions: AtomicUsize,
}

#[async_trait]
impl ExecutionBackend for PendingPreflightBackend {
    fn requires_provider_selection(&self) -> bool {
        true
    }

    async fn preflight_turn(
        &self,
        request: &ExecutionRequest,
        _first_sampling_turn: bool,
    ) -> Result<(), RuntimeCoreError> {
        if self.pending_once.swap(false, Ordering::AcqRel) {
            return Err(RuntimeCoreError::PendingRoute {
                session_id: request.session.session_id.clone(),
                provider: request.provider_preference().map(str::to_string),
                model: request.model_preference().map(str::to_string),
                reason_code: "fixture_route_pending".to_string(),
            });
        }
        Ok(())
    }

    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.executions.fetch_add(1, Ordering::AcqRel);
        sink.emit(RuntimeEvent::new("turn.accepted", json!({})))?;
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

#[async_trait]
impl ExecutionBackend for BlockingBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.started
            .send(request.turn.turn_id.clone())
            .map_err(|_| RuntimeCoreError::Backend("turn start observer dropped".to_string()))?;
        self.release
            .acquire()
            .await
            .map_err(|_| RuntimeCoreError::Backend("turn release closed".to_string()))?
            .forget();
        sink.emit(RuntimeEvent::new("turn.accepted", json!({})))?;
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

#[async_trait]
impl ExecutionBackend for RecordingBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.requests
            .lock()
            .expect("requests mutex poisoned")
            .push(request);
        sink.emit(RuntimeEvent::new("turn.accepted", json!({})))
    }

    async fn start_turn_with_provider_history(
        &self,
        request: ExecutionRequest,
        provider_history: crate::runtime::provider_history::ProviderTurnHistory,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.requests
            .lock()
            .expect("requests mutex poisoned")
            .push(request);
        self.histories
            .lock()
            .expect("histories mutex poisoned")
            .push(provider_history.into_messages());
        sink.emit(RuntimeEvent::new("turn.accepted", json!({})))
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

fn setup() -> (
    tempfile::TempDir,
    RuntimeCore,
    Arc<ProjectionStore>,
    Arc<RecordingBackend>,
) {
    let backend = Arc::new(RecordingBackend::default());
    let (temp, core, store) = setup_with_backend(backend.clone());
    (temp, core, store, backend)
}

fn setup_with_backend(
    backend: Arc<dyn ExecutionBackend>,
) -> (tempfile::TempDir, RuntimeCore, Arc<ProjectionStore>) {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let event_log_writer =
        Arc::new(EventLogWriter::new(temp.path().join("event-log")).expect("event log writer"));
    let core = RuntimeCore::with_backend(backend)
        .with_event_log_writer(event_log_writer)
        .with_projection_store(store.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("root-session".to_string()),
        thread_id: Some("root-thread".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("root session");
    core.start_session(AgentSessionStartParams {
        session_id: Some("child-session".to_string()),
        thread_id: Some("child-thread".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("child session");
    futures::executor::block_on(store.upsert_agent_identity(AgentIdentity {
        root_thread_id: ThreadId::new("root-thread"),
        thread_id: ThreadId::new("root-thread"),
        agent_path: "/root".to_string(),
        nickname: None,
        role: None,
        last_task_message: None,
    }))
    .expect("root identity");
    futures::executor::block_on(store.upsert_agent_identity(AgentIdentity {
        root_thread_id: ThreadId::new("root-thread"),
        thread_id: ThreadId::new("child-thread"),
        agent_path: "/root/child".to_string(),
        nickname: Some("child".to_string()),
        role: Some("worker".to_string()),
        last_task_message: None,
    }))
    .expect("child identity");
    (temp, core, store)
}

fn message(id: &str, mode: AgentMailboxDeliveryMode) -> AgentMailboxMessage {
    AgentMailboxMessage {
        message_id: id.to_string(),
        root_thread_id: ThreadId::new("root-thread"),
        sender_thread_id: ThreadId::new("root-thread"),
        recipient_thread_id: ThreadId::new("child-thread"),
        content: format!("content-{id}"),
        kind: AgentMailboxMessageKind::Message,
        source_turn_id: None,
        result_status: None,
        delivery_mode: mode,
        delivery_status: AgentMailboxDeliveryStatus::Pending,
        created_at_ms: 1,
        delivered_at_ms: None,
    }
}

#[test]
fn mailbox_message_event_uses_ordered_typed_input() {
    let message = message("typed-input", AgentMailboxDeliveryMode::TriggerTurn);
    let event = mailbox_message_runtime_events(&message, "mailbox-item-typed-input", true)
        .into_iter()
        .next()
        .expect("mailbox message event");
    let input =
        serde_json::from_value::<Vec<agent_protocol::AgentInput>>(event.payload["input"].clone())
            .expect("typed mailbox input");
    assert_eq!(
        input,
        vec![agent_protocol::AgentInput::text("content-typed-input")]
    );
}

fn result(id: &str) -> AgentMailboxMessage {
    AgentMailboxMessage {
        message_id: id.to_string(),
        root_thread_id: ThreadId::new("root-thread"),
        sender_thread_id: ThreadId::new("child-thread"),
        recipient_thread_id: ThreadId::new("child-thread"),
        content: format!("result-{id}"),
        kind: AgentMailboxMessageKind::Result,
        source_turn_id: Some(agent_protocol::TurnId::new(format!("source-{id}"))),
        result_status: Some(thread_store::AgentMailboxResultStatus::Completed),
        delivery_mode: AgentMailboxDeliveryMode::QueueOnly,
        delivery_status: AgentMailboxDeliveryStatus::Pending,
        created_at_ms: 1,
        delivered_at_ms: None,
    }
}

#[test]
fn durable_message_maps_to_typed_runtime_input_without_losing_identity() {
    use agent_runtime::session_loop::{
        RuntimeSessionInterAgentDeliveryMode, RuntimeSessionInterAgentMessageKind,
        RuntimeSessionInterAgentResultStatus,
    };

    let input = from_mailbox_message(&result("typed-result"));
    assert_eq!(input.message_id, "typed-result");
    assert_eq!(input.root_thread_id, "root-thread");
    assert_eq!(input.sender_thread_id, "child-thread");
    assert_eq!(input.recipient_thread_id, "child-thread");
    assert_eq!(input.content, "result-typed-result");
    assert_eq!(input.kind, RuntimeSessionInterAgentMessageKind::Result);
    assert_eq!(input.source_turn_id.as_deref(), Some("source-typed-result"));
    assert_eq!(
        input.result_status,
        Some(RuntimeSessionInterAgentResultStatus::Completed)
    );
    assert_eq!(
        input.delivery_mode,
        RuntimeSessionInterAgentDeliveryMode::QueueOnly
    );
}

fn append(store: &ProjectionStore, message: AgentMailboxMessage) {
    futures::executor::block_on(
        store.append_agent_mailbox_message(AppendAgentMailboxMessageParams { message }),
    )
    .expect("append mailbox");
}

async fn pending(store: &ProjectionStore) -> Vec<AgentMailboxMessage> {
    store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("root-thread"),
            ThreadId::new("child-thread"),
        )
        .await
        .expect("list pending")
}

async fn has_item(store: &ProjectionStore, message_id: &str) -> bool {
    canonical_mailbox_item_is_terminal(
        store,
        &ThreadId::new("child-thread"),
        &mailbox_item_id(message_id),
    )
    .await
    .expect("read canonical item")
}

async fn canonical_turn_status(store: &ProjectionStore, turn_id: &str) -> Option<TurnStatus> {
    store
        .read_thread(ReadThreadParams {
            thread_id: ThreadId::new("child-thread"),
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        })
        .await
        .expect("read canonical thread")
        .and_then(|thread| {
            thread
                .turns
                .into_iter()
                .find(|turn| turn.turn_id.as_str() == turn_id)
                .map(|turn| turn.status)
        })
}

async fn canonical_item_status(
    store: &ProjectionStore,
    item_id: &str,
) -> Option<agent_protocol::ItemStatus> {
    store
        .read_thread(ReadThreadParams {
            thread_id: ThreadId::new("child-thread"),
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        })
        .await
        .expect("read canonical thread")
        .and_then(|thread| {
            thread
                .turns
                .into_iter()
                .flat_map(|turn| turn.items)
                .find(|item| item.item_id.as_str() == item_id)
                .map(|item| item.status)
        })
}

#[tokio::test]
async fn trigger_turn_appends_canonical_item_before_ack_and_starts_recipient() {
    let (_temp, core, store, backend) = setup();
    append(
        &store,
        message("trigger-1", AgentMailboxDeliveryMode::TriggerTurn),
    );

    assert_eq!(
        core.process_pending_agent_mailbox_triggers("child-session", RuntimeHostContext::default())
            .await
            .expect("process trigger"),
        1
    );

    assert!(pending(&store).await.is_empty());
    assert!(has_item(&store, "trigger-1").await);
    let requests = backend.requests.lock().expect("requests mutex poisoned");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].input.concat_text(), "content-trigger-1");
    assert_eq!(requests[0].turn.turn_id, mailbox_turn_id("trigger-1"));
    let histories = backend.histories.lock().expect("histories mutex poisoned");
    assert!(
        histories[0].is_empty(),
        "trigger input must not be duplicated in history"
    );
    let mailbox_events = core
        .events_for_session("child-session")
        .expect("events")
        .into_iter()
        .filter(|event| event.payload["mailbox"]["messageId"] == "trigger-1")
        .collect::<Vec<_>>();
    assert_eq!(
        mailbox_events
            .iter()
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        vec!["item.started", "message.created", "item.completed"]
    );
    let mailbox_message = mailbox_events
        .iter()
        .find(|event| event.event_type == "message.created")
        .expect("mailbox message event");
    assert_eq!(
        mailbox_message.payload["itemId"],
        mailbox_item_id("trigger-1")
    );
}

#[tokio::test]
async fn concurrent_trigger_schedules_serialize_without_losing_tail_mail() {
    let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let backend = Arc::new(BlockingBackend {
        started: started_tx,
        release: Arc::clone(&release),
    });
    let (_temp, core, store) = setup_with_backend(backend);
    append(
        &store,
        message("serial-trigger-1", AgentMailboxDeliveryMode::TriggerTurn),
    );

    core.schedule_pending_agent_mailbox_triggers(
        "child-session".to_string(),
        RuntimeHostContext::default(),
        None,
    )
    .await;
    assert_eq!(
        started_rx.recv().await.as_deref(),
        Some(mailbox_turn_id("serial-trigger-1").as_str())
    );

    append(
        &store,
        message("serial-trigger-2", AgentMailboxDeliveryMode::TriggerTurn),
    );
    let second_core = core.clone();
    let second = tokio::spawn(async move {
        second_core
            .schedule_pending_agent_mailbox_triggers(
                "child-session".to_string(),
                RuntimeHostContext::default(),
                None,
            )
            .await;
    });
    let duplicate_core = core.clone();
    let duplicate = tokio::spawn(async move {
        duplicate_core
            .schedule_pending_agent_mailbox_triggers(
                "child-session".to_string(),
                RuntimeHostContext::default(),
                None,
            )
            .await;
    });
    tokio::task::yield_now().await;
    assert!(!second.is_finished());
    assert!(!duplicate.is_finished());

    release.add_permits(1);
    let second_turn = tokio::time::timeout(std::time::Duration::from_secs(1), started_rx.recv())
        .await
        .expect("tail mailbox turn should start after the active recovery")
        .expect("turn start observer");
    assert_eq!(second_turn, mailbox_turn_id("serial-trigger-2"));
    assert!(!second.is_finished());
    assert!(!duplicate.is_finished());

    release.add_permits(1);
    second.await.expect("second schedule task");
    duplicate.await.expect("duplicate schedule task");
    assert!(started_rx.try_recv().is_err());
    assert!(pending(&store).await.is_empty());
    assert!(has_item(&store, "serial-trigger-1").await);
    assert!(has_item(&store, "serial-trigger-2").await);
}

#[tokio::test]
async fn detached_pending_work_wake_returns_before_target_turn_terminal() {
    let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let backend = Arc::new(BlockingBackend {
        started: started_tx,
        release: Arc::clone(&release),
    });
    let (_temp, core, store) = setup_with_backend(backend);
    append(
        &store,
        message("detached-trigger", AgentMailboxDeliveryMode::TriggerTurn),
    );

    core.wake_pending_session_work(
        "child-session".to_string(),
        RuntimeHostContext::default(),
        None,
    );

    let turn_id = tokio::time::timeout(std::time::Duration::from_secs(1), started_rx.recv())
        .await
        .expect("detached wake should start the target turn")
        .expect("turn start observer");
    assert_eq!(turn_id, mailbox_turn_id("detached-trigger"));
    assert!(!matches!(
        canonical_turn_status(&store, &turn_id).await,
        Some(TurnStatus::Completed | TurnStatus::Failed | TurnStatus::Interrupted)
    ));

    release.add_permits(1);
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            if canonical_turn_status(&store, &turn_id).await == Some(TurnStatus::Completed) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("target turn should complete after release");
}

#[tokio::test]
async fn pending_route_preflight_keeps_trigger_mail_unmaterialized_until_retry() {
    let backend = Arc::new(PendingPreflightBackend {
        pending_once: AtomicBool::new(true),
        executions: AtomicUsize::new(0),
    });
    let (_temp, core, store) = setup_with_backend(backend.clone());
    append(
        &store,
        message("preflight-trigger", AgentMailboxDeliveryMode::TriggerTurn),
    );
    let runtime_options = Some(RuntimeOptions {
        runtime_request: Some(RuntimeRequest {
            provider_preference: Some("provider-current".to_string()),
            model_preference: Some("model-current".to_string()),
            ..RuntimeRequest::default()
        }),
        ..RuntimeOptions::default()
    });

    let error = core
        .process_pending_session_work_with_options(
            "child-session",
            RuntimeHostContext::default(),
            runtime_options.clone(),
            None,
        )
        .await
        .expect_err("first route preflight should remain pending");
    assert!(matches!(
        error,
        RuntimeCoreError::PendingRoute { reason_code, .. }
            if reason_code == "fixture_route_pending"
    ));
    assert_eq!(backend.executions.load(Ordering::Acquire), 0);
    assert_eq!(pending(&store).await.len(), 1);
    assert!(!has_item(&store, "preflight-trigger").await);

    assert_eq!(
        core.process_pending_session_work_with_options(
            "child-session",
            RuntimeHostContext::default(),
            runtime_options,
            None,
        )
        .await
        .expect("retry ready route"),
        1
    );
    assert_eq!(backend.executions.load(Ordering::Acquire), 1);
    assert!(pending(&store).await.is_empty());
    assert!(has_item(&store, "preflight-trigger").await);
}

#[tokio::test]
async fn terminal_turn_wakes_trigger_mail_that_arrived_while_recipient_was_active() {
    let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let backend = Arc::new(BlockingBackend {
        started: started_tx,
        release: Arc::clone(&release),
    });
    let (_temp, core, store) = setup_with_backend(backend);
    let active_core = core.clone();
    let active_turn = tokio::spawn(async move {
        active_core
            .start_turn(
                AgentSessionTurnStartParams {
                    session_id: "child-session".to_string(),
                    turn_id: Some("active-before-mail".to_string()),
                    input: AgentInput {
                        text: "active child work".to_string(),
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
    assert_eq!(
        started_rx.recv().await.as_deref(),
        Some("active-before-mail")
    );

    append(
        &store,
        message(
            "arrived-while-active",
            AgentMailboxDeliveryMode::TriggerTurn,
        ),
    );
    core.wake_pending_session_work(
        "child-session".to_string(),
        RuntimeHostContext::default(),
        None,
    );
    tokio::task::yield_now().await;
    assert!(started_rx.try_recv().is_err());

    release.add_permits(1);
    active_turn
        .await
        .expect("active turn task")
        .expect("active turn completion");
    let mailbox_turn = tokio::time::timeout(std::time::Duration::from_secs(1), started_rx.recv())
        .await
        .expect("terminal wake should start pending mailbox work")
        .expect("mailbox turn observer");
    assert_eq!(mailbox_turn, mailbox_turn_id("arrived-while-active"));

    release.add_permits(1);
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            if canonical_turn_status(&store, &mailbox_turn).await == Some(TurnStatus::Completed) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("pending mailbox work should complete after release");
}

#[tokio::test]
async fn queue_only_stays_pending_until_a_real_turn_without_starting_one() {
    let (_temp, core, store, backend) = setup();
    append(
        &store,
        message("queue-1", AgentMailboxDeliveryMode::QueueOnly),
    );

    assert_eq!(
        core.process_pending_agent_mailbox_triggers("child-session", RuntimeHostContext::default())
            .await
            .expect("process queue only"),
        0
    );
    assert_eq!(
        backend
            .requests
            .lock()
            .expect("requests mutex poisoned")
            .len(),
        0
    );
    assert!(core
        .has_pending_agent_mailbox_activity("child-session")
        .await
        .expect("mailbox activity"));

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "child-session".to_string(),
            turn_id: Some("user-turn".to_string()),
            input: AgentInput {
                text: "real user turn".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("start real turn");

    assert!(pending(&store).await.is_empty());
    assert!(has_item(&store, "queue-1").await);
    assert!(!core
        .has_pending_agent_mailbox_activity("child-session")
        .await
        .expect("mailbox activity after delivery"));
    let requests = backend.requests.lock().expect("requests mutex poisoned");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].input.concat_text(), "real user turn");
    let histories = backend.histories.lock().expect("histories mutex poisoned");
    assert_eq!(histories.len(), 1);
    assert_eq!(
        histories[0]
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|content| match content {
                model_provider::current_client::CurrentProviderContent::Text(text) => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec!["content-queue-1"]
    );
}

#[tokio::test]
async fn multiple_results_complete_distinct_canonical_items_before_ack() {
    let (_temp, core, store, _backend) = setup();
    append(&store, result("result-1"));
    append(&store, result("result-2"));

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "child-session".to_string(),
            turn_id: Some("result-turn".to_string()),
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
    .expect("deliver distinct result items");

    assert!(pending(&store).await.is_empty());
    let thread = store
        .read_thread(ReadThreadParams {
            thread_id: ThreadId::new("child-thread"),
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        })
        .await
        .expect("read canonical thread")
        .expect("canonical thread");
    let result_item_ids = [mailbox_item_id("result-1"), mailbox_item_id("result-2")];
    let result_items = thread
        .turns
        .iter()
        .flat_map(|turn| turn.items.iter())
        .filter(|item| {
            result_item_ids
                .iter()
                .any(|item_id| item.item_id.as_str() == item_id)
        })
        .collect::<Vec<_>>();
    assert_eq!(result_items.len(), 2);
    assert!(result_items
        .iter()
        .all(|item| item.status == agent_protocol::ItemStatus::Completed));
    let events = core.events_for_session("child-session").expect("events");
    for message_id in ["result-1", "result-2"] {
        let item_id = mailbox_item_id(message_id);
        let item_events = events
            .iter()
            .filter(|event| {
                event.payload.get("itemId") == Some(&json!(item_id))
                    || event.payload.pointer("/item/itemId") == Some(&json!(item_id))
            })
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            item_events,
            vec!["item.started", "message.delta", "item.completed"],
            "scenario=multiple-results messageId={message_id} itemId={item_id}"
        );
    }
}

#[tokio::test]
async fn failed_result_projects_failed_terminal_item() {
    let (_temp, core, store, _backend) = setup();
    let mut failed = result("failed-result");
    failed.result_status = Some(thread_store::AgentMailboxResultStatus::Failed);
    append(&store, failed);

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "child-session".to_string(),
            turn_id: Some("failed-result-turn".to_string()),
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
    .expect("deliver failed result item");

    assert_eq!(
        canonical_item_status(&store, &mailbox_item_id("failed-result")).await,
        Some(agent_protocol::ItemStatus::Failed)
    );
    assert!(pending(&store).await.is_empty());
}

#[tokio::test]
async fn in_progress_result_is_completed_before_retry_ack() {
    let (_temp, core, store, _backend) = setup();
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "child-session".to_string(),
            turn_id: Some("partial-result-turn".to_string()),
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
    .expect("start recipient turn");
    let result = result("partial-result");
    append(&store, result.clone());
    let (session, turns) = core.session_snapshot("child-session").expect("session");
    let turn = turns
        .into_iter()
        .find(|turn| turn.turn_id == "partial-result-turn")
        .expect("recipient turn");
    let item_id = mailbox_item_id(&result.message_id);
    let connection = Connection::open(store.path()).expect("open projection database");
    connection
        .execute(
            "UPDATE canonical_threads SET last_sequence = 9999 WHERE thread_id = ?1",
            ["child-thread"],
        )
        .expect("force canonical sequence conflict");
    let error = core
        .append_external_runtime_events(
            &session.session_id,
            Some(&turn.turn_id),
            vec![mailbox_message_runtime_events(&result, &item_id, false)
                .into_iter()
                .next()
                .expect("result delta")],
        )
        .expect_err("canonical projection must fail after EventLog append");
    assert!(error
        .to_string()
        .contains("canonical mailbox Item must persist before delivery acknowledgement"));
    assert_eq!(canonical_item_status(&store, &item_id).await, None);
    connection
        .execute(
            "UPDATE canonical_threads SET last_sequence = NULL WHERE thread_id = ?1",
            ["child-thread"],
        )
        .expect("repair canonical sequence");
    core.deliver_pending_agent_mailbox_for_turn(
        &session,
        &turn,
        &[agent_protocol::AgentInput::text("continue")],
    )
    .await
    .expect("retry result delivery");

    assert!(pending(&store).await.is_empty());
    assert_eq!(
        canonical_item_status(&store, &item_id).await,
        Some(agent_protocol::ItemStatus::Completed)
    );
    let events = core.events_for_session("child-session").expect("events");
    let item_events = events
        .iter()
        .filter(|event| {
            event.payload.get("itemId") == Some(&json!(item_id))
                || event.payload.pointer("/item/itemId") == Some(&json!(item_id))
        })
        .map(|event| event.event_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        item_events,
        vec!["item.started", "message.delta", "item.completed"],
        "scenario=partial-result-retry messageId={} itemId={item_id}",
        result.message_id
    );
}

#[tokio::test]
async fn existing_canonical_item_is_acknowledged_without_a_duplicate_visible_item() {
    let (_temp, core, store, _backend) = setup();
    let message = message("retry-1", AgentMailboxDeliveryMode::TriggerTurn);
    let turn_id = mailbox_turn_id(&message.message_id);
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "child-session".to_string(),
            turn_id: Some(turn_id),
            input: AgentInput {
                text: "placeholder".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("start turn before mailbox append");
    append(&store, message.clone());
    let (session, turns) = core.session_snapshot("child-session").expect("session");
    let turn = turns
        .into_iter()
        .find(|turn| turn.turn_id == mailbox_turn_id(&message.message_id))
        .expect("turn");
    core.append_external_runtime_events(
        &session.session_id,
        Some(&turn.turn_id),
        mailbox_message_runtime_events(&message, &mailbox_item_id(&message.message_id), true),
    )
    .expect("persist mailbox item before simulated crash");

    let retry = core
        .deliver_pending_agent_mailbox_for_turn(
            &session,
            &turn,
            &[agent_protocol::AgentInput::text(message.content.clone())],
        )
        .await
        .expect("retry delivery");
    assert!(retry.events.is_empty());
    assert!(pending(&store).await.is_empty());
    let mailbox_events = core
        .events_for_session("child-session")
        .expect("events")
        .into_iter()
        .filter(|event| event.payload["mailbox"]["messageId"] == "retry-1")
        .collect::<Vec<_>>();
    assert_eq!(
        mailbox_events
            .iter()
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        vec!["item.started", "message.created", "item.completed"]
    );
}

#[tokio::test]
async fn canonical_mailbox_item_failure_keeps_message_pending_and_rolls_back_turn() {
    let (_temp, core, store, backend) = setup();
    append(
        &store,
        message("canonical-failure", AgentMailboxDeliveryMode::TriggerTurn),
    );
    let connection = Connection::open(store.path()).expect("open projection database");
    connection
        .execute(
            "UPDATE canonical_threads SET last_sequence = 9999 WHERE thread_id = ?1",
            ["child-thread"],
        )
        .expect("force canonical sequence conflict");

    let error = core
        .process_pending_agent_mailbox_triggers("child-session", RuntimeHostContext::default())
        .await
        .expect_err("canonical mailbox item must fail closed");
    assert!(error
        .to_string()
        .contains("canonical mailbox Item must persist before delivery acknowledgement"));
    assert_eq!(pending(&store).await.len(), 1);
    assert!(backend
        .requests
        .lock()
        .expect("requests mutex poisoned")
        .is_empty());
    assert!(core
        .session_snapshot("child-session")
        .expect("child session")
        .1
        .is_empty());
    assert!(core
        .events_for_session("child-session")
        .expect("child events")
        .is_empty());
}

#[tokio::test]
async fn event_log_first_queue_only_retry_recovers_canonical_item_before_ack() {
    let (temp, core, store, backend) = setup();
    append(
        &store,
        message("event-log-retry", AgentMailboxDeliveryMode::QueueOnly),
    );
    let connection = Connection::open(store.path()).expect("open projection database");
    connection
        .execute(
            "UPDATE canonical_threads SET last_sequence = 9999 WHERE thread_id = ?1",
            ["child-thread"],
        )
        .expect("force canonical sequence conflict");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "child-session".to_string(),
            turn_id: Some("first-user-turn".to_string()),
            input: AgentInput {
                text: "first user input".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect_err("first canonical projection must fail after EventLog append");
    assert_eq!(pending(&store).await.len(), 1);
    assert!(backend
        .requests
        .lock()
        .expect("requests mutex poisoned")
        .is_empty());
    assert!(core
        .events_for_session("child-session")
        .expect("child events")
        .is_empty());
    let event_log_writer =
        EventLogWriter::new(temp.path().join("event-log")).expect("reopen event log writer");
    assert_eq!(
        event_log_writer
            .read_session_events("child-session")
            .expect("durable mailbox event")
            .len(),
        6
    );

    connection
        .execute(
            "UPDATE canonical_threads SET last_sequence = NULL WHERE thread_id = ?1",
            ["child-thread"],
        )
        .expect("repair canonical sequence");

    let retry = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "child-session".to_string(),
                turn_id: Some("second-user-turn".to_string()),
                input: AgentInput {
                    text: "second user input".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("retry mailbox delivery");
    assert_eq!(retry.response.turn.turn_id, "second-user-turn");
    assert!(pending(&store).await.is_empty());
    assert!(has_item(&store, "event-log-retry").await);
    assert_eq!(
        backend
            .requests
            .lock()
            .expect("requests mutex poisoned")
            .len(),
        1
    );
    let child_events = core
        .events_for_session("child-session")
        .expect("child events");
    let mailbox_events = child_events
        .iter()
        .filter(|event| event.payload["mailbox"]["messageId"] == "event-log-retry")
        .collect::<Vec<_>>();
    assert_eq!(
        mailbox_events
            .iter()
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        vec!["item.started", "message.created", "item.completed"]
    );
    assert!(child_events.iter().any(|event| {
        event.turn_id.as_deref() == Some("second-user-turn")
            && event.payload.pointer("/input/0/text") == Some(&json!("second user input"))
    }));
    assert!(child_events.iter().any(|event| {
        event.turn_id.as_deref() == Some("first-user-turn")
            && event.event_type == "turn.canceled"
            && event.payload["reason"] == "mailbox_projection_recovery"
    }));
    assert_eq!(
        canonical_turn_status(&store, "first-user-turn").await,
        Some(TurnStatus::Interrupted)
    );
}

#[tokio::test]
async fn event_log_first_trigger_retry_recovers_the_deterministic_turn_before_ack() {
    let (temp, core, store, backend) = setup();
    append(
        &store,
        message("trigger-log-retry", AgentMailboxDeliveryMode::TriggerTurn),
    );
    let connection = Connection::open(store.path()).expect("open projection database");
    connection
        .execute(
            "UPDATE canonical_threads SET last_sequence = 9999 WHERE thread_id = ?1",
            ["child-thread"],
        )
        .expect("force canonical sequence conflict");

    core.process_pending_agent_mailbox_triggers("child-session", RuntimeHostContext::default())
        .await
        .expect_err("first canonical projection must fail after EventLog append");
    assert_eq!(pending(&store).await.len(), 1);
    assert!(backend
        .requests
        .lock()
        .expect("requests mutex poisoned")
        .is_empty());
    let event_log_writer =
        EventLogWriter::new(temp.path().join("event-log")).expect("reopen event log writer");
    assert_eq!(
        event_log_writer
            .read_session_events("child-session")
            .expect("durable mailbox event")
            .len(),
        3
    );

    connection
        .execute(
            "UPDATE canonical_threads SET last_sequence = NULL WHERE thread_id = ?1",
            ["child-thread"],
        )
        .expect("repair canonical sequence");

    assert_eq!(
        core.process_pending_agent_mailbox_triggers("child-session", RuntimeHostContext::default())
            .await
            .expect("retry mailbox delivery"),
        1
    );
    assert!(pending(&store).await.is_empty());
    assert!(has_item(&store, "trigger-log-retry").await);
    let requests = backend.requests.lock().expect("requests mutex poisoned");
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].turn.turn_id,
        mailbox_turn_id("trigger-log-retry")
    );
}
