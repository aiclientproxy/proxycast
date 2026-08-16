use super::*;
use crate::runtime::agent_control::AgentControlSpawnRequest;
use crate::runtime::agent_mailbox_delivery::{canonical_mailbox_item_is_terminal, mailbox_item_id};
use crate::runtime::agent_terminal_activity::terminal_result_payload;
use agent_protocol::{
    ItemId, ItemKind, ItemStatus, SessionId, ThreadId, ThreadItem, ThreadItemPayload,
    ThreadTurnsView, Turn, TurnId, TurnStatus,
};
use async_trait::async_trait;
use std::sync::Arc;
use thread_store::{
    AgentGraphStore, AgentIdentity, AgentIdentityStore, AgentMailboxMessageKind,
    AgentMailboxResultStatus, AgentMailboxStore, ReadThreadParams, ThreadSpawnEdgeStatus,
    ThreadStore,
};
use tokio::sync::{oneshot, Mutex};
use tokio::time::{timeout, Duration};
use tool_runtime::agent_control::{
    AgentControlCaller, AgentControlCommand, AgentControlGatewayRequest,
};

#[derive(Clone, Copy)]
enum ChildOutcome {
    Completed,
    LongCompleted,
    Failed,
    Interrupted,
}

struct TerminalBackend {
    outcome: ChildOutcome,
}

#[async_trait]
impl ExecutionBackend for TerminalBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.accepted", json!({})))?;
        if request.session.session_id == "root-session" {
            return Ok(());
        }
        match self.outcome {
            ChildOutcome::Completed | ChildOutcome::LongCompleted => {
                let text = if matches!(self.outcome, ChildOutcome::LongCompleted) {
                    "研究".repeat(2_500)
                } else {
                    "durable child result".to_string()
                };
                sink.emit(RuntimeEvent::new(
                    "message.delta",
                    json!({
                        "itemId": "child-final-answer",
                        "role": "assistant",
                        "phase": "final_answer",
                        "text": text,
                    }),
                ))?;
                sink.emit(RuntimeEvent::new("turn.completed", json!({})))
            }
            ChildOutcome::Failed => sink.emit(RuntimeEvent::new(
                "turn.failed",
                json!({ "error": "child provider failed" }),
            )),
            ChildOutcome::Interrupted => sink.emit(RuntimeEvent::new("turn.canceled", json!({}))),
        }
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

fn start_params(session_id: &str, thread_id: &str) -> AgentSessionStartParams {
    AgentSessionStartParams {
        session_id: Some(session_id.to_string()),
        thread_id: Some(thread_id.to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-activity".to_string()),
        business_object_ref: None,
        locale: None,
    }
}

async fn setup(
    outcome: ChildOutcome,
) -> (
    tempfile::TempDir,
    RuntimeCore,
    Arc<ProjectionStore>,
    AgentSession,
    AgentTurn,
) {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::with_backend(Arc::new(TerminalBackend { outcome }))
        .with_event_log_writer(Arc::new(
            EventLogWriter::new(temp.path().join("event-log")).expect("event log writer"),
        ))
        .with_projection_store(store.clone());
    let root = core
        .start_session(start_params("root-session", "root-thread"))
        .expect("root session")
        .session;
    core.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
        parent_session_id: root.session_id.clone(),
        child_session_id: Some("child-session".to_string()),
        child_thread_id: Some("child-thread".to_string()),
        fork_mode: tool_runtime::agent_control::SpawnAgentForkMode::None,
    })
    .await
    .expect("child session");
    store
        .upsert_agent_identity(AgentIdentity {
            root_thread_id: ThreadId::new("root-thread"),
            thread_id: ThreadId::new("root-thread"),
            agent_path: "/root".to_string(),
            nickname: None,
            role: None,
            last_task_message: None,
        })
        .await
        .expect("root identity");
    store
        .upsert_agent_identity(AgentIdentity {
            root_thread_id: ThreadId::new("root-thread"),
            thread_id: ThreadId::new("child-thread"),
            agent_path: "/root/research".to_string(),
            nickname: None,
            role: None,
            last_task_message: Some("research".to_string()),
        })
        .await
        .expect("child identity");
    let root_turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: root.session_id.clone(),
                turn_id: Some("root-turn".to_string()),
                input: AgentInput {
                    text: "delegate".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("root turn")
        .response
        .turn;
    (temp, core, store, root, root_turn)
}

fn child_turn_params() -> AgentSessionTurnStartParams {
    AgentSessionTurnStartParams {
        session_id: "child-session".to_string(),
        turn_id: Some("child-turn".to_string()),
        input: AgentInput {
            text: "work".to_string(),
            attachments: Vec::new(),
        },
        runtime_options: None,
        queue_if_busy: false,
        skip_pre_submit_resume: false,
    }
}

fn completed_turn_with_messages(messages: &[(&str, Option<&str>)]) -> Turn {
    Turn {
        session_id: SessionId::new("child-session"),
        thread_id: ThreadId::new("child-thread"),
        turn_id: TurnId::new("child-turn"),
        status: TurnStatus::Completed,
        admission: Default::default(),
        queue: Default::default(),
        approval: Default::default(),
        items: messages
            .iter()
            .enumerate()
            .map(|(index, (text, phase))| ThreadItem {
                session_id: SessionId::new("child-session"),
                thread_id: ThreadId::new("child-thread"),
                turn_id: TurnId::new("child-turn"),
                item_id: ItemId::new(format!("message-{index}")),
                sequence: index as u64 + 1,
                ordinal: index as u64,
                created_at_ms: index as i64,
                updated_at_ms: index as i64,
                completed_at_ms: Some(index as i64),
                kind: ItemKind::AgentMessage,
                status: ItemStatus::Completed,
                payload: ThreadItemPayload::AgentMessage {
                    text: (*text).to_string(),
                    phase: phase.and_then(agent_protocol::response_item::MessagePhase::from_wire),
                    content_parts: Vec::new(),
                },
                metadata: json!({}),
            })
            .collect(),
        items_view: Default::default(),
        error: None,
        created_at_ms: 0,
        updated_at_ms: 3,
        started_at_ms: Some(0),
        completed_at_ms: Some(3),
        duration_ms: Some(3),
    }
}

#[test]
fn completed_child_result_selects_only_the_last_final_answer() {
    let turn = completed_turn_with_messages(&[
        ("commentary", Some("commentary")),
        ("first final", Some("final_answer")),
        ("last final", Some("final_answer")),
        ("legacy trailing message", None),
    ]);

    assert_eq!(
        terminal_result_payload(&turn),
        Some((
            AgentMailboxResultStatus::Completed,
            "last final".to_string(),
        ))
    );
}

#[test]
fn completed_child_result_never_promotes_commentary_or_unphased_messages() {
    let turn = completed_turn_with_messages(&[
        ("commentary", Some("commentary")),
        ("legacy trailing message", None),
    ]);

    assert_eq!(
        terminal_result_payload(&turn),
        Some((AgentMailboxResultStatus::Completed, String::new()))
    );
}

async fn complete_child(core: &RuntimeCore) -> AgentTurn {
    core.start_turn(child_turn_params(), RuntimeHostContext::default())
        .await
        .expect("child turn")
        .response
        .turn
}

#[tokio::test]
async fn terminal_result_notifies_an_existing_parent_session_actor() {
    let (_temp, core, _store, root, _root_turn) = setup(ChildOutcome::Completed).await;
    let session = core
        .session_loops
        .get_or_create(&root.session_id, &root.thread_id)
        .await
        .expect("bind parent session actor");
    let (ready_tx, ready_rx) = oneshot::channel();
    let ready_tx = Arc::new(Mutex::new(Some(ready_tx)));
    let (activity_tx, activity_rx) = oneshot::channel();
    let activity_tx = Arc::new(Mutex::new(Some(activity_tx)));
    let task = agent_runtime::session_loop::RuntimeSessionClosureTask::new(
        "parent-terminal-activity-observer",
        Vec::new(),
        move |context, _input, _cancel| {
            let ready_tx = Arc::clone(&ready_tx);
            let activity_tx = Arc::clone(&activity_tx);
            Box::pin(async move {
                if let Some(sender) = ready_tx.lock().await.take() {
                    let _ = sender.send(());
                }
                context.wait_for_pending_input().await;
                if let Some(sender) = activity_tx.lock().await.take() {
                    let _ = sender.send(());
                }
                Ok(())
            })
        },
    );
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("activity observer submission");
    ready_rx.await.expect("activity observer ready");

    complete_child(&core).await;
    timeout(Duration::from_secs(2), activity_rx)
        .await
        .expect("terminal activity notification timeout")
        .expect("terminal activity notification");
    assert_eq!(
        submission.completion.await.expect("observer completion"),
        Ok(agent_runtime::session_loop::RuntimeSessionTaskOutcome::Completed)
    );
}

async fn wait_for_recovered_child_result(
    temp: &tempfile::TempDir,
    store: Arc<ProjectionStore>,
) -> serde_json::Value {
    let restarted = RuntimeCore::default()
        .with_event_log_writer(Arc::new(
            EventLogWriter::new(temp.path().join("event-log")).expect("reopen event log writer"),
        ))
        .with_projection_store(store);
    restarted
        .ensure_current_session_hydrated("root-session")
        .await
        .expect("hydrate root after restart");
    let (root, turns) = restarted
        .session_snapshot("root-session")
        .expect("root snapshot");
    let root_turn = turns
        .into_iter()
        .find(|turn| turn.turn_id == "root-turn")
        .expect("root turn");
    let gateway =
        restarted.agent_control_gateway_for_turn(&root, &root_turn, RuntimeHostContext::default());
    let waited = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: caller(&root, &root_turn, "wait-recovered-result"),
            command: AgentControlCommand::WaitAgent { timeout_ms: 50 },
            cancel_token: None,
        })
        .await
        .expect("wait recovered result");
    let repeated = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: caller(&root, &root_turn, "wait-recovered-result-again"),
            command: AgentControlCommand::WaitAgent { timeout_ms: 0 },
            cancel_token: None,
        })
        .await
        .expect("repeat wait after recovery");
    assert_eq!(repeated.output["message"], "Wait timed out.");
    waited.output
}

fn caller(root: &AgentSession, root_turn: &AgentTurn, call_id: &str) -> AgentControlCaller {
    AgentControlCaller {
        session_id: root.session_id.clone(),
        thread_id: root.thread_id.clone(),
        turn_id: root_turn.turn_id.clone(),
        call_id: call_id.to_string(),
    }
}

#[tokio::test]
async fn completed_child_result_is_durable_consumed_once_and_visible_to_wait() {
    let (_temp, core, store, root, root_turn) = setup(ChildOutcome::Completed).await;
    complete_child(&core).await;

    let pending = store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("root-thread"),
            ThreadId::new("root-thread"),
        )
        .await
        .expect("parent activity");
    assert_eq!(pending.len(), 1);
    let result = &pending[0];
    assert_eq!(result.kind, AgentMailboxMessageKind::Result);
    assert_eq!(
        result.result_status,
        Some(AgentMailboxResultStatus::Completed)
    );
    assert_eq!(result.source_turn_id, Some(TurnId::new("child-turn")));
    assert!(result.content.contains("Message Type: FINAL_ANSWER"));
    assert!(result.content.contains("Task name: /root"));
    assert!(result.content.contains("Sender: /root/research"));
    assert!(result.content.contains("durable child result"));

    let reopened = ProjectionStore::initialize(store.path()).expect("reopen store");
    assert_eq!(
        reopened
            .list_pending_agent_mailbox_messages(
                ThreadId::new("root-thread"),
                ThreadId::new("root-thread"),
            )
            .await
            .expect("restarted parent activity"),
        pending
    );

    let terminal_events = core
        .events_for_session("child-session")
        .expect("child events")
        .into_iter()
        .filter(|event| event.event_type == "turn.completed")
        .collect::<Vec<_>>();
    store
        .append_terminal_agent_results_sync(&ThreadId::new("child-thread"), &terminal_events)
        .expect("duplicate terminal append");
    assert_eq!(
        store
            .list_pending_agent_mailbox_messages(
                ThreadId::new("root-thread"),
                ThreadId::new("root-thread"),
            )
            .await
            .expect("deduplicated parent activity")
            .len(),
        1
    );

    let gateway =
        core.agent_control_gateway_for_turn(&root, &root_turn, RuntimeHostContext::default());
    let waited = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: caller(&root, &root_turn, "wait-result"),
            command: AgentControlCommand::WaitAgent { timeout_ms: 50 },
            cancel_token: None,
        })
        .await
        .expect("wait result");
    assert_eq!(waited.output["message"], "Wait completed.");
    assert_eq!(waited.output["timed_out"], false);
    assert_eq!(waited.output["activity"][0]["kind"], "result");
    assert_eq!(waited.output["activity"][0]["sender"], "/root/research");
    assert!(waited.output["activity"][0]["content"]
        .as_str()
        .expect("activity content")
        .contains("durable child result"));
    assert!(store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("root-thread"),
            ThreadId::new("root-thread"),
        )
        .await
        .expect("activity consumed")
        .is_empty());
    assert!(canonical_mailbox_item_is_terminal(
        &store,
        &ThreadId::new("root-thread"),
        &mailbox_item_id(&result.message_id),
    )
    .await
    .expect("canonical parent result item"));
    let root_thread = store
        .read_thread(ReadThreadParams {
            thread_id: ThreadId::new("root-thread"),
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        })
        .await
        .expect("read root thread")
        .expect("root thread");
    let canonical_result = root_thread
        .turns
        .iter()
        .flat_map(|turn| turn.items.iter())
        .find(|item| item.item_id.as_str() == mailbox_item_id(&result.message_id))
        .expect("canonical result item");
    assert_eq!(canonical_result.kind, ItemKind::AgentMessage);
    assert!(matches!(
        &canonical_result.payload,
        ThreadItemPayload::AgentMessage { text, .. } if text.contains("durable child result")
    ));
    assert_eq!(
        canonical_result.metadata.pointer("/mailbox/kind"),
        Some(&json!("result"))
    );
    assert_eq!(
        canonical_result.metadata.pointer("/mailbox/sourceTurnId"),
        Some(&json!("child-turn"))
    );
    let parent_result_event = core
        .events_for_session("root-session")
        .expect("root events")
        .into_iter()
        .find(|event| {
            event.event_type == "message.delta"
                && event.payload["itemId"] == mailbox_item_id(&result.message_id)
        })
        .expect("parent result event");
    assert_eq!(parent_result_event.event_type, "message.delta");
    assert_eq!(parent_result_event.payload["role"], "assistant");

    let repeated = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: caller(&root, &root_turn, "wait-again"),
            command: AgentControlCommand::WaitAgent { timeout_ms: 0 },
            cancel_token: None,
        })
        .await
        .expect("repeat wait");
    assert_eq!(repeated.output["message"], "Wait timed out.");
    assert_eq!(repeated.output["timed_out"], true);
}

#[tokio::test]
async fn failed_child_returns_error_result_but_interrupted_child_does_not() {
    let (_failed_temp, failed_core, failed_store, _root, _root_turn) =
        setup(ChildOutcome::Failed).await;
    complete_child(&failed_core).await;
    let failed = failed_store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("root-thread"),
            ThreadId::new("root-thread"),
        )
        .await
        .expect("failed result");
    assert_eq!(failed.len(), 1);
    assert_eq!(
        failed[0].result_status,
        Some(AgentMailboxResultStatus::Failed)
    );
    assert!(failed[0]
        .content
        .contains("Agent errored: child provider failed"));
    assert!(failed[0].content.contains("give it another task"));

    let (_interrupted_temp, interrupted_core, interrupted_store, _root, _root_turn) =
        setup(ChildOutcome::Interrupted).await;
    complete_child(&interrupted_core).await;
    assert!(interrupted_store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("root-thread"),
            ThreadId::new("root-thread"),
        )
        .await
        .expect("interrupted result")
        .is_empty());
}

#[tokio::test]
async fn completed_child_result_preserves_long_unicode_final_answer() {
    let (_temp, core, store, _root, _root_turn) = setup(ChildOutcome::LongCompleted).await;
    complete_child(&core).await;
    let pending = store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("root-thread"),
            ThreadId::new("root-thread"),
        )
        .await
        .expect("long completed result");
    assert_eq!(pending.len(), 1);
    let expected = "研究".repeat(2_500);
    assert!(pending[0].content.ends_with(&expected));
    assert_eq!(
        pending[0]
            .content
            .chars()
            .filter(|character| *character == '研' || *character == '究')
            .count(),
        expected.chars().count()
    );
}

#[tokio::test]
async fn repairable_child_event_log_tail_does_not_block_parent_wait() {
    use std::io::Write;

    let (temp, core, _store, root, root_turn) = setup(ChildOutcome::Completed).await;
    complete_child(&core).await;
    let child_log = temp
        .path()
        .join("event-log/sessions/session_child-session.jsonl");
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&child_log)
        .expect("open child event log");
    file.write_all(br#"{"partial":"crash""#)
        .expect("write repairable tail");
    file.flush().expect("flush repairable tail");
    drop(file);

    let gateway =
        core.agent_control_gateway_for_turn(&root, &root_turn, RuntimeHostContext::default());
    let waited = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: caller(&root, &root_turn, "wait-repairable-tail"),
            command: AgentControlCommand::WaitAgent { timeout_ms: 100 },
            cancel_token: None,
        })
        .await
        .expect("wait repairs child log tail");
    assert_eq!(waited.output["message"], "Wait completed.");
    assert_eq!(waited.output["activity"][0]["kind"], "result");
    assert!(EventLogWriter::new(temp.path().join("event-log"))
        .expect("event log writer")
        .read_session_events("child-session")
        .expect("repaired child log")
        .iter()
        .any(|record| record.event.event_type == "turn.completed"));
}

#[tokio::test]
async fn empty_prefix_child_crash_tail_does_not_fail_parent_wait() {
    let (temp, core, _store, root, root_turn) = setup(ChildOutcome::Completed).await;
    let child_log = temp
        .path()
        .join("event-log/sessions/session_child-session.jsonl");
    std::fs::create_dir_all(child_log.parent().expect("child log parent"))
        .expect("create child log parent");
    std::fs::write(&child_log, br#"{"partial":"crash""#).expect("write empty-prefix tail");

    let gateway =
        core.agent_control_gateway_for_turn(&root, &root_turn, RuntimeHostContext::default());
    let waited = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: caller(&root, &root_turn, "wait-empty-prefix-tail"),
            command: AgentControlCommand::WaitAgent { timeout_ms: 0 },
            cancel_token: None,
        })
        .await
        .expect("empty-prefix crash tail is repairable");
    assert_eq!(waited.output["message"], "Wait timed out.");
    assert!(EventLogWriter::new(temp.path().join("event-log"))
        .expect("event log writer")
        .read_session_events("child-session")
        .expect("repaired empty child log")
        .is_empty());
}

#[tokio::test]
async fn wait_distinguishes_new_user_steer_from_mailbox_activity() {
    let (_temp, core, _store, root, root_turn) = setup(ChildOutcome::Completed).await;
    let gateway =
        core.agent_control_gateway_for_turn(&root, &root_turn, RuntimeHostContext::default());
    let wait = tokio::spawn(async move {
        gateway
            .gateway()
            .execute(AgentControlGatewayRequest {
                caller: caller(&root, &root_turn, "wait-steer"),
                command: AgentControlCommand::WaitAgent { timeout_ms: 500 },
                cancel_token: None,
            })
            .await
    });
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "root-session".to_string(),
            turn_id: Some("queued-steer".to_string()),
            input: AgentInput {
                text: "new user input".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: true,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("queued steer");
    let waited = tokio::time::timeout(std::time::Duration::from_secs(1), wait)
        .await
        .expect("wait observes steer")
        .expect("wait task")
        .expect("wait result");
    assert_eq!(waited.output["message"], "Wait interrupted by new input.");
    assert_eq!(waited.output["timed_out"], false);
    assert!(waited.output.get("activity").is_none());
}

#[tokio::test]
async fn wait_prioritizes_a_prequeued_user_steer_over_pending_mailbox_activity() {
    let (_temp, core, store, root, root_turn) = setup(ChildOutcome::Completed).await;
    complete_child(&core).await;
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "root-session".to_string(),
            turn_id: Some("prequeued-steer".to_string()),
            input: AgentInput {
                text: "already queued input".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: true,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("prequeued steer");

    let gateway =
        core.agent_control_gateway_for_turn(&root, &root_turn, RuntimeHostContext::default());
    let waited = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: caller(&root, &root_turn, "wait-prequeued-steer"),
            command: AgentControlCommand::WaitAgent { timeout_ms: 50 },
            cancel_token: None,
        })
        .await
        .expect("wait result");
    assert_eq!(waited.output["message"], "Wait interrupted by new input.");
    assert_eq!(waited.output["timed_out"], false);
    assert_eq!(
        store
            .list_pending_agent_mailbox_messages(
                ThreadId::new("root-thread"),
                ThreadId::new("root-thread"),
            )
            .await
            .expect("mailbox remains pending")
            .len(),
        1
    );
}

#[tokio::test]
async fn active_wait_recovers_terminal_result_written_after_initial_repair() {
    let (_temp, core, store, root, root_turn) = setup(ChildOutcome::Completed).await;
    let gateway =
        core.agent_control_gateway_for_turn(&root, &root_turn, RuntimeHostContext::default());
    let wait = tokio::spawn(async move {
        gateway
            .gateway()
            .execute(AgentControlGatewayRequest {
                caller: caller(&root, &root_turn, "wait-late-recovery"),
                command: AgentControlCommand::WaitAgent { timeout_ms: 500 },
                cancel_token: None,
            })
            .await
    });
    tokio::time::sleep(std::time::Duration::from_millis(40)).await;
    store
        .delete_agent_identity(ThreadId::new("child-thread"))
        .await
        .expect("remove child identity");
    core.start_turn(child_turn_params(), RuntimeHostContext::default())
        .await
        .expect_err("terminal mailbox effect must fail");
    store
        .upsert_agent_identity(AgentIdentity {
            root_thread_id: ThreadId::new("root-thread"),
            thread_id: ThreadId::new("child-thread"),
            agent_path: "/root/research".to_string(),
            nickname: None,
            role: None,
            last_task_message: Some("research".to_string()),
        })
        .await
        .expect("restore child identity");

    let waited = tokio::time::timeout(std::time::Duration::from_secs(1), wait)
        .await
        .expect("wait observes recovered terminal")
        .expect("wait task")
        .expect("wait result");
    assert_eq!(waited.output["message"], "Wait completed.");
    assert_eq!(waited.output["activity"][0]["kind"], "result");
}

#[tokio::test]
async fn expired_wait_recovers_terminal_activity_before_timing_out() {
    let (_temp, core, store, root, root_turn) = setup(ChildOutcome::Completed).await;
    store
        .delete_agent_identity(ThreadId::new("child-thread"))
        .await
        .expect("remove child identity");
    core.start_turn(child_turn_params(), RuntimeHostContext::default())
        .await
        .expect_err("terminal mailbox effect must fail");
    store
        .upsert_agent_identity(AgentIdentity {
            root_thread_id: ThreadId::new("root-thread"),
            thread_id: ThreadId::new("child-thread"),
            agent_path: "/root/research".to_string(),
            nickname: None,
            role: None,
            last_task_message: Some("research".to_string()),
        })
        .await
        .expect("restore child identity");

    let gateway =
        core.agent_control_gateway_for_turn(&root, &root_turn, RuntimeHostContext::default());
    let waited = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: caller(&root, &root_turn, "wait-expired-recovery"),
            command: AgentControlCommand::WaitAgent { timeout_ms: 0 },
            cancel_token: None,
        })
        .await
        .expect("wait result");
    assert_eq!(waited.output["message"], "Wait completed.");
    assert_eq!(waited.output["activity"][0]["kind"], "result");
}

#[tokio::test]
async fn concurrent_waits_report_each_mailbox_activity_once() {
    let (temp, core, _store, root, root_turn) = setup(ChildOutcome::Completed).await;
    complete_child(&core).await;
    let first_gateway =
        core.agent_control_gateway_for_turn(&root, &root_turn, RuntimeHostContext::default());
    let second_gateway =
        core.agent_control_gateway_for_turn(&root, &root_turn, RuntimeHostContext::default());
    let first_caller = caller(&root, &root_turn, "wait-concurrent-first");
    let second_caller = caller(&root, &root_turn, "wait-concurrent-second");
    let first = tokio::spawn(async move {
        first_gateway
            .gateway()
            .execute(AgentControlGatewayRequest {
                caller: first_caller,
                command: AgentControlCommand::WaitAgent { timeout_ms: 100 },
                cancel_token: None,
            })
            .await
    });
    let second = tokio::spawn(async move {
        second_gateway
            .gateway()
            .execute(AgentControlGatewayRequest {
                caller: second_caller,
                command: AgentControlCommand::WaitAgent { timeout_ms: 100 },
                cancel_token: None,
            })
            .await
    });
    let (first, second) = tokio::join!(first, second);
    let outputs = [
        first.expect("first wait task").expect("first wait").output,
        second
            .expect("second wait task")
            .expect("second wait")
            .output,
    ];
    assert_eq!(
        outputs
            .iter()
            .filter(|output| output["message"] == "Wait completed.")
            .count(),
        1
    );
    assert_eq!(
        outputs
            .iter()
            .filter(|output| output["message"] == "Wait timed out.")
            .count(),
        1
    );
    let mailbox_events = core
        .events_for_session("root-session")
        .expect("root events")
        .into_iter()
        .filter(|event| event.payload.pointer("/mailbox/kind") == Some(&json!("result")))
        .count();
    assert_eq!(mailbox_events, 1);
    let durable_mailbox_events = EventLogWriter::new(temp.path().join("event-log"))
        .expect("event log writer")
        .read_session_events("root-session")
        .expect("root event log")
        .into_iter()
        .filter(|record| record.event.payload.pointer("/mailbox/kind") == Some(&json!("result")))
        .count();
    assert_eq!(durable_mailbox_events, 1);
}

#[tokio::test]
async fn grandchild_result_targets_only_its_direct_parent() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::with_backend(Arc::new(TerminalBackend {
        outcome: ChildOutcome::Completed,
    }))
    .with_projection_store(store.clone());
    for (session_id, thread_id) in [
        ("root-session", "root-thread"),
        ("parent-session", "parent-thread"),
        ("child-session", "child-thread"),
    ] {
        core.start_session(start_params(session_id, thread_id))
            .expect("canonical session");
    }
    store
        .upsert_thread_spawn_edge(
            ThreadId::new("root-thread"),
            ThreadId::new("parent-thread"),
            ThreadSpawnEdgeStatus::Open,
        )
        .await
        .expect("parent edge");
    store
        .upsert_thread_spawn_edge(
            ThreadId::new("parent-thread"),
            ThreadId::new("child-thread"),
            ThreadSpawnEdgeStatus::Open,
        )
        .await
        .expect("child edge");
    for (thread_id, path) in [
        ("root-thread", "/root"),
        ("parent-thread", "/root/lead"),
        ("child-thread", "/root/lead/research"),
    ] {
        store
            .upsert_agent_identity(AgentIdentity {
                root_thread_id: ThreadId::new("root-thread"),
                thread_id: ThreadId::new(thread_id),
                agent_path: path.to_string(),
                nickname: None,
                role: None,
                last_task_message: None,
            })
            .await
            .expect("agent identity");
    }

    complete_child(&core).await;
    let parent = store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("root-thread"),
            ThreadId::new("parent-thread"),
        )
        .await
        .expect("direct parent result");
    assert_eq!(parent.len(), 1);
    assert_eq!(parent[0].sender_thread_id, ThreadId::new("child-thread"));
    assert!(parent[0].content.contains("Task name: /root/lead"));
    assert!(parent[0].content.contains("Sender: /root/lead/research"));
    assert!(store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("root-thread"),
            ThreadId::new("root-thread"),
        )
        .await
        .expect("root result isolation")
        .is_empty());
}

#[tokio::test]
async fn restart_recovers_result_after_canonical_terminal_apply_failure() {
    let (temp, core, store, _root, _root_turn) = setup(ChildOutcome::Completed).await;
    let connection = rusqlite::Connection::open(store.path()).expect("open projection database");
    connection
        .execute(
            "UPDATE canonical_threads SET last_sequence = 9999 WHERE thread_id = ?1",
            ["child-thread"],
        )
        .expect("force canonical sequence conflict");

    let error = core
        .start_turn(child_turn_params(), RuntimeHostContext::default())
        .await
        .expect_err("canonical child terminal must fail after EventLog append");
    assert!(
        error
            .to_string()
            .contains("canonical child terminal Turn must persist before parent activity"),
        "expected canonical terminal persistence failure, got: {error}"
    );
    assert!(store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("root-thread"),
            ThreadId::new("root-thread"),
        )
        .await
        .expect("no result before canonical terminal")
        .is_empty());
    connection
        .execute(
            "UPDATE canonical_threads SET last_sequence = NULL WHERE thread_id = ?1",
            ["child-thread"],
        )
        .expect("repair canonical sequence");
    drop(core);

    let waited = wait_for_recovered_child_result(&temp, store.clone()).await;
    assert_eq!(waited["message"], "Wait completed.");
    assert_eq!(waited["activity"][0]["kind"], "result");
    assert!(waited["activity"][0]["content"]
        .as_str()
        .expect("result content")
        .contains("durable child result"));
    assert!(store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("root-thread"),
            ThreadId::new("root-thread"),
        )
        .await
        .expect("recovered result remains delivered")
        .is_empty());
}

#[tokio::test]
async fn restart_recovers_result_after_canonical_terminal_before_mailbox_failure() {
    let (temp, core, store, _root, _root_turn) = setup(ChildOutcome::Completed).await;
    store
        .delete_agent_identity(ThreadId::new("child-thread"))
        .await
        .expect("remove child identity before terminal result");

    let error = core
        .start_turn(child_turn_params(), RuntimeHostContext::default())
        .await
        .expect_err("mailbox append must fail after canonical terminal");
    assert!(
        error
            .to_string()
            .contains("spawned child child-thread has no durable identity"),
        "expected missing child identity failure, got: {error}"
    );
    store
        .upsert_agent_identity(AgentIdentity {
            root_thread_id: ThreadId::new("root-thread"),
            thread_id: ThreadId::new("child-thread"),
            agent_path: "/root/research".to_string(),
            nickname: None,
            role: None,
            last_task_message: Some("research".to_string()),
        })
        .await
        .expect("restore child identity");
    drop(core);

    let waited = wait_for_recovered_child_result(&temp, store.clone()).await;
    assert_eq!(waited["message"], "Wait completed.");
    assert_eq!(waited["activity"][0]["kind"], "result");
    assert!(store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("root-thread"),
            ThreadId::new("root-thread"),
        )
        .await
        .expect("recovered result remains delivered")
        .is_empty());
}
