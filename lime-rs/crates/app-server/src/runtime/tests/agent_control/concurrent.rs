use super::*;
use std::collections::HashMap;

struct ConcurrentChildBackend {
    child_started: tokio::sync::mpsc::UnboundedSender<ExecutionRequest>,
    child_releases:
        tokio::sync::Mutex<HashMap<String, tokio::sync::oneshot::Sender<ChildTerminal>>>,
}

enum ChildTerminal {
    Completed,
    Failed,
}

fn spawn_request(
    session: &AgentSession,
    turn: &AgentTurn,
    task_name: &str,
    call_id: &str,
) -> AgentControlGatewayRequest {
    AgentControlGatewayRequest {
        caller: AgentControlCaller {
            session_id: session.session_id.clone(),
            thread_id: session.thread_id.clone(),
            turn_id: turn.turn_id.clone(),
            call_id: call_id.to_string(),
        },
        command: AgentControlCommand::SpawnAgent {
            task_name: task_name.to_string(),
            message: format!("inspect {task_name}"),
            fork_mode: SpawnAgentForkMode::None,
            model_overrides: SpawnAgentModelOverrides::default(),
        },
        cancel_token: None,
    }
}

impl ConcurrentChildBackend {
    async fn release(&self, session_id: &str, terminal: ChildTerminal) {
        let sender = self
            .child_releases
            .lock()
            .await
            .remove(session_id)
            .unwrap_or_else(|| panic!("missing release gate for child session {session_id}"));
        sender
            .send(terminal)
            .unwrap_or_else(|_| panic!("child release receiver should be live"));
    }
}

#[async_trait::async_trait]
impl ExecutionBackend for ConcurrentChildBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        if request.session.session_id == "parent-session" {
            return Ok(());
        }

        let session_id = request.session.session_id.clone();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        self.child_releases
            .lock()
            .await
            .insert(session_id.clone(), release_tx);
        self.child_started
            .send(request)
            .map_err(|_| RuntimeCoreError::Backend("child start observer dropped".to_string()))?;
        let terminal = release_rx
            .await
            .map_err(|_| RuntimeCoreError::Backend("child release dropped".to_string()))?;
        match terminal {
            ChildTerminal::Completed => sink.emit(RuntimeEvent::new("turn.completed", json!({}))),
            ChildTerminal::Failed => Err(RuntimeCoreError::Backend(format!(
                "fixture child {session_id} failed"
            ))),
        }
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.canceled", json!({})))
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
async fn concurrent_children_keep_mailbox_routes_and_terminal_state_isolated() {
    let (child_started_tx, mut child_started_rx) = tokio::sync::mpsc::unbounded_channel();
    let backend = Arc::new(ConcurrentChildBackend {
        child_started: child_started_tx,
        child_releases: tokio::sync::Mutex::new(HashMap::new()),
    });
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::with_backend(backend.clone()).with_projection_store(store.clone());
    let session = core
        .start_session(start_params("parent-session", "parent-thread"))
        .expect("parent")
        .session;
    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: session.session_id.clone(),
                turn_id: Some("parent-turn".to_string()),
                input: AgentInput {
                    text: "delegate two independent tasks".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("parent turn")
        .response
        .turn;
    let gateway =
        core.agent_control_gateway_for_turn(&session, &turn, RuntimeHostContext::default());
    let spawn_request = |task_name: &str, call_id: &str| AgentControlGatewayRequest {
        caller: AgentControlCaller {
            session_id: session.session_id.clone(),
            thread_id: session.thread_id.clone(),
            turn_id: turn.turn_id.clone(),
            call_id: call_id.to_string(),
        },
        command: AgentControlCommand::SpawnAgent {
            task_name: task_name.to_string(),
            message: format!("inspect {task_name}"),
            fork_mode: SpawnAgentForkMode::None,
            model_overrides: SpawnAgentModelOverrides::default(),
        },
        cancel_token: None,
    };

    let (alpha_result, beta_result) = tokio::join!(
        gateway
            .gateway()
            .execute(spawn_request("alpha", "spawn-alpha")),
        gateway
            .gateway()
            .execute(spawn_request("beta", "spawn-beta")),
    );
    let alpha_result = alpha_result.expect("alpha spawn");
    let beta_result = beta_result.expect("beta spawn");
    let alpha_message_id = alpha_result.output["message_id"]
        .as_str()
        .expect("alpha message id")
        .to_string();
    let beta_message_id = beta_result.output["message_id"]
        .as_str()
        .expect("beta message id")
        .to_string();
    assert_ne!(alpha_message_id, beta_message_id);
    assert_eq!(alpha_result.output["task_name"], "/root/alpha");
    assert_eq!(beta_result.output["task_name"], "/root/beta");

    let mut started = HashMap::new();
    for _ in 0..2 {
        let request = tokio::time::timeout(Duration::from_secs(1), child_started_rx.recv())
            .await
            .expect("both child turns should start")
            .expect("child start request");
        started.insert(
            request.session.session_id.clone(),
            (
                request.session.thread_id.clone(),
                request.turn.turn_id.clone(),
            ),
        );
    }
    assert_eq!(started.len(), 2, "children must use independent sessions");

    let alpha_identity = spawned_child_identity(&store, "parent-thread", "alpha").await;
    let beta_identity = spawned_child_identity(&store, "parent-thread", "beta").await;
    assert_ne!(alpha_identity.thread_id, beta_identity.thread_id);
    let alpha_session_id = store
        .read_thread(ReadThreadParams {
            thread_id: alpha_identity.thread_id.clone(),
            include_archived: true,
            turns_view: ThreadTurnsView::NotLoaded,
        })
        .await
        .expect("alpha thread")
        .expect("alpha thread exists")
        .session_id
        .to_string();
    let beta_session_id = store
        .read_thread(ReadThreadParams {
            thread_id: beta_identity.thread_id.clone(),
            include_archived: true,
            turns_view: ThreadTurnsView::NotLoaded,
        })
        .await
        .expect("beta thread")
        .expect("beta thread exists")
        .session_id
        .to_string();
    assert!(started.contains_key(&alpha_session_id));
    assert!(started.contains_key(&beta_session_id));

    for (thread_id, message_id, expected_text) in [
        (
            &alpha_identity.thread_id,
            &alpha_message_id,
            "inspect alpha",
        ),
        (&beta_identity.thread_id, &beta_message_id, "inspect beta"),
    ] {
        let pending = store
            .list_pending_agent_mailbox_messages(ThreadId::new("parent-thread"), thread_id.clone())
            .await
            .expect("pending mailbox");
        assert!(
            pending.is_empty(),
            "started child mailbox should be consumed"
        );
        let thread = store
            .read_thread(ReadThreadParams {
                thread_id: thread_id.clone(),
                include_archived: true,
                turns_view: ThreadTurnsView::Full,
            })
            .await
            .expect("child thread read")
            .expect("child thread exists");
        let mailbox_item_id = super::super::agent_mailbox_delivery::mailbox_item_id(message_id);
        assert!(
            thread.turns.iter().any(|child_turn| {
                child_turn.items.iter().any(|item| {
                    item.item_id.as_str() == mailbox_item_id
                        && serde_json::to_string(&item.payload)
                            .expect("item payload json")
                            .contains(expected_text)
                })
            }),
            "child thread must contain only its routed mailbox item: {thread:#?}"
        );
        let other_text = if expected_text.ends_with("alpha") {
            "inspect beta"
        } else {
            "inspect alpha"
        };
        assert!(!thread.turns.iter().any(|child_turn| {
            child_turn.items.iter().any(|item| {
                serde_json::to_string(&item.payload)
                    .expect("item payload json")
                    .contains(other_text)
            })
        }));
    }

    backend
        .release(&alpha_session_id, ChildTerminal::Completed)
        .await;
    backend
        .release(&beta_session_id, ChildTerminal::Completed)
        .await;

    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let alpha = store
                .read_thread(ReadThreadParams {
                    thread_id: alpha_identity.thread_id.clone(),
                    include_archived: true,
                    turns_view: ThreadTurnsView::Full,
                })
                .await
                .expect("alpha thread read")
                .expect("alpha thread");
            let beta = store
                .read_thread(ReadThreadParams {
                    thread_id: beta_identity.thread_id.clone(),
                    include_archived: true,
                    turns_view: ThreadTurnsView::Full,
                })
                .await
                .expect("beta thread read")
                .expect("beta thread");
            if alpha
                .turns
                .iter()
                .any(|turn| turn.status == agent_protocol::TurnStatus::Completed)
                && beta
                    .turns
                    .iter()
                    .any(|turn| turn.status == agent_protocol::TurnStatus::Completed)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("both child turns should complete independently");

    let wait = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                call_id: "wait-after-concurrent".to_string(),
                session_id: session.session_id.clone(),
                thread_id: session.thread_id.clone(),
                turn_id: turn.turn_id.clone(),
            },
            command: AgentControlCommand::WaitAgent { timeout_ms: 0 },
            cancel_token: None,
        })
        .await
        .expect("wait for concurrent children");
    assert_eq!(wait.output["timed_out"], false);
    let activity = wait.output["activity"].as_array().expect("wait activity");
    assert_eq!(
        activity.len(),
        2,
        "both child results must be aggregated once"
    );
    for path in ["/root/alpha", "/root/beta"] {
        assert!(activity.iter().any(|entry| {
            entry["sender"] == path
                && entry["kind"] == "result"
                && entry["result_status"] == "completed"
        }));
    }
    assert!(wait.projection_facts.is_empty());
    assert_eq!(wait.state_facts.len(), 2);
    for thread_id in [&alpha_identity.thread_id, &beta_identity.thread_id] {
        assert!(wait.state_facts.iter().any(|fact| {
            &fact.target_thread_id == thread_id
                && fact.state.status == agent_protocol::CollabAgentStatus::Completed
        }));
    }
    let second_wait = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                call_id: "wait-after-concurrent-again".to_string(),
                session_id: session.session_id.clone(),
                thread_id: session.thread_id.clone(),
                turn_id: turn.turn_id.clone(),
            },
            command: AgentControlCommand::WaitAgent { timeout_ms: 0 },
            cancel_token: None,
        })
        .await
        .expect("second wait");
    assert_eq!(second_wait.output["timed_out"], true);
    assert!(second_wait.state_facts.is_empty());

    let listed = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                call_id: "list-after-concurrent".to_string(),
                session_id: session.session_id.clone(),
                thread_id: session.thread_id.clone(),
                turn_id: turn.turn_id.clone(),
            },
            command: AgentControlCommand::ListAgents { path_prefix: None },
            cancel_token: None,
        })
        .await
        .expect("list agents");
    let agents = listed.output["agents"].as_array().expect("agents array");
    assert_eq!(
        agents
            .iter()
            .filter(
                |agent| agent["agent_name"] == "/root/alpha" || agent["agent_name"] == "/root/beta"
            )
            .count(),
        2
    );
    assert!(agents.iter().any(|agent| {
        agent["agent_name"] == "/root/alpha" && agent["agent_status"] == "completed"
    }));
    assert!(agents.iter().any(|agent| {
        agent["agent_name"] == "/root/beta" && agent["agent_status"] == "completed"
    }));

    let send_request = |target: &str, call_id: &str, message: &str| AgentControlGatewayRequest {
        caller: AgentControlCaller {
            session_id: session.session_id.clone(),
            thread_id: session.thread_id.clone(),
            turn_id: turn.turn_id.clone(),
            call_id: call_id.to_string(),
        },
        command: AgentControlCommand::SendMessage {
            target: target.to_string(),
            message: message.to_string(),
        },
        cancel_token: None,
    };
    let (alpha_send, beta_send) = tokio::join!(
        gateway
            .gateway()
            .execute(send_request("alpha", "send-alpha", "alpha-only")),
        gateway
            .gateway()
            .execute(send_request("beta", "send-beta", "beta-only")),
    );
    let alpha_send = alpha_send.expect("alpha send");
    let beta_send = beta_send.expect("beta send");
    assert_ne!(
        alpha_send.output["message_id"],
        beta_send.output["message_id"]
    );
    let alpha_pending = store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("parent-thread"),
            alpha_identity.thread_id.clone(),
        )
        .await
        .expect("alpha pending mailbox");
    let beta_pending = store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("parent-thread"),
            beta_identity.thread_id.clone(),
        )
        .await
        .expect("beta pending mailbox");
    assert_eq!(alpha_pending.len(), 1);
    assert_eq!(beta_pending.len(), 1);
    assert_eq!(alpha_pending[0].content, "alpha-only");
    assert_eq!(beta_pending[0].content, "beta-only");
}

#[tokio::test]
async fn root_tree_execution_limit_rejects_fourth_child_and_reuses_terminal_slot() {
    let (child_started_tx, mut child_started_rx) = tokio::sync::mpsc::unbounded_channel();
    let backend = Arc::new(ConcurrentChildBackend {
        child_started: child_started_tx,
        child_releases: tokio::sync::Mutex::new(HashMap::new()),
    });
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::with_backend(backend.clone()).with_projection_store(store.clone());
    let session = core
        .start_session(start_params("parent-session", "parent-thread"))
        .expect("parent")
        .session;
    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: session.session_id.clone(),
                turn_id: Some("parent-turn".to_string()),
                input: AgentInput {
                    text: "delegate up to the root-tree limit".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("parent turn")
        .response
        .turn;
    let gateway =
        core.agent_control_gateway_for_turn(&session, &turn, RuntimeHostContext::default());

    for index in 1..=3 {
        gateway
            .gateway()
            .execute(spawn_request(
                &session,
                &turn,
                &format!("child_{index}"),
                &format!("spawn-{index}"),
            ))
            .await
            .unwrap_or_else(|error| panic!("child {index} should be admitted: {error}"));
    }
    let mut active_sessions = Vec::new();
    for _ in 0..3 {
        active_sessions.push(
            tokio::time::timeout(Duration::from_secs(1), child_started_rx.recv())
                .await
                .expect("three child turns should start")
                .expect("child start request")
                .session
                .session_id,
        );
    }

    let error = gateway
        .gateway()
        .execute(spawn_request(
            &session,
            &turn,
            "over_capacity",
            "spawn-over-capacity",
        ))
        .await
        .expect_err("fourth active child must fail closed");
    assert_eq!(
        error.message(),
        "agent_limit_reached: maximum concurrent child turns is 3"
    );
    assert!(
        store
            .list_agent_identities(ThreadId::new("parent-thread"))
            .await
            .expect("agent identities")
            .iter()
            .all(|identity| identity.agent_path != "/root/over_capacity"),
        "capacity failure must happen before a durable child is created"
    );

    let released_session = active_sessions.remove(0);
    backend
        .release(&released_session, ChildTerminal::Completed)
        .await;
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let completed = core
                .read_session(AgentSessionReadParams {
                    session_id: released_session.clone(),
                    history_limit: None,
                    history_offset: None,
                    history_before_message_id: None,
                })
                .expect("released child session")
                .turns
                .iter()
                .any(|turn| turn.status == AgentTurnStatus::Completed);
            if completed {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("completed child should release its slot");

    gateway
        .gateway()
        .execute(spawn_request(
            &session,
            &turn,
            "replacement",
            "spawn-replacement",
        ))
        .await
        .expect("terminal child slot should be reusable");
    let replacement_session = tokio::time::timeout(Duration::from_secs(1), child_started_rx.recv())
        .await
        .expect("replacement child should start")
        .expect("replacement start request")
        .session
        .session_id;

    for session_id in active_sessions
        .into_iter()
        .chain(std::iter::once(replacement_session))
    {
        backend.release(&session_id, ChildTerminal::Completed).await;
    }
}

#[tokio::test]
async fn completed_child_is_lru_unloaded_but_durable_and_reloadable() {
    let (child_started_tx, mut child_started_rx) = tokio::sync::mpsc::unbounded_channel();
    let backend = Arc::new(ConcurrentChildBackend {
        child_started: child_started_tx,
        child_releases: tokio::sync::Mutex::new(HashMap::new()),
    });
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::with_backend(backend.clone()).with_projection_store(store.clone());
    let session = core
        .start_session(start_params("parent-session", "parent-thread"))
        .expect("parent")
        .session;
    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: session.session_id.clone(),
                turn_id: Some("parent-turn".to_string()),
                input: AgentInput {
                    text: "exercise resident child reload".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("parent turn")
        .response
        .turn;
    let gateway =
        core.agent_control_gateway_for_turn(&session, &turn, RuntimeHostContext::default());

    let mut child_sessions = Vec::new();
    for index in 1..=3 {
        gateway
            .gateway()
            .execute(spawn_request(
                &session,
                &turn,
                &format!("resident_{index}"),
                &format!("spawn-resident-{index}"),
            ))
            .await
            .expect("resident child admission");
        child_sessions.push(
            tokio::time::timeout(Duration::from_secs(1), child_started_rx.recv())
                .await
                .expect("resident child should start")
                .expect("resident child request")
                .session
                .session_id,
        );
    }
    for child_session in &child_sessions {
        backend
            .release(child_session, ChildTerminal::Completed)
            .await;
    }
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let all_terminal = child_sessions.iter().all(|child_session| {
                core.read_session(AgentSessionReadParams {
                    session_id: child_session.clone(),
                    history_limit: None,
                    history_offset: None,
                    history_before_message_id: None,
                })
                .expect("resident child session")
                .turns
                .iter()
                .any(|turn| turn.status == AgentTurnStatus::Completed)
            });
            if all_terminal {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("resident children should become terminal");

    gateway
        .gateway()
        .execute(spawn_request(
            &session,
            &turn,
            "resident_4",
            "spawn-resident-4",
        ))
        .await
        .expect("fourth resident should evict the oldest terminal actor");
    let fourth_session = tokio::time::timeout(Duration::from_secs(1), child_started_rx.recv())
        .await
        .expect("fourth resident should start")
        .expect("fourth resident request")
        .session
        .session_id;

    let oldest_identity = spawned_child_identity(&store, "parent-thread", "resident_1").await;
    assert!(
        core.session_loops
            .get_existing(&child_sessions[0])
            .await
            .is_none(),
        "oldest terminal child actor should be unloaded"
    );
    assert!(
        store
            .read_thread(ReadThreadParams {
                thread_id: oldest_identity.thread_id.clone(),
                include_archived: true,
                turns_view: ThreadTurnsView::Full,
            })
            .await
            .expect("durable oldest child")
            .is_some(),
        "LRU unload must retain the canonical child thread"
    );

    let followup = gateway.gateway().execute(AgentControlGatewayRequest {
        caller: AgentControlCaller {
            session_id: session.session_id.clone(),
            thread_id: session.thread_id.clone(),
            turn_id: turn.turn_id.clone(),
            call_id: "reload-oldest".to_string(),
        },
        command: AgentControlCommand::FollowupTask {
            target: "resident_1".to_string(),
            message: "run after cold reload".to_string(),
        },
        cancel_token: None,
    });
    backend
        .release(&fourth_session, ChildTerminal::Completed)
        .await;
    followup.await.expect("completed cold child should reload");
    let reloaded = tokio::time::timeout(Duration::from_secs(1), child_started_rx.recv())
        .await
        .expect("reloaded child should start")
        .expect("reloaded child request");
    assert_eq!(reloaded.session.session_id, child_sessions[0]);
    assert!(
        core.session_loops
            .get_existing(&child_sessions[0])
            .await
            .is_some(),
        "followup should restore the completed child actor"
    );
    backend
        .release(&child_sessions[0], ChildTerminal::Completed)
        .await;
}

#[tokio::test]
async fn one_failed_child_does_not_pollute_a_completed_sibling() {
    let (child_started_tx, mut child_started_rx) = tokio::sync::mpsc::unbounded_channel();
    let backend = Arc::new(ConcurrentChildBackend {
        child_started: child_started_tx,
        child_releases: tokio::sync::Mutex::new(HashMap::new()),
    });
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::with_backend(backend.clone()).with_projection_store(store.clone());
    let session = core
        .start_session(start_params("parent-session", "parent-thread"))
        .expect("parent")
        .session;
    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: session.session_id.clone(),
                turn_id: Some("parent-turn".to_string()),
                input: AgentInput {
                    text: "delegate isolated work".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("parent turn")
        .response
        .turn;
    let gateway =
        core.agent_control_gateway_for_turn(&session, &turn, RuntimeHostContext::default());
    let spawn_request = |task_name: &str, call_id: &str| AgentControlGatewayRequest {
        caller: AgentControlCaller {
            session_id: session.session_id.clone(),
            thread_id: session.thread_id.clone(),
            turn_id: turn.turn_id.clone(),
            call_id: call_id.to_string(),
        },
        command: AgentControlCommand::SpawnAgent {
            task_name: task_name.to_string(),
            message: format!("run {task_name}"),
            fork_mode: SpawnAgentForkMode::None,
            model_overrides: SpawnAgentModelOverrides::default(),
        },
        cancel_token: None,
    };

    let (failed_spawn, completed_spawn) = tokio::join!(
        gateway
            .gateway()
            .execute(spawn_request("will_fail", "spawn-failed")),
        gateway
            .gateway()
            .execute(spawn_request("will_complete", "spawn-completed")),
    );
    failed_spawn.expect("failed child should be admitted");
    completed_spawn.expect("completed child should be admitted");

    for _ in 0..2 {
        tokio::time::timeout(Duration::from_secs(1), child_started_rx.recv())
            .await
            .expect("both child turns should start")
            .expect("child start request");
    }
    let failed_identity = spawned_child_identity(&store, "parent-thread", "will_fail").await;
    let completed_identity = spawned_child_identity(&store, "parent-thread", "will_complete").await;
    let failed_session_id = store
        .read_thread(ReadThreadParams {
            thread_id: failed_identity.thread_id.clone(),
            include_archived: true,
            turns_view: ThreadTurnsView::NotLoaded,
        })
        .await
        .expect("failed child thread")
        .expect("failed child exists")
        .session_id
        .to_string();
    let completed_session_id = store
        .read_thread(ReadThreadParams {
            thread_id: completed_identity.thread_id.clone(),
            include_archived: true,
            turns_view: ThreadTurnsView::NotLoaded,
        })
        .await
        .expect("completed child thread")
        .expect("completed child exists")
        .session_id
        .to_string();

    backend
        .release(&failed_session_id, ChildTerminal::Failed)
        .await;
    backend
        .release(&completed_session_id, ChildTerminal::Completed)
        .await;

    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let failed = core
                .read_session(AgentSessionReadParams {
                    session_id: failed_session_id.clone(),
                    history_limit: None,
                    history_offset: None,
                    history_before_message_id: None,
                })
                .expect("failed child session");
            let completed = core
                .read_session(AgentSessionReadParams {
                    session_id: completed_session_id.clone(),
                    history_limit: None,
                    history_offset: None,
                    history_before_message_id: None,
                })
                .expect("completed child session");
            if failed
                .turns
                .iter()
                .any(|turn| turn.status == AgentTurnStatus::Failed)
                && completed
                    .turns
                    .iter()
                    .any(|turn| turn.status == AgentTurnStatus::Completed)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("siblings should reach independent terminal states");

    let wait = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                session_id: session.session_id.clone(),
                thread_id: session.thread_id.clone(),
                turn_id: turn.turn_id.clone(),
                call_id: "wait-mixed-terminal".to_string(),
            },
            command: AgentControlCommand::WaitAgent { timeout_ms: 0 },
            cancel_token: None,
        })
        .await
        .expect("wait mixed terminal");
    let activity = wait.output["activity"].as_array().expect("wait activity");
    assert_eq!(activity.len(), 2);
    assert!(activity.iter().any(|entry| {
        entry["sender"] == "/root/will_fail" && entry["result_status"] == "failed"
    }));
    assert!(activity.iter().any(|entry| {
        entry["sender"] == "/root/will_complete" && entry["result_status"] == "completed"
    }));
    assert!(wait.projection_facts.is_empty());
    assert_eq!(wait.state_facts.len(), 2);
    assert!(wait.state_facts.iter().any(|fact| {
        fact.target_thread_id == failed_identity.thread_id
            && fact.state.status == agent_protocol::CollabAgentStatus::Errored
    }));
    assert!(wait.state_facts.iter().any(|fact| {
        fact.target_thread_id == completed_identity.thread_id
            && fact.state.status == agent_protocol::CollabAgentStatus::Completed
    }));

    let listed = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                session_id: session.session_id,
                thread_id: session.thread_id,
                turn_id: turn.turn_id,
                call_id: "list-mixed-terminal".to_string(),
            },
            command: AgentControlCommand::ListAgents { path_prefix: None },
            cancel_token: None,
        })
        .await
        .expect("list mixed terminal");
    let agents = listed.output["agents"].as_array().expect("agents array");
    assert!(agents.iter().any(|agent| {
        agent["agent_name"] == "/root/will_fail" && agent["agent_status"] == "errored"
    }));
    assert!(agents.iter().any(|agent| {
        agent["agent_name"] == "/root/will_complete" && agent["agent_status"] == "completed"
    }));
}
