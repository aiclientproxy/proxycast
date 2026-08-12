use super::*;
use agent_protocol::{ItemKind, ItemStatus, ThreadItemPayload, ThreadTurnsView};
use agent_runtime::session_loop::{
    RuntimeSessionClosureTask, RuntimeSessionInterAgentDeliveryMode, RuntimeSessionInterAgentInput,
    RuntimeSessionInterAgentMessageKind,
};
use app_server_protocol::{
    AgentInput, AgentSessionReadParams, AgentSessionReadResponse, AgentSessionTurnStartParams,
};
use futures::executor::block_on;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use thread_store::{
    AgentGraphStore, AgentIdentityStore, AgentMailboxDeliveryMode, AgentMailboxStore,
    ReadThreadParams, ThreadSpawnEdgeStatus, ThreadStore,
};
use tool_runtime::agent_control::{
    AgentControlCaller, AgentControlCommand, AgentControlGatewayRequest, SpawnAgentForkMode,
    SpawnAgentModelOverrides, SubAgentProjectionActivity,
};

#[path = "agent_control/concurrent.rs"]
mod concurrent;
#[path = "agent_control/effective_route.rs"]
mod effective_route;
#[path = "agent_control/fork.rs"]
mod fork;
#[path = "agent_control/restart.rs"]
mod restart;

struct BlockingChildBackend {
    child_started: tokio::sync::mpsc::UnboundedSender<ExecutionRequest>,
    child_release: tokio::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}

#[async_trait::async_trait]
impl ExecutionBackend for BlockingChildBackend {
    fn effective_turn_runtime_options(
        &self,
        request: &ExecutionRequest,
        _first_sampling_turn: bool,
    ) -> Option<app_server_protocol::RuntimeOptions> {
        let mut options = request.runtime_options.clone()?;
        if request
            .runtime_metadata()
            .and_then(|metadata| metadata.get("fixture").and_then(serde_json::Value::as_str))
            == Some("effective-child-route")
        {
            let runtime_request = options.runtime_request_mut();
            runtime_request.provider_preference = Some("resolved-provider".to_string());
            runtime_request.model_preference = Some("resolved-model".to_string());
            runtime_request.reasoning_effort = Some("high".to_string());
            runtime_request.working_dir = Some("/tmp/effective-child-route".to_string());
        }
        Some(options)
    }

    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        if request.session.session_id == "parent-session" {
            return Ok(());
        }

        self.child_started
            .send(request)
            .map_err(|_| RuntimeCoreError::Backend("child start observer dropped".to_string()))?;
        let release = self.child_release.lock().await.take().ok_or_else(|| {
            RuntimeCoreError::Backend("child release already consumed".to_string())
        })?;
        release
            .await
            .map_err(|_| RuntimeCoreError::Backend("child release dropped".to_string()))?;
        sink.emit(RuntimeEvent::new("turn.completed", json!({})))
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

fn start_params(session_id: &str, thread_id: &str) -> AgentSessionStartParams {
    AgentSessionStartParams {
        session_id: Some(session_id.to_string()),
        thread_id: Some(thread_id.to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-control".to_string()),
        business_object_ref: None,
        locale: None,
    }
}

fn core() -> (tempfile::TempDir, RuntimeCore, Arc<ProjectionStore>) {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::default().with_projection_store(store.clone());
    (temp, core, store)
}

async fn append_completed_parent_turn(
    core: &RuntimeCore,
    session: &AgentSession,
    turn_id: &str,
    user_text: &str,
    assistant_text: &str,
) {
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: session.session_id.clone(),
            turn_id: Some(turn_id.to_string()),
            input: AgentInput {
                text: user_text.to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("start parent turn");
    core.append_external_runtime_events(
        &session.session_id,
        Some(turn_id),
        vec![
            RuntimeEvent::new(
                "message.delta",
                json!({
                    "itemId": format!("commentary-{turn_id}"),
                    "phase": "commentary",
                    "text": format!("commentary for {assistant_text}"),
                    "trace_id": format!("parent-trace-{turn_id}"),
                }),
            ),
            RuntimeEvent::new(
                "message.completed",
                json!({
                    "itemId": format!("commentary-{turn_id}"),
                    "phase": "commentary",
                    "status": "completed",
                    "text": format!("commentary for {assistant_text}"),
                    "trace_id": format!("parent-trace-{turn_id}"),
                }),
            ),
            RuntimeEvent::new("reasoning.delta", json!({ "text": "private reasoning" })),
            RuntimeEvent::new(
                "message.delta",
                json!({
                    "itemId": format!("final-{turn_id}"),
                    "phase": "final_answer",
                    "text": assistant_text,
                    "request_id": format!("parent-request-{turn_id}"),
                }),
            ),
            RuntimeEvent::new(
                "message.completed",
                json!({
                    "itemId": format!("final-{turn_id}"),
                    "phase": "final_answer",
                    "status": "completed",
                    "text": assistant_text,
                    "request_id": format!("parent-request-{turn_id}"),
                }),
            ),
            RuntimeEvent::new("turn.completed", json!({})),
        ],
    )
    .expect("complete parent turn");
}

async fn spawned_child_identity(
    store: &ProjectionStore,
    root_thread_id: &str,
    task_name: &str,
) -> thread_store::AgentIdentity {
    store
        .list_agent_identities(ThreadId::new(root_thread_id))
        .await
        .expect("list durable identities")
        .into_iter()
        .find(|identity| identity.agent_path == format!("/root/{task_name}"))
        .expect("spawned child identity")
}

async fn wait_for_session_turn(
    core: &RuntimeCore,
    session_id: &str,
    expected_turn_id: Option<&str>,
) -> AgentSessionReadResponse {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            match core.read_session(AgentSessionReadParams {
                session_id: session_id.to_string(),
                history_limit: None,
                history_offset: None,
                history_before_message_id: None,
            }) {
                Ok(response)
                    if response.turns.iter().any(|turn| {
                        expected_turn_id
                            .map(|expected| turn.turn_id == expected)
                            .unwrap_or(true)
                    }) =>
                {
                    return response;
                }
                Ok(_) | Err(RuntimeCoreError::SessionNotFound(_)) => {
                    tokio::task::yield_now().await;
                }
                Err(error) => panic!("child session read failed: {error}"),
            }
        }
    })
    .await
    .expect("child session did not expose the expected turn")
}

#[tokio::test]
async fn spawn_gateway_returns_before_child_terminal_and_inherits_runtime_request() {
    let (child_started_tx, mut child_started_rx) = tokio::sync::mpsc::unbounded_channel();
    let (child_release_tx, child_release_rx) = tokio::sync::oneshot::channel();
    let backend = Arc::new(BlockingChildBackend {
        child_started: child_started_tx,
        child_release: tokio::sync::Mutex::new(Some(child_release_rx)),
    });
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::with_backend(backend).with_projection_store(store.clone());
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
                    text: "delegate".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: Some(app_server_protocol::RuntimeOptions {
                    capability_id: None,
                    stream: true,
                    event_name: Some("parent-event".to_string()),
                    queued_turn_id: Some("parent-queue".to_string()),
                    runtime_request: Some(app_server_protocol::RuntimeRequest {
                        provider_config: Some(app_server_protocol::RuntimeProviderConfig {
                            base_url: Some("http://127.0.0.1:43123/v1".to_string()),
                            ..app_server_protocol::RuntimeProviderConfig::default()
                        }),
                        provider_preference: Some("fixture-provider".to_string()),
                        model_preference: Some("fixture-model".to_string()),
                        metadata: Some(json!({ "fixture": "agent-control" })),
                        ..app_server_protocol::RuntimeRequest::default()
                    }),
                    expected_output: Some(json!({ "type": "parent-only" })),
                    structured_output: Some(
                        app_server_protocol::StructuredOutputContract::default(),
                    ),
                    output_schema: Some(json!({ "type": "object" })),
                }),
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

    let result = tokio::time::timeout(
        Duration::from_millis(200),
        gateway.gateway().execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                session_id: session.session_id.clone(),
                thread_id: session.thread_id.clone(),
                turn_id: turn.turn_id.clone(),
                call_id: "async-spawn-call".to_string(),
            },
            command: AgentControlCommand::SpawnAgent {
                task_name: "research".to_string(),
                message: "inspect the current owner".to_string(),
                fork_mode: SpawnAgentForkMode::None,
                model_overrides: SpawnAgentModelOverrides::default(),
            },
            cancel_token: None,
        }),
    )
    .await
    .expect("spawn_agent must return before child terminal")
    .expect("spawn result");

    let child_request = tokio::time::timeout(Duration::from_secs(1), child_started_rx.recv())
        .await
        .expect("child turn should start")
        .expect("child request");
    let child_options = child_request
        .runtime_options
        .expect("child runtime options");
    assert_eq!(child_options.capability_id, None);
    assert!(child_options.stream);
    assert_eq!(child_options.event_name, None);
    assert_eq!(child_options.queued_turn_id, None);
    assert_eq!(child_options.expected_output, None);
    assert_eq!(child_options.structured_output, None);
    assert_eq!(child_options.output_schema, None);
    let child_runtime_request = child_options
        .runtime_request
        .expect("child runtime request");
    assert_eq!(
        child_runtime_request.provider_preference.as_deref(),
        Some("fixture-provider")
    );
    assert_eq!(
        child_runtime_request.model_preference.as_deref(),
        Some("fixture-model")
    );
    assert_eq!(
        child_runtime_request
            .provider_config
            .as_ref()
            .and_then(|config| config.base_url.as_deref()),
        None
    );

    let child_identity = spawned_child_identity(&store, "parent-thread", "research").await;
    let child_thread = store
        .read_thread(ReadThreadParams {
            thread_id: child_identity.thread_id.clone(),
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        })
        .await
        .expect("child thread read")
        .expect("child thread");
    assert_eq!(
        uuid::Uuid::parse_str(child_identity.thread_id.as_str())
            .expect("AgentControl child UUID")
            .get_version_num(),
        7
    );
    assert_eq!(
        child_thread.session_id.as_str(),
        child_identity.thread_id.as_str()
    );
    let message_id = result.output["message_id"].as_str().expect("message id");
    let mailbox_item_id = super::super::agent_mailbox_delivery::mailbox_item_id(message_id);
    assert!(child_thread.turns.iter().any(|child_turn| {
        child_turn
            .items
            .iter()
            .any(|item| item.item_id.as_str() == mailbox_item_id)
            && !matches!(
                child_turn.status,
                agent_protocol::TurnStatus::Completed
                    | agent_protocol::TurnStatus::Failed
                    | agent_protocol::TurnStatus::Interrupted
            )
    }));

    child_release_tx.send(()).expect("release child");
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let thread = store
                .read_thread(ReadThreadParams {
                    thread_id: child_identity.thread_id.clone(),
                    include_archived: true,
                    turns_view: ThreadTurnsView::Full,
                })
                .await
                .expect("child thread read")
                .expect("child thread");
            if thread
                .turns
                .iter()
                .any(|child_turn| child_turn.status == agent_protocol::TurnStatus::Completed)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("child turn should complete after release");
}

#[test]
fn spawn_creates_canonical_child_and_open_edge() {
    let (_temp, core, store) = core();
    core.start_session(start_params("parent-session", "parent-thread"))
        .expect("parent");

    let response = block_on(core.create_open_agent_control_child_for_test(
        AgentControlSpawnRequest {
            parent_session_id: "parent-session".to_string(),
            child_session_id: Some("child-session".to_string()),
            child_thread_id: Some("child-thread".to_string()),
            fork_mode: SpawnAgentForkMode::None,
        },
    ))
    .expect("spawn child");

    assert_eq!(response.parent_thread_id, "parent-thread");
    assert_eq!(response.session.session_id, "child-session");
    assert_eq!(response.session.thread_id, "child-thread");
    assert_eq!(
        block_on(store.list_thread_spawn_children(
            ThreadId::new("parent-thread"),
            Some(ThreadSpawnEdgeStatus::Open),
        ))
        .expect("children"),
        vec![ThreadId::new("child-thread")]
    );
}

#[tokio::test]
async fn spawn_forks_all_last_n_or_no_parent_turns_into_canonical_history() {
    let (_temp, core, store) = core();
    let parent = core
        .start_session(start_params("parent-session", "parent-thread"))
        .expect("parent")
        .session;
    for (index, user, assistant) in [
        (1, "user one", "assistant one"),
        (2, "user two", "assistant two"),
        (3, "user three", "assistant three"),
    ] {
        append_completed_parent_turn(
            &core,
            &parent,
            &format!("parent-turn-{index}"),
            user,
            assistant,
        )
        .await;
    }

    for (session_id, thread_id, fork_mode, expected_users) in [
        (
            "child-all-session",
            "child-all-thread",
            SpawnAgentForkMode::FullHistory,
            vec!["user one", "user two", "user three"],
        ),
        (
            "child-last-session",
            "child-last-thread",
            SpawnAgentForkMode::LastNTurns(2),
            vec!["user two", "user three"],
        ),
        (
            "child-none-session",
            "child-none-thread",
            SpawnAgentForkMode::None,
            Vec::new(),
        ),
    ] {
        let expects_fork_lineage = fork_mode != SpawnAgentForkMode::None;
        core.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
            parent_session_id: parent.session_id.clone(),
            child_session_id: Some(session_id.to_string()),
            child_thread_id: Some(thread_id.to_string()),
            fork_mode,
        })
        .await
        .expect("spawn forked child");

        let child = core
            .read_session(AgentSessionReadParams {
                session_id: session_id.to_string(),
                history_limit: None,
                history_offset: None,
                history_before_message_id: None,
            })
            .expect("read forked child");
        assert_eq!(child.turns.len(), expected_users.len());
        let actual_users = {
            let state = core
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state.sessions.get(session_id).expect("stored child");
            child
                .turns
                .iter()
                .map(|turn| {
                    stored
                        .turn_inputs
                        .get(&turn.turn_id)
                        .map(|input| super::super::turn_start::user_input_text(input))
                        .expect("forked turn input")
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(actual_users, expected_users);
        let provider_history = {
            let state = core
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state.sessions.get(session_id).expect("stored child");
            crate::runtime::provider_history::provider_history_excluding_current_turn_input(
                stored,
                core.sidecar_store.as_deref(),
                "future-child-turn",
            )
            .expect("provider history")
        };
        let provider_text = provider_history
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|content| match content {
                model_provider::current_client::CurrentProviderContent::Text(text) => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let expected_provider_text = expected_users
            .iter()
            .flat_map(|user| {
                let suffix = user.strip_prefix("user ").expect("user fixture prefix");
                [
                    *user,
                    match suffix {
                        "one" => "assistant one",
                        "two" => "assistant two",
                        "three" => "assistant three",
                        _ => unreachable!("known fixture suffix"),
                    },
                ]
            })
            .collect::<Vec<_>>();
        assert_eq!(provider_text, expected_provider_text);
        assert!(provider_history
            .iter()
            .flat_map(|message| message.content.iter())
            .all(|content| !matches!(
                content,
                model_provider::current_client::CurrentProviderContent::Reasoning(_)
                    | model_provider::current_client::CurrentProviderContent::ToolCall(_)
                    | model_provider::current_client::CurrentProviderContent::ToolResult(_)
            )));

        let canonical = store
            .read_thread(ReadThreadParams {
                thread_id: ThreadId::new(thread_id),
                include_archived: true,
                turns_view: ThreadTurnsView::Full,
            })
            .await
            .expect("read canonical child")
            .expect("canonical child");
        assert_eq!(canonical.turns.len(), expected_users.len());
        assert_eq!(
            canonical.parent_thread_id,
            Some(ThreadId::new("parent-thread"))
        );
        assert_eq!(
            canonical.forked_from_id,
            expects_fork_lineage.then(|| ThreadId::new("parent-thread"))
        );
        assert!(canonical
            .turns
            .iter()
            .all(|turn| turn.turn_id.as_str().starts_with("fork-")));
        assert!(canonical
            .turns
            .iter()
            .flat_map(|turn| turn.items.iter())
            .all(|item| matches!(item.kind, ItemKind::UserMessage | ItemKind::AgentMessage)));
        assert!(canonical
            .turns
            .iter()
            .flat_map(|turn| turn.items.iter())
            .all(|item| item.status == ItemStatus::Completed));
        let assistant_items = canonical
            .turns
            .iter()
            .flat_map(|turn| turn.items.iter())
            .filter(|item| item.kind == ItemKind::AgentMessage)
            .collect::<Vec<_>>();
        assert_eq!(assistant_items.len(), expected_users.len());
        assert!(assistant_items.iter().all(|item| {
            matches!(
                &item.payload,
                ThreadItemPayload::AgentMessage { phase, .. }
                    if phase
                        == &Some(agent_protocol::response_item::MessagePhase::FinalAnswer)
            )
        }));
        assert!(assistant_items
            .iter()
            .all(|item| item.item_id.as_str().starts_with("item_fork-")));
        assert_eq!(
            assistant_items
                .iter()
                .map(|item| {
                    item.metadata["forkedFromTurnId"]
                        .as_str()
                        .expect("source Turn metadata")
                })
                .collect::<Vec<_>>(),
            expected_users
                .iter()
                .map(|user| match *user {
                    "user one" => "parent-turn-1",
                    "user two" => "parent-turn-2",
                    "user three" => "parent-turn-3",
                    _ => unreachable!("known fork user"),
                })
                .collect::<Vec<_>>()
        );
    }
}

#[tokio::test]
async fn spawn_forks_each_completed_final_answer_in_source_order() {
    let (_temp, core, store) = core();
    let parent = core
        .start_session(start_params("multi-parent-session", "multi-parent-thread"))
        .expect("parent")
        .session;
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: parent.session_id.clone(),
            turn_id: Some("multi-parent-turn".to_string()),
            input: AgentInput {
                text: "user input".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("start parent turn");
    core.append_external_runtime_events(
        &parent.session_id,
        Some("multi-parent-turn"),
        vec![
            RuntimeEvent::new(
                "message.delta",
                json!({"itemId": "final-one", "phase": "final_answer", "text": "first"}),
            ),
            RuntimeEvent::new(
                "message.completed",
                json!({"itemId": "final-one", "phase": "final_answer", "status": "completed", "text": "first"}),
            ),
            RuntimeEvent::new(
                "message.delta",
                json!({"itemId": "final-two", "phase": "final_answer", "text": "second"}),
            ),
            RuntimeEvent::new(
                "message.completed",
                json!({"itemId": "final-two", "phase": "final_answer", "status": "completed", "text": "second"}),
            ),
            RuntimeEvent::new("turn.completed", json!({})),
        ],
    )
    .expect("complete parent turn");

    core.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
        parent_session_id: parent.session_id,
        child_session_id: Some("multi-child-session".to_string()),
        child_thread_id: Some("multi-child-thread".to_string()),
        fork_mode: SpawnAgentForkMode::FullHistory,
    })
    .await
    .expect("spawn forked child");

    let canonical = store
        .read_thread(ReadThreadParams {
            thread_id: ThreadId::new("multi-child-thread"),
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        })
        .await
        .expect("read child")
        .expect("child thread");
    let assistant_items = canonical.turns[0]
        .items
        .iter()
        .filter(|item| item.kind == ItemKind::AgentMessage)
        .collect::<Vec<_>>();
    assert_eq!(assistant_items.len(), 2);
    assert_eq!(
        assistant_items[0].metadata["forkedFromItemId"],
        "item_final-one"
    );
    assert_eq!(
        assistant_items[1].metadata["forkedFromItemId"],
        "item_final-two"
    );
    assert!(assistant_items
        .iter()
        .all(|item| item.status == ItemStatus::Completed));
}

#[test]
fn spawn_fails_closed_without_projection_store() {
    let core = RuntimeCore::default();
    core.start_session(start_params("parent-session", "parent-thread"))
        .expect("parent");

    let error = block_on(
        core.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
            parent_session_id: "parent-session".to_string(),
            child_session_id: None,
            child_thread_id: None,
            fork_mode: SpawnAgentForkMode::None,
        }),
    )
    .expect_err("projection store is required");
    assert!(error
        .to_string()
        .contains("agent control requires canonical ProjectionStore"));
}

#[test]
fn spawn_reservation_failure_never_creates_child_session_or_thread() {
    let (_temp, core, store) = core();
    core.start_session(start_params("parent-session", "parent-thread"))
        .expect("parent");
    block_on(store.upsert_thread_spawn_edge(
        ThreadId::new("ancestor-thread"),
        ThreadId::new("parent-thread"),
        ThreadSpawnEdgeStatus::Open,
    ))
    .expect("precondition edge");

    let error = block_on(
        core.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
            parent_session_id: "parent-session".to_string(),
            child_session_id: Some("unlinked-child-session".to_string()),
            child_thread_id: Some("ancestor-thread".to_string()),
            fork_mode: SpawnAgentForkMode::None,
        }),
    )
    .expect_err("cyclic graph must reject child edge");
    assert!(error
        .to_string()
        .contains("failed to reserve canonical child thread spawn"));
    assert!(matches!(
        core.read_session(AgentSessionReadParams {
            session_id: "unlinked-child-session".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        }),
        Err(RuntimeCoreError::SessionNotFound(session_id)) if session_id == "unlinked-child-session"
    ));
    assert!(block_on(store.read_thread(ReadThreadParams {
        thread_id: ThreadId::new("ancestor-thread"),
        include_archived: false,
        turns_view: ThreadTurnsView::NotLoaded,
    }))
    .expect("read failed child thread")
    .is_none());
}

#[test]
fn unloaded_parent_fails_without_legacy_session_fallback() {
    let temp = tempfile::tempdir().expect("tempdir");
    let path = temp.path().join("projection.sqlite");
    let store = Arc::new(ProjectionStore::initialize(&path).expect("store"));
    let core = RuntimeCore::default().with_projection_store(store);
    core.start_session(start_params("parent-session", "parent-thread"))
        .expect("parent");
    drop(core);

    let restarted = RuntimeCore::default().with_projection_store(Arc::new(
        ProjectionStore::initialize(path).expect("reopen store"),
    ));
    assert!(matches!(
        block_on(restarted.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
            parent_session_id: "parent-session".to_string(),
            child_session_id: None,
            child_thread_id: None,
            fork_mode: SpawnAgentForkMode::None,
        })),
        Err(RuntimeCoreError::SessionNotFound(session_id)) if session_id == "parent-session"
    ));
}

#[tokio::test]
async fn gateway_rejects_caller_outside_its_bound_turn() {
    let (_temp, core, _store) = core();
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
                    text: "work".to_string(),
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

    let error = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                session_id: session.session_id.clone(),
                thread_id: session.thread_id.clone(),
                turn_id: "other-turn".to_string(),
                call_id: "call-1".to_string(),
            },
            command: AgentControlCommand::ListAgents { path_prefix: None },
            cancel_token: None,
        })
        .await
        .expect_err("bound gateway must reject another turn");

    assert!(error
        .to_string()
        .contains("outside its per-turn gateway scope"));
}

#[tokio::test]
async fn spawn_gateway_projects_and_starts_the_initial_child_task_before_success() {
    let (_temp, core, store) = core();
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
        .expect("parent turn")
        .response
        .turn;
    let gateway =
        core.agent_control_gateway_for_turn(&session, &turn, RuntimeHostContext::default());

    let result = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                session_id: session.session_id.clone(),
                thread_id: session.thread_id.clone(),
                turn_id: turn.turn_id.clone(),
                call_id: "spawn-call".to_string(),
            },
            command: AgentControlCommand::SpawnAgent {
                task_name: "research".to_string(),
                message: "inspect the current owner".to_string(),
                fork_mode: SpawnAgentForkMode::None,
                model_overrides: SpawnAgentModelOverrides::default(),
            },
            cancel_token: None,
        })
        .await
        .expect("spawn result");

    let child_identity = spawned_child_identity(&store, "parent-thread", "research").await;
    let child_thread_id = child_identity.thread_id.clone();
    let child_session_id = store
        .read_thread(ReadThreadParams {
            thread_id: child_thread_id.clone(),
            include_archived: true,
            turns_view: ThreadTurnsView::NotLoaded,
        })
        .await
        .expect("child thread")
        .expect("child thread exists")
        .session_id
        .to_string();
    let message_id = result.output["message_id"].as_str().expect("message id");
    assert_eq!(result.output["task_name"], "/root/research");
    assert!(result.output.get("nickname").is_none());
    assert_eq!(result.projection_facts.len(), 1);
    assert_eq!(result.projection_facts[0].target_thread_id, child_thread_id);
    assert_eq!(
        result.projection_facts[0].activity,
        SubAgentProjectionActivity::Started
    );
    assert_eq!(
        result.projection_facts[0].detail.as_deref(),
        Some("/root/research")
    );
    assert!(block_on(store.read_agent_identity(child_thread_id.clone()))
        .expect("child identity")
        .is_some());
    let initial_turn_id = super::super::agent_mailbox_delivery::mailbox_turn_id(message_id);
    let child = wait_for_session_turn(&core, &child_session_id, Some(&initial_turn_id)).await;
    assert!(child
        .turns
        .iter()
        .any(|child_turn| child_turn.turn_id == initial_turn_id));
    let thread = block_on(store.read_thread(ReadThreadParams {
        thread_id: child_thread_id,
        include_archived: true,
        turns_view: ThreadTurnsView::Full,
    }))
    .expect("child thread")
    .expect("child thread exists");
    let initial_item_id = super::super::agent_mailbox_delivery::mailbox_item_id(message_id);
    let item_turn_ids = thread
        .turns
        .iter()
        .filter(|child_turn| {
            child_turn
                .items
                .iter()
                .any(|item| item.item_id.as_str() == initial_item_id)
        })
        .map(|child_turn| child_turn.turn_id.to_string())
        .collect::<Vec<_>>();
    assert_eq!(item_turn_ids, vec![initial_turn_id], "{thread:#?}");
}

#[tokio::test]
async fn wait_gateway_returns_interrupted_when_the_turn_is_cancelled() {
    let (_temp, core, _store) = core();
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
                    text: "wait".to_string(),
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
    let cancel_token = tokio_util::sync::CancellationToken::new();
    cancel_token.cancel();

    let result = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                session_id: session.session_id,
                thread_id: session.thread_id,
                turn_id: turn.turn_id,
                call_id: "wait-call".to_string(),
            },
            command: AgentControlCommand::WaitAgent { timeout_ms: 1_000 },
            cancel_token: Some(cancel_token),
        })
        .await
        .expect("wait result");

    assert_eq!(result.output["message"], "Wait interrupted.");
    assert_eq!(result.output["timed_out"], false);
    assert!(result.projection_facts.is_empty());
}

#[tokio::test]
async fn wait_agent_wakes_for_new_runtimecore_queued_steer() {
    let (_temp, core, _store) = core();
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
                    text: "work".to_string(),
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
    let caller = AgentControlCaller {
        session_id: session.session_id.clone(),
        thread_id: session.thread_id.clone(),
        turn_id: turn.turn_id.clone(),
        call_id: "call-wait".to_string(),
    };
    let wait_task = tokio::spawn(async move {
        gateway
            .gateway()
            .execute(AgentControlGatewayRequest {
                caller,
                command: AgentControlCommand::WaitAgent { timeout_ms: 500 },
                cancel_token: None,
            })
            .await
    });
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: session.session_id.clone(),
            turn_id: Some("queued-steer".to_string()),
            input: AgentInput {
                text: "steer".to_string(),
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
    let result = tokio::time::timeout(std::time::Duration::from_secs(1), wait_task)
        .await
        .expect("wait must observe the new steer")
        .expect("wait task")
        .expect("wait result");

    assert_eq!(result.output["timed_out"], false);
    assert!(result.projection_facts.is_empty());
}

#[tokio::test]
async fn gateway_persists_root_identity_and_hides_closed_child_targets() {
    let (_temp, core, store) = core();
    let session = core
        .start_session(start_params("root-session", "root-thread"))
        .expect("root")
        .session;
    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: session.session_id.clone(),
                turn_id: Some("root-turn".to_string()),
                input: AgentInput {
                    text: "work".to_string(),
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
    let gateway =
        core.agent_control_gateway_for_turn(&session, &turn, RuntimeHostContext::default());
    let caller = AgentControlCaller {
        session_id: session.session_id.clone(),
        thread_id: session.thread_id.clone(),
        turn_id: turn.turn_id.clone(),
        call_id: "call-list".to_string(),
    };

    gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: caller.clone(),
            command: AgentControlCommand::ListAgents { path_prefix: None },
            cancel_token: None,
        })
        .await
        .expect("list root");
    assert_eq!(
        store
            .read_agent_identity(ThreadId::new("root-thread"))
            .await
            .expect("read durable root identity")
            .expect("root identity")
            .agent_path,
        "/root"
    );

    core.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
        parent_session_id: session.session_id.clone(),
        child_session_id: Some("child-session".to_string()),
        child_thread_id: Some("child-thread".to_string()),
        fork_mode: SpawnAgentForkMode::None,
    })
    .await
    .expect("child");
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new("root-thread"),
            thread_id: ThreadId::new("child-thread"),
            agent_path: "/root/child".to_string(),
            nickname: None,
            role: None,
            last_task_message: Some("work".to_string()),
        })
        .await
        .expect("child identity");
    store
        .set_thread_spawn_edge_status(ThreadId::new("child-thread"), ThreadSpawnEdgeStatus::Closed)
        .await
        .expect("close child");

    let error = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller,
            command: AgentControlCommand::SendMessage {
                target: "child".to_string(),
                message: "continue".to_string(),
            },
            cancel_token: None,
        })
        .await
        .expect_err("closed child must not be addressable");
    assert!(error
        .to_string()
        .contains("not in the current durable root-thread tree"));
    assert!(store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("root-thread"),
            ThreadId::new("child-thread"),
        )
        .await
        .expect("pending child mail")
        .is_empty());
}

#[tokio::test]
async fn direct_message_publishes_activity_only_to_an_existing_recipient_session_loop() {
    let (_temp, core, store) = core();
    let session = core
        .start_session(start_params("root-session", "root-thread"))
        .expect("root")
        .session;
    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: session.session_id.clone(),
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
    let child = core
        .create_open_agent_control_child_for_test(AgentControlSpawnRequest {
            parent_session_id: session.session_id.clone(),
            child_session_id: Some("child-session".to_string()),
            child_thread_id: Some("child-thread".to_string()),
            fork_mode: SpawnAgentForkMode::None,
        })
        .await
        .expect("child");
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new(session.thread_id.clone()),
            thread_id: ThreadId::new(child.session.thread_id.clone()),
            agent_path: "/root/child".to_string(),
            nickname: None,
            role: None,
            last_task_message: Some("work".to_string()),
        })
        .await
        .expect("child identity");
    let gateway =
        core.agent_control_gateway_for_turn(&session, &turn, RuntimeHostContext::default());
    let caller = AgentControlCaller {
        session_id: session.session_id.clone(),
        thread_id: session.thread_id.clone(),
        turn_id: turn.turn_id.clone(),
        call_id: "cold-send".to_string(),
    };

    gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: caller.clone(),
            command: AgentControlCommand::SendMessage {
                target: "child".to_string(),
                message: "persist without a live loop".to_string(),
            },
            cancel_token: None,
        })
        .await
        .expect("queue message without recipient loop");
    assert!(
        !core
            .session_loops
            .notify_inter_agent_communication(
                &child.session.session_id,
                RuntimeSessionInterAgentInput {
                    message_id: "cold-notification-probe".to_string(),
                    root_thread_id: session.thread_id.clone(),
                    sender_thread_id: session.thread_id.clone(),
                    recipient_thread_id: child.session.thread_id.clone(),
                    content: "probe".to_string(),
                    kind: RuntimeSessionInterAgentMessageKind::Message,
                    source_turn_id: Some(turn.turn_id.clone()),
                    result_status: None,
                    delivery_mode: RuntimeSessionInterAgentDeliveryMode::QueueOnly,
                },
            )
            .await
            .expect("check absent recipient loop"),
        "durable mailbox append must not create a recipient session loop"
    );

    let (activity_tx, activity_rx) = tokio::sync::oneshot::channel();
    let activity_tx = Arc::new(tokio::sync::Mutex::new(Some(activity_tx)));
    let task = RuntimeSessionClosureTask::new(
        "mailbox-activity-turn",
        Vec::new(),
        move |context, _input, _cancellation_token| {
            let activity_tx = Arc::clone(&activity_tx);
            Box::pin(async move {
                context.wait_for_pending_input().await;
                if let Some(activity_tx) = activity_tx.lock().await.take() {
                    let _ = activity_tx.send(());
                }
                Ok(())
            })
        },
    );
    let recipient_loop = core
        .session_loops
        .get_or_create(&child.session.session_id, &child.session.thread_id)
        .await
        .expect("bind recipient session actor");
    let submission = recipient_loop
        .submit_replacing(Arc::new(task))
        .await
        .expect("start recipient mailbox waiter");

    gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                call_id: "live-send".to_string(),
                ..caller
            },
            command: AgentControlCommand::SendMessage {
                target: "child".to_string(),
                message: "wake the existing loop".to_string(),
            },
            cancel_token: None,
        })
        .await
        .expect("queue message with recipient loop");
    tokio::time::timeout(Duration::from_secs(1), activity_rx)
        .await
        .expect("recipient loop activity timeout")
        .expect("recipient loop activity sender");
    tokio::time::timeout(Duration::from_secs(1), submission.completion)
        .await
        .expect("recipient waiter completion timeout")
        .expect("recipient waiter completion sender")
        .expect("recipient waiter completion");
}

#[tokio::test]
async fn gateway_queue_followup_and_interrupt_keep_the_durable_contract() {
    let (_temp, core, store) = core();
    let session = core
        .start_session(start_params("root-session", "root-thread"))
        .expect("root")
        .session;
    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: session.session_id.clone(),
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
    let gateway =
        core.agent_control_gateway_for_turn(&session, &turn, RuntimeHostContext::default());
    let caller = AgentControlCaller {
        session_id: session.session_id.clone(),
        thread_id: session.thread_id.clone(),
        turn_id: turn.turn_id.clone(),
        call_id: "spawn-call".to_string(),
    };
    gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: caller.clone(),
            command: AgentControlCommand::SpawnAgent {
                task_name: "research".to_string(),
                message: "inspect the current owner".to_string(),
                fork_mode: SpawnAgentForkMode::None,
                model_overrides: SpawnAgentModelOverrides::default(),
            },
            cancel_token: None,
        })
        .await
        .expect("spawn child");

    let child_identity = spawned_child_identity(&store, "root-thread", "research").await;
    let child_thread_id = child_identity.thread_id.clone();
    let child_session_id = store
        .read_thread(ReadThreadParams {
            thread_id: child_thread_id.clone(),
            include_archived: true,
            turns_view: ThreadTurnsView::NotLoaded,
        })
        .await
        .expect("child thread")
        .expect("child thread exists")
        .session_id
        .to_string();
    let child = wait_for_session_turn(&core, &child_session_id, None).await;
    let initial_turn_count = child.turns.len();

    let queued = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                call_id: "send-call".to_string(),
                ..caller.clone()
            },
            command: AgentControlCommand::SendMessage {
                target: "research".to_string(),
                message: "wait for the review".to_string(),
            },
            cancel_token: None,
        })
        .await
        .expect("queue-only message");
    let queued_message_id = queued.output["message_id"].as_str().expect("message id");
    assert_eq!(queued.projection_facts.len(), 1);
    assert_eq!(queued.projection_facts[0].target_thread_id, child_thread_id);
    assert_eq!(
        queued.projection_facts[0].activity,
        SubAgentProjectionActivity::Interacted
    );
    assert_eq!(
        queued.projection_facts[0].detail.as_deref(),
        Some("/root/research")
    );
    let child_after_queue = core
        .read_session(AgentSessionReadParams {
            session_id: child_session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("child session after queue");
    assert_eq!(child_after_queue.turns.len(), initial_turn_count);
    assert!(store
        .list_pending_agent_mailbox_messages(ThreadId::new("root-thread"), child_thread_id.clone(),)
        .await
        .expect("pending mailbox")
        .iter()
        .any(|message| {
            message.message_id == queued_message_id
                && message.delivery_mode == AgentMailboxDeliveryMode::QueueOnly
        }));

    let root_followup = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                call_id: "followup-root-call".to_string(),
                ..caller.clone()
            },
            command: AgentControlCommand::FollowupTask {
                target: "/root".to_string(),
                message: "forbidden".to_string(),
            },
            cancel_token: None,
        })
        .await
        .expect_err("followup must not target root");
    assert!(root_followup
        .to_string()
        .contains("followup_task cannot target the root agent"));

    let interrupted = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                call_id: "interrupt-call".to_string(),
                ..caller.clone()
            },
            command: AgentControlCommand::InterruptAgent {
                target: "research".to_string(),
            },
            cancel_token: None,
        })
        .await
        .expect("interrupt child");
    assert_eq!(interrupted.output["previous_status"], "pending_init");
    assert_eq!(interrupted.projection_facts.len(), 1);
    assert_eq!(
        interrupted.projection_facts[0].target_thread_id,
        child_thread_id
    );
    assert_eq!(
        interrupted.projection_facts[0].activity,
        SubAgentProjectionActivity::Interrupted
    );
    assert_eq!(
        interrupted.projection_facts[0].detail.as_deref(),
        Some("/root/research")
    );
    assert_eq!(
        store
            .list_thread_spawn_children(
                ThreadId::new("root-thread"),
                Some(ThreadSpawnEdgeStatus::Open),
            )
            .await
            .expect("open child edge"),
        vec![child_thread_id.clone()]
    );

    let followup = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                call_id: "followup-child-call".to_string(),
                ..caller
            },
            command: AgentControlCommand::FollowupTask {
                target: "/root/research".to_string(),
                message: "continue after the interrupt".to_string(),
            },
            cancel_token: None,
        })
        .await
        .expect("followup child");
    assert_eq!(followup.projection_facts.len(), 1);
    assert_eq!(
        followup.projection_facts[0].target_thread_id,
        child_thread_id
    );
    assert_eq!(
        followup.projection_facts[0].activity,
        SubAgentProjectionActivity::Interacted
    );
    assert_eq!(
        followup.projection_facts[0].detail.as_deref(),
        Some("/root/research")
    );
}

#[tokio::test]
async fn list_gateway_sorts_prefixes_and_isolates_root_trees() {
    let (_temp, core, store) = core();
    let session = core
        .start_session(start_params("root-session", "root-thread"))
        .expect("root")
        .session;
    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: session.session_id.clone(),
                turn_id: Some("root-turn".to_string()),
                input: AgentInput {
                    text: "list".to_string(),
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
    for (task_name, child_session_id, child_thread_id) in [
        ("zeta", "zeta-session", "zeta-thread"),
        ("alpha", "alpha-session", "alpha-thread"),
    ] {
        core.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
            parent_session_id: session.session_id.clone(),
            child_session_id: Some(child_session_id.to_string()),
            child_thread_id: Some(child_thread_id.to_string()),
            fork_mode: SpawnAgentForkMode::None,
        })
        .await
        .expect("child");
        store
            .upsert_agent_identity(thread_store::AgentIdentity {
                root_thread_id: ThreadId::new("root-thread"),
                thread_id: ThreadId::new(child_thread_id),
                agent_path: format!("/root/{task_name}"),
                nickname: None,
                role: None,
                last_task_message: None,
            })
            .await
            .expect("child identity");
    }
    core.start_session(start_params("other-session", "other-thread"))
        .expect("other root");
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new("other-thread"),
            thread_id: ThreadId::new("other-thread"),
            agent_path: "/root".to_string(),
            nickname: None,
            role: None,
            last_task_message: None,
        })
        .await
        .expect("other identity");

    let gateway =
        core.agent_control_gateway_for_turn(&session, &turn, RuntimeHostContext::default());
    let caller = AgentControlCaller {
        session_id: session.session_id,
        thread_id: session.thread_id,
        turn_id: turn.turn_id,
        call_id: "list-call".to_string(),
    };
    let listed = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: caller.clone(),
            command: AgentControlCommand::ListAgents { path_prefix: None },
            cancel_token: None,
        })
        .await
        .expect("list root tree");
    assert!(listed.projection_facts.is_empty());
    assert_eq!(
        listed.output["agents"]
            .as_array()
            .expect("agents")
            .iter()
            .filter_map(|entry| entry["agent_name"].as_str())
            .collect::<Vec<_>>(),
        vec!["/root", "/root/alpha", "/root/zeta"]
    );

    let prefixed = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller,
            command: AgentControlCommand::ListAgents {
                path_prefix: Some("alpha".to_string()),
            },
            cancel_token: None,
        })
        .await
        .expect("list relative prefix");
    assert!(prefixed.projection_facts.is_empty());
    assert_eq!(
        prefixed.output["agents"]
            .as_array()
            .expect("agents")
            .iter()
            .filter_map(|entry| entry["agent_name"].as_str())
            .collect::<Vec<_>>(),
        vec!["/root/alpha"]
    );
}
