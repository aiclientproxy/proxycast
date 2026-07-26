use super::support::*;
use super::*;

use crate::runtime::agent_control::AgentControlSpawnRequest;
use agent_protocol::ThreadId;
use async_trait::async_trait;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use thread_store::{
    AgentIdentity, AgentIdentityStore, AgentMailboxDeliveryMode, AgentMailboxDeliveryStatus,
    AgentMailboxMessage, AgentMailboxMessageKind, AgentMailboxStore,
    AppendAgentMailboxMessageParams,
};
use tool_runtime::agent_control::SpawnAgentForkMode;

const FIXTURE_API_KEY: &str = "fixture-secret-must-not-reach-durable-log";
const FIXTURE_BASE_URL: &str = "https://user:pass@provider.example/v1?api_key=fixture-secret#token";

struct PersistentRuntimeFixture {
    _temp: tempfile::TempDir,
    event_log_writer: Arc<EventLogWriter>,
    projection_store: Arc<ProjectionStore>,
}

impl PersistentRuntimeFixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().expect("tempdir");
        let roots = StorageRoots::initialize(temp.path(), temp.path().join("app-server"))
            .expect("storage roots");
        let event_log_writer =
            Arc::new(EventLogWriter::new(&roots.event_log_root).expect("event log writer"));
        let projection_store = Arc::new(
            ProjectionStore::initialize(&roots.projection_db_path).expect("projection store"),
        );
        Self {
            _temp: temp,
            event_log_writer,
            projection_store,
        }
    }

    fn core(&self, backend: Arc<dyn ExecutionBackend>) -> RuntimeCore {
        RuntimeCore::with_backend(backend)
            .with_event_log_writer(self.event_log_writer.clone())
            .with_projection_store(self.projection_store.clone())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ObservedRecoveryRequest {
    session_id: String,
    turn_id: String,
    input: String,
    provider: Option<String>,
    model: Option<String>,
    metadata: Option<serde_json::Value>,
    queued_turn_id: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
}

#[derive(Default)]
struct SelectiveRecoveryBackend {
    pending_once: Mutex<BTreeSet<String>>,
    fail_once: Mutex<BTreeSet<String>>,
    requests: Mutex<Vec<ObservedRecoveryRequest>>,
}

impl SelectiveRecoveryBackend {
    fn pending_once_for<I, S>(session_ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            pending_once: Mutex::new(session_ids.into_iter().map(Into::into).collect()),
            fail_once: Mutex::new(BTreeSet::new()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn fail_once_for<I, S>(session_ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            pending_once: Mutex::new(BTreeSet::new()),
            fail_once: Mutex::new(session_ids.into_iter().map(Into::into).collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<ObservedRecoveryRequest> {
        self.requests
            .lock()
            .expect("recovery requests mutex poisoned")
            .clone()
    }

    fn observe_request(&self, request: &ExecutionRequest) -> ObservedRecoveryRequest {
        let observed = ObservedRecoveryRequest {
            session_id: request.session.session_id.clone(),
            turn_id: request.turn.turn_id.clone(),
            input: request.input.concat_text(),
            provider: request.provider_preference().map(str::to_string),
            model: request.model_preference().map(str::to_string),
            metadata: request.runtime_metadata().cloned(),
            queued_turn_id: request.queued_turn_id.clone(),
            api_key: request
                .runtime_options
                .as_ref()
                .and_then(|options| options.runtime_request.as_ref())
                .and_then(|runtime_request| runtime_request.provider_config.as_ref())
                .and_then(|provider_config| provider_config.api_key.clone()),
            base_url: request
                .runtime_options
                .as_ref()
                .and_then(|options| options.runtime_request.as_ref())
                .and_then(|runtime_request| runtime_request.provider_config.as_ref())
                .and_then(|provider_config| provider_config.base_url.clone()),
        };
        self.requests
            .lock()
            .expect("recovery requests mutex poisoned")
            .push(observed.clone());
        observed
    }
}

#[async_trait]
impl ExecutionBackend for SelectiveRecoveryBackend {
    async fn preflight_turn(
        &self,
        request: &ExecutionRequest,
        _first_sampling_turn: bool,
    ) -> Result<(), RuntimeCoreError> {
        let observed = self.observe_request(request);
        if self
            .pending_once
            .lock()
            .expect("pending route sessions mutex poisoned")
            .remove(&observed.session_id)
        {
            return Err(RuntimeCoreError::PendingRoute {
                session_id: observed.session_id,
                provider: observed.provider,
                model: observed.model,
                reason_code: "provider_and_model_missing".to_string(),
            });
        }
        Ok(())
    }

    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let session_id = request.session.session_id.clone();

        if self
            .fail_once
            .lock()
            .expect("failed recovery sessions mutex poisoned")
            .remove(&session_id)
        {
            return Err(RuntimeCoreError::Backend(format!(
                "fixture recovery failure for {session_id}"
            )));
        }

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

fn route_options(marker: &str, include_api_key: bool) -> RuntimeOptions {
    RuntimeOptions {
        event_name: Some(format!("pending-work-{marker}")),
        runtime_request: Some(RuntimeRequest {
            provider_config: Some(RuntimeProviderConfig {
                provider_id: Some(format!("provider-id-{marker}")),
                provider_name: Some(format!("provider-{marker}")),
                model_name: Some(format!("model-{marker}")),
                api_key: include_api_key.then(|| FIXTURE_API_KEY.to_string()),
                base_url: Some(FIXTURE_BASE_URL.to_string()),
                ..RuntimeProviderConfig::default()
            }),
            provider_preference: Some(format!("provider-{marker}")),
            model_preference: Some(format!("model-{marker}")),
            collaboration_mode: Some(agent_protocol::CollaborationMode {
                mode: agent_protocol::ModeKind::Plan,
                settings: agent_protocol::CollaborationModeSettings {
                    model: format!("model-{marker}"),
                    reasoning_effort: None,
                    developer_instructions: None,
                },
            }),
            metadata: Some(json!({
                "clientUserMessageId": format!("client-message-{marker}"),
                "fixtureMarker": marker,
                "nested": { "source": "must-not-enter-durable-intent" }
            })),
            ..RuntimeRequest::default()
        }),
        ..RuntimeOptions::default()
    }
}

async fn seed_queued_session(
    core: &RuntimeCore,
    session_id: &str,
    thread_id: &str,
    queued_turns: &[(&str, &str, RuntimeOptions)],
) {
    core.start_session(AgentSessionStartParams {
        session_id: Some(session_id.to_string()),
        thread_id: Some(thread_id.to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("start queued session");
    let active_turn_id = format!("{session_id}-active");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: session_id.to_string(),
            turn_id: Some(active_turn_id.clone()),
            input: AgentInput {
                text: format!("active input for {session_id}"),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("start active turn");
    for (turn_id, input, runtime_options) in queued_turns {
        core.start_turn(
            AgentSessionTurnStartParams {
                session_id: session_id.to_string(),
                turn_id: Some((*turn_id).to_string()),
                input: AgentInput {
                    text: (*input).to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: Some(runtime_options.clone()),
                queue_if_busy: true,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("queue turn");
    }
    core.append_external_runtime_events(
        session_id,
        Some(&active_turn_id),
        vec![RuntimeEvent::new("turn.completed", json!({}))],
    )
    .expect("complete active turn");
}

fn assert_session_not_hydrated(core: &RuntimeCore, session_id: &str) {
    assert!(
        !core
            .state
            .lock()
            .expect("runtime state mutex poisoned")
            .sessions
            .contains_key(session_id),
        "restarted core must discover {session_id} from durable state"
    );
}

fn queued_turn_ids(core: &RuntimeCore, session_id: &str) -> Vec<String> {
    core.state
        .lock()
        .expect("runtime state mutex poisoned")
        .sessions
        .get(session_id)
        .unwrap_or_else(|| panic!("missing hydrated session {session_id}"))
        .turns
        .iter()
        .filter(|turn| turn.status == AgentTurnStatus::Queued)
        .map(|turn| turn.turn_id.clone())
        .collect()
}

#[tokio::test]
async fn cold_restart_recovers_durable_route_without_persisting_direct_credentials() {
    let fixture = PersistentRuntimeFixture::new();
    let setup = fixture.core(Arc::new(RecordingBackend::default()));
    seed_queued_session(
        &setup,
        "cold-route-session",
        "cold-route-thread",
        &[(
            "cold-route-queued",
            "continue with the durable route",
            route_options("cold", true),
        )],
    )
    .await;

    let records = fixture
        .event_log_writer
        .read_session_events("cold-route-session")
        .expect("read durable event log");
    let queue_event = records
        .iter()
        .map(|record| &record.event)
        .find(|event| {
            event.event_type == "queue.added"
                && event.turn_id.as_deref() == Some("cold-route-queued")
        })
        .expect("durable queue intent event");
    assert!(
        !queue_event.payload.to_string().contains(FIXTURE_API_KEY),
        "direct API key must not enter the durable queue intent"
    );
    assert!(
        !queue_event.payload.to_string().contains(FIXTURE_BASE_URL),
        "direct provider base URL must not enter the durable queue intent"
    );
    assert!(
        queue_event
            .payload
            .pointer("/queuedTurnIntent/runtimeOptions/runtimeRequest/providerConfig/apiKey")
            .is_none_or(serde_json::Value::is_null),
        "durable provider config must redact the direct API key"
    );
    drop(setup);

    let backend = Arc::new(SelectiveRecoveryBackend::default());
    let restarted = fixture.core(backend.clone());
    assert_session_not_hydrated(&restarted, "cold-route-session");
    restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect("recover durable queued route");

    let requests = backend.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].session_id, "cold-route-session");
    assert_eq!(requests[0].turn_id, "cold-route-queued");
    assert_eq!(requests[0].input, "continue with the durable route");
    assert_eq!(requests[0].provider.as_deref(), Some("provider-cold"));
    assert_eq!(requests[0].model.as_deref(), Some("model-cold"));
    assert_eq!(
        requests[0]
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("clientUserMessageId"))
            .and_then(serde_json::Value::as_str),
        Some("client-message-cold")
    );
    assert!(requests[0]
        .metadata
        .as_ref()
        .is_some_and(
            |metadata| metadata.get("fixtureMarker").is_none() && metadata.get("nested").is_none()
        ));
    assert_eq!(
        requests[0].queued_turn_id.as_deref(),
        Some("cold-route-queued")
    );
    assert_eq!(requests[0].api_key, None);
    assert_eq!(requests[0].base_url, None);
    assert!(queued_turn_ids(&restarted, "cold-route-session").is_empty());
}

#[tokio::test]
async fn queued_user_fifo_runs_before_trigger_turn_mailbox_for_the_same_session() {
    let fixture = PersistentRuntimeFixture::new();
    let setup = fixture.core(Arc::new(RecordingBackend::default()));
    let parent = setup
        .start_session(AgentSessionStartParams {
            session_id: Some("priority-root-session".to_string()),
            thread_id: Some("priority-root-thread".to_string()),
            app_id: "agent-chat".to_string(),
            workspace_id: Some("workspace-current".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("start root session")
        .session;
    fixture
        .projection_store
        .upsert_agent_identity(AgentIdentity {
            root_thread_id: ThreadId::new("priority-root-thread"),
            thread_id: ThreadId::new("priority-root-thread"),
            agent_path: "/root".to_string(),
            nickname: None,
            role: None,
            last_task_message: None,
        })
        .await
        .expect("persist root identity");
    let child = setup
        .create_open_agent_control_child_for_test(AgentControlSpawnRequest {
            parent_session_id: parent.session_id,
            child_session_id: Some("priority-child-session".to_string()),
            child_thread_id: Some("priority-child-thread".to_string()),
            fork_mode: SpawnAgentForkMode::None,
        })
        .await
        .expect("create open child")
        .session;
    fixture
        .projection_store
        .upsert_agent_identity(AgentIdentity {
            root_thread_id: ThreadId::new("priority-root-thread"),
            thread_id: ThreadId::new("priority-child-thread"),
            agent_path: "/root/priority".to_string(),
            nickname: None,
            role: None,
            last_task_message: None,
        })
        .await
        .expect("persist child identity");

    let active_turn_id = "priority-child-active";
    setup
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: child.session_id.clone(),
                turn_id: Some(active_turn_id.to_string()),
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
        .expect("start active child turn");
    for (turn_id, input, marker) in [
        ("priority-user-first", "first user follow-up", "first"),
        ("priority-user-second", "second user follow-up", "second"),
    ] {
        setup
            .start_turn(
                AgentSessionTurnStartParams {
                    session_id: child.session_id.clone(),
                    turn_id: Some(turn_id.to_string()),
                    input: AgentInput {
                        text: input.to_string(),
                        attachments: Vec::new(),
                    },
                    runtime_options: Some(route_options(marker, false)),
                    queue_if_busy: true,
                    skip_pre_submit_resume: false,
                },
                RuntimeHostContext::default(),
            )
            .await
            .expect("queue child user turn");
    }
    setup
        .append_external_runtime_events(
            &child.session_id,
            Some(active_turn_id),
            vec![RuntimeEvent::new("turn.completed", json!({}))],
        )
        .expect("complete active child turn");
    fixture
        .projection_store
        .append_agent_mailbox_message(AppendAgentMailboxMessageParams {
            message: AgentMailboxMessage {
                message_id: "priority-trigger-mail".to_string(),
                root_thread_id: ThreadId::new("priority-root-thread"),
                sender_thread_id: ThreadId::new("priority-root-thread"),
                recipient_thread_id: ThreadId::new("priority-child-thread"),
                content: "mailbox work after user follow-ups".to_string(),
                kind: AgentMailboxMessageKind::Message,
                source_turn_id: None,
                result_status: None,
                delivery_mode: AgentMailboxDeliveryMode::TriggerTurn,
                delivery_status: AgentMailboxDeliveryStatus::Pending,
                created_at_ms: 1,
                delivered_at_ms: None,
            },
        })
        .await
        .expect("append trigger mailbox");
    drop(setup);

    let backend = Arc::new(SelectiveRecoveryBackend::default());
    let restarted = fixture.core(backend.clone());
    assert_session_not_hydrated(&restarted, "priority-child-session");
    restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect("recover queued child work");

    assert_eq!(
        backend
            .requests()
            .iter()
            .map(|request| request.turn_id.clone())
            .collect::<Vec<_>>(),
        vec![
            "priority-user-first".to_string(),
            "priority-user-second".to_string(),
            super::super::agent_mailbox_delivery::mailbox_turn_id("priority-trigger-mail"),
        ]
    );
    assert!(fixture
        .projection_store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("priority-root-thread"),
            ThreadId::new("priority-child-thread"),
        )
        .await
        .expect("read pending child mailbox")
        .is_empty());
}

#[tokio::test]
async fn pending_route_restores_original_queue_state_then_retries_the_same_turn_once() {
    let fixture = PersistentRuntimeFixture::new();
    let setup = fixture.core(Arc::new(RecordingBackend::default()));
    seed_queued_session(
        &setup,
        "retry-session",
        "retry-thread",
        &[
            (
                "retry-first",
                "first queued input",
                route_options("retry-first", false),
            ),
            (
                "retry-second",
                "second queued input",
                route_options("retry-second", false),
            ),
        ],
    )
    .await;
    drop(setup);

    let backend = Arc::new(SelectiveRecoveryBackend::pending_once_for([
        "retry-session",
    ]));
    let restarted = fixture.core(backend.clone());
    restarted
        .ensure_current_session_hydrated("retry-session")
        .await
        .expect("hydrate retry session before route attempt");
    let original_turn_order = {
        let state = restarted
            .state
            .lock()
            .expect("runtime state mutex poisoned");
        state
            .sessions
            .get("retry-session")
            .expect("hydrated retry session")
            .turns
            .iter()
            .map(|turn| (turn.turn_id.clone(), turn.status))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        original_turn_order
            .iter()
            .filter(|(_, status)| *status == AgentTurnStatus::Queued)
            .map(|(turn_id, _)| turn_id.as_str())
            .collect::<Vec<_>>(),
        vec!["retry-first", "retry-second"]
    );
    let pending = restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect_err("first queued route should remain pending");
    assert!(matches!(
        pending,
        RuntimeCoreError::PendingRoute {
            session_id,
            provider,
            model,
            reason_code,
        } if session_id == "retry-session"
            && provider.as_deref() == Some("provider-retry-first")
            && model.as_deref() == Some("model-retry-first")
            && reason_code == "provider_and_model_missing"
    ));
    {
        let state = restarted
            .state
            .lock()
            .expect("runtime state mutex poisoned");
        let stored = state
            .sessions
            .get("retry-session")
            .expect("hydrated retry session");
        assert_eq!(
            stored
                .turns
                .iter()
                .map(|turn| (turn.turn_id.clone(), turn.status))
                .collect::<Vec<_>>(),
            original_turn_order,
            "PendingRoute must restore the first queued turn at its original FIFO index"
        );
        assert_eq!(
            stored
                .turn_inputs
                .get("retry-first")
                .map(|input| super::super::turn_start::user_input_text(input)),
            Some("first queued input".to_string())
        );
        let restored_options = stored
            .turn_runtime_options
            .get("retry-first")
            .expect("restored runtime options");
        assert_eq!(
            restored_options.provider_preference(),
            Some("provider-retry-first")
        );
        assert_eq!(
            restored_options.model_preference(),
            Some("model-retry-first")
        );
        assert_eq!(
            restored_options
                .runtime_metadata()
                .and_then(|metadata| metadata.get("clientUserMessageId"))
                .and_then(serde_json::Value::as_str),
            Some("client-message-retry-first")
        );
        assert_eq!(restored_options.queued_turn_id, None);
    }

    restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect("retry restored queued route");
    restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect("completed queue must not run again");
    let requests = backend.requests();
    assert_eq!(
        requests
            .iter()
            .map(|request| request.turn_id.as_str())
            .collect::<Vec<_>>(),
        vec!["retry-first", "retry-first", "retry-second"]
    );
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.turn_id == "retry-first")
            .count(),
        2,
        "one PendingRoute attempt must be followed by exactly one successful retry"
    );
    assert!(queued_turn_ids(&restarted, "retry-session").is_empty());
}

#[tokio::test]
async fn one_pending_route_does_not_block_another_durable_queued_session() {
    let fixture = PersistentRuntimeFixture::new();
    let setup = fixture.core(Arc::new(RecordingBackend::default()));
    seed_queued_session(
        &setup,
        "a-blocked-session",
        "a-blocked-thread",
        &[(
            "a-blocked-queued",
            "blocked until the next generation",
            route_options("blocked", false),
        )],
    )
    .await;
    seed_queued_session(
        &setup,
        "z-ready-session",
        "z-ready-thread",
        &[(
            "z-ready-queued",
            "ready in this generation",
            route_options("ready", false),
        )],
    )
    .await;
    drop(setup);

    let backend = Arc::new(SelectiveRecoveryBackend::pending_once_for([
        "a-blocked-session",
    ]));
    let restarted = fixture.core(backend.clone());
    let pending = restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect_err("one session should remain pending");
    assert!(matches!(
        pending,
        RuntimeCoreError::PendingRoute { session_id, .. }
            if session_id == "a-blocked-session"
    ));
    assert_eq!(
        backend
            .requests()
            .iter()
            .map(|request| (request.session_id.as_str(), request.turn_id.as_str()))
            .collect::<Vec<_>>(),
        vec![
            ("a-blocked-session", "a-blocked-queued"),
            ("z-ready-session", "z-ready-queued"),
        ],
        "PendingRoute must be aggregated only after later sessions are attempted"
    );
    assert_eq!(
        queued_turn_ids(&restarted, "a-blocked-session"),
        vec!["a-blocked-queued"]
    );
    assert!(queued_turn_ids(&restarted, "z-ready-session").is_empty());
}

#[tokio::test]
async fn recovery_ignores_event_only_queued_work_and_recovers_projection_work() {
    let fixture = PersistentRuntimeFixture::new();
    let setup = fixture.core(Arc::new(RecordingBackend::default()));
    seed_queued_session(
        &setup,
        "a-event-only-failure",
        "a-event-only-thread",
        &[(
            "a-event-only-queued",
            "recover from the event log",
            route_options("event-only", false),
        )],
    )
    .await;
    seed_queued_session(
        &setup,
        "b-projection-success",
        "b-projection-thread",
        &[(
            "b-projection-queued",
            "continue after the failed session",
            route_options("projection", false),
        )],
    )
    .await;
    fixture
        .projection_store
        .clear_session("a-event-only-failure")
        .expect("remove queued projection session");
    assert_eq!(
        fixture
            .projection_store
            .list_queued_session_ids()
            .expect("projection queued sessions"),
        vec!["b-projection-success".to_string()]
    );
    drop(setup);

    let backend = Arc::new(SelectiveRecoveryBackend::fail_once_for([
        "a-event-only-failure",
    ]));
    let restarted = fixture.core(backend.clone());
    restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect("projection-backed queued work recovers");

    let requests = backend.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].session_id, "b-projection-success");
    assert_session_not_hydrated(&restarted, "a-event-only-failure");
    assert!(queued_turn_ids(&restarted, "b-projection-success").is_empty());
}
