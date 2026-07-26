use super::support::*;
use super::*;

use async_trait::async_trait;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{oneshot, Notify};
use tokio::time::timeout;

const SESSION_ID: &str = "route-generation-session";
const THREAD_ID: &str = "route-generation-thread";
const ACTIVE_TURN_ID: &str = "route-generation-active";
const QUEUED_TURN_ID: &str = "route-generation-queued";

struct CommittedRouteSignal {
    generation: Arc<AtomicU64>,
    reads: AtomicUsize,
    read_notify: Notify,
}

impl CommittedRouteSignal {
    fn new(generation: Arc<AtomicU64>) -> Self {
        Self {
            generation,
            reads: AtomicUsize::new(0),
            read_notify: Notify::new(),
        }
    }

    fn commit_credential_generation(&self, generation: u64) {
        self.generation.store(generation, Ordering::Release);
    }

    async fn wait_for_reads(&self, expected: usize) {
        timeout(Duration::from_secs(2), async {
            loop {
                let notified = self.read_notify.notified();
                if self.reads.load(Ordering::Acquire) >= expected {
                    return;
                }
                notified.await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("route generation was not read {expected} times"));
    }
}

impl SessionAppDataSource for CommittedRouteSignal {}
impl WorkspaceAppDataSource for CommittedRouteSignal {}
impl SkillAppDataSource for CommittedRouteSignal {}
impl WorkspaceSkillBindingAppDataSource for CommittedRouteSignal {}
impl GatewayAppDataSource for CommittedRouteSignal {}
impl MediaAppDataSource for CommittedRouteSignal {}
impl VoiceAppDataSource for CommittedRouteSignal {}
impl PluginDataSource for CommittedRouteSignal {}
impl KnowledgeAppDataSource for CommittedRouteSignal {}
impl AutomationOverviewAppDataSource for CommittedRouteSignal {}
impl McpAppDataSource for CommittedRouteSignal {}
impl AutomationManagementAppDataSource for CommittedRouteSignal {}
impl MemoryAppDataSource for CommittedRouteSignal {}
impl DiagnosticsAppDataSource for CommittedRouteSignal {}
impl UsageStatsAppDataSource for CommittedRouteSignal {}
impl ConnectAppDataSource for CommittedRouteSignal {}
impl RightSurfaceAppDataSource for CommittedRouteSignal {}

#[async_trait]
impl ModelProviderAppDataSource for CommittedRouteSignal {
    async fn read_model_route_generation(&self) -> Result<u64, RuntimeCoreError> {
        let generation = self.generation.load(Ordering::Acquire);
        self.reads.fetch_add(1, Ordering::AcqRel);
        self.read_notify.notify_waiters();
        Ok(generation)
    }
}

struct GenerationControlledBackend {
    generation: Arc<AtomicU64>,
    attempts: Mutex<Vec<(String, u64)>>,
    successful_executions: AtomicUsize,
    attempt_notify: Notify,
    success_notify: Notify,
    block_next_generation_one: AtomicBool,
    generation_one_release: Mutex<Option<oneshot::Receiver<()>>>,
}

impl GenerationControlledBackend {
    fn new(generation: Arc<AtomicU64>) -> Self {
        Self {
            generation,
            attempts: Mutex::new(Vec::new()),
            successful_executions: AtomicUsize::new(0),
            attempt_notify: Notify::new(),
            success_notify: Notify::new(),
            block_next_generation_one: AtomicBool::new(false),
            generation_one_release: Mutex::new(None),
        }
    }

    fn attempts(&self) -> Vec<(String, u64)> {
        self.attempts
            .lock()
            .expect("route attempts mutex poisoned")
            .clone()
    }

    fn reset_attempts(&self) {
        self.attempts
            .lock()
            .expect("route attempts mutex poisoned")
            .clear();
    }

    fn block_next_generation_one_attempt(&self) -> oneshot::Sender<()> {
        let (release_tx, release_rx) = oneshot::channel();
        *self
            .generation_one_release
            .lock()
            .expect("generation one release mutex poisoned") = Some(release_rx);
        self.block_next_generation_one
            .store(true, Ordering::Release);
        release_tx
    }

    async fn wait_for_attempts(&self, expected: usize) {
        timeout(Duration::from_secs(2), async {
            loop {
                let notified = self.attempt_notify.notified();
                if self.attempts().len() >= expected {
                    return;
                }
                notified.await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("route backend did not receive {expected} attempts"));
    }

    async fn wait_for_successes(&self, expected: usize) {
        timeout(Duration::from_secs(2), async {
            loop {
                let notified = self.success_notify.notified();
                if self.successful_executions.load(Ordering::Acquire) >= expected {
                    return;
                }
                notified.await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("route backend did not complete {expected} turns"));
    }
}

#[async_trait]
impl ExecutionBackend for GenerationControlledBackend {
    fn requires_provider_selection(&self) -> bool {
        true
    }

    async fn preflight_turn(
        &self,
        request: &ExecutionRequest,
        _first_sampling_turn: bool,
    ) -> Result<(), RuntimeCoreError> {
        let generation = self.generation.load(Ordering::Acquire);
        let turn_id = request.turn.turn_id.clone();
        self.attempts
            .lock()
            .expect("route attempts mutex poisoned")
            .push((turn_id.clone(), generation));
        self.attempt_notify.notify_waiters();

        let release =
            if generation == 1 && self.block_next_generation_one.swap(false, Ordering::AcqRel) {
                self.generation_one_release
                    .lock()
                    .expect("generation one release mutex poisoned")
                    .take()
            } else {
                None
            };
        if let Some(release) = release {
            let _ = release.await;
        }

        if generation < 2 {
            let session_id = request.session.session_id.clone();
            let provider = request.provider_preference().map(str::to_string);
            let model = request.model_preference().map(str::to_string);
            return Err(RuntimeCoreError::PendingRoute {
                session_id,
                provider,
                model,
                reason_code: "credential_generation_pending".to_string(),
            });
        }
        Ok(())
    }

    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(RuntimeEvent::new("turn.completed", json!({})))?;
        self.successful_executions.fetch_add(1, Ordering::AcqRel);
        self.success_notify.notify_waiters();
        Ok(())
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

fn route_options() -> RuntimeOptions {
    RuntimeOptions {
        runtime_request: Some(RuntimeRequest {
            provider_config: Some(RuntimeProviderConfig {
                provider_id: Some("committed-provider".to_string()),
                provider_name: Some("openai".to_string()),
                model_name: Some("committed-model".to_string()),
                ..RuntimeProviderConfig::default()
            }),
            provider_preference: Some("committed-provider".to_string()),
            model_preference: Some("committed-model".to_string()),
            ..RuntimeRequest::default()
        }),
        ..RuntimeOptions::default()
    }
}

async fn seed_durable_queued_turn(core: &RuntimeCore) {
    core.start_session(AgentSessionStartParams {
        session_id: Some(SESSION_ID.to_string()),
        thread_id: Some(THREAD_ID.to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("start route generation session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: SESSION_ID.to_string(),
            turn_id: Some(ACTIVE_TURN_ID.to_string()),
            input: AgentInput {
                text: "active turn before provider commit".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("start active route generation turn");
    let queued = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: SESSION_ID.to_string(),
                turn_id: Some(QUEUED_TURN_ID.to_string()),
                input: AgentInput {
                    text: "resume after credential commit".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: Some(route_options()),
                queue_if_busy: true,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("persist queued route generation turn");
    assert_eq!(queued.response.turn.status, AgentTurnStatus::Queued);
    core.append_external_runtime_events(
        SESSION_ID,
        Some(ACTIVE_TURN_ID),
        vec![RuntimeEvent::new("turn.completed", json!({}))],
    )
    .expect("complete active route generation turn");
}

#[tokio::test]
async fn committed_generation_retries_durable_queued_turn_once_after_pending_route() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots = StorageRoots::initialize(temp.path(), temp.path().join("app-server"))
        .expect("storage roots");
    let event_log_writer =
        Arc::new(EventLogWriter::new(&roots.event_log_root).expect("event log writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection store"));
    let setup = RuntimeCore::with_backend(Arc::new(RecordingBackend::default()))
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());
    seed_durable_queued_turn(&setup).await;
    assert_eq!(
        projection_store
            .list_queued_session_ids()
            .expect("list queued sessions"),
        vec![SESSION_ID.to_string()]
    );
    drop(setup);

    let generation = Arc::new(AtomicU64::new(1));
    let signal = Arc::new(CommittedRouteSignal::new(generation.clone()));
    let backend = Arc::new(GenerationControlledBackend::new(generation));
    let restarted = RuntimeCore::with_backend(backend.clone())
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store.clone())
        .with_app_data_source(signal.clone());

    let pending = restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect_err("queued route must wait for a committed credential generation");
    assert!(matches!(
        pending,
        RuntimeCoreError::PendingRoute {
            session_id,
            provider,
            model,
            reason_code,
        } if session_id == SESSION_ID
            && provider.as_deref() == Some("committed-provider")
            && model.as_deref() == Some("committed-model")
            && reason_code == "credential_generation_pending"
    ));
    assert_eq!(
        projection_store
            .list_queued_session_ids()
            .expect("queued route remains durable after PendingRoute"),
        vec![SESSION_ID.to_string()]
    );
    assert_eq!(
        restarted
            .session_snapshot(SESSION_ID)
            .expect("hydrated route generation session")
            .1
            .iter()
            .filter(|turn| turn.status == AgentTurnStatus::Queued)
            .map(|turn| turn.turn_id.as_str())
            .collect::<Vec<_>>(),
        vec![QUEUED_TURN_ID]
    );

    backend.reset_attempts();
    let release_generation_one = backend.block_next_generation_one_attempt();
    restarted.schedule_pending_route_recovery(RuntimeHostContext::default());
    backend.wait_for_attempts(1).await;

    restarted.schedule_pending_route_recovery(RuntimeHostContext::default());
    signal.wait_for_reads(2).await;
    signal.commit_credential_generation(2);
    restarted.schedule_pending_route_recovery(RuntimeHostContext::default());
    signal.wait_for_reads(3).await;
    release_generation_one
        .send(())
        .expect("release generation one recovery");
    backend.wait_for_successes(1).await;

    assert_eq!(
        backend.attempts(),
        vec![
            (QUEUED_TURN_ID.to_string(), 1),
            (QUEUED_TURN_ID.to_string(), 2),
        ],
        "the same committed generation must be coalesced before the next generation retries"
    );
    assert_eq!(
        backend.successful_executions.load(Ordering::Acquire),
        1,
        "the durable queued turn must execute exactly once"
    );
    assert!(projection_store
        .list_queued_session_ids()
        .expect("list queued sessions after recovery")
        .is_empty());
    let (_, turns) = restarted
        .session_snapshot(SESSION_ID)
        .expect("recovered route generation session");
    assert_eq!(
        turns
            .iter()
            .filter(|turn| turn.turn_id == QUEUED_TURN_ID)
            .count(),
        1,
        "generation retry must reuse the durable queued turn id"
    );
    assert!(turns.iter().any(|turn| {
        turn.turn_id == QUEUED_TURN_ID && turn.status == AgentTurnStatus::Completed
    }));
}
