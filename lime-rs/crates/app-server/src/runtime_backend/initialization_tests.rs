use super::*;
use crate::execution_process::ExecutionProcessServer;
use crate::runtime::RuntimeHostContext;
use crate::{ActionRespondRequest, CancelExecutionRequest, ExecutionBackend};
use agent_protocol::{ItemStatus, ThreadItem, ThreadItemPayload};
use agent_runtime::session_loop::{
    RuntimeSessionClosureTask, RuntimeSessionRegistry, RuntimeSessionTaskFailure,
};
use app_server_protocol::{
    AgentInput, AgentSession, AgentSessionActionRespondParams, AgentSessionActionScope,
    AgentSessionActionType, AgentSessionApprovalDecision, AgentSessionStartParams,
    AgentSessionStatus, AgentSessionTurnStartParams, AgentTurn, AgentTurnStatus, RuntimeOptions,
};
use async_trait::async_trait;
use lime_core::database::schema::create_tables;
use rusqlite::Connection;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};

#[derive(Default)]
struct TestRuntimeEventSink {
    events: Vec<RuntimeEvent>,
}

impl RuntimeEventSink for TestRuntimeEventSink {
    fn emit(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        self.events.push(event);
        Ok(())
    }
}

struct BridgeRuntimeEventSink {
    events: Arc<Mutex<Vec<RuntimeEvent>>>,
    approval_tx: mpsc::UnboundedSender<String>,
}

struct RestartPendingActionSeedBackend;

#[async_trait]
impl ExecutionBackend for RestartPendingActionSeedBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(RuntimeEvent::new(
            "action.required",
            json!({
                "requestId": "ask-restart-1",
                "actionType": "ask_user",
                "prompt": "Choose a restart option",
                "requestedSchema": { "type": "string" },
                "availableDecisions": ["allow_once", "decline"],
                "createdAtMs": 1_783_900_000_000_u64,
                "deadlineAtMs": 1_999_999_999_999_u64,
            }),
        ))
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

impl RuntimeEventSink for BridgeRuntimeEventSink {
    fn emit(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        if event.event_type == "action.required" {
            if let Some(request_id) = event
                .payload
                .get("request_id")
                .or_else(|| event.payload.get("requestId"))
                .and_then(Value::as_str)
            {
                let _ = self.approval_tx.send(request_id.to_string());
            }
        }
        self.events
            .lock()
            .expect("record runtime event")
            .push(event);
        Ok(())
    }
}

struct LocalOpenAiFixture {
    base_url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    server_task: JoinHandle<()>,
}

impl LocalOpenAiFixture {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind local OpenAI fixture");
        let addr = listener.local_addr().expect("fixture local addr");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let server_requests = requests.clone();
        let server_task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let requests = server_requests.clone();
                tokio::spawn(async move {
                    let _ = handle_openai_fixture_connection(stream, requests).await;
                });
            }
        });

        Self {
            base_url: format!("http://{addr}/v1"),
            requests,
            server_task,
        }
    }
}

impl Drop for LocalOpenAiFixture {
    fn drop(&mut self) {
        self.server_task.abort();
    }
}

fn test_db() -> DbConnection {
    let conn = Connection::open_in_memory().expect("open in-memory db");
    create_tables(&conn).expect("create schema");
    Arc::new(Mutex::new(conn))
}

#[tokio::test]
async fn main_turn_initializes_agent_before_live_execution_hook() {
    let db = test_db();
    let db = provider_config::initialize_runtime_database(Some(&db)).expect("runtime database");
    let backend = RuntimeBackend::with_db_and_execution_process_server(
        db.clone(),
        ExecutionProcessServer::default(),
    );

    assert!(!backend.agent_state.is_initialized().await);

    backend
        .ensure_agent_initialized(&db)
        .await
        .expect("main turn should initialize agent before hook installation");
    backend
        .install_live_execution_process_hook_if_available()
        .await
        .expect("live execution hook should install after agent initialization");

    assert!(backend.agent_state.is_initialized().await);
}

#[tokio::test]
async fn mcp_resource_read_rehydrates_session_runtime_without_starting_turn() {
    let backend = RuntimeBackend::with_db(test_db());
    ExecutionBackend::set_app_data_source(&backend, Arc::new(crate::NoopAppDataSource))
        .expect("set app data source");

    assert!(!backend.agent_state.is_initialized().await);

    let error = ExecutionBackend::read_mcp_runtime_resource(
        &backend,
        "session-history",
        "thread-history",
        "missing-server",
        "ui://missing-server/app.html",
        None,
    )
    .await
    .expect_err("empty fixture runtime has no MCP server");

    assert!(backend.agent_state.is_initialized().await);
    assert!(!error.to_string().contains("MCP runtime is not initialized"));
}

#[tokio::test]
async fn unexecutable_route_returns_typed_rejection_before_provider_call() {
    let backend = RuntimeBackend::with_db(test_db());
    let request = crate::runtime_backend::tests::request_for_test(
        "do not reach the provider",
        Some(app_server_protocol::RuntimeRequest {
            provider_config: Some(app_server_protocol::RuntimeProviderConfig {
                provider_id: Some("fixture-openai".to_string()),
                provider_name: Some("openai".to_string()),
                model_name: Some("fixture-model".to_string()),
                api_key: Some("fixture-key".to_string()),
                base_url: Some("http://127.0.0.1:1".to_string()),
                ..app_server_protocol::RuntimeProviderConfig::default()
            }),
            ..app_server_protocol::RuntimeRequest::default()
        }),
        None,
    );
    let mut sink = TestRuntimeEventSink::default();

    let error = ExecutionBackend::start_turn(&backend, request, &mut sink)
        .await
        .expect_err("missing capability snapshot must reject the current route");

    assert!(matches!(
        &error,
        RuntimeCoreError::RouteRejected {
            session_id,
            provider: Some(provider),
            model: Some(model),
            category: app_server_protocol::RouteFailureCategory::CapabilityGap,
            reason_code,
        } if session_id == "session-1"
            && provider == "fixture-openai"
            && model == "fixture-model"
            && reason_code == "capability_snapshot_missing"
    ));
    let rpc_error = error.into_jsonrpc_error();
    assert_eq!(
        rpc_error
            .data
            .as_ref()
            .and_then(|data| data.get("type"))
            .and_then(serde_json::Value::as_str),
        Some("RouteRejected")
    );
    assert_eq!(
        rpc_error
            .data
            .as_ref()
            .and_then(|data| data.get("retryable"))
            .and_then(serde_json::Value::as_bool),
        Some(false)
    );
    assert!(sink
        .events
        .iter()
        .any(|event| event.event_type == "routing.not_possible"));
    assert!(!backend.agent_state.is_initialized().await);
}

#[test]
fn internal_route_failure_does_not_enter_pending_route_recovery() {
    let selection = super::request_context::RuntimeModelSelection {
        provider: "fixture-provider".to_string(),
        model: "fixture-model".to_string(),
        source: "test",
        reasoning_effort: None,
    };
    let failure = app_server_protocol::RouteFailure {
        category: app_server_protocol::RouteFailureCategory::InternalError,
        reason_code: "route_owner_failed".to_string(),
        ..app_server_protocol::RouteFailure::default()
    };

    let error = super::runtime_error_from_route_failure("session-1", &selection, &failure);

    assert!(
        matches!(error, RuntimeCoreError::Backend(message) if message.contains("route_owner_failed"))
    );
}

#[test]
fn fixed_capability_gap_returns_typed_route_rejection() {
    let selection = super::request_context::RuntimeModelSelection {
        provider: "fixture-provider".to_string(),
        model: "fixture-model".to_string(),
        source: "test",
        reasoning_effort: None,
    };
    let failure = app_server_protocol::RouteFailure {
        category: app_server_protocol::RouteFailureCategory::CapabilityGap,
        reason_code: "capability_gap".to_string(),
        capability_gap: Some("task_family:vision_understanding".to_string()),
        ..app_server_protocol::RouteFailure::default()
    };

    let error = super::runtime_error_from_route_failure("session-1", &selection, &failure);

    assert!(matches!(
        error,
        RuntimeCoreError::RouteRejected {
            category: app_server_protocol::RouteFailureCategory::CapabilityGap,
            reason_code,
            ..
        } if reason_code == "capability_gap"
    ));
}

#[test]
fn missing_model_catalog_enters_pending_route_recovery() {
    let selection = super::request_context::RuntimeModelSelection {
        provider: "fixture-provider".to_string(),
        model: "fixture-model".to_string(),
        source: "test",
        reasoning_effort: None,
    };
    let failure = app_server_protocol::RouteFailure {
        category: app_server_protocol::RouteFailureCategory::ModelUnavailable,
        reason_code: "model_registry_metadata_missing".to_string(),
        provider_id: Some("fixture-provider".to_string()),
        model_id: Some("fixture-model".to_string()),
        ..app_server_protocol::RouteFailure::default()
    };

    let error = super::runtime_error_from_route_failure("session-1", &selection, &failure);

    assert!(matches!(
        error,
        RuntimeCoreError::PendingRoute {
            provider: Some(provider),
            model: Some(model),
            reason_code,
            ..
        } if provider == "fixture-provider"
            && model == "fixture-model"
            && reason_code == "model_registry_metadata_missing"
    ));
}

#[tokio::test]
async fn respond_action_initializes_agent_before_runtime_resume() {
    let db = test_db();
    let db = provider_config::initialize_runtime_database(Some(&db)).expect("runtime database");
    let backend = RuntimeBackend::with_db(db.clone());
    let mut sink = TestRuntimeEventSink::default();

    assert!(!backend.agent_state.is_initialized().await);

    let error = ExecutionBackend::respond_action(
        &backend,
        ActionRespondRequest {
            host: RuntimeHostContext::default(),
            session: AgentSession {
                session_id: "session-respond-init".to_string(),
                thread_id: "thread-respond-init".to_string(),
                app_id: "agent".to_string(),
                workspace_id: None,
                business_object_ref: None,
                status: AgentSessionStatus::Running,
                created_at: chrono::Utc::now().to_rfc3339(),
                updated_at: chrono::Utc::now().to_rfc3339(),
            },
            turn: None,
            request_id: "ask-respond-init".to_string(),
            action_type: AgentSessionActionType::AskUser,
            decision: None,
            confirmed: false,
            response: None,
            user_data: None,
            metadata: None,
            event_name: None,
            action_scope: None,
            pending_action_descriptor: None,
        },
        &mut sink,
    )
    .await
    .expect_err("unknown action must fail closed after agent initialization");

    assert!(backend.agent_state.is_initialized().await);
    assert!(matches!(
        error,
        RuntimeCoreError::ActionResponse { ref code, ref request_id }
            if code == "action_scope_missing" && request_id == "ask-respond-init"
    ));
    assert!(sink.events.is_empty());
}

#[tokio::test]
async fn respond_action_restores_descriptor_and_fails_closed_without_continuation() {
    let db = test_db();
    let db = provider_config::initialize_runtime_database(Some(&db)).expect("runtime database");
    let backend = RuntimeBackend::with_db(db);
    let session = AgentSession {
        session_id: "session-restored-action".to_string(),
        thread_id: "thread-restored-action".to_string(),
        app_id: "agent".to_string(),
        workspace_id: None,
        business_object_ref: None,
        status: AgentSessionStatus::WaitingAction,
        created_at: "2026-07-12T15:00:00Z".to_string(),
        updated_at: "2026-07-12T15:00:00Z".to_string(),
    };
    let descriptor = agent_runtime::action_required::PendingActionDescriptor {
        request_id: "approval-restored-1".to_string(),
        action_type: "tool_confirmation".to_string(),
        tool_id: Some("tool-restored-1".to_string()),
        message: Some("Allow restored action?".to_string()),
        requested_schema: None,
        available_decisions: vec!["allow_once".to_string(), "decline".to_string()],
        scope: lime_agent::AgentActionRequiredScope::from_parts(
            Some("session-restored-action".to_string()),
            Some("thread-restored-action".to_string()),
            Some("turn-restored-action".to_string()),
        ),
        created_at_ms: Some(1_783_900_000_000_u64),
        deadline_at_ms: Some(1_999_999_999_999_u64),
        status: agent_runtime::action_required::PendingActionStatus::Pending,
    };

    let action_request = ActionRespondRequest {
        host: RuntimeHostContext::default(),
        session,
        turn: Some(AgentTurn {
            turn_id: "turn-restored-action".to_string(),
            session_id: "session-restored-action".to_string(),
            thread_id: "thread-restored-action".to_string(),
            status: AgentTurnStatus::WaitingAction,
            started_at: Some("2026-07-12T15:00:00Z".to_string()),
            completed_at: None,
        }),
        request_id: "approval-restored-1".to_string(),
        action_type: AgentSessionActionType::ToolConfirmation,
        decision: Some(AgentSessionApprovalDecision::AllowOnce),
        confirmed: true,
        response: None,
        user_data: None,
        metadata: None,
        event_name: None,
        action_scope: Some(app_server_protocol::AgentSessionActionScope {
            session_id: Some("session-restored-action".to_string()),
            thread_id: Some("thread-restored-action".to_string()),
            turn_id: Some("turn-restored-action".to_string()),
        }),
        pending_action_descriptor: Some(descriptor),
    };
    let mut sink = TestRuntimeEventSink::default();
    let error = ExecutionBackend::respond_action(&backend, action_request.clone(), &mut sink)
        .await
        .expect_err("restored descriptor must not fake a live continuation");

    assert!(
        matches!(
            error,
            RuntimeCoreError::ActionResponse { ref code, ref request_id }
                if code == "action_not_resumable" && request_id == "approval-restored-1"
        ),
        "unexpected restored action error: {error:?}"
    );
    assert!(sink.events.is_empty());

    let repeated = ExecutionBackend::respond_action(&backend, action_request, &mut sink)
        .await
        .expect_err("repeated restored response must preserve terminal reason");
    assert!(matches!(
        repeated,
        RuntimeCoreError::ActionResponse { ref code, ref request_id }
            if code == "action_not_resumable" && request_id == "approval-restored-1"
    ));
    assert!(sink.events.is_empty());
}

#[tokio::test]
async fn denied_ask_user_cancels_restored_action_without_resolved_event() {
    let session = AgentSession {
        session_id: "session-restored-cancel".to_string(),
        thread_id: "thread-restored-cancel".to_string(),
        app_id: "agent".to_string(),
        workspace_id: None,
        business_object_ref: None,
        status: AgentSessionStatus::WaitingAction,
        created_at: "2026-07-12T15:00:00Z".to_string(),
        updated_at: "2026-07-12T15:00:00Z".to_string(),
    };
    let turn = AgentTurn {
        turn_id: "turn-restored-cancel".to_string(),
        session_id: session.session_id.clone(),
        thread_id: session.thread_id.clone(),
        status: AgentTurnStatus::WaitingAction,
        started_at: Some("2026-07-12T15:00:00Z".to_string()),
        completed_at: None,
    };
    let descriptor = agent_runtime::action_required::PendingActionDescriptor {
        request_id: "ask-restored-cancel".to_string(),
        action_type: "ask_user".to_string(),
        tool_id: None,
        message: Some("Continue?".to_string()),
        requested_schema: None,
        available_decisions: Vec::new(),
        scope: lime_agent::AgentActionRequiredScope::from_parts(
            Some(session.session_id.clone()),
            Some(session.thread_id.clone()),
            Some(turn.turn_id.clone()),
        ),
        created_at_ms: Some(1_783_900_000_000_u64),
        deadline_at_ms: Some(1_999_999_999_999_u64),
        status: agent_runtime::action_required::PendingActionStatus::Pending,
    };
    let backend = RuntimeBackend::with_db(test_db());
    let mut sink = TestRuntimeEventSink::default();

    ExecutionBackend::respond_action(
        &backend,
        ActionRespondRequest {
            host: RuntimeHostContext::default(),
            session,
            turn: Some(turn),
            request_id: "ask-restored-cancel".to_string(),
            action_type: AgentSessionActionType::AskUser,
            decision: None,
            confirmed: false,
            response: None,
            user_data: None,
            metadata: None,
            event_name: None,
            action_scope: Some(AgentSessionActionScope {
                session_id: Some("session-restored-cancel".to_string()),
                thread_id: Some("thread-restored-cancel".to_string()),
                turn_id: Some("turn-restored-cancel".to_string()),
            }),
            pending_action_descriptor: Some(descriptor),
        },
        &mut sink,
    )
    .await
    .expect("denied restored ask-user action");

    assert_eq!(sink.events.len(), 1);
    assert_eq!(sink.events[0].event_type, "action.canceled");
    assert!(sink
        .events
        .iter()
        .all(|event| event.event_type != "action.resolved"));
}

#[tokio::test]
async fn runtime_core_hydrates_persisted_descriptor_into_runtime_backend() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots = crate::StorageRoots::initialize(temp.path(), temp.path().join("app-server"))
        .expect("roots");
    let event_log_writer =
        Arc::new(crate::EventLogWriter::new(&roots.event_log_root).expect("event log"));
    let projection_store = Arc::new(
        crate::ProjectionStore::initialize(&roots.projection_db_path).expect("projection store"),
    );
    let session_id = "session-restart-production";
    let thread_id = "thread-restart-production";
    let turn_id = "turn-restart-production";

    let seed_core = crate::RuntimeCore::with_backend(Arc::new(RestartPendingActionSeedBackend))
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());
    seed_core
        .start_session(AgentSessionStartParams {
            session_id: Some(session_id.to_string()),
            thread_id: Some(thread_id.to_string()),
            app_id: "agent-chat".to_string(),
            workspace_id: Some("workspace-restart".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("seed session");
    seed_core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: session_id.to_string(),
                turn_id: Some(turn_id.to_string()),
                input: AgentInput {
                    text: "wait for restart response".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: true,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("seed pending action");
    drop(seed_core);
    drop(event_log_writer);
    drop(projection_store);

    let event_log_writer =
        Arc::new(crate::EventLogWriter::new(&roots.event_log_root).expect("reopen event log"));
    let projection_store = Arc::new(
        crate::ProjectionStore::initialize(&roots.projection_db_path)
            .expect("reopen projection store"),
    );

    let restarted_core =
        crate::RuntimeCore::with_backend(Arc::new(RuntimeBackend::with_db(test_db())))
            .with_event_log_writer(event_log_writer.clone())
            .with_projection_store(projection_store);
    let type_mismatch = restarted_core
        .respond_action(
            AgentSessionActionRespondParams {
                session_id: session_id.to_string(),
                request_id: "ask-restart-1".to_string(),
                action_type: AgentSessionActionType::ToolConfirmation,
                decision: Some(AgentSessionApprovalDecision::AllowOnce),
                confirmed: None,
                response: None,
                user_data: Some(json!({ "confirmed": true })),
                metadata: None,
                event_name: None,
                action_scope: Some(AgentSessionActionScope {
                    session_id: Some(session_id.to_string()),
                    thread_id: Some(thread_id.to_string()),
                    turn_id: Some(turn_id.to_string()),
                }),
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect_err("caller action type must match persisted action");
    assert!(matches!(
        type_mismatch,
        RuntimeCoreError::ActionResponse { ref code, ref request_id }
            if code == "action_type_mismatch" && request_id == "ask-restart-1"
    ));
    let error = restarted_core
        .respond_action(
            AgentSessionActionRespondParams {
                session_id: session_id.to_string(),
                request_id: "ask-restart-1".to_string(),
                action_type: AgentSessionActionType::AskUser,
                decision: None,
                confirmed: Some(true),
                response: Some("continue".to_string()),
                user_data: Some(json!({ "answer": "continue" })),
                metadata: None,
                event_name: None,
                action_scope: Some(AgentSessionActionScope {
                    session_id: Some(session_id.to_string()),
                    thread_id: Some(thread_id.to_string()),
                    turn_id: Some(turn_id.to_string()),
                }),
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect_err("restored continuation must fail closed");

    assert!(
        matches!(
            error,
            RuntimeCoreError::ActionResponse { ref code, ref request_id }
                if code == "action_not_resumable" && request_id == "ask-restart-1"
        ),
        "unexpected runtime restart error: {error:?}"
    );
    let persisted_events = event_log_writer
        .read_session_events(session_id)
        .expect("persisted events");
    assert!(persisted_events
        .iter()
        .all(|record| record.event.event_type != "action.resolved"));
}

#[test]
fn action_response_error_preserves_stable_jsonrpc_data() {
    let error = RuntimeCoreError::ActionResponse {
        code: "action_not_resumable".to_string(),
        request_id: "ask-restart-1".to_string(),
    }
    .into_jsonrpc_error();

    assert_eq!(error.code, app_server_protocol::error_codes::RUNTIME_ERROR);
    assert_eq!(
        error.message,
        "action response failed: action_not_resumable"
    );
    assert_eq!(
        error.data,
        Some(json!({
            "code": "action_not_resumable",
            "requestId": "ask-restart-1",
        }))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn respond_action_tool_confirmation_resumes_pending_agent_tool_future() {
    let workspace = TempDir::new().expect("workspace");
    let provider = LocalOpenAiFixture::start().await;
    let db = test_db();
    let db = provider_config::initialize_runtime_database(Some(&db)).expect("runtime database");
    let backend = Arc::new(RuntimeBackend::with_db_and_execution_process_server(
        db.clone(),
        ExecutionProcessServer::default(),
    ));
    backend
        .ensure_agent_initialized(&db)
        .await
        .expect("initialize runtime before confirmation scenario");
    let run_id = uuid::Uuid::new_v4().simple().to_string();
    let request = execution_request_for_tool_confirmation_bridge_test(
        &provider.base_url,
        workspace.path().to_string_lossy().as_ref(),
        &run_id,
    );
    let action_session = request.session.clone();
    let action_turn = request.turn.clone();
    let (approval_tx, mut approval_rx) = mpsc::unbounded_channel::<String>();
    let stream_events = Arc::new(Mutex::new(Vec::new()));
    let stream_backend = backend.clone();
    let stream_events_for_sink = stream_events.clone();
    let registry = RuntimeSessionRegistry::default();
    let session_loop = registry
        .get_or_create(&action_session.session_id, &action_session.thread_id)
        .await
        .expect("bind confirmation session actor");
    let turn_id = action_turn.turn_id.clone();
    let task = RuntimeSessionClosureTask::new(
        turn_id,
        Vec::new(),
        move |context, _input, cancellation_token| {
            let stream_backend = stream_backend.clone();
            let request = request.clone();
            let stream_events = stream_events_for_sink.clone();
            let approval_tx = approval_tx.clone();
            Box::pin(async move {
                let mut sink = BridgeRuntimeEventSink {
                    events: stream_events,
                    approval_tx,
                };
                ExecutionBackend::start_turn_with_provider_history_and_session_input(
                    &*stream_backend,
                    request,
                    crate::runtime::provider_history::ProviderTurnHistory::default(),
                    Some(context.input_handle()),
                    Some(cancellation_token),
                    &mut sink,
                )
                .await
                .map_err(|error| RuntimeSessionTaskFailure {
                    message: error.to_string(),
                    reason_code: None,
                })
            })
        },
    );
    let submission = session_loop
        .submit(Arc::new(task), false)
        .await
        .expect("submit runtime turn task");
    let mut completion = submission.completion;

    let request_id = tokio::select! {
        request_id = approval_rx.recv() => match request_id {
            Some(request_id) => request_id,
            None => {
                panic!(
                    "tool confirmation request channel closed before request id; events={:?}; provider_requests={:?}",
                    stream_events.lock().expect("read stream events"),
                    provider.requests.lock().expect("read provider requests")
                );
            }
        },
        result = &mut completion => {
            panic!(
                "runtime turn finished before tool confirmation; result={:?}; events={:?}",
                result,
                stream_events.lock().expect("read stream events")
            );
        }
        _ = tokio::time::sleep(Duration::from_secs(10)) => {
            let _ = session_loop.interrupt().await;
            panic!(
                "timed out waiting for tool confirmation; events={:?}; provider_requests={:?}",
                stream_events.lock().expect("read stream events"),
                provider.requests.lock().expect("read provider requests")
            );
        }
    };
    assert!(!request_id.trim().is_empty());

    let mut action_sink = TestRuntimeEventSink::default();
    ExecutionBackend::respond_action(
        &*backend,
        ActionRespondRequest {
            host: RuntimeHostContext::default(),
            session: action_session.clone(),
            turn: Some(action_turn.clone()),
            request_id: request_id.clone(),
            action_type: AgentSessionActionType::ToolConfirmation,
            decision: Some(AgentSessionApprovalDecision::AllowOnce),
            confirmed: true,
            response: None,
            user_data: None,
            metadata: None,
            event_name: None,
            action_scope: Some(app_server_protocol::AgentSessionActionScope {
                session_id: Some(action_session.session_id.clone()),
                thread_id: Some(action_session.thread_id.clone()),
                turn_id: Some(action_turn.turn_id.clone()),
            }),
            pending_action_descriptor: None,
        },
        &mut action_sink,
    )
    .await
    .expect("respond_action should release pending tool confirmation");

    session_loop
        .approve(
            Some(&action_turn.turn_id),
            &request_id,
            json!({ "confirmed": true }),
        )
        .await
        .expect("typed approval should resume runtime task");

    timeout(Duration::from_secs(20), &mut completion)
        .await
        .expect("runtime turn should finish after confirmation")
        .expect("runtime task should complete")
        .expect("runtime turn should succeed");
    registry
        .shutdown(&action_session.session_id)
        .await
        .expect("shutdown runtime session");

    assert_eq!(action_sink.events.len(), 1);
    assert_eq!(action_sink.events[0].event_type, "action.resolved");
    assert_eq!(
        action_sink.events[0].payload["requestId"].as_str(),
        Some(request_id.as_str())
    );

    let events = stream_events.lock().expect("read stream events");
    assert!(events.iter().any(|event| {
        if event.event_type != "item.completed" {
            return false;
        }
        let Some(item) = event
            .payload
            .get("item")
            .cloned()
            .and_then(|item| serde_json::from_value::<ThreadItem>(item).ok())
        else {
            return false;
        };
        let ThreadItemPayload::Command { output, .. } = item.payload else {
            return false;
        };
        item.status == ItemStatus::Completed
            && item.item_id == agent_protocol::ItemId::new("req-runtime-confirm")
            && output.is_some_and(|output| output.contains("runtime-confirmed"))
    }));
    assert!(events.iter().any(|event| {
        event.event_type == "message.delta"
            && event.payload["text"]
                .as_str()
                .is_some_and(|text| text.contains("provider observed resumed tool"))
    }));
    assert!(events
        .iter()
        .any(|event| event.event_type == "turn.completed"));
    let provider_steps = events
        .iter()
        .filter(|event| event.event_type == "provider.step")
        .collect::<Vec<_>>();
    assert_eq!(provider_steps.len(), 2);
    assert_eq!(provider_steps[0].payload["attempt"], 1);
    assert_eq!(provider_steps[0].payload["tool_call_count"], 1);
    assert_eq!(provider_steps[1].payload["attempt"], 2);
    assert_eq!(provider_steps[1].payload["finish_reason"], "stop");
    let turn_completed = events
        .iter()
        .find(|event| event.event_type == "turn.completed")
        .expect("turn completed event");
    assert_eq!(turn_completed.payload["usage"]["input_tokens"], 20);
    assert_eq!(turn_completed.payload["usage"]["output_tokens"], 8);
    assert!(provider
        .requests
        .lock()
        .expect("read provider requests")
        .iter()
        .any(request_contains_tool_response));
}

fn execution_request_for_tool_confirmation_bridge_test(
    base_url: &str,
    workspace: &str,
    run_id: &str,
) -> ExecutionRequest {
    let session_id = format!("session-bridge-tool-confirm-{run_id}");
    let thread_id = format!("thread-bridge-tool-confirm-{run_id}");
    let turn_id = format!("turn-bridge-tool-confirm-{run_id}");
    ExecutionRequest {
        host: RuntimeHostContext::default(),
        session: AgentSession {
            session_id: session_id.clone(),
            thread_id: thread_id.clone(),
            app_id: "agent".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: AgentSessionStatus::Running,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        },
        turn: AgentTurn {
            turn_id,
            session_id,
            thread_id,
            status: AgentTurnStatus::Accepted,
            started_at: None,
            completed_at: None,
        },
        forked_from_thread_id: None,
        input: agent_runtime::reply_input::RuntimeReplyInput::text(
            "run the runtime confirmation command",
        ),
        runtime_options: Some(RuntimeOptions {
            stream: true,
            runtime_request: Some(app_server_protocol::RuntimeRequest {
                approval_policy: Some("on-request".to_string()),
                sandbox_policy: Some("workspace-write".to_string()),
                metadata: Some(json!({
                    "harness": {
                        "projectRoot": workspace,
                        "cwd": workspace
                    }
                })),
                provider_config: Some(app_server_protocol::RuntimeProviderConfig {
                    provider_id: Some("fixture-openai".to_string()),
                    provider_name: Some("openai".to_string()),
                    model_name: Some("fixture-model".to_string()),
                    api_key: Some("fixture-key".to_string()),
                    base_url: Some(base_url.to_string()),
                    model_capabilities: Some(json!({
                        "capabilities": {
                            "tools": true,
                            "streaming": true,
                            "jsonMode": true,
                            "functionCalling": true
                        },
                        "taskFamilies": ["chat"],
                        "inputModalities": ["text"],
                        "outputModalities": ["text"],
                        "runtimeFeatures": ["streaming", "tool_calling"]
                    })),
                    ..app_server_protocol::RuntimeProviderConfig::default()
                }),
                ..app_server_protocol::RuntimeRequest::default()
            }),
            ..RuntimeOptions::default()
        }),
        event_name: None,
        expected_output: None,
        structured_output: None,
        output_schema: None,
        queued_turn_id: None,
        queue_if_busy: false,
        skip_pre_submit_resume: false,
        agent_control_gateway: None,
        rollout_budget_reminder_source: None,
    }
}

async fn handle_openai_fixture_connection(
    mut stream: TcpStream,
    requests: Arc<Mutex<Vec<Value>>>,
) -> std::io::Result<()> {
    let body = read_http_json_body(&mut stream).await?;
    requests
        .lock()
        .expect("record provider request")
        .push(body.clone());
    let response_body = if body.get("stream").and_then(Value::as_bool) == Some(true) {
        streaming_response_for_request(&body)
    } else {
        json_response_for_request(&body)
    };
    let content_type = if body.get("stream").and_then(Value::as_bool) == Some(true) {
        "text/event-stream"
    } else {
        "application/json"
    };
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{response_body}",
        response_body.len()
    );
    stream.write_all(response.as_bytes()).await
}

async fn read_http_json_body(stream: &mut TcpStream) -> std::io::Result<Value> {
    let mut buffer = Vec::new();
    let header_end = loop {
        let mut chunk = [0_u8; 1024];
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Ok(json!({}));
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(header_end) = find_header_end(&buffer) {
            break header_end;
        }
    };
    let headers = String::from_utf8_lossy(&buffer[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    let body_start = header_end + 4;
    while buffer.len().saturating_sub(body_start) < content_length {
        let mut chunk = vec![0_u8; content_length - (buffer.len() - body_start)];
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    serde_json::from_slice(&buffer[body_start..body_start + content_length])
        .map_err(std::io::Error::other)
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn streaming_response_for_request(body: &Value) -> String {
    if request_contains_tool_response(body) {
        return sse(&[
            json!({
                "id": "chatcmpl-final",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "fixture-model",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "provider observed resumed tool"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14
                }
            })
            .to_string(),
            "[DONE]".to_string(),
        ]);
    }

    sse(&[
        json!({
            "id": "chatcmpl-tool",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "fixture-model",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": "req-runtime-confirm",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": ""
                        }
                    }]
                },
                "finish_reason": null
            }]
        })
        .to_string(),
        json!({
            "id": "chatcmpl-tool",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "fixture-model",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": "{\"cmd\":\"printf runtime-confirmed\"}"
                        }
                    }]
                },
                "finish_reason": null
            }]
        })
        .to_string(),
        json!({
            "id": "chatcmpl-tool",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "fixture-model",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls"
            }],
            "usage": null
        })
        .to_string(),
        json!({
            "id": "chatcmpl-tool",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "fixture-model",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14
            }
        })
        .to_string(),
        "[DONE]".to_string(),
    ])
}

fn json_response_for_request(_body: &Value) -> String {
    json!({
        "id": "chatcmpl-title",
        "object": "chat.completion",
        "created": 0,
        "model": "fixture-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "approval resume"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2
        }
    })
    .to_string()
}

fn sse(lines: &[String]) -> String {
    lines
        .iter()
        .map(|line| format!("data: {line}\n\n"))
        .collect::<String>()
}

fn request_contains_tool_response(body: &Value) -> bool {
    body.get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            messages.iter().any(|message| {
                message.get("role").and_then(Value::as_str) == Some("tool")
                    || message
                        .get("tool_call_id")
                        .and_then(Value::as_str)
                        .is_some_and(|id| id == "req-runtime-confirm")
            })
        })
}
