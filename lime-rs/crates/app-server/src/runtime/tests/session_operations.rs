use super::support::{
    canonical_tool_completed_event, canonical_tool_started_event, wait_for_runtime_event,
    TestSessionDataSource,
};
use super::*;
use app_server_protocol::protocol::v2::{ThreadCompactStartParams, ThreadCompactStartResponse};
use lime_core::models::model_registry::{
    EnhancedModelMetadata, ModelCapabilityProvenance, ModelTaskFamily, ModelVisibility,
};
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

#[derive(Default)]
struct ModelSelectionCaptureBackend {
    selection: Mutex<Option<(String, String)>>,
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

#[async_trait]
impl ExecutionBackend for ModelSelectionCaptureBackend {
    fn requires_provider_selection(&self) -> bool {
        true
    }

    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let runtime_request = request.runtime_request().ok_or_else(|| {
            RuntimeCoreError::Backend("captured turn requires runtime request".to_string())
        })?;
        let provider = runtime_request
            .provider_preference
            .clone()
            .ok_or_else(|| RuntimeCoreError::Backend("captured provider is missing".to_string()))?;
        let model = runtime_request
            .model_preference
            .clone()
            .ok_or_else(|| RuntimeCoreError::Backend("captured model is missing".to_string()))?;
        *self.selection.lock().expect("selection mutex poisoned") = Some((provider, model));
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
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

#[tokio::test]
async fn catalog_refresh_preserves_current_selectable_model() {
    let (_temp, core) = model_selection_core(vec![model_catalog(
        "provider-a",
        vec![chat_model("provider-a", "model-a")],
    )]);

    let changed = core
        .reconcile_thread_model_selection("thread-model-refresh")
        .await
        .expect("reconcile current model");

    assert!(changed.is_none());
    let (_, settings, _) = core
        .loaded_thread_settings("thread-model-refresh")
        .expect("read current settings")
        .expect("persisted settings");
    assert_eq!(settings.model_provider, "provider-a");
    assert_eq!(settings.model, "model-a");
    assert_eq!(settings.effort.as_deref(), Some("high"));
}

#[tokio::test]
async fn catalog_refresh_reselects_same_provider_before_next_provider() {
    let mut hidden = chat_model("provider-a", "hidden-model");
    hidden.visibility = ModelVisibility::Hide;
    let mut inferred = chat_model("provider-a", "inferred-model");
    inferred.capability_provenance = ModelCapabilityProvenance::InferredHint;
    let mut image = chat_model("provider-a", "image-model");
    image.task_families = vec![ModelTaskFamily::ImageGeneration];
    let (_temp, core) = model_selection_core(vec![
        model_catalog("provider-b", vec![chat_model("provider-b", "model-c")]),
        model_catalog(
            "provider-a",
            vec![hidden, inferred, image, chat_model("provider-a", "model-b")],
        ),
    ]);

    let changed = core
        .reconcile_thread_model_selection("thread-model-refresh")
        .await
        .expect("reselect model")
        .expect("selection changed");

    assert_eq!(changed.model_provider, "provider-a");
    assert_eq!(changed.model, "model-b");
    assert_eq!(changed.effort.as_deref(), Some("medium"));
    assert_eq!(
        changed.collaboration_mode.settings.model, "model-b",
        "collaboration settings must follow the durable model selection"
    );
    assert_eq!(
        changed
            .collaboration_mode
            .settings
            .reasoning_effort
            .as_deref(),
        Some("medium")
    );
    let (_, persisted, _) = core
        .loaded_thread_settings("thread-model-refresh")
        .expect("read reselected settings")
        .expect("persisted settings");
    assert_eq!(persisted, changed);
}

#[tokio::test]
async fn catalog_refresh_without_executable_model_fails_closed() {
    let mut inferred = chat_model("provider-a", "inferred-model");
    inferred.capability_provenance = ModelCapabilityProvenance::InferredHint;
    let mut image = chat_model("provider-b", "image-model");
    image.task_families = vec![ModelTaskFamily::ImageGeneration];
    let (_temp, core) = model_selection_core(vec![
        model_catalog("provider-a", vec![inferred]),
        model_catalog("provider-b", vec![image]),
    ]);

    let error = core
        .reconcile_thread_model_selection("thread-model-refresh")
        .await
        .expect_err("catalog without executable chat model must fail");

    assert!(matches!(
        error,
        RuntimeCoreError::RouteRejected {
            reason_code,
            ..
        } if reason_code == "model_catalog_has_no_executable_selection"
    ));
    let (_, settings, _) = core
        .loaded_thread_settings("thread-model-refresh")
        .expect("read unchanged settings")
        .expect("persisted settings");
    assert_eq!(settings.model_provider, "provider-a");
    assert_eq!(settings.model, "model-a");
}

#[tokio::test]
async fn catalog_refresh_skips_backend_without_provider_selection() {
    let (_temp, core) = model_selection_core_with_metadata_and_backend(
        vec![model_catalog(
            "provider-b",
            vec![chat_model("provider-b", "model-b")],
        )],
        default_model_selection_metadata(),
        Arc::new(MockBackend),
    );

    let changed = core
        .reconcile_thread_model_selection("thread-model-refresh")
        .await
        .expect("backend without provider selection must bypass catalog reconciliation");

    assert!(changed.is_none());
    let (_, settings, _) = core
        .loaded_thread_settings("thread-model-refresh")
        .expect("read unchanged settings")
        .expect("persisted settings");
    assert_eq!(settings.model_provider, "provider-a");
    assert_eq!(settings.model, "model-a");
}

#[tokio::test]
async fn catalog_refresh_preserves_durable_direct_provider_route() {
    let (_temp, core) = model_selection_core_with_metadata(
        vec![model_catalog(
            "provider-b",
            vec![chat_model("provider-b", "model-b")],
        )],
        json!({
            "providerSelector": "provider-direct",
            "providerName": "provider-direct",
            "modelName": "model-direct",
            "collaborationMode": {
                "mode": "default",
                "settings": { "model": "model-direct" }
            },
            "agentControlRoute": {
                "schemaVersion": 2,
                "routeSource": "direct_provider_config",
                "providerPreference": "provider-direct",
                "modelPreference": "model-direct",
                "providerConfig": {
                    "providerId": "provider-direct",
                    "providerName": "provider-direct",
                    "modelName": "model-direct"
                },
                "routeProtocol": "openai_responses",
                "authKind": "direct_api_key",
                "effectiveGeneration": 1
            }
        }),
    );

    let changed = core
        .reconcile_thread_model_selection("thread-model-refresh")
        .await
        .expect("preserve direct route");

    assert!(changed.is_none());
    let (_, settings, direct) = core
        .loaded_thread_settings("thread-model-refresh")
        .expect("read direct settings")
        .expect("persisted direct settings");
    assert!(direct);
    assert_eq!(settings.model_provider, "provider-direct");
    assert_eq!(settings.model, "model-direct");
}

#[tokio::test]
async fn runtime_turn_entry_uses_reconciled_model_selection() {
    let backend = Arc::new(ModelSelectionCaptureBackend::default());
    let (_temp, core) = model_selection_core_with_metadata_and_backend(
        vec![model_catalog(
            "provider-a",
            vec![chat_model("provider-a", "model-b")],
        )],
        default_model_selection_metadata(),
        backend.clone(),
    );

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "session-model-refresh".to_string(),
            turn_id: None,
            input: AgentInput {
                text: "continue with refreshed model".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("start reconciled runtime turn");

    assert_eq!(
        backend
            .selection
            .lock()
            .expect("selection mutex poisoned")
            .clone(),
        Some(("provider-a".to_string(), "model-b".to_string()))
    );
}

fn model_selection_core(catalogs: Vec<ProviderModelCatalog>) -> (tempfile::TempDir, RuntimeCore) {
    model_selection_core_with_metadata(catalogs, default_model_selection_metadata())
}

fn default_model_selection_metadata() -> Value {
    json!({
        "providerSelector": "provider-a",
        "providerName": "provider-a",
        "modelName": "model-a",
        "reasoningEffort": "high",
        "collaborationMode": {
            "mode": "default",
            "settings": {
                "model": "model-a",
                "reasoning_effort": "high"
            }
        }
    })
}

fn model_selection_core_with_metadata(
    catalogs: Vec<ProviderModelCatalog>,
    metadata: Value,
) -> (tempfile::TempDir, RuntimeCore) {
    model_selection_core_with_metadata_and_backend(
        catalogs,
        metadata,
        Arc::new(ModelSelectionCaptureBackend::default()),
    )
}

fn model_selection_core_with_metadata_and_backend(
    catalogs: Vec<ProviderModelCatalog>,
    metadata: Value,
    backend: Arc<dyn ExecutionBackend>,
) -> (tempfile::TempDir, RuntimeCore) {
    let temp = tempfile::tempdir().expect("model selection temp dir");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("model selection projection store"),
    );
    let data_source = Arc::new(TestSessionDataSource::new().with_model_catalogs(catalogs));
    let core = RuntimeCore::with_backend(backend)
        .with_projection_store(projection_store)
        .with_app_data_source(data_source);
    core.start_session(AgentSessionStartParams {
        session_id: Some("session-model-refresh".to_string()),
        thread_id: Some("thread-model-refresh".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: None,
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.thread".to_string(),
            id: "thread-model-refresh".to_string(),
            title: None,
            uri: None,
            metadata: Some(metadata),
        }),
        locale: None,
    })
    .expect("start model selection session");
    (temp, core)
}

fn model_catalog(provider: &str, models: Vec<EnhancedModelMetadata>) -> ProviderModelCatalog {
    ProviderModelCatalog {
        provider_id: provider.to_string(),
        sort_order: 0,
        models,
    }
}

fn chat_model(provider: &str, model: &str) -> EnhancedModelMetadata {
    let mut metadata = EnhancedModelMetadata::new(
        model.to_string(),
        model.to_string(),
        provider.to_string(),
        provider.to_string(),
    );
    metadata.capability_provenance = ModelCapabilityProvenance::ProviderExplicit;
    metadata.task_families = vec![ModelTaskFamily::Chat];
    metadata.capabilities.reasoning_effort = Some(
        lime_core::models::model_registry::ModelReasoningEffortSupport {
            supported: true,
            levels: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
            options: Vec::new(),
            default: Some("medium".to_string()),
            source: None,
        },
    );
    metadata
}
