use super::*;

struct RecordingChildBackend {
    child_started: tokio::sync::mpsc::UnboundedSender<ExecutionRequest>,
}

#[async_trait::async_trait]
impl ExecutionBackend for RecordingChildBackend {
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

struct RoutePreparingChildBackend {
    inner: RecordingChildBackend,
    service_tiers: Vec<&'static str>,
}

#[async_trait::async_trait]
impl ExecutionBackend for RoutePreparingChildBackend {
    async fn prepare_turn_runtime_options(
        &self,
        request: &ExecutionRequest,
        _first_sampling_turn: bool,
    ) -> Result<Option<app_server_protocol::RuntimeOptions>, RuntimeCoreError> {
        let Some(mut options) = request.runtime_options.clone() else {
            return Ok(None);
        };
        let runtime_request = options.runtime_request_mut();
        let Some(provider) = runtime_request.provider_preference.clone() else {
            return Ok(Some(options));
        };
        let Some(model) = runtime_request.model_preference.clone() else {
            return Ok(Some(options));
        };
        let reasoning_effort = runtime_request.reasoning_effort.clone();
        let service_tier = runtime_request.service_tier.clone();
        let service_tiers = self
            .service_tiers
            .iter()
            .map(|id| json!({ "id": id, "name": id, "description": "" }))
            .collect::<Vec<_>>();
        let metadata = runtime_request.metadata.get_or_insert_with(|| json!({}));
        metadata["agentControlRoute"] = json!({
            "schemaVersion": 2,
            "providerPreference": provider,
            "modelPreference": model,
            "serviceTier": service_tier,
            "providerConfig": {
                "providerId": provider,
                "providerName": provider,
                "modelName": model,
                "reasoningEffort": reasoning_effort,
                "toolshim": false,
                "toolshimModel": null,
                "supportsWebsockets": false
            },
            "routeProtocol": "openai_responses",
            "authKind": "api_key_ref",
            "credentialRef": "credential-1",
            "effectiveGeneration": 1,
            "modelRegistry": {
                "status": "matched",
                "model": {
                    "service_tiers": service_tiers,
                    "default_service_tier": "priority"
                }
            }
        });
        Ok(Some(options))
    }

    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.inner.start_turn(request, sink).await
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.inner.cancel_turn(_request, sink).await
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.inner.respond_action(_request, _sink).await
    }
}

#[tokio::test]
async fn explicit_spawn_model_controls_replace_the_inherited_route() {
    let (child_started_tx, mut child_started_rx) = tokio::sync::mpsc::unbounded_channel();
    let backend = Arc::new(RoutePreparingChildBackend {
        inner: RecordingChildBackend {
            child_started: child_started_tx,
        },
        service_tiers: vec!["priority"],
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
                    runtime_request: Some(app_server_protocol::RuntimeRequest {
                        provider_preference: Some("openai".to_string()),
                        model_preference: Some("parent-model".to_string()),
                        reasoning_effort: Some("medium".to_string()),
                        service_tier: Some("default".to_string()),
                        ..app_server_protocol::RuntimeRequest::default()
                    }),
                    ..app_server_protocol::RuntimeOptions::default()
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

    gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                session_id: session.session_id,
                thread_id: session.thread_id,
                turn_id: turn.turn_id,
                call_id: "spawn-model-override".to_string(),
            },
            command: AgentControlCommand::SpawnAgent {
                task_name: "review".to_string(),
                message: "review the route".to_string(),
                fork_mode: SpawnAgentForkMode::None,
                model_overrides: SpawnAgentModelOverrides {
                    model: Some("child-model".to_string()),
                    reasoning_effort: Some("high".to_string()),
                    service_tier: Some("priority".to_string()),
                },
            },
            cancel_token: None,
        })
        .await
        .expect("spawn child with explicit model controls");

    let child_request = tokio::time::timeout(Duration::from_secs(1), child_started_rx.recv())
        .await
        .expect("child turn should start")
        .expect("child request");
    let runtime_request = child_request
        .runtime_request()
        .expect("child runtime request");
    assert_eq!(
        runtime_request.provider_preference.as_deref(),
        Some("openai")
    );
    assert_eq!(
        runtime_request.model_preference.as_deref(),
        Some("child-model")
    );
    assert_eq!(runtime_request.reasoning_effort.as_deref(), Some("high"));
    assert_eq!(runtime_request.service_tier.as_deref(), Some("priority"));

    let identity = spawned_child_identity(&store, "parent-thread", "review").await;
    let child_session_id = store
        .read_thread(ReadThreadParams {
            thread_id: identity.thread_id,
            include_archived: true,
            turns_view: ThreadTurnsView::NotLoaded,
        })
        .await
        .expect("read child thread")
        .expect("child thread exists")
        .session_id
        .to_string();
    let child = core
        .read_session(AgentSessionReadParams {
            session_id: child_session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("child session")
        .session;
    let metadata = child
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref())
        .expect("child route metadata");
    assert_eq!(metadata["modelName"], "child-model");
    assert_eq!(metadata["serviceTier"], "priority");
    assert_eq!(
        metadata["agentControlRoute"]["modelPreference"],
        "child-model"
    );
    assert_eq!(
        metadata["agentControlRoute"]["modelRegistry"],
        json!({
            "status": "matched",
            "model": {
                "service_tiers": [{
                    "id": "priority",
                    "name": "priority",
                    "description": ""
                }],
                "default_service_tier": "priority"
            }
        })
    );
}

#[tokio::test]
async fn unsupported_spawn_service_tier_has_no_durable_child_side_effects() {
    let (child_started_tx, mut child_started_rx) = tokio::sync::mpsc::unbounded_channel();
    let backend = Arc::new(RoutePreparingChildBackend {
        inner: RecordingChildBackend {
            child_started: child_started_tx,
        },
        service_tiers: vec!["priority"],
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
                    runtime_request: Some(app_server_protocol::RuntimeRequest {
                        provider_preference: Some("openai".to_string()),
                        model_preference: Some("parent-model".to_string()),
                        reasoning_effort: Some("medium".to_string()),
                        ..app_server_protocol::RuntimeRequest::default()
                    }),
                    ..app_server_protocol::RuntimeOptions::default()
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

    let error = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                session_id: session.session_id,
                thread_id: session.thread_id,
                turn_id: turn.turn_id,
                call_id: "spawn-unsupported-tier".to_string(),
            },
            command: AgentControlCommand::SpawnAgent {
                task_name: "review".to_string(),
                message: "review the route".to_string(),
                fork_mode: SpawnAgentForkMode::None,
                model_overrides: SpawnAgentModelOverrides {
                    model: Some("child-model".to_string()),
                    reasoning_effort: Some("high".to_string()),
                    service_tier: Some("flex".to_string()),
                },
            },
            cancel_token: None,
        })
        .await
        .expect_err("unsupported service tier must fail closed");

    assert!(error
        .message()
        .contains("spawn_agent_service_tier_unsupported"));
    assert!(child_started_rx.try_recv().is_err());
    let identities = store
        .list_agent_identities(ThreadId::new("parent-thread"))
        .await
        .expect("list identities after rejected spawn");
    assert_eq!(
        identities
            .iter()
            .map(|identity| identity.agent_path.as_str())
            .collect::<Vec<_>>(),
        vec!["/root"]
    );
    assert!(store
        .list_thread_spawn_children(ThreadId::new("parent-thread"), None)
        .await
        .expect("list children after rejected spawn")
        .is_empty());
    assert!(store
        .list_pending_agent_mailbox_trigger_recipients()
        .await
        .expect("list mailbox recipients after rejected spawn")
        .is_empty());
    let sessions = core
        .list_agent_sessions(app_server_protocol::AgentSessionListParams::default())
        .await
        .expect("list sessions after rejected spawn");
    assert_eq!(sessions.sessions.len(), 1);
    assert_eq!(sessions.sessions[0].session_id, "parent-session");
}

#[tokio::test]
async fn gateway_reads_preflight_route_from_canonical_turn_options() {
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
    let core = RuntimeCore::with_backend(backend).with_projection_store(store);
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
                    event_name: Some("parent-event".to_string()),
                    queued_turn_id: Some("parent-queue".to_string()),
                    runtime_request: Some(app_server_protocol::RuntimeRequest {
                        metadata: Some(json!({ "fixture": "effective-child-route" })),
                        ..app_server_protocol::RuntimeRequest::default()
                    }),
                    expected_output: Some(json!({ "type": "parent-only" })),
                    structured_output: Some(
                        app_server_protocol::StructuredOutputContract::default(),
                    ),
                    output_schema: Some(json!({ "type": "object" })),
                    ..app_server_protocol::RuntimeOptions::default()
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

    gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                session_id: session.session_id,
                thread_id: session.thread_id,
                turn_id: turn.turn_id,
                call_id: "effective-route-call".to_string(),
            },
            command: AgentControlCommand::SpawnAgent {
                task_name: "route_check".to_string(),
                message: "inspect the effective route".to_string(),
                fork_mode: SpawnAgentForkMode::None,
                model_overrides: SpawnAgentModelOverrides::default(),
            },
            cancel_token: None,
        })
        .await
        .expect("spawn child");

    let child_request = tokio::time::timeout(Duration::from_secs(1), child_started_rx.recv())
        .await
        .expect("child turn should start")
        .expect("child request");
    let child_options = child_request
        .runtime_options
        .expect("child runtime options");
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
        Some("resolved-provider")
    );
    assert_eq!(
        child_runtime_request.model_preference.as_deref(),
        Some("resolved-model")
    );
    assert_eq!(
        child_runtime_request.reasoning_effort.as_deref(),
        Some("high")
    );
    assert_eq!(
        child_runtime_request.working_dir.as_deref(),
        Some("/tmp/effective-child-route")
    );

    child_release_tx.send(()).expect("release child");
}

#[tokio::test]
async fn warm_followup_keeps_the_target_effective_route() {
    let (child_started_tx, mut child_started_rx) = tokio::sync::mpsc::unbounded_channel();
    let backend = Arc::new(RecordingChildBackend {
        child_started: child_started_tx,
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
                    runtime_request: Some(app_server_protocol::RuntimeRequest {
                        provider_preference: Some("parent-provider".to_string()),
                        model_preference: Some("parent-model".to_string()),
                        ..app_server_protocol::RuntimeRequest::default()
                    }),
                    ..app_server_protocol::RuntimeOptions::default()
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
    let caller = AgentControlCaller {
        session_id: session.session_id.clone(),
        thread_id: session.thread_id.clone(),
        turn_id: turn.turn_id.clone(),
        call_id: "spawn-route-target".to_string(),
    };
    gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: caller.clone(),
            command: AgentControlCommand::SpawnAgent {
                task_name: "route_target".to_string(),
                message: "start with the parent route".to_string(),
                fork_mode: SpawnAgentForkMode::None,
                model_overrides: SpawnAgentModelOverrides::default(),
            },
            cancel_token: None,
        })
        .await
        .expect("spawn child");

    let initial_child_request =
        tokio::time::timeout(Duration::from_secs(1), child_started_rx.recv())
            .await
            .expect("initial child should start")
            .expect("initial child request");
    assert_eq!(
        initial_child_request.provider_preference(),
        Some("parent-provider")
    );
    let child_identity = spawned_child_identity(&store, "parent-thread", "route_target").await;
    let child_session_id = initial_child_request.session.session_id;
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let child = core
                .read_session(AgentSessionReadParams {
                    session_id: child_session_id.clone(),
                    history_limit: None,
                    history_offset: None,
                    history_before_message_id: None,
                })
                .expect("child session");
            if child
                .turns
                .iter()
                .any(|turn| turn.status == app_server_protocol::AgentTurnStatus::Completed)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("initial child should complete");
    {
        let mut state = core
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let child = state
            .sessions
            .get_mut(&child_session_id)
            .expect("warm child session");
        let latest_turn_id = child
            .turns
            .last()
            .expect("initial child turn")
            .turn_id
            .clone();
        let options = child
            .turn_runtime_options
            .get_mut(&latest_turn_id)
            .expect("initial child effective options");
        options.event_name = Some("target-only-event".to_string());
        options.expected_output = Some(json!({ "type": "target-only" }));
        let runtime_request = options.runtime_request_mut();
        runtime_request.provider_preference = Some("target-provider".to_string());
        runtime_request.model_preference = Some("target-model".to_string());
    }

    gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                call_id: "followup-route-target".to_string(),
                ..caller
            },
            command: AgentControlCommand::FollowupTask {
                target: child_identity.agent_path,
                message: "continue with your own route".to_string(),
            },
            cancel_token: None,
        })
        .await
        .expect("followup child");

    let followup_request = tokio::time::timeout(Duration::from_secs(1), child_started_rx.recv())
        .await
        .expect("followup child should start")
        .expect("followup child request");
    assert_eq!(
        followup_request.provider_preference(),
        Some("target-provider")
    );
    assert_eq!(followup_request.model_preference(), Some("target-model"));
    let followup_options = followup_request
        .runtime_options
        .expect("followup runtime options");
    assert_eq!(followup_options.event_name, None);
    assert_eq!(followup_options.expected_output, None);
}
