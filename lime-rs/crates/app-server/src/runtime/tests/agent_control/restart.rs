use super::*;
use app_server_protocol::AgentSessionReadResponse;

fn persistent_core(
    temp: &tempfile::TempDir,
) -> (Arc<EventLogWriter>, Arc<ProjectionStore>, RuntimeCore) {
    let roots = StorageRoots::initialize(temp.path(), temp.path().join("app-server"))
        .expect("storage roots");
    let event_log_writer =
        Arc::new(EventLogWriter::new(&roots.event_log_root).expect("event log writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection store"));
    let core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());
    (event_log_writer, projection_store, core)
}

struct ProviderSelectionFailureBackend;

#[async_trait::async_trait]
impl ExecutionBackend for ProviderSelectionFailureBackend {
    fn requires_provider_selection(&self) -> bool {
        true
    }

    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Err(RuntimeCoreError::PendingRoute {
            session_id: _request.session.session_id,
            provider: None,
            model: None,
            reason_code: "provider_and_model_missing".to_string(),
        })
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

struct ProviderSelectionRecoveryBackend;

#[async_trait::async_trait]
impl ExecutionBackend for ProviderSelectionRecoveryBackend {
    fn requires_provider_selection(&self) -> bool {
        true
    }

    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
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

#[test]
fn recovery_route_fills_missing_selection_from_session_defaults() {
    let session = app_server_protocol::AgentSession {
        session_id: "recovery-session".to_string(),
        thread_id: "recovery-thread".to_string(),
        app_id: "agent-chat".to_string(),
        workspace_id: None,
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.session".to_string(),
            id: "recovery-session".to_string(),
            title: None,
            uri: None,
            metadata: Some(serde_json::json!({
                "providerSelector": "durable-provider",
                "providerName": "durable-provider",
                "modelName": "durable-model"
            })),
        }),
        status: app_server_protocol::AgentSessionStatus::Idle,
        created_at: "2026-07-18T00:00:00Z".to_string(),
        updated_at: "2026-07-18T00:00:00Z".to_string(),
    };

    let explicit = app_server_protocol::RuntimeOptions {
        runtime_request: Some(app_server_protocol::RuntimeRequest {
            provider_preference: Some("explicit-provider".to_string()),
            ..app_server_protocol::RuntimeRequest::default()
        }),
        ..app_server_protocol::RuntimeOptions::default()
    };
    let merged =
        super::super::super::agent_control::runtime_options_with_agent_control_session_defaults(
            Some(explicit),
            &session,
        )
        .expect("route options");
    let request = merged.runtime_request.as_ref().expect("runtime request");
    assert_eq!(
        request.provider_preference.as_deref(),
        Some("explicit-provider")
    );
    assert_eq!(request.model_preference.as_deref(), Some("durable-model"));
    assert!(super::super::super::agent_control::has_agent_control_runtime_route(Some(&merged)));
}

fn assert_session_unloaded(core: &RuntimeCore, session_id: &str) {
    assert!(matches!(
        core.read_session(AgentSessionReadParams {
            session_id: session_id.to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        }),
        Err(RuntimeCoreError::SessionNotFound(missing)) if missing == session_id
    ));
}

async fn wait_for_session_with_turn(
    core: &RuntimeCore,
    session_id: &str,
    scenario: &str,
) -> AgentSessionReadResponse {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            match core.read_session(AgentSessionReadParams {
                session_id: session_id.to_string(),
                history_limit: None,
                history_offset: None,
                history_before_message_id: None,
            }) {
                Ok(response) if !response.turns.is_empty() => return response,
                Ok(_) | Err(RuntimeCoreError::SessionNotFound(_)) => tokio::task::yield_now().await,
                Err(error) => panic!("{scenario}: {error}"),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{scenario}: session {session_id} did not expose a turn"))
}

async fn session_id_for_thread(store: &ProjectionStore, thread_id: ThreadId) -> String {
    store
        .read_thread(ReadThreadParams {
            thread_id,
            include_archived: true,
            turns_view: ThreadTurnsView::NotLoaded,
        })
        .await
        .expect("read child thread")
        .expect("child thread exists")
        .session_id
        .to_string()
}

async fn agent_identity_for_path(
    store: &ProjectionStore,
    root_thread_id: &str,
    agent_path: &str,
) -> thread_store::AgentIdentity {
    store
        .list_agent_identities(ThreadId::new(root_thread_id))
        .await
        .expect("list durable identities")
        .into_iter()
        .find(|identity| identity.agent_path == agent_path)
        .expect("agent identity")
}

#[tokio::test]
async fn runtime_gateway_does_not_reconcile_an_inflight_pending_spawn() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (_event_log_writer, store, core) = persistent_core(&temp);
    let root = core
        .start_session(start_params("live-root-session", "live-root-thread"))
        .expect("root")
        .session;
    let root_turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: root.session_id.clone(),
                turn_id: Some("live-root-turn".to_string()),
                input: AgentInput {
                    text: "inspect live tree".to_string(),
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
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new("live-root-thread"),
            thread_id: ThreadId::new("live-root-thread"),
            agent_path: "/root".to_string(),
            nickname: None,
            role: None,
            last_task_message: None,
        })
        .await
        .expect("root identity");
    store
        .create_pending_thread_spawn_edge(
            ThreadId::new("live-root-thread"),
            ThreadId::new("inflight-child-thread"),
            "inflight-child-session".to_string(),
        )
        .await
        .expect("inflight reservation");

    let gateway =
        core.agent_control_gateway_for_turn(&root, &root_turn, RuntimeHostContext::default());
    gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                session_id: root.session_id,
                thread_id: root.thread_id,
                turn_id: root_turn.turn_id,
                call_id: "list-during-inflight-spawn".to_string(),
            },
            command: AgentControlCommand::ListAgents { path_prefix: None },
            cancel_token: None,
        })
        .await
        .expect("list agents while another spawn is pending");

    assert_eq!(
        store
            .read_thread_spawn_parent(ThreadId::new("inflight-child-thread"))
            .await
            .expect("read inflight reservation"),
        Some(thread_store::ThreadSpawnParent {
            parent_thread_id: ThreadId::new("live-root-thread"),
            status: ThreadSpawnEdgeStatus::Pending,
        })
    );
}

#[tokio::test]
async fn restart_hydrates_open_none_fork_child_from_session_created_without_items() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (event_log_writer, store, core) = persistent_core(&temp);
    let parent = core
        .start_session(start_params(
            "empty-child-parent-session",
            "empty-child-parent-thread",
        ))
        .expect("parent")
        .session;
    let child = core
        .stage_agent_control_spawn(AgentControlSpawnRequest {
            parent_session_id: parent.session_id,
            child_session_id: Some("empty-child-session".to_string()),
            child_thread_id: Some("empty-child-thread".to_string()),
            fork_mode: SpawnAgentForkMode::None,
        })
        .await
        .expect("stage empty child")
        .session;
    core.commit_agent_control_spawn(&child)
        .await
        .expect("commit empty child");

    let child_events = event_log_writer
        .read_session_events(&child.session_id)
        .expect("read child EventLog");
    assert_eq!(child_events.len(), 1);
    assert_eq!(child_events[0].event.event_type, "session.created");
    assert!(child_events[0].event.turn_id.is_none());
    drop(core);

    let restarted = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(store.clone());
    assert_session_unloaded(&restarted, &child.session_id);
    restarted
        .ensure_current_session_hydrated(&child.session_id)
        .await
        .expect("hydrate empty child from session.created");

    let restored = restarted
        .read_session(AgentSessionReadParams {
            session_id: child.session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read restored empty child");
    assert_eq!(
        restored.session.status,
        app_server_protocol::AgentSessionStatus::Idle
    );
    assert!(restored.turns.is_empty());

    let canonical = store
        .read_thread(ReadThreadParams {
            thread_id: ThreadId::new("empty-child-thread"),
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        })
        .await
        .expect("read canonical empty child")
        .expect("canonical empty child");
    assert_eq!(canonical.status, agent_protocol::ThreadStatus::Idle);
    assert!(canonical.turns.is_empty());
    assert_eq!(
        canonical.agent_state.map(|state| state.status),
        Some(agent_protocol::CollabAgentStatus::PendingInit)
    );
}

#[tokio::test]
async fn restart_preserves_agent_control_provider_defaults() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (event_log_writer, store, core) = persistent_core(&temp);
    let parent = core
        .start_session(start_params(
            "provider-parent-session",
            "provider-parent-thread",
        ))
        .expect("parent")
        .session;
    let parent_turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: parent.session_id.clone(),
                turn_id: Some("provider-parent-turn".to_string()),
                input: AgentInput {
                    text: "delegate with a durable route".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: Some(app_server_protocol::RuntimeOptions {
                    runtime_request: Some(app_server_protocol::RuntimeRequest {
                        provider_preference: Some("fixture-provider".to_string()),
                        model_preference: Some("fixture-model".to_string()),
                        service_tier: Some("priority".to_string()),
                        provider_config: Some(app_server_protocol::RuntimeProviderConfig {
                            provider_id: Some("fixture-provider-id".to_string()),
                            provider_name: Some("fixture-provider".to_string()),
                            model_name: Some("fixture-model".to_string()),
                            api_key: Some("fixture-secret-must-not-persist".to_string()),
                            base_url: Some("https://provider.example/v1".to_string()),
                            tool_call_strategy: Some(
                                app_server_protocol::RuntimeToolCallStrategy::ToolShim,
                            ),
                            toolshim_model: Some("fixture-toolshim".to_string()),
                            model_capabilities: Some(serde_json::json!({
                                "streaming": true,
                                "vision": true
                            })),
                            supports_websockets: Some(true),
                        }),
                        metadata: Some(serde_json::json!({
                            "agentControlRoute": {
                                "schemaVersion": 2,
                                "providerPreference": "fixture-provider",
                                "modelPreference": "fixture-model",
                                "serviceTier": "priority",
                                "providerConfig": {
                                    "providerId": "fixture-provider-id",
                                    "providerName": "fixture-provider",
                                    "modelName": "fixture-model",
                                    "toolshimModel": "fixture-toolshim",
                                    "supportsWebsockets": true
                                },
                                "routeProtocol": "openai_responses",
                                "authKind": "api_key_ref",
                                "credentialRef": "credential-ref-1",
                                "effectiveGeneration": 7
                            }
                        })),
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
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new("provider-parent-thread"),
            thread_id: ThreadId::new("provider-parent-thread"),
            agent_path: "/root".to_string(),
            nickname: None,
            role: None,
            last_task_message: None,
        })
        .await
        .expect("root identity");

    let gateway =
        core.agent_control_gateway_for_turn(&parent, &parent_turn, RuntimeHostContext::default());
    gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                session_id: parent.session_id.clone(),
                thread_id: parent.thread_id.clone(),
                turn_id: parent_turn.turn_id.clone(),
                call_id: "provider-defaults-spawn".to_string(),
            },
            command: AgentControlCommand::SpawnAgent {
                task_name: "provider_child".to_string(),
                message: "continue after restart".to_string(),
                fork_mode: SpawnAgentForkMode::None,
                model_overrides: SpawnAgentModelOverrides::default(),
            },
            cancel_token: None,
        })
        .await
        .expect("spawn child");

    let identity =
        agent_identity_for_path(&store, "provider-parent-thread", "/root/provider_child").await;
    let child_session_id = session_id_for_thread(&store, identity.thread_id).await;
    let child = core
        .read_session(AgentSessionReadParams {
            session_id: child_session_id.clone(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("child before restart")
        .session;
    let metadata = child
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref())
        .expect("child provider defaults");
    assert_eq!(metadata["providerSelector"], "fixture-provider");
    assert_eq!(metadata["providerName"], "fixture-provider");
    assert_eq!(metadata["modelName"], "fixture-model");
    assert_eq!(metadata["serviceTier"], "priority");
    let route_snapshot = metadata["agentControlRoute"]
        .as_object()
        .expect("route snapshot");
    let provider_config = route_snapshot["providerConfig"]
        .as_object()
        .expect("provider config snapshot");
    assert_eq!(provider_config["providerId"], "fixture-provider-id");
    assert_eq!(provider_config["providerName"], "fixture-provider");
    assert_eq!(provider_config["modelName"], "fixture-model");
    assert!(provider_config.get("baseUrl").is_none());
    assert!(provider_config.get("apiKeyPresent").is_none());
    assert!(provider_config.get("apiKey").is_none());
    assert_eq!(route_snapshot["credentialRef"], "credential-ref-1");
    assert_eq!(route_snapshot["routeProtocol"], "openai_responses");
    assert_eq!(route_snapshot["serviceTier"], "priority");
    assert_eq!(route_snapshot["effectiveGeneration"], 7);
    store
        .append_agent_mailbox_message(thread_store::AppendAgentMailboxMessageParams {
            message: thread_store::AgentMailboxMessage {
                message_id: "provider-defaults-restart-message".to_string(),
                root_thread_id: ThreadId::new("provider-parent-thread"),
                sender_thread_id: ThreadId::new("provider-parent-thread"),
                recipient_thread_id: ThreadId::new(child.thread_id.clone()),
                content: "resume with the persisted route".to_string(),
                kind: thread_store::AgentMailboxMessageKind::Message,
                source_turn_id: None,
                result_status: None,
                delivery_mode: AgentMailboxDeliveryMode::TriggerTurn,
                delivery_status: thread_store::AgentMailboxDeliveryStatus::Pending,
                created_at_ms: 2,
                delivered_at_ms: None,
            },
        })
        .await
        .expect("pending restart mailbox");
    drop(gateway);
    drop(core);

    let restarted = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(store);
    restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect("recover child with persisted route");
    restarted
        .ensure_current_session_hydrated(&child_session_id)
        .await
        .expect("hydrate child after restart");
    let restored = restarted
        .read_session(AgentSessionReadParams {
            session_id: child_session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("child after restart")
        .session;
    let metadata = restored
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref())
        .expect("restored provider defaults");
    assert_eq!(metadata["providerSelector"], "fixture-provider");
    assert_eq!(metadata["modelName"], "fixture-model");
    let restored_options =
        super::super::super::agent_control::runtime_options_with_agent_control_session_defaults(
            None, &restored,
        )
        .expect("restore route options");
    let restored_request = restored_options
        .runtime_request
        .as_ref()
        .expect("restored runtime request");
    assert_eq!(restored_request.service_tier.as_deref(), Some("priority"));
    let restored_provider_config = restored_request
        .provider_config
        .as_ref()
        .expect("restored provider config");
    assert_eq!(
        restored_provider_config.provider_id.as_deref(),
        Some("fixture-provider-id")
    );
    assert!(restored_provider_config.base_url.is_none());
    assert_eq!(
        restored_provider_config.toolshim_model.as_deref(),
        Some("fixture-toolshim")
    );
    assert!(restored_provider_config.api_key.is_none());
    let restored_metadata = restored_request
        .metadata
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .expect("restored route metadata");
    let restored_route = restored_metadata["agentControlRoute"]
        .as_object()
        .expect("restored durable route");
    assert_eq!(restored_route["credentialRef"], "credential-ref-1");
    assert_eq!(restored_route["routeProtocol"], "openai_responses");
    assert_eq!(restored_route["serviceTier"], "priority");
    assert_eq!(restored_route["effectiveGeneration"], 7);
}

#[tokio::test]
async fn restart_defers_agent_control_recovery_without_provider_defaults() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (event_log_writer, store, core) = persistent_core(&temp);
    let parent = core
        .start_session(start_params(
            "missing-provider-parent-session",
            "missing-provider-parent-thread",
        ))
        .expect("parent")
        .session;
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new("missing-provider-parent-thread"),
            thread_id: ThreadId::new("missing-provider-parent-thread"),
            agent_path: "/root".to_string(),
            nickname: None,
            role: None,
            last_task_message: None,
        })
        .await
        .expect("root identity");
    let child = core
        .stage_agent_control_spawn(AgentControlSpawnRequest {
            parent_session_id: parent.session_id,
            child_session_id: Some("missing-provider-child-session".to_string()),
            child_thread_id: Some("missing-provider-child-thread".to_string()),
            fork_mode: SpawnAgentForkMode::None,
        })
        .await
        .expect("stage child")
        .session;
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new("missing-provider-parent-thread"),
            thread_id: ThreadId::new("missing-provider-child-thread"),
            agent_path: "/root/missing_provider".to_string(),
            nickname: None,
            role: None,
            last_task_message: Some("retry when a provider is configured".to_string()),
        })
        .await
        .expect("child identity");
    store
        .append_agent_mailbox_message(thread_store::AppendAgentMailboxMessageParams {
            message: thread_store::AgentMailboxMessage {
                message_id: "missing-provider-message".to_string(),
                root_thread_id: ThreadId::new("missing-provider-parent-thread"),
                sender_thread_id: ThreadId::new("missing-provider-parent-thread"),
                recipient_thread_id: ThreadId::new("missing-provider-child-thread"),
                content: "wait for provider selection".to_string(),
                kind: thread_store::AgentMailboxMessageKind::Message,
                source_turn_id: None,
                result_status: None,
                delivery_mode: AgentMailboxDeliveryMode::TriggerTurn,
                delivery_status: thread_store::AgentMailboxDeliveryStatus::Pending,
                created_at_ms: 1,
                delivered_at_ms: None,
            },
        })
        .await
        .expect("pending mailbox");
    core.commit_agent_control_spawn(&child)
        .await
        .expect("commit child");
    drop(core);

    let restarted = RuntimeCore::with_backend(Arc::new(ProviderSelectionFailureBackend))
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(store.clone());
    let pending = restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect_err("missing provider/model route must remain pending");
    assert!(matches!(
        &pending,
        RuntimeCoreError::PendingRoute {
            session_id,
            reason_code,
            ..
        } if session_id == "missing-provider-child-session"
            && reason_code == "provider_and_model_missing"
    ));
    assert!(pending.is_provider_selection_required());
    tokio::time::timeout(
        Duration::from_secs(1),
        restarted.schedule_pending_agent_mailbox_triggers(
            "missing-provider-child-session".to_string(),
            RuntimeHostContext::default(),
            None,
        ),
    )
    .await
    .expect("deferred scheduling should not block admission");
    drop(restarted);

    let ready = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(store.clone());
    ready
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect("recovery retries once a backend route is available");
    assert_eq!(
        store
            .list_pending_agent_mailbox_messages(
                ThreadId::new("missing-provider-parent-thread"),
                ThreadId::new("missing-provider-child-thread"),
            )
            .await
            .expect("read deferred mailbox")
            .len(),
        0
    );
}

#[tokio::test]
async fn restart_continues_recovery_after_one_recipient_has_pending_route() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (event_log_writer, store, core) = persistent_core(&temp);
    let parent = core
        .start_session(start_params(
            "multi-recipient-parent-session",
            "multi-recipient-parent-thread",
        ))
        .expect("parent")
        .session;
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new("multi-recipient-parent-thread"),
            thread_id: ThreadId::new("multi-recipient-parent-thread"),
            agent_path: "/root".to_string(),
            nickname: None,
            role: None,
            last_task_message: None,
        })
        .await
        .expect("root identity");

    let missing = core
        .stage_agent_control_spawn(AgentControlSpawnRequest {
            parent_session_id: parent.session_id.clone(),
            child_session_id: Some("multi-recipient-missing-session".to_string()),
            child_thread_id: Some("a-missing-child-thread".to_string()),
            fork_mode: SpawnAgentForkMode::None,
        })
        .await
        .expect("stage missing-route child")
        .session;
    let ready = core
        .stage_agent_control_spawn_with_runtime_options(
            AgentControlSpawnRequest {
                parent_session_id: parent.session_id,
                child_session_id: Some("multi-recipient-ready-session".to_string()),
                child_thread_id: Some("z-ready-child-thread".to_string()),
                fork_mode: SpawnAgentForkMode::None,
            },
            Some(app_server_protocol::RuntimeOptions {
                runtime_request: Some(app_server_protocol::RuntimeRequest {
                    provider_preference: Some("fixture-provider".to_string()),
                    model_preference: Some("fixture-model".to_string()),
                    ..app_server_protocol::RuntimeRequest::default()
                }),
                ..app_server_protocol::RuntimeOptions::default()
            }),
        )
        .await
        .expect("stage routed child")
        .session;

    for (child, path) in [(&missing, "/root/missing"), (&ready, "/root/ready")] {
        store
            .upsert_agent_identity(thread_store::AgentIdentity {
                root_thread_id: ThreadId::new("multi-recipient-parent-thread"),
                thread_id: ThreadId::new(child.thread_id.clone()),
                agent_path: path.to_string(),
                nickname: None,
                role: None,
                last_task_message: None,
            })
            .await
            .expect("child identity");
    }
    for (child, message_id, content, created_at_ms) in [
        (
            &missing,
            "multi-recipient-missing-message",
            "wait for provider selection",
            1,
        ),
        (
            &ready,
            "multi-recipient-ready-message",
            "resume with the persisted route",
            2,
        ),
    ] {
        store
            .append_agent_mailbox_message(thread_store::AppendAgentMailboxMessageParams {
                message: thread_store::AgentMailboxMessage {
                    message_id: message_id.to_string(),
                    root_thread_id: ThreadId::new("multi-recipient-parent-thread"),
                    sender_thread_id: ThreadId::new("multi-recipient-parent-thread"),
                    recipient_thread_id: ThreadId::new(child.thread_id.clone()),
                    content: content.to_string(),
                    kind: thread_store::AgentMailboxMessageKind::Message,
                    source_turn_id: None,
                    result_status: None,
                    delivery_mode: AgentMailboxDeliveryMode::TriggerTurn,
                    delivery_status: thread_store::AgentMailboxDeliveryStatus::Pending,
                    created_at_ms,
                    delivered_at_ms: None,
                },
            })
            .await
            .expect("pending mailbox");
    }
    core.commit_agent_control_spawn(&missing)
        .await
        .expect("commit missing-route child");
    core.commit_agent_control_spawn(&ready)
        .await
        .expect("commit routed child");
    drop(core);

    let restarted = RuntimeCore::with_backend(Arc::new(ProviderSelectionRecoveryBackend))
        .with_event_log_writer(event_log_writer)
        .with_projection_store(store.clone());
    let pending = restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect_err("one missing route must remain pending after other recipients recover");
    assert!(matches!(
        pending,
        RuntimeCoreError::PendingRoute {
            session_id,
            reason_code,
            ..
        } if session_id == "multi-recipient-missing-session"
            && reason_code == "provider_and_model_missing"
    ));

    assert_eq!(
        store
            .list_pending_agent_mailbox_messages(
                ThreadId::new("multi-recipient-parent-thread"),
                ThreadId::new("a-missing-child-thread"),
            )
            .await
            .expect("read missing-route mailbox")
            .len(),
        1
    );
    assert!(store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("multi-recipient-parent-thread"),
            ThreadId::new("z-ready-child-thread"),
        )
        .await
        .expect("read routed mailbox")
        .is_empty());
    let ready_thread = store
        .read_thread(ReadThreadParams {
            thread_id: ThreadId::new("z-ready-child-thread"),
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        })
        .await
        .expect("read routed child")
        .expect("routed child");
    assert!(ready_thread.turns.iter().any(|turn| {
        turn.turn_id.as_str()
            == super::super::super::agent_mailbox_delivery::mailbox_turn_id(
                "multi-recipient-ready-message",
            )
    }));
}

#[tokio::test]
async fn restart_removes_every_pending_spawn_prefix_and_allows_stable_retry() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (event_log_writer, store, core) = persistent_core(&temp);
    let parent = core
        .start_session(start_params(
            "pending-parent-session",
            "pending-parent-thread",
        ))
        .expect("parent")
        .session;
    append_completed_parent_turn(&core, &parent, "pending-parent-turn", "delegate", "ready").await;
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new("pending-parent-thread"),
            thread_id: ThreadId::new("pending-parent-thread"),
            agent_path: "/root".to_string(),
            nickname: None,
            role: None,
            last_task_message: None,
        })
        .await
        .expect("root identity");

    store
        .create_pending_thread_spawn_edge(
            ThreadId::new("pending-parent-thread"),
            ThreadId::new("intent-only-thread"),
            "intent-only-session".to_string(),
        )
        .await
        .expect("intent-only prefix");
    core.stage_agent_control_spawn(AgentControlSpawnRequest {
        parent_session_id: parent.session_id.clone(),
        child_session_id: Some("materialized-session".to_string()),
        child_thread_id: Some("materialized-thread".to_string()),
        fork_mode: SpawnAgentForkMode::FullHistory,
    })
    .await
    .expect("materialized prefix");
    let mailbox_child = core
        .stage_agent_control_spawn(AgentControlSpawnRequest {
            parent_session_id: parent.session_id.clone(),
            child_session_id: Some("mailbox-session".to_string()),
            child_thread_id: Some("mailbox-thread".to_string()),
            fork_mode: SpawnAgentForkMode::None,
        })
        .await
        .expect("mailbox prefix")
        .session;
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new("pending-parent-thread"),
            thread_id: ThreadId::new("mailbox-thread"),
            agent_path: "/root/mailbox".to_string(),
            nickname: None,
            role: None,
            last_task_message: Some("initial task".to_string()),
        })
        .await
        .expect("partial child identity");
    store
        .append_agent_mailbox_message(thread_store::AppendAgentMailboxMessageParams {
            message: thread_store::AgentMailboxMessage {
                message_id: "pending-bootstrap-message".to_string(),
                root_thread_id: ThreadId::new("pending-parent-thread"),
                sender_thread_id: ThreadId::new("pending-parent-thread"),
                recipient_thread_id: ThreadId::new("mailbox-thread"),
                content: "initial task".to_string(),
                kind: thread_store::AgentMailboxMessageKind::Message,
                source_turn_id: None,
                result_status: None,
                delivery_mode: AgentMailboxDeliveryMode::TriggerTurn,
                delivery_status: thread_store::AgentMailboxDeliveryStatus::Pending,
                created_at_ms: 1,
                delivered_at_ms: None,
            },
        })
        .await
        .expect("partial bootstrap mailbox");

    for (session_id, thread_id) in [
        ("materialized-session", "materialized-thread"),
        ("mailbox-session", "mailbox-thread"),
    ] {
        assert_session_unloaded(&core, session_id);
        assert!(store
            .read_thread(ReadThreadParams {
                thread_id: ThreadId::new(thread_id),
                include_archived: true,
                turns_view: ThreadTurnsView::Full,
            })
            .await
            .expect("read hidden pending child")
            .is_none());
        assert!(matches!(
            core.read_session_current(AgentSessionReadParams {
                session_id: session_id.to_string(),
                history_limit: None,
                history_offset: None,
                history_before_message_id: None,
            })
            .await,
            Err(RuntimeCoreError::SessionNotFound(missing)) if missing == session_id
        ));
    }
    let listed = core
        .list_agent_sessions(app_server_protocol::AgentSessionListParams::default())
        .await
        .expect("list sessions while spawn is pending");
    assert!(listed.sessions.iter().all(|session| {
        session.session_id != "materialized-session" && session.session_id != "mailbox-session"
    }));
    drop(core);

    let restarted = RuntimeCore::default()
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(store.clone());
    restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect("recover pending prefixes");

    for (session_id, thread_id) in [
        ("intent-only-session", "intent-only-thread"),
        ("materialized-session", "materialized-thread"),
        ("mailbox-session", "mailbox-thread"),
    ] {
        assert!(store
            .read_thread_spawn_parent(ThreadId::new(thread_id))
            .await
            .expect("read cleaned edge")
            .is_none());
        assert!(store
            .read_agent_identity(ThreadId::new(thread_id))
            .await
            .expect("read cleaned identity")
            .is_none());
        assert!(event_log_writer
            .read_session_events(session_id)
            .expect("read cleaned EventLog")
            .is_empty());
        assert_session_unloaded(&restarted, session_id);
    }
    assert!(store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("pending-parent-thread"),
            ThreadId::new("mailbox-thread"),
        )
        .await
        .expect("read cleaned mailbox")
        .is_empty());

    restarted
        .ensure_current_session_hydrated("pending-parent-session")
        .await
        .expect("hydrate parent for retry");
    let retry = restarted
        .stage_agent_control_spawn(AgentControlSpawnRequest {
            parent_session_id: "pending-parent-session".to_string(),
            child_session_id: Some(mailbox_child.session_id),
            child_thread_id: Some(mailbox_child.thread_id),
            fork_mode: SpawnAgentForkMode::None,
        })
        .await
        .expect("stable retry after recovery");
    assert_eq!(retry.session.session_id, "mailbox-session");
    assert_eq!(retry.session.thread_id, "mailbox-thread");
    restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect("clean retry fixture");
}

#[tokio::test]
async fn restart_drops_open_child_when_pending_mail_targets_missing_session_history() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (event_log_writer, store, core) = persistent_core(&temp);
    let parent = core
        .start_session(start_params(
            "orphan-parent-session",
            "orphan-parent-thread",
        ))
        .expect("parent")
        .session;
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new("orphan-parent-thread"),
            thread_id: ThreadId::new("orphan-parent-thread"),
            agent_path: "/root".to_string(),
            nickname: None,
            role: None,
            last_task_message: None,
        })
        .await
        .expect("root identity");
    let child = core
        .create_open_agent_control_child_for_test(AgentControlSpawnRequest {
            parent_session_id: parent.session_id,
            child_session_id: Some("orphan-child-session".to_string()),
            child_thread_id: Some("orphan-child-thread".to_string()),
            fork_mode: SpawnAgentForkMode::None,
        })
        .await
        .expect("open child")
        .session;
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new("orphan-parent-thread"),
            thread_id: ThreadId::new("orphan-child-thread"),
            agent_path: "/root/orphan".to_string(),
            nickname: None,
            role: None,
            last_task_message: Some("resume orphan".to_string()),
        })
        .await
        .expect("child identity");
    store
        .append_agent_mailbox_message(thread_store::AppendAgentMailboxMessageParams {
            message: thread_store::AgentMailboxMessage {
                message_id: "orphan-bootstrap-message".to_string(),
                root_thread_id: ThreadId::new("orphan-parent-thread"),
                sender_thread_id: ThreadId::new("orphan-parent-thread"),
                recipient_thread_id: ThreadId::new("orphan-child-thread"),
                content: "resume orphan".to_string(),
                kind: thread_store::AgentMailboxMessageKind::Message,
                source_turn_id: None,
                result_status: None,
                delivery_mode: AgentMailboxDeliveryMode::TriggerTurn,
                delivery_status: thread_store::AgentMailboxDeliveryStatus::Pending,
                created_at_ms: 1,
                delivered_at_ms: None,
            },
        })
        .await
        .expect("pending bootstrap mailbox");
    store
        .clear_session(&child.session_id)
        .expect("remove child projection history");
    event_log_writer
        .clear_session(&child.session_id)
        .expect("remove child event history");
    drop(core);

    let restarted = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(store.clone());
    restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect("drop unusable open child without blocking startup");

    assert!(store
        .read_thread_spawn_parent(ThreadId::new("orphan-child-thread"))
        .await
        .expect("read cleaned edge")
        .is_none());
    assert!(store
        .read_agent_identity(ThreadId::new("orphan-child-thread"))
        .await
        .expect("read cleaned identity")
        .is_none());
    assert!(store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("orphan-parent-thread"),
            ThreadId::new("orphan-child-thread"),
        )
        .await
        .expect("read cleaned mailbox")
        .is_empty());
    assert_session_unloaded(&restarted, "orphan-child-session");
}

#[tokio::test]
async fn restart_resumes_only_open_children_with_pending_trigger_mail() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (event_log_writer, store, core) = persistent_core(&temp);
    let parent = core
        .start_session(start_params(
            "resume-parent-session",
            "resume-parent-thread",
        ))
        .expect("parent")
        .session;
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new("resume-parent-thread"),
            thread_id: ThreadId::new("resume-parent-thread"),
            agent_path: "/root".to_string(),
            nickname: None,
            role: None,
            last_task_message: None,
        })
        .await
        .expect("root identity");
    let child = core
        .stage_agent_control_spawn(AgentControlSpawnRequest {
            parent_session_id: parent.session_id.clone(),
            child_session_id: Some("resume-child-session".to_string()),
            child_thread_id: Some("resume-child-thread".to_string()),
            fork_mode: SpawnAgentForkMode::None,
        })
        .await
        .expect("stage resumable child")
        .session;
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new("resume-parent-thread"),
            thread_id: ThreadId::new("resume-child-thread"),
            agent_path: "/root/resume".to_string(),
            nickname: None,
            role: None,
            last_task_message: Some("resume after restart".to_string()),
        })
        .await
        .expect("child identity");
    store
        .append_agent_mailbox_message(thread_store::AppendAgentMailboxMessageParams {
            message: thread_store::AgentMailboxMessage {
                message_id: "resume-bootstrap-message".to_string(),
                root_thread_id: ThreadId::new("resume-parent-thread"),
                sender_thread_id: ThreadId::new("resume-parent-thread"),
                recipient_thread_id: ThreadId::new("resume-child-thread"),
                content: "resume after restart".to_string(),
                kind: thread_store::AgentMailboxMessageKind::Message,
                source_turn_id: None,
                result_status: None,
                delivery_mode: AgentMailboxDeliveryMode::TriggerTurn,
                delivery_status: thread_store::AgentMailboxDeliveryStatus::Pending,
                created_at_ms: 1,
                delivered_at_ms: None,
            },
        })
        .await
        .expect("bootstrap mailbox");
    core.commit_agent_control_spawn(&child)
        .await
        .expect("commit resumable child");
    core.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
        parent_session_id: parent.session_id,
        child_session_id: Some("dormant-child-session".to_string()),
        child_thread_id: Some("dormant-child-thread".to_string()),
        fork_mode: SpawnAgentForkMode::None,
    })
    .await
    .expect("dormant child");
    drop(core);

    let restarted = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(store.clone());
    restarted
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect("resume durable bootstrap");

    let resumed = restarted
        .read_session(AgentSessionReadParams {
            session_id: "resume-child-session".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("resumed child");
    assert!(resumed.turns.iter().any(|turn| {
        turn.turn_id
            == super::super::super::agent_mailbox_delivery::mailbox_turn_id(
                "resume-bootstrap-message",
            )
    }));
    assert!(store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("resume-parent-thread"),
            ThreadId::new("resume-child-thread"),
        )
        .await
        .expect("delivered bootstrap mailbox")
        .is_empty());
    assert_session_unloaded(&restarted, "dormant-child-session");
}

#[tokio::test]
async fn restart_restores_forked_child_history_without_parent_hydration() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (event_log_writer, store, core) = persistent_core(&temp);
    let parent = core
        .start_session(start_params("fork-parent-session", "fork-parent-thread"))
        .expect("parent")
        .session;
    append_completed_parent_turn(
        &core,
        &parent,
        "fork-parent-turn-1",
        "user one",
        "assistant one",
    )
    .await;
    append_completed_parent_turn(
        &core,
        &parent,
        "fork-parent-turn-2",
        "user two",
        "assistant two",
    )
    .await;
    core.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
        parent_session_id: parent.session_id,
        child_session_id: Some("fork-child-session".to_string()),
        child_thread_id: Some("fork-child-thread".to_string()),
        fork_mode: SpawnAgentForkMode::FullHistory,
    })
    .await
    .expect("spawn forked child");
    let fork_events = event_log_writer
        .read_session_events("fork-child-session")
        .expect("read forked child EventLog");
    assert_eq!(
        fork_events
            .iter()
            .filter(|record| record.event.event_type == "message.delta")
            .count(),
        2
    );
    assert_eq!(
        fork_events
            .iter()
            .filter(|record| {
                record.event.event_type == "item.completed"
                    && record.event.payload["itemType"] == "agent_message"
            })
            .count(),
        2
    );
    assert!(fork_events.iter().all(|record| {
        record.event.payload.get("trace_id").is_none()
            && record.event.payload.get("request_id").is_none()
            && record.event.payload["phase"] != "commentary"
    }));
    drop(core);

    let restarted = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(store.clone());
    assert_session_unloaded(&restarted, "fork-parent-session");
    assert_session_unloaded(&restarted, "fork-child-session");
    restarted
        .ensure_current_session_hydrated("fork-child-session")
        .await
        .expect("hydrate forked child");
    assert_session_unloaded(&restarted, "fork-parent-session");

    let child = restarted
        .read_session(AgentSessionReadParams {
            session_id: "fork-child-session".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read restored child");
    assert_eq!(child.turns.len(), 2);
    let canonical = store
        .read_thread(ReadThreadParams {
            thread_id: ThreadId::new("fork-child-thread"),
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        })
        .await
        .expect("read canonical child")
        .expect("canonical child");
    assert_eq!(canonical.turns.len(), 2);
    assert_eq!(
        canonical
            .turns
            .iter()
            .flat_map(|turn| turn.items.iter())
            .filter(|item| item.kind == agent_protocol::ItemKind::AgentMessage)
            .count(),
        2
    );
    assert!(canonical
        .turns
        .iter()
        .flat_map(|turn| turn.items.iter())
        .all(|item| item.status == agent_protocol::ItemStatus::Completed));
    assert_eq!(
        canonical.forked_from_id,
        Some(ThreadId::new("fork-parent-thread"))
    );
}

#[tokio::test]
async fn pending_reservation_failure_keeps_event_log_clean_across_stable_retry() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (event_log_writer, store, core) = persistent_core(&temp);
    let parent = core
        .start_session(start_params(
            "cleanup-parent-session",
            "cleanup-parent-thread",
        ))
        .expect("parent")
        .session;
    append_completed_parent_turn(
        &core,
        &parent,
        "cleanup-parent-turn",
        "user input",
        "assistant output",
    )
    .await;
    store
        .upsert_thread_spawn_edge(
            ThreadId::new("cleanup-ancestor-thread"),
            ThreadId::new("cleanup-parent-thread"),
            ThreadSpawnEdgeStatus::Open,
        )
        .await
        .expect("precondition edge");

    for _attempt in 0..2 {
        let error = core
            .create_open_agent_control_child_for_test(AgentControlSpawnRequest {
                parent_session_id: parent.session_id.clone(),
                child_session_id: Some("cleanup-child-session".to_string()),
                child_thread_id: Some("cleanup-ancestor-thread".to_string()),
                fork_mode: SpawnAgentForkMode::FullHistory,
            })
            .await
            .expect_err("cyclic graph must reject child edge");
        assert!(error
            .to_string()
            .contains("failed to reserve canonical child thread spawn"));
        assert!(event_log_writer
            .read_session_events("cleanup-child-session")
            .expect("read cleaned child EventLog")
            .is_empty());
        assert!(store
            .read_thread(ReadThreadParams {
                thread_id: ThreadId::new("cleanup-ancestor-thread"),
                include_archived: true,
                turns_view: ThreadTurnsView::Full,
            })
            .await
            .expect("read cleaned canonical child")
            .is_none());
        assert_session_unloaded(&core, "cleanup-child-session");
    }
}

#[tokio::test]
async fn restart_hydrates_only_the_exact_open_child_on_demand() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (event_log_writer, store, core) = persistent_core(&temp);
    let root = core
        .start_session(start_params("root-session", "root-thread"))
        .expect("root")
        .session;
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
    store
        .upsert_agent_identity(thread_store::AgentIdentity {
            root_thread_id: ThreadId::new("root-thread"),
            thread_id: ThreadId::new("root-thread"),
            agent_path: "/root".to_string(),
            nickname: None,
            role: None,
            last_task_message: None,
        })
        .await
        .expect("root identity");

    let seed_gateway =
        core.agent_control_gateway_for_turn(&root, &root_turn, RuntimeHostContext::default());
    let seed_caller = AgentControlCaller {
        session_id: root.session_id.clone(),
        thread_id: root.thread_id.clone(),
        turn_id: root_turn.turn_id.clone(),
        call_id: String::new(),
    };
    for task_name in ["followup", "interrupt", "closed"] {
        seed_gateway
            .gateway()
            .execute(AgentControlGatewayRequest {
                caller: AgentControlCaller {
                    call_id: format!("spawn-{task_name}"),
                    ..seed_caller.clone()
                },
                command: AgentControlCommand::SpawnAgent {
                    task_name: task_name.to_string(),
                    message: format!("initial {task_name} task"),
                    fork_mode: SpawnAgentForkMode::None,
                    model_overrides: SpawnAgentModelOverrides::default(),
                },
                cancel_token: None,
            })
            .await
            .expect("spawn child through current gateway");
    }
    let followup_identity = spawned_child_identity(&store, "root-thread", "followup").await;
    let interrupt_identity = spawned_child_identity(&store, "root-thread", "interrupt").await;
    let closed_identity = spawned_child_identity(&store, "root-thread", "closed").await;
    let followup_session_id =
        session_id_for_thread(&store, followup_identity.thread_id.clone()).await;
    let interrupt_session_id =
        session_id_for_thread(&store, interrupt_identity.thread_id.clone()).await;
    let closed_session_id = session_id_for_thread(&store, closed_identity.thread_id.clone()).await;

    let followup_before_restart =
        wait_for_session_with_turn(&core, &followup_session_id, "followup child before restart")
            .await;
    let followup_turn = followup_before_restart
        .turns
        .first()
        .expect("followup child initial turn")
        .clone();
    let followup_gateway = core.agent_control_gateway_for_turn(
        &followup_before_restart.session,
        &followup_turn,
        RuntimeHostContext::default(),
    );
    followup_gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                session_id: followup_before_restart.session.session_id,
                thread_id: followup_before_restart.session.thread_id,
                turn_id: followup_turn.turn_id,
                call_id: "spawn-reviewer".to_string(),
            },
            command: AgentControlCommand::SpawnAgent {
                task_name: "reviewer".to_string(),
                message: "review the child work".to_string(),
                fork_mode: SpawnAgentForkMode::None,
                model_overrides: SpawnAgentModelOverrides::default(),
            },
            cancel_token: None,
        })
        .await
        .expect("spawn grandchild through current gateway");
    let reviewer_identity =
        agent_identity_for_path(&store, "root-thread", "/root/followup/reviewer").await;
    let reviewer_session_id =
        session_id_for_thread(&store, reviewer_identity.thread_id.clone()).await;
    let interrupt_turn_id = wait_for_session_with_turn(
        &core,
        &interrupt_session_id,
        "interrupt child before restart",
    )
    .await
    .turns
    .first()
    .expect("interrupt child initial turn")
    .turn_id
    .clone();
    store
        .set_thread_spawn_edge_status(
            closed_identity.thread_id.clone(),
            ThreadSpawnEdgeStatus::Closed,
        )
        .await
        .expect("close child");
    drop(followup_gateway);
    drop(seed_gateway);
    drop(core);

    let restarted = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(store.clone());
    assert_session_unloaded(&restarted, "root-session");
    assert_session_unloaded(&restarted, &followup_session_id);
    assert_session_unloaded(&restarted, &interrupt_session_id);
    assert_session_unloaded(&restarted, &closed_session_id);
    assert_session_unloaded(&restarted, &reviewer_session_id);

    restarted
        .ensure_current_session_hydrated("root-session")
        .await
        .expect("hydrate root");
    assert_session_unloaded(&restarted, &followup_session_id);
    assert_session_unloaded(&restarted, &interrupt_session_id);
    assert_session_unloaded(&restarted, &closed_session_id);
    assert_session_unloaded(&restarted, &reviewer_session_id);

    let gateway =
        restarted.agent_control_gateway_for_turn(&root, &root_turn, RuntimeHostContext::default());
    let caller = AgentControlCaller {
        session_id: root.session_id,
        thread_id: root.thread_id,
        turn_id: root_turn.turn_id,
        call_id: "queue-after-restart".to_string(),
    };
    let queued = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: caller.clone(),
            command: AgentControlCommand::SendMessage {
                target: "followup".to_string(),
                message: "queue only".to_string(),
            },
            cancel_token: None,
        })
        .await
        .expect("queue message");
    let queued_message_id = queued.output["message_id"].as_str().expect("message id");
    assert_session_unloaded(&restarted, &followup_session_id);
    assert!(store
        .list_pending_agent_mailbox_messages(
            ThreadId::new("root-thread"),
            followup_identity.thread_id.clone(),
        )
        .await
        .expect("pending queue")
        .iter()
        .any(|message| {
            message.message_id == queued_message_id
                && message.delivery_mode == AgentMailboxDeliveryMode::QueueOnly
        }));

    let closed = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                call_id: "closed-after-restart".to_string(),
                ..caller.clone()
            },
            command: AgentControlCommand::SendMessage {
                target: "closed".to_string(),
                message: "must stay closed".to_string(),
            },
            cancel_token: None,
        })
        .await
        .expect_err("closed child must stay unaddressable");
    assert!(closed
        .to_string()
        .contains("not in the current durable root-thread tree"));
    assert_session_unloaded(&restarted, &closed_session_id);
    assert_eq!(
        store
            .read_thread_spawn_parent(closed_identity.thread_id.clone())
            .await
            .expect("closed edge audit"),
        Some(thread_store::ThreadSpawnParent {
            parent_thread_id: ThreadId::new("root-thread"),
            status: ThreadSpawnEdgeStatus::Closed,
        })
    );
    assert!(store
        .read_agent_identity(closed_identity.thread_id.clone())
        .await
        .expect("closed identity audit")
        .is_some());
    assert!(store
        .read_thread(ReadThreadParams {
            thread_id: closed_identity.thread_id.clone(),
            include_archived: true,
            turns_view: ThreadTurnsView::NotLoaded,
        })
        .await
        .expect("closed thread audit")
        .is_some());

    gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                call_id: "followup-after-restart".to_string(),
                ..caller.clone()
            },
            command: AgentControlCommand::FollowupTask {
                target: "followup".to_string(),
                message: "start exact child".to_string(),
            },
            cancel_token: None,
        })
        .await
        .expect("followup exact child");
    wait_for_session_with_turn(&restarted, &followup_session_id, "followup child hydrated").await;
    assert_session_unloaded(&restarted, &interrupt_session_id);
    assert_session_unloaded(&restarted, &closed_session_id);
    assert_session_unloaded(&restarted, &reviewer_session_id);

    let interrupted = gateway
        .gateway()
        .execute(AgentControlGatewayRequest {
            caller: AgentControlCaller {
                call_id: "interrupt-after-restart".to_string(),
                ..caller
            },
            command: AgentControlCommand::InterruptAgent {
                target: "interrupt".to_string(),
            },
            cancel_token: None,
        })
        .await
        .expect("interrupt exact child");
    assert_eq!(interrupted.output["previous_status"], "pending_init");
    let interrupted_child = restarted
        .read_session(AgentSessionReadParams {
            session_id: interrupt_session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("interrupt child hydrated");
    assert_eq!(
        interrupted_child
            .turns
            .iter()
            .find(|turn| turn.turn_id == interrupt_turn_id)
            .expect("interrupted initial turn")
            .status,
        app_server_protocol::AgentTurnStatus::Canceled
    );
    assert_session_unloaded(&restarted, &closed_session_id);
    assert_session_unloaded(&restarted, &reviewer_session_id);
}
