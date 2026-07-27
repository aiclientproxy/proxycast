use super::*;

struct PhasedAgentMessagesBackend;

#[async_trait]
impl ExecutionBackend for PhasedAgentMessagesBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(RuntimeEvent::new(
            "message.delta",
            json!({
                "itemId": "agent-commentary-1",
                "text": "我先搜索公开资料。",
                "phase": "commentary"
            }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "message.completed",
            json!({
                "itemId": "agent-commentary-1",
                "phase": "commentary",
                "status": "completed"
            }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "message.delta",
            json!({
                "itemId": "agent-final-1",
                "text": "最终答复。",
                "phase": "final_answer"
            }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "message.completed",
            json!({
                "itemId": "agent-final-1",
                "phase": "final_answer",
                "status": "completed"
            }),
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
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

struct ProviderSafetyBufferingBackend;

#[async_trait]
impl ExecutionBackend for ProviderSafetyBufferingBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(RuntimeEvent::new(
            "provider_safety_buffering",
            json!({
                "kind": "provider_safety_buffering",
                "backend": "runtime",
                "provider": "openai",
                "model": "gpt-5-codex",
                "useCases": ["cyber"],
                "reasons": ["policy"],
                "showBufferingUi": true,
                "retryModel": "gpt-5-mini",
                "fallbackHeaderModel": "legacy-fast",
                "source": "payload_retry_model",
                "runtimeEvent": {
                    "type": "provider_stream_event",
                    "payload": {
                        "retryModel": "gpt-5-mini"
                    }
                }
            }),
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
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

#[tokio::test]
async fn read_session_projects_runtime_turns_into_gui_messages() {
    let core = RuntimeCore::with_backend(Arc::new(CompletedBackend));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_messages".to_string()),
        thread_id: Some("thread_messages".to_string()),
        app_id: "desktop".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.session".to_string(),
            id: "sess_messages".to_string(),
            title: Some("Messages Read".to_string()),
            uri: None,
            metadata: None,
        }),
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_messages".to_string(),
            turn_id: Some("turn_messages".to_string()),
            input: AgentInput {
                text: "你好，帮我整理今天的计划".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("turn");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_messages".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");
    let messages = detail["messages"].as_array().expect("messages");

    assert_eq!(detail["messages_count"], 2);
    assert_eq!(detail["history_cursor"]["loaded_count"], 2);
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["id"], "turn_messages:user");
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["runtimeTurnId"], "turn_messages");
    assert_eq!(messages[0]["runtime_turn_id"], "turn_messages");
    assert_eq!(
        messages[0]["content"][0]["text"],
        "你好，帮我整理今天的计划"
    );
    assert_eq!(messages[1]["id"], "turn_messages:assistant");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["runtimeTurnId"], "turn_messages");
    assert_eq!(messages[1]["runtime_turn_id"], "turn_messages");
    assert_eq!(
        messages[1]["content"][0]["text"],
        "你好！有什么可以帮你的吗？"
    );
}

#[tokio::test]
async fn read_session_projects_provider_safety_buffering_into_diagnostics() {
    let core = RuntimeCore::with_backend(Arc::new(ProviderSafetyBufferingBackend));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_safety_buffering".to_string()),
        thread_id: Some("thread_safety_buffering".to_string()),
        app_id: "desktop".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_safety_buffering".to_string(),
            turn_id: Some("turn_safety_buffering".to_string()),
            input: AgentInput {
                text: "触发 safety buffering".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("turn");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_safety_buffering".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");
    let diagnostics = &detail["thread_read"]["diagnostics"];
    let latest = &diagnostics["latest_provider_safety_buffering"];

    assert_eq!(diagnostics["provider_safety_buffering_count"], 1);
    assert_eq!(latest["thread_id"], "thread_safety_buffering");
    assert_eq!(latest["turn_id"], "turn_safety_buffering");
    assert_eq!(latest["provider"], "openai");
    assert_eq!(latest["model"], "gpt-5-codex");
    assert_eq!(latest["use_cases"], json!(["cyber"]));
    assert_eq!(latest["reasons"], json!(["policy"]));
    assert_eq!(latest["show_buffering_ui"], true);
    assert_eq!(latest["retry_model"], "gpt-5-mini");
    assert_eq!(latest["fallback_header_model"], "legacy-fast");
    assert_eq!(latest["source"], "payload_retry_model");
    assert_eq!(latest["backend"], "runtime");
    assert!(latest.get("runtimeEvent").is_none());
    assert!(latest.get("retryModel").is_none());
    assert_eq!(
        detail["thread_read"]["runtime_summary"]["latestProviderSafetyBuffering"],
        diagnostics["latest_provider_safety_buffering"]
    );
}

#[tokio::test]
async fn read_session_does_not_project_commentary_phase_as_final_message() {
    let core = RuntimeCore::with_backend(Arc::new(PhasedAgentMessagesBackend));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_phased_messages".to_string()),
        thread_id: Some("thread_phased_messages".to_string()),
        app_id: "desktop".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_phased_messages".to_string(),
            turn_id: Some("turn_phased_messages".to_string()),
            input: AgentInput {
                text: "搜索资料".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("turn");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_phased_messages".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");
    let messages = detail["messages"].as_array().expect("messages");
    let items = detail["items"].as_array().expect("items");

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"][0]["text"], "最终答复。");
    assert!(items.iter().any(|item| {
        item["type"] == "agent_message"
            && item["id"] == "agent-commentary-1"
            && item["phase"] == "commentary"
            && item["text"] == "我先搜索公开资料。"
    }));
    assert!(items.iter().any(|item| {
        item["type"] == "agent_message"
            && item["id"] == "agent-final-1"
            && item["phase"] == "final_answer"
            && item["text"] == "最终答复。"
    }));
}

#[tokio::test]
async fn read_session_projects_failed_runtime_event_into_diagnostics_and_error_item() {
    let core = RuntimeCore::with_backend(Arc::new(PartialFailureBackend));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_failed_read".to_string()),
        thread_id: Some("thread_failed_read".to_string()),
        app_id: "desktop".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.session".to_string(),
            id: "sess_failed_read".to_string(),
            title: Some("Failed Read".to_string()),
            uri: None,
            metadata: None,
        }),
        locale: None,
    })
    .expect("session");

    let error = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_failed_read".to_string(),
                turn_id: Some("turn_failed_read".to_string()),
                input: AgentInput {
                    text: "整理今天的国际新闻".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect_err("backend failure should propagate");
    let expected_error_message = error.to_string();
    assert!(expected_error_message.contains("provider stream timed out"));
    let events = core
        .events_for_session("sess_failed_read")
        .expect("failed runtime events");
    let failed = events
        .iter()
        .find(|event| event.event_type == "turn.failed")
        .expect("turn failed event");
    assert_eq!(failed.payload["reason"], "turn_error");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_failed_read".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read failed session");
    let detail = read.detail.expect("session detail");

    assert_eq!(
        detail["thread_read"]["diagnostics"]["latest_turn_status"],
        "failed"
    );
    assert_eq!(
        detail["thread_read"]["diagnostics"]["latest_turn_error_message"].as_str(),
        Some(expected_error_message.as_str())
    );
    assert_eq!(
        detail["thread_read"]["runtime_summary"]["latestTurnErrorMessage"].as_str(),
        Some(expected_error_message.as_str())
    );

    let messages = detail["messages"].as_array().expect("messages");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"][0]["text"], "整理今天的国际新闻");

    let items = detail["items"].as_array().expect("items");
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["type"], "error");
    assert_eq!(items[0]["status"], "failed");
    assert_eq!(
        items[0]["message"].as_str(),
        Some(expected_error_message.as_str())
    );
}

#[tokio::test]
async fn usage_limit_failure_preserves_structured_terminal_reason() {
    let core = RuntimeCore::with_backend(Arc::new(UsageLimitFailureBackend));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_usage_limit_read".to_string()),
        thread_id: Some("thread_usage_limit_read".to_string()),
        app_id: "desktop".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_usage_limit_read".to_string(),
            turn_id: Some("turn_usage_limit_read".to_string()),
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
    .expect_err("usage limit must fail the turn");

    let events = core
        .events_for_session("sess_usage_limit_read")
        .expect("usage-limit runtime events");
    let failed = events
        .iter()
        .find(|event| event.event_type == "turn.failed")
        .expect("turn failed event");
    assert_eq!(failed.payload["message"], "provider quota exhausted");
    assert_eq!(failed.payload["reason"], "usage_limit_exceeded");
}

#[tokio::test]
async fn runtime_failure_updates_bound_thread_goal_with_codex_status() {
    let cases: Vec<(
        Arc<dyn ExecutionBackend>,
        &str,
        app_server_protocol::protocol::v2::ThreadGoalStatus,
    )> = vec![
        (
            Arc::new(PartialFailureBackend),
            "blocked",
            app_server_protocol::protocol::v2::ThreadGoalStatus::Blocked,
        ),
        (
            Arc::new(UsageLimitFailureBackend),
            "usage-limited",
            app_server_protocol::protocol::v2::ThreadGoalStatus::UsageLimited,
        ),
    ];

    for (backend, suffix, expected_status) in cases {
        let temp = tempfile::tempdir().expect("goal failure tempdir");
        let projection_store = Arc::new(
            ProjectionStore::initialize(temp.path().join("state.sqlite"))
                .expect("goal failure projection store"),
        );
        let core =
            RuntimeCore::with_backend(backend).with_projection_store(Arc::clone(&projection_store));
        let session_id = format!("sess_goal_failure_{suffix}");
        let thread_id = format!("thread_goal_failure_{suffix}");
        let turn_id = format!("turn_goal_failure_{suffix}");
        core.start_session(AgentSessionStartParams {
            session_id: Some(session_id.clone()),
            thread_id: Some(thread_id.clone()),
            app_id: "desktop".to_string(),
            workspace_id: Some("workspace-main".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("goal failure session");
        core.set_thread_goal(app_server_protocol::protocol::v2::ThreadGoalSetParams {
            thread_id: thread_id.clone(),
            objective: Some("finish the active goal".to_string()),
            status: None,
            token_budget: None,
        })
        .expect("set active thread goal");

        core.start_turn(
            AgentSessionTurnStartParams {
                session_id,
                turn_id: Some(turn_id),
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
        .expect_err("terminal backend error must propagate");

        let events = core
            .events_for_session(&format!("sess_goal_failure_{suffix}"))
            .expect("goal failure runtime events");
        assert!(
            events
                .iter()
                .any(|event| event.event_type == "turn.accepted"),
            "case={suffix} must preserve canonical turn admission"
        );
        let goal = core
            .get_thread_goal(&thread_id)
            .expect("read terminal thread goal")
            .expect("terminal thread goal");
        assert_eq!(goal.status, expected_status, "case={suffix}");
    }
}

#[tokio::test]
async fn read_session_projects_model_routing_into_thread_read() {
    let core = RuntimeCore::with_backend(Arc::new(RoutingDecisionReadModelBackend));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_routing_read".to_string()),
        thread_id: Some("thread_routing_read".to_string()),
        app_id: "desktop".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.session".to_string(),
            id: "sess_routing_read".to_string(),
            title: Some("Routing Read".to_string()),
            uri: None,
            metadata: None,
        }),
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_routing_read".to_string(),
            turn_id: Some("turn_routing_read".to_string()),
            input: AgentInput {
                text: "帮我修改代码".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("turn");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_routing_read".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");
    let routing = &detail["thread_read"]["model_routing"];

    assert_eq!(
        detail["thread_read"]["service_model_slot"].as_str(),
        Some("coding")
    );
    assert_eq!(
        routing["decisionSource"].as_str(),
        Some("profile_model_slot")
    );
    assert_eq!(
        routing["decisionReason"].as_str(),
        Some("profile_slot_selected")
    );
    assert_eq!(routing["serviceModelSlot"].as_str(), Some("coding"));
    assert_eq!(routing["selectedProvider"].as_str(), Some("custom-coding"));
    assert_eq!(routing["selectedModel"].as_str(), Some("coder-large"));
    assert_eq!(
        routing["providerReadiness"]["status"].as_str(),
        Some("ready")
    );
    assert_eq!(
        routing["modelRegistry"]["reasonCode"].as_str(),
        Some("matched_provider_models")
    );
    assert_eq!(
        routing["modelRegistry"]["modelCapabilities"]["capabilities"]["reasoning"].as_bool(),
        Some(true)
    );
    assert_eq!(
        routing["modelRegistry"]["modelAlias"]["providerModelId"].as_str(),
        Some("coder-large")
    );
    assert_eq!(
        routing["modelSlot"]["slots"][1]["slot"].as_str(),
        Some("review")
    );
    assert_eq!(
        routing["modelTaskRequest"]["requirements"]["taskFamilies"][0].as_str(),
        Some("chat")
    );
    assert_eq!(
        routing["resolvedRoute"]["protocol"].as_str(),
        Some("openai_chat")
    );
    assert_eq!(
        routing["resolvedRoute"]["modelRef"]["providerId"].as_str(),
        Some("custom-coding")
    );
    assert_eq!(
        routing["resolvedRoute"]["decision"]["serviceModelSlot"].as_str(),
        Some("coding")
    );
    assert_eq!(
        detail["thread_read"]["runtime_summary"]["decisionSource"].as_str(),
        Some("profile_model_slot")
    );
}
