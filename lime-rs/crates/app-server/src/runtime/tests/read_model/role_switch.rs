use super::super::*;

struct ExpertRoleSwitchBackend;

#[async_trait]
impl ExecutionBackend for ExpertRoleSwitchBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let role_switch = request
            .runtime_metadata()
            .and_then(|metadata| metadata.pointer("/harness/expert_role_switch"))
            .expect("role switch metadata");
        assert_eq!(role_switch["kind"], "expert_profile_switch");
        assert_eq!(role_switch["scope"], "thread");

        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(RuntimeEvent::new(
            "message.delta",
            json!({ "text": "已按文案专家继续处理。" }),
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
async fn read_session_projects_thread_expert_role_switch_metadata_into_items() {
    let core = RuntimeCore::with_backend(Arc::new(ExpertRoleSwitchBackend));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_expert_role_switch".to_string()),
        thread_id: Some("thread_expert_role_switch".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_expert_role_switch".to_string(),
            turn_id: Some("turn_expert_role_switch".to_string()),
            input: AgentInput {
                text: "把上面的商业分析改成投资人能读懂的版本".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: Some(RuntimeOptions {
                runtime_request: Some(RuntimeRequest {
                    metadata: Some(json!({
                    "expert": {
                        "expertId": "copywriter",
                        "releaseId": "copywriter-release-1"
                    },
                    "harness": {
                        "expert": {
                            "expert_id": "copywriter",
                            "release_id": "copywriter-release-1"
                        },
                        "expert_role_switch": {
                            "kind": "expert_profile_switch",
                            "scope": "thread",
                            "source": "expert_info_panel",
                            "previous_expert_id": "business-analyst",
                            "previous_release_id": "business-release-1",
                            "next_expert_id": "copywriter",
                            "next_release_id": "copywriter-release-1",
                            "switched_at": "2026-07-05T10:00:00.000Z"
                        }
                    }
                    })),
                    ..RuntimeRequest::default()
                }),
                ..RuntimeOptions::default()
            }),
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("turn");

    let read = core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_expert_role_switch".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read session");
    let detail = read.detail.expect("detail");
    let items = detail["items"].as_array().expect("items");
    let role_switch_item = items
        .iter()
        .find(|item| item["type"].as_str() == Some("expert_profile_switch"))
        .expect("expert profile switch item");

    assert_eq!(role_switch_item["turn_id"], "turn_expert_role_switch");
    assert_eq!(
        role_switch_item["previous_expert_id"].as_str(),
        Some("business-analyst")
    );
    assert_eq!(
        role_switch_item["next_expert_id"].as_str(),
        Some("copywriter")
    );
    assert_eq!(
        role_switch_item["metadata"]["harness"]["expert_role_switch"]["kind"].as_str(),
        Some("expert_profile_switch")
    );
    assert_eq!(
        role_switch_item["metadata"]["harness"]["expert"]["expert_id"].as_str(),
        Some("copywriter")
    );
}
