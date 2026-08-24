use super::*;
use crate::runtime::session_control::QueuedTurnResume;
use async_trait::async_trait;

struct ReasoningHistoryBackend;

#[async_trait]
impl ExecutionBackend for ReasoningHistoryBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(RuntimeEvent::new(
            "reasoning.delta",
            json!({
                "text": format!("reasoning：{}", request.input.concat_text()),
                "metadata": {
                    "source": "reasoning"
                }
            }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "message.delta",
            json!({
                "text": format!("最终答复：{}", request.input.concat_text()),
                "phase": "final"
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
async fn queued_resume_helper_hydrates_projection_history_without_mutation() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let core = RuntimeCore::with_backend(Arc::new(ReasoningHistoryBackend))
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_resume_projection_history".to_string()),
        thread_id: Some("thread_resume_projection_history".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    for (turn_id, text) in [
        (
            "turn_resume_projection_1",
            "第一轮：建立 thread read 同构基线。",
        ),
        (
            "turn_resume_projection_2",
            "第二轮：验证 resume 后仍读取 projection。",
        ),
    ] {
        core.start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_resume_projection_history".to_string(),
                turn_id: Some(turn_id.to_string()),
                input: AgentInput {
                    text: text.to_string(),
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
    }

    let restarted_core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store);
    let before = restarted_core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_resume_projection_history".to_string(),
            history_limit: Some(10),
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read projection before resume");
    assert_projection_history_detail(before.detail.as_ref().expect("before detail"), true);

    let resume = restarted_core
        .resume_next_queued_turn_if_idle(
            "sess_resume_projection_history",
            RuntimeHostContext::default(),
        )
        .await
        .expect("resume hydrated session");
    assert!(matches!(resume, QueuedTurnResume::Empty));

    let after = restarted_core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_resume_projection_history".to_string(),
            history_limit: Some(10),
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read projection after resume");
    assert_projection_history_detail(after.detail.as_ref().expect("after detail"), false);
}

fn assert_projection_history_detail(detail: &serde_json::Value, expect_projection_source: bool) {
    let messages = detail["messages"].as_array().expect("messages");
    if expect_projection_source {
        assert_eq!(detail["projection_source"], "runtime.projection_1");
    }
    assert_eq!(detail["messages_count"].as_u64(), Some(4));
    assert_eq!(messages.len(), 4);
    assert_eq!(
        messages[0]["content"][0]["text"].as_str(),
        Some("第一轮：建立 thread read 同构基线。")
    );
    assert_eq!(
        messages[1]["content"][0]["text"].as_str(),
        Some("最终答复：第一轮：建立 thread read 同构基线。")
    );
    assert_eq!(
        messages[2]["content"][0]["text"].as_str(),
        Some("第二轮：验证 resume 后仍读取 projection。")
    );
    assert_eq!(
        messages[3]["content"][0]["text"].as_str(),
        Some("最终答复：第二轮：验证 resume 后仍读取 projection。")
    );

    let items = detail["items"].as_array().expect("items");
    assert!(items.iter().any(|item| {
        item["type"].as_str() == Some("reasoning")
            && item["text"]
                .as_str()
                .is_some_and(|text| text.contains("第一轮：建立 thread read 同构基线"))
    }));
    assert!(items.iter().any(|item| {
        item["type"].as_str() == Some("reasoning")
            && item["text"]
                .as_str()
                .is_some_and(|text| text.contains("第二轮：验证 resume 后仍读取 projection"))
    }));
    assert_eq!(detail["thread_read"]["thread_items"], detail["items"]);
    assert_eq!(
        detail["thread_read"]["turns"]
            .as_array()
            .expect("turns")
            .len(),
        2
    );
}
