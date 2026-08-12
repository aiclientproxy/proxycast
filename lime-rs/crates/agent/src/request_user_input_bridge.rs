//! request_user_input 工具桥接
//!
//! 将 current request_user_input callback 桥接到 session-scoped action state，
//! 通过 elicitation 事件把问题发送到前端并等待用户输入。

use crate::protocol::AgentEvent;
use agent_protocol::action_required::ActionRequiredScope as RuntimeActionRequiredScope;
use agent_protocol::action_required::ASK_USER_ACTION_TYPE;
use agent_runtime::action_required::ActionRequiredState;
use agent_runtime::request_user_input::{
    run_request_user_input, RequestUserInputAction, RequestUserInputCallback,
    RequestUserInputGateway, RequestUserInputRequest, RequestUserInputRunRequest,
    DEFAULT_REQUEST_USER_INPUT_TIMEOUT_SECS,
};
use agent_runtime::session_loop::{RuntimeSessionInputHandle, RuntimeSessionResponseKind};
use futures::future::BoxFuture;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;

struct RuntimeActionRequiredGateway {
    state: Arc<ActionRequiredState>,
    response_handle: RuntimeSessionInputHandle,
    item_id: String,
    event_sender: UnboundedSender<AgentEvent>,
}

impl RequestUserInputGateway for RuntimeActionRequiredGateway {
    fn request_user_input<'a>(
        &'a self,
        action: RequestUserInputAction,
    ) -> BoxFuture<'a, anyhow::Result<serde_json::Value>> {
        Box::pin(async move {
            let event_sender = self.event_sender.clone();
            self.state
                .request_action_and_wait_with_notification(
                    self.response_handle.clone(),
                    RuntimeSessionResponseKind::AskUser,
                    ASK_USER_ACTION_TYPE,
                    Some(self.item_id.clone()),
                    Vec::new(),
                    action.scope,
                    action.prompt,
                    action.requested_schema,
                    action.timeout,
                    move |queued| {
                        let questions = queued
                            .requested_schema
                            .get(agent_runtime::request_user_input::REQUEST_USER_INPUT_QUESTIONS_SCHEMA_KEY)
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!([]));
                        let _ = event_sender.send(AgentEvent::ActionRequired {
                            request_id: queued.id.clone(),
                            action_type: queued.action_type.clone(),
                            data: serde_json::json!({
                                "actionType": ASK_USER_ACTION_TYPE,
                                "toolCallId": queued.tool_id,
                                "prompt": queued.message,
                                "questions": questions,
                                "autoResolutionMs": action.auto_resolution_ms,
                                "createdAtMs": queued.created_at_ms,
                                "deadlineAtMs": queued.deadline_at_ms,
                            }),
                            scope: queued.scope.clone(),
                        });
                    },
                )
                .await
        })
    }
}

/// 创建 request_user_input 回调
pub(crate) fn create_request_user_input_callback(
    state: Arc<ActionRequiredState>,
    response_handle: RuntimeSessionInputHandle,
    item_id: String,
    scope: Option<RuntimeActionRequiredScope>,
    event_sender: UnboundedSender<AgentEvent>,
) -> RequestUserInputCallback {
    Arc::new(move |request: RequestUserInputRequest| {
        let state = Arc::clone(&state);
        let response_handle = response_handle.clone();
        let item_id = item_id.clone();
        let scope = scope.clone();
        let event_sender = event_sender.clone();
        Box::pin(async move {
            let run_request = RequestUserInputRunRequest::new(
                request,
                scope,
                Duration::from_secs(DEFAULT_REQUEST_USER_INPUT_TIMEOUT_SECS),
            );
            let gateway = RuntimeActionRequiredGateway {
                state,
                response_handle,
                item_id,
                event_sender,
            };

            match run_request_user_input(&gateway, run_request).await {
                Ok(response) => response,
                Err(err) => {
                    tracing::warn!(
                        "[AgentRuntime][RequestUserInputBridge] 用户输入等待失败: prompt='{}', err={}",
                        err.prompt(),
                        err.message()
                    );
                    None
                }
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_runtime::request_user_input::{RequestUserInputOption, RequestUserInputQuestion};
    use agent_runtime::session_loop::{RuntimeSessionClosureTask, RuntimeSessionRegistry};
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    #[test]
    fn request_user_input_callback_uses_current_dto() {
        let request = RequestUserInputRequest {
            questions: vec![
                RequestUserInputQuestion::new("第一问"),
                RequestUserInputQuestion {
                    id: Some("mode".to_string()),
                    question: "第二问".to_string(),
                    header: Some("mode".to_string()),
                    options: vec![
                        RequestUserInputOption::with_label("auto", "自动执行"),
                        RequestUserInputOption::with_label("confirm", "确认后执行"),
                    ],
                    multi_select: false,
                },
            ],
            auto_resolution_ms: None,
        };

        assert_eq!(request.questions.len(), 2);
        assert_eq!(request.questions[0].question, "第一问");
        assert_eq!(request.questions[1].id.as_deref(), Some("mode"));
        assert_eq!(request.questions[1].header.as_deref(), Some("mode"));
        assert_eq!(request.questions[1].options[0].value, "auto");
        assert_eq!(
            request.questions[1].options[0].label.as_deref(),
            Some("自动执行")
        );
        assert!(!request.questions[1].multi_select);
    }

    #[tokio::test]
    async fn callback_emits_action_and_resumes_matching_pending_request() {
        let state = Arc::new(ActionRequiredState::default());
        let scope = RuntimeActionRequiredScope {
            session_id: Some("session-1".to_string()),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
        };
        let (event_sender, mut event_receiver) = tokio::sync::mpsc::unbounded_channel();
        let request = RequestUserInputRequest {
            questions: vec![RequestUserInputQuestion {
                id: Some("mode".to_string()),
                question: "请选择执行模式".to_string(),
                header: Some("mode".to_string()),
                options: vec![
                    RequestUserInputOption::with_label("auto", "自动执行"),
                    RequestUserInputOption::with_label("confirm", "确认后执行"),
                ],
                multi_select: false,
            }],
            auto_resolution_ms: Some(60_000),
        };
        let registry = RuntimeSessionRegistry::default();
        let session = registry
            .get_or_create("session-1", "thread-1")
            .await
            .expect("bind user-input bridge session actor");
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        let response_tx = Arc::new(StdMutex::new(Some(response_tx)));
        let task_state = Arc::clone(&state);
        let task_scope = scope.clone();
        let task = RuntimeSessionClosureTask::new(
            "turn-1",
            Vec::new(),
            move |context, _input, _cancel| {
                let callback = create_request_user_input_callback(
                    Arc::clone(&task_state),
                    context.input_handle(),
                    "item-request-user-input-1".to_string(),
                    Some(task_scope.clone()),
                    event_sender.clone(),
                );
                let request = request.clone();
                let response_tx = Arc::clone(&response_tx);
                Box::pin(async move {
                    let response = callback(request).await;
                    if let Some(response_tx) =
                        response_tx.lock().expect("response sender lock").take()
                    {
                        let _ = response_tx.send(response);
                    }
                    Ok(())
                })
            },
        );
        let submission = session
            .submit(Arc::new(task), false)
            .await
            .expect("request task");
        let event = tokio::time::timeout(Duration::from_secs(1), event_receiver.recv())
            .await
            .expect("action event timeout")
            .expect("action event");
        let AgentEvent::ActionRequired {
            request_id,
            action_type,
            data,
            scope: event_scope,
            ..
        } = event
        else {
            panic!("expected action_required event");
        };
        assert_eq!(action_type, "ask_user");
        assert_eq!(data["toolCallId"], "item-request-user-input-1");
        assert_eq!(data["autoResolutionMs"], 60_000);
        assert_eq!(event_scope, Some(scope.clone()));

        let descriptors = state.pending_action_descriptors().await;
        assert_eq!(descriptors.len(), 1);
        assert_eq!(descriptors[0].request_id, request_id);
        assert_eq!(descriptors[0].action_type, ASK_USER_ACTION_TYPE);
        assert_eq!(descriptors[0].scope, Some(scope.clone()));

        state
            .resolve_action(&request_id, Some(&scope))
            .await
            .expect("resolve action");
        session
            .answer_user_input(
                Some("turn-1"),
                &request_id,
                serde_json::json!({ "answer": "确认后执行" }),
            )
            .await
            .expect("typed response");
        let response = response_rx.await.expect("pending callback");
        assert!(response.is_some());
        assert_eq!(
            submission.completion.await.expect("task completion"),
            Ok(agent_runtime::session_loop::RuntimeSessionTaskOutcome::Completed)
        );
        registry.shutdown("session-1").await.expect("shutdown");
    }
}
