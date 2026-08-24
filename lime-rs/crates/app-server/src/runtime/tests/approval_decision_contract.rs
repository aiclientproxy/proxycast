use super::support::canonical_tool_started_event;
use super::*;
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct ShellApprovalOnceOnlyBackend {
    respond_count: AtomicUsize,
}

#[async_trait]
impl ExecutionBackend for ShellApprovalOnceOnlyBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(canonical_tool_started_event(
            "sess_shell_approval_decision_contract",
            "thread_shell_approval_decision_contract",
            "turn_shell_approval_decision_contract",
            "shell-tool-1",
            "Bash",
        ))?;
        sink.emit(RuntimeEvent::new(
            "action.required",
            json!({
                "requestId": "shell-approval-1",
                "actionId": "shell-approval-1",
                "actionType": "tool_confirmation",
                "actionKind": "tool_execution_policy",
                "availableDecisions": ["allow_once", "allow_for_session", "decline", "cancel"],
                "toolCallId": "shell-tool-1",
                "toolName": "Bash",
                "toolFamily": "shell_command",
                "runtime_contract": {
                    "contract_key": "shell_command",
                    "tool_family": "shell_command",
                    "session_cache_supported": false
                },
                "approvalScope": {
                    "contractKey": "shell_command",
                    "toolFamily": "shell_command",
                    "riskClass": "shell_command_requires_approval",
                    "workingDirHash": "sha256:test"
                },
                "prompt": "是否允许执行命令？",
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
        self.respond_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
async fn shell_allow_for_session_fails_when_contract_does_not_support_it() {
    let backend = Arc::new(ShellApprovalOnceOnlyBackend {
        respond_count: AtomicUsize::new(0),
    });
    let core = RuntimeCore::with_backend(backend.clone());
    let session_id = "sess_shell_approval_decision_contract".to_string();
    let thread_id = "thread_shell_approval_decision_contract".to_string();
    let turn_id = "turn_shell_approval_decision_contract".to_string();

    core.start_session(AgentSessionStartParams {
        session_id: Some(session_id.clone()),
        thread_id: Some(thread_id.clone()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-permission".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: session_id.clone(),
            turn_id: Some(turn_id.clone()),
            input: AgentInput {
                text: "运行需要确认的命令".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: true,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("turn should enter waiting action");

    let error = core
        .respond_action(
            AgentSessionActionRespondParams {
                session_id: session_id.clone(),
                request_id: "shell-approval-1".to_string(),
                action_type: AgentSessionActionType::ToolConfirmation,
                decision: Some(AgentSessionApprovalDecision::AllowForSession),
                confirmed: None,
                response: Some("{\"answer\":\"本会话允许\"}".to_string()),
                user_data: Some(json!({ "answer": "本会话允许" })),
                metadata: None,
                event_name: None,
                action_scope: Some(AgentSessionActionScope {
                    session_id: Some(session_id),
                    thread_id: Some(thread_id),
                    turn_id: Some(turn_id),
                }),
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect_err("unsupported shell allow_for_session must fail closed");

    assert!(error.to_string().contains(
        "approval decision 'allow_for_session' is not available for tool_confirmation request 'shell-approval-1'"
    ));
    assert_eq!(backend.respond_count.load(Ordering::SeqCst), 0);
}
