use crate::server_request::{PendingServerRequest, ServerRequestError, ServerRequestOwner};
use crate::AppServer;
use app_server_protocol::protocol::v2::{
    CommandExecutionApprovalDecision, CommandExecutionRequestApprovalParams,
    CommandExecutionRequestApprovalResponse, FileChangeApprovalDecision,
    FileChangeRequestApprovalParams, FileChangeRequestApprovalResponse, ServerNotification,
    ServerRequestResolvedNotification, ToolRequestUserInputOption, ToolRequestUserInputParams,
    ToolRequestUserInputQuestion, ToolRequestUserInputResponse,
    METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL, METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
    METHOD_ITEM_TOOL_REQUEST_USER_INPUT,
};
use app_server_protocol::{
    error_codes, AgentEvent, AgentSessionActionRespondParams, AgentSessionActionScope,
    AgentSessionActionType, AgentSessionApprovalDecision, RequestId,
};
use serde_json::Value;
use std::collections::BTreeMap;

impl AppServer {
    pub(crate) async fn handle_command_approval_request(&self, event: AgentEvent) {
        let request = match action_server_request(&event) {
            Ok(Some(request)) => request,
            Ok(None) => return,
            Err(error) => {
                tracing::warn!(event_id = %event.event_id, %error, "invalid action server request event");
                return;
            }
        };
        let response = match request {
            ActionServerRequest::Command(request) => {
                let CommandApprovalRequest { params, identity } = request;
                let response = self
                    .wait_server_request::<_, CommandExecutionRequestApprovalResponse>(
                        METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
                        &identity.thread_id,
                        params,
                    )
                    .await;
                let decision = match response {
                    Ok(response) => approval_decision(response.decision),
                    Err(error) => {
                        let Some(decision) = failed_approval_decision(&error) else {
                            return;
                        };
                        if let WaitServerRequestError::Failed(error) = error {
                            tracing::warn!(%error, "command approval server request failed closed");
                        }
                        decision
                    }
                };
                RuntimeActionResponse::approval(identity, decision)
            }
            ActionServerRequest::FileChange(request) => {
                let FileChangeApprovalRequest { params, identity } = request;
                let response = self
                    .wait_server_request::<_, FileChangeRequestApprovalResponse>(
                        METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
                        &identity.thread_id,
                        params,
                    )
                    .await;
                let decision = match response {
                    Ok(response) => file_approval_decision(response.decision),
                    Err(error) => {
                        let Some(decision) = failed_approval_decision(&error) else {
                            return;
                        };
                        if let WaitServerRequestError::Failed(error) = error {
                            tracing::warn!(%error, "file approval server request failed closed");
                        }
                        decision
                    }
                };
                RuntimeActionResponse::approval(identity, decision)
            }
            ActionServerRequest::UserInput(request) => {
                let UserInputRequest { params, identity } = request;
                let response = self
                    .wait_server_request::<_, ToolRequestUserInputResponse>(
                        METHOD_ITEM_TOOL_REQUEST_USER_INPUT,
                        &identity.thread_id,
                        params,
                    )
                    .await;
                let response = match response {
                    Ok(response) => response,
                    Err(WaitServerRequestError::Transition) => return,
                    Err(WaitServerRequestError::Failed(error)) => {
                        tracing::warn!(%error, "user-input server request failed closed");
                        ToolRequestUserInputResponse {
                            answers: BTreeMap::new(),
                        }
                    }
                };
                RuntimeActionResponse::user_input(identity, response)
            }
        };

        if let Err(error) = self.respond_to_runtime_action(response).await {
            tracing::warn!(event_id = %event.event_id, %error, "typed server request response rejected by runtime");
        }
    }

    pub(crate) async fn wait_server_request<P, R>(
        &self,
        method: &str,
        thread_id: &str,
        params: P,
    ) -> Result<R, WaitServerRequestError>
    where
        P: serde::Serialize,
        R: serde::de::DeserializeOwned,
    {
        let pending = self
            .begin_server_request(method, params)
            .await
            .map_err(|error| WaitServerRequestError::Failed(error.to_string()))?;
        self.finish_server_request(thread_id, pending).await
    }

    async fn finish_server_request<R>(
        &self,
        thread_id: &str,
        mut pending: PendingServerRequest,
    ) -> Result<R, WaitServerRequestError>
    where
        R: serde::de::DeserializeOwned,
    {
        let request_id = pending.id().clone();
        let terminal = pending.wait_terminal().await;
        if !terminal.resolved_before_transition {
            if let Err(error) = self
                .publish_server_request_resolved(thread_id, request_id, terminal.owner)
                .await
            {
                tracing::warn!(%thread_id, %error, "failed to publish typed server-request terminal notification");
            }
        }
        let value = match terminal.result {
            Ok(value) => value,
            Err(ServerRequestError::ClientRejected { error, .. })
                if error.code == error_codes::REQUEST_CANCELLED =>
            {
                return Err(WaitServerRequestError::Transition);
            }
            Err(error) => return Err(WaitServerRequestError::Failed(error.to_string())),
        };
        serde_json::from_value(value)
            .map_err(|error| WaitServerRequestError::Failed(error.to_string()))
    }

    async fn publish_server_request_resolved(
        &self,
        thread_id: &str,
        request_id: RequestId,
        owner: Option<ServerRequestOwner>,
    ) -> Result<(), String> {
        let Some(owner) = owner else {
            return Ok(());
        };
        let origin_connection_id = match owner {
            ServerRequestOwner::Transport(connection_id) => Some(connection_id),
            #[cfg(test)]
            ServerRequestOwner::Subscriber => None,
        };
        let notification =
            ServerNotification::ServerRequestResolved(ServerRequestResolvedNotification {
                thread_id: thread_id.to_string(),
                request_id,
            });
        let (completion_tx, completion_rx) = tokio::sync::oneshot::channel();
        self.event_bridge()
            .send_thread_command(
                agent_protocol::ThreadId::new(thread_id),
                crate::thread_state::ThreadListenerCommand::PublishNotification {
                    notification: notification.into(),
                    origin_connection_id,
                    completion_tx: Some(completion_tx),
                },
            )
            .await?;
        completion_rx.await.map_err(|error| {
            format!("server-request resolved completion channel closed: {error}")
        })?
    }

    async fn respond_to_runtime_action(
        &self,
        response: RuntimeActionResponse,
    ) -> Result<(), String> {
        let params = AgentSessionActionRespondParams {
            session_id: response.identity.session_id.clone(),
            request_id: response.identity.request_id,
            action_type: response.action_type,
            decision: response.decision,
            confirmed: response.confirmed,
            response: None,
            user_data: response.user_data,
            metadata: None,
            event_name: None,
            action_scope: Some(AgentSessionActionScope {
                session_id: Some(response.identity.session_id),
                thread_id: Some(response.identity.thread_id.clone()),
                turn_id: Some(response.identity.turn_id),
            }),
        };
        let output = self
            .processor
            .runtime()
            .respond_action(params, crate::RuntimeHostContext::default())
            .await
            .map_err(|error| error.to_string())?;
        let bridge = self.event_bridge();
        let thread_id = agent_protocol::ThreadId::new(response.identity.thread_id);
        for event in output.events {
            bridge
                .send_thread_command(
                    thread_id.clone(),
                    crate::thread_state::ThreadListenerCommand::PublishRuntimeEvent {
                        event,
                        completion_tx: None,
                    },
                )
                .await
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum WaitServerRequestError {
    Transition,
    Failed(String),
}

fn failed_approval_decision(
    error: &WaitServerRequestError,
) -> Option<AgentSessionApprovalDecision> {
    match error {
        WaitServerRequestError::Transition => None,
        WaitServerRequestError::Failed(_) => Some(AgentSessionApprovalDecision::Decline),
    }
}

enum ActionServerRequest {
    Command(CommandApprovalRequest),
    FileChange(FileChangeApprovalRequest),
    UserInput(UserInputRequest),
}

struct RuntimeActionIdentity {
    request_id: String,
    session_id: String,
    thread_id: String,
    turn_id: String,
}

struct RuntimeActionResponse {
    identity: RuntimeActionIdentity,
    action_type: AgentSessionActionType,
    decision: Option<AgentSessionApprovalDecision>,
    confirmed: Option<bool>,
    user_data: Option<Value>,
}

impl RuntimeActionResponse {
    fn approval(identity: RuntimeActionIdentity, decision: AgentSessionApprovalDecision) -> Self {
        Self {
            identity,
            action_type: AgentSessionActionType::ToolConfirmation,
            decision: Some(decision),
            confirmed: None,
            user_data: None,
        }
    }

    fn user_input(identity: RuntimeActionIdentity, response: ToolRequestUserInputResponse) -> Self {
        let answered = !response.answers.is_empty();
        let answers = response
            .answers
            .into_iter()
            .map(|(id, answer)| {
                let value = match answer.answers.as_slice() {
                    [answer] => Value::String(answer.clone()),
                    answers => serde_json::json!(answers),
                };
                (id, value)
            })
            .collect::<serde_json::Map<_, _>>();
        Self {
            identity,
            action_type: AgentSessionActionType::AskUser,
            decision: None,
            confirmed: Some(answered),
            user_data: Some(Value::Object(answers)),
        }
    }
}

struct CommandApprovalRequest {
    params: CommandExecutionRequestApprovalParams,
    identity: RuntimeActionIdentity,
}

struct FileChangeApprovalRequest {
    params: FileChangeRequestApprovalParams,
    identity: RuntimeActionIdentity,
}

struct UserInputRequest {
    params: ToolRequestUserInputParams,
    identity: RuntimeActionIdentity,
}

fn action_server_request(event: &AgentEvent) -> Result<Option<ActionServerRequest>, String> {
    if let Some(request) = user_input_request(event)? {
        return Ok(Some(ActionServerRequest::UserInput(request)));
    }
    if let Some(request) = file_change_approval_request(event)? {
        return Ok(Some(ActionServerRequest::FileChange(request)));
    }
    command_approval_request(event).map(|request| request.map(ActionServerRequest::Command))
}

fn command_approval_request(event: &AgentEvent) -> Result<Option<CommandApprovalRequest>, String> {
    if event.event_type != "action.required" {
        return Ok(None);
    }
    let action_type = payload_string(&event.payload, &["actionType", "action_type"]);
    if action_type.as_deref() != Some("tool_confirmation") {
        return Ok(None);
    }
    let session_id = required_id(&event.session_id, "sessionId")?;
    let thread_id = required_optional_id(event.thread_id.as_deref(), "threadId")?;
    let turn_id = required_optional_id(event.turn_id.as_deref(), "turnId")?;
    let request_id = payload_string(
        &event.payload,
        &["requestId", "request_id", "actionId", "action_id"],
    )
    .ok_or_else(|| "action.required tool_confirmation has no request id".to_string())?;
    let item_id = payload_string(&event.payload, &["toolCallId", "tool_call_id"])
        .unwrap_or_else(|| request_id.clone());
    let command = payload_value(&event.payload, &["arguments"])
        .and_then(Value::as_object)
        .and_then(|arguments| arguments.get("command"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|command| !command.is_empty())
        .map(str::to_string);
    let available_decisions = payload_value(
        &event.payload,
        &["availableDecisions", "available_decisions"],
    )
    .and_then(Value::as_array)
    .map(|values| {
        values
            .iter()
            .map(Value::as_str)
            .map(|value| match value {
                Some("allow_once") => Ok(CommandExecutionApprovalDecision::Accept),
                Some("allow_for_session") => Ok(CommandExecutionApprovalDecision::AcceptForSession),
                Some("decline") => Ok(CommandExecutionApprovalDecision::Decline),
                Some("cancel") => Ok(CommandExecutionApprovalDecision::Cancel),
                _ => Err("action.required has unsupported approval decision".to_string()),
            })
            .collect::<Result<Vec<_>, _>>()
    })
    .transpose()?;
    Ok(Some(CommandApprovalRequest {
        params: CommandExecutionRequestApprovalParams {
            thread_id: thread_id.clone(),
            turn_id: turn_id.clone(),
            item_id,
            started_at_ms: timestamp_millis(&event.timestamp)?,
            approval_id: Some(request_id.clone()),
            reason: payload_string(&event.payload, &["prompt", "message"]),
            command,
            cwd: payload_string(&event.payload, &["cwd"]),
            available_decisions,
        },
        identity: RuntimeActionIdentity {
            request_id,
            session_id,
            thread_id,
            turn_id,
        },
    }))
}

fn file_change_approval_request(
    event: &AgentEvent,
) -> Result<Option<FileChangeApprovalRequest>, String> {
    if event.event_type != "action.required"
        || payload_string(&event.payload, &["actionType", "action_type"]).as_deref()
            != Some("tool_confirmation")
        || payload_string(&event.payload, &["toolName", "tool_name"]).as_deref()
            != Some("apply_patch")
    {
        return Ok(None);
    }
    let identity = action_identity(event)?;
    let item_id = required_payload_id(&event.payload, &["toolCallId", "tool_call_id"], "itemId")?;
    Ok(Some(FileChangeApprovalRequest {
        params: FileChangeRequestApprovalParams {
            thread_id: identity.thread_id.clone(),
            turn_id: identity.turn_id.clone(),
            item_id,
            started_at_ms: payload_i64(&event.payload, &["createdAtMs", "created_at_ms"])
                .unwrap_or(timestamp_millis(&event.timestamp)?),
            reason: payload_string(&event.payload, &["prompt", "message"]),
            grant_root: payload_string(&event.payload, &["grantRoot", "grant_root"]),
        },
        identity,
    }))
}

fn user_input_request(event: &AgentEvent) -> Result<Option<UserInputRequest>, String> {
    if event.event_type != "action.required"
        || payload_string(&event.payload, &["actionType", "action_type"]).as_deref()
            != Some("ask_user")
    {
        return Ok(None);
    }
    let identity = action_identity(event)?;
    let item_id = required_payload_id(&event.payload, &["toolCallId", "tool_call_id"], "itemId")?;
    let questions = payload_value(&event.payload, &["questions"])
        .and_then(Value::as_array)
        .ok_or_else(|| "action.required ask_user has no questions".to_string())?
        .iter()
        .map(tool_request_user_input_question)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(UserInputRequest {
        params: ToolRequestUserInputParams {
            thread_id: identity.thread_id.clone(),
            turn_id: identity.turn_id.clone(),
            item_id,
            questions,
            auto_resolution_ms: payload_value(
                &event.payload,
                &["autoResolutionMs", "auto_resolution_ms"],
            )
            .and_then(Value::as_u64),
        },
        identity,
    }))
}

fn action_identity(event: &AgentEvent) -> Result<RuntimeActionIdentity, String> {
    Ok(RuntimeActionIdentity {
        request_id: required_payload_id(
            &event.payload,
            &["requestId", "request_id", "actionId", "action_id"],
            "requestId",
        )?,
        session_id: required_id(&event.session_id, "sessionId")?,
        thread_id: required_optional_id(event.thread_id.as_deref(), "threadId")?,
        turn_id: required_optional_id(event.turn_id.as_deref(), "turnId")?,
    })
}

fn tool_request_user_input_question(value: &Value) -> Result<ToolRequestUserInputQuestion, String> {
    let id = required_payload_id(value, &["id"], "question.id")?;
    let header = required_payload_id(value, &["header"], "question.header")?;
    let question = required_payload_id(value, &["question"], "question.question")?;
    let options = value
        .get("options")
        .and_then(Value::as_array)
        .map(|options| {
            options
                .iter()
                .map(|option| {
                    let label = payload_string(option, &["label", "value"])
                        .ok_or_else(|| "request_user_input option has no label".to_string())?;
                    Ok(ToolRequestUserInputOption {
                        label,
                        description: payload_string(option, &["description"]).unwrap_or_default(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .transpose()?;
    Ok(ToolRequestUserInputQuestion {
        id,
        header,
        question,
        is_other: value
            .get("isOther")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        is_secret: value
            .get("isSecret")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        options,
    })
}

fn approval_decision(decision: CommandExecutionApprovalDecision) -> AgentSessionApprovalDecision {
    match decision {
        CommandExecutionApprovalDecision::Accept => AgentSessionApprovalDecision::AllowOnce,
        CommandExecutionApprovalDecision::AcceptForSession => {
            AgentSessionApprovalDecision::AllowForSession
        }
        CommandExecutionApprovalDecision::Decline => AgentSessionApprovalDecision::Decline,
        CommandExecutionApprovalDecision::Cancel => AgentSessionApprovalDecision::Cancel,
    }
}

fn file_approval_decision(decision: FileChangeApprovalDecision) -> AgentSessionApprovalDecision {
    match decision {
        FileChangeApprovalDecision::Accept => AgentSessionApprovalDecision::AllowOnce,
        FileChangeApprovalDecision::AcceptForSession => {
            AgentSessionApprovalDecision::AllowForSession
        }
        FileChangeApprovalDecision::Decline => AgentSessionApprovalDecision::Decline,
        FileChangeApprovalDecision::Cancel => AgentSessionApprovalDecision::Cancel,
    }
}

fn required_id(value: &str, field: &str) -> Result<String, String> {
    let value = value.trim();
    (!value.is_empty())
        .then(|| value.to_string())
        .ok_or_else(|| format!("action.required has no {field}"))
}

fn required_optional_id(value: Option<&str>, field: &str) -> Result<String, String> {
    value
        .map(|value| required_id(value, field))
        .transpose()?
        .ok_or_else(|| format!("action.required has no {field}"))
}

fn payload_string(payload: &Value, keys: &[&str]) -> Option<String> {
    payload_value(payload, keys).and_then(|value| {
        value
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

fn payload_value<'a>(payload: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| {
        payload
            .get(key)
            .or_else(|| payload.get("data").and_then(|data| data.get(key)))
    })
}

fn payload_i64(payload: &Value, keys: &[&str]) -> Option<i64> {
    payload_value(payload, keys).and_then(Value::as_i64)
}

fn required_payload_id(payload: &Value, keys: &[&str], field: &str) -> Result<String, String> {
    payload_string(payload, keys).ok_or_else(|| format!("action.required has no {field}"))
}

fn timestamp_millis(value: &str) -> Result<i64, String> {
    chrono::DateTime::parse_from_rfc3339(value)
        .map(|value| value.timestamp_millis())
        .map_err(|error| format!("action.required timestamp is invalid: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JsonRpcMessage;
    use serde_json::json;

    fn event(payload: Value) -> AgentEvent {
        AgentEvent {
            event_id: "event-1".to_string(),
            sequence: 1,
            session_id: "session-1".to_string(),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            event_type: "action.required".to_string(),
            timestamp: "2026-07-20T00:00:01Z".to_string(),
            payload,
        }
    }

    #[test]
    fn lowers_tool_confirmation_to_codex_command_approval_request() {
        let request = command_approval_request(&event(json!({
            "requestId": "approval-1",
            "actionType": "tool_confirmation",
            "toolCallId": "item-command-1",
            "prompt": "允许执行测试？",
            "arguments": { "command": "npm test" },
            "availableDecisions": ["allow_once", "allow_for_session", "decline", "cancel"]
        })))
        .expect("valid approval")
        .expect("command approval");

        assert_eq!(request.identity.request_id, "approval-1");
        assert_eq!(request.params.item_id, "item-command-1");
        assert_eq!(request.params.command.as_deref(), Some("npm test"));
        assert_eq!(
            request.params.available_decisions,
            Some(vec![
                CommandExecutionApprovalDecision::Accept,
                CommandExecutionApprovalDecision::AcceptForSession,
                CommandExecutionApprovalDecision::Decline,
                CommandExecutionApprovalDecision::Cancel,
            ])
        );
    }

    #[test]
    fn routes_apply_patch_to_codex_file_change_approval_request() {
        let request = action_server_request(&event(json!({
            "requestId": "approval-file-1",
            "actionType": "tool_confirmation",
            "toolCallId": "item-file-1",
            "toolName": "apply_patch",
            "prompt": "允许修改文件？",
            "createdAtMs": 1784560000000_i64
        })))
        .expect("valid file approval")
        .expect("file approval request");

        let ActionServerRequest::FileChange(request) = request else {
            panic!("apply_patch must use file change approval");
        };
        assert_eq!(request.params.item_id, "item-file-1");
        assert_eq!(request.params.reason.as_deref(), Some("允许修改文件？"));
        assert_eq!(request.params.started_at_ms, 1784560000000);
    }

    #[test]
    fn lowers_nested_ask_user_to_codex_tool_request() {
        let request = action_server_request(&event(json!({
            "requestId": "ask-1",
            "data": {
                "actionType": "ask_user",
                "toolCallId": "item-ask-1",
                "autoResolutionMs": 60000,
                "questions": [{
                    "id": "mode",
                    "header": "模式",
                    "question": "请选择执行模式",
                    "options": [
                        {"value": "auto", "label": "自动", "description": "直接继续"},
                        {"value": "confirm", "label": "确认", "description": "再次确认"}
                    ]
                }]
            }
        })))
        .expect("valid user input")
        .expect("user input request");

        let ActionServerRequest::UserInput(request) = request else {
            panic!("ask_user must use tool request user input");
        };
        assert_eq!(request.params.item_id, "item-ask-1");
        assert_eq!(request.params.auto_resolution_ms, Some(60000));
        assert_eq!(request.params.questions[0].id, "mode");
        assert_eq!(
            request.params.questions[0].options.as_ref().unwrap()[0].label,
            "自动"
        );
    }

    #[test]
    fn leaves_non_command_actions_for_their_typed_server_request_owner() {
        assert!(command_approval_request(&event(json!({
            "requestId": "ask-1",
            "actionType": "ask_user"
        })))
        .expect("not malformed")
        .is_none());
    }

    #[test]
    fn ordinary_approval_request_failure_declines_without_canceling_the_turn() {
        assert_eq!(
            failed_approval_decision(&WaitServerRequestError::Failed(
                "client disconnected".to_string()
            )),
            Some(AgentSessionApprovalDecision::Decline)
        );
        assert_eq!(
            failed_approval_decision(&WaitServerRequestError::Transition),
            None
        );
    }

    #[tokio::test]
    async fn typed_response_publishes_resolved_before_wait_completes() {
        let server = AppServer::new();
        let mut outbound = server.subscribe_outbound_messages();
        let pending = server.server_requests.register(
            METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
            Some(json!({ "threadId": "thread-1" })),
        );
        let request_id = pending.id().clone();
        let waiting_server = server.clone();
        let waiter = tokio::spawn(async move {
            waiting_server
                .finish_server_request::<CommandExecutionRequestApprovalResponse>(
                    "thread-1", pending,
                )
                .await
        });

        server
            .server_requests
            .resolve_response(
                request_id.clone(),
                json!({ "decision": "acceptForSession" }),
            )
            .expect("resolve typed approval response");
        let response = waiter
            .await
            .expect("approval waiter task")
            .expect("typed approval response");
        assert_eq!(
            response.decision,
            CommandExecutionApprovalDecision::AcceptForSession
        );

        let JsonRpcMessage::Notification(notification) = outbound
            .try_recv()
            .expect("resolved notification must be published before waiter returns")
        else {
            panic!("expected resolved notification");
        };
        assert_eq!(
            ServerNotification::try_from(notification).expect("typed resolved notification"),
            ServerNotification::ServerRequestResolved(ServerRequestResolvedNotification {
                thread_id: "thread-1".to_string(),
                request_id
            })
        );
    }

    #[tokio::test]
    async fn transition_cancel_publishes_resolved_without_domain_response() {
        let server = AppServer::new();
        let mut outbound = server.subscribe_outbound_messages();
        let pending = server.server_requests.register(
            METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
            Some(json!({ "threadId": "thread-1" })),
        );
        let request_id = pending.id().clone();
        let waiting_server = server.clone();
        let waiter = tokio::spawn(async move {
            waiting_server
                .finish_server_request::<CommandExecutionRequestApprovalResponse>(
                    "thread-1", pending,
                )
                .await
        });

        server
            .server_requests
            .resolve_error(
                request_id.clone(),
                app_server_protocol::JsonRpcError::new(
                    error_codes::REQUEST_CANCELLED,
                    "turn interrupted",
                ),
            )
            .expect("cancel typed approval request");
        let error = waiter
            .await
            .expect("approval waiter task")
            .expect_err("transition cancellation must stay terminal");
        assert_eq!(error, WaitServerRequestError::Transition);

        let JsonRpcMessage::Notification(notification) = outbound
            .try_recv()
            .expect("transition cancellation must resolve the outer request")
        else {
            panic!("expected resolved notification");
        };
        assert_eq!(
            ServerNotification::try_from(notification).expect("typed resolved notification"),
            ServerNotification::ServerRequestResolved(ServerRequestResolvedNotification {
                thread_id: "thread-1".to_string(),
                request_id
            })
        );
    }
}
