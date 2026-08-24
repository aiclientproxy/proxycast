use super::{
    dispatch_result, parse_params, thread::projection::project_thread_read_response,
    thread_goal::goal_updated_notification, to_jsonrpc_error, RequestProcessor, RpcDispatch,
};
use app_server_protocol::protocol::v2::{
    ServerNotification, ThreadForkParams, ThreadForkResponse, ThreadStartedNotification,
};
use app_server_protocol::{JsonRpcError, JsonRpcNotification};
use serde_json::Value;

impl RequestProcessor {
    pub(super) async fn handle_thread_fork_v2(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadForkParams = parse_params(params)?;
        let include_turns = !params.exclude_turns;
        let inherit_goal = params.defer_goal_continuation;
        let canonical = self
            .runtime
            .fork_thread(params)
            .await
            .map_err(to_jsonrpc_error)?;
        let thread = project_thread_read_response(agent_protocol::thread::ThreadReadResponse {
            thread: canonical,
        })?
        .thread;
        let thread_id = thread.id.clone();
        let metadata = thread.extra.clone().unwrap_or(Value::Null);
        let token_usage = include_turns
            .then(|| self.runtime.thread_token_usage_snapshot(&thread_id))
            .flatten();
        let inherited_goal = if inherit_goal {
            match self.runtime.get_thread_goal(&thread_id) {
                Ok(goal) => goal,
                Err(error) => {
                    tracing::warn!(%thread_id, %error, "failed to read inherited thread goal after fork");
                    None
                }
            }
        } else {
            None
        };
        let mut started_thread = thread.clone();
        started_thread.turns.clear();
        let notification: JsonRpcNotification =
            ServerNotification::ThreadStarted(ThreadStartedNotification {
                thread: started_thread,
            })
            .into();
        let response = ThreadForkResponse {
            model: metadata_string(&metadata, "modelName"),
            model_provider: thread.model_provider.clone(),
            service_tier: metadata_optional_string(&metadata, "serviceTier"),
            cwd: thread.cwd.to_string_lossy().to_string(),
            runtime_workspace_roots: metadata_string_array(&metadata, "runtimeWorkspaceRoots"),
            instruction_sources: metadata_string_array(&metadata, "instructionSources"),
            approval_policy: metadata_value(&metadata, "approvalPolicy"),
            approvals_reviewer: metadata_value(&metadata, "approvalsReviewer"),
            sandbox: metadata_value(&metadata, "sandbox"),
            active_permission_profile: metadata.get("activePermissionProfile").cloned(),
            reasoning_effort: metadata_optional_string(&metadata, "reasoningEffort"),
            multi_agent_mode: agent_protocol::MultiAgentMode::default(),
            thread,
        };
        let mut notifications = Vec::with_capacity(3);
        if let Some(snapshot) = token_usage {
            notifications.push(crate::thread_token_usage_notification(&thread_id, snapshot));
        }
        notifications.push(notification);
        if let Some(goal) = inherited_goal {
            notifications.push(goal_updated_notification(&goal));
        }
        Ok(dispatch_result(response)?.with_notifications(notifications))
    }
}

fn metadata_string(metadata: &Value, key: &str) -> String {
    metadata_optional_string(metadata, key).unwrap_or_default()
}

fn metadata_optional_string(metadata: &Value, key: &str) -> Option<String> {
    metadata
        .get(key)
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn metadata_string_array(metadata: &Value, key: &str) -> Vec<String> {
    metadata
        .get(key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToString::to_string)
        .collect()
}

fn metadata_value(metadata: &Value, key: &str) -> Value {
    metadata.get(key).cloned().unwrap_or(Value::Null)
}
