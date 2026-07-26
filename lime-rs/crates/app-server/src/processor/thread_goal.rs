use super::{
    dispatch_result, parse_params, to_jsonrpc_error, JsonRpcError, RequestProcessor, RpcDispatch,
};
use app_server_protocol::protocol::v2::{
    ServerNotification, ThreadGoal, ThreadGoalClearParams, ThreadGoalClearResponse,
    ThreadGoalClearedNotification, ThreadGoalGetParams, ThreadGoalGetResponse, ThreadGoalSetParams,
    ThreadGoalSetResponse, ThreadGoalUpdatedNotification,
};
use app_server_protocol::{error_codes, JsonRpcNotification};

impl RequestProcessor {
    pub(super) async fn handle_thread_goal_set(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadGoalSetParams = parse_params(params)?;
        self.resolve_persisted_v2_thread_session(&params.thread_id)
            .await?;
        let goal = self
            .runtime
            .set_thread_goal(params)
            .map_err(to_jsonrpc_error)?;
        Ok(
            dispatch_result(ThreadGoalSetResponse { goal: goal.clone() })?
                .with_notification(goal_updated_notification(&goal)),
        )
    }

    pub(super) async fn handle_thread_goal_get(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadGoalGetParams = parse_params(params)?;
        let thread_id = required_thread_id(&params.thread_id)?;
        let goal = self
            .runtime
            .get_thread_goal(&thread_id)
            .map_err(to_jsonrpc_error)?;
        dispatch_result(ThreadGoalGetResponse { goal })
    }

    pub(super) async fn handle_thread_goal_clear(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadGoalClearParams = parse_params(params)?;
        let thread_id = required_thread_id(&params.thread_id)?;
        let cleared = self
            .runtime
            .clear_thread_goal(&thread_id)
            .map_err(to_jsonrpc_error)?;
        let response = dispatch_result(ThreadGoalClearResponse { cleared })?;
        if cleared {
            Ok(response.with_notification(
                ServerNotification::ThreadGoalCleared(ThreadGoalClearedNotification { thread_id })
                    .into(),
            ))
        } else {
            Ok(response)
        }
    }

    pub(crate) fn thread_goal_snapshot_notification(
        &self,
        thread_id: &str,
    ) -> Result<JsonRpcNotification, String> {
        match self.runtime.get_thread_goal(thread_id) {
            Ok(Some(goal)) => Ok(goal_updated_notification(&goal)),
            Ok(None) => Ok(
                ServerNotification::ThreadGoalCleared(ThreadGoalClearedNotification {
                    thread_id: thread_id.to_string(),
                })
                .into(),
            ),
            Err(error) => Err(error.to_string()),
        }
    }
}

pub(super) fn goal_updated_notification(goal: &ThreadGoal) -> JsonRpcNotification {
    ServerNotification::ThreadGoalUpdated(ThreadGoalUpdatedNotification {
        thread_id: goal.thread_id.clone(),
        turn_id: None,
        goal: goal.clone(),
    })
    .into()
}

fn required_thread_id(value: &str) -> Result<String, JsonRpcError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(JsonRpcError::new(
            error_codes::INVALID_PARAMS,
            "threadId must not be empty",
        ));
    }
    Ok(value.to_string())
}
