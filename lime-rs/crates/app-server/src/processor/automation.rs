//! automation domain handlers for the App Server processor.

use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use crate::scheduled_task_notifications::{changed_notification, run_notification_from_projection};
use app_server_protocol::{JsonRpcError, ScheduledTaskChange, ScheduledTaskRunListParams};

impl RequestProcessor {
    pub(super) async fn handle_scheduled_task_list_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::ScheduledTaskListParams = parse_params(params)?;
        dispatch_result(
            self.runtime
                .list_scheduled_tasks(params)
                .await
                .map_err(to_jsonrpc_error)?,
        )
    }

    pub(super) async fn handle_scheduled_task_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::ScheduledTaskIdParams = parse_params(params)?;
        dispatch_result(
            self.runtime
                .read_scheduled_task(params)
                .await
                .map_err(to_jsonrpc_error)?,
        )
    }

    pub(super) async fn handle_scheduled_task_create_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::ScheduledTaskCreateParams = parse_params(params)?;
        let response = self
            .runtime
            .create_scheduled_task(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_server_notification(changed_notification(
            response.task.id.clone(),
            ScheduledTaskChange::Created,
        ))
        .await;
        dispatch_result(response)
    }

    pub(super) async fn handle_scheduled_task_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::ScheduledTaskUpdateParams = parse_params(params)?;
        let response = self
            .runtime
            .update_scheduled_task(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_server_notification(changed_notification(
            response.task.id.clone(),
            ScheduledTaskChange::Updated,
        ))
        .await;
        dispatch_result(response)
    }

    pub(super) async fn handle_scheduled_task_delete_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::ScheduledTaskIdParams = parse_params(params)?;
        let task_id = params.id.clone();
        let response = self
            .runtime
            .delete_scheduled_task(params)
            .await
            .map_err(to_jsonrpc_error)?;
        if response.deleted {
            self.publish_server_notification(changed_notification(
                task_id,
                ScheduledTaskChange::Deleted,
            ))
            .await;
        }
        dispatch_result(response)
    }

    pub(super) async fn handle_scheduled_task_enabled_set_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::ScheduledTaskEnabledSetParams = parse_params(params)?;
        let response = self
            .runtime
            .set_scheduled_task_enabled(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_server_notification(changed_notification(
            response.task.id.clone(),
            ScheduledTaskChange::Enabled,
        ))
        .await;
        dispatch_result(response)
    }

    pub(super) async fn handle_scheduled_task_run_start_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::ScheduledTaskIdParams = parse_params(params)?;
        let task_id = params.id.clone();
        let task = self
            .runtime
            .read_scheduled_task(params.clone())
            .await
            .ok()
            .and_then(|response| response.task);
        let previous_runs = self
            .runtime
            .list_scheduled_task_runs(ScheduledTaskRunListParams {
                task_id: task_id.clone(),
                limit: Some(1),
            })
            .await
            .ok();
        let previous_run_id = previous_runs
            .as_ref()
            .and_then(|response| response.runs.first())
            .map(|run| run.id.clone());
        let host = self.runtime_host_context();
        let result = self.runtime.start_scheduled_task_run(params, host).await;
        let expected_run_id = result
            .as_ref()
            .ok()
            .map(|response| response.run.id.as_str());
        if let (Some(task), Some(_)) = (task.as_ref(), previous_runs.as_ref()) {
            if let Ok(response) = self
                .runtime
                .list_scheduled_task_runs(ScheduledTaskRunListParams {
                    task_id,
                    limit: Some(1),
                })
                .await
            {
                if let Some(run) = response.runs.first() {
                    let is_new = expected_run_id
                        .map(|expected| expected == run.id.as_str())
                        .unwrap_or_else(|| previous_run_id.as_deref() != Some(run.id.as_str()));
                    if is_new {
                        if let Some(notification) = run_notification_from_projection(task, run) {
                            self.publish_server_notification(notification).await;
                        }
                    }
                }
            }
        }
        dispatch_result(result.map_err(to_jsonrpc_error)?)
    }

    pub(super) async fn handle_scheduled_task_run_list_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::ScheduledTaskRunListParams = parse_params(params)?;
        dispatch_result(
            self.runtime
                .list_scheduled_task_runs(params)
                .await
                .map_err(to_jsonrpc_error)?,
        )
    }

    pub(super) async fn handle_scheduled_task_schedule_preview_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::ScheduledTaskSchedulePreviewParams = parse_params(params)?;
        dispatch_result(
            self.runtime
                .preview_scheduled_task_schedule(params)
                .await
                .map_err(to_jsonrpc_error)?,
        )
    }
}
