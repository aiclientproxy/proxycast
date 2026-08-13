//! automation domain handlers for the App Server processor.

use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::{
    AutomationJobCreateParams, AutomationJobHealthParams, AutomationJobIdParams,
    AutomationJobRunHistoryParams, AutomationJobUpdateParams, AutomationScheduleParams,
    AutomationSchedulerConfigUpdateParams, JsonRpcError,
};

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
        dispatch_result(
            self.runtime
                .create_scheduled_task(params)
                .await
                .map_err(to_jsonrpc_error)?,
        )
    }

    pub(super) async fn handle_scheduled_task_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::ScheduledTaskUpdateParams = parse_params(params)?;
        dispatch_result(
            self.runtime
                .update_scheduled_task(params)
                .await
                .map_err(to_jsonrpc_error)?,
        )
    }

    pub(super) async fn handle_scheduled_task_delete_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::ScheduledTaskIdParams = parse_params(params)?;
        dispatch_result(
            self.runtime
                .delete_scheduled_task(params)
                .await
                .map_err(to_jsonrpc_error)?,
        )
    }

    pub(super) async fn handle_scheduled_task_enabled_set_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::ScheduledTaskEnabledSetParams = parse_params(params)?;
        dispatch_result(
            self.runtime
                .set_scheduled_task_enabled(params)
                .await
                .map_err(to_jsonrpc_error)?,
        )
    }

    pub(super) async fn handle_scheduled_task_run_start_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: app_server_protocol::ScheduledTaskIdParams = parse_params(params)?;
        let host = self.runtime_host_context();
        dispatch_result(
            self.runtime
                .start_scheduled_task_run(params, host)
                .await
                .map_err(to_jsonrpc_error)?,
        )
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

    pub(super) async fn handle_automation_job_list_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_automation_jobs()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_automation_scheduler_config_read_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .read_automation_scheduler_config()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_automation_scheduler_config_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AutomationSchedulerConfigUpdateParams = parse_params(params)?;
        let response = self
            .runtime
            .update_automation_scheduler_config(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_automation_scheduler_status_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .read_automation_scheduler_status()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_automation_job_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AutomationJobIdParams = parse_params(params)?;
        let response = self
            .runtime
            .read_automation_job(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_automation_job_create_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AutomationJobCreateParams = parse_params(params)?;
        let response = self
            .runtime
            .create_automation_job(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_automation_job_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AutomationJobUpdateParams = parse_params(params)?;
        let response = self
            .runtime
            .update_automation_job(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_automation_job_delete_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AutomationJobIdParams = parse_params(params)?;
        let response = self
            .runtime
            .delete_automation_job(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_automation_job_run_now_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AutomationJobIdParams = parse_params(params)?;
        let host = self.runtime_host_context();
        let response = self
            .runtime
            .run_automation_job_now(params, host)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_automation_job_health_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AutomationJobHealthParams = parse_params(params)?;
        let response = self
            .runtime
            .read_automation_health(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_automation_job_run_history_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AutomationJobRunHistoryParams = parse_params(params)?;
        let response = self
            .runtime
            .read_automation_run_history(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_automation_schedule_preview_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AutomationScheduleParams = parse_params(params)?;
        let response = self
            .runtime
            .preview_automation_schedule(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_automation_schedule_validate_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AutomationScheduleParams = parse_params(params)?;
        let response = self
            .runtime
            .validate_automation_schedule(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }
}
