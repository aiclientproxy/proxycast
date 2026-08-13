use super::{RuntimeCore, RuntimeCoreError, RuntimeHostContext};
use crate::automation_execution::{
    AutomationRunFailure, AutomationRunFinish, AutomationRunIdentity, AutomationRunStart,
};
use app_server_protocol::*;
use serde_json::Value;

impl RuntimeCore {
    pub async fn list_scheduled_tasks(
        &self,
        params: ScheduledTaskListParams,
    ) -> Result<ScheduledTaskListResponse, RuntimeCoreError> {
        self.app_data_source.list_scheduled_tasks(params).await
    }

    pub async fn read_scheduled_task(
        &self,
        params: ScheduledTaskIdParams,
    ) -> Result<ScheduledTaskReadResponse, RuntimeCoreError> {
        self.app_data_source.read_scheduled_task(params).await
    }

    pub async fn create_scheduled_task(
        &self,
        params: ScheduledTaskCreateParams,
    ) -> Result<ScheduledTaskWriteResponse, RuntimeCoreError> {
        self.app_data_source.create_scheduled_task(params).await
    }

    pub async fn update_scheduled_task(
        &self,
        params: ScheduledTaskUpdateParams,
    ) -> Result<ScheduledTaskWriteResponse, RuntimeCoreError> {
        self.app_data_source.update_scheduled_task(params).await
    }

    pub async fn delete_scheduled_task(
        &self,
        params: ScheduledTaskIdParams,
    ) -> Result<ScheduledTaskDeleteResponse, RuntimeCoreError> {
        self.app_data_source.delete_scheduled_task(params).await
    }

    pub async fn set_scheduled_task_enabled(
        &self,
        params: ScheduledTaskEnabledSetParams,
    ) -> Result<ScheduledTaskWriteResponse, RuntimeCoreError> {
        self.app_data_source
            .set_scheduled_task_enabled(params)
            .await
    }

    pub async fn list_scheduled_task_runs(
        &self,
        params: ScheduledTaskRunListParams,
    ) -> Result<ScheduledTaskRunListResponse, RuntimeCoreError> {
        self.app_data_source.list_scheduled_task_runs(params).await
    }

    pub async fn preview_scheduled_task_schedule(
        &self,
        params: ScheduledTaskSchedulePreviewParams,
    ) -> Result<ScheduledTaskSchedulePreviewResponse, RuntimeCoreError> {
        self.app_data_source
            .preview_scheduled_task_schedule(params)
            .await
    }

    pub async fn start_scheduled_task_run(
        &self,
        params: ScheduledTaskIdParams,
        host: RuntimeHostContext,
    ) -> Result<ScheduledTaskRunStartResponse, RuntimeCoreError> {
        let task = self
            .read_scheduled_task(params.clone())
            .await?
            .task
            .ok_or_else(|| RuntimeCoreError::Backend(format!("已安排任务不存在: {}", params.id)))?;
        let identity = match task.execution.thread_mode {
            ScheduledTaskThreadMode::NewThread => None,
            ScheduledTaskThreadMode::ContinueThread => {
                let thread_id = task
                    .execution
                    .source_thread_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        RuntimeCoreError::InvalidRequest(
                            "continue_thread 必须提供 sourceThreadId".to_string(),
                        )
                    })?;
                let resumed = self
                    .resume_thread(agent_protocol::ThreadId::new(thread_id))
                    .await?;
                Some(AutomationRunIdentity {
                    session_id: resumed.thread.session_id.to_string(),
                    thread_id: resumed.thread.thread_id.to_string(),
                })
            }
        };
        let result = self
            .execute_scheduled_task_now(params.id.clone(), identity, host)
            .await?;
        let value = result.result;
        Ok(ScheduledTaskRunStartResponse {
            run: ScheduledTaskRunSummary {
                id: value
                    .get("run_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                task_id: params.id,
                status: value
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("running")
                    .to_string(),
                scheduled_for: None,
                started_at: value
                    .get("started_at")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                finished_at: value
                    .get("finished_at")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                session_id: value
                    .get("session_id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                thread_id: value
                    .get("thread_id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                turn_id: value
                    .get("turn_id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                summary: None,
                error: None,
            },
        })
    }
    pub(crate) async fn start_automation_job_run(
        &self,
        id: String,
    ) -> Result<AutomationRunStart, RuntimeCoreError> {
        self.app_data_source.start_automation_job_run(id).await
    }

    pub(crate) async fn start_scheduled_task_run_record(
        &self,
        id: String,
        identity: Option<AutomationRunIdentity>,
    ) -> Result<AutomationRunStart, RuntimeCoreError> {
        self.app_data_source
            .start_scheduled_task_run_record(id, identity)
            .await
    }

    pub(crate) async fn finish_automation_job_run(
        &self,
        finish: AutomationRunFinish,
    ) -> Result<(), RuntimeCoreError> {
        self.app_data_source.finish_automation_job_run(finish).await
    }

    pub(crate) async fn fail_automation_job_run(
        &self,
        failure: AutomationRunFailure,
    ) -> Result<(), RuntimeCoreError> {
        self.app_data_source.fail_automation_job_run(failure).await
    }

    pub async fn list_automation_jobs(
        &self,
    ) -> Result<AutomationJobListResponse, RuntimeCoreError> {
        self.app_data_source.list_automation_jobs().await
    }

    pub async fn read_automation_scheduler_config(
        &self,
    ) -> Result<AutomationSchedulerConfigReadResponse, RuntimeCoreError> {
        self.app_data_source
            .read_automation_scheduler_config()
            .await
    }

    pub async fn update_automation_scheduler_config(
        &self,
        params: AutomationSchedulerConfigUpdateParams,
    ) -> Result<AutomationSchedulerConfigUpdateResponse, RuntimeCoreError> {
        self.app_data_source
            .update_automation_scheduler_config(params)
            .await
    }

    pub async fn read_automation_scheduler_status(
        &self,
    ) -> Result<AutomationSchedulerStatusResponse, RuntimeCoreError> {
        self.app_data_source
            .read_automation_scheduler_status()
            .await
    }

    pub async fn read_automation_job(
        &self,
        params: AutomationJobIdParams,
    ) -> Result<AutomationJobReadResponse, RuntimeCoreError> {
        self.app_data_source.read_automation_job(params).await
    }

    pub async fn create_automation_job(
        &self,
        params: AutomationJobCreateParams,
    ) -> Result<AutomationJobWriteResponse, RuntimeCoreError> {
        self.app_data_source.create_automation_job(params).await
    }

    pub async fn update_automation_job(
        &self,
        params: AutomationJobUpdateParams,
    ) -> Result<AutomationJobWriteResponse, RuntimeCoreError> {
        self.app_data_source.update_automation_job(params).await
    }

    pub async fn delete_automation_job(
        &self,
        params: AutomationJobIdParams,
    ) -> Result<AutomationJobDeleteResponse, RuntimeCoreError> {
        self.app_data_source.delete_automation_job(params).await
    }

    pub async fn run_automation_job_now(
        &self,
        params: AutomationJobIdParams,
        host: RuntimeHostContext,
    ) -> Result<AutomationJobRunNowResponse, RuntimeCoreError> {
        self.execute_automation_job_now(params, host).await
    }

    pub async fn read_automation_health(
        &self,
        params: AutomationJobHealthParams,
    ) -> Result<AutomationJobHealthResponse, RuntimeCoreError> {
        self.app_data_source.read_automation_health(params).await
    }

    pub async fn read_automation_run_history(
        &self,
        params: AutomationJobRunHistoryParams,
    ) -> Result<AutomationJobRunHistoryResponse, RuntimeCoreError> {
        self.app_data_source
            .read_automation_run_history(params)
            .await
    }

    pub async fn preview_automation_schedule(
        &self,
        params: AutomationScheduleParams,
    ) -> Result<AutomationSchedulePreviewResponse, RuntimeCoreError> {
        self.app_data_source
            .preview_automation_schedule(params)
            .await
    }

    pub async fn validate_automation_schedule(
        &self,
        params: AutomationScheduleParams,
    ) -> Result<AutomationScheduleValidateResponse, RuntimeCoreError> {
        self.app_data_source
            .validate_automation_schedule(params)
            .await
    }
}
