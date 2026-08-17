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
        let value = result;
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

    pub(crate) async fn finish_scheduled_task_run_for_terminal_event(
        &self,
        event: app_server_protocol::AgentEvent,
    ) -> Result<Option<app_server_protocol::protocol::v2::ServerNotification>, RuntimeCoreError>
    {
        self.app_data_source
            .finish_scheduled_task_run_for_terminal_event(event)
            .await
    }

    pub(crate) async fn fail_automation_job_run(
        &self,
        failure: AutomationRunFailure,
    ) -> Result<(), RuntimeCoreError> {
        self.app_data_source.fail_automation_job_run(failure).await
    }
}
