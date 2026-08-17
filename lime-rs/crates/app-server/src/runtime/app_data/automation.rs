use super::unavailable;
use super::NoopAppDataSource;
use super::RuntimeCoreError;
use crate::automation_execution::AutomationRunFailure;
use crate::automation_execution::AutomationRunFinish;
use crate::automation_execution::AutomationRunIdentity;
use crate::automation_execution::AutomationRunStart;
use app_server_protocol::*;
use async_trait::async_trait;

#[async_trait]
pub trait AutomationOverviewAppDataSource: Send + Sync {
    async fn list_scheduled_tasks(
        &self,
        _params: ScheduledTaskListParams,
    ) -> Result<ScheduledTaskListResponse, RuntimeCoreError> {
        Err(unavailable("scheduledTask/list"))
    }
}

#[async_trait]
pub trait AutomationManagementAppDataSource: Send + Sync {
    async fn read_scheduled_task(
        &self,
        _params: ScheduledTaskIdParams,
    ) -> Result<ScheduledTaskReadResponse, RuntimeCoreError> {
        Err(unavailable("scheduledTask/read"))
    }

    async fn create_scheduled_task(
        &self,
        _params: ScheduledTaskCreateParams,
    ) -> Result<ScheduledTaskWriteResponse, RuntimeCoreError> {
        Err(unavailable("scheduledTask/create"))
    }

    async fn update_scheduled_task(
        &self,
        _params: ScheduledTaskUpdateParams,
    ) -> Result<ScheduledTaskWriteResponse, RuntimeCoreError> {
        Err(unavailable("scheduledTask/update"))
    }

    async fn delete_scheduled_task(
        &self,
        _params: ScheduledTaskIdParams,
    ) -> Result<ScheduledTaskDeleteResponse, RuntimeCoreError> {
        Err(unavailable("scheduledTask/delete"))
    }

    async fn set_scheduled_task_enabled(
        &self,
        _params: ScheduledTaskEnabledSetParams,
    ) -> Result<ScheduledTaskWriteResponse, RuntimeCoreError> {
        Err(unavailable("scheduledTask/enabled/set"))
    }

    async fn list_scheduled_task_runs(
        &self,
        _params: ScheduledTaskRunListParams,
    ) -> Result<ScheduledTaskRunListResponse, RuntimeCoreError> {
        Err(unavailable("scheduledTask/run/list"))
    }

    async fn preview_scheduled_task_schedule(
        &self,
        _params: ScheduledTaskSchedulePreviewParams,
    ) -> Result<ScheduledTaskSchedulePreviewResponse, RuntimeCoreError> {
        Err(unavailable("scheduledTask/schedule/preview"))
    }
    async fn start_scheduled_task_run_record(
        &self,
        _id: String,
        _identity: Option<AutomationRunIdentity>,
    ) -> Result<AutomationRunStart, RuntimeCoreError> {
        Err(unavailable("scheduledTask/run/start"))
    }

    async fn finish_automation_job_run(
        &self,
        _finish: AutomationRunFinish,
    ) -> Result<(), RuntimeCoreError> {
        Err(unavailable("scheduledTask/run/start"))
    }

    async fn finish_scheduled_task_run_for_terminal_event(
        &self,
        _event: app_server_protocol::AgentEvent,
    ) -> Result<Option<app_server_protocol::protocol::v2::ServerNotification>, RuntimeCoreError>
    {
        Ok(None)
    }

    async fn fail_automation_job_run(
        &self,
        _failure: AutomationRunFailure,
    ) -> Result<(), RuntimeCoreError> {
        Err(unavailable("scheduledTask/run/start"))
    }
}

impl AutomationOverviewAppDataSource for NoopAppDataSource {}
impl AutomationManagementAppDataSource for NoopAppDataSource {}
