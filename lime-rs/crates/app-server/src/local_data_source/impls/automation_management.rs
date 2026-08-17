use super::super::*;
use async_trait::async_trait;

#[async_trait]
impl AutomationManagementAppDataSource for LocalAppDataSource {
    async fn read_scheduled_task(
        &self,
        params: ScheduledTaskIdParams,
    ) -> Result<ScheduledTaskReadResponse, RuntimeCoreError> {
        automation::read_scheduled_task(&self.db, params)
    }

    async fn create_scheduled_task(
        &self,
        params: ScheduledTaskCreateParams,
    ) -> Result<ScheduledTaskWriteResponse, RuntimeCoreError> {
        automation::create_scheduled_task(&self.db, params)
    }

    async fn update_scheduled_task(
        &self,
        params: ScheduledTaskUpdateParams,
    ) -> Result<ScheduledTaskWriteResponse, RuntimeCoreError> {
        automation::update_scheduled_task(&self.db, params)
    }

    async fn delete_scheduled_task(
        &self,
        params: ScheduledTaskIdParams,
    ) -> Result<ScheduledTaskDeleteResponse, RuntimeCoreError> {
        automation::delete_scheduled_task(&self.db, params)
    }

    async fn set_scheduled_task_enabled(
        &self,
        params: ScheduledTaskEnabledSetParams,
    ) -> Result<ScheduledTaskWriteResponse, RuntimeCoreError> {
        automation::set_scheduled_task_enabled(&self.db, params)
    }

    async fn list_scheduled_task_runs(
        &self,
        params: ScheduledTaskRunListParams,
    ) -> Result<ScheduledTaskRunListResponse, RuntimeCoreError> {
        automation::list_scheduled_task_runs(&self.db, params)
    }

    async fn preview_scheduled_task_schedule(
        &self,
        params: ScheduledTaskSchedulePreviewParams,
    ) -> Result<ScheduledTaskSchedulePreviewResponse, RuntimeCoreError> {
        automation::preview_scheduled_task_schedule(params)
    }
    async fn start_scheduled_task_run_record(
        &self,
        id: String,
        identity: Option<crate::automation_execution::AutomationRunIdentity>,
    ) -> Result<crate::automation_execution::AutomationRunStart, RuntimeCoreError> {
        automation::start_scheduled_task_run_record(&self.db, id, identity)
    }

    async fn finish_automation_job_run(
        &self,
        finish: crate::automation_execution::AutomationRunFinish,
    ) -> Result<(), RuntimeCoreError> {
        automation::finish_automation_job_run(&self.db, finish).map(|_| ())
    }

    async fn finish_scheduled_task_run_for_terminal_event(
        &self,
        event: app_server_protocol::AgentEvent,
    ) -> Result<Option<app_server_protocol::protocol::v2::ServerNotification>, RuntimeCoreError>
    {
        automation::finish_scheduled_task_run_for_terminal_event(&self.db, &event)
    }

    async fn fail_automation_job_run(
        &self,
        failure: crate::automation_execution::AutomationRunFailure,
    ) -> Result<(), RuntimeCoreError> {
        automation::fail_automation_job_run(&self.db, failure)
    }
}
