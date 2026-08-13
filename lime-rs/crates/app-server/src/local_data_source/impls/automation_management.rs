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
    async fn read_automation_scheduler_config(
        &self,
    ) -> Result<AutomationSchedulerConfigReadResponse, RuntimeCoreError> {
        automation::read_automation_scheduler_config()
    }

    async fn update_automation_scheduler_config(
        &self,
        params: AutomationSchedulerConfigUpdateParams,
    ) -> Result<AutomationSchedulerConfigUpdateResponse, RuntimeCoreError> {
        automation::update_automation_scheduler_config(params)
    }

    async fn read_automation_scheduler_status(
        &self,
    ) -> Result<AutomationSchedulerStatusResponse, RuntimeCoreError> {
        automation::read_automation_scheduler_status()
    }

    async fn read_automation_job(
        &self,
        params: AutomationJobIdParams,
    ) -> Result<AutomationJobReadResponse, RuntimeCoreError> {
        automation::read_automation_job(&self.db, params)
    }

    async fn create_automation_job(
        &self,
        params: AutomationJobCreateParams,
    ) -> Result<AutomationJobWriteResponse, RuntimeCoreError> {
        automation::create_automation_job(&self.db, params)
    }

    async fn update_automation_job(
        &self,
        params: AutomationJobUpdateParams,
    ) -> Result<AutomationJobWriteResponse, RuntimeCoreError> {
        automation::update_automation_job(&self.db, params)
    }

    async fn delete_automation_job(
        &self,
        params: AutomationJobIdParams,
    ) -> Result<AutomationJobDeleteResponse, RuntimeCoreError> {
        automation::delete_automation_job(&self.db, params)
    }

    async fn start_automation_job_run(
        &self,
        id: String,
    ) -> Result<crate::automation_execution::AutomationRunStart, RuntimeCoreError> {
        automation::start_automation_job_run(&self.db, id)
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
        automation::finish_automation_job_run(&self.db, finish)
    }

    async fn fail_automation_job_run(
        &self,
        failure: crate::automation_execution::AutomationRunFailure,
    ) -> Result<(), RuntimeCoreError> {
        automation::fail_automation_job_run(&self.db, failure)
    }

    async fn read_automation_health(
        &self,
        params: AutomationJobHealthParams,
    ) -> Result<AutomationJobHealthResponse, RuntimeCoreError> {
        automation::read_automation_health(&self.db, params)
    }

    async fn read_automation_run_history(
        &self,
        params: AutomationJobRunHistoryParams,
    ) -> Result<AutomationJobRunHistoryResponse, RuntimeCoreError> {
        automation::read_automation_run_history(&self.db, params)
    }

    async fn preview_automation_schedule(
        &self,
        params: AutomationScheduleParams,
    ) -> Result<AutomationSchedulePreviewResponse, RuntimeCoreError> {
        automation::preview_automation_schedule(params)
    }

    async fn validate_automation_schedule(
        &self,
        params: AutomationScheduleParams,
    ) -> Result<AutomationScheduleValidateResponse, RuntimeCoreError> {
        automation::validate_automation_schedule(params)
    }
}
