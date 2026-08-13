use super::super::*;
use async_trait::async_trait;

#[async_trait]
impl AutomationOverviewAppDataSource for LocalAppDataSource {
    async fn list_automation_jobs(&self) -> Result<AutomationJobListResponse, RuntimeCoreError> {
        automation::list_automation_jobs(&self.db)
    }

    async fn list_scheduled_tasks(
        &self,
        params: ScheduledTaskListParams,
    ) -> Result<ScheduledTaskListResponse, RuntimeCoreError> {
        automation::list_scheduled_tasks(&self.db, params)
    }
}
