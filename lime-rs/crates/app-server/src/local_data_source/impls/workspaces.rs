use super::super::*;
use async_trait::async_trait;

#[async_trait]
impl WorkspaceAppDataSource for LocalAppDataSource {
    async fn list_workspaces(&self) -> Result<WorkspaceListResponse, RuntimeCoreError> {
        workspaces::list_workspaces(&self.db)
    }

    async fn read_workspace(
        &self,
        params: WorkspaceReadParams,
    ) -> Result<WorkspaceReadResponse, RuntimeCoreError> {
        workspaces::read_workspace(&self.db, params)
    }

    async fn update_workspace(
        &self,
        params: WorkspaceUpdateParams,
    ) -> Result<WorkspaceUpdateResponse, RuntimeCoreError> {
        workspaces::update_workspace(&self.db, params)
    }

    async fn delete_workspace(
        &self,
        params: WorkspaceDeleteParams,
    ) -> Result<WorkspaceDeleteResponse, RuntimeCoreError> {
        workspaces::delete_workspace(&self.db, params)
    }

    async fn read_workspace_by_path(
        &self,
        params: WorkspacePathReadParams,
    ) -> Result<WorkspaceReadResponse, RuntimeCoreError> {
        workspaces::read_workspace_by_path(&self.db, params)
    }

    async fn ensure_project_workspace(
        &self,
        params: WorkspaceEnsureProjectParams,
    ) -> Result<WorkspaceEnsureProjectResponse, RuntimeCoreError> {
        let response = workspaces::ensure_project_workspace(&self.db, params)?;
        trigger_image_task_recovery_for_workspace(response.workspace.get("root_path"), &self.db);
        Ok(response)
    }

    async fn read_default_workspace(&self) -> Result<WorkspaceReadResponse, RuntimeCoreError> {
        workspaces::read_default_workspace(&self.db)
    }

    async fn ensure_default_workspace(&self) -> Result<WorkspaceReadResponse, RuntimeCoreError> {
        let response = workspaces::ensure_default_workspace(&self.db)?;
        if let Some(workspace) = response.workspace.as_ref() {
            trigger_image_task_recovery_for_workspace(workspace.get("root_path"), &self.db);
        }
        Ok(response)
    }

    async fn ensure_workspace_ready(
        &self,
        params: WorkspaceEnsureParams,
    ) -> Result<WorkspaceEnsureReadyResponse, RuntimeCoreError> {
        let response = workspaces::ensure_workspace_ready(&self.db, params)?;
        trigger_image_task_recovery_for_workspace(response.result.get("rootPath"), &self.db);
        Ok(response)
    }

    async fn read_workspace_projects_root(
        &self,
    ) -> Result<WorkspaceProjectsRootReadResponse, RuntimeCoreError> {
        workspaces::read_workspace_projects_root()
    }

    async fn resolve_workspace_project_path(
        &self,
        params: WorkspaceProjectPathResolveParams,
    ) -> Result<WorkspaceProjectPathResolveResponse, RuntimeCoreError> {
        workspaces::resolve_workspace_project_path(params)
    }
}

fn trigger_image_task_recovery_for_workspace(
    root_path: Option<&serde_json::Value>,
    db: &lime_core::database::DbConnection,
) {
    let Some(root_path) = root_path
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return;
    };
    if let Err(error) = crate::media_task_worker::spawn_pending_image_task_workers_for_workspace(
        root_path,
        Some(8),
        crate::media_task_worker::MediaTaskWorkerContext::new(db.clone()),
    ) {
        tracing::warn!(
            workspace_root = %root_path,
            error = %error,
            "failed to recover pending image tasks for workspace"
        );
    }
}
