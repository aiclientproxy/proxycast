use super::NoopAppDataSource;
use super::RuntimeCoreError;
use app_server_protocol::WorkspaceRightSurfacePendingListParams;
use app_server_protocol::WorkspaceRightSurfacePendingRequest;
use async_trait::async_trait;

#[async_trait]
pub trait RightSurfaceAppDataSource: Send + Sync {
    fn workspace_right_surface_pending_persistence_enabled(&self) -> bool {
        false
    }

    async fn save_workspace_right_surface_pending(
        &self,
        _request: WorkspaceRightSurfacePendingRequest,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn list_workspace_right_surface_pending(
        &self,
        _params: WorkspaceRightSurfacePendingListParams,
    ) -> Result<Vec<WorkspaceRightSurfacePendingRequest>, RuntimeCoreError> {
        Ok(Vec::new())
    }

    async fn delete_workspace_right_surface_pending(
        &self,
        _request_ids: Vec<String>,
    ) -> Result<Vec<String>, RuntimeCoreError> {
        Ok(Vec::new())
    }
}

impl RightSurfaceAppDataSource for NoopAppDataSource {}
