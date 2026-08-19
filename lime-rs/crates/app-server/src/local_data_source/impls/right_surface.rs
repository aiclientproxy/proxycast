use super::super::*;
use async_trait::async_trait;

#[async_trait]
impl RightSurfaceAppDataSource for LocalAppDataSource {
    fn workspace_right_surface_pending_persistence_enabled(&self) -> bool {
        true
    }

    async fn save_workspace_right_surface_pending(
        &self,
        request: WorkspaceRightSurfacePendingRequest,
    ) -> Result<(), RuntimeCoreError> {
        right_surface::save_pending_request(&self.db, request)
    }

    async fn list_workspace_right_surface_pending(
        &self,
        params: WorkspaceRightSurfacePendingListParams,
    ) -> Result<Vec<WorkspaceRightSurfacePendingRequest>, RuntimeCoreError> {
        right_surface::list_pending_requests(&self.db, params)
    }

    async fn delete_workspace_right_surface_pending(
        &self,
        request_ids: Vec<String>,
    ) -> Result<Vec<String>, RuntimeCoreError> {
        right_surface::delete_pending_requests(&self.db, request_ids)
    }
}
