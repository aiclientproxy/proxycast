use app_server_protocol::{
    protocol::v2::GrantedPermissionProfile, ExecutionProcessDrainOutputParams,
    ExecutionProcessDrainOutputResponse, ExecutionProcessEmptyResponse, ExecutionProcessIdParams,
    ExecutionProcessStartParams, ExecutionProcessStartResponse, ExecutionProcessStatusResponse,
    ExecutionProcessWriteStdinParams,
};
use async_trait::async_trait;

/// App Server-owned process control used by the current provider tool loop.
#[async_trait]
pub trait RuntimeLiveExecutionGateway: Send + Sync {
    async fn start_process(
        &self,
        thread_id: &str,
        display_command: &str,
        params: ExecutionProcessStartParams,
        granted_permissions: Option<GrantedPermissionProfile>,
    ) -> Result<ExecutionProcessStartResponse, String>;

    fn write_stdin(
        &self,
        params: ExecutionProcessWriteStdinParams,
    ) -> Result<ExecutionProcessEmptyResponse, String>;

    fn terminate(
        &self,
        params: ExecutionProcessIdParams,
    ) -> Result<ExecutionProcessStatusResponse, String>;

    fn status(
        &self,
        params: ExecutionProcessIdParams,
    ) -> Result<ExecutionProcessStatusResponse, String>;

    fn drain_output(
        &self,
        params: ExecutionProcessDrainOutputParams,
    ) -> Result<ExecutionProcessDrainOutputResponse, String>;
}
