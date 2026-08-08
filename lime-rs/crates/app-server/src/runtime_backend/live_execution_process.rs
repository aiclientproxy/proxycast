use crate::execution_process::ExecutionProcessServer;
use app_server_protocol::protocol::v2::GrantedPermissionProfile;
use async_trait::async_trait;
use tool_runtime::execution_process::{
    live::{
        LiveExecutionOutputBatch, LiveExecutionOutputQuery, LiveExecutionRequest,
        RuntimeLiveExecutionGateway,
    },
    ExecutionProcessSnapshot,
};

#[async_trait]
impl RuntimeLiveExecutionGateway for ExecutionProcessServer {
    async fn start_process(
        &self,
        thread_id: &str,
        display_command: &str,
        request: LiveExecutionRequest,
        granted_permissions: Option<GrantedPermissionProfile>,
    ) -> Result<ExecutionProcessSnapshot, String> {
        self.start_thread_process_with_permissions(
            thread_id,
            display_command,
            request,
            granted_permissions,
        )
        .await
        .map_err(|error| error.to_string())
    }

    fn write_stdin(&self, process_id: &str, data: &[u8]) -> Result<(), String> {
        self.write_stdin(process_id, data)
            .map_err(|error| error.to_string())
    }

    fn terminate(&self, process_id: &str) -> Result<ExecutionProcessSnapshot, String> {
        self.terminate(process_id)
            .map_err(|error| error.to_string())
    }

    fn status(&self, process_id: &str) -> Result<ExecutionProcessSnapshot, String> {
        self.status(process_id).map_err(|error| error.to_string())
    }

    fn drain_output(
        &self,
        query: LiveExecutionOutputQuery,
    ) -> Result<LiveExecutionOutputBatch, String> {
        self.drain_output(query).map_err(|error| error.to_string())
    }
}
