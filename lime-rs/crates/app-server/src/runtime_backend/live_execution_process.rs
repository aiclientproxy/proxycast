use crate::execution_process::ExecutionProcessServer;
use app_server_protocol::{
    ExecutionProcessDrainOutputParams, ExecutionProcessDrainOutputResponse,
    ExecutionProcessEmptyResponse, ExecutionProcessIdParams, ExecutionProcessStartParams,
    ExecutionProcessStartResponse, ExecutionProcessStatusResponse,
    ExecutionProcessWriteStdinParams,
};
use async_trait::async_trait;
use tool_runtime::execution_process::live::RuntimeLiveExecutionGateway;

#[async_trait]
impl RuntimeLiveExecutionGateway for ExecutionProcessServer {
    async fn start_process(
        &self,
        thread_id: &str,
        display_command: &str,
        params: ExecutionProcessStartParams,
    ) -> Result<ExecutionProcessStartResponse, String> {
        self.start_thread_process(thread_id, display_command, params)
            .await
            .map_err(|error| error.to_string())
    }

    fn write_stdin(
        &self,
        params: ExecutionProcessWriteStdinParams,
    ) -> Result<ExecutionProcessEmptyResponse, String> {
        self.write_stdin(params).map_err(|error| error.to_string())
    }

    fn terminate(
        &self,
        params: ExecutionProcessIdParams,
    ) -> Result<ExecutionProcessStatusResponse, String> {
        self.terminate(params).map_err(|error| error.to_string())
    }

    fn status(
        &self,
        params: ExecutionProcessIdParams,
    ) -> Result<ExecutionProcessStatusResponse, String> {
        self.status(params).map_err(|error| error.to_string())
    }

    fn drain_output(
        &self,
        params: ExecutionProcessDrainOutputParams,
    ) -> Result<ExecutionProcessDrainOutputResponse, String> {
        self.drain_output(params).map_err(|error| error.to_string())
    }
}
