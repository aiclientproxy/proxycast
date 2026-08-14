use crate::execution_process::ExecutionProcessServer;
use async_trait::async_trait;
use tool_runtime::execution_process::{
    live::{
        LiveExecutionOutputBatch, LiveExecutionOutputQuery, LiveExecutionRequest,
        RuntimeLiveExecutionGateway,
    },
    ExecutionProcessSnapshot,
};
use tool_runtime::tool_executor::{RuntimeToolExecutionError, RuntimeToolPolicyErrorKind};

#[async_trait]
impl RuntimeLiveExecutionGateway for ExecutionProcessServer {
    async fn start_process(
        &self,
        thread_id: &str,
        display_command: &str,
        request: LiveExecutionRequest,
    ) -> Result<ExecutionProcessSnapshot, RuntimeToolExecutionError> {
        self.start_thread_process(thread_id, display_command, request)
            .await
            .map_err(runtime_execution_process_error)
    }

    fn write_stdin(&self, process_id: &str, data: &[u8]) -> Result<(), RuntimeToolExecutionError> {
        self.write_stdin(process_id, data)
            .map_err(runtime_execution_process_error)
    }

    fn terminate(
        &self,
        process_id: &str,
    ) -> Result<ExecutionProcessSnapshot, RuntimeToolExecutionError> {
        self.terminate(process_id)
            .map_err(runtime_execution_process_error)
    }

    fn status(
        &self,
        process_id: &str,
    ) -> Result<ExecutionProcessSnapshot, RuntimeToolExecutionError> {
        self.status(process_id)
            .map_err(runtime_execution_process_error)
    }

    fn drain_output(
        &self,
        query: LiveExecutionOutputQuery,
    ) -> Result<LiveExecutionOutputBatch, RuntimeToolExecutionError> {
        self.drain_output(query)
            .map_err(runtime_execution_process_error)
    }
}

fn runtime_execution_process_error(
    error: crate::execution_process::ExecutionProcessError,
) -> RuntimeToolExecutionError {
    use crate::execution_process::ExecutionProcessError;

    let message = error.to_string();
    let kind = match error {
        ExecutionProcessError::Policy(_) => RuntimeToolPolicyErrorKind::SafetyCheckFailed(
            "execution_process_policy_denied".to_string(),
        ),
        ExecutionProcessError::SandboxDenied { reason_code, .. } => {
            RuntimeToolPolicyErrorKind::SandboxDenied(reason_code)
        }
        ExecutionProcessError::ManagedNetworkDenied {
            reason_code, host, ..
        } => RuntimeToolPolicyErrorKind::ManagedNetworkDenied { reason_code, host },
        ExecutionProcessError::Canceled(_) => {
            RuntimeToolPolicyErrorKind::Canceled("execution_process_cancelled".to_string())
        }
        _ => RuntimeToolPolicyErrorKind::ExecutionFailed("execution_process".to_string()),
    };
    RuntimeToolExecutionError::new(message, Some(kind))
}
