use super::{ExecutionOutputDelta, ExecutionProcessSnapshot};
use crate::execution_orchestrator::RuntimeToolExecutionAttempt;
use crate::tool_executor::RuntimeToolExecutionError;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct LiveExecutionRequest {
    pub process_id: String,
    pub tool_id: String,
    pub tool_name: String,
    pub command: Vec<String>,
    pub working_directory: PathBuf,
    pub tty: bool,
    pub approval_policy: Option<String>,
    pub sandbox_policy: Option<String>,
    pub runtime_metadata: Option<Value>,
    pub env: HashMap<String, String>,
    pub attempt: Option<RuntimeToolExecutionAttempt>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LiveExecutionOutputQuery {
    pub process_id: Option<String>,
    pub after_sequence: Option<u64>,
    pub limit: Option<u16>,
    pub max_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveExecutionOutputBatch {
    pub deltas: Vec<ExecutionOutputDelta>,
    pub next_sequence: Option<u64>,
}

/// App Server-owned process control used by the current provider tool loop.
#[async_trait]
pub trait RuntimeLiveExecutionGateway: Send + Sync {
    async fn start_process(
        &self,
        thread_id: &str,
        display_command: &str,
        request: LiveExecutionRequest,
    ) -> Result<ExecutionProcessSnapshot, RuntimeToolExecutionError>;

    fn write_stdin(&self, process_id: &str, data: &[u8]) -> Result<(), RuntimeToolExecutionError>;

    fn terminate(
        &self,
        process_id: &str,
    ) -> Result<ExecutionProcessSnapshot, RuntimeToolExecutionError>;

    fn status(
        &self,
        process_id: &str,
    ) -> Result<ExecutionProcessSnapshot, RuntimeToolExecutionError>;

    fn drain_output(
        &self,
        query: LiveExecutionOutputQuery,
    ) -> Result<LiveExecutionOutputBatch, RuntimeToolExecutionError>;
}
