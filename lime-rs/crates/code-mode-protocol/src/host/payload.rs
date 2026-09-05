//! Operation payloads exchanged by the host message envelopes.

use crate::{
    RuntimeCodeModeExecuteRequest, RuntimeCodeModeSessionLimits, RuntimeCodeModeTool,
    RuntimeCodeModeWaitRequest,
};
use serde::{Deserialize, Serialize};

pub type WireSessionCellExecutionLimits = RuntimeCodeModeSessionLimits;
pub type WireWaitRequest = RuntimeCodeModeWaitRequest;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct WireExecuteRequest {
    pub tool_call_id: String,
    pub source: String,
    pub enabled_tools: Vec<RuntimeCodeModeTool>,
    pub yield_time_ms: Option<u64>,
    pub max_output_tokens: Option<u64>,
}

impl TryFrom<RuntimeCodeModeExecuteRequest> for WireExecuteRequest {
    type Error = String;

    fn try_from(request: RuntimeCodeModeExecuteRequest) -> Result<Self, Self::Error> {
        Ok(Self {
            tool_call_id: request.tool_call_id,
            source: request.source,
            enabled_tools: request.enabled_tools,
            yield_time_ms: request.yield_time_ms,
            max_output_tokens: request
                .max_output_tokens
                .map(u64::try_from)
                .transpose()
                .map_err(|_| "code mode output token limit exceeds u64".to_string())?,
        })
    }
}

impl TryFrom<WireExecuteRequest> for RuntimeCodeModeExecuteRequest {
    type Error = String;

    fn try_from(request: WireExecuteRequest) -> Result<Self, Self::Error> {
        Ok(Self {
            tool_call_id: request.tool_call_id,
            source: request.source,
            enabled_tools: request.enabled_tools,
            yield_time_ms: request.yield_time_ms,
            max_output_tokens: request
                .max_output_tokens
                .map(usize::try_from)
                .transpose()
                .map_err(|_| "code mode output token limit exceeds usize".to_string())?,
            cancellation_token: None,
        })
    }
}
