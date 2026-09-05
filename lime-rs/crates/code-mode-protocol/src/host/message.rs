//! Client/host message envelopes.

use super::payload::{WireExecuteRequest, WireSessionCellExecutionLimits, WireWaitRequest};
use crate::{
    RuntimeCodeModeCellId, RuntimeCodeModeNestedToolCall, RuntimeCodeModeResponse,
    RuntimeCodeModeWaitOutcome,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ClientHello {
    pub supported_versions: Vec<u32>,
    pub required_capabilities: Vec<String>,
    pub optional_capabilities: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct HostHello {
    pub selected_version: u32,
    pub capabilities: Vec<String>,
    pub host_pid: u32,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "type", rename_all_fields = "camelCase")]
pub enum ClientToHost {
    #[serde(rename = "connection/hello")]
    ClientHello(ClientHello),
    #[serde(rename = "operation/request")]
    Request { id: u64, request: HostRequest },
    #[serde(rename = "operation/cancel")]
    CancelRequest { id: u64 },
    #[serde(rename = "delegate/response")]
    DelegateResponse {
        id: u64,
        result: WireResult<DelegateResponse>,
    },
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "type", rename_all_fields = "camelCase")]
pub enum HostToClient {
    #[serde(rename = "connection/ready")]
    HostHello(HostHello),
    #[serde(rename = "connection/rejected")]
    HandshakeRejected { reason: String },
    #[serde(rename = "operation/response")]
    Response {
        id: u64,
        result: WireResult<HostResponse>,
    },
    #[serde(rename = "execute/initialResponse")]
    InitialResponse {
        id: u64,
        result: WireResult<RuntimeCodeModeResponse>,
    },
    #[serde(rename = "delegate/request")]
    DelegateRequest {
        id: u64,
        session_id: String,
        request: DelegateRequest,
    },
    #[serde(rename = "delegate/cancel")]
    CancelDelegateRequest { id: u64 },
    #[serde(rename = "cell/closed")]
    CellClosed {
        session_id: String,
        cell_id: RuntimeCodeModeCellId,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "method", rename_all_fields = "camelCase")]
pub enum HostRequest {
    #[serde(rename = "session/open")]
    OpenSession {
        session_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cell_execution_limits: Option<WireSessionCellExecutionLimits>,
    },
    #[serde(rename = "session/execute")]
    Execute {
        session_id: String,
        request: WireExecuteRequest,
    },
    #[serde(rename = "session/wait")]
    Wait {
        session_id: String,
        request: WireWaitRequest,
    },
    #[serde(rename = "session/terminate")]
    Terminate {
        session_id: String,
        cell_id: RuntimeCodeModeCellId,
    },
    #[serde(rename = "session/shutdown")]
    ShutdownSession { session_id: String },
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "type", rename_all_fields = "camelCase")]
pub enum HostResponse {
    #[serde(rename = "session/ready")]
    SessionReady { session_id: String },
    #[serde(rename = "execution/started")]
    ExecutionStarted { cell_id: RuntimeCodeModeCellId },
    #[serde(rename = "wait/completed")]
    WaitCompleted { outcome: RuntimeCodeModeWaitOutcome },
    #[serde(rename = "session/closed")]
    SessionClosed { session_id: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "type", rename_all_fields = "camelCase")]
pub enum DelegateRequest {
    #[serde(rename = "tool/invoke")]
    InvokeTool {
        invocation: RuntimeCodeModeNestedToolCall,
    },
    #[serde(rename = "notification/send")]
    Notify {
        tool_call_id: String,
        cell_id: RuntimeCodeModeCellId,
        text: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "type", rename_all_fields = "camelCase")]
pub enum DelegateResponse {
    #[serde(rename = "tool/result")]
    ToolResult { result: Value },
    #[serde(rename = "notification/delivered")]
    NotificationDelivered,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "status", rename_all_fields = "camelCase")]
pub enum WireResult<T> {
    #[serde(rename = "ok")]
    Ok { value: T },
    #[serde(rename = "error")]
    Err { message: String },
}

impl<T> WireResult<T> {
    pub fn from_result(result: Result<T, String>) -> Self {
        match result {
            Ok(value) => Self::Ok { value },
            Err(message) => Self::Err { message },
        }
    }

    pub fn into_result(self) -> Result<T, String> {
        match self {
            Self::Ok { value } => Ok(value),
            Self::Err { message } => Err(message),
        }
    }
}
