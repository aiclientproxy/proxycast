use super::super::{
    RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest, RuntimeCodeModeNestedToolCall,
    RuntimeCodeModeResponse, RuntimeCodeModeSessionLimits, RuntimeCodeModeTool,
    RuntimeCodeModeWaitOutcome, RuntimeCodeModeWaitRequest,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::io;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_IN_FLIGHT_REQUESTS: usize = 1_024;
pub const MAX_PENDING_DELEGATE_CALLS: usize = 1_024;
pub const SESSION_LIMITS_CAPABILITY: &str = "session-cell-execution-resource-limits";

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
        cell_execution_limits: Option<RuntimeCodeModeSessionLimits>,
    },
    #[serde(rename = "session/execute")]
    Execute {
        session_id: String,
        request: WireExecuteRequest,
    },
    #[serde(rename = "session/wait")]
    Wait {
        session_id: String,
        request: RuntimeCodeModeWaitRequest,
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

pub struct FramedReader<R> {
    reader: R,
}

impl<R: AsyncRead + Unpin> FramedReader<R> {
    pub fn new(reader: R) -> Self {
        Self { reader }
    }

    pub async fn read<T: DeserializeOwned>(&mut self) -> io::Result<Option<T>> {
        let mut length_bytes = [0_u8; 4];
        if self.reader.read(&mut length_bytes[..1]).await? == 0 {
            return Ok(None);
        }
        self.reader.read_exact(&mut length_bytes[1..]).await?;
        let length = u32::from_le_bytes(length_bytes) as usize;
        if length > MAX_FRAME_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("code-mode IPC frame length {length} exceeds {MAX_FRAME_BYTES} bytes"),
            ));
        }
        let mut payload = vec![0_u8; length];
        self.reader.read_exact(&mut payload).await?;
        serde_json::from_slice(&payload).map(Some).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to decode code-mode IPC frame: {error}"),
            )
        })
    }
}

pub struct FramedWriter<W> {
    writer: W,
}

impl<W: AsyncWrite + Unpin> FramedWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub async fn write<T: Serialize>(&mut self, message: &T) -> io::Result<()> {
        let payload = serde_json::to_vec(message).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to encode code-mode IPC frame: {error}"),
            )
        })?;
        if payload.len() > MAX_FRAME_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "code-mode IPC frame length {} exceeds {MAX_FRAME_BYTES} bytes",
                    payload.len()
                ),
            ));
        }
        let length = u32::try_from(payload.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "code-mode IPC frame exceeds u32",
            )
        })?;
        self.writer.write_all(&length.to_le_bytes()).await?;
        self.writer.write_all(&payload).await?;
        self.writer.flush().await
    }
}
