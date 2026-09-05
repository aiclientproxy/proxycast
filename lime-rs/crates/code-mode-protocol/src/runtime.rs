//! Runtime request, response and session primitives.

use crate::{
    CodeModeToolKind, FunctionCallOutputContentItem, RuntimeCodeModeTool,
    DEFAULT_CODE_MODE_EXEC_YIELD_TIME_MS, DEFAULT_CODE_MODE_MAX_OUTPUT_TOKENS,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RuntimeCodeModeCellId(String);

impl RuntimeCodeModeCellId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RuntimeCodeModeCellId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone)]
pub struct RuntimeCodeModeExecuteRequest {
    pub tool_call_id: String,
    pub source: String,
    pub enabled_tools: Vec<RuntimeCodeModeTool>,
    pub yield_time_ms: Option<u64>,
    pub max_output_tokens: Option<usize>,
    pub cancellation_token: Option<CancellationToken>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCodeModeWaitRequest {
    pub cell_id: RuntimeCodeModeCellId,
    pub yield_time_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RuntimeCodeModeResponse {
    Yielded {
        cell_id: RuntimeCodeModeCellId,
        content_items: Vec<FunctionCallOutputContentItem>,
        #[serde(skip_serializing_if = "Option::is_none")]
        code_mode_host_duration: Option<Duration>,
    },
    Terminated {
        cell_id: RuntimeCodeModeCellId,
        content_items: Vec<FunctionCallOutputContentItem>,
        #[serde(skip_serializing_if = "Option::is_none")]
        code_mode_host_duration: Option<Duration>,
    },
    Result {
        cell_id: RuntimeCodeModeCellId,
        content_items: Vec<FunctionCallOutputContentItem>,
        error_text: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        code_mode_host_duration: Option<Duration>,
    },
}

impl RuntimeCodeModeResponse {
    pub fn cell_id(&self) -> &RuntimeCodeModeCellId {
        match self {
            Self::Yielded { cell_id, .. }
            | Self::Terminated { cell_id, .. }
            | Self::Result { cell_id, .. } => cell_id,
        }
    }

    pub fn is_terminal(&self) -> bool {
        !matches!(self, Self::Yielded { .. })
    }

    pub fn code_mode_host_duration(&self) -> Option<Duration> {
        match self {
            Self::Yielded {
                code_mode_host_duration,
                ..
            }
            | Self::Terminated {
                code_mode_host_duration,
                ..
            }
            | Self::Result {
                code_mode_host_duration,
                ..
            } => *code_mode_host_duration,
        }
    }

    pub fn with_code_mode_host_duration(mut self, duration: Duration) -> Self {
        match &mut self {
            Self::Yielded {
                code_mode_host_duration: value,
                ..
            }
            | Self::Terminated {
                code_mode_host_duration: value,
                ..
            }
            | Self::Result {
                code_mode_host_duration: value,
                ..
            } => *value = Some(duration),
        }
        self
    }

    pub fn into_tool_result(self) -> RuntimeCodeModeToolResult {
        self.into_tool_result_with_max_tokens(DEFAULT_CODE_MODE_MAX_OUTPUT_TOKENS)
    }

    pub fn into_tool_result_with_max_tokens(self, max: usize) -> RuntimeCodeModeToolResult {
        let result = match self {
            Self::Yielded {
                cell_id,
                content_items,
                ..
            } => RuntimeCodeModeToolResult {
                output: format!(
                    "Script running with cell ID {cell_id}\nOutput:\n{}",
                    format_content_items(&content_items)
                ),
                cell_id,
                success: true,
                error: None,
            },
            Self::Terminated {
                cell_id,
                content_items,
                ..
            } => RuntimeCodeModeToolResult {
                cell_id,
                success: true,
                output: format!(
                    "Script terminated\nOutput:\n{}",
                    format_content_items(&content_items)
                ),
                error: None,
            },
            Self::Result {
                cell_id,
                content_items,
                error_text,
                ..
            } => {
                let success = error_text.is_none();
                RuntimeCodeModeToolResult {
                    cell_id,
                    success,
                    output: format!(
                        "Script {}\nOutput:\n{}{}",
                        if success { "completed" } else { "failed" },
                        format_content_items(&content_items),
                        error_text
                            .as_deref()
                            .map(|error| format!("\nScript error:\n{error}"))
                            .unwrap_or_default()
                    ),
                    error: error_text,
                }
            }
        };
        let limit = max.max(1).saturating_mul(4);
        if result.output.chars().count() > limit {
            RuntimeCodeModeToolResult {
                output: format!(
                    "{}\n[output truncated]",
                    result.output.chars().take(limit).collect::<String>()
                ),
                ..result
            }
        } else {
            result
        }
    }
}

fn format_content_items(items: &[FunctionCallOutputContentItem]) -> String {
    items
        .iter()
        .map(|item| match item {
            FunctionCallOutputContentItem::InputText { text } => text.clone(),
            FunctionCallOutputContentItem::InputImage { image_url, detail } => serde_json::json!({
                "type": "input_image",
                "image_url": image_url,
                "detail": detail,
            })
            .to_string(),
            FunctionCallOutputContentItem::InputAudio { audio_url } => {
                serde_json::json!({"type": "input_audio", "audio_url": audio_url}).to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeCodeModeToolResult {
    pub cell_id: RuntimeCodeModeCellId,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

impl RuntimeCodeModeToolResult {
    pub fn failure(cell_id: RuntimeCodeModeCellId, error: impl Into<String>) -> Self {
        let error = error.into();
        Self {
            cell_id,
            success: false,
            output: format!("Script failed\nOutput:\n\nScript error:\n{error}"),
            error: Some(error),
        }
    }
}

pub type RuntimeCodeModeFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, String>> + Send + 'a>>;

pub struct RuntimeCodeModeStartedCell {
    pub cell_id: RuntimeCodeModeCellId,
    initial_response: RuntimeCodeModeFuture<'static, RuntimeCodeModeResponse>,
}

impl RuntimeCodeModeStartedCell {
    pub fn new(
        cell_id: RuntimeCodeModeCellId,
        initial_response: RuntimeCodeModeFuture<'static, RuntimeCodeModeResponse>,
    ) -> Self {
        Self {
            cell_id,
            initial_response,
        }
    }

    pub fn from_result_receiver(
        cell_id: RuntimeCodeModeCellId,
        receiver: oneshot::Receiver<Result<RuntimeCodeModeResponse, String>>,
    ) -> Self {
        Self::new(
            cell_id,
            Box::pin(async move {
                receiver
                    .await
                    .map_err(|_| "code mode runtime ended unexpectedly".to_string())?
            }),
        )
    }

    pub async fn initial_response(self) -> Result<RuntimeCodeModeResponse, String> {
        self.initial_response.await
    }
}

impl fmt::Debug for RuntimeCodeModeStartedCell {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeCodeModeStartedCell")
            .field("cell_id", &self.cell_id)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RuntimeCodeModeWaitOutcome {
    LiveCell(RuntimeCodeModeResponse),
    MissingCell(RuntimeCodeModeResponse),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MissingCodeModeHostDuration;

impl fmt::Display for MissingCodeModeHostDuration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("code-mode response is missing host duration")
    }
}

impl Error for MissingCodeModeHostDuration {}

impl RuntimeCodeModeWaitOutcome {
    pub fn into_response(self) -> RuntimeCodeModeResponse {
        match self {
            Self::LiveCell(response) | Self::MissingCell(response) => response,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeCodeModeNestedToolCall {
    pub cell_id: RuntimeCodeModeCellId,
    pub runtime_tool_call_id: String,
    pub tool_name: String,
    #[serde(default)]
    pub kind: CodeModeToolKind,
    pub input: Option<Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCodeModeSessionLimits {
    pub max_yield_time_ms: Option<u64>,
    pub max_heap_size_bytes: Option<usize>,
}

pub type CellId = RuntimeCodeModeCellId;
pub type ExecuteRequest = RuntimeCodeModeExecuteRequest;
pub type WaitRequest = RuntimeCodeModeWaitRequest;
pub type WaitOutcome = RuntimeCodeModeWaitOutcome;
pub type RuntimeResponse = RuntimeCodeModeResponse;
pub type StartedCell = RuntimeCodeModeStartedCell;

pub const DEFAULT_EXEC_YIELD_TIME_MS: u64 = DEFAULT_CODE_MODE_EXEC_YIELD_TIME_MS;
pub const DEFAULT_WAIT_YIELD_TIME_MS: u64 = crate::DEFAULT_CODE_MODE_WAIT_YIELD_TIME_MS;
pub const DEFAULT_MAX_OUTPUT_TOKENS_PER_EXEC_CALL: usize = DEFAULT_CODE_MODE_MAX_OUTPUT_TOKENS;
