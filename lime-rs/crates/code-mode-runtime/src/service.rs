use crate::session_runtime::Error as SessionRuntimeError;
use crate::session_runtime::{
    CellEvent, CellId, CreateCellRequest, ImageDetail, NestedToolCall, ObserveMode, OutputItem,
    SessionRuntime, SessionRuntimeDelegate, ToolDefinition, ToolKind, ToolName,
};
use code_mode_protocol::{FunctionCallOutputContentItem, ImageDetail as ProtocolImageDetail};
use code_mode_protocol::{
    RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest, RuntimeCodeModeFuture,
    RuntimeCodeModeNestedToolCall, RuntimeCodeModeResponse, RuntimeCodeModeSession,
    RuntimeCodeModeSessionDelegate, RuntimeCodeModeSessionHandle, RuntimeCodeModeSessionLimits,
    RuntimeCodeModeSessionProvider, RuntimeCodeModeSessionProviderFuture,
    RuntimeCodeModeStartedCell, RuntimeCodeModeWaitOutcome, RuntimeCodeModeWaitRequest,
    DEFAULT_CODE_MODE_EXEC_YIELD_TIME_MS,
};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

const YIELD_GRACE_PERIOD: Duration = Duration::from_secs(1);
const MIN_YIELD_TIME_FOR_GRACE: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug, Default)]
pub struct V8CodeModeSessionProvider;

impl RuntimeCodeModeSessionProvider for V8CodeModeSessionProvider {
    fn availability(&self) -> Result<(), String> {
        crate::v8_init::ensure_v8_initialized()
    }

    fn create_session<'a>(
        &'a self,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    ) -> RuntimeCodeModeSessionProviderFuture<'a> {
        self.create_session_with_limits(delegate, RuntimeCodeModeSessionLimits::default())
    }

    fn create_session_with_limits<'a>(
        &'a self,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
        limits: RuntimeCodeModeSessionLimits,
    ) -> RuntimeCodeModeSessionProviderFuture<'a> {
        Box::pin(async move {
            if limits.max_heap_size_bytes.is_some() {
                return Err(
                    "in-process V8 code mode does not support a per-session heap limit".to_string(),
                );
            }
            crate::v8_init::ensure_v8_initialized()?;
            Ok(RuntimeCodeModeSessionHandle::new(Arc::new(
                InProcessCodeModeSession::with_delegate_and_limits(delegate, limits),
            )))
        })
    }
}

/// In-process sandbox-enabled V8 session. Cells share state for the lifetime
/// of this session and are isolated from other sessions.
pub struct InProcessCodeModeSession {
    runtime: Arc<SessionRuntime<Delegate>>,
    limits: RuntimeCodeModeSessionLimits,
}

impl InProcessCodeModeSession {
    pub fn new() -> Self {
        Self::with_delegate(Arc::new(
            code_mode_protocol::NoopRuntimeCodeModeSessionDelegate,
        ))
    }

    pub fn with_delegate(delegate: Arc<dyn RuntimeCodeModeSessionDelegate>) -> Self {
        Self::with_delegate_and_limits(delegate, RuntimeCodeModeSessionLimits::default())
    }

    pub fn with_delegate_and_limits(
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
        limits: RuntimeCodeModeSessionLimits,
    ) -> Self {
        Self {
            runtime: Arc::new(SessionRuntime::new(Arc::new(Delegate { delegate }))),
            limits,
        }
    }

    fn resolve_yield_timeout(&self, yield_time_ms: u64) -> Duration {
        let yield_time = Duration::from_millis(yield_time_ms);
        let timeout = if yield_time >= MIN_YIELD_TIME_FOR_GRACE {
            yield_time.saturating_add(YIELD_GRACE_PERIOD)
        } else {
            yield_time
        };
        self.limits
            .max_yield_time_ms
            .map(Duration::from_millis)
            .map_or(timeout, |limit| timeout.min(limit))
    }

    fn missing_cell(cell_id: RuntimeCodeModeCellId) -> RuntimeCodeModeResponse {
        RuntimeCodeModeResponse::Result {
            error_text: Some(format!("exec cell {cell_id} not found")),
            cell_id,
            content_items: Vec::new(),
            code_mode_host_duration: None,
        }
    }
}

impl Default for InProcessCodeModeSession {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeCodeModeSession for InProcessCodeModeSession {
    fn execute<'a>(
        &'a self,
        request: RuntimeCodeModeExecuteRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeStartedCell> {
        Box::pin(async move {
            if request
                .cancellation_token
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
            {
                return Err("code mode execution cancelled".to_string());
            }
            let yield_time_ms = request
                .yield_time_ms
                .unwrap_or(DEFAULT_CODE_MODE_EXEC_YIELD_TIME_MS);
            let started = self
                .runtime
                .execute(
                    create_cell_request(request),
                    ObserveMode::YieldAfter(self.resolve_yield_timeout(yield_time_ms)),
                )
                .await
                .map_err(|error| error.to_string())?;
            let runtime_cell_id = started.cell_id.clone();
            let cell_id = public_cell_id(&runtime_cell_id);
            let response_cell_id = cell_id.clone();
            Ok(RuntimeCodeModeStartedCell::new(
                cell_id,
                Box::pin(async move {
                    started
                        .initial_event()
                        .await
                        .map_err(|error| error.to_string())
                        .and_then(|event| runtime_response(response_cell_id, event))
                }),
            ))
        })
    }

    fn wait<'a>(
        &'a self,
        request: RuntimeCodeModeWaitRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            let runtime_cell_id = runtime_cell_id(&request.cell_id);
            match self
                .runtime
                .observe(
                    &runtime_cell_id,
                    ObserveMode::YieldAfter(self.resolve_yield_timeout(request.yield_time_ms)),
                )
                .await
            {
                Ok(event) => Ok(RuntimeCodeModeWaitOutcome::LiveCell(runtime_response(
                    request.cell_id,
                    event,
                )?)),
                Err(SessionRuntimeError::MissingCell(_) | SessionRuntimeError::ClosedCell(_)) => {
                    Ok(RuntimeCodeModeWaitOutcome::MissingCell(Self::missing_cell(
                        request.cell_id,
                    )))
                }
                Err(error) => Err(error.to_string()),
            }
        })
    }

    fn terminate<'a>(
        &'a self,
        cell_id: RuntimeCodeModeCellId,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            match self.runtime.terminate(&runtime_cell_id(&cell_id)).await {
                Ok(event) => Ok(RuntimeCodeModeWaitOutcome::LiveCell(runtime_response(
                    cell_id, event,
                )?)),
                Err(SessionRuntimeError::MissingCell(_) | SessionRuntimeError::ClosedCell(_)) => {
                    Ok(RuntimeCodeModeWaitOutcome::MissingCell(Self::missing_cell(
                        cell_id,
                    )))
                }
                Err(error) => Err(error.to_string()),
            }
        })
    }

    fn shutdown(&self) -> RuntimeCodeModeFuture<'_, ()> {
        Box::pin(async move {
            self.runtime
                .shutdown()
                .await
                .map_err(|error| error.to_string())
        })
    }
}

struct Delegate {
    delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
}

impl SessionRuntimeDelegate for Delegate {
    fn invoke_tool(
        &self,
        invocation: NestedToolCall,
        cancellation_token: CancellationToken,
    ) -> impl Future<Output = Result<serde_json::Value, String>> + Send {
        self.delegate.invoke_tool(
            RuntimeCodeModeNestedToolCall {
                cell_id: public_cell_id(&invocation.cell_id),
                runtime_tool_call_id: invocation.runtime_tool_call_id,
                tool_name: invocation.tool_name.name,
                kind: match invocation.tool_kind {
                    ToolKind::Function => code_mode_protocol::CodeModeToolKind::Function,
                    ToolKind::Freeform => code_mode_protocol::CodeModeToolKind::Freeform,
                },
                input: invocation.input,
            },
            cancellation_token,
        )
    }

    fn notify(
        &self,
        call_id: String,
        cell_id: CellId,
        text: String,
        cancellation_token: CancellationToken,
    ) -> impl Future<Output = Result<(), String>> + Send {
        self.delegate
            .notify(call_id, public_cell_id(&cell_id), text, cancellation_token)
    }

    fn cell_closed(&self, cell_id: &CellId) {
        self.delegate.cell_closed(&public_cell_id(cell_id));
    }
}

fn create_cell_request(request: RuntimeCodeModeExecuteRequest) -> CreateCellRequest {
    CreateCellRequest {
        tool_call_id: request.tool_call_id,
        enabled_tools: request
            .enabled_tools
            .into_iter()
            .map(|tool| ToolDefinition {
                name: tool.global_name.clone(),
                tool_name: ToolName {
                    name: tool.global_name,
                    namespace: None,
                },
                description: tool.definition.description,
                kind: match tool.kind {
                    code_mode_protocol::CodeModeToolKind::Function => ToolKind::Function,
                    code_mode_protocol::CodeModeToolKind::Freeform => ToolKind::Freeform,
                },
            })
            .collect(),
        source: request.source,
    }
}

fn runtime_cell_id(cell_id: &RuntimeCodeModeCellId) -> CellId {
    CellId::new(cell_id.as_str().to_string())
}

fn public_cell_id(cell_id: &CellId) -> RuntimeCodeModeCellId {
    RuntimeCodeModeCellId::new(cell_id.as_str())
}

fn runtime_response(
    cell_id: RuntimeCodeModeCellId,
    event: CellEvent,
) -> Result<RuntimeCodeModeResponse, String> {
    match event {
        CellEvent::Yielded { content_items } => Ok(RuntimeCodeModeResponse::Yielded {
            cell_id,
            content_items: content_items.into_iter().map(output_item).collect(),
            code_mode_host_duration: None,
        }),
        CellEvent::Completed {
            content_items,
            error_text,
        } => Ok(RuntimeCodeModeResponse::Result {
            cell_id,
            content_items: content_items.into_iter().map(output_item).collect(),
            error_text,
            code_mode_host_duration: None,
        }),
        CellEvent::Terminated { content_items } => Ok(RuntimeCodeModeResponse::Terminated {
            cell_id,
            content_items: content_items.into_iter().map(output_item).collect(),
            code_mode_host_duration: None,
        }),
        CellEvent::Pending { .. } => {
            Err("cell returned a pending frontier unexpectedly".to_string())
        }
    }
}

impl From<FunctionCallOutputContentItem> for OutputItem {
    fn from(item: FunctionCallOutputContentItem) -> Self {
        match item {
            FunctionCallOutputContentItem::InputText { text } => Self::Text { text },
            FunctionCallOutputContentItem::InputImage { image_url, detail } => Self::Image {
                image_url,
                detail: detail.map(|detail| match detail {
                    ProtocolImageDetail::Auto => ImageDetail::Auto,
                    ProtocolImageDetail::Low => ImageDetail::Low,
                    ProtocolImageDetail::High => ImageDetail::High,
                    ProtocolImageDetail::Original => ImageDetail::Original,
                }),
            },
            FunctionCallOutputContentItem::InputAudio { audio_url } => Self::Audio { audio_url },
        }
    }
}

fn output_item(item: OutputItem) -> FunctionCallOutputContentItem {
    match item {
        OutputItem::Text { text } => FunctionCallOutputContentItem::InputText { text },
        OutputItem::Image { image_url, detail } => FunctionCallOutputContentItem::InputImage {
            image_url,
            detail: detail.map(|detail| match detail {
                ImageDetail::Auto => ProtocolImageDetail::Auto,
                ImageDetail::Low => ProtocolImageDetail::Low,
                ImageDetail::High => ProtocolImageDetail::High,
                ImageDetail::Original => ProtocolImageDetail::Original,
            }),
        },
        OutputItem::Audio { audio_url } => FunctionCallOutputContentItem::InputAudio { audio_url },
    }
}

#[cfg(test)]
#[path = "service_audio_tests.rs"]
mod audio_tests;
#[cfg(test)]
#[path = "service_contract_tests.rs"]
mod contract_tests;
#[cfg(test)]
#[path = "service_tests.rs"]
mod tests;
