mod cell_actor;
mod protocol;
mod runtime;
mod session_runtime;
mod v8_init;

use super::{
    RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest, RuntimeCodeModeFuture,
    RuntimeCodeModeNestedToolCall, RuntimeCodeModeResponse, RuntimeCodeModeSession,
    RuntimeCodeModeSessionDelegate, RuntimeCodeModeSessionHandle, RuntimeCodeModeSessionLimits,
    RuntimeCodeModeSessionProvider, RuntimeCodeModeSessionProviderFuture,
    RuntimeCodeModeStartedCell, RuntimeCodeModeWaitOutcome, RuntimeCodeModeWaitRequest,
    DEFAULT_CODE_MODE_EXEC_YIELD_TIME_MS,
};
use protocol::FunctionCallOutputContentItem;
use session_runtime::{
    CellEvent, CellId, CreateCellRequest, ImageDetail, NestedToolCall, ObserveMode, OutputItem,
    SessionRuntime, SessionRuntimeDelegate, ToolDefinition, ToolName,
};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

type TaskFailureHandler = Arc<dyn Fn(String) + Send + Sync>;

const YIELD_GRACE_PERIOD: Duration = Duration::from_secs(1);
const MIN_YIELD_TIME_FOR_GRACE: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct V8CodeModeSessionProvider;

impl RuntimeCodeModeSessionProvider for V8CodeModeSessionProvider {
    fn availability(&self) -> Result<(), String> {
        v8_init::ensure_v8_initialized()
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
            v8_init::ensure_v8_initialized()?;
            Ok(RuntimeCodeModeSessionHandle::new(Arc::new(
                V8CodeModeSession::new(delegate, limits),
            )))
        })
    }
}

struct V8CodeModeSession {
    runtime: Arc<SessionRuntime<Delegate>>,
    limits: RuntimeCodeModeSessionLimits,
}

impl V8CodeModeSession {
    fn new(
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
            output: String::new(),
        }
    }
}

impl RuntimeCodeModeSession for V8CodeModeSession {
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
                Err(
                    session_runtime::Error::MissingCell(_) | session_runtime::Error::ClosedCell(_),
                ) => Ok(RuntimeCodeModeWaitOutcome::MissingCell(Self::missing_cell(
                    request.cell_id,
                ))),
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
                Err(
                    session_runtime::Error::MissingCell(_) | session_runtime::Error::ClosedCell(_),
                ) => Ok(RuntimeCodeModeWaitOutcome::MissingCell(Self::missing_cell(
                    cell_id,
                ))),
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
            output: format_output_items(content_items),
        }),
        CellEvent::Completed {
            content_items,
            error_text,
        } => Ok(RuntimeCodeModeResponse::Result {
            cell_id,
            output: format_output_items(content_items),
            error_text,
        }),
        CellEvent::Terminated { content_items } => Ok(RuntimeCodeModeResponse::Terminated {
            cell_id,
            output: format_output_items(content_items),
        }),
        CellEvent::Pending { .. } => {
            Err("cell returned a pending frontier unexpectedly".to_string())
        }
    }
}

fn format_output_items(items: Vec<OutputItem>) -> String {
    items
        .into_iter()
        .map(|item| match item {
            OutputItem::Text { text } => text,
            OutputItem::Image { image_url, detail } => serde_json::json!({
                "type": "input_image",
                "image_url": image_url,
                "detail": detail.map(|detail| match detail {
                    ImageDetail::Auto => "auto",
                    ImageDetail::Low => "low",
                    ImageDetail::High => "high",
                    ImageDetail::Original => "original",
                }),
            })
            .to_string(),
            OutputItem::Audio { audio_url } => serde_json::json!({
                "type": "input_audio",
                "audio_url": audio_url,
            })
            .to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

impl From<FunctionCallOutputContentItem> for OutputItem {
    fn from(item: FunctionCallOutputContentItem) -> Self {
        match item {
            FunctionCallOutputContentItem::InputText { text } => Self::Text { text },
            FunctionCallOutputContentItem::InputImage { image_url, detail } => Self::Image {
                image_url,
                detail: detail.map(|detail| match detail {
                    protocol::ImageDetail::Auto => ImageDetail::Auto,
                    protocol::ImageDetail::Low => ImageDetail::Low,
                    protocol::ImageDetail::High => ImageDetail::High,
                    protocol::ImageDetail::Original => ImageDetail::Original,
                }),
            },
            FunctionCallOutputContentItem::InputAudio { audio_url } => Self::Audio { audio_url },
        }
    }
}

#[cfg(test)]
mod tests;
