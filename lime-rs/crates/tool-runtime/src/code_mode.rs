use crate::tool_definition::RuntimeToolDefinition;
use crate::turn_snapshot::{RuntimeToolIdentity, RuntimeToolSnapshot};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

mod process;
mod v8;

pub use process::{default_code_mode_host_path, ProcessCodeModeSessionProvider};

#[doc(hidden)]
pub async fn run_code_mode_host_stdio() -> Result<(), String> {
    process::run_stdio().await
}

pub const CODE_MODE_EXEC_TOOL_NAME: &str = "exec";
pub const CODE_MODE_WAIT_TOOL_NAME: &str = "wait";
pub const DEFAULT_CODE_MODE_EXEC_YIELD_TIME_MS: u64 = 10_000;
pub const DEFAULT_CODE_MODE_WAIT_YIELD_TIME_MS: u64 = 10_000;
pub const DEFAULT_CODE_MODE_MAX_OUTPUT_TOKENS: usize = 10_000;
pub const CODE_MODE_EXEC_PRAGMA_PREFIX: &str = "// @exec:";
const MAX_JS_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
pub const CODE_MODE_EXEC_FREEFORM_GRAMMAR: &str = r#"
start: pragma_source | plain_source
pragma_source: PRAGMA_LINE NEWLINE SOURCE
plain_source: SOURCE

PRAGMA_LINE: /[ \t]*\/\/ @exec:[^\r\n]*/
NEWLINE: /\r?\n/
SOURCE: /[\s\S]+/
"#;

pub fn code_mode_exec_tool_description(tools: &[RuntimeCodeModeTool]) -> String {
    let mut description = format!(
        "Run JavaScript code to orchestrate tool calls. Each cell evaluates as an async module in a fresh sandbox-enabled V8 isolate; cells in the same thread share values written with store/load. There is no Node.js, file system, network, or console access. The input is raw JavaScript source, not JSON. Long-running cells yield after {} ms and return a cell ID. Nested tools are asynchronous and must be awaited.\n\nGlobal helpers: text(value), image(value, detail?), audio(value), generatedImage(result), store(key, value), load(key), notify(value), setTimeout(callback, delayMs?), clearTimeout(id?), yield_control(), and exit().",
        DEFAULT_CODE_MODE_EXEC_YIELD_TIME_MS
    );
    if !tools.is_empty() {
        description.push_str("\n\nAvailable nested tools:\n");
        for tool in tools {
            description.push_str(&format!(
                "- `{}`: {}\n",
                tool.global_name, tool.definition.description
            ));
        }
    }
    description
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeCodeModeParsedExecSource {
    pub code: String,
    pub yield_time_ms: Option<u64>,
    pub max_output_tokens: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeCodeModeExecPragma {
    #[serde(default)]
    yield_time_ms: Option<u64>,
    #[serde(default)]
    max_output_tokens: Option<usize>,
}

pub fn parse_code_mode_exec_source(input: &str) -> Result<RuntimeCodeModeParsedExecSource, String> {
    if input.trim().is_empty() {
        return Err("exec expects non-empty raw JavaScript source text".to_string());
    }
    let mut parsed = RuntimeCodeModeParsedExecSource {
        code: input.to_string(),
        yield_time_ms: None,
        max_output_tokens: None,
    };
    let mut lines = input.splitn(2, '\n');
    let first_line = lines.next().unwrap_or_default();
    let rest = lines.next().unwrap_or_default();
    let Some(directive) = first_line
        .trim_start()
        .strip_prefix(CODE_MODE_EXEC_PRAGMA_PREFIX)
    else {
        return Ok(parsed);
    };
    if rest.trim().is_empty() {
        return Err("exec pragma must be followed by JavaScript source".to_string());
    }
    let pragma: RuntimeCodeModeExecPragma = serde_json::from_str(directive.trim()).map_err(|error| {
        format!(
            "exec pragma must be a JSON object containing only `yield_time_ms` and `max_output_tokens`: {error}"
        )
    })?;
    if pragma
        .yield_time_ms
        .is_some_and(|value| value > MAX_JS_SAFE_INTEGER)
    {
        return Err("exec pragma `yield_time_ms` must be a JavaScript safe integer".to_string());
    }
    if pragma.max_output_tokens.is_some_and(|value| {
        u64::try_from(value)
            .map(|value| value > MAX_JS_SAFE_INTEGER)
            .unwrap_or(true)
    }) {
        return Err(
            "exec pragma `max_output_tokens` must be a JavaScript safe integer".to_string(),
        );
    }
    parsed.code = rest.to_string();
    parsed.yield_time_ms = pragma.yield_time_ms;
    parsed.max_output_tokens = pragma.max_output_tokens;
    Ok(parsed)
}

pub fn code_mode_wait_tool_definition() -> RuntimeToolDefinition {
    RuntimeToolDefinition::new(
        CODE_MODE_WAIT_TOOL_NAME,
        "Waits on a yielded exec cell and returns new output or completion.",
        serde_json::json!({
            "type": "object",
            "required": ["cell_id"],
            "additionalProperties": false,
            "properties": {
                "cell_id": {
                    "type": "string",
                    "description": "Identifier of the running exec cell."
                },
                "yield_time_ms": {
                    "type": "number",
                    "description": "Wait before yielding more output. Defaults to 10000 ms."
                },
                "max_tokens": {
                    "type": "number",
                    "description": "Output token budget for this wait call. Defaults to 10000 tokens."
                },
                "terminate": {
                    "type": "boolean",
                    "description": "True stops the running exec cell; false or omitted waits for output."
                }
            }
        }),
    )
}

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeCodeModeResponse {
    Yielded {
        cell_id: RuntimeCodeModeCellId,
        output: String,
    },
    Terminated {
        cell_id: RuntimeCodeModeCellId,
        output: String,
    },
    Result {
        cell_id: RuntimeCodeModeCellId,
        output: String,
        error_text: Option<String>,
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

    pub fn into_tool_result(self) -> RuntimeCodeModeToolResult {
        self.into_tool_result_with_max_tokens(DEFAULT_CODE_MODE_MAX_OUTPUT_TOKENS)
    }

    pub fn into_tool_result_with_max_tokens(
        self,
        max_output_tokens: usize,
    ) -> RuntimeCodeModeToolResult {
        let mut result = match self {
            Self::Yielded { cell_id, output } => RuntimeCodeModeToolResult {
                output: format_code_mode_output(
                    &format!("Script running with cell ID {cell_id}"),
                    &output,
                    None,
                ),
                cell_id,
                success: true,
                error: None,
            },
            Self::Terminated { cell_id, output } => RuntimeCodeModeToolResult {
                output: format_code_mode_output("Script terminated", &output, None),
                cell_id,
                success: true,
                error: None,
            },
            Self::Result {
                cell_id,
                output,
                error_text,
            } => {
                let success = error_text.is_none();
                let status = if success {
                    "Script completed"
                } else {
                    "Script failed"
                };
                RuntimeCodeModeToolResult {
                    cell_id,
                    success,
                    output: format_code_mode_output(status, &output, error_text.as_deref()),
                    error: error_text,
                }
            }
        };
        result.output = crate::tool_io::format_tool_output_for_model(
            &result.output,
            crate::tool_io::ToolOutputTruncationPolicy::Tokens(max_output_tokens.max(1)),
        );
        result
    }
}

fn format_code_mode_output(status: &str, output: &str, error_text: Option<&str>) -> String {
    let mut formatted = format!("{status}\nOutput:\n{output}");
    if let Some(error_text) = error_text {
        formatted.push_str("\nScript error:\n");
        formatted.push_str(error_text);
    }
    formatted
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
            output: format_code_mode_output("Script failed", "", Some(&error)),
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
        initial_response: oneshot::Receiver<Result<RuntimeCodeModeResponse, String>>,
    ) -> Self {
        Self::new(
            cell_id,
            Box::pin(async move {
                initial_response
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeCodeModeWaitOutcome {
    LiveCell(RuntimeCodeModeResponse),
    MissingCell(RuntimeCodeModeResponse),
}

impl RuntimeCodeModeWaitOutcome {
    pub fn into_response(self) -> RuntimeCodeModeResponse {
        match self {
            Self::LiveCell(response) | Self::MissingCell(response) => response,
        }
    }
}

pub trait RuntimeCodeModeSession: Send + Sync {
    fn execute<'a>(
        &'a self,
        request: RuntimeCodeModeExecuteRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeStartedCell>;

    /// Binds the nested-tool delegate for one cell without changing the session-wide host delegate.
    /// Providers that do not support nested dispatch keep the default execute-only behavior.
    fn execute_with_delegate<'a>(
        &'a self,
        request: RuntimeCodeModeExecuteRequest,
        _delegate: Option<Arc<dyn RuntimeCodeModeSessionDelegate>>,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeStartedCell> {
        self.execute(request)
    }

    fn wait<'a>(
        &'a self,
        request: RuntimeCodeModeWaitRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome>;

    fn terminate<'a>(
        &'a self,
        cell_id: RuntimeCodeModeCellId,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome>;

    fn shutdown(&self) -> RuntimeCodeModeFuture<'_, ()>;
}

#[derive(Clone)]
pub struct RuntimeCodeModeSessionHandle(Arc<dyn RuntimeCodeModeSession>);

impl RuntimeCodeModeSessionHandle {
    pub fn new(session: Arc<dyn RuntimeCodeModeSession>) -> Self {
        Self(session)
    }

    pub async fn execute(
        &self,
        request: RuntimeCodeModeExecuteRequest,
    ) -> Result<RuntimeCodeModeStartedCell, String> {
        self.0.execute(request).await
    }

    pub async fn execute_with_delegate(
        &self,
        request: RuntimeCodeModeExecuteRequest,
        delegate: Option<Arc<dyn RuntimeCodeModeSessionDelegate>>,
    ) -> Result<RuntimeCodeModeStartedCell, String> {
        self.0.execute_with_delegate(request, delegate).await
    }

    pub async fn wait(
        &self,
        request: RuntimeCodeModeWaitRequest,
    ) -> Result<RuntimeCodeModeWaitOutcome, String> {
        self.0.wait(request).await
    }

    pub async fn terminate(
        &self,
        cell_id: RuntimeCodeModeCellId,
    ) -> Result<RuntimeCodeModeWaitOutcome, String> {
        self.0.terminate(cell_id).await
    }

    pub async fn shutdown(&self) -> Result<(), String> {
        self.0.shutdown().await
    }
}

impl fmt::Debug for RuntimeCodeModeSessionHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RuntimeCodeModeSessionHandle(<runtime>)")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeCodeModeNestedToolCall {
    pub cell_id: RuntimeCodeModeCellId,
    pub runtime_tool_call_id: String,
    pub tool_name: String,
    pub input: Option<Value>,
}

pub trait RuntimeCodeModeSessionDelegate: Send + Sync {
    fn invoke_tool<'a>(
        &'a self,
        invocation: RuntimeCodeModeNestedToolCall,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, Value>;

    fn notify<'a>(
        &'a self,
        tool_call_id: String,
        cell_id: RuntimeCodeModeCellId,
        text: String,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, ()>;

    fn cell_closed(&self, cell_id: &RuntimeCodeModeCellId);
}

pub struct NoopRuntimeCodeModeSessionDelegate;

impl RuntimeCodeModeSessionDelegate for NoopRuntimeCodeModeSessionDelegate {
    fn invoke_tool<'a>(
        &'a self,
        _invocation: RuntimeCodeModeNestedToolCall,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, Value> {
        Box::pin(async move {
            cancellation_token.cancelled().await;
            Err("code mode nested tools are unavailable".to_string())
        })
    }

    fn notify<'a>(
        &'a self,
        _tool_call_id: String,
        _cell_id: RuntimeCodeModeCellId,
        _text: String,
        _cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

    fn cell_closed(&self, _cell_id: &RuntimeCodeModeCellId) {}
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCodeModeSessionLimits {
    pub max_yield_time_ms: Option<u64>,
    pub max_heap_size_bytes: Option<usize>,
}

pub type RuntimeCodeModeSessionProviderFuture<'a> =
    RuntimeCodeModeFuture<'a, RuntimeCodeModeSessionHandle>;

pub trait RuntimeCodeModeSessionProvider: Send + Sync {
    fn availability(&self) -> Result<(), String> {
        Ok(())
    }

    fn create_session<'a>(
        &'a self,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    ) -> RuntimeCodeModeSessionProviderFuture<'a>;

    fn create_session_with_limits<'a>(
        &'a self,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
        limits: RuntimeCodeModeSessionLimits,
    ) -> RuntimeCodeModeSessionProviderFuture<'a> {
        if limits == RuntimeCodeModeSessionLimits::default() {
            self.create_session(delegate)
        } else {
            Box::pin(async {
                Err("code mode session provider does not support resource limits".to_string())
            })
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeToolMode {
    #[default]
    Direct,
    CodeMode,
    CodeModeOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeToolModeResolution {
    pub requested: RuntimeToolMode,
    pub effective: RuntimeToolMode,
    pub used_direct_fallback: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeToolModeResolutionError {
    CodeModeUnavailable { requested: RuntimeToolMode },
    ReservedToolNameCollision { tool_name: String },
}

impl fmt::Display for RuntimeToolModeResolutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CodeModeUnavailable { requested } => {
                write!(
                    formatter,
                    "{requested:?} requested but code mode is unavailable"
                )
            }
            Self::ReservedToolNameCollision { tool_name } => {
                write!(
                    formatter,
                    "tool '{tool_name}' collides with a code mode reserved name"
                )
            }
        }
    }
}

impl std::error::Error for RuntimeToolModeResolutionError {}

pub fn resolve_runtime_tool_mode(
    requested: RuntimeToolMode,
    code_mode_available: bool,
    disable_direct_fallback: bool,
) -> Result<RuntimeToolModeResolution, RuntimeToolModeResolutionError> {
    let (effective, used_direct_fallback) = match requested {
        RuntimeToolMode::Direct => (RuntimeToolMode::Direct, false),
        RuntimeToolMode::CodeMode => {
            if code_mode_available {
                (RuntimeToolMode::CodeMode, false)
            } else if disable_direct_fallback {
                return Err(RuntimeToolModeResolutionError::CodeModeUnavailable { requested });
            } else {
                (RuntimeToolMode::Direct, true)
            }
        }
        RuntimeToolMode::CodeModeOnly => {
            if !code_mode_available {
                return Err(RuntimeToolModeResolutionError::CodeModeUnavailable { requested });
            }
            (RuntimeToolMode::CodeModeOnly, false)
        }
    };

    Ok(RuntimeToolModeResolution {
        requested,
        effective,
        used_direct_fallback,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCodeModeTool {
    pub identity: RuntimeToolIdentity,
    pub definition: RuntimeToolDefinition,
    pub code_name: String,
    pub global_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeCodeModeToolPlan {
    pub resolution: RuntimeToolModeResolution,
    pub model_visible_tools: Vec<RuntimeToolSnapshot>,
    pub searchable_tools: Vec<RuntimeToolSnapshot>,
    pub nested_tools: Vec<RuntimeCodeModeTool>,
    pub shadowed_nested_tools: Vec<RuntimeCodeModeTool>,
}

pub fn plan_runtime_code_mode_tools(
    tools: &[RuntimeToolSnapshot],
    requested: RuntimeToolMode,
    code_mode_available: bool,
    disable_direct_fallback: bool,
) -> Result<RuntimeCodeModeToolPlan, RuntimeToolModeResolutionError> {
    let resolution =
        resolve_runtime_tool_mode(requested, code_mode_available, disable_direct_fallback)?;
    let mut model_visible_tools = Vec::new();
    let mut searchable_tools = Vec::new();
    let mut nested_tools = Vec::new();
    let mut shadowed_nested_tools = Vec::new();
    let mut nested_global_names = HashSet::new();

    for tool in tools {
        let exposure = tool.exposure;
        let code_name = code_mode_name_for_identity(&tool.identity);
        if resolution.effective != RuntimeToolMode::Direct
            && exposure != crate::tool_definition::RuntimeToolExposure::Hidden
            && matches!(
                code_name.as_str(),
                CODE_MODE_EXEC_TOOL_NAME | CODE_MODE_WAIT_TOOL_NAME
            )
        {
            return Err(RuntimeToolModeResolutionError::ReservedToolNameCollision {
                tool_name: code_name,
            });
        }
        if exposure.is_deferred() {
            searchable_tools.push(tool.clone());
        }

        let model_visible = match resolution.effective {
            RuntimeToolMode::Direct | RuntimeToolMode::CodeMode => exposure.is_direct(),
            RuntimeToolMode::CodeModeOnly => {
                exposure.is_direct() && !exposure.is_available_in_code_mode()
            }
        };
        if model_visible {
            let mut tool = tool.clone();
            tool.model_visible = true;
            model_visible_tools.push(tool);
        }

        if resolution.effective == RuntimeToolMode::Direct || !exposure.is_available_in_code_mode()
        {
            continue;
        }

        let global_name = normalize_code_mode_identifier(&code_name);
        let nested = RuntimeCodeModeTool {
            identity: tool.identity.clone(),
            definition: tool.definition.clone(),
            code_name,
            global_name: global_name.clone(),
        };
        if nested_global_names.insert(global_name) {
            nested_tools.push(nested);
        } else {
            shadowed_nested_tools.push(nested);
        }
    }

    Ok(RuntimeCodeModeToolPlan {
        resolution,
        model_visible_tools,
        searchable_tools,
        nested_tools,
        shadowed_nested_tools,
    })
}

pub fn code_mode_name_for_identity(identity: &RuntimeToolIdentity) -> String {
    let Some(namespace) = identity.namespace.as_deref() else {
        return identity.name.clone();
    };
    if namespace.ends_with('_') || identity.name.starts_with('_') {
        format!("{namespace}{}", identity.name)
    } else {
        format!("{namespace}__{}", identity.name)
    }
}

pub fn normalize_code_mode_identifier(tool_name: &str) -> String {
    let mut identifier = String::new();
    for (index, character) in tool_name.chars().enumerate() {
        let valid = if index == 0 {
            character == '_' || character == '$' || character.is_ascii_alphabetic()
        } else {
            character == '_' || character == '$' || character.is_ascii_alphanumeric()
        };
        identifier.push(if valid { character } else { '_' });
    }
    if identifier.is_empty() {
        "_".to_string()
    } else {
        identifier
    }
}

#[cfg(test)]
#[path = "code_mode/tests.rs"]
mod tests;
