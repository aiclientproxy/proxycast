//! Code Mode tool descriptions, planning, and source parsing.

use crate::{
    CODE_MODE_EXEC_PRAGMA_PREFIX, CODE_MODE_EXEC_TOOL_NAME, CODE_MODE_WAIT_TOOL_NAME,
    DEFAULT_CODE_MODE_EXEC_YIELD_TIME_MS,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeToolExposure {
    #[default]
    Direct,
    Deferred,
    DeferredModelOnly,
    DirectModelOnly,
    CodeModeOnly,
    Hidden,
}
impl RuntimeToolExposure {
    pub fn is_direct(self) -> bool {
        matches!(self, Self::Direct | Self::DirectModelOnly)
    }
    pub fn is_deferred(self) -> bool {
        matches!(self, Self::Deferred | Self::DeferredModelOnly)
    }
    pub fn is_available_in_code_mode(self) -> bool {
        matches!(self, Self::Direct | Self::Deferred | Self::CodeModeOnly)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}
impl RuntimeToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RuntimeToolIdentity {
    pub namespace: Option<String>,
    pub name: String,
}
impl RuntimeToolIdentity {
    pub fn plain(name: impl Into<String>) -> Self {
        Self {
            namespace: None,
            name: name.into(),
        }
    }
    pub fn namespaced(namespace: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            namespace: Some(namespace.into()),
            name: name.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeToolSnapshot {
    pub identity: RuntimeToolIdentity,
    pub definition: RuntimeToolDefinition,
    pub exposure: RuntimeToolExposure,
    pub supports_parallel: bool,
    pub model_visible: bool,
}
impl RuntimeToolSnapshot {
    pub fn new(
        identity: RuntimeToolIdentity,
        definition: RuntimeToolDefinition,
        exposure: RuntimeToolExposure,
        supports_parallel: bool,
        model_visible: bool,
    ) -> Self {
        Self {
            identity,
            definition,
            exposure,
            supports_parallel,
            model_visible,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCodeModeTool {
    pub identity: RuntimeToolIdentity,
    pub definition: RuntimeToolDefinition,
    #[serde(default)]
    pub kind: CodeModeToolKind,
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CodeModeUnavailable { requested } => {
                write!(f, "{requested:?} requested but code mode is unavailable")
            }
            Self::ReservedToolNameCollision { tool_name } => write!(
                f,
                "tool '{tool_name}' collides with a code mode reserved name"
            ),
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
        RuntimeToolMode::CodeMode if code_mode_available => (RuntimeToolMode::CodeMode, false),
        RuntimeToolMode::CodeMode if disable_direct_fallback => {
            return Err(RuntimeToolModeResolutionError::CodeModeUnavailable { requested })
        }
        RuntimeToolMode::CodeMode => (RuntimeToolMode::Direct, true),
        RuntimeToolMode::CodeModeOnly if code_mode_available => {
            (RuntimeToolMode::CodeModeOnly, false)
        }
        RuntimeToolMode::CodeModeOnly => {
            return Err(RuntimeToolModeResolutionError::CodeModeUnavailable { requested })
        }
    };
    Ok(RuntimeToolModeResolution {
        requested,
        effective,
        used_direct_fallback,
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
    let mut names = std::collections::HashSet::new();
    for tool in tools {
        let code_name = code_mode_name_for_identity(&tool.identity);
        if resolution.effective != RuntimeToolMode::Direct
            && tool.exposure != RuntimeToolExposure::Hidden
            && matches!(
                code_name.as_str(),
                CODE_MODE_EXEC_TOOL_NAME | CODE_MODE_WAIT_TOOL_NAME
            )
        {
            return Err(RuntimeToolModeResolutionError::ReservedToolNameCollision {
                tool_name: code_name,
            });
        }
        if tool.exposure.is_deferred() {
            searchable_tools.push(tool.clone());
        }
        let visible = match resolution.effective {
            RuntimeToolMode::Direct | RuntimeToolMode::CodeMode => tool.exposure.is_direct(),
            RuntimeToolMode::CodeModeOnly => {
                tool.exposure.is_direct() && !tool.exposure.is_available_in_code_mode()
            }
        };
        if visible {
            let mut snapshot = tool.clone();
            snapshot.model_visible = true;
            model_visible_tools.push(snapshot);
        }
        if resolution.effective == RuntimeToolMode::Direct
            || !tool.exposure.is_available_in_code_mode()
        {
            continue;
        }
        let global_name = normalize_code_mode_identifier(&code_name);
        let nested = RuntimeCodeModeTool {
            identity: tool.identity.clone(),
            definition: tool.definition.clone(),
            kind: CodeModeToolKind::Function,
            code_name,
            global_name: global_name.clone(),
        };
        if names.insert(global_name) {
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
pub fn code_mode_exec_tool_description(tools: &[RuntimeCodeModeTool]) -> String {
    let mut description = format!("Run JavaScript code to orchestrate tool calls. Each cell evaluates as an async module in a fresh sandbox-enabled V8 isolate; cells in the same thread share values written with store/load. There is no Node.js, file system, network, or console access. The input is raw JavaScript source, not JSON. Long-running cells yield after {} ms and return a cell ID. Nested tools are asynchronous and must be awaited.\n\nGlobal helpers: text(value), image(value, detail?), audio(value), generatedImage(result), store(key, value), load(key), notify(value), setTimeout(callback, delayMs?), clearTimeout(id?), yield_control(), and exit().", DEFAULT_CODE_MODE_EXEC_YIELD_TIME_MS);
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
pub fn parse_code_mode_exec_source(input: &str) -> Result<RuntimeCodeModeParsedExecSource, String> {
    const MAX_SAFE: u64 = (1_u64 << 53) - 1;
    if input.trim().is_empty() {
        return Err("exec expects non-empty raw JavaScript source text".to_string());
    }
    let mut parsed = RuntimeCodeModeParsedExecSource {
        code: input.to_string(),
        yield_time_ms: None,
        max_output_tokens: None,
    };
    let mut lines = input.splitn(2, '\n');
    let first = lines.next().unwrap_or_default();
    let rest = lines.next().unwrap_or_default();
    let Some(directive) = first
        .trim_start()
        .strip_prefix(CODE_MODE_EXEC_PRAGMA_PREFIX)
    else {
        return Ok(parsed);
    };
    if rest.trim().is_empty() {
        return Err("exec pragma must be followed by JavaScript source".to_string());
    }
    let value: Value = serde_json::from_str(directive.trim())
        .map_err(|e| format!("exec pragma must be valid JSON: {e}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "exec pragma must be a JSON object".to_string())?;
    for key in object.keys() {
        if !matches!(key.as_str(), "yield_time_ms" | "max_output_tokens") {
            return Err(format!(
                "exec pragma only supports `yield_time_ms` and `max_output_tokens`; got `{key}`"
            ));
        }
    }
    parsed.yield_time_ms = object.get("yield_time_ms").and_then(Value::as_u64);
    parsed.max_output_tokens = object
        .get("max_output_tokens")
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok());
    if parsed.yield_time_ms.is_some_and(|v| v > MAX_SAFE)
        || parsed
            .max_output_tokens
            .map(|v| v as u64)
            .is_some_and(|v| v > MAX_SAFE)
    {
        return Err("exec pragma values must be JavaScript safe integers".to_string());
    }
    parsed.code = rest.to_string();
    Ok(parsed)
}
pub fn code_mode_wait_tool_definition() -> RuntimeToolDefinition {
    RuntimeToolDefinition::new(
        CODE_MODE_WAIT_TOOL_NAME,
        "Waits on a yielded exec cell and returns new output or completion.",
        serde_json::json!({"type":"object","required":["cell_id"],"additionalProperties":false,"properties":{"cell_id":{"type":"string"},"yield_time_ms":{"type":"number"},"max_tokens":{"type":"number"},"terminate":{"type":"boolean"}}}),
    )
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeModeToolKind {
    #[default]
    Function,
    Freeform,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolNamespaceDescription {
    pub name: String,
    pub description: String,
}

pub fn is_code_mode_nested_tool(tool_name: &str) -> bool {
    tool_name != crate::CODE_MODE_EXEC_TOOL_NAME && tool_name != crate::CODE_MODE_WAIT_TOOL_NAME
}

pub fn build_exec_tool_description(tools: &[RuntimeCodeModeTool]) -> String {
    code_mode_exec_tool_description(tools)
}
