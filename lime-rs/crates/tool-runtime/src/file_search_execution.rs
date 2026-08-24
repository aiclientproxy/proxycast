use crate::tool_definition::RuntimeToolDefinition;
use crate::tool_result_projection::{
    runtime_tool_result_to_call_tool_result, RuntimeToolResultParts,
};
use glob::glob as glob_match;
use regex::{Regex, RegexBuilder};
use rmcp::model::{CallToolResult, ErrorCode, ErrorData};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::cmp::Reverse;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tokio_util::sync::CancellationToken;

const DEFAULT_MAX_RESULTS: usize = 100;
const DEFAULT_MAX_CONTEXT_LINES: usize = 5;
const MAX_OUTPUT_SIZE: usize = 100_000;
pub const GLOB_TOOL_NAME: &str = "Glob";
pub const GREP_TOOL_NAME: &str = "Grep";
const GLOB_LEGACY_ALIASES: &[&str] = &["GlobTool", "mcp__system__glob"];
const GREP_LEGACY_ALIASES: &[&str] = &["GrepTool", "ripgrep", "mcp__system__grep"];

pub struct RuntimeFileSearchRequest<'a> {
    pub tool_name: &'a str,
    pub params: &'a Value,
    pub working_directory: PathBuf,
    pub cancel_token: Option<CancellationToken>,
    pub environment_id: Option<String>,
    pub filesystem_gateway: Option<Arc<dyn crate::filesystem_gateway::RuntimeFileSystemGateway>>,
    pub sandbox_policy: Option<String>,
}

pub fn file_search_tool_definitions() -> Vec<RuntimeToolDefinition> {
    [GLOB_TOOL_NAME, GREP_TOOL_NAME]
        .into_iter()
        .filter_map(file_search_tool_definition)
        .collect()
}

pub fn file_search_tool_definition(tool_name: &str) -> Option<RuntimeToolDefinition> {
    match file_search_canonical_tool_name(tool_name)? {
        GLOB_TOOL_NAME => Some(RuntimeToolDefinition::new(
            GLOB_TOOL_NAME,
            "Find files using glob patterns. Supports wildcards like *, **, ?, and character classes. \
             Results are sorted by modification time (newest first).",
            json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files. Examples: '*.rs', 'src/**/*.ts', 'test_*.py'"
                    },
                    "path": {
                        "type": "string",
                        "description": "Base path to search from. Defaults to working directory."
                    },
                    "exclude": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Patterns to exclude from results (e.g., ['node_modules', '.git'])"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Default: 100"
                    }
                },
                "required": ["pattern"]
            }),
        )),
        GREP_TOOL_NAME => Some(RuntimeToolDefinition::new(
            GREP_TOOL_NAME,
            "Search file contents using regex patterns. Supports multiple output modes: content (default), \
             files_with_matches, and count.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to search in. Defaults to working directory."
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "description": "Output mode. 'content' returns matching lines, 'files_with_matches' returns file names, 'count' returns match counts."
                    },
                    "context_before": {
                        "type": "integer",
                        "description": "Number of lines to show before each match. Default: 0"
                    },
                    "context_after": {
                        "type": "integer",
                        "description": "Number of lines to show after each match. Default: 0"
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Whether to ignore case. Default: false"
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Whether to search hidden files. Default: false"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Default: 100"
                    }
                },
                "required": ["pattern"]
            }),
        )),
        _ => None,
    }
}

pub fn file_search_canonical_tool_name(tool_name: &str) -> Option<&'static str> {
    let trimmed = tool_name.trim();
    if trimmed.is_empty() {
        return None;
    }

    file_search_canonical_tool_name_direct(trimmed).or_else(|| {
        model_visible_namespace_tail(trimmed).and_then(file_search_canonical_tool_name_direct)
    })
}

fn file_search_canonical_tool_name_direct(tool_name: &str) -> Option<&'static str> {
    if tool_name.eq_ignore_ascii_case(GLOB_TOOL_NAME) {
        return Some(GLOB_TOOL_NAME);
    }
    if tool_name.eq_ignore_ascii_case(GREP_TOOL_NAME) {
        return Some(GREP_TOOL_NAME);
    }
    if GLOB_LEGACY_ALIASES
        .iter()
        .any(|alias| tool_name.eq_ignore_ascii_case(alias))
    {
        return Some(GLOB_TOOL_NAME);
    }
    if GREP_LEGACY_ALIASES
        .iter()
        .any(|alias| tool_name.eq_ignore_ascii_case(alias))
    {
        return Some(GREP_TOOL_NAME);
    }

    None
}

fn model_visible_namespace_tail(name: &str) -> Option<&str> {
    for prefix in [
        "functions.",
        "functions__",
        "function.",
        "function__",
        "tools.",
        "tools__",
        "tool.",
        "tool__",
        "native.",
        "native__",
        "builtin.",
        "builtin__",
    ] {
        if name
            .get(..prefix.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
        {
            let tail = name[prefix.len()..].trim();
            if !tail.is_empty() {
                return Some(tail);
            }
        }
    }

    None
}

#[derive(Debug, Deserialize)]
struct GlobInput {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    exclude: Vec<String>,
    #[serde(default)]
    max_results: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct GrepInput {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    mode: GrepOutputMode,
    #[serde(default)]
    context_before: Option<usize>,
    #[serde(default)]
    context_after: Option<usize>,
    #[serde(default)]
    case_insensitive: bool,
    #[serde(default)]
    include_hidden: bool,
    #[serde(default)]
    max_results: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum GrepOutputMode {
    #[default]
    Content,
    FilesWithMatches,
    Count,
}

impl GrepOutputMode {
    fn metadata_value(self) -> &'static str {
        match self {
            Self::Content => "content",
            Self::FilesWithMatches => "files_with_matches",
            Self::Count => "count",
        }
    }
}

#[derive(Debug, Clone)]
struct SearchResult {
    path: PathBuf,
    line_number: Option<usize>,
    line_content: Option<String>,
    context_before: Vec<String>,
    context_after: Vec<String>,
    mtime: Option<SystemTime>,
    size: Option<u64>,
    match_count: Option<usize>,
}

impl SearchResult {
    fn file_match(path: PathBuf) -> Self {
        Self {
            path,
            line_number: None,
            line_content: None,
            context_before: Vec::new(),
            context_after: Vec::new(),
            mtime: None,
            size: None,
            match_count: None,
        }
    }

    fn content_match(path: PathBuf, line_number: usize, line_content: String) -> Self {
        Self {
            path,
            line_number: Some(line_number),
            line_content: Some(line_content),
            context_before: Vec::new(),
            context_after: Vec::new(),
            mtime: None,
            size: None,
            match_count: None,
        }
    }

    fn count_match(path: PathBuf, count: usize) -> Self {
        Self {
            path,
            line_number: None,
            line_content: None,
            context_before: Vec::new(),
            context_after: Vec::new(),
            mtime: None,
            size: None,
            match_count: Some(count),
        }
    }

    fn with_context(mut self, before: Vec<String>, after: Vec<String>) -> Self {
        self.context_before = before;
        self.context_after = after;
        self
    }

    fn with_metadata(mut self, mtime: Option<SystemTime>, size: Option<u64>) -> Self {
        self.mtime = mtime;
        self.size = size;
        self
    }
}

pub async fn execute_runtime_file_search_tool(
    request: RuntimeFileSearchRequest<'_>,
) -> Option<Result<CallToolResult, ErrorData>> {
    if request
        .cancel_token
        .as_ref()
        .is_some_and(CancellationToken::is_cancelled)
    {
        return Some(Err(runtime_search_error(
            ErrorCode::INTERNAL_ERROR,
            "Tool execution cancelled",
        )));
    }

    let canonical_tool_name = file_search_canonical_tool_name(request.tool_name)?;
    let remote_environment_id = request
        .environment_id
        .as_deref()
        .filter(|environment_id| *environment_id != "local");
    if let Some(environment_id) = remote_environment_id {
        let Some(gateway) = request.filesystem_gateway.as_ref() else {
            return Some(Err(runtime_search_error(
                ErrorCode::INTERNAL_ERROR,
                format!("remote Environment '{environment_id}' filesystem gateway is unavailable"),
            )));
        };
        return match canonical_tool_name {
            GLOB_TOOL_NAME => Some(
                execute_remote_glob(
                    request.params,
                    &request.working_directory,
                    environment_id,
                    Arc::clone(gateway),
                    request.sandbox_policy.as_deref(),
                )
                .await,
            ),
            GREP_TOOL_NAME => Some(
                execute_remote_grep(
                    request.params,
                    &request.working_directory,
                    environment_id,
                    Arc::clone(gateway),
                    request.sandbox_policy.as_deref(),
                )
                .await,
            ),
            _ => None,
        };
    }
    match canonical_tool_name {
        GLOB_TOOL_NAME => Some(execute_glob(request.params, &request.working_directory)),
        GREP_TOOL_NAME => Some(execute_grep(request.params, &request.working_directory)),
        _ => None,
    }
}

async fn execute_remote_glob(
    params: &Value,
    working_directory: &Path,
    environment_id: &str,
    gateway: Arc<dyn crate::filesystem_gateway::RuntimeFileSystemGateway>,
    sandbox_policy: Option<&str>,
) -> Result<CallToolResult, ErrorData> {
    let input: GlobInput = serde_json::from_value(params.clone()).map_err(|error| {
        runtime_search_error(
            ErrorCode::INVALID_PARAMS,
            format!("Invalid Glob parameters: {error}"),
        )
    })?;
    let pattern = normalize_required_string(&input.pattern, "pattern")?;
    let base_path = input
        .path
        .as_deref()
        .map(|path| resolve_path(path, working_directory))
        .unwrap_or_else(|| working_directory.to_path_buf());
    let entries = gateway
        .walk(
            environment_id,
            &base_path,
            crate::filesystem_gateway::RuntimeFileWalkOptions {
                max_depth: 64,
                max_directories: 10_000,
                max_entries: 50_000,
                follow_directory_symlinks: false,
                prune_hidden_directories: false,
            },
            sandbox_policy,
        )
        .await
        .map_err(|error| runtime_search_error(ErrorCode::INTERNAL_ERROR, error))?;
    let mut results = entries
        .into_iter()
        .filter(|entry| entry.is_file && glob_pattern_matches(&pattern, &base_path, &entry.path))
        .filter(|entry| !excluded_path(&entry.path, &input.exclude))
        .map(|entry| SearchResult::file_match(entry.path))
        .collect::<Vec<_>>();
    let max_results = input.max_results.unwrap_or(DEFAULT_MAX_RESULTS);
    let (results, truncated) = truncate_results(std::mem::take(&mut results), max_results);
    Ok(runtime_tool_result_to_call_tool_result(
        RuntimeToolResultParts {
            success: true,
            output: Some(format_search_results(&results)),
            error: None,
            metadata: HashMap::from([
                ("count".to_string(), json!(results.len())),
                ("truncated".to_string(), json!(truncated)),
                ("pattern".to_string(), json!(pattern)),
                ("environment_id".to_string(), json!(environment_id)),
            ]),
        },
    ))
}

async fn execute_remote_grep(
    params: &Value,
    working_directory: &Path,
    environment_id: &str,
    gateway: Arc<dyn crate::filesystem_gateway::RuntimeFileSystemGateway>,
    sandbox_policy: Option<&str>,
) -> Result<CallToolResult, ErrorData> {
    let input: GrepInput = serde_json::from_value(params.clone()).map_err(|error| {
        runtime_search_error(
            ErrorCode::INVALID_PARAMS,
            format!("Invalid Grep parameters: {error}"),
        )
    })?;
    let pattern = normalize_required_string(&input.pattern, "pattern")?;
    let regex = RegexBuilder::new(&pattern)
        .case_insensitive(input.case_insensitive)
        .build()
        .map_err(|error| {
            runtime_search_error(
                ErrorCode::INVALID_PARAMS,
                format!("Invalid regex pattern: {error}"),
            )
        })?;
    let base_path = input
        .path
        .as_deref()
        .map(|path| resolve_path(path, working_directory))
        .unwrap_or_else(|| working_directory.to_path_buf());
    let entries = gateway
        .walk(
            environment_id,
            &base_path,
            crate::filesystem_gateway::RuntimeFileWalkOptions {
                max_depth: 64,
                max_directories: 10_000,
                max_entries: 50_000,
                follow_directory_symlinks: false,
                prune_hidden_directories: !input.include_hidden,
            },
            sandbox_policy,
        )
        .await
        .map_err(|error| runtime_search_error(ErrorCode::INTERNAL_ERROR, error))?;
    let before = input
        .context_before
        .unwrap_or(0)
        .min(DEFAULT_MAX_CONTEXT_LINES);
    let after = input
        .context_after
        .unwrap_or(0)
        .min(DEFAULT_MAX_CONTEXT_LINES);
    let hard_limit = input
        .max_results
        .unwrap_or(DEFAULT_MAX_RESULTS)
        .saturating_mul(10)
        .max(DEFAULT_MAX_RESULTS);
    let mut results = Vec::new();
    for entry in entries.into_iter().filter(|entry| entry.is_file) {
        if results.len() >= hard_limit {
            break;
        }
        let bytes = gateway
            .read_file(environment_id, &entry.path, sandbox_policy)
            .await
            .map_err(|error| runtime_search_error(ErrorCode::INTERNAL_ERROR, error))?;
        let Ok(text) = String::from_utf8(bytes) else {
            continue;
        };
        let lines = text.lines().map(ToString::to_string).collect::<Vec<_>>();
        let mut match_count = 0;
        for (index, line) in lines.iter().enumerate() {
            if !regex.is_match(line) {
                continue;
            }
            match_count += 1;
            if input.mode == GrepOutputMode::Content {
                results.push(
                    SearchResult::content_match(entry.path.clone(), index + 1, line.clone())
                        .with_context(
                            lines[index.saturating_sub(before)..index].to_vec(),
                            lines[(index + 1)..(index + after + 1).min(lines.len())].to_vec(),
                        ),
                );
            }
        }
        match input.mode {
            GrepOutputMode::FilesWithMatches if match_count > 0 => {
                results.push(SearchResult::file_match(entry.path.clone()))
            }
            GrepOutputMode::Count if match_count > 0 => {
                results.push(SearchResult::count_match(entry.path.clone(), match_count))
            }
            _ => {}
        }
    }
    let max_results = input.max_results.unwrap_or(DEFAULT_MAX_RESULTS);
    let (results, result_truncated) = truncate_results(results, max_results);
    let (output, output_truncated) = truncate_output(&format_search_results(&results));
    Ok(runtime_tool_result_to_call_tool_result(
        RuntimeToolResultParts {
            success: true,
            output: Some(output),
            error: None,
            metadata: HashMap::from([
                ("count".to_string(), json!(results.len())),
                (
                    "truncated".to_string(),
                    json!(result_truncated || output_truncated),
                ),
                ("mode".to_string(), json!(input.mode.metadata_value())),
                ("pattern".to_string(), json!(pattern)),
                ("environment_id".to_string(), json!(environment_id)),
            ]),
        },
    ))
}

fn glob_pattern_matches(pattern: &str, base_path: &Path, path: &Path) -> bool {
    let relative = path
        .strip_prefix(base_path)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    let pattern = pattern.replace('\\', "/");
    glob::Pattern::new(&pattern)
        .map(|pattern| pattern.matches(&relative) || pattern.matches(&path.to_string_lossy()))
        .unwrap_or(false)
}

fn execute_glob(params: &Value, working_directory: &Path) -> Result<CallToolResult, ErrorData> {
    let input: GlobInput = serde_json::from_value(params.clone()).map_err(|error| {
        runtime_search_error(
            ErrorCode::INVALID_PARAMS,
            format!("Invalid Glob parameters: {error}"),
        )
    })?;
    let pattern = normalize_required_string(&input.pattern, "pattern")?;
    let base_path = input
        .path
        .as_deref()
        .map(|path| resolve_path(path, working_directory))
        .unwrap_or_else(|| working_directory.to_path_buf());
    let max_results = input.max_results.unwrap_or(DEFAULT_MAX_RESULTS);

    let results = glob_search(&pattern, &base_path, &input.exclude)?;
    let (results, truncated) = truncate_results(results, max_results);
    let output = format_search_results(&results);
    let metadata = HashMap::from([
        ("count".to_string(), json!(results.len())),
        ("truncated".to_string(), json!(truncated)),
        ("pattern".to_string(), json!(pattern)),
        ("path".to_string(), json!(base_path.display().to_string())),
    ]);

    Ok(runtime_tool_result_to_call_tool_result(
        RuntimeToolResultParts {
            success: true,
            output: Some(output),
            error: None,
            metadata,
        },
    ))
}

fn execute_grep(params: &Value, working_directory: &Path) -> Result<CallToolResult, ErrorData> {
    let input: GrepInput = serde_json::from_value(params.clone()).map_err(|error| {
        runtime_search_error(
            ErrorCode::INVALID_PARAMS,
            format!("Invalid Grep parameters: {error}"),
        )
    })?;
    let pattern = normalize_required_string(&input.pattern, "pattern")?;
    let path = input
        .path
        .as_deref()
        .map(|path| resolve_path(path, working_directory))
        .unwrap_or_else(|| working_directory.to_path_buf());
    let context_before = input
        .context_before
        .unwrap_or(0)
        .min(DEFAULT_MAX_CONTEXT_LINES);
    let context_after = input
        .context_after
        .unwrap_or(0)
        .min(DEFAULT_MAX_CONTEXT_LINES);
    let max_results = input.max_results.unwrap_or(DEFAULT_MAX_RESULTS);
    let regex = RegexBuilder::new(&pattern)
        .case_insensitive(input.case_insensitive)
        .build()
        .map_err(|error| {
            runtime_search_error(
                ErrorCode::INVALID_PARAMS,
                format!("Invalid regex pattern: {error}"),
            )
        })?;

    let results = grep_search(
        &regex,
        &path,
        input.mode,
        context_before,
        context_after,
        input.include_hidden,
        max_results.saturating_mul(10).max(max_results),
    )?;
    let (results, result_truncated) = truncate_results(results, max_results);
    let output = format_search_results(&results);
    let (output, output_truncated) = truncate_output(&output);
    let metadata = HashMap::from([
        ("count".to_string(), json!(results.len())),
        (
            "truncated".to_string(),
            json!(result_truncated || output_truncated),
        ),
        ("mode".to_string(), json!(input.mode.metadata_value())),
        ("pattern".to_string(), json!(pattern)),
        ("path".to_string(), json!(path.display().to_string())),
    ]);

    Ok(runtime_tool_result_to_call_tool_result(
        RuntimeToolResultParts {
            success: true,
            output: Some(output),
            error: None,
            metadata,
        },
    ))
}

fn glob_search(
    pattern: &str,
    base_path: &Path,
    exclude_patterns: &[String],
) -> Result<Vec<SearchResult>, ErrorData> {
    let full_pattern = glob_pattern(pattern, base_path);
    let paths = glob_match(&full_pattern).map_err(|error| {
        runtime_search_error(
            ErrorCode::INVALID_PARAMS,
            format!("Invalid glob pattern: {error}"),
        )
    })?;
    let mut results = Vec::new();

    for entry in paths {
        let Ok(path) = entry else {
            continue;
        };
        if path.is_dir() || excluded_path(&path, exclude_patterns) {
            continue;
        }
        let (mtime, size) = fs::metadata(&path)
            .ok()
            .map(|metadata| (metadata.modified().ok(), Some(metadata.len())))
            .unwrap_or((None, None));
        results.push(SearchResult::file_match(path).with_metadata(mtime, size));
    }

    results.sort_by(|left, right| match (&left.mtime, &right.mtime) {
        (Some(left), Some(right)) => Reverse(left).cmp(&Reverse(right)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => left.path.cmp(&right.path),
    });
    Ok(results)
}

fn grep_search(
    regex: &Regex,
    path: &Path,
    mode: GrepOutputMode,
    context_before: usize,
    context_after: usize,
    include_hidden: bool,
    hard_result_limit: usize,
) -> Result<Vec<SearchResult>, ErrorData> {
    if !path.exists() {
        return Err(runtime_search_error(
            ErrorCode::INTERNAL_ERROR,
            format!("Path not found: {}", path.display()),
        ));
    }

    let mut results = Vec::new();
    if path.is_file() {
        search_file(
            regex,
            path,
            mode,
            context_before,
            context_after,
            &mut results,
        )?;
    } else {
        search_directory(
            regex,
            path,
            mode,
            context_before,
            context_after,
            include_hidden,
            hard_result_limit,
            &mut results,
        )?;
    }
    results.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then(left.line_number.cmp(&right.line_number))
    });
    Ok(results)
}

#[allow(clippy::too_many_arguments)]
fn search_directory(
    regex: &Regex,
    path: &Path,
    mode: GrepOutputMode,
    context_before: usize,
    context_after: usize,
    include_hidden: bool,
    hard_result_limit: usize,
    results: &mut Vec<SearchResult>,
) -> Result<(), ErrorData> {
    let entries = fs::read_dir(path).map_err(io_error)?;
    let mut paths = Vec::new();
    for entry in entries.flatten() {
        let entry_path = entry.path();
        if !include_hidden && is_hidden_path(&entry_path) {
            continue;
        }
        if entry
            .file_type()
            .map(|file_type| file_type.is_symlink())
            .unwrap_or(false)
        {
            continue;
        }
        paths.push(entry_path);
    }
    paths.sort();

    for entry_path in paths {
        if results.len() >= hard_result_limit {
            break;
        }
        if entry_path.is_file() {
            search_file(
                regex,
                &entry_path,
                mode,
                context_before,
                context_after,
                results,
            )?;
        } else if entry_path.is_dir() {
            search_directory(
                regex,
                &entry_path,
                mode,
                context_before,
                context_after,
                include_hidden,
                hard_result_limit,
                results,
            )?;
        }
    }
    Ok(())
}

fn search_file(
    regex: &Regex,
    path: &Path,
    mode: GrepOutputMode,
    context_before: usize,
    context_after: usize,
    results: &mut Vec<SearchResult>,
) -> Result<(), ErrorData> {
    if is_binary_file(path) {
        return Ok(());
    }
    let file = fs::File::open(path).map_err(io_error)?;
    let lines = BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .collect::<Vec<_>>();
    let mut match_count = 0usize;
    let mut file_has_match = false;

    for (index, line) in lines.iter().enumerate() {
        if !regex.is_match(line) {
            continue;
        }
        file_has_match = true;
        match_count += 1;
        if mode == GrepOutputMode::Content {
            let before = lines[index.saturating_sub(context_before)..index].to_vec();
            let after_end = (index + context_after + 1).min(lines.len());
            let after = if index + 1 < after_end {
                lines[index + 1..after_end].to_vec()
            } else {
                Vec::new()
            };
            results.push(
                SearchResult::content_match(path.to_path_buf(), index + 1, line.clone())
                    .with_context(before, after),
            );
        }
    }

    match mode {
        GrepOutputMode::FilesWithMatches if file_has_match => {
            results.push(SearchResult::file_match(path.to_path_buf()));
        }
        GrepOutputMode::Count if match_count > 0 => {
            results.push(SearchResult::count_match(path.to_path_buf(), match_count));
        }
        _ => {}
    }

    Ok(())
}

fn glob_pattern(pattern: &str, base_path: &Path) -> String {
    let pattern_path = PathBuf::from(pattern);
    if pattern_path.is_absolute() {
        pattern.to_string()
    } else {
        base_path.join(pattern_path).display().to_string()
    }
}

fn excluded_path(path: &Path, exclude_patterns: &[String]) -> bool {
    let path = path.to_string_lossy();
    exclude_patterns
        .iter()
        .any(|exclude| !exclude.is_empty() && path.contains(exclude))
}

fn normalize_required_string(value: &str, field: &str) -> Result<String, ErrorData> {
    let value = value.trim();
    if value.is_empty() {
        Err(runtime_search_error(
            ErrorCode::INVALID_PARAMS,
            format!("Missing required parameter: {field}"),
        ))
    } else {
        Ok(value.to_string())
    }
}

fn resolve_path(path: &str, working_directory: &Path) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        working_directory.join(path)
    }
}

fn format_search_results(results: &[SearchResult]) -> String {
    results
        .iter()
        .map(|result| {
            if let Some(line_number) = result.line_number {
                format!(
                    "{}:{}:{}",
                    result.path.display(),
                    line_number,
                    result.line_content.as_deref().unwrap_or("")
                )
            } else if let Some(count) = result.match_count {
                format!("{}:{}", result.path.display(), count)
            } else {
                result.path.display().to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn truncate_results(results: Vec<SearchResult>, max_results: usize) -> (Vec<SearchResult>, bool) {
    if results.len() > max_results {
        (results.into_iter().take(max_results).collect(), true)
    } else {
        (results, false)
    }
}

fn truncate_output(output: &str) -> (String, bool) {
    if output.len() <= MAX_OUTPUT_SIZE {
        return (output.to_string(), false);
    }
    let mut safe_length = MAX_OUTPUT_SIZE;
    while safe_length > 0 && !output.is_char_boundary(safe_length) {
        safe_length -= 1;
    }
    let truncated = output.get(..safe_length).unwrap_or(output);
    let last_newline = truncated.rfind('\n').unwrap_or(truncated.len());
    (
        format!(
            "{}\n\n[Output truncated. Showing first {} bytes of {} bytes total.]",
            truncated.get(..last_newline).unwrap_or(truncated),
            last_newline,
            output.len()
        ),
        true,
    )
}

fn is_hidden_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
}

fn is_binary_file(path: &Path) -> bool {
    const BINARY_EXTENSIONS: &[&str] = &[
        "exe", "dll", "so", "dylib", "bin", "obj", "o", "a", "lib", "png", "jpg", "jpeg", "gif",
        "bmp", "ico", "webp", "mp3", "mp4", "avi", "mov", "mkv", "wav", "flac", "zip", "tar", "gz",
        "bz2", "xz", "7z", "rar", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "wasm",
        "pyc", "class",
    ];
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| BINARY_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
    {
        return true;
    }
    if let Ok(mut file) = fs::File::open(path) {
        let mut buffer = [0u8; 512];
        if let Ok(bytes) = file.read(&mut buffer) {
            return buffer[..bytes].contains(&0);
        }
    }
    false
}

fn runtime_search_error(code: ErrorCode, message: impl Into<String>) -> ErrorData {
    ErrorData::new(code, message.into(), None)
}

fn io_error(error: std::io::Error) -> ErrorData {
    runtime_search_error(ErrorCode::INTERNAL_ERROR, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn result_text(result: &CallToolResult) -> &str {
        result.content[0]
            .as_text()
            .expect("text content")
            .text
            .as_str()
    }

    #[tokio::test]
    async fn glob_returns_files_sorted_and_truncated() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("b.rs"), "b").unwrap();
        fs::write(dir.path().join("a.rs"), "a").unwrap();

        let result = execute_runtime_file_search_tool(RuntimeFileSearchRequest {
            tool_name: "Glob",
            params: &json!({ "pattern": "*.rs", "max_results": 1 }),
            working_directory: dir.path().to_path_buf(),
            cancel_token: None,
            environment_id: None,
            filesystem_gateway: None,
            sandbox_policy: None,
        })
        .await
        .expect("glob tool")
        .expect("glob result");

        assert_eq!(result.is_error, Some(false));
        assert_eq!(
            result.structured_content.as_ref().unwrap().get("count"),
            Some(&json!(1))
        );
        assert_eq!(
            result.structured_content.as_ref().unwrap().get("truncated"),
            Some(&json!(true))
        );
        assert!(result_text(&result).contains(".rs"));
    }

    #[tokio::test]
    async fn grep_content_mode_searches_text_files() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("sample.txt"), "alpha\nbeta\nAlpha\n").unwrap();

        let result = execute_runtime_file_search_tool(RuntimeFileSearchRequest {
            tool_name: "Grep",
            params: &json!({
                "pattern": "alpha",
                "path": "sample.txt",
                "case_insensitive": true
            }),
            working_directory: dir.path().to_path_buf(),
            cancel_token: None,
            environment_id: None,
            filesystem_gateway: None,
            sandbox_policy: None,
        })
        .await
        .expect("grep tool")
        .expect("grep result");

        let text = result_text(&result);
        assert!(text.contains("sample.txt:1:alpha"));
        assert!(text.contains("sample.txt:3:Alpha"));
        assert_eq!(
            result.structured_content.as_ref().unwrap().get("mode"),
            Some(&json!("content"))
        );
    }

    #[tokio::test]
    async fn unknown_tool_falls_back_to_registry() {
        let result = execute_runtime_file_search_tool(RuntimeFileSearchRequest {
            tool_name: "Read",
            params: &json!({ "pattern": "*.rs" }),
            working_directory: PathBuf::from("."),
            cancel_token: None,
            environment_id: None,
            filesystem_gateway: None,
            sandbox_policy: None,
        })
        .await;

        assert!(result.is_none());
    }

    #[test]
    fn definitions_and_aliases_are_owned_by_tool_runtime() {
        let names = file_search_tool_definitions()
            .into_iter()
            .map(|definition| definition.name)
            .collect::<Vec<_>>();
        assert_eq!(names, vec![GLOB_TOOL_NAME, GREP_TOOL_NAME]);
        assert_eq!(
            file_search_canonical_tool_name("functions.GlobTool"),
            Some(GLOB_TOOL_NAME)
        );
        assert_eq!(
            file_search_canonical_tool_name("ripgrep"),
            Some(GREP_TOOL_NAME)
        );
    }
}
