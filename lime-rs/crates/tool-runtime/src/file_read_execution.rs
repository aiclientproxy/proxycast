use crate::tool_definition::RuntimeToolDefinition;
use crate::tool_result_projection::{
    runtime_tool_result_to_call_tool_result, RuntimeToolResultParts,
};
use rmcp::model::{CallToolResult, ErrorCode, ErrorData};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

const MAX_TEXT_FILE_SIZE: u64 = 10 * 1024 * 1024;
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp", "ico"];
const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "jsx", "tsx", "java", "c", "cpp", "cc", "cxx", "h", "hpp", "go", "rb",
    "php", "swift", "kt", "scala", "sh", "bash", "zsh", "sql",
];
const CONFIG_EXTENSIONS: &[&str] = &[
    "json", "yaml", "yml", "toml", "xml", "ini", "cfg", "conf", "env",
];
const DOCUMENTATION_EXTENSIONS: &[&str] = &["md", "txt", "rst", "adoc"];
pub const FILE_READ_TOOL_NAME: &str = "Read";
const FILE_READ_LEGACY_ALIASES: &[&str] = &[
    "ReadTool",
    "FileReadTool",
    "read_file",
    "developer__read",
    "mcp__system__read_file",
];

pub struct RuntimeFileReadRequest<'a> {
    pub tool_name: &'a str,
    pub params: &'a Value,
    pub working_directory: PathBuf,
    pub cancel_token: Option<CancellationToken>,
    pub environment_id: Option<String>,
    pub filesystem_gateway: Option<Arc<dyn crate::filesystem_gateway::RuntimeFileSystemGateway>>,
    pub sandbox_policy: Option<String>,
}

pub fn file_read_tool_definition() -> RuntimeToolDefinition {
    RuntimeToolDefinition::new(
        FILE_READ_TOOL_NAME,
        "File reader for text, document previews, SVG, and Jupyter notebooks. \
         Text files are returned as direct line-numbered content by default, \
         with an optional enhanced analysis mode when explicitly requested. \
         Images must use the current view_image tool; PDF multimodal ingestion is not provided by Read. \
         Optimized for reliable file reading in agent workflows.",
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to working directory or absolute)"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line number (1-indexed, for text files only)",
                    "minimum": 1
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line number (1-indexed, inclusive, for text files only)",
                    "minimum": 1
                },
                "text_output_mode": {
                    "type": "string",
                    "enum": ["plain", "enhanced"],
                    "description": "For text files only. `plain` returns direct line-numbered content and is the default. `enhanced` adds file-analysis headers and hints."
                }
            },
            "required": ["path"]
        }),
    )
}

pub fn file_read_canonical_tool_name(tool_name: &str) -> Option<&'static str> {
    let trimmed = tool_name.trim();
    if trimmed.is_empty() {
        return None;
    }

    file_read_canonical_tool_name_direct(trimmed).or_else(|| {
        model_visible_namespace_tail(trimmed).and_then(file_read_canonical_tool_name_direct)
    })
}

fn file_read_canonical_tool_name_direct(tool_name: &str) -> Option<&'static str> {
    if tool_name.eq_ignore_ascii_case(FILE_READ_TOOL_NAME) {
        return Some(FILE_READ_TOOL_NAME);
    }

    FILE_READ_LEGACY_ALIASES
        .iter()
        .any(|alias| tool_name.eq_ignore_ascii_case(alias))
        .then_some(FILE_READ_TOOL_NAME)
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
struct ReadInput {
    #[serde(default, alias = "file_path", alias = "filePath")]
    path: Option<String>,
    #[serde(default, alias = "startLine")]
    start_line: Option<usize>,
    #[serde(default, alias = "endLine")]
    end_line: Option<usize>,
    #[serde(default)]
    head: Option<usize>,
    #[serde(default)]
    text_output_mode: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LineRange {
    start: usize,
    end: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextOutputMode {
    Plain,
    Enhanced,
}

impl TextOutputMode {
    fn from_input(input: Option<&str>) -> Self {
        match input.map(str::trim) {
            Some(value) if value.eq_ignore_ascii_case("enhanced") => Self::Enhanced,
            _ => Self::Plain,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Enhanced => "enhanced",
        }
    }
}

pub async fn execute_runtime_file_read_tool(
    request: RuntimeFileReadRequest<'_>,
) -> Option<Result<CallToolResult, ErrorData>> {
    if file_read_canonical_tool_name(request.tool_name).is_none() {
        return None;
    }
    if request
        .cancel_token
        .as_ref()
        .is_some_and(CancellationToken::is_cancelled)
    {
        return Some(Err(runtime_read_error(
            ErrorCode::INTERNAL_ERROR,
            "Tool execution cancelled",
        )));
    }

    if let Some(environment_id) = request
        .environment_id
        .as_deref()
        .filter(|environment_id| *environment_id != "local")
    {
        let Some(gateway) = request.filesystem_gateway.as_ref() else {
            return Some(Err(runtime_read_error(
                ErrorCode::INTERNAL_ERROR,
                format!("remote Environment '{environment_id}' filesystem gateway is unavailable"),
            )));
        };
        return Some(
            execute_remote_read(
                request.params,
                &request.working_directory,
                environment_id,
                Arc::clone(gateway),
                request.sandbox_policy.as_deref(),
            )
            .await,
        );
    }
    let params = request.params.clone();
    let working_directory = request.working_directory;
    let cancel_token = request.cancel_token;
    Some(execute_local_read(params, working_directory, cancel_token).await)
}

async fn execute_local_read(
    params: Value,
    working_directory: PathBuf,
    cancel_token: Option<CancellationToken>,
) -> Result<CallToolResult, ErrorData> {
    let read = tokio::task::spawn_blocking(move || execute_read(&params, &working_directory));
    match cancel_token {
        Some(cancel_token) => {
            tokio::select! {
                result = read => join_local_read(result),
                _ = cancel_token.cancelled() => Err(runtime_read_error(
                    ErrorCode::INTERNAL_ERROR,
                    "Tool execution cancelled",
                )),
            }
        }
        None => join_local_read(read.await),
    }
}

fn join_local_read(
    result: Result<Result<CallToolResult, ErrorData>, tokio::task::JoinError>,
) -> Result<CallToolResult, ErrorData> {
    result.map_err(|error| {
        runtime_read_error(
            ErrorCode::INTERNAL_ERROR,
            format!("Read worker failed: {error}"),
        )
    })?
}

async fn execute_remote_read(
    params: &Value,
    working_directory: &Path,
    environment_id: &str,
    gateway: Arc<dyn crate::filesystem_gateway::RuntimeFileSystemGateway>,
    sandbox_policy: Option<&str>,
) -> Result<CallToolResult, ErrorData> {
    let input: ReadInput = serde_json::from_value(params.clone()).map_err(|error| {
        runtime_read_error(
            ErrorCode::INVALID_PARAMS,
            format!("Invalid Read parameters: {error}"),
        )
    })?;
    let path = input
        .path
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .ok_or_else(|| {
            runtime_read_error(
                ErrorCode::INVALID_PARAMS,
                "Missing required parameter: path",
            )
        })?;
    let full_path = resolve_path(path, working_directory);
    let bytes = gateway
        .read_file(environment_id, &full_path, sandbox_policy)
        .await
        .map_err(|error| runtime_read_error(ErrorCode::INTERNAL_ERROR, error))?;
    let text = String::from_utf8(bytes.clone()).map_err(|_| {
        runtime_read_error(
            ErrorCode::INTERNAL_ERROR,
            format!("Remote file is not UTF-8 text: {}", full_path.display()),
        )
    })?;
    let output = format_text_with_lines(&text, line_range_from_input(&input));
    Ok(runtime_tool_result_to_call_tool_result(
        RuntimeToolResultParts {
            success: true,
            output: Some(output),
            error: None,
            metadata: HashMap::from([
                ("file_type".to_string(), json!("text")),
                ("analysis_type".to_string(), json!("textual")),
                ("path".to_string(), json!(full_path.display().to_string())),
                ("content_bytes".to_string(), json!(bytes.len())),
                ("environment_id".to_string(), json!(environment_id)),
            ]),
        },
    ))
}

fn execute_read(params: &Value, working_directory: &Path) -> Result<CallToolResult, ErrorData> {
    let input: ReadInput = serde_json::from_value(params.clone()).map_err(|error| {
        runtime_read_error(
            ErrorCode::INVALID_PARAMS,
            format!("Invalid Read parameters: {error}"),
        )
    })?;
    let path = input
        .path
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .ok_or_else(|| {
            runtime_read_error(
                ErrorCode::INVALID_PARAMS,
                "Missing required parameter: path",
            )
        })?;
    let full_path = resolve_path(path, working_directory);

    if is_image_file(&full_path) {
        return Err(runtime_read_error(
            ErrorCode::INTERNAL_ERROR,
            format!(
                "Image reading through Read is retired: {}. Use the current view_image tool.",
                full_path.display()
            ),
        ));
    }
    if is_pdf_file(&full_path) {
        return Err(runtime_read_error(
            ErrorCode::INTERNAL_ERROR,
            format!(
                "PDF multimodal reading through Read is retired: {}. Use current document preview or document ingestion instead.",
                full_path.display()
            ),
        ));
    }

    let range = line_range_from_input(&input);
    let output_mode = TextOutputMode::from_input(input.text_output_mode.as_deref());
    let read_output = if is_svg_file(&full_path) {
        read_svg(&full_path)?
    } else if is_notebook_file(&full_path) {
        read_notebook(&full_path)?
    } else if document_preview::is_supported_document(&full_path) {
        read_document_preview(&full_path)?
    } else {
        read_text(&full_path, range, output_mode)?
    };

    Ok(runtime_tool_result_to_call_tool_result(
        RuntimeToolResultParts {
            success: true,
            output: Some(read_output.output),
            error: None,
            metadata: read_output.metadata,
        },
    ))
}

fn line_range_from_input(input: &ReadInput) -> Option<LineRange> {
    match (input.start_line, input.end_line, input.head) {
        (Some(start), end, _) => Some(LineRange { start, end }),
        (None, Some(end), _) => Some(LineRange {
            start: 1,
            end: Some(end),
        }),
        (None, None, Some(head)) => Some(LineRange {
            start: 1,
            end: Some(head),
        }),
        (None, None, None) => None,
    }
}

struct FileReadOutput {
    output: String,
    metadata: HashMap<String, Value>,
}

fn read_text(
    path: &Path,
    range: Option<LineRange>,
    output_mode: TextOutputMode,
) -> Result<FileReadOutput, ErrorData> {
    let (content, metadata, text) = load_text_file(path)?;
    let formatted = format_text_with_lines(&text, range);
    let line_count = text.lines().count();
    let output = match output_mode {
        TextOutputMode::Plain => formatted,
        TextOutputMode::Enhanced => {
            let language = detect_programming_language(path, &text);
            let category = categorize_file_type(path);
            format!(
                "[Text File: {}]\nFile type: {}\nLanguage: {}\nSize: {} bytes\nLines: {}\n\n{}",
                path.display(),
                category,
                language.as_deref().unwrap_or("unknown"),
                metadata.len(),
                line_count,
                formatted
            )
        }
    };

    Ok(FileReadOutput {
        output,
        metadata: HashMap::from([
            ("file_type".to_string(), json!("text")),
            ("analysis_type".to_string(), json!("textual")),
            ("text_output_mode".to_string(), json!(output_mode.as_str())),
            ("path".to_string(), json!(path.display().to_string())),
            ("size_bytes".to_string(), json!(metadata.len())),
            ("line_count".to_string(), json!(line_count)),
            ("content_bytes".to_string(), json!(content.len())),
        ]),
    })
}

fn read_document_preview(path: &Path) -> Result<FileReadOutput, ErrorData> {
    ensure_readable_file(path)?;
    let metadata = fs::metadata(path).map_err(io_error)?;
    let text =
        document_preview::extract_document_text_from_path(path, Some(MAX_TEXT_FILE_SIZE as usize))
            .map_err(|error| runtime_read_error(ErrorCode::INTERNAL_ERROR, error.to_string()))?;
    Ok(FileReadOutput {
        output: format!(
            "[Document Text Preview: {}]\nSize: {} bytes\n\n{}",
            path.display(),
            metadata.len(),
            text
        ),
        metadata: HashMap::from([
            ("file_type".to_string(), json!("document")),
            ("analysis_type".to_string(), json!("document_text")),
            ("path".to_string(), json!(path.display().to_string())),
            ("size_bytes".to_string(), json!(metadata.len())),
        ]),
    })
}

fn read_svg(path: &Path) -> Result<FileReadOutput, ErrorData> {
    let (_, metadata, text) = load_text_file(path)?;
    let preview = text
        .lines()
        .take(10)
        .enumerate()
        .filter_map(|(index, line)| {
            let line = line.trim();
            (!line.is_empty()).then(|| format!("  {}: {}", index + 1, line))
        })
        .collect::<Vec<_>>()
        .join("\n");
    let output = format!(
        "[SVG File: {}]\nSize: {} bytes\n\nSVG Content Preview:\n{}\n\nFull SVG Content:\n{}",
        path.display(),
        metadata.len(),
        preview,
        text
    );
    Ok(FileReadOutput {
        output,
        metadata: HashMap::from([
            ("file_type".to_string(), json!("svg")),
            ("analysis_type".to_string(), json!("vector_text")),
            ("path".to_string(), json!(path.display().to_string())),
            ("size_bytes".to_string(), json!(metadata.len())),
        ]),
    })
}

fn read_notebook(path: &Path) -> Result<FileReadOutput, ErrorData> {
    let (content, metadata, text) = load_text_file(path)?;
    let notebook: Value = serde_json::from_str(&text).map_err(|error| {
        runtime_read_error(
            ErrorCode::INTERNAL_ERROR,
            format!("Failed to parse notebook JSON: {error}"),
        )
    })?;
    let cells = notebook
        .get("cells")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            runtime_read_error(
                ErrorCode::INTERNAL_ERROR,
                "Invalid notebook format: missing cells",
            )
        })?;
    let mut output = vec![
        format!("[Notebook: {}]", path.display()),
        format!("Size: {} bytes", metadata.len()),
        format!("Total cells: {}", cells.len()),
        String::new(),
    ];
    for (index, cell) in cells.iter().enumerate() {
        output.extend(format_notebook_cell(index + 1, cell));
        output.push(String::new());
    }

    Ok(FileReadOutput {
        output: output.join("\n"),
        metadata: HashMap::from([
            ("file_type".to_string(), json!("notebook")),
            ("analysis_type".to_string(), json!("notebook_text")),
            ("path".to_string(), json!(path.display().to_string())),
            ("size_bytes".to_string(), json!(metadata.len())),
            ("content_bytes".to_string(), json!(content.len())),
            ("cell_count".to_string(), json!(cells.len())),
        ]),
    })
}

fn format_notebook_cell(cell_number: usize, cell: &Value) -> Vec<String> {
    let cell_type = cell
        .get("cell_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let source = cell
        .get("source")
        .map(extract_notebook_text)
        .unwrap_or_default();
    let mut output = vec![format!("## Cell {} [{}]", cell_number, cell_type)];
    if cell_type == "code" {
        output.push("```python".to_string());
        output.push(source);
        output.push("```".to_string());
    } else {
        output.push(source);
    }
    if let Some(outputs) = cell.get("outputs").and_then(Value::as_array) {
        for (index, item) in outputs.iter().enumerate() {
            if let Some(text) = extract_notebook_output_text(item) {
                output.push(format!("### Output {}", index + 1));
                output.push("```".to_string());
                output.push(text);
                output.push("```".to_string());
            }
        }
    }
    output
}

fn extract_notebook_output_text(output: &Value) -> Option<String> {
    output.get("text").map(extract_notebook_text).or_else(|| {
        output
            .get("data")?
            .get("text/plain")
            .map(extract_notebook_text)
    })
}

fn extract_notebook_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(values) => values.iter().filter_map(Value::as_str).collect(),
        _ => String::new(),
    }
}

fn load_text_file(path: &Path) -> Result<(Vec<u8>, fs::Metadata, String), ErrorData> {
    ensure_readable_file(path)?;
    let metadata = fs::metadata(path).map_err(io_error)?;
    if metadata.len() > MAX_TEXT_FILE_SIZE {
        return Err(runtime_read_error(
            ErrorCode::INTERNAL_ERROR,
            format!(
                "File too large: {} bytes (max: {} bytes)",
                metadata.len(),
                MAX_TEXT_FILE_SIZE
            ),
        ));
    }
    let content = fs::read(path).map_err(io_error)?;
    let text = String::from_utf8_lossy(&content).to_string();
    Ok((content, metadata, text))
}

fn ensure_readable_file(path: &Path) -> Result<(), ErrorData> {
    if !path.exists() {
        return Err(runtime_read_error(
            ErrorCode::INTERNAL_ERROR,
            format!("File not found: {}", path.display()),
        ));
    }
    if path.is_dir() {
        return Err(runtime_read_error(
            ErrorCode::INTERNAL_ERROR,
            format!("Path is a directory: {}", path.display()),
        ));
    }
    Ok(())
}

fn format_text_with_lines(text: &str, range: Option<LineRange>) -> String {
    let lines = text.lines().collect::<Vec<_>>();
    let total_lines = lines.len();
    let (start, end) = match range {
        Some(range) => {
            let start = range.start.saturating_sub(1).min(total_lines);
            let end = range
                .end
                .map(|end| end.min(total_lines))
                .unwrap_or(total_lines);
            (start, end)
        }
        None => (0, total_lines),
    };
    let line_width = end.max(1).to_string().len();
    lines[start..end]
        .iter()
        .enumerate()
        .map(|(index, line)| {
            let line_number = start + index + 1;
            format!("{line_number:>line_width$} | {line}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn resolve_path(path: &str, working_directory: &Path) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        working_directory.join(path)
    }
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
}

fn is_svg_file(path: &Path) -> bool {
    extension(path).is_some_and(|extension| extension == "svg")
}

fn is_pdf_file(path: &Path) -> bool {
    extension(path).is_some_and(|extension| extension == "pdf")
}

fn is_notebook_file(path: &Path) -> bool {
    extension(path).is_some_and(|extension| extension == "ipynb")
}

fn is_image_file(path: &Path) -> bool {
    extension(path).is_some_and(|extension| IMAGE_EXTENSIONS.contains(&extension.as_str()))
}

fn categorize_file_type(path: &Path) -> &'static str {
    let Some(extension) = extension(path) else {
        return "Text File";
    };
    if SOURCE_EXTENSIONS.contains(&extension.as_str()) {
        "Source Code"
    } else if CONFIG_EXTENSIONS.contains(&extension.as_str()) {
        "Configuration"
    } else if DOCUMENTATION_EXTENSIONS.contains(&extension.as_str()) {
        "Documentation"
    } else if matches!(
        extension.as_str(),
        "html" | "css" | "scss" | "sass" | "less"
    ) {
        "Web Content"
    } else if matches!(extension.as_str(), "csv" | "tsv" | "log") {
        "Data File"
    } else {
        "Text File"
    }
}

fn detect_programming_language(path: &Path, content: &str) -> Option<&'static str> {
    match extension(path).as_deref() {
        Some("rs") => Some("Rust"),
        Some("py") => Some("Python"),
        Some("js") => Some("JavaScript"),
        Some("ts") => Some("TypeScript"),
        Some("jsx") => Some("React JSX"),
        Some("tsx") => Some("React TSX"),
        Some("java") => Some("Java"),
        Some("c") => Some("C"),
        Some("cpp" | "cc" | "cxx") => Some("C++"),
        Some("h" | "hpp") => Some("C/C++ Header"),
        Some("go") => Some("Go"),
        Some("rb") => Some("Ruby"),
        Some("php") => Some("PHP"),
        Some("swift") => Some("Swift"),
        Some("kt") => Some("Kotlin"),
        Some("scala") => Some("Scala"),
        Some("sh" | "bash" | "zsh") => Some("Shell Script"),
        Some("sql") => Some("SQL"),
        Some("html") => Some("HTML"),
        Some("css") => Some("CSS"),
        Some("scss" | "sass") => Some("SCSS/Sass"),
        Some("xml") => Some("XML"),
        Some("json") => Some("JSON"),
        Some("yaml" | "yml") => Some("YAML"),
        Some("toml") => Some("TOML"),
        Some("md") => Some("Markdown"),
        _ if content.starts_with("#!/bin/bash") || content.starts_with("#!/bin/sh") => {
            Some("Shell Script")
        }
        _ if content.starts_with("#!/usr/bin/env python") => Some("Python"),
        _ if content.starts_with("#!/usr/bin/env node") => Some("JavaScript"),
        _ => None,
    }
}

fn runtime_read_error(code: ErrorCode, message: impl Into<String>) -> ErrorData {
    ErrorData::new(code, message.into(), None)
}

fn io_error(error: std::io::Error) -> ErrorData {
    runtime_read_error(ErrorCode::INTERNAL_ERROR, error.to_string())
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
    async fn reads_line_range_with_line_numbers() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("sample.rs");
        fs::write(&file, "one\ntwo\nthree\n").unwrap();

        let result = execute_runtime_file_read_tool(RuntimeFileReadRequest {
            tool_name: "Read",
            params: &json!({ "path": "sample.rs", "start_line": 2, "end_line": 3 }),
            working_directory: dir.path().to_path_buf(),
            cancel_token: None,
            environment_id: None,
            filesystem_gateway: None,
            sandbox_policy: None,
        })
        .await
        .expect("read tool")
        .expect("read result");

        assert_eq!(result.is_error, Some(false));
        assert_eq!(result_text(&result), "2 | two\n3 | three");
        assert_eq!(
            result.structured_content.as_ref().unwrap().get("file_type"),
            Some(&json!("text"))
        );
    }

    #[tokio::test]
    async fn rejects_retired_image_read_path() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("image.png"), b"png").unwrap();

        let result = execute_runtime_file_read_tool(RuntimeFileReadRequest {
            tool_name: "Read",
            params: &json!({ "path": "image.png" }),
            working_directory: dir.path().to_path_buf(),
            cancel_token: None,
            environment_id: None,
            filesystem_gateway: None,
            sandbox_policy: None,
        })
        .await
        .expect("read tool");

        assert!(result.unwrap_err().message.contains("view_image"));
    }

    #[tokio::test]
    async fn unknown_tool_falls_back_to_registry() {
        let result = execute_runtime_file_read_tool(RuntimeFileReadRequest {
            tool_name: "Grep",
            params: &json!({ "path": "sample.rs" }),
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
    fn definition_and_aliases_are_owned_by_tool_runtime() {
        let definition = file_read_tool_definition();
        assert_eq!(definition.name, FILE_READ_TOOL_NAME);
        assert!(definition.description.contains("view_image"));
        assert_eq!(
            file_read_canonical_tool_name("functions.read_file"),
            Some(FILE_READ_TOOL_NAME)
        );
        assert_eq!(
            file_read_canonical_tool_name("FileReadTool"),
            Some(FILE_READ_TOOL_NAME)
        );
    }
}
