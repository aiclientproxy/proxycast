//! 文件读取工具模块
//!
//! 提供文件内容读取功能，支持行号显示、行范围读取、大文件检测、目录列表和语言检测
//! 符合 Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
//!
//! ## 功能
//! - 文件内容读取（带行号）
//! - 行范围读取（start_line, end_line）
//! - 大文件检测和推荐
//! - 目录列表功能
//! - 语言检测

use super::registry::Tool;
use super::security::SecurityManager;
use super::types::{JsonSchema, PropertySchema, ToolDefinition, ToolError, ToolResult};
use async_trait::async_trait;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// 大文件阈值（行数）
const LARGE_FILE_THRESHOLD: usize = 500;

/// 最大读取行数（无行范围时）
const MAX_LINES_WITHOUT_RANGE: usize = 2000;

/// 文件读取工具
///
/// 读取文件内容并返回带行号的结果
/// Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
pub struct ReadFileTool {
    /// 安全管理器
    security: Arc<SecurityManager>,
}

impl ReadFileTool {
    /// 创建新的文件读取工具
    pub fn new(security: Arc<SecurityManager>) -> Self {
        Self { security }
    }

    /// 读取文件内容
    ///
    /// Requirements: 4.1 - THE File_Reader SHALL return the file content with line numbers
    /// Requirements: 4.2 - THE File_Reader SHALL support reading specific line ranges
    pub fn read_file(
        &self,
        path: &Path,
        start_line: Option<usize>,
        end_line: Option<usize>,
    ) -> Result<ReadFileResult, ToolError> {
        // 验证路径安全性
        let validated_path = self
            .security
            .validate_path(path)
            .map_err(|e| ToolError::Security(e.to_string()))?;

        // 检查文件是否存在
        // Requirements: 4.3 - IF the file does not exist, THEN THE File_Reader SHALL return a clear error message
        if !validated_path.exists() {
            return Err(ToolError::ExecutionFailed(format!(
                "文件不存在: {}",
                path.display()
            )));
        }

        // 检查是否为目录
        // Requirements: 4.6 - IF the path is a directory, THEN THE File_Reader SHALL list the directory contents
        if validated_path.is_dir() {
            return self.list_directory(&validated_path);
        }

        // 读取文件内容
        let content = fs::read_to_string(&validated_path).map_err(|e| {
            ToolError::ExecutionFailed(format!("无法读取文件 {}: {}", path.display(), e))
        })?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        // 检测语言
        // Requirements: 4.5 - THE File_Reader SHALL detect and report the file's programming language
        let language = detect_language(&validated_path);

        // 检查是否为大文件
        // Requirements: 4.4 - WHEN reading a large file without line range, THE File_Reader SHALL recommend using line ranges
        let is_large_file = total_lines > LARGE_FILE_THRESHOLD;
        let recommend_range = is_large_file && start_line.is_none() && end_line.is_none();

        // 计算实际读取范围
        let (actual_start, actual_end) = calculate_line_range(total_lines, start_line, end_line);

        // 提取指定范围的行
        let selected_lines: Vec<(usize, &str)> = lines
            .iter()
            .enumerate()
            .skip(actual_start.saturating_sub(1))
            .take(actual_end - actual_start.saturating_sub(1))
            .map(|(i, line)| (i + 1, *line))
            .collect();

        // 格式化输出（带行号）
        let formatted_content = format_lines_with_numbers(&selected_lines);

        // 检查是否被截断
        let truncated =
            total_lines > MAX_LINES_WITHOUT_RANGE && start_line.is_none() && end_line.is_none();

        Ok(ReadFileResult {
            content: formatted_content,
            total_lines,
            start_line: actual_start,
            end_line: actual_end.min(total_lines),
            language,
            is_directory: false,
            recommend_range,
            truncated,
        })
    }

    /// 列出目录内容
    ///
    /// Requirements: 4.6 - IF the path is a directory, THEN THE File_Reader SHALL list the directory contents
    fn list_directory(&self, path: &Path) -> Result<ReadFileResult, ToolError> {
        let entries = fs::read_dir(path).map_err(|e| {
            ToolError::ExecutionFailed(format!("无法读取目录 {}: {}", path.display(), e))
        })?;

        let mut items: Vec<DirectoryEntry> = Vec::new();

        for entry in entries {
            let entry = entry
                .map_err(|e| ToolError::ExecutionFailed(format!("读取目录条目失败: {}", e)))?;

            let file_name = entry.file_name().to_string_lossy().to_string();
            let file_type = entry
                .file_type()
                .map_err(|e| ToolError::ExecutionFailed(format!("获取文件类型失败: {}", e)))?;

            let metadata = entry.metadata().ok();
            let size = metadata.as_ref().map(|m| m.len());

            items.push(DirectoryEntry {
                name: file_name,
                is_dir: file_type.is_dir(),
                is_symlink: file_type.is_symlink(),
                size,
            });
        }

        // 排序：目录在前，然后按名称排序
        items.sort_by(|a, b| match (a.is_dir, b.is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.cmp(&b.name),
        });

        // 格式化目录列表
        let content = format_directory_listing(&items);

        Ok(ReadFileResult {
            content,
            total_lines: items.len(),
            start_line: 1,
            end_line: items.len(),
            language: None,
            is_directory: true,
            recommend_range: false,
            truncated: false,
        })
    }
}

/// 文件读取结果
#[derive(Debug, Clone)]
pub struct ReadFileResult {
    /// 格式化后的内容（带行号）
    pub content: String,
    /// 文件总行数
    pub total_lines: usize,
    /// 实际读取的起始行
    pub start_line: usize,
    /// 实际读取的结束行
    pub end_line: usize,
    /// 检测到的编程语言
    pub language: Option<String>,
    /// 是否为目录
    pub is_directory: bool,
    /// 是否推荐使用行范围
    pub recommend_range: bool,
    /// 内容是否被截断
    pub truncated: bool,
}

/// 目录条目
#[derive(Debug, Clone)]
struct DirectoryEntry {
    /// 文件/目录名
    name: String,
    /// 是否为目录
    is_dir: bool,
    /// 是否为符号链接
    is_symlink: bool,
    /// 文件大小（字节）
    size: Option<u64>,
}

/// 计算实际的行范围
///
/// Requirements: 4.2 - THE File_Reader SHALL support reading specific line ranges
fn calculate_line_range(
    total_lines: usize,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> (usize, usize) {
    let start = start_line.unwrap_or(1).max(1);
    let end = end_line
        .unwrap_or(total_lines)
        .min(total_lines)
        .min(start + MAX_LINES_WITHOUT_RANGE - 1);

    (start, end.max(start))
}

/// 格式化带行号的内容
///
/// Requirements: 4.1 - THE File_Reader SHALL return the file content with line numbers
fn format_lines_with_numbers(lines: &[(usize, &str)]) -> String {
    if lines.is_empty() {
        return String::new();
    }

    // 计算行号宽度
    let max_line_num = lines.last().map(|(n, _)| *n).unwrap_or(1);
    let width = max_line_num.to_string().len();

    lines
        .iter()
        .map(|(num, line)| format!("{:>width$} | {}", num, line, width = width))
        .collect::<Vec<_>>()
        .join("\n")
}

/// 格式化目录列表
fn format_directory_listing(entries: &[DirectoryEntry]) -> String {
    if entries.is_empty() {
        return "（空目录）".to_string();
    }

    let mut output = String::new();
    output.push_str("目录内容:\n\n");

    for entry in entries {
        let type_indicator = if entry.is_symlink {
            "🔗"
        } else if entry.is_dir {
            "📁"
        } else {
            "📄"
        };

        let size_str = if entry.is_dir {
            String::new()
        } else {
            entry
                .size
                .map(|s| format!(" ({} bytes)", s))
                .unwrap_or_default()
        };

        output.push_str(&format!("{} {}{}\n", type_indicator, entry.name, size_str));
    }

    output
}

/// 检测文件的编程语言
///
/// Requirements: 4.5 - THE File_Reader SHALL detect and report the file's programming language
fn detect_language(path: &Path) -> Option<String> {
    let extension = path.extension()?.to_str()?;

    let language = match extension.to_lowercase().as_str() {
        // Rust
        "rs" => "rust",
        // JavaScript/TypeScript
        "js" => "javascript",
        "jsx" => "javascript",
        "ts" => "typescript",
        "tsx" => "typescript",
        "mjs" => "javascript",
        "cjs" => "javascript",
        // Python
        "py" => "python",
        "pyi" => "python",
        "pyw" => "python",
        // Go
        "go" => "go",
        // Java
        "java" => "java",
        // C/C++
        "c" => "c",
        "h" => "c",
        "cpp" | "cc" | "cxx" => "cpp",
        "hpp" | "hh" | "hxx" => "cpp",
        // C#
        "cs" => "csharp",
        // Ruby
        "rb" => "ruby",
        // PHP
        "php" => "php",
        // Swift
        "swift" => "swift",
        // Kotlin
        "kt" | "kts" => "kotlin",
        // Scala
        "scala" => "scala",
        // Shell
        "sh" | "bash" | "zsh" => "shell",
        "ps1" => "powershell",
        // Web
        "html" | "htm" => "html",
        "css" => "css",
        "scss" | "sass" => "scss",
        "less" => "less",
        // Data formats
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        "xml" => "xml",
        "csv" => "csv",
        // Markdown
        "md" | "markdown" => "markdown",
        // SQL
        "sql" => "sql",
        // Docker
        "dockerfile" => "dockerfile",
        // Config
        "ini" | "cfg" | "conf" => "ini",
        "env" => "dotenv",
        // Other
        "txt" => "plaintext",
        "log" => "log",
        _ => return None,
    };

    Some(language.to_string())
}

#[async_trait]
impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "read_file",
            "Read the contents of a file or list directory contents. Returns file content with \
             line numbers for easy reference. Supports reading specific line ranges for large files. \
             If the path is a directory, lists its contents instead.",
        )
        .with_parameters(
            JsonSchema::new()
                .add_property(
                    "path",
                    PropertySchema::string(
                        "The path to the file or directory to read. Can be relative or absolute.",
                    ),
                    true,
                )
                .add_property(
                    "start_line",
                    PropertySchema::integer(
                        "Optional starting line number (1-indexed). If not specified, starts from line 1.",
                    ),
                    false,
                )
                .add_property(
                    "end_line",
                    PropertySchema::integer(
                        "Optional ending line number (inclusive). If not specified, reads to the end of file.",
                    ),
                    false,
                ),
        )
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult, ToolError> {
        // 解析参数
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments("缺少 path 参数".to_string()))?;

        let start_line = args
            .get("start_line")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let end_line = args
            .get("end_line")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let path = PathBuf::from(path_str);

        info!(
            "[ReadFileTool] 读取文件: {} (行范围: {:?}-{:?})",
            path_str, start_line, end_line
        );

        // 读取文件
        let result = self.read_file(&path, start_line, end_line)?;

        // 构建输出
        let mut output = String::new();

        if result.is_directory {
            output.push_str(&result.content);
        } else {
            // 添加文件信息头
            if let Some(ref lang) = result.language {
                output.push_str(&format!("语言: {}\n", lang));
            }
            output.push_str(&format!(
                "行数: {} (显示: {}-{})\n",
                result.total_lines, result.start_line, result.end_line
            ));

            if result.recommend_range {
                output.push_str(
                    "\n⚠️ 这是一个大文件，建议使用 start_line 和 end_line 参数读取特定范围。\n",
                );
            }

            if result.truncated {
                output.push_str(&format!(
                    "\n⚠️ 文件内容已截断（最多显示 {} 行）。请使用 start_line 和 end_line 参数读取更多内容。\n",
                    MAX_LINES_WITHOUT_RANGE
                ));
            }

            output.push_str("\n");
            output.push_str(&result.content);
        }

        debug!(
            "[ReadFileTool] 读取完成: {} 行",
            result.end_line - result.start_line + 1
        );

        Ok(ToolResult::success(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup_test_tool() -> (ReadFileTool, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let security = Arc::new(SecurityManager::new(temp_dir.path()));
        let tool = ReadFileTool::new(security);
        (tool, temp_dir)
    }

    #[test]
    fn test_tool_definition() {
        let temp_dir = TempDir::new().unwrap();
        let security = Arc::new(SecurityManager::new(temp_dir.path()));
        let tool = ReadFileTool::new(security);
        let def = tool.definition();

        assert_eq!(def.name, "read_file");
        assert!(!def.description.is_empty());
        assert!(def.parameters.required.contains(&"path".to_string()));
    }

    #[test]
    fn test_read_simple_file() {
        let (tool, temp_dir) = setup_test_tool();

        // 创建测试文件
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "Line 1\nLine 2\nLine 3").unwrap();

        let result = tool.read_file(Path::new("test.txt"), None, None);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.total_lines, 3);
        assert_eq!(result.start_line, 1);
        assert_eq!(result.end_line, 3);
        assert!(!result.is_directory);
        assert!(result.content.contains("Line 1"));
        assert!(result.content.contains("Line 2"));
        assert!(result.content.contains("Line 3"));
    }

    #[test]
    fn test_read_file_with_line_numbers() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "First\nSecond\nThird").unwrap();

        let result = tool.read_file(Path::new("test.txt"), None, None).unwrap();

        // 验证行号格式
        assert!(result.content.contains("1 | First"));
        assert!(result.content.contains("2 | Second"));
        assert!(result.content.contains("3 | Third"));
    }

    #[test]
    fn test_read_file_line_range() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        let content = (1..=10)
            .map(|i| format!("Line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&file_path, &content).unwrap();

        // 读取第 3-5 行
        let result = tool
            .read_file(Path::new("test.txt"), Some(3), Some(5))
            .unwrap();

        assert_eq!(result.start_line, 3);
        assert_eq!(result.end_line, 5);
        assert!(result.content.contains("Line 3"));
        assert!(result.content.contains("Line 4"));
        assert!(result.content.contains("Line 5"));
        assert!(!result.content.contains("Line 2"));
        assert!(!result.content.contains("Line 6"));
    }

    #[test]
    fn test_read_nonexistent_file() {
        let (tool, _temp_dir) = setup_test_tool();

        let result = tool.read_file(Path::new("nonexistent.txt"), None, None);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
    }

    #[test]
    fn test_read_directory() {
        let (tool, temp_dir) = setup_test_tool();

        // 创建一些文件和目录
        let sub_dir = temp_dir.path().join("subdir");
        fs::create_dir(&sub_dir).unwrap();
        fs::write(temp_dir.path().join("file1.txt"), "content").unwrap();
        fs::write(temp_dir.path().join("file2.rs"), "fn main() {}").unwrap();

        // 读取目录
        let result = tool.read_file(Path::new("."), None, None).unwrap();

        assert!(result.is_directory);
        assert!(result.content.contains("subdir"));
        assert!(result.content.contains("file1.txt"));
        assert!(result.content.contains("file2.rs"));
    }

    #[test]
    fn test_language_detection() {
        // Rust
        assert_eq!(
            detect_language(Path::new("main.rs")),
            Some("rust".to_string())
        );
        // TypeScript
        assert_eq!(
            detect_language(Path::new("app.tsx")),
            Some("typescript".to_string())
        );
        // Python
        assert_eq!(
            detect_language(Path::new("script.py")),
            Some("python".to_string())
        );
        // JSON
        assert_eq!(
            detect_language(Path::new("config.json")),
            Some("json".to_string())
        );
        // Unknown
        assert_eq!(detect_language(Path::new("file.xyz")), None);
        // No extension
        assert_eq!(detect_language(Path::new("Makefile")), None);
    }

    #[test]
    fn test_large_file_recommendation() {
        let (tool, temp_dir) = setup_test_tool();

        // 创建大文件
        let file_path = temp_dir.path().join("large.txt");
        let content = (1..=600)
            .map(|i| format!("Line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&file_path, &content).unwrap();

        // 不指定行范围读取
        let result = tool.read_file(Path::new("large.txt"), None, None).unwrap();

        assert!(result.recommend_range);
        assert!(result.total_lines > LARGE_FILE_THRESHOLD);
    }

    #[test]
    fn test_line_range_boundary() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "Line 1\nLine 2\nLine 3").unwrap();

        // 起始行超出范围
        let result = tool
            .read_file(Path::new("test.txt"), Some(10), None)
            .unwrap();
        assert!(result.content.is_empty() || result.start_line <= result.total_lines);

        // 结束行超出范围
        let result = tool
            .read_file(Path::new("test.txt"), Some(1), Some(100))
            .unwrap();
        assert_eq!(result.end_line, 3);
    }

    #[test]
    fn test_format_lines_with_numbers() {
        let lines = vec![(1, "First"), (2, "Second"), (10, "Tenth")];
        let formatted = format_lines_with_numbers(&lines);

        assert!(formatted.contains(" 1 | First"));
        assert!(formatted.contains(" 2 | Second"));
        assert!(formatted.contains("10 | Tenth"));
    }

    #[test]
    fn test_calculate_line_range() {
        // 默认范围
        let (start, end) = calculate_line_range(100, None, None);
        assert_eq!(start, 1);
        assert!(end <= 100);

        // 指定起始行
        let (start, end) = calculate_line_range(100, Some(50), None);
        assert_eq!(start, 50);

        // 指定结束行
        let (start, end) = calculate_line_range(100, None, Some(30));
        assert_eq!(start, 1);
        assert_eq!(end, 30);

        // 起始行为 0（应该修正为 1）
        let (start, _) = calculate_line_range(100, Some(0), None);
        assert_eq!(start, 1);
    }

    #[tokio::test]
    async fn test_tool_execute() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "Hello, World!").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "path": "test.txt"
            }))
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert!(result.output.contains("Hello, World!"));
    }

    #[tokio::test]
    async fn test_tool_execute_with_line_range() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        let content = (1..=10)
            .map(|i| format!("Line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&file_path, &content).unwrap();

        let result = tool
            .execute(serde_json::json!({
                "path": "test.txt",
                "start_line": 3,
                "end_line": 5
            }))
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert!(result.output.contains("Line 3"));
        assert!(result.output.contains("Line 5"));
    }

    #[tokio::test]
    async fn test_tool_execute_missing_path() {
        let (tool, _temp_dir) = setup_test_tool();

        let result = tool.execute(serde_json::json!({})).await;

        assert!(result.is_err());
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use std::fs;
    use tempfile::TempDir;

    /// 生成有效的文件内容（多行，使用唯一标识符避免内容重复）
    fn arb_file_lines() -> impl Strategy<Value = Vec<String>> {
        prop::collection::vec("[a-zA-Z0-9]{5,20}", 1..100)
    }

    /// 生成有效的行范围
    fn arb_line_range(max_lines: usize) -> impl Strategy<Value = (Option<usize>, Option<usize>)> {
        let max = max_lines.max(1);
        prop_oneof![
            // 无范围
            Just((None, None)),
            // 只有起始行
            (1..=max).prop_map(|s| (Some(s), None)),
            // 只有结束行
            (1..=max).prop_map(|e| (None, Some(e))),
            // 完整范围
            (1..=max, 1..=max).prop_map(|(s, e)| {
                let (start, end) = if s <= e { (s, e) } else { (e, s) };
                (Some(start), Some(end))
            }),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: agent-tool-calling, Property 7: 文件行范围读取**
        /// **Validates: Requirements 4.2**
        ///
        /// *For any* 多行文件和有效的行范围 [start, end]，read_file 返回的内容
        /// 应该恰好包含第 start 到第 end 行。
        #[test]
        fn prop_file_line_range_read(lines in arb_file_lines()) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = ReadFileTool::new(security);

            // 创建测试文件，使用唯一标识符确保每行内容唯一
            let unique_lines: Vec<String> = lines.iter()
                .enumerate()
                .map(|(i, content)| format!("LINE{}_{}", i + 1, content))
                .collect();

            let file_path = temp_dir.path().join("test.txt");
            let content = unique_lines.join("\n");
            fs::write(&file_path, &content).unwrap();

            let total_lines = unique_lines.len();

            // 生成随机行范围
            let start = if total_lines > 1 {
                (rand::random::<usize>() % total_lines) + 1
            } else {
                1
            };
            let end = if total_lines > start {
                start + (rand::random::<usize>() % (total_lines - start + 1))
            } else {
                start
            };

            // 读取指定范围
            let result = tool.read_file(Path::new("test.txt"), Some(start), Some(end));

            prop_assert!(result.is_ok(), "读取文件应该成功");
            let result = result.unwrap();

            // 验证返回的行范围
            prop_assert_eq!(
                result.start_line, start,
                "起始行应该匹配: expected {}, got {}",
                start, result.start_line
            );

            // 结束行应该是 min(end, total_lines)
            let expected_end = end.min(total_lines);
            prop_assert_eq!(
                result.end_line, expected_end,
                "结束行应该匹配: expected {}, got {}",
                expected_end, result.end_line
            );

            // 验证内容包含正确的行（使用唯一标识符）
            for i in start..=expected_end {
                if i <= unique_lines.len() {
                    let unique_marker = format!("LINE{}_", i);
                    prop_assert!(
                        result.content.contains(&unique_marker),
                        "内容应该包含第 {} 行的唯一标识符: '{}'",
                        i, unique_marker
                    );
                }
            }

            // 验证内容不包含范围外的行（使用唯一标识符）
            for i in 1..start {
                if i <= unique_lines.len() {
                    let unique_marker = format!("LINE{}_", i);
                    prop_assert!(
                        !result.content.contains(&unique_marker),
                        "内容不应该包含第 {} 行的唯一标识符: '{}'",
                        i, unique_marker
                    );
                }
            }

            // 验证范围后的行也不应该出现
            for i in (expected_end + 1)..=total_lines {
                let unique_marker = format!("LINE{}_", i);
                prop_assert!(
                    !result.content.contains(&unique_marker),
                    "内容不应该包含第 {} 行的唯一标识符: '{}'",
                    i, unique_marker
                );
            }
        }

        /// **Feature: agent-tool-calling, Property 7: 文件行范围读取 - 行数正确**
        /// **Validates: Requirements 4.2**
        ///
        /// *For any* 文件和行范围，返回的行数应该等于 end - start + 1（或文件实际行数）。
        #[test]
        fn prop_file_line_range_count(lines in arb_file_lines()) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = ReadFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            let content = lines.join("\n");
            fs::write(&file_path, &content).unwrap();

            let total_lines = lines.len();

            // 生成随机行范围
            let start = if total_lines > 1 {
                (rand::random::<usize>() % total_lines) + 1
            } else {
                1
            };
            let end = if total_lines > start {
                start + (rand::random::<usize>() % (total_lines - start + 1))
            } else {
                start
            };

            let result = tool.read_file(Path::new("test.txt"), Some(start), Some(end)).unwrap();

            // 计算预期的行数
            let expected_count = (result.end_line - result.start_line + 1).min(total_lines);

            // 统计实际返回的行数（通过计算行数）
            let actual_count = result.content.lines().count();

            prop_assert_eq!(
                actual_count, expected_count,
                "返回的行数应该匹配: expected {}, got {}",
                expected_count, actual_count
            );
        }

        /// **Feature: agent-tool-calling, Property 7: 文件行范围读取 - 边界处理**
        /// **Validates: Requirements 4.2**
        ///
        /// *For any* 超出文件范围的行号，read_file 应该正确处理边界情况。
        #[test]
        fn prop_file_line_range_boundary(lines in arb_file_lines()) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = ReadFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            let content = lines.join("\n");
            fs::write(&file_path, &content).unwrap();

            let total_lines = lines.len();

            // 测试超出范围的结束行
            let result = tool.read_file(
                Path::new("test.txt"),
                Some(1),
                Some(total_lines + 100)
            ).unwrap();

            prop_assert!(
                result.end_line <= total_lines,
                "结束行不应该超过文件总行数: end_line={}, total_lines={}",
                result.end_line, total_lines
            );

            // 测试起始行为 0（应该修正为 1）
            let result = tool.read_file(
                Path::new("test.txt"),
                Some(0),
                None
            ).unwrap();

            prop_assert!(
                result.start_line >= 1,
                "起始行应该至少为 1: start_line={}",
                result.start_line
            );
        }

        /// **Feature: agent-tool-calling, Property 7: 文件行范围读取 - 总行数一致**
        /// **Validates: Requirements 4.1, 4.2**
        ///
        /// *For any* 文件，无论读取什么范围，total_lines 应该始终等于文件的实际行数。
        #[test]
        fn prop_file_total_lines_consistent(lines in arb_file_lines()) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = ReadFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            let content = lines.join("\n");
            fs::write(&file_path, &content).unwrap();

            let expected_total = lines.len();

            // 读取不同范围，total_lines 应该一致
            let result1 = tool.read_file(Path::new("test.txt"), None, None).unwrap();
            let result2 = tool.read_file(Path::new("test.txt"), Some(1), Some(5)).unwrap();
            let result3 = tool.read_file(Path::new("test.txt"), Some(3), None).unwrap();

            prop_assert_eq!(
                result1.total_lines, expected_total,
                "total_lines 应该等于文件实际行数"
            );
            prop_assert_eq!(
                result2.total_lines, expected_total,
                "total_lines 应该等于文件实际行数（即使指定了范围）"
            );
            prop_assert_eq!(
                result3.total_lines, expected_total,
                "total_lines 应该等于文件实际行数（即使指定了起始行）"
            );
        }

        /// **Feature: agent-tool-calling, Property 7: 文件行范围读取 - 行号格式正确**
        /// **Validates: Requirements 4.1**
        ///
        /// *For any* 文件内容，返回的每一行都应该有正确的行号前缀。
        #[test]
        fn prop_file_line_numbers_format(lines in arb_file_lines()) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = ReadFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            let content = lines.join("\n");
            fs::write(&file_path, &content).unwrap();

            let result = tool.read_file(Path::new("test.txt"), None, None).unwrap();

            // 验证每一行都有行号格式
            for (i, line) in result.content.lines().enumerate() {
                let line_num = result.start_line + i;
                prop_assert!(
                    line.contains(" | "),
                    "每一行应该包含 ' | ' 分隔符: line {}",
                    line_num
                );

                // 验证行号在分隔符之前
                let parts: Vec<&str> = line.splitn(2, " | ").collect();
                prop_assert!(
                    parts.len() == 2,
                    "行应该被 ' | ' 分成两部分"
                );

                let num_str = parts[0].trim();
                let parsed_num: Result<usize, _> = num_str.parse();
                prop_assert!(
                    parsed_num.is_ok(),
                    "行号应该是有效的数字: '{}'",
                    num_str
                );
            }
        }
    }
}
