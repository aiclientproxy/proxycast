//! 文件写入工具模块
//!
//! 提供文件创建和写入功能，支持父目录自动创建、换行符规范化、尾部换行符保证
//! 符合 Requirements 5.1, 5.2, 5.3, 5.4, 5.5
//!
//! ## 功能
//! - 文件创建/覆盖
//! - 父目录自动创建
//! - 换行符规范化（Unix: LF, Windows: CRLF）
//! - 尾部换行符保证

use super::registry::Tool;
use super::security::SecurityManager;
use super::types::{JsonSchema, PropertySchema, ToolDefinition, ToolError, ToolResult};
use async_trait::async_trait;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// 文件写入工具
///
/// 创建或覆盖文件，支持自动创建父目录和换行符规范化
/// Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
pub struct WriteFileTool {
    /// 安全管理器
    security: Arc<SecurityManager>,
}

impl WriteFileTool {
    /// 创建新的文件写入工具
    pub fn new(security: Arc<SecurityManager>) -> Self {
        Self { security }
    }

    /// 写入文件内容
    ///
    /// Requirements: 5.1 - THE File_Writer SHALL create or overwrite the file with the provided content
    /// Requirements: 5.2 - THE File_Writer SHALL create parent directories if they do not exist
    /// Requirements: 5.3 - THE File_Writer SHALL normalize line endings based on the platform
    /// Requirements: 5.4 - THE File_Writer SHALL ensure files end with a trailing newline
    pub fn write_file(&self, path: &Path, content: &str) -> Result<WriteFileResult, ToolError> {
        // 验证路径安全性（不检查符号链接，因为文件可能不存在）
        let validated_path = self
            .security
            .validate_path_no_symlink_check(path)
            .map_err(|e| ToolError::Security(e.to_string()))?;

        // 规范化换行符
        // Requirements: 5.3 - THE File_Writer SHALL normalize line endings based on the platform
        let normalized_content = normalize_line_endings(content);

        // 确保尾部换行符
        // Requirements: 5.4 - THE File_Writer SHALL ensure files end with a trailing newline
        let final_content = ensure_trailing_newline(&normalized_content);

        // 创建父目录
        // Requirements: 5.2 - THE File_Writer SHALL create parent directories if they do not exist
        if let Some(parent) = validated_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    ToolError::ExecutionFailed(format!(
                        "无法创建父目录 {}: {}",
                        parent.display(),
                        e
                    ))
                })?;
                debug!("[WriteFileTool] 创建父目录: {:?}", parent);
            }
        }

        // 检查文件是否已存在（用于返回结果）
        let file_existed = validated_path.exists();

        // 写入文件
        // Requirements: 5.1 - THE File_Writer SHALL create or overwrite the file
        // Requirements: 5.5 - IF the write operation fails, THEN THE File_Writer SHALL return a descriptive error message
        fs::write(&validated_path, &final_content).map_err(|e| {
            ToolError::ExecutionFailed(format!("无法写入文件 {}: {}", path.display(), e))
        })?;

        let bytes_written = final_content.len();
        let line_count = final_content.lines().count();

        info!(
            "[WriteFileTool] 写入文件: {} ({} 字节, {} 行, {})",
            path.display(),
            bytes_written,
            line_count,
            if file_existed { "覆盖" } else { "新建" }
        );

        Ok(WriteFileResult {
            path: validated_path,
            bytes_written,
            line_count,
            created: !file_existed,
            overwritten: file_existed,
        })
    }
}

/// 文件写入结果
#[derive(Debug, Clone)]
pub struct WriteFileResult {
    /// 写入的文件路径
    pub path: PathBuf,
    /// 写入的字节数
    pub bytes_written: usize,
    /// 写入的行数
    pub line_count: usize,
    /// 是否为新创建的文件
    pub created: bool,
    /// 是否覆盖了已有文件
    pub overwritten: bool,
}

/// 规范化换行符
///
/// Requirements: 5.3 - THE File_Writer SHALL normalize line endings based on the platform
/// - Unix/macOS: LF (\n)
/// - Windows: CRLF (\r\n)
fn normalize_line_endings(content: &str) -> String {
    // 首先将所有换行符统一为 LF
    let unified = content
        .replace("\r\n", "\n") // CRLF -> LF
        .replace("\r", "\n"); // CR -> LF

    // 根据平台转换
    #[cfg(windows)]
    {
        // Windows: LF -> CRLF
        unified.replace("\n", "\r\n")
    }

    #[cfg(not(windows))]
    {
        // Unix/macOS: 保持 LF
        unified
    }
}

/// 确保内容以换行符结尾
///
/// Requirements: 5.4 - THE File_Writer SHALL ensure files end with a trailing newline
fn ensure_trailing_newline(content: &str) -> String {
    if content.is_empty() {
        return String::new();
    }

    #[cfg(windows)]
    {
        if content.ends_with("\r\n") {
            content.to_string()
        } else if content.ends_with('\n') {
            // 已有 LF，转换为 CRLF
            let mut result = content[..content.len() - 1].to_string();
            result.push_str("\r\n");
            result
        } else {
            let mut result = content.to_string();
            result.push_str("\r\n");
            result
        }
    }

    #[cfg(not(windows))]
    {
        if content.ends_with('\n') {
            content.to_string()
        } else {
            let mut result = content.to_string();
            result.push('\n');
            result
        }
    }
}

/// 获取平台的换行符
#[allow(dead_code)]
fn platform_line_ending() -> &'static str {
    #[cfg(windows)]
    {
        "\r\n"
    }

    #[cfg(not(windows))]
    {
        "\n"
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "write_file",
            "Create a new file or overwrite an existing file with the provided content. \
             Automatically creates parent directories if they don't exist. \
             Line endings are normalized based on the platform (LF for Unix, CRLF for Windows). \
             Files are guaranteed to end with a trailing newline.",
        )
        .with_parameters(
            JsonSchema::new()
                .add_property(
                    "path",
                    PropertySchema::string(
                        "The path to the file to write. Can be relative or absolute.",
                    ),
                    true,
                )
                .add_property(
                    "content",
                    PropertySchema::string("The content to write to the file."),
                    true,
                ),
        )
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult, ToolError> {
        // 解析参数
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments("缺少 path 参数".to_string()))?;

        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments("缺少 content 参数".to_string()))?;

        let path = PathBuf::from(path_str);

        info!("[WriteFileTool] 写入文件: {}", path_str);

        // 写入文件
        let result = self.write_file(&path, content)?;

        // 构建输出
        let action = if result.created { "创建" } else { "覆盖" };
        let output = format!(
            "成功{}文件: {}\n写入 {} 字节, {} 行",
            action, path_str, result.bytes_written, result.line_count
        );

        debug!("[WriteFileTool] 写入完成: {}", output);

        Ok(ToolResult::success(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup_test_tool() -> (WriteFileTool, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let security = Arc::new(SecurityManager::new(temp_dir.path()));
        let tool = WriteFileTool::new(security);
        (tool, temp_dir)
    }

    #[test]
    fn test_tool_definition() {
        let temp_dir = TempDir::new().unwrap();
        let security = Arc::new(SecurityManager::new(temp_dir.path()));
        let tool = WriteFileTool::new(security);
        let def = tool.definition();

        assert_eq!(def.name, "write_file");
        assert!(!def.description.is_empty());
        assert!(def.parameters.required.contains(&"path".to_string()));
        assert!(def.parameters.required.contains(&"content".to_string()));
    }

    #[test]
    fn test_write_new_file() {
        let (tool, temp_dir) = setup_test_tool();

        let result = tool.write_file(Path::new("test.txt"), "Hello, World!");
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.created);
        assert!(!result.overwritten);

        // 验证文件内容
        let file_path = temp_dir.path().join("test.txt");
        let content = fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("Hello, World!"));
        assert!(content.ends_with('\n') || content.ends_with("\r\n"));
    }

    #[test]
    fn test_overwrite_existing_file() {
        let (tool, temp_dir) = setup_test_tool();

        // 创建初始文件
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "Original content").unwrap();

        // 覆盖文件
        let result = tool.write_file(Path::new("test.txt"), "New content");
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(!result.created);
        assert!(result.overwritten);

        // 验证文件内容已更新
        let content = fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("New content"));
        assert!(!content.contains("Original"));
    }

    #[test]
    fn test_create_parent_directories() {
        let (tool, temp_dir) = setup_test_tool();

        // 写入嵌套目录中的文件
        let result = tool.write_file(Path::new("a/b/c/test.txt"), "Nested content");
        assert!(result.is_ok());

        // 验证目录和文件都已创建
        let file_path = temp_dir.path().join("a/b/c/test.txt");
        assert!(file_path.exists());
        let content = fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("Nested content"));
    }

    #[test]
    fn test_trailing_newline() {
        let (tool, temp_dir) = setup_test_tool();

        // 写入不带换行符的内容
        let result = tool.write_file(Path::new("test.txt"), "No newline");
        assert!(result.is_ok());

        // 验证文件以换行符结尾
        let file_path = temp_dir.path().join("test.txt");
        let content = fs::read_to_string(&file_path).unwrap();

        #[cfg(windows)]
        assert!(content.ends_with("\r\n"));

        #[cfg(not(windows))]
        assert!(content.ends_with('\n'));
    }

    #[test]
    fn test_normalize_line_endings_crlf_to_lf() {
        let input = "Line 1\r\nLine 2\r\nLine 3";
        let result = normalize_line_endings(input);

        #[cfg(windows)]
        {
            assert!(result.contains("\r\n"));
            assert!(!result.contains("\r\n\r\n")); // 不应该有双换行
        }

        #[cfg(not(windows))]
        {
            assert!(!result.contains("\r\n"));
            assert!(result.contains('\n'));
        }
    }

    #[test]
    fn test_normalize_line_endings_mixed() {
        let input = "Line 1\r\nLine 2\nLine 3\rLine 4";
        let result = normalize_line_endings(input);

        // 所有换行符应该被统一
        let line_count = result.lines().count();
        assert_eq!(line_count, 4);
    }

    #[test]
    fn test_empty_content() {
        let (tool, temp_dir) = setup_test_tool();

        let result = tool.write_file(Path::new("empty.txt"), "");
        assert!(result.is_ok());

        let file_path = temp_dir.path().join("empty.txt");
        let content = fs::read_to_string(&file_path).unwrap();
        assert!(content.is_empty());
    }

    #[test]
    fn test_security_path_traversal() {
        let (tool, _temp_dir) = setup_test_tool();

        // 尝试路径遍历攻击
        let result = tool.write_file(Path::new("../../../etc/passwd"), "malicious");
        assert!(result.is_err());
        assert!(matches!(result, Err(ToolError::Security(_))));
    }

    #[test]
    fn test_bytes_and_line_count() {
        let (tool, _temp_dir) = setup_test_tool();

        let content = "Line 1\nLine 2\nLine 3";
        let result = tool.write_file(Path::new("test.txt"), content).unwrap();

        assert_eq!(result.line_count, 3);
        assert!(result.bytes_written > 0);
    }

    #[tokio::test]
    async fn test_tool_execute() {
        let (tool, temp_dir) = setup_test_tool();

        let result = tool
            .execute(serde_json::json!({
                "path": "test.txt",
                "content": "Hello from execute!"
            }))
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert!(result.output.contains("成功"));

        // 验证文件已创建
        let file_path = temp_dir.path().join("test.txt");
        assert!(file_path.exists());
    }

    #[tokio::test]
    async fn test_tool_execute_missing_path() {
        let (tool, _temp_dir) = setup_test_tool();

        let result = tool
            .execute(serde_json::json!({
                "content": "Some content"
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn test_tool_execute_missing_content() {
        let (tool, _temp_dir) = setup_test_tool();

        let result = tool
            .execute(serde_json::json!({
                "path": "test.txt"
            }))
            .await;

        assert!(result.is_err());
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[test]
    fn test_ensure_trailing_newline() {
        // 已有换行符
        let with_newline = "content\n";
        let result = ensure_trailing_newline(with_newline);

        #[cfg(windows)]
        assert!(result.ends_with("\r\n"));

        #[cfg(not(windows))]
        assert!(result.ends_with('\n'));

        // 无换行符
        let without_newline = "content";
        let result = ensure_trailing_newline(without_newline);

        #[cfg(windows)]
        assert!(result.ends_with("\r\n"));

        #[cfg(not(windows))]
        assert!(result.ends_with('\n'));

        // 空内容
        let empty = "";
        let result = ensure_trailing_newline(empty);
        assert!(result.is_empty());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::agent::tools::read_file::ReadFileTool;
    use proptest::prelude::*;
    use std::fs;
    use tempfile::TempDir;

    /// 生成有效的文件内容（多行文本）
    fn arb_file_content() -> impl Strategy<Value = String> {
        prop::collection::vec("[a-zA-Z0-9 ,.!?]{1,100}", 1..50).prop_map(|lines| lines.join("\n"))
    }

    /// 生成有效的文件名
    fn arb_valid_filename() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9_-]{0,20}\\.[a-z]{1,4}"
    }

    /// 生成包含各种换行符的内容
    fn arb_content_with_mixed_line_endings() -> impl Strategy<Value = String> {
        prop::collection::vec("[a-zA-Z0-9 ,.!?]{1,50}", 1..20).prop_map(|lines| {
            // 随机使用不同的换行符
            let mut result = String::new();
            for (i, line) in lines.iter().enumerate() {
                result.push_str(line);
                match i % 3 {
                    0 => result.push('\n'),       // LF
                    1 => result.push_str("\r\n"), // CRLF
                    _ => result.push('\r'),       // CR
                }
            }
            result
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: agent-tool-calling, Property 6: 文件读写 Round-Trip**
        /// **Validates: Requirements 4.1, 5.1**
        ///
        /// *For any* 有效的文件内容，使用 write_file 写入后再使用 read_file 读取，
        /// 应该得到等价的内容（考虑换行符规范化）。
        #[test]
        fn prop_file_write_read_roundtrip(content in arb_file_content()) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let write_tool = WriteFileTool::new(security.clone());
            let read_tool = ReadFileTool::new(security);

            let filename = "roundtrip_test.txt";

            // 写入文件
            let write_result = write_tool.write_file(Path::new(filename), &content);
            prop_assert!(
                write_result.is_ok(),
                "写入文件应该成功: {:?}",
                write_result.err()
            );

            // 读取文件
            let read_result = read_tool.read_file(Path::new(filename), None, None);
            prop_assert!(
                read_result.is_ok(),
                "读取文件应该成功: {:?}",
                read_result.err()
            );

            let read_result = read_result.unwrap();

            // 提取实际内容（去除行号格式）
            let read_lines: Vec<&str> = read_result.content
                .lines()
                .map(|line| {
                    // 格式: "  N | content"，提取 | 后面的内容
                    if let Some(pos) = line.find(" | ") {
                        &line[pos + 3..]
                    } else {
                        line
                    }
                })
                .collect();

            // 规范化原始内容进行比较
            let normalized_original = normalize_line_endings(&content);
            let original_lines: Vec<&str> = normalized_original.lines().collect();

            // 比较行数
            prop_assert_eq!(
                read_lines.len(),
                original_lines.len(),
                "读取的行数应该与写入的行数相同"
            );

            // 比较每一行的内容
            for (i, (read_line, orig_line)) in read_lines.iter().zip(original_lines.iter()).enumerate() {
                prop_assert_eq!(
                    *read_line,
                    *orig_line,
                    "第 {} 行内容应该匹配",
                    i + 1
                );
            }
        }

        /// **Feature: agent-tool-calling, Property 6: 文件读写 Round-Trip - 字节级验证**
        /// **Validates: Requirements 4.1, 5.1**
        ///
        /// *For any* 有效的文件内容，写入后直接读取文件字节，
        /// 应该得到规范化后的内容加上尾部换行符。
        #[test]
        fn prop_file_write_read_bytes_roundtrip(content in arb_file_content()) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let write_tool = WriteFileTool::new(security);

            let file_path = temp_dir.path().join("bytes_test.txt");

            // 写入文件
            let write_result = write_tool.write_file(Path::new("bytes_test.txt"), &content);
            prop_assert!(write_result.is_ok());

            // 直接读取文件字节
            let read_bytes = fs::read_to_string(&file_path).unwrap();

            // 计算预期内容
            let normalized = normalize_line_endings(&content);
            let expected = ensure_trailing_newline(&normalized);

            prop_assert_eq!(
                read_bytes,
                expected,
                "文件内容应该是规范化后的内容加尾部换行符"
            );
        }

        /// **Feature: agent-tool-calling, Property 6: 文件读写 Round-Trip - 多次写入**
        /// **Validates: Requirements 5.1**
        ///
        /// *For any* 两个不同的内容，第二次写入应该完全覆盖第一次的内容。
        #[test]
        fn prop_file_overwrite_roundtrip(
            content1 in arb_file_content(),
            content2 in arb_file_content()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let write_tool = WriteFileTool::new(security);

            let file_path = temp_dir.path().join("overwrite_test.txt");

            // 第一次写入
            let result1 = write_tool.write_file(Path::new("overwrite_test.txt"), &content1);
            prop_assert!(result1.is_ok());
            prop_assert!(result1.unwrap().created);

            // 第二次写入（覆盖）
            let result2 = write_tool.write_file(Path::new("overwrite_test.txt"), &content2);
            prop_assert!(result2.is_ok());
            prop_assert!(result2.unwrap().overwritten);

            // 验证文件内容是第二次写入的内容
            let read_bytes = fs::read_to_string(&file_path).unwrap();
            let expected = ensure_trailing_newline(&normalize_line_endings(&content2));

            prop_assert_eq!(
                read_bytes,
                expected,
                "文件内容应该是第二次写入的内容"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: agent-tool-calling, Property 8: 文件换行符规范化**
        /// **Validates: Requirements 5.3, 5.4**
        ///
        /// *For any* 写入的文件内容，最终文件应该以换行符结尾，
        /// 且换行符符合平台规范（Unix: LF, Windows: CRLF）。
        #[test]
        fn prop_file_trailing_newline(content in arb_file_content()) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let write_tool = WriteFileTool::new(security);

            let file_path = temp_dir.path().join("newline_test.txt");

            // 写入文件
            let result = write_tool.write_file(Path::new("newline_test.txt"), &content);
            prop_assert!(result.is_ok());

            // 读取文件内容
            let read_bytes = fs::read_to_string(&file_path).unwrap();

            // 验证尾部换行符
            if !content.is_empty() {
                #[cfg(windows)]
                prop_assert!(
                    read_bytes.ends_with("\r\n"),
                    "Windows 平台文件应该以 CRLF 结尾"
                );

                #[cfg(not(windows))]
                prop_assert!(
                    read_bytes.ends_with('\n'),
                    "Unix 平台文件应该以 LF 结尾"
                );
            }
        }

        /// **Feature: agent-tool-calling, Property 8: 文件换行符规范化 - 混合换行符**
        /// **Validates: Requirements 5.3**
        ///
        /// *For any* 包含混合换行符（LF, CRLF, CR）的内容，
        /// 写入后所有换行符应该被统一为平台规范的换行符。
        #[test]
        fn prop_file_normalize_mixed_line_endings(content in arb_content_with_mixed_line_endings()) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let write_tool = WriteFileTool::new(security);

            let file_path = temp_dir.path().join("mixed_newline_test.txt");

            // 写入文件
            let result = write_tool.write_file(Path::new("mixed_newline_test.txt"), &content);
            prop_assert!(result.is_ok());

            // 读取文件内容
            let read_bytes = fs::read_to_string(&file_path).unwrap();

            // 验证换行符已被规范化
            #[cfg(windows)]
            {
                // Windows: 不应该有单独的 LF 或 CR
                let without_crlf = read_bytes.replace("\r\n", "");
                prop_assert!(
                    !without_crlf.contains('\n') && !without_crlf.contains('\r'),
                    "Windows 平台所有换行符应该是 CRLF"
                );
            }

            #[cfg(not(windows))]
            {
                // Unix: 不应该有 CR
                prop_assert!(
                    !read_bytes.contains('\r'),
                    "Unix 平台不应该有 CR 字符"
                );
            }
        }

        /// **Feature: agent-tool-calling, Property 8: 文件换行符规范化 - 行数保持**
        /// **Validates: Requirements 5.3**
        ///
        /// *For any* 内容，规范化换行符后行数应该保持不变。
        #[test]
        fn prop_file_line_count_preserved(content in arb_content_with_mixed_line_endings()) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let write_tool = WriteFileTool::new(security);

            let file_path = temp_dir.path().join("line_count_test.txt");

            // 计算原始行数（统一换行符后）
            let unified = content
                .replace("\r\n", "\n")
                .replace('\r', "\n");
            let original_line_count = unified.lines().count();

            // 写入文件
            let result = write_tool.write_file(Path::new("line_count_test.txt"), &content);
            prop_assert!(result.is_ok());

            // 读取文件并计算行数
            let read_bytes = fs::read_to_string(&file_path).unwrap();
            let read_line_count = read_bytes.lines().count();

            prop_assert_eq!(
                read_line_count,
                original_line_count,
                "规范化后行数应该保持不变"
            );
        }

        /// **Feature: agent-tool-calling, Property 8: 文件换行符规范化 - 空内容**
        /// **Validates: Requirements 5.4**
        ///
        /// *For any* 空内容，写入后文件应该为空（不添加换行符）。
        #[test]
        fn prop_empty_content_no_newline(_dummy in Just(())) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let write_tool = WriteFileTool::new(security);

            let file_path = temp_dir.path().join("empty_test.txt");

            // 写入空内容
            let result = write_tool.write_file(Path::new("empty_test.txt"), "");
            prop_assert!(result.is_ok());

            // 读取文件内容
            let read_bytes = fs::read_to_string(&file_path).unwrap();

            prop_assert!(
                read_bytes.is_empty(),
                "空内容写入后文件应该为空"
            );
        }
    }
}
