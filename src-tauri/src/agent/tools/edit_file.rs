//! 文件编辑工具模块
//!
//! 提供文件精确编辑功能，支持字符串替换、多次出现检测、unified diff、历史栈和撤销
//! 符合 Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
//!
//! ## 功能
//! - 精确字符串替换（old_str → new_str）
//! - 多次出现检测（返回错误要求更多上下文）
//! - 不存在检测（返回错误和指导）
//! - Unified diff 支持
//! - 历史栈和撤销功能
//! - 返回变更上下文片段

use super::registry::Tool;
use super::security::SecurityManager;
use super::types::{JsonSchema, PropertySchema, ToolDefinition, ToolError, ToolResult};
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// 上下文行数（显示变更前后的行数）
const CONTEXT_LINES: usize = 3;

/// 最大历史记录数
const MAX_HISTORY_SIZE: usize = 100;

/// 文件编辑工具
///
/// 提供精确的文件编辑功能，支持撤销操作
/// Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
pub struct EditFileTool {
    /// 安全管理器
    security: Arc<SecurityManager>,
    /// 编辑历史栈（文件路径 -> 历史记录列表）
    history: Arc<RwLock<HashMap<PathBuf, Vec<EditHistory>>>>,
}

impl EditFileTool {
    /// 创建新的文件编辑工具
    pub fn new(security: Arc<SecurityManager>) -> Self {
        Self {
            security,
            history: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 编辑文件（精确字符串替换）
    ///
    /// Requirements: 6.1 - WHEN the Agent calls the edit_file tool with old_str and new_str,
    ///               THE File_Editor SHALL replace the exact match
    /// Requirements: 6.2 - IF old_str appears multiple times, THEN THE File_Editor SHALL
    ///               return an error requiring more context
    /// Requirements: 6.3 - IF old_str does not appear in the file, THEN THE File_Editor SHALL
    ///               return an error with guidance
    pub fn edit_file(
        &self,
        path: &Path,
        old_str: &str,
        new_str: &str,
    ) -> Result<EditFileResult, ToolError> {
        // 验证路径安全性
        let validated_path = self
            .security
            .validate_path(path)
            .map_err(|e| ToolError::Security(e.to_string()))?;

        // 检查文件是否存在
        if !validated_path.exists() {
            return Err(ToolError::ExecutionFailed(format!(
                "文件不存在: {}",
                path.display()
            )));
        }

        // 读取文件内容
        let original_content = fs::read_to_string(&validated_path).map_err(|e| {
            ToolError::ExecutionFailed(format!("无法读取文件 {}: {}", path.display(), e))
        })?;

        // 检查 old_str 出现次数
        // Requirements: 6.2 - IF old_str appears multiple times, THEN return error
        let occurrences = count_occurrences(&original_content, old_str);

        if occurrences == 0 {
            // Requirements: 6.3 - IF old_str does not appear, THEN return error with guidance
            return Err(ToolError::ExecutionFailed(format!(
                "在文件 {} 中未找到要替换的字符串。\n\
                 请确保 old_str 与文件内容完全匹配（包括空格和换行符）。\n\
                 提示：可以先使用 read_file 工具查看文件内容。",
                path.display()
            )));
        }

        if occurrences > 1 {
            // Requirements: 6.2 - Multiple occurrences, require more context
            let positions = find_occurrence_positions(&original_content, old_str);
            let context_snippets = positions
                .iter()
                .take(3)
                .map(|&pos| get_context_snippet(&original_content, pos, old_str.len()))
                .collect::<Vec<_>>()
                .join("\n---\n");

            return Err(ToolError::ExecutionFailed(format!(
                "在文件 {} 中找到 {} 处匹配，需要更多上下文来确定要替换哪一处。\n\
                 请在 old_str 中包含更多周围的内容以唯一标识要替换的位置。\n\n\
                 匹配位置示例:\n{}",
                path.display(),
                occurrences,
                context_snippets
            )));
        }

        // 执行替换
        // Requirements: 6.1 - Replace the exact match
        let new_content = original_content.replacen(old_str, new_str, 1);

        // 保存历史记录
        // Requirements: 6.5 - THE File_Editor SHALL maintain a history stack for undo operations
        self.save_history(&validated_path, &original_content);

        // 写入文件
        fs::write(&validated_path, &new_content).map_err(|e| {
            ToolError::ExecutionFailed(format!("无法写入文件 {}: {}", path.display(), e))
        })?;

        // 生成变更上下文片段
        // Requirements: 6.6 - WHEN an edit is applied, THE File_Editor SHALL return a snippet
        //               showing the changed context
        let change_position = original_content.find(old_str).unwrap_or(0);
        let context_snippet = generate_change_context(&new_content, change_position, new_str.len());

        // 生成 unified diff
        // Requirements: 6.4 - THE File_Editor SHALL support unified diff format
        let diff = generate_unified_diff(path, &original_content, &new_content);

        info!(
            "[EditFileTool] 编辑文件: {} (替换 {} 字节 -> {} 字节)",
            path.display(),
            old_str.len(),
            new_str.len()
        );

        Ok(EditFileResult {
            path: validated_path,
            old_str_len: old_str.len(),
            new_str_len: new_str.len(),
            context_snippet,
            diff,
        })
    }

    /// 应用 unified diff
    ///
    /// Requirements: 6.4 - THE File_Editor SHALL support unified diff format for multi-line changes
    pub fn apply_diff(&self, path: &Path, diff: &str) -> Result<EditFileResult, ToolError> {
        // 验证路径安全性
        let validated_path = self
            .security
            .validate_path(path)
            .map_err(|e| ToolError::Security(e.to_string()))?;

        // 检查文件是否存在
        if !validated_path.exists() {
            return Err(ToolError::ExecutionFailed(format!(
                "文件不存在: {}",
                path.display()
            )));
        }

        // 读取文件内容
        let original_content = fs::read_to_string(&validated_path).map_err(|e| {
            ToolError::ExecutionFailed(format!("无法读取文件 {}: {}", path.display(), e))
        })?;

        // 解析并应用 diff
        let new_content = apply_unified_diff(&original_content, diff)?;

        // 保存历史记录
        self.save_history(&validated_path, &original_content);

        // 写入文件
        fs::write(&validated_path, &new_content).map_err(|e| {
            ToolError::ExecutionFailed(format!("无法写入文件 {}: {}", path.display(), e))
        })?;

        info!("[EditFileTool] 应用 diff: {}", path.display());

        Ok(EditFileResult {
            path: validated_path,
            old_str_len: original_content.len(),
            new_str_len: new_content.len(),
            context_snippet: "Diff applied successfully".to_string(),
            diff: diff.to_string(),
        })
    }

    /// 撤销上一次编辑
    ///
    /// Requirements: 6.5 - THE File_Editor SHALL maintain a history stack for undo operations
    pub fn undo_edit(&self, path: &Path) -> Result<UndoResult, ToolError> {
        // 验证路径安全性
        let validated_path = self
            .security
            .validate_path(path)
            .map_err(|e| ToolError::Security(e.to_string()))?;

        // 获取历史记录
        let previous_content = {
            let mut history = self.history.write();
            let file_history = history.get_mut(&validated_path).ok_or_else(|| {
                ToolError::ExecutionFailed(format!("没有可撤销的编辑历史: {}", path.display()))
            })?;

            file_history.pop().ok_or_else(|| {
                ToolError::ExecutionFailed(format!("没有可撤销的编辑历史: {}", path.display()))
            })?
        };

        // 读取当前内容
        let current_content = fs::read_to_string(&validated_path).map_err(|e| {
            ToolError::ExecutionFailed(format!("无法读取文件 {}: {}", path.display(), e))
        })?;

        // 恢复之前的内容
        fs::write(&validated_path, &previous_content.content).map_err(|e| {
            ToolError::ExecutionFailed(format!("无法写入文件 {}: {}", path.display(), e))
        })?;

        info!("[EditFileTool] 撤销编辑: {}", path.display());

        Ok(UndoResult {
            path: validated_path,
            restored_content_len: previous_content.content.len(),
            previous_content_len: current_content.len(),
        })
    }

    /// 获取文件的编辑历史数量
    pub fn history_count(&self, path: &Path) -> usize {
        let validated_path = match self.security.validate_path(path) {
            Ok(p) => p,
            Err(_) => return 0,
        };

        self.history
            .read()
            .get(&validated_path)
            .map(|h| h.len())
            .unwrap_or(0)
    }

    /// 清除文件的编辑历史
    pub fn clear_history(&self, path: &Path) {
        if let Ok(validated_path) = self.security.validate_path(path) {
            let mut history = self.history.write();
            history.remove(&validated_path);
            debug!("[EditFileTool] 清除历史: {:?}", validated_path);
        }
    }

    /// 清除所有编辑历史
    pub fn clear_all_history(&self) {
        let mut history = self.history.write();
        history.clear();
        debug!("[EditFileTool] 清除所有历史");
    }

    /// 保存历史记录
    fn save_history(&self, path: &PathBuf, content: &str) {
        let mut history = self.history.write();
        let file_history = history.entry(path.clone()).or_insert_with(Vec::new);

        // 限制历史记录数量
        if file_history.len() >= MAX_HISTORY_SIZE {
            file_history.remove(0);
        }

        file_history.push(EditHistory {
            content: content.to_string(),
            timestamp: std::time::SystemTime::now(),
        });
    }
}

/// 文件编辑结果
#[derive(Debug, Clone)]
pub struct EditFileResult {
    /// 编辑的文件路径
    pub path: PathBuf,
    /// 原字符串长度
    pub old_str_len: usize,
    /// 新字符串长度
    pub new_str_len: usize,
    /// 变更上下文片段
    pub context_snippet: String,
    /// Unified diff
    pub diff: String,
}

/// 撤销结果
#[derive(Debug, Clone)]
pub struct UndoResult {
    /// 文件路径
    pub path: PathBuf,
    /// 恢复的内容长度
    pub restored_content_len: usize,
    /// 之前的内容长度
    pub previous_content_len: usize,
}

/// 编辑历史记录
#[derive(Debug, Clone)]
struct EditHistory {
    /// 编辑前的内容
    content: String,
    /// 时间戳
    timestamp: std::time::SystemTime,
}

/// 统计字符串出现次数
fn count_occurrences(content: &str, pattern: &str) -> usize {
    if pattern.is_empty() {
        return 0;
    }
    content.matches(pattern).count()
}

/// 查找所有出现位置
fn find_occurrence_positions(content: &str, pattern: &str) -> Vec<usize> {
    if pattern.is_empty() {
        return Vec::new();
    }
    content.match_indices(pattern).map(|(pos, _)| pos).collect()
}

/// 获取指定位置的上下文片段
fn get_context_snippet(content: &str, position: usize, match_len: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();

    // 找到位置所在的行
    let mut current_pos = 0;
    let mut target_line = 0;

    for (i, line) in lines.iter().enumerate() {
        let line_end = current_pos + line.len() + 1; // +1 for newline
        if position < line_end {
            target_line = i;
            break;
        }
        current_pos = line_end;
    }

    // 获取上下文行
    let start_line = target_line.saturating_sub(CONTEXT_LINES);
    let end_line = (target_line + CONTEXT_LINES + 1).min(lines.len());

    let mut result = String::new();
    for i in start_line..end_line {
        let line_num = i + 1;
        let marker = if i == target_line { ">>>" } else { "   " };
        result.push_str(&format!("{} {:4} | {}\n", marker, line_num, lines[i]));
    }

    result
}

/// 生成变更上下文
fn generate_change_context(content: &str, position: usize, new_len: usize) -> String {
    get_context_snippet(content, position, new_len)
}

/// 生成 unified diff
///
/// Requirements: 6.4 - THE File_Editor SHALL support unified diff format
fn generate_unified_diff(path: &Path, old_content: &str, new_content: &str) -> String {
    let old_lines: Vec<&str> = old_content.lines().collect();
    let new_lines: Vec<&str> = new_content.lines().collect();

    let mut diff = String::new();
    diff.push_str(&format!("--- a/{}\n", path.display()));
    diff.push_str(&format!("+++ b/{}\n", path.display()));

    // 简单的行级 diff（找出不同的行）
    let mut i = 0;
    let mut j = 0;

    while i < old_lines.len() || j < new_lines.len() {
        if i < old_lines.len() && j < new_lines.len() && old_lines[i] == new_lines[j] {
            i += 1;
            j += 1;
            continue;
        }

        // 找到差异块的起始位置
        let context_start = i.saturating_sub(CONTEXT_LINES);
        let old_start = i;
        let new_start = j;

        // 找到差异块的结束位置
        let mut old_end = i;
        let mut new_end = j;

        // 跳过不同的行
        while old_end < old_lines.len() && new_end < new_lines.len() {
            if old_lines.get(old_end) == new_lines.get(new_end) {
                // 检查是否有足够的相同行来结束差异块
                let mut same_count = 0;
                while old_end + same_count < old_lines.len()
                    && new_end + same_count < new_lines.len()
                    && old_lines[old_end + same_count] == new_lines[new_end + same_count]
                {
                    same_count += 1;
                    if same_count > CONTEXT_LINES * 2 {
                        break;
                    }
                }
                if same_count > CONTEXT_LINES * 2 {
                    break;
                }
            }
            if old_end < old_lines.len() {
                old_end += 1;
            }
            if new_end < new_lines.len() {
                new_end += 1;
            }
        }

        // 添加上下文后的结束位置
        let context_end_old = (old_end + CONTEXT_LINES).min(old_lines.len());
        let context_end_new = (new_end + CONTEXT_LINES).min(new_lines.len());

        // 输出 hunk header
        diff.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            context_start + 1,
            context_end_old - context_start,
            context_start + 1,
            context_end_new - context_start
        ));

        // 输出上下文和差异
        let mut oi = context_start;
        let mut ni = context_start;

        while oi < context_end_old || ni < context_end_new {
            if oi < old_start && ni < new_start {
                // 前置上下文
                if oi < old_lines.len() {
                    diff.push_str(&format!(" {}\n", old_lines[oi]));
                }
                oi += 1;
                ni += 1;
            } else if oi < old_end {
                // 删除的行
                if oi < old_lines.len() {
                    diff.push_str(&format!("-{}\n", old_lines[oi]));
                }
                oi += 1;
            } else if ni < new_end {
                // 添加的行
                if ni < new_lines.len() {
                    diff.push_str(&format!("+{}\n", new_lines[ni]));
                }
                ni += 1;
            } else {
                // 后置上下文
                if oi < old_lines.len() && ni < new_lines.len() {
                    diff.push_str(&format!(" {}\n", old_lines[oi]));
                }
                oi += 1;
                ni += 1;
            }
        }

        i = old_end;
        j = new_end;
    }

    diff
}

/// 应用 unified diff
///
/// Requirements: 6.4 - THE File_Editor SHALL support unified diff format
fn apply_unified_diff(content: &str, diff: &str) -> Result<String, ToolError> {
    let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    let diff_lines: Vec<&str> = diff.lines().collect();

    let mut i = 0;
    while i < diff_lines.len() {
        let line = diff_lines[i];

        // 跳过文件头
        if line.starts_with("---") || line.starts_with("+++") {
            i += 1;
            continue;
        }

        // 解析 hunk header
        if line.starts_with("@@") {
            // 解析 @@ -start,count +start,count @@
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                i += 1;
                continue;
            }

            let old_range = parts[1].trim_start_matches('-');
            let old_start: usize = old_range
                .split(',')
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1);

            let mut current_line = old_start.saturating_sub(1);
            i += 1;

            // 应用 hunk 中的变更
            while i < diff_lines.len() && !diff_lines[i].starts_with("@@") {
                let diff_line = diff_lines[i];

                if diff_line.starts_with('-') {
                    // 删除行
                    if current_line < lines.len() {
                        lines.remove(current_line);
                    }
                } else if diff_line.starts_with('+') {
                    // 添加行
                    let new_line = diff_line.strip_prefix('+').unwrap_or("");
                    lines.insert(current_line, new_line.to_string());
                    current_line += 1;
                } else if diff_line.starts_with(' ') || diff_line.is_empty() {
                    // 上下文行
                    current_line += 1;
                }

                i += 1;
            }
        } else {
            i += 1;
        }
    }

    Ok(lines.join("\n"))
}

#[async_trait]
impl Tool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "edit_file",
            "Make precise edits to an existing file by replacing exact string matches. \
             The old_str must match exactly one location in the file. \
             If old_str appears multiple times, include more surrounding context to uniquely identify the location. \
             Supports undo operations to revert changes.",
        )
        .with_parameters(
            JsonSchema::new()
                .add_property(
                    "path",
                    PropertySchema::string(
                        "The path to the file to edit. Can be relative or absolute.",
                    ),
                    true,
                )
                .add_property(
                    "old_str",
                    PropertySchema::string(
                        "The exact string to find and replace. Must match exactly one location in the file.",
                    ),
                    true,
                )
                .add_property(
                    "new_str",
                    PropertySchema::string(
                        "The string to replace old_str with.",
                    ),
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

        let old_str = args
            .get("old_str")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments("缺少 old_str 参数".to_string()))?;

        let new_str = args
            .get("new_str")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments("缺少 new_str 参数".to_string()))?;

        let path = PathBuf::from(path_str);

        info!(
            "[EditFileTool] 编辑文件: {} (old_str: {} 字节, new_str: {} 字节)",
            path_str,
            old_str.len(),
            new_str.len()
        );

        // 执行编辑
        let result = self.edit_file(&path, old_str, new_str)?;

        // 构建输出
        let output = format!(
            "成功编辑文件: {}\n\
             替换: {} 字节 -> {} 字节\n\n\
             变更上下文:\n{}\n\n\
             Diff:\n{}",
            path_str, result.old_str_len, result.new_str_len, result.context_snippet, result.diff
        );

        debug!("[EditFileTool] 编辑完成: {}", path_str);

        Ok(ToolResult::success(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup_test_tool() -> (EditFileTool, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let security = Arc::new(SecurityManager::new(temp_dir.path()));
        let tool = EditFileTool::new(security);
        (tool, temp_dir)
    }

    #[test]
    fn test_tool_definition() {
        let temp_dir = TempDir::new().unwrap();
        let security = Arc::new(SecurityManager::new(temp_dir.path()));
        let tool = EditFileTool::new(security);
        let def = tool.definition();

        assert_eq!(def.name, "edit_file");
        assert!(!def.description.is_empty());
        assert!(def.parameters.required.contains(&"path".to_string()));
        assert!(def.parameters.required.contains(&"old_str".to_string()));
        assert!(def.parameters.required.contains(&"new_str".to_string()));
    }

    #[test]
    fn test_edit_file_simple_replacement() {
        let (tool, temp_dir) = setup_test_tool();

        // 创建测试文件
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "Hello, World!").unwrap();

        // 执行编辑
        let result = tool.edit_file(Path::new("test.txt"), "World", "Rust");
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.old_str_len, 5);
        assert_eq!(result.new_str_len, 4);

        // 验证文件内容
        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "Hello, Rust!");
    }

    #[test]
    fn test_edit_file_multiline() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "Line 1\nLine 2\nLine 3").unwrap();

        // 替换多行内容
        let result = tool.edit_file(Path::new("test.txt"), "Line 2", "Modified Line");
        assert!(result.is_ok());

        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "Line 1\nModified Line\nLine 3");
    }

    #[test]
    fn test_edit_file_not_found() {
        let (tool, _temp_dir) = setup_test_tool();

        let result = tool.edit_file(Path::new("nonexistent.txt"), "old", "new");
        assert!(result.is_err());
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }

    #[test]
    fn test_edit_file_old_str_not_found() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "Hello, World!").unwrap();

        // 尝试替换不存在的字符串
        let result = tool.edit_file(Path::new("test.txt"), "NotFound", "New");
        assert!(result.is_err());

        let err = result.unwrap_err();
        if let ToolError::ExecutionFailed(msg) = err {
            assert!(msg.contains("未找到"));
        } else {
            panic!("Expected ExecutionFailed error");
        }
    }

    #[test]
    fn test_edit_file_multiple_occurrences() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "foo bar foo baz foo").unwrap();

        // 尝试替换出现多次的字符串
        let result = tool.edit_file(Path::new("test.txt"), "foo", "qux");
        assert!(result.is_err());

        let err = result.unwrap_err();
        if let ToolError::ExecutionFailed(msg) = err {
            assert!(msg.contains("3 处匹配"));
            assert!(msg.contains("更多上下文"));
        } else {
            panic!("Expected ExecutionFailed error");
        }

        // 验证文件未被修改
        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "foo bar foo baz foo");
    }

    #[test]
    fn test_edit_file_with_context() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "foo bar foo baz foo").unwrap();

        // 使用更多上下文来唯一标识
        let result = tool.edit_file(Path::new("test.txt"), "bar foo", "bar qux");
        assert!(result.is_ok());

        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "foo bar qux baz foo");
    }

    #[test]
    fn test_undo_edit() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        let original_content = "Hello, World!";
        fs::write(&file_path, original_content).unwrap();

        // 执行编辑
        let result = tool.edit_file(Path::new("test.txt"), "World", "Rust");
        assert!(result.is_ok());

        // 验证编辑后的内容
        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "Hello, Rust!");

        // 撤销编辑
        let undo_result = tool.undo_edit(Path::new("test.txt"));
        assert!(undo_result.is_ok());

        // 验证内容已恢复
        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, original_content);
    }

    #[test]
    fn test_undo_no_history() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "Hello, World!").unwrap();

        // 尝试撤销（没有历史记录）
        let result = tool.undo_edit(Path::new("test.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_edits_and_undos() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "A B C").unwrap();

        // 第一次编辑
        tool.edit_file(Path::new("test.txt"), "A", "X").unwrap();
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "X B C");

        // 第二次编辑
        tool.edit_file(Path::new("test.txt"), "B", "Y").unwrap();
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "X Y C");

        // 第三次编辑
        tool.edit_file(Path::new("test.txt"), "C", "Z").unwrap();
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "X Y Z");

        // 撤销第三次
        tool.undo_edit(Path::new("test.txt")).unwrap();
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "X Y C");

        // 撤销第二次
        tool.undo_edit(Path::new("test.txt")).unwrap();
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "X B C");

        // 撤销第一次
        tool.undo_edit(Path::new("test.txt")).unwrap();
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "A B C");
    }

    #[test]
    fn test_history_count() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "A B C D E").unwrap();

        assert_eq!(tool.history_count(Path::new("test.txt")), 0);

        tool.edit_file(Path::new("test.txt"), "A", "X").unwrap();
        assert_eq!(tool.history_count(Path::new("test.txt")), 1);

        tool.edit_file(Path::new("test.txt"), "B", "Y").unwrap();
        assert_eq!(tool.history_count(Path::new("test.txt")), 2);
    }

    #[test]
    fn test_clear_history() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "A B C").unwrap();

        tool.edit_file(Path::new("test.txt"), "A", "X").unwrap();
        tool.edit_file(Path::new("test.txt"), "B", "Y").unwrap();
        assert_eq!(tool.history_count(Path::new("test.txt")), 2);

        tool.clear_history(Path::new("test.txt"));
        assert_eq!(tool.history_count(Path::new("test.txt")), 0);
    }

    #[test]
    fn test_security_path_traversal() {
        let (tool, _temp_dir) = setup_test_tool();

        let result = tool.edit_file(Path::new("../../../etc/passwd"), "old", "new");
        assert!(result.is_err());
        assert!(matches!(result, Err(ToolError::Security(_))));
    }

    #[test]
    fn test_count_occurrences() {
        assert_eq!(count_occurrences("foo bar foo baz foo", "foo"), 3);
        assert_eq!(count_occurrences("hello world", "foo"), 0);
        assert_eq!(count_occurrences("aaa", "a"), 3);
        assert_eq!(count_occurrences("aaa", "aa"), 1); // non-overlapping
        assert_eq!(count_occurrences("", "foo"), 0);
        assert_eq!(count_occurrences("foo", ""), 0);
    }

    #[test]
    fn test_find_occurrence_positions() {
        let positions = find_occurrence_positions("foo bar foo baz foo", "foo");
        assert_eq!(positions, vec![0, 8, 16]);

        let positions = find_occurrence_positions("hello world", "foo");
        assert!(positions.is_empty());
    }

    #[test]
    fn test_generate_unified_diff() {
        let old_content = "Line 1\nLine 2\nLine 3";
        let new_content = "Line 1\nModified\nLine 3";

        let diff = generate_unified_diff(Path::new("test.txt"), old_content, new_content);

        assert!(diff.contains("--- a/test.txt"));
        assert!(diff.contains("+++ b/test.txt"));
        assert!(diff.contains("-Line 2"));
        assert!(diff.contains("+Modified"));
    }

    #[tokio::test]
    async fn test_tool_execute() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "Hello, World!").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "path": "test.txt",
                "old_str": "World",
                "new_str": "Rust"
            }))
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert!(result.output.contains("成功编辑"));

        // 验证文件已修改
        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "Hello, Rust!");
    }

    #[tokio::test]
    async fn test_tool_execute_missing_args() {
        let (tool, _temp_dir) = setup_test_tool();

        // 缺少 path
        let result = tool
            .execute(serde_json::json!({
                "old_str": "old",
                "new_str": "new"
            }))
            .await;
        assert!(result.is_err());

        // 缺少 old_str
        let result = tool
            .execute(serde_json::json!({
                "path": "test.txt",
                "new_str": "new"
            }))
            .await;
        assert!(result.is_err());

        // 缺少 new_str
        let result = tool
            .execute(serde_json::json!({
                "path": "test.txt",
                "old_str": "old"
            }))
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_edit_preserves_whitespace() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "  indented\n\ttabbed\n").unwrap();

        let result = tool.edit_file(Path::new("test.txt"), "  indented", "    more indented");
        assert!(result.is_ok());

        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "    more indented\n\ttabbed\n");
    }

    #[test]
    fn test_edit_empty_new_str() {
        let (tool, temp_dir) = setup_test_tool();

        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "Hello, World!").unwrap();

        // 删除字符串（替换为空）
        let result = tool.edit_file(Path::new("test.txt"), ", World", "");
        assert!(result.is_ok());

        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "Hello!");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use std::fs;
    use tempfile::TempDir;

    /// 生成有效的文件内容（多行文本，每行唯一）
    fn arb_file_content_with_unique_lines() -> impl Strategy<Value = Vec<String>> {
        prop::collection::vec("[a-zA-Z0-9 ,.!?]{5,50}", 3..20).prop_map(|lines| {
            lines
                .iter()
                .enumerate()
                .map(|(i, content)| format!("LINE{}_{}", i + 1, content))
                .collect()
        })
    }

    /// 生成有效的替换字符串
    fn arb_replacement_str() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9 ,.!?]{1,100}"
    }

    /// 生成包含重复内容的文件
    fn arb_file_with_duplicates() -> impl Strategy<Value = (Vec<String>, String)> {
        (
            prop::collection::vec("[a-zA-Z0-9]{5,20}", 2..10),
            "[a-zA-Z0-9]{3,10}",
        )
            .prop_map(|(unique_parts, duplicate)| {
                let mut lines = Vec::new();
                for (i, part) in unique_parts.iter().enumerate() {
                    if i % 2 == 0 {
                        lines.push(format!("{} {} {}", part, duplicate, part));
                    } else {
                        lines.push(part.clone());
                    }
                }
                (lines, duplicate)
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: agent-tool-calling, Property 9: 文件编辑精确替换**
        /// **Validates: Requirements 6.1**
        ///
        /// *For any* 文件内容和唯一出现的 old_str，edit_file 执行后
        /// 文件中应该不再包含 old_str，而包含 new_str。
        #[test]
        fn prop_edit_file_exact_replacement(
            lines in arb_file_content_with_unique_lines(),
            new_str in arb_replacement_str()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = EditFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            let content = lines.join("\n");
            fs::write(&file_path, &content).unwrap();

            // 选择一个唯一的行作为 old_str
            if lines.is_empty() {
                return Ok(());
            }
            let target_line_idx = 0; // 使用第一行
            let old_str = &lines[target_line_idx];

            // 执行编辑
            let result = tool.edit_file(Path::new("test.txt"), old_str, &new_str);

            prop_assert!(
                result.is_ok(),
                "编辑应该成功: {:?}",
                result.err()
            );

            // 读取编辑后的文件
            let edited_content = fs::read_to_string(&file_path).unwrap();

            // 验证 old_str 不再存在
            prop_assert!(
                !edited_content.contains(old_str),
                "编辑后文件不应该包含 old_str: '{}'",
                old_str
            );

            // 验证 new_str 存在
            prop_assert!(
                edited_content.contains(&new_str),
                "编辑后文件应该包含 new_str: '{}'",
                new_str
            );
        }

        /// **Feature: agent-tool-calling, Property 9: 文件编辑精确替换 - 其他内容不变**
        /// **Validates: Requirements 6.1**
        ///
        /// *For any* 文件内容和唯一出现的 old_str，edit_file 执行后
        /// 除了被替换的部分，其他内容应该保持不变。
        #[test]
        fn prop_edit_file_preserves_other_content(
            lines in arb_file_content_with_unique_lines(),
            new_str in arb_replacement_str()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = EditFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            let content = lines.join("\n");
            fs::write(&file_path, &content).unwrap();

            if lines.len() < 2 {
                return Ok(());
            }

            // 选择第一行作为 old_str
            let old_str = &lines[0];

            // 执行编辑
            let result = tool.edit_file(Path::new("test.txt"), old_str, &new_str);
            prop_assert!(result.is_ok());

            // 读取编辑后的文件
            let edited_content = fs::read_to_string(&file_path).unwrap();

            // 验证其他行仍然存在
            for (i, line) in lines.iter().enumerate() {
                if i == 0 {
                    continue; // 跳过被替换的行
                }
                prop_assert!(
                    edited_content.contains(line),
                    "编辑后文件应该保留第 {} 行: '{}'",
                    i + 1,
                    line
                );
            }
        }

        /// **Feature: agent-tool-calling, Property 9: 文件编辑精确替换 - 返回正确的长度**
        /// **Validates: Requirements 6.1**
        ///
        /// *For any* 编辑操作，返回的 old_str_len 和 new_str_len 应该正确。
        #[test]
        fn prop_edit_file_returns_correct_lengths(
            lines in arb_file_content_with_unique_lines(),
            new_str in arb_replacement_str()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = EditFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            let content = lines.join("\n");
            fs::write(&file_path, &content).unwrap();

            if lines.is_empty() {
                return Ok(());
            }

            let old_str = &lines[0];

            // 执行编辑
            let result = tool.edit_file(Path::new("test.txt"), old_str, &new_str);
            prop_assert!(result.is_ok());

            let result = result.unwrap();

            prop_assert_eq!(
                result.old_str_len,
                old_str.len(),
                "old_str_len 应该等于 old_str 的长度"
            );

            prop_assert_eq!(
                result.new_str_len,
                new_str.len(),
                "new_str_len 应该等于 new_str 的长度"
            );
        }

        /// **Feature: agent-tool-calling, Property 9: 文件编辑精确替换 - 生成 diff**
        /// **Validates: Requirements 6.1, 6.4**
        ///
        /// *For any* 编辑操作，应该生成包含变更信息的 diff。
        #[test]
        fn prop_edit_file_generates_diff(
            lines in arb_file_content_with_unique_lines(),
            new_str in arb_replacement_str()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = EditFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            let content = lines.join("\n");
            fs::write(&file_path, &content).unwrap();

            if lines.is_empty() {
                return Ok(());
            }

            let old_str = &lines[0];

            // 执行编辑
            let result = tool.edit_file(Path::new("test.txt"), old_str, &new_str);
            prop_assert!(result.is_ok());

            let result = result.unwrap();

            // 验证 diff 包含文件头
            prop_assert!(
                result.diff.contains("--- a/test.txt"),
                "diff 应该包含旧文件头"
            );
            prop_assert!(
                result.diff.contains("+++ b/test.txt"),
                "diff 应该包含新文件头"
            );
        }
    }
}

#[cfg(test)]
mod proptests_multiple_occurrences {
    use super::*;
    use proptest::prelude::*;
    use std::fs;
    use tempfile::TempDir;

    /// 生成包含重复字符串的文件内容
    fn arb_content_with_duplicates() -> impl Strategy<Value = (String, String, usize)> {
        (
            "[a-zA-Z0-9]{3,15}",  // 重复的字符串
            "[a-zA-Z0-9 ]{5,30}", // 唯一的前缀/后缀
            2..6usize,            // 重复次数
        )
            .prop_map(|(duplicate, unique, count)| {
                let mut content = String::new();
                for i in 0..count {
                    content.push_str(&format!(
                        "{}_{} {} {}_{}\n",
                        unique, i, duplicate, unique, i
                    ));
                }
                (content, duplicate, count)
            })
    }

    /// 生成替换字符串
    fn arb_replacement() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9]{5,20}"
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: agent-tool-calling, Property 10: 文件编辑多次出现错误**
        /// **Validates: Requirements 6.2**
        ///
        /// *For any* 文件内容中出现多次的字符串 old_str，
        /// edit_file 应该返回错误而不修改文件。
        #[test]
        fn prop_edit_file_multiple_occurrences_error(
            (content, duplicate, count) in arb_content_with_duplicates(),
            new_str in arb_replacement()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = EditFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            fs::write(&file_path, &content).unwrap();

            // 尝试替换出现多次的字符串
            let result = tool.edit_file(Path::new("test.txt"), &duplicate, &new_str);

            // 应该返回错误
            prop_assert!(
                result.is_err(),
                "出现 {} 次的字符串 '{}' 应该返回错误，但结果是 {:?}",
                count,
                duplicate,
                result
            );

            // 验证错误消息包含出现次数
            if let Err(ToolError::ExecutionFailed(msg)) = result {
                prop_assert!(
                    msg.contains(&format!("{} 处匹配", count)),
                    "错误消息应该包含出现次数 '{}': {}",
                    count,
                    msg
                );
                prop_assert!(
                    msg.contains("更多上下文"),
                    "错误消息应该提示需要更多上下文: {}",
                    msg
                );
            } else {
                prop_assert!(false, "应该返回 ExecutionFailed 错误");
            }
        }

        /// **Feature: agent-tool-calling, Property 10: 文件编辑多次出现错误 - 文件未修改**
        /// **Validates: Requirements 6.2**
        ///
        /// *For any* 文件内容中出现多次的字符串 old_str，
        /// edit_file 返回错误后文件内容应该保持不变。
        #[test]
        fn prop_edit_file_multiple_occurrences_no_modification(
            (content, duplicate, _count) in arb_content_with_duplicates(),
            new_str in arb_replacement()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = EditFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            fs::write(&file_path, &content).unwrap();

            // 尝试替换出现多次的字符串
            let _ = tool.edit_file(Path::new("test.txt"), &duplicate, &new_str);

            // 验证文件内容未被修改
            let after_content = fs::read_to_string(&file_path).unwrap();
            prop_assert_eq!(
                after_content,
                content,
                "文件内容应该保持不变"
            );
        }

        /// **Feature: agent-tool-calling, Property 10: 文件编辑多次出现错误 - 无历史记录**
        /// **Validates: Requirements 6.2**
        ///
        /// *For any* 失败的编辑操作，不应该添加历史记录。
        #[test]
        fn prop_edit_file_multiple_occurrences_no_history(
            (content, duplicate, _count) in arb_content_with_duplicates(),
            new_str in arb_replacement()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = EditFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            fs::write(&file_path, &content).unwrap();

            // 验证初始历史记录为空
            prop_assert_eq!(
                tool.history_count(Path::new("test.txt")),
                0,
                "初始历史记录应该为空"
            );

            // 尝试替换出现多次的字符串（应该失败）
            let _ = tool.edit_file(Path::new("test.txt"), &duplicate, &new_str);

            // 验证历史记录仍然为空
            prop_assert_eq!(
                tool.history_count(Path::new("test.txt")),
                0,
                "失败的编辑不应该添加历史记录"
            );
        }

        /// **Feature: agent-tool-calling, Property 10: 文件编辑多次出现错误 - 提供上下文示例**
        /// **Validates: Requirements 6.2**
        ///
        /// *For any* 出现多次的字符串，错误消息应该包含匹配位置的上下文示例。
        #[test]
        fn prop_edit_file_multiple_occurrences_shows_context(
            (content, duplicate, _count) in arb_content_with_duplicates(),
            new_str in arb_replacement()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = EditFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            fs::write(&file_path, &content).unwrap();

            // 尝试替换出现多次的字符串
            let result = tool.edit_file(Path::new("test.txt"), &duplicate, &new_str);

            // 验证错误消息包含上下文示例
            if let Err(ToolError::ExecutionFailed(msg)) = result {
                prop_assert!(
                    msg.contains("匹配位置示例"),
                    "错误消息应该包含匹配位置示例: {}",
                    msg
                );
            }
        }
    }
}

#[cfg(test)]
mod proptests_undo {
    use super::*;
    use proptest::prelude::*;
    use std::fs;
    use tempfile::TempDir;

    /// 生成有效的文件内容（多行文本，每行唯一）
    fn arb_file_content_unique() -> impl Strategy<Value = Vec<String>> {
        prop::collection::vec("[a-zA-Z0-9 ,.!?]{5,50}", 3..15).prop_map(|lines| {
            lines
                .iter()
                .enumerate()
                .map(|(i, content)| format!("UNIQUE{}_{}", i + 1, content))
                .collect()
        })
    }

    /// 生成有效的替换字符串
    fn arb_replacement() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9 ,.!?]{1,50}"
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: agent-tool-calling, Property 11: 文件编辑撤销 Round-Trip**
        /// **Validates: Requirements 6.5**
        ///
        /// *For any* 成功的文件编辑操作，执行 undo_edit 后
        /// 文件内容应该恢复到编辑前的状态。
        #[test]
        fn prop_edit_undo_roundtrip(
            lines in arb_file_content_unique(),
            new_str in arb_replacement()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = EditFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            let original_content = lines.join("\n");
            fs::write(&file_path, &original_content).unwrap();

            if lines.is_empty() {
                return Ok(());
            }

            // 选择第一行作为 old_str
            let old_str = &lines[0];

            // 执行编辑
            let edit_result = tool.edit_file(Path::new("test.txt"), old_str, &new_str);
            prop_assert!(edit_result.is_ok(), "编辑应该成功");

            // 验证文件已被修改
            let edited_content = fs::read_to_string(&file_path).unwrap();
            prop_assert_ne!(
                edited_content,
                original_content.clone(),
                "编辑后文件内容应该改变"
            );

            // 执行撤销
            let undo_result = tool.undo_edit(Path::new("test.txt"));
            prop_assert!(undo_result.is_ok(), "撤销应该成功");

            // 验证文件内容已恢复
            let restored_content = fs::read_to_string(&file_path).unwrap();
            prop_assert_eq!(
                restored_content,
                original_content,
                "撤销后文件内容应该恢复到原始状态"
            );
        }

        /// **Feature: agent-tool-calling, Property 11: 文件编辑撤销 Round-Trip - 多次编辑**
        /// **Validates: Requirements 6.5**
        ///
        /// *For any* 多次成功的文件编辑操作，每次 undo_edit 应该
        /// 恢复到上一次编辑前的状态。
        #[test]
        fn prop_edit_multiple_undo_roundtrip(
            lines in arb_file_content_unique(),
            new_str1 in arb_replacement(),
            new_str2 in arb_replacement()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = EditFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            let original_content = lines.join("\n");
            fs::write(&file_path, &original_content).unwrap();

            if lines.len() < 2 {
                return Ok(());
            }

            // 第一次编辑
            let old_str1 = &lines[0];
            let edit1_result = tool.edit_file(Path::new("test.txt"), old_str1, &new_str1);
            prop_assert!(edit1_result.is_ok(), "第一次编辑应该成功");

            let after_edit1 = fs::read_to_string(&file_path).unwrap();

            // 第二次编辑
            let old_str2 = &lines[1];
            let edit2_result = tool.edit_file(Path::new("test.txt"), old_str2, &new_str2);
            prop_assert!(edit2_result.is_ok(), "第二次编辑应该成功");

            // 撤销第二次编辑
            let undo2_result = tool.undo_edit(Path::new("test.txt"));
            prop_assert!(undo2_result.is_ok(), "撤销第二次编辑应该成功");

            let after_undo2 = fs::read_to_string(&file_path).unwrap();
            prop_assert_eq!(
                after_undo2,
                after_edit1,
                "撤销第二次编辑后应该恢复到第一次编辑后的状态"
            );

            // 撤销第一次编辑
            let undo1_result = tool.undo_edit(Path::new("test.txt"));
            prop_assert!(undo1_result.is_ok(), "撤销第一次编辑应该成功");

            let after_undo1 = fs::read_to_string(&file_path).unwrap();
            prop_assert_eq!(
                after_undo1,
                original_content.clone(),
                "撤销第一次编辑后应该恢复到原始状态"
            );
        }

        /// **Feature: agent-tool-calling, Property 11: 文件编辑撤销 Round-Trip - 历史记录正确**
        /// **Validates: Requirements 6.5**
        ///
        /// *For any* 编辑操作，历史记录数量应该正确增减。
        #[test]
        fn prop_edit_undo_history_count(
            lines in arb_file_content_unique(),
            new_str in arb_replacement()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = EditFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            let content = lines.join("\n");
            fs::write(&file_path, &content).unwrap();

            if lines.is_empty() {
                return Ok(());
            }

            // 初始历史记录为 0
            prop_assert_eq!(
                tool.history_count(Path::new("test.txt")),
                0,
                "初始历史记录应该为 0"
            );

            // 编辑后历史记录为 1
            let old_str = &lines[0];
            tool.edit_file(Path::new("test.txt"), old_str, &new_str).unwrap();
            prop_assert_eq!(
                tool.history_count(Path::new("test.txt")),
                1,
                "编辑后历史记录应该为 1"
            );

            // 撤销后历史记录为 0
            tool.undo_edit(Path::new("test.txt")).unwrap();
            prop_assert_eq!(
                tool.history_count(Path::new("test.txt")),
                0,
                "撤销后历史记录应该为 0"
            );
        }

        /// **Feature: agent-tool-calling, Property 11: 文件编辑撤销 Round-Trip - 撤销结果正确**
        /// **Validates: Requirements 6.5**
        ///
        /// *For any* 撤销操作，返回的结果应该包含正确的长度信息。
        #[test]
        fn prop_edit_undo_result_correct(
            lines in arb_file_content_unique(),
            new_str in arb_replacement()
        ) {
            let temp_dir = TempDir::new().unwrap();
            let security = Arc::new(SecurityManager::new(temp_dir.path()));
            let tool = EditFileTool::new(security);

            // 创建测试文件
            let file_path = temp_dir.path().join("test.txt");
            let original_content = lines.join("\n");
            fs::write(&file_path, &original_content).unwrap();

            if lines.is_empty() {
                return Ok(());
            }

            // 执行编辑
            let old_str = &lines[0];
            tool.edit_file(Path::new("test.txt"), old_str, &new_str).unwrap();

            let edited_content = fs::read_to_string(&file_path).unwrap();

            // 执行撤销
            let undo_result = tool.undo_edit(Path::new("test.txt")).unwrap();

            // 验证撤销结果
            prop_assert_eq!(
                undo_result.restored_content_len,
                original_content.len(),
                "restored_content_len 应该等于原始内容长度"
            );
            prop_assert_eq!(
                undo_result.previous_content_len,
                edited_content.len(),
                "previous_content_len 应该等于编辑后内容长度"
            );
        }
    }
}
