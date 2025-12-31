//! Agent 工具系统模块
//!
//! 提供工具定义、注册、执行的核心框架
//! 参考 Claude Code 和 goose 项目的工具系统设计
//!
//! ## 模块结构
//! - `types`: 工具类型定义（ToolDefinition, ToolCall, ToolResult 等）
//! - `registry`: 工具注册表和 Tool trait
//! - `security`: 安全管理器（路径验证、符号链接检查等）
//! - `bash`: Bash 命令执行工具
//! - `read_file`: 文件读取工具
//! - `write_file`: 文件写入工具
//! - `edit_file`: 文件编辑工具
//! - `prompt`: 工具 Prompt 生成器（System Prompt 工具注入）

pub mod bash;
pub mod edit_file;
pub mod prompt;
pub mod read_file;
pub mod registry;
pub mod security;
pub mod types;
pub mod write_file;

pub use bash::{BashExecutionResult, BashTool, ShellType};
pub use edit_file::{EditFileResult, EditFileTool, UndoResult};
pub use prompt::{generate_tools_prompt, PromptFormat, ToolPromptGenerator};
pub use read_file::{ReadFileResult, ReadFileTool};
pub use registry::{Tool, ToolRegistry};
pub use security::{SecurityError, SecurityManager};
pub use types::*;
pub use write_file::{WriteFileResult, WriteFileTool};
