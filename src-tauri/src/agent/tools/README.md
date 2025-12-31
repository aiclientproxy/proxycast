# 工具系统模块

<!-- 一旦我所属的文件夹有所变化，请更新我 -->

## 架构说明

Agent 工具系统模块，提供工具定义、注册、执行的核心框架。

### 设计决策

- **可扩展架构**：通过 `Tool` trait 定义工具接口，便于添加新工具
- **类型安全**：使用 JSON Schema 定义参数，支持必需和可选参数验证
- **动态注册**：工具可在运行时注册/注销，无需重启
- **安全优先**：所有工具执行前进行参数验证，SecurityManager 提供路径安全检查

## 文件索引

| 文件 | 说明 |
|------|------|
| `mod.rs` | 模块入口，导出公共类型 |
| `types.rs` | 工具类型定义（ToolDefinition, ToolCall, ToolResult, ToolError） |
| `registry.rs` | Tool trait 和 ToolRegistry 实现 |
| `security.rs` | 安全管理器（路径验证、符号链接检查、目录遍历防护） |
| `bash.rs` | Bash 命令执行工具（shell 检测、命令执行、超时控制、环境变量设置） |
| `read_file.rs` | 文件读取工具（带行号读取、行范围读取、大文件检测、目录列表、语言检测） |
| `write_file.rs` | 文件写入工具（文件创建/覆盖、父目录自动创建、换行符规范化、尾部换行符保证） |
| `edit_file.rs` | 文件编辑工具（精确字符串替换、多次出现检测、unified diff、历史栈、撤销功能） |
| `prompt.rs` | 工具 Prompt 生成器（System Prompt 工具注入、XML/JSON 格式转换） |

## 核心类型

### 工具定义
- `ToolDefinition`: 工具定义结构（名称、描述、参数 Schema）
- `JsonSchema`: JSON Schema 参数定义
- `PropertySchema`: 属性 Schema（类型、描述、默认值、枚举值）

### 工具调用
- `ToolCall`: 工具调用请求（ID、名称、参数）
- `ToolResult`: 工具执行结果（成功/失败、输出、错误信息）

### 错误类型
- `ToolError`: 工具执行错误（NotFound, InvalidArguments, ExecutionFailed, Security, Timeout）
- `ToolValidationError`: 工具定义验证错误（EmptyName, EmptyDescription, RequiredPropertyNotDefined, DuplicateName）
- `SecurityError`: 安全错误（PathTraversal, OutsideBaseDir, SymlinkNotAllowed, InvalidPath）

### 工具接口
- `Tool` trait: 工具接口，包含 `definition()` 和 `execute()` 方法
- `ToolRegistry`: 工具注册表，管理所有已注册的工具

### 安全管理
- `SecurityManager`: 安全管理器，验证文件操作的安全性
  - `validate_path()`: 完整路径验证（".." 检查、基础目录检查、符号链接检查）
  - `quick_check()`: 快速检查（仅检查 ".." 组件）
  - `validate_path_no_symlink_check()`: 不检查符号链接的路径验证

### Bash 工具
- `BashTool`: Bash 命令执行工具
  - `execute_command()`: 执行 shell 命令，捕获 stdout/stderr
  - `detect_shell()`: 检测用户默认 shell（bash/zsh/powershell）
  - `get_non_interactive_env()`: 获取防止交互的环境变量
- `ShellType`: Shell 类型枚举（Bash, Zsh, PowerShell, Cmd, Sh）
- `BashExecutionResult`: 命令执行结果（stdout, stderr, exit_code, timed_out）

### 文件读取工具
- `ReadFileTool`: 文件读取工具
  - `read_file()`: 读取文件内容，支持行范围
  - 自动检测编程语言
  - 大文件推荐使用行范围
  - 目录自动列出内容
- `ReadFileResult`: 文件读取结果（content, total_lines, start_line, end_line, language, is_directory, recommend_range, truncated）

### 文件写入工具
- `WriteFileTool`: 文件写入工具
  - `write_file()`: 创建或覆盖文件
  - 自动创建父目录
  - 换行符规范化（Unix: LF, Windows: CRLF）
  - 确保文件以换行符结尾
- `WriteFileResult`: 文件写入结果（path, bytes_written, line_count, created, overwritten）

### 文件编辑工具
- `EditFileTool`: 文件编辑工具
  - `edit_file()`: 精确字符串替换（old_str → new_str）
  - `apply_diff()`: 应用 unified diff 格式的变更
  - `undo_edit()`: 撤销上一次编辑
  - `history_count()`: 获取编辑历史数量
  - `clear_history()`: 清除编辑历史
  - 多次出现检测（返回错误要求更多上下文）
  - 不存在检测（返回错误和指导）
  - 返回变更上下文片段
- `EditFileResult`: 文件编辑结果（path, old_str_len, new_str_len, context_snippet, diff）
- `UndoResult`: 撤销结果（path, restored_content_len, previous_content_len）

### Prompt 生成器
- `ToolPromptGenerator`: 工具 Prompt 生成器
  - `generate_system_prompt()`: 生成包含工具定义的 System Prompt
  - `tool_to_xml()`: 将工具定义转换为 XML 格式
  - `tool_to_json()`: 将工具定义转换为 JSON 格式
- `PromptFormat`: Prompt 输出格式枚举（Xml, Json）
- `generate_tools_prompt()`: 便捷函数，生成工具 Prompt

## 使用示例

### 定义工具

```rust
use crate::agent::tools::{Tool, ToolDefinition, ToolResult, ToolError, JsonSchema, PropertySchema};
use async_trait::async_trait;

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("echo", "Echo the input message")
            .with_parameters(
                JsonSchema::new()
                    .add_property("message", PropertySchema::string("The message to echo"), true)
            )
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult, ToolError> {
        let message = args.get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments("缺少 message 参数".to_string()))?;
        
        Ok(ToolResult::success(message))
    }
}
```

### 注册和执行工具

```rust
use crate::agent::tools::ToolRegistry;

let registry = ToolRegistry::new();

// 注册工具
registry.register(EchoTool)?;

// 执行工具
let result = registry.execute("echo", serde_json::json!({"message": "Hello!"})).await?;
assert!(result.success);
assert_eq!(result.output, "Hello!");
```

### 使用文件读取工具

```rust
use crate::agent::tools::{ReadFileTool, SecurityManager};
use std::sync::Arc;

let security = Arc::new(SecurityManager::new("/path/to/project"));
let tool = ReadFileTool::new(security);

// 读取整个文件
let result = tool.read_file(Path::new("src/main.rs"), None, None)?;
println!("语言: {:?}", result.language);
println!("总行数: {}", result.total_lines);

// 读取指定行范围
let result = tool.read_file(Path::new("src/main.rs"), Some(10), Some(20))?;
println!("内容:\n{}", result.content);
```

### 使用文件写入工具

```rust
use crate::agent::tools::{WriteFileTool, SecurityManager};
use std::sync::Arc;

let security = Arc::new(SecurityManager::new("/path/to/project"));
let tool = WriteFileTool::new(security);

// 写入新文件
let result = tool.write_file(Path::new("output.txt"), "Hello, World!")?;
println!("创建: {}, 字节数: {}", result.created, result.bytes_written);

// 覆盖已有文件
let result = tool.write_file(Path::new("output.txt"), "New content")?;
println!("覆盖: {}", result.overwritten);

// 自动创建父目录
let result = tool.write_file(Path::new("a/b/c/nested.txt"), "Nested content")?;
println!("路径: {:?}", result.path);
```

### 使用文件编辑工具

```rust
use crate::agent::tools::{EditFileTool, SecurityManager};
use std::sync::Arc;

let security = Arc::new(SecurityManager::new("/path/to/project"));
let tool = EditFileTool::new(security);

// 精确字符串替换
let result = tool.edit_file(Path::new("src/main.rs"), "old_code", "new_code")?;
println!("替换: {} 字节 -> {} 字节", result.old_str_len, result.new_str_len);
println!("变更上下文:\n{}", result.context_snippet);
println!("Diff:\n{}", result.diff);

// 撤销编辑
let undo_result = tool.undo_edit(Path::new("src/main.rs"))?;
println!("已恢复: {} 字节", undo_result.restored_content_len);

// 查看历史记录数量
let count = tool.history_count(Path::new("src/main.rs"));
println!("历史记录: {} 条", count);
```

### 使用 Prompt 生成器

```rust
use crate::agent::tools::{ToolPromptGenerator, PromptFormat, ToolDefinition, JsonSchema, PropertySchema};

// 创建工具定义
let tools = vec![
    ToolDefinition::new("bash", "Execute a bash command")
        .with_parameters(
            JsonSchema::new()
                .add_property("command", PropertySchema::string("The command to execute"), true)
        ),
    ToolDefinition::new("read_file", "Read file contents")
        .with_parameters(
            JsonSchema::new()
                .add_property("path", PropertySchema::string("The file path"), true)
        ),
];

// 生成 XML 格式的 System Prompt
let generator = ToolPromptGenerator::new().with_format(PromptFormat::Xml);
let system_prompt = generator.generate_system_prompt(&tools);
println!("System Prompt:\n{}", system_prompt);

// 生成 JSON 格式的 System Prompt
let json_generator = ToolPromptGenerator::new().with_format(PromptFormat::Json);
let json_prompt = json_generator.generate_system_prompt(&tools);
println!("JSON Prompt:\n{}", json_prompt);

// 使用便捷函数
use crate::agent::tools::generate_tools_prompt;
let prompt = generate_tools_prompt(&tools, PromptFormat::Xml);
```

## 需求追溯

- Requirements 2.1: 工具定义包含 name, description, JSON Schema parameters
- Requirements 2.2: 注册时验证工具定义
- Requirements 2.4: 运行时添加工具无需重启
- Requirements 2.5: 支持必需和可选参数类型验证
- Requirements 3.1: Bash 工具在用户默认 shell 中执行命令
- Requirements 3.2: Bash 工具捕获 stdout 和 stderr
- Requirements 3.3: Bash 工具支持超时控制
- Requirements 3.4: Bash 工具设置防止交互的环境变量
- Requirements 3.5: Bash 工具返回退出码和错误输出
- Requirements 3.6: Bash 工具支持可配置的工作目录
- Requirements 4.1: 文件读取工具返回带行号的内容
- Requirements 4.2: 文件读取工具支持行范围读取
- Requirements 4.3: 文件不存在时返回清晰错误信息
- Requirements 4.4: 大文件推荐使用行范围
- Requirements 4.5: 检测并报告文件的编程语言
- Requirements 4.6: 路径为目录时列出目录内容
- Requirements 5.1: 文件写入工具创建或覆盖文件
- Requirements 5.2: 文件写入工具自动创建父目录
- Requirements 5.3: 文件写入工具规范化换行符（Unix: LF, Windows: CRLF）
- Requirements 5.4: 文件写入工具确保文件以换行符结尾
- Requirements 5.5: 写入失败时返回描述性错误信息
- Requirements 6.1: 文件编辑工具精确替换匹配的字符串
- Requirements 6.2: 多次出现时返回错误要求更多上下文
- Requirements 6.3: 字符串不存在时返回错误和指导
- Requirements 6.4: 支持 unified diff 格式
- Requirements 6.5: 维护历史栈支持撤销操作
- Requirements 6.6: 编辑后返回变更上下文片段
- Requirements 8.1: 验证所有文件路径防止目录遍历攻击
- Requirements 8.2: 拒绝包含 ".." 组件的路径
- Requirements 8.3: 拒绝符号链接操作
- Requirements 8.4: Bash 工具设置环境变量禁用交互式编辑器和提示
- Requirements 8.5: 强制执行可配置的基础目录
- Requirements 2.3: System Prompt 包含所有可用工具定义

## 更新提醒

任何文件变更后，请更新此文档和相关的上级文档。
