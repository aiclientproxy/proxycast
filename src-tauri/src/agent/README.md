# Agent 模块

<!-- 一旦我所属的文件夹有所变化，请更新我 -->

## 架构说明

AI Agent 集成模块，提供原生 Rust Agent 功能，支持**连续对话**和**工具调用循环**。

### 设计决策

- **原生 Rust 实现**：直接在 Rust 中处理 Agent 功能，复用现有 provider 和流式处理能力
- **会话管理**：支持多会话，每个会话独立维护消息历史和系统提示词
- **连续对话**：每次请求携带 session_id，自动包含历史消息
- **流式响应**：通过 Tauri 事件系统向前端推送流式内容
- **工具系统**：可扩展的工具定义和执行框架，支持 Bash、文件操作等
- **工具调用循环**：自动执行工具调用并继续对话，直到产生最终响应

## 文件索引

| 文件/目录 | 说明 |
|------|------|
| `mod.rs` | 模块入口，导出公共类型 |
| `types.rs` | Agent 相关类型定义（会话、消息、工具、配置） |
| `native_agent.rs` | 原生 Rust Agent 实现（NativeAgent、NativeAgentState） |
| `tool_loop.rs` | 工具调用循环引擎（ToolLoopEngine、ToolLoopConfig） |
| `tools/` | 工具系统子模块（类型定义、注册表、具体工具实现） |

## 核心类型

### 会话管理
- `AgentSession`: 会话状态，包含消息历史和系统提示词
- `AgentMessage`: 消息结构，支持文本、图片、工具调用

### 消息内容
- `MessageContent`: 消息内容（文本或多部分）
- `ContentPart`: 内容部分（文本/图片）

### 工具系统
- `ToolDefinition`: 工具定义（名称、描述、参数 Schema）
- `JsonSchema`: JSON Schema 参数定义
- `PropertySchema`: 属性 Schema（类型、描述、默认值）
- `ToolCall`: 工具调用请求
- `ToolResult`: 工具执行结果
- `ToolError`: 工具错误类型
- `Tool` trait: 工具接口（definition + execute）
- `ToolRegistry`: 工具注册表（注册、查找、验证、执行）

### 工具调用循环
- `ToolLoopEngine`: 工具循环引擎，执行工具调用并继续对话
- `ToolLoopConfig`: 循环配置（最大迭代次数等）
- `ToolLoopState`: 循环状态跟踪
- `ToolCallResult`: 工具调用结果

### Agent 实现
- `NativeAgent`: Agent 核心实现
- `NativeAgentState`: Tauri 状态管理器

## 使用示例

```rust
// 创建会话
let session_id = agent_state.create_session(
    Some("claude-sonnet-4-20250514".to_string()),
    Some("你是一个有帮助的助手".to_string()),
)?;

// 发送消息（自动包含历史）
let request = NativeChatRequest {
    session_id: Some(session_id.clone()),
    message: "你好".to_string(),
    model: None,
    images: None,
    stream: false,
};
let response = agent_state.chat(request).await?;

// 使用工具调用循环
let registry = Arc::new(ToolRegistry::new());
registry.register(BashTool::new(security.clone()))?;
let engine = ToolLoopEngine::new(registry);

let (tx, rx) = mpsc::channel(100);
let result = agent_state.chat_stream_with_tools(request, tx, &engine).await?;
```

## 更新提醒

任何文件变更后，请更新此文档和相关的上级文档。
