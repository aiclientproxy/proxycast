//! Agent 类型定义
//!
//! 定义 Agent 模块使用的核心类型
//! 参考 goose 项目的 Conversation 设计，支持连续对话和工具调用

use serde::{Deserialize, Serialize};

/// Agent 会话状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSession {
    /// 会话 ID
    pub id: String,
    /// 使用的模型
    pub model: String,
    /// 会话消息历史（支持连续对话）
    pub messages: Vec<AgentMessage>,
    /// 系统提示词
    pub system_prompt: Option<String>,
    /// 创建时间
    pub created_at: String,
    /// 最后活动时间
    pub updated_at: String,
}

/// Agent 消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    /// 角色: user, assistant, system, tool
    pub role: String,
    /// 消息内容（文本或结构化内容）
    pub content: MessageContent,
    /// 时间戳
    pub timestamp: String,
    /// 工具调用（assistant 消息可能包含）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// 工具调用 ID（tool 角色消息需要）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// 消息内容类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// 纯文本
    Text(String),
    /// 多部分内容（文本 + 图片）
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// 获取文本内容
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

/// 内容部分
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// 文本
    Text { text: String },
    /// 图片 URL
    ImageUrl { image_url: ImageUrl },
}

/// 图片 URL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// 工具调用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// 工具调用 ID
    pub id: String,
    /// 工具类型
    #[serde(rename = "type")]
    pub call_type: String,
    /// 函数调用详情
    pub function: FunctionCall,
}

/// 函数调用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// 函数名
    pub name: String,
    /// 参数（JSON 字符串）
    pub arguments: String,
}

/// 工具定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// 工具类型
    #[serde(rename = "type")]
    pub tool_type: String,
    /// 函数定义
    pub function: FunctionDefinition,
}

/// 函数定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// 函数名
    pub name: String,
    /// 函数描述
    pub description: String,
    /// 参数 schema
    pub parameters: serde_json::Value,
}

/// Agent 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// 模型名称
    pub model: String,
    /// 系统提示词
    pub system_prompt: Option<String>,
    /// 温度参数
    pub temperature: Option<f32>,
    /// 最大 token 数
    pub max_tokens: Option<u32>,
    /// 可用工具
    pub tools: Vec<ToolDefinition>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-4-20250514".to_string(),
            system_prompt: None,
            temperature: Some(0.7),
            max_tokens: Some(4096),
            tools: Vec::new(),
        }
    }
}

/// 聊天请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeChatRequest {
    /// 会话 ID（用于连续对话）
    pub session_id: Option<String>,
    /// 用户消息
    pub message: String,
    /// 模型名称（可选）
    pub model: Option<String>,
    /// 图片列表（可选）
    pub images: Option<Vec<ImageData>>,
    /// 是否流式响应
    pub stream: bool,
}

/// 图片数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    /// base64 编码的图片数据
    pub data: String,
    /// MIME 类型
    pub media_type: String,
}

/// 聊天响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeChatResponse {
    /// 响应内容
    pub content: String,
    /// 使用的模型
    pub model: String,
    /// Token 使用量
    pub usage: Option<TokenUsage>,
    /// 是否成功
    pub success: bool,
    /// 错误信息
    pub error: Option<String>,
}

/// Token 使用量
///
/// 记录 API 调用的 token 消耗
/// Requirements: 1.3 - THE Streaming_Handler SHALL emit a done event with token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    /// 输入 token 数
    pub input_tokens: u32,
    /// 输出 token 数
    pub output_tokens: u32,
}

impl TokenUsage {
    /// 创建新的 TokenUsage
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
        }
    }

    /// 计算总 token 数
    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

/// 流式响应事件
///
/// 定义流式输出过程中的各种事件类型
/// Requirements: 1.1, 1.3, 1.4
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum StreamEvent {
    /// 文本增量
    /// Requirements: 1.1 - THE Streaming_Handler SHALL emit text deltas to the frontend in real-time
    #[serde(rename = "text_delta")]
    TextDelta { text: String },

    /// 工具调用开始
    /// Requirements: 7.6 - WHILE the Tool_Loop is executing, THE Frontend SHALL display the current tool being executed
    #[serde(rename = "tool_start")]
    ToolStart {
        /// 工具名称
        tool_name: String,
        /// 工具调用 ID
        tool_id: String,
    },

    /// 工具调用结束
    /// Requirements: 7.6 - 工具执行完成后通知前端
    #[serde(rename = "tool_end")]
    ToolEnd {
        /// 工具调用 ID
        tool_id: String,
        /// 工具执行结果
        result: ToolExecutionResult,
    },

    /// 完成
    /// Requirements: 1.3 - THE Streaming_Handler SHALL emit a done event with token usage statistics
    #[serde(rename = "done")]
    Done { usage: Option<TokenUsage> },

    /// 错误
    /// Requirements: 1.4 - IF a streaming error occurs, THEN THE Streaming_Handler SHALL emit an error event
    #[serde(rename = "error")]
    Error { message: String },
}

/// 工具执行结果（用于 StreamEvent）
///
/// 简化版的工具结果，用于前端显示
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolExecutionResult {
    /// 是否成功
    pub success: bool,
    /// 输出内容
    pub output: String,
    /// 错误信息（如果失败）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ToolExecutionResult {
    /// 创建成功结果
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
        }
    }

    /// 创建失败结果
    pub fn failure(error: impl Into<String>) -> Self {
        let error_msg = error.into();
        Self {
            success: false,
            output: String::new(),
            error: Some(error_msg),
        }
    }

    /// 创建带输出的失败结果
    pub fn failure_with_output(output: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: output.into(),
            error: Some(error.into()),
        }
    }
}

/// 流式响应结果
///
/// 流式处理完成后的最终结果
/// Requirements: 1.1, 1.3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamResult {
    /// 完整的响应内容
    pub content: String,
    /// 工具调用列表（如果有）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Token 使用量
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

impl StreamResult {
    /// 创建新的流式结果
    pub fn new(content: String) -> Self {
        Self {
            content,
            tool_calls: None,
            usage: None,
        }
    }

    /// 设置工具调用
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// 设置 token 使用量
    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// 是否有工具调用
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls
            .as_ref()
            .map(|tc| !tc.is_empty())
            .unwrap_or(false)
    }
}
