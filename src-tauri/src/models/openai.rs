//! OpenAI API 数据模型
//!
//! 支持标准 OpenAI 格式以及扩展的工具类型（如 web_search）。
//!
//! # 工具类型支持
//!
//! - `function`: 标准函数调用工具
//! - `web_search`: 联网搜索工具（Claude Code 使用 `web_search_20250305`）
//!
//! # 更新日志
//!
//! - 2025-12-27: 添加 web_search 工具支持，修复 Issue #49
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn get_content_text(&self) -> String {
        match &self.content {
            Some(MessageContent::Text(s)) => s.clone(),
            Some(MessageContent::Parts(parts)) => parts
                .iter()
                .filter_map(|p| {
                    if let ContentPart::Text { text } = p {
                        Some(text.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(""),
            None => String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// 工具定义
///
/// 支持多种工具类型：
/// - `function`: 标准函数调用工具，包含 function 字段
/// - `web_search`: 联网搜索工具，无需额外字段
/// - `web_search_20250305`: Claude Code 的联网搜索工具类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Tool {
    /// 标准函数调用工具
    #[serde(rename = "function")]
    Function { function: FunctionDef },
    /// 联网搜索工具（Codex/Kiro 格式）
    #[serde(rename = "web_search")]
    WebSearch,
    /// 联网搜索工具（Claude Code 格式）
    #[serde(rename = "web_search_20250305")]
    WebSearch20250305,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// 思维链强度：none, low, medium, high
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: StreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

// ============================================================================
// 图像生成 API 数据模型
// ============================================================================

/// OpenAI 图像生成请求
///
/// 兼容 OpenAI Images API，支持通过 Antigravity 生成图像。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    /// 图像生成提示词
    pub prompt: String,

    /// 模型名称 (默认: gemini-3-pro-image-preview)
    #[serde(default = "default_image_model")]
    pub model: String,

    /// 生成图像数量 (默认: 1)
    #[serde(default = "default_n")]
    pub n: u32,

    /// 图像尺寸 (可选，Antigravity 可能忽略)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,

    /// 响应格式: "url" 或 "b64_json" (默认: "url")
    #[serde(default = "default_response_format")]
    pub response_format: String,

    /// 图像质量 (可选，Antigravity 可能忽略)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,

    /// 图像风格 (可选，Antigravity 可能忽略)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,

    /// 用户标识 (可选)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

fn default_image_model() -> String {
    "gemini-3-pro-image-preview".to_string()
}

fn default_n() -> u32 {
    1
}

fn default_response_format() -> String {
    "url".to_string()
}

/// OpenAI 图像生成响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    /// 创建时间戳 (Unix epoch seconds)
    pub created: i64,

    /// 生成的图像数组
    pub data: Vec<ImageData>,
}

/// 单个图像数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    /// Base64 编码的图像数据 (当 response_format="b64_json")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,

    /// 图像 URL (当 response_format="url"，返回 data URL)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// 修订后的提示词 (如果 Antigravity 返回了文本)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}
