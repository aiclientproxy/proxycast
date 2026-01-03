//! OpenAI 格式转换为 Antigravity (Gemini CLI) 格式
//!
//! 本模块实现 OpenAI Chat Completions API 到 Antigravity/Gemini CLI API 的转换。
//! 参考 CLIProxyAPI 的实现，确保请求格式与 Gemini CLI 兼容。
//!
//! ## 主要功能
//! - 消息格式转换（system/user/assistant/tool）
//! - 工具定义转换（parameters → parametersJsonSchema）
//! - 安全设置自动附加
//! - 思维链配置（reasoning_effort）

#![allow(dead_code)]
//!
//! ## 更新日志
//! - 2025-12-28: 修复请求格式，对齐 CLIProxyAPI 实现

use crate::models::openai::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// 常量定义
// ============================================================================

/// Gemini CLI 函数调用的 thought signature 标记
const GEMINI_CLI_FUNCTION_THOUGHT_SIGNATURE: &str = "skip_thought_signature_validator";

// ============================================================================
// 数据结构定义
// ============================================================================

/// Antigravity/Gemini 内容部分
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<InlineData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<GeminiFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<GeminiFunctionResponse>,
    /// 思维签名，用于函数调用
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    pub response: GeminiFunctionResponseBody,
}

/// Function Response 的响应体结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionResponseBody {
    pub result: serde_json::Value,
}

/// Antigravity/Gemini 内容
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

/// Antigravity/Gemini 工具定义
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_declarations: Option<Vec<GeminiFunctionDeclaration>>,
    /// Google Search 工具（透传）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_search: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiFunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Gemini CLI 使用 parametersJsonSchema 而非 parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters_json_schema: Option<serde_json::Value>,
}

/// 安全设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySetting {
    pub category: String,
    pub threshold: String,
}

/// Antigravity/Gemini 生成配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
    /// 响应模态（TEXT, IMAGE）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_modalities: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<i32>,
}

/// Antigravity 请求体内部结构
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AntigravityRequestInner {
    pub contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GeminiGenerationConfig>,
    /// 工具定义 - 支持 Gemini 格式（function_declarations）和 Claude 格式（custom + input_schema）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// 安全设置
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_settings: Option<Vec<SafetySetting>>,
}

// ============================================================================
// 辅助函数
// ============================================================================

// ============================================================================
// 辅助函数
// ============================================================================

/// 生成随机请求 ID
fn generate_request_id() -> String {
    format!("agent-{}", Uuid::new_v4())
}

/// 生成随机会话 ID
fn generate_session_id() -> String {
    let uuid = Uuid::new_v4();
    let bytes = uuid.as_bytes();
    let n: u64 = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]) % 9_000_000_000_000_000_000;
    format!("-{}", n)
}

/// 获取默认安全设置
fn default_safety_settings() -> Vec<SafetySetting> {
    vec![
        SafetySetting {
            category: "HARM_CATEGORY_HARASSMENT".to_string(),
            threshold: "OFF".to_string(),
        },
        SafetySetting {
            category: "HARM_CATEGORY_HATE_SPEECH".to_string(),
            threshold: "OFF".to_string(),
        },
        SafetySetting {
            category: "HARM_CATEGORY_SEXUALLY_EXPLICIT".to_string(),
            threshold: "OFF".to_string(),
        },
        SafetySetting {
            category: "HARM_CATEGORY_DANGEROUS_CONTENT".to_string(),
            threshold: "OFF".to_string(),
        },
        SafetySetting {
            category: "HARM_CATEGORY_CIVIC_INTEGRITY".to_string(),
            threshold: "BLOCK_NONE".to_string(),
        },
    ]
}

/// 模型名称映射
fn model_mapping(model: &str) -> &str {
    match model {
        "claude-sonnet-4-5-thinking" => "claude-sonnet-4-5",
        "claude-opus-4-5" => "claude-opus-4-5-thinking",
        "gemini-2.5-flash-thinking" => "gemini-2.5-flash",
        "gemini-2.5-computer-use-preview-10-2025" => "rev19-uic3-1p",
        "gemini-3-pro-image-preview" => "gemini-3-pro-image",
        "gemini-3-pro-preview" => "gemini-3-pro-high",
        "gemini-claude-sonnet-4-5" => "claude-sonnet-4-5",
        "gemini-claude-sonnet-4-5-thinking" => "claude-sonnet-4-5-thinking",
        _ => model,
    }
}

/// 检查模型是否支持思维链
fn model_supports_thinking(model: &str) -> bool {
    model.contains("2.5") || model.contains("3-pro") || model.contains("thinking")
}

/// 检查模型是否使用离散思维级别（Gemini 3）
fn model_uses_thinking_levels(model: &str) -> bool {
    model.contains("gemini-3")
}

/// 是否启用思维链
fn is_enable_thinking(model: &str) -> bool {
    model.ends_with("-thinking")
        || model == "gemini-2.5-pro"
        || model.starts_with("gemini-3-pro-")
        || model == "rev19-uic3-1p"
        || model == "gpt-oss-120b-medium"
}

/// 检查是否是图片生成模型
fn is_image_generation_model(model: &str) -> bool {
    model == "gemini-3-pro-image" || model == "gemini-3-pro-image-preview"
}

/// 检查是否是 Claude 模型（通过 Antigravity 访问）
/// Claude 模型需要使用不同的工具格式（custom + input_schema）
fn is_claude_model(model: &str) -> bool {
    model.starts_with("claude-") || model.contains("claude")
}

// ============================================================================
// 主转换函数
// ============================================================================

// ============================================================================
// 主转换函数
// ============================================================================

/// 将 OpenAI ChatCompletionRequest 转换为 Antigravity 请求体
///
/// 参考 CLIProxyAPI 的实现，确保请求格式正确。
pub fn convert_openai_to_antigravity_with_context(
    request: &ChatCompletionRequest,
    project_id: &str,
) -> serde_json::Value {
    let actual_model = model_mapping(&request.model);
    let supports_thinking = model_supports_thinking(actual_model);

    let mut contents: Vec<GeminiContent> = Vec::new();
    let mut system_instruction: Option<GeminiContent> = None;

    // 第一遍：收集 assistant tool_calls 的 id -> name 映射
    let mut tc_id_to_name: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for msg in &request.messages {
        if msg.role == "assistant" {
            if let Some(tool_calls) = &msg.tool_calls {
                for tc in tool_calls {
                    tc_id_to_name.insert(tc.id.clone(), tc.function.name.clone());
                }
            }
        }
    }

    // 第二遍：收集 tool 响应
    let mut tool_responses: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for msg in &request.messages {
        if msg.role == "tool" {
            if let Some(tool_call_id) = &msg.tool_call_id {
                let content = msg.get_content_text();
                tool_responses.insert(tool_call_id.clone(), content);
            }
        }
    }

    // 第三遍：构建消息
    let messages_len = request.messages.len();
    for (idx, msg) in request.messages.iter().enumerate() {
        match msg.role.as_str() {
            "system" => {
                // system 消息只有在有其他消息时才作为 systemInstruction
                if messages_len > 1 {
                    let text = msg.get_content_text();
                    if !text.is_empty() {
                        system_instruction = Some(GeminiContent {
                            role: "user".to_string(),
                            parts: vec![GeminiPart {
                                text: Some(text),
                                inline_data: None,
                                function_call: None,
                                function_response: None,
                                thought_signature: None,
                            }],
                        });
                    }
                } else {
                    // 只有 system 消息时，作为 user 消息
                    let parts = convert_user_content(msg);
                    if !parts.is_empty() {
                        contents.push(GeminiContent {
                            role: "user".to_string(),
                            parts,
                        });
                    }
                }
            }
            "user" => {
                let parts = convert_user_content(msg);
                if !parts.is_empty() {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts,
                    });
                }
            }
            "assistant" => {
                let mut parts = Vec::new();

                // 文本内容
                let text = msg.get_content_text();
                if !text.is_empty() {
                    parts.push(GeminiPart {
                        text: Some(text),
                        inline_data: None,
                        function_call: None,
                        function_response: None,
                        thought_signature: None,
                    });
                }

                // 处理多模态内容（如图片）
                if let Some(MessageContent::Parts(content_parts)) = &msg.content {
                    for part in content_parts {
                        if let ContentPart::ImageUrl { image_url } = part {
                            if let Some((mime, data)) = parse_data_url(&image_url.url) {
                                parts.push(GeminiPart {
                                    text: None,
                                    inline_data: Some(InlineData {
                                        mime_type: mime,
                                        data,
                                    }),
                                    function_call: None,
                                    function_response: None,
                                    thought_signature: None,
                                });
                            }
                        }
                    }
                }

                // 工具调用
                if let Some(tool_calls) = &msg.tool_calls {
                    let mut function_ids: Vec<String> = Vec::new();

                    for tc in tool_calls {
                        let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::json!({}));

                        parts.push(GeminiPart {
                            text: None,
                            inline_data: None,
                            function_call: Some(GeminiFunctionCall {
                                id: Some(tc.id.clone()),
                                name: tc.function.name.clone(),
                                args, // 直接使用 args，不要包装
                            }),
                            function_response: None,
                            thought_signature: Some(
                                GEMINI_CLI_FUNCTION_THOUGHT_SIGNATURE.to_string(),
                            ),
                        });

                        function_ids.push(tc.id.clone());
                    }

                    // 添加 model 消息
                    if !parts.is_empty() {
                        contents.push(GeminiContent {
                            role: "model".to_string(),
                            parts,
                        });
                    }

                    // 紧接着添加 tool 响应作为 user 消息
                    let mut tool_parts: Vec<GeminiPart> = Vec::new();
                    for fid in &function_ids {
                        if let Some(name) = tc_id_to_name.get(fid) {
                            let resp = tool_responses.get(fid).cloned().unwrap_or_default();

                            // 解析响应内容
                            let result_value: serde_json::Value =
                                if resp.is_empty() || resp == "null" {
                                    serde_json::json!({})
                                } else {
                                    serde_json::from_str(&resp).unwrap_or_else(|_| {
                                        // 非 JSON 内容，作为字符串
                                        serde_json::Value::String(resp.clone())
                                    })
                                };

                            tool_parts.push(GeminiPart {
                                text: None,
                                inline_data: None,
                                function_call: None,
                                function_response: Some(GeminiFunctionResponse {
                                    id: Some(fid.clone()),
                                    name: name.clone(),
                                    response: GeminiFunctionResponseBody {
                                        result: result_value,
                                    },
                                }),
                                thought_signature: None,
                            });
                        }
                    }

                    if !tool_parts.is_empty() {
                        contents.push(GeminiContent {
                            role: "user".to_string(),
                            parts: tool_parts,
                        });
                    }
                } else if !parts.is_empty() {
                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
            }
            "tool" => {
                // tool 消息已经在 assistant 处理时合并了，这里跳过
                // 但如果前面没有对应的 assistant tool_calls，需要单独处理
                let tool_id = msg.tool_call_id.clone().unwrap_or_default();

                // 检查是否已经被处理过
                let already_processed = idx > 0
                    && request.messages[..idx].iter().rev().any(|m| {
                        m.role == "assistant"
                            && m.tool_calls
                                .as_ref()
                                .map(|tcs| tcs.iter().any(|tc| tc.id == tool_id))
                                .unwrap_or(false)
                    });

                if !already_processed {
                    let content = msg.get_content_text();
                    let function_name = tc_id_to_name.get(&tool_id).cloned().unwrap_or_default();

                    let result_value: serde_json::Value = if content.is_empty() || content == "null"
                    {
                        serde_json::json!({})
                    } else {
                        serde_json::from_str(&content)
                            .unwrap_or_else(|_| serde_json::Value::String(content.clone()))
                    };

                    let function_response = GeminiPart {
                        text: None,
                        inline_data: None,
                        function_call: None,
                        function_response: Some(GeminiFunctionResponse {
                            id: Some(tool_id),
                            name: function_name,
                            response: GeminiFunctionResponseBody {
                                result: result_value,
                            },
                        }),
                        thought_signature: None,
                    };

                    // 检查是否需要合并到上一条 user 消息
                    let should_merge = contents
                        .last()
                        .map(|last| {
                            last.role == "user"
                                && last.parts.iter().any(|p| p.function_response.is_some())
                        })
                        .unwrap_or(false);

                    if should_merge {
                        if let Some(last) = contents.last_mut() {
                            last.parts.push(function_response);
                        }
                    } else {
                        contents.push(GeminiContent {
                            role: "user".to_string(),
                            parts: vec![function_response],
                        });
                    }
                }
            }
            _ => {}
        }
    }

    // 构建生成配置
    let mut generation_config = GeminiGenerationConfig {
        temperature: request.temperature,
        max_output_tokens: request.max_tokens.map(|t| t as i32),
        top_p: request.top_p,
        top_k: None,
        stop_sequences: None,
        candidate_count: None,
        thinking_config: None,
        response_modalities: None,
    };

    // 为图片生成模型设置 response_modalities
    if is_image_generation_model(actual_model) {
        generation_config.response_modalities = Some(vec!["TEXT".to_string(), "IMAGE".to_string()]);
        tracing::info!(
            "[ANTIGRAVITY] 图片生成模型 {} 已启用 IMAGE 响应模态",
            actual_model
        );
        eprintln!(
            "[ANTIGRAVITY] 图片生成模型 {} 已启用 IMAGE 响应模态",
            actual_model
        );
    } else {
        eprintln!(
            "[ANTIGRAVITY] 模型 {} 不是图片生成模型，不启用 IMAGE 响应模态",
            actual_model
        );
    }

    // 处理 reasoning_effort（思维链配置）
    if supports_thinking {
        if let Some(ref effort) = request.reasoning_effort {
            let effort_lower = effort.to_lowercase();
            if effort_lower != "none" {
                if model_uses_thinking_levels(actual_model) {
                    // Gemini 3 使用离散级别
                    generation_config.thinking_config = Some(ThinkingConfig {
                        include_thoughts: Some(true),
                        thinking_budget: None,
                    });
                } else {
                    // Gemini 2.5 使用数值预算
                    let budget = match effort_lower.as_str() {
                        "low" => 1024,
                        "medium" => 8192,
                        "high" => 24576,
                        _ => 8192,
                    };
                    generation_config.thinking_config = Some(ThinkingConfig {
                        include_thoughts: Some(true),
                        thinking_budget: Some(budget),
                    });
                }
            }
        } else if is_enable_thinking(&request.model) {
            // 默认启用思维链
            generation_config.thinking_config = Some(ThinkingConfig {
                include_thoughts: Some(true),
                thinking_budget: Some(8192),
            });
        }
    }

    // 转换工具定义
    // 注意：Antigravity API 统一使用 Gemini 格式（function_declarations）
    // Claude 模型在 Antigravity 内部会自动转换
    let tools: Option<serde_json::Value> = request.tools.as_ref().and_then(|tools| {
        let is_claude = is_claude_model(actual_model);

        // Gemini 模型使用 function_declarations + parametersJsonSchema
        // Claude 模型使用 function_declarations + inputSchema（注意字段名不同）
        let mut function_declarations: Vec<serde_json::Value> = Vec::new();

        for t in tools {
            match t {
                Tool::Function { function } => {
                    let params_schema = function
                        .parameters
                        .as_ref()
                        .map(|p| {
                            let mut schema = clean_parameters(Some(p.clone())).unwrap_or_default();
                            // 确保有 type 和 properties
                            if schema.get("type").is_none() {
                                schema["type"] = serde_json::json!("object");
                            }
                            if schema.get("properties").is_none() {
                                schema["properties"] = serde_json::json!({});
                            }
                            schema
                        })
                        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));

                    if is_claude {
                        // Claude 模型使用 inputSchema 字段名
                        function_declarations.push(serde_json::json!({
                            "name": function.name,
                            "description": function.description.clone().unwrap_or_default(),
                            "inputSchema": params_schema
                        }));
                    } else {
                        // Gemini 模型使用 parametersJsonSchema 字段名
                        function_declarations.push(serde_json::json!({
                            "name": function.name,
                            "description": function.description.clone(),
                            "parametersJsonSchema": params_schema
                        }));
                    }
                }
                Tool::WebSearch | Tool::WebSearch20250305 => {
                    // web_search 工具不转换
                }
            }
        }

        if function_declarations.is_empty() {
            None
        } else {
            Some(serde_json::json!([{
                "functionDeclarations": function_declarations
            }]))
        }
    });

    let inner = AntigravityRequestInner {
        contents,
        system_instruction,
        generation_config: Some(generation_config),
        tools,
        tool_config: None,
        session_id: Some(generate_session_id()),
        safety_settings: Some(default_safety_settings()),
    };

    // 构建完整的 Antigravity 请求体
    serde_json::json!({
        "project": project_id,
        "requestId": generate_request_id(),
        "request": inner,
        "model": actual_model,
        "userAgent": "antigravity"
    })
}

// ============================================================================
// 辅助转换函数
// ============================================================================

/// 清理参数中不需要的字段
fn clean_parameters(params: Option<serde_json::Value>) -> Option<serde_json::Value> {
    params.map(clean_value)
}

fn clean_value(value: serde_json::Value) -> serde_json::Value {
    const EXCLUDED_KEYS: &[&str] = &[
        "$schema",
        "additionalProperties",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "strict", // Gemini 不支持 strict
    ];

    match value {
        serde_json::Value::Object(map) => {
            let cleaned: serde_json::Map<String, serde_json::Value> = map
                .into_iter()
                .filter(|(k, _)| !EXCLUDED_KEYS.contains(&k.as_str()))
                .map(|(k, v)| (k, clean_value(v)))
                .collect();
            serde_json::Value::Object(cleaned)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(clean_value).collect())
        }
        other => other,
    }
}

/// 兼容旧接口
pub fn convert_openai_to_antigravity(request: &ChatCompletionRequest) -> serde_json::Value {
    convert_openai_to_antigravity_with_context(request, "")
}

/// 转换用户消息内容
fn convert_user_content(msg: &ChatMessage) -> Vec<GeminiPart> {
    let mut parts = Vec::new();

    match &msg.content {
        Some(MessageContent::Text(text)) => {
            parts.push(GeminiPart {
                text: Some(text.clone()),
                inline_data: None,
                function_call: None,
                function_response: None,
                thought_signature: None,
            });
        }
        Some(MessageContent::Parts(content_parts)) => {
            for part in content_parts {
                match part {
                    ContentPart::Text { text } => {
                        parts.push(GeminiPart {
                            text: Some(text.clone()),
                            inline_data: None,
                            function_call: None,
                            function_response: None,
                            thought_signature: None,
                        });
                    }
                    ContentPart::ImageUrl { image_url } => {
                        // 处理 base64 图片
                        if let Some((mime, data)) = parse_data_url(&image_url.url) {
                            parts.push(GeminiPart {
                                text: None,
                                inline_data: Some(InlineData {
                                    mime_type: mime,
                                    data,
                                }),
                                function_call: None,
                                function_response: None,
                                thought_signature: None,
                            });
                        }
                    }
                }
            }
        }
        None => {}
    }

    parts
}

/// 解析 data URL
fn parse_data_url(url: &str) -> Option<(String, String)> {
    if url.starts_with("data:") {
        let parts: Vec<&str> = url.splitn(2, ',').collect();
        if parts.len() == 2 {
            let meta = parts[0].strip_prefix("data:")?;
            let mime = meta.split(';').next()?.to_string();
            let data = parts[1].to_string();
            return Some((mime, data));
        }
    }
    None
}

// ============================================================================
// 响应转换函数
// ============================================================================

/// 将 Antigravity 响应转换为 OpenAI 格式
///
/// Antigravity 响应结构：
/// ```json
/// {
///   "response": {
///     "candidates": [...],
///     "usageMetadata": {...},
///     "modelVersion": "...",
///     "responseId": "..."
///   }
/// }
/// ```
pub fn convert_antigravity_to_openai_response(
    antigravity_resp: &serde_json::Value,
    model: &str,
) -> serde_json::Value {
    // Antigravity 响应可能在 response 字段下，也可能直接是 Gemini 格式
    let resp = antigravity_resp.get("response").unwrap_or(antigravity_resp);

    let mut choices = Vec::new();
    let mut reasoning_content: Option<String> = None;

    if let Some(candidates) = resp.get("candidates").and_then(|c| c.as_array()) {
        for (i, candidate) in candidates.iter().enumerate() {
            let mut content = String::new();
            let mut tool_calls: Vec<serde_json::Value> = Vec::new();

            if let Some(parts) = candidate
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
            {
                for part in parts {
                    // 检查是否是思维内容
                    let is_thought = part
                        .get("thought")
                        .and_then(|t| t.as_bool())
                        .unwrap_or(false);

                    // 跳过纯 thoughtSignature 部分
                    let has_thought_signature = part
                        .get("thoughtSignature")
                        .or_else(|| part.get("thought_signature"))
                        .and_then(|s| s.as_str())
                        .map(|s| !s.is_empty())
                        .unwrap_or(false);

                    let has_content = part.get("text").is_some()
                        || part.get("functionCall").is_some()
                        || part.get("inlineData").is_some();

                    if has_thought_signature && !has_content {
                        continue;
                    }

                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        if is_thought {
                            // 思维内容
                            if let Some(ref mut rc) = reasoning_content {
                                rc.push_str(text);
                            } else {
                                reasoning_content = Some(text.to_string());
                            }
                        } else {
                            content.push_str(text);
                        }
                    }

                    if let Some(fc) = part.get("functionCall") {
                        // 优先使用响应中的 id，否则生成新的
                        let call_id = fc
                            .get("id")
                            .and_then(|id| id.as_str())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| {
                                format!("call_{}", &uuid::Uuid::new_v4().to_string()[..8])
                            });

                        let default_args = serde_json::json!({});
                        let args = fc.get("args").unwrap_or(&default_args);
                        let args_str = if args.is_string() {
                            args.as_str().unwrap_or("{}").to_string()
                        } else {
                            serde_json::to_string(args).unwrap_or_default()
                        };

                        tool_calls.push(serde_json::json!({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": fc.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                                "arguments": args_str
                            }
                        }));
                    }

                    // 处理图片输出
                    if let Some(inline_data) =
                        part.get("inlineData").or_else(|| part.get("inline_data"))
                    {
                        if let Some(data) = inline_data.get("data").and_then(|d| d.as_str()) {
                            let mime_type = inline_data
                                .get("mimeType")
                                .or_else(|| inline_data.get("mime_type"))
                                .and_then(|m| m.as_str())
                                .unwrap_or("image/png");

                            // 将图片作为 data URL 添加到内容中
                            let image_url = format!("data:{};base64,{}", mime_type, data);
                            if !content.is_empty() {
                                content.push_str("\n\n");
                            }
                            content.push_str(&format!("![image]({})", image_url));
                        }
                    }
                }
            }

            let finish_reason = candidate
                .get("finishReason")
                .and_then(|r| r.as_str())
                .map(|r| match r.to_uppercase().as_str() {
                    "STOP" => "stop",
                    "MAX_TOKENS" => "length",
                    "SAFETY" => "content_filter",
                    "RECITATION" => "content_filter",
                    _ => "stop",
                })
                .unwrap_or(if !tool_calls.is_empty() {
                    "tool_calls"
                } else {
                    "stop"
                });

            let mut message = serde_json::json!({
                "role": "assistant",
                "content": if content.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(content) }
            });

            if let Some(ref rc) = reasoning_content {
                message["reasoning_content"] = serde_json::Value::String(rc.clone());
            }

            if !tool_calls.is_empty() {
                message["tool_calls"] = serde_json::json!(tool_calls);
            }

            choices.push(serde_json::json!({
                "index": i,
                "message": message,
                "finish_reason": finish_reason
            }));
        }
    }

    // 构建 usage
    let usage = resp.get("usageMetadata").map(|u| {
        let prompt_tokens = u
            .get("promptTokenCount")
            .and_then(|t| t.as_i64())
            .unwrap_or(0);
        let completion_tokens = u
            .get("candidatesTokenCount")
            .and_then(|t| t.as_i64())
            .unwrap_or(0);
        let total_tokens = u
            .get("totalTokenCount")
            .and_then(|t| t.as_i64())
            .unwrap_or(0);
        let thoughts_tokens = u
            .get("thoughtsTokenCount")
            .and_then(|t| t.as_i64())
            .unwrap_or(0);
        let cached_tokens = u
            .get("cachedContentTokenCount")
            .and_then(|t| t.as_i64())
            .unwrap_or(0);

        let mut usage_obj = serde_json::json!({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        });

        if thoughts_tokens > 0 {
            usage_obj["completion_tokens_details"] = serde_json::json!({
                "reasoning_tokens": thoughts_tokens
            });
        }

        if cached_tokens > 0 {
            usage_obj["prompt_tokens_details"] = serde_json::json!({
                "cached_tokens": cached_tokens
            });
        }

        usage_obj
    });

    // 获取响应 ID
    let response_id = resp
        .get("responseId")
        .and_then(|id| id.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("chatcmpl-{}", uuid::Uuid::new_v4()));

    let mut response = serde_json::json!({
        "id": response_id,
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": model,
        "choices": choices
    });

    if let Some(u) = usage {
        response["usage"] = u;
    }

    response
}

// ============================================================================
// 图像生成 API 转换函数
// ============================================================================

use crate::models::openai::{ImageData, ImageGenerationRequest, ImageGenerationResponse};

/// 图像生成模型名称映射
///
/// 注意：Antigravity API 使用内部模型名称 `gemini-3-pro-image`，
/// 而不是用户友好名称 `gemini-3-pro-image-preview`。
fn image_model_mapping(model: &str) -> &str {
    match model {
        // OpenAI 兼容模型名 -> Antigravity 内部名称
        "dall-e-3" | "dall-e-2" => "gemini-3-pro-image",
        // 用户友好名称 -> 内部名称
        "gemini-3-pro-image-preview" => "gemini-3-pro-image",
        _ => model,
    }
}

/// 将 OpenAI 图像生成请求转换为 Antigravity 格式
///
/// # 参数
/// - `request`: OpenAI 图像生成请求
/// - `project_id`: Antigravity 项目 ID
///
/// # 返回
/// Antigravity 格式的请求 JSON
pub fn convert_image_request_to_antigravity(
    request: &ImageGenerationRequest,
    project_id: &str,
) -> serde_json::Value {
    // 模型映射
    let actual_model = image_model_mapping(&request.model);

    // 构建 Gemini 内容结构
    let contents = vec![serde_json::json!({
        "role": "user",
        "parts": [{"text": request.prompt}]
    })];

    // 构建生成配置
    let generation_config = serde_json::json!({
        "temperature": 1.0,
        "maxOutputTokens": 8096,
        "responseModalities": ["TEXT", "IMAGE"],
        "candidateCount": request.n
    });

    // 构建安全设置
    let safety_settings = default_safety_settings();

    // 构建完整请求
    serde_json::json!({
        "project": project_id,
        "requestId": format!("img-{}", Uuid::new_v4()),
        "request": {
            "contents": contents,
            "generationConfig": generation_config,
            "safetySettings": safety_settings
        },
        "model": actual_model,
        "userAgent": "antigravity"
    })
}

/// 将 Antigravity 图像响应转换为 OpenAI 格式
///
/// # 参数
/// - `antigravity_resp`: Antigravity 响应 JSON
/// - `response_format`: 响应格式 ("url" 或 "b64_json")
///
/// # 返回
/// OpenAI 格式的图像生成响应，或错误信息
pub fn convert_antigravity_image_response(
    antigravity_resp: &serde_json::Value,
    response_format: &str,
) -> Result<ImageGenerationResponse, String> {
    let resp = antigravity_resp.get("response").unwrap_or(antigravity_resp);

    let mut images = Vec::new();
    let mut revised_prompt: Option<String> = None;

    if let Some(candidates) = resp.get("candidates").and_then(|c| c.as_array()) {
        for candidate in candidates {
            if let Some(parts) = candidate
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
            {
                for part in parts {
                    // 提取文本作为 revised_prompt
                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        if !text.is_empty() {
                            revised_prompt = Some(text.to_string());
                        }
                    }

                    // 提取图像数据
                    if let Some(inline_data) =
                        part.get("inlineData").or_else(|| part.get("inline_data"))
                    {
                        if let (Some(data), Some(mime_type)) = (
                            inline_data.get("data").and_then(|d| d.as_str()),
                            inline_data
                                .get("mimeType")
                                .or_else(|| inline_data.get("mime_type"))
                                .and_then(|m| m.as_str()),
                        ) {
                            let image_data = if response_format == "b64_json" {
                                ImageData {
                                    b64_json: Some(data.to_string()),
                                    url: None,
                                    revised_prompt: revised_prompt.clone(),
                                }
                            } else {
                                // 构建 data URL
                                let data_url = format!("data:{};base64,{}", mime_type, data);
                                ImageData {
                                    b64_json: None,
                                    url: Some(data_url),
                                    revised_prompt: revised_prompt.clone(),
                                }
                            };
                            images.push(image_data);
                        }
                    }
                }
            }
        }
    }

    if images.is_empty() {
        return Err("No image generated".to_string());
    }

    Ok(ImageGenerationResponse {
        created: chrono::Utc::now().timestamp(),
        data: images,
    })
}

// ============================================================================
// 图像生成 API 测试
// ============================================================================

#[cfg(test)]
mod image_tests {
    use super::*;

    #[test]
    fn test_image_model_mapping() {
        // OpenAI 兼容模型名映射到内部名称
        assert_eq!(image_model_mapping("dall-e-3"), "gemini-3-pro-image");
        assert_eq!(image_model_mapping("dall-e-2"), "gemini-3-pro-image");
        // 用户友好名称映射到内部名称
        assert_eq!(
            image_model_mapping("gemini-3-pro-image-preview"),
            "gemini-3-pro-image"
        );
        // 内部名称保持不变
        assert_eq!(
            image_model_mapping("gemini-3-pro-image"),
            "gemini-3-pro-image"
        );
        // 其他模型名称透传
        assert_eq!(image_model_mapping("other-model"), "other-model");
    }

    #[test]
    fn test_convert_image_request_basic() {
        let request = ImageGenerationRequest {
            prompt: "A cute cat".to_string(),
            model: "gemini-3-pro-image-preview".to_string(),
            n: 1,
            size: None,
            response_format: "url".to_string(),
            quality: None,
            style: None,
            user: None,
        };

        let result = convert_image_request_to_antigravity(&request, "test-project");

        // 验证基本结构
        assert_eq!(result["project"], "test-project");
        // 模型名应该映射为内部名称
        assert_eq!(result["model"], "gemini-3-pro-image");
        assert!(result["requestId"].as_str().unwrap().starts_with("img-"));

        // 验证内容
        let contents = result["request"]["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "A cute cat");

        // 验证生成配置
        let gen_config = &result["request"]["generationConfig"];
        let modalities = gen_config["responseModalities"].as_array().unwrap();
        assert!(modalities.contains(&serde_json::json!("TEXT")));
        assert!(modalities.contains(&serde_json::json!("IMAGE")));
        assert_eq!(gen_config["candidateCount"], 1);
    }

    #[test]
    fn test_convert_image_request_with_n() {
        let request = ImageGenerationRequest {
            prompt: "A beautiful sunset".to_string(),
            model: "dall-e-3".to_string(),
            n: 3,
            size: Some("1024x1024".to_string()),
            response_format: "b64_json".to_string(),
            quality: Some("hd".to_string()),
            style: Some("vivid".to_string()),
            user: None,
        };

        let result = convert_image_request_to_antigravity(&request, "project-123");

        // dall-e-3 应该映射为内部名称
        assert_eq!(result["model"], "gemini-3-pro-image");
        assert_eq!(result["request"]["generationConfig"]["candidateCount"], 3);
    }

    #[test]
    fn test_convert_antigravity_image_response_b64_json() {
        let antigravity_resp = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [
                            {"text": "Here is your image"},
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                                }
                            }
                        ]
                    }
                }]
            }
        });

        let result = convert_antigravity_image_response(&antigravity_resp, "b64_json").unwrap();

        assert!(result.created > 0);
        assert_eq!(result.data.len(), 1);
        assert!(result.data[0].b64_json.is_some());
        assert!(result.data[0].url.is_none());
        assert_eq!(
            result.data[0].revised_prompt,
            Some("Here is your image".to_string())
        );
    }

    #[test]
    fn test_convert_antigravity_image_response_url() {
        let antigravity_resp = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": "base64data"
                            }
                        }]
                    }
                }]
            }
        });

        let result = convert_antigravity_image_response(&antigravity_resp, "url").unwrap();

        assert!(result.created > 0);
        assert_eq!(result.data.len(), 1);
        assert!(result.data[0].b64_json.is_none());
        assert!(result.data[0].url.is_some());
        assert_eq!(
            result.data[0].url,
            Some("data:image/jpeg;base64,base64data".to_string())
        );
    }

    #[test]
    fn test_convert_antigravity_image_response_no_image() {
        let antigravity_resp = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Sorry, I cannot generate that image"}]
                    }
                }]
            }
        });

        let result = convert_antigravity_image_response(&antigravity_resp, "url");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "No image generated");
    }

    #[test]
    fn test_convert_antigravity_image_response_snake_case() {
        // 测试 snake_case 字段名兼容性
        let antigravity_resp = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": "testdata"
                            }
                        }]
                    }
                }]
            }
        });

        let result = convert_antigravity_image_response(&antigravity_resp, "b64_json").unwrap();
        assert_eq!(result.data.len(), 1);
        assert_eq!(result.data[0].b64_json, Some("testdata".to_string()));
    }
}

// ============================================================================
// 图像生成 API 属性测试
// ============================================================================

#[cfg(test)]
mod image_property_tests {
    use super::*;
    use proptest::prelude::*;

    // 生成随机提示词
    fn arb_prompt() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9 .,!?]{1,200}".prop_map(|s| s)
    }

    // 生成随机模型名称
    fn arb_image_model() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("dall-e-3".to_string()),
            Just("dall-e-2".to_string()),
            Just("gemini-3-pro-image-preview".to_string()),
            Just("gemini-3-pro-image".to_string()),
            Just("other-model".to_string()),
        ]
    }

    // 生成随机 n 值
    fn arb_n() -> impl Strategy<Value = u32> {
        1u32..5u32
    }

    // 生成随机响应格式
    fn arb_response_format() -> impl Strategy<Value = String> {
        prop_oneof![Just("url".to_string()), Just("b64_json".to_string()),]
    }

    // 生成随机图像请求
    fn arb_image_request() -> impl Strategy<Value = ImageGenerationRequest> {
        (
            arb_prompt(),
            arb_image_model(),
            arb_n(),
            arb_response_format(),
        )
            .prop_map(
                |(prompt, model, n, response_format)| ImageGenerationRequest {
                    prompt,
                    model,
                    n,
                    size: None,
                    response_format,
                    quality: None,
                    style: None,
                    user: None,
                },
            )
    }

    // 生成随机 Base64 数据
    fn arb_base64_data() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9+/]{10,100}={0,2}".prop_map(|s| s)
    }

    // 生成随机 MIME 类型
    fn arb_mime_type() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("image/png".to_string()),
            Just("image/jpeg".to_string()),
            Just("image/gif".to_string()),
            Just("image/webp".to_string()),
        ]
    }

    proptest! {
        /// Property 1: Request Conversion Correctness
        ///
        /// *For any* valid OpenAI image generation request with a non-empty prompt,
        /// the converted Antigravity request SHALL:
        /// - Contain the prompt text in `request.contents[0].parts[0].text`
        /// - Have `responseModalities` set to `["TEXT", "IMAGE"]`
        /// - Use the correct mapped model name
        /// - Include the specified `n` value in `candidateCount`
        ///
        /// **Feature: antigravity-image-api, Property 1: Request Conversion Correctness**
        /// **Validates: Requirements 1.2, 1.3, 1.4, 2.1, 2.2**
        #[test]
        fn prop_request_conversion_correctness(request in arb_image_request()) {
            let result = convert_image_request_to_antigravity(&request, "test-project");

            // 验证 prompt 正确传递
            let contents = result["request"]["contents"].as_array().unwrap();
            prop_assert_eq!(contents.len(), 1);
            prop_assert_eq!(contents[0]["parts"][0]["text"].as_str().unwrap(), request.prompt.as_str());

            // 验证 responseModalities 设置正确
            let modalities = result["request"]["generationConfig"]["responseModalities"]
                .as_array()
                .unwrap();
            prop_assert!(modalities.contains(&serde_json::json!("TEXT")));
            prop_assert!(modalities.contains(&serde_json::json!("IMAGE")));

            // 验证模型映射正确
            let expected_model = image_model_mapping(&request.model);
            prop_assert_eq!(result["model"].as_str().unwrap(), expected_model);

            // 验证 n 值正确传递
            prop_assert_eq!(
                result["request"]["generationConfig"]["candidateCount"].as_u64().unwrap(),
                request.n as u64
            );
        }

        /// Property 2: Response Format Correctness
        ///
        /// *For any* Antigravity response containing image data:
        /// - WHEN response_format is "b64_json", the output SHALL have `b64_json` field set and `url` field null
        /// - WHEN response_format is "url", the output SHALL have `url` field as a valid data URL and `b64_json` field null
        ///
        /// **Feature: antigravity-image-api, Property 2: Response Format Correctness**
        /// **Validates: Requirements 1.6, 1.7, 3.2, 3.3**
        #[test]
        fn prop_response_format_correctness(
            base64_data in arb_base64_data(),
            mime_type in arb_mime_type(),
            response_format in arb_response_format()
        ) {
            let antigravity_resp = serde_json::json!({
                "response": {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "inlineData": {
                                    "mimeType": mime_type.clone(),
                                    "data": base64_data.clone()
                                }
                            }]
                        }
                    }]
                }
            });

            let result = convert_antigravity_image_response(&antigravity_resp, &response_format).unwrap();

            prop_assert!(result.data.len() >= 1);

            if response_format == "b64_json" {
                // b64_json 格式
                prop_assert!(result.data[0].b64_json.is_some());
                prop_assert!(result.data[0].url.is_none());
                prop_assert_eq!(result.data[0].b64_json.as_ref().unwrap(), &base64_data);
            } else {
                // url 格式
                prop_assert!(result.data[0].b64_json.is_none());
                prop_assert!(result.data[0].url.is_some());

                // 验证 data URL 格式
                let url = result.data[0].url.as_ref().unwrap();
                let expected_url = format!("data:{};base64,{}", mime_type, base64_data);
                prop_assert_eq!(url, &expected_url);
            }
        }

        /// Property 3: OpenAI Response Structure Compliance
        ///
        /// *For any* successful image generation response:
        /// - The response SHALL have a `created` field with a valid Unix timestamp (positive integer)
        /// - The response SHALL have a `data` field that is a non-empty array
        /// - Each item in `data` SHALL have either `url` or `b64_json` field (not both)
        ///
        /// **Feature: antigravity-image-api, Property 3: OpenAI Response Structure Compliance**
        /// **Validates: Requirements 3.4, 3.5, 5.3, 5.4**
        #[test]
        fn prop_openai_response_structure(
            base64_data in arb_base64_data(),
            mime_type in arb_mime_type(),
            response_format in arb_response_format()
        ) {
            let antigravity_resp = serde_json::json!({
                "response": {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": base64_data
                                }
                            }]
                        }
                    }]
                }
            });

            let result = convert_antigravity_image_response(&antigravity_resp, &response_format).unwrap();

            // 验证 created 是有效时间戳
            prop_assert!(result.created > 0);

            // 验证 data 是非空数组
            prop_assert!(!result.data.is_empty());

            // 验证每个 item 只有 url 或 b64_json 之一
            for item in &result.data {
                let has_url = item.url.is_some();
                let has_b64 = item.b64_json.is_some();
                prop_assert!(has_url != has_b64, "Each item should have exactly one of url or b64_json");
            }
        }

        /// Property 4: Model Name Mapping Correctness
        ///
        /// *For any* model name in the request:
        /// - "dall-e-3" SHALL map to "gemini-3-pro-image"
        /// - "dall-e-2" SHALL map to "gemini-3-pro-image"
        /// - "gemini-3-pro-image-preview" SHALL map to "gemini-3-pro-image"
        /// - Other model names SHALL pass through unchanged
        ///
        /// **Feature: antigravity-image-api, Property 4: Model Name Mapping Correctness**
        /// **Validates: Requirements 2.3, 2.4**
        #[test]
        fn prop_model_name_mapping(model in arb_image_model()) {
            let mapped = image_model_mapping(&model);

            match model.as_str() {
                "dall-e-3" | "dall-e-2" | "gemini-3-pro-image-preview" => {
                    prop_assert_eq!(mapped, "gemini-3-pro-image");
                }
                other => {
                    prop_assert_eq!(mapped, other);
                }
            }
        }

        /// Property 5: Image Data Extraction
        ///
        /// *For any* Antigravity response with `inlineData` containing `data` and `mimeType`:
        /// - The converter SHALL successfully extract the Base64 data
        /// - The converter SHALL successfully extract the MIME type
        /// - If text is present alongside the image, it SHALL be included in `revised_prompt`
        ///
        /// **Feature: antigravity-image-api, Property 5: Image Data Extraction**
        /// **Validates: Requirements 3.1, 3.6**
        #[test]
        fn prop_image_data_extraction(
            base64_data in arb_base64_data(),
            mime_type in arb_mime_type(),
            text in proptest::option::of("[a-zA-Z0-9 ]{0,100}")
        ) {
            let mut parts = vec![];

            // 可选的文本部分
            if let Some(ref t) = text {
                if !t.is_empty() {
                    parts.push(serde_json::json!({"text": t}));
                }
            }

            // 图像部分
            parts.push(serde_json::json!({
                "inlineData": {
                    "mimeType": mime_type.clone(),
                    "data": base64_data.clone()
                }
            }));

            let antigravity_resp = serde_json::json!({
                "response": {
                    "candidates": [{
                        "content": {
                            "parts": parts
                        }
                    }]
                }
            });

            let result = convert_antigravity_image_response(&antigravity_resp, "b64_json").unwrap();

            // 验证 Base64 数据提取
            prop_assert_eq!(result.data[0].b64_json.as_ref().unwrap(), &base64_data);

            // 验证 revised_prompt
            if let Some(ref t) = text {
                if !t.is_empty() {
                    prop_assert_eq!(result.data[0].revised_prompt.as_ref().unwrap(), t);
                }
            }
        }
    }
}
