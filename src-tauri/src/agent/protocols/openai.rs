//! OpenAI 协议实现
//!
//! 实现 OpenAI Chat Completions API 协议
//! 适用于 OpenAI、Qwen、Codex、Antigravity、IFlow、Kiro 等兼容服务

use super::Protocol;
use crate::agent::parsers::OpenAISSEParser;
use crate::agent::types::{
    AgentConfig, AgentMessage, ContentPart, ImageData, MessageContent, StreamEvent, StreamResult,
};
use crate::models::openai::{
    ChatCompletionRequest, ChatMessage, ContentPart as OpenAIContentPart,
    MessageContent as OpenAIMessageContent, Tool,
};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

/// OpenAI 协议处理器
pub struct OpenAIProtocol;

impl OpenAIProtocol {
    /// 将 AgentMessage 转换为 OpenAI ChatMessage
    fn convert_to_chat_message(msg: &AgentMessage) -> ChatMessage {
        let content = match &msg.content {
            MessageContent::Text(text) => Some(OpenAIMessageContent::Text(text.clone())),
            MessageContent::Parts(parts) => {
                let openai_parts: Vec<OpenAIContentPart> = parts
                    .iter()
                    .map(|p| match p {
                        ContentPart::Text { text } => {
                            OpenAIContentPart::Text { text: text.clone() }
                        }
                        ContentPart::ImageUrl { image_url } => OpenAIContentPart::ImageUrl {
                            image_url: crate::models::openai::ImageUrl {
                                url: image_url.url.clone(),
                                detail: image_url.detail.clone(),
                            },
                        },
                    })
                    .collect();
                Some(OpenAIMessageContent::Parts(openai_parts))
            }
        };

        ChatMessage {
            role: msg.role.clone(),
            content,
            tool_calls: msg.tool_calls.as_ref().map(|calls| {
                calls
                    .iter()
                    .map(|tc| crate::models::openai::ToolCall {
                        id: tc.id.clone(),
                        call_type: tc.call_type.clone(),
                        function: crate::models::openai::FunctionCall {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    })
                    .collect()
            }),
            tool_call_id: msg.tool_call_id.clone(),
        }
    }

    /// 构建消息列表
    fn build_messages(
        history: &[AgentMessage],
        user_message: &str,
        images: Option<&[ImageData]>,
        config: &AgentConfig,
    ) -> Vec<ChatMessage> {
        let mut messages = Vec::new();

        // 添加系统提示词
        if let Some(prompt) = &config.system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(OpenAIMessageContent::Text(prompt.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // 添加历史消息
        for msg in history {
            messages.push(Self::convert_to_chat_message(msg));
        }

        // 添加当前用户消息
        let user_msg = if let Some(imgs) = images {
            let mut parts = vec![OpenAIContentPart::Text {
                text: user_message.to_string(),
            }];

            for img in imgs {
                parts.push(OpenAIContentPart::ImageUrl {
                    image_url: crate::models::openai::ImageUrl {
                        url: format!("data:{};base64,{}", img.media_type, img.data),
                        detail: None,
                    },
                });
            }

            ChatMessage {
                role: "user".to_string(),
                content: Some(OpenAIMessageContent::Parts(parts)),
                tool_calls: None,
                tool_call_id: None,
            }
        } else {
            ChatMessage {
                role: "user".to_string(),
                content: Some(OpenAIMessageContent::Text(user_message.to_string())),
                tool_calls: None,
                tool_call_id: None,
            }
        };

        messages.push(user_msg);
        messages
    }

    /// 从历史构建消息（不添加新用户消息）
    fn build_messages_from_history(
        history: &[AgentMessage],
        config: &AgentConfig,
    ) -> Vec<ChatMessage> {
        let mut messages = Vec::new();

        // 添加系统提示词
        if let Some(prompt) = &config.system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(OpenAIMessageContent::Text(prompt.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // 添加所有历史消息
        for msg in history {
            messages.push(Self::convert_to_chat_message(msg));
        }

        messages
    }

    /// 处理 SSE 流
    async fn process_stream(
        response: reqwest::Response,
        tx: mpsc::Sender<StreamEvent>,
        send_done: bool,
    ) -> Result<StreamResult, String> {
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut parser = OpenAISSEParser::new();
        let mut final_usage = None;

        eprintln!("[OpenAIProtocol] 开始处理 SSE 流...");

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    eprintln!(
                        "[OpenAIProtocol] 收到 chunk: {} bytes, 内容: {}",
                        bytes.len(),
                        if text.len() > 200 {
                            format!("{}...", &text[..200])
                        } else {
                            text.to_string()
                        }
                    );
                    buffer.push_str(&text);

                    // 处理完整的 SSE 事件（以 \n\n 分隔）
                    while let Some(pos) = buffer.find("\n\n") {
                        let event = buffer[..pos].to_string();
                        buffer = buffer[pos + 2..].to_string();

                        for line in event.lines() {
                            if let Some(data) = line.strip_prefix("data: ") {
                                debug!("[OpenAIProtocol] SSE data: {}", data);
                                let (text_delta, is_done, usage) = parser.parse_data(data);

                                if usage.is_some() {
                                    final_usage = usage;
                                }

                                if let Some(text) = text_delta {
                                    let _ = tx.send(StreamEvent::TextDelta { text }).await;
                                }

                                if is_done {
                                    let full_content = parser.get_full_content();
                                    let tool_calls = if parser.has_tool_calls() {
                                        Some(parser.finalize_tool_calls())
                                    } else {
                                        None
                                    };

                                    if send_done {
                                        let _ = tx
                                            .send(StreamEvent::Done {
                                                usage: final_usage.clone(),
                                            })
                                            .await;
                                    }

                                    return Ok(StreamResult {
                                        content: full_content,
                                        tool_calls,
                                        usage: final_usage,
                                    });
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("[OpenAIProtocol] 流读取错误: {}", e);
                    let _ = tx
                        .send(StreamEvent::Error {
                            message: format!("流读取错误: {}", e),
                        })
                        .await;
                    return Err(format!("流读取错误: {}", e));
                }
            }
        }

        // 流正常结束但没有收到 [DONE]
        let full_content = parser.get_full_content();
        let tool_calls = if parser.has_tool_calls() {
            Some(parser.finalize_tool_calls())
        } else {
            None
        };

        if send_done {
            let _ = tx
                .send(StreamEvent::Done {
                    usage: final_usage.clone(),
                })
                .await;
        }

        Ok(StreamResult {
            content: full_content,
            tool_calls,
            usage: final_usage,
        })
    }
}

#[async_trait]
impl Protocol for OpenAIProtocol {
    async fn chat_stream(
        &self,
        client: &Client,
        base_url: &str,
        api_key: &str,
        messages: &[AgentMessage],
        user_message: &str,
        images: Option<&[ImageData]>,
        model: &str,
        config: &AgentConfig,
        tools: Option<&[Tool]>,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<StreamResult, String> {
        info!(
            "[OpenAIProtocol] 发送流式请求: model={}, history_len={}, tools_count={}",
            model,
            messages.len(),
            tools.map(|t| t.len()).unwrap_or(0)
        );

        let chat_messages = Self::build_messages(messages, user_message, images, config);

        let request = ChatCompletionRequest {
            model: model.to_string(),
            messages: chat_messages,
            stream: true,
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: None,
            tools: tools.map(|t| t.to_vec()),
            tool_choice: if tools.is_some() {
                Some(serde_json::json!("auto"))
            } else {
                None
            },
            reasoning_effort: None,
        };

        let url = format!("{}{}", base_url, self.endpoint());

        eprintln!(
            "[OpenAIProtocol] 发送请求到: {} model={} stream={}",
            url, model, request.stream
        );

        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                eprintln!("[OpenAIProtocol] 请求发送失败: {}", e);
                format!("请求失败: {}", e)
            })?;

        let status = response.status();
        eprintln!("[OpenAIProtocol] 响应状态: {}", status);
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!("[OpenAIProtocol] 请求失败: {} - {}", status, body);
            let _ = tx
                .send(StreamEvent::Error {
                    message: format!("API 错误 ({}): {}", status, body),
                })
                .await;
            return Err(format!("API 错误: {}", status));
        }

        Self::process_stream(response, tx, true).await
    }

    async fn chat_stream_continue(
        &self,
        client: &Client,
        base_url: &str,
        api_key: &str,
        messages: &[AgentMessage],
        model: &str,
        config: &AgentConfig,
        tools: Option<&[Tool]>,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<StreamResult, String> {
        debug!(
            "[OpenAIProtocol] 继续流式对话: model={}, history_len={}, tools_count={}",
            model,
            messages.len(),
            tools.map(|t| t.len()).unwrap_or(0)
        );

        let chat_messages = Self::build_messages_from_history(messages, config);

        let request = ChatCompletionRequest {
            model: model.to_string(),
            messages: chat_messages,
            stream: true,
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: None,
            tools: tools.map(|t| t.to_vec()),
            tool_choice: if tools.is_some() {
                Some(serde_json::json!("auto"))
            } else {
                None
            },
            reasoning_effort: None,
        };

        let url = format!("{}{}", base_url, self.endpoint());

        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("请求失败: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!("[OpenAIProtocol] 请求失败: {} - {}", status, body);
            let _ = tx
                .send(StreamEvent::Error {
                    message: format!("API 错误 ({}): {}", status, body),
                })
                .await;
            return Err(format!("API 错误: {}", status));
        }

        // 继续对话时不发送 Done 事件（工具循环可能还会继续）
        Self::process_stream(response, tx, false).await
    }

    fn endpoint(&self) -> &'static str {
        "/v1/chat/completions"
    }
}
