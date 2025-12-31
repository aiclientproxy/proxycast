//! 原生 Rust Agent 实现
//!
//! 支持连续对话（Conversation History）和工具调用（Tools）
//! 参考 goose 项目的 Agent 设计
//!
//! ## 流式处理
//! - 实现 SSE 流解析，支持 text_delta 和 tool_calls 解析
//! - Requirements: 1.1, 1.3, 1.4
//!
//! ## 工具调用循环
//! - 实现工具调用检测、执行和结果收集
//! - Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6

use crate::agent::tool_loop::{ToolCallResult, ToolLoopConfig, ToolLoopEngine, ToolLoopState};
use crate::agent::tools::ToolRegistry;
use crate::agent::types::*;
use crate::models::openai::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ContentPart as OpenAIContentPart,
    MessageContent as OpenAIMessageContent,
};
use futures::StreamExt;
use parking_lot::RwLock;
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// SSE 流解析器
///
/// 解析 Server-Sent Events 流，提取 text_delta 和 tool_calls
/// Requirements: 1.1, 1.3, 1.4
#[derive(Debug, Default)]
struct SSEParser {
    /// 累积的完整内容
    full_content: String,
    /// 累积的工具调用
    tool_calls: Vec<ToolCallDelta>,
    /// 当前正在构建的工具调用索引
    current_tool_indices: HashMap<usize, ToolCallDelta>,
}

/// 工具调用增量数据
#[derive(Debug, Clone, Default)]
struct ToolCallDelta {
    /// 工具调用索引
    index: usize,
    /// 工具调用 ID
    id: String,
    /// 工具类型
    call_type: String,
    /// 函数名
    function_name: String,
    /// 函数参数（累积的 JSON 字符串）
    function_arguments: String,
}

impl SSEParser {
    fn new() -> Self {
        Self::default()
    }

    /// 解析 SSE 数据行
    ///
    /// 返回 (text_delta, is_done, usage)
    fn parse_data(&mut self, data: &str) -> (Option<String>, bool, Option<TokenUsage>) {
        if data.trim() == "[DONE]" {
            return (None, true, None);
        }

        let json: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                warn!("[SSEParser] 解析 JSON 失败: {} - data: {}", e, data);
                return (None, false, None);
            }
        };

        // 提取 usage 信息（如果存在）
        let usage = json.get("usage").and_then(|u| {
            let input = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let output = u
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            if input > 0 || output > 0 {
                Some(TokenUsage::new(input, output))
            } else {
                None
            }
        });

        // 检查是否有 choices
        let choices = match json.get("choices").and_then(|c| c.as_array()) {
            Some(c) => c,
            None => return (None, false, usage),
        };

        if choices.is_empty() {
            return (None, false, usage);
        }

        let choice = &choices[0];
        let delta = match choice.get("delta") {
            Some(d) => d,
            None => return (None, false, usage),
        };

        // 检查 finish_reason
        let finish_reason = choice
            .get("finish_reason")
            .and_then(|f| f.as_str())
            .unwrap_or("");
        let is_done = finish_reason == "stop" || finish_reason == "tool_calls";

        // 提取文本内容
        let text_delta = delta
            .get("content")
            .and_then(|c| c.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| {
                self.full_content.push_str(s);
                s.to_string()
            });

        // 提取工具调用
        if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
            for tc in tool_calls {
                self.parse_tool_call_delta(tc);
            }
        }

        (text_delta, is_done, usage)
    }

    /// 解析工具调用增量
    fn parse_tool_call_delta(&mut self, tc: &Value) {
        let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;

        // 获取或创建工具调用
        let tool_call = self
            .current_tool_indices
            .entry(index)
            .or_insert_with(|| ToolCallDelta {
                index,
                ..Default::default()
            });

        // 更新 ID
        if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
            tool_call.id = id.to_string();
        }

        // 更新类型
        if let Some(t) = tc.get("type").and_then(|t| t.as_str()) {
            tool_call.call_type = t.to_string();
        }

        // 更新函数信息
        if let Some(function) = tc.get("function") {
            if let Some(name) = function.get("name").and_then(|n| n.as_str()) {
                tool_call.function_name = name.to_string();
            }
            if let Some(args) = function.get("arguments").and_then(|a| a.as_str()) {
                tool_call.function_arguments.push_str(args);
            }
        }
    }

    /// 完成解析，返回最终的工具调用列表
    fn finalize_tool_calls(&mut self) -> Vec<ToolCall> {
        // 按索引排序并转换为 ToolCall
        let mut indices: Vec<_> = self.current_tool_indices.keys().cloned().collect();
        indices.sort();

        indices
            .into_iter()
            .filter_map(|idx| {
                let delta = self.current_tool_indices.get(&idx)?;
                if delta.id.is_empty() || delta.function_name.is_empty() {
                    return None;
                }
                Some(ToolCall {
                    id: delta.id.clone(),
                    call_type: if delta.call_type.is_empty() {
                        "function".to_string()
                    } else {
                        delta.call_type.clone()
                    },
                    function: FunctionCall {
                        name: delta.function_name.clone(),
                        arguments: delta.function_arguments.clone(),
                    },
                })
            })
            .collect()
    }

    /// 获取完整内容
    fn get_full_content(&self) -> String {
        self.full_content.clone()
    }

    /// 是否有工具调用
    fn has_tool_calls(&self) -> bool {
        !self.current_tool_indices.is_empty()
    }
}

/// 原生 Agent 实现
pub struct NativeAgent {
    client: Client,
    base_url: String,
    api_key: String,
    sessions: Arc<RwLock<HashMap<String, AgentSession>>>,
    config: AgentConfig,
}

impl NativeAgent {
    pub fn new(base_url: String, api_key: String) -> Result<Self, String> {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(30))
            .no_proxy()
            .build()
            .map_err(|e| format!("创建 HTTP 客户端失败: {}", e))?;

        Ok(Self {
            client,
            base_url,
            api_key,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config: AgentConfig::default(),
        })
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.config.model = model;
        self
    }

    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.config.system_prompt = Some(prompt);
        self
    }

    /// 将 AgentMessage 转换为 OpenAI ChatMessage
    fn convert_to_chat_message(&self, msg: &AgentMessage) -> ChatMessage {
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

    /// 构建完整的消息列表（包含历史）
    fn build_messages_with_history(
        &self,
        session: &AgentSession,
        user_message: &str,
        images: Option<&[ImageData]>,
    ) -> Vec<ChatMessage> {
        let mut messages = Vec::new();

        // 1. 添加系统提示词
        let system_prompt = session
            .system_prompt
            .as_ref()
            .or(self.config.system_prompt.as_ref());
        if let Some(prompt) = system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(OpenAIMessageContent::Text(prompt.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // 2. 添加历史消息
        for msg in &session.messages {
            messages.push(self.convert_to_chat_message(msg));
        }

        // 3. 添加当前用户消息
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

    /// 发送聊天请求（支持连续对话）
    pub async fn chat(&self, request: NativeChatRequest) -> Result<NativeChatResponse, String> {
        let model = request.model.unwrap_or_else(|| self.config.model.clone());
        let session_id = request.session_id.clone();
        let has_images = request.images.as_ref().map(|i| i.len()).unwrap_or(0);

        info!(
            "[NativeAgent] 发送聊天请求: model={}, session={:?}, images={}",
            model, session_id, has_images
        );

        // 获取或创建会话
        let session = if let Some(sid) = &session_id {
            self.sessions.read().get(sid).cloned()
        } else {
            None
        };

        let messages = if let Some(ref sess) = session {
            // 使用会话历史构建消息
            self.build_messages_with_history(sess, &request.message, request.images.as_deref())
        } else {
            // 无会话，单次对话
            self.build_single_messages(&request.message, request.images.as_deref())
        };

        // 打印消息结构用于调试
        for (i, msg) in messages.iter().enumerate() {
            let content_type = match &msg.content {
                Some(OpenAIMessageContent::Text(_)) => "text",
                Some(OpenAIMessageContent::Parts(parts)) => {
                    let has_image = parts
                        .iter()
                        .any(|p| matches!(p, OpenAIContentPart::ImageUrl { .. }));
                    if has_image {
                        "parts_with_image"
                    } else {
                        "parts_text_only"
                    }
                }
                None => "none",
            };
            debug!(
                "[NativeAgent] 消息[{}]: role={}, content_type={}",
                i, msg.role, content_type
            );
        }

        let chat_request = ChatCompletionRequest {
            model: model.clone(),
            messages,
            stream: false,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            top_p: None,
            tools: None, // TODO: 添加工具支持
            tool_choice: None,
            reasoning_effort: None,
        };

        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&chat_request)
            .send()
            .await
            .map_err(|e| format!("请求失败: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!("[NativeAgent] 请求失败: {} - {}", status, body);
            return Ok(NativeChatResponse {
                content: String::new(),
                model,
                usage: None,
                success: false,
                error: Some(format!("API 错误 ({}): {}", status, body)),
            });
        }

        let body: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| format!("解析响应失败: {}", e))?;

        let content = body
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let usage = Some(TokenUsage {
            input_tokens: body.usage.prompt_tokens,
            output_tokens: body.usage.completion_tokens,
        });

        // 更新会话历史
        if let Some(sid) = session_id {
            self.add_message_to_session(
                &sid,
                "user",
                MessageContent::Text(request.message.clone()),
                request.images.as_deref(),
            );
            self.add_message_to_session(
                &sid,
                "assistant",
                MessageContent::Text(content.clone()),
                None,
            );
        }

        info!("[NativeAgent] 聊天完成: content_len={}", content.len());

        Ok(NativeChatResponse {
            content,
            model: body.model,
            usage,
            success: true,
            error: None,
        })
    }

    /// 构建单次对话消息（无历史）
    fn build_single_messages(
        &self,
        user_message: &str,
        images: Option<&[ImageData]>,
    ) -> Vec<ChatMessage> {
        let mut messages = Vec::new();

        if let Some(system_prompt) = &self.config.system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(OpenAIMessageContent::Text(system_prompt.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }

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

    /// 添加消息到会话
    fn add_message_to_session(
        &self,
        session_id: &str,
        role: &str,
        content: MessageContent,
        images: Option<&[ImageData]>,
    ) {
        let mut sessions = self.sessions.write();
        if let Some(session) = sessions.get_mut(session_id) {
            let final_content = if let Some(imgs) = images {
                // 如果有图片，转换为 Parts
                let mut parts = vec![ContentPart::Text {
                    text: content.as_text(),
                }];
                for img in imgs {
                    parts.push(ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: format!("data:{};base64,{}", img.media_type, img.data),
                            detail: None,
                        },
                    });
                }
                MessageContent::Parts(parts)
            } else {
                content
            };

            session.messages.push(AgentMessage {
                role: role.to_string(),
                content: final_content,
                timestamp: chrono::Utc::now().to_rfc3339(),
                tool_calls: None,
                tool_call_id: None,
            });
            session.updated_at = chrono::Utc::now().to_rfc3339();
        }
    }

    /// 流式聊天（支持连续对话）
    ///
    /// 实现 SSE 流解析，支持 text_delta 和 tool_calls 解析
    /// Requirements: 1.1, 1.3, 1.4
    pub async fn chat_stream(
        &self,
        request: NativeChatRequest,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<StreamResult, String> {
        let model = request.model.unwrap_or_else(|| self.config.model.clone());
        let session_id = request.session_id.clone();

        debug!(
            "[NativeAgent] 发送流式聊天请求: model={}, session={:?}",
            model, session_id
        );

        // 获取会话
        let session = if let Some(sid) = &session_id {
            self.sessions.read().get(sid).cloned()
        } else {
            None
        };

        let messages = if let Some(ref sess) = session {
            self.build_messages_with_history(sess, &request.message, request.images.as_deref())
        } else {
            self.build_single_messages(&request.message, request.images.as_deref())
        };

        let chat_request = ChatCompletionRequest {
            model: model.clone(),
            messages,
            stream: true,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            top_p: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
        };

        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&chat_request)
            .send()
            .await
            .map_err(|e| format!("请求失败: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!("[NativeAgent] 流式请求失败: {} - {}", status, body);
            let _ = tx
                .send(StreamEvent::Error {
                    message: format!("API 错误 ({}): {}", status, body),
                })
                .await;
            return Err(format!("API 错误: {}", status));
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut parser = SSEParser::new();
        let mut final_usage: Option<TokenUsage> = None;

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    buffer.push_str(&text);

                    // 处理完整的 SSE 事件（以 \n\n 分隔）
                    while let Some(pos) = buffer.find("\n\n") {
                        let event = buffer[..pos].to_string();
                        buffer = buffer[pos + 2..].to_string();

                        for line in event.lines() {
                            if let Some(data) = line.strip_prefix("data: ") {
                                let (text_delta, is_done, usage) = parser.parse_data(data);

                                // 更新 usage
                                if usage.is_some() {
                                    final_usage = usage;
                                }

                                // 发送文本增量
                                if let Some(text) = text_delta {
                                    let _ = tx.send(StreamEvent::TextDelta { text }).await;
                                }

                                // 检查是否完成
                                if is_done {
                                    // 获取最终结果
                                    let full_content = parser.get_full_content();
                                    let tool_calls = if parser.has_tool_calls() {
                                        Some(parser.finalize_tool_calls())
                                    } else {
                                        None
                                    };

                                    // 更新会话历史
                                    if let Some(sid) = &session_id {
                                        self.add_message_to_session(
                                            sid,
                                            "user",
                                            MessageContent::Text(request.message.clone()),
                                            request.images.as_deref(),
                                        );
                                        self.add_assistant_message_to_session(
                                            sid,
                                            MessageContent::Text(full_content.clone()),
                                            tool_calls.clone(),
                                        );
                                    }

                                    // 发送完成事件
                                    let _ = tx
                                        .send(StreamEvent::Done {
                                            usage: final_usage.clone(),
                                        })
                                        .await;

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
                    error!("[NativeAgent] 流读取错误: {}", e);
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

        // 更新会话历史
        if let Some(sid) = &session_id {
            self.add_message_to_session(
                sid,
                "user",
                MessageContent::Text(request.message.clone()),
                request.images.as_deref(),
            );
            self.add_assistant_message_to_session(
                sid,
                MessageContent::Text(full_content.clone()),
                tool_calls.clone(),
            );
        }

        let _ = tx
            .send(StreamEvent::Done {
                usage: final_usage.clone(),
            })
            .await;

        Ok(StreamResult {
            content: full_content,
            tool_calls,
            usage: final_usage,
        })
    }

    /// 添加 assistant 消息到会话（支持工具调用）
    fn add_assistant_message_to_session(
        &self,
        session_id: &str,
        content: MessageContent,
        tool_calls: Option<Vec<ToolCall>>,
    ) {
        let mut sessions = self.sessions.write();
        if let Some(session) = sessions.get_mut(session_id) {
            session.messages.push(AgentMessage {
                role: "assistant".to_string(),
                content,
                timestamp: chrono::Utc::now().to_rfc3339(),
                tool_calls,
                tool_call_id: None,
            });
            session.updated_at = chrono::Utc::now().to_rfc3339();
        }
    }

    /// 添加工具结果消息到会话
    ///
    /// Requirements: 7.2 - THE Tool_Loop SHALL send tool results back to the Agent as tool role messages
    fn add_tool_result_to_session(&self, session_id: &str, tool_result: &ToolCallResult) {
        let mut sessions = self.sessions.write();
        if let Some(session) = sessions.get_mut(session_id) {
            session.messages.push(tool_result.to_agent_message());
            session.updated_at = chrono::Utc::now().to_rfc3339();
        }
    }

    /// 流式聊天（支持工具调用循环）
    ///
    /// 实现完整的工具调用循环：
    /// 1. 发送请求到 LLM
    /// 2. 如果响应包含工具调用，执行工具
    /// 3. 将工具结果发送回 LLM
    /// 4. 重复直到 LLM 产生最终响应或达到最大迭代次数
    ///
    /// Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
    pub async fn chat_stream_with_tools(
        &self,
        request: NativeChatRequest,
        tx: mpsc::Sender<StreamEvent>,
        tool_loop_engine: &ToolLoopEngine,
    ) -> Result<StreamResult, String> {
        let session_id = request.session_id.clone();
        let mut state = ToolLoopState::new();

        // 首次请求
        let mut current_result = self.chat_stream(request.clone(), tx.clone()).await?;

        // 工具调用循环
        // Requirements: 7.3 - THE Tool_Loop SHALL continue until the Agent produces a final response without tool_calls
        while tool_loop_engine.should_continue(&current_result, state.iteration) {
            state.increment_iteration();

            let tool_calls = current_result.tool_calls.as_ref().unwrap();
            state.add_tool_calls(tool_calls.len());

            info!(
                "[NativeAgent] 工具循环迭代 {}: 执行 {} 个工具调用",
                state.iteration,
                tool_calls.len()
            );

            // 执行所有工具调用
            // Requirements: 7.1 - THE Tool_Loop SHALL execute each tool and collect results
            // Requirements: 7.6 - WHILE the Tool_Loop is executing, THE Frontend SHALL display the current tool
            let tool_results = tool_loop_engine
                .execute_all_tool_calls(tool_calls, Some(&tx))
                .await;

            // 将工具结果添加到会话
            // Requirements: 7.2 - THE Tool_Loop SHALL send tool results back to the Agent as tool role messages
            if let Some(sid) = &session_id {
                for result in &tool_results {
                    self.add_tool_result_to_session(sid, result);
                }
            }

            // 构建继续对话的请求
            let continue_request = NativeChatRequest {
                session_id: session_id.clone(),
                message: String::new(), // 空消息，因为我们使用会话历史
                model: request.model.clone(),
                images: None,
                stream: true,
            };

            // 继续对话
            current_result = self
                .chat_stream_continue(continue_request, tx.clone())
                .await?;
        }

        // 检查是否因为达到最大迭代次数而停止
        // Requirements: 7.5 - THE Tool_Loop SHALL enforce a maximum iteration limit
        if state.iteration >= tool_loop_engine.max_iterations() && current_result.has_tool_calls() {
            warn!(
                "[NativeAgent] 达到最大迭代次数 {}，强制停止工具循环",
                tool_loop_engine.max_iterations()
            );
            let _ = tx
                .send(StreamEvent::Error {
                    message: format!(
                        "达到最大工具调用迭代次数限制 ({})",
                        tool_loop_engine.max_iterations()
                    ),
                })
                .await;
        }

        state.mark_completed(current_result.content.clone());

        info!(
            "[NativeAgent] 工具循环完成: {} 次迭代, {} 个工具调用",
            state.iteration, state.total_tool_calls
        );

        Ok(current_result)
    }

    /// 继续流式对话（使用会话历史）
    ///
    /// 用于工具调用循环中继续对话
    async fn chat_stream_continue(
        &self,
        request: NativeChatRequest,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<StreamResult, String> {
        let model = request.model.unwrap_or_else(|| self.config.model.clone());
        let session_id = request.session_id.as_ref().ok_or("需要 session_id")?;

        debug!(
            "[NativeAgent] 继续流式对话: model={}, session={}",
            model, session_id
        );

        // 获取会话
        let session = self
            .sessions
            .read()
            .get(session_id)
            .cloned()
            .ok_or_else(|| format!("会话不存在: {}", session_id))?;

        // 构建消息（使用会话历史，不添加新的用户消息）
        let messages = self.build_messages_from_session(&session);

        let chat_request = ChatCompletionRequest {
            model: model.clone(),
            messages,
            stream: true,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            top_p: None,
            tools: None, // TODO: 添加工具定义
            tool_choice: None,
            reasoning_effort: None,
        };

        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&chat_request)
            .send()
            .await
            .map_err(|e| format!("请求失败: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!("[NativeAgent] 流式请求失败: {} - {}", status, body);
            let _ = tx
                .send(StreamEvent::Error {
                    message: format!("API 错误 ({}): {}", status, body),
                })
                .await;
            return Err(format!("API 错误: {}", status));
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut parser = SSEParser::new();
        let mut final_usage: Option<TokenUsage> = None;

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    buffer.push_str(&text);

                    while let Some(pos) = buffer.find("\n\n") {
                        let event = buffer[..pos].to_string();
                        buffer = buffer[pos + 2..].to_string();

                        for line in event.lines() {
                            if let Some(data) = line.strip_prefix("data: ") {
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

                                    // 更新会话历史
                                    self.add_assistant_message_to_session(
                                        session_id,
                                        MessageContent::Text(full_content.clone()),
                                        tool_calls.clone(),
                                    );

                                    // 不发送 Done 事件，因为工具循环可能还会继续

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
                    error!("[NativeAgent] 流读取错误: {}", e);
                    let _ = tx
                        .send(StreamEvent::Error {
                            message: format!("流读取错误: {}", e),
                        })
                        .await;
                    return Err(format!("流读取错误: {}", e));
                }
            }
        }

        // 流正常结束
        let full_content = parser.get_full_content();
        let tool_calls = if parser.has_tool_calls() {
            Some(parser.finalize_tool_calls())
        } else {
            None
        };

        self.add_assistant_message_to_session(
            session_id,
            MessageContent::Text(full_content.clone()),
            tool_calls.clone(),
        );

        Ok(StreamResult {
            content: full_content,
            tool_calls,
            usage: final_usage,
        })
    }

    /// 从会话构建消息列表（不添加新的用户消息）
    fn build_messages_from_session(&self, session: &AgentSession) -> Vec<ChatMessage> {
        let mut messages = Vec::new();

        // 添加系统提示词
        let system_prompt = session
            .system_prompt
            .as_ref()
            .or(self.config.system_prompt.as_ref());
        if let Some(prompt) = system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(OpenAIMessageContent::Text(prompt.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // 添加所有历史消息
        for msg in &session.messages {
            messages.push(self.convert_to_chat_message(msg));
        }

        messages
    }

    pub fn create_session(&self, model: Option<String>, system_prompt: Option<String>) -> String {
        let session_id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let session = AgentSession {
            id: session_id.clone(),
            model: model.unwrap_or_else(|| self.config.model.clone()),
            messages: Vec::new(),
            system_prompt,
            created_at: now.clone(),
            updated_at: now,
        };

        self.sessions.write().insert(session_id.clone(), session);
        info!("[NativeAgent] 创建会话: {}", session_id);

        session_id
    }

    pub fn get_session(&self, session_id: &str) -> Option<AgentSession> {
        self.sessions.read().get(session_id).cloned()
    }

    pub fn delete_session(&self, session_id: &str) -> bool {
        self.sessions.write().remove(session_id).is_some()
    }

    pub fn list_sessions(&self) -> Vec<AgentSession> {
        self.sessions.read().values().cloned().collect()
    }

    pub fn clear_session_messages(&self, session_id: &str) -> bool {
        let mut sessions = self.sessions.write();
        if let Some(session) = sessions.get_mut(session_id) {
            session.messages.clear();
            session.updated_at = chrono::Utc::now().to_rfc3339();
            true
        } else {
            false
        }
    }

    pub fn get_session_messages(&self, session_id: &str) -> Option<Vec<AgentMessage>> {
        self.sessions
            .read()
            .get(session_id)
            .map(|s| s.messages.clone())
    }
}

/// Tauri 状态：原生 Agent 管理器
#[derive(Clone, Default)]
pub struct NativeAgentState {
    agent: Arc<RwLock<Option<NativeAgent>>>,
}

impl NativeAgentState {
    pub fn new() -> Self {
        Self {
            agent: Arc::new(RwLock::new(None)),
        }
    }

    pub fn init(&self, base_url: String, api_key: String) -> Result<(), String> {
        let agent = NativeAgent::new(base_url, api_key)?;
        *self.agent.write() = Some(agent);
        Ok(())
    }

    pub fn is_initialized(&self) -> bool {
        self.agent.read().is_some()
    }

    pub fn reset(&self) {
        *self.agent.write() = None;
    }

    pub async fn chat(&self, request: NativeChatRequest) -> Result<NativeChatResponse, String> {
        let (base_url, api_key, config, sessions) = {
            let guard = self.agent.read();
            let agent = guard.as_ref().ok_or_else(|| "Agent 未初始化".to_string())?;
            (
                agent.base_url.clone(),
                agent.api_key.clone(),
                agent.config.clone(),
                agent.sessions.clone(),
            )
        };

        // 创建临时 Agent，共享 sessions
        let temp_agent = NativeAgent {
            client: Client::builder()
                .timeout(Duration::from_secs(300))
                .connect_timeout(Duration::from_secs(30))
                .no_proxy()
                .build()
                .map_err(|e| format!("创建 HTTP 客户端失败: {}", e))?,
            base_url,
            api_key,
            sessions,
            config,
        };

        temp_agent.chat(request).await
    }

    pub async fn chat_stream(
        &self,
        request: NativeChatRequest,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<StreamResult, String> {
        let (base_url, api_key, config, sessions) = {
            let guard = self.agent.read();
            let agent = guard.as_ref().ok_or_else(|| "Agent 未初始化".to_string())?;
            (
                agent.base_url.clone(),
                agent.api_key.clone(),
                agent.config.clone(),
                agent.sessions.clone(),
            )
        };

        let temp_agent = NativeAgent {
            client: Client::builder()
                .timeout(Duration::from_secs(300))
                .connect_timeout(Duration::from_secs(30))
                .no_proxy()
                .build()
                .map_err(|e| format!("创建 HTTP 客户端失败: {}", e))?,
            base_url,
            api_key,
            sessions,
            config,
        };

        temp_agent.chat_stream(request, tx).await
    }

    /// 流式聊天（支持工具调用循环）
    ///
    /// Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
    pub async fn chat_stream_with_tools(
        &self,
        request: NativeChatRequest,
        tx: mpsc::Sender<StreamEvent>,
        tool_loop_engine: &ToolLoopEngine,
    ) -> Result<StreamResult, String> {
        let (base_url, api_key, config, sessions) = {
            let guard = self.agent.read();
            let agent = guard.as_ref().ok_or_else(|| "Agent 未初始化".to_string())?;
            (
                agent.base_url.clone(),
                agent.api_key.clone(),
                agent.config.clone(),
                agent.sessions.clone(),
            )
        };

        let temp_agent = NativeAgent {
            client: Client::builder()
                .timeout(Duration::from_secs(300))
                .connect_timeout(Duration::from_secs(30))
                .no_proxy()
                .build()
                .map_err(|e| format!("创建 HTTP 客户端失败: {}", e))?,
            base_url,
            api_key,
            sessions,
            config,
        };

        temp_agent
            .chat_stream_with_tools(request, tx, tool_loop_engine)
            .await
    }

    pub fn create_session(
        &self,
        model: Option<String>,
        system_prompt: Option<String>,
    ) -> Result<String, String> {
        let guard = self.agent.read();
        let agent = guard.as_ref().ok_or_else(|| "Agent 未初始化".to_string())?;
        Ok(agent.create_session(model, system_prompt))
    }

    pub fn get_session(&self, session_id: &str) -> Result<Option<AgentSession>, String> {
        let guard = self.agent.read();
        let agent = guard.as_ref().ok_or_else(|| "Agent 未初始化".to_string())?;
        Ok(agent.get_session(session_id))
    }

    pub fn delete_session(&self, session_id: &str) -> bool {
        let guard = self.agent.read();
        if let Some(agent) = guard.as_ref() {
            agent.delete_session(session_id)
        } else {
            false
        }
    }

    pub fn list_sessions(&self) -> Vec<AgentSession> {
        let guard = self.agent.read();
        if let Some(agent) = guard.as_ref() {
            agent.list_sessions()
        } else {
            Vec::new()
        }
    }

    pub fn clear_session_messages(&self, session_id: &str) -> bool {
        let guard = self.agent.read();
        if let Some(agent) = guard.as_ref() {
            agent.clear_session_messages(session_id)
        } else {
            false
        }
    }

    pub fn get_session_messages(&self, session_id: &str) -> Option<Vec<AgentMessage>> {
        let guard = self.agent.read();
        guard
            .as_ref()
            .and_then(|a| a.get_session_messages(session_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_parser_text_delta() {
        let mut parser = SSEParser::new();

        // 模拟 SSE 数据
        let data1 = r#"{"choices":[{"delta":{"content":"Hello"}}]}"#;
        let data2 = r#"{"choices":[{"delta":{"content":" World"}}]}"#;
        let data3 = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;

        let (text1, done1, _) = parser.parse_data(data1);
        assert_eq!(text1, Some("Hello".to_string()));
        assert!(!done1);

        let (text2, done2, _) = parser.parse_data(data2);
        assert_eq!(text2, Some(" World".to_string()));
        assert!(!done2);

        let (text3, done3, _) = parser.parse_data(data3);
        assert!(text3.is_none());
        assert!(done3);

        assert_eq!(parser.get_full_content(), "Hello World");
    }

    #[test]
    fn test_sse_parser_tool_calls() {
        let mut parser = SSEParser::new();

        // 模拟工具调用的 SSE 数据
        let data1 = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"bash"}}]}}]}"#;
        let data2 = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":"}}]}}]}"#;
        let data3 = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls -la\"}"}}]}}]}"#;
        let data4 = r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#;

        parser.parse_data(data1);
        parser.parse_data(data2);
        parser.parse_data(data3);
        let (_, done, _) = parser.parse_data(data4);

        assert!(done);
        assert!(parser.has_tool_calls());

        let tool_calls = parser.finalize_tool_calls();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_123");
        assert_eq!(tool_calls[0].function.name, "bash");
        assert_eq!(tool_calls[0].function.arguments, r#"{"command":"ls -la"}"#);
    }

    #[test]
    fn test_sse_parser_usage() {
        let mut parser = SSEParser::new();

        let data = r#"{"choices":[{"delta":{"content":"Hi"}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let (text, _, usage) = parser.parse_data(data);

        assert_eq!(text, Some("Hi".to_string()));
        assert!(usage.is_some());
        let usage = usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn test_sse_parser_done_signal() {
        let mut parser = SSEParser::new();

        let (_, done, _) = parser.parse_data("[DONE]");
        assert!(done);
    }

    #[test]
    fn test_sse_parser_invalid_json() {
        let mut parser = SSEParser::new();

        let (text, done, usage) = parser.parse_data("invalid json");
        assert!(text.is_none());
        assert!(!done);
        assert!(usage.is_none());
    }

    #[test]
    fn test_sse_parser_multiple_tool_calls() {
        let mut parser = SSEParser::new();

        // 两个工具调用
        let data1 = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":"{}"}}]}}]}"#;
        let data2 = r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"read_file","arguments":"{}"}}]}}]}"#;

        parser.parse_data(data1);
        parser.parse_data(data2);

        let tool_calls = parser.finalize_tool_calls();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].function.name, "bash");
        assert_eq!(tool_calls[1].function.name, "read_file");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// 生成有效的文本内容（不包含特殊字符）
    fn arb_text_content() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9 ,.!?]{1,100}".prop_map(|s| s)
    }

    /// 生成文本片段列表
    fn arb_text_chunks() -> impl Strategy<Value = Vec<String>> {
        prop::collection::vec(arb_text_content(), 1..10)
    }

    /// 生成有效的工具名称
    fn arb_tool_name() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("bash".to_string()),
            Just("read_file".to_string()),
            Just("write_file".to_string()),
            Just("edit_file".to_string()),
        ]
    }

    /// 生成有效的工具调用 ID
    fn arb_tool_id() -> impl Strategy<Value = String> {
        "call_[a-zA-Z0-9]{8}".prop_map(|s| s)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: agent-tool-calling, Property 1: 流式事件完整性**
        /// **Validates: Requirements 1.1, 1.3**
        ///
        /// *For any* Agent 响应流，流式处理器发送的所有 text_delta 事件的文本拼接后，
        /// 应该等于最终的完整响应内容。
        #[test]
        fn prop_streaming_text_completeness(chunks in arb_text_chunks()) {
            let mut parser = SSEParser::new();
            let mut collected_deltas = String::new();

            // 模拟流式处理
            for chunk in &chunks {
                // 转义 JSON 特殊字符
                let escaped = chunk.replace('\\', "\\\\").replace('"', "\\\"");
                let data = format!(r#"{{"choices":[{{"delta":{{"content":"{}"}}}}]}}"#, escaped);
                let (text_delta, _, _) = parser.parse_data(&data);

                if let Some(text) = text_delta {
                    collected_deltas.push_str(&text);
                }
            }

            // 验证：收集的 text_delta 拼接后等于 parser 的完整内容
            prop_assert_eq!(
                collected_deltas,
                parser.get_full_content(),
                "收集的 text_delta 应该等于完整内容"
            );

            // 验证：完整内容等于原始 chunks 拼接
            let expected = chunks.join("");
            prop_assert_eq!(
                parser.get_full_content(),
                expected,
                "完整内容应该等于原始 chunks 拼接"
            );
        }

        /// **Feature: agent-tool-calling, Property 1: 流式事件完整性 - 工具调用**
        /// **Validates: Requirements 1.1, 1.3**
        ///
        /// *For any* 包含工具调用的响应流，工具调用信息应该被正确解析和累积。
        #[test]
        fn prop_streaming_tool_calls_completeness(
            tool_name in arb_tool_name(),
            tool_id in arb_tool_id(),
            arg_key in "[a-z]{3,10}",
            arg_value in "[a-zA-Z0-9]{1,20}"
        ) {
            let mut parser = SSEParser::new();

            // 使用 serde_json 构建正确的 JSON，避免手动转义问题
            let args_json = serde_json::json!({arg_key.clone(): arg_value.clone()}).to_string();

            // 第一个 chunk: 工具 ID 和名称
            let data1 = serde_json::json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": tool_id.clone(),
                            "type": "function",
                            "function": {
                                "name": tool_name.clone()
                            }
                        }]
                    }
                }]
            }).to_string();

            // 第二个 chunk: 参数的前半部分
            let args_first_half = &args_json[..args_json.len()/2];
            let data2 = serde_json::json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {
                                "arguments": args_first_half
                            }
                        }]
                    }
                }]
            }).to_string();

            // 第三个 chunk: 参数的后半部分
            let args_second_half = &args_json[args_json.len()/2..];
            let data3 = serde_json::json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {
                                "arguments": args_second_half
                            }
                        }]
                    }
                }]
            }).to_string();

            parser.parse_data(&data1);
            parser.parse_data(&data2);
            parser.parse_data(&data3);

            prop_assert!(parser.has_tool_calls(), "应该检测到工具调用");

            let tool_calls = parser.finalize_tool_calls();
            prop_assert_eq!(tool_calls.len(), 1, "应该有一个工具调用");
            prop_assert_eq!(&tool_calls[0].id, &tool_id, "工具调用 ID 应该匹配");
            prop_assert_eq!(&tool_calls[0].function.name, &tool_name, "工具名称应该匹配");

            // 验证参数被正确累积
            prop_assert_eq!(
                &tool_calls[0].function.arguments,
                &args_json,
                "工具参数应该被正确累积"
            );
        }

        /// **Feature: agent-tool-calling, Property 1: 流式事件完整性 - Done 事件**
        /// **Validates: Requirements 1.1, 1.3**
        ///
        /// *For any* 完成的响应流，应该正确识别 finish_reason。
        #[test]
        fn prop_streaming_done_detection(
            finish_reason in prop_oneof![Just("stop"), Just("tool_calls"), Just("length")]
        ) {
            let mut parser = SSEParser::new();

            let data = format!(
                r#"{{"choices":[{{"delta":{{}},"finish_reason":"{}"}}]}}"#,
                finish_reason
            );
            let (_, is_done, _) = parser.parse_data(&data);

            if finish_reason == "stop" || finish_reason == "tool_calls" {
                prop_assert!(is_done, "finish_reason={} 应该标记为完成", finish_reason);
            } else {
                prop_assert!(!is_done, "finish_reason={} 不应该标记为完成", finish_reason);
            }
        }

        /// **Feature: agent-tool-calling, Property 1: 流式事件完整性 - Usage 统计**
        /// **Validates: Requirements 1.3**
        ///
        /// *For any* 包含 usage 的响应，应该正确解析 token 使用量。
        #[test]
        fn prop_streaming_usage_parsing(
            input_tokens in 0u32..10000,
            output_tokens in 0u32..10000
        ) {
            let mut parser = SSEParser::new();

            let data = format!(
                r#"{{"choices":[{{"delta":{{"content":"test"}}}}],"usage":{{"prompt_tokens":{},"completion_tokens":{}}}}}"#,
                input_tokens, output_tokens
            );
            let (_, _, usage) = parser.parse_data(&data);

            if input_tokens > 0 || output_tokens > 0 {
                prop_assert!(usage.is_some(), "应该解析出 usage");
                let usage = usage.unwrap();
                prop_assert_eq!(usage.input_tokens, input_tokens, "input_tokens 应该匹配");
                prop_assert_eq!(usage.output_tokens, output_tokens, "output_tokens 应该匹配");
            }
        }
    }
}
