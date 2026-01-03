//! 原生 Rust Agent 实现
//!
//! 支持连续对话（Conversation History）和工具调用（Tools）
//! 使用策略模式支持多种 API 协议（OpenAI、Anthropic、Kiro、Gemini）
//!
//! ## 架构设计
//! - protocols/ - 协议策略实现
//! - parsers/ - SSE 流解析器
//! - NativeAgent - 核心 Agent 逻辑
//! - NativeAgentState - Tauri 状态管理
//!
//! ## 流式处理
//! - Requirements: 1.1, 1.3, 1.4
//!
//! ## 工具调用循环
//! - Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6

#![allow(dead_code)]

use crate::agent::protocols::{create_protocol, Protocol};
use crate::agent::tool_loop::{ToolCallResult, ToolLoopEngine, ToolLoopState};
use crate::agent::tools::{create_default_registry, ToolRegistry};
use crate::agent::types::*;
use crate::models::openai::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ContentPart as OpenAIContentPart,
    MessageContent as OpenAIMessageContent,
};
use parking_lot::RwLock;
use reqwest::Client;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// 原生 Agent 实现
pub struct NativeAgent {
    client: Client,
    base_url: String,
    api_key: String,
    sessions: Arc<RwLock<HashMap<String, AgentSession>>>,
    config: AgentConfig,
    /// Provider 类型，决定使用哪种协议
    provider_type: ProviderType,
    /// 协议处理器
    protocol: Box<dyn Protocol>,
}

impl NativeAgent {
    pub fn new(
        base_url: String,
        api_key: String,
        provider_type: ProviderType,
    ) -> Result<Self, String> {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(30))
            .no_proxy()
            .build()
            .map_err(|e| format!("创建 HTTP 客户端失败: {}", e))?;

        let protocol = create_protocol(provider_type);

        info!(
            "[NativeAgent] 创建 Agent: base_url={}, provider={:?}, protocol_endpoint={}",
            base_url,
            provider_type,
            protocol.endpoint()
        );

        Ok(Self {
            client,
            base_url,
            api_key,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config: AgentConfig::default(),
            provider_type,
            protocol,
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

    /// 发送聊天请求（非流式，用于简单场景）
    pub async fn chat(&self, request: NativeChatRequest) -> Result<NativeChatResponse, String> {
        let model = request.model.unwrap_or_else(|| self.config.model.clone());
        let session_id = request.session_id.clone();
        let has_images = request.images.as_ref().map(|i| i.len()).unwrap_or(0);

        info!(
            "[NativeAgent] 发送聊天请求: model={}, session={:?}, images={}",
            model, session_id, has_images
        );

        // 获取会话
        let session = if let Some(sid) = &session_id {
            self.sessions.read().get(sid).cloned()
        } else {
            None
        };

        // 构建消息
        let messages = self.build_openai_messages(
            session.as_ref(),
            &request.message,
            request.images.as_deref(),
        );

        let chat_request = ChatCompletionRequest {
            model: model.clone(),
            messages,
            stream: false,
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

    /// 流式聊天（使用协议策略模式）
    ///
    /// Requirements: 1.1, 1.3, 1.4
    pub async fn chat_stream(
        &self,
        request: NativeChatRequest,
        tools: Option<&[crate::models::openai::Tool]>,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<StreamResult, String> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.config.model.clone());
        let session_id = request.session_id.clone();

        info!(
            "[NativeAgent] 发送流式聊天请求: model={}, session={:?}, provider={:?}, tools_count={}",
            model,
            session_id,
            self.provider_type,
            tools.map(|t| t.len()).unwrap_or(0)
        );

        // 获取会话
        let session = if let Some(sid) = &session_id {
            self.sessions.read().get(sid).cloned()
        } else {
            None
        };

        // 获取会话历史和配置
        let history: Vec<AgentMessage> = session
            .as_ref()
            .map(|s| s.messages.clone())
            .unwrap_or_default();

        let config = if let Some(ref sess) = session {
            let mut cfg = self.config.clone();
            if sess.system_prompt.is_some() {
                cfg.system_prompt = sess.system_prompt.clone();
            }
            cfg
        } else {
            self.config.clone()
        };

        // 使用协议策略发送请求
        let result = self
            .protocol
            .chat_stream(
                &self.client,
                &self.base_url,
                &self.api_key,
                &history,
                &request.message,
                request.images.as_deref(),
                &model,
                &config,
                tools,
                tx,
            )
            .await?;

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
                MessageContent::Text(result.content.clone()),
                result.tool_calls.clone(),
            );
        }

        Ok(result)
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
        let session_id = request.session_id.clone();
        let mut state = ToolLoopState::new();

        // 获取工具定义
        let tools = tool_loop_engine.registry().list_definitions_api();
        let tools_ref = if tools.is_empty() {
            None
        } else {
            Some(tools.as_slice())
        };

        // 首次请求
        let mut current_result = self
            .chat_stream(request.clone(), tools_ref, tx.clone())
            .await?;

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
            let tool_results = tool_loop_engine
                .execute_all_tool_calls(tool_calls, Some(&tx))
                .await;

            // 将工具结果添加到会话
            if let Some(sid) = &session_id {
                for result in &tool_results {
                    self.add_tool_result_to_session(sid, result);
                }
            }

            // 继续对话
            let continue_request = NativeChatRequest {
                session_id: session_id.clone(),
                message: String::new(),
                model: request.model.clone(),
                images: None,
                stream: true,
            };

            current_result = self
                .chat_stream_continue(continue_request, tools_ref, tx.clone())
                .await?;
        }

        // 检查是否因为达到最大迭代次数而停止
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

        // 发送 FinalDone 事件，通知前端整个对话（包括工具循环）已完成
        let _ = tx
            .send(StreamEvent::FinalDone {
                usage: current_result.usage.clone(),
            })
            .await;

        Ok(current_result)
    }

    /// 继续流式对话（使用会话历史）
    async fn chat_stream_continue(
        &self,
        request: NativeChatRequest,
        tools: Option<&[crate::models::openai::Tool]>,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<StreamResult, String> {
        let model = request.model.unwrap_or_else(|| self.config.model.clone());
        let session_id = request.session_id.as_ref().ok_or("需要 session_id")?;

        debug!(
            "[NativeAgent] 继续流式对话: model={}, session={}, tools_count={}",
            model,
            session_id,
            tools.map(|t| t.len()).unwrap_or(0)
        );

        // 获取会话
        let session = self
            .sessions
            .read()
            .get(session_id)
            .cloned()
            .ok_or_else(|| format!("会话不存在: {}", session_id))?;

        // 获取配置
        let config = {
            let mut cfg = self.config.clone();
            if session.system_prompt.is_some() {
                cfg.system_prompt = session.system_prompt.clone();
            }
            cfg
        };

        // 使用协议策略继续对话
        let result = self
            .protocol
            .chat_stream_continue(
                &self.client,
                &self.base_url,
                &self.api_key,
                &session.messages,
                &model,
                &config,
                tools,
                tx,
            )
            .await?;

        // 更新会话历史
        self.add_assistant_message_to_session(
            session_id,
            MessageContent::Text(result.content.clone()),
            result.tool_calls.clone(),
        );

        Ok(result)
    }

    // ==================== 会话管理方法 ====================

    /// 构建 OpenAI 格式消息（用于非流式请求）
    fn build_openai_messages(
        &self,
        session: Option<&AgentSession>,
        user_message: &str,
        images: Option<&[ImageData]>,
    ) -> Vec<ChatMessage> {
        let mut messages = Vec::new();

        // 系统提示词
        let system_prompt = session
            .and_then(|s| s.system_prompt.as_ref())
            .or(self.config.system_prompt.as_ref());
        if let Some(prompt) = system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(OpenAIMessageContent::Text(prompt.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // 历史消息
        if let Some(sess) = session {
            for msg in &sess.messages {
                messages.push(self.convert_to_chat_message(msg));
            }
        }

        // 用户消息
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
    fn add_tool_result_to_session(&self, session_id: &str, tool_result: &ToolCallResult) {
        let mut sessions = self.sessions.write();
        if let Some(session) = sessions.get_mut(session_id) {
            session.messages.push(tool_result.to_agent_message());
            session.updated_at = chrono::Utc::now().to_rfc3339();
        }
    }

    // ==================== 公开会话管理 API ====================

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

// ==================== Tauri 状态管理 ====================

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

    pub fn init(
        &self,
        base_url: String,
        api_key: String,
        provider_type: ProviderType,
    ) -> Result<(), String> {
        let agent = NativeAgent::new(base_url, api_key, provider_type)?;
        *self.agent.write() = Some(agent);
        Ok(())
    }

    pub fn is_initialized(&self) -> bool {
        self.agent.read().is_some()
    }

    /// 获取当前 Agent 的 provider 类型
    pub fn get_provider_type(&self) -> Option<ProviderType> {
        self.agent.read().as_ref().map(|a| a.provider_type)
    }

    pub fn reset(&self) {
        *self.agent.write() = None;
    }

    /// 获取工具注册表
    pub fn get_tool_registry(&self) -> Result<Arc<ToolRegistry>, String> {
        let base_dir = dirs::home_dir().ok_or_else(|| "无法获取用户 home 目录".to_string())?;
        let registry = create_default_registry(base_dir);
        Ok(Arc::new(registry))
    }

    /// 创建临时 Agent 用于异步操作
    fn create_temp_agent(&self) -> Result<NativeAgent, String> {
        let guard = self.agent.read();
        let agent = guard.as_ref().ok_or_else(|| "Agent 未初始化".to_string())?;

        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(30))
            .no_proxy()
            .build()
            .map_err(|e| format!("创建 HTTP 客户端失败: {}", e))?;

        let protocol = create_protocol(agent.provider_type);

        Ok(NativeAgent {
            client,
            base_url: agent.base_url.clone(),
            api_key: agent.api_key.clone(),
            sessions: agent.sessions.clone(),
            config: agent.config.clone(),
            provider_type: agent.provider_type,
            protocol,
        })
    }

    pub async fn chat(&self, request: NativeChatRequest) -> Result<NativeChatResponse, String> {
        let temp_agent = self.create_temp_agent()?;
        temp_agent.chat(request).await
    }

    pub async fn chat_stream(
        &self,
        request: NativeChatRequest,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<StreamResult, String> {
        let temp_agent = self.create_temp_agent()?;
        temp_agent.chat_stream(request, None, tx).await
    }

    pub async fn chat_stream_with_tools(
        &self,
        request: NativeChatRequest,
        tx: mpsc::Sender<StreamEvent>,
        tool_loop_engine: &ToolLoopEngine,
    ) -> Result<StreamResult, String> {
        let temp_agent = self.create_temp_agent()?;
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
    use crate::agent::parsers::OpenAISSEParser;

    #[test]
    fn test_sse_parser_text_delta() {
        let mut parser = OpenAISSEParser::new();

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
        let mut parser = OpenAISSEParser::new();

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
}
