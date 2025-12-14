//! OpenAI 格式转换为 CodeWhisperer 格式
use crate::models::codewhisperer::*;
use crate::models::openai::*;
use std::collections::HashMap;
use uuid::Uuid;

/// 模型映射表
pub fn get_model_map() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();
    map.insert("claude-opus-4-5", "claude-opus-4.5");
    map.insert("claude-opus-4-5-20251101", "claude-opus-4.5");
    map.insert("claude-haiku-4-5", "claude-haiku-4.5");
    map.insert("claude-haiku-4-5-20251001", "claude-haiku-4.5");
    map.insert("claude-sonnet-4-5", "CLAUDE_SONNET_4_5_20250929_V1_0");
    map.insert(
        "claude-sonnet-4-5-20250929",
        "CLAUDE_SONNET_4_5_20250929_V1_0",
    );
    map.insert("claude-sonnet-4-20250514", "CLAUDE_SONNET_4_20250514_V1_0");
    map.insert(
        "claude-3-7-sonnet-20250219",
        "CLAUDE_3_7_SONNET_20250219_V1_0",
    );
    map.insert(
        "claude-3-5-sonnet-20241022",
        "CLAUDE_3_7_SONNET_20250219_V1_0",
    );
    map.insert(
        "claude-3-5-sonnet-latest",
        "CLAUDE_3_7_SONNET_20250219_V1_0",
    );
    map
}

pub const DEFAULT_MODEL: &str = "CLAUDE_SONNET_4_5_20250929_V1_0";

/// 预处理消息：合并连续的 tool 消息到前一个 assistant 消息后的 user 消息
fn preprocess_messages(messages: &[&ChatMessage]) -> Vec<ProcessedMessage> {
    let mut result: Vec<ProcessedMessage> = Vec::new();
    let mut pending_tool_results: Vec<CWToolResult> = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "tool" => {
                // 收集 tool 结果
                let content = msg.get_content_text();
                let tool_id = msg.tool_call_id.clone().unwrap_or_default();
                pending_tool_results.push(CWToolResult {
                    content: vec![CWTextContent { text: content }],
                    status: "success".to_string(),
                    tool_use_id: tool_id,
                });
            }
            "user" => {
                // 如果有待处理的 tool results，合并到这个 user 消息
                let content = msg.get_content_text();
                let mut tool_results = pending_tool_results.clone();
                pending_tool_results.clear();

                // 去重 tool_results
                let mut seen_ids = std::collections::HashSet::new();
                tool_results.retain(|tr| seen_ids.insert(tr.tool_use_id.clone()));

                result.push(ProcessedMessage {
                    role: "user".to_string(),
                    content,
                    tool_calls: None,
                    tool_results: if tool_results.is_empty() {
                        None
                    } else {
                        Some(tool_results)
                    },
                });
            }
            "assistant" => {
                // 如果有待处理的 tool results，先创建一个 user 消息
                if !pending_tool_results.is_empty() {
                    let mut tool_results = pending_tool_results.clone();
                    pending_tool_results.clear();

                    // 去重 tool_results
                    let mut seen_ids = std::collections::HashSet::new();
                    tool_results.retain(|tr| seen_ids.insert(tr.tool_use_id.clone()));

                    result.push(ProcessedMessage {
                        role: "user".to_string(),
                        content: "Tool results provided.".to_string(),
                        tool_calls: None,
                        tool_results: Some(tool_results),
                    });
                }

                let content = msg.get_content_text();
                let tool_calls = msg.tool_calls.as_ref().map(|calls| {
                    calls
                        .iter()
                        .map(|tc| CWToolUse {
                            input: serde_json::from_str(&tc.function.arguments)
                                .unwrap_or(serde_json::json!({})),
                            name: tc.function.name.clone(),
                            tool_use_id: tc.id.clone(),
                        })
                        .collect()
                });

                result.push(ProcessedMessage {
                    role: "assistant".to_string(),
                    content,
                    tool_calls,
                    tool_results: None,
                });
            }
            _ => {}
        }
    }

    // 处理末尾的 tool results
    if !pending_tool_results.is_empty() {
        let mut tool_results = pending_tool_results;
        let mut seen_ids = std::collections::HashSet::new();
        tool_results.retain(|tr| seen_ids.insert(tr.tool_use_id.clone()));

        result.push(ProcessedMessage {
            role: "user".to_string(),
            content: "Tool results provided.".to_string(),
            tool_calls: None,
            tool_results: Some(tool_results),
        });
    }

    result
}

#[derive(Debug, Clone)]
struct ProcessedMessage {
    role: String,
    content: String,
    tool_calls: Option<Vec<CWToolUse>>,
    tool_results: Option<Vec<CWToolResult>>,
}

/// 将 OpenAI ChatCompletionRequest 转换为 CodeWhisperer 请求
pub fn convert_openai_to_codewhisperer(
    request: &ChatCompletionRequest,
    profile_arn: Option<String>,
) -> CodeWhispererRequest {
    let model_map = get_model_map();
    let cw_model = model_map
        .get(request.model.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let conversation_id = Uuid::new_v4().to_string();

    // 提取 system prompt 和消息
    let mut system_prompt = String::new();
    let mut raw_messages: Vec<&ChatMessage> = Vec::new();

    for msg in &request.messages {
        if msg.role == "system" {
            system_prompt = msg.get_content_text();
        } else {
            raw_messages.push(msg);
        }
    }

    // 预处理消息：合并 tool 消息
    let messages = preprocess_messages(&raw_messages);

    // 构建历史记录
    let mut history: Vec<HistoryItem> = Vec::new();
    let mut start_idx = 0;

    // 处理 system prompt - 合并到第一条用户消息
    if !system_prompt.is_empty() && !messages.is_empty() && messages[0].role == "user" {
        let first_content = &messages[0].content;
        let combined = format!("{system_prompt}\n\n{first_content}");

        let mut user_msg = UserInputMessage {
            content: combined,
            model_id: cw_model.clone(),
            origin: "AI_EDITOR".to_string(),
            images: None,
            user_input_message_context: None,
        };

        // 如果第一条消息有 tool_results，也要包含
        if let Some(ref tool_results) = messages[0].tool_results {
            user_msg.user_input_message_context = Some(UserInputMessageContext {
                tools: None,
                tool_results: Some(tool_results.clone()),
            });
        }

        history.push(HistoryItem::User(UserHistoryItem {
            user_input_message: user_msg,
        }));
        start_idx = 1;
    }

    // 处理历史消息（除最后一条）
    for msg in messages
        .iter()
        .take(messages.len().saturating_sub(1))
        .skip(start_idx)
    {
        match msg.role.as_str() {
            "user" => {
                let content = if msg.content.is_empty() {
                    if msg.tool_results.is_some() {
                        "Tool results provided.".to_string()
                    } else {
                        "Continue".to_string()
                    }
                } else {
                    msg.content.clone()
                };

                let mut user_msg = UserInputMessage {
                    content,
                    model_id: cw_model.clone(),
                    origin: "AI_EDITOR".to_string(),
                    images: None,
                    user_input_message_context: None,
                };

                if let Some(ref tool_results) = msg.tool_results {
                    user_msg.user_input_message_context = Some(UserInputMessageContext {
                        tools: None,
                        tool_results: Some(tool_results.clone()),
                    });
                }

                history.push(HistoryItem::User(UserHistoryItem {
                    user_input_message: user_msg,
                }));
            }
            "assistant" => {
                let content = if msg.content.is_empty() {
                    "I understand.".to_string()
                } else {
                    msg.content.clone()
                };

                history.push(HistoryItem::Assistant(AssistantHistoryItem {
                    assistant_response_message: AssistantResponseMessage {
                        content,
                        tool_uses: msg.tool_calls.clone(),
                    },
                }));
            }
            _ => {}
        }
    }

    // 修复历史记录交替顺序
    let history = fix_history_alternation(history, &cw_model);

    // 构建当前消息
    let (current_content, current_tool_results) = if messages.is_empty() {
        ("Continue".to_string(), None)
    } else {
        let last_msg = messages.last().unwrap();
        if last_msg.role == "assistant" {
            ("Continue".to_string(), None)
        } else {
            let content = if last_msg.content.is_empty() {
                if last_msg.tool_results.is_some() {
                    "Tool results provided.".to_string()
                } else {
                    "Continue".to_string()
                }
            } else {
                last_msg.content.clone()
            };
            (content, last_msg.tool_results.clone())
        }
    };

    // 构建 tools
    let tools = request.tools.as_ref().map(|tools| {
        tools
            .iter()
            .take(50)
            .map(|t| {
                let params = t
                    .function
                    .parameters
                    .clone()
                    .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));

                let desc = t
                    .function
                    .description
                    .clone()
                    .unwrap_or_else(|| format!("Tool: {}", t.function.name));

                CWTool {
                    tool_specification: ToolSpecification {
                        name: t.function.name.clone(),
                        description: if desc.len() > 500 {
                            format!("{}...", &desc[..497])
                        } else {
                            desc
                        },
                        input_schema: InputSchema { json: params },
                    },
                }
            })
            .collect()
    });

    let user_input_message_context = if tools.is_some() || current_tool_results.is_some() {
        Some(UserInputMessageContext {
            tools,
            tool_results: current_tool_results,
        })
    } else {
        None
    };

    CodeWhispererRequest {
        conversation_state: ConversationState {
            chat_trigger_type: "MANUAL".to_string(),
            conversation_id,
            current_message: CurrentMessage {
                user_input_message: UserInputMessage {
                    content: current_content,
                    model_id: cw_model,
                    origin: "AI_EDITOR".to_string(),
                    images: None,
                    user_input_message_context,
                },
            },
            history: if history.is_empty() {
                None
            } else {
                Some(history)
            },
        },
        profile_arn,
    }
}

/// 修复历史记录，确保 user/assistant 严格交替
fn fix_history_alternation(history: Vec<HistoryItem>, model_id: &str) -> Vec<HistoryItem> {
    if history.is_empty() {
        return history;
    }

    let mut fixed: Vec<HistoryItem> = Vec::new();

    for item in history {
        match &item {
            HistoryItem::User(user_item) => {
                // 如果上一条也是 user，合并 tool_results 或插入占位 assistant
                if let Some(HistoryItem::User(last_user)) = fixed.last_mut() {
                    // 尝试合并 tool_results
                    let has_tool_results = user_item
                        .user_input_message
                        .user_input_message_context
                        .as_ref()
                        .map(|ctx| ctx.tool_results.is_some())
                        .unwrap_or(false);

                    if has_tool_results {
                        // 合并 tool_results 到上一个 user 消息
                        let new_results = user_item
                            .user_input_message
                            .user_input_message_context
                            .as_ref()
                            .and_then(|ctx| ctx.tool_results.clone())
                            .unwrap_or_default();

                        if let Some(ref mut ctx) = last_user.user_input_message.user_input_message_context {
                            if let Some(ref mut existing) = ctx.tool_results {
                                existing.extend(new_results);
                            } else {
                                ctx.tool_results = Some(new_results);
                            }
                        } else {
                            last_user.user_input_message.user_input_message_context =
                                Some(UserInputMessageContext {
                                    tools: None,
                                    tool_results: Some(new_results),
                                });
                        }
                        continue;
                    } else {
                        // 插入占位 assistant
                        fixed.push(HistoryItem::Assistant(AssistantHistoryItem {
                            assistant_response_message: AssistantResponseMessage {
                                content: "I understand.".to_string(),
                                tool_uses: None,
                            },
                        }));
                    }
                }
                fixed.push(item);
            }
            HistoryItem::Assistant(_) => {
                // 如果上一条也是 assistant，插入占位 user
                if let Some(HistoryItem::Assistant(_)) = fixed.last() {
                    fixed.push(HistoryItem::User(UserHistoryItem {
                        user_input_message: UserInputMessage {
                            content: "Continue".to_string(),
                            model_id: model_id.to_string(),
                            origin: "AI_EDITOR".to_string(),
                            images: None,
                            user_input_message_context: None,
                        },
                    }));
                }
                // 如果历史为空，先插入 user
                if fixed.is_empty() {
                    fixed.push(HistoryItem::User(UserHistoryItem {
                        user_input_message: UserInputMessage {
                            content: "Continue".to_string(),
                            model_id: model_id.to_string(),
                            origin: "AI_EDITOR".to_string(),
                            images: None,
                            user_input_message_context: None,
                        },
                    }));
                }
                fixed.push(item);
            }
        }
    }

    // 确保以 assistant 结尾
    if let Some(HistoryItem::User(_)) = fixed.last() {
        fixed.push(HistoryItem::Assistant(AssistantHistoryItem {
            assistant_response_message: AssistantResponseMessage {
                content: "I understand.".to_string(),
                tool_uses: None,
            },
        }));
    }

    fixed
}
