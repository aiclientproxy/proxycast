use super::McpClientManager;
use crate::types::*;
use tracing::{debug, error, info, warn};

impl McpClientManager {
    // ========================================================================
    // 提示词管理方法
    // ========================================================================

    /// 获取所有提示词
    ///
    /// 从所有运行中的服务器获取提示词定义。
    ///
    /// # Returns
    ///
    /// 返回所有可用提示词的定义列表。
    ///
    /// # 实现步骤（Task 4.4）
    ///
    /// 1. 遍历所有运行中的服务器
    /// 2. 检查服务器是否支持提示词
    /// 3. 调用 list_all_prompts 获取提示词列表
    /// 4. 转换为 McpPromptDefinition 格式
    /// 5. 返回合并后的提示词列表
    pub async fn list_prompts(&self) -> Result<Vec<McpPromptDefinition>, McpError> {
        info!("获取所有 MCP 提示词");

        let mut all_prompts: Vec<McpPromptDefinition> = Vec::new();
        let clients = self.clients.read().await;

        for (server_name, wrapper) in clients.iter() {
            // 检查服务器是否支持提示词
            if let Some(ref info) = wrapper.server_info {
                if !info.supports_prompts {
                    debug!(server_name = %server_name, "服务器不支持提示词，跳过");
                    continue;
                }
            }

            // 获取 rmcp 服务
            let service = match wrapper.running_service() {
                Some(s) => s,
                None => {
                    warn!(server_name = %server_name, "服务器无运行服务，跳过");
                    continue;
                }
            };

            // 调用 list_all_prompts 获取所有提示词
            match service.list_all_prompts().await {
                Ok(prompts) => {
                    debug!(
                        server_name = %server_name,
                        prompt_count = prompts.len(),
                        "获取服务器提示词列表成功"
                    );
                    for prompt in prompts {
                        all_prompts.push(Self::convert_prompt_to_definition(
                            prompt,
                            server_name.clone(),
                        ));
                    }
                }
                Err(e) => {
                    warn!(
                        server_name = %server_name,
                        error = %e,
                        "获取服务器提示词列表失败"
                    );
                    // 继续处理其他服务器，不中断
                }
            }
        }

        info!(prompt_count = all_prompts.len(), "提示词列表已获取");
        Ok(all_prompts)
    }

    /// 将 rmcp Prompt 转换为 McpPromptDefinition
    pub(super) fn convert_prompt_to_definition(
        prompt: rmcp::model::Prompt,
        server_name: String,
    ) -> McpPromptDefinition {
        let arguments = prompt
            .arguments
            .unwrap_or_default()
            .into_iter()
            .map(|arg| McpPromptArgument {
                name: arg.name,
                description: arg.description,
                required: arg.required.unwrap_or(false),
            })
            .collect();

        McpPromptDefinition {
            name: prompt.name.to_string(),
            description: prompt.description.map(|s| s.to_string()),
            arguments,
            server_name,
        }
    }

    /// 获取提示词内容
    ///
    /// # Arguments
    ///
    /// * `server_name` - 精确的服务器名称
    /// * `name` - 提示词名称
    /// * `arguments` - 提示词参数
    ///
    /// # Returns
    ///
    /// 返回提示词内容，包含描述和消息列表。
    ///
    /// # 实现步骤（Task 4.4）
    ///
    /// 1. 按服务器名称精确选择运行连接
    /// 2. 验证必需参数是否提供
    /// 3. 调用服务器的 get_prompt 方法
    /// 4. 转换结果为 McpPromptResult
    /// 5. 返回结果
    pub async fn get_prompt(
        &self,
        server_name: &str,
        name: &str,
        arguments: serde_json::Map<String, serde_json::Value>,
    ) -> Result<McpPromptResult, McpError> {
        let (server_name, name) = validate_prompt_target(server_name, name)?;
        info!(server_name = %server_name, prompt_name = %name, "获取 MCP 提示词内容");
        let clients = self.clients.read().await;
        let wrapper = clients
            .get(server_name)
            .ok_or_else(|| McpError::ServerNotRunning(server_name.to_string()))?;

        let service = wrapper
            .running_service()
            .ok_or_else(|| McpError::ServerNotRunning(server_name.to_string()))?;

        let args: Option<serde_json::Map<String, serde_json::Value>> = if arguments.is_empty() {
            None
        } else {
            Some(arguments)
        };

        let get_prompt_param = rmcp::model::GetPromptRequestParam {
            name: name.to_string(),
            arguments: args,
        };

        let result = service.get_prompt(get_prompt_param).await.map_err(|e| {
            error!(
                prompt_name = %name,
                server_name = %server_name,
                error = %e,
                "获取提示词失败"
            );
            McpError::ToolCallFailed(format!("获取提示词失败: {e}"))
        })?;

        let mcp_result = Self::convert_get_prompt_result(result);

        info!(
            prompt_name = %name,
            server_name = %server_name,
            message_count = mcp_result.messages.len(),
            "提示词获取完成"
        );

        Ok(mcp_result)
    }

    /// 转换 rmcp GetPromptResult 为 McpPromptResult
    pub(super) fn convert_get_prompt_result(
        result: rmcp::model::GetPromptResult,
    ) -> McpPromptResult {
        let messages: Vec<McpPromptMessage> = result
            .messages
            .into_iter()
            .map(Self::convert_prompt_message)
            .collect();

        McpPromptResult {
            description: result.description.map(|s| s.to_string()),
            messages,
        }
    }

    /// 转换 rmcp PromptMessage 为 McpPromptMessage
    pub(super) fn convert_prompt_message(msg: rmcp::model::PromptMessage) -> McpPromptMessage {
        let role = match msg.role {
            rmcp::model::PromptMessageRole::User => "user".to_string(),
            rmcp::model::PromptMessageRole::Assistant => "assistant".to_string(),
        };

        let content = Self::convert_prompt_message_content(msg.content);

        McpPromptMessage { role, content }
    }

    /// 转换 rmcp PromptMessageContent 为 McpContent
    pub(super) fn convert_prompt_message_content(
        content: rmcp::model::PromptMessageContent,
    ) -> McpContent {
        match content {
            rmcp::model::PromptMessageContent::Text { text } => McpContent::Text { text },
            rmcp::model::PromptMessageContent::Image { image } => McpContent::Image {
                data: image.data.clone(),
                mime_type: image.mime_type.clone(),
            },
            rmcp::model::PromptMessageContent::Resource { resource } => {
                let (uri, text, blob) = match &resource.resource {
                    rmcp::model::ResourceContents::TextResourceContents { uri, text, .. } => {
                        (uri.clone(), Some(text.clone()), None)
                    }
                    rmcp::model::ResourceContents::BlobResourceContents { uri, blob, .. } => {
                        (uri.clone(), None, Some(blob.clone()))
                    }
                };
                McpContent::Resource { uri, text, blob }
            }
            rmcp::model::PromptMessageContent::ResourceLink { link } => McpContent::Resource {
                uri: link.uri.clone(),
                text: Some(link.name.clone()),
                blob: None,
            },
        }
    }
}

fn validate_prompt_target<'a>(
    server_name: &'a str,
    name: &'a str,
) -> Result<(&'a str, &'a str), McpError> {
    let server_name =
        crate::naming::validate_server_name(server_name).map_err(McpError::ConfigError)?;
    let name = name.trim();
    if name.is_empty() {
        return Err(McpError::ConfigError(
            "MCP prompt name cannot be empty".to_string(),
        ));
    }
    Ok((server_name, name))
}
