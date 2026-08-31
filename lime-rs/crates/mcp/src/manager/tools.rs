use super::McpClientManager;
use crate::naming::parse_runtime_tool_name;
use crate::tool_policy;
use crate::types::*;
use lime_core::tool_calling::ToolSurfaceMetadata;
use std::collections::HashMap;
use tracing::{debug, error, info, warn};

pub(super) struct McpAppToolAuthority {
    pub(super) connector_id: Option<String>,
    pub(super) link_id: Option<String>,
    pub(super) requires_explicit_link_id: bool,
}

impl McpClientManager {
    // ========================================================================
    // 工具管理方法
    // ========================================================================

    /// 获取所有工具定义
    ///
    /// 从所有运行中的服务器获取工具定义，并使用缓存优化性能。
    ///
    /// # Returns
    ///
    /// 返回所有可用工具的定义列表。
    ///
    /// # 实现步骤（Task 4.3）
    ///
    /// 1. 检查缓存是否有效
    /// 2. 如果缓存有效，直接返回缓存
    /// 3. 从所有运行中的服务器获取工具
    /// 4. 解决名称冲突（添加服务器前缀）
    /// 5. 更新缓存
    /// 6. 发送 mcp:tools_updated 事件
    /// 7. 返回工具列表
    pub async fn list_tools(&self) -> Result<Vec<McpToolDefinition>, McpError> {
        // 1. 检查缓存是否有效
        if let Some(cached_tools) = self.get_cached_tools().await {
            debug!(tool_count = cached_tools.len(), "返回缓存的工具列表");
            return Ok(cached_tools);
        }

        // 2. 从所有运行中的服务器获取工具
        let mut all_tools: Vec<McpToolDefinition> = Vec::new();
        let clients = self.clients.read().await;
        let config_by_server = clients
            .iter()
            .map(|(server_name, wrapper)| (server_name.clone(), wrapper.config.clone()))
            .collect::<HashMap<_, _>>();

        for (server_name, wrapper) in clients.iter() {
            // 检查服务器是否支持工具
            if let Some(ref info) = wrapper.server_info {
                if !info.supports_tools {
                    debug!(server_name = %server_name, "服务器不支持工具，跳过");
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

            // 调用 list_tools（使用 list_all_tools 获取所有工具）
            match service.list_all_tools().await {
                Ok(tools) => {
                    debug!(
                        server_name = %server_name,
                        tool_count = tools.len(),
                        "获取服务器工具列表成功"
                    );
                    for tool in tools {
                        let input_schema = normalize_tool_input_schema(serde_json::Value::Object(
                            (*tool.input_schema).clone(),
                        ));
                        let output_schema = tool.output_schema.map(|schema| {
                            tool_result_output_schema(serde_json::Value::Object((*schema).clone()))
                        });
                        let metadata =
                            Self::extract_tool_metadata(tool.name.as_ref(), &input_schema);
                        all_tools.push(McpToolDefinition {
                            name: tool.name.to_string(),
                            description: tool
                                .description
                                .clone()
                                .map(|s| s.to_string())
                                .unwrap_or_default(),
                            input_schema,
                            output_schema,
                            server_name: server_name.clone(),
                            deferred_loading: metadata.deferred_loading,
                            always_visible: metadata.always_visible,
                            allowed_callers: metadata.allowed_callers,
                            input_examples: (!metadata.input_examples.is_empty())
                                .then_some(metadata.input_examples),
                            tags: metadata.tags,
                        });
                    }
                }
                Err(e) => {
                    warn!(
                        server_name = %server_name,
                        error = %e,
                        "获取服务器工具列表失败"
                    );
                    // 继续处理其他服务器，不中断
                }
            }
        }
        drop(clients);

        // 3. 收口到 MCP current runtime 名称
        let resolved_tools =
            Self::apply_default_loading_policy(tool_policy::apply_server_tool_filters(
                Self::apply_runtime_tool_names(all_tools),
                &config_by_server,
            ));

        // 4. 更新缓存
        self.update_tool_cache(resolved_tools.clone()).await;

        // 5. 发送 mcp:tools_updated 事件
        self.emit_tools_updated(resolved_tools.clone());

        info!(tool_count = resolved_tools.len(), "工具列表已更新");
        Ok(resolved_tools)
    }

    pub(super) async fn app_tool_authority(
        &self,
        server_name: &str,
        tool_name: &str,
    ) -> Result<McpAppToolAuthority, McpError> {
        let clients = self.clients.read().await;
        let wrapper = clients
            .get(server_name)
            .ok_or_else(|| McpError::ServerNotRunning(server_name.to_string()))?;
        if !wrapper.config.tool_is_enabled(tool_name) {
            return Err(McpError::ToolNotFound(format!("{server_name}/{tool_name}")));
        }
        let service = wrapper
            .running_service()
            .ok_or_else(|| McpError::ServerNotRunning(server_name.to_string()))?;
        let tools = service.list_all_tools().await.map_err(|error| {
            McpError::ToolCallFailed(format!("读取 MCP 工具 authority 失败: {error}"))
        })?;
        let tool = tools
            .iter()
            .find(|tool| tool.name.as_ref() == tool_name)
            .ok_or_else(|| McpError::ToolNotFound(format!("{server_name}/{tool_name}")))?;
        Ok(app_tool_authority_from_meta(tool.meta.as_ref()))
    }

    /// 根据上下文过滤工具列表
    ///
    /// - `caller`: 调用方（assistant/code_execution/tool_search）
    /// - `include_deferred`: 是否包含延迟加载工具
    pub async fn list_tools_for_context(
        &self,
        caller: Option<&str>,
        include_deferred: bool,
    ) -> Result<Vec<McpToolDefinition>, McpError> {
        let tools = self.list_tools().await?;

        let filtered = tools
            .into_iter()
            .filter(|tool| tool_policy::tool_visible_for_context(tool, caller, include_deferred))
            .collect();

        Ok(filtered)
    }

    pub async fn list_tools_for_context_fresh(
        &self,
        caller: Option<&str>,
        include_deferred: bool,
    ) -> Result<Vec<McpToolDefinition>, McpError> {
        let tools = self.list_tools_fresh().await?;
        Ok(tools
            .into_iter()
            .filter(|tool| tool_policy::tool_visible_for_context(tool, caller, include_deferred))
            .collect())
    }

    /// 搜索工具
    ///
    /// 搜索默认包含 deferred 工具，便于模型通过 tool_search 检索按需加载。
    pub async fn search_tools(
        &self,
        query: &str,
        limit: usize,
        caller: Option<&str>,
    ) -> Result<Vec<McpToolDefinition>, McpError> {
        let query = query.trim().to_ascii_lowercase();
        let limit = limit.clamp(1, 100);
        let mut tools = self.list_tools_for_context(caller, true).await?;

        // 空查询：优先 always_visible，再按名称排序返回前 N
        if query.is_empty() {
            tools.sort_by(|a, b| {
                let a_visible = a.always_visible.unwrap_or(false);
                let b_visible = b.always_visible.unwrap_or(false);
                b_visible
                    .cmp(&a_visible)
                    .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
            });
            tools.truncate(limit);
            return Ok(tools);
        }

        let mut scored: Vec<(i32, McpToolDefinition)> = tools
            .into_iter()
            .filter_map(|tool| {
                let score = Self::score_tool_match(&tool, &query);
                (score > 0).then_some((score, tool))
            })
            .collect();

        scored.sort_by(|(score_a, tool_a), (score_b, tool_b)| {
            score_b
                .cmp(score_a)
                .then_with(|| tool_a.name.to_lowercase().cmp(&tool_b.name.to_lowercase()))
        });

        let mut result = scored
            .into_iter()
            .take(limit)
            .map(|(_, tool)| tool)
            .collect::<Vec<_>>();
        result.truncate(limit);
        Ok(result)
    }

    pub(super) fn extract_tool_metadata(
        tool_name: &str,
        input_schema: &serde_json::Value,
    ) -> ToolSurfaceMetadata {
        tool_policy::extract_tool_metadata(tool_name, input_schema)
    }

    fn score_tool_match(tool: &McpToolDefinition, query: &str) -> i32 {
        tool_policy::score_tool_match(tool, query)
    }

    pub(super) fn apply_default_loading_policy(
        tools: Vec<McpToolDefinition>,
    ) -> Vec<McpToolDefinition> {
        tool_policy::apply_default_loading_policy(tools)
    }

    /// 将 MCP 工具名统一收口到当前 runtime 命名。
    ///
    /// # Arguments
    ///
    /// * `tools` - 原始工具列表
    ///
    /// # Returns
    ///
    /// 返回统一后的工具列表，名称始终为 `mcp__<server>__<tool>`。
    pub(super) fn apply_runtime_tool_names(
        tools: Vec<McpToolDefinition>,
    ) -> Vec<McpToolDefinition> {
        tool_policy::apply_runtime_tool_names(tools)
    }

    /// 调用工具
    ///
    /// # Arguments
    ///
    /// * `tool_name` - 工具名称，格式为 `mcp__<server>__<tool>`
    /// * `arguments` - 工具参数
    ///
    /// # Returns
    ///
    /// 返回工具调用结果。
    ///
    /// # 实现步骤（Task 4.3）
    ///
    /// 1. 解析工具名称，确定目标服务器
    /// 2. 路由到正确的客户端
    /// 3. 执行工具调用
    /// 4. 转换结果为 McpToolResult
    /// 5. 返回结果
    pub async fn call_tool_with_caller(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
        caller: Option<&str>,
    ) -> Result<McpToolResult, McpError> {
        let caller = caller
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_ascii_lowercase());

        if let Some(caller) = caller {
            let tools = self.list_tools().await?;
            if let Some(tool) = tools.iter().find(|t| t.name == tool_name) {
                if !tool_policy::caller_is_allowed(tool, &caller) {
                    return Err(McpError::ToolCallFailed(format!(
                        "调用方 '{}' 无权调用工具 '{}'",
                        caller, tool_name
                    )));
                }
            }
        }

        self.call_tool(tool_name, arguments).await
    }

    /// # 实现步骤（Task 4.3）
    ///
    /// 1. 解析工具名称，确定目标服务器
    /// 2. 路由到正确的客户端
    /// 3. 执行工具调用
    /// 4. 转换结果为 McpToolResult
    /// 5. 返回结果
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolResult, McpError> {
        info!(tool_name = %tool_name, "调用 MCP 工具");

        // 1. 解析工具名称，确定目标服务器和实际工具名
        let (server_name, actual_tool_name) = self.resolve_tool_target(tool_name).await?;

        debug!(
            tool_name = %tool_name,
            server_name = %server_name,
            actual_tool_name = %actual_tool_name,
            "解析工具目标"
        );

        // 2. 获取目标服务器的客户端
        let clients = self.clients.read().await;
        let wrapper = clients
            .get(&server_name)
            .ok_or_else(|| McpError::ServerNotRunning(server_name.clone()))?;

        let service = wrapper
            .running_service()
            .ok_or_else(|| McpError::ServerNotRunning(server_name.clone()))?;
        let tool_timeout = super::bridge_tool_timeout(&wrapper.config);
        let service = service.clone();
        drop(clients);

        // 3. 构建工具调用参数
        let args = match arguments {
            serde_json::Value::Object(map) => Some(map),
            serde_json::Value::Null => None,
            _ => {
                return Err(McpError::ToolCallFailed(
                    "参数必须是 JSON 对象或 null".to_string(),
                ));
            }
        };

        // 4. 复用 connection-local active-time timeout，避免第二套 wall-clock timer。
        let client = crate::bridge_client::McpBridgeClient::new(service, tool_timeout);
        let result = client
            .call_tool(
                &actual_tool_name,
                args,
                Default::default(),
                None,
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .map_err(|error| {
                if matches!(&error, rmcp::service::ServiceError::Timeout { .. }) {
                    error!(
                        tool_name = %actual_tool_name,
                        server_name = %server_name,
                        timeout = ?tool_timeout,
                        "工具调用超时"
                    );
                    return McpError::Timeout;
                }
                error!(
                    tool_name = %actual_tool_name,
                    server_name = %server_name,
                    error = %error,
                    "工具调用失败"
                );
                McpError::ToolCallFailed(error.to_string())
            })?;

        // 5. 转换结果为 McpToolResult
        let mcp_result = Self::convert_call_tool_result(result);

        info!(
            tool_name = %actual_tool_name,
            server_name = %server_name,
            is_error = mcp_result.is_error,
            "工具调用完成"
        );

        Ok(mcp_result)
    }

    /// 解析工具目标（服务器名称和实际工具名）
    ///
    /// # Arguments
    ///
    /// * `tool_name` - 工具名称，格式为 `mcp__<server>__<tool>`
    ///
    /// # Returns
    ///
    /// 返回 (服务器名称, 实际工具名) 元组。
    ///
    /// # 解析逻辑
    ///
    /// 1. 按 `mcp__<server>__<tool>` 解析运行时工具名
    /// 2. 使用最长 server 名匹配，避免 server 名中包含 `__` 时误切割
    /// 3. 解析失败则视为未知 current 工具名
    async fn resolve_tool_target(&self, tool_name: &str) -> Result<(String, String), McpError> {
        let clients = self.clients.read().await;

        if let Some((server_name, actual_tool_name)) =
            parse_runtime_tool_name(tool_name, clients.keys())
        {
            return Ok((server_name, actual_tool_name));
        }

        // 工具未找到
        Err(McpError::ToolNotFound(tool_name.to_string()))
    }

    /// 转换 rmcp CallToolResult 为 McpToolResult
    pub(super) fn convert_call_tool_result(result: rmcp::model::CallToolResult) -> McpToolResult {
        let content: Vec<McpContent> = result
            .content
            .into_iter()
            .map(Self::convert_content)
            .collect();

        McpToolResult {
            content,
            structured_content: result.structured_content,
            is_error: result.is_error.unwrap_or(false),
        }
    }

    /// 转换 rmcp Content 为 McpContent
    pub(super) fn convert_content(content: rmcp::model::Content) -> McpContent {
        // Content 是 Annotated<RawContent>，需要访问内部的 raw 字段
        match content.raw {
            rmcp::model::RawContent::Text(text_content) => McpContent::Text {
                text: text_content.text,
            },
            rmcp::model::RawContent::Image(image_content) => McpContent::Image {
                data: image_content.data,
                mime_type: image_content.mime_type,
            },
            rmcp::model::RawContent::Resource(resource_content) => {
                let (uri, text, blob) = match resource_content.resource {
                    rmcp::model::ResourceContents::TextResourceContents { uri, text, .. } => {
                        (uri, Some(text), None)
                    }
                    rmcp::model::ResourceContents::BlobResourceContents { uri, blob, .. } => {
                        (uri, None, Some(blob))
                    }
                };
                McpContent::Resource { uri, text, blob }
            }
            rmcp::model::RawContent::Audio(audio_content) => {
                // 将音频内容作为 Image 类型处理（因为 McpContent 没有 Audio 变体）
                McpContent::Image {
                    data: audio_content.data,
                    mime_type: audio_content.mime_type,
                }
            }
            rmcp::model::RawContent::ResourceLink(resource_link) => McpContent::Resource {
                uri: resource_link.uri.clone(),
                text: Some(resource_link.name.clone()),
                blob: None,
            },
        }
    }
}

pub(super) fn app_tool_authority_from_meta(
    meta: Option<&rmcp::model::Meta>,
) -> McpAppToolAuthority {
    let connector_id = meta.and_then(|meta| non_empty_meta_string(meta, "connector_id"));
    let link_id = meta.and_then(|meta| non_empty_meta_string(meta, "link_id"));
    let requires_explicit_link_id = meta
        .and_then(|meta| meta.get("_codex_apps"))
        .and_then(serde_json::Value::as_object)
        .and_then(|meta| meta.get("requires_explicit_link_id"))
        .and_then(serde_json::Value::as_bool)
        == Some(true);
    McpAppToolAuthority {
        connector_id,
        link_id,
        requires_explicit_link_id,
    }
}

fn non_empty_meta_string(meta: &rmcp::model::Meta, key: &str) -> Option<String> {
    meta.get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

pub(crate) fn normalize_tool_input_schema(mut schema: serde_json::Value) -> serde_json::Value {
    if let serde_json::Value::Object(object) = &mut schema {
        if object
            .get("properties")
            .is_none_or(serde_json::Value::is_null)
        {
            object.insert(
                "properties".to_string(),
                serde_json::Value::Object(serde_json::Map::new()),
            );
        }
    }

    schema
}

pub(crate) fn tool_result_output_schema(
    structured_content_schema: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "array",
                "items": {
                    "type": "object"
                }
            },
            "structuredContent": structured_content_schema,
            "isError": {
                "type": "boolean"
            },
            "_meta": {
                "type": "object"
            }
        },
        "required": ["content"],
        "additionalProperties": false
    })
}
