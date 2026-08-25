use super::McpClientManager;
use crate::types::*;
use rmcp::model::{
    ClientRequest, Extensions, Meta, ReadResourceRequest, ReadResourceRequestParam, ServerResult,
};
use tracing::{debug, error, info, warn};

impl McpClientManager {
    /// 获取指定运行中服务器的一页资源。
    pub async fn list_resource_page(
        &self,
        server_name: &str,
        cursor: Option<String>,
    ) -> Result<McpResourcePage, McpError> {
        let server_name = validate_server_name(server_name)?;
        let clients = self.clients.read().await;
        let wrapper = clients
            .get(server_name)
            .ok_or_else(|| McpError::ServerNotRunning(server_name.to_string()))?;
        if let Some(info) = &wrapper.server_info {
            if !info.supports_resources {
                return Ok(McpResourcePage {
                    resources: Vec::new(),
                    next_cursor: None,
                });
            }
        }
        let service = wrapper
            .running_service()
            .ok_or_else(|| McpError::ServerNotRunning(server_name.to_string()))?;
        let result = service
            .list_resources(cursor.map(|cursor| rmcp::model::PaginatedRequestParam {
                cursor: Some(cursor),
            }))
            .await
            .map_err(|error| McpError::ProtocolError(error.to_string()))?;
        Ok(McpResourcePage {
            resources: result
                .resources
                .into_iter()
                .map(|resource| {
                    Self::convert_resource_to_definition(resource, server_name.to_string())
                })
                .collect(),
            next_cursor: result.next_cursor,
        })
    }

    // ========================================================================
    // 资源管理方法
    // ========================================================================

    /// 获取所有资源
    ///
    /// 从所有运行中的服务器获取资源定义。
    ///
    /// # Returns
    ///
    /// 返回所有可用资源的定义列表。
    ///
    /// # 实现步骤（Task 4.5）
    ///
    /// 1. 遍历所有运行中的服务器
    /// 2. 检查服务器是否支持资源
    /// 3. 调用 list_all_resources 获取资源列表
    /// 4. 转换为 McpResourceDefinition 格式
    /// 5. 返回合并后的资源列表
    pub async fn list_resources(&self) -> Result<Vec<McpResourceDefinition>, McpError> {
        info!("获取所有 MCP 资源");

        let mut all_resources: Vec<McpResourceDefinition> = Vec::new();
        let clients = self.clients.read().await;

        for (server_name, wrapper) in clients.iter() {
            // 检查服务器是否支持资源
            if let Some(ref info) = wrapper.server_info {
                if !info.supports_resources {
                    debug!(server_name = %server_name, "服务器不支持资源，跳过");
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

            // 调用 list_all_resources 获取所有资源
            match service.list_all_resources().await {
                Ok(resources) => {
                    debug!(
                        server_name = %server_name,
                        resource_count = resources.len(),
                        "获取服务器资源列表成功"
                    );
                    for resource in resources {
                        all_resources.push(Self::convert_resource_to_definition(
                            resource,
                            server_name.clone(),
                        ));
                    }
                }
                Err(e) => {
                    warn!(
                        server_name = %server_name,
                        error = %e,
                        "获取服务器资源列表失败"
                    );
                    // 继续处理其他服务器，不中断
                }
            }
        }

        info!(resource_count = all_resources.len(), "资源列表已获取");
        Ok(all_resources)
    }

    /// 获取所有资源模板。
    pub async fn list_resource_templates(
        &self,
    ) -> Result<Vec<McpResourceTemplateDefinition>, McpError> {
        info!("获取所有 MCP 资源模板");

        let mut all_templates: Vec<McpResourceTemplateDefinition> = Vec::new();
        let clients = self.clients.read().await;

        for (server_name, wrapper) in clients.iter() {
            if let Some(ref info) = wrapper.server_info {
                if !info.supports_resources {
                    debug!(server_name = %server_name, "服务器不支持资源，跳过资源模板");
                    continue;
                }
            }

            let service = match wrapper.running_service() {
                Some(service) => service,
                None => {
                    warn!(server_name = %server_name, "服务器无运行服务，跳过资源模板");
                    continue;
                }
            };

            match service.list_all_resource_templates().await {
                Ok(templates) => {
                    debug!(
                        server_name = %server_name,
                        template_count = templates.len(),
                        "获取服务器资源模板列表成功"
                    );
                    for template in templates {
                        all_templates.push(Self::convert_resource_template_to_definition(
                            template,
                            server_name.clone(),
                        ));
                    }
                }
                Err(error) => {
                    warn!(
                        server_name = %server_name,
                        error = %error,
                        "获取服务器资源模板列表失败"
                    );
                }
            }
        }

        info!(template_count = all_templates.len(), "资源模板列表已获取");
        Ok(all_templates)
    }

    /// 将 rmcp Resource 转换为 McpResourceDefinition
    pub(super) fn convert_resource_to_definition(
        resource: rmcp::model::Resource,
        server_name: String,
    ) -> McpResourceDefinition {
        McpResourceDefinition {
            uri: resource.uri.clone(),
            name: resource.name.clone(),
            description: resource.description.clone(),
            mime_type: resource.mime_type.clone(),
            server_name,
            meta: resource
                .meta
                .as_ref()
                .map(|meta| serde_json::Value::Object(meta.0.clone())),
        }
    }

    /// 将 rmcp ResourceTemplate 转换为 McpResourceTemplateDefinition
    pub(super) fn convert_resource_template_to_definition(
        template: rmcp::model::ResourceTemplate,
        server_name: String,
    ) -> McpResourceTemplateDefinition {
        McpResourceTemplateDefinition {
            uri_template: template.uri_template.clone(),
            name: template.name.clone(),
            title: template.title.clone(),
            description: template.description.clone(),
            mime_type: template.mime_type.clone(),
            server_name,
        }
    }

    /// 读取资源内容
    ///
    /// # Arguments
    ///
    /// * `server_name` - 精确的服务器名称
    /// * `uri` - 资源 URI
    ///
    /// # Returns
    ///
    /// 返回资源内容。
    ///
    /// # 实现步骤（Task 4.5）
    ///
    /// 1. 按服务器名称精确选择运行连接
    /// 2. 调用服务器的 read_resource 方法
    /// 3. 转换并返回资源内容
    pub async fn read_resource(
        &self,
        server_name: &str,
        uri: &str,
    ) -> Result<McpResourceContent, McpError> {
        self.read_resource_with_meta(server_name, uri, None).await
    }

    /// Read one resource while preserving request-scoped MCP metadata on the wire.
    pub async fn read_resource_with_meta(
        &self,
        server_name: &str,
        uri: &str,
        meta: Option<serde_json::Value>,
    ) -> Result<McpResourceContent, McpError> {
        let (server_name, uri) = validate_resource_target(server_name, uri)?;
        info!(server_name = %server_name, uri = %uri, "读取 MCP 资源");
        let clients = self.clients.read().await;
        let wrapper = clients
            .get(server_name)
            .ok_or_else(|| McpError::ServerNotRunning(server_name.to_string()))?;

        let service = wrapper
            .running_service()
            .ok_or_else(|| McpError::ServerNotRunning(server_name.to_string()))?;

        let read_param = ReadResourceRequestParam {
            uri: uri.to_string(),
        };
        let mut request = ReadResourceRequest::new(read_param);
        if let Some(meta) = meta {
            let meta = meta.as_object().cloned().ok_or_else(|| {
                McpError::ConfigError(
                    "MCP resource request _meta must be a JSON object".to_string(),
                )
            })?;
            let mut extensions = Extensions::new();
            extensions.insert(Meta(meta));
            request.extensions = extensions;
        }
        let result = service
            .peer()
            .send_request(ClientRequest::ReadResourceRequest(request))
            .await
            .and_then(|result| match result {
                ServerResult::ReadResourceResult(result) => Ok(result),
                _ => Err(rmcp::service::ServiceError::UnexpectedResponse),
            })
            .map_err(|e| {
                error!(
                    uri = %uri,
                    server_name = %server_name,
                    error = %e,
                    "读取资源失败"
                );
                McpError::ToolCallFailed(format!("读取资源失败: {e}"))
            })?;

        let mcp_result = Self::convert_read_resource_result(uri, result);

        info!(
            uri = %uri,
            server_name = %server_name,
            "资源读取完成"
        );

        Ok(mcp_result)
    }

    pub async fn read_resource_for_origin(
        &self,
        server_name: &str,
        thread_id: &str,
        origin: &crate::types::McpResourceOrigin,
    ) -> Result<McpResourceContent, McpError> {
        if server_name != "codex_apps" {
            return Err(McpError::ConfigError(
                "MCP resource origins are only valid for codex_apps".to_string(),
            ));
        }
        let thread_id = thread_id.trim();
        if thread_id.is_empty() {
            return Err(McpError::ConfigError(
                "MCP resource origin requires thread identity".to_string(),
            ));
        }
        if origin.uri != origin.uri.trim() || origin.uri.is_empty() {
            return Err(McpError::ConfigError(
                "MCP resource origin URI is invalid".to_string(),
            ));
        }
        if origin.ambiguous_account {
            return Err(McpError::ConfigError(
                "originating MCP tool call has ambiguous account selection".to_string(),
            ));
        }

        let authority = self.app_tool_authority(server_name, &origin.tool).await?;
        if authority.connector_id.as_deref() != Some(origin.connector_id.as_str()) {
            return Err(McpError::ConfigError(
                "originating MCP tool connector does not match its app context".to_string(),
            ));
        }
        if authority.link_id.as_deref() != origin.link_id.as_deref() {
            return Err(McpError::ConfigError(
                "originating MCP tool link does not match its app context".to_string(),
            ));
        }
        if origin.link_id.is_none() && authority.requires_explicit_link_id {
            return Err(McpError::ConfigError(
                "originating MCP tool requires an explicit account link".to_string(),
            ));
        }

        self.read_resource_with_meta(
            server_name,
            &origin.uri,
            Some(serde_json::json!({
                "threadId": thread_id,
                "x-codex-turn-metadata": {
                    "mcp_request_meta": {
                        "selected_connector_ids": [&origin.connector_id],
                        "link_id": &origin.link_id,
                    },
                },
            })),
        )
        .await
    }

    /// 订阅资源更新。
    pub async fn subscribe_resource(&self, server_name: &str, uri: &str) -> Result<(), McpError> {
        let (server_name, uri) = validate_resource_target(server_name, uri)?;
        info!(server_name = %server_name, uri = %uri, "订阅 MCP 资源");
        let clients = self.clients.read().await;
        let wrapper = clients
            .get(server_name)
            .ok_or_else(|| McpError::ServerNotRunning(server_name.to_string()))?;
        let service = wrapper
            .running_service()
            .ok_or_else(|| McpError::ServerNotRunning(server_name.to_string()))?;

        service
            .subscribe(rmcp::model::SubscribeRequestParam {
                uri: uri.to_string(),
            })
            .await
            .map_err(|error| {
                error!(
                    uri = %uri,
                    server_name = %server_name,
                    error = %error,
                    "订阅资源失败"
                );
                McpError::ToolCallFailed(format!("订阅资源失败: {error}"))
            })?;

        info!(uri = %uri, server_name = %server_name, "资源订阅完成");
        Ok(())
    }

    /// 取消订阅资源更新。
    pub async fn unsubscribe_resource(&self, server_name: &str, uri: &str) -> Result<(), McpError> {
        let (server_name, uri) = validate_resource_target(server_name, uri)?;
        info!(server_name = %server_name, uri = %uri, "取消订阅 MCP 资源");
        let clients = self.clients.read().await;
        let wrapper = clients
            .get(server_name)
            .ok_or_else(|| McpError::ServerNotRunning(server_name.to_string()))?;
        let service = wrapper
            .running_service()
            .ok_or_else(|| McpError::ServerNotRunning(server_name.to_string()))?;

        service
            .unsubscribe(rmcp::model::UnsubscribeRequestParam {
                uri: uri.to_string(),
            })
            .await
            .map_err(|error| {
                error!(
                    uri = %uri,
                    server_name = %server_name,
                    error = %error,
                    "取消订阅资源失败"
                );
                McpError::ToolCallFailed(format!("取消订阅资源失败: {error}"))
            })?;

        info!(uri = %uri, server_name = %server_name, "资源取消订阅完成");
        Ok(())
    }

    /// 转换 rmcp ReadResourceResult 为 McpResourceContent
    pub(super) fn convert_read_resource_result(
        uri: &str,
        result: rmcp::model::ReadResourceResult,
    ) -> McpResourceContent {
        // 获取第一个内容（通常只有一个）
        if let Some(content) = result.contents.into_iter().next() {
            match content {
                rmcp::model::ResourceContents::TextResourceContents {
                    uri: content_uri,
                    mime_type,
                    text,
                    meta,
                } => McpResourceContent {
                    uri: content_uri,
                    mime_type,
                    text: Some(text),
                    blob: None,
                    meta: meta.map(|value| serde_json::Value::Object(value.0)),
                },
                rmcp::model::ResourceContents::BlobResourceContents {
                    uri: content_uri,
                    mime_type,
                    blob,
                    meta,
                } => McpResourceContent {
                    uri: content_uri,
                    mime_type,
                    text: None,
                    blob: Some(blob),
                    meta: meta.map(|value| serde_json::Value::Object(value.0)),
                },
            }
        } else {
            // 如果没有内容，返回空的资源内容
            McpResourceContent {
                uri: uri.to_string(),
                mime_type: None,
                text: None,
                blob: None,
                meta: None,
            }
        }
    }
}

fn validate_server_name(server_name: &str) -> Result<&str, McpError> {
    let server_name = server_name.trim();
    if server_name.is_empty() {
        return Err(McpError::ConfigError(
            "MCP resource server cannot be empty".to_string(),
        ));
    }
    Ok(server_name)
}

fn validate_resource_target<'a>(
    server_name: &'a str,
    uri: &'a str,
) -> Result<(&'a str, &'a str), McpError> {
    let server_name = server_name.trim();
    let uri = uri.trim();
    if server_name.is_empty() {
        return Err(McpError::ConfigError(
            "MCP resource server cannot be empty".to_string(),
        ));
    }
    if uri.is_empty() {
        return Err(McpError::ConfigError(
            "MCP resource URI cannot be empty".to_string(),
        ));
    }
    Ok((server_name, uri))
}
