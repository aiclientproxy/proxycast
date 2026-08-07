use super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::protocol::v2;
use app_server_protocol::*;

impl RuntimeCore {
    pub async fn list_mcp_servers(&self) -> Result<McpServerListResponse, RuntimeCoreError> {
        self.app_data_source.list_mcp_servers().await
    }

    pub async fn list_mcp_servers_with_status(
        &self,
    ) -> Result<McpServerStatusListResponse, RuntimeCoreError> {
        self.app_data_source.list_mcp_servers_with_status().await
    }

    pub async fn create_mcp_server(
        &self,
        params: McpServerCreateParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        self.app_data_source.create_mcp_server(params).await
    }

    pub async fn update_mcp_server(
        &self,
        params: McpServerUpdateParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        self.app_data_source.update_mcp_server(params).await
    }

    pub async fn delete_mcp_server(
        &self,
        params: McpServerDeleteParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        self.app_data_source.delete_mcp_server(params).await
    }

    pub async fn set_mcp_server_enabled(
        &self,
        params: McpServerEnabledSetParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        self.app_data_source.set_mcp_server_enabled(params).await
    }

    pub async fn import_mcp_servers_from_app(
        &self,
        params: McpServerImportFromAppParams,
    ) -> Result<McpServerImportFromAppResponse, RuntimeCoreError> {
        self.app_data_source
            .import_mcp_servers_from_app(params)
            .await
    }

    pub async fn sync_all_mcp_servers_to_live(
        &self,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        self.app_data_source.sync_all_mcp_servers_to_live().await
    }

    pub async fn start_mcp_server(
        &self,
        params: McpServerStartParams,
    ) -> Result<McpServerLifecycleResponse, RuntimeCoreError> {
        self.app_data_source.start_mcp_server(params).await
    }

    pub async fn stop_mcp_server(
        &self,
        params: McpServerStopParams,
    ) -> Result<McpServerLifecycleResponse, RuntimeCoreError> {
        self.app_data_source.stop_mcp_server(params).await
    }

    pub async fn login_mcp_server_oauth(
        &self,
        params: McpServerOauthLoginParams,
    ) -> Result<lime_mcp::McpOAuthLoginHandle, RuntimeCoreError> {
        self.app_data_source.login_mcp_server_oauth(params).await
    }

    pub async fn list_mcp_tools(&self) -> Result<McpToolListResponse, RuntimeCoreError> {
        self.app_data_source.list_mcp_tools().await
    }

    pub async fn list_mcp_tools_for_context(
        &self,
        params: McpToolListForContextParams,
    ) -> Result<McpToolListResponse, RuntimeCoreError> {
        self.app_data_source
            .list_mcp_tools_for_context(params)
            .await
    }

    pub async fn search_mcp_tools(
        &self,
        params: McpToolSearchParams,
    ) -> Result<McpToolListResponse, RuntimeCoreError> {
        self.app_data_source.search_mcp_tools(params).await
    }

    pub async fn call_mcp_server_tool(
        &self,
        params: v2::McpServerToolCallParams,
    ) -> Result<v2::McpServerToolCallResponse, RuntimeCoreError> {
        let thread_id = params.thread_id.trim();
        let server = params.server.trim();
        let tool = params.tool.trim();
        if thread_id.is_empty() {
            return Err(RuntimeCoreError::InvalidRequest(
                "mcpServer/tool/call requires threadId".to_string(),
            ));
        }
        if server.is_empty() || tool.is_empty() {
            return Err(RuntimeCoreError::InvalidRequest(
                "mcpServer/tool/call requires server and tool".to_string(),
            ));
        }

        // Exact Codex calls are thread-scoped. Resolve the canonical thread
        // first, then execute through its Session-owned MCP runtime.
        let thread = self
            .read_thread(agent_protocol::thread::ThreadReadParams {
                thread_id: agent_protocol::ThreadId::from(thread_id.to_string()),
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .await?;
        let response = self
            .backend
            .call_mcp_runtime_tool(
                &thread.thread.session_id.to_string(),
                thread_id,
                server,
                tool,
                params
                    .arguments
                    .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new())),
            )
            .await?;

        Ok(v2::McpServerToolCallResponse {
            content: response
                .content
                .into_iter()
                .map(lower_mcp_content)
                .collect(),
            structured_content: response.structured_content,
            is_error: Some(response.is_error),
            // Request metadata is not provider result metadata. The current
            // manager response does not preserve result `_meta`.
            meta: None,
        })
    }

    pub async fn list_mcp_prompts(&self) -> Result<McpPromptListResponse, RuntimeCoreError> {
        self.app_data_source.list_mcp_prompts().await
    }

    pub async fn get_mcp_prompt(
        &self,
        params: McpPromptGetParams,
    ) -> Result<McpPromptGetResponse, RuntimeCoreError> {
        self.app_data_source.get_mcp_prompt(params).await
    }

    pub async fn list_mcp_resources(&self) -> Result<McpResourceListResponse, RuntimeCoreError> {
        self.app_data_source.list_mcp_resources().await
    }

    pub async fn read_mcp_server_resource(
        &self,
        params: v2::McpServerResourceReadParams,
    ) -> Result<v2::McpServerResourceReadResponse, RuntimeCoreError> {
        let server = params.server.trim();
        let uri = params.uri.trim();
        if server.is_empty() || uri.is_empty() {
            return Err(RuntimeCoreError::InvalidRequest(
                "mcpServer/resource/read requires server and uri".to_string(),
            ));
        }

        let thread_id = match params.thread_id.as_deref() {
            Some(value) if value.trim().is_empty() => {
                return Err(RuntimeCoreError::InvalidRequest(
                    "mcpServer/resource/read threadId cannot be empty".to_string(),
                ));
            }
            Some(value) => Some(value.trim()),
            None => None,
        };
        if let Some(thread_id) = thread_id {
            let thread = self
                .read_thread(agent_protocol::thread::ThreadReadParams {
                    thread_id: agent_protocol::ThreadId::from(thread_id.to_string()),
                    turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
                })
                .await?;
            self.backend
                .read_mcp_runtime_resource(
                    &thread.thread.session_id.to_string(),
                    thread_id,
                    server,
                    uri,
                )
                .await
        } else {
            self.app_data_source
                .read_mcp_server_resource(v2::McpServerResourceReadParams {
                    thread_id: None,
                    server: server.to_string(),
                    uri: uri.to_string(),
                })
                .await
        }
    }

    pub async fn subscribe_mcp_resource(
        &self,
        params: McpResourceSubscribeParams,
    ) -> Result<McpResourceSubscriptionResponse, RuntimeCoreError> {
        self.app_data_source.subscribe_mcp_resource(params).await
    }

    pub async fn unsubscribe_mcp_resource(
        &self,
        params: McpResourceUnsubscribeParams,
    ) -> Result<McpResourceSubscriptionResponse, RuntimeCoreError> {
        self.app_data_source.unsubscribe_mcp_resource(params).await
    }
}

fn lower_mcp_content(content: lime_mcp::McpContent) -> serde_json::Value {
    match content {
        lime_mcp::McpContent::Text { text } => serde_json::json!({
            "type": "text",
            "text": text,
        }),
        lime_mcp::McpContent::Image { data, mime_type } => serde_json::json!({
            "type": "image",
            "data": data,
            "mimeType": mime_type,
        }),
        lime_mcp::McpContent::Resource { uri, text, blob } => {
            let mut value = serde_json::Map::from_iter([
                (
                    "type".to_string(),
                    serde_json::Value::String("resource".to_string()),
                ),
                ("uri".to_string(), serde_json::Value::String(uri)),
            ]);
            if let Some(text) = text {
                value.insert("text".to_string(), serde_json::Value::String(text));
            }
            if let Some(blob) = blob {
                value.insert("blob".to_string(), serde_json::Value::String(blob));
            }
            serde_json::Value::Object(value)
        }
    }
}
