use super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::protocol::v2;
use app_server_protocol::*;
use std::collections::VecDeque;

const MAX_MCP_RESOURCE_ORIGINS: usize = 64;
const MAX_MCP_RESOURCE_ORIGIN_BYTES: usize = 1024;

impl RuntimeCore {
    pub async fn list_mcp_servers(&self) -> Result<McpServerListResponse, RuntimeCoreError> {
        self.app_data_source.list_mcp_servers().await
    }

    pub async fn list_mcp_servers_with_status_v2(
        &self,
        params: v2::ListMcpServerStatusParams,
    ) -> Result<v2::ListMcpServerStatusResponse, RuntimeCoreError> {
        self.app_data_source
            .list_mcp_servers_with_status_v2(params)
            .await
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
        let origin_call_id =
            normalize_mcp_resource_identity(params.origin_call_id, "originCallId")?;
        let connector_id = normalize_mcp_resource_identity(params.connector_id, "connectorId")?;
        if thread_id.is_none() && origin_call_id.is_some() {
            return Err(RuntimeCoreError::InvalidRequest(
                "originCallId requires threadId".to_string(),
            ));
        }
        if let Some(thread_id) = thread_id {
            let effective_origin_call_id =
                origin_call_id.filter(|_| server == lime_skills::APPS_MCP_SERVER_NAME);
            let thread = self
                .read_thread(agent_protocol::thread::ThreadReadParams {
                    thread_id: agent_protocol::ThreadId::from(thread_id.to_string()),
                    turns_view: if effective_origin_call_id.is_some() {
                        agent_protocol::ThreadTurnsView::Full
                    } else {
                        agent_protocol::ThreadTurnsView::NotLoaded
                    },
                })
                .await?;
            let origin = effective_origin_call_id
                .as_deref()
                .map(|origin_call_id| {
                    find_mcp_resource_origin(
                        thread
                            .thread
                            .turns
                            .iter()
                            .flat_map(|turn| turn.items.iter()),
                        origin_call_id,
                        uri,
                    )
                })
                .transpose()?;
            let mut response = if let Some(origin) = origin.as_ref() {
                self.backend
                    .read_mcp_runtime_resource_for_origin(
                        &thread.thread.session_id.to_string(),
                        thread_id,
                        server,
                        origin,
                    )
                    .await?
            } else {
                self.backend
                    .read_mcp_runtime_resource(
                        &thread.thread.session_id.to_string(),
                        thread_id,
                        server,
                        uri,
                        mcp_resource_request_meta(connector_id.as_deref()),
                    )
                    .await?
            };
            response.origin_call_id = effective_origin_call_id;
            Ok(response)
        } else {
            self.app_data_source
                .read_mcp_server_resource(v2::McpServerResourceReadParams {
                    thread_id: None,
                    origin_call_id: None,
                    server: server.to_string(),
                    uri: uri.to_string(),
                    connector_id,
                })
                .await
        }
    }

    pub async fn subscribe_mcp_server_events(
        &self,
        thread_id: &str,
    ) -> Result<tokio::sync::broadcast::Receiver<lime_mcp::McpServerNotification>, RuntimeCoreError>
    {
        let thread = self
            .read_thread(agent_protocol::thread::ThreadReadParams {
                thread_id: agent_protocol::ThreadId::from(thread_id.to_string()),
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .await?;
        self.backend
            .subscribe_mcp_runtime_events(&thread.thread.session_id.to_string(), thread_id)
            .await
    }

    pub async fn open_mcp_server_event_stream(
        &self,
        thread_id: &str,
        server: &str,
        name: &str,
        arguments: serde_json::Value,
        meta: Option<serde_json::Value>,
    ) -> Result<lime_mcp::McpEventStream, RuntimeCoreError> {
        let thread = self
            .read_thread(agent_protocol::thread::ThreadReadParams {
                thread_id: agent_protocol::ThreadId::from(thread_id.to_string()),
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .await?;
        self.backend
            .open_mcp_runtime_event_stream(
                &thread.thread.session_id.to_string(),
                thread_id,
                server,
                name,
                arguments,
                meta,
            )
            .await
    }

    pub async fn has_mcp_server_for_thread(
        &self,
        thread_id: &str,
        server: &str,
    ) -> Result<bool, RuntimeCoreError> {
        let thread = self
            .read_thread(agent_protocol::thread::ThreadReadParams {
                thread_id: agent_protocol::ThreadId::from(thread_id.to_string()),
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .await?;
        self.backend
            .has_mcp_runtime_server(&thread.thread.session_id.to_string(), thread_id, server)
            .await
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

fn normalize_mcp_resource_identity(
    value: Option<String>,
    field: &str,
) -> Result<Option<String>, RuntimeCoreError> {
    value
        .map(|value| {
            let value = value.trim();
            if value.is_empty() {
                Err(RuntimeCoreError::InvalidRequest(format!(
                    "mcpServer/resource/read {field} cannot be empty"
                )))
            } else {
                Ok(value.to_string())
            }
        })
        .transpose()
}

fn mcp_resource_request_meta(connector_id: Option<&str>) -> Option<serde_json::Value> {
    connector_id.map(|connector_id| {
        serde_json::json!({
            "x-codex-turn-metadata": {
                "mcp_request_meta": {
                    "selected_connector_ids": [connector_id],
                },
            },
        })
    })
}

fn find_mcp_resource_origin<'a>(
    items: impl Iterator<Item = &'a agent_protocol::ThreadItem>,
    origin_call_id: &str,
    uri: &str,
) -> Result<lime_mcp::McpResourceOrigin, RuntimeCoreError> {
    let origin = collect_mcp_resource_origins(items)
        .into_iter()
        .rev()
        .find(|origin| origin.call_id == origin_call_id)
        .ok_or_else(origin_call_not_found)?;
    if origin.uri != uri {
        return Err(RuntimeCoreError::InvalidRequest(
            "originating MCP tool call does not match the requested resource".to_string(),
        ));
    }
    Ok(origin)
}

fn collect_mcp_resource_origins<'a>(
    items: impl Iterator<Item = &'a agent_protocol::ThreadItem>,
) -> VecDeque<lime_mcp::McpResourceOrigin> {
    let mut origins = VecDeque::new();
    for item in items {
        let agent_protocol::ThreadItemPayload::McpToolCall {
            server_name,
            tool_name,
            app_context: Some(app_context),
            arguments,
            ..
        } = &item.payload
        else {
            continue;
        };
        if item.status != agent_protocol::ItemStatus::Completed
            || server_name != lime_skills::APPS_MCP_SERVER_NAME
        {
            continue;
        }
        let connector_id = app_context.connector_id.trim();
        let Some(uri) = app_context
            .resource_uri
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        if connector_id.is_empty() {
            continue;
        }
        let link_id = app_context
            .link_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let argument_link_id = mcp_link_argument(arguments);
        let origin = lime_mcp::McpResourceOrigin {
            call_id: item.item_id.to_string(),
            turn_id: Some(item.turn_id.to_string()),
            tool: tool_name.clone(),
            connector_id: connector_id.to_string(),
            link_id: link_id.map(ToOwned::to_owned),
            uri: uri.to_string(),
            ambiguous_account: argument_link_id
                .as_deref()
                .is_some_and(|argument_link_id| Some(argument_link_id) != link_id),
        };
        if mcp_resource_origin_bytes(&origin) > MAX_MCP_RESOURCE_ORIGIN_BYTES {
            continue;
        }
        if let Some(index) = origins
            .iter()
            .position(|existing: &lime_mcp::McpResourceOrigin| existing.call_id == origin.call_id)
        {
            origins.remove(index);
        }
        if origins.len() >= MAX_MCP_RESOURCE_ORIGINS {
            origins.pop_front();
        }
        origins.push_back(origin);
    }
    origins
}

fn mcp_link_argument(arguments: &[agent_protocol::ToolArgument]) -> Option<String> {
    let value = arguments
        .iter()
        .find(|argument| argument.name == "link_id")?
        .value
        .trim();
    if value.is_empty() {
        return None;
    }
    serde_json::from_str::<serde_json::Value>(value)
        .ok()
        .and_then(|value| value.as_str().map(str::trim).map(ToOwned::to_owned))
        .filter(|value| !value.is_empty())
        .or_else(|| Some(value.to_string()))
}

fn mcp_resource_origin_bytes(origin: &lime_mcp::McpResourceOrigin) -> usize {
    origin.call_id.len()
        + origin.turn_id.as_ref().map_or(0, String::len)
        + origin.tool.len()
        + origin.connector_id.len()
        + origin.link_id.as_ref().map_or(0, String::len)
        + origin.uri.len()
}

fn origin_call_not_found() -> RuntimeCoreError {
    RuntimeCoreError::InvalidRequest(
        "originating MCP tool call was not found or did not complete successfully".to_string(),
    )
}

#[cfg(test)]
mod resource_read_tests {
    use super::*;
    use agent_protocol::{
        ItemId, ItemKind, ItemStatus, McpToolCallAppContext, SessionId, ThreadId, ThreadItem,
        ThreadItemPayload, ToolArgument, TurnId,
    };

    fn origin_item(index: usize) -> ThreadItem {
        ThreadItem {
            session_id: SessionId::new("session-1"),
            thread_id: ThreadId::new("thread-1"),
            turn_id: TurnId::new(format!("turn-{index}")),
            item_id: ItemId::new(format!("origin-{index}")),
            sequence: index as u64 + 1,
            ordinal: index as u64 + 1,
            created_at_ms: 1,
            updated_at_ms: 2,
            completed_at_ms: Some(2),
            kind: ItemKind::McpToolCall,
            status: ItemStatus::Completed,
            payload: ThreadItemPayload::McpToolCall {
                call_id: "provider-call".to_string(),
                server_name: lime_skills::APPS_MCP_SERVER_NAME.to_string(),
                tool_name: "search".to_string(),
                app_context: Some(McpToolCallAppContext {
                    connector_id: "calendar".to_string(),
                    link_id: Some("link-calendar".to_string()),
                    resource_uri: Some("ui://calendar/event".to_string()),
                    app_name: Some("Calendar".to_string()),
                    action_name: Some("search".to_string()),
                }),
                mcp_app_resource_uri: Some("ui://calendar/event".to_string()),
                plugin_id: None,
                arguments: vec![ToolArgument {
                    name: "link_id".to_string(),
                    value: "link-calendar".to_string(),
                }],
                output: None,
            },
            metadata: serde_json::Value::Null,
        }
    }

    #[test]
    fn origin_resource_uses_completed_canonical_app_context_and_matching_uri() {
        let valid = origin_item(0);
        let origin_call_id = valid.item_id.to_string();
        let origin = find_mcp_resource_origin(
            std::iter::once(&valid),
            &origin_call_id,
            "ui://calendar/event",
        )
        .expect("matching completed origin");
        assert_eq!(origin.connector_id, "calendar");
        assert_eq!(origin.link_id.as_deref(), Some("link-calendar"));
        assert_eq!(origin.turn_id.as_deref(), Some("turn-0"));
        assert!(!origin.ambiguous_account);

        let mut in_progress = valid.clone();
        in_progress.status = ItemStatus::InProgress;
        let mut wrong_server = valid.clone();
        let ThreadItemPayload::McpToolCall { server_name, .. } = &mut wrong_server.payload else {
            unreachable!();
        };
        *server_name = "docs".to_string();
        let mut missing_context = valid.clone();
        let ThreadItemPayload::McpToolCall { app_context, .. } = &mut missing_context.payload
        else {
            unreachable!();
        };
        *app_context = None;
        for invalid in [in_progress, wrong_server, missing_context] {
            assert!(find_mcp_resource_origin(
                std::iter::once(&invalid),
                &invalid.item_id.to_string(),
                "ui://calendar/event",
            )
            .is_err());
        }

        let mismatch = find_mcp_resource_origin(
            std::iter::once(&valid),
            &origin_call_id,
            "ui://calendar/other",
        )
        .expect_err("mismatched resource URI");
        assert!(mismatch
            .to_string()
            .contains("does not match the requested resource"));
    }

    #[test]
    fn origin_resource_marks_account_ambiguity_from_canonical_arguments() {
        let mut item = origin_item(0);
        let ThreadItemPayload::McpToolCall { arguments, .. } = &mut item.payload else {
            unreachable!();
        };
        arguments[0].value = "\"link-other\"".to_string();
        let origins = collect_mcp_resource_origins(std::iter::once(&item));
        assert_eq!(origins.len(), 1);
        assert!(origins[0].ambiguous_account);
    }

    #[test]
    fn origin_resource_history_is_bounded_and_survives_canonical_round_trip() {
        let items = (0..=MAX_MCP_RESOURCE_ORIGINS)
            .map(origin_item)
            .collect::<Vec<_>>();
        let restored = serde_json::from_value::<Vec<ThreadItem>>(
            serde_json::to_value(&items).expect("serialize canonical items"),
        )
        .expect("restore canonical items");
        let origins = collect_mcp_resource_origins(restored.iter());
        assert_eq!(origins.len(), MAX_MCP_RESOURCE_ORIGINS);
        assert!(origins
            .iter()
            .all(|origin| origin.call_id != "item_origin-0"));
        assert_eq!(
            origins.back().map(|origin| origin.call_id.as_str()),
            Some("item_origin-64")
        );
    }

    #[test]
    fn origin_resource_rejects_entries_larger_than_codex_bound() {
        let mut item = origin_item(0);
        let ThreadItemPayload::McpToolCall {
            app_context: Some(app_context),
            ..
        } = &mut item.payload
        else {
            unreachable!();
        };
        app_context.connector_id = "x".repeat(MAX_MCP_RESOURCE_ORIGIN_BYTES);
        assert!(collect_mcp_resource_origins(std::iter::once(&item)).is_empty());
    }

    #[test]
    fn connector_meta_matches_codex_selected_connector_shape_for_non_origin_reads() {
        assert_eq!(
            mcp_resource_request_meta(Some("calendar")),
            Some(serde_json::json!({
                "x-codex-turn-metadata": {
                    "mcp_request_meta": {
                        "selected_connector_ids": ["calendar"],
                    },
                },
            }))
        );
        assert_eq!(mcp_resource_request_meta(None), None);
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
