use super::data_error;
use super::values_from_serializable_vec;
use crate::RuntimeCoreError;
use app_server_protocol::protocol::v2::{
    McpServerResourceContent, McpServerResourceReadParams, McpServerResourceReadResponse,
};
use app_server_protocol::McpContent;
use app_server_protocol::McpPromptGetParams;
use app_server_protocol::McpPromptGetResponse;
use app_server_protocol::McpPromptListResponse;
use app_server_protocol::McpPromptMessage;
use app_server_protocol::McpResourceListResponse;
use app_server_protocol::McpResourceSubscribeParams;
use app_server_protocol::McpResourceSubscriptionResponse;
use app_server_protocol::McpResourceUnsubscribeParams;
use app_server_protocol::McpServerCreateParams;
use app_server_protocol::McpServerDeleteParams;
use app_server_protocol::McpServerEnabledSetParams;
use app_server_protocol::McpServerImportFromAppParams;
use app_server_protocol::McpServerImportFromAppResponse;
use app_server_protocol::McpServerLifecycleResponse;
use app_server_protocol::McpServerListResponse;
use app_server_protocol::McpServerOauthLoginParams;
use app_server_protocol::McpServerStartParams;
use app_server_protocol::McpServerStatusListResponse;
use app_server_protocol::McpServerStopParams;
use app_server_protocol::McpServerUpdateParams;
use app_server_protocol::McpToolListForContextParams;
use app_server_protocol::McpToolListResponse;
use app_server_protocol::McpToolSearchParams;
use lime_core::database::DbConnection;
use lime_core::models::McpServer;
use lime_mcp::McpError;
use lime_mcp::McpManagerState;
use lime_mcp::McpServerConfig;
use lime_services::mcp_service::McpService;
use serde_json::json;
use serde_json::Value;
use std::collections::HashSet;
use std::path::Path;

pub(crate) fn list_mcp_servers(
    db: &DbConnection,
) -> Result<McpServerListResponse, RuntimeCoreError> {
    Ok(McpServerListResponse {
        servers: values_from_serializable_vec(McpService::get_all(db).map_err(data_error)?)?,
    })
}

pub(crate) async fn list_mcp_servers_with_status(
    db: &DbConnection,
    manager: &McpManagerState,
) -> Result<McpServerStatusListResponse, RuntimeCoreError> {
    let servers = McpService::get_all(db).map_err(data_error)?;
    let manager = manager.lock().await;
    let mut result = Vec::with_capacity(servers.len());
    for server in servers {
        let parsed_config = parse_mcp_server_config(&server.server_config);
        let runtime_status = manager
            .get_server_runtime_status(&server.name, Some(&parsed_config))
            .await;
        result.push(json!({
            "id": server.id,
            "name": server.name,
            "description": server.description,
            "config": parsed_config,
            "is_running": runtime_status.is_running,
            "server_info": runtime_status.server_info,
            "runtime_status": runtime_status,
            "enabled_lime": server.enabled_lime,
            "enabled_claude": server.enabled_claude,
            "enabled_codex": server.enabled_codex,
            "enabled_gemini": server.enabled_gemini,
        }));
    }
    Ok(McpServerStatusListResponse { servers: result })
}

pub(crate) fn create_mcp_server(
    db: &DbConnection,
    params: McpServerCreateParams,
) -> Result<McpServerListResponse, RuntimeCoreError> {
    let server = mcp_server_from_value(params.server)?;
    McpService::add(db, server).map_err(data_error)?;
    list_mcp_servers(db)
}

pub(crate) fn update_mcp_server(
    db: &DbConnection,
    params: McpServerUpdateParams,
) -> Result<McpServerListResponse, RuntimeCoreError> {
    let server = mcp_server_from_value(params.server)?;
    McpService::update(db, server).map_err(data_error)?;
    list_mcp_servers(db)
}

pub(crate) fn delete_mcp_server(
    db: &DbConnection,
    params: McpServerDeleteParams,
) -> Result<McpServerListResponse, RuntimeCoreError> {
    McpService::delete(db, &params.id).map_err(data_error)?;
    list_mcp_servers(db)
}

pub(crate) fn set_mcp_server_enabled(
    db: &DbConnection,
    params: McpServerEnabledSetParams,
) -> Result<McpServerListResponse, RuntimeCoreError> {
    McpService::toggle_enabled(db, &params.id, &params.app_type, params.enabled)
        .map_err(data_error)?;
    list_mcp_servers(db)
}

pub(crate) fn import_mcp_servers_from_app(
    db: &DbConnection,
    params: McpServerImportFromAppParams,
) -> Result<McpServerImportFromAppResponse, RuntimeCoreError> {
    let imported_count = McpService::import_from_app(db, &params.app_type).map_err(data_error)?;
    let servers = list_mcp_servers(db)?.servers;
    Ok(McpServerImportFromAppResponse {
        imported_count,
        servers,
    })
}

pub(crate) fn sync_all_mcp_servers_to_live(
    db: &DbConnection,
) -> Result<McpServerListResponse, RuntimeCoreError> {
    McpService::sync_all_to_live(db).map_err(data_error)?;
    list_mcp_servers(db)
}

pub(crate) async fn start_mcp_server(
    db: &DbConnection,
    manager: &McpManagerState,
    params: McpServerStartParams,
) -> Result<McpServerLifecycleResponse, RuntimeCoreError> {
    let server = McpService::get_all(db)
        .map_err(data_error)?
        .into_iter()
        .find(|server| server.name == params.name)
        .ok_or_else(|| {
            RuntimeCoreError::Backend(format!("MCP server not found: {}", params.name))
        })?;
    let config = parse_mcp_server_config(&server.server_config);
    let manager = manager.lock().await;
    manager
        .start_server(&params.name, &config)
        .await
        .map_err(mcp_error)?;
    Ok(McpServerLifecycleResponse::default())
}

pub(crate) async fn stop_mcp_server(
    manager: &McpManagerState,
    params: McpServerStopParams,
) -> Result<McpServerLifecycleResponse, RuntimeCoreError> {
    let manager = manager.lock().await;
    manager.stop_server(&params.name).await.map_err(mcp_error)?;
    Ok(McpServerLifecycleResponse::default())
}

pub(crate) async fn login_mcp_server_oauth(
    db: &DbConnection,
    manager: &McpManagerState,
    params: McpServerOauthLoginParams,
) -> Result<lime_mcp::McpOAuthLoginHandle, RuntimeCoreError> {
    let server = McpService::get_all(db)
        .map_err(data_error)?
        .into_iter()
        .find(|server| server.name == params.name)
        .ok_or_else(|| {
            RuntimeCoreError::Backend(format!("MCP server not found: {}", params.name))
        })?;
    let config = parse_mcp_server_config(&server.server_config);
    let manager = manager.lock().await;
    manager
        .start_oauth_login(
            &params.name,
            &config,
            params.scopes.clone(),
            params.timeout_secs,
        )
        .await
        .map_err(mcp_error)
}

pub(crate) async fn list_mcp_tools(
    manager: &McpManagerState,
) -> Result<McpToolListResponse, RuntimeCoreError> {
    let manager = manager.lock().await;
    Ok(McpToolListResponse {
        tools: values_from_serializable_vec(manager.list_tools().await.map_err(mcp_error)?)?,
    })
}

pub(crate) fn list_mcp_runtime_server_specs(
    db: &DbConnection,
    plugin_data_root: &Path,
) -> Result<Vec<lime_mcp::McpRuntimeServerSpec>, RuntimeCoreError> {
    let mut specs = McpService::get_all(db)
        .map_err(data_error)?
        .into_iter()
        .filter(|server| server.enabled_lime)
        .filter_map(|server| {
            let config = parse_mcp_server_config(&server.server_config);
            config.enabled.then(|| lime_mcp::McpRuntimeServerSpec {
                name: server.name,
                plugin_id: None,
                config,
            })
        })
        .collect::<Vec<_>>();
    let mut names = specs
        .iter()
        .map(|spec| spec.name.clone())
        .collect::<HashSet<_>>();
    match super::plugin_catalog::list_plugin_mcp_runtime_server_specs(plugin_data_root) {
        Ok(plugin_specs) => {
            for spec in plugin_specs {
                if !names.insert(spec.name.clone()) {
                    tracing::warn!(
                        server_name = %spec.name,
                        "跳过与现有 MCP server 同名的 Plugin MCP server"
                    );
                    continue;
                }
                specs.push(spec);
            }
        }
        Err(error) => {
            tracing::warn!(%error, "Plugin MCP runtime 配置读取失败，跳过 Plugin MCP servers")
        }
    }
    specs.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(specs)
}

pub(crate) async fn list_mcp_tools_for_context(
    manager: &McpManagerState,
    params: McpToolListForContextParams,
) -> Result<McpToolListResponse, RuntimeCoreError> {
    let manager = manager.lock().await;
    Ok(McpToolListResponse {
        tools: values_from_serializable_vec(
            manager
                .list_tools_for_context(params.caller.as_deref(), params.include_deferred)
                .await
                .map_err(mcp_error)?,
        )?,
    })
}

pub(crate) async fn search_mcp_tools(
    manager: &McpManagerState,
    params: McpToolSearchParams,
) -> Result<McpToolListResponse, RuntimeCoreError> {
    let manager = manager.lock().await;
    Ok(McpToolListResponse {
        tools: values_from_serializable_vec(
            manager
                .search_tools(&params.query, params.limit, params.caller.as_deref())
                .await
                .map_err(mcp_error)?,
        )?,
    })
}

pub(crate) async fn list_mcp_prompts(
    manager: &McpManagerState,
) -> Result<McpPromptListResponse, RuntimeCoreError> {
    let manager = manager.lock().await;
    Ok(McpPromptListResponse {
        prompts: values_from_serializable_vec(manager.list_prompts().await.map_err(mcp_error)?)?,
    })
}

pub(crate) async fn get_mcp_prompt(
    manager: &McpManagerState,
    params: McpPromptGetParams,
) -> Result<McpPromptGetResponse, RuntimeCoreError> {
    let manager = manager.lock().await;
    let result = manager
        .get_prompt(&params.server, &params.name, params.arguments)
        .await
        .map_err(mcp_error)?;
    Ok(to_mcp_prompt_get_response(result))
}

pub(crate) async fn list_mcp_resources(
    manager: &McpManagerState,
) -> Result<McpResourceListResponse, RuntimeCoreError> {
    let manager = manager.lock().await;
    Ok(McpResourceListResponse {
        resources: values_from_serializable_vec(
            manager.list_resources().await.map_err(mcp_error)?,
        )?,
        resource_templates: values_from_serializable_vec(
            manager.list_resource_templates().await.map_err(mcp_error)?,
        )?,
        next_cursor: None,
    })
}

pub(crate) async fn read_mcp_server_resource(
    manager: &McpManagerState,
    params: McpServerResourceReadParams,
) -> Result<McpServerResourceReadResponse, RuntimeCoreError> {
    let manager = manager.lock().await;
    let result = manager
        .read_resource(&params.server, &params.uri)
        .await
        .map_err(mcp_error)?;
    let content = match (result.text, result.blob) {
        (Some(text), None) => Some(McpServerResourceContent::Text {
            uri: result.uri,
            mime_type: result.mime_type,
            text,
            meta: result.meta,
        }),
        (None, Some(blob)) => Some(McpServerResourceContent::Blob {
            uri: result.uri,
            mime_type: result.mime_type,
            blob,
            meta: result.meta,
        }),
        (None, None) => None,
        (Some(_), Some(_)) => {
            return Err(RuntimeCoreError::Backend(
                "MCP resource response contained both text and blob".to_string(),
            ));
        }
    };
    Ok(McpServerResourceReadResponse {
        contents: content.into_iter().collect(),
    })
}

pub(crate) async fn subscribe_mcp_resource(
    manager: &McpManagerState,
    params: McpResourceSubscribeParams,
) -> Result<McpResourceSubscriptionResponse, RuntimeCoreError> {
    let manager = manager.lock().await;
    manager
        .subscribe_resource(&params.server, &params.uri)
        .await
        .map_err(mcp_error)?;
    Ok(McpResourceSubscriptionResponse::default())
}

pub(crate) async fn unsubscribe_mcp_resource(
    manager: &McpManagerState,
    params: McpResourceUnsubscribeParams,
) -> Result<McpResourceSubscriptionResponse, RuntimeCoreError> {
    let manager = manager.lock().await;
    manager
        .unsubscribe_resource(&params.server, &params.uri)
        .await
        .map_err(mcp_error)?;
    Ok(McpResourceSubscriptionResponse::default())
}

fn parse_mcp_server_config(config_value: &Value) -> McpServerConfig {
    McpServerConfig::from_value(config_value.clone()).unwrap_or_default()
}

fn mcp_server_from_value(value: Value) -> Result<McpServer, RuntimeCoreError> {
    serde_json::from_value(value).map_err(data_error)
}

fn mcp_error(error: McpError) -> RuntimeCoreError {
    RuntimeCoreError::Backend(format!("MCP current runtime error: {error}"))
}

fn to_mcp_prompt_get_response(result: lime_mcp::McpPromptResult) -> McpPromptGetResponse {
    McpPromptGetResponse {
        description: result.description,
        messages: result
            .messages
            .into_iter()
            .map(|message| McpPromptMessage {
                role: message.role,
                content: to_mcp_content(message.content),
            })
            .collect(),
    }
}

fn to_mcp_content(content: lime_mcp::McpContent) -> McpContent {
    match content {
        lime_mcp::McpContent::Text { text } => McpContent::Text { text },
        lime_mcp::McpContent::Image { data, mime_type } => McpContent::Image { data, mime_type },
        lime_mcp::McpContent::Resource { uri, text, blob } => {
            McpContent::Resource { uri, text, blob }
        }
    }
}
