use super::image_tools;
use super::mcp_resource_tools;
use super::memory_tools;
use super::tool_search_tools;
use crate::AppDataSource;
use crate::RuntimeCoreError;
use lime_agent::AgentRuntimeState;
use std::sync::{Arc, RwLock};

pub(crate) async fn register_current_native_tools_if_available(
    agent_state: &AgentRuntimeState,
    app_data_source: &RwLock<Option<Arc<dyn AppDataSource>>>,
    current_time_gateway: &RwLock<Option<Arc<dyn tool_runtime::current_time::CurrentTimeGateway>>>,
) -> Result<(), RuntimeCoreError> {
    if !agent_state.is_initialized().await {
        return Ok(());
    }
    let current_time_gateway = current_time_gateway
        .read()
        .map_err(|_| RuntimeCoreError::Backend("current-time gateway lock poisoned".to_string()))?
        .clone();
    if let Some(current_time_gateway) = current_time_gateway {
        agent_state
            .register_current_time_tools(current_time_gateway)
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
    }
    let app_data_source = app_data_source
        .read()
        .map_err(|_| {
            RuntimeCoreError::Backend("native tool app data source lock poisoned".to_string())
        })?
        .clone();
    let Some(app_data_source) = app_data_source else {
        return Ok(());
    };
    agent_state
        .register_memory_store_tools(memory_tools::memory_store_gateway(app_data_source.clone()))
        .await
        .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
    agent_state
        .register_image_task_tools(image_tools::image_task_gateway(app_data_source.clone()))
        .await
        .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
    agent_state
        .register_tool_search_tools(tool_search_tools::tool_search_gateway(
            app_data_source.clone(),
        ))
        .await
        .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
    agent_state
        .register_mcp_resource_tools(mcp_resource_tools::mcp_resource_gateway(app_data_source))
        .await
        .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
    Ok(())
}
