//! MCP 事件 Payload。

use crate::types::McpToolDefinition;
use serde::Serialize;

/// 工具列表更新事件
#[derive(Debug, Clone, Serialize)]
pub struct McpToolsUpdatedPayload {
    pub tools: Vec<McpToolDefinition>,
}

/// 资源列表更新事件
#[derive(Debug, Clone, Serialize)]
pub struct McpResourcesUpdatedPayload {
    pub server_name: String,
}

/// 资源内容更新事件
#[derive(Debug, Clone, Serialize)]
pub struct McpResourceUpdatedPayload {
    pub server_name: String,
    pub uri: String,
}
