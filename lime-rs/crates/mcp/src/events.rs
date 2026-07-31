//! MCP 事件 Payload。

use crate::types::{McpServerCapabilities, McpToolDefinition};
use serde::Serialize;

/// 服务器启动事件
#[derive(Debug, Clone, Serialize)]
pub struct McpServerStartedPayload {
    pub server_name: String,
    pub server_info: Option<McpServerCapabilities>,
}

/// 服务器停止事件
#[derive(Debug, Clone, Serialize)]
pub struct McpServerStoppedPayload {
    pub server_name: String,
}

/// 服务器错误事件
#[derive(Debug, Clone, Serialize)]
pub struct McpServerErrorPayload {
    pub server_name: String,
    pub error: String,
}

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
