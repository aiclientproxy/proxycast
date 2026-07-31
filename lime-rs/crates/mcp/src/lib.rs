//! Lime MCP Crate
//!
//! MCP（Model Context Protocol）集成模块，提供 MCP 协议的客户端实现。
//! 使用 DynEmitter 进行事件发射，与具体桌面宿主解耦。

mod active_time;
pub mod auth_status;
pub mod bridge_client;
mod client;
mod client_service;
pub mod elicitation;
mod environment;
pub mod events;
pub mod extension_surface;
pub mod manager;
pub mod naming;
pub mod oauth;
pub mod oauth_store;
mod stdio_launcher;
mod stdio_process;
mod streamable_http;
pub mod tool_converter;
pub mod tool_policy;
pub mod types;

pub use auth_status::{McpServerAuthActionPlan, McpServerAuthStatus};
pub use bridge_client::McpBridgeClient;
pub use client_service::LimeMcpClientService;
pub use elicitation::{
    ClaimedElicitationResolution, ElicitationAction, ElicitationRequest, ElicitationRequestId,
    ElicitationRequestRouter, ElicitationResponse, ElicitationRouterError,
};
pub use events::{
    McpResourceUpdatedPayload, McpResourcesUpdatedPayload, McpServerErrorPayload,
    McpServerStartedPayload, McpServerStoppedPayload, McpToolsUpdatedPayload,
};
pub use extension_surface::{build_runtime_extension_surface, runtime_extension_name};
pub use manager::{McpBridgeSnapshot, McpClientManager, McpRuntimeOwner, McpRuntimeServerSpec};
pub use oauth::{McpOAuthLoginHandle, McpOAuthLoginParams, McpOAuthRegistry};
pub use tool_converter::ToolConverter;
pub use types::{
    McpContent, McpError, McpManagerState, McpPromptArgument, McpPromptDefinition,
    McpPromptMessage, McpPromptResult, McpResourceContent, McpResourceDefinition,
    McpServerCapabilities, McpServerConfig, McpServerInfo, McpServerRuntimeStatus,
    McpServerTransport, McpToolCall, McpToolDefinition, McpToolResult,
    DEFAULT_MCP_SERVER_ENVIRONMENT_ID,
};
