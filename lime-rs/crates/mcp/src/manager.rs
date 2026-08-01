//! MCP 客户端管理器
//!
//! 本模块提供 MCP 客户端的集中管理，包括：
//! - 服务器生命周期管理（启动、停止、重启）
//! - 客户端连接池管理
//! - 工具定义缓存
//! - 事件发送
//!
//! # 架构设计
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                  McpClientManager                        │
//! │  ┌─────────────────────────────────────────────────┐   │
//! │  │           clients (连接池)                        │   │
//! │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │   │
//! │  │  │ Client1 │  │ Client2 │  │ Client3 │         │   │
//! │  │  └─────────┘  └─────────┘  └─────────┘         │   │
//! │  └─────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────┐   │
//! │  │           tool_cache (工具缓存)                   │   │
//! │  │  缓存所有运行中服务器的工具定义                      │   │
//! │  └─────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────┘
//! ```

#![allow(dead_code)]

use lime_core::DynEmitter;
use rmcp::service::RunningService;
use rmcp::RoleClient;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::client::McpClientWrapper;
use crate::environment::McpEnvironmentRegistry;
use crate::events::*;
use crate::oauth::{McpOAuthLoginHandle, McpOAuthRegistry};
use crate::types::*;

mod lifecycle;
mod prompts;
mod resources;
#[cfg(test)]
mod tests;
mod tools;

/// MCP 客户端管理器
///
/// 负责管理所有 MCP 服务器的连接和生命周期。
///
/// # 功能
///
/// - **连接池管理**: 维护所有运行中的 MCP 客户端连接
/// - **工具缓存**: 缓存工具定义以避免重复查询
/// - **事件通知**: 通过 DynEmitter 通知前端状态变化
///
/// # 线程安全
///
/// 所有内部状态都使用 `Arc<RwLock<_>>` 包装，支持并发访问。
///
/// # 示例
///
/// ```rust,ignore
/// let manager = McpClientManager::new(Some(app_handle));
///
/// // 启动服务器
/// manager.start_server("my-server", &config).await?;
///
/// // 获取工具列表
/// let tools = manager.list_tools().await?;
///
/// // 调用工具
/// let result = manager.call_tool("my-tool", args).await?;
///
/// // 停止服务器
/// manager.stop_server("my-server").await?;
/// ```
pub struct McpClientManager {
    /// 运行中的客户端 (server_name -> client)
    ///
    /// 使用 HashMap 存储所有活跃的 MCP 客户端连接。
    /// 键为服务器名称，值为客户端包装器。
    clients: Arc<RwLock<HashMap<String, McpClientWrapper>>>,

    /// 工具定义缓存
    ///
    /// 缓存所有运行中服务器的工具定义。
    /// 当服务器启动或停止时，缓存会被失效。
    /// 使用 Option 表示缓存状态：
    /// - None: 缓存无效，需要重新获取
    /// - Some(tools): 缓存有效
    tool_cache: Arc<RwLock<Option<Vec<McpToolDefinition>>>>,

    /// 事件发射器
    ///
    /// 用于向前端发送 MCP 相关事件，如：
    /// - mcp:tools_updated
    /// - mcp:resources_updated
    /// - mcp:resource_updated
    emitter: Option<DynEmitter>,

    oauth_registry: McpOAuthRegistry,

    environment_registry: McpEnvironmentRegistry,

    elicitation_router: Option<crate::elicitation::ElicitationRequestRouter>,
    runtime_owner: Option<McpRuntimeOwner>,
}

/// Immutable owner bound when a runtime MCP generation is created.
///
/// It never crosses the App Server reverse-request protocol. The public
/// request exposes only its `thread_id` and optional per-call `turn_id`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct McpRuntimeOwner {
    pub session_id: String,
    pub thread_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpRuntimeServerSpec {
    pub name: String,
    pub config: McpServerConfig,
}

/// 当前运行中的 MCP bridge 快照。
///
/// 该类型只服务已绑定 owner 的 App Server runtime generation。
/// 每个 `McpThreadRuntime` 自己持有 manager 与 `RunningService`，快照只导出
/// 该 generation 的不可变 connection handle，不会复用管理控制面连接。
#[derive(Clone)]
pub struct McpBridgeSnapshot {
    pub server_name: String,
    pub environment_id: String,
    pub description: String,
    pub tools: Vec<McpToolDefinition>,
    pub auth_scopes: Option<Vec<String>>,
    pub supports_parallel_tool_calls: bool,
    pub running_service:
        Arc<RunningService<RoleClient, crate::client_service::LimeMcpClientService>>,
    pub manager: Arc<McpClientManager>,
    pub tool_timeout: Duration,
}

impl McpClientManager {
    /// 创建新的管理器实例
    ///
    /// # Arguments
    ///
    /// * `emitter` - 事件发射器，用于发送事件到前端。
    ///   如果为 None，则不会发送事件。
    ///
    /// # Returns
    ///
    /// 返回初始化的 McpClientManager 实例，连接池和缓存均为空。
    pub fn new(emitter: Option<DynEmitter>) -> Self {
        info!("创建 MCP 客户端管理器");
        Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
            tool_cache: Arc::new(RwLock::new(None)),
            emitter,
            oauth_registry: McpOAuthRegistry::new(),
            environment_registry: McpEnvironmentRegistry::local(),
            elicitation_router: None,
            runtime_owner: None,
        }
    }

    pub fn new_runtime(
        emitter: Option<DynEmitter>,
        elicitation_router: crate::elicitation::ElicitationRequestRouter,
        session_id: impl Into<String>,
        thread_id: impl Into<String>,
    ) -> Self {
        let session_id = session_id.into();
        let thread_id = thread_id.into();
        assert!(
            !session_id.trim().is_empty(),
            "runtime MCP session id is required"
        );
        assert!(
            !thread_id.trim().is_empty(),
            "runtime MCP thread id is required"
        );
        Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
            tool_cache: Arc::new(RwLock::new(None)),
            emitter,
            oauth_registry: McpOAuthRegistry::new(),
            environment_registry: McpEnvironmentRegistry::local(),
            elicitation_router: Some(elicitation_router),
            runtime_owner: Some(McpRuntimeOwner {
                session_id,
                thread_id,
            }),
        }
    }

    /// 注入 OAuth 凭据根。未注入时凭据操作 fail closed，不解析平台默认目录。
    pub fn with_oauth_store_root(mut self, root_dir: impl Into<std::path::PathBuf>) -> Self {
        self.oauth_registry = McpOAuthRegistry::new_in(root_dir);
        self
    }

    /// 设置事件发射器
    pub fn set_emitter(&mut self, emitter: DynEmitter) {
        self.emitter = Some(emitter);
    }

    // ========================================================================
    // 连接池管理方法
    // ========================================================================

    /// 获取客户端连接池的只读引用
    ///
    /// 用于需要遍历所有客户端的场景。
    pub fn clients(&self) -> Arc<RwLock<HashMap<String, McpClientWrapper>>> {
        self.clients.clone()
    }

    /// 获取指定服务器的客户端（检查是否存在）
    ///
    /// # Arguments
    ///
    /// * `name` - 服务器名称
    ///
    /// # Returns
    ///
    /// 如果服务器正在运行，返回 true；否则返回 false。
    ///
    /// 注意：由于 McpClientWrapper 包含不可克隆的字段（如 tokio::process::Child），
    /// 我们不能直接返回客户端的克隆。如需操作客户端，请使用 clients() 获取连接池引用。
    pub async fn has_client(&self, name: &str) -> bool {
        let clients = self.clients.read().await;
        clients.contains_key(name)
    }

    /// 获取指定服务器的配置（如果存在）
    ///
    /// # Arguments
    ///
    /// * `name` - 服务器名称
    ///
    /// # Returns
    ///
    /// 如果服务器正在运行，返回 Some(配置的克隆)；
    /// 否则返回 None。
    pub async fn get_client_config(&self, name: &str) -> Option<McpServerConfig> {
        let clients = self.clients.read().await;
        clients.get(name).map(|c| c.config.clone())
    }

    /// 获取指定服务器的能力信息（如果存在）
    ///
    /// # Arguments
    ///
    /// * `name` - 服务器名称
    ///
    /// # Returns
    ///
    /// 如果服务器正在运行且有能力信息，返回 Some(能力信息的克隆)；
    /// 否则返回 None。
    pub async fn get_client_capabilities(&self, name: &str) -> Option<McpServerCapabilities> {
        let clients = self.clients.read().await;
        clients.get(name).and_then(|c| c.server_info.clone())
    }

    /// 添加客户端到连接池
    ///
    /// # Arguments
    ///
    /// * `name` - 服务器名称
    /// * `client` - 客户端包装器
    ///
    /// # Returns
    ///
    /// 如果服务器已存在，返回错误；否则添加成功。
    pub async fn add_client(&self, name: String, client: McpClientWrapper) -> Result<(), McpError> {
        let mut clients = self.clients.write().await;
        if clients.contains_key(&name) {
            return Err(McpError::ServerAlreadyRunning(name));
        }
        debug!(server_name = %name, "添加客户端到连接池");
        clients.insert(name, client);
        Ok(())
    }

    /// 从连接池移除客户端
    ///
    /// # Arguments
    ///
    /// * `name` - 服务器名称
    ///
    /// # Returns
    ///
    /// 如果服务器存在，返回移除的客户端包装器；
    /// 否则返回 None。
    pub async fn remove_client(&self, name: &str) -> Option<McpClientWrapper> {
        let mut clients = self.clients.write().await;
        let removed = clients.remove(name);
        if removed.is_some() {
            debug!(server_name = %name, "从连接池移除客户端");
        }
        removed
    }

    /// 获取所有运行中的服务器名称
    ///
    /// # Returns
    ///
    /// 返回所有运行中服务器的名称列表。
    pub async fn get_running_servers(&self) -> Vec<String> {
        let clients = self.clients.read().await;
        clients.keys().cloned().collect()
    }

    pub async fn get_server_runtime_status(
        &self,
        name: &str,
        fallback_config: Option<&McpServerConfig>,
    ) -> McpServerRuntimeStatus {
        let clients = self.clients.read().await;
        let running_snapshot = clients
            .get(name)
            .map(|wrapper| (wrapper.config.clone(), wrapper.server_info.clone()));
        drop(clients);

        if let Some((config, server_info)) = running_snapshot {
            let has_credentials = self
                .oauth_registry
                .has_credentials(name, &config)
                .await
                .unwrap_or(false);
            return Self::runtime_status_from_config(
                name,
                &config,
                true,
                server_info,
                has_credentials,
            );
        }

        let config = fallback_config.cloned().unwrap_or(McpServerConfig {
            transport: McpServerTransport::default(),
            environment_id: crate::DEFAULT_MCP_SERVER_ENVIRONMENT_ID.to_string(),
            enabled: false,
            startup_timeout: 30,
            tool_timeout: None,
            enabled_tools: None,
            disabled_tools: Vec::new(),
            required: false,
            supports_parallel_tool_calls: false,
            scopes: None,
            oauth: None,
            oauth_resource: None,
        });
        let has_credentials = self
            .oauth_registry
            .has_credentials(name, &config)
            .await
            .unwrap_or(false);
        Self::runtime_status_from_config(name, &config, false, None, has_credentials)
    }

    pub async fn start_oauth_login(
        &self,
        name: &str,
        config: &McpServerConfig,
        scopes: Option<Vec<String>>,
        timeout_secs: Option<u64>,
    ) -> Result<McpOAuthLoginHandle, McpError> {
        self.oauth_registry
            .start_login(name, config, scopes, timeout_secs)
            .await
    }

    fn runtime_status_from_config(
        name: &str,
        config: &McpServerConfig,
        is_running: bool,
        server_info: Option<McpServerCapabilities>,
        has_credentials: bool,
    ) -> McpServerRuntimeStatus {
        McpServerRuntimeStatus {
            name: name.to_string(),
            transport: config.transport_kind().to_string(),
            enabled: config.enabled,
            is_running,
            required: config.required,
            supports_parallel_tool_calls: config.supports_parallel_tool_calls,
            startup_timeout: config.startup_timeout_secs(),
            tool_timeout: config.tool_timeout_secs(),
            enabled_tools: config.enabled_tools.clone(),
            disabled_tools: config.disabled_tools.clone(),
            server_info,
            auth_status: config.auth_status_with_credentials(has_credentials),
        }
    }

    /// 获取运行中的服务器数量
    pub async fn running_server_count(&self) -> usize {
        let clients = self.clients.read().await;
        clients.len()
    }

    pub async fn bridge_snapshots(self: &Arc<Self>) -> Result<Vec<McpBridgeSnapshot>, McpError> {
        self.runtime_owner.as_ref().ok_or_else(|| {
            McpError::ConfigError(
                "management MCP connections cannot become runtime bridges".to_string(),
            )
        })?;
        let tools = self.list_tools().await?;
        let mut tools_by_server: HashMap<String, Vec<McpToolDefinition>> = HashMap::new();
        for tool in tools {
            tools_by_server
                .entry(tool.server_name.clone())
                .or_default()
                .push(tool);
        }

        let clients = self.clients.read().await;
        let mut snapshots = Vec::new();
        for (server_name, wrapper) in clients.iter() {
            let Some(running_service) = wrapper.running_service_arc() else {
                continue;
            };
            let mut server_tools = tools_by_server.remove(server_name).unwrap_or_default();
            if server_tools.is_empty() {
                continue;
            }
            server_tools.sort_by(|left, right| left.name.cmp(&right.name));

            let server_info = running_service.peer_info().cloned();
            let description = server_info
                .as_ref()
                .and_then(|info| info.instructions.clone())
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| format!("MCP server {server_name} tools"));

            snapshots.push(McpBridgeSnapshot {
                server_name: server_name.clone(),
                environment_id: wrapper.config.environment_id.clone(),
                description,
                tools: server_tools,
                auth_scopes: wrapper.config.oauth_scopes(),
                supports_parallel_tool_calls: wrapper.config.supports_parallel_tool_calls,
                running_service,
                manager: Arc::clone(self),
                tool_timeout: bridge_tool_timeout(&wrapper.config),
            });
        }
        snapshots.sort_by(|left, right| left.server_name.cmp(&right.server_name));
        Ok(snapshots)
    }

    pub async fn shutdown(&self) {
        let names = self.get_running_servers().await;
        for name in names {
            if let Err(error) = self.stop_server(&name).await {
                tracing::warn!(server_name = %name, %error, "failed to shut down runtime MCP server");
            }
        }
    }

    // ========================================================================
    // 缓存管理方法
    // ========================================================================

    /// 失效工具缓存
    ///
    /// 当服务器启动或停止时调用此方法，
    /// 确保下次获取工具列表时会重新查询所有服务器。
    pub async fn invalidate_tool_cache(&self) {
        let mut cache = self.tool_cache.write().await;
        if cache.is_some() {
            debug!("失效工具缓存");
        }
        *cache = None;
    }

    /// 检查工具缓存是否有效
    pub async fn is_tool_cache_valid(&self) -> bool {
        let cache = self.tool_cache.read().await;
        cache.is_some()
    }

    /// 获取缓存的工具列表（如果有效）
    ///
    /// # Returns
    ///
    /// 如果缓存有效，返回 Some(工具列表)；
    /// 否则返回 None。
    pub async fn get_cached_tools(&self) -> Option<Vec<McpToolDefinition>> {
        let cache = self.tool_cache.read().await;
        cache.clone()
    }

    /// 更新工具缓存
    ///
    /// # Arguments
    ///
    /// * `tools` - 新的工具列表
    pub async fn update_tool_cache(&self, tools: Vec<McpToolDefinition>) {
        let mut cache = self.tool_cache.write().await;
        debug!(tool_count = tools.len(), "更新工具缓存");
        *cache = Some(tools);
    }

    // ========================================================================
    // 事件发送方法
    // ========================================================================

    /// 发送事件到前端
    ///
    /// # Arguments
    ///
    /// * `event` - 事件名称
    /// * `payload` - 事件数据
    pub fn emit_event<T: serde::Serialize + Clone>(&self, event: &str, payload: T) {
        if let Some(ref emitter) = self.emitter {
            if let Ok(value) = serde_json::to_value(&payload) {
                if let Err(e) = emitter.emit_event(event, &value) {
                    warn!(
                        event = %event,
                        error = %e,
                        "发送事件失败"
                    );
                } else {
                    debug!(event = %event, "发送事件");
                }
            }
        }
    }

    /// 发送工具列表更新事件
    pub fn emit_tools_updated(&self, tools: Vec<McpToolDefinition>) {
        debug!(tool_count = tools.len(), "工具列表已更新");
        self.emit_event("mcp:tools_updated", McpToolsUpdatedPayload { tools });
    }
}

fn bridge_tool_timeout(config: &McpServerConfig) -> Duration {
    Duration::from_secs(config.tool_timeout_secs())
}

/// MCP 管理器共享状态包装器
pub type McpManagerState = Arc<tokio::sync::Mutex<McpClientManager>>;

/// 创建 MCP 管理器状态
pub fn create_mcp_manager_state(emitter: Option<DynEmitter>) -> McpManagerState {
    Arc::new(tokio::sync::Mutex::new(McpClientManager::new(emitter)))
}
