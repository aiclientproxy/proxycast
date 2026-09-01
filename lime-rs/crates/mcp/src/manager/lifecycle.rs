use super::McpClientManager;
use crate::environment::McpEnvironment;
use crate::naming::validate_server_name;
use crate::stdio_launcher::{LocalStdioLauncher, StdioLaunch};
use crate::streamable_http::{
    build_oauth_streamable_http_transport, build_streamable_http_transport,
};
use crate::types::{McpError, McpServerCapabilities, McpServerConfig, McpServerTransport};
use rmcp::ServiceExt;
use std::time::Duration;
use tracing::{debug, error, info};

impl McpClientManager {
    // ========================================================================
    // 服务器生命周期管理方法
    // ========================================================================

    /// 启动 MCP 服务器
    ///
    /// # Arguments
    ///
    /// * `name` - 服务器名称
    /// * `config` - 服务器配置
    ///
    /// # Returns
    ///
    /// 成功返回 Ok(())，失败返回错误。
    ///
    /// # 实现步骤（Task 4.2）
    ///
    /// 1. 检查服务器是否已运行
    /// 2. 按配置启动 stdio 或 streamable HTTP transport
    /// 3. 建立 MCP 连接
    /// 4. 初始化 MCP 客户端
    /// 5. 失效工具缓存
    pub async fn start_server(&self, name: &str, config: &McpServerConfig) -> Result<(), McpError> {
        validate_server_name(name).map_err(McpError::ConfigError)?;
        config.validate().map_err(McpError::ConfigError)?;
        if !config.enabled {
            return Err(McpError::ConfigError(format!("MCP 服务器 '{name}' 已禁用")));
        }
        let environment = self
            .environment_registry
            .resolve_server_environment(name, config)
            .map_err(McpError::ConfigError)?;
        info!(server_name = %name, transport = %config.transport_kind(), "启动 MCP 服务器");

        // 1. 检查服务器是否已运行
        if self.is_server_running(name).await {
            return Err(McpError::ServerAlreadyRunning(name.to_string()));
        }

        match (&config.transport, environment) {
            (McpServerTransport::Stdio { .. }, McpEnvironment::Local) => {
                self.start_stdio_server(name, config).await
            }
            (McpServerTransport::StreamableHttp { .. }, McpEnvironment::Local) => {
                self.start_streamable_http_server(name, config).await
            }
        }
    }

    async fn start_stdio_server(
        &self,
        name: &str,
        config: &McpServerConfig,
    ) -> Result<(), McpError> {
        debug!(server_name = %name, command = %config.command(), "启动 stdio MCP transport");

        let StdioLaunch {
            transport,
            process: process_handle,
            mut stderr_task,
        } = match LocalStdioLauncher::launch(name, config) {
            Ok(launch) => launch,
            Err(e) => {
                let error_msg = format!("无法启动服务器进程: {e}");
                error!(server_name = %name, error = %e, "启动 MCP 服务器进程失败");
                return Err(McpError::ProcessSpawnFailed(error_msg));
            }
        };

        // 初始化 MCP 客户端
        let client_service = self.client_service(name);

        // 启动 deadline 由配置唯一决定；不为特定命令隐式延长超时。
        let timeout_secs = config.startup_timeout_secs();
        let connect_result = tokio::time::timeout(
            Duration::from_secs(timeout_secs),
            client_service.serve(transport),
        )
        .await;

        let running_service = match connect_result {
            Ok(Ok(service)) => service,
            Ok(Err(e)) => {
                process_handle.terminate();
                if let Some(task) = stderr_task.take() {
                    task.abort();
                }
                let error_msg = format!("MCP 连接失败: {e}");
                error!(server_name = %name, error = %e, "MCP 客户端初始化失败");
                return Err(McpError::ConnectionFailed(error_msg));
            }
            Err(_) => {
                process_handle.terminate();
                if let Some(task) = stderr_task.take() {
                    task.abort();
                }
                error!(server_name = %name, timeout = timeout_secs, "MCP 连接超时");
                return Err(McpError::Timeout);
            }
        };

        // 获取服务器信息
        let server_info = running_service
            .peer_info()
            .map(Self::server_capabilities_from_peer_info);

        // 创建客户端包装器
        let mut wrapper = crate::client::McpClientWrapper::new(
            name.to_string(),
            config.clone(),
            self.emitter.clone(),
        );
        if let Some(ref info) = server_info {
            wrapper.set_server_info(info.clone());
        }
        wrapper.set_stdio_lifecycle(process_handle, stderr_task);
        wrapper.set_running_service(running_service);

        // 添加到连接池
        self.add_client(name.to_string(), wrapper).await?;

        // 5. 失效工具缓存
        self.invalidate_tool_cache().await;

        info!(server_name = %name, "MCP 服务器启动成功");
        Ok(())
    }

    async fn start_streamable_http_server(
        &self,
        name: &str,
        config: &McpServerConfig,
    ) -> Result<(), McpError> {
        let McpServerTransport::StreamableHttp { url, .. } = &config.transport else {
            return Err(McpError::ConfigError(
                "streamable HTTP 启动收到非 HTTP 配置".to_string(),
            ));
        };

        if config.has_oauth_settings() {
            let auth_manager = self
                .oauth_registry
                .authorized_manager_for(name, config)
                .await?
                .ok_or_else(|| {
                    McpError::ConfigError(
                        "MCP OAuth credentials are missing; run mcpServer/oauth/login first"
                            .to_string(),
                    )
                })?;
            let transport = build_oauth_streamable_http_transport(config, auth_manager).await?;
            return self
                .start_streamable_http_transport(name, config, url, transport)
                .await;
        }

        let transport = build_streamable_http_transport(config)?;
        self.start_streamable_http_transport(name, config, url, transport)
            .await
    }

    async fn start_streamable_http_transport<T, E, A>(
        &self,
        name: &str,
        config: &McpServerConfig,
        url: &str,
        transport: T,
    ) -> Result<(), McpError>
    where
        T: rmcp::transport::IntoTransport<rmcp::RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let client_service = self.client_service(name);
        let timeout_secs = config.startup_timeout_secs();
        let connect_result = tokio::time::timeout(
            Duration::from_secs(timeout_secs),
            client_service.serve(transport),
        )
        .await;

        let running_service = match connect_result {
            Ok(Ok(service)) => service,
            Ok(Err(error)) => {
                let error_msg = format!("MCP streamable HTTP 连接失败: {error}");
                error!(server_name = %name, url = %url, error = %error, "MCP HTTP 客户端初始化失败");
                return Err(McpError::ConnectionFailed(error_msg));
            }
            Err(_) => {
                error!(server_name = %name, url = %url, timeout = timeout_secs, "MCP HTTP 连接超时");
                return Err(McpError::Timeout);
            }
        };

        let server_info = running_service
            .peer_info()
            .map(Self::server_capabilities_from_peer_info);
        let mut wrapper = crate::client::McpClientWrapper::new(
            name.to_string(),
            config.clone(),
            self.emitter.clone(),
        );
        if let Some(ref info) = server_info {
            wrapper.set_server_info(info.clone());
        }
        wrapper.set_running_service(running_service);
        self.add_client(name.to_string(), wrapper).await?;
        self.invalidate_tool_cache().await;
        info!(server_name = %name, url = %url, "MCP streamable HTTP 服务器启动成功");
        Ok(())
    }

    fn server_capabilities_from_peer_info(
        info: &rmcp::model::InitializeResult,
    ) -> McpServerCapabilities {
        McpServerCapabilities {
            name: info.server_info.name.clone(),
            version: info.server_info.version.clone(),
            supports_tools: info
                .capabilities
                .tools
                .as_ref()
                .map(|_| true)
                .unwrap_or(false),
            supports_prompts: info
                .capabilities
                .prompts
                .as_ref()
                .map(|_| true)
                .unwrap_or(false),
            supports_resources: info
                .capabilities
                .resources
                .as_ref()
                .map(|_| true)
                .unwrap_or(false),
        }
    }

    /// 停止 MCP 服务器
    ///
    /// # Arguments
    ///
    /// * `name` - 服务器名称
    ///
    /// # Returns
    ///
    /// 成功返回 Ok(())，失败返回错误。
    /// 如果服务器未运行，也返回 Ok()（幂等操作）。
    ///
    /// # 实现步骤（Task 4.2）
    ///
    /// 1. 检查服务器是否在运行
    /// 2. 终止子进程
    /// 3. 清理客户端连接
    /// 4. 失效工具缓存
    pub async fn stop_server(&self, name: &str) -> Result<(), McpError> {
        info!(server_name = %name, "停止 MCP 服务器");

        // 1. 检查服务器是否在运行
        if !self.is_server_running(name).await {
            debug!(server_name = %name, "服务器未运行，跳过停止操作");
            return Ok(()); // 幂等操作
        }

        // 2. 从连接池移除客户端
        let mut wrapper = match self.remove_client(name).await {
            Some(w) => w,
            None => {
                debug!(server_name = %name, "客户端已被移除");
                return Ok(());
            }
        };

        // 3. 取消 rmcp 服务并终止本地 stdio 进程组。
        wrapper.shutdown();
        debug!(server_name = %name, "已停止 MCP 服务");

        // 4. 失效工具缓存
        self.invalidate_tool_cache().await;

        info!(server_name = %name, "MCP 服务器已停止");
        Ok(())
    }

    /// 检查服务器是否在运行
    ///
    /// # Arguments
    ///
    /// * `name` - 服务器名称
    ///
    /// # Returns
    ///
    /// 如果服务器正在运行返回 true，否则返回 false。
    pub async fn is_server_running(&self, name: &str) -> bool {
        let clients = self.clients.read().await;
        clients.contains_key(name)
    }

    /// 重启 MCP 服务器
    ///
    /// 先停止服务器，然后重新启动。
    ///
    /// # Arguments
    ///
    /// * `name` - 服务器名称
    /// * `config` - 服务器配置
    ///
    /// # Returns
    ///
    /// 成功返回 Ok(())，失败返回错误。
    pub async fn restart_server(
        &self,
        name: &str,
        config: &McpServerConfig,
    ) -> Result<(), McpError> {
        // 先停止（忽略未运行的错误）
        let _ = self.stop_server(name).await;
        // 再启动
        self.start_server(name, config).await
    }
}

impl McpClientManager {
    fn client_service(&self, name: &str) -> crate::client_service::LimeMcpClientService {
        match (&self.elicitation_router, &self.runtime_owner) {
            (Some(router), Some(owner)) => {
                crate::client_service::LimeMcpClientService::with_runtime_elicitation_router_and_notifications(
                    name.to_string(),
                    self.emitter.clone(),
                    router.clone(),
                    owner.session_id.clone(),
                    owner.thread_id.clone(),
                    Some(self.server_notifications.clone()),
                )
            }
            _ => crate::client_service::LimeMcpClientService::new(
                name.to_string(),
                self.emitter.clone(),
            ),
        }
    }
}
