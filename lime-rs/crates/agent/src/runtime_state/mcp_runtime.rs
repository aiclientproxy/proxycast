use crate::mcp_bridge::McpBridgeRuntimeRegistry;
use futures::future::join_all;
use lime_mcp::{ElicitationRequestRouter, McpClientManager, McpRuntimeServerSpec};
use std::sync::Arc;
use tokio::sync::broadcast;
use tool_runtime::mcp_connection::McpConnectionRegistry;

pub(crate) struct McpThreadRuntime {
    session_id: String,
    thread_id: String,
    manager: Arc<McpClientManager>,
    connections: McpConnectionRegistry,
    bridge_registry: McpBridgeRuntimeRegistry,
    elicitation_router: ElicitationRequestRouter,
    server_specs: Vec<McpRuntimeServerSpec>,
}

impl McpThreadRuntime {
    pub(crate) fn new(
        session_id: String,
        thread_id: String,
        elicitation_router: ElicitationRequestRouter,
        server_specs: Vec<McpRuntimeServerSpec>,
    ) -> Self {
        Self {
            manager: Arc::new(McpClientManager::new_runtime(
                None,
                elicitation_router.clone(),
                session_id.clone(),
                thread_id.clone(),
            )),
            session_id,
            thread_id,
            connections: McpConnectionRegistry::new(),
            bridge_registry: McpBridgeRuntimeRegistry::new(),
            elicitation_router,
            server_specs,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(session_id: &str, thread_id: &str) -> Self {
        Self::new(
            session_id.to_string(),
            thread_id.to_string(),
            ElicitationRequestRouter::default(),
            Vec::new(),
        )
    }

    pub(crate) fn thread_id(&self) -> &str {
        &self.thread_id
    }

    pub(crate) fn connections(&self) -> &McpConnectionRegistry {
        &self.connections
    }

    pub(crate) fn server_specs(&self) -> Vec<McpRuntimeServerSpec> {
        self.server_specs.clone()
    }

    pub(crate) fn subscribe_server_notifications(
        &self,
    ) -> broadcast::Receiver<lime_mcp::McpServerNotification> {
        self.manager.subscribe_server_notifications()
    }

    pub(crate) async fn open_event_stream(
        &self,
        server_name: &str,
        name: &str,
        arguments: serde_json::Value,
        meta: Option<serde_json::Value>,
    ) -> Result<lime_mcp::McpEventStream, String> {
        self.manager
            .open_event_stream(server_name, name, arguments, meta)
            .await
            .map_err(|error| error.to_string())
    }

    pub(crate) async fn has_server(&self, server_name: &str) -> bool {
        self.manager.has_client(server_name).await
    }

    pub(crate) async fn read_resource(
        &self,
        server_name: &str,
        uri: &str,
    ) -> Result<lime_mcp::McpResourceContent, String> {
        self.read_resource_with_meta(server_name, uri, None).await
    }

    pub(crate) async fn read_resource_with_meta(
        &self,
        server_name: &str,
        uri: &str,
        meta: Option<serde_json::Value>,
    ) -> Result<lime_mcp::McpResourceContent, String> {
        self.manager
            .read_resource_with_meta(server_name, uri, meta)
            .await
            .map_err(|error| error.to_string())
    }

    pub(crate) async fn read_resource_for_origin(
        &self,
        server_name: &str,
        thread_id: &str,
        origin: &lime_mcp::McpResourceOrigin,
    ) -> Result<lime_mcp::McpResourceContent, String> {
        self.manager
            .read_resource_for_origin(server_name, thread_id, origin)
            .await
            .map_err(|error| error.to_string())
    }

    pub(crate) async fn list_resource_page(
        &self,
        server_name: &str,
        cursor: Option<String>,
    ) -> Result<lime_mcp::McpResourcePage, String> {
        self.manager
            .list_resource_page(server_name, cursor)
            .await
            .map_err(|error| error.to_string())
    }

    pub(crate) async fn list_resources(
        &self,
    ) -> Result<
        (
            Vec<lime_mcp::McpResourceDefinition>,
            Vec<lime_mcp::McpResourceTemplateDefinition>,
        ),
        String,
    > {
        let resources = self
            .manager
            .list_resources()
            .await
            .map_err(|error| error.to_string())?;
        let templates = self
            .manager
            .list_resource_templates()
            .await
            .map_err(|error| error.to_string())?;
        Ok((resources, templates))
    }

    pub(crate) async fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<lime_mcp::McpToolResult, String> {
        self.manager
            .call_tool(&format!("mcp__{server_name}__{tool_name}"), arguments)
            .await
            .map_err(|error| error.to_string())
    }

    pub(crate) async fn start(&self) -> Result<(), String> {
        let mut unavailable_servers = Vec::new();
        let mut required_failures = Vec::new();
        let results = join_all(self.server_specs.iter().cloned().map(|spec| {
            let manager = Arc::clone(&self.manager);
            async move {
                let result = manager.start_server(&spec.name, &spec.config).await;
                (spec, result)
            }
        }))
        .await;

        for (spec, result) in results {
            if let Err(error) = result {
                unavailable_servers.push(spec.name.clone());
                tracing::warn!(
                    session_id = %self.session_id,
                    thread_id = %self.thread_id,
                    server_name = %spec.name,
                    %error,
                    "runtime MCP server is unavailable; excluding it from this generation"
                );
                if spec.config.required {
                    required_failures.push(format!("{}: {error}", spec.name));
                }
            }
        }
        unavailable_servers.sort();
        required_failures.sort();
        if !required_failures.is_empty() {
            self.manager.shutdown().await;
            return Err(format!(
                "required MCP runtime server startup failed: {}",
                required_failures.join("; ")
            ));
        }
        let mut snapshots = self
            .manager
            .bridge_snapshots()
            .await
            .map_err(|error| error.to_string())?;
        for snapshot in &mut snapshots {
            snapshot.plugin_id = self
                .server_specs
                .iter()
                .find(|spec| spec.name == snapshot.server_name)
                .and_then(|spec| spec.plugin_id.clone());
        }
        let bridge_count = self
            .bridge_registry
            .sync(&self.connections, snapshots)
            .await;
        tracing::info!(
            session_id = %self.session_id,
            thread_id = %self.thread_id,
            configured_server_count = self.server_specs.len(),
            unavailable_servers = ?unavailable_servers,
            bridge_count,
            "runtime MCP generation published"
        );
        Ok(())
    }

    pub(crate) async fn shutdown(&self) {
        self.elicitation_router.cancel_session(&self.session_id);
        self.manager.shutdown().await;
    }
}
