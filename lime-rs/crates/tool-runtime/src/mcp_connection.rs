use async_trait::async_trait;
use rmcp::model::{JsonObject, ListToolsResult};
use tokio_util::sync::CancellationToken;

mod registry;
mod scope;
mod step_snapshot;

pub use registry::{McpConnectionCall, McpConnectionHandle, McpConnectionRegistry};
pub use scope::McpCallScope;
pub use step_snapshot::{McpStepRouteAppContext, McpStepRouteIdentity, McpStepSnapshot};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct McpConnectionProvenance {
    environment_id: String,
    auth_scopes: Option<Vec<String>>,
    server_name: Option<String>,
    plugin_id: Option<String>,
}

impl Default for McpConnectionProvenance {
    fn default() -> Self {
        Self::new("local", None)
    }
}

impl McpConnectionProvenance {
    pub fn new(environment_id: impl Into<String>, auth_scopes: Option<Vec<String>>) -> Self {
        Self {
            environment_id: environment_id.into(),
            auth_scopes,
            server_name: None,
            plugin_id: None,
        }
    }

    pub fn with_server_name(mut self, server_name: Option<String>) -> Self {
        self.server_name = server_name
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        self
    }

    pub fn with_plugin_id(mut self, plugin_id: Option<String>) -> Self {
        self.plugin_id = plugin_id
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        self
    }

    pub fn environment_id(&self) -> &str {
        &self.environment_id
    }

    pub fn auth_scopes(&self) -> Option<&[String]> {
        self.auth_scopes.as_deref()
    }

    pub fn server_name(&self) -> Option<&str> {
        self.server_name.as_deref()
    }

    pub fn plugin_id(&self) -> Option<&str> {
        self.plugin_id.as_deref()
    }
}

pub type McpConnectionError = rmcp::ServiceError;

#[async_trait]
pub trait McpConnection: Send + Sync {
    async fn list_tools(
        &self,
        next_cursor: Option<String>,
        cancel_token: CancellationToken,
    ) -> Result<ListToolsResult, McpConnectionError>;

    async fn start_call_tool(
        &self,
        name: &str,
        arguments: Option<JsonObject>,
        scope: &McpCallScope,
        cancel_token: CancellationToken,
    ) -> Result<McpConnectionCall, McpConnectionError>;
}
