use async_trait::async_trait;
use rmcp::model::{JsonObject, ListToolsResult};
use tokio_util::sync::CancellationToken;

mod registry;
mod scope;
mod step_snapshot;

pub use registry::{McpConnectionCall, McpConnectionHandle, McpConnectionRegistry};
pub use scope::McpCallScope;
pub use step_snapshot::{McpStepRouteIdentity, McpStepSnapshot};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct McpConnectionProvenance {
    environment_id: String,
    auth_scopes: Option<Vec<String>>,
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
        }
    }

    pub fn environment_id(&self) -> &str {
        &self.environment_id
    }

    pub fn auth_scopes(&self) -> Option<&[String]> {
        self.auth_scopes.as_deref()
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
