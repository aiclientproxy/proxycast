use super::{McpConnection, McpConnectionProvenance, McpStepSnapshot};
use crate::tool_extension::{RuntimeExtensionConfig, RuntimeToolCaller};
use futures::Stream;
use rmcp::model::{CallToolResult, ErrorData, ServerNotification, Tool};
use rmcp::service::ServiceError;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

pub type McpConnectionHandle = Arc<Mutex<Box<dyn McpConnection>>>;

pub struct McpConnectionCall {
    pub response: Pin<Box<dyn Future<Output = Result<CallToolResult, ErrorData>> + Send + 'static>>,
    pub notifications: Pin<Box<dyn Stream<Item = ServerNotification> + Send + 'static>>,
}

struct McpConnectionEntry {
    config: RuntimeExtensionConfig,
    provenance: McpConnectionProvenance,
    supports_parallel_tool_calls: bool,
    connection: McpConnectionHandle,
}

impl McpConnectionEntry {
    fn new(
        config: RuntimeExtensionConfig,
        provenance: McpConnectionProvenance,
        supports_parallel_tool_calls: bool,
        connection: McpConnectionHandle,
    ) -> Self {
        Self {
            config,
            provenance,
            supports_parallel_tool_calls,
            connection,
        }
    }
}

const DEFAULT_TOOL_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Default)]
pub struct McpConnectionRegistry {
    connections: Mutex<HashMap<String, McpConnectionEntry>>,
    generation: AtomicU64,
}

impl McpConnectionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn register(
        &self,
        name: String,
        config: RuntimeExtensionConfig,
        provenance: McpConnectionProvenance,
        supports_parallel_tool_calls: bool,
        connection: McpConnectionHandle,
    ) {
        let mut connections = self.connections.lock().await;
        connections.insert(
            name,
            McpConnectionEntry::new(config, provenance, supports_parallel_tool_calls, connection),
        );
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    pub async fn remove(&self, name: &str) -> bool {
        let mut connections = self.connections.lock().await;
        let removed = connections.remove(name).is_some();
        if removed {
            self.generation.fetch_add(1, Ordering::Relaxed);
        }
        removed
    }

    pub async fn inherit_from(&self, other: &Self) {
        let inherited = {
            let connections = other.connections.lock().await;
            connections
                .iter()
                .map(|(name, entry)| {
                    (
                        name.clone(),
                        McpConnectionEntry::new(
                            entry.config.clone(),
                            entry.provenance.clone(),
                            entry.supports_parallel_tool_calls,
                            Arc::clone(&entry.connection),
                        ),
                    )
                })
                .collect::<Vec<_>>()
        };
        if inherited.is_empty() {
            return;
        }
        let mut connections = self.connections.lock().await;
        connections.extend(inherited);
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    pub async fn names(&self) -> Vec<String> {
        let mut names = self
            .connections
            .lock()
            .await
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    pub async fn configs(&self) -> Vec<RuntimeExtensionConfig> {
        self.connections
            .lock()
            .await
            .values()
            .map(|entry| entry.config.clone())
            .collect()
    }

    pub async fn list_tools(
        &self,
        connection_name: Option<&str>,
    ) -> Result<Vec<Tool>, ServiceError> {
        Ok(self
            .step_snapshot(
                connection_name,
                RuntimeToolCaller::assistant(),
                HashSet::new(),
                DEFAULT_TOOL_DISCOVERY_TIMEOUT,
            )
            .await
            .tools()
            .to_vec())
    }

    pub async fn step_snapshot(
        &self,
        connection_name: Option<&str>,
        caller: RuntimeToolCaller,
        selected_deferred_tools: HashSet<String>,
        discovery_timeout: Duration,
    ) -> McpStepSnapshot {
        let (generation, connections) = {
            let entries = self.connections.lock().await;
            let connections = entries
                .iter()
                .filter(|(name, _)| connection_name.is_none_or(|filter| name.as_str() == filter))
                .map(|(name, entry)| {
                    (
                        name.clone(),
                        entry.config.clone(),
                        entry.provenance.clone(),
                        entry.supports_parallel_tool_calls,
                        Arc::clone(&entry.connection),
                    )
                })
                .collect::<Vec<_>>();
            (self.generation.load(Ordering::Relaxed), connections)
        };
        McpStepSnapshot::capture(
            connections,
            selected_deferred_tools,
            caller,
            discovery_timeout,
            generation,
        )
        .await
    }
}
