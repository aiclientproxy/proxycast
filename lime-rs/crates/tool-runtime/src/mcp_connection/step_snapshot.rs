use super::{McpCallScope, McpConnectionCall, McpConnectionHandle, McpConnectionProvenance};
use crate::tool_extension::{RuntimeExtensionConfig, RuntimeToolCaller};
use lime_core::tool_calling::extract_tool_surface_metadata;
use rmcp::model::{CallToolRequestParam, ErrorCode, ErrorData, Tool};
use rmcp::service::ServiceError;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
struct McpStepRoute {
    server_name: String,
    tool_name: String,
    app_context: Option<McpStepRouteAppContext>,
    mcp_app_resource_uri: Option<String>,
    allowed_callers: Option<Vec<String>>,
    provenance: McpConnectionProvenance,
    supports_parallel_tool_calls: bool,
    connection: McpConnectionHandle,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct McpStepRouteAppContext {
    pub connector_id: String,
    pub link_id: Option<String>,
    pub resource_uri: Option<String>,
    pub app_name: Option<String>,
    pub action_name: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct McpStepRouteIdentity {
    pub server_name: String,
    pub tool_name: String,
    pub runtime_tool_name: String,
    pub app_context: Option<McpStepRouteAppContext>,
    pub mcp_app_resource_uri: Option<String>,
    pub plugin_id: Option<String>,
}

#[derive(Clone)]
pub struct McpStepSnapshot {
    caller: RuntimeToolCaller,
    generation: u64,
    tools: Arc<Vec<Tool>>,
    routes: Arc<HashMap<String, McpStepRoute>>,
}

impl McpStepSnapshot {
    pub fn empty(caller: RuntimeToolCaller) -> Self {
        Self {
            caller,
            generation: 0,
            tools: Arc::new(Vec::new()),
            routes: Arc::new(HashMap::new()),
        }
    }

    /// Generation of the connection registry captured for this sampling step.
    ///
    /// The snapshot remains immutable even when the registry changes; callers can use this
    /// value to correlate tool exposure and execution evidence with the captured registry state.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn tools(&self) -> &[Tool] {
        self.tools.as_ref()
    }

    pub fn supports_parallel_tool_calls(&self, tool_name: &str) -> bool {
        self.routes
            .get(tool_name)
            .is_some_and(|route| route.supports_parallel_tool_calls)
    }

    pub fn environment_id(&self, tool_name: &str) -> Option<&str> {
        self.routes
            .get(tool_name)
            .map(|route| route.provenance.environment_id())
    }

    pub fn route_identity(&self, runtime_tool_name: &str) -> Option<McpStepRouteIdentity> {
        self.routes
            .get(runtime_tool_name)
            .map(|route| McpStepRouteIdentity {
                server_name: route.server_name.clone(),
                tool_name: route.tool_name.clone(),
                runtime_tool_name: runtime_tool_name.to_string(),
                app_context: route.app_context.clone(),
                mcp_app_resource_uri: route.mcp_app_resource_uri.clone(),
                plugin_id: route.provenance.plugin_id().map(ToOwned::to_owned),
            })
    }

    /// 返回排除指定 MCP server 的不可变 step snapshot。
    pub fn without_server(&self, server_name: &str) -> Self {
        let routes = self
            .routes
            .iter()
            .filter(|(_, route)| route.server_name != server_name)
            .map(|(name, route)| (name.clone(), route.clone()))
            .collect::<HashMap<_, _>>();
        let tools = self
            .tools
            .iter()
            .filter(|tool| routes.contains_key(tool.name.as_ref()))
            .cloned()
            .collect::<Vec<_>>();
        Self {
            caller: self.caller.clone(),
            generation: self.generation,
            tools: Arc::new(tools),
            routes: Arc::new(routes),
        }
    }

    pub async fn dispatch(
        &self,
        tool_call: CallToolRequestParam,
        scope: McpCallScope,
        cancellation_token: CancellationToken,
    ) -> Result<McpConnectionCall, ErrorData> {
        let Some(route) = self.routes.get(tool_call.name.as_ref()).cloned() else {
            return Err(ErrorData::new(
                ErrorCode::RESOURCE_NOT_FOUND,
                tool_call.name,
                None,
            ));
        };
        if !self.caller.is_allowed(route.allowed_callers.as_deref()) {
            return Err(ErrorData::new(
                ErrorCode::RESOURCE_NOT_FOUND,
                tool_call.name,
                None,
            ));
        }

        let scope = scope
            .with_snapshot_generation(self.generation)
            .with_environment_id(route.provenance.environment_id().to_string())
            .with_auth_scopes(
                route
                    .provenance
                    .auth_scopes()
                    .map(ToOwned::to_owned)
                    .unwrap_or_default(),
            );
        let call = route
            .connection
            .lock()
            .await
            .start_call_tool(
                &route.tool_name,
                tool_call.arguments,
                &scope,
                cancellation_token,
            )
            .await
            .map_err(service_error_data);
        call
    }

    pub(super) async fn capture(
        connections: Vec<(
            String,
            RuntimeExtensionConfig,
            McpConnectionProvenance,
            bool,
            McpConnectionHandle,
        )>,
        selected_deferred_tools: HashSet<String>,
        caller: RuntimeToolCaller,
        discovery_timeout: Duration,
        generation: u64,
    ) -> Self {
        let mut tools = Vec::new();
        let mut routes = HashMap::new();
        let selected_deferred_tools = Arc::new(selected_deferred_tools);
        let mut pending = JoinSet::new();
        for (connection_name, config, provenance, supports_parallel_tool_calls, connection) in
            connections
        {
            let selected_deferred_tools = Arc::clone(&selected_deferred_tools);
            let caller = caller.clone();
            pending.spawn(async move {
                capture_connection(
                    connection_name,
                    config,
                    provenance,
                    supports_parallel_tool_calls,
                    connection,
                    selected_deferred_tools,
                    caller,
                    discovery_timeout,
                )
                .await
            });
        }
        while let Some(result) = pending.join_next().await {
            let Ok(Some(captured)) = result else {
                continue;
            };
            for (tool, route) in captured.entries {
                let prefixed_name = tool.name.to_string();
                if routes.insert(prefixed_name, route).is_none() {
                    tools.push(tool);
                }
            }
        }
        tools.sort_by(|left, right| left.name.cmp(&right.name));
        Self {
            caller,
            generation,
            tools: Arc::new(tools),
            routes: Arc::new(routes),
        }
    }
}

struct CapturedConnectionTools {
    entries: Vec<(Tool, McpStepRoute)>,
}

async fn capture_connection(
    connection_name: String,
    config: RuntimeExtensionConfig,
    provenance: McpConnectionProvenance,
    supports_parallel_tool_calls: bool,
    connection: McpConnectionHandle,
    selected_deferred_tools: Arc<HashSet<String>>,
    caller: RuntimeToolCaller,
    discovery_timeout: Duration,
) -> Option<CapturedConnectionTools> {
    let cancellation_token = CancellationToken::new();
    let discovery = async {
        let connection_guard = connection.lock().await;
        let mut page = connection_guard
            .list_tools(None, cancellation_token.clone())
            .await?;
        let mut entries = Vec::new();
        let mut seen = HashSet::new();
        loop {
            for tool in page.tools {
                let tool_name = tool.name.to_string();
                let mcp_app_resource_uri = mcp_app_resource_uri(&tool);
                let server_name = provenance
                    .server_name()
                    .unwrap_or(&connection_name)
                    .to_string();
                let app_context =
                    mcp_app_context(&server_name, &tool, mcp_app_resource_uri.clone());
                let prefixed_name = format!("{connection_name}__{tool_name}");
                let visible = config.is_tool_exposed_by_default(&tool.name)
                    || selected_deferred_tools.contains(&prefixed_name);
                let allowed_callers = tool_allowed_callers(&tool, &config);
                if !config.is_tool_available(&tool.name)
                    || !visible
                    || !caller.is_allowed(allowed_callers.as_deref())
                    || !seen.insert(prefixed_name.clone())
                {
                    continue;
                }
                let tool_supports_parallel = supports_parallel_tool_calls
                    || tool
                        .annotations
                        .as_ref()
                        .and_then(|annotations| annotations.read_only_hint)
                        .unwrap_or(false);
                entries.push((
                    Tool {
                        name: prefixed_name.into(),
                        description: tool.description,
                        input_schema: tool.input_schema,
                        annotations: tool.annotations,
                        output_schema: tool.output_schema,
                        icons: tool.icons,
                        title: tool.title,
                        meta: tool.meta,
                    },
                    McpStepRoute {
                        server_name,
                        tool_name,
                        app_context,
                        mcp_app_resource_uri,
                        allowed_callers,
                        provenance: provenance.clone(),
                        supports_parallel_tool_calls: tool_supports_parallel,
                        connection: Arc::clone(&connection),
                    },
                ));
            }
            let Some(cursor) = page.next_cursor else {
                break;
            };
            page = connection_guard
                .list_tools(Some(cursor), cancellation_token.clone())
                .await?;
        }
        Ok::<_, ServiceError>(CapturedConnectionTools { entries })
    };

    match tokio::time::timeout(discovery_timeout, discovery).await {
        Ok(Ok(captured)) => Some(captured),
        Ok(Err(_)) => None,
        Err(_) => {
            cancellation_token.cancel();
            None
        }
    }
}

fn mcp_app_resource_uri(tool: &Tool) -> Option<String> {
    let meta = tool.meta.as_ref()?;
    meta.get("ui")
        .and_then(Value::as_object)
        .and_then(|ui| ui.get("resourceUri"))
        .and_then(Value::as_str)
        .or_else(|| meta.get("ui/resourceUri").and_then(Value::as_str))
        .or_else(|| meta.get("openai/outputTemplate").and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn mcp_app_context(
    server_name: &str,
    tool: &Tool,
    resource_uri: Option<String>,
) -> Option<McpStepRouteAppContext> {
    if server_name != "codex_apps" {
        return None;
    }
    let meta = tool.meta.as_ref()?;
    let connector_id = non_empty_meta_string(meta, "connector_id")?;
    let codex_apps = meta.get("_codex_apps").and_then(Value::as_object);
    let action_name = codex_apps
        .and_then(|value| value.get("resource_uri"))
        .and_then(Value::as_str)
        .map(str::trim)
        .and_then(|value| value.trim_matches('/').rsplit('/').next())
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    Some(McpStepRouteAppContext {
        connector_id,
        link_id: non_empty_meta_string(meta, "link_id"),
        resource_uri,
        app_name: non_empty_meta_string(meta, "connector_name"),
        action_name,
    })
}

fn non_empty_meta_string(meta: &rmcp::model::Meta, key: &str) -> Option<String> {
    meta.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn tool_allowed_callers(tool: &Tool, config: &RuntimeExtensionConfig) -> Option<Vec<String>> {
    extract_tool_surface_metadata(
        tool.name.as_ref(),
        &Value::Object((*tool.input_schema).clone()),
    )
    .allowed_callers
    .or_else(|| {
        config
            .allowed_caller()
            .map(|caller| vec![caller.to_string()])
    })
}

fn service_error_data(error: ServiceError) -> ErrorData {
    match error {
        ServiceError::McpError(error) => error,
        error => ErrorData::new(ErrorCode::INTERNAL_ERROR, error.to_string(), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp_connection::{McpConnection, McpConnectionError, McpConnectionRegistry};
    use async_trait::async_trait;
    use rmcp::model::{
        CallToolResult, Content, JsonObject, ListToolsResult, Meta, ToolAnnotations,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Mutex;

    #[derive(Clone, Copy)]
    enum DiscoveryMode {
        Ready,
        Error,
        Slow,
    }

    struct TestConnection {
        tools: Vec<Tool>,
        output: String,
        discovery_mode: DiscoveryMode,
        call_count: Arc<AtomicUsize>,
        observed_scope: Option<Arc<Mutex<Option<McpCallScope>>>>,
    }

    #[async_trait]
    impl McpConnection for TestConnection {
        async fn list_tools(
            &self,
            _next_cursor: Option<String>,
            _cancel_token: CancellationToken,
        ) -> Result<ListToolsResult, McpConnectionError> {
            match self.discovery_mode {
                DiscoveryMode::Ready => Ok(ListToolsResult::with_all_items(self.tools.clone())),
                DiscoveryMode::Error => Err(ServiceError::UnexpectedResponse),
                DiscoveryMode::Slow => std::future::pending().await,
            }
        }

        async fn start_call_tool(
            &self,
            name: &str,
            _arguments: Option<JsonObject>,
            scope: &McpCallScope,
            _cancel_token: CancellationToken,
        ) -> Result<McpConnectionCall, McpConnectionError> {
            let observed_scope = self.observed_scope.clone();
            let scope = scope.clone();
            let call_count = Arc::clone(&self.call_count);
            let output = self.output.clone();
            let name = name.to_string();
            Ok(McpConnectionCall {
                response: Box::pin(async move {
                    if let Some(observed_scope) = observed_scope {
                        *observed_scope.lock().await = Some(scope);
                    }
                    call_count.fetch_add(1, Ordering::SeqCst);
                    Ok(CallToolResult::success(vec![Content::text(format!(
                        "{output}:{name}"
                    ))]))
                }),
                notifications: Box::pin(futures::stream::empty()),
            })
        }
    }

    #[tokio::test]
    async fn captured_step_keeps_old_route_after_registry_replace() {
        let registry = McpConnectionRegistry::new();
        register(&registry, vec!["search"], "old").await;
        let captured = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;

        registry.remove("docs").await;
        register(&registry, vec!["search", "new_tool"], "new").await;

        let old_result = captured
            .dispatch(call("docs__search"), scope(), CancellationToken::default())
            .await
            .expect("dispatch captured route")
            .response
            .await
            .expect("old result");
        let fresh = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;
        let new_result = fresh
            .dispatch(call("docs__search"), scope(), CancellationToken::default())
            .await
            .expect("dispatch fresh route")
            .response
            .await
            .expect("new result");

        assert!(fresh.generation() > captured.generation());
        assert!(result_text(&old_result).contains("old:search"));
        assert!(result_text(&new_result).contains("new:search"));
    }

    #[tokio::test]
    async fn snapshot_generation_is_stable_and_monotonic_across_registry_mutations() {
        let registry = McpConnectionRegistry::new();
        let empty = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;
        assert_eq!(empty.generation(), 0);

        register(&registry, vec!["search"], "first").await;
        let first = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;
        assert!(first.generation() > empty.generation());
        let first_generation = first.generation();

        registry.remove("docs").await;
        let removed = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;
        assert!(removed.generation() > first_generation);
        assert_eq!(first.generation(), first_generation);
    }

    #[tokio::test]
    async fn without_server_filters_tool_definitions_and_dispatch_routes() {
        let registry = McpConnectionRegistry::new();
        register_named(&registry, "codex_apps", DiscoveryMode::Ready).await;
        register_named(&registry, "docs", DiscoveryMode::Ready).await;
        let captured = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;

        let filtered = captured.without_server("codex_apps");

        assert!(filtered
            .tools()
            .iter()
            .any(|tool| tool.name == "docs__search"));
        assert!(!filtered
            .tools()
            .iter()
            .any(|tool| tool.name == "codex_apps__search"));
        assert!(filtered
            .dispatch(
                call("codex_apps__search"),
                scope(),
                CancellationToken::default(),
            )
            .await
            .is_err());
        assert!(filtered
            .dispatch(call("docs__search"), scope(), CancellationToken::default(),)
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn snapshot_keeps_server_route_metadata_immutable() {
        let registry = McpConnectionRegistry::new();
        let config = RuntimeExtensionConfig::new(
            "docs",
            "Docs tools",
            vec!["search".to_string()],
            false,
            vec!["search".to_string()],
            None,
        );
        let connection: McpConnectionHandle = Arc::new(Mutex::new(Box::new(TestConnection {
            tools: vec![tool("search", None)],
            output: "docs".to_string(),
            discovery_mode: DiscoveryMode::Ready,
            call_count: Arc::new(AtomicUsize::new(0)),
            observed_scope: None,
        })));
        registry
            .register(
                "docs".to_string(),
                config.clone(),
                McpConnectionProvenance::new("remote-tools", None),
                true,
                Arc::clone(&connection),
            )
            .await;

        let parallel = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;
        registry.remove("docs").await;
        registry
            .register(
                "docs".to_string(),
                config,
                McpConnectionProvenance::default(),
                false,
                connection,
            )
            .await;
        let serial = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;

        assert!(parallel.supports_parallel_tool_calls("docs__search"));
        assert!(!serial.supports_parallel_tool_calls("docs__search"));
        assert!(parallel.supports_parallel_tool_calls("docs__search"));
        assert!(!parallel.supports_parallel_tool_calls("missing"));
        assert_eq!(
            parallel.environment_id("docs__search"),
            Some("remote-tools")
        );
        assert_eq!(serial.environment_id("docs__search"), Some("local"));
        assert_eq!(
            parallel.environment_id("docs__search"),
            Some("remote-tools")
        );
        assert_eq!(parallel.environment_id("missing"), None);
        assert_eq!(
            parallel.route_identity("docs__search"),
            Some(McpStepRouteIdentity {
                server_name: "docs".to_string(),
                tool_name: "search".to_string(),
                runtime_tool_name: "docs__search".to_string(),
                app_context: None,
                mcp_app_resource_uri: None,
                plugin_id: None,
            })
        );
        assert_eq!(parallel.route_identity("missing"), None);
    }

    #[tokio::test]
    async fn snapshot_freezes_plugin_identity_and_mcp_app_resource_uri() {
        let registry = McpConnectionRegistry::new();
        let config = RuntimeExtensionConfig::new(
            "mcp__plugin__docs__demo",
            "Plugin docs tools",
            vec!["search".to_string()],
            false,
            vec!["search".to_string()],
            None,
        );
        let mut app_tool = tool("search", None);
        app_tool.meta = Some(Meta(
            serde_json::json!({
                "ui": { "resourceUri": "ui://plugin/docs.html" },
                "ui/resourceUri": "ui://plugin/flat.html",
                "openai/outputTemplate": "ui://plugin/fallback.html"
            })
            .as_object()
            .expect("tool meta object")
            .clone(),
        ));
        let connection: McpConnectionHandle = Arc::new(Mutex::new(Box::new(TestConnection {
            tools: vec![app_tool],
            output: "docs".to_string(),
            discovery_mode: DiscoveryMode::Ready,
            call_count: Arc::new(AtomicUsize::new(0)),
            observed_scope: None,
        })));
        registry
            .register(
                "mcp__plugin__docs__demo".to_string(),
                config,
                McpConnectionProvenance::default()
                    .with_server_name(Some("plugin__docs__demo".to_string()))
                    .with_plugin_id(Some("docs-plugin".to_string())),
                false,
                connection,
            )
            .await;

        let captured = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;
        registry.remove("mcp__plugin__docs__demo").await;

        assert_eq!(
            captured.route_identity("mcp__plugin__docs__demo__search"),
            Some(McpStepRouteIdentity {
                server_name: "plugin__docs__demo".to_string(),
                tool_name: "search".to_string(),
                runtime_tool_name: "mcp__plugin__docs__demo__search".to_string(),
                app_context: None,
                mcp_app_resource_uri: Some("ui://plugin/docs.html".to_string()),
                plugin_id: Some("docs-plugin".to_string()),
            })
        );
    }

    #[tokio::test]
    async fn read_only_tool_annotation_allows_parallel_execution_without_server_opt_in() {
        let registry = McpConnectionRegistry::new();
        let config = RuntimeExtensionConfig::new(
            "docs",
            "Docs tools",
            vec!["readonly".to_string(), "mutating".to_string()],
            false,
            vec!["readonly".to_string(), "mutating".to_string()],
            None,
        );
        let mut readonly = tool("readonly", None);
        readonly.annotations = Some(ToolAnnotations {
            read_only_hint: Some(true),
            ..ToolAnnotations::default()
        });
        let mut mutating = tool("mutating", None);
        mutating.annotations = Some(ToolAnnotations {
            read_only_hint: Some(false),
            ..ToolAnnotations::default()
        });
        register_connection(
            &registry,
            "docs",
            config,
            vec![readonly, mutating],
            "docs",
            DiscoveryMode::Ready,
            Arc::new(AtomicUsize::new(0)),
        )
        .await;

        let snapshot = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;
        assert!(snapshot.supports_parallel_tool_calls("docs__readonly"));
        assert!(!snapshot.supports_parallel_tool_calls("docs__mutating"));
    }

    #[tokio::test]
    async fn missing_read_only_tool_annotation_keeps_serial_default() {
        let registry = McpConnectionRegistry::new();
        register(&registry, vec!["search"], "docs").await;

        let snapshot = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;
        assert!(!snapshot.supports_parallel_tool_calls("docs__search"));
    }

    #[tokio::test]
    async fn dispatch_binds_snapshot_generation_to_call_scope() {
        let registry = McpConnectionRegistry::new();
        let observed_scope = Arc::new(Mutex::new(None));
        let config = RuntimeExtensionConfig::new(
            "docs",
            "Docs tools",
            vec!["search".to_string()],
            false,
            vec!["search".to_string()],
            None,
        );
        let connection: McpConnectionHandle = Arc::new(Mutex::new(Box::new(TestConnection {
            tools: vec![tool("search", None)],
            output: "docs".to_string(),
            discovery_mode: DiscoveryMode::Ready,
            call_count: Arc::new(AtomicUsize::new(0)),
            observed_scope: Some(Arc::clone(&observed_scope)),
        })));
        registry
            .register(
                "docs".to_string(),
                config,
                McpConnectionProvenance::new("local", Some(vec!["search.read".to_string()])),
                false,
                connection,
            )
            .await;

        let snapshot = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;
        let generation = snapshot.generation();
        snapshot
            .dispatch(call("docs__search"), scope(), CancellationToken::default())
            .await
            .expect("dispatch route")
            .response
            .await
            .expect("call result");

        assert_eq!(
            observed_scope
                .lock()
                .await
                .as_ref()
                .and_then(McpCallScope::snapshot_generation),
            Some(generation)
        );
        assert_eq!(
            observed_scope
                .lock()
                .await
                .as_ref()
                .and_then(McpCallScope::auth_scopes),
            Some(["search.read".to_string()].as_slice())
        );
        assert_eq!(
            observed_scope
                .lock()
                .await
                .as_ref()
                .and_then(McpCallScope::environment_id),
            Some("local")
        );
    }

    #[tokio::test]
    async fn tools_added_after_capture_are_visible_only_to_next_step() {
        let registry = McpConnectionRegistry::new();
        register(&registry, vec!["search"], "old").await;
        let captured = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;

        registry.remove("docs").await;
        register(&registry, vec!["search", "new_tool"], "new").await;
        let fresh = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;

        assert!(!captured
            .tools()
            .iter()
            .any(|tool| tool.name == "docs__new_tool"));
        assert!(captured
            .dispatch(
                call("docs__new_tool"),
                scope(),
                CancellationToken::default(),
            )
            .await
            .is_err());
        assert!(fresh
            .tools()
            .iter()
            .any(|tool| tool.name == "docs__new_tool"));
        assert!(fresh
            .dispatch(
                call("docs__new_tool"),
                scope(),
                CancellationToken::default(),
            )
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn deferred_selection_changes_only_the_next_snapshot() {
        let registry = McpConnectionRegistry::new();
        let config = RuntimeExtensionConfig::new(
            "docs",
            "Docs tools",
            vec!["search".to_string(), "query".to_string()],
            true,
            vec!["search".to_string()],
            None,
        );
        register_connection(
            &registry,
            "docs",
            config,
            vec![tool("search", None), tool("query", None)],
            "docs",
            DiscoveryMode::Ready,
            Arc::new(AtomicUsize::new(0)),
        )
        .await;

        let before = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;
        let selected = HashSet::from(["docs__query".to_string()]);
        let after = snapshot(&registry, selected, Duration::from_secs(1)).await;

        assert!(before
            .tools()
            .iter()
            .any(|tool| tool.name == "docs__search"));
        assert!(!before.tools().iter().any(|tool| tool.name == "docs__query"));
        assert!(before
            .dispatch(call("docs__query"), scope(), CancellationToken::default())
            .await
            .is_err());
        assert!(after.tools().iter().any(|tool| tool.name == "docs__query"));
        assert!(after
            .dispatch(call("docs__query"), scope(), CancellationToken::default())
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn assistant_snapshot_filters_and_rechecks_per_tool_caller_policy() {
        let registry = McpConnectionRegistry::new();
        let call_count = Arc::new(AtomicUsize::new(0));
        let config = RuntimeExtensionConfig::new(
            "mixed",
            "Mixed caller tools",
            vec![
                "public".to_string(),
                "review".to_string(),
                "execute".to_string(),
            ],
            false,
            Vec::new(),
            None,
        );
        let connection = register_connection(
            &registry,
            "mixed",
            config,
            vec![
                tool("public", None),
                tool("review", Some(&["assistant"])),
                tool("execute", Some(&["code_execution"])),
            ],
            "mixed",
            DiscoveryMode::Ready,
            call_count.clone(),
        )
        .await;

        let captured = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;
        let names = captured
            .tools()
            .iter()
            .map(|tool| tool.name.as_ref())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["mixed__public", "mixed__review"]);
        assert!(captured
            .dispatch(
                call("mixed__execute"),
                scope(),
                CancellationToken::default(),
            )
            .await
            .is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), 0);

        let denied_route = McpStepRoute {
            server_name: "mixed".to_string(),
            tool_name: "execute".to_string(),
            app_context: None,
            mcp_app_resource_uri: None,
            allowed_callers: Some(vec!["code_execution".to_string()]),
            provenance: McpConnectionProvenance::default(),
            supports_parallel_tool_calls: false,
            connection,
        };
        let forged_snapshot = McpStepSnapshot {
            caller: RuntimeToolCaller::assistant(),
            generation: 0,
            tools: Arc::new(Vec::new()),
            routes: Arc::new(HashMap::from([(
                "mixed__execute".to_string(),
                denied_route,
            )])),
        };
        assert!(forged_snapshot
            .dispatch(
                call("mixed__execute"),
                scope(),
                CancellationToken::default(),
            )
            .await
            .is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), 0);

        let code_execution = snapshot_for_caller(
            &registry,
            RuntimeToolCaller::parse("code_execution").expect("code execution caller"),
            HashSet::new(),
            Duration::from_secs(1),
        )
        .await;
        let code_names = code_execution
            .tools()
            .iter()
            .map(|tool| tool.name.as_ref())
            .collect::<Vec<_>>();
        assert_eq!(code_names, vec!["mixed__execute", "mixed__public"]);
        let code_result = code_execution
            .dispatch(
                call("mixed__execute"),
                scope(),
                CancellationToken::default(),
            )
            .await
            .expect("code execution route")
            .response
            .await
            .expect("code execution result");
        assert!(result_text(&code_result).contains("mixed:execute"));
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn codex_apps_route_context_comes_from_trusted_tool_metadata() {
        let mut tool = tool("calendar_search", None);
        tool.meta = Some(Meta(
            serde_json::json!({
                "connector_id": "calendar",
                "connector_name": "Calendar",
                "link_id": "link-calendar",
                "ui": { "resourceUri": "ui://calendar/event" },
                "_codex_apps": {
                    "resource_uri": "/calendar/link-calendar/calendar_search"
                }
            })
            .as_object()
            .expect("tool metadata")
            .clone(),
        ));

        assert_eq!(
            mcp_app_context("codex_apps", &tool, mcp_app_resource_uri(&tool),),
            Some(McpStepRouteAppContext {
                connector_id: "calendar".to_string(),
                link_id: Some("link-calendar".to_string()),
                resource_uri: Some("ui://calendar/event".to_string()),
                app_name: Some("Calendar".to_string()),
                action_name: Some("calendar_search".to_string()),
            })
        );
        assert_eq!(mcp_app_context("docs", &tool, None), None);
    }

    #[tokio::test]
    async fn per_tool_caller_schema_precedes_extension_fallback() {
        let registry = McpConnectionRegistry::new();
        let config = RuntimeExtensionConfig::new(
            "precedence",
            "Caller precedence",
            vec!["review".to_string()],
            false,
            Vec::new(),
            Some("code_execution".to_string()),
        );
        register_connection(
            &registry,
            "precedence",
            config,
            vec![tool("review", Some(&["assistant"]))],
            "precedence",
            DiscoveryMode::Ready,
            Arc::new(AtomicUsize::new(0)),
        )
        .await;

        let assistant = snapshot(&registry, HashSet::new(), Duration::from_secs(1)).await;
        assert!(assistant
            .tools()
            .iter()
            .any(|tool| tool.name == "precedence__review"));

        let code_execution = snapshot_for_caller(
            &registry,
            RuntimeToolCaller::parse("code_execution").expect("code execution caller"),
            HashSet::new(),
            Duration::from_secs(1),
        )
        .await;
        assert!(code_execution.tools().is_empty());
    }

    #[tokio::test]
    async fn discovery_error_keeps_healthy_server_tools() {
        let registry = McpConnectionRegistry::new();
        register_named(&registry, "fast", DiscoveryMode::Ready).await;
        register_named(&registry, "broken", DiscoveryMode::Error).await;

        let captured = snapshot(&registry, HashSet::new(), Duration::from_millis(50)).await;

        assert!(captured
            .tools()
            .iter()
            .any(|tool| tool.name == "fast__search"));
        assert!(!captured
            .tools()
            .iter()
            .any(|tool| tool.name == "broken__search"));
        assert!(captured
            .dispatch(call("fast__search"), scope(), CancellationToken::default())
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn discovery_timeout_keeps_concurrent_healthy_server_tools() {
        let registry = McpConnectionRegistry::new();
        register_named(&registry, "slow", DiscoveryMode::Slow).await;
        register_named(&registry, "fast", DiscoveryMode::Ready).await;

        let captured = tokio::time::timeout(
            Duration::from_millis(200),
            snapshot(&registry, HashSet::new(), Duration::from_millis(10)),
        )
        .await
        .expect("per-server timeout should keep snapshot capture bounded");

        assert!(captured
            .tools()
            .iter()
            .any(|tool| tool.name == "fast__search"));
        assert!(!captured
            .tools()
            .iter()
            .any(|tool| tool.name == "slow__search"));
        assert_eq!(registry.names().await, vec!["fast", "slow"]);
    }

    async fn register(registry: &McpConnectionRegistry, tools: Vec<&str>, output: &str) {
        let config = RuntimeExtensionConfig::new(
            "docs",
            "Docs tools",
            tools.iter().map(|name| (*name).to_string()).collect(),
            false,
            tools.iter().map(|name| (*name).to_string()).collect(),
            None,
        );
        register_connection(
            registry,
            "docs",
            config,
            tools.into_iter().map(|name| tool(name, None)).collect(),
            output,
            DiscoveryMode::Ready,
            Arc::new(AtomicUsize::new(0)),
        )
        .await;
    }

    async fn register_named(
        registry: &McpConnectionRegistry,
        name: &str,
        discovery_mode: DiscoveryMode,
    ) {
        let config = RuntimeExtensionConfig::new(
            name,
            format!("{name} tools"),
            vec!["search".to_string()],
            false,
            vec!["search".to_string()],
            None,
        );
        register_connection(
            registry,
            name,
            config,
            vec![tool("search", None)],
            name,
            discovery_mode,
            Arc::new(AtomicUsize::new(0)),
        )
        .await;
    }

    async fn register_connection(
        registry: &McpConnectionRegistry,
        name: &str,
        config: RuntimeExtensionConfig,
        tools: Vec<Tool>,
        output: &str,
        discovery_mode: DiscoveryMode,
        call_count: Arc<AtomicUsize>,
    ) -> McpConnectionHandle {
        let connection: McpConnectionHandle = Arc::new(Mutex::new(Box::new(TestConnection {
            tools,
            output: output.to_string(),
            discovery_mode,
            call_count,
            observed_scope: None,
        })));
        registry
            .register(
                name.to_string(),
                config,
                McpConnectionProvenance::default(),
                false,
                Arc::clone(&connection),
            )
            .await;
        connection
    }

    async fn snapshot(
        registry: &McpConnectionRegistry,
        selected_deferred_tools: HashSet<String>,
        discovery_timeout: Duration,
    ) -> McpStepSnapshot {
        snapshot_for_caller(
            registry,
            RuntimeToolCaller::assistant(),
            selected_deferred_tools,
            discovery_timeout,
        )
        .await
    }

    async fn snapshot_for_caller(
        registry: &McpConnectionRegistry,
        caller: RuntimeToolCaller,
        selected_deferred_tools: HashSet<String>,
        discovery_timeout: Duration,
    ) -> McpStepSnapshot {
        registry
            .step_snapshot(None, caller, selected_deferred_tools, discovery_timeout)
            .await
    }

    fn tool(name: &str, allowed_callers: Option<&[&str]>) -> Tool {
        let input_schema = allowed_callers.map_or_else(JsonObject::new, |allowed_callers| {
            serde_json::json!({
                "type": "object",
                "x-lime": { "allowed_callers": allowed_callers }
            })
            .as_object()
            .expect("tool schema object")
            .clone()
        });
        Tool {
            name: name.to_string().into(),
            title: None,
            description: Some(format!("{name} tool").into()),
            input_schema: Arc::new(input_schema),
            output_schema: None,
            annotations: None,
            icons: None,
            meta: None,
        }
    }

    fn call(name: &str) -> CallToolRequestParam {
        CallToolRequestParam {
            name: name.to_string().into(),
            arguments: None,
        }
    }

    fn scope() -> McpCallScope {
        McpCallScope::new(Some("turn-1")).expect("turn correlation")
    }

    fn result_text(result: &CallToolResult) -> String {
        serde_json::to_string(result).expect("serialize result")
    }
}
