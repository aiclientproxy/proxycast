use super::{is_web_tool, tool_executor::CurrentTurnToolExecutor};
use crate::model_request_policy::{
    native_tool_policy_disallowed_tool_names, native_tool_policy_from_turn_context,
};
use crate::protocol::AgentEvent;
use crate::request_tool_policy::{is_same_tool, RequestToolPolicy};
use crate::runtime_state::AgentRuntimeState;
use agent_protocol::ThreadId;
use agent_runtime::provider_turn::{
    RuntimeToolStepSnapshot, RuntimeToolStepSnapshotFuture, RuntimeToolStepSnapshotSource,
    RuntimeToolStepSnapshotSourceHandle,
};
use agent_runtime::session_loop::RuntimeSessionInputHandle;
use app_server_protocol::protocol::v2::{
    DynamicToolFunctionSpec, DynamicToolNamespaceTool, DynamicToolSpec,
};
use rmcp::model::CallToolResult;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::{mpsc::UnboundedSender, Mutex};
use tool_runtime::tool_definition::RuntimeToolDefinition;
use tool_runtime::tool_definition::RuntimeToolExposure;
use tool_runtime::tool_executor::RuntimeToolExecutorHandle;
use tool_runtime::tool_extension::RuntimeToolCaller;
use tool_runtime::turn_snapshot::{RuntimeToolIdentity, RuntimeToolSnapshot};
use tool_runtime::turn_tool_surface::{
    runtime_turn_tool_scope_from_metadata, runtime_turn_tool_surface_allows_tool_name,
    runtime_turn_tool_surface_mode_from_metadata,
};

const MCP_TOOL_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(2);
const DYNAMIC_TOOLS_METADATA_KEY: &str = "dynamicTools";

#[derive(Clone, Default)]
pub(super) struct DeferredToolSelections(Arc<Mutex<HashSet<String>>>);

#[derive(Clone, Default)]
pub(super) struct McpToolRoutes(
    Arc<RwLock<HashMap<String, tool_runtime::mcp_connection::McpStepRouteIdentity>>>,
);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DynamicToolRoute {
    pub(super) runtime_tool_name: String,
    pub(super) namespace: Option<String>,
    pub(super) tool: String,
}

#[derive(Clone, Default)]
pub(super) struct DynamicToolRoutes(Arc<RwLock<HashMap<String, DynamicToolRoute>>>);

impl DynamicToolRoutes {
    fn replace(&self, routes: Vec<DynamicToolRoute>) {
        *self.0.write().expect("dynamic tool routes lock poisoned") = routes
            .into_iter()
            .map(|route| (route.runtime_tool_name.clone(), route))
            .collect();
    }

    pub(super) fn get(&self, runtime_tool_name: &str) -> Option<DynamicToolRoute> {
        self.0
            .read()
            .expect("dynamic tool routes lock poisoned")
            .get(runtime_tool_name)
            .cloned()
    }

    #[cfg(test)]
    pub(super) fn replace_for_test(&self, routes: impl IntoIterator<Item = DynamicToolRoute>) {
        self.replace(routes.into_iter().collect());
    }
}

impl McpToolRoutes {
    fn replace_from_snapshot(&self, snapshot: &tool_runtime::mcp_connection::McpStepSnapshot) {
        let routes = snapshot
            .tools()
            .iter()
            .filter_map(|tool| snapshot.route_identity(tool.name.as_ref()))
            .map(|route| (route.runtime_tool_name.clone(), route))
            .collect();
        *self.0.write().expect("MCP tool routes lock poisoned") = routes;
    }

    pub(super) fn get(
        &self,
        runtime_tool_name: &str,
    ) -> Option<tool_runtime::mcp_connection::McpStepRouteIdentity> {
        self.0
            .read()
            .expect("MCP tool routes lock poisoned")
            .get(runtime_tool_name)
            .cloned()
    }

    #[cfg(test)]
    pub(super) fn replace_for_test(
        &self,
        routes: impl IntoIterator<Item = tool_runtime::mcp_connection::McpStepRouteIdentity>,
    ) {
        *self.0.write().expect("MCP tool routes lock poisoned") = routes
            .into_iter()
            .map(|route| (route.runtime_tool_name.clone(), route))
            .collect();
    }
}

impl DeferredToolSelections {
    async fn snapshot(&self) -> HashSet<String> {
        self.0.lock().await.clone()
    }

    pub(super) async fn activate_from_tool_search_result(
        &self,
        result: &mut CallToolResult,
    ) -> bool {
        let Some(structured_content) = result.structured_content.as_mut() else {
            return false;
        };
        let matches = structured_content
            .get("matches")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        let mut selected = self.0.lock().await;
        let updated = matches
            .into_iter()
            .fold(false, |updated, name| selected.insert(name) || updated);
        if let Some(object) = structured_content.as_object_mut() {
            object.insert("tool_surface_updated".to_string(), Value::Bool(updated));
        }
        updated
    }
}

pub(super) fn current_tool_step_snapshot_source(
    state: AgentRuntimeState,
    policy: RequestToolPolicy,
    turn_context: Option<agent_protocol::turn_context::TurnContextOverride>,
    event_sender: UnboundedSender<AgentEvent>,
    session_id: String,
    thread_id: ThreadId,
    agent_control_gateway: Option<tool_runtime::agent_control::AgentControlGatewayHandle>,
    pending_input: Option<RuntimeSessionInputHandle>,
    mcp_tool_routes: McpToolRoutes,
    dynamic_tool_routes: DynamicToolRoutes,
    tool_mode: tool_runtime::code_mode::RuntimeToolMode,
    supports_custom_tools: bool,
) -> RuntimeToolStepSnapshotSourceHandle {
    let deferred_tools = DeferredToolSelections::default();
    RuntimeToolStepSnapshotSourceHandle::new(Arc::new(CurrentTurnToolStepSnapshotSource {
        state,
        policy,
        turn_context,
        event_sender,
        session_id,
        thread_id,
        agent_control_gateway,
        pending_input,
        deferred_tools,
        mcp_tool_routes,
        dynamic_tool_routes,
        tool_mode,
        supports_custom_tools,
    }))
}

pub(super) async fn mcp_step_snapshot(
    state: &AgentRuntimeState,
    session_id: &str,
    thread_id: &ThreadId,
    timeout_duration: Duration,
    deferred_tools: &DeferredToolSelections,
) -> tool_runtime::mcp_connection::McpStepSnapshot {
    let Ok(runtime) = state.mcp_runtime(session_id, thread_id.as_str()).await else {
        return tool_runtime::mcp_connection::McpStepSnapshot::empty(RuntimeToolCaller::assistant());
    };
    runtime
        .connections()
        .step_snapshot(
            None,
            RuntimeToolCaller::assistant(),
            deferred_tools.snapshot().await,
            timeout_duration,
        )
        .await
}

fn tool_definitions(
    state: &AgentRuntimeState,
    policy: &RequestToolPolicy,
    turn_context: Option<&agent_protocol::turn_context::TurnContextOverride>,
    mcp_snapshot: &tool_runtime::mcp_connection::McpStepSnapshot,
    agent_control_gateway: Option<&tool_runtime::agent_control::AgentControlGatewayHandle>,
) -> Result<(Vec<RuntimeToolDefinition>, Vec<DynamicToolRoute>), String> {
    let native_policy = native_tool_policy_from_turn_context(turn_context);
    let tool_surface_mode = turn_context
        .and_then(|context| runtime_turn_tool_surface_mode_from_metadata(&context.metadata));
    let tool_scope = turn_context
        .map(|context| runtime_turn_tool_scope_from_metadata(&context.metadata))
        .unwrap_or_default();
    let blocked_by_model = native_tool_policy_disallowed_tool_names(native_policy.as_ref())
        .into_iter()
        .map(str::to_string)
        .collect::<HashSet<_>>();
    let native_dispatch = tool_runtime::native_dispatch::runtime_native_dispatch();
    let mut definitions = native_dispatch.definitions();
    definitions.extend(tool_runtime::unified_exec::unified_exec_tool_definitions());
    definitions.push(tool_runtime::request_permissions::request_permissions_tool_definition());
    definitions.push(tool_runtime::request_user_input::request_user_input_tool_definition());
    if let Some(gateway) = agent_control_gateway {
        definitions.extend(gateway.tool_definitions());
    }
    definitions.extend(state.gateway_tools().definitions());
    definitions.extend(mcp_tool_definitions(mcp_snapshot));
    let (dynamic_definitions, dynamic_routes) = dynamic_tool_definitions(turn_context)?;
    let existing_names = definitions
        .iter()
        .map(|definition| definition.name.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    if let Some(collision) = dynamic_definitions
        .iter()
        .find(|definition| existing_names.contains(&definition.name.to_ascii_lowercase()))
    {
        return Err(format!(
            "dynamic tool '{}' collides with an existing runtime tool",
            collision.name
        ));
    }
    definitions.extend(dynamic_definitions);
    let canonical_name = |name: &str| {
        native_dispatch
            .canonical_name(name)
            .map(ToOwned::to_owned)
            .or_else(|| state.gateway_tools().canonical_name(name))
    };

    let mut seen = HashSet::new();
    definitions.retain(|definition| {
        let key = definition.name.to_ascii_lowercase();
        seen.insert(key)
            && !blocked_by_model
                .iter()
                .any(|name| is_same_tool(name, &definition.name))
            && !policy.matches_any_disallowed_tool(&definition.name)
            && (policy.allows_web_search() || !is_web_tool(&definition.name))
            && runtime_turn_tool_surface_allows_tool_name(
                &definition.name,
                tool_surface_mode.as_ref(),
                &tool_scope.allowed_tools,
                &canonical_name,
            )
    });
    definitions.sort_by(|left, right| left.name.cmp(&right.name));
    Ok((definitions, dynamic_routes))
}

fn dynamic_tool_definitions(
    turn_context: Option<&agent_protocol::turn_context::TurnContextOverride>,
) -> Result<(Vec<RuntimeToolDefinition>, Vec<DynamicToolRoute>), String> {
    let Some(specs) = turn_context
        .and_then(|context| context.metadata.get("runtime_request"))
        .and_then(|metadata| metadata.get(DYNAMIC_TOOLS_METADATA_KEY))
    else {
        return Ok((Vec::new(), Vec::new()));
    };
    let specs = serde_json::from_value::<Vec<DynamicToolSpec>>(specs.clone())
        .map_err(|error| format!("invalid trusted dynamicTools metadata: {error}"))?;
    let mut definitions = Vec::new();
    let mut routes = Vec::new();
    let mut names = HashSet::new();
    for spec in specs {
        match spec {
            DynamicToolSpec::Function(tool) => {
                push_dynamic_tool(None, tool, &mut names, &mut definitions, &mut routes)?;
            }
            DynamicToolSpec::Namespace(namespace) => {
                let namespace_name = dynamic_tool_text(&namespace.name, "namespace name")?;
                dynamic_tool_text(&namespace.description, "namespace description")?;
                for tool in namespace.tools {
                    match tool {
                        DynamicToolNamespaceTool::Function(tool) => {
                            push_dynamic_tool(
                                Some(namespace_name.clone()),
                                tool,
                                &mut names,
                                &mut definitions,
                                &mut routes,
                            )?;
                        }
                    }
                }
            }
        }
    }
    Ok((definitions, routes))
}

fn push_dynamic_tool(
    namespace: Option<String>,
    tool: DynamicToolFunctionSpec,
    names: &mut HashSet<String>,
    definitions: &mut Vec<RuntimeToolDefinition>,
    routes: &mut Vec<DynamicToolRoute>,
) -> Result<(), String> {
    let tool_name = dynamic_tool_text(&tool.name, "tool name")?;
    let description = dynamic_tool_text(&tool.description, "tool description")?;
    if tool.defer_loading {
        return Err(
            "dynamic tool deferLoading is not supported by the current runtime".to_string(),
        );
    }
    if !tool.input_schema.is_object() {
        return Err("dynamic tool inputSchema must be an object".to_string());
    }
    let runtime_tool_name = match namespace.as_deref() {
        Some(namespace) => format!("{namespace}__{tool_name}"),
        None => tool_name.clone(),
    };
    if !names.insert(runtime_tool_name.to_ascii_lowercase()) {
        return Err(format!(
            "dynamic tool runtime name '{runtime_tool_name}' is duplicated"
        ));
    }
    definitions.push(RuntimeToolDefinition {
        name: runtime_tool_name.clone(),
        description,
        input_schema: tool.input_schema,
    });
    routes.push(DynamicToolRoute {
        runtime_tool_name,
        namespace,
        tool: tool_name,
    });
    Ok(())
}

fn dynamic_tool_text(value: &str, field: &str) -> Result<String, String> {
    let value = value.trim();
    if value.is_empty() || value.contains("__") {
        return Err(format!("dynamic tool {field} is invalid"));
    }
    Ok(value.to_string())
}

fn mcp_tool_definitions(
    snapshot: &tool_runtime::mcp_connection::McpStepSnapshot,
) -> Vec<RuntimeToolDefinition> {
    snapshot
        .tools()
        .iter()
        .map(|tool| RuntimeToolDefinition {
            name: tool.name.to_string(),
            description: tool
                .description
                .clone()
                .map(|value| value.to_string())
                .unwrap_or_default(),
            input_schema: Value::Object((*tool.input_schema).clone()),
        })
        .collect()
}

fn resolved_tool_environment_ids(
    definitions: &[RuntimeToolDefinition],
    mut explicit_environment_ids: HashMap<String, String>,
    default_environment_id: Option<&str>,
) -> HashMap<String, String> {
    if let Some(environment_id) = default_environment_id {
        for definition in definitions {
            explicit_environment_ids
                .entry(definition.name.clone())
                .or_insert_with(|| environment_id.to_string());
        }
    }
    explicit_environment_ids
}

#[derive(Clone)]
struct CurrentTurnToolStepSnapshotSource {
    state: AgentRuntimeState,
    policy: RequestToolPolicy,
    turn_context: Option<agent_protocol::turn_context::TurnContextOverride>,
    event_sender: UnboundedSender<AgentEvent>,
    thread_id: ThreadId,
    session_id: String,
    agent_control_gateway: Option<tool_runtime::agent_control::AgentControlGatewayHandle>,
    pending_input: Option<RuntimeSessionInputHandle>,
    deferred_tools: DeferredToolSelections,
    mcp_tool_routes: McpToolRoutes,
    dynamic_tool_routes: DynamicToolRoutes,
    tool_mode: tool_runtime::code_mode::RuntimeToolMode,
    supports_custom_tools: bool,
}

struct CodeModeStepPlan {
    tool_plan: tool_runtime::code_mode::RuntimeCodeModeToolPlan,
    attach_session: bool,
}

fn code_mode_step_plan(
    runtime_tools: &[RuntimeToolSnapshot],
    requested: tool_runtime::code_mode::RuntimeToolMode,
    supports_custom_tools: bool,
    has_session: bool,
) -> Result<CodeModeStepPlan, String> {
    let code_mode_available = supports_custom_tools && has_session;
    let tool_plan = tool_runtime::code_mode::plan_runtime_code_mode_tools(
        runtime_tools,
        requested,
        code_mode_available,
        false,
    )
    .map_err(|error| error.to_string())?;
    let attach_session = code_mode_available
        && tool_plan.resolution.effective != tool_runtime::code_mode::RuntimeToolMode::Direct;
    Ok(CodeModeStepPlan {
        tool_plan,
        attach_session,
    })
}

impl RuntimeToolStepSnapshotSource for CurrentTurnToolStepSnapshotSource {
    fn capture(&self) -> RuntimeToolStepSnapshotFuture<'_> {
        Box::pin(async move {
            let mcp_snapshot = mcp_step_snapshot(
                &self.state,
                &self.session_id,
                &self.thread_id,
                MCP_TOOL_DISCOVERY_TIMEOUT,
                &self.deferred_tools,
            )
            .await;
            let mcp_snapshot = if orchestrator_mcp_enabled(self.turn_context.as_ref()) {
                mcp_snapshot
            } else {
                mcp_snapshot.without_server(lime_skills::APPS_MCP_SERVER_NAME)
            };
            self.mcp_tool_routes.replace_from_snapshot(&mcp_snapshot);
            let (definitions, dynamic_routes) = tool_definitions(
                &self.state,
                &self.policy,
                self.turn_context.as_ref(),
                &mcp_snapshot,
                self.agent_control_gateway.as_ref(),
            )?;
            self.dynamic_tool_routes.replace(dynamic_routes);
            let serial_mcp_tool_names = mcp_snapshot
                .tools()
                .iter()
                .filter(|tool| !mcp_snapshot.supports_parallel_tool_calls(tool.name.as_ref()))
                .map(|tool| tool.name.to_string())
                .collect::<Vec<_>>();
            let mcp_tool_environment_ids =
                mcp_snapshot
                    .tools()
                    .iter()
                    .map(|tool| {
                        let tool_name = tool.name.to_string();
                        let environment_id = mcp_snapshot.environment_id(&tool_name).ok_or_else(|| {
                        format!("MCP tool '{tool_name}' is missing captured environment provenance")
                    })?;
                        Ok((tool_name, environment_id.to_string()))
                    })
                    .collect::<Result<HashMap<_, _>, String>>()?;
            let default_environment_id =
                super::environment::primary_environment_id(self.turn_context.as_ref());
            let executor = RuntimeToolExecutorHandle::new(Arc::new(CurrentTurnToolExecutor {
                state: self.state.clone(),
                policy: self.policy.clone(),
                event_sender: self.event_sender.clone(),
                thread_id: self.thread_id.clone(),
                mcp_snapshot,
                deferred_tools: self.deferred_tools.clone(),
                agent_control_gateway: self.agent_control_gateway.clone(),
                pending_input: self.pending_input.clone(),
                dynamic_tool_routes: self.dynamic_tool_routes.clone(),
            }));
            let runtime_tools = definitions
                .iter()
                .cloned()
                .map(|definition| {
                    let supports_parallel = !serial_mcp_tool_names.contains(&definition.name);
                    RuntimeToolSnapshot::new(
                        RuntimeToolIdentity::plain(definition.name.clone()),
                        definition,
                        RuntimeToolExposure::Direct,
                        supports_parallel,
                        true,
                    )
                })
                .collect::<Vec<_>>();
            let code_mode_session = self
                .pending_input
                .as_ref()
                .and_then(RuntimeSessionInputHandle::code_mode_session);
            let CodeModeStepPlan {
                tool_plan,
                attach_session,
            } = code_mode_step_plan(
                &runtime_tools,
                self.tool_mode,
                self.supports_custom_tools,
                code_mode_session.is_some(),
            )?;
            let definitions: Vec<RuntimeToolDefinition> = tool_plan
                .model_visible_tools
                .into_iter()
                .map(|tool| tool.definition)
                .collect();
            let nested_tools = tool_plan.nested_tools;
            let tool_environment_ids = resolved_tool_environment_ids(
                &definitions,
                mcp_tool_environment_ids,
                default_environment_id.as_deref(),
            );
            let snapshot = RuntimeToolStepSnapshot::with_tool_metadata(
                definitions,
                executor,
                serial_mcp_tool_names,
                tool_environment_ids,
            );
            let snapshot = match self.state.filesystem_gateway() {
                Some(gateway) => snapshot.with_filesystem_gateway(gateway),
                None => snapshot,
            };
            if !attach_session {
                return Ok(snapshot);
            }
            let session = code_mode_session
                .ok_or_else(|| "CodeMode plan requires an executable session".to_string())?;
            Ok(snapshot.with_code_mode_session(session, nested_tools))
        })
    }
}

fn orchestrator_mcp_enabled(
    turn_context: Option<&agent_protocol::turn_context::TurnContextOverride>,
) -> bool {
    let Some(config) = turn_context.and_then(|context| context.metadata.get("config")) else {
        return true;
    };
    if config.pointer("/orchestrator/loadError").is_some() {
        return false;
    }
    config
        .pointer("/orchestrator/mcp/enabled")
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_tool_policy::{
        resolve_request_tool_policy_with_mode, RequestToolPolicyMode,
    };
    use serde_json::json;
    use std::collections::HashMap;
    use tool_runtime::turn_tool_surface::{
        RUNTIME_METADATA_KEY, RUNTIME_TOOL_SURFACE_KEY, TURN_TOOL_SURFACE_COMPACT_TOOLS,
        TURN_TOOL_SURFACE_DIRECT_ANSWER,
    };

    fn turn_context_with_tool_surface(
        surface: &str,
    ) -> agent_protocol::turn_context::TurnContextOverride {
        agent_protocol::turn_context::TurnContextOverride {
            metadata: HashMap::from([(
                RUNTIME_METADATA_KEY.to_string(),
                json!({ RUNTIME_TOOL_SURFACE_KEY: surface }),
            )]),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn provider_step_applies_structured_turn_tool_surface() {
        let state = AgentRuntimeState::new();
        state
            .register_tool_search_tools(Arc::new(EmptyToolSearchGateway))
            .await
            .expect("tool search registration");
        let policy = resolve_request_tool_policy_with_mode(None, Some(RequestToolPolicyMode::Auto));
        let snapshot =
            tool_runtime::mcp_connection::McpStepSnapshot::empty(RuntimeToolCaller::assistant());
        let full = tool_definitions(&state, &policy, None, &snapshot, None)
            .expect("full definitions")
            .0;
        let compact_context = turn_context_with_tool_surface(TURN_TOOL_SURFACE_COMPACT_TOOLS);
        let compact = tool_definitions(&state, &policy, Some(&compact_context), &snapshot, None)
            .expect("compact definitions")
            .0;
        let direct_context = turn_context_with_tool_surface(TURN_TOOL_SURFACE_DIRECT_ANSWER);
        let direct = tool_definitions(&state, &policy, Some(&direct_context), &snapshot, None)
            .expect("direct definitions")
            .0;

        assert!(compact.len() < full.len());
        assert!(compact.iter().any(|tool| tool.name == "WebSearch"));
        assert!(compact.iter().any(|tool| tool.name == "tool_search"));
        assert!(compact.iter().any(|tool| tool.name == "exec_command"));
        assert!(compact.iter().any(|tool| tool.name == "write_stdin"));
        assert!(compact.iter().any(|tool| tool.name == "apply_patch"));
        assert!(compact.iter().any(|tool| tool.name == "request_user_input"));
        assert!(!compact.iter().any(|tool| tool.name == "update_plan"));
        assert!(direct.is_empty());
    }

    #[test]
    fn primary_environment_defaults_native_tools_without_overriding_mcp_provenance() {
        let definitions = vec![
            RuntimeToolDefinition::new("exec_command", "Execute", json!({})),
            RuntimeToolDefinition::new("docs__search", "Search docs", json!({})),
        ];
        let environment_ids = resolved_tool_environment_ids(
            &definitions,
            HashMap::from([("docs__search".to_string(), "local".to_string())]),
            Some("remote-primary"),
        );

        assert_eq!(
            environment_ids.get("exec_command").map(String::as_str),
            Some("remote-primary")
        );
        assert_eq!(
            environment_ids.get("docs__search").map(String::as_str),
            Some("local")
        );
    }

    #[test]
    fn compact_surface_defers_agent_control_unless_explicitly_allowed() {
        let state = AgentRuntimeState::new();
        let policy = resolve_request_tool_policy_with_mode(None, Some(RequestToolPolicyMode::Auto));
        let snapshot =
            tool_runtime::mcp_connection::McpStepSnapshot::empty(RuntimeToolCaller::assistant());
        let compact_context = turn_context_with_tool_surface(TURN_TOOL_SURFACE_COMPACT_TOOLS);

        let without_gateway =
            tool_definitions(&state, &policy, Some(&compact_context), &snapshot, None)
                .expect("definitions")
                .0;
        assert!(!without_gateway
            .iter()
            .any(|tool| { tool_runtime::agent_control::is_agent_control_tool_name(&tool.name) }));

        let gateway = tool_runtime::agent_control::AgentControlGatewayHandle::new(Arc::new(
            RejectingAgentControlGateway,
        ));
        let with_gateway = tool_definitions(
            &state,
            &policy,
            Some(&compact_context),
            &snapshot,
            Some(&gateway),
        )
        .expect("definitions")
        .0;
        assert!(!with_gateway
            .iter()
            .any(|tool| tool_runtime::agent_control::is_agent_control_tool_name(&tool.name)));

        let mut explicitly_allowed_context = compact_context;
        explicitly_allowed_context.metadata.insert(
            "tool_scope".to_string(),
            json!({ "allowed_tools": ["spawn_agent", "list_agents"] }),
        );
        let explicitly_allowed = tool_definitions(
            &state,
            &policy,
            Some(&explicitly_allowed_context),
            &snapshot,
            Some(&gateway),
        )
        .expect("definitions")
        .0;
        let names = explicitly_allowed
            .iter()
            .filter(|tool| tool_runtime::agent_control::is_agent_control_tool_name(&tool.name))
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["list_agents", "spawn_agent"]);

        let full = tool_definitions(&state, &policy, None, &snapshot, Some(&gateway))
            .expect("definitions")
            .0;
        assert_eq!(
            full.iter()
                .filter(|tool| tool_runtime::agent_control::is_agent_control_tool_name(&tool.name))
                .count(),
            tool_runtime::agent_control::agent_control_tool_definitions().len()
        );
    }

    struct RejectingAgentControlGateway;

    struct EmptyToolSearchGateway;

    #[async_trait::async_trait]
    impl tool_runtime::tool_search::ToolSearchGateway for EmptyToolSearchGateway {
        async fn search_tools(
            &self,
            _params: app_server_protocol::McpToolSearchParams,
        ) -> Result<app_server_protocol::McpToolListResponse, String> {
            Ok(app_server_protocol::McpToolListResponse { tools: Vec::new() })
        }
    }

    #[async_trait::async_trait]
    impl tool_runtime::agent_control::AgentControlGateway for RejectingAgentControlGateway {
        async fn execute(
            &self,
            _request: tool_runtime::agent_control::AgentControlGatewayRequest,
        ) -> Result<
            tool_runtime::agent_control::AgentControlGatewayResult,
            tool_runtime::agent_control::AgentControlGatewayError,
        > {
            Err(tool_runtime::agent_control::AgentControlGatewayError::new(
                "test gateway must not execute",
            ))
        }
    }

    #[tokio::test]
    async fn deferred_tool_selections_are_turn_local_and_report_real_updates() {
        let first_turn = DeferredToolSelections::default();
        let second_turn = DeferredToolSelections::default();
        let mut result = CallToolResult::success(Vec::new());
        result.structured_content = Some(serde_json::json!({
            "matches": ["docs__query", "docs__query", "  "],
            "tool_surface_updated": false
        }));

        assert!(
            first_turn
                .activate_from_tool_search_result(&mut result)
                .await
        );
        assert_eq!(
            result
                .structured_content
                .as_ref()
                .and_then(|value| value.get("tool_surface_updated")),
            Some(&Value::Bool(true))
        );
        assert_eq!(
            first_turn.snapshot().await,
            HashSet::from(["docs__query".to_string()])
        );
        assert!(second_turn.snapshot().await.is_empty());

        assert!(
            !first_turn
                .activate_from_tool_search_result(&mut result)
                .await
        );
        assert_eq!(
            result
                .structured_content
                .as_ref()
                .and_then(|value| value.get("tool_surface_updated")),
            Some(&Value::Bool(false))
        );
    }

    #[test]
    fn trusted_dynamic_tools_freeze_exact_namespace_route_and_reject_collisions() {
        let state = AgentRuntimeState::new();
        let policy = resolve_request_tool_policy_with_mode(None, Some(RequestToolPolicyMode::Auto));
        let snapshot =
            tool_runtime::mcp_connection::McpStepSnapshot::empty(RuntimeToolCaller::assistant());
        let context = agent_protocol::turn_context::TurnContextOverride {
            metadata: HashMap::from([(
                "runtime_request".to_string(),
                json!({
                    "dynamicTools": [{
                        "type": "namespace",
                        "name": "desktop",
                        "description": "Desktop host",
                        "tools": [{
                            "type": "function",
                            "name": "appInfo",
                            "description": "Read app information",
                            "inputSchema": {"type": "object", "properties": {}}
                        }]
                    }]
                }),
            )]),
            ..Default::default()
        };
        let (definitions, routes) =
            tool_definitions(&state, &policy, Some(&context), &snapshot, None)
                .expect("dynamic definitions");
        assert!(definitions
            .iter()
            .any(|definition| definition.name == "desktop__appInfo"));
        assert_eq!(
            routes,
            vec![DynamicToolRoute {
                runtime_tool_name: "desktop__appInfo".to_string(),
                namespace: Some("desktop".to_string()),
                tool: "appInfo".to_string(),
            }]
        );

        let collision = agent_protocol::turn_context::TurnContextOverride {
            metadata: HashMap::from([(
                "runtime_request".to_string(),
                json!({
                    "dynamicTools": [{
                        "type": "function",
                        "name": "exec_command",
                        "description": "Collision",
                        "inputSchema": {"type": "object"}
                    }]
                }),
            )]),
            ..Default::default()
        };
        assert!(tool_definitions(&state, &policy, Some(&collision), &snapshot, None).is_err());
    }

    #[test]
    fn code_mode_step_requires_model_capability_and_executable_session() {
        let tools = vec![RuntimeToolSnapshot::new(
            RuntimeToolIdentity::plain("read"),
            RuntimeToolDefinition::new("read", "Read a file", json!({})),
            RuntimeToolExposure::Direct,
            false,
            true,
        )];

        let direct = code_mode_step_plan(
            &tools,
            tool_runtime::code_mode::RuntimeToolMode::Direct,
            true,
            true,
        )
        .expect("direct mode");
        assert_eq!(
            direct.tool_plan.resolution.effective,
            tool_runtime::code_mode::RuntimeToolMode::Direct
        );
        assert!(!direct.attach_session);

        let missing_capability = code_mode_step_plan(
            &tools,
            tool_runtime::code_mode::RuntimeToolMode::CodeMode,
            false,
            true,
        )
        .expect("regular CodeMode falls back to direct");
        assert_eq!(
            missing_capability.tool_plan.resolution.effective,
            tool_runtime::code_mode::RuntimeToolMode::Direct
        );
        assert!(!missing_capability.attach_session);

        let executable = code_mode_step_plan(
            &tools,
            tool_runtime::code_mode::RuntimeToolMode::CodeMode,
            true,
            true,
        )
        .expect("executable CodeMode");
        assert_eq!(
            executable.tool_plan.resolution.effective,
            tool_runtime::code_mode::RuntimeToolMode::CodeMode
        );
        assert!(executable.attach_session);
        assert_eq!(executable.tool_plan.nested_tools[0].global_name, "read");

        assert!(code_mode_step_plan(
            &tools,
            tool_runtime::code_mode::RuntimeToolMode::CodeModeOnly,
            true,
            false,
        )
        .is_err());
    }
}
