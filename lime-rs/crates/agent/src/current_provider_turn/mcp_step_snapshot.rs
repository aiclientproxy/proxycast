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
use rmcp::model::CallToolResult;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::{mpsc::UnboundedSender, Mutex};
use tool_runtime::tool_definition::RuntimeToolDefinition;
use tool_runtime::tool_executor::RuntimeToolExecutorHandle;
use tool_runtime::tool_extension::RuntimeToolCaller;
use tool_runtime::turn_tool_surface::{
    runtime_turn_tool_scope_from_metadata, runtime_turn_tool_surface_allows_tool_name,
    runtime_turn_tool_surface_mode_from_metadata,
};

const MCP_TOOL_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone, Default)]
pub(super) struct DeferredToolSelections(Arc<Mutex<HashSet<String>>>);

#[derive(Clone, Default)]
pub(super) struct McpToolRoutes(
    Arc<RwLock<HashMap<String, tool_runtime::mcp_connection::McpStepRouteIdentity>>>,
);

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
) -> Vec<RuntimeToolDefinition> {
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
    definitions.push(tool_runtime::request_user_input::request_user_input_tool_definition());
    if agent_control_gateway.is_some() {
        definitions.extend(tool_runtime::agent_control::agent_control_tool_definitions());
    }
    definitions.extend(state.gateway_tools().definitions());
    definitions.extend(mcp_tool_definitions(mcp_snapshot));
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
    definitions
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
            self.mcp_tool_routes.replace_from_snapshot(&mcp_snapshot);
            let definitions = tool_definitions(
                &self.state,
                &self.policy,
                self.turn_context.as_ref(),
                &mcp_snapshot,
                self.agent_control_gateway.as_ref(),
            );
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
            let executor = RuntimeToolExecutorHandle::new(Arc::new(CurrentTurnToolExecutor {
                state: self.state.clone(),
                policy: self.policy.clone(),
                event_sender: self.event_sender.clone(),
                thread_id: self.thread_id.clone(),
                mcp_snapshot,
                deferred_tools: self.deferred_tools.clone(),
                agent_control_gateway: self.agent_control_gateway.clone(),
                pending_input: self.pending_input.clone(),
            }));
            Ok(RuntimeToolStepSnapshot::with_tool_metadata(
                definitions,
                executor,
                serial_mcp_tool_names,
                mcp_tool_environment_ids,
            ))
        })
    }
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
        let full = tool_definitions(&state, &policy, None, &snapshot, None);
        let compact_context = turn_context_with_tool_surface(TURN_TOOL_SURFACE_COMPACT_TOOLS);
        let compact = tool_definitions(&state, &policy, Some(&compact_context), &snapshot, None);
        let direct_context = turn_context_with_tool_surface(TURN_TOOL_SURFACE_DIRECT_ANSWER);
        let direct = tool_definitions(&state, &policy, Some(&direct_context), &snapshot, None);

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
    fn compact_surface_defers_agent_control_unless_explicitly_allowed() {
        let state = AgentRuntimeState::new();
        let policy = resolve_request_tool_policy_with_mode(None, Some(RequestToolPolicyMode::Auto));
        let snapshot =
            tool_runtime::mcp_connection::McpStepSnapshot::empty(RuntimeToolCaller::assistant());
        let compact_context = turn_context_with_tool_surface(TURN_TOOL_SURFACE_COMPACT_TOOLS);

        let without_gateway =
            tool_definitions(&state, &policy, Some(&compact_context), &snapshot, None);
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
        );
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
        );
        let names = explicitly_allowed
            .iter()
            .filter(|tool| tool_runtime::agent_control::is_agent_control_tool_name(&tool.name))
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["list_agents", "spawn_agent"]);

        let full = tool_definitions(&state, &policy, None, &snapshot, Some(&gateway));
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
}
