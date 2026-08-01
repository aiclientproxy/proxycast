use super::{
    mcp_step_snapshot::{DynamicToolRoute, DynamicToolRoutes, McpToolRoutes},
    CurrentTurnHostEvent,
};
use crate::protocol::{
    canonical_tool_item_event, AgentEvent, CanonicalSubAgentActivity, ToolItemLifecycleContext,
};
use agent_protocol::{
    DynamicToolCallContentItem, SessionId, ThreadId, ThreadItem, ThreadItemPayload,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tokio::sync::mpsc::UnboundedSender;
use tool_runtime::tool_lifecycle::{
    ToolLifecycleEmissionFuture, ToolLifecycleEmitter, ToolLifecycleEvent, ToolLifecyclePhase,
};

pub(super) struct CurrentTurnToolLifecycleEmitter {
    event_sender: UnboundedSender<CurrentTurnHostEvent>,
    session_id: SessionId,
    thread_id: ThreadId,
    next_sequence: AtomicU64,
    next_ordinal: AtomicU64,
    items: Mutex<HashMap<String, ToolItemLifecycleState>>,
    mcp_tool_routes: McpToolRoutes,
    dynamic_tool_routes: DynamicToolRoutes,
}

#[derive(Clone, Copy)]
struct ToolItemLifecycleState {
    ordinal: u64,
    created_at_ms: i64,
}

impl CurrentTurnToolLifecycleEmitter {
    #[cfg(test)]
    pub(super) fn new(
        event_sender: UnboundedSender<CurrentTurnHostEvent>,
        session_id: impl Into<String>,
        thread_id: impl Into<String>,
    ) -> Self {
        Self::with_tool_routes(
            event_sender,
            session_id,
            thread_id,
            McpToolRoutes::default(),
            DynamicToolRoutes::default(),
        )
    }

    pub(super) fn with_tool_routes(
        event_sender: UnboundedSender<CurrentTurnHostEvent>,
        session_id: impl Into<String>,
        thread_id: impl Into<String>,
        mcp_tool_routes: McpToolRoutes,
        dynamic_tool_routes: DynamicToolRoutes,
    ) -> Self {
        Self {
            event_sender,
            session_id: SessionId::new(session_id),
            thread_id: ThreadId::new(thread_id),
            next_sequence: AtomicU64::new(0),
            next_ordinal: AtomicU64::new(0),
            items: Mutex::new(HashMap::new()),
            mcp_tool_routes,
            dynamic_tool_routes,
        }
    }

    #[cfg(test)]
    pub(super) fn project(&self, event: ToolLifecycleEvent) -> Option<AgentEvent> {
        self.project_all(event).into_iter().next()
    }

    pub(super) fn project_all(&self, event: ToolLifecycleEvent) -> Vec<AgentEvent> {
        let lifecycle_completed = matches!(event.phase, ToolLifecyclePhase::Completed);
        if lifecycle_completed && event.output.is_none() {
            return Vec::new();
        }
        if event.tool_name == tool_runtime::unified_exec::WRITE_STDIN_TOOL_NAME
            && matches!(event.phase, ToolLifecyclePhase::Started)
        {
            return Vec::new();
        }

        let subagent_activity = CanonicalSubAgentActivity::from_tool_event(&event);
        let mcp_route = self.mcp_tool_routes.get(&event.tool_name);
        let dynamic_tool_route = self.dynamic_tool_routes.get(&event.tool_name);
        let dynamic_tool_arguments = dynamic_tool_route.as_ref().map(|_| event.arguments.clone());
        let now = chrono::Utc::now().timestamp_millis();
        let lifecycle_call_id = event
            .output
            .as_ref()
            .and_then(|output| output.metadata.get("exec_command_call_id"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or(&event.call_id);
        let key = format!("{}\0{}", event.turn_id, lifecycle_call_id);
        let state = {
            let mut items = self
                .items
                .lock()
                .expect("tool item lifecycle mutex poisoned");
            let state = items.get(&key).copied().unwrap_or_else(|| {
                let state = ToolItemLifecycleState {
                    ordinal: self.next_ordinal.fetch_add(1, Ordering::Relaxed) + 1,
                    created_at_ms: now,
                };
                items.insert(key.clone(), state);
                state
            });
            state
        };
        let event_count = if subagent_activity.is_some() { 2 } else { 1 };
        let first_sequence = self.next_sequence.fetch_add(event_count, Ordering::Relaxed) + 1;
        let mut projected = Vec::with_capacity(event_count as usize);
        if let Some(event) = canonical_tool_item_event(
            event,
            ToolItemLifecycleContext {
                session_id: self.session_id.clone(),
                thread_id: self.thread_id.clone(),
                sequence: first_sequence,
                ordinal: state.ordinal,
                created_at_ms: state.created_at_ms,
                updated_at_ms: now,
            },
        ) {
            if matches!(&event, AgentEvent::ItemCompleted { .. }) {
                self.items
                    .lock()
                    .expect("tool item lifecycle mutex poisoned")
                    .remove(&key);
            }
            let event = project_dynamic_tool_event(
                event,
                dynamic_tool_route.as_ref(),
                dynamic_tool_arguments.as_ref(),
            );
            projected.push(project_mcp_tool_event(event, mcp_route.as_ref()));
        }
        if let Some(activity) = subagent_activity {
            projected.push(activity.into_event(ToolItemLifecycleContext {
                session_id: self.session_id.clone(),
                thread_id: self.thread_id.clone(),
                sequence: first_sequence + 1,
                ordinal: self.next_ordinal.fetch_add(1, Ordering::Relaxed) + 1,
                created_at_ms: now,
                updated_at_ms: now,
            }));
        }
        projected
    }
}

fn project_dynamic_tool_event(
    event: AgentEvent,
    route: Option<&DynamicToolRoute>,
    arguments: Option<&serde_json::Value>,
) -> AgentEvent {
    let Some(route) = route else {
        return event;
    };
    match event {
        AgentEvent::ItemStarted { item } => AgentEvent::ItemStarted {
            item: project_dynamic_tool_item(item, route, arguments),
        },
        AgentEvent::ItemUpdated { item } => AgentEvent::ItemUpdated {
            item: project_dynamic_tool_item(item, route, arguments),
        },
        AgentEvent::ItemCompleted { item } => AgentEvent::ItemCompleted {
            item: project_dynamic_tool_item(item, route, arguments),
        },
        event => event,
    }
}

fn project_dynamic_tool_item(
    mut item: ThreadItem,
    route: &DynamicToolRoute,
    arguments: Option<&serde_json::Value>,
) -> ThreadItem {
    let mut metadata = item.metadata.as_object().cloned().unwrap_or_default();
    let content_items = metadata
        .remove("dynamic_tool_content_items")
        .and_then(|value| serde_json::from_value::<Vec<DynamicToolCallContentItem>>(value).ok())
        .unwrap_or_default();
    let success = metadata.remove("success").and_then(|value| value.as_bool());
    metadata.remove("dynamic_tool");
    metadata.remove("duration_ms");
    item.payload = match item.payload {
        ThreadItemPayload::Tool {
            call_id, output, ..
        } => ThreadItemPayload::DynamicToolCall {
            call_id,
            namespace: route.namespace.clone(),
            tool: route.tool.clone(),
            arguments: arguments.cloned().unwrap_or(serde_json::Value::Null),
            content_items,
            success,
            duration_ms: output.and_then(|output| output.duration_ms),
        },
        payload => payload,
    };
    item.kind = item.payload.kind();
    item.metadata = serde_json::Value::Object(metadata);
    item
}

fn project_mcp_tool_event(
    event: AgentEvent,
    route: Option<&tool_runtime::mcp_connection::McpStepRouteIdentity>,
) -> AgentEvent {
    let Some(route) = route else {
        return event;
    };
    match event {
        AgentEvent::ItemStarted { item } => AgentEvent::ItemStarted {
            item: project_mcp_tool_item(item, route),
        },
        AgentEvent::ItemUpdated { item } => AgentEvent::ItemUpdated {
            item: project_mcp_tool_item(item, route),
        },
        AgentEvent::ItemCompleted { item } => AgentEvent::ItemCompleted {
            item: project_mcp_tool_item(item, route),
        },
        event => event,
    }
}

fn project_mcp_tool_item(
    mut item: ThreadItem,
    route: &tool_runtime::mcp_connection::McpStepRouteIdentity,
) -> ThreadItem {
    item.payload = match item.payload {
        ThreadItemPayload::Tool {
            call_id,
            arguments,
            output,
            ..
        } => ThreadItemPayload::McpToolCall {
            call_id,
            server_name: route.server_name.clone(),
            tool_name: route.tool_name.clone(),
            arguments,
            output,
        },
        payload => payload,
    };
    item.kind = item.payload.kind();
    item
}

impl ToolLifecycleEmitter for CurrentTurnToolLifecycleEmitter {
    fn emit<'a>(&'a self, event: ToolLifecycleEvent) -> ToolLifecycleEmissionFuture<'a> {
        Box::pin(async move {
            for event in self.project_all(event) {
                let _ = self
                    .event_sender
                    .send(CurrentTurnHostEvent::ToolLifecycle(event));
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use tool_runtime::tool_call::ToolEnvironment;
    use tool_runtime::tool_result_projection::NormalizedToolOutput;

    #[test]
    fn exact_snapshot_route_projects_mcp_item() {
        let routes = McpToolRoutes::default();
        routes.replace_for_test([tool_runtime::mcp_connection::McpStepRouteIdentity {
            server_name: "docs".to_string(),
            tool_name: "search".to_string(),
            runtime_tool_name: "docs__search".to_string(),
        }]);
        let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
        let emitter = CurrentTurnToolLifecycleEmitter::with_tool_routes(
            sender,
            "session-1",
            "thread-1",
            routes,
            DynamicToolRoutes::default(),
        );

        let event = emitter
            .project(ToolLifecycleEvent {
                turn_id: "turn-1".to_string(),
                call_id: "mcp-call-1".to_string(),
                tool_name: "docs__search".to_string(),
                arguments: serde_json::json!({ "query": "lime" }),
                provider_metadata: serde_json::Value::Null,
                environments: vec![ToolEnvironment::new("local", PathBuf::from("/workspace"))],
                phase: ToolLifecyclePhase::Started,
                output: None,
            })
            .expect("MCP item started");

        let AgentEvent::ItemStarted { item } = event else {
            panic!("expected item started");
        };
        assert_eq!(item.item_id.as_str(), "item_mcp-call-1");
        assert!(matches!(
            item.payload,
            ThreadItemPayload::McpToolCall {
                ref server_name,
                ref tool_name,
                ..
            } if server_name == "docs" && tool_name == "search"
        ));
    }

    #[test]
    fn exact_snapshot_route_projects_typed_dynamic_tool_lifecycle() {
        let routes = DynamicToolRoutes::default();
        routes.replace_for_test([DynamicToolRoute {
            runtime_tool_name: "desktop__appInfo".to_string(),
            namespace: Some("desktop".to_string()),
            tool: "appInfo".to_string(),
        }]);
        let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
        let emitter = CurrentTurnToolLifecycleEmitter::with_tool_routes(
            sender,
            "session-1",
            "thread-1",
            McpToolRoutes::default(),
            routes,
        );
        let arguments = serde_json::json!({
            "includeLocale": true,
            "options": {"platform": "darwin"}
        });

        let started = emitter
            .project(ToolLifecycleEvent {
                turn_id: "turn-1".to_string(),
                call_id: "dynamic-call-1".to_string(),
                tool_name: "desktop__appInfo".to_string(),
                arguments: arguments.clone(),
                provider_metadata: serde_json::Value::Null,
                environments: Vec::new(),
                phase: ToolLifecyclePhase::Started,
                output: None,
            })
            .expect("dynamic item started");
        let AgentEvent::ItemStarted { item: started } = started else {
            panic!("expected dynamic item started");
        };
        assert_eq!(started.kind, agent_protocol::ItemKind::DynamicToolCall);
        assert!(matches!(
            &started.payload,
            ThreadItemPayload::DynamicToolCall {
                call_id,
                namespace: Some(namespace),
                tool,
                arguments: actual_arguments,
                content_items,
                success: None,
                duration_ms: None,
            } if call_id == "dynamic-call-1"
                && namespace == "desktop"
                && tool == "appInfo"
                && actual_arguments == &arguments
                && content_items.is_empty()
        ));

        let content_items = serde_json::json!([
            {"type": "inputText", "text": "Lime 1.116.0"},
            {"type": "inputImage", "imageUrl": "data:image/png;base64,AA=="},
            {"type": "inputAudio", "audioUrl": "data:audio/wav;base64,AA=="}
        ]);
        let completed = emitter
            .project(ToolLifecycleEvent {
                turn_id: "turn-1".to_string(),
                call_id: "dynamic-call-1".to_string(),
                tool_name: "desktop__appInfo".to_string(),
                arguments: arguments.clone(),
                provider_metadata: serde_json::Value::Null,
                environments: Vec::new(),
                phase: ToolLifecyclePhase::Completed,
                output: Some(NormalizedToolOutput {
                    success: true,
                    text: content_items.to_string(),
                    structured_content: Some(content_items.clone()),
                    error: None,
                    duration_ms: 17,
                    truncation: None,
                    sidecar_reference: None,
                    metadata: HashMap::from([(
                        "dynamic_tool_content_items".to_string(),
                        content_items,
                    )]),
                    agent_control_projection_facts: Vec::new(),
                    agent_control_state_facts: Vec::new(),
                }),
            })
            .expect("dynamic item completed");
        let AgentEvent::ItemCompleted { item: completed } = completed else {
            panic!("expected dynamic item completed");
        };
        assert_eq!(completed.item_id, started.item_id);
        assert_eq!(completed.ordinal, started.ordinal);
        assert_eq!(completed.created_at_ms, started.created_at_ms);
        let ThreadItemPayload::DynamicToolCall {
            call_id,
            namespace,
            tool,
            arguments: actual_arguments,
            content_items,
            success,
            duration_ms,
        } = completed.payload
        else {
            panic!("expected typed dynamic payload");
        };
        assert_eq!(call_id, "dynamic-call-1");
        assert_eq!(namespace.as_deref(), Some("desktop"));
        assert_eq!(tool, "appInfo");
        assert_eq!(actual_arguments, arguments);
        assert_eq!(
            content_items,
            vec![
                DynamicToolCallContentItem::InputText {
                    text: "Lime 1.116.0".to_string(),
                },
                DynamicToolCallContentItem::InputImage {
                    image_url: "data:image/png;base64,AA==".to_string(),
                },
                DynamicToolCallContentItem::InputAudio {
                    audio_url: "data:audio/wav;base64,AA==".to_string(),
                },
            ]
        );
        assert_eq!(success, Some(true));
        assert_eq!(duration_ms, Some(17));
        assert!(completed.metadata.get("dynamic_tool").is_none());
        assert!(completed
            .metadata
            .get("dynamic_tool_content_items")
            .is_none());
        assert!(completed.metadata.get("success").is_none());
        assert!(completed.metadata.get("duration_ms").is_none());
    }
}
