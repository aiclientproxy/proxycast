use super::{mcp_step_snapshot::McpToolRoutes, CurrentTurnHostEvent};
use crate::protocol::{
    canonical_tool_item_event, AgentEvent, CanonicalSubAgentActivity, ToolItemLifecycleContext,
};
use agent_protocol::{SessionId, ThreadId, ThreadItem, ThreadItemPayload};
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
        Self::with_mcp_routes(
            event_sender,
            session_id,
            thread_id,
            McpToolRoutes::default(),
        )
    }

    pub(super) fn with_mcp_routes(
        event_sender: UnboundedSender<CurrentTurnHostEvent>,
        session_id: impl Into<String>,
        thread_id: impl Into<String>,
        mcp_tool_routes: McpToolRoutes,
    ) -> Self {
        Self {
            event_sender,
            session_id: SessionId::new(session_id),
            thread_id: ThreadId::new(thread_id),
            next_sequence: AtomicU64::new(0),
            next_ordinal: AtomicU64::new(0),
            items: Mutex::new(HashMap::new()),
            mcp_tool_routes,
        }
    }

    #[cfg(test)]
    pub(super) fn project(&self, event: ToolLifecycleEvent) -> Option<AgentEvent> {
        self.project_all(event).into_iter().next()
    }

    pub(super) fn project_all(&self, event: ToolLifecycleEvent) -> Vec<AgentEvent> {
        let terminal = matches!(event.phase, ToolLifecyclePhase::Completed);
        if terminal && event.output.is_none() {
            return Vec::new();
        }

        let subagent_activity = CanonicalSubAgentActivity::from_tool_event(&event);
        let mcp_route = self.mcp_tool_routes.get(&event.tool_name);
        let now = chrono::Utc::now().timestamp_millis();
        let key = format!("{}\0{}", event.turn_id, event.call_id);
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
            if terminal {
                items.remove(&key);
            }
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
    use std::path::PathBuf;
    use tool_runtime::tool_call::ToolEnvironment;

    #[test]
    fn exact_snapshot_route_projects_mcp_item() {
        let routes = McpToolRoutes::default();
        routes.replace_for_test([tool_runtime::mcp_connection::McpStepRouteIdentity {
            server_name: "docs".to_string(),
            tool_name: "search".to_string(),
            runtime_tool_name: "docs__search".to_string(),
        }]);
        let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
        let emitter = CurrentTurnToolLifecycleEmitter::with_mcp_routes(
            sender,
            "session-1",
            "thread-1",
            routes,
        );

        let event = emitter
            .project(ToolLifecycleEvent {
                turn_id: "turn-1".to_string(),
                call_id: "mcp-call-1".to_string(),
                tool_name: "docs__search".to_string(),
                arguments: serde_json::json!({ "query": "lime" }),
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
}
