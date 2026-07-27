use super::*;
use agent_protocol::{SubAgentActivityKind, ThreadItemPayload};
use std::collections::HashMap;
use tool_runtime::agent_control::{
    SubAgentProjectionActivity, SubAgentProjectionFact, FOLLOWUP_TASK_TOOL_NAME,
    INTERRUPT_AGENT_TOOL_NAME, SEND_MESSAGE_TOOL_NAME, SPAWN_AGENT_TOOL_NAME,
};
use tool_runtime::tool_result_projection::NormalizedToolOutput;

fn terminal_event(tool_name: &str, activity: SubAgentProjectionActivity) -> ToolLifecycleEvent {
    ToolLifecycleEvent {
        turn_id: "turn-1".to_string(),
        call_id: format!("{tool_name}-call"),
        tool_name: tool_name.to_string(),
        arguments: serde_json::json!({ "target": "research" }),
        provider_metadata: serde_json::Value::Null,
        environments: Vec::new(),
        phase: ToolLifecyclePhase::Completed,
        output: Some(NormalizedToolOutput {
            success: true,
            text: "accepted".to_string(),
            structured_content: None,
            error: None,
            duration_ms: 4,
            truncation: None,
            sidecar_reference: None,
            metadata: HashMap::new(),
            agent_control_projection_facts: vec![SubAgentProjectionFact {
                target_thread_id: ThreadId::new("thread-child"),
                activity,
                detail: Some("/root/research".to_string()),
            }],
        }),
    }
}

#[test]
fn targeted_agent_control_completion_emits_tool_then_subagent() {
    for (tool_name, fact_activity, item_activity) in [
        (
            SPAWN_AGENT_TOOL_NAME,
            SubAgentProjectionActivity::Started,
            SubAgentActivityKind::Started,
        ),
        (
            SEND_MESSAGE_TOOL_NAME,
            SubAgentProjectionActivity::Interacted,
            SubAgentActivityKind::Interacted,
        ),
        (
            FOLLOWUP_TASK_TOOL_NAME,
            SubAgentProjectionActivity::Interacted,
            SubAgentActivityKind::Interacted,
        ),
        (
            INTERRUPT_AGENT_TOOL_NAME,
            SubAgentProjectionActivity::Interrupted,
            SubAgentActivityKind::Interrupted,
        ),
    ] {
        let (sender, _receiver) = mpsc::unbounded_channel();
        let emitter = CurrentTurnToolLifecycleEmitter::new(sender, "session-1", "thread-1");
        let projected = emitter.project_all(terminal_event(tool_name, fact_activity));

        assert_eq!(projected.len(), 2);
        let AgentEvent::ItemCompleted { item: tool_item } = &projected[0] else {
            panic!("expected ordinary tool completion first");
        };
        assert!(matches!(tool_item.payload, ThreadItemPayload::Tool { .. }));
        assert_eq!(
            tool_item.metadata.as_object().expect("tool metadata").len(),
            3,
            "internal activity facts must not enter ordinary Tool metadata"
        );
        let AgentEvent::ItemCompleted {
            item: subagent_item,
        } = &projected[1]
        else {
            panic!("expected subagent completion second");
        };
        assert_eq!(subagent_item.sequence, tool_item.sequence + 1);
        assert_ne!(subagent_item.ordinal, tool_item.ordinal);
        assert_ne!(subagent_item.item_id, tool_item.item_id);
        assert!(matches!(
            &subagent_item.payload,
            ThreadItemPayload::SubAgent {
                child_thread_id,
                activity,
                detail: Some(detail),
            } if child_thread_id.as_str() == "thread-child"
                && *activity == item_activity
                && detail == "/root/research"
        ));
    }
}
