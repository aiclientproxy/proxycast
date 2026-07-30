use super::*;

fn lifecycle_context(sequence: u64) -> ToolItemLifecycleContext {
    ToolItemLifecycleContext {
        session_id: SessionId::new("session-1"),
        thread_id: ThreadId::new("thread-1"),
        sequence,
        ordinal: 1,
        created_at_ms: 100,
        updated_at_ms: 200,
    }
}

fn wait_lifecycle_event(
    phase: ToolLifecyclePhase,
    output: Option<NormalizedToolOutput>,
) -> ToolLifecycleEvent {
    ToolLifecycleEvent {
        turn_id: "turn-1".to_string(),
        call_id: "wait-call".to_string(),
        tool_name: tool_runtime::agent_control::WAIT_AGENT_TOOL_NAME.to_string(),
        arguments: serde_json::json!({ "timeout_ms": 1000 }),
        provider_metadata: serde_json::Value::Null,
        environments: Vec::new(),
        phase,
        output,
    }
}

fn targeted_lifecycle_event(
    tool_name: &str,
    activity: tool_runtime::agent_control::SubAgentProjectionActivity,
    success: bool,
) -> ToolLifecycleEvent {
    let facts = vec![tool_runtime::agent_control::SubAgentProjectionFact {
        target_thread_id: ThreadId::new("thread-child"),
        activity,
        detail: Some("/root/research".to_string()),
    }];
    ToolLifecycleEvent {
        turn_id: "turn-1".to_string(),
        call_id: format!("{tool_name}-call"),
        tool_name: tool_name.to_string(),
        arguments: serde_json::json!({ "target": "research" }),
        provider_metadata: serde_json::Value::Null,
        environments: Vec::new(),
        phase: ToolLifecyclePhase::Completed,
        output: Some(NormalizedToolOutput {
            success,
            text: "accepted".to_string(),
            structured_content: None,
            error: (!success).then(|| "failed".to_string()),
            duration_ms: 10,
            truncation: None,
            sidecar_reference: None,
            metadata: HashMap::new(),
            agent_control_projection_facts: facts,
            agent_control_state_facts: Vec::new(),
        }),
    }
}

#[test]
fn canonical_wait_agent_projects_one_collab_lifecycle_item() {
    let started = canonical_tool_item_event(
        wait_lifecycle_event(ToolLifecyclePhase::Started, None),
        lifecycle_context(1),
    )
    .expect("wait started item");
    let AgentEvent::ItemStarted { item: started } = started else {
        panic!("expected wait started item");
    };
    assert_eq!(started.kind, agent_protocol::ItemKind::CollabAgentToolCall);
    assert_eq!(started.status, ItemStatus::InProgress);
    assert!(matches!(
        started.payload,
        ThreadItemPayload::CollabAgentToolCall {
            operation: CollabAgentOperation::Wait,
            target_thread_id: None,
            message: None,
            output: None,
            ..
        }
    ));

    let completed = canonical_tool_item_event(
        wait_lifecycle_event(
            ToolLifecyclePhase::Completed,
            Some(NormalizedToolOutput {
                success: true,
                text: "Wait completed.".to_string(),
                structured_content: Some(serde_json::json!({ "timed_out": false })),
                error: None,
                duration_ms: 25,
                truncation: None,
                sidecar_reference: None,
                metadata: HashMap::new(),
                agent_control_projection_facts: Vec::new(),
                agent_control_state_facts: vec![
                    tool_runtime::agent_control::AgentStateProjectionFact {
                        target_thread_id: ThreadId::new("thread-child"),
                        state: agent_protocol::CollabAgentState {
                            status: agent_protocol::CollabAgentStatus::Completed,
                            message: None,
                        },
                    },
                ],
            }),
        ),
        lifecycle_context(2),
    )
    .expect("wait completed item");
    let AgentEvent::ItemCompleted { item: completed } = completed else {
        panic!("expected wait completed item");
    };
    assert_eq!(completed.item_id, started.item_id);
    assert_eq!(completed.ordinal, started.ordinal);
    assert_eq!(completed.status, ItemStatus::Completed);
    let ThreadItemPayload::CollabAgentToolCall {
        operation,
        target_thread_id,
        message,
        output,
        agent_states,
        ..
    } = completed.payload
    else {
        panic!("expected canonical wait collab payload");
    };
    assert_eq!(operation, CollabAgentOperation::Wait);
    assert!(target_thread_id.is_none());
    assert!(message.is_none());
    assert_eq!(
        agent_states[&ThreadId::new("thread-child")].status,
        agent_protocol::CollabAgentStatus::Completed
    );
    let output = output.expect("wait terminal output");
    assert_eq!(output.text.as_deref(), Some("Wait completed."));
    assert_eq!(
        output.structured_content,
        Some(serde_json::json!({ "timed_out": false }))
    );
    assert_eq!(output.duration_ms, Some(25));
}

#[test]
fn targeted_agent_control_projects_distinct_terminal_subagent_items() {
    use agent_protocol::SubAgentActivityKind;
    use tool_runtime::agent_control::{
        SubAgentProjectionActivity, FOLLOWUP_TASK_TOOL_NAME, INTERRUPT_AGENT_TOOL_NAME,
        SEND_MESSAGE_TOOL_NAME, SPAWN_AGENT_TOOL_NAME,
    };

    for (tool_name, projection_activity, protocol_activity) in [
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
        let lifecycle = targeted_lifecycle_event(tool_name, projection_activity, true);
        let tool_event = canonical_tool_item_event(lifecycle.clone(), lifecycle_context(1))
            .expect("ordinary tool completion");
        let AgentEvent::ItemCompleted { item: tool_item } = tool_event else {
            panic!("expected ordinary tool completion");
        };
        assert!(matches!(tool_item.payload, ThreadItemPayload::Tool { .. }));
        assert_eq!(
            tool_item.metadata.as_object().expect("tool metadata").len(),
            3,
            "internal activity facts must not enter ordinary Tool metadata"
        );

        let subagent_event = CanonicalSubAgentActivity::from_tool_event(&lifecycle)
            .expect("canonical subagent activity")
            .into_event(lifecycle_context(2));
        let AgentEvent::ItemCompleted {
            item: subagent_item,
        } = subagent_event
        else {
            panic!("expected terminal subagent item");
        };
        assert_ne!(subagent_item.item_id, tool_item.item_id);
        assert_eq!(subagent_item.sequence, tool_item.sequence + 1);
        assert_eq!(subagent_item.status, ItemStatus::Completed);
        assert!(matches!(
            subagent_item.payload,
            ThreadItemPayload::SubAgent {
                child_thread_id,
                activity,
                detail: Some(ref detail),
            } if child_thread_id.as_str() == "thread-child"
                && activity == protocol_activity
                && detail == "/root/research"
        ));

        let repeated = CanonicalSubAgentActivity::from_tool_event(&lifecycle)
            .expect("repeat canonical subagent activity")
            .into_event(lifecycle_context(99));
        let AgentEvent::ItemCompleted { item: repeated } = repeated else {
            panic!("expected repeated terminal subagent item");
        };
        assert_eq!(repeated.item_id, subagent_item.item_id);
    }
}

#[test]
fn failed_or_unmatched_agent_control_does_not_fabricate_subagent_activity() {
    use tool_runtime::agent_control::{
        SubAgentProjectionActivity, LIST_AGENTS_TOOL_NAME, SPAWN_AGENT_TOOL_NAME,
    };

    let failed = targeted_lifecycle_event(
        SPAWN_AGENT_TOOL_NAME,
        SubAgentProjectionActivity::Started,
        false,
    );
    assert!(CanonicalSubAgentActivity::from_tool_event(&failed).is_none());

    let list = targeted_lifecycle_event(
        LIST_AGENTS_TOOL_NAME,
        SubAgentProjectionActivity::Started,
        true,
    );
    assert!(CanonicalSubAgentActivity::from_tool_event(&list).is_none());

    let mismatched = targeted_lifecycle_event(
        SPAWN_AGENT_TOOL_NAME,
        SubAgentProjectionActivity::Interacted,
        true,
    );
    assert!(CanonicalSubAgentActivity::from_tool_event(&mismatched).is_none());

    let mut started = targeted_lifecycle_event(
        SPAWN_AGENT_TOOL_NAME,
        SubAgentProjectionActivity::Started,
        true,
    );
    started.phase = ToolLifecyclePhase::Started;
    assert!(CanonicalSubAgentActivity::from_tool_event(&started).is_none());

    let mut empty_target = targeted_lifecycle_event(
        SPAWN_AGENT_TOOL_NAME,
        SubAgentProjectionActivity::Started,
        true,
    );
    empty_target
        .output
        .as_mut()
        .expect("output")
        .agent_control_projection_facts[0]
        .target_thread_id = ThreadId::new("");
    assert!(CanonicalSubAgentActivity::from_tool_event(&empty_target).is_none());

    let mut duplicate = targeted_lifecycle_event(
        SPAWN_AGENT_TOOL_NAME,
        SubAgentProjectionActivity::Started,
        true,
    );
    let fact = duplicate
        .output
        .as_ref()
        .expect("output")
        .agent_control_projection_facts[0]
        .clone();
    duplicate
        .output
        .as_mut()
        .expect("output")
        .agent_control_projection_facts
        .push(fact);
    assert!(CanonicalSubAgentActivity::from_tool_event(&duplicate).is_none());
}
