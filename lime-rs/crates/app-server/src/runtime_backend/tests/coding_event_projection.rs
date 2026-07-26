use super::*;
use crate::runtime_backend::event_mapper::emit_runtime_agent_event_with_coding_mirror_and_plan_parser;
use agent_protocol::{
    ItemId, ItemStatus, SessionId, ThreadId, ThreadItem, ThreadItemPayload, ToolArgument,
    ToolOutput, TurnId,
};

fn tool_started(tool_name: &str, tool_id: &str, arguments: Option<Value>) -> RuntimeAgentEvent {
    canonical_tool_event(tool_name, tool_id, arguments, None)
}

fn tool_completed(tool_id: &str, result: AgentToolResult) -> RuntimeAgentEvent {
    canonical_tool_event("tool", tool_id, None, Some(result))
}

fn canonical_tool_event(
    tool_name: &str,
    tool_id: &str,
    arguments: Option<Value>,
    result: Option<AgentToolResult>,
) -> RuntimeAgentEvent {
    let arguments = arguments
        .map(|arguments| match arguments {
            Value::Object(arguments) => arguments
                .into_iter()
                .map(|(name, value)| ToolArgument {
                    name,
                    value: value
                        .as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|| value.to_string()),
                })
                .collect(),
            value => vec![ToolArgument {
                name: "value".to_string(),
                value: value
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| value.to_string()),
            }],
        })
        .unwrap_or_default();
    let (status, output, metadata) = match result {
        Some(result) => {
            let status = if result.success {
                ItemStatus::Completed
            } else {
                ItemStatus::Failed
            };
            let metadata = result
                .metadata
                .map(|metadata| Value::Object(metadata.into_iter().collect()))
                .unwrap_or_else(|| json!({}));
            let output = ToolOutput {
                text: Some(result.output),
                structured_content: result.structured_content,
                error: result.error,
                duration_ms: metadata.get("duration_ms").and_then(Value::as_u64),
                truncated: metadata
                    .get("truncated")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                output_ref: metadata
                    .get("output_ref")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            };
            (status, Some(output), metadata)
        }
        None => (ItemStatus::InProgress, None, json!({})),
    };
    let payload = ThreadItemPayload::Tool {
        call_id: tool_id.to_string(),
        name: tool_name.to_string(),
        arguments,
        output,
    };
    let item = ThreadItem {
        session_id: SessionId::new("session-test"),
        thread_id: ThreadId::new("thread-test"),
        turn_id: TurnId::new("turn-test"),
        item_id: ItemId::new(tool_id),
        sequence: 1,
        ordinal: 1,
        created_at_ms: 1,
        updated_at_ms: 2,
        completed_at_ms: status.is_terminal().then_some(2),
        kind: payload.kind(),
        status,
        payload,
        metadata,
    };
    if status.is_terminal() {
        RuntimeAgentEvent::ItemCompleted { item }
    } else {
        RuntimeAgentEvent::ItemStarted { item }
    }
}

#[test]
fn runtime_agent_tool_events_are_mirrored_to_coding_facts() {
    let mut sink = TestRuntimeEventSink::default();
    let mut mirror = coding_events::CodingEventMirror::default();

    emit_runtime_agent_event_with_coding_mirror(
        &tool_started(
            "Bash",
            "tool-test",
            Some(json!({ "command": "cargo test -p app-server coding_events" })),
        ),
        &mut sink,
        &mut mirror,
    )
    .expect("tool start should emit");
    emit_runtime_agent_event_with_coding_mirror(
        &RuntimeAgentEvent::ToolOutputDelta {
            tool_id: "tool-test".to_string(),
            delta: "running tests".to_string(),
            output_kind: Some("stdout".to_string()),
            metadata: None,
        },
        &mut sink,
        &mut mirror,
    )
    .expect("tool output should emit");
    emit_runtime_agent_event_with_coding_mirror(
        &tool_completed(
            "tool-test",
            AgentToolResult {
                success: true,
                output: "ok".to_string(),
                error: None,
                structured_content: None,
                images: None,
                metadata: Some(HashMap::from([
                    ("exit_code".to_string(), json!(0)),
                    (
                        "command".to_string(),
                        json!("cargo test -p app-server coding_events"),
                    ),
                ])),
            },
        ),
        &mut sink,
        &mut mirror,
    )
    .expect("tool end should emit");

    let event_types = sink
        .events
        .iter()
        .map(|event| event.event_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec![
            "item.started",
            "command.started",
            "test.started",
            "tool.output.delta",
            "command.output",
            "item.completed",
            "command.exited",
            "test.completed"
        ]
    );
    let item_started = sink
        .events
        .iter()
        .find(|event| event.event_type == "item.started")
        .expect("canonical tool item start");
    assert_eq!(
        item_started.payload["item"]["payload"]["call_id"].as_str(),
        Some("tool-test")
    );
    let arguments = item_started.payload["item"]["payload"]["arguments"]
        .as_array()
        .expect("typed tool arguments");
    assert!(arguments.iter().any(|argument| {
        argument["name"] == "command"
            && argument["value"] == "cargo test -p app-server coding_events"
    }));
    let command_started = sink
        .events
        .iter()
        .find(|event| event.event_type == "command.started")
        .expect("command started event");
    assert_eq!(
        command_started.payload["canonicalCommand"].as_str(),
        Some("cargo test -p app-server coding_events")
    );
    assert_eq!(
        command_started.payload["commandSummary"].as_str(),
        Some("cargo test -p app-server")
    );
    assert_eq!(
        command_started.payload["commandArgv"]
            .as_array()
            .expect("argv"),
        &vec![
            json!("cargo"),
            json!("test"),
            json!("-p"),
            json!("app-server"),
            json!("coding_events")
        ]
    );
    let command_exited = sink
        .events
        .iter()
        .find(|event| event.event_type == "command.exited")
        .expect("command exited event");
    assert_eq!(
        command_exited.payload["canonicalCommand"].as_str(),
        Some("cargo test -p app-server coding_events")
    );
}

#[test]
fn model_effective_event_records_selected_model_and_reasoning_policy() {
    let selection = RuntimeModelSelection {
        provider: "openai".to_string(),
        model: "gpt-codex".to_string(),
        source: "runtime_options",
        reasoning_effort: Some("high".to_string()),
    };
    let provider_config = SessionProviderConfig {
        provider_name: "openai".to_string(),
        provider_selector: Some("openai-main".to_string()),
        model_name: "gpt-codex".to_string(),
        api_key: Some("sk-test".to_string()),
        base_url: Some("https://api.example.test/v1".to_string()),
        credential_uuid: None,
        reasoning_effort: Some("high".to_string()),
        service_tier: None,
        route_protocol: None,
        toolshim: false,
        toolshim_model: None,
        model_capabilities: None,
        supports_websockets: false,
    };

    let snapshot = reasoning_capability_snapshot(&["low", "medium", "high", "max", "xhigh"]);
    let event = model_effective_event_from_runtime(
        &selection,
        &selection,
        &provider_config,
        "coding",
        &snapshot,
    );

    assert_eq!(event.event_type, "model.effective");
    assert_eq!(event.payload["model"]["providerId"], "openai-main");
    assert_eq!(event.payload["model"]["modelId"], "gpt-codex");
    assert_eq!(event.payload["modelRef"]["providerId"], "openai-main");
    assert_eq!(event.payload["reasoning"]["supported"], true);
    assert_eq!(event.payload["reasoning"]["requestedLevel"], "high");
    assert_eq!(event.payload["reasoning"]["effectiveLevel"], "high");
    assert_eq!(event.payload["toolCalling"]["supported"], true);
    assert_eq!(event.payload["requestedReasoningEffort"], "high");
    assert_eq!(event.payload["effectiveReasoningEffort"], "high");
    assert_eq!(event.payload["serviceModelSlot"], "coding");
    assert_eq!(event.payload["source"], "runtime_options");
}

#[test]
fn model_effective_event_preserves_snapshot_declared_custom_reasoning() {
    let requested_selection = RuntimeModelSelection {
        provider: "openai".to_string(),
        model: "gpt-codex".to_string(),
        source: "profile_model_slot",
        reasoning_effort: Some("ultra".to_string()),
    };
    let snapshot = reasoning_capability_snapshot(&["low", "high", "ultra"]);
    let effective_selection = selection_with_capability_reasoning(&requested_selection, &snapshot);
    let provider_config = SessionProviderConfig {
        provider_name: "openai".to_string(),
        provider_selector: Some("openai-main".to_string()),
        model_name: "gpt-codex".to_string(),
        api_key: Some("sk-test".to_string()),
        base_url: Some("https://api.example.test/v1".to_string()),
        credential_uuid: None,
        reasoning_effort: effective_selection.reasoning_effort.clone(),
        service_tier: None,
        route_protocol: None,
        toolshim: false,
        toolshim_model: None,
        model_capabilities: None,
        supports_websockets: false,
    };

    let event = model_effective_event_from_runtime(
        &requested_selection,
        &effective_selection,
        &provider_config,
        "fast",
        &snapshot,
    );

    assert_eq!(event.payload["reasoning"]["requestedLevel"], "ultra");
    assert_eq!(event.payload["reasoning"]["effectiveLevel"], "ultra");
    assert!(event.payload["reasoning"].get("downgradeReason").is_none());
    assert_eq!(event.payload["requestedReasoningEffort"], "ultra");
    assert_eq!(event.payload["effectiveReasoningEffort"], "ultra");
}

#[test]
fn model_effective_event_uses_route_capability_snapshot_over_model_slug() {
    let selection = RuntimeModelSelection {
        provider: "openai".to_string(),
        model: "gpt-codex".to_string(),
        source: "runtime_options",
        reasoning_effort: Some("high".to_string()),
    };
    let provider_config = SessionProviderConfig {
        provider_name: "openai".to_string(),
        provider_selector: Some("relay-openai".to_string()),
        model_name: "gpt-codex".to_string(),
        api_key: Some("sk-test".to_string()),
        base_url: Some("https://api.example.test/v1".to_string()),
        credential_uuid: None,
        reasoning_effort: Some("high".to_string()),
        service_tier: None,
        route_protocol: None,
        toolshim: false,
        toolshim_model: None,
        model_capabilities: None,
        supports_websockets: false,
    };
    let snapshot = no_reasoning_capability_snapshot();

    let event = model_effective_event_from_runtime(
        &selection,
        &selection,
        &provider_config,
        "coding",
        &snapshot,
    );

    assert_eq!(event.payload["reasoning"]["supported"], false);
    assert_eq!(event.payload["reasoning"]["requestedLevel"], "high");
    assert!(event.payload["reasoning"].get("effectiveLevel").is_none());
    assert_eq!(
        event.payload["reasoning"]["downgradeReason"],
        "selected model does not support reasoning"
    );
    assert!(event.payload.get("effectiveReasoningEffort").is_none());
}

fn reasoning_capability_snapshot(levels: &[&str]) -> app_server_protocol::CapabilitySnapshot {
    app_server_protocol::CapabilitySnapshot {
        runtime_features: vec!["streaming".to_string(), "reasoning".to_string()],
        capabilities: app_server_protocol::ModelCapabilitiesInfo {
            tools: true,
            streaming: true,
            reasoning: true,
            reasoning_effort: Some(app_server_protocol::ModelReasoningEffortSupportInfo {
                supported: true,
                levels: levels.iter().map(|level| (*level).to_string()).collect(),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn no_reasoning_capability_snapshot() -> app_server_protocol::CapabilitySnapshot {
    app_server_protocol::CapabilitySnapshot {
        runtime_features: vec!["streaming".to_string()],
        capabilities: app_server_protocol::ModelCapabilitiesInfo {
            tools: true,
            streaming: true,
            reasoning: false,
            reasoning_effort: Some(app_server_protocol::ModelReasoningEffortSupportInfo {
                supported: false,
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn runtime_reasoning_summary_and_content_emit_distinct_lifecycle_events() {
    let mut sink = TestRuntimeEventSink::default();
    let mut mirror = coding_events::CodingEventMirror::default();
    let mut proposed_plan_parser = proposed_plan_parser::ProposedPlanParser::default();
    let mut reasoning_state = reasoning_events::ReasoningEventState::default();

    emit_runtime_agent_event_with_coding_mirror_and_plan_parser(
        &RuntimeAgentEvent::ReasoningStart {
            item_id: "reasoning-1".to_string(),
        },
        &mut sink,
        &mut mirror,
        &mut proposed_plan_parser,
        &mut reasoning_state,
    )
    .expect("reasoning start should emit");
    emit_runtime_agent_event_with_coding_mirror_and_plan_parser(
        &RuntimeAgentEvent::ReasoningSummaryPartAdded {
            item_id: "reasoning-1".to_string(),
            summary_index: 0,
        },
        &mut sink,
        &mut mirror,
        &mut proposed_plan_parser,
        &mut reasoning_state,
    )
    .expect("reasoning summary part should emit");
    emit_runtime_agent_event_with_coding_mirror_and_plan_parser(
        &RuntimeAgentEvent::ReasoningSummaryDelta {
            item_id: "reasoning-1".to_string(),
            text: "摘要".to_string(),
            summary_index: 0,
        },
        &mut sink,
        &mut mirror,
        &mut proposed_plan_parser,
        &mut reasoning_state,
    )
    .expect("reasoning summary delta should emit");
    emit_runtime_agent_event_with_coding_mirror_and_plan_parser(
        &RuntimeAgentEvent::ReasoningContentDelta {
            item_id: "reasoning-1".to_string(),
            text: "原始推理".to_string(),
            content_index: 0,
        },
        &mut sink,
        &mut mirror,
        &mut proposed_plan_parser,
        &mut reasoning_state,
    )
    .expect("reasoning content delta should emit");
    emit_runtime_agent_event_with_coding_mirror_and_plan_parser(
        &RuntimeAgentEvent::ReasoningEnd {
            item_id: "reasoning-1".to_string(),
        },
        &mut sink,
        &mut mirror,
        &mut proposed_plan_parser,
        &mut reasoning_state,
    )
    .expect("reasoning end should emit");

    let event_types = sink
        .events
        .iter()
        .map(|event| event.event_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec![
            "reasoning.started",
            "reasoning.summary_part_added",
            "reasoning.summary",
            "reasoning.delta",
            "reasoning.final",
            "reasoning.ended",
        ]
    );
    assert_eq!(sink.events[0].payload["reasoningId"], "reasoning-1");
    assert_eq!(sink.events[1].payload["itemId"], "reasoning-1");
    assert_eq!(sink.events[1].payload["summaryIndex"], 0);
    assert_eq!(sink.events[2].payload["summary"], "摘要");
    assert_eq!(sink.events[2].payload["summaryIndex"], 0);
    assert_eq!(sink.events[3].payload["delta"], "原始推理");
    assert_eq!(sink.events[3].payload["contentIndex"], 0);
    assert_eq!(sink.events[4].payload["text"], "摘要");
    assert_eq!(sink.events[4].payload["summary"], json!(["摘要"]));
    assert_eq!(sink.events[4].payload["content"], json!(["原始推理"]));
    assert_eq!(sink.events[5].payload["status"], "completed");
}

#[test]
fn runtime_agent_message_emits_its_own_terminal_before_turn_terminal() {
    let mut sink = TestRuntimeEventSink::default();
    let mut mirror = coding_events::CodingEventMirror::default();
    let mut proposed_plan_parser = proposed_plan_parser::ProposedPlanParser::default();
    let mut reasoning_state = reasoning_events::ReasoningEventState::default();

    emit_runtime_agent_event_with_coding_mirror_and_plan_parser(
        &RuntimeAgentEvent::TextStart {
            item_id: "message-1".to_string(),
        },
        &mut sink,
        &mut mirror,
        &mut proposed_plan_parser,
        &mut reasoning_state,
    )
    .expect("message start should emit");
    emit_runtime_agent_event_with_coding_mirror_and_plan_parser(
        &RuntimeAgentEvent::TextDelta {
            item_id: "message-1".to_string(),
            text: "answer".to_string(),
        },
        &mut sink,
        &mut mirror,
        &mut proposed_plan_parser,
        &mut reasoning_state,
    )
    .expect("message delta should emit");
    emit_runtime_agent_event_with_coding_mirror_and_plan_parser(
        &RuntimeAgentEvent::TextEnd {
            item_id: "message-1".to_string(),
            phase: lime_agent::protocol::AgentMessagePhase::FinalAnswer,
        },
        &mut sink,
        &mut mirror,
        &mut proposed_plan_parser,
        &mut reasoning_state,
    )
    .expect("message end should emit");
    sink.emit(RuntimeEvent::new("turn.completed", json!({})))
        .expect("turn completion should emit");

    assert_eq!(
        sink.events
            .iter()
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        vec!["message.delta", "message.completed", "turn.completed"]
    );
    assert_eq!(sink.events[1].payload["status"], "completed");
    assert_eq!(sink.events[1].payload["phase"], "final_answer");
    assert_eq!(sink.events[1].payload["itemId"], "message-1");
}

#[test]
fn runtime_output_items_keep_distinct_provider_identities() {
    let mut sink = TestRuntimeEventSink::default();
    let mut mirror = coding_events::CodingEventMirror::default();
    let mut parser = proposed_plan_parser::ProposedPlanParser::default();
    let mut reasoning_state = reasoning_events::ReasoningEventState::default();
    let events = [
        RuntimeAgentEvent::ReasoningStart {
            item_id: "reasoning-1".to_string(),
        },
        RuntimeAgentEvent::ReasoningContentDelta {
            item_id: "reasoning-1".to_string(),
            text: "first reasoning".to_string(),
            content_index: 0,
        },
        RuntimeAgentEvent::ReasoningEnd {
            item_id: "reasoning-1".to_string(),
        },
        RuntimeAgentEvent::TextStart {
            item_id: "message-1".to_string(),
        },
        RuntimeAgentEvent::TextDelta {
            item_id: "message-1".to_string(),
            text: "first answer".to_string(),
        },
        RuntimeAgentEvent::TextEnd {
            item_id: "message-1".to_string(),
            phase: lime_agent::protocol::AgentMessagePhase::Commentary,
        },
        RuntimeAgentEvent::ReasoningStart {
            item_id: "reasoning-2".to_string(),
        },
        RuntimeAgentEvent::ReasoningContentDelta {
            item_id: "reasoning-2".to_string(),
            text: "second reasoning".to_string(),
            content_index: 0,
        },
        RuntimeAgentEvent::ReasoningEnd {
            item_id: "reasoning-2".to_string(),
        },
        RuntimeAgentEvent::TextStart {
            item_id: "message-2".to_string(),
        },
        RuntimeAgentEvent::TextDelta {
            item_id: "message-2".to_string(),
            text: "second answer".to_string(),
        },
        RuntimeAgentEvent::TextEnd {
            item_id: "message-2".to_string(),
            phase: lime_agent::protocol::AgentMessagePhase::FinalAnswer,
        },
    ];

    for event in &events {
        emit_runtime_agent_event_with_coding_mirror_and_plan_parser(
            event,
            &mut sink,
            &mut mirror,
            &mut parser,
            &mut reasoning_state,
        )
        .expect("provider output item should map");
    }

    let item_events = sink
        .events
        .iter()
        .filter(|event| {
            matches!(
                event.event_type.as_str(),
                "reasoning.started"
                    | "reasoning.delta"
                    | "reasoning.final"
                    | "reasoning.ended"
                    | "message.delta"
                    | "message.completed"
            )
        })
        .map(|event| {
            (
                event.event_type.as_str(),
                event.payload["itemId"].as_str().expect("itemId"),
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(
        item_events,
        vec![
            ("reasoning.started", "reasoning-1"),
            ("reasoning.delta", "reasoning-1"),
            ("reasoning.final", "reasoning-1"),
            ("reasoning.ended", "reasoning-1"),
            ("message.delta", "message-1"),
            ("message.completed", "message-1"),
            ("reasoning.started", "reasoning-2"),
            ("reasoning.delta", "reasoning-2"),
            ("reasoning.final", "reasoning-2"),
            ("reasoning.ended", "reasoning-2"),
            ("message.delta", "message-2"),
            ("message.completed", "message-2"),
        ]
    );
    let message_phases = sink
        .events
        .iter()
        .filter(|event| event.event_type == "message.completed")
        .map(|event| event.payload["phase"].as_str().expect("message phase"))
        .collect::<Vec<_>>();
    assert_eq!(message_phases, vec!["commentary", "final_answer"]);
}

#[test]
fn canceled_runtime_agent_message_uses_interrupted_item_status() {
    let mut sink = TestRuntimeEventSink::default();
    let mut parser = proposed_plan_parser::ProposedPlanParser::default();
    for event in proposed_plan_parser::split_runtime_event(
        RuntimeEvent::new("message.delta", json!({"text": "partial"})),
        &mut parser,
    ) {
        sink.emit(event).expect("message delta should emit");
    }

    emit_agent_message_finish(&mut parser, "interrupted", &mut sink)
        .expect("message interruption should emit");

    assert_eq!(sink.events[1].event_type, "message.completed");
    assert_eq!(sink.events[1].payload["status"], "interrupted");
    assert_eq!(sink.events[1].payload["phase"], "commentary");
}

#[test]
fn failed_runtime_output_items_finish_before_turn_failure() {
    let mut sink = TestRuntimeEventSink::default();
    let mut mirror = coding_events::CodingEventMirror::default();
    let mut parser = proposed_plan_parser::ProposedPlanParser::default();
    let mut reasoning_state = reasoning_events::ReasoningEventState::default();
    for event in [
        RuntimeAgentEvent::ReasoningStart {
            item_id: "reasoning-1".to_string(),
        },
        RuntimeAgentEvent::ReasoningContentDelta {
            item_id: "reasoning-1".to_string(),
            text: "partial reasoning".to_string(),
            content_index: 0,
        },
        RuntimeAgentEvent::TextStart {
            item_id: "message-1".to_string(),
        },
        RuntimeAgentEvent::TextDelta {
            item_id: "message-1".to_string(),
            text: "partial answer".to_string(),
        },
    ] {
        emit_runtime_agent_event_with_coding_mirror_and_plan_parser(
            &event,
            &mut sink,
            &mut mirror,
            &mut parser,
            &mut reasoning_state,
        )
        .expect("partial provider output should map");
    }

    emit_reasoning_finish(&mut reasoning_state, "failed", &mut sink)
        .expect("reasoning failure should emit");
    emit_agent_message_finish(&mut parser, "failed", &mut sink)
        .expect("message failure should emit");

    let reasoning_terminal = sink
        .events
        .iter()
        .find(|event| event.event_type == "reasoning.ended")
        .expect("reasoning terminal");
    let message_terminal = sink
        .events
        .iter()
        .find(|event| event.event_type == "message.completed")
        .expect("message terminal");
    assert_eq!(reasoning_terminal.payload["status"], "failed");
    assert_eq!(message_terminal.payload["status"], "failed");
    assert_eq!(message_terminal.payload["phase"], "commentary");
}

#[test]
fn runtime_agent_failed_shell_tool_is_mirrored_to_coding_facts() {
    let mut sink = TestRuntimeEventSink::default();
    let mut mirror = coding_events::CodingEventMirror::default();

    emit_runtime_agent_event_with_coding_mirror(
        &tool_started(
            "Bash",
            "tool-failed",
            Some(json!({ "command": "cargo test -p app-server missing" })),
        ),
        &mut sink,
        &mut mirror,
    )
    .expect("tool start should emit");
    emit_runtime_agent_event_with_coding_mirror(
        &tool_completed(
            "tool-failed",
            AgentToolResult {
                success: false,
                output: "test failed".to_string(),
                error: Some("exit code 101".to_string()),
                structured_content: None,
                images: None,
                metadata: Some(HashMap::from([
                    ("exit_code".to_string(), json!(101)),
                    ("failureCategory".to_string(), json!("test_failed")),
                    (
                        "command".to_string(),
                        json!("cargo test -p app-server missing"),
                    ),
                ])),
            },
        ),
        &mut sink,
        &mut mirror,
    )
    .expect("failed tool end should emit");

    let event_types = sink
        .events
        .iter()
        .map(|event| event.event_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec![
            "item.started",
            "command.started",
            "test.started",
            "item.completed",
            "command.output",
            "command.exited",
            "test.completed"
        ]
    );
    let failed_event = sink
        .events
        .iter()
        .find(|event| event.event_type == "item.completed")
        .expect("failed canonical item event");
    assert_eq!(
        failed_event.payload["item"]["payload"]["call_id"],
        "tool-failed"
    );
    assert_eq!(failed_event.payload["item"]["status"], "failed");
    assert_eq!(
        failed_event.payload["item"]["payload"]["output"]["error"].as_str(),
        Some("exit code 101")
    );
    assert_eq!(
        failed_event.payload["item"]["payload"]["output"]["text"].as_str(),
        Some("test failed")
    );
}

#[test]
fn runtime_agent_permission_denied_fact_precedes_tool_failed_terminal() {
    let mut sink = TestRuntimeEventSink::default();
    let mut mirror = coding_events::CodingEventMirror::default();

    emit_runtime_agent_event_with_coding_mirror(
        &tool_started(
            "Bash",
            "tool-denied",
            Some(json!({ "command": "rm -rf important" })),
        ),
        &mut sink,
        &mut mirror,
    )
    .expect("tool start should emit");
    emit_runtime_agent_event_with_coding_mirror(
        &tool_completed(
            "tool-denied",
            AgentToolResult {
                success: false,
                output: String::new(),
                error: Some("policy denied this command".to_string()),
                structured_content: None,
                images: None,
                metadata: Some(HashMap::from([(
                    "reasonCode".to_string(),
                    json!("dangerous_command"),
                )])),
            },
        ),
        &mut sink,
        &mut mirror,
    )
    .expect("failed tool end should emit");

    let event_types = sink
        .events
        .iter()
        .map(|event| event.event_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec![
            "item.started",
            "command.started",
            "permission.denied",
            "item.completed",
            "command.exited"
        ]
    );
    let denied = sink
        .events
        .iter()
        .find(|event| event.event_type == "permission.denied")
        .expect("permission denied event");
    assert_eq!(denied.payload["toolCallId"].as_str(), Some("tool-denied"));
    assert_eq!(
        denied.payload["reasonCode"].as_str(),
        Some("dangerous_command")
    );
}

#[test]
fn runtime_agent_read_tool_result_is_mirrored_to_file_read() {
    let mut sink = TestRuntimeEventSink::default();
    let mut mirror = coding_events::CodingEventMirror::default();

    emit_runtime_agent_event_with_coding_mirror(
        &tool_started("Read", "tool-read", Some(json!({ "path": "src/App.tsx" }))),
        &mut sink,
        &mut mirror,
    )
    .expect("tool start should emit");
    emit_runtime_agent_event_with_coding_mirror(
        &tool_completed(
            "tool-read",
            AgentToolResult {
                success: true,
                output: "1 | export {}".to_string(),
                error: None,
                structured_content: None,
                images: None,
                metadata: Some(HashMap::from([("file_type".to_string(), json!("text"))])),
            },
        ),
        &mut sink,
        &mut mirror,
    )
    .expect("tool end should emit");

    let event_types = sink
        .events
        .iter()
        .map(|event| event.event_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec!["item.started", "item.completed", "file.read"]
    );
    assert_eq!(
        sink.events.last().expect("file read event").payload["path"].as_str(),
        Some("src/App.tsx")
    );
}

#[test]
fn runtime_agent_shell_apply_patch_is_mirrored_to_patch_lifecycle() {
    let mut sink = TestRuntimeEventSink::default();
    let mut mirror = coding_events::CodingEventMirror::default();

    emit_runtime_agent_event_with_coding_mirror(
        &tool_started(
            "Bash",
            "tool-patch-shell",
            Some(
                json!({
                    "command": "apply_patch <<'PATCH'\n*** Begin Patch\n*** Add File: notes/live.md\n+hello\n*** End Patch\nPATCH"
                }),
            ),
        ),
        &mut sink,
        &mut mirror,
    )
    .expect("tool start should emit");
    emit_runtime_agent_event_with_coding_mirror(
        &tool_completed(
            "tool-patch-shell",
            AgentToolResult {
                success: true,
                output: "ok".to_string(),
                error: None,
                structured_content: None,
                images: None,
                metadata: Some(HashMap::from([
                    ("exit_code".to_string(), json!(0)),
                    ("command".to_string(), json!("apply_patch <<'PATCH'")),
                ])),
            },
        ),
        &mut sink,
        &mut mirror,
    )
    .expect("tool end should emit");

    let event_types = sink
        .events
        .iter()
        .map(|event| event.event_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec![
            "item.started",
            "patch.started",
            "command.started",
            "item.completed",
            "command.output",
            "command.exited",
            "patch.applied",
        ]
    );
}
