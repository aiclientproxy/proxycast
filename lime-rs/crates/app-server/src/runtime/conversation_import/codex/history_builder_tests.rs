use super::*;

fn tool_event(
    phase: CodexToolPhase,
    call_id: Option<&str>,
    output: Option<Value>,
) -> CodexRolloutEvent {
    CodexRolloutEvent::Tool(CodexToolCall {
        phase,
        call_id: call_id.map(str::to_string),
        name: Some("read_file".to_string()),
        arguments: Some(json!({ "path": "src/lib.rs" })),
        output,
        source: CodexToolSource {
            source_client: Some("codex".to_string()),
            source_event_type: Some("test_tool_event".to_string()),
            ..CodexToolSource::default()
        },
    })
}

#[test]
fn normalizer_deduplicates_tool_start_and_terminal() {
    let mut normalizer = CodexHistoryBuilder::new();
    let start = tool_event(CodexToolPhase::Started, Some("call-1"), None);
    let terminal = tool_event(
        CodexToolPhase::Completed,
        Some("call-1"),
        Some(Value::String("done".to_string())),
    );

    let mut events = normalizer.push(start.clone());
    events.extend(normalizer.push(start));
    events.extend(normalizer.push(terminal.clone()));
    events.extend(normalizer.push(terminal));
    events.extend(normalizer.finish());

    assert_eq!(
        events
            .iter()
            .filter_map(CodexRolloutEvent::tool_call)
            .filter(|tool| tool.phase == CodexToolPhase::Started)
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter_map(CodexRolloutEvent::tool_call)
            .filter(|tool| tool.phase == CodexToolPhase::Completed)
            .count(),
        1
    );
}

#[test]
fn terminal_only_synthetic_start_does_not_copy_terminal_state() {
    let mut terminal = match tool_event(
        CodexToolPhase::Failed,
        Some("call-failed"),
        Some(json!({ "error": "denied" })),
    ) {
        CodexRolloutEvent::Tool(tool) => tool,
        CodexRolloutEvent::Runtime { .. } => unreachable!(),
    };
    terminal.source.success = Some(false);
    terminal.source.failure_category = Some("permission_denied".to_string());
    terminal.source.source_provenance = Some(json!({ "sourceEventSeq": 11 }));

    let mut normalizer = CodexHistoryBuilder::new();
    let events = normalizer.push(CodexRolloutEvent::Tool(terminal));
    let start = events[0].tool_call().expect("synthetic start");
    let terminal = events[1].tool_call().expect("terminal");

    assert_eq!(start.phase, CodexToolPhase::Started);
    assert!(start.source.synthetic);
    assert!(start.source.success.is_none());
    assert!(start.source.failure_category.is_none());
    assert!(start.source.source_provenance.is_none());
    assert_eq!(terminal.source.success, Some(false));
    assert_eq!(
        terminal.source.failure_category.as_deref(),
        Some("permission_denied")
    );
}

#[test]
fn missing_tool_call_ids_are_unique_and_structured_output_is_preserved() {
    let mut first = match tool_event(
        CodexToolPhase::Completed,
        None,
        Some(json!([{ "type": "input_text", "text": "first" }])),
    ) {
        CodexRolloutEvent::Tool(tool) => tool,
        CodexRolloutEvent::Runtime { .. } => unreachable!(),
    };
    first.source.source_provenance = Some(json!({ "sourceEventSeq": 21 }));
    let mut second = first.clone();
    second.source.source_provenance = Some(json!({ "sourceEventSeq": 22 }));

    let mut normalizer = CodexHistoryBuilder::new();
    let mut events = normalizer.push(CodexRolloutEvent::Tool(first));
    events.extend(normalizer.push(CodexRolloutEvent::Tool(second)));
    let ids = events
        .iter()
        .filter_map(CodexRolloutEvent::tool_call)
        .filter(|tool| tool.phase == CodexToolPhase::Completed)
        .filter_map(|tool| tool.call_id.clone())
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["imported-tool-21", "imported-tool-22"]);

    let lowered = project_rollout_events_to_canonical(&events, "session-1", "thread-1", "turn-1");
    let completed = lowered
        .iter()
        .find(|event| event.event_type() == "item.completed")
        .and_then(CodexRolloutEvent::payload)
        .expect("canonical completed item");
    assert_eq!(
        completed["item"]["payload"]["output"]["structuredContent"],
        json!([{ "type": "input_text", "text": "first" }])
    );
    assert_eq!(completed["item"]["payload"]["output"]["text"], "first");
}

#[test]
fn imported_structured_tool_output_is_reduced_to_historical_summary() {
    let mut terminal = match tool_event(
        CodexToolPhase::Completed,
        Some("call-large-output"),
        Some(json!({
            "text": "Script completed\nOutput:\nready",
            "structuredContent": [{"type": "input_text", "text": "large raw payload"}]
        })),
    ) {
        CodexRolloutEvent::Tool(tool) => tool,
        CodexRolloutEvent::Runtime { .. } => unreachable!(),
    };
    terminal.source.imported = true;
    terminal.source.output_bytes = Some(42_000_000);

    let lowered = project_rollout_events_to_canonical(
        &[CodexRolloutEvent::Tool(terminal)],
        "session-1",
        "thread-1",
        "turn-1",
    );
    let completed = lowered
        .iter()
        .find(|event| event.event_type() == "item.completed")
        .and_then(CodexRolloutEvent::payload)
        .expect("canonical completed item");
    let output = &completed["item"]["payload"]["output"];
    assert_eq!(output["text"], "Script completed\nOutput:\nready");
    assert_eq!(output["structuredContent"], Value::Null);
    assert_eq!(output["outputBytes"], Value::Null);
}

#[test]
fn read_file_terminal_projects_a_file_changed_event_for_history_artifacts() {
    let mut normalizer = CodexHistoryBuilder::new();
    let start = CodexRolloutEvent::Tool(CodexToolCall {
        phase: CodexToolPhase::Started,
        call_id: Some("call-read".to_string()),
        name: Some("read_file".to_string()),
        arguments: Some(json!({ "path": "/workspace/docs/readme.md" })),
        output: None,
        source: CodexToolSource::default(),
    });
    let terminal = CodexRolloutEvent::Tool(CodexToolCall {
        phase: CodexToolPhase::Completed,
        call_id: Some("call-read".to_string()),
        name: Some("read_file".to_string()),
        arguments: Some(json!({ "path": "/workspace/docs/readme.md" })),
        output: Some(json!("readme content")),
        source: CodexToolSource::default(),
    });

    let mut events = normalizer.push(start);
    events.extend(normalizer.push(terminal));

    let file_event = events
        .iter()
        .find(|event| event.event_type() == "file.changed")
        .expect("read_file should produce a history file artifact event");
    let payload = file_event.payload().expect("file event payload");
    assert_eq!(payload["path"], "/workspace/docs/readme.md");
    assert_eq!(payload["content"], "readme content");
    assert_eq!(payload["eventClass"], "file.read");
    assert_eq!(payload["operation"], "read");
    assert_eq!(payload["toolName"], "read_file");
}
