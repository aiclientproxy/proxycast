use super::*;
use app_server_protocol::AgentEvent;
use serde_json::json;
use std::fs;
use std::io::Read;
use tool_runtime::tool_lifecycle::{CodeCellRuntimeStatus, CodeCellTraceEvent};
use zip::ZipArchive;

fn traced_event(trace_id: &str, session_id: &str, sequence: u64) -> AgentEvent {
    AgentEvent {
        event_id: format!("evt-{sequence}"),
        sequence,
        session_id: session_id.to_string(),
        thread_id: Some("thread-a".to_string()),
        turn_id: Some("turn-a".to_string()),
        event_type: "message.delta".to_string(),
        timestamp: "2026-06-14T00:00:00.000Z".to_string(),
        payload: json!({
            "text": "secret assistant text",
            "trace_id": trace_id,
            "server_event_emitted_at": 1_780_000_000_000i64,
            "trace": {
                "schemaVersion": 1,
                "checkpoint": "app_server.message_delta.emitted",
                "traceId": trace_id,
                "runId": "run-a",
                "requestId": "request-a",
                "w3cTraceId": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "w3cTraceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"
            }
        }),
    }
}

#[test]
fn trace_writer_persists_summary_only_events() {
    let temp = tempfile::tempdir().expect("tempdir");
    let writer = TraceEventWriter::new(temp.path()).expect("writer");

    writer
        .append_agent_events(&[traced_event("trace-a", "session-a", 1)])
        .expect("append trace");
    writer
        .append_agent_events(&[traced_event("trace-a", "session-a", 2)])
        .expect("append trace again");

    let records = writer
        .read_raw_trace_events("session-a", "trace-a")
        .expect("records");
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].event.seq, 1);
    assert_eq!(records[1].event.seq, 2);
    assert_eq!(records[0].event.trace_id, "trace-a");
    assert_eq!(
        records[0].event.checkpoint,
        "app_server.message_delta.emitted"
    );
    assert_eq!(records[0].event.metrics["text_chars"], json!(21));
    assert_eq!(
        records[0].event.metrics["w3c_trace_id"],
        json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    );
    assert_eq!(
        records[0].event.metrics["w3c_traceparent"],
        json!("00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01")
    );
    assert_eq!(records[0].event.redaction.mode, "summary_only");
    let raw = fs::read_to_string(&records[0].path).expect("trace file");
    assert!(!raw.contains("secret assistant text"));
    assert!(!raw.contains("\"text\""));

    let list = writer
        .list_trace_events(DiagnosticsTraceListParams {
            session_id: Some("session-a".to_string()),
            limit: None,
        })
        .expect("list traces");
    assert!(list.available);
    assert_eq!(list.trace_root, None);
    assert_eq!(list.traces.len(), 1);
    assert_eq!(list.traces[0].trace_id, "trace-a");
    assert_eq!(
        list.traces[0].path,
        "sessions/session_session-a/trace_trace-a.jsonl"
    );
    assert_eq!(list.traces[0].event_count, 2);
    assert!(!list.traces[0].path.starts_with('/'));

    let read = writer
        .read_trace_events(DiagnosticsTraceReadParams {
            session_id: "session-a".to_string(),
            trace_id: "trace-a".to_string(),
            max_events: Some(1),
        })
        .expect("read trace");
    assert!(read.available);
    assert_eq!(read.trace.expect("trace").event_count, 2);
    assert_eq!(read.events.len(), 1);
    assert_eq!(read.events[0].redaction.mode, "summary_only");
}

#[test]
fn trace_writer_clear_session_is_scoped_and_idempotent() {
    let temp = tempfile::tempdir().expect("tempdir");
    let writer = TraceEventWriter::new(temp.path()).expect("writer");
    writer
        .append_agent_events(&[
            traced_event("trace-a", "session-a", 1),
            traced_event("trace-b", "session-b", 1),
        ])
        .expect("append traces");

    writer.clear_session("session-a").expect("clear session");
    writer
        .clear_session("session-a")
        .expect("clear missing session");

    assert!(writer
        .read_raw_trace_events("session-a", "trace-a")
        .expect("cleared trace")
        .is_empty());
    assert_eq!(
        writer
            .read_raw_trace_events("session-b", "trace-b")
            .expect("retained trace")
            .len(),
        1
    );
}

#[test]
fn trace_writer_exports_summary_only_zip() {
    let temp = tempfile::tempdir().expect("tempdir");
    let trace_root = temp.path().join("trace-store");
    let export_root = temp.path().join("exports");
    let writer = TraceEventWriter::new(&trace_root).expect("writer");

    writer
        .append_agent_events(&[traced_event("trace-a", "session-a", 1)])
        .expect("append trace");

    let response = writer
        .export_trace_events_to_directory(
            DiagnosticsTraceExportParams {
                session_id: "session-a".to_string(),
                trace_id: "trace-a".to_string(),
            },
            export_root.clone(),
        )
        .expect("export trace");

    assert!(response.available);
    assert!(response.exported);
    assert_eq!(
        response.output_directory,
        Some(export_root.to_string_lossy().to_string())
    );
    assert_eq!(
        response.included_sections,
        vec![
            "meta/manifest.json",
            "meta/trace-summary.json",
            "trace/events.jsonl",
            "README.txt"
        ]
    );
    assert!(response
        .omitted_sections
        .iter()
        .any(|section| section == "assistant delta text"));
    assert_eq!(response.redaction.mode, "summary_only");

    let bundle_path = response.bundle_path.expect("bundle path");
    assert!(std::path::Path::new(&bundle_path).is_file());
    let bundle = fs::File::open(&bundle_path).expect("open export zip");
    let mut archive = ZipArchive::new(bundle).expect("read export zip");
    let mut names = Vec::new();
    for index in 0..archive.len() {
        names.push(
            archive
                .by_index(index)
                .expect("zip entry")
                .name()
                .to_string(),
        );
    }
    assert_eq!(
        names,
        vec![
            "meta/manifest.json",
            "meta/trace-summary.json",
            "trace/events.jsonl",
            "README.txt"
        ]
    );

    let mut manifest = String::new();
    archive
        .by_name("meta/manifest.json")
        .expect("manifest")
        .read_to_string(&mut manifest)
        .expect("read manifest");
    assert!(manifest.contains("\"summaryOnlyTraceEventsIncluded\": true"));
    assert!(manifest.contains("\"prompt text\""));

    let mut events = String::new();
    archive
        .by_name("trace/events.jsonl")
        .expect("events")
        .read_to_string(&mut events)
        .expect("read events");
    assert!(events.contains("\"checkpoint\":\"app_server.message_delta.emitted\""));
    assert!(events.contains("\"text_chars\":21"));
    assert!(!events.contains("secret assistant text"));
    assert!(!events.contains("\"text\""));
}

#[test]
fn trace_writer_does_not_export_missing_trace() {
    let temp = tempfile::tempdir().expect("tempdir");
    let writer = TraceEventWriter::new(temp.path().join("trace-store")).expect("writer");

    let response = writer
        .export_trace_events_to_directory(
            DiagnosticsTraceExportParams {
                session_id: "session-a".to_string(),
                trace_id: "missing".to_string(),
            },
            temp.path().join("exports"),
        )
        .expect("export missing trace");

    assert!(response.available);
    assert!(!response.exported);
    assert_eq!(response.trace, None);
    assert_eq!(response.bundle_path, None);
    assert_eq!(response.output_directory, None);
    assert_eq!(response.generated_at, None);
    assert!(response.included_sections.is_empty());
    assert!(response
        .omitted_sections
        .iter()
        .any(|section| section == "unparsed raw JSONL bytes"));
}

#[test]
fn trace_writer_keeps_recent_trace_files_per_session() {
    let temp = tempfile::tempdir().expect("tempdir");
    let writer = TraceEventWriter::new(temp.path()).expect("writer");

    for index in 0..(TRACE_EVENT_MAX_FILES_PER_SESSION + 2) {
        writer
            .append_agent_events(&[traced_event(
                &format!("trace-{index:03}"),
                "session-a",
                index as u64 + 1,
            )])
            .expect("append trace");
    }

    assert!(writer
        .read_raw_trace_events("session-a", "trace-000")
        .expect("old trace")
        .is_empty());
    assert!(
        writer
            .read_trace_events(DiagnosticsTraceReadParams {
                session_id: "session-a".to_string(),
                trace_id: format!("trace-{:03}", TRACE_EVENT_MAX_FILES_PER_SESSION + 1),
                max_events: None,
            })
            .expect("new trace")
            .events
            .len()
            == 1
    );
}

#[test]
fn code_cell_trace_replays_pending_source_nested_wait_and_output_links() {
    let temp = tempfile::tempdir().expect("tempdir");
    let writer = TraceEventWriter::new(temp.path()).expect("writer");
    let append = |event| {
        writer
            .append_code_cell_trace_event("session-code", "thread-code", event)
            .expect("append CodeCell trace")
    };

    append(CodeCellTraceEvent::Started {
        turn_id: "turn-1".to_string(),
        runtime_cell_id: "cell-1".to_string(),
        model_visible_call_id: "call-exec".to_string(),
        source_js: "await tools.read({ path: 'README.md' });".to_string(),
    });
    append(CodeCellTraceEvent::InitialResponse {
        turn_id: "turn-1".to_string(),
        runtime_cell_id: "cell-1".to_string(),
        status: CodeCellRuntimeStatus::Yielded,
        response_chars: 7,
    });
    append(CodeCellTraceEvent::NestedToolStarted {
        turn_id: "turn-1".to_string(),
        runtime_cell_id: "cell-1".to_string(),
        tool_call_id: "code-mode-read-1".to_string(),
        runtime_tool_call_id: "read-1".to_string(),
        tool_name: "read".to_string(),
    });
    append(CodeCellTraceEvent::NestedToolEnded {
        turn_id: "turn-1".to_string(),
        runtime_cell_id: "cell-1".to_string(),
        tool_call_id: "code-mode-read-1".to_string(),
        status: "completed".to_string(),
    });

    let pending = writer
        .read_code_cell_trace("session-code", "thread-code")
        .expect("pending projection");
    assert!(pending.code_cells.is_empty());
    assert_eq!(pending.pending_code_cell_ids, vec!["thread-code:call-exec"]);

    append(CodeCellTraceEvent::SourceItemObserved {
        turn_id: "turn-1".to_string(),
        model_visible_call_id: "call-exec".to_string(),
        source_item_id: "item_call-exec".to_string(),
    });
    append(CodeCellTraceEvent::OutputItemObserved {
        turn_id: "turn-1".to_string(),
        runtime_cell_id: "cell-1".to_string(),
        output_item_id: "item_call-exec".to_string(),
    });
    append(CodeCellTraceEvent::WaitToolObserved {
        turn_id: "turn-2".to_string(),
        runtime_cell_id: "cell-1".to_string(),
        tool_call_id: "wait-call-1".to_string(),
    });
    append(CodeCellTraceEvent::Ended {
        turn_id: "turn-2".to_string(),
        runtime_cell_id: "cell-1".to_string(),
        status: CodeCellRuntimeStatus::Completed,
        response_chars: 4,
    });

    let projection = writer
        .read_code_cell_trace("session-code", "thread-code")
        .expect("CodeCell projection");
    assert!(projection.pending_code_cell_ids.is_empty());
    let cell = &projection.code_cells["thread-code:call-exec"];
    assert_eq!(cell.code_cell_id, "thread-code:call-exec");
    assert_eq!(cell.model_visible_call_id, "call-exec");
    assert_eq!(cell.thread_id, "thread-code");
    assert_eq!(cell.started_by_turn_id, "turn-1");
    assert_eq!(cell.source_item_id, "item_call-exec");
    assert_eq!(cell.output_item_ids, vec!["item_call-exec"]);
    assert_eq!(cell.runtime_cell_id, "cell-1");
    assert_eq!(cell.runtime_status, CodeCellRuntimeStatus::Completed);
    assert_eq!(cell.nested_tool_call_ids, vec!["code-mode-read-1"]);
    assert_eq!(cell.nested_tool_statuses["code-mode-read-1"], "completed");
    assert_eq!(cell.wait_tool_call_ids, vec!["wait-call-1"]);
    assert!(cell.initial_response_seq.is_some());
    assert!(cell.yielded_seq.is_some());
    assert!(cell.ended_seq.is_some());
    assert_eq!(cell.source_js_chars, 40);
    assert_eq!(cell.source_js_sha256.len(), 64);
    assert!(cell.started_seq < cell.ended_seq.expect("ended sequence"));

    let raw = writer
        .read_raw_trace_events("session-code", "code-cell-thread-code")
        .expect("raw CodeCell trace");
    assert_eq!(raw.len(), 8);
    let raw_text = fs::read_to_string(&raw[0].path).expect("raw trace file");
    assert!(!raw_text.contains("README.md"));
    assert!(!raw_text.contains("await tools.read"));
    assert!(raw_text.contains("source_js_sha256"));

    let diagnostics = writer
        .read_trace_events(DiagnosticsTraceReadParams {
            session_id: "session-code".to_string(),
            trace_id: "code-cell-thread-code".to_string(),
            max_events: None,
        })
        .expect("diagnostics CodeCell trace");
    assert_eq!(diagnostics.events.len(), 8);
    assert!(diagnostics
        .events
        .iter()
        .any(|event| event.event_type == "code_cell.ended"));
    assert!(!serde_json::to_string(&diagnostics)
        .expect("serialize diagnostics trace")
        .contains("README.md"));
}

#[test]
fn interrupted_turn_closes_running_code_cell_and_rejects_late_terminal() {
    let temp = tempfile::tempdir().expect("tempdir");
    let writer = TraceEventWriter::new(temp.path()).expect("writer");
    for event in [
        CodeCellTraceEvent::SourceItemObserved {
            turn_id: "turn-cancel".to_string(),
            model_visible_call_id: "call-cancel".to_string(),
            source_item_id: "call-cancel".to_string(),
        },
        CodeCellTraceEvent::Started {
            turn_id: "turn-cancel".to_string(),
            runtime_cell_id: "cell-cancel".to_string(),
            model_visible_call_id: "call-cancel".to_string(),
            source_js: "await new Promise(() => {});".to_string(),
        },
        CodeCellTraceEvent::InitialResponse {
            turn_id: "turn-cancel".to_string(),
            runtime_cell_id: "cell-cancel".to_string(),
            status: CodeCellRuntimeStatus::Yielded,
            response_chars: 0,
        },
    ] {
        writer
            .append_code_cell_trace_event("session-cancel", "thread-cancel", event)
            .expect("append running cell trace");
    }
    writer
        .close_code_cells_for_turn(
            "session-cancel",
            "thread-cancel",
            "turn-cancel",
            CodeCellRuntimeStatus::Terminated,
        )
        .expect("close interrupted cell");

    let late = writer.append_code_cell_trace_event(
        "session-cancel",
        "thread-cancel",
        CodeCellTraceEvent::Ended {
            turn_id: "turn-cancel".to_string(),
            runtime_cell_id: "cell-cancel".to_string(),
            status: CodeCellRuntimeStatus::Completed,
            response_chars: 4,
        },
    );
    assert!(late
        .expect_err("late terminal must fail closed")
        .contains("terminal cell"));

    let projection = writer
        .read_code_cell_trace("session-cancel", "thread-cancel")
        .expect("terminal projection");
    let cell = &projection.code_cells["thread-cancel:call-cancel"];
    assert_eq!(cell.runtime_status, CodeCellRuntimeStatus::Terminated);
    assert!(cell.ended_seq.is_some());
    assert_eq!(
        writer
            .read_raw_trace_events("session-cancel", "code-cell-thread-cancel")
            .expect("raw terminal trace")
            .len(),
        4
    );
}
