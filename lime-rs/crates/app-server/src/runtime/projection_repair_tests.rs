use super::event_log::{EventLogIssue, EventLogWriter};
use super::projection_repair::{ProjectionRepair, ProjectionRepairStatus};
use super::projection_store::ProjectionStore;
use super::StorageRoots;
use app_server_protocol::AgentEvent;
use serde_json::json;
use std::fs;
use std::io::Write;

fn event(sequence: u64, event_type: &str) -> AgentEvent {
    AgentEvent {
        event_id: format!("evt-{sequence}"),
        sequence,
        session_id: "sess_repair_audit".to_string(),
        thread_id: Some("thread_repair_audit".to_string()),
        turn_id: Some("turn_repair_audit".to_string()),
        event_type: event_type.to_string(),
        timestamp: "2026-07-12T00:00:00.000Z".to_string(),
        payload: json!({ "sequence": sequence }),
    }
}

fn setup() -> (
    tempfile::TempDir,
    StorageRoots,
    EventLogWriter,
    ProjectionStore,
) {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log = EventLogWriter::new(&roots.event_log_root).expect("event log");
    let projection = ProjectionStore::initialize(&roots.projection_db_path).expect("projection");
    (temp, roots, event_log, projection)
}

fn event_log_path(roots: &StorageRoots) -> std::path::PathBuf {
    roots
        .event_log_root
        .join("sessions/session_sess_repair_audit.jsonl")
}

fn write_event_log_fixture(roots: &StorageRoots, events: &[AgentEvent]) {
    let path = event_log_path(roots);
    fs::create_dir_all(path.parent().expect("event log parent")).expect("create event log parent");
    let mut contents = events
        .iter()
        .map(|event| serde_json::to_string(event).expect("serialize event"))
        .collect::<Vec<_>>()
        .join("\n");
    contents.push('\n');
    fs::write(path, contents).expect("write event log fixture");
}

#[test]
fn repair_audit_records_tail_truncation_and_valid_prefix_fingerprint() {
    let (_temp, roots, event_log, projection) = setup();
    event_log
        .append(&event(1, "turn.accepted"))
        .expect("append accepted");
    event_log
        .append(&event(2, "turn.completed"))
        .expect("append completed");
    let path = event_log_path(&roots);
    let valid_len = fs::metadata(&path).expect("metadata").len();
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("open tail")
        .write_all(br#"{"eventId":"evt-3"#)
        .expect("write tail");
    let original_len = fs::metadata(&path).expect("original metadata").len();

    let repair = ProjectionRepair::new(event_log.clone(), projection.clone());
    let audit = repair
        .repair_session_with_audit("sess_repair_audit")
        .expect("repair with audit");

    assert_eq!(
        audit.status,
        ProjectionRepairStatus::RebuiltAfterTailTruncation
    );
    assert_eq!(audit.records_scanned, 2);
    assert_eq!(audit.events_applied, 2);
    assert_eq!(audit.last_valid_offset, valid_len);
    assert_eq!(audit.file_len, original_len);
    assert_eq!(audit.fingerprint.len(), 64);
    assert!(matches!(
        audit.issue,
        Some(EventLogIssue::MalformedTail { .. })
    ));
    assert_eq!(
        fs::metadata(path).expect("repaired metadata").len(),
        valid_len
    );
    assert_eq!(
        projection
            .read_watermark("sess_repair_audit")
            .expect("watermark")
            .expect("watermark row")
            .last_sequence,
        2
    );
}

#[test]
fn repair_rejects_missing_middle_without_mutating_projection() {
    let (_temp, roots, event_log, projection) = setup();
    write_event_log_fixture(
        &roots,
        &[event(1, "turn.accepted"), event(3, "turn.completed")],
    );

    let repair = ProjectionRepair::new(event_log, projection.clone());
    let error = repair
        .repair_session_with_audit("sess_repair_audit")
        .expect_err("gap must fail closed");

    assert!(error.contains("SequenceGap"));
    assert_eq!(
        projection
            .read_watermark("sess_repair_audit")
            .expect("watermark"),
        None
    );
}

#[test]
fn repair_rejects_equal_sequence_divergence() {
    let (_temp, roots, event_log, projection) = setup();
    let first = event(1, "turn.accepted");
    let mut divergent = event(1, "turn.completed");
    divergent.event_id = "evt-divergent".to_string();
    write_event_log_fixture(&roots, &[first, divergent]);

    let repair = ProjectionRepair::new(event_log, projection);
    let error = repair
        .repair_session_with_audit("sess_repair_audit")
        .expect_err("divergence must fail closed");

    assert!(error.contains("EqualSequenceDivergence"));
}

#[test]
fn read_repair_rejects_log_shorter_than_projection_watermark() {
    let (_temp, _roots, event_log, projection) = setup();
    let first = event(1, "turn.accepted");
    let second = event(2, "turn.completed");
    event_log.append(&first).expect("append first");
    projection.apply_event(&first).expect("project first");
    projection.apply_event(&second).expect("project second");

    let repair = ProjectionRepair::new(event_log, projection);
    let error = repair
        .read_repaired_session("sess_repair_audit", None)
        .expect_err("shorter log must fail closed");

    assert!(error.contains("shorter event log"));
    assert!(error.contains("projection_sequence=2"));
    assert!(error.contains("event_log_sequence=1"));
}

#[test]
fn malformed_middle_log_is_neither_truncated_nor_projected() {
    let (_temp, roots, event_log, projection) = setup();
    let path = event_log_path(&roots);
    fs::create_dir_all(path.parent().expect("parent")).expect("create parent");
    let first = serde_json::to_string(&event(1, "turn.accepted")).expect("first");
    let third = serde_json::to_string(&event(3, "turn.completed")).expect("third");
    fs::write(&path, format!("{first}\n{{malformed}}\n{third}\n")).expect("write log");
    let original = fs::read(&path).expect("original");

    let repair = ProjectionRepair::new(event_log, projection.clone());
    let error = repair
        .repair_session_with_audit("sess_repair_audit")
        .expect_err("middle corruption must fail closed");

    assert!(error.contains("MalformedRecord"));
    assert_eq!(fs::read(path).expect("after"), original);
    assert_eq!(
        projection
            .read_watermark("sess_repair_audit")
            .expect("watermark"),
        None
    );
}
