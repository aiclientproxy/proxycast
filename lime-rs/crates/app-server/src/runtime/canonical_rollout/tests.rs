use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use agent_protocol::{
    SessionId, Thread, ThreadHistoryChangeSet, ThreadId, ThreadStatus, ThreadTurnsView,
};
use serde_json::{json, Value};

use super::{scan_rollout, RolloutStore};

fn thread(id: &str) -> Thread {
    Thread {
        session_id: SessionId::new(format!("session-{id}")),
        thread_id: ThreadId::new(id),
        status: ThreadStatus::Idle,
        created_at_ms: 1_700_000_000_000,
        updated_at_ms: 1_700_000_000_000,
        archived: false,
        recency_at_ms: None,
        parent_thread_id: None,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
        last_task_message: None,
        agent_state: None,
        forked_from_id: None,
        preview: format!("preview-{id}"),
        model_provider: "test".to_string(),
        product: None,
        name: None,
        metadata: json!({}),
        turns: Vec::new(),
        turns_view: ThreadTurnsView::NotLoaded,
    }
}

fn setup(id: &str) -> (tempfile::TempDir, RolloutStore, Thread, PathBuf, PathBuf) {
    let temp = tempfile::tempdir().expect("tempdir");
    let agent_root = temp.path().join("agent-root");
    let store = RolloutStore::new(&agent_root);
    let source = thread(id);
    let relative_path = store.path_for_thread(&source).expect("rollout path");
    store
        .ensure_thread(&relative_path, &source)
        .expect("ensure rollout");
    let absolute_path = agent_root.join(&relative_path);
    (temp, store, source, relative_path, absolute_path)
}

fn changes(sequence: u64) -> ThreadHistoryChangeSet {
    ThreadHistoryChangeSet {
        sequence,
        ..Default::default()
    }
}

fn fingerprint(character: char) -> String {
    character.to_string().repeat(64)
}

fn append(
    store: &RolloutStore,
    relative_path: &Path,
    source: &Thread,
    sequence: u64,
    fingerprint: &str,
) -> Result<bool, String> {
    store.append_history(
        relative_path,
        source.session_id.as_str(),
        source.thread_id.as_str(),
        fingerprint,
        &changes(sequence),
    )
}

#[test]
fn append_history_does_not_recompute_earlier_history_digests() {
    let (_temp, store, source, relative_path, absolute_path) = setup("tail-only");
    append(&store, &relative_path, &source, 1, &fingerprint('a')).expect("append sequence 1");
    append(&store, &relative_path, &source, 2, &fingerprint('b')).expect("append sequence 2");

    let contents = fs::read_to_string(&absolute_path).expect("read rollout");
    let mut records = contents
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("rollout JSON"))
        .collect::<Vec<_>>();
    records[1]["content_digest"] = Value::String("0".repeat(64));
    let rewritten = records
        .iter()
        .map(|record| serde_json::to_string(record).expect("encode rollout JSON"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(&absolute_path, rewritten).expect("rewrite rollout fixture");

    assert!(
        append(&store, &relative_path, &source, 3, &fingerprint('c'))
            .expect("append from latest valid history")
    );
    assert!(scan_rollout(&absolute_path)
        .expect_err("cold scan must retain full integrity validation")
        .contains("invalid rollout history record"));
}

#[test]
fn append_history_is_idempotent_for_the_latest_sequence() {
    let (_temp, store, source, relative_path, _absolute_path) = setup("idempotent");
    let expected_fingerprint = fingerprint('a');
    assert!(
        append(&store, &relative_path, &source, 1, &expected_fingerprint).expect("initial append")
    );
    assert!(
        !append(&store, &relative_path, &source, 1, &expected_fingerprint)
            .expect("idempotent append")
    );
}

#[test]
fn append_history_rejects_latest_collision_and_stale_sequence() {
    let (_temp, store, source, relative_path, _absolute_path) = setup("ordering");
    append(&store, &relative_path, &source, 2, &fingerprint('a')).expect("append sequence 2");

    let collision = append(&store, &relative_path, &source, 2, &fingerprint('b'))
        .expect_err("same sequence with another fingerprint must fail");
    assert!(collision.contains("sequence collision at 2"));

    let stale = append(&store, &relative_path, &source, 1, &fingerprint('c'))
        .expect_err("older sequence must fail");
    assert!(stale.contains("sequence 1 is stale"));
}

#[test]
fn append_history_finds_latest_history_before_trailing_metadata() {
    let (_temp, store, source, relative_path, _absolute_path) = setup("metadata-tail");
    let expected_fingerprint = fingerprint('a');
    append(&store, &relative_path, &source, 1, &expected_fingerprint).expect("append history");
    let mut next = source.clone();
    next.updated_at_ms += 1;
    next.metadata = json!({"blob": "x".repeat(100_000)});
    store
        .append_metadata(&relative_path, &source, &next)
        .expect("append metadata");

    assert!(
        !append(&store, &relative_path, &source, 1, &expected_fingerprint)
            .expect("latest history remains idempotent")
    );
    assert!(
        append(&store, &relative_path, &source, 2, &fingerprint('b'))
            .expect("append after trailing metadata")
    );
}

#[test]
fn append_history_rejects_a_corrupted_latest_history() {
    let (_temp, store, source, relative_path, absolute_path) = setup("corrupt-latest");
    append(&store, &relative_path, &source, 1, &fingerprint('a')).expect("append history");

    let contents = fs::read_to_string(&absolute_path).expect("read rollout");
    let mut records = contents
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("rollout JSON"))
        .collect::<Vec<_>>();
    records[1]["content_digest"] = Value::String("0".repeat(64));
    let rewritten = records
        .iter()
        .map(|record| serde_json::to_string(record).expect("encode rollout JSON"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(&absolute_path, rewritten).expect("rewrite rollout fixture");

    let error = append(&store, &relative_path, &source, 2, &fingerprint('b'))
        .expect_err("corrupted latest history must fail closed");
    assert!(error.contains("invalid rollout history record"));
}

#[test]
fn append_history_preserves_a_valid_unterminated_record() {
    let (_temp, store, source, relative_path, absolute_path) = setup("valid-crash-tail");
    append(&store, &relative_path, &source, 1, &fingerprint('a')).expect("append sequence 1");
    append(&store, &relative_path, &source, 2, &fingerprint('b')).expect("append sequence 2");

    let mut contents = fs::read(&absolute_path).expect("read rollout");
    assert_eq!(contents.pop(), Some(b'\n'));
    fs::write(&absolute_path, contents).expect("remove final newline");

    append(&store, &relative_path, &source, 3, &fingerprint('c'))
        .expect("append after valid crash tail");
    let repaired = fs::read_to_string(&absolute_path).expect("read repaired rollout");
    assert!(repaired.ends_with('\n'));
    assert_eq!(
        scan_rollout(&absolute_path)
            .expect("scan repaired rollout")
            .history
            .iter()
            .map(|record| record.sequence)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
}

#[test]
fn append_history_truncates_only_a_malformed_crash_tail_and_keeps_sequence_gap() {
    let (_temp, store, source, relative_path, absolute_path) = setup("partial-crash-tail");
    append(&store, &relative_path, &source, 1, &fingerprint('a')).expect("append sequence 1");
    fs::OpenOptions::new()
        .append(true)
        .open(&absolute_path)
        .and_then(|mut file| file.write_all(b"{\"type\":\"thread_history\",\"sequence\":2"))
        .expect("append partial sequence 2");

    append(&store, &relative_path, &source, 3, &fingerprint('c'))
        .expect("append after malformed crash tail");
    let repaired = fs::read_to_string(&absolute_path).expect("read repaired rollout");
    assert!(!repaired.contains("\"sequence\":2"));
    assert!(repaired.ends_with('\n'));
    assert_eq!(
        scan_rollout(&absolute_path)
            .expect("scan repaired rollout")
            .history
            .iter()
            .map(|record| record.sequence)
            .collect::<Vec<_>>(),
        vec![1, 3]
    );
}

#[test]
fn cold_snapshot_repairs_a_malformed_crash_tail_before_materialization() {
    let (_temp, store, source, relative_path, absolute_path) = setup("cold-crash-tail");
    append(&store, &relative_path, &source, 1, &fingerprint('a')).expect("append sequence 1");
    fs::OpenOptions::new()
        .append(true)
        .open(&absolute_path)
        .and_then(|mut file| file.write_all(b"{\"type\":\"thread_history\""))
        .expect("append crash tail");

    let restarted = RolloutStore::new(store.agent_root.clone());
    let snapshots = restarted.snapshots().expect("repair cold rollout");
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].history.len(), 1);
    assert_eq!(snapshots[0].history[0].sequence, 1);
    assert!(fs::read(&absolute_path)
        .expect("read repaired rollout")
        .ends_with(b"\n"));

    append(&restarted, &relative_path, &source, 3, &fingerprint('c'))
        .expect("continue after cold repair");
}

#[test]
fn crash_tail_repair_leaves_middle_corruption_untouched() {
    let (_temp, store, source, relative_path, absolute_path) = setup("middle-corruption");
    append(&store, &relative_path, &source, 1, &fingerprint('a')).expect("append sequence 1");
    fs::OpenOptions::new()
        .append(true)
        .open(&absolute_path)
        .and_then(|mut file| file.write_all(b"{\"broken\":}\n{\"type\":\"thread_history\""))
        .expect("append middle corruption and crash tail");
    let before = fs::read(&absolute_path).expect("read corrupted rollout");

    let error = store
        .snapshots()
        .expect_err("middle corruption must fail closed");
    assert!(error.contains("invalid rollout JSONL record"));
    assert_eq!(fs::read(&absolute_path).expect("reread rollout"), before);
}

#[test]
fn crash_tail_repair_rejects_a_complete_semantically_invalid_record() {
    let (_temp, store, source, relative_path, absolute_path) = setup("invalid-complete-tail");
    append(&store, &relative_path, &source, 1, &fingerprint('a')).expect("append sequence 1");
    let invalid = serde_json::to_vec(&json!({
        "type": "thread_history",
        "schema_version": 1,
        "session_id": source.session_id.as_str(),
        "thread_id": "another-thread",
        "sequence": 2,
        "fingerprint": fingerprint('b'),
        "content_digest": "0".repeat(64),
        "changes": changes(2),
    }))
    .expect("encode invalid tail");
    fs::OpenOptions::new()
        .append(true)
        .open(&absolute_path)
        .and_then(|mut file| file.write_all(&invalid))
        .expect("append invalid complete tail");
    let before = fs::read(&absolute_path).expect("read invalid rollout");

    let error = append(&store, &relative_path, &source, 3, &fingerprint('c'))
        .expect_err("semantic divergence must fail closed");
    assert!(error.contains("invalid rollout history record"));
    assert_eq!(fs::read(&absolute_path).expect("reread rollout"), before);
}
