use super::super::{PickerState, SessionPickerAction, SessionSelection, SessionStatus};
use super::ArchiveState;
use app_server_protocol::protocol::v2::{
    SessionSource, Thread, ThreadHistoryMode, ThreadStatus, ThreadUnarchiveResponse,
};
use std::path::PathBuf;

fn thread(id: &str) -> Thread {
    Thread {
        id: id.to_string(),
        extra: None,
        session_id: format!("session-{id}"),
        forked_from_id: None,
        parent_thread_id: None,
        preview: format!("preview-{id}"),
        ephemeral: false,
        section: None,
        section_entered_at: None,
        project_id: None,
        history_mode: ThreadHistoryMode::default(),
        model_provider: "fixture".to_string(),
        created_at: 1,
        updated_at: 1,
        recency_at: None,
        status: ThreadStatus::Idle,
        path: None,
        cwd: PathBuf::from("/workspace"),
        cli_version: "test".to_string(),
        source: SessionSource::Cli,
        can_accept_direct_input: Some(true),
        thread_source: None,
        agent_nickname: None,
        agent_role: None,
        git_info: None,
        name: None,
        turns: Vec::new(),
    }
}

fn state(threads: Vec<Thread>, status: SessionStatus) -> PickerState {
    PickerState::new(threads, SessionPickerAction::Resume, status, None, true)
}

#[test]
fn archive_request_is_deduplicated_until_completion() {
    let mut picker = state(vec![thread("one"), thread("two")], SessionStatus::Active);

    assert_eq!(
        picker.request_archive_for_selected_session(),
        Some("one".into())
    );
    assert_eq!(picker.request_archive_for_selected_session(), None);
    assert_eq!(
        picker.archive_state,
        ArchiveState::Pending {
            thread_id: "one".into()
        }
    );
}

#[test]
fn archive_failure_preserves_selected_thread_and_allows_retry() {
    let mut picker = state(vec![thread("one")], SessionStatus::Active);
    let thread_id = picker
        .request_archive_for_selected_session()
        .expect("selected thread");

    picker.handle_archive_result(thread_id.clone(), Err(anyhow::anyhow!("server refused")));

    assert_eq!(picker.threads.len(), 1);
    assert_eq!(
        picker.status_message.as_deref(),
        Some("Failed to archive session: server refused")
    );
    assert_eq!(
        picker.request_archive_for_selected_session(),
        Some(thread_id)
    );
}

#[test]
fn archive_success_removes_thread_and_clamps_selection() {
    let mut picker = state(vec![thread("one"), thread("two")], SessionStatus::Active);
    picker.selected = 1;
    let thread_id = picker
        .request_archive_for_selected_session()
        .expect("selected thread");

    picker.handle_archive_result(thread_id, Ok(()));

    assert_eq!(picker.threads.len(), 1);
    assert_eq!(picker.selected_thread_id(), Some("one"));
    assert_eq!(picker.archive_state, ArchiveState::Idle);
    assert!(picker.status_message.is_none());
}

#[test]
fn restore_request_is_deduplicated_and_returns_canonical_target() {
    let mut picker = state(vec![thread("archived")], SessionStatus::Archived);
    let thread_id = picker
        .request_unarchive_for_selected_session()
        .expect("selected archived thread");
    assert_eq!(picker.request_unarchive_for_selected_session(), None);
    assert_eq!(
        picker.archive_state,
        ArchiveState::Restoring {
            thread_id: "archived".into()
        }
    );

    let selection = picker.handle_unarchive_result(
        thread_id,
        Ok(ThreadUnarchiveResponse {
            thread: thread("archived"),
        }),
    );
    assert!(matches!(
        selection,
        Some(SessionSelection::Resume(target)) if target.thread_id == "archived"
    ));
    assert_eq!(picker.archive_state, ArchiveState::Idle);
}

#[test]
fn restore_failure_preserves_archived_thread_and_allows_retry() {
    let mut picker = state(vec![thread("archived")], SessionStatus::Archived);
    let thread_id = picker
        .request_unarchive_for_selected_session()
        .expect("selected archived thread");

    assert!(
        picker
            .handle_unarchive_result(thread_id.clone(), Err(anyhow::anyhow!("restore denied")))
            .is_none()
    );
    assert_eq!(picker.threads.len(), 1);
    assert_eq!(
        picker.status_message.as_deref(),
        Some("Failed to restore archived session: restore denied")
    );
    assert_eq!(
        picker.request_unarchive_for_selected_session(),
        Some(thread_id)
    );
}
