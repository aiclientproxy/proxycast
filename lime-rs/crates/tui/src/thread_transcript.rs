//! Canonical persisted thread transcript projection for TUI resume/preview.
//!
//! Codex keeps this owner separate from the interactive chat widget. Lime does
//! the same, but its only source is App Server's v2 `Thread/Turn/Item` model;
//! no rollout files or private local history database are consulted.

use std::io;

use app_server_client::RequestHandle;
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_READ, Thread, ThreadReadParams, ThreadReadResponse,
};

use crate::app_server_session::AppServerSession;
use crate::projection::{ConversationProjection, TranscriptEntry};

#[allow(dead_code)]
pub(crate) async fn load_session_transcript(
    app_server: &AppServerSession,
    thread_id: impl Into<String>,
) -> io::Result<Vec<TranscriptEntry>> {
    let response = app_server
        .read_thread(thread_id, true)
        .await
        .map_err(io::Error::other)?;
    Ok(thread_to_transcript_entries(response.thread))
}

pub(crate) async fn load_session_transcript_with_handle(
    request_handle: RequestHandle,
    thread_id: impl Into<String>,
) -> io::Result<Vec<TranscriptEntry>> {
    let response: ThreadReadResponse = request_handle
        .request(
            METHOD_THREAD_READ,
            ThreadReadParams {
                thread_id: thread_id.into(),
                include_turns: true,
            },
        )
        .await
        .map_err(io::Error::other)?;
    Ok(thread_to_transcript_entries(response.thread))
}

pub(crate) fn thread_to_transcript_entries(thread: Thread) -> Vec<TranscriptEntry> {
    let mut projection = ConversationProjection::default();
    projection.hydrate_thread(thread);
    projection.entries().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::protocol::v2::{
        SessionSource, ThreadActiveFlag, ThreadHistoryMode, ThreadItem, ThreadStatus, Turn,
        TurnStatus,
    };
    use std::path::PathBuf;

    fn thread(items: Vec<ThreadItem>) -> Thread {
        Thread {
            id: "thread-1".into(),
            extra: None,
            session_id: "session-1".into(),
            forked_from_id: None,
            parent_thread_id: None,
            preview: "preview".into(),
            ephemeral: false,
            section: None,
            section_entered_at: None,
            project_id: None,
            history_mode: ThreadHistoryMode::default(),
            model_provider: "fixture".into(),
            created_at: 1,
            updated_at: 1,
            recency_at: None,
            status: ThreadStatus::Active {
                active_flags: vec![ThreadActiveFlag::WaitingOnUserInput],
            },
            path: None,
            cwd: PathBuf::from("/workspace"),
            cli_version: "test".into(),
            source: SessionSource::Cli,
            can_accept_direct_input: Some(true),
            thread_source: None,
            agent_nickname: None,
            agent_role: None,
            git_info: None,
            name: None,
            turns: vec![Turn {
                id: "turn-1".into(),
                status: TurnStatus::Completed,
                error: None,
                items,
                items_view: Default::default(),
                started_at: None,
                completed_at: None,
                duration_ms: None,
            }],
        }
    }

    #[test]
    fn persisted_items_share_the_live_projection_shape() {
        let entries = thread_to_transcript_entries(thread(vec![ThreadItem::AgentMessage {
            id: "assistant-1".into(),
            metadata: None,
            text: "done".into(),
            phase: None,
            memory_citation: None,
            delivery: None,
        }]));

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "assistant-1");
        assert_eq!(entries[0].text, "done");
    }

    #[test]
    fn empty_threads_produce_an_empty_canonical_transcript() {
        assert!(thread_to_transcript_entries(thread(Vec::new())).is_empty());
    }
}
