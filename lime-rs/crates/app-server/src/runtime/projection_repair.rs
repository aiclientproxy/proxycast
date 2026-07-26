use super::event_log::{EventLogIssue, EventLogWriter};
use super::projection_store::ProjectionReadSession;
use super::projection_store::ProjectionReadWindow;
use super::projection_store::ProjectionStore;
use super::turn_input_events;
use super::StoredSession;
use agent_protocol::{ThreadId, ThreadTurnsView};
use std::collections::HashMap;
use thread_store::ReadThreadParams;

#[derive(Debug, Clone)]
pub struct ProjectionRepair {
    event_log_writer: EventLogWriter,
    projection_store: ProjectionStore,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProjectionRepairStatus {
    Empty,
    Rebuilt,
    RebuiltAfterTailTruncation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProjectionRepairAudit {
    pub(crate) session_id: String,
    pub(crate) status: ProjectionRepairStatus,
    pub(crate) records_scanned: usize,
    pub(crate) events_applied: usize,
    pub(crate) last_valid_offset: u64,
    pub(crate) file_len: u64,
    pub(crate) fingerprint: String,
    pub(crate) issue: Option<EventLogIssue>,
}

impl ProjectionRepair {
    pub fn new(event_log_writer: EventLogWriter, projection_store: ProjectionStore) -> Self {
        Self {
            event_log_writer,
            projection_store,
        }
    }

    pub fn repair_session(&self, session_id: &str) -> Result<usize, String> {
        Ok(self.repair_session_with_audit(session_id)?.events_applied)
    }

    fn repair_canonical_session(
        &self,
        projection: &ProjectionReadSession,
        events: &[app_server_protocol::AgentEvent],
    ) -> Result<(), String> {
        let events = self.canonical_repair_events(projection, events)?;
        let stored = StoredSession {
            session: projection.session.clone(),
            turns: projection.turns.clone(),
            turn_inputs: turn_input_events::turn_inputs_from_events(&events),
            turn_runtime_options: HashMap::new(),
            events: events.clone(),
            output_blobs: HashMap::new(),
        };
        self.projection_store
            .repair_canonical_history(&stored, &events)
    }

    fn canonical_repair_events(
        &self,
        projection: &ProjectionReadSession,
        events: &[app_server_protocol::AgentEvent],
    ) -> Result<Vec<app_server_protocol::AgentEvent>, String> {
        let Some(thread) = self
            .projection_store
            .read_thread_sync(ReadThreadParams {
                thread_id: ThreadId::new(projection.session.thread_id.clone()),
                include_archived: true,
                turns_view: ThreadTurnsView::Full,
            })
            .map_err(|error| error.to_string())?
        else {
            return Ok(events.to_vec());
        };
        if thread.forked_from_id.is_none() {
            return Ok(events.to_vec());
        }
        let Some(fork_sequence) = thread
            .metadata
            .get("forkSequence")
            .and_then(serde_json::Value::as_u64)
        else {
            // AgentControl topology forks have their own persisted history owner and do not use
            // the public thread/fork canonical seed boundary.
            return Ok(events.to_vec());
        };
        let items = thread
            .turns
            .iter()
            .flat_map(|turn| turn.items.iter())
            .filter(|item| item.sequence <= fork_sequence)
            .cloned()
            .collect::<Vec<_>>();
        let item_turn_ids = items
            .iter()
            .map(|item| item.turn_id.as_str())
            .collect::<std::collections::HashSet<_>>();
        let turns = thread
            .turns
            .iter()
            .filter(|turn| item_turn_ids.contains(turn.turn_id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let seed = super::thread_fork::fork_history_seed_events(
            &projection.session,
            &turns,
            &items,
            fork_sequence,
        )
        .map_err(|error| error.to_string())?;
        super::thread_fork::merge_fork_history_events(seed, events.to_vec())
            .map_err(|error| error.to_string())
    }

    pub(crate) fn repair_session_with_audit(
        &self,
        session_id: &str,
    ) -> Result<ProjectionRepairAudit, String> {
        let initial_scan = self.event_log_writer.scan_session_events(session_id)?;
        let original_file_len = initial_scan.file_len;
        let repaired_issue = initial_scan.issue.clone();
        let had_repairable_tail = initial_scan
            .issue
            .as_ref()
            .is_some_and(EventLogIssue::is_repairable_tail);
        let scan = if had_repairable_tail {
            self.event_log_writer.repair_session_event_log(session_id)?
        } else {
            if let Some(issue) = initial_scan.issue.clone() {
                return Err(format!(
                    "Projection repair refused event log {}: {issue:?}",
                    initial_scan.path.display()
                ));
            }
            initial_scan
        };
        let events = scan
            .records
            .iter()
            .map(|record| record.event.clone())
            .collect::<Vec<_>>();
        let events_applied = self.projection_store.repair_session(session_id, &events)?;
        if let Some(projection) = self
            .projection_store
            .read_session_projection(session_id, ProjectionReadWindow::default())?
        {
            self.repair_canonical_session(&projection, &events)?;
        }
        Ok(ProjectionRepairAudit {
            session_id: session_id.to_string(),
            status: if had_repairable_tail {
                ProjectionRepairStatus::RebuiltAfterTailTruncation
            } else if events.is_empty() {
                ProjectionRepairStatus::Empty
            } else {
                ProjectionRepairStatus::Rebuilt
            },
            records_scanned: events.len(),
            events_applied,
            last_valid_offset: scan.last_valid_offset,
            file_len: original_file_len,
            fingerprint: scan.fingerprint,
            issue: repaired_issue,
        })
    }

    pub fn read_repaired_session(
        &self,
        session_id: &str,
        existing_projection_window: Option<ProjectionReadWindow>,
    ) -> Result<Option<(ProjectionReadSession, Vec<app_server_protocol::AgentEvent>)>, String> {
        let scan = self.event_log_writer.scan_session_events(session_id)?;
        let scan = if scan
            .issue
            .as_ref()
            .is_some_and(EventLogIssue::is_repairable_tail)
        {
            self.event_log_writer.repair_session_event_log(session_id)?
        } else {
            if let Some(issue) = scan.issue.clone() {
                return Err(format!(
                    "无法读取 event log 进行 projection repair: {issue:?}"
                ));
            }
            scan
        };
        let events = scan
            .records
            .into_iter()
            .map(|record| record.event)
            .collect::<Vec<_>>();
        let latest_sequence = events.last().map(|event| event.sequence).unwrap_or(0);
        let watermark = self.projection_store.read_watermark(session_id)?;
        if let Some(watermark) = watermark {
            if watermark.last_sequence > latest_sequence {
                return Err(format!(
                    "Projection repair refused shorter event log: session_id={session_id} projection_sequence={} event_log_sequence={latest_sequence}",
                    watermark.last_sequence
                ));
            }
            if watermark.last_sequence == latest_sequence {
                if let Some(window) = existing_projection_window {
                    if let Some(session) = self
                        .projection_store
                        .read_session_projection(session_id, window)?
                    {
                        self.repair_canonical_session(&session, &events)?;
                        return Ok(Some((session, Vec::new())));
                    }
                }
            }
        }
        if events.is_empty() {
            self.projection_store.repair_session(session_id, &[])?;
            return Ok(None);
        }
        let needs_repair = match watermark {
            Some(watermark) => watermark.last_sequence < latest_sequence,
            None => true,
        };
        if needs_repair {
            self.projection_store.repair_session(session_id, &events)?;
        }
        let Some(session) = self
            .projection_store
            .read_session_projection(session_id, existing_projection_window.unwrap_or_default())?
        else {
            return Ok(None);
        };
        self.repair_canonical_session(&session, &events)?;
        Ok(Some((session, events)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::StorageRoots;
    use agent_protocol::{ItemStatus, ThreadItemPayload, ThreadTurnsView};
    use app_server_protocol::{AgentEvent, AgentSession, AgentSessionStatus};
    use serde_json::json;
    use thread_store::ReadThreadParams;

    #[test]
    fn repair_session_rebuilds_projection_from_jsonl_event_log() {
        let temp = tempfile::tempdir().expect("tempdir");
        let roots =
            StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
        let event_log_writer = EventLogWriter::new(&roots.event_log_root).expect("writer");
        let projection_store =
            ProjectionStore::initialize(&roots.projection_db_path).expect("projection");
        let accepted = event(1, "turn.accepted");
        let completed = event(2, "turn.completed");

        event_log_writer.append(&accepted).expect("append accepted");
        event_log_writer
            .append(&completed)
            .expect("append completed");
        projection_store
            .apply_event(&accepted)
            .expect("apply stale accepted");
        projection_store
            .clear_session("sess_repair")
            .expect("clear projection");

        let repair = ProjectionRepair::new(event_log_writer, projection_store.clone());
        let repaired = repair
            .repair_session("sess_repair")
            .expect("repair projection");

        assert_eq!(repaired, 2);
        let session = projection_store
            .read_session("sess_repair")
            .expect("read projection")
            .expect("session");
        assert_eq!(session.status, "completed");
        assert_eq!(session.last_event_sequence, 2);
    }

    #[test]
    fn read_repaired_session_repairs_missing_projection_before_read() {
        let temp = tempfile::tempdir().expect("tempdir");
        let roots =
            StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
        let event_log_writer = EventLogWriter::new(&roots.event_log_root).expect("writer");
        let projection_store =
            ProjectionStore::initialize(&roots.projection_db_path).expect("projection");
        let accepted = event(1, "turn.accepted");
        let completed = event(2, "turn.completed");

        event_log_writer.append(&accepted).expect("append accepted");
        event_log_writer
            .append(&completed)
            .expect("append completed");

        let repair = ProjectionRepair::new(event_log_writer, projection_store.clone());
        let (session, events) = repair
            .read_repaired_session("sess_repair", None)
            .expect("read repaired")
            .expect("session");

        assert_eq!(events.len(), 2);
        assert_eq!(session.session.session_id, "sess_repair");
        assert_eq!(
            session.session.status,
            app_server_protocol::AgentSessionStatus::Completed
        );
        assert_eq!(session.turns.len(), 1);
        assert_eq!(session.last_event_sequence, 2);
        assert!(projection_store
            .read_thread_sync(thread_store::ReadThreadParams {
                thread_id: agent_protocol::ThreadId::new("thread_repair"),
                include_archived: true,
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .expect("read canonical thread")
            .is_some());
    }

    #[test]
    fn read_repaired_session_rebuilds_partial_canonical_message_history() {
        let temp = tempfile::tempdir().expect("tempdir");
        let roots =
            StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
        let event_log_writer = EventLogWriter::new(&roots.event_log_root).expect("writer");
        let projection_store =
            ProjectionStore::initialize(&roots.projection_db_path).expect("projection");
        let events = vec![
            event_with_payload(
                1,
                "message.created",
                json!({"role": "user", "input": [{"type": "text", "text": "inspect"}]}),
            ),
            event_with_payload(2, "reasoning.started", json!({"status": "in_progress"})),
            event_with_payload(
                3,
                "reasoning.final",
                json!({"text": "inspect inputs", "status": "completed"}),
            ),
            event_with_payload(
                4,
                "message.delta",
                json!({"role": "assistant", "text": "done"}),
            ),
            event_with_payload(
                5,
                "message.completed",
                json!({"role": "assistant", "status": "completed"}),
            ),
            event_with_payload(6, "turn.completed", json!({})),
        ];
        for event in &events {
            event_log_writer.append(event).expect("append event");
        }

        let stored = StoredSession {
            session: AgentSession {
                session_id: "sess_repair".to_string(),
                thread_id: "thread_repair".to_string(),
                app_id: "agent-chat".to_string(),
                workspace_id: None,
                business_object_ref: None,
                status: AgentSessionStatus::Completed,
                created_at: "2026-06-14T00:00:00.000Z".to_string(),
                updated_at: "2026-06-14T00:00:06.000Z".to_string(),
            },
            turns: Vec::new(),
            turn_inputs: HashMap::new(),
            turn_runtime_options: HashMap::new(),
            events: events.clone(),
            output_blobs: HashMap::new(),
        };
        projection_store
            .apply_canonical_events(&stored, &events[1..3])
            .expect("seed partial reasoning-only history");
        let partial = projection_store
            .read_thread_sync(ReadThreadParams {
                thread_id: agent_protocol::ThreadId::new("thread_repair"),
                include_archived: true,
                turns_view: ThreadTurnsView::Full,
            })
            .expect("read partial thread")
            .expect("partial thread");
        assert_eq!(partial.turns[0].items.len(), 1);
        assert!(matches!(
            partial.turns[0].items[0].payload,
            ThreadItemPayload::Reasoning { .. }
        ));

        let repair = ProjectionRepair::new(event_log_writer, projection_store.clone());
        repair
            .read_repaired_session("sess_repair", None)
            .expect("repair partial canonical history")
            .expect("repaired session");
        let repaired = projection_store
            .read_thread_sync(ReadThreadParams {
                thread_id: agent_protocol::ThreadId::new("thread_repair"),
                include_archived: true,
                turns_view: ThreadTurnsView::Full,
            })
            .expect("read repaired thread")
            .expect("repaired thread");
        let items = &repaired.turns[0].items;
        assert_eq!(items.len(), 3);
        assert!(matches!(
            items[0].payload,
            ThreadItemPayload::UserMessage { .. }
        ));
        assert!(matches!(
            items[1].payload,
            ThreadItemPayload::Reasoning { .. }
        ));
        assert!(matches!(
            items[2].payload,
            ThreadItemPayload::AgentMessage { .. }
        ));
        assert!(items
            .iter()
            .all(|item| item.status == ItemStatus::Completed));
        assert!(items.iter().all(|item| item.completed_at_ms.is_some()));

        repair
            .read_repaired_session("sess_repair", None)
            .expect("repeat idempotent canonical repair")
            .expect("repaired session");
        let repeated = projection_store
            .read_thread_sync(ReadThreadParams {
                thread_id: agent_protocol::ThreadId::new("thread_repair"),
                include_archived: true,
                turns_view: ThreadTurnsView::Full,
            })
            .expect("read repeated thread")
            .expect("repeated thread");
        assert_eq!(repeated, repaired);
    }

    fn event(sequence: u64, event_type: &str) -> AgentEvent {
        event_with_payload(sequence, event_type, json!({ "sequence": sequence }))
    }

    fn event_with_payload(
        sequence: u64,
        event_type: &str,
        payload: serde_json::Value,
    ) -> AgentEvent {
        AgentEvent {
            event_id: format!("evt-{sequence}"),
            sequence,
            session_id: "sess_repair".to_string(),
            thread_id: Some("thread_repair".to_string()),
            turn_id: Some("turn_repair".to_string()),
            event_type: event_type.to_string(),
            timestamp: format!("2026-06-14T00:00:{sequence:02}.000Z"),
            payload,
        }
    }
}
