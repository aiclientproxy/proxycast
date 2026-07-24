use app_server_protocol::AgentEvent;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

#[derive(Debug, Clone, PartialEq)]
pub struct EventLogRecord {
    pub path: PathBuf,
    pub event: AgentEvent,
}

/// A single issue found while scanning the canonical JSONL event log.
///
/// The scanner deliberately stops at the first issue.  A projection may only
/// be rebuilt from the contiguous, fingerprinted prefix; a caller must not
/// silently sort or skip records after an issue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EventLogIssue {
    UnterminatedTail {
        offset: u64,
    },
    MalformedTail {
        offset: u64,
        message: String,
    },
    MalformedRecord {
        offset: u64,
        message: String,
    },
    SequenceGap {
        offset: u64,
        expected: u64,
        actual: u64,
    },
    SequenceRegression {
        offset: u64,
        previous: u64,
        actual: u64,
    },
    EqualSequenceDivergence {
        offset: u64,
        sequence: u64,
        previous_event_id: String,
        event_id: String,
    },
    DuplicateEventId {
        offset: u64,
        event_id: String,
    },
    SessionMismatch {
        offset: u64,
        expected: String,
        actual: String,
    },
}

impl EventLogIssue {
    pub(crate) fn is_repairable_tail(&self) -> bool {
        matches!(
            self,
            Self::UnterminatedTail { .. } | Self::MalformedTail { .. }
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EventLogScan {
    pub(crate) path: PathBuf,
    pub(crate) records: Vec<EventLogRecord>,
    /// SHA-256 over the canonical serialized records in the valid prefix.
    pub(crate) fingerprint: String,
    /// Byte offset immediately after the last valid, newline-terminated record.
    pub(crate) last_valid_offset: u64,
    pub(crate) file_len: u64,
    pub(crate) issue: Option<EventLogIssue>,
}

#[derive(Debug, Clone)]
pub struct EventLogWriter {
    root: PathBuf,
    append_states: Arc<Mutex<BTreeMap<PathBuf, EventLogAppendState>>>,
    io_locks: Arc<Mutex<BTreeMap<PathBuf, Arc<Mutex<()>>>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EventLogAppendState {
    file_len: u64,
    modified: Option<SystemTime>,
    last_sequence: Option<u64>,
    event_ids: BTreeSet<String>,
}

impl PartialEq for EventLogWriter {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
    }
}

impl Eq for EventLogWriter {}

pub const WORKFLOW_AUDIT_ACTIVE_COMPACT_AFTER_RECORDS: usize = 1024;
pub const WORKFLOW_AUDIT_ACTIVE_RETAIN_RECENT_RECORDS: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowAuditCompactionReport {
    pub session_id: String,
    pub before_count: usize,
    pub archived_count: usize,
    pub retained_count: usize,
    pub archive_path: Option<PathBuf>,
    pub active_path: PathBuf,
}

impl EventLogWriter {
    pub fn new(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|error| {
            format!(
                "无法创建 App Server event log 目录 {}: {error}",
                root.display()
            )
        })?;
        Ok(Self {
            root,
            append_states: Arc::new(Mutex::new(BTreeMap::new())),
            io_locks: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }

    pub fn append(&self, event: &AgentEvent) -> Result<PathBuf, String> {
        self.append_events(std::slice::from_ref(event))?
            .into_iter()
            .next()
            .ok_or_else(|| format!("无法写入 event log: event {} 未产生路径", event.event_id))
    }

    pub fn append_events(&self, events: &[AgentEvent]) -> Result<Vec<PathBuf>, String> {
        if events.is_empty() {
            return Ok(Vec::new());
        }
        let mut events_by_path: BTreeMap<PathBuf, Vec<&AgentEvent>> = BTreeMap::new();
        for event in events {
            events_by_path
                .entry(self.session_path(&event.session_id))
                .or_default()
                .push(event);
        }

        let mut paths = Vec::with_capacity(events_by_path.len());
        for (path, events) in events_by_path {
            let io_lock = self.io_lock_for(&path)?;
            let _io_guard = io_lock
                .lock()
                .map_err(|_| "event log I/O lock poisoned".to_string())?;
            let session_id = events
                .first()
                .map(|event| event.session_id.as_str())
                .ok_or_else(|| "event log append group cannot be empty".to_string())?;
            let append_state = self.prepare_append_state(&path, session_id)?;
            validate_appended_events(&append_state, &events)?;
            append_events_to_path(&path, &events)?;
            self.update_append_state(&path, append_state, &events)?;
            paths.push(path);
        }
        Ok(paths)
    }

    pub fn read_session_events(&self, session_id: &str) -> Result<Vec<EventLogRecord>, String> {
        let path = self.session_path(session_id);
        let io_lock = self.io_lock_for(&path)?;
        let _io_guard = io_lock
            .lock()
            .map_err(|_| "event log I/O lock poisoned".to_string())?;
        let scan = scan_event_log_path(&path, session_id)?;
        if let Some(issue) = scan.issue {
            return Err(format!(
                "event log {} requires repair at offset {}: {issue:?}",
                scan.path.display(),
                scan.last_valid_offset
            ));
        }
        Ok(scan.records)
    }

    #[cfg(test)]
    pub(crate) fn list_session_ids(&self) -> Result<Vec<String>, String> {
        let sessions_dir = self.root.join("sessions");
        let entries = match fs::read_dir(&sessions_dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(format!(
                    "无法读取 event log session 目录 {}: {error}",
                    sessions_dir.display()
                ));
            }
        };
        let mut session_ids = BTreeSet::new();
        for entry in entries {
            let entry = entry.map_err(|error| {
                format!(
                    "无法读取 event log session 目录项 {}: {error}",
                    sessions_dir.display()
                )
            })?;
            let path = entry.path();
            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if !path.is_file()
                || !file_name.starts_with("session_")
                || !file_name.ends_with(".jsonl")
            {
                continue;
            }
            let io_lock = self.io_lock_for(&path)?;
            let _io_guard = io_lock
                .lock()
                .map_err(|_| "event log I/O lock poisoned".to_string())?;
            let file = fs::File::open(&path)
                .map_err(|error| format!("无法读取 event log {}: {error}", path.display()))?;
            for line in BufReader::new(file).lines() {
                let line = line
                    .map_err(|error| format!("无法读取 event log {}: {error}", path.display()))?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let event = serde_json::from_str::<AgentEvent>(trimmed)
                    .map_err(|error| format!("无法解析 event log {}: {error}", path.display()))?;
                session_ids.insert(event.session_id);
                break;
            }
        }
        Ok(session_ids.into_iter().collect())
    }

    #[cfg(test)]
    pub(crate) fn list_queued_session_ids(&self) -> Result<Vec<String>, String> {
        let mut queued_session_ids = Vec::new();
        for session_id in self.list_session_ids()? {
            let mut scan = self.scan_session_events(&session_id)?;
            if scan
                .issue
                .as_ref()
                .is_some_and(EventLogIssue::is_repairable_tail)
            {
                scan = self.repair_session_event_log(&session_id)?;
            }
            if let Some(issue) = scan.issue {
                return Err(format!(
                    "event log {} is not safely readable for queued work discovery: {issue:?}",
                    scan.path.display()
                ));
            }
            let mut queued_turn_ids = BTreeSet::new();
            for record in scan.records {
                let event = record.event;
                match event.event_type.as_str() {
                    "queue.added" => {
                        if let Some(turn_id) = event.turn_id {
                            queued_turn_ids.insert(turn_id);
                        }
                    }
                    "queue.removed" => {
                        if let Some(turn_id) = event
                            .payload
                            .get("queuedTurnId")
                            .or_else(|| event.payload.get("queued_turn_id"))
                            .and_then(Value::as_str)
                        {
                            queued_turn_ids.remove(turn_id);
                        }
                    }
                    "turn.accepted" | "turn.started" | "turn.completed" | "turn.failed"
                    | "turn.canceled" => {
                        if let Some(turn_id) = event.turn_id {
                            queued_turn_ids.remove(&turn_id);
                        }
                    }
                    _ => {}
                }
            }
            if !queued_turn_ids.is_empty() {
                queued_session_ids.push(session_id);
            }
        }
        Ok(queued_session_ids)
    }

    /// Scan one session log and return only its contiguous valid prefix.
    ///
    /// No mutation happens during a scan.  A final malformed or unterminated
    /// record is isolated in `issue`; sequence gaps and divergence are never
    /// interpreted as stale events.
    pub(crate) fn scan_session_events(&self, session_id: &str) -> Result<EventLogScan, String> {
        let path = self.session_path(session_id);
        let io_lock = self.io_lock_for(&path)?;
        let _io_guard = io_lock
            .lock()
            .map_err(|_| "event log I/O lock poisoned".to_string())?;
        scan_event_log_path(&path, session_id)
    }

    /// Truncate only a repairable malformed/unterminated tail.  Middle-log
    /// corruption and sequence issues fail closed and are left untouched.
    pub(crate) fn repair_session_event_log(
        &self,
        session_id: &str,
    ) -> Result<EventLogScan, String> {
        let path = self.session_path(session_id);
        let io_lock = self.io_lock_for(&path)?;
        let _io_guard = io_lock
            .lock()
            .map_err(|_| "event log I/O lock poisoned".to_string())?;
        let scan = scan_event_log_path(&path, session_id)?;
        let Some(issue) = scan.issue.as_ref() else {
            return Ok(scan);
        };
        if !issue.is_repairable_tail() {
            return Err(format!(
                "event log {} is not safely repairable: {issue:?}",
                path.display()
            ));
        }
        let file = fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .map_err(|error| format!("无法打开 event log 以截断 {}: {error}", path.display()))?;
        file.set_len(scan.last_valid_offset).map_err(|error| {
            format!(
                "无法截断 event log {} 到 {}: {error}",
                path.display(),
                scan.last_valid_offset
            )
        })?;
        scan_event_log_path(&path, session_id)
    }

    pub fn append_workflow_audit_events(
        &self,
        session_id: &str,
        events: &[AgentEvent],
    ) -> Result<PathBuf, String> {
        let path = self.workflow_audit_path(session_id);
        let redacted_events = events
            .iter()
            .map(redact_workflow_audit_event)
            .collect::<Vec<_>>();
        let events = redacted_events.iter().collect::<Vec<_>>();
        append_events_to_path(&path, &events)?;
        if let Err(error) = self.compact_session_workflow_audit_events_if_needed(session_id) {
            tracing::warn!(
                "[event-log] failed to compact workflow audit events for session {}: {}",
                session_id,
                error
            );
        }
        Ok(path)
    }

    pub fn read_session_workflow_audit_events(
        &self,
        session_id: &str,
    ) -> Result<Vec<EventLogRecord>, String> {
        let mut records = Vec::new();
        for path in self.workflow_audit_archive_paths(session_id)? {
            records.extend(read_events_from_path(&path)?);
        }
        let path = self.workflow_audit_path(session_id);
        records.extend(read_events_from_path(&path)?);
        Ok(records)
    }

    pub fn compact_session_workflow_audit_events(
        &self,
        session_id: &str,
        retain_recent: usize,
    ) -> Result<WorkflowAuditCompactionReport, String> {
        if retain_recent == 0 {
            return Err(
                "workflow audit compaction retain_recent must be greater than 0".to_string(),
            );
        }
        let active_path = self.workflow_audit_path(session_id);
        let current_records = read_events_from_path(&active_path)?;
        let before_count = current_records.len();
        if before_count <= retain_recent {
            return Ok(WorkflowAuditCompactionReport {
                session_id: session_id.to_string(),
                before_count,
                archived_count: 0,
                retained_count: before_count,
                archive_path: None,
                active_path,
            });
        }

        let archive_count = before_count - retain_recent;
        let archived_records = &current_records[..archive_count];
        let retained_records = &current_records[archive_count..];
        let archive_path = self.workflow_audit_archive_path(session_id, archived_records);
        write_events_to_path(
            &archive_path,
            &archived_records
                .iter()
                .map(|record| &record.event)
                .collect::<Vec<_>>(),
        )?;
        write_events_to_path(
            &active_path,
            &retained_records
                .iter()
                .map(|record| &record.event)
                .collect::<Vec<_>>(),
        )?;
        Ok(WorkflowAuditCompactionReport {
            session_id: session_id.to_string(),
            before_count,
            archived_count: archived_records.len(),
            retained_count: retained_records.len(),
            archive_path: Some(archive_path),
            active_path,
        })
    }

    pub fn compact_session_workflow_audit_events_if_needed(
        &self,
        session_id: &str,
    ) -> Result<Option<WorkflowAuditCompactionReport>, String> {
        let active_path = self.workflow_audit_path(session_id);
        let active_count = read_events_from_path(&active_path)?.len();
        if active_count <= WORKFLOW_AUDIT_ACTIVE_COMPACT_AFTER_RECORDS {
            return Ok(None);
        }
        self.compact_session_workflow_audit_events(
            session_id,
            WORKFLOW_AUDIT_ACTIVE_RETAIN_RECENT_RECORDS,
        )
        .map(Some)
    }

    pub fn workflow_audit_path(&self, session_id: &str) -> PathBuf {
        self.workflow_audit_dir(session_id)
            .join("workflow-events.jsonl")
    }

    fn workflow_audit_dir(&self, session_id: &str) -> PathBuf {
        self.root
            .join("sessions")
            .join(format!("session_{}", safe_file_stem(session_id)))
    }

    fn workflow_audit_archive_paths(&self, session_id: &str) -> Result<Vec<PathBuf>, String> {
        let dir = self.workflow_audit_dir(session_id);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut paths = Vec::new();
        let entries = fs::read_dir(&dir)
            .map_err(|error| format!("无法读取 workflow audit 目录 {}: {error}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                format!("无法读取 workflow audit 目录项 {}: {error}", dir.display())
            })?;
            let path = entry.path();
            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if file_name.starts_with("workflow-events.archive.") && file_name.ends_with(".jsonl") {
                paths.push(path);
            }
        }
        paths.sort();
        Ok(paths)
    }

    fn workflow_audit_archive_path(&self, session_id: &str, records: &[EventLogRecord]) -> PathBuf {
        let first_sequence = records
            .first()
            .map(|record| record.event.sequence)
            .unwrap_or(0);
        let last_sequence = records
            .last()
            .map(|record| record.event.sequence)
            .unwrap_or(first_sequence);
        self.workflow_audit_dir(session_id).join(format!(
            "workflow-events.archive.{first_sequence:020}-{last_sequence:020}.jsonl"
        ))
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.root
            .join("sessions")
            .join(format!("session_{}.jsonl", safe_file_stem(session_id)))
    }

    fn io_lock_for(&self, path: &Path) -> Result<Arc<Mutex<()>>, String> {
        let mut io_locks = self
            .io_locks
            .lock()
            .map_err(|_| "event log lock registry poisoned".to_string())?;
        Ok(io_locks
            .entry(path.to_path_buf())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone())
    }

    fn prepare_append_state(
        &self,
        path: &Path,
        session_id: &str,
    ) -> Result<EventLogAppendState, String> {
        let stamp = event_log_file_stamp(path)?;
        if let Some(state) = self
            .append_states
            .lock()
            .map_err(|_| "event log append state lock poisoned".to_string())?
            .get(path)
            .filter(|state| state.matches(stamp.0, stamp.1))
            .cloned()
        {
            return Ok(state);
        }

        let scan = prepare_session_event_log_for_append(path, session_id)?;
        let state = EventLogAppendState::from_scan(&scan);
        self.append_states
            .lock()
            .map_err(|_| "event log append state lock poisoned".to_string())?
            .insert(path.to_path_buf(), state.clone());
        Ok(state)
    }

    fn update_append_state(
        &self,
        path: &Path,
        mut state: EventLogAppendState,
        events: &[&AgentEvent],
    ) -> Result<(), String> {
        for event in events {
            state.last_sequence = Some(event.sequence);
            state.event_ids.insert(event.event_id.clone());
        }
        let (file_len, modified) = event_log_file_stamp(path)?;
        state.file_len = file_len;
        state.modified = modified;
        self.append_states
            .lock()
            .map_err(|_| "event log append state lock poisoned".to_string())?
            .insert(path.to_path_buf(), state);
        Ok(())
    }
}

fn prepare_session_event_log_for_append(
    path: &Path,
    session_id: &str,
) -> Result<EventLogScan, String> {
    let scan = scan_event_log_path(path, session_id)?;
    match scan.issue.as_ref() {
        None => Ok(scan),
        Some(EventLogIssue::UnterminatedTail { .. }) => {
            // Codex rollouts repair a complete non-newline-terminated record
            // before appending.  Preserve that behavior, then rescan so a
            // sequence or identity conflict still fails closed.
            let mut file = fs::OpenOptions::new()
                .append(true)
                .open(path)
                .map_err(|error| {
                    format!(
                        "无法打开 unterminated event log {}: {error}",
                        path.display()
                    )
                })?;
            file.write_all(b"\n")
                .and_then(|_| file.flush())
                .map_err(|error| format!("无法终止 event log 尾部 {}: {error}", path.display()))?;
            let repaired = scan_event_log_path(path, session_id)?;
            if let Some(issue) = repaired.issue.as_ref() {
                return Err(format!(
                    "event log {} remains invalid after newline repair: {issue:?}",
                    path.display()
                ));
            }
            Ok(repaired)
        }
        Some(EventLogIssue::MalformedTail { .. }) => {
            let file = fs::OpenOptions::new()
                .write(true)
                .open(path)
                .map_err(|error| {
                    format!("无法打开 malformed event log {}: {error}", path.display())
                })?;
            file.set_len(scan.last_valid_offset).map_err(|error| {
                format!(
                    "无法截断 malformed event log {} 到 {}: {error}",
                    path.display(),
                    scan.last_valid_offset
                )
            })?;
            scan_event_log_path(path, session_id)
        }
        Some(issue) => Err(format!(
            "event log {} is not appendable: {issue:?}",
            path.display()
        )),
    }
}

impl EventLogAppendState {
    fn from_scan(scan: &EventLogScan) -> Self {
        Self {
            file_len: scan.file_len,
            modified: event_log_file_stamp(&scan.path)
                .ok()
                .and_then(|(_, modified)| modified),
            last_sequence: scan.records.last().map(|record| record.event.sequence),
            event_ids: scan
                .records
                .iter()
                .map(|record| record.event.event_id.clone())
                .collect(),
        }
    }

    fn matches(&self, file_len: u64, modified: Option<SystemTime>) -> bool {
        self.file_len == file_len && self.modified == modified
    }
}

fn event_log_file_stamp(path: &Path) -> Result<(u64, Option<SystemTime>), String> {
    match fs::metadata(path) {
        Ok(metadata) => Ok((metadata.len(), metadata.modified().ok())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok((0, None)),
        Err(error) => Err(format!(
            "无法读取 event log metadata {}: {error}",
            path.display()
        )),
    }
}

fn validate_appended_events(
    state: &EventLogAppendState,
    events: &[&AgentEvent],
) -> Result<(), String> {
    let expected_session_id = events
        .first()
        .map(|event| event.session_id.as_str())
        .unwrap_or_default();
    let mut previous_sequence = state.last_sequence;
    let mut event_ids = state.event_ids.clone();
    for event in events {
        if event.session_id != expected_session_id {
            return Err(format!(
                "event log append session mismatch: expected {expected_session_id}, got {}",
                event.session_id
            ));
        }
        if let Some(previous) = previous_sequence {
            let expected = previous
                .checked_add(1)
                .ok_or_else(|| "event log sequence overflow".to_string())?;
            if event.sequence != expected {
                return Err(format!(
                    "event log append sequence mismatch: expected {expected}, got {} ({})",
                    event.sequence, event.event_id
                ));
            }
        }
        if !event_ids.insert(event.event_id.clone()) {
            return Err(format!(
                "event log append duplicate event_id: {}",
                event.event_id
            ));
        }
        previous_sequence = Some(event.sequence);
    }
    Ok(())
}

fn scan_event_log_path(path: &Path, expected_session_id: &str) -> Result<EventLogScan, String> {
    if !path.exists() {
        return Ok(EventLogScan {
            path: path.to_path_buf(),
            records: Vec::new(),
            fingerprint: fingerprint_events(&[]),
            last_valid_offset: 0,
            file_len: 0,
            issue: None,
        });
    }

    let bytes = fs::read(path)
        .map_err(|error| format!("无法读取 event log {}: {error}", path.display()))?;
    let file_len = bytes.len() as u64;
    let mut records = Vec::new();
    let mut event_ids = BTreeMap::<String, u64>::new();
    let mut fingerprint = Sha256::new();
    let mut offset = 0_u64;
    let mut last_valid_offset = 0_u64;
    let mut previous_sequence = None;
    let mut issue = None;

    for raw_line in bytes.split_inclusive(|byte| *byte == b'\n') {
        let line_offset = offset;
        offset += raw_line.len() as u64;
        let has_newline = raw_line.last() == Some(&b'\n');
        let content = raw_line.strip_suffix(&[b'\n']).unwrap_or(raw_line);
        let content = content.strip_suffix(&[b'\r']).unwrap_or(content);
        if content.iter().all(u8::is_ascii_whitespace) {
            if has_newline {
                last_valid_offset = offset;
            }
            continue;
        }

        let event = match serde_json::from_slice::<AgentEvent>(content) {
            Ok(event) => event,
            Err(error) => {
                issue = Some(if !has_newline && offset == file_len {
                    EventLogIssue::MalformedTail {
                        offset: line_offset,
                        message: error.to_string(),
                    }
                } else {
                    EventLogIssue::MalformedRecord {
                        offset: line_offset,
                        message: error.to_string(),
                    }
                });
                break;
            }
        };

        if event.session_id != expected_session_id {
            issue = Some(EventLogIssue::SessionMismatch {
                offset: line_offset,
                expected: expected_session_id.to_string(),
                actual: event.session_id.clone(),
            });
            break;
        }
        if let Some(previous) = previous_sequence {
            if event.sequence == previous {
                let previous_event_id = records
                    .last()
                    .map(|record: &EventLogRecord| record.event.event_id.clone())
                    .unwrap_or_default();
                issue = Some(EventLogIssue::EqualSequenceDivergence {
                    offset: line_offset,
                    sequence: event.sequence,
                    previous_event_id,
                    event_id: event.event_id.clone(),
                });
                break;
            }
            if event.sequence < previous {
                issue = Some(EventLogIssue::SequenceRegression {
                    offset: line_offset,
                    previous,
                    actual: event.sequence,
                });
                break;
            }
            if event.sequence != previous.saturating_add(1) {
                issue = Some(EventLogIssue::SequenceGap {
                    offset: line_offset,
                    expected: previous.saturating_add(1),
                    actual: event.sequence,
                });
                break;
            }
        }
        if event_ids.contains_key(&event.event_id) {
            issue = Some(EventLogIssue::DuplicateEventId {
                offset: line_offset,
                event_id: event.event_id.clone(),
            });
            break;
        }

        // A complete JSON value without its terminating newline is treated as
        // a crash tail and isolated until an explicit repair truncates it.
        // Validate identity and ordering first so divergence cannot masquerade
        // as a repairable tail.
        if !has_newline && offset == file_len {
            issue = Some(EventLogIssue::UnterminatedTail {
                offset: line_offset,
            });
            break;
        }

        let canonical = serde_json::to_vec(&event)
            .map_err(|error| format!("无法规范化 event {}: {error}", event.event_id))?;
        fingerprint.update(&canonical);
        fingerprint.update([b'\n']);
        event_ids.insert(event.event_id.clone(), event.sequence);
        previous_sequence = Some(event.sequence);
        records.push(EventLogRecord {
            path: path.to_path_buf(),
            event,
        });
        last_valid_offset = offset;
    }

    Ok(EventLogScan {
        path: path.to_path_buf(),
        records,
        fingerprint: hex::encode(fingerprint.finalize()),
        last_valid_offset,
        file_len,
        issue,
    })
}

fn fingerprint_events(events: &[AgentEvent]) -> String {
    let mut hasher = Sha256::new();
    for event in events {
        if let Ok(canonical) = serde_json::to_vec(event) {
            hasher.update(canonical);
            hasher.update([b'\n']);
        }
    }
    hex::encode(hasher.finalize())
}

fn read_events_from_path(path: &Path) -> Result<Vec<EventLogRecord>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let mut records = Vec::new();
    let file = fs::File::open(&path)
        .map_err(|error| format!("无法读取 event log {}: {error}", path.display()))?;
    for line in BufReader::new(file).lines() {
        let line =
            line.map_err(|error| format!("无法读取 event log {}: {error}", path.display()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let event = serde_json::from_str::<AgentEvent>(trimmed)
            .map_err(|error| format!("无法解析 event log {}: {error}", path.display()))?;
        records.push(EventLogRecord {
            path: path.to_path_buf(),
            event,
        });
    }
    Ok(records)
}

fn append_events_to_path(path: &Path, events: &[&AgentEvent]) -> Result<(), String> {
    if events.is_empty() {
        return Ok(());
    }
    let mut file = open_event_log_for_append(path)?;
    write_events(&mut file, path, events)
}

fn write_events_to_path(path: &Path, events: &[&AgentEvent]) -> Result<(), String> {
    let temp_path = path.with_extension("jsonl.tmp");
    if let Some(parent) = temp_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "无法创建 event log 临时父目录 {}: {error}",
                parent.display()
            )
        })?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&temp_path)
        .map_err(|error| {
            format!(
                "无法打开 event log 临时文件 {}: {error}",
                temp_path.display()
            )
        })?;
    write_events(&mut file, &temp_path, events)?;
    file.flush().map_err(|error| {
        format!(
            "无法刷新 event log 临时文件 {}: {error}",
            temp_path.display()
        )
    })?;
    fs::rename(&temp_path, path).map_err(|error| {
        format!(
            "无法替换 event log {} <- {}: {error}",
            path.display(),
            temp_path.display()
        )
    })
}

fn open_event_log_for_append(path: &Path) -> Result<fs::File, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("无法创建 event log 父目录 {}: {error}", parent.display()))?;
    }
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("无法打开 event log {}: {error}", path.display()))
}

fn write_events(file: &mut fs::File, path: &Path, events: &[&AgentEvent]) -> Result<(), String> {
    for event in events {
        let json = serde_json::to_vec(event)
            .map_err(|error| format!("无法序列化 event {}: {error}", event.event_id))?;
        file.write_all(&json)
            .and_then(|_| file.write_all(b"\n"))
            .map_err(|error| format!("无法写入 event log {}: {error}", path.display()))?;
    }
    Ok(())
}

impl EventLogWriter {
    pub fn clear_session(&self, session_id: &str) -> Result<(), String> {
        let session_event_path = self.session_path(session_id);
        let io_lock = self.io_lock_for(&session_event_path)?;
        let _io_guard = io_lock
            .lock()
            .map_err(|_| "event log I/O lock poisoned".to_string())?;
        let workflow_audit_path = self.workflow_audit_path(session_id);
        let workflow_audit_archive_paths = self.workflow_audit_archive_paths(session_id)?;
        remove_event_log_path(&session_event_path)?;
        remove_event_log_path(&workflow_audit_path)?;
        for archive_path in workflow_audit_archive_paths {
            remove_event_log_path(&archive_path)?;
        }
        self.append_states
            .lock()
            .map_err(|_| "event log append state lock poisoned".to_string())?
            .remove(&session_event_path);
        Ok(())
    }
}

fn remove_event_log_path(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("无法删除 event log {}: {error}", path.display())),
    }
}

fn redact_workflow_audit_event(event: &AgentEvent) -> AgentEvent {
    let mut event = event.clone();
    redact_workflow_audit_value(&mut event.payload, None);
    ensure_workflow_audit_redaction_policy(&mut event.payload);
    event
}

fn redact_workflow_audit_value(value: &mut Value, key: Option<&str>) {
    if key.is_some_and(workflow_audit_sensitive_key) {
        *value = workflow_audit_redacted_value();
        return;
    }

    match value {
        Value::Object(object) => {
            for (child_key, child_value) in object.iter_mut() {
                redact_workflow_audit_value(child_value, Some(child_key.as_str()));
            }
        }
        Value::Array(items) => {
            for item in items {
                redact_workflow_audit_value(item, None);
            }
        }
        Value::String(text) => {
            if workflow_audit_string_looks_sensitive(text) {
                *text = "[redacted:workflow_audit_metadata_only]".to_string();
            } else {
                truncate_workflow_audit_string(text);
            }
        }
        _ => {}
    }
}

fn ensure_workflow_audit_redaction_policy(payload: &mut Value) {
    let policy = json!({
        "policy": "workflow_audit_metadata_only",
        "mode": "metadata_only",
        "promptText": false,
        "providerPayload": false,
        "rawContent": false,
    });
    if let Some(object) = payload.as_object_mut() {
        object.insert("redaction".to_string(), policy);
    }
}

fn workflow_audit_sensitive_key(key: &str) -> bool {
    let normalized = key
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    matches!(
        normalized.as_str(),
        "apikey"
            | "authorization"
            | "body"
            | "content"
            | "contents"
            | "credential"
            | "credentials"
            | "documenttext"
            | "finalmarkdown"
            | "headers"
            | "input"
            | "inputsummary"
            | "markdown"
            | "message"
            | "messages"
            | "password"
            | "prompt"
            | "providerconfig"
            | "providerpayload"
            | "query"
            | "request"
            | "response"
            | "result"
            | "searchquery"
            | "secret"
            | "summary"
            | "text"
            | "token"
            | "url"
    )
}

fn workflow_audit_redacted_value() -> Value {
    json!({
        "redacted": true,
        "policy": "workflow_audit_metadata_only",
    })
}

fn workflow_audit_string_looks_sensitive(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    lower.contains("bearer ")
        || lower.contains("api_key")
        || lower.contains("apikey")
        || lower.contains("authorization:")
        || lower.contains("password=")
        || lower.contains("secret=")
        || lower.contains("token=")
}

fn truncate_workflow_audit_string(value: &mut String) {
    const MAX_WORKFLOW_AUDIT_STRING_CHARS: usize = 512;
    if value.chars().count() <= MAX_WORKFLOW_AUDIT_STRING_CHARS {
        return;
    }
    let truncated = value
        .chars()
        .take(MAX_WORKFLOW_AUDIT_STRING_CHARS)
        .collect::<String>();
    *value = format!("{truncated}...[truncated:workflow_audit_metadata_only]");
}

fn safe_file_stem(value: &str) -> String {
    let stem = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    let stem = stem.trim_matches('_');
    if stem.is_empty() {
        "unknown".to_string()
    } else {
        stem.to_string()
    }
}

#[cfg(test)]
mod tests;
