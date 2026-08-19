use app_server_protocol::AgentEvent;
use app_server_protocol::DiagnosticsTraceEvent;
use app_server_protocol::DiagnosticsTraceExportParams;
use app_server_protocol::DiagnosticsTraceExportResponse;
use app_server_protocol::DiagnosticsTraceListParams;
use app_server_protocol::DiagnosticsTraceListResponse;
use app_server_protocol::DiagnosticsTraceReadParams;
use app_server_protocol::DiagnosticsTraceReadResponse;
use app_server_protocol::DiagnosticsTraceRedactionPolicy;
use app_server_protocol::DiagnosticsTraceSummary;
use chrono::DateTime;
use chrono::SecondsFormat;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

mod code_cell;
mod export;
mod summary;

pub(crate) use code_cell::CodeCellTraceProjection;
pub(crate) use summary::summarize_trace_event_store;

pub(crate) const RAW_TRACE_EVENT_SCHEMA_VERSION: u32 = 1;
pub(crate) const TRACE_EVENT_MAX_FILES_PER_SESSION: usize = 100;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RawTraceRedactionPolicy {
    pub(crate) mode: String,
    pub(crate) raw_agent_event_payload: bool,
    pub(crate) prompt_text: bool,
    pub(crate) provider_payload: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RawTraceEvent {
    pub(crate) schema_version: u32,
    pub(crate) seq: u64,
    pub(crate) wall_time_unix_ms: i64,
    pub(crate) trace_id: String,
    pub(crate) run_id: Option<String>,
    pub(crate) request_id: Option<String>,
    pub(crate) session_id: String,
    pub(crate) thread_id: Option<String>,
    pub(crate) turn_id: Option<String>,
    pub(crate) event_id: String,
    pub(crate) event_sequence: u64,
    pub(crate) event_type: String,
    pub(crate) checkpoint: String,
    pub(crate) metrics: BTreeMap<String, Value>,
    pub(crate) redaction: RawTraceRedactionPolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawTraceEventRecord {
    pub(crate) path: PathBuf,
    pub(crate) event: RawTraceEvent,
}

#[derive(Debug)]
pub struct TraceEventWriter {
    root: PathBuf,
    state: Mutex<TraceEventWriterState>,
}

#[derive(Debug, Default)]
struct TraceEventWriterState {
    next_seq_by_path: BTreeMap<PathBuf, u64>,
}

#[derive(Debug)]
struct RawTraceEventBatch {
    session_id: String,
    events: Vec<RawTraceEvent>,
}

impl TraceEventWriter {
    pub fn new(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|error| {
            format!(
                "无法创建 App Server trace event 目录 {}: {error}",
                root.display()
            )
        })?;
        Ok(Self {
            root,
            state: Mutex::new(TraceEventWriterState::default()),
        })
    }

    pub fn append_agent_events(&self, events: &[AgentEvent]) -> Result<Vec<PathBuf>, String> {
        let mut events_by_path: BTreeMap<PathBuf, RawTraceEventBatch> = BTreeMap::new();
        for event in events {
            let Some((trace_id, raw_event)) = raw_trace_event_from_agent_event(event) else {
                continue;
            };
            events_by_path
                .entry(self.trace_path(&event.session_id, &trace_id))
                .or_insert_with(|| RawTraceEventBatch {
                    session_id: event.session_id.clone(),
                    events: Vec::new(),
                })
                .events
                .push(raw_event);
        }

        let mut state = self.lock_state();
        let mut paths = Vec::with_capacity(events_by_path.len());
        for (path, mut batch) in events_by_path {
            let should_enforce_retention = !path.exists();
            let start_seq = state.next_seq_for_path(&path)?;
            for (index, event) in batch.events.iter_mut().enumerate() {
                event.seq = start_seq.saturating_add(index as u64);
            }
            append_raw_trace_events_to_path(&path, &batch.events)?;
            state.advance_next_seq(&path, start_seq, batch.events.len());
            paths.push(path.clone());
            if should_enforce_retention {
                self.enforce_session_retention(&batch.session_id)?;
            }
        }
        Ok(paths)
    }

    pub(crate) fn append_code_cell_trace_event(
        &self,
        session_id: &str,
        thread_id: &str,
        event: tool_runtime::tool_lifecycle::CodeCellTraceEvent,
    ) -> Result<(), String> {
        code_cell::append(self, session_id, thread_id, event)
    }

    pub(crate) fn close_code_cells_for_turn(
        &self,
        session_id: &str,
        thread_id: &str,
        turn_id: &str,
        status: tool_runtime::tool_lifecycle::CodeCellRuntimeStatus,
    ) -> Result<(), String> {
        code_cell::close_for_turn(self, session_id, thread_id, turn_id, status)
    }

    pub(crate) fn read_code_cell_trace(
        &self,
        session_id: &str,
        thread_id: &str,
    ) -> Result<CodeCellTraceProjection, String> {
        code_cell::read_projection(self, session_id, thread_id)
    }

    pub fn clear_session(&self, session_id: &str) -> Result<(), String> {
        let session_dir = self
            .root
            .join("sessions")
            .join(format!("session_{}", safe_file_stem(session_id)));
        let mut state = self.lock_state();
        if session_dir
            .try_exists()
            .map_err(|error| format!("无法检查 trace event session 目录: {error}"))?
        {
            fs::remove_dir_all(&session_dir).map_err(|error| {
                format!(
                    "无法删除 trace event session 目录 {}: {error}",
                    session_dir.display()
                )
            })?;
        }
        state
            .next_seq_by_path
            .retain(|path, _| !path.starts_with(&session_dir));
        Ok(())
    }

    pub fn list_trace_events(
        &self,
        params: DiagnosticsTraceListParams,
    ) -> Result<DiagnosticsTraceListResponse, String> {
        let limit = normalize_limit(params.limit, 100, 500);
        let mut summaries = Vec::new();
        for path in self.trace_paths_for_session_filter(params.session_id.as_deref())? {
            let records = read_trace_event_records(&path)?;
            if let Some(summary) = trace_summary_from_records(&path, &records) {
                summaries.push(summary);
            }
        }

        summaries.sort_by(|left, right| {
            right
                .last_wall_time_unix_ms
                .cmp(&left.last_wall_time_unix_ms)
                .then_with(|| right.modified_at.cmp(&left.modified_at))
                .then_with(|| right.path.cmp(&left.path))
        });
        summaries.truncate(limit);

        Ok(DiagnosticsTraceListResponse {
            available: true,
            trace_root: None,
            traces: summaries,
            redaction: protocol_redaction_policy(),
        })
    }

    pub fn read_trace_events(
        &self,
        params: DiagnosticsTraceReadParams,
    ) -> Result<DiagnosticsTraceReadResponse, String> {
        let max_events = normalize_limit(params.max_events, 5_000, 10_000);
        let path = self.trace_path(&params.session_id, &params.trace_id);
        if !path.exists() {
            return Ok(DiagnosticsTraceReadResponse {
                available: true,
                trace: None,
                events: Vec::new(),
                redaction: protocol_redaction_policy(),
            });
        }

        let records = read_trace_event_records(&path)?;
        if records
            .iter()
            .any(|record| record.event.event_type.starts_with("code_cell."))
        {
            let thread_id = records
                .iter()
                .find_map(|record| record.event.thread_id.as_deref())
                .ok_or_else(|| "CodeCell trace is missing thread identity".to_string())?;
            if params.trace_id != code_cell::trace_id(thread_id) {
                return Err(format!(
                    "CodeCell trace id {} does not match thread {thread_id}",
                    params.trace_id
                ));
            }
            let projection = self.read_code_cell_trace(&params.session_id, thread_id)?;
            let _validated_cell_count = projection
                .code_cells
                .len()
                .saturating_add(projection.pending_code_cell_ids.len());
        }
        let trace = trace_summary_from_records(&path, &records);
        let events = records
            .into_iter()
            .take(max_events)
            .map(|record| protocol_trace_event(record.event))
            .collect();

        Ok(DiagnosticsTraceReadResponse {
            available: true,
            trace,
            events,
            redaction: protocol_redaction_policy(),
        })
    }

    pub fn export_trace_events(
        &self,
        params: DiagnosticsTraceExportParams,
    ) -> Result<DiagnosticsTraceExportResponse, String> {
        self.export_trace_events_to_directory(params, export::default_trace_export_output_dir())
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn export_trace_events_to_directory(
        &self,
        params: DiagnosticsTraceExportParams,
        output_directory: PathBuf,
    ) -> Result<DiagnosticsTraceExportResponse, String> {
        let path = self.trace_path(&params.session_id, &params.trace_id);
        if !path.exists() {
            return Ok(DiagnosticsTraceExportResponse {
                available: true,
                exported: false,
                trace: None,
                bundle_path: None,
                output_directory: None,
                generated_at: None,
                included_sections: Vec::new(),
                omitted_sections: export::trace_export_omitted_sections(),
                redaction: protocol_redaction_policy(),
            });
        }

        let records = read_trace_event_records(&path)?;
        let Some(trace) = trace_summary_from_records(&path, &records) else {
            return Ok(DiagnosticsTraceExportResponse {
                available: true,
                exported: false,
                trace: None,
                bundle_path: None,
                output_directory: None,
                generated_at: None,
                included_sections: Vec::new(),
                omitted_sections: export::trace_export_omitted_sections(),
                redaction: protocol_redaction_policy(),
            });
        };

        fs::create_dir_all(&output_directory).map_err(|error| {
            format!(
                "无法创建 trace export 输出目录 {}: {error}",
                output_directory.display()
            )
        })?;
        let generated_at = Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true);
        let timestamp = Utc::now().format("%Y%m%d-%H%M%S").to_string();
        let bundle_name = format!(
            "claw-trace-{}-{}-{timestamp}.zip",
            safe_file_stem(&params.session_id),
            safe_file_stem(&params.trace_id)
        );
        let bundle_path = output_directory.join(bundle_name);
        export::write_trace_export_zip(&bundle_path, &generated_at, &trace, &records)?;

        Ok(DiagnosticsTraceExportResponse {
            available: true,
            exported: true,
            trace: Some(trace),
            bundle_path: Some(bundle_path.to_string_lossy().to_string()),
            output_directory: Some(output_directory.to_string_lossy().to_string()),
            generated_at: Some(generated_at),
            included_sections: export::trace_export_included_sections(),
            omitted_sections: export::trace_export_omitted_sections(),
            redaction: protocol_redaction_policy(),
        })
    }

    #[cfg(test)]
    pub(crate) fn read_raw_trace_events(
        &self,
        session_id: &str,
        trace_id: &str,
    ) -> Result<Vec<RawTraceEventRecord>, String> {
        let path = self.trace_path(session_id, trace_id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        read_trace_event_records(&path)
    }

    fn trace_path(&self, session_id: &str, trace_id: &str) -> PathBuf {
        trace_event_path(&self.root, session_id, trace_id)
    }

    fn enforce_session_retention(&self, session_id: &str) -> Result<(), String> {
        let session_dir = self
            .root
            .join("sessions")
            .join(format!("session_{}", safe_file_stem(session_id)));
        let Ok(entries) = fs::read_dir(&session_dir) else {
            return Ok(());
        };

        let mut files = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| {
                format!(
                    "无法读取 trace event session 目录 {}: {error}",
                    session_dir.display()
                )
            })?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
                continue;
            }
            let modified = entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            files.push((modified, path));
        }
        if files.len() <= TRACE_EVENT_MAX_FILES_PER_SESSION {
            return Ok(());
        }

        files.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        let remove_count = files.len() - TRACE_EVENT_MAX_FILES_PER_SESSION;
        for (_, path) in files.into_iter().take(remove_count) {
            fs::remove_file(&path).map_err(|error| {
                format!("无法删除旧 trace event 文件 {}: {error}", path.display())
            })?;
        }
        Ok(())
    }

    fn trace_paths_for_session_filter(
        &self,
        session_id: Option<&str>,
    ) -> Result<Vec<PathBuf>, String> {
        let sessions_root = self.root.join("sessions");
        let mut session_dirs = Vec::new();
        if let Some(session_id) = session_id {
            session_dirs
                .push(sessions_root.join(format!("session_{}", safe_file_stem(session_id))));
        } else {
            let Ok(entries) = fs::read_dir(&sessions_root) else {
                return Ok(Vec::new());
            };
            for entry in entries {
                let entry = entry.map_err(|error| {
                    format!(
                        "无法读取 trace event sessions 目录 {}: {error}",
                        sessions_root.display()
                    )
                })?;
                let path = entry.path();
                if path.is_dir()
                    && path
                        .file_name()
                        .and_then(|value| value.to_str())
                        .is_some_and(|name| name.starts_with("session_"))
                {
                    session_dirs.push(path);
                }
            }
        }

        let mut paths = Vec::new();
        for session_dir in session_dirs {
            let Ok(entries) = fs::read_dir(&session_dir) else {
                continue;
            };
            for entry in entries {
                let entry = entry.map_err(|error| {
                    format!(
                        "无法读取 trace event session 目录 {}: {error}",
                        session_dir.display()
                    )
                })?;
                let path = entry.path();
                if path.extension().and_then(|value| value.to_str()) == Some("jsonl") {
                    paths.push(path);
                }
            }
        }
        Ok(paths)
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, TraceEventWriterState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

pub(crate) fn export_trace_events_from_store_to_path(
    root: &Path,
    params: &DiagnosticsTraceExportParams,
    bundle_path: &Path,
    generated_at: &str,
) -> Result<Option<DiagnosticsTraceSummary>, String> {
    let path = trace_event_path(root, &params.session_id, &params.trace_id);
    if !path.exists() {
        return Ok(None);
    }

    let records = read_trace_event_records(&path)?;
    let Some(trace) = trace_summary_from_records(&path, &records) else {
        return Ok(None);
    };
    if let Some(parent) = bundle_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "无法创建 trace export 输出目录 {}: {error}",
                parent.display()
            )
        })?;
    }
    export::write_trace_export_zip(bundle_path, generated_at, &trace, &records)?;
    Ok(Some(trace))
}

fn trace_event_path(root: &Path, session_id: &str, trace_id: &str) -> PathBuf {
    root.join("sessions")
        .join(format!("session_{}", safe_file_stem(session_id)))
        .join(format!("trace_{}.jsonl", safe_file_stem(trace_id)))
}

fn read_trace_event_records(path: &Path) -> Result<Vec<RawTraceEventRecord>, String> {
    let mut records = Vec::new();
    let file = fs::File::open(path)
        .map_err(|error| format!("无法读取 trace event log {}: {error}", path.display()))?;
    for line in BufReader::new(file).lines() {
        let line =
            line.map_err(|error| format!("无法读取 trace event log {}: {error}", path.display()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let event = serde_json::from_str::<RawTraceEvent>(trimmed)
            .map_err(|error| format!("无法解析 trace event log {}: {error}", path.display()))?;
        records.push(RawTraceEventRecord {
            path: path.to_path_buf(),
            event,
        });
    }
    Ok(records)
}

fn trace_summary_from_records(
    path: &Path,
    records: &[RawTraceEventRecord],
) -> Option<DiagnosticsTraceSummary> {
    let first = records.first()?;
    let last = records.last().unwrap_or(first);
    let metadata = fs::metadata(path).ok();
    Some(DiagnosticsTraceSummary {
        session_id: first.event.session_id.clone(),
        trace_id: first.event.trace_id.clone(),
        path: trace_logical_path(path, &first.event.session_id, &first.event.trace_id),
        size_bytes: metadata.as_ref().map(fs::Metadata::len).unwrap_or_default(),
        event_count: records.len() as u64,
        first_wall_time_unix_ms: Some(first.event.wall_time_unix_ms),
        last_wall_time_unix_ms: Some(last.event.wall_time_unix_ms),
        modified_at: metadata
            .and_then(|metadata| metadata.modified().ok())
            .map(system_time_rfc3339),
    })
}

fn trace_logical_path(path: &Path, session_id: &str, trace_id: &str) -> String {
    let session_stem = safe_file_stem(session_id);
    path.file_name()
        .and_then(|value| value.to_str())
        .filter(|name| name.ends_with(".jsonl"))
        .map(|name| format!("sessions/session_{session_stem}/{name}"))
        .unwrap_or_else(|| {
            format!(
                "sessions/session_{}/trace_{}.jsonl",
                session_stem,
                safe_file_stem(trace_id)
            )
        })
}

fn protocol_trace_event(event: RawTraceEvent) -> DiagnosticsTraceEvent {
    DiagnosticsTraceEvent {
        schema_version: event.schema_version,
        seq: event.seq,
        wall_time_unix_ms: event.wall_time_unix_ms,
        trace_id: event.trace_id,
        run_id: event.run_id,
        request_id: event.request_id,
        session_id: event.session_id,
        thread_id: event.thread_id,
        turn_id: event.turn_id,
        event_id: event.event_id,
        event_sequence: event.event_sequence,
        event_type: event.event_type,
        checkpoint: event.checkpoint,
        metrics: event.metrics,
        redaction: protocol_redaction_policy(),
    }
}

fn protocol_redaction_policy() -> DiagnosticsTraceRedactionPolicy {
    DiagnosticsTraceRedactionPolicy {
        mode: "summary_only".to_string(),
        raw_agent_event_payload: false,
        prompt_text: false,
        provider_payload: false,
    }
}

fn normalize_limit(value: Option<usize>, default_value: usize, max_value: usize) -> usize {
    value.unwrap_or(default_value).clamp(1, max_value)
}

fn system_time_rfc3339(value: SystemTime) -> String {
    DateTime::<Utc>::from(value).to_rfc3339_opts(SecondsFormat::Millis, true)
}

impl TraceEventWriterState {
    fn next_seq_for_path(&mut self, path: &Path) -> Result<u64, String> {
        if let Some(next_seq) = self.next_seq_by_path.get(path) {
            if path.exists() {
                return Ok(*next_seq);
            }
            self.next_seq_by_path.insert(path.to_path_buf(), 1);
            return Ok(1);
        }
        let next_seq = count_existing_events(path)?.saturating_add(1);
        self.next_seq_by_path.insert(path.to_path_buf(), next_seq);
        Ok(next_seq)
    }

    fn advance_next_seq(&mut self, path: &Path, start_seq: u64, event_count: usize) {
        self.next_seq_by_path.insert(
            path.to_path_buf(),
            start_seq.saturating_add(event_count as u64),
        );
    }
}

fn raw_trace_event_from_agent_event(event: &AgentEvent) -> Option<(String, RawTraceEvent)> {
    let payload = event.payload.as_object()?;
    let trace = payload.get("trace")?.as_object()?;
    let trace_id = string_field(trace, &["traceId", "trace_id"])
        .or_else(|| string_field(payload, &["trace_id"]))?;
    let checkpoint = string_field(trace, &["checkpoint"])?;
    let wall_time_unix_ms = number_field(payload, &["server_event_emitted_at"])
        .or_else(|| timestamp_ms(&event.timestamp))
        .unwrap_or_default();

    let raw_event = RawTraceEvent {
        schema_version: RAW_TRACE_EVENT_SCHEMA_VERSION,
        seq: 0,
        wall_time_unix_ms,
        trace_id: trace_id.clone(),
        run_id: string_field(trace, &["runId", "run_id"])
            .or_else(|| string_field(payload, &["run_id"])),
        request_id: string_field(trace, &["requestId", "request_id"])
            .or_else(|| string_field(payload, &["request_id"])),
        session_id: event.session_id.clone(),
        thread_id: event.thread_id.clone(),
        turn_id: event.turn_id.clone(),
        event_id: event.event_id.clone(),
        event_sequence: event.sequence,
        event_type: event.event_type.clone(),
        checkpoint,
        metrics: safe_metrics_for_trace_event(payload, trace),
        redaction: RawTraceRedactionPolicy {
            mode: "summary_only".to_string(),
            raw_agent_event_payload: false,
            prompt_text: false,
            provider_payload: false,
        },
    };
    Some((trace_id, raw_event))
}

fn safe_metrics_for_trace_event(
    payload: &Map<String, Value>,
    trace: &Map<String, Value>,
) -> BTreeMap<String, Value> {
    let mut metrics = BTreeMap::new();
    for key in [
        "attempt",
        "backend",
        "elapsed_ms",
        "failure_category",
        "model",
        "provider",
        "provider_request_id",
        "provider_request_id_header",
        "retryable",
        "source",
        "stage",
        "status",
        "text_chars",
        "server_event_emitted_at",
    ] {
        insert_scalar_metric(&mut metrics, key, payload.get(key));
    }
    insert_scalar_metric(&mut metrics, "submitted_at", trace.get("submittedAt"));
    insert_scalar_metric(&mut metrics, "w3c_trace_id", trace.get("w3cTraceId"));
    insert_scalar_metric(&mut metrics, "w3c_traceparent", trace.get("w3cTraceparent"));
    insert_text_length_metric(&mut metrics, "text_chars", payload.get("text"));
    insert_nested_text_length_metric(
        &mut metrics,
        "input_text_chars",
        payload.get("input"),
        "text",
    );
    metrics
}

fn insert_scalar_metric(metrics: &mut BTreeMap<String, Value>, key: &str, value: Option<&Value>) {
    let Some(value) = value else {
        return;
    };
    if value.is_string() || value.is_number() || value.is_boolean() || value.is_null() {
        metrics.insert(key.to_string(), value.clone());
    }
}

fn insert_text_length_metric(
    metrics: &mut BTreeMap<String, Value>,
    key: &str,
    value: Option<&Value>,
) {
    if metrics.contains_key(key) {
        return;
    }
    let Some(text) = value.and_then(Value::as_str) else {
        return;
    };
    metrics.insert(key.to_string(), Value::Number(text.chars().count().into()));
}

fn insert_nested_text_length_metric(
    metrics: &mut BTreeMap<String, Value>,
    key: &str,
    value: Option<&Value>,
    text_key: &str,
) {
    let Some(text) = value
        .and_then(Value::as_object)
        .and_then(|object| object.get(text_key))
        .and_then(Value::as_str)
    else {
        return;
    };
    metrics.insert(key.to_string(), Value::Number(text.chars().count().into()));
}

fn append_raw_trace_events_to_path(path: &Path, events: &[RawTraceEvent]) -> Result<(), String> {
    if events.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "无法创建 trace event log 父目录 {}: {error}",
                parent.display()
            )
        })?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("无法打开 trace event log {}: {error}", path.display()))?;
    for event in events {
        let json = serde_json::to_vec(event).map_err(|error| {
            format!(
                "无法序列化 trace event {}:{}: {error}",
                event.trace_id, event.event_id
            )
        })?;
        file.write_all(&json)
            .and_then(|_| file.write_all(b"\n"))
            .map_err(|error| format!("无法写入 trace event log {}: {error}", path.display()))?;
    }
    Ok(())
}

fn count_existing_events(path: &Path) -> Result<u64, String> {
    if !path.exists() {
        return Ok(0);
    }
    let file = fs::File::open(path)
        .map_err(|error| format!("无法读取 trace event log {}: {error}", path.display()))?;
    let mut count = 0_u64;
    for line in BufReader::new(file).lines() {
        let line =
            line.map_err(|error| format!("无法读取 trace event log {}: {error}", path.display()))?;
        if !line.trim().is_empty() {
            count = count.saturating_add(1);
        }
    }
    Ok(count)
}

fn string_field(payload: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| payload.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn number_field(payload: &Map<String, Value>, keys: &[&str]) -> Option<i64> {
    keys.iter()
        .filter_map(|key| payload.get(*key))
        .find_map(|value| {
            value
                .as_i64()
                .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
        })
}

fn timestamp_ms(timestamp: &str) -> Option<i64> {
    DateTime::parse_from_rfc3339(timestamp)
        .ok()
        .map(|value| value.timestamp_millis())
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
