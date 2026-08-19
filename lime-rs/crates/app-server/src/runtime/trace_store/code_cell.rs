use super::{
    append_raw_trace_events_to_path, read_trace_event_records, RawTraceEvent, RawTraceEventRecord,
    RawTraceRedactionPolicy, TraceEventWriter, RAW_TRACE_EVENT_SCHEMA_VERSION,
};
use chrono::Utc;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use tool_runtime::tool_lifecycle::{CodeCellRuntimeStatus, CodeCellTraceEvent};
use uuid::Uuid;

const SOURCE_ITEM_OBSERVED: &str = "code_cell.source_item_observed";
const OUTPUT_ITEM_OBSERVED: &str = "code_cell.output_item_observed";
const STARTED: &str = "code_cell.started";
const INITIAL_RESPONSE: &str = "code_cell.initial_response";
const ENDED: &str = "code_cell.ended";
const NESTED_TOOL_STARTED: &str = "code_cell.nested_tool_started";
const NESTED_TOOL_ENDED: &str = "code_cell.nested_tool_ended";
const WAIT_TOOL_OBSERVED: &str = "code_cell.wait_tool_observed";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CodeCellTraceProjection {
    pub(crate) code_cells: BTreeMap<String, CodeCellTrace>,
    pub(crate) pending_code_cell_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CodeCellTrace {
    pub(crate) code_cell_id: String,
    pub(crate) model_visible_call_id: String,
    pub(crate) thread_id: String,
    pub(crate) started_by_turn_id: String,
    pub(crate) source_item_id: String,
    pub(crate) output_item_ids: Vec<String>,
    pub(crate) runtime_cell_id: String,
    pub(crate) runtime_status: CodeCellRuntimeStatus,
    pub(crate) started_seq: u64,
    pub(crate) initial_response_seq: Option<u64>,
    pub(crate) yielded_seq: Option<u64>,
    pub(crate) ended_seq: Option<u64>,
    pub(crate) source_js_chars: usize,
    pub(crate) source_js_sha256: String,
    pub(crate) nested_tool_call_ids: Vec<String>,
    pub(crate) nested_tool_statuses: BTreeMap<String, String>,
    pub(crate) wait_tool_call_ids: Vec<String>,
}

#[derive(Debug, Clone)]
struct CodeCellState {
    code_cell_id: String,
    model_visible_call_id: String,
    thread_id: String,
    started_by_turn_id: String,
    source_item_id: Option<String>,
    output_item_ids: Vec<String>,
    runtime_cell_id: String,
    runtime_status: CodeCellRuntimeStatus,
    started_seq: u64,
    initial_response_seq: Option<u64>,
    yielded_seq: Option<u64>,
    ended_seq: Option<u64>,
    source_js_chars: usize,
    source_js_sha256: String,
    nested_tool_call_ids: Vec<String>,
    nested_tool_statuses: BTreeMap<String, String>,
    wait_tool_call_ids: Vec<String>,
}

#[derive(Debug, Default)]
struct CodeCellTraceReducer {
    cells: BTreeMap<String, CodeCellState>,
    source_items: HashMap<String, String>,
    runtime_cells: HashMap<String, String>,
}

pub(super) fn trace_id(thread_id: &str) -> String {
    format!("code-cell-{thread_id}")
}

pub(super) fn append(
    writer: &TraceEventWriter,
    session_id: &str,
    thread_id: &str,
    event: CodeCellTraceEvent,
) -> Result<(), String> {
    append_batch(writer, session_id, thread_id, vec![event])
}

pub(super) fn close_for_turn(
    writer: &TraceEventWriter,
    session_id: &str,
    thread_id: &str,
    turn_id: &str,
    status: CodeCellRuntimeStatus,
) -> Result<(), String> {
    if !status.is_terminal() {
        return Err("CodeCell turn terminal status must be terminal".to_string());
    }
    let trace_id = trace_id(thread_id);
    let path = writer.trace_path(session_id, &trace_id);
    let mut state = writer.lock_state();
    let records = if path.exists() {
        read_trace_event_records(&path)?
    } else {
        Vec::new()
    };
    let mut reducer = CodeCellTraceReducer::replay(&records)?;
    let events = reducer
        .cells
        .values()
        .filter(|cell| cell.started_by_turn_id == turn_id && cell.ended_seq.is_none())
        .map(|cell| CodeCellTraceEvent::Ended {
            turn_id: turn_id.to_string(),
            runtime_cell_id: cell.runtime_cell_id.clone(),
            status,
            response_chars: 0,
        })
        .collect::<Vec<_>>();
    if events.is_empty() {
        return Ok(());
    }
    append_validated_batch(
        writer,
        &mut state,
        &path,
        session_id,
        thread_id,
        &trace_id,
        &mut reducer,
        events,
    )
}

pub(super) fn read_projection(
    writer: &TraceEventWriter,
    session_id: &str,
    thread_id: &str,
) -> Result<CodeCellTraceProjection, String> {
    let path = writer.trace_path(session_id, &trace_id(thread_id));
    let records = if path.exists() {
        read_trace_event_records(&path)?
    } else {
        Vec::new()
    };
    CodeCellTraceReducer::replay(&records).map(CodeCellTraceReducer::finish)
}

fn append_batch(
    writer: &TraceEventWriter,
    session_id: &str,
    thread_id: &str,
    events: Vec<CodeCellTraceEvent>,
) -> Result<(), String> {
    if session_id.trim().is_empty() || thread_id.trim().is_empty() {
        return Err("CodeCell trace requires non-empty session and thread identities".to_string());
    }
    let trace_id = trace_id(thread_id);
    let path = writer.trace_path(session_id, &trace_id);
    let mut state = writer.lock_state();
    let records = if path.exists() {
        read_trace_event_records(&path)?
    } else {
        Vec::new()
    };
    let mut reducer = CodeCellTraceReducer::replay(&records)?;
    append_validated_batch(
        writer,
        &mut state,
        &path,
        session_id,
        thread_id,
        &trace_id,
        &mut reducer,
        events,
    )
}

#[allow(clippy::too_many_arguments)]
fn append_validated_batch(
    writer: &TraceEventWriter,
    state: &mut super::TraceEventWriterState,
    path: &std::path::Path,
    session_id: &str,
    thread_id: &str,
    trace_id: &str,
    reducer: &mut CodeCellTraceReducer,
    events: Vec<CodeCellTraceEvent>,
) -> Result<(), String> {
    let should_enforce_retention = !path.exists();
    let start_seq = state.next_seq_for_path(path)?;
    let mut raw_events = Vec::with_capacity(events.len());
    for (index, event) in events.into_iter().enumerate() {
        let seq = start_seq.saturating_add(index as u64);
        let raw = raw_event(session_id, thread_id, trace_id, seq, event)?;
        reducer.apply(&raw)?;
        raw_events.push(raw);
    }
    append_raw_trace_events_to_path(path, &raw_events)?;
    state.advance_next_seq(path, start_seq, raw_events.len());
    if should_enforce_retention {
        writer.enforce_session_retention(session_id)?;
    }
    Ok(())
}

fn raw_event(
    session_id: &str,
    thread_id: &str,
    trace_id: &str,
    seq: u64,
    event: CodeCellTraceEvent,
) -> Result<RawTraceEvent, String> {
    let mut metrics = BTreeMap::new();
    let (event_type, turn_id) = match event {
        CodeCellTraceEvent::SourceItemObserved {
            turn_id,
            model_visible_call_id,
            source_item_id,
        } => {
            insert_string(&mut metrics, "model_visible_call_id", model_visible_call_id);
            insert_string(&mut metrics, "source_item_id", source_item_id);
            (SOURCE_ITEM_OBSERVED, turn_id)
        }
        CodeCellTraceEvent::OutputItemObserved {
            turn_id,
            runtime_cell_id,
            output_item_id,
        } => {
            insert_string(&mut metrics, "runtime_cell_id", runtime_cell_id);
            insert_string(&mut metrics, "output_item_id", output_item_id);
            (OUTPUT_ITEM_OBSERVED, turn_id)
        }
        CodeCellTraceEvent::Started {
            turn_id,
            runtime_cell_id,
            model_visible_call_id,
            source_js,
        } => {
            insert_string(&mut metrics, "runtime_cell_id", runtime_cell_id);
            insert_string(&mut metrics, "model_visible_call_id", model_visible_call_id);
            metrics.insert(
                "source_js_chars".to_string(),
                Value::from(source_js.chars().count() as u64),
            );
            insert_string(
                &mut metrics,
                "source_js_sha256",
                hex::encode(Sha256::digest(source_js.as_bytes())),
            );
            (STARTED, turn_id)
        }
        CodeCellTraceEvent::InitialResponse {
            turn_id,
            runtime_cell_id,
            status,
            response_chars,
        } => {
            insert_string(&mut metrics, "runtime_cell_id", runtime_cell_id);
            insert_status(&mut metrics, status);
            metrics.insert(
                "response_chars".to_string(),
                Value::from(response_chars as u64),
            );
            (INITIAL_RESPONSE, turn_id)
        }
        CodeCellTraceEvent::Ended {
            turn_id,
            runtime_cell_id,
            status,
            response_chars,
        } => {
            insert_string(&mut metrics, "runtime_cell_id", runtime_cell_id);
            insert_status(&mut metrics, status);
            metrics.insert(
                "response_chars".to_string(),
                Value::from(response_chars as u64),
            );
            (ENDED, turn_id)
        }
        CodeCellTraceEvent::NestedToolStarted {
            turn_id,
            runtime_cell_id,
            tool_call_id,
            runtime_tool_call_id,
            tool_name,
        } => {
            insert_string(&mut metrics, "runtime_cell_id", runtime_cell_id);
            insert_string(&mut metrics, "tool_call_id", tool_call_id);
            insert_string(&mut metrics, "runtime_tool_call_id", runtime_tool_call_id);
            insert_string(&mut metrics, "tool_name", tool_name);
            (NESTED_TOOL_STARTED, turn_id)
        }
        CodeCellTraceEvent::NestedToolEnded {
            turn_id,
            runtime_cell_id,
            tool_call_id,
            status,
        } => {
            insert_string(&mut metrics, "runtime_cell_id", runtime_cell_id);
            insert_string(&mut metrics, "tool_call_id", tool_call_id);
            insert_string(&mut metrics, "status", status);
            (NESTED_TOOL_ENDED, turn_id)
        }
        CodeCellTraceEvent::WaitToolObserved {
            turn_id,
            runtime_cell_id,
            tool_call_id,
        } => {
            insert_string(&mut metrics, "runtime_cell_id", runtime_cell_id);
            insert_string(&mut metrics, "tool_call_id", tool_call_id);
            (WAIT_TOOL_OBSERVED, turn_id)
        }
    };
    Ok(RawTraceEvent {
        schema_version: RAW_TRACE_EVENT_SCHEMA_VERSION,
        seq,
        wall_time_unix_ms: Utc::now().timestamp_millis(),
        trace_id: trace_id.to_string(),
        run_id: None,
        request_id: None,
        session_id: session_id.to_string(),
        thread_id: Some(thread_id.to_string()),
        turn_id: Some(turn_id),
        event_id: format!("code-cell-trace-{}", Uuid::now_v7()),
        event_sequence: seq,
        event_type: event_type.to_string(),
        checkpoint: event_type.to_string(),
        metrics,
        redaction: RawTraceRedactionPolicy {
            mode: "summary_only".to_string(),
            raw_agent_event_payload: false,
            prompt_text: false,
            provider_payload: false,
        },
    })
}

impl CodeCellTraceReducer {
    fn replay(records: &[RawTraceEventRecord]) -> Result<Self, String> {
        let mut reducer = Self::default();
        for record in records {
            if record.event.event_type.starts_with("code_cell.") {
                reducer.apply(&record.event)?;
            }
        }
        Ok(reducer)
    }

    fn apply(&mut self, event: &RawTraceEvent) -> Result<(), String> {
        match event.event_type.as_str() {
            SOURCE_ITEM_OBSERVED => self.observe_source_item(event),
            OUTPUT_ITEM_OBSERVED => self.observe_output_item(event),
            STARTED => self.start(event),
            INITIAL_RESPONSE => self.initial_response(event),
            ENDED => self.end(event),
            NESTED_TOOL_STARTED => self.nested_tool_started(event),
            NESTED_TOOL_ENDED => self.nested_tool_ended(event),
            WAIT_TOOL_OBSERVED => self.wait_tool_observed(event),
            _ => Ok(()),
        }
    }

    fn observe_source_item(&mut self, event: &RawTraceEvent) -> Result<(), String> {
        let call_id = metric_string(event, "model_visible_call_id")?;
        let source_item_id = metric_string(event, "source_item_id")?;
        if let Some(existing) = self.source_items.get(call_id) {
            return if existing == source_item_id {
                Err(format!("duplicate CodeCell source item for call {call_id}"))
            } else {
                Err(format!(
                    "conflicting CodeCell source item for call {call_id}"
                ))
            };
        }
        self.source_items
            .insert(call_id.to_string(), source_item_id.to_string());
        if let Some(cell) = self
            .cells
            .values_mut()
            .find(|cell| cell.model_visible_call_id == call_id)
        {
            cell.source_item_id = Some(source_item_id.to_string());
        }
        Ok(())
    }

    fn observe_output_item(&mut self, event: &RawTraceEvent) -> Result<(), String> {
        let runtime_cell_id = metric_string(event, "runtime_cell_id")?;
        let output_item_id = metric_string(event, "output_item_id")?;
        let Some(cell) = self.cell_mut(runtime_cell_id) else {
            return Ok(());
        };
        push_unique(&mut cell.output_item_ids, output_item_id);
        Ok(())
    }

    fn start(&mut self, event: &RawTraceEvent) -> Result<(), String> {
        let thread_id = event
            .thread_id
            .as_deref()
            .ok_or_else(|| "CodeCell start is missing thread identity".to_string())?;
        let turn_id = event
            .turn_id
            .as_deref()
            .ok_or_else(|| "CodeCell start is missing turn identity".to_string())?;
        let runtime_cell_id = metric_string(event, "runtime_cell_id")?;
        let call_id = metric_string(event, "model_visible_call_id")?;
        let code_cell_id = code_cell_id(thread_id, call_id);
        if self.cells.contains_key(&code_cell_id) {
            return Err(format!("duplicate CodeCell start for {code_cell_id}"));
        }
        if self.runtime_cells.contains_key(runtime_cell_id) {
            return Err(format!(
                "duplicate runtime CodeCell id {runtime_cell_id} in thread {thread_id}"
            ));
        }
        let source_item_id = self.source_items.get(call_id).cloned();
        self.runtime_cells
            .insert(runtime_cell_id.to_string(), code_cell_id.clone());
        self.cells.insert(
            code_cell_id.clone(),
            CodeCellState {
                code_cell_id,
                model_visible_call_id: call_id.to_string(),
                thread_id: thread_id.to_string(),
                started_by_turn_id: turn_id.to_string(),
                source_item_id,
                output_item_ids: Vec::new(),
                runtime_cell_id: runtime_cell_id.to_string(),
                runtime_status: CodeCellRuntimeStatus::Starting,
                started_seq: event.seq,
                initial_response_seq: None,
                yielded_seq: None,
                ended_seq: None,
                source_js_chars: metric_usize(event, "source_js_chars")?,
                source_js_sha256: metric_string(event, "source_js_sha256")?.to_string(),
                nested_tool_call_ids: Vec::new(),
                nested_tool_statuses: BTreeMap::new(),
                wait_tool_call_ids: Vec::new(),
            },
        );
        Ok(())
    }

    fn initial_response(&mut self, event: &RawTraceEvent) -> Result<(), String> {
        let runtime_cell_id = metric_string(event, "runtime_cell_id")?;
        let status = metric_status(event)?;
        let cell = self.known_live_cell_mut(runtime_cell_id, "initial response")?;
        if cell.initial_response_seq.is_some() {
            return Err(format!(
                "duplicate CodeCell initial response for {}",
                cell.code_cell_id
            ));
        }
        cell.initial_response_seq = Some(event.seq);
        if status == CodeCellRuntimeStatus::Yielded {
            cell.yielded_seq = Some(event.seq);
        }
        cell.runtime_status = status;
        Ok(())
    }

    fn end(&mut self, event: &RawTraceEvent) -> Result<(), String> {
        let runtime_cell_id = metric_string(event, "runtime_cell_id")?;
        let status = metric_status(event)?;
        if !status.is_terminal() {
            return Err(format!("CodeCell end status is not terminal: {status:?}"));
        }
        let cell = self.known_live_cell_mut(runtime_cell_id, "end")?;
        if cell.initial_response_seq.is_none() {
            cell.initial_response_seq = Some(event.seq);
        }
        cell.ended_seq = Some(event.seq);
        cell.runtime_status = status;
        Ok(())
    }

    fn nested_tool_started(&mut self, event: &RawTraceEvent) -> Result<(), String> {
        let runtime_cell_id = metric_string(event, "runtime_cell_id")?;
        let tool_call_id = metric_string(event, "tool_call_id")?;
        let cell = self.known_live_cell_mut(runtime_cell_id, "nested tool start")?;
        if cell
            .nested_tool_call_ids
            .iter()
            .any(|id| id == tool_call_id)
        {
            return Err(format!("duplicate nested tool start for {tool_call_id}"));
        }
        cell.nested_tool_call_ids.push(tool_call_id.to_string());
        cell.nested_tool_statuses
            .insert(tool_call_id.to_string(), "running".to_string());
        Ok(())
    }

    fn nested_tool_ended(&mut self, event: &RawTraceEvent) -> Result<(), String> {
        let runtime_cell_id = metric_string(event, "runtime_cell_id")?;
        let tool_call_id = metric_string(event, "tool_call_id")?;
        let status = metric_string(event, "status")?;
        let cell = self.known_live_cell_mut(runtime_cell_id, "nested tool end")?;
        if !cell
            .nested_tool_call_ids
            .iter()
            .any(|id| id == tool_call_id)
        {
            return Err(format!(
                "nested tool end referenced unknown call {tool_call_id}"
            ));
        }
        let current = cell.nested_tool_statuses.get(tool_call_id);
        if current.is_some_and(|current| current != "running") {
            return Err(format!("duplicate nested tool end for {tool_call_id}"));
        }
        cell.nested_tool_statuses
            .insert(tool_call_id.to_string(), status.to_string());
        Ok(())
    }

    fn wait_tool_observed(&mut self, event: &RawTraceEvent) -> Result<(), String> {
        let runtime_cell_id = metric_string(event, "runtime_cell_id")?;
        let tool_call_id = metric_string(event, "tool_call_id")?;
        let Some(cell) = self.cell_mut(runtime_cell_id) else {
            return Ok(());
        };
        push_unique(&mut cell.wait_tool_call_ids, tool_call_id);
        Ok(())
    }

    fn known_live_cell_mut(
        &mut self,
        runtime_cell_id: &str,
        operation: &str,
    ) -> Result<&mut CodeCellState, String> {
        let cell = self.cell_mut(runtime_cell_id).ok_or_else(|| {
            format!("CodeCell {operation} referenced unknown runtime cell {runtime_cell_id}")
        })?;
        if cell.ended_seq.is_some() {
            return Err(format!(
                "late CodeCell {operation} referenced terminal cell {}",
                cell.code_cell_id
            ));
        }
        Ok(cell)
    }

    fn cell_mut(&mut self, runtime_cell_id: &str) -> Option<&mut CodeCellState> {
        let code_cell_id = self.runtime_cells.get(runtime_cell_id)?.clone();
        self.cells.get_mut(&code_cell_id)
    }

    fn finish(self) -> CodeCellTraceProjection {
        let mut code_cells = BTreeMap::new();
        let mut pending_code_cell_ids = Vec::new();
        for (id, cell) in self.cells {
            let Some(source_item_id) = cell.source_item_id else {
                pending_code_cell_ids.push(id);
                continue;
            };
            code_cells.insert(
                id,
                CodeCellTrace {
                    code_cell_id: cell.code_cell_id,
                    model_visible_call_id: cell.model_visible_call_id,
                    thread_id: cell.thread_id,
                    started_by_turn_id: cell.started_by_turn_id,
                    source_item_id,
                    output_item_ids: cell.output_item_ids,
                    runtime_cell_id: cell.runtime_cell_id,
                    runtime_status: cell.runtime_status,
                    started_seq: cell.started_seq,
                    initial_response_seq: cell.initial_response_seq,
                    yielded_seq: cell.yielded_seq,
                    ended_seq: cell.ended_seq,
                    source_js_chars: cell.source_js_chars,
                    source_js_sha256: cell.source_js_sha256,
                    nested_tool_call_ids: cell.nested_tool_call_ids,
                    nested_tool_statuses: cell.nested_tool_statuses,
                    wait_tool_call_ids: cell.wait_tool_call_ids,
                },
            );
        }
        CodeCellTraceProjection {
            code_cells,
            pending_code_cell_ids,
        }
    }
}

fn code_cell_id(thread_id: &str, call_id: &str) -> String {
    format!("{thread_id}:{call_id}")
}

fn insert_string(metrics: &mut BTreeMap<String, Value>, key: &str, value: String) {
    metrics.insert(key.to_string(), Value::String(value));
}

fn insert_status(metrics: &mut BTreeMap<String, Value>, status: CodeCellRuntimeStatus) {
    metrics.insert(
        "status".to_string(),
        Value::String(
            serde_json::to_value(status)
                .ok()
                .and_then(|value| value.as_str().map(str::to_string))
                .unwrap_or_else(|| "failed".to_string()),
        ),
    );
}

fn metric_string<'a>(event: &'a RawTraceEvent, key: &str) -> Result<&'a str, String> {
    event
        .metrics
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{} trace event is missing {key}", event.event_type))
}

fn metric_usize(event: &RawTraceEvent, key: &str) -> Result<usize, String> {
    event
        .metrics
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("{} trace event has invalid {key}", event.event_type))
}

fn metric_status(event: &RawTraceEvent) -> Result<CodeCellRuntimeStatus, String> {
    let value = metric_string(event, "status")?;
    serde_json::from_value(Value::String(value.to_string())).map_err(|error| {
        format!(
            "{} trace event has invalid status: {error}",
            event.event_type
        )
    })
}

fn push_unique(values: &mut Vec<String>, value: &str) {
    if !values.iter().any(|existing| existing == value) {
        values.push(value.to_string());
    }
}
