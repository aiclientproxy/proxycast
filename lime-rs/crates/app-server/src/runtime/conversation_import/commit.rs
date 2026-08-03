use super::codex::{self, events::CodexRolloutEvent, CodexTimelineItem};
use super::import_status;
use super::provenance::{self, ImportProvenance};
use crate::runtime::{new_id, timestamp};
use crate::{RuntimeCore, RuntimeCoreError};
use agent_protocol::{AgentInput, ImageDetail};
use app_server_protocol::{
    AgentAttachment, AgentSessionStartParams, AgentSessionStatus, AgentTurn, AgentTurnStatus,
    BusinessObjectRef, ConversationImportJobPhase, ConversationImportJobProgress,
    ConversationImportSourceClient, ConversationImportSourceProvenance,
    ConversationImportThreadCommitParams, ConversationImportThreadCommitResponse,
    ConversationImportThreadStatus,
};
use serde_json::json;

const DEFAULT_IMPORT_APP_ID: &str = "content-studio";

#[cfg(test)]
pub(super) fn commit_conversation_import_thread(
    core: &RuntimeCore,
    params: ConversationImportThreadCommitParams,
) -> Result<ConversationImportThreadCommitResponse, RuntimeCoreError> {
    commit_conversation_import_thread_with_progress(core, params, &mut |_| Ok(()))
}

pub(super) fn commit_conversation_import_thread_with_progress(
    core: &RuntimeCore,
    params: ConversationImportThreadCommitParams,
    report_progress: &mut dyn FnMut(ConversationImportJobProgress) -> Result<(), RuntimeCoreError>,
) -> Result<ConversationImportThreadCommitResponse, RuntimeCoreError> {
    if !params.confirmed {
        return Err(RuntimeCoreError::Backend(
            "conversation import commit requires explicit user confirmation".to_string(),
        ));
    }

    commit_codex_thread(core, params, report_progress)
}

fn commit_codex_thread(
    core: &RuntimeCore,
    params: ConversationImportThreadCommitParams,
    report_progress: &mut dyn FnMut(ConversationImportJobProgress) -> Result<(), RuntimeCoreError>,
) -> Result<ConversationImportThreadCommitResponse, RuntimeCoreError> {
    report_progress(import_progress(
        ConversationImportJobPhase::ReadingSource,
        0,
        0,
        0,
        0,
    ))?;
    let source_root = codex::resolve_home(params.source_root.as_deref()).ok_or_else(|| {
        RuntimeCoreError::Backend("unable to resolve source home directory".to_string())
    })?;
    if !source_root.is_dir() {
        return Err(RuntimeCoreError::Backend(
            "source home directory does not exist".to_string(),
        ));
    }

    let (source_path, indexed_thread) = resolve_source_path(&source_root, &params)?;
    let mut preview = codex::parse_rollout_for_import(&source_path)?;
    if let Some(indexed_thread) = indexed_thread {
        codex::merge_indexed_thread_metadata(&mut preview.thread, indexed_thread);
    }
    preview.thread.source_path = Some(codex::path_to_string(&source_path));
    preview.thread.import_status = ConversationImportThreadStatus::Imported;

    let turns = history_turns(&preview.timeline);
    if turns.is_empty() {
        return Err(RuntimeCoreError::Backend(
            "source rollout does not contain importable user messages".to_string(),
        ));
    }
    let total_turns = turns.len();
    let total_items = turns
        .iter()
        .map(|turn| turn.events.len().saturating_add(1))
        .sum();
    report_progress(import_progress(
        ConversationImportJobPhase::BuildingHistory,
        0,
        total_items,
        0,
        total_turns,
    ))?;
    if let Some(session) = import_status::imported_session_for_thread(
        core,
        ConversationImportSourceClient::Codex,
        &preview.thread.source_thread_id,
    ) {
        if params.replace_existing.unwrap_or(false) {
            clear_existing_imported_session(core, &session.session_id)?;
        } else {
            preview.thread.import_status = ConversationImportThreadStatus::Imported;
            preview.summary.dry_run.will_create_session = false;
            preview.summary.dry_run.will_append_to_existing_session = true;
            let warnings = preview.summary.warnings.clone();
            return Ok(ConversationImportThreadCommitResponse {
                session,
                thread: preview.thread,
                imported_messages: preview.summary.dry_run.will_import_messages,
                imported_turns: preview.summary.dry_run.will_import_turns,
                summary: preview.summary,
                can_continue: true,
                warnings,
            });
        }
    }
    preview.summary.fidelity.budget_dropped = 0;

    let provenance = ImportProvenance::for_thread(
        &preview.thread,
        &source_root,
        &source_path,
        &preview.summary.fidelity,
    );
    let prepared_turns = prepare_history_turns(turns)?;
    report_progress(import_progress(
        ConversationImportJobPhase::PersistingHistory,
        0,
        total_items,
        0,
        total_turns,
    ))?;

    let created_session = core
        .start_session(AgentSessionStartParams {
            session_id: None,
            thread_id: None,
            app_id: params
                .app_id
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_IMPORT_APP_ID.to_string()),
            workspace_id: params.workspace_id.filter(|value| !value.trim().is_empty()),
            business_object_ref: Some(import_business_object_ref(
                preview.thread.source_thread_id.as_str(),
                preview.thread.title.as_deref(),
                &source_root,
                &source_path,
                &preview.thread,
                &preview.summary.fidelity,
            )),
            locale: None,
        })?
        .session;

    let commit_result = (|| {
        let imported_turns = append_history_turns(
            core,
            &created_session.session_id,
            prepared_turns,
            &provenance,
            report_progress,
        )?;
        let imported_messages = preview.summary.dry_run.will_import_messages;
        let warnings = provenance::commit_warnings(
            &preview.summary.warnings,
            preview.summary.unsupported_count,
            preview.summary.rollout_event_items,
        );

        report_progress(import_progress(
            ConversationImportJobPhase::Finalizing,
            total_items,
            total_items,
            imported_turns,
            total_turns,
        ))?;
        let (mut session, _) = core.session_snapshot(&created_session.session_id)?;
        session.status = AgentSessionStatus::Completed;

        Ok(ConversationImportThreadCommitResponse {
            session,
            thread: preview.thread,
            summary: preview.summary,
            imported_messages,
            imported_turns,
            can_continue: true,
            warnings,
        })
    })();

    match commit_result {
        Ok(response) => Ok(response),
        Err(import_error) => {
            match clear_existing_imported_session(core, &created_session.session_id) {
                Ok(()) => Err(import_error),
                Err(cleanup_error) => Err(RuntimeCoreError::Backend(format!(
                    "Codex import failed after creating session {}: {import_error}; compensating cleanup also failed: {cleanup_error}",
                    created_session.session_id
                ))),
            }
        }
    }
}

fn clear_existing_imported_session(
    core: &RuntimeCore,
    session_id: &str,
) -> Result<(), RuntimeCoreError> {
    let mut errors = Vec::new();
    if let Some(projection_store) = core.projection_store.as_ref() {
        if let Err(error) = projection_store.delete_session_data(session_id) {
            errors.push(format!("projection cleanup failed: {error}"));
        }
    }
    {
        let mut state = core
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        super::super::approval_cache::remove_session(&mut state.session_approval_cache, session_id);
        if let Some(thread_id) = state
            .sessions
            .get(session_id)
            .map(|stored| stored.session.thread_id.clone())
        {
            state.thread_elicitation_counts.remove(&thread_id);
        }
        state.sessions.remove(session_id);
    }
    if let Some(event_log_writer) = core.event_log_writer.as_ref() {
        if let Err(error) = event_log_writer.clear_session(session_id) {
            errors.push(format!("event log cleanup failed: {error}"));
        }
    }
    if let Some(sidecar_store) = core.sidecar_store.as_ref() {
        if let Err(error) = sidecar_store.clear_session(session_id) {
            errors.push(format!("sidecar cleanup failed: {error}"));
        }
    }
    if !errors.is_empty() {
        return Err(RuntimeCoreError::Backend(format!(
            "failed to clear imported session {session_id}: {}",
            errors.join("; ")
        )));
    }
    Ok(())
}

fn resolve_source_path(
    source_root: &std::path::Path,
    params: &ConversationImportThreadCommitParams,
) -> Result<
    (
        std::path::PathBuf,
        Option<app_server_protocol::ImportedThreadSummary>,
    ),
    RuntimeCoreError,
> {
    let (source_path, indexed_thread) = match codex::normalize_filter(params.source_path.as_deref())
    {
        Some(path) => (
            codex::resolve_user_supplied_rollout_path(source_root, &path).ok_or_else(|| {
                RuntimeCoreError::Backend(
                    "source path must be a Codex rollout JSONL file inside source root".to_string(),
                )
            })?,
            None,
        ),
        None => {
            let thread_id = codex::normalize_filter(params.source_thread_id.as_deref())
                .ok_or_else(|| {
                    RuntimeCoreError::Backend(
                        "conversation import commit requires sourceThreadId or sourcePath"
                            .to_string(),
                    )
                })?;
            let thread = codex::find_thread(source_root, &thread_id).ok_or_else(|| {
                RuntimeCoreError::Backend(
                    "unable to resolve source rollout path for thread".to_string(),
                )
            })?;
            let path = thread
                .source_path
                .as_deref()
                .and_then(|value| codex::resolve_existing_source_path(source_root, Some(value)))
                .ok_or_else(|| {
                    RuntimeCoreError::Backend(
                        "unable to resolve source rollout path for thread".to_string(),
                    )
                })?;
            (path, Some(thread))
        }
    };
    Ok((source_path, indexed_thread))
}

struct HistoryTurn {
    user_text: String,
    user_attachments: Vec<AgentAttachment>,
    user_timestamp: Option<String>,
    completed_timestamp: Option<String>,
    user_provenance: Option<app_server_protocol::ConversationImportSourceProvenance>,
    events: Vec<HistoryTurnEvent>,
}

struct PreparedHistoryTurn {
    user_input: Vec<AgentInput>,
    user_timestamp: Option<String>,
    completed_timestamp: Option<String>,
    user_provenance: Option<app_server_protocol::ConversationImportSourceProvenance>,
    user_message: CodexRolloutEvent,
    events: Vec<CodexRolloutEvent>,
}

enum HistoryTurnEvent {
    AssistantMessage(HistoricalAssistantMessage),
    Runtime {
        event: CodexRolloutEvent,
        timestamp: Option<String>,
    },
}

struct HistoricalAssistantMessage {
    text: String,
    phase: Option<String>,
    source_item_id: Option<String>,
    provenance: Option<app_server_protocol::ConversationImportSourceProvenance>,
    timestamp: Option<String>,
}

fn history_turns(timeline: &[CodexTimelineItem]) -> Vec<HistoryTurn> {
    let mut turns = Vec::new();
    let mut pending_user: Option<(
        String,
        Vec<AgentAttachment>,
        Option<String>,
        Option<String>,
        Option<app_server_protocol::ConversationImportSourceProvenance>,
    )> = None;
    let mut pending_events = Vec::new();
    let mut leading_runtime_events = Vec::new();

    for item in timeline {
        match item {
            CodexTimelineItem::Message(message) => match message.preview.role.as_str() {
                "user" => {
                    if pending_user.as_mut().is_some_and(
                        |(text, attachments, source_type, timestamp, provenance)| {
                            if normalized_duplicate_user_message_text(text)
                                != normalized_duplicate_user_message_text(&message.preview.text)
                                || !is_duplicate_source_user_message(
                                    source_type.as_deref(),
                                    message.preview.source_type.as_deref(),
                                )
                                || has_intervening_runtime_event(&pending_events)
                            {
                                return false;
                            }
                            merge_duplicate_user_attachments(
                                attachments,
                                &message.preview.attachments,
                            );
                            if timestamp.is_none() {
                                *timestamp = message.preview.timestamp.clone();
                            }
                            if provenance.is_none() {
                                *provenance = message.preview.provenance.clone();
                            }
                            if message.preview.source_type.as_deref() == Some("event_msg") {
                                *text = message.preview.text.clone();
                            }
                            true
                        },
                    ) {
                        continue;
                    }
                    flush_history_turn(&mut turns, &mut pending_user, &mut pending_events);
                    pending_user = Some((
                        message.preview.text.clone(),
                        message.preview.attachments.clone(),
                        message.preview.source_type.clone(),
                        message.preview.timestamp.clone(),
                        message.preview.provenance.clone(),
                    ));
                    pending_events.append(&mut leading_runtime_events);
                }
                "assistant" => {
                    if pending_user.is_some() {
                        let text = message.preview.text.trim();
                        if !text.is_empty() {
                            pending_events.push(HistoryTurnEvent::AssistantMessage(
                                HistoricalAssistantMessage {
                                    text: text.to_string(),
                                    phase: message.phase.clone(),
                                    source_item_id: message.source_item_id.clone(),
                                    provenance: message.preview.provenance.clone(),
                                    timestamp: message.preview.timestamp.clone(),
                                },
                            ));
                        }
                    }
                }
                _ => {}
            },
            CodexTimelineItem::RolloutEvent { event, timestamp } => {
                if pending_user.is_some() {
                    pending_events.push(HistoryTurnEvent::Runtime {
                        event: event.clone(),
                        timestamp: timestamp.clone(),
                    });
                } else {
                    leading_runtime_events.push(HistoryTurnEvent::Runtime {
                        event: event.clone(),
                        timestamp: timestamp.clone(),
                    });
                }
            }
        }
    }
    flush_history_turn(&mut turns, &mut pending_user, &mut pending_events);
    turns
}

fn flush_history_turn(
    turns: &mut Vec<HistoryTurn>,
    pending_user: &mut Option<(
        String,
        Vec<AgentAttachment>,
        Option<String>,
        Option<String>,
        Option<app_server_protocol::ConversationImportSourceProvenance>,
    )>,
    pending_events: &mut Vec<HistoryTurnEvent>,
) {
    let Some((user_text, user_attachments, _source_type, user_timestamp, user_provenance)) =
        pending_user.take()
    else {
        pending_events.clear();
        return;
    };
    let completed_timestamp = pending_events
        .iter()
        .rev()
        .find_map(history_turn_event_timestamp)
        .or_else(|| user_timestamp.clone());
    turns.push(HistoryTurn {
        user_text,
        user_attachments,
        user_timestamp,
        completed_timestamp,
        user_provenance,
        events: std::mem::take(pending_events),
    });
}

fn history_turn_event_timestamp(event: &HistoryTurnEvent) -> Option<String> {
    match event {
        HistoryTurnEvent::AssistantMessage(message) => message.timestamp.clone(),
        HistoryTurnEvent::Runtime { timestamp, .. } => timestamp.clone(),
    }
}

fn is_duplicate_source_user_message(
    existing_type: Option<&str>,
    candidate_type: Option<&str>,
) -> bool {
    matches!(
        (existing_type, candidate_type),
        (Some("event_msg"), Some("response_item")) | (Some("response_item"), Some("event_msg"))
    )
}

fn has_intervening_runtime_event(events: &[HistoryTurnEvent]) -> bool {
    events
        .iter()
        .any(|event| matches!(event, HistoryTurnEvent::Runtime { .. }))
}

fn normalized_duplicate_user_message_text(value: &str) -> String {
    let mut without_image_tags = String::with_capacity(value.len());
    let lower = value.to_ascii_lowercase();
    let mut cursor = 0;
    while let Some(relative_start) = lower[cursor..].find("<image") {
        let start = cursor + relative_start;
        without_image_tags.push_str(&value[cursor..start]);
        let Some(relative_end) = lower[start..].find('>') else {
            without_image_tags.push_str(&value[start..]);
            cursor = value.len();
            break;
        };
        cursor = start + relative_end + 1;
    }
    if cursor < value.len() {
        without_image_tags.push_str(&value[cursor..]);
    }

    let without_closing_tags = without_image_tags
        .replace("</image>", "")
        .replace("</IMAGE>", "");
    let mut without_placeholders = String::with_capacity(without_closing_tags.len());
    let lower = without_closing_tags.to_ascii_lowercase();
    let mut cursor = 0;
    while let Some(relative_start) = lower[cursor..].find("[image") {
        let start = cursor + relative_start;
        let Some(relative_end) = lower[start..].find(']') else {
            without_placeholders.push_str(&without_closing_tags[cursor..]);
            cursor = without_closing_tags.len();
            break;
        };
        let end = start + relative_end;
        let marker = &lower[start..=end];
        let numbered_placeholder = marker
            .strip_prefix("[image #")
            .and_then(|value| value.strip_suffix(']'));
        let is_image_placeholder = marker == "[image]"
            || numbered_placeholder.is_some_and(|value| {
                !value.is_empty() && value.chars().all(|character| character.is_ascii_digit())
            });
        if !is_image_placeholder {
            without_placeholders.push_str(&without_closing_tags[cursor..=end]);
        } else {
            without_placeholders.push_str(&without_closing_tags[cursor..start]);
        }
        cursor = end + 1;
    }
    if cursor < without_closing_tags.len() {
        without_placeholders.push_str(&without_closing_tags[cursor..]);
    }

    without_placeholders
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn attachment_index(attachment: &AgentAttachment) -> Option<u64> {
    attachment
        .metadata
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .and_then(|metadata| metadata.get("index"))
        .and_then(|value| value.as_u64())
}

fn attachment_is_inline_image(attachment: &AgentAttachment) -> bool {
    attachment.kind.eq_ignore_ascii_case("image")
        && attachment
            .uri
            .as_deref()
            .is_some_and(|uri| uri.trim().to_ascii_lowercase().starts_with("data:"))
}

fn merge_duplicate_user_attachments(
    attachments: &mut Vec<AgentAttachment>,
    candidates: &[AgentAttachment],
) {
    for candidate in candidates {
        let candidate_index = attachment_index(candidate);
        let existing_index = attachments.iter().position(|existing| {
            existing.kind.eq_ignore_ascii_case(&candidate.kind)
                && ((candidate_index.is_some() && candidate_index == attachment_index(existing))
                    || existing.uri == candidate.uri)
        });
        if let Some(existing_index) = existing_index {
            if attachment_is_inline_image(candidate)
                && !attachment_is_inline_image(&attachments[existing_index])
            {
                attachments[existing_index] = candidate.clone();
            }
            continue;
        }
        attachments.push(candidate.clone());
    }
}

fn append_history_turns(
    core: &RuntimeCore,
    session_id: &str,
    turns: Vec<PreparedHistoryTurn>,
    provenance: &ImportProvenance,
    report_progress: &mut dyn FnMut(ConversationImportJobProgress) -> Result<(), RuntimeCoreError>,
) -> Result<usize, RuntimeCoreError> {
    let (session, _) = core.session_snapshot(session_id)?;
    let thread_id = session.thread_id.clone();
    let mut imported = 0;
    let total_turns = turns.len();
    let total_items = turns
        .iter()
        .map(|turn| turn.events.len().saturating_add(1))
        .sum();
    let mut completed_items: usize = 0;

    for history_turn in turns {
        let turn_id = new_id("turn");
        let started_at = history_turn
            .user_timestamp
            .clone()
            .unwrap_or_else(timestamp);
        {
            let mut state = core
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get_mut(session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
            let turn = AgentTurn {
                turn_id: turn_id.clone(),
                session_id: session_id.to_string(),
                thread_id: thread_id.clone(),
                status: AgentTurnStatus::Running,
                started_at: Some(started_at.clone()),
                completed_at: None,
            };
            stored.session.status = AgentSessionStatus::Running;
            stored.session.updated_at = started_at;
            stored
                .turn_inputs
                .insert(turn_id.clone(), history_turn.user_input);
            stored.turn_runtime_options.insert(
                turn_id.clone(),
                provenance.turn_runtime_options(history_turn.user_provenance.as_ref()),
            );
            stored.turns.push(turn);
        }

        let persisted_items = history_turn.events.len().saturating_add(1);
        let mut source_events = Vec::with_capacity(persisted_items);
        source_events.push(history_turn.user_message);
        source_events.extend(history_turn.events);
        let events = codex::build_canonical_history_events(
            source_events,
            session_id,
            &thread_id,
            &turn_id,
            history_turn.user_timestamp.as_deref(),
            history_turn.completed_timestamp.as_deref(),
        );
        core.append_runtime_events(session_id, &thread_id, Some(&turn_id), events)?;
        imported += 1;
        completed_items = completed_items.saturating_add(persisted_items);
        report_progress(import_progress(
            ConversationImportJobPhase::PersistingHistory,
            completed_items,
            total_items,
            imported,
            total_turns,
        ))?;
    }

    Ok(imported)
}

fn import_progress(
    phase: ConversationImportJobPhase,
    completed_items: usize,
    total_items: usize,
    completed_turns: usize,
    total_turns: usize,
) -> ConversationImportJobProgress {
    ConversationImportJobProgress {
        phase,
        completed_items,
        total_items,
        completed_turns,
        total_turns,
    }
}

fn prepare_history_turns(
    turns: Vec<HistoryTurn>,
) -> Result<Vec<PreparedHistoryTurn>, RuntimeCoreError> {
    turns
        .into_iter()
        .map(|turn| {
            let user_input = imported_user_input(&turn.user_text, &turn.user_attachments);
            let user_message = message_rollout_event(
                "user",
                &turn.user_text,
                &turn.user_attachments,
                Some(&user_input),
                turn.user_provenance.as_ref(),
                None,
                None,
            )?;
            let mut events = Vec::with_capacity(turn.events.len());
            for event in turn.events {
                match event {
                    HistoryTurnEvent::AssistantMessage(message) => {
                        events.push(message_rollout_event(
                            "assistant",
                            &message.text,
                            &[],
                            None,
                            message.provenance.as_ref(),
                            message.source_item_id.as_deref(),
                            message.phase.as_deref(),
                        )?);
                    }
                    HistoryTurnEvent::Runtime { event, .. } => events.push(event),
                }
            }
            Ok(PreparedHistoryTurn {
                user_input,
                user_timestamp: turn.user_timestamp,
                completed_timestamp: turn.completed_timestamp,
                user_provenance: turn.user_provenance,
                user_message,
                events,
            })
        })
        .collect()
}

fn imported_user_input(text: &str, attachments: &[AgentAttachment]) -> Vec<AgentInput> {
    let mut input = Vec::with_capacity(attachments.len() + 1);
    if !text.is_empty() {
        input.push(AgentInput::text(text));
    }
    input.extend(attachments.iter().filter_map(imported_image_input));
    input
}

fn imported_image_input(attachment: &AgentAttachment) -> Option<AgentInput> {
    if !attachment.kind.eq_ignore_ascii_case("image") {
        return None;
    }
    let uri = attachment
        .uri
        .as_deref()
        .map(str::trim)
        .filter(|uri| !uri.is_empty())?;
    let detail = attachment
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("detail"))
        .and_then(|detail| serde_json::from_value::<ImageDetail>(detail.clone()).ok());
    let local_path = attachment
        .metadata
        .as_ref()
        .and_then(|metadata| {
            metadata
                .get("localPath")
                .or_else(|| metadata.get("local_path"))
        })
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty());
    if let Some(path) = local_path {
        return Some(AgentInput::LocalImage {
            path: path.to_string(),
            detail,
        });
    }
    Some(AgentInput::Image {
        uri: uri.to_string(),
        detail,
    })
}

fn message_rollout_event(
    role: &'static str,
    text: &str,
    attachments: &[AgentAttachment],
    input: Option<&[AgentInput]>,
    provenance: Option<&ConversationImportSourceProvenance>,
    source_item_id: Option<&str>,
    phase: Option<&str>,
) -> Result<CodexRolloutEvent, RuntimeCoreError> {
    let provenance = provenance.ok_or_else(|| {
        RuntimeCoreError::Backend(format!(
            "Codex import {role} message is missing source provenance"
        ))
    })?;
    let source_event_seq = provenance.source_event_seq.ok_or_else(|| {
        RuntimeCoreError::Backend(format!(
            "Codex import {role} message is missing source event sequence"
        ))
    })?;
    let ordinal = u64::try_from(source_event_seq).map_err(|_| {
        RuntimeCoreError::Backend(format!(
            "Codex import {role} message source event sequence is out of range: {source_event_seq}"
        ))
    })?;
    let item_role = if role == "user" { "user" } else { "agent" };
    let item_id = source_item_id
        .or(provenance.source_call_id.as_deref())
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("imported-{item_role}_{source_event_seq}"));
    let source_provenance = serde_json::to_value(provenance).map_err(|error| {
        RuntimeCoreError::Backend(format!(
            "unable to serialize Codex import {role} message provenance: {error}"
        ))
    })?;
    let attachments = serde_json::to_value(attachments).map_err(|error| {
        RuntimeCoreError::Backend(format!(
            "unable to serialize Codex import {role} message attachments: {error}"
        ))
    })?;

    let mut payload = json!({
        "itemId": item_id,
        "ordinal": ordinal,
        "role": role,
        "text": text,
        "phase": phase,
        "attachments": attachments,
        "imported": true,
        "sourceClient": "codex",
        "sourceEventSeq": source_event_seq,
        "sourceProvenance": source_provenance,
    });
    if role == "user" {
        let input = input.ok_or_else(|| {
            RuntimeCoreError::Backend(
                "Codex import user message is missing typed input".to_string(),
            )
        })?;
        payload["input"] = serde_json::to_value(input).map_err(|error| {
            RuntimeCoreError::Backend(format!(
                "unable to serialize Codex import {role} typed input: {error}"
            ))
        })?;
    }

    Ok(CodexRolloutEvent::new(
        if role == "user" {
            "import.message"
        } else {
            "message.delta"
        },
        payload,
    ))
}

fn import_business_object_ref(
    source_thread_id: &str,
    title: Option<&str>,
    source_root: &std::path::Path,
    source_path: &std::path::Path,
    thread: &app_server_protocol::ImportedThreadSummary,
    fidelity: &app_server_protocol::ConversationImportFidelitySummary,
) -> BusinessObjectRef {
    let metadata = provenance::import_session_metadata(
        source_thread_id,
        source_root,
        source_path,
        thread,
        fidelity,
    );
    BusinessObjectRef {
        kind: import_status::IMPORTED_CONVERSATION_KIND.to_string(),
        id: source_thread_id.to_string(),
        title: title.map(str::to_string),
        uri: Some(codex::path_to_string(source_path)),
        metadata: Some(metadata),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn imported_user_input_preserves_text_media_order_and_detail() {
        let attachments = vec![
            AgentAttachment {
                kind: "image".to_string(),
                uri: Some("https://example.test/image.png".to_string()),
                metadata: Some(json!({"detail": "high"})),
            },
            AgentAttachment {
                kind: "image".to_string(),
                uri: Some("file:///tmp/image.png".to_string()),
                metadata: Some(json!({
                    "localPath": "/tmp/image.png",
                    "detail": "low"
                })),
            },
        ];

        assert_eq!(
            imported_user_input("describe", &attachments),
            vec![
                AgentInput::text("describe"),
                AgentInput::Image {
                    uri: "https://example.test/image.png".to_string(),
                    detail: Some(ImageDetail::High),
                },
                AgentInput::LocalImage {
                    path: "/tmp/image.png".to_string(),
                    detail: Some(ImageDetail::Low),
                },
            ]
        );
    }

    #[test]
    fn message_rollout_event_only_attaches_typed_input_to_user_messages() {
        let provenance = ConversationImportSourceProvenance {
            source_event_seq: Some(7),
            ..ConversationImportSourceProvenance::default()
        };
        let input = vec![AgentInput::text("question")];
        let user = message_rollout_event(
            "user",
            "question",
            &[],
            Some(&input),
            Some(&provenance),
            None,
            None,
        )
        .expect("user rollout event");
        let assistant = message_rollout_event(
            "assistant",
            "answer",
            &[],
            None,
            Some(&provenance),
            None,
            None,
        )
        .expect("assistant rollout event");

        assert_eq!(user.payload().expect("user payload")["input"], json!(input));
        assert!(assistant
            .payload()
            .expect("assistant payload")
            .get("input")
            .is_none());
    }
}
