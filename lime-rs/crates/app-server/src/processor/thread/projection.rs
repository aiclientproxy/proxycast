//! Typed v2 projection at the App Server thread command boundary.

use agent_protocol as canonical;
use app_server_protocol::protocol::v2;
use app_server_protocol::{error_codes, AgentEvent, JsonRpcError};
use serde_json::Value;
use std::collections::HashMap;

use super::ProjectedEvent;

pub(super) fn lower_thread_read_params(
    params: &v2::ThreadReadParams,
) -> Result<canonical::thread::ThreadReadParams, JsonRpcError> {
    Ok(canonical::thread::ThreadReadParams {
        thread_id: canonical::ThreadId::new(non_empty(&params.thread_id, "threadId")?),
        turns_view: if params.include_turns {
            canonical::ThreadTurnsView::Full
        } else {
            canonical::ThreadTurnsView::NotLoaded
        },
    })
}

pub(super) fn lower_thread_list_params(
    params: &v2::ThreadListParams,
) -> Result<canonical::ThreadListParams, JsonRpcError> {
    if params.ancestor_thread_id.is_some() {
        return Err(invalid_params(
            "thread/list ancestorThreadId is not implemented by the canonical store boundary",
        ));
    }
    if params
        .parent_thread_id
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(invalid_params(
            "thread/list parentThreadId must not be empty",
        ));
    }

    Ok(canonical::ThreadListParams {
        page: canonical::PageCursor {
            cursor: params.cursor.clone(),
            limit: params.limit,
            sort_direction: lower_sort_direction(params.sort_direction),
        },
        // The current store's flag means "include archived". The projection
        // below applies the v2 exact archived filter to the returned page.
        include_archived: params.archived.unwrap_or(false),
        turns_view: canonical::ThreadTurnsView::NotLoaded,
    })
}

pub(super) fn lower_thread_search_params(
    params: &v2::ThreadSearchParams,
) -> Result<thread_store::SearchThreadsParams, JsonRpcError> {
    let cursor = params
        .cursor
        .clone()
        .map(thread_store::StoreCursor::new)
        .transpose()
        .map_err(invalid_params)?;
    let source_kinds = match params.source_kinds.as_deref() {
        None | Some([]) => vec![
            thread_store::ThreadSearchSourceKind::Cli,
            thread_store::ThreadSearchSourceKind::VsCode,
        ],
        Some(source_kinds) => source_kinds
            .iter()
            .copied()
            .map(lower_thread_search_source_kind)
            .collect(),
    };
    Ok(thread_store::SearchThreadsParams {
        cursor,
        page_size: params.limit.unwrap_or(25).clamp(1, 100) as usize,
        sort_key: match params.sort_key.unwrap_or(v2::ThreadSortKey::CreatedAt) {
            v2::ThreadSortKey::CreatedAt => thread_store::ThreadSearchSortKey::CreatedAt,
            v2::ThreadSortKey::UpdatedAt => thread_store::ThreadSearchSortKey::UpdatedAt,
            v2::ThreadSortKey::RecencyAt => thread_store::ThreadSearchSortKey::RecencyAt,
        },
        sort_direction: lower_sort_direction(params.sort_direction),
        source_kinds,
        archived: params.archived.unwrap_or(false),
        search_term: params.search_term.trim().to_string(),
    })
}

pub(super) fn lower_thread_turns_list_params(
    params: &v2::ThreadTurnsListParams,
) -> Result<canonical::ThreadTurnsListParams, JsonRpcError> {
    Ok(canonical::ThreadTurnsListParams {
        thread_id: canonical::ThreadId::new(non_empty(&params.thread_id, "threadId")?),
        page: canonical::PageCursor {
            cursor: params.cursor.clone(),
            limit: params.limit,
            sort_direction: lower_sort_direction(params.sort_direction),
        },
        items_view: lower_items_view(params.items_view.unwrap_or_default()),
    })
}

pub(super) fn lower_thread_items_list_params(
    params: &v2::ThreadItemsListParams,
) -> Result<canonical::ThreadItemsListParams, JsonRpcError> {
    Ok(canonical::ThreadItemsListParams {
        thread_id: canonical::ThreadId::new(non_empty(&params.thread_id, "threadId")?),
        turn_id: params
            .turn_id
            .as_deref()
            .map(|value| non_empty(value, "turnId").map(canonical::TurnId::new))
            .transpose()?,
        page: canonical::PageCursor {
            cursor: params.cursor.clone(),
            limit: params.limit,
            sort_direction: lower_sort_direction(params.sort_direction),
        },
    })
}

pub(in crate::processor) fn project_thread_read_response(
    response: canonical::thread::ThreadReadResponse,
) -> Result<v2::ThreadReadResponse, JsonRpcError> {
    Ok(v2::ThreadReadResponse {
        thread: project_thread(response.thread)?,
    })
}

pub(super) fn project_thread_list_response(
    response: canonical::ThreadListResponse,
    params: &v2::ThreadListParams,
) -> Result<v2::ThreadListResponse, JsonRpcError> {
    let data = response
        .data
        .into_iter()
        .filter(|thread| thread_matches_list_filters(thread, params))
        .map(project_thread)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(v2::ThreadListResponse {
        data,
        next_cursor: response.next_cursor,
        backwards_cursor: response.backwards_cursor,
    })
}

pub(super) fn project_thread_search_response(
    response: thread_store::ThreadSearchPage,
) -> Result<v2::ThreadSearchResponse, JsonRpcError> {
    Ok(v2::ThreadSearchResponse {
        data: response
            .data
            .into_iter()
            .map(|result| {
                Ok(v2::ThreadSearchResult {
                    thread: project_thread(result.thread)?,
                    snippet: result.snippet,
                })
            })
            .collect::<Result<Vec<_>, JsonRpcError>>()?,
        next_cursor: response
            .next_cursor
            .map(thread_store::StoreCursor::into_string),
        backwards_cursor: response
            .backwards_cursor
            .map(thread_store::StoreCursor::into_string),
    })
}

pub(super) fn project_thread_turns_list_response(
    response: canonical::ThreadTurnsListResponse,
) -> Result<v2::ThreadTurnsListResponse, JsonRpcError> {
    Ok(v2::ThreadTurnsListResponse {
        data: response
            .data
            .into_iter()
            .map(project_turn)
            .collect::<Result<Vec<_>, _>>()?,
        next_cursor: response.next_cursor,
        backwards_cursor: response.backwards_cursor,
    })
}

pub(super) fn project_thread_items_list_response(
    response: canonical::ThreadItemsListResponse,
) -> Result<v2::ThreadItemsListResponse, JsonRpcError> {
    Ok(v2::ThreadItemsListResponse {
        data: response
            .data
            .into_iter()
            .filter(|item| !is_out_of_band_control_item(item))
            .map(|item| {
                let turn_id = item.turn_id.as_str().to_string();
                Ok(v2::ThreadItemEntry {
                    turn_id,
                    item: project_item(item)?,
                })
            })
            .collect::<Result<Vec<_>, JsonRpcError>>()?,
        next_cursor: response.next_cursor,
        backwards_cursor: response.backwards_cursor,
    })
}

pub(super) fn project_event(event: &AgentEvent) -> Option<ProjectedEvent> {
    match event.event_type.as_str() {
        "thread.created" | "thread.started" | "thread.updated" => {
            canonical_entity(&event.payload, "thread")
                .and_then(|thread| project_thread(thread).ok())
                .map(ProjectedEvent::Thread)
        }
        "turn.accepted" | "turn.started" | "turn.completed" | "turn.failed" | "turn.canceled" => {
            canonical_entity(&event.payload, "turn")
                .and_then(|turn| project_turn(turn).ok())
                .map(ProjectedEvent::Turn)
        }
        _ => canonical_entity(&event.payload, "item")
            .and_then(|item| project_item(item).ok())
            .map(ProjectedEvent::Item),
    }
}

fn canonical_entity<T>(payload: &Value, key: &str) -> Option<T>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_value(payload.get(key)?.clone()).ok()
}

fn project_thread(thread: canonical::Thread) -> Result<v2::Thread, JsonRpcError> {
    let can_accept_direct_input = thread.parent_thread_id.is_none();
    let metadata = thread.metadata.clone();
    let cwd = metadata_string(&metadata, &["workingDir", "working_dir", "cwd"]).unwrap_or_default();
    let source = metadata_string(&metadata, &["source", "sourceKind", "source_kind"])
        .unwrap_or_else(|| "appServer".to_string());
    let git_info = project_git_info(&metadata);
    let history_mode = match metadata_string(&metadata, &["historyMode", "history_mode"]).as_deref()
    {
        Some("paginated") => v2::ThreadHistoryMode::Paginated,
        _ => v2::ThreadHistoryMode::Legacy,
    };
    let extra = (!metadata.is_null()).then_some(metadata);

    Ok(v2::Thread {
        id: thread.thread_id.as_str().to_string(),
        extra,
        session_id: thread.session_id.as_str().to_string(),
        forked_from_id: thread
            .forked_from_id
            .map(|value| value.as_str().to_string()),
        parent_thread_id: thread
            .parent_thread_id
            .map(|value| value.as_str().to_string()),
        preview: thread.preview,
        ephemeral: metadata_bool(&thread.metadata, &["ephemeral"]).unwrap_or(false),
        is_pinned: metadata_bool(&thread.metadata, &["isPinned"]).unwrap_or(false),
        history_mode,
        model_provider: thread.model_provider,
        created_at: millis_to_seconds(thread.created_at_ms),
        updated_at: millis_to_seconds(thread.updated_at_ms),
        recency_at: thread.recency_at_ms.map(millis_to_seconds),
        status: Some(project_thread_status(thread.status)),
        path: metadata_string(&thread.metadata, &["path", "rolloutPath", "rollout_path"]),
        cwd,
        cli_version: metadata_string(&thread.metadata, &["cliVersion", "cli_version"])
            .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string()),
        source,
        can_accept_direct_input: Some(can_accept_direct_input),
        thread_source: metadata_string(&thread.metadata, &["threadSource", "thread_source"]),
        agent_nickname: thread.agent_nickname,
        agent_role: thread.agent_role,
        git_info,
        name: thread.name,
        turns: thread
            .turns
            .into_iter()
            .map(project_turn)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn project_turn(turn: canonical::Turn) -> Result<v2::Turn, JsonRpcError> {
    Ok(v2::Turn {
        id: turn.turn_id.as_str().to_string(),
        items: turn
            .items
            .into_iter()
            .filter(|item| !is_out_of_band_control_item(item))
            .map(project_item)
            .collect::<Result<Vec<_>, _>>()?,
        items_view: project_items_view(turn.items_view),
        status: project_turn_status(turn.status),
        error: turn.error.map(|error| v2::TurnError {
            message: error.message,
            error_info: error.code.map(Value::String),
            additional_details: error.details,
        }),
        started_at: turn.started_at_ms.map(millis_to_seconds),
        completed_at: turn.completed_at_ms.map(millis_to_seconds),
        duration_ms: turn.duration_ms.map(saturating_i64),
    })
}

fn is_out_of_band_control_item(item: &canonical::ThreadItem) -> bool {
    matches!(&item.payload, canonical::ThreadItemPayload::Approval { .. })
}

fn project_item(item: canonical::ThreadItem) -> Result<v2::ThreadItem, JsonRpcError> {
    let id = item.item_id.as_str().to_string();
    let status = item.status;
    let metadata = item.metadata;
    match item.payload {
        canonical::ThreadItemPayload::UserMessage { content, client_id } => {
            Ok(v2::ThreadItem::UserMessage {
                id,
                client_id,
                content: content.into_iter().map(project_user_input).collect(),
            })
        }
        canonical::ThreadItemPayload::AgentMessage { text, phase, .. } => {
            Ok(v2::ThreadItem::AgentMessage {
                id,
                text,
                phase,
                memory_citation: None,
            })
        }
        canonical::ThreadItemPayload::Plan { text, .. } => {
            Ok(v2::ThreadItem::Plan { id, text })
        }
        canonical::ThreadItemPayload::Reasoning { summary, content } => {
            Ok(v2::ThreadItem::Reasoning {
                id,
                summary,
                content,
            })
        }
        canonical::ThreadItemPayload::Tool {
            name,
            arguments,
            output,
            ..
        } => {
            let duration_ms = output.as_ref().and_then(|value| value.duration_ms);
            let success = output
                .as_ref()
                .map(|value| value.error.is_none())
                .or_else(|| terminal_success(status));
            Ok(v2::ThreadItem::DynamicToolCall {
                id,
                namespace: None,
                tool: name,
                arguments: serde_json::to_value(arguments)
                    .map_err(|error| projection_error(format!("tool arguments: {error}")))?,
                status: project_dynamic_tool_status(status),
                content_items: output_content_items(output.as_ref()),
                success,
                duration_ms: duration_ms.map(saturating_i64),
            })
        }
        canonical::ThreadItemPayload::McpToolCall {
            server_name,
            tool_name,
            arguments,
            output,
            ..
        } => {
            let result = output.as_ref().map(project_mcp_tool_result);
            let error = output
                .as_ref()
                .and_then(|value| value.error.clone())
                .map(|message| serde_json::json!({ "message": message }));
            Ok(v2::ThreadItem::McpToolCall {
                id,
                server: server_name,
                tool: tool_name,
                status: project_mcp_status(status),
                arguments: serde_json::to_value(arguments)
                    .map_err(|error| projection_error(format!("MCP arguments: {error}")))?,
                app_context: None,
                mcp_app_resource_uri: None,
                plugin_id: None,
                result,
                error,
                duration_ms: output
                    .as_ref()
                    .and_then(|value| value.duration_ms)
                    .map(saturating_i64),
            })
        }
        canonical::ThreadItemPayload::CollabAgentToolCall {
            operation,
            target_thread_id,
            message,
            ..
        } => Ok(v2::ThreadItem::CollabAgentToolCall {
            id,
            tool: project_collab_tool(operation),
            status: project_collab_status(status),
            sender_thread_id: item.thread_id.as_str().to_string(),
            receiver_thread_ids: target_thread_id
                .map(|value| vec![value.as_str().to_string()])
                .unwrap_or_default(),
            prompt: message,
            model: metadata_string(&metadata, &["model", "modelName", "model_name"]),
            reasoning_effort: metadata_string(
                &metadata,
                &["reasoningEffort", "reasoning_effort"],
            ),
            agents_states: HashMap::new(),
        }),
        canonical::ThreadItemPayload::Approval { .. } => Err(projection_error(format!(
            "canonical approval item {id} has no v2 ThreadItem representation"
        ))),
        canonical::ThreadItemPayload::Command {
            command,
            cwd,
            output,
            exit_code,
        } => Ok(v2::ThreadItem::CommandExecution {
            id,
            command,
            cwd: cwd.unwrap_or_default(),
            process_id: metadata_string(&metadata, &["processId", "process_id"]),
            source: project_command_source(&metadata),
            status: project_command_status(status),
            command_actions: Vec::new(),
            aggregated_output: output,
            exit_code,
            duration_ms: metadata_u64(&metadata, &["durationMs", "duration_ms"])
                .map(saturating_i64),
        }),
        canonical::ThreadItemPayload::File { changes, status } => Ok(v2::ThreadItem::FileChange {
            id,
            changes: changes
                .into_iter()
                .map(|change| v2::FileUpdateChange {
                    path: change.path,
                    kind: project_patch_change_kind(change.kind),
                    diff: change.diff,
                })
                .collect(),
            status: project_patch_status(status),
        }),
        canonical::ThreadItemPayload::Media {
            uri,
            mime_type,
            ..
        } if mime_type.starts_with("image/") => Ok(v2::ThreadItem::ImageView { id, path: uri }),
        canonical::ThreadItemPayload::Media { mime_type, .. } => Err(projection_error(format!(
            "canonical media item {id} with MIME type {mime_type} has no v2 ThreadItem representation"
        ))),
        canonical::ThreadItemPayload::SubAgent {
            child_thread_id,
            activity,
            ..
        } => Ok(v2::ThreadItem::SubAgentActivity {
            id,
            kind: project_subagent_activity(activity),
            agent_thread_id: child_thread_id.as_str().to_string(),
            agent_path: metadata_string(&metadata, &["agentPath", "agent_path"])
                .unwrap_or_else(|| child_thread_id.as_str().to_string()),
        }),
        canonical::ThreadItemPayload::ContextCompaction { .. } => {
            Ok(v2::ThreadItem::ContextCompaction { id })
        }
        canonical::ThreadItemPayload::Extension { name, .. } => Err(projection_error(format!(
            "canonical extension item {id} ({name}) has no v2 ThreadItem representation"
        ))),
    }
}

fn project_user_input(input: canonical::AgentInput) -> v2::UserInput {
    match input {
        canonical::AgentInput::Text {
            text,
            text_elements,
        } => v2::UserInput::Text {
            text,
            text_elements,
        },
        canonical::AgentInput::Image { uri, detail } => v2::UserInput::Image { detail, url: uri },
        canonical::AgentInput::LocalImage { path, detail } => {
            v2::UserInput::LocalImage { detail, path }
        }
        canonical::AgentInput::Skill { name, path } => v2::UserInput::Skill { name, path },
        canonical::AgentInput::Mention { name, path } => v2::UserInput::Mention { name, path },
    }
}

fn project_command_source(metadata: &Value) -> v2::CommandExecutionSource {
    match metadata_string(
        metadata,
        &["commandExecutionSource", "command_execution_source"],
    )
    .as_deref()
    {
        Some("userShell") | Some("user_shell") => v2::CommandExecutionSource::UserShell,
        _ => v2::CommandExecutionSource::Agent,
    }
}

fn thread_matches_list_filters(thread: &canonical::Thread, params: &v2::ThreadListParams) -> bool {
    if thread.archived != params.archived.unwrap_or(false) {
        return false;
    }
    if params.is_pinned.is_some_and(|is_pinned| {
        metadata_bool(&thread.metadata, &["isPinned"]).unwrap_or(false) != is_pinned
    }) {
        return false;
    }
    if params.model_providers.as_ref().is_some_and(|providers| {
        !providers
            .iter()
            .any(|provider| provider == &thread.model_provider)
    }) {
        return false;
    }
    if params.parent_thread_id.as_ref().is_some_and(|parent| {
        thread
            .parent_thread_id
            .as_ref()
            .map(canonical::ThreadId::as_str)
            != Some(parent.trim())
    }) {
        return false;
    }
    if params.cwd.as_ref().is_some_and(|filter| {
        let cwd = metadata_string(&thread.metadata, &["workingDir", "working_dir", "cwd"])
            .unwrap_or_default();
        !cwd_matches(filter, &cwd)
    }) {
        return false;
    }
    if params.search_term.as_ref().is_some_and(|search| {
        let search = search.trim().to_lowercase();
        !search.is_empty()
            && !thread.preview.to_lowercase().contains(&search)
            && !thread
                .name
                .as_deref()
                .unwrap_or_default()
                .to_lowercase()
                .contains(&search)
    }) {
        return false;
    }
    if params.source_kinds.as_ref().is_some_and(|kinds| {
        let source = metadata_string(
            &thread.metadata,
            &["sourceKind", "source_kind", "source", "threadSource"],
        )
        .unwrap_or_else(|| "appServer".to_string());
        !kinds.iter().any(|kind| source_kind_matches(*kind, &source))
    }) {
        return false;
    }
    true
}

fn cwd_matches(filter: &v2::ThreadListCwdFilter, cwd: &str) -> bool {
    let cwd = normalize_path(cwd);
    match filter {
        v2::ThreadListCwdFilter::One(value) => normalize_path(value) == cwd,
        v2::ThreadListCwdFilter::Many(values) => {
            values.iter().any(|value| normalize_path(value) == cwd)
        }
    }
}

fn normalize_path(value: &str) -> &str {
    value.trim().trim_end_matches(&['/', '\\'][..])
}

fn source_kind_matches(kind: v2::ThreadSourceKind, source: &str) -> bool {
    let normalized = source
        .chars()
        .filter(|value| value.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    match kind {
        v2::ThreadSourceKind::Cli => normalized == "cli",
        v2::ThreadSourceKind::VsCode => normalized == "vscode",
        v2::ThreadSourceKind::Exec => normalized == "exec",
        v2::ThreadSourceKind::AppServer => normalized == "appserver",
        v2::ThreadSourceKind::SubAgent => normalized == "subagent",
        v2::ThreadSourceKind::SubAgentReview => normalized == "subagentreview",
        v2::ThreadSourceKind::SubAgentCompact => normalized == "subagentcompact",
        v2::ThreadSourceKind::SubAgentThreadSpawn => normalized == "subagentthreadspawn",
        v2::ThreadSourceKind::SubAgentOther => normalized == "subagentother",
        v2::ThreadSourceKind::Unknown => !matches!(
            normalized.as_str(),
            "cli"
                | "vscode"
                | "exec"
                | "appserver"
                | "subagent"
                | "subagentreview"
                | "subagentcompact"
                | "subagentthreadspawn"
                | "subagentother"
        ),
    }
}

fn project_git_info(metadata: &Value) -> Option<v2::GitInfo> {
    let git = metadata.get("gitInfo").or_else(|| metadata.get("git_info"));
    let sha = git.and_then(|value| metadata_string(value, &["sha", "commitHash", "commit_hash"]));
    let branch = git.and_then(|value| metadata_string(value, &["branch"]));
    let origin_url = git.and_then(|value| {
        metadata_string(
            value,
            &["originUrl", "origin_url", "repositoryUrl", "repository_url"],
        )
    });
    (sha.is_some() || branch.is_some() || origin_url.is_some()).then_some(v2::GitInfo {
        sha,
        branch,
        origin_url,
    })
}

fn project_thread_status(status: canonical::ThreadStatus) -> v2::ThreadStatus {
    match status {
        canonical::ThreadStatus::NotLoaded => v2::ThreadStatus::NotLoaded,
        canonical::ThreadStatus::Idle => v2::ThreadStatus::Idle,
        canonical::ThreadStatus::SystemError => v2::ThreadStatus::SystemError,
        canonical::ThreadStatus::Active { active_flags } => v2::ThreadStatus::Active {
            active_flags: active_flags
                .into_iter()
                .map(|flag| match flag {
                    canonical::ThreadActiveFlag::WaitingOnApproval => {
                        v2::ThreadActiveFlag::WaitingOnApproval
                    }
                    canonical::ThreadActiveFlag::WaitingOnUserInput => {
                        v2::ThreadActiveFlag::WaitingOnUserInput
                    }
                })
                .collect(),
        },
    }
}

fn project_turn_status(status: canonical::TurnStatus) -> v2::TurnStatus {
    match status {
        canonical::TurnStatus::InProgress => v2::TurnStatus::InProgress,
        canonical::TurnStatus::Completed => v2::TurnStatus::Completed,
        canonical::TurnStatus::Interrupted => v2::TurnStatus::Interrupted,
        canonical::TurnStatus::Failed => v2::TurnStatus::Failed,
    }
}

fn lower_items_view(view: v2::TurnItemsView) -> canonical::TurnItemsView {
    match view {
        v2::TurnItemsView::NotLoaded => canonical::TurnItemsView::NotLoaded,
        v2::TurnItemsView::Summary => canonical::TurnItemsView::Summary,
        v2::TurnItemsView::Full => canonical::TurnItemsView::Full,
    }
}

fn project_items_view(view: canonical::TurnItemsView) -> v2::TurnItemsView {
    match view {
        canonical::TurnItemsView::NotLoaded => v2::TurnItemsView::NotLoaded,
        canonical::TurnItemsView::Summary => v2::TurnItemsView::Summary,
        canonical::TurnItemsView::Full => v2::TurnItemsView::Full,
    }
}

fn lower_sort_direction(direction: Option<v2::SortDirection>) -> canonical::SortDirection {
    match direction.unwrap_or(v2::SortDirection::Desc) {
        v2::SortDirection::Asc => canonical::SortDirection::Asc,
        v2::SortDirection::Desc => canonical::SortDirection::Desc,
    }
}

fn lower_thread_search_source_kind(
    kind: v2::ThreadSourceKind,
) -> thread_store::ThreadSearchSourceKind {
    match kind {
        v2::ThreadSourceKind::Cli => thread_store::ThreadSearchSourceKind::Cli,
        v2::ThreadSourceKind::VsCode => thread_store::ThreadSearchSourceKind::VsCode,
        v2::ThreadSourceKind::Exec => thread_store::ThreadSearchSourceKind::Exec,
        v2::ThreadSourceKind::AppServer => thread_store::ThreadSearchSourceKind::AppServer,
        v2::ThreadSourceKind::SubAgent => thread_store::ThreadSearchSourceKind::SubAgent,
        v2::ThreadSourceKind::SubAgentReview => {
            thread_store::ThreadSearchSourceKind::SubAgentReview
        }
        v2::ThreadSourceKind::SubAgentCompact => {
            thread_store::ThreadSearchSourceKind::SubAgentCompact
        }
        v2::ThreadSourceKind::SubAgentThreadSpawn => {
            thread_store::ThreadSearchSourceKind::SubAgentThreadSpawn
        }
        v2::ThreadSourceKind::SubAgentOther => thread_store::ThreadSearchSourceKind::SubAgentOther,
        v2::ThreadSourceKind::Unknown => thread_store::ThreadSearchSourceKind::Unknown,
    }
}

fn project_command_status(status: canonical::ItemStatus) -> v2::CommandExecutionStatus {
    match status {
        canonical::ItemStatus::Pending | canonical::ItemStatus::InProgress => {
            v2::CommandExecutionStatus::InProgress
        }
        canonical::ItemStatus::Completed => v2::CommandExecutionStatus::Completed,
        canonical::ItemStatus::Failed => v2::CommandExecutionStatus::Failed,
        canonical::ItemStatus::Interrupted | canonical::ItemStatus::Cancelled => {
            v2::CommandExecutionStatus::Declined
        }
    }
}

fn project_dynamic_tool_status(status: canonical::ItemStatus) -> v2::DynamicToolCallStatus {
    match status {
        canonical::ItemStatus::Pending | canonical::ItemStatus::InProgress => {
            v2::DynamicToolCallStatus::InProgress
        }
        canonical::ItemStatus::Completed => v2::DynamicToolCallStatus::Completed,
        canonical::ItemStatus::Failed
        | canonical::ItemStatus::Interrupted
        | canonical::ItemStatus::Cancelled => v2::DynamicToolCallStatus::Failed,
    }
}

fn project_mcp_status(status: canonical::ItemStatus) -> v2::McpToolCallStatus {
    match status {
        canonical::ItemStatus::Pending | canonical::ItemStatus::InProgress => {
            v2::McpToolCallStatus::InProgress
        }
        canonical::ItemStatus::Completed => v2::McpToolCallStatus::Completed,
        canonical::ItemStatus::Failed
        | canonical::ItemStatus::Interrupted
        | canonical::ItemStatus::Cancelled => v2::McpToolCallStatus::Failed,
    }
}

fn project_mcp_tool_result(output: &canonical::ToolOutput) -> Value {
    let content = output
        .text
        .as_ref()
        .filter(|text| !text.is_empty())
        .map(|text| vec![serde_json::json!({ "type": "text", "text": text })])
        .unwrap_or_default();
    let mut result = serde_json::Map::from_iter([("content".to_string(), Value::Array(content))]);
    if let Some(structured_content) = &output.structured_content {
        result.insert("structuredContent".to_string(), structured_content.clone());
    }

    let mut metadata = serde_json::Map::new();
    if output.truncated {
        metadata.insert("truncated".to_string(), Value::Bool(true));
    }
    if let Some(output_ref) = &output.output_ref {
        metadata.insert("outputRef".to_string(), Value::String(output_ref.clone()));
    }
    if !metadata.is_empty() {
        result.insert("_meta".to_string(), Value::Object(metadata));
    }
    Value::Object(result)
}

fn project_collab_status(status: canonical::ItemStatus) -> v2::CollabAgentToolCallStatus {
    match status {
        canonical::ItemStatus::Pending | canonical::ItemStatus::InProgress => {
            v2::CollabAgentToolCallStatus::InProgress
        }
        canonical::ItemStatus::Completed => v2::CollabAgentToolCallStatus::Completed,
        canonical::ItemStatus::Failed
        | canonical::ItemStatus::Interrupted
        | canonical::ItemStatus::Cancelled => v2::CollabAgentToolCallStatus::Failed,
    }
}

fn project_patch_status(status: canonical::FileChangeStatus) -> v2::PatchApplyStatus {
    match status {
        canonical::FileChangeStatus::Proposed => v2::PatchApplyStatus::InProgress,
        canonical::FileChangeStatus::Applied => v2::PatchApplyStatus::Completed,
        canonical::FileChangeStatus::Rejected => v2::PatchApplyStatus::Declined,
        canonical::FileChangeStatus::Failed => v2::PatchApplyStatus::Failed,
    }
}

fn project_patch_change_kind(kind: canonical::FileChangeKind) -> v2::PatchChangeKind {
    match kind {
        canonical::FileChangeKind::Add => v2::PatchChangeKind::Add,
        canonical::FileChangeKind::Delete => v2::PatchChangeKind::Delete,
        canonical::FileChangeKind::Update { move_path } => {
            v2::PatchChangeKind::Update { move_path }
        }
    }
}

fn project_collab_tool(operation: canonical::CollabAgentOperation) -> v2::CollabAgentTool {
    match operation {
        canonical::CollabAgentOperation::Spawn => v2::CollabAgentTool::SpawnAgent,
        canonical::CollabAgentOperation::SendMessage
        | canonical::CollabAgentOperation::FollowUp => v2::CollabAgentTool::SendInput,
        canonical::CollabAgentOperation::Wait => v2::CollabAgentTool::Wait,
        canonical::CollabAgentOperation::Resume => v2::CollabAgentTool::ResumeAgent,
        canonical::CollabAgentOperation::Interrupt | canonical::CollabAgentOperation::Close => {
            v2::CollabAgentTool::CloseAgent
        }
    }
}

fn project_subagent_activity(
    activity: canonical::SubAgentActivityKind,
) -> v2::SubAgentActivityKind {
    match activity {
        canonical::SubAgentActivityKind::Started => v2::SubAgentActivityKind::Started,
        canonical::SubAgentActivityKind::Interacted => v2::SubAgentActivityKind::Interacted,
        canonical::SubAgentActivityKind::Interrupted => v2::SubAgentActivityKind::Interrupted,
    }
}

fn output_content_items(
    output: Option<&canonical::ToolOutput>,
) -> Option<Vec<v2::DynamicToolCallOutputContentItem>> {
    let output = output?;
    let mut items = Vec::new();
    if let Some(text) = output.text.as_ref().filter(|value| !value.is_empty()) {
        items.push(v2::DynamicToolCallOutputContentItem::InputText { text: text.clone() });
    }
    if let Some(value) = &output.structured_content {
        items.push(v2::DynamicToolCallOutputContentItem::InputText {
            text: value.to_string(),
        });
    }
    if let Some(error) = output.error.as_ref().filter(|value| !value.is_empty()) {
        items.push(v2::DynamicToolCallOutputContentItem::InputText {
            text: error.clone(),
        });
    }
    (!items.is_empty()).then_some(items)
}

fn terminal_success(status: canonical::ItemStatus) -> Option<bool> {
    match status {
        canonical::ItemStatus::Completed => Some(true),
        canonical::ItemStatus::Failed
        | canonical::ItemStatus::Interrupted
        | canonical::ItemStatus::Cancelled => Some(false),
        canonical::ItemStatus::Pending | canonical::ItemStatus::InProgress => None,
    }
}

fn metadata_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(key).and_then(Value::as_str))
        .map(ToString::to_string)
        .filter(|value| !value.trim().is_empty())
}

fn metadata_bool(value: &Value, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .find_map(|key| value.get(key).and_then(Value::as_bool))
}

fn metadata_u64(value: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| value.get(key).and_then(Value::as_u64))
}

fn millis_to_seconds(value: i64) -> i64 {
    value.div_euclid(1_000)
}

fn saturating_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn non_empty(value: &str, field: &str) -> Result<String, JsonRpcError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(invalid_params(format!("thread request requires {field}")));
    }
    Ok(value.to_string())
}

fn invalid_params(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(error_codes::INVALID_PARAMS, message)
}

fn projection_error(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(error_codes::RUNTIME_ERROR, message)
}

#[cfg(test)]
mod tests;
