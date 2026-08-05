use super::fields::{
    compact_value, explicit_item_id, map_bool, map_i64, map_string, map_u64, message_text,
    payload_source, string_list,
};
use super::lifecycle::{approval_decision, is_action_resolution_event};
use agent_protocol::{
    AgentInput, ApprovalAction, ApprovalDecision, ApprovalScope, CollabAgentOperation, FileChange,
    FileChangeKind, FileChangeStatus, MessageContentPart, PlanStep, PlanStepStatus,
    SubAgentActivityKind, ThreadId, ThreadItemPayload, ToolOutput,
};
use serde_json::{Map, Value};
use std::collections::BTreeSet;

const UNKNOWN_ITEM_FIELD_NAME_LIMIT: usize = 12;
const UNKNOWN_ITEM_REDACTED_FIELD_NAME: &str = "[redacted]";

#[derive(Clone, Copy)]
pub(super) enum ItemFamily {
    UserMessage,
    AgentMessage,
    Plan,
    Reasoning,
    Tool,
    McpToolCall,
    CollabAgentToolCall,
    Approval,
    Command,
    File,
    Media,
    SubAgent,
    ContextCompaction,
    Unknown,
}

impl ItemFamily {
    pub(super) fn explicit_item_id(self, payload: &Map<String, Value>) -> Option<String> {
        if matches!(self, Self::Approval) {
            return map_string(
                payload,
                &[
                    "itemId",
                    "item_id",
                    "requestId",
                    "request_id",
                    "actionId",
                    "action_id",
                    "id",
                ],
            );
        }
        explicit_item_id(payload)
    }

    fn stable_name(self) -> &'static str {
        match self {
            Self::UserMessage => "user",
            Self::AgentMessage => "agent",
            Self::Plan => "plan",
            Self::Reasoning => "reasoning",
            Self::Tool => "tool",
            Self::McpToolCall => "mcp-tool",
            Self::CollabAgentToolCall => "collab-tool",
            Self::Approval => "approval",
            Self::Command => "command",
            Self::File => "file",
            Self::Media => "media",
            Self::SubAgent => "subagent",
            Self::ContextCompaction => "compaction",
            Self::Unknown => "unknown",
        }
    }

    pub(super) fn item_id(
        self,
        payload: &Map<String, Value>,
        turn_id: &str,
        event_id: &str,
    ) -> Option<String> {
        if matches!(self, Self::Unknown) {
            return explicit_item_id(payload);
        }
        if matches!(self, Self::Plan) {
            if let Some(item_id) = explicit_item_id(payload) {
                return Some(item_id);
            }
            let revision_id = map_string(payload, &["revisionId", "revision_id"])?;
            return Some(format!("{}_{turn_id}_{revision_id}", self.stable_name()));
        }
        Some(
            self.explicit_item_id(payload)
                .unwrap_or_else(|| self.fallback_id(turn_id, event_id)),
        )
    }

    fn fallback_id(self, turn_id: &str, event_id: &str) -> String {
        match self {
            Self::UserMessage | Self::AgentMessage | Self::Reasoning | Self::ContextCompaction => {
                format!("{}-{turn_id}", self.stable_name())
            }
            _ => format!("{}-{event_id}", self.stable_name()),
        }
    }
}

pub(super) fn item_family(event_type: &str, payload: &Value) -> Option<ItemFamily> {
    let normalized = event_type.to_ascii_lowercase();
    if normalized.starts_with("item.") {
        let source = payload_source(payload);
        let upstream_type = map_string(source, &["kind", "type", "itemType", "item_type"])?;
        let kind = upstream_type.to_ascii_lowercase();
        return match kind.as_str() {
            "user_message" | "usermessage" | "user" => Some(ItemFamily::UserMessage),
            "agent_message" | "agentmessage" | "assistant" | "message" => {
                Some(ItemFamily::AgentMessage)
            }
            "plan" => Some(ItemFamily::Plan),
            "reasoning" | "reasoning_message" => Some(ItemFamily::Reasoning),
            "tool" | "tool_call" | "tool_result" | "web_search" => Some(tool_family(source)),
            "mcp_tool" | "mcp_tool_call" => Some(ItemFamily::McpToolCall),
            "collab_agent_tool_call" | "subagent_tool_call" => {
                Some(ItemFamily::CollabAgentToolCall)
            }
            "approval" | "action_required" | "request_user_input" => Some(ItemFamily::Approval),
            "command" | "command_execution" => Some(ItemFamily::Command),
            "file" | "file_change" | "patch" => Some(ItemFamily::File),
            "media" | "artifact" | "image" | "video" | "audio" => Some(ItemFamily::Media),
            "subagent" | "sub_agent" | "subagent_activity" => Some(ItemFamily::SubAgent),
            "context_compaction" | "compaction" => Some(ItemFamily::ContextCompaction),
            _ if matches!(
                normalized.as_str(),
                "item.started" | "item.updated" | "item.completed"
            ) && is_canonical_unknown_item_type(&upstream_type) =>
            {
                Some(ItemFamily::Unknown)
            }
            _ => None,
        };
    }
    if normalized.starts_with("message.") {
        let role = super::fields::value_string(payload, &["role", "author"]).unwrap_or_default();
        let user = normalized == "message.created"
            || role.eq_ignore_ascii_case("user")
            || payload.get("input").is_some();
        return Some(if user {
            ItemFamily::UserMessage
        } else {
            ItemFamily::AgentMessage
        });
    }
    if normalized.starts_with("plan.") {
        return Some(ItemFamily::Plan);
    }
    if normalized.starts_with("reasoning.") {
        return Some(ItemFamily::Reasoning);
    }
    if matches!(normalized.as_str(), "tool.progress" | "tool.output.delta") {
        return Some(tool_family(payload_source(payload)));
    }
    if normalized.starts_with("mcp.") {
        return Some(ItemFamily::McpToolCall);
    }
    if normalized == "approval.session_cache.hit" {
        return None;
    }
    if normalized.starts_with("action.") || normalized.starts_with("approval.") {
        return Some(ItemFamily::Approval);
    }
    if normalized.starts_with("command.") {
        return Some(ItemFamily::Command);
    }
    if normalized.starts_with("patch.") || normalized.starts_with("file.") {
        return Some(ItemFamily::File);
    }
    if normalized.starts_with("artifact.") || normalized.starts_with("media.") {
        return Some(ItemFamily::Media);
    }
    if normalized.starts_with("subagent.") || normalized.starts_with("sub_agent.") {
        return Some(ItemFamily::SubAgent);
    }
    if normalized.starts_with("context.compaction") {
        return Some(ItemFamily::ContextCompaction);
    }
    None
}

fn tool_family(payload: &Map<String, Value>) -> ItemFamily {
    if map_string(
        payload,
        &["serverName", "server_name", "mcpServer", "mcp_server"],
    )
    .is_some()
    {
        return ItemFamily::McpToolCall;
    }
    ItemFamily::Tool
}

pub(super) fn typed_payload(
    family: ItemFamily,
    event_type: &str,
    payload: &Map<String, Value>,
    fallback_call_id: &str,
    timestamp_ms: i64,
) -> Option<ThreadItemPayload> {
    Some(match family {
        ItemFamily::UserMessage => ThreadItemPayload::UserMessage {
            content: user_message_content(payload)?,
            client_id: map_string(payload, &["clientId", "client_id"]),
        },
        ItemFamily::AgentMessage => ThreadItemPayload::AgentMessage {
            text: message_text(payload),
            phase: map_string(payload, &["phase", "messagePhase", "message_phase"])
                .as_deref()
                .and_then(agent_protocol::response_item::MessagePhase::from_wire),
            content_parts: message_content_parts(payload)?,
        },
        ItemFamily::Plan => ThreadItemPayload::Plan {
            text: message_text(payload),
            revision_id: map_string(payload, &["revisionId", "revision_id"])
                .unwrap_or_else(|| fallback_call_id.to_string()),
            source: map_string(payload, &["source"]),
            plan: plan_steps(payload),
            explanation: map_string(payload, &["explanation"]),
            tool_call_id: map_string(payload, &["toolCallId", "tool_call_id"]),
            source_item_id: map_string(payload, &["sourceItemId", "source_item_id"]),
        },
        ItemFamily::Reasoning => ThreadItemPayload::Reasoning {
            summary: string_list(payload, &["summary", "summaries"]),
            content: string_list(payload, &["content", "text", "delta"]),
        },
        ItemFamily::Tool => ThreadItemPayload::Tool {
            call_id: call_id(payload, fallback_call_id),
            name: map_string(payload, &["toolName", "tool_name", "name"])
                .unwrap_or_else(|| "tool".to_string()),
            arguments: tool_arguments(payload),
            output: tool_output(payload),
        },
        ItemFamily::McpToolCall => ThreadItemPayload::McpToolCall {
            call_id: call_id(payload, fallback_call_id),
            server_name: map_string(
                payload,
                &["serverName", "server_name", "mcpServer", "mcp_server"],
            )
            .unwrap_or_else(|| "unknown".to_string()),
            tool_name: map_string(payload, &["toolName", "tool_name", "name"])
                .unwrap_or_else(|| "tool".to_string()),
            mcp_app_resource_uri: map_string(
                payload,
                &["mcpAppResourceUri", "mcp_app_resource_uri"],
            ),
            plugin_id: map_string(payload, &["pluginId", "plugin_id"]),
            arguments: tool_arguments(payload),
            output: tool_output(payload),
        },
        ItemFamily::CollabAgentToolCall => ThreadItemPayload::CollabAgentToolCall {
            call_id: call_id(payload, fallback_call_id),
            operation: collab_operation(payload)?,
            target_thread_id: map_string(
                payload,
                &[
                    "targetThreadId",
                    "target_thread_id",
                    "childThreadId",
                    "child_thread_id",
                ],
            )
            .map(ThreadId::new),
            message: map_string(payload, &["message", "prompt", "detail"]),
            output: tool_output(payload),
            agent_states: Default::default(),
        },
        ItemFamily::Approval => {
            let decision = approval_decision(event_type, payload);
            let resolved = is_action_resolution_event(event_type);
            let scope = approval_scope(payload, decision);
            ThreadItemPayload::Approval {
                request_id: map_string(
                    payload,
                    &["requestId", "request_id", "actionId", "action_id"],
                )
                .unwrap_or_else(|| fallback_call_id.to_string()),
                action: ApprovalAction {
                    kind: map_string(
                        payload,
                        &["actionType", "action_type", "actionKind", "action_kind"],
                    )
                    .unwrap_or_else(|| "approval".to_string()),
                    description: map_string(
                        payload,
                        &["description", "prompt", "message", "reason"],
                    )
                    .unwrap_or_default(),
                },
                scope,
                available_decisions: approval_available_decisions(payload),
                decision,
                requested_at_ms: (!resolved).then_some(timestamp_ms),
                resolved_at_ms: resolved.then_some(timestamp_ms),
                reason_code: map_string(payload, &["reasonCode", "reason_code", "code"]),
                expires_at_ms: map_i64(payload, &["expiresAtMs", "expires_at_ms"]),
            }
        }
        ItemFamily::Command => ThreadItemPayload::Command {
            command: map_string(payload, &["command", "cmd"])
                .unwrap_or_else(|| "command".to_string()),
            cwd: map_string(payload, &["cwd", "workingDirectory", "working_dir"]),
            output: map_string(
                payload,
                &[
                    "output",
                    "stdout",
                    "stderr",
                    "result",
                    "outputPreview",
                    "output_preview",
                    "preview",
                    "summary",
                ],
            ),
            exit_code: map_i64(payload, &["exitCode", "exit_code"]).map(|value| value as i32),
        },
        ItemFamily::File => ThreadItemPayload::File {
            changes: file_changes(payload),
            status: file_change_status(event_type, payload),
        },
        ItemFamily::Media if event_type == "artifact.snapshot" => {
            let artifact = payload.get("artifact").and_then(Value::as_object);
            ThreadItemPayload::File {
                changes: vec![FileChange {
                    path: map_string(payload, &["path", "filePath", "file_path"])
                        .or_else(|| {
                            artifact.and_then(|value| map_string(value, &["filePath", "path"]))
                        })
                        .unwrap_or_else(|| "artifact".to_string()),
                    kind: FileChangeKind::Update { move_path: None },
                    diff: map_string(payload, &["content", "preview"])
                        .or_else(|| {
                            artifact
                                .and_then(|value| map_string(value, &["content", "previewText"]))
                        })
                        .unwrap_or_default(),
                }],
                status: FileChangeStatus::Applied,
            }
        }
        ItemFamily::Media => ThreadItemPayload::Media {
            uri: map_string(
                payload,
                &["uri", "url", "artifactRef", "artifact_ref", "path"],
            )
            .unwrap_or_else(|| "unknown".to_string()),
            mime_type: map_string(
                payload,
                &["mimeType", "mime_type", "contentType", "content_type"],
            )
            .unwrap_or_else(|| "application/octet-stream".to_string()),
            preview: map_string(payload, &["preview", "thumbnail", "summary"]),
        },
        ItemFamily::SubAgent => ThreadItemPayload::SubAgent {
            child_thread_id: ThreadId::new(
                map_string(
                    payload,
                    &[
                        "childThreadId",
                        "child_thread_id",
                        "agentThreadId",
                        "agent_thread_id",
                        "sessionId",
                        "session_id",
                        "threadId",
                        "thread_id",
                    ],
                )
                .unwrap_or_else(|| "unknown".to_string()),
            ),
            activity: subagent_activity(event_type, payload),
            detail: map_string(payload, &["detail", "message", "reason", "status"]),
        },
        ItemFamily::ContextCompaction => ThreadItemPayload::ContextCompaction {
            summary: map_string(payload, &["summary", "message", "text"]),
            replacement_history: payload
                .get("replacementHistory")
                .or_else(|| payload.get("replacement_history"))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            window_number: map_u64(payload, &["windowNumber", "window_number"]),
            first_window_id: map_string(payload, &["firstWindowId", "first_window_id"]),
            previous_window_id: map_string(payload, &["previousWindowId", "previous_window_id"]),
            window_id: map_string(payload, &["windowId", "window_id", "contextWindowId"]),
            tail_start_turn_id: map_string(payload, &["tailStartTurnId", "tail_start_turn_id"]),
        },
        ItemFamily::Unknown => ThreadItemPayload::Unknown {
            upstream_type: unknown_item_upstream_type(payload)?,
            field_names: unknown_item_field_names(payload),
        },
    })
}

fn unknown_item_upstream_type(payload: &Map<String, Value>) -> Option<String> {
    let upstream_type = map_string(payload, &["kind", "type", "itemType", "item_type"])?;
    is_canonical_unknown_item_type(&upstream_type).then_some(upstream_type)
}

fn is_canonical_unknown_item_type(value: &str) -> bool {
    let mut chars = value.chars();
    chars.next().is_some_and(|first| first.is_ascii_lowercase())
        && chars.all(|character| character.is_ascii_alphanumeric())
}

pub(super) fn unknown_payload_is_safe(upstream_type: &str, field_names: &[String]) -> bool {
    if !is_canonical_unknown_item_type(upstream_type)
        || field_names.len() > UNKNOWN_ITEM_FIELD_NAME_LIMIT
    {
        return false;
    }
    let sanitized = field_names.iter().cloned().collect::<BTreeSet<_>>();
    sanitized.len() == field_names.len()
        && sanitized.iter().eq(field_names.iter())
        && field_names.iter().all(|name| {
            name == UNKNOWN_ITEM_REDACTED_FIELD_NAME
                || (is_safe_unknown_item_field_name(name)
                    && !is_sensitive_unknown_item_field_name(name))
        })
}

fn unknown_item_field_names(payload: &Map<String, Value>) -> Vec<String> {
    payload
        .keys()
        .filter(|name| !is_unknown_item_identity_field(name))
        .map(|name| sanitize_unknown_item_field_name(name))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .take(UNKNOWN_ITEM_FIELD_NAME_LIMIT)
        .collect()
}

fn is_unknown_item_identity_field(name: &str) -> bool {
    matches!(
        name,
        "id" | "itemId"
            | "item_id"
            | "kind"
            | "type"
            | "itemType"
            | "item_type"
            | "sessionId"
            | "session_id"
            | "threadId"
            | "thread_id"
            | "turnId"
            | "turn_id"
    )
}

fn sanitize_unknown_item_field_name(name: &str) -> String {
    if is_safe_unknown_item_field_name(name) && !is_sensitive_unknown_item_field_name(name) {
        name.to_string()
    } else {
        UNKNOWN_ITEM_REDACTED_FIELD_NAME.to_string()
    }
}

fn is_safe_unknown_item_field_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_alphabetic()
        && name.len() <= 64
        && chars
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

fn is_sensitive_unknown_item_field_name(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    [
        "authorization",
        "cookie",
        "credential",
        "password",
        "secret",
        "token",
    ]
    .iter()
    .any(|sensitive| normalized.contains(sensitive))
        || normalized.contains("apikey")
        || normalized.contains("api_key")
        || normalized.contains("api-key")
}

fn user_message_content(payload: &Map<String, Value>) -> Option<Vec<AgentInput>> {
    let content = match payload.get("input") {
        Some(input) => serde_json::from_value::<Vec<AgentInput>>(input.clone()).ok()?,
        None => vec![AgentInput::text(message_text(payload))],
    };
    super::super::super::turn_start::validate_user_input(&content)
        .is_ok()
        .then_some(content)
}

fn message_content_parts(payload: &Map<String, Value>) -> Option<Vec<MessageContentPart>> {
    let list_value = matching_alias(payload, "contentParts", "content_parts")?;
    let single_value = matching_alias(payload, "contentPart", "content_part")?;
    let mut parts = list_value
        .map(|value| serde_json::from_value::<Vec<MessageContentPart>>(value.clone()))
        .transpose()
        .ok()?
        .unwrap_or_default();
    let single = single_value
        .map(|value| serde_json::from_value::<MessageContentPart>(value.clone()))
        .transpose()
        .ok()?;
    if let Some(single) = single {
        if parts.is_empty() {
            parts.push(single);
        } else if parts.first() != Some(&single) {
            return None;
        }
    }
    parts
        .iter()
        .all(MessageContentPart::is_safe)
        .then_some(parts)
}

fn matching_alias<'a>(
    payload: &'a Map<String, Value>,
    first: &str,
    second: &str,
) -> Option<Option<&'a Value>> {
    match (payload.get(first), payload.get(second)) {
        (Some(first), Some(second)) if first != second => None,
        (Some(value), _) | (_, Some(value)) => Some(Some(value)),
        (None, None) => Some(None),
    }
}

fn plan_steps(payload: &Map<String, Value>) -> Vec<PlanStep> {
    payload
        .get("plan")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|value| {
            let value = value.as_object()?;
            let step = map_string(value, &["step"])?;
            let status = match map_string(value, &["status"])
                .unwrap_or_else(|| "pending".to_string())
                .to_ascii_lowercase()
                .as_str()
            {
                "inprogress" | "in_progress" | "running" => PlanStepStatus::InProgress,
                "completed" | "complete" | "done" => PlanStepStatus::Completed,
                _ => PlanStepStatus::Pending,
            };
            Some(PlanStep { step, status })
        })
        .collect()
}

fn tool_arguments(payload: &Map<String, Value>) -> Vec<agent_protocol::ToolArgument> {
    let Some(arguments) = ["arguments", "args", "input"]
        .iter()
        .find_map(|key| payload.get(*key))
    else {
        return Vec::new();
    };
    match arguments {
        Value::Object(object) => object
            .iter()
            .map(|(name, value)| agent_protocol::ToolArgument {
                name: name.clone(),
                value: compact_value(value),
            })
            .collect(),
        Value::Array(values) => values
            .iter()
            .enumerate()
            .map(|(index, value)| agent_protocol::ToolArgument {
                name: index.to_string(),
                value: compact_value(value),
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn call_id(payload: &Map<String, Value>, fallback: &str) -> String {
    map_string(
        payload,
        &["callId", "call_id", "toolCallId", "tool_call_id"],
    )
    .unwrap_or_else(|| fallback.to_string())
}

fn tool_output(payload: &Map<String, Value>) -> Option<ToolOutput> {
    let raw_output = ["output", "result"]
        .iter()
        .find_map(|key| payload.get(*key));
    let text = raw_output
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| map_string(payload, &["outputText", "output_text"]));
    let structured_content = payload
        .get("structuredContent")
        .or_else(|| payload.get("structured_content"))
        .cloned()
        .or_else(|| raw_output.filter(|value| !value.is_string()).cloned());
    let output = ToolOutput {
        text,
        structured_content,
        error: map_string(payload, &["error", "errorMessage", "error_message"]),
        duration_ms: map_u64(payload, &["durationMs", "duration_ms"]),
        truncated: map_bool(
            payload,
            &["truncated", "outputTruncated", "output_truncated"],
        )
        .unwrap_or(false),
        output_ref: map_string(
            payload,
            &[
                "outputRef",
                "output_ref",
                super::super::super::output_refs::SIDECAR_REF_FIELD,
                "sidecar_ref",
            ],
        ),
    };
    (output != ToolOutput::default()).then_some(output)
}

fn collab_operation(payload: &Map<String, Value>) -> Option<CollabAgentOperation> {
    let value = map_string(
        payload,
        &[
            "operation",
            "collabOperation",
            "collab_operation",
            "toolName",
            "tool_name",
            "name",
        ],
    )
    .unwrap_or_default()
    .to_ascii_lowercase();
    match value.as_str() {
        "wait" | "wait_agent" => Some(CollabAgentOperation::Wait),
        _ => None,
    }
}

fn approval_scope(
    payload: &Map<String, Value>,
    decision: Option<ApprovalDecision>,
) -> ApprovalScope {
    if decision == Some(ApprovalDecision::ApprovedForSession) {
        return ApprovalScope::Session;
    }
    match map_string(
        payload,
        &[
            "decisionScope",
            "decision_scope",
            "approvalScope",
            "approval_scope",
            "scope",
        ],
    )
    .unwrap_or_default()
    .to_ascii_lowercase()
    .as_str()
    {
        "turn" => ApprovalScope::Turn,
        "session" => ApprovalScope::Session,
        _ => ApprovalScope::Once,
    }
}

fn approval_available_decisions(payload: &Map<String, Value>) -> Vec<ApprovalDecision> {
    let Some(values) = ["availableDecisions", "available_decisions", "decisions"]
        .iter()
        .find_map(|key| payload.get(*key))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    values
        .iter()
        .filter_map(Value::as_str)
        .filter_map(approval_decision_value)
        .collect()
}

fn approval_decision_value(value: &str) -> Option<ApprovalDecision> {
    match value.to_ascii_lowercase().as_str() {
        "approved" | "approve" | "allow" | "allowed" | "allow_once" => {
            Some(ApprovalDecision::Approved)
        }
        "approved_for_session" | "approvedforsession" | "allow_for_session" => {
            Some(ApprovalDecision::ApprovedForSession)
        }
        "denied" | "deny" | "decline" | "declined" | "reject" | "rejected" => {
            Some(ApprovalDecision::Denied)
        }
        "abort" | "cancelled" | "canceled" | "cancel" => Some(ApprovalDecision::Abort),
        "timed_out" | "timedout" | "timeout" => Some(ApprovalDecision::TimedOut),
        _ => None,
    }
}

fn file_change_status(event_type: &str, payload: &Map<String, Value>) -> FileChangeStatus {
    match map_string(payload, &["status", "state"])
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "applied" => FileChangeStatus::Applied,
        "rejected" | "denied" | "declined" => FileChangeStatus::Rejected,
        "failed" | "error" => FileChangeStatus::Failed,
        _ if event_type.ends_with("applied") => FileChangeStatus::Applied,
        _ if event_type.ends_with("declined") => FileChangeStatus::Rejected,
        _ if event_type.ends_with("failed") => FileChangeStatus::Failed,
        _ => FileChangeStatus::Proposed,
    }
}

fn file_changes(payload: &Map<String, Value>) -> Vec<FileChange> {
    let changes = payload
        .get("changes")
        .and_then(Value::as_array)
        .map(|changes| {
            changes
                .iter()
                .filter_map(file_change_from_value)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !changes.is_empty() {
        return changes;
    }

    let path = map_string(payload, &["path", "filePath", "file_path"]).or_else(|| {
        string_list(payload, &["paths", "changedFiles", "changed_files"])
            .into_iter()
            .next()
    });
    path.map(|path| FileChange {
        path,
        kind: FileChangeKind::Update { move_path: None },
        diff: map_string(payload, &["diff", "patch", "content"]).unwrap_or_default(),
    })
    .into_iter()
    .collect()
}

fn file_change_from_value(value: &Value) -> Option<FileChange> {
    let value = value.as_object()?;
    let kind = map_string(value, &["kind", "changeKind", "change_kind"])
        .unwrap_or_else(|| "update".to_string())
        .to_ascii_lowercase();
    let move_path = map_string(value, &["movePath", "move_path"]);
    let destination_path = map_string(value, &["path", "filePath", "file_path"])?;
    let (path, kind) = match kind.as_str() {
        "add" => (destination_path, FileChangeKind::Add),
        "delete" => (destination_path, FileChangeKind::Delete),
        "update" => (destination_path, FileChangeKind::Update { move_path }),
        "move" | "move_update" => (
            map_string(value, &["sourcePath", "source_path"]).unwrap_or(destination_path.clone()),
            FileChangeKind::Update {
                move_path: move_path.or(Some(destination_path)),
            },
        ),
        _ => return None,
    };
    Some(FileChange {
        path,
        kind,
        diff: value
            .get("diff")
            .map(compact_value)
            .or_else(|| map_string(value, &["patch", "content"]))
            .unwrap_or_default(),
    })
}

fn subagent_activity(event_type: &str, payload: &Map<String, Value>) -> SubAgentActivityKind {
    let value = map_string(payload, &["activity", "kind", "status"])
        .unwrap_or_else(|| {
            event_type
                .rsplit('.')
                .next()
                .unwrap_or_default()
                .to_string()
        })
        .to_ascii_lowercase();
    match value.as_str() {
        "started" => SubAgentActivityKind::Started,
        "interrupted" => SubAgentActivityKind::Interrupted,
        _ => SubAgentActivityKind::Interacted,
    }
}
