use agent_protocol::{ItemStatus, ThreadItem, ThreadItemPayload};
use app_server_protocol::AgentEvent;
use serde_json::{Map, Value};
use std::collections::HashMap;

#[derive(Default)]
struct ToolLifecycleState {
    tools: HashMap<String, ToolState>,
}

#[derive(Default)]
struct ToolState {
    owner: ToolOwner,
    gate: ToolGateState,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct ToolOwner {
    assistant_owner_id: Option<String>,
    item_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolLifecycleSnapshot {
    tool_call_id: String,
    owner: ToolOwner,
    gate: ToolGateState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolGateState {
    Open,
    PendingApproval(ToolApprovalAction),
    Blocked(ToolBlockDecision),
}

impl Default for ToolGateState {
    fn default() -> Self {
        Self::Open
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolApprovalAction {
    action_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolBlockDecision {
    source: ToolBlockSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolBlockSource {
    Approval,
    Permission,
    Sandbox,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolLifecycleViolation {
    code: &'static str,
    event_id: String,
    tool_call_id: Option<String>,
}

#[cfg(test)]
pub(super) fn validate_tool_lifecycle_event(
    existing_events: &[AgentEvent],
    candidate: &AgentEvent,
) -> Result<(), String> {
    let mut validator = ToolLifecycleValidator::from_events(
        existing_events,
        &candidate.session_id,
        candidate.turn_id.as_deref(),
    );
    validator.validate_and_observe(candidate)
}

pub(super) struct ToolLifecycleValidator {
    state: ToolLifecycleState,
}

impl ToolLifecycleValidator {
    pub(super) fn from_events(
        existing_events: &[AgentEvent],
        session_id: &str,
        turn_id: Option<&str>,
    ) -> Self {
        let mut state = ToolLifecycleState::default();
        for event in existing_events
            .iter()
            .filter(|event| event.session_id == session_id && event.turn_id.as_deref() == turn_id)
        {
            state.push_existing(event);
        }
        Self { state }
    }

    pub(super) fn validate_and_observe(&mut self, candidate: &AgentEvent) -> Result<(), String> {
        let violations = self.state.validate_candidate(candidate);
        if violations.is_empty() {
            self.state.push_existing(candidate);
            return Ok(());
        }

        tool_lifecycle_validation_result(&violations)
    }
}

fn tool_lifecycle_validation_result(violations: &[ToolLifecycleViolation]) -> Result<(), String> {
    if violations.is_empty() {
        return Ok(());
    }

    Err(format!(
        "agent runtime tool lifecycle validation failed: {}",
        violations
            .iter()
            .map(format_violation)
            .collect::<Vec<_>>()
            .join("; ")
    ))
}

pub(super) fn normalize_policy_event_payload(
    existing_events: &[AgentEvent],
    turn_id: Option<&str>,
    event_type: &str,
    payload: Value,
) -> Value {
    let event_class = normalize_event_class(event_type);
    if !is_policy_event_class(event_class) {
        return payload;
    }

    let Value::Object(mut object) = payload else {
        return payload;
    };

    if is_action_event_class(event_class) {
        normalize_action_aliases(&mut object);
        if payload_string_from_map(
            &object,
            &["toolCallId", "tool_call_id", "toolId", "tool_id"],
        )
        .is_none()
        {
            let tool_call_id = if event_class == "action.required" {
                active_tool_call_id_for_action(existing_events, turn_id, &object)
            } else {
                action_id_from_map(&object).and_then(|action_id| {
                    previous_action_tool_call_id(existing_events, turn_id, &action_id)
                })
            };
            if let Some(tool_call_id) = tool_call_id {
                object.insert("toolCallId".to_string(), Value::String(tool_call_id));
            }
        }
        if event_class == "action.required" {
            normalize_action_required_kind(&mut object);
        }
    }

    Value::Object(object)
}

impl ToolLifecycleState {
    fn push_existing(&mut self, event: &AgentEvent) {
        match normalize_event_class(&event.event_type) {
            "item.started" => {
                if let Some(tool) = canonical_tool_item(event) {
                    self.tools.insert(
                        tool.call_id,
                        ToolState {
                            owner: ToolOwner::from_event(event),
                            gate: ToolGateState::Open,
                        },
                    );
                }
            }
            "item.completed" => {
                if let Some(tool) = canonical_tool_item(event) {
                    self.tools.remove(&tool.call_id);
                }
            }
            "action.required" => {
                if let Some(tool_call_id) = explicit_tool_call_id(event) {
                    if let Some(tool) = self.tools.get_mut(&tool_call_id) {
                        tool.gate = ToolGateState::PendingApproval(ToolApprovalAction {
                            action_id: action_id(event),
                        });
                    }
                }
            }
            event_class if is_action_terminal_event_class(event_class) => {
                if let Some(tool_call_id) = explicit_tool_call_id(event) {
                    if let Some(tool) = self.tools.get_mut(&tool_call_id) {
                        let action_id = action_id(event);
                        if tool.pending_approval_action_id() == action_id.as_deref() {
                            tool.gate = if action_denies_tool(event) {
                                ToolGateState::Blocked(ToolBlockDecision {
                                    source: ToolBlockSource::Approval,
                                })
                            } else {
                                ToolGateState::Open
                            };
                        }
                    }
                }
            }
            "permission.denied" | "sandbox.blocked" => {
                if let Some(tool_call_id) = explicit_tool_call_id(event) {
                    if let Some(tool) = self.tools.get_mut(&tool_call_id) {
                        let event_class = normalize_event_class(&event.event_type);
                        tool.gate = ToolGateState::Blocked(ToolBlockDecision {
                            source: match event_class {
                                "permission.denied" => ToolBlockSource::Permission,
                                _ => ToolBlockSource::Sandbox,
                            },
                        });
                    }
                }
            }
            _ => {}
        }
    }

    fn validate_candidate(&mut self, event: &AgentEvent) -> Vec<ToolLifecycleViolation> {
        let mut violations = Vec::new();
        let event_class = normalize_event_class(&event.event_type);
        match event_class {
            "item.started" => {
                if let Some(tool) = canonical_tool_item(event) {
                    if self.tool_snapshot(&tool.call_id).is_some() {
                        violations.push(ToolLifecycleViolation {
                            code: "tool_item_started_already_active",
                            event_id: event.event_id.clone(),
                            tool_call_id: Some(tool.call_id),
                        });
                    }
                }
            }
            "tool.progress" => {
                if let Some(tool_call_id) = tool_call_id(event) {
                    if let Some(snapshot) = self.tool_snapshot(&tool_call_id) {
                        validate_tool_owner(
                            &mut violations,
                            event,
                            &snapshot,
                            OwnerRequirement::ExplicitOnly,
                        );
                    } else {
                        violations.push(ToolLifecycleViolation {
                            code: "tool_progress_without_start",
                            event_id: event.event_id.clone(),
                            tool_call_id: Some(tool_call_id),
                        });
                    }
                }
            }
            "tool.output.delta" => {
                if let Some(tool_call_id) = tool_call_id(event) {
                    if let Some(snapshot) = self.tool_snapshot(&tool_call_id) {
                        validate_tool_owner(
                            &mut violations,
                            event,
                            &snapshot,
                            OwnerRequirement::ExplicitOnly,
                        );
                        if snapshot.pending_approval_action_id().is_some() {
                            violations.push(ToolLifecycleViolation {
                                code: "tool_output_before_action_resolved",
                                event_id: event.event_id.clone(),
                                tool_call_id: Some(snapshot.tool_call_id.clone()),
                            });
                        } else if let Some(code) = snapshot.output_blocked_violation_code() {
                            violations.push(snapshot.violation(event, code));
                        }
                    } else {
                        violations.push(ToolLifecycleViolation {
                            code: "tool_output_without_start",
                            event_id: event.event_id.clone(),
                            tool_call_id: Some(tool_call_id),
                        });
                    }
                }
            }
            "item.updated" => {
                if let Some(tool) = canonical_tool_item(event) {
                    if let Some(snapshot) = self.tool_snapshot(&tool.call_id) {
                        validate_tool_owner(
                            &mut violations,
                            event,
                            &snapshot,
                            OwnerRequirement::ExplicitOnly,
                        );
                        if tool.has_output && snapshot.pending_approval_action_id().is_some() {
                            violations.push(ToolLifecycleViolation {
                                code: "tool_output_before_action_resolved",
                                event_id: event.event_id.clone(),
                                tool_call_id: Some(snapshot.tool_call_id.clone()),
                            });
                        } else if tool.has_output {
                            if let Some(code) = snapshot.output_blocked_violation_code() {
                                violations.push(snapshot.violation(event, code));
                            }
                        }
                    } else {
                        violations.push(ToolLifecycleViolation {
                            code: "tool_output_without_start",
                            event_id: event.event_id.clone(),
                            tool_call_id: Some(tool.call_id),
                        });
                    }
                }
            }
            "item.completed" => {
                if let Some(tool) = canonical_tool_item(event) {
                    if let Some(snapshot) = self.tool_snapshot(&tool.call_id) {
                        validate_tool_owner(
                            &mut violations,
                            event,
                            &snapshot,
                            OwnerRequirement::RequiredForTerminal,
                        );
                        if snapshot.pending_approval_action_id().is_some() {
                            violations.push(ToolLifecycleViolation {
                                code: if tool.status == ItemStatus::Completed {
                                    "tool_result_before_action_resolved"
                                } else {
                                    "tool_failed_before_action_resolved"
                                },
                                event_id: event.event_id.clone(),
                                tool_call_id: Some(snapshot.tool_call_id.clone()),
                            });
                        } else if tool.status == ItemStatus::Completed {
                            if let Some(code) = snapshot.result_blocked_violation_code() {
                                violations.push(snapshot.violation(event, code));
                            }
                        }
                    } else {
                        violations.push(ToolLifecycleViolation {
                            code: "tool_item_completed_without_start",
                            event_id: event.event_id.clone(),
                            tool_call_id: Some(tool.call_id),
                        });
                    }
                }
            }
            "permission.denied" | "sandbox.blocked" | "action.required" => {
                if let Some(tool_call_id) = explicit_tool_call_id(event) {
                    if let Some(snapshot) = self.tool_snapshot(&tool_call_id) {
                        validate_tool_owner(
                            &mut violations,
                            event,
                            &snapshot,
                            OwnerRequirement::ExplicitOnly,
                        );
                    } else {
                        violations.push(ToolLifecycleViolation {
                            code: "tool_policy_event_without_active_tool",
                            event_id: event.event_id.clone(),
                            tool_call_id: Some(tool_call_id),
                        });
                    }
                }
            }
            _ => {}
        }
        violations
    }

    fn tool_snapshot(&self, tool_call_id: &str) -> Option<ToolLifecycleSnapshot> {
        self.tools
            .get(tool_call_id)
            .map(|tool| tool.snapshot(tool_call_id))
    }
}

#[derive(Debug, Clone)]
struct ActiveToolCandidate {
    name: Option<String>,
    arguments: Option<Value>,
}

#[derive(Debug, Clone)]
struct CanonicalToolItem {
    call_id: String,
    name: String,
    arguments: Value,
    item_id: String,
    status: ItemStatus,
    has_output: bool,
}

#[derive(Clone, Copy)]
enum OwnerRequirement {
    ExplicitOnly,
    RequiredForTerminal,
}

impl ToolState {
    fn snapshot(&self, tool_call_id: &str) -> ToolLifecycleSnapshot {
        ToolLifecycleSnapshot {
            tool_call_id: tool_call_id.to_string(),
            owner: self.owner.clone(),
            gate: self.gate.clone(),
        }
    }

    fn pending_approval_action_id(&self) -> Option<&str> {
        match &self.gate {
            ToolGateState::PendingApproval(action) => action.action_id.as_deref(),
            _ => None,
        }
    }
}

impl ToolLifecycleSnapshot {
    fn pending_approval_action_id(&self) -> Option<&str> {
        match &self.gate {
            ToolGateState::PendingApproval(action) => action.action_id.as_deref(),
            _ => None,
        }
    }

    fn output_blocked_violation_code(&self) -> Option<&'static str> {
        match &self.gate {
            ToolGateState::Blocked(block) => Some(match block.source {
                ToolBlockSource::Approval => "tool_output_after_action_denied",
                ToolBlockSource::Permission => "tool_output_after_permission_denied",
                ToolBlockSource::Sandbox => "tool_output_after_sandbox_blocked",
            }),
            _ => None,
        }
    }

    fn result_blocked_violation_code(&self) -> Option<&'static str> {
        match &self.gate {
            ToolGateState::Blocked(block) => Some(match block.source {
                ToolBlockSource::Approval => "tool_result_after_action_denied",
                ToolBlockSource::Permission => "tool_result_after_permission_denied",
                ToolBlockSource::Sandbox => "tool_result_after_sandbox_blocked",
            }),
            _ => None,
        }
    }

    fn violation(&self, event: &AgentEvent, code: &'static str) -> ToolLifecycleViolation {
        ToolLifecycleViolation {
            code,
            event_id: event.event_id.clone(),
            tool_call_id: Some(self.tool_call_id.clone()),
        }
    }
}

impl ToolOwner {
    fn from_event(event: &AgentEvent) -> Self {
        if let Some(tool) = canonical_tool_item(event) {
            return Self {
                assistant_owner_id: None,
                item_id: Some(tool.item_id),
            };
        }
        Self {
            assistant_owner_id: string_field(
                &event.payload,
                &[
                    "assistantMessageId",
                    "assistant_message_id",
                    "messageId",
                    "message_id",
                ],
            ),
            item_id: string_field(&event.payload, &["itemId", "item_id"]),
        }
    }

    fn is_empty(&self) -> bool {
        self.assistant_owner_id.is_none() && self.item_id.is_none()
    }

    fn has_comparable_field_with(&self, other: &ToolOwner) -> bool {
        (self.assistant_owner_id.is_some() && other.assistant_owner_id.is_some())
            || (self.item_id.is_some() && other.item_id.is_some())
    }

    fn matches(&self, other: &ToolOwner) -> bool {
        if self
            .assistant_owner_id
            .as_deref()
            .zip(other.assistant_owner_id.as_deref())
            .is_some_and(|(left, right)| left != right)
        {
            return false;
        }
        if self
            .item_id
            .as_deref()
            .zip(other.item_id.as_deref())
            .is_some_and(|(left, right)| left != right)
        {
            return false;
        }
        self.has_comparable_field_with(other)
    }
}

fn validate_tool_owner(
    violations: &mut Vec<ToolLifecycleViolation>,
    event: &AgentEvent,
    snapshot: &ToolLifecycleSnapshot,
    requirement: OwnerRequirement,
) {
    let event_owner = ToolOwner::from_event(event);
    if snapshot.owner.is_empty() {
        return;
    }
    if event_owner.is_empty() {
        if matches!(requirement, OwnerRequirement::RequiredForTerminal) {
            violations.push(ToolLifecycleViolation {
                code: "tool_terminal_missing_owner",
                event_id: event.event_id.clone(),
                tool_call_id: Some(snapshot.tool_call_id.clone()),
            });
        }
        return;
    }
    if !snapshot.owner.matches(&event_owner) {
        violations.push(ToolLifecycleViolation {
            code: "tool_event_owner_mismatch",
            event_id: event.event_id.clone(),
            tool_call_id: Some(snapshot.tool_call_id.clone()),
        });
    }
}

fn normalize_event_class(event_type: &str) -> &str {
    match event_type {
        "tool_args_delta" => "tool.args.delta",
        "tool_output_delta" => "tool.output.delta",
        "tool_input_delta" => "tool.input.delta",
        value => value,
    }
}

fn is_policy_event_class(event_class: &str) -> bool {
    is_action_event_class(event_class)
        || matches!(event_class, "permission.denied" | "sandbox.blocked")
}

fn is_action_event_class(event_class: &str) -> bool {
    event_class == "action.required" || is_action_terminal_event_class(event_class)
}

fn normalize_action_aliases(object: &mut Map<String, Value>) {
    if let Some(action_id) = payload_string_from_map(object, &["actionId", "action_id"])
        .or_else(|| payload_string_from_map(object, &["requestId", "request_id"]))
    {
        insert_string_if_absent(object, "actionId", action_id.clone());
        insert_string_if_absent(object, "requestId", action_id);
    }

    if let Some(action_type) = payload_string_from_map(object, &["actionType", "action_type"]) {
        insert_string_if_absent(object, "actionType", action_type);
    }
}

fn normalize_action_required_kind(object: &mut Map<String, Value>) {
    if payload_string_from_map(object, &["actionKind", "action_kind"]).is_some() {
        return;
    }
    let action_type = payload_string_from_map(object, &["actionType", "action_type"])
        .unwrap_or_else(|| "runtime_action".to_string());
    let action_kind = match action_type.as_str() {
        "tool_confirmation" => "approve-tool",
        "ask_user" | "elicitation" => "provide-input",
        _ => "runtime-action",
    };
    object.insert(
        "actionKind".to_string(),
        Value::String(action_kind.to_string()),
    );
}

fn active_tool_call_id_for_action(
    existing_events: &[AgentEvent],
    turn_id: Option<&str>,
    action_payload: &Map<String, Value>,
) -> Option<String> {
    let active_tools = active_tools_for_turn(existing_events, turn_id);
    let action_id = action_id_from_map(action_payload);
    if let Some(action_id) = action_id.as_deref() {
        if active_tools.contains_key(action_id) {
            return Some(action_id.to_string());
        }
    }

    let action_tool_name = action_tool_name(action_payload)?;
    let action_arguments = action_arguments(action_payload);
    active_tools
        .iter()
        .find(|(_, tool)| {
            tool.name
                .as_deref()
                .is_some_and(|tool_name| lookup_key(tool_name) == lookup_key(&action_tool_name))
                && action_arguments_match(tool.arguments.as_ref(), action_arguments.as_ref())
        })
        .map(|(tool_call_id, _)| tool_call_id.clone())
}

fn previous_action_tool_call_id(
    existing_events: &[AgentEvent],
    turn_id: Option<&str>,
    action_id: &str,
) -> Option<String> {
    existing_events
        .iter()
        .rev()
        .filter(|event| same_turn(event, turn_id))
        .find(|event| {
            normalize_event_class(&event.event_type) == "action.required"
                && event_action_id(event).as_deref() == Some(action_id)
        })
        .and_then(|event| explicit_tool_call_id(event).or_else(|| tool_call_id(event)))
}

fn active_tools_for_turn(
    existing_events: &[AgentEvent],
    turn_id: Option<&str>,
) -> HashMap<String, ActiveToolCandidate> {
    let mut active_tools = HashMap::new();
    for event in existing_events
        .iter()
        .filter(|event| same_turn(event, turn_id))
    {
        match normalize_event_class(&event.event_type) {
            "item.started" => {
                if let Some(tool) = canonical_tool_item(event) {
                    active_tools.insert(
                        tool.call_id,
                        ActiveToolCandidate {
                            name: Some(tool.name),
                            arguments: Some(tool.arguments),
                        },
                    );
                }
            }
            "item.completed" => {
                if let Some(tool) = canonical_tool_item(event) {
                    active_tools.remove(&tool.call_id);
                }
            }
            _ => {}
        }
    }
    active_tools
}

fn same_turn(event: &AgentEvent, turn_id: Option<&str>) -> bool {
    event.turn_id.as_deref() == turn_id
}

fn event_action_id(event: &AgentEvent) -> Option<String> {
    string_field(
        &event.payload,
        &["actionId", "action_id", "requestId", "request_id", "id"],
    )
}

fn action_id_from_map(object: &Map<String, Value>) -> Option<String> {
    payload_string_from_map(
        object,
        &["actionId", "action_id", "requestId", "request_id", "id"],
    )
}

fn action_tool_name(object: &Map<String, Value>) -> Option<String> {
    payload_string_from_map(object, &["toolName", "tool_name"]).or_else(|| {
        object
            .get("data")
            .and_then(Value::as_object)
            .and_then(|data| payload_string_from_map(data, &["toolName", "tool_name"]))
    })
}

fn action_arguments(object: &Map<String, Value>) -> Option<Value> {
    object
        .get("arguments")
        .cloned()
        .or_else(|| object.get("data")?.as_object()?.get("arguments").cloned())
}

fn action_arguments_match(
    tool_arguments: Option<&Value>,
    action_arguments: Option<&Value>,
) -> bool {
    match (tool_arguments, action_arguments) {
        (Some(left), Some(right)) => canonical_json(left) == canonical_json(right),
        (_, None) => true,
        _ => false,
    }
}

fn canonical_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
}

fn lookup_key(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

fn payload_string_from_map(object: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn insert_string_if_absent(object: &mut Map<String, Value>, key: &str, value: String) {
    if object
        .get(key)
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
    {
        return;
    }
    object.insert(key.to_string(), Value::String(value));
}

fn canonical_tool_item(event: &AgentEvent) -> Option<CanonicalToolItem> {
    if !matches!(
        normalize_event_class(&event.event_type),
        "item.started" | "item.updated" | "item.completed"
    ) {
        return None;
    }
    let item = serde_json::from_value::<ThreadItem>(event.payload.get("item")?.clone()).ok()?;
    let (call_id, name, arguments, output) = match item.payload {
        ThreadItemPayload::Tool {
            call_id,
            name,
            arguments,
            output,
        } => (call_id, name, arguments, output),
        ThreadItemPayload::McpToolCall {
            call_id,
            tool_name,
            arguments,
            output,
            ..
        } => (call_id, tool_name, arguments, output),
        _ => return None,
    };
    let arguments = arguments
        .into_iter()
        .map(|argument| {
            let value = serde_json::from_str(&argument.value)
                .unwrap_or_else(|_| Value::String(argument.value));
            (argument.name, value)
        })
        .collect::<Map<_, _>>();
    Some(CanonicalToolItem {
        call_id,
        name,
        arguments: Value::Object(arguments),
        item_id: item.item_id.to_string(),
        status: item.status,
        has_output: output.is_some(),
    })
}

fn tool_call_id(event: &AgentEvent) -> Option<String> {
    if let Some(tool) = canonical_tool_item(event) {
        return Some(tool.call_id);
    }
    string_field(
        &event.payload,
        &["toolCallId", "tool_call_id", "toolId", "tool_id", "id"],
    )
}

fn explicit_tool_call_id(event: &AgentEvent) -> Option<String> {
    if let Some(tool) = canonical_tool_item(event) {
        return Some(tool.call_id);
    }
    string_field(
        &event.payload,
        &["toolCallId", "tool_call_id", "toolId", "tool_id"],
    )
}

fn action_id(event: &AgentEvent) -> Option<String> {
    string_field(
        &event.payload,
        &["actionId", "action_id", "requestId", "request_id"],
    )
}

fn is_action_terminal_event_class(event_class: &str) -> bool {
    matches!(
        event_class,
        "action.resolved" | "action.cancelled" | "action.canceled" | "action.expired"
    )
}

fn action_denies_tool(event: &AgentEvent) -> bool {
    let event_class = normalize_event_class(&event.event_type);
    if matches!(
        event_class,
        "action.cancelled" | "action.canceled" | "action.expired"
    ) {
        return true;
    }
    let Some(object) = event.payload.as_object() else {
        return false;
    };
    ["decision", "status", "result"]
        .iter()
        .filter_map(|key| object.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .map(|value| {
            matches!(
                value,
                "deny"
                    | "denied"
                    | "decline"
                    | "declined"
                    | "reject"
                    | "rejected"
                    | "cancel"
                    | "canceled"
                    | "cancelled"
            )
        })
        .unwrap_or(false)
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    let object = value.as_object()?;
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn format_violation(violation: &ToolLifecycleViolation) -> String {
    match violation.tool_call_id.as_deref() {
        Some(tool_call_id) => format!(
            "{} event_id={} tool_call_id={}",
            violation.code, violation.event_id, tool_call_id
        ),
        None => format!("{} event_id={}", violation.code, violation.event_id),
    }
}

#[cfg(test)]
#[path = "tool_lifecycle_tests.rs"]
mod tests;
