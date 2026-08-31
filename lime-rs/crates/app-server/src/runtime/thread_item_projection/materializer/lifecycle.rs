use super::fields::{map_string, payload_source, value_i64, value_string, value_u64};
use agent_protocol::{ApprovalDecision, ItemStatus, TurnApprovalState, TurnQueueState};
use app_server_protocol::AgentEvent;
use chrono::{DateTime, FixedOffset};
use serde_json::{Map, Value};

pub(super) fn item_status(event_type: &str, payload: &Value) -> ItemStatus {
    if matches!(
        event_type,
        "message.created" | "plan.final" | "artifact.snapshot"
    ) {
        return ItemStatus::Completed;
    }
    if event_type == "tool.progress" {
        return ItemStatus::InProgress;
    }
    if let Some(status) = value_string(payload, &["status", "state"]) {
        match status.to_ascii_lowercase().as_str() {
            "completed" | "complete" | "success" | "succeeded" | "applied" => {
                return ItemStatus::Completed
            }
            "failed" | "error" => return ItemStatus::Failed,
            "cancelled" | "canceled" => return ItemStatus::Cancelled,
            "interrupted" => return ItemStatus::Interrupted,
            _ => {}
        }
    }
    if event_type.ends_with("completed")
        || event_type.ends_with("result")
        || event_type.ends_with("exited")
        || event_type.ends_with("applied")
        || event_type.ends_with("declined")
        || event_type.ends_with("resolved")
    {
        ItemStatus::Completed
    } else if event_type.ends_with("failed") || event_type.ends_with("denied") {
        ItemStatus::Failed
    } else if event_type.ends_with("cancelled") || event_type.ends_with("canceled") {
        ItemStatus::Cancelled
    } else if event_type.ends_with("started") || event_type.ends_with("delta") {
        ItemStatus::InProgress
    } else {
        ItemStatus::Pending
    }
}

pub(super) fn approval_decision(
    event_type: &str,
    payload: &Map<String, Value>,
) -> Option<ApprovalDecision> {
    let explicit = map_string(
        payload,
        &["decision", "approvalDecision", "approval_decision"],
    )
    .unwrap_or_default()
    .to_ascii_lowercase();
    if explicit.contains("approve")
        || explicit == "allow"
        || explicit == "allowed"
        || explicit == "allow_once"
        || explicit == "allow_for_session"
    {
        if explicit == "allow_for_session"
            || explicit == "approved_for_session"
            || explicit == "approvedforsession"
        {
            Some(ApprovalDecision::ApprovedForSession)
        } else {
            Some(ApprovalDecision::Approved)
        }
    } else if explicit.contains("deny")
        || explicit == "decline"
        || explicit == "declined"
        || explicit == "reject"
        || explicit == "rejected"
    {
        Some(ApprovalDecision::Denied)
    } else if explicit.contains("timeout") || event_type.ends_with("expired") {
        Some(ApprovalDecision::TimedOut)
    } else if matches!(explicit.as_str(), "cancel" | "cancelled" | "canceled")
        || event_type.ends_with("cancelled")
        || event_type.ends_with("canceled")
    {
        Some(ApprovalDecision::Abort)
    } else if event_type.ends_with("required") || explicit == "pending" {
        None
    } else {
        None
    }
}

pub(super) fn is_action_resolution_event(event_type: &str) -> bool {
    event_type.ends_with("resolved")
        || event_type.ends_with("cancelled")
        || event_type.ends_with("canceled")
        || event_type.ends_with("expired")
}

pub(super) fn turn_queue_state(event: &AgentEvent) -> TurnQueueState {
    match event.event_type.as_str() {
        "turn.queued" | "queue.added" => TurnQueueState::Queued {
            position: value_u64(
                &event.payload,
                &["position", "queuePosition", "queue_position"],
            )
            .map(|value| value.min(u32::MAX as u64) as u32),
        },
        "turn.accepted" | "turn.started" => TurnQueueState::Running,
        _ => TurnQueueState::default(),
    }
}

pub(super) fn queued_turn_id(event: &AgentEvent) -> Option<String> {
    value_string(&event.payload, &["queuedTurnId", "queued_turn_id"])
}

pub(super) fn turn_approval_state(event: &AgentEvent) -> TurnApprovalState {
    match event.event_type.as_str() {
        "action.required" | "approval.required" => TurnApprovalState::Pending,
        "action.resolved" | "approval.resolved" => {
            match approval_decision(event.event_type.as_str(), payload_source(&event.payload)) {
                Some(ApprovalDecision::Approved | ApprovalDecision::ApprovedForSession) => {
                    TurnApprovalState::Approved
                }
                Some(ApprovalDecision::Denied) => TurnApprovalState::Denied,
                Some(ApprovalDecision::Abort) => TurnApprovalState::Cancelled,
                Some(ApprovalDecision::TimedOut) => TurnApprovalState::TimedOut,
                None => TurnApprovalState::Resolved,
            }
        }
        "action.cancelled" | "action.canceled" | "approval.cancelled" => {
            TurnApprovalState::Cancelled
        }
        "action.expired" | "approval.expired" => TurnApprovalState::TimedOut,
        _ => TurnApprovalState::default(),
    }
}

pub(super) fn event_timestamp_ms(event: &AgentEvent) -> i64 {
    parse_timestamp_ms(&event.timestamp)
        .or_else(|| {
            value_i64(
                &event.payload,
                &["timestampMs", "timestamp_ms", "createdAtMs"],
            )
        })
        .unwrap_or(event.sequence as i64)
}

pub(super) fn parse_timestamp_ms(value: &str) -> Option<i64> {
    DateTime::<FixedOffset>::parse_from_rfc3339(value)
        .ok()
        .map(|date| date.timestamp_millis())
}

pub(super) fn rollback_target(event: &AgentEvent) -> Option<u64> {
    if !matches!(
        event.event_type.as_str(),
        "turn.rollback" | "history.rollback" | "thread.rollback" | "turn.canceled"
    ) {
        return value_u64(
            &event.payload,
            &["rollbackToSequence", "rollback_to_sequence"],
        );
    }
    value_u64(
        &event.payload,
        &[
            "rollbackToSequence",
            "rollback_to_sequence",
            "targetSequence",
            "target_sequence",
            "sequence",
        ],
    )
}
