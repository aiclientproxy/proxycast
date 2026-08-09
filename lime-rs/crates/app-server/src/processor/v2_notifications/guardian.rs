use super::{projection_error, required_event_id, EventProjection};
use app_server_protocol::protocol::v2::{self, ServerNotification};
use app_server_protocol::AgentEvent;
use serde_json::Value;

pub(super) fn project_started(event: &AgentEvent) -> EventProjection {
    let Some(payload) = payload(event) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(thread_id) = required_event_id(event.thread_id.as_deref()) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(turn_id) = required_event_id(event.turn_id.as_deref()).or_else(|| {
        payload
            .get("turnId")
            .and_then(Value::as_str)
            .map(str::to_string)
    }) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(review_id) = payload
        .get("reviewId")
        .or_else(|| payload.get("review_id"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
    else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(started_at_ms) = payload
        .get("startedAtMs")
        .or_else(|| payload.get("started_at_ms"))
        .and_then(Value::as_i64)
    else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(action) = payload.get("action").cloned().and_then(parse_action) else {
        return EventProjection::Reject(projection_error(event));
    };
    let target_item_id = payload
        .get("targetItemId")
        .or_else(|| payload.get("target_item_id"))
        .and_then(Value::as_str)
        .map(str::to_string);
    EventProjection::Direct(vec![ServerNotification::ItemAutoApprovalReviewStarted(
        v2::ItemGuardianApprovalReviewStartedNotification {
            thread_id,
            turn_id,
            started_at_ms,
            review_id,
            target_item_id,
            review: v2::GuardianApprovalReview {
                status: v2::GuardianApprovalReviewStatus::InProgress,
                risk_level: None,
                user_authorization: None,
                rationale: None,
            },
            action,
        },
    )
    .into()])
}

pub(super) fn project_completed(event: &AgentEvent) -> EventProjection {
    let Some(payload) = payload(event) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(thread_id) = required_event_id(event.thread_id.as_deref()) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(turn_id) = required_event_id(event.turn_id.as_deref()).or_else(|| {
        payload
            .get("turnId")
            .and_then(Value::as_str)
            .map(str::to_string)
    }) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(review_id) = payload
        .get("reviewId")
        .or_else(|| payload.get("review_id"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
    else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(started_at_ms) = payload
        .get("startedAtMs")
        .or_else(|| payload.get("started_at_ms"))
        .and_then(Value::as_i64)
    else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(completed_at_ms) = payload
        .get("completedAtMs")
        .or_else(|| payload.get("completed_at_ms"))
        .and_then(Value::as_i64)
    else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(status) = payload
        .get("status")
        .and_then(Value::as_str)
        .and_then(parse_status)
    else {
        return EventProjection::Reject(projection_error(event));
    };
    if matches!(status, v2::GuardianApprovalReviewStatus::InProgress) {
        return EventProjection::Reject(projection_error(event));
    }
    let decision_source = match payload
        .get("decisionSource")
        .or_else(|| payload.get("decision_source"))
        .and_then(Value::as_str)
    {
        Some("agent") => v2::AutoReviewDecisionSource::Agent,
        _ => return EventProjection::Reject(projection_error(event)),
    };
    let Some(action) = payload.get("action").cloned().and_then(parse_action) else {
        return EventProjection::Reject(projection_error(event));
    };
    let target_item_id = payload
        .get("targetItemId")
        .or_else(|| payload.get("target_item_id"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let risk_level = payload
        .get("riskLevel")
        .or_else(|| payload.get("risk_level"))
        .and_then(Value::as_str)
        .and_then(parse_risk_level);
    let user_authorization = payload
        .get("userAuthorization")
        .or_else(|| payload.get("user_authorization"))
        .and_then(Value::as_str)
        .and_then(parse_user_authorization);
    let rationale = payload
        .get("rationale")
        .and_then(Value::as_str)
        .map(str::to_string);
    EventProjection::Direct(vec![ServerNotification::ItemAutoApprovalReviewCompleted(
        v2::ItemGuardianApprovalReviewCompletedNotification {
            thread_id,
            turn_id,
            started_at_ms,
            completed_at_ms,
            review_id,
            target_item_id,
            decision_source,
            review: v2::GuardianApprovalReview {
                status,
                risk_level,
                user_authorization,
                rationale,
            },
            action,
        },
    )
    .into()])
}

fn payload(event: &AgentEvent) -> Option<&Value> {
    event
        .payload
        .get("runtimeEvent")
        .filter(|value| value.is_object())
        .or(Some(&event.payload))
}

fn parse_action(value: Value) -> Option<v2::GuardianApprovalReviewAction> {
    serde_json::from_value(value).ok()
}

fn parse_status(value: &str) -> Option<v2::GuardianApprovalReviewStatus> {
    match value {
        "inProgress" | "in_progress" => Some(v2::GuardianApprovalReviewStatus::InProgress),
        "approved" => Some(v2::GuardianApprovalReviewStatus::Approved),
        "denied" => Some(v2::GuardianApprovalReviewStatus::Denied),
        "timedOut" | "timed_out" => Some(v2::GuardianApprovalReviewStatus::TimedOut),
        "aborted" => Some(v2::GuardianApprovalReviewStatus::Aborted),
        _ => None,
    }
}

fn parse_risk_level(value: &str) -> Option<v2::GuardianRiskLevel> {
    match value {
        "low" => Some(v2::GuardianRiskLevel::Low),
        "medium" => Some(v2::GuardianRiskLevel::Medium),
        "high" => Some(v2::GuardianRiskLevel::High),
        "critical" => Some(v2::GuardianRiskLevel::Critical),
        _ => None,
    }
}

fn parse_user_authorization(value: &str) -> Option<v2::GuardianUserAuthorization> {
    match value {
        "unknown" => Some(v2::GuardianUserAuthorization::Unknown),
        "low" => Some(v2::GuardianUserAuthorization::Low),
        "medium" => Some(v2::GuardianUserAuthorization::Medium),
        "high" => Some(v2::GuardianUserAuthorization::High),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(event_type: &str, payload: Value) -> AgentEvent {
        AgentEvent {
            event_id: "guardian-event-1".to_string(),
            sequence: 1,
            session_id: "session-guardian".to_string(),
            thread_id: Some("thread-guardian".to_string()),
            turn_id: Some("turn-guardian".to_string()),
            event_type: event_type.to_string(),
            timestamp: "2026-08-09T00:00:00.000Z".to_string(),
            payload,
        }
    }

    fn action() -> Value {
        json!({
            "command": "git status --short",
            "cwd": "/workspace",
            "source": "shell",
            "type": "command"
        })
    }

    #[test]
    fn projects_guardian_review_lifecycle_to_typed_notifications() {
        let EventProjection::Direct(started) = project_started(&event(
            "guardian.review.started",
            json!({
                "action": action(),
                "reviewId": "guardian-1",
                "startedAtMs": 1_783_814_400_100i64,
                "targetItemId": "item-command"
            }),
        )) else {
            panic!("Guardian start should project directly");
        };
        assert_eq!(started[0].method, "item/autoApprovalReview/started");
        let started_params = started[0].params.as_ref().expect("started params");
        assert_eq!(started_params["review"]["status"], "inProgress");
        assert_eq!(started_params["action"]["type"], "command");

        let EventProjection::Direct(completed) = project_completed(&event(
            "guardian.review.completed",
            json!({
                "action": action(),
                "completedAtMs": 1_783_814_401_100i64,
                "decisionSource": "agent",
                "rationale": "workspace read only",
                "reviewId": "guardian-1",
                "riskLevel": "low",
                "startedAtMs": 1_783_814_400_100i64,
                "status": "approved",
                "targetItemId": "item-command",
                "userAuthorization": "high"
            }),
        )) else {
            panic!("Guardian completion should project directly");
        };
        assert_eq!(completed[0].method, "item/autoApprovalReview/completed");
        let completed_params = completed[0].params.as_ref().expect("completed params");
        assert_eq!(completed_params["review"]["status"], "approved");
        assert_eq!(completed_params["review"]["riskLevel"], "low");
    }

    #[test]
    fn rejects_guardian_review_with_unknown_status_or_action() {
        assert!(matches!(
            project_completed(&event(
                "guardian.review.completed",
                json!({
                    "action": { "type": "future" },
                    "completedAtMs": 2,
                    "decisionSource": "agent",
                    "reviewId": "guardian-1",
                    "startedAtMs": 1,
                    "status": "approved"
                }),
            )),
            EventProjection::Reject(_)
        ));
        assert!(matches!(
            project_completed(&event(
                "guardian.review.completed",
                json!({
                    "action": action(),
                    "completedAtMs": 2,
                    "decisionSource": "agent",
                    "reviewId": "guardian-1",
                    "startedAtMs": 1,
                    "status": "inProgress"
                }),
            )),
            EventProjection::Reject(_)
        ));
    }
}
