use super::{projection_error, required_event_id, EventProjection};
use app_server_protocol::protocol::v2::{
    self, ServerNotification, TurnPlanStep, TurnPlanStepStatus,
};
use app_server_protocol::AgentEvent;
use serde_json::Value;
use tool_runtime::update_plan::{PlanStep, PlanStepStatus};

pub(super) fn project(event: &AgentEvent) -> EventProjection {
    let Some(thread_id) = required_event_id(event.thread_id.as_deref()) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(turn_id) = required_event_id(event.turn_id.as_deref()) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(plan_value) = event.payload.get("plan") else {
        return EventProjection::Reject(projection_error(event));
    };
    let Ok(plan) = serde_json::from_value::<Vec<PlanStep>>(plan_value.clone()) else {
        return EventProjection::Reject(projection_error(event));
    };
    let explanation = match event.payload.get("explanation") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.clone()),
        Some(_) => return EventProjection::Reject(projection_error(event)),
    };

    EventProjection::Direct(vec![ServerNotification::TurnPlanUpdated(
        v2::TurnPlanUpdatedNotification {
            thread_id,
            turn_id,
            explanation,
            plan: plan.into_iter().map(project_step).collect(),
        },
    )
    .into()])
}

fn project_step(step: PlanStep) -> TurnPlanStep {
    TurnPlanStep {
        step: step.step,
        status: match step.status {
            PlanStepStatus::Pending => TurnPlanStepStatus::Pending,
            PlanStepStatus::InProgress => TurnPlanStepStatus::InProgress,
            PlanStepStatus::Completed => TurnPlanStepStatus::Completed,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(payload: Value) -> AgentEvent {
        AgentEvent {
            event_id: "evt-plan-1".to_string(),
            sequence: 3,
            session_id: "session-plan-1".to_string(),
            thread_id: Some("thread-plan-1".to_string()),
            turn_id: Some("turn-plan-1".to_string()),
            event_type: "turn.plan.updated".to_string(),
            timestamp: "2026-07-31T00:00:00.000Z".to_string(),
            payload,
        }
    }

    #[test]
    fn projects_update_plan_checklist_without_plan_item_lifecycle() {
        let EventProjection::Direct(notifications) = project(&event(json!({
            "explanation": "继续实现",
            "plan": [
                { "step": "读现状", "status": "completed" },
                { "step": "补主链", "status": "in_progress" }
            ]
        }))) else {
            panic!("turn plan update should project directly");
        };

        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].method, "turn/plan/updated");
        let params = notifications[0].params.as_ref().expect("params");
        assert_eq!(params["threadId"], "thread-plan-1");
        assert_eq!(params["turnId"], "turn-plan-1");
        assert_eq!(params["plan"][1]["status"], "inProgress");
    }

    #[test]
    fn rejects_invalid_update_plan_status() {
        assert!(matches!(
            project(&event(json!({
                "plan": [{ "step": "补主链", "status": "running" }]
            }))),
            EventProjection::Reject(_)
        ));
    }
}
