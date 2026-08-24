use app_server_protocol::AgentSessionStatus;
use app_server_protocol::AgentTurnStatus;

pub(super) fn session_status_from_event(event_type: &str) -> &'static str {
    match event_type {
        "session.created" => "idle",
        "turn.completed" => "completed",
        "turn.failed" => "failed",
        "turn.canceled" => "canceled",
        "history.rollback" => "idle",
        "action.required" => "waitingAction",
        "turn.accepted" | "turn.started" | "message.created" | "message.delta" => "running",
        _ => "active",
    }
}

pub(super) fn agent_session_status_from_projection(status: &str) -> AgentSessionStatus {
    match status {
        "running" | "active" => AgentSessionStatus::Running,
        "waitingAction" | "waiting_action" => AgentSessionStatus::WaitingAction,
        "failed" => AgentSessionStatus::Failed,
        "canceled" => AgentSessionStatus::Canceled,
        "completed" => AgentSessionStatus::Completed,
        _ => AgentSessionStatus::Idle,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waiting_action_projection_preserves_session_and_turn_status() {
        assert_eq!(
            agent_session_status_from_projection("waitingAction"),
            AgentSessionStatus::WaitingAction
        );
        assert_eq!(
            agent_turn_status_from_projection("waitingAction"),
            AgentTurnStatus::WaitingAction
        );
    }
}

pub(super) fn turn_status_from_event(event_type: &str) -> &'static str {
    match event_type {
        "turn.completed" => "completed",
        "turn.failed" => "failed",
        "turn.canceled" => "canceled",
        "action.required" => "waitingAction",
        "queue.added" => "queued",
        "turn.accepted" => "accepted",
        "turn.started" | "message.created" | "message.delta" => "running",
        _ => "active",
    }
}

pub(super) fn agent_turn_status_from_projection(status: &str) -> AgentTurnStatus {
    match status {
        "accepted" => AgentTurnStatus::Accepted,
        "queued" => AgentTurnStatus::Queued,
        "running" | "active" => AgentTurnStatus::Running,
        "waitingAction" | "waiting_action" => AgentTurnStatus::WaitingAction,
        "completed" => AgentTurnStatus::Completed,
        "failed" => AgentTurnStatus::Failed,
        "canceled" | "aborted" => AgentTurnStatus::Canceled,
        _ => AgentTurnStatus::Running,
    }
}
