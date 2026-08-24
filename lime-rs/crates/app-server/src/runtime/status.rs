use app_server_protocol::AgentSessionStatus;
use app_server_protocol::AgentTurn;
use app_server_protocol::AgentTurnStatus;

pub(super) fn agent_turn_is_active(status: AgentTurnStatus) -> bool {
    matches!(
        status,
        AgentTurnStatus::Accepted
            | AgentTurnStatus::Queued
            | AgentTurnStatus::Running
            | AgentTurnStatus::WaitingAction
    )
}

pub(super) fn agent_turn_is_terminal(status: AgentTurnStatus) -> bool {
    matches!(
        status,
        AgentTurnStatus::Completed | AgentTurnStatus::Failed | AgentTurnStatus::Canceled
    )
}

pub(super) fn agent_turn_advances_queue(status: AgentTurnStatus) -> bool {
    matches!(status, AgentTurnStatus::Completed | AgentTurnStatus::Failed)
}

pub(super) fn agent_turn_blocks_queue_resume(status: AgentTurnStatus) -> bool {
    matches!(
        status,
        AgentTurnStatus::Accepted | AgentTurnStatus::Running | AgentTurnStatus::WaitingAction
    )
}

pub(super) fn agent_session_status_label(status: AgentSessionStatus) -> &'static str {
    match status {
        AgentSessionStatus::Idle => "idle",
        AgentSessionStatus::Running => "running",
        AgentSessionStatus::WaitingAction => "waitingAction",
        AgentSessionStatus::Completed => "completed",
        AgentSessionStatus::Failed => "failed",
        AgentSessionStatus::Canceled => "canceled",
    }
}

pub(super) fn agent_turn_status_label(status: AgentTurnStatus) -> &'static str {
    match status {
        AgentTurnStatus::Accepted => "accepted",
        AgentTurnStatus::Queued => "queued",
        AgentTurnStatus::Running => "running",
        AgentTurnStatus::WaitingAction => "waitingAction",
        AgentTurnStatus::Completed => "completed",
        AgentTurnStatus::Failed => "failed",
        AgentTurnStatus::Canceled => "canceled",
    }
}

pub(super) fn session_status_from_turn_status(turn_status: AgentTurnStatus) -> AgentSessionStatus {
    match turn_status {
        AgentTurnStatus::Accepted | AgentTurnStatus::Queued => AgentSessionStatus::Running,
        AgentTurnStatus::Running => AgentSessionStatus::Running,
        AgentTurnStatus::WaitingAction => AgentSessionStatus::WaitingAction,
        AgentTurnStatus::Completed => AgentSessionStatus::Completed,
        AgentTurnStatus::Failed => AgentSessionStatus::Failed,
        AgentTurnStatus::Canceled => AgentSessionStatus::Canceled,
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RuntimeTurnSnapshot<'a> {
    pub turn_id: &'a str,
    pub status: &'a str,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct SessionRuntimeState {
    pub thread_status: String,
    pub latest_turn_status: Option<String>,
    pub active_turn_id: Option<String>,
    pub queued_turn_count: usize,
}

pub(super) fn resolve_session_runtime_state<'a>(
    session_status: &str,
    pending_request_count: usize,
    turns: impl IntoIterator<Item = RuntimeTurnSnapshot<'a>>,
    active_turn_id: Option<&str>,
) -> SessionRuntimeState {
    let turns = turns.into_iter().collect::<Vec<_>>();
    let session_status = normalize_session_runtime_status(session_status);
    let latest_turn_status = turns
        .last()
        .map(|turn| effective_turn_runtime_status(turn, active_turn_id));

    if session_runtime_status_is_terminal(session_status.as_str()) {
        return SessionRuntimeState {
            thread_status: canonical_terminal_status(session_status.as_str()).to_string(),
            latest_turn_status,
            active_turn_id: None,
            queued_turn_count: 0,
        };
    }

    let queued_turn_count = turns
        .iter()
        .filter(|turn| normalize_turn_runtime_status(turn.status) == "queued")
        .count();
    let has_waiting_turn = turns.iter().any(|turn| {
        active_turn_id == Some(turn.turn_id)
            && normalize_turn_runtime_status(turn.status) == "waitingAction"
    });
    let active_turn_id = active_turn_id.map(ToString::to_string);

    let thread_status =
        if active_turn_id.is_some() && (pending_request_count > 0 || has_waiting_turn) {
            "waitingAction"
        } else if active_turn_id.is_some() || queued_turn_count > 0 {
            "running"
        } else if matches!(
            session_status.as_str(),
            "running" | "active" | "waitingAction"
        ) {
            "idle"
        } else {
            session_status.as_str()
        };

    SessionRuntimeState {
        thread_status: thread_status.to_string(),
        latest_turn_status,
        active_turn_id,
        queued_turn_count,
    }
}

pub(super) fn resolve_agent_session_runtime_state(
    session_status: AgentSessionStatus,
    pending_request_count: usize,
    turns: &[AgentTurn],
    active_turn_id: Option<&str>,
) -> SessionRuntimeState {
    resolve_session_runtime_state(
        agent_session_status_label(session_status),
        pending_request_count,
        turns.iter().map(runtime_turn_state_from_agent_turn),
        active_turn_id,
    )
}

pub(super) fn normalize_agent_session_runtime_snapshot(
    session: &mut app_server_protocol::AgentSession,
    turns: &mut [AgentTurn],
    active_turn_id: Option<&str>,
) {
    let runtime_state =
        resolve_agent_session_runtime_state(session.status, 0, turns, active_turn_id);
    for turn in turns {
        if matches!(
            turn.status,
            AgentTurnStatus::Accepted | AgentTurnStatus::Running | AgentTurnStatus::WaitingAction
        ) && active_turn_id != Some(turn.turn_id.as_str())
        {
            turn.status = AgentTurnStatus::Canceled;
        }
    }
    session.status = match runtime_state.thread_status.as_str() {
        "running" => AgentSessionStatus::Running,
        "waitingAction" => AgentSessionStatus::WaitingAction,
        "completed" => AgentSessionStatus::Completed,
        "failed" => AgentSessionStatus::Failed,
        "canceled" => AgentSessionStatus::Canceled,
        _ => AgentSessionStatus::Idle,
    };
}

pub(super) fn runtime_turn_state_from_agent_turn(turn: &AgentTurn) -> RuntimeTurnSnapshot<'_> {
    RuntimeTurnSnapshot {
        turn_id: turn.turn_id.as_str(),
        status: agent_turn_status_label(turn.status),
    }
}

fn effective_turn_runtime_status(
    turn: &RuntimeTurnSnapshot<'_>,
    active_turn_id: Option<&str>,
) -> String {
    let status = normalize_turn_runtime_status(turn.status);
    if matches!(status.as_str(), "accepted" | "running" | "waitingAction")
        && active_turn_id != Some(turn.turn_id)
    {
        "canceled".to_string()
    } else {
        status
    }
}

fn normalize_session_runtime_status(status: &str) -> String {
    let normalized = normalize_runtime_status_token(status);
    match normalized.as_str() {
        "waitingaction" | "waiting_action" => "waitingAction".to_string(),
        "cancelled" | "aborted" => "canceled".to_string(),
        "" => "idle".to_string(),
        _ => normalized,
    }
}

fn normalize_turn_runtime_status(status: &str) -> String {
    let normalized = normalize_runtime_status_token(status);
    match normalized.as_str() {
        "waitingaction" | "waiting_action" => "waitingAction".to_string(),
        "cancelled" | "aborted" => "canceled".to_string(),
        "active" | "in_progress" | "processing" | "streaming" => "running".to_string(),
        "" => "running".to_string(),
        _ => normalized,
    }
}

fn normalize_runtime_status_token(status: &str) -> String {
    status.trim().to_lowercase().replace([' ', '-'], "_")
}

fn session_runtime_status_is_terminal(status: &str) -> bool {
    matches!(status, "completed" | "failed" | "canceled")
}

fn canonical_terminal_status(status: &str) -> &'static str {
    match status {
        "failed" => "failed",
        "canceled" => "canceled",
        _ => "completed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_turn_keeps_thread_running_without_becoming_active() {
        let state = resolve_session_runtime_state(
            "running",
            0,
            [
                RuntimeTurnSnapshot {
                    turn_id: "turn-completed",
                    status: "completed",
                },
                RuntimeTurnSnapshot {
                    turn_id: "turn-queued",
                    status: "queued",
                },
            ],
            None,
        );

        assert_eq!(state.thread_status, "running");
        assert_eq!(state.latest_turn_status.as_deref(), Some("queued"));
        assert_eq!(state.active_turn_id, None);
        assert_eq!(state.queued_turn_count, 1);
    }

    #[test]
    fn only_execution_owner_keeps_turn_running() {
        let turns = [
            RuntimeTurnSnapshot {
                turn_id: "turn-orphan",
                status: "running",
            },
            RuntimeTurnSnapshot {
                turn_id: "turn-live",
                status: "running",
            },
        ];

        let live = resolve_session_runtime_state("running", 0, turns, Some("turn-live"));
        assert_eq!(live.thread_status, "running");
        assert_eq!(live.latest_turn_status.as_deref(), Some("running"));
        assert_eq!(live.active_turn_id.as_deref(), Some("turn-live"));

        let cold = resolve_session_runtime_state("running", 0, turns, None);
        assert_eq!(cold.thread_status, "idle");
        assert_eq!(cold.latest_turn_status.as_deref(), Some("canceled"));
        assert_eq!(cold.active_turn_id, None);
    }
}
