use super::{projection_error, required_event_id, EventProjection};
use app_server_protocol::protocol::v2::{self, ServerNotification};
use app_server_protocol::AgentEvent;
use std::collections::HashSet;

pub(super) fn project_started(
    started_run_ids: &mut HashSet<String>,
    event: &AgentEvent,
) -> EventProjection {
    let Some(thread_id) = required_event_id(event.thread_id.as_deref()) else {
        return EventProjection::Reject(projection_error(event));
    };
    let run = match hook_run(event) {
        Some(run) => run,
        None => return EventProjection::Reject(projection_error(event)),
    };
    if run.status != agent_protocol::hook::HookRunStatus::Running {
        return EventProjection::Reject(projection_error(event));
    }
    if !started_run_ids.insert(run.id.clone()) {
        return EventProjection::Direct(Vec::new());
    }
    EventProjection::Direct(vec![ServerNotification::HookStarted(
        v2::HookStartedNotification {
            thread_id,
            turn_id: required_event_id(event.turn_id.as_deref()),
            run: (&run).into(),
        },
    )
    .into()])
}

pub(super) fn project_completed(
    started_run_ids: &HashSet<String>,
    completed_run_ids: &mut HashSet<String>,
    event: &AgentEvent,
) -> EventProjection {
    let Some(thread_id) = required_event_id(event.thread_id.as_deref()) else {
        return EventProjection::Reject(projection_error(event));
    };
    let run = match hook_run(event) {
        Some(run) => run,
        None => return EventProjection::Reject(projection_error(event)),
    };
    if run.status == agent_protocol::hook::HookRunStatus::Running
        || !started_run_ids.contains(&run.id)
    {
        return EventProjection::Reject(projection_error(event));
    }
    if !completed_run_ids.insert(run.id.clone()) {
        return EventProjection::Direct(Vec::new());
    }
    EventProjection::Direct(vec![ServerNotification::HookCompleted(
        v2::HookCompletedNotification {
            thread_id,
            turn_id: required_event_id(event.turn_id.as_deref()),
            run: (&run).into(),
        },
    )
    .into()])
}

fn hook_run(event: &AgentEvent) -> Option<agent_protocol::hook::HookRunSummary> {
    serde_json::from_value::<agent_protocol::hook::HookRunSummary>(
        event.payload.get("run")?.clone(),
    )
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_protocol::hook::{
        HookEventName, HookExecutionMode, HookHandlerType, HookRunStatus, HookScope, HookSource,
    };
    use serde_json::json;
    use std::path::PathBuf;

    fn event(event_type: &str, status: HookRunStatus) -> AgentEvent {
        let run = agent_protocol::hook::HookRunSummary {
            id: "run-1".to_string(),
            event_name: HookEventName::PreToolUse,
            handler_type: HookHandlerType::Command,
            execution_mode: HookExecutionMode::Sync,
            scope: HookScope::Turn,
            source_path: PathBuf::from("/tmp/hook.sh"),
            source: HookSource::Project,
            display_order: 0,
            status,
            status_message: None,
            started_at: 1,
            completed_at: (status != HookRunStatus::Running).then_some(2),
            duration_ms: (status != HookRunStatus::Running).then_some(1),
            entries: Vec::new(),
        };
        AgentEvent {
            event_id: event_type.to_string(),
            sequence: 1,
            session_id: "session-1".to_string(),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            event_type: event_type.to_string(),
            timestamp: "2026-08-06T00:00:00Z".to_string(),
            payload: json!({"run": run}),
        }
    }

    #[test]
    fn projects_paired_hook_lifecycle_with_same_run_id() {
        let mut started = HashSet::new();
        let mut completed = HashSet::new();
        assert!(matches!(
            project_started(&mut started, &event("hook.started", HookRunStatus::Running)),
            EventProjection::Direct(_)
        ));
        assert!(matches!(
            project_completed(
                &started,
                &mut completed,
                &event("hook.completed", HookRunStatus::Completed)
            ),
            EventProjection::Direct(_)
        ));
        assert!(matches!(
            project_completed(
                &started,
                &mut completed,
                &event("hook.completed", HookRunStatus::Completed)
            ),
            EventProjection::Direct(notifications) if notifications.is_empty()
        ));
    }
}
