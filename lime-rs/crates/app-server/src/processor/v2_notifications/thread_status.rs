use agent_protocol::action_required::{ASK_USER_ACTION_TYPE, TOOL_CONFIRMATION_ACTION_TYPE};
use app_server_protocol::protocol::v2::{
    ServerNotification, ThreadActiveFlag, ThreadStatus, ThreadStatusChangedNotification,
};
use app_server_protocol::AgentEvent;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Default)]
pub(super) struct ThreadStatusProjector {
    runtime_by_thread_id: HashMap<String, RuntimeFacts>,
}

impl ThreadStatusProjector {
    pub(super) fn note_thread_started(&mut self, thread_id: &str) {
        self.runtime_by_thread_id
            .entry(thread_id.to_string())
            .or_default();
    }

    pub(super) fn project_turn_started(&mut self, thread_id: &str) -> Option<ServerNotification> {
        self.update(thread_id, |runtime| runtime.running = true)
    }

    pub(super) fn project_turn_terminal(&mut self, thread_id: &str) -> Option<ServerNotification> {
        self.update_existing(thread_id, |runtime| {
            runtime.running = false;
            runtime.pending_actions.clear();
        })
    }

    pub(super) fn project_action_required(
        &mut self,
        event: &AgentEvent,
    ) -> Option<ServerNotification> {
        let active_flag =
            match payload_string(&event.payload, &["actionType", "action_type"])?.as_str() {
                TOOL_CONFIRMATION_ACTION_TYPE => ThreadActiveFlag::WaitingOnApproval,
                ASK_USER_ACTION_TYPE => ThreadActiveFlag::WaitingOnUserInput,
                _ => return None,
            };
        let thread_id = event_id(event.thread_id.as_deref())?;
        let request_id = payload_string(
            &event.payload,
            &["requestId", "request_id", "actionId", "action_id"],
        )?;
        self.update(thread_id, |runtime| {
            runtime.running = true;
            runtime.pending_actions.insert(request_id, active_flag);
        })
    }

    pub(super) fn project_action_terminal(
        &mut self,
        event: &AgentEvent,
    ) -> Option<ServerNotification> {
        let thread_id = event_id(event.thread_id.as_deref())?;
        let request_id = payload_string(
            &event.payload,
            &["requestId", "request_id", "actionId", "action_id"],
        )?;
        self.update_existing(thread_id, |runtime| {
            runtime.pending_actions.remove(&request_id);
        })
    }

    fn update(
        &mut self,
        thread_id: &str,
        mutate: impl FnOnce(&mut RuntimeFacts),
    ) -> Option<ServerNotification> {
        let previous = self.status_for(thread_id);
        mutate(
            self.runtime_by_thread_id
                .entry(thread_id.to_string())
                .or_default(),
        );
        self.changed_notification(thread_id, previous)
    }

    fn update_existing(
        &mut self,
        thread_id: &str,
        mutate: impl FnOnce(&mut RuntimeFacts),
    ) -> Option<ServerNotification> {
        let previous = self.status_for(thread_id)?;
        mutate(self.runtime_by_thread_id.get_mut(thread_id)?);
        self.changed_notification(thread_id, Some(previous))
    }

    fn changed_notification(
        &self,
        thread_id: &str,
        previous: Option<ThreadStatus>,
    ) -> Option<ServerNotification> {
        let status = self.status_for(thread_id)?;
        if previous.as_ref() == Some(&status) {
            return None;
        }
        Some(ServerNotification::ThreadStatusChanged(
            ThreadStatusChangedNotification {
                thread_id: thread_id.to_string(),
                status,
            },
        ))
    }

    fn status_for(&self, thread_id: &str) -> Option<ThreadStatus> {
        self.runtime_by_thread_id
            .get(thread_id)
            .map(RuntimeFacts::status)
    }
}

#[derive(Default)]
struct RuntimeFacts {
    running: bool,
    pending_actions: HashMap<String, ThreadActiveFlag>,
}

impl RuntimeFacts {
    fn status(&self) -> ThreadStatus {
        let mut active_flags = Vec::with_capacity(2);
        if self
            .pending_actions
            .values()
            .any(|flag| *flag == ThreadActiveFlag::WaitingOnApproval)
        {
            active_flags.push(ThreadActiveFlag::WaitingOnApproval);
        }
        if self
            .pending_actions
            .values()
            .any(|flag| *flag == ThreadActiveFlag::WaitingOnUserInput)
        {
            active_flags.push(ThreadActiveFlag::WaitingOnUserInput);
        }
        if self.running || !active_flags.is_empty() {
            ThreadStatus::Active { active_flags }
        } else {
            ThreadStatus::Idle
        }
    }
}

fn event_id(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn payload_string(payload: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| payload.get(*key))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(event_type: &str, payload: Value) -> AgentEvent {
        AgentEvent {
            event_id: format!("evt-{event_type}"),
            sequence: 1,
            session_id: "session-1".to_string(),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            event_type: event_type.to_string(),
            timestamp: "2026-07-26T00:00:00Z".to_string(),
            payload,
        }
    }

    fn status(notification: Option<ServerNotification>) -> Option<ThreadStatus> {
        match notification {
            Some(ServerNotification::ThreadStatusChanged(notification)) => {
                Some(notification.status)
            }
            None => None,
            Some(_) => panic!("unexpected notification"),
        }
    }

    #[test]
    fn tracks_turn_and_both_pending_action_flags() {
        let mut projector = ThreadStatusProjector::default();
        projector.note_thread_started("thread-1");
        assert_eq!(
            status(projector.project_turn_started("thread-1")),
            Some(ThreadStatus::Active {
                active_flags: vec![]
            })
        );
        assert_eq!(
            status(projector.project_action_required(&event(
                "action.required",
                json!({"requestId": "approval-1", "actionType": "tool_confirmation"}),
            ))),
            Some(ThreadStatus::Active {
                active_flags: vec![ThreadActiveFlag::WaitingOnApproval]
            })
        );
        assert_eq!(
            status(projector.project_action_required(&event(
                "action.required",
                json!({"requestId": "input-1", "actionType": "ask_user"}),
            ))),
            Some(ThreadStatus::Active {
                active_flags: vec![
                    ThreadActiveFlag::WaitingOnApproval,
                    ThreadActiveFlag::WaitingOnUserInput,
                ]
            })
        );
        assert_eq!(
            status(projector.project_action_terminal(&event(
                "action.resolved",
                json!({"requestId": "approval-1"}),
            ))),
            Some(ThreadStatus::Active {
                active_flags: vec![ThreadActiveFlag::WaitingOnUserInput]
            })
        );
        assert_eq!(
            status(projector.project_action_terminal(&event(
                "action.canceled",
                json!({"requestId": "input-1"}),
            ))),
            Some(ThreadStatus::Active {
                active_flags: vec![]
            })
        );
        assert_eq!(
            status(projector.project_turn_terminal("thread-1")),
            Some(ThreadStatus::Idle)
        );
    }

    #[test]
    fn malformed_and_duplicate_events_do_not_mutate_or_repeat_status() {
        let mut projector = ThreadStatusProjector::default();
        projector.note_thread_started("thread-1");
        assert!(projector
            .project_action_required(&event("action.required", json!({})))
            .is_none());
        assert!(projector.project_turn_terminal("thread-1").is_none());

        let required = event(
            "action.required",
            json!({"requestId": "approval-1", "actionType": "tool_confirmation"}),
        );
        assert!(projector.project_action_required(&required).is_some());
        assert!(projector.project_action_required(&required).is_none());
        assert!(projector
            .project_action_terminal(&event("action.resolved", json!({"requestId": "unknown"}),))
            .is_none());
    }
}
