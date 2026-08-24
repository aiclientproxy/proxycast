use app_server_protocol::AgentEvent;

pub(super) const HISTORY_ROLLBACK_EVENT_TYPE: &str = "history.rollback";

pub(super) fn rollback_target(event: &AgentEvent) -> Option<u64> {
    (event.event_type == HISTORY_ROLLBACK_EVENT_TYPE)
        .then(|| {
            event
                .payload
                .get("rollbackToSequence")
                .or_else(|| event.payload.get("rollback_to_sequence"))
                .and_then(serde_json::Value::as_u64)
        })
        .flatten()
}

/// Resolve append-only replacement markers without rewriting the immutable event log.
pub(super) fn effective_events(events: &[AgentEvent]) -> Vec<AgentEvent> {
    let mut effective = Vec::with_capacity(events.len());
    for event in events {
        if let Some(target) = rollback_target(event) {
            effective.retain(|candidate: &AgentEvent| candidate.sequence <= target);
        }
        effective.push(event.clone());
    }
    effective
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(sequence: u64, event_type: &str, rollback_to_sequence: Option<u64>) -> AgentEvent {
        AgentEvent {
            event_id: format!("event-{sequence}"),
            sequence,
            session_id: "session-1".to_string(),
            thread_id: Some("thread-1".to_string()),
            turn_id: None,
            event_type: event_type.to_string(),
            timestamp: "2026-08-24T00:00:00Z".to_string(),
            payload: rollback_to_sequence
                .map(|target| json!({"rollbackToSequence": target}))
                .unwrap_or_else(|| json!({})),
        }
    }

    #[test]
    fn repeated_replacements_keep_only_the_latest_effective_lineage() {
        let events = vec![
            event(1, "turn.started", None),
            event(2, "turn.completed", None),
            event(3, "turn.started", None),
            event(4, "turn.completed", None),
            event(5, HISTORY_ROLLBACK_EVENT_TYPE, Some(2)),
            event(6, "turn.started", None),
            event(7, "turn.completed", None),
            event(8, HISTORY_ROLLBACK_EVENT_TYPE, Some(2)),
            event(9, "turn.started", None),
        ];

        assert_eq!(
            effective_events(&events)
                .into_iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 8, 9]
        );
    }

    #[test]
    fn malformed_replacement_marker_never_discards_history() {
        let events = vec![
            event(1, "turn.started", None),
            event(2, HISTORY_ROLLBACK_EVENT_TYPE, None),
        ];

        assert_eq!(effective_events(&events), events);
    }
}
