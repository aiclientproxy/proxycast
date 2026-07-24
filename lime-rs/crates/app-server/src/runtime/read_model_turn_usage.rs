use super::thread_usage::thread_token_usage_snapshot_from_events;
use app_server_protocol::{AgentEvent, AgentTurn};
use serde_json::{json, Value};

pub(super) fn turns_with_usage(turns: &[AgentTurn], events: &[AgentEvent]) -> Vec<Value> {
    turns
        .iter()
        .map(|turn| {
            let mut value = serde_json::to_value(turn).unwrap_or_else(|_| {
                json!({
                    "turnId": turn.turn_id,
                    "sessionId": turn.session_id,
                    "threadId": turn.thread_id,
                    "status": turn.status,
                    "startedAt": turn.started_at,
                    "completedAt": turn.completed_at,
                })
            });
            if let Some(usage) = usage_for_turn(events, &turn.turn_id) {
                if let Some(record) = value.as_object_mut() {
                    record.insert("usage".to_string(), usage);
                }
            }
            value
        })
        .collect()
}

pub(super) fn latest_usage_for_turn(events: &[AgentEvent], turn_id: Option<&str>) -> Option<Value> {
    canonical_usage_for_turn(events, turn_id).or_else(|| {
        events
            .iter()
            .rev()
            .filter(|event| event.event_type == "turn.completed")
            .filter(|event| match turn_id {
                Some(turn_id) => event.turn_id.as_deref() == Some(turn_id),
                None => true,
            })
            .find_map(|event| normalized_usage(&event.payload))
    })
}

fn usage_for_turn(events: &[AgentEvent], turn_id: &str) -> Option<Value> {
    latest_usage_for_turn(events, Some(turn_id))
}

fn canonical_usage_for_turn(events: &[AgentEvent], turn_id: Option<&str>) -> Option<Value> {
    let turn_id = turn_id.map(str::trim).filter(|value| !value.is_empty());
    let turn_events = events
        .iter()
        .filter(|event| turn_id.is_none_or(|turn_id| event.turn_id.as_deref() == Some(turn_id)))
        .cloned()
        .collect::<Vec<_>>();
    let snapshot = thread_token_usage_snapshot_from_events(&turn_events)?;
    Some(json!({
        "input_tokens": snapshot.last_token_usage.input_tokens,
        "cached_input_tokens": snapshot.last_token_usage.cached_input_tokens,
        "cache_write_input_tokens": snapshot.last_token_usage.cache_write_input_tokens,
        "output_tokens": snapshot.last_token_usage.output_tokens,
        "reasoning_output_tokens": snapshot.last_token_usage.reasoning_output_tokens,
        "total_tokens": snapshot.last_token_usage.total_tokens,
    }))
}

fn normalized_usage(payload: &Value) -> Option<Value> {
    let usage = payload.get("usage")?;
    if !usage.is_object() {
        return None;
    }
    Some(usage.clone())
}

#[cfg(test)]
mod tests {
    use super::latest_usage_for_turn;
    use app_server_protocol::AgentEvent;
    use serde_json::json;

    fn event(
        sequence: u64,
        turn_id: &str,
        event_type: &str,
        payload: serde_json::Value,
    ) -> AgentEvent {
        AgentEvent {
            event_id: format!("event-{sequence}"),
            sequence,
            session_id: "session-usage".to_string(),
            thread_id: Some("thread-usage".to_string()),
            turn_id: Some(turn_id.to_string()),
            event_type: event_type.to_string(),
            timestamp: "2026-07-20T08:00:00Z".to_string(),
            payload,
        }
    }

    fn provider_usage(
        sequence: u64,
        turn_id: &str,
        input: i64,
        cache_write: i64,
        output: i64,
    ) -> AgentEvent {
        event(
            sequence,
            turn_id,
            "provider.usage",
            json!({
                "backend": "runtime",
                "attempt": 0,
                "usage": {
                    "input_tokens": input,
                    "output_tokens": output,
                    "cached_input_tokens": 0,
                    "cache_write_input_tokens": cache_write,
                    "reasoning_output_tokens": 0,
                    "total_tokens": input + output,
                    "total_token_usage": {
                        "input_tokens": input,
                        "cached_input_tokens": 0,
                        "cache_write_input_tokens": cache_write,
                        "output_tokens": output,
                        "reasoning_output_tokens": 0,
                        "total_tokens": input + output
                    },
                    "last_token_usage": {
                        "input_tokens": input,
                        "cached_input_tokens": 0,
                        "cache_write_input_tokens": cache_write,
                        "output_tokens": output,
                        "reasoning_output_tokens": 0,
                        "total_tokens": input + output
                    },
                    "model_context_window": 128000
                }
            }),
        )
    }

    #[test]
    fn provider_usage_survives_terminal_without_usage() {
        let events = vec![
            provider_usage(1, "turn-image", 31_000, 1_200, 0),
            event(2, "turn-image", "turn.completed", json!({})),
        ];

        assert_eq!(
            latest_usage_for_turn(&events, Some("turn-image")),
            Some(json!({
                "input_tokens": 31_000,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 1_200,
                "output_tokens": 0,
                "reasoning_output_tokens": 0,
                "total_tokens": 31_000
            }))
        );
    }

    #[test]
    fn usage_is_scoped_to_the_requested_turn() {
        let events = vec![
            provider_usage(1, "turn-first", 10, 1, 2),
            provider_usage(2, "turn-second", 20, 2, 3),
        ];

        assert_eq!(
            latest_usage_for_turn(&events, Some("turn-first"))
                .and_then(|usage| usage.get("input_tokens").cloned()),
            Some(json!(10))
        );
        assert_eq!(
            latest_usage_for_turn(&events, Some("turn-second"))
                .and_then(|usage| usage.get("input_tokens").cloned()),
            Some(json!(20))
        );
    }
}
