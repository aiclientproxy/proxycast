use super::status::agent_turn_is_active;
use super::*;
use app_server_protocol::{AgentEvent, RuntimeOptions};
use serde_json::{json, Value};

const AUTO_COMPACT_SOURCE: &str = "turn/start";
const AUTO_COMPACT_EVENT_NAME: &str = "turn/start:autoCompact";
const AUTO_COMPACT_TRIGGER: &str = "auto_context_limit";
const DEFAULT_EFFECTIVE_CONTEXT_WINDOW_PERCENT: u64 = 95;
const AUTO_COMPACT_CONTEXT_WINDOW_RATIO_NUMERATOR: u64 = 9;
const AUTO_COMPACT_CONTEXT_WINDOW_RATIO_DENOMINATOR: u64 = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
struct AutoCompactionDecision {
    active_context_tokens: u64,
    max_tokens: u64,
    remaining_tokens: u64,
    source: String,
    usage_event_sequence: u64,
    usage_turn_id: Option<String>,
}

impl AutoCompactionDecision {
    fn payload(&self) -> Value {
        json!({
            "reason": "context_limit",
            "activeContextTokens": self.active_context_tokens,
            "maxTokens": self.max_tokens,
            "remainingTokens": self.remaining_tokens,
            "source": self.source,
            "usageEventSequence": self.usage_event_sequence,
            "usageTurnId": self.usage_turn_id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ContextBudgetPolicy {
    max_tokens: u64,
    source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UsageSample {
    sequence: u64,
    turn_id: Option<String>,
    input_tokens: u64,
}

impl RuntimeCore {
    pub(in crate::runtime) async fn maybe_auto_compact_before_turn(
        &self,
        session_id: &str,
        runtime_options: Option<&RuntimeOptions>,
    ) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
        let decision = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
            if stored
                .turns
                .iter()
                .any(|turn| agent_turn_is_active(turn.status))
            {
                return Ok(Vec::new());
            }
            auto_compaction_decision(runtime_options, &stored.events)
        };

        let Some(decision) = decision else {
            return Ok(Vec::new());
        };

        let output = self
            .compact_session_with_trigger(
                session_id,
                Some(AUTO_COMPACT_EVENT_NAME),
                AUTO_COMPACT_SOURCE,
                AUTO_COMPACT_TRIGGER,
                Some(decision.payload()),
            )
            .await?;
        Ok(output.events)
    }
}

fn auto_compaction_decision(
    runtime_options: Option<&RuntimeOptions>,
    events: &[AgentEvent],
) -> Option<AutoCompactionDecision> {
    let metadata = runtime_options.and_then(|options| options.runtime_metadata())?;
    if !auto_compact_enabled(metadata) {
        return None;
    }
    let policy = context_budget_policy(metadata)?;
    let usage = latest_usage_after_latest_compaction(events)?;
    if usage.input_tokens < policy.max_tokens {
        return None;
    }
    Some(AutoCompactionDecision {
        active_context_tokens: usage.input_tokens,
        max_tokens: policy.max_tokens,
        remaining_tokens: policy.max_tokens.saturating_sub(usage.input_tokens),
        source: policy.source,
        usage_event_sequence: usage.sequence,
        usage_turn_id: usage.turn_id,
    })
}

fn latest_usage_after_latest_compaction(events: &[AgentEvent]) -> Option<UsageSample> {
    let latest_compaction_sequence = events
        .iter()
        .rev()
        .find(|event| event.event_type == "context.compaction.completed")
        .map(|event| event.sequence)
        .unwrap_or(0);
    events
        .iter()
        .rev()
        .filter(|event| {
            event.event_type == "turn.completed" && event.sequence > latest_compaction_sequence
        })
        .filter_map(|event| {
            let usage = event.payload.get("usage")?;
            let input_tokens = positive_u64_field(
                usage,
                &[
                    "input_tokens",
                    "inputTokens",
                    "active_context_tokens",
                    "activeContextTokens",
                    "accumulated_input_tokens",
                    "accumulatedInputTokens",
                ],
            )?;
            Some(UsageSample {
                sequence: event.sequence,
                turn_id: event.turn_id.clone(),
                input_tokens,
            })
        })
        .next()
}

fn context_budget_policy(metadata: &Value) -> Option<ContextBudgetPolicy> {
    let policy = context_policy_value(metadata)?;
    let context_window = positive_u64_field(policy, &["context_window", "contextWindow"]);
    let max_context_window =
        positive_u64_field(policy, &["max_context_window", "maxContextWindow"]);
    let resolved_context_window = positive_u64_field(
        policy,
        &["resolved_context_window", "resolvedContextWindow"],
    )
    .or(context_window)
    .or(max_context_window);
    let effective_context_window_percent = positive_u64_field(
        policy,
        &[
            "effective_context_window_percent",
            "effectiveContextWindowPercent",
        ],
    )
    .filter(|percent| *percent <= 100)
    .unwrap_or(DEFAULT_EFFECTIVE_CONTEXT_WINDOW_PERCENT);
    let model_context_window =
        positive_u64_field(policy, &["model_context_window", "modelContextWindow"]).or_else(|| {
            resolved_context_window
                .map(|window| window.saturating_mul(effective_context_window_percent) / 100)
        });
    let auto_compact_token_limit = positive_u64_field(
        policy,
        &["auto_compact_token_limit", "autoCompactTokenLimit"],
    )
    .map(|limit| {
        resolved_context_window.map_or(limit, |window| {
            let max_limit = window.saturating_mul(AUTO_COMPACT_CONTEXT_WINDOW_RATIO_NUMERATOR)
                / AUTO_COMPACT_CONTEXT_WINDOW_RATIO_DENOMINATOR;
            limit.min(max_limit)
        })
    })
    .or_else(|| {
        resolved_context_window.map(|window| {
            window.saturating_mul(AUTO_COMPACT_CONTEXT_WINDOW_RATIO_NUMERATOR)
                / AUTO_COMPACT_CONTEXT_WINDOW_RATIO_DENOMINATOR
        })
    });
    let max_tokens = match (model_context_window, auto_compact_token_limit) {
        (Some(model_window), Some(compact_limit)) => Some(model_window.min(compact_limit)),
        (Some(model_window), None) => Some(model_window),
        (None, Some(compact_limit)) => Some(compact_limit),
        (None, None) => None,
    }?;
    let source =
        string_field(policy, &["source"]).unwrap_or_else(|| "model_request_policy".to_string());
    Some(ContextBudgetPolicy { max_tokens, source })
}

fn auto_compact_enabled(metadata: &Value) -> bool {
    [
        metadata.pointer("/lime_runtime/auto_compact"),
        metadata.pointer("/limeRuntime/autoCompact"),
        metadata.get("auto_compact"),
        metadata.get("autoCompact"),
    ]
    .into_iter()
    .flatten()
    .find_map(Value::as_bool)
    .unwrap_or(true)
}

fn context_policy_value(value: &Value) -> Option<&Value> {
    model_request_policy_value(value)
        .and_then(|policy| object_field(policy, &["context_policy", "contextPolicy"]))
        .or_else(|| value.pointer("/lime_runtime/context_policy"))
        .or_else(|| value.pointer("/limeRuntime/contextPolicy"))
}

fn model_request_policy_value(value: &Value) -> Option<&Value> {
    direct_model_request_policy_value(value)
        .or_else(|| nested_metadata_value(value).and_then(model_request_policy_value))
        .or_else(|| {
            [
                "runtime_options",
                "runtimeOptions",
                "runtime_request",
                "runtimeRequest",
                "config",
            ]
            .into_iter()
            .filter_map(|key| value.get(key))
            .find_map(model_request_policy_value)
        })
        .or_else(|| looks_like_policy_value(value).then_some(value))
}

fn direct_model_request_policy_value(value: &Value) -> Option<&Value> {
    value
        .pointer("/request_metadata/harness/model_request_policy")
        .or_else(|| value.pointer("/requestMetadata/harness/modelRequestPolicy"))
        .or_else(|| value.pointer("/harness/model_request_policy"))
        .or_else(|| value.pointer("/harness/modelRequestPolicy"))
        .or_else(|| value.get("model_request_policy"))
        .or_else(|| value.get("modelRequestPolicy"))
}

fn nested_metadata_value(value: &Value) -> Option<&Value> {
    value
        .get("metadata")
        .or_else(|| value.get("request_metadata"))
        .or_else(|| value.get("requestMetadata"))
}

fn looks_like_policy_value(value: &Value) -> bool {
    value.as_object().is_some_and(|object| {
        object.contains_key("context_policy") || object.contains_key("contextPolicy")
    })
}

fn object_field<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find(|value| value.is_object())
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn positive_u64_field(value: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(Value::as_u64)
        .filter(|value| *value > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::AgentEvent;

    fn completed_usage_event(sequence: u64, input_tokens: u64, output_tokens: u64) -> AgentEvent {
        AgentEvent {
            event_id: format!("event_{sequence}"),
            sequence,
            session_id: "sess".to_string(),
            thread_id: Some("thread".to_string()),
            turn_id: Some(format!("turn_{sequence}")),
            event_type: "turn.completed".to_string(),
            timestamp: "2026-07-06T00:00:00.000Z".to_string(),
            payload: json!({
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            }),
        }
    }

    fn completed_compaction_event(sequence: u64) -> AgentEvent {
        AgentEvent {
            event_id: format!("compact_{sequence}"),
            sequence,
            session_id: "sess".to_string(),
            thread_id: Some("thread".to_string()),
            turn_id: None,
            event_type: "context.compaction.completed".to_string(),
            timestamp: "2026-07-06T00:00:00.000Z".to_string(),
            payload: json!({ "trigger": "auto_context_limit" }),
        }
    }

    fn runtime_options_with_context_limit(limit: u64) -> RuntimeOptions {
        RuntimeOptions {
            runtime_request: Some(app_server_protocol::RuntimeRequest {
                metadata: Some(json!({
                    "request_metadata": {
                        "harness": {
                            "model_request_policy": {
                                "context_policy": {
                                    "model_context_window": 120_000,
                                    "auto_compact_token_limit": limit
                                }
                            }
                        }
                    }
                })),
                ..app_server_protocol::RuntimeRequest::default()
            }),
            ..RuntimeOptions::default()
        }
    }

    #[test]
    fn decision_uses_input_tokens_not_output_tokens() {
        let options = runtime_options_with_context_limit(90_000);
        let events = vec![completed_usage_event(1, 89_999, 200_000)];

        assert_eq!(auto_compaction_decision(Some(&options), &events), None);
    }

    #[test]
    fn decision_ignores_usage_before_latest_compaction() {
        let options = runtime_options_with_context_limit(90_000);
        let events = vec![
            completed_usage_event(1, 91_000, 0),
            completed_compaction_event(2),
        ];

        assert_eq!(auto_compaction_decision(Some(&options), &events), None);
    }
}
