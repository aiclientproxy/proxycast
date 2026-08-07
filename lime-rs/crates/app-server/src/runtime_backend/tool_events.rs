use super::plan_events;
use super::tool_process_metadata::SoulStyleMetadata;
use super::tool_process_runtime_metadata;
use crate::RuntimeCoreError;
use crate::RuntimeEvent;
use agent_protocol::provider_trace::runtime_event_type_for_provider_trace_stage;
#[cfg(test)]
use agent_protocol::provider_trace::{ProviderTraceEvent, ProviderTraceStage};
use agent_protocol::{
    ItemId, ItemStatus, SessionId, ThreadId, ThreadItem, ThreadItemPayload, ToolOutput, TurnId,
};
use lime_agent::AgentEvent as RuntimeAgentEvent;
use model_provider::safety::SAFETY_BUFFERING_RUNTIME_EVENT_KIND;
use serde_json::{json, Value};

#[cfg(test)]
pub(super) fn runtime_events_from_agent_event(
    event: &RuntimeAgentEvent,
) -> Result<Vec<RuntimeEvent>, RuntimeCoreError> {
    runtime_events_from_agent_event_with_soul_style(event, None)
}

pub(super) fn runtime_events_from_agent_event_with_soul_style(
    event: &RuntimeAgentEvent,
    soul_style: Option<&SoulStyleMetadata>,
) -> Result<Vec<RuntimeEvent>, RuntimeCoreError> {
    if let Some(events) = hook_runtime_events(event) {
        return Ok(events);
    }
    if let RuntimeAgentEvent::ProviderStreamEvent {
        runtime_event_kind,
        payload,
    } = event
    {
        let mut payload = payload.clone();
        if let Some(payload_object) = payload.as_object_mut() {
            payload_object.insert("backend".to_string(), Value::String("runtime".to_string()));
            payload_object.insert(
                "runtimeEvent".to_string(),
                serde_json::to_value(event).map_err(event_error)?,
            );
        }
        return Ok(vec![RuntimeEvent::new(
            provider_stream_runtime_event_type(runtime_event_kind),
            payload,
        )]);
    }

    let runtime_event = serde_json::to_value(event).map_err(event_error)?;
    let raw_type = runtime_event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("runtime_event")
        .to_string();
    let mut payload = runtime_event
        .as_object()
        .cloned()
        .map(Value::Object)
        .unwrap_or_else(|| json!({ "value": runtime_event.clone() }));
    if let Some(payload_object) = payload.as_object_mut() {
        payload_object.insert("backend".to_string(), Value::String("runtime".to_string()));
        payload_object.insert("runtimeEvent".to_string(), runtime_event);
        tool_process_runtime_metadata::enrich_runtime_tool_process_payload(
            event,
            payload_object,
            soul_style,
        );
        enrich_reasoning_payload(event, payload_object);
    }
    let mut events = vec![RuntimeEvent::new(
        runtime_event_type_for_agent_event(event, &raw_type),
        payload,
    )];
    if let Some(plan_event) = update_plan_event_from_completed_tool(event) {
        events.push(plan_event);
    }
    Ok(events)
}

fn hook_runtime_events(event: &RuntimeAgentEvent) -> Option<Vec<RuntimeEvent>> {
    let (event_type, status, run) = match event {
        RuntimeAgentEvent::HookStarted { run, .. } => ("hook.started", ItemStatus::InProgress, run),
        RuntimeAgentEvent::HookCompleted { run, .. } => {
            let status = match run.status {
                agent_protocol::hook::HookRunStatus::Failed
                | agent_protocol::hook::HookRunStatus::Blocked => ItemStatus::Failed,
                agent_protocol::hook::HookRunStatus::Stopped => ItemStatus::Interrupted,
                _ => ItemStatus::Completed,
            };
            ("hook.completed", status, run)
        }
        _ => return None,
    };
    let mut item = ThreadItem::new(
        SessionId::new("hook-session"),
        ThreadId::new("hook-thread"),
        TurnId::new("hook-turn"),
        0,
        0,
        ThreadItemPayload::Hook { run: run.clone() },
    );
    item.item_id = ItemId::new(format!("item_{}", run.id));
    item.status = status;
    item.created_at_ms = run.started_at;
    item.updated_at_ms = run.completed_at.unwrap_or(run.started_at);
    item.completed_at_ms = status.is_terminal().then_some(item.updated_at_ms);
    item.metadata = serde_json::json!({
        "hookRunId": run.id,
        "source": "codex_hook_runtime",
    });
    Some(vec![RuntimeEvent::new(
        event_type,
        serde_json::json!({
            "run": run,
            "item": item,
        }),
    )])
}

pub(super) fn runtime_event_type_from_raw(raw_type: &str) -> &'static str {
    match raw_type {
        "thread_started" => "thread.started",
        "turn_started" => "turn.started",
        "turn_completed" => "turn.completed",
        "turn_failed" => "turn.failed",
        "item_started" => "item.started",
        "item_updated" => "item.updated",
        "item_completed" => "item.completed",
        "hook_started" => "hook.started",
        "hook_completed" => "hook.completed",
        "text_delta" => "message.delta",
        "text_delta_batch" => "message.delta_batch",
        "reasoning_summary_delta" => "reasoning.summary",
        "reasoning_summary_part_added" => "reasoning.summary_part_added",
        "reasoning_content_delta" => "reasoning.delta",
        "tool_progress" => "tool.progress",
        "tool_output_delta" => "tool.output.delta",
        "tool_input_delta" => "tool.input.delta",
        "artifact_snapshot" => "artifact.snapshot",
        "action_required" => "action.required",
        "action_resolved" => "action.resolved",
        "dynamic_tool_call_requested" => "dynamic_tool.requested",
        "turn_context" => "turn.context",
        "model_change" => "model.changed",
        "server_model" => "model.server_reported",
        "model_reroute" => "model.rerouted",
        "model_verification" => "model.verification",
        "provider_trace" => "provider.trace",
        "provider_usage" => "provider.usage",
        "provider_step" => "provider.step",
        "provider_stream_event" => "runtime.event",
        "context_trace" => "context.trace",
        "context_compaction_started" => "context.compaction.started",
        "context_compaction_completed" => "context.compaction.completed",
        "runtime_status" => "runtime.status",
        "task_profile_resolved" => "task.profile.resolved",
        "candidate_set_resolved" => "routing.candidates.resolved",
        "routing_decision_made" => "routing.decision.made",
        "routing_fallback_applied" => "routing.fallback.applied",
        "routing_not_possible" => "routing.not_possible",
        "limit_state_updated" => "limit.state.updated",
        "single_candidate_only" => "limit.single_candidate_only",
        "single_candidate_capability_gap" => "limit.single_candidate_capability_gap",
        "cost_estimated" => "cost.estimated",
        "cost_recorded" => "cost.recorded",
        "rate_limit_hit" => "rate_limit.hit",
        "quota_low" => "quota.low",
        "quota_blocked" => "quota.blocked",
        "queue_added" => "queue.added",
        "queue_removed" => "queue.removed",
        "queue_started" => "queue.started",
        "queue_cleared" => "queue.cleared",
        "error" => "turn.failed",
        "warning" => "runtime.warning",
        "message" => "message",
        _ => "runtime.event",
    }
}

fn enrich_reasoning_payload(
    event: &RuntimeAgentEvent,
    payload_object: &mut serde_json::Map<String, Value>,
) {
    let item_id = match event {
        RuntimeAgentEvent::ReasoningSummaryDelta {
            item_id,
            text,
            summary_index,
        } => {
            payload_object.remove("text");
            payload_object.insert("summary".to_string(), Value::String(text.clone()));
            payload_object.insert("summaryIndex".to_string(), Value::from(*summary_index));
            item_id
        }
        RuntimeAgentEvent::ReasoningSummaryPartAdded {
            item_id,
            summary_index,
        } => {
            payload_object.insert("summaryIndex".to_string(), Value::from(*summary_index));
            item_id
        }
        RuntimeAgentEvent::ReasoningContentDelta {
            item_id,
            text,
            content_index,
        } => {
            payload_object.remove("text");
            payload_object.insert("delta".to_string(), Value::String(text.clone()));
            payload_object.insert("contentIndex".to_string(), Value::from(*content_index));
            item_id
        }
        _ => return,
    };
    payload_object.insert("reasoningId".to_string(), Value::String(item_id.clone()));
    payload_object.insert("itemId".to_string(), Value::String(item_id.clone()));
}

fn update_plan_event_from_completed_tool(event: &RuntimeAgentEvent) -> Option<RuntimeEvent> {
    let RuntimeAgentEvent::ItemCompleted { .. } = event else {
        return None;
    };
    let tool = canonical_tool_item(event)?;
    let output = tool.output?;
    if tool.name != tool_runtime::update_plan::UPDATE_PLAN_NAME
        || !tool.item.status.is_terminal()
        || output.error.is_some()
    {
        return None;
    }
    plan_events::turn_plan_updated_event_from_update_plan_result(tool.call_id, output)
}

fn runtime_event_type_for_agent_event(event: &RuntimeAgentEvent, raw_type: &str) -> &'static str {
    match event {
        RuntimeAgentEvent::ProviderTrace { event } => {
            runtime_event_type_for_provider_trace_stage(event.stage)
        }
        _ => runtime_event_type_from_raw(raw_type),
    }
}

fn provider_stream_runtime_event_type(runtime_event_kind: &str) -> &'static str {
    match runtime_event_kind {
        SAFETY_BUFFERING_RUNTIME_EVENT_KIND => SAFETY_BUFFERING_RUNTIME_EVENT_KIND,
        _ => "runtime.event",
    }
}

struct CanonicalToolItem<'a> {
    item: &'a ThreadItem,
    call_id: &'a str,
    name: &'a str,
    output: Option<&'a ToolOutput>,
}

fn canonical_tool_item(event: &RuntimeAgentEvent) -> Option<CanonicalToolItem<'_>> {
    let item = match event {
        RuntimeAgentEvent::ItemStarted { item }
        | RuntimeAgentEvent::ItemUpdated { item }
        | RuntimeAgentEvent::ItemCompleted { item } => item,
        _ => return None,
    };
    let ThreadItemPayload::Tool {
        call_id,
        name,
        output,
        ..
    } = &item.payload
    else {
        return None;
    };
    Some(CanonicalToolItem {
        item,
        call_id,
        name,
        output: output.as_ref(),
    })
}

fn event_error(error: impl std::fmt::Display) -> RuntimeCoreError {
    RuntimeCoreError::Backend(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_protocol::{ItemId, ItemStatus, SessionId, ThreadId, ToolArgument, TurnId};
    use lime_agent::AgentToolResult;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn final_done_raw_runtime_event_does_not_map_to_current_terminal_event() {
        assert_eq!(runtime_event_type_from_raw("final_done"), "runtime.event");
    }

    #[test]
    fn runtime_status_raw_runtime_event_maps_to_non_terminal_status_event() {
        assert_eq!(
            runtime_event_type_from_raw("runtime_status"),
            "runtime.status"
        );
    }

    #[test]
    fn model_facts_map_to_canonical_runtime_event_types() {
        let server_model = runtime_events_from_agent_event(&RuntimeAgentEvent::ServerModel {
            model: "gpt-5-codex".to_string(),
        })
        .expect("server model event");
        assert_eq!(server_model[0].event_type, "model.server_reported");
        assert_eq!(server_model[0].payload["model"], "gpt-5-codex");

        let reroute = runtime_events_from_agent_event(&RuntimeAgentEvent::ModelReroute {
            from_model: "gpt-5-codex".to_string(),
            to_model: "gpt-5.1-codex".to_string(),
            reason: model_provider::current_client::ModelRerouteReason::HighRiskCyberActivity,
        })
        .expect("model reroute event");
        assert_eq!(reroute[0].event_type, "model.rerouted");
        assert_eq!(reroute[0].payload["from_model"], "gpt-5-codex");
        assert_eq!(reroute[0].payload["to_model"], "gpt-5.1-codex");
        assert_eq!(reroute[0].payload["reason"], "high_risk_cyber_activity");

        let verification = runtime_events_from_agent_event(&RuntimeAgentEvent::ModelVerification {
            verifications: vec![
                model_provider::current_client::ModelVerification::TrustedAccessForCyber,
            ],
        })
        .expect("model verification event");
        assert_eq!(verification[0].event_type, "model.verification");
        assert_eq!(
            verification[0].payload["verifications"],
            json!(["trusted_access_for_cyber"])
        );
    }

    #[test]
    fn provider_step_maps_usage_and_output_summary_to_current_event() {
        let events = runtime_events_from_agent_event(&RuntimeAgentEvent::ProviderStep {
            attempt: 2,
            completed: true,
            finish_reason: Some("tool_call".to_string()),
            text_output_chars: 7,
            reasoning_output_chars: 42,
            tool_call_count: 1,
            usage: Some(lime_agent::AgentTokenUsage {
                input_tokens: 120,
                output_tokens: 30,
                cached_input_tokens: Some(20),
                cache_creation_input_tokens: None,
            }),
        })
        .expect("provider step should emit");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "provider.step");
        assert_eq!(events[0].payload["attempt"], 2);
        assert_eq!(events[0].payload["tool_call_count"], 1);
        assert_eq!(events[0].payload["usage"]["input_tokens"], 120);
        assert_eq!(events[0].payload["runtimeEvent"]["type"], "provider_step");
    }

    #[test]
    fn provider_usage_maps_attempt_snapshot_to_durable_runtime_event() {
        let events = runtime_events_from_agent_event(&RuntimeAgentEvent::ProviderUsage {
            attempt: 3,
            usage: lime_agent::AgentTokenUsage {
                input_tokens: 120,
                output_tokens: 30,
                cached_input_tokens: Some(20),
                cache_creation_input_tokens: None,
            },
        })
        .expect("provider usage should emit");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "provider.usage");
        assert_eq!(events[0].payload["backend"], "runtime");
        assert_eq!(events[0].payload["attempt"], 3);
        assert_eq!(events[0].payload["usage"]["input_tokens"], 120);
        assert_eq!(events[0].payload["runtimeEvent"]["type"], "provider_usage");
    }

    #[test]
    fn reasoning_summary_delta_maps_to_standard_reasoning_summary_event() {
        let events = runtime_events_from_agent_event(&RuntimeAgentEvent::ReasoningSummaryDelta {
            item_id: "reasoning-1".to_string(),
            text: "先理解目标".to_string(),
            summary_index: 2,
        })
        .expect("reasoning summary delta should emit");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "reasoning.summary");
        assert_eq!(events[0].payload["summary"], "先理解目标");
        assert_eq!(events[0].payload["summaryIndex"], 2);
        assert!(events[0].payload.get("text").is_none());
        assert_eq!(events[0].payload["reasoningId"], "reasoning-1");
        assert_eq!(events[0].payload["itemId"], "reasoning-1");
    }

    #[test]
    fn reasoning_content_delta_maps_to_standard_reasoning_delta_event() {
        let events = runtime_events_from_agent_event(&RuntimeAgentEvent::ReasoningContentDelta {
            item_id: "reasoning-1".to_string(),
            text: "原始推理".to_string(),
            content_index: 3,
        })
        .expect("reasoning content delta should emit");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "reasoning.delta");
        assert_eq!(events[0].payload["delta"], "原始推理");
        assert_eq!(events[0].payload["contentIndex"], 3);
        assert!(events[0].payload.get("text").is_none());
        assert_eq!(events[0].payload["reasoningId"], "reasoning-1");
        assert_eq!(events[0].payload["itemId"], "reasoning-1");
    }

    #[test]
    fn reasoning_summary_part_added_maps_to_distinct_runtime_event() {
        let events =
            runtime_events_from_agent_event(&RuntimeAgentEvent::ReasoningSummaryPartAdded {
                item_id: "reasoning-1".to_string(),
                summary_index: 4,
            })
            .expect("reasoning summary part should emit");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "reasoning.summary_part_added");
        assert_eq!(events[0].payload["summaryIndex"], 4);
        assert_eq!(events[0].payload["itemId"], "reasoning-1");
    }

    #[test]
    fn provider_trace_stage_maps_to_provider_runtime_event() {
        let events = runtime_events_from_agent_event(&RuntimeAgentEvent::ProviderTrace {
            event: ProviderTraceEvent {
                stage: ProviderTraceStage::FirstTextDeltaReceived,
                provider: "openai".to_string(),
                model: "gpt-4.1".to_string(),
                attempt: 1,
                elapsed_ms: Some(1234),
                text_chars: Some(8),
                status: "running".to_string(),
                failure_category: None,
                retryable: None,
                non_retryable_provider_rejection: None,
                cancel_reason: None,
                provider_request_id: Some("req-provider-1".to_string()),
                provider_request_id_header: Some("x-request-id".to_string()),
                runtime_provider_backend: Some("current".to_string()),
                runtime_provider_selector: Some("codex".to_string()),
                runtime_provider_protocol: Some("responses".to_string()),
                runtime_provider_active_model: Some("gpt-4.1".to_string()),
                tool_names: vec!["Read".to_string(), "apply_patch".to_string()],
            },
        })
        .expect("provider trace should emit");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "provider.first_text_delta.received");
        assert_eq!(events[0].payload["stage"], "first_text_delta_received");
        assert_eq!(events[0].payload["provider"], "openai");
        assert_eq!(events[0].payload["model"], "gpt-4.1");
        assert_eq!(events[0].payload["provider_request_id"], "req-provider-1");
        assert_eq!(
            events[0].payload["provider_request_id_header"],
            "x-request-id"
        );
        assert_eq!(events[0].payload["runtime_provider_backend"], "current");
        assert_eq!(events[0].payload["runtime_provider_selector"], "codex");
        assert_eq!(events[0].payload["runtime_provider_protocol"], "responses");
        assert_eq!(
            events[0].payload["tool_names"],
            json!(["Read", "apply_patch"])
        );
        assert_eq!(
            events[0].payload["runtime_provider_active_model"],
            "gpt-4.1"
        );
        assert_eq!(events[0].payload["elapsed_ms"], json!(1234));
        assert_eq!(events[0].payload["text_chars"], json!(8));
    }

    #[test]
    fn provider_stream_event_maps_to_declared_runtime_event_kind() {
        let events = runtime_events_from_agent_event(&RuntimeAgentEvent::ProviderStreamEvent {
            runtime_event_kind: "provider_safety_buffering".to_string(),
            payload: json!({
                "kind": "provider_safety_buffering",
                "provider": "openai",
                "model": "gpt-5-codex",
                "useCases": ["policy"],
                "reasons": ["buffering"],
                "showBufferingUi": true,
                "retryModel": "gpt-5-mini",
                "source": "payload_retry_model"
            }),
        })
        .expect("provider stream event should emit");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "provider_safety_buffering");
        assert_eq!(events[0].payload["backend"], "runtime");
        assert_eq!(events[0].payload["provider"], "openai");
        assert_eq!(events[0].payload["model"], "gpt-5-codex");
        assert_eq!(events[0].payload["retryModel"], "gpt-5-mini");
        assert_eq!(events[0].payload["source"], "payload_retry_model");
        assert_eq!(
            events[0].payload["runtimeEvent"]["type"],
            "provider_stream_event"
        );
        assert!(events[0].payload.get("retry_model").is_none());
        assert!(events[0].payload.get("fasterModel").is_none());
    }

    #[test]
    fn unknown_provider_stream_event_falls_back_to_generic_runtime_event() {
        let events = runtime_events_from_agent_event(&RuntimeAgentEvent::ProviderStreamEvent {
            runtime_event_kind: "provider.future_event".to_string(),
            payload: json!({ "kind": "provider.future_event" }),
        })
        .expect("provider stream event should emit");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "runtime.event");
        assert_eq!(events[0].payload["kind"], "provider.future_event");
        assert_eq!(
            events[0].payload["runtimeEvent"]["runtime_event_kind"],
            "provider.future_event"
        );
    }

    fn canonical_tool_event(
        tool_id: &str,
        tool_name: &str,
        arguments: Option<Value>,
        result: Option<AgentToolResult>,
    ) -> RuntimeAgentEvent {
        let arguments = arguments.map(canonical_test_arguments).unwrap_or_default();
        let (status, output, metadata) = match result {
            Some(result) => {
                let status = if result.success {
                    ItemStatus::Completed
                } else {
                    ItemStatus::Failed
                };
                let metadata = result
                    .metadata
                    .map(|metadata| Value::Object(metadata.into_iter().collect()))
                    .unwrap_or_else(|| json!({}));
                let output = ToolOutput {
                    text: Some(result.output),
                    structured_content: result.structured_content,
                    error: result.error,
                    duration_ms: metadata.get("duration_ms").and_then(Value::as_u64),
                    truncated: metadata
                        .get("truncated")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    output_ref: metadata
                        .get("output_ref")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                };
                (status, Some(output), metadata)
            }
            None => (ItemStatus::InProgress, None, json!({})),
        };
        let payload = ThreadItemPayload::Tool {
            call_id: tool_id.to_string(),
            name: tool_name.to_string(),
            arguments,
            output,
        };
        let item = ThreadItem {
            session_id: SessionId::new("session-test"),
            thread_id: ThreadId::new("thread-test"),
            turn_id: TurnId::new("turn-test"),
            item_id: ItemId::new(tool_id),
            sequence: 1,
            ordinal: 1,
            created_at_ms: 1,
            updated_at_ms: 2,
            completed_at_ms: status.is_terminal().then_some(2),
            kind: payload.kind(),
            status,
            payload,
            metadata,
        };
        if status.is_terminal() {
            RuntimeAgentEvent::ItemCompleted { item }
        } else {
            RuntimeAgentEvent::ItemStarted { item }
        }
    }

    fn canonical_test_arguments(arguments: Value) -> Vec<ToolArgument> {
        match arguments {
            Value::Object(arguments) => arguments
                .into_iter()
                .map(|(name, value)| ToolArgument {
                    name,
                    value: value
                        .as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|| value.to_string()),
                })
                .collect(),
            value => vec![ToolArgument {
                name: "value".to_string(),
                value: value
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| value.to_string()),
            }],
        }
    }

    #[test]
    fn canonical_tool_start_emits_only_typed_item_arguments() {
        let events = runtime_events_from_agent_event(&canonical_tool_event(
            "tool-json-args",
            "Bash",
            Some(json!({ "command": "cargo test" })),
            None,
        ))
        .expect("tool item start should emit");

        assert_eq!(
            events
                .iter()
                .map(|event| event.event_type.as_str())
                .collect::<Vec<_>>(),
            vec!["item.started"]
        );
        assert_eq!(
            events[0].payload["item"]["payload"]["call_id"],
            "tool-json-args"
        );
        assert_eq!(
            events[0].payload["item"]["payload"]["arguments"][0]["name"],
            "command"
        );
        assert_eq!(
            events[0].payload["item"]["payload"]["arguments"][0]["value"],
            "cargo test"
        );
    }

    #[test]
    fn action_required_preserves_runtime_available_decisions_without_override() {
        let events = runtime_events_from_agent_event(&RuntimeAgentEvent::ActionRequired {
            request_id: "approval-1".to_string(),
            action_type: "tool_confirmation".to_string(),
            data: json!({
                "availableDecisions": [
                    "allow_once",
                    "allow_for_session",
                    "decline",
                    "cancel"
                ]
            }),
            scope: None,
        })
        .expect("action required should emit");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "action.required");
        assert!(events[0].payload.get("availableDecisions").is_none());
        assert_eq!(
            events[0].payload["data"]["availableDecisions"],
            json!(["allow_once", "allow_for_session", "decline", "cancel"])
        );
    }

    #[test]
    fn canonical_tool_start_without_arguments_emits_only_item() {
        let events = runtime_events_from_agent_event(&canonical_tool_event(
            "tool-no-args",
            "WebFetch",
            None,
            None,
        ))
        .expect("tool item start should emit");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "item.started");
    }

    #[test]
    fn canonical_tool_process_metadata_is_nested_on_item() {
        let soul_style = SoulStyleMetadata {
            profile_id: Some("cool_confident_operator".to_string()),
            pack_id: Some("com.lime.soul.cool-confident-operator".to_string()),
            tone_variant: Some("cool_confident".to_string()),
        };
        let events = runtime_events_from_agent_event_with_soul_style(
            &canonical_tool_event(
                "tool-search-style",
                "web_search",
                Some(json!({ "query": "runtime facts" })),
                None,
            ),
            Some(&soul_style),
        )
        .expect("tool item start should emit");

        let metadata = &events[0].payload["item"]["metadata"];
        assert_eq!(
            metadata["tool_process_summary"]["pre"]["key"],
            "toolCall.processSummary.webSearch.searchFirstWithQuery"
        );
        assert_eq!(
            metadata["tool_process_facts"]["profileId"],
            "cool_confident_operator"
        );
        assert_eq!(metadata["soul_lifecycle"]["status"], "started");
        assert!(metadata.get("metadata").is_none());
    }

    #[test]
    fn canonical_tool_completion_preserves_output_and_metadata() {
        let events = runtime_events_from_agent_event(&canonical_tool_event(
            "tool-mcp-structured",
            "mcp_lookup",
            Some(json!({ "query": "facts" })),
            Some(AgentToolResult {
                success: true,
                output: "ok".to_string(),
                error: None,
                structured_content: Some(json!({ "answer": "ok", "ids": ["doc-1"] })),
                images: None,
                metadata: Some(HashMap::from([
                    ("source".to_string(), json!("mcp")),
                    ("duration_ms".to_string(), json!(42)),
                    ("truncated".to_string(), json!(true)),
                    ("output_ref".to_string(), json!("sidecar://tool-output-1")),
                ])),
            }),
        ))
        .expect("tool item completion should emit");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "item.completed");
        let item = &events[0].payload["item"];
        assert_eq!(
            item["payload"]["output"]["structuredContent"]["answer"],
            "ok"
        );
        assert_eq!(item["payload"]["output"]["durationMs"], 42);
        assert_eq!(item["payload"]["output"]["truncated"], true);
        assert_eq!(
            item["payload"]["output"]["outputRef"],
            "sidecar://tool-output-1"
        );
        assert_eq!(item["metadata"]["source"], "mcp");
        assert_eq!(
            item["metadata"]["tool_process_facts"]["status"],
            "completed"
        );
    }

    #[test]
    fn canonical_update_plan_completion_emits_turn_plan_updated_fact() {
        let events = runtime_events_from_agent_event(&canonical_tool_event(
            "tool-plan",
            "update_plan",
            Some(json!({})),
            Some(AgentToolResult {
                success: true,
                output: "Plan updated".to_string(),
                error: None,
                structured_content: Some(json!({
                    "tool_family": "update_plan",
                    "explanation": "开始实现",
                    "plan": [
                        { "step": "读现状", "status": "completed" },
                        { "step": "打通主链", "status": "in_progress" }
                    ]
                })),
                images: None,
                metadata: None,
            }),
        ))
        .expect("update_plan item completion should emit");

        assert_eq!(
            events
                .iter()
                .map(|event| event.event_type.as_str())
                .collect::<Vec<_>>(),
            vec!["item.completed", "turn.plan.updated"]
        );
        assert_eq!(events[1].payload["toolCallId"], "tool-plan");
        assert_eq!(events[1].payload["explanation"], "开始实现");
        assert_eq!(events[1].payload["plan"][1]["status"], "in_progress");
    }

    #[test]
    fn non_update_plan_completion_does_not_emit_turn_plan_updated_fact() {
        let events = runtime_events_from_agent_event(&canonical_tool_event(
            "tool-other",
            "other_tool",
            Some(json!({})),
            Some(AgentToolResult {
                success: true,
                output: "Plan updated".to_string(),
                error: None,
                structured_content: Some(json!({
                    "plan": [{ "step": "不应投影", "status": "completed" }]
                })),
                images: None,
                metadata: None,
            }),
        ))
        .expect("other tool item completion should emit");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "item.completed");
    }

    #[test]
    fn canonical_failed_tool_stays_failed_item_and_gets_process_summary() {
        let events = runtime_events_from_agent_event(&canonical_tool_event(
            "tool-failed",
            "Bash",
            Some(json!({ "command": "cargo test" })),
            Some(AgentToolResult {
                success: false,
                output: "test failed".to_string(),
                error: Some("exit code 101".to_string()),
                structured_content: None,
                images: None,
                metadata: Some(HashMap::from([(
                    "failureCategory".to_string(),
                    json!("test_failed"),
                )])),
            }),
        ))
        .expect("failed tool item should emit");

        assert_eq!(events[0].event_type, "item.completed");
        let item = &events[0].payload["item"];
        assert_eq!(item["status"], "failed");
        assert_eq!(item["payload"]["output"]["error"], "exit code 101");
        assert_eq!(
            item["metadata"]["tool_process_summary"]["failed"]["key"],
            "toolCall.processSummary.error.failed"
        );
    }
}
