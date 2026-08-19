use super::coding_events;
use super::proposed_plan_parser;
use super::reasoning_events;
use super::tool_events;
use super::tool_process_metadata::SoulStyleMetadata;
use crate::RuntimeCoreError;
use crate::RuntimeEventSink;
use lime_agent::AgentEvent as RuntimeAgentEvent;
use serde_json::json;

pub(super) struct ModelRouteEvidence {
    pub provider: String,
    pub requested_model: String,
    pub selected_model: String,
    pub route_attempt: usize,
}

#[cfg(test)]
pub(super) fn emit_runtime_agent_event_with_coding_mirror(
    event: &RuntimeAgentEvent,
    sink: &mut dyn RuntimeEventSink,
    coding_event_mirror: &mut coding_events::CodingEventMirror,
) -> Result<(), RuntimeCoreError> {
    let mut proposed_plan_parser = proposed_plan_parser::ProposedPlanParser::default();
    let mut reasoning_event_state = reasoning_events::ReasoningEventState::default();
    emit_runtime_agent_event_with_coding_mirror_and_plan_parser(
        event,
        sink,
        coding_event_mirror,
        &mut proposed_plan_parser,
        &mut reasoning_event_state,
    )
}

#[cfg(test)]
pub(super) fn emit_runtime_agent_event_with_coding_mirror_and_plan_parser(
    event: &RuntimeAgentEvent,
    sink: &mut dyn RuntimeEventSink,
    coding_event_mirror: &mut coding_events::CodingEventMirror,
    proposed_plan_parser: &mut proposed_plan_parser::ProposedPlanParser,
    reasoning_event_state: &mut reasoning_events::ReasoningEventState,
) -> Result<(), RuntimeCoreError> {
    emit_runtime_agent_event_with_coding_mirror_and_plan_parser_with_soul_style(
        event,
        sink,
        coding_event_mirror,
        proposed_plan_parser,
        reasoning_event_state,
        None,
        None,
    )
}

pub(super) fn emit_runtime_agent_event_with_coding_mirror_and_plan_parser_with_soul_style(
    event: &RuntimeAgentEvent,
    sink: &mut dyn RuntimeEventSink,
    coding_event_mirror: &mut coding_events::CodingEventMirror,
    proposed_plan_parser: &mut proposed_plan_parser::ProposedPlanParser,
    reasoning_event_state: &mut reasoning_events::ReasoningEventState,
    soul_style: Option<&SoulStyleMetadata>,
    model_route_evidence: Option<&ModelRouteEvidence>,
) -> Result<(), RuntimeCoreError> {
    if let RuntimeAgentEvent::CodeCellTrace { event } = event {
        return sink.emit_code_cell_trace(event.clone());
    }
    if let RuntimeAgentEvent::RolloutBudgetReminder {
        durable_event_id, ..
    } = event
    {
        return sink.emit_preappended_by_id(durable_event_id);
    }
    let coding_events = coding_event_mirror.process_event(event);
    let persistence_event = coding_events::runtime_event_for_persistence(event);
    for event in coding_events.before_raw {
        sink.emit(event)?;
    }
    match event {
        RuntimeAgentEvent::TextStart { item_id } => {
            proposed_plan_parser.observe_message_start(item_id);
        }
        RuntimeAgentEvent::TextEnd { item_id, phase } => {
            emit_agent_message_item_finish(
                proposed_plan_parser,
                item_id,
                phase.as_str(),
                "completed",
                sink,
            )?;
        }
        RuntimeAgentEvent::ReasoningStart { item_id } => {
            reasoning_event_state
                .start(item_id)
                .map_err(RuntimeCoreError::Backend)?;
        }
        RuntimeAgentEvent::ReasoningSummaryDelta {
            item_id,
            text,
            summary_index,
        } => {
            for event in reasoning_event_state
                .observe_summary_delta(item_id, text, *summary_index)
                .map_err(RuntimeCoreError::Backend)?
            {
                sink.emit(event)?;
            }
            emit_presentation_events(
                event,
                soul_style,
                model_route_evidence,
                proposed_plan_parser,
                sink,
            )?;
        }
        RuntimeAgentEvent::ReasoningSummaryPartAdded {
            item_id,
            summary_index,
        } => {
            for event in reasoning_event_state
                .observe_summary_part_added(item_id, *summary_index)
                .map_err(RuntimeCoreError::Backend)?
            {
                sink.emit(event)?;
            }
            emit_presentation_events(
                event,
                soul_style,
                model_route_evidence,
                proposed_plan_parser,
                sink,
            )?;
        }
        RuntimeAgentEvent::ReasoningContentDelta {
            item_id,
            text,
            content_index,
        } => {
            for event in reasoning_event_state
                .observe_content_delta(item_id, text, *content_index)
                .map_err(RuntimeCoreError::Backend)?
            {
                sink.emit(event)?;
            }
            emit_presentation_events(
                event,
                soul_style,
                model_route_evidence,
                proposed_plan_parser,
                sink,
            )?;
        }
        RuntimeAgentEvent::ReasoningEnd { item_id } => {
            for event in reasoning_event_state
                .end(item_id, "completed")
                .map_err(RuntimeCoreError::Backend)?
            {
                sink.emit(event)?;
            }
        }
        _ => emit_presentation_events(
            persistence_event.as_ref(),
            soul_style,
            model_route_evidence,
            proposed_plan_parser,
            sink,
        )?,
    }
    for event in coding_events.after_raw {
        sink.emit(event)?;
    }
    Ok(())
}

fn emit_presentation_events(
    event: &RuntimeAgentEvent,
    soul_style: Option<&SoulStyleMetadata>,
    model_route_evidence: Option<&ModelRouteEvidence>,
    proposed_plan_parser: &mut proposed_plan_parser::ProposedPlanParser,
    sink: &mut dyn RuntimeEventSink,
) -> Result<(), RuntimeCoreError> {
    for mut event in
        tool_events::runtime_events_from_agent_event_with_soul_style(event, soul_style)?
    {
        if matches!(
            event.event_type.as_str(),
            "model.server_reported" | "provider.usage" | "provider.step"
        ) {
            if let (Some(evidence), Some(payload)) =
                (model_route_evidence, event.payload.as_object_mut())
            {
                if event.event_type == "model.server_reported" {
                    payload.insert("provider".to_string(), evidence.provider.clone().into());
                    payload.insert(
                        "requestedModel".to_string(),
                        evidence.requested_model.clone().into(),
                    );
                    payload.insert(
                        "selectedModel".to_string(),
                        evidence.selected_model.clone().into(),
                    );
                }
                payload.insert("routeAttempt".to_string(), evidence.route_attempt.into());
            }
        }
        for event in proposed_plan_parser::split_runtime_event(event, proposed_plan_parser) {
            if event.event_type == "model.rerouted" {
                sink.emit_transient(event)?;
            } else {
                sink.emit(event)?;
            }
        }
    }
    Ok(())
}

pub(super) fn emit_reasoning_finish(
    reasoning_event_state: &mut reasoning_events::ReasoningEventState,
    status: &str,
    sink: &mut dyn RuntimeEventSink,
) -> Result<(), RuntimeCoreError> {
    for event in reasoning_event_state
        .finish(status)
        .map_err(RuntimeCoreError::Backend)?
    {
        sink.emit(event)?;
    }
    Ok(())
}

pub(super) fn emit_agent_message_finish(
    proposed_plan_parser: &mut proposed_plan_parser::ProposedPlanParser,
    status: &str,
    sink: &mut dyn RuntimeEventSink,
) -> Result<(), RuntimeCoreError> {
    let phase = if status == "completed" {
        "final_answer"
    } else {
        "commentary"
    };
    for message in proposed_plan_parser::finish_runtime_messages(proposed_plan_parser) {
        emit_finished_message(message, phase, status, sink)?;
    }
    Ok(())
}

fn emit_agent_message_item_finish(
    proposed_plan_parser: &mut proposed_plan_parser::ProposedPlanParser,
    item_id: &str,
    phase: &str,
    status: &str,
    sink: &mut dyn RuntimeEventSink,
) -> Result<(), RuntimeCoreError> {
    let message = proposed_plan_parser::finish_runtime_message(proposed_plan_parser, item_id)
        .map_err(RuntimeCoreError::Backend)?;
    emit_finished_message(message, phase, status, sink)
}

fn emit_finished_message(
    message: proposed_plan_parser::FinishedMessage,
    phase: &str,
    status: &str,
    sink: &mut dyn RuntimeEventSink,
) -> Result<(), RuntimeCoreError> {
    for event in message.events {
        sink.emit(event)?;
    }
    if message.has_message_output {
        let mut payload = json!({
            "role": "assistant",
            "phase": phase,
            "status": status,
        });
        if let (Some(item_id), Some(payload)) = (message.item_id, payload.as_object_mut()) {
            payload.insert("itemId".to_string(), item_id.into());
        }
        sink.emit(crate::RuntimeEvent::new("message.completed", payload))?;
    }
    Ok(())
}
