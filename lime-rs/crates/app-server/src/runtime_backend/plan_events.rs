use crate::RuntimeEvent;
use agent_protocol::ToolOutput;
use serde_json::{json, Value};

pub fn plan_delta_event(text: impl Into<String>, revision_id: impl Into<String>) -> RuntimeEvent {
    let text = text.into();
    RuntimeEvent::new(
        "plan.delta",
        json!({
            "text": text,
            "delta": text,
            "revisionId": revision_id.into(),
        }),
    )
}

pub fn proposed_plan_delta_event(
    text: impl Into<String>,
    delta: impl Into<String>,
    revision_id: impl Into<String>,
) -> RuntimeEvent {
    let mut event = plan_delta_event(text, revision_id);
    if let Some(payload_object) = event.payload.as_object_mut() {
        payload_object.insert("delta".to_string(), Value::String(delta.into()));
        payload_object.insert(
            "source".to_string(),
            Value::String("proposed_plan".to_string()),
        );
    }
    event
}

pub fn plan_final_event(
    text: impl Into<String>,
    revision_id: impl Into<String>,
    plan: Option<Value>,
) -> RuntimeEvent {
    let mut payload = json!({
        "text": text.into(),
        "revisionId": revision_id.into(),
    });
    if let Some(plan) = plan {
        if let Some(object) = payload.as_object_mut() {
            object.insert("plan".to_string(), plan);
        }
    }
    RuntimeEvent::new("plan.final", payload)
}

pub fn proposed_plan_final_event(
    text: impl Into<String>,
    revision_id: impl Into<String>,
) -> RuntimeEvent {
    let text = text.into();
    let mut event = plan_final_event(
        text.clone(),
        revision_id,
        plan_value_from_markdown_text(&text),
    );
    if let Some(payload_object) = event.payload.as_object_mut() {
        payload_object.insert(
            "source".to_string(),
            Value::String("proposed_plan".to_string()),
        );
    }
    event
}

pub fn turn_plan_updated_event_from_update_plan_result(
    tool_id: &str,
    output: &ToolOutput,
) -> Option<RuntimeEvent> {
    if output.text.as_deref()?.trim() != tool_runtime::update_plan::PLAN_UPDATED_MESSAGE {
        return None;
    }
    let structured_content = output.structured_content.as_ref()?.as_object()?;
    let plan = serde_json::from_value::<Vec<tool_runtime::update_plan::PlanStep>>(
        structured_content.get("plan")?.clone(),
    )
    .ok()?;
    let explanation = match structured_content.get("explanation") {
        None | Some(Value::Null) => None,
        Some(Value::String(explanation)) => Some(explanation.clone()),
        Some(_) => return None,
    };
    Some(RuntimeEvent::new(
        "turn.plan.updated",
        json!({
            "toolCallId": tool_id,
            "explanation": explanation,
            "plan": plan,
        }),
    ))
}

fn plan_value_from_markdown_text(text: &str) -> Option<Value> {
    let items = text
        .lines()
        .filter_map(plan_item_from_markdown_line)
        .collect::<Vec<_>>();
    (!items.is_empty()).then(|| Value::Array(items))
}

fn plan_item_from_markdown_line(line: &str) -> Option<Value> {
    let mut text = line.trim();
    if text.is_empty() {
        return None;
    }
    text = text
        .strip_prefix("- ")
        .or_else(|| text.strip_prefix("* "))
        .or_else(|| text.strip_prefix("+ "))
        .unwrap_or(text)
        .trim();
    if let Some((index, separator)) = text
        .char_indices()
        .find(|(_, ch)| *ch == '.' || *ch == ')')
        .filter(|(index, _)| *index > 0 && text[..*index].chars().all(|ch| ch.is_ascii_digit()))
    {
        text = text[index + separator.len_utf8()..].trim();
    }
    if text.is_empty() {
        return None;
    }
    let (status, step) = if let Some(rest) = text.strip_prefix("[x]") {
        ("completed", rest.trim())
    } else if let Some(rest) = text.strip_prefix("[X]") {
        ("completed", rest.trim())
    } else if let Some(rest) = text.strip_prefix("[~]") {
        ("in_progress", rest.trim())
    } else if let Some(rest) = text.strip_prefix("[ ]") {
        ("pending", rest.trim())
    } else {
        ("pending", text)
    };
    (!step.is_empty()).then(|| {
        json!({
            "step": step,
            "status": status,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_plan_delta_skeleton_event() {
        let event = plan_delta_event("整理计划", "rev-1");

        assert_eq!(event.event_type, "plan.delta");
        assert_eq!(event.payload["text"], "整理计划");
        assert_eq!(event.payload["delta"], "整理计划");
        assert_eq!(event.payload["revisionId"], "rev-1");
    }

    #[test]
    fn builds_proposed_plan_final_event_with_plan_items() {
        let event = proposed_plan_final_event("- [x] 读现状\n- 补主链", "plan:1");

        assert_eq!(event.event_type, "plan.final");
        assert_eq!(event.payload["revisionId"], "plan:1");
        assert_eq!(event.payload["source"], "proposed_plan");
        assert_eq!(event.payload["plan"][0]["status"], "completed");
        assert_eq!(event.payload["plan"][1]["step"], "补主链");
    }

    #[test]
    fn builds_turn_plan_updated_event_from_canonical_tool_output() {
        let event = turn_plan_updated_event_from_update_plan_result(
            "tool-plan-1",
            &ToolOutput {
                text: Some("Plan updated".to_string()),
                structured_content: Some(json!({
                    "tool_family": "update_plan",
                    "explanation": "继续实现",
                    "plan": [
                        { "step": "读现状", "status": "completed" },
                        { "step": "补主链", "status": "in_progress" }
                    ]
                })),
                ..ToolOutput::default()
            },
        )
        .expect("update_plan result should become turn.plan.updated");

        assert_eq!(event.event_type, "turn.plan.updated");
        assert_eq!(event.payload["toolCallId"], "tool-plan-1");
        assert_eq!(event.payload["explanation"], "继续实现");
        assert_eq!(event.payload["plan"][1]["status"], "in_progress");
    }

    #[test]
    fn ignores_update_plan_ack_without_canonical_structured_output() {
        let event = turn_plan_updated_event_from_update_plan_result(
            "tool-plan-1",
            &ToolOutput {
                text: Some("Plan updated".to_string()),
                structured_content: None,
                ..ToolOutput::default()
            },
        );

        assert!(event.is_none());
    }
}
