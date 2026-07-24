use crate::RuntimeEvent;
use serde_json::{json, Value};
use std::collections::HashMap;

const UPDATE_PLAN_ACK: &str = "Plan updated";

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

pub fn plan_final_event_from_update_plan_result(
    tool_id: &str,
    output: &str,
    metadata: Option<&HashMap<String, Value>>,
) -> Option<RuntimeEvent> {
    if output.trim() != UPDATE_PLAN_ACK {
        return None;
    }
    let metadata = metadata?;
    let plan = metadata.get("plan")?.clone();
    let text = plan_text_from_plan_value(&plan)?;
    let revision_id = format!("update_plan:{tool_id}");
    let mut event = plan_final_event(text, revision_id, Some(plan));
    if let Some(payload_object) = event.payload.as_object_mut() {
        payload_object.insert("toolCallId".to_string(), Value::String(tool_id.to_string()));
        payload_object.insert(
            "source".to_string(),
            Value::String("update_plan".to_string()),
        );
        if let Some(explanation) = metadata.get("explanation") {
            if !explanation.is_null() {
                payload_object.insert("explanation".to_string(), explanation.clone());
            }
        }
    }
    Some(event)
}

fn plan_text_from_plan_value(plan: &Value) -> Option<String> {
    let lines = plan
        .as_array()?
        .iter()
        .filter_map(|item| {
            let step = item.get("step")?.as_str()?.trim();
            if step.is_empty() {
                return None;
            }
            let status = item
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("pending");
            let marker = if status == "completed" { "[x]" } else { "[ ]" };
            Some(format!("- {marker} {step}"))
        })
        .collect::<Vec<_>>();
    (!lines.is_empty()).then(|| lines.join("\n"))
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
    fn builds_update_plan_final_event_from_tool_metadata() {
        let event = plan_final_event_from_update_plan_result(
            "tool-plan-1",
            "Plan updated",
            Some(&HashMap::from([
                (
                    "plan".to_string(),
                    json!([
                        { "step": "读现状", "status": "completed" },
                        { "step": "补主链", "status": "in_progress" }
                    ]),
                ),
                ("explanation".to_string(), json!("继续实现")),
            ])),
        )
        .expect("update_plan result should become plan.final");

        assert_eq!(event.event_type, "plan.final");
        assert_eq!(event.payload["revisionId"], "update_plan:tool-plan-1");
        assert_eq!(event.payload["toolCallId"], "tool-plan-1");
        assert_eq!(event.payload["source"], "update_plan");
        assert_eq!(event.payload["explanation"], "继续实现");
        assert_eq!(event.payload["text"], "- [x] 读现状\n- [ ] 补主链");
        assert_eq!(event.payload["plan"][1]["status"], "in_progress");
    }
}
