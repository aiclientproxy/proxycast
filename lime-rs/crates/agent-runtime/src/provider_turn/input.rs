use crate::reply_input::RuntimeReplyInputPart;
use crate::session_loop::{
    RuntimeSessionInput, RuntimeSessionInterAgentDeliveryMode, RuntimeSessionInterAgentInput,
    RuntimeSessionInterAgentMessageKind, RuntimeSessionInterAgentResultStatus,
};
use model_provider::current_client::{CurrentProviderContent, CurrentProviderMessage};

pub(super) fn escape_xml_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

pub(super) fn runtime_session_input_message(
    input: RuntimeSessionInput,
) -> Option<CurrentProviderMessage> {
    match input {
        RuntimeSessionInput::User(input) => {
            let content = input
                .parts
                .into_iter()
                .filter_map(|part| match part {
                    RuntimeReplyInputPart::Text { text, .. } => {
                        (!text.is_empty()).then_some(CurrentProviderContent::Text(text))
                    }
                    RuntimeReplyInputPart::Image(image) => Some(CurrentProviderContent::Image {
                        uri: image.uri,
                        media_type: image.media_type,
                        provider_data: image.provider_data,
                        detail: image.detail,
                    }),
                    RuntimeReplyInputPart::Skill { .. } | RuntimeReplyInputPart::Mention { .. } => {
                        None
                    }
                })
                .collect::<Vec<_>>();
            (!content.is_empty()).then(|| CurrentProviderMessage::user(content))
        }
        RuntimeSessionInput::Developer(text) => (!text.trim().is_empty())
            .then(|| CurrentProviderMessage::developer(vec![CurrentProviderContent::Text(text)])),
        RuntimeSessionInput::RawResponseItem(item) => {
            Some(CurrentProviderMessage::raw_response_item(item))
        }
        RuntimeSessionInput::InterAgent(input) => {
            let text = runtime_inter_agent_text(&input);
            (!text.trim().is_empty())
                .then(|| CurrentProviderMessage::user(vec![CurrentProviderContent::Text(text)]))
        }
    }
}

pub(super) fn runtime_inter_agent_text(input: &RuntimeSessionInterAgentInput) -> String {
    if input.content.trim().is_empty() {
        return String::new();
    }
    let mut lines = vec!["<inter_agent_message>".to_string()];
    lines.push(format!(
        "<message_id>{}</message_id>",
        escape_xml_text(&input.message_id)
    ));
    lines.push(format!(
        "<root_thread_id>{}</root_thread_id>",
        escape_xml_text(&input.root_thread_id)
    ));
    lines.push(format!(
        "<sender_thread_id>{}</sender_thread_id>",
        escape_xml_text(&input.sender_thread_id)
    ));
    lines.push(format!(
        "<recipient_thread_id>{}</recipient_thread_id>",
        escape_xml_text(&input.recipient_thread_id)
    ));
    if let Some(source_turn_id) = input.source_turn_id.as_deref() {
        lines.push(format!(
            "<source_turn_id>{}</source_turn_id>",
            escape_xml_text(source_turn_id)
        ));
    }
    let kind = match input.kind {
        RuntimeSessionInterAgentMessageKind::Message => "message",
        RuntimeSessionInterAgentMessageKind::Result => "result",
    };
    lines.push(format!("<kind>{kind}</kind>"));
    if let Some(status) = input.result_status {
        let status = match status {
            RuntimeSessionInterAgentResultStatus::Completed => "completed",
            RuntimeSessionInterAgentResultStatus::Failed => "failed",
        };
        lines.push(format!("<result_status>{status}</result_status>"));
    }
    let delivery_mode = match input.delivery_mode {
        RuntimeSessionInterAgentDeliveryMode::QueueOnly => "queue_only",
        RuntimeSessionInterAgentDeliveryMode::TriggerTurn => "trigger_turn",
    };
    lines.push(format!("<delivery_mode>{delivery_mode}</delivery_mode>"));
    lines.push(format!(
        "<content>{}</content>",
        escape_xml_text(&input.content)
    ));
    lines.push("</inter_agent_message>".to_string());
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_provider::current_client::CurrentProviderRole;
    use serde_json::json;

    #[test]
    fn developer_input_keeps_provider_role() {
        let message = runtime_session_input_message(RuntimeSessionInput::Developer(
            "approve the exact denied action".to_string(),
        ))
        .expect("developer message");

        assert_eq!(message.role, CurrentProviderRole::Developer);
        assert!(matches!(
            message.content.as_slice(),
            [CurrentProviderContent::Text(text)] if text == "approve the exact denied action"
        ));
    }

    #[test]
    fn raw_response_item_keeps_exact_provider_value() {
        let item = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "injected"}],
            "provider_extension": {"keep": true}
        });
        let message =
            runtime_session_input_message(RuntimeSessionInput::RawResponseItem(item.clone()))
                .expect("raw response item");

        assert_eq!(message.role, CurrentProviderRole::Assistant);
        assert!(matches!(
            message.content.as_slice(),
            [CurrentProviderContent::RawResponseItem(value)] if value == &item
        ));
    }
}
