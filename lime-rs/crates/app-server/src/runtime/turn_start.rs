use agent_protocol::{AgentInput, ImageDetail};
use app_server_protocol::{AgentAttachment, AgentSessionTurnStartParams, RuntimeOptions};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ReviewContext {
    pub(super) user_facing_hint: String,
    pub(super) target: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TurnStartInputKind {
    User,
    QueuedUser,
    PendingTriggerUser,
    GoalContinuation,
    Review,
}

impl TurnStartInputKind {
    pub(super) fn is_agent_only(self) -> bool {
        matches!(self, Self::GoalContinuation | Self::Review)
    }

    pub(super) fn runs_idle_scheduler(self) -> bool {
        !matches!(self, Self::QueuedUser | Self::PendingTriggerUser)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TurnStartRequest {
    pub session_id: String,
    pub turn_id: Option<String>,
    pub input: Vec<AgentInput>,
    pub runtime_options: Option<RuntimeOptions>,
    pub queue_if_busy: bool,
    pub skip_pre_submit_resume: bool,
}

impl From<AgentSessionTurnStartParams> for TurnStartRequest {
    fn from(params: AgentSessionTurnStartParams) -> Self {
        Self {
            session_id: params.session_id,
            turn_id: params.turn_id,
            input: legacy_user_input(params.input),
            runtime_options: params.runtime_options,
            queue_if_busy: params.queue_if_busy,
            skip_pre_submit_resume: params.skip_pre_submit_resume,
        }
    }
}

pub(super) fn validate_user_input(input: &[AgentInput]) -> Result<(), String> {
    if input.is_empty() {
        return Err("turn input must not be empty".to_string());
    }
    for part in input {
        part.validate().map_err(|error| error.to_string())?;
    }
    if input.iter().all(|part| {
        matches!(
            part,
            AgentInput::Text { text, .. } if text.trim().is_empty()
        )
    }) {
        return Err("turn input must not be empty".to_string());
    }
    Ok(())
}

pub(super) fn user_input_text(input: &[AgentInput]) -> String {
    input
        .iter()
        .filter_map(|part| match part {
            AgentInput::Text { text, .. } => Some(text.as_str()),
            AgentInput::Image { .. }
            | AgentInput::LocalImage { .. }
            | AgentInput::Skill { .. }
            | AgentInput::Mention { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn legacy_user_input(input: app_server_protocol::AgentInput) -> Vec<AgentInput> {
    let mut parts = Vec::with_capacity(input.attachments.len() + 1);
    if !input.text.is_empty() {
        parts.push(AgentInput::text(input.text));
    }
    parts.extend(input.attachments.iter().filter_map(legacy_image_input));
    parts
}

fn legacy_image_input(attachment: &AgentAttachment) -> Option<AgentInput> {
    if !attachment.kind.eq_ignore_ascii_case("image") {
        return None;
    }
    let uri = attachment
        .uri
        .as_deref()
        .map(str::trim)
        .filter(|uri| !uri.is_empty())?;
    let detail = attachment
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("detail"))
        .and_then(|detail| serde_json::from_value::<ImageDetail>(detail.clone()).ok());
    let local_path = attachment
        .metadata
        .as_ref()
        .and_then(|metadata| {
            metadata
                .get("localPath")
                .or_else(|| metadata.get("local_path"))
        })
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty());
    if let Some(path) = local_path {
        return Some(AgentInput::LocalImage {
            path: path.to_string(),
            detail,
        });
    }
    Some(AgentInput::Image {
        uri: uri.to_string(),
        detail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn legacy_input_is_converted_once_into_ordered_user_parts() {
        let request = TurnStartRequest::from(AgentSessionTurnStartParams {
            session_id: "session-1".to_string(),
            turn_id: None,
            input: app_server_protocol::AgentInput {
                text: "describe".to_string(),
                attachments: vec![AgentAttachment {
                    kind: "image".to_string(),
                    uri: Some("/tmp/input.png".to_string()),
                    metadata: Some(json!({
                        "localPath": "/tmp/input.png",
                        "detail": "high"
                    })),
                }],
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        });

        assert_eq!(
            request.input,
            vec![
                AgentInput::text("describe"),
                AgentInput::LocalImage {
                    path: "/tmp/input.png".to_string(),
                    detail: Some(ImageDetail::High),
                },
            ]
        );
    }

    #[test]
    fn structured_only_input_is_valid() {
        let input = vec![AgentInput::Skill {
            name: "review".to_string(),
            path: "/skills/review/SKILL.md".to_string(),
        }];

        assert!(validate_user_input(&input).is_ok());
        assert_eq!(user_input_text(&input), "");
    }
}
