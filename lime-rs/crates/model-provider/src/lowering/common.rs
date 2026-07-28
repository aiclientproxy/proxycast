use app_server_protocol::ProtocolKind;
use runtime_core::{CanonicalRequest, ContentPart};
use serde_json::Value;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolMappingError {
    UnsupportedInputPart {
        protocol: ProtocolKind,
        part_type: &'static str,
    },
    UnsupportedOption {
        protocol: ProtocolKind,
        option: &'static str,
    },
    EmptyInput,
}

impl fmt::Display for ProtocolMappingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedInputPart {
                protocol,
                part_type,
            } => write!(
                f,
                "input part `{part_type}` is not supported by protocol {protocol:?}"
            ),
            Self::UnsupportedOption { protocol, option } => write!(
                f,
                "provider option `{option}` is not supported by protocol {protocol:?}"
            ),
            Self::EmptyInput => f.write_str("canonical LLM request must contain input"),
        }
    }
}

pub(crate) fn unsupported_option(
    protocol: ProtocolKind,
    option: &'static str,
) -> ProtocolMappingError {
    ProtocolMappingError::UnsupportedOption { protocol, option }
}

impl std::error::Error for ProtocolMappingError {}

pub(crate) fn unsupported(protocol: ProtocolKind, part_type: &'static str) -> ProtocolMappingError {
    ProtocolMappingError::UnsupportedInputPart {
        protocol,
        part_type,
    }
}

pub(crate) fn json_with_value_fields<const N: usize>(
    value: Value,
    fields: [(&str, Option<Value>); N],
) -> Value {
    let mut value = value;
    if let Some(object) = value.as_object_mut() {
        for (key, field) in fields {
            if let Some(field) = field {
                object.insert(key.to_string(), field);
            }
        }
    }
    value
}

pub(crate) fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

pub(crate) fn canonical_generation_prompt(
    request: &CanonicalRequest,
    protocol: ProtocolKind,
    allow_media: bool,
) -> Result<String, ProtocolMappingError> {
    if !request.tools.is_empty() {
        return Err(unsupported(protocol, "tools"));
    }

    let mut sections = Vec::new();
    for part in request.system.iter().chain(
        request
            .messages
            .iter()
            .flat_map(|message| message.content.iter()),
    ) {
        match part {
            ContentPart::Text { text, .. } => {
                if let Some(text) = non_empty(Some(text)) {
                    sections.push(text.to_string());
                }
            }
            ContentPart::Media { .. } if allow_media => {}
            ContentPart::Media { .. } => return Err(unsupported(protocol, "media")),
            ContentPart::Reasoning { .. } => return Err(unsupported(protocol, "reasoning")),
            ContentPart::ToolCall { .. } => return Err(unsupported(protocol, "tool_call")),
            ContentPart::ToolResult { .. } => return Err(unsupported(protocol, "tool_result")),
            ContentPart::RawResponseItem { .. } => {
                return Err(unsupported(protocol, "raw_response_item"));
            }
        }
    }

    if sections.is_empty() {
        Err(ProtocolMappingError::EmptyInput)
    } else {
        Ok(sections.join("\n\n"))
    }
}

pub(crate) fn canonical_media_references(request: &CanonicalRequest) -> Vec<&str> {
    request
        .messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|part| match part {
            ContentPart::Media { uri, .. } => non_empty(Some(uri)),
            _ => None,
        })
        .collect()
}
