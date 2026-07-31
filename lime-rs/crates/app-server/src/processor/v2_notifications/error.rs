use app_server_protocol::protocol::v2::{
    CodexErrorInfo, ErrorNotification, ServerNotification, TurnError,
};
use app_server_protocol::AgentEvent;
use serde_json::Value;

pub(super) struct ProjectedError {
    pub(super) notification: ServerNotification,
    pub(super) turn_id: String,
    pub(super) will_retry: bool,
}

pub(super) fn project(
    event: &AgentEvent,
    forced_will_retry: Option<bool>,
) -> Option<ProjectedError> {
    let thread_id = required_id(event.thread_id.as_deref())?;
    let turn_id = required_id(event.turn_id.as_deref())?;
    let message = error_message(&event.payload)?;
    let additional_details = optional_string(
        &event.payload,
        &["additionalDetails", "additional_details", "details"],
    )
    .or_else(|| nested_turn_error_string(&event.payload, "additionalDetails"))
    .or_else(|| nested_turn_error_string(&event.payload, "details"));
    if has_invalid_optional_string(
        &event.payload,
        &["additionalDetails", "additional_details", "details"],
    ) {
        return None;
    }
    let will_retry = match forced_will_retry {
        Some(will_retry) => will_retry,
        None => explicit_will_retry(&event.payload)?,
    };
    let code = payload_string(&event.payload, &["errorCode", "error_code", "code"])
        .or_else(|| nested_turn_error_string(&event.payload, "code"));

    Some(ProjectedError {
        notification: ServerNotification::Error(ErrorNotification {
            error: TurnError {
                message,
                codex_error_info: codex_error_info_from_code(code.as_deref()),
                additional_details,
            },
            will_retry,
            thread_id,
            turn_id: turn_id.clone(),
        }),
        turn_id,
        will_retry,
    })
}

pub(crate) fn codex_error_info_from_code(code: Option<&str>) -> Option<CodexErrorInfo> {
    let code = code?;
    let normalized = code
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    Some(match normalized.as_str() {
        "contextwindowexceeded" => CodexErrorInfo::ContextWindowExceeded,
        "sessionbudgetexceeded" => CodexErrorInfo::SessionBudgetExceeded,
        "usagelimitexceeded" => CodexErrorInfo::UsageLimitExceeded,
        "serveroverloaded" => CodexErrorInfo::ServerOverloaded,
        "cyberpolicy" => CodexErrorInfo::CyberPolicy,
        "httpconnectionfailed" => CodexErrorInfo::HttpConnectionFailed {
            http_status_code: None,
        },
        "responsestreamconnectionfailed" => CodexErrorInfo::ResponseStreamConnectionFailed {
            http_status_code: None,
        },
        "internalservererror" => CodexErrorInfo::InternalServerError,
        "unauthorized" => CodexErrorInfo::Unauthorized,
        "badrequest" => CodexErrorInfo::BadRequest,
        "threadrollbackfailed" => CodexErrorInfo::ThreadRollbackFailed,
        "sandboxerror" => CodexErrorInfo::SandboxError,
        "responsestreamdisconnected" => CodexErrorInfo::ResponseStreamDisconnected {
            http_status_code: None,
        },
        "responsetoomanyfailedattempts" => CodexErrorInfo::ResponseTooManyFailedAttempts {
            http_status_code: None,
        },
        _ => CodexErrorInfo::Other,
    })
}

fn error_message(payload: &Value) -> Option<String> {
    payload_string(payload, &["message", "errorMessage", "error_message"])
        .or_else(|| payload_string(payload, &["error"]))
        .or_else(|| nested_error_string(payload, "message"))
        .or_else(|| nested_turn_error_string(payload, "message"))
}

fn explicit_will_retry(payload: &Value) -> Option<bool> {
    let mut result = None;
    for key in ["willRetry", "will_retry"] {
        let Some(value) = payload.get(key) else {
            continue;
        };
        let value = value.as_bool()?;
        if result.is_some_and(|current| current != value) {
            return None;
        }
        result = Some(value);
    }
    Some(result.unwrap_or(false))
}

fn required_id(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn payload_string(payload: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

fn optional_string(payload: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| match payload.get(key) {
        Some(Value::String(value)) => Some(value.clone()),
        _ => None,
    })
}

fn has_invalid_optional_string(payload: &Value, keys: &[&str]) -> bool {
    keys.iter().any(|key| {
        !matches!(
            payload.get(key),
            None | Some(Value::Null) | Some(Value::String(_))
        )
    })
}

fn nested_error_string(payload: &Value, key: &str) -> Option<String> {
    payload
        .get("error")
        .and_then(Value::as_object)
        .and_then(|error| error.get(key))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn nested_turn_error_string(payload: &Value, key: &str) -> Option<String> {
    payload
        .get("turn")
        .and_then(Value::as_object)
        .and_then(|turn| turn.get("error"))
        .and_then(Value::as_object)
        .and_then(|error| error.get(key))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}
