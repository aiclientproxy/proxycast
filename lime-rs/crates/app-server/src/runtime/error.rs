use app_server_protocol::{error_codes, JsonRpcError, RuntimeOptions};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeCoreError {
    #[error("{0}")]
    MethodNotFound(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("session not found: {0}")]
    SessionNotFound(String),
    #[error("turn not active: {0}")]
    TurnNotActive(String),
    #[error("session already exists: {0}")]
    SessionAlreadyExists(String),
    #[error("turn already active: {0}")]
    TurnAlreadyActive(String),
    #[error("agent execution limit reached: {max_threads}")]
    AgentLimitReached { max_threads: usize },
    #[error("capability denied: {0}")]
    CapabilityDenied(String),
    #[error("request canceled")]
    RequestCanceled,
    #[error("{0}")]
    UsageLimitExceeded(String),
    #[error("rollout budget is exhausted")]
    RolloutBudgetExhausted,
    #[error("invalid provider rollout budget units")]
    InvalidRolloutBudgetUnits,
    #[error("invalid rollout budget config: {0}")]
    InvalidRolloutBudgetConfig(String),
    #[error("pending route for session {session_id}: {reason_code}")]
    PendingRoute {
        session_id: String,
        provider: Option<String>,
        model: Option<String>,
        reason_code: String,
    },
    #[error("route rejected for session {session_id}: {reason_code}")]
    RouteRejected {
        session_id: String,
        provider: Option<String>,
        model: Option<String>,
        category: app_server_protocol::RouteFailureCategory,
        reason_code: String,
    },
    #[error("execution backend error: {0}")]
    Backend(String),
    #[error("action response error ({code}): {request_id}")]
    ActionResponse { code: String, request_id: String },
}

impl RuntimeCoreError {
    pub fn is_provider_selection_required(&self) -> bool {
        matches!(self, Self::PendingRoute { .. })
    }

    pub fn pending_route_for_session(
        session_id: impl Into<String>,
        runtime_options: Option<&RuntimeOptions>,
    ) -> Self {
        let request = runtime_options.and_then(|options| options.runtime_request.as_ref());
        let provider = request.and_then(|request| {
            non_empty_route_hint(request.provider_preference.as_deref()).or_else(|| {
                request.provider_config.as_ref().and_then(|config| {
                    non_empty_route_hint(config.provider_id.as_deref())
                        .or_else(|| non_empty_route_hint(config.provider_name.as_deref()))
                })
            })
        });
        let model = request.and_then(|request| {
            non_empty_route_hint(request.model_preference.as_deref()).or_else(|| {
                request
                    .provider_config
                    .as_ref()
                    .and_then(|config| non_empty_route_hint(config.model_name.as_deref()))
            })
        });
        let reason_code = match (provider.is_some(), model.is_some()) {
            (false, false) => "provider_and_model_missing",
            (false, true) => "provider_missing",
            (true, false) => "model_missing",
            (true, true) => "route_unavailable",
        };
        Self::PendingRoute {
            session_id: session_id.into(),
            provider,
            model,
            reason_code: reason_code.to_string(),
        }
    }

    pub fn into_jsonrpc_error(self) -> JsonRpcError {
        match self {
            Self::MethodNotFound(message) => {
                JsonRpcError::new(error_codes::METHOD_NOT_FOUND, message)
            }
            Self::InvalidRequest(message) => {
                JsonRpcError::new(error_codes::INVALID_REQUEST, message)
            }
            Self::SessionNotFound(session_id) => JsonRpcError::new(
                error_codes::SESSION_NOT_FOUND,
                format!("session not found: {session_id}"),
            ),
            Self::TurnNotActive(turn_id) => JsonRpcError::new(
                error_codes::TURN_NOT_ACTIVE,
                format!("turn not active: {turn_id}"),
            ),
            Self::SessionAlreadyExists(session_id) => JsonRpcError::new(
                error_codes::SESSION_ALREADY_EXISTS,
                format!("session already exists: {session_id}"),
            ),
            Self::TurnAlreadyActive(turn_id) => JsonRpcError::new(
                error_codes::TURN_ALREADY_ACTIVE,
                format!("turn already active: {turn_id}"),
            ),
            Self::AgentLimitReached { max_threads } => JsonRpcError {
                code: error_codes::RUNTIME_ERROR,
                message: "agent execution capacity is exhausted".to_string(),
                data: Some(serde_json::json!({
                    "type": "AgentLimitReached",
                    "reason": "agent_limit_reached",
                    "maxThreads": max_threads,
                    "retryable": true,
                })),
            },
            Self::CapabilityDenied(capability_id) => JsonRpcError::new(
                error_codes::CAPABILITY_DENIED,
                format!("capability denied: {capability_id}"),
            ),
            Self::PendingRoute {
                session_id,
                provider,
                model,
                reason_code,
            } => JsonRpcError {
                code: error_codes::RUNTIME_ERROR,
                message: "App Server runtime backend requires provider/model selection. Start or resume the canonical thread with a complete modelProvider/model route before starting a turn.".to_string(),
                data: Some(serde_json::json!({
                    "type": "PendingRoute",
                    "sessionId": session_id,
                    "provider": provider,
                    "model": model,
                    "reasonCode": reason_code,
                    "retryable": true,
                })),
            },
            Self::RouteRejected {
                session_id,
                provider,
                model,
                category,
                reason_code,
            } => JsonRpcError {
                code: error_codes::RUNTIME_ERROR,
                message: "runtime model route is not executable".to_string(),
                data: Some(serde_json::json!({
                    "type": "RouteRejected",
                    "sessionId": session_id,
                    "provider": provider,
                    "model": model,
                    "category": category,
                    "reasonCode": reason_code,
                    "retryable": false,
                })),
            },
            Self::RequestCanceled => {
                JsonRpcError::new(error_codes::REQUEST_CANCELLED, "request canceled")
            }
            Self::UsageLimitExceeded(message) => {
                JsonRpcError::new(error_codes::RUNTIME_ERROR, message)
            }
            Self::RolloutBudgetExhausted => JsonRpcError {
                code: error_codes::RUNTIME_ERROR,
                message: "shared rollout budget is exhausted".to_string(),
                data: Some(serde_json::json!({"reason": "rollout_budget_exhausted", "retryable": false})),
            },
            Self::InvalidRolloutBudgetUnits => JsonRpcError {
                code: error_codes::RUNTIME_ERROR,
                message: "provider returned invalid rollout budget units".to_string(),
                data: Some(serde_json::json!({"reason": "invalid_rollout_budget_units", "retryable": false})),
            },
            Self::InvalidRolloutBudgetConfig(message) => {
                JsonRpcError::new(error_codes::INVALID_REQUEST, message)
            }
            Self::Backend(message) => JsonRpcError::new(error_codes::RUNTIME_ERROR, message),
            Self::ActionResponse { code, request_id } => JsonRpcError {
                code: error_codes::RUNTIME_ERROR,
                message: format!("action response failed: {code}"),
                data: Some(serde_json::json!({
                    "code": code,
                    "requestId": request_id,
                })),
            },
        }
    }

    pub(crate) fn turn_failure_reason(&self) -> &'static str {
        match self {
            Self::UsageLimitExceeded(_) | Self::RolloutBudgetExhausted => "usage_limit_exceeded",
            Self::InvalidRolloutBudgetUnits => "invalid_rollout_budget_units",
            Self::InvalidRolloutBudgetConfig(_) => "invalid_rollout_budget_config",
            _ => "turn_error",
        }
    }
}

fn non_empty_route_hint(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}
