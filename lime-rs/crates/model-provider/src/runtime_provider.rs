//! Runtime provider 配置与错误判定边界。

use crate::ModelProviderProtocol;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeProviderProtocol {
    ChatCompletions,
    Responses,
    AnthropicMessages,
}

impl RuntimeProviderProtocol {
    pub fn uses_responses_api(self) -> bool {
        matches!(self, Self::Responses)
    }

    pub fn to_model_provider_protocol(self) -> ModelProviderProtocol {
        match self {
            Self::ChatCompletions => ModelProviderProtocol::ChatCompletions,
            Self::Responses => ModelProviderProtocol::Responses,
            Self::AnthropicMessages => ModelProviderProtocol::AnthropicMessages,
        }
    }
}

/// Runtime provider 配置。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeProviderConfig {
    /// Provider 名称 (openai, anthropic, google 等)
    pub provider_name: String,
    /// Provider 选择器（优先保留前端 provider_id / runtime_provider_type）
    pub provider_selector: Option<String>,
    /// 模型名称
    pub model_name: String,
    /// API Key
    pub api_key: Option<String>,
    /// Base URL
    pub base_url: Option<String>,
    /// 凭证 UUID（用于记录使用和健康状态）
    pub credential_uuid: String,
    /// 当前回合显式推理强度
    pub reasoning_effort: Option<String>,
    /// 当前回合显式服务等级
    pub service_tier: Option<String>,
    /// App Server RouteResolver 派生出的 provider 执行协议
    pub protocol: Option<RuntimeProviderProtocol>,
    /// Provider 是否显式支持 Responses WebSocket transport
    pub supports_websockets: bool,
    /// 当前回合是否启用 toolshim
    pub toolshim: bool,
    /// toolshim 解释器模型
    pub toolshim_model: Option<String>,
}

pub fn message_is_non_retryable_provider_rejection(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    normalized.contains("authentication error")
        || normalized.contains("unauthorized")
        || normalized.contains("forbidden")
        || !is_retryable_request_failed_message(message)
}

fn is_retryable_request_failed_message(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    let non_retryable_markers = [
        "bad request (400)",
        "resource not found (404)",
        "invalid_request_error",
        "status: 400",
        "status: 401",
        "status: 403",
        "status: 404",
        "status 400",
        "status 401",
        "status 403",
        "status 404",
    ];

    !non_retryable_markers
        .iter()
        .any(|marker| normalized.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::message_is_non_retryable_provider_rejection;

    #[test]
    fn classifies_non_retryable_provider_rejections() {
        assert!(message_is_non_retryable_provider_rejection(
            "Request failed: Bad request (400): 当前模型未在租户白名单中开放"
        ));
        assert!(message_is_non_retryable_provider_rejection(
            "Authentication error: invalid key"
        ));
        assert!(!message_is_non_retryable_provider_rejection(
            "connection failed"
        ));
        assert!(!message_is_non_retryable_provider_rejection(
            "Server error: temporarily unavailable"
        ));
    }
}
