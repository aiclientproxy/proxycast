//! 处理错误类型
//!
//! 定义请求处理过程中可能发生的错误

#![allow(dead_code)]

use thiserror::Error;

/// 处理错误
#[derive(Error, Debug, Clone)]
pub enum ProcessError {
    /// 认证失败
    #[error("认证失败: {0}")]
    AuthError(String),

    /// 路由失败
    #[error("路由失败: 无可用 Provider 处理模型 {model}")]
    RoutingError { model: String },

    /// Provider 调用失败
    #[error("Provider 调用失败: {0}")]
    ProviderError(String),

    /// 重试耗尽
    #[error("重试耗尽: 尝试 {attempts} 次后失败")]
    RetriesExhausted { attempts: u32 },

    /// 请求超时
    #[error("请求超时: {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// 流式响应空闲超时
    #[error("流式响应空闲超时: {timeout_ms}ms")]
    StreamIdleTimeout { timeout_ms: u64 },

    /// 参数注入错误
    #[error("参数注入错误: {0}")]
    InjectionError(String),

    /// 配置错误
    #[error("配置错误: {0}")]
    ConfigError(String),

    /// 内部错误
    #[error("内部错误: {0}")]
    InternalError(String),

    /// 请求被取消
    #[error("请求已取消")]
    Cancelled,
}

impl ProcessError {
    /// 获取对应的 HTTP 状态码
    pub fn status_code(&self) -> u16 {
        match self {
            ProcessError::AuthError(_) => 401,
            ProcessError::RoutingError { .. } => 404,
            ProcessError::ProviderError(_) => 502,
            ProcessError::RetriesExhausted { .. } => 503,
            ProcessError::Timeout { .. } => 408,
            ProcessError::StreamIdleTimeout { .. } => 408,
            ProcessError::InjectionError(_) => 400,
            ProcessError::ConfigError(_) => 500,
            ProcessError::InternalError(_) => 500,
            ProcessError::Cancelled => 499,
        }
    }

    /// 检查是否为可重试错误
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ProcessError::ProviderError(_)
                | ProcessError::Timeout { .. }
                | ProcessError::StreamIdleTimeout { .. }
        )
    }

    /// 检查是否应该触发故障转移
    pub fn should_failover(&self) -> bool {
        matches!(
            self,
            ProcessError::ProviderError(_) | ProcessError::RetriesExhausted { .. }
        )
    }

    /// 转换为 JSON 错误响应
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {
                "message": self.to_string(),
                "type": self.error_type(),
                "code": self.status_code()
            }
        })
    }

    /// 获取错误类型字符串
    pub fn error_type(&self) -> &'static str {
        match self {
            ProcessError::AuthError(_) => "authentication_error",
            ProcessError::RoutingError { .. } => "routing_error",
            ProcessError::ProviderError(_) => "provider_error",
            ProcessError::RetriesExhausted { .. } => "retries_exhausted",
            ProcessError::Timeout { .. } => "timeout_error",
            ProcessError::StreamIdleTimeout { .. } => "stream_idle_timeout",
            ProcessError::InjectionError(_) => "injection_error",
            ProcessError::ConfigError(_) => "config_error",
            ProcessError::InternalError(_) => "internal_error",
            ProcessError::Cancelled => "cancelled",
        }
    }

    /// 记录带上下文的错误日志
    pub fn log_with_context(&self, request_id: &str, provider: &str, model: &str) {
        tracing::error!(
            request_id = %request_id,
            provider = %provider,
            model = %model,
            error_type = %self.error_type(),
            error_message = %self.to_string(),
            "Request processing failed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_error_status_codes() {
        assert_eq!(
            ProcessError::AuthError("test".to_string()).status_code(),
            401
        );
        assert_eq!(
            ProcessError::RoutingError {
                model: "test".to_string()
            }
            .status_code(),
            404
        );
        assert_eq!(
            ProcessError::ProviderError("test".to_string()).status_code(),
            502
        );
        assert_eq!(
            ProcessError::RetriesExhausted { attempts: 3 }.status_code(),
            503
        );
        assert_eq!(
            ProcessError::Timeout { timeout_ms: 5000 }.status_code(),
            408
        );
        assert_eq!(ProcessError::Cancelled.status_code(), 499);
    }

    #[test]
    fn test_process_error_is_retryable() {
        assert!(ProcessError::ProviderError("test".to_string()).is_retryable());
        assert!(ProcessError::Timeout { timeout_ms: 5000 }.is_retryable());
        assert!(!ProcessError::AuthError("test".to_string()).is_retryable());
        assert!(!ProcessError::RoutingError {
            model: "test".to_string()
        }
        .is_retryable());
    }

    #[test]
    fn test_process_error_should_failover() {
        assert!(ProcessError::ProviderError("test".to_string()).should_failover());
        assert!(ProcessError::RetriesExhausted { attempts: 3 }.should_failover());
        assert!(!ProcessError::AuthError("test".to_string()).should_failover());
        assert!(!ProcessError::Timeout { timeout_ms: 5000 }.should_failover());
    }

    #[test]
    fn test_process_error_to_json() {
        let error = ProcessError::AuthError("Invalid API key".to_string());
        let json = error.to_json();

        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Invalid API key"));
        assert_eq!(json["error"]["type"], "authentication_error");
        assert_eq!(json["error"]["code"], 401);
    }
}
