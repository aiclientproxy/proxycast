//! 请求上下文
//!
//! 定义请求处理过程中的上下文信息

use crate::models::provider_type::ProviderType;
use chrono::{DateTime, Utc};
use std::time::Instant;

/// 请求上下文
///
/// 在请求处理管道中传递的上下文信息
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// 请求唯一标识
    pub request_id: String,
    /// 请求开始时间
    pub start_time: Instant,
    /// 请求时间戳
    pub timestamp: DateTime<Utc>,
    /// 原始模型名称（请求中的模型）
    pub original_model: String,
    /// 解析后的模型名称（经过别名映射）
    pub resolved_model: String,
    /// 选择的 Provider
    pub provider: Option<ProviderType>,
    /// 使用的凭证 ID
    pub credential_id: Option<String>,
    /// 重试次数
    pub retry_count: u32,
    /// 是否为流式请求
    pub is_stream: bool,
    /// 元数据
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl RequestContext {
    /// 创建新的请求上下文
    pub fn new(model: String) -> Self {
        let request_id = uuid::Uuid::new_v4().to_string();
        Self {
            request_id: request_id.clone(),
            start_time: Instant::now(),
            timestamp: Utc::now(),
            original_model: model.clone(),
            resolved_model: model,
            provider: None,
            credential_id: None,
            retry_count: 0,
            is_stream: false,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// 设置流式请求标志
    pub fn with_stream(mut self, is_stream: bool) -> Self {
        self.is_stream = is_stream;
        self
    }

    /// 设置 Provider
    pub fn set_provider(&mut self, provider: ProviderType) {
        self.provider = Some(provider);
    }

    /// 设置凭证 ID
    pub fn set_credential_id(&mut self, credential_id: String) {
        self.credential_id = Some(credential_id);
    }

    /// 设置解析后的模型名称
    pub fn set_resolved_model(&mut self, model: String) {
        self.resolved_model = model;
    }

    /// 增加重试计数
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    /// 获取已耗时（毫秒）
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// 添加元数据
    pub fn set_metadata(&mut self, key: &str, value: serde_json::Value) {
        self.metadata.insert(key.to_string(), value);
    }

    /// 获取元数据
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }
}

impl Default for RequestContext {
    fn default() -> Self {
        Self::new(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_context_new() {
        let ctx = RequestContext::new("claude-sonnet-4-5".to_string());

        assert!(!ctx.request_id.is_empty());
        assert_eq!(ctx.original_model, "claude-sonnet-4-5");
        assert_eq!(ctx.resolved_model, "claude-sonnet-4-5");
        assert!(ctx.provider.is_none());
        assert!(ctx.credential_id.is_none());
        assert_eq!(ctx.retry_count, 0);
        assert!(!ctx.is_stream);
    }

    #[test]
    fn test_request_context_with_stream() {
        let ctx = RequestContext::new("model".to_string()).with_stream(true);
        assert!(ctx.is_stream);
    }

    #[test]
    fn test_request_context_set_provider() {
        let mut ctx = RequestContext::new("model".to_string());
        ctx.set_provider(ProviderType::Gemini);
        assert_eq!(ctx.provider, Some(ProviderType::Gemini));
    }

    #[test]
    fn test_request_context_increment_retry() {
        let mut ctx = RequestContext::new("model".to_string());
        assert_eq!(ctx.retry_count, 0);
        ctx.increment_retry();
        assert_eq!(ctx.retry_count, 1);
        ctx.increment_retry();
        assert_eq!(ctx.retry_count, 2);
    }

    #[test]
    fn test_request_context_metadata() {
        let mut ctx = RequestContext::new("model".to_string());
        ctx.set_metadata("key", serde_json::json!("value"));

        let value = ctx.get_metadata("key");
        assert!(value.is_some());
        assert_eq!(value.unwrap(), &serde_json::json!("value"));
    }
}
