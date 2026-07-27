//! Lime LLM Provider 实现
//!
//! 使用 API Key Provider 选择凭证并调用 LLM API。
//! trait 定义（LlmProvider, SkillError）已迁移到 lime-skills crate。

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;

use lime_core::database::DbConnection;
use lime_core::models::{
    runtime_api_key_id_from_credential_uuid, RuntimeCredentialData, RuntimeProviderCredential,
    RuntimeProviderType,
};
use lime_services::api_key_provider_service::ApiKeyProviderService;
use model_provider::current_client::{
    CanonicalLlmEvent, CurrentProviderClient, CurrentProviderContent, CurrentProviderMessage,
    CurrentProviderRequest,
};
use model_provider::runtime_provider::{
    RuntimeProviderAuth, RuntimeProviderConfig, RuntimeProviderProtocol,
};

use crate::{LlmProvider, SkillError};

/// Lime LLM Provider
///
/// 使用 API Key Provider 选择凭证并调用 LLM API。
/// 实现 agent-rust 定义的 LlmProvider trait。
pub struct LimeLlmProvider {
    /// API Key Provider 服务
    api_key_service: Arc<ApiKeyProviderService>,
    /// 数据库连接
    db: DbConnection,
    /// 偏好的 Provider 类型（可选）
    preferred_provider: Option<String>,
}

impl LimeLlmProvider {
    /// 创建新的 LimeLlmProvider 实例
    ///
    /// # Arguments
    /// * `api_key_service` - API Key 服务
    /// * `db` - 数据库连接
    pub fn new(api_key_service: Arc<ApiKeyProviderService>, db: DbConnection) -> Self {
        Self {
            api_key_service,
            db,
            preferred_provider: None,
        }
    }

    /// 创建带有偏好 Provider 的实例
    ///
    /// # Arguments
    /// * `api_key_service` - API Key 服务
    /// * `db` - 数据库连接
    /// * `preferred_provider` - 偏好的 Provider 类型
    pub fn with_preferred_provider(
        api_key_service: Arc<ApiKeyProviderService>,
        db: DbConnection,
        preferred_provider: String,
    ) -> Self {
        Self {
            api_key_service,
            db,
            preferred_provider: Some(preferred_provider),
        }
    }

    /// 设置偏好的 Provider 类型
    pub fn set_preferred_provider(&mut self, provider: Option<String>) {
        self.preferred_provider = provider;
    }

    /// 获取偏好的 Provider 类型
    pub fn preferred_provider(&self) -> Option<&str> {
        self.preferred_provider.as_deref()
    }

    /// 根据凭证调用 LLM API
    ///
    /// # Arguments
    /// * `credential` - 选中的凭证
    /// * `system_prompt` - 系统提示词
    /// * `user_message` - 用户消息
    /// * `model` - 模型名称
    ///
    /// # Returns
    /// LLM 响应文本或错误
    async fn call_llm_with_credential(
        &self,
        credential: &RuntimeProviderCredential,
        system_prompt: &str,
        user_message: &str,
        model: &str,
    ) -> Result<String, SkillError> {
        let (api_key, base_url) = match &credential.credential {
            RuntimeCredentialData::OpenAIKey { api_key, base_url }
            | RuntimeCredentialData::ClaudeKey { api_key, base_url }
            | RuntimeCredentialData::AnthropicKey { api_key, base_url } => {
                (api_key.as_str(), base_url.as_deref())
            }
            RuntimeCredentialData::GeminiApiKey {
                api_key, base_url, ..
            } => (api_key.as_str(), base_url.as_deref()),
            RuntimeCredentialData::VertexKey { .. } => {
                return Err(SkillError::ProviderError(format!(
                    "当前 model-provider 不支持 runtime credential: {:?}",
                    credential.provider_type
                )));
            }
        };

        let protocol = match credential.provider_type {
            RuntimeProviderType::Claude
            | RuntimeProviderType::Anthropic
            | RuntimeProviderType::AnthropicCompatible => {
                RuntimeProviderProtocol::AnthropicMessages
            }
            RuntimeProviderType::OpenAI => RuntimeProviderProtocol::ChatCompletions,
            RuntimeProviderType::GeminiApiKey => RuntimeProviderProtocol::GeminiGenerateContent,
            unsupported => {
                return Err(SkillError::ProviderError(format!(
                    "当前 model-provider 不支持 runtime provider: {unsupported}"
                )));
            }
        };

        let client = CurrentProviderClient::new(RuntimeProviderConfig {
            provider_name: match protocol {
                RuntimeProviderProtocol::AnthropicMessages => "anthropic".to_string(),
                RuntimeProviderProtocol::GeminiGenerateContent => "google".to_string(),
                _ => "openai".to_string(),
            },
            provider_selector: Some(credential.provider_type.to_string()),
            model_name: model.to_string(),
            api_key: Some(api_key.to_string()),
            auth: RuntimeProviderAuth::ApiKey,
            base_url: base_url.map(ToOwned::to_owned),
            credential_uuid: credential.uuid.clone(),
            reasoning_effort: None,
            service_tier: None,
            protocol: Some(protocol),
            supports_websockets: false,
            toolshim: false,
            toolshim_model: None,
        })
        .map_err(|error| SkillError::ProviderError(error.to_string()))?;

        let request = CurrentProviderRequest::new(vec![CurrentProviderMessage::user(vec![
            CurrentProviderContent::Text(user_message.to_string()),
        ])])
        .with_system_prompt((!system_prompt.trim().is_empty()).then(|| system_prompt.to_string()))
        .with_generation(model_provider::current_client::GenerationOptions {
            max_tokens: Some(4096),
            ..Default::default()
        });

        let mut stream = client
            .stream(request)
            .await
            .map_err(|error| SkillError::ProviderError(error.to_string()))?;
        let mut content = String::new();
        while let Some(event) = stream.next().await {
            match event.map_err(|error| SkillError::ProviderError(error.to_string()))? {
                CanonicalLlmEvent::TextDelta { text, .. } => content.push_str(&text),
                CanonicalLlmEvent::ProviderError { message, .. } => {
                    return Err(SkillError::ProviderError(message));
                }
                CanonicalLlmEvent::Finish { .. } | CanonicalLlmEvent::StepFinish { .. } => break,
                _ => {}
            }
        }

        Ok(content)
    }
}

#[async_trait]
impl LlmProvider for LimeLlmProvider {
    /// 调用 LLM 进行对话
    ///
    /// # 实现说明
    /// 1. 使用 API Key Provider 选择凭证
    /// 2. 如果指定了 preferred_provider，优先选择该类型的凭证
    /// 3. 如果指定了 model，传递给底层 provider
    /// 4. 如果没有可用凭证，返回 ProviderError
    ///
    /// # Requirements
    /// - 1.2: 使用 API Key Provider 选择可用凭证
    /// - 1.3: 优先选择指定 provider 类型的凭证
    /// - 1.4: 将 model 参数传递给底层 provider
    /// - 1.5: 没有可用凭证时返回 ProviderError
    async fn chat(
        &self,
        system_prompt: &str,
        user_message: &str,
        model: Option<&str>,
    ) -> Result<String, SkillError> {
        // 确定要使用的 provider 类型
        let provider_type = self.preferred_provider.as_deref().unwrap_or("claude"); // 默认使用 Claude

        // 确定要使用的模型
        let model_name = model.unwrap_or("claude-sonnet-4-5-20250514");

        tracing::info!(
            "[LimeLlmProvider] chat 调用: provider_type={}, model={}",
            provider_type,
            model_name
        );

        // 使用 API Key Provider 选择凭证（Requirements 1.2, 1.3）
        let credential = self
            .api_key_service
            .select_credential_for_provider(
                &self.db,
                provider_type,
                Some(provider_type),
                None, // client_type
            )
            .await
            .map_err(|e| SkillError::ProviderError(format!("选择凭证失败: {}", e)))?
            .ok_or_else(|| {
                // Requirements 1.5: 没有可用凭证时返回 ProviderError
                SkillError::ProviderError(format!(
                    "没有可用的凭证: provider_type={}, model={}",
                    provider_type, model_name
                ))
            })?;

        tracing::info!(
            "[LimeLlmProvider] 选中凭证: uuid={}, type={:?}",
            &credential.uuid[..8],
            credential.provider_type
        );

        // 调用 LLM API（Requirements 1.4: 传递 model 参数）
        let result = self
            .call_llm_with_credential(&credential, system_prompt, user_message, model_name)
            .await;

        // 记录使用情况
        match &result {
            Ok(_) => {
                if let Some(api_key_id) = runtime_api_key_id_from_credential_uuid(&credential.uuid)
                {
                    let _ = self.api_key_service.record_usage(&self.db, api_key_id);
                }
            }
            Err(e) => {
                tracing::debug!(
                    "[LimeLlmProvider] 调用失败，API Key Provider 不写回旧 credential 健康状态: {}",
                    e
                );
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_error_display() {
        let provider_err = SkillError::ProviderError("没有可用凭证".to_string());
        assert!(provider_err.to_string().contains("Provider error"));
        assert!(provider_err.to_string().contains("没有可用凭证"));

        let exec_err = SkillError::ExecutionError("执行失败".to_string());
        assert!(exec_err.to_string().contains("Execution error"));

        let config_err = SkillError::ConfigError("配置错误".to_string());
        assert!(config_err.to_string().contains("Config error"));
    }
}
