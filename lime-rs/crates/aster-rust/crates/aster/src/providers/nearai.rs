//! NEAR AI Cloud Provider（OpenAI 兼容 API）
//!
//! NEAR AI Cloud 提供 OpenAI Chat Completions 兼容的 TEE 推理网关，并通过
//! 公开 `/v1/model/list` 目录暴露可用模型。

use super::base::{ConfigKey, MessageStream, ModelInfo, Provider, ProviderMetadata, ProviderUsage};
use super::errors::ProviderError;
use super::openai::OpenAiProvider;
use crate::config::declarative_providers::{DeclarativeProviderConfig, ProviderEngine};
use crate::conversation::message::Message;
use crate::model::ModelConfig;
use anyhow::Result;
use async_trait::async_trait;
use rmcp::model::Tool;
use serde::Deserialize;

pub const NEARAI_DEFAULT_MODEL: &str = "zai-org/GLM-5.1-FP8";
pub const NEARAI_DEFAULT_FAST_MODEL: &str = "google/gemma-4-31B-it";
pub const NEARAI_DEFAULT_HOST: &str = "https://cloud-api.near.ai";
pub const NEARAI_DEFAULT_BASE_PATH: &str = "v1/chat/completions";
pub const NEARAI_MODEL_LIST_PATH: &str = "v1/model/list";
pub const NEARAI_DOC_URL: &str = "https://cloud.near.ai";

#[derive(Debug, serde::Serialize)]
pub struct NearAiProvider {
    #[serde(skip)]
    inner: OpenAiProvider,
    model_list_url: String,
}

impl NearAiProvider {
    pub async fn from_env(model: ModelConfig) -> Result<Self> {
        let model = model.with_fast(NEARAI_DEFAULT_FAST_MODEL.to_string());

        let config = crate::config::Config::global();
        let host: String = config
            .get_param("NEARAI_HOST")
            .unwrap_or_else(|_| NEARAI_DEFAULT_HOST.to_string());
        let base_path: String = config
            .get_param("NEARAI_BASE_PATH")
            .unwrap_or_else(|_| NEARAI_DEFAULT_BASE_PATH.to_string());
        let timeout_seconds: u64 = config.get_param("NEARAI_TIMEOUT").unwrap_or(600);

        let provider_config = DeclarativeProviderConfig {
            name: "nearai".to_string(),
            engine: ProviderEngine::OpenAI,
            display_name: "NEAR AI Cloud".to_string(),
            description: Some("NEAR AI Cloud TEE inference provider".to_string()),
            api_key_env: "NEARAI_API_KEY".to_string(),
            base_url: Self::build_chat_base_url(&host, &base_path),
            models: Self::known_models(),
            headers: None,
            timeout_seconds: Some(timeout_seconds),
            supports_streaming: Some(true),
        };

        Ok(Self {
            inner: OpenAiProvider::from_custom_config(model, provider_config)?,
            model_list_url: Self::build_model_list_url(&host),
        })
    }

    fn known_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo::with_cost("zai-org/GLM-5.1-FP8", 202_752, 850e-9, 3300e-9),
            ModelInfo::with_cost("Qwen/Qwen3.6-35B-A3B-FP8", 262_144, 170e-9, 1100e-9),
            ModelInfo::with_cost("google/gemma-4-31B-it", 262_144, 130e-9, 400e-9),
            ModelInfo::with_cost("Qwen/Qwen3-VL-30B-A3B-Instruct", 256_000, 150e-9, 550e-9),
        ]
    }

    fn trim_url_parts(host: &str, path: &str) -> String {
        let host = host.trim().trim_end_matches('/');
        let path = path.trim().trim_matches('/');

        if path.is_empty() {
            host.to_string()
        } else {
            format!("{host}/{path}")
        }
    }

    fn build_chat_base_url(host: &str, base_path: &str) -> String {
        Self::trim_url_parts(host, base_path)
    }

    fn build_model_list_url(host: &str) -> String {
        let host = host.trim().trim_end_matches('/');

        if host.ends_with("/model/list") {
            return host.to_string();
        }

        if let Some((prefix, _)) = host.split_once("/v1") {
            return format!("{prefix}/{NEARAI_MODEL_LIST_PATH}");
        }

        Self::trim_url_parts(host, NEARAI_MODEL_LIST_PATH)
    }

    fn parse_model_list(body: &str) -> Result<Vec<String>, ProviderError> {
        let response: NearAiModelListResponse = serde_json::from_str(body).map_err(|e| {
            ProviderError::UsageError(format!("Failed to parse NEAR AI model list: {e}"))
        })?;

        let mut models: Vec<String> = response
            .models
            .into_iter()
            .filter(|model| model.is_chat_model())
            .map(|model| model.model_id)
            .collect();
        models.sort();
        Ok(models)
    }
}

#[async_trait]
impl Provider for NearAiProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::with_models(
            "nearai",
            "NEAR AI Cloud",
            "NEAR AI Cloud TEE inference through an OpenAI-compatible API",
            NEARAI_DEFAULT_MODEL,
            Self::known_models(),
            NEARAI_DOC_URL,
            vec![
                ConfigKey::new("NEARAI_API_KEY", true, true, None),
                ConfigKey::new("NEARAI_HOST", false, false, Some(NEARAI_DEFAULT_HOST)),
                ConfigKey::new(
                    "NEARAI_BASE_PATH",
                    false,
                    false,
                    Some(NEARAI_DEFAULT_BASE_PATH),
                ),
                ConfigKey::new("NEARAI_TIMEOUT", false, false, Some("600")),
            ],
        )
    }

    fn get_name(&self) -> &str {
        self.inner.get_name()
    }

    fn get_model_config(&self) -> ModelConfig {
        self.inner.get_model_config()
    }

    async fn complete_with_model(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        self.inner
            .complete_with_model(model_config, system, messages, tools)
            .await
    }

    async fn fetch_supported_models(&self) -> Result<Option<Vec<String>>, ProviderError> {
        let response = reqwest::Client::new()
            .get(&self.model_list_url)
            .send()
            .await
            .map_err(|e| ProviderError::RequestFailed(e.to_string()))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| ProviderError::RequestFailed(e.to_string()))?;

        if !status.is_success() {
            return Err(ProviderError::RequestFailed(format!(
                "NEAR AI model list request failed with status {status}: {body}"
            )));
        }

        Self::parse_model_list(&body).map(Some)
    }

    async fn stream(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        self.inner.stream(system, messages, tools).await
    }

    fn supports_streaming(&self) -> bool {
        self.inner.supports_streaming()
    }
}

#[derive(Debug, Deserialize)]
struct NearAiModelListResponse {
    #[serde(default)]
    models: Vec<NearAiModel>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NearAiModel {
    model_id: String,
    #[serde(default)]
    metadata: Option<NearAiModelMetadata>,
}

impl NearAiModel {
    fn is_chat_model(&self) -> bool {
        if self
            .model_id
            .trim()
            .eq_ignore_ascii_case("openai/privacy-filter")
            || self
                .model_id
                .trim()
                .to_ascii_lowercase()
                .contains("reranker")
        {
            return false;
        }

        let Some(metadata) = self.metadata.as_ref() else {
            return false;
        };
        let Some(architecture) = metadata.architecture.as_ref() else {
            return false;
        };

        architecture
            .output_modalities
            .iter()
            .any(|modality| modality.eq_ignore_ascii_case("text"))
            && architecture
                .input_modalities
                .iter()
                .any(|modality| modality.eq_ignore_ascii_case("text"))
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NearAiModelMetadata {
    #[serde(default)]
    architecture: Option<NearAiModelArchitecture>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NearAiModelArchitecture {
    #[serde(default)]
    input_modalities: Vec<String>,
    #[serde(default)]
    output_modalities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use crate::providers::base::Provider;

    use super::{NearAiProvider, NEARAI_DEFAULT_BASE_PATH, NEARAI_DEFAULT_HOST};

    #[test]
    fn metadata_uses_nearai_credentials_and_gateway_defaults() {
        let metadata = NearAiProvider::metadata();

        assert_eq!(metadata.name, "nearai");
        assert_eq!(metadata.display_name, "NEAR AI Cloud");
        assert_eq!(metadata.config_keys[0].name, "NEARAI_API_KEY");
        assert_eq!(
            metadata.config_keys[1].default.as_deref(),
            Some(NEARAI_DEFAULT_HOST)
        );
        assert_eq!(
            metadata.config_keys[2].default.as_deref(),
            Some(NEARAI_DEFAULT_BASE_PATH)
        );
    }

    #[test]
    fn build_model_list_url_uses_nearai_public_catalog_path() {
        assert_eq!(
            NearAiProvider::build_model_list_url("https://cloud-api.near.ai/v1"),
            "https://cloud-api.near.ai/v1/model/list"
        );
        assert_eq!(
            NearAiProvider::build_model_list_url("https://cloud-api.near.ai/v1/chat/completions"),
            "https://cloud-api.near.ai/v1/model/list"
        );
    }

    #[test]
    fn parse_model_list_filters_non_chat_catalog_entries() {
        let models = NearAiProvider::parse_model_list(
            r#"{
              "models": [
                {
                  "modelId": "zai-org/GLM-5.1-FP8",
                  "metadata": {
                    "architecture": {
                      "inputModalities": ["text"],
                      "outputModalities": ["text"]
                    }
                  }
                },
                {
                  "modelId": "Qwen/Qwen3-Embedding-0.6B",
                  "metadata": {
                    "architecture": {
                      "inputModalities": ["text"],
                      "outputModalities": ["embedding"]
                    }
                  }
                },
                {
                  "modelId": "openai/privacy-filter",
                  "metadata": {
                    "architecture": {
                      "inputModalities": ["text"],
                      "outputModalities": ["text"]
                    }
                  }
                }
              ]
            }"#,
        )
        .expect("nearai model list should parse");

        assert_eq!(models, vec!["zai-org/GLM-5.1-FP8".to_string()]);
    }
}
