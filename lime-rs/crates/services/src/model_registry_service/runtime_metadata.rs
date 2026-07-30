use super::{FetchModelsResult, ModelRegistryService};
use lime_core::database::dao::api_key_provider::ProviderWithKeys;
use lime_core::models::model_registry::{EnhancedModelMetadata, ProviderModelConfig};
use lime_core::models::{RuntimeCredentialData, RuntimeProviderCredential};
use serde::Serialize;

#[derive(Clone, Copy)]
pub enum ProviderModelCacheAccess<'a> {
    Credential(&'a RuntimeProviderCredential),
    Keyless,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderModelRegistryMetadataSource {
    ProviderModelsCache,
    ProviderDeclaredModel,
    RuntimeSelectionOnly,
}

impl ProviderModelRegistryMetadataSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ProviderModelsCache => "provider_models_cache",
            Self::ProviderDeclaredModel => "provider_declared_model",
            Self::RuntimeSelectionOnly => "runtime_selection_only",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderModelRegistryMetadata {
    pub provider_id: String,
    pub requested_model_id: String,
    pub source: ProviderModelRegistryMetadataSource,
    pub reason_code: &'static str,
    pub matched_model_id: Option<String>,
    pub cached_model_count: Option<usize>,
    pub from_cache: bool,
    pub provider_declared_model: bool,
    pub model: Option<EnhancedModelMetadata>,
}

impl ProviderModelRegistryMetadata {
    fn runtime_selection_only(
        provider_id: &str,
        requested_model_id: &str,
        reason_code: &'static str,
        cached_model_count: Option<usize>,
    ) -> Self {
        Self {
            provider_id: provider_id.to_string(),
            requested_model_id: requested_model_id.to_string(),
            source: ProviderModelRegistryMetadataSource::RuntimeSelectionOnly,
            reason_code,
            matched_model_id: None,
            cached_model_count,
            from_cache: false,
            provider_declared_model: false,
            model: None,
        }
    }
}

impl ModelRegistryService {
    pub fn get_cached_provider_models_for_access(
        &self,
        provider_id: &str,
        api_host: &str,
        provider_type: Option<lime_core::database::dao::api_key_provider::ApiProviderType>,
        cache_access: ProviderModelCacheAccess<'_>,
    ) -> Result<Option<FetchModelsResult>, String> {
        let credential_fingerprint = match cache_access {
            ProviderModelCacheAccess::Credential(credential) => {
                Self::credential_cache_fingerprint(runtime_credential_api_key(credential))
            }
            ProviderModelCacheAccess::Keyless => None,
            ProviderModelCacheAccess::Unavailable => return Ok(None),
        };
        self.get_cached_provider_models_scoped(
            provider_id,
            api_host,
            provider_type,
            credential_fingerprint.as_deref(),
        )
    }

    pub fn resolve_provider_model_metadata(
        &self,
        provider: Option<&ProviderWithKeys>,
        provider_id: &str,
        model_id: &str,
        cache_access: ProviderModelCacheAccess<'_>,
    ) -> Result<ProviderModelRegistryMetadata, String> {
        let Some(provider) = provider else {
            return Ok(ProviderModelRegistryMetadata::runtime_selection_only(
                provider_id,
                model_id,
                "provider_not_configured",
                None,
            ));
        };

        let provider_id = provider.provider.id.as_str();
        let requested_model_id = model_id.trim();
        let provider_type = provider.provider.effective_provider_type();
        let api_host = provider.provider.api_host.trim();
        if !api_host.is_empty() {
            let cached = if matches!(cache_access, ProviderModelCacheAccess::Keyless)
                && Self::requires_api_key_for_runtime(provider_id, api_host, provider_type)
            {
                None
            } else {
                self.get_cached_provider_models_for_access(
                    provider_id,
                    api_host,
                    Some(provider_type),
                    cache_access,
                )?
            };
            if let Some(cached) = cached {
                let cached_model_count = Some(cached.models.len());
                if let Some(model) = find_model_metadata(&cached.models, requested_model_id) {
                    return Ok(ProviderModelRegistryMetadata {
                        provider_id: provider_id.to_string(),
                        requested_model_id: requested_model_id.to_string(),
                        source: ProviderModelRegistryMetadataSource::ProviderModelsCache,
                        reason_code: "matched_provider_models_cache",
                        matched_model_id: Some(model.id.clone()),
                        cached_model_count,
                        from_cache: cached.from_cache,
                        provider_declared_model: false,
                        model: Some(model.clone()),
                    });
                }

                if cached_model_count != Some(0) {
                    return Ok(ProviderModelRegistryMetadata::runtime_selection_only(
                        provider_id,
                        requested_model_id,
                        "provider_models_cache_missing_requested_model",
                        cached_model_count,
                    ));
                }
            }
        }

        if let Some(model_config) = find_declared_model(&provider.provider.models, model_id) {
            let model = self.build_declared_model_metadata(provider_id, model_config);
            return Ok(ProviderModelRegistryMetadata {
                provider_id: provider_id.to_string(),
                requested_model_id: requested_model_id.to_string(),
                source: ProviderModelRegistryMetadataSource::ProviderDeclaredModel,
                reason_code: "matched_provider_models",
                matched_model_id: Some(model.id.clone()),
                cached_model_count: None,
                from_cache: false,
                provider_declared_model: true,
                model: Some(model),
            });
        }

        Ok(ProviderModelRegistryMetadata::runtime_selection_only(
            provider_id,
            requested_model_id,
            "model_registry_metadata_missing",
            None,
        ))
    }
}

fn runtime_credential_api_key(credential: &RuntimeProviderCredential) -> &str {
    match &credential.credential {
        RuntimeCredentialData::OpenAIKey { api_key, .. }
        | RuntimeCredentialData::ClaudeKey { api_key, .. }
        | RuntimeCredentialData::VertexKey { api_key, .. }
        | RuntimeCredentialData::GeminiApiKey { api_key, .. }
        | RuntimeCredentialData::AnthropicKey { api_key, .. } => api_key,
    }
}

fn find_model_metadata<'a>(
    models: &'a [EnhancedModelMetadata],
    requested_model_id: &str,
) -> Option<&'a EnhancedModelMetadata> {
    let requested = normalized_model_id(requested_model_id)?;
    models.iter().find(|model| {
        model_id_candidates(model)
            .into_iter()
            .filter_map(|value| normalized_model_id(&value))
            .any(|candidate| candidate == requested)
    })
}

fn model_id_candidates(model: &EnhancedModelMetadata) -> Vec<String> {
    let mut candidates = vec![model.id.clone()];
    if let Some(value) = model.provider_model_id.clone() {
        candidates.push(value);
    }
    if let Some(value) = model.canonical_model_id.clone() {
        candidates.push(value);
    }
    candidates
}

fn find_declared_model<'a>(
    models: &'a [ProviderModelConfig],
    requested_model_id: &str,
) -> Option<&'a ProviderModelConfig> {
    let requested = normalized_model_id(requested_model_id)?;
    models
        .iter()
        .find(|model| normalized_model_id(&model.id).as_deref() == Some(requested.as_str()))
}

fn normalized_model_id(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lime_core::database::dao::api_key_provider::{
        ApiKeyProvider, ApiProviderType, ProviderGroup, ProviderWithKeys,
    };
    use lime_core::database::schema::create_tables;
    use lime_core::models::model_registry::{
        ModelCapabilities, ModelCapabilityProvenance, ModelModality, ModelReasoningEffortSource,
        ModelReasoningEffortSupport, ModelRuntimeFeature, ModelTaskFamily,
    };
    use lime_core::models::RuntimeProviderType;
    use rusqlite::Connection;
    use std::sync::{Arc, Mutex};

    fn service() -> ModelRegistryService {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        create_tables(&conn).expect("create schema");
        ModelRegistryService::new(Arc::new(Mutex::new(conn)))
    }

    fn provider(provider_id: &str, api_host: &str) -> ProviderWithKeys {
        let now = chrono::Utc::now();
        ProviderWithKeys {
            provider: ApiKeyProvider {
                id: provider_id.to_string(),
                name: provider_id.to_string(),
                provider_type: ApiProviderType::Openai,
                api_host: api_host.to_string(),
                is_system: false,
                group: ProviderGroup::Custom,
                enabled: true,
                sort_order: 1,
                api_version: None,
                project: None,
                location: None,
                region: None,
                models: Vec::new(),
                prompt_cache_mode: None,
                created_at: now,
                updated_at: now,
            },
            api_keys: Vec::new(),
        }
    }

    fn credential(id: &str, api_key: &str) -> RuntimeProviderCredential {
        RuntimeProviderCredential {
            uuid: id.to_string(),
            provider_type: RuntimeProviderType::OpenAI,
            credential: RuntimeCredentialData::OpenAIKey {
                api_key: api_key.to_string(),
                base_url: None,
            },
            name: None,
            prompt_cache_mode_override: None,
        }
    }

    #[test]
    fn resolves_cached_provider_model_by_provider_model_id() {
        let service = service();
        let provider = provider("cached-provider", "https://gateway.example.com/v1");
        let api_key = "sk-cache-a";
        let mut model = EnhancedModelMetadata::new(
            "stable-coder".to_string(),
            "Stable Coder".to_string(),
            "cached-provider".to_string(),
            "Cached Provider".to_string(),
        );
        model.provider_model_id = Some("upstream-coder".to_string());
        model.capabilities = ModelCapabilities {
            tools: true,
            streaming: true,
            reasoning: true,
            reasoning_effort: Some(ModelReasoningEffortSupport {
                supported: true,
                levels: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
                options: Vec::new(),
                default: Some("medium".to_string()),
                source: Some(ModelReasoningEffortSource::Api),
            }),
            ..Default::default()
        };
        model.task_families = vec![ModelTaskFamily::Chat, ModelTaskFamily::Reasoning];
        model.runtime_features = vec![
            ModelRuntimeFeature::Streaming,
            ModelRuntimeFeature::Reasoning,
        ];

        service
            .save_provider_models_cache_scoped(
                "cached-provider",
                "https://gateway.example.com/v1",
                Some(ApiProviderType::Openai),
                &[model],
                Some("https://gateway.example.com/v1/models".to_string()),
                chrono::Utc::now().timestamp(),
                ModelRegistryService::credential_cache_fingerprint(api_key).as_deref(),
            )
            .expect("save cache");

        let metadata = service
            .resolve_provider_model_metadata(
                Some(&provider),
                "cached-provider",
                "upstream-coder",
                ProviderModelCacheAccess::Credential(&credential("key-a", api_key)),
            )
            .expect("metadata");

        assert_eq!(
            metadata.source,
            ProviderModelRegistryMetadataSource::ProviderModelsCache
        );
        assert_eq!(metadata.reason_code, "matched_provider_models_cache");
        assert_eq!(metadata.matched_model_id.as_deref(), Some("stable-coder"));
        assert_eq!(metadata.cached_model_count, Some(1));
        assert!(metadata.from_cache);
        let model = metadata.model.expect("model metadata");
        assert_eq!(model.provider_model_id.as_deref(), Some("upstream-coder"));
        assert!(model.capabilities.reasoning);
        assert!(model.capabilities.reasoning_effort.is_some());
    }

    #[test]
    fn provider_model_metadata_does_not_cross_credential_scopes() {
        let service = service();
        let provider = provider("cached-provider", "https://gateway.example.com/v1");
        let model = EnhancedModelMetadata::new(
            "credential-a-model".to_string(),
            "Credential A Model".to_string(),
            "cached-provider".to_string(),
            "Cached Provider".to_string(),
        );
        let credential_a = "sk-cache-a";
        let runtime_credential_a = credential("key-a", credential_a);
        let runtime_credential_b = credential("key-b", "sk-cache-b");

        service
            .save_provider_models_cache_scoped(
                "cached-provider",
                "https://gateway.example.com/v1",
                Some(ApiProviderType::Openai),
                &[model],
                Some("https://gateway.example.com/v1/models".to_string()),
                chrono::Utc::now().timestamp(),
                ModelRegistryService::credential_cache_fingerprint(credential_a).as_deref(),
            )
            .expect("save credential A cache");

        let matching = service
            .resolve_provider_model_metadata(
                Some(&provider),
                "cached-provider",
                "credential-a-model",
                ProviderModelCacheAccess::Credential(&runtime_credential_a),
            )
            .expect("matching credential metadata");
        let isolated = service
            .resolve_provider_model_metadata(
                Some(&provider),
                "cached-provider",
                "credential-a-model",
                ProviderModelCacheAccess::Credential(&runtime_credential_b),
            )
            .expect("isolated credential metadata");

        assert_eq!(
            matching.source,
            ProviderModelRegistryMetadataSource::ProviderModelsCache
        );
        assert_eq!(isolated.reason_code, "model_registry_metadata_missing");
        assert!(isolated.model.is_none());
    }

    #[test]
    fn keyless_provider_can_read_unscoped_cache() {
        let service = service();
        let mut provider = provider("ollama", "http://127.0.0.1:11434");
        provider.provider.provider_type = ApiProviderType::Ollama;
        let model = EnhancedModelMetadata::new(
            "qwen3".to_string(),
            "Qwen 3".to_string(),
            "ollama".to_string(),
            "Ollama".to_string(),
        );

        service
            .save_provider_models_cache(
                "ollama",
                "http://127.0.0.1:11434",
                Some(ApiProviderType::Ollama),
                &[model],
                Some("http://127.0.0.1:11434/api/tags".to_string()),
                chrono::Utc::now().timestamp(),
            )
            .expect("save keyless cache");

        let metadata = service
            .resolve_provider_model_metadata(
                Some(&provider),
                "ollama",
                "qwen3",
                ProviderModelCacheAccess::Keyless,
            )
            .expect("keyless metadata");

        assert_eq!(
            metadata.source,
            ProviderModelRegistryMetadataSource::ProviderModelsCache
        );
    }

    #[test]
    fn key_required_provider_does_not_read_unscoped_cache() {
        let service = service();
        let provider = provider("cached-provider", "https://gateway.example.com/v1");
        let model = EnhancedModelMetadata::new(
            "credential-model".to_string(),
            "Credential Model".to_string(),
            "cached-provider".to_string(),
            "Cached Provider".to_string(),
        );

        service
            .save_provider_models_cache(
                "cached-provider",
                "https://gateway.example.com/v1",
                Some(ApiProviderType::Openai),
                &[model],
                Some("https://gateway.example.com/v1/models".to_string()),
                chrono::Utc::now().timestamp(),
            )
            .expect("save unscoped cache");

        let metadata = service
            .resolve_provider_model_metadata(
                Some(&provider),
                "cached-provider",
                "credential-model",
                ProviderModelCacheAccess::Unavailable,
            )
            .expect("credential metadata");

        assert_eq!(metadata.reason_code, "model_registry_metadata_missing");
        assert!(metadata.model.is_none());
    }

    #[test]
    fn lime_hub_declared_canonical_model_keeps_executable_capability() {
        let service = service();
        let mut provider = provider(
            "lime-hub",
            "https://llm.limeai.run#lime_tenant_id=tenant-test",
        );
        provider.provider.models = vec![ProviderModelConfig::hint("gpt-5.2-pro")];

        let metadata = service
            .resolve_provider_model_metadata(
                Some(&provider),
                "lime-hub",
                "gpt-5.2-pro",
                ProviderModelCacheAccess::Unavailable,
            )
            .expect("lime-hub model metadata");

        assert_eq!(
            metadata.source,
            ProviderModelRegistryMetadataSource::ProviderDeclaredModel
        );
        let model = metadata.model.expect("canonical model metadata");
        assert_eq!(
            model.capability_provenance,
            ModelCapabilityProvenance::Canonical
        );
        assert_eq!(
            model.canonical_model_id.as_deref(),
            Some("openai/gpt-5.2-pro")
        );
        assert!(model.task_families.contains(&ModelTaskFamily::Chat));
        assert!(model.input_modalities.contains(&ModelModality::Text));
        assert!(model.output_modalities.contains(&ModelModality::Text));
        assert!(model
            .runtime_features
            .contains(&ModelRuntimeFeature::Streaming));
    }
}
