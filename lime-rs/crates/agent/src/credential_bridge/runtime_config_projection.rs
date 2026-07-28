use lime_core::database::dao::api_key_provider::ApiProviderType;
use lime_core::models::{RuntimeCredentialData, RuntimeProviderCredential};
use model_provider::runtime_provider::{RuntimeProviderAuth, RuntimeProviderConfig};

use super::provider_mapping::{normalize_provider_selector, resolve_runtime_provider_name};

pub(super) fn runtime_provider_config_from_credential(
    credential: &RuntimeProviderCredential,
    model: &str,
    provider_type_hint: &str,
    resolved_api_type: Option<ApiProviderType>,
) -> RuntimeProviderConfig {
    let (provider_name, api_key, base_url) = match &credential.credential {
        RuntimeCredentialData::OpenAIKey { api_key, base_url } => {
            let provider = resolve_runtime_provider_name(provider_type_hint, resolved_api_type);
            tracing::info!(
                "[CredentialBridge] OpenAIKey: provider_type_hint={}, resolved_api_type={:?} -> runtime_provider={}",
                provider_type_hint,
                resolved_api_type,
                provider
            );
            (
                provider.to_string(),
                Some(api_key.clone()),
                base_url.clone(),
            )
        }
        RuntimeCredentialData::ClaudeKey { api_key, base_url }
        | RuntimeCredentialData::AnthropicKey { api_key, base_url } => (
            "anthropic".to_string(),
            Some(api_key.clone()),
            base_url.clone(),
        ),
        RuntimeCredentialData::GeminiApiKey {
            api_key, base_url, ..
        } => (
            "google".to_string(),
            Some(api_key.clone()),
            base_url.clone(),
        ),
        RuntimeCredentialData::VertexKey {
            api_key, base_url, ..
        } => (
            "gcpvertexai".to_string(),
            Some(api_key.clone()),
            base_url.clone(),
        ),
    };

    RuntimeProviderConfig {
        provider_name,
        provider_selector: normalize_provider_selector(Some(provider_type_hint)),
        model_name: model.to_string(),
        api_key,
        auth: RuntimeProviderAuth::ApiKey,
        base_url,
        api_version: None,
        credential_uuid: credential.uuid.clone(),
        reasoning_effort: None,
        service_tier: None,
        protocol: None,
        supports_websockets: resolved_api_type
            .is_some_and(|provider_type| provider_type.runtime_spec().supports_websockets),
        toolshim: false,
        toolshim_model: None,
    }
}

#[cfg(test)]
mod tests {
    use super::runtime_provider_config_from_credential;
    use lime_core::database::dao::api_key_provider::ApiProviderType;
    use lime_core::models::{
        RuntimeCredentialData, RuntimeProviderCredential, RuntimeProviderType,
    };

    fn openai_credential() -> RuntimeProviderCredential {
        RuntimeProviderCredential::new(
            RuntimeProviderType::OpenAI,
            RuntimeCredentialData::OpenAIKey {
                api_key: "test-key".to_string(),
                base_url: Some("https://api.openai.com/v1".to_string()),
            },
        )
    }

    #[test]
    fn provider_runtime_spec_controls_websocket_capability() {
        let credential = openai_credential();
        let openai = runtime_provider_config_from_credential(
            &credential,
            "gpt-5.4",
            "openai",
            Some(ApiProviderType::Openai),
        );
        let gateway = runtime_provider_config_from_credential(
            &credential,
            "gpt-5.4",
            "gateway",
            Some(ApiProviderType::Gateway),
        );

        assert!(openai.supports_websockets);
        assert!(!gateway.supports_websockets);
    }
}
