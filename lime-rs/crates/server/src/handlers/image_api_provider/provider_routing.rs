use lime_core::api_host_utils::is_openai_responses_compatible_host;
use lime_core::database::dao::api_key_provider::{ApiKeyProvider, ApiProviderType};
use lime_core::image_generation_matcher::{
    is_likely_fal_image_model_id, is_likely_image_generation_model_id,
};
use lime_core::models::model_registry::ProviderModelConfig;
use lime_core::models::openai::ImageGenerationRequest;

use super::{ConfiguredImageProviderKind, ImageProviderRoutingConfig};

pub(super) fn resolve_configured_image_provider_kind(
    provider: &ApiKeyProvider,
    request: &ImageGenerationRequest,
    routing: &ImageProviderRoutingConfig,
) -> Option<ConfiguredImageProviderKind> {
    if is_fal_provider(provider) {
        return Some(ConfiguredImageProviderKind::Fal);
    }

    if is_gemini_image_provider(provider) {
        return Some(ConfiguredImageProviderKind::Gemini);
    }

    if is_zhipu_image_provider(provider) {
        return Some(ConfiguredImageProviderKind::Zhipu);
    }

    if supports_openai_compatible_image_provider(provider)
        && resolve_compatible_image_model(
            request.model.as_str(),
            routing.preferred_model_id.as_deref(),
            &provider.models,
        )
        .is_some()
    {
        return Some(ConfiguredImageProviderKind::OpenAiCompatible);
    }

    None
}

pub(super) fn is_gemini_image_provider(provider: &ApiKeyProvider) -> bool {
    if matches!(
        provider.effective_provider_type(),
        ApiProviderType::Gemini | ApiProviderType::Vertexai
    ) {
        return true;
    }

    let normalized_provider_id = provider.id.trim().to_ascii_lowercase();
    let normalized_host = provider.api_host.trim().to_ascii_lowercase();

    matches!(
        normalized_provider_id.as_str(),
        "gemini" | "google" | "vertexai" | "google-vertex"
    ) || normalized_host.contains("generativelanguage.googleapis.com")
}

pub(super) fn supports_openai_compatible_image_provider(provider: &ApiKeyProvider) -> bool {
    if is_catalog_zhipu_image_provider(provider) {
        return false;
    }

    matches!(
        provider.effective_provider_type(),
        ApiProviderType::Openai
            | ApiProviderType::OpenaiResponse
            | ApiProviderType::Codex
            | ApiProviderType::NewApi
            | ApiProviderType::Gateway
    ) || is_catalog_openai_compatible_image_provider(provider)
}

pub(super) fn resolve_compatible_image_model(
    request_model: &str,
    preferred_model_id: Option<&str>,
    models: &[ProviderModelConfig],
) -> Option<String> {
    let normalized_request_model = request_model.trim();
    if is_likely_image_generation_model_id(normalized_request_model) {
        return Some(normalized_request_model.to_string());
    }

    if let Some(preferred_model) = preferred_model_id
        .map(str::trim)
        .filter(|value| is_likely_image_generation_model_id(value))
    {
        return Some(preferred_model.to_string());
    }

    models
        .iter()
        .map(|model| model.id.trim())
        .find(|model| is_likely_image_generation_model_id(model))
        .map(|model| model.to_string())
}

pub(super) fn resolve_fal_model(request_model: &str, preferred_model_id: Option<&str>) -> String {
    let trimmed = request_model.trim();
    if trimmed.is_empty() {
        return normalize_preferred_fal_model(preferred_model_id)
            .unwrap_or_else(|| super::FAL_DEFAULT_MODEL.to_string());
    }

    if !is_likely_fal_image_model_id(trimmed) {
        return normalize_preferred_fal_model(preferred_model_id)
            .unwrap_or_else(|| super::FAL_DEFAULT_MODEL.to_string());
    }

    normalize_fal_model(trimmed)
}

pub(super) fn resolve_openai_responses_image_orchestration_model(
    provider: &ApiKeyProvider,
    image_model: &str,
) -> String {
    let trimmed = image_model.trim();
    if !trimmed.is_empty() && !is_likely_image_generation_model_id(trimmed) {
        return trimmed.to_string();
    }

    provider
        .models
        .iter()
        .map(|model| model.id.as_str())
        .map(str::trim)
        .find(|candidate| !candidate.is_empty() && !is_likely_image_generation_model_id(candidate))
        .map(ToString::to_string)
        .unwrap_or_else(|| super::OPENAI_RESPONSES_IMAGE_ORCHESTRATOR_MODEL.to_string())
}

pub(super) fn should_prefer_openai_responses_image_api(provider: &ApiKeyProvider) -> bool {
    matches!(
        provider.effective_provider_type(),
        ApiProviderType::OpenaiResponse | ApiProviderType::Codex
    ) || is_openai_responses_compatible_host(&provider.api_host)
}

pub(super) fn is_zhipu_image_provider(provider: &ApiKeyProvider) -> bool {
    is_catalog_zhipu_image_provider(provider)
}

fn is_fal_provider(provider: &ApiKeyProvider) -> bool {
    if provider.provider_type == ApiProviderType::Fal || provider.id == "fal" {
        return true;
    }

    let normalized_host = provider.api_host.trim().to_ascii_lowercase();
    normalized_host.contains("fal.run") || normalized_host.contains("queue.fal.run")
}

fn is_catalog_openai_compatible_image_provider(provider: &ApiKeyProvider) -> bool {
    let provider_id = provider.id.trim().to_ascii_lowercase();
    let api_host = provider.api_host.trim().to_ascii_lowercase();

    provider_id.contains("siliconflow")
        || provider_id.contains("dmxapi")
        || provider_id.contains("tokenflux")
        || provider_id.contains("aihubmix")
        || api_host.contains("siliconflow")
        || api_host.contains("dmxapi")
        || api_host.contains("tokenflux")
        || api_host.contains("aihubmix")
}

fn is_catalog_zhipu_image_provider(provider: &ApiKeyProvider) -> bool {
    let provider_id = provider.id.trim().to_ascii_lowercase();
    let api_host = provider.api_host.trim().to_ascii_lowercase();

    provider_id.contains("zhipu")
        || provider_id.contains("glm")
        || api_host.contains("bigmodel.cn/api/paas")
}

fn normalize_preferred_fal_model(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(normalize_fal_model)
}

fn normalize_fal_model(model: &str) -> String {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return super::FAL_DEFAULT_MODEL.to_string();
    }

    if trimmed.starts_with("fal-ai/") {
        trimmed.to_string()
    } else {
        format!("fal-ai/{trimmed}")
    }
}
