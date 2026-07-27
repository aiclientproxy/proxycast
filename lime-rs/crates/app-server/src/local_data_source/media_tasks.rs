use crate::media_task;
use crate::media_task_payload::{image_model_task_request, video_model_task_request};
use crate::model_route_assembly::{resolved_route_from_task_with_credential, ModelRouteSelection};
use crate::model_task_contract::{
    capability_snapshot_from_model_capabilities, MediaRouteAssessment,
};
use crate::runtime::SidecarStore;
use crate::runtime_backend::model_route_credential;
use app_server_protocol::MediaTaskArtifactAudioCompleteParams;
use app_server_protocol::MediaTaskArtifactAudioCreateParams;
use app_server_protocol::MediaTaskArtifactImageCompleteParams;
use app_server_protocol::MediaTaskArtifactImageCreateParams;
use app_server_protocol::MediaTaskArtifactListParams;
use app_server_protocol::MediaTaskArtifactListResponse;
use app_server_protocol::MediaTaskArtifactLookupParams;
use app_server_protocol::MediaTaskArtifactResponse;
use app_server_protocol::MediaTaskArtifactVideoCreateParams;
use app_server_protocol::{ModelRefSource, ModelTaskRequest};
use lime_core::config::{Config, ConfigManager};
use lime_core::database::dao::api_key_provider::{
    ApiKeyProvider, ApiProviderType, ProviderProtocolFamily, ProviderWithKeys,
};
use lime_core::database::DbConnection;
use lime_core::models::model_registry::{EnhancedModelMetadata, ModelCapabilityProvenance};
use lime_services::api_key_provider_service::ApiKeyProviderService;
use lime_services::model_registry_service::{ModelRegistryService, ProviderModelRegistryMetadata};
use serde_json::{json, Value};

pub(crate) fn create_image_media_task_artifact(
    params: MediaTaskArtifactImageCreateParams,
    route_assessment: MediaRouteAssessment,
) -> Result<MediaTaskArtifactResponse, String> {
    media_task::create_image_generation_task_artifact(params, Some(route_assessment))
}

pub(crate) fn create_audio_media_task_artifact(
    params: MediaTaskArtifactAudioCreateParams,
) -> Result<MediaTaskArtifactResponse, String> {
    media_task::create_audio_generation_task_artifact(params)
}

pub(crate) fn create_video_media_task_artifact(
    params: MediaTaskArtifactVideoCreateParams,
    route_assessment: MediaRouteAssessment,
) -> Result<MediaTaskArtifactResponse, String> {
    media_task::create_video_generation_task_artifact(params, Some(route_assessment))
}

pub(crate) fn complete_audio_media_task_artifact(
    params: MediaTaskArtifactAudioCompleteParams,
    sidecar_store: Option<&SidecarStore>,
) -> Result<MediaTaskArtifactResponse, String> {
    media_task::complete_audio_generation_task_artifact(params, sidecar_store)
}

pub(crate) async fn complete_image_media_task_artifact(
    params: MediaTaskArtifactImageCompleteParams,
    sidecar_store: Option<&SidecarStore>,
) -> Result<MediaTaskArtifactResponse, String> {
    media_task::complete_image_generation_task_artifact(params, sidecar_store).await
}

pub(crate) fn get_media_task_artifact(
    params: MediaTaskArtifactLookupParams,
) -> Result<MediaTaskArtifactResponse, String> {
    media_task::get_media_task_artifact(params)
}

pub(crate) fn list_media_task_artifacts(
    params: MediaTaskArtifactListParams,
) -> Result<MediaTaskArtifactListResponse, String> {
    media_task::list_media_task_artifacts(params)
}

pub(crate) fn cancel_media_task_artifact(
    params: MediaTaskArtifactLookupParams,
) -> Result<MediaTaskArtifactResponse, String> {
    media_task::cancel_media_task_artifact(params)
}

pub(crate) async fn assess_image_route(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    model_registry_service: &ModelRegistryService,
    params: &MediaTaskArtifactImageCreateParams,
) -> Result<MediaRouteAssessment, String> {
    assess_media_route(
        db,
        api_key_provider_service,
        model_registry_service,
        &image_model_task_request(params),
    )
    .await
}

pub(crate) fn normalize_image_create_params_for_task_submission(
    params: MediaTaskArtifactImageCreateParams,
) -> Result<NormalizedImageCreateParams, String> {
    normalize_image_create_params_with_defaults(params, configured_image_generation_defaults())
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NormalizedImageCreateParams {
    pub(crate) params: MediaTaskArtifactImageCreateParams,
    pub(crate) provider_from_defaults: bool,
    pub(crate) model_from_defaults: bool,
}

impl NormalizedImageCreateParams {
    fn new(
        params: MediaTaskArtifactImageCreateParams,
        provider_from_defaults: bool,
        model_from_defaults: bool,
    ) -> Self {
        Self {
            params,
            provider_from_defaults,
            model_from_defaults,
        }
    }
}

fn normalize_image_create_params_with_defaults(
    params: MediaTaskArtifactImageCreateParams,
    defaults: ImageGenerationDefaults,
) -> Result<NormalizedImageCreateParams, String> {
    let mut normalized = params;
    let explicit_provider_id = normalize_optional_task_field(normalized.provider_id.clone());
    let explicit_model = normalize_optional_task_field(normalized.model.clone());
    let provider_from_defaults = explicit_provider_id.is_none() && defaults.provider_id.is_some();
    let model_from_defaults = explicit_model.is_none() && defaults.model_id.is_some();

    normalized.mode = normalize_image_task_mode(normalized.mode);
    normalized.provider_id = explicit_provider_id.or(defaults.provider_id);
    normalized.model = explicit_model.or(defaults.model_id);
    normalized.executor_mode = normalize_image_executor_mode(normalized.executor_mode);
    normalized.outer_model = normalize_optional_task_field(normalized.outer_model);
    normalized.session_id = normalize_optional_task_field(normalized.session_id);
    normalized.thread_id = normalize_optional_task_field(normalized.thread_id);
    normalized.turn_id = normalize_optional_task_field(normalized.turn_id);
    normalized.project_id = normalize_optional_task_field(normalized.project_id);
    normalized.modality_contract_key =
        normalize_optional_task_field(normalized.modality_contract_key);
    normalized.modality = normalize_optional_task_field(normalized.modality);
    normalized.routing_slot = normalize_optional_task_field(normalized.routing_slot);
    normalized.requested_target = normalize_optional_task_field(normalized.requested_target);
    normalized.slot_id = normalize_optional_task_field(normalized.slot_id);
    normalized.target_output_id = normalize_optional_task_field(normalized.target_output_id);
    normalized.target_output_ref_id =
        normalize_optional_task_field(normalized.target_output_ref_id);

    if normalized.provider_id.is_none() || normalized.model.is_none() {
        return Err(
            "图片生成缺少默认 Provider 或模型，请在 设置 -> 媒体生成 -> 图片服务模型 选择可用图片模型后重试。"
                .to_string(),
        );
    }

    Ok(NormalizedImageCreateParams::new(
        normalized,
        provider_from_defaults,
        model_from_defaults,
    ))
}

pub(crate) fn resolve_image_provider_for_task_submission(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    normalized: NormalizedImageCreateParams,
) -> Result<MediaTaskArtifactImageCreateParams, String> {
    let mut params = normalized.params;
    let provider_id = params
        .provider_id
        .as_deref()
        .ok_or_else(image_model_ref_missing_message)?;
    let model = params
        .model
        .as_deref()
        .ok_or_else(image_model_ref_missing_message)?;
    let provider = api_key_provider_service
        .get_provider(db, provider_id)
        .map_err(|error| format!("读取图片 Provider 失败: {error}"))?;

    if let Some(provider) = provider {
        validate_image_provider_for_submission(&provider, model)?;
        return Ok(params);
    }

    if !normalized.provider_from_defaults && !normalized.model_from_defaults {
        return Err(format!(
            "图片 Provider {provider_id} 不存在，请重新选择图片模型后重试。"
        ));
    }

    let fallback = select_ready_image_provider_fallback(db, api_key_provider_service)
        .map_err(|error| format!("读取图片 Provider 失败: {error}"))?
        .ok_or_else(|| {
            format!(
                "默认图片 Provider {provider_id} 不存在，且没有其他已启用且带 API Key 的图片 Provider。请在 设置 -> 媒体生成 -> 图片服务模型 选择可用图片模型后重试。"
            )
        })?;
    params.provider_id = Some(fallback.provider_id);
    params.model = Some(fallback.model);
    Ok(params)
}

fn image_model_ref_missing_message() -> String {
    "图片生成缺少默认 Provider 或模型，请在 设置 -> 媒体生成 -> 图片服务模型 选择可用图片模型后重试。"
        .to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ImageProviderFallback {
    provider_id: String,
    model: String,
}

fn select_ready_image_provider_fallback(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
) -> Result<Option<ImageProviderFallback>, String> {
    let providers = api_key_provider_service.get_all_providers(db)?;
    Ok(providers
        .into_iter()
        .filter(|provider| image_provider_is_ready(&provider))
        .find_map(|provider| {
            first_image_model_for_provider(&provider.provider).map(|model| ImageProviderFallback {
                provider_id: provider.provider.id,
                model,
            })
        }))
}

fn validate_image_provider_for_submission(
    provider: &ProviderWithKeys,
    model: &str,
) -> Result<(), String> {
    if !provider.provider.enabled {
        return Err(format!(
            "图片 Provider {} 当前未启用，请重新选择图片模型后重试。",
            provider.provider.id
        ));
    }
    if !provider.api_keys.iter().any(|key| key.enabled) {
        return Err(format!(
            "图片 Provider {} 没有可用 API Key，请在设置中启用 API Key 后重试。",
            provider.provider.id
        ));
    }
    if !image_provider_has_supported_endpoint(&provider.provider, Some(model)) {
        return Err(format!(
            "图片 Provider {} 不支持当前图片模型 {}，请重新选择图片模型后重试。",
            provider.provider.id, model
        ));
    }
    if is_placeholder_provider_host(&provider.provider.api_host) {
        return Err(format!(
            "图片 Provider {} 的 API Host 是示例地址，请配置真实地址后重试。",
            provider.provider.id
        ));
    }
    Ok(())
}

fn image_provider_is_ready(provider: &ProviderWithKeys) -> bool {
    provider.provider.enabled
        && provider.api_keys.iter().any(|key| key.enabled)
        && !is_placeholder_provider_host(&provider.provider.api_host)
        && first_image_model_for_provider(&provider.provider).is_some()
}

fn first_image_model_for_provider(provider: &ApiKeyProvider) -> Option<String> {
    provider
        .models
        .iter()
        .find_map(|model| {
            let model = normalize_optional_task_field(Some(model.id.clone()))?;
            image_provider_has_supported_endpoint(provider, Some(&model)).then_some(model)
        })
        .or_else(|| default_image_model_for_provider(provider))
}

fn default_image_model_for_provider(provider: &ApiKeyProvider) -> Option<String> {
    match provider.effective_provider_type() {
        ApiProviderType::Openai
        | ApiProviderType::OpenaiResponse
        | ApiProviderType::Codex
        | ApiProviderType::NewApi
        | ApiProviderType::Gateway => Some("gpt-images-2".to_string()),
        ApiProviderType::Gemini => Some("gemini-2.5-flash-image".to_string()),
        ApiProviderType::Fal => provider
            .models
            .iter()
            .find_map(|model| normalize_optional_task_field(Some(model.id.clone()))),
        _ => None,
    }
}

fn image_provider_has_supported_endpoint(
    provider: &ApiKeyProvider,
    model_id: Option<&str>,
) -> bool {
    if is_zhipu_image_provider(provider) || is_dashscope_image_provider(provider, model_id) {
        return true;
    }

    let effective_type = provider.effective_provider_type();
    let spec = effective_type.runtime_spec();
    matches!(
        effective_type,
        ApiProviderType::Openai
            | ApiProviderType::OpenaiResponse
            | ApiProviderType::Codex
            | ApiProviderType::NewApi
            | ApiProviderType::Gateway
            | ApiProviderType::Gemini
    ) && (spec.protocol_family == ProviderProtocolFamily::OpenAiCompatible
        || spec.protocol_family == ProviderProtocolFamily::Gemini
        || matches!(effective_type, ApiProviderType::Codex))
}

fn is_placeholder_provider_host(api_host: &str) -> bool {
    let normalized = api_host.trim().to_ascii_lowercase();
    normalized.is_empty() || normalized.contains("example.invalid")
}

fn is_zhipu_image_provider(provider: &ApiKeyProvider) -> bool {
    let provider_id = provider.id.to_ascii_lowercase();
    let provider_name = provider.name.to_ascii_lowercase();
    let api_host = provider.api_host.to_ascii_lowercase();
    let has_zhipu_identity = provider_id.contains("zhipu")
        || provider_id.contains("bigmodel")
        || provider_name.contains("zhipu")
        || provider_name.contains("智谱")
        || api_host.contains("bigmodel.cn/api/paas");
    let has_zhipu_model = provider.models.iter().any(|model| {
        let normalized = model.id.trim().to_ascii_lowercase();
        matches!(
            normalized.as_str(),
            "glm-image" | "cogview-4-250304" | "cogview-4" | "cogview-3-flash"
        ) || normalized.contains("cogview")
            || normalized.contains("glm-image")
    });
    has_zhipu_identity || has_zhipu_model
}

fn is_dashscope_image_provider(provider: &ApiKeyProvider, model_id: Option<&str>) -> bool {
    let provider_id = provider.id.to_ascii_lowercase();
    let provider_name = provider.name.to_ascii_lowercase();
    let api_host = provider.api_host.to_ascii_lowercase();
    let has_dashscope_identity = provider_id.contains("dashscope")
        || provider_id.contains("alibaba")
        || provider_id.contains("qwen")
        || provider_id.contains("tongyi")
        || provider_name.contains("dashscope")
        || provider_name.contains("通义")
        || provider_name.contains("百炼")
        || api_host.contains("dashscope.aliyuncs.com")
        || api_host.contains("dashscope-intl.aliyuncs.com")
        || api_host.contains("maas.aliyuncs.com");
    let has_dashscope_image_model = model_id.map(is_dashscope_image_model_id).unwrap_or(false)
        || provider
            .models
            .iter()
            .any(|model| is_dashscope_image_model_id(&model.id));
    has_dashscope_identity && has_dashscope_image_model
}

fn is_dashscope_image_model_id(model_id: &str) -> bool {
    let normalized = model_id.trim().to_ascii_lowercase();
    normalized.contains("qwen-image")
        || normalized.contains("wanx")
        || normalized.contains("wan2.")
        || normalized.contains("wan2-")
}

fn normalize_image_task_mode(value: Option<String>) -> Option<String> {
    match normalize_optional_task_field(value).as_deref() {
        Some("generate") => Some("generate".to_string()),
        Some("edit") => Some("edit".to_string()),
        Some("variation") => Some("variation".to_string()),
        _ => None,
    }
}

fn normalize_image_executor_mode(value: Option<String>) -> Option<String> {
    match normalize_optional_string(value).as_deref() {
        Some("images_api") => Some("images_api".to_string()),
        Some("responses_image_generation") => Some("responses_image_generation".to_string()),
        _ => None,
    }
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value.and_then(|raw| {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn normalize_optional_task_field(value: Option<String>) -> Option<String> {
    normalize_optional_string(value).filter(|value| !is_placeholder_task_field(value))
}

fn is_placeholder_task_field(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "default"
            | "auto"
            | "automatic"
            | "system_default"
            | "system-default"
            | "__auto__"
            | "__default__"
    )
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ImageGenerationDefaults {
    provider_id: Option<String>,
    model_id: Option<String>,
}

fn configured_image_generation_defaults() -> ImageGenerationDefaults {
    image_generation_defaults_from_config_file()
}

fn image_generation_defaults_from_config_file() -> ImageGenerationDefaults {
    let path = ConfigManager::default_config_path();
    ConfigManager::load(&path)
        .map(|manager| image_generation_defaults_from_config(manager.config()))
        .unwrap_or_default()
}

fn image_generation_defaults_from_config(config: &Config) -> ImageGenerationDefaults {
    let image = &config.workspace_preferences.media_defaults.image;

    ImageGenerationDefaults {
        provider_id: normalize_optional_task_field(image.preferred_provider_id.clone()),
        model_id: normalize_optional_task_field(image.preferred_model_id.clone()),
    }
}

pub(crate) async fn assess_video_route(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    model_registry_service: &ModelRegistryService,
    params: &MediaTaskArtifactVideoCreateParams,
) -> Result<MediaRouteAssessment, String> {
    assess_media_route(
        db,
        api_key_provider_service,
        model_registry_service,
        &video_model_task_request(params),
    )
    .await
}

async fn assess_media_route(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    model_registry_service: &ModelRegistryService,
    task_request: &ModelTaskRequest,
) -> Result<MediaRouteAssessment, String> {
    let model_ref = task_request
        .model_ref
        .as_ref()
        .ok_or_else(|| "media_model_ref_missing".to_string())?;
    let provider = api_key_provider_service
        .get_provider(db, &model_ref.provider_id)
        .map_err(|error| format!("provider_lookup_failed: {error}"))?
        .ok_or_else(|| "provider_not_configured".to_string())?;
    let route_credential = model_route_credential::resolve_route_credential(
        db,
        api_key_provider_service,
        &model_ref.provider_id,
        Some(&provider),
        None,
        None,
    )
    .await?;
    let requires_api_key = ModelRegistryService::requires_api_key_for_runtime(
        &provider.provider.id,
        &provider.provider.api_host,
        provider.provider.effective_provider_type(),
    );
    if requires_api_key && route_credential.credential_ref().is_none() {
        return Err("resolved_credential_unavailable".to_string());
    }
    let model_registry = model_registry_service.resolve_provider_model_metadata(
        Some(&provider),
        &model_ref.provider_id,
        &model_ref.model_id,
        route_credential.model_cache_access(Some(&provider)),
    )?;
    let model = model_registry
        .model
        .as_ref()
        .ok_or_else(|| model_registry.reason_code.to_string())?;
    if !model_has_declared_capability_snapshot(model) {
        return Err("model_capability_snapshot_missing".to_string());
    }
    let snapshot = capability_snapshot_from_model_capabilities(&model_capabilities_value(model));
    let assessment = MediaRouteAssessment::from_snapshot(task_request, snapshot);
    let route = resolved_route_from_task_with_credential(
        task_request,
        ModelRouteSelection {
            provider_id: &model_ref.provider_id,
            model_id: &model_ref.model_id,
            model_ref_source: ModelRefSource::Task,
            reasoning_effort: None,
        },
        &media_route_payload(task_request, &model_registry, &provider),
        Some(&provider),
        route_credential.credential_ref(),
        None,
    );
    Ok(assessment.with_resolved_route(route))
}

fn model_has_declared_capability_snapshot(model: &EnhancedModelMetadata) -> bool {
    matches!(
        model.capability_provenance,
        ModelCapabilityProvenance::Canonical | ModelCapabilityProvenance::ProviderExplicit
    )
}

fn model_capabilities_value(model: &EnhancedModelMetadata) -> Value {
    json!({
        "provenance": model.capability_provenance,
        "capabilities": model.capabilities,
        "taskFamilies": model.task_families,
        "task_families": model.task_families,
        "runtimeFeatures": model.runtime_features,
        "runtime_features": model.runtime_features,
        "inputModalities": model.input_modalities,
        "input_modalities": model.input_modalities,
        "outputModalities": model.output_modalities,
        "output_modalities": model.output_modalities,
    })
}

fn media_route_payload(
    task_request: &ModelTaskRequest,
    model_registry: &ProviderModelRegistryMetadata,
    provider: &ProviderWithKeys,
) -> Value {
    let model_ref = task_request.model_ref.as_ref();
    let provider_readiness = readiness_payload(provider);
    let registry_payload = model_registry_payload(model_registry);
    json!({
        "backend": "media_task_artifact",
        "routingMode": "task_route",
        "routing_mode": "task_route",
        "decisionSource": "media_task_artifact",
        "decision_source": "media_task_artifact",
        "decisionReason": "explicit_task_model",
        "decision_reason": "explicit_task_model",
        "settingsSource": "media_task_artifact",
        "settings_source": "media_task_artifact",
        "serviceModelSlot": task_request.routing_slot,
        "service_model_slot": task_request.routing_slot,
        "selectedProvider": model_ref.map(|model_ref| model_ref.provider_id.as_str()),
        "selected_provider": model_ref.map(|model_ref| model_ref.provider_id.as_str()),
        "selectedModel": model_ref.map(|model_ref| model_ref.model_id.as_str()),
        "selected_model": model_ref.map(|model_ref| model_ref.model_id.as_str()),
        "provider": model_ref.map(|model_ref| model_ref.provider_id.as_str()),
        "model": model_ref.map(|model_ref| model_ref.model_id.as_str()),
        "requiredCapabilities": task_request.requirements.capabilities,
        "required_capabilities": task_request.requirements.capabilities,
        "providerReadiness": provider_readiness,
        "provider_readiness": provider_readiness,
        "modelRegistry": registry_payload,
        "model_registry": registry_payload,
    })
}

fn readiness_payload(provider: &ProviderWithKeys) -> Value {
    let enabled_key_count = provider.api_keys.iter().filter(|key| key.enabled).count();
    let total_key_count = provider.api_keys.len();
    let requires_api_key = ModelRegistryService::requires_api_key_for_runtime(
        &provider.provider.id,
        &provider.provider.api_host,
        provider.provider.effective_provider_type(),
    );
    let ready = provider.provider.enabled && (!requires_api_key || enabled_key_count > 0);
    let reason_code = if !provider.provider.enabled {
        Some("provider_disabled")
    } else if requires_api_key && enabled_key_count == 0 {
        Some("missing_enabled_api_key")
    } else {
        None
    };
    json!({
        "ready": ready,
        "status": if ready { "ready" } else { "needs_setup" },
        "source": "provider_store",
        "reasonCode": reason_code,
        "reason_code": reason_code,
        "providerType": provider.provider.provider_type.to_string(),
        "provider_type": provider.provider.provider_type.to_string(),
        "enabled": provider.provider.enabled,
        "requiresApiKey": requires_api_key,
        "requires_api_key": requires_api_key,
        "enabledKeyCount": enabled_key_count,
        "enabled_key_count": enabled_key_count,
        "totalKeyCount": total_key_count,
        "total_key_count": total_key_count,
        "directRequestConfig": false,
        "direct_request_config": false,
    })
}

fn model_registry_payload(model_registry: &ProviderModelRegistryMetadata) -> Value {
    let model = model_registry
        .model
        .as_ref()
        .expect("media route requires resolved model metadata");
    let model_value = serde_json::to_value(model).unwrap_or(Value::Null);
    let capabilities = model_capabilities_value(model);
    json!({
        "source": model_registry.source.as_str(),
        "sourceLabel": model_registry.source.as_str(),
        "source_label": model_registry.source.as_str(),
        "status": "matched",
        "reasonCode": model_registry.reason_code,
        "reason_code": model_registry.reason_code,
        "providerId": model_registry.provider_id,
        "provider_id": model_registry.provider_id,
        "requestedModelId": model_registry.requested_model_id,
        "requested_model_id": model_registry.requested_model_id,
        "matchedModelId": model_registry.matched_model_id,
        "matched_model_id": model_registry.matched_model_id,
        "model": model_value,
        "modelCapabilities": capabilities,
        "model_capabilities": capabilities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use lime_core::database::dao::api_key_provider::{
        ApiKeyEntry, ApiKeyProvider, ApiProviderType, ProviderGroup,
    };
    use lime_core::database::schema::create_tables;
    use lime_core::models::model_registry::{
        ModelCapabilities, ModelModality, ModelRuntimeFeature, ModelTaskFamily,
        ProviderModelCapability, ProviderModelConfig,
    };
    use lime_core::models::runtime_api_key_credential_uuid;
    use lime_services::model_registry_service::ProviderModelCacheAccess;
    use rusqlite::Connection;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn image_create_params() -> MediaTaskArtifactImageCreateParams {
        MediaTaskArtifactImageCreateParams {
            project_root_path: "/tmp/project".to_string(),
            prompt: "画一张广州夏天的图".to_string(),
            ..MediaTaskArtifactImageCreateParams::default()
        }
    }

    fn explicit_image_model(id: &str) -> ProviderModelConfig {
        ProviderModelConfig {
            id: id.to_string(),
            display_name: None,
            capability: Some(ProviderModelCapability {
                task_families: vec![ModelTaskFamily::ImageGeneration],
                input_modalities: vec![ModelModality::Text],
                output_modalities: vec![ModelModality::Image],
                runtime_features: vec![ModelRuntimeFeature::ImagesApi],
                capabilities: ModelCapabilities::default(),
            }),
        }
    }

    #[test]
    fn image_defaults_support_current_config_preferences() {
        let mut config = Config::default();
        config
            .workspace_preferences
            .media_defaults
            .image
            .preferred_provider_id = Some("custom-provider".to_string());
        config
            .workspace_preferences
            .media_defaults
            .image
            .preferred_model_id = Some("agnes-image-2.0-flash".to_string());

        let defaults = image_generation_defaults_from_config(&config);

        assert_eq!(defaults.provider_id.as_deref(), Some("custom-provider"));
        assert_eq!(defaults.model_id.as_deref(), Some("agnes-image-2.0-flash"));
    }

    #[test]
    fn image_task_submission_uses_config_defaults_and_cleans_direct_executor_mode() {
        let params = MediaTaskArtifactImageCreateParams {
            executor_mode: Some("direct".to_string()),
            ..image_create_params()
        };
        let normalized = normalize_image_create_params_with_defaults(
            params,
            ImageGenerationDefaults {
                provider_id: Some("custom-provider".to_string()),
                model_id: Some("agnes-image-2.0-flash".to_string()),
            },
        )
        .expect("normalized image params");

        assert_eq!(
            normalized.params.provider_id.as_deref(),
            Some("custom-provider")
        );
        assert_eq!(
            normalized.params.model.as_deref(),
            Some("agnes-image-2.0-flash")
        );
        assert_eq!(normalized.params.executor_mode, None);
        assert!(normalized.provider_from_defaults);
        assert!(normalized.model_from_defaults);
    }

    #[test]
    fn image_task_submission_treats_default_model_as_config_fallback() {
        let params = MediaTaskArtifactImageCreateParams {
            mode: Some("default".to_string()),
            provider_id: Some("custom-provider".to_string()),
            model: Some("default".to_string()),
            outer_model: Some("default".to_string()),
            session_id: Some("default".to_string()),
            thread_id: Some("default".to_string()),
            turn_id: Some("default".to_string()),
            project_id: Some("default".to_string()),
            routing_slot: Some("default".to_string()),
            requested_target: Some("default".to_string()),
            slot_id: Some("default".to_string()),
            target_output_id: Some("default".to_string()),
            target_output_ref_id: Some("default".to_string()),
            ..image_create_params()
        };
        let normalized = normalize_image_create_params_with_defaults(
            params,
            ImageGenerationDefaults {
                provider_id: Some("custom-provider".to_string()),
                model_id: Some("agnes-image-2.1-flash".to_string()),
            },
        )
        .expect("normalized image params");

        assert_eq!(
            normalized.params.provider_id.as_deref(),
            Some("custom-provider")
        );
        assert_eq!(
            normalized.params.model.as_deref(),
            Some("agnes-image-2.1-flash")
        );
        assert_eq!(normalized.params.mode, None);
        assert_eq!(normalized.params.session_id, None);
        assert_eq!(normalized.params.thread_id, None);
        assert_eq!(normalized.params.turn_id, None);
        assert_eq!(normalized.params.project_id, None);
        assert_eq!(normalized.params.routing_slot, None);
        assert_eq!(normalized.params.requested_target, None);
        assert_eq!(normalized.params.slot_id, None);
        assert_eq!(normalized.params.target_output_id, None);
        assert_eq!(normalized.params.target_output_ref_id, None);
        assert_eq!(normalized.params.outer_model, None);
        assert!(!normalized.provider_from_defaults);
        assert!(normalized.model_from_defaults);
    }

    #[test]
    fn image_defaults_ignore_placeholder_values() {
        let mut config = Config::default();
        config
            .workspace_preferences
            .media_defaults
            .image
            .preferred_provider_id = Some("default".to_string());
        config
            .workspace_preferences
            .media_defaults
            .image
            .preferred_model_id = Some("auto".to_string());

        let defaults = image_generation_defaults_from_config(&config);

        assert_eq!(defaults, ImageGenerationDefaults::default());
    }

    #[test]
    fn image_task_submission_fails_without_explicit_or_default_model_ref() {
        let error =
            normalize_image_create_params_with_defaults(image_create_params(), Default::default())
                .expect_err("missing image defaults should fail closed");

        assert!(error.contains("图片生成缺少默认 Provider 或模型"));
    }

    #[test]
    fn image_task_submission_rejects_missing_explicit_provider_before_task_write() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let normalized = NormalizedImageCreateParams::new(
            MediaTaskArtifactImageCreateParams {
                provider_id: Some("stale-provider".to_string()),
                model: Some("gpt-images-2".to_string()),
                ..image_create_params()
            },
            false,
            false,
        );

        let error = resolve_image_provider_for_task_submission(&db, &service, normalized)
            .expect_err("missing explicit provider should fail");

        assert!(error.contains("图片 Provider stale-provider 不存在"));
    }

    #[test]
    fn image_task_submission_falls_back_when_default_provider_was_deleted() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        insert_provider_with_key(
            &db,
            "ready-images",
            "gpt-images-2",
            "https://images.example/v1",
        );
        let normalized = NormalizedImageCreateParams::new(
            MediaTaskArtifactImageCreateParams {
                provider_id: Some("deleted-default".to_string()),
                model: Some("gpt-images-2".to_string()),
                ..image_create_params()
            },
            true,
            true,
        );

        let resolved = resolve_image_provider_for_task_submission(&db, &service, normalized)
            .expect("deleted default provider should fall back to ready image provider");

        assert_eq!(resolved.provider_id.as_deref(), Some("ready-images"));
        assert_eq!(resolved.model.as_deref(), Some("gpt-images-2"));
    }

    #[test]
    fn image_task_submission_rejects_placeholder_provider_host() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        insert_provider_with_key(
            &db,
            "placeholder-images",
            "gpt-images-2",
            "https://example.invalid/v1",
        );
        let normalized = NormalizedImageCreateParams::new(
            MediaTaskArtifactImageCreateParams {
                provider_id: Some("placeholder-images".to_string()),
                model: Some("gpt-images-2".to_string()),
                ..image_create_params()
            },
            false,
            false,
        );

        let error = resolve_image_provider_for_task_submission(&db, &service, normalized)
            .expect_err("placeholder provider should fail");

        assert!(error.contains("API Host 是示例地址"));
    }

    #[tokio::test]
    async fn media_route_rejects_missing_scoped_model_metadata() {
        let db = test_db();
        let api_key_service = ApiKeyProviderService::new();
        let model_registry_service = ModelRegistryService::new(db.clone());
        let provider = api_key_service
            .add_custom_provider(
                &db,
                "Missing Media Metadata".to_string(),
                ApiProviderType::Openai,
                "http://127.0.0.1:9".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create media provider");
        api_key_service
            .add_api_key(&db, &provider.id, "missing-media-metadata-key", None, false)
            .expect("add media key");

        let error = assess_image_route(
            &db,
            &api_key_service,
            &model_registry_service,
            &MediaTaskArtifactImageCreateParams {
                provider_id: Some(provider.id),
                model: Some("missing-image-model".to_string()),
                ..image_create_params()
            },
        )
        .await
        .expect_err("missing scoped metadata must reject task creation");

        assert_eq!(error, "model_registry_metadata_missing");
    }

    #[tokio::test]
    async fn media_route_binds_scoped_metadata_and_execution_to_one_credential() {
        let db = test_db();
        let api_key_service = ApiKeyProviderService::new();
        let model_registry_service = ModelRegistryService::new(db.clone());
        let model_id = "gpt-image-edit-2";
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind local models fixture");
        let address = listener.local_addr().expect("local models address");
        let models_body = serde_json::json!({
            "data": [{
                "id": model_id,
                "task_families": ["image_generation", "image_edit"],
                "input_modalities": ["text", "image"],
                "output_modalities": ["image"],
                "runtime_features": ["images_api"]
            }]
        })
        .to_string();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept models request");
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).await;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                models_body.len(),
                models_body
            );
            stream
                .write_all(response.as_bytes())
                .await
                .expect("write models response");
        });
        let api_host = format!("http://{address}");
        let provider = api_key_service
            .add_custom_provider(
                &db,
                "Scoped Media Fixture".to_string(),
                ApiProviderType::Openai,
                api_host.clone(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create media provider");
        let key_a = api_key_service
            .add_api_key(&db, &provider.id, "media-scope-key-a", None, false)
            .expect("add key A");
        let key_b = api_key_service
            .add_api_key(&db, &provider.id, "media-scope-key-b", None, false)
            .expect("add key B");
        let ref_a = runtime_api_key_credential_uuid(&key_a.id);
        let ref_b = runtime_api_key_credential_uuid(&key_b.id);
        let first = api_key_service
            .select_credential_for_provider(&db, &provider.id, Some(&provider.id), None)
            .await
            .expect("advance media credential rotation")
            .expect("first media credential");
        let (expected_ref, expected_api_key, other_ref) = if first.uuid == ref_a {
            (&ref_b, "media-scope-key-b", &ref_a)
        } else {
            (&ref_a, "media-scope-key-a", &ref_b)
        };

        model_registry_service
            .fetch_models_from_api_with_hints(
                &provider.id,
                &api_host,
                expected_api_key,
                Some(ApiProviderType::Openai),
                &[],
            )
            .await
            .expect("seed selected credential model cache");

        let provider_with_keys = api_key_service
            .get_provider(&db, &provider.id)
            .expect("read provider")
            .expect("media provider");
        let expected_credential = api_key_service
            .select_runtime_credential_by_ref(&db, &provider.id, expected_ref)
            .expect("read expected credential")
            .expect("expected credential");
        let other_credential = api_key_service
            .select_runtime_credential_by_ref(&db, &provider.id, other_ref)
            .expect("read other credential")
            .expect("other credential");
        let selected_metadata = model_registry_service
            .resolve_provider_model_metadata(
                Some(&provider_with_keys),
                &provider.id,
                model_id,
                ProviderModelCacheAccess::Credential(&expected_credential),
            )
            .expect("selected credential metadata");
        let other_metadata = model_registry_service
            .resolve_provider_model_metadata(
                Some(&provider_with_keys),
                &provider.id,
                model_id,
                ProviderModelCacheAccess::Credential(&other_credential),
            )
            .expect("other credential metadata");
        assert_eq!(selected_metadata.source.as_str(), "provider_models_cache");
        assert_eq!(other_metadata.source.as_str(), "runtime_selection_only");
        assert!(other_metadata.model.is_none());

        let assessment = assess_image_route(
            &db,
            &api_key_service,
            &model_registry_service,
            &MediaTaskArtifactImageCreateParams {
                provider_id: Some(provider.id.clone()),
                model: Some(model_id.to_string()),
                ..image_create_params()
            },
        )
        .await
        .expect("media route assessment");
        let route = assessment.resolved_route.expect("resolved media route");

        assert!(
            route.failure.is_none(),
            "route failure: {:?}",
            route.failure
        );
        assert_eq!(
            route.auth.credential_ref.as_deref(),
            Some(expected_ref.as_str())
        );
        let evidence = serde_json::to_string(&route).expect("serialize route");
        assert!(!evidence.contains("media-scope-key-a"));
        assert!(!evidence.contains("media-scope-key-b"));
    }

    fn test_db() -> DbConnection {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        create_tables(&conn).expect("create schema");
        Arc::new(Mutex::new(conn))
    }

    fn insert_provider_with_key(db: &DbConnection, id: &str, model: &str, api_host: &str) {
        let now = Utc::now();
        let provider = ApiKeyProvider {
            id: id.to_string(),
            name: id.to_string(),
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
            models: vec![explicit_image_model(model)],
            prompt_cache_mode: None,
            created_at: now,
            updated_at: now,
        };
        let key = ApiKeyEntry {
            id: format!("{id}-key"),
            provider_id: id.to_string(),
            api_key_encrypted: "encrypted-test-key".to_string(),
            alias: None,
            enabled: true,
            usage_count: 0,
            error_count: 0,
            last_used_at: None,
            created_at: now,
        };
        let conn = lime_core::database::lock_db(db).expect("lock db");
        lime_core::database::dao::api_key_provider::ApiKeyProviderDao::insert_provider(
            &conn, &provider,
        )
        .expect("insert provider");
        lime_core::database::dao::api_key_provider::ApiKeyProviderDao::insert_api_key(&conn, &key)
            .expect("insert api key");
    }
}
