use super::request_context::RuntimeModelSelection;
use crate::ExecutionRequest;
use lime_core::database::DbConnection;
use lime_core::models::model_registry::{
    EnhancedModelMetadata, ModelCapabilities, ModelRuntimeFeature, ProviderModelConfig,
};
use lime_services::api_key_provider_service::ApiKeyProviderService;
use lime_services::model_registry_service::ModelRegistryService;
use runtime_core::{CandidateModel, CandidateModelSet, CandidateRequirements};
use serde::Serialize;

pub(super) fn build(
    db: &DbConnection,
    service: &ApiKeyProviderService,
    request: &ExecutionRequest,
    selection: &RuntimeModelSelection,
) -> Result<CandidateModelSet, String> {
    let mut requirements = CandidateRequirements {
        task_families: vec!["chat".to_string()],
        input_modalities: vec!["text".to_string()],
        output_modalities: vec!["text".to_string()],
        runtime_features: vec!["streaming".to_string()],
        capabilities: vec![
            "coding".to_string(),
            "tools".to_string(),
            "streaming".to_string(),
        ],
    };
    if request.input.has_images() {
        requirements
            .task_families
            .push("vision_understanding".to_string());
        requirements.input_modalities.push("image".to_string());
        requirements.capabilities.push("vision".to_string());
    }

    let mut candidate_set = CandidateModelSet::new(requirements);
    candidate_set.push_unique(CandidateModel::from_selection(selection));
    let providers = service.get_all_providers(db)?;
    let registry = ModelRegistryService::new(db.clone());
    for provider in providers {
        for model in provider
            .provider
            .models
            .iter()
            .map(candidate_from_declared_model)
        {
            candidate_set.push_unique(CandidateModel {
                provider: provider.provider.id.clone(),
                ..model
            });
        }

        if let Some(cached) = registry.get_cached_provider_models(
            &provider.provider.id,
            &provider.provider.api_host,
            Some(provider.provider.effective_provider_type()),
        )? {
            for model in cached.models {
                candidate_set.push_unique(candidate_from_registry_model(&model));
            }
        }
    }
    Ok(candidate_set)
}

fn candidate_from_declared_model(model: &ProviderModelConfig) -> CandidateModel {
    let capability = model.capability.as_ref();
    CandidateModel {
        provider: String::new(),
        model: model.id.clone(),
        source: "provider_declared_model".to_string(),
        status: "declared".to_string(),
        task_families: capability
            .map(|value| serialized_tokens(&value.task_families))
            .unwrap_or_default(),
        input_modalities: capability
            .map(|value| serialized_tokens(&value.input_modalities))
            .unwrap_or_default(),
        output_modalities: capability
            .map(|value| serialized_tokens(&value.output_modalities))
            .unwrap_or_default(),
        runtime_features: capability
            .map(|value| serialized_tokens(&value.runtime_features))
            .unwrap_or_default(),
        capabilities: capability
            .map(|value| capability_tokens(&value.capabilities, &value.runtime_features))
            .unwrap_or_default(),
        estimated_cost_class: None,
        limit_state: None,
        continuity_key: Some(model.id.clone()),
    }
}

fn candidate_from_registry_model(model: &EnhancedModelMetadata) -> CandidateModel {
    CandidateModel {
        provider: model.provider_id.clone(),
        model: model.id.clone(),
        source: format!("provider_{}", &model.source),
        status: model.status.to_string(),
        task_families: serialized_tokens(&model.task_families),
        input_modalities: serialized_tokens(&model.input_modalities),
        output_modalities: serialized_tokens(&model.output_modalities),
        runtime_features: serialized_tokens(&model.runtime_features),
        capabilities: capability_tokens(&model.capabilities, &model.runtime_features),
        estimated_cost_class: model.pricing.as_ref().map(|pricing| {
            let prices = [
                pricing.input_per_million,
                pricing.output_per_million,
                pricing.cache_read_per_million,
                pricing.cache_write_per_million,
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
            if prices.is_empty() {
                "unknown".to_string()
            } else if prices.iter().all(|value| *value == 0.0) {
                "free".to_string()
            } else {
                "priced".to_string()
            }
        }),
        limit_state: model
            .limits
            .context_length
            .or(model.limits.max_output_tokens)
            .or(model.limits.requests_per_minute)
            .or(model.limits.tokens_per_minute)
            .map(|_| "declared".to_string()),
        continuity_key: model
            .family
            .clone()
            .or_else(|| model.canonical_model_id.clone())
            .or_else(|| Some(model.id.clone())),
    }
}

fn serialized_tokens<T: Serialize>(values: &[T]) -> Vec<String> {
    serde_json::to_value(values)
        .ok()
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(ToString::to_string))
        .collect()
}

fn capability_tokens(
    capabilities: &ModelCapabilities,
    runtime_features: &[ModelRuntimeFeature],
) -> Vec<String> {
    let mut values = serialized_tokens(runtime_features);
    for (name, enabled) in [
        ("vision", capabilities.vision),
        ("tools", capabilities.tools),
        ("streaming", capabilities.streaming),
        ("json_mode", capabilities.json_mode),
        ("function_calling", capabilities.function_calling),
        ("reasoning", capabilities.reasoning),
    ] {
        if enabled && !values.iter().any(|value| value == name) {
            values.push(name.to_string());
        }
    }
    values
}
