use super::super::{ProviderModelCatalog, RuntimeCore, RuntimeCoreError};
use app_server_protocol::protocol::v2::MultiAgentVersion;
use app_server_protocol::protocol::v2::{ThreadSettings, ThreadSettingsUpdateParams};
use app_server_protocol::*;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use lime_core::models::model_registry::{
    EnhancedModelMetadata, ModelCapabilityProvenance, ModelModality, ModelMultiAgentVersion,
    ModelVisibility,
};

impl RuntimeCore {
    pub async fn list_models(
        &self,
        params: ModelListParams,
    ) -> Result<ModelListResponse, RuntimeCoreError> {
        model_list_from_catalogs(params, self.model_catalog(None).await?)
    }

    pub(crate) async fn resolve_thread_start_model_selection(
        &self,
        model: Option<&str>,
        model_provider: Option<&str>,
        service_tier: Option<Option<String>>,
    ) -> Result<ThreadStartModelSelection, RuntimeCoreError> {
        let model = normalize_optional_thread_start_value(model, "model")?;
        let model_provider =
            normalize_optional_thread_start_value(model_provider, "modelProvider")?;
        match (model, model_provider) {
            (None, None) => {
                let catalogs = self.model_catalog(None).await?;
                let candidate = selectable_chat_models(&catalogs)
                    .into_iter()
                    .next()
                    .ok_or_else(|| RuntimeCoreError::RouteRejected {
                        session_id: "thread/start".to_string(),
                        provider: None,
                        model: None,
                        category: RouteFailureCategory::ModelUnavailable,
                        reason_code: "model_catalog_has_no_executable_selection".to_string(),
                    })?;
                Ok(ThreadStartModelSelection {
                    model: candidate.model,
                    model_provider: candidate.provider,
                    service_tier: service_tier.unwrap_or(candidate.default_service_tier),
                })
            }
            (Some(model), Some(model_provider)) => {
                let catalogs = self.model_catalog(Some(&model_provider)).await?;
                let metadata = find_catalog_model(&catalogs, &model_provider, &model);
                let selected_model = metadata.map(effective_provider_model_id).unwrap_or(model);
                let default_service_tier = metadata.and_then(supported_default_service_tier);
                Ok(ThreadStartModelSelection {
                    model: selected_model,
                    model_provider,
                    service_tier: service_tier.unwrap_or(default_service_tier),
                })
            }
            _ => Err(RuntimeCoreError::InvalidRequest(
                "thread/start model and modelProvider must be provided together".to_string(),
            )),
        }
    }

    pub(crate) async fn resolve_scheduled_task_model_selection(
        &self,
        model_id: Option<&str>,
    ) -> Result<ThreadStartModelSelection, RuntimeCoreError> {
        let Some(model_id) = model_id else {
            return self
                .resolve_thread_start_model_selection(None, None, None)
                .await;
        };
        let model_id = non_empty(model_id).ok_or_else(|| {
            RuntimeCoreError::InvalidRequest(
                "scheduled task modelId must be a non-empty string".to_string(),
            )
        })?;
        let route = decode_model_route_selector(model_id)?;
        let catalogs = self.model_catalog(None).await?;
        let mut candidates = selectable_chat_models(&catalogs)
            .into_iter()
            .filter(|candidate| match route.as_ref() {
                Some((provider, model)) => {
                    candidate.provider == *provider && candidate.matches_model(model)
                }
                None => candidate.matches_model(model_id),
            })
            .collect::<Vec<_>>();

        match candidates.len() {
            1 => {
                let candidate = candidates.pop().expect("single scheduled task model");
                Ok(ThreadStartModelSelection {
                    model: candidate.model,
                    model_provider: candidate.provider,
                    service_tier: candidate.default_service_tier,
                })
            }
            0 => Err(RuntimeCoreError::RouteRejected {
                session_id: "scheduledTask/run/start".to_string(),
                provider: route.as_ref().map(|(provider, _)| provider.clone()),
                model: Some(
                    route
                        .as_ref()
                        .map(|(_, model)| model.clone())
                        .unwrap_or_else(|| model_id.to_string()),
                ),
                category: RouteFailureCategory::ModelUnavailable,
                reason_code: "scheduled_task_model_unavailable".to_string(),
            }),
            _ => Err(RuntimeCoreError::InvalidRequest(format!(
                "scheduled task modelId is ambiguous across providers: {model_id}"
            ))),
        }
    }

    pub(crate) async fn reconcile_thread_model_selection(
        &self,
        thread_id: &str,
    ) -> Result<Option<ThreadSettings>, RuntimeCoreError> {
        self.reconcile_thread_model_selection_for_turn(thread_id, None)
            .await
    }

    pub(in crate::runtime) async fn reconcile_thread_model_selection_for_turn(
        &self,
        thread_id: &str,
        runtime_options: Option<&RuntimeOptions>,
    ) -> Result<Option<ThreadSettings>, RuntimeCoreError> {
        const MAX_GENERATION_ATTEMPTS: usize = 3;
        if !self.backend.requires_provider_selection()
            || !self.app_data_source.model_catalog_reconciliation_enabled()
            || runtime_options_has_direct_provider_config(runtime_options)
        {
            return Ok(None);
        }
        let mut reconciled_settings = None;
        let mut refreshed_providers = std::collections::HashSet::new();

        for _ in 0..MAX_GENERATION_ATTEMPTS {
            let generation = self.app_data_source.read_model_route_generation().await?;
            let Some((session_id, current, has_agent_control_route)) =
                self.loaded_thread_settings(thread_id)?
            else {
                return Ok(None);
            };
            if has_agent_control_route {
                return Ok(None);
            }
            let catalogs = self.model_catalog(None).await?;
            if self.app_data_source.read_model_route_generation().await? != generation {
                continue;
            }

            let candidates = selectable_chat_models(&catalogs);
            let current_candidate = candidates.iter().find(|candidate| {
                candidate.provider == current.model_provider
                    && candidate.matches_model(&current.model)
            });
            let should_probe_missing_catalog = current_candidate.is_none()
                && !refreshed_providers.contains(&current.model_provider)
                && !self
                    .has_model_provider_last_success(&current.model_provider)
                    .await?;
            let mut last_route_error = None;
            if current_candidate.is_some() || should_probe_missing_catalog {
                let session = self.session_snapshot(&session_id)?.0;
                match self
                    .backend
                    .preflight_thread_settings(&session, &current)
                    .await
                {
                    Ok(()) if current_candidate.is_some() => return Ok(reconciled_settings),
                    Ok(()) => {}
                    Err(
                        error @ (RuntimeCoreError::RouteRejected { .. }
                        | RuntimeCoreError::PendingRoute { .. }),
                    ) => {
                        last_route_error = Some(error);
                    }
                    Err(error) => return Err(error),
                }
            }

            if let Some(provider_id) = last_route_error
                .as_ref()
                .and_then(refreshable_model_catalog_provider)
                .filter(|provider_id| *provider_id == current.model_provider)
            {
                let has_last_success = if should_probe_missing_catalog {
                    false
                } else {
                    self.has_model_provider_last_success(provider_id).await?
                };
                if !has_last_success && refreshed_providers.insert(provider_id.to_string()) {
                    match self.refresh_model_provider_catalog(provider_id).await {
                        Ok(response) if response.source == "Api" => continue,
                        Ok(response) => tracing::warn!(
                            provider_id,
                            source = response.source,
                            error_kind = response.error_kind.as_deref().unwrap_or("unknown"),
                            "turn start model catalog refresh did not produce API metadata"
                        ),
                        Err(error) => tracing::warn!(
                            provider_id,
                            error = %error,
                            "turn start model catalog refresh failed"
                        ),
                    }
                }
            }

            let mut candidates = candidates;
            candidates.sort_by_key(|candidate| candidate.provider != current.model_provider);

            let mut changed_settings = None;
            for candidate in candidates {
                let mut collaboration_mode = current.collaboration_mode.clone();
                collaboration_mode.settings.model = candidate.model.clone();
                collaboration_mode.settings.reasoning_effort = candidate.reasoning_effort.clone();
                let params = ThreadSettingsUpdateParams {
                    thread_id: thread_id.to_string(),
                    model: Some(candidate.model),
                    model_provider: Some(candidate.provider),
                    service_tier: Some(candidate.default_service_tier),
                    collaboration_mode: Some(collaboration_mode),
                    ..Default::default()
                };
                match self.update_thread_settings(params).await {
                    Ok(settings) => {
                        reconciled_settings = Some(settings.clone());
                        changed_settings = Some(settings);
                        break;
                    }
                    Err(error @ RuntimeCoreError::RouteRejected { .. }) => {
                        last_route_error = Some(error);
                    }
                    Err(error) => return Err(error),
                }
            }

            let Some(_) = changed_settings else {
                return Err(last_route_error.unwrap_or(RuntimeCoreError::RouteRejected {
                    session_id,
                    provider: Some(current.model_provider),
                    model: Some(current.model),
                    category: RouteFailureCategory::ModelUnavailable,
                    reason_code: "model_catalog_has_no_executable_selection".to_string(),
                }));
            };
            if self.app_data_source.read_model_route_generation().await? == generation {
                return Ok(reconciled_settings);
            }
        }

        Err(RuntimeCoreError::Backend(
            "model route generation changed repeatedly during model selection reconciliation"
                .to_string(),
        ))
    }
}

fn refreshable_model_catalog_provider(error: &RuntimeCoreError) -> Option<&str> {
    match error {
        RuntimeCoreError::PendingRoute {
            provider: Some(provider),
            reason_code,
            ..
        }
        | RuntimeCoreError::RouteRejected {
            provider: Some(provider),
            reason_code,
            ..
        } if matches!(
            reason_code.as_str(),
            "model_registry_metadata_missing" | "capability_snapshot_missing"
        ) =>
        {
            non_empty(provider)
        }
        _ => None,
    }
}

fn runtime_options_has_direct_provider_config(runtime_options: Option<&RuntimeOptions>) -> bool {
    runtime_options
        .and_then(RuntimeOptions::runtime_request)
        .and_then(|request| request.provider_config.as_ref())
        .is_some_and(|config| config.api_key.is_some() || config.base_url.is_some())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SelectableModel {
    provider: String,
    model: String,
    aliases: Vec<String>,
    reasoning_effort: Option<String>,
    default_service_tier: Option<String>,
}

pub(crate) struct ThreadStartModelSelection {
    pub(crate) model: String,
    pub(crate) model_provider: String,
    pub(crate) service_tier: Option<String>,
}

impl SelectableModel {
    fn matches_model(&self, model: &str) -> bool {
        self.model == model || self.aliases.iter().any(|alias| alias == model)
    }
}

fn selectable_chat_models(catalogs: &[ProviderModelCatalog]) -> Vec<SelectableModel> {
    catalogs
        .iter()
        .flat_map(|catalog| {
            catalog.models.iter().filter_map(|metadata| {
                if metadata.visibility != ModelVisibility::List
                    || !is_executable_chat_model(metadata)
                {
                    return None;
                }
                let model = effective_provider_model_id(metadata);
                let mut aliases = vec![metadata.id.clone()];
                if let Some(canonical) = metadata
                    .canonical_model_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty() && *value != model)
                {
                    aliases.push(canonical.to_string());
                }
                let reasoning_effort =
                    metadata
                        .capabilities
                        .reasoning_effort
                        .as_ref()
                        .and_then(|support| {
                            support.default.clone().or_else(|| {
                                support
                                    .options
                                    .iter()
                                    .find(|option| option.default)
                                    .map(|option| option.value.clone())
                            })
                        });
                Some(SelectableModel {
                    provider: catalog.provider_id.clone(),
                    model,
                    aliases,
                    reasoning_effort,
                    default_service_tier: supported_default_service_tier(metadata),
                })
            })
        })
        .collect()
}

pub(in crate::runtime) fn catalog_model_default_service_tier(
    catalogs: &[ProviderModelCatalog],
    provider: &str,
    model: &str,
) -> Option<Option<String>> {
    find_catalog_model(catalogs, provider, model).map(supported_default_service_tier)
}

fn find_catalog_model<'a>(
    catalogs: &'a [ProviderModelCatalog],
    provider: &str,
    model: &str,
) -> Option<&'a EnhancedModelMetadata> {
    catalogs
        .iter()
        .find(|catalog| catalog.provider_id == provider)
        .and_then(|catalog| {
            catalog.models.iter().find(|metadata| {
                metadata.visibility != ModelVisibility::None
                    && metadata.capability_provenance != ModelCapabilityProvenance::InferredHint
                    && (effective_provider_model_id(metadata) == model
                        || metadata.id == model
                        || metadata.canonical_model_id.as_deref() == Some(model))
            })
        })
}

fn supported_default_service_tier(metadata: &EnhancedModelMetadata) -> Option<String> {
    metadata.default_service_tier.clone().filter(|default| {
        metadata
            .service_tiers
            .iter()
            .any(|tier| tier.id == *default)
    })
}

fn normalize_optional_thread_start_value(
    value: Option<&str>,
    field: &str,
) -> Result<Option<String>, RuntimeCoreError> {
    value
        .map(|value| {
            non_empty(value).map(str::to_string).ok_or_else(|| {
                RuntimeCoreError::InvalidRequest(format!(
                    "thread/start {field} must be a non-empty string"
                ))
            })
        })
        .transpose()
}

pub(in crate::runtime) fn is_executable_chat_model(metadata: &EnhancedModelMetadata) -> bool {
    if metadata.capability_provenance == ModelCapabilityProvenance::InferredHint {
        return false;
    }

    let input_modalities = chat_wire_input_modalities(metadata);
    let snapshot = chat_wire_capability_snapshot(metadata, &input_modalities);
    runtime_core::route_capability_gap(&chat_task_request(metadata), &snapshot).is_none()
}

fn chat_task_request(metadata: &EnhancedModelMetadata) -> ModelTaskRequest {
    runtime_core::build_model_task_request(runtime_core::ModelTaskRequestInput {
        task_kind: ModelTaskKind::Chat,
        source: ModelTaskSource::AgentTurn,
        provider_id: Some(metadata.provider_id.clone()),
        model_id: Some(effective_provider_model_id(metadata)),
        model_ref_source: ModelRefSource::Explicit,
        modality_contract_key: Some("chat".to_string()),
        routing_slot: Some("coding".to_string()),
        task_families: vec!["chat".to_string()],
        input_modalities: vec!["text".to_string()],
        output_modalities: vec!["text".to_string()],
        runtime_features: vec!["streaming".to_string()],
        capabilities: vec!["streaming".to_string()],
        session_id: None,
        thread_id: None,
        turn_id: None,
        content_id: None,
        trace_id: None,
    })
}

fn chat_wire_input_modalities(metadata: &EnhancedModelMetadata) -> Vec<ModelModality> {
    metadata
        .input_modalities
        .iter()
        .filter(|modality| matches!(modality, ModelModality::Text | ModelModality::Image))
        .cloned()
        .collect()
}

fn chat_wire_capability_snapshot(
    metadata: &EnhancedModelMetadata,
    input_modalities: &[ModelModality],
) -> CapabilitySnapshot {
    let mut snapshot =
        runtime_core::capability_snapshot_from_model_capabilities(&serde_json::json!({
            "taskFamilies": &metadata.task_families,
            "inputModalities": input_modalities,
            "outputModalities": &metadata.output_modalities,
            "runtimeFeatures": &metadata.runtime_features,
            "capabilities": &metadata.capabilities,
        }));
    snapshot.source = Some(
        match metadata.capability_provenance {
            ModelCapabilityProvenance::Canonical => "canonical",
            ModelCapabilityProvenance::ProviderExplicit => "provider_explicit",
            ModelCapabilityProvenance::InferredHint => "inferred_hint",
        }
        .to_string(),
    );
    snapshot
}

fn effective_provider_model_id(metadata: &EnhancedModelMetadata) -> String {
    metadata
        .provider_model_id
        .as_deref()
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .unwrap_or(metadata.id.as_str())
        .to_string()
}

fn model_list_from_catalogs(
    params: ModelListParams,
    catalogs: Vec<ProviderModelCatalog>,
) -> Result<ModelListResponse, RuntimeCoreError> {
    let ModelListParams {
        cursor,
        limit,
        include_hidden,
    } = params;
    let include_hidden = include_hidden.unwrap_or(false);
    let mut models = catalogs
        .into_iter()
        .flat_map(|catalog| {
            let provider_id = catalog.provider_id;
            catalog
                .models
                .into_iter()
                .filter(move |metadata| {
                    is_executable_chat_model(metadata)
                        && (include_hidden || metadata.visibility == ModelVisibility::List)
                })
                .map(move |metadata| model_from_catalog(&provider_id, metadata))
        })
        .collect::<Vec<_>>();
    let default_index = models
        .iter()
        .position(|model| !model.hidden)
        .or_else(|| (!models.is_empty()).then_some(0));
    if let Some(default_index) = default_index {
        models[default_index].is_default = true;
    }
    let total = models.len();

    if total == 0 {
        return Ok(ModelListResponse {
            data: Vec::new(),
            next_cursor: None,
        });
    }

    let effective_limit = limit.unwrap_or(total as u32).max(1) as usize;
    let effective_limit = effective_limit.min(total);
    let start = match cursor {
        Some(cursor) => cursor
            .parse::<usize>()
            .map_err(|_| RuntimeCoreError::InvalidRequest(format!("invalid cursor: {cursor}")))?,
        None => 0,
    };
    if start > total {
        return Err(RuntimeCoreError::InvalidRequest(format!(
            "cursor {start} exceeds total models {total}"
        )));
    }

    let end = start.saturating_add(effective_limit).min(total);
    let data = models[start..end].to_vec();
    let next_cursor = (end < total).then(|| end.to_string());
    Ok(ModelListResponse { data, next_cursor })
}

fn model_from_catalog(provider_id: &str, metadata: EnhancedModelMetadata) -> Model {
    let provider_model_id = effective_provider_model_id(&metadata);
    let runtime_input_modalities = chat_wire_input_modalities(&metadata);
    let capability_snapshot = chat_wire_capability_snapshot(&metadata, &runtime_input_modalities);
    let reasoning = metadata.capabilities.reasoning_effort.as_ref();
    let supported_reasoning_efforts = reasoning
        .map(|support| {
            if support.options.is_empty() {
                support
                    .levels
                    .iter()
                    .filter_map(|effort| non_empty(effort))
                    .map(|effort| ReasoningEffortOption {
                        reasoning_effort: effort.to_string(),
                        description: effort.to_string(),
                    })
                    .collect()
            } else {
                support
                    .options
                    .iter()
                    .filter_map(|option| {
                        let effort = non_empty(&option.value)?;
                        let description = option
                            .description
                            .as_deref()
                            .and_then(non_empty)
                            .or_else(|| non_empty(&option.label))
                            .unwrap_or(effort);
                        Some(ReasoningEffortOption {
                            reasoning_effort: effort.to_string(),
                            description: description.to_string(),
                        })
                    })
                    .collect()
            }
        })
        .unwrap_or_default();
    let default_reasoning_effort = reasoning
        .and_then(|support| support.default.as_deref())
        .and_then(non_empty)
        .or_else(|| {
            reasoning.and_then(|support| {
                support
                    .options
                    .iter()
                    .find(|option| option.default)
                    .and_then(|option| non_empty(&option.value))
            })
        })
        .unwrap_or("none")
        .to_string();
    let input_modalities = runtime_input_modalities
        .iter()
        .filter_map(|modality| match modality {
            ModelModality::Text => Some(InputModality::Text),
            ModelModality::Image => Some(InputModality::Image),
            _ => None,
        })
        .collect::<Vec<_>>();
    let service_tiers = metadata
        .service_tiers
        .iter()
        .map(|tier| ModelServiceTier {
            id: tier.id.clone(),
            name: tier.name.clone(),
            description: tier.description.clone(),
        })
        .collect::<Vec<_>>();
    let default_service_tier = supported_default_service_tier(&metadata);

    Model {
        id: encode_model_route_selector(provider_id, &metadata.id),
        provider_id: provider_id.to_string(),
        model: provider_model_id,
        upgrade: None,
        upgrade_info: None,
        availability_nux: None,
        display_name: metadata.display_name,
        description: metadata.description.unwrap_or_default(),
        hidden: metadata.visibility != ModelVisibility::List,
        supported_reasoning_efforts,
        default_reasoning_effort,
        input_modalities,
        capability_snapshot,
        context_window: metadata.limits.context_length,
        max_output_tokens: metadata.limits.max_output_tokens,
        supports_personality: false,
        multi_agent_version: metadata.multi_agent_version.map(|version| match version {
            ModelMultiAgentVersion::Disabled => MultiAgentVersion::Disabled,
            ModelMultiAgentVersion::V1 => MultiAgentVersion::V1,
            ModelMultiAgentVersion::V2 => MultiAgentVersion::V2,
        }),
        additional_speed_tiers: Vec::new(),
        service_tiers,
        default_service_tier,
        is_default: false,
    }
}

fn encode_model_route_selector(provider_id: &str, model_id: &str) -> String {
    format!(
        "route:{}.{}",
        URL_SAFE_NO_PAD.encode(provider_id.as_bytes()),
        URL_SAFE_NO_PAD.encode(model_id.as_bytes())
    )
}

fn decode_model_route_selector(
    selector: &str,
) -> Result<Option<(String, String)>, RuntimeCoreError> {
    let Some(encoded) = selector.strip_prefix("route:") else {
        return Ok(None);
    };
    let mut parts = encoded.split('.');
    let provider = parts.next();
    let model = parts.next();
    if provider.is_none() || model.is_none() || parts.next().is_some() {
        return Err(invalid_model_route_selector(selector));
    }
    let decode = |value: &str| {
        URL_SAFE_NO_PAD
            .decode(value)
            .ok()
            .and_then(|value| String::from_utf8(value).ok())
            .and_then(|value| non_empty(&value).map(str::to_string))
    };
    match (decode(provider.unwrap()), decode(model.unwrap())) {
        (Some(provider), Some(model)) => Ok(Some((provider, model))),
        _ => Err(invalid_model_route_selector(selector)),
    }
}

fn invalid_model_route_selector(selector: &str) -> RuntimeCoreError {
    RuntimeCoreError::InvalidRequest(format!(
        "scheduled task modelId is not a valid model route selector: {selector}"
    ))
}

fn non_empty(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lime_core::models::model_registry::{ModelRuntimeFeature, ModelTaskFamily};

    fn model(provider_id: &str, id: &str, visibility: ModelVisibility) -> EnhancedModelMetadata {
        let mut model = EnhancedModelMetadata::new(
            id.to_string(),
            id.to_string(),
            provider_id.to_string(),
            provider_id.to_string(),
        );
        model.visibility = visibility;
        model.capability_provenance = ModelCapabilityProvenance::ProviderExplicit;
        model.task_families = vec![ModelTaskFamily::Chat];
        model.input_modalities = vec![ModelModality::Text];
        model.output_modalities = vec![ModelModality::Text];
        model.runtime_features = vec![ModelRuntimeFeature::Streaming];
        model.capabilities.streaming = true;
        model
    }

    fn catalog(
        provider_id: &str,
        sort_order: i32,
        models: Vec<EnhancedModelMetadata>,
    ) -> ProviderModelCatalog {
        ProviderModelCatalog {
            provider_id: provider_id.to_string(),
            sort_order,
            models,
        }
    }

    #[test]
    fn model_list_filters_hidden_models_by_default() {
        let response = model_list_from_catalogs(
            ModelListParams::default(),
            vec![catalog(
                "openai",
                0,
                vec![
                    model("openai", "visible", ModelVisibility::List),
                    model("openai", "hidden", ModelVisibility::Hide),
                    model("openai", "disabled", ModelVisibility::None),
                ],
            )],
        )
        .expect("list visible models");

        assert_eq!(
            response
                .data
                .iter()
                .map(|model| model.model.as_str())
                .collect::<Vec<_>>(),
            vec!["visible"]
        );
        assert!(!response.data[0].hidden);
    }

    #[test]
    fn model_list_includes_and_marks_hidden_models_when_requested() {
        let response = model_list_from_catalogs(
            ModelListParams {
                include_hidden: Some(true),
                ..ModelListParams::default()
            },
            vec![catalog(
                "openai",
                0,
                vec![
                    model("openai", "visible", ModelVisibility::List),
                    model("openai", "hidden", ModelVisibility::Hide),
                    model("openai", "disabled", ModelVisibility::None),
                ],
            )],
        )
        .expect("list all models");

        assert_eq!(response.data.len(), 3);
        assert_eq!(
            response
                .data
                .iter()
                .map(|model| model.hidden)
                .collect::<Vec<_>>(),
            vec![false, true, true]
        );
    }

    #[test]
    fn model_list_excludes_inferred_and_non_chat_models_even_when_hidden_requested() {
        let mut inferred = model("openai", "inferred", ModelVisibility::List);
        inferred.capability_provenance = ModelCapabilityProvenance::InferredHint;
        let mut image = model("openai", "image", ModelVisibility::List);
        image.task_families = vec![ModelTaskFamily::ImageGeneration];
        let mut audio = model("openai", "audio", ModelVisibility::List);
        audio.input_modalities = vec![ModelModality::Audio];
        let mut image_output = model("openai", "image-output", ModelVisibility::List);
        image_output.output_modalities = vec![ModelModality::Image];
        let mut non_streaming = model("openai", "non-streaming", ModelVisibility::List);
        non_streaming.runtime_features.clear();
        non_streaming.capabilities.streaming = false;
        let chat = model("openai", "chat", ModelVisibility::Hide);

        let response = model_list_from_catalogs(
            ModelListParams {
                include_hidden: Some(true),
                ..ModelListParams::default()
            },
            vec![catalog(
                "openai",
                0,
                vec![inferred, image, audio, image_output, non_streaming, chat],
            )],
        )
        .expect("list executable chat models");

        assert_eq!(
            response
                .data
                .iter()
                .map(|model| model.model.as_str())
                .collect::<Vec<_>>(),
            vec!["chat"]
        );
        assert!(response.data[0].hidden);
    }

    #[test]
    fn model_list_default_matches_thread_start_default_selection() {
        let catalogs = vec![
            catalog(
                "provider-a",
                0,
                vec![model("provider-a", "model-a", ModelVisibility::List)],
            ),
            catalog(
                "provider-b",
                10,
                vec![model("provider-b", "model-b", ModelVisibility::List)],
            ),
        ];
        let selected = selectable_chat_models(&catalogs)
            .into_iter()
            .next()
            .expect("thread/start default selection");
        let listed = model_list_from_catalogs(ModelListParams::default(), catalogs)
            .expect("model/list default");
        let listed_default = listed
            .data
            .iter()
            .find(|model| model.is_default)
            .expect("one model/list default");

        assert_eq!(listed_default.provider_id, selected.provider);
        assert_eq!(listed_default.model, selected.model);
    }

    #[test]
    fn model_list_projects_grok_style_multimodal_capability_and_limits() {
        let mut metadata = model("grok", "grok-4", ModelVisibility::List);
        metadata.task_families = vec![
            ModelTaskFamily::Chat,
            ModelTaskFamily::Reasoning,
            ModelTaskFamily::VisionUnderstanding,
        ];
        metadata.input_modalities = vec![
            ModelModality::Text,
            ModelModality::Image,
            ModelModality::Video,
            ModelModality::File,
        ];
        metadata.output_modalities = vec![ModelModality::Text, ModelModality::Json];
        metadata.runtime_features = vec![
            ModelRuntimeFeature::Streaming,
            ModelRuntimeFeature::ToolCalling,
        ];
        metadata.capabilities.vision = true;
        metadata.capabilities.tools = true;
        metadata.limits.context_length = Some(256_000);
        metadata.limits.max_output_tokens = Some(64_000);

        let projected = model_from_catalog("grok", metadata);

        assert_eq!(projected.provider_id, "grok");
        assert_eq!(
            projected.input_modalities,
            vec![InputModality::Text, InputModality::Image]
        );
        assert_eq!(
            projected.capability_snapshot.input_modalities,
            vec!["text", "image"]
        );
        assert_eq!(
            projected.capability_snapshot.task_families,
            vec!["chat", "reasoning", "vision_understanding"]
        );
        assert_eq!(
            projected.capability_snapshot.output_modalities,
            vec!["text", "json"]
        );
        assert_eq!(
            projected.capability_snapshot.runtime_features,
            vec!["streaming", "tool_calling"]
        );
        assert!(projected.capability_snapshot.capabilities.vision);
        assert!(projected.capability_snapshot.capabilities.tools);
        assert_eq!(
            projected.capability_snapshot.source.as_deref(),
            Some("provider_explicit")
        );
        assert_eq!(projected.context_window, Some(256_000));
        assert_eq!(projected.max_output_tokens, Some(64_000));
    }

    #[test]
    fn model_list_preserves_only_explicit_multi_agent_version() {
        let implicit = model("openai", "implicit", ModelVisibility::List);
        let mut explicit = model("openai", "explicit", ModelVisibility::List);
        explicit.multi_agent_version = Some(ModelMultiAgentVersion::V2);

        assert_eq!(
            model_from_catalog("openai", implicit).multi_agent_version,
            None
        );
        assert_eq!(
            model_from_catalog("openai", explicit).multi_agent_version,
            Some(MultiAgentVersion::V2)
        );
    }

    #[test]
    fn model_list_uses_codex_offset_pagination_boundaries() {
        let catalogs = vec![catalog(
            "openai",
            0,
            vec![
                model("openai", "first", ModelVisibility::List),
                model("openai", "second", ModelVisibility::List),
            ],
        )];
        let first_page = model_list_from_catalogs(
            ModelListParams {
                limit: Some(0),
                ..ModelListParams::default()
            },
            catalogs.clone(),
        )
        .expect("zero limit is promoted to one");
        let terminal_page = model_list_from_catalogs(
            ModelListParams {
                cursor: Some("2".to_string()),
                limit: Some(1),
                ..ModelListParams::default()
            },
            catalogs,
        )
        .expect("cursor at total is valid");

        assert_eq!(first_page.data.len(), 1);
        assert_eq!(first_page.next_cursor.as_deref(), Some("1"));
        assert!(first_page.data[0].is_default);
        assert!(terminal_page.data.is_empty());
        assert_eq!(terminal_page.next_cursor, None);
    }

    #[test]
    fn model_list_marks_one_stable_default_before_pagination() {
        let catalogs = vec![catalog(
            "openai",
            0,
            vec![
                model("openai", "first", ModelVisibility::List),
                model("openai", "second", ModelVisibility::List),
            ],
        )];
        let full = model_list_from_catalogs(ModelListParams::default(), catalogs.clone())
            .expect("list models");
        let second_page = model_list_from_catalogs(
            ModelListParams {
                cursor: Some("1".to_string()),
                limit: Some(1),
                ..ModelListParams::default()
            },
            catalogs,
        )
        .expect("list second page");

        assert_eq!(
            full.data
                .iter()
                .filter(|model| model.is_default)
                .map(|model| model.model.as_str())
                .collect::<Vec<_>>(),
            vec!["first"]
        );
        assert!(!second_page.data[0].is_default);
    }

    #[test]
    fn model_list_preserves_provider_and_model_catalog_order() {
        let response = model_list_from_catalogs(
            ModelListParams::default(),
            vec![
                catalog(
                    "provider-b",
                    10,
                    vec![
                        model("provider-b", "b-2", ModelVisibility::List),
                        model("provider-b", "b-1", ModelVisibility::List),
                    ],
                ),
                catalog(
                    "provider-a",
                    20,
                    vec![model("provider-a", "a-1", ModelVisibility::List)],
                ),
            ],
        )
        .expect("list ordered models");

        assert_eq!(
            response
                .data
                .iter()
                .map(|model| model.model.as_str())
                .collect::<Vec<_>>(),
            vec!["b-2", "b-1", "a-1"]
        );
    }
}
