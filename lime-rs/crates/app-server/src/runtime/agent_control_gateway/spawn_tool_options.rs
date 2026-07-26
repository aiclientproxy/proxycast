use super::*;
use app_server_protocol::RuntimeOptions;
use lime_core::models::model_registry::{EnhancedModelMetadata, ModelVisibility};
use std::collections::HashSet;
use tool_runtime::agent_control::{SpawnAgentModelOption, SpawnAgentToolOptions};

pub(in crate::runtime) async fn load(
    core: &RuntimeCore,
    runtime_options: Option<&RuntimeOptions>,
) -> SpawnAgentToolOptions {
    let provider_id = runtime_options
        .and_then(RuntimeOptions::runtime_request)
        .and_then(|request| {
            request.provider_preference.as_deref().or_else(|| {
                request
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.pointer("/agentControlRoute/providerPreference"))
                    .and_then(serde_json::Value::as_str)
            })
        })
        .map(str::trim)
        .filter(|provider| !provider.is_empty())
        .map(str::to_string);
    let Some(provider_id) = provider_id else {
        return SpawnAgentToolOptions::default();
    };
    match core.model_catalog(Some(&provider_id)).await {
        Ok(catalogs) => from_models(
            catalogs
                .into_iter()
                .flat_map(|catalog| catalog.models)
                .collect(),
        ),
        Err(error) => {
            tracing::warn!(
                provider_id,
                error = %error,
                "failed to load spawn_agent model override catalog"
            );
            SpawnAgentToolOptions::default()
        }
    }
}

fn from_models(models: Vec<EnhancedModelMetadata>) -> SpawnAgentToolOptions {
    let mut seen = HashSet::new();
    let available_models = models
        .into_iter()
        .filter(|model| model.visibility == ModelVisibility::List)
        .filter(|model| seen.insert(model.id.clone()))
        .map(|model| SpawnAgentModelOption {
            description: model
                .description
                .clone()
                .filter(|description| !description.trim().is_empty())
                .unwrap_or_else(|| model.display_name.clone()),
            supported_reasoning_efforts: reasoning_efforts(&model),
            default_reasoning_effort: model
                .capabilities
                .reasoning_effort
                .as_ref()
                .and_then(|support| support.default.clone()),
            service_tiers: model
                .service_tiers
                .iter()
                .map(|tier| tier.id.clone())
                .collect(),
            model: model.id,
        })
        .collect();
    SpawnAgentToolOptions { available_models }
}

fn reasoning_efforts(model: &EnhancedModelMetadata) -> Vec<String> {
    let Some(support) = model.capabilities.reasoning_effort.as_ref() else {
        return Vec::new();
    };
    let values = if support.options.is_empty() {
        support.levels.iter().collect::<Vec<_>>()
    } else {
        support
            .options
            .iter()
            .map(|option| &option.value)
            .collect::<Vec<_>>()
    };
    values
        .into_iter()
        .map(|effort| effort.trim())
        .filter(|effort| !effort.is_empty())
        .fold(Vec::new(), |mut efforts, effort| {
            if !efforts.iter().any(|current| current == effort) {
                efforts.push(effort.to_string());
            }
            efforts
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lime_core::models::model_registry::{
        ModelCapabilities, ModelReasoningEffortOption, ModelReasoningEffortSupport,
        ModelServiceTier,
    };

    #[test]
    fn projects_only_picker_visible_models_with_typed_controls() {
        let mut visible = EnhancedModelMetadata::new(
            "gpt-5.6-sol".to_string(),
            "GPT-5.6 Sol".to_string(),
            "openai".to_string(),
            "OpenAI".to_string(),
        );
        visible.description = Some("Frontier coding model".to_string());
        visible.capabilities = ModelCapabilities {
            reasoning: true,
            reasoning_effort: Some(ModelReasoningEffortSupport {
                supported: true,
                levels: Vec::new(),
                options: vec![
                    ModelReasoningEffortOption {
                        id: "low".to_string(),
                        value: "low".to_string(),
                        label: "Low".to_string(),
                        description: None,
                        default: false,
                    },
                    ModelReasoningEffortOption {
                        id: "medium".to_string(),
                        value: "medium".to_string(),
                        label: "Medium".to_string(),
                        description: None,
                        default: true,
                    },
                ],
                default: Some("medium".to_string()),
                source: None,
            }),
            ..ModelCapabilities::default()
        };
        visible.service_tiers = vec![ModelServiceTier {
            id: "priority".to_string(),
            name: "Priority".to_string(),
            description: "Fast queue".to_string(),
        }];
        let mut hidden = EnhancedModelMetadata::new(
            "internal-model".to_string(),
            "Internal Model".to_string(),
            "openai".to_string(),
            "OpenAI".to_string(),
        );
        hidden.visibility = ModelVisibility::Hide;

        let options = from_models(vec![visible, hidden]);

        assert_eq!(options.available_models.len(), 1);
        assert_eq!(options.available_models[0].model, "gpt-5.6-sol");
        assert_eq!(
            options.available_models[0].supported_reasoning_efforts,
            vec!["low", "medium"]
        );
        assert_eq!(options.available_models[0].service_tiers, vec!["priority"]);
    }
}
