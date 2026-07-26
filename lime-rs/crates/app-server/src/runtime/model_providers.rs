use super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::protocol::v2::ModelProviderCapabilitiesReadResponse;
use app_server_protocol::*;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use lime_core::models::model_registry::{EnhancedModelMetadata, ModelModality, ModelVisibility};
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Default)]
pub(in crate::runtime) struct RouteRecoveryCoordinator {
    state: Arc<RouteRecoveryState>,
}

#[derive(Default)]
struct RouteRecoveryState {
    attempted_generation: AtomicU64,
    gate: tokio::sync::Mutex<()>,
}

impl RouteRecoveryCoordinator {
    async fn run<F, Fut, T>(&self, generation: u64, work: F) -> Option<(u64, T)>
    where
        F: FnOnce(u64) -> Fut,
        Fut: Future<Output = T>,
    {
        let _guard = self.state.gate.lock().await;
        if generation <= self.state.attempted_generation.load(Ordering::Acquire) {
            return None;
        }
        let output = work(generation).await;
        self.state
            .attempted_generation
            .store(generation, Ordering::Release);
        Some((generation, output))
    }
}

impl RuntimeCore {
    pub(crate) fn schedule_pending_route_recovery(&self, host: super::RuntimeHostContext) {
        let recovery = self.route_recovery.clone();
        let core = self.clone();
        tokio::spawn(async move {
            let generation = match core.app_data_source.read_model_route_generation().await {
                Ok(generation) => generation,
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "failed to read committed provider route generation"
                    );
                    return;
                }
            };
            let Some((attempted_generation, result)) = recovery
                .run(generation, move |_| async move {
                    core.recover_agent_control_spawns(host, None).await
                })
                .await
            else {
                return;
            };
            if let Err(error) = result {
                if matches!(&error, RuntimeCoreError::PendingRoute { .. }) {
                    tracing::debug!(
                        generation = attempted_generation,
                        error = %error,
                        "provider route recovery is waiting for another committed generation"
                    );
                } else {
                    tracing::warn!(
                        generation = attempted_generation,
                        error = %error,
                        "failed to recover pending routes after provider configuration commit"
                    );
                }
            }
        });
    }

    pub async fn list_models(
        &self,
        params: ModelListParams,
    ) -> Result<ModelListResponse, RuntimeCoreError> {
        model_list_from_catalogs(params, self.model_catalog(None).await?)
    }

    pub(crate) async fn model_catalog(
        &self,
        provider_id: Option<&str>,
    ) -> Result<Vec<super::ProviderModelCatalog>, RuntimeCoreError> {
        self.app_data_source
            .model_catalog(super::ModelCatalogQuery {
                provider_id: provider_id.map(str::to_string),
            })
            .await
    }

    pub async fn list_model_preferences(
        &self,
    ) -> Result<ModelPreferencesListResponse, RuntimeCoreError> {
        self.app_data_source.list_model_preferences().await
    }

    pub async fn read_model_sync_state(
        &self,
    ) -> Result<ModelSyncStateReadResponse, RuntimeCoreError> {
        self.app_data_source.read_model_sync_state().await
    }

    pub async fn list_model_providers(
        &self,
    ) -> Result<ModelProviderListResponse, RuntimeCoreError> {
        self.app_data_source.list_model_providers().await
    }

    pub async fn read_model_provider_capabilities(
        &self,
    ) -> Result<ModelProviderCapabilitiesReadResponse, RuntimeCoreError> {
        self.app_data_source
            .read_model_provider_capabilities()
            .await
    }

    pub async fn list_model_provider_catalog(
        &self,
    ) -> Result<ModelProviderCatalogListResponse, RuntimeCoreError> {
        self.app_data_source.list_model_provider_catalog().await
    }

    pub async fn read_model_provider(
        &self,
        params: ModelProviderReadParams,
    ) -> Result<ModelProviderReadResponse, RuntimeCoreError> {
        self.app_data_source.read_model_provider(params).await
    }

    pub async fn create_model_provider(
        &self,
        params: ModelProviderCreateParams,
    ) -> Result<ModelProviderWriteResponse, RuntimeCoreError> {
        self.app_data_source.create_model_provider(params).await
    }

    pub async fn update_model_provider(
        &self,
        params: ModelProviderUpdateParams,
    ) -> Result<ModelProviderWriteResponse, RuntimeCoreError> {
        self.app_data_source.update_model_provider(params).await
    }

    pub async fn delete_model_provider(
        &self,
        params: ModelProviderDeleteParams,
    ) -> Result<ModelProviderDeleteResponse, RuntimeCoreError> {
        self.app_data_source.delete_model_provider(params).await
    }

    pub async fn update_model_provider_sort_orders(
        &self,
        params: ModelProviderSortOrdersUpdateParams,
    ) -> Result<ModelProviderMutationResponse, RuntimeCoreError> {
        self.app_data_source
            .update_model_provider_sort_orders(params)
            .await
    }

    pub async fn export_model_provider_config(
        &self,
        params: ModelProviderConfigExportParams,
    ) -> Result<ModelProviderConfigExportResponse, RuntimeCoreError> {
        self.app_data_source
            .export_model_provider_config(params)
            .await
    }

    pub async fn import_model_provider_config(
        &self,
        params: ModelProviderConfigImportParams,
    ) -> Result<ModelProviderConfigImportResponse, RuntimeCoreError> {
        self.app_data_source
            .import_model_provider_config(params)
            .await
    }

    pub async fn test_model_provider_connection(
        &self,
        params: ModelProviderTestConnectionParams,
    ) -> Result<ModelProviderTestConnectionResponse, RuntimeCoreError> {
        self.app_data_source
            .test_model_provider_connection(params)
            .await
    }

    pub async fn test_model_provider_chat(
        &self,
        params: ModelProviderTestChatParams,
    ) -> Result<ModelProviderTestChatResponse, RuntimeCoreError> {
        self.app_data_source.test_model_provider_chat(params).await
    }

    pub async fn fetch_model_provider_models(
        &self,
        params: ModelProviderFetchModelsParams,
    ) -> Result<ModelProviderFetchModelsResponse, RuntimeCoreError> {
        self.app_data_source
            .fetch_model_provider_models(params)
            .await
    }

    pub async fn create_model_provider_key(
        &self,
        params: ModelProviderKeyCreateParams,
    ) -> Result<ModelProviderKeyWriteResponse, RuntimeCoreError> {
        self.app_data_source.create_model_provider_key(params).await
    }

    pub async fn update_model_provider_key(
        &self,
        params: ModelProviderKeyUpdateParams,
    ) -> Result<ModelProviderKeyWriteResponse, RuntimeCoreError> {
        self.app_data_source.update_model_provider_key(params).await
    }

    pub async fn delete_model_provider_key(
        &self,
        params: ModelProviderKeyDeleteParams,
    ) -> Result<ModelProviderKeyDeleteResponse, RuntimeCoreError> {
        self.app_data_source.delete_model_provider_key(params).await
    }

    pub async fn read_model_provider_ui_state(
        &self,
        params: ModelProviderUiStateReadParams,
    ) -> Result<ModelProviderUiStateReadResponse, RuntimeCoreError> {
        self.app_data_source
            .read_model_provider_ui_state(params)
            .await
    }

    pub async fn write_model_provider_ui_state(
        &self,
        params: ModelProviderUiStateWriteParams,
    ) -> Result<ModelProviderMutationResponse, RuntimeCoreError> {
        self.app_data_source
            .write_model_provider_ui_state(params)
            .await
    }

    pub async fn read_model_provider_alias(
        &self,
        params: ModelProviderAliasReadParams,
    ) -> Result<ModelProviderAliasReadResponse, RuntimeCoreError> {
        self.app_data_source.read_model_provider_alias(params).await
    }

    pub async fn list_model_provider_aliases(
        &self,
    ) -> Result<ModelProviderAliasListResponse, RuntimeCoreError> {
        self.app_data_source.list_model_provider_aliases().await
    }
}

fn model_list_from_catalogs(
    params: ModelListParams,
    catalogs: Vec<super::ProviderModelCatalog>,
) -> Result<ModelListResponse, RuntimeCoreError> {
    let ModelListParams {
        cursor,
        limit,
        include_hidden,
    } = params;
    let include_hidden = include_hidden.unwrap_or(false);
    let models = catalogs
        .into_iter()
        .flat_map(|catalog| {
            let provider_id = catalog.provider_id;
            catalog
                .models
                .into_iter()
                .filter(move |metadata| {
                    include_hidden || metadata.visibility == ModelVisibility::List
                })
                .map(move |metadata| model_from_catalog(&provider_id, metadata))
        })
        .collect::<Vec<_>>();
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
    let provider_model_id = metadata
        .provider_model_id
        .as_deref()
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .unwrap_or(metadata.id.as_str())
        .to_string();
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
    let mut input_modalities = metadata
        .input_modalities
        .iter()
        .filter_map(|modality| match modality {
            ModelModality::Text => Some(InputModality::Text),
            ModelModality::Image => Some(InputModality::Image),
            ModelModality::Audio => Some(InputModality::Audio),
            _ => None,
        })
        .collect::<Vec<_>>();
    if input_modalities.is_empty() {
        input_modalities.push(InputModality::Text);
    }
    let service_tiers = metadata
        .service_tiers
        .iter()
        .map(|tier| ModelServiceTier {
            id: tier.id.clone(),
            name: tier.name.clone(),
            description: tier.description.clone(),
        })
        .collect::<Vec<_>>();
    let default_service_tier = metadata
        .default_service_tier
        .filter(|default| service_tiers.iter().any(|tier| tier.id == *default));

    Model {
        id: encode_model_route_selector(provider_id, &metadata.id),
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
        supports_personality: false,
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

fn non_empty(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn model(provider_id: &str, id: &str, visibility: ModelVisibility) -> EnhancedModelMetadata {
        let mut model = EnhancedModelMetadata::new(
            id.to_string(),
            id.to_string(),
            provider_id.to_string(),
            provider_id.to_string(),
        );
        model.visibility = visibility;
        model
    }

    fn catalog(
        provider_id: &str,
        sort_order: i32,
        models: Vec<EnhancedModelMetadata>,
    ) -> super::super::ProviderModelCatalog {
        super::super::ProviderModelCatalog {
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
        assert!(terminal_page.data.is_empty());
        assert_eq!(terminal_page.next_cursor, None);
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

    #[tokio::test]
    async fn recovery_coalesces_commits_visible_before_the_worker_runs() {
        let recovery = RouteRecoveryCoordinator::default();
        let committed_generation = 2;
        let calls = Arc::new(AtomicUsize::new(0));

        let first_calls = calls.clone();
        let first = recovery
            .run(committed_generation, move |target_generation| async move {
                first_calls.fetch_add(1, Ordering::AcqRel);
                target_generation
            })
            .await;
        let second_calls = calls.clone();
        let second = recovery
            .run(committed_generation, move |target_generation| async move {
                second_calls.fetch_add(1, Ordering::AcqRel);
                target_generation
            })
            .await;

        assert_eq!(first, Some((committed_generation, committed_generation)));
        assert_eq!(second, None);
        assert_eq!(calls.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn recovery_preserves_a_commit_that_arrives_during_an_attempt() {
        let recovery = RouteRecoveryCoordinator::default();
        let first_generation = 1;
        let calls = Arc::new(AtomicUsize::new(0));
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();

        let first_recovery = recovery.clone();
        let first_calls = calls.clone();
        let first = tokio::spawn(async move {
            first_recovery
                .run(first_generation, move |target_generation| async move {
                    first_calls.fetch_add(1, Ordering::AcqRel);
                    let _ = entered_tx.send(());
                    let _ = release_rx.await;
                    target_generation
                })
                .await
        });
        entered_rx.await.expect("first recovery entered");

        let second_generation = 2;
        let second_recovery = recovery.clone();
        let second_calls = calls.clone();
        let second = tokio::spawn(async move {
            second_recovery
                .run(second_generation, move |target_generation| async move {
                    second_calls.fetch_add(1, Ordering::AcqRel);
                    target_generation
                })
                .await
        });

        release_tx.send(()).expect("release first recovery");
        assert_eq!(
            first.await.expect("first recovery task"),
            Some((first_generation, first_generation))
        );
        assert_eq!(
            second.await.expect("second recovery task"),
            Some((second_generation, second_generation))
        );
        assert_eq!(calls.load(Ordering::Acquire), 2);
    }
}
