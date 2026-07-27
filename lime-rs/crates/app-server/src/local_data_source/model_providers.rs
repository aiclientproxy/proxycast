use super::data_error;
use super::model_projection::model_info_from_value;
use super::model_projection::provider_info_from_value;
use super::model_projection::provider_key_info_from_value;
use super::values_from_serializable_vec;
use crate::runtime_backend::configured_provider_readiness;
use crate::RuntimeCoreError;
use app_server_protocol::protocol::v2::ModelProviderCapabilitiesReadResponse;
use app_server_protocol::ModelPreferencesListResponse;
use app_server_protocol::ModelProviderAliasListResponse;
use app_server_protocol::ModelProviderAliasReadParams;
use app_server_protocol::ModelProviderAliasReadResponse;
use app_server_protocol::ModelProviderCatalogListResponse;
use app_server_protocol::ModelProviderConfigExportParams;
use app_server_protocol::ModelProviderConfigExportResponse;
use app_server_protocol::ModelProviderConfigImportParams;
use app_server_protocol::ModelProviderConfigImportResponse;
use app_server_protocol::ModelProviderCreateParams;
use app_server_protocol::ModelProviderDeleteParams;
use app_server_protocol::ModelProviderDeleteResponse;
use app_server_protocol::ModelProviderFetchModelsParams;
use app_server_protocol::ModelProviderFetchModelsResponse;
use app_server_protocol::ModelProviderKeyCreateParams;
use app_server_protocol::ModelProviderKeyDeleteParams;
use app_server_protocol::ModelProviderKeyDeleteResponse;
use app_server_protocol::ModelProviderKeyUpdateParams;
use app_server_protocol::ModelProviderKeyWriteResponse;
use app_server_protocol::ModelProviderListResponse;
use app_server_protocol::ModelProviderMutationResponse;
use app_server_protocol::ModelProviderReadParams;
use app_server_protocol::ModelProviderReadResponse;
use app_server_protocol::ModelProviderSortOrdersUpdateParams;
use app_server_protocol::ModelProviderTestChatParams;
use app_server_protocol::ModelProviderTestChatResponse;
use app_server_protocol::ModelProviderTestConnectionParams;
use app_server_protocol::ModelProviderTestConnectionResponse;
use app_server_protocol::ModelProviderUiStateReadParams;
use app_server_protocol::ModelProviderUiStateReadResponse;
use app_server_protocol::ModelProviderUiStateWriteParams;
use app_server_protocol::ModelProviderUpdateParams;
use app_server_protocol::ModelProviderWriteResponse;
use app_server_protocol::ModelSyncStateReadResponse;
use app_server_protocol::ProviderModelConfig as ProtocolProviderModelConfig;
use lime_core::database::dao::api_key_provider::ApiKeyEntry;
use lime_core::database::dao::api_key_provider::ApiKeyProvider;
use lime_core::database::dao::api_key_provider::ApiProviderPromptCacheMode;
use lime_core::database::dao::api_key_provider::ApiProviderType;
use lime_core::database::dao::api_key_provider::ProviderWithKeys;
use lime_core::database::dao::route_state::RouteStateDao;
use lime_core::database::system_providers::get_system_providers;
use lime_core::database::system_providers::SystemProviderDef;
use lime_core::database::DbConnection;
use lime_core::models::model_registry::EnhancedModelMetadata;
use lime_core::models::model_registry::{
    ModelCapabilities as CoreModelCapabilities, ModelReasoningEffortOption,
    ModelReasoningEffortSupport, ProviderModelCapability as CoreProviderModelCapability,
    ProviderModelConfig as CoreProviderModelConfig,
};
use lime_core::models::runtime_api_key_credential_uuid;
use lime_services::api_key_provider_service::ApiKeyProviderService;
use lime_services::model_registry_service::{
    FetchModelsResult, ModelRegistryService, ProviderModelCacheAccess,
};
use serde_json::json;
use serde_json::Map;
use serde_json::Value;
use std::collections::HashSet;

pub(crate) fn read_model_route_generation(db: &DbConnection) -> Result<u64, RuntimeCoreError> {
    let conn = lime_core::database::lock_db(db).map_err(data_error)?;
    RouteStateDao::read_generation(&conn).map_err(data_error)
}

pub(crate) fn model_catalog(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    model_registry_service: &ModelRegistryService,
    query: crate::ModelCatalogQuery,
) -> Result<Vec<crate::ProviderModelCatalog>, RuntimeCoreError> {
    let provider_filter = query
        .provider_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let providers = api_key_provider_service
        .get_all_providers(db)
        .map_err(data_error)?;
    let mut catalogs = Vec::new();

    for provider in providers {
        if !configured_provider_readiness(&provider).ready {
            continue;
        }
        let provider_id = provider.provider.id.as_str();
        if provider_filter.is_some_and(|filter| filter != provider_id) {
            continue;
        }
        let mut models = Vec::new();
        let mut seen = HashSet::new();

        for model_id in &provider.provider.models {
            append_model(
                &mut models,
                &mut seen,
                model_registry_service.build_declared_model_metadata(provider_id, model_id),
            );
        }

        append_cached_provider_models(
            db,
            api_key_provider_service,
            model_registry_service,
            &provider,
            &mut models,
            &mut seen,
        );
        catalogs.push(crate::ProviderModelCatalog {
            provider_id: provider_id.to_string(),
            sort_order: provider.provider.sort_order,
            models,
        });
    }

    catalogs.sort_by(|left, right| {
        left.sort_order
            .cmp(&right.sort_order)
            .then_with(|| left.provider_id.cmp(&right.provider_id))
    });
    Ok(catalogs)
}

fn append_model(
    models: &mut Vec<EnhancedModelMetadata>,
    seen: &mut HashSet<String>,
    model: EnhancedModelMetadata,
) {
    let key = model_dedupe_key(&model);
    if seen.insert(key) {
        models.push(model);
    }
}

fn append_cached_provider_models(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    model_registry_service: &ModelRegistryService,
    provider: &ProviderWithKeys,
    models: &mut Vec<EnhancedModelMetadata>,
    seen: &mut HashSet<String>,
) {
    let provider_id = provider.provider.id.as_str();
    let api_host = provider.provider.api_host.as_str();
    let provider_type = provider.provider.effective_provider_type();
    let enabled_keys = provider.api_keys.iter().filter(|key| key.enabled);
    let mut has_enabled_key = false;

    for key in enabled_keys {
        has_enabled_key = true;
        let credential_ref = runtime_api_key_credential_uuid(&key.id);
        let credential = match api_key_provider_service.select_runtime_credential_by_ref(
            db,
            provider_id,
            &credential_ref,
        ) {
            Ok(Some(credential)) => credential,
            Ok(None) => continue,
            Err(error) => {
                tracing::warn!(
                    "[ModelProvider] 读取 Provider scoped 模型缓存凭证失败: provider={}, key_id={}, error={}",
                    provider_id,
                    key.id,
                    error
                );
                continue;
            }
        };
        append_cached_models_for_access(
            model_registry_service,
            provider_id,
            api_host,
            provider_type,
            ProviderModelCacheAccess::Credential(&credential),
            models,
            seen,
        );
    }

    if !has_enabled_key
        && !ModelRegistryService::requires_api_key_for_runtime(provider_id, api_host, provider_type)
    {
        append_cached_models_for_access(
            model_registry_service,
            provider_id,
            api_host,
            provider_type,
            ProviderModelCacheAccess::Keyless,
            models,
            seen,
        );
    }
}

fn append_cached_models_for_access(
    model_registry_service: &ModelRegistryService,
    provider_id: &str,
    api_host: &str,
    provider_type: ApiProviderType,
    cache_access: ProviderModelCacheAccess<'_>,
    models: &mut Vec<EnhancedModelMetadata>,
    seen: &mut HashSet<String>,
) {
    match model_registry_service.get_cached_provider_models_for_access(
        provider_id,
        api_host,
        Some(provider_type),
        cache_access,
    ) {
        Ok(Some(result)) => {
            for model in result.models {
                append_model(models, seen, model);
            }
        }
        Ok(None) => {}
        Err(error) => {
            tracing::warn!(
                "[ModelProvider] 读取 Provider 模型缓存失败，跳过缓存模型: provider={}, error={}",
                provider_id,
                error
            );
        }
    }
}

fn model_dedupe_key(model: &EnhancedModelMetadata) -> String {
    model.id.trim().to_ascii_lowercase()
}

pub(crate) async fn list_model_preferences(
    model_registry_service: &ModelRegistryService,
) -> Result<ModelPreferencesListResponse, RuntimeCoreError> {
    let preferences = model_registry_service
        .get_all_preferences()
        .await
        .map_err(data_error)?;
    Ok(ModelPreferencesListResponse {
        preferences: values_from_serializable_vec(preferences)?,
    })
}

pub(crate) async fn read_model_sync_state(
    model_registry_service: &ModelRegistryService,
) -> Result<ModelSyncStateReadResponse, RuntimeCoreError> {
    Ok(ModelSyncStateReadResponse {
        sync_state: serde_json::to_value(model_registry_service.get_sync_state().await)
            .map_err(data_error)?,
    })
}

pub(crate) fn list_model_providers(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
) -> Result<ModelProviderListResponse, RuntimeCoreError> {
    let providers = api_key_provider_service
        .get_all_providers(db)
        .map_err(data_error)?
        .iter()
        .map(|provider| provider_with_keys_to_value(provider, api_key_provider_service))
        .map(|provider| provider_info_from_value(&provider))
        .collect();
    Ok(ModelProviderListResponse { providers })
}

pub(crate) fn read_model_provider_capabilities(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
) -> Result<ModelProviderCapabilitiesReadResponse, RuntimeCoreError> {
    let config = lime_core::config::load_config().map_err(data_error)?;
    resolve_model_provider_capabilities(
        db,
        api_key_provider_service,
        config.default_provider.trim(),
    )
}

fn resolve_model_provider_capabilities(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    provider_id: &str,
) -> Result<ModelProviderCapabilitiesReadResponse, RuntimeCoreError> {
    let unavailable = || ModelProviderCapabilitiesReadResponse {
        namespace_tools: false,
        image_generation: false,
        web_search: false,
    };
    let provider_id = provider_id.trim();
    if provider_id.is_empty() {
        return Ok(unavailable());
    }
    let Some(provider) = api_key_provider_service
        .get_provider(db, provider_id)
        .map_err(data_error)?
    else {
        return Ok(unavailable());
    };
    if !configured_provider_readiness(&provider).ready {
        return Ok(unavailable());
    }
    let provider_type = provider.provider.effective_provider_type().to_string();
    let Some(capabilities) =
        model_provider::provider_capabilities::ProviderCapabilities::from_provider_route(
            &provider_type,
            Some(&provider.provider.api_host),
        )
    else {
        return Ok(unavailable());
    };

    Ok(ModelProviderCapabilitiesReadResponse {
        namespace_tools: capabilities.namespace_tools,
        image_generation: capabilities.image_generation,
        web_search: capabilities.web_search,
    })
}

pub(crate) fn list_model_provider_catalog(
) -> Result<ModelProviderCatalogListResponse, RuntimeCoreError> {
    let providers: Vec<Value> = get_system_providers()
        .into_iter()
        .map(system_provider_to_value)
        .collect();
    Ok(ModelProviderCatalogListResponse {
        providers: providers.iter().map(provider_info_from_value).collect(),
    })
}

pub(crate) fn read_model_provider(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    params: ModelProviderReadParams,
) -> Result<ModelProviderReadResponse, RuntimeCoreError> {
    let provider = api_key_provider_service
        .get_provider(db, &params.provider_id)
        .map_err(data_error)?
        .map(|provider| provider_with_keys_to_value(&provider, api_key_provider_service))
        .map(|provider| provider_info_from_value(&provider));
    Ok(ModelProviderReadResponse { provider })
}

pub(crate) fn create_model_provider(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    params: ModelProviderCreateParams,
) -> Result<ModelProviderWriteResponse, RuntimeCoreError> {
    let provider_type = params
        .provider_type
        .parse::<ApiProviderType>()
        .map_err(data_error)?;
    let provider = api_key_provider_service
        .add_custom_provider(
            db,
            params.name,
            provider_type,
            params.api_host,
            params.api_version,
            params.project,
            params.location,
            params.region,
            parse_prompt_cache_mode(params.prompt_cache_mode)?,
        )
        .map_err(data_error)?;
    let provider = provider_to_value(&provider, 0);
    Ok(ModelProviderWriteResponse {
        provider: provider_info_from_value(&provider),
    })
}

pub(crate) fn update_model_provider(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    params: ModelProviderUpdateParams,
) -> Result<ModelProviderWriteResponse, RuntimeCoreError> {
    let provider_type = params
        .provider_type
        .map(|value| value.parse::<ApiProviderType>())
        .transpose()
        .map_err(data_error)?;
    let models = params
        .models
        .map(|models| {
            models
                .into_iter()
                .map(provider_model_config_to_core)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let provider = api_key_provider_service
        .update_provider(
            db,
            &params.provider_id,
            params.name,
            provider_type,
            params.api_host,
            params.enabled,
            params.sort_order,
            params.api_version,
            params.project,
            params.location,
            params.region,
            parse_prompt_cache_mode(params.prompt_cache_mode)?,
            models,
        )
        .map_err(data_error)?;
    let api_key_count = api_key_provider_service
        .get_provider(db, &params.provider_id)
        .map_err(data_error)?
        .map(|provider| provider.api_keys.len())
        .unwrap_or(0);
    let provider = provider_to_value(&provider, api_key_count);
    Ok(ModelProviderWriteResponse {
        provider: provider_info_from_value(&provider),
    })
}

fn provider_model_config_to_core(
    model: ProtocolProviderModelConfig,
) -> Result<CoreProviderModelConfig, RuntimeCoreError> {
    Ok(CoreProviderModelConfig {
        id: model.id,
        display_name: model.display_name,
        capability: model
            .capability
            .map(|capability| {
                Ok(CoreProviderModelCapability {
                    task_families: parse_provider_capability_values(
                        "taskFamilies",
                        capability.task_families,
                    )?,
                    input_modalities: parse_provider_capability_values(
                        "inputModalities",
                        capability.input_modalities,
                    )?,
                    output_modalities: parse_provider_capability_values(
                        "outputModalities",
                        capability.output_modalities,
                    )?,
                    runtime_features: parse_provider_capability_values(
                        "runtimeFeatures",
                        capability.runtime_features,
                    )?,
                    capabilities: CoreModelCapabilities {
                        vision: capability.capabilities.vision,
                        tools: capability.capabilities.tools,
                        streaming: capability.capabilities.streaming,
                        json_mode: capability.capabilities.json_mode,
                        function_calling: capability.capabilities.function_calling,
                        reasoning: capability.capabilities.reasoning,
                        reasoning_effort: capability
                            .capabilities
                            .reasoning_effort
                            .map(|support| {
                                Ok(ModelReasoningEffortSupport {
                                    supported: support.supported,
                                    levels: support.levels,
                                    options: support
                                        .options
                                        .into_iter()
                                        .map(|option| ModelReasoningEffortOption {
                                            id: option.id,
                                            value: option.value,
                                            label: option.label,
                                            description: option.description,
                                            default: option.default,
                                        })
                                        .collect(),
                                    default: support.default,
                                    source: support
                                        .source
                                        .map(|source| {
                                            parse_provider_capability_value(
                                                "reasoningEffort.source",
                                                source,
                                            )
                                        })
                                        .transpose()?,
                                })
                            })
                            .transpose()?,
                    },
                })
            })
            .transpose()?,
    })
}

fn parse_provider_capability_values<T>(
    field: &str,
    values: Vec<String>,
) -> Result<Vec<T>, RuntimeCoreError>
where
    T: serde::de::DeserializeOwned,
{
    values
        .into_iter()
        .map(|value| parse_provider_capability_value(field, value))
        .collect()
}

fn parse_provider_capability_value<T>(field: &str, value: String) -> Result<T, RuntimeCoreError>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_value(Value::String(value.clone())).map_err(|error| {
        data_error(format!(
            "Provider model capability {field} contains invalid value `{value}`: {error}"
        ))
    })
}

pub(crate) fn delete_model_provider(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    params: ModelProviderDeleteParams,
) -> Result<ModelProviderDeleteResponse, RuntimeCoreError> {
    let deleted = api_key_provider_service
        .delete_custom_provider(db, &params.provider_id)
        .map_err(data_error)?;
    Ok(ModelProviderDeleteResponse { deleted })
}

pub(crate) fn update_model_provider_sort_orders(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    params: ModelProviderSortOrdersUpdateParams,
) -> Result<ModelProviderMutationResponse, RuntimeCoreError> {
    let sort_orders = params
        .sort_orders
        .into_iter()
        .map(|item| (item.provider_id, item.sort_order))
        .collect();
    api_key_provider_service
        .update_provider_sort_orders(db, sort_orders)
        .map_err(data_error)?;
    Ok(ModelProviderMutationResponse::default())
}

pub(crate) fn export_model_provider_config(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    params: ModelProviderConfigExportParams,
) -> Result<ModelProviderConfigExportResponse, RuntimeCoreError> {
    let config = api_key_provider_service
        .export_config(db, params.include_keys.unwrap_or(false))
        .map_err(data_error)?;
    let config_json = serde_json::to_string_pretty(&config).map_err(data_error)?;
    Ok(ModelProviderConfigExportResponse { config_json })
}

pub(crate) fn import_model_provider_config(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    params: ModelProviderConfigImportParams,
) -> Result<ModelProviderConfigImportResponse, RuntimeCoreError> {
    let result = api_key_provider_service
        .import_config(db, &params.config_json)
        .map_err(data_error)?;
    Ok(ModelProviderConfigImportResponse {
        success: result.success,
        imported_providers: result.imported_providers,
        imported_api_keys: result.imported_api_keys,
        skipped_providers: result.skipped_providers,
        errors: result.errors,
    })
}

pub(crate) async fn test_model_provider_connection(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    params: ModelProviderTestConnectionParams,
) -> Result<ModelProviderTestConnectionResponse, RuntimeCoreError> {
    let result = api_key_provider_service
        .test_connection_with_fallback_models(
            db,
            &params.provider_id,
            params.model_name,
            Vec::new(),
        )
        .await
        .map_err(data_error)?;
    Ok(ModelProviderTestConnectionResponse {
        success: result.success,
        latency_ms: result.latency_ms,
        error: result.error,
        models: result.models,
    })
}

pub(crate) async fn test_model_provider_chat(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    params: ModelProviderTestChatParams,
) -> Result<ModelProviderTestChatResponse, RuntimeCoreError> {
    let result = api_key_provider_service
        .test_chat_with_fallback_models(
            db,
            &params.provider_id,
            params.model_name,
            params.prompt,
            Vec::new(),
        )
        .await
        .map_err(data_error)?;
    Ok(ModelProviderTestChatResponse {
        success: result.success,
        latency_ms: result.latency_ms,
        error: result.error,
        content: result.content,
        raw: result.raw,
    })
}

pub(crate) async fn fetch_model_provider_models(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    model_registry_service: &ModelRegistryService,
    params: ModelProviderFetchModelsParams,
) -> Result<ModelProviderFetchModelsResponse, RuntimeCoreError> {
    let provider = api_key_provider_service
        .get_provider(db, &params.provider_id)
        .map_err(data_error)?
        .ok_or_else(|| data_error(format!("Provider 不存在: {}", params.provider_id)))?;
    let api_host = provider.provider.api_host.clone();
    if api_host.trim().is_empty() {
        return Err(data_error("Provider 没有配置 API Host"));
    }
    let provider_type = provider.provider.effective_provider_type();
    let requires_api_key = ModelRegistryService::requires_api_key_for_model_fetch(
        &params.provider_id,
        &api_host,
        provider_type,
    );
    let api_key = if requires_api_key {
        api_key_provider_service
            .get_next_api_key(db, &params.provider_id)
            .map_err(data_error)?
            .ok_or_else(|| {
                data_error(format!(
                    "Provider {} 没有可用的 API Key",
                    params.provider_id
                ))
            })?
    } else {
        api_key_provider_service
            .get_next_api_key(db, &params.provider_id)
            .map_err(data_error)?
            .unwrap_or_default()
    };
    let result = model_registry_service
        .fetch_models_from_api_with_hints(
            &params.provider_id,
            &api_host,
            &api_key,
            Some(provider_type),
            &provider.provider.models,
        )
        .await
        .map_err(data_error)?;
    fetch_models_result_to_response(result)
}

pub(crate) fn create_model_provider_key(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    params: ModelProviderKeyCreateParams,
) -> Result<ModelProviderKeyWriteResponse, RuntimeCoreError> {
    let key = api_key_provider_service
        .add_api_key(
            db,
            &params.provider_id,
            &params.api_key,
            params.alias,
            params.replace_existing.unwrap_or(false),
        )
        .map_err(data_error)?;
    Ok(ModelProviderKeyWriteResponse {
        key: provider_key_info_from_value(&api_key_to_value(&key, api_key_provider_service)),
    })
}

pub(crate) fn update_model_provider_key(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    params: ModelProviderKeyUpdateParams,
) -> Result<ModelProviderKeyWriteResponse, RuntimeCoreError> {
    let key = if let Some(enabled) = params.enabled {
        api_key_provider_service
            .toggle_api_key(db, &params.key_id, enabled)
            .map_err(data_error)?
    } else {
        api_key_provider_service
            .update_api_key_alias(db, &params.key_id, params.alias.clone())
            .map_err(data_error)?
    };
    let key = if params.enabled.is_some() && params.alias.is_some() {
        api_key_provider_service
            .update_api_key_alias(db, &params.key_id, params.alias)
            .map_err(data_error)?
    } else {
        key
    };
    Ok(ModelProviderKeyWriteResponse {
        key: provider_key_info_from_value(&api_key_to_value(&key, api_key_provider_service)),
    })
}

pub(crate) fn delete_model_provider_key(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    params: ModelProviderKeyDeleteParams,
) -> Result<ModelProviderKeyDeleteResponse, RuntimeCoreError> {
    let deleted = api_key_provider_service
        .delete_api_key(db, &params.key_id)
        .map_err(data_error)?;
    Ok(ModelProviderKeyDeleteResponse { deleted })
}

pub(crate) fn read_model_provider_ui_state(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    params: ModelProviderUiStateReadParams,
) -> Result<ModelProviderUiStateReadResponse, RuntimeCoreError> {
    let value = api_key_provider_service
        .get_ui_state(db, &params.key)
        .map_err(data_error)?;
    Ok(ModelProviderUiStateReadResponse { value })
}

pub(crate) fn write_model_provider_ui_state(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    params: ModelProviderUiStateWriteParams,
) -> Result<ModelProviderMutationResponse, RuntimeCoreError> {
    api_key_provider_service
        .set_ui_state(db, &params.key, &params.value)
        .map_err(data_error)?;
    Ok(ModelProviderMutationResponse::default())
}

pub(crate) async fn read_model_provider_alias(
    model_registry_service: &ModelRegistryService,
    params: ModelProviderAliasReadParams,
) -> Result<ModelProviderAliasReadResponse, RuntimeCoreError> {
    Ok(ModelProviderAliasReadResponse {
        config: model_registry_service
            .get_provider_alias_config(&params.provider)
            .await
            .map(serde_json::to_value)
            .transpose()
            .map_err(data_error)?,
    })
}

pub(crate) async fn list_model_provider_aliases(
    model_registry_service: &ModelRegistryService,
) -> Result<ModelProviderAliasListResponse, RuntimeCoreError> {
    let mut configs = Map::new();
    for (provider, config) in model_registry_service.get_all_alias_configs().await {
        configs.insert(provider, serde_json::to_value(config).map_err(data_error)?);
    }
    Ok(ModelProviderAliasListResponse { configs })
}

fn provider_with_keys_to_value(
    provider_with_keys: &ProviderWithKeys,
    service: &ApiKeyProviderService,
) -> Value {
    let provider = &provider_with_keys.provider;
    let api_keys: Vec<Value> = provider_with_keys
        .api_keys
        .iter()
        .map(|api_key| api_key_to_value(api_key, service))
        .collect();
    json!({
        "id": provider.id,
        "name": provider.name,
        "type": provider.effective_provider_type().to_string(),
        "api_host": provider.api_host,
        "is_system": provider.is_system,
        "group": provider.group.to_string(),
        "enabled": provider.enabled,
        "sort_order": provider.sort_order,
        "api_version": provider.api_version,
        "project": provider.project,
        "location": provider.location,
        "region": provider.region,
        "models": provider.models,
        "prompt_cache_mode": provider.effective_prompt_cache_mode().map(|mode| mode.to_string()),
        "api_key_count": provider_with_keys.api_keys.len(),
        "created_at": provider.created_at.to_rfc3339(),
        "updated_at": provider.updated_at.to_rfc3339(),
        "api_keys": api_keys,
    })
}

fn provider_to_value(provider: &ApiKeyProvider, api_key_count: usize) -> Value {
    json!({
        "id": provider.id,
        "name": provider.name,
        "type": provider.effective_provider_type().to_string(),
        "api_host": provider.api_host,
        "is_system": provider.is_system,
        "group": provider.group.to_string(),
        "enabled": provider.enabled,
        "sort_order": provider.sort_order,
        "api_version": provider.api_version,
        "project": provider.project,
        "location": provider.location,
        "region": provider.region,
        "models": provider.models,
        "prompt_cache_mode": provider.effective_prompt_cache_mode().map(|mode| mode.to_string()),
        "api_key_count": api_key_count,
        "created_at": provider.created_at.to_rfc3339(),
        "updated_at": provider.updated_at.to_rfc3339(),
    })
}

fn api_key_to_value(api_key: &ApiKeyEntry, service: &ApiKeyProviderService) -> Value {
    let api_key_masked = service
        .decrypt_api_key(&api_key.api_key_encrypted)
        .map(|decrypted| mask_api_key(&decrypted))
        .unwrap_or_else(|_| "****".to_string());
    json!({
        "id": api_key.id,
        "provider_id": api_key.provider_id,
        "api_key_masked": api_key_masked,
        "alias": api_key.alias,
        "enabled": api_key.enabled,
        "usage_count": api_key.usage_count,
        "error_count": api_key.error_count,
        "last_used_at": api_key.last_used_at.map(|value| value.to_rfc3339()),
        "created_at": api_key.created_at.to_rfc3339(),
    })
}

fn fetch_models_result_to_response(
    result: FetchModelsResult,
) -> Result<ModelProviderFetchModelsResponse, RuntimeCoreError> {
    let models = values_from_serializable_vec(result.models)?;
    Ok(ModelProviderFetchModelsResponse {
        models: models.iter().map(model_info_from_value).collect(),
        source: serde_json::to_value(result.source)
            .map_err(data_error)?
            .as_str()
            .unwrap_or("Error")
            .to_string(),
        error: result.error,
        request_url: result.request_url,
        diagnostic_hint: result.diagnostic_hint,
        error_kind: result
            .error_kind
            .map(serde_json::to_value)
            .transpose()
            .map_err(data_error)?
            .and_then(|value| value.as_str().map(str::to_string)),
        should_prompt_error: result.should_prompt_error,
        from_cache: result.from_cache,
    })
}

fn parse_prompt_cache_mode(
    value: Option<String>,
) -> Result<Option<ApiProviderPromptCacheMode>, RuntimeCoreError> {
    value
        .map(|mode| {
            mode.parse::<ApiProviderPromptCacheMode>()
                .map_err(data_error)
        })
        .transpose()
}

fn mask_api_key(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    if chars.len() <= 12 {
        "****".to_string()
    } else {
        let prefix: String = chars[..6].iter().collect();
        let suffix: String = chars[chars.len() - 4..].iter().collect();
        format!("{prefix}****{suffix}")
    }
}

fn system_provider_to_value(provider: SystemProviderDef) -> Value {
    json!({
        "id": provider.id,
        "name": provider.name,
        "type": provider.provider_type.to_string(),
        "api_host": provider.api_host,
        "group": provider.group.to_string(),
        "sort_order": provider.sort_order,
        "api_version": provider.api_version,
        "legacy_ids": legacy_provider_ids(provider.id),
    })
}

fn legacy_provider_ids(provider_id: &str) -> Vec<String> {
    match provider_id {
        "lime-hub" => vec![format!("{}{}", "lobe", "hub")],
        "google" => vec!["gemini".to_string()],
        "zhipuai" => vec!["zhipu".to_string()],
        "alibaba" => vec!["dashscope".to_string(), "qwen".to_string()],
        "moonshotai" => vec!["moonshot".to_string()],
        "xai" => vec!["grok".to_string()],
        "github-models" => vec!["github".to_string()],
        "github-copilot" => vec!["copilot".to_string()],
        "google-vertex" => vec!["vertexai".to_string()],
        "azure-openai" => vec!["azure".to_string()],
        "amazon-bedrock" => vec!["aws-bedrock".to_string(), "bedrock".to_string()],
        "togetherai" => vec!["together".to_string()],
        "fireworks-ai" => vec!["fireworks".to_string(), "fireworksai".to_string()],
        "xiaomi" => vec!["mimo".to_string(), "xiaomimimo".to_string()],
        "siliconflow" => vec!["silicon".to_string(), "siliconcloud".to_string()],
        "302ai" => vec!["ai302".to_string()],
        "new-api" => vec!["newapi".to_string()],
        "vercel-gateway" => vec!["vercelaigateway".to_string()],
        "fal" => vec!["falai".to_string()],
        "yi" => vec!["zeroone".to_string()],
        "infini" => vec!["infiniai".to_string()],
        "doubao" => vec!["volcengine".to_string()],
        "airgate-openai-images" => vec!["airgate".to_string(), "k8ray".to_string()],
        "baidu-cloud" => vec!["wenxin".to_string()],
        "tencent-cloud-ti" => vec!["tencentcloud".to_string()],
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lime_core::database::schema;
    use rusqlite::{params, Connection};
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn setup_model_provider_db() -> DbConnection {
        let conn = Connection::open_in_memory().expect("open db");
        schema::create_tables(&conn).expect("create schema");
        Arc::new(Mutex::new(conn))
    }

    async fn spawn_models_fixture() -> (String, tokio::task::JoinHandle<String>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind models fixture");
        let address = listener.local_addr().expect("models fixture address");
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept models request");
            let mut request = [0_u8; 4096];
            let size = stream
                .read(&mut request)
                .await
                .expect("read models request");
            let body = r#"{"models":[{"name":"qwen3:14b","details":{"family":"qwen3"}}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .await
                .expect("write models response");
            String::from_utf8_lossy(&request[..size]).into_owned()
        });
        (format!("http://{address}"), server)
    }

    #[test]
    fn provider_model_config_to_core_preserves_explicit_capability() {
        let model: ProtocolProviderModelConfig = serde_json::from_value(serde_json::json!({
            "id": "grok-4.5",
            "displayName": "Grok 4.5",
            "capability": {
                "taskFamilies": ["chat", "reasoning"],
                "inputModalities": ["text", "image"],
                "outputModalities": ["text"],
                "runtimeFeatures": ["streaming", "tool_calling"],
                "capabilities": {
                    "vision": true,
                    "tools": true,
                    "streaming": true,
                    "jsonMode": true,
                    "functionCalling": true,
                    "reasoning": true,
                    "reasoningEffort": {
                        "supported": true,
                        "levels": ["medium", "xhigh"],
                        "options": [{
                            "id": "deep",
                            "value": "xhigh",
                            "label": "Deep",
                            "description": "Maximum reasoning",
                            "default": true
                        }],
                        "default": "xhigh",
                        "source": "api"
                    }
                }
            }
        }))
        .expect("deserialize typed provider model");

        let model = provider_model_config_to_core(model).expect("convert provider model");

        assert_eq!(model.id, "grok-4.5");
        assert_eq!(model.display_name.as_deref(), Some("Grok 4.5"));
        assert_eq!(
            serde_json::to_value(model.capability.expect("explicit capability"))
                .expect("serialize capability"),
            serde_json::json!({
                "task_families": ["chat", "reasoning"],
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
                "runtime_features": ["streaming", "tool_calling"],
                "capabilities": {
                    "vision": true,
                    "tools": true,
                    "streaming": true,
                    "json_mode": true,
                    "function_calling": true,
                    "reasoning": true,
                    "reasoning_effort": {
                        "supported": true,
                        "levels": ["medium", "xhigh"],
                        "options": [{
                            "id": "deep",
                            "value": "xhigh",
                            "label": "Deep",
                            "description": "Maximum reasoning",
                            "default": true
                        }],
                        "default": "xhigh",
                        "source": "api"
                    }
                }
            })
        );
    }

    #[test]
    fn provider_model_config_to_core_rejects_unknown_capability_taxonomy() {
        let model: ProtocolProviderModelConfig = serde_json::from_value(serde_json::json!({
            "id": "unknown-model",
            "capability": {
                "taskFamilies": ["unknown_task"],
                "inputModalities": [],
                "outputModalities": [],
                "runtimeFeatures": [],
                "capabilities": {
                    "vision": false,
                    "tools": false,
                    "streaming": false,
                    "jsonMode": false,
                    "functionCalling": false,
                    "reasoning": false
                }
            }
        }))
        .expect("deserialize protocol model");

        let error = provider_model_config_to_core(model).expect_err("reject unknown taxonomy");
        let message = error.to_string();
        assert!(message.contains("taskFamilies"));
        assert!(message.contains("unknown_task"));
    }

    fn insert_provider(db: &DbConnection, provider_id: &str, enabled: bool, models: &[&str]) {
        insert_typed_provider(db, provider_id, "openai", enabled, models);
    }

    fn insert_typed_provider(
        db: &DbConnection,
        provider_id: &str,
        provider_type: &str,
        enabled: bool,
        models: &[&str],
    ) {
        insert_typed_provider_with_host(
            db,
            provider_id,
            provider_type,
            "https://llm.limeai.run#lime_tenant_id=tenant-0001",
            enabled,
            models,
        );
    }

    fn insert_typed_provider_with_host(
        db: &DbConnection,
        provider_id: &str,
        provider_type: &str,
        api_host: &str,
        enabled: bool,
        models: &[&str],
    ) {
        let models = models
            .iter()
            .map(|model| CoreProviderModelConfig::hint(*model))
            .collect::<Vec<_>>();
        let models = serde_json::to_string(&models).expect("serialize models");
        let conn = db.lock().expect("lock db");
        conn.execute(
            "INSERT INTO api_key_providers (
                id, name, type, api_host, is_system, group_name, enabled, sort_order,
                models, created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, 0, 'cloud', ?5, 0, ?6, ?7, ?7)",
            params![
                provider_id,
                provider_id,
                provider_type,
                api_host,
                if enabled { 1 } else { 0 },
                models,
                "2026-07-06T00:00:00Z",
            ],
        )
        .expect("insert provider");
    }

    #[test]
    fn provider_capabilities_read_current_configured_provider_only() {
        let db = setup_model_provider_db();
        insert_typed_provider(&db, "bedrock-route", "aws-bedrock", true, &[]);
        insert_typed_provider_with_host(
            &db,
            "openai-route",
            "openai",
            "https://api.openai.com/v1",
            true,
            &[],
        );
        let service = ApiKeyProviderService::new();
        service
            .add_api_key(&db, "openai-route", "sk-test", None, true)
            .expect("add enabled OpenAI key");

        let response = resolve_model_provider_capabilities(&db, &service, "openai-route")
            .expect("read current provider capabilities");

        assert!(!response.namespace_tools);
        assert!(!response.image_generation);
        assert!(!response.web_search);

        insert_typed_provider_with_host(
            &db,
            "responses-route",
            "openai-response",
            "https://api.openai.com/v1",
            true,
            &[],
        );
        service
            .add_api_key(&db, "responses-route", "sk-test", None, true)
            .expect("add enabled Responses key");
        let response = resolve_model_provider_capabilities(&db, &service, "responses-route")
            .expect("read official Responses capabilities");
        assert!(!response.namespace_tools);
        assert!(response.image_generation);
        assert!(response.web_search);

        insert_typed_provider_with_host(
            &db,
            "custom-responses-route",
            "openai-response",
            "https://gateway.example.com/v1",
            true,
            &[],
        );
        service
            .add_api_key(&db, "custom-responses-route", "sk-test", None, true)
            .expect("add custom Responses key");
        let response = resolve_model_provider_capabilities(&db, &service, "custom-responses-route")
            .expect("read custom Responses capabilities");
        assert!(!response.image_generation);
        assert!(!response.web_search);

        let response = resolve_model_provider_capabilities(&db, &service, "bedrock-route")
            .expect("read unsupported current provider capabilities");

        assert!(!response.namespace_tools);
        assert!(!response.image_generation);
        assert!(!response.web_search);
    }

    #[test]
    fn provider_capabilities_fail_closed_without_a_ready_current_provider() {
        let db = setup_model_provider_db();
        insert_typed_provider(&db, "bedrock-route", "aws-bedrock", true, &[]);
        insert_typed_provider_with_host(
            &db,
            "missing-key-route",
            "openai",
            "https://api.openai.com/v1",
            true,
            &[],
        );
        insert_typed_provider(&db, "disabled-route", "openai", false, &[]);
        insert_typed_provider(&db, "media-route", "fal", true, &[]);
        let service = ApiKeyProviderService::new();

        for provider_id in [
            "",
            "missing-route",
            "missing-key-route",
            "disabled-route",
            "bedrock-route",
            "media-route",
        ] {
            let response = resolve_model_provider_capabilities(&db, &service, provider_id)
                .expect("read unavailable current provider capabilities");

            assert!(!response.namespace_tools, "provider_id={provider_id}");
            assert!(!response.image_generation, "provider_id={provider_id}");
            assert!(!response.web_search, "provider_id={provider_id}");
        }
    }

    #[tokio::test]
    async fn model_catalog_includes_enabled_provider_declared_models() {
        let db = setup_model_provider_db();
        insert_provider(&db, "lime-hub", true, &["gpt-5.1", "gpt-5.1"]);
        let api_key_provider_service = ApiKeyProviderService::new();
        let model_registry_service = ModelRegistryService::new(db.clone());

        let catalogs = model_catalog(
            &db,
            &api_key_provider_service,
            &model_registry_service,
            crate::ModelCatalogQuery {
                provider_id: Some("lime-hub".to_string()),
            },
        )
        .expect("list models");

        assert_eq!(catalogs.len(), 1);
        assert_eq!(catalogs[0].models.len(), 1);
        assert_eq!(catalogs[0].models[0].id, "gpt-5.1");
        assert_eq!(catalogs[0].provider_id, "lime-hub");
    }

    #[tokio::test]
    async fn model_catalog_orders_providers_by_sort_order_then_id() {
        let db = setup_model_provider_db();
        insert_typed_provider_with_host(
            &db,
            "provider-z",
            "openai",
            "https://llm.limeai.run#lime_tenant_id=tenant-0001",
            true,
            &["model-z"],
        );
        insert_typed_provider_with_host(
            &db,
            "provider-a",
            "openai",
            "https://llm.limeai.run#lime_tenant_id=tenant-0001",
            true,
            &["model-a"],
        );
        {
            let conn = db.lock().expect("lock db");
            conn.execute(
                "UPDATE api_key_providers SET sort_order = 20 WHERE id = 'provider-a'",
                [],
            )
            .expect("update provider sort order");
            conn.execute(
                "UPDATE api_key_providers SET sort_order = 10 WHERE id = 'provider-z'",
                [],
            )
            .expect("update provider sort order");
        }
        let api_key_provider_service = ApiKeyProviderService::new();
        let model_registry_service = ModelRegistryService::new(db.clone());

        let catalogs = model_catalog(
            &db,
            &api_key_provider_service,
            &model_registry_service,
            crate::ModelCatalogQuery::default(),
        )
        .expect("list ordered model catalog");

        assert_eq!(
            catalogs
                .iter()
                .map(|catalog| catalog.provider_id.as_str())
                .collect::<Vec<_>>(),
            vec!["provider-z", "provider-a"]
        );
    }

    #[tokio::test]
    async fn model_catalog_skips_disabled_provider_declared_models() {
        let db = setup_model_provider_db();
        insert_provider(&db, "disabled-hub", false, &["gpt-disabled"]);
        let api_key_provider_service = ApiKeyProviderService::new();
        let model_registry_service = ModelRegistryService::new(db.clone());

        let catalogs = model_catalog(
            &db,
            &api_key_provider_service,
            &model_registry_service,
            crate::ModelCatalogQuery::default(),
        )
        .expect("list models");

        assert!(catalogs.is_empty());
    }

    #[tokio::test]
    async fn model_catalog_skips_provider_without_required_key() {
        let db = setup_model_provider_db();
        insert_typed_provider_with_host(
            &db,
            "missing-key-route",
            "openai",
            "https://api.openai.com/v1",
            true,
            &["gpt-unready"],
        );
        let api_key_provider_service = ApiKeyProviderService::new();
        let model_registry_service = ModelRegistryService::new(db.clone());

        let catalogs = model_catalog(
            &db,
            &api_key_provider_service,
            &model_registry_service,
            crate::ModelCatalogQuery {
                provider_id: Some("missing-key-route".to_string()),
            },
        )
        .expect("list models without required key");

        assert!(catalogs.is_empty());
    }

    #[tokio::test]
    async fn model_catalog_includes_provider_with_enabled_key() {
        let db = setup_model_provider_db();
        insert_typed_provider_with_host(
            &db,
            "ready-openai-route",
            "openai",
            "https://api.openai.com/v1",
            true,
            &["gpt-ready"],
        );
        let api_key_provider_service = ApiKeyProviderService::new();
        api_key_provider_service
            .add_api_key(&db, "ready-openai-route", "sk-test", None, true)
            .expect("add enabled OpenAI key");
        let model_registry_service = ModelRegistryService::new(db.clone());

        let catalogs = model_catalog(
            &db,
            &api_key_provider_service,
            &model_registry_service,
            crate::ModelCatalogQuery {
                provider_id: Some("ready-openai-route".to_string()),
            },
        )
        .expect("list models with enabled key");

        assert_eq!(catalogs.len(), 1);
        assert_eq!(catalogs[0].models.len(), 1);
        assert_eq!(catalogs[0].models[0].id, "gpt-ready");
    }

    #[tokio::test]
    async fn model_catalog_includes_gemini_with_enabled_key() {
        let db = setup_model_provider_db();
        insert_typed_provider_with_host(
            &db,
            "ready-gemini-route",
            "gemini",
            "https://generativelanguage.googleapis.com",
            true,
            &["gemini-2.5-flash"],
        );
        let api_key_provider_service = ApiKeyProviderService::new();
        api_key_provider_service
            .add_api_key(&db, "ready-gemini-route", "gemini-test", None, true)
            .expect("add enabled Gemini key");
        let model_registry_service = ModelRegistryService::new(db.clone());

        let catalogs = model_catalog(
            &db,
            &api_key_provider_service,
            &model_registry_service,
            crate::ModelCatalogQuery {
                provider_id: Some("ready-gemini-route".to_string()),
            },
        )
        .expect("list Gemini models with enabled key");

        assert_eq!(catalogs.len(), 1);
        assert_eq!(catalogs[0].models.len(), 1);
        assert_eq!(catalogs[0].models[0].id, "gemini-2.5-flash");
    }

    #[tokio::test]
    async fn model_catalog_includes_keyless_ollama_responses_adapter() {
        let db = setup_model_provider_db();
        insert_typed_provider_with_host(
            &db,
            "local-ollama",
            "ollama",
            "http://127.0.0.1:11434",
            true,
            &["qwen3:14b"],
        );
        let api_key_provider_service = ApiKeyProviderService::new();
        let model_registry_service = ModelRegistryService::new(db.clone());

        let catalogs = model_catalog(
            &db,
            &api_key_provider_service,
            &model_registry_service,
            crate::ModelCatalogQuery {
                provider_id: Some("local-ollama".to_string()),
            },
        )
        .expect("list keyless Ollama models");

        assert_eq!(catalogs.len(), 1);
        assert_eq!(catalogs[0].provider_id, "local-ollama");
        assert_eq!(catalogs[0].models.len(), 1);
        assert_eq!(catalogs[0].models[0].id, "qwen3:14b");
    }

    #[tokio::test]
    async fn model_catalog_skips_non_chat_and_unknown_provider_types() {
        let db = setup_model_provider_db();
        insert_typed_provider(&db, "media-route", "fal", true, &["fal-image"]);
        insert_typed_provider(&db, "azure-route", "azure-openai", true, &["azure-model"]);
        insert_typed_provider(&db, "vertex-route", "vertexai", true, &["vertex-model"]);
        insert_typed_provider(
            &db,
            "bedrock-route",
            "aws-bedrock",
            true,
            &["bedrock-model"],
        );
        insert_typed_provider(
            &db,
            "unknown-route",
            "future-provider",
            true,
            &["future-model"],
        );
        let api_key_provider_service = ApiKeyProviderService::new();
        let model_registry_service = ModelRegistryService::new(db.clone());

        let catalogs = model_catalog(
            &db,
            &api_key_provider_service,
            &model_registry_service,
            crate::ModelCatalogQuery::default(),
        )
        .expect("list filtered provider models");

        assert!(catalogs.is_empty());
    }

    #[tokio::test]
    async fn keyless_fetch_models_uses_enabled_key_and_scopes_cache() {
        let db = setup_model_provider_db();
        let api_key_provider_service = ApiKeyProviderService::new();
        let model_registry_service = ModelRegistryService::new(db.clone());
        let (api_host, server) = spawn_models_fixture().await;
        let provider = api_key_provider_service
            .add_custom_provider(
                &db,
                "Keyless Models Fixture".to_string(),
                ApiProviderType::Ollama,
                api_host.clone(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create keyless provider");
        api_key_provider_service
            .add_api_key(&db, &provider.id, "scope-keyless-cache", None, false)
            .expect("add keyless provider key");

        let response = fetch_model_provider_models(
            &db,
            &api_key_provider_service,
            &model_registry_service,
            ModelProviderFetchModelsParams {
                provider_id: provider.id.clone(),
            },
        )
        .await
        .expect("fetch keyless models");
        let request = server.await.expect("join models fixture");

        assert_eq!(response.models.len(), 1);
        assert_eq!(response.models[0].id, "qwen3:14b");
        assert!(request.contains("scope-keyless-cache"));
        assert!(model_registry_service
            .get_cached_provider_models(&provider.id, &api_host, Some(ApiProviderType::Ollama))
            .expect("read unscoped keyless cache")
            .is_none());
        let cached = model_registry_service
            .fetch_models_from_api_with_hints(
                &provider.id,
                &api_host,
                "scope-keyless-cache",
                Some(ApiProviderType::Ollama),
                &[],
            )
            .await
            .expect("read scoped keyless cache");
        assert!(cached.from_cache);
        assert_eq!(cached.models.len(), 1);
        assert_eq!(cached.models[0].id, "qwen3:14b");

        let catalogs = model_catalog(
            &db,
            &api_key_provider_service,
            &model_registry_service,
            crate::ModelCatalogQuery {
                provider_id: Some(provider.id.clone()),
            },
        )
        .expect("credential-scoped discovery should enter model catalog");
        assert_eq!(catalogs.len(), 1);
        assert_eq!(catalogs[0].models.len(), 1);
        assert_eq!(catalogs[0].models[0].id, "qwen3:14b");
        assert_eq!(
            catalogs[0].models[0].capability_provenance,
            lime_core::models::model_registry::ModelCapabilityProvenance::InferredHint
        );
    }

    #[tokio::test]
    async fn keyless_fetch_models_without_enabled_key_writes_unscoped_cache() {
        let db = setup_model_provider_db();
        let api_key_provider_service = ApiKeyProviderService::new();
        let model_registry_service = ModelRegistryService::new(db.clone());
        let (api_host, server) = spawn_models_fixture().await;
        let provider = api_key_provider_service
            .add_custom_provider(
                &db,
                "Unscoped Keyless Models Fixture".to_string(),
                ApiProviderType::Ollama,
                api_host.clone(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create keyless provider without key");

        let response = fetch_model_provider_models(
            &db,
            &api_key_provider_service,
            &model_registry_service,
            ModelProviderFetchModelsParams {
                provider_id: provider.id.clone(),
            },
        )
        .await
        .expect("fetch keyless models without key");
        let request = server.await.expect("join models fixture");

        assert_eq!(response.models.len(), 1);
        assert_eq!(response.models[0].id, "qwen3:14b");
        assert!(!request.to_ascii_lowercase().contains("authorization:"));
        let cached = model_registry_service
            .get_cached_provider_models(&provider.id, &api_host, Some(ApiProviderType::Ollama))
            .expect("read unscoped keyless cache")
            .expect("unscoped keyless cache");
        assert_eq!(cached.models.len(), 1);
        assert_eq!(cached.models[0].id, "qwen3:14b");
    }
}
