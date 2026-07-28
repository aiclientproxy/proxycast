use super::unavailable;
use super::NoopAppDataSource;
use super::RuntimeCoreError;
use app_server_protocol::*;
use async_trait::async_trait;
use lime_core::models::model_registry::EnhancedModelMetadata;
use serde_json::json;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelCatalogQuery {
    pub provider_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ProviderModelCatalog {
    pub provider_id: String,
    pub sort_order: i32,
    pub models: Vec<EnhancedModelMetadata>,
}

#[async_trait]
pub trait ModelProviderAppDataSource: Send + Sync {
    fn model_catalog_reconciliation_enabled(&self) -> bool {
        false
    }

    async fn read_model_route_generation(&self) -> Result<u64, RuntimeCoreError> {
        Err(unavailable("modelProvider/routeGeneration/read"))
    }

    async fn model_catalog(
        &self,
        _query: ModelCatalogQuery,
    ) -> Result<Vec<ProviderModelCatalog>, RuntimeCoreError> {
        Ok(Vec::new())
    }

    async fn has_model_provider_last_success(
        &self,
        _provider_id: &str,
    ) -> Result<bool, RuntimeCoreError> {
        Ok(false)
    }

    async fn list_model_preferences(
        &self,
    ) -> Result<ModelPreferencesListResponse, RuntimeCoreError> {
        Ok(ModelPreferencesListResponse::default())
    }

    async fn read_model_sync_state(&self) -> Result<ModelSyncStateReadResponse, RuntimeCoreError> {
        Ok(ModelSyncStateReadResponse {
            sync_state: json!({
                "last_sync_at": null,
                "model_count": 0,
                "is_syncing": false,
                "last_error": null,
            }),
        })
    }

    async fn list_model_providers(&self) -> Result<ModelProviderListResponse, RuntimeCoreError> {
        Ok(ModelProviderListResponse::default())
    }

    async fn list_model_provider_catalog(
        &self,
    ) -> Result<ModelProviderCatalogListResponse, RuntimeCoreError> {
        Ok(ModelProviderCatalogListResponse::default())
    }

    async fn read_model_provider(
        &self,
        _params: ModelProviderReadParams,
    ) -> Result<ModelProviderReadResponse, RuntimeCoreError> {
        Err(unavailable("modelProvider/read"))
    }

    async fn create_model_provider(
        &self,
        _params: ModelProviderCreateParams,
    ) -> Result<ModelProviderWriteResponse, RuntimeCoreError> {
        Err(unavailable("modelProvider/create"))
    }

    async fn update_model_provider(
        &self,
        _params: ModelProviderUpdateParams,
    ) -> Result<ModelProviderWriteResponse, RuntimeCoreError> {
        Err(unavailable("modelProvider/update"))
    }

    async fn delete_model_provider(
        &self,
        _params: ModelProviderDeleteParams,
    ) -> Result<ModelProviderDeleteResponse, RuntimeCoreError> {
        Err(unavailable("modelProvider/delete"))
    }

    async fn update_model_provider_sort_orders(
        &self,
        _params: ModelProviderSortOrdersUpdateParams,
    ) -> Result<ModelProviderMutationResponse, RuntimeCoreError> {
        Err(unavailable("modelProvider/sortOrders/update"))
    }

    async fn export_model_provider_config(
        &self,
        _params: ModelProviderConfigExportParams,
    ) -> Result<ModelProviderConfigExportResponse, RuntimeCoreError> {
        Err(unavailable("modelProviderConfig/export"))
    }

    async fn import_model_provider_config(
        &self,
        _params: ModelProviderConfigImportParams,
    ) -> Result<ModelProviderConfigImportResponse, RuntimeCoreError> {
        Err(unavailable("modelProviderConfig/import"))
    }

    async fn test_model_provider_connection(
        &self,
        _params: ModelProviderTestConnectionParams,
    ) -> Result<ModelProviderTestConnectionResponse, RuntimeCoreError> {
        Err(unavailable("modelProvider/testConnection"))
    }

    async fn test_model_provider_chat(
        &self,
        _params: ModelProviderTestChatParams,
    ) -> Result<ModelProviderTestChatResponse, RuntimeCoreError> {
        Err(unavailable("modelProvider/testChat"))
    }

    async fn fetch_model_provider_models(
        &self,
        _params: ModelProviderFetchModelsParams,
    ) -> Result<ModelProviderFetchModelsResponse, RuntimeCoreError> {
        Err(unavailable("modelProvider/fetchModels"))
    }

    async fn create_model_provider_key(
        &self,
        _params: ModelProviderKeyCreateParams,
    ) -> Result<ModelProviderKeyWriteResponse, RuntimeCoreError> {
        Err(unavailable("modelProviderKey/create"))
    }

    async fn update_model_provider_key(
        &self,
        _params: ModelProviderKeyUpdateParams,
    ) -> Result<ModelProviderKeyWriteResponse, RuntimeCoreError> {
        Err(unavailable("modelProviderKey/update"))
    }

    async fn delete_model_provider_key(
        &self,
        _params: ModelProviderKeyDeleteParams,
    ) -> Result<ModelProviderKeyDeleteResponse, RuntimeCoreError> {
        Err(unavailable("modelProviderKey/delete"))
    }

    async fn read_model_provider_ui_state(
        &self,
        _params: ModelProviderUiStateReadParams,
    ) -> Result<ModelProviderUiStateReadResponse, RuntimeCoreError> {
        Err(unavailable("modelProviderUiState/read"))
    }

    async fn write_model_provider_ui_state(
        &self,
        _params: ModelProviderUiStateWriteParams,
    ) -> Result<ModelProviderMutationResponse, RuntimeCoreError> {
        Err(unavailable("modelProviderUiState/write"))
    }

    async fn read_model_provider_alias(
        &self,
        _params: ModelProviderAliasReadParams,
    ) -> Result<ModelProviderAliasReadResponse, RuntimeCoreError> {
        Ok(ModelProviderAliasReadResponse::default())
    }

    async fn list_model_provider_aliases(
        &self,
    ) -> Result<ModelProviderAliasListResponse, RuntimeCoreError> {
        Ok(ModelProviderAliasListResponse::default())
    }
}

impl ModelProviderAppDataSource for NoopAppDataSource {}
