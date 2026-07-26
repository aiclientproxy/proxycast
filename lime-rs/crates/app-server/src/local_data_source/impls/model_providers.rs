use super::super::*;
use crate::{ModelCatalogQuery, ProviderModelCatalog};
use app_server_protocol::protocol::v2::ModelProviderCapabilitiesReadResponse;
use async_trait::async_trait;

#[async_trait]
impl ModelProviderAppDataSource for LocalAppDataSource {
    async fn read_model_route_generation(&self) -> Result<u64, RuntimeCoreError> {
        model_providers::read_model_route_generation(&self.db)
    }

    async fn model_catalog(
        &self,
        query: ModelCatalogQuery,
    ) -> Result<Vec<ProviderModelCatalog>, RuntimeCoreError> {
        model_providers::model_catalog(
            &self.db,
            &self.api_key_provider_service,
            &self.model_registry_service,
            query,
        )
    }

    async fn list_model_preferences(
        &self,
    ) -> Result<ModelPreferencesListResponse, RuntimeCoreError> {
        model_providers::list_model_preferences(&self.model_registry_service).await
    }

    async fn read_model_sync_state(&self) -> Result<ModelSyncStateReadResponse, RuntimeCoreError> {
        model_providers::read_model_sync_state(&self.model_registry_service).await
    }

    async fn list_model_providers(&self) -> Result<ModelProviderListResponse, RuntimeCoreError> {
        model_providers::list_model_providers(&self.db, &self.api_key_provider_service)
    }

    async fn read_model_provider_capabilities(
        &self,
    ) -> Result<ModelProviderCapabilitiesReadResponse, RuntimeCoreError> {
        model_providers::read_model_provider_capabilities(&self.db, &self.api_key_provider_service)
    }

    async fn list_model_provider_catalog(
        &self,
    ) -> Result<ModelProviderCatalogListResponse, RuntimeCoreError> {
        model_providers::list_model_provider_catalog()
    }

    async fn read_model_provider(
        &self,
        params: ModelProviderReadParams,
    ) -> Result<ModelProviderReadResponse, RuntimeCoreError> {
        model_providers::read_model_provider(&self.db, &self.api_key_provider_service, params)
    }

    async fn create_model_provider(
        &self,
        params: ModelProviderCreateParams,
    ) -> Result<ModelProviderWriteResponse, RuntimeCoreError> {
        model_providers::create_model_provider(&self.db, &self.api_key_provider_service, params)
    }

    async fn update_model_provider(
        &self,
        params: ModelProviderUpdateParams,
    ) -> Result<ModelProviderWriteResponse, RuntimeCoreError> {
        model_providers::update_model_provider(&self.db, &self.api_key_provider_service, params)
    }

    async fn delete_model_provider(
        &self,
        params: ModelProviderDeleteParams,
    ) -> Result<ModelProviderDeleteResponse, RuntimeCoreError> {
        model_providers::delete_model_provider(&self.db, &self.api_key_provider_service, params)
    }

    async fn update_model_provider_sort_orders(
        &self,
        params: ModelProviderSortOrdersUpdateParams,
    ) -> Result<ModelProviderMutationResponse, RuntimeCoreError> {
        model_providers::update_model_provider_sort_orders(
            &self.db,
            &self.api_key_provider_service,
            params,
        )
    }

    async fn export_model_provider_config(
        &self,
        params: ModelProviderConfigExportParams,
    ) -> Result<ModelProviderConfigExportResponse, RuntimeCoreError> {
        model_providers::export_model_provider_config(
            &self.db,
            &self.api_key_provider_service,
            params,
        )
    }

    async fn import_model_provider_config(
        &self,
        params: ModelProviderConfigImportParams,
    ) -> Result<ModelProviderConfigImportResponse, RuntimeCoreError> {
        model_providers::import_model_provider_config(
            &self.db,
            &self.api_key_provider_service,
            params,
        )
    }

    async fn test_model_provider_connection(
        &self,
        params: ModelProviderTestConnectionParams,
    ) -> Result<ModelProviderTestConnectionResponse, RuntimeCoreError> {
        model_providers::test_model_provider_connection(
            &self.db,
            &self.api_key_provider_service,
            params,
        )
        .await
    }

    async fn test_model_provider_chat(
        &self,
        params: ModelProviderTestChatParams,
    ) -> Result<ModelProviderTestChatResponse, RuntimeCoreError> {
        model_providers::test_model_provider_chat(&self.db, &self.api_key_provider_service, params)
            .await
    }

    async fn fetch_model_provider_models(
        &self,
        params: ModelProviderFetchModelsParams,
    ) -> Result<ModelProviderFetchModelsResponse, RuntimeCoreError> {
        model_providers::fetch_model_provider_models(
            &self.db,
            &self.api_key_provider_service,
            &self.model_registry_service,
            params,
        )
        .await
    }

    async fn create_model_provider_key(
        &self,
        params: ModelProviderKeyCreateParams,
    ) -> Result<ModelProviderKeyWriteResponse, RuntimeCoreError> {
        model_providers::create_model_provider_key(&self.db, &self.api_key_provider_service, params)
    }

    async fn update_model_provider_key(
        &self,
        params: ModelProviderKeyUpdateParams,
    ) -> Result<ModelProviderKeyWriteResponse, RuntimeCoreError> {
        model_providers::update_model_provider_key(&self.db, &self.api_key_provider_service, params)
    }

    async fn delete_model_provider_key(
        &self,
        params: ModelProviderKeyDeleteParams,
    ) -> Result<ModelProviderKeyDeleteResponse, RuntimeCoreError> {
        model_providers::delete_model_provider_key(&self.db, &self.api_key_provider_service, params)
    }

    async fn read_model_provider_ui_state(
        &self,
        params: ModelProviderUiStateReadParams,
    ) -> Result<ModelProviderUiStateReadResponse, RuntimeCoreError> {
        model_providers::read_model_provider_ui_state(
            &self.db,
            &self.api_key_provider_service,
            params,
        )
    }

    async fn write_model_provider_ui_state(
        &self,
        params: ModelProviderUiStateWriteParams,
    ) -> Result<ModelProviderMutationResponse, RuntimeCoreError> {
        model_providers::write_model_provider_ui_state(
            &self.db,
            &self.api_key_provider_service,
            params,
        )
    }

    async fn read_model_provider_alias(
        &self,
        params: ModelProviderAliasReadParams,
    ) -> Result<ModelProviderAliasReadResponse, RuntimeCoreError> {
        model_providers::read_model_provider_alias(&self.model_registry_service, params).await
    }

    async fn list_model_provider_aliases(
        &self,
    ) -> Result<ModelProviderAliasListResponse, RuntimeCoreError> {
        model_providers::list_model_provider_aliases(&self.model_registry_service).await
    }
}
