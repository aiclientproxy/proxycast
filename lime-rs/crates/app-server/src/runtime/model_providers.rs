use super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::*;
use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

mod selection;

pub(in crate::runtime) use selection::{
    catalog_model_default_service_tier, is_executable_chat_model,
};

#[derive(Clone, Default)]
pub(in crate::runtime) struct RouteRecoveryCoordinator {
    state: Arc<RouteRecoveryState>,
}

#[derive(Default)]
struct RouteRecoveryState {
    attempted_generation: AtomicU64,
    gate: tokio::sync::Mutex<()>,
}

#[derive(Clone, Default)]
pub(in crate::runtime) struct CatalogRefreshCoordinator {
    entries: Arc<tokio::sync::Mutex<HashMap<String, Arc<CatalogRefreshEntry>>>>,
}

#[derive(Default)]
struct CatalogRefreshEntry {
    gate: tokio::sync::Mutex<()>,
    retry_in_flight: AtomicBool,
}

pub(crate) struct CatalogRetryPermit {
    entry: Arc<CatalogRefreshEntry>,
}

impl Drop for CatalogRetryPermit {
    fn drop(&mut self) {
        self.entry.retry_in_flight.store(false, Ordering::Release);
    }
}

impl CatalogRefreshCoordinator {
    async fn entry(&self, provider_id: &str) -> Arc<CatalogRefreshEntry> {
        let mut entries = self.entries.lock().await;
        entries
            .entry(provider_id.trim().to_ascii_lowercase())
            .or_insert_with(|| Arc::new(CatalogRefreshEntry::default()))
            .clone()
    }

    async fn run<F, Fut, T>(&self, provider_id: &str, work: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        let entry = self.entry(provider_id).await;
        let _guard = entry.gate.lock().await;
        work().await
    }

    async fn acquire_retry(&self, provider_id: &str) -> Option<CatalogRetryPermit> {
        let entry = self.entry(provider_id).await;
        entry
            .retry_in_flight
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| CatalogRetryPermit { entry })
    }
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
    pub(crate) async fn model_catalog_generation(&self) -> Result<u64, RuntimeCoreError> {
        self.app_data_source.read_model_route_generation().await
    }

    pub(crate) async fn has_model_provider_last_success(
        &self,
        provider_id: &str,
    ) -> Result<bool, RuntimeCoreError> {
        self.app_data_source
            .has_model_provider_last_success(provider_id)
            .await
    }

    pub(crate) async fn acquire_model_provider_catalog_retry(
        &self,
        provider_id: &str,
    ) -> Option<CatalogRetryPermit> {
        self.catalog_refresh.acquire_retry(provider_id).await
    }

    pub(crate) async fn refresh_model_provider_catalog(
        &self,
        provider_id: &str,
    ) -> Result<ModelProviderFetchModelsResponse, RuntimeCoreError> {
        let provider_id = provider_id.trim().to_string();
        let refresh_provider_id = provider_id.clone();
        let coordinator = self.catalog_refresh.clone();
        coordinator
            .run(&provider_id, || async move {
                self.fetch_model_provider_models(ModelProviderFetchModelsParams {
                    provider_id: refresh_provider_id,
                })
                .await
            })
            .await
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

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

    #[tokio::test]
    async fn catalog_retry_permit_is_single_flight_per_provider() {
        let coordinator = CatalogRefreshCoordinator::default();
        let first = coordinator
            .acquire_retry("provider-a")
            .await
            .expect("first retry owns provider flight");
        assert!(coordinator.acquire_retry("provider-a").await.is_none());
        assert!(coordinator.acquire_retry("provider-b").await.is_some());

        drop(first);
        assert!(coordinator.acquire_retry("provider-a").await.is_some());
    }
}
