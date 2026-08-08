use super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::*;

impl RuntimeCore {
    pub async fn read_project_memory(
        &self,
        params: ProjectMemoryReadParams,
    ) -> Result<ProjectMemoryReadResponse, RuntimeCoreError> {
        self.app_data_source.read_project_memory(params).await
    }

    pub async fn list_memory_store(
        &self,
        params: MemoryStoreListParams,
    ) -> Result<MemoryStoreListResponse, RuntimeCoreError> {
        self.app_data_source.list_memory_store(params).await
    }

    pub async fn read_memory_store(
        &self,
        params: MemoryStoreReadParams,
    ) -> Result<MemoryStoreReadResponse, RuntimeCoreError> {
        self.app_data_source.read_memory_store(params).await
    }

    pub async fn search_memory_store(
        &self,
        params: MemoryStoreSearchParams,
    ) -> Result<MemoryStoreSearchResponse, RuntimeCoreError> {
        self.app_data_source.search_memory_store(params).await
    }

    pub async fn add_memory_store_note(
        &self,
        params: MemoryStoreAddNoteParams,
    ) -> Result<MemoryStoreAddNoteResponse, RuntimeCoreError> {
        self.app_data_source.add_memory_store_note(params).await
    }

    pub async fn consolidate_memory_store(
        &self,
        params: MemoryStoreConsolidateParams,
    ) -> Result<MemoryStoreConsolidateResponse, RuntimeCoreError> {
        self.app_data_source.consolidate_memory_store(params).await
    }

    pub async fn list_memory_store_review_notes(
        &self,
        params: MemoryStoreReviewListParams,
    ) -> Result<MemoryStoreReviewListResponse, RuntimeCoreError> {
        self.app_data_source
            .list_memory_store_review_notes(params)
            .await
    }

    pub async fn resolve_memory_store_review_note(
        &self,
        params: MemoryStoreReviewResolveParams,
    ) -> Result<MemoryStoreReviewResolveResponse, RuntimeCoreError> {
        self.app_data_source
            .resolve_memory_store_review_note(params)
            .await
    }

    pub async fn health_memory_store(
        &self,
        params: MemoryStoreRootParams,
    ) -> Result<MemoryStoreHealthResponse, RuntimeCoreError> {
        self.app_data_source.health_memory_store(params).await
    }

    pub async fn reset_memory(&self) -> Result<(), RuntimeCoreError> {
        self.app_data_source.reset_memory().await
    }

    pub async fn rebuild_memory_store_index(
        &self,
        params: MemoryStoreRootParams,
    ) -> Result<MemoryStoreIndexRebuildResponse, RuntimeCoreError> {
        self.app_data_source
            .rebuild_memory_store_index(params)
            .await
    }
}
