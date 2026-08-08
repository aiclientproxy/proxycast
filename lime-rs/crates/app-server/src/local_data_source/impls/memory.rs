use super::super::*;
use async_trait::async_trait;

#[async_trait]
impl MemoryAppDataSource for LocalAppDataSource {
    async fn read_project_memory(
        &self,
        params: ProjectMemoryReadParams,
    ) -> Result<ProjectMemoryReadResponse, RuntimeCoreError> {
        let memory = lime_core::memory::read_project_memory(self.db.clone(), &params.project_id)
            .map_err(data_error)?;
        Ok(ProjectMemoryReadResponse {
            memory: serde_json::to_value(memory).map_err(data_error)?,
        })
    }

    async fn list_memory_store(
        &self,
        params: MemoryStoreListParams,
    ) -> Result<MemoryStoreListResponse, RuntimeCoreError> {
        self.memory_backend.list(params).await
    }

    async fn read_memory_store(
        &self,
        params: MemoryStoreReadParams,
    ) -> Result<MemoryStoreReadResponse, RuntimeCoreError> {
        self.memory_backend.read(params).await
    }

    async fn search_memory_store(
        &self,
        params: MemoryStoreSearchParams,
    ) -> Result<MemoryStoreSearchResponse, RuntimeCoreError> {
        self.memory_backend.search(params).await
    }

    async fn add_memory_store_note(
        &self,
        params: MemoryStoreAddNoteParams,
    ) -> Result<MemoryStoreAddNoteResponse, RuntimeCoreError> {
        self.memory_backend.add_note(params).await
    }

    async fn write_memory_rollout_summary(
        &self,
        params: RolloutSummaryWriteParams,
    ) -> Result<MemoryStoreAddNoteResponse, RuntimeCoreError> {
        self.memory_backend.write_rollout_summary(params).await
    }

    async fn consolidate_memory_store(
        &self,
        params: MemoryStoreConsolidateParams,
    ) -> Result<MemoryStoreConsolidateResponse, RuntimeCoreError> {
        self.memory_backend.consolidate(params).await
    }

    async fn list_memory_store_review_notes(
        &self,
        params: MemoryStoreReviewListParams,
    ) -> Result<MemoryStoreReviewListResponse, RuntimeCoreError> {
        self.memory_backend.list_review(params).await
    }

    async fn resolve_memory_store_review_note(
        &self,
        params: MemoryStoreReviewResolveParams,
    ) -> Result<MemoryStoreReviewResolveResponse, RuntimeCoreError> {
        self.memory_backend.resolve_review(params).await
    }

    async fn health_memory_store(
        &self,
        params: MemoryStoreRootParams,
    ) -> Result<MemoryStoreHealthResponse, RuntimeCoreError> {
        self.memory_backend.health(params).await
    }

    async fn reset_memory(&self) -> Result<(), RuntimeCoreError> {
        self.memory_backend.reset().await
    }

    async fn rebuild_memory_store_index(
        &self,
        params: MemoryStoreRootParams,
    ) -> Result<MemoryStoreIndexRebuildResponse, RuntimeCoreError> {
        self.memory_backend.rebuild_index(params).await
    }
}
