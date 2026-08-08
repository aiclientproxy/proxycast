use super::unavailable;
use super::NoopAppDataSource;
use super::RuntimeCoreError;
use crate::RolloutSummaryWriteParams;
use app_server_protocol::*;
use async_trait::async_trait;

#[async_trait]
pub trait MemoryAppDataSource: Send + Sync {
    async fn read_project_memory(
        &self,
        _params: ProjectMemoryReadParams,
    ) -> Result<ProjectMemoryReadResponse, RuntimeCoreError> {
        Ok(ProjectMemoryReadResponse::default())
    }

    async fn list_memory_store(
        &self,
        _params: MemoryStoreListParams,
    ) -> Result<MemoryStoreListResponse, RuntimeCoreError> {
        Err(unavailable("memoryStore/list"))
    }

    async fn read_memory_store(
        &self,
        _params: MemoryStoreReadParams,
    ) -> Result<MemoryStoreReadResponse, RuntimeCoreError> {
        Err(unavailable("memoryStore/read"))
    }

    async fn search_memory_store(
        &self,
        _params: MemoryStoreSearchParams,
    ) -> Result<MemoryStoreSearchResponse, RuntimeCoreError> {
        Err(unavailable("memoryStore/search"))
    }

    async fn add_memory_store_note(
        &self,
        _params: MemoryStoreAddNoteParams,
    ) -> Result<MemoryStoreAddNoteResponse, RuntimeCoreError> {
        Err(unavailable("memoryStore/addNote"))
    }

    async fn write_memory_rollout_summary(
        &self,
        _params: RolloutSummaryWriteParams,
    ) -> Result<MemoryStoreAddNoteResponse, RuntimeCoreError> {
        Err(unavailable("memory rollout summary write"))
    }

    async fn consolidate_memory_store(
        &self,
        _params: MemoryStoreConsolidateParams,
    ) -> Result<MemoryStoreConsolidateResponse, RuntimeCoreError> {
        Err(unavailable("memoryStore/consolidate"))
    }

    async fn list_memory_store_review_notes(
        &self,
        _params: MemoryStoreReviewListParams,
    ) -> Result<MemoryStoreReviewListResponse, RuntimeCoreError> {
        Err(unavailable("memoryStore/review/list"))
    }

    async fn resolve_memory_store_review_note(
        &self,
        _params: MemoryStoreReviewResolveParams,
    ) -> Result<MemoryStoreReviewResolveResponse, RuntimeCoreError> {
        Err(unavailable("memoryStore/review/resolve"))
    }

    async fn health_memory_store(
        &self,
        _params: MemoryStoreRootParams,
    ) -> Result<MemoryStoreHealthResponse, RuntimeCoreError> {
        Err(unavailable("memoryStore/health"))
    }

    async fn reset_memory(&self) -> Result<(), RuntimeCoreError> {
        Err(unavailable("memory/reset"))
    }

    async fn rebuild_memory_store_index(
        &self,
        _params: MemoryStoreRootParams,
    ) -> Result<MemoryStoreIndexRebuildResponse, RuntimeCoreError> {
        Err(unavailable("memoryStore/index/rebuild"))
    }
}

impl MemoryAppDataSource for NoopAppDataSource {}
