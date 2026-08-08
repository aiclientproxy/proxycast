//! memory store domain handlers for the App Server processor.

use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::{
    JsonRpcError, MemoryResetResponse, MemoryStoreAddNoteParams, MemoryStoreConsolidateParams,
    MemoryStoreListParams, MemoryStoreReadParams, MemoryStoreReviewListParams,
    MemoryStoreReviewResolveParams, MemoryStoreRootParams, MemoryStoreSearchParams,
};
use serde_json::Value;

impl RequestProcessor {
    pub(super) async fn handle_memory_store_list_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MemoryStoreListParams = parse_params(params)?;
        let response = self
            .runtime
            .list_memory_store(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_memory_store_read_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MemoryStoreReadParams = parse_params(params)?;
        let response = self
            .runtime
            .read_memory_store(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_memory_store_search_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MemoryStoreSearchParams = parse_params(params)?;
        let response = self
            .runtime
            .search_memory_store(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_memory_store_add_note_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MemoryStoreAddNoteParams = parse_params(params)?;
        let response = self
            .runtime
            .add_memory_store_note(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_memory_store_consolidate_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MemoryStoreConsolidateParams = parse_params(params)?;
        let response = self
            .runtime
            .consolidate_memory_store(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_memory_store_review_list_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MemoryStoreReviewListParams = parse_params(params)?;
        let response = self
            .runtime
            .list_memory_store_review_notes(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_memory_store_review_resolve_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MemoryStoreReviewResolveParams = parse_params(params)?;
        let response = self
            .runtime
            .resolve_memory_store_review_note(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_memory_store_health_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MemoryStoreRootParams = parse_params(params)?;
        let response = self
            .runtime
            .health_memory_store(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_memory_reset_impl(&self) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        self.runtime
            .reset_memory()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(MemoryResetResponse {})
    }

    pub(super) async fn handle_memory_store_index_rebuild_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MemoryStoreRootParams = parse_params(params)?;
        let response = self
            .runtime
            .rebuild_memory_store_index(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }
}
