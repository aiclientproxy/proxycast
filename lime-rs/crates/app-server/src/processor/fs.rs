use super::{dispatch_result, parse_params, ConnectionRequestId, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{
    FsCopyParams, FsCreateDirectoryParams, FsGetMetadataParams, FsReadDirectoryParams,
    FsReadFileParams, FsRemoveParams, FsUnwatchParams, FsWatchParams, FsWriteFileParams,
};
use app_server_protocol::{error_codes, JsonRpcError};
use serde_json::Value;

fn connection_id(
    request: Option<ConnectionRequestId>,
) -> Result<app_server_transport::ConnectionId, JsonRpcError> {
    request.map(|request| request.connection_id).ok_or_else(|| {
        JsonRpcError::new(
            error_codes::INVALID_REQUEST,
            "fs watch requires transport connection",
        )
    })
}

impl RequestProcessor {
    pub(super) async fn handle_fs_read_file_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: FsReadFileParams = parse_params(params)?;
        dispatch_result(self.fs.read_file(params).await?)
    }

    pub(super) async fn handle_fs_write_file_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: FsWriteFileParams = parse_params(params)?;
        dispatch_result(self.fs.write_file(params).await?)
    }

    pub(super) async fn handle_fs_create_directory_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: FsCreateDirectoryParams = parse_params(params)?;
        dispatch_result(self.fs.create_directory(params).await?)
    }

    pub(super) async fn handle_fs_get_metadata_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: FsGetMetadataParams = parse_params(params)?;
        dispatch_result(self.fs.get_metadata(params).await?)
    }

    pub(super) async fn handle_fs_read_directory_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: FsReadDirectoryParams = parse_params(params)?;
        dispatch_result(self.fs.read_directory(params).await?)
    }

    pub(super) async fn handle_fs_remove_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: FsRemoveParams = parse_params(params)?;
        dispatch_result(self.fs.remove(params).await?)
    }

    pub(super) async fn handle_fs_copy_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: FsCopyParams = parse_params(params)?;
        dispatch_result(self.fs.copy(params).await?)
    }

    pub(super) async fn handle_fs_watch_impl(
        &self,
        params: Option<Value>,
        request: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: FsWatchParams = parse_params(params)?;
        dispatch_result(self.fs.watch(connection_id(request)?, params).await?)
    }

    pub(super) async fn handle_fs_unwatch_impl(
        &self,
        params: Option<Value>,
        request: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: FsUnwatchParams = parse_params(params)?;
        dispatch_result(self.fs.unwatch(connection_id(request)?, params).await?)
    }
}
