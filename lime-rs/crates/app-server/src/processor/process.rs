use super::{dispatch_result, parse_params, ConnectionRequestId, RequestProcessor, RpcDispatch};
use app_server_protocol::error_codes;
use app_server_protocol::protocol::v2::{
    ProcessKillParams, ProcessResizePtyParams, ProcessSpawnParams, ProcessWriteStdinParams,
};
use app_server_protocol::JsonRpcError;
use serde_json::Value;

impl RequestProcessor {
    pub(super) async fn handle_process_spawn_impl(
        &self,
        params: Option<Value>,
        request: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let connection_id = request
            .map(|request| request.connection_id)
            .ok_or_else(|| {
                JsonRpcError::new(
                    error_codes::INVALID_REQUEST,
                    "process requires transport connection",
                )
            })?;
        let params: ProcessSpawnParams = parse_params(params)?;
        dispatch_result(self.process.spawn(connection_id, params).await?)
    }

    pub(super) async fn handle_process_write_stdin_impl(
        &self,
        params: Option<Value>,
        request: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let connection_id = request
            .map(|request| request.connection_id)
            .ok_or_else(|| {
                JsonRpcError::new(
                    error_codes::INVALID_REQUEST,
                    "process requires transport connection",
                )
            })?;
        let params: ProcessWriteStdinParams = parse_params(params)?;
        dispatch_result(self.process.write_stdin(connection_id, params).await?)
    }

    pub(super) async fn handle_process_resize_pty_impl(
        &self,
        params: Option<Value>,
        request: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let connection_id = request
            .map(|request| request.connection_id)
            .ok_or_else(|| {
                JsonRpcError::new(
                    error_codes::INVALID_REQUEST,
                    "process requires transport connection",
                )
            })?;
        let params: ProcessResizePtyParams = parse_params(params)?;
        dispatch_result(self.process.resize_pty(connection_id, params).await?)
    }

    pub(super) async fn handle_process_kill_impl(
        &self,
        params: Option<Value>,
        request: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let connection_id = request
            .map(|request| request.connection_id)
            .ok_or_else(|| {
                JsonRpcError::new(
                    error_codes::INVALID_REQUEST,
                    "process requires transport connection",
                )
            })?;
        let params: ProcessKillParams = parse_params(params)?;
        dispatch_result(self.process.kill(connection_id, params).await?)
    }
}
