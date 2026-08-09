use super::{dispatch_result, parse_params, ConnectionRequestId, RequestProcessor, RpcDispatch};
use app_server_protocol::{error_codes, JsonRpcError};
use serde_json::Value;

impl RequestProcessor {
    pub(super) async fn handle_command_exec_impl(
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
                    "command/exec requires transport connection",
                )
            })?;
        dispatch_result(
            self.command_exec
                .exec(connection_id, parse_params(params)?)
                .await?,
        )
    }

    pub(super) async fn handle_command_exec_write_impl(
        &self,
        params: Option<Value>,
        request: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let connection_id = connection_id(request)?;
        dispatch_result(
            self.command_exec
                .write(connection_id, parse_params(params)?)
                .await?,
        )
    }

    pub(super) async fn handle_command_exec_resize_impl(
        &self,
        params: Option<Value>,
        request: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let connection_id = connection_id(request)?;
        dispatch_result(
            self.command_exec
                .resize(connection_id, parse_params(params)?)
                .await?,
        )
    }

    pub(super) async fn handle_command_exec_terminate_impl(
        &self,
        params: Option<Value>,
        request: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let connection_id = connection_id(request)?;
        dispatch_result(
            self.command_exec
                .terminate(connection_id, parse_params(params)?)
                .await?,
        )
    }
}

fn connection_id(
    request: Option<ConnectionRequestId>,
) -> Result<app_server_transport::ConnectionId, JsonRpcError> {
    request.map(|request| request.connection_id).ok_or_else(|| {
        JsonRpcError::new(
            error_codes::INVALID_REQUEST,
            "command/exec requires transport connection",
        )
    })
}
