//! Host response routing.

use std::sync::Arc;

use code_mode_protocol::host::{ClientToHost, HostResponse, HostToClient};
use code_mode_protocol::RuntimeCodeModeResponse;
use tokio::sync::mpsc;

use super::commands::register_request;
use super::state::ConnectionState;
use super::types::PendingRequest;

pub(crate) fn handle_host_message(
    state: &Arc<ConnectionState>,
    message: HostToClient,
    outgoing: &mpsc::Sender<ClientToHost>,
) -> Result<(), String> {
    match message {
        HostToClient::Response { id, result } => complete_response(state, id, result.into_result()),
        HostToClient::InitialResponse { id, result } => {
            complete_initial_response(state, id, result.into_result())
        }
        HostToClient::DelegateRequest {
            id,
            session_id,
            request,
        } => {
            let cell_id = match &request {
                code_mode_protocol::host::DelegateRequest::InvokeTool { invocation } => {
                    &invocation.cell_id
                }
                code_mode_protocol::host::DelegateRequest::Notify { cell_id, .. } => cell_id,
            };
            let delegate = state.sessions.delegate(&session_id, cell_id)?;
            state
                .delegates
                .spawn(id, cell_id.clone(), delegate, request, outgoing.clone())
        }
        HostToClient::CancelDelegateRequest { id } => {
            state.delegates.cancel(id);
            Ok(())
        }
        HostToClient::CellClosed {
            session_id,
            cell_id,
        } => state.close_cell(&session_id, &cell_id),
        HostToClient::HostHello(_) | HostToClient::HandshakeRejected { .. } => {
            Err("code mode host sent a handshake message after initialization".to_string())
        }
    }
}

fn complete_response(
    state: &Arc<ConnectionState>,
    id: u64,
    result: Result<HostResponse, String>,
) -> Result<(), String> {
    let Some(pending) = state.remove_pending(id) else {
        return Err(format!("unexpected code mode response id {id}"));
    };
    match pending {
        PendingRequest::Standard(sender) => {
            let _ = sender.send(result);
            Ok(())
        }
        PendingRequest::Execute {
            session_id,
            started,
            initial,
        } => match result {
            Ok(HostResponse::ExecutionStarted { cell_id }) => {
                if let Err(error) = state.register_cell(&session_id, cell_id.clone()) {
                    let _ = started.send(Err(error.clone()));
                    let _ = initial.send(Err(error.clone()));
                    return Err(error);
                }
                register_request(
                    state,
                    id,
                    PendingRequest::ExecuteStarted {
                        cell_id: cell_id.clone(),
                        initial,
                    },
                )?;
                let _ = started.send(Ok(HostResponse::ExecutionStarted { cell_id }));
                Ok(())
            }
            Ok(response) => {
                let error = format!("unexpected code mode execute response: {response:?}");
                state.finish_caller_cancellation_watcher(id);
                let _ = started.send(Err(error.clone()));
                let _ = initial.send(Err(error.clone()));
                Err(error)
            }
            Err(error) => {
                state.finish_caller_cancellation_watcher(id);
                let _ = started.send(Err(error.clone()));
                let _ = initial.send(Err(error));
                Ok(())
            }
        },
        PendingRequest::ExecuteStarted { initial, .. } => {
            let _ = initial.send(Err(format!("duplicate code mode response id {id}")));
            Err(format!("duplicate code mode response id {id}"))
        }
    }
}

fn complete_initial_response(
    state: &Arc<ConnectionState>,
    id: u64,
    result: Result<RuntimeCodeModeResponse, String>,
) -> Result<(), String> {
    let Some(PendingRequest::ExecuteStarted {
        cell_id, initial, ..
    }) = state.remove_pending(id)
    else {
        return Err(format!("unexpected code mode initial response id {id}"));
    };
    state.finish_caller_cancellation_watcher(id);
    let result = match result {
        Ok(response) if response.cell_id() == &cell_id => Ok(response),
        Ok(response) => Err(format!(
            "code mode host returned initial response for cell {} after starting {cell_id}",
            response.cell_id()
        )),
        Err(error) => Err(error),
    };
    let protocol_error = result
        .as_ref()
        .err()
        .filter(|error| error.contains("after starting"))
        .cloned();
    let _ = initial.send(result);
    protocol_error.map_or(Ok(()), Err)
}
