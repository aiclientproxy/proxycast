//! Caller command admission and cancellation helpers.

use std::sync::Arc;

use code_mode_protocol::host::ClientToHost;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::state::ConnectionState;
use super::types::PendingRequest;

pub(crate) fn register_request(
    state: &ConnectionState,
    id: u64,
    request: PendingRequest,
) -> Result<(), String> {
    state.requests.register(id, request)
}

pub(crate) fn watch_caller_cancellation(
    state: Arc<ConnectionState>,
    id: u64,
    caller: CancellationToken,
    outgoing: mpsc::Sender<ClientToHost>,
) {
    let finished = CancellationToken::new();
    state.requests.register_caller_watcher(id, finished.clone());
    tokio::spawn(async move {
        tokio::select! {
            _ = caller.cancelled() => {
                if state.requests.contains(id) {
                    let _ = outgoing.send(ClientToHost::CancelRequest { id }).await;
                }
            }
            _ = finished.cancelled() => {}
        }
    });
}
