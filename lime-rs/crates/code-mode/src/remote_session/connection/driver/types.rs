//! Driver-local request types.
//!
//! Wire envelopes remain owned by `code-mode-protocol`; these types describe
//! the local promises that connect a host request to its caller.

use code_mode_protocol::host::{HostResponse, HostToClient};
use code_mode_protocol::{RuntimeCodeModeCellId, RuntimeCodeModeResponse};
use tokio::sync::oneshot;

pub(crate) enum PendingRequest {
    Standard(oneshot::Sender<Result<HostResponse, String>>),
    Execute {
        session_id: String,
        started: oneshot::Sender<Result<HostResponse, String>>,
        initial: oneshot::Sender<Result<RuntimeCodeModeResponse, String>>,
    },
    ExecuteStarted {
        cell_id: RuntimeCodeModeCellId,
        initial: oneshot::Sender<Result<RuntimeCodeModeResponse, String>>,
    },
}

pub(crate) enum DriverEvent {
    HostMessage(HostToClient),
    Failed(String),
}
