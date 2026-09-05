//! Process-host connection driver owners.
//!
//! The V1 transport is intentionally small, but its state is split by
//! responsibility to keep the Codex-shaped layout meaningful: commands admit
//! caller work, responses route host messages, request tracking owns promises,
//! session registry owns live cells, and delegate runtime owns callbacks.

mod cell_ids;
mod cleanup;
mod commands;
mod delegate_runtime;
mod request_tracker;
mod responses;
mod session_registry;
mod state;
mod types;

use code_mode_protocol::host::ClientToHost;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub(crate) use cell_ids::{
    public_started_cell, public_wait_outcome, remote_cell_id, GenerationDelegate,
};
pub(crate) use commands::{register_request, watch_caller_cancellation};
pub(crate) use state::ConnectionState;
pub(crate) use types::{DriverEvent, PendingRequest};

pub(crate) struct ConnectionDriver {
    state: Arc<ConnectionState>,
    events: mpsc::Receiver<DriverEvent>,
    outgoing: mpsc::Sender<ClientToHost>,
    cancellation: CancellationToken,
}

impl ConnectionDriver {
    pub(crate) fn new(
        state: Arc<ConnectionState>,
        events: mpsc::Receiver<DriverEvent>,
        outgoing: mpsc::Sender<ClientToHost>,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            state,
            events,
            outgoing,
            cancellation,
        }
    }

    pub(crate) async fn run(mut self) {
        loop {
            tokio::select! {
                _ = self.cancellation.cancelled() => return,
                event = self.events.recv() => {
                    let Some(event) = event else {
                        self.state.fail("code mode host event stream closed");
                        return;
                    };
                    match event {
                        DriverEvent::HostMessage(message) => {
                            if let Err(error) = self.state.handle_host_message(message, &self.outgoing) {
                                self.state.fail(error);
                                return;
                            }
                        }
                        DriverEvent::Failed(reason) => {
                            self.state.fail(reason);
                            return;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "driver_tests.rs"]
mod tests;
