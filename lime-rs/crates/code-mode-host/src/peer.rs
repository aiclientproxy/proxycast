//! Transport peer lifecycle state.
//!
//! Framed stdio and gRPC both need a fail-closed disconnected signal.  The
//! concrete transports keep their protocol-specific queues, while this small
//! owner centralizes the observable failure state used by future peers.

use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
pub(crate) struct PeerState {
    disconnected: CancellationToken,
    failure: Arc<Mutex<Option<String>>>,
}

impl PeerState {
    pub(crate) fn new() -> Self {
        Self {
            disconnected: CancellationToken::new(),
            failure: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn fail(&self, reason: impl Into<String>) {
        let mut failure = self
            .failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if failure.is_none() {
            *failure = Some(reason.into());
        }
        self.disconnected.cancel();
    }

    pub(crate) fn is_disconnected(&self) -> bool {
        self.disconnected.is_cancelled()
    }
}
