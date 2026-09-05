//! Connection lifecycle state.
//!
//! This module only composes the driver owners. Request routing, session
//! admission, delegate execution and cleanup live in their focused modules.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use code_mode_protocol::host::{ClientToHost, HostToClient};
use code_mode_protocol::{RuntimeCodeModeCellId, RuntimeCodeModeSessionDelegate};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::cleanup::Cleanup;
use super::delegate_runtime::DelegateRuntime;
use super::request_tracker::RequestTracker;
use super::session_registry::SessionRegistry;
use super::types::PendingRequest;

pub(crate) struct ConnectionState {
    pub(crate) alive: AtomicBool,
    pub(crate) failure: Mutex<Option<String>>,
    pub(crate) requests: RequestTracker,
    pub(crate) sessions: SessionRegistry,
    pub(crate) delegates: DelegateRuntime,
    pub(crate) cleanup: Cleanup,
    pub(crate) cancellation: CancellationToken,
}

impl ConnectionState {
    pub(crate) fn new(cancellation: CancellationToken) -> Self {
        Self {
            alive: AtomicBool::new(true),
            failure: Mutex::new(None),
            requests: RequestTracker::new(),
            sessions: SessionRegistry::new(),
            delegates: DelegateRuntime::new(),
            cleanup: Cleanup::new(cancellation.clone()),
            cancellation,
        }
    }

    pub(crate) fn insert_session(
        &self,
        session_id: String,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    ) {
        self.sessions.insert(session_id, delegate);
    }

    pub(crate) fn remove_session(&self, session_id: &str) {
        let cells = self.sessions.remove(session_id);
        for cell in &cells {
            self.delegates.cancel_cell(&cell.cell_id);
        }
        self.cleanup.close_cells(cells);
    }

    pub(crate) fn register_cell(
        &self,
        session_id: &str,
        cell_id: RuntimeCodeModeCellId,
    ) -> Result<(), String> {
        self.sessions.register_cell(session_id, cell_id)
    }

    pub(crate) fn remove_pending(&self, id: u64) -> Option<PendingRequest> {
        self.requests.remove(id)
    }

    #[cfg(test)]
    pub(crate) fn pending_request_count(&self) -> usize {
        self.requests.len()
    }

    #[cfg(test)]
    pub(crate) fn register_pending(&self, id: u64, request: PendingRequest) -> Result<(), String> {
        self.requests.register(id, request)
    }

    pub(crate) fn finish_caller_cancellation_watcher(&self, id: u64) {
        self.requests.finish_caller_watcher(id);
    }

    pub(crate) fn failure_message(&self) -> String {
        self.failure
            .lock()
            .expect("code mode connection failure poisoned")
            .clone()
            .unwrap_or_else(|| "code mode host connection closed".to_string())
    }

    pub(crate) fn fail(&self, reason: impl Into<String>) {
        if !self.alive.swap(false, Ordering::AcqRel) {
            return;
        }
        let reason = reason.into();
        self.failure
            .lock()
            .expect("code mode connection failure poisoned")
            .get_or_insert_with(|| reason.clone());
        self.requests.fail_all(&reason);
        let cells = self.sessions.drain();
        self.cleanup.close_cells(cells);
        self.delegates.cancel_all();
        self.cleanup.fail();
        self.cancellation.cancel();
    }

    pub(crate) fn close_cell(
        &self,
        session_id: &str,
        cell_id: &RuntimeCodeModeCellId,
    ) -> Result<(), String> {
        let cell = self
            .sessions
            .close_cell(session_id, cell_id)
            .ok_or_else(|| {
                format!("code mode host closed unknown cell {cell_id} in session {session_id}")
            })?;
        self.delegates.cancel_cell(cell_id);
        self.cleanup.close_cells([cell]);
        Ok(())
    }

    pub(crate) fn handle_host_message(
        self: &Arc<Self>,
        message: HostToClient,
        outgoing: &mpsc::Sender<ClientToHost>,
    ) -> Result<(), String> {
        super::responses::handle_host_message(self, message, outgoing)
    }
}
