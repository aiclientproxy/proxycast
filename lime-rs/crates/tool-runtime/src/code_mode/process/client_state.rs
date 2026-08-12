use super::protocol::{
    ClientToHost, DelegateRequest, DelegateResponse, HostResponse, HostToClient, WireResult,
    MAX_IN_FLIGHT_REQUESTS, MAX_PENDING_DELEGATE_CALLS,
};
use crate::code_mode::{RuntimeCodeModeResponse, RuntimeCodeModeSessionDelegate};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub(super) struct ConnectionState {
    pub(super) alive: AtomicBool,
    failure: Mutex<Option<String>>,
    pending: Mutex<HashMap<u64, PendingRequest>>,
    sessions: Mutex<HashMap<String, Arc<dyn RuntimeCodeModeSessionDelegate>>>,
    delegate_cancellations: Mutex<HashMap<u64, CancellationToken>>,
    caller_cancellation_watchers: Mutex<HashMap<u64, CancellationToken>>,
    pub(super) cancellation: CancellationToken,
}

pub(super) enum PendingRequest {
    Standard(oneshot::Sender<Result<HostResponse, String>>),
    Execute {
        started: oneshot::Sender<Result<HostResponse, String>>,
        initial: oneshot::Sender<Result<RuntimeCodeModeResponse, String>>,
    },
    ExecuteStarted(oneshot::Sender<Result<RuntimeCodeModeResponse, String>>),
}

impl ConnectionState {
    pub(super) fn new(cancellation: CancellationToken) -> Self {
        Self {
            alive: AtomicBool::new(true),
            failure: Mutex::new(None),
            pending: Mutex::new(HashMap::new()),
            sessions: Mutex::new(HashMap::new()),
            delegate_cancellations: Mutex::new(HashMap::new()),
            caller_cancellation_watchers: Mutex::new(HashMap::new()),
            cancellation,
        }
    }

    pub(super) fn insert_session(
        &self,
        session_id: String,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    ) {
        self.sessions
            .lock()
            .expect("code mode session delegates poisoned")
            .insert(session_id, delegate);
    }

    pub(super) fn remove_session(&self, session_id: &str) {
        self.sessions
            .lock()
            .expect("code mode session delegates poisoned")
            .remove(session_id);
    }

    pub(super) fn register_pending(&self, id: u64, request: PendingRequest) -> Result<(), String> {
        let mut pending = self
            .pending
            .lock()
            .expect("code mode pending requests poisoned");
        if pending.len() >= MAX_IN_FLIGHT_REQUESTS {
            return Err(format!(
                "code mode host request limit {MAX_IN_FLIGHT_REQUESTS} exceeded"
            ));
        }
        if pending.insert(id, request).is_some() {
            return Err(format!("duplicate code mode host request id {id}"));
        }
        Ok(())
    }

    pub(super) fn remove_pending(&self, id: u64) -> Option<PendingRequest> {
        self.pending
            .lock()
            .expect("code mode pending requests poisoned")
            .remove(&id)
    }

    pub(super) fn has_pending(&self, id: u64) -> bool {
        self.pending
            .lock()
            .expect("code mode pending requests poisoned")
            .contains_key(&id)
    }

    pub(super) fn register_caller_cancellation_watcher(&self, id: u64, token: CancellationToken) {
        self.caller_cancellation_watchers
            .lock()
            .expect("code mode caller cancellation watchers poisoned")
            .insert(id, token);
    }

    pub(super) fn failure_message(&self) -> String {
        self.failure
            .lock()
            .expect("code mode connection failure poisoned")
            .clone()
            .unwrap_or_else(|| "code mode host connection closed".to_string())
    }

    pub(super) fn fail(&self, reason: impl Into<String>) {
        if !self.alive.swap(false, Ordering::AcqRel) {
            return;
        }
        let reason = reason.into();
        self.failure
            .lock()
            .expect("code mode connection failure poisoned")
            .get_or_insert_with(|| reason.clone());
        let pending = std::mem::take(
            &mut *self
                .pending
                .lock()
                .expect("code mode pending requests poisoned"),
        );
        for request in pending.into_values() {
            match request {
                PendingRequest::Standard(sender) => {
                    let _ = sender.send(Err(reason.clone()));
                }
                PendingRequest::Execute { started, initial } => {
                    let _ = started.send(Err(reason.clone()));
                    let _ = initial.send(Err(reason.clone()));
                }
                PendingRequest::ExecuteStarted(initial) => {
                    let _ = initial.send(Err(reason.clone()));
                }
            }
        }
        for token in self
            .delegate_cancellations
            .lock()
            .expect("code mode delegate cancellations poisoned")
            .drain()
            .map(|(_, token)| token)
        {
            token.cancel();
        }
        for token in self
            .caller_cancellation_watchers
            .lock()
            .expect("code mode caller cancellation watchers poisoned")
            .drain()
            .map(|(_, token)| token)
        {
            token.cancel();
        }
        self.cancellation.cancel();
    }

    pub(super) fn handle_host_message(
        self: &Arc<Self>,
        message: HostToClient,
        outgoing: &mpsc::Sender<ClientToHost>,
    ) -> Result<(), String> {
        match message {
            HostToClient::Response { id, result } => self.complete_response(id, result),
            HostToClient::InitialResponse { id, result } => {
                let Some(PendingRequest::ExecuteStarted(initial)) = self.remove_pending(id) else {
                    return Err(format!("unexpected code mode initial response id {id}"));
                };
                self.finish_caller_cancellation_watcher(id);
                let _ = initial.send(result.into_result());
                Ok(())
            }
            HostToClient::DelegateRequest {
                id,
                session_id,
                request,
            } => self.spawn_delegate_request(id, session_id, request, outgoing.clone()),
            HostToClient::CancelDelegateRequest { id } => {
                if let Some(token) = self
                    .delegate_cancellations
                    .lock()
                    .expect("code mode delegate cancellations poisoned")
                    .remove(&id)
                {
                    token.cancel();
                }
                Ok(())
            }
            HostToClient::CellClosed {
                session_id,
                cell_id,
            } => {
                if let Some(delegate) = self
                    .sessions
                    .lock()
                    .expect("code mode session delegates poisoned")
                    .get(&session_id)
                    .cloned()
                {
                    delegate.cell_closed(&cell_id);
                }
                Ok(())
            }
            HostToClient::HostHello(_) | HostToClient::HandshakeRejected { .. } => {
                Err("code mode host sent a handshake message after initialization".to_string())
            }
        }
    }

    fn complete_response(&self, id: u64, result: WireResult<HostResponse>) -> Result<(), String> {
        let mut pending = self
            .pending
            .lock()
            .expect("code mode pending requests poisoned");
        match pending.remove(&id) {
            Some(PendingRequest::Standard(sender)) => {
                let _ = sender.send(result.into_result());
                Ok(())
            }
            Some(PendingRequest::Execute { started, initial }) => {
                match result.into_result() {
                    Ok(response) => {
                        pending.insert(id, PendingRequest::ExecuteStarted(initial));
                        let _ = started.send(Ok(response));
                    }
                    Err(message) => {
                        self.finish_caller_cancellation_watcher(id);
                        let _ = started.send(Err(message.clone()));
                        let _ = initial.send(Err(message));
                    }
                }
                Ok(())
            }
            Some(PendingRequest::ExecuteStarted(initial)) => {
                pending.insert(id, PendingRequest::ExecuteStarted(initial));
                Err(format!("duplicate code mode execute response id {id}"))
            }
            None => Err(format!("unexpected code mode response id {id}")),
        }
    }

    fn finish_caller_cancellation_watcher(&self, id: u64) {
        if let Some(token) = self
            .caller_cancellation_watchers
            .lock()
            .expect("code mode caller cancellation watchers poisoned")
            .remove(&id)
        {
            token.cancel();
        }
    }

    fn spawn_delegate_request(
        self: &Arc<Self>,
        id: u64,
        session_id: String,
        request: DelegateRequest,
        outgoing: mpsc::Sender<ClientToHost>,
    ) -> Result<(), String> {
        let delegate = self
            .sessions
            .lock()
            .expect("code mode session delegates poisoned")
            .get(&session_id)
            .cloned()
            .ok_or_else(|| format!("delegate request referenced unknown session {session_id}"))?;
        let cancellation_token = CancellationToken::new();
        {
            let mut cancellations = self
                .delegate_cancellations
                .lock()
                .expect("code mode delegate cancellations poisoned");
            if cancellations.len() >= MAX_PENDING_DELEGATE_CALLS {
                return Err(format!(
                    "code mode delegate request limit {MAX_PENDING_DELEGATE_CALLS} exceeded"
                ));
            }
            if cancellations
                .insert(id, cancellation_token.clone())
                .is_some()
            {
                return Err(format!("duplicate code mode delegate request id {id}"));
            }
        }
        let state = Arc::clone(self);
        tokio::spawn(async move {
            let result = match request {
                DelegateRequest::InvokeTool { invocation } => delegate
                    .invoke_tool(invocation, cancellation_token)
                    .await
                    .map(|result| DelegateResponse::ToolResult { result }),
                DelegateRequest::Notify {
                    tool_call_id,
                    cell_id,
                    text,
                } => delegate
                    .notify(tool_call_id, cell_id, text, cancellation_token)
                    .await
                    .map(|()| DelegateResponse::NotificationDelivered),
            };
            state
                .delegate_cancellations
                .lock()
                .expect("code mode delegate cancellations poisoned")
                .remove(&id);
            let _ = outgoing
                .send(ClientToHost::DelegateResponse {
                    id,
                    result: WireResult::from_result(result),
                })
                .await;
        });
        Ok(())
    }
}
