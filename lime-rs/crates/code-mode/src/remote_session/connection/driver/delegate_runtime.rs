//! Delegate callback lifecycle for process-host sessions.

use std::collections::HashMap;
use std::sync::Arc;

use code_mode_protocol::host::{
    ClientToHost, DelegateRequest, DelegateResponse, WireResult, MAX_PENDING_DELEGATE_CALLS,
};
use code_mode_protocol::RuntimeCodeModeCellId;
use code_mode_protocol::RuntimeCodeModeSessionDelegate;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub(crate) struct DelegateRuntime {
    calls: Arc<std::sync::Mutex<HashMap<u64, DelegateCall>>>,
}

struct DelegateCall {
    cell_id: RuntimeCodeModeCellId,
    cancellation: CancellationToken,
}

impl DelegateRuntime {
    pub(crate) fn new() -> Self {
        Self {
            calls: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    pub(crate) fn spawn(
        &self,
        id: u64,
        cell_id: RuntimeCodeModeCellId,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
        request: DelegateRequest,
        outgoing: mpsc::Sender<ClientToHost>,
    ) -> Result<(), String> {
        let cancellation = CancellationToken::new();
        {
            let mut calls = self
                .calls
                .lock()
                .expect("code mode delegate cancellations poisoned");
            if calls.len() >= MAX_PENDING_DELEGATE_CALLS {
                return Err(format!(
                    "code mode delegate request limit {MAX_PENDING_DELEGATE_CALLS} exceeded"
                ));
            }
            if calls
                .insert(
                    id,
                    DelegateCall {
                        cell_id,
                        cancellation: cancellation.clone(),
                    },
                )
                .is_some()
            {
                return Err(format!("duplicate code mode delegate request id {id}"));
            }
        }
        let calls = self.calls.clone();
        tokio::spawn(async move {
            let result = match request {
                DelegateRequest::InvokeTool { invocation } => delegate
                    .invoke_tool(invocation, cancellation.clone())
                    .await
                    .map(|result| DelegateResponse::ToolResult { result }),
                DelegateRequest::Notify {
                    tool_call_id,
                    cell_id,
                    text,
                } => delegate
                    .notify(tool_call_id, cell_id, text, cancellation.clone())
                    .await
                    .map(|()| DelegateResponse::NotificationDelivered),
            };
            calls
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

    pub(crate) fn cancel(&self, id: u64) {
        if let Some(call) = self
            .calls
            .lock()
            .expect("code mode delegate cancellations poisoned")
            .remove(&id)
        {
            call.cancellation.cancel();
        }
    }

    pub(crate) fn cancel_cell(&self, cell_id: &RuntimeCodeModeCellId) {
        let mut calls = self
            .calls
            .lock()
            .expect("code mode delegate cancellations poisoned");
        let cancelled = calls
            .extract_if(|_, call| call.cell_id == *cell_id)
            .map(|(_, call)| call.cancellation)
            .collect::<Vec<_>>();
        drop(calls);
        for token in cancelled {
            token.cancel();
        }
    }

    pub(crate) fn cancel_all(&self) {
        for call in self
            .calls
            .lock()
            .expect("code mode delegate cancellations poisoned")
            .drain()
            .map(|(_, call)| call)
        {
            call.cancellation.cancel();
        }
    }
}
