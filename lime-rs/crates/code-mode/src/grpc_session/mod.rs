//! gRPC client-side Code Mode session provider.

use code_mode_protocol::grpc::{self as proto, code_mode_host_client::CodeModeHostClient};
use code_mode_protocol::{
    RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest, RuntimeCodeModeFuture,
    RuntimeCodeModeSession, RuntimeCodeModeSessionDelegate, RuntimeCodeModeSessionHandle,
    RuntimeCodeModeSessionLimits, RuntimeCodeModeSessionProvider,
    RuntimeCodeModeSessionProviderFuture, RuntimeCodeModeStartedCell, RuntimeCodeModeWaitOutcome,
    RuntimeCodeModeWaitRequest,
};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;

pub(super) fn validate_identifier(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("code-mode gRPC {field} must not be empty"));
    }
    if value.len() > code_mode_protocol::grpc::MAX_IDENTIFIER_BYTES {
        return Err(format!(
            "code-mode gRPC {field} exceeds {} bytes",
            code_mode_protocol::grpc::MAX_IDENTIFIER_BYTES
        ));
    }
    Ok(())
}

mod callbacks;
mod completion;
#[cfg(test)]
#[path = "completion_tests.rs"]
mod completion_tests;
mod conversion;
#[cfg(test)]
#[path = "conversion_tests.rs"]
mod conversion_tests;
mod deadline;
#[cfg(test)]
#[path = "deadline_tests.rs"]
mod deadline_tests;
mod generation;
#[cfg(test)]
#[path = "generation_tests.rs"]
mod generation_tests;
mod operations;
mod reconnect;
mod state;
#[cfg(test)]
#[path = "state_tests.rs"]
mod state_tests;
mod transport;

/// Connects Code Mode sessions to a standalone gRPC host.
#[derive(Clone)]
pub struct GrpcCodeModeSessionProvider {
    endpoint: String,
}

impl GrpcCodeModeSessionProvider {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
        }
    }

    async fn open_binding(
        &self,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
        limits: RuntimeCodeModeSessionLimits,
    ) -> Result<Arc<GrpcCodeModeSession>, String> {
        let channel = transport::connect(&self.endpoint).await?;
        let mut client = CodeModeHostClient::new(channel.clone());
        let max_heap_size_bytes = limits
            .max_heap_size_bytes
            .map(u64::try_from)
            .transpose()
            .map_err(|error| format!("invalid code-mode heap size limit: {error}"))?;
        let limits = (limits.max_yield_time_ms.is_some() || max_heap_size_bytes.is_some())
            .then_some(proto::SessionCellExecutionLimits {
                max_yield_time_ms: limits.max_yield_time_ms,
                max_heap_size_bytes,
            });
        let mut events = client
            .open_session(proto::OpenSessionRequest {
                cell_execution_limits: limits,
            })
            .await
            .map_err(|error| format!("failed to open code-mode gRPC session: {error}"))?
            .into_inner();
        let opened = events
            .message()
            .await
            .map_err(|error| format!("failed to read code-mode session opening: {error}"))?
            .ok_or_else(|| "code-mode gRPC session closed during opening".to_string())?;
        let session_id = match opened.event {
            Some(proto::session_event::Event::Opened(opened))
                if validate_identifier(&opened.session_id, "session ID").is_ok() =>
            {
                opened.session_id
            }
            _ => {
                return Err(
                    "code-mode gRPC host returned an invalid session opening event".to_string(),
                )
            }
        };
        let mut subscriptions = CodeModeHostClient::new(channel.clone())
            .subscribe_to_tool_calls(proto::SubscribeToToolCallsRequest {
                session_id: session_id.clone(),
                tool_names: Vec::new(),
            })
            .await
            .map_err(|error| format!("failed to subscribe to code-mode tool calls: {error}"))?
            .into_inner();
        let session = Arc::new(GrpcCodeModeSession {
            channel,
            session_id,
            delegate,
            closed: AtomicBool::new(false),
            pending_callbacks: Mutex::new(HashMap::new()),
            cancelled_callbacks: Mutex::new(HashSet::new()),
        });
        let callback_session = Arc::clone(&session);
        tokio::spawn(async move {
            loop {
                match subscriptions.message().await {
                    Ok(Some(call)) => {
                        let callback_session = Arc::clone(&callback_session);
                        tokio::spawn(async move {
                            if let Err(error) = callback_session.handle_tool_call(call).await {
                                tracing::warn!(%error, "code-mode gRPC tool callback failed");
                            }
                        });
                    }
                    Ok(None) | Err(_) => {
                        callback_session.mark_closed().await;
                        break;
                    }
                }
            }
        });
        let event_session = Arc::clone(&session);
        tokio::spawn(async move {
            loop {
                match events.message().await {
                    Ok(Some(event)) => {
                        if let Err(error) = event_session.handle_session_event(event).await {
                            tracing::warn!(%error, "code-mode gRPC session event failed");
                        }
                    }
                    Ok(None) | Err(_) => {
                        event_session.mark_closed().await;
                        break;
                    }
                }
            }
        });
        Ok(session)
    }
}

impl RuntimeCodeModeSessionProvider for GrpcCodeModeSessionProvider {
    fn create_session<'a>(
        &'a self,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    ) -> RuntimeCodeModeSessionProviderFuture<'a> {
        self.create_session_with_limits(delegate, RuntimeCodeModeSessionLimits::default())
    }

    fn create_session_with_limits<'a>(
        &'a self,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
        limits: RuntimeCodeModeSessionLimits,
    ) -> RuntimeCodeModeSessionProviderFuture<'a> {
        Box::pin(async move {
            let session = reconnect::ReconnectableSession::new(self.clone(), delegate, limits);
            session.initialize().await?;
            Ok(RuntimeCodeModeSessionHandle::new(Arc::new(session)))
        })
    }
}

struct GrpcCodeModeSession {
    channel: Channel,
    session_id: String,
    delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    closed: AtomicBool,
    pending_callbacks: Mutex<HashMap<String, CancellationToken>>,
    cancelled_callbacks: Mutex<HashSet<String>>,
}

impl GrpcCodeModeSession {
    async fn mark_closed(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let callbacks = std::mem::take(&mut *self.pending_callbacks.lock().await);
        for cancellation in callbacks.into_values() {
            cancellation.cancel();
        }
        self.cancelled_callbacks.lock().await.clear();
    }

    async fn handle_tool_call(&self, call: proto::ToolCall) -> Result<(), String> {
        callbacks::tool_call(self, call).await
    }

    async fn handle_session_event(&self, event: proto::SessionEvent) -> Result<(), String> {
        callbacks::session_event(self, event).await
    }
}

impl RuntimeCodeModeSession for GrpcCodeModeSession {
    fn execute(
        &self,
        request: RuntimeCodeModeExecuteRequest,
    ) -> RuntimeCodeModeFuture<'_, RuntimeCodeModeStartedCell> {
        Box::pin(operations::execute(self, request))
    }

    fn wait(
        &self,
        request: RuntimeCodeModeWaitRequest,
    ) -> RuntimeCodeModeFuture<'_, RuntimeCodeModeWaitOutcome> {
        Box::pin(operations::wait(self, request))
    }

    fn terminate(
        &self,
        cell_id: RuntimeCodeModeCellId,
    ) -> RuntimeCodeModeFuture<'_, RuntimeCodeModeWaitOutcome> {
        Box::pin(operations::terminate(self, cell_id))
    }

    fn shutdown(&self) -> RuntimeCodeModeFuture<'_, ()> {
        Box::pin(operations::shutdown(self))
    }
}
