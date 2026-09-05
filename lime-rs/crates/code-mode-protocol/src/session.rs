//! Session lifecycle and delegate contracts for Code Mode.

use crate::runtime::{
    RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest, RuntimeCodeModeFuture,
    RuntimeCodeModeNestedToolCall, RuntimeCodeModeSessionLimits, RuntimeCodeModeStartedCell,
    RuntimeCodeModeWaitOutcome, RuntimeCodeModeWaitRequest,
};
use serde_json::Value;
use std::fmt;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub trait RuntimeCodeModeSession: Send + Sync {
    fn execute(
        &self,
        request: RuntimeCodeModeExecuteRequest,
    ) -> RuntimeCodeModeFuture<'_, RuntimeCodeModeStartedCell>;
    fn execute_with_delegate(
        &self,
        request: RuntimeCodeModeExecuteRequest,
        _delegate: Option<Arc<dyn RuntimeCodeModeSessionDelegate>>,
    ) -> RuntimeCodeModeFuture<'_, RuntimeCodeModeStartedCell> {
        self.execute(request)
    }
    fn wait(
        &self,
        request: RuntimeCodeModeWaitRequest,
    ) -> RuntimeCodeModeFuture<'_, RuntimeCodeModeWaitOutcome>;
    fn terminate(
        &self,
        cell_id: RuntimeCodeModeCellId,
    ) -> RuntimeCodeModeFuture<'_, RuntimeCodeModeWaitOutcome>;
    fn shutdown(&self) -> RuntimeCodeModeFuture<'_, ()>;
}

#[derive(Clone)]
pub struct RuntimeCodeModeSessionHandle(Arc<dyn RuntimeCodeModeSession>);

impl RuntimeCodeModeSessionHandle {
    pub fn new(session: Arc<dyn RuntimeCodeModeSession>) -> Self {
        Self(session)
    }

    pub async fn execute(
        &self,
        request: RuntimeCodeModeExecuteRequest,
    ) -> Result<RuntimeCodeModeStartedCell, String> {
        self.0.execute(request).await
    }

    pub async fn execute_with_delegate(
        &self,
        request: RuntimeCodeModeExecuteRequest,
        delegate: Option<Arc<dyn RuntimeCodeModeSessionDelegate>>,
    ) -> Result<RuntimeCodeModeStartedCell, String> {
        self.0.execute_with_delegate(request, delegate).await
    }

    pub async fn wait(
        &self,
        request: RuntimeCodeModeWaitRequest,
    ) -> Result<RuntimeCodeModeWaitOutcome, String> {
        self.0.wait(request).await
    }

    pub async fn terminate(
        &self,
        cell_id: RuntimeCodeModeCellId,
    ) -> Result<RuntimeCodeModeWaitOutcome, String> {
        self.0.terminate(cell_id).await
    }

    pub async fn shutdown(&self) -> Result<(), String> {
        self.0.shutdown().await
    }
}

impl fmt::Debug for RuntimeCodeModeSessionHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RuntimeCodeModeSessionHandle(<runtime>)")
    }
}

pub trait RuntimeCodeModeSessionDelegate: Send + Sync {
    fn invoke_tool<'a>(
        &'a self,
        invocation: RuntimeCodeModeNestedToolCall,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, Value>;
    fn notify<'a>(
        &'a self,
        tool_call_id: String,
        cell_id: RuntimeCodeModeCellId,
        text: String,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, ()>;
    fn cell_closed(&self, cell_id: &RuntimeCodeModeCellId);
}

pub struct NoopRuntimeCodeModeSessionDelegate;

impl RuntimeCodeModeSessionDelegate for NoopRuntimeCodeModeSessionDelegate {
    fn invoke_tool<'a>(
        &'a self,
        _invocation: RuntimeCodeModeNestedToolCall,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, Value> {
        Box::pin(async move {
            cancellation_token.cancelled().await;
            Err("code mode nested tools are unavailable".to_string())
        })
    }

    fn notify<'a>(
        &'a self,
        _tool_call_id: String,
        _cell_id: RuntimeCodeModeCellId,
        _text: String,
        _cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

    fn cell_closed(&self, _cell_id: &RuntimeCodeModeCellId) {}
}

pub type RuntimeCodeModeSessionProviderFuture<'a> =
    RuntimeCodeModeFuture<'a, RuntimeCodeModeSessionHandle>;

pub trait RuntimeCodeModeSessionProvider: Send + Sync {
    fn availability(&self) -> Result<(), String> {
        Ok(())
    }

    fn create_session<'a>(
        &'a self,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    ) -> RuntimeCodeModeSessionProviderFuture<'a>;

    fn create_session_with_limits<'a>(
        &'a self,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
        limits: RuntimeCodeModeSessionLimits,
    ) -> RuntimeCodeModeSessionProviderFuture<'a> {
        if limits == RuntimeCodeModeSessionLimits::default() {
            self.create_session(delegate)
        } else {
            Box::pin(async {
                Err("code mode session provider does not support resource limits".to_string())
            })
        }
    }
}

pub type CodeModeSession = dyn RuntimeCodeModeSession;
pub type CodeModeSessionDelegate = dyn RuntimeCodeModeSessionDelegate;
pub type CodeModeSessionProvider = dyn RuntimeCodeModeSessionProvider;
pub type CodeModeSessionCellExecutionLimits = RuntimeCodeModeSessionLimits;
