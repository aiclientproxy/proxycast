//! Reconnect policy and binding lifecycle for gRPC Code Mode sessions.

use super::{GrpcCodeModeSession, GrpcCodeModeSessionProvider};
use code_mode_protocol::{
    RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest, RuntimeCodeModeFuture,
    RuntimeCodeModeSession, RuntimeCodeModeSessionDelegate, RuntimeCodeModeSessionLimits,
    RuntimeCodeModeStartedCell, RuntimeCodeModeWaitOutcome, RuntimeCodeModeWaitRequest,
};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

pub(super) struct ReconnectableSession {
    inner: Arc<ReconnectInner>,
}

struct ReconnectInner {
    provider: GrpcCodeModeSessionProvider,
    delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    limits: RuntimeCodeModeSessionLimits,
    binding: Mutex<Option<SessionBinding>>,
    opening: Semaphore,
    next_generation: AtomicU64,
    shutdown: CancellationToken,
}

#[derive(Clone)]
struct SessionBinding {
    session: Arc<GrpcCodeModeSession>,
    generation: u64,
}

impl ReconnectableSession {
    pub(super) fn new(
        provider: GrpcCodeModeSessionProvider,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
        limits: RuntimeCodeModeSessionLimits,
    ) -> Self {
        Self {
            inner: Arc::new(ReconnectInner {
                provider,
                delegate,
                limits,
                binding: Mutex::new(None),
                opening: Semaphore::new(1),
                next_generation: AtomicU64::new(1),
                shutdown: CancellationToken::new(),
            }),
        }
    }

    pub(super) async fn initialize(&self) -> Result<(), String> {
        self.binding().await.map(|_| ())
    }

    async fn binding(&self) -> Result<SessionBinding, String> {
        if self.inner.shutdown.is_cancelled() {
            return Err("code-mode session is shutting down".to_string());
        }
        if let Some(binding) = self.inner.live_binding() {
            return Ok(binding);
        }
        let _permit = self
            .inner
            .opening
            .acquire()
            .await
            .map_err(|_| "code-mode session opening coordinator closed".to_string())?;
        if self.inner.shutdown.is_cancelled() {
            return Err("code-mode session is shutting down".to_string());
        }
        if let Some(binding) = self.inner.live_binding() {
            return Ok(binding);
        }
        let previous = {
            self.inner
                .binding
                .lock()
                .expect("code mode binding poisoned")
                .clone()
        };
        if let Some(previous) = previous {
            // A broken stream retires the binding immediately. Best-effort close
            // releases the host lease before the next generation is published.
            let _ = previous.session.shutdown().await;
        }
        let generation = self.inner.next_generation.fetch_add(1, Ordering::Relaxed);
        let delegate = Arc::new(super::generation::GenerationDelegate {
            delegate: Arc::clone(&self.inner.delegate),
            generation,
        });
        let session = self
            .inner
            .provider
            .open_binding(delegate, self.inner.limits.clone())
            .await?;
        if self.inner.shutdown.is_cancelled() {
            let _ = session.shutdown().await;
            return Err("code-mode session is shutting down".to_string());
        }
        let binding = SessionBinding {
            session,
            generation,
        };
        *self
            .inner
            .binding
            .lock()
            .expect("code mode binding poisoned") = Some(binding.clone());
        Ok(binding)
    }
}

impl ReconnectInner {
    fn live_binding(&self) -> Option<SessionBinding> {
        self.binding
            .lock()
            .expect("code mode binding poisoned")
            .as_ref()
            .filter(|binding| !binding.session.closed.load(Ordering::Acquire))
            .cloned()
    }
}

impl RuntimeCodeModeSession for ReconnectableSession {
    fn execute(
        &self,
        request: RuntimeCodeModeExecuteRequest,
    ) -> RuntimeCodeModeFuture<'_, RuntimeCodeModeStartedCell> {
        Box::pin(async move {
            let binding = self.binding().await?;
            let started = binding.session.execute(request).await?;
            Ok(super::generation::public_started_cell(
                binding.generation,
                started,
            ))
        })
    }

    fn wait(
        &self,
        request: RuntimeCodeModeWaitRequest,
    ) -> RuntimeCodeModeFuture<'_, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            let binding = self.binding().await?;
            let request = RuntimeCodeModeWaitRequest {
                cell_id: super::generation::remote_cell_id(binding.generation, &request.cell_id)?,
                yield_time_ms: request.yield_time_ms,
            };
            let outcome = binding.session.wait(request).await?;
            Ok(super::generation::public_wait_outcome(
                binding.generation,
                outcome,
            ))
        })
    }

    fn terminate(
        &self,
        cell_id: RuntimeCodeModeCellId,
    ) -> RuntimeCodeModeFuture<'_, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            let binding = self.binding().await?;
            let cell_id = super::generation::remote_cell_id(binding.generation, &cell_id)?;
            let outcome = binding.session.terminate(cell_id).await?;
            Ok(super::generation::public_wait_outcome(
                binding.generation,
                outcome,
            ))
        })
    }

    fn shutdown(&self) -> RuntimeCodeModeFuture<'_, ()> {
        Box::pin(async move {
            self.inner.shutdown.cancel();
            let binding = self
                .inner
                .binding
                .lock()
                .expect("code mode binding poisoned")
                .take();
            if let Some(binding) = binding {
                binding.session.shutdown().await?;
            }
            Ok(())
        })
    }
}
