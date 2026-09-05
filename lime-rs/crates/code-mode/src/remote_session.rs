use crate::default_code_mode_host_path;
mod connection;

#[cfg(test)]
#[path = "remote_session_tests.rs"]
mod tests;

use self::connection::driver::{
    public_started_cell, public_wait_outcome, remote_cell_id, GenerationDelegate,
};
use self::connection::ProcessConnection;
use code_mode_protocol::{
    RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest, RuntimeCodeModeFuture,
    RuntimeCodeModeSession, RuntimeCodeModeSessionDelegate, RuntimeCodeModeSessionHandle,
    RuntimeCodeModeSessionLimits, RuntimeCodeModeSessionProvider,
    RuntimeCodeModeSessionProviderFuture, RuntimeCodeModeStartedCell, RuntimeCodeModeWaitOutcome,
    RuntimeCodeModeWaitRequest,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tokio_util::sync::CancellationToken;

/// Shared process-owned host lifecycle.
///
/// A provider may create multiple logical sessions, all multiplexed over one
/// sidecar connection while it is alive. Once the connection dies, the next
/// binding request replaces it with a fresh host process.
struct ProcessHost {
    host_path: PathBuf,
    connection: AsyncMutex<Option<Arc<ProcessConnection>>>,
    connect_permit: Semaphore,
}

impl ProcessHost {
    fn new(host_path: PathBuf) -> Self {
        Self {
            host_path,
            connection: AsyncMutex::new(None),
            connect_permit: Semaphore::new(1),
        }
    }

    async fn connection(&self) -> Result<Arc<ProcessConnection>, String> {
        if let Some(connection) = self.live_connection().await {
            return Ok(connection);
        }

        let _permit = self
            .connect_permit
            .acquire()
            .await
            .map_err(|_| "code mode host connection coordinator closed".to_string())?;
        if let Some(connection) = self.live_connection().await {
            return Ok(connection);
        }

        let connection = Arc::new(ProcessConnection::spawn(&self.host_path).await?);
        self.connection
            .lock()
            .await
            .replace(Arc::clone(&connection));
        Ok(connection)
    }

    async fn live_connection(&self) -> Option<Arc<ProcessConnection>> {
        self.connection
            .lock()
            .await
            .as_ref()
            .filter(|connection| connection.is_alive())
            .cloned()
    }
}

/// Process-owned Code Mode provider.
pub struct ProcessCodeModeSessionProvider {
    host: Arc<ProcessHost>,
    next_session_id: Arc<AtomicU64>,
}

impl ProcessCodeModeSessionProvider {
    pub fn with_host_path(host_path: PathBuf) -> Self {
        Self {
            host: Arc::new(ProcessHost::new(host_path)),
            next_session_id: Arc::new(AtomicU64::new(1)),
        }
    }

    fn next_session_id(&self) -> String {
        format!(
            "session-{}",
            self.next_session_id.fetch_add(1, Ordering::Relaxed)
        )
    }

    async fn connection(&self) -> Result<Arc<ProcessConnection>, String> {
        self.host.connection().await
    }
}

impl Clone for ProcessCodeModeSessionProvider {
    fn clone(&self) -> Self {
        Self {
            host: Arc::clone(&self.host),
            next_session_id: Arc::clone(&self.next_session_id),
        }
    }
}

impl Default for ProcessCodeModeSessionProvider {
    fn default() -> Self {
        Self::with_host_path(default_code_mode_host_path())
    }
}

impl RuntimeCodeModeSessionProvider for ProcessCodeModeSessionProvider {
    fn availability(&self) -> Result<(), String> {
        if self.host.host_path.is_file() {
            Ok(())
        } else {
            Err(format!(
                "code mode host executable was not found: {}",
                self.host.host_path.display()
            ))
        }
    }

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
            self.availability()?;
            let session = ReconnectableSession::new(self.clone(), delegate, limits);
            session.initialize().await?;
            Ok(RuntimeCodeModeSessionHandle::new(Arc::new(session)))
        })
    }
}

struct ReconnectableSession {
    inner: Arc<ReconnectInner>,
}

struct ReconnectInner {
    provider: ProcessCodeModeSessionProvider,
    delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    limits: RuntimeCodeModeSessionLimits,
    binding: Mutex<Option<SessionBinding>>,
    opening: Semaphore,
    next_generation: AtomicU64,
    shutdown: CancellationToken,
}

#[derive(Clone)]
struct SessionBinding {
    connection: Arc<ProcessConnection>,
    session_id: String,
    generation: u64,
}

impl ReconnectableSession {
    fn new(
        provider: ProcessCodeModeSessionProvider,
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

    async fn initialize(&self) -> Result<(), String> {
        self.binding().await.map(|_| ())
    }

    async fn binding(&self) -> Result<SessionBinding, String> {
        if self.inner.shutdown.is_cancelled() {
            return Err("code mode session is shutting down".to_string());
        }
        if let Some(binding) = self.inner.live_binding() {
            return Ok(binding);
        }

        let _permit = self
            .inner
            .opening
            .acquire()
            .await
            .map_err(|_| "code mode session opening coordinator closed".to_string())?;
        if self.inner.shutdown.is_cancelled() {
            return Err("code mode session is shutting down".to_string());
        }
        if let Some(binding) = self.inner.live_binding() {
            return Ok(binding);
        }

        let previous = self
            .inner
            .binding
            .lock()
            .expect("code mode binding poisoned")
            .take();
        if let Some(previous) = previous {
            // Retire the old lease before publishing the next generation.
            let _ = previous.shutdown().await;
        }

        let generation = self.inner.next_generation.fetch_add(1, Ordering::Relaxed);
        let connection = self.inner.provider.connection().await?;
        let session_id = self.inner.provider.next_session_id();
        let delegate = Arc::new(GenerationDelegate {
            delegate: Arc::clone(&self.inner.delegate),
            generation,
        });
        connection
            .open_session(session_id.clone(), delegate, self.inner.limits.clone())
            .await?;
        if self.inner.shutdown.is_cancelled() {
            let _ = connection.shutdown_session(session_id).await;
            return Err("code mode session is shutting down".to_string());
        }

        let binding = SessionBinding {
            connection,
            session_id,
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
            .filter(|binding| binding.connection.is_alive())
            .cloned()
    }
}

impl SessionBinding {
    async fn shutdown(&self) -> Result<(), String> {
        self.connection
            .shutdown_session(self.session_id.clone())
            .await
    }
}

impl RuntimeCodeModeSession for ReconnectableSession {
    fn execute<'a>(
        &'a self,
        request: RuntimeCodeModeExecuteRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeStartedCell> {
        Box::pin(async move {
            let binding = self.binding().await?;
            let started = binding
                .connection
                .execute(binding.session_id.clone(), request)
                .await?;
            Ok(public_started_cell(binding.generation, started))
        })
    }

    fn wait<'a>(
        &'a self,
        request: RuntimeCodeModeWaitRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            let binding = self.binding().await?;
            let request = RuntimeCodeModeWaitRequest {
                cell_id: remote_cell_id(binding.generation, &request.cell_id)?,
                yield_time_ms: request.yield_time_ms,
            };
            let outcome = binding
                .connection
                .wait(binding.session_id.clone(), request)
                .await?;
            Ok(public_wait_outcome(binding.generation, outcome))
        })
    }

    fn terminate<'a>(
        &'a self,
        cell_id: RuntimeCodeModeCellId,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            let binding = self.binding().await?;
            let cell_id = remote_cell_id(binding.generation, &cell_id)?;
            let outcome = binding
                .connection
                .terminate(binding.session_id.clone(), cell_id)
                .await?;
            Ok(public_wait_outcome(binding.generation, outcome))
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
                binding.shutdown().await?;
            }
            Ok(())
        })
    }
}

impl Drop for ReconnectInner {
    fn drop(&mut self) {
        self.shutdown.cancel();
        let binding = self
            .binding
            .get_mut()
            .expect("code mode binding poisoned")
            .take();
        let Some(binding) = binding else {
            return;
        };
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::spawn(async move {
                let _ = binding.shutdown().await;
            });
        }
    }
}
