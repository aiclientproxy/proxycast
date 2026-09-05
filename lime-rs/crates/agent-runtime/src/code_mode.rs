use ::code_mode::{
    NoopRuntimeCodeModeSessionDelegate, ProcessCodeModeSessionProvider, RuntimeCodeModeCellId,
    RuntimeCodeModeExecuteRequest, RuntimeCodeModeFuture, RuntimeCodeModeSession,
    RuntimeCodeModeSessionDelegate, RuntimeCodeModeSessionHandle, RuntimeCodeModeSessionLimits,
    RuntimeCodeModeSessionProvider, RuntimeCodeModeStartedCell, RuntimeCodeModeWaitOutcome,
    RuntimeCodeModeWaitRequest,
};
use futures::future::join_all;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{watch, OnceCell};

type DelegateFactory =
    dyn Fn(&str) -> Result<Arc<dyn RuntimeCodeModeSessionDelegate>, String> + Send + Sync;

#[derive(Clone)]
pub struct RuntimeCodeModeServiceFactory {
    provider: Arc<dyn RuntimeCodeModeSessionProvider>,
    delegate_factory: Arc<DelegateFactory>,
    limits: RuntimeCodeModeSessionLimits,
}

impl RuntimeCodeModeServiceFactory {
    pub fn production() -> Self {
        Self::new(
            Arc::new(ProcessCodeModeSessionProvider::default()),
            |_thread_id| Ok(Arc::new(NoopRuntimeCodeModeSessionDelegate)),
        )
    }

    pub fn new(
        provider: Arc<dyn RuntimeCodeModeSessionProvider>,
        delegate_factory: impl Fn(&str) -> Result<Arc<dyn RuntimeCodeModeSessionDelegate>, String>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            provider,
            delegate_factory: Arc::new(delegate_factory),
            limits: RuntimeCodeModeSessionLimits::default(),
        }
    }

    pub fn with_limits(mut self, limits: RuntimeCodeModeSessionLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn create(&self, thread_id: &str) -> Result<RuntimeCodeModeService, String> {
        self.provider.availability()?;
        let delegate = (self.delegate_factory)(thread_id)?;
        Ok(RuntimeCodeModeService::new(
            thread_id,
            Arc::clone(&self.provider),
            delegate,
            self.limits.clone(),
        ))
    }
}

#[derive(Clone)]
pub struct RuntimeCodeModeService {
    inner: Arc<RuntimeCodeModeServiceInner>,
}

struct RuntimeCodeModeServiceInner {
    thread_id: Arc<str>,
    provider: Arc<dyn RuntimeCodeModeSessionProvider>,
    delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    limits: RuntimeCodeModeSessionLimits,
    session: OnceCell<RuntimeCodeModeSessionHandle>,
    active_cells: Arc<Mutex<HashSet<RuntimeCodeModeCellId>>>,
    dispatch_routes: Arc<Mutex<HashMap<RuntimeCodeModeCellId, CodeModeDelegate>>>,
    dispatch_gates: Arc<Mutex<HashMap<RuntimeCodeModeCellId, watch::Sender<bool>>>>,
    closed_cells: Arc<Mutex<HashSet<RuntimeCodeModeCellId>>>,
    shutting_down: AtomicBool,
}

type CodeModeDelegate = Arc<dyn RuntimeCodeModeSessionDelegate>;

struct RuntimeCodeModeDispatchDelegate {
    fallback: CodeModeDelegate,
    routes: Arc<Mutex<HashMap<RuntimeCodeModeCellId, CodeModeDelegate>>>,
    gates: Arc<Mutex<HashMap<RuntimeCodeModeCellId, watch::Sender<bool>>>>,
    closed_cells: Arc<Mutex<HashSet<RuntimeCodeModeCellId>>>,
}

impl RuntimeCodeModeService {
    fn new(
        thread_id: &str,
        provider: Arc<dyn RuntimeCodeModeSessionProvider>,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
        limits: RuntimeCodeModeSessionLimits,
    ) -> Self {
        let dispatch_routes = Arc::new(Mutex::new(HashMap::new()));
        let dispatch_gates = Arc::new(Mutex::new(HashMap::new()));
        let closed_cells = Arc::new(Mutex::new(HashSet::new()));
        let delegate = Arc::new(RuntimeCodeModeDispatchDelegate {
            fallback: delegate,
            routes: Arc::clone(&dispatch_routes),
            gates: Arc::clone(&dispatch_gates),
            closed_cells: Arc::clone(&closed_cells),
        });
        Self {
            inner: Arc::new(RuntimeCodeModeServiceInner {
                thread_id: Arc::from(thread_id),
                provider,
                delegate,
                limits,
                session: OnceCell::new(),
                active_cells: Arc::new(Mutex::new(HashSet::new())),
                dispatch_routes,
                dispatch_gates,
                closed_cells,
                shutting_down: AtomicBool::new(false),
            }),
        }
    }

    pub fn thread_id(&self) -> &str {
        &self.inner.thread_id
    }

    pub fn session_handle(&self) -> RuntimeCodeModeSessionHandle {
        RuntimeCodeModeSessionHandle::new(Arc::new(self.clone()))
    }

    pub async fn interrupt_active_cells(&self) {
        let Some(session) = self.inner.session.get().cloned() else {
            return;
        };
        let cell_ids = self
            .inner
            .active_cells
            .lock()
            .expect("code mode active cell set poisoned")
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let results = join_all(cell_ids.into_iter().map(|cell_id| {
            let session = session.clone();
            async move {
                let result = session.terminate(cell_id.clone()).await;
                (cell_id, result)
            }
        }))
        .await;
        let mut active_cells = self
            .inner
            .active_cells
            .lock()
            .expect("code mode active cell set poisoned");
        for (cell_id, result) in results {
            if result.is_ok() {
                active_cells.remove(&cell_id);
                self.remove_dispatch_route(&cell_id);
                self.mark_cell_closed(&cell_id);
            }
        }
    }

    pub async fn shutdown(&self) -> Result<(), String> {
        if self.inner.shutting_down.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let result = match self
            .inner
            .session
            .get_or_try_init(|| async {
                Err::<RuntimeCodeModeSessionHandle, String>(
                    "code mode session is shutting down".to_string(),
                )
            })
            .await
        {
            Ok(session) => session.shutdown().await,
            Err(_) => Ok(()),
        };
        self.inner
            .active_cells
            .lock()
            .expect("code mode active cell set poisoned")
            .clear();
        self.inner
            .dispatch_routes
            .lock()
            .expect("code mode dispatch routes poisoned")
            .drain()
            .for_each(|(cell_id, delegate)| delegate.cell_closed(&cell_id));
        self.inner
            .dispatch_gates
            .lock()
            .expect("code mode dispatch gates poisoned")
            .clear();
        result
    }

    async fn session(&self) -> Result<RuntimeCodeModeSessionHandle, String> {
        if self.inner.shutting_down.load(Ordering::Acquire) {
            return Err("code mode session is shutting down".to_string());
        }
        self.inner
            .session
            .get_or_try_init(|| async {
                if self.inner.shutting_down.load(Ordering::Acquire) {
                    return Err("code mode session is shutting down".to_string());
                }
                let session = self
                    .inner
                    .provider
                    .create_session_with_limits(
                        Arc::clone(&self.inner.delegate),
                        self.inner.limits.clone(),
                    )
                    .await?;
                if self.inner.shutting_down.load(Ordering::Acquire) {
                    let _ = session.shutdown().await;
                    return Err("code mode session is shutting down".to_string());
                }
                Ok(session)
            })
            .await
            .cloned()
    }

    fn register_cell(&self, cell_id: RuntimeCodeModeCellId, delegate: Option<CodeModeDelegate>) {
        self.inner
            .closed_cells
            .lock()
            .expect("code mode closed cells poisoned")
            .remove(&cell_id);
        self.inner
            .active_cells
            .lock()
            .expect("code mode active cell set poisoned")
            .insert(cell_id.clone());
        if let Some(delegate) = delegate {
            self.inner
                .dispatch_routes
                .lock()
                .expect("code mode dispatch routes poisoned")
                .insert(cell_id.clone(), delegate);
        }
        dispatch_gate(&self.inner.dispatch_gates, &cell_id).send_replace(true);
    }

    fn remove_dispatch_route(&self, cell_id: &RuntimeCodeModeCellId) {
        let delegate = self
            .inner
            .dispatch_routes
            .lock()
            .expect("code mode dispatch routes poisoned")
            .remove(cell_id);
        if let Some(delegate) = delegate {
            delegate.cell_closed(cell_id);
        }
        self.inner
            .dispatch_gates
            .lock()
            .expect("code mode dispatch gates poisoned")
            .remove(cell_id);
    }

    fn finish_cell(&self, cell_id: &RuntimeCodeModeCellId) {
        self.inner
            .active_cells
            .lock()
            .expect("code mode active cell set poisoned")
            .remove(cell_id);
        self.remove_dispatch_route(cell_id);
        self.mark_cell_closed(cell_id);
    }

    fn mark_cell_closed(&self, cell_id: &RuntimeCodeModeCellId) {
        self.inner
            .closed_cells
            .lock()
            .expect("code mode closed cells poisoned")
            .insert(cell_id.clone());
    }

    fn execute_with_delegate(
        &self,
        request: RuntimeCodeModeExecuteRequest,
        delegate: Option<CodeModeDelegate>,
    ) -> RuntimeCodeModeFuture<'_, RuntimeCodeModeStartedCell> {
        Box::pin(async move {
            let started = self.session().await?.execute(request).await?;
            let cell_id = started.cell_id.clone();
            self.register_cell(cell_id.clone(), delegate);
            let service = self.clone();
            let response_cell_id = cell_id.clone();
            Ok(RuntimeCodeModeStartedCell::new(
                cell_id,
                Box::pin(async move {
                    let result = started.initial_response().await;
                    let finished = match &result {
                        Ok(response) => response.is_terminal(),
                        Err(_) => true,
                    };
                    if finished {
                        service.finish_cell(&response_cell_id);
                    }
                    result
                }),
            ))
        })
    }
}

impl RuntimeCodeModeSession for RuntimeCodeModeService {
    fn execute<'a>(
        &'a self,
        request: RuntimeCodeModeExecuteRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeStartedCell> {
        RuntimeCodeModeService::execute_with_delegate(self, request, None)
    }

    fn execute_with_delegate<'a>(
        &'a self,
        request: RuntimeCodeModeExecuteRequest,
        delegate: Option<CodeModeDelegate>,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeStartedCell> {
        RuntimeCodeModeService::execute_with_delegate(self, request, delegate)
    }

    fn wait<'a>(
        &'a self,
        request: RuntimeCodeModeWaitRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            let cell_id = request.cell_id.clone();
            let outcome = self.session().await?.wait(request).await?;
            if matches!(&outcome, RuntimeCodeModeWaitOutcome::MissingCell(_))
                || outcome.clone().into_response().is_terminal()
            {
                self.finish_cell(&cell_id);
            }
            Ok(outcome)
        })
    }

    fn terminate<'a>(
        &'a self,
        cell_id: RuntimeCodeModeCellId,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            let outcome = self.session().await?.terminate(cell_id.clone()).await?;
            self.finish_cell(&cell_id);
            Ok(outcome)
        })
    }

    fn shutdown(&self) -> RuntimeCodeModeFuture<'_, ()> {
        Box::pin(RuntimeCodeModeService::shutdown(self))
    }
}

fn dispatch_gate(
    gates: &Mutex<HashMap<RuntimeCodeModeCellId, watch::Sender<bool>>>,
    cell_id: &RuntimeCodeModeCellId,
) -> watch::Sender<bool> {
    gates
        .lock()
        .expect("code mode dispatch gates poisoned")
        .entry(cell_id.clone())
        .or_insert_with(|| watch::channel(false).0)
        .clone()
}

async fn wait_for_dispatch_route(
    routes: &Mutex<HashMap<RuntimeCodeModeCellId, CodeModeDelegate>>,
    gates: &Mutex<HashMap<RuntimeCodeModeCellId, watch::Sender<bool>>>,
    closed_cells: &Mutex<HashSet<RuntimeCodeModeCellId>>,
    cell_id: &RuntimeCodeModeCellId,
    cancellation_token: &tokio_util::sync::CancellationToken,
) -> Result<Option<CodeModeDelegate>, String> {
    if closed_cells
        .lock()
        .expect("code mode closed cells poisoned")
        .contains(cell_id)
    {
        return Err(format!("code mode cell {cell_id} is already closed"));
    }
    if let Some(delegate) = routes
        .lock()
        .expect("code mode dispatch routes poisoned")
        .get(cell_id)
        .cloned()
    {
        return Ok(Some(delegate));
    }
    let mut ready = dispatch_gate(gates, cell_id).subscribe();
    loop {
        if *ready.borrow_and_update() {
            if closed_cells
                .lock()
                .expect("code mode closed cells poisoned")
                .contains(cell_id)
            {
                return Err(format!("code mode cell {cell_id} is already closed"));
            }
            return Ok(routes
                .lock()
                .expect("code mode dispatch routes poisoned")
                .get(cell_id)
                .cloned());
        }
        tokio::select! {
            changed = ready.changed() => {
                if changed.is_err() {
                    return Ok(None);
                }
            }
            _ = cancellation_token.cancelled() => {
                return Err("code mode nested dispatch cancelled".to_string());
            }
        }
    }
}

impl RuntimeCodeModeSessionDelegate for RuntimeCodeModeDispatchDelegate {
    fn invoke_tool<'a>(
        &'a self,
        invocation: code_mode::RuntimeCodeModeNestedToolCall,
        cancellation_token: tokio_util::sync::CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, serde_json::Value> {
        Box::pin(async move {
            let delegate = wait_for_dispatch_route(
                &self.routes,
                &self.gates,
                &self.closed_cells,
                &invocation.cell_id,
                &cancellation_token,
            )
            .await?;
            match delegate {
                Some(delegate) => delegate.invoke_tool(invocation, cancellation_token).await,
                None => {
                    self.fallback
                        .invoke_tool(invocation, cancellation_token)
                        .await
                }
            }
        })
    }

    fn notify<'a>(
        &'a self,
        tool_call_id: String,
        cell_id: RuntimeCodeModeCellId,
        text: String,
        cancellation_token: tokio_util::sync::CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, ()> {
        Box::pin(async move {
            let delegate = wait_for_dispatch_route(
                &self.routes,
                &self.gates,
                &self.closed_cells,
                &cell_id,
                &cancellation_token,
            )
            .await?;
            match delegate {
                Some(delegate) => {
                    delegate
                        .notify(tool_call_id, cell_id, text, cancellation_token)
                        .await
                }
                None => {
                    self.fallback
                        .notify(tool_call_id, cell_id, text, cancellation_token)
                        .await
                }
            }
        })
    }

    fn cell_closed(&self, cell_id: &RuntimeCodeModeCellId) {
        self.closed_cells
            .lock()
            .expect("code mode closed cells poisoned")
            .insert(cell_id.clone());
        if let Some(delegate) = self
            .routes
            .lock()
            .expect("code mode dispatch routes poisoned")
            .remove(cell_id)
        {
            delegate.cell_closed(cell_id);
        }
        self.gates
            .lock()
            .expect("code mode dispatch gates poisoned")
            .remove(cell_id);
        self.fallback.cell_closed(cell_id);
    }
}

#[cfg(test)]
mod tests;
