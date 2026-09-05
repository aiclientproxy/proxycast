use super::*;
use crate::session_loop::{
    RuntimeSessionClosureTask, RuntimeSessionRegistry, RuntimeSessionTaskFailure,
    RuntimeSessionTaskOutcome,
};
use code_mode::{
    FunctionCallOutputContentItem, NoopRuntimeCodeModeSessionDelegate,
    RuntimeCodeModeNestedToolCall, RuntimeCodeModeResponse, RuntimeCodeModeSessionProviderFuture,
};
use std::future::pending;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::oneshot;
use tokio::time::{timeout, Duration};
use tokio_util::sync::CancellationToken;

#[derive(Default)]
struct RecordingSession {
    operations: Mutex<Vec<String>>,
}

impl RecordingSession {
    fn operations(&self) -> Vec<String> {
        self.operations
            .lock()
            .expect("recording session operations poisoned")
            .clone()
    }
}

impl RuntimeCodeModeSession for RecordingSession {
    fn execute<'a>(
        &'a self,
        request: RuntimeCodeModeExecuteRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeStartedCell> {
        Box::pin(async move {
            self.operations
                .lock()
                .expect("record execute")
                .push(format!("execute:{}", request.tool_call_id));
            let cell_id = RuntimeCodeModeCellId::new("cell-active");
            Ok(RuntimeCodeModeStartedCell::new(
                cell_id,
                Box::pin(pending()),
            ))
        })
    }

    fn wait<'a>(
        &'a self,
        request: RuntimeCodeModeWaitRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            Ok(RuntimeCodeModeWaitOutcome::LiveCell(
                RuntimeCodeModeResponse::Yielded {
                    cell_id: request.cell_id,
                    content_items: vec![FunctionCallOutputContentItem::InputText {
                        text: "pending".to_string(),
                    }],
                    code_mode_host_duration: None,
                },
            ))
        })
    }

    fn terminate<'a>(
        &'a self,
        cell_id: RuntimeCodeModeCellId,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            self.operations
                .lock()
                .expect("record terminate")
                .push(format!("terminate:{cell_id}"));
            Ok(RuntimeCodeModeWaitOutcome::LiveCell(
                RuntimeCodeModeResponse::Terminated {
                    cell_id,
                    content_items: Vec::new(),
                    code_mode_host_duration: None,
                },
            ))
        })
    }

    fn shutdown(&self) -> RuntimeCodeModeFuture<'_, ()> {
        Box::pin(async move {
            self.operations
                .lock()
                .expect("record shutdown")
                .push("shutdown".to_string());
            Ok(())
        })
    }
}

struct RecordingProvider {
    session: RuntimeCodeModeSessionHandle,
    creates: AtomicUsize,
    availability_error: Option<String>,
}

impl RecordingProvider {
    fn new(session: Arc<RecordingSession>) -> Self {
        Self {
            session: RuntimeCodeModeSessionHandle::new(session),
            creates: AtomicUsize::new(0),
            availability_error: None,
        }
    }

    fn unavailable(session: Arc<RecordingSession>) -> Self {
        Self {
            availability_error: Some("isolated host is unavailable".to_string()),
            ..Self::new(session)
        }
    }
}

impl RuntimeCodeModeSessionProvider for RecordingProvider {
    fn availability(&self) -> Result<(), String> {
        self.availability_error.clone().map_or(Ok(()), Err)
    }

    fn create_session<'a>(
        &'a self,
        _delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    ) -> RuntimeCodeModeSessionProviderFuture<'a> {
        Box::pin(async move {
            self.creates.fetch_add(1, Ordering::SeqCst);
            Ok(self.session.clone())
        })
    }
}

fn factory(provider: Arc<RecordingProvider>) -> RuntimeCodeModeServiceFactory {
    RuntimeCodeModeServiceFactory::new(provider, |_thread_id| {
        Ok(Arc::new(NoopRuntimeCodeModeSessionDelegate))
    })
}

#[derive(Default)]
struct DelegatingSession {
    delegate: Mutex<Option<Arc<dyn RuntimeCodeModeSessionDelegate>>>,
}

impl RuntimeCodeModeSession for DelegatingSession {
    fn execute<'a>(
        &'a self,
        _request: RuntimeCodeModeExecuteRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeStartedCell> {
        let delegate = self
            .delegate
            .lock()
            .expect("delegating session delegate poisoned")
            .clone()
            .expect("provider delegate installed before execute");
        Box::pin(async move {
            let cell_id = RuntimeCodeModeCellId::new("cell-nested");
            Ok(RuntimeCodeModeStartedCell::new(
                cell_id.clone(),
                Box::pin(async move {
                    let nested = delegate
                        .invoke_tool(
                            RuntimeCodeModeNestedToolCall {
                                cell_id: cell_id.clone(),
                                runtime_tool_call_id: "nested-call-1".to_string(),
                                tool_name: "read".to_string(),
                                kind: code_mode::CodeModeToolKind::Function,
                                input: Some(serde_json::json!({"path": "README.md"})),
                            },
                            CancellationToken::new(),
                        )
                        .await?;
                    Ok(RuntimeCodeModeResponse::Result {
                        cell_id,
                        content_items: vec![FunctionCallOutputContentItem::InputText {
                            text: nested.to_string(),
                        }],
                        code_mode_host_duration: None,
                        error_text: None,
                    })
                }),
            ))
        })
    }

    fn wait<'a>(
        &'a self,
        request: RuntimeCodeModeWaitRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            Ok(RuntimeCodeModeWaitOutcome::MissingCell(
                RuntimeCodeModeResponse::Result {
                    cell_id: request.cell_id,
                    content_items: Vec::new(),
                    code_mode_host_duration: None,
                    error_text: Some("cell not found".to_string()),
                },
            ))
        })
    }

    fn terminate<'a>(
        &'a self,
        cell_id: RuntimeCodeModeCellId,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            Ok(RuntimeCodeModeWaitOutcome::MissingCell(
                RuntimeCodeModeResponse::Result {
                    cell_id,
                    content_items: Vec::new(),
                    code_mode_host_duration: None,
                    error_text: Some("cell not found".to_string()),
                },
            ))
        })
    }

    fn shutdown(&self) -> RuntimeCodeModeFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}

#[derive(Default)]
struct RecordingNestedDelegate {
    calls: Mutex<Vec<RuntimeCodeModeNestedToolCall>>,
}

impl RuntimeCodeModeSessionDelegate for RecordingNestedDelegate {
    fn invoke_tool<'a>(
        &'a self,
        invocation: RuntimeCodeModeNestedToolCall,
        _cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, serde_json::Value> {
        self.calls
            .lock()
            .expect("nested delegate calls poisoned")
            .push(invocation);
        Box::pin(async { Ok(serde_json::json!({"content": "ok"})) })
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

struct DelegatingProvider {
    session: Arc<DelegatingSession>,
}

impl RuntimeCodeModeSessionProvider for DelegatingProvider {
    fn create_session<'a>(
        &'a self,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    ) -> RuntimeCodeModeSessionProviderFuture<'a> {
        self.session
            .delegate
            .lock()
            .expect("delegating provider delegate poisoned")
            .replace(delegate);
        let session: Arc<dyn RuntimeCodeModeSession> = self.session.clone();
        let session = RuntimeCodeModeSessionHandle::new(session);
        Box::pin(async move { Ok(session) })
    }
}

#[tokio::test]
async fn registry_requires_stable_thread_identity_and_skips_unavailable_provider() {
    let runtime = Arc::new(RecordingSession::default());
    let provider = Arc::new(RecordingProvider::unavailable(Arc::clone(&runtime)));
    let registry = RuntimeSessionRegistry::with_code_mode(factory(Arc::clone(&provider)));
    let session = registry
        .get_or_create("session-identity", "thread-identity")
        .await
        .expect("bind canonical identity");
    assert_eq!(session.thread_id(), "thread-identity");
    assert!(registry
        .get_or_create("session-identity", "thread-other")
        .await
        .is_err());

    let task = RuntimeSessionClosureTask::new(
        "turn-identity",
        Vec::new(),
        move |context, _input, _cancel| {
            Box::pin(async move {
                if context.thread_id() != "thread-identity" {
                    return Err(RuntimeSessionTaskFailure {
                        message: "task context lost canonical thread identity".to_string(),
                        reason_code: None,
                    });
                }
                if context.code_mode_session().is_some() {
                    return Err(RuntimeSessionTaskFailure {
                        message: "unavailable provider created a CodeMode owner".to_string(),
                        reason_code: None,
                    });
                }
                Ok(())
            })
        },
    );
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("submit identity task");
    assert_eq!(
        submission.completion.await.expect("identity completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    registry
        .shutdown("session-identity")
        .await
        .expect("shutdown identity actor");
    assert_eq!(provider.creates.load(Ordering::SeqCst), 0);
    assert!(runtime.operations().is_empty());
}

#[tokio::test]
async fn unused_service_shutdown_does_not_create_a_runtime_session() {
    let runtime = Arc::new(RecordingSession::default());
    let provider = Arc::new(RecordingProvider::new(Arc::clone(&runtime)));
    let registry = RuntimeSessionRegistry::with_code_mode(factory(Arc::clone(&provider)));
    registry
        .get_or_create("session-unused", "thread-unused")
        .await
        .expect("bind unused service");
    registry
        .shutdown("session-unused")
        .await
        .expect("shutdown unused service");
    assert_eq!(provider.creates.load(Ordering::SeqCst), 0);
    assert!(runtime.operations().is_empty());
}

#[tokio::test]
async fn actor_interrupt_terminates_active_cells_and_shutdown_closes_the_session() {
    let runtime = Arc::new(RecordingSession::default());
    let provider = Arc::new(RecordingProvider::new(Arc::clone(&runtime)));
    let registry = RuntimeSessionRegistry::with_code_mode(factory(Arc::clone(&provider)));
    let session = registry
        .get_or_create("session-active", "thread-active")
        .await
        .expect("bind active service");
    let (started_tx, started_rx) = oneshot::channel();
    let started_tx = Arc::new(Mutex::new(Some(started_tx)));
    let task = RuntimeSessionClosureTask::new(
        "turn-active",
        Vec::new(),
        move |context, _input, _cancel| {
            let started_tx = Arc::clone(&started_tx);
            Box::pin(async move {
                let code_mode =
                    context
                        .code_mode_session()
                        .ok_or_else(|| RuntimeSessionTaskFailure {
                            message: "CodeMode service is unavailable".to_string(),
                            reason_code: None,
                        })?;
                let started = code_mode
                    .execute(RuntimeCodeModeExecuteRequest {
                        tool_call_id: "call-active".to_string(),
                        source: "await tools.read({ path: 'README.md' })".to_string(),
                        enabled_tools: Vec::new(),
                        yield_time_ms: None,
                        max_output_tokens: None,
                        cancellation_token: None,
                    })
                    .await
                    .map_err(|message| RuntimeSessionTaskFailure {
                        message,
                        reason_code: None,
                    })?;
                if let Some(sender) = started_tx.lock().expect("started sender poisoned").take() {
                    let _ = sender.send(started.cell_id.clone());
                }
                started
                    .initial_response()
                    .await
                    .map(|_| ())
                    .map_err(|message| RuntimeSessionTaskFailure {
                        message,
                        reason_code: None,
                    })
            })
        },
    );
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("submit active CodeMode task");
    assert_eq!(
        started_rx.await.expect("active cell id"),
        RuntimeCodeModeCellId::new("cell-active")
    );
    assert!(session.interrupt().await.expect("interrupt active actor"));
    assert_eq!(
        submission.completion.await.expect("interrupt completion"),
        Ok(RuntimeSessionTaskOutcome::Interrupted)
    );
    registry
        .shutdown("session-active")
        .await
        .expect("shutdown active service");
    assert_eq!(provider.creates.load(Ordering::SeqCst), 1);
    assert_eq!(
        runtime.operations(),
        vec![
            "execute:call-active".to_string(),
            "terminate:cell-active".to_string(),
            "shutdown".to_string(),
        ]
    );
}

#[tokio::test]
async fn execute_binds_a_cell_delegate_before_nested_dispatch() {
    let session = Arc::new(DelegatingSession::default());
    let provider = Arc::new(DelegatingProvider {
        session: Arc::clone(&session),
    });
    let service = factory_for_delegating_provider(provider)
        .create("thread-nested")
        .expect("create CodeMode service");
    let handle = service.session_handle();
    let nested = Arc::new(RecordingNestedDelegate::default());
    let started = handle
        .execute_with_delegate(
            RuntimeCodeModeExecuteRequest {
                tool_call_id: "call-nested".to_string(),
                source: "await tools.read({ path: 'README.md' })".to_string(),
                enabled_tools: Vec::new(),
                yield_time_ms: None,
                max_output_tokens: None,
                cancellation_token: None,
            },
            Some(nested.clone()),
        )
        .await
        .expect("start nested cell");
    let response = started.initial_response().await.expect("nested response");
    assert!(response.is_terminal());
    let calls = nested.calls.lock().expect("nested calls").clone();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].cell_id, RuntimeCodeModeCellId::new("cell-nested"));
    assert_eq!(calls[0].tool_name, "read");
    service.shutdown().await.expect("shutdown nested service");
}

#[tokio::test]
async fn late_nested_dispatch_after_cell_completion_fails_without_waiting() {
    let session = Arc::new(DelegatingSession::default());
    let provider = Arc::new(DelegatingProvider {
        session: Arc::clone(&session),
    });
    let service = factory_for_delegating_provider(provider)
        .create("thread-late-dispatch")
        .expect("create CodeMode service");
    let handle = service.session_handle();
    let started = handle
        .execute_with_delegate(
            RuntimeCodeModeExecuteRequest {
                tool_call_id: "call-late".to_string(),
                source: "1".to_string(),
                enabled_tools: Vec::new(),
                yield_time_ms: None,
                max_output_tokens: None,
                cancellation_token: None,
            },
            Some(Arc::new(RecordingNestedDelegate::default())),
        )
        .await
        .expect("start late-dispatch cell");
    let cell_id = started.cell_id.clone();
    started
        .initial_response()
        .await
        .expect("complete late-dispatch cell");

    let delegate = session
        .delegate
        .lock()
        .expect("delegating session delegate poisoned")
        .clone()
        .expect("provider delegate installed before late dispatch");
    let result = timeout(
        Duration::from_millis(100),
        delegate.notify(
            "late-notify".to_string(),
            cell_id.clone(),
            "late output".to_string(),
            CancellationToken::new(),
        ),
    )
    .await
    .expect("late notify must not wait on a recreated gate");
    assert_eq!(
        result.expect_err("closed cell must reject late notify"),
        format!("code mode cell {cell_id} is already closed")
    );

    service
        .shutdown()
        .await
        .expect("shutdown late-dispatch service");
}

fn factory_for_delegating_provider(
    provider: Arc<DelegatingProvider>,
) -> RuntimeCodeModeServiceFactory {
    RuntimeCodeModeServiceFactory::new(provider, |_thread_id| {
        Ok(Arc::new(NoopRuntimeCodeModeSessionDelegate))
    })
}
