use anyhow::{Context, Result};
use code_mode::GrpcCodeModeSessionProvider;
use code_mode_host::GrpcCodeModeHost;
use code_mode_protocol::grpc::code_mode_host_server::CodeModeHostServer;
use code_mode_protocol::{
    FunctionCallOutputContentItem, RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest,
    RuntimeCodeModeNestedToolCall, RuntimeCodeModeResponse, RuntimeCodeModeSessionDelegate,
    RuntimeCodeModeSessionHandle, RuntimeCodeModeSessionProvider, RuntimeCodeModeTool,
    RuntimeCodeModeWaitOutcome, RuntimeToolDefinition, RuntimeToolIdentity,
};
use serde_json::json;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::transport::Server;

const TEST_TIMEOUT: Duration = Duration::from_secs(15);

struct HostHarness {
    endpoint: String,
    task: JoinHandle<std::result::Result<(), tonic::transport::Error>>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl HostHarness {
    async fn start() -> Result<Self> {
        Self::start_at(SocketAddr::from(([127, 0, 0, 1], 0))).await
    }

    async fn start_at(address: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(address)
            .await
            .context("bind code-mode gRPC test host")?;
        let address = listener.local_addr().context("read test host address")?;
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(
            Server::builder()
                .add_service(CodeModeHostServer::new(GrpcCodeModeHost::new()))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = shutdown_rx.await;
                }),
        );
        tokio::time::timeout(TEST_TIMEOUT, async {
            loop {
                match tonic::transport::Endpoint::from_shared(format!("http://{address}"))
                    .context("build code-mode endpoint")?
                    .connect()
                    .await
                {
                    Ok(_) => return Ok::<_, anyhow::Error>(()),
                    Err(_) => tokio::task::yield_now().await,
                }
            }
        })
        .await
        .context("code-mode gRPC test host did not start")??;
        Ok(Self {
            endpoint: format!("http://{address}"),
            task,
            shutdown: Some(shutdown_tx),
        })
    }

    async fn stop(mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if tokio::time::timeout(Duration::from_secs(1), &mut self.task)
            .await
            .is_err()
        {
            self.task.abort();
            let _ = self.task.await;
        }
    }
}

#[derive(Default)]
struct RecordingDelegate {
    invocations: Mutex<Vec<RuntimeCodeModeNestedToolCall>>,
    notifications: Mutex<Vec<(String, RuntimeCodeModeCellId, String)>>,
    closed_cells: std::sync::Mutex<Vec<RuntimeCodeModeCellId>>,
    invocation_started: Notify,
    callback_cancelled: AtomicBool,
    block_invocations: AtomicBool,
}

impl RuntimeCodeModeSessionDelegate for RecordingDelegate {
    fn invoke_tool<'a>(
        &'a self,
        invocation: RuntimeCodeModeNestedToolCall,
        cancellation_token: CancellationToken,
    ) -> code_mode_protocol::RuntimeCodeModeFuture<'a, serde_json::Value> {
        Box::pin(async move {
            self.invocations.lock().await.push(invocation.clone());
            self.invocation_started.notify_waiters();
            if self.block_invocations.load(Ordering::Acquire) {
                cancellation_token.cancelled().await;
                self.callback_cancelled.store(true, Ordering::Release);
                return Err("nested invocation cancelled".to_string());
            }
            Ok(json!({"value": "output"}))
        })
    }

    fn notify<'a>(
        &'a self,
        tool_call_id: String,
        cell_id: RuntimeCodeModeCellId,
        text: String,
        _cancellation_token: CancellationToken,
    ) -> code_mode_protocol::RuntimeCodeModeFuture<'a, ()> {
        Box::pin(async move {
            self.notifications
                .lock()
                .await
                .push((tool_call_id, cell_id, text));
            Ok(())
        })
    }

    fn cell_closed(&self, cell_id: &RuntimeCodeModeCellId) {
        self.closed_cells
            .lock()
            .expect("closed cells lock")
            .push(cell_id.clone());
    }
}

fn tool(name: &str) -> RuntimeCodeModeTool {
    RuntimeCodeModeTool {
        identity: RuntimeToolIdentity::plain(name),
        definition: RuntimeToolDefinition::new(name, format!("{name} test tool"), json!({})),
        kind: code_mode_protocol::CodeModeToolKind::Function,
        code_name: name.to_string(),
        global_name: name.to_string(),
    }
}

fn request(source: &str) -> RuntimeCodeModeExecuteRequest {
    RuntimeCodeModeExecuteRequest {
        tool_call_id: "call-1".to_string(),
        source: source.to_string(),
        enabled_tools: Vec::new(),
        yield_time_ms: Some(5_000),
        max_output_tokens: Some(1_000),
        cancellation_token: None,
    }
}

async fn execute(
    session: &RuntimeCodeModeSessionHandle,
    request: RuntimeCodeModeExecuteRequest,
) -> Result<RuntimeCodeModeResponse> {
    tokio::time::timeout(TEST_TIMEOUT, async {
        session
            .execute(request)
            .await
            .map_err(anyhow::Error::msg)?
            .initial_response()
            .await
            .map_err(anyhow::Error::msg)
    })
    .await
    .context("timed out executing remote code-mode cell")?
}

async fn wait_for_cell_closed(delegate: &RecordingDelegate, cell_id: &RuntimeCodeModeCellId) {
    tokio::time::timeout(TEST_TIMEOUT, async {
        loop {
            if delegate
                .closed_cells
                .lock()
                .expect("closed cells lock")
                .iter()
                .any(|closed| closed == cell_id)
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("remote code-mode cell was not closed");
}

#[tokio::test]
async fn tcp_facade_executes_nested_tool_and_notification() -> Result<()> {
    let host = HostHarness::start().await?;
    let provider = GrpcCodeModeSessionProvider::new(host.endpoint.clone());
    let delegate = Arc::new(RecordingDelegate::default());
    let session = provider
        .create_session(delegate.clone())
        .await
        .map_err(anyhow::Error::msg)?;

    let mut request = request(
        r#"const result = await tools.echo({value: 7}); notify("notice"); text(result.value);"#,
    );
    request.enabled_tools = vec![tool("echo")];
    let response = execute(&session, request).await?;
    assert!(matches!(
        response,
        RuntimeCodeModeResponse::Result {
            content_items,
            error_text: None,
            ..
        } if content_items == vec![FunctionCallOutputContentItem::InputText { text: "output".to_string() }]
    ));
    tokio::time::timeout(TEST_TIMEOUT, async {
        loop {
            if delegate.notifications.lock().await.len() == 1 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .context("notification callback was not completed")?;
    assert_eq!(delegate.invocations.lock().await.len(), 1);
    assert_eq!(delegate.notifications.lock().await[0].2, "notice");
    session.shutdown().await.map_err(anyhow::Error::msg)?;
    host.stop().await;
    Ok(())
}

#[tokio::test]
async fn dropping_started_cell_reaps_remote_execution() -> Result<()> {
    let host = HostHarness::start().await?;
    let provider = GrpcCodeModeSessionProvider::new(host.endpoint.clone());
    let delegate = Arc::new(RecordingDelegate::default());
    let session = provider
        .create_session(delegate.clone())
        .await
        .map_err(anyhow::Error::msg)?;
    let mut pending = request("await new Promise(() => {});");
    pending.yield_time_ms = Some(1);
    let started = session.execute(pending).await.map_err(anyhow::Error::msg)?;
    let cell_id = started.cell_id.clone();
    drop(started);
    wait_for_cell_closed(&delegate, &cell_id).await;
    session.shutdown().await.map_err(anyhow::Error::msg)?;
    host.stop().await;
    Ok(())
}

#[tokio::test]
async fn terminate_cancels_a_remote_delegate_callback() -> Result<()> {
    let host = HostHarness::start().await?;
    let provider = GrpcCodeModeSessionProvider::new(host.endpoint.clone());
    let delegate = Arc::new(RecordingDelegate::default());
    delegate.block_invocations.store(true, Ordering::Release);
    let session = provider
        .create_session(delegate.clone())
        .await
        .map_err(anyhow::Error::msg)?;
    let mut pending = request("await tools.block({});");
    pending.enabled_tools = vec![tool("block")];
    pending.yield_time_ms = Some(1);
    let started = session.execute(pending).await.map_err(anyhow::Error::msg)?;
    let cell_id = started.cell_id.clone();
    tokio::time::timeout(TEST_TIMEOUT, delegate.invocation_started.notified())
        .await
        .context("nested callback did not start")?;
    let initial = started
        .initial_response()
        .await
        .map_err(anyhow::Error::msg)?;
    assert!(matches!(initial, RuntimeCodeModeResponse::Yielded { .. }));
    let outcome = session
        .terminate(cell_id)
        .await
        .map_err(anyhow::Error::msg)?;
    assert!(matches!(
        outcome,
        RuntimeCodeModeWaitOutcome::LiveCell(RuntimeCodeModeResponse::Terminated { .. })
    ));
    session.shutdown().await.map_err(anyhow::Error::msg)?;
    tokio::time::timeout(TEST_TIMEOUT, async {
        while !delegate.callback_cancelled.load(Ordering::Acquire) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .context("nested callback was not cancelled")?;
    host.stop().await;
    Ok(())
}
