use super::protocol::{
    ClientHello, ClientToHost, FramedReader, FramedWriter, HostHello, HostToClient,
    MAX_FRAME_BYTES, PROTOCOL_VERSION, SESSION_LIMITS_CAPABILITY,
};
use super::ProcessCodeModeSessionProvider;
use crate::code_mode::{
    NoopRuntimeCodeModeSessionDelegate, RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest,
    RuntimeCodeModeFuture, RuntimeCodeModeNestedToolCall, RuntimeCodeModeResponse,
    RuntimeCodeModeSessionDelegate, RuntimeCodeModeSessionHandle, RuntimeCodeModeSessionProvider,
    RuntimeCodeModeTool, RuntimeCodeModeWaitRequest,
};
use crate::tool_definition::RuntimeToolDefinition;
use crate::turn_snapshot::RuntimeToolIdentity;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;
use tokio::io::{duplex, AsyncWriteExt};
use tokio::sync::Notify;
use tokio::time::{timeout, Duration};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn length_prefixed_codec_round_trips_handshake() {
    let (client, server) = duplex(16 * 1024);
    let (client_read, client_write) = tokio::io::split(client);
    let (server_read, server_write) = tokio::io::split(server);
    let client_task = tokio::spawn(async move {
        let mut writer = FramedWriter::new(client_write);
        let mut reader = FramedReader::new(client_read);
        writer
            .write(&ClientToHost::ClientHello(ClientHello {
                supported_versions: vec![PROTOCOL_VERSION],
                required_capabilities: vec![SESSION_LIMITS_CAPABILITY.to_string()],
                optional_capabilities: Vec::new(),
            }))
            .await
            .expect("write client hello");
        reader
            .read::<HostToClient>()
            .await
            .expect("read host hello")
            .expect("host hello frame")
    });
    let server_task = tokio::spawn(async move {
        let mut reader = FramedReader::new(server_read);
        let mut writer = FramedWriter::new(server_write);
        let hello = reader
            .read::<ClientToHost>()
            .await
            .expect("read client hello")
            .expect("client hello frame");
        assert!(matches!(hello, ClientToHost::ClientHello(_)));
        writer
            .write(&HostToClient::HostHello(HostHello {
                selected_version: PROTOCOL_VERSION,
                capabilities: vec![SESSION_LIMITS_CAPABILITY.to_string()],
                host_pid: 42,
            }))
            .await
            .expect("write host hello");
    });

    server_task.await.expect("server codec task");
    assert_eq!(
        client_task.await.expect("client codec task"),
        HostToClient::HostHello(HostHello {
            selected_version: PROTOCOL_VERSION,
            capabilities: vec![SESSION_LIMITS_CAPABILITY.to_string()],
            host_pid: 42,
        })
    );
}

#[tokio::test]
async fn length_prefixed_codec_rejects_oversized_frame_before_allocation() {
    let (mut writer, reader) = duplex(16);
    writer
        .write_all(&u32::try_from(MAX_FRAME_BYTES + 1).unwrap().to_le_bytes())
        .await
        .expect("write oversized frame header");
    let error = FramedReader::new(reader)
        .read::<ClientToHost>()
        .await
        .expect_err("oversized frame must fail");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("exceeds"));
}

#[test]
fn process_provider_fails_closed_when_host_is_missing() {
    let temp = TempDir::new().expect("missing host temp dir");
    let host_path = temp.path().join("code-mode-host-missing");
    let provider = ProcessCodeModeSessionProvider::with_host_path(host_path.clone());
    assert_eq!(
        provider.availability().expect_err("missing host must fail"),
        format!(
            "code mode host executable was not found: {}",
            host_path.display()
        )
    );
}

#[tokio::test]
async fn process_host_executes_in_a_distinct_process_when_test_binary_is_available() {
    let provider = process_provider();
    let delegate: Arc<dyn RuntimeCodeModeSessionDelegate> =
        Arc::new(NoopRuntimeCodeModeSessionDelegate);
    let session = provider
        .create_session(delegate)
        .await
        .expect("open process-owned session");
    let started = session
        .execute(RuntimeCodeModeExecuteRequest {
            tool_call_id: "process-cell".to_string(),
            source: "text('process-ok')".to_string(),
            enabled_tools: Vec::new(),
            yield_time_ms: Some(1_000),
            max_output_tokens: None,
            cancellation_token: None,
        })
        .await
        .expect("start process cell");
    let response = started.initial_response().await.expect("process response");
    assert!(response.is_terminal());
    assert!(response.into_tool_result().output.contains("process-ok"));

    let missing = session
        .wait(RuntimeCodeModeWaitRequest {
            cell_id: crate::code_mode::RuntimeCodeModeCellId::new("missing-cell"),
            yield_time_ms: 10,
        })
        .await
        .expect("missing cell response");
    assert!(missing.into_response().is_terminal());
    session.shutdown().await.expect("shutdown process session");
}

#[derive(Default)]
struct RecordingDelegate {
    calls: Mutex<Vec<RuntimeCodeModeNestedToolCall>>,
    notifications: Mutex<Vec<String>>,
    closed_cells: Mutex<Vec<RuntimeCodeModeCellId>>,
    closed: Notify,
}

impl RuntimeCodeModeSessionDelegate for RecordingDelegate {
    fn invoke_tool<'a>(
        &'a self,
        invocation: RuntimeCodeModeNestedToolCall,
        _cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, Value> {
        Box::pin(async move {
            self.calls.lock().expect("process calls").push(invocation);
            Ok(json!({ "answer": 42 }))
        })
    }

    fn notify<'a>(
        &'a self,
        _tool_call_id: String,
        _cell_id: RuntimeCodeModeCellId,
        text: String,
        _cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, ()> {
        Box::pin(async move {
            self.notifications
                .lock()
                .expect("process notifications")
                .push(text);
            Ok(())
        })
    }

    fn cell_closed(&self, cell_id: &RuntimeCodeModeCellId) {
        self.closed_cells
            .lock()
            .expect("process closed cells")
            .push(cell_id.clone());
        self.closed.notify_waiters();
    }
}

#[tokio::test]
async fn process_host_routes_nested_tools_notifications_and_cell_close_in_order() {
    let provider = process_provider();
    let delegate = Arc::new(RecordingDelegate::default());
    let session = process_session(&provider, delegate.clone()).await;
    let mut request = process_request(
        "const result = await tools.lookup({ value: 41 }); notify('working'); text(result.answer);",
    );
    request.enabled_tools.push(RuntimeCodeModeTool {
        identity: RuntimeToolIdentity::plain("lookup"),
        definition: RuntimeToolDefinition::new(
            "lookup",
            "Returns an answer.",
            json!({ "type": "object" }),
        ),
        code_name: "lookup".to_string(),
        global_name: "lookup".to_string(),
    });
    let started = session
        .execute(request)
        .await
        .expect("start nested process cell");
    let cell_id = started.cell_id.clone();
    let response = started
        .initial_response()
        .await
        .expect("nested process response");
    assert!(matches!(
        response,
        RuntimeCodeModeResponse::Result { output, error_text: None, .. } if output == "42"
    ));
    wait_for_cell_closed(&delegate, &cell_id).await;

    let calls = delegate.calls.lock().expect("process calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tool_name, "lookup");
    assert_eq!(calls[0].input, Some(json!({ "value": 41 })));
    assert_eq!(
        delegate
            .notifications
            .lock()
            .expect("process notifications")
            .as_slice(),
        ["working"]
    );
    session
        .shutdown()
        .await
        .expect("shutdown nested process session");
}

#[tokio::test]
async fn process_host_yields_waits_and_terminates_cells() {
    let provider = process_provider();
    let session = process_session(&provider, Arc::new(RecordingDelegate::default())).await;
    let mut delayed = process_request(
        "await new Promise(resolve => setTimeout(resolve, 25)); text('completed');",
    );
    delayed.yield_time_ms = Some(1);
    let yielded = session
        .execute(delayed)
        .await
        .expect("start delayed process cell")
        .initial_response()
        .await
        .expect("yield delayed process cell");
    let delayed_cell_id = yielded.cell_id().clone();
    assert!(matches!(yielded, RuntimeCodeModeResponse::Yielded { .. }));
    let waited = session
        .wait(RuntimeCodeModeWaitRequest {
            cell_id: delayed_cell_id,
            yield_time_ms: 1_000,
        })
        .await
        .expect("wait for delayed process cell")
        .into_response();
    assert!(matches!(
        waited,
        RuntimeCodeModeResponse::Result { output, error_text: None, .. } if output == "completed"
    ));

    let mut infinite = process_request("while (true) {}");
    infinite.yield_time_ms = Some(1);
    let running = session
        .execute(infinite)
        .await
        .expect("start infinite process cell");
    let infinite_cell_id = running.cell_id.clone();
    assert!(matches!(
        running
            .initial_response()
            .await
            .expect("yield infinite process cell"),
        RuntimeCodeModeResponse::Yielded { .. }
    ));
    assert!(matches!(
        session
            .terminate(infinite_cell_id)
            .await
            .expect("terminate process cell")
            .into_response(),
        RuntimeCodeModeResponse::Terminated { .. }
    ));
    session
        .shutdown()
        .await
        .expect("shutdown yielded process session");
}

fn process_provider() -> ProcessCodeModeSessionProvider {
    ProcessCodeModeSessionProvider::with_host_path(test_host_path().expect(
        "code-mode-host must be built before the standalone process test; run `cargo build -p tool-runtime --bin code-mode-host`",
    ))
}

async fn process_session(
    provider: &ProcessCodeModeSessionProvider,
    delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
) -> RuntimeCodeModeSessionHandle {
    provider
        .create_session(delegate)
        .await
        .expect("open process-owned session")
}

fn process_request(source: &str) -> RuntimeCodeModeExecuteRequest {
    RuntimeCodeModeExecuteRequest {
        tool_call_id: "process-call".to_string(),
        source: source.to_string(),
        enabled_tools: Vec::new(),
        yield_time_ms: Some(1_000),
        max_output_tokens: None,
        cancellation_token: None,
    }
}

async fn wait_for_cell_closed(delegate: &RecordingDelegate, cell_id: &RuntimeCodeModeCellId) {
    timeout(Duration::from_secs(5), async {
        loop {
            let notified = delegate.closed.notified();
            if delegate
                .closed_cells
                .lock()
                .expect("process closed cells")
                .contains(cell_id)
            {
                return;
            }
            notified.await;
        }
    })
    .await
    .expect("process host did not deliver cell_closed");
}

fn test_host_path() -> Option<PathBuf> {
    std::env::var_os("CARGO_BIN_EXE_code-mode-host")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            let binary_name = if cfg!(windows) {
                "code-mode-host.exe"
            } else {
                "code-mode-host"
            };
            std::env::current_exe()
                .ok()?
                .parent()?
                .parent()?
                .join(binary_name)
                .is_file()
                .then(|| {
                    std::env::current_exe()
                        .expect("current test binary")
                        .parent()
                        .expect("test deps directory")
                        .parent()
                        .expect("cargo target profile directory")
                        .join(binary_name)
                })
        })
}
