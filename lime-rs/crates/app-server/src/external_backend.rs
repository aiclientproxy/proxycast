use crate::ActionRespondRequest;
use crate::CancelExecutionRequest;
use crate::ExecutionBackend;
use crate::ExecutionRequest;
use crate::ProviderTurnHistory;
use crate::RuntimeCoreError;
use crate::RuntimeEvent;
use crate::RuntimeEventSink;
use crate::RuntimeHostContext;
use agent_runtime::session_loop::RuntimeSessionInputHandle;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::process::Child;
use tokio::process::Command;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

pub const DEFAULT_EXTERNAL_BACKEND_TIMEOUT_MS: u64 = 30_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalBackendConfig {
    pub command: String,
    pub args: Vec<String>,
    pub timeout_ms: u64,
}

impl ExternalBackendConfig {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            timeout_ms: DEFAULT_EXTERNAL_BACKEND_TIMEOUT_MS,
        }
    }

    pub fn with_args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }
}

#[derive(Clone)]
pub struct ExternalBackend {
    config: ExternalBackendConfig,
}

impl ExternalBackend {
    pub fn new(config: ExternalBackendConfig) -> Self {
        Self { config }
    }

    async fn invoke(
        &self,
        kind: &str,
        request: serde_json::Value,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.invoke_with_cancellation(kind, request, sink, None)
            .await
    }

    async fn invoke_with_cancellation(
        &self,
        kind: &str,
        request: serde_json::Value,
        sink: &mut dyn RuntimeEventSink,
        cancellation_token: Option<CancellationToken>,
    ) -> Result<(), RuntimeCoreError> {
        invoke_external_backend(&self.config, kind, request, sink, cancellation_token).await
    }
}

#[async_trait]
impl ExecutionBackend for ExternalBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.invoke("turnStart", start_turn_request_value(request), sink)
            .await
    }

    async fn start_turn_with_provider_history_and_session_input(
        &self,
        request: ExecutionRequest,
        _provider_history: ProviderTurnHistory,
        _pending_input: Option<RuntimeSessionInputHandle>,
        cancellation_token: Option<CancellationToken>,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.invoke_with_cancellation(
            "turnStart",
            start_turn_request_value(request),
            sink,
            cancellation_token,
        )
        .await
    }

    async fn cancel_turn(
        &self,
        request: CancelExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.invoke("turnCancel", cancel_turn_request_value(request), sink)
            .await
    }

    async fn respond_action(
        &self,
        request: ActionRespondRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.invoke("actionRespond", action_respond_request_value(request), sink)
            .await
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExternalBackendResponse {
    #[serde(default)]
    events: Vec<ExternalBackendEvent>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExternalBackendEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    payload: serde_json::Value,
}

async fn invoke_external_backend(
    config: &ExternalBackendConfig,
    kind: &str,
    request: serde_json::Value,
    sink: &mut dyn RuntimeEventSink,
    cancellation_token: Option<CancellationToken>,
) -> Result<(), RuntimeCoreError> {
    if config.command.trim().is_empty() {
        return Err(RuntimeCoreError::Backend(
            "external app-server backend command is not configured".to_string(),
        ));
    }

    let input = json!({
        "kind": kind,
        "request": request,
    });

    let mut child = Command::new(&config.command)
        .args(&config.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            RuntimeCoreError::Backend(format!(
                "failed to spawn external app-server backend: {error}"
            ))
        })?;

    if let Some(mut stdin) = child.stdin.take() {
        let input = serde_json::to_vec(&input).map_err(|error| {
            RuntimeCoreError::Backend(format!(
                "failed to encode external app-server backend request: {error}"
            ))
        })?;
        stdin.write_all(&input).await.map_err(|error| {
            RuntimeCoreError::Backend(format!(
                "failed to write external app-server backend request: {error}"
            ))
        })?;
        stdin.write_all(b"\n").await.map_err(|error| {
            RuntimeCoreError::Backend(format!(
                "failed to finish external app-server backend request: {error}"
            ))
        })?;
    }

    let stdout = child.stdout.take().ok_or_else(|| {
        RuntimeCoreError::Backend("external app-server backend stdout is not available".to_string())
    })?;
    let stderr = child.stderr.take().ok_or_else(|| {
        RuntimeCoreError::Backend("external app-server backend stderr is not available".to_string())
    })?;
    let mut stdout_lines = BufReader::new(stdout).lines();
    let stderr_task = tokio::spawn(async move {
        let mut stderr = BufReader::new(stderr);
        let mut buffer = Vec::new();
        stderr.read_to_end(&mut buffer).await.map(|_| buffer)
    });

    let mut line_count = 0usize;
    loop {
        let next_line = if let Some(cancellation_token) = cancellation_token.as_ref() {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                    let _ = stderr_task.await;
                    return Ok(());
                }
                result = timeout(
                    Duration::from_millis(config.timeout_ms),
                    stdout_lines.next_line(),
                ) => match result {
                    Ok(result) => result.map_err(|error| {
                        RuntimeCoreError::Backend(format!(
                            "failed to read external app-server backend response: {error}"
                        ))
                    })?,
                    Err(_) => {
                        return Err(cleanup_external_backend_after_timeout(
                            &mut child,
                            stderr_task,
                            config.timeout_ms,
                            "reading stdout",
                        )
                        .await);
                    }
                },
            }
        } else {
            match timeout(
                Duration::from_millis(config.timeout_ms),
                stdout_lines.next_line(),
            )
            .await
            {
                Ok(result) => result.map_err(|error| {
                    RuntimeCoreError::Backend(format!(
                        "failed to read external app-server backend response: {error}"
                    ))
                })?,
                Err(_) => {
                    return Err(cleanup_external_backend_after_timeout(
                        &mut child,
                        stderr_task,
                        config.timeout_ms,
                        "reading stdout",
                    )
                    .await);
                }
            }
        };
        let Some(line) = next_line else {
            break;
        };
        if line.trim().is_empty() {
            continue;
        }
        line_count += 1;
        emit_external_backend_line(&line, sink)?;
    }

    let status = if let Some(cancellation_token) = cancellation_token.as_ref() {
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                let _ = stderr_task.await;
                return Ok(());
            }
            result = timeout(Duration::from_millis(config.timeout_ms), child.wait()) => match result {
                Ok(result) => result.map_err(|error| {
                    RuntimeCoreError::Backend(format!(
                        "failed to wait for external app-server backend: {error}"
                    ))
                })?,
                Err(_) => {
                    return Err(cleanup_external_backend_after_timeout(
                        &mut child,
                        stderr_task,
                        config.timeout_ms,
                        "waiting for exit",
                    )
                    .await);
                }
            },
        }
    } else {
        match timeout(Duration::from_millis(config.timeout_ms), child.wait()).await {
            Ok(result) => result.map_err(|error| {
                RuntimeCoreError::Backend(format!(
                    "failed to wait for external app-server backend: {error}"
                ))
            })?,
            Err(_) => {
                return Err(cleanup_external_backend_after_timeout(
                    &mut child,
                    stderr_task,
                    config.timeout_ms,
                    "waiting for exit",
                )
                .await);
            }
        }
    };

    let stderr = stderr_task
        .await
        .map_err(|error| {
            RuntimeCoreError::Backend(format!(
                "failed to join external app-server backend stderr reader: {error}"
            ))
        })?
        .map_err(|error| {
            RuntimeCoreError::Backend(format!(
                "failed to read external app-server backend stderr: {error}"
            ))
        })?;

    if !status.success() {
        return Err(RuntimeCoreError::Backend(format!(
            "external app-server backend exited with status {}: {}",
            status,
            String::from_utf8_lossy(&stderr).trim()
        )));
    }

    if line_count == 0 {
        return Err(RuntimeCoreError::Backend(
            "external app-server backend produced no response".to_string(),
        ));
    }

    Ok(())
}

async fn cleanup_external_backend_after_timeout(
    child: &mut Child,
    stderr_task: JoinHandle<std::io::Result<Vec<u8>>>,
    timeout_ms: u64,
    phase: &str,
) -> RuntimeCoreError {
    let _ = child.start_kill();
    let _ = child.wait().await;
    RuntimeCoreError::Backend(format!(
        "external app-server backend timed out after {timeout_ms}ms while {phase}: {}",
        external_backend_stderr_summary(stderr_task).await
    ))
}

async fn external_backend_stderr_summary(
    stderr_task: JoinHandle<std::io::Result<Vec<u8>>>,
) -> String {
    match stderr_task.await {
        Ok(Ok(stderr)) => {
            let stderr = String::from_utf8_lossy(&stderr).trim().to_string();
            if stderr.is_empty() {
                "stderr was empty".to_string()
            } else {
                stderr
            }
        }
        Ok(Err(error)) => format!("failed to read stderr: {error}"),
        Err(error) => format!("failed to join stderr reader: {error}"),
    }
}

fn emit_external_backend_line(
    line: &str,
    sink: &mut dyn RuntimeEventSink,
) -> Result<(), RuntimeCoreError> {
    let value = serde_json::from_str::<serde_json::Value>(line).map_err(|error| {
        RuntimeCoreError::Backend(format!(
            "failed to decode external app-server backend response line: {error}"
        ))
    })?;

    if value.get("events").is_some() {
        let response: ExternalBackendResponse = serde_json::from_value(value).map_err(|error| {
            RuntimeCoreError::Backend(format!(
                "failed to decode external app-server backend response line: {error}"
            ))
        })?;
        for event in response.events {
            sink.emit(RuntimeEvent::new(event.event_type, event.payload))?;
        }
        return Ok(());
    }

    let event: ExternalBackendEvent = serde_json::from_value(value).map_err(|error| {
        RuntimeCoreError::Backend(format!(
            "failed to decode external app-server backend response line: {error}"
        ))
    })?;
    sink.emit(RuntimeEvent::new(event.event_type, event.payload))
}

fn host_value(host: RuntimeHostContext) -> serde_json::Value {
    json!({
        "clientName": host.client_name,
        "clientVersion": host.client_version,
    })
}

fn start_turn_request_value(request: ExecutionRequest) -> serde_json::Value {
    let provider_preference = request.provider_preference().map(str::to_string);
    let model_preference = request.model_preference().map(str::to_string);
    let metadata = request.runtime_metadata().cloned();
    json!({
        "host": host_value(request.host),
        "session": request.session,
        "turn": request.turn,
        "input": request.input,
        "runtimeOptions": request.runtime_options,
        "eventName": request.event_name,
        "providerPreference": provider_preference,
        "modelPreference": model_preference,
        "metadata": metadata,
        "queuedTurnId": request.queued_turn_id,
        "queueIfBusy": request.queue_if_busy,
        "skipPreSubmitResume": request.skip_pre_submit_resume,
    })
}

fn cancel_turn_request_value(request: CancelExecutionRequest) -> serde_json::Value {
    json!({
        "host": host_value(request.host),
        "session": request.session,
        "turn": request.turn,
    })
}

fn action_respond_request_value(request: ActionRespondRequest) -> serde_json::Value {
    let decision = request.decision.map(|decision| decision.as_str());
    let decision_scope = request.decision.map(|decision| decision.scope());
    let metadata = request.runtime_metadata().cloned();
    json!({
        "host": host_value(request.host),
        "session": request.session,
        "turn": request.turn,
        "requestId": request.request_id,
        "actionType": request.action_type,
        "decision": decision,
        "decisionScope": decision_scope,
        "confirmed": request.confirmed,
        "response": request.response,
        "userData": request.user_data,
        "metadata": metadata,
        "eventName": request.event_name,
        "actionScope": request.action_scope,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RuntimeCore;
    use app_server_protocol::AgentSessionStartParams;
    use app_server_protocol::AgentSessionTurnStartParams;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;

    #[tokio::test]
    async fn external_backend_invokes_process_and_maps_runtime_events() {
        let Some(node) = node_binary() else {
            return;
        };
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let script_path = temp_dir.path().join("external-backend.mjs");
        fs::write(
            &script_path,
            r#"
              import { readFileSync } from 'node:fs';
              const input = JSON.parse(readFileSync(0, 'utf8'));
              const text = (input.request.input?.parts ?? [])
                .map((part) => part.Text?.text ?? '')
                .join('');
              console.log(JSON.stringify({
                events: [
                  {
                    type: 'message.delta',
                    payload: {
                      backend: 'external',
                      kind: input.kind,
                      inputTextLength: text.length
                    }
                  }
                ]
              }));
            "#,
        )
        .expect("write backend script");

        let core = RuntimeCore::with_backend(Arc::new(ExternalBackend::new(
            ExternalBackendConfig::new(node).with_args([script_path.to_string_lossy().to_string()]),
        )));
        core.start_session(AgentSessionStartParams {
            session_id: Some("sess_external".to_string()),
            thread_id: Some("thread_external".to_string()),
            app_id: "content-studio".to_string(),
            workspace_id: None,
            business_object_ref: None,
            locale: None,
        })
        .expect("session");

        let output = core
            .start_turn(
                AgentSessionTurnStartParams {
                    session_id: "sess_external".to_string(),
                    turn_id: Some("turn_external".to_string()),
                    input: app_server_protocol::AgentInput {
                        text: "draft".to_string(),
                        attachments: Vec::new(),
                    },
                    runtime_options: None,
                    queue_if_busy: false,
                    skip_pre_submit_resume: false,
                },
                RuntimeHostContext {
                    client_name: Some("content-studio".to_string()),
                    client_version: Some("0.1.0".to_string()),
                },
            )
            .await
            .expect("turn");

        assert_eq!(
            output
                .events
                .iter()
                .map(|event| event.event_type.as_str())
                .collect::<Vec<_>>(),
            vec![
                "item.started",
                "message.created",
                "item.completed",
                "turn.accepted",
                "item.started",
                "message.delta",
            ]
        );
        assert_eq!(output.events[1].payload["input"][0]["type"], "text");
        assert_eq!(output.events[1].payload["input"][0]["text"], "draft");
        assert_eq!(output.events[5].payload["backend"], "external");
        assert_eq!(output.events[5].payload["kind"], "turnStart");
        assert_eq!(output.events[5].payload["inputTextLength"], 5);
    }

    #[tokio::test]
    async fn external_backend_reads_jsonl_event_stream() {
        let Some(node) = node_binary() else {
            return;
        };
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let script_path = temp_dir.path().join("external-backend-stream.mjs");
        fs::write(
            &script_path,
            r#"
              console.log(JSON.stringify({
                type: 'message.delta',
                payload: { chunk: 1, text: 'hello' }
              }));
              console.log(JSON.stringify({
                type: 'message.delta',
                payload: { chunk: 2, text: 'world' }
              }));
              console.log(JSON.stringify({
                events: [
                  {
                    type: 'artifact.snapshot',
                    payload: {
                      artifactId: 'stream-artifact',
                      title: 'Stream Artifact'
                    }
                  }
                ]
              }));
            "#,
        )
        .expect("write backend script");

        let output = start_external_test_turn(
            ExternalBackendConfig::new(node).with_args([script_path.to_string_lossy().to_string()]),
            "sess_external_stream",
            "turn_external_stream",
        )
        .await;

        assert_eq!(
            output
                .events
                .iter()
                .map(|event| event.event_type.as_str())
                .collect::<Vec<_>>(),
            vec![
                "item.started",
                "message.created",
                "item.completed",
                "turn.accepted",
                "item.started",
                "message.delta",
                "message.delta",
                "artifact.snapshot",
            ]
        );
        assert_eq!(output.events[5].payload["chunk"], 1);
        assert_eq!(output.events[6].payload["chunk"], 2);
        assert_eq!(output.events[7].payload["artifactId"], "stream-artifact");
    }

    #[tokio::test]
    async fn external_backend_timeout_kills_process_and_reports_stderr_while_reading_stdout() {
        let Some(node) = node_binary() else {
            return;
        };
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let script_path = temp_dir.path().join("external-backend-timeout-stdout.mjs");
        fs::write(
            &script_path,
            r#"
              console.error('backend still starting');
              setTimeout(() => {}, 10_000);
            "#,
        )
        .expect("write backend script");

        let error = start_external_test_turn_error(
            ExternalBackendConfig::new(node)
                .with_args([script_path.to_string_lossy().to_string()])
                .with_timeout_ms(500),
            "sess_external_timeout_stdout",
            "turn_external_timeout_stdout",
        )
        .await;

        assert!(error.contains("timed out after 500ms while reading stdout"));
    }

    #[tokio::test]
    async fn external_backend_timeout_kills_process_and_reports_stderr_while_waiting_for_exit() {
        let Some(node) = node_binary() else {
            return;
        };
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let script_path = temp_dir.path().join("external-backend-timeout-exit.mjs");
        fs::write(
            &script_path,
            r#"
              console.log(JSON.stringify({
                type: 'message.delta',
                payload: { text: 'before hang' }
              }));
              await new Promise((resolve) => process.stdout.write('', resolve));
              await new Promise((resolve) => process.stdout.end(resolve));
              console.error('backend cleanup hung');
              setTimeout(() => {}, 10_000);
            "#,
        )
        .expect("write backend script");

        let error = start_external_test_turn_error(
            ExternalBackendConfig::new(node)
                .with_args([script_path.to_string_lossy().to_string()])
                .with_timeout_ms(500),
            "sess_external_timeout_exit",
            "turn_external_timeout_exit",
        )
        .await;

        assert!(error.contains("timed out after"), "{error}");
    }

    #[tokio::test]
    async fn external_backend_cancellation_kills_hanging_process() {
        let Some(node) = node_binary() else {
            return;
        };
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let script_path = temp_dir.path().join("external-backend-cancel.mjs");
        fs::write(
            &script_path,
            r#"
              setInterval(() => {}, 1_000);
              await new Promise(() => {});
            "#,
        )
        .expect("write backend script");

        let cancellation = CancellationToken::new();
        let cancel_after_delay = cancellation.clone();
        let cancel_task = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            cancel_after_delay.cancel();
        });
        let mut sink = RecordingRuntimeEventSink;
        let result = invoke_external_backend(
            &ExternalBackendConfig::new(node)
                .with_args([script_path.to_string_lossy().to_string()])
                .with_timeout_ms(5_000),
            "turnStart",
            json!({"request": "cancel"}),
            &mut sink,
            Some(cancellation),
        )
        .await;

        cancel_task.await.expect("cancel task");
        assert!(result.is_ok(), "canceled backend should complete cleanly");
    }

    #[test]
    fn external_backend_config_keeps_command_and_args_separate() {
        let config = ExternalBackendConfig::new("/bin/backend")
            .with_args(["--mode", "agent"])
            .with_timeout_ms(42);

        assert_eq!(config.command, "/bin/backend");
        assert_eq!(config.args, vec!["--mode".to_string(), "agent".to_string()]);
        assert_eq!(config.timeout_ms, 42);
    }

    struct RecordingRuntimeEventSink;

    impl RuntimeEventSink for RecordingRuntimeEventSink {
        fn emit(&mut self, _event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
            Ok(())
        }
    }

    fn node_binary() -> Option<String> {
        let candidates = std::env::var("NODE")
            .ok()
            .into_iter()
            .chain(["node".to_string()]);
        for candidate in candidates {
            if std::process::Command::new(&candidate)
                .arg("--version")
                .output()
                .is_ok_and(|output| output.status.success())
            {
                return Some(candidate);
            }
        }
        None
    }

    async fn start_external_test_turn(
        config: ExternalBackendConfig,
        session_id: &str,
        turn_id: &str,
    ) -> crate::RuntimeCoreOutput<app_server_protocol::AgentSessionTurnStartResponse> {
        let core = RuntimeCore::with_backend(Arc::new(ExternalBackend::new(config)));
        core.start_session(AgentSessionStartParams {
            session_id: Some(session_id.to_string()),
            thread_id: Some(format!("thread_{session_id}")),
            app_id: "content-studio".to_string(),
            workspace_id: None,
            business_object_ref: None,
            locale: None,
        })
        .expect("session");

        core.start_turn(
            AgentSessionTurnStartParams {
                session_id: session_id.to_string(),
                turn_id: Some(turn_id.to_string()),
                input: app_server_protocol::AgentInput {
                    text: "draft".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext {
                client_name: Some("content-studio".to_string()),
                client_version: Some("0.1.0".to_string()),
            },
        )
        .await
        .expect("turn")
    }

    async fn start_external_test_turn_error(
        config: ExternalBackendConfig,
        session_id: &str,
        turn_id: &str,
    ) -> String {
        let core = RuntimeCore::with_backend(Arc::new(ExternalBackend::new(config)));
        core.start_session(AgentSessionStartParams {
            session_id: Some(session_id.to_string()),
            thread_id: Some(format!("thread_{session_id}")),
            app_id: "content-studio".to_string(),
            workspace_id: None,
            business_object_ref: None,
            locale: None,
        })
        .expect("session");

        let error = core
            .start_turn(
                AgentSessionTurnStartParams {
                    session_id: session_id.to_string(),
                    turn_id: Some(turn_id.to_string()),
                    input: app_server_protocol::AgentInput {
                        text: "draft".to_string(),
                        attachments: Vec::new(),
                    },
                    runtime_options: None,
                    queue_if_busy: false,
                    skip_pre_submit_resume: false,
                },
                RuntimeHostContext {
                    client_name: Some("content-studio".to_string()),
                    client_version: Some("0.1.0".to_string()),
                },
            )
            .await
            .expect_err("external backend should fail");

        match error {
            RuntimeCoreError::Backend(message) => message,
            other => panic!("unexpected runtime error: {other:?}"),
        }
    }

    #[allow(dead_code)]
    fn _assert_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ExternalBackend>();
        assert_send_sync::<ExternalBackendConfig>();
        let _ = PathBuf::new();
    }
}
