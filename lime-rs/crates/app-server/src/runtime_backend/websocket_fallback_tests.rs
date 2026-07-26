use super::*;
use crate::runtime::RuntimeHostContext;
use crate::{ExecutionBackend, RuntimeEventSink};
use app_server_protocol::{
    AgentSession, AgentSessionStatus, AgentTurn, AgentTurnStatus, RuntimeOptions,
    RuntimeProviderConfig, RuntimeRequest,
};
use lime_core::database::schema::create_tables;
use rusqlite::Connection;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};

#[derive(Default)]
struct RecordingSink {
    events: Vec<RuntimeEvent>,
}

impl RuntimeEventSink for RecordingSink {
    fn emit(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        self.events.push(event);
        Ok(())
    }
}

#[derive(Debug)]
struct CapturedRequest {
    method: String,
    path: String,
    body: Option<Value>,
}

struct ResponsesFallbackFixture {
    base_url: String,
    requests: Arc<Mutex<Vec<CapturedRequest>>>,
    server_task: JoinHandle<()>,
}

impl ResponsesFallbackFixture {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind Responses fallback fixture");
        let address = listener.local_addr().expect("fixture address");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let server_requests = Arc::clone(&requests);
        let server_task = tokio::spawn(async move {
            for index in 0..3 {
                let (mut stream, _) = listener.accept().await.expect("accept provider request");
                let request = read_request(&mut stream)
                    .await
                    .expect("read provider request");
                server_requests
                    .lock()
                    .expect("record provider request")
                    .push(request);
                if index == 0 {
                    write_response(
                        &mut stream,
                        "426 Upgrade Required",
                        "text/plain",
                        "unsupported",
                    )
                    .await;
                    continue;
                }
                let response_id = format!("resp-http-{index}");
                let body = format!(
                    "data: {}\n\ndata: {}\n\n",
                    json!({
                        "type": "response.output_text.delta",
                        "item_id": format!("message-{index}"),
                        "delta": format!("fallback turn {index}")
                    }),
                    json!({
                        "type": "response.completed",
                        "response": {
                            "id": response_id,
                            "output": [],
                            "usage": {
                                "input_tokens": 2,
                                "output_tokens": 1,
                                "total_tokens": 3
                            }
                        }
                    })
                );
                write_response(&mut stream, "200 OK", "text/event-stream", &body).await;
            }
        });
        Self {
            base_url: format!("http://{address}/v1"),
            requests,
            server_task,
        }
    }
}

impl Drop for ResponsesFallbackFixture {
    fn drop(&mut self) {
        self.server_task.abort();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn responses_websocket_fallback_is_sticky_across_runtime_turns() {
    let fixture = ResponsesFallbackFixture::start().await;
    let backend = RuntimeBackend::with_db(test_db());

    for index in 1..=2 {
        let mut sink = RecordingSink::default();
        timeout(
            Duration::from_secs(10),
            ExecutionBackend::start_turn(
                &backend,
                execution_request(&fixture.base_url, index),
                &mut sink,
            ),
        )
        .await
        .expect("runtime turn timeout")
        .expect("runtime turn");
        assert!(sink
            .events
            .iter()
            .any(|event| event.event_type == "turn.completed"));
    }

    timeout(Duration::from_secs(2), async {
        while !fixture.server_task.is_finished() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("provider fixture completion");
    let requests = fixture.requests.lock().expect("provider requests");
    assert_eq!(
        requests
            .iter()
            .map(|request| request.method.as_str())
            .collect::<Vec<_>>(),
        ["GET", "POST", "POST"]
    );
    assert!(requests
        .iter()
        .all(|request| request.path == "/v1/responses"));
    assert!(requests.iter().skip(1).all(|request| {
        request
            .body
            .as_ref()
            .and_then(|body| body.get("stream"))
            .and_then(Value::as_bool)
            == Some(true)
    }));
}

fn execution_request(base_url: &str, turn_index: usize) -> ExecutionRequest {
    let session_id = "session-responses-websocket-fallback".to_string();
    let thread_id = "thread-responses-websocket-fallback".to_string();
    let turn_id = format!("turn-responses-websocket-fallback-{turn_index}");
    ExecutionRequest {
        host: RuntimeHostContext::default(),
        session: AgentSession {
            session_id: session_id.clone(),
            thread_id: thread_id.clone(),
            app_id: "agent".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: AgentSessionStatus::Running,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        },
        turn: AgentTurn {
            turn_id,
            session_id,
            thread_id,
            status: AgentTurnStatus::Accepted,
            started_at: None,
            completed_at: None,
        },
        forked_from_thread_id: None,
        input: agent_runtime::reply_input::RuntimeReplyInput::text(format!("turn {turn_index}")),
        runtime_options: Some(RuntimeOptions {
            stream: true,
            runtime_request: Some(RuntimeRequest {
                provider_config: Some(RuntimeProviderConfig {
                    provider_id: Some("fixture-openai-responses".to_string()),
                    provider_name: Some("openai-responses".to_string()),
                    model_name: Some("fixture-responses-model".to_string()),
                    api_key: Some("fixture-key".to_string()),
                    base_url: Some(base_url.to_string()),
                    model_capabilities: Some(json!({
                        "capabilities": {
                            "streaming": true,
                            "tools": true,
                            "functionCalling": true
                        },
                        "taskFamilies": ["chat"],
                        "inputModalities": ["text"],
                        "outputModalities": ["text"],
                        "runtimeFeatures": ["streaming", "tool_calling", "responses_api"]
                    })),
                    supports_websockets: Some(true),
                    ..RuntimeProviderConfig::default()
                }),
                ..RuntimeRequest::default()
            }),
            ..RuntimeOptions::default()
        }),
        event_name: None,
        expected_output: None,
        structured_output: None,
        output_schema: None,
        queued_turn_id: None,
        queue_if_busy: false,
        skip_pre_submit_resume: false,
        agent_control_gateway: None,
    }
}

fn test_db() -> DbConnection {
    let connection = Connection::open_in_memory().expect("open in-memory db");
    create_tables(&connection).expect("create schema");
    Arc::new(Mutex::new(connection))
}

async fn read_request(stream: &mut TcpStream) -> std::io::Result<CapturedRequest> {
    let mut buffer = Vec::new();
    let header_end = loop {
        let mut chunk = [0_u8; 1024];
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(header_end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break header_end;
        }
    };
    let headers = String::from_utf8_lossy(&buffer[..header_end]);
    let request_line = headers.lines().next().unwrap_or_default();
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().unwrap_or_default().to_string();
    let path = request_parts.next().unwrap_or_default().to_string();
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    let body_start = header_end + 4;
    while buffer.len().saturating_sub(body_start) < content_length {
        let mut chunk = vec![0_u8; content_length - (buffer.len() - body_start)];
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    let body = if content_length == 0 {
        None
    } else {
        Some(
            serde_json::from_slice(&buffer[body_start..body_start + content_length])
                .map_err(std::io::Error::other)?,
        )
    };
    Ok(CapturedRequest { method, path, body })
}

async fn write_response(stream: &mut TcpStream, status: &str, content_type: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .await
        .expect("write provider response");
    stream.shutdown().await.expect("close provider response");
}
