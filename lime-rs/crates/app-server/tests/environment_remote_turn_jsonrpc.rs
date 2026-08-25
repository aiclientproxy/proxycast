mod support;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use app_server::{run_json_lines, AppServer, AppServerError, EventLogWriter, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_ENVIRONMENT_ADD, METHOD_ENVIRONMENT_STATUS, METHOD_TURN_COMPLETED,
};
use app_server_protocol::{
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_READ, METHOD_THREAD_RESUME,
    METHOD_THREAD_START, METHOD_TURN_START,
};
use base64::Engine;
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

const PROVIDER_ID: &str = "remote-turn-provider";
const MODEL_ID: &str = "remote-turn-model";
const ENVIRONMENT_ID: &str = "remote-turn-environment";
const REMOTE_CWD: &str = "/remote/workspace";
const REMOTE_OUTPUT: &str = "remote-process-output";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn remote_environment_executes_provider_tool_and_survives_cold_resume() {
    let temp = TempDir::new().expect("remote turn temp dir");
    let provider = ProviderFixture::start().await;
    let remote = RemoteExecFixture::start().await;

    let mut first = TransportClient::start(
        AppServer::with_runtime(runtime(&temp, &provider.base_url)),
        "environment-remote-turn-first",
    )
    .await;
    first
        .request_ok(
            2,
            METHOD_ENVIRONMENT_ADD,
            json!({
                "environmentId": ENVIRONMENT_ID,
                "execServerUrl": remote.url,
            }),
        )
        .await;
    wait_for_environment_ready(&mut first, 10).await;

    let started = first
        .request_ok(
            30,
            METHOD_THREAD_START,
            json!({
                "model": MODEL_ID,
                "modelProvider": PROVIDER_ID,
                "cwd": temp.path(),
                "historyMode": "paginated",
                "environments": [{
                    "environmentId": ENVIRONMENT_ID,
                    "cwd": REMOTE_CWD,
                    "runtimeWorkspaceRoots": [REMOTE_CWD]
                }]
            }),
        )
        .await;
    let thread_id = required_string(&started, "/result/thread/id", "thread/start id");

    let first_turn_id = start_turn(&mut first, 31, &thread_id, "first remote command").await;
    first
        .wait_for_turn_completed(&thread_id, &first_turn_id, &remote, &provider)
        .await;
    let first_read = first
        .request_ok(
            32,
            METHOD_THREAD_READ,
            json!({"threadId": thread_id, "includeTurns": true}),
        )
        .await;
    assert_remote_command_item(&first_read, 1, &provider.requests(), &remote.requests());
    first.shutdown().await;

    let mut resumed = TransportClient::start(
        AppServer::with_runtime(runtime(&temp, &provider.base_url)),
        "environment-remote-turn-resumed",
    )
    .await;
    wait_for_environment_ready(&mut resumed, 40).await;
    let resume = resumed
        .request_ok(
            60,
            METHOD_THREAD_RESUME,
            json!({"threadId": thread_id, "excludeTurns": true}),
        )
        .await;
    assert_eq!(
        resume.pointer("/result/thread/id").and_then(Value::as_str),
        Some(thread_id.as_str())
    );

    let second_turn_id = start_turn(&mut resumed, 61, &thread_id, "second remote command").await;
    resumed
        .wait_for_turn_completed(&thread_id, &second_turn_id, &remote, &provider)
        .await;
    let second_read = resumed
        .request_ok(
            62,
            METHOD_THREAD_READ,
            json!({"threadId": thread_id, "includeTurns": true}),
        )
        .await;
    assert_remote_command_item(&second_read, 2, &provider.requests(), &remote.requests());

    let remote_requests = remote.requests();
    let starts = remote_requests
        .iter()
        .filter(|request| request["method"] == "process/start")
        .collect::<Vec<_>>();
    assert_eq!(starts.len(), 2, "each turn must start one remote process");
    for start in starts {
        assert_eq!(
            start.pointer("/params/cwd"),
            Some(&json!("file:///remote/workspace"))
        );
        assert!(start
            .pointer("/params/argv")
            .and_then(Value::as_array)
            .is_some_and(|argv| argv.iter().any(|part| {
                part.as_str()
                    .is_some_and(|part| part.contains("local-command-must-not-run"))
            })));
    }
    assert!(
        remote_requests
            .iter()
            .filter(|request| request["method"] == "initialize")
            .count()
            >= 2,
        "cold App Server must hydrate and reconnect the persisted Environment registry"
    );
    assert!(
        provider
            .requests()
            .iter()
            .filter(|request| request["stream"] == true)
            .count()
            >= 4,
        "two turns must each sample before and after the remote tool result"
    );

    resumed.shutdown().await;
}

fn runtime(temp: &TempDir, provider_base_url: &str) -> RuntimeCore {
    support::runtime_core_with_chat_provider_at(temp, PROVIDER_ID, MODEL_ID, provider_base_url)
        .with_event_log_writer(Arc::new(
            EventLogWriter::new(temp.path().join("event-log")).expect("event log writer"),
        ))
        .with_app_config_path(temp.path().join("config").join("config.yaml"))
}

async fn start_turn(
    client: &mut TransportClient,
    id: u64,
    thread_id: &str,
    prompt: &str,
) -> String {
    let response = client
        .request_ok(
            id,
            METHOD_TURN_START,
            json!({
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt}],
                "model": MODEL_ID,
                "approvalPolicy": "never",
                "sandboxPolicy": "workspace-write"
            }),
        )
        .await;
    required_string(&response, "/result/turn/id", "turn/start id")
}

async fn wait_for_environment_ready(client: &mut TransportClient, first_id: u64) {
    for id in first_id..first_id + 20 {
        let status = client
            .request_ok(
                id,
                METHOD_ENVIRONMENT_STATUS,
                json!({"environmentId": ENVIRONMENT_ID}),
            )
            .await;
        if status.pointer("/result/status") == Some(&json!("ready")) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("remote Environment did not become ready");
}

fn assert_remote_command_item(
    read: &Value,
    expected_turns: usize,
    provider_requests: &[Value],
    remote_requests: &[Value],
) {
    let turns = read
        .pointer("/result/thread/turns")
        .and_then(Value::as_array)
        .expect("thread/read turns");
    assert_eq!(turns.len(), expected_turns);
    let command = turns
        .iter()
        .flat_map(|turn| {
            turn.get("items")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .rev()
        .find(|item| item["type"] == "commandExecution")
        .expect("remote commandExecution item");
    assert_eq!(
        command["status"],
        "completed",
        "remote command projection: {command:#}; provider tool results: {:?}; remote methods: {:?}",
        provider_tool_results(provider_requests),
        remote_requests
            .iter()
            .filter_map(|request| request["method"].as_str())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        command["exitCode"], 0,
        "remote command projection: {command:#}"
    );
    assert_eq!(
        command["aggregatedOutput"], REMOTE_OUTPUT,
        "remote command projection: {command:#}"
    );
}

fn provider_tool_results(requests: &[Value]) -> Vec<Value> {
    requests
        .iter()
        .flat_map(|request| request["messages"].as_array())
        .flatten()
        .filter(|message| message["role"] == "tool")
        .map(|message| {
            json!({
                "tool_call_id": message["tool_call_id"].clone(),
                "content": message["content"].clone()
            })
        })
        .collect()
}

fn required_string(value: &Value, pointer: &str, scenario: &str) -> String {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing {scenario}: {value:#}"))
        .to_string()
}

struct ProviderFixture {
    base_url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    task: JoinHandle<()>,
}

impl ProviderFixture {
    async fn start() -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind provider fixture");
        let address = listener.local_addr().expect("provider fixture address");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let fixture_requests = Arc::clone(&requests);
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let requests = Arc::clone(&fixture_requests);
                tokio::spawn(async move {
                    let _ = handle_provider_connection(stream, requests).await;
                });
            }
        });
        Self {
            base_url: format!("http://{address}/v1"),
            requests,
            task,
        }
    }

    fn requests(&self) -> Vec<Value> {
        self.requests.lock().expect("provider requests").clone()
    }
}

impl Drop for ProviderFixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn handle_provider_connection(
    mut stream: TcpStream,
    requests: Arc<Mutex<Vec<Value>>>,
) -> std::io::Result<()> {
    let body = read_http_json_body(&mut stream).await?;
    requests
        .lock()
        .expect("record provider request")
        .push(body.clone());
    let streaming = body["stream"] == true;
    let response_body = if streaming {
        streaming_provider_response(&body)
    } else {
        json!({
            "id": "chatcmpl-title",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Remote turn"},
                "finish_reason": "stop"
            }]
        })
        .to_string()
    };
    let content_type = if streaming {
        "text/event-stream"
    } else {
        "application/json"
    };
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{response_body}",
        response_body.len()
    );
    stream.write_all(response.as_bytes()).await
}

fn streaming_provider_response(request: &Value) -> String {
    let has_tool_result = request
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| messages.iter().any(|message| message["role"] == "tool"));
    if has_tool_result {
        return sse(&[
            json!({
                "id": "chatcmpl-final",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": "remote command completed"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
            })
            .to_string(),
            "[DONE]".to_string(),
        ]);
    }

    let turn_number = request
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| {
            messages.iter().find_map(|message| {
                (message["role"] == "user"
                    && message["content"].as_str() == Some("second remote command"))
                .then_some(2)
            })
        })
        .unwrap_or(1);
    let tool_call_id = format!("remote-exec-call-{turn_number}");

    sse(&[
        json!({
            "id": "chatcmpl-tool",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": "{\"cmd\":\"printf local-command-must-not-run\"}"
                        }
                    }]
                },
                "finish_reason": null
            }]
        })
        .to_string(),
        json!({
            "id": "chatcmpl-tool",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}
        })
        .to_string(),
        "[DONE]".to_string(),
    ])
}

fn sse(events: &[String]) -> String {
    events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect()
}

async fn read_http_json_body(stream: &mut TcpStream) -> std::io::Result<Value> {
    let mut buffer = Vec::new();
    let header_end = loop {
        let mut chunk = [0_u8; 1024];
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Ok(json!({}));
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(index) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break index;
        }
    };
    let headers = String::from_utf8_lossy(&buffer[..header_end]);
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
    serde_json::from_slice(&buffer[body_start..body_start + content_length])
        .map_err(std::io::Error::other)
}

struct RemoteExecFixture {
    url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    task: JoinHandle<()>,
}

impl RemoteExecFixture {
    async fn start() -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind remote exec fixture");
        let address = listener.local_addr().expect("remote exec fixture address");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let fixture_requests = Arc::clone(&requests);
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let requests = Arc::clone(&fixture_requests);
                tokio::spawn(async move {
                    let Ok(mut socket) = accept_async(stream).await else {
                        return;
                    };
                    while let Some(message) = socket.next().await {
                        let Ok(Message::Text(text)) = message else {
                            continue;
                        };
                        let request: Value =
                            serde_json::from_str(&text).expect("remote exec request JSON");
                        requests
                            .lock()
                            .expect("remote exec requests")
                            .push(request.clone());
                        let Some(id) = request.get("id") else {
                            continue;
                        };
                        let result = match request["method"].as_str().unwrap_or_default() {
                            "initialize" => json!({"sessionId": "remote-turn-session"}),
                            "environment/info" => json!({
                                "shell": {"name": "fixture-sh", "path": "/bin/fixture-sh"},
                                "cwd": "file:///remote/workspace"
                            }),
                            "environment/status" => json!({"status": "ready"}),
                            "process/start" => json!({
                                "processId": request["params"]["processId"].clone()
                            }),
                            "process/read" => json!({
                                "chunks": [{
                                    "seq": 0,
                                    "stream": "stdout",
                                    "chunk": base64::engine::general_purpose::STANDARD
                                        .encode(REMOTE_OUTPUT)
                                }],
                                "nextSeq": 1,
                                "exited": true,
                                "exitCode": 0,
                                "closed": false,
                                "failure": null,
                                "sandboxDenied": false
                            }),
                            "process/write" | "process/signal" | "process/terminate" => json!({}),
                            method => panic!("unexpected remote exec method: {method}"),
                        };
                        socket
                            .send(Message::Text(
                                json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string(),
                            ))
                            .await
                            .expect("send remote exec response");
                    }
                });
            }
        });
        Self {
            url: format!("ws://{address}"),
            requests,
            task,
        }
    }

    fn requests(&self) -> Vec<Value> {
        self.requests.lock().expect("remote exec requests").clone()
    }
}

impl Drop for RemoteExecFixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}

struct TransportClient {
    input: DuplexStream,
    lines: Lines<BufReader<DuplexStream>>,
    pending: VecDeque<Value>,
    runner: JoinHandle<Result<(), AppServerError>>,
}

impl TransportClient {
    async fn start(server: AppServer, name: &str) -> Self {
        let (input, input_server) = tokio::io::duplex(128 * 1024);
        let (output_server, output) = tokio::io::duplex(128 * 1024);
        let runner = tokio::spawn(run_json_lines(server, input_server, output_server));
        let mut client = Self {
            input,
            lines: BufReader::new(output).lines(),
            pending: VecDeque::new(),
            runner,
        };
        client
            .request_ok(
                1,
                METHOD_INITIALIZE,
                json!({
                    "clientInfo": {"name": name, "version": "1.0.0"},
                    "capabilities": {"experimentalApi": true}
                }),
            )
            .await;
        client
            .write(json!({
                "jsonrpc": "2.0",
                "method": METHOD_INITIALIZED,
                "params": {}
            }))
            .await;
        client
    }

    async fn request_ok(&mut self, id: u64, method: &str, params: Value) -> Value {
        self.write(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        }))
        .await;
        if let Some(position) = self
            .pending
            .iter()
            .position(|message| message.get("id") == Some(&json!(id)))
        {
            return Self::assert_ok_response(
                method,
                self.pending.remove(position).expect("pending response"),
            );
        }
        loop {
            let message = self.read_message(method).await;
            if message.get("id") == Some(&json!(id)) {
                return Self::assert_ok_response(method, message);
            }
            self.pending.push_back(message);
        }
    }

    async fn wait_for_turn_completed(
        &mut self,
        thread_id: &str,
        turn_id: &str,
        remote: &RemoteExecFixture,
        provider: &ProviderFixture,
    ) {
        timeout(Duration::from_secs(10), async {
            loop {
                let pending_position = self
                    .pending
                    .iter()
                    .position(|message| turn_completed_matches(message, thread_id, turn_id));
                let message = match pending_position {
                    Some(position) => self.pending.remove(position).expect("pending terminal"),
                    None => self.read_message("turn/completed").await,
                };
                let matches = turn_completed_matches(&message, thread_id, turn_id);
                if matches {
                    assert_eq!(
                        message.pointer("/params/turn/status"),
                        Some(&json!("completed"))
                    );
                    return;
                }
                self.pending.push_back(message);
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "timed out waiting for turn/completed: {turn_id}; remote methods: {:?}; provider requests: {:?}",
                remote
                    .requests()
                    .iter()
                    .filter_map(|request| request["method"].as_str())
                    .collect::<Vec<_>>(),
                provider
                    .requests()
                    .iter()
                    .map(|request| {
                        json!({
                            "stream": request["stream"].clone(),
                            "roles": request["messages"]
                                .as_array()
                                .map(|messages| messages.iter().filter_map(|message| message["role"].as_str()).collect::<Vec<_>>()),
                            "last": request["messages"].as_array().and_then(|messages| messages.last()).cloned()
                        })
                    })
                    .collect::<Vec<_>>()
            )
        });
    }

    fn assert_ok_response(method: &str, message: Value) -> Value {
        assert!(
            message.get("error").is_none(),
            "{method} returned an error: {message:#}"
        );
        message
    }

    async fn read_message(&mut self, scenario: &str) -> Value {
        let line = timeout(Duration::from_secs(10), self.lines.next_line())
            .await
            .unwrap_or_else(|_| panic!("timed out waiting for JSON-RPC message: {scenario}"))
            .expect("read JSON-RPC message")
            .expect("JSON-RPC output closed");
        serde_json::from_str(&line).expect("decode JSON-RPC message")
    }

    async fn write(&mut self, message: Value) {
        self.input
            .write_all(format!("{message}\n").as_bytes())
            .await
            .expect("write JSON-RPC message");
    }

    async fn shutdown(self) {
        drop(self.input);
        timeout(Duration::from_secs(3), self.runner)
            .await
            .expect("JSON lines runner should stop")
            .expect("JSON lines runner task")
            .expect("JSON lines runner result");
    }
}

fn turn_completed_matches(message: &Value, thread_id: &str, turn_id: &str) -> bool {
    message.get("method") == Some(&json!(METHOD_TURN_COMPLETED))
        && message.pointer("/params/threadId") == Some(&json!(thread_id))
        && message.pointer("/params/turn/id") == Some(&json!(turn_id))
}
