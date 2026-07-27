use super::*;
use crate::runtime::RuntimeHostContext;
use crate::ExecutionBackend;
use app_server_protocol::{
    AgentSession, AgentSessionStatus, AgentTurn, AgentTurnStatus, RuntimeOptions, RuntimeRequest,
};
use lime_core::database::dao::api_key_provider::ApiProviderType;
use lime_core::database::schema::create_tables;
use rusqlite::Connection;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};

const PRIMARY_MODEL: &str = "gpt-4.1-mini";
const BACKUP_MODEL: &str = "openai/gpt-4.1-mini";

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

struct RuntimeRerouteFixture {
    base_url: String,
    models: Arc<Mutex<Vec<String>>>,
    server_task: JoinHandle<()>,
}

impl RuntimeRerouteFixture {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind runtime reroute fixture");
        let address = listener.local_addr().expect("fixture address");
        let models = Arc::new(Mutex::new(Vec::new()));
        let server_models = Arc::clone(&models);
        let server_task = tokio::spawn(async move {
            for _ in 0..8 {
                let (mut stream, _) = listener.accept().await.expect("accept provider request");
                let body = read_request_body(&mut stream)
                    .await
                    .expect("read provider request");
                let model = body
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                server_models
                    .lock()
                    .expect("record provider model")
                    .push(model.clone());
                if model == PRIMARY_MODEL {
                    write_response(
                        &mut stream,
                        "503 Service Unavailable",
                        "application/json",
                        &json!({ "error": { "message": "primary unavailable" } }).to_string(),
                    )
                    .await;
                    continue;
                }

                let body = format!(
                    "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                    json!({
                        "id": "chatcmpl-runtime-reroute",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": BACKUP_MODEL,
                        "choices": [{
                            "index": 0,
                            "delta": { "content": "backup answer" },
                            "finish_reason": null
                        }]
                    }),
                    json!({
                        "id": "chatcmpl-runtime-reroute",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": BACKUP_MODEL,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 2,
                            "total_tokens": 4
                        }
                    })
                );
                write_response(&mut stream, "200 OK", "text/event-stream", &body).await;
                break;
            }
        });
        Self {
            base_url: format!("http://{address}/v1"),
            models,
            server_task,
        }
    }
}

impl Drop for RuntimeRerouteFixture {
    fn drop(&mut self) {
        self.server_task.abort();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retryable_provider_failure_reroutes_to_ready_profile_candidate() {
    let fixture = RuntimeRerouteFixture::start().await;
    let db = test_db();
    let backend = RuntimeBackend::with_db(db.clone());
    backend
        .api_key_provider_service
        .initialize_system_providers(&db)
        .expect("initialize providers");
    let primary_provider = add_provider(
        &backend.api_key_provider_service,
        &db,
        "openai",
        "Primary",
        &fixture.base_url,
        PRIMARY_MODEL,
    );
    let backup_provider = add_provider(
        &backend.api_key_provider_service,
        &db,
        "openrouter",
        "Backup",
        &fixture.base_url,
        BACKUP_MODEL,
    );
    let request = execution_request(&primary_provider, &backup_provider);
    let mut sink = RecordingSink::default();

    timeout(
        Duration::from_secs(15),
        ExecutionBackend::start_turn(&backend, request, &mut sink),
    )
    .await
    .expect("runtime reroute timeout")
    .expect("runtime reroute turn");

    timeout(Duration::from_secs(2), async {
        while !fixture.server_task.is_finished() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("provider fixture completion");
    let models = fixture.models.lock().expect("provider request models");
    assert!(models.len() >= 2);
    assert!(models[..models.len() - 1]
        .iter()
        .all(|model| model == PRIMARY_MODEL));
    assert_eq!(models.last().map(String::as_str), Some(BACKUP_MODEL));

    let fallback = sink
        .events
        .iter()
        .find(|event| {
            event.event_type == "routing.fallback.applied"
                && event.payload["fallbackReason"] == "runtime_provider_failure"
        })
        .expect("runtime fallback evidence");
    assert_eq!(
        fallback.payload["runtimeFailure"]["provider"].as_str(),
        Some(primary_provider.as_str())
    );
    assert_eq!(
        fallback.payload["runtimeFailure"]["classification"].as_str(),
        Some("provider-internal")
    );
    assert_eq!(
        fallback.payload["selectedProvider"].as_str(),
        Some(backup_provider.as_str())
    );
    assert!(sink
        .events
        .iter()
        .any(|event| event.event_type == "turn.completed"));
    assert!(sink
        .events
        .iter()
        .all(|event| event.event_type != "routing.not_possible"));
    assert!(sink
        .events
        .iter()
        .all(|event| event.event_type != "model.rerouted"));
}

fn add_provider(
    service: &ApiKeyProviderService,
    db: &DbConnection,
    provider_id: &str,
    name: &str,
    base_url: &str,
    model: &str,
) -> String {
    let provider = service
        .update_provider(
            db,
            provider_id,
            Some(name.to_string()),
            Some(ApiProviderType::Openai),
            Some(base_url.to_string()),
            Some(true),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(vec![
                lime_core::models::model_registry::ProviderModelConfig::hint(model),
            ]),
        )
        .expect("configure provider");
    service
        .add_api_key(db, &provider.id, &format!("{name}-key"), None, true)
        .expect("provider api key");
    provider.id
}

fn execution_request(primary_provider: &str, backup_provider: &str) -> ExecutionRequest {
    let session_id = "session-runtime-reroute".to_string();
    let thread_id = "thread-runtime-reroute".to_string();
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
            turn_id: "turn-runtime-reroute".to_string(),
            session_id,
            thread_id,
            status: AgentTurnStatus::Accepted,
            started_at: None,
            completed_at: None,
        },
        forked_from_thread_id: None,
        input: agent_runtime::reply_input::RuntimeReplyInput::text("hello"),
        runtime_options: Some(RuntimeOptions {
            stream: true,
            runtime_request: Some(RuntimeRequest {
                provider_preference: Some(primary_provider.to_string()),
                model_preference: Some(PRIMARY_MODEL.to_string()),
                metadata: Some(json!({
                    "harness": {
                        "coding_model_slots": {
                            "coding": {
                                "provider": primary_provider,
                                "model": PRIMARY_MODEL
                            },
                            "base": {
                                "provider": backup_provider,
                                "model": BACKUP_MODEL
                            }
                        }
                    }
                })),
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

async fn read_request_body(stream: &mut TcpStream) -> std::io::Result<Value> {
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
