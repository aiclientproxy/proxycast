use super::*;
use futures::StreamExt;
use serde_json::{json, Value};
use std::net::SocketAddr;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::oneshot,
    task::JoinHandle,
};

#[derive(Debug)]
struct HttpCapture {
    path: String,
    headers: String,
    body: Value,
}

fn provider_config(
    base_url: String,
    provider_name: &str,
    model_name: &str,
    protocol: RuntimeProviderProtocol,
) -> RuntimeProviderConfig {
    RuntimeProviderConfig {
        provider_name: provider_name.to_string(),
        provider_selector: Some(provider_name.to_string()),
        model_name: model_name.to_string(),
        api_key: Some("capture-key".to_string()),
        auth: RuntimeProviderAuth::ApiKey,
        base_url: Some(base_url),
        api_version: None,
        credential_uuid: format!("credential-{provider_name}"),
        reasoning_effort: None,
        service_tier: None,
        protocol: Some(protocol),
        supports_websockets: false,
        toolshim: false,
        toolshim_model: None,
    }
}

fn request_with_multimodal_tool_history() -> CurrentProviderRequest {
    CurrentProviderRequest::new(vec![
        CurrentProviderMessage::user(vec![
            CurrentProviderContent::Text("inspect this image".to_string()),
            CurrentProviderContent::Image {
                uri: "sidecar://image-1".to_string(),
                media_type: "image/png".to_string(),
                provider_data: Some("data:image/png;base64,abc".to_string()),
                detail: Some(ImageDetail::High),
            },
        ]),
        CurrentProviderMessage::assistant(vec![CurrentProviderContent::ToolCall(
            CurrentProviderToolCall::new("call-1", "read_file", json!({ "path": "README.md" })),
        )]),
        CurrentProviderMessage::tool(vec![CurrentProviderContent::ToolResult(
            CurrentProviderToolResult {
                call_id: "call-1".to_string(),
                name: "read_file".to_string(),
                success: true,
                output: "Lime".to_string(),
                error: None,
            },
        )]),
    ])
    .with_system_prompt(Some("Follow repository rules".to_string()))
    .with_tools(vec![CurrentProviderTool::function(
        "read_file",
        "Read a file",
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        }),
    )])
    .with_generation(GenerationOptions {
        max_tokens: Some(128),
        temperature: Some(0.2),
        top_p: Some(0.8),
        top_k: Some(16),
    })
}

fn request_with_custom_tool_history() -> CurrentProviderRequest {
    let mut call = CurrentProviderCustomToolCall::new(
        "custom-call-1",
        "run_code",
        "const answer = await exec({cmd: \"pwd\"}); return answer.stdout;",
    );
    call.namespace = Some("codemode".to_string());
    CurrentProviderRequest::new(vec![
        CurrentProviderMessage::user(vec![CurrentProviderContent::Text(
            "inspect the workspace".to_string(),
        )]),
        CurrentProviderMessage::assistant(vec![CurrentProviderContent::CustomToolCall(call)]),
        CurrentProviderMessage::tool(vec![CurrentProviderContent::CustomToolResult(
            CurrentProviderToolResult {
                call_id: "custom-call-1".to_string(),
                name: "run_code".to_string(),
                success: true,
                output: "{\"stdout\":\"/workspace\"}".to_string(),
                error: None,
            },
        )]),
    ])
    .with_system_prompt(Some("Follow repository rules".to_string()))
    .with_tools(vec![CurrentProviderTool::custom(
        "run_code",
        "Run a bounded CodeMode program",
        FreeformToolFormat {
            r#type: "grammar".to_string(),
            syntax: "lark".to_string(),
            definition: "program := statement*".to_string(),
        },
    )])
}

async fn capture_provider_request(
    provider_name: &str,
    model_name: &str,
    protocol: RuntimeProviderProtocol,
    response_body: &'static str,
) -> (HttpCapture, Vec<CanonicalLlmEvent>) {
    capture_provider_request_with(
        request_with_multimodal_tool_history(),
        provider_name,
        model_name,
        protocol,
        response_body,
        false,
    )
    .await
}

async fn capture_provider_request_with(
    request: CurrentProviderRequest,
    provider_name: &str,
    model_name: &str,
    protocol: RuntimeProviderProtocol,
    response_body: &'static str,
    official_openai_loopback: bool,
) -> (HttpCapture, Vec<CanonicalLlmEvent>) {
    let (base_url, capture, server) = spawn_http_capture_fixture(response_body).await;
    let local_address = base_url
        .strip_prefix("http://")
        .expect("capture fixture uses HTTP")
        .parse::<SocketAddr>()
        .expect("capture fixture address");
    let provider_base_url = if official_openai_loopback {
        format!("http://api.openai.com:{}/v1", local_address.port())
    } else {
        base_url
    };
    let mut client_builder = Client::builder().no_proxy();
    if official_openai_loopback {
        client_builder = client_builder.resolve("api.openai.com", local_address);
    }
    let client = CurrentProviderClient::with_client(
        provider_config(provider_base_url, provider_name, model_name, protocol),
        client_builder.build().expect("capture fixture HTTP client"),
    );
    let events = client
        .stream(request)
        .await
        .expect("open provider capture stream")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("collect provider capture events");
    let capture = capture.await.expect("capture provider request");
    server.await.expect("join provider capture fixture");
    (capture, events)
}

#[tokio::test]
async fn openai_chat_capture_proves_native_request_and_terminal_stream() {
    let response_body = concat!(
        "data: {\"id\":\"chatcmpl-capture\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
        "data: [DONE]\n\n"
    );
    let (capture, events) = capture_provider_request(
        "openai",
        "gpt-5",
        RuntimeProviderProtocol::ChatCompletions,
        response_body,
    )
    .await;

    assert_eq!(capture.path, "/v1/chat/completions");
    let headers = capture.headers.to_ascii_lowercase();
    assert!(headers.contains("\r\nauthorization: bearer capture-key\r\n"));
    assert!(!headers.contains("\r\nx-api-key:"));
    assert!(!headers.contains("\r\nanthropic-version:"));
    assert_eq!(capture.body["model"], "gpt-5");
    assert_eq!(capture.body["messages"][0]["role"], "system");
    assert_eq!(
        capture.body["messages"][1]["content"][1]["type"],
        "image_url"
    );
    assert_eq!(
        capture.body["messages"][2]["tool_calls"][0]["function"]["name"],
        "read_file"
    );
    assert_eq!(capture.body["messages"][3]["tool_call_id"], "call-1");
    assert_eq!(capture.body["tools"][0]["function"]["name"], "read_file");
    assert_eq!(capture.body["max_tokens"], 128);
    assert_eq!(capture.body["stream_options"]["include_usage"], true);
    assert!(terminal_matches(&events, "chatcmpl-capture", Some(3)));
}

#[tokio::test]
async fn openai_responses_http_capture_proves_native_request_and_terminal_stream() {
    let response_body = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"message-1\",\"delta\":\"done\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-capture\",\"output\":[],\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n"
    );
    let (capture, events) = capture_provider_request(
        "openai",
        "gpt-5",
        RuntimeProviderProtocol::Responses,
        response_body,
    )
    .await;

    assert_eq!(capture.path, "/v1/responses");
    let headers = capture.headers.to_ascii_lowercase();
    assert!(headers.contains("\r\nauthorization: bearer capture-key\r\n"));
    assert!(!headers.contains("\r\nx-api-key:"));
    assert_eq!(capture.body["model"], "gpt-5");
    assert_eq!(capture.body["instructions"], "Follow repository rules");
    assert_eq!(
        capture.body["input"][0]["content"][1]["type"],
        "input_image"
    );
    assert_eq!(capture.body["input"][0]["content"][1]["detail"], "high");
    assert_eq!(capture.body["input"][1]["type"], "function_call");
    assert_eq!(capture.body["input"][2]["type"], "function_call_output");
    assert_eq!(capture.body["tools"][0]["name"], "read_file");
    assert_eq!(capture.body["max_output_tokens"], 128);
    assert_eq!(capture.body["store"], false);
    assert!(terminal_matches(&events, "resp-capture", Some(3)));
}

#[tokio::test]
async fn official_responses_custom_tool_capture_proves_native_wire_and_typed_event() {
    let response_body = concat!(
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"custom_tool_call\",\"call_id\":\"custom-call-2\",\"name\":\"run_code\",\"namespace\":\"codemode\"}}\n\n",
        "data: {\"type\":\"response.custom_tool_call_input.delta\",\"call_id\":\"custom-call-2\",\"name\":\"run_code\",\"namespace\":\"codemode\",\"delta\":\"return 42;\"}\n\n",
        "data: {\"type\":\"response.custom_tool_call_input.done\",\"call_id\":\"custom-call-2\",\"name\":\"run_code\",\"namespace\":\"codemode\",\"input\":\"return 42;\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-custom-capture\",\"output\":[],\"usage\":{\"input_tokens\":3,\"output_tokens\":2,\"total_tokens\":5}}}\n\n"
    );
    let (capture, events) = capture_provider_request_with(
        request_with_custom_tool_history(),
        "openai",
        "gpt-5",
        RuntimeProviderProtocol::Responses,
        response_body,
        true,
    )
    .await;

    assert_eq!(capture.path, "/v1/responses");
    assert_eq!(
        capture.body["tools"][0],
        json!({
            "type": "custom",
            "name": "run_code",
            "description": "Run a bounded CodeMode program",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": "program := statement*"
            }
        })
    );
    assert_eq!(
        capture.body["input"][1],
        json!({
            "type": "custom_tool_call",
            "call_id": "custom-call-1",
            "name": "run_code",
            "input": "const answer = await exec({cmd: \"pwd\"}); return answer.stdout;",
            "namespace": "codemode"
        })
    );
    assert_eq!(
        capture.body["input"][2],
        json!({
            "type": "custom_tool_call_output",
            "call_id": "custom-call-1",
            "output": "{\"stdout\":\"/workspace\"}"
        })
    );
    assert!(events.iter().any(|event| matches!(
        event,
        CanonicalLlmEvent::CustomToolCall {
            id,
            name,
            input,
            namespace: Some(namespace),
            ..
        } if id == "custom-call-2"
            && name == "run_code"
            && input == "return 42;"
            && namespace == "codemode"
    )));
    assert!(terminal_matches(&events, "resp-custom-capture", Some(5)));
}

#[tokio::test]
async fn anthropic_capture_proves_native_request_and_terminal_stream() {
    let response_body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-capture\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4\",\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"done\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":2,\"output_tokens\":1}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );
    let (capture, events) = capture_provider_request(
        "anthropic",
        "claude-sonnet-4",
        RuntimeProviderProtocol::AnthropicMessages,
        response_body,
    )
    .await;

    assert_eq!(capture.path, "/v1/messages");
    let headers = capture.headers.to_ascii_lowercase();
    assert!(headers.contains("\r\nx-api-key: capture-key\r\n"));
    assert!(headers.contains("\r\nanthropic-version: 2023-06-01\r\n"));
    assert!(!headers.contains("\r\nauthorization:"));
    assert_eq!(capture.body["model"], "claude-sonnet-4");
    assert_eq!(capture.body["system"], "Follow repository rules");
    assert_eq!(capture.body["messages"][0]["content"][1]["type"], "image");
    assert_eq!(
        capture.body["messages"][1]["content"][0]["type"],
        "tool_use"
    );
    assert_eq!(
        capture.body["messages"][2]["content"][0]["type"],
        "tool_result"
    );
    assert_eq!(capture.body["tools"][0]["name"], "read_file");
    assert_eq!(capture.body["max_tokens"], 128);
    assert_eq!(capture.body["top_k"], 16);
    assert!(terminal_matches(&events, "msg-capture", None));
    assert!(events.iter().any(|event| matches!(
        event,
        CanonicalLlmEvent::Finish {
            usage: Some(usage),
            ..
        } if usage.input_tokens == Some(2) && usage.output_tokens == Some(1)
    )));
}

fn terminal_matches(
    events: &[CanonicalLlmEvent],
    response_id: &str,
    total_tokens: Option<u64>,
) -> bool {
    events.iter().any(|event| {
        matches!(
            event,
            CanonicalLlmEvent::Finish {
                usage: Some(usage),
                response_id: Some(actual_response_id),
                ..
            } if actual_response_id == response_id
                && total_tokens.map_or(true, |total| usage.total_tokens == Some(total))
        )
    })
}

async fn spawn_http_capture_fixture(
    response_body: &'static str,
) -> (String, oneshot::Receiver<HttpCapture>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind provider capture fixture");
    let address = listener.local_addr().expect("provider fixture address");
    let (capture_tx, capture_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept provider request");
        let request = read_http_request(&mut stream).await;
        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("provider request headers")
            + 4;
        let headers = String::from_utf8_lossy(&request[..header_end]).into_owned();
        let path = headers
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or_default()
            .to_string();
        let body = serde_json::from_slice(&request[header_end..]).expect("provider request body");
        let _ = capture_tx.send(HttpCapture {
            path,
            headers,
            body,
        });
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
            response_body.len()
        );
        stream
            .write_all(response.as_bytes())
            .await
            .expect("write provider response");
        stream.shutdown().await.expect("close provider response");
    });
    (format!("http://{address}"), capture_rx, server)
}

async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 2048];
    let mut expected_length = None;
    loop {
        let read = stream
            .read(&mut buffer)
            .await
            .expect("read provider request");
        if read == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..read]);
        let Some(header_end) = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|index| index + 4)
        else {
            continue;
        };
        let content_length = *expected_length.get_or_insert_with(|| {
            String::from_utf8_lossy(&request[..header_end])
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .expect("provider request content length")
        });
        if request.len() >= header_end + content_length {
            request.truncate(header_end + content_length);
            break;
        }
    }
    request
}
