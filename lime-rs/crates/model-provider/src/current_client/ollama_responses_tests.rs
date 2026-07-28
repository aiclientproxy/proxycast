use super::*;
use futures::StreamExt;
use serde_json::{json, Value};
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

fn ollama_config(base_url: String) -> RuntimeProviderConfig {
    RuntimeProviderConfig {
        provider_name: "ollama".to_string(),
        provider_selector: Some("ollama".to_string()),
        model_name: "qwen3:14b".to_string(),
        api_key: None,
        auth: RuntimeProviderAuth::NoAuth,
        base_url: Some(base_url),
        api_version: None,
        credential_uuid: String::new(),
        reasoning_effort: None,
        service_tier: None,
        protocol: Some(RuntimeProviderProtocol::Responses),
        supports_websockets: false,
        toolshim: false,
        toolshim_model: None,
    }
}

async fn spawn_ollama_responses_fixture(
    response_body: &'static str,
) -> (String, oneshot::Receiver<HttpCapture>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind Ollama Responses fixture");
    let address = listener.local_addr().expect("Ollama fixture address");
    let (capture_tx, capture_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept Ollama request");
        let request = read_http_request(&mut stream).await;
        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("Ollama request headers")
            + 4;
        let headers = String::from_utf8_lossy(&request[..header_end]).into_owned();
        let path = headers
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or_default()
            .to_string();
        let body = serde_json::from_slice(&request[header_end..]).expect("Ollama request body");
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
            .expect("write Ollama response");
        stream.shutdown().await.expect("close Ollama response");
    });
    (format!("http://{address}"), capture_rx, server)
}

async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 2048];
    let mut expected_length = None;
    loop {
        let read = stream.read(&mut buffer).await.expect("read Ollama request");
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
                .expect("Ollama request content length")
        });
        if request.len() >= header_end + content_length {
            request.truncate(header_end + content_length);
            break;
        }
    }
    request
}

fn request_with_tool_history() -> CurrentProviderRequest {
    CurrentProviderRequest::new(vec![
        CurrentProviderMessage::user(vec![CurrentProviderContent::Text(
            "inspect workspace".to_string(),
        )]),
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
    .with_tools(vec![CurrentProviderTool {
        name: "read_file".to_string(),
        description: "Read a file".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        }),
    }])
}

#[tokio::test]
async fn ollama_uses_keyless_responses_endpoint_and_canonical_stream() {
    let response_body = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"message-1\",\"delta\":\"done\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-ollama\",\"output\":[],\"usage\":{\"input_tokens\":8,\"output_tokens\":1,\"total_tokens\":9}}}\n\n",
    );
    let (base_url, capture, server) = spawn_ollama_responses_fixture(response_body).await;
    let client = CurrentProviderClient::with_client(
        ollama_config(base_url),
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("Ollama fixture HTTP client"),
    );

    let events = client
        .stream(request_with_tool_history())
        .await
        .expect("open Ollama Responses stream")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("collect Ollama Responses events");
    let capture = capture.await.expect("capture Ollama Responses request");
    server.await.expect("join Ollama Responses fixture");

    assert_eq!(capture.path, "/v1/responses");
    let headers = capture.headers.to_ascii_lowercase();
    assert!(!headers.contains("authorization:"));
    assert_eq!(capture.body["model"], "qwen3:14b");
    assert_eq!(capture.body["instructions"], "Follow repository rules");
    assert_eq!(capture.body["input"][0]["role"], "user");
    assert_eq!(capture.body["input"][1]["type"], "function_call");
    assert_eq!(capture.body["input"][2]["type"], "function_call_output");
    assert_eq!(capture.body["tools"][0]["name"], "read_file");
    assert_eq!(capture.body["stream"], true);
    assert_eq!(capture.body["store"], false);
    assert!(events.iter().any(|event| matches!(
        event,
        CanonicalLlmEvent::TextDelta { text, .. } if text == "done"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        CanonicalLlmEvent::Finish {
            response_id: Some(response_id),
            ..
        } if response_id == "resp-ollama"
    )));
}
