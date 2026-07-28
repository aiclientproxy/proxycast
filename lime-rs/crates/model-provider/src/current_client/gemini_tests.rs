use super::*;
use futures::StreamExt;
use serde_json::{json, Value};
use std::collections::BTreeMap;
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

fn gemini_config(base_url: String) -> RuntimeProviderConfig {
    RuntimeProviderConfig {
        provider_name: "google".to_string(),
        provider_selector: Some("gemini".to_string()),
        model_name: "gemini-2.5-flash".to_string(),
        api_key: Some("gemini-test-key".to_string()),
        auth: RuntimeProviderAuth::ApiKey,
        base_url: Some(base_url),
        api_version: None,
        credential_uuid: "credential-gemini".to_string(),
        reasoning_effort: None,
        service_tier: None,
        protocol: Some(RuntimeProviderProtocol::GeminiGenerateContent),
        supports_websockets: false,
        toolshim: false,
        toolshim_model: None,
    }
}

async fn spawn_gemini_fixture(
    response_body: &'static str,
) -> (String, oneshot::Receiver<HttpCapture>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind Gemini fixture");
    let address = listener.local_addr().expect("Gemini fixture address");
    let (capture_tx, capture_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept Gemini request");
        let request = read_http_request(&mut stream).await;
        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("Gemini request headers")
            + 4;
        let headers = String::from_utf8_lossy(&request[..header_end]).into_owned();
        let path = headers
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or_default()
            .to_string();
        let body = serde_json::from_slice(&request[header_end..]).expect("Gemini request body");
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
            .expect("write Gemini response");
        stream.shutdown().await.expect("close Gemini response");
    });
    (format!("http://{address}"), capture_rx, server)
}

async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 2048];
    let mut expected_length = None;
    loop {
        let read = stream.read(&mut buffer).await.expect("read Gemini request");
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
                .expect("Gemini request content length")
        });
        if request.len() >= header_end + content_length {
            request.truncate(header_end + content_length);
            break;
        }
    }
    request
}

fn request_with_history() -> CurrentProviderRequest {
    CurrentProviderRequest::new(vec![
        CurrentProviderMessage::user(vec![
            CurrentProviderContent::Text("inspect".to_string()),
            CurrentProviderContent::Image {
                uri: "sidecar://image-1".to_string(),
                media_type: "image/png".to_string(),
                provider_data: Some("data:image/png;base64,AAE=".to_string()),
                detail: Some(ImageDetail::High),
            },
        ]),
        CurrentProviderMessage::assistant(vec![CurrentProviderContent::ToolCall(
            CurrentProviderToolCall::new("call-1", "lookup", json!({ "query": "weather" }))
                .with_provider_metadata(BTreeMap::from([(
                    "google".to_string(),
                    json!({ "thoughtSignature": "history-sig" }),
                )])),
        )]),
        CurrentProviderMessage::tool(vec![CurrentProviderContent::ToolResult(
            CurrentProviderToolResult {
                call_id: "call-1".to_string(),
                name: "lookup".to_string(),
                success: true,
                output: "sunny".to_string(),
                error: None,
            },
        )]),
    ])
    .with_system_prompt(Some("Be concise.".to_string()))
    .with_tools(vec![CurrentProviderTool {
        name: "lookup".to_string(),
        description: "Lookup data".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "query": { "type": ["string", "null"] },
                "ignored": { "$ref": "#/$defs/ignored" }
            },
            "required": ["query", "missing"],
            "additionalProperties": false
        }),
    }])
    .with_generation(GenerationOptions {
        max_tokens: Some(256),
        temperature: Some(0.2),
        top_p: Some(0.8),
        top_k: Some(40),
    })
}

#[tokio::test]
async fn gemini_capture_proves_endpoint_auth_and_canonical_lowering() {
    let response_body = concat!(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"done\"}]},\"finishReason\":\"STOP\"}]}\n\n",
        "data: {\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":2,\"totalTokenCount\":7}}\n\n"
    );
    let (base_url, capture, server) = spawn_gemini_fixture(response_body).await;
    let client = CurrentProviderClient::with_client(
        gemini_config(base_url),
        Client::builder().no_proxy().build().expect("HTTP client"),
    );

    let events = client
        .stream(request_with_history())
        .await
        .expect("Gemini stream")
        .map(|event| event.expect("Gemini event"))
        .collect::<Vec<_>>()
        .await;
    let capture = capture.await.expect("Gemini request capture");
    server.await.expect("Gemini fixture");

    assert_eq!(
        capture.path,
        "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    );
    let headers = capture.headers.to_ascii_lowercase();
    assert!(headers.contains("x-goog-api-key: gemini-test-key"));
    assert!(!headers.contains("authorization:"));
    assert_eq!(
        capture.body["systemInstruction"]["parts"][0]["text"],
        "Be concise."
    );
    assert_eq!(capture.body["contents"][0]["role"], "user");
    assert_eq!(
        capture.body["contents"][0]["parts"][1]["inlineData"]["data"],
        "AAE="
    );
    assert_eq!(capture.body["contents"][1]["role"], "model");
    assert_eq!(
        capture.body["contents"][1]["parts"][0]["functionCall"]["name"],
        "lookup"
    );
    assert_eq!(
        capture.body["contents"][1]["parts"][0]["thoughtSignature"],
        "history-sig"
    );
    assert_eq!(
        capture.body["contents"][2]["parts"][0]["functionResponse"]["name"],
        "lookup"
    );
    assert_eq!(
        capture.body["tools"][0]["functionDeclarations"][0]["parameters"]["required"],
        json!(["query"])
    );
    assert!(
        capture.body["tools"][0]["functionDeclarations"][0]["parameters"]["properties"]["ignored"]
            .is_null()
    );
    assert_eq!(capture.body["generationConfig"]["maxOutputTokens"], 256);
    assert!(events.iter().any(|event| matches!(
        event,
        CanonicalLlmEvent::Finish { reason: FinishReason::Stop, usage: Some(usage), .. }
            if usage.total_tokens == Some(7)
    )));
}

#[tokio::test]
async fn gemini_stream_waits_for_usage_trailer_and_projects_reasoning_tool_lifecycle() {
    let response_body = concat!(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"checking\",\"thought\":true},{\"functionCall\":{\"name\":\"lookup\",\"args\":{\"query\":\"weather\"}},\"thoughtSignature\":\"sig\"}]},\"finishReason\":\"STOP\"}]}\n\n",
        "data: {\"usageMetadata\":{\"promptTokenCount\":8,\"cachedContentTokenCount\":3,\"candidatesTokenCount\":2,\"thoughtsTokenCount\":4,\"totalTokenCount\":14}}\n\n"
    );
    let (base_url, _capture, server) = spawn_gemini_fixture(response_body).await;
    let client = CurrentProviderClient::with_client(
        gemini_config(base_url),
        Client::builder().no_proxy().build().expect("HTTP client"),
    );
    let events = client
        .stream(CurrentProviderRequest::new(vec![
            CurrentProviderMessage::user(vec![CurrentProviderContent::Text("go".to_string())]),
        ]))
        .await
        .expect("Gemini stream")
        .map(|event| event.expect("Gemini event"))
        .collect::<Vec<_>>()
        .await;
    server.await.expect("Gemini fixture");

    let lifecycle = events
        .iter()
        .filter_map(|event| match event {
            CanonicalLlmEvent::ReasoningStart { .. } => Some("reasoning-start"),
            CanonicalLlmEvent::ReasoningContentDelta { .. } => Some("reasoning-delta"),
            CanonicalLlmEvent::ReasoningEnd { .. } => Some("reasoning-end"),
            CanonicalLlmEvent::ToolInputStart { .. } => Some("tool-start"),
            CanonicalLlmEvent::ToolInputDelta { .. } => Some("tool-delta"),
            CanonicalLlmEvent::ToolInputEnd { .. } => Some("tool-end"),
            CanonicalLlmEvent::ToolCall { .. } => Some("tool-call"),
            CanonicalLlmEvent::Usage { .. } => Some("usage"),
            CanonicalLlmEvent::Finish { .. } => Some("finish"),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        lifecycle,
        [
            "reasoning-start",
            "reasoning-delta",
            "reasoning-end",
            "tool-start",
            "tool-delta",
            "tool-end",
            "tool-call",
            "usage",
            "finish"
        ]
    );
    assert!(matches!(
        events.last(),
        Some(CanonicalLlmEvent::Finish { reason: FinishReason::ToolCall, usage: Some(usage), .. })
            if usage.input_tokens == Some(8)
                && usage.output_tokens == Some(6)
                && usage.reasoning_tokens == Some(4)
                && usage.cache_read_input_tokens == Some(3)
                && usage.non_cached_input_tokens == Some(5)
    ));
    assert!(events.iter().any(|event| matches!(
        event,
        CanonicalLlmEvent::ToolCall { provider_metadata, .. }
            if provider_metadata.get("google")
                .and_then(|value| value.get("thoughtSignature"))
                == Some(&json!("sig"))
    )));
}

#[tokio::test]
async fn gemini_blocked_prompt_and_malformed_output_fail_closed() {
    let (base_url, _capture, server) =
        spawn_gemini_fixture("data: {\"promptFeedback\":{\"blockReason\":\"SAFETY\"}}\n\n").await;
    let client = CurrentProviderClient::with_client(
        gemini_config(base_url),
        Client::builder().no_proxy().build().expect("HTTP client"),
    );
    let events = client
        .stream(CurrentProviderRequest::new(vec![
            CurrentProviderMessage::user(vec![CurrentProviderContent::Text("blocked".to_string())]),
        ]))
        .await
        .expect("Gemini blocked stream")
        .collect::<Vec<_>>()
        .await;
    server.await.expect("Gemini blocked fixture");
    assert!(matches!(
        events.as_slice(),
        [Ok(CanonicalLlmEvent::ProviderError {
            classification: Some(FailureClassification::ContentPolicy),
            retryable: Some(false),
            ..
        })]
    ));

    let (base_url, _capture, server) = spawn_gemini_fixture(
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":\"AAE=\"}}]}}]}\n\n",
    )
    .await;
    let client = CurrentProviderClient::with_client(
        gemini_config(base_url),
        Client::builder().no_proxy().build().expect("HTTP client"),
    );
    let events = client
        .stream(CurrentProviderRequest::new(vec![
            CurrentProviderMessage::user(vec![CurrentProviderContent::Text(
                "malformed".to_string(),
            )]),
        ]))
        .await
        .expect("Gemini malformed stream")
        .collect::<Vec<_>>()
        .await;
    server.await.expect("Gemini malformed fixture");
    assert!(matches!(
        events.as_slice(),
        [Err(CurrentProviderError {
            classification: Some(FailureClassification::InvalidRequest),
            retryable: false,
            ..
        })]
    ));
}

#[tokio::test]
async fn gemini_truncated_stream_is_retryable_and_remote_media_fails_closed() {
    let (base_url, _capture, server) = spawn_gemini_fixture(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"partial\"}]}}]}\n\n",
    )
    .await;
    let client = CurrentProviderClient::with_client(
        gemini_config(base_url),
        Client::builder().no_proxy().build().expect("HTTP client"),
    );
    let events = client
        .stream(CurrentProviderRequest::new(vec![
            CurrentProviderMessage::user(vec![CurrentProviderContent::Text("go".to_string())]),
        ]))
        .await
        .expect("Gemini stream")
        .map(|event| event.expect("Gemini event"))
        .collect::<Vec<_>>()
        .await;
    server.await.expect("Gemini fixture");
    assert!(matches!(
        events.last(),
        Some(CanonicalLlmEvent::ProviderError {
            classification: Some(FailureClassification::Transport),
            retryable: Some(true),
            ..
        })
    ));

    let config = gemini_config("https://generativelanguage.googleapis.com".to_string());
    let canonical = CurrentProviderRequest::new(vec![CurrentProviderMessage::user(vec![
        CurrentProviderContent::Image {
            uri: "https://example.com/image.png".to_string(),
            media_type: "image/png".to_string(),
            provider_data: None,
            detail: None,
        },
    ])])
    .into_canonical("gemini-2.5-flash")
    .expect("canonical request");
    let error = super::gemini::request(&canonical, &Default::default())
        .expect_err("arbitrary remote media must fail closed");
    assert_eq!(
        error.classification,
        Some(FailureClassification::InvalidRequest)
    );
    let spoofed = CurrentProviderRequest::new(vec![CurrentProviderMessage::user(vec![
        CurrentProviderContent::Image {
            uri: "https://example.com/generativelanguage.googleapis.com/file.png".to_string(),
            media_type: "image/png".to_string(),
            provider_data: None,
            detail: None,
        },
    ])])
    .into_canonical("gemini-2.5-flash")
    .expect("canonical request");
    let error = super::gemini::request(&spoofed, &Default::default())
        .expect_err("spoofed Google file host must fail closed");
    assert_eq!(
        error.classification,
        Some(FailureClassification::InvalidRequest)
    );
    assert_eq!(
        config.protocol,
        Some(RuntimeProviderProtocol::GeminiGenerateContent)
    );
}
