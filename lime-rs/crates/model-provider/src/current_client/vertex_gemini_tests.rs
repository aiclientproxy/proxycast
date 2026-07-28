use super::*;
use futures::StreamExt;
use serde_json::Value;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::oneshot,
};

#[derive(Debug)]
struct HttpCapture {
    path: String,
    headers: String,
    body: Value,
}

fn vertex_config(base_url: Option<String>) -> RuntimeProviderConfig {
    RuntimeProviderConfig {
        provider_name: "gcpvertexai".to_string(),
        provider_selector: Some("vertexai".to_string()),
        model_name: "gemini-2.5-flash".to_string(),
        api_key: Some("vertex-access-token".to_string()),
        auth: RuntimeProviderAuth::ApiKey,
        base_url,
        api_version: None,
        credential_uuid: "credential-vertex".to_string(),
        reasoning_effort: None,
        service_tier: None,
        protocol: Some(RuntimeProviderProtocol::VertexGemini),
        supports_websockets: false,
        toolshim: false,
        toolshim_model: None,
    }
}

#[tokio::test]
async fn vertex_capture_proves_project_endpoint_bearer_auth_and_gemini_lowering() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind Vertex fixture");
    let address = listener.local_addr().expect("Vertex fixture address");
    let (capture_tx, capture_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept Vertex request");
        let request = read_http_request(&mut stream).await;
        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("Vertex request headers")
            + 4;
        let headers = String::from_utf8_lossy(&request[..header_end]).into_owned();
        let path = headers
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or_default()
            .to_string();
        let body = serde_json::from_slice(&request[header_end..]).expect("Vertex request body");
        let _ = capture_tx.send(HttpCapture {
            path,
            headers,
            body,
        });
        let response_body = concat!(
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"done\"}]},\"finishReason\":\"STOP\"}]}\n\n",
            "data: {\"usageMetadata\":{\"promptTokenCount\":2,\"candidatesTokenCount\":1,\"totalTokenCount\":3}}\n\n"
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
            response_body.len()
        );
        stream
            .write_all(response.as_bytes())
            .await
            .expect("write Vertex response");
        stream.shutdown().await.expect("close Vertex response");
    });
    let base_url = vertex_gemini_base_url(
        Some(&format!("http://{address}")),
        Some("project-alpha"),
        Some("us-central1"),
    )
    .expect("resolve Vertex fixture endpoint");
    let client = CurrentProviderClient::with_client(
        vertex_config(Some(base_url)),
        Client::builder().no_proxy().build().expect("HTTP client"),
    );

    let events = client
        .stream(CurrentProviderRequest::new(vec![
            CurrentProviderMessage::user(vec![CurrentProviderContent::Text("hello".to_string())]),
        ]))
        .await
        .expect("Vertex stream")
        .map(|event| event.expect("Vertex event"))
        .collect::<Vec<_>>()
        .await;
    let capture = capture_rx.await.expect("Vertex request capture");
    server.await.expect("Vertex fixture");

    assert_eq!(
        capture.path,
        "/v1/projects/project-alpha/locations/us-central1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    );
    let headers = capture.headers.to_ascii_lowercase();
    assert!(headers.contains("authorization: bearer vertex-access-token"));
    assert!(!headers.contains("x-goog-api-key:"));
    assert_eq!(capture.body["contents"][0]["role"], "user");
    assert_eq!(capture.body["contents"][0]["parts"][0]["text"], "hello");
    assert!(events.iter().any(|event| matches!(
        event,
        CanonicalLlmEvent::Finish { reason: FinishReason::Stop, usage: Some(usage), .. }
            if usage.total_tokens == Some(3)
    )));
}

#[test]
fn vertex_endpoint_defaults_follow_regional_and_global_google_hosts() {
    assert_eq!(
        vertex_gemini_base_url(None, Some("project-a"), Some("europe-west4"))
            .expect("regional Vertex endpoint"),
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/project-a/locations/europe-west4/publishers/google"
    );
    assert_eq!(
        vertex_gemini_base_url(None, Some("project-a"), Some("global"))
            .expect("global Vertex endpoint"),
        "https://aiplatform.googleapis.com/v1/projects/project-a/locations/global/publishers/google"
    );
}

#[tokio::test]
async fn vertex_missing_context_and_unresolved_endpoint_fail_before_network() {
    for (project, location) in [
        (None, Some("us-central1")),
        (Some("project-a"), None),
        (Some("project-a"), Some("../global")),
    ] {
        assert!(vertex_gemini_base_url(None, project, location).is_err());
    }
    assert!(vertex_gemini_base_url(
        Some("https://example.com/proxy"),
        Some("project-a"),
        Some("us-central1")
    )
    .is_err());

    let client = CurrentProviderClient::new(vertex_config(None)).expect("Vertex client");
    let error = match client
        .stream(CurrentProviderRequest::new(vec![
            CurrentProviderMessage::user(vec![CurrentProviderContent::Text("hello".to_string())]),
        ]))
        .await
    {
        Ok(_) => panic!("unresolved Vertex endpoint must fail before network"),
        Err(error) => error,
    };
    assert!(error.message.contains("resolved project endpoint"));
}

async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 2048];
    let mut expected_length = None;
    loop {
        let read = stream.read(&mut buffer).await.expect("read Vertex request");
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
        if expected_length.is_none() {
            let headers = String::from_utf8_lossy(&request[..header_end]);
            expected_length = headers.lines().find_map(|line| {
                line.split_once(':').and_then(|(name, value)| {
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
            });
        }
        if expected_length.is_some_and(|length| request.len() >= header_end + length) {
            break;
        }
    }
    request
}
