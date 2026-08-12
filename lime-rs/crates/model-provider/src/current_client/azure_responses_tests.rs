use super::*;
use futures::StreamExt;
use serde_json::{json, Value};
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

fn azure_config(base_url: String) -> RuntimeProviderConfig {
    RuntimeProviderConfig {
        provider_name: "azure".to_string(),
        provider_selector: Some("azure-openai".to_string()),
        model_name: "gpt-5.4".to_string(),
        api_key: Some("azure-test-key".to_string()),
        auth: RuntimeProviderAuth::ApiKey,
        base_url: Some(base_url),
        api_version: Some("2025-04-01-preview".to_string()),
        credential_uuid: "azure-credential".to_string(),
        reasoning_effort: None,
        service_tier: None,
        protocol: Some(RuntimeProviderProtocol::AzureResponses),
        supports_websockets: true,
        toolshim: false,
        toolshim_model: None,
    }
}

async fn spawn_azure_fixture() -> (
    String,
    oneshot::Receiver<HttpCapture>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind Azure Responses fixture");
    let address = listener.local_addr().expect("Azure fixture address");
    let (capture_tx, capture_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept Azure request");
        let request = read_http_request(&mut stream).await;
        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("Azure request headers")
            + 4;
        let headers = String::from_utf8_lossy(&request[..header_end]).into_owned();
        let path = headers
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or_default()
            .to_string();
        let body = serde_json::from_slice(&request[header_end..]).expect("Azure request body");
        let _ = capture_tx.send(HttpCapture {
            path,
            headers,
            body,
        });

        let response_body = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"message-1\",\"delta\":\"done\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-azure\",\"output\":[],\"usage\":{\"input_tokens\":8,\"output_tokens\":1,\"total_tokens\":9}}}\n\n",
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
            response_body.len()
        );
        stream
            .write_all(response.as_bytes())
            .await
            .expect("write Azure response");
        stream.shutdown().await.expect("close Azure response");
    });
    (
        format!("http://{address}/openai/v1?region=west"),
        capture_rx,
        server,
    )
}

async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 2048];
    let mut expected_length = None;
    loop {
        let read = stream.read(&mut buffer).await.expect("read Azure request");
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
                .expect("Azure request content length")
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
    .with_tools(vec![CurrentProviderTool::function(
        "read_file",
        "Read a file",
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        }),
    )])
}

#[tokio::test]
async fn azure_uses_responses_query_api_key_and_canonical_stream() {
    let (base_url, capture, server) = spawn_azure_fixture().await;
    let client = CurrentProviderClient::with_client(
        azure_config(base_url),
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("Azure fixture HTTP client"),
    );

    assert!(!client.responses_websocket_enabled());
    let events = client
        .stream(request_with_tool_history())
        .await
        .expect("open Azure Responses stream")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("collect Azure Responses events");
    let capture = capture.await.expect("capture Azure Responses request");
    server.await.expect("join Azure Responses fixture");

    assert_eq!(
        capture.path,
        "/openai/v1/responses?region=west&api-version=2025-04-01-preview"
    );
    let headers = capture.headers.to_ascii_lowercase();
    assert!(headers.contains("\r\ncontent-type: application/json\r\n"));
    assert!(headers.contains("\r\naccept: text/event-stream\r\n"));
    assert!(headers.contains("\r\napi-key: azure-test-key\r\n"));
    assert!(!headers.contains("\r\nauthorization:"));
    assert_eq!(capture.body["model"], "gpt-5.4");
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
            usage: Some(usage),
            response_id: Some(response_id),
            ..
        } if response_id == "resp-azure"
            && usage.input_tokens == Some(8)
            && usage.output_tokens == Some(1)
            && usage.total_tokens == Some(9)
    )));
}

#[test]
fn azure_endpoint_uses_typed_api_version_without_duplicate_query() {
    assert_eq!(
        azure_responses_endpoint(
            "https://resource.openai.azure.com/openai/v1?api-version=2026-01-01",
            Some("2027-01-01"),
        )
        .expect("Azure Responses endpoint"),
        "https://resource.openai.azure.com/openai/v1/responses?api-version=2027-01-01"
    );
    assert_eq!(
        azure_responses_endpoint("https://resource.openai.azure.com", None)
            .expect("default Azure Responses endpoint"),
        "https://resource.openai.azure.com/openai/v1/responses?api-version=v1"
    );
    assert!(azure_responses_endpoint(
        "https://resource.openai.azure.com/openai/deployments/legacy",
        None,
    )
    .is_err());
}

#[test]
fn azure_endpoint_rejects_non_http_and_malformed_resource_urls() {
    for base_url in [
        "file:///openai/v1",
        "ftp://resource.openai.azure.com/openai/v1",
        "/openai/v1",
    ] {
        assert!(
            azure_responses_endpoint(base_url, None).is_err(),
            "base_url={base_url}"
        );
    }
}

#[tokio::test]
async fn azure_rejects_missing_api_key_before_network() {
    for (auth, api_key) in [
        (RuntimeProviderAuth::NoAuth, None),
        (RuntimeProviderAuth::ApiKey, Some("   ".to_string())),
    ] {
        let mut config = azure_config("http://127.0.0.1:9".to_string());
        config.auth = auth;
        config.api_key = api_key;
        let client = CurrentProviderClient::with_client(
            config,
            reqwest::Client::builder()
                .no_proxy()
                .build()
                .expect("Azure preflight HTTP client"),
        );

        let error = match client.stream(request_with_tool_history()).await {
            Ok(_) => panic!("Azure request without an API key must fail before network"),
            Err(error) => error,
        };
        assert!(
            error.message.contains("API-key") || error.message.contains("API key"),
            "error={error}"
        );
    }
}
