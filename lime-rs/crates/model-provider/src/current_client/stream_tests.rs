use super::{
    stream::{anthropic_sse, openai_chat_sse, responses_sse, ResponsesEventReducer},
    CurrentProviderError,
};
use futures::{Stream, StreamExt};
use reqwest::{Client, Response};
use runtime_core::CanonicalLlmEvent;
use std::pin::Pin;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::oneshot,
    time::{timeout, Duration},
};

type TestProviderStream =
    Pin<Box<dyn Stream<Item = Result<CanonicalLlmEvent, CurrentProviderError>> + Send + 'static>>;

async fn collect_openai_events(body: &'static str) -> Vec<CanonicalLlmEvent> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture server");
    let address = listener.local_addr().expect("fixture address");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = socket.read(&mut buffer).await.expect("read request");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
        }

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len(),
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");
    });

    let response = Client::builder()
        .no_proxy()
        .build()
        .expect("HTTP client")
        .get(format!("http://{address}"))
        .send()
        .await
        .expect("SSE response");
    let events = openai_chat_sse(response)
        .map(|event| event.expect("valid OpenAI-compatible SSE event"))
        .collect()
        .await;
    server.await.expect("fixture server");
    events
}

async fn collect_responses_events(body: &'static str) -> Vec<CanonicalLlmEvent> {
    collect_responses_events_with_headers(body, "", true).await
}

async fn collect_responses_events_with_headers(
    body: &'static str,
    extra_headers: &'static str,
    allow_model_verification: bool,
) -> Vec<CanonicalLlmEvent> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture server");
    let address = listener.local_addr().expect("fixture address");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = socket.read(&mut buffer).await.expect("read request");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
        }

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n{extra_headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len(),
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");
    });

    let response = Client::builder()
        .no_proxy()
        .build()
        .expect("HTTP client")
        .get(format!("http://{address}"))
        .send()
        .await
        .expect("SSE response");
    let events = responses_sse(response, allow_model_verification)
        .map(|event| event.expect("valid Responses SSE event"))
        .collect()
        .await;
    server.await.expect("fixture server");
    events
}

#[tokio::test]
async fn responses_projects_reported_models_from_headers_without_trusting_response_model() {
    let events = collect_responses_events_with_headers(
        concat!(
            "data: {\"type\":\"response.metadata\",\"headers\":{\"X-OpenAI-Model\":[\"gpt-5\"]}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"response\":{\"headers\":{\"OpenAI-Model\":\"gpt-5-mini\"}},\"headers\":{\"openai-model\":\"ignored-by-precedence\"},\"item\":{\"type\":\"message\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-model\",\"model\":\"must-not-be-trusted\",\"output\":[]}}\n\n",
        ),
        "OpenAI-Model: gpt-5\r\n",
        true,
    )
    .await;

    let models = events
        .iter()
        .filter_map(|event| match event {
            CanonicalLlmEvent::ServerModel { model } => Some(model.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(models, ["gpt-5", "gpt-5-mini"]);
}

#[tokio::test]
async fn responses_verification_is_typed_deduplicated_and_fail_closed() {
    let events = collect_responses_events(concat!(
        "data: {\"type\":\"response.output_item.added\",\"metadata\":{\"openai_verification_recommendation\":[\"trusted_access_for_cyber\"]},\"item\":{\"type\":\"message\"}}\n\n",
        "data: {\"type\":\"response.metadata\",\"metadata\":{\"openai_verification_recommendation\":[\"unknown\",\"trusted_access_for_cyber\",\"trusted_access_for_cyber\"]}}\n\n",
        "data: {\"type\":\"response.metadata\",\"metadata\":{\"openai_verification_recommendation\":[\"trusted_access_for_cyber\"]}}\n\n",
        "data: {\"type\":\"response.metadata\",\"metadata\":{\"openai_verification_recommendation\":\"trusted_access_for_cyber\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-verification\",\"output\":[]}}\n\n",
    ))
    .await;

    let verifications = events
        .iter()
        .filter_map(|event| match event {
            CanonicalLlmEvent::ModelVerification { verifications } => {
                Some(verifications.as_slice())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(verifications.len(), 1);
    assert_eq!(
        verifications[0],
        [runtime_core::ModelVerification::TrustedAccessForCyber]
    );
}

#[tokio::test]
async fn responses_verification_is_ignored_for_untrusted_routes() {
    let events = collect_responses_events_with_headers(
        concat!(
            "data: {\"type\":\"response.metadata\",\"metadata\":{\"openai_verification_recommendation\":[\"trusted_access_for_cyber\"]}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-untrusted\",\"output\":[]}}\n\n",
        ),
        "OpenAI-Verification-Recommendation: trusted_access_for_cyber\r\n",
        false,
    )
    .await;

    assert!(!events
        .iter()
        .any(|event| matches!(event, CanonicalLlmEvent::ModelVerification { .. })));
}

#[tokio::test]
async fn responses_projects_hosted_web_search_without_local_tool_finish_reason() {
    let events = collect_responses_events(concat!(
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"in_progress\",\"action\":{\"type\":\"search\",\"query\":\"Rust release\"}}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"query\":\"Rust release\"}}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"delta\":\"Rust 1.90\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-search\",\"output\":[]}}\n\n",
    ))
    .await;

    let calls = events
        .iter()
        .filter_map(|event| match event {
            CanonicalLlmEvent::ToolCall {
                id,
                name,
                input,
                provider_executed,
                provider_metadata,
                ..
            } => Some((id, name, input, provider_executed, provider_metadata)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "ws_1");
    assert_eq!(calls[0].1, "web_search");
    assert_eq!(calls[0].2["query"], "Rust release");
    assert_eq!(*calls[0].3, Some(true));
    assert_eq!(calls[0].4["raw_response_item"]["type"], "web_search_call");
    assert!(events.iter().any(|event| matches!(
        event,
        CanonicalLlmEvent::ToolResult {
            id,
            provider_executed: Some(true),
            ..
        } if id == "ws_1"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        CanonicalLlmEvent::Finish {
            reason: runtime_core::FinishReason::Stop,
            ..
        }
    )));
}

#[tokio::test]
async fn responses_projects_hosted_image_generation_exactly_once_and_finishes_stop() {
    let events = collect_responses_events(concat!(
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"ig_1\",\"type\":\"image_generation_call\",\"status\":\"in_progress\",\"revised_prompt\":\"a blue square\"}}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"ig_1\",\"type\":\"image_generation_call\",\"status\":\"completed\",\"revised_prompt\":\"a blue square\",\"result\":\"Zm9v\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-image\",\"output\":[{\"id\":\"ig_1\",\"type\":\"image_generation_call\",\"status\":\"completed\",\"revised_prompt\":\"a blue square\",\"result\":\"Zm9v\"}]}}\n\n",
    ))
    .await;

    let calls = events
        .iter()
        .filter_map(|event| match event {
            CanonicalLlmEvent::ToolCall {
                id,
                name,
                input,
                provider_executed,
                provider_metadata,
                ..
            } => Some((id, name, input, provider_executed, provider_metadata)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "ig_1");
    assert_eq!(calls[0].1, "image_generation");
    assert_eq!(calls[0].2["revised_prompt"], "a blue square");
    assert_eq!(*calls[0].3, Some(true));
    assert_eq!(
        calls[0].4["raw_response_item"]["type"],
        "image_generation_call"
    );

    let results = events
        .iter()
        .filter_map(|event| match event {
            CanonicalLlmEvent::ToolResult {
                id,
                result: runtime_core::ToolResultValue::Json { value },
                provider_executed: Some(true),
                ..
            } => Some((id, value)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "ig_1");
    assert_eq!(results[0].1["result"], "Zm9v");
    assert!(events.iter().any(|event| matches!(
        event,
        CanonicalLlmEvent::Finish {
            reason: runtime_core::FinishReason::Stop,
            ..
        }
    )));
}

#[test]
fn responses_rejects_completed_image_generation_without_result() {
    let mut reducer = ResponsesEventReducer::new(None, true);
    let error = match reducer.push(&serde_json::json!({
        "type": "response.output_item.done",
        "item": {
            "id": "ig_missing",
            "type": "image_generation_call",
            "status": "completed",
            "revised_prompt": "missing bytes"
        }
    })) {
        Ok(_) => panic!("completed image generation must include result"),
        Err(error) => error,
    };

    assert!(error.message.contains("completed without a string result"));
}

async fn assert_finish_releases_http_body(
    body: &'static str,
    stream_from_response: impl FnOnce(Response) -> TestProviderStream,
) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture server");
    let address = listener.local_addr().expect("fixture address");
    let (peer_closed_tx, peer_closed_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = socket.read(&mut buffer).await.expect("read request");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
        }

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n{:X}\r\n{body}\r\n",
            body.len(),
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");

        let peer_closed = socket.read(&mut buffer).await.expect("read peer close") == 0;
        let _ = peer_closed_tx.send(peer_closed);
    });

    let client = Client::builder()
        .no_proxy()
        .pool_idle_timeout(Duration::from_secs(300))
        .build()
        .expect("HTTP client");
    let response = client
        .get(format!("http://{address}"))
        .send()
        .await
        .expect("SSE response");
    let mut stream = stream_from_response(response);
    while let Some(event) = stream.next().await {
        if matches!(
            event.expect("provider event"),
            CanonicalLlmEvent::Finish { .. }
        ) {
            break;
        }
    }

    assert!(timeout(Duration::from_secs(2), peer_closed_rx)
        .await
        .expect("provider connection should close before another stream poll")
        .expect("peer close signal"));
    server.await.expect("fixture server");
}

#[tokio::test]
async fn openai_finish_releases_http_body_before_consumer_polls_again() {
    assert_finish_releases_http_body(
        concat!(
            "data: {\"id\":\"chatcmpl-close\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        ),
        |response| Box::pin(openai_chat_sse(response)),
    )
    .await;
}

#[tokio::test]
async fn openai_chunk_without_id_keeps_valid_delta_and_finish_reason() {
    let events = collect_openai_events(concat!(
        "data: {\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5.6-sol\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"你好\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    ))
    .await;

    assert_eq!(events.len(), 4);
    assert!(matches!(
        &events[0],
        CanonicalLlmEvent::TextStart { id } if id == "text-0"
    ));
    assert!(matches!(
        &events[1],
        CanonicalLlmEvent::TextDelta { id, text } if id == "text-0" && text == "你好"
    ));
    assert!(matches!(
        &events[2],
        CanonicalLlmEvent::TextEnd { id } if id == "text-0"
    ));
    assert!(matches!(
        &events[3],
        CanonicalLlmEvent::Finish {
            response_id: None,
            ..
        }
    ));
}

#[tokio::test]
async fn openai_null_id_does_not_clear_an_id_from_an_earlier_chunk() {
    let events = collect_openai_events(concat!(
        "data: {\"id\":\"chatcmpl-keep\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":null,\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    ))
    .await;

    assert!(matches!(
        events.last(),
        Some(CanonicalLlmEvent::Finish {
            response_id: Some(response_id),
            ..
        }) if response_id == "chatcmpl-keep"
    ));
}

#[tokio::test]
async fn openai_finish_reason_is_terminal_without_done_sentinel() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture server");
    let address = listener.local_addr().expect("fixture address");
    let (peer_closed_tx, peer_closed_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = socket.read(&mut buffer).await.expect("read request");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
        }

        let body = concat!(
            "data: {\"id\":\"chatcmpl-finish-only\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"id\":\"chatcmpl-finish-only\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"fixture\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n",
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n{:X}\r\n{body}\r\n",
            body.len(),
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");

        let peer_closed = socket.read(&mut buffer).await.expect("read peer close") == 0;
        let _ = peer_closed_tx.send(peer_closed);
    });

    let client = Client::builder()
        .no_proxy()
        .pool_idle_timeout(Duration::from_secs(300))
        .build()
        .expect("HTTP client");
    let response = client
        .get(format!("http://{address}"))
        .send()
        .await
        .expect("SSE response");
    let mut stream = Box::pin(openai_chat_sse(response));
    let terminal = timeout(Duration::from_secs(1), async {
        while let Some(event) = stream.next().await {
            if matches!(
                event.expect("provider event"),
                CanonicalLlmEvent::Finish {
                    usage: Some(usage),
                    ..
                } if usage.input_tokens == Some(7) && usage.output_tokens == Some(3)
            ) {
                return true;
            }
        }
        false
    })
    .await
    .expect("finish_reason should terminate the stream without [DONE]");
    drop(stream);

    assert!(terminal);
    assert!(timeout(Duration::from_secs(2), peer_closed_rx)
        .await
        .expect("provider connection should close after finish_reason")
        .expect("peer close signal"));
    server.await.expect("fixture server");
}

#[tokio::test]
async fn openai_node_style_chunked_keep_alive_finishes_after_done_sentinel() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture server");
    let address = listener.local_addr().expect("fixture address");
    let body = concat!(
        "data: {\"id\":\"chatcmpl-node\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"DESKTOP_GO_TASK_VISIBLE\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-node\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
        "data: [DONE]\n\n",
    );
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = socket.read(&mut buffer).await.expect("read request");
            if read == 0 {
                return;
            }
            request.extend_from_slice(&buffer[..read]);
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n{:X}\r\n{body}\r\n0\r\n\r\n",
            body.len(),
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");
        tokio::time::sleep(Duration::from_secs(2)).await;
    });

    let response = Client::builder()
        .no_proxy()
        .pool_idle_timeout(Duration::from_secs(300))
        .build()
        .expect("HTTP client")
        .get(format!("http://{address}"))
        .send()
        .await
        .expect("SSE response");
    let mut stream = Box::pin(openai_chat_sse(response));
    let events = timeout(Duration::from_secs(1), async {
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event.expect("provider event"));
        }
        events
    })
    .await
    .expect("chunked keep-alive response should finish");

    assert!(events.iter().any(|event| {
        matches!(
            event,
            CanonicalLlmEvent::TextDelta { text, .. }
                if text == "DESKTOP_GO_TASK_VISIBLE"
        )
    }));
    assert!(events
        .iter()
        .any(|event| matches!(event, CanonicalLlmEvent::Finish { .. })));
    server.await.expect("fixture server");
}

#[tokio::test]
async fn responses_finish_releases_http_body_before_consumer_polls_again() {
    assert_finish_releases_http_body(
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-close\",\"output\":[]}}\n\n",
        |response| Box::pin(responses_sse(response, true)),
    )
    .await;
}

#[tokio::test]
async fn responses_separates_reasoning_summary_from_raw_content() {
    let events = collect_responses_events(concat!(
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"reasoning\",\"id\":\"reasoning-wire\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_part.added\",\"output_index\":0,\"summary_index\":0}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"摘要\",\"summary_index\":0}\n\n",
        "data: {\"type\":\"response.reasoning_text.delta\",\"output_index\":0,\"delta\":\"原始推理\",\"content_index\":0}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-reasoning\",\"output\":[]}}\n\n",
    ))
    .await;

    assert!(matches!(
        events.as_slice(),
        [
            CanonicalLlmEvent::ReasoningStart { id: start_id },
            CanonicalLlmEvent::ReasoningSummaryPartAdded { id: part_id, summary_index: part_index },
            CanonicalLlmEvent::ReasoningSummaryDelta { id: summary_id, text: summary, summary_index },
            CanonicalLlmEvent::ReasoningContentDelta { id: content_id, text: content, content_index },
            CanonicalLlmEvent::ReasoningEnd { id: end_id },
            CanonicalLlmEvent::Finish { .. },
        ] if start_id == "reasoning-reasoning-wire"
            && part_id == "reasoning-reasoning-wire"
            && *part_index == 0
            && summary_id == "reasoning-reasoning-wire"
            && summary == "摘要"
            && *summary_index == 0
            && content_id == "reasoning-reasoning-wire"
            && content == "原始推理"
            && *content_index == 0
            && end_id == "reasoning-reasoning-wire"
    ));
}

#[tokio::test]
async fn responses_reasoning_deltas_without_indexes_are_ignored() {
    let events = collect_responses_events(concat!(
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":\"1\",\"delta\":\"摘要\"}\n\n",
        "data: {\"type\":\"response.reasoning_text.delta\",\"item_id\":\"1\",\"delta\":\"原始推理\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-reasoning\",\"output\":[]}}\n\n",
    ))
    .await;

    assert!(matches!(
        events.as_slice(),
        [CanonicalLlmEvent::Finish { .. }]
    ));
}

#[tokio::test]
async fn responses_eof_before_completed_fails_closed_without_finish() {
    let events = collect_responses_events(
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"message-1\",\"delta\":\"partial\"}\n\n",
    )
    .await;

    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, CanonicalLlmEvent::ProviderError { .. }))
            .count(),
        1
    );
    assert!(events.iter().any(|event| matches!(
        event,
        CanonicalLlmEvent::ProviderError { message, .. }
            if message == "stream closed before response.completed"
    )));
    assert!(!events
        .iter()
        .any(|event| matches!(event, CanonicalLlmEvent::Finish { .. })));
}

#[tokio::test]
async fn anthropic_finish_releases_http_body_before_consumer_polls_again() {
    assert_finish_releases_http_body("data: {\"type\":\"message_stop\"}\n\n", |response| {
        Box::pin(anthropic_sse(response))
    })
    .await;
}
