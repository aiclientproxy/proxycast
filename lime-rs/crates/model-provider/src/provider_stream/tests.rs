use super::*;
use crate::runtime_provider::{RuntimeProviderConfig, RuntimeProviderProtocol};
use crate::safety::{
    ProviderSafetyBufferingRetryModelSource, SAFETY_BUFFERING_ENABLED_HEADER,
    SAFETY_BUFFERING_FASTER_MODEL_HEADER,
};
use agent_protocol::provider_trace::ProviderTraceEvent;
use serde_json::json;

fn runtime_config() -> RuntimeProviderConfig {
    RuntimeProviderConfig {
        provider_name: "openai".to_string(),
        provider_selector: Some("codex".to_string()),
        model_name: "gpt-5.3-codex".to_string(),
        api_key: None,
        auth: crate::runtime_provider::RuntimeProviderAuth::ApiKey,
        base_url: Some("https://example.com/openai".to_string()),
        credential_uuid: "credential-1".to_string(),
        reasoning_effort: Some("medium".to_string()),
        service_tier: None,
        protocol: Some(RuntimeProviderProtocol::Responses),
        supports_websockets: true,
        toolshim: true,
        toolshim_model: Some("gpt-4o-mini".to_string()),
    }
}

#[test]
fn provider_handle_projects_runtime_config_without_provider_trait() {
    let handle =
        RuntimeReplyProviderHandle::from_config(&runtime_config(), RuntimeProviderBackend::Current)
            .with_capabilities(RuntimeReplyProviderCapabilities {
                supports_streaming: true,
                supports_embeddings: false,
                active_model_name: Some("gpt-5.3-codex".to_string()),
            });

    assert_eq!(handle.provider_name(), "openai");
    assert_eq!(handle.model_name(), "gpt-5.3-codex");
    assert_eq!(handle.backend, RuntimeProviderBackend::Current);
    assert_eq!(
        handle.identity.protocol,
        Some(ModelProviderProtocol::Responses)
    );
    assert!(handle.capabilities.supports_streaming);
}

#[test]
fn provider_handle_projects_provider_trace_metadata_without_provider_trait() {
    let handle =
        RuntimeReplyProviderHandle::from_config(&runtime_config(), RuntimeProviderBackend::Current)
            .with_capabilities(RuntimeReplyProviderCapabilities {
                supports_streaming: true,
                supports_embeddings: false,
                active_model_name: Some("gpt-5.3-codex-active".to_string()),
            });
    let mut event = ProviderTraceEvent::request_started("", "", 1);

    apply_runtime_provider_metadata(&mut event, Some(&handle));

    assert_eq!(event.provider, "openai");
    assert_eq!(event.model, "gpt-5.3-codex");
    assert_eq!(event.runtime_provider_backend.as_deref(), Some("current"));
    assert_eq!(event.runtime_provider_selector.as_deref(), Some("codex"));
    assert_eq!(
        event.runtime_provider_protocol.as_deref(),
        Some("responses")
    );
    assert_eq!(
        event.runtime_provider_active_model.as_deref(),
        Some("gpt-5.3-codex-active")
    );
}

#[test]
fn provider_trace_metadata_keeps_existing_provider_and_model() {
    let handle =
        RuntimeReplyProviderHandle::from_config(&runtime_config(), RuntimeProviderBackend::Current);
    let mut event = ProviderTraceEvent::request_started("anthropic", "claude", 1);

    handle
        .provider_trace_metadata()
        .apply_to_provider_trace_event(&mut event);

    assert_eq!(event.provider, "anthropic");
    assert_eq!(event.model, "claude");
    assert_eq!(event.runtime_provider_backend.as_deref(), Some("current"));
}

#[test]
fn provider_binding_keeps_handle_with_backend_without_provider_trait_object() {
    let handle =
        RuntimeReplyProviderHandle::from_config(&runtime_config(), RuntimeProviderBackend::Current);
    let binding = RuntimeReplyProviderBinding::new(handle.clone(), "backend-marker".to_string());

    assert_eq!(binding.handle(), &handle);
    assert_eq!(binding.backend(), "backend-marker");

    let (actual_handle, backend) = binding.into_parts();
    assert_eq!(actual_handle, handle);
    assert_eq!(backend, "backend-marker");
}

#[test]
fn provider_response_event_contract_carries_response_items_without_turn_owner() {
    let item = RuntimeReplyResponseItem::new(
        "call-1",
        "tool_call",
        RuntimeReplyResponseItemPayload::ToolCall {
            tool_name: "apply_patch".to_string(),
            arguments: Some(json!({ "patch": "*** Begin Patch" })),
            output: None,
            success: None,
            error: None,
            metadata: Some(json!({ "source": "provider_response" })),
        },
    );

    let event = RuntimeReplyResponseEvent::OutputItemAdded { item: item.clone() };

    assert_eq!(event, RuntimeReplyResponseEvent::OutputItemAdded { item });
}

#[test]
fn provider_response_content_projects_text_notification_and_tool_delta_without_agent_message() {
    assert_eq!(
        provider_stream_response_text_chars([
            RuntimeReplyProviderResponseContent::text("  "),
            RuntimeReplyProviderResponseContent::text(" hello "),
        ]),
        Some(5)
    );

    let notification = provider_stream_notification_text(
        RuntimeReplyProviderStreamEvent::NOTIFICATION_KIND_SAFETY_BUFFERING,
        json!({"type": "response.in_progress"}),
        Vec::new(),
    );
    assert!(provider_stream_response_has_notification_text([
        RuntimeReplyProviderResponseContent::text("plain"),
        RuntimeReplyProviderResponseContent::system_notification(notification.as_str()),
    ]));

    let events = provider_stream_response_tool_input_delta_events([
        RuntimeReplyProviderResponseContent::tool_input_delta(
            "call-1",
            Some("apply_patch"),
            "{\"patch\"",
            Some("{\"patch\""),
            Some("openai"),
        ),
    ])
    .expect("tool input delta events");
    assert!(matches!(
        events.as_slice(),
        [RuntimeReplyResponseEvent::ToolCallInputDelta {
            call_id,
            tool_name,
            delta,
            accumulated_arguments,
            provider,
        }] if call_id == "call-1"
            && tool_name.as_deref() == Some("apply_patch")
            && delta == "{\"patch\""
            && accumulated_arguments.as_deref() == Some("{\"patch\"")
            && provider.as_deref() == Some("openai")
    ));
}

#[test]
fn provider_response_content_keeps_mixed_tool_delta_messages_fail_closed() {
    assert!(provider_stream_response_tool_input_delta_events([
        RuntimeReplyProviderResponseContent::tool_input_delta(
            "call-1",
            Some("web_search"),
            "{\"query\"",
            Some("{\"query\""),
            Some("openai"),
        ),
        RuntimeReplyProviderResponseContent::text("final answer"),
    ])
    .is_none());
}

#[test]
fn provider_direct_answer_policy_strips_structured_tool_requests_without_agent_message() {
    assert!(provider_stream_direct_answer_should_bypass_tool_execution(
        true, false,
    ));
    assert!(!provider_stream_direct_answer_should_bypass_tool_execution(
        true, true,
    ));
    assert!(!provider_stream_direct_answer_should_bypass_tool_execution(
        false, false,
    ));
    assert!(provider_stream_direct_answer_should_strip_response_content(
        RuntimeReplyProviderResponseContent::structured_tool_request(),
    ));
    assert!(
        !provider_stream_direct_answer_should_strip_response_content(
            RuntimeReplyProviderResponseContent::text("answer"),
        )
    );
    assert!(
        !provider_stream_direct_answer_should_strip_response_content(
            RuntimeReplyProviderResponseContent::Other,
        )
    );
}

#[test]
fn provider_response_route_prioritizes_provider_stream_side_channels() {
    let notification = provider_stream_notification_text(
        RuntimeReplyProviderStreamEvent::NOTIFICATION_KIND_SAFETY_BUFFERING,
        json!({"type": "response.in_progress"}),
        Vec::new(),
    );

    let route = provider_stream_response_route(
        [RuntimeReplyProviderResponseContent::tool_input_delta(
            "call-1",
            Some("apply_patch"),
            "{\"patch\"",
            Some("{\"patch\""),
            Some("openai"),
        )],
        true,
        false,
    );
    assert!(matches!(
        route,
        RuntimeReplyProviderResponseRoute::ToolInputDeltaEvents(events)
            if matches!(
                events.as_slice(),
                [RuntimeReplyResponseEvent::ToolCallInputDelta { call_id, .. }]
                    if call_id == "call-1"
            )
    ));

    assert!(matches!(
        provider_stream_response_route(
            [
                RuntimeReplyProviderResponseContent::text("plain"),
                RuntimeReplyProviderResponseContent::system_notification(notification.as_str()),
            ],
            true,
            false,
        ),
        RuntimeReplyProviderResponseRoute::Notification
    ));
    assert!(matches!(
        provider_stream_response_route(
            [
                RuntimeReplyProviderResponseContent::text("answer"),
                RuntimeReplyProviderResponseContent::structured_tool_request(),
            ],
            true,
            false,
        ),
        RuntimeReplyProviderResponseRoute::DirectAnswer
    ));
    assert!(matches!(
        provider_stream_response_route(
            [
                RuntimeReplyProviderResponseContent::text("answer"),
                RuntimeReplyProviderResponseContent::structured_tool_request(),
            ],
            true,
            true,
        ),
        RuntimeReplyProviderResponseRoute::ToolExecution
    ));
}

#[test]
fn provider_response_outcome_tracks_progress_model_change_and_route_without_agent_loop() {
    let mut session = RuntimeReplyProviderResponseSession::new();
    let notification = provider_stream_notification_text(
        RuntimeReplyProviderStreamEvent::NOTIFICATION_KIND_SAFETY_BUFFERING,
        json!({"type": "response.in_progress"}),
        Vec::new(),
    );

    let first = session.accept_response(
        Some([
            RuntimeReplyProviderResponseContent::text("  hello "),
            RuntimeReplyProviderResponseContent::system_notification(notification.as_str()),
        ]),
        Some("gpt-worker"),
        Some(RuntimeReplyProviderLeadWorkerModels::new(
            "gpt-lead",
            "gpt-worker",
        )),
        false,
        true,
    );

    assert!(first.first_event_received);
    assert_eq!(first.first_text_delta_chars, Some(5));
    assert_eq!(
        first.model_change,
        Some(RuntimeReplyProviderModelChange {
            model: "gpt-worker".to_string(),
            mode: RuntimeReplyProviderModelChangeMode::Worker,
        })
    );
    assert!(matches!(
        first.route,
        Some(RuntimeReplyProviderResponseRoute::Notification)
    ));
    assert!(session.stream_progress().first_event_seen());

    let second = session.accept_response(
        Some([RuntimeReplyProviderResponseContent::text("second")]),
        Some("gpt-lead"),
        Some(RuntimeReplyProviderLeadWorkerModels::new(
            "gpt-lead",
            "gpt-worker",
        )),
        false,
        false,
    );

    assert!(!second.first_event_received);
    assert_eq!(second.first_text_delta_chars, None);
    assert_eq!(
        second.model_change,
        Some(RuntimeReplyProviderModelChange {
            model: "gpt-lead".to_string(),
            mode: RuntimeReplyProviderModelChangeMode::Lead,
        })
    );
    assert!(matches!(
        second.route,
        Some(RuntimeReplyProviderResponseRoute::ToolExecution)
    ));
}

#[test]
fn provider_response_outcome_keeps_usage_only_event_without_response_route() {
    let mut session = RuntimeReplyProviderResponseSession::new();

    let outcome = session.accept_response(
        Option::<[RuntimeReplyProviderResponseContent<'_>; 0]>::None,
        Some("gpt-other"),
        Some(RuntimeReplyProviderLeadWorkerModels::new(
            "gpt-lead",
            "gpt-worker",
        )),
        true,
        false,
    );

    assert!(outcome.first_event_received);
    assert_eq!(outcome.first_text_delta_chars, None);
    assert_eq!(
        outcome.model_change,
        Some(RuntimeReplyProviderModelChange {
            model: "gpt-other".to_string(),
            mode: RuntimeReplyProviderModelChangeMode::Unknown,
        })
    );
    assert_eq!(outcome.route, None);
}

#[test]
fn provider_sampling_session_accepts_response_text_delta_once_without_agent_loop() {
    let request =
        RuntimeReplyProviderSamplingRequest::new("openai", "gpt-5-codex", 1, 0, 32, None, true);
    let mut session = RuntimeReplyProviderSamplingSession::start(request);

    assert_eq!(
        session.accept_response_text_delta([
            RuntimeReplyProviderResponseContent::text("  "),
            RuntimeReplyProviderResponseContent::text(" hello "),
        ],),
        Some(5),
    );
    assert!(session.stream_progress().first_text_delta_seen());
    assert_eq!(
        session.accept_response_text_delta([RuntimeReplyProviderResponseContent::text("second")],),
        None,
    );
}

#[test]
fn plaintext_tool_use_stream_projects_split_chunks_without_agent_message_state() {
    let mut stream = RuntimeReplyProviderPlaintextToolUseStream::default();

    let first = stream.push_text("Before <tool_use name=\"WebSearch\">{\"query\":\"");
    assert!(stream.is_pending());
    assert!(matches!(
        first.as_slice(),
        [
            RuntimeReplyProviderPlaintextToolUseStreamEvent::Text(prefix),
            RuntimeReplyProviderPlaintextToolUseStreamEvent::ToolInputDelta(progress),
        ] if prefix == "Before"
            && progress.tool_name.as_deref() == Some("WebSearch")
            && progress.delta == "{\"query\":\""
    ));

    let second = stream.push_text("rust\"}</tool_use>");
    assert!(!stream.is_pending());
    assert!(matches!(
        second.as_slice(),
        [RuntimeReplyProviderPlaintextToolUseStreamEvent::ToolUse(tool_use)]
            if tool_use.prefix.is_empty()
                && tool_use.tool_calls.len() == 1
                && tool_use.tool_calls[0].name == "WebSearch"
                && tool_use.tool_calls[0].arguments.as_ref()
                    == Some(&serde_json::Map::from_iter([(
                        "query".to_string(),
                        json!("rust"),
                    )]))
    ));
}

#[test]
fn plaintext_tool_use_stream_keeps_plain_text_and_flushes_incomplete_markup() {
    let mut stream = RuntimeReplyProviderPlaintextToolUseStream::default();

    assert_eq!(
        stream.push_text("plain answer"),
        vec![RuntimeReplyProviderPlaintextToolUseStreamEvent::Text(
            "plain answer".to_string()
        )]
    );

    stream.push_text("<tool_use name=\"WebSearch\">{\"query\":");
    assert_eq!(
        stream.finish(),
        Some(RuntimeReplyProviderPlaintextToolUseStreamEvent::Text(
            "<tool_use name=\"WebSearch\">{\"query\":".to_string()
        ))
    );
    assert!(!stream.is_pending());
}

#[test]
fn plaintext_tool_use_stream_normalizes_inline_search_call() {
    let mut stream = RuntimeReplyProviderPlaintextToolUseStream::default();

    let events = stream.push_text("<Search query=\"codex runtime\" />");

    assert!(matches!(
        events.as_slice(),
        [RuntimeReplyProviderPlaintextToolUseStreamEvent::ToolUse(tool_use)]
            if tool_use.tool_calls.len() == 1
                && tool_use.tool_calls[0].name == "WebSearch"
                && tool_use.tool_calls[0].arguments.as_ref()
                    == Some(&serde_json::Map::from_iter([(
                        "query".to_string(),
                        json!("codex runtime"),
                    )]))
    ));
}

#[test]
fn provider_stream_response_context_extracts_request_id_without_reqwest() {
    let context = provider_stream_response_context_from_header_pairs([
        ("content-type", "application/json"),
        ("X-OAI-Request-ID", " req_provider_123 "),
    ])
    .expect("provider response context");

    assert_eq!(
        context.provider_request_id.as_deref(),
        Some("req_provider_123")
    );
    assert_eq!(
        context.provider_request_id_header.as_deref(),
        Some("x-oai-request-id")
    );
}

#[test]
fn provider_stream_response_context_rejects_empty_long_or_non_visible_ids() {
    assert!(
        provider_stream_response_context_from_header_pairs([("x-request-id", "   ")]).is_none()
    );
    assert!(provider_stream_response_context_from_header_pairs([(
        "x-request-id",
        "x".repeat(257),
    )])
    .is_none());
    assert!(
        provider_stream_response_context_from_header_pairs([("x-request-id", "req\nbad",)])
            .is_none()
    );
}

#[test]
fn provider_sampling_request_selects_streaming_mode_without_agent_agent() {
    let request = RuntimeReplyProviderSamplingRequest::new(
        "openai",
        "gpt-5-codex",
        3,
        2,
        128,
        Some("local_workspace".to_string()),
        true,
    );

    assert_eq!(
        request.sampling_mode(),
        RuntimeReplyProviderSamplingMode::Streaming
    );
    assert_eq!(request.provider_name, "openai");
    assert_eq!(request.model_name, "gpt-5-codex");
    assert_eq!(request.message_count, 3);
    assert_eq!(request.tool_count, 2);
    assert_eq!(request.system_chars, 128);
    assert_eq!(request.tool_surface.as_deref(), Some("local_workspace"));
}

#[test]
fn provider_sampling_request_selects_non_streaming_mode_without_agent_agent() {
    let request =
        RuntimeReplyProviderSamplingRequest::new("openai", "gpt-5-codex", 1, 0, 32, None, false);

    assert_eq!(
        request.sampling_mode(),
        RuntimeReplyProviderSamplingMode::NonStreaming
    );
}

#[tokio::test]
async fn provider_sampling_session_runs_streaming_branch_without_agent_agent() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let request =
        RuntimeReplyProviderSamplingRequest::new("openai", "gpt-5-codex", 1, 0, 32, None, true);
    let session = RuntimeReplyProviderSamplingSession::start(request);
    let stream_calls = Arc::new(AtomicUsize::new(0));
    let complete_calls = Arc::new(AtomicUsize::new(0));
    let stream_calls_for_open = stream_calls.clone();
    let complete_calls_for_complete = complete_calls.clone();

    let output = session
        .open_stream(
            || async move {
                stream_calls_for_open.fetch_add(1, Ordering::SeqCst);
                Ok::<_, &'static str>("stream".to_string())
            },
            || async move {
                complete_calls_for_complete.fetch_add(1, Ordering::SeqCst);
                Ok::<_, &'static str>("complete")
            },
            |value| format!("single:{value}"),
            |_| RuntimeReplyProviderSamplingFailureLogLevel::Warn,
        )
        .await
        .expect("streaming branch");

    assert_eq!(output, "stream");
    assert_eq!(stream_calls.load(Ordering::SeqCst), 1);
    assert_eq!(complete_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn provider_sampling_session_runs_non_streaming_branch_without_agent_agent() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let request =
        RuntimeReplyProviderSamplingRequest::new("openai", "gpt-5-codex", 1, 0, 32, None, false);
    let session = RuntimeReplyProviderSamplingSession::start(request);
    let stream_calls = Arc::new(AtomicUsize::new(0));
    let complete_calls = Arc::new(AtomicUsize::new(0));
    let stream_calls_for_open = stream_calls.clone();
    let complete_calls_for_complete = complete_calls.clone();

    let output = session
        .open_stream(
            || async move {
                stream_calls_for_open.fetch_add(1, Ordering::SeqCst);
                Ok::<_, &'static str>("stream".to_string())
            },
            || async move {
                complete_calls_for_complete.fetch_add(1, Ordering::SeqCst);
                Ok::<_, &'static str>("complete")
            },
            |value| format!("single:{value}"),
            |_| RuntimeReplyProviderSamplingFailureLogLevel::Warn,
        )
        .await
        .expect("non-streaming branch");

    assert_eq!(output, "single:complete");
    assert_eq!(stream_calls.load(Ordering::SeqCst), 0);
    assert_eq!(complete_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn provider_empty_first_content_retry_marker_is_current_policy() {
    assert!(provider_stream_should_retry_empty_first_content(
        false,
        "Anthropic stream ended without assistant content or tool call"
    ));
    assert!(!provider_stream_should_retry_empty_first_content(
        true,
        "Anthropic stream ended without assistant content or tool call"
    ));
    assert!(!provider_stream_should_retry_empty_first_content(
        false,
        "different provider error"
    ));
}

#[test]
fn provider_sampling_session_accepts_stream_item_and_retry_outcome_without_agent_stream() {
    let request =
        RuntimeReplyProviderSamplingRequest::new("anthropic", "claude", 1, 0, 32, None, true);
    let mut session = RuntimeReplyProviderSamplingSession::start(request);

    let retry = session.accept_stream_item::<String, (), _>(Err(
        "Anthropic stream ended without assistant content or tool call",
    ));
    assert!(matches!(
        retry,
        RuntimeReplyProviderSamplingStreamItem::RetryEmptyFirstContent(_)
    ));
    assert!(!session.stream_progress().first_content_seen());

    let item = session.accept_stream_item::<String, usize, &'static str>(Ok((
        Some("hello".to_string()),
        Some(42),
    )));
    assert_eq!(
        item,
        RuntimeReplyProviderSamplingStreamItem::Item {
            message: Some("hello".to_string()),
            usage: Some(42),
        }
    );
    assert!(session.stream_progress().first_content_seen());

    let error_after_content = session.accept_stream_item::<String, (), _>(Err(
        "Anthropic stream ended without assistant content or tool call",
    ));
    assert!(matches!(
        error_after_content,
        RuntimeReplyProviderSamplingStreamItem::Error(_)
    ));
}

#[test]
fn provider_stream_failure_logging_classifies_transient_and_internal_failures() {
    assert!(provider_stream_failure_should_log_as_error(
        RuntimeReplyProviderFailure::new(RuntimeReplyProviderFailureKind::Server, true, false)
    ));
    assert!(provider_stream_failure_should_log_as_error(
        RuntimeReplyProviderFailure::new(RuntimeReplyProviderFailureKind::Execution, false, false)
    ));
    assert!(provider_stream_failure_should_log_as_error(
        RuntimeReplyProviderFailure::new(RuntimeReplyProviderFailureKind::Usage, false, false)
    ));
    assert!(provider_stream_failure_should_log_as_error(
        RuntimeReplyProviderFailure::new(RuntimeReplyProviderFailureKind::Request, true, false)
    ));
    assert!(!provider_stream_failure_should_log_as_error(
        RuntimeReplyProviderFailure::new(
            RuntimeReplyProviderFailureKind::Authentication,
            false,
            true
        )
    ));
    assert!(!provider_stream_failure_should_log_as_error(
        RuntimeReplyProviderFailure::new(
            RuntimeReplyProviderFailureKind::ContextLength,
            false,
            false
        )
    ));
}

#[test]
fn provider_stream_failure_accepts_provider_telemetry_categories() {
    let failure = RuntimeReplyProviderFailure::from_category("server", true, false);
    assert_eq!(failure.kind, RuntimeReplyProviderFailureKind::Server);
    assert_eq!(failure.kind.as_category(), "server");
    assert!(provider_stream_failure_should_log_as_error(failure));

    let unknown = RuntimeReplyProviderFailure::from_category("provider_custom", false, false);
    assert_eq!(unknown.kind, RuntimeReplyProviderFailureKind::Unknown);
    assert_eq!(unknown.kind.as_category(), "unknown");
    assert!(!provider_stream_failure_should_log_as_error(unknown));
}

#[test]
fn provider_stream_failure_message_logging_keeps_provider_rejections_quiet() {
    assert!(!provider_stream_failure_message_should_log_as_warning(
        "Request failed: Bad request (400): 当前模型未在租户白名单中开放"
    ));
    assert!(!provider_stream_failure_message_should_log_as_warning(
        "Authentication error: invalid key"
    ));
    assert!(provider_stream_failure_message_should_log_as_warning(
        "connection failed"
    ));
    assert!(provider_stream_failure_message_should_log_as_warning(
        "Server error: temporarily unavailable"
    ));
}

#[test]
fn provider_stream_first_text_delta_chars_uses_first_non_empty_text() {
    assert_eq!(
        provider_stream_first_text_delta_chars(["", " \n\t", " 你好 "]),
        Some(2)
    );
    assert_eq!(provider_stream_first_text_delta_chars(["", " \n\t"]), None);
}

#[test]
fn provider_stream_tool_input_delta_events_projects_complete_delta_content() {
    let events = provider_stream_tool_input_delta_events([
        Some(RuntimeReplyProviderToolInputDelta::new(
            "call-1",
            Some("apply_patch"),
            "{\"patch\"",
            Some("{\"patch\""),
            Some("openai"),
        )),
        Some(RuntimeReplyProviderToolInputDelta::new(
            "call-1",
            Some("apply_patch"),
            ":\"...\"}",
            Some("{\"patch\":\"...\"}"),
            Some("openai"),
        )),
    ])
    .expect("tool input delta events");

    assert_eq!(
        events,
        vec![
            RuntimeReplyResponseEvent::ToolCallInputDelta {
                call_id: "call-1".to_string(),
                tool_name: Some("apply_patch".to_string()),
                delta: "{\"patch\"".to_string(),
                accumulated_arguments: Some("{\"patch\"".to_string()),
                provider: Some("openai".to_string()),
            },
            RuntimeReplyResponseEvent::ToolCallInputDelta {
                call_id: "call-1".to_string(),
                tool_name: Some("apply_patch".to_string()),
                delta: ":\"...\"}".to_string(),
                accumulated_arguments: Some("{\"patch\":\"...\"}".to_string()),
                provider: Some("openai".to_string()),
            },
        ]
    );
}

#[test]
fn provider_stream_tool_input_delta_events_rejects_mixed_or_empty_content() {
    assert_eq!(
        provider_stream_tool_input_delta_events([
            Some(RuntimeReplyProviderToolInputDelta::new(
                "call-1", None, "{}", None, None
            )),
            None,
        ]),
        None
    );
    assert_eq!(
        provider_stream_tool_input_delta_events(std::iter::empty()),
        None
    );
    assert_eq!(
        provider_stream_tool_input_delta_events([Some(RuntimeReplyProviderToolInputDelta::new(
            "", None, "{}", None, None
        ))]),
        Some(Vec::new())
    );
}

#[test]
fn provider_stream_progress_tracks_first_milestones_once() {
    let mut progress = RuntimeReplyProviderStreamProgress::new();

    assert!(!progress.first_event_seen());
    assert!(progress.note_first_event());
    assert!(progress.first_event_seen());
    assert!(!progress.note_first_event());

    assert!(!progress.first_content_seen());
    assert!(!progress.note_first_content(false));
    assert!(progress.should_retry_empty_first_content(PROVIDER_EMPTY_STREAM_RETRY_MARKER));
    assert!(progress.note_first_content(true));
    assert!(progress.first_content_seen());
    assert!(!progress.note_first_content(true));
    assert!(!progress.should_retry_empty_first_content(PROVIDER_EMPTY_STREAM_RETRY_MARKER));

    assert!(!progress.first_text_delta_seen());
    assert_eq!(progress.note_first_text_delta(None), None);
    assert_eq!(progress.note_first_text_delta(Some(4)), Some(4));
    assert!(progress.first_text_delta_seen());
    assert_eq!(progress.note_first_text_delta(Some(8)), None);
}

#[test]
fn provider_stream_model_change_classifies_active_lead_worker_model() {
    let lead = provider_stream_model_change("gpt-5-lead", "gpt-5-lead", "gpt-5-worker");
    assert_eq!(lead.model, "gpt-5-lead");
    assert_eq!(lead.mode, RuntimeReplyProviderModelChangeMode::Lead);
    assert_eq!(lead.mode.as_str(), "lead");

    let worker = provider_stream_model_change("gpt-5-worker", "gpt-5-lead", "gpt-5-worker");
    assert_eq!(worker.model, "gpt-5-worker");
    assert_eq!(worker.mode, RuntimeReplyProviderModelChangeMode::Worker);
    assert_eq!(worker.mode.as_str(), "worker");

    let unknown = provider_stream_model_change("gpt-5-other", "gpt-5-lead", "gpt-5-worker");
    assert_eq!(unknown.model, "gpt-5-other");
    assert_eq!(unknown.mode, RuntimeReplyProviderModelChangeMode::Unknown);
    assert_eq!(unknown.mode.as_str(), "unknown");
}

#[test]
fn provider_stream_notification_text_round_trips_current_prefix() {
    let text = provider_stream_notification_text(
        RuntimeReplyProviderStreamEvent::NOTIFICATION_KIND_SAFETY_BUFFERING,
        json!({
            "type": "response.output_text.delta",
            "safety_buffering": {
                "use_cases": ["cyber"],
                "reasons": ["policy"],
            },
        }),
        vec![(
            SAFETY_BUFFERING_ENABLED_HEADER.to_string(),
            "true".to_string(),
        )],
    );

    assert!(text.starts_with(PROVIDER_STREAM_EVENT_NOTIFICATION_PREFIX));
    assert!(!text.contains("__runtime_provider_stream_event__"));

    let payload =
        provider_stream_notification_payload_from_text(&text).expect("notification payload");

    assert_eq!(
        payload["eventKind"],
        json!(RuntimeReplyProviderStreamEvent::NOTIFICATION_KIND_SAFETY_BUFFERING)
    );
    assert_eq!(
        payload["responseEvent"]["safety_buffering"]["reasons"],
        json!(["policy"])
    );
    assert_eq!(
        payload["headers"][0]["name"],
        json!(SAFETY_BUFFERING_ENABLED_HEADER)
    );
    assert!(provider_stream_notification_payload_from_text(
        "__runtime_provider_stream_event__:{\"eventKind\":\"legacy\"}"
    )
    .is_none());
    assert!(provider_stream_has_notification_text([text.as_str()]));
    assert!(provider_stream_notification_payload_from_texts([text.as_str()]).is_some());
    assert!(!provider_stream_has_notification_text(["plain text"]));
}

#[test]
fn provider_stream_plaintext_tool_uses_extracts_xml_blocks_without_agent_message() {
    let parsed = provider_stream_plaintext_tool_uses(
        "我先做只读验证。\n\
        <tool_use name=\"mcp__system__shell\">{\"command\":\"pwd\"}</tool_use>\n\
        <tool_use name=\"mcp__system__read_file\">```json\n{\"path\":\"package.json\",\"head\":20}\n```</tool_use>\n\
        抱歉，当前环境没有暴露工具接口。",
    )
    .expect("plaintext tool use");

    assert_eq!(parsed.prefix.trim(), "我先做只读验证。");
    assert_eq!(parsed.tool_calls.len(), 2);
    assert_eq!(parsed.tool_calls[0].name, "mcp__system__shell");
    assert_eq!(
        parsed.tool_calls[0]
            .arguments
            .as_ref()
            .and_then(|arguments| arguments.get("command"))
            .and_then(|value| value.as_str()),
        Some("pwd")
    );
    assert_eq!(parsed.tool_calls[1].name, "mcp__system__read_file");
    assert_eq!(
        parsed.tool_calls[1]
            .arguments
            .as_ref()
            .and_then(|arguments| arguments.get("head"))
            .and_then(|value| value.as_i64()),
        Some(20)
    );
}

#[test]
fn provider_stream_plaintext_tool_uses_converts_search_aliases() {
    let web_search = provider_stream_plaintext_tool_uses(
        "我需要检索。\n<WebSearch query=\"2026年6月1日 国际新闻 今日 要闻\" />",
    )
    .expect("web search tag");
    assert_eq!(web_search.tool_calls[0].name, "WebSearch");
    assert_eq!(
        web_search.tool_calls[0]
            .arguments
            .as_ref()
            .and_then(|arguments| arguments.get("query"))
            .and_then(|value| value.as_str()),
        Some("2026年6月1日 国际新闻 今日 要闻")
    );

    let search =
        provider_stream_plaintext_tool_uses("<Search query=\"today international news\" />")
            .expect("search alias tag");
    assert_eq!(search.tool_calls[0].name, "WebSearch");
    assert_eq!(
        search.tool_calls[0]
            .arguments
            .as_ref()
            .and_then(|arguments| arguments.get("query"))
            .and_then(|value| value.as_str()),
        Some("today international news")
    );
}

#[test]
fn provider_stream_plaintext_tool_use_progress_tracks_split_xml_block() {
    let open = "<tool_use name=\"mcp__system__shell\">";
    let first_progress =
        provider_stream_plaintext_tool_use_progress(open, open).expect("first progress");
    assert_eq!(
        first_progress.tool_name.as_deref(),
        Some("mcp__system__shell")
    );
    assert_eq!(first_progress.delta, "");
    assert_eq!(first_progress.accumulated_arguments, None);

    let accumulated = format!("{open}{{\"command\":\"pwd\"}}");
    let second_progress =
        provider_stream_plaintext_tool_use_progress(&accumulated, "{\"command\":\"pwd\"}")
            .expect("second progress");
    assert_eq!(second_progress.delta, "{\"command\":\"pwd\"}");
    assert_eq!(
        second_progress.accumulated_arguments.as_deref(),
        Some("{\"command\":\"pwd\"}")
    );
    assert_eq!(
        provider_stream_plaintext_tool_use_start(&format!("prefix\n{open}")),
        Some("prefix\n".len())
    );
    assert!(!provider_stream_plaintext_tool_use_is_complete(
        &accumulated
    ));
    assert!(provider_stream_plaintext_tool_use_is_complete(&format!(
        "{accumulated}</tool_use>"
    )));
}

#[test]
fn stream_request_carries_current_provider_handle() {
    let handle =
        RuntimeReplyProviderHandle::from_config(&runtime_config(), RuntimeProviderBackend::Current);
    let request = RuntimeReplyStreamRequest::new(
        "session-1",
        RuntimeReplyInputKind::UserMessage,
        42,
        Some(handle),
    );

    assert_eq!(request.session_id, "session-1");
    assert_eq!(
        request.provider_backend(),
        Some(RuntimeProviderBackend::Current)
    );
    assert_eq!(request.provider_name(), Some("openai"));
    assert_eq!(request.model_name(), Some("gpt-5.3-codex"));
}

#[test]
fn provider_stream_start_accepts_matching_handle() {
    let handle =
        RuntimeReplyProviderHandle::from_config(&runtime_config(), RuntimeProviderBackend::Current);
    let request = RuntimeReplyStreamRequest::new(
        "session-1",
        RuntimeReplyInputKind::UserMessage,
        42,
        Some(handle.clone()),
    );

    let start =
        RuntimeReplyProviderStreamStart::new(request, &handle).expect("provider stream start");

    assert_eq!(start.stream_request().provider_name(), Some("openai"));
    assert_eq!(start.stream_request().model_name(), Some("gpt-5.3-codex"));
    let trace = start.trace();
    assert_eq!(trace.session_id, "session-1");
    assert_eq!(trace.input_kind, RuntimeReplyInputKind::UserMessage);
    assert_eq!(trace.message_chars, 42);
    assert_eq!(
        trace.provider_backend,
        Some(RuntimeProviderBackend::Current)
    );
    assert_eq!(trace.provider_name, Some("openai"));
    assert_eq!(trace.model_name, Some("gpt-5.3-codex"));
}

#[test]
fn provider_stream_start_rejects_missing_handle() {
    let expected =
        RuntimeReplyProviderHandle::from_config(&runtime_config(), RuntimeProviderBackend::Current);
    let request =
        RuntimeReplyStreamRequest::new("session-1", RuntimeReplyInputKind::UserMessage, 42, None);

    let error = RuntimeReplyProviderStreamStart::new(request, &expected)
        .expect_err("missing provider handle");

    assert!(error
        .message
        .contains("requires a configured provider handle"));
}

#[test]
fn provider_stream_start_rejects_mismatched_handle() {
    let expected =
        RuntimeReplyProviderHandle::from_config(&runtime_config(), RuntimeProviderBackend::Current);
    let mut other_config = runtime_config();
    other_config.provider_name = "anthropic".to_string();
    other_config.model_name = "claude-sonnet-4.5".to_string();
    let actual =
        RuntimeReplyProviderHandle::from_config(&other_config, RuntimeProviderBackend::Current);
    let request = RuntimeReplyStreamRequest::new(
        "session-1",
        RuntimeReplyInputKind::UserMessage,
        42,
        Some(actual),
    );

    let error = RuntimeReplyProviderStreamStart::new(request, &expected)
        .expect_err("mismatched provider handle");

    assert!(error.message.contains("Provider stream handle mismatch"));
    assert!(error.message.contains("openai/gpt-5.3-codex"));
    assert!(error.message.contains("anthropic/claude-sonnet-4.5"));
}

#[test]
fn provider_stream_poll_contract_carries_cancel_wait_rule() {
    assert_eq!(
        provider_stream_cancel_poll_interval(true),
        Some(PROVIDER_STREAM_CANCEL_POLL_INTERVAL)
    );
    assert_eq!(provider_stream_cancel_poll_interval(false), None);
    assert_eq!(
        provider_stream_timeout_poll::<()>(false),
        ProviderStreamPoll::Pending
    );
    assert_eq!(
        provider_stream_timeout_poll::<()>(true),
        ProviderStreamPoll::Canceled(ProviderStreamCancelReason::WhileWaiting)
    );
    assert_eq!(
        ProviderStreamCancelReason::WhileWaiting.as_str(),
        PROVIDER_STREAM_CANCEL_WHILE_WAITING_REASON
    );
}

#[test]
fn provider_stream_poll_contract_classifies_event_boundary_cancel() {
    assert_eq!(
        provider_stream_event_poll::<u8>(None, false),
        ProviderStreamPoll::End
    );
    assert_eq!(
        provider_stream_event_poll(Some(7_u8), false),
        ProviderStreamPoll::Item(7)
    );
    assert_eq!(
        provider_stream_event_poll(Some(7_u8), true),
        ProviderStreamPoll::Canceled(ProviderStreamCancelReason::BeforeEventProcessing)
    );
    assert_eq!(
        ProviderStreamCancelReason::BeforeEventProcessing.as_str(),
        PROVIDER_STREAM_CANCEL_BEFORE_EVENT_REASON
    );
}

#[test]
fn stream_event_projects_safety_buffering_payload_from_response_event() {
    let request = RuntimeReplyStreamRequest::new(
        "session-1",
        RuntimeReplyInputKind::UserMessage,
        42,
        Some(RuntimeReplyProviderHandle::from_config(
            &runtime_config(),
            RuntimeProviderBackend::Current,
        )),
    );
    let response_event = json!({
        "type": "response.output_text.delta",
        "safety_buffering": {
            "use_cases": ["cyber"],
            "reasons": ["user_risk"],
        },
    });

    let event = RuntimeReplyProviderStreamEvent::safety_buffering_from_response_event(
        &request,
        &response_event,
        [
            (SAFETY_BUFFERING_ENABLED_HEADER, "true"),
            (SAFETY_BUFFERING_FASTER_MODEL_HEADER, " retry-target "),
        ],
    )
    .expect("safety buffering event");

    assert_eq!(
        event.runtime_event_kind(),
        SAFETY_BUFFERING_RUNTIME_EVENT_KIND
    );
    let RuntimeReplyProviderStreamEvent::SafetyBuffering(payload) = &event;
    assert_eq!(payload.provider.as_deref(), Some("openai"));
    assert_eq!(payload.model.as_deref(), Some("gpt-5.3-codex"));
    assert_eq!(payload.use_cases, ["cyber"]);
    assert_eq!(payload.reasons, ["user_risk"]);
    assert!(payload.show_buffering_ui);
    assert_eq!(payload.retry_model.as_deref(), Some("retry-target"));
    assert_eq!(
        payload.fallback_header_model.as_deref(),
        Some("retry-target")
    );
    assert_eq!(
        payload.source,
        ProviderSafetyBufferingRetryModelSource::LegacyHeader
    );

    let payload_json = event.payload_json_value();
    assert_eq!(payload_json["source"], json!("legacy_header"));
    assert!(payload_json.get("retry_model").is_none());
    assert!(payload_json.get("faster_model").is_none());
}

#[test]
fn stream_event_projects_safety_buffering_payload_from_notification_payload() {
    let request = RuntimeReplyStreamRequest::new(
        "session-1",
        RuntimeReplyInputKind::UserMessage,
        42,
        Some(RuntimeReplyProviderHandle::from_config(
            &runtime_config(),
            RuntimeProviderBackend::Current,
        )),
    );
    let payload = json!({
        "eventKind": RuntimeReplyProviderStreamEvent::NOTIFICATION_KIND_SAFETY_BUFFERING,
        "responseEvent": {
            "type": "response.output_text.delta",
            "safety_buffering": {
                "use_cases": ["cyber"],
                "reasons": ["policy"],
                "retry_model": "gpt-5-mini"
            }
        },
        "headers": [
            {
                "name": SAFETY_BUFFERING_ENABLED_HEADER,
                "value": "true"
            },
            {
                "name": SAFETY_BUFFERING_FASTER_MODEL_HEADER,
                "value": "legacy-fast"
            }
        ]
    });

    let event = RuntimeReplyProviderStreamEvent::from_notification_payload(&request, &payload)
        .expect("safety buffering event");

    let RuntimeReplyProviderStreamEvent::SafetyBuffering(payload) = &event;
    assert_eq!(payload.provider.as_deref(), Some("openai"));
    assert_eq!(payload.model.as_deref(), Some("gpt-5.3-codex"));
    assert_eq!(payload.use_cases, ["cyber"]);
    assert_eq!(payload.reasons, ["policy"]);
    assert_eq!(payload.retry_model.as_deref(), Some("gpt-5-mini"));
    assert_eq!(payload.fallback_header_model, None);
    assert_eq!(
        payload.source,
        ProviderSafetyBufferingRetryModelSource::PayloadRetryModel
    );
}

#[test]
fn stream_event_ignores_unknown_notification_payload_kind() {
    let request =
        RuntimeReplyStreamRequest::new("session-1", RuntimeReplyInputKind::UserMessage, 42, None);
    let payload = json!({
        "eventKind": "vendor.unknown",
        "responseEvent": {
            "type": "response.output_text.delta",
            "safety_buffering": {
                "use_cases": ["cyber"],
                "reasons": ["policy"]
            }
        }
    });

    assert!(
        RuntimeReplyProviderStreamEvent::from_notification_payload(&request, &payload).is_none()
    );
}

#[test]
fn stream_event_ignores_false_safety_buffering_response_field() {
    let request = RuntimeReplyStreamRequest::new(
        "session-1",
        RuntimeReplyInputKind::UserMessage,
        42,
        Some(RuntimeReplyProviderHandle::from_config(
            &runtime_config(),
            RuntimeProviderBackend::Current,
        )),
    );
    let response_event = json!({
        "type": "response.created",
        "safety_buffering": false,
    });

    assert!(
        RuntimeReplyProviderStreamEvent::safety_buffering_from_response_event(
            &request,
            &response_event,
            [(SAFETY_BUFFERING_ENABLED_HEADER, "true")],
        )
        .is_none()
    );
}

#[test]
fn stream_request_carries_model_request_policy() {
    let request =
        RuntimeReplyStreamRequest::new("session-1", RuntimeReplyInputKind::UserMessage, 42, None)
            .with_model_request_policy(RuntimeReplyModelRequestPolicy::new(
                Some(RuntimeReplyResponsesPolicy {
                    use_responses_lite: true,
                    request_mode: "responses_lite".to_string(),
                    instructions_location: "input_prefix".to_string(),
                    tools_location: "input_prefix".to_string(),
                    reasoning_context: "all_turns".to_string(),
                    parallel_tool_calls_allowed: false,
                    requires_responses_lite_header: true,
                }),
                Some(RuntimeReplyToolCallPolicy {
                    supports_parallel_tool_calls: true,
                    parallel_tool_calls: false,
                }),
                None,
            ));

    let policy = request.model_request_policy.as_ref().expect("policy");
    assert!(policy.use_responses_lite());
    assert_eq!(policy.reasoning_context(), Some("all_turns"));
    assert!(policy.requires_responses_lite_header());
    assert_eq!(policy.parallel_tool_calls(), Some(false));
}

#[test]
fn stream_request_projects_responses_lite_wire_shape() {
    let request =
        RuntimeReplyStreamRequest::new("session-1", RuntimeReplyInputKind::UserMessage, 42, None)
            .with_model_request_policy(RuntimeReplyModelRequestPolicy::new(
                Some(RuntimeReplyResponsesPolicy {
                    use_responses_lite: true,
                    request_mode: "responses_lite".to_string(),
                    instructions_location: "input_prefix".to_string(),
                    tools_location: "input_prefix".to_string(),
                    reasoning_context: "all_turns".to_string(),
                    parallel_tool_calls_allowed: false,
                    requires_responses_lite_header: true,
                }),
                Some(RuntimeReplyToolCallPolicy {
                    supports_parallel_tool_calls: true,
                    parallel_tool_calls: false,
                }),
                None,
            ));

    let wire_shape = request.provider_request_wire_shape();

    assert!(wire_shape.use_responses_lite);
    assert_eq!(
        wire_shape.instructions_location.as_deref(),
        Some("input_prefix")
    );
    assert_eq!(wire_shape.tools_location.as_deref(), Some("input_prefix"));
    assert_eq!(wire_shape.reasoning_context.as_deref(), Some("all_turns"));
    assert_eq!(wire_shape.parallel_tool_calls, Some(false));
    assert_eq!(
        wire_shape.headers,
        vec![RuntimeReplyProviderRequestHeader {
            name: RuntimeReplyProviderRequestWireShape::RESPONSES_LITE_HEADER_NAME.to_string(),
            value: RuntimeReplyProviderRequestWireShape::RESPONSES_LITE_HEADER_VALUE.to_string(),
        }]
    );
    assert!(wire_shape.requires_responses_lite_wire_support());
}

#[test]
fn stream_request_accepts_responses_lite_wire_for_current_backend() {
    let handle =
        RuntimeReplyProviderHandle::from_config(&runtime_config(), RuntimeProviderBackend::Current);
    let request = RuntimeReplyStreamRequest::new(
        "session-1",
        RuntimeReplyInputKind::UserMessage,
        42,
        Some(handle),
    )
    .with_model_request_policy(RuntimeReplyModelRequestPolicy::new(
        Some(RuntimeReplyResponsesPolicy {
            use_responses_lite: true,
            request_mode: "responses_lite".to_string(),
            instructions_location: "input_prefix".to_string(),
            tools_location: "input_prefix".to_string(),
            reasoning_context: "all_turns".to_string(),
            parallel_tool_calls_allowed: false,
            requires_responses_lite_header: true,
        }),
        None,
        None,
    ));

    assert_eq!(
        request.provider_request_wire_shape().parallel_tool_calls,
        Some(false)
    );
    assert!(request.provider_request_wire_support_issue().is_none());
}

#[test]
fn model_policy_keeps_parallel_tool_calls_fail_closed_without_explicit_support() {
    let policy = RuntimeReplyModelRequestPolicy::new(
        None,
        Some(RuntimeReplyToolCallPolicy {
            supports_parallel_tool_calls: false,
            parallel_tool_calls: true,
        }),
        None,
    )
    .expect("tool policy");

    assert_eq!(policy.parallel_tool_calls(), Some(false));
}

#[test]
fn stream_request_accepts_responses_lite_wire_for_openai_responses_compat() {
    let handle =
        RuntimeReplyProviderHandle::from_config(&runtime_config(), RuntimeProviderBackend::Current);
    let request = RuntimeReplyStreamRequest::new(
        "session-1",
        RuntimeReplyInputKind::UserMessage,
        42,
        Some(handle),
    )
    .with_model_request_policy(RuntimeReplyModelRequestPolicy::new(
        Some(RuntimeReplyResponsesPolicy {
            use_responses_lite: true,
            request_mode: "responses_lite".to_string(),
            instructions_location: "input_prefix".to_string(),
            tools_location: "input_prefix".to_string(),
            reasoning_context: "all_turns".to_string(),
            parallel_tool_calls_allowed: false,
            requires_responses_lite_header: true,
        }),
        None,
        None,
    ));

    assert!(request.provider_request_wire_support_issue().is_none());
}

#[test]
fn stream_request_rejects_responses_lite_wire_for_chat_compat() {
    let mut config = runtime_config();
    config.protocol = Some(RuntimeProviderProtocol::ChatCompletions);
    let handle = RuntimeReplyProviderHandle::from_config(&config, RuntimeProviderBackend::Current);
    let request = RuntimeReplyStreamRequest::new(
        "session-1",
        RuntimeReplyInputKind::UserMessage,
        42,
        Some(handle),
    )
    .with_model_request_policy(RuntimeReplyModelRequestPolicy::new(
        Some(RuntimeReplyResponsesPolicy {
            use_responses_lite: true,
            request_mode: "responses_lite".to_string(),
            instructions_location: "input_prefix".to_string(),
            tools_location: "input_prefix".to_string(),
            reasoning_context: "all_turns".to_string(),
            parallel_tool_calls_allowed: false,
            requires_responses_lite_header: true,
        }),
        None,
        None,
    ));

    let issue = request
        .provider_request_wire_support_issue()
        .expect("wire support issue");

    assert_eq!(
        issue.provider_backend,
        Some(RuntimeProviderBackend::Current)
    );
    assert_eq!(issue.provider_name.as_deref(), Some("openai"));
    assert_eq!(issue.model_name.as_deref(), Some("gpt-5.3-codex"));
    assert!(issue.wire_shape.requires_responses_lite_wire_support());
    assert_eq!(
        issue.message(),
        RuntimeReplyProviderWireSupportIssue::MESSAGE
    );
}

#[test]
fn stream_request_projects_plain_responses_parallel_tool_calls_without_lite_header() {
    let request =
        RuntimeReplyStreamRequest::new("session-1", RuntimeReplyInputKind::UserMessage, 42, None)
            .with_model_request_policy(RuntimeReplyModelRequestPolicy::new(
                Some(RuntimeReplyResponsesPolicy {
                    use_responses_lite: false,
                    request_mode: "responses".to_string(),
                    instructions_location: "request_field".to_string(),
                    tools_location: "request_field".to_string(),
                    reasoning_context: "default".to_string(),
                    parallel_tool_calls_allowed: true,
                    requires_responses_lite_header: false,
                }),
                Some(RuntimeReplyToolCallPolicy {
                    supports_parallel_tool_calls: true,
                    parallel_tool_calls: true,
                }),
                None,
            ));

    let wire_shape = request.provider_request_wire_shape();

    assert!(!wire_shape.use_responses_lite);
    assert_eq!(wire_shape.reasoning_context.as_deref(), Some("default"));
    assert_eq!(wire_shape.parallel_tool_calls, Some(true));
    assert!(wire_shape.headers.is_empty());
    assert!(!wire_shape.requires_responses_lite_wire_support());
}

#[test]
fn stream_request_projects_reasoning_output_wire_shape() {
    let request =
        RuntimeReplyStreamRequest::new("session-1", RuntimeReplyInputKind::UserMessage, 42, None)
            .with_model_request_policy(RuntimeReplyModelRequestPolicy::new(
                None,
                None,
                Some(RuntimeReplyReasoningOutputPolicy {
                    default_reasoning_summary: "detailed".to_string(),
                    support_verbosity: true,
                    default_verbosity: Some("low".to_string()),
                    can_set_verbosity: true,
                }),
            ));

    let wire_shape = request.provider_request_wire_shape();

    assert_eq!(wire_shape.reasoning_summary.as_deref(), Some("detailed"));
    assert_eq!(wire_shape.text_verbosity.as_deref(), Some("low"));
}

#[test]
fn reasoning_output_wire_shape_omits_none_summary_and_unsupported_verbosity() {
    let request =
        RuntimeReplyStreamRequest::new("session-1", RuntimeReplyInputKind::UserMessage, 42, None)
            .with_model_request_policy(RuntimeReplyModelRequestPolicy::new(
                None,
                None,
                Some(RuntimeReplyReasoningOutputPolicy {
                    default_reasoning_summary: "none".to_string(),
                    support_verbosity: false,
                    default_verbosity: Some("high".to_string()),
                    can_set_verbosity: false,
                }),
            ));

    let wire_shape = request.provider_request_wire_shape();

    assert_eq!(wire_shape.reasoning_summary, None);
    assert_eq!(wire_shape.text_verbosity, None);
}

#[test]
fn provider_token_usage_calculates_total_when_provider_omits_it() {
    let usage = RuntimeReplyProviderTokenUsage::new(Some(11), Some(7), None);

    assert_eq!(usage.input_tokens, Some(11));
    assert_eq!(usage.output_tokens, Some(7));
    assert_eq!(usage.total_tokens, Some(18));
}

#[test]
fn provider_usage_combines_token_and_cache_counts() {
    let first = RuntimeReplyProviderUsage::new(
        "gpt-5.3-codex".to_string(),
        RuntimeReplyProviderTokenUsage {
            input_tokens: Some(10),
            output_tokens: None,
            total_tokens: None,
            cached_input_tokens: Some(3),
            cache_creation_input_tokens: None,
        },
    );
    let second = RuntimeReplyProviderUsage::new(
        "worker-model".to_string(),
        RuntimeReplyProviderTokenUsage::new(Some(2), Some(5), Some(9))
            .with_cached_input_tokens(Some(4))
            .with_cache_creation_input_tokens(Some(1)),
    );

    let combined = first.combine_with(&second);

    assert_eq!(combined.model, "gpt-5.3-codex");
    assert_eq!(combined.usage.input_tokens, Some(12));
    assert_eq!(combined.usage.output_tokens, Some(5));
    assert_eq!(combined.usage.total_tokens, Some(9));
    assert_eq!(combined.usage.cached_input_tokens, Some(7));
    assert_eq!(combined.usage.cache_creation_input_tokens, Some(1));
}

#[test]
fn provider_message_outputs_attach_usage_to_first_message() {
    let outputs = provider_stream_message_outputs(["first", "second"], None, Some(42));

    assert_eq!(
        outputs,
        vec![
            RuntimeReplyProviderMessageOutput::Message {
                message: "first",
                usage: Some(42),
            },
            RuntimeReplyProviderMessageOutput::Message {
                message: "second",
                usage: None,
            },
        ]
    );
}

#[test]
fn provider_message_outputs_flush_pending_message_when_usage_has_no_message() {
    let outputs = provider_stream_message_outputs(Vec::<&str>::new(), Some("pending"), Some(7));

    assert_eq!(
        outputs,
        vec![RuntimeReplyProviderMessageOutput::Message {
            message: "pending",
            usage: Some(7),
        }]
    );
}

#[test]
fn provider_message_outputs_emit_usage_only_when_message_is_absent() {
    let outputs = provider_stream_message_outputs(Vec::<&str>::new(), None, Some(5));

    assert_eq!(outputs, vec![RuntimeReplyProviderMessageOutput::Usage(5)]);
}

#[test]
fn provider_single_message_output_keeps_direct_answer_usage_attached() {
    let outputs = provider_stream_single_message_output("answer", Some(3));

    assert_eq!(
        outputs,
        vec![RuntimeReplyProviderMessageOutput::Message {
            message: "answer",
            usage: Some(3),
        }]
    );
}
