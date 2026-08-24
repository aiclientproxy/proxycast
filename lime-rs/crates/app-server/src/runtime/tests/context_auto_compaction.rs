use super::*;
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

struct AutoCompactionUsageBackend {
    requests: Mutex<Vec<ExecutionRequest>>,
    histories: Mutex<Vec<Vec<model_provider::current_client::CurrentProviderMessage>>>,
    call_count: AtomicUsize,
    high_usage_calls: usize,
}

impl AutoCompactionUsageBackend {
    fn complete_turn(
        &self,
        request: ExecutionRequest,
        provider_history: Vec<model_provider::current_client::CurrentProviderMessage>,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let call_index = self.call_count.fetch_add(1, Ordering::SeqCst);
        self.requests
            .lock()
            .expect("test backend requests mutex poisoned")
            .push(request);
        self.histories
            .lock()
            .expect("test backend histories mutex poisoned")
            .push(provider_history);
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        let input_tokens = if call_index < self.high_usage_calls {
            91_000
        } else {
            1_000
        };
        sink.emit(RuntimeEvent::new(
            "turn.completed",
            json!({
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 200_000
                }
            }),
        ))
    }
}

#[async_trait]
impl ExecutionBackend for AutoCompactionUsageBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.complete_turn(request, Vec::new(), sink)
    }

    async fn start_turn_with_provider_history(
        &self,
        request: ExecutionRequest,
        provider_history: crate::runtime::provider_history::ProviderTurnHistory,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.complete_turn(request, provider_history.into_messages(), sink)
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

#[tokio::test]
async fn auto_compacts_before_turn_when_context_limit_is_due() {
    let backend = Arc::new(AutoCompactionUsageBackend {
        requests: Mutex::new(Vec::new()),
        histories: Mutex::new(Vec::new()),
        call_count: AtomicUsize::new(0),
        high_usage_calls: 1,
    });
    let core = RuntimeCore::with_backend(backend.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_auto_compact".to_string()),
        thread_id: Some("thread_auto_compact".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_auto_compact".to_string(),
            turn_id: Some("turn_auto_compact_1".to_string()),
            input: AgentInput {
                text: "第一轮".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("first turn");

    let metadata = json!({
        "harness": {
            "model_request_policy": {
                "context_policy": {
                    "model_context_window": 120_000,
                    "auto_compact_token_limit": 90_000
                }
            }
        }
    });
    let output = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_auto_compact".to_string(),
                turn_id: Some("turn_auto_compact_2".to_string()),
                input: AgentInput {
                    text: "继续".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: Some(RuntimeOptions {
                    runtime_request: Some(RuntimeRequest {
                        metadata: Some(metadata.clone()),
                        ..RuntimeRequest::default()
                    }),
                    ..RuntimeOptions::default()
                }),
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("second turn");

    let compact_index = output
        .events
        .iter()
        .position(|event| event.event_type == "context.compaction.completed")
        .expect("auto compact event");
    let turn_started_index = output
        .events
        .iter()
        .position(|event| {
            event.event_type == "turn.started"
                && event.turn_id.as_deref() == Some("turn_auto_compact_2")
        })
        .expect("turn started event");
    assert!(
        compact_index < turn_started_index,
        "auto compaction must run before backend sampling"
    );
    let completed = &output.events[compact_index];
    assert_eq!(
        completed.payload["trigger"].as_str(),
        Some("auto_context_limit")
    );
    assert_eq!(
        completed.payload["triggerContext"]["activeContextTokens"].as_u64(),
        Some(91_000)
    );
    assert_eq!(
        completed.payload["triggerContext"]["maxTokens"].as_u64(),
        Some(90_000)
    );
    assert!(completed.payload["triggerContext"]
        .get("outputTokens")
        .is_none());

    let requests = backend
        .requests
        .lock()
        .expect("test backend requests mutex poisoned");
    assert_eq!(requests.len(), 2);
    let second_metadata = requests[1]
        .runtime_options
        .as_ref()
        .and_then(app_server_protocol::RuntimeOptions::runtime_metadata)
        .expect("second turn metadata");
    assert!(
        second_metadata
            .get(crate::runtime::memory_prompt::SESSION_COMPACTION_PROMPT_CONTEXT_KEY)
            .is_none(),
        "replacement history must not be duplicated as prompt metadata"
    );
    drop(requests);

    let third = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_auto_compact".to_string(),
                turn_id: Some("turn_auto_compact_3".to_string()),
                input: AgentInput {
                    text: "继续第二次".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: Some(RuntimeOptions {
                    runtime_request: Some(RuntimeRequest {
                        metadata: Some(metadata),
                        ..RuntimeRequest::default()
                    }),
                    ..RuntimeOptions::default()
                }),
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("third turn");
    assert!(
        third
            .events
            .iter()
            .all(|event| event.event_type != "context.compaction.completed"),
        "old usage before latest compaction must not retrigger auto compact"
    );
}

#[tokio::test]
async fn auto_compaction_replaces_provider_prefix_with_summary_and_bounded_tail() {
    let backend = Arc::new(AutoCompactionUsageBackend {
        requests: Mutex::new(Vec::new()),
        histories: Mutex::new(Vec::new()),
        call_count: AtomicUsize::new(0),
        high_usage_calls: 6,
    });
    let core = RuntimeCore::with_backend(backend.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_auto_compact_history".to_string()),
        thread_id: Some("thread_auto_compact_history".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    for index in 1..=6 {
        core.start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_auto_compact_history".to_string(),
                turn_id: Some(format!("turn_history_{index}")),
                input: AgentInput {
                    text: format!("fact-from-turn-{index}"),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("seed turn");
    }

    let output = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_auto_compact_history".to_string(),
                turn_id: Some("turn_history_7".to_string()),
                input: AgentInput {
                    text: "current-turn-input".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: Some(RuntimeOptions {
                    runtime_request: Some(RuntimeRequest {
                        metadata: Some(json!({
                            "harness": {
                                "model_request_policy": {
                                    "context_policy": {
                                        "model_context_window": 120_000,
                                        "auto_compact_token_limit": 90_000
                                    }
                                }
                            }
                        })),
                        ..RuntimeRequest::default()
                    }),
                    ..RuntimeOptions::default()
                }),
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("compacted continuation turn");

    let completed = output
        .events
        .iter()
        .find(|event| event.event_type == "context.compaction.completed")
        .expect("auto compaction completed event");
    assert_eq!(
        completed.payload["tailStartTurnId"].as_str(),
        Some("turn_history_3")
    );
    let summary = completed.payload["summary"]
        .as_str()
        .expect("compaction summary");
    assert!(summary.contains("fact-from-turn-1"));
    assert!(summary.contains("fact-from-turn-2"));
    assert!(!summary.contains("fact-from-turn-3"));
    assert_eq!(
        completed.payload["artifact"]["policy"]["durableHistoryRewrite"].as_bool(),
        Some(false)
    );
    assert_eq!(
        completed.payload["artifact"]["policy"]["providerHistoryRewrite"].as_bool(),
        Some(true)
    );

    let histories = backend
        .histories
        .lock()
        .expect("test backend histories mutex poisoned");
    let provider_text = histories
        .last()
        .expect("seventh provider history")
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|content| match content {
            model_provider::current_client::CurrentProviderContent::Text(text) => {
                Some(text.as_str())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    // Replacement history contains the original user boundary and the compact summary. A third
    // occurrence would mean the durable prefix was appended again after replacement.
    assert_eq!(provider_text.matches("fact-from-turn-1").count(), 2);
    assert_eq!(provider_text.matches("fact-from-turn-2").count(), 2);
    for index in 3..=6 {
        assert!(
            provider_text.contains(&format!("fact-from-turn-{index}")),
            "provider history must retain raw tail turn {index}"
        );
    }
    assert!(!provider_text.contains("current-turn-input"));
    drop(histories);

    let durable_events = core
        .events_for_session("sess_auto_compact_history")
        .expect("durable session events");
    assert!(durable_events.iter().any(|event| {
        event.turn_id.as_deref() == Some("turn_history_1")
            && event.payload.to_string().contains("fact-from-turn-1")
    }));
}
