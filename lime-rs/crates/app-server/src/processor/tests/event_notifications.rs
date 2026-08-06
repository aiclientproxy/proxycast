use crate::processor::{
    project_event_notifications_jsonrpc, v2_notifications::V2NotificationProjector,
};
use app_server_protocol::AgentEvent;
use app_server_protocol::JsonRpcError;
use app_server_protocol::JsonRpcMessage;
use serde_json::json;

fn event_notifications_jsonrpc(event: AgentEvent) -> Result<Vec<JsonRpcMessage>, JsonRpcError> {
    let mut projector = V2NotificationProjector::default();
    project_event_notifications_jsonrpc(&mut projector, event)
}

fn single_event_notification(event: AgentEvent) -> Result<JsonRpcMessage, JsonRpcError> {
    let mut messages = event_notifications_jsonrpc(event)?;
    assert_eq!(messages.len(), 1, "expected one projected notification");
    Ok(messages.remove(0))
}

#[test]
fn event_notifications_jsonrpc_emits_direct_agent_message_delta() {
    let message = single_event_notification(AgentEvent {
        event_id: "evt_1".to_string(),
        sequence: 1,
        session_id: "sess_1".to_string(),
        thread_id: Some("thread_1".to_string()),
        turn_id: Some("turn_1".to_string()),
        event_type: "message.delta".to_string(),
        timestamp: "2026-07-05T00:00:00Z".to_string(),
        payload: json!({
            "itemId": "agent-message-final",
            "text": "typed delta",
            "phase": "final_answer",
        }),
    })
    .expect("notification");

    let JsonRpcMessage::Notification(notification) = message else {
        panic!("expected notification");
    };
    assert_eq!(notification.method, "item/agentMessage/delta");
    assert_eq!(
        notification.params.expect("params"),
        json!({
            "threadId": "thread_1",
            "turnId": "turn_1",
            "itemId": "agent-message-final",
            "delta": "typed delta"
        })
    );
}

#[test]
fn event_notifications_jsonrpc_emits_direct_turn_plan_update() {
    let message = single_event_notification(AgentEvent {
        event_id: "evt_plan_1".to_string(),
        sequence: 2,
        session_id: "sess_1".to_string(),
        thread_id: Some("thread_1".to_string()),
        turn_id: Some("turn_1".to_string()),
        event_type: "turn.plan.updated".to_string(),
        timestamp: "2026-07-05T00:00:00Z".to_string(),
        payload: json!({
            "explanation": "继续执行",
            "plan": [
                { "step": "读取现状", "status": "completed" },
                { "step": "补齐主链", "status": "in_progress" }
            ]
        }),
    })
    .expect("notification");

    let JsonRpcMessage::Notification(notification) = message else {
        panic!("expected notification");
    };
    assert_eq!(notification.method, "turn/plan/updated");
    assert_eq!(
        notification.params.expect("params"),
        json!({
            "threadId": "thread_1",
            "turnId": "turn_1",
            "explanation": "继续执行",
            "plan": [
                { "step": "读取现状", "status": "completed" },
                { "step": "补齐主链", "status": "inProgress" }
            ]
        })
    );
}

#[test]
fn event_notifications_jsonrpc_emits_typed_warning_with_localization_code() {
    let message = single_event_notification(AgentEvent {
        event_id: "evt_warning".to_string(),
        sequence: 2,
        session_id: "sess_1".to_string(),
        thread_id: Some("thread_1".to_string()),
        turn_id: Some("turn_1".to_string()),
        event_type: "runtime.warning".to_string(),
        timestamp: "2026-07-05T00:00:00Z".to_string(),
        payload: json!({
            "code": "skill_not_available",
            "message": "技能不可用，已继续执行。"
        }),
    })
    .expect("notification");

    let JsonRpcMessage::Notification(notification) = message else {
        panic!("expected notification");
    };
    assert_eq!(notification.method, "warning");
    assert_eq!(
        notification.params.expect("params"),
        json!({
            "threadId": "thread_1",
            "message": "技能不可用，已继续执行。",
            "code": "skill_not_available"
        })
    );
}

#[test]
fn malformed_runtime_warning_rejects_without_retired_side_channel() {
    for payload in [
        json!({}),
        json!({ "message": "   " }),
        json!({ "code": 42, "message": "warning" }),
    ] {
        let error = event_notifications_jsonrpc(AgentEvent {
            event_id: "evt_warning_invalid".to_string(),
            sequence: 3,
            session_id: "sess_1".to_string(),
            thread_id: Some("thread_1".to_string()),
            turn_id: Some("turn_1".to_string()),
            event_type: "runtime.warning".to_string(),
            timestamp: "2026-07-05T00:00:00Z".to_string(),
            payload,
        })
        .expect_err("malformed warning must reject");

        assert_eq!(error.code, app_server_protocol::error_codes::RUNTIME_ERROR);
        assert!(error.message.contains("runtime.warning"));
    }
}

#[test]
fn warning_alias_is_rejected_without_retired_side_channel() {
    let error = event_notifications_jsonrpc(AgentEvent {
        event_id: "evt_warning_alias".to_string(),
        sequence: 4,
        session_id: "sess_1".to_string(),
        thread_id: Some("thread_1".to_string()),
        turn_id: Some("turn_1".to_string()),
        event_type: "warning".to_string(),
        timestamp: "2026-07-05T00:00:00Z".to_string(),
        payload: json!({
            "code": "skill_not_available",
            "message": "技能不可用，已继续执行。"
        }),
    })
    .expect_err("retired warning alias must reject");

    assert_eq!(error.code, app_server_protocol::error_codes::RUNTIME_ERROR);
    assert!(error.message.contains("warning"));
}

fn failed_turn_event(event_id: &str, message: &str) -> AgentEvent {
    AgentEvent {
        event_id: event_id.to_string(),
        sequence: 9,
        session_id: "sess_1".to_string(),
        thread_id: Some("thread_1".to_string()),
        turn_id: Some("turn_1".to_string()),
        event_type: "turn.failed".to_string(),
        timestamp: "2026-07-05T00:00:01Z".to_string(),
        payload: json!({
            "message": message,
            "turn": {
                "sessionId": "sess_1",
                "threadId": "thread_1",
                "turnId": "turn_1",
                "status": "failed",
                "createdAtMs": 100,
                "updatedAtMs": 120,
                "startedAtMs": 100,
                "completedAtMs": 120,
                "error": {"message": message}
            }
        }),
    }
}

#[test]
fn event_notifications_project_retry_and_terminal_errors_with_exact_semantics() {
    let mut projector = V2NotificationProjector::default();
    let retry = project_event_notifications_jsonrpc(
        &mut projector,
        AgentEvent {
            event_id: "evt_retry".to_string(),
            sequence: 7,
            session_id: "sess_1".to_string(),
            thread_id: Some("thread_1".to_string()),
            turn_id: Some("turn_1".to_string()),
            event_type: "plugin_worker.retry".to_string(),
            timestamp: "2026-07-05T00:00:00Z".to_string(),
            payload: json!({
                "message": "provider stream reconnecting",
                "errorCode": "PLUGIN_WORKER_RETRYABLE_FAILURE",
                "retryable": false
            }),
        },
    )
    .expect("retry notification");
    assert_eq!(retry.len(), 1);
    let JsonRpcMessage::Notification(retry) = &retry[0] else {
        panic!("expected retry notification");
    };
    assert_eq!(retry.method, "error");
    assert_eq!(retry.params.as_ref().expect("params")["willRetry"], true);

    let terminal = project_event_notifications_jsonrpc(
        &mut projector,
        failed_turn_event("evt_failed_after_retry", "retry budget exhausted"),
    )
    .expect("terminal notifications");
    let terminal = terminal
        .into_iter()
        .map(|message| match message {
            JsonRpcMessage::Notification(notification) => notification,
            other => panic!("expected notification, got {other:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        terminal
            .iter()
            .map(|notification| notification.method.as_str())
            .collect::<Vec<_>>(),
        ["error", "turn/completed"]
    );
    assert_eq!(
        terminal[0].params.as_ref().expect("error params")["willRetry"],
        false
    );
    assert_eq!(
        terminal[1].params.as_ref().expect("completion params")["turn"]["status"],
        "failed"
    );
}

#[test]
fn runtime_error_is_terminal_even_when_legacy_retryable_is_true_and_is_not_duplicated() {
    let mut projector = V2NotificationProjector::default();
    let runtime_error = project_event_notifications_jsonrpc(
        &mut projector,
        AgentEvent {
            event_id: "evt_runtime_error".to_string(),
            sequence: 8,
            session_id: "sess_1".to_string(),
            thread_id: Some("thread_1".to_string()),
            turn_id: Some("turn_1".to_string()),
            event_type: "runtime.error".to_string(),
            timestamp: "2026-07-05T00:00:00Z".to_string(),
            payload: json!({
                "message": "retry budget exhausted",
                "errorCode": "PLUGIN_WORKER_RETRYABLE_FAILURE",
                "retryable": true
            }),
        },
    )
    .expect("runtime error notification");
    let JsonRpcMessage::Notification(runtime_error) = &runtime_error[0] else {
        panic!("expected runtime error notification");
    };
    let params = runtime_error.params.as_ref().expect("runtime error params");
    assert_eq!(runtime_error.method, "error");
    assert_eq!(params["willRetry"], false);
    assert_eq!(params["error"]["codexErrorInfo"], "other");

    let completion = project_event_notifications_jsonrpc(
        &mut projector,
        failed_turn_event("evt_failed_after_runtime_error", "retry budget exhausted"),
    )
    .expect("completion after terminal error");
    assert_eq!(completion.len(), 1);
    let JsonRpcMessage::Notification(completion) = &completion[0] else {
        panic!("expected turn completion");
    };
    assert_eq!(completion.method, "turn/completed");
}

#[test]
fn turn_failed_without_runtime_error_emits_error_before_completion() {
    let messages = event_notifications_jsonrpc(failed_turn_event(
        "evt_failed_without_error",
        "provider stream timed out",
    ))
    .expect("failed turn notifications");
    let methods = messages
        .iter()
        .map(|message| match message {
            JsonRpcMessage::Notification(notification) => notification.method.as_str(),
            other => panic!("expected notification, got {other:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(methods, ["error", "turn/completed"]);
}

#[test]
fn malformed_typed_error_sources_fail_closed() {
    for (index, event) in [
        AgentEvent {
            event_id: "missing_thread".to_string(),
            sequence: 1,
            session_id: "sess_1".to_string(),
            thread_id: None,
            turn_id: Some("turn_1".to_string()),
            event_type: "runtime.error".to_string(),
            timestamp: "2026-07-05T00:00:00Z".to_string(),
            payload: json!({"message": "failed"}),
        },
        AgentEvent {
            event_id: "missing_turn".to_string(),
            sequence: 1,
            session_id: "sess_1".to_string(),
            thread_id: Some("thread_1".to_string()),
            turn_id: None,
            event_type: "runtime.error".to_string(),
            timestamp: "2026-07-05T00:00:00Z".to_string(),
            payload: json!({"message": "failed"}),
        },
        AgentEvent {
            event_id: "missing_message".to_string(),
            sequence: 1,
            session_id: "sess_1".to_string(),
            thread_id: Some("thread_1".to_string()),
            turn_id: Some("turn_1".to_string()),
            event_type: "runtime.error".to_string(),
            timestamp: "2026-07-05T00:00:00Z".to_string(),
            payload: json!({}),
        },
        AgentEvent {
            event_id: "invalid_retry".to_string(),
            sequence: 1,
            session_id: "sess_1".to_string(),
            thread_id: Some("thread_1".to_string()),
            turn_id: Some("turn_1".to_string()),
            event_type: "runtime.error".to_string(),
            timestamp: "2026-07-05T00:00:00Z".to_string(),
            payload: json!({"message": "failed", "willRetry": "false"}),
        },
        AgentEvent {
            event_id: "invalid_details".to_string(),
            sequence: 1,
            session_id: "sess_1".to_string(),
            thread_id: Some("thread_1".to_string()),
            turn_id: Some("turn_1".to_string()),
            event_type: "runtime.error".to_string(),
            timestamp: "2026-07-05T00:00:00Z".to_string(),
            payload: json!({"message": "failed", "additionalDetails": 42}),
        },
    ]
    .into_iter()
    .enumerate()
    {
        let error = match event_notifications_jsonrpc(event) {
            Ok(messages) => panic!("case {index} projected {messages:?}"),
            Err(error) => error,
        };
        assert_eq!(error.code, app_server_protocol::error_codes::RUNTIME_ERROR);
    }
}

#[test]
fn event_notifications_jsonrpc_lowers_turn_failed_to_error_then_completion() {
    let messages = event_notifications_jsonrpc(AgentEvent {
        event_id: "evt_failed".to_string(),
        sequence: 2,
        session_id: "sess_1".to_string(),
        thread_id: Some("thread_1".to_string()),
        turn_id: Some("turn_1".to_string()),
        event_type: "turn.failed".to_string(),
        timestamp: "2026-07-05T00:00:01Z".to_string(),
        payload: json!({
            "message": "provider stream timed out",
            "turn": {
                "sessionId": "sess_1",
                "threadId": "thread_1",
                "turnId": "turn_1",
                "status": "failed",
                "createdAtMs": 100,
                "updatedAtMs": 120,
                "startedAtMs": 100,
                "completedAtMs": 120,
                "error": {"message": "provider stream timed out"}
            }
        }),
    })
    .expect("notifications");

    assert_eq!(messages.len(), 2);
    let JsonRpcMessage::Notification(notification) = &messages[1] else {
        panic!("expected notification");
    };
    assert_eq!(notification.method, "turn/completed");
    let params = notification.params.as_ref().expect("params");
    assert_eq!(params["threadId"], "thread_1");
    assert_eq!(params["turn"]["id"], "turn_1");
    assert_eq!(params["turn"]["status"], "failed");
}

#[test]
fn direct_delta_uses_the_canonical_item_identity() {
    let message = single_event_notification(AgentEvent {
        event_id: "evt_item".to_string(),
        sequence: 3,
        session_id: "sess_1".to_string(),
        thread_id: Some("thread_1".to_string()),
        turn_id: Some("turn_1".to_string()),
        event_type: "message.delta".to_string(),
        timestamp: "2026-07-05T00:00:02Z".to_string(),
        payload: json!({
            "text": "hello",
            "item": {
                "sessionId": "sess_1",
                "threadId": "thread_1",
                "turnId": "turn_1",
                "itemId": "agent-turn_1",
                "sequence": 3,
                "ordinal": 3,
                "createdAtMs": 100,
                "updatedAtMs": 120,
                "kind": "agentMessage",
                "status": "inProgress",
                "payload": {
                    "type": "agentMessage",
                    "text": "hello"
                }
            }
        }),
    })
    .expect("notification");

    let JsonRpcMessage::Notification(notification) = message else {
        panic!("expected notification");
    };
    assert_eq!(notification.method, "item/agentMessage/delta");
    let params = notification.params.expect("params");
    assert_eq!(params["itemId"], "agent-turn_1");
    assert_eq!(params["delta"], "hello");
}

#[test]
fn current_turn_canceled_projects_interrupted_and_retired_name_is_not_accepted() {
    let turn = json!({
        "sessionId": "sess_1",
        "threadId": "thread_1",
        "turnId": "turn_1",
        "status": "interrupted",
        "createdAtMs": 100,
        "updatedAtMs": 120
    });
    let notification = single_event_notification(AgentEvent {
        event_id: "evt_canceled".to_string(),
        sequence: 4,
        session_id: "sess_1".to_string(),
        thread_id: Some("thread_1".to_string()),
        turn_id: Some("turn_1".to_string()),
        event_type: "turn.canceled".to_string(),
        timestamp: "2026-07-05T00:00:03Z".to_string(),
        payload: json!({ "turn": turn.clone() }),
    })
    .expect("notification");
    let JsonRpcMessage::Notification(notification) = notification else {
        panic!("expected notification");
    };
    assert_eq!(notification.method, "turn/completed");
    let params = notification.params.expect("params");
    assert_eq!(params["turn"]["id"], "turn_1");
    assert_eq!(params["turn"]["status"], "interrupted");

    let error = event_notifications_jsonrpc(AgentEvent {
        event_id: "evt_interrupted".to_string(),
        sequence: 5,
        session_id: "sess_1".to_string(),
        thread_id: Some("thread_1".to_string()),
        turn_id: Some("turn_1".to_string()),
        event_type: "turn.interrupted".to_string(),
        timestamp: "2026-07-05T00:00:04Z".to_string(),
        payload: json!({ "turn": turn }),
    })
    .expect_err("retired turn.interrupted alias must reject");
    assert_eq!(error.code, app_server_protocol::error_codes::RUNTIME_ERROR);
    assert!(error.message.contains("turn.interrupted"));
}

#[test]
fn malformed_direct_lifecycle_does_not_fall_back_to_agent_session_event() {
    let error = event_notifications_jsonrpc(AgentEvent {
        event_id: "evt_malformed_item".to_string(),
        sequence: 6,
        session_id: "sess_1".to_string(),
        thread_id: Some("thread_1".to_string()),
        turn_id: Some("turn_1".to_string()),
        event_type: "item.completed".to_string(),
        timestamp: "2026-07-05T00:00:05Z".to_string(),
        payload: json!({}),
    })
    .expect_err("malformed direct lifecycle must reject");

    assert_eq!(error.code, app_server_protocol::error_codes::RUNTIME_ERROR);
    assert!(error.message.contains("item.completed"));
}

#[test]
fn terminal_usage_emits_completion_and_token_usage_notifications() {
    let messages = event_notifications_jsonrpc(AgentEvent {
        event_id: "evt_terminal_usage".to_string(),
        sequence: 7,
        session_id: "sess_1".to_string(),
        thread_id: Some("thread_1".to_string()),
        turn_id: Some("turn_1".to_string()),
        event_type: "turn.completed".to_string(),
        timestamp: "2026-07-05T00:00:06Z".to_string(),
        payload: json!({
            "turn": {
                "sessionId": "sess_1",
                "threadId": "thread_1",
                "turnId": "turn_1",
                "status": "completed",
                "createdAtMs": 100,
                "updatedAtMs": 120,
                "startedAtMs": 100,
                "completedAtMs": 120
            },
            "usage": {
                "total_token_usage": {
                    "total_tokens": 31_000,
                    "input_tokens": 31_000,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_output_tokens": 0
                },
                "last_token_usage": {
                    "total_tokens": 31_000,
                    "input_tokens": 31_000,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_output_tokens": 0
                }
            }
        }),
    })
    .expect("terminal notifications");

    assert_eq!(messages.len(), 2);
    let methods = messages
        .into_iter()
        .map(|message| match message {
            JsonRpcMessage::Notification(notification) => notification.method,
            other => panic!("expected notification, got {other:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(methods, ["thread/tokenUsage/updated", "turn/completed"]);
}
