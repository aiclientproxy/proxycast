use super::support::*;
use super::*;
use std::sync::Arc;
use tokio::time::sleep;

fn read_params(session_id: &str) -> AgentSessionReadParams {
    AgentSessionReadParams {
        session_id: session_id.to_string(),
        history_limit: None,
        history_offset: None,
        history_before_message_id: None,
    }
}

fn queued_ids(queued_turns: &[serde_json::Value]) -> Vec<&str> {
    queued_turns
        .iter()
        .map(|turn| {
            turn.get("queued_turn_id")
                .and_then(serde_json::Value::as_str)
                .expect("queued turn id")
        })
        .collect()
}

fn queued_positions(queued_turns: &[serde_json::Value]) -> Vec<u64> {
    queued_turns
        .iter()
        .map(|turn| {
            turn.get("position")
                .and_then(serde_json::Value::as_u64)
                .expect("queued turn position")
        })
        .collect()
}

async fn read_queued_turns(core: &RuntimeCore, session_id: &str) -> Vec<serde_json::Value> {
    let read = core
        .read_session_current(read_params(session_id))
        .await
        .expect("read current session");
    let detail = read.detail.expect("read detail");
    let queued_turns = detail
        .get("queued_turns")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .expect("detail queued turns");
    assert_eq!(
        detail
            .get("thread_read")
            .and_then(|thread_read| thread_read.get("queued_turns")),
        Some(&serde_json::Value::Array(queued_turns.clone())),
        "thread_read queued_turns must hydrate from the same ordered snapshot"
    );
    queued_turns
}

#[tokio::test]
async fn queued_turn_read_model_reindexes_after_pop_front_resume() {
    let backend = Arc::new(ExternallyTerminatedBackend::default());
    let core = RuntimeCore::with_backend(backend.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_queue_order".to_string()),
        thread_id: Some("thread_queue_order".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn_admitted(
        AgentSessionTurnStartParams {
            session_id: "sess_queue_order".to_string(),
            turn_id: Some("turn_running".to_string()),
            input: AgentInput {
                text: "running".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("running turn");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_queue_order".to_string(),
            turn_id: Some("turn_queued_first".to_string()),
            input: AgentInput {
                text: "first queued steer".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: true,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("first queued turn");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_queue_order".to_string(),
            turn_id: Some("turn_queued_second".to_string()),
            input: AgentInput {
                text: "second queued steer".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: true,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("second queued turn");

    let queued_before_resume = read_queued_turns(&core, "sess_queue_order").await;
    assert_eq!(
        queued_ids(&queued_before_resume),
        vec!["turn_queued_first", "turn_queued_second"]
    );
    assert_eq!(queued_positions(&queued_before_resume), vec![0, 1]);
    assert_eq!(
        queued_before_resume[0]["message_text"],
        "first queued steer"
    );
    assert_eq!(
        queued_before_resume[1]["message_text"],
        "second queued steer"
    );

    core.append_external_runtime_events(
        "sess_queue_order",
        Some("turn_running"),
        vec![RuntimeEvent::new("turn.completed", json!({}))],
    )
    .expect("complete running turn");
    backend.release_initial.notify_one();
    timeout(Duration::from_secs(2), async {
        loop {
            let requests = backend
                .requests
                .lock()
                .expect("test backend requests mutex poisoned");
            if requests
                .iter()
                .any(|request| request.turn.turn_id == "turn_queued_first")
            {
                break;
            }
            drop(requests);
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("automatic queue resume");

    let queued_after_resume = read_queued_turns(&core, "sess_queue_order").await;
    assert_eq!(queued_ids(&queued_after_resume), vec!["turn_queued_second"]);
    assert_eq!(queued_positions(&queued_after_resume), vec![0]);
    assert_eq!(
        queued_after_resume[0]["message_text"], "second queued steer",
        "pop-front must not downgrade the remaining pending steer snapshot"
    );

    let requests = backend
        .requests
        .lock()
        .expect("test backend requests mutex poisoned");
    let resumed_request = requests
        .iter()
        .find(|request| request.turn.turn_id == "turn_queued_first")
        .expect("resumed queued request");
    assert_eq!(resumed_request.input.concat_text(), "first queued steer");
    assert_eq!(
        resumed_request.queued_turn_id.as_deref(),
        Some("turn_queued_first")
    );
}
