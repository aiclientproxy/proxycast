use super::*;
use crate::runtime::thread_usage::TokenUsageSnapshot;
use agent_protocol::{SessionId, Thread, ThreadId, ThreadStatus, ThreadTurnsView};
use app_server_protocol::protocol::v2::{ThreadGoalSetParams, ThreadGoalStatus};
use app_server_protocol::{AgentEvent, AgentSession, AgentSessionStatus};
use futures::executor::block_on;
use serde_json::json;
use std::collections::HashMap;
use thread_store::{CreateThreadParams, ThreadStore};

fn thread(id: &str) -> Thread {
    Thread {
        session_id: SessionId::new(format!("session-{id}")),
        thread_id: ThreadId::new(id),
        status: ThreadStatus::Idle,
        created_at_ms: 1,
        updated_at_ms: 1,
        archived: false,
        recency_at_ms: None,
        parent_thread_id: None,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
        last_task_message: None,
        agent_state: None,
        forked_from_id: None,
        preview: String::new(),
        model_provider: "test".to_string(),
        product: None,
        name: None,
        metadata: json!({}),
        turns: Vec::new(),
        turns_view: ThreadTurnsView::NotLoaded,
    }
}

fn stored(thread: &Thread) -> StoredSession {
    StoredSession {
        session: AgentSession {
            session_id: thread.session_id.to_string(),
            thread_id: thread.thread_id.to_string(),
            app_id: "agent-chat".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: AgentSessionStatus::Running,
            created_at: "2026-07-20T00:00:00Z".to_string(),
            updated_at: "2026-07-20T00:00:05Z".to_string(),
        },
        turns: Vec::new(),
        turn_inputs: HashMap::new(),
        turn_runtime_options: HashMap::new(),
        events: Vec::new(),
        output_blobs: HashMap::new(),
    }
}

fn event(
    stored: &StoredSession,
    turn_id: &str,
    sequence: u64,
    event_type: &str,
    timestamp: &str,
    payload: serde_json::Value,
) -> AgentEvent {
    AgentEvent {
        event_id: format!("{turn_id}-{sequence}"),
        sequence,
        session_id: stored.session.session_id.clone(),
        thread_id: Some(stored.session.thread_id.clone()),
        turn_id: Some(turn_id.to_string()),
        event_type: event_type.to_string(),
        timestamp: timestamp.to_string(),
        payload,
    }
}

fn usage(total_input: i64, cached_input: i64, output: i64) -> serde_json::Value {
    json!({
        "usage": {
            "total_token_usage": {
                "input_tokens": total_input,
                "cached_input_tokens": cached_input,
                "output_tokens": output,
                "reasoning_output_tokens": 0,
                "total_tokens": total_input + output
            },
            "last_token_usage": {
                "input_tokens": total_input,
                "cached_input_tokens": cached_input,
                "output_tokens": output,
                "reasoning_output_tokens": 0,
                "total_tokens": total_input + output
            },
            "model_context_window": 128000
        }
    })
}

fn provider_usage(
    attempt: u32,
    total_input: i64,
    cached_input: i64,
    output: i64,
) -> serde_json::Value {
    let mut payload = usage(total_input, cached_input, output);
    let object = payload
        .as_object_mut()
        .expect("provider usage payload must be an object");
    object.insert("attempt".to_string(), json!(attempt));
    object.insert("backend".to_string(), json!("runtime"));
    payload
}

fn tool_finish_payload(
    stored: &StoredSession,
    turn_id: &str,
    sequence: u64,
    status: &str,
    handler_executed: bool,
) -> serde_json::Value {
    json!({
        "item": {
            "sessionId": stored.session.session_id,
            "threadId": stored.session.thread_id,
            "turnId": turn_id,
            "itemId": format!("tool-{sequence}"),
            "sequence": sequence,
            "ordinal": 1,
            "createdAtMs": 1_784_521_600_000_i64,
            "updatedAtMs": 1_784_521_604_000_i64,
            "completedAtMs": 1_784_521_604_000_i64,
            "kind": "tool",
            "status": status,
            "payload": {
                "type": "tool",
                "call_id": format!("call-{sequence}"),
                "name": "exec_command",
                "arguments": [],
                "output": {
                    "text": "done",
                    "truncated": false
                }
            },
            "metadata": {
                tool_runtime::tool_result_projection::TOOL_HANDLER_EXECUTED_METADATA_KEY:
                    handler_executed
            }
        }
    })
}

fn store_with_goal(id: &str) -> (tempfile::TempDir, ProjectionStore, Thread, StoredSession) {
    let temp = tempfile::tempdir().expect("tempdir");
    let store =
        ProjectionStore::initialize(temp.path().join("state.sqlite")).expect("projection store");
    let thread = thread(id);
    block_on(store.create_thread(CreateThreadParams {
        thread: thread.clone(),
    }))
    .expect("create canonical thread");
    store
        .set_thread_goal_sync(ThreadGoalSetParams {
            thread_id: thread.thread_id.to_string(),
            objective: Some("finish the goal projection".to_string()),
            status: None,
            token_budget: None,
        })
        .expect("set thread goal");
    let stored = stored(&thread);
    (temp, store, thread, stored)
}

fn timestamp_ms(value: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(value)
        .expect("valid test timestamp")
        .timestamp_millis()
}

fn token_snapshot(
    input_tokens: i64,
    cached_input_tokens: i64,
    output_tokens: i64,
) -> TokenUsageSnapshot {
    TokenUsageSnapshot {
        input_tokens,
        cached_input_tokens,
        cache_write_input_tokens: 0,
        output_tokens,
        reasoning_output_tokens: 0,
        total_tokens: input_tokens + output_tokens,
    }
}

fn active_binding(
    turn_id: &str,
    plan_mode: bool,
    source_sequence: u64,
    usage: TokenUsageSnapshot,
    observed_at: &str,
) -> ActiveTurnGoalBinding {
    ActiveTurnGoalBinding {
        turn_id: turn_id.to_string(),
        plan_mode,
        source_sequence,
        token_usage: usage,
        observed_at_ms: timestamp_ms(observed_at),
    }
}

fn goal_id(store: &ProjectionStore, thread_id: &str) -> String {
    store
        .open_thread_store()
        .expect("open goal identity store")
        .query_row(
            "SELECT goal_id FROM thread_goals WHERE thread_id = ?1",
            rusqlite::params![thread_id],
            |row| row.get(0),
        )
        .expect("read goal identity")
}

#[test]
fn idle_usage_flushes_at_turn_admission_and_terminal_rearms_active_goal() {
    let (_temp, store, thread, stored) = store_with_goal("goal-idle-admission");
    let turn_id = "turn-idle-admission";
    // Leave one whole-second margin for the event timestamp's millisecond projection boundary.
    store.backdate_thread_goal_idle_for_test(thread.thread_id.as_str(), 6);
    let accepted_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Nanos, true);
    let accepted = event(
        &stored,
        turn_id,
        1,
        "turn.accepted",
        &accepted_at,
        json!({"goalAccountingMode": "default"}),
    );

    store
        .apply_canonical_events(&stored, std::slice::from_ref(&accepted))
        .expect("apply turn admission");
    assert!(!store.thread_goal_idle_is_active_for_test(thread.thread_id.as_str()));
    let admitted = store
        .get_thread_goal_sync(thread.thread_id.as_str())
        .expect("read admitted goal")
        .expect("admitted goal");
    assert!((5..=6).contains(&admitted.time_used_seconds));
    assert_eq!(admitted.tokens_used, 0);
    let admitted_time_seconds = admitted.time_used_seconds;

    store
        .apply_canonical_events(&stored, std::slice::from_ref(&accepted))
        .expect("replay turn admission");
    assert_eq!(
        store
            .get_thread_goal_sync(thread.thread_id.as_str())
            .expect("read replayed goal")
            .expect("replayed goal")
            .time_used_seconds,
        admitted_time_seconds
    );

    let completed = event(
        &stored,
        turn_id,
        2,
        "turn.completed",
        &accepted_at,
        usage(0, 0, 0),
    );
    store
        .apply_canonical_events(&stored, std::slice::from_ref(&completed))
        .expect("complete admitted turn");
    assert!(store.thread_goal_idle_is_active_for_test(thread.thread_id.as_str()));
    let conn = store.open_thread_store().expect("open idle outbox");
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM thread_goal_update_outbox WHERE source_sequence = 1",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("count idle admission outbox"),
        1
    );
}

#[test]
fn delayed_turn_admission_does_not_overlap_idle_and_turn_time() {
    let (_temp, store, thread, stored) = store_with_goal("goal-idle-delayed-admission");
    store.backdate_thread_goal_idle_for_test(thread.thread_id.as_str(), 6);
    let accepted_at = (chrono::Utc::now() - chrono::Duration::seconds(2))
        .to_rfc3339_opts(chrono::SecondsFormat::Nanos, true);
    let accepted = event(
        &stored,
        "turn-idle-delayed-admission",
        1,
        "turn.accepted",
        &accepted_at,
        json!({"goalAccountingMode": "default"}),
    );

    store
        .apply_canonical_events(&stored, std::slice::from_ref(&accepted))
        .expect("apply delayed turn admission");
    let goal = store
        .get_thread_goal_sync(thread.thread_id.as_str())
        .expect("read delayed-admission goal")
        .expect("delayed-admission goal");
    assert!((3..=4).contains(&goal.time_used_seconds));

    let completed_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Nanos, true);
    let completed = event(
        &stored,
        "turn-idle-delayed-admission",
        2,
        "turn.completed",
        &completed_at,
        usage(0, 0, 0),
    );
    store
        .apply_canonical_events(&stored, std::slice::from_ref(&completed))
        .expect("complete delayed turn");
    let completed_goal = store
        .get_thread_goal_sync(thread.thread_id.as_str())
        .expect("read completed delayed-admission goal")
        .expect("completed delayed-admission goal");
    assert!((5..=6).contains(&completed_goal.time_used_seconds));
}

#[test]
fn idle_usage_and_turn_binding_roll_back_together() {
    let (_temp, store, thread, stored) = store_with_goal("goal-idle-bind-rollback");
    store.backdate_thread_goal_idle_for_test(thread.thread_id.as_str(), 5);
    let conn = store.open_thread_store().expect("open idle rollback store");
    conn.execute_batch(
        r#"CREATE TRIGGER fail_goal_turn_bind
           BEFORE INSERT ON thread_goal_turn_accounting
           BEGIN
               SELECT RAISE(ABORT, 'injected goal turn bind failure');
           END;"#,
    )
    .expect("install goal turn bind failure");
    drop(conn);

    let accepted_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Nanos, true);
    let accepted = event(
        &stored,
        "turn-idle-bind-rollback",
        1,
        "turn.accepted",
        &accepted_at,
        json!({"goalAccountingMode": "default"}),
    );
    let error = store
        .apply_canonical_events(&stored, std::slice::from_ref(&accepted))
        .expect_err("injected bind failure must reject admission accounting");
    assert!(error.contains("injected goal turn bind failure"));
    let goal = store
        .get_thread_goal_sync(thread.thread_id.as_str())
        .expect("read rolled-back idle goal")
        .expect("rolled-back idle goal");
    assert_eq!(goal.time_used_seconds, 0);
    assert!(store.thread_goal_idle_is_active_for_test(thread.thread_id.as_str()));
    let conn = store
        .open_thread_store()
        .expect("open rolled-back idle store");
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM thread_goal_update_outbox",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("count rolled-back idle outbox"),
        0
    );
}

#[test]
fn plan_terminal_does_not_rearm_idle_goal_accounting() {
    let (_temp, store, thread, stored) = store_with_goal("goal-idle-plan");
    let turn_id = "turn-idle-plan";
    store.backdate_thread_goal_idle_for_test(thread.thread_id.as_str(), 5);
    let events = [
        event(
            &stored,
            turn_id,
            1,
            "turn.accepted",
            "2026-07-20T00:00:00Z",
            json!({"goalAccountingMode": "plan"}),
        ),
        event(
            &stored,
            turn_id,
            2,
            "turn.completed",
            "2026-07-20T00:00:05Z",
            usage(50, 10, 20),
        ),
    ];

    store
        .apply_canonical_events(&stored, &events)
        .expect("apply plan turn");
    assert!(!store.thread_goal_idle_is_active_for_test(thread.thread_id.as_str()));
    let goal = store
        .get_thread_goal_sync(thread.thread_id.as_str())
        .expect("read plan goal")
        .expect("plan goal");
    assert_eq!(goal.tokens_used, 0);
    assert_eq!(goal.time_used_seconds, 0);
}

#[test]
fn durable_plan_marker_prevents_goal_charging_without_runtime_options() {
    let (_temp, store, thread, stored) = store_with_goal("goal-plan-durable");
    let turn_id = "turn-plan-durable";
    let events = [
        event(
            &stored,
            turn_id,
            1,
            "turn.accepted",
            "2026-07-20T00:00:00Z",
            json!({"goalAccountingMode": "plan"}),
        ),
        event(
            &stored,
            turn_id,
            2,
            "turn.completed",
            "2026-07-20T00:00:05Z",
            usage(40, 10, 20),
        ),
    ];

    store
        .apply_canonical_events(&stored, &events)
        .expect("apply durable plan accounting events");

    let goal = store
        .get_thread_goal_sync(thread.thread_id.as_str())
        .expect("read plan goal")
        .expect("plan goal");
    assert_eq!(goal.tokens_used, 0);
    assert_eq!(goal.time_used_seconds, 0);
    let conn = store.open_thread_store().expect("open goal store");
    let persisted: (String, Option<i64>) = conn
        .query_row(
            "SELECT turn_mode, terminal_sequence FROM thread_goal_turn_accounting",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("read durable plan accounting state");
    assert_eq!(persisted, ("plan".to_string(), Some(2)));
}

#[test]
fn successful_tool_finish_flushes_active_goal_progress_exactly_once() {
    let (_temp, store, thread, stored) = store_with_goal("goal-tool-finish");
    let turn_id = "turn-tool-finish";
    let events = [
        event(
            &stored,
            turn_id,
            1,
            "turn.accepted",
            "2026-07-20T00:00:00Z",
            json!({"goalAccountingMode": "default"}),
        ),
        event(
            &stored,
            turn_id,
            2,
            "provider.usage",
            "2026-07-20T00:00:03Z",
            provider_usage(1, 20, 5, 10),
        ),
        event(
            &stored,
            turn_id,
            3,
            "item.completed",
            "2026-07-20T00:00:04Z",
            tool_finish_payload(&stored, turn_id, 3, "completed", true),
        ),
    ];

    store
        .apply_canonical_events(&stored, &events)
        .expect("apply tool-finish accounting batch");
    store
        .apply_canonical_events(&stored, &events)
        .expect("replay tool-finish accounting batch");

    let goal = store
        .get_thread_goal_sync(thread.thread_id.as_str())
        .expect("read tool-finish goal")
        .expect("tool-finish goal");
    assert_eq!(goal.tokens_used, 25);
    assert_eq!(goal.time_used_seconds, 4);
    let conn = store.open_thread_store().expect("open tool-finish store");
    let accounting: (i64, Option<i64>) = conn
        .query_row(
            "SELECT last_source_sequence, terminal_sequence FROM thread_goal_turn_accounting",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("read tool-finish accounting state");
    assert_eq!(accounting, (3, None));
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM thread_goal_update_outbox",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("count tool-finish outbox"),
        1
    );
}

#[test]
fn failed_tool_finish_counts_only_after_handler_execution() {
    for handler_executed in [false, true] {
        let suffix = if handler_executed {
            "executed"
        } else {
            "blocked"
        };
        let (_temp, store, thread, stored) =
            store_with_goal(&format!("goal-failed-tool-finish-{suffix}"));
        let turn_id = format!("turn-failed-tool-finish-{suffix}");
        let events = [
            event(
                &stored,
                &turn_id,
                1,
                "turn.accepted",
                "2026-07-20T00:00:00Z",
                json!({"goalAccountingMode": "default"}),
            ),
            event(
                &stored,
                &turn_id,
                2,
                "provider.usage",
                "2026-07-20T00:00:03Z",
                provider_usage(1, 20, 5, 10),
            ),
            event(
                &stored,
                &turn_id,
                3,
                "item.completed",
                "2026-07-20T00:00:04Z",
                tool_finish_payload(&stored, &turn_id, 3, "failed", handler_executed),
            ),
        ];

        store
            .apply_canonical_events(&stored, &events)
            .expect("apply failed tool-finish batch");

        let goal = store
            .get_thread_goal_sync(thread.thread_id.as_str())
            .expect("read failed tool-finish goal")
            .expect("failed tool-finish goal");
        assert_eq!(goal.tokens_used, if handler_executed { 25 } else { 0 });
        assert_eq!(goal.time_used_seconds, if handler_executed { 4 } else { 0 });
        let conn = store
            .open_thread_store()
            .expect("open failed tool-finish store");
        assert_eq!(
            conn.query_row(
                "SELECT COUNT(*) FROM thread_goal_update_outbox",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("count failed tool-finish outbox"),
            i64::from(handler_executed)
        );
    }
}

#[test]
fn aborted_tool_finish_does_not_count_after_handler_started() {
    let (_temp, store, thread, stored) = store_with_goal("goal-aborted-tool-finish");
    let turn_id = "turn-aborted-tool-finish";
    let mut aborted_item = tool_finish_payload(&stored, turn_id, 3, "cancelled", true);
    aborted_item["item"]["metadata"]
        [tool_runtime::tool_result_projection::TOOL_OUTCOME_METADATA_KEY] =
        json!(tool_runtime::tool_result_projection::TOOL_OUTCOME_ABORTED);
    let events = [
        event(
            &stored,
            turn_id,
            1,
            "turn.accepted",
            "2026-07-20T00:00:00Z",
            json!({"goalAccountingMode": "default"}),
        ),
        event(
            &stored,
            turn_id,
            2,
            "provider.usage",
            "2026-07-20T00:00:03Z",
            provider_usage(1, 20, 5, 10),
        ),
        event(
            &stored,
            turn_id,
            3,
            "item.completed",
            "2026-07-20T00:00:04Z",
            aborted_item,
        ),
    ];

    store
        .apply_canonical_events(&stored, &events)
        .expect("apply aborted tool-finish batch");

    let goal = store
        .get_thread_goal_sync(thread.thread_id.as_str())
        .expect("read aborted tool-finish goal")
        .expect("aborted tool-finish goal");
    assert_eq!(goal.tokens_used, 0);
    assert_eq!(goal.time_used_seconds, 0);
    let conn = store
        .open_thread_store()
        .expect("open aborted tool-finish store");
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM thread_goal_update_outbox",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("count aborted tool-finish outbox"),
        0
    );
}

#[test]
fn failed_and_canceled_turns_account_batch_usage_then_close_exactly_once() {
    for terminal in ["turn.failed", "turn.canceled"] {
        let id = terminal.replace('.', "-");
        let (_temp, store, thread, stored) = store_with_goal(&format!("goal-{id}"));
        let turn_id = format!("turn-{id}");
        let events = [
            event(
                &stored,
                &turn_id,
                1,
                "turn.accepted",
                "2026-07-20T00:00:00Z",
                json!({"goalAccountingMode": "default"}),
            ),
            event(
                &stored,
                &turn_id,
                2,
                "provider.usage",
                "2026-07-20T00:00:03Z",
                provider_usage(1, 20, 5, 10),
            ),
            event(
                &stored,
                &turn_id,
                3,
                terminal,
                "2026-07-20T00:00:05Z",
                json!({}),
            ),
        ];

        store
            .apply_canonical_events(&stored, &events)
            .expect("apply terminal accounting batch");
        store
            .apply_canonical_events(&stored, &events)
            .expect("replay terminal accounting batch");

        let goal = store
            .get_thread_goal_sync(thread.thread_id.as_str())
            .expect("read terminal goal")
            .expect("terminal goal");
        assert_eq!(goal.tokens_used, 25, "terminal={terminal}");
        assert_eq!(goal.time_used_seconds, 5, "terminal={terminal}");
        assert_eq!(
            goal.status,
            if terminal == "turn.failed" {
                app_server_protocol::protocol::v2::ThreadGoalStatus::Blocked
            } else {
                app_server_protocol::protocol::v2::ThreadGoalStatus::Active
            },
            "terminal={terminal}"
        );
        let conn = store.open_thread_store().expect("open goal store");
        let terminal_sequence: Option<i64> = conn
            .query_row(
                "SELECT terminal_sequence FROM thread_goal_turn_accounting",
                [],
                |row| row.get(0),
            )
            .expect("read terminal sequence");
        assert_eq!(terminal_sequence, Some(3), "terminal={terminal}");
        let outbox_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM thread_goal_update_outbox",
                [],
                |row| row.get(0),
            )
            .expect("count goal update outbox");
        assert_eq!(outbox_count, 1, "terminal={terminal}");
    }
}

#[test]
fn usage_limit_failure_transitions_goal_exactly_once() {
    let (_temp, store, thread, stored) = store_with_goal("goal-usage-limited");
    let turn_id = "turn-usage-limited";
    let events = [
        event(
            &stored,
            turn_id,
            1,
            "turn.accepted",
            "2026-07-20T00:00:00Z",
            json!({"goalAccountingMode": "default"}),
        ),
        event(
            &stored,
            turn_id,
            2,
            "provider.usage",
            "2026-07-20T00:00:03Z",
            provider_usage(1, 20, 5, 10),
        ),
        event(
            &stored,
            turn_id,
            3,
            "turn.failed",
            "2026-07-20T00:00:05Z",
            json!({"reason": "usage_limit_exceeded"}),
        ),
    ];

    store
        .apply_canonical_events(&stored, &events)
        .expect("apply usage-limit accounting batch");
    store
        .apply_canonical_events(&stored, &events)
        .expect("replay usage-limit accounting batch");

    let goal = store
        .get_thread_goal_sync(thread.thread_id.as_str())
        .expect("read usage-limited goal")
        .expect("usage-limited goal");
    assert_eq!(goal.tokens_used, 25);
    assert_eq!(goal.time_used_seconds, 5);
    assert_eq!(
        goal.status,
        app_server_protocol::protocol::v2::ThreadGoalStatus::UsageLimited
    );
    let conn = store.open_thread_store().expect("open goal store");
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM thread_goal_update_outbox",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("count usage-limit outbox"),
        1
    );
}

#[test]
fn goal_created_during_turn_binds_current_baseline_without_charging_prior_usage() {
    for plan_mode in [false, true] {
        let suffix = if plan_mode { "plan" } else { "default" };
        let temp = tempfile::tempdir().expect("late goal tempdir");
        let store = ProjectionStore::initialize(temp.path().join("state.sqlite"))
            .expect("late goal projection store");
        let thread = thread(&format!("goal-late-bind-{suffix}"));
        block_on(store.create_thread(CreateThreadParams {
            thread: thread.clone(),
        }))
        .expect("create late goal thread");
        let stored = stored(&thread);
        let turn_id = format!("turn-late-bind-{suffix}");
        let before_goal = vec![
            event(
                &stored,
                &turn_id,
                1,
                "turn.accepted",
                "2026-07-20T00:00:00Z",
                json!({
                    "goalAccountingMode": if plan_mode { "plan" } else { "default" }
                }),
            ),
            event(
                &stored,
                &turn_id,
                2,
                "provider.usage",
                "2026-07-20T00:00:03Z",
                provider_usage(1, 20, 5, 10),
            ),
        ];
        store
            .apply_canonical_events(&stored, &before_goal)
            .expect("apply events before goal creation");

        let binding = ActiveTurnGoalBinding {
            turn_id: turn_id.clone(),
            plan_mode,
            source_sequence: 2,
            token_usage: TokenUsageSnapshot {
                input_tokens: 20,
                cached_input_tokens: 5,
                cache_write_input_tokens: 0,
                output_tokens: 10,
                reasoning_output_tokens: 0,
                total_tokens: 30,
            },
            observed_at_ms: timestamp_ms("2026-07-20T00:00:03Z"),
        };
        let params = ThreadGoalSetParams {
            thread_id: thread.thread_id.to_string(),
            objective: Some("finish work created during the turn".to_string()),
            status: None,
            token_budget: None,
        };
        store
            .set_thread_goal_with_active_turn_sync(params.clone(), Some(&binding))
            .expect("create and bind late goal");
        store
            .set_thread_goal_with_active_turn_sync(params, Some(&binding))
            .expect("replay late goal mutation");
        store
            .apply_canonical_events(&stored, &before_goal)
            .expect("replay pre-goal events without resetting late baseline");

        let mut stored_after_goal = stored.clone();
        stored_after_goal.events = before_goal;
        let after_goal = [
            event(
                &stored_after_goal,
                &turn_id,
                3,
                "provider.usage",
                "2026-07-20T00:00:06Z",
                provider_usage(2, 30, 5, 15),
            ),
            event(
                &stored_after_goal,
                &turn_id,
                4,
                "turn.completed",
                "2026-07-20T00:00:08Z",
                json!({}),
            ),
        ];
        store
            .apply_canonical_events(&stored_after_goal, &after_goal)
            .expect("account usage after late goal creation");

        let goal = store
            .get_thread_goal_sync(thread.thread_id.as_str())
            .expect("read late-bound goal")
            .expect("late-bound goal");
        assert_eq!(
            goal.tokens_used,
            if plan_mode { 0 } else { 15 },
            "mode={suffix}"
        );
        assert_eq!(
            goal.time_used_seconds,
            if plan_mode { 0 } else { 5 },
            "mode={suffix}"
        );
        let conn = store.open_thread_store().expect("open late goal store");
        let persisted: (i64, i64, Option<i64>) = conn
            .query_row(
                r#"SELECT last_input_tokens, last_source_sequence, terminal_sequence
                   FROM thread_goal_turn_accounting
                   WHERE thread_id = ?1 AND turn_id = ?2"#,
                rusqlite::params![thread.thread_id.as_str(), turn_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("read late goal accounting baseline");
        assert_eq!(persisted, (30, 4, Some(4)), "mode={suffix}");
    }
}

#[test]
fn active_goal_patch_flushes_then_resets_baseline_without_stale_outbox() {
    for plan_mode in [false, true] {
        let suffix = if plan_mode { "plan" } else { "default" };
        let (_temp, store, thread, stored) = store_with_goal(&format!("goal-patch-{suffix}"));
        let turn_id = format!("turn-patch-{suffix}");
        let before_mutation = vec![
            event(
                &stored,
                &turn_id,
                1,
                "turn.accepted",
                "2026-07-20T00:00:00Z",
                json!({
                    "goalAccountingMode": if plan_mode { "plan" } else { "default" }
                }),
            ),
            event(
                &stored,
                &turn_id,
                2,
                "provider.usage",
                "2026-07-20T00:00:03Z",
                provider_usage(1, 20, 5, 10),
            ),
        ];
        store
            .apply_canonical_events(&stored, &before_mutation)
            .expect("bind goal before active patch");
        let original_goal_id = goal_id(&store, thread.thread_id.as_str());
        let binding = active_binding(
            &turn_id,
            plan_mode,
            2,
            token_snapshot(20, 5, 10),
            "2026-07-20T00:00:03Z",
        );
        let params = ThreadGoalSetParams {
            thread_id: thread.thread_id.to_string(),
            objective: Some(format!("patched {suffix} objective")),
            status: None,
            token_budget: None,
        };

        let patched = store
            .set_thread_goal_with_active_turn_sync(params.clone(), Some(&binding))
            .expect("flush and patch active goal");
        store
            .set_thread_goal_with_active_turn_sync(params, Some(&binding))
            .expect("replay active goal patch");
        assert_eq!(goal_id(&store, thread.thread_id.as_str()), original_goal_id);
        assert_eq!(patched.tokens_used, if plan_mode { 0 } else { 25 });
        assert_eq!(patched.time_used_seconds, if plan_mode { 0 } else { 3 });

        let conn = store
            .open_thread_store()
            .expect("open patched accounting store");
        let baseline: (String, i64, i64, i64, Option<i64>) = conn
            .query_row(
                r#"SELECT goal_id, started_at_ms, last_input_tokens,
                          last_source_sequence, terminal_sequence
                   FROM thread_goal_turn_accounting
                   WHERE thread_id = ?1 AND turn_id = ?2"#,
                rusqlite::params![thread.thread_id.as_str(), turn_id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .expect("read patched accounting baseline");
        assert_eq!(
            baseline,
            (
                original_goal_id.clone(),
                timestamp_ms("2026-07-20T00:00:03Z"),
                20,
                2,
                None,
            )
        );
        assert_eq!(
            conn.query_row(
                "SELECT COUNT(*) FROM thread_goal_update_outbox",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("count mutation outbox"),
            0
        );
        drop(conn);

        let delayed_patch = store
            .set_thread_goal_with_active_turn_sync(
                ThreadGoalSetParams {
                    thread_id: thread.thread_id.to_string(),
                    objective: Some(format!("delayed {suffix} objective")),
                    status: None,
                    token_budget: None,
                },
                Some(&active_binding(
                    &turn_id,
                    plan_mode,
                    2,
                    token_snapshot(20, 5, 10),
                    "2026-07-20T00:00:05Z",
                )),
            )
            .expect("account time between same-sequence mutations");
        assert_eq!(
            delayed_patch.time_used_seconds,
            if plan_mode { 0 } else { 5 }
        );

        let mut stored_after_mutation = stored.clone();
        stored_after_mutation.events = before_mutation;
        let after_mutation = [
            event(
                &stored_after_mutation,
                &turn_id,
                3,
                "provider.usage",
                "2026-07-20T00:00:06Z",
                provider_usage(2, 30, 5, 15),
            ),
            event(
                &stored_after_mutation,
                &turn_id,
                4,
                "turn.completed",
                "2026-07-20T00:00:08Z",
                json!({}),
            ),
        ];
        store
            .apply_canonical_events(&stored_after_mutation, &after_mutation)
            .expect("account post-patch usage");
        store
            .apply_canonical_events(&stored_after_mutation, &after_mutation)
            .expect("replay post-patch terminal");

        let goal = store
            .get_thread_goal_sync(thread.thread_id.as_str())
            .expect("read patched goal")
            .expect("patched goal");
        assert_eq!(goal.tokens_used, if plan_mode { 0 } else { 40 });
        assert_eq!(goal.time_used_seconds, if plan_mode { 0 } else { 8 });
        let conn = store
            .open_thread_store()
            .expect("open terminal outbox store");
        assert_eq!(
            conn.query_row(
                "SELECT COUNT(*) FROM thread_goal_update_outbox",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("count terminal outbox"),
            if plan_mode { 0 } else { 1 },
        );
    }
}

#[test]
fn pause_resume_excludes_progress_observed_while_goal_is_paused() {
    let (_temp, store, thread, stored) = store_with_goal("goal-pause-resume");
    let turn_id = "turn-pause-resume";
    let before_pause = vec![
        event(
            &stored,
            turn_id,
            1,
            "turn.accepted",
            "2026-07-20T00:00:00Z",
            json!({"goalAccountingMode": "default"}),
        ),
        event(
            &stored,
            turn_id,
            2,
            "provider.usage",
            "2026-07-20T00:00:03Z",
            provider_usage(1, 20, 5, 10),
        ),
    ];
    store
        .apply_canonical_events(&stored, &before_pause)
        .expect("bind goal before pause");
    let paused = store
        .set_thread_goal_with_active_turn_sync(
            ThreadGoalSetParams {
                thread_id: thread.thread_id.to_string(),
                objective: None,
                status: Some(ThreadGoalStatus::Paused),
                token_budget: None,
            },
            Some(&active_binding(
                turn_id,
                false,
                2,
                token_snapshot(20, 5, 10),
                "2026-07-20T00:00:03Z",
            )),
        )
        .expect("pause active goal");
    assert_eq!(paused.tokens_used, 25);
    assert_eq!(paused.time_used_seconds, 3);

    let resume_binding = active_binding(
        turn_id,
        false,
        3,
        token_snapshot(40, 5, 20),
        "2026-07-20T00:00:05Z",
    );
    let resume_params = ThreadGoalSetParams {
        thread_id: thread.thread_id.to_string(),
        objective: None,
        status: Some(ThreadGoalStatus::Active),
        token_budget: None,
    };
    let resumed = store
        .set_thread_goal_with_active_turn_sync(resume_params.clone(), Some(&resume_binding))
        .expect("resume paused goal");
    store
        .set_thread_goal_with_active_turn_sync(resume_params, Some(&resume_binding))
        .expect("replay goal resume");
    assert_eq!(resumed.tokens_used, 25);
    assert_eq!(resumed.time_used_seconds, 3);

    let mut stored_after_resume = stored.clone();
    stored_after_resume.events = before_pause;
    stored_after_resume.events.push(event(
        &stored_after_resume,
        turn_id,
        3,
        "provider.usage",
        "2026-07-20T00:00:05Z",
        provider_usage(1, 40, 5, 20),
    ));
    let after_resume = [
        event(
            &stored_after_resume,
            turn_id,
            4,
            "provider.usage",
            "2026-07-20T00:00:07Z",
            provider_usage(1, 50, 5, 25),
        ),
        event(
            &stored_after_resume,
            turn_id,
            5,
            "turn.completed",
            "2026-07-20T00:00:08Z",
            json!({}),
        ),
    ];
    store
        .apply_canonical_events(&stored_after_resume, &after_resume)
        .expect("account resumed goal usage");

    let goal = store
        .get_thread_goal_sync(thread.thread_id.as_str())
        .expect("read resumed goal")
        .expect("resumed goal");
    assert_eq!(goal.tokens_used, 40);
    assert_eq!(goal.time_used_seconds, 6);
}

#[test]
fn clear_then_recreate_replaces_active_turn_goal_binding() {
    let (_temp, store, thread, stored) = store_with_goal("goal-clear-recreate");
    let turn_id = "turn-clear-recreate";
    let before_clear = vec![
        event(
            &stored,
            turn_id,
            1,
            "turn.accepted",
            "2026-07-20T00:00:00Z",
            json!({"goalAccountingMode": "default"}),
        ),
        event(
            &stored,
            turn_id,
            2,
            "provider.usage",
            "2026-07-20T00:00:03Z",
            provider_usage(1, 20, 5, 10),
        ),
    ];
    store
        .apply_canonical_events(&stored, &before_clear)
        .expect("bind goal before clear");
    let old_goal_id = goal_id(&store, thread.thread_id.as_str());
    let binding = active_binding(
        turn_id,
        false,
        2,
        token_snapshot(20, 5, 10),
        "2026-07-20T00:00:03Z",
    );
    assert!(store
        .clear_thread_goal_with_active_turn_sync(thread.thread_id.as_str(), Some(&binding))
        .expect("clear active goal"));

    let created = store
        .set_thread_goal_with_active_turn_sync(
            ThreadGoalSetParams {
                thread_id: thread.thread_id.to_string(),
                objective: Some("replacement goal".to_string()),
                status: None,
                token_budget: None,
            },
            Some(&binding),
        )
        .expect("create replacement goal");
    let new_goal_id = goal_id(&store, thread.thread_id.as_str());
    assert_ne!(new_goal_id, old_goal_id);
    assert_eq!(created.tokens_used, 0);

    let mut stored_after_recreate = stored.clone();
    stored_after_recreate.events = before_clear;
    let after_recreate = [
        event(
            &stored_after_recreate,
            turn_id,
            3,
            "provider.usage",
            "2026-07-20T00:00:06Z",
            provider_usage(2, 30, 5, 15),
        ),
        event(
            &stored_after_recreate,
            turn_id,
            4,
            "turn.completed",
            "2026-07-20T00:00:08Z",
            json!({}),
        ),
    ];
    store
        .apply_canonical_events(&stored_after_recreate, &after_recreate)
        .expect("account replacement goal usage");

    let goal = store
        .get_thread_goal_sync(thread.thread_id.as_str())
        .expect("read replacement goal")
        .expect("replacement goal");
    assert_eq!(goal.tokens_used, 15);
    assert_eq!(goal.time_used_seconds, 5);
    let conn = store
        .open_thread_store()
        .expect("open replacement accounting store");
    let persisted: (String, i64, Option<i64>) = conn
        .query_row(
            r#"SELECT goal_id, last_source_sequence, terminal_sequence
               FROM thread_goal_turn_accounting
               WHERE thread_id = ?1 AND turn_id = ?2"#,
            rusqlite::params![thread.thread_id.as_str(), turn_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("read replacement binding");
    assert_eq!(persisted, (new_goal_id.clone(), 4, Some(4)));
    let outbox_goal_id: String = conn
        .query_row("SELECT goal_id FROM thread_goal_update_outbox", [], |row| {
            row.get(0)
        })
        .expect("read replacement outbox goal id");
    assert_eq!(outbox_goal_id, new_goal_id);
}

#[test]
fn goal_mutation_failure_rolls_back_flushed_usage_and_baseline() {
    let (_temp, store, thread, stored) = store_with_goal("goal-mutation-rollback");
    let turn_id = "turn-mutation-rollback";
    let before_mutation = vec![
        event(
            &stored,
            turn_id,
            1,
            "turn.accepted",
            "2026-07-20T00:00:00Z",
            json!({"goalAccountingMode": "default"}),
        ),
        event(
            &stored,
            turn_id,
            2,
            "provider.usage",
            "2026-07-20T00:00:03Z",
            provider_usage(1, 20, 5, 10),
        ),
    ];
    store
        .apply_canonical_events(&stored, &before_mutation)
        .expect("bind goal before rollback test");
    let conn = store
        .open_thread_store()
        .expect("open rollback trigger store");
    conn.execute_batch(
        r#"CREATE TRIGGER reject_goal_objective_patch
           BEFORE UPDATE OF objective ON thread_goals
           WHEN NEW.objective = 'reject this patch'
           BEGIN
               SELECT RAISE(ABORT, 'injected goal mutation failure');
           END;"#,
    )
    .expect("install rollback trigger");
    drop(conn);

    let error = store
        .set_thread_goal_with_active_turn_sync(
            ThreadGoalSetParams {
                thread_id: thread.thread_id.to_string(),
                objective: Some("reject this patch".to_string()),
                status: None,
                token_budget: None,
            },
            Some(&active_binding(
                turn_id,
                false,
                2,
                token_snapshot(20, 5, 10),
                "2026-07-20T00:00:03Z",
            )),
        )
        .expect_err("injected mutation must fail");
    assert!(error.to_string().contains("injected goal mutation failure"));
    let stale_error = store
        .set_thread_goal_with_active_turn_sync(
            ThreadGoalSetParams {
                thread_id: thread.thread_id.to_string(),
                objective: Some("stale patch".to_string()),
                status: None,
                token_budget: None,
            },
            Some(&active_binding(
                turn_id,
                false,
                0,
                token_snapshot(0, 0, 0),
                "2026-07-20T00:00:01Z",
            )),
        )
        .expect_err("stale mutation baseline must fail");
    assert!(stale_error.to_string().contains("is before"));

    let goal = store
        .get_thread_goal_sync(thread.thread_id.as_str())
        .expect("read rolled-back goal")
        .expect("rolled-back goal");
    assert_eq!(goal.objective, "finish the goal projection");
    assert_eq!(goal.tokens_used, 0);
    assert_eq!(goal.time_used_seconds, 0);
    let conn = store
        .open_thread_store()
        .expect("open rolled-back accounting store");
    let accounting: (i64, i64, i64) = conn
        .query_row(
            r#"SELECT last_input_tokens, last_source_sequence, last_accounted_time_seconds
               FROM thread_goal_turn_accounting
               WHERE thread_id = ?1 AND turn_id = ?2"#,
            rusqlite::params![thread.thread_id.as_str(), turn_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("read rolled-back accounting");
    assert_eq!(accounting, (0, 1, 0));
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM thread_goal_update_outbox",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("count rolled-back outbox"),
        0
    );
}
