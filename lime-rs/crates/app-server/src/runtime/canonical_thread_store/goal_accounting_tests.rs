use super::*;

fn usage(input: i64, cached: i64, output: i64) -> TokenUsageSnapshot {
    TokenUsageSnapshot {
        input_tokens: input,
        cached_input_tokens: cached,
        cache_write_input_tokens: 0,
        output_tokens: output,
        reasoning_output_tokens: 0,
        total_tokens: input.saturating_add(output),
    }
}

fn test_connection() -> Connection {
    let conn = Connection::open_in_memory().expect("open in-memory goal accounting store");
    install_test_schema(&conn);
    conn
}

fn install_test_schema(conn: &Connection) {
    conn.execute_batch(
        r#"PRAGMA foreign_keys = ON;
           CREATE TABLE canonical_threads (thread_id TEXT PRIMARY KEY);
           CREATE TABLE thread_goals (
               thread_id TEXT PRIMARY KEY NOT NULL REFERENCES canonical_threads(thread_id)
                   ON DELETE CASCADE,
               goal_id TEXT NOT NULL, objective TEXT NOT NULL, status TEXT NOT NULL,
               token_budget INTEGER, tokens_used INTEGER NOT NULL DEFAULT 0,
               time_used_seconds INTEGER NOT NULL DEFAULT 0, created_at_ms INTEGER NOT NULL,
               updated_at_ms INTEGER NOT NULL
           );"#,
    )
    .expect("create base goal tables");
    conn.execute_batch(GOAL_ACCOUNTING_SCHEMA_SQL)
        .expect("create goal accounting tables");
    conn.execute(
        "INSERT INTO canonical_threads (thread_id) VALUES ('thread-1')",
        [],
    )
    .expect("insert canonical thread");
}

fn insert_goal(conn: &Connection, goal_id: &str, status: &str, token_budget: Option<i64>) {
    conn.execute(
        r#"INSERT INTO thread_goals (
               thread_id, goal_id, objective, status, token_budget, tokens_used,
               time_used_seconds, created_at_ms, updated_at_ms
           ) VALUES ('thread-1', ?1, 'finish the task', ?2, ?3, 0, 0, 1000, 1000)
           ON CONFLICT(thread_id) DO UPDATE SET
               goal_id = excluded.goal_id,
               status = excluded.status,
               token_budget = excluded.token_budget,
               tokens_used = 0,
               time_used_seconds = 0"#,
        params![goal_id, status, token_budget],
    )
    .expect("insert thread goal");
}

fn bind(
    conn: &mut Connection,
    goal_id: &str,
    turn_id: &str,
    turn_mode: GoalTurnMode,
    baseline: &TokenUsageSnapshot,
) -> GoalTurnBindOutcome {
    bind_goal_turn(
        conn,
        BindGoalTurn {
            thread_id: "thread-1",
            turn_id,
            expected_goal_id: goal_id,
            turn_mode,
            source_sequence: 10,
            token_usage_at_start: baseline,
            started_at_ms: 2_000,
        },
    )
    .expect("bind goal turn")
}

fn account(
    conn: &mut Connection,
    goal_id: &str,
    turn_id: &str,
    source_sequence: u64,
    current: &TokenUsageSnapshot,
    observed_at_ms: i64,
    terminal: bool,
) -> GoalTurnUsageOutcome {
    account_with_terminal_status(
        conn,
        goal_id,
        turn_id,
        source_sequence,
        current,
        observed_at_ms,
        terminal,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn account_with_terminal_status(
    conn: &mut Connection,
    goal_id: &str,
    turn_id: &str,
    source_sequence: u64,
    current: &TokenUsageSnapshot,
    observed_at_ms: i64,
    terminal: bool,
    terminal_status: Option<GoalTerminalStatus>,
) -> GoalTurnUsageOutcome {
    account_goal_turn_usage(
        conn,
        AccountGoalTurnUsage {
            thread_id: "thread-1",
            turn_id,
            expected_goal_id: goal_id,
            source_sequence,
            token_usage: current,
            observed_at_ms,
            status_scope: GoalAccountingMode::ActiveOnly,
            terminal,
            terminal_status,
        },
    )
    .expect("account goal turn usage")
}

fn goal_usage(conn: &Connection) -> (String, String, i64, i64) {
    conn.query_row(
        "SELECT goal_id, status, tokens_used, time_used_seconds FROM thread_goals",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    )
    .expect("read goal usage")
}

#[test]
fn turn_binding_is_exact_baselined_and_replay_safe() {
    let mut conn = test_connection();
    insert_goal(&conn, "goal-old", "active", Some(100));
    assert_eq!(
        bind(
            &mut conn,
            "goal-old",
            "turn-1",
            GoalTurnMode::Default,
            &usage(0, 0, 0)
        ),
        GoalTurnBindOutcome::Bound
    );
    insert_goal(&conn, "goal-new", "active", Some(100));

    let outcome = account(
        &mut conn,
        "goal-old",
        "turn-1",
        11,
        &usage(25, 0, 5),
        5_000,
        true,
    );

    assert_eq!(
        outcome,
        GoalTurnUsageOutcome::Unchanged(GoalUsageIgnored::GoalMismatch)
    );
    assert_eq!(
        goal_usage(&conn),
        ("goal-new".to_string(), "active".to_string(), 0, 0)
    );
    assert!(pending_goal_updates(&conn, 10)
        .expect("read outbox")
        .is_empty());

    insert_goal(&conn, "goal-1", "active", None);
    let baseline = usage(100, 10, 30);
    assert_eq!(
        bind(
            &mut conn,
            "goal-1",
            "turn-2",
            GoalTurnMode::Default,
            &baseline
        ),
        GoalTurnBindOutcome::Bound
    );

    let outcome = account(
        &mut conn,
        "goal-1",
        "turn-2",
        11,
        &usage(120, 14, 42),
        2_500,
        false,
    );
    let GoalTurnUsageOutcome::Updated(update) = outcome else {
        panic!("usage after the turn-start baseline should update the goal");
    };
    assert_eq!(update.notification.goal.tokens_used, 28);
    assert_eq!(goal_usage(&conn).2, 28);
    let replay = account(
        &mut conn,
        "goal-1",
        "turn-2",
        11,
        &usage(120, 14, 42),
        2_500,
        false,
    );
    assert_eq!(
        replay,
        GoalTurnUsageOutcome::Unchanged(GoalUsageIgnored::Replayed)
    );
    assert_eq!(
        pending_goal_updates(&conn, 10).expect("read outbox").len(),
        1
    );
}

#[test]
fn plan_turn_records_replay_cursor_without_charging_goal() {
    let mut conn = test_connection();
    insert_goal(&conn, "goal-1", "active", None);
    bind(
        &mut conn,
        "goal-1",
        "turn-plan",
        GoalTurnMode::Plan,
        &usage(0, 0, 0),
    );
    let sample = usage(50, 10, 20);

    let first = account(&mut conn, "goal-1", "turn-plan", 11, &sample, 6_000, true);
    let replay = account(&mut conn, "goal-1", "turn-plan", 11, &sample, 6_000, true);

    assert_eq!(
        first,
        GoalTurnUsageOutcome::Unchanged(GoalUsageIgnored::PlanMode)
    );
    assert_eq!(
        replay,
        GoalTurnUsageOutcome::Unchanged(GoalUsageIgnored::Replayed)
    );
    assert_eq!(goal_usage(&conn).2, 0);
    assert!(pending_goal_updates(&conn, 10)
        .expect("read outbox")
        .is_empty());
}

#[test]
fn terminal_sample_accounts_token_and_elapsed_time_delta_then_closes_turn() {
    let mut conn = test_connection();
    insert_goal(&conn, "goal-1", "active", None);
    bind(
        &mut conn,
        "goal-1",
        "turn-1",
        GoalTurnMode::Default,
        &usage(10, 2, 3),
    );

    let outcome = account(
        &mut conn,
        "goal-1",
        "turn-1",
        11,
        &usage(30, 5, 11),
        7_900,
        true,
    );
    let after_terminal = account(
        &mut conn,
        "goal-1",
        "turn-1",
        12,
        &usage(40, 5, 15),
        9_000,
        false,
    );

    let GoalTurnUsageOutcome::Updated(update) = outcome else {
        panic!("terminal usage should update the goal");
    };
    assert_eq!(update.notification.goal.tokens_used, 25);
    assert_eq!(update.notification.goal.time_used_seconds, 5);
    assert_eq!(
        after_terminal,
        GoalTurnUsageOutcome::Unchanged(GoalUsageIgnored::TurnClosed)
    );
}

#[test]
fn budget_limited_goal_keeps_counting_in_flight_turn_usage() {
    let mut conn = test_connection();
    insert_goal(&conn, "goal-1", "active", Some(25));
    bind(
        &mut conn,
        "goal-1",
        "turn-1",
        GoalTurnMode::Default,
        &usage(0, 0, 0),
    );

    account(
        &mut conn,
        "goal-1",
        "turn-1",
        11,
        &usage(20, 5, 10),
        3_000,
        false,
    );
    assert_eq!(
        goal_usage(&conn),
        ("goal-1".to_string(), "budget_limited".to_string(), 25, 1)
    );

    let outcome = account(
        &mut conn,
        "goal-1",
        "turn-1",
        12,
        &usage(24, 5, 16),
        5_000,
        true,
    );

    let GoalTurnUsageOutcome::Updated(update) = outcome else {
        panic!("in-flight budget-limited usage should remain accountable");
    };
    assert_eq!(
        update.notification.goal.status,
        ThreadGoalStatus::BudgetLimited
    );
    assert_eq!(update.notification.goal.tokens_used, 35);
    assert_eq!(update.notification.goal.time_used_seconds, 3);
}

#[test]
fn terminal_error_updates_status_without_fabricating_usage() {
    let mut conn = test_connection();
    insert_goal(&conn, "goal-1", "active", None);
    bind(
        &mut conn,
        "goal-1",
        "turn-1",
        GoalTurnMode::Default,
        &usage(0, 0, 0),
    );

    let outcome = account_with_terminal_status(
        &mut conn,
        "goal-1",
        "turn-1",
        11,
        &usage(0, 0, 0),
        2_000,
        true,
        Some(GoalTerminalStatus::Blocked),
    );

    let GoalTurnUsageOutcome::Updated(update) = outcome else {
        panic!("terminal error should update the goal status");
    };
    assert_eq!(update.notification.goal.status, ThreadGoalStatus::Blocked);
    assert_eq!(update.notification.goal.tokens_used, 0);
    assert_eq!(update.notification.goal.time_used_seconds, 0);
    assert_eq!(goal_usage(&conn).1, "blocked");
}

#[test]
fn terminal_status_obeys_budget_limited_and_plan_guards() {
    let mut blocked_conn = test_connection();
    insert_goal(&blocked_conn, "goal-1", "budget_limited", Some(1));
    bind(
        &mut blocked_conn,
        "goal-1",
        "turn-1",
        GoalTurnMode::Default,
        &usage(0, 0, 0),
    );
    let blocked = account_with_terminal_status(
        &mut blocked_conn,
        "goal-1",
        "turn-1",
        11,
        &usage(0, 0, 0),
        2_000,
        true,
        Some(GoalTerminalStatus::Blocked),
    );
    assert_eq!(
        blocked,
        GoalTurnUsageOutcome::Unchanged(GoalUsageIgnored::EmptyDelta)
    );
    assert_eq!(goal_usage(&blocked_conn).1, "budget_limited");

    let mut limited_conn = test_connection();
    insert_goal(&limited_conn, "goal-1", "budget_limited", Some(1));
    bind(
        &mut limited_conn,
        "goal-1",
        "turn-1",
        GoalTurnMode::Default,
        &usage(0, 0, 0),
    );
    let limited = account_with_terminal_status(
        &mut limited_conn,
        "goal-1",
        "turn-1",
        11,
        &usage(0, 0, 0),
        2_000,
        true,
        Some(GoalTerminalStatus::UsageLimited),
    );
    let GoalTurnUsageOutcome::Updated(update) = limited else {
        panic!("usage limit should replace budget-limited status");
    };
    assert_eq!(
        update.notification.goal.status,
        ThreadGoalStatus::UsageLimited
    );

    let mut plan_conn = test_connection();
    insert_goal(&plan_conn, "goal-1", "active", None);
    bind(
        &mut plan_conn,
        "goal-1",
        "turn-plan",
        GoalTurnMode::Plan,
        &usage(0, 0, 0),
    );
    let plan = account_with_terminal_status(
        &mut plan_conn,
        "goal-1",
        "turn-plan",
        11,
        &usage(0, 0, 0),
        2_000,
        true,
        Some(GoalTerminalStatus::UsageLimited),
    );
    assert_eq!(
        plan,
        GoalTurnUsageOutcome::Unchanged(GoalUsageIgnored::PlanMode)
    );
    assert_eq!(goal_usage(&plan_conn).1, "active");
}

#[test]
fn durable_update_outbox_survives_restart_until_delivery_is_acknowledged() {
    let temp = tempfile::tempdir().expect("create tempdir");
    let path = temp.path().join("goal-accounting.sqlite");
    let mut conn = Connection::open(&path).expect("open durable goal store");
    install_test_schema(&conn);
    insert_goal(&conn, "goal-1", "active", None);
    bind(
        &mut conn,
        "goal-1",
        "turn-1",
        GoalTurnMode::Default,
        &usage(0, 0, 0),
    );
    account(
        &mut conn,
        "goal-1",
        "turn-1",
        11,
        &usage(20, 5, 8),
        4_000,
        true,
    );
    drop(conn);

    let conn = Connection::open(&path).expect("reopen durable goal store");
    let pending = pending_goal_updates(&conn, 10).expect("read durable outbox");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].goal_id, "goal-1");
    assert_eq!(pending[0].source_sequence, 11);
    assert_eq!(pending[0].notification.turn_id.as_deref(), Some("turn-1"));
    assert!(
        mark_goal_update_delivered(&conn, pending[0].outbox_id, 5_000)
            .expect("acknowledge delivery")
    );
    assert!(pending_goal_updates(&conn, 10)
        .expect("read outbox")
        .is_empty());
}

#[test]
fn thread_outbox_preserves_order_and_resume_watermark_does_not_ack_new_updates() {
    let mut conn = test_connection();
    insert_goal(&conn, "goal-1", "active", None);
    bind(
        &mut conn,
        "goal-1",
        "turn-1",
        GoalTurnMode::Default,
        &usage(0, 0, 0),
    );
    let GoalTurnUsageOutcome::Updated(first) = account(
        &mut conn,
        "goal-1",
        "turn-1",
        11,
        &usage(20, 5, 8),
        4_000,
        true,
    ) else {
        panic!("first terminal sample should update the goal");
    };

    bind(
        &mut conn,
        "goal-1",
        "turn-2",
        GoalTurnMode::Default,
        &usage(20, 5, 8),
    );
    let GoalTurnUsageOutcome::Updated(second) = account(
        &mut conn,
        "goal-1",
        "turn-2",
        12,
        &usage(32, 7, 14),
        6_000,
        true,
    ) else {
        panic!("second terminal sample should update the goal");
    };

    let ordered = pending_goal_updates_for_thread(&conn, "thread-1", Some(second.outbox_id))
        .expect("read ordered thread outbox");
    assert_eq!(
        ordered
            .iter()
            .map(|update| update.outbox_id)
            .collect::<Vec<_>>(),
        vec![first.outbox_id, second.outbox_id]
    );

    assert_eq!(
        mark_thread_goal_updates_delivered_through(&conn, "thread-1", first.outbox_id, 7_000)
            .expect("acknowledge captured resume watermark"),
        1
    );
    let remaining =
        pending_goal_updates_for_thread(&conn, "thread-1", None).expect("read post-resume outbox");
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].outbox_id, second.outbox_id);
}
