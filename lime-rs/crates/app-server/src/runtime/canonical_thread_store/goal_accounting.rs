//! Durable per-turn accounting for canonical thread goals.
//!
//! The projection owner binds a turn to an exact goal identity and the cumulative token snapshot
//! observed when goal work starts. Later usage samples are applied transactionally with their
//! source sequence, so replay cannot charge the same sample twice and a replaced goal cannot
//! inherit an older turn's usage. Goal updates and their notification outbox entry commit together.

use super::super::thread_goal::GoalAccountingMode;
use super::super::thread_usage::{goal_token_delta_since, TokenUsageSnapshot};
use super::goal::ThreadGoalStoreError;
use app_server_protocol::protocol::v2::{
    ThreadGoal, ThreadGoalStatus, ThreadGoalUpdatedNotification,
};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};

pub(super) const GOAL_ACCOUNTING_SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS thread_goal_turn_accounting (
    thread_id TEXT NOT NULL REFERENCES canonical_threads(thread_id) ON DELETE CASCADE,
    turn_id TEXT NOT NULL, goal_id TEXT NOT NULL,
    turn_mode TEXT NOT NULL CHECK(turn_mode IN ('default', 'plan')),
    started_at_ms INTEGER NOT NULL CHECK(started_at_ms >= 0),
    last_accounted_time_seconds INTEGER NOT NULL DEFAULT 0 CHECK(last_accounted_time_seconds >= 0),
    last_input_tokens INTEGER NOT NULL CHECK(last_input_tokens >= 0),
    last_cached_input_tokens INTEGER NOT NULL CHECK(last_cached_input_tokens >= 0),
    last_output_tokens INTEGER NOT NULL CHECK(last_output_tokens >= 0),
    last_reasoning_output_tokens INTEGER NOT NULL CHECK(last_reasoning_output_tokens >= 0),
    last_total_tokens INTEGER NOT NULL CHECK(last_total_tokens >= 0),
    last_source_sequence INTEGER NOT NULL CHECK(last_source_sequence >= 0),
    terminal_sequence INTEGER CHECK(terminal_sequence IS NULL OR terminal_sequence >= 0),
    PRIMARY KEY (thread_id, turn_id)
);
CREATE TABLE IF NOT EXISTS thread_goal_update_outbox (
    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL REFERENCES canonical_threads(thread_id) ON DELETE CASCADE,
    turn_id TEXT NOT NULL, goal_id TEXT NOT NULL,
    source_sequence INTEGER NOT NULL CHECK(source_sequence >= 0),
    notification_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    delivered_at_ms INTEGER,
    UNIQUE (thread_id, turn_id, source_sequence)
);
CREATE INDEX IF NOT EXISTS idx_thread_goal_update_outbox_pending
    ON thread_goal_update_outbox(outbox_id) WHERE delivered_at_ms IS NULL;
"#;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GoalTurnMode {
    Default,
    Plan,
}

impl GoalTurnMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Plan => "plan",
        }
    }

    fn parse(value: &str) -> Result<Self, ThreadGoalStoreError> {
        match value {
            "default" => Ok(Self::Default),
            "plan" => Ok(Self::Plan),
            other => Err(store_error(format!(
                "unknown goal turn accounting mode `{other}`"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GoalTerminalStatus {
    Blocked,
    UsageLimited,
}

pub(super) struct BindGoalTurn<'a> {
    pub thread_id: &'a str,
    pub turn_id: &'a str,
    pub expected_goal_id: &'a str,
    pub turn_mode: GoalTurnMode,
    pub source_sequence: u64,
    pub token_usage_at_start: &'a TokenUsageSnapshot,
    pub started_at_ms: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GoalTurnBindOutcome {
    Bound,
    Replayed,
    GoalUnavailable,
}

pub(super) struct AccountGoalTurnUsage<'a> {
    pub thread_id: &'a str,
    pub turn_id: &'a str,
    pub expected_goal_id: &'a str,
    pub source_sequence: u64,
    pub token_usage: &'a TokenUsageSnapshot,
    pub observed_at_ms: i64,
    pub status_scope: GoalAccountingMode,
    pub terminal: bool,
    pub terminal_status: Option<GoalTerminalStatus>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GoalUsageIgnored {
    Replayed,
    PlanMode,
    UnboundTurn,
    GoalMismatch,
    StatusFiltered,
    EmptyDelta,
    TurnClosed,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum GoalTurnUsageOutcome {
    Updated(GoalUpdateOutboxRecord),
    Unchanged(GoalUsageIgnored),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct GoalUpdateOutboxRecord {
    pub outbox_id: i64,
    pub goal_id: String,
    pub source_sequence: u64,
    pub notification: ThreadGoalUpdatedNotification,
}

struct AppliedGoalUsageUpdate {
    goal_id: String,
    source_sequence: u64,
    notification: ThreadGoalUpdatedNotification,
    created_at_ms: i64,
}

enum AppliedGoalTurnUsage {
    Updated(AppliedGoalUsageUpdate),
    Unchanged(GoalUsageIgnored),
}

#[derive(Debug)]
struct PersistedTurnAccounting {
    goal_id: String,
    turn_mode: GoalTurnMode,
    started_at_ms: i64,
    last_accounted_time_seconds: i64,
    last_token_usage: TokenUsageSnapshot,
    last_source_sequence: u64,
    terminal_sequence: Option<u64>,
}

#[derive(Debug)]
struct PersistedGoal {
    thread_id: String,
    objective: String,
    status: ThreadGoalStatus,
    token_budget: Option<i64>,
    tokens_used: i64,
    time_used_seconds: i64,
    created_at_ms: i64,
    updated_at_ms: i64,
}

#[cfg(test)]
pub(super) fn bind_goal_turn(
    conn: &mut Connection,
    input: BindGoalTurn<'_>,
) -> Result<GoalTurnBindOutcome, ThreadGoalStoreError> {
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(store_error)?;
    let outcome = bind_goal_turn_in_tx(&tx, input)?;
    tx.commit().map_err(store_error)?;
    Ok(outcome)
}

pub(super) fn bind_goal_turn_in_tx(
    conn: &Connection,
    input: BindGoalTurn<'_>,
) -> Result<GoalTurnBindOutcome, ThreadGoalStoreError> {
    let thread_id = required_identity(input.thread_id, "thread id")?;
    let turn_id = required_identity(input.turn_id, "turn id")?;
    let expected_goal_id = required_identity(input.expected_goal_id, "goal id")?;
    validate_usage(input.token_usage_at_start)?;
    if input.started_at_ms < 0 {
        return Err(invalid_request("goal turn start time must not be negative"));
    }
    let source_sequence = sequence_to_i64(input.source_sequence)?;

    if let Some(existing) = read_turn_accounting(conn, thread_id, turn_id)? {
        let same_identity = existing.goal_id == expected_goal_id
            && existing.turn_mode == input.turn_mode
            && existing.started_at_ms == input.started_at_ms;
        let replayed = same_identity
            && input.source_sequence <= existing.last_source_sequence
            && (input.source_sequence < existing.last_source_sequence
                || existing.last_token_usage == *input.token_usage_at_start);
        return if replayed {
            Ok(GoalTurnBindOutcome::Replayed)
        } else {
            Err(invalid_request(format!(
                "turn {turn_id} is already bound to a different goal accounting baseline"
            )))
        };
    }

    let goal_available = conn
        .query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM thread_goals
                 WHERE thread_id = ?1 AND goal_id = ?2
                   AND status IN ('active', 'budget_limited')
             )",
            params![thread_id, expected_goal_id],
            |row| row.get::<_, bool>(0),
        )
        .map_err(store_error)?;
    if !goal_available {
        return Ok(GoalTurnBindOutcome::GoalUnavailable);
    }

    conn.execute(
        r#"INSERT INTO thread_goal_turn_accounting (
               thread_id, turn_id, goal_id, turn_mode, started_at_ms,
               last_accounted_time_seconds, last_input_tokens, last_cached_input_tokens,
               last_output_tokens, last_reasoning_output_tokens, last_total_tokens,
               last_source_sequence, terminal_sequence
           ) VALUES (?1, ?2, ?3, ?4, ?5, 0, ?6, ?7, ?8, ?9, ?10, ?11, NULL)"#,
        params![
            thread_id,
            turn_id,
            expected_goal_id,
            input.turn_mode.as_str(),
            input.started_at_ms,
            input.token_usage_at_start.input_tokens,
            input.token_usage_at_start.cached_input_tokens,
            input.token_usage_at_start.output_tokens,
            input.token_usage_at_start.reasoning_output_tokens,
            input.token_usage_at_start.total_tokens,
            source_sequence,
        ],
    )
    .map_err(store_error)?;
    Ok(GoalTurnBindOutcome::Bound)
}

pub(super) fn account_goal_turn_usage(
    conn: &mut Connection,
    input: AccountGoalTurnUsage<'_>,
) -> Result<GoalTurnUsageOutcome, ThreadGoalStoreError> {
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(store_error)?;
    let applied = apply_goal_turn_usage_in_tx(&tx, input, false)?;
    let outcome = match applied {
        AppliedGoalTurnUsage::Updated(update) => {
            let notification_json =
                serde_json::to_string(&update.notification).map_err(store_error)?;
            tx.execute(
                r#"INSERT INTO thread_goal_update_outbox (
                       thread_id, turn_id, goal_id, source_sequence,
                       notification_json, created_at_ms
                   ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)"#,
                params![
                    update.notification.thread_id,
                    update.notification.turn_id,
                    update.goal_id,
                    sequence_to_i64(update.source_sequence)?,
                    notification_json,
                    update.created_at_ms,
                ],
            )
            .map_err(store_error)?;
            GoalTurnUsageOutcome::Updated(GoalUpdateOutboxRecord {
                outbox_id: tx.last_insert_rowid(),
                goal_id: update.goal_id,
                source_sequence: update.source_sequence,
                notification: update.notification,
            })
        }
        AppliedGoalTurnUsage::Unchanged(reason) => GoalTurnUsageOutcome::Unchanged(reason),
    };
    tx.commit().map_err(store_error)?;
    Ok(outcome)
}

pub(super) fn flush_goal_turn_usage_in_tx(
    conn: &Connection,
    input: AccountGoalTurnUsage<'_>,
) -> Result<(), ThreadGoalStoreError> {
    apply_goal_turn_usage_in_tx(conn, input, true).map(|_| ())
}

fn apply_goal_turn_usage_in_tx(
    conn: &Connection,
    input: AccountGoalTurnUsage<'_>,
    allow_same_sequence_time: bool,
) -> Result<AppliedGoalTurnUsage, ThreadGoalStoreError> {
    let thread_id = required_identity(input.thread_id, "thread id")?;
    let turn_id = required_identity(input.turn_id, "turn id")?;
    let expected_goal_id = required_identity(input.expected_goal_id, "goal id")?;
    validate_usage(input.token_usage)?;
    if input.observed_at_ms < 0 {
        return Err(invalid_request(
            "goal usage observation time must not be negative",
        ));
    }
    if input.terminal_status.is_some() && !input.terminal {
        return Err(invalid_request(
            "goal terminal status requires a terminal usage sample",
        ));
    }
    let source_sequence = sequence_to_i64(input.source_sequence)?;

    let Some(turn) = read_turn_accounting(conn, thread_id, turn_id)? else {
        return Ok(AppliedGoalTurnUsage::Unchanged(
            GoalUsageIgnored::UnboundTurn,
        ));
    };
    if turn.goal_id != expected_goal_id {
        return Ok(AppliedGoalTurnUsage::Unchanged(
            GoalUsageIgnored::GoalMismatch,
        ));
    }
    if input.source_sequence < turn.last_source_sequence
        || (input.source_sequence == turn.last_source_sequence && !allow_same_sequence_time)
    {
        return Ok(AppliedGoalTurnUsage::Unchanged(GoalUsageIgnored::Replayed));
    }
    if turn.terminal_sequence.is_some() {
        return Ok(AppliedGoalTurnUsage::Unchanged(
            GoalUsageIgnored::TurnClosed,
        ));
    }

    let total_time_seconds = input
        .observed_at_ms
        .saturating_sub(turn.started_at_ms)
        .max(0)
        .div_euclid(1_000);
    let time_delta_seconds = total_time_seconds
        .saturating_sub(turn.last_accounted_time_seconds)
        .max(0);
    let token_delta = goal_token_delta_since(&turn.last_token_usage, input.token_usage).max(0);

    if turn.turn_mode == GoalTurnMode::Plan {
        advance_turn_accounting(
            conn,
            thread_id,
            turn_id,
            input.token_usage,
            total_time_seconds,
            source_sequence,
            input.terminal,
        )?;
        return Ok(AppliedGoalTurnUsage::Unchanged(GoalUsageIgnored::PlanMode));
    }

    let Some(current_goal) = read_exact_goal(conn, thread_id, expected_goal_id)? else {
        advance_turn_accounting(
            conn,
            thread_id,
            turn_id,
            input.token_usage,
            total_time_seconds,
            source_sequence,
            input.terminal,
        )?;
        return Ok(AppliedGoalTurnUsage::Unchanged(
            GoalUsageIgnored::GoalMismatch,
        ));
    };
    if !status_is_accountable(input.status_scope, current_goal.status) {
        advance_turn_accounting(
            conn,
            thread_id,
            turn_id,
            input.token_usage,
            total_time_seconds,
            source_sequence,
            input.terminal,
        )?;
        return Ok(AppliedGoalTurnUsage::Unchanged(
            GoalUsageIgnored::StatusFiltered,
        ));
    }

    let accounted_status = if status_can_become_budget_limited(current_goal.status)
        && current_goal
            .token_budget
            .is_some_and(|budget| current_goal.tokens_used.saturating_add(token_delta) >= budget)
    {
        ThreadGoalStatus::BudgetLimited
    } else {
        current_goal.status
    };
    let next_status = terminal_goal_status(accounted_status, input.terminal_status);
    if token_delta == 0 && time_delta_seconds == 0 && next_status == current_goal.status {
        advance_turn_accounting(
            conn,
            thread_id,
            turn_id,
            input.token_usage,
            total_time_seconds,
            source_sequence,
            input.terminal,
        )?;
        return Ok(AppliedGoalTurnUsage::Unchanged(
            GoalUsageIgnored::EmptyDelta,
        ));
    }
    let updated_at_ms = input.observed_at_ms.max(current_goal.updated_at_ms);
    let updated_rows = conn
        .execute(
            r#"UPDATE thread_goals
               SET tokens_used = tokens_used + ?1,
                   time_used_seconds = time_used_seconds + ?2,
                   status = ?3,
                   updated_at_ms = ?4
               WHERE thread_id = ?5 AND goal_id = ?6 AND status = ?7"#,
            params![
                token_delta,
                time_delta_seconds,
                status_as_str(next_status),
                updated_at_ms,
                thread_id,
                expected_goal_id,
                status_as_str(current_goal.status),
            ],
        )
        .map_err(store_error)?;
    if updated_rows != 1 {
        return Err(store_error(format!(
            "exact goal accounting update affected {updated_rows} rows"
        )));
    }

    let updated_goal = read_exact_goal(conn, thread_id, expected_goal_id)?
        .ok_or_else(|| store_error("updated goal disappeared during accounting"))?;
    let notification = ThreadGoalUpdatedNotification {
        thread_id: thread_id.to_string(),
        turn_id: Some(turn_id.to_string()),
        goal: project_goal(&updated_goal),
    };
    advance_turn_accounting(
        conn,
        thread_id,
        turn_id,
        input.token_usage,
        total_time_seconds,
        source_sequence,
        input.terminal,
    )?;

    Ok(AppliedGoalTurnUsage::Updated(AppliedGoalUsageUpdate {
        goal_id: expected_goal_id.to_string(),
        source_sequence: input.source_sequence,
        notification,
        created_at_ms: updated_at_ms,
    }))
}

#[cfg(test)]
pub(super) fn pending_goal_updates(
    conn: &Connection,
    limit: u32,
) -> Result<Vec<GoalUpdateOutboxRecord>, ThreadGoalStoreError> {
    let mut statement = conn
        .prepare(
            r#"SELECT outbox_id, goal_id, source_sequence, notification_json
               FROM thread_goal_update_outbox
               WHERE delivered_at_ms IS NULL
               ORDER BY outbox_id ASC
               LIMIT ?1"#,
        )
        .map_err(store_error)?;
    let rows = statement
        .query_map(params![i64::from(limit)], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .map_err(store_error)?;
    rows.map(|row| {
        let (outbox_id, goal_id, source_sequence, notification_json) = row.map_err(store_error)?;
        Ok(GoalUpdateOutboxRecord {
            outbox_id,
            goal_id,
            source_sequence: sequence_from_i64(source_sequence)?,
            notification: serde_json::from_str(&notification_json).map_err(store_error)?,
        })
    })
    .collect()
}

pub(super) fn pending_goal_updates_for_thread(
    conn: &Connection,
    thread_id: &str,
    through_outbox_id: Option<i64>,
) -> Result<Vec<GoalUpdateOutboxRecord>, ThreadGoalStoreError> {
    let thread_id = required_identity(thread_id, "thread id")?;
    if through_outbox_id.is_some_and(|outbox_id| outbox_id <= 0) {
        return Err(invalid_request("goal update outbox id must be positive"));
    }
    let mut statement = conn
        .prepare(
            r#"SELECT outbox_id, goal_id, source_sequence, notification_json
               FROM thread_goal_update_outbox
               WHERE thread_id = ?1 AND delivered_at_ms IS NULL
                 AND (?2 IS NULL OR outbox_id <= ?2)
               ORDER BY outbox_id ASC"#,
        )
        .map_err(store_error)?;
    let rows = statement
        .query_map(params![thread_id, through_outbox_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .map_err(store_error)?;
    rows.map(|row| {
        let (outbox_id, goal_id, source_sequence, notification_json) = row.map_err(store_error)?;
        Ok(GoalUpdateOutboxRecord {
            outbox_id,
            goal_id,
            source_sequence: sequence_from_i64(source_sequence)?,
            notification: serde_json::from_str(&notification_json).map_err(store_error)?,
        })
    })
    .collect()
}

pub(super) fn mark_goal_update_delivered(
    conn: &Connection,
    outbox_id: i64,
    delivered_at_ms: i64,
) -> Result<bool, ThreadGoalStoreError> {
    if outbox_id <= 0 {
        return Err(invalid_request("goal update outbox id must be positive"));
    }
    if delivered_at_ms < 0 {
        return Err(invalid_request(
            "goal update delivery time must not be negative",
        ));
    }
    conn.execute(
        r#"UPDATE thread_goal_update_outbox
           SET delivered_at_ms = ?2
           WHERE outbox_id = ?1 AND delivered_at_ms IS NULL"#,
        params![outbox_id, delivered_at_ms],
    )
    .map(|updated| updated == 1)
    .map_err(store_error)
}

pub(super) fn mark_thread_goal_updates_delivered_through(
    conn: &Connection,
    thread_id: &str,
    through_outbox_id: i64,
    delivered_at_ms: i64,
) -> Result<usize, ThreadGoalStoreError> {
    let thread_id = required_identity(thread_id, "thread id")?;
    if through_outbox_id <= 0 {
        return Err(invalid_request("goal update outbox id must be positive"));
    }
    if delivered_at_ms < 0 {
        return Err(invalid_request(
            "goal update delivery time must not be negative",
        ));
    }
    conn.execute(
        r#"UPDATE thread_goal_update_outbox
           SET delivered_at_ms = ?3
           WHERE thread_id = ?1 AND outbox_id <= ?2 AND delivered_at_ms IS NULL"#,
        params![thread_id, through_outbox_id, delivered_at_ms],
    )
    .map_err(store_error)
}

fn read_turn_accounting(
    conn: &Connection,
    thread_id: &str,
    turn_id: &str,
) -> Result<Option<PersistedTurnAccounting>, ThreadGoalStoreError> {
    conn.query_row(
        r#"SELECT goal_id, turn_mode, started_at_ms, last_accounted_time_seconds,
                  last_input_tokens, last_cached_input_tokens, last_output_tokens,
                  last_reasoning_output_tokens, last_total_tokens, last_source_sequence,
                  terminal_sequence
           FROM thread_goal_turn_accounting
           WHERE thread_id = ?1 AND turn_id = ?2"#,
        params![thread_id, turn_id],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                TokenUsageSnapshot {
                    input_tokens: row.get(4)?,
                    cached_input_tokens: row.get(5)?,
                    cache_write_input_tokens: 0,
                    output_tokens: row.get(6)?,
                    reasoning_output_tokens: row.get(7)?,
                    total_tokens: row.get(8)?,
                },
                row.get::<_, i64>(9)?,
                row.get::<_, Option<i64>>(10)?,
            ))
        },
    )
    .optional()
    .map_err(store_error)?
    .map(
        |(
            goal_id,
            turn_mode,
            started_at_ms,
            last_accounted_time_seconds,
            last_token_usage,
            last_source_sequence,
            terminal_sequence,
        )| {
            Ok(PersistedTurnAccounting {
                goal_id,
                turn_mode: GoalTurnMode::parse(&turn_mode)?,
                started_at_ms,
                last_accounted_time_seconds,
                last_token_usage,
                last_source_sequence: sequence_from_i64(last_source_sequence)?,
                terminal_sequence: terminal_sequence.map(sequence_from_i64).transpose()?,
            })
        },
    )
    .transpose()
}

fn read_exact_goal(
    conn: &Connection,
    thread_id: &str,
    goal_id: &str,
) -> Result<Option<PersistedGoal>, ThreadGoalStoreError> {
    conn.query_row(
        r#"SELECT thread_id, objective, status, token_budget, tokens_used,
                  time_used_seconds, created_at_ms, updated_at_ms
           FROM thread_goals WHERE thread_id = ?1 AND goal_id = ?2"#,
        params![thread_id, goal_id],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<i64>>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, i64>(7)?,
            ))
        },
    )
    .optional()
    .map_err(store_error)?
    .map(
        |(
            thread_id,
            objective,
            status,
            token_budget,
            tokens_used,
            time_used_seconds,
            created_at_ms,
            updated_at_ms,
        )| {
            Ok(PersistedGoal {
                thread_id,
                objective,
                status: parse_status(&status)?,
                token_budget,
                tokens_used,
                time_used_seconds,
                created_at_ms,
                updated_at_ms,
            })
        },
    )
    .transpose()
}

#[allow(clippy::too_many_arguments)]
fn advance_turn_accounting(
    conn: &Connection,
    thread_id: &str,
    turn_id: &str,
    usage: &TokenUsageSnapshot,
    total_time_seconds: i64,
    source_sequence: i64,
    terminal: bool,
) -> Result<(), ThreadGoalStoreError> {
    let updated = conn
        .execute(
            r#"UPDATE thread_goal_turn_accounting
               SET last_accounted_time_seconds = ?1,
                   last_input_tokens = ?2,
                   last_cached_input_tokens = ?3,
                   last_output_tokens = ?4,
                   last_reasoning_output_tokens = ?5,
                   last_total_tokens = ?6,
                   last_source_sequence = ?7,
                   terminal_sequence = CASE WHEN ?8 THEN ?7 ELSE terminal_sequence END
               WHERE thread_id = ?9 AND turn_id = ?10"#,
            params![
                total_time_seconds,
                usage.input_tokens,
                usage.cached_input_tokens,
                usage.output_tokens,
                usage.reasoning_output_tokens,
                usage.total_tokens,
                source_sequence,
                terminal,
                thread_id,
                turn_id,
            ],
        )
        .map_err(store_error)?;
    if updated != 1 {
        return Err(store_error(format!(
            "goal turn accounting update affected {updated} rows"
        )));
    }
    Ok(())
}

fn project_goal(goal: &PersistedGoal) -> ThreadGoal {
    ThreadGoal {
        thread_id: goal.thread_id.clone(),
        objective: goal.objective.clone(),
        status: goal.status,
        token_budget: goal.token_budget,
        tokens_used: goal.tokens_used,
        time_used_seconds: goal.time_used_seconds,
        created_at: goal.created_at_ms.div_euclid(1_000),
        updated_at: goal.updated_at_ms.div_euclid(1_000),
    }
}

fn status_is_accountable(mode: GoalAccountingMode, status: ThreadGoalStatus) -> bool {
    match mode {
        GoalAccountingMode::ActiveOnly => matches!(
            status,
            ThreadGoalStatus::Active | ThreadGoalStatus::BudgetLimited
        ),
    }
}

fn status_can_become_budget_limited(status: ThreadGoalStatus) -> bool {
    status == ThreadGoalStatus::Active
}

fn terminal_goal_status(
    status: ThreadGoalStatus,
    terminal_status: Option<GoalTerminalStatus>,
) -> ThreadGoalStatus {
    match (status, terminal_status) {
        (ThreadGoalStatus::Active, Some(GoalTerminalStatus::Blocked)) => ThreadGoalStatus::Blocked,
        (
            ThreadGoalStatus::Active | ThreadGoalStatus::BudgetLimited,
            Some(GoalTerminalStatus::UsageLimited),
        ) => ThreadGoalStatus::UsageLimited,
        _ => status,
    }
}

fn parse_status(value: &str) -> Result<ThreadGoalStatus, ThreadGoalStoreError> {
    match value {
        "active" => Ok(ThreadGoalStatus::Active),
        "paused" => Ok(ThreadGoalStatus::Paused),
        "blocked" => Ok(ThreadGoalStatus::Blocked),
        "usage_limited" => Ok(ThreadGoalStatus::UsageLimited),
        "budget_limited" => Ok(ThreadGoalStatus::BudgetLimited),
        "complete" => Ok(ThreadGoalStatus::Complete),
        other => Err(store_error(format!(
            "unknown persisted thread goal status `{other}`"
        ))),
    }
}

fn status_as_str(status: ThreadGoalStatus) -> &'static str {
    match status {
        ThreadGoalStatus::Active => "active",
        ThreadGoalStatus::Paused => "paused",
        ThreadGoalStatus::Blocked => "blocked",
        ThreadGoalStatus::UsageLimited => "usage_limited",
        ThreadGoalStatus::BudgetLimited => "budget_limited",
        ThreadGoalStatus::Complete => "complete",
    }
}

fn validate_usage(usage: &TokenUsageSnapshot) -> Result<(), ThreadGoalStoreError> {
    if [
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.output_tokens,
        usage.reasoning_output_tokens,
        usage.total_tokens,
    ]
    .into_iter()
    .any(|value| value < 0)
    {
        return Err(invalid_request(
            "goal accounting token counters must not be negative",
        ));
    }
    Ok(())
}

fn required_identity<'a>(value: &'a str, field: &str) -> Result<&'a str, ThreadGoalStoreError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(invalid_request(format!("{field} must not be empty")));
    }
    Ok(value)
}

fn sequence_to_i64(value: u64) -> Result<i64, ThreadGoalStoreError> {
    i64::try_from(value).map_err(|_| invalid_request("source sequence exceeds SQLite range"))
}

fn sequence_from_i64(value: i64) -> Result<u64, ThreadGoalStoreError> {
    u64::try_from(value).map_err(|_| store_error("persisted source sequence is negative"))
}

fn invalid_request(message: impl Into<String>) -> ThreadGoalStoreError {
    ThreadGoalStoreError::InvalidRequest(message.into())
}

fn store_error(source: impl std::fmt::Display) -> ThreadGoalStoreError {
    ThreadGoalStoreError::Store(source.to_string())
}

#[cfg(test)]
#[path = "goal_accounting_tests.rs"]
mod tests;
