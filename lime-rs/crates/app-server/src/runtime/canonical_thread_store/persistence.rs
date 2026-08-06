use super::*;

pub(super) fn create_thread_store_schema(
    conn: &Connection,
    separate_history_database: bool,
) -> ThreadStoreResult<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS canonical_threads (
            thread_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            thread_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            recency_at_ms INTEGER,
            archived INTEGER NOT NULL DEFAULT 0,
            last_sequence INTEGER,
            rollout_path TEXT
        );
        CREATE TABLE IF NOT EXISTS canonical_thread_spawn_edges (
            parent_thread_id TEXT NOT NULL,
            child_thread_id TEXT NOT NULL PRIMARY KEY,
            status TEXT NOT NULL,
            pending_session_id TEXT
        );
        CREATE TABLE IF NOT EXISTS thread_sections (
            section_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            ordinal INTEGER NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS thread_section_members (
            thread_id TEXT PRIMARY KEY NOT NULL
                REFERENCES canonical_threads(thread_id) ON DELETE CASCADE,
            section_id TEXT NOT NULL
                REFERENCES thread_sections(section_id) ON DELETE CASCADE,
            position INTEGER NOT NULL,
            section_entered_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS thread_goals (
            thread_id TEXT PRIMARY KEY NOT NULL
                REFERENCES canonical_threads(thread_id) ON DELETE CASCADE,
            goal_id TEXT NOT NULL,
            objective TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN (
                'active',
                'paused',
                'blocked',
                'usage_limited',
                'budget_limited',
                'complete'
            )),
            token_budget INTEGER CHECK(token_budget IS NULL OR token_budget > 0),
            tokens_used INTEGER NOT NULL DEFAULT 0 CHECK(tokens_used >= 0),
            time_used_seconds INTEGER NOT NULL DEFAULT 0 CHECK(time_used_seconds >= 0),
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS thread_goal_continuation_deferrals (
            thread_id TEXT PRIMARY KEY NOT NULL
                REFERENCES thread_goals(thread_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_canonical_threads_archive_recency
            ON canonical_threads(archived, recency_at_ms DESC, updated_at_ms DESC, thread_id DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_canonical_threads_session
            ON canonical_threads(session_id);
        CREATE INDEX IF NOT EXISTS idx_canonical_thread_spawn_edges_parent_status
            ON canonical_thread_spawn_edges(parent_thread_id, status);
        CREATE INDEX IF NOT EXISTS idx_thread_section_members_order
            ON thread_section_members(section_id, position, thread_id);
        "#,
    )
    .map_err(store_error)?;
    conn.execute(
        "INSERT OR IGNORE INTO thread_sections (section_id, name, ordinal) VALUES (?1, ?2, 0)",
        params![PINNED_THREAD_SECTION_ID, PINNED_THREAD_SECTION_NAME],
    )
    .map_err(store_error)?;
    conn.execute_batch(super::goal_accounting::GOAL_ACCOUNTING_SCHEMA_SQL)
        .map_err(store_error)?;
    let history_schema = separate_history_database
        .then_some("thread_history.")
        .unwrap_or_default();
    conn.execute_batch(&format!(
        r#"
        CREATE TABLE IF NOT EXISTS {history_schema}canonical_turns (
            thread_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            last_sequence INTEGER NOT NULL,
            turn_json TEXT NOT NULL,
            PRIMARY KEY (thread_id, turn_id),
            UNIQUE (thread_id, ordinal)
        );
        CREATE TABLE IF NOT EXISTS {history_schema}canonical_items (
            thread_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            sequence INTEGER NOT NULL,
            item_json TEXT NOT NULL,
            PRIMARY KEY (thread_id, item_id),
            UNIQUE (thread_id, ordinal),
            FOREIGN KEY (thread_id, turn_id)
                REFERENCES canonical_turns(thread_id, turn_id)
                ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS {history_schema}canonical_history_applies (
            thread_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            PRIMARY KEY (thread_id, sequence)
        );
        CREATE INDEX IF NOT EXISTS {history_schema}idx_canonical_turns_page
            ON canonical_turns(thread_id, ordinal, turn_id);
        CREATE INDEX IF NOT EXISTS {history_schema}idx_canonical_items_page
            ON canonical_items(thread_id, ordinal, item_id);
        CREATE INDEX IF NOT EXISTS {history_schema}idx_canonical_items_turn_page
            ON canonical_items(thread_id, turn_id, ordinal, item_id);
        "#,
    ))
    .map_err(store_error)?;
    ensure_table_column(conn, "canonical_threads", "rollout_path", "TEXT")?;
    ensure_table_column(
        conn,
        "canonical_thread_spawn_edges",
        "pending_session_id",
        "TEXT",
    )?;
    conn.execute_batch(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_canonical_threads_rollout_path
            ON canonical_threads(rollout_path) WHERE rollout_path IS NOT NULL;",
    )
    .map_err(store_error)
}

fn ensure_table_column(
    conn: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> ThreadStoreResult<()> {
    let mut statement = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(store_error)?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(store_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(store_error)?;
    if columns.iter().any(|existing| existing == column) {
        return Ok(());
    }
    conn.execute(
        &format!("ALTER TABLE {table} ADD COLUMN {column} {definition}"),
        [],
    )
    .map_err(store_error)?;
    Ok(())
}

pub(super) fn apply_change_set(
    tx: &Transaction<'_>,
    params: &ApplyThreadHistoryParams,
) -> ThreadStoreResult<()> {
    if let Some(target) = params.changes.rollback_to_sequence {
        let target = to_i64(target, "rollback sequence")?;
        tx.execute(
            "DELETE FROM canonical_turns WHERE thread_id = ?1 AND last_sequence > ?2",
            params![params.thread_id.as_str(), target],
        )
        .map_err(store_error)?;
        tx.execute(
            "DELETE FROM canonical_items WHERE thread_id = ?1 AND sequence > ?2",
            params![params.thread_id.as_str(), target],
        )
        .map_err(store_error)?;
    }
    for turn_id in &params.changes.removed_turn_ids {
        tx.execute(
            "DELETE FROM canonical_turns WHERE thread_id = ?1 AND turn_id = ?2",
            params![params.thread_id.as_str(), turn_id.as_str()],
        )
        .map_err(store_error)?;
    }
    for item_id in &params.changes.removed_item_ids {
        tx.execute(
            "DELETE FROM canonical_items WHERE thread_id = ?1 AND item_id = ?2",
            params![params.thread_id.as_str(), item_id.as_str()],
        )
        .map_err(store_error)?;
    }
    for turn in &params.changes.changed_turns {
        upsert_turn(tx, turn, params.changes.sequence)?;
    }
    for item in &params.changes.changed_items {
        upsert_item_snapshot(tx, item)?;
    }
    Ok(())
}

fn upsert_turn(tx: &Transaction<'_>, turn: &Turn, sequence: u64) -> ThreadStoreResult<()> {
    let existing = tx
        .query_row(
            "SELECT ordinal, turn_json FROM canonical_turns WHERE thread_id = ?1 AND turn_id = ?2",
            params![turn.thread_id.as_str(), turn.turn_id.as_str()],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(store_error)?;
    let (ordinal, snapshot) = if let Some((ordinal, json)) = existing {
        let previous = decode_json(&json)?;
        (
            ordinal,
            super::super::thread_item_projection::merge_turn_snapshot(previous, turn.clone()),
        )
    } else {
        let ordinal = tx
            .query_row(
                "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM canonical_turns WHERE thread_id = ?1",
                params![turn.thread_id.as_str()],
                |row| row.get::<_, i64>(0),
            )
            .map_err(store_error)?;
        (ordinal, turn.clone())
    };
    tx.execute(
        "INSERT INTO canonical_turns (thread_id, turn_id, ordinal, last_sequence, turn_json)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(thread_id, turn_id) DO UPDATE SET
             last_sequence = excluded.last_sequence,
             turn_json = excluded.turn_json",
        params![
            turn.thread_id.as_str(),
            turn.turn_id.as_str(),
            ordinal,
            to_i64(sequence, "turn sequence")?,
            encode_json(&turn_without_items(snapshot))?,
        ],
    )
    .map_err(store_error)?;
    Ok(())
}

fn upsert_item_snapshot(tx: &Transaction<'_>, item: &ThreadItem) -> ThreadStoreResult<()> {
    if item.kind != item.payload_kind() {
        return Err(error(format!(
            "item {} kind does not match payload",
            item.item_id
        )));
    }
    let existing = tx
        .query_row(
            "SELECT turn_id FROM canonical_items WHERE thread_id = ?1 AND item_id = ?2",
            params![item.thread_id.as_str(), item.item_id.as_str()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(store_error)?;
    if let Some(existing_turn_id) = existing {
        if existing_turn_id != item.turn_id.as_str() {
            return Err(error(format!(
                "item {} changed turn identity",
                item.item_id
            )));
        }
    }
    tx.execute(
        "INSERT INTO canonical_items (
            thread_id, turn_id, item_id, ordinal, sequence, item_json
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(thread_id, item_id) DO UPDATE SET
             ordinal = excluded.ordinal,
             sequence = excluded.sequence,
             item_json = excluded.item_json",
        params![
            item.thread_id.as_str(),
            item.turn_id.as_str(),
            item.item_id.as_str(),
            to_i64(item.ordinal, "item ordinal")?,
            to_i64(item.sequence, "item sequence")?,
            encode_json(item)?,
        ],
    )
    .map_err(|source| error(format!("cannot persist canonical item: {source}")))?;
    Ok(())
}

pub(super) fn refresh_thread_snapshot(
    tx: &Transaction<'_>,
    thread_id: &ThreadId,
    sequence: u64,
) -> ThreadStoreResult<()> {
    let Some((mut thread, _)) = read_thread_row(tx, thread_id)? else {
        return Err(error(format!("thread {thread_id} does not exist")));
    };
    let latest_turn = tx
        .query_row(
            "SELECT turn_json FROM canonical_turns
             WHERE thread_id = ?1 ORDER BY ordinal DESC LIMIT 1",
            params![thread_id.as_str()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(store_error)?
        .map(|value| decode_json::<Turn>(&value))
        .transpose()?;
    if let Some(turn) = latest_turn {
        thread.status = match turn.status {
            TurnStatus::InProgress => ThreadStatus::Active {
                active_flags: (turn.approval == agent_protocol::TurnApprovalState::Pending)
                    .then_some(ThreadActiveFlag::WaitingOnApproval)
                    .into_iter()
                    .collect(),
            },
            TurnStatus::Completed | TurnStatus::Interrupted | TurnStatus::Failed => {
                ThreadStatus::Idle
            }
        };
    } else {
        thread.status = ThreadStatus::Idle;
    }
    persist_thread_snapshot(tx, &thread)?;
    tx.execute(
        "UPDATE canonical_threads SET last_sequence = ?2 WHERE thread_id = ?1",
        params![thread_id.as_str(), to_i64(sequence, "history sequence")?],
    )
    .map_err(store_error)?;
    Ok(())
}
