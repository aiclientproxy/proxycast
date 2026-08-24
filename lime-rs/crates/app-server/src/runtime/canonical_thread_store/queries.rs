use super::*;

pub(super) fn hydrate_thread(
    conn: &Connection,
    thread: &mut Thread,
    view: ThreadTurnsView,
) -> ThreadStoreResult<()> {
    thread.turns_view = view;
    thread.turns = match view {
        ThreadTurnsView::NotLoaded => Vec::new(),
        ThreadTurnsView::Summary => {
            let mut turns = query_all_turns(conn, &thread.thread_id)?;
            for turn in &mut turns {
                turn.items.clear();
                turn.items_view = TurnItemsView::NotLoaded;
            }
            turns
        }
        ThreadTurnsView::Full => {
            let mut turns = query_all_turns(conn, &thread.thread_id)?;
            for turn in &mut turns {
                hydrate_turn(conn, turn, TurnItemsView::Full)?;
            }
            turns
        }
    };
    Ok(())
}

pub(super) fn hydrate_turn(
    conn: &Connection,
    turn: &mut Turn,
    view: TurnItemsView,
) -> ThreadStoreResult<()> {
    turn.items_view = view;
    turn.items = match view {
        TurnItemsView::NotLoaded | TurnItemsView::Summary => Vec::new(),
        TurnItemsView::Full => query_all_items(conn, &turn.thread_id, Some(&turn.turn_id))?,
    };
    Ok(())
}

fn query_all_turns(conn: &Connection, thread_id: &ThreadId) -> ThreadStoreResult<Vec<Turn>> {
    let mut stmt = conn
        .prepare(
            "SELECT turn_json FROM canonical_turns
             WHERE thread_id = ?1 ORDER BY ordinal ASC, turn_id ASC",
        )
        .map_err(store_error)?;
    let rows = stmt
        .query_map(params![thread_id.as_str()], |row| row.get::<_, String>(0))
        .map_err(store_error)?
        .map(|row| decode_json(&row.map_err(store_error)?))
        .collect();
    rows
}

fn query_all_items(
    conn: &Connection,
    thread_id: &ThreadId,
    turn_id: Option<&TurnId>,
) -> ThreadStoreResult<Vec<ThreadItem>> {
    let (sql, turn) = if let Some(turn_id) = turn_id {
        (
            "SELECT item_json FROM canonical_items
             WHERE thread_id = ?1 AND turn_id = ?2 ORDER BY ordinal ASC, item_id ASC",
            Some(turn_id.as_str()),
        )
    } else {
        (
            "SELECT item_json FROM canonical_items
             WHERE thread_id = ?1 ORDER BY ordinal ASC, item_id ASC",
            None,
        )
    };
    let mut stmt = conn.prepare(sql).map_err(store_error)?;
    let values = if let Some(turn) = turn {
        stmt.query_map(params![thread_id.as_str(), turn], |row| {
            row.get::<_, String>(0)
        })
        .map_err(store_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(store_error)?
    } else {
        stmt.query_map(params![thread_id.as_str()], |row| row.get::<_, String>(0))
            .map_err(store_error)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(store_error)?
    };
    values
        .into_iter()
        .map(|value| decode_json(&value))
        .collect()
}

pub(super) fn query_thread_page(
    conn: &Connection,
    include_archived: bool,
    project: Option<&Option<String>>,
    direction: SortDirection,
    cursor: Option<&CursorValue>,
    limit: u32,
) -> ThreadStoreResult<Vec<(Thread, i64, String)>> {
    let comparator = if direction == SortDirection::Asc {
        ">"
    } else {
        "<"
    };
    let order = if direction == SortDirection::Asc {
        "ASC"
    } else {
        "DESC"
    };
    let cursor_clause = cursor.map_or(String::new(), |_| {
        format!("AND ((COALESCE(recency_at_ms, updated_at_ms) {comparator} ?2) OR (COALESCE(recency_at_ms, updated_at_ms) = ?2 AND thread_id {comparator} ?3))")
    });
    let sql = format!(
        "SELECT thread_json, COALESCE(recency_at_ms, updated_at_ms), thread_id
         FROM canonical_threads
         WHERE (?1 = 1 OR archived = 0)
           AND NOT EXISTS (
                SELECT 1 FROM canonical_thread_spawn_edges AS edge
                WHERE edge.child_thread_id = canonical_threads.thread_id
                  AND edge.status = 'pending'
           )
           AND (?4 = 0
                OR (?4 = 1 AND json_extract(thread_json, '$.metadata.projectId') IS NULL)
                OR (?4 = 2 AND json_extract(thread_json, '$.metadata.projectId') = ?5))
           {cursor_clause}
         ORDER BY COALESCE(recency_at_ms, updated_at_ms) {order}, thread_id {order} LIMIT ?6"
    );
    let mut stmt = conn.prepare(&sql).map_err(store_error)?;
    let fallback = cursor_fallback(CursorKind::Threads, direction);
    let cursor = cursor.unwrap_or(&fallback);
    let rows = stmt
        .query_map(
            params![
                i64::from(include_archived),
                cursor.position,
                cursor.id,
                project_filter_mode(project),
                project.and_then(Option::as_deref),
                i64::from(limit)
            ],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .map_err(store_error)?
        .map(|row| {
            let (json, position, id) = row.map_err(store_error)?;
            Ok((decode_json(&json)?, position, id))
        })
        .collect();
    rows
}

pub(super) fn project_filter_mode(project: Option<&Option<String>>) -> i64 {
    match project {
        None => 0,
        Some(None) => 1,
        Some(Some(_)) => 2,
    }
}

pub(super) fn query_turn_page(
    conn: &Connection,
    thread_id: &ThreadId,
    direction: SortDirection,
    cursor: Option<&CursorValue>,
    limit: u32,
) -> ThreadStoreResult<Vec<(Turn, i64, String)>> {
    query_entity_page(
        conn,
        "canonical_turns",
        "turn_json",
        "turn_id",
        thread_id,
        None,
        direction,
        cursor,
        limit,
    )
    .and_then(decode_page)
}

pub(super) fn query_item_page(
    conn: &Connection,
    thread_id: &ThreadId,
    turn_id: Option<&TurnId>,
    direction: SortDirection,
    cursor: Option<&CursorValue>,
    limit: u32,
) -> ThreadStoreResult<Vec<(ThreadItem, i64, String)>> {
    query_entity_page(
        conn,
        "canonical_items",
        "item_json",
        "item_id",
        thread_id,
        turn_id.map(TurnId::as_str),
        direction,
        cursor,
        limit,
    )
    .and_then(decode_page)
}

#[allow(clippy::too_many_arguments)]
fn query_entity_page(
    conn: &Connection,
    table: &str,
    json_column: &str,
    id_column: &str,
    thread_id: &ThreadId,
    turn_id: Option<&str>,
    direction: SortDirection,
    cursor: Option<&CursorValue>,
    limit: u32,
) -> ThreadStoreResult<Vec<(String, i64, String)>> {
    let comparator = if direction == SortDirection::Asc {
        ">"
    } else {
        "<"
    };
    let order = if direction == SortDirection::Asc {
        "ASC"
    } else {
        "DESC"
    };
    let turn_clause = turn_id.map_or("", |_| "AND turn_id = ?2");
    let cursor_clause = cursor.map_or(String::new(), |cursor| {
        let anchor_comparator = if cursor.inclusive {
            if direction == SortDirection::Asc {
                ">="
            } else {
                "<="
            }
        } else {
            comparator
        };
        format!("AND ((ordinal {comparator} ?3) OR (ordinal = ?3 AND {id_column} {anchor_comparator} ?4))")
    });
    let sql = format!(
        "SELECT {json_column}, ordinal, {id_column} FROM {table}
         WHERE thread_id = ?1 {turn_clause} {cursor_clause}
         ORDER BY ordinal {order}, {id_column} {order} LIMIT ?5"
    );
    let fallback = cursor_fallback(CursorKind::Turns, direction);
    let cursor = cursor.unwrap_or(&fallback);
    let mut stmt = conn.prepare(&sql).map_err(store_error)?;
    let rows = stmt
        .query_map(
            params![
                thread_id.as_str(),
                turn_id,
                cursor.position,
                cursor.id,
                i64::from(limit)
            ],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(store_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(store_error);
    rows
}

fn cursor_fallback(kind: CursorKind, direction: SortDirection) -> CursorValue {
    CursorValue {
        kind,
        position: if direction == SortDirection::Asc {
            i64::MIN
        } else {
            i64::MAX
        },
        id: String::new(),
        inclusive: false,
    }
}

fn decode_page<T: serde::de::DeserializeOwned>(
    rows: Vec<(String, i64, String)>,
) -> ThreadStoreResult<Vec<(T, i64, String)>> {
    rows.into_iter()
        .map(|(json, position, id)| Ok((decode_json(&json)?, position, id)))
        .collect()
}

pub(super) fn read_thread_row(
    conn: &Connection,
    thread_id: &ThreadId,
) -> ThreadStoreResult<Option<(Thread, bool)>> {
    conn.query_row(
        "SELECT thread_json, archived FROM canonical_threads WHERE thread_id = ?1",
        params![thread_id.as_str()],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? != 0)),
    )
    .optional()
    .map_err(store_error)?
    .map(|(json, archived)| {
        let mut thread: Thread = decode_json(&json)?;
        thread.archived = archived;
        Ok((thread, archived))
    })
    .transpose()
}

pub(super) fn is_pending_spawn_thread(
    conn: &Connection,
    thread_id: &ThreadId,
) -> ThreadStoreResult<bool> {
    conn.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM canonical_thread_spawn_edges
            WHERE child_thread_id = ?1 AND status = 'pending'
         )",
        params![thread_id.as_str()],
        |row| row.get(0),
    )
    .map_err(store_error)
}

pub(super) fn ensure_thread_visible(
    conn: &Connection,
    thread_id: &ThreadId,
    include_archived: bool,
) -> ThreadStoreResult<()> {
    if is_pending_spawn_thread(conn, thread_id)? {
        return Err(error(format!("thread {thread_id} does not exist")));
    }
    let Some((_, archived)) = read_thread_row(conn, thread_id)? else {
        return Err(error(format!("thread {thread_id} does not exist")));
    };
    if archived && !include_archived {
        return Err(error(format!("thread {thread_id} is archived")));
    }
    Ok(())
}

pub(super) fn persist_thread_snapshot(conn: &Connection, thread: &Thread) -> ThreadStoreResult<()> {
    conn.execute(
        "UPDATE canonical_threads SET
            thread_json = ?2, updated_at_ms = ?3, recency_at_ms = ?4
         WHERE thread_id = ?1",
        params![
            thread.thread_id.as_str(),
            encode_json(&thread_without_turns(thread.clone()))?,
            thread.updated_at_ms,
            thread.recency_at_ms,
        ],
    )
    .map_err(store_error)?;
    Ok(())
}
