use super::*;

pub(super) fn hydrate_thread_section(
    conn: &Connection,
    thread: &mut Thread,
) -> ThreadStoreResult<()> {
    if let Some(metadata) = thread.metadata.as_object_mut() {
        metadata.remove("section");
        metadata.remove("sectionEnteredAt");
    }
    let section = conn
        .query_row(
            "SELECT s.section_id, s.name, m.section_entered_at_ms
             FROM thread_section_members AS m
             JOIN thread_sections AS s ON s.section_id = m.section_id
             WHERE m.thread_id = ?1",
            params![thread.thread_id.as_str()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .optional()
        .map_err(store_error)?;
    if let Some((id, name, entered_at_ms)) = section {
        let metadata = metadata_object(&mut thread.metadata);
        metadata.insert(
            "section".to_string(),
            serde_json::json!({ "id": id, "name": name }),
        );
        metadata.insert(
            "sectionEnteredAt".to_string(),
            serde_json::Value::Number(entered_at_ms.into()),
        );
    }
    Ok(())
}

pub(super) fn list_thread_sections(
    store: &ProjectionStore,
    params: ListThreadSectionsParams,
) -> ThreadStoreResult<ThreadSectionPage> {
    let conn = store.open_thread_store()?;
    let limit = page_limit(params.limit)?;
    let cursor = decode_cursor(params.cursor.as_ref(), CursorKind::Sections)?;
    let cursor_clause = cursor.as_ref().map_or("", |_| {
        "WHERE (ordinal > ?1 OR (ordinal = ?1 AND section_id > ?2))"
    });
    let sql = format!(
        "SELECT section_id, name, ordinal FROM thread_sections
         {cursor_clause} ORDER BY ordinal ASC, section_id ASC LIMIT ?3"
    );
    let fallback = CursorValue {
        kind: CursorKind::Sections,
        position: i64::MIN,
        id: String::new(),
        inclusive: false,
    };
    let cursor = cursor.as_ref().unwrap_or(&fallback);
    let mut statement = conn.prepare(&sql).map_err(store_error)?;
    let mut rows = statement
        .query_map(
            params![cursor.position, cursor.id, i64::from(limit + 1)],
            |row| {
                Ok((
                    StoredThreadSection {
                        id: row.get(0)?,
                        name: row.get(1)?,
                    },
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .map_err(store_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(store_error)?;
    let has_more = rows.len() > limit as usize;
    rows.truncate(limit as usize);
    let next_cursor = has_more
        .then(|| rows.last())
        .flatten()
        .map(|(section, ordinal)| encode_cursor(CursorKind::Sections, *ordinal, &section.id))
        .transpose()?;
    Ok(ThreadSectionPage {
        data: rows.into_iter().map(|(section, _)| section).collect(),
        next_cursor,
    })
}

pub(super) fn create_thread_section(
    store: &ProjectionStore,
    params: CreateThreadSectionParams,
) -> ThreadStoreResult<StoredThreadSection> {
    let name = non_empty(&params.name, "section name")?;
    let mut conn = store.open_thread_store()?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(store_error)?;
    let ordinal = tx
        .query_row(
            "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM thread_sections",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(store_error)?;
    let section = StoredThreadSection {
        id: uuid::Uuid::now_v7().to_string(),
        name: name.to_string(),
    };
    tx.execute(
        "INSERT INTO thread_sections (section_id, name, ordinal) VALUES (?1, ?2, ?3)",
        params![section.id, section.name, ordinal],
    )
    .map_err(store_error)?;
    tx.commit().map_err(store_error)?;
    Ok(section)
}

pub(super) fn rename_thread_section(
    store: &ProjectionStore,
    params: RenameThreadSectionParams,
) -> ThreadStoreResult<Option<StoredThreadSection>> {
    let section_id = non_empty(&params.section_id, "section id")?;
    if section_id == PINNED_THREAD_SECTION_ID {
        return Err(ThreadStoreError::invalid_request(
            "the built-in pinned section cannot be renamed",
        ));
    }
    let name = non_empty(&params.name, "section name")?;
    let conn = store.open_thread_store()?;
    let changed = conn
        .execute(
            "UPDATE thread_sections SET name = ?2 WHERE section_id = ?1",
            params![section_id, name],
        )
        .map_err(store_error)?;
    Ok((changed > 0).then(|| StoredThreadSection {
        id: section_id.to_string(),
        name: name.to_string(),
    }))
}

pub(super) fn delete_thread_section(
    store: &ProjectionStore,
    params: DeleteThreadSectionParams,
) -> ThreadStoreResult<bool> {
    let section_id = non_empty(&params.section_id, "section id")?;
    if section_id == PINNED_THREAD_SECTION_ID {
        return Err(ThreadStoreError::invalid_request(
            "the built-in pinned section cannot be deleted",
        ));
    }
    let conn = store.open_thread_store()?;
    conn.execute(
        "DELETE FROM thread_sections WHERE section_id = ?1",
        params![section_id],
    )
    .map(|changed| changed > 0)
    .map_err(store_error)
}

pub(super) fn move_thread_to_section(
    store: &ProjectionStore,
    params: MoveThreadToSectionParams,
) -> ThreadStoreResult<()> {
    if params
        .section
        .as_deref()
        .is_some_and(|section| section.trim().is_empty())
    {
        return Err(ThreadStoreError::invalid_request(
            "sectionId must not be empty",
        ));
    }
    if params.section.is_none() && params.before_thread_id.is_some() {
        return Err(ThreadStoreError::invalid_request(
            "beforeThreadId requires a non-null sectionId",
        ));
    }
    if params.before_thread_id.as_ref() == Some(&params.thread_id) {
        return Ok(());
    }

    let mut conn = store.open_thread_store()?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(store_error)?;
    let thread_exists = tx
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM canonical_threads WHERE thread_id = ?1)",
            params![params.thread_id.as_str()],
            |row| row.get::<_, bool>(0),
        )
        .map_err(store_error)?;
    if !thread_exists {
        return Err(ThreadStoreError::thread_not_found(format!(
            "thread not found: {}",
            params.thread_id
        )));
    }

    if let Some(section_id) = params.section.as_deref() {
        let section_exists = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM thread_sections WHERE section_id = ?1)",
                params![section_id],
                |row| row.get::<_, bool>(0),
            )
            .map_err(store_error)?;
        if !section_exists {
            return Err(ThreadStoreError::invalid_request(format!(
                "thread section not found: {section_id}"
            )));
        }
        if let Some(before_thread_id) = params.before_thread_id.as_ref() {
            let before_section = tx
                .query_row(
                    "SELECT section_id FROM thread_section_members WHERE thread_id = ?1",
                    params![before_thread_id.as_str()],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .map_err(store_error)?;
            if before_section.as_deref() != Some(section_id) {
                return Err(ThreadStoreError::invalid_request(format!(
                    "before thread {before_thread_id} is not in section {section_id}"
                )));
            }
        }
    }

    let current = tx
        .query_row(
            "SELECT section_id, section_entered_at_ms FROM thread_section_members
             WHERE thread_id = ?1",
            params![params.thread_id.as_str()],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()
        .map_err(store_error)?;
    tx.execute(
        "DELETE FROM thread_section_members WHERE thread_id = ?1",
        params![params.thread_id.as_str()],
    )
    .map_err(store_error)?;
    if let Some((current_section, _)) = current.as_ref() {
        normalize_section_positions(&tx, current_section)?;
    }

    if let Some(section_id) = params.section.as_deref() {
        let mut thread_ids = section_thread_ids(&tx, section_id)?;
        let index = params
            .before_thread_id
            .as_ref()
            .and_then(|before| thread_ids.iter().position(|thread| thread == before))
            .unwrap_or(thread_ids.len());
        thread_ids.insert(index, params.thread_id.clone());
        let entered_at_ms = current
            .as_ref()
            .filter(|(current_section, _)| current_section == section_id)
            .map(|(_, entered_at_ms)| *entered_at_ms)
            .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
        tx.execute(
            "INSERT INTO thread_section_members
             (thread_id, section_id, position, section_entered_at_ms)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                params.thread_id.as_str(),
                section_id,
                index as i64,
                entered_at_ms
            ],
        )
        .map_err(store_error)?;
        for (position, thread_id) in thread_ids.iter().enumerate() {
            tx.execute(
                "UPDATE thread_section_members SET position = ?2 WHERE thread_id = ?1",
                params![thread_id.as_str(), position as i64],
            )
            .map_err(store_error)?;
        }
    }
    tx.commit().map_err(store_error)
}

pub(super) fn query_section_thread_page(
    conn: &Connection,
    include_archived: bool,
    section_id: &str,
    sort_by_section_position: bool,
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
    let position = if sort_by_section_position {
        "m.position"
    } else {
        "COALESCE(t.recency_at_ms, t.updated_at_ms)"
    };
    let cursor_clause = cursor.map_or(String::new(), |_| {
        format!(
            "AND (({position} {comparator} ?3) OR ({position} = ?3 AND t.thread_id {comparator} ?4))"
        )
    });
    let sql = format!(
        "SELECT t.thread_json, {position}, t.thread_id
         FROM canonical_threads AS t
         JOIN thread_section_members AS m ON m.thread_id = t.thread_id
         WHERE (?1 = 1 OR t.archived = 0) AND m.section_id = ?2
           AND NOT EXISTS (
                SELECT 1 FROM canonical_thread_spawn_edges AS edge
                WHERE edge.child_thread_id = t.thread_id
                  AND edge.status = 'pending'
           )
         {cursor_clause}
         ORDER BY {position} {order}, t.thread_id {order} LIMIT ?5"
    );
    query_thread_rows(
        conn,
        &sql,
        include_archived,
        Some(section_id),
        direction,
        cursor,
        limit,
    )
}

pub(super) fn query_unsectioned_thread_page(
    conn: &Connection,
    include_archived: bool,
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
        format!("AND ((COALESCE(t.recency_at_ms, t.updated_at_ms) {comparator} ?3) OR (COALESCE(t.recency_at_ms, t.updated_at_ms) = ?3 AND t.thread_id {comparator} ?4))")
    });
    let sql = format!(
        "SELECT t.thread_json, COALESCE(t.recency_at_ms, t.updated_at_ms), t.thread_id
         FROM canonical_threads AS t
         WHERE (?1 = 1 OR t.archived = 0)
           AND NOT EXISTS (
               SELECT 1 FROM thread_section_members AS m WHERE m.thread_id = t.thread_id
           )
           AND NOT EXISTS (
                SELECT 1 FROM canonical_thread_spawn_edges AS edge
                WHERE edge.child_thread_id = t.thread_id
                  AND edge.status = 'pending'
           )
         {cursor_clause}
         ORDER BY COALESCE(t.recency_at_ms, t.updated_at_ms) {order}, t.thread_id {order}
         LIMIT ?5"
    );
    query_thread_rows(conn, &sql, include_archived, None, direction, cursor, limit)
}

fn query_thread_rows(
    conn: &Connection,
    sql: &str,
    include_archived: bool,
    section_id: Option<&str>,
    direction: SortDirection,
    cursor: Option<&CursorValue>,
    limit: u32,
) -> ThreadStoreResult<Vec<(Thread, i64, String)>> {
    let fallback = CursorValue {
        kind: CursorKind::Threads,
        position: if direction == SortDirection::Asc {
            i64::MIN
        } else {
            i64::MAX
        },
        id: String::new(),
        inclusive: false,
    };
    let cursor = cursor.unwrap_or(&fallback);
    let mut statement = conn.prepare(sql).map_err(store_error)?;
    let rows = statement
        .query_map(
            params![
                i64::from(include_archived),
                section_id,
                cursor.position,
                cursor.id,
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

fn section_thread_ids(tx: &Transaction<'_>, section_id: &str) -> ThreadStoreResult<Vec<ThreadId>> {
    let mut statement = tx
        .prepare(
            "SELECT thread_id FROM thread_section_members
             WHERE section_id = ?1 ORDER BY position ASC, thread_id ASC",
        )
        .map_err(store_error)?;
    let thread_ids = statement
        .query_map(params![section_id], |row| row.get::<_, String>(0))
        .map_err(store_error)?
        .map(|row| row.map(ThreadId::new).map_err(store_error))
        .collect();
    thread_ids
}

fn normalize_section_positions(tx: &Transaction<'_>, section_id: &str) -> ThreadStoreResult<()> {
    for (position, thread_id) in section_thread_ids(tx, section_id)?.iter().enumerate() {
        tx.execute(
            "UPDATE thread_section_members SET position = ?2 WHERE thread_id = ?1",
            params![thread_id.as_str(), position as i64],
        )
        .map_err(store_error)?;
    }
    Ok(())
}

fn non_empty<'a>(value: &'a str, field: &str) -> ThreadStoreResult<&'a str> {
    let value = value.trim();
    if value.is_empty() {
        Err(ThreadStoreError::invalid_request(format!(
            "{field} must not be empty"
        )))
    } else {
        Ok(value)
    }
}

fn metadata_object(
    metadata: &mut serde_json::Value,
) -> &mut serde_json::Map<String, serde_json::Value> {
    if !metadata.is_object() {
        *metadata = serde_json::Value::Object(serde_json::Map::new());
    }
    metadata
        .as_object_mut()
        .expect("metadata was normalized to an object")
}
