use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::runtime) struct ThreadRevertBoundary {
    pub(in crate::runtime) rollback_to_sequence: u64,
    pub(in crate::runtime) turns_backwards_cursor: Option<StoreCursor>,
    pub(in crate::runtime) items_backwards_cursor: Option<StoreCursor>,
}

impl ProjectionStore {
    pub(in crate::runtime) fn thread_revert_boundary_sync(
        &self,
        thread_id: &ThreadId,
        before_turn_id: &TurnId,
    ) -> ThreadStoreResult<ThreadRevertBoundary> {
        let conn = self.open_thread_store()?;
        ensure_thread_visible(&conn, thread_id, false)?;
        let target_ordinal = conn
            .query_row(
                "SELECT ordinal FROM canonical_turns
                 WHERE thread_id = ?1 AND turn_id = ?2",
                params![thread_id.as_str(), before_turn_id.as_str()],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .map_err(store_error)?
            .ok_or_else(|| error(format!("turn not found: {before_turn_id}")))?;
        let rollback_to_sequence = conn
            .query_row(
                "SELECT COALESCE(MAX(last_sequence), 0) FROM canonical_turns
                 WHERE thread_id = ?1 AND ordinal < ?2",
                params![thread_id.as_str(), target_ordinal],
                |row| row.get::<_, i64>(0),
            )
            .map_err(store_error)?
            .max(0) as u64;
        let retained_turn = conn
            .query_row(
                "SELECT ordinal, turn_id FROM canonical_turns
                 WHERE thread_id = ?1 AND ordinal < ?2
                 ORDER BY ordinal DESC, turn_id DESC LIMIT 1",
                params![thread_id.as_str(), target_ordinal],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()
            .map_err(store_error)?;
        let retained_item = conn
            .query_row(
                "SELECT item.ordinal, item.item_id
                 FROM canonical_items AS item
                 INNER JOIN canonical_turns AS turn
                   ON turn.thread_id = item.thread_id AND turn.turn_id = item.turn_id
                 WHERE item.thread_id = ?1 AND turn.ordinal < ?2
                 ORDER BY item.ordinal DESC, item.item_id DESC LIMIT 1",
                params![thread_id.as_str(), target_ordinal],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()
            .map_err(store_error)?;

        Ok(ThreadRevertBoundary {
            rollback_to_sequence,
            turns_backwards_cursor: retained_turn
                .map(|(ordinal, id)| {
                    encode_cursor_with_inclusive(CursorKind::Turns, ordinal, &id, true)
                })
                .transpose()?,
            items_backwards_cursor: retained_item
                .map(|(ordinal, id)| {
                    encode_cursor_with_inclusive(CursorKind::Items, ordinal, &id, true)
                })
                .transpose()?,
        })
    }
}
