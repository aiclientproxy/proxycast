// Adapted from Codex thread-store/store.rs
// (5c19155cbd93bfa099016e7487259f61669823ff), Apache-2.0; see repository NOTICE.

use std::any::Any;
use std::future::Future;
use std::pin::Pin;

use agent_protocol::{Thread, ThreadId};

use crate::{
    AppendThreadItemsParams, ApplyThreadHistoryParams, ApplyThreadHistoryResult,
    ArchiveThreadParams, CreateThreadParams, DeleteThreadParams, ItemPage, ListItemsParams,
    ListThreadsParams, ListTurnsParams, ReadThreadParams, SearchThreadOccurrencesParams,
    SearchThreadsParams, ThreadOccurrenceSearchPage, ThreadPage, ThreadSearchPage,
    ThreadStoreResult, TurnPage, UpdateThreadMetadataParams,
};

/// Future returned by [`ThreadStore`] operations.
pub type ThreadStoreFuture<'a, T> = Pin<Box<dyn Future<Output = ThreadStoreResult<T>> + Send + 'a>>;

/// Storage-neutral persistence boundary for canonical Thread/Turn/Item history.
///
/// Implementations own storage details such as SQLite transactions and cursor encoding. Callers
/// only exchange canonical protocol values and opaque cursors; filesystem paths and provider wire
/// payloads do not cross this boundary.
pub trait ThreadStore: Any + Send + Sync {
    /// Returns this store as [`Any`] for implementation-owned diagnostics.
    fn as_any(&self) -> &dyn Any;

    /// Creates the durable metadata row for a canonical thread.
    fn create_thread(&self, params: CreateThreadParams) -> ThreadStoreFuture<'_, ()>;

    /// Reads a canonical thread and the requested amount of turn detail.
    fn read_thread(&self, params: ReadThreadParams) -> ThreadStoreFuture<'_, Option<Thread>>;

    /// Lists canonical threads using a store-owned opaque cursor.
    fn list_threads(&self, params: ListThreadsParams) -> ThreadStoreFuture<'_, ThreadPage>;

    /// Searches visible user and assistant conversation text across canonical threads.
    fn search_threads(
        &self,
        params: SearchThreadsParams,
    ) -> ThreadStoreFuture<'_, ThreadSearchPage>;

    /// Appends already-canonical items without deriving or changing thread metadata.
    ///
    /// The append is idempotent by `(thread_id, sequence, item contents)` and
    /// uses the same foreign-key and sequence collision rules as [`Self::apply_history`].
    fn append_items(
        &self,
        params: AppendThreadItemsParams,
    ) -> ThreadStoreFuture<'_, ApplyThreadHistoryResult>;

    /// Applies one typed materializer change set atomically.
    ///
    /// The store must reject identity/sequence collisions. An exact retry is idempotent and
    /// returns `applied = false`.
    fn apply_history(
        &self,
        params: ApplyThreadHistoryParams,
    ) -> ThreadStoreFuture<'_, ApplyThreadHistoryResult>;

    /// Lists canonical turns using stable ordinals encoded by the store cursor.
    fn list_turns(&self, params: ListTurnsParams) -> ThreadStoreFuture<'_, TurnPage>;

    /// Lists canonical items using stable ordinals encoded by the store cursor.
    fn list_items(&self, params: ListItemsParams) -> ThreadStoreFuture<'_, ItemPage>;

    /// Searches visible user and final assistant message occurrences in chronological order.
    fn search_thread_occurrences(
        &self,
        params: SearchThreadOccurrencesParams,
    ) -> ThreadStoreFuture<'_, ThreadOccurrenceSearchPage>;

    /// Applies a literal metadata patch and returns the updated canonical thread.
    fn update_thread_metadata(
        &self,
        params: UpdateThreadMetadataParams,
    ) -> ThreadStoreFuture<'_, Thread>;

    /// Archives a thread without changing its runtime status.
    fn archive_thread(&self, params: ArchiveThreadParams) -> ThreadStoreFuture<'_, ()>;

    /// Unarchives a thread and returns its current canonical snapshot.
    fn unarchive_thread(&self, params: ArchiveThreadParams) -> ThreadStoreFuture<'_, Thread>;

    /// Deletes all persisted state owned by a thread.
    fn delete_thread(&self, params: DeleteThreadParams) -> ThreadStoreFuture<'_, ()>;

    /// Returns the latest materialized sequence for repair/resume comparisons.
    fn history_sequence(&self, thread_id: ThreadId) -> ThreadStoreFuture<'_, Option<u64>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_object_safe(_: &dyn ThreadStore) {}

    #[allow(dead_code)]
    fn assert_store_contract<T: ThreadStore>(store: &T) {
        assert_object_safe(store);
    }
}
