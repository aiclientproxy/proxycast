//! Canonical Thread/Turn/Item history reduction.
//!
//! The builder operates on values that are already in the `agent-protocol`
//! canonical shape.  It deliberately does not decode provider payloads or
//! runtime events.  Event-to-item lowering belongs to the runtime projection;
//! this module only gives that projection a deterministic coalescing and
//! rollback boundary before the raw history is handed to [`ThreadStore`].

use agent_protocol::{
    ItemId, SessionId, SortDirection, Thread, ThreadHistoryChangeSet, ThreadId, ThreadItem, Turn,
    TurnId,
};
use std::collections::{HashMap, HashSet};
use std::fmt;

#[path = "history_merge.rs"]
mod merge;
use merge::{merge_item_snapshot, merge_turn_snapshot, ChangeAccumulator};

#[path = "history_validation.rs"]
mod validation;
use validation::{validate_identity, validate_item_content};

/// A page of the builder's raw canonical history.
#[derive(Clone, Debug, PartialEq)]
pub struct ThreadHistoryPage<T> {
    pub data: Vec<T>,
    pub next_offset: Option<usize>,
    pub previous_offset: Option<usize>,
}

/// A storage-neutral snapshot of raw canonical history.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CanonicalHistory {
    pub session_id: Option<SessionId>,
    pub thread_id: Option<ThreadId>,
    pub sequence: Option<u64>,
    pub turns: Vec<Turn>,
    pub turn_sequences: HashMap<TurnId, u64>,
    pub items: Vec<ThreadItem>,
}

/// Errors raised before a history batch is applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ThreadHistoryBuilderError {
    SessionIdentityMismatch {
        expected: SessionId,
        actual: SessionId,
    },
    ThreadIdentityMismatch {
        expected: ThreadId,
        actual: ThreadId,
    },
    SequenceRegression {
        current: u64,
        next: u64,
    },
    SequenceCollision {
        sequence: u64,
    },
    InvalidRollback {
        target: u64,
        applied: u64,
    },
    DuplicateTurn {
        turn_id: TurnId,
    },
    DuplicateItem {
        item_id: ItemId,
    },
    MissingTurn {
        turn_id: TurnId,
    },
    ItemTurnIdentityMismatch {
        item_id: ItemId,
        expected: TurnId,
        actual: TurnId,
    },
    UnsafeItemContent {
        item_id: ItemId,
    },
}

impl fmt::Display for ThreadHistoryBuilderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SessionIdentityMismatch { expected, actual } => write!(
                formatter,
                "history session identity mismatch: expected {expected}, got {actual}"
            ),
            Self::ThreadIdentityMismatch { expected, actual } => write!(
                formatter,
                "history thread identity mismatch: expected {expected}, got {actual}"
            ),
            Self::SequenceRegression { current, next } => {
                write!(
                    formatter,
                    "history sequence regressed from {current} to {next}"
                )
            }
            Self::SequenceCollision { sequence } => {
                write!(formatter, "history sequence collision at {sequence}")
            }
            Self::InvalidRollback { target, applied } => write!(
                formatter,
                "rollback target {target} must precede applied sequence {applied}"
            ),
            Self::DuplicateTurn { turn_id } => {
                write!(formatter, "duplicate canonical turn {turn_id}")
            }
            Self::DuplicateItem { item_id } => {
                write!(formatter, "duplicate canonical item {item_id}")
            }
            Self::MissingTurn { turn_id } => {
                write!(
                    formatter,
                    "canonical item references missing turn {turn_id}"
                )
            }
            Self::ItemTurnIdentityMismatch {
                item_id,
                expected,
                actual,
            } => write!(
                formatter,
                "canonical item {item_id} changed turn identity from {expected} to {actual}"
            ),
            Self::UnsafeItemContent { item_id } => {
                write!(
                    formatter,
                    "canonical item {item_id} contains unsafe content"
                )
            }
        }
    }
}

impl std::error::Error for ThreadHistoryBuilderError {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ItemKey {
    turn_id: TurnId,
    item_id: ItemId,
}

/// Reduces already-canonical history snapshots into one deterministic raw
/// Thread/Turn/Item view.
///
/// This is intentionally a small storage-neutral counterpart to Codex's
/// `ThreadHistoryBuilder`: repeated snapshots replace the latest value while
/// retaining first-seen ordering, and rollback removes only records after its
/// target sequence.  It never infers thread metadata from item content.
#[derive(Clone, Debug, Default)]
pub struct ThreadHistoryBuilder {
    session_id: Option<SessionId>,
    thread_id: Option<ThreadId>,
    turns: Vec<Turn>,
    turn_indexes: HashMap<TurnId, usize>,
    turn_sequences: HashMap<TurnId, u64>,
    items: Vec<ThreadItem>,
    item_indexes: HashMap<ItemKey, usize>,
    sequence: Option<u64>,
    active_turn_id: Option<TurnId>,
}

impl ThreadHistoryBuilder {
    /// Creates an empty builder. Identity is established by the first valid
    /// turn/item batch, or can be fixed up-front with [`Self::for_thread`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty builder with a fixed session/thread identity.
    pub fn for_thread(session_id: SessionId, thread_id: ThreadId) -> Self {
        Self {
            session_id: Some(session_id),
            thread_id: Some(thread_id),
            ..Self::default()
        }
    }

    /// Rebuilds the reducer from a canonical read snapshot. Durable callers
    /// should prefer [`Self::from_snapshot`] when exact turn sequences are
    /// available.
    pub fn from_thread(
        thread: &Thread,
        sequence: Option<u64>,
    ) -> Result<Self, ThreadHistoryBuilderError> {
        let items = thread
            .turns
            .iter()
            .flat_map(|turn| turn.items.iter().cloned())
            .collect();
        Self::from_snapshot(CanonicalHistory {
            session_id: Some(thread.session_id.clone()),
            thread_id: Some(thread.thread_id.clone()),
            sequence,
            turns: thread.turns.clone(),
            turn_sequences: HashMap::new(),
            items,
        })
    }

    /// Rebuilds all indexes from a durable canonical snapshot. Snapshot
    /// validation is atomic and rejects duplicate identity or items whose turn
    /// is absent.
    pub fn from_snapshot(mut history: CanonicalHistory) -> Result<Self, ThreadHistoryBuilderError> {
        if history.items.is_empty() {
            history.items = history
                .turns
                .iter()
                .flat_map(|turn| turn.items.iter().cloned())
                .collect();
        }
        let mut session_id = history.session_id.take();
        let mut thread_id = history.thread_id.take();
        for turn in &history.turns {
            validate_identity(
                &mut session_id,
                &mut thread_id,
                &turn.session_id,
                &turn.thread_id,
            )?;
        }
        for item in &history.items {
            validate_identity(
                &mut session_id,
                &mut thread_id,
                &item.session_id,
                &item.thread_id,
            )?;
            validate_item_content(item)?;
        }

        let mut builder = Self {
            session_id,
            thread_id,
            sequence: history.sequence,
            ..Self::default()
        };
        for mut turn in history.turns {
            if builder.turn_indexes.contains_key(&turn.turn_id) {
                return Err(ThreadHistoryBuilderError::DuplicateTurn {
                    turn_id: turn.turn_id,
                });
            }
            turn.items.clear();
            let index = builder.turns.len();
            builder.turn_indexes.insert(turn.turn_id.clone(), index);
            builder.turns.push(turn);
        }
        for item in history.items {
            builder.insert_snapshot_item(item)?;
        }

        let inferred_sequence = builder
            .items
            .iter()
            .map(|item| item.sequence)
            .chain(history.turn_sequences.values().copied())
            .max();
        if let (Some(current), Some(next)) = (inferred_sequence, builder.sequence) {
            if next < current {
                return Err(ThreadHistoryBuilderError::SequenceRegression { current, next });
            }
        }
        builder.sequence = builder.sequence.or(inferred_sequence);
        for turn in &builder.turns {
            let sequence = history
                .turn_sequences
                .get(&turn.turn_id)
                .copied()
                .or_else(|| turn.items.iter().map(|item| item.sequence).max())
                .or(builder.sequence)
                .unwrap_or_default();
            builder
                .turn_sequences
                .insert(turn.turn_id.clone(), sequence);
            if turn.status.is_terminal() {
                if builder.active_turn_id.as_ref() == Some(&turn.turn_id) {
                    builder.active_turn_id = None;
                }
            } else {
                builder.active_turn_id = Some(turn.turn_id.clone());
            }
        }
        Ok(builder)
    }

    /// Clears all raw history and identity state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Returns the latest durable sequence accepted by this builder.
    pub fn sequence(&self) -> Option<u64> {
        self.sequence
    }

    /// Returns raw canonical items in first-seen order.
    pub fn raw_items(&self) -> &[ThreadItem] {
        &self.items
    }

    /// Alias used by replay callers that consume the builder.
    pub fn into_raw_items(self) -> Vec<ThreadItem> {
        self.items
    }

    /// Returns canonical turn snapshots in first-seen order.
    pub fn turns(&self) -> &[Turn] {
        &self.turns
    }

    /// Returns a value snapshot suitable for cold-read/replay comparisons.
    pub fn snapshot(&self) -> CanonicalHistory {
        CanonicalHistory {
            session_id: self.session_id.clone(),
            thread_id: self.thread_id.clone(),
            sequence: self.sequence,
            turns: self.turns.clone(),
            turn_sequences: self.turn_sequences.clone(),
            items: self.items.clone(),
        }
    }

    /// Validates and reduces one durable change set against the current
    /// canonical snapshot. The reducer is transactional: any identity,
    /// sequence, rollback, or parent-turn error leaves `self` unchanged.
    pub fn apply_change_set(
        &mut self,
        changes: ThreadHistoryChangeSet,
    ) -> Result<ThreadHistoryChangeSet, ThreadHistoryBuilderError> {
        let checkpoint = self.clone();
        match self.apply_change_set_inner(changes) {
            Ok(changes) => Ok(changes),
            Err(error) => {
                *self = checkpoint;
                Err(error)
            }
        }
    }

    fn apply_change_set_inner(
        &mut self,
        changes: ThreadHistoryChangeSet,
    ) -> Result<ThreadHistoryChangeSet, ThreadHistoryBuilderError> {
        self.validate_turns(&changes.changed_turns)?;
        self.validate_items(&changes.changed_items)?;
        self.validate_sequence(changes.sequence)?;
        if changes
            .rollback_to_sequence
            .is_some_and(|target| target >= changes.sequence)
        {
            return Err(ThreadHistoryBuilderError::InvalidRollback {
                target: changes.rollback_to_sequence.unwrap_or_default(),
                applied: changes.sequence,
            });
        }
        if self.sequence == Some(changes.sequence) {
            if self.change_set_already_applied(&changes) {
                return Ok(ThreadHistoryChangeSet {
                    sequence: changes.sequence,
                    ..Default::default()
                });
            }
            return Err(ThreadHistoryBuilderError::SequenceCollision {
                sequence: changes.sequence,
            });
        }

        let mut normalized = ThreadHistoryChangeSet {
            sequence: changes.sequence,
            rollback_to_sequence: changes.rollback_to_sequence,
            ..Default::default()
        };
        if let Some(target) = changes.rollback_to_sequence {
            let rolled_back = self.rollback_at(changes.sequence, target)?;
            extend_unique(
                &mut normalized.removed_item_ids,
                rolled_back.removed_item_ids,
            );
            extend_unique(
                &mut normalized.removed_turn_ids,
                rolled_back.removed_turn_ids,
            );
        }
        for turn_id in changes.removed_turn_ids {
            if self.remove_turn(&turn_id) {
                push_unique(&mut normalized.removed_turn_ids, turn_id);
            }
        }
        for item_id in changes.removed_item_ids {
            if self.remove_item(&item_id) {
                push_unique(&mut normalized.removed_item_ids, item_id);
            }
        }

        let mut coalesced = ChangeAccumulator::default();
        for turn in changes.changed_turns {
            if let Some(turn) = self.upsert_turn(turn, changes.sequence)? {
                coalesced.push_turn(turn);
            }
        }
        for item in changes.changed_items {
            let key = ItemKey {
                turn_id: item.turn_id.clone(),
                item_id: item.item_id.clone(),
            };
            if let Some(item) = self.upsert_item(item)? {
                coalesced.push_item((key.turn_id, key.item_id), item);
            }
        }
        let changed = coalesced.finish(changes.sequence);
        normalized.changed_turns = changed.changed_turns;
        normalized.changed_items = changed.changed_items;
        self.sequence = Some(changes.sequence);
        Ok(normalized)
    }

    /// Returns the active turn, or the newest turn when no turn is open.
    pub fn active_turn_snapshot(&self) -> Option<Turn> {
        self.active_turn_id
            .as_ref()
            .and_then(|turn_id| self.turn_snapshot(turn_id.as_str()))
            .or_else(|| self.turns.last().cloned())
    }

    /// Returns one turn snapshot by canonical ID.
    pub fn turn_snapshot(&self, turn_id: &str) -> Option<Turn> {
        self.turn_indexes
            .get(&TurnId::new(turn_id))
            .and_then(|index| self.turns.get(*index))
            .cloned()
    }

    /// Appends canonical items at the caller-owned durable event sequence.
    ///
    /// An exact retry at the same sequence is idempotent.  A different payload
    /// or a new item at an already-applied sequence is rejected before any
    /// in-memory state changes.
    pub fn append_items_at<I>(
        &mut self,
        sequence: u64,
        items: I,
    ) -> Result<ThreadHistoryChangeSet, ThreadHistoryBuilderError>
    where
        I: AsRef<[ThreadItem]>,
    {
        let items = items.as_ref();
        self.validate_items(items)?;
        self.validate_sequence(sequence)?;
        if self.sequence == Some(sequence) {
            let retry = ThreadHistoryChangeSet {
                sequence,
                changed_items: items.to_vec(),
                ..Default::default()
            };
            if self.change_set_already_applied(&retry) {
                return Ok(ThreadHistoryChangeSet {
                    sequence,
                    ..Default::default()
                });
            }
            return Err(ThreadHistoryBuilderError::SequenceCollision { sequence });
        }
        if let Some(item) = items.first() {
            self.adopt_identity(&item.session_id, &item.thread_id);
        }

        let mut changes = ChangeAccumulator::default();
        let mut affected_turn_ids = HashSet::new();
        for item in items {
            affected_turn_ids.insert(item.turn_id.clone());
            let key = ItemKey {
                turn_id: item.turn_id.clone(),
                item_id: item.item_id.clone(),
            };
            let snapshot = if let Some(index) = self.item_indexes.get(&key).copied() {
                let previous = self.items[index].clone();
                let merged = merge_item_snapshot(previous.clone(), item.clone());
                if merged == previous {
                    continue;
                }
                self.items[index] = merged.clone();
                merged
            } else {
                let index = self.items.len();
                self.item_indexes.insert(key.clone(), index);
                self.items.push(item.clone());
                item.clone()
            };
            changes.push_item((key.turn_id, key.item_id), snapshot);
            if self
                .turns
                .iter()
                .find(|turn| turn.turn_id == item.turn_id)
                .is_some_and(|turn| !turn.status.is_terminal())
            {
                self.active_turn_id = Some(item.turn_id.clone());
            }
        }
        for turn_id in affected_turn_ids {
            self.sync_turn_items(&turn_id);
        }

        if !items.is_empty() {
            self.sequence = Some(sequence);
        }
        Ok(changes.finish(sequence))
    }

    /// Appends turn snapshots at one durable sequence.
    pub fn append_turns_at<I>(
        &mut self,
        sequence: u64,
        turns: I,
    ) -> Result<ThreadHistoryChangeSet, ThreadHistoryBuilderError>
    where
        I: AsRef<[Turn]>,
    {
        let turns = turns.as_ref();
        self.validate_turns(turns)?;
        self.validate_sequence(sequence)?;
        if self.sequence == Some(sequence) {
            let retry = ThreadHistoryChangeSet {
                sequence,
                changed_turns: turns.to_vec(),
                ..Default::default()
            };
            if self.change_set_already_applied(&retry) {
                return Ok(ThreadHistoryChangeSet {
                    sequence,
                    ..Default::default()
                });
            }
            return Err(ThreadHistoryBuilderError::SequenceCollision { sequence });
        }
        if let Some(turn) = turns.first() {
            self.adopt_identity(&turn.session_id, &turn.thread_id);
        }

        let mut changes = ChangeAccumulator::default();
        for turn in turns {
            let turn_id = turn.turn_id.clone();
            let index = if let Some(index) = self.turn_indexes.get(&turn_id).copied() {
                let previous = self.turns[index].clone();
                let merged = merge_turn_snapshot(previous.clone(), turn.clone());
                if merged == previous {
                    self.turn_sequences.insert(turn_id.clone(), sequence);
                    self.sync_turn_items(&turn_id);
                    continue;
                }
                self.turns[index] = merged.clone();
                index
            } else {
                let index = self.turns.len();
                self.turn_indexes.insert(turn_id.clone(), index);
                self.turns.push(turn.clone());
                index
            };
            self.turn_sequences.insert(turn_id.clone(), sequence);
            self.sync_turn_items(&turn_id);
            let snapshot = self.turns[index].clone();
            changes.push_turn(snapshot);
            if self.turns[index].status.is_terminal() {
                if self.active_turn_id.as_ref() == Some(&turn_id) {
                    self.active_turn_id = None;
                }
            } else {
                self.active_turn_id = Some(turn_id);
            }
        }

        if !turns.is_empty() {
            self.sequence = Some(sequence);
        }
        Ok(changes.finish(sequence))
    }

    /// Rolls the in-memory history back to `target` and emits a new durable
    /// change sequence.  The target must precede the applied rollback record.
    pub fn rollback_to_sequence(
        &mut self,
        target: u64,
    ) -> Result<ThreadHistoryChangeSet, ThreadHistoryBuilderError> {
        let applied = self.sequence.unwrap_or_default().saturating_add(1);
        self.rollback_at(applied, target)
    }

    /// Variant for callers that already own the durable rollback event
    /// sequence.
    pub fn rollback_at(
        &mut self,
        applied: u64,
        target: u64,
    ) -> Result<ThreadHistoryChangeSet, ThreadHistoryBuilderError> {
        if applied <= target {
            return Err(ThreadHistoryBuilderError::InvalidRollback { target, applied });
        }
        self.validate_sequence(applied)?;

        let mut seen_item_ids = HashSet::new();
        let removed_item_ids = self
            .items
            .iter()
            .filter(|item| item.sequence > target)
            .filter_map(|item| {
                seen_item_ids
                    .insert(item.item_id.clone())
                    .then_some(item.item_id.clone())
            })
            .collect::<Vec<_>>();
        self.items.retain(|item| item.sequence <= target);
        self.item_indexes.clear();
        for (index, item) in self.items.iter().enumerate() {
            self.item_indexes.insert(
                ItemKey {
                    turn_id: item.turn_id.clone(),
                    item_id: item.item_id.clone(),
                },
                index,
            );
        }

        let removed_turn_ids = self
            .turns
            .iter()
            .filter(|turn| {
                self.turn_sequences
                    .get(&turn.turn_id)
                    .is_some_and(|sequence| *sequence > target)
            })
            .map(|turn| turn.turn_id.clone())
            .collect::<Vec<_>>();
        for turn_id in &removed_turn_ids {
            if let Some(index) = self.turn_indexes.remove(turn_id) {
                self.turns[index].items.clear();
            }
            self.turn_sequences.remove(turn_id);
        }
        self.turns
            .retain(|turn| !removed_turn_ids.contains(&turn.turn_id));
        for turn in &mut self.turns {
            turn.items.retain(|item| item.sequence <= target);
        }
        self.turn_indexes.clear();
        for (index, turn) in self.turns.iter().enumerate() {
            self.turn_indexes.insert(turn.turn_id.clone(), index);
        }
        let remaining_turn_ids = self
            .turns
            .iter()
            .map(|turn| turn.turn_id.clone())
            .collect::<Vec<_>>();
        for turn_id in remaining_turn_ids {
            self.sync_turn_items(&turn_id);
        }

        if self
            .active_turn_id
            .as_ref()
            .is_some_and(|turn_id| removed_turn_ids.contains(turn_id))
        {
            self.active_turn_id = None;
        }
        self.sequence = Some(applied);
        Ok(ThreadHistoryChangeSet {
            sequence: applied,
            removed_item_ids,
            removed_turn_ids,
            rollback_to_sequence: Some(target),
            ..Default::default()
        })
    }

    /// Pages raw items without exposing store-owned cursor encoding.
    pub fn page_items(
        &self,
        offset: usize,
        limit: usize,
        sort_direction: SortDirection,
    ) -> ThreadHistoryPage<ThreadItem> {
        page(&self.items, offset, limit, sort_direction)
    }

    /// Pages turn snapshots without exposing store-owned cursor encoding.
    pub fn page_turns(
        &self,
        offset: usize,
        limit: usize,
        sort_direction: SortDirection,
    ) -> ThreadHistoryPage<Turn> {
        page(&self.turns, offset, limit, sort_direction)
    }

    fn insert_snapshot_item(&mut self, item: ThreadItem) -> Result<(), ThreadHistoryBuilderError> {
        if !self.turn_indexes.contains_key(&item.turn_id) {
            return Err(ThreadHistoryBuilderError::MissingTurn {
                turn_id: item.turn_id,
            });
        }
        if let Some(previous) = self
            .items
            .iter()
            .find(|previous| previous.item_id == item.item_id)
        {
            if previous.turn_id != item.turn_id {
                return Err(ThreadHistoryBuilderError::ItemTurnIdentityMismatch {
                    item_id: item.item_id,
                    expected: previous.turn_id.clone(),
                    actual: item.turn_id,
                });
            }
            return Err(ThreadHistoryBuilderError::DuplicateItem {
                item_id: item.item_id,
            });
        }
        let key = ItemKey {
            turn_id: item.turn_id.clone(),
            item_id: item.item_id.clone(),
        };
        let index = self.items.len();
        self.item_indexes.insert(key, index);
        if let Some(turn_index) = self.turn_indexes.get(&item.turn_id).copied() {
            self.turns[turn_index].items.push(item.clone());
        }
        self.items.push(item);
        Ok(())
    }

    fn change_set_already_applied(&self, changes: &ThreadHistoryChangeSet) -> bool {
        changes.rollback_to_sequence.is_none()
            && changes
                .removed_turn_ids
                .iter()
                .all(|turn_id| !self.turn_indexes.contains_key(turn_id))
            && changes
                .removed_item_ids
                .iter()
                .all(|item_id| !self.items.iter().any(|item| &item.item_id == item_id))
            && changes.changed_turns.iter().all(|turn| {
                self.turn_snapshot(turn.turn_id.as_str())
                    .is_some_and(|previous| {
                        merge_turn_snapshot(previous.clone(), turn.clone()) == previous
                    })
            })
            && changes.changed_items.iter().all(|item| {
                self.items
                    .iter()
                    .find(|previous| previous.item_id == item.item_id)
                    .is_some_and(|previous| {
                        merge_item_snapshot(previous.clone(), item.clone()) == *previous
                    })
            })
    }

    fn upsert_turn(
        &mut self,
        mut turn: Turn,
        sequence: u64,
    ) -> Result<Option<Turn>, ThreadHistoryBuilderError> {
        if self.turn_indexes.contains_key(&turn.turn_id) {
            let index = self.turn_indexes[&turn.turn_id];
            turn.items.clear();
            let merged = merge_turn_snapshot(self.turns[index].clone(), turn);
            if merged == self.turns[index] {
                self.turn_sequences
                    .insert(self.turns[index].turn_id.clone(), sequence);
                let turn_id = self.turns[index].turn_id.clone();
                self.sync_turn_items(&turn_id);
                return Ok(None);
            }
            self.turns[index] = merged.clone();
            self.turn_sequences.insert(merged.turn_id.clone(), sequence);
            let turn_id = merged.turn_id.clone();
            self.sync_turn_items(&turn_id);
            let merged = self.turns[index].clone();
            if merged.status.is_terminal() {
                if self.active_turn_id.as_ref() == Some(&merged.turn_id) {
                    self.active_turn_id = None;
                }
            } else {
                self.active_turn_id = Some(merged.turn_id.clone());
            }
            Ok(Some(merged))
        } else {
            turn.items.clear();
            let index = self.turns.len();
            self.turn_indexes.insert(turn.turn_id.clone(), index);
            self.turn_sequences.insert(turn.turn_id.clone(), sequence);
            self.turns.push(turn);
            let turn_id = self.turns[index].turn_id.clone();
            self.sync_turn_items(&turn_id);
            let snapshot = self.turns[index].clone();
            if !snapshot.status.is_terminal() {
                self.active_turn_id = Some(turn_id);
            }
            Ok(Some(snapshot))
        }
    }

    fn sync_turn_items(&mut self, turn_id: &TurnId) {
        let Some(turn_index) = self.turn_indexes.get(turn_id).copied() else {
            return;
        };
        self.turns[turn_index].items = self
            .items
            .iter()
            .filter(|item| &item.turn_id == turn_id)
            .cloned()
            .collect();
    }

    fn upsert_item(
        &mut self,
        item: ThreadItem,
    ) -> Result<Option<ThreadItem>, ThreadHistoryBuilderError> {
        if !self.turn_indexes.contains_key(&item.turn_id) {
            return Err(ThreadHistoryBuilderError::MissingTurn {
                turn_id: item.turn_id,
            });
        }
        if let Some(previous) = self
            .items
            .iter()
            .find(|previous| previous.item_id == item.item_id)
        {
            if previous.turn_id != item.turn_id {
                return Err(ThreadHistoryBuilderError::ItemTurnIdentityMismatch {
                    item_id: item.item_id,
                    expected: previous.turn_id.clone(),
                    actual: item.turn_id,
                });
            }
        }
        let key = ItemKey {
            turn_id: item.turn_id.clone(),
            item_id: item.item_id.clone(),
        };
        let snapshot = if let Some(index) = self.item_indexes.get(&key).copied() {
            let previous = self.items[index].clone();
            let merged = merge_item_snapshot(previous.clone(), item);
            if merged == previous {
                return Ok(None);
            }
            self.items[index] = merged.clone();
            merged
        } else {
            let index = self.items.len();
            self.item_indexes.insert(key, index);
            self.items.push(item.clone());
            item
        };
        let turn_index = self.turn_indexes[&snapshot.turn_id];
        if let Some(existing) = self.turns[turn_index]
            .items
            .iter_mut()
            .find(|existing| existing.item_id == snapshot.item_id)
        {
            *existing = snapshot.clone();
        } else {
            self.turns[turn_index].items.push(snapshot.clone());
        }
        Ok(Some(snapshot))
    }

    fn remove_item(&mut self, item_id: &ItemId) -> bool {
        let before = self.items.len();
        self.items.retain(|item| &item.item_id != item_id);
        if before == self.items.len() {
            return false;
        }
        for turn in &mut self.turns {
            turn.items.retain(|item| &item.item_id != item_id);
        }
        self.rebuild_item_indexes();
        true
    }

    fn remove_turn(&mut self, turn_id: &TurnId) -> bool {
        let Some(index) = self.turn_indexes.remove(turn_id) else {
            return false;
        };
        self.turns.remove(index);
        self.turn_sequences.remove(turn_id);
        self.items.retain(|item| &item.turn_id != turn_id);
        if self.active_turn_id.as_ref() == Some(turn_id) {
            self.active_turn_id = self
                .turns
                .iter()
                .rev()
                .find(|turn| !turn.status.is_terminal())
                .map(|turn| turn.turn_id.clone());
        }
        self.rebuild_turn_indexes();
        self.rebuild_item_indexes();
        true
    }

    fn rebuild_turn_indexes(&mut self) {
        self.turn_indexes.clear();
        for (index, turn) in self.turns.iter().enumerate() {
            self.turn_indexes.insert(turn.turn_id.clone(), index);
        }
    }

    fn rebuild_item_indexes(&mut self) {
        self.item_indexes.clear();
        for (index, item) in self.items.iter().enumerate() {
            self.item_indexes.insert(
                ItemKey {
                    turn_id: item.turn_id.clone(),
                    item_id: item.item_id.clone(),
                },
                index,
            );
        }
    }

    fn validate_sequence(&self, next: u64) -> Result<(), ThreadHistoryBuilderError> {
        if let Some(current) = self.sequence {
            if next < current {
                return Err(ThreadHistoryBuilderError::SequenceRegression { current, next });
            }
        }
        Ok(())
    }

    fn validate_items(&self, items: &[ThreadItem]) -> Result<(), ThreadHistoryBuilderError> {
        let mut session_id = self.session_id.clone();
        let mut thread_id = self.thread_id.clone();
        for item in items {
            validate_identity(
                &mut session_id,
                &mut thread_id,
                &item.session_id,
                &item.thread_id,
            )?;
            validate_item_content(item)?;
        }
        Ok(())
    }

    fn validate_turns(&self, turns: &[Turn]) -> Result<(), ThreadHistoryBuilderError> {
        let mut session_id = self.session_id.clone();
        let mut thread_id = self.thread_id.clone();
        for turn in turns {
            validate_identity(
                &mut session_id,
                &mut thread_id,
                &turn.session_id,
                &turn.thread_id,
            )?;
        }
        Ok(())
    }

    fn adopt_identity(&mut self, session_id: &SessionId, thread_id: &ThreadId) {
        if self.session_id.is_none() {
            self.session_id = Some(session_id.clone());
        }
        if self.thread_id.is_none() {
            self.thread_id = Some(thread_id.clone());
        }
    }
}

fn push_unique<T: PartialEq>(values: &mut Vec<T>, value: T) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn extend_unique<T: PartialEq>(values: &mut Vec<T>, incoming: Vec<T>) {
    for value in incoming {
        push_unique(values, value);
    }
}

fn page<T: Clone>(
    values: &[T],
    offset: usize,
    limit: usize,
    sort_direction: SortDirection,
) -> ThreadHistoryPage<T> {
    if limit == 0 || values.is_empty() || offset >= values.len() {
        return ThreadHistoryPage {
            data: Vec::new(),
            next_offset: None,
            previous_offset: (offset > 0).then_some(offset.saturating_sub(limit.max(1))),
        };
    }
    let end = offset.saturating_add(limit).min(values.len());
    let data = match sort_direction {
        SortDirection::Asc => values[offset..end].to_vec(),
        SortDirection::Desc => values
            .iter()
            .rev()
            .skip(offset)
            .take(limit)
            .cloned()
            .collect(),
    };
    let total = values.len();
    let consumed = offset.saturating_add(data.len());
    ThreadHistoryPage {
        data,
        next_offset: (consumed < total).then_some(consumed),
        previous_offset: (offset > 0).then_some(offset.saturating_sub(limit.max(1))),
    }
}

#[cfg(test)]
#[path = "history_tests.rs"]
mod tests;
