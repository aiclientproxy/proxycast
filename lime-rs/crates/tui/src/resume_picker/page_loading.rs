//! Cursor pagination state for the Codex-shaped resume picker.
//!
//! The App Server owns the persisted thread index. This module only tracks
//! picker-local request ordering and scan progress so stale pages cannot
//! overwrite a newer search.

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PageCursor {
    AppServer(String),
}

/// Selects whether thread listing trusts indexed metadata or uses the store's
/// normal behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
pub(super) enum PageLoadMode {
    /// Return only indexed metadata.
    StateDbOnly,
    /// Use the store's normal listing behavior.
    StoreDefault,
}

/// Tracks the current request and the cursor policy inherited by its next page.
#[derive(Debug)]
pub(super) struct PaginationState {
    pub(super) next_cursor: Option<PageCursor>,
    next_page_mode: PageLoadMode,
    pub(super) num_scanned_files: usize,
    pub(super) reached_scan_cap: bool,
    loading: LoadingState,
}

#[derive(Clone, Copy, Debug)]
enum LoadingState {
    Idle,
    Pending(PendingLoad),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PendingLoad {
    request_token: usize,
    pub(super) search_token: Option<usize>,
    pub(super) mode: PageLoadMode,
}

impl PaginationState {
    pub(super) fn new() -> Self {
        Self {
            next_cursor: None,
            next_page_mode: PageLoadMode::StoreDefault,
            num_scanned_files: 0,
            reached_scan_cap: false,
            loading: LoadingState::Idle,
        }
    }

    pub(super) fn reset(&mut self) {
        *self = Self::new();
    }

    pub(super) fn start_load(
        &mut self,
        request_token: usize,
        search_token: Option<usize>,
        mode: PageLoadMode,
    ) {
        self.loading = LoadingState::Pending(PendingLoad {
            request_token,
            search_token,
            mode,
        });
    }

    /// Completes only the matching in-flight request.
    pub(super) fn finish_load(&mut self, request_token: usize) -> Option<PendingLoad> {
        let LoadingState::Pending(pending) = self.loading else {
            return None;
        };
        if pending.request_token != request_token {
            return None;
        }
        self.loading = LoadingState::Idle;
        self.next_page_mode = pending.mode;
        Some(pending)
    }

    /// Records a page cursor and accumulates scan progress.
    pub(super) fn complete_page(
        &mut self,
        next_cursor: Option<PageCursor>,
        num_scanned_files: usize,
        reached_scan_cap: bool,
    ) {
        self.next_cursor = next_cursor;
        self.num_scanned_files = self.num_scanned_files.saturating_add(num_scanned_files);
        self.reached_scan_cap |= reached_scan_cap;
    }

    pub(super) fn next_page(&self) -> Option<(PageCursor, PageLoadMode)> {
        if self.is_loading() {
            return None;
        }
        self.next_cursor
            .clone()
            .map(|cursor| (cursor, self.next_page_mode))
    }

    pub(super) fn is_loading(&self) -> bool {
        matches!(self.loading, LoadingState::Pending(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_request_cannot_finish_newer_load() {
        let mut state = PaginationState::new();
        state.start_load(1, Some(7), PageLoadMode::StateDbOnly);
        state.start_load(2, Some(8), PageLoadMode::StoreDefault);

        assert!(state.finish_load(1).is_none());
        assert!(state.is_loading());
        assert_eq!(
            state.finish_load(2),
            Some(PendingLoad {
                request_token: 2,
                search_token: Some(8),
                mode: PageLoadMode::StoreDefault,
            })
        );
        assert!(!state.is_loading());
    }

    #[test]
    fn next_page_is_hidden_while_loading() {
        let mut state = PaginationState::new();
        state.complete_page(
            Some(PageCursor::AppServer(String::from("cursor-1"))),
            25,
            false,
        );
        assert_eq!(
            state.next_page(),
            Some((
                PageCursor::AppServer(String::from("cursor-1")),
                PageLoadMode::StoreDefault
            ))
        );

        state.start_load(3, None, PageLoadMode::StateDbOnly);
        assert!(state.next_page().is_none());
    }

    #[test]
    fn scan_progress_accumulates_and_cap_is_sticky() {
        let mut state = PaginationState::new();
        state.complete_page(None, usize::MAX, false);
        state.complete_page(None, 1, true);

        assert_eq!(state.num_scanned_files, usize::MAX);
        assert!(state.reached_scan_cap);
    }

    #[test]
    fn reset_discards_cursor_and_scan_progress() {
        let mut state = PaginationState::new();
        state.complete_page(
            Some(PageCursor::AppServer(String::from("cursor-1"))),
            4,
            true,
        );
        state.reset();

        assert!(state.next_page().is_none());
        assert_eq!(state.num_scanned_files, 0);
        assert!(!state.reached_scan_cap);
        assert!(!state.is_loading());
    }
}
