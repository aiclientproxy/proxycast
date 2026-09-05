//! Tracks when the canonical TUI transcript should be rebuilt after resize.
//!
//! This state machine is intentionally renderer agnostic. The App owns the
//! projection; this module only records observed/rebuilt widths and ensures a
//! resize during streaming receives one final source-backed repaint.

use std::time::{Duration, Instant};

pub(crate) const TRANSCRIPT_REFLOW_DEBOUNCE: Duration = Duration::from_millis(75);

#[derive(Debug, Default)]
pub(crate) struct TranscriptReflowState {
    last_observed_width: Option<u16>,
    last_reflow_width: Option<u16>,
    pending_reflow_width: Option<u16>,
    pending_until: Option<Instant>,
    visible_history_rows: Option<u16>,
    ran_during_stream: bool,
    resize_requested_during_stream: bool,
}

impl TranscriptReflowState {
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn set_visible_history_rows(&mut self, rows: u16) {
        self.visible_history_rows = Some(rows.max(1));
    }

    pub(crate) fn visible_history_rows(&self) -> Option<u16> {
        self.visible_history_rows
    }

    pub(crate) fn note_width(&mut self, width: u16) -> TranscriptWidthChange {
        let previous_width = self.last_observed_width.replace(width);
        if previous_width.is_none() {
            self.last_reflow_width = Some(width);
        }
        TranscriptWidthChange {
            changed: previous_width.is_some_and(|previous| previous != width),
            initialized: previous_width.is_none(),
        }
    }

    pub(crate) fn reflow_needed_for_width(&self, width: u16) -> bool {
        self.last_reflow_width != Some(width) && self.pending_reflow_width != Some(width)
    }

    pub(crate) fn schedule_debounced(&mut self, target_width: Option<u16>) -> bool {
        if let Some(target_width) = target_width {
            self.pending_reflow_width = Some(target_width);
        }
        self.pending_until = Some(Instant::now() + TRANSCRIPT_REFLOW_DEBOUNCE);
        false
    }

    pub(crate) fn schedule_immediate(&mut self) {
        self.pending_reflow_width = None;
        self.pending_until = Some(Instant::now());
    }

    #[cfg(test)]
    pub(crate) fn set_due_for_test(&mut self) {
        self.pending_until = Some(Instant::now() - Duration::from_millis(1));
    }

    pub(crate) fn pending_is_due(&self, now: Instant) -> bool {
        self.pending_until.is_some_and(|deadline| now >= deadline)
    }

    pub(crate) fn pending_until(&self) -> Option<Instant> {
        self.pending_until
    }

    pub(crate) fn has_pending_reflow(&self) -> bool {
        self.pending_until.is_some()
    }

    pub(crate) fn clear_pending_reflow(&mut self) {
        self.pending_until = None;
        self.pending_reflow_width = None;
    }

    pub(crate) fn mark_reflowed_width(&mut self, width: u16) -> bool {
        self.last_reflow_width.replace(width) != Some(width)
    }

    pub(crate) fn mark_ran_during_stream(&mut self) {
        self.ran_during_stream = true;
    }

    pub(crate) fn mark_resize_requested_during_stream(&mut self) {
        self.resize_requested_during_stream = true;
    }

    pub(crate) fn take_stream_finish_reflow_needed(&mut self) -> bool {
        let needed = self.ran_during_stream || self.resize_requested_during_stream;
        self.ran_during_stream = false;
        self.resize_requested_during_stream = false;
        needed
    }

    pub(crate) fn clear_stream_flags(&mut self) {
        self.ran_during_stream = false;
        self.resize_requested_during_stream = false;
    }
}

pub(crate) struct TranscriptWidthChange {
    pub(crate) changed: bool,
    pub(crate) initialized: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_width_sets_baseline_without_reflow() {
        let mut state = TranscriptReflowState::default();
        let change = state.note_width(80);
        assert!(change.initialized);
        assert!(!change.changed);
        assert!(!state.reflow_needed_for_width(80));
    }

    #[test]
    fn changed_width_is_debounced_and_due_is_observable() {
        let mut state = TranscriptReflowState::default();
        state.note_width(80);
        let change = state.note_width(100);
        assert!(change.changed);
        state.schedule_debounced(Some(100));
        assert!(state.has_pending_reflow());
        state.set_due_for_test();
        assert!(state.pending_is_due(Instant::now()));
        state.clear_pending_reflow();
        assert!(state.reflow_needed_for_width(100));
    }

    #[test]
    fn stream_reflow_request_is_drained_once() {
        let mut state = TranscriptReflowState::default();
        state.mark_ran_during_stream();
        state.mark_resize_requested_during_stream();
        assert!(state.take_stream_finish_reflow_needed());
        assert!(!state.take_stream_finish_reflow_needed());
    }
}
