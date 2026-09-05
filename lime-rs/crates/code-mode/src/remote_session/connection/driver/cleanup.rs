//! Session cleanup coordination for the process connection driver.

use code_mode_protocol::{RuntimeCodeModeCellId, RuntimeCodeModeSessionDelegate};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use super::session_registry::ClosedCell;

pub(crate) struct Cleanup {
    cancellation: CancellationToken,
}

impl Cleanup {
    pub(crate) fn new(cancellation: CancellationToken) -> Self {
        Self { cancellation }
    }

    pub(crate) fn close_cells(&self, cells: impl IntoIterator<Item = ClosedCell>) {
        for cell in cells {
            notify_cell_closed(&cell.delegate, &cell.cell_id);
        }
    }

    pub(crate) fn fail(&self) {
        self.cancellation.cancel();
    }
}

pub(crate) fn notify_cell_closed(
    delegate: &Arc<dyn RuntimeCodeModeSessionDelegate>,
    cell_id: &RuntimeCodeModeCellId,
) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| delegate.cell_closed(cell_id)));
}
