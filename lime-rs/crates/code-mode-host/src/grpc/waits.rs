//! Active wait cancellation and retirement state.

use std::sync::Arc;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
pub(super) struct WaitControl {
    pub(super) cancellation: CancellationToken,
    pub(super) retired: Arc<Notify>,
}
