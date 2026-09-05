//! Lifecycle state for a remote gRPC session.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SessionState {
    Open,
    Closed,
}

pub(super) fn is_closed(closed: &std::sync::atomic::AtomicBool) -> bool {
    closed.load(std::sync::atomic::Ordering::Acquire)
}

pub(super) fn lifecycle(closed: &std::sync::atomic::AtomicBool) -> SessionState {
    if is_closed(closed) {
        SessionState::Closed
    } else {
        SessionState::Open
    }
}
