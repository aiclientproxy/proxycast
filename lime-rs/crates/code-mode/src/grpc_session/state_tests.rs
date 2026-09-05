use super::state::{is_closed, lifecycle, SessionState};
use std::sync::atomic::AtomicBool;

#[test]
fn lifecycle_tracks_closed_flag() {
    let closed = AtomicBool::new(false);
    assert!(!is_closed(&closed));
    assert_eq!(lifecycle(&closed), SessionState::Open);
    closed.store(true, std::sync::atomic::Ordering::Release);
    assert!(is_closed(&closed));
    assert_eq!(lifecycle(&closed), SessionState::Closed);
}
