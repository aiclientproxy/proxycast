use super::peer::PeerState;

#[test]
fn peer_failure_is_idempotent_and_disconnects() {
    let peer = PeerState::new();
    assert!(!peer.is_disconnected());
    peer.fail("first failure");
    peer.fail("second failure");
    assert!(peer.is_disconnected());
}
