use super::DEFAULT_LISTEN_URL;

#[test]
fn default_transport_is_stdio() {
    assert_eq!(DEFAULT_LISTEN_URL, "stdio");
}
