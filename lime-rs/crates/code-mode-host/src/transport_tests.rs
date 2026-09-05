use super::{parse_listen_url, ListenTransport};

#[test]
fn parses_stdio_variants() {
    for value in ["", "stdio", "stdio://"] {
        assert_eq!(parse_listen_url(value), Ok(ListenTransport::Stdio));
    }
}

#[test]
fn parses_grpc_socket_address() {
    assert_eq!(
        parse_listen_url("grpc://127.0.0.1:4100"),
        Ok(ListenTransport::Grpc("127.0.0.1:4100".parse().unwrap()))
    );
}

#[test]
fn rejects_unknown_transport() {
    let error = parse_listen_url("ws://127.0.0.1:4100").expect_err("unknown transport");
    assert!(error.contains("unsupported"));
}
