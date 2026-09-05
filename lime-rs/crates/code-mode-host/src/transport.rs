//! Host transport selection.
//!
//! The host owns transport parsing; protocol and runtime crates must remain
//! unaware of listen URLs.  Stdio is the default local sidecar transport and
//! gRPC is the explicit remote transport.

use std::net::SocketAddr;

use crate::grpc_transport;

pub const DEFAULT_LISTEN_URL: &str = "stdio";

#[derive(Debug, Clone, Eq, PartialEq)]
enum ListenTransport {
    Stdio,
    Grpc(SocketAddr),
}

pub(crate) async fn run_transport(listen_url: &str) -> Result<(), String> {
    match parse_listen_url(listen_url)? {
        ListenTransport::Stdio => crate::run_stdio().await,
        ListenTransport::Grpc(address) => grpc_transport::run_tcp_listener(address).await,
    }
}

fn parse_listen_url(listen_url: &str) -> Result<ListenTransport, String> {
    if matches!(listen_url, "" | "stdio" | "stdio://") {
        return Ok(ListenTransport::Stdio);
    }

    if let Some(address) = listen_url.strip_prefix("grpc://") {
        return address
            .parse::<SocketAddr>()
            .map(ListenTransport::Grpc)
            .map_err(|error| {
                format!(
                    "invalid gRPC --listen URL `{listen_url}`; expected `grpc://IP:PORT`: {error}"
                )
            });
    }

    Err(format!(
        "unsupported --listen URL `{listen_url}`; expected `grpc://IP:PORT`, `stdio`, or `stdio://`"
    ))
}

#[cfg(test)]
#[path = "transport_tests.rs"]
mod tests;
