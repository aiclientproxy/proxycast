//! gRPC listener owner.
//!
//! URL parsing belongs to `transport`; this module owns only the listener
//! startup boundary and delegates service construction to the gRPC service
//! module.

use std::net::SocketAddr;

pub(crate) async fn run_tcp_listener(address: SocketAddr) -> Result<(), String> {
    crate::grpc::run_grpc_address(address).await
}
