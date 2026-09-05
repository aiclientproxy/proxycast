//! gRPC endpoint resolution and channel establishment.

use tonic::transport::{Channel, Endpoint};

pub(super) async fn connect(endpoint: &str) -> Result<Channel, String> {
    let endpoint = endpoint
        .strip_prefix("grpc://")
        .map(|address| format!("http://{address}"))
        .unwrap_or_else(|| endpoint.to_string());
    Endpoint::from_shared(endpoint)
        .map_err(|error| format!("invalid code-mode gRPC endpoint: {error}"))?
        .connect()
        .await
        .map_err(|error| format!("failed to connect to code-mode gRPC host: {error}"))
}
