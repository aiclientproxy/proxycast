//! Deadline policy for gRPC session operations.

use std::future::Future;
use std::time::Duration;

pub(super) async fn request<T, F>(label: &str, timeout: Duration, future: F) -> Result<T, String>
where
    F: Future<Output = Result<T, tonic::Status>>,
{
    if timeout.is_zero() {
        return future
            .await
            .map_err(|error| format!("{label} failed: {error}"));
    }
    tokio::time::timeout(timeout, future)
        .await
        .map_err(|_| format!("{label} timed out"))?
        .map_err(|error| format!("{label} failed: {error}"))
}
