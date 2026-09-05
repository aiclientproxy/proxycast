use super::deadline::request;
use std::time::Duration;

#[tokio::test]
async fn zero_timeout_preserves_transport_error() {
    let result = request("wait", Duration::ZERO, async {
        Err::<(), _>(tonic::Status::not_found("missing"))
    })
    .await;
    assert_eq!(
        result,
        Err("wait failed: status: NotFound, message: \"missing\", details: [], metadata: MetadataMap { headers: {} }".to_string())
    );
}

#[tokio::test(start_paused = true)]
async fn nonzero_timeout_fails_when_transport_stalls() {
    let task = tokio::spawn(request("wait", Duration::from_secs(1), async {
        std::future::pending::<Result<(), tonic::Status>>().await
    }));
    tokio::task::yield_now().await;
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(
        task.await.expect("deadline task"),
        Err("wait timed out".to_string())
    );
}
