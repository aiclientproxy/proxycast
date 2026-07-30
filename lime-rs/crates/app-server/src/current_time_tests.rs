use app_server_protocol::protocol::v2::METHOD_CURRENT_TIME_READ;
use app_server_protocol::{JsonRpcError, JsonRpcMessage};
use app_server_transport::ConnectionId;
use chrono::{DateTime, Utc};
use serde_json::json;
use std::time::Duration;
use tokio::sync::mpsc;

use crate::current_time::CURRENT_TIME_REQUEST_TIMEOUT;
use crate::server_request::ServerRequestError;
use crate::{AppServer, AppServerError, CurrentTimeReadError};

async fn register_client(
    server: &AppServer,
    connection_id: ConnectionId,
    thread_id: Option<&str>,
) -> mpsc::Receiver<app_server_transport::QueuedOutgoingMessage> {
    let (writer, outbound) = mpsc::channel(4);
    server.register_transport_writer(connection_id, writer, None);
    server
        .thread_states
        .connection_initialized(connection_id)
        .await;
    if let Some(thread_id) = thread_id {
        assert!(
            server
                .thread_states
                .subscribe_connection(agent_protocol::ThreadId::new(thread_id), connection_id)
                .await
        );
    }
    outbound
}

async fn next_request(
    outbound: &mut mpsc::Receiver<app_server_transport::QueuedOutgoingMessage>,
) -> app_server_protocol::JsonRpcRequest {
    let queued = outbound.recv().await.expect("current-time request");
    let JsonRpcMessage::Request(request) = queued.message.into_json_rpc_message() else {
        panic!("expected current-time JSON-RPC request");
    };
    request
}

#[tokio::test]
async fn current_time_round_trips_through_the_only_transport_client() {
    let server = AppServer::new();
    let connection_id = ConnectionId(7);
    let mut outbound = register_client(&server, connection_id, Some("thread-7")).await;
    let pending = tokio::spawn({
        let server = server.clone();
        async move { server.read_current_time("thread-7").await }
    });

    let request = next_request(&mut outbound).await;
    assert_eq!(request.method, METHOD_CURRENT_TIME_READ);
    assert_eq!(request.params, Some(json!({ "threadId": "thread-7" })));
    server.resolve_transport_server_request_response(
        connection_id,
        request.id,
        json!({ "currentTimeAt": 1_783_860_000_i64 }),
    );

    assert_eq!(
        pending
            .await
            .expect("current-time task")
            .expect("current time"),
        DateTime::<Utc>::from_timestamp(1_783_860_000, 0).expect("valid timestamp")
    );
    assert_eq!(server.server_requests.pending_count(), 0);
}

#[tokio::test]
async fn current_time_requires_exactly_one_thread_subscriber_without_leaking_waiters() {
    let server = AppServer::new();
    let unavailable = server
        .read_current_time_with_timeout("thread-none".to_string(), Duration::from_millis(20))
        .await
        .expect_err("missing client must fail closed");
    assert!(matches!(unavailable, CurrentTimeReadError::TimedOut { .. }));
    assert_eq!(server.server_requests.pending_count(), 0);

    let _first = register_client(&server, ConnectionId(1), Some("thread-many")).await;
    let _second = register_client(&server, ConnectionId(2), Some("thread-many")).await;
    let ambiguous = server
        .read_current_time("thread-many")
        .await
        .expect_err("multiple clients must fail closed");
    assert!(matches!(
        ambiguous,
        CurrentTimeReadError::AppServer(AppServerError::ServerRequest(
            ServerRequestError::ClientAmbiguous { client_count: 2 }
        ))
    ));
    assert_eq!(server.server_requests.pending_count(), 0);

    let server = AppServer::new();
    let mut subscribed = register_client(&server, ConnectionId(3), Some("thread-one")).await;
    let _unrelated = register_client(&server, ConnectionId(4), None).await;
    let pending = tokio::spawn({
        let server = server.clone();
        async move { server.read_current_time("thread-one").await }
    });
    let request = next_request(&mut subscribed).await;
    server.resolve_transport_server_request_response(
        ConnectionId(3),
        request.id,
        json!({ "currentTimeAt": 1_783_860_000_i64 }),
    );
    pending
        .await
        .expect("current-time task")
        .expect("unrelated connections must not make the route ambiguous");
}

#[tokio::test]
async fn current_time_rejects_non_integer_and_out_of_range_responses() {
    let cases = [
        (json!({ "currentTimeAt": 1.5 }), "fractional"),
        (json!({ "currentTimeAt": "1783860000" }), "string"),
        (json!({ "currentTimeAt": i64::MAX }), "out-of-range"),
    ];

    for (response, scenario) in cases {
        let server = AppServer::new();
        let connection_id = ConnectionId(9);
        let mut outbound = register_client(&server, connection_id, Some("thread-invalid")).await;
        let pending = tokio::spawn({
            let server = server.clone();
            async move { server.read_current_time("thread-invalid").await }
        });
        let request = next_request(&mut outbound).await;
        server.resolve_transport_server_request_response(connection_id, request.id, response);

        let error = pending
            .await
            .expect("current-time task")
            .expect_err(scenario);
        match scenario {
            "out-of-range" => {
                assert!(matches!(error, CurrentTimeReadError::OutOfRange { .. }))
            }
            _ => assert!(matches!(
                error,
                CurrentTimeReadError::InvalidResponse { .. }
            )),
        }
        assert_eq!(server.server_requests.pending_count(), 0, "{scenario}");
    }
}

#[tokio::test]
async fn current_time_propagates_json_rpc_errors() {
    let server = AppServer::new();
    let connection_id = ConnectionId(11);
    let mut outbound = register_client(&server, connection_id, Some("thread-rejected")).await;
    let pending = tokio::spawn({
        let server = server.clone();
        async move { server.read_current_time("thread-rejected").await }
    });
    let request = next_request(&mut outbound).await;
    server.resolve_transport_server_request_error(
        connection_id,
        request.id,
        JsonRpcError::new(-32000, "host clock unavailable"),
    );

    assert!(matches!(
        pending.await.expect("current-time task"),
        Err(CurrentTimeReadError::AppServer(
            AppServerError::ServerRequest(ServerRequestError::ClientRejected { .. })
        ))
    ));
    assert_eq!(server.server_requests.pending_count(), 0);
}

#[tokio::test]
async fn current_time_deadline_cancels_the_exact_waiter() {
    assert_eq!(CURRENT_TIME_REQUEST_TIMEOUT, Duration::from_secs(10));

    let server = AppServer::new();
    let connection_id = ConnectionId(13);
    let mut outbound = register_client(&server, connection_id, Some("thread-timeout")).await;
    let pending = tokio::spawn({
        let server = server.clone();
        async move {
            server
                .read_current_time_with_timeout(
                    "thread-timeout".to_string(),
                    Duration::from_millis(20),
                )
                .await
        }
    });
    let request = next_request(&mut outbound).await;
    let request_id = request.id;

    assert!(matches!(
        pending.await.expect("current-time task"),
        Err(CurrentTimeReadError::TimedOut { .. })
    ));
    assert_eq!(server.server_requests.pending_count(), 0);
    assert!(matches!(
        server.server_requests.resolve_transport_response(
            connection_id,
            request_id.clone(),
            json!({ "currentTimeAt": 1_783_860_000_i64 }),
        ),
        Err(ServerRequestError::RequestNotFound { id }) if id == request_id
    ));
}
