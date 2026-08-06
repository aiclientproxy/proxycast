use super::*;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::time::{sleep, timeout, Duration};

fn notification_method(message: &JsonRpcMessage) -> Option<&str> {
    match message {
        JsonRpcMessage::Notification(notification) => Some(notification.method.as_str()),
        _ => None,
    }
}

#[tokio::test]
async fn duplicate_resume_barrier_fails_during_prepare() {
    let server = AppServer::new();
    let bridge = server.event_bridge();
    let thread_id = agent_protocol::ThreadId::new("thread-duplicate-resume");
    let barrier = thread_state::ThreadResumeBarrier::new(
        ConnectionId(1),
        RequestId::String("resume-duplicate".to_string()),
    );

    bridge
        .prepare_thread_resume(thread_id.clone(), barrier.clone())
        .await
        .expect("prepare first resume barrier");
    let error = bridge
        .prepare_thread_resume(thread_id, barrier)
        .await
        .expect_err("duplicate resume barrier must fail closed");

    assert!(error.contains("already has resume barrier"));
    server.thread_states.clear_all_listeners().await;
}

#[tokio::test]
async fn goal_notification_waits_for_resume_and_fans_out_once() {
    let server = AppServer::new();
    let bridge = server.event_bridge();
    let thread_id = agent_protocol::ThreadId::new("thread-goal-notification-order");
    let origin_connection = ConnectionId(11);
    let peer_connection = ConnectionId(12);
    let (origin_writer, mut origin_messages) = mpsc::channel(4);
    let (peer_writer, mut peer_messages) = mpsc::channel(4);

    server.register_transport_writer(origin_connection, origin_writer, None);
    server.register_transport_writer(peer_connection, peer_writer, None);
    server.mark_transport_initialized(origin_connection);
    server.mark_transport_initialized(peer_connection);
    server
        .thread_states
        .connection_initialized(origin_connection)
        .await;
    server
        .thread_states
        .connection_initialized(peer_connection)
        .await;
    assert!(
        server
            .thread_states
            .subscribe_connection(thread_id.clone(), peer_connection)
            .await
    );

    let barrier = thread_state::ThreadResumeBarrier::new(
        origin_connection,
        RequestId::String("resume-goal-notification".to_string()),
    );
    bridge
        .prepare_thread_resume(thread_id.clone(), barrier.clone())
        .await
        .expect("prepare goal notification resume barrier");

    let notification = JsonRpcNotification::new(
        app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_CLEARED,
        Some(json!({ "threadId": thread_id.as_str() })),
    );
    let expected_notification = JsonRpcMessage::Notification(notification.clone());
    let mut outbound = server.subscribe_outbound_messages();
    let (notification_tx, notification_rx) = oneshot::channel();
    bridge
        .send_thread_command(
            thread_id.clone(),
            thread_state::ThreadListenerCommand::PublishNotification {
                notification,
                origin_connection_id: Some(origin_connection),
                completion_tx: Some(notification_tx),
            },
        )
        .await
        .expect("enqueue goal notification behind resume barrier");
    tokio::task::yield_now().await;

    assert!(matches!(
        origin_messages.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert!(matches!(
        peer_messages.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert!(matches!(
        outbound.try_recv(),
        Err(broadcast::error::TryRecvError::Empty)
    ));

    let response = JsonRpcMessage::Response(
        app_server_protocol::JsonRpcResponse::new(
            RequestId::String("resume-goal-notification".to_string()),
            json!({ "thread": { "id": thread_id.as_str() } }),
        )
        .expect("goal notification resume response"),
    );
    let (resume_tx, resume_rx) = oneshot::channel();
    bridge
        .send_thread_command(
            thread_id.clone(),
            thread_state::ThreadListenerCommand::CompleteResume {
                barrier,
                connection_id: origin_connection,
                messages: vec![response.clone()],
                subscribe: false,
                completion_tx: resume_tx,
            },
        )
        .await
        .expect("complete goal notification resume barrier");
    resume_rx
        .await
        .expect("resume listener completion")
        .expect("send response before goal notification");
    notification_rx
        .await
        .expect("goal notification completion")
        .expect("publish goal notification");

    assert_eq!(next_queued_message(&mut origin_messages).await, response);
    assert_eq!(
        next_queued_message(&mut origin_messages).await,
        expected_notification
    );
    assert_eq!(
        next_queued_message(&mut peer_messages).await,
        expected_notification
    );
    assert!(matches!(
        peer_messages.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert_eq!(
        timeout(Duration::from_secs(1), outbound.recv())
            .await
            .expect("goal notification broadcast timeout")
            .expect("goal notification broadcast"),
        expected_notification
    );
    assert!(matches!(
        outbound.try_recv(),
        Err(broadcast::error::TryRecvError::Empty)
    ));

    server.thread_states.clear_all_listeners().await;
}

#[tokio::test]
async fn goal_notification_removes_stale_peer_without_failing_origin() {
    let server = AppServer::new();
    let bridge = server.event_bridge();
    let thread_id = agent_protocol::ThreadId::new("thread-goal-stale-peer");
    let origin_connection = ConnectionId(21);
    let stale_connection = ConnectionId(22);
    let (origin_writer, mut origin_messages) = mpsc::channel(2);
    let (stale_writer, stale_messages) = mpsc::channel(2);
    drop(stale_messages);

    server.register_transport_writer(origin_connection, origin_writer, None);
    server.register_transport_writer(stale_connection, stale_writer, None);
    server.mark_transport_initialized(origin_connection);
    server.mark_transport_initialized(stale_connection);
    server
        .thread_states
        .connection_initialized(origin_connection)
        .await;
    server
        .thread_states
        .connection_initialized(stale_connection)
        .await;
    assert!(
        server
            .thread_states
            .subscribe_connection(thread_id.clone(), stale_connection)
            .await
    );

    let notification = JsonRpcNotification::new(
        app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_CLEARED,
        Some(json!({ "threadId": thread_id.as_str() })),
    );
    let expected_notification = JsonRpcMessage::Notification(notification.clone());
    let (completion_tx, completion_rx) = oneshot::channel();
    bridge
        .send_thread_command(
            thread_id.clone(),
            thread_state::ThreadListenerCommand::PublishNotification {
                notification,
                origin_connection_id: Some(origin_connection),
                completion_tx: Some(completion_tx),
            },
        )
        .await
        .expect("enqueue goal notification with stale peer");
    completion_rx
        .await
        .expect("stale peer notification completion")
        .expect("stale peer must not fail goal notification fanout");

    assert_eq!(
        next_queued_message(&mut origin_messages).await,
        expected_notification
    );
    assert!(!server
        .thread_states
        .subscribed_connection_ids(&thread_id)
        .await
        .contains(&stale_connection));

    server.thread_states.clear_all_listeners().await;
}

#[tokio::test]
async fn thread_delete_responds_then_fans_out_child_to_root_and_removes_state() {
    let server = AppServer::new();
    let origin_connection = ConnectionId(25);
    let peer_connection = ConnectionId(26);
    let (origin_writer, mut origin_messages) = mpsc::channel(8);
    let (peer_writer, mut peer_messages) = mpsc::channel(8);
    server.register_transport_writer(origin_connection, origin_writer, None);
    server.register_transport_writer(peer_connection, peer_writer, None);
    server.mark_transport_initialized(origin_connection);
    server.mark_transport_initialized(peer_connection);
    server
        .thread_states
        .connection_initialized(origin_connection)
        .await;
    server
        .thread_states
        .connection_initialized(peer_connection)
        .await;

    let child_thread = agent_protocol::ThreadId::new("thread-delete-child");
    let root_thread = agent_protocol::ThreadId::new("thread-delete-root");
    let retained_thread = agent_protocol::ThreadId::new("thread-delete-retained");
    for thread_id in [&child_thread, &root_thread, &retained_thread] {
        assert!(
            server
                .thread_states
                .subscribe_connection(thread_id.clone(), peer_connection)
                .await
        );
    }

    let response = JsonRpcMessage::Response(
        app_server_protocol::JsonRpcResponse::new(
            RequestId::String("delete-1".to_string()),
            json!({}),
        )
        .expect("thread/delete response"),
    );
    let child_deleted: JsonRpcNotification =
        app_server_protocol::protocol::v2::ServerNotification::ThreadDeleted(
            app_server_protocol::protocol::v2::ThreadDeletedNotification {
                thread_id: child_thread.to_string(),
            },
        )
        .into();
    let root_deleted: JsonRpcNotification =
        app_server_protocol::protocol::v2::ServerNotification::ThreadDeleted(
            app_server_protocol::protocol::v2::ThreadDeletedNotification {
                thread_id: root_thread.to_string(),
            },
        )
        .into();
    let child_deleted = JsonRpcMessage::Notification(child_deleted);
    let root_deleted = JsonRpcMessage::Notification(root_deleted);
    let mut messages = vec![
        response.clone(),
        child_deleted.clone(),
        root_deleted.clone(),
    ];

    assert!(
        publish_thread_delete_transport_result(&server, origin_connection, &mut messages,).await
    );

    assert_eq!(next_queued_message(&mut origin_messages).await, response);
    assert_eq!(
        next_queued_message(&mut origin_messages).await,
        child_deleted
    );
    assert_eq!(
        next_queued_message(&mut origin_messages).await,
        root_deleted
    );
    assert_eq!(next_queued_message(&mut peer_messages).await, child_deleted);
    assert_eq!(next_queued_message(&mut peer_messages).await, root_deleted);
    assert!(server
        .thread_states
        .subscribed_connection_ids(&child_thread)
        .await
        .is_empty());
    assert!(server
        .thread_states
        .subscribed_connection_ids(&root_thread)
        .await
        .is_empty());
    assert_eq!(
        server
            .thread_states
            .subscribed_connection_ids(&retained_thread)
            .await,
        vec![peer_connection]
    );
    assert!(matches!(
        origin_messages.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert!(matches!(
        peer_messages.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));

    server.thread_states.clear_all_listeners().await;
}

#[tokio::test]
async fn completed_turn_publishes_durable_goal_update_once_in_listener_fifo() {
    let temp = tempfile::tempdir().expect("goal listener tempdir");
    let database_path = temp.path().join("state.sqlite");
    let projection_store = Arc::new(
        ProjectionStore::initialize(&database_path).expect("goal listener projection store"),
    );
    let thread_id = "thread-goal-accounting-listener";
    let turn_id = "turn-goal-accounting-listener";
    let notification = app_server_protocol::protocol::v2::ThreadGoalUpdatedNotification {
        thread_id: thread_id.to_string(),
        turn_id: Some(turn_id.to_string()),
        goal: app_server_protocol::protocol::v2::ThreadGoal {
            thread_id: thread_id.to_string(),
            objective: "publish accounted usage".to_string(),
            status: app_server_protocol::protocol::v2::ThreadGoalStatus::Active,
            token_budget: Some(100),
            tokens_used: 42,
            time_used_seconds: 3,
            created_at: 1,
            updated_at: 4,
        },
    };
    let connection = rusqlite::Connection::open(&database_path).expect("open goal listener store");
    connection
        .execute(
            r#"INSERT INTO canonical_threads (
                   thread_id, session_id, thread_json, created_at_ms, updated_at_ms, archived
               ) VALUES (?1, 'session-goal-accounting-listener', '{}', 1, 1, 0)"#,
            [thread_id],
        )
        .expect("insert listener canonical thread");
    connection
        .execute(
            r#"INSERT INTO thread_goal_update_outbox (
                   thread_id, turn_id, goal_id, source_sequence, notification_json, created_at_ms
               ) VALUES (?1, ?2, 'goal-listener', 7, ?3, 4)"#,
            rusqlite::params![
                thread_id,
                turn_id,
                serde_json::to_string(&notification).expect("serialize goal update")
            ],
        )
        .expect("insert goal update outbox");
    drop(connection);

    let server = AppServer::with_runtime(
        RuntimeCore::default().with_projection_store(projection_store.clone()),
    );
    let bridge = server.event_bridge();
    let connection_id = ConnectionId(31);
    let (writer, mut messages) = mpsc::channel(4);
    server.register_transport_writer(connection_id, writer, None);
    server.mark_transport_initialized(connection_id);
    server
        .thread_states
        .connection_initialized(connection_id)
        .await;
    let canonical_thread_id = agent_protocol::ThreadId::new(thread_id);
    assert!(
        server
            .thread_states
            .subscribe_connection(canonical_thread_id.clone(), connection_id)
            .await
    );

    let completed = AgentEvent {
        event_id: "event-goal-accounting-completed".to_string(),
        sequence: 7,
        session_id: "session-goal-accounting-listener".to_string(),
        thread_id: Some(thread_id.to_string()),
        turn_id: Some(turn_id.to_string()),
        event_type: "turn.completed".to_string(),
        timestamp: "2026-07-20T00:00:04Z".to_string(),
        payload: json!({
            "turn": {
                "sessionId": "session-goal-accounting-listener",
                "threadId": thread_id,
                "turnId": turn_id,
                "status": "completed",
                "createdAtMs": 1,
                "updatedAtMs": 4,
                "startedAtMs": 1,
                "completedAtMs": 4,
                "items": [],
                "itemsView": "full"
            }
        }),
    };
    let (completion_tx, completion_rx) = oneshot::channel();
    bridge
        .send_thread_command(
            canonical_thread_id.clone(),
            thread_state::ThreadListenerCommand::PublishRuntimeEvent {
                event: completed.clone(),
                completion_tx: Some(completion_tx),
            },
        )
        .await
        .expect("publish completed event");
    let projected = completion_rx
        .await
        .expect("completed listener result")
        .expect("completed listener success");
    assert_eq!(projected.len(), 2);
    assert_eq!(
        notification_method(&projected[0]),
        Some(app_server_protocol::protocol::v2::METHOD_TURN_COMPLETED)
    );
    assert_eq!(
        notification_method(&projected[1]),
        Some(app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_UPDATED)
    );
    assert_eq!(
        notification_method(&next_queued_message(&mut messages).await),
        Some(app_server_protocol::protocol::v2::METHOD_TURN_COMPLETED)
    );
    assert_eq!(
        notification_method(&next_queued_message(&mut messages).await),
        Some(app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_UPDATED)
    );

    let (replay_tx, replay_rx) = oneshot::channel();
    bridge
        .send_thread_command(
            canonical_thread_id.clone(),
            thread_state::ThreadListenerCommand::PublishRuntimeEvent {
                event: completed,
                completion_tx: Some(replay_tx),
            },
        )
        .await
        .expect("replay completed event");
    let replayed = replay_rx
        .await
        .expect("replay listener result")
        .expect("replay listener success");
    assert_eq!(replayed.len(), 1);
    assert_eq!(
        notification_method(&replayed[0]),
        Some(app_server_protocol::protocol::v2::METHOD_TURN_COMPLETED)
    );
    assert_eq!(
        notification_method(&next_queued_message(&mut messages).await),
        Some(app_server_protocol::protocol::v2::METHOD_TURN_COMPLETED)
    );
    assert!(matches!(
        messages.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    let connection =
        rusqlite::Connection::open(&database_path).expect("reopen goal listener store");
    assert!(connection
        .query_row(
            "SELECT delivered_at_ms IS NOT NULL FROM thread_goal_update_outbox",
            [],
            |row| row.get::<_, bool>(0)
        )
        .expect("read goal outbox acknowledgement"));

    server.thread_states.clear_all_listeners().await;
}

#[tokio::test]
async fn successful_tool_finish_publishes_durable_goal_update_in_listener_fifo() {
    let temp = tempfile::tempdir().expect("tool-finish listener tempdir");
    let database_path = temp.path().join("state.sqlite");
    let projection_store = Arc::new(
        ProjectionStore::initialize(&database_path).expect("tool-finish listener projection store"),
    );
    let thread_id = "thread-tool-finish-listener";
    let turn_id = "turn-tool-finish-listener";
    let notification = app_server_protocol::protocol::v2::ThreadGoalUpdatedNotification {
        thread_id: thread_id.to_string(),
        turn_id: Some(turn_id.to_string()),
        goal: app_server_protocol::protocol::v2::ThreadGoal {
            thread_id: thread_id.to_string(),
            objective: "publish tool-finish progress".to_string(),
            status: app_server_protocol::protocol::v2::ThreadGoalStatus::Active,
            token_budget: Some(100),
            tokens_used: 25,
            time_used_seconds: 4,
            created_at: 1,
            updated_at: 4,
        },
    };
    let connection =
        rusqlite::Connection::open(&database_path).expect("open tool-finish listener store");
    connection
        .execute(
            r#"INSERT INTO canonical_threads (
                   thread_id, session_id, thread_json, created_at_ms, updated_at_ms, archived
               ) VALUES (?1, 'session-tool-finish-listener', '{}', 1, 1, 0)"#,
            [thread_id],
        )
        .expect("insert tool-finish canonical thread");
    connection
        .execute(
            r#"INSERT INTO thread_goal_update_outbox (
                   thread_id, turn_id, goal_id, source_sequence, notification_json, created_at_ms
               ) VALUES (?1, ?2, 'goal-tool-finish-listener', 7, ?3, 4)"#,
            rusqlite::params![
                thread_id,
                turn_id,
                serde_json::to_string(&notification).expect("serialize tool-finish goal update")
            ],
        )
        .expect("insert tool-finish goal update outbox");
    drop(connection);

    let server = AppServer::with_runtime(
        RuntimeCore::default().with_projection_store(projection_store.clone()),
    );
    let bridge = server.event_bridge();
    let canonical_thread_id = agent_protocol::ThreadId::new(thread_id);
    let mut outbound = server.subscribe_outbound_messages();
    let tool_finish = AgentEvent {
        event_id: "event-tool-finish-listener".to_string(),
        sequence: 7,
        session_id: "session-tool-finish-listener".to_string(),
        thread_id: Some(thread_id.to_string()),
        turn_id: Some(turn_id.to_string()),
        event_type: "item.completed".to_string(),
        timestamp: "2026-07-20T00:00:04Z".to_string(),
        payload: json!({
            "item": {
                "sessionId": "session-tool-finish-listener",
                "threadId": thread_id,
                "turnId": turn_id,
                "itemId": "tool-finish-listener",
                "sequence": 7,
                "ordinal": 1,
                "createdAtMs": 1_784_521_600_000_i64,
                "updatedAtMs": 1_784_521_604_000_i64,
                "completedAtMs": 1_784_521_604_000_i64,
                "kind": "tool",
                "status": "completed",
                "payload": {
                    "type": "tool",
                    "call_id": "call-tool-finish-listener",
                    "name": "exec_command",
                    "arguments": [],
                    "output": { "text": "done", "truncated": false }
                },
                "metadata": {}
            }
        }),
    };
    let (completion_tx, completion_rx) = oneshot::channel();
    bridge
        .send_thread_command(
            canonical_thread_id,
            thread_state::ThreadListenerCommand::PublishRuntimeEvent {
                event: tool_finish,
                completion_tx: Some(completion_tx),
            },
        )
        .await
        .expect("publish tool-finish event");
    let projected = completion_rx
        .await
        .expect("tool-finish listener result")
        .expect("tool-finish listener success");
    assert_eq!(projected.len(), 2);
    assert_eq!(
        notification_method(&projected[0]),
        Some(app_server_protocol::protocol::v2::METHOD_ITEM_COMPLETED)
    );
    assert_eq!(
        notification_method(&projected[1]),
        Some(app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_UPDATED)
    );
    for expected in [
        app_server_protocol::protocol::v2::METHOD_ITEM_COMPLETED,
        app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_UPDATED,
    ] {
        let message = timeout(Duration::from_secs(1), outbound.recv())
            .await
            .expect("tool-finish broadcast timeout")
            .expect("tool-finish broadcast");
        assert_eq!(notification_method(&message), Some(expected));
    }
    let connection =
        rusqlite::Connection::open(&database_path).expect("reopen tool-finish listener store");
    assert!(connection
        .query_row(
            "SELECT delivered_at_ms IS NOT NULL FROM thread_goal_update_outbox",
            [],
            |row| row.get::<_, bool>(0)
        )
        .expect("read tool-finish outbox acknowledgement"));

    server.thread_states.clear_all_listeners().await;
}

#[tokio::test]
async fn resume_barrier_orders_raw_jsonl_response_replay_before_deferred_live_event() {
    let temp = tempfile::TempDir::new().expect("thread listener fixture temp dir");
    let runtime = RuntimeCore::default().with_projection_store(Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("thread listener projection store"),
    ));
    let server = AppServer::with_runtime(runtime);
    let (mut input_client, input_server) = tokio::io::duplex(32 * 1024);
    let (output_server, output_client) = tokio::io::duplex(32 * 1024);
    let mut runner = tokio::spawn(run_json_lines(server.clone(), input_server, output_server));
    let mut output_lines = BufReader::new(output_client).lines();

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": METHOD_INITIALIZE,
            "params": {
                "clientInfo": {
                    "name": "thread-resume-listener-test",
                    "version": "1.0.0"
                }
            }
        }),
    )
    .await;
    next_response(&mut output_lines, RequestId::Integer(1)).await;
    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "method": METHOD_INITIALIZED,
            "params": {}
        }),
    )
    .await;

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": METHOD_THREAD_START,
            "params": {
                "model": "fixture-model",
                "modelProvider": "fixture-provider",
                "cwd": temp.path()
            }
        }),
    )
    .await;
    let start_response = next_response(&mut output_lines, RequestId::Integer(2)).await;
    let thread_id = start_response
        .result
        .pointer("/thread/id")
        .and_then(serde_json::Value::as_str)
        .expect("thread/start thread id")
        .to_string();

    let connection_id = server
        .transport_writers
        .lock()
        .expect("transport writer mutex")
        .keys()
        .copied()
        .next()
        .expect("stdio connection id");
    let pending = server.server_requests.register_for_owner(
        server_request::ServerRequestOwner::Transport(connection_id),
        "item/tool/requestUserInput",
        Some(json!({ "threadId": thread_id, "question": "continue?" })),
    );
    let pending_request_id = pending.id().clone();
    let resume_request_id = RequestId::Integer(3);
    let hold_barrier = thread_state::ThreadResumeBarrier::new(
        connection_id,
        RequestId::String("test-hold-live-event".to_string()),
    );
    let bridge = server.event_bridge();
    bridge
        .prepare_thread_resume(
            agent_protocol::ThreadId::new(thread_id.clone()),
            hold_barrier.clone(),
        )
        .await
        .expect("prepare resume barrier");
    bridge
        .send_thread_command(
            agent_protocol::ThreadId::new(thread_id.clone()),
            thread_state::ThreadListenerCommand::PublishRuntimeEvent {
                event: AgentEvent {
                    event_id: "event-deferred-live".to_string(),
                    sequence: 1,
                    session_id: start_response
                        .result
                        .pointer("/thread/sessionId")
                        .and_then(serde_json::Value::as_str)
                        .expect("thread/start session id")
                        .to_string(),
                    thread_id: Some(thread_id.clone()),
                    turn_id: Some("turn-deferred-live".to_string()),
                    event_type: "provider.request.started".to_string(),
                    timestamp: "2026-07-20T00:00:00Z".to_string(),
                    payload: json!({ "phase": "streaming" }),
                },
                completion_tx: None,
            },
        )
        .await
        .expect("enqueue deferred live event");

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": app_server_protocol::protocol::v2::METHOD_THREAD_RESUME,
            "params": { "threadId": thread_id }
        }),
    )
    .await;

    let _resume_response = tokio::select! {
        response = next_response(&mut output_lines, resume_request_id) => response,
        result = &mut runner => {
            panic!("JSONL runner exited before resume response: {result:?}");
        }
    };
    let JsonRpcMessage::Notification(goal_snapshot) = next_message(&mut output_lines).await else {
        panic!("expected goal snapshot after resume response");
    };
    assert_eq!(
        goal_snapshot.method,
        app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_CLEARED
    );
    let JsonRpcMessage::Request(replayed_request) = next_message(&mut output_lines).await else {
        panic!("expected pending server request replay after resume response");
    };
    assert_eq!(replayed_request.id, pending_request_id);
    assert_eq!(replayed_request.method, "item/tool/requestUserInput");

    // The transport-owned resume barrier has completed; release the test hold
    // so the actor can flush the event that was queued before the request.
    let (completion_tx, completion_rx) = oneshot::channel();
    bridge
        .send_thread_command(
            agent_protocol::ThreadId::new(thread_id.clone()),
            thread_state::ThreadListenerCommand::CompleteResume {
                barrier: hold_barrier,
                connection_id,
                messages: Vec::new(),
                subscribe: false,
                completion_tx,
            },
        )
        .await
        .expect("release test resume barrier");
    completion_rx
        .await
        .expect("test resume barrier completion")
        .expect("flush deferred live event");

    let JsonRpcMessage::Notification(live_notification) = next_message(&mut output_lines).await
    else {
        panic!("expected deferred live event after pending request replay");
    };
    assert_eq!(live_notification.method, METHOD_AGENT_SESSION_EVENT);
    assert_eq!(
        live_notification
            .params
            .as_ref()
            .and_then(|params| params.pointer("/event/type")),
        Some(&json!("provider.request.started"))
    );

    write_message(
        &mut input_client,
        serde_json::to_value(JsonRpcMessage::Response(
            app_server_protocol::JsonRpcResponse::new(
                replayed_request.id,
                json!({ "answer": "yes" }),
            )
            .expect("server request response"),
        ))
        .expect("serialize server request response"),
    )
    .await;
    assert_eq!(
        pending.wait().await.expect("pending request result"),
        json!({ "answer": "yes" })
    );

    drop(input_client);
    drop(output_lines);
    timeout(Duration::from_secs(2), runner)
        .await
        .expect("jsonl runner timeout")
        .expect("jsonl runner task")
        .expect("jsonl runner result");
}

#[tokio::test]
async fn reconnect_claims_thread_request_and_orders_resume_replay_before_live_event() {
    let temp = tempfile::TempDir::new().expect("thread reconnect fixture temp dir");
    let runtime = RuntimeCore::default().with_projection_store(Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("thread reconnect projection store"),
    ));
    let server = AppServer::with_runtime(runtime);
    let (transport_tx, transport_rx) = mpsc::channel(OUTBOUND_MESSAGE_CAPACITY);
    let shutdown = tokio_util::sync::CancellationToken::new();
    let mut runner = tokio::spawn(run_transport_events(
        server.clone(),
        transport_rx,
        false,
        shutdown.clone(),
    ));

    let first_connection = ConnectionId(101);
    let mut first_messages = open_initialized_connection(
        &transport_tx,
        first_connection,
        RequestId::Integer(10),
        "thread-reconnect-first",
    )
    .await;
    send_transport_message(
        &transport_tx,
        first_connection,
        json!({
            "jsonrpc": "2.0",
            "id": 11,
            "method": METHOD_THREAD_START,
            "params": {
                "model": "fixture-model",
                "modelProvider": "fixture-provider",
                "cwd": temp.path()
            }
        }),
    )
    .await;
    let start_response = next_queued_response(&mut first_messages, RequestId::Integer(11)).await;
    let thread_id = start_response
        .result
        .pointer("/thread/id")
        .and_then(serde_json::Value::as_str)
        .expect("thread/start thread id")
        .to_string();
    let session_id = start_response
        .result
        .pointer("/thread/sessionId")
        .and_then(serde_json::Value::as_str)
        .expect("thread/start session id")
        .to_string();

    let mut pending = server.server_requests.register_for_owner(
        server_request::ServerRequestOwner::Transport(first_connection),
        "item/tool/requestUserInput",
        Some(json!({ "threadId": thread_id, "question": "continue?" })),
    );
    let pending_request_id = pending.id().clone();
    transport_tx
        .send(TransportEvent::ConnectionClosed {
            connection_id: first_connection,
        })
        .await
        .expect("close first transport connection");

    let second_connection = ConnectionId(202);
    let mut second_messages = open_initialized_connection(
        &transport_tx,
        second_connection,
        RequestId::Integer(20),
        "thread-reconnect-second",
    )
    .await;
    assert!(
        server
            .thread_states
            .contains_thread(&agent_protocol::ThreadId::new(&thread_id))
            .await,
        "idle thread listener must stay loaded during the unload grace period"
    );
    assert_eq!(
        server.server_requests.current_owner(&pending_request_id),
        None,
        "thread-scoped request must detach when its connection closes"
    );

    let hold_barrier = thread_state::ThreadResumeBarrier::new(
        second_connection,
        RequestId::String("test-reconnect-hold-live-event".to_string()),
    );
    let bridge = server.event_bridge();
    bridge
        .prepare_thread_resume(
            agent_protocol::ThreadId::new(thread_id.clone()),
            hold_barrier.clone(),
        )
        .await
        .expect("prepare reconnect hold barrier");
    bridge
        .send_thread_command(
            agent_protocol::ThreadId::new(thread_id.clone()),
            thread_state::ThreadListenerCommand::PublishRuntimeEvent {
                event: AgentEvent {
                    event_id: "event-reconnect-live".to_string(),
                    sequence: 1,
                    session_id,
                    thread_id: Some(thread_id.clone()),
                    turn_id: Some("turn-reconnect-live".to_string()),
                    event_type: "provider.request.started".to_string(),
                    timestamp: "2026-07-20T00:00:00Z".to_string(),
                    payload: json!({ "phase": "streaming" }),
                },
                completion_tx: None,
            },
        )
        .await
        .expect("enqueue reconnect live event");

    send_transport_message(
        &transport_tx,
        second_connection,
        json!({
            "jsonrpc": "2.0",
            "id": 21,
            "method": app_server_protocol::protocol::v2::METHOD_THREAD_RESUME,
            "params": { "threadId": thread_id }
        }),
    )
    .await;

    let JsonRpcMessage::Response(resume_response) = next_queued_message(&mut second_messages).await
    else {
        panic!("expected resume response before reconnect replay");
    };
    assert_eq!(resume_response.id, RequestId::Integer(21));
    let JsonRpcMessage::Notification(goal_snapshot) =
        next_queued_message(&mut second_messages).await
    else {
        panic!("expected goal snapshot after reconnect response");
    };
    assert_eq!(
        goal_snapshot.method,
        app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_CLEARED
    );
    let JsonRpcMessage::Request(replayed_request) = next_queued_message(&mut second_messages).await
    else {
        panic!("expected pending request replay after reconnect response");
    };
    assert_eq!(replayed_request.id, pending_request_id);
    assert_eq!(replayed_request.method, "item/tool/requestUserInput");
    assert_eq!(
        server.server_requests.current_owner(&pending_request_id),
        Some(server_request::ServerRequestOwner::Transport(
            second_connection
        ))
    );
    assert!(matches!(
        server.server_requests.resolve_transport_response(
            first_connection,
            pending_request_id.clone(),
            json!({ "answer": "stale" }),
        ),
        Err(server_request::ServerRequestError::ClientMismatch { .. })
    ));

    let (completion_tx, completion_rx) = oneshot::channel();
    bridge
        .send_thread_command(
            agent_protocol::ThreadId::new(thread_id),
            thread_state::ThreadListenerCommand::CompleteResume {
                barrier: hold_barrier,
                connection_id: second_connection,
                messages: Vec::new(),
                subscribe: false,
                completion_tx,
            },
        )
        .await
        .expect("release reconnect hold barrier");
    completion_rx
        .await
        .expect("reconnect hold barrier completion")
        .expect("flush reconnect live event");

    let JsonRpcMessage::Notification(live_notification) =
        next_queued_message(&mut second_messages).await
    else {
        panic!("expected live event after reconnect replay");
    };
    assert_eq!(live_notification.method, METHOD_AGENT_SESSION_EVENT);
    assert_eq!(
        live_notification
            .params
            .as_ref()
            .and_then(|params| params.pointer("/event/type")),
        Some(&json!("provider.request.started"))
    );

    send_transport_message(
        &transport_tx,
        second_connection,
        serde_json::to_value(JsonRpcMessage::Response(
            app_server_protocol::JsonRpcResponse::new(
                replayed_request.id,
                json!({ "answer": "yes" }),
            )
            .expect("reconnected server request response"),
        ))
        .expect("serialize reconnected server request response"),
    )
    .await;
    assert_eq!(
        pending
            .wait_terminal()
            .await
            .result
            .expect("pending result"),
        json!({ "answer": "yes" })
    );

    shutdown.cancel();
    drop(transport_tx);
    timeout(Duration::from_secs(2), &mut runner)
        .await
        .expect("transport runner timeout")
        .expect("transport runner task")
        .expect("transport runner result");
}

#[tokio::test]
async fn reconnect_over_real_stdio_jsonl_reclaims_pending_request_and_preserves_order() {
    let temp = tempfile::TempDir::new().expect("raw reconnect fixture temp dir");
    let projection_path = temp.path().join("projection.sqlite");
    let runtime = RuntimeCore::default().with_projection_store(Arc::new(
        ProjectionStore::initialize(&projection_path).expect("raw reconnect projection store"),
    ));
    let server = AppServer::with_runtime(runtime);
    let (transport_tx, transport_rx) = mpsc::channel(OUTBOUND_MESSAGE_CAPACITY);
    let shutdown = tokio_util::sync::CancellationToken::new();
    let mut runner = tokio::spawn(run_transport_events(
        server.clone(),
        transport_rx,
        false,
        shutdown.clone(),
    ));

    let (mut first_input_client, first_input_server) = tokio::io::duplex(32 * 1024);
    let (first_output_server, first_output_client) = tokio::io::duplex(32 * 1024);
    let first_handles = app_server_transport::start_stdio_connection(
        transport_tx.clone(),
        first_input_server,
        first_output_server,
    )
    .await
    .expect("start first stdio connection");
    let mut first_output_lines = BufReader::new(first_output_client).lines();

    write_message(
        &mut first_input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": METHOD_INITIALIZE,
            "params": {
                "clientInfo": {
                    "name": "raw-reconnect-first",
                    "version": "1.0.0"
                }
            }
        }),
    )
    .await;
    next_response(&mut first_output_lines, RequestId::Integer(1)).await;
    write_message(
        &mut first_input_client,
        json!({
            "jsonrpc": "2.0",
            "method": METHOD_INITIALIZED,
            "params": {}
        }),
    )
    .await;
    write_message(
        &mut first_input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": METHOD_THREAD_START,
            "params": {
                "model": "fixture-model",
                "modelProvider": "fixture-provider",
                "cwd": temp.path()
            }
        }),
    )
    .await;
    let start_response = next_response(&mut first_output_lines, RequestId::Integer(2)).await;
    let thread_id = start_response
        .result
        .pointer("/thread/id")
        .and_then(serde_json::Value::as_str)
        .expect("first thread/start thread id")
        .to_string();
    let session_id = start_response
        .result
        .pointer("/thread/sessionId")
        .and_then(serde_json::Value::as_str)
        .expect("first thread/start session id")
        .to_string();
    write_message(
        &mut first_input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_SET,
            "params": {
                "threadId": thread_id.clone(),
                "objective": "survive raw reconnect",
                "status": "active"
            }
        }),
    )
    .await;
    next_response(&mut first_output_lines, RequestId::Integer(3)).await;
    let JsonRpcMessage::Notification(goal_updated) = next_message(&mut first_output_lines).await
    else {
        panic!("expected thread goal update after set");
    };
    assert_eq!(
        goal_updated.method,
        app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_UPDATED
    );
    let goal_update =
        serde_json::from_value::<app_server_protocol::protocol::v2::ThreadGoalUpdatedNotification>(
            goal_updated
                .params
                .clone()
                .expect("thread goal update params"),
        )
        .expect("decode thread goal update");
    let connection =
        rusqlite::Connection::open(&projection_path).expect("open reconnect goal outbox");
    let goal_id = connection
        .query_row(
            "SELECT goal_id FROM thread_goals WHERE thread_id = ?1",
            [&thread_id],
            |row| row.get::<_, String>(0),
        )
        .expect("read reconnect goal id");
    connection
        .execute(
            r#"INSERT INTO thread_goal_update_outbox (
                   thread_id, turn_id, goal_id, source_sequence, notification_json, created_at_ms
               ) VALUES (?1, 'turn-before-reconnect', ?2, 99, ?3, 4)"#,
            rusqlite::params![
                thread_id,
                goal_id,
                serde_json::to_string(&goal_update).expect("serialize reconnect goal update")
            ],
        )
        .expect("seed pending reconnect goal outbox");
    drop(connection);
    let first_connection = wait_for_new_transport_connection(&server, &[]).await;

    let mut pending = server
        .begin_server_request(
            "item/tool/requestUserInput",
            json!({ "threadId": thread_id, "question": "continue?" }),
        )
        .await
        .expect("register reverse request for first connection");
    let pending_request_id = pending.id().clone();
    let JsonRpcMessage::Request(first_reverse_request) =
        next_message(&mut first_output_lines).await
    else {
        panic!("expected reverse request on first connection");
    };
    assert_eq!(first_reverse_request.id, pending_request_id);
    assert_eq!(first_reverse_request.method, "item/tool/requestUserInput");

    drop(first_input_client);
    drop(first_output_lines);
    wait_for_detached_transport_connection(&server, first_connection, &pending_request_id).await;
    assert!(matches!(
        server.server_requests.resolve_transport_response(
            first_connection,
            pending_request_id.clone(),
            json!({ "answer": "stale" }),
        ),
        Err(server_request::ServerRequestError::ClientMismatch { .. })
    ));

    let (mut second_input_client, second_input_server) = tokio::io::duplex(32 * 1024);
    let (second_output_server, second_output_client) = tokio::io::duplex(32 * 1024);
    let second_handles = app_server_transport::start_stdio_connection(
        transport_tx.clone(),
        second_input_server,
        second_output_server,
    )
    .await
    .expect("start second stdio connection");
    let mut second_output_lines = BufReader::new(second_output_client).lines();
    let second_connection = wait_for_new_transport_connection(&server, &[first_connection]).await;

    write_message(
        &mut second_input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 20,
            "method": METHOD_INITIALIZE,
            "params": {
                "clientInfo": {
                    "name": "raw-reconnect-second",
                    "version": "1.0.0"
                }
            }
        }),
    )
    .await;
    next_response(&mut second_output_lines, RequestId::Integer(20)).await;
    write_message(
        &mut second_input_client,
        json!({
            "jsonrpc": "2.0",
            "method": METHOD_INITIALIZED,
            "params": {}
        }),
    )
    .await;

    let hold_barrier = thread_state::ThreadResumeBarrier::new(
        second_connection,
        RequestId::String("raw-reconnect-hold-live-event".to_string()),
    );
    let bridge = server.event_bridge();
    bridge
        .prepare_thread_resume(
            agent_protocol::ThreadId::new(thread_id.clone()),
            hold_barrier.clone(),
        )
        .await
        .expect("prepare raw reconnect hold barrier");
    bridge
        .send_thread_command(
            agent_protocol::ThreadId::new(thread_id.clone()),
            thread_state::ThreadListenerCommand::PublishRuntimeEvent {
                event: AgentEvent {
                    event_id: "event-raw-reconnect-live".to_string(),
                    sequence: 1,
                    session_id,
                    thread_id: Some(thread_id.clone()),
                    turn_id: Some("turn-raw-reconnect-live".to_string()),
                    event_type: "provider.request.started".to_string(),
                    timestamp: "2026-07-20T00:00:00Z".to_string(),
                    payload: json!({ "phase": "streaming" }),
                },
                completion_tx: None,
            },
        )
        .await
        .expect("enqueue raw reconnect live event");

    write_message(
        &mut second_input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 21,
            "method": app_server_protocol::protocol::v2::METHOD_THREAD_RESUME,
            "params": { "threadId": thread_id }
        }),
    )
    .await;
    let resume_response = next_response(&mut second_output_lines, RequestId::Integer(21)).await;
    assert_eq!(
        resume_response
            .result
            .pointer("/thread/id")
            .and_then(serde_json::Value::as_str),
        Some(thread_id.as_str())
    );
    let JsonRpcMessage::Notification(goal_snapshot) = next_message(&mut second_output_lines).await
    else {
        panic!("expected goal snapshot after raw resume response");
    };
    assert_eq!(
        goal_snapshot.method,
        app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_UPDATED
    );
    assert_eq!(
        goal_snapshot
            .params
            .as_ref()
            .and_then(|params| params.pointer("/goal/objective")),
        Some(&json!("survive raw reconnect"))
    );
    wait_for_goal_outbox_delivery(&projection_path, 99).await;
    let JsonRpcMessage::Request(replayed_request) = next_message(&mut second_output_lines).await
    else {
        panic!("expected replayed pending request after raw resume response");
    };
    assert_eq!(replayed_request.id, pending_request_id);
    assert_eq!(replayed_request.method, "item/tool/requestUserInput");
    assert_eq!(
        server.server_requests.current_owner(&pending_request_id),
        Some(server_request::ServerRequestOwner::Transport(
            second_connection
        ))
    );
    assert!(matches!(
        server.server_requests.resolve_transport_response(
            first_connection,
            pending_request_id.clone(),
            json!({ "answer": "stale-after-claim" }),
        ),
        Err(server_request::ServerRequestError::ClientMismatch { .. })
    ));

    let (completion_tx, completion_rx) = oneshot::channel();
    bridge
        .send_thread_command(
            agent_protocol::ThreadId::new(thread_id),
            thread_state::ThreadListenerCommand::CompleteResume {
                barrier: hold_barrier,
                connection_id: second_connection,
                messages: Vec::new(),
                subscribe: false,
                completion_tx,
            },
        )
        .await
        .expect("release raw reconnect hold barrier");
    completion_rx
        .await
        .expect("raw reconnect hold barrier completion")
        .expect("flush raw reconnect live event");
    let JsonRpcMessage::Notification(live_notification) =
        next_message(&mut second_output_lines).await
    else {
        panic!("expected deferred live event after raw replay");
    };
    assert_eq!(live_notification.method, METHOD_AGENT_SESSION_EVENT);
    assert_eq!(
        live_notification
            .params
            .as_ref()
            .and_then(|params| params.pointer("/event/type")),
        Some(&json!("provider.request.started"))
    );

    write_message(
        &mut second_input_client,
        serde_json::to_value(JsonRpcMessage::Response(
            app_server_protocol::JsonRpcResponse::new(
                replayed_request.id,
                json!({ "answer": "yes" }),
            )
            .expect("raw reconnect response"),
        ))
        .expect("serialize raw reconnect response"),
    )
    .await;
    let terminal = pending.wait_terminal().await;
    assert_eq!(
        terminal.owner,
        Some(server_request::ServerRequestOwner::Transport(
            second_connection
        ))
    );
    assert_eq!(
        terminal.result.expect("raw reconnect result"),
        json!({ "answer": "yes" })
    );

    shutdown.cancel();
    drop(second_input_client);
    drop(second_output_lines);
    drop(transport_tx);
    timeout(Duration::from_secs(2), &mut runner)
        .await
        .expect("raw reconnect runner timeout")
        .expect("raw reconnect runner task")
        .expect("raw reconnect runner result");
    for handle in first_handles.into_iter().chain(second_handles) {
        handle.abort();
    }
}

async fn wait_for_new_transport_connection(
    server: &AppServer,
    excluded: &[ConnectionId],
) -> ConnectionId {
    timeout(Duration::from_secs(2), async {
        loop {
            let connection_id = server
                .transport_writers
                .lock()
                .expect("transport writer mutex")
                .keys()
                .copied()
                .find(|connection_id| !excluded.contains(connection_id));
            if let Some(connection_id) = connection_id {
                return connection_id;
            }
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("transport connection open timeout")
}

async fn wait_for_detached_transport_connection(
    server: &AppServer,
    connection_id: ConnectionId,
    pending_request_id: &RequestId,
) {
    timeout(Duration::from_secs(2), async {
        loop {
            let writer_gone = !server
                .transport_writers
                .lock()
                .expect("transport writer mutex")
                .contains_key(&connection_id);
            let owner_detached = server
                .server_requests
                .current_owner(pending_request_id)
                .is_none();
            if writer_gone && owner_detached {
                return;
            }
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("transport connection detach timeout");
}

async fn wait_for_goal_outbox_delivery(database_path: &std::path::Path, source_sequence: i64) {
    timeout(Duration::from_secs(2), async {
        loop {
            let delivered = {
                let connection = rusqlite::Connection::open(database_path)
                    .expect("open goal outbox delivery store");
                connection
                    .query_row(
                        r#"SELECT EXISTS(
                               SELECT 1 FROM thread_goal_update_outbox
                               WHERE source_sequence = ?1 AND delivered_at_ms IS NOT NULL
                           )"#,
                        [source_sequence],
                        |row| row.get::<_, bool>(0),
                    )
                    .expect("read goal outbox delivery")
            };
            if delivered {
                return;
            }
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("goal outbox delivery timeout");
}

async fn write_message(writer: &mut tokio::io::DuplexStream, message: serde_json::Value) {
    writer
        .write_all(format!("{message}\n").as_bytes())
        .await
        .expect("write JSONL message");
}

async fn next_message(lines: &mut Lines<BufReader<tokio::io::DuplexStream>>) -> JsonRpcMessage {
    let line = timeout(Duration::from_secs(2), lines.next_line())
        .await
        .expect("JSONL message timeout")
        .expect("read JSONL message")
        .expect("JSONL stream closed");
    serde_json::from_str(&line).expect("decode JSONL message")
}

async fn next_response(
    lines: &mut Lines<BufReader<tokio::io::DuplexStream>>,
    expected_id: RequestId,
) -> app_server_protocol::JsonRpcResponse {
    loop {
        let line = timeout(Duration::from_secs(2), lines.next_line())
            .await
            .unwrap_or_else(|_| panic!("JSONL response timeout for request id {expected_id:?}"))
            .expect("read JSONL response")
            .expect("JSONL stream closed");
        let message = serde_json::from_str::<JsonRpcMessage>(&line).expect("decode JSONL message");
        match message {
            JsonRpcMessage::Response(response) if response.id == expected_id => return response,
            JsonRpcMessage::Error(response) if response.id == expected_id => {
                panic!(
                    "JSON-RPC request {expected_id} failed: code={} message={}",
                    response.error.code, response.error.message
                );
            }
            _ => {}
        }
    }
}

async fn open_initialized_connection(
    transport_tx: &mpsc::Sender<TransportEvent>,
    connection_id: ConnectionId,
    initialize_request_id: RequestId,
    client_name: &str,
) -> mpsc::Receiver<QueuedOutgoingMessage> {
    let (writer, mut messages) = mpsc::channel(32);
    transport_tx
        .send(TransportEvent::ConnectionOpened {
            connection_id,
            origin: app_server_transport::ConnectionOrigin::InProcess,
            writer,
            disconnect_sender: None,
        })
        .await
        .expect("open transport connection");
    send_transport_message(
        transport_tx,
        connection_id,
        json!({
            "jsonrpc": "2.0",
            "id": initialize_request_id,
            "method": METHOD_INITIALIZE,
            "params": {
                "clientInfo": {
                    "name": client_name,
                    "version": "1.0.0"
                }
            }
        }),
    )
    .await;
    next_queued_response(&mut messages, initialize_request_id).await;
    send_transport_message(
        transport_tx,
        connection_id,
        json!({
            "jsonrpc": "2.0",
            "method": METHOD_INITIALIZED,
            "params": {}
        }),
    )
    .await;
    messages
}

async fn send_transport_message(
    transport_tx: &mpsc::Sender<TransportEvent>,
    connection_id: ConnectionId,
    message: serde_json::Value,
) {
    transport_tx
        .send(TransportEvent::IncomingMessage {
            connection_id,
            message: serde_json::from_value(message).expect("decode transport test message"),
        })
        .await
        .expect("send transport test message");
}

async fn next_queued_message(
    messages: &mut mpsc::Receiver<QueuedOutgoingMessage>,
) -> JsonRpcMessage {
    timeout(Duration::from_secs(2), messages.recv())
        .await
        .expect("queued transport message timeout")
        .expect("queued transport writer closed")
        .message
        .into_json_rpc_message()
}

async fn next_queued_response(
    messages: &mut mpsc::Receiver<QueuedOutgoingMessage>,
    expected_id: RequestId,
) -> app_server_protocol::JsonRpcResponse {
    loop {
        match next_queued_message(messages).await {
            JsonRpcMessage::Response(response) if response.id == expected_id => return response,
            JsonRpcMessage::Error(response) if response.id == expected_id => {
                panic!(
                    "queued JSON-RPC request {expected_id} failed: code={} message={}",
                    response.error.code, response.error.message
                );
            }
            _ => {}
        }
    }
}
