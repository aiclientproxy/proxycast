use super::GrpcCodeModeHost;
use code_mode_protocol::grpc as proto;
use code_mode_protocol::grpc::code_mode_host_server::CodeModeHost;
use tokio_stream::StreamExt;
use tonic::{Code, Request};

fn id() -> String {
    uuid::Uuid::new_v4().to_string()
}

async fn open_session(host: &GrpcCodeModeHost) -> (String, super::EventStream) {
    let mut stream = host
        .open_session(Request::new(proto::OpenSessionRequest {
            cell_execution_limits: None,
        }))
        .await
        .expect("open code-mode session")
        .into_inner();
    let event = stream
        .next()
        .await
        .expect("session opening event")
        .expect("session event");
    let Some(proto::session_event::Event::Opened(opened)) = event.event else {
        panic!("expected session opened event");
    };
    (opened.session_id, stream)
}

fn execute_request(session_id: &str, execution_id: &str, source: &str) -> proto::ExecuteRequest {
    proto::ExecuteRequest {
        session_id: session_id.to_string(),
        execution_id: execution_id.to_string(),
        tool_call_id: "outer-call".to_string(),
        source: source.to_string(),
        enabled_tools: Vec::new(),
        yield_time_ms: Some(60_000),
        max_output_tokens: None,
    }
}

async fn execute_started(
    host: &GrpcCodeModeHost,
    request: proto::ExecuteRequest,
) -> (String, super::ExecuteStream) {
    let mut stream = host
        .execute(Request::new(request))
        .await
        .expect("execute code-mode cell")
        .into_inner();
    let event = stream
        .next()
        .await
        .expect("execution start event")
        .expect("execution event");
    let Some(proto::execute_event::Event::Started(started)) = event.event else {
        panic!("expected execution started event");
    };
    (started.cell_id, stream)
}

fn tool(name: &str) -> proto::ToolDefinition {
    proto::ToolDefinition {
        name: name.to_string(),
        tool_name: Some(proto::ToolName {
            name: name.to_string(),
            namespace: None,
        }),
        description: String::new(),
        kind: proto::ToolKind::Function as i32,
        input_schema_json: None,
        output_schema_json: None,
    }
}

#[tokio::test]
async fn execute_wait_and_cell_closed_preserve_identity_and_tool_sequence() {
    let host = GrpcCodeModeHost::new();
    let (session_id, mut session_events) = open_session(&host).await;
    let execution_id = id();
    let mut subscription = host
        .subscribe_to_tool_calls(Request::new(proto::SubscribeToToolCallsRequest {
            session_id: session_id.clone(),
            tool_names: vec![proto::ToolName {
                name: "echo".to_string(),
                namespace: None,
            }],
        }))
        .await
        .expect("subscribe to tool calls")
        .into_inner();
    let mut request = execute_request(
        &session_id,
        &execution_id,
        r#"text(await tools.echo({value: 1})); text(await tools.echo({value: 2}));"#,
    );
    request.enabled_tools = vec![tool("echo")];
    let (cell_id, mut execution) = execute_started(&host, request).await;

    for sequence in [1, 2] {
        let invocation = subscription
            .next()
            .await
            .expect("tool invocation")
            .expect("tool call stream item");
        assert_eq!(invocation.session_id, session_id);
        assert_eq!(invocation.execution_id, execution_id);
        assert_eq!(invocation.cell_id, cell_id);
        assert_eq!(invocation.sequence, sequence);
        host.complete_tool_call(Request::new(proto::CompleteToolCallRequest {
            session_id: invocation.session_id,
            invocation_id: invocation.invocation_id,
            outcome: Some(proto::complete_tool_call_request::Outcome::Succeeded(
                proto::ToolCallSucceeded {
                    output_json: format!(r#""result-{sequence}""#).into_bytes(),
                },
            )),
        }))
        .await
        .expect("complete tool invocation");
    }

    let outcome = execution
        .next()
        .await
        .expect("execution outcome")
        .expect("execution event");
    assert!(matches!(
        outcome.event,
        Some(proto::execute_event::Event::Outcome(
            proto::ExecutionOutcome {
                outcome: Some(proto::execution_outcome::Outcome::Completed(_)),
                ..
            }
        ))
    ));
    let closed = session_events
        .next()
        .await
        .expect("cell closed event")
        .expect("session event");
    assert_eq!(
        closed.event,
        Some(proto::session_event::Event::CellClosed(proto::CellClosed {
            execution_id,
            cell_id,
            final_tool_call_sequence: 2,
        }))
    );
}

#[tokio::test]
async fn cancel_wait_before_admission_is_preserved() {
    let host = GrpcCodeModeHost::new();
    let (session_id, _events) = open_session(&host).await;
    let execution_id = id();
    let mut request = execute_request(&session_id, &execution_id, "await new Promise(() => {});");
    request.yield_time_ms = Some(1);
    let (cell_id, mut execution) = execute_started(&host, request).await;
    execution
        .next()
        .await
        .expect("initial outcome")
        .expect("event");

    let wait_id = id();
    host.cancel_wait(Request::new(proto::CancelWaitRequest {
        session_id: session_id.clone(),
        wait_id: wait_id.clone(),
    }))
    .await
    .expect("record pre-cancelled wait");
    let error = host
        .wait(Request::new(proto::WaitRequest {
            session_id: session_id.clone(),
            cell_id: cell_id.clone(),
            wait_id,
            yield_time_ms: 60_000,
        }))
        .await
        .expect_err("pre-cancelled wait should fail closed");
    assert_eq!(error.code(), Code::Cancelled);
    host.terminate(Request::new(proto::TerminateRequest {
        session_id,
        cell_id,
    }))
    .await
    .expect("terminate pending cell");
}

#[tokio::test]
async fn close_session_cancels_in_flight_wait() {
    let host = GrpcCodeModeHost::new();
    let (session_id, mut events) = open_session(&host).await;
    let mut request = execute_request(&session_id, &id(), "await new Promise(() => {});");
    request.yield_time_ms = Some(1);
    let (cell_id, mut execution) = execute_started(&host, request).await;
    execution
        .next()
        .await
        .expect("initial outcome")
        .expect("event");
    let wait_id = id();
    let wait_host = host.clone();
    let wait_session_id = session_id.clone();
    let wait_cell_id = cell_id.clone();
    let wait = tokio::spawn(async move {
        wait_host
            .wait(Request::new(proto::WaitRequest {
                session_id: wait_session_id,
                cell_id: wait_cell_id,
                wait_id,
                yield_time_ms: 60_000,
            }))
            .await
    });
    tokio::task::yield_now().await;
    host.close_session(Request::new(proto::CloseSessionRequest { session_id }))
        .await
        .expect("close code-mode session");
    let error = wait
        .await
        .expect("wait task")
        .expect_err("in-flight wait should be cancelled");
    assert_eq!(error.code(), Code::Cancelled);
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(1), events.next())
            .await
            .expect("close should finish the event stream")
            .is_none()
    );
}

#[tokio::test]
async fn dropping_session_event_stream_releases_the_session_lease() {
    let host = GrpcCodeModeHost::new();
    let (session_id, events) = open_session(&host).await;
    drop(events);

    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            if host.session(&session_id).await.is_err() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropping event stream should release the session lease");
}

#[tokio::test]
async fn execution_id_cannot_be_reused_after_admission() {
    let host = GrpcCodeModeHost::new();
    let (session_id, _) = open_session(&host).await;
    let execution_id = id();
    let request = execute_request(&session_id, &execution_id, "await new Promise(() => {});");
    let (cell_id, _execution) = execute_started(&host, request.clone()).await;
    let error = host
        .execute(Request::new(request))
        .await
        .expect_err("execution IDs must not be reused");
    assert_eq!(error.code(), Code::AlreadyExists);
    host.terminate(Request::new(proto::TerminateRequest {
        session_id,
        cell_id,
    }))
    .await
    .expect("terminate admitted execution");
}

#[tokio::test]
async fn heap_limit_is_rejected_at_session_admission() {
    let host = GrpcCodeModeHost::new();
    let error = host
        .open_session(Request::new(proto::OpenSessionRequest {
            cell_execution_limits: Some(proto::SessionCellExecutionLimits {
                max_yield_time_ms: None,
                max_heap_size_bytes: Some(1),
            }),
        }))
        .await
        .expect_err("in-process V8 must reject heap limits");
    assert_eq!(error.code(), Code::FailedPrecondition);
}

#[tokio::test]
async fn request_admission_fails_closed_when_global_limit_is_saturated() {
    let host = GrpcCodeModeHost::new();
    let permits = (0..super::MAX_IN_FLIGHT_REQUESTS)
        .map(|_| host.limits.request_permit().expect("request permit"))
        .collect::<Vec<_>>();
    let result = host
        .open_session(Request::new(proto::OpenSessionRequest {
            cell_execution_limits: None,
        }))
        .await;
    assert!(matches!(result, Err(status) if status.code() == Code::ResourceExhausted));
    drop(permits);
}

#[tokio::test]
async fn active_cell_permit_is_released_after_cell_closed() {
    let host = GrpcCodeModeHost::new();
    let permits = (0..super::MAX_ACTIVE_CELLS - 1)
        .map(|_| {
            host.limits
                .active_cell_permit()
                .expect("active cell permit")
        })
        .collect::<Vec<_>>();
    let (session_id, mut session_events) = open_session(&host).await;
    let mut request = execute_request(&session_id, &id(), "await new Promise(() => {});");
    request.yield_time_ms = Some(1);
    let (cell_id, mut execution) = execute_started(&host, request).await;
    execution
        .next()
        .await
        .expect("initial outcome")
        .expect("event");
    assert!(host.limits.active_cell_permit().is_err());

    host.terminate(Request::new(proto::TerminateRequest {
        session_id,
        cell_id,
    }))
    .await
    .expect("terminate active cell");
    session_events
        .next()
        .await
        .expect("cell closed event")
        .expect("session event");
    assert!(host.limits.active_cell_permit().is_ok());
    drop(permits);
}

#[tokio::test]
async fn notification_reports_the_owning_execution_identity() {
    let host = GrpcCodeModeHost::new();
    let (session_id, mut session_events) = open_session(&host).await;
    let execution_id = id();
    let (cell_id, mut execution) = execute_started(
        &host,
        execute_request(
            &session_id,
            &execution_id,
            "notify('notice'); text('done');",
        ),
    )
    .await;
    let notification = loop {
        let event = session_events
            .next()
            .await
            .expect("notification event")
            .expect("session event");
        if let Some(proto::session_event::Event::Notification(notification)) = event.event {
            break notification;
        }
    };
    assert_eq!(notification.execution_id, execution_id);
    assert_eq!(notification.cell_id, cell_id);
    host.acknowledge_notification(Request::new(proto::AcknowledgeNotificationRequest {
        session_id,
        notification_id: notification.notification_id,
    }))
    .await
    .expect("acknowledge notification");
    execution
        .next()
        .await
        .expect("execution outcome")
        .expect("event");
}

#[tokio::test]
async fn terminate_cancels_an_unacknowledged_notification() {
    let host = GrpcCodeModeHost::new();
    let (session_id, mut session_events) = open_session(&host).await;
    let (cell_id, mut execution) = execute_started(
        &host,
        execute_request(
            &session_id,
            &id(),
            "notify('pending'); await new Promise(() => {});",
        ),
    )
    .await;
    let notification_id = loop {
        let event = session_events
            .next()
            .await
            .expect("notification event")
            .expect("session event");
        if let Some(proto::session_event::Event::Notification(notification)) = event.event {
            break notification.notification_id;
        }
    };
    host.terminate(Request::new(proto::TerminateRequest {
        session_id,
        cell_id,
    }))
    .await
    .expect("terminate pending notification");
    let cancelled = loop {
        let event = session_events
            .next()
            .await
            .expect("notification cancellation event")
            .expect("session event");
        if let Some(proto::session_event::Event::NotificationCancelled(cancelled)) = event.event {
            break cancelled;
        }
    };
    assert_eq!(cancelled.notification_id, notification_id);
    execution
        .next()
        .await
        .expect("execution outcome")
        .expect("event");
}
