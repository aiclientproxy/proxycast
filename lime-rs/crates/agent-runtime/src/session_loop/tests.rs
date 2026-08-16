use super::input_queue::{PendingInputQueue, RuntimeSessionTaskState};
use super::resources::RuntimeSessionResources;
use super::*;
use crate::reply_input::RuntimeReplyInput;
use futures::future::BoxFuture;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{oneshot, Mutex, Notify};
use tokio::time::{sleep, Duration};
use tokio_util::sync::CancellationToken;

fn task(
    turn_id: &str,
    started: Arc<AtomicUsize>,
    finished: Arc<AtomicUsize>,
) -> Arc<dyn RuntimeSessionTask> {
    let turn_id = turn_id.to_string();
    Arc::new(RuntimeSessionClosureTask::new(
        turn_id,
        Vec::new(),
        move |_context, _input, cancellation_token| {
            let started = Arc::clone(&started);
            let finished = Arc::clone(&finished);
            Box::pin(async move {
                started.fetch_add(1, Ordering::SeqCst);
                while !cancellation_token.is_cancelled() {
                    sleep(Duration::from_millis(1)).await;
                    if started.load(Ordering::SeqCst) > 1 {
                        break;
                    }
                }
                finished.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        },
    ))
}

fn inter_agent_input(message_id: &str, content: &str) -> RuntimeSessionInterAgentInput {
    RuntimeSessionInterAgentInput {
        message_id: message_id.to_string(),
        root_thread_id: "thread-root".to_string(),
        sender_thread_id: "thread-sender".to_string(),
        recipient_thread_id: "thread-recipient".to_string(),
        content: content.to_string(),
        kind: RuntimeSessionInterAgentMessageKind::Message,
        source_turn_id: Some("turn-source".to_string()),
        result_status: None,
        delivery_mode: RuntimeSessionInterAgentDeliveryMode::QueueOnly,
    }
}

async fn actor(registry: &RuntimeSessionRegistry, session_id: &str) -> RuntimeSessionHandle {
    let thread_id = format!("thread-{session_id}");
    registry
        .get_or_create(session_id, &thread_id)
        .await
        .expect("bind session actor identity")
}

#[tokio::test]
async fn session_snapshot_is_actor_ordered_and_reports_only_the_live_turn() {
    let registry = RuntimeSessionRegistry::default();
    assert_eq!(
        registry
            .snapshot("session-snapshot")
            .await
            .expect("missing session snapshot"),
        None
    );

    let session = actor(&registry, "session-snapshot").await;
    let started = Arc::new(AtomicUsize::new(0));
    let finished = Arc::new(AtomicUsize::new(0));
    let submission = session
        .submit(
            task(
                "turn-snapshot-live",
                Arc::clone(&started),
                Arc::clone(&finished),
            ),
            false,
        )
        .await
        .expect("snapshot task submission");

    assert_eq!(
        registry
            .snapshot("session-snapshot")
            .await
            .expect("active session snapshot")
            .expect("loaded session snapshot")
            .active_turn_id
            .as_deref(),
        Some("turn-snapshot-live")
    );

    session.interrupt().await.expect("interrupt snapshot task");
    assert_eq!(
        submission.completion.await.expect("snapshot completion"),
        Ok(RuntimeSessionTaskOutcome::Interrupted)
    );
    assert_eq!(
        registry
            .snapshot("session-snapshot")
            .await
            .expect("idle session snapshot")
            .expect("loaded idle snapshot")
            .active_turn_id,
        None
    );
}

#[tokio::test]
async fn session_loop_serializes_and_queues_tasks() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-1").await;
    let started = Arc::new(AtomicUsize::new(0));
    let finished = Arc::new(AtomicUsize::new(0));
    let first = session
        .submit(
            task("turn-1", Arc::clone(&started), Arc::clone(&finished)),
            false,
        )
        .await
        .expect("first submission");
    let second = session
        .submit(
            task("turn-2", Arc::clone(&started), Arc::clone(&finished)),
            true,
        )
        .await
        .expect("queued submission");
    assert_eq!(first.result, RuntimeSessionSubmitResult::Started);
    assert_eq!(
        second.result,
        RuntimeSessionSubmitResult::Queued { position: 1 }
    );
    session.interrupt().await.expect("interrupt");
    assert_eq!(
        first.completion.await.expect("first completion"),
        Ok(RuntimeSessionTaskOutcome::Interrupted)
    );
    assert_eq!(
        second.completion.await.expect("second completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    assert_eq!(started.load(Ordering::SeqCst), 2);
    assert_eq!(finished.load(Ordering::SeqCst), 2);
    registry.shutdown("session-1").await.expect("shutdown");
}

#[tokio::test]
async fn operation_submission_preserves_identity_and_trace_metadata() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-operation-envelope").await;
    let context_metadata = Arc::new(Mutex::new(
        Option::<(String, Option<String>, Option<RuntimeSessionTraceContext>)>::None,
    ));
    let context_metadata_for_task = Arc::clone(&context_metadata);
    let task = RuntimeSessionClosureTask::new(
        "turn-operation-envelope",
        Vec::new(),
        move |context, _input, _cancellation_token| {
            let context_metadata = Arc::clone(&context_metadata_for_task);
            Box::pin(async move {
                *context_metadata.lock().await = Some((
                    context.submission_id().to_string(),
                    context.client_user_message_id().map(str::to_string),
                    context.trace().cloned(),
                ));
                Ok(())
            })
        },
    );
    let trace = RuntimeSessionTraceContext {
        traceparent: Some("00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01".to_string()),
        tracestate: Some("vendor=value".to_string()),
    };
    let envelope = RuntimeSessionOperationSubmission::with_metadata(
        RuntimeSessionOperation::StartTask {
            task: Arc::new(task),
            queue_if_busy: false,
            replace_active: false,
        },
        Some("client-message-1".to_string()),
        Some(trace.clone()),
    );
    let submission_id = envelope.id.clone();
    let result = session.dispatch(envelope).await.expect("dispatch");
    let RuntimeSessionOperationResult::Submission(submission) = result else {
        panic!("expected submit receipt");
    };
    let expected_context = (
        submission_id.clone(),
        Some("client-message-1".to_string()),
        Some(trace.clone()),
    );
    assert_eq!(submission.id, expected_context.0);
    assert_eq!(
        submission.client_user_message_id.as_deref(),
        Some("client-message-1")
    );
    assert_eq!(submission.trace, expected_context.2);
    let completion = submission.completion;
    assert_eq!(
        completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    assert_eq!(*context_metadata.lock().await, Some(expected_context));
    registry
        .shutdown("session-operation-envelope")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn replacing_task_operations_share_the_session_dispatcher() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-typed-task-operations").await;
    let operations = [
        RuntimeSessionTaskKind::Review,
        RuntimeSessionTaskKind::Compact,
    ];

    for (index, kind) in operations.into_iter().enumerate() {
        let task: Arc<dyn RuntimeSessionTask> = Arc::new(
            RuntimeSessionClosureTask::new(
                format!("turn-typed-operation-{index}"),
                Vec::new(),
                |_context, _input, _cancel| Box::pin(async { Ok(()) }),
            )
            .with_kind(kind),
        );
        let operation = match kind {
            RuntimeSessionTaskKind::Review => RuntimeSessionOperation::Review { task },
            RuntimeSessionTaskKind::Compact => RuntimeSessionOperation::Compact { task },
            RuntimeSessionTaskKind::Regular | RuntimeSessionTaskKind::RunShell => unreachable!(),
        };
        let RuntimeSessionOperationResult::Submission(submission) = session
            .dispatch(RuntimeSessionOperationSubmission::new(operation))
            .await
            .expect("typed task dispatch")
        else {
            panic!("typed task operation must return a submission receipt");
        };
        assert_eq!(submission.result, RuntimeSessionSubmitResult::Started);
        assert_eq!(
            submission.completion.await.expect("completion"),
            Ok(RuntimeSessionTaskOutcome::Completed)
        );
    }

    registry
        .shutdown("session-typed-task-operations")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn inline_operations_run_in_submission_order_without_replacing_the_active_task() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-inline-operations").await;
    let active = RuntimeSessionClosureTask::new(
        "turn-inline-active",
        Vec::new(),
        |_context, _input, cancellation_token| {
            Box::pin(async move {
                cancellation_token.cancelled().await;
                Ok(())
            })
        },
    );
    let active_submission = session
        .submit(Arc::new(active), false)
        .await
        .expect("active task");
    let seen = Arc::new(Mutex::new(Vec::new()));
    let handler = |name: &'static str| {
        let seen = Arc::clone(&seen);
        RuntimeSessionHandler::new(move |context| {
            let seen = Arc::clone(&seen);
            Box::pin(async move {
                seen.lock().await.push((name, context));
                Ok(())
            })
        })
    };
    let operations = vec![
        RuntimeSessionOperation::ThreadSettings {
            handler: handler("settings"),
        },
        RuntimeSessionOperation::SetMemoryMode {
            handler: handler("memory"),
        },
        RuntimeSessionOperation::RefreshMcp {
            handler: handler("mcp"),
        },
        RuntimeSessionOperation::ReloadConfig {
            handler: handler("config"),
        },
    ];

    let trace = RuntimeSessionTraceContext {
        traceparent: Some("00-cccccccccccccccccccccccccccccccc-dddddddddddddddd-01".to_string()),
        tracestate: Some("runtime=inline".to_string()),
    };
    let mut submission_ids = Vec::new();
    for operation in operations {
        let envelope = RuntimeSessionOperationSubmission::with_metadata(
            operation,
            Some("inline-client-message".to_string()),
            Some(trace.clone()),
        );
        submission_ids.push(envelope.id.clone());
        assert!(matches!(
            session.dispatch(envelope).await.expect("inline operation"),
            RuntimeSessionOperationResult::Accepted { .. }
        ));
    }
    let seen = seen.lock().await;
    assert_eq!(seen.len(), 4);
    assert_eq!(
        seen.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        vec!["settings", "memory", "mcp", "config"]
    );
    for ((_, context), submission_id) in seen.iter().zip(submission_ids) {
        assert_eq!(context.session_id, "session-inline-operations");
        assert_eq!(context.submission_id, submission_id);
        assert_eq!(
            context.active_turn_id.as_deref(),
            Some("turn-inline-active")
        );
        assert_eq!(
            context.client_user_message_id.as_deref(),
            Some("inline-client-message")
        );
        assert_eq!(context.trace.as_ref(), Some(&trace));
    }
    drop(seen);
    assert!(session.interrupt().await.expect("interrupt active task"));
    assert_eq!(
        active_submission
            .completion
            .await
            .expect("active completion"),
        Ok(RuntimeSessionTaskOutcome::Interrupted)
    );
    registry
        .shutdown("session-inline-operations")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn shell_operation_attaches_to_the_active_turn_and_shares_cancellation() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-shell-active").await;
    let active = RuntimeSessionClosureTask::new(
        "turn-shell-active",
        Vec::new(),
        |_context, _input, cancellation_token| {
            Box::pin(async move {
                cancellation_token.cancelled().await;
                Ok(())
            })
        },
    );
    let active_submission = session
        .submit(Arc::new(active), false)
        .await
        .expect("active task");

    let seen = Arc::new(Mutex::new(None));
    let seen_for_handler = Arc::clone(&seen);
    let auxiliary =
        RuntimeSessionHandler::new_with_cancellation(move |context, cancellation_token| {
            let seen = Arc::clone(&seen_for_handler);
            Box::pin(async move {
                *seen.lock().await = Some((context, cancellation_token));
                Ok(())
            })
        });
    let idle_task_started = Arc::new(AtomicUsize::new(0));
    let idle_task_started_for_task = Arc::clone(&idle_task_started);
    let idle_task = RuntimeSessionClosureTask::new(
        "turn-shell-idle-candidate",
        Vec::new(),
        move |_context, _input, _cancellation_token| {
            let started = Arc::clone(&idle_task_started_for_task);
            Box::pin(async move {
                started.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        },
    )
    .with_kind(RuntimeSessionTaskKind::RunShell);

    let result = session
        .dispatch(RuntimeSessionOperationSubmission::new(
            RuntimeSessionOperation::RunShell {
                auxiliary,
                task: Arc::new(idle_task),
            },
        ))
        .await
        .expect("active shell operation");
    assert!(matches!(
        result,
        RuntimeSessionOperationResult::Accepted {
            turn_id: Some(ref turn_id),
            ..
        } if turn_id == "turn-shell-active"
    ));
    assert_eq!(idle_task_started.load(Ordering::SeqCst), 0);

    let (context, cancellation_token) = seen.lock().await.clone().expect("auxiliary context");
    assert_eq!(context.active_turn_id.as_deref(), Some("turn-shell-active"));
    assert!(!cancellation_token.is_cancelled());

    assert!(session.interrupt().await.expect("interrupt active task"));
    assert_eq!(
        active_submission
            .completion
            .await
            .expect("active completion"),
        Ok(RuntimeSessionTaskOutcome::Interrupted)
    );
    assert!(cancellation_token.is_cancelled());
    registry
        .shutdown("session-shell-active")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn idle_shell_operation_owns_the_session_task_lifecycle() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-shell-idle").await;
    let auxiliary_calls = Arc::new(AtomicUsize::new(0));
    let auxiliary_calls_for_handler = Arc::clone(&auxiliary_calls);
    let auxiliary = RuntimeSessionHandler::new(move |_context| {
        let calls = Arc::clone(&auxiliary_calls_for_handler);
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    });
    let (started_tx, started_rx) = oneshot::channel();
    let started_tx = Arc::new(Mutex::new(Some(started_tx)));
    let release = Arc::new(Notify::new());
    let shell_task = RuntimeSessionClosureTask::new("turn-shell-idle", Vec::new(), {
        let started_tx = Arc::clone(&started_tx);
        let release = Arc::clone(&release);
        move |_context, _input, _cancellation_token| {
            let started_tx = Arc::clone(&started_tx);
            let release = Arc::clone(&release);
            Box::pin(async move {
                if let Some(started_tx) = started_tx.lock().await.take() {
                    let _ = started_tx.send(());
                }
                release.notified().await;
                Ok(())
            })
        }
    })
    .with_kind(RuntimeSessionTaskKind::RunShell);

    let RuntimeSessionOperationResult::Submission(shell_submission) = session
        .dispatch(RuntimeSessionOperationSubmission::new(
            RuntimeSessionOperation::RunShell {
                auxiliary,
                task: Arc::new(shell_task),
            },
        ))
        .await
        .expect("idle shell operation")
    else {
        panic!("idle shell operation must return a submission receipt");
    };
    assert_eq!(shell_submission.result, RuntimeSessionSubmitResult::Started);
    started_rx.await.expect("shell task started");
    assert_eq!(auxiliary_calls.load(Ordering::SeqCst), 0);

    let busy = session
        .submit(
            Arc::new(RuntimeSessionClosureTask::new(
                "turn-after-shell",
                Vec::new(),
                |_context, _input, _cancellation_token| Box::pin(async { Ok(()) }),
            )),
            false,
        )
        .await
        .expect("busy receipt");
    assert_eq!(busy.result, RuntimeSessionSubmitResult::Busy);

    release.notify_one();
    assert_eq!(
        shell_submission.completion.await.expect("shell completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    registry
        .shutdown("session-shell-idle")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn user_input_starts_an_idle_candidate_with_the_submitted_input() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-user-input-start").await;
    let (seen_tx, seen_rx) = oneshot::channel();
    let seen_tx = Arc::new(Mutex::new(Some(seen_tx)));
    let task = RuntimeSessionClosureTask::new(
        "turn-user-input-start",
        Vec::new(),
        move |_context, input, _cancellation_token| {
            let seen_tx = Arc::clone(&seen_tx);
            Box::pin(async move {
                if let Some(sender) = seen_tx.lock().await.take() {
                    let _ = sender.send(input);
                }
                Ok(())
            })
        },
    );

    let result = session
        .submit_user_input_with_metadata(
            Arc::new(task),
            vec![RuntimeSessionInput::User(RuntimeReplyInput::text("start"))],
            false,
            Some("client-user-input-start".to_string()),
            None,
        )
        .await
        .expect("user input submission");
    let RuntimeSessionUserInputResult::Submitted(submission) = result else {
        panic!("expected an idle user input to start its candidate task");
    };
    assert_eq!(submission.result, RuntimeSessionSubmitResult::Started);
    assert_eq!(
        submission.client_user_message_id.as_deref(),
        Some("client-user-input-start")
    );
    assert_eq!(
        submission.completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    let seen = seen_rx.await.expect("submitted input");
    assert_eq!(seen.len(), 1);
    let RuntimeSessionInput::User(seen) = &seen[0] else {
        panic!("expected user input");
    };
    assert_eq!(seen.concat_text(), "start");
    registry
        .shutdown("session-user-input-start")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn user_input_steers_an_active_regular_task_without_starting_the_candidate() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-user-input-steer").await;
    let ready = Arc::new(Notify::new());
    let ready_for_task = Arc::clone(&ready);
    let release = Arc::new(Notify::new());
    let release_for_task = Arc::clone(&release);
    let (seen_tx, seen_rx) = oneshot::channel();
    let seen_tx = Arc::new(Mutex::new(Some(seen_tx)));
    let active = RuntimeSessionClosureTask::new(
        "turn-user-input-active",
        Vec::new(),
        move |context, _input, _cancellation_token| {
            let ready = Arc::clone(&ready_for_task);
            let release = Arc::clone(&release_for_task);
            let seen_tx = Arc::clone(&seen_tx);
            Box::pin(async move {
                ready.notify_one();
                release.notified().await;
                let input = context.take_pending_input(false).await;
                if let Some(sender) = seen_tx.lock().await.take() {
                    let _ = sender.send(input);
                }
                Ok(())
            })
        },
    );
    let active_submission = session
        .submit(Arc::new(active), false)
        .await
        .expect("active submission");
    ready.notified().await;

    let candidate_starts = Arc::new(AtomicUsize::new(0));
    let candidate_starts_for_task = Arc::clone(&candidate_starts);
    let candidate = RuntimeSessionClosureTask::new(
        "turn-user-input-candidate",
        Vec::new(),
        move |_context, _input, _cancellation_token| {
            let candidate_starts = Arc::clone(&candidate_starts_for_task);
            Box::pin(async move {
                candidate_starts.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        },
    );
    let result = session
        .submit_user_input_with_metadata(
            Arc::new(candidate),
            vec![RuntimeSessionInput::User(RuntimeReplyInput::text("steer"))],
            false,
            None,
            None,
        )
        .await
        .expect("user input steer");
    let RuntimeSessionUserInputResult::Steered { turn_id, .. } = result else {
        panic!("expected user input to steer the active task");
    };
    assert_eq!(turn_id, "turn-user-input-active");
    release.notify_one();
    let seen = seen_rx.await.expect("steered input");
    assert_eq!(seen.len(), 1);
    let RuntimeSessionInput::User(seen) = &seen[0] else {
        panic!("expected user input");
    };
    assert_eq!(seen.concat_text(), "steer");
    assert_eq!(
        active_submission
            .completion
            .await
            .expect("active completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    assert_eq!(candidate_starts.load(Ordering::SeqCst), 0);
    registry
        .shutdown("session-user-input-steer")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn promoted_task_context_preserves_submission_metadata() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-promoted-metadata").await;
    let first = RuntimeSessionClosureTask::new(
        "turn-promoted-blocker",
        Vec::new(),
        |_context, _input, cancellation_token| {
            Box::pin(async move {
                cancellation_token.cancelled().await;
                Ok(())
            })
        },
    );
    let (seen_tx, seen_rx) = oneshot::channel();
    let seen_tx = Arc::new(Mutex::new(Some(seen_tx)));
    let second = RuntimeSessionClosureTask::new(
        "turn-promoted-target",
        Vec::new(),
        move |context, _input, _cancellation_token| {
            let seen_tx = Arc::clone(&seen_tx);
            Box::pin(async move {
                let metadata = (
                    context.submission_id().to_string(),
                    context.client_user_message_id().map(str::to_string),
                    context.trace().cloned(),
                );
                if let Some(sender) = seen_tx.lock().await.take() {
                    let _ = sender.send(metadata);
                }
                Ok(())
            })
        },
    );
    let first_submission = session
        .submit(Arc::new(first), false)
        .await
        .expect("blocker submission");
    let trace = RuntimeSessionTraceContext {
        traceparent: Some("00-cccccccccccccccccccccccccccccccccc-dddddddddddddddd-01".to_string()),
        tracestate: Some("queued=true".to_string()),
    };
    let envelope = RuntimeSessionOperationSubmission::with_metadata(
        RuntimeSessionOperation::StartTask {
            task: Arc::new(second),
            queue_if_busy: true,
            replace_active: false,
        },
        Some("queued-client-message".to_string()),
        Some(trace.clone()),
    );
    let submission_id = envelope.id.clone();
    let result = session.dispatch(envelope).await.expect("queued dispatch");
    let RuntimeSessionOperationResult::Submission(second_submission) = result else {
        panic!("expected queued submit receipt");
    };
    assert_eq!(second_submission.id, submission_id);
    assert_eq!(
        second_submission.result,
        RuntimeSessionSubmitResult::Queued { position: 1 }
    );
    assert!(session.interrupt().await.expect("interrupt blocker"));
    assert_eq!(
        first_submission
            .completion
            .await
            .expect("blocker completion"),
        Ok(RuntimeSessionTaskOutcome::Interrupted)
    );
    assert_eq!(
        second_submission
            .completion
            .await
            .expect("promoted completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    let metadata = tokio::time::timeout(Duration::from_secs(1), seen_rx)
        .await
        .expect("metadata deadline")
        .expect("metadata sender");
    assert_eq!(
        metadata,
        (
            submission_id,
            Some("queued-client-message".to_string()),
            Some(trace),
        )
    );
    registry
        .shutdown("session-promoted-metadata")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn shutdown_and_wait_allows_multiple_waiters() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-shutdown-waiters").await;
    let submission = session
        .submit(
            Arc::new(RuntimeSessionClosureTask::new(
                "turn-shutdown-waiters",
                Vec::new(),
                move |_context, _input, cancellation_token| {
                    Box::pin(async move {
                        cancellation_token.cancelled().await;
                        Ok(())
                    })
                },
            )),
            false,
        )
        .await
        .expect("submission");
    let first = session.clone();
    let second = session.clone();
    let (first_result, second_result) =
        tokio::join!(first.shutdown_and_wait(), second.shutdown_and_wait(),);
    first_result.expect("first shutdown waiter");
    second_result.expect("second shutdown waiter");
    assert_eq!(
        submission.completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Shutdown)
    );
    registry
        .shutdown("session-shutdown-waiters")
        .await
        .expect("registry shutdown");
}

#[tokio::test]
async fn interrupt_preserves_session_mailbox_for_the_next_task() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-mailbox-interrupt").await;
    let (ready_tx, ready_rx) = oneshot::channel();
    let ready_tx = Arc::new(Mutex::new(Some(ready_tx)));
    let first = RuntimeSessionClosureTask::new(
        "turn-first",
        Vec::new(),
        move |_context, _input, cancellation_token| {
            let ready_tx = Arc::clone(&ready_tx);
            Box::pin(async move {
                if let Some(ready_tx) = ready_tx.lock().await.take() {
                    let _ = ready_tx.send(());
                }
                while !cancellation_token.is_cancelled() {
                    sleep(Duration::from_millis(1)).await;
                }
                Ok(())
            })
        },
    );
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_for_task = Arc::clone(&seen);
    let second = RuntimeSessionClosureTask::new(
        "turn-second",
        Vec::new(),
        move |context, _input, _cancel| {
            let seen = Arc::clone(&seen_for_task);
            Box::pin(async move {
                seen.lock()
                    .await
                    .extend(context.take_pending_input(true).await);
                Ok(())
            })
        },
    )
    .with_mailbox_loader(|| {
        Box::pin(async {
            Ok(vec![RuntimeSessionInput::InterAgent(inter_agent_input(
                "mailbox-preserve",
                "preserve",
            ))])
        })
    });
    let first_submission = session
        .submit(Arc::new(first), false)
        .await
        .expect("first submission");
    ready_rx.await.expect("first task ready");
    session
        .notify_inter_agent_communication(inter_agent_input("mailbox-preserve", "preserve"))
        .await
        .expect("mailbox");
    let second_submission = session
        .submit(Arc::new(second), true)
        .await
        .expect("second submission");
    assert!(session.interrupt().await.expect("interrupt"));
    assert_eq!(
        first_submission.completion.await.expect("first completion"),
        Ok(RuntimeSessionTaskOutcome::Interrupted)
    );
    assert_eq!(
        second_submission
            .completion
            .await
            .expect("second completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    assert_eq!(seen.lock().await.len(), 1);
    registry
        .shutdown("session-mailbox-interrupt")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn steer_and_mailbox_are_kept_in_separate_queues() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-2").await;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_for_task = Arc::clone(&seen);
    let (ready_sender, ready_receiver) = oneshot::channel();
    let ready_sender = Arc::new(Mutex::new(Some(ready_sender)));
    let task: Arc<dyn RuntimeSessionTask> = Arc::new(
        RuntimeSessionClosureTask::new("turn-1", Vec::new(), move |context, _input, _cancel| {
            let seen = Arc::clone(&seen_for_task);
            let ready_sender = Arc::clone(&ready_sender);
            Box::pin(async move {
                if let Some(sender) = ready_sender.lock().await.take() {
                    let _ = sender.send(());
                }
                context.wait_for_pending_input().await;
                let steer = context.take_pending_input(false).await;
                seen.lock().await.push(steer);
                context.wait_for_pending_input().await;
                let all = context.take_pending_input(true).await;
                seen.lock().await.push(all);
                Ok(())
            })
        })
        .with_mailbox_loader(|| {
            Box::pin(async {
                Ok(vec![RuntimeSessionInput::InterAgent(inter_agent_input(
                    "mailbox-separated",
                    "mail",
                ))])
            })
        }),
    );
    let submission = session.submit(task, false).await.expect("submit");
    ready_receiver.await.expect("task ready");
    session
        .steer_for_turn(
            Some("turn-1"),
            vec![RuntimeSessionInput::User(RuntimeReplyInput::text("steer"))],
        )
        .await
        .expect("steer");
    session
        .notify_inter_agent_communication(inter_agent_input("mailbox-separated", "mail"))
        .await
        .expect("mailbox");
    assert_eq!(
        submission.completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    let seen = seen.lock().await;
    assert_eq!(seen[0].len(), 1);
    assert_eq!(seen[1].len(), 1);
    registry.shutdown("session-2").await.expect("shutdown");
}

#[tokio::test]
async fn mailbox_loader_is_deferred_until_the_mailbox_boundary() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-loader").await;
    let loader_calls = Arc::new(AtomicUsize::new(0));
    let loader_calls_for_loader = Arc::clone(&loader_calls);
    let loader_calls_for_task = Arc::clone(&loader_calls);
    let loader = move || -> BoxFuture<'static, Result<Vec<RuntimeSessionInput>, String>> {
        let loader_calls = Arc::clone(&loader_calls_for_loader);
        Box::pin(async move {
            loader_calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![RuntimeSessionInput::InterAgent(inter_agent_input(
                "mailbox-durable",
                "durable",
            ))])
        })
    };
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_for_task = Arc::clone(&seen);
    let task =
        RuntimeSessionClosureTask::new("turn-1", Vec::new(), move |context, _input, _cancel| {
            let seen = Arc::clone(&seen_for_task);
            let loader_calls = Arc::clone(&loader_calls_for_task);
            Box::pin(async move {
                let input = context.input_handle();
                assert!(input
                    .try_take_pending_input(false)
                    .await
                    .expect("steer-only input")
                    .is_empty());
                assert_eq!(loader_calls.load(Ordering::SeqCst), 0);
                seen.lock().await.extend(
                    input
                        .try_take_pending_input(true)
                        .await
                        .expect("mailbox input"),
                );
                Ok(())
            })
        })
        .with_mailbox_loader(loader);
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("submission");
    assert_eq!(
        submission.completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    assert_eq!(loader_calls.load(Ordering::SeqCst), 1);
    assert_eq!(seen.lock().await.len(), 1);
    registry.shutdown("session-loader").await.expect("shutdown");
}

#[tokio::test]
async fn busy_submission_completes_with_a_failure_instead_of_hanging() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-busy").await;
    let started = Arc::new(AtomicUsize::new(0));
    let finished = Arc::new(AtomicUsize::new(0));
    let first = session
        .submit(
            task("turn-1", Arc::clone(&started), Arc::clone(&finished)),
            false,
        )
        .await
        .expect("first submission");
    let second = session
        .submit(
            task("turn-2", Arc::clone(&started), Arc::clone(&finished)),
            false,
        )
        .await
        .expect("busy submission");

    assert_eq!(second.result, RuntimeSessionSubmitResult::Busy);
    assert_eq!(
        second.completion.await.expect("busy completion channel"),
        Err(RuntimeSessionTaskFailure {
            message: "runtime session is busy".to_string(),
            reason_code: None,
        })
    );
    session.interrupt().await.expect("interrupt");
    assert_eq!(
        first.completion.await.expect("first completion"),
        Ok(RuntimeSessionTaskOutcome::Interrupted)
    );
    registry.shutdown("session-busy").await.expect("shutdown");
}

#[tokio::test]
async fn steer_for_turn_rejects_a_late_target() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-target").await;
    let started = Arc::new(AtomicUsize::new(0));
    let finished = Arc::new(AtomicUsize::new(0));
    let submission = session
        .submit(
            task("turn-1", Arc::clone(&started), Arc::clone(&finished)),
            false,
        )
        .await
        .expect("submission");
    assert!(session
        .steer_for_turn(
            Some("turn-other"),
            vec![RuntimeSessionInput::User(RuntimeReplyInput::text("late"))]
        )
        .await
        .is_err());
    session.interrupt().await.expect("interrupt");
    let _ = submission.completion.await;
    registry.shutdown("session-target").await.expect("shutdown");
}

#[tokio::test]
async fn steer_returns_the_actor_confirmed_active_turn_id() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-steer-id").await;
    let submission = session
        .submit(
            task(
                "turn-steer-id",
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
            ),
            false,
        )
        .await
        .expect("submission");
    let turn_id = session
        .steer_for_turn_id(
            Some("turn-steer-id"),
            vec![RuntimeSessionInput::User(RuntimeReplyInput::text("steer"))],
        )
        .await
        .expect("steer");
    assert_eq!(turn_id, "turn-steer-id");
    session.interrupt().await.expect("interrupt");
    assert_eq!(
        submission.completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Interrupted)
    );
    registry
        .shutdown("session-steer-id")
        .await
        .expect("shutdown");
}

struct HangingAbortTask {
    turn_id: String,
}

impl RuntimeSessionTask for HangingAbortTask {
    fn turn_id(&self) -> &str {
        &self.turn_id
    }

    fn run(
        self: Arc<Self>,
        _context: RuntimeSessionTaskContext,
        _input: Vec<RuntimeSessionInput>,
        _cancellation_token: CancellationToken,
    ) -> BoxFuture<'static, Result<(), RuntimeSessionTaskFailure>> {
        Box::pin(std::future::pending())
    }

    fn abort(&self, _context: RuntimeSessionTaskContext) -> BoxFuture<'static, ()> {
        Box::pin(std::future::pending())
    }
}

#[tokio::test]
async fn hanging_abort_is_forcefully_reaped() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-hanging-abort").await;
    let submission = session
        .submit(
            Arc::new(HangingAbortTask {
                turn_id: "turn-1".to_string(),
            }),
            false,
        )
        .await
        .expect("submission");

    let interrupted = tokio::time::timeout(Duration::from_secs(1), session.interrupt())
        .await
        .expect("interrupt deadline")
        .expect("interrupt result");
    assert!(interrupted);
    assert_eq!(
        submission.completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Interrupted)
    );
    registry
        .shutdown("session-hanging-abort")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn dropping_last_session_handle_shuts_down_active_and_queued_tasks() {
    let session = RuntimeSessionActor::spawn(
        "session-channel-close".to_string(),
        "thread-channel-close".to_string(),
        None,
    );
    let started = Arc::new(AtomicUsize::new(0));
    let finished = Arc::new(AtomicUsize::new(0));
    let active = session
        .submit(
            task("turn-active", Arc::clone(&started), Arc::clone(&finished)),
            false,
        )
        .await
        .expect("active submission");
    let queued = session
        .submit(
            task("turn-queued", Arc::clone(&started), Arc::clone(&finished)),
            true,
        )
        .await
        .expect("queued submission");
    drop(session);

    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), active.completion)
            .await
            .expect("active completion deadline")
            .expect("active completion channel"),
        Ok(RuntimeSessionTaskOutcome::Shutdown)
    );
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), queued.completion)
            .await
            .expect("queued completion deadline")
            .expect("queued completion channel"),
        Ok(RuntimeSessionTaskOutcome::Shutdown)
    );
}

#[tokio::test]
async fn stale_completion_cannot_finish_a_promoted_task() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-stale-completion").await;
    let (ready_tx, ready_rx) = oneshot::channel();
    let ready_tx = Arc::new(Mutex::new(Some(ready_tx)));
    let first = RuntimeSessionClosureTask::new(
        "turn-first",
        Vec::new(),
        move |_context, _input, _cancel| {
            let ready_tx = Arc::clone(&ready_tx);
            Box::pin(async move {
                if let Some(ready_tx) = ready_tx.lock().await.take() {
                    let _ = ready_tx.send(());
                }
                sleep(Duration::from_millis(20)).await;
                Ok(())
            })
        },
    );
    let second =
        RuntimeSessionClosureTask::new("turn-second", Vec::new(), |_context, _input, _cancel| {
            Box::pin(async { Ok(()) })
        });
    let first_submission = session
        .submit(Arc::new(first), false)
        .await
        .expect("first submission");
    ready_rx.await.expect("first task ready");
    let second_submission = session
        .submit(Arc::new(second), true)
        .await
        .expect("second submission");
    assert!(session.interrupt().await.expect("interrupt"));
    assert_eq!(
        first_submission.completion.await.expect("first completion"),
        Ok(RuntimeSessionTaskOutcome::Interrupted)
    );
    assert_eq!(
        second_submission
            .completion
            .await
            .expect("second completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    registry
        .shutdown("session-stale-completion")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn replace_submission_closes_the_old_task_and_rejects_compact_steer() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-replace").await;
    let first = RuntimeSessionClosureTask::new(
        "turn-compact",
        Vec::new(),
        |_context, _input, cancellation_token| {
            Box::pin(async move {
                while !cancellation_token.is_cancelled() {
                    sleep(Duration::from_millis(1)).await;
                }
                Ok(())
            })
        },
    )
    .with_kind(RuntimeSessionTaskKind::Compact);
    let first_submission = session
        .submit(Arc::new(first), false)
        .await
        .expect("compact submission");
    assert!(session
        .steer(vec![RuntimeSessionInput::User(RuntimeReplyInput::text(
            "not accepted",
        ))])
        .await
        .is_err());

    let second = RuntimeSessionClosureTask::new(
        "turn-replacement",
        Vec::new(),
        |_context, _input, _cancel| Box::pin(async { Ok(()) }),
    );
    let second_submission = session
        .submit_replacing(Arc::new(second))
        .await
        .expect("replacement submission");
    assert_eq!(
        first_submission.completion.await.expect("old completion"),
        Ok(RuntimeSessionTaskOutcome::Replaced)
    );
    assert_eq!(
        second_submission
            .completion
            .await
            .expect("replacement completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    registry
        .shutdown("session-replace")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn step_context_and_usage_are_turn_scoped_and_monotonic() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-step-context").await;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let usage = Arc::new(Mutex::new(RuntimeSessionTokenUsage::default()));
    let seen_for_task = Arc::clone(&seen);
    let usage_for_task = Arc::clone(&usage);
    let task = RuntimeSessionClosureTask::new(
        "turn-step-context",
        Vec::new(),
        move |context, _input, _cancel| {
            let seen = Arc::clone(&seen_for_task);
            let usage = Arc::clone(&usage_for_task);
            Box::pin(async move {
                seen.lock().await.push(context.capture_step_context().await);
                context.record_token_usage(10, 3, 1, 7).await;
                seen.lock().await.push(context.capture_step_context().await);
                assert_eq!(context.advance_context_epoch().await, 1);
                seen.lock().await.push(context.capture_step_context().await);
                *usage.lock().await = context.token_usage().await;
                Ok(())
            })
        },
    );
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("step task");
    assert_eq!(
        submission.completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    let seen = seen.lock().await;
    assert_eq!(seen.len(), 3);
    assert_eq!(seen[0].step_index, 1);
    assert_eq!(seen[1].step_index, 2);
    assert_eq!(seen[2].step_index, 3);
    assert_eq!(seen[0].context_epoch, 0);
    assert_eq!(seen[2].context_epoch, 1);
    assert_eq!(seen[0].session_id, "session-step-context");
    assert_eq!(seen[0].turn_id, "turn-step-context");
    assert_eq!(
        *usage.lock().await,
        RuntimeSessionTokenUsage {
            input_tokens: 10,
            output_tokens: 3,
            reasoning_tokens: 1,
            cache_write_input_tokens: 7,
        }
    );
    registry
        .shutdown("session-step-context")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn context_rollover_is_consumed_by_the_next_step_snapshot() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-context-rollover").await;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_for_task = Arc::clone(&seen);
    let task = RuntimeSessionClosureTask::new(
        "turn-context-rollover",
        Vec::new(),
        move |context, _input, _cancel| {
            let seen = Arc::clone(&seen_for_task);
            Box::pin(async move {
                seen.lock().await.push(context.capture_step_context().await);
                context.request_context_rollover().await;
                assert!(context.context_rollover_requested().await);
                seen.lock().await.push(context.capture_step_context().await);
                assert!(!context.context_rollover_requested().await);
                seen.lock().await.push(context.capture_step_context().await);
                Ok(())
            })
        },
    );
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("rollover task");
    assert_eq!(
        submission.completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    let seen = seen.lock().await;
    assert_eq!(seen[0].context_epoch, 0);
    assert_eq!(seen[1].context_epoch, 1);
    assert_eq!(seen[2].context_epoch, 1);
    registry
        .shutdown("session-context-rollover")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn final_answer_defers_mailbox_until_steer_reopens_the_turn() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-mailbox-phase").await;
    let (ready_tx, ready_rx) = oneshot::channel();
    let ready_tx = Arc::new(Mutex::new(Some(ready_tx)));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_for_task = Arc::clone(&seen);
    let task = RuntimeSessionClosureTask::new(
        "turn-mailbox-phase",
        Vec::new(),
        move |context, _input, _cancel| {
            let ready_tx = Arc::clone(&ready_tx);
            let seen = Arc::clone(&seen_for_task);
            Box::pin(async move {
                let handle = context.input_handle();
                handle.mark_mailbox_delivery_for_next_turn().await;
                if let Some(sender) = ready_tx.lock().await.take() {
                    let _ = sender.send(());
                }
                assert!(!handle.has_pending_input(true).await);
                handle.wait_for_pending_input().await;
                seen.lock().await.extend(
                    handle
                        .try_take_pending_input(true)
                        .await
                        .expect("reopened input"),
                );
                Ok(())
            })
        },
    )
    .with_mailbox_loader(|| {
        Box::pin(async {
            Ok(vec![RuntimeSessionInput::InterAgent(inter_agent_input(
                "mailbox-phase",
                "mail",
            ))])
        })
    });
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("phase task");
    ready_rx.await.expect("phase ready");
    session
        .notify_inter_agent_communication(inter_agent_input("mailbox-phase", "mail"))
        .await
        .expect("mailbox");
    tokio::task::yield_now().await;
    assert!(session
        .steer_for_turn(
            Some("turn-mailbox-phase"),
            vec![RuntimeSessionInput::User(RuntimeReplyInput::text(
                "follow up"
            ))]
        )
        .await
        .is_ok());
    assert_eq!(
        submission.completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    let seen = seen.lock().await;
    assert_eq!(seen.len(), 2);
    assert!(matches!(seen[0], RuntimeSessionInput::User(_)));
    assert!(matches!(seen[1], RuntimeSessionInput::InterAgent(_)));
    registry
        .shutdown("session-mailbox-phase")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn stale_final_defer_does_not_override_a_steer() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-stale-defer").await;
    let (ready_tx, ready_rx) = oneshot::channel();
    let ready_tx = Arc::new(Mutex::new(Some(ready_tx)));
    let task = RuntimeSessionClosureTask::new(
        "turn-stale-defer",
        Vec::new(),
        move |context, _input, _cancel| {
            let ready_tx = Arc::clone(&ready_tx);
            Box::pin(async move {
                let handle = context.input_handle();
                handle.mark_mailbox_delivery_for_next_turn().await;
                if let Some(sender) = ready_tx.lock().await.take() {
                    let _ = sender.send(handle);
                }
                std::future::pending::<Result<(), RuntimeSessionTaskFailure>>().await
            })
        },
    );
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("stale defer task");
    let handle = ready_rx.await.expect("handle");
    session
        .steer_for_turn(
            Some("turn-stale-defer"),
            vec![RuntimeSessionInput::User(RuntimeReplyInput::text("steer"))],
        )
        .await
        .expect("steer");
    handle.mark_mailbox_delivery_for_next_turn().await;
    assert_eq!(
        handle.mailbox_delivery_phase().await,
        RuntimeSessionMailboxDeliveryPhase::CurrentTurn
    );
    session.interrupt().await.expect("interrupt");
    assert_eq!(
        submission.completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Interrupted)
    );
    registry
        .shutdown("session-stale-defer")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn response_waiter_uses_the_active_turn_generation() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-response").await;
    let (ready_tx, ready_rx) = oneshot::channel();
    let ready_tx = Arc::new(Mutex::new(Some(ready_tx)));
    let seen = Arc::new(Mutex::new(None));
    let seen_for_task = Arc::clone(&seen);
    let task = RuntimeSessionClosureTask::new(
        "turn-response",
        Vec::new(),
        move |context, _input, _cancel| {
            let ready_tx = Arc::clone(&ready_tx);
            let seen = Arc::clone(&seen_for_task);
            Box::pin(async move {
                let receiver = context
                    .register_response(RuntimeSessionResponseKind::Approval, "request-1")
                    .await
                    .expect("register response");
                if let Some(sender) = ready_tx.lock().await.take() {
                    let _ = sender.send(());
                }
                *seen.lock().await = Some(receiver.wait().await.expect("response"));
                Ok(())
            })
        },
    );
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("response task");
    ready_rx.await.expect("waiter ready");
    assert!(session
        .approve(
            Some("turn-other"),
            "request-1",
            serde_json::json!({"confirmed":true}),
        )
        .await
        .is_err());
    session
        .approve(
            Some("turn-response"),
            "request-1",
            serde_json::json!({"confirmed":true}),
        )
        .await
        .expect("resolve response");
    assert_eq!(
        submission.completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    assert_eq!(
        *seen.lock().await,
        Some(serde_json::json!({"confirmed":true}))
    );
    registry
        .shutdown("session-response")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn typed_response_operations_route_to_distinct_waiters() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-typed-responses").await;
    let (ready_tx, ready_rx) = oneshot::channel();
    let ready_tx = Arc::new(Mutex::new(Some(ready_tx)));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_for_task = Arc::clone(&seen);
    let task = RuntimeSessionClosureTask::new(
        "turn-typed-responses",
        Vec::new(),
        move |context, _input, _cancel| {
            let ready_tx = Arc::clone(&ready_tx);
            let seen = Arc::clone(&seen_for_task);
            Box::pin(async move {
                let waiters = [
                    RuntimeSessionResponseKind::Approval,
                    RuntimeSessionResponseKind::AskUser,
                    RuntimeSessionResponseKind::Permission,
                    RuntimeSessionResponseKind::DynamicTool,
                    RuntimeSessionResponseKind::McpElicitation,
                ];
                let mut receivers = Vec::new();
                for kind in waiters {
                    receivers.push(context.register_response(kind, format!("{kind:?}")).await?);
                }
                if let Some(sender) = ready_tx.lock().await.take() {
                    let _ = sender.send(());
                }
                for receiver in receivers {
                    seen.lock().await.push(receiver.wait().await?);
                }
                Ok(())
            })
        },
    );
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("typed response task");
    ready_rx.await.expect("waiters ready");

    session
        .approve(
            Some("turn-typed-responses"),
            format!("{:?}", RuntimeSessionResponseKind::Approval),
            serde_json::json!({"type":"approval"}),
        )
        .await
        .expect("approval response");
    session
        .answer_user_input(
            Some("turn-typed-responses"),
            format!("{:?}", RuntimeSessionResponseKind::AskUser),
            serde_json::json!({"type":"ask-user"}),
        )
        .await
        .expect("ask-user response");
    session
        .respond_permission(
            Some("turn-typed-responses"),
            format!("{:?}", RuntimeSessionResponseKind::Permission),
            serde_json::json!({"type":"permission"}),
        )
        .await
        .expect("permission response");
    session
        .respond_dynamic_tool(
            Some("turn-typed-responses"),
            format!("{:?}", RuntimeSessionResponseKind::DynamicTool),
            serde_json::json!({"type":"dynamic-tool"}),
        )
        .await
        .expect("dynamic tool response");
    session
        .resolve_mcp_elicitation(
            Some("turn-typed-responses"),
            format!("{:?}", RuntimeSessionResponseKind::McpElicitation),
            serde_json::json!({"type":"elicitation"}),
        )
        .await
        .expect("elicitation response");

    assert_eq!(
        submission.completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    assert_eq!(
        *seen.lock().await,
        vec![
            serde_json::json!({"type":"approval"}),
            serde_json::json!({"type":"ask-user"}),
            serde_json::json!({"type":"permission"}),
            serde_json::json!({"type":"dynamic-tool"}),
            serde_json::json!({"type":"elicitation"}),
        ]
    );
    registry
        .shutdown("session-typed-responses")
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn mailbox_loader_failure_preserves_steer_input() {
    let registry = RuntimeSessionRegistry::default();
    let session = actor(&registry, "session-loader-error").await;
    let (ready_tx, ready_rx) = oneshot::channel();
    let ready_tx = Arc::new(Mutex::new(Some(ready_tx)));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_for_task = Arc::clone(&seen);
    let task = RuntimeSessionClosureTask::new(
        "turn-loader-error",
        Vec::new(),
        move |context, _input, _cancel| {
            let ready_tx = Arc::clone(&ready_tx);
            let seen = Arc::clone(&seen_for_task);
            Box::pin(async move {
                let handle = context.input_handle();
                if let Some(sender) = ready_tx.lock().await.take() {
                    let _ = sender.send(());
                }
                while !handle.has_pending_input(false).await {
                    tokio::task::yield_now().await;
                }
                handle
                    .try_take_pending_input(true)
                    .await
                    .expect_err("loader should fail");
                seen.lock().await.extend(
                    handle
                        .try_take_pending_input(false)
                        .await
                        .expect("steer remains"),
                );
                Ok(())
            })
        },
    )
    .with_mailbox_loader(|| Box::pin(async { Err("mailbox unavailable".to_string()) }));
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("loader error task");
    ready_rx.await.expect("loader task ready");
    session
        .steer(vec![RuntimeSessionInput::User(RuntimeReplyInput::text(
            "preserve",
        ))])
        .await
        .expect("steer");
    assert_eq!(
        submission.completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    assert!(matches!(
        seen.lock().await.as_slice(),
        [RuntimeSessionInput::User(input)] if input.concat_text() == "preserve"
    ));
    registry
        .shutdown("session-loader-error")
        .await
        .expect("shutdown");
}

fn input_handle_with_loader(
    pending_input: Arc<PendingInputQueue>,
    loader: RuntimeSessionMailboxLoader,
) -> RuntimeSessionInputHandle {
    RuntimeSessionInputHandle {
        session_id: Arc::from("session-activity"),
        pending_input,
        turn_id: Arc::from("turn-activity"),
        kind: RuntimeSessionTaskKind::Regular,
        mailbox_loader: Some(loader),
        resources: Arc::new(RuntimeSessionResources::new("thread-activity", None)),
        state: Arc::new(RuntimeSessionTaskState::default()),
    }
}

#[tokio::test]
async fn activity_subscription_notifies_mailbox_and_steer() {
    let pending_input = Arc::new(PendingInputQueue::default());
    let handle = input_handle_with_loader(
        Arc::clone(&pending_input),
        Arc::new(|| Box::pin(async { Ok(Vec::new()) })),
    );
    let (mut activity, pending_activity) = handle.subscribe_activity().await;
    assert_eq!(pending_activity, None);

    pending_input.notify_mailbox_activity().await;
    activity.changed().await.expect("mailbox activity");
    assert_eq!(
        *activity.borrow_and_update(),
        RuntimeSessionInputActivity::Mailbox
    );

    assert!(
        handle
            .push_steer(vec![RuntimeSessionInput::User(RuntimeReplyInput::text(
                "steer"
            ))])
            .await
    );
    activity.changed().await.expect("steer activity");
    assert_eq!(
        *activity.borrow_and_update(),
        RuntimeSessionInputActivity::Steer
    );
}

#[tokio::test]
async fn async_hook_context_targets_active_turn_then_next_turn_mailbox() {
    let active = input_handle_with_loader(
        Arc::new(PendingInputQueue::default()),
        Arc::new(|| Box::pin(async { Ok(Vec::new()) })),
    );
    assert!(active.inject_developer_input("active context").await);
    let active_inputs = active
        .try_take_pending_input(false)
        .await
        .expect("active context");
    assert!(matches!(
        active_inputs.as_slice(),
        [RuntimeSessionInput::Developer(text)] if text == "active context"
    ));

    let idle = input_handle_with_loader(
        Arc::new(PendingInputQueue::default()),
        Arc::new(|| Box::pin(async { Ok(Vec::new()) })),
    );
    idle.mark_finishing().await;
    assert!(!idle.inject_developer_input("idle context").await);
    idle.accept_mailbox_delivery_for_current_turn().await;
    let idle_inputs = idle
        .try_take_pending_input(true)
        .await
        .expect("idle context");
    assert!(matches!(
        idle_inputs.as_slice(),
        [RuntimeSessionInput::Developer(text)] if text == "idle context"
    ));
}

#[tokio::test]
async fn activity_subscription_reports_pending_and_prioritizes_steer() {
    let pending_input = Arc::new(PendingInputQueue::default());
    let handle = input_handle_with_loader(
        Arc::clone(&pending_input),
        Arc::new(|| Box::pin(async { Ok(Vec::new()) })),
    );

    pending_input.notify_mailbox_activity().await;
    let (_, pending_activity) = handle.subscribe_activity().await;
    assert_eq!(pending_activity, Some(RuntimeSessionInputActivity::Mailbox));

    assert!(
        handle
            .push_steer(vec![RuntimeSessionInput::User(RuntimeReplyInput::text(
                "steer"
            ))])
            .await
    );
    let (_, pending_activity) = handle.subscribe_activity().await;
    assert_eq!(pending_activity, Some(RuntimeSessionInputActivity::Steer));
}

#[tokio::test]
async fn next_turn_mailbox_activity_is_not_deliverable_to_current_turn() {
    let pending_input = Arc::new(PendingInputQueue::default());
    let handle = input_handle_with_loader(
        Arc::clone(&pending_input),
        Arc::new(|| Box::pin(async { Ok(Vec::new()) })),
    );
    handle.mark_mailbox_delivery_for_next_turn().await;
    let (mut activity, pending_activity) = handle.subscribe_activity().await;
    assert_eq!(pending_activity, None);

    pending_input.notify_mailbox_activity().await;
    activity.changed().await.expect("mailbox activity");
    assert_eq!(
        *activity.borrow_and_update(),
        RuntimeSessionInputActivity::Mailbox
    );
    assert!(!handle.has_pending_input(true).await);
    assert_eq!(
        handle.mailbox_delivery_phase().await,
        RuntimeSessionMailboxDeliveryPhase::NextTurn
    );
}

#[tokio::test]
async fn mailbox_loader_failure_preserves_activity_generation() {
    let pending_input = Arc::new(PendingInputQueue::default());
    let handle = input_handle_with_loader(
        Arc::clone(&pending_input),
        Arc::new(|| Box::pin(async { Err("mailbox unavailable".to_string()) })),
    );
    pending_input.notify_mailbox_activity().await;

    handle
        .try_take_pending_input(true)
        .await
        .expect_err("loader should fail");

    assert!(handle.has_pending_input(true).await);
    let (_, pending_activity) = handle.subscribe_activity().await;
    assert_eq!(pending_activity, Some(RuntimeSessionInputActivity::Mailbox));
}

#[tokio::test]
async fn mailbox_loader_acknowledges_only_the_captured_generation() {
    let pending_input = Arc::new(PendingInputQueue::default());
    let calls = Arc::new(AtomicUsize::new(0));
    let pending_for_loader = Arc::clone(&pending_input);
    let calls_for_loader = Arc::clone(&calls);
    let handle = input_handle_with_loader(
        Arc::clone(&pending_input),
        Arc::new(move || {
            let pending_input = Arc::clone(&pending_for_loader);
            let calls = Arc::clone(&calls_for_loader);
            Box::pin(async move {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    pending_input.notify_mailbox_activity().await;
                }
                Ok(vec![RuntimeSessionInput::InterAgent(inter_agent_input(
                    "mailbox-first-generation",
                    "first generation",
                ))])
            })
        }),
    );
    pending_input.notify_mailbox_activity().await;

    let delivered = handle
        .try_take_pending_input(true)
        .await
        .expect("loader succeeds");
    assert_eq!(delivered.len(), 1);

    assert!(handle.has_pending_input(true).await);
    let (_, pending_activity) = handle.subscribe_activity().await;
    assert_eq!(pending_activity, Some(RuntimeSessionInputActivity::Mailbox));
}

#[tokio::test]
async fn mailbox_loader_serializes_final_defer_with_delivery_side_effects() {
    let pending_input = Arc::new(PendingInputQueue::default());
    let loader_started = Arc::new(Notify::new());
    let release_loader = Arc::new(Notify::new());
    let started_for_loader = Arc::clone(&loader_started);
    let release_for_loader = Arc::clone(&release_loader);
    let handle = input_handle_with_loader(
        Arc::clone(&pending_input),
        Arc::new(move || {
            let loader_started = Arc::clone(&started_for_loader);
            let release_loader = Arc::clone(&release_for_loader);
            Box::pin(async move {
                loader_started.notify_one();
                release_loader.notified().await;
                Ok(vec![RuntimeSessionInput::InterAgent(inter_agent_input(
                    "mailbox-loaded",
                    "loaded",
                ))])
            })
        }),
    );
    pending_input.notify_mailbox_activity().await;

    let handle_for_delivery = handle.clone();
    let delivery = tokio::spawn(async move {
        handle_for_delivery
            .try_take_pending_input(true)
            .await
            .expect("loader succeeds")
    });
    loader_started.notified().await;

    let handle_for_defer = handle.clone();
    let defer = tokio::spawn(async move {
        handle_for_defer.mark_mailbox_delivery_for_next_turn().await;
    });
    tokio::task::yield_now().await;
    assert!(!defer.is_finished());

    release_loader.notify_one();
    assert_eq!(delivery.await.expect("delivery task").len(), 1);
    defer.await.expect("defer task");
    assert_eq!(
        handle.mailbox_delivery_phase().await,
        RuntimeSessionMailboxDeliveryPhase::NextTurn
    );
}

#[tokio::test]
async fn handle_notification_is_visible_to_a_late_task_subscriber() {
    let registry = RuntimeSessionRegistry::default();
    assert!(!registry
        .notify_inter_agent_communication(
            "session-late-activity",
            inter_agent_input("mailbox-late-missing", "missing"),
        )
        .await
        .expect("missing session lookup"));
    let session = actor(&registry, "session-late-activity").await;
    assert!(registry
        .notify_inter_agent_communication(
            "session-late-activity",
            inter_agent_input("mailbox-late-existing", "existing"),
        )
        .await
        .expect("mailbox activity"));
    let (seen_tx, seen_rx) = oneshot::channel();
    let seen_tx = Arc::new(Mutex::new(Some(seen_tx)));
    let task = RuntimeSessionClosureTask::new(
        "turn-late-activity",
        Vec::new(),
        move |context, _input, _cancel| {
            let seen_tx = Arc::clone(&seen_tx);
            Box::pin(async move {
                let (_, pending_activity) = context.subscribe_activity().await;
                if let Some(sender) = seen_tx.lock().await.take() {
                    let _ = sender.send(pending_activity);
                }
                Ok(())
            })
        },
    );
    let submission = session
        .submit(Arc::new(task), false)
        .await
        .expect("task submission");

    assert_eq!(
        seen_rx.await.expect("activity snapshot"),
        Some(RuntimeSessionInputActivity::Mailbox)
    );
    assert_eq!(
        submission.completion.await.expect("completion"),
        Ok(RuntimeSessionTaskOutcome::Completed)
    );
    registry
        .shutdown("session-late-activity")
        .await
        .expect("shutdown");
}
