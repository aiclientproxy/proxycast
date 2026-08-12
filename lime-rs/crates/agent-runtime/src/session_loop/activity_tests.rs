use super::*;
use crate::reply_input::RuntimeReplyInput;
use futures::future;
use std::sync::Arc;
use tokio::time::{timeout, Duration};

fn waiting_task(turn_id: &str, kind: RuntimeSessionTaskKind) -> Arc<dyn RuntimeSessionTask> {
    Arc::new(
        RuntimeSessionClosureTask::new(turn_id, Vec::new(), |_context, _input, cancel| {
            Box::pin(async move {
                cancel.cancelled().await;
                Ok(())
            })
        })
        .with_kind(kind),
    )
}

#[tokio::test]
async fn session_handle_activity_subscription_observes_steer() {
    let registry = RuntimeSessionRegistry::default();
    let session = registry
        .get_or_create("session-activity-handle", "thread-activity-handle")
        .await
        .expect("bind activity session actor");
    let active = session
        .submit(
            waiting_task("turn-active", RuntimeSessionTaskKind::Regular),
            false,
        )
        .await
        .expect("submit active task");
    let (mut activity, pending_activity) = session
        .subscribe_input_activity()
        .await
        .expect("subscribe input activity");
    assert_eq!(pending_activity, None);

    session
        .steer(vec![RuntimeSessionInput::User(RuntimeReplyInput::text(
            "new input",
        ))])
        .await
        .expect("steer active task");
    timeout(Duration::from_secs(1), activity.changed())
        .await
        .expect("steer activity timeout")
        .expect("steer activity sender");
    assert_eq!(
        *activity.borrow_and_update(),
        RuntimeSessionInputActivity::Steer
    );

    assert!(session.interrupt().await.expect("interrupt active task"));
    assert_eq!(
        active.completion.await.expect("active completion"),
        Ok(RuntimeSessionTaskOutcome::Interrupted)
    );
    registry
        .shutdown("session-activity-handle")
        .await
        .expect("shutdown session");
}

#[tokio::test]
async fn queued_user_admission_is_visible_to_late_activity_subscriber() {
    let registry = RuntimeSessionRegistry::default();
    let session = registry
        .get_or_create("session-queued-activity", "thread-queued-activity")
        .await
        .expect("bind queued activity session actor");
    let active = session
        .submit(
            waiting_task("turn-review", RuntimeSessionTaskKind::Review),
            false,
        )
        .await
        .expect("submit review task");
    let queued_task =
        RuntimeSessionClosureTask::new("turn-queued", Vec::new(), |_context, _input, _cancel| {
            Box::pin(future::ready(Ok(())))
        });
    let queued = session
        .submit_user_input_with_metadata(
            Arc::new(queued_task),
            vec![RuntimeSessionInput::User(RuntimeReplyInput::text(
                "queued input",
            ))],
            true,
            None,
            None,
        )
        .await
        .expect("queue user input");
    let RuntimeSessionUserInputResult::Submitted(queued) = queued else {
        panic!("expected queued submission");
    };
    assert!(matches!(
        queued.result,
        RuntimeSessionSubmitResult::Queued { position: 1 }
    ));

    let (_, pending_activity) = session
        .subscribe_input_activity()
        .await
        .expect("subscribe queued activity");
    assert_eq!(pending_activity, Some(RuntimeSessionInputActivity::Steer));

    session.shutdown().await.expect("shutdown session");
    assert_eq!(
        active.completion.await.expect("active completion"),
        Ok(RuntimeSessionTaskOutcome::Shutdown)
    );
    assert_eq!(
        queued.completion.await.expect("queued completion"),
        Ok(RuntimeSessionTaskOutcome::Shutdown)
    );
}
