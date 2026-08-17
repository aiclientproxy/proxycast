use crate::processor;
use crate::thread_state;
use crate::{error_codes, AppServerEventBridge, JsonRpcError, RuntimeCoreError, RuntimeEvent};
use app_server_protocol::protocol::v2::ServerNotification;
use app_server_protocol::{AgentEvent, JsonRpcMessage, JsonRpcNotification};
use app_server_transport::ConnectionId;
use std::sync::Arc;
use tokio::sync::oneshot;

impl AppServerEventBridge {
    pub async fn append_external_runtime_events(
        &self,
        session_id: &str,
        turn_id: Option<&str>,
        runtime_events: Vec<RuntimeEvent>,
    ) -> Result<Vec<JsonRpcMessage>, JsonRpcError> {
        let events = self
            .runtime_events
            .append_external_runtime_events(session_id, turn_id, runtime_events)
            .map_err(RuntimeCoreError::into_jsonrpc_error)?;
        let mut messages = Vec::new();
        for event in events {
            let thread_id = event
                .thread_id
                .as_deref()
                .map(str::trim)
                .filter(|thread_id| !thread_id.is_empty())
                .map(agent_protocol::ThreadId::new)
                .ok_or_else(|| {
                    JsonRpcError::new(
                        error_codes::RUNTIME_ERROR,
                        format!(
                            "runtime event {} ({}) is missing canonical threadId",
                            event.event_id, event.event_type
                        ),
                    )
                })?;
            let (completion_tx, completion_rx) = oneshot::channel();
            self.send_thread_command(
                thread_id,
                thread_state::ThreadListenerCommand::PublishRuntimeEvent {
                    event,
                    completion_tx: Some(completion_tx),
                },
            )
            .await
            .map_err(|error| JsonRpcError::new(error_codes::RUNTIME_ERROR, error))?;
            let projected = completion_rx
                .await
                .map_err(|error| {
                    JsonRpcError::new(
                        error_codes::RUNTIME_ERROR,
                        format!("thread listener response channel closed: {error}"),
                    )
                })?
                .map_err(|error| JsonRpcError::new(error_codes::RUNTIME_ERROR, error))?;
            messages.extend(projected);
        }
        Ok(messages)
    }

    pub(crate) async fn send_thread_command(
        &self,
        thread_id: agent_protocol::ThreadId,
        command: thread_state::ThreadListenerCommand,
    ) -> Result<(), String> {
        let thread_state = self
            .thread_states
            .thread_state_for_listener(thread_id.clone())
            .await
            .ok_or_else(|| "thread listeners are shutting down".to_string())?;
        let (command_tx, registration) = {
            let mut state = thread_state.lock().await;
            if let Some(command_tx) = state.listener_command_tx() {
                (command_tx, None)
            } else {
                let registration = state.set_listener();
                let command_tx = state
                    .listener_command_tx()
                    .expect("thread listener sender missing after registration");
                (command_tx, Some(registration))
            }
        };

        if let Some(registration) = registration {
            tokio::spawn(run_thread_listener(
                thread_id.clone(),
                thread_state,
                self.clone(),
                registration,
            ));
        }

        command_tx
            .send(command)
            .map_err(|_| format!("thread listener command channel is closed for {thread_id}"))
    }

    pub(crate) async fn prepare_thread_resume(
        &self,
        thread_id: agent_protocol::ThreadId,
        barrier: thread_state::ThreadResumeBarrier,
    ) -> Result<(), String> {
        let thread_state = self
            .thread_states
            .thread_state_for_listener(thread_id.clone())
            .await
            .ok_or_else(|| format!("thread listeners are shutting down for {thread_id}"))?;
        let registration = {
            let mut state = thread_state.lock().await;
            if !state.begin_resume(barrier.clone()) {
                return Err(format!(
                    "thread {thread_id} already has resume barrier {barrier:?}"
                ));
            }
            let registration = state
                .listener_command_tx()
                .is_none()
                .then(|| state.set_listener());
            registration
        };

        if let Some(registration) = registration {
            tokio::spawn(run_thread_listener(
                thread_id,
                thread_state,
                self.clone(),
                registration,
            ));
        }
        Ok(())
    }

    async fn publish_runtime_event(
        &self,
        thread_id: &agent_protocol::ThreadId,
        projector: &mut processor::v2_notifications::V2NotificationProjector,
        event: AgentEvent,
    ) -> Result<Vec<JsonRpcMessage>, String> {
        let messages = projector
            .project(event)
            .map_err(|error| error.message.to_string())?
            .into_iter()
            .map(JsonRpcNotification::into)
            .map(JsonRpcMessage::Notification)
            .collect::<Vec<_>>();

        for message in &messages {
            let _ = self.outbound_messages.send(message.clone());
        }
        let connection_ids = self
            .thread_states
            .subscribed_connection_ids(thread_id)
            .await;
        for connection_id in connection_ids {
            if let Err(error) = self
                .send_messages_to_connection(connection_id, &messages)
                .await
            {
                tracing::warn!(
                    %thread_id,
                    %connection_id,
                    %error,
                    "failed to publish thread runtime event; removing stale subscription"
                );
                self.thread_states
                    .unsubscribe_connection(thread_id, connection_id)
                    .await;
            }
        }
        Ok(messages)
    }

    async fn publish_notification(
        &self,
        thread_id: &agent_protocol::ThreadId,
        notification: JsonRpcNotification,
        origin_connection_id: Option<ConnectionId>,
    ) -> Result<(), String> {
        let message = JsonRpcMessage::Notification(notification);
        let _ = self.outbound_messages.send(message.clone());
        let mut connection_ids = self
            .thread_states
            .subscribed_connection_ids(thread_id)
            .await;
        if let Some(origin_connection_id) = origin_connection_id {
            if !connection_ids.contains(&origin_connection_id) {
                connection_ids.push(origin_connection_id);
            }
        }
        for connection_id in connection_ids {
            if let Err(error) = self
                .send_messages_to_connection(connection_id, std::slice::from_ref(&message))
                .await
            {
                tracing::warn!(
                    %thread_id,
                    %connection_id,
                    %error,
                    "failed to publish thread notification; removing stale subscription"
                );
                self.thread_states
                    .unsubscribe_connection(thread_id, connection_id)
                    .await;
            }
        }
        Ok(())
    }

    async fn subscribe_and_send(
        &self,
        thread_id: &agent_protocol::ThreadId,
        connection_id: ConnectionId,
        messages: &[JsonRpcMessage],
    ) -> Result<(), String> {
        if !self
            .thread_states
            .subscribe_connection(thread_id.clone(), connection_id)
            .await
        {
            return Err(format!(
                "connection {connection_id} is not live for thread {thread_id}"
            ));
        }
        if let Err(error) = self
            .send_messages_to_connection(connection_id, messages)
            .await
        {
            self.thread_states
                .unsubscribe_connection(thread_id, connection_id)
                .await;
            return Err(error);
        }
        Ok(())
    }
}

async fn run_thread_listener(
    thread_id: agent_protocol::ThreadId,
    thread_state: Arc<tokio::sync::Mutex<thread_state::ThreadState>>,
    bridge: AppServerEventBridge,
    mut registration: thread_state::ThreadListenerRegistration,
) {
    let mut projector = processor::v2_notifications::V2NotificationProjector::default();
    let mut deferred_publishes = Vec::new();
    loop {
        tokio::select! {
            biased;
            _ = registration.cancellation.cancelled() => break,
            command = registration.command_rx.recv() => {
                let Some(command) = command else {
                    break;
                };
                match command {
                    thread_state::ThreadListenerCommand::PublishRuntimeEvent {
                        event,
                        completion_tx,
                    } => {
                        if thread_state.lock().await.has_pending_resume() {
                            deferred_publishes.push(DeferredThreadPublish::RuntimeEvent {
                                event,
                                completion_tx,
                            });
                        } else {
                            publish_thread_listener_event(
                                &bridge,
                                &thread_id,
                                &mut projector,
                                event,
                                completion_tx,
                            )
                            .await;
                        }
                    }
                    thread_state::ThreadListenerCommand::PublishNotification {
                        notification,
                        origin_connection_id,
                        completion_tx,
                    } => {
                        if thread_state.lock().await.has_pending_resume() {
                            deferred_publishes.push(DeferredThreadPublish::Notification {
                                notification,
                                origin_connection_id,
                                completion_tx,
                            });
                        } else {
                            publish_thread_listener_notification(
                                &bridge,
                                &thread_id,
                                notification,
                                origin_connection_id,
                                completion_tx,
                            )
                            .await;
                        }
                    }
                    thread_state::ThreadListenerCommand::SubscribeAndSend {
                        connection_id,
                        messages,
                        completion_tx,
                    } => {
                        let result = bridge
                            .subscribe_and_send(&thread_id, connection_id, &messages)
                            .await;
                        let _ = completion_tx.send(result);
                    }
                    thread_state::ThreadListenerCommand::CompleteResume {
                        barrier,
                        connection_id,
                        messages,
                        subscribe,
                        completion_tx,
                    } => {
                        let mut result = if subscribe {
                            bridge
                                .subscribe_and_send(&thread_id, connection_id, &messages)
                                .await
                        } else {
                            bridge
                                .send_messages_to_connection(connection_id, &messages)
                                .await
                        };
                        let finish = thread_state.lock().await.finish_resume(&barrier);
                        match finish {
                            Some(true) => {
                                for publish in deferred_publishes.drain(..) {
                                    if let Some(error) = publish_deferred_thread_publish(
                                        &bridge,
                                        &thread_id,
                                        &mut projector,
                                        publish,
                                    )
                                    .await
                                    {
                                        if result.is_ok() {
                                            result = Err(error);
                                        }
                                    }
                                }
                            }
                            Some(false) => {}
                            None => {
                                if result.is_ok() {
                                    result = Err(format!(
                                        "thread {thread_id} completed an unknown resume barrier"
                                    ));
                                }
                            }
                        }
                        let _ = completion_tx.send(result);
                    }
                }
            }
        }
    }

    thread_state
        .lock()
        .await
        .clear_listener_if_generation(registration.generation);
}

enum DeferredThreadPublish {
    RuntimeEvent {
        event: AgentEvent,
        completion_tx: Option<oneshot::Sender<Result<Vec<JsonRpcMessage>, String>>>,
    },
    Notification {
        notification: JsonRpcNotification,
        origin_connection_id: Option<ConnectionId>,
        completion_tx: Option<oneshot::Sender<Result<(), String>>>,
    },
}

async fn publish_deferred_thread_publish(
    bridge: &AppServerEventBridge,
    thread_id: &agent_protocol::ThreadId,
    projector: &mut processor::v2_notifications::V2NotificationProjector,
    publish: DeferredThreadPublish,
) -> Option<String> {
    match publish {
        DeferredThreadPublish::RuntimeEvent {
            event,
            completion_tx,
        } => {
            publish_thread_listener_event(bridge, thread_id, projector, event, completion_tx).await
        }
        DeferredThreadPublish::Notification {
            notification,
            origin_connection_id,
            completion_tx,
        } => {
            publish_thread_listener_notification(
                bridge,
                thread_id,
                notification,
                origin_connection_id,
                completion_tx,
            )
            .await
        }
    }
}

async fn publish_thread_listener_event(
    bridge: &AppServerEventBridge,
    thread_id: &agent_protocol::ThreadId,
    projector: &mut processor::v2_notifications::V2NotificationProjector,
    event: AgentEvent,
    completion_tx: Option<oneshot::Sender<Result<Vec<JsonRpcMessage>, String>>>,
) -> Option<String> {
    let terminal_event = matches!(
        event.event_type.as_str(),
        "turn.completed" | "turn.failed" | "turn.canceled"
    )
    .then(|| event.clone());
    let pending_goal_updates = bridge
        .runtime_events
        .pending_thread_goal_updates_for_event(&event);
    let mut result = bridge
        .publish_runtime_event(thread_id, projector, event)
        .await;
    if result.is_ok() {
        match pending_goal_updates {
            Ok(updates) => {
                for update in updates {
                    let notification: JsonRpcNotification =
                        ServerNotification::ThreadGoalUpdated(update.notification).into();
                    let message = JsonRpcMessage::Notification(notification.clone());
                    if let Err(error) = bridge
                        .publish_notification(thread_id, notification, None)
                        .await
                    {
                        result = Err(error);
                        break;
                    }
                    match bridge
                        .runtime_events
                        .mark_thread_goal_update_delivered(update.outbox_id)
                    {
                        Ok(_) => {
                            if let Ok(messages) = result.as_mut() {
                                messages.push(message);
                            }
                        }
                        Err(error) => {
                            result = Err(error.to_string());
                            break;
                        }
                    }
                }
            }
            Err(error) => result = Err(error.to_string()),
        }
    }
    if let Some(event) = terminal_event {
        match bridge
            .runtime
            .finish_scheduled_task_run_for_terminal_event(event.clone())
            .await
        {
            Ok(Some(notification)) => bridge.publish_server_notification(notification).await,
            Ok(None) => {}
            Err(error) => tracing::error!(
                event_id = %event.event_id,
                session_id = %event.session_id,
                turn_id = ?event.turn_id,
                %error,
                "failed to finish scheduled task run from canonical turn terminal event"
            ),
        }
    }
    let error = result.as_ref().err().cloned();
    if let Some(completion_tx) = completion_tx {
        let _ = completion_tx.send(result);
    } else if let Some(error) = error.as_ref() {
        tracing::error!(%error, "failed to publish background runtime event");
    }
    error
}

async fn publish_thread_listener_notification(
    bridge: &AppServerEventBridge,
    thread_id: &agent_protocol::ThreadId,
    notification: JsonRpcNotification,
    origin_connection_id: Option<ConnectionId>,
    completion_tx: Option<oneshot::Sender<Result<(), String>>>,
) -> Option<String> {
    let result = bridge
        .publish_notification(thread_id, notification, origin_connection_id)
        .await;
    let error = result.as_ref().err().cloned();
    if let Some(completion_tx) = completion_tx {
        let _ = completion_tx.send(result);
    } else if let Some(error) = error.as_ref() {
        tracing::error!(%error, "failed to publish background thread notification");
    }
    error
}
