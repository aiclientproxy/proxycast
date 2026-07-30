use agent_protocol::ThreadId;
use app_server_protocol::{AgentEvent, JsonRpcMessage, JsonRpcNotification, RequestId};
use app_server_transport::ConnectionId;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex, Notify};
use tokio_util::sync::CancellationToken;

pub(crate) enum ThreadListenerCommand {
    PublishRuntimeEvent {
        event: AgentEvent,
        completion_tx: Option<oneshot::Sender<Result<Vec<JsonRpcMessage>, String>>>,
    },
    PublishNotification {
        notification: JsonRpcNotification,
        origin_connection_id: Option<ConnectionId>,
        completion_tx: Option<oneshot::Sender<Result<(), String>>>,
    },
    SubscribeAndSend {
        connection_id: ConnectionId,
        messages: Vec<JsonRpcMessage>,
        completion_tx: oneshot::Sender<Result<(), String>>,
    },
    CompleteResume {
        barrier: ThreadResumeBarrier,
        connection_id: ConnectionId,
        messages: Vec<JsonRpcMessage>,
        subscribe: bool,
        completion_tx: oneshot::Sender<Result<(), String>>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ThreadResumeBarrier {
    connection_id: ConnectionId,
    request_id: RequestId,
}

impl ThreadResumeBarrier {
    pub(crate) fn new(connection_id: ConnectionId, request_id: RequestId) -> Self {
        Self {
            connection_id,
            request_id,
        }
    }
}

pub(crate) struct ThreadListenerRegistration {
    pub(crate) generation: u64,
    pub(crate) command_rx: mpsc::UnboundedReceiver<ThreadListenerCommand>,
    pub(crate) cancellation: CancellationToken,
}

#[derive(Default)]
pub(crate) struct ThreadState {
    listener_generation: u64,
    listener_command_tx: Option<mpsc::UnboundedSender<ThreadListenerCommand>>,
    listener_cancellation: Option<CancellationToken>,
    pending_resume_barriers: HashSet<ThreadResumeBarrier>,
}

impl ThreadState {
    pub(crate) fn set_listener(&mut self) -> ThreadListenerRegistration {
        if let Some(previous) = self.listener_cancellation.take() {
            previous.cancel();
        }

        self.listener_generation = self
            .listener_generation
            .checked_add(1)
            .expect("thread listener generation exhausted");
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let cancellation = CancellationToken::new();
        self.listener_command_tx = Some(command_tx);
        self.listener_cancellation = Some(cancellation.clone());

        ThreadListenerRegistration {
            generation: self.listener_generation,
            command_rx,
            cancellation,
        }
    }

    pub(crate) fn listener_command_tx(
        &self,
    ) -> Option<mpsc::UnboundedSender<ThreadListenerCommand>> {
        self.listener_command_tx.clone()
    }

    pub(crate) fn begin_resume(&mut self, barrier: ThreadResumeBarrier) -> bool {
        self.pending_resume_barriers.insert(barrier)
    }

    pub(crate) fn finish_resume(&mut self, barrier: &ThreadResumeBarrier) -> Option<bool> {
        self.pending_resume_barriers
            .remove(barrier)
            .then(|| self.pending_resume_barriers.is_empty())
    }

    pub(crate) fn has_pending_resume(&self) -> bool {
        !self.pending_resume_barriers.is_empty()
    }

    pub(crate) fn clear_listener_if_generation(&mut self, generation: u64) -> bool {
        if self.listener_generation != generation {
            return false;
        }

        self.clear_listener();
        true
    }

    fn clear_listener(&mut self) {
        if let Some(cancellation) = self.listener_cancellation.take() {
            cancellation.cancel();
        }
        self.listener_command_tx = None;
        self.pending_resume_barriers.clear();
    }
}

#[derive(Default)]
struct ThreadEntry {
    state: Arc<Mutex<ThreadState>>,
    connection_ids: HashSet<ConnectionId>,
    unload_generation: u64,
    unloading: bool,
}

#[derive(Default)]
struct ThreadStateManagerInner {
    live_connections: HashSet<ConnectionId>,
    threads: HashMap<ThreadId, ThreadEntry>,
    thread_ids_by_connection: HashMap<ConnectionId, HashSet<ThreadId>>,
    listeners_shutting_down: bool,
}

#[derive(Clone, Default)]
pub(crate) struct ThreadStateManager {
    inner: Arc<Mutex<ThreadStateManagerInner>>,
    subscriber_changed: Arc<Notify>,
}

impl ThreadStateManager {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) async fn connection_initialized(&self, connection_id: ConnectionId) {
        self.inner
            .lock()
            .await
            .live_connections
            .insert(connection_id);
    }

    #[cfg(test)]
    pub(crate) async fn thread_state(&self, thread_id: ThreadId) -> Arc<Mutex<ThreadState>> {
        self.inner
            .lock()
            .await
            .threads
            .entry(thread_id)
            .or_default()
            .state
            .clone()
    }

    #[cfg(test)]
    pub(crate) async fn contains_thread(&self, thread_id: &ThreadId) -> bool {
        self.inner.lock().await.threads.contains_key(thread_id)
    }

    pub(crate) async fn thread_state_for_listener(
        &self,
        thread_id: ThreadId,
    ) -> Option<Arc<Mutex<ThreadState>>> {
        let mut inner = self.inner.lock().await;
        if inner.listeners_shutting_down {
            return None;
        }
        let entry = inner.threads.entry(thread_id).or_default();
        (!entry.unloading).then(|| entry.state.clone())
    }

    pub(crate) async fn subscribe_connection(
        &self,
        thread_id: ThreadId,
        connection_id: ConnectionId,
    ) -> bool {
        let mut inner = self.inner.lock().await;
        if !inner.live_connections.contains(&connection_id) {
            return false;
        }
        inner
            .thread_ids_by_connection
            .entry(connection_id)
            .or_default()
            .insert(thread_id.clone());
        let entry = inner.threads.entry(thread_id).or_default();
        if entry.unloading {
            return false;
        }
        entry.unload_generation = entry
            .unload_generation
            .checked_add(1)
            .expect("thread unload generation exhausted");
        entry.connection_ids.insert(connection_id);
        self.subscriber_changed.notify_waiters();
        true
    }

    pub(crate) async fn unsubscribe_connection(
        &self,
        thread_id: &ThreadId,
        connection_id: ConnectionId,
    ) -> bool {
        let mut inner = self.inner.lock().await;
        let removed = inner
            .threads
            .get_mut(thread_id)
            .is_some_and(|entry| entry.connection_ids.remove(&connection_id));
        if !removed {
            return false;
        }

        let remove_connection_entry = inner
            .thread_ids_by_connection
            .get_mut(&connection_id)
            .is_some_and(|thread_ids| {
                thread_ids.remove(thread_id);
                thread_ids.is_empty()
            });
        if remove_connection_entry {
            inner.thread_ids_by_connection.remove(&connection_id);
        }
        true
    }

    pub(crate) async fn disconnect_connection(&self, connection_id: ConnectionId) -> Vec<ThreadId> {
        let mut inner = self.inner.lock().await;
        inner.live_connections.remove(&connection_id);
        let thread_ids = inner
            .thread_ids_by_connection
            .remove(&connection_id)
            .unwrap_or_default();

        for thread_id in &thread_ids {
            if let Some(entry) = inner.threads.get_mut(thread_id) {
                entry.connection_ids.remove(&connection_id);
            }
        }

        thread_ids
            .into_iter()
            .filter(|thread_id| {
                inner
                    .threads
                    .get(thread_id)
                    .is_some_and(|entry| entry.connection_ids.is_empty())
            })
            .collect()
    }

    pub(crate) async fn subscribed_connection_ids(
        &self,
        thread_id: &ThreadId,
    ) -> Vec<ConnectionId> {
        self.inner
            .lock()
            .await
            .threads
            .get(thread_id)
            .map(|entry| entry.connection_ids.iter().copied().collect())
            .unwrap_or_default()
    }

    pub(crate) async fn wait_for_thread_subscriber(&self, thread_id: &ThreadId) {
        loop {
            let changed = self.subscriber_changed.notified();
            if !self.subscribed_connection_ids(thread_id).await.is_empty() {
                return;
            }
            changed.await;
        }
    }

    pub(crate) async fn schedule_unload_if_idle(&self, thread_id: &ThreadId) -> Option<u64> {
        let mut inner = self.inner.lock().await;
        let entry = inner.threads.get_mut(thread_id)?;
        if entry.unloading || !entry.connection_ids.is_empty() {
            return None;
        }
        entry.unload_generation = entry
            .unload_generation
            .checked_add(1)
            .expect("thread unload generation exhausted");
        Some(entry.unload_generation)
    }

    pub(crate) async fn unload_ticket_is_current(
        &self,
        thread_id: &ThreadId,
        generation: u64,
    ) -> bool {
        self.inner
            .lock()
            .await
            .threads
            .get(thread_id)
            .is_some_and(|entry| {
                !entry.unloading
                    && entry.connection_ids.is_empty()
                    && entry.unload_generation == generation
            })
    }

    pub(crate) async fn begin_unload(&self, thread_id: &ThreadId, generation: u64) -> bool {
        let mut inner = self.inner.lock().await;
        let Some(entry) = inner.threads.get_mut(thread_id) else {
            return false;
        };
        if entry.unloading
            || !entry.connection_ids.is_empty()
            || entry.unload_generation != generation
        {
            return false;
        }
        entry.unloading = true;
        true
    }

    pub(crate) async fn cancel_unload(&self, thread_id: &ThreadId, generation: u64) {
        let mut inner = self.inner.lock().await;
        let Some(entry) = inner.threads.get_mut(thread_id) else {
            return;
        };
        if entry.unload_generation == generation {
            entry.unloading = false;
        }
    }

    pub(crate) async fn remove_thread(&self, thread_id: &ThreadId) {
        let entry = {
            let mut inner = self.inner.lock().await;
            let Some(entry) = inner.threads.remove(thread_id) else {
                return;
            };

            for connection_id in &entry.connection_ids {
                let remove_connection_entry = inner
                    .thread_ids_by_connection
                    .get_mut(connection_id)
                    .is_some_and(|thread_ids| {
                        thread_ids.remove(thread_id);
                        thread_ids.is_empty()
                    });
                if remove_connection_entry {
                    inner.thread_ids_by_connection.remove(connection_id);
                }
            }
            entry
        };

        entry.state.lock().await.clear_listener();
    }

    pub(crate) async fn clear_all_listeners(&self) {
        let thread_states = {
            let mut inner = self.inner.lock().await;
            inner.listeners_shutting_down = true;
            inner
                .threads
                .values()
                .map(|entry| entry.state.clone())
                .collect::<Vec<_>>()
        };
        for thread_state in thread_states {
            thread_state.lock().await.clear_listener();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn listener_replacement_cancels_previous_without_stale_cleanup() {
        let manager = ThreadStateManager::new();
        let thread_state = manager.thread_state(ThreadId::new("thread-1")).await;

        let first = thread_state.lock().await.set_listener();
        let first_generation = first.generation;
        let first_cancellation = first.cancellation.clone();
        let _first_commands = first.command_rx;

        let second = thread_state.lock().await.set_listener();
        let second_generation = second.generation;
        let second_cancellation = second.cancellation.clone();
        let _second_commands = second.command_rx;

        assert_eq!(first_generation, 1);
        assert_eq!(second_generation, 2);
        assert!(first_cancellation.is_cancelled());
        assert!(!second_cancellation.is_cancelled());

        let mut state = thread_state.lock().await;
        assert!(!state.clear_listener_if_generation(first_generation));
        assert!(state.listener_command_tx().is_some());
        assert!(!second_cancellation.is_cancelled());

        assert!(state.clear_listener_if_generation(second_generation));
        assert!(state.listener_command_tx().is_none());
        assert!(second_cancellation.is_cancelled());
    }

    #[tokio::test]
    async fn disconnect_removes_connection_from_every_thread() {
        let manager = ThreadStateManager::new();
        let first_thread = ThreadId::new("thread-1");
        let second_thread = ThreadId::new("thread-2");
        let first_connection = ConnectionId(1);
        let second_connection = ConnectionId(2);

        manager.connection_initialized(first_connection).await;
        manager.connection_initialized(second_connection).await;

        assert!(
            manager
                .subscribe_connection(first_thread.clone(), first_connection)
                .await
        );
        assert!(
            manager
                .subscribe_connection(first_thread.clone(), second_connection)
                .await
        );
        assert!(
            manager
                .subscribe_connection(second_thread.clone(), first_connection)
                .await
        );

        let empty_threads = manager.disconnect_connection(first_connection).await;

        assert_eq!(empty_threads, vec![second_thread.clone()]);
        assert_eq!(
            manager.subscribed_connection_ids(&first_thread).await,
            vec![second_connection]
        );
        assert!(manager
            .subscribed_connection_ids(&second_thread)
            .await
            .is_empty());
        assert!(manager
            .disconnect_connection(first_connection)
            .await
            .is_empty());
        assert!(
            manager
                .unsubscribe_connection(&first_thread, second_connection)
                .await
        );
        assert!(manager
            .subscribed_connection_ids(&first_thread)
            .await
            .is_empty());
    }

    #[tokio::test]
    async fn subscription_requires_a_live_connection_and_disconnect_closes_it() {
        let manager = ThreadStateManager::new();
        let thread_id = ThreadId::new("thread-1");
        let connection_id = ConnectionId(1);

        assert!(
            !manager
                .subscribe_connection(thread_id.clone(), connection_id)
                .await
        );

        manager.connection_initialized(connection_id).await;
        assert!(
            manager
                .subscribe_connection(thread_id.clone(), connection_id)
                .await
        );
        assert!(
            manager
                .subscribe_connection(thread_id.clone(), connection_id)
                .await
        );
        assert_eq!(
            manager.subscribed_connection_ids(&thread_id).await,
            vec![connection_id]
        );

        assert_eq!(
            manager.disconnect_connection(connection_id).await,
            vec![thread_id.clone()]
        );
        assert!(
            !manager
                .subscribe_connection(thread_id.clone(), connection_id)
                .await
        );
        assert!(manager
            .subscribed_connection_ids(&thread_id)
            .await
            .is_empty());
    }

    #[tokio::test]
    async fn resubscribe_invalidates_idle_unload_ticket() {
        let manager = ThreadStateManager::new();
        let thread_id = ThreadId::new("thread-idle-unload");
        let first_connection = ConnectionId(1);
        let second_connection = ConnectionId(2);
        manager.connection_initialized(first_connection).await;
        manager.connection_initialized(second_connection).await;
        assert!(
            manager
                .subscribe_connection(thread_id.clone(), first_connection)
                .await
        );
        assert!(
            manager
                .unsubscribe_connection(&thread_id, first_connection)
                .await
        );
        let first_ticket = manager
            .schedule_unload_if_idle(&thread_id)
            .await
            .expect("idle thread unload ticket");

        assert!(
            manager
                .subscribe_connection(thread_id.clone(), second_connection)
                .await
        );
        assert!(
            !manager
                .unload_ticket_is_current(&thread_id, first_ticket)
                .await
        );
        assert!(!manager.begin_unload(&thread_id, first_ticket).await);

        assert!(
            manager
                .unsubscribe_connection(&thread_id, second_connection)
                .await
        );
        let second_ticket = manager
            .schedule_unload_if_idle(&thread_id)
            .await
            .expect("rescheduled idle thread unload ticket");
        assert_ne!(second_ticket, first_ticket);
        assert!(manager.begin_unload(&thread_id, second_ticket).await);
        assert!(manager
            .thread_state_for_listener(thread_id.clone())
            .await
            .is_none());
        manager.cancel_unload(&thread_id, second_ticket).await;
        assert!(manager.thread_state_for_listener(thread_id).await.is_some());
    }

    #[test]
    fn resume_barrier_is_idempotent_and_only_flushes_after_its_owner_finishes() {
        let mut state = ThreadState::default();
        let barrier =
            ThreadResumeBarrier::new(ConnectionId(1), RequestId::String("resume-1".to_string()));

        assert!(state.begin_resume(barrier.clone()));
        assert!(!state.begin_resume(barrier.clone()));
        assert!(state.has_pending_resume());
        assert_eq!(state.finish_resume(&barrier), Some(true));
        assert!(!state.has_pending_resume());
        assert_eq!(state.finish_resume(&barrier), None);
    }

    #[tokio::test]
    async fn clear_all_listeners_cancels_every_generation() {
        let manager = ThreadStateManager::new();
        let first_state = manager.thread_state(ThreadId::new("thread-1")).await;
        let second_state = manager.thread_state(ThreadId::new("thread-2")).await;
        let first = first_state.lock().await.set_listener();
        let second = second_state.lock().await.set_listener();

        manager.clear_all_listeners().await;

        assert!(first.cancellation.is_cancelled());
        assert!(second.cancellation.is_cancelled());
        assert!(first_state.lock().await.listener_command_tx().is_none());
        assert!(second_state.lock().await.listener_command_tx().is_none());
        assert!(manager
            .thread_state_for_listener(ThreadId::new("thread-3"))
            .await
            .is_none());
    }

    #[tokio::test]
    async fn remove_thread_cancels_listener_and_clears_both_connection_indexes() {
        let manager = ThreadStateManager::new();
        let deleted_thread = ThreadId::new("thread-deleted");
        let retained_thread = ThreadId::new("thread-retained");
        let first_connection = ConnectionId(1);
        let second_connection = ConnectionId(2);

        manager.connection_initialized(first_connection).await;
        manager.connection_initialized(second_connection).await;
        assert!(
            manager
                .subscribe_connection(deleted_thread.clone(), first_connection)
                .await
        );
        assert!(
            manager
                .subscribe_connection(deleted_thread.clone(), second_connection)
                .await
        );
        assert!(
            manager
                .subscribe_connection(retained_thread.clone(), first_connection)
                .await
        );

        let state = manager.thread_state(deleted_thread.clone()).await;
        let registration = state.lock().await.set_listener();
        let cancellation = registration.cancellation.clone();
        let _commands = registration.command_rx;
        assert!(state.lock().await.begin_resume(ThreadResumeBarrier::new(
            first_connection,
            RequestId::String("resume-deleted".to_string()),
        )));

        manager.remove_thread(&deleted_thread).await;

        assert!(cancellation.is_cancelled());
        assert!(state.lock().await.listener_command_tx().is_none());
        assert!(!state.lock().await.has_pending_resume());
        assert!(manager
            .subscribed_connection_ids(&deleted_thread)
            .await
            .is_empty());
        assert_eq!(
            manager.disconnect_connection(first_connection).await,
            vec![retained_thread]
        );
        assert!(manager
            .disconnect_connection(second_connection)
            .await
            .is_empty());
    }
}
