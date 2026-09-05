//! Per-session state owned by the gRPC host.

use super::waits::WaitControl;
use code_mode_protocol::grpc as proto;
use code_mode_protocol::RuntimeCodeModeSessionHandle;
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_util::sync::CancellationToken;
use tonic::Status;

pub(crate) struct GrpcSession {
    pub(super) id: String,
    pub(crate) peer: crate::peer::PeerState,
    pub(super) closed: AtomicBool,
    pub(super) runtime: Mutex<Option<RuntimeCodeModeSessionHandle>>,
    pub(super) subscribers: Mutex<Vec<Subscriber>>,
    pub(super) session_events: Mutex<Option<mpsc::Sender<Result<proto::SessionEvent, Status>>>>,
    pub(super) event_shutdown: CancellationToken,
    pub(super) pending: Mutex<HashMap<String, oneshot::Sender<Result<Value, String>>>>,
    pub(crate) pending_notifications: Mutex<HashMap<String, oneshot::Sender<Result<(), String>>>>,
    pub(super) next_subscriber: AtomicU64,
    pub(super) waits: Mutex<HashMap<String, WaitControl>>,
    pub(super) cancelled_waits: Mutex<HashSet<String>>,
    pub(super) pending_executions: Mutex<HashSet<String>>,
    pub(super) seen_executions: Mutex<BoundedIds>,
    pub(super) execution_ids: Mutex<HashMap<String, String>>,
    pub(super) tool_call_sequences: Mutex<HashMap<String, u64>>,
    pub(super) active_cells: Mutex<HashMap<String, OwnedSemaphorePermit>>,
}

pub(super) struct Subscriber {
    pub(super) tool_names: Vec<proto::ToolName>,
    pub(super) sender: mpsc::Sender<Result<proto::ToolCall, Status>>,
}

const MAX_RECENT_EXECUTION_IDS: usize = 4096;

#[derive(Default)]
pub(super) struct BoundedIds {
    ids: HashSet<String>,
    order: VecDeque<String>,
}

impl BoundedIds {
    fn remember(&mut self, id: String) -> bool {
        if !self.ids.insert(id.clone()) {
            return false;
        }
        self.order.push_back(id);
        while self.order.len() > MAX_RECENT_EXECUTION_IDS {
            if let Some(expired) = self.order.pop_front() {
                self.ids.remove(&expired);
            }
        }
        true
    }
}

impl GrpcSession {
    pub(super) async fn runtime(&self) -> Result<RuntimeCodeModeSessionHandle, Status> {
        if self.closed.load(Ordering::Acquire) || self.peer.is_disconnected() {
            return Err(Status::cancelled("code-mode session is closed"));
        }
        self.runtime
            .lock()
            .await
            .clone()
            .ok_or_else(|| Status::failed_precondition("code-mode session is not ready"))
    }

    pub(super) async fn register_execution(
        &self,
        cell_id: &str,
        execution_id: &str,
        permit: OwnedSemaphorePermit,
    ) -> Result<(), String> {
        if self.closed.load(Ordering::Acquire) {
            return Err("code-mode session is closed".to_string());
        }
        if self.pending_executions.lock().await.remove(execution_id) == false {
            return Err("code-mode execution was abandoned or already admitted".to_string());
        }
        self.execution_ids
            .lock()
            .await
            .insert(cell_id.to_string(), execution_id.to_string());
        self.tool_call_sequences
            .lock()
            .await
            .insert(cell_id.to_string(), 0);
        self.active_cells
            .lock()
            .await
            .insert(cell_id.to_string(), permit);
        Ok(())
    }

    pub(super) async fn reserve_execution(&self, execution_id: &str) -> Result<(), Status> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Status::cancelled("code-mode session is closed"));
        }
        let mut seen = self.seen_executions.lock().await;
        if !seen.remember(execution_id.to_string()) {
            return Err(Status::already_exists(
                "code-mode execution ID was already used",
            ));
        }
        drop(seen);
        self.pending_executions
            .lock()
            .await
            .insert(execution_id.to_string());
        Ok(())
    }

    pub(super) async fn abandon_execution(&self, execution_id: &str) {
        self.pending_executions.lock().await.remove(execution_id);
    }

    pub(crate) async fn execution_id_for_cell(&self, cell_id: &str) -> Option<String> {
        self.execution_ids.lock().await.get(cell_id).cloned()
    }

    pub(super) async fn take_execution(&self, cell_id: &str) -> (String, u64) {
        let execution_id = self
            .execution_ids
            .lock()
            .await
            .remove(cell_id)
            .unwrap_or_else(|| cell_id.to_string());
        let sequence = self
            .tool_call_sequences
            .lock()
            .await
            .remove(cell_id)
            .unwrap_or_default();
        self.active_cells.lock().await.remove(cell_id);
        (execution_id, sequence)
    }

    pub(super) async fn next_tool_call_sequence(&self, cell_id: &str) -> Result<u64, String> {
        let mut sequences = self.tool_call_sequences.lock().await;
        let sequence = sequences
            .get_mut(cell_id)
            .ok_or_else(|| "code-mode cell has no owning execution".to_string())?;
        *sequence = sequence
            .checked_add(1)
            .ok_or_else(|| "code-mode tool-call sequence exhausted".to_string())?;
        Ok(*sequence)
    }

    pub(crate) async fn publish_event(
        &self,
        event: proto::session_event::Event,
    ) -> Result<(), String> {
        let sender = self
            .session_events
            .lock()
            .await
            .clone()
            .ok_or_else(|| "code-mode session event stream is closed".to_string())?;
        sender
            .send(Ok(proto::SessionEvent { event: Some(event) }))
            .await
            .map_err(|error| format!("code-mode session event stream closed: {error}"))
    }

    pub(super) async fn acknowledge_notification(
        &self,
        notification_id: &str,
    ) -> Result<(), Status> {
        let sender = self
            .pending_notifications
            .lock()
            .await
            .remove(notification_id)
            .ok_or_else(|| Status::not_found("code-mode notification not found"))?;
        sender
            .send(Ok(()))
            .map_err(|_| Status::aborted("code-mode notification already completed"))
    }

    pub(super) async fn close_pending(&self, reason: &str) {
        self.peer.fail(reason.to_string());
        self.closed.store(true, Ordering::Release);
        let pending = std::mem::take(&mut *self.pending.lock().await);
        for (invocation_id, sender) in pending {
            let _ = sender.send(Err(reason.to_string()));
            super::events::tool_call_cancelled(self, invocation_id).await;
        }
        let notifications = std::mem::take(&mut *self.pending_notifications.lock().await);
        for (notification_id, sender) in notifications {
            let _ = sender.send(Err(reason.to_string()));
            super::events::notification_cancelled(self, notification_id).await;
        }
        for control in std::mem::take(&mut *self.waits.lock().await).into_values() {
            control.cancellation.cancel();
            control.retired.notify_waiters();
        }
        self.cancelled_waits.lock().await.clear();
        self.pending_executions.lock().await.clear();
        self.event_shutdown.cancel();
        self.session_events.lock().await.take();
    }

    pub(super) async fn clear_execution_state(&self) {
        self.execution_ids.lock().await.clear();
        self.tool_call_sequences.lock().await.clear();
        self.active_cells.lock().await.clear();
    }
}
