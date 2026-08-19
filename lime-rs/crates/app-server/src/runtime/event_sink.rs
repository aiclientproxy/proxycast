use super::RuntimeCoreError;
use app_server_protocol::AgentEvent;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeEvent {
    pub event_type: String,
    pub payload: serde_json::Value,
}

impl RuntimeEvent {
    pub fn new(event_type: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            event_type: event_type.into(),
            payload,
        }
    }
}

/// Owned runtime event channel used by App Server's background projection pump.
///
/// RuntimeCore owns persistence; App Server owns transport projection. Keeping the
/// receiver behind the hub lets a turn task publish after the request future has
/// returned without capturing a borrowed request callback.
#[derive(Clone)]
pub struct RuntimeEventHub {
    sender: mpsc::UnboundedSender<AgentEvent>,
    receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<AgentEvent>>>>,
}

impl RuntimeEventHub {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        Self {
            sender,
            receiver: Arc::new(Mutex::new(Some(receiver))),
        }
    }

    pub(crate) fn take_receiver(&self) -> Option<mpsc::UnboundedReceiver<AgentEvent>> {
        self.receiver
            .lock()
            .expect("runtime event hub mutex poisoned")
            .take()
    }

    pub(crate) fn publish(&self, event: AgentEvent) {
        let _ = self.sender.send(event);
    }
}

pub(super) type RuntimeEventCallback<'a> =
    dyn FnMut(AgentEvent) -> Result<(), RuntimeCoreError> + Send + 'a;

pub trait RuntimeEventSink: Send {
    fn emit(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError>;

    /// Persists internal CodeMode evidence without creating a RuntimeEvent or
    /// public ThreadItem.
    fn emit_code_cell_trace(
        &mut self,
        _event: tool_runtime::tool_lifecycle::CodeCellTraceEvent,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    fn emit_transient(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        self.emit(event)
    }

    /// Forward an event that was already persisted by an external runtime boundary.
    ///
    /// The default is intentionally a no-op for sinks that only collect backend events. The
    /// app-server sink overrides this to notify the current request without writing a duplicate
    /// event to the session log.
    fn emit_preappended(&mut self, _event: AgentEvent) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    /// Forward an event that the runtime already persisted and identifies by canonical id.
    fn emit_preappended_by_id(&mut self, event_id: &str) -> Result<(), RuntimeCoreError> {
        Err(RuntimeCoreError::Backend(format!(
            "runtime event sink cannot resolve preappended event: {event_id}"
        )))
    }
}
