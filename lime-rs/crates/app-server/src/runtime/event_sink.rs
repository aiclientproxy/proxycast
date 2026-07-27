use super::{RuntimeCoreError, RuntimeEvent};
use app_server_protocol::AgentEvent;

pub trait RuntimeEventSink: Send {
    fn emit(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError>;

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
}
