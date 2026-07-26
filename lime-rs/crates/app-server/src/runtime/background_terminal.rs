use super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::protocol::v2::ThreadBackgroundTerminal;

impl RuntimeCore {
    pub fn list_background_terminals(
        &self,
        thread_id: &str,
    ) -> Result<Vec<ThreadBackgroundTerminal>, RuntimeCoreError> {
        self.ensure_background_terminal_thread_loaded(thread_id)?;
        self.background_terminal_server()?
            .list_background_terminals(thread_id)
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))
    }

    pub fn terminate_background_terminal(
        &self,
        thread_id: &str,
        process_id: u64,
    ) -> Result<bool, RuntimeCoreError> {
        self.ensure_background_terminal_thread_loaded(thread_id)?;
        self.background_terminal_server()?
            .terminate_background_terminal(thread_id, process_id)
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))
    }

    pub fn clean_background_terminals(&self, thread_id: &str) -> Result<(), RuntimeCoreError> {
        self.ensure_background_terminal_thread_loaded(thread_id)?;
        self.background_terminal_server()?
            .clean_background_terminals(thread_id)
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))
    }

    fn ensure_background_terminal_thread_loaded(
        &self,
        thread_id: &str,
    ) -> Result<(), RuntimeCoreError> {
        if self.loaded_session_id_for_thread(thread_id).is_none() {
            return Err(RuntimeCoreError::InvalidRequest(format!(
                "thread not found: {thread_id}"
            )));
        }
        Ok(())
    }

    fn background_terminal_server(
        &self,
    ) -> Result<crate::execution_process::ExecutionProcessServer, RuntimeCoreError> {
        self.execution_process_server().ok_or_else(|| {
            RuntimeCoreError::Backend("local execution environment is not configured".to_string())
        })
    }
}
