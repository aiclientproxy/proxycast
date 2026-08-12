use crate::code_mode::{RuntimeCodeModeService, RuntimeCodeModeServiceFactory};
use std::sync::Arc;
use tool_runtime::code_mode::RuntimeCodeModeSessionHandle;

pub(super) struct RuntimeSessionResources {
    thread_id: Arc<str>,
    code_mode: Option<RuntimeCodeModeService>,
}

impl RuntimeSessionResources {
    pub(super) fn new(
        thread_id: impl Into<Arc<str>>,
        code_mode_factory: Option<&RuntimeCodeModeServiceFactory>,
    ) -> Self {
        let thread_id = thread_id.into();
        let code_mode = code_mode_factory.and_then(|factory| factory.create(&thread_id).ok());
        Self {
            thread_id,
            code_mode,
        }
    }

    pub(super) fn thread_id(&self) -> &str {
        &self.thread_id
    }

    pub(super) fn code_mode_session(&self) -> Option<RuntimeCodeModeSessionHandle> {
        self.code_mode
            .as_ref()
            .map(RuntimeCodeModeService::session_handle)
    }

    pub(super) async fn interrupt_code_mode(&self) {
        if let Some(code_mode) = self.code_mode.as_ref() {
            code_mode.interrupt_active_cells().await;
        }
    }

    pub(super) async fn shutdown_code_mode(&self) -> Result<(), String> {
        match self.code_mode.as_ref() {
            Some(code_mode) => code_mode.shutdown().await,
            None => Ok(()),
        }
    }
}
