use super::actor::RuntimeSessionActor;
use super::{RuntimeSessionHandle, RuntimeSessionLoopError, RuntimeSessionSnapshot};
use crate::code_mode::RuntimeCodeModeServiceFactory;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{watch, Mutex};

#[derive(Clone, Default)]
pub struct RuntimeSessionRegistry {
    sessions: Arc<Mutex<HashMap<String, RuntimeSessionHandle>>>,
    code_mode_factory: Option<RuntimeCodeModeServiceFactory>,
}

impl RuntimeSessionRegistry {
    pub fn with_code_mode(code_mode_factory: RuntimeCodeModeServiceFactory) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            code_mode_factory: Some(code_mode_factory),
        }
    }

    pub async fn get_existing(&self, session_id: &str) -> Option<RuntimeSessionHandle> {
        let sessions = self.sessions.lock().await;
        sessions.get(session_id).cloned()
    }

    pub async fn get_or_create(
        &self,
        session_id: &str,
        thread_id: &str,
    ) -> Result<RuntimeSessionHandle, RuntimeSessionLoopError> {
        if session_id.trim().is_empty() || thread_id.trim().is_empty() {
            return Err(RuntimeSessionLoopError::InvalidTask(
                "runtime session actor requires canonical session_id and thread_id".to_string(),
            ));
        }
        let mut sessions = self.sessions.lock().await;
        if let Some(handle) = sessions.get(session_id) {
            if handle.thread_id() != thread_id {
                return Err(RuntimeSessionLoopError::InvalidTask(format!(
                    "runtime session identity mismatch: session `{session_id}` is bound to thread `{}`",
                    handle.thread_id()
                )));
            }
            return Ok(handle.clone());
        }
        let handle = RuntimeSessionActor::spawn(
            session_id.to_string(),
            thread_id.to_string(),
            self.code_mode_factory.as_ref(),
        );
        sessions.insert(session_id.to_string(), handle.clone());
        Ok(handle)
    }

    pub async fn shutdown(&self, session_id: &str) -> Result<(), RuntimeSessionLoopError> {
        let mut sessions = self.sessions.lock().await;
        let Some(handle) = sessions.get(session_id).cloned() else {
            return Ok(());
        };
        let result = handle.shutdown().await;
        sessions.remove(session_id);
        if let Err(error) = result {
            return Err(error);
        }
        Ok(())
    }

    pub async fn notify_inter_agent_communication(
        &self,
        session_id: &str,
        input: super::RuntimeSessionInterAgentInput,
    ) -> Result<bool, RuntimeSessionLoopError> {
        let handle = {
            let sessions = self.sessions.lock().await;
            sessions.get(session_id).cloned()
        };
        let Some(handle) = handle else {
            return Ok(false);
        };
        handle.notify_inter_agent_communication(input).await?;
        Ok(true)
    }

    pub async fn subscribe_input_activity(
        &self,
        session_id: &str,
    ) -> Result<
        Option<(
            watch::Receiver<super::RuntimeSessionInputActivity>,
            Option<super::RuntimeSessionInputActivity>,
        )>,
        RuntimeSessionLoopError,
    > {
        let handle = {
            let sessions = self.sessions.lock().await;
            sessions.get(session_id).cloned()
        };
        let Some(handle) = handle else {
            return Ok(None);
        };
        handle.subscribe_input_activity().await.map(Some)
    }

    pub async fn snapshot(
        &self,
        session_id: &str,
    ) -> Result<Option<RuntimeSessionSnapshot>, RuntimeSessionLoopError> {
        let handle = {
            let sessions = self.sessions.lock().await;
            sessions.get(session_id).cloned()
        };
        let Some(handle) = handle else {
            return Ok(None);
        };
        handle.snapshot().await.map(Some)
    }
}
