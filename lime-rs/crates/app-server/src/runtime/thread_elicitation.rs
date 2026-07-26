use super::{RuntimeCore, RuntimeCoreError};

impl RuntimeCore {
    pub(crate) fn increment_thread_elicitation(
        &self,
        thread_id: &str,
    ) -> Result<i64, RuntimeCoreError> {
        let thread_id = required_thread_id(thread_id)?;
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        ensure_thread_loaded(&state, thread_id)?;
        let count = state
            .thread_elicitation_counts
            .entry(thread_id.to_string())
            .or_default();
        *count = count.checked_add(1).ok_or_else(|| {
            RuntimeCoreError::Backend("out-of-band elicitation count overflowed".to_string())
        })?;
        Ok(*count)
    }

    pub(crate) fn decrement_thread_elicitation(
        &self,
        thread_id: &str,
    ) -> Result<i64, RuntimeCoreError> {
        let thread_id = required_thread_id(thread_id)?;
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        ensure_thread_loaded(&state, thread_id)?;
        let Some(count) = state.thread_elicitation_counts.get_mut(thread_id) else {
            return Err(RuntimeCoreError::InvalidRequest(
                "out-of-band elicitation count is already zero".to_string(),
            ));
        };
        *count -= 1;
        let count = *count;
        if count == 0 {
            state.thread_elicitation_counts.remove(thread_id);
        }
        Ok(count)
    }
}

fn required_thread_id(thread_id: &str) -> Result<&str, RuntimeCoreError> {
    let thread_id = thread_id.trim();
    if thread_id.is_empty() {
        return Err(RuntimeCoreError::InvalidRequest(
            "threadId is required for elicitation accounting".to_string(),
        ));
    }
    Ok(thread_id)
}

fn ensure_thread_loaded(
    state: &super::RuntimeCoreState,
    thread_id: &str,
) -> Result<(), RuntimeCoreError> {
    if state
        .sessions
        .values()
        .any(|stored| stored.session.thread_id == thread_id)
    {
        return Ok(());
    }
    Err(RuntimeCoreError::InvalidRequest(format!(
        "thread not found: {thread_id}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MockBackend, ProjectionStore};
    use agent_protocol::ThreadId;
    use app_server_protocol::AgentSessionStartParams;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn loaded_core() -> RuntimeCore {
        let core = RuntimeCore::default();
        core.start_session(AgentSessionStartParams {
            session_id: Some("session-elicitation".to_string()),
            thread_id: Some("thread-elicitation".to_string()),
            app_id: "test".to_string(),
            workspace_id: None,
            business_object_ref: None,
            locale: None,
        })
        .expect("start elicitation test thread");
        core
    }

    #[test]
    fn count_is_thread_local_and_rejects_underflow() {
        let core = loaded_core();

        assert_eq!(
            core.increment_thread_elicitation("thread-elicitation")
                .expect("first increment"),
            1
        );
        assert_eq!(
            core.increment_thread_elicitation("thread-elicitation")
                .expect("second increment"),
            2
        );
        assert_eq!(
            core.decrement_thread_elicitation("thread-elicitation")
                .expect("first decrement"),
            1
        );
        assert_eq!(
            core.decrement_thread_elicitation("thread-elicitation")
                .expect("second decrement"),
            0
        );
        assert!(matches!(
            core.decrement_thread_elicitation("thread-elicitation"),
            Err(RuntimeCoreError::InvalidRequest(message))
                if message == "out-of-band elicitation count is already zero"
        ));
    }

    #[test]
    fn increment_rejects_overflow() {
        let core = loaded_core();
        core.state
            .lock()
            .expect("runtime core state mutex poisoned")
            .thread_elicitation_counts
            .insert("thread-elicitation".to_string(), i64::MAX);

        assert!(matches!(
            core.increment_thread_elicitation("thread-elicitation"),
            Err(RuntimeCoreError::Backend(message))
                if message == "out-of-band elicitation count overflowed"
        ));
    }

    #[tokio::test]
    async fn archive_clears_the_loaded_thread_registration() {
        let temp = TempDir::new().expect("elicitation archive temp dir");
        let projection = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("elicitation archive projection store");
        let core = RuntimeCore::with_backend(Arc::new(MockBackend))
            .with_projection_store(Arc::new(projection));
        core.start_session(AgentSessionStartParams {
            session_id: Some("session-elicitation-archive".to_string()),
            thread_id: Some("thread-elicitation-archive".to_string()),
            app_id: "test".to_string(),
            workspace_id: None,
            business_object_ref: None,
            locale: None,
        })
        .expect("start archived elicitation test thread");
        core.increment_thread_elicitation("thread-elicitation-archive")
            .expect("register elicitation before archive");

        assert!(core
            .archive_thread(ThreadId::new("thread-elicitation-archive"))
            .await
            .expect("archive elicitation test thread"));
        assert!(!core
            .state
            .lock()
            .expect("runtime core state mutex poisoned")
            .thread_elicitation_counts
            .contains_key("thread-elicitation-archive"));
    }
}
