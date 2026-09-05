//! In-flight request tracking owner.

use std::collections::HashMap;
use std::sync::Mutex;

use code_mode_protocol::host::MAX_IN_FLIGHT_REQUESTS;
use tokio_util::sync::CancellationToken;

use super::types::PendingRequest;

pub(crate) struct RequestTracker {
    pending: Mutex<HashMap<u64, PendingRequest>>,
    caller_cancellation_watchers: Mutex<HashMap<u64, CancellationToken>>,
}

impl RequestTracker {
    pub(crate) fn new() -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            caller_cancellation_watchers: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn register(&self, id: u64, request: PendingRequest) -> Result<(), String> {
        let mut pending = self
            .pending
            .lock()
            .expect("code mode pending requests poisoned");
        if pending.len() >= MAX_IN_FLIGHT_REQUESTS {
            return Err(format!(
                "code mode host request limit {MAX_IN_FLIGHT_REQUESTS} exceeded"
            ));
        }
        if pending.insert(id, request).is_some() {
            return Err(format!("duplicate code mode host request id {id}"));
        }
        Ok(())
    }

    pub(crate) fn remove(&self, id: u64) -> Option<PendingRequest> {
        self.pending
            .lock()
            .expect("code mode pending requests poisoned")
            .remove(&id)
    }

    pub(crate) fn contains(&self, id: u64) -> bool {
        self.pending
            .lock()
            .expect("code mode pending requests poisoned")
            .contains_key(&id)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.pending
            .lock()
            .expect("code mode pending requests poisoned")
            .len()
    }

    pub(crate) fn register_caller_watcher(&self, id: u64, finished: CancellationToken) {
        self.caller_cancellation_watchers
            .lock()
            .expect("code mode caller cancellation watchers poisoned")
            .insert(id, finished);
    }

    pub(crate) fn finish_caller_watcher(&self, id: u64) {
        if let Some(token) = self
            .caller_cancellation_watchers
            .lock()
            .expect("code mode caller cancellation watchers poisoned")
            .remove(&id)
        {
            token.cancel();
        }
    }

    pub(crate) fn fail_all(&self, reason: &str) {
        let pending = std::mem::take(
            &mut *self
                .pending
                .lock()
                .expect("code mode pending requests poisoned"),
        );
        for request in pending.into_values() {
            request.fail(reason.to_string());
        }
        for token in self
            .caller_cancellation_watchers
            .lock()
            .expect("code mode caller cancellation watchers poisoned")
            .drain()
            .map(|(_, token)| token)
        {
            token.cancel();
        }
    }
}

impl PendingRequest {
    pub(crate) fn fail(self, reason: String) {
        match self {
            Self::Standard(sender) => {
                let _ = sender.send(Err(reason));
            }
            Self::Execute {
                started, initial, ..
            } => {
                let _ = started.send(Err(reason.clone()));
                let _ = initial.send(Err(reason));
            }
            Self::ExecuteStarted { initial, .. } => {
                let _ = initial.send(Err(reason));
            }
        }
    }
}
