use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Limits concurrently running V2 child turns per durable root tree.
/// The root turn is not counted here, so the default total capacity of four
/// leaves three child execution slots.
#[derive(Debug)]
pub(in crate::runtime) struct AgentExecutionLimiter {
    state: Mutex<AgentExecutionState>,
    max_threads: Mutex<usize>,
}

#[derive(Debug, Default)]
struct AgentExecutionState {
    active_by_root: HashMap<String, usize>,
    active_by_session: HashMap<String, String>,
    pending_by_session: HashMap<String, String>,
}

pub(in crate::runtime) struct AgentExecutionGuard {
    limiter: Arc<AgentExecutionLimiter>,
    session_id: String,
}

pub(in crate::runtime) struct AgentExecutionReservation {
    limiter: Arc<AgentExecutionLimiter>,
    session_id: String,
    root_thread_id: String,
    owns_slot: bool,
}

impl AgentExecutionLimiter {
    pub(in crate::runtime) fn new(max_threads: usize) -> Self {
        Self {
            state: Mutex::new(AgentExecutionState::default()),
            max_threads: Mutex::new(max_threads),
        }
    }

    /// Reserves before a gateway commits TriggerTurn work. The turn admission
    /// later claims this exact slot by session id.
    pub(in crate::runtime) fn reserve_for_session(
        self: &Arc<Self>,
        root_thread_id: &str,
        session_id: &str,
    ) -> Result<AgentExecutionReservation, usize> {
        let max_threads = self.max_threads();
        let mut state = self.state.lock().expect("agent execution mutex poisoned");
        if let Some(existing_root) = state.pending_by_session.get(session_id) {
            debug_assert_eq!(existing_root, root_thread_id);
            return Ok(AgentExecutionReservation {
                limiter: Arc::clone(self),
                session_id: session_id.to_string(),
                root_thread_id: root_thread_id.to_string(),
                owns_slot: false,
            });
        }
        reserve_root_slot(&mut state, root_thread_id, max_threads)?;
        state
            .pending_by_session
            .insert(session_id.to_string(), root_thread_id.to_string());
        Ok(AgentExecutionReservation {
            limiter: Arc::clone(self),
            session_id: session_id.to_string(),
            root_thread_id: root_thread_id.to_string(),
            owns_slot: true,
        })
    }

    /// Claims a gateway reservation or atomically reserves for recovery and
    /// direct turn starts that did not originate from a live gateway call.
    pub(in crate::runtime) fn claim_for_session(
        self: &Arc<Self>,
        root_thread_id: &str,
        session_id: &str,
    ) -> Result<AgentExecutionGuard, usize> {
        let max_threads = self.max_threads();
        let mut state = self.state.lock().expect("agent execution mutex poisoned");
        let claimed = state
            .pending_by_session
            .remove(session_id)
            .is_some_and(|reserved_root| reserved_root == root_thread_id);
        if !claimed {
            reserve_root_slot(&mut state, root_thread_id, max_threads)?;
        }
        state
            .active_by_session
            .insert(session_id.to_string(), root_thread_id.to_string());
        Ok(AgentExecutionGuard {
            limiter: Arc::clone(self),
            session_id: session_id.to_string(),
        })
    }

    /// Claims an existing gateway reservation without re-reading the durable
    /// agent identity on the turn-admission hot path.
    pub(in crate::runtime) fn claim_reserved_session(
        self: &Arc<Self>,
        session_id: &str,
    ) -> Option<AgentExecutionGuard> {
        let mut state = self.state.lock().expect("agent execution mutex poisoned");
        let root_thread_id = state.pending_by_session.remove(session_id)?;
        state
            .active_by_session
            .insert(session_id.to_string(), root_thread_id);
        Some(AgentExecutionGuard {
            limiter: Arc::clone(self),
            session_id: session_id.to_string(),
        })
    }

    fn max_threads(&self) -> usize {
        *self
            .max_threads
            .lock()
            .expect("agent execution limit mutex poisoned")
    }

    pub(in crate::runtime) fn release_session(&self, session_id: &str) {
        let mut state = self.state.lock().expect("agent execution mutex poisoned");
        let Some(root_thread_id) = state.active_by_session.remove(session_id) else {
            return;
        };
        let Some(active) = state.active_by_root.get_mut(&root_thread_id) else {
            debug_assert!(false, "agent execution guard has no reservation");
            return;
        };
        *active -= 1;
        if *active == 0 {
            state.active_by_root.remove(&root_thread_id);
        }
    }

    fn release_unclaimed(&self, session_id: &str, root_thread_id: &str) {
        let mut state = self.state.lock().expect("agent execution mutex poisoned");
        let Some(reserved_root) = state.pending_by_session.remove(session_id) else {
            return;
        };
        if reserved_root != root_thread_id {
            return;
        }
        if let Some(active) = state.active_by_root.get_mut(root_thread_id) {
            *active = active.saturating_sub(1);
            if *active == 0 {
                state.active_by_root.remove(root_thread_id);
            }
        }
    }
}

impl Drop for AgentExecutionGuard {
    fn drop(&mut self) {
        self.limiter.release_session(&self.session_id);
    }
}

impl Drop for AgentExecutionReservation {
    fn drop(&mut self) {
        if !self.owns_slot {
            return;
        }
        self.limiter
            .release_unclaimed(&self.session_id, &self.root_thread_id);
    }
}

fn reserve_root_slot(
    state: &mut AgentExecutionState,
    root_thread_id: &str,
    max_threads: usize,
) -> Result<(), usize> {
    let active = state
        .active_by_root
        .entry(root_thread_id.to_string())
        .or_default();
    if *active >= max_threads {
        return Err(max_threads);
    }
    *active += 1;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::AgentExecutionLimiter;
    use std::sync::Arc;

    #[test]
    fn reservations_are_root_scoped_and_claimed_once() {
        let limiter = Arc::new(AgentExecutionLimiter::new(1));
        let pending = limiter
            .reserve_for_session("root-a", "child-a")
            .expect("pending reservation");
        assert!(limiter.reserve_for_session("root-a", "child-b").is_err());
        assert!(limiter.reserve_for_session("root-b", "child-c").is_ok());

        let guard = limiter
            .claim_for_session("root-a", "child-a")
            .expect("claim reservation");
        drop(pending);
        assert!(limiter.reserve_for_session("root-a", "child-d").is_err());
        drop(guard);
        assert!(limiter.reserve_for_session("root-a", "child-d").is_ok());
    }

    #[test]
    fn reserved_session_claim_does_not_consume_a_second_slot() {
        let limiter = Arc::new(AgentExecutionLimiter::new(1));
        let pending = limiter
            .reserve_for_session("root-a", "child-a")
            .expect("pending reservation");
        let guard = limiter
            .claim_reserved_session("child-a")
            .expect("claim pending session");
        drop(pending);
        assert!(limiter.reserve_for_session("root-a", "child-b").is_err());
        drop(guard);
        assert!(limiter.reserve_for_session("root-a", "child-b").is_ok());
    }
}
