use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResidentAgent {
    session_id: String,
    thread_id: String,
}

#[derive(Debug, Default)]
struct RootResidency {
    residents: VecDeque<ResidentAgent>,
    pending: HashMap<String, ResidentAgent>,
    lost_sessions: std::collections::HashSet<String>,
}

#[derive(Clone, Debug)]
pub(in crate::runtime) struct AgentResidency {
    roots: Arc<Mutex<HashMap<String, RootResidency>>>,
    max_residents: usize,
}

#[derive(Debug)]
pub(in crate::runtime) struct AgentResidencySlot {
    residency: AgentResidency,
    root_thread_id: String,
    agent: ResidentAgent,
    active: bool,
}

impl Default for AgentResidency {
    fn default() -> Self {
        Self {
            roots: Arc::new(Mutex::new(HashMap::new())),
            max_residents: 3,
        }
    }
}

impl AgentResidency {
    pub(in crate::runtime) fn reserve(
        &self,
        root_thread_id: &str,
        session_id: &str,
        thread_id: &str,
    ) -> Result<AgentResidencySlot, Option<(String, String)>> {
        let agent = ResidentAgent {
            session_id: session_id.to_string(),
            thread_id: thread_id.to_string(),
        };
        let mut roots = self.roots.lock().expect("agent residency mutex poisoned");
        let root = roots.entry(root_thread_id.to_string()).or_default();
        if let Some(index) = root
            .residents
            .iter()
            .position(|resident| resident.session_id == session_id)
        {
            let resident = root
                .residents
                .remove(index)
                .expect("resident index must exist");
            root.residents.push_back(resident);
            return Ok(AgentResidencySlot {
                residency: self.clone(),
                root_thread_id: root_thread_id.to_string(),
                agent,
                active: false,
            });
        }
        if root.residents.len().saturating_add(root.pending.len()) < self.max_residents {
            root.pending.insert(session_id.to_string(), agent.clone());
            return Ok(AgentResidencySlot {
                residency: self.clone(),
                root_thread_id: root_thread_id.to_string(),
                agent,
                active: true,
            });
        }
        Err(root
            .residents
            .pop_front()
            .map(|candidate| (candidate.session_id, candidate.thread_id)))
    }

    pub(in crate::runtime) fn restore_candidate(
        &self,
        root_thread_id: &str,
        session_id: String,
        thread_id: String,
    ) {
        let mut roots = self.roots.lock().expect("agent residency mutex poisoned");
        let root = roots.entry(root_thread_id.to_string()).or_default();
        root.residents
            .retain(|resident| resident.session_id != session_id);
        root.residents.push_back(ResidentAgent {
            session_id,
            thread_id,
        });
    }

    pub(in crate::runtime) fn forget(&self, root_thread_id: &str, session_id: &str) {
        let mut roots = self.roots.lock().expect("agent residency mutex poisoned");
        let Some(root) = roots.get_mut(root_thread_id) else {
            return;
        };
        root.pending.remove(session_id);
        root.lost_sessions.remove(session_id);
        root.residents
            .retain(|resident| resident.session_id != session_id);
        if root.pending.is_empty() && root.residents.is_empty() && root.lost_sessions.is_empty() {
            roots.remove(root_thread_id);
        }
    }

    pub(in crate::runtime) fn mark_lost(&self, root_thread_id: &str, session_id: &str) {
        let mut roots = self.roots.lock().expect("agent residency mutex poisoned");
        roots
            .entry(root_thread_id.to_string())
            .or_default()
            .lost_sessions
            .insert(session_id.to_string());
    }

    pub(in crate::runtime) fn is_lost(&self, root_thread_id: &str, session_id: &str) -> bool {
        self.roots
            .lock()
            .expect("agent residency mutex poisoned")
            .get(root_thread_id)
            .is_some_and(|root| root.lost_sessions.contains(session_id))
    }
}

impl AgentResidencySlot {
    pub(in crate::runtime) fn commit(mut self) {
        if !self.active {
            return;
        }
        let mut roots = self
            .residency
            .roots
            .lock()
            .expect("agent residency mutex poisoned");
        let root = roots.entry(self.root_thread_id.clone()).or_default();
        root.pending.remove(&self.agent.session_id);
        root.residents
            .retain(|resident| resident.session_id != self.agent.session_id);
        root.residents.push_back(self.agent.clone());
        self.active = false;
    }
}

impl Drop for AgentResidencySlot {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut roots = self
            .residency
            .roots
            .lock()
            .expect("agent residency mutex poisoned");
        if let Some(root) = roots.get_mut(&self.root_thread_id) {
            root.pending.remove(&self.agent.session_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AgentResidency;

    #[test]
    fn lru_is_root_scoped_and_pending_slots_are_released() {
        let residency = AgentResidency {
            max_residents: 1,
            ..AgentResidency::default()
        };
        residency
            .reserve("root-a", "session-a", "thread-a")
            .expect("first slot")
            .commit();
        let candidate = residency
            .reserve("root-a", "session-b", "thread-b")
            .expect_err("root-a should need eviction");
        assert_eq!(candidate, Some(("session-a".into(), "thread-a".into())));
        assert!(residency.reserve("root-b", "session-c", "thread-c").is_ok());
    }

    #[test]
    fn forgetting_another_session_preserves_lost_tombstones() {
        let residency = AgentResidency::default();
        residency.mark_lost("root", "lost-session");
        residency
            .reserve("root", "completed-session", "completed-thread")
            .expect("resident slot")
            .commit();

        residency.forget("root", "completed-session");

        assert!(residency.is_lost("root", "lost-session"));
    }
}
