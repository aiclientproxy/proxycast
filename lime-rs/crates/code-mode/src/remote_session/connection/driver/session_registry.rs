//! Remote session and active-cell registry.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use code_mode_protocol::{RuntimeCodeModeCellId, RuntimeCodeModeSessionDelegate};

pub(crate) struct SessionRegistry {
    sessions: std::sync::Mutex<HashMap<String, SessionRecord>>,
}

struct SessionRecord {
    delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    cells: HashSet<RuntimeCodeModeCellId>,
}

pub(crate) struct ClosedCell {
    pub(crate) delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    pub(crate) cell_id: RuntimeCodeModeCellId,
}

impl SessionRegistry {
    pub(crate) fn new() -> Self {
        Self {
            sessions: std::sync::Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn insert(
        &self,
        session_id: String,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    ) {
        self.sessions
            .lock()
            .expect("code mode session delegates poisoned")
            .insert(
                session_id,
                SessionRecord {
                    delegate,
                    cells: HashSet::new(),
                },
            );
    }

    pub(crate) fn remove(&self, session_id: &str) -> Vec<ClosedCell> {
        let Some(record) = self
            .sessions
            .lock()
            .expect("code mode session delegates poisoned")
            .remove(session_id)
        else {
            return Vec::new();
        };
        record
            .cells
            .into_iter()
            .map(|cell_id| ClosedCell {
                delegate: Arc::clone(&record.delegate),
                cell_id,
            })
            .collect()
    }

    pub(crate) fn register_cell(
        &self,
        session_id: &str,
        cell_id: RuntimeCodeModeCellId,
    ) -> Result<(), String> {
        let mut sessions = self
            .sessions
            .lock()
            .expect("code mode session delegates poisoned");
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| format!("code mode cell referenced unknown session {session_id}"))?;
        if !session.cells.insert(cell_id.clone()) {
            return Err(format!(
                "duplicate code mode cell id {cell_id} in session {session_id}"
            ));
        }
        Ok(())
    }

    pub(crate) fn close_cell(
        &self,
        session_id: &str,
        cell_id: &RuntimeCodeModeCellId,
    ) -> Option<ClosedCell> {
        let mut sessions = self
            .sessions
            .lock()
            .expect("code mode session delegates poisoned");
        let session = sessions.get_mut(session_id)?;
        if !session.cells.remove(cell_id) {
            return None;
        }
        Some(ClosedCell {
            delegate: Arc::clone(&session.delegate),
            cell_id: cell_id.clone(),
        })
    }

    pub(crate) fn delegate(
        &self,
        session_id: &str,
        cell_id: &RuntimeCodeModeCellId,
    ) -> Result<Arc<dyn RuntimeCodeModeSessionDelegate>, String> {
        let sessions = self
            .sessions
            .lock()
            .expect("code mode session delegates poisoned");
        let session = sessions
            .get(session_id)
            .ok_or_else(|| format!("delegate request referenced unknown session {session_id}"))?;
        if !session.cells.contains(cell_id) {
            return Err(format!(
                "delegate request referenced unknown cell {cell_id} in session {session_id}"
            ));
        }
        Ok(Arc::clone(&session.delegate))
    }

    pub(crate) fn drain(&self) -> Vec<ClosedCell> {
        let sessions = std::mem::take(
            &mut *self
                .sessions
                .lock()
                .expect("code mode session delegates poisoned"),
        );
        sessions
            .into_values()
            .flat_map(|record| {
                record.cells.into_iter().map(move |cell_id| ClosedCell {
                    delegate: Arc::clone(&record.delegate),
                    cell_id,
                })
            })
            .collect()
    }
}
