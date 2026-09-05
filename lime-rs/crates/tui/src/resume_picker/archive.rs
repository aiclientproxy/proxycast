use super::PickerState;

/// Tracks an in-flight archive request so repeated key presses are idempotent.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) enum ArchiveState {
    #[default]
    Idle,
    Pending {
        thread_id: String,
    },
    Restoring {
        thread_id: String,
    },
}

impl PickerState {
    pub(super) fn archive_shortcut_available(&self) -> bool {
        matches!(self.archive_state, ArchiveState::Idle) && self.selected_thread_id().is_some()
    }

    pub(super) fn request_archive_for_selected_session(&mut self) -> Option<String> {
        if !self.archive_shortcut_available() {
            return None;
        }
        let thread_id = self.selected_thread_id()?.to_string();
        self.archive_state = ArchiveState::Pending {
            thread_id: thread_id.clone(),
        };
        self.status_message = None;
        Some(thread_id)
    }

    pub(super) fn request_unarchive_for_selected_session(&mut self) -> Option<String> {
        if !matches!(self.archive_state, ArchiveState::Idle) {
            return None;
        }
        let thread_id = self.selected_thread_id()?.to_string();
        self.archive_state = ArchiveState::Restoring {
            thread_id: thread_id.clone(),
        };
        self.status_message = None;
        Some(thread_id)
    }

    pub(super) fn handle_archive_result(&mut self, thread_id: String, result: anyhow::Result<()>) {
        if self.archive_state
            != (ArchiveState::Pending {
                thread_id: thread_id.clone(),
            })
        {
            return;
        }
        self.archive_state = ArchiveState::Idle;

        if let Err(error) = result {
            self.status_message = Some(format!("Failed to archive session: {error}"));
            return;
        }

        self.threads.retain(|thread| thread.id != thread_id);
        self.transcript_previews.remove(&thread_id);
        self.transcripts.remove(&thread_id);
        if self.expanded_thread_id.as_deref() == Some(thread_id.as_str()) {
            self.expanded_thread_id = None;
        }
        self.selected = self.selected.min(self.threads.len().saturating_sub(1));
        self.status_message = None;
    }

    pub(super) fn handle_unarchive_result(
        &mut self,
        thread_id: String,
        result: anyhow::Result<app_server_protocol::protocol::v2::ThreadUnarchiveResponse>,
    ) -> Option<super::SessionSelection> {
        if self.archive_state
            != (ArchiveState::Restoring {
                thread_id: thread_id.clone(),
            })
        {
            return None;
        }
        self.archive_state = ArchiveState::Idle;

        match result {
            Ok(response) => {
                let thread = response.thread;
                let target = super::SessionTarget {
                    path: thread.path,
                    thread_id: thread.id,
                    history_mode: Some(thread.history_mode),
                };
                Some(super::SessionSelection::Resume(target))
            }
            Err(error) => {
                self.status_message = Some(format!("Failed to restore archived session: {error}"));
                None
            }
        }
    }
}

#[cfg(test)]
#[path = "archive_tests.rs"]
mod tests;
