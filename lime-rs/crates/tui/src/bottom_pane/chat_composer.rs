use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use unicode_segmentation::UnicodeSegmentation;

const MAX_HISTORY_ENTRIES: usize = 200;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InputResult {
    None,
    Changed,
    Submitted(String),
    Queued(String),
    Interrupt,
    DecreaseEffort,
    IncreaseEffort,
    PreviousPermissions,
    NextPermissions,
    OpenExternalEditor,
    Quit,
}

#[derive(Debug, Default)]
struct HistorySearchState {
    query: String,
    draft: String,
    selected_index: Option<usize>,
}

#[derive(Debug, Default)]
pub(crate) struct ChatComposer {
    text: String,
    cursor: usize,
    history: Vec<String>,
    history_index: Option<usize>,
    saved_draft: Option<String>,
    history_search: Option<HistorySearchState>,
}

impl ChatComposer {
    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    pub(crate) fn cursor(&self) -> usize {
        self.cursor
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub(crate) fn history_search_active(&self) -> bool {
        self.history_search.is_some()
    }

    pub(crate) fn history_search_query(&self) -> Option<&str> {
        self.history_search
            .as_ref()
            .map(|search| search.query.as_str())
    }

    pub(crate) fn replace(&mut self, text: String) {
        self.replace_text(text);
        self.reset_history_navigation();
        self.history_search = None;
    }

    pub(crate) fn load_history<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = String>,
    {
        self.history = entries
            .into_iter()
            .filter(|entry| !entry.trim().is_empty())
            .collect();
        if self.history.len() > MAX_HISTORY_ENTRIES {
            let keep_from = self.history.len() - MAX_HISTORY_ENTRIES;
            self.history.drain(..keep_from);
        }
        self.history_index = None;
        self.saved_draft = None;
        self.history_search = None;
    }

    pub(crate) fn insert(&mut self, value: &str) {
        self.text.insert_str(self.cursor, value);
        self.cursor += value.len();
        self.reset_history_navigation();
    }

    pub(crate) fn handle_key_event(&mut self, key: KeyEvent) -> InputResult {
        if self.history_search.is_some() {
            return self.handle_history_search_key(key);
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            return match key.code {
                KeyCode::Char('c') => InputResult::Interrupt,
                KeyCode::Char('d') if self.is_empty() => InputResult::Quit,
                KeyCode::Char('j') => {
                    self.insert("\n");
                    InputResult::Changed
                }
                KeyCode::Char('a') => {
                    self.cursor = self.line_start();
                    InputResult::Changed
                }
                KeyCode::Char('e') => {
                    self.cursor = self.line_end();
                    InputResult::Changed
                }
                KeyCode::Char('g') => InputResult::OpenExternalEditor,
                KeyCode::Char('r') => self.start_history_search(),
                _ => InputResult::None,
            };
        }

        match key.code {
            KeyCode::Char(',') if key.modifiers.contains(KeyModifiers::ALT) => {
                InputResult::DecreaseEffort
            }
            KeyCode::Char('.') if key.modifiers.contains(KeyModifiers::ALT) => {
                InputResult::IncreaseEffort
            }
            KeyCode::F(7) => InputResult::PreviousPermissions,
            KeyCode::F(8) => InputResult::NextPermissions,
            KeyCode::Tab => self.queue(),
            KeyCode::Enter
                if key
                    .modifiers
                    .intersects(KeyModifiers::ALT | KeyModifiers::SHIFT) =>
            {
                self.insert("\n");
                InputResult::Changed
            }
            KeyCode::Enter => self.submit(),
            KeyCode::Char(ch) => {
                self.insert(&ch.to_string());
                InputResult::Changed
            }
            KeyCode::Backspace => {
                if self.remove_previous_grapheme() {
                    InputResult::Changed
                } else {
                    InputResult::None
                }
            }
            KeyCode::Delete => {
                if self.remove_next_grapheme() {
                    InputResult::Changed
                } else {
                    InputResult::None
                }
            }
            KeyCode::Left => {
                self.cursor = self.previous_grapheme_start();
                InputResult::Changed
            }
            KeyCode::Right => {
                self.cursor = self.next_grapheme_end();
                InputResult::Changed
            }
            KeyCode::Home => {
                self.cursor = self.line_start();
                InputResult::Changed
            }
            KeyCode::End => {
                self.cursor = self.line_end();
                InputResult::Changed
            }
            KeyCode::Up if !self.text.contains('\n') => self.history_previous(),
            KeyCode::Down if !self.text.contains('\n') => self.history_next(),
            _ => InputResult::None,
        }
    }

    fn submit(&mut self) -> InputResult {
        self.take_submission(InputResult::Submitted)
    }

    fn queue(&mut self) -> InputResult {
        self.take_submission(InputResult::Queued)
    }

    fn take_submission<F>(&mut self, action: F) -> InputResult
    where
        F: FnOnce(String) -> InputResult,
    {
        if self.text.trim().is_empty() {
            return InputResult::None;
        }
        let submitted = std::mem::take(&mut self.text);
        self.cursor = 0;
        self.history.push(submitted.clone());
        if self.history.len() > MAX_HISTORY_ENTRIES {
            self.history.remove(0);
        }
        self.history_index = None;
        self.saved_draft = None;
        self.history_search = None;
        action(submitted)
    }

    fn start_history_search(&mut self) -> InputResult {
        self.history_index = None;
        self.saved_draft = None;
        self.history_search = Some(HistorySearchState {
            query: String::new(),
            draft: self.text.clone(),
            selected_index: None,
        });
        InputResult::Changed
    }

    fn handle_history_search_key(&mut self, key: KeyEvent) -> InputResult {
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            return match key.code {
                KeyCode::Char('r') => {
                    self.select_history_match(true);
                    InputResult::Changed
                }
                KeyCode::Char('s') => {
                    self.select_history_match_newer();
                    InputResult::Changed
                }
                KeyCode::Char('c') => {
                    let draft = self
                        .history_search
                        .take()
                        .map(|search| search.draft)
                        .unwrap_or_default();
                    self.replace_text(draft);
                    InputResult::Changed
                }
                _ => InputResult::None,
            };
        }

        match key.code {
            KeyCode::Esc => {
                let draft = self
                    .history_search
                    .take()
                    .map(|search| search.draft)
                    .unwrap_or_default();
                self.replace_text(draft);
                InputResult::Changed
            }
            KeyCode::Enter => {
                if self
                    .history_search
                    .as_ref()
                    .is_some_and(|search| search.selected_index.is_some())
                {
                    self.history_search = None;
                    self.submit()
                } else {
                    let draft = self
                        .history_search
                        .take()
                        .map(|search| search.draft)
                        .unwrap_or_default();
                    self.replace_text(draft);
                    InputResult::Changed
                }
            }
            KeyCode::Backspace => {
                if let Some(search) = self.history_search.as_mut() {
                    if let Some((start, _)) = search.query.grapheme_indices(true).next_back() {
                        search.query.truncate(start);
                        search.selected_index = None;
                    }
                }
                self.select_history_match(false);
                InputResult::Changed
            }
            KeyCode::Char(ch) => {
                if let Some(search) = self.history_search.as_mut() {
                    search.query.push(ch);
                    search.selected_index = None;
                }
                self.select_history_match(false);
                InputResult::Changed
            }
            KeyCode::Up => {
                self.select_history_match(true);
                InputResult::Changed
            }
            KeyCode::Down => {
                self.select_history_match_newer();
                InputResult::Changed
            }
            KeyCode::Left | KeyCode::Right | KeyCode::Home | KeyCode::End => {
                self.history_search = None;
                self.handle_key_event(key)
            }
            _ => InputResult::None,
        }
    }

    fn select_history_match(&mut self, older: bool) {
        let Some(search) = self.history_search.as_ref() else {
            return;
        };
        let selected_index = search.selected_index;
        let query = search.query.to_lowercase();
        if query.is_empty() {
            return;
        }
        let start = selected_index
            .map(|index| {
                if older {
                    index.saturating_sub(1)
                } else {
                    index
                }
            })
            .unwrap_or_else(|| self.history.len().saturating_sub(1));
        let found = (0..=start).rev().find(|index| {
            self.history
                .get(*index)
                .is_some_and(|entry| entry.to_lowercase().contains(&query))
        });
        if let Some(index) = found {
            if let Some(search) = self.history_search.as_mut() {
                search.selected_index = Some(index);
            }
            self.replace_text(self.history[index].clone());
        } else if selected_index.is_none() {
            let draft = self
                .history_search
                .as_ref()
                .map(|search| search.draft.clone())
                .unwrap_or_default();
            self.replace_text(draft);
        }
    }

    fn select_history_match_newer(&mut self) {
        let Some(search) = self.history_search.as_ref() else {
            return;
        };
        let Some(current) = search.selected_index else {
            return;
        };
        let query = search.query.to_lowercase();
        if query.is_empty() {
            return;
        }
        let found = ((current + 1)..self.history.len()).find(|index| {
            self.history
                .get(*index)
                .is_some_and(|entry| entry.to_lowercase().contains(&query))
        });
        if let Some(index) = found {
            if let Some(search) = self.history_search.as_mut() {
                search.selected_index = Some(index);
            }
            self.replace_text(self.history[index].clone());
        }
    }

    fn previous_grapheme_start(&self) -> usize {
        self.text[..self.cursor]
            .grapheme_indices(true)
            .next_back()
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn next_grapheme_end(&self) -> usize {
        self.text[self.cursor..]
            .graphemes(true)
            .next()
            .map(|grapheme| self.cursor + grapheme.len())
            .unwrap_or(self.text.len())
    }

    fn remove_previous_grapheme(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        let start = self.previous_grapheme_start();
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
        self.reset_history_navigation();
        true
    }

    fn remove_next_grapheme(&mut self) -> bool {
        if self.cursor == self.text.len() {
            return false;
        }
        let end = self.next_grapheme_end();
        self.text.replace_range(self.cursor..end, "");
        self.reset_history_navigation();
        true
    }

    fn line_start(&self) -> usize {
        self.text[..self.cursor]
            .rfind('\n')
            .map(|index| index + 1)
            .unwrap_or(0)
    }

    fn line_end(&self) -> usize {
        self.text[self.cursor..]
            .find('\n')
            .map(|index| self.cursor + index)
            .unwrap_or(self.text.len())
    }

    fn history_previous(&mut self) -> InputResult {
        if self.history.is_empty() {
            return InputResult::None;
        }
        if self.history_index.is_none() {
            self.saved_draft = Some(self.text.clone());
        }
        let index = self
            .history_index
            .map(|index| index.saturating_sub(1))
            .unwrap_or(self.history.len() - 1);
        self.history_index = Some(index);
        self.replace_text(self.history[index].clone());
        InputResult::Changed
    }

    fn history_next(&mut self) -> InputResult {
        let Some(index) = self.history_index else {
            return InputResult::None;
        };
        if index + 1 < self.history.len() {
            let next = index + 1;
            self.history_index = Some(next);
            self.replace_text(self.history[next].clone());
        } else {
            self.history_index = None;
            let draft = self.saved_draft.take().unwrap_or_default();
            self.replace_text(draft);
        }
        InputResult::Changed
    }

    fn replace_text(&mut self, text: String) {
        self.text = text;
        self.cursor = self.text.len();
    }

    fn reset_history_navigation(&mut self) {
        self.history_index = None;
        self.saved_draft = None;
        self.history_search = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn cursor_and_backspace_respect_extended_graphemes() {
        let mut composer = ChatComposer::default();
        composer.insert("a👩🏽‍💻界");

        composer.handle_key_event(key(KeyCode::Left));
        composer.handle_key_event(key(KeyCode::Backspace));

        assert_eq!(composer.text(), "a界");
        assert_eq!(composer.cursor(), 1);
    }

    #[test]
    fn paste_inserts_at_a_unicode_boundary() {
        let mut composer = ChatComposer::default();
        composer.insert("你好");
        composer.handle_key_event(key(KeyCode::Left));
        composer.insert("\nworld");

        assert_eq!(composer.text(), "你\nworld好");
    }

    #[test]
    fn history_restores_the_unsent_draft() {
        let mut composer = ChatComposer::default();
        composer.insert("first");
        assert!(matches!(
            composer.handle_key_event(key(KeyCode::Enter)),
            InputResult::Submitted(_)
        ));
        composer.insert("draft");

        composer.handle_key_event(key(KeyCode::Up));
        assert_eq!(composer.text(), "first");
        composer.handle_key_event(key(KeyCode::Down));
        assert_eq!(composer.text(), "draft");
    }

    #[test]
    fn modified_enter_adds_a_newline_and_plain_enter_submits() {
        let mut composer = ChatComposer::default();
        composer.insert("first");
        composer.handle_key_event(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
        composer.insert("second");

        assert_eq!(composer.text(), "first\nsecond");
        assert_eq!(
            composer.handle_key_event(key(KeyCode::Enter)),
            InputResult::Submitted("first\nsecond".to_string())
        );
    }

    #[test]
    fn tab_queues_non_empty_draft_and_clears_composer() {
        let mut composer = ChatComposer::default();
        composer.insert("follow up");

        assert_eq!(
            composer.handle_key_event(key(KeyCode::Tab)),
            InputResult::Queued("follow up".to_string())
        );
        assert!(composer.is_empty());
    }

    #[test]
    fn loaded_history_is_bounded_to_the_recent_entries() {
        let mut composer = ChatComposer::default();
        composer.load_history((0..205).map(|index| format!("prompt-{index}")));

        composer.handle_key_event(key(KeyCode::Up));
        assert_eq!(composer.text(), "prompt-204");
        for _ in 0..199 {
            composer.handle_key_event(key(KeyCode::Up));
        }
        assert_eq!(composer.text(), "prompt-5");
    }

    #[test]
    fn ctrl_r_searches_history_without_replacing_the_saved_draft() {
        let mut composer = ChatComposer::default();
        composer.load_history([
            "git status".to_string(),
            "cargo test".to_string(),
            "git diff".to_string(),
        ]);
        composer.insert("draft");

        assert_eq!(
            composer.handle_key_event(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL)),
            InputResult::Changed
        );
        assert_eq!(composer.text(), "draft");
        assert_eq!(composer.history_search_query(), Some(""));
        composer.handle_key_event(key(KeyCode::Char('g')));
        composer.handle_key_event(key(KeyCode::Char('i')));
        composer.handle_key_event(key(KeyCode::Char('t')));
        assert_eq!(composer.text(), "git diff");
        assert_eq!(composer.history_search_query(), Some("git"));
        composer.handle_key_event(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
        assert_eq!(composer.text(), "git status");
        composer.handle_key_event(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL));
        assert_eq!(composer.text(), "git diff");
        composer.handle_key_event(key(KeyCode::Esc));
        assert_eq!(composer.text(), "draft");
        assert!(!composer.history_search_active());
    }

    #[test]
    fn ctrl_r_search_accepts_a_match_with_enter() {
        let mut composer = ChatComposer::default();
        composer.load_history(["first prompt".to_string()]);
        composer.handle_key_event(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
        for character in "first".chars() {
            composer.handle_key_event(key(KeyCode::Char(character)));
        }

        assert_eq!(
            composer.handle_key_event(key(KeyCode::Enter)),
            InputResult::Submitted("first prompt".to_string())
        );
        assert!(!composer.history_search_active());
    }

    #[test]
    fn history_search_is_case_insensitive_and_restores_on_no_match() {
        let mut composer = ChatComposer::default();
        composer.load_history(["Deploy Lime".to_string()]);
        composer.insert("draft");
        composer.handle_key_event(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
        for character in "DEP".chars() {
            composer.handle_key_event(key(KeyCode::Char(character)));
        }
        assert_eq!(composer.text(), "Deploy Lime");
        composer.handle_key_event(key(KeyCode::Char('x')));
        assert_eq!(composer.text(), "draft");
        assert_eq!(composer.history_search_query(), Some("DEPx"));
    }
}
