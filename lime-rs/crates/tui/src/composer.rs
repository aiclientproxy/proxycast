use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use unicode_segmentation::UnicodeSegmentation;

const MAX_HISTORY_ENTRIES: usize = 200;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ComposerAction {
    None,
    Changed,
    Submit(String),
    Queue(String),
    Interrupt,
    DecreaseEffort,
    IncreaseEffort,
    PreviousPermissions,
    NextPermissions,
    OpenExternalEditor,
    Quit,
}

#[derive(Debug, Default)]
pub(crate) struct Composer {
    text: String,
    cursor: usize,
    history: Vec<String>,
    history_index: Option<usize>,
    saved_draft: Option<String>,
}

impl Composer {
    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    pub(crate) fn cursor(&self) -> usize {
        self.cursor
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub(crate) fn replace(&mut self, text: String) {
        self.replace_text(text);
        self.reset_history_navigation();
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
    }

    pub(crate) fn insert(&mut self, value: &str) {
        self.text.insert_str(self.cursor, value);
        self.cursor += value.len();
        self.reset_history_navigation();
    }

    pub(crate) fn handle_key(&mut self, key: KeyEvent) -> ComposerAction {
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            return match key.code {
                KeyCode::Char('c') => ComposerAction::Interrupt,
                KeyCode::Char('d') if self.is_empty() => ComposerAction::Quit,
                KeyCode::Char('j') => {
                    self.insert("\n");
                    ComposerAction::Changed
                }
                KeyCode::Char('a') => {
                    self.cursor = self.line_start();
                    ComposerAction::Changed
                }
                KeyCode::Char('e') => {
                    self.cursor = self.line_end();
                    ComposerAction::Changed
                }
                KeyCode::Char('g') => ComposerAction::OpenExternalEditor,
                _ => ComposerAction::None,
            };
        }

        match key.code {
            KeyCode::Char(',') if key.modifiers.contains(KeyModifiers::ALT) => {
                ComposerAction::DecreaseEffort
            }
            KeyCode::Char('.') if key.modifiers.contains(KeyModifiers::ALT) => {
                ComposerAction::IncreaseEffort
            }
            KeyCode::F(7) => ComposerAction::PreviousPermissions,
            KeyCode::F(8) => ComposerAction::NextPermissions,
            KeyCode::Tab => self.queue(),
            KeyCode::Enter
                if key
                    .modifiers
                    .intersects(KeyModifiers::ALT | KeyModifiers::SHIFT) =>
            {
                self.insert("\n");
                ComposerAction::Changed
            }
            KeyCode::Enter => self.submit(),
            KeyCode::Char(ch) => {
                self.insert(&ch.to_string());
                ComposerAction::Changed
            }
            KeyCode::Backspace => {
                if self.remove_previous_grapheme() {
                    ComposerAction::Changed
                } else {
                    ComposerAction::None
                }
            }
            KeyCode::Delete => {
                if self.remove_next_grapheme() {
                    ComposerAction::Changed
                } else {
                    ComposerAction::None
                }
            }
            KeyCode::Left => {
                self.cursor = self.previous_grapheme_start();
                ComposerAction::Changed
            }
            KeyCode::Right => {
                self.cursor = self.next_grapheme_end();
                ComposerAction::Changed
            }
            KeyCode::Home => {
                self.cursor = self.line_start();
                ComposerAction::Changed
            }
            KeyCode::End => {
                self.cursor = self.line_end();
                ComposerAction::Changed
            }
            KeyCode::Up if !self.text.contains('\n') => self.history_previous(),
            KeyCode::Down if !self.text.contains('\n') => self.history_next(),
            _ => ComposerAction::None,
        }
    }

    fn submit(&mut self) -> ComposerAction {
        self.take_submission(ComposerAction::Submit)
    }

    fn queue(&mut self) -> ComposerAction {
        self.take_submission(ComposerAction::Queue)
    }

    fn take_submission<F>(&mut self, action: F) -> ComposerAction
    where
        F: FnOnce(String) -> ComposerAction,
    {
        if self.text.trim().is_empty() {
            return ComposerAction::None;
        }
        let submitted = std::mem::take(&mut self.text);
        self.cursor = 0;
        self.history.push(submitted.clone());
        if self.history.len() > MAX_HISTORY_ENTRIES {
            self.history.remove(0);
        }
        self.history_index = None;
        self.saved_draft = None;
        action(submitted)
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

    fn history_previous(&mut self) -> ComposerAction {
        if self.history.is_empty() {
            return ComposerAction::None;
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
        ComposerAction::Changed
    }

    fn history_next(&mut self) -> ComposerAction {
        let Some(index) = self.history_index else {
            return ComposerAction::None;
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
        ComposerAction::Changed
    }

    fn replace_text(&mut self, text: String) {
        self.text = text;
        self.cursor = self.text.len();
    }

    fn reset_history_navigation(&mut self) {
        self.history_index = None;
        self.saved_draft = None;
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
        let mut composer = Composer::default();
        composer.insert("a👩🏽‍💻界");

        composer.handle_key(key(KeyCode::Left));
        composer.handle_key(key(KeyCode::Backspace));

        assert_eq!(composer.text(), "a界");
        assert_eq!(composer.cursor(), 1);
    }

    #[test]
    fn paste_inserts_at_a_unicode_boundary() {
        let mut composer = Composer::default();
        composer.insert("你好");
        composer.handle_key(key(KeyCode::Left));
        composer.insert("\nworld");

        assert_eq!(composer.text(), "你\nworld好");
    }

    #[test]
    fn history_restores_the_unsent_draft() {
        let mut composer = Composer::default();
        composer.insert("first");
        assert!(matches!(
            composer.handle_key(key(KeyCode::Enter)),
            ComposerAction::Submit(_)
        ));
        composer.insert("draft");

        composer.handle_key(key(KeyCode::Up));
        assert_eq!(composer.text(), "first");
        composer.handle_key(key(KeyCode::Down));
        assert_eq!(composer.text(), "draft");
    }

    #[test]
    fn modified_enter_adds_a_newline_and_plain_enter_submits() {
        let mut composer = Composer::default();
        composer.insert("first");
        composer.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
        composer.insert("second");

        assert_eq!(composer.text(), "first\nsecond");
        assert_eq!(
            composer.handle_key(key(KeyCode::Enter)),
            ComposerAction::Submit("first\nsecond".to_string())
        );
    }

    #[test]
    fn tab_queues_non_empty_draft_and_clears_composer() {
        let mut composer = Composer::default();
        composer.insert("follow up");

        assert_eq!(
            composer.handle_key(key(KeyCode::Tab)),
            ComposerAction::Queue("follow up".to_string())
        );
        assert!(composer.is_empty());
    }

    #[test]
    fn loaded_history_is_bounded_to_the_recent_entries() {
        let mut composer = Composer::default();
        composer.load_history((0..205).map(|index| format!("prompt-{index}")));

        composer.handle_key(key(KeyCode::Up));
        assert_eq!(composer.text(), "prompt-204");
        for _ in 0..199 {
            composer.handle_key(key(KeyCode::Up));
        }
        assert_eq!(composer.text(), "prompt-5");
    }
}
