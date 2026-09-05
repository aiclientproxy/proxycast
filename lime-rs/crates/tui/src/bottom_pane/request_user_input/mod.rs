use std::collections::BTreeMap;

use app_server_protocol::RequestId;
use app_server_protocol::protocol::v2::{
    ToolRequestUserInputAnswer, ToolRequestUserInputParams, ToolRequestUserInputResponse,
};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::{AppServerResponse, ChatComposer, InputResult};

pub(super) mod render;

#[derive(Debug)]
pub(super) struct RequestUserInputOverlay {
    pub(super) id: RequestId,
    pub(super) params: ToolRequestUserInputParams,
    pub(super) question_index: usize,
    pub(super) selected: usize,
    pub(super) editing: bool,
    pub(super) composer: ChatComposer,
    answers: BTreeMap<String, ToolRequestUserInputAnswer>,
}

impl RequestUserInputOverlay {
    pub(super) fn new(id: RequestId, params: ToolRequestUserInputParams) -> Self {
        let editing = params
            .questions
            .first()
            .and_then(|question| question.options.as_ref())
            .is_none_or(Vec::is_empty);
        Self {
            id,
            params,
            question_index: 0,
            selected: 0,
            editing,
            composer: ChatComposer::default(),
            answers: BTreeMap::new(),
        }
    }

    pub(super) fn handle_key_event(&mut self, key: KeyEvent) -> Option<AppServerResponse> {
        if self.params.questions.is_empty() {
            return Some(self.finish());
        }
        match key {
            key if key.kind == KeyEventKind::Press
                && key.modifiers.contains(KeyModifiers::CONTROL)
                && key.code == KeyCode::Char('c') =>
            {
                Some(self.cancel())
            }
            key if key.kind == KeyEventKind::Press => match key.code {
                KeyCode::Esc => Some(self.cancel()),
                KeyCode::Tab if self.has_options() => {
                    self.editing = !self.editing;
                    None
                }
                KeyCode::Up if !self.editing => {
                    self.selected = self.selected.saturating_sub(1);
                    None
                }
                KeyCode::Down if !self.editing => {
                    let count = self.option_count();
                    if count > 0 {
                        self.selected = (self.selected + 1) % count;
                    }
                    None
                }
                _ if self.editing && key.code == KeyCode::Enter && self.composer.is_empty() => {
                    let answer = self.selected_option_label().into_iter().collect();
                    self.commit(answer)
                }
                _ if self.editing => match self.composer.handle_key_event(key) {
                    InputResult::Submitted(text) => {
                        let mut answers =
                            self.selected_option_label().into_iter().collect::<Vec<_>>();
                        let note = text.trim();
                        if !note.is_empty() {
                            if self.has_options() {
                                answers.push(format!("user_note: {note}"));
                            } else {
                                answers.push(note.to_string());
                            }
                        }
                        self.commit(answers)
                    }
                    InputResult::Interrupt | InputResult::Quit => Some(self.cancel()),
                    InputResult::Queued(text) => {
                        self.composer.insert(&text);
                        None
                    }
                    InputResult::None
                    | InputResult::Changed
                    | InputResult::DecreaseEffort
                    | InputResult::IncreaseEffort
                    | InputResult::PreviousPermissions
                    | InputResult::NextPermissions
                    | InputResult::OpenExternalEditor => None,
                },
                KeyCode::Enter => {
                    let answer = self.selected_option_label().into_iter().collect();
                    self.commit(answer)
                }
                KeyCode::Char(ch) if ch.is_ascii_digit() && ch != '0' => {
                    let index = ch.to_digit(10).unwrap_or_default() as usize - 1;
                    if index < self.option_count() {
                        self.selected = index;
                        let answer = self.selected_option_label().into_iter().collect();
                        self.commit(answer)
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn commit(&mut self, answers: Vec<String>) -> Option<AppServerResponse> {
        if let Some(question) = self.params.questions.get(self.question_index) {
            self.answers
                .insert(question.id.clone(), ToolRequestUserInputAnswer { answers });
        }
        if self.question_index + 1 >= self.params.questions.len() {
            return Some(self.finish());
        }
        self.question_index += 1;
        self.selected = 0;
        self.composer = ChatComposer::default();
        self.editing = !self.has_options();
        None
    }

    fn current_options(
        &self,
    ) -> Option<&[app_server_protocol::protocol::v2::ToolRequestUserInputOption]> {
        self.params
            .questions
            .get(self.question_index)
            .and_then(|question| question.options.as_deref())
    }

    fn has_options(&self) -> bool {
        self.current_options()
            .is_some_and(|options| !options.is_empty())
    }

    fn option_count(&self) -> usize {
        let options = self.current_options().map_or(0, <[_]>::len);
        if self.other_option_enabled() {
            options + 1
        } else {
            options
        }
    }

    fn other_option_enabled(&self) -> bool {
        self.params
            .questions
            .get(self.question_index)
            .is_some_and(|question| question.is_other && self.has_options())
    }

    fn selected_option_label(&self) -> Option<String> {
        let options = self.current_options()?;
        if let Some(option) = options.get(self.selected) {
            return Some(option.label.clone());
        }
        (self.selected == options.len() && self.other_option_enabled()).then(|| "Other".to_string())
    }

    fn finish(&self) -> AppServerResponse {
        AppServerResponse::UserInput {
            id: self.id.clone(),
            response: ToolRequestUserInputResponse {
                answers: self.answers.clone(),
            },
        }
    }

    fn cancel(&self) -> AppServerResponse {
        AppServerResponse::UserInput {
            id: self.id.clone(),
            response: ToolRequestUserInputResponse {
                answers: BTreeMap::new(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::protocol::v2::{
        ToolRequestUserInputOption, ToolRequestUserInputQuestion,
    };
    use crossterm::event::{KeyEvent, KeyModifiers};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn collects_option_and_freeform_questions_before_responding() {
        let mut request = RequestUserInputOverlay::new(
            RequestId::Integer(9),
            ToolRequestUserInputParams {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "question-1".to_string(),
                questions: vec![
                    ToolRequestUserInputQuestion {
                        id: "mode".to_string(),
                        header: "Mode".to_string(),
                        question: "Choose a mode".to_string(),
                        is_other: false,
                        is_secret: false,
                        options: Some(vec![ToolRequestUserInputOption {
                            label: "Fast".to_string(),
                            description: "Continue immediately".to_string(),
                        }]),
                    },
                    ToolRequestUserInputQuestion {
                        id: "note".to_string(),
                        header: "Note".to_string(),
                        question: "Add a note".to_string(),
                        is_other: false,
                        is_secret: false,
                        options: None,
                    },
                ],
                auto_resolution_ms: None,
            },
        );

        assert_eq!(request.handle_key_event(key(KeyCode::Enter)), None);
        request.handle_key_event(key(KeyCode::Char('好')));
        let response = request.handle_key_event(key(KeyCode::Enter));

        let Some(AppServerResponse::UserInput { response, .. }) = response else {
            panic!("expected user input response");
        };
        assert_eq!(response.answers["mode"].answers, ["Fast"]);
        assert_eq!(response.answers["note"].answers, ["好"]);
    }

    #[test]
    fn escape_returns_an_empty_fail_closed_response() {
        let mut request = RequestUserInputOverlay::new(
            RequestId::Integer(9),
            ToolRequestUserInputParams {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "question-1".to_string(),
                questions: Vec::new(),
                auto_resolution_ms: None,
            },
        );

        let response = request.handle_key_event(key(KeyCode::Esc));
        let Some(AppServerResponse::UserInput { response, .. }) = response else {
            panic!("expected user input response");
        };
        assert!(response.answers.is_empty());
    }

    #[test]
    fn ctrl_c_returns_an_empty_fail_closed_response() {
        let mut request = RequestUserInputOverlay::new(
            RequestId::Integer(10),
            ToolRequestUserInputParams {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "question-1".to_string(),
                questions: vec![ToolRequestUserInputQuestion {
                    id: "secret".to_string(),
                    header: "Secret".to_string(),
                    question: "Enter a secret".to_string(),
                    is_other: false,
                    is_secret: true,
                    options: None,
                }],
                auto_resolution_ms: None,
            },
        );
        request.editing = true;
        request.composer.insert("sensitive");

        let response =
            request.handle_key_event(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        let Some(AppServerResponse::UserInput { response, .. }) = response else {
            panic!("expected user input response");
        };
        assert!(response.answers.is_empty());
    }

    #[test]
    fn option_notes_follow_codex_answer_shape() {
        let mut request = RequestUserInputOverlay::new(
            RequestId::Integer(11),
            ToolRequestUserInputParams {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "question-1".to_string(),
                questions: vec![ToolRequestUserInputQuestion {
                    id: "mode".to_string(),
                    header: "Mode".to_string(),
                    question: "Choose a mode".to_string(),
                    is_other: false,
                    is_secret: false,
                    options: Some(vec![ToolRequestUserInputOption {
                        label: "Fast".to_string(),
                        description: "Continue immediately".to_string(),
                    }]),
                }],
                auto_resolution_ms: None,
            },
        );

        request.handle_key_event(key(KeyCode::Tab));
        request.editing = true;
        request.composer.insert("keep logs");
        let response = request.handle_key_event(key(KeyCode::Enter));

        let Some(AppServerResponse::UserInput { response, .. }) = response else {
            panic!("expected user input response");
        };
        assert_eq!(
            response.answers["mode"].answers,
            ["Fast", "user_note: keep logs"]
        );
    }

    #[test]
    fn empty_notes_submit_the_selected_option() {
        let mut request = RequestUserInputOverlay::new(
            RequestId::Integer(13),
            ToolRequestUserInputParams {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "question-1".to_string(),
                questions: vec![ToolRequestUserInputQuestion {
                    id: "mode".to_string(),
                    header: "Mode".to_string(),
                    question: "Choose a mode".to_string(),
                    is_other: false,
                    is_secret: false,
                    options: Some(vec![ToolRequestUserInputOption {
                        label: "Fast".to_string(),
                        description: "Continue immediately".to_string(),
                    }]),
                }],
                auto_resolution_ms: None,
            },
        );

        request.handle_key_event(key(KeyCode::Tab));
        let response = request.handle_key_event(key(KeyCode::Enter));

        let Some(AppServerResponse::UserInput { response, .. }) = response else {
            panic!("expected user input response");
        };
        assert_eq!(response.answers["mode"].answers, ["Fast"]);
    }

    #[test]
    fn other_option_is_only_added_when_the_contract_enables_it() {
        let mut request = RequestUserInputOverlay::new(
            RequestId::Integer(12),
            ToolRequestUserInputParams {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "question-1".to_string(),
                questions: vec![ToolRequestUserInputQuestion {
                    id: "mode".to_string(),
                    header: "Mode".to_string(),
                    question: "Choose a mode".to_string(),
                    is_other: true,
                    is_secret: false,
                    options: Some(vec![ToolRequestUserInputOption {
                        label: "Fast".to_string(),
                        description: "Continue immediately".to_string(),
                    }]),
                }],
                auto_resolution_ms: None,
            },
        );

        request.handle_key_event(key(KeyCode::Down));
        let response = request.handle_key_event(key(KeyCode::Enter));

        let Some(AppServerResponse::UserInput { response, .. }) = response else {
            panic!("expected user input response");
        };
        assert_eq!(response.answers["mode"].answers, ["Other"]);
    }
}
