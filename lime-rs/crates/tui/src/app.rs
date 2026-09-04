use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};

use crate::bottom_pane::{AppServerResponse, BottomPane};
use crate::composer::{Composer, ComposerAction};
use crate::model_picker::{ModelPicker, ModelPickerAction, ModelSelection};
use crate::projection::ConversationProjection;

#[derive(Debug, PartialEq)]
pub(crate) enum AppAction {
    None,
    Submit(String),
    Queue(String),
    Interrupt,
    DecreaseEffort,
    IncreaseEffort,
    PreviousPermissions,
    NextPermissions,
    OpenExternalEditor,
    ScrollUp,
    ScrollDown,
    ScrollTop,
    ScrollBottom,
    SelectModel(ModelSelection),
    Respond(AppServerResponse),
    Quit,
}

#[derive(Debug, Default)]
pub(crate) struct App {
    pub(crate) bottom_pane: BottomPane,
    pub(crate) composer: Composer,
    pub(crate) projection: ConversationProjection,
    pub(crate) model_picker: Option<ModelPicker>,
    pub(crate) model: Option<String>,
    pub(crate) model_provider: Option<String>,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) permissions: Option<String>,
    pub(crate) transcript_scroll: usize,
}

impl App {
    pub(crate) fn set_settings(
        &mut self,
        model: Option<String>,
        model_provider: Option<String>,
        reasoning_effort: Option<String>,
        permissions: Option<String>,
    ) {
        self.model = model;
        self.model_provider = model_provider;
        self.reasoning_effort = reasoning_effort;
        self.permissions = permissions;
    }

    pub(crate) fn open_model_picker(
        &mut self,
        models: Vec<app_server_protocol::protocol::v2::Model>,
    ) {
        self.model_picker = Some(ModelPicker::new(models));
    }

    pub(crate) fn handle_terminal_event(&mut self, event: Event) -> AppAction {
        if let Event::Key(ref key) = event {
            if key.kind == KeyEventKind::Press {
                match key.code {
                    KeyCode::PageUp => return AppAction::ScrollUp,
                    KeyCode::PageDown => return AppAction::ScrollDown,
                    KeyCode::Home if key.modifiers.contains(KeyModifiers::ALT) => {
                        return AppAction::ScrollTop;
                    }
                    KeyCode::End if key.modifiers.contains(KeyModifiers::ALT) => {
                        return AppAction::ScrollBottom;
                    }
                    _ => {}
                }
            }
        }

        if self.bottom_pane.is_active() {
            return self
                .bottom_pane
                .handle_event(event)
                .map(AppAction::Respond)
                .unwrap_or(AppAction::None);
        }

        if let Some(picker) = self.model_picker.as_mut() {
            return match picker.handle_event(event) {
                ModelPickerAction::Select(index) => {
                    let selection = picker.selected_model(index);
                    self.model_picker = None;
                    selection
                        .map(AppAction::SelectModel)
                        .unwrap_or(AppAction::None)
                }
                ModelPickerAction::Cancel => {
                    self.model_picker = None;
                    AppAction::None
                }
                ModelPickerAction::None => AppAction::None,
            };
        }

        match event {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                match self.composer.handle_key(key) {
                    ComposerAction::Submit(text) => AppAction::Submit(text),
                    ComposerAction::Queue(text) => AppAction::Queue(text),
                    ComposerAction::Interrupt => AppAction::Interrupt,
                    ComposerAction::DecreaseEffort => AppAction::DecreaseEffort,
                    ComposerAction::IncreaseEffort => AppAction::IncreaseEffort,
                    ComposerAction::PreviousPermissions => AppAction::PreviousPermissions,
                    ComposerAction::NextPermissions => AppAction::NextPermissions,
                    ComposerAction::OpenExternalEditor => AppAction::OpenExternalEditor,
                    ComposerAction::Quit => AppAction::Quit,
                    ComposerAction::None | ComposerAction::Changed => AppAction::None,
                }
            }
            Event::Paste(text) => {
                self.composer.insert(&text);
                AppAction::None
            }
            _ => AppAction::None,
        }
    }

    pub(crate) fn scroll_up(&mut self, amount: usize) {
        self.transcript_scroll = self.transcript_scroll.saturating_add(amount);
    }

    pub(crate) fn scroll_down(&mut self, amount: usize) {
        self.transcript_scroll = self.transcript_scroll.saturating_sub(amount);
    }

    pub(crate) fn scroll_top(&mut self) {
        self.transcript_scroll = usize::MAX;
    }

    pub(crate) fn scroll_bottom(&mut self) {
        self.transcript_scroll = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::protocol::v2::{
        CommandExecutionApprovalDecision, CommandExecutionRequestApprovalParams, ServerRequest,
    };
    use app_server_protocol::RequestId;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    #[test]
    fn active_bottom_pane_receives_input_before_the_chat_composer() {
        let mut app = App::default();
        app.composer.insert("draft");
        app.bottom_pane
            .enqueue(ServerRequest::ItemCommandExecutionRequestApproval {
                id: RequestId::Integer(7),
                params: CommandExecutionRequestApprovalParams {
                    thread_id: "thread-1".to_string(),
                    turn_id: "turn-1".to_string(),
                    item_id: "command-1".to_string(),
                    started_at_ms: 1,
                    approval_id: None,
                    reason: None,
                    network_approval_context: None,
                    command: Some("cargo test".to_string()),
                    cwd: Some("/workspace".to_string()),
                    available_decisions: None,
                },
            })
            .expect("queue approval");

        let ignored = app.handle_terminal_event(Event::Key(KeyEvent::new(
            KeyCode::Char('x'),
            KeyModifiers::NONE,
        )));
        assert_eq!(ignored, AppAction::None);
        assert_eq!(app.composer.text(), "draft");

        let response = app.handle_terminal_event(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));
        assert!(matches!(
            response,
            AppAction::Respond(AppServerResponse::Command {
                response:
                    app_server_protocol::protocol::v2::CommandExecutionRequestApprovalResponse {
                        decision: CommandExecutionApprovalDecision::Accept,
                    },
                ..
            })
        ));
        assert!(!app.bottom_pane.is_active());
        assert_eq!(app.composer.text(), "draft");
    }

    #[test]
    fn tab_queues_a_follow_up_without_submitting_the_active_turn() {
        let mut app = App::default();
        app.composer.insert("follow up");
        app.projection.apply(
            app_server_protocol::protocol::v2::ServerNotification::TurnStarted(
                app_server_protocol::protocol::v2::TurnStartedNotification {
                    thread_id: "thread-1".to_string(),
                    turn: app_server_protocol::protocol::v2::Turn {
                        id: "turn-1".to_string(),
                        items: Vec::new(),
                        items_view: Default::default(),
                        status: app_server_protocol::protocol::v2::TurnStatus::InProgress,
                        error: None,
                        started_at: None,
                        completed_at: None,
                        duration_ms: None,
                    },
                },
            ),
        );

        let action =
            app.handle_terminal_event(Event::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)));

        assert_eq!(action, AppAction::Queue("follow up".to_string()));
        assert!(app.composer.is_empty());
    }

    #[test]
    fn codex_style_effort_and_permission_shortcuts_are_not_inserted_into_draft() {
        let mut app = App::default();
        app.composer.insert("draft");
        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(
                KeyCode::Char('.'),
                KeyModifiers::ALT,
            ))),
            AppAction::IncreaseEffort
        );
        assert_eq!(
            app.handle_terminal_event(Event::Key(
                KeyEvent::new(KeyCode::F(8), KeyModifiers::NONE,)
            )),
            AppAction::NextPermissions
        );
        assert_eq!(app.composer.text(), "draft");
    }
}
