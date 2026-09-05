use app_server_protocol::protocol::v2::{QueuedSubmission, ServerNotification, Thread};
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::bottom_pane::{AppServerResponse, BottomPane, ChatComposer, InputResult};
use crate::command_popup::{CommandPopup, CommandPopupAction};
use crate::locale::Locale;
use crate::model_picker::{ModelPicker, ModelPickerAction, ModelSelection};
use crate::pager_overlay::{PagerAction, PagerOverlay, StatusFacts};
use crate::pending_input_preview::can_restore_submission;
use crate::projection::ConversationProjection;
use crate::settings::{PERMISSION_PROFILES, cycle_setting};
use crate::slash_command::{SlashCommand, command_from_prompt};

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
    CopyLastResponse,
    PasteImage,
    EditQueuedSubmission(QueuedSubmission),
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
    pub(crate) composer: ChatComposer,
    pub(crate) projection: ConversationProjection,
    pub(crate) model_picker: Option<ModelPicker>,
    pub(crate) command_popup: Option<CommandPopup>,
    pub(crate) pager_overlay: Option<PagerOverlay>,
    pub(crate) thread_id: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) model_provider: Option<String>,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) permissions: Option<String>,
    pub(crate) permission_profiles: Vec<String>,
    pub(crate) transcript_scroll: usize,
    pub(crate) locale: Locale,
    pub(crate) cwd: PathBuf,
    pub(crate) clipboard_lease: Option<crate::clipboard_copy::ClipboardLease>,
    pub(crate) pending_images: Vec<PathBuf>,
    pub(crate) queued_submissions: Vec<QueuedSubmission>,
    active_turn_started_at: Option<Instant>,
}

impl App {
    pub(crate) fn set_cwd(&mut self, cwd: PathBuf) {
        self.cwd = cwd;
    }

    pub(crate) fn set_thread_id(&mut self, thread_id: String) {
        if self.thread_id.as_deref() != Some(thread_id.as_str()) {
            self.queued_submissions.clear();
        }
        self.thread_id = Some(thread_id);
    }

    pub(crate) fn set_locale(&mut self, locale: Locale) {
        self.locale = locale;
    }

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

    pub(crate) fn set_permission_profiles(&mut self, profiles: impl IntoIterator<Item = String>) {
        let mut next = Vec::new();
        for profile in profiles {
            let profile = profile.trim();
            if profile.is_empty() || next.iter().any(|value| value == profile) {
                continue;
            }
            next.push(profile.to_string());
        }
        self.permission_profiles = next;
    }

    pub(crate) fn cycle_permission_profile(&self, current: Option<&str>, direction: i8) -> String {
        if self.permission_profiles.is_empty() {
            return cycle_setting(&PERMISSION_PROFILES, current, direction);
        }
        cycle_setting(&self.permission_profiles, current, direction)
    }

    pub(crate) fn hydrate_thread(&mut self, thread: Thread) {
        self.projection.hydrate_thread(thread);
        self.active_turn_started_at = self
            .projection
            .active_turn_id()
            .is_some()
            .then(Instant::now);
    }

    pub(crate) fn start_turn(&mut self, turn_id: String) {
        self.projection.start_turn(turn_id);
        self.active_turn_started_at = Some(Instant::now());
    }

    pub(crate) fn apply_notification(&mut self, notification: ServerNotification) {
        let previous_turn_id = self.projection.active_turn_id().map(str::to_owned);
        self.projection.apply(notification);
        let active_turn_id = self.projection.active_turn_id();
        if active_turn_id != previous_turn_id.as_deref() {
            self.active_turn_started_at = active_turn_id.map(|_| Instant::now());
        }
    }

    pub(crate) fn active_turn_elapsed(&self, now: Instant) -> Option<Duration> {
        self.projection.active_turn_id()?;
        Some(
            self.active_turn_started_at
                .map(|started_at| now.saturating_duration_since(started_at))
                .unwrap_or_default(),
        )
    }

    pub(crate) fn open_model_picker(
        &mut self,
        models: Vec<app_server_protocol::protocol::v2::Model>,
    ) {
        self.model_picker = Some(ModelPicker::new(models));
    }

    pub(crate) fn replace_composer(&mut self, text: String) {
        self.composer.replace(text);
        self.sync_command_popup();
    }

    pub(crate) fn handle_terminal_event(&mut self, event: Event) -> AppAction {
        if let Some(pager) = self.pager_overlay.as_mut() {
            if pager.handle_event(&event) == PagerAction::Close {
                self.pager_overlay = None;
            }
            return AppAction::None;
        }

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

        if let Some(popup) = self.command_popup.as_mut() {
            match popup.handle_event(&event) {
                CommandPopupAction::Pass => {}
                CommandPopupAction::Consumed => return AppAction::None,
                CommandPopupAction::Cancel => {
                    self.command_popup = None;
                    return AppAction::None;
                }
                CommandPopupAction::Complete(command) => {
                    self.complete_slash_command(command);
                    return AppAction::None;
                }
                CommandPopupAction::Execute(command) => {
                    self.composer.replace(format!("/{}", command.command()));
                    self.command_popup = None;
                    if let Some(action) = self.run_local_command() {
                        return action;
                    }
                    let action = self
                        .composer
                        .handle_key_event(crossterm::event::KeyEvent::new(
                            KeyCode::Enter,
                            KeyModifiers::NONE,
                        ));
                    return self.map_composer_action(action);
                }
            }
        }

        if self.composer.history_search_active() {
            if let Event::Key(key) = event {
                let action = self.composer.handle_key_event(key);
                return self.map_composer_action(action);
            }
            return AppAction::None;
        }

        match event {
            Event::Key(key)
                if key.kind == KeyEventKind::Press
                    && key.code == KeyCode::Esc
                    && self.projection.active_turn_id().is_some() =>
            {
                AppAction::Interrupt
            }
            Event::Key(key)
                if key.kind == KeyEventKind::Press
                    && key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                    && matches!(key.code, KeyCode::Char(value) if value.eq_ignore_ascii_case(&'v')) =>
            {
                AppAction::PasteImage
            }
            Event::Key(key)
                if key.kind == KeyEventKind::Press
                    && key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char(value) if value.eq_ignore_ascii_case(&'o')) =>
            {
                AppAction::CopyLastResponse
            }
            Event::Key(key)
                if key.kind == KeyEventKind::Press
                    && key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char(value) if value.eq_ignore_ascii_case(&'t')) =>
            {
                self.command_popup = None;
                self.pager_overlay = Some(PagerOverlay::transcript(self.locale));
                AppAction::None
            }
            Event::Key(key)
                if key.kind == KeyEventKind::Press
                    && key.code == KeyCode::Up
                    && key.modifiers.contains(KeyModifiers::ALT)
                    && self.composer.is_empty()
                    && self.pending_images.is_empty() =>
            {
                self.queued_submissions
                    .last()
                    .filter(|submission| can_restore_submission(submission))
                    .cloned()
                    .map(AppAction::EditQueuedSubmission)
                    .unwrap_or(AppAction::None)
            }
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                if key.code == KeyCode::Enter
                    && !key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SHIFT)
                {
                    if let Some(action) = self.run_local_command() {
                        return action;
                    }
                }
                if matches!(key.code, KeyCode::Enter | KeyCode::Tab)
                    && self.composer.is_empty()
                    && !self.pending_images.is_empty()
                {
                    return if key.code == KeyCode::Tab {
                        AppAction::Queue(String::new())
                    } else {
                        AppAction::Submit(String::new())
                    };
                }
                if key.code == KeyCode::Backspace
                    && self.composer.is_empty()
                    && self.pending_images.pop().is_some()
                {
                    return AppAction::None;
                }
                let action = self.composer.handle_key_event(key);
                self.map_composer_action(action)
            }
            Event::Paste(text) => {
                self.composer.insert(&text);
                self.sync_command_popup();
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

    pub(crate) fn attach_image(&mut self, path: PathBuf) {
        self.pending_images.push(path);
    }

    pub(crate) fn take_pending_images(&mut self) -> Vec<PathBuf> {
        std::mem::take(&mut self.pending_images)
    }

    pub(crate) fn restore_pending_images(&mut self, images: Vec<PathBuf>) {
        self.pending_images = images;
    }

    pub(crate) fn set_queued_submissions(&mut self, submissions: Vec<QueuedSubmission>) {
        self.queued_submissions = submissions;
    }

    pub(crate) fn upsert_queued_submission(&mut self, submission: QueuedSubmission) {
        if let Some(existing) = self
            .queued_submissions
            .iter_mut()
            .find(|existing| existing.id == submission.id)
        {
            *existing = submission;
        } else {
            self.queued_submissions.push(submission);
        }
    }

    pub(crate) fn restore_queued_submission_for_edit(
        &mut self,
        submission: QueuedSubmission,
    ) -> bool {
        if !self.composer.is_empty()
            || !self.pending_images.is_empty()
            || !can_restore_submission(&submission)
        {
            return false;
        }
        let submission_id = submission.id.clone();
        let mut text = String::new();
        let mut images = Vec::new();
        for input in submission.input {
            match input {
                app_server_protocol::protocol::v2::UserInput::Text { text: value, .. } => {
                    text = value;
                }
                app_server_protocol::protocol::v2::UserInput::LocalImage { path, .. } => {
                    images.push(PathBuf::from(path));
                }
                _ => return false,
            }
        }
        self.queued_submissions
            .retain(|queued| queued.id != submission_id);
        self.replace_composer(text);
        self.pending_images = images;
        true
    }

    fn map_composer_action(&mut self, action: InputResult) -> AppAction {
        match action {
            InputResult::Submitted(text) => {
                self.command_popup = None;
                AppAction::Submit(text)
            }
            InputResult::Queued(text) => {
                self.command_popup = None;
                AppAction::Queue(text)
            }
            InputResult::Interrupt => AppAction::Interrupt,
            InputResult::DecreaseEffort => AppAction::DecreaseEffort,
            InputResult::IncreaseEffort => AppAction::IncreaseEffort,
            InputResult::PreviousPermissions => AppAction::PreviousPermissions,
            InputResult::NextPermissions => AppAction::NextPermissions,
            InputResult::OpenExternalEditor => AppAction::OpenExternalEditor,
            InputResult::Quit => AppAction::Quit,
            InputResult::Changed => {
                if self.composer.history_search_active() {
                    self.command_popup = None;
                } else {
                    self.sync_command_popup();
                }
                AppAction::None
            }
            InputResult::None => AppAction::None,
        }
    }

    fn complete_slash_command(&mut self, command: SlashCommand) {
        let suffix = if command.requires_argument() { " " } else { "" };
        self.composer
            .replace(format!("/{}{suffix}", command.command()));
        self.command_popup = None;
    }

    fn sync_command_popup(&mut self) {
        if self
            .command_popup
            .as_mut()
            .is_some_and(|popup| popup.update(self.composer.text()))
        {
            return;
        }
        self.command_popup = CommandPopup::for_composer(self.composer.text());
    }

    fn run_local_command(&mut self) -> Option<AppAction> {
        if self.composer.text().split_whitespace().count() != 1 {
            return None;
        }
        let command = command_from_prompt(self.composer.text())?;
        let action = match command {
            SlashCommand::Status => {
                self.open_status_pager();
                AppAction::None
            }
            SlashCommand::Copy => AppAction::CopyLastResponse,
            _ => return None,
        };
        self.composer.replace(String::new());
        self.command_popup = None;
        Some(action)
    }

    fn open_status_pager(&mut self) {
        let cwd = self.cwd.to_string_lossy();
        self.pager_overlay = Some(PagerOverlay::status(
            self.locale,
            StatusFacts {
                thread_id: self.thread_id.as_deref(),
                model: self.model.as_deref(),
                provider: self.model_provider.as_deref(),
                effort: self.reasoning_effort.as_deref(),
                permissions: self.permissions.as_deref(),
                cwd: &cwd,
                status: self.projection.status(),
            },
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::RequestId;
    use app_server_protocol::protocol::v2::{
        CommandExecutionApprovalDecision, CommandExecutionRequestApprovalParams, ServerRequest,
        UserInput,
    };
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
    fn escape_interrupts_only_an_active_turn_and_preserves_the_draft() {
        let mut app = App::default();
        app.composer.insert("keep this draft");
        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE,))),
            AppAction::None
        );

        app.start_turn("turn-1".to_string());
        assert!(app.active_turn_elapsed(Instant::now()).is_some());
        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE,))),
            AppAction::Interrupt
        );
        assert_eq!(app.composer.text(), "keep this draft");
    }

    #[test]
    fn turn_completion_clears_the_active_status_timer() {
        use app_server_protocol::protocol::v2::{
            Turn, TurnCompletedNotification, TurnItemsView, TurnStatus,
        };

        let mut app = App::default();
        app.start_turn("turn-1".to_string());
        app.apply_notification(ServerNotification::TurnCompleted(
            TurnCompletedNotification {
                thread_id: "thread-1".to_string(),
                turn: Turn {
                    id: "turn-1".to_string(),
                    items: Vec::new(),
                    items_view: TurnItemsView::Full,
                    status: TurnStatus::Completed,
                    error: None,
                    started_at: Some(1),
                    completed_at: Some(2),
                    duration_ms: Some(1),
                },
            },
        ));

        assert!(app.projection.active_turn_id().is_none());
        assert!(app.active_turn_elapsed(Instant::now()).is_none());
    }

    #[test]
    fn permission_profile_catalog_is_trimmed_deduplicated_and_used_for_cycles() {
        let mut app = App::default();
        app.set_permission_profiles([
            " custom-read ".to_string(),
            "custom-write".to_string(),
            "custom-read".to_string(),
            "".to_string(),
        ]);

        assert_eq!(
            app.permission_profiles,
            vec!["custom-read".to_string(), "custom-write".to_string()]
        );
        assert_eq!(
            app.cycle_permission_profile(Some("custom-read"), 1),
            "custom-write"
        );
        assert_eq!(
            app.cycle_permission_profile(Some("custom-write"), 1),
            "custom-read"
        );

        app.set_permission_profiles([" ".to_string(), "custom-read".to_string()]);
        assert_eq!(app.permission_profiles, vec!["custom-read".to_string()]);
        app.set_permission_profiles(std::iter::empty::<String>());
        assert!(app.permission_profiles.is_empty());
        assert_eq!(
            app.cycle_permission_profile(Some(":read-only"), 1),
            ":workspace"
        );
    }

    #[test]
    fn an_open_popup_owns_escape_before_active_turn_interruption() {
        let mut app = App::default();
        app.start_turn("turn-1".to_string());
        app.handle_terminal_event(Event::Key(KeyEvent::new(
            KeyCode::Char('/'),
            KeyModifiers::NONE,
        )));
        assert!(app.command_popup.is_some());

        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE,))),
            AppAction::None
        );
        assert!(app.command_popup.is_none());
        assert!(app.projection.active_turn_id().is_some());
    }

    #[test]
    fn history_search_owns_escape_before_active_turn_interruption() {
        let mut app = App::default();
        app.composer.load_history(["previous prompt".to_string()]);
        app.composer.insert("previous");
        app.start_turn("turn-1".to_string());

        app.handle_terminal_event(Event::Key(KeyEvent::new(
            KeyCode::Char('r'),
            KeyModifiers::CONTROL,
        )));
        assert!(app.composer.history_search_active());
        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE,))),
            AppAction::None
        );
        assert!(!app.composer.history_search_active());
        assert_eq!(app.composer.text(), "previous");
        assert!(app.projection.active_turn_id().is_some());
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

    #[test]
    fn copy_shortcut_and_slash_command_do_not_become_turn_input() {
        let mut app = App::default();
        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(
                KeyCode::Char('o'),
                KeyModifiers::CONTROL,
            ))),
            AppAction::CopyLastResponse
        );
        app.composer.insert("/copy");
        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            ))),
            AppAction::CopyLastResponse
        );
        assert!(app.composer.is_empty());
    }

    #[test]
    fn slash_popup_filters_and_executes_immediate_commands() {
        let mut app = App::default();
        for character in ['/', 'm'] {
            assert_eq!(
                app.handle_terminal_event(Event::Key(KeyEvent::new(
                    KeyCode::Char(character),
                    KeyModifiers::NONE,
                ))),
                AppAction::None
            );
        }
        assert_eq!(
            app.command_popup.as_ref().and_then(CommandPopup::selected),
            Some(SlashCommand::Model)
        );

        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            ))),
            AppAction::Submit("/model".to_string())
        );
        assert!(app.command_popup.is_none());
        assert!(app.composer.is_empty());
    }

    #[test]
    fn slash_popup_completes_argument_commands_and_reopens_after_cancelled_input_changes() {
        let mut app = App::default();
        app.handle_terminal_event(Event::Key(KeyEvent::new(
            KeyCode::Char('/'),
            KeyModifiers::NONE,
        )));
        assert!(app.command_popup.is_some());
        app.handle_terminal_event(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)));
        assert!(app.command_popup.is_none());

        app.handle_terminal_event(Event::Key(KeyEvent::new(
            KeyCode::Char('e'),
            KeyModifiers::NONE,
        )));
        assert_eq!(
            app.command_popup.as_ref().and_then(CommandPopup::selected),
            Some(SlashCommand::Effort)
        );
        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            ))),
            AppAction::None
        );
        assert_eq!(app.composer.text(), "/effort ");
        assert!(app.command_popup.is_none());
    }

    #[test]
    fn status_command_opens_an_ephemeral_pager_and_consumes_input_until_closed() {
        let mut app = App::default();
        app.composer.insert("real prompt");
        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            ))),
            AppAction::Submit("real prompt".to_string())
        );
        app.set_thread_id("thread-1".to_string());
        app.set_settings(
            Some("gpt-5".to_string()),
            Some("openai".to_string()),
            Some("high".to_string()),
            Some(":workspace".to_string()),
        );
        app.composer.insert("/status");

        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            ))),
            AppAction::None
        );
        assert!(app.pager_overlay.is_some());
        assert!(app.composer.is_empty());

        app.handle_terminal_event(Event::Key(KeyEvent::new(
            KeyCode::Char('x'),
            KeyModifiers::NONE,
        )));
        assert!(app.composer.is_empty());
        assert!(app.pager_overlay.is_some());
        app.handle_terminal_event(Event::Key(KeyEvent::new(
            KeyCode::Char('q'),
            KeyModifiers::NONE,
        )));
        assert!(app.pager_overlay.is_none());

        app.handle_terminal_event(Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE)));
        assert_eq!(app.composer.text(), "real prompt");
        assert!(app.projection.entries().is_empty());
    }

    #[test]
    fn ctrl_t_opens_transcript_without_copying_or_mutating_conversation_state() {
        let mut app = App::default();
        app.composer.insert("draft");

        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(
                KeyCode::Char('t'),
                KeyModifiers::CONTROL,
            ))),
            AppAction::None
        );
        assert!(
            app.pager_overlay
                .as_ref()
                .is_some_and(PagerOverlay::is_transcript)
        );
        assert_eq!(app.composer.text(), "draft");
        assert!(app.projection.entries().is_empty());

        app.handle_terminal_event(Event::Key(KeyEvent::new(
            KeyCode::Char('t'),
            KeyModifiers::CONTROL,
        )));
        assert!(app.pager_overlay.is_none());
        assert_eq!(app.composer.text(), "draft");
    }

    #[test]
    fn image_shortcut_attaches_and_allows_image_only_submission() {
        let mut app = App::default();
        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(
                KeyCode::Char('v'),
                KeyModifiers::CONTROL,
            ))),
            AppAction::PasteImage
        );
        app.attach_image(PathBuf::from("/tmp/input.png"));
        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            ))),
            AppAction::Submit(String::new())
        );
        assert_eq!(
            app.take_pending_images(),
            vec![PathBuf::from("/tmp/input.png")]
        );
    }

    #[test]
    fn failed_image_submission_can_restore_pending_attachments() {
        let mut app = App::default();
        app.attach_image(PathBuf::from("/tmp/one.png"));
        app.attach_image(PathBuf::from("/tmp/two.png"));

        let images = app.take_pending_images();
        assert!(app.pending_images.is_empty());
        app.restore_pending_images(images);

        assert_eq!(
            app.pending_images,
            vec![PathBuf::from("/tmp/one.png"), PathBuf::from("/tmp/two.png")]
        );
    }

    #[test]
    fn queued_submission_projection_updates_by_id_and_clears_on_thread_change() {
        let queued = |id: &str, text: &str| QueuedSubmission {
            id: id.to_string(),
            input: vec![UserInput::Text {
                text: text.to_string(),
                text_elements: Vec::new(),
            }],
            client_user_message_id: format!("client-{id}"),
        };
        let mut app = App::default();
        app.set_thread_id("thread-1".to_string());
        app.upsert_queued_submission(queued("queue-1", "first"));
        app.upsert_queued_submission(queued("queue-1", "revised"));
        app.upsert_queued_submission(queued("queue-2", "second"));

        assert_eq!(app.queued_submissions.len(), 2);
        assert!(matches!(
            app.queued_submissions[0].input.as_slice(),
            [UserInput::Text { text, .. }] if text == "revised"
        ));

        app.set_thread_id("thread-2".to_string());
        assert!(app.queued_submissions.is_empty());
    }

    #[test]
    fn alt_up_requests_server_delete_before_restoring_the_last_queued_input() {
        let submission = QueuedSubmission {
            id: "queue-1".to_string(),
            input: vec![
                UserInput::LocalImage {
                    detail: None,
                    path: "/tmp/queued.png".to_string(),
                },
                UserInput::Text {
                    text: "revise this follow-up".to_string(),
                    text_elements: Vec::new(),
                },
            ],
            client_user_message_id: "client-queue-1".to_string(),
        };
        let mut app = App::default();
        app.set_queued_submissions(vec![submission.clone()]);

        let action =
            app.handle_terminal_event(Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT)));

        assert_eq!(action, AppAction::EditQueuedSubmission(submission.clone()));
        assert_eq!(app.queued_submissions, vec![submission.clone()]);
        assert!(app.composer.is_empty());
        assert!(app.pending_images.is_empty());

        assert!(app.restore_queued_submission_for_edit(submission));
        assert!(app.queued_submissions.is_empty());
        assert_eq!(app.composer.text(), "revise this follow-up");
        assert_eq!(app.pending_images, vec![PathBuf::from("/tmp/queued.png")]);
    }

    #[test]
    fn alt_up_does_not_offer_a_lossy_or_overwriting_queue_edit() {
        let remote_image = QueuedSubmission {
            id: "queue-remote".to_string(),
            input: vec![UserInput::Image {
                detail: None,
                url: "https://example.test/input.png".to_string(),
            }],
            client_user_message_id: "client-remote".to_string(),
        };
        let mut app = App::default();
        app.set_queued_submissions(vec![remote_image]);
        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT,))),
            AppAction::None
        );

        app.set_queued_submissions(vec![QueuedSubmission {
            id: "queue-text".to_string(),
            input: vec![UserInput::Text {
                text: "queued".to_string(),
                text_elements: Vec::new(),
            }],
            client_user_message_id: "client-text".to_string(),
        }]);
        app.composer.insert("unsent draft");
        assert_eq!(
            app.handle_terminal_event(Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT,))),
            AppAction::None
        );
        assert_eq!(app.composer.text(), "unsent draft");
        assert_eq!(app.queued_submissions.len(), 1);
    }
}
