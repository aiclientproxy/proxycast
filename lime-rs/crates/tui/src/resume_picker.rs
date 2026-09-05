use anyhow::{Context, Result};
use app_server_client::RequestHandle;
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_ARCHIVE, METHOD_THREAD_UNARCHIVE, SortDirection, Thread, ThreadArchiveParams,
    ThreadHistoryMode, ThreadListCwdFilter, ThreadListParams, ThreadListResponse, ThreadSortKey,
    ThreadStatus, ThreadUnarchiveParams, ThreadUnarchiveResponse,
};
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use futures::StreamExt;
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use tokio::sync::mpsc;

use crate::clipboard_paste::normalize_pasted_search_query;
use crate::locale::Locale;
use crate::projection::{EntryKind, TranscriptEntry};
use crate::runtime::{TuiOptions, connect_session};
use crate::text_formatting::center_truncate_path;
use crate::tui::{TerminalGuard, TuiEvent};
use crate::width::display_width;
use crate::wrapping::{RtOptions, adaptive_wrap_line};

mod archive;
mod page_loading;

use page_loading::{PageCursor, PageLoadMode, PaginationState};

#[path = "resume_picker_transcript_preview.rs"]
mod transcript_preview;

const MAX_THREADS: u32 = 100;
const MAX_PAGES: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) struct SessionTarget {
    pub(crate) path: Option<PathBuf>,
    pub(crate) thread_id: String,
    pub(crate) history_mode: Option<ThreadHistoryMode>,
}

#[allow(dead_code)]
impl SessionTarget {
    pub(crate) fn display_label(&self) -> String {
        self.path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| format!("thread {}", self.thread_id))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum SessionSelection {
    StartFresh,
    AgentsOverview,
    Resume(SessionTarget),
    Fork(SessionTarget),
    Exit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum SessionPickerAction {
    Resume,
    Fork,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum SessionPickerLaunchContext {
    Startup,
    ExistingSession { current_thread_id: Option<String> },
}

#[allow(dead_code)]
impl SessionPickerAction {
    pub(crate) fn title(self) -> &'static str {
        match self {
            Self::Resume => "Resume a previous session",
            Self::Fork => "Fork a previous session",
        }
    }

    pub(crate) fn action_label(self) -> &'static str {
        match self {
            Self::Resume => "resume",
            Self::Fork => "fork",
        }
    }

    pub(crate) fn selection(self, target_session: SessionTarget) -> SessionSelection {
        match self {
            Self::Resume => SessionSelection::Resume(target_session),
            Self::Fork => SessionSelection::Fork(target_session),
        }
    }
}

/// Codex-shaped resume picker entry point backed by App Server's Thread list.
pub(crate) async fn run_resume_picker_with_app_server(
    options: &TuiOptions,
) -> Result<Option<String>> {
    run_session_picker_with_action(options, SessionPickerAction::Resume).await
}

#[allow(dead_code)]
pub(crate) async fn run_fork_picker_with_app_server(
    options: &TuiOptions,
) -> Result<Option<String>> {
    let source = run_session_picker_with_action(options, SessionPickerAction::Fork).await?;
    let Some(source) = source else {
        return Ok(None);
    };
    let session = connect_session(options).await?;
    let result = session
        .fork_thread(
            source,
            Some(options.cwd.clone()),
            options.model.clone(),
            options.model_provider.clone(),
        )
        .await;
    let shutdown = session.shutdown().await;
    let response = result?;
    shutdown?;
    Ok(Some(response.thread.id))
}

async fn load_threads_with_handle(
    request_handle: RequestHandle,
    status: SessionStatus,
    filter_cwd: Option<&std::path::Path>,
    query: &str,
    sort_key: ThreadSortKey,
) -> Result<Vec<Thread>> {
    let mut pagination = PaginationState::new();
    let mut cursor = None;
    let mut data = Vec::new();
    let mut seen_cursors = std::collections::HashSet::new();
    let mut seen_thread_ids = std::collections::HashSet::new();

    for request_token in 0..MAX_PAGES {
        if request_token == 0 {
            pagination.reset();
        }
        // Lime's current App Server owns the canonical index; do not switch
        // to Codex's private local state DB mode for later pages.
        let mode = PageLoadMode::StoreDefault;
        pagination.start_load(request_token, None, mode);
        let page: ThreadListResponse = request_handle
            .request(
                app_server_protocol::protocol::v2::METHOD_THREAD_LIST,
                ThreadListParams {
                    cursor: cursor.clone(),
                    limit: Some(MAX_THREADS),
                    sort_key: Some(sort_key),
                    sort_direction: Some(SortDirection::Desc),
                    archived: Some(status == SessionStatus::Archived),
                    cwd: filter_cwd
                        .map(|cwd| ThreadListCwdFilter::One(cwd.to_string_lossy().into_owned())),
                    search_term: (!query.trim().is_empty()).then(|| query.trim().to_string()),
                    ..ThreadListParams::default()
                },
            )
            .await?;
        pagination
            .finish_load(request_token)
            .ok_or_else(|| anyhow::anyhow!("thread list response became stale"))?;
        let next_cursor = page.next_cursor.clone();
        let page_len = page.data.len();
        data.extend(
            page.data
                .into_iter()
                .filter(|thread| seen_thread_ids.insert(thread.id.clone())),
        );
        pagination.complete_page(
            next_cursor.clone().map(PageCursor::AppServer),
            page_len,
            false,
        );
        let Some((PageCursor::AppServer(next), _mode)) = pagination.next_page() else {
            return Ok(filter_threads(data, query));
        };
        if !seen_cursors.insert(next.clone()) {
            return Err(anyhow::anyhow!(
                "thread list pagination repeated cursor {next}"
            ));
        }
        cursor = Some(next);
    }

    Err(anyhow::anyhow!(
        "thread list pagination exceeded {MAX_PAGES} pages"
    ))
}

fn filter_threads(mut threads: Vec<Thread>, query: &str) -> Vec<Thread> {
    threads.retain(|thread| !thread.ephemeral);
    let query = query.trim().to_ascii_lowercase();
    if query.is_empty() {
        return threads;
    }
    threads
        .into_iter()
        .filter(|thread| {
            thread
                .name
                .as_deref()
                .unwrap_or_default()
                .to_ascii_lowercase()
                .contains(&query)
                || thread.preview.to_ascii_lowercase().contains(&query)
                || thread
                    .cwd
                    .to_string_lossy()
                    .to_ascii_lowercase()
                    .contains(&query)
                || thread.id.to_ascii_lowercase().contains(&query)
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PickerAction {
    None,
    Reload,
    MoveUp,
    MoveDown,
    Select,
    Archive,
    Restore,
    ToggleStatus,
    ToggleFilter,
    ToggleSort,
    ToggleDensity,
    ToggleExpanded,
    Cancel,
}

enum PickerLoadEvent {
    Threads {
        token: usize,
        result: Result<Vec<Thread>>,
    },
    Preview {
        thread_id: String,
        result: std::io::Result<Vec<transcript_preview::TranscriptPreviewLine>>,
    },
    Transcript {
        thread_id: String,
        result: std::io::Result<Vec<TranscriptEntry>>,
    },
    Archive {
        thread_id: String,
        result: Result<()>,
    },
    Unarchive {
        thread_id: String,
        result: Box<Result<ThreadUnarchiveResponse>>,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum SessionStatus {
    #[default]
    Active,
    Archived,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum SessionListDensity {
    #[default]
    Comfortable,
    Dense,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SessionTranscriptState {
    Loading,
    Loaded(Vec<TranscriptEntry>),
    Failed,
}

impl SessionListDensity {
    fn toggle(self) -> Self {
        match self {
            Self::Comfortable => Self::Dense,
            Self::Dense => Self::Comfortable,
        }
    }
}

#[derive(Debug)]
struct PickerState {
    action: SessionPickerAction,
    threads: Vec<Thread>,
    selected: usize,
    query: String,
    status: SessionStatus,
    filter_cwd: Option<PathBuf>,
    show_all: bool,
    sort_key: ThreadSortKey,
    density: SessionListDensity,
    transcript_previews: HashMap<String, Vec<transcript_preview::TranscriptPreviewLine>>,
    preview_loading: HashSet<String>,
    expanded_thread_id: Option<String>,
    transcripts: HashMap<String, SessionTranscriptState>,
    archive_state: archive::ArchiveState,
    status_message: Option<String>,
    loading: bool,
    load_token: usize,
}

impl PickerState {
    fn new(
        threads: Vec<Thread>,
        action: SessionPickerAction,
        status: SessionStatus,
        filter_cwd: Option<PathBuf>,
        show_all: bool,
    ) -> Self {
        Self {
            action,
            threads: filter_threads(threads, ""),
            selected: 0,
            query: String::new(),
            status,
            filter_cwd,
            show_all,
            sort_key: ThreadSortKey::UpdatedAt,
            density: SessionListDensity::Comfortable,
            transcript_previews: HashMap::new(),
            preview_loading: HashSet::new(),
            expanded_thread_id: None,
            transcripts: HashMap::new(),
            archive_state: archive::ArchiveState::Idle,
            status_message: None,
            loading: false,
            load_token: 0,
        }
    }

    fn selected_thread_id(&self) -> Option<&str> {
        self.threads
            .get(self.selected)
            .map(|thread| thread.id.as_str())
    }

    fn set_transcript_preview(
        &mut self,
        thread_id: String,
        preview: Vec<transcript_preview::TranscriptPreviewLine>,
    ) {
        self.preview_loading.remove(&thread_id);
        self.transcript_previews.insert(thread_id, preview);
    }

    fn preview_needs_load(&self, thread_id: &str) -> bool {
        !self.transcript_previews.contains_key(thread_id)
            && !self.preview_loading.contains(thread_id)
    }

    fn mark_preview_loading(&mut self, thread_id: impl Into<String>) {
        self.preview_loading.insert(thread_id.into());
    }

    fn toggle_selected_expansion(&mut self) -> Option<String> {
        let thread_id = self.selected_thread_id()?.to_string();
        if self.expanded_thread_id.as_deref() == Some(thread_id.as_str()) {
            self.expanded_thread_id = None;
            return None;
        }
        self.expanded_thread_id = Some(thread_id.clone());
        if !matches!(
            self.transcripts.get(&thread_id),
            Some(SessionTranscriptState::Loaded(_))
        ) {
            self.transcripts
                .insert(thread_id.clone(), SessionTranscriptState::Loading);
            return Some(thread_id);
        }
        None
    }

    fn set_transcript(&mut self, thread_id: String, result: std::io::Result<Vec<TranscriptEntry>>) {
        self.transcripts.insert(
            thread_id,
            match result {
                Ok(entries) => SessionTranscriptState::Loaded(entries),
                Err(_) => SessionTranscriptState::Failed,
            },
        );
    }

    fn begin_load(&mut self) -> usize {
        self.load_token = self.load_token.wrapping_add(1);
        self.loading = true;
        self.status_message = None;
        self.load_token
    }

    fn apply_threads(&mut self, token: usize, threads: Vec<Thread>) {
        if token != self.load_token {
            return;
        }
        self.loading = false;
        self.set_threads(threads);
    }

    fn transcript_preview_text(&self, thread_id: &str) -> Option<String> {
        let lines = self.transcript_previews.get(thread_id)?;
        (!lines.is_empty()).then(|| {
            lines
                .iter()
                .map(|line| {
                    let prefix = match line.speaker {
                        transcript_preview::TranscriptPreviewSpeaker::User => "you: ",
                        transcript_preview::TranscriptPreviewSpeaker::Assistant => "assistant: ",
                    };
                    format!("{prefix}{}", line.text)
                })
                .collect::<Vec<_>>()
                .join(" | ")
        })
    }

    fn handle_event(&mut self, event: Event) -> PickerAction {
        if let Event::Paste(text) = event {
            if let Some(text) = normalize_pasted_search_query(&text) {
                if !self.query.is_empty() && !self.query.ends_with(char::is_whitespace) {
                    self.query.push(' ');
                }
                self.query.push_str(&text);
            }
            return PickerAction::Reload;
        }
        let Event::Key(key) = event else {
            return PickerAction::None;
        };
        if key.kind != KeyEventKind::Press {
            return PickerAction::None;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('d'))
        {
            return PickerAction::Cancel;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && key.code == KeyCode::Char('a')
            && self.status == SessionStatus::Active
            && self.archive_shortcut_available()
        {
            return PickerAction::Archive;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('s') => return PickerAction::ToggleStatus,
                KeyCode::Char('f') => return PickerAction::ToggleFilter,
                KeyCode::Char('r') => return PickerAction::ToggleSort,
                KeyCode::Char('o') => return PickerAction::ToggleDensity,
                KeyCode::Char('e') => return PickerAction::ToggleExpanded,
                _ => {}
            }
        }
        if key.modifiers.is_empty() && key.code == KeyCode::Char('\u{0005}') {
            return PickerAction::ToggleExpanded;
        }
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected = self.selected.saturating_sub(1);
                PickerAction::MoveUp
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if !self.threads.is_empty() {
                    self.selected = (self.selected + 1).min(self.threads.len() - 1);
                }
                PickerAction::MoveDown
            }
            KeyCode::Enter if self.status == SessionStatus::Archived => PickerAction::Restore,
            KeyCode::Enter => PickerAction::Select,
            KeyCode::Backspace => {
                self.query.pop();
                PickerAction::Reload
            }
            KeyCode::Esc if self.query.is_empty() => PickerAction::Cancel,
            KeyCode::Esc => {
                self.query.clear();
                PickerAction::Reload
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::ALT) => {
                self.query.push(c);
                PickerAction::Reload
            }
            _ => PickerAction::None,
        }
    }

    fn toggle_status(&mut self) {
        self.status = match self.status {
            SessionStatus::Active => SessionStatus::Archived,
            SessionStatus::Archived => SessionStatus::Active,
        };
        self.selected = 0;
        self.transcript_previews.clear();
        self.preview_loading.clear();
        self.expanded_thread_id = None;
        self.transcripts.clear();
    }

    fn toggle_filter(&mut self) {
        self.show_all = !self.show_all;
        self.selected = 0;
        self.transcript_previews.clear();
        self.preview_loading.clear();
        self.expanded_thread_id = None;
        self.transcripts.clear();
    }

    fn toggle_sort(&mut self) {
        self.sort_key = match self.sort_key {
            ThreadSortKey::UpdatedAt => ThreadSortKey::CreatedAt,
            ThreadSortKey::CreatedAt
            | ThreadSortKey::RecencyAt
            | ThreadSortKey::SectionPosition => ThreadSortKey::UpdatedAt,
        };
        self.selected = 0;
        self.transcript_previews.clear();
        self.preview_loading.clear();
        self.expanded_thread_id = None;
        self.transcripts.clear();
    }

    fn set_threads(&mut self, threads: Vec<Thread>) {
        self.threads = filter_threads(threads, &self.query);
        self.selected = self.selected.min(self.threads.len().saturating_sub(1));
        self.transcript_previews.clear();
        self.preview_loading.clear();
        self.expanded_thread_id = None;
        self.transcripts.clear();
    }
}

fn spawn_thread_load(
    request_handle: RequestHandle,
    sender: &mpsc::UnboundedSender<PickerLoadEvent>,
    picker: &mut PickerState,
) {
    let token = picker.begin_load();
    let status = picker.status;
    let cwd = (!picker.show_all)
        .then(|| picker.filter_cwd.clone())
        .flatten();
    let query = picker.query.clone();
    let sort_key = picker.sort_key;
    let sender = sender.clone();
    tokio::spawn(async move {
        let result =
            load_threads_with_handle(request_handle, status, cwd.as_deref(), &query, sort_key)
                .await;
        let _ = sender.send(PickerLoadEvent::Threads { token, result });
    });
}

fn spawn_preview_load(
    request_handle: RequestHandle,
    sender: &mpsc::UnboundedSender<PickerLoadEvent>,
    picker: &mut PickerState,
    thread_id: String,
) {
    if !picker.preview_needs_load(&thread_id) {
        return;
    }
    picker.mark_preview_loading(thread_id.clone());
    let sender = sender.clone();
    tokio::spawn(async move {
        let result = load_transcript_preview_with_handle(request_handle, thread_id.clone()).await;
        let _ = sender.send(PickerLoadEvent::Preview { thread_id, result });
    });
}

fn spawn_transcript_load(
    request_handle: RequestHandle,
    sender: &mpsc::UnboundedSender<PickerLoadEvent>,
    thread_id: String,
) {
    let sender = sender.clone();
    tokio::spawn(async move {
        let result = crate::thread_transcript::load_session_transcript_with_handle(
            request_handle,
            thread_id.clone(),
        )
        .await;
        let _ = sender.send(PickerLoadEvent::Transcript { thread_id, result });
    });
}

async fn load_transcript_preview_with_handle(
    request_handle: RequestHandle,
    thread_id: String,
) -> std::io::Result<Vec<transcript_preview::TranscriptPreviewLine>> {
    let loaded_entries =
        crate::thread_transcript::load_session_transcript_with_handle(request_handle, thread_id)
            .await?;
    let preview_entries = loaded_entries
        .into_iter()
        .filter_map(|entry| {
            let speaker = match entry.kind {
                crate::projection::EntryKind::User => {
                    transcript_preview::TranscriptPreviewSpeaker::User
                }
                crate::projection::EntryKind::Assistant => {
                    transcript_preview::TranscriptPreviewSpeaker::Assistant
                }
                _ => return None,
            };
            Some((speaker, entry.text))
        })
        .collect();
    transcript_preview::preview_from_entries(preview_entries)
}

fn spawn_archive_request(
    request_handle: RequestHandle,
    sender: &mpsc::UnboundedSender<PickerLoadEvent>,
    thread_id: String,
) {
    let sender = sender.clone();
    tokio::spawn(async move {
        let result = request_handle
            .request::<_, app_server_protocol::protocol::v2::ThreadArchiveResponse>(
                METHOD_THREAD_ARCHIVE,
                ThreadArchiveParams {
                    thread_id: thread_id.clone(),
                },
            )
            .await
            .map(|_| ())
            .map_err(anyhow::Error::from);
        let _ = sender.send(PickerLoadEvent::Archive { thread_id, result });
    });
}

fn spawn_unarchive_request(
    request_handle: RequestHandle,
    sender: &mpsc::UnboundedSender<PickerLoadEvent>,
    thread_id: String,
) {
    let sender = sender.clone();
    tokio::spawn(async move {
        let result = request_handle
            .request::<_, ThreadUnarchiveResponse>(
                METHOD_THREAD_UNARCHIVE,
                ThreadUnarchiveParams {
                    thread_id: thread_id.clone(),
                },
            )
            .await
            .map_err(anyhow::Error::from);
        let _ = sender.send(PickerLoadEvent::Unarchive {
            thread_id,
            result: Box::new(result),
        });
    });
}

#[allow(dead_code)]
pub(crate) async fn run_session_picker_with_app_server(
    options: &TuiOptions,
) -> Result<Option<String>> {
    run_session_picker_with_action(options, SessionPickerAction::Resume).await
}

async fn run_session_picker_with_action(
    options: &TuiOptions,
    action: SessionPickerAction,
) -> Result<Option<String>> {
    let session = connect_session(options).await?;
    let request_handle = session.request_handle();
    let (load_tx, mut load_rx) = mpsc::unbounded_channel();
    let mut picker = PickerState::new(
        Vec::new(),
        action,
        SessionStatus::Active,
        Some(options.cwd.clone()),
        false,
    );
    spawn_thread_load(request_handle.clone(), &load_tx, &mut picker);
    let locale = Locale::resolve(options.locale.as_deref());
    let mut terminal = match TerminalGuard::enter().context("failed to initialize terminal") {
        Ok(terminal) => terminal,
        Err(error) => {
            let _ = session.shutdown().await;
            return Err(error);
        }
    };
    let mut input = terminal.event_stream();
    let selected = loop {
        if let Some(thread_id) = picker.selected_thread_id().map(ToOwned::to_owned) {
            spawn_preview_load(request_handle.clone(), &load_tx, &mut picker, thread_id);
        }
        terminal
            .terminal_mut()
            .draw(|frame| render_with_locale(frame, &picker, locale))
            .context("failed to render session picker")?;
        tokio::select! {
            load_event = load_rx.recv() => {
                let Some(load_event) = load_event else { break None; };
                match load_event {
                    PickerLoadEvent::Threads { token, result } => match result {
                        Ok(threads) => picker.apply_threads(token, threads),
                        Err(error) if token == picker.load_token => {
                            picker.loading = false;
                            picker.status_message = Some(error.to_string());
                        }
                        Err(_) => {}
                    },
                    PickerLoadEvent::Preview { thread_id, result } => {
                        picker.set_transcript_preview(thread_id, result.unwrap_or_default());
                    }
                    PickerLoadEvent::Transcript { thread_id, result } => {
                        picker.set_transcript(thread_id, result);
                    }
                    PickerLoadEvent::Archive { thread_id, result } => {
                        picker.handle_archive_result(thread_id, result);
                    }
                    PickerLoadEvent::Unarchive { thread_id, result } => {
                        let selection = picker.handle_unarchive_result(thread_id, *result);
                        if let Some(crate::resume_picker::SessionSelection::Resume(target)) = selection {
                            break Some(target.thread_id);
                        }
                    }
                }
            }
            event = input.next() => {
                let Some(event) = event else { break None; };
                let event = match event {
                    TuiEvent::Key(key) => Event::Key(key),
                    TuiEvent::Paste(text) => Event::Paste(text),
                    TuiEvent::Resize(size) => Event::Resize(size.width, size.height),
                    TuiEvent::FocusGained => Event::FocusGained,
                    TuiEvent::FocusLost => Event::FocusLost,
                    TuiEvent::Draw | TuiEvent::Resume => continue,
                };
                let action = picker.handle_event(event);
                match action {
                    PickerAction::Select => break picker.selected_thread_id().map(ToOwned::to_owned),
                    PickerAction::Restore => {
                        if let Some(thread_id) = picker.request_unarchive_for_selected_session() {
                            spawn_unarchive_request(request_handle.clone(), &load_tx, thread_id);
                        }
                    }
                    PickerAction::Archive => {
                        if let Some(thread_id) = picker.request_archive_for_selected_session() {
                            spawn_archive_request(request_handle.clone(), &load_tx, thread_id);
                        }
                    }
                    PickerAction::ToggleStatus | PickerAction::ToggleFilter | PickerAction::ToggleSort | PickerAction::Reload => {
                        match action {
                            PickerAction::ToggleStatus => picker.toggle_status(),
                            PickerAction::ToggleFilter => picker.toggle_filter(),
                            PickerAction::ToggleSort => picker.toggle_sort(),
                            PickerAction::Reload => {}
                            _ => unreachable!(),
                        }
                        spawn_thread_load(request_handle.clone(), &load_tx, &mut picker);
                    }
                    PickerAction::ToggleDensity => {
                        picker.density = picker.density.toggle();
                    }
                    PickerAction::ToggleExpanded => {
                        if let Some(thread_id) = picker.toggle_selected_expansion() {
                            spawn_transcript_load(request_handle.clone(), &load_tx, thread_id);
                        }
                    }
                    PickerAction::Cancel => break None,
                    PickerAction::None | PickerAction::MoveUp | PickerAction::MoveDown => {}
                }
            }
        }
    };
    let restore_result = terminal.restore().context("failed to restore terminal");
    let shutdown_result = session.shutdown().await;
    restore_result?;
    shutdown_result?;
    Ok(selected)
}

#[cfg(test)]
fn render(frame: &mut Frame<'_>, picker: &PickerState) {
    render_with_locale(frame, picker, Locale::default());
}

fn render_with_locale(frame: &mut Frame<'_>, picker: &PickerState, locale: Locale) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(1),
            Constraint::Length(2),
        ])
        .split(area);

    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            format!(
                " {}: {} ",
                locale.resume_label(),
                locale.resume_picker_title(matches!(picker.action, SessionPickerAction::Fork))
            ),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::styled(locale.resume_title(), Style::default().fg(Color::DarkGray)),
    ]));
    frame.render_widget(title, chunks[0]);

    let search = if picker.query.is_empty() {
        locale.resume_search_placeholder().to_string()
    } else {
        format!("{}: {}", locale.resume_search_placeholder(), picker.query)
    };
    let toolbar = format!(
        "{} | {} | {}",
        locale.resume_status_label(picker.status == SessionStatus::Archived),
        locale.resume_filter_label(picker.show_all),
        locale.resume_sort_label(picker.sort_key == ThreadSortKey::CreatedAt),
    );
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("{search} | {toolbar}"),
            Style::default().fg(Color::DarkGray),
        ))),
        chunks[0].inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 0,
        }),
    );

    let items = picker
        .threads
        .iter()
        .enumerate()
        .map(|(index, thread)| {
            let preview = picker.transcript_preview_text(&thread.id);
            let title = thread_line_with_preview(
                thread,
                usize::from(chunks[1].width),
                locale,
                preview.as_deref(),
            );
            let is_expanded = picker.selected == index
                && picker.expanded_thread_id.as_deref() == Some(thread.id.as_str());
            if is_expanded {
                let mut lines = vec![title];
                lines.extend(render_expanded_session_details(
                    thread,
                    picker,
                    usize::from(chunks[1].width),
                    locale,
                ));
                ListItem::new(lines)
            } else if picker.density == SessionListDensity::Dense {
                ListItem::new(title)
            } else {
                let metadata = format!(
                    "  {}  {}",
                    thread.cwd.display(),
                    if thread.updated_at == 0 {
                        String::from("-")
                    } else {
                        thread.updated_at.to_string()
                    }
                );
                ListItem::new(vec![
                    title,
                    Line::from(Span::styled(
                        truncate_display(&metadata, usize::from(chunks[1].width)),
                        Style::default().fg(Color::DarkGray),
                    )),
                ])
            }
        })
        .collect::<Vec<_>>();
    let mut state = ListState::default();
    if !picker.threads.is_empty() {
        state.select(Some(picker.selected));
    }
    frame.render_stateful_widget(
        List::new(items)
            .block(Block::default().borders(Borders::TOP | Borders::BOTTOM))
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("> "),
        chunks[1],
        &mut state,
    );
    let footer = if picker.loading {
        locale.resume_loading().to_string()
    } else if picker.threads.is_empty() {
        locale.resume_empty().to_string()
    } else if let Some(message) = picker.status_message.as_deref() {
        message.to_string()
    } else {
        format!(
            "{} | {}  {} | {}",
            locale.resume_enter_hint(
                matches!(picker.action, SessionPickerAction::Fork),
                picker.status == SessionStatus::Archived,
            ),
            locale.resume_controls_hint(),
            locale.resume_expand_hint(),
            locale.resume_density_label(picker.density == SessionListDensity::Dense),
        )
    };
    frame.render_widget(
        Paragraph::new(footer)
            .style(Style::default().fg(Color::DarkGray))
            .wrap(ratatui::widgets::Wrap { trim: true }),
        chunks[2],
    );
}

fn render_expanded_session_details(
    thread: &Thread,
    picker: &PickerState,
    width: usize,
    locale: Locale,
) -> Vec<Line<'static>> {
    let indent = "  ";
    let available = width.saturating_sub(display_width(indent));
    let title = thread
        .name
        .as_deref()
        .filter(|name| !name.trim().is_empty())
        .unwrap_or(thread.preview.as_str());
    let cwd = center_truncate_path(&thread.cwd.to_string_lossy(), available.saturating_sub(16));
    let state = match &thread.status {
        ThreadStatus::Active { .. } => "active",
        ThreadStatus::Idle => "idle",
        ThreadStatus::NotLoaded => "not loaded",
        ThreadStatus::SystemError => "error",
    };
    let mut lines = vec![
        metadata_line(
            locale.thread_label(),
            &format!("{title} ({})", thread.id),
            width,
        ),
        metadata_line(locale.cwd_label(), &cwd, width),
        metadata_line(locale.state_label(), &locale.thread_state(state), width),
        metadata_line(
            locale.created_label(),
            &thread.created_at.to_string(),
            width,
        ),
        metadata_line(
            locale.updated_label(),
            &thread.updated_at.to_string(),
            width,
        ),
    ];

    match picker.transcripts.get(&thread.id) {
        Some(SessionTranscriptState::Loading) => {
            lines.push(Line::from(Span::styled(
                format!("{indent}{}", locale.resume_transcript_loading()),
                Style::default().fg(Color::DarkGray),
            )));
        }
        Some(SessionTranscriptState::Failed) => {
            lines.push(Line::from(Span::styled(
                format!("{indent}{}", locale.resume_transcript_failed()),
                Style::default().fg(Color::Red),
            )));
        }
        Some(SessionTranscriptState::Loaded(entries)) if entries.is_empty() => {
            lines.push(Line::from(Span::styled(
                format!("{indent}{}", locale.resume_transcript_empty()),
                Style::default().fg(Color::DarkGray),
            )));
        }
        Some(SessionTranscriptState::Loaded(entries)) => {
            lines.push(Line::from(Span::styled(
                format!("{indent}{}", locale.transcript_title()),
                Style::default().add_modifier(Modifier::BOLD),
            )));
            for entry in entries {
                lines.extend(render_transcript_content_lines(entry, width));
            }
        }
        None => {}
    }
    lines
}

fn metadata_line(label: &str, value: &str, width: usize) -> Line<'static> {
    let prefix = format!("  {label}: ");
    let available = width.saturating_sub(display_width(&prefix));
    Line::from(vec![
        Span::styled(prefix, Style::default().fg(Color::DarkGray)),
        Span::raw(truncate_display(value, available)),
    ])
}

fn render_transcript_content_lines(entry: &TranscriptEntry, width: usize) -> Vec<Line<'static>> {
    let (prefix, style) = match entry.kind {
        EntryKind::User => ("  you  ", Style::default().fg(Color::Cyan)),
        EntryKind::Assistant => ("  assistant  ", Style::default().fg(Color::Green)),
        EntryKind::Reasoning => ("  reasoning  ", Style::default().fg(Color::DarkGray)),
        EntryKind::Command => ("  command  ", Style::default().fg(Color::Yellow)),
        _ => ("  event  ", Style::default().fg(Color::DarkGray)),
    };
    let content_width = width.saturating_sub(display_width(prefix)).max(1);
    let mut output = Vec::new();
    for text in entry.text.lines() {
        let source_line = Line::from(Span::raw(text.to_string()));
        let wrapped = adaptive_wrap_line(&source_line, RtOptions::new(content_width));
        if wrapped.is_empty() {
            output.push(Line::from(Span::styled(prefix, style)));
            continue;
        }
        for (index, line) in wrapped.into_iter().enumerate() {
            let line_text = line.to_string();
            let marker = if index == 0 {
                prefix.to_string()
            } else {
                " ".repeat(display_width(prefix))
            };
            output.push(Line::from(vec![
                Span::styled(marker, style),
                Span::raw(line_text),
            ]));
        }
    }
    if output.is_empty() {
        output.push(Line::from(Span::styled(prefix, style)));
    }
    output
}

fn thread_line_with_preview(
    thread: &Thread,
    width: usize,
    locale: Locale,
    transcript_preview: Option<&str>,
) -> Line<'static> {
    let title = thread
        .name
        .as_deref()
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| {
            let preview = transcript_preview.unwrap_or(thread.preview.as_str());
            if preview.trim().is_empty() {
                locale.untitled_conversation()
            } else {
                preview
            }
        });
    let state = match &thread.status {
        ThreadStatus::Active { .. } => "active",
        ThreadStatus::Idle => "idle",
        ThreadStatus::NotLoaded => "not loaded",
        ThreadStatus::SystemError => "error",
    };
    let cwd = thread.cwd.to_string_lossy();
    let max_width = width.saturating_sub(2);
    let state_text = format!("[{}]", locale.thread_state(state));
    let reserved = display_width(&format!("{title}  {state_text}  "));
    let cwd = center_truncate_path(&cwd, max_width.saturating_sub(reserved));
    let truncated = truncate_display(&format!("{title}  {state_text}  {cwd}"), max_width);
    Line::from(Span::raw(truncated))
}

fn truncate_display(text: &str, max_width: usize) -> String {
    if display_width(text) <= max_width {
        return text.to_string();
    }
    if max_width <= 1 {
        return "…".to_string();
    }
    let mut output = String::new();
    let mut width = 0;
    for grapheme in unicode_segmentation::UnicodeSegmentation::graphemes(text, true) {
        let grapheme_width = display_width(grapheme);
        if width + grapheme_width + 1 > max_width {
            break;
        }
        output.push_str(grapheme);
        width += grapheme_width;
    }
    output.push('…');
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::protocol::v2::{SessionSource, ThreadActiveFlag, ThreadHistoryMode};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use std::path::PathBuf;

    fn thread(id: &str, preview: &str, ephemeral: bool) -> Thread {
        Thread {
            id: id.to_string(),
            extra: None,
            session_id: format!("session-{id}"),
            forked_from_id: None,
            parent_thread_id: None,
            preview: preview.to_string(),
            ephemeral,
            section: None,
            section_entered_at: None,
            project_id: None,
            history_mode: ThreadHistoryMode::default(),
            model_provider: "fixture".to_string(),
            created_at: 1,
            updated_at: 1,
            recency_at: None,
            status: ThreadStatus::Active {
                active_flags: vec![ThreadActiveFlag::WaitingOnUserInput],
            },
            path: None,
            cwd: PathBuf::from("/workspace"),
            cli_version: "test".to_string(),
            source: SessionSource::Cli,
            can_accept_direct_input: Some(true),
            thread_source: None,
            agent_nickname: None,
            agent_role: None,
            git_info: None,
            name: None,
            turns: Vec::new(),
        }
    }

    #[test]
    fn picker_filters_ephemeral_threads_and_navigates() {
        let mut picker = PickerState::new(
            vec![
                thread("hidden", "temporary", true),
                thread("one", "first", false),
                thread("two", "second", false),
            ],
            SessionPickerAction::Resume,
            SessionStatus::Active,
            None,
            true,
        );
        assert_eq!(picker.threads.len(), 2);
        assert_eq!(picker.selected_thread_id(), Some("one"));
        assert_eq!(
            picker.handle_event(Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Down,
                KeyModifiers::NONE,
            ))),
            PickerAction::MoveDown
        );
        assert_eq!(picker.selected_thread_id(), Some("two"));
        assert_eq!(
            picker.handle_event(Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            ))),
            PickerAction::Select
        );
    }

    #[test]
    fn picker_render_is_bounded_for_narrow_unicode_terminal() {
        let picker = PickerState::new(
            vec![thread("one", "你好，这是一段很长的预览", false)],
            SessionPickerAction::Resume,
            SessionStatus::Active,
            None,
            true,
        );
        let mut terminal = Terminal::new(TestBackend::new(24, 8)).expect("terminal");
        terminal.draw(|frame| render(frame, &picker)).expect("draw");
        assert!(
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .all(|cell| cell.symbol().chars().count() <= 1)
        );
    }

    #[test]
    fn truncation_preserves_display_width() {
        let text = truncate_display("你好abc", 5);
        assert!(display_width(&text) <= 5);
        assert!(text.ends_with('…'));
    }

    #[test]
    fn session_picker_action_keeps_resume_and_fork_selection_distinct() {
        let target = SessionTarget {
            path: Some(PathBuf::from("/workspace")),
            thread_id: String::from("thread-1"),
            history_mode: Some(ThreadHistoryMode::Legacy),
        };
        assert_eq!(SessionPickerAction::Resume.action_label(), "resume");
        assert_eq!(
            SessionPickerAction::Fork.selection(target.clone()),
            SessionSelection::Fork(target)
        );
    }

    #[test]
    fn session_target_uses_thread_id_when_path_is_not_available() {
        let target = SessionTarget {
            path: None,
            thread_id: String::from("thread-1"),
            history_mode: None,
        };
        assert_eq!(target.display_label(), "thread thread-1");
    }

    #[test]
    fn picker_search_filters_preview_name_path_and_thread_id() {
        let mut named = thread("named", "unrelated", false);
        named.name = Some("Release checklist".to_string());
        let mut picker = PickerState::new(
            vec![
                named,
                thread("preview-id", "Need a deployment review", false),
                thread("other", "unrelated", false),
            ],
            SessionPickerAction::Resume,
            SessionStatus::Active,
            Some(PathBuf::from("/workspace")),
            true,
        );

        picker.query = "release".to_string();
        picker.set_threads(picker.threads.clone());
        assert_eq!(picker.threads.len(), 1);
        assert_eq!(picker.selected_thread_id(), Some("named"));

        picker.query = "preview-id".to_string();
        picker.set_threads(vec![thread("preview-id", "unrelated", false)]);
        assert_eq!(picker.selected_thread_id(), Some("preview-id"));
    }

    #[test]
    fn picker_controls_match_codex_resume_shortcut_shapes() {
        let mut picker = PickerState::new(
            vec![thread("one", "first", false)],
            SessionPickerAction::Resume,
            SessionStatus::Active,
            Some(PathBuf::from("/workspace")),
            false,
        );
        assert_eq!(
            picker.handle_event(Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Char('s'),
                KeyModifiers::CONTROL,
            ))),
            PickerAction::ToggleStatus
        );
        picker.toggle_status();
        assert_eq!(picker.status, SessionStatus::Archived);
        assert_eq!(
            picker.handle_event(Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Char('o'),
                KeyModifiers::CONTROL,
            ))),
            PickerAction::ToggleDensity
        );
        picker.density = picker.density.toggle();
        assert_eq!(picker.density, SessionListDensity::Dense);
    }

    #[test]
    fn ctrl_e_toggles_selected_session_expansion_and_transcript_loading() {
        let mut picker = PickerState::new(
            vec![thread("one", "first", false)],
            SessionPickerAction::Resume,
            SessionStatus::Active,
            None,
            true,
        );
        let key = Event::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('e'),
            KeyModifiers::CONTROL,
        ));
        assert_eq!(
            picker.handle_event(key.clone()),
            PickerAction::ToggleExpanded
        );
        assert_eq!(
            picker.toggle_selected_expansion(),
            Some(String::from("one"))
        );
        assert_eq!(picker.expanded_thread_id.as_deref(), Some("one"));
        assert!(matches!(
            picker.transcripts.get("one"),
            Some(SessionTranscriptState::Loading)
        ));

        assert_eq!(picker.handle_event(key), PickerAction::ToggleExpanded);
        assert_eq!(picker.toggle_selected_expansion(), None);
        assert_eq!(picker.expanded_thread_id, None);
    }

    #[test]
    fn raw_ctrl_e_keycode_matches_codex_resume_shortcut() {
        let mut picker = PickerState::new(
            vec![thread("one", "first", false)],
            SessionPickerAction::Resume,
            SessionStatus::Active,
            None,
            true,
        );
        assert_eq!(
            picker.handle_event(Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Char('\u{0005}'),
                KeyModifiers::NONE,
            ))),
            PickerAction::ToggleExpanded
        );
    }

    #[test]
    fn expanded_transcript_uses_canonical_entries_and_stays_bounded() {
        let mut picker = PickerState::new(
            vec![thread("one", "first", false)],
            SessionPickerAction::Resume,
            SessionStatus::Active,
            None,
            true,
        );
        picker.expanded_thread_id = Some(String::from("one"));
        picker.transcripts.insert(
            String::from("one"),
            SessionTranscriptState::Loaded(vec![TranscriptEntry {
                id: String::from("entry-1"),
                kind: EntryKind::Assistant,
                text: String::from("a long assistant response that wraps safely"),
                streaming: false,
                status: None,
                summary: Vec::new(),
            }]),
        );
        let lines = render_expanded_session_details(&picker.threads[0], &picker, 24, Locale::EnUs);
        assert!(
            lines
                .iter()
                .any(|line| line.to_string().contains("assistant"))
        );
        assert!(
            lines
                .iter()
                .all(|line| display_width(&line.to_string()) <= 24)
        );
    }

    #[test]
    fn transcript_load_event_records_loaded_and_failed_states() {
        let mut picker = PickerState::new(
            vec![thread("one", "first", false)],
            SessionPickerAction::Resume,
            SessionStatus::Active,
            None,
            true,
        );
        picker.set_transcript(
            String::from("one"),
            Ok(vec![TranscriptEntry {
                id: String::from("entry-1"),
                kind: EntryKind::User,
                text: String::from("hello"),
                streaming: false,
                status: None,
                summary: Vec::new(),
            }]),
        );
        assert!(matches!(
            picker.transcripts.get("one"),
            Some(SessionTranscriptState::Loaded(entries)) if entries.len() == 1
        ));

        picker.set_transcript(String::from("one"), Err(std::io::Error::other("offline")));
        assert!(matches!(
            picker.transcripts.get("one"),
            Some(SessionTranscriptState::Failed)
        ));
    }

    #[test]
    fn picker_only_reloads_for_query_mutations() {
        let mut picker = PickerState::new(
            vec![thread("one", "first", false)],
            SessionPickerAction::Resume,
            SessionStatus::Active,
            Some(PathBuf::from("/workspace")),
            false,
        );

        assert_eq!(
            picker.handle_event(Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::F(5),
                KeyModifiers::NONE,
            ))),
            PickerAction::None
        );
        assert_eq!(
            picker.handle_event(Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Char('x'),
                KeyModifiers::NONE,
            ))),
            PickerAction::Reload
        );
    }

    #[test]
    fn stale_thread_load_results_do_not_clear_newer_loading_state() {
        let mut picker = PickerState::new(
            Vec::new(),
            SessionPickerAction::Resume,
            SessionStatus::Active,
            None,
            true,
        );
        let first = picker.begin_load();
        let second = picker.begin_load();
        picker.apply_threads(first, vec![thread("stale", "stale", false)]);

        assert!(picker.loading);
        assert!(picker.threads.is_empty());

        picker.apply_threads(second, vec![thread("fresh", "fresh", false)]);
        assert!(!picker.loading);
        assert_eq!(picker.selected_thread_id(), Some("fresh"));
    }
}
