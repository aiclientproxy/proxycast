use anyhow::{Context, Result};
use app_server_protocol::protocol::v2::{Thread, ThreadStatus};
use crossterm::event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers};
use futures::StreamExt;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph};
use ratatui::Frame;

use crate::app_server_session::AppServerSession;
use crate::runtime::{stdio_config, TuiOptions};
use crate::terminal::TerminalGuard;
use crate::width::display_width;

const MAX_THREADS: u32 = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PickerAction {
    None,
    MoveUp,
    MoveDown,
    Select,
    Cancel,
}

#[derive(Debug, Default)]
struct SessionPicker {
    threads: Vec<Thread>,
    selected: usize,
}

impl SessionPicker {
    fn new(threads: Vec<Thread>) -> Self {
        Self {
            threads: threads
                .into_iter()
                .filter(|thread| !thread.ephemeral)
                .collect(),
            selected: 0,
        }
    }

    fn selected_thread_id(&self) -> Option<&str> {
        self.threads
            .get(self.selected)
            .map(|thread| thread.id.as_str())
    }

    fn handle_event(&mut self, event: Event) -> PickerAction {
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
            KeyCode::Enter => PickerAction::Select,
            KeyCode::Esc | KeyCode::Char('q') => PickerAction::Cancel,
            _ => PickerAction::None,
        }
    }
}

pub(crate) async fn pick_session(options: &TuiOptions) -> Result<Option<String>> {
    let config = stdio_config(options)?;
    let session = AppServerSession::connect(config).await?;
    let threads = match session.list_threads(MAX_THREADS).await {
        Ok(response) => response.data,
        Err(error) => {
            let _ = session.shutdown().await;
            return Err(error);
        }
    };
    let mut picker = SessionPicker::new(threads);
    let mut terminal = match TerminalGuard::enter().context("failed to initialize terminal") {
        Ok(terminal) => terminal,
        Err(error) => {
            let _ = session.shutdown().await;
            return Err(error);
        }
    };
    let mut input = EventStream::new();
    let selected = loop {
        terminal
            .terminal_mut()
            .draw(|frame| render(frame, &picker))
            .context("failed to render session picker")?;
        let Some(event) = input.next().await else {
            break None;
        };
        let action = picker.handle_event(event.context("failed to read terminal event")?);
        match action {
            PickerAction::Select => break picker.selected_thread_id().map(ToOwned::to_owned),
            PickerAction::Cancel => break None,
            PickerAction::None | PickerAction::MoveUp | PickerAction::MoveDown => {}
        }
    };
    let restore_result = terminal.restore().context("failed to restore terminal");
    let shutdown_result = session.shutdown().await;
    restore_result?;
    shutdown_result?;
    Ok(selected)
}

fn render(frame: &mut Frame<'_>, picker: &SessionPicker) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(area);

    let title = Paragraph::new(Line::from(vec![
        Span::styled(" Resume ", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(
            "Select a conversation",
            Style::default().fg(Color::DarkGray),
        ),
    ]));
    frame.render_widget(title, chunks[0]);

    let items = picker
        .threads
        .iter()
        .map(|thread| ListItem::new(thread_line(thread, usize::from(chunks[1].width))))
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
    let footer = if picker.threads.is_empty() {
        "No resumable conversations. Press Esc to cancel."
    } else {
        "Up/Down move   Enter resume   Esc cancel"
    };
    frame.render_widget(
        Paragraph::new(footer).style(Style::default().fg(Color::DarkGray)),
        chunks[2],
    );
}

fn thread_line(thread: &Thread, width: usize) -> Line<'static> {
    let title = thread
        .name
        .as_deref()
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| {
            if thread.preview.trim().is_empty() {
                "Untitled conversation"
            } else {
                thread.preview.as_str()
            }
        });
    let state = match &thread.status {
        ThreadStatus::Active { .. } => "active",
        ThreadStatus::Idle => "idle",
        ThreadStatus::NotLoaded => "not loaded",
        ThreadStatus::SystemError => "error",
    };
    let cwd = thread.cwd.to_string_lossy();
    let text = format!("{title}  [{state}]  {cwd}");
    let max_width = width.saturating_sub(2);
    let truncated = truncate_display(&text, max_width);
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
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
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
        let mut picker = SessionPicker::new(vec![
            thread("hidden", "temporary", true),
            thread("one", "first", false),
            thread("two", "second", false),
        ]);
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
        let picker = SessionPicker::new(vec![thread("one", "你好，这是一段很长的预览", false)]);
        let mut terminal = Terminal::new(TestBackend::new(24, 8)).expect("terminal");
        terminal.draw(|frame| render(frame, &picker)).expect("draw");
        assert!(terminal.backend().buffer().content().iter().all(|cell| cell
            .symbol()
            .chars()
            .count()
            <= 1));
    }

    #[test]
    fn truncation_preserves_display_width() {
        let text = truncate_display("你好abc", 5);
        assert!(display_width(&text) <= 5);
        assert!(text.ends_with('…'));
    }
}
