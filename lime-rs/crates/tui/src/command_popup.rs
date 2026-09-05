use crossterm::event::{Event, KeyCode, KeyEventKind};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};

use crate::line_truncation::truncate_line_with_ellipsis_if_overflow;
use crate::locale::Locale;
use crate::slash_command::{SlashCommand, command_filter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandPopupAction {
    Pass,
    Consumed,
    Cancel,
    Complete(SlashCommand),
    Execute(SlashCommand),
}

#[derive(Debug)]
pub(crate) struct CommandPopup {
    filter: String,
    selected: usize,
}

impl CommandPopup {
    pub(crate) fn for_composer(text: &str) -> Option<Self> {
        let filter = command_filter(text)?.to_ascii_lowercase();
        let popup = Self {
            filter,
            selected: 0,
        };
        (!popup.matches().is_empty()).then_some(popup)
    }

    pub(crate) fn update(&mut self, text: &str) -> bool {
        let Some(filter) = command_filter(text) else {
            return false;
        };
        self.filter = filter.to_ascii_lowercase();
        let matches = self.matches();
        if matches.is_empty() {
            return false;
        }
        self.selected = self.selected.min(matches.len() - 1);
        true
    }

    pub(crate) fn commands(&self) -> Vec<SlashCommand> {
        self.matches()
    }

    pub(crate) fn selected(&self) -> Option<SlashCommand> {
        self.matches().get(self.selected).copied()
    }

    pub(crate) fn handle_event(&mut self, event: &Event) -> CommandPopupAction {
        let Event::Key(key) = event else {
            return CommandPopupAction::Pass;
        };
        if key.kind != KeyEventKind::Press {
            return CommandPopupAction::Pass;
        }
        match key.code {
            KeyCode::Up => {
                let len = self.matches().len();
                if len > 0 {
                    self.selected = self.selected.checked_sub(1).unwrap_or(len - 1);
                }
                CommandPopupAction::Consumed
            }
            KeyCode::Down => {
                let len = self.matches().len();
                if len > 0 {
                    self.selected = (self.selected + 1) % len;
                }
                CommandPopupAction::Consumed
            }
            KeyCode::Esc => CommandPopupAction::Cancel,
            KeyCode::Tab => self
                .selected()
                .map(CommandPopupAction::Complete)
                .unwrap_or(CommandPopupAction::Consumed),
            KeyCode::Enter => self
                .selected()
                .map(|command| {
                    if command.requires_argument() {
                        CommandPopupAction::Complete(command)
                    } else {
                        CommandPopupAction::Execute(command)
                    }
                })
                .unwrap_or(CommandPopupAction::Consumed),
            _ => CommandPopupAction::Pass,
        }
    }

    fn matches(&self) -> Vec<SlashCommand> {
        SlashCommand::ALL
            .into_iter()
            .filter(|command| command.command().starts_with(&self.filter))
            .collect()
    }
}

pub(crate) fn render(
    frame: &mut Frame<'_>,
    composer_area: Rect,
    popup: &CommandPopup,
    locale: Locale,
) {
    let commands = popup.commands();
    if commands.is_empty() || composer_area.y == 0 || composer_area.width == 0 {
        return;
    }
    let height = u16::try_from(commands.len())
        .unwrap_or(u16::MAX)
        .min(composer_area.y);
    let area = Rect::new(
        composer_area.x,
        composer_area.y.saturating_sub(height),
        composer_area.width,
        height,
    );
    let lines = commands
        .into_iter()
        .enumerate()
        .map(|(index, command)| {
            let selected = index == popup.selected;
            let marker = if selected { "> " } else { "  " };
            let command_style = if selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().add_modifier(Modifier::BOLD)
            };
            truncate_line_with_ellipsis_if_overflow(
                Line::from(vec![
                    Span::styled(marker, command_style),
                    Span::styled(format!("/{}", command.command()), command_style),
                    Span::raw("  "),
                    Span::styled(
                        command.description(locale),
                        Style::default().add_modifier(Modifier::DIM),
                    ),
                ]),
                usize::from(area.width),
            )
        })
        .collect::<Vec<_>>();
    frame.render_widget(Clear, area);
    frame.render_widget(Paragraph::new(lines), area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEvent, KeyModifiers};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    #[test]
    fn filters_by_prefix_and_wraps_selection() {
        let mut popup = CommandPopup::for_composer("/").expect("popup");
        assert_eq!(popup.selected(), Some(SlashCommand::Model));
        assert_eq!(
            popup.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE,))),
            CommandPopupAction::Consumed
        );
        assert_eq!(popup.selected(), Some(SlashCommand::Copy));
        assert!(popup.update("/p"));
        assert_eq!(popup.commands(), vec![SlashCommand::Permissions]);
        assert_eq!(popup.selected(), Some(SlashCommand::Permissions));
        assert!(!popup.update("/unknown"));
    }

    #[test]
    fn enter_executes_immediate_commands_and_completes_argument_commands() {
        let event = Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let mut model = CommandPopup::for_composer("/m").expect("model popup");
        assert_eq!(
            model.handle_event(&event),
            CommandPopupAction::Execute(SlashCommand::Model)
        );
        let mut effort = CommandPopup::for_composer("/e").expect("effort popup");
        assert_eq!(
            effort.handle_event(&event),
            CommandPopupAction::Complete(SlashCommand::Effort)
        );
    }

    #[test]
    fn test_backend_renders_commands_and_localized_descriptions() {
        let popup = CommandPopup::for_composer("/").expect("popup");
        let mut terminal = Terminal::new(TestBackend::new(72, 8)).expect("terminal");
        terminal
            .draw(|frame| render(frame, Rect::new(0, 5, 72, 3), &popup, Locale::ZhCn))
            .expect("draw");
        let buffer = terminal.backend().buffer();
        let text = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        let compact = text
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();

        assert!(text.contains("/model"));
        assert!(compact.contains("选择模型"), "{text}");
        assert!(text.contains("/copy"));
    }
}
