use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use super::approval_overlay::ApprovalRequest;
use super::request_user_input::render as request_user_input_render;
use super::{BottomPane, PendingInteraction};
use crate::locale::Locale;

pub(crate) fn desired_height_with_locale(pane: &BottomPane, locale: Locale) -> u16 {
    let lines = lines_with_locale(pane, locale, usize::MAX)
        .len()
        .saturating_add(2);
    u16::try_from(lines).unwrap_or(u16::MAX).clamp(5, 14)
}

pub(crate) fn render_with_locale(
    frame: &mut Frame<'_>,
    area: Rect,
    pane: &BottomPane,
    locale: Locale,
) {
    let block = Block::default()
        .borders(Borders::TOP | Borders::BOTTOM)
        .border_style(Style::default().fg(Color::Yellow));
    let inner = block.inner(area);
    let content = lines_with_locale(pane, locale, inner.width as usize);
    frame.render_widget(
        Paragraph::new(content.clone())
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );

    let Some(PendingInteraction::UserInput(request)) = pane.current() else {
        return;
    };
    request_user_input_render::set_cursor_position(frame, inner, request, content.len());
}

fn lines_with_locale(pane: &BottomPane, locale: Locale, width: usize) -> Vec<Line<'static>> {
    match pane.current() {
        Some(PendingInteraction::Approval(approval)) => {
            let (kind, details) = match &approval.request {
                ApprovalRequest::Exec { params, .. } => (
                    "command",
                    vec![
                        params.command.clone().unwrap_or_default(),
                        params.cwd.clone().unwrap_or_default(),
                        params.reason.clone().unwrap_or_default(),
                    ],
                ),
                ApprovalRequest::ApplyPatch { params, .. } => (
                    "file",
                    vec![
                        params.grant_root.clone().unwrap_or_default(),
                        params.reason.clone().unwrap_or_default(),
                    ],
                ),
                ApprovalRequest::Permissions { params, .. } => (
                    "permissions",
                    vec![
                        params.cwd.clone(),
                        params.reason.clone().unwrap_or_default(),
                        serde_json::to_string(&params.permissions).unwrap_or_default(),
                    ],
                ),
            };
            let mut lines = vec![Line::styled(
                locale.approval_title(kind),
                Style::default().add_modifier(Modifier::BOLD),
            )];
            lines.extend(
                details
                    .into_iter()
                    .filter(|detail| !detail.is_empty())
                    .map(|detail| Line::styled(detail, Style::default().fg(Color::DarkGray))),
            );
            lines.extend(
                approval
                    .option_labels()
                    .into_iter()
                    .enumerate()
                    .map(|(index, label)| {
                        option_line(index == approval.selected, locale.approval_option(&label))
                    }),
            );
            lines
        }
        Some(PendingInteraction::UserInput(request)) => {
            request_user_input_render::lines_with_locale_with_width(request, locale, width)
        }
        None => Vec::new(),
    }
}

fn option_line(selected: bool, label: String) -> Line<'static> {
    let prefix = if selected { "> " } else { "  " };
    let style = if selected {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    };
    Line::styled(format!("{prefix}{label}"), style)
}
