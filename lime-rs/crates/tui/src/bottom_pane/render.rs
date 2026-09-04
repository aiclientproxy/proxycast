use ratatui::layout::{Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use super::approval::ApprovalRequest;
use super::{BottomPane, PendingInteraction};
use crate::width::display_width;

pub(crate) fn desired_height(pane: &BottomPane) -> u16 {
    let lines = lines(pane).len().saturating_add(2);
    u16::try_from(lines).unwrap_or(u16::MAX).clamp(5, 14)
}

pub(crate) fn render(frame: &mut Frame<'_>, area: Rect, pane: &BottomPane) {
    let block = Block::default()
        .borders(Borders::TOP | Borders::BOTTOM)
        .border_style(Style::default().fg(Color::Yellow));
    let inner = block.inner(area);
    let content = lines(pane);
    frame.render_widget(
        Paragraph::new(content.clone())
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );

    let Some(PendingInteraction::UserInput(request)) = pane.current() else {
        return;
    };
    if !request.editing || inner.width == 0 || inner.height == 0 {
        return;
    }
    let row = content.len().saturating_sub(1);
    let value = &request.composer.text()[..request.composer.cursor()];
    let value_width = request
        .params
        .questions
        .get(request.question_index)
        .filter(|question| question.is_secret)
        .map_or_else(|| display_width(value), |_| value.chars().count());
    let x = u16::try_from(value_width)
        .unwrap_or(u16::MAX)
        .saturating_add(2)
        .min(inner.width.saturating_sub(1));
    let y = u16::try_from(row)
        .unwrap_or(u16::MAX)
        .min(inner.height.saturating_sub(1));
    frame.set_cursor_position(Position::new(
        inner.x.saturating_add(x),
        inner.y.saturating_add(y),
    ));
}

fn lines(pane: &BottomPane) -> Vec<Line<'static>> {
    match pane.current() {
        Some(PendingInteraction::Approval(approval)) => {
            let (title, details) = match &approval.request {
                ApprovalRequest::Command { params, .. } => (
                    "Approve command?",
                    vec![
                        params.command.clone().unwrap_or_default(),
                        params.cwd.clone().unwrap_or_default(),
                        params.reason.clone().unwrap_or_default(),
                    ],
                ),
                ApprovalRequest::FileChange { params, .. } => (
                    "Approve file changes?",
                    vec![
                        params.grant_root.clone().unwrap_or_default(),
                        params.reason.clone().unwrap_or_default(),
                    ],
                ),
                ApprovalRequest::Permissions { params, .. } => (
                    "Grant additional permissions?",
                    vec![
                        params.cwd.clone(),
                        params.reason.clone().unwrap_or_default(),
                        serde_json::to_string(&params.permissions).unwrap_or_default(),
                    ],
                ),
            };
            let mut lines = vec![Line::styled(
                title,
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
                    .map(|(index, label)| option_line(index == approval.selected, label)),
            );
            lines
        }
        Some(PendingInteraction::UserInput(request)) => {
            let Some(question) = request.params.questions.get(request.question_index) else {
                return vec![Line::from("No questions")];
            };
            let mut lines = vec![
                Line::styled(
                    format!(
                        "{} ({}/{})",
                        question.header,
                        request.question_index + 1,
                        request.params.questions.len()
                    ),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Line::from(question.question.clone()),
            ];
            if let Some(options) = question
                .options
                .as_ref()
                .filter(|options| !options.is_empty())
            {
                lines.extend(options.iter().enumerate().map(|(index, option)| {
                    option_line(
                        !request.editing && index == request.selected,
                        format!("{}  {}", option.label, option.description),
                    )
                }));
                if question.is_other {
                    lines.push(option_line(
                        !request.editing && request.selected == options.len(),
                        "Other  Type a custom answer".to_string(),
                    ));
                }
            }
            if request.editing {
                let value = if question.is_secret {
                    "*".repeat(request.composer.text().chars().count())
                } else {
                    request.composer.text().to_string()
                };
                lines.push(Line::from(vec![
                    Span::styled("> ", Style::default().fg(Color::Cyan)),
                    Span::raw(value),
                ]));
            } else if question.options.is_some() {
                lines.push(Line::styled(
                    "Tab to add notes",
                    Style::default().fg(Color::DarkGray),
                ));
            }
            lines
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
