use ratatui::Frame;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_segmentation::UnicodeSegmentation;

use crate::line_truncation::line_width;
use crate::locale::Locale;
use crate::width::display_width;

use super::RequestUserInputOverlay;

pub(in crate::bottom_pane) fn lines_with_locale(
    request: &RequestUserInputOverlay,
    locale: Locale,
) -> Vec<Line<'static>> {
    lines_with_locale_unbounded(request, locale)
}

pub(in crate::bottom_pane) fn lines_with_locale_with_width(
    request: &RequestUserInputOverlay,
    locale: Locale,
    width: usize,
) -> Vec<Line<'static>> {
    let lines = if width == usize::MAX {
        lines_with_locale(request, locale)
    } else {
        lines_with_locale_unbounded(request, locale)
    };
    if width == usize::MAX {
        lines
    } else {
        lines
            .into_iter()
            .map(|line| truncate_line_word_boundary_with_ellipsis(line, width))
            .collect()
    }
}

fn lines_with_locale_unbounded(
    request: &RequestUserInputOverlay,
    locale: Locale,
) -> Vec<Line<'static>> {
    let Some(question) = request.params.questions.get(request.question_index) else {
        return vec![Line::from(locale.no_questions())];
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
                locale.other_option().to_string(),
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
            Span::styled("> ", Style::default().fg(ratatui::style::Color::Cyan)),
            Span::raw(value),
        ]));
    } else if question.options.is_some() {
        lines.push(Line::styled(
            locale.add_notes(),
            Style::default().fg(ratatui::style::Color::DarkGray),
        ));
    }
    lines
}

pub(in crate::bottom_pane) fn set_cursor_position(
    frame: &mut Frame<'_>,
    inner: Rect,
    request: &RequestUserInputOverlay,
    content_len: usize,
) {
    if !request.editing || inner.width == 0 || inner.height == 0 {
        return;
    }
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
    let y = u16::try_from(content_len.saturating_sub(1))
        .unwrap_or(u16::MAX)
        .min(inner.height.saturating_sub(1));
    frame.set_cursor_position(Position::new(
        inner.x.saturating_add(x),
        inner.y.saturating_add(y),
    ));
}

fn option_line(selected: bool, label: String) -> Line<'static> {
    let prefix = if selected { "> " } else { "  " };
    let style = if selected {
        Style::default().fg(ratatui::style::Color::Cyan)
    } else {
        Style::default()
    };
    Line::styled(format!("{prefix}{label}"), style)
}

/// Truncate a styled line at a grapheme-safe word boundary and append an ellipsis.
///
/// The available width reserves one cell for the ellipsis. Whitespace is preferred as the
/// break point, while a grapheme boundary is used when no word boundary fits.
pub(super) fn truncate_line_word_boundary_with_ellipsis(
    line: Line<'static>,
    max_width: usize,
) -> Line<'static> {
    if max_width == 0 {
        return Line::from(Vec::<Span<'static>>::new());
    }

    if line_width(&line) <= max_width {
        return line;
    }

    let ellipsis = "…";
    let ellipsis_width = display_width(ellipsis);
    if ellipsis_width >= max_width {
        return Line::from(ellipsis);
    }
    let limit = max_width.saturating_sub(ellipsis_width);

    #[derive(Clone, Copy)]
    struct BreakPoint {
        span_idx: usize,
        byte_end: usize,
    }

    let mut used = 0usize;
    let mut last_fit = None;
    let mut last_word_break = None;
    let mut overflowed = false;

    'outer: for (span_idx, span) in line.spans.iter().enumerate() {
        for (byte_idx, grapheme) in span.content.as_ref().grapheme_indices(true) {
            let grapheme_width = display_width(grapheme);
            if used.saturating_add(grapheme_width) > limit {
                overflowed = true;
                break 'outer;
            }
            used = used.saturating_add(grapheme_width);
            let break_point = BreakPoint {
                span_idx,
                byte_end: byte_idx + grapheme.len(),
            };
            last_fit = Some(break_point);
            if grapheme.chars().all(char::is_whitespace) {
                last_word_break = Some(break_point);
            }
        }
    }

    if !overflowed {
        return line;
    }

    let Some(chosen_break) = last_word_break.or(last_fit) else {
        return Line::from(ellipsis);
    };

    let line_style = line.style;
    let mut spans_out = Vec::new();
    for (idx, span) in line.spans.into_iter().enumerate() {
        if idx < chosen_break.span_idx {
            spans_out.push(span);
            continue;
        }
        if idx == chosen_break.span_idx {
            let text = span.content.into_owned();
            let truncated = text[..chosen_break.byte_end].to_string();
            if !truncated.is_empty() {
                spans_out.push(Span::styled(truncated, span.style));
            }
        }
        break;
    }

    while let Some(last) = spans_out.last_mut() {
        let trimmed = last
            .content
            .trim_end_matches(char::is_whitespace)
            .to_string();
        if trimmed.is_empty() {
            spans_out.pop();
        } else {
            last.content = trimmed.into();
            break;
        }
    }

    let ellipsis_style = spans_out
        .last()
        .map(|span| span.style)
        .unwrap_or(line_style);
    spans_out.push(Span::styled(ellipsis, ellipsis_style));
    Line::from(spans_out).style(line_style)
}

#[cfg(test)]
#[path = "render_tests.rs"]
mod tests;
