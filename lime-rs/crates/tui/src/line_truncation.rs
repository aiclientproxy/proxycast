//! Styled line truncation adapted from Codex TUI.

use ratatui::text::{Line, Span};
use unicode_segmentation::UnicodeSegmentation;

use crate::width::display_width;

pub(crate) fn line_width(line: &Line<'_>) -> usize {
    line.iter()
        .map(|span| display_width(span.content.as_ref()))
        .sum()
}

pub(crate) fn truncate_line_to_width(line: Line<'static>, max_width: usize) -> Line<'static> {
    if max_width == 0 {
        return Line::from(Vec::<Span<'static>>::new());
    }

    let Line {
        style,
        alignment,
        spans,
    } = line;
    let mut used = 0usize;
    let mut spans_out = Vec::with_capacity(spans.len());

    for span in spans {
        let span_width = display_width(span.content.as_ref());
        if span_width == 0 {
            spans_out.push(span);
            continue;
        }
        if used >= max_width {
            break;
        }
        if used + span_width <= max_width {
            used += span_width;
            spans_out.push(span);
            continue;
        }

        let style = span.style;
        let text = span.content.as_ref();
        let mut end_idx = 0usize;
        for (idx, grapheme) in text.grapheme_indices(true) {
            let grapheme_width = display_width(grapheme);
            if used + grapheme_width > max_width {
                break;
            }
            end_idx = idx + grapheme.len();
            used += grapheme_width;
        }
        if end_idx > 0 {
            spans_out.push(Span::styled(text[..end_idx].to_string(), style));
        }
        break;
    }

    Line {
        style,
        alignment,
        spans: spans_out,
    }
}

pub(crate) fn truncate_line_with_ellipsis_if_overflow(
    line: Line<'static>,
    max_width: usize,
) -> Line<'static> {
    if max_width == 0 {
        return Line::from(Vec::<Span<'static>>::new());
    }
    if line_width(&line) <= max_width {
        return line;
    }

    let truncated = truncate_line_to_width(line, max_width.saturating_sub(1));
    let Line {
        style,
        alignment,
        mut spans,
    } = truncated;
    let ellipsis_style = spans.last().map(|span| span.style).unwrap_or_default();
    spans.push(Span::styled("…", ellipsis_style));
    Line {
        style,
        alignment,
        spans,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halfwidth_sound_marks_stay_with_their_grapheme_when_truncated() {
        let line = Line::from("abｶﾞc");

        assert_eq!(line_width(&line), 5);
        assert_eq!(truncate_line_to_width(line.clone(), 3), Line::from("ab"));
        assert_eq!(truncate_line_to_width(line, 4), Line::from("abｶﾞ"));
    }

    #[test]
    fn ellipsis_inherits_the_last_visible_span_style() {
        let line = Line::from(vec![
            Span::raw("ab"),
            Span::styled(
                "界cd",
                ratatui::style::Style::default().fg(ratatui::style::Color::Cyan),
            ),
        ]);
        let truncated = truncate_line_with_ellipsis_if_overflow(line, 5);

        assert_eq!(line_width(&truncated), 5);
        assert_eq!(truncated.spans[1].content, "界");
        assert_eq!(truncated.spans[2].content, "…");
        assert_eq!(truncated.spans[1].style, truncated.spans[2].style);
    }
}
