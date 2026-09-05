use std::collections::VecDeque;
use std::path::Path;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::diff_render;
use crate::locale::Locale;
use crate::markdown_render;
use crate::projection::{EntryKind, EntryStatus, TranscriptEntry};
use crate::terminal_hyperlinks::{HyperlinkLine, prefix_hyperlink_lines};

const COMMAND_OUTPUT_HEAD_LINES: usize = 50;
const COMMAND_OUTPUT_TAIL_LINES: usize = 50;
const COMMAND_OUTPUT_MAX_LINE_BYTES: usize = 16 * 1024;

#[cfg(test)]
pub(crate) fn lines(entry: &TranscriptEntry) -> Vec<Line<'static>> {
    hyperlink_lines_with_locale(entry, Locale::default(), None, Path::new(""))
        .into_iter()
        .map(|line| line.line)
        .collect()
}

pub(crate) fn hyperlink_lines_with_locale(
    entry: &TranscriptEntry,
    locale: Locale,
    width: Option<usize>,
    cwd: &Path,
) -> Vec<HyperlinkLine> {
    let (prefix, prefix_style, text_style) = styles(entry.kind);
    let rich_lines = match entry.kind {
        EntryKind::Assistant | EntryKind::Reasoning => Some(
            markdown_render::render_markdown_lines_with_width(&entry.text, text_style, width),
        ),
        EntryKind::Patch => Some(
            diff_render::render(&entry.text, width, cwd)
                .into_iter()
                .map(HyperlinkLine::new)
                .collect(),
        ),
        _ => None,
    };
    if let Some(mut rendered) = rich_lines {
        if rendered.is_empty() {
            rendered.push(HyperlinkLine::default());
        }
        let status_suffix = entry
            .status
            .map(|status| format!(" [{}]", locale.status(status.label())));
        rendered = prefix_hyperlink_lines(
            rendered,
            Span::styled(prefix, prefix_style),
            Span::styled("  ", prefix_style),
        );
        if let Some(suffix) = status_suffix {
            if let Some(line) = rendered.first_mut() {
                line.push_span(Span::styled(suffix, text_style), None);
            }
        }
        rendered.extend(entry.summary.iter().map(|detail| {
            let detail = format!("- {}", locale.detail(detail));
            HyperlinkLine::new(format_line(
                entry.kind,
                "  ",
                prefix_style,
                continuation_style(entry.kind, text_style),
                &detail,
            ))
        }));
        return rendered;
    }
    let mut source = entry.text.lines();
    let first = source.next().unwrap_or("");
    let first = if matches!(entry.kind, EntryKind::Tool | EntryKind::System) {
        locale.detail(first)
    } else {
        first.to_string()
    };
    let first = with_status_suffix(&first, entry.status, locale);
    let mut lines = vec![HyperlinkLine::new(format_line(
        entry.kind,
        prefix,
        prefix_style,
        text_style,
        &first,
    ))];
    if entry.kind == EntryKind::Command {
        let output_lines = bounded_command_output_lines(source, locale);
        lines.extend(output_lines.iter().map(|line| {
            HyperlinkLine::new(format_line(
                entry.kind,
                "  ",
                prefix_style,
                continuation_style(entry.kind, text_style),
                line,
            ))
        }));
    } else {
        lines.extend(source.map(|line| {
            HyperlinkLine::new(format_line(
                entry.kind,
                "  ",
                prefix_style,
                continuation_style(entry.kind, text_style),
                line,
            ))
        }));
    }
    lines.extend(entry.summary.iter().map(|detail| {
        let detail = format!("- {}", locale.detail(detail));
        HyperlinkLine::new(format_line(
            entry.kind,
            "  ",
            prefix_style,
            continuation_style(entry.kind, text_style),
            &detail,
        ))
    }));
    lines
}

fn bounded_command_output_lines<'a>(
    source: impl Iterator<Item = &'a str>,
    locale: Locale,
) -> Vec<String> {
    let mut head = Vec::with_capacity(COMMAND_OUTPUT_HEAD_LINES);
    let mut tail = VecDeque::with_capacity(COMMAND_OUTPUT_TAIL_LINES);
    let mut total = 0usize;

    for line in source {
        total = total.saturating_add(1);
        let line = truncate_command_output_line(line, locale);
        if head.len() < COMMAND_OUTPUT_HEAD_LINES {
            head.push(line);
        } else {
            if tail.len() == COMMAND_OUTPUT_TAIL_LINES {
                tail.pop_front();
            }
            tail.push_back(line);
        }
    }

    let omitted = total.saturating_sub(head.len().saturating_add(tail.len()));
    if omitted > 0 {
        head.push(locale.output_omitted_lines(omitted));
    }
    head.extend(tail);
    head
}

fn truncate_command_output_line(line: &str, locale: Locale) -> String {
    if line.len() <= COMMAND_OUTPUT_MAX_LINE_BYTES {
        return line.to_string();
    }

    // Budget the localized omission marker as part of the line cap. Using the
    // full input length for the estimate keeps the marker length conservative
    // even when the actual UTF-8 boundary adjustment omits a few extra bytes.
    let marker_budget = locale.output_omitted_bytes(line.len());
    let retained_budget = COMMAND_OUTPUT_MAX_LINE_BYTES.saturating_sub(marker_budget.len());
    let head_budget = retained_budget / 2;
    let tail_budget = retained_budget.saturating_sub(head_budget);
    let mut head_end = head_budget.min(line.len());
    while !line.is_char_boundary(head_end) {
        head_end = head_end.saturating_sub(1);
    }
    let mut tail_start = line.len().saturating_sub(tail_budget);
    while !line.is_char_boundary(tail_start) {
        tail_start = tail_start.saturating_add(1);
    }
    let omitted = tail_start.saturating_sub(head_end);

    format!(
        "{}{}{}",
        &line[..head_end],
        locale.output_omitted_bytes(omitted),
        &line[tail_start..]
    )
}

fn with_status_suffix(text: &str, status: Option<EntryStatus>, locale: Locale) -> String {
    status
        .map(|status| format!("{text} [{}]", locale.status(status.label())))
        .unwrap_or_else(|| text.to_string())
}

fn styles(kind: EntryKind) -> (&'static str, Style, Style) {
    match kind {
        EntryKind::User => (
            "> ",
            Style::default().fg(Color::Cyan),
            Style::default().fg(Color::Cyan),
        ),
        EntryKind::Assistant => ("  ", Style::default(), Style::default()),
        EntryKind::Reasoning => (
            "· ",
            Style::default().fg(Color::DarkGray),
            Style::default().fg(Color::DarkGray),
        ),
        EntryKind::Command => (
            "$ ",
            Style::default().fg(Color::Yellow),
            Style::default().fg(Color::Yellow),
        ),
        EntryKind::Patch => ("Δ ", Style::default().fg(Color::Blue), Style::default()),
        EntryKind::Mcp => ("@ ", Style::default().fg(Color::Magenta), Style::default()),
        EntryKind::Plan => (
            "• ",
            Style::default().fg(Color::Cyan),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        EntryKind::MultiAgent => (
            "& ",
            Style::default().fg(Color::LightCyan),
            Style::default(),
        ),
        EntryKind::Tool => ("• ", Style::default().fg(Color::Yellow), Style::default()),
        EntryKind::System => (
            "! ",
            Style::default().fg(Color::Red),
            Style::default().fg(Color::Red),
        ),
    }
}

fn continuation_style(kind: EntryKind, base_style: Style) -> Style {
    if kind == EntryKind::Command {
        Style::default().fg(Color::DarkGray)
    } else {
        base_style
    }
}

fn format_line(
    kind: EntryKind,
    prefix: &'static str,
    prefix_style: Style,
    base_style: Style,
    text: &str,
) -> Line<'static> {
    let text_style = match kind {
        EntryKind::Patch if text.starts_with('+') => Style::default().fg(Color::Green),
        EntryKind::Patch if text.starts_with('-') => Style::default().fg(Color::Red),
        EntryKind::Patch if text.starts_with("@@") => Style::default().fg(Color::Cyan),
        EntryKind::Plan if text.starts_with("[x]") => Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::CROSSED_OUT),
        EntryKind::Plan if text.starts_with("[~]") => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
        EntryKind::Plan if text.starts_with("[ ]") => Style::default().fg(Color::DarkGray),
        _ => base_style,
    };
    Line::from(vec![
        Span::styled(prefix, prefix_style),
        Span::styled(text.to_string(), text_style),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(kind: EntryKind, text: &str) -> TranscriptEntry {
        TranscriptEntry {
            id: "entry-1".to_string(),
            kind,
            text: text.to_string(),
            streaming: false,
            status: (kind == EntryKind::Command).then_some(EntryStatus::Running),
            summary: Vec::new(),
        }
    }

    #[test]
    fn item_kinds_have_distinct_terminal_layouts() {
        let command = lines(&entry(EntryKind::Command, "cargo test\nfinished"));
        let patch = lines(&entry(EntryKind::Patch, "updated src/lib.rs\n+new"));
        let mcp = lines(&entry(EntryKind::Mcp, "server.tool [Completed]"));
        let plan = lines(&entry(EntryKind::Plan, "[~] run tests"));
        let multi_agent = lines(&entry(EntryKind::MultiAgent, "SpawnAgent [InProgress]"));

        assert_eq!(command[0].spans[0].content.as_ref(), "$ ");
        assert!(command[0].spans[1].content.contains("[running]"));
        assert_eq!(patch[0].spans[0].content.as_ref(), "Δ ");
        assert_eq!(mcp[0].spans[0].content.as_ref(), "@ ");
        assert_eq!(plan[0].spans[0].content.as_ref(), "• ");
        assert_eq!(multi_agent[0].spans[0].content.as_ref(), "& ");
        assert!(
            patch[1]
                .spans
                .iter()
                .any(|span| span.style.fg == Some(Color::Green))
        );
        assert_eq!(command[1].spans[1].style.fg, Some(Color::DarkGray));
    }

    #[test]
    fn plan_statuses_use_stable_checkbox_styles() {
        let completed = lines(&entry(EntryKind::Plan, "[x] inspect"));
        let running = lines(&entry(EntryKind::Plan, "[~] test"));
        let pending = lines(&entry(EntryKind::Plan, "[ ] ship"));

        assert!(
            completed[0].spans[1]
                .style
                .add_modifier
                .contains(Modifier::CROSSED_OUT)
        );
        assert_eq!(running[0].spans[1].style.fg, Some(Color::Cyan));
        assert_eq!(pending[0].spans[1].style.fg, Some(Color::DarkGray));
    }

    #[test]
    fn assistant_markdown_is_rendered_at_the_transcript_boundary() {
        let rendered = lines(&entry(
            EntryKind::Assistant,
            "# Result\n\n**completed** with `cargo test`",
        ));
        let text = rendered
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert!(text.contains("# Result"));
        assert!(text.contains("completed"));
        assert!(rendered.iter().any(|line| {
            line.spans
                .iter()
                .any(|span| span.style.add_modifier.contains(Modifier::BOLD))
        }));
        assert!(rendered.iter().any(|line| {
            line.spans
                .iter()
                .any(|span| span.style.fg == Some(Color::Cyan))
        }));
    }

    #[test]
    fn reasoning_markdown_keeps_reasoning_tone_and_inline_emphasis() {
        let rendered = lines(&entry(EntryKind::Reasoning, "**Inspecting** `src/lib.rs`"));

        assert!(rendered.iter().flat_map(|line| &line.spans).any(|span| {
            span.style.fg == Some(Color::DarkGray)
                && span.style.add_modifier.contains(Modifier::BOLD)
        }));
        assert!(
            rendered
                .iter()
                .flat_map(|line| &line.spans)
                .any(|span| span.content == "src/lib.rs" && span.style.fg == Some(Color::Cyan))
        );
    }

    #[test]
    fn command_output_keeps_head_and_tail_with_omitted_line_marker() {
        let output = (0..101)
            .map(|index| format!("line-{index}"))
            .collect::<Vec<_>>()
            .join("\n");
        let rendered = lines(&entry(EntryKind::Command, &format!("printf\n{output}")));
        let text = rendered
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(text.contains("line-0"));
        assert!(text.contains("line-49"));
        assert!(text.contains("… 1 lines omitted …"));
        assert!(!text.contains("line-50"));
        assert!(text.contains("line-51"));
        assert!(text.contains("line-100"));
    }

    #[test]
    fn command_output_long_line_preserves_utf8_head_and_tail() {
        let line = format!("{}尾", "界".repeat(COMMAND_OUTPUT_MAX_LINE_BYTES));
        let rendered = lines(&entry(EntryKind::Command, &format!("printf\n{line}")));
        let text = rendered
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(text.contains("bytes omitted"));
        assert!(text.ends_with("尾"));
    }

    #[test]
    fn command_output_long_line_includes_marker_in_the_byte_budget() {
        let line = "界".repeat(COMMAND_OUTPUT_MAX_LINE_BYTES);
        let truncated = truncate_command_output_line(&line, Locale::EnUs);

        assert!(truncated.len() <= COMMAND_OUTPUT_MAX_LINE_BYTES);
        assert!(truncated.contains("bytes omitted"));
    }
}
