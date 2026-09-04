use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::projection::{EntryKind, EntryStatus, TranscriptEntry};

pub(crate) fn lines(entry: &TranscriptEntry) -> Vec<Line<'static>> {
    let (prefix, prefix_style, text_style) = styles(entry.kind);
    let mut source = entry.text.lines();
    let first = source.next().unwrap_or("");
    let first = with_status_suffix(first, entry.status);
    let mut lines = vec![format_line(
        entry.kind,
        prefix,
        prefix_style,
        text_style,
        &first,
    )];
    lines.extend(source.map(|line| {
        format_line(
            entry.kind,
            "  ",
            prefix_style,
            continuation_style(entry.kind, text_style),
            line,
        )
    }));
    lines.extend(entry.summary.iter().map(|detail| {
        let detail = format!("- {detail}");
        format_line(
            entry.kind,
            "  ",
            prefix_style,
            continuation_style(entry.kind, text_style),
            &detail,
        )
    }));
    lines
}

fn with_status_suffix(text: &str, status: Option<EntryStatus>) -> String {
    status
        .map(|status| format!("{text} [{}]", status.label()))
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
        assert_eq!(patch[1].spans[1].style.fg, Some(Color::Green));
        assert_eq!(command[1].spans[1].style.fg, Some(Color::DarkGray));
    }

    #[test]
    fn plan_statuses_use_stable_checkbox_styles() {
        let completed = lines(&entry(EntryKind::Plan, "[x] inspect"));
        let running = lines(&entry(EntryKind::Plan, "[~] test"));
        let pending = lines(&entry(EntryKind::Plan, "[ ] ship"));

        assert!(completed[0].spans[1]
            .style
            .add_modifier
            .contains(Modifier::CROSSED_OUT));
        assert_eq!(running[0].spans[1].style.fg, Some(Color::Cyan));
        assert_eq!(pending[0].spans[1].style.fg, Some(Color::DarkGray));
    }
}
