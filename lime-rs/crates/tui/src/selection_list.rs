//! Shared list-row formatting used by Codex-shaped TUI selectors.
//!
//! The production TUI currently has several list surfaces backed by ratatui's
//! `ListItem`. Keeping the numbering and selected/dim styles here prevents
//! those surfaces from drifting while leaving layout ownership with each
//! caller.

use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use unicode_width::UnicodeWidthStr;

pub(crate) fn selection_option_row(
    index: usize,
    label: impl Into<String>,
    is_selected: bool,
) -> Line<'static> {
    selection_option_row_with_dim(index, label, is_selected, false)
}

pub(crate) fn selection_option_row_with_dim(
    index: usize,
    label: impl Into<String>,
    is_selected: bool,
    dim: bool,
) -> Line<'static> {
    let prefix = if is_selected {
        format!("› {}. ", index.saturating_add(1))
    } else {
        format!("  {}. ", index.saturating_add(1))
    };
    let style = if is_selected {
        Style::default().cyan().add_modifier(Modifier::BOLD)
    } else if dim {
        Style::default().dim()
    } else {
        Style::default()
    };
    let prefix_width = UnicodeWidthStr::width(prefix.as_str());
    let label = label.into();
    let line = Line::from(vec![
        ratatui::text::Span::styled(prefix, style),
        ratatui::text::Span::styled(label, style),
    ]);
    debug_assert_eq!(
        prefix_width,
        UnicodeWidthStr::width(line.spans[0].content.as_ref())
    );
    line.style(style)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_rows_use_codex_marker_and_bold_cyan_style() {
        let row = selection_option_row(0, "Model", true);
        let line = &row;

        assert_eq!(line.spans[0].content, "› 1. ");
        assert_eq!(line.spans[1].content, "Model");
        assert_eq!(line.spans[0].style.fg, Some(ratatui::style::Color::Cyan));
        assert!(line.spans[0].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn unselected_dim_rows_keep_numbering_and_dim_style() {
        let row = selection_option_row_with_dim(2, "Provider", false, true);
        let line = &row;

        assert_eq!(line.spans[0].content, "  3. ");
        assert!(line.spans[1].style.add_modifier.contains(Modifier::DIM));
    }

    #[test]
    fn unicode_labels_remain_single_stable_row() {
        let row = selection_option_row(4, "模型", false);
        assert_eq!(row.spans.len(), 2);
        assert_eq!(UnicodeWidthStr::width(row.spans[1].content.as_ref()), 4);
    }
}
