use ratatui::style::Style;

use crate::markdown;
use crate::terminal_hyperlinks::HyperlinkLine;

/// Codex-shaped markdown rendering entry point backed by Lime's canonical renderer.
pub(crate) fn render_markdown_lines_with_width(
    input: &str,
    base_style: Style,
    width: Option<usize>,
) -> Vec<HyperlinkLine> {
    markdown::render(input, base_style, width)
}

#[allow(dead_code)]
pub(crate) fn render_markdown_text(input: &str, base_style: Style) -> Vec<HyperlinkLine> {
    render_markdown_lines_with_width(input, base_style, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_markdown_text_keeps_plain_content() {
        let lines = render_markdown_text("hello", Style::default());
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].line.spans[0].content, "hello");
    }
}
