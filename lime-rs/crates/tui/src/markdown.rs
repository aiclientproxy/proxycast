use pulldown_cmark::{Alignment, CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;

use crate::highlight::highlight_code_to_lines;
use crate::terminal_hyperlinks::HyperlinkLine;

mod local_links;
mod table;

const LIST_INDENT: &str = "    ";

#[derive(Clone, Copy, Debug)]
enum ListKind {
    Unordered,
    Ordered(u64),
}

struct LinkState {
    destination: String,
    image: bool,
    local_label: Option<Vec<Span<'static>>>,
}

#[derive(Clone, Default)]
struct TableRow {
    cells: Vec<HyperlinkLine>,
    header: bool,
}

struct TableState {
    alignments: Vec<Alignment>,
    rows: Vec<TableRow>,
    current_row: TableRow,
    current_cell: HyperlinkLine,
    in_header: bool,
}

impl TableState {
    fn new(alignments: Vec<Alignment>) -> Self {
        Self {
            alignments,
            rows: Vec::new(),
            current_row: TableRow::default(),
            current_cell: HyperlinkLine::default(),
            in_header: false,
        }
    }

    fn push_text(
        &mut self,
        text: &str,
        style: Style,
        destination: Option<&str>,
        detect_bare_links: bool,
    ) {
        let text = text.replace('\n', " ");
        if !text.is_empty() {
            let span = Span::styled(text, style);
            if destination.is_some() {
                self.current_cell.push_span(span, destination);
            } else if detect_bare_links {
                self.current_cell.push_auto_link_span(span);
            } else {
                self.current_cell.push_span(span, None);
            }
        }
    }

    fn finish_cell(&mut self) {
        self.current_row
            .cells
            .push(std::mem::take(&mut self.current_cell));
    }

    fn finish_row(&mut self) {
        self.current_row.header = self.in_header;
        self.rows.push(std::mem::take(&mut self.current_row));
    }
}

struct Renderer {
    lines: Vec<HyperlinkLine>,
    current: HyperlinkLine,
    base_style: Style,
    width: Option<usize>,
    styles: Vec<Style>,
    blockquote_depth: usize,
    lists: Vec<ListKind>,
    item_marker: Option<String>,
    heading_marker: Option<String>,
    code_block: bool,
    code_block_lang: Option<String>,
    code_block_buffer: String,
    link: Option<LinkState>,
    table: Option<TableState>,
}

impl Renderer {
    fn new(base_style: Style, width: Option<usize>) -> Self {
        Self {
            lines: Vec::new(),
            current: HyperlinkLine::default(),
            base_style,
            width,
            styles: vec![base_style],
            blockquote_depth: 0,
            lists: Vec::new(),
            item_marker: None,
            heading_marker: None,
            code_block: false,
            code_block_lang: None,
            code_block_buffer: String::new(),
            link: None,
            table: None,
        }
    }

    fn style(&self) -> Style {
        self.styles.last().copied().unwrap_or(self.base_style)
    }

    fn push_style(&mut self, style: Style) {
        self.styles.push(self.style().patch(style));
    }

    fn pop_style(&mut self) {
        if self.styles.len() > 1 {
            self.styles.pop();
        }
    }

    fn ensure_prefix(&mut self) {
        if !self.current.line.spans.is_empty() {
            return;
        }

        let mut prefix = "> ".repeat(self.blockquote_depth);
        if let Some(marker) = self.heading_marker.take() {
            prefix.push_str(&marker);
        } else if let Some(marker) = self.item_marker.take() {
            prefix.push_str(&LIST_INDENT.repeat(self.lists.len().saturating_sub(1)));
            prefix.push_str(&marker);
        } else if !self.lists.is_empty() {
            prefix.push_str(&LIST_INDENT.repeat(self.lists.len()));
        }
        if self.code_block {
            prefix.push_str("    ");
        }
        if !prefix.is_empty() {
            let style = if self.blockquote_depth > 0 {
                Style::default().fg(Color::Green)
            } else {
                Style::default()
            };
            self.current.push_span(Span::styled(prefix, style), None);
        }
    }

    fn push_text(&mut self, text: &str, style: Style) {
        self.push_text_with_link_detection(text, style, true);
    }

    fn push_text_with_link_detection(&mut self, text: &str, style: Style, detect_bare_links: bool) {
        if let Some(label) = self
            .link
            .as_mut()
            .and_then(|link| link.local_label.as_mut())
        {
            let text = text.replace('\n', " ");
            if !text.is_empty() {
                label.push(Span::styled(text, style));
            }
            return;
        }
        let destination = self
            .link
            .as_ref()
            .filter(|link| !link.image)
            .map(|link| link.destination.clone());
        let detect_bare_links = detect_bare_links && destination.is_none() && !self.code_block;
        if let Some(table) = self.table.as_mut() {
            table.push_text(text, style, destination.as_deref(), detect_bare_links);
            return;
        }
        for part in text.split_inclusive('\n') {
            let has_newline = part.ends_with('\n');
            let content = part.strip_suffix('\n').unwrap_or(part);
            if !content.is_empty() {
                self.ensure_prefix();
                let style = if self.blockquote_depth > 0 {
                    self.style().patch(Style::default().fg(Color::Green))
                } else {
                    style
                };
                let span = Span::styled(content.to_string(), style);
                if destination.is_some() {
                    self.current.push_span(span, destination.as_deref());
                } else if detect_bare_links {
                    self.current.push_auto_link_span(span);
                } else {
                    self.current.push_span(span, None);
                }
            }
            if has_newline {
                self.finish_line();
            }
        }
    }

    fn finish_line(&mut self) {
        self.lines.push(std::mem::take(&mut self.current));
    }

    fn blank_line(&mut self) {
        if !self
            .lines
            .last()
            .is_some_and(|line| line.line.spans.is_empty())
        {
            self.lines.push(HyperlinkLine::default());
        }
    }

    fn finish_block(&mut self) {
        if !self.current.line.spans.is_empty() {
            self.finish_line();
        }
        if self.blockquote_depth == 0 && self.lists.is_empty() {
            self.blank_line();
        }
    }

    fn render(mut self, input: &str) -> Vec<HyperlinkLine> {
        let mut options =
            Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES | Options::ENABLE_TASKLISTS;
        options.insert(Options::ENABLE_FOOTNOTES);

        for event in Parser::new_ext(input, options) {
            match event {
                Event::Start(tag) => self.start(tag),
                Event::End(tag) => self.end(tag),
                Event::Text(text) => {
                    if self.code_block && self.code_block_lang.is_some() {
                        self.code_block_buffer.push_str(&text);
                    } else {
                        self.push_text(&text, self.style());
                    }
                }
                Event::Code(code) => {
                    self.push_text_with_link_detection(&code, self.style().fg(Color::Cyan), false);
                }
                Event::Html(html) | Event::InlineHtml(html) => {
                    self.push_text(&html, self.style());
                }
                Event::SoftBreak | Event::HardBreak => {
                    let style = self.style();
                    if let Some(table) = self.table.as_mut() {
                        table.push_text(" ", style, None, false);
                    } else {
                        self.finish_line();
                    }
                }
                Event::Rule => {
                    self.push_text("────────", Style::default().fg(Color::DarkGray));
                    self.finish_line();
                }
                Event::TaskListMarker(checked) => {
                    let marker = if checked { "[x] " } else { "[ ] " };
                    self.push_text(marker, Style::default().fg(Color::DarkGray));
                }
                Event::FootnoteReference(label) => {
                    self.push_text(&format!("[^{label}]"), self.style());
                }
            }
        }

        if !self.current.line.spans.is_empty() {
            self.finish_line();
        }
        while self
            .lines
            .last()
            .is_some_and(|line| line.line.spans.is_empty())
        {
            self.lines.pop();
        }
        self.lines
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {}
            Tag::Heading { level, .. } => {
                self.heading_marker = Some(format!("{} ", "#".repeat(level as usize)));
                let style = match level {
                    HeadingLevel::H1 => {
                        Style::default().add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
                    }
                    HeadingLevel::H2 => Style::default().add_modifier(Modifier::BOLD),
                    HeadingLevel::H3 => {
                        Style::default().add_modifier(Modifier::BOLD | Modifier::ITALIC)
                    }
                    _ => Style::default().add_modifier(Modifier::ITALIC),
                };
                self.push_style(style);
            }
            Tag::BlockQuote => self.blockquote_depth += 1,
            Tag::CodeBlock(kind) => {
                self.code_block = true;
                self.push_style(Style::default().fg(Color::Cyan));
                self.code_block_lang = match kind {
                    CodeBlockKind::Fenced(info) => info
                        .split([',', ' ', '\t'])
                        .next()
                        .filter(|language| !language.is_empty())
                        .map(str::to_string),
                    CodeBlockKind::Indented => None,
                };
                self.code_block_buffer.clear();
            }
            Tag::List(start) => {
                if !self.lists.is_empty() {
                    self.finish_line_if_needed();
                }
                self.lists
                    .push(start.map_or(ListKind::Unordered, ListKind::Ordered));
            }
            Tag::Item => {
                self.item_marker = self.lists.last_mut().map(|list| match list {
                    ListKind::Unordered => "- ".to_string(),
                    ListKind::Ordered(next) => {
                        let marker = format!("{next}. ");
                        *next += 1;
                        marker
                    }
                });
            }
            Tag::Emphasis => self.push_style(Style::default().add_modifier(Modifier::ITALIC)),
            Tag::Strong => self.push_style(Style::default().add_modifier(Modifier::BOLD)),
            Tag::Strikethrough => {
                self.push_style(Style::default().add_modifier(Modifier::CROSSED_OUT))
            }
            Tag::Link { dest_url, .. } => {
                let destination = dest_url.into_string();
                self.link = Some(LinkState {
                    local_label: local_links::is_local_path_like_link(&destination).then(Vec::new),
                    destination,
                    image: false,
                });
                self.push_style(
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::UNDERLINED),
                );
            }
            Tag::Image { dest_url, .. } => {
                self.link = Some(LinkState {
                    destination: dest_url.into_string(),
                    image: true,
                    local_label: None,
                });
                self.push_style(Style::default().fg(Color::Cyan));
            }
            Tag::Table(alignments) => {
                self.finish_line_if_needed();
                self.table = Some(TableState::new(alignments));
            }
            Tag::TableHead => {
                if let Some(table) = self.table.as_mut() {
                    table.in_header = true;
                }
            }
            Tag::TableRow => {}
            Tag::TableCell => {}
            Tag::FootnoteDefinition(_) | Tag::MetadataBlock(_) | Tag::HtmlBlock => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => self.finish_block(),
            TagEnd::Heading(_) => {
                self.finish_block();
                self.pop_style();
            }
            TagEnd::BlockQuote => {
                self.finish_line_if_needed();
                self.blockquote_depth = self.blockquote_depth.saturating_sub(1);
                if self.blockquote_depth == 0 {
                    self.blank_line();
                }
            }
            TagEnd::CodeBlock => {
                self.finish_highlighted_code_block();
                self.finish_line_if_needed();
                self.code_block = false;
                self.pop_style();
                self.blank_line();
            }
            TagEnd::List(_) => {
                self.finish_line_if_needed();
                self.lists.pop();
                if self.lists.is_empty() {
                    self.blank_line();
                }
            }
            TagEnd::Item => self.finish_line_if_needed(),
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => {
                self.pop_style();
            }
            TagEnd::Link | TagEnd::Image => {
                self.pop_style();
                self.finish_link();
            }
            TagEnd::Table => self.finish_table(),
            TagEnd::TableHead => {
                if let Some(table) = self.table.as_mut() {
                    if !table.current_row.cells.is_empty() {
                        table.finish_row();
                    }
                    table.in_header = false;
                }
            }
            TagEnd::TableRow => {
                if let Some(table) = self.table.as_mut() {
                    table.finish_row();
                }
            }
            TagEnd::TableCell => {
                if let Some(table) = self.table.as_mut() {
                    table.finish_cell();
                }
            }
            TagEnd::FootnoteDefinition | TagEnd::MetadataBlock(_) | TagEnd::HtmlBlock => {}
        }
    }

    fn finish_line_if_needed(&mut self) {
        if !self.current.line.spans.is_empty() {
            self.finish_line();
        }
    }

    fn finish_highlighted_code_block(&mut self) {
        let Some(language) = self.code_block_lang.take() else {
            return;
        };
        let code = std::mem::take(&mut self.code_block_buffer);
        if code.is_empty() {
            return;
        }
        for line in highlight_code_to_lines(&code, &language) {
            self.ensure_prefix();
            for span in line.spans {
                let style = self.style().patch(span.style);
                self.current
                    .push_span(Span::styled(span.content.into_owned(), style), None);
            }
            self.finish_line();
        }
    }

    fn finish_link(&mut self) {
        let Some(link) = self.link.take() else {
            return;
        };
        if link.image || link.destination.is_empty() {
            return;
        }
        if let Some(label) = link.local_label {
            let Some(target) = local_links::render_local_link_target(&link.destination) else {
                return;
            };
            let label_text = label
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>();
            if local_links::should_render_local_link_label(&label_text, &link.destination) {
                for span in label {
                    self.push_text_with_link_detection(&span.content, span.style, false);
                }
                self.push_text_with_link_detection(" (", self.base_style, false);
                self.push_text_with_link_detection(
                    &target,
                    self.base_style
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::UNDERLINED),
                    false,
                );
                self.push_text_with_link_detection(")", self.base_style, false);
            } else {
                self.push_text_with_link_detection(
                    &target,
                    self.base_style
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::UNDERLINED),
                    false,
                );
            }
            return;
        }
        self.push_text_with_link_detection(" (", self.base_style, false);
        self.link = Some(LinkState {
            destination: link.destination.clone(),
            image: false,
            local_label: None,
        });
        self.push_text_with_link_detection(
            &link.destination,
            self.base_style
                .fg(Color::Cyan)
                .add_modifier(Modifier::UNDERLINED),
            false,
        );
        self.link = None;
        self.push_text_with_link_detection(")", self.base_style, false);
    }

    fn finish_table(&mut self) {
        let Some(table) = self.table.take() else {
            return;
        };
        self.lines.extend(table::render(table, self.width));
        self.blank_line();
    }
}

pub(crate) fn render(input: &str, base_style: Style, width: Option<usize>) -> Vec<HyperlinkLine> {
    Renderer::new(base_style, width).render(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(lines: &[HyperlinkLine]) -> Vec<String> {
        lines
            .iter()
            .map(|line| {
                line.line
                    .spans
                    .iter()
                    .map(|span| span.content.clone())
                    .collect()
            })
            .collect()
    }

    fn render_unconstrained(input: &str, style: Style) -> Vec<HyperlinkLine> {
        render(input, style, None)
    }

    #[test]
    fn renders_headings_emphasis_code_lists_quotes_and_links() {
        let lines = render_unconstrained(
            "# Title\n\n**strong** *emphasis* `inline` [link](https://example.com)\n\n```rust\nlet x = 1;\n```\n\n- one\n- two\n\n> quote",
            Style::default(),
        );
        let text = plain(&lines).join("\n");
        assert!(text.contains("# Title"));
        assert!(text.contains("strong emphasis inline link"));
        assert!(text.contains("let x = 1;"));
        assert!(text.contains("- one"));
        assert!(text.contains("> quote"));
        assert!(lines.iter().any(|line| {
            line.line
                .spans
                .iter()
                .any(|span| span.style.add_modifier.contains(Modifier::BOLD))
        }));
        assert!(lines.iter().any(|line| {
            line.line
                .spans
                .iter()
                .any(|span| span.style.fg == Some(Color::Cyan))
        }));
    }

    #[test]
    fn fenced_code_uses_the_first_info_token_for_syntax_highlighting() {
        let lines = render_unconstrained(
            "```rust,no_run title=demo\nfn main() { let answer = 42; }\n```",
            Style::default(),
        );
        let code = lines
            .iter()
            .find(|line| plain(std::slice::from_ref(line))[0].contains("fn main"))
            .expect("rendered Rust code line");

        assert!(code.line.spans.len() > 2);
        assert!(code.line.spans.iter().any(|span| {
            span.content.contains("fn")
                && (span.style.fg.is_some() || !span.style.add_modifier.is_empty())
        }));
    }

    #[test]
    fn unknown_fenced_language_preserves_plain_code() {
        let lines = render_unconstrained("```unknown-language\nalpha\nbeta\n```", Style::default());

        assert_eq!(plain(&lines), vec!["    alpha", "    beta"]);
    }

    #[test]
    fn preserves_multiline_and_empty_input() {
        assert!(render_unconstrained("", Style::default()).is_empty());
        let lines = render_unconstrained("first\nsecond", Style::default());
        assert_eq!(plain(&lines), vec!["first", "second"]);
    }

    #[test]
    fn keeps_link_destinations_visible_and_underlined() {
        let lines = render_unconstrained(
            "Read [the guide](https://example.com/guide).",
            Style::default(),
        );

        assert_eq!(
            plain(&lines),
            vec!["Read the guide (https://example.com/guide)."]
        );
        let destination = lines[0]
            .line
            .spans
            .iter()
            .find(|span| span.content == "https://example.com/guide")
            .expect("visible link destination");
        assert_eq!(destination.style.fg, Some(Color::Cyan));
        assert!(
            destination
                .style
                .add_modifier
                .contains(Modifier::UNDERLINED)
        );
        assert_eq!(lines[0].hyperlinks.len(), 2);
        assert!(
            lines[0]
                .hyperlinks
                .iter()
                .all(|link| link.destination == "https://example.com/guide")
        );
    }

    #[test]
    fn bare_urls_are_linked_but_code_and_unsafe_schemes_are_not() {
        let lines = render_unconstrained(
            "See https://example.com/a~b. `https://code.example` [mail](mailto:a@example.com) [bad](javascript:alert(1))",
            Style::default(),
        );
        let links = lines
            .iter()
            .flat_map(|line| &line.hyperlinks)
            .collect::<Vec<_>>();

        assert_eq!(links.len(), 1);
        assert_eq!(links[0].destination, "https://example.com/a~b");
    }

    #[test]
    fn local_file_links_collapse_matching_labels_and_keep_descriptions() {
        let lines = render_unconstrained(
            "[src/lib.rs](./src/lib.rs) and [open generated source](file:///tmp/My%20File.rs#L12C3)",
            Style::default(),
        );

        assert_eq!(
            plain(&lines),
            vec!["./src/lib.rs and open generated source (/tmp/My File.rs:12:3)"]
        );
        assert!(lines.iter().all(|line| line.hyperlinks.is_empty()));
    }

    #[test]
    fn local_file_links_preserve_tilde_unc_and_invalid_percent_spelling() {
        let lines = render_unconstrained(
            "[notes](~/notes) [share](file://server/share/My%20File.rs) [bad](/tmp/bad%FF.rs)",
            Style::default(),
        );
        let text = plain(&lines).join("\n");

        assert!(text.contains("~/notes"));
        assert!(text.contains("share (//server/share/My File.rs)"));
        assert!(text.contains("bad (/tmp/bad%FF.rs)"));
    }

    #[test]
    fn renders_tables_as_aligned_terminal_columns() {
        let lines = render_unconstrained(
            "| Name | Count |\n| :--- | ---: |\n| Lime | 2 |",
            Style::default(),
        );

        assert_eq!(
            plain(&lines),
            vec![" Name    Count ", "━━━━━━  ━━━━━━━", " Lime        2 "]
        );
        assert!(
            lines[0]
                .line
                .spans
                .iter()
                .any(|span| span.style.add_modifier.contains(Modifier::BOLD))
        );
    }

    #[test]
    fn constrained_tables_wrap_without_exceeding_available_width() {
        let lines = render(
            "| Name | Description |\n| --- | --- |\n| Lime | terminal interface with readable wrapped content |",
            Style::default(),
            Some(24),
        );

        assert!(lines.len() > 3);
        assert!(lines.iter().all(|line| line.width() <= 24));
        let text = plain(&lines).join("\n");
        assert!(text.contains("Name"));
        assert!(text.contains("terminal"));
        assert!(text.contains("content"));
    }

    #[test]
    fn constrained_halfwidth_table_keeps_display_width_bounded() {
        let lines = render(
            "| Kana | Value |\n| --- | --- |\n| ｶﾞﾊﾟ | ｶﾞﾊﾟtail |",
            Style::default(),
            Some(18),
        );

        assert!(lines.iter().all(|line| line.width() <= 18));
        assert!(plain(&lines).join("\n").contains("ｶﾞﾊﾟ"));
    }

    #[test]
    fn systemic_compact_fragmentation_uses_stacked_records() {
        let lines = render(
            "| c1 | c2 | c3 | c4 |\n| --- | --- | --- | --- |\n| alpha-long-token | beta-long-token | gamma-long-token | delta-long-token |",
            Style::default(),
            Some(16),
        );
        let text = plain(&lines);
        let compact = text.concat().replace(char::is_whitespace, "");

        assert!(lines.iter().all(|line| line.width() <= 16));
        assert!(text.iter().any(|line| line.trim() == "c1"));
        assert!(compact.contains("alpha-long-token"));
        assert!(!text.iter().any(|line| line.contains('━')));
    }

    #[test]
    fn nested_lists_keep_a_scannable_hierarchy() {
        let lines = render_unconstrained("- outer\n    - inner", Style::default());

        assert_eq!(plain(&lines), vec!["- outer", "    - inner"]);
    }
}
