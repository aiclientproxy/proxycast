//! Semantic terminal hyperlinks carried separately from visible TUI text.
//!
//! Layout code measures and wraps ordinary ratatui lines. Hyperlink annotations are applied only
//! when text reaches the terminal buffer, so OSC 8 bytes never affect geometry.

mod paragraph;

pub(crate) use paragraph::HyperlinkParagraph;

use std::num::NonZeroU16;
use std::ops::Range;

use ratatui::buffer::{Buffer, CellDiffOption, CellWidth};
use ratatui::layout::Rect;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Paragraph, Widget, Wrap};
use unicode_segmentation::UnicodeSegmentation;
use url::Url;

use crate::line_truncation::line_width;
use crate::width::display_width;

// Destinations are repeated in every linked buffer cell. Leave oversized URLs as plain text.
const MAX_HYPERLINK_DESTINATION_BYTES: usize = 8 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TerminalHyperlink {
    pub(crate) columns: Range<usize>,
    pub(crate) destination: String,
}

impl TerminalHyperlink {
    pub(crate) fn web(columns: Range<usize>, destination: String) -> Self {
        Self {
            columns,
            destination,
        }
    }

    fn with_columns(&self, columns: Range<usize>) -> Self {
        Self {
            columns,
            destination: self.destination.clone(),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct HyperlinkLine {
    pub(crate) line: Line<'static>,
    pub(crate) hyperlinks: Vec<TerminalHyperlink>,
}

impl HyperlinkLine {
    pub(crate) fn new(line: Line<'static>) -> Self {
        Self {
            line,
            hyperlinks: Vec::new(),
        }
    }

    pub(crate) fn width(&self) -> usize {
        line_width(&self.line)
    }

    pub(crate) fn push_span(&mut self, span: Span<'static>, destination: Option<&str>) {
        let start = self.width();
        let end = start + display_width(span.content.as_ref());
        self.line.push_span(span);
        if end > start {
            if let Some(destination) = destination.and_then(web_destination) {
                self.hyperlinks
                    .push(TerminalHyperlink::web(start..end, destination));
            }
        }
    }

    pub(crate) fn push_auto_link_span(&mut self, span: Span<'static>) {
        let shift = self.width();
        let links = web_links_in_text(span.content.as_ref());
        self.line.push_span(span);
        self.hyperlinks.extend(links.into_iter().map(|mut link| {
            link.columns = link.columns.start + shift..link.columns.end + shift;
            link
        }));
    }

    pub(crate) fn append(&mut self, mut other: Self) {
        let shift = self.width();
        self.line.spans.append(&mut other.line.spans);
        self.hyperlinks
            .extend(other.hyperlinks.into_iter().map(|mut link| {
                link.columns = link.columns.start + shift..link.columns.end + shift;
                link
            }));
    }
}

impl From<Line<'static>> for HyperlinkLine {
    fn from(line: Line<'static>) -> Self {
        Self::new(line)
    }
}

impl From<&'static str> for HyperlinkLine {
    fn from(text: &'static str) -> Self {
        Self::new(Line::from(text))
    }
}

impl From<String> for HyperlinkLine {
    fn from(text: String) -> Self {
        Self::new(Line::from(text))
    }
}

pub(crate) fn visible_lines_ref(lines: &[HyperlinkLine]) -> Vec<Line<'static>> {
    lines.iter().map(|line| line.line.clone()).collect()
}

pub(crate) fn prefix_hyperlink_lines(
    lines: Vec<HyperlinkLine>,
    initial_prefix: Span<'static>,
    subsequent_prefix: Span<'static>,
) -> Vec<HyperlinkLine> {
    lines
        .into_iter()
        .enumerate()
        .map(|(index, mut line)| {
            let prefix = if index == 0 {
                initial_prefix.clone()
            } else {
                subsequent_prefix.clone()
            };
            let shift = display_width(prefix.content.as_ref());
            line.line.spans.insert(0, prefix);
            for hyperlink in &mut line.hyperlinks {
                hyperlink.columns = hyperlink.columns.start + shift..hyperlink.columns.end + shift;
            }
            line
        })
        .collect()
}

/// Re-attach source hyperlink ranges after visible-text wrapping has split a line.
pub(crate) fn remap_wrapped_line(
    source: &HyperlinkLine,
    wrapped: Vec<Line<'static>>,
) -> Vec<HyperlinkLine> {
    let mut out = wrapped
        .into_iter()
        .map(HyperlinkLine::new)
        .collect::<Vec<_>>();
    if source.hyperlinks.is_empty() {
        return out;
    }

    let source_text = line_text(&source.line);
    let mut source_byte = 0usize;
    let mut source_column = 0usize;
    let mut link_index = 0usize;
    for (index, line) in out.iter_mut().enumerate() {
        if index > 0 {
            let trimmed = source_text[source_byte..].trim_start_matches(char::is_whitespace);
            let skipped = source_text[source_byte..].len() - trimmed.len();
            source_column += display_width(&source_text[source_byte..source_byte + skipped]);
            source_byte += skipped;
        }

        let rendered = line_text(&line.line);
        let remaining = &source_text[source_byte..];
        let Some(rendered_start) = longest_suffix_matching_prefix(&rendered, remaining) else {
            continue;
        };
        let mapped = &rendered[rendered_start..];
        let mut output_column = display_width(&rendered[..rendered_start]);
        for grapheme in mapped.graphemes(true) {
            let width = display_width(grapheme);
            while source
                .hyperlinks
                .get(link_index)
                .is_some_and(|link| link.columns.end <= source_column)
            {
                link_index += 1;
            }
            if let Some(link) = source
                .hyperlinks
                .get(link_index)
                .filter(|link| link.columns.contains(&source_column))
            {
                push_link_range(line, output_column..output_column + width, link);
            }
            source_column += width;
            output_column += width;
        }
        source_byte += mapped.len();
    }
    out
}

pub(crate) fn wrap_hyperlink_line(source: &HyperlinkLine, width: usize) -> Vec<HyperlinkLine> {
    if !source.hyperlinks.is_empty() {
        let wrapped =
            crate::wrapping::word_wrap_line(&source.line, crate::wrapping::RtOptions::new(width));
        return remap_wrapped_line(source, crate::wrapping::own_lines(wrapped));
    }
    if crate::wrapping::line_contains_url_like(&source.line) {
        let wrapped = crate::wrapping::adaptive_wrap_line(
            &source.line,
            crate::wrapping::RtOptions::new(width),
        );
        return remap_wrapped_line(source, crate::wrapping::own_lines(wrapped));
    }
    let width = width.max(1).min(usize::from(u16::MAX)) as u16;
    let paragraph = Paragraph::new(Text::from(source.line.clone())).wrap(Wrap { trim: false });
    let height = paragraph
        .line_count(width)
        .max(1)
        .min(usize::from(u16::MAX)) as u16;
    let area = Rect::new(0, 0, width, height);
    let mut buffer = Buffer::empty(area);
    paragraph.render(area, &mut buffer);
    remap_wrapped_line(source, visible_buffer_lines(&buffer, area))
}

fn line_text(line: &Line<'_>) -> String {
    line.spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
}

fn longest_suffix_matching_prefix(rendered: &str, source: &str) -> Option<usize> {
    rendered
        .grapheme_indices(true)
        .map(|(index, _)| index)
        .chain(std::iter::once(rendered.len()))
        .find(|index| source.starts_with(&rendered[*index..]) && *index < rendered.len())
}

fn push_link_range(line: &mut HyperlinkLine, range: Range<usize>, link: &TerminalHyperlink) {
    if range.is_empty() {
        return;
    }
    if let Some(previous) = line.hyperlinks.last_mut() {
        if previous.destination == link.destination && previous.columns.end == range.start {
            previous.columns.end = range.end;
            return;
        }
    }
    line.hyperlinks.push(link.with_columns(range));
}

pub(crate) fn web_links_in_text(text: &str) -> Vec<TerminalHyperlink> {
    let mut links = Vec::new();
    let mut search_from = 0usize;
    let mut source_byte = 0usize;
    let mut source_column = 0usize;
    for raw_token in text.split_whitespace() {
        let Some(relative_start) = text[search_from..].find(raw_token) else {
            continue;
        };
        let raw_start = search_from + relative_start;
        search_from = raw_start + raw_token.len();
        let trimmed_start = raw_token
            .find(|ch: char| !is_leading_punctuation(ch))
            .unwrap_or(raw_token.len());
        let trimmed_end = trailing_url_end(&raw_token[trimmed_start..]) + trimmed_start;
        if trimmed_start >= trimmed_end {
            continue;
        }
        let candidate = &raw_token[trimmed_start..trimmed_end];
        let Some(destination) = web_destination(candidate) else {
            continue;
        };
        let candidate_start = raw_start + trimmed_start;
        source_column += display_width(&text[source_byte..candidate_start]);
        source_byte = candidate_start;
        let end = source_column + display_width(candidate);
        links.push(TerminalHyperlink::web(source_column..end, destination));
    }
    links
}

fn is_leading_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ',' | '.' | ';' | '!' | '\'' | '"'
    )
}

fn trailing_url_end(candidate: &str) -> usize {
    let mut balances = [0isize; 4];
    for ch in candidate.chars() {
        match ch {
            '(' => balances[0] += 1,
            ')' => balances[0] -= 1,
            '[' => balances[1] += 1,
            ']' => balances[1] -= 1,
            '{' => balances[2] += 1,
            '}' => balances[2] -= 1,
            '<' => balances[3] += 1,
            '>' => balances[3] -= 1,
            _ => {}
        }
    }
    let mut end = candidate.len();
    while end > 0 {
        let remaining = &candidate[..end];
        let Some(ch) = remaining.chars().next_back() else {
            break;
        };
        let balance = match ch {
            ')' => Some(&mut balances[0]),
            ']' => Some(&mut balances[1]),
            '}' => Some(&mut balances[2]),
            '>' => Some(&mut balances[3]),
            _ => None,
        };
        let trim = if let Some(balance) = balance {
            let unmatched = *balance < 0;
            *balance += 1;
            unmatched
        } else {
            matches!(ch, ',' | '.' | ';' | '!' | '\'' | '"')
        };
        if !trim {
            break;
        }
        end -= ch.len_utf8();
    }
    end
}

pub(crate) fn web_destination(destination: &str) -> Option<String> {
    let safe_destination = sanitized_destination(destination)?;
    let parsed = Url::parse(&safe_destination).ok()?;
    matches!(parsed.scheme(), "http" | "https")
        .then(|| parsed.host_str())
        .flatten()?;
    Some(safe_destination)
}

fn sanitized_destination(destination: &str) -> Option<String> {
    if destination.len() > MAX_HYPERLINK_DESTINATION_BYTES {
        return None;
    }
    Some(destination.chars().filter(|ch| !ch.is_control()).collect())
}

fn osc8_hyperlink(destination: &str, text: &str) -> String {
    let Some(safe_destination) = web_destination(destination) else {
        return text.to_string();
    };
    format!("\x1b]8;;{safe_destination}\x07{text}\x1b]8;;\x07")
}

#[cfg(test)]
pub(crate) fn strip_osc8(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut stripped = String::with_capacity(text.len());
    let mut index = 0usize;

    while index < bytes.len() {
        if bytes[index..].starts_with(b"\x1b]8;;") {
            index += 5;
            while index < bytes.len() {
                if bytes[index] == b'\x07' {
                    index += 1;
                    break;
                }
                if index + 1 < bytes.len() && bytes[index] == b'\x1b' && bytes[index + 1] == b'\\' {
                    index += 2;
                    break;
                }
                index += 1;
            }
            continue;
        }
        let ch = text[index..]
            .chars()
            .next()
            .expect("current byte index starts a character");
        stripped.push(ch);
        index += ch.len_utf8();
    }

    stripped
}

pub(crate) fn mark_buffer_hyperlinks(
    buffer: &mut Buffer,
    area: Rect,
    lines: &[HyperlinkLine],
    scroll_rows: usize,
) {
    if area.width == 0 || area.height == 0 || lines.iter().all(|line| line.hyperlinks.is_empty()) {
        return;
    }
    let viewport_end = scroll_rows.saturating_add(usize::from(area.height));
    let mut logical_row = 0usize;
    for line in lines {
        if logical_row >= viewport_end {
            break;
        }
        let paragraph = Paragraph::new(Text::from(line.line.clone())).wrap(Wrap { trim: false });
        let rendered_height = paragraph.line_count(area.width).max(1);
        if line.hyperlinks.is_empty() || logical_row.saturating_add(rendered_height) <= scroll_rows
        {
            logical_row += rendered_height;
            continue;
        }

        let layout_area = Rect::new(
            0,
            0,
            area.width,
            u16::try_from(rendered_height).unwrap_or(u16::MAX),
        );
        let mut layout = Buffer::empty(layout_area);
        paragraph.render(layout_area, &mut layout);
        for (row, rendered) in remap_wrapped_line(line, visible_buffer_lines(&layout, layout_area))
            .iter()
            .enumerate()
        {
            let row = logical_row + row;
            if row < scroll_rows || row >= viewport_end {
                continue;
            }
            for link in &rendered.hyperlinks {
                let Some(destination) = web_destination(&link.destination) else {
                    continue;
                };
                let mut trailing_columns = 0usize;
                for column in link.columns.clone() {
                    if trailing_columns > 0 {
                        trailing_columns -= 1;
                        continue;
                    }
                    let Ok(column) = u16::try_from(column) else {
                        break;
                    };
                    if column >= area.width {
                        break;
                    }
                    let x = area.x + column;
                    let y = area.y + u16::try_from(row - scroll_rows).unwrap_or(u16::MAX);
                    let cell = &mut buffer[(x, y)];
                    if cell.diff_option == CellDiffOption::Skip {
                        continue;
                    }
                    trailing_columns = usize::from(cell.cell_width()).saturating_sub(1);
                    let symbol = osc8_hyperlink(&destination, cell.symbol());
                    let width = NonZeroU16::new(cell.cell_width()).unwrap_or(NonZeroU16::MIN);
                    cell.set_symbol(&symbol)
                        .set_diff_option(CellDiffOption::ForcedWidth(width));
                }
            }
        }
        logical_row += rendered_height;
    }
}

fn visible_buffer_lines(buffer: &Buffer, area: Rect) -> Vec<Line<'static>> {
    (0..area.height)
        .map(|row| {
            let mut spans = Vec::<Span<'static>>::new();
            let mut trailing_columns = 0usize;
            for column in 0..area.width {
                if trailing_columns > 0 {
                    trailing_columns -= 1;
                    continue;
                }
                let cell = &buffer[(column, row)];
                if cell.diff_option == CellDiffOption::Skip {
                    continue;
                }
                trailing_columns = usize::from(cell.cell_width()).saturating_sub(1);
                if let Some(previous) = spans.last_mut() {
                    if previous.style == cell.style() {
                        previous.content.to_mut().push_str(cell.symbol());
                        continue;
                    }
                }
                spans.push(Span::styled(cell.symbol().to_string(), cell.style()));
            }
            while let Some(last) = spans.last_mut() {
                let trimmed = last.content.trim_end().to_string();
                if trimmed.is_empty() {
                    spans.pop();
                } else {
                    last.content = trimmed.into();
                    break;
                }
            }
            Line::from(spans)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_safe_web_destinations_receive_osc8() {
        assert!(osc8_hyperlink("https://example.com/a", "a").contains("\x1b]8;;"));
        assert_eq!(osc8_hyperlink("mailto:a@example.com", "a"), "a");
        assert_eq!(osc8_hyperlink("javascript:alert(1)", "a"), "a");
        assert_eq!(
            osc8_hyperlink("https://example.com/\u{1b}]8;;bad\u{7}safe", "a"),
            "\x1b]8;;https://example.com/]8;;badsafe\x07a\x1b]8;;\x07"
        );
    }

    #[test]
    fn discovers_punctuated_and_balanced_web_urls() {
        let destination = "https://en.wikipedia.org/wiki/Function_(mathematics)";
        assert_eq!(
            web_links_in_text(&format!("See ({destination}).")),
            vec![TerminalHyperlink::web(
                5..5 + display_width(destination),
                destination.to_string(),
            )]
        );
    }

    #[test]
    fn oversized_destinations_remain_plain_text() {
        let prefix = "https://example.com/";
        let destination = format!(
            "{prefix}{}",
            "a".repeat(MAX_HYPERLINK_DESTINATION_BYTES - prefix.len())
        );
        assert_eq!(web_destination(&destination), Some(destination.clone()));
        assert_eq!(web_destination(&format!("{destination}é")), None);
    }

    #[test]
    fn buffer_hyperlinks_follow_scrolled_wrapped_rows() {
        let hidden_destination = "https://example.com/hidden";
        let visible_destination = "https://example.com/visible";
        let mut hidden = HyperlinkLine::new(Line::default());
        hidden.push_span("hidden".into(), Some(hidden_destination));
        let mut visible = HyperlinkLine::new(Line::from("prefix "));
        visible.push_span("visible-link".into(), Some(visible_destination));
        let lines = vec![hidden, visible];
        let area = Rect::new(0, 0, 8, 2);
        let mut buffer = Buffer::empty(area);

        HyperlinkParagraph::new(&lines)
            .scroll(2)
            .render(area, &mut buffer);

        let linked_text = area
            .positions()
            .filter_map(|position| {
                let symbol = buffer[position].symbol();
                symbol
                    .contains(&format!("\x1b]8;;{visible_destination}\x07"))
                    .then(|| strip_osc8(symbol))
            })
            .collect::<String>();
        assert_eq!(linked_text, "visible-link");
    }

    #[test]
    fn forced_width_hyperlinks_preserve_wide_and_halfwidth_cells() {
        let destination = "https://example.com/rendered";
        let mut line = HyperlinkLine::new(Line::from("prefix "));
        line.push_span("漢字 ｶﾞ".into(), Some(destination));
        line.push_span(" tail".into(), None);
        let wrapped = wrap_hyperlink_line(&line, 14);
        assert!(
            wrapped.iter().any(|line| !line.hyperlinks.is_empty()),
            "source={line:#?} wrapped={wrapped:#?}"
        );
        let area = Rect::new(0, 0, 14, 3);
        let mut buffer = Buffer::empty(area);

        HyperlinkParagraph::new(&[line]).render(area, &mut buffer);

        let linked = area
            .positions()
            .filter_map(|position| {
                let cell = &buffer[position];
                cell.symbol()
                    .contains(&format!("\x1b]8;;{destination}\x07"))
                    .then(|| {
                        (
                            strip_osc8(cell.symbol()),
                            cell.cell_width(),
                            cell.diff_option,
                        )
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            linked
                .iter()
                .map(|(text, _, _)| text.as_str())
                .collect::<String>(),
            "漢字 ｶﾞ",
            "buffer={buffer:#?}"
        );
        assert!(linked.iter().all(|(_, width, option)| {
            matches!(option, CellDiffOption::ForcedWidth(forced) if forced.get() == *width)
        }));
    }

    #[test]
    fn wrapping_keeps_explicit_destination_on_every_fragment() {
        let destination = "https://example.com/a/very/long/path";
        let mut source = HyperlinkLine::new(Line::default());
        source.push_span(destination.into(), Some(destination));

        let wrapped = wrap_hyperlink_line(&source, 10);

        assert!(wrapped.len() > 1);
        assert!(wrapped.iter().all(|line| {
            !line.hyperlinks.is_empty()
                && line
                    .hyperlinks
                    .iter()
                    .all(|link| link.destination == destination)
        }));
    }

    #[test]
    fn strips_osc8_without_changing_visible_text() {
        let encoded = osc8_hyperlink("https://example.com", "visible");
        assert_eq!(strip_osc8(&encoded), "visible");
    }
}
