//! Codex-shaped terminal line wrapping with URL-aware heuristics.
//!
//! Wrapping remains a render concern: the canonical Thread/Turn/Item model is
//! never rewritten. The module preserves ratatui span styles while ensuring
//! URLs are not split at `/` or `-`.

use std::borrow::Cow;
use std::ops::Range;

use ratatui::text::{Line, Span};
use textwrap::{Options, WordSeparator, WordSplitter};
use unicode_segmentation::UnicodeSegmentation;

use crate::line_truncation::line_width;

/// Textwrap options using ratatui lines for styled indentation.
#[derive(Debug, Clone)]
pub(crate) struct RtOptions<'a> {
    pub(crate) width: usize,
    pub(crate) line_ending: textwrap::LineEnding,
    pub(crate) initial_indent: Line<'a>,
    pub(crate) subsequent_indent: Line<'a>,
    pub(crate) break_words: bool,
    pub(crate) wrap_algorithm: textwrap::WrapAlgorithm,
    pub(crate) word_separator: textwrap::WordSeparator,
    pub(crate) word_splitter: textwrap::WordSplitter,
}

impl From<usize> for RtOptions<'_> {
    fn from(width: usize) -> Self {
        Self::new(width)
    }
}

impl<'a> RtOptions<'a> {
    pub(crate) fn new(width: usize) -> Self {
        Self {
            width,
            line_ending: textwrap::LineEnding::LF,
            initial_indent: Line::default(),
            subsequent_indent: Line::default(),
            break_words: true,
            wrap_algorithm: textwrap::WrapAlgorithm::FirstFit,
            word_separator: WordSeparator::new(),
            word_splitter: WordSplitter::HyphenSplitter,
        }
    }

    pub(crate) fn initial_indent(self, value: Line<'a>) -> Self {
        Self {
            initial_indent: value,
            ..self
        }
    }

    #[allow(dead_code)]
    pub(crate) fn subsequent_indent(self, value: Line<'a>) -> Self {
        Self {
            subsequent_indent: value,
            ..self
        }
    }

    pub(crate) fn break_words(self, value: bool) -> Self {
        Self {
            break_words: value,
            ..self
        }
    }

    pub(crate) fn word_separator(self, value: WordSeparator) -> Self {
        Self {
            word_separator: value,
            ..self
        }
    }

    #[allow(dead_code)]
    pub(crate) fn wrap_algorithm(self, value: textwrap::WrapAlgorithm) -> Self {
        Self {
            wrap_algorithm: value,
            ..self
        }
    }

    pub(crate) fn word_splitter(self, value: WordSplitter) -> Self {
        Self {
            word_splitter: value,
            ..self
        }
    }
}

/// Return source ranges for wrapped lines, including a one-byte cursor sentinel.
#[allow(dead_code)]
pub(crate) fn wrap_ranges<'a, O>(text: &str, width_or_options: O) -> Vec<Range<usize>>
where
    O: Into<Options<'a>>,
{
    ranges_for_wrapped(text, width_or_options.into(), true)
}

/// Return source ranges for wrapped lines without trailing whitespace.
pub(crate) fn wrap_ranges_trim<'a, O>(text: &str, width_or_options: O) -> Vec<Range<usize>>
where
    O: Into<Options<'a>>,
{
    ranges_for_wrapped(text, width_or_options.into(), false)
}

fn ranges_for_wrapped<'a>(text: &str, options: Options<'a>, sentinel: bool) -> Vec<Range<usize>> {
    let mut cursor = 0usize;
    let mut ranges = Vec::new();
    for (index, wrapped) in textwrap::wrap(text, &options).iter().enumerate() {
        let indent = if index == 0 {
            options.initial_indent
        } else {
            options.subsequent_indent
        };
        let range = match wrapped {
            Cow::Borrowed(slice) => borrowed_range(text, slice)
                .unwrap_or_else(|| map_owned_range(text, cursor, slice, indent)),
            Cow::Owned(slice) => map_owned_range(text, cursor, slice, indent),
        };
        cursor = range.end;
        if sentinel {
            ranges.push(range.start..range.end.saturating_add(1).min(text.len() + 1));
        } else {
            ranges.push(range);
        }
    }
    ranges
}

fn borrowed_range(text: &str, slice: &str) -> Option<Range<usize>> {
    let text_start = text.as_ptr() as usize;
    let slice_start = slice.as_ptr() as usize;
    let offset = slice_start.checked_sub(text_start)?;
    (offset + slice.len() <= text.len()).then_some(offset..offset + slice.len())
}

fn map_owned_range(text: &str, cursor: usize, wrapped: &str, indent: &str) -> Range<usize> {
    let content = wrapped.strip_prefix(indent).unwrap_or(wrapped);
    if content.is_empty() {
        return cursor..cursor;
    }
    let search = &text[cursor.min(text.len())..];
    if let Some(relative) = search.find(content) {
        let start = cursor + relative;
        return start..start + content.len();
    }

    let mut end = cursor;
    for grapheme in content.graphemes(true) {
        if text[end..].starts_with(grapheme) {
            end += grapheme.len();
        } else {
            break;
        }
    }
    cursor..end
}

/// Wrap one styled ratatui line with standard textwrap behavior.
#[must_use]
pub(crate) fn word_wrap_line<'a, O>(line: &'a Line<'a>, width_or_options: O) -> Vec<Line<'a>>
where
    O: Into<RtOptions<'a>>,
{
    wrap_line(line, width_or_options.into())
}

/// Wrap one styled ratatui line, preserving URL-like tokens.
#[must_use]
pub(crate) fn adaptive_wrap_line<'a>(line: &'a Line<'a>, options: RtOptions<'a>) -> Vec<Line<'a>> {
    if text_contains_url_like(&line_text(line)) {
        word_wrap_line(line, url_preserving_wrap_options(options))
    } else {
        word_wrap_line(line, options)
    }
}

fn wrap_line<'a>(line: &'a Line<'a>, options: RtOptions<'a>) -> Vec<Line<'a>> {
    let (flat, spans) = flatten_line(line);
    let initial_width = options
        .width
        .saturating_sub(line_width(&options.initial_indent))
        .max(1);
    let subsequent_width = options
        .width
        .saturating_sub(line_width(&options.subsequent_indent))
        .max(1);
    let base = textwrap_options(&options, initial_width);
    let first = wrap_ranges_trim(&flat, base);
    if first.is_empty() {
        return vec![options.initial_indent.clone()];
    }

    let mut output = Vec::new();
    for (index, range) in first.iter().enumerate() {
        let indent = if index == 0 {
            &options.initial_indent
        } else {
            &options.subsequent_indent
        };
        let mut wrapped = indent.clone();
        wrapped.style = line.style;
        wrapped
            .spans
            .extend(slice_line_spans(line, &spans, range).spans);
        output.push(wrapped);
    }

    if let Some(last) = first.last() {
        let rest_start = flat[last.end..]
            .char_indices()
            .find_map(|(offset, ch)| (!ch.is_whitespace()).then_some(last.end + offset))
            .unwrap_or(flat.len());
        if rest_start < flat.len() {
            let rest_options = textwrap_options(&options, subsequent_width);
            for range in wrap_ranges_trim(&flat[rest_start..], rest_options) {
                let range = (range.start + rest_start)..(range.end + rest_start);
                let mut wrapped = options.subsequent_indent.clone();
                wrapped.style = line.style;
                wrapped
                    .spans
                    .extend(slice_line_spans(line, &spans, &range).spans);
                output.push(wrapped);
            }
        }
    }
    output
}

fn textwrap_options<'a>(options: &RtOptions<'a>, width: usize) -> Options<'a> {
    Options::new(width)
        .line_ending(options.line_ending)
        .break_words(options.break_words)
        .wrap_algorithm(options.wrap_algorithm)
        .word_separator(options.word_separator)
        .word_splitter(options.word_splitter.clone())
}

fn flatten_line(line: &Line<'_>) -> (String, Vec<(Range<usize>, ratatui::style::Style)>) {
    let mut text = String::new();
    let mut spans = Vec::new();
    let mut cursor = 0;
    for span in &line.spans {
        let start = cursor;
        text.push_str(span.content.as_ref());
        cursor += span.content.len();
        spans.push((start..cursor, span.style));
    }
    (text, spans)
}

fn slice_line_spans<'a>(
    original: &'a Line<'a>,
    bounds: &[(Range<usize>, ratatui::style::Style)],
    requested: &Range<usize>,
) -> Line<'a> {
    let mut output = Vec::new();
    for (index, (bound, style)) in bounds.iter().enumerate() {
        if bound.end <= requested.start {
            continue;
        }
        if bound.start >= requested.end {
            break;
        }
        let start = bound.start.max(requested.start) - bound.start;
        let end = bound.end.min(requested.end) - bound.start;
        if start < end {
            let content = original.spans[index].content.as_ref();
            output.push(Span {
                style: *style,
                content: Cow::Borrowed(&content[start..end]),
            });
        }
    }
    Line {
        style: original.style,
        alignment: original.alignment,
        spans: output,
    }
}

fn line_text(line: &Line<'_>) -> String {
    line.spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
}

/// Wrap a sequence of lines and return owned output suitable for a transcript.
#[allow(dead_code, private_bounds)]
pub(crate) fn word_wrap_lines<'a, I, O, L>(lines: I, options: O) -> Vec<Line<'static>>
where
    I: IntoIterator<Item = L>,
    L: IntoLineInput<'a>,
    O: Into<RtOptions<'a>>,
{
    wrap_lines(lines, options.into(), false)
}

/// Multi-line URL-aware counterpart of [`word_wrap_lines`].
#[allow(dead_code, private_bounds)]
pub(crate) fn adaptive_wrap_lines<'a, I, L>(lines: I, options: RtOptions<'a>) -> Vec<Line<'static>>
where
    I: IntoIterator<Item = L>,
    L: IntoLineInput<'a>,
{
    wrap_lines(lines, options, true)
}

fn wrap_lines<'a, I, L>(lines: I, options: RtOptions<'a>, adaptive: bool) -> Vec<Line<'static>>
where
    I: IntoIterator<Item = L>,
    L: IntoLineInput<'a>,
{
    let mut output = Vec::new();
    for (index, input) in lines.into_iter().enumerate() {
        let line = input.into_line_input();
        let line_options = if index == 0 {
            options.clone()
        } else {
            options
                .clone()
                .initial_indent(options.subsequent_indent.clone())
        };
        let wrapped = if adaptive {
            adaptive_wrap_line(line.as_ref(), line_options)
        } else {
            word_wrap_line(line.as_ref(), line_options)
        };
        output.extend(wrapped.iter().map(owned_line));
    }
    output
}

fn owned_line(line: &Line<'_>) -> Line<'static> {
    Line {
        style: line.style,
        alignment: line.alignment,
        spans: line
            .spans
            .iter()
            .map(|span| Span::styled(span.content.to_string(), span.style))
            .collect(),
    }
}

pub(crate) fn own_lines<'a>(lines: Vec<Line<'a>>) -> Vec<Line<'static>> {
    lines.iter().map(owned_line).collect()
}

#[derive(Debug)]
enum LineInput<'a> {
    Borrowed(&'a Line<'a>),
    Owned(Line<'a>),
}

impl<'a> LineInput<'a> {
    fn as_ref(&self) -> &Line<'a> {
        match self {
            Self::Borrowed(line) => line,
            Self::Owned(line) => line,
        }
    }
}

#[allow(dead_code)]
trait IntoLineInput<'a> {
    fn into_line_input(self) -> LineInput<'a>;
}

impl<'a> IntoLineInput<'a> for &'a Line<'a> {
    fn into_line_input(self) -> LineInput<'a> {
        LineInput::Borrowed(self)
    }
}

#[allow(dead_code)]
impl<'a> IntoLineInput<'a> for &'a mut Line<'a> {
    fn into_line_input(self) -> LineInput<'a> {
        LineInput::Borrowed(self)
    }
}

impl<'a> IntoLineInput<'a> for Line<'a> {
    fn into_line_input(self) -> LineInput<'a> {
        LineInput::Owned(self)
    }
}

impl<'a> IntoLineInput<'a> for &'a str {
    fn into_line_input(self) -> LineInput<'a> {
        LineInput::Owned(Line::from(self))
    }
}

impl<'a> IntoLineInput<'a> for String {
    fn into_line_input(self) -> LineInput<'a> {
        LineInput::Owned(Line::from(self))
    }
}

#[allow(dead_code)]
impl<'a> IntoLineInput<'a> for Cow<'a, str> {
    fn into_line_input(self) -> LineInput<'a> {
        LineInput::Owned(Line::from(self))
    }
}

#[allow(dead_code)]
impl<'a> IntoLineInput<'a> for Span<'a> {
    fn into_line_input(self) -> LineInput<'a> {
        LineInput::Owned(Line::from(self))
    }
}

#[allow(dead_code)]
impl<'a> IntoLineInput<'a> for Vec<Span<'a>> {
    fn into_line_input(self) -> LineInput<'a> {
        LineInput::Owned(Line::from(self))
    }
}

/// Whether any whitespace-delimited token resembles a URL.
pub(crate) fn text_contains_url_like(text: &str) -> bool {
    text.split_ascii_whitespace().any(is_url_like_token)
}

pub(crate) fn line_contains_url_like(line: &Line<'_>) -> bool {
    text_contains_url_like(&line_text(line))
}

#[allow(dead_code)]
pub(crate) fn line_has_mixed_url_and_non_url_tokens(line: &Line<'_>) -> bool {
    let mut has_url = false;
    let mut has_non_url = false;
    for token in line_text(line).split_ascii_whitespace() {
        if is_url_like_token(token) {
            has_url = true;
        } else if token.chars().any(char::is_alphanumeric) {
            has_non_url = true;
        }
        if has_url && has_non_url {
            return true;
        }
    }
    false
}

pub(crate) fn url_preserving_wrap_options<'a>(options: RtOptions<'a>) -> RtOptions<'a> {
    options
        .word_separator(WordSeparator::AsciiSpace)
        .word_splitter(WordSplitter::NoHyphenation)
        .break_words(false)
}

fn is_url_like_token(raw: &str) -> bool {
    let token = raw.trim_matches(|ch: char| {
        matches!(
            ch,
            '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ',' | '.' | ';' | ':' | '!'
        )
    });
    if token.contains("://") {
        return url::Url::parse(token)
            .map(|url| url.host_str().is_some())
            .unwrap_or_else(|_| {
                token
                    .split_once("://")
                    .is_some_and(|(_, rest)| !rest.is_empty())
            });
    }

    let host = token.split(['/', '?', '#']).next().unwrap_or_default();
    let Some((_, tld)) = host.rsplit_once('.') else {
        return host == "localhost" && token.len() > host.len();
    };
    host.len() > 3
        && tld.len() >= 2
        && tld.chars().all(|ch| ch.is_ascii_alphabetic())
        && host
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '.')
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::{Color, Style};

    fn rendered(lines: &[Line<'_>]) -> String {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn word_wrap_preserves_styled_spans_and_indent() {
        let line = Line::from(vec![
            Span::styled("hello ", Style::default().fg(Color::Red)),
            Span::styled("world", Style::default().fg(Color::Green)),
        ]);
        let wrapped = word_wrap_line(&line, RtOptions::new(8).subsequent_indent(Line::from("  ")));
        assert_eq!(rendered(&wrapped), "hello\n  world");
        assert_eq!(wrapped[0].spans[0].style.fg, Some(Color::Red));
        assert_eq!(
            wrapped[1].spans.last().unwrap().style.fg,
            Some(Color::Green)
        );
    }

    #[test]
    fn adaptive_wrap_keeps_url_token_intact() {
        let line = Line::from("see https://example.com/a-long/path now");
        let wrapped = adaptive_wrap_line(&line, RtOptions::new(18));
        let text = rendered(&wrapped);
        assert!(text.contains("https://example.com/a-long/path"));
        assert!(line_contains_url_like(&line));
    }

    #[test]
    fn url_detection_rejects_file_paths() {
        assert!(
            text_contains_url_like("open https://example.com")
                .then_some(())
                .is_some()
        );
        assert!(!text_contains_url_like("src/main.rs foo/bar"));
    }

    #[test]
    fn wrap_ranges_are_utf8_safe() {
        let text = "你好 世界";
        let ranges = wrap_ranges_trim(text, Options::new(5));
        assert_eq!(
            ranges
                .iter()
                .map(|range| &text[range.clone()])
                .collect::<Vec<_>>(),
            vec!["你好", "世界"]
        );
        assert!(
            wrap_ranges(text, Options::new(5))
                .iter()
                .all(|range| range.end <= text.len() + 1)
        );
    }

    #[test]
    fn multiline_wrap_returns_owned_lines() {
        let lines = word_wrap_lines(["one two", "three"], 5usize);
        assert_eq!(rendered(&lines), "one\ntwo\nthree");
        assert!(lines.iter().all(|line| {
            line.spans
                .iter()
                .all(|span| matches!(span.content, Cow::Owned(_)))
        }));
    }
}
