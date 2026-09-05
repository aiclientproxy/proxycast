//! Canonical markdown table and fenced-code detection helpers.
//!
//! These helpers are intentionally independent from the renderer. Streaming
//! markdown and the terminal renderer can therefore share the same structural
//! decisions without creating a second transcript or parser owner.

#![allow(dead_code)]

/// Split a pipe-delimited line into trimmed segments.
pub(crate) fn parse_table_segments(line: &str) -> Option<Vec<&str>> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    let has_outer_pipe = trimmed.starts_with('|') || trimmed.ends_with('|');
    let content = trimmed.strip_prefix('|').unwrap_or(trimmed);
    let content = content.strip_suffix('|').unwrap_or(content);
    let raw_segments = split_unescaped_pipe(content);
    if !has_outer_pipe && raw_segments.len() <= 1 {
        return None;
    }

    let segments = raw_segments.into_iter().map(str::trim).collect::<Vec<_>>();
    (!segments.is_empty()).then_some(segments)
}

fn split_unescaped_pipe(content: &str) -> Vec<&str> {
    let mut segments = Vec::with_capacity(8);
    let mut start = 0;
    let bytes = content.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            index += 2;
        } else if bytes[index] == b'|' {
            segments.push(&content[start..index]);
            start = index + 1;
            index += 1;
        } else {
            index += 1;
        }
    }
    segments.push(&content[start..]);
    segments
}

/// Whether a line can be a markdown table header.
#[inline]
pub(crate) fn is_table_header_line(line: &str) -> bool {
    parse_table_segments(line).is_some_and(|segments| segments.iter().any(|s| !s.is_empty()))
}

fn is_table_delimiter_segment(segment: &str) -> bool {
    let trimmed = segment.trim();
    if trimmed.is_empty() {
        return false;
    }
    let without_leading = trimmed.strip_prefix(':').unwrap_or(trimmed);
    let without_ends = without_leading.strip_suffix(':').unwrap_or(without_leading);
    without_ends.len() >= 3 && without_ends.chars().all(|ch| ch == '-')
}

/// Whether a line is a valid GFM table delimiter row.
#[inline]
pub(crate) fn is_table_delimiter_line(line: &str) -> bool {
    parse_table_segments(line)
        .is_some_and(|segments| segments.into_iter().all(is_table_delimiter_segment))
}

/// Whether two adjacent lines form a table header and delimiter pair.
pub(crate) fn is_table_start(header: &str, delimiter: &str) -> bool {
    is_table_header_line(header) && is_table_delimiter_line(delimiter)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FenceKind {
    Outside,
    Markdown,
    Other,
}

/// Incremental fenced-code-block tracker used by streaming table detection.
#[derive(Debug, Default)]
pub(crate) struct FenceTracker {
    state: Option<(char, usize, FenceKind)>,
}

impl FenceTracker {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Advance the tracker with one raw source line.
    pub(crate) fn advance(&mut self, raw_line: &str) {
        let leading_spaces = raw_line
            .as_bytes()
            .iter()
            .take_while(|byte| **byte == b' ')
            .count();
        if leading_spaces > 3 {
            return;
        }

        let trimmed = &raw_line[leading_spaces..];
        let fence_text = strip_blockquote_prefix(trimmed);
        let Some((marker, length)) = parse_fence_marker(fence_text) else {
            return;
        };

        if let Some((open_marker, open_length, _)) = self.state {
            if marker == open_marker
                && length >= open_length
                && fence_text[length..].trim().is_empty()
            {
                self.state = None;
            }
            return;
        }

        let kind = if is_markdown_fence_info(fence_text, length) {
            FenceKind::Markdown
        } else {
            FenceKind::Other
        };
        self.state = Some((marker, length, kind));
    }

    pub(crate) fn kind(&self) -> FenceKind {
        self.state.map_or(FenceKind::Outside, |(_, _, kind)| kind)
    }
}

/// Return the fence marker character and run length for a potential fence.
pub(crate) fn parse_fence_marker(line: &str) -> Option<(char, usize)> {
    let first = line.as_bytes().first().copied()?;
    if first != b'`' && first != b'~' {
        return None;
    }
    let length = line.bytes().take_while(|byte| *byte == first).count();
    (length >= 3).then_some((first as char, length))
}

/// Whether the info string identifies a markdown fence.
pub(crate) fn is_markdown_fence_info(line: &str, marker_length: usize) -> bool {
    let info = line[marker_length..]
        .split_whitespace()
        .next()
        .unwrap_or_default();
    info.eq_ignore_ascii_case("md") || info.eq_ignore_ascii_case("markdown")
}

/// Remove one or more blockquote prefixes before structural parsing.
pub(crate) fn strip_blockquote_prefix(line: &str) -> &str {
    let mut rest = line.trim_start();
    loop {
        let Some(stripped) = rest.strip_prefix('>') else {
            return rest;
        };
        rest = stripped.strip_prefix(' ').unwrap_or(stripped).trim_start();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_table_segments_supports_outer_and_escaped_pipes() {
        assert_eq!(
            parse_table_segments("| A | B \\| literal |"),
            Some(vec!["A", "B \\| literal"])
        );
        assert_eq!(parse_table_segments("plain text"), None);
    }

    #[test]
    fn table_start_requires_header_and_delimiter() {
        assert!(is_table_start("A | B", "--- | :---:"));
        assert!(!is_table_start("A | B", "-- | ---"));
        assert!(!is_table_start("plain text", "--- | ---"));
    }

    #[test]
    fn fence_tracker_tracks_markdown_other_and_blockquotes() {
        let mut tracker = FenceTracker::new();
        assert_eq!(tracker.kind(), FenceKind::Outside);
        tracker.advance("```rust");
        assert_eq!(tracker.kind(), FenceKind::Other);
        tracker.advance("| code | not a table |");
        assert_eq!(tracker.kind(), FenceKind::Other);
        tracker.advance("```");
        assert_eq!(tracker.kind(), FenceKind::Outside);
        tracker.advance("> ```markdown");
        assert_eq!(tracker.kind(), FenceKind::Markdown);
        tracker.advance("> ```");
        assert_eq!(tracker.kind(), FenceKind::Outside);
    }

    #[test]
    fn fence_helpers_reject_short_or_indented_markers() {
        assert_eq!(parse_fence_marker("``"), None);
        assert_eq!(parse_fence_marker("~~~"), Some(('~', 3)));
        assert_eq!(strip_blockquote_prefix("> > | a | b |"), "| a | b |");
        assert!(is_markdown_fence_info("```MD title", 3));
        assert!(!is_markdown_fence_info("```rust", 3));
    }
}
