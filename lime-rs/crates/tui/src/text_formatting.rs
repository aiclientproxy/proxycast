//! Text formatting helpers copied from the Codex TUI owner.
//!
//! These helpers are presentation-only. They operate on terminal text and paths;
//! they do not alter canonical App Server messages or persisted thread state.

use unicode_segmentation::UnicodeSegmentation;

use crate::width::display_width;

#[allow(dead_code)]
pub(crate) fn capitalize_first(input: &str) -> String {
    let mut chars = input.chars();
    match chars.next() {
        Some(first) => {
            let mut capitalized = first.to_uppercase().collect::<String>();
            capitalized.push_str(chars.as_str());
            capitalized
        }
        None => String::new(),
    }
}

/// Formats valid JSON compactly before truncating it to a terminal budget.
#[allow(dead_code)]
pub(crate) fn format_and_truncate_tool_result(
    text: &str,
    max_lines: usize,
    line_width: usize,
) -> String {
    let max_graphemes = (max_lines * line_width).saturating_sub(max_lines);
    format_json_compact(text).as_deref().map_or_else(
        || truncate_text(text, max_graphemes),
        |json| truncate_text(json, max_graphemes),
    )
}

/// Formats JSON on one line while retaining spaces after separators.
pub(crate) fn format_json_compact(text: &str) -> Option<String> {
    let json = serde_json::from_str::<serde_json::Value>(text).ok()?;
    let pretty = serde_json::to_string_pretty(&json).unwrap_or_else(|_| json.to_string());

    let mut result = String::new();
    let mut chars = pretty.chars().peekable();
    let mut in_string = false;
    let mut escape_next = false;
    while let Some(ch) = chars.next() {
        match ch {
            '"' if !escape_next => {
                in_string = !in_string;
                result.push(ch);
            }
            '\\' if in_string => {
                escape_next = !escape_next;
                result.push(ch);
            }
            '\n' | '\r' if !in_string => {}
            ' ' | '\t' if !in_string => {
                if let Some(&next) = chars.peek() {
                    if let Some(last) = result.chars().last() {
                        if (last == ':' || last == ',') && !matches!(next, '}' | ']') {
                            result.push(' ');
                        }
                    }
                }
            }
            _ => {
                if escape_next && in_string {
                    escape_next = false;
                }
                result.push(ch);
            }
        }
    }
    Some(result)
}

/// Truncates by grapheme count so combining marks and emoji stay intact.
pub(crate) fn truncate_text(text: &str, max_graphemes: usize) -> String {
    let mut graphemes = text.grapheme_indices(true);
    let Some(byte_index) = graphemes.nth(max_graphemes).map(|(index, _)| index) else {
        return text.to_string();
    };
    if max_graphemes < 3 {
        return text[..byte_index].to_string();
    }
    let truncate_byte_index = text
        .grapheme_indices(true)
        .nth(max_graphemes - 3)
        .map(|(index, _)| index)
        .unwrap_or(byte_index);
    format!("{}...", &text[..truncate_byte_index])
}

/// Truncates a path while retaining useful leading and trailing segments.
pub(crate) fn center_truncate_path(path: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if display_width(path) <= max_width {
        return path.to_string();
    }

    let sep = std::path::MAIN_SEPARATOR;
    let has_leading_sep = path.starts_with(sep);
    let has_trailing_sep = path.ends_with(sep);
    let mut raw_segments: Vec<&str> = path.split(sep).collect();
    if has_leading_sep && !raw_segments.is_empty() && raw_segments[0].is_empty() {
        raw_segments.remove(0);
    }
    if has_trailing_sep
        && !raw_segments.is_empty()
        && raw_segments.last().is_some_and(|last| last.is_empty())
    {
        raw_segments.pop();
    }
    if raw_segments.is_empty() {
        if has_leading_sep {
            let root = sep.to_string();
            if display_width(&root) <= max_width {
                return root;
            }
        }
        return "…".to_string();
    }

    struct Segment<'a> {
        original: &'a str,
        text: String,
        truncatable: bool,
        is_suffix: bool,
    }

    let assemble = |leading: bool, segments: &[Segment<'_>]| -> String {
        let mut result = String::new();
        if leading {
            result.push(sep);
        }
        for segment in segments {
            if !result.is_empty() && !result.ends_with(sep) {
                result.push(sep);
            }
            result.push_str(&segment.text);
        }
        result
    };
    let front_truncate = |original: &str, allowed_width: usize| -> String {
        if allowed_width == 0 {
            return String::new();
        }
        if display_width(original) <= allowed_width {
            return original.to_string();
        }
        if allowed_width == 1 {
            return "…".to_string();
        }
        let mut kept = Vec::new();
        let mut used_width = 1;
        for grapheme in original.graphemes(true).rev() {
            let grapheme_width = display_width(grapheme);
            if used_width + grapheme_width > allowed_width {
                break;
            }
            used_width += grapheme_width;
            kept.push(grapheme);
        }
        kept.reverse();
        format!("…{}", kept.concat())
    };

    let segment_count = raw_segments.len();
    let mut combos = Vec::new();
    for left in 1..=segment_count {
        let min_right = if left == segment_count { 0 } else { 1 };
        for right in min_right..=(segment_count - left) {
            combos.push((left, right));
        }
    }
    let desired_suffix = if segment_count > 1 {
        std::cmp::min(2, segment_count - 1)
    } else {
        0
    };
    let (mut prioritized, mut fallback): (Vec<_>, Vec<_>) = combos
        .into_iter()
        .partition(|(_, right)| *right >= desired_suffix);
    let sort_combos = |items: &mut Vec<(usize, usize)>| {
        items.sort_by(|(left_a, right_a), (left_b, right_b)| {
            left_b
                .cmp(left_a)
                .then_with(|| right_b.cmp(right_a))
                .then_with(|| (left_b + right_b).cmp(&(left_a + right_a)))
        });
    };
    sort_combos(&mut prioritized);
    sort_combos(&mut fallback);

    let fit_segments = |segments: &mut Vec<Segment<'_>>, allow_front_truncate: bool| loop {
        let candidate = assemble(has_leading_sep, segments);
        let width = display_width(&candidate);
        if width <= max_width {
            return Some(candidate);
        }
        if !allow_front_truncate {
            return None;
        }
        let mut indices = Vec::new();
        for (index, segment) in segments.iter().enumerate().rev() {
            if segment.truncatable && segment.is_suffix {
                indices.push(index);
            }
        }
        for (index, segment) in segments.iter().enumerate().rev() {
            if segment.truncatable && !segment.is_suffix {
                indices.push(index);
            }
        }
        let mut changed = false;
        for index in indices {
            let original_width = display_width(segments[index].original);
            if original_width <= max_width && segment_count > 2 {
                continue;
            }
            let segment_width = display_width(&segments[index].text);
            let other_width = width.saturating_sub(segment_width);
            let allowed_width = max_width.saturating_sub(other_width).max(1);
            let shortened = front_truncate(segments[index].original, allowed_width);
            if shortened != segments[index].text {
                segments[index].text = shortened;
                changed = true;
                break;
            }
        }
        if !changed {
            return None;
        }
    };

    for (left_count, right_count) in prioritized.into_iter().chain(fallback) {
        let mut segments = raw_segments[..left_count]
            .iter()
            .map(|segment| Segment {
                original: segment,
                text: (*segment).to_string(),
                truncatable: true,
                is_suffix: false,
            })
            .collect::<Vec<_>>();
        let need_ellipsis = left_count + right_count < segment_count;
        if need_ellipsis {
            segments.push(Segment {
                original: "…",
                text: "…".to_string(),
                truncatable: false,
                is_suffix: false,
            });
        }
        if right_count > 0 {
            segments.extend(
                raw_segments[segment_count - right_count..]
                    .iter()
                    .map(|segment| Segment {
                        original: segment,
                        text: (*segment).to_string(),
                        truncatable: true,
                        is_suffix: true,
                    }),
            );
        }
        if let Some(candidate) = fit_segments(&mut segments, need_ellipsis || segment_count <= 2) {
            return candidate;
        }
    }
    front_truncate(path, max_width)
}

/// Joins labels with natural English punctuation.
#[allow(dead_code)]
pub(crate) fn proper_join<T: AsRef<str>>(items: &[T]) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].as_ref().to_string(),
        2 => format!("{} and {}", items[0].as_ref(), items[1].as_ref()),
        _ => {
            let prefix = items[..items.len() - 1]
                .iter()
                .map(AsRef::as_ref)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{prefix} and {}", items[items.len() - 1].as_ref())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capitalize_first() {
        assert_eq!(capitalize_first("hello"), "Hello");
        assert_eq!(capitalize_first(""), "");
    }

    #[test]
    fn test_format_and_truncate_tool_result() {
        let result = format_and_truncate_tool_result(r#"{"ok":true,"value":"done"}"#, 2, 20);
        assert_eq!(result, r#"{"ok": true, "value": "done"}"#);
    }

    #[test]
    fn test_truncate_text() {
        assert_eq!(truncate_text("Hello, world!", 8), "Hello...");
    }

    #[test]
    fn test_truncate_empty_string() {
        assert_eq!(truncate_text("", 5), "");
    }

    #[test]
    fn test_truncate_max_graphemes_zero() {
        assert_eq!(truncate_text("Hello", 0), "");
    }

    #[test]
    fn test_truncate_max_graphemes_one() {
        assert_eq!(truncate_text("Hello", 1), "H");
    }

    #[test]
    fn test_truncate_max_graphemes_two() {
        assert_eq!(truncate_text("Hello", 2), "He");
    }

    #[test]
    fn test_truncate_max_graphemes_three_boundary() {
        assert_eq!(truncate_text("Hello", 3), "...");
    }

    #[test]
    fn test_truncate_text_shorter_than_limit() {
        assert_eq!(truncate_text("Hi", 10), "Hi");
    }

    #[test]
    fn test_truncate_text_exact_length() {
        assert_eq!(truncate_text("Hello", 5), "Hello");
    }

    #[test]
    fn test_truncate_emoji() {
        assert_eq!(truncate_text("👋🌍🚀✨💫", 3), "...");
        assert_eq!(truncate_text("👋🌍🚀✨💫", 4), "👋...");
    }

    #[test]
    fn test_truncate_unicode_combining_characters() {
        assert_eq!(truncate_text("é́ñ̃", 2), "é́ñ̃");
    }

    #[test]
    fn test_truncate_very_long_text() {
        let text = "a".repeat(1000);
        assert_eq!(truncate_text(&text, 10), "aaaaaaa...");
    }

    #[test]
    fn test_format_json_compact_simple_object() {
        assert_eq!(
            format_json_compact(r#"{ "name": "John", "age": 30 }"#).unwrap(),
            r#"{"name": "John", "age": 30}"#
        );
    }

    #[test]
    fn test_format_json_compact_nested_object() {
        let result =
            format_json_compact(r#"{ "user": { "name": "John", "details": { "age": 30 } } }"#)
                .unwrap();
        assert_eq!(
            result,
            r#"{"user": {"name": "John", "details": {"age": 30}}}"#
        );
    }

    #[test]
    fn test_center_truncate_doesnt_truncate_short_path() {
        let sep = std::path::MAIN_SEPARATOR;
        let path = format!("{sep}Users{sep}codex{sep}Public");
        assert_eq!(center_truncate_path(&path, 40), path);
    }

    #[test]
    fn test_center_truncate_truncates_long_path() {
        let sep = std::path::MAIN_SEPARATOR;
        let path = format!("~{sep}hello{sep}the{sep}fox{sep}is{sep}very{sep}fast");
        assert_eq!(
            center_truncate_path(&path, 24),
            format!("~{sep}hello{sep}the{sep}…{sep}very{sep}fast")
        );
    }

    #[test]
    fn test_center_truncate_handles_long_segment() {
        let sep = std::path::MAIN_SEPARATOR;
        let path = format!("~{sep}supercalifragilisticexpialidocious");
        assert_eq!(
            center_truncate_path(&path, 18),
            format!("~{sep}…cexpialidocious")
        );
    }

    #[test]
    fn test_format_json_compact_invalid_json() {
        assert!(format_json_compact(r#"{"invalid": json syntax}"#).is_none());
    }

    #[test]
    fn test_format_json_compact_primitive_values() {
        assert_eq!(format_json_compact("42").unwrap(), "42");
        assert_eq!(format_json_compact("true").unwrap(), "true");
        assert_eq!(format_json_compact("null").unwrap(), "null");
    }

    #[test]
    fn test_proper_join() {
        let empty: Vec<String> = vec![];
        assert_eq!(proper_join(&empty), "");
        assert_eq!(proper_join(&["apple"]), "apple");
        assert_eq!(proper_join(&["apple", "banana"]), "apple and banana");
        assert_eq!(
            proper_join(&["apple", "banana", "cherry"]),
            "apple, banana and cherry"
        );
    }
}
