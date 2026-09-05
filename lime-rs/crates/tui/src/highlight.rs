//! Syntax highlighting for terminal code blocks and diffs.
//!
//! This is the terminal-only subset of Codex's highlighter. It deliberately
//! uses the ANSI theme so colors follow the user's terminal palette. Inputs
//! outside the bounded limits fall back to plain text.

use std::sync::OnceLock;

use ratatui::style::{Color as RtColor, Modifier, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::{Color as SyntectColor, FontStyle, Style as SyntectStyle, Theme};
use syntect::parsing::{SyntaxReference, SyntaxSet};
use syntect::util::LinesWithEndings;
use two_face::theme::EmbeddedThemeName;

static SYNTAX_SET: OnceLock<SyntaxSet> = OnceLock::new();
static THEME: OnceLock<Theme> = OnceLock::new();

const ANSI_ALPHA_INDEX: u8 = 0x00;
const ANSI_ALPHA_DEFAULT: u8 = 0x01;
const OPAQUE_ALPHA: u8 = 0xff;
const MAX_HIGHLIGHT_BYTES: usize = 512 * 1024;
const MAX_HIGHLIGHT_LINES: usize = 10_000;
const MAX_HIGHLIGHT_LINE_BYTES: usize = 4 * 1024;

fn syntax_set() -> &'static SyntaxSet {
    SYNTAX_SET.get_or_init(two_face::syntax::extra_newlines)
}

fn theme() -> &'static Theme {
    THEME.get_or_init(|| {
        two_face::theme::extra()
            .get(EmbeddedThemeName::Ansi)
            .clone()
    })
}

#[allow(clippy::disallowed_methods)]
fn ansi_palette_color(index: u8) -> RtColor {
    match index {
        0x00 => RtColor::Black,
        0x01 => RtColor::Red,
        0x02 => RtColor::Green,
        0x03 => RtColor::Yellow,
        0x04 => RtColor::Blue,
        0x05 => RtColor::Magenta,
        0x06 => RtColor::Cyan,
        0x07 => RtColor::Gray,
        value => crate::terminal_palette::indexed_color(value),
    }
}

#[allow(clippy::disallowed_methods)]
fn convert_syntect_color(color: SyntectColor) -> Option<RtColor> {
    match color.a {
        ANSI_ALPHA_INDEX => Some(ansi_palette_color(color.r)),
        ANSI_ALPHA_DEFAULT => None,
        OPAQUE_ALPHA => Some(RtColor::Rgb(color.r, color.g, color.b)),
        _ => Some(RtColor::Rgb(color.r, color.g, color.b)),
    }
}

fn convert_style(syntect: SyntectStyle) -> Style {
    let mut style = Style::default();
    if let Some(foreground) = convert_syntect_color(syntect.foreground) {
        style = style.fg(foreground);
    }
    if syntect.font_style.contains(FontStyle::BOLD) {
        style = style.add_modifier(Modifier::BOLD);
    }
    style
}

fn find_syntax(language: &str) -> Option<&'static SyntaxReference> {
    let syntaxes = syntax_set();
    let normalized = language.to_ascii_lowercase();
    let patched = match normalized.as_str() {
        "csharp" | "c-sharp" => "c#",
        "cu" | "cuh" | "cppm" | "cxxm" | "ixx" => "cpp",
        "golang" => "go",
        "python3" => "python",
        "shell" => "bash",
        _ => language,
    };

    syntaxes
        .find_syntax_by_token(patched)
        .or_else(|| syntaxes.find_syntax_by_name(patched))
        .or_else(|| {
            let lower = patched.to_ascii_lowercase();
            syntaxes
                .syntaxes()
                .iter()
                .find(|syntax| syntax.name.to_ascii_lowercase() == lower)
        })
        .or_else(|| syntaxes.find_syntax_by_extension(language))
}

pub(crate) fn exceeds_highlight_limits(total_bytes: usize, total_lines: usize) -> bool {
    total_bytes > MAX_HIGHLIGHT_BYTES || total_lines > MAX_HIGHLIGHT_LINES
}

fn highlight_to_line_spans(code: &str, language: &str) -> Option<Vec<Vec<Span<'static>>>> {
    if code.is_empty()
        || exceeds_highlight_limits(code.len(), code.lines().count())
        || code
            .lines()
            .any(|line| line.len() > MAX_HIGHLIGHT_LINE_BYTES)
    {
        return None;
    }

    let syntax = find_syntax(language)?;
    let mut highlighter = HighlightLines::new(syntax, theme());
    LinesWithEndings::from(code)
        .map(|line| {
            let ranges = highlighter.highlight_line(line, syntax_set()).ok()?;
            Some(highlighted_line_spans(ranges))
        })
        .collect()
}

fn highlighted_line_spans(ranges: Vec<(SyntectStyle, &str)>) -> Vec<Span<'static>> {
    let mut spans = ranges
        .into_iter()
        .filter_map(|(style, text)| {
            let text = text.trim_end_matches(['\n', '\r']);
            (!text.is_empty()).then(|| Span::styled(text.to_string(), convert_style(style)))
        })
        .collect::<Vec<_>>();
    if spans.is_empty() {
        spans.push(Span::raw(String::new()));
    }
    spans
}

pub(crate) struct CodeLineHighlighter {
    highlighter: Option<HighlightLines<'static>>,
}

impl CodeLineHighlighter {
    pub(crate) fn new(language: &str) -> Option<Self> {
        Some(Self {
            highlighter: Some(HighlightLines::new(find_syntax(language)?, theme())),
        })
    }

    pub(crate) fn highlight_line(&mut self, line: &str) -> Option<Vec<Span<'static>>> {
        if line.len() > MAX_HIGHLIGHT_LINE_BYTES {
            self.highlighter = None;
            return None;
        }
        let source = format!("{line}\n");
        let ranges = match self
            .highlighter
            .as_mut()?
            .highlight_line(&source, syntax_set())
        {
            Ok(ranges) => ranges,
            Err(_) => {
                self.highlighter = None;
                return None;
            }
        };
        Some(highlighted_line_spans(ranges))
    }
}

pub(crate) fn highlight_code_to_lines(code: &str, language: &str) -> Vec<Line<'static>> {
    highlight_to_line_spans(code, language)
        .map(|lines| lines.into_iter().map(Line::from).collect())
        .unwrap_or_else(|| {
            let mut lines = code
                .lines()
                .map(|line| Line::from(line.to_string()))
                .collect::<Vec<_>>();
            if lines.is_empty() {
                lines.push(Line::default());
            }
            lines
        })
}

#[cfg(test)]
pub(crate) fn highlight_code_to_styled_spans(
    code: &str,
    language: &str,
) -> Option<Vec<Vec<Span<'static>>>> {
    highlight_to_line_spans(code, language)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reconstructed(lines: &[Line<'static>]) -> String {
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
    fn rust_keywords_receive_terminal_palette_styles() {
        let lines = highlight_code_to_lines("fn main() {}", "rust");
        assert_eq!(reconstructed(&lines), "fn main() {}");
        let keyword = lines[0]
            .spans
            .iter()
            .find(|span| span.content.as_ref() == "fn")
            .expect("Rust keyword should have its own span");
        assert!(keyword.style.fg.is_some() || !keyword.style.add_modifier.is_empty());
        assert!(!matches!(keyword.style.fg, Some(RtColor::Rgb(..))));
    }

    #[test]
    fn unknown_language_falls_back_to_plain_text() {
        let lines = highlight_code_to_lines("some random text", "xyzlang");
        assert_eq!(reconstructed(&lines), "some random text");
        assert!(
            lines
                .iter()
                .flat_map(|line| &line.spans)
                .all(|span| span.style == Style::default())
        );
    }

    #[test]
    fn trailing_newline_does_not_create_a_phantom_line() {
        let lines = highlight_code_to_lines("hello world\n", "xyzlang");
        assert_eq!(lines.len(), 1);
        assert_eq!(reconstructed(&lines), "hello world");
    }

    #[test]
    fn crlf_does_not_leak_carriage_returns() {
        let lines = highlight_code_to_lines("fn main() {\r\n}\r\n", "rust");
        assert!(
            lines
                .iter()
                .flat_map(|line| &line.spans)
                .all(|span| !span.content.contains('\r'))
        );
    }

    #[test]
    fn multiline_python_preserves_content() {
        let code = "def hello():\n    print(\"hi\")\n    return 42";
        let lines = highlight_code_to_lines(code, "python");
        assert_eq!(lines.len(), 3);
        assert_eq!(reconstructed(&lines), code);
    }

    #[test]
    fn aliases_resolve_to_extended_syntaxes() {
        for language in ["csharp", "cuh", "golang", "python3", "shell"] {
            assert!(find_syntax(language).is_some(), "missing alias {language}");
        }
    }

    #[test]
    fn oversized_input_falls_back_without_highlighting() {
        let code = "x".repeat(MAX_HIGHLIGHT_BYTES + 1);
        assert!(highlight_code_to_styled_spans(&code, "rust").is_none());
    }

    #[test]
    fn too_many_lines_fall_back_without_highlighting() {
        let code = "let x = 1;\n".repeat(MAX_HIGHLIGHT_LINES + 1);
        assert!(highlight_code_to_styled_spans(&code, "rust").is_none());
    }

    #[test]
    fn long_single_line_falls_back_and_preserves_text() {
        let code = "x".repeat(MAX_HIGHLIGHT_LINE_BYTES + 1);
        assert!(highlight_code_to_styled_spans(&code, "rust").is_none());
        assert_eq!(
            highlight_code_to_lines(&code, "rust"),
            vec![Line::from(code)]
        );
    }

    #[test]
    fn ansi_alpha_markers_map_to_terminal_colors() {
        assert_eq!(
            convert_syntect_color(SyntectColor {
                r: 0x02,
                g: 0,
                b: 0,
                a: ANSI_ALPHA_INDEX,
            }),
            Some(RtColor::Green)
        );
        assert_eq!(
            convert_syntect_color(SyntectColor {
                r: 0,
                g: 0,
                b: 0,
                a: ANSI_ALPHA_DEFAULT,
            }),
            None
        );
    }
}
