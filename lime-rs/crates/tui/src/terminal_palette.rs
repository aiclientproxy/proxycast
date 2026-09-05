//! Terminal color capability and palette helpers.
//!
//! The public names mirror Codex TUI. Lime does not have Codex's startup
//! terminal probe, so default foreground/background queries intentionally
//! return `None` until a current probe owner exists. Color quantization itself
//! remains deterministic and safe for snapshot tests.

use ratatui::style::Color;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StdoutColorLevel {
    TrueColor,
    Ansi256,
    Ansi16,
    Unknown,
}

#[allow(dead_code)]
pub(crate) fn stdout_color_level() -> StdoutColorLevel {
    if std::env::var_os("NO_COLOR").is_some() {
        return StdoutColorLevel::Ansi16;
    }

    let color_term = std::env::var("COLORTERM")
        .unwrap_or_default()
        .to_ascii_lowercase();
    if matches!(color_term.as_str(), "truecolor" | "24bit") {
        return StdoutColorLevel::TrueColor;
    }

    let term = std::env::var("TERM")
        .unwrap_or_default()
        .to_ascii_lowercase();
    if term.contains("direct") || term.contains("truecolor") {
        StdoutColorLevel::TrueColor
    } else if term.contains("256color") {
        StdoutColorLevel::Ansi256
    } else if term.is_empty() {
        StdoutColorLevel::Unknown
    } else {
        StdoutColorLevel::Ansi16
    }
}

#[allow(dead_code)]
pub(crate) fn effective_stdout_color_level() -> StdoutColorLevel {
    stdout_color_level()
}

#[allow(clippy::disallowed_methods)]
pub(crate) fn rgb_color((red, green, blue): (u8, u8, u8)) -> Color {
    Color::Rgb(red, green, blue)
}

#[allow(clippy::disallowed_methods)]
pub(crate) fn indexed_color(index: u8) -> Color {
    Color::Indexed(index)
}

#[allow(dead_code)]
pub(crate) fn best_color(target: (u8, u8, u8)) -> Color {
    best_color_for_level(target, stdout_color_level())
}

pub(crate) fn best_color_for_level(target: (u8, u8, u8), level: StdoutColorLevel) -> Color {
    best_color_for_color_level(target, level)
}

#[allow(dead_code)]
fn best_color_for_color_level(target: (u8, u8, u8), level: StdoutColorLevel) -> Color {
    match level {
        StdoutColorLevel::TrueColor => rgb_color(target),
        StdoutColorLevel::Ansi256 => xterm_fixed_colors()
            .min_by_key(|(_, color)| color_distance(*color, target))
            .map_or_else(Color::default, |(index, _)| indexed_color(index)),
        StdoutColorLevel::Ansi16 | StdoutColorLevel::Unknown => Color::default(),
    }
}

/// Terminal default colors are not available without a current probe owner.
#[allow(dead_code)]
pub(crate) fn default_colors() -> Option<DefaultColors> {
    None
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DefaultColors {
    pub(crate) fg: (u8, u8, u8),
    pub(crate) bg: (u8, u8, u8),
}

#[allow(dead_code)]
pub(crate) fn default_fg() -> Option<(u8, u8, u8)> {
    default_colors().map(|colors| colors.fg)
}

#[allow(dead_code)]
pub(crate) fn default_bg() -> Option<(u8, u8, u8)> {
    default_colors().map(|colors| colors.bg)
}

#[allow(dead_code)]
pub(crate) fn with_test_default_colors<T>(render: impl FnOnce() -> T) -> T {
    render()
}

#[allow(dead_code)]
pub(crate) fn set_default_colors_from_startup_probe(_colors: Option<DefaultColors>) {}

fn color_distance(left: (u8, u8, u8), right: (u8, u8, u8)) -> u32 {
    let red = i32::from(left.0) - i32::from(right.0);
    let green = i32::from(left.1) - i32::from(right.1);
    let blue = i32::from(left.2) - i32::from(right.2);
    (red * red + green * green + blue * blue) as u32
}

fn xterm_fixed_colors() -> impl Iterator<Item = (u8, (u8, u8, u8))> {
    let cube = (0..216).map(|offset| {
        let red = offset / 36;
        let green = (offset / 6) % 6;
        let blue = offset % 6;
        let level = |value: u8| if value == 0 { 0 } else { 55 + value * 40 };
        (16 + offset, (level(red), level(green), level(blue)))
    });
    let grayscale = (0..24).map(|offset| {
        let value = 8 + offset * 10;
        (232 + offset, (value, value, value))
    });
    cube.chain(grayscale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_color_uses_truecolor_without_quantization() {
        assert_eq!(
            best_color_for_level((12, 34, 56), StdoutColorLevel::TrueColor),
            Color::Rgb(12, 34, 56)
        );
    }

    #[test]
    fn best_color_resets_for_ansi16() {
        assert_eq!(
            best_color_for_level((12, 34, 56), StdoutColorLevel::Ansi16),
            Color::Reset
        );
    }

    #[test]
    fn ansi256_palette_contains_cube_and_grayscale() {
        let colors = xterm_fixed_colors().collect::<Vec<_>>();
        assert_eq!(colors.len(), 240);
        assert_eq!(colors.first(), Some(&(16, (0, 0, 0))));
        assert_eq!(colors.last(), Some(&(255, (238, 238, 238))));
    }

    #[test]
    fn nearest_ansi256_color_is_indexed() {
        assert!(matches!(
            best_color_for_level((255, 0, 0), StdoutColorLevel::Ansi256),
            Color::Indexed(_)
        ));
    }

    #[test]
    fn default_color_queries_fail_closed_without_probe() {
        assert_eq!(default_colors(), None);
        assert_eq!(default_fg(), None);
        assert_eq!(default_bg(), None);
    }
}
