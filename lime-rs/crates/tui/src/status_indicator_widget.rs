use std::time::Duration;

use ratatui::Frame;
use ratatui::layout::Rect;

use crate::locale::Locale;
use crate::status_indicator;

pub(crate) fn render(frame: &mut Frame<'_>, area: Rect, locale: Locale, elapsed: Duration) {
    status_indicator::render(frame, area, locale, elapsed);
}

pub fn fmt_elapsed_compact(elapsed_secs: u64) -> String {
    if elapsed_secs < 60 {
        return format!("{elapsed_secs}s");
    }
    if elapsed_secs < 3_600 {
        return format!("{}m {:02}s", elapsed_secs / 60, elapsed_secs % 60);
    }
    format!(
        "{}h {:02}m {:02}s",
        elapsed_secs / 3_600,
        (elapsed_secs % 3_600) / 60,
        elapsed_secs % 60
    )
}
