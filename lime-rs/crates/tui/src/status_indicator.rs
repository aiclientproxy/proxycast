use std::time::Duration;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::Paragraph;

use crate::line_truncation::truncate_line_with_ellipsis_if_overflow;
use crate::locale::Locale;

pub(crate) fn render(frame: &mut Frame<'_>, area: Rect, locale: Locale, elapsed: Duration) {
    if area.is_empty() {
        return;
    }

    frame.render_widget(
        Paragraph::new(status_line(locale, elapsed, area.width)),
        area,
    );
}

fn status_line(locale: Locale, elapsed: Duration, width: u16) -> Line<'static> {
    truncate_line_with_ellipsis_if_overflow(
        Line::styled(
            format!(
                "• {} ({} • {})",
                locale.working_label(),
                fmt_elapsed_compact(elapsed.as_secs()),
                locale.interrupt_hint()
            ),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
        usize::from(width),
    )
}

fn fmt_elapsed_compact(elapsed_secs: u64) -> String {
    crate::status_indicator_widget::fmt_elapsed_compact(elapsed_secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::line_truncation::line_width;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    #[test]
    fn elapsed_time_matches_codex_compact_format() {
        assert_eq!(fmt_elapsed_compact(0), "0s");
        assert_eq!(fmt_elapsed_compact(59), "59s");
        assert_eq!(fmt_elapsed_compact(60), "1m 00s");
        assert_eq!(fmt_elapsed_compact(3_599), "59m 59s");
        assert_eq!(fmt_elapsed_compact(3_600), "1h 00m 00s");
        assert_eq!(fmt_elapsed_compact(7_389), "2h 03m 09s");
    }

    #[test]
    fn working_status_is_localized_and_width_bounded() {
        let cases = [
            (Locale::ZhCn, "处理中", "Esc 中断"),
            (Locale::ZhTw, "處理中", "Esc 中斷"),
            (Locale::EnUs, "Working", "esc to interrupt"),
            (Locale::JaJp, "処理中", "Esc で中断"),
            (Locale::KoKr, "작업 중", "Esc로 중단"),
        ];

        for (locale, working, interrupt) in cases {
            let line = status_line(locale, Duration::from_secs(63), 80);
            let text = line_text(&line);
            assert!(text.contains(working), "{locale:?}: {text}");
            assert!(text.contains("1m 03s"), "{locale:?}: {text}");
            assert!(text.contains(interrupt), "{locale:?}: {text}");

            let narrow = status_line(locale, Duration::ZERO, 18);
            assert!(line_width(&narrow) <= 18, "{locale:?}: {narrow:?}");
            assert!(line_text(&narrow).ends_with('…'), "{locale:?}: {narrow:?}");
        }
    }

    #[test]
    fn fixed_elapsed_test_backend_matches_wide_and_narrow_layouts() {
        let mut wide = Terminal::new(TestBackend::new(48, 1)).expect("wide terminal");
        wide.draw(|frame| render(frame, frame.area(), Locale::EnUs, Duration::from_secs(63)))
            .expect("wide draw");
        let wide_text = wide
            .backend()
            .buffer()
            .content
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect::<String>();
        assert!(
            wide_text.starts_with("• Working (1m 03s • esc to interrupt)"),
            "{wide_text}"
        );

        let mut narrow = Terminal::new(TestBackend::new(18, 1)).expect("narrow terminal");
        narrow
            .draw(|frame| render(frame, frame.area(), Locale::EnUs, Duration::ZERO))
            .expect("narrow draw");
        let narrow_text = narrow
            .backend()
            .buffer()
            .content
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect::<String>();
        assert!(
            narrow_text.starts_with("• Working (0s • e…"),
            "{narrow_text}"
        );
    }
}
