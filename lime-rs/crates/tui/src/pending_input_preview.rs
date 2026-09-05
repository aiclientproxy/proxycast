use app_server_protocol::protocol::v2::{QueuedSubmission, UserInput};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::line_truncation::truncate_line_with_ellipsis_if_overflow;
use crate::locale::Locale;
use crate::terminal_hyperlinks::{HyperlinkLine, wrap_hyperlink_line};

const MAX_VISIBLE_SUBMISSIONS: usize = 2;
const MAX_LINES_PER_SUBMISSION: usize = 2;

pub(crate) fn desired_height(submissions: &[QueuedSubmission], width: u16, locale: Locale) -> u16 {
    u16::try_from(preview_lines(submissions, width, locale).len()).unwrap_or(u16::MAX)
}

pub(crate) fn render(
    frame: &mut Frame<'_>,
    area: Rect,
    submissions: &[QueuedSubmission],
    locale: Locale,
) {
    if area.is_empty() {
        return;
    }
    frame.render_widget(
        Paragraph::new(preview_lines(submissions, area.width, locale)),
        area,
    );
}

fn preview_lines(
    submissions: &[QueuedSubmission],
    width: u16,
    locale: Locale,
) -> Vec<Line<'static>> {
    if submissions.is_empty() || width < 4 {
        return Vec::new();
    }

    let mut lines = vec![truncate_line_with_ellipsis_if_overflow(
        Line::styled(
            format!("• {} ({})", locale.status("queued"), submissions.len()),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
        usize::from(width),
    )];
    for submission in submissions.iter().take(MAX_VISIBLE_SUBMISSIONS) {
        lines.extend(submission_preview_lines(submission, width, locale));
    }
    if submissions.len() > MAX_VISIBLE_SUBMISSIONS {
        lines.push(truncate_line_with_ellipsis_if_overflow(
            Line::styled(
                format!("   … +{}", submissions.len() - MAX_VISIBLE_SUBMISSIONS),
                Style::default().fg(Color::DarkGray),
            ),
            usize::from(width),
        ));
    }
    if submissions.last().is_some_and(can_restore_submission) {
        lines.push(truncate_line_with_ellipsis_if_overflow(
            Line::styled(
                format!("   {}", locale.edit_queued_input_hint()),
                Style::default().fg(Color::DarkGray),
            ),
            usize::from(width),
        ));
    }
    lines
}

pub(crate) fn can_restore_submission(submission: &QueuedSubmission) -> bool {
    let mut text_count = 0usize;
    !submission.input.is_empty()
        && submission.input.iter().all(|input| match input {
            UserInput::Text { text_elements, .. } => {
                text_count += 1;
                text_count <= 1 && text_elements.is_empty()
            }
            UserInput::LocalImage { detail, .. } => detail.is_none(),
            UserInput::Image { .. } | UserInput::Skill { .. } | UserInput::Mention { .. } => false,
        })
}

fn submission_preview_lines(
    submission: &QueuedSubmission,
    width: u16,
    locale: Locale,
) -> Vec<Line<'static>> {
    let content_width = usize::from(width.saturating_sub(4).max(1));
    let summary = submission_summary(submission, locale);
    let mut wrapped = summary
        .lines()
        .flat_map(|line| {
            wrap_hyperlink_line(
                &HyperlinkLine::new(Line::styled(
                    line.to_string(),
                    Style::default().add_modifier(Modifier::ITALIC),
                )),
                content_width,
            )
        })
        .map(|line| line.line)
        .collect::<Vec<_>>();
    if wrapped.is_empty() {
        wrapped.push(Line::styled(
            locale.not_set_label(),
            Style::default().add_modifier(Modifier::ITALIC),
        ));
    }

    let overflow = wrapped.len() > MAX_LINES_PER_SUBMISSION;
    wrapped.truncate(MAX_LINES_PER_SUBMISSION);
    let mut lines = wrapped
        .into_iter()
        .enumerate()
        .map(|(index, line)| {
            let mut spans = vec![Span::styled(
                if index == 0 { " ↳ " } else { "   " },
                Style::default().fg(Color::DarkGray),
            )];
            spans.extend(line.spans);
            Line::from(spans)
        })
        .collect::<Vec<_>>();
    if overflow {
        lines.push(Line::styled("   …", Style::default().fg(Color::DarkGray)));
    }
    lines
}

fn submission_summary(submission: &QueuedSubmission, locale: Locale) -> String {
    let mut parts = Vec::new();
    let mut image_count = 0usize;
    for input in &submission.input {
        match input {
            UserInput::Text { text, .. } if !text.trim().is_empty() => {
                parts.push(text.trim().to_string());
            }
            UserInput::Image { .. } | UserInput::LocalImage { .. } => image_count += 1,
            UserInput::Skill { name, .. } => parts.push(format!("${name}")),
            UserInput::Mention { name, .. } => parts.push(format!("@{name}")),
            UserInput::Text { .. } => {}
        }
    }
    if image_count > 0 {
        parts.push(format!("[{} ×{image_count}]", locale.image_label()));
    }
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::line_truncation::line_width;

    fn submission(id: &str, input: Vec<UserInput>) -> QueuedSubmission {
        QueuedSubmission {
            id: id.to_string(),
            input,
            client_user_message_id: format!("client-{id}"),
        }
    }

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    #[test]
    fn preview_uses_canonical_multimodal_queue_and_caps_visible_items() {
        let submissions = vec![
            submission(
                "queue-1",
                vec![
                    UserInput::LocalImage {
                        detail: None,
                        path: "/tmp/one.png".to_string(),
                    },
                    UserInput::Text {
                        text: "first follow-up".to_string(),
                        text_elements: Vec::new(),
                    },
                ],
            ),
            submission(
                "queue-2",
                vec![UserInput::Text {
                    text: "second follow-up".to_string(),
                    text_elements: Vec::new(),
                }],
            ),
            submission(
                "queue-3",
                vec![UserInput::Text {
                    text: "third follow-up".to_string(),
                    text_elements: Vec::new(),
                }],
            ),
        ];

        let text = preview_lines(&submissions, 80, Locale::EnUs)
            .iter()
            .map(line_text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("queued (3)"), "{text}");
        assert!(text.contains("first follow-up"), "{text}");
        assert!(text.contains("[image ×1]"), "{text}");
        assert!(text.contains("second follow-up"), "{text}");
        assert!(!text.contains("third follow-up"), "{text}");
        assert!(text.contains("… +1"), "{text}");
    }

    #[test]
    fn narrow_preview_wraps_and_caps_each_submission() {
        let submissions = vec![submission(
            "queue-1",
            vec![UserInput::Text {
                text: "one two three four five six seven eight nine".to_string(),
                text_elements: Vec::new(),
            }],
        )];

        let lines = preview_lines(&submissions, 12, Locale::EnUs);

        assert_eq!(lines.len(), 5);
        assert_eq!(line_text(&lines[3]), "   …");
        assert!(line_text(lines.last().expect("edit hint")).contains("Alt+Up"));
        assert!(lines.iter().all(|line| line_width(line) <= 12));
    }

    #[test]
    fn edit_hint_requires_a_lossless_composer_projection() {
        let local = submission(
            "local",
            vec![UserInput::LocalImage {
                detail: None,
                path: "/tmp/local.png".to_string(),
            }],
        );
        let remote = submission(
            "remote",
            vec![UserInput::Image {
                detail: None,
                url: "https://example.test/image.png".to_string(),
            }],
        );
        let structured = submission(
            "structured",
            vec![
                UserInput::Text {
                    text: "first segment".to_string(),
                    text_elements: Vec::new(),
                },
                UserInput::Text {
                    text: "second segment".to_string(),
                    text_elements: Vec::new(),
                },
            ],
        );

        assert!(can_restore_submission(&local));
        assert!(!can_restore_submission(&remote));
        assert!(!can_restore_submission(&structured));
        assert!(
            !preview_lines(&[remote], 80, Locale::EnUs)
                .iter()
                .map(line_text)
                .collect::<String>()
                .contains("Alt+Up")
        );
    }

    #[test]
    fn queue_header_and_image_label_cover_all_product_locales() {
        let cases = [
            (Locale::ZhCn, "已排队", "图片"),
            (Locale::ZhTw, "已排隊", "圖片"),
            (Locale::EnUs, "queued", "image"),
            (Locale::JaJp, "キューに追加済み", "画像"),
            (Locale::KoKr, "대기열에 추가됨", "이미지"),
        ];
        for (locale, queued, image) in cases {
            let lines = preview_lines(
                &[submission(
                    "queue-1",
                    vec![UserInput::Image {
                        detail: None,
                        url: "https://example.test/image.png".to_string(),
                    }],
                )],
                80,
                locale,
            );
            let text = lines.iter().map(line_text).collect::<Vec<_>>().join("\n");
            assert!(text.contains(queued), "{locale:?}: {text}");
            assert!(text.contains(image), "{locale:?}: {text}");
        }
    }
}
