use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::app::App;
use crate::bottom_pane;
use crate::command_popup;
use crate::entry;
use crate::line_truncation::truncate_line_with_ellipsis_if_overflow;
use crate::model_picker;
use crate::pending_input_preview;
use crate::status_indicator_widget;
use crate::terminal_hyperlinks::HyperlinkParagraph;
use crate::width::{display_width, usable_content_width_u16};
use std::time::Instant;

pub(crate) fn render(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();
    if let Some(pager) = app.pager_overlay.as_ref() {
        let transcript_lines = if pager.is_transcript() {
            projected_transcript_lines(app, area.width, true)
        } else {
            Vec::new()
        };
        pager.render(frame, area, app.locale, &transcript_lines);
        return;
    }
    let chunks = screen_chunks(area, app);

    render_header(frame, chunks[0], app);
    render_transcript(frame, chunks[1], app);
    if !app.bottom_pane.is_active() {
        if let Some(elapsed) = app.active_turn_elapsed(Instant::now()) {
            status_indicator_widget::render(frame, chunks[2], app.locale, elapsed);
        }
        pending_input_preview::render(frame, chunks[3], &app.queued_submissions, app.locale);
    }
    if app.bottom_pane.is_active() {
        bottom_pane::render_with_locale(frame, chunks[4], &app.bottom_pane, app.locale);
    } else {
        render_composer(frame, chunks[4], app);
    }
    render_footer(frame, chunks[5], app);
    if !app.bottom_pane.is_active() {
        if let Some(popup) = app.command_popup.as_ref() {
            command_popup::render(frame, chunks[4], popup, app.locale);
        }
    }
    if let Some(picker) = app.model_picker.as_ref() {
        model_picker::render_with_locale(frame, area, picker, app.locale);
    }
}

fn screen_chunks(area: Rect, app: &App) -> [Rect; 6] {
    let status_height =
        u16::from(!app.bottom_pane.is_active() && app.projection.active_turn_id().is_some());
    let preview_height = if app.bottom_pane.is_active() {
        0
    } else {
        pending_input_preview::desired_height(&app.queued_submissions, area.width, app.locale)
            .min(8)
            .min(area.height.saturating_sub(6 + status_height))
    };
    let input_height = if app.bottom_pane.is_active() {
        bottom_pane::desired_height_with_locale(&app.bottom_pane, app.locale)
    } else {
        let desired = app
            .composer
            .text()
            .lines()
            .count()
            .max(1)
            .saturating_add(app.pending_images.len())
            .saturating_add(2)
            .clamp(3, 12) as u16;
        desired.min(
            area.height
                .saturating_sub(preview_height)
                .saturating_sub(status_height)
                .saturating_sub(3)
                .max(1),
        )
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(status_height),
            Constraint::Length(preview_height),
            Constraint::Length(input_height),
            Constraint::Length(1),
        ])
        .split(area);
    [
        chunks[0], chunks[1], chunks[2], chunks[3], chunks[4], chunks[5],
    ]
}

pub(crate) fn transcript_page_size(width: u16, height: u16, app: &App) -> usize {
    let transcript = screen_chunks(Rect::new(0, 0, width, height), app)[1];
    usize::from(transcript.height.saturating_sub(1).max(1))
}

fn render_header(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let status = status_text(app);
    let model = app.model.as_deref().unwrap_or("auto");
    let effort = app.reasoning_effort.as_deref().unwrap_or("auto");
    let permissions = app.permissions.as_deref().unwrap_or("default");
    let suffix = format!(
        "  {}:{model} {}:{effort} {}:{permissions}",
        app.locale.model_label(),
        app.locale.effort_label(),
        app.locale.permissions_label()
    );
    let line = Line::from(vec![
        Span::styled(" Lime ", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("{}{suffix}", status),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(truncate_line_with_ellipsis_if_overflow(
            line,
            usize::from(area.width),
        )),
        area,
    );
}

fn render_transcript(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let lines = projected_transcript_lines(app, area.width, false);
    let paragraph = HyperlinkParagraph::new(&lines);
    let scroll = transcript_scroll_offset(
        paragraph.line_count(area.width),
        area,
        app.transcript_scroll,
    );
    frame.render_widget(paragraph.scroll(scroll), area);
}

fn projected_transcript_lines(
    app: &App,
    viewport_width: u16,
    separate_entries: bool,
) -> Vec<crate::terminal_hyperlinks::HyperlinkLine> {
    let content_width = Some(usize::from(viewport_width.saturating_sub(2).max(1)));
    let mut lines = Vec::new();
    for entry in app.projection.entries() {
        if separate_entries && !lines.is_empty() {
            lines.push(crate::terminal_hyperlinks::HyperlinkLine::default());
        }
        lines.extend(entry::hyperlink_lines_with_locale(
            entry,
            app.locale,
            content_width,
            &app.cwd,
        ));
    }
    lines
}

fn transcript_scroll_offset(
    rendered_line_count: usize,
    area: Rect,
    distance_from_bottom: usize,
) -> u16 {
    let max_scroll = rendered_line_count.saturating_sub(usize::from(area.height));
    let offset = max_scroll.saturating_sub(distance_from_bottom.min(max_scroll));
    u16::try_from(offset).unwrap_or(u16::MAX)
}

fn render_composer(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::TOP | Borders::BOTTOM)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    let mut lines = app
        .pending_images
        .iter()
        .enumerate()
        .map(|(index, _)| {
            Line::styled(
                format!("[Image #{}]", index + 1),
                Style::default().fg(Color::Cyan),
            )
        })
        .collect::<Vec<_>>();
    lines.extend(app.composer.text().split('\n').map(Line::raw));
    frame.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );

    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let before_cursor = &app.composer.text()[..app.composer.cursor()];
    let row = app
        .pending_images
        .len()
        .saturating_add(before_cursor.chars().filter(|ch| *ch == '\n').count());
    let column_text = before_cursor.rsplit('\n').next().unwrap_or("");
    let column = display_width(column_text);
    let x = inner.x.saturating_add(
        u16::try_from(column)
            .unwrap_or(u16::MAX)
            .min(inner.width.saturating_sub(1)),
    );
    let y = inner.y.saturating_add(
        u16::try_from(row)
            .unwrap_or(u16::MAX)
            .min(inner.height.saturating_sub(1)),
    );
    frame.set_cursor_position(Position::new(x, y));
}

fn render_footer(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if let Some(query) = app.composer.history_search_query() {
        let width = usable_content_width_u16(area.width, 1).unwrap_or_default();
        let line = truncate_line_with_ellipsis_if_overflow(
            Line::styled(
                format!(" {}{}", app.locale.history_search_label(), query),
                Style::default().fg(Color::DarkGray),
            ),
            width,
        );
        frame.render_widget(Paragraph::new(line), area);
        return;
    }
    let active = app
        .projection
        .active_turn_id()
        .map(|turn| format!(" {} {turn}", app.locale.turn_label()))
        .unwrap_or_default();
    let width = usable_content_width_u16(area.width, 1).unwrap_or_default();
    let line = truncate_line_with_ellipsis_if_overflow(
        Line::styled(format!(" {active}"), Style::default().fg(Color::DarkGray)),
        width,
    );
    frame.render_widget(Paragraph::new(line), area);
}

fn status_text(app: &App) -> String {
    match app.projection.status() {
        "" => app.locale.ready_label().to_string(),
        status => app.locale.status(status),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::locale::Locale;
    use app_server_protocol::RequestId;
    use app_server_protocol::protocol::v2::{
        AgentMessageDeltaNotification, CommandExecutionOutputDeltaNotification,
        CommandExecutionRequestApprovalParams, CommandExecutionSource, FileUpdateChange,
        ItemCompletedNotification, ItemStartedNotification, PatchApplyStatus, PatchChangeKind,
        QueuedSubmission, ServerNotification, ServerRequest, ThreadItem,
        ToolRequestUserInputParams, ToolRequestUserInputQuestion, TurnDiffUpdatedNotification,
        TurnPlanStep, TurnPlanStepStatus, TurnPlanUpdatedNotification, UserInput,
    };
    use crossterm::event::Event;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    #[test]
    fn wrapped_transcript_scroll_uses_visual_rows() {
        let paragraph = Paragraph::new("abcdefghij").wrap(Wrap { trim: false });
        let narrow = Rect::new(0, 0, 5, 1);

        let narrow_count = paragraph.line_count(narrow.width);
        assert_eq!(transcript_scroll_offset(narrow_count, narrow, 0), 1);
        assert_eq!(transcript_scroll_offset(narrow_count, narrow, 1), 0);
        assert_eq!(
            transcript_scroll_offset(narrow_count, narrow, usize::MAX),
            0
        );
        assert_eq!(
            transcript_scroll_offset(paragraph.line_count(10), Rect::new(0, 0, 10, 1), 0,),
            0
        );
    }

    #[test]
    fn transcript_page_size_tracks_resize() {
        let mut app = App::default();

        assert_eq!(transcript_page_size(80, 10, &app), 4);
        assert_eq!(transcript_page_size(80, 6, &app), 1);
        app.attach_image(std::path::PathBuf::from("/tmp/one.png"));
        app.attach_image(std::path::PathBuf::from("/tmp/two.png"));
        assert_eq!(transcript_page_size(80, 10, &app), 2);
    }

    fn buffer_text(terminal: &Terminal<TestBackend>) -> String {
        let buffer = terminal.backend().buffer();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| crate::terminal_hyperlinks::strip_osc8(buffer[(x, y)].symbol()))
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn test_backend_renders_streaming_unicode_and_composer() {
        let mut app = App::default();
        app.set_settings(
            Some("fixture-model".to_string()),
            Some("fixture-provider".to_string()),
            Some("high".to_string()),
            Some(":workspace".to_string()),
        );
        app.projection.apply(ServerNotification::AgentMessageDelta(
            AgentMessageDeltaNotification {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "item-1".to_string(),
                delta: "你好，terminal".to_string(),
            },
        ));
        app.composer.insert("继续");
        let mut terminal = Terminal::new(TestBackend::new(80, 10)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        assert!(text.contains("Lime"));
        assert!(text.contains('你'));
        assert!(text.contains('好'));
        assert!(text.contains("terminal"));
        assert!(text.contains('继'));
        assert!(text.contains('续'));
        assert!(text.contains("model:fixture-model"));
        assert!(text.contains("effort:high"));
    }

    #[test]
    fn test_backend_renders_pending_images_above_composer_text() {
        let mut app = App::default();
        app.attach_image(std::path::PathBuf::from("/tmp/one.png"));
        app.attach_image(std::path::PathBuf::from("/tmp/two.png"));
        app.composer.insert("describe these");
        let mut terminal = Terminal::new(TestBackend::new(40, 10)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        assert!(text.contains("[Image #1]"));
        assert!(text.contains("[Image #2]"));
        assert!(text.contains("describe these"));
    }

    #[test]
    fn test_backend_renders_canonical_queue_between_transcript_and_composer() {
        let mut app = App::default();
        app.set_queued_submissions(vec![QueuedSubmission {
            id: "queue-1".to_string(),
            input: vec![UserInput::Text {
                text: "follow up after this turn".to_string(),
                text_elements: Vec::new(),
            }],
            client_user_message_id: "client-queue-1".to_string(),
        }]);
        app.composer.insert("current draft");
        let mut terminal = Terminal::new(TestBackend::new(40, 10)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        let queued_row = text.find("queued (1)").expect("queued header");
        let message_row = text
            .find("follow up after this turn")
            .expect("queued message");
        let composer_row = text.find("current draft").expect("composer");
        assert!(
            queued_row < message_row && message_row < composer_row,
            "{text}"
        );
    }

    #[test]
    fn active_turn_status_precedes_canonical_queue_and_composer() {
        let mut app = App::default();
        app.start_turn("turn-1".to_string());
        app.set_queued_submissions(vec![QueuedSubmission {
            id: "queue-1".to_string(),
            input: vec![UserInput::Text {
                text: "follow up after this turn".to_string(),
                text_elements: Vec::new(),
            }],
            client_user_message_id: "client-queue-1".to_string(),
        }]);
        app.composer.insert("current draft");
        let mut terminal = Terminal::new(TestBackend::new(48, 12)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        let status_row = text.find("Working (").expect("active status");
        let interrupt_hint = text.find("esc to interrupt").expect("interrupt hint");
        let queued_row = text.find("queued (1)").expect("queued header");
        let composer_row = text.find("current draft").expect("composer");
        assert!(
            status_row < interrupt_hint && interrupt_hint < queued_row && queued_row < composer_row,
            "{text}"
        );
    }

    #[test]
    fn active_turn_status_does_not_overflow_a_tiny_terminal() {
        let mut app = App::default();
        app.start_turn("turn-1".to_string());
        app.set_queued_submissions(vec![QueuedSubmission {
            id: "queue-1".to_string(),
            input: vec![UserInput::Text {
                text: "queued text that cannot fit".to_string(),
                text_elements: Vec::new(),
            }],
            client_user_message_id: "client-queue-1".to_string(),
        }]);
        app.composer.insert("界界界界");
        let mut terminal = Terminal::new(TestBackend::new(12, 7)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        assert_eq!(text.lines().count(), 7, "{text}");
        assert!(text.contains("Working"), "{text}");
        assert!(text.contains('…'), "{text}");
    }

    #[test]
    fn history_search_footer_shows_localized_query_without_hiding_composer() {
        let mut app = App::default();
        app.set_locale(Locale::ZhCn);
        app.composer.load_history(["git status".to_string()]);
        app.composer.insert("git");
        app.handle_terminal_event(Event::Key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('r'),
            crossterm::event::KeyModifiers::CONTROL,
        )));
        for character in "git".chars() {
            app.handle_terminal_event(Event::Key(crossterm::event::KeyEvent::new(
                crossterm::event::KeyCode::Char(character),
                crossterm::event::KeyModifiers::NONE,
            )));
        }
        let mut terminal = Terminal::new(TestBackend::new(48, 8)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        let compact = text
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        assert!(compact.contains("反向搜索：git"), "{text}");
        assert!(text.contains("git status"), "{text}");
    }

    #[test]
    fn test_backend_renders_filtered_slash_command_popup_above_composer() {
        let mut app = App::default();
        app.set_locale(Locale::ZhCn);
        for character in ['/', 'p'] {
            app.handle_terminal_event(Event::Key(crossterm::event::KeyEvent::new(
                crossterm::event::KeyCode::Char(character),
                crossterm::event::KeyModifiers::NONE,
            )));
        }
        let mut terminal = Terminal::new(TestBackend::new(48, 10)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        let compact = text
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        assert!(text.contains("> /permissions"));
        assert!(compact.contains("设置权限配置"), "{text}");
        assert!(!text.contains("/model"));
        assert!(text.contains("/p"));
    }

    #[test]
    fn status_pager_owns_the_frame_and_renders_current_session_facts() {
        let mut app = App::default();
        app.set_thread_id("thread-1".to_string());
        app.set_cwd(std::path::PathBuf::from("/workspace"));
        app.set_settings(
            Some("gpt-5".to_string()),
            Some("openai".to_string()),
            Some("high".to_string()),
            Some(":workspace".to_string()),
        );
        app.composer.insert("/status");
        app.handle_terminal_event(Event::Key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        )));
        let mut terminal = Terminal::new(TestBackend::new(72, 12)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        for value in [
            "/ STATUS",
            "thread-1",
            "gpt-5",
            "openai",
            ":workspace",
            "/workspace",
            "100%",
        ] {
            assert!(text.contains(value), "missing {value}: {text}");
        }
        assert!(!text.contains("Lime"));
    }

    #[test]
    fn transcript_overlay_renders_live_canonical_projection_with_markdown_and_links() {
        let destination = "https://example.com/transcript";
        let mut app = App::default();
        app.composer.insert("draft remains private");
        app.projection.apply(ServerNotification::AgentMessageDelta(
            AgentMessageDeltaNotification {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "assistant-1".to_string(),
                delta: format!("**first** [link]({destination})"),
            },
        ));
        app.handle_terminal_event(Event::Key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('t'),
            crossterm::event::KeyModifiers::CONTROL,
        )));
        let mut terminal = Terminal::new(TestBackend::new(48, 9)).expect("terminal");
        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        assert!(text.contains("/ T R A N S C R I P T"), "{text}");
        assert!(text.contains("first"), "{text}");
        assert!(text.contains("link"), "{text}");
        assert!(!text.contains("draft remains private"), "{text}");
        assert!(terminal.backend().buffer().content.iter().any(|cell| {
            cell.symbol()
                .contains(&format!("\x1b]8;;{destination}\x07"))
        }));

        app.projection.apply(ServerNotification::AgentMessageDelta(
            AgentMessageDeltaNotification {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "assistant-2".to_string(),
                delta: "latest canonical update".to_string(),
            },
        ));
        terminal.draw(|frame| render(frame, &app)).expect("redraw");
        assert!(
            buffer_text(&terminal).contains("latest canonical update"),
            "{}",
            buffer_text(&terminal)
        );
    }

    #[test]
    fn user_visible_header_labels_cover_all_product_locales() {
        let cases = [
            (Locale::ZhCn, "模型:fixture-model", "设置已更新"),
            (Locale::ZhTw, "模型:fixture-model", "設定已更新"),
            (Locale::EnUs, "model:fixture-model", "settings updated"),
            (Locale::JaJp, "モデル:fixture-model", "設定を更新しました"),
            (Locale::KoKr, "모델:fixture-model", "설정이 업데이트됨"),
        ];
        for (locale, model_label, status_label) in cases {
            let mut app = App::default();
            app.set_locale(locale);
            app.set_settings(
                Some("fixture-model".to_string()),
                Some("fixture-provider".to_string()),
                Some("high".to_string()),
                Some(":workspace".to_string()),
            );
            app.projection.set_status("settings updated");
            let mut terminal = Terminal::new(TestBackend::new(80, 10)).expect("terminal");
            terminal.draw(|frame| render(frame, &app)).expect("draw");
            let text = buffer_text(&terminal);
            let compact = text
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect::<String>();
            assert!(compact.contains(model_label), "{locale:?}: {text}");
            let compact_status = status_label
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect::<String>();
            assert!(compact.contains(&compact_status), "{locale:?}: {text}");
        }
    }

    #[test]
    fn test_backend_renders_specialized_plan_and_patch_layouts() {
        let mut app = App::default();
        app.projection.apply(ServerNotification::TurnPlanUpdated(
            TurnPlanUpdatedNotification {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                explanation: None,
                plan: vec![TurnPlanStep {
                    step: "run tests".to_string(),
                    status: TurnPlanStepStatus::InProgress,
                }],
            },
        ));
        app.projection.apply(ServerNotification::TurnDiffUpdated(
            TurnDiffUpdatedNotification {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                diff: "+new line".to_string(),
            },
        ));
        let mut terminal = Terminal::new(TestBackend::new(40, 10)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        assert!(text.contains("• [~] run tests"));
        assert!(text.contains("Δ +new line"));
    }

    #[test]
    fn test_backend_renders_markdown_and_numbered_diff() {
        let mut app = App::default();
        app.projection.apply(ServerNotification::AgentMessageDelta(
            AgentMessageDeltaNotification {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "assistant-1".to_string(),
                delta: "## Result\n\nRead [the guide](https://example.com/guide).".to_string(),
            },
        ));
        app.projection.apply(ServerNotification::TurnDiffUpdated(
            TurnDiffUpdatedNotification {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                diff: "@@ -1 +1 @@\n-old\n+new".to_string(),
            },
        ));
        let mut terminal = Terminal::new(TestBackend::new(80, 16)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        assert!(text.contains("## Result"));
        assert!(text.contains("the guide (https://example.com/guide)"));
        assert!(text.contains("1 -old"));
        assert!(text.contains("1 +new"));
    }

    #[test]
    fn test_backend_marks_markdown_links_with_osc8() {
        let destination = "https://example.com/guide";
        let mut app = App::default();
        app.projection.apply(ServerNotification::AgentMessageDelta(
            AgentMessageDeltaNotification {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "assistant-1".to_string(),
                delta: format!("Read [the guide]({destination})."),
            },
        ));
        let mut terminal = Terminal::new(TestBackend::new(40, 10)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let buffer = terminal.backend().buffer();
        assert!(buffer.content.iter().any(|cell| {
            cell.symbol()
                .contains(&format!("\x1b]8;;{destination}\x07"))
        }));
        let visible = buffer_text(&terminal);
        assert!(visible.contains("the guide"));
        assert!(visible.contains("https://example.com/guide"));
    }

    #[test]
    fn test_backend_renders_completed_item_summaries() {
        let mut app = App::default();
        app.projection.apply(ServerNotification::ItemCompleted(
            ItemCompletedNotification {
                item: ThreadItem::CommandExecution {
                    id: "command-1".to_string(),
                    metadata: None,
                    plugin_id: None,
                    script_path: None,
                    command: "cargo test -p tui".to_string(),
                    cwd: "/workspace".to_string(),
                    process_id: None,
                    source: CommandExecutionSource::Agent,
                    status: app_server_protocol::protocol::v2::CommandExecutionStatus::Completed,
                    command_actions: Vec::new(),
                    aggregated_output: Some("ok".to_string()),
                    exit_code: Some(0),
                    duration_ms: Some(42),
                    terminal_interactions: Vec::new(),
                },
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                completed_at_ms: 1,
            },
        ));
        app.projection.apply(ServerNotification::ItemCompleted(
            ItemCompletedNotification {
                item: ThreadItem::FileChange {
                    id: "patch-1".to_string(),
                    metadata: None,
                    changes: vec![FileUpdateChange {
                        path: "src/lib.rs".to_string(),
                        kind: PatchChangeKind::Add,
                        diff: "+new".to_string(),
                    }],
                    status: PatchApplyStatus::Completed,
                },
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                completed_at_ms: 2,
            },
        ));
        let mut terminal = Terminal::new(TestBackend::new(80, 16)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        assert!(text.contains("cargo test -p tui [completed]"));
        assert!(text.contains("- exit 0"));
        assert!(text.contains("- duration 42ms"));
        assert!(text.contains("src/lib.rs (+1 -0) [completed]"));
        assert!(text.contains("- files: 1"));
        assert!(text.contains("+new"));
    }

    #[test]
    fn test_backend_renders_live_command_output_below_the_command() {
        let mut app = App::default();
        app.projection
            .apply(ServerNotification::ItemStarted(ItemStartedNotification {
                item: ThreadItem::CommandExecution {
                    id: "command-1".to_string(),
                    metadata: None,
                    plugin_id: None,
                    script_path: None,
                    command: "printf data".to_string(),
                    cwd: "/workspace".to_string(),
                    process_id: None,
                    source: CommandExecutionSource::Agent,
                    status: app_server_protocol::protocol::v2::CommandExecutionStatus::InProgress,
                    command_actions: Vec::new(),
                    aggregated_output: None,
                    exit_code: None,
                    duration_ms: None,
                    terminal_interactions: Vec::new(),
                },
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                started_at_ms: 1,
            }));
        for delta in ["std", "out\nstderr\n"] {
            app.projection
                .apply(ServerNotification::CommandExecutionOutputDelta(
                    CommandExecutionOutputDeltaNotification {
                        thread_id: "thread-1".to_string(),
                        turn_id: "turn-1".to_string(),
                        item_id: "command-1".to_string(),
                        delta: delta.to_string(),
                    },
                ));
        }
        let mut terminal = Terminal::new(TestBackend::new(60, 10)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        let command = text.find("$ printf data [running]").expect("command row");
        let stdout = text.find("stdout").expect("stdout row");
        let stderr = text.find("stderr").expect("stderr row");
        assert!(command < stdout && stdout < stderr, "{text}");
        assert!(!text.contains("datastdout"), "{text}");
    }

    #[test]
    fn test_backend_renders_bounded_command_output_marker() {
        let mut app = App::default();
        app.projection
            .apply(ServerNotification::ItemStarted(ItemStartedNotification {
                item: ThreadItem::CommandExecution {
                    id: "command-large".to_string(),
                    metadata: None,
                    plugin_id: None,
                    script_path: None,
                    command: "printf output".to_string(),
                    cwd: "/workspace".to_string(),
                    process_id: None,
                    source: CommandExecutionSource::Agent,
                    status: app_server_protocol::protocol::v2::CommandExecutionStatus::InProgress,
                    command_actions: Vec::new(),
                    aggregated_output: None,
                    exit_code: None,
                    duration_ms: None,
                    terminal_interactions: Vec::new(),
                },
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                started_at_ms: 1,
            }));
        let output = (0..101)
            .map(|index| format!("line-{index}"))
            .collect::<Vec<_>>()
            .join("\n");
        app.projection
            .apply(ServerNotification::CommandExecutionOutputDelta(
                CommandExecutionOutputDeltaNotification {
                    thread_id: "thread-1".to_string(),
                    turn_id: "turn-1".to_string(),
                    item_id: "command-large".to_string(),
                    delta: output,
                },
            ));
        let mut terminal = Terminal::new(TestBackend::new(80, 110)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        assert!(text.contains("… 1 lines omitted …"), "{text}");
        assert!(text.contains("line-0"), "{text}");
        assert!(text.contains("line-100"), "{text}");
        assert!(!text.contains("line-50"), "{text}");
    }

    #[test]
    fn narrow_terminal_does_not_overflow_or_panic() {
        let mut app = App::default();
        app.projection.set_status("a-status-that-does-not-fit");
        app.composer.insert("界界界界");
        let mut terminal = Terminal::new(TestBackend::new(8, 6)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        assert_eq!(text.lines().count(), 6);
        assert!(text.contains("Lime"));
    }

    #[test]
    fn approval_replaces_the_composer_with_actionable_options() {
        let mut app = App::default();
        app.composer.insert("unsent draft");
        app.bottom_pane
            .enqueue(ServerRequest::ItemCommandExecutionRequestApproval {
                id: RequestId::Integer(7),
                params: CommandExecutionRequestApprovalParams {
                    thread_id: "thread-1".to_string(),
                    turn_id: "turn-1".to_string(),
                    item_id: "command-1".to_string(),
                    started_at_ms: 1,
                    approval_id: None,
                    reason: Some("run the focused regression".to_string()),
                    network_approval_context: None,
                    command: Some("cargo test -p tui".to_string()),
                    cwd: Some("/workspace".to_string()),
                    available_decisions: None,
                },
            })
            .expect("queue approval");
        let mut terminal = Terminal::new(TestBackend::new(60, 16)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        assert!(text.contains("Approve command?"));
        assert!(text.contains("cargo test -p tui"));
        assert!(text.contains("Allow once"));
        assert!(!text.contains("unsent draft"));
    }

    #[test]
    fn approval_hides_an_open_slash_command_popup() {
        let mut app = App::default();
        app.handle_terminal_event(Event::Key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('/'),
            crossterm::event::KeyModifiers::NONE,
        )));
        app.bottom_pane
            .enqueue(ServerRequest::ItemCommandExecutionRequestApproval {
                id: RequestId::Integer(9),
                params: CommandExecutionRequestApprovalParams {
                    thread_id: "thread-1".to_string(),
                    turn_id: "turn-1".to_string(),
                    item_id: "command-1".to_string(),
                    started_at_ms: 1,
                    approval_id: None,
                    reason: None,
                    network_approval_context: None,
                    command: Some("cargo test".to_string()),
                    cwd: Some("/workspace".to_string()),
                    available_decisions: None,
                },
            })
            .expect("queue approval");
        let mut terminal = Terminal::new(TestBackend::new(72, 12)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        assert!(text.contains("Approve command?"));
        assert!(!text.contains("/model"));
        assert!(!text.contains("copy the last response"));
    }

    #[test]
    fn secret_user_input_is_masked_in_the_test_backend() {
        let mut app = App::default();
        app.bottom_pane
            .enqueue(ServerRequest::ItemToolRequestUserInput {
                id: RequestId::Integer(8),
                params: ToolRequestUserInputParams {
                    thread_id: "thread-1".to_string(),
                    turn_id: "turn-1".to_string(),
                    item_id: "question-1".to_string(),
                    questions: vec![ToolRequestUserInputQuestion {
                        id: "token".to_string(),
                        header: "Token".to_string(),
                        question: "Enter token".to_string(),
                        is_other: false,
                        is_secret: true,
                        options: None,
                    }],
                    auto_resolution_ms: None,
                },
            })
            .expect("queue user input");
        app.handle_terminal_event(Event::Paste("sensitive".to_string()));
        let mut terminal = Terminal::new(TestBackend::new(40, 10)).expect("terminal");

        terminal.draw(|frame| render(frame, &app)).expect("draw");

        let text = buffer_text(&terminal);
        assert!(text.contains("*********"));
        assert!(!text.contains("sensitive"));
    }
}
