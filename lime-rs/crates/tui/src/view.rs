use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::App;
use crate::bottom_pane;
use crate::entry;
use crate::line_truncation::truncate_line_with_ellipsis_if_overflow;
use crate::model_picker;
use crate::width::{display_width, usable_content_width_u16};

pub(crate) fn render(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();
    let chunks = screen_chunks(area, app);

    render_header(frame, chunks[0], app);
    render_transcript(frame, chunks[1], app);
    if app.bottom_pane.is_active() {
        bottom_pane::render(frame, chunks[2], &app.bottom_pane);
    } else {
        render_composer(frame, chunks[2], app);
    }
    render_footer(frame, chunks[3], app);
    if let Some(picker) = app.model_picker.as_ref() {
        model_picker::render(frame, area, picker);
    }
}

fn screen_chunks(area: Rect, app: &App) -> [Rect; 4] {
    let input_height = if app.bottom_pane.is_active() {
        bottom_pane::desired_height(&app.bottom_pane)
    } else {
        app.composer
            .text()
            .lines()
            .count()
            .saturating_add(2)
            .clamp(3, 7) as u16
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(input_height),
            Constraint::Length(1),
        ])
        .split(area);
    [chunks[0], chunks[1], chunks[2], chunks[3]]
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
    let suffix = format!("  model:{model} effort:{effort} permissions:{permissions}");
    let line = Line::from(vec![
        Span::styled(" Lime ", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("{status}{suffix}"),
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
    let lines = app
        .projection
        .entries()
        .iter()
        .flat_map(entry::lines)
        .collect::<Vec<_>>();
    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    let scroll = transcript_scroll_offset(&paragraph, area, app.transcript_scroll);
    frame.render_widget(paragraph.scroll((scroll, 0)), area);
}

fn transcript_scroll_offset(
    paragraph: &Paragraph<'_>,
    area: Rect,
    distance_from_bottom: usize,
) -> u16 {
    let max_scroll = paragraph
        .line_count(area.width)
        .saturating_sub(usize::from(area.height));
    let offset = max_scroll.saturating_sub(distance_from_bottom.min(max_scroll));
    u16::try_from(offset).unwrap_or(u16::MAX)
}

fn render_composer(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::TOP | Borders::BOTTOM)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    frame.render_widget(
        Paragraph::new(app.composer.text().to_string())
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );

    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let before_cursor = &app.composer.text()[..app.composer.cursor()];
    let row = before_cursor.chars().filter(|ch| *ch == '\n').count();
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
    let active = app
        .projection
        .active_turn_id()
        .map(|turn| format!(" turn {turn}"))
        .unwrap_or_default();
    let width = usable_content_width_u16(area.width, 1).unwrap_or_default();
    let line = truncate_line_with_ellipsis_if_overflow(
        Line::styled(format!(" {active}"), Style::default().fg(Color::DarkGray)),
        width,
    );
    frame.render_widget(Paragraph::new(line), area);
}

fn status_text(app: &App) -> &str {
    match app.projection.status() {
        "" => "ready",
        status => status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::protocol::v2::{
        AgentMessageDeltaNotification, CommandExecutionRequestApprovalParams,
        CommandExecutionSource, FileUpdateChange, ItemCompletedNotification, PatchApplyStatus,
        PatchChangeKind, ServerNotification, ServerRequest, ThreadItem, ToolRequestUserInputParams,
        ToolRequestUserInputQuestion, TurnDiffUpdatedNotification, TurnPlanStep,
        TurnPlanStepStatus, TurnPlanUpdatedNotification,
    };
    use app_server_protocol::RequestId;
    use crossterm::event::Event;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn wrapped_transcript_scroll_uses_visual_rows() {
        let paragraph = Paragraph::new("abcdefghij").wrap(Wrap { trim: false });
        let narrow = Rect::new(0, 0, 5, 1);

        assert_eq!(transcript_scroll_offset(&paragraph, narrow, 0), 1);
        assert_eq!(transcript_scroll_offset(&paragraph, narrow, 1), 0);
        assert_eq!(transcript_scroll_offset(&paragraph, narrow, usize::MAX), 0);
        assert_eq!(
            transcript_scroll_offset(&paragraph, Rect::new(0, 0, 10, 1), 0),
            0
        );
    }

    #[test]
    fn transcript_page_size_tracks_resize() {
        let app = App::default();

        assert_eq!(transcript_page_size(80, 10, &app), 4);
        assert_eq!(transcript_page_size(80, 6, &app), 1);
    }

    fn buffer_text(terminal: &Terminal<TestBackend>) -> String {
        let buffer = terminal.backend().buffer();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
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
            Some("workspace-write".to_string()),
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
        assert!(text.contains("src/lib.rs [completed]"));
        assert!(text.contains("- files: 1"));
        assert!(text.contains("+new"));
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
