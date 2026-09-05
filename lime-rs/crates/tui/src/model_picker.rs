use app_server_protocol::protocol::v2::Model;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph};

use crate::line_truncation::truncate_line_with_ellipsis_if_overflow;
use crate::locale::Locale;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelSelection {
    pub(crate) model: String,
    pub(crate) provider: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelPickerAction {
    None,
    Cancel,
    Select(usize),
}

#[derive(Debug, Default)]
pub(crate) struct ModelPicker {
    models: Vec<Model>,
    selected: usize,
    query: String,
}

impl ModelPicker {
    pub(crate) fn new(models: Vec<Model>) -> Self {
        let mut models = models
            .into_iter()
            .filter(|model| !model.hidden)
            .collect::<Vec<_>>();
        models.sort_by_key(|model| (!model.is_default, model.display_name.to_lowercase()));
        Self {
            models,
            selected: 0,
            query: String::new(),
        }
    }

    pub(crate) fn selected_model(&self, index: usize) -> Option<ModelSelection> {
        self.visible_indices()
            .get(index)
            .and_then(|model_index| self.models.get(*model_index))
            .map(|model| ModelSelection {
                model: model.model.clone(),
                provider: model.provider_id.clone(),
            })
    }

    pub(crate) fn query(&self) -> &str {
        &self.query
    }

    pub(crate) fn visible_models(&self) -> Vec<&Model> {
        self.visible_indices()
            .into_iter()
            .filter_map(|index| self.models.get(index))
            .collect()
    }

    pub(crate) fn handle_event(&mut self, event: Event) -> ModelPickerAction {
        match event {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('d'))
                {
                    return ModelPickerAction::Cancel;
                }
                match key.code {
                    KeyCode::Esc => ModelPickerAction::Cancel,
                    KeyCode::Up | KeyCode::Char('k') => {
                        self.selected = self.selected.saturating_sub(1);
                        ModelPickerAction::None
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        let count = self.visible_indices().len();
                        if count > 0 {
                            self.selected = (self.selected + 1).min(count - 1);
                        }
                        ModelPickerAction::None
                    }
                    KeyCode::Enter => self
                        .selected_model(self.selected)
                        .map(|_| ModelPickerAction::Select(self.selected))
                        .unwrap_or(ModelPickerAction::None),
                    KeyCode::Backspace => {
                        self.query.pop();
                        self.selected = 0;
                        ModelPickerAction::None
                    }
                    KeyCode::Char(ch) => {
                        self.query.push(ch);
                        self.selected = 0;
                        ModelPickerAction::None
                    }
                    _ => ModelPickerAction::None,
                }
            }
            Event::Paste(text) => {
                self.query.push_str(&text);
                self.selected = 0;
                ModelPickerAction::None
            }
            _ => ModelPickerAction::None,
        }
    }

    fn visible_indices(&self) -> Vec<usize> {
        let query = self.query.trim().to_lowercase();
        self.models
            .iter()
            .enumerate()
            .filter(|(_, model)| {
                query.is_empty()
                    || model.display_name.to_lowercase().contains(&query)
                    || model.model.to_lowercase().contains(&query)
                    || model.provider_id.to_lowercase().contains(&query)
            })
            .map(|(index, _)| index)
            .collect()
    }
}

#[cfg(test)]
pub(crate) fn render(frame: &mut Frame<'_>, area: Rect, picker: &ModelPicker) {
    render_with_locale(frame, area, picker, Locale::default());
}

pub(crate) fn render_with_locale(
    frame: &mut Frame<'_>,
    area: Rect,
    picker: &ModelPicker,
    locale: Locale,
) {
    let width = area.width.saturating_mul(4).saturating_div(5).clamp(28, 72);
    let height = area.height.saturating_mul(3).saturating_div(4).clamp(7, 18);
    let x = area.x.saturating_add(area.width.saturating_sub(width) / 2);
    let y = area
        .y
        .saturating_add(area.height.saturating_sub(height) / 2);
    let popup = Rect::new(x, y, width.min(area.width), height.min(area.height));

    frame.render_widget(Clear, popup);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(popup);
    let title = Line::from(vec![
        Span::styled(
            format!(" {} ", locale.model_label()),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            if picker.query().is_empty() {
                locale.picker_title().to_string()
            } else {
                picker.query().to_string()
            },
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(truncate_line_with_ellipsis_if_overflow(
            title,
            usize::from(chunks[0].width.saturating_sub(2)),
        ))
        .block(Block::default().borders(Borders::TOP)),
        chunks[0],
    );

    let items = picker
        .visible_models()
        .into_iter()
        .map(|model| {
            let label = format!("{}  [{}]", model.display_name, model.provider_id);
            ListItem::new(truncate_line_with_ellipsis_if_overflow(
                Line::from(label),
                usize::from(chunks[1].width.saturating_sub(2)),
            ))
        })
        .collect::<Vec<_>>();
    let mut state = ListState::default();
    if !items.is_empty() {
        state.select(Some(picker.selected));
    }
    frame.render_stateful_widget(
        List::new(items)
            .block(Block::default().borders(Borders::LEFT | Borders::RIGHT))
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("> "),
        chunks[1],
        &mut state,
    );
    let footer = if picker.visible_models().is_empty() {
        locale.picker_empty()
    } else {
        locale.picker_footer()
    };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            footer,
            Style::default().fg(Color::DarkGray),
        )))
        .block(Block::default().borders(Borders::BOTTOM)),
        chunks[2],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::CapabilitySnapshot;
    use app_server_protocol::protocol::v2::{InputModality, Model};
    use crossterm::event::{KeyEvent, KeyModifiers};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn model(id: &str, provider: &str, hidden: bool, is_default: bool) -> Model {
        Model {
            id: id.to_string(),
            provider_id: provider.to_string(),
            model: id.to_string(),
            upgrade: None,
            upgrade_info: None,
            availability_nux: None,
            display_name: id.to_string(),
            description: String::new(),
            hidden,
            supported_reasoning_efforts: Vec::new(),
            default_reasoning_effort: "medium".to_string(),
            input_modalities: vec![InputModality::Text],
            capability_snapshot: CapabilitySnapshot::default(),
            context_window: None,
            max_output_tokens: None,
            supports_personality: false,
            multi_agent_version: None,
            additional_speed_tiers: Vec::new(),
            service_tiers: Vec::new(),
            default_service_tier: None,
            is_default,
        }
    }

    #[test]
    fn picker_filters_hidden_models_and_selects_default_first() {
        let mut picker = ModelPicker::new(vec![
            model("slow", "fixture", false, false),
            model("hidden", "fixture", true, false),
            model("fast", "fixture", false, true),
        ]);
        assert_eq!(picker.visible_models().len(), 2);
        assert_eq!(picker.selected_model(0).expect("default").model, "fast");
        picker.handle_event(Event::Key(KeyEvent::new(
            KeyCode::Char('l'),
            KeyModifiers::NONE,
        )));
        assert_eq!(picker.selected_model(0).expect("filtered").model, "slow");
        assert_eq!(
            picker.handle_event(Event::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            ))),
            ModelPickerAction::Select(0)
        );
    }

    #[test]
    fn picker_escape_cancels_and_render_stays_bounded() {
        let mut picker = ModelPicker::new(vec![model("模型", "提供方", false, true)]);
        assert_eq!(
            picker.handle_event(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE,))),
            ModelPickerAction::Cancel
        );
        let mut terminal = Terminal::new(TestBackend::new(12, 6)).expect("terminal");
        terminal
            .draw(|frame| render(frame, frame.area(), &picker))
            .expect("draw");
    }
}
