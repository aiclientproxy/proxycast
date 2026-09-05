//! Width-aware Markdown table rendering adapted from Codex TUI.

use pulldown_cmark::Alignment;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::{TableRow, TableState};
use crate::terminal_hyperlinks::{HyperlinkLine, wrap_hyperlink_line};
use crate::width::display_width;

const COLUMN_GAP: usize = 2;
const CELL_PADDING: usize = 1;
const MIN_COLUMN_WIDTH: usize = 3;
const MIN_SCANNABLE_EXPANSIVE_WIDTH: usize = 12;
const RECORD_FIELD_GAP: usize = 2;
const STACKED_VALUE_INDENT: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ColumnKind {
    TokenHeavy,
    Narrative,
    Compact,
}

#[derive(Clone, Copy, Debug)]
struct ColumnMetrics {
    max_width: usize,
    header_token_width: usize,
    body_token_width: usize,
    kind: ColumnKind,
}

pub(super) fn render(table: TableState, width: Option<usize>) -> Vec<HyperlinkLine> {
    let column_count = table
        .rows
        .iter()
        .map(|row| row.cells.len())
        .max()
        .unwrap_or(0);
    if column_count == 0 {
        return Vec::new();
    }

    let mut header = table
        .rows
        .iter()
        .find(|row| row.header)
        .cloned()
        .unwrap_or_else(|| TableRow {
            cells: vec![HyperlinkLine::default(); column_count],
            header: true,
        });
    normalize_row(&mut header, column_count);
    let mut rows = table
        .rows
        .into_iter()
        .filter(|row| !row.header)
        .collect::<Vec<_>>();
    for row in &mut rows {
        normalize_row(row, column_count);
    }

    let metrics = collect_metrics(&header, &rows, column_count);
    let available_content_width = width.map(|width| {
        width.saturating_sub(
            column_count * CELL_PADDING * 2 + column_count.saturating_sub(1) * COLUMN_GAP,
        )
    });
    let Some(column_widths) = compute_column_widths(&metrics, available_content_width) else {
        return render_records(&header, &rows, width);
    };
    if should_render_records(&rows, &column_widths, &metrics) {
        return render_records(&header, &rows, width);
    }

    let mut out = render_row(&header, &column_widths, &table.alignments, true);
    out.push(render_separator(&column_widths, '━'));
    for (index, row) in rows.iter().enumerate() {
        out.extend(render_row(row, &column_widths, &table.alignments, false));
        if index + 1 < rows.len() {
            out.push(render_separator(&column_widths, '─'));
        }
    }
    out
}

fn normalize_row(row: &mut TableRow, column_count: usize) {
    row.cells.truncate(column_count);
    row.cells.resize(column_count, HyperlinkLine::default());
}

fn collect_metrics(
    header: &TableRow,
    rows: &[TableRow],
    column_count: usize,
) -> Vec<ColumnMetrics> {
    (0..column_count)
        .map(|column| {
            let header_text = line_text(&header.cells[column]);
            let header_token_width = longest_token_width(&header_text);
            let mut max_width = header.cells[column].width();
            let mut body_token_width = 0usize;
            let mut body_token_count = 0usize;
            let mut long_body_token_count = 0usize;
            let mut total_words = 0usize;
            let mut total_cells = 0usize;
            let mut total_cell_width = 0usize;

            for row in rows {
                let cell = &row.cells[column];
                max_width = max_width.max(cell.width());
                let text = line_text(cell);
                let mut word_count = 0usize;
                for token in text.split_whitespace() {
                    let token_width = display_width(token);
                    body_token_width = body_token_width.max(token_width);
                    long_body_token_count += usize::from(token_width >= 20);
                    word_count += 1;
                }
                if word_count > 0 {
                    body_token_count += word_count;
                    total_words += word_count;
                    total_cells += 1;
                    total_cell_width += display_width(&text);
                }
            }

            let avg_words = if total_cells == 0 {
                header_text.split_whitespace().count() as f64
            } else {
                total_words as f64 / total_cells as f64
            };
            let avg_width = if total_cells == 0 {
                display_width(&header_text) as f64
            } else {
                total_cell_width as f64 / total_cells as f64
            };
            let kind = if long_body_token_count > 0
                && long_body_token_count >= body_token_count.saturating_sub(long_body_token_count)
            {
                ColumnKind::TokenHeavy
            } else if avg_words >= 4.0 || avg_width >= 28.0 {
                ColumnKind::Narrative
            } else {
                ColumnKind::Compact
            };

            ColumnMetrics {
                max_width,
                header_token_width,
                body_token_width,
                kind,
            }
        })
        .collect()
}

fn compute_column_widths(
    metrics: &[ColumnMetrics],
    available_width: Option<usize>,
) -> Option<Vec<usize>> {
    let mut widths = metrics
        .iter()
        .map(|column| column.max_width.max(MIN_COLUMN_WIDTH))
        .collect::<Vec<_>>();
    let Some(max_width) = available_width else {
        return Some(widths);
    };
    if max_width < metrics.len() * MIN_COLUMN_WIDTH {
        return None;
    }

    let mut floors = metrics
        .iter()
        .map(preferred_column_floor)
        .collect::<Vec<_>>();
    let floor_total = floors.iter().sum::<usize>();
    if floor_total > max_width {
        let minimums = vec![MIN_COLUMN_WIDTH; floors.len()];
        shrink_columns(&mut floors, &minimums, metrics, floor_total - max_width);
    }

    let total = widths.iter().sum::<usize>();
    if total > max_width && shrink_columns(&mut widths, &floors, metrics, total - max_width) > 0 {
        return None;
    }
    Some(widths)
}

fn preferred_column_floor(metrics: &ColumnMetrics) -> usize {
    let target = match metrics.kind {
        ColumnKind::Narrative | ColumnKind::TokenHeavy => 16,
        ColumnKind::Compact => metrics
            .header_token_width
            .max(metrics.body_token_width.min(16)),
    };
    target
        .max(MIN_COLUMN_WIDTH)
        .min(metrics.max_width.max(MIN_COLUMN_WIDTH))
}

fn shrink_columns(
    widths: &mut [usize],
    floors: &[usize],
    metrics: &[ColumnMetrics],
    mut amount: usize,
) -> usize {
    for kind in [
        ColumnKind::TokenHeavy,
        ColumnKind::Narrative,
        ColumnKind::Compact,
    ] {
        while amount > 0 {
            let Some(index) = widths
                .iter()
                .enumerate()
                .filter(|(index, width)| metrics[*index].kind == kind && **width > floors[*index])
                .max_by_key(|(index, width)| width.saturating_sub(floors[*index]))
                .map(|(index, _)| index)
            else {
                break;
            };
            widths[index] -= 1;
            amount -= 1;
        }
        if amount == 0 {
            break;
        }
    }
    amount
}

fn should_render_records(rows: &[TableRow], widths: &[usize], metrics: &[ColumnMetrics]) -> bool {
    if rows.is_empty() {
        return false;
    }
    let affected_rows = rows
        .iter()
        .filter(|row| {
            let fragmented =
                row.cells
                    .iter()
                    .zip(widths)
                    .zip(metrics)
                    .any(|((cell, width), metrics)| {
                        let fragmented_token = line_text(cell)
                            .split_whitespace()
                            .any(|token| display_width(token) > *width);
                        match metrics.kind {
                            ColumnKind::Compact => fragmented_token,
                            ColumnKind::TokenHeavy => {
                                *width < MIN_SCANNABLE_EXPANSIVE_WIDTH && fragmented_token
                            }
                            ColumnKind::Narrative => false,
                        }
                    });
            let starved_expansive = row
                .cells
                .iter()
                .zip(widths)
                .zip(metrics)
                .filter(|((_, width), metrics)| {
                    metrics.kind != ColumnKind::Compact && **width < MIN_SCANNABLE_EXPANSIVE_WIDTH
                })
                .filter(|((cell, width), _)| wrap_hyperlink_line(cell, **width).len() >= 4)
                .count()
                >= 2;
            fragmented || starved_expansive
        })
        .count();
    let threshold = if rows.len() == 1 {
        1
    } else {
        2.max(rows.len().div_ceil(3))
    };
    affected_rows >= threshold
}

fn render_row(
    row: &TableRow,
    widths: &[usize],
    alignments: &[Alignment],
    header: bool,
) -> Vec<HyperlinkLine> {
    let wrapped = row
        .cells
        .iter()
        .zip(widths)
        .map(|(cell, width)| wrap_hyperlink_line(cell, *width))
        .collect::<Vec<_>>();
    let row_height = wrapped.iter().map(Vec::len).max().unwrap_or(1);
    let mut out = Vec::with_capacity(row_height);

    for row_line in 0..row_height {
        let mut line = HyperlinkLine::default();
        for (column, width) in widths.iter().copied().enumerate() {
            if column > 0 {
                line.push_span(Span::raw(" ".repeat(COLUMN_GAP)), None);
            }
            line.push_span(Span::raw(" ".repeat(CELL_PADDING)), None);
            let mut cell = wrapped[column].get(row_line).cloned().unwrap_or_default();
            if header {
                for span in &mut cell.line.spans {
                    span.style = span
                        .style
                        .patch(Style::default().add_modifier(Modifier::BOLD));
                }
            }
            let padding = width.saturating_sub(cell.width());
            let (left, right) = match alignments.get(column).copied().unwrap_or(Alignment::None) {
                Alignment::Right => (padding, 0),
                Alignment::Center => (padding / 2, padding - padding / 2),
                Alignment::None | Alignment::Left => (0, padding),
            };
            line.push_span(Span::raw(" ".repeat(left)), None);
            line.append(cell);
            line.push_span(Span::raw(" ".repeat(right + CELL_PADDING)), None);
        }
        out.push(line);
    }
    out
}

fn render_separator(widths: &[usize], separator: char) -> HyperlinkLine {
    let segment = separator.to_string();
    let text = widths
        .iter()
        .map(|width| segment.repeat(width + CELL_PADDING * 2))
        .collect::<Vec<_>>()
        .join(&" ".repeat(COLUMN_GAP));
    HyperlinkLine::new(Line::styled(text, Style::default().fg(Color::DarkGray)))
}

fn render_records(
    header: &TableRow,
    rows: &[TableRow],
    width: Option<usize>,
) -> Vec<HyperlinkLine> {
    if rows.is_empty() {
        return render_pipe_fallback(header, &[]);
    }
    let label_width = header
        .cells
        .iter()
        .map(|cell| display_width(&line_text(cell)))
        .max()
        .unwrap_or(0);
    let aligned = width.is_none_or(|width| {
        CELL_PADDING + label_width + RECORD_FIELD_GAP + MIN_SCANNABLE_EXPANSIVE_WIDTH <= width
    });
    let mut out = Vec::new();
    for (row_index, row) in rows.iter().enumerate() {
        for (label, value) in header.cells.iter().zip(&row.cells) {
            if aligned {
                render_aligned_field(&mut out, label, value, label_width, width);
            } else {
                render_stacked_field(&mut out, label, value, width);
            }
        }
        if row_index + 1 < rows.len() {
            let separator_width =
                width.unwrap_or_else(|| out.iter().map(HyperlinkLine::width).max().unwrap_or(1));
            out.push(HyperlinkLine::new(Line::styled(
                "─".repeat(separator_width.max(1)),
                Style::default().fg(Color::DarkGray),
            )));
        }
    }
    out
}

fn render_aligned_field(
    out: &mut Vec<HyperlinkLine>,
    label: &HyperlinkLine,
    value: &HyperlinkLine,
    label_width: usize,
    width: Option<usize>,
) {
    let value_indent = CELL_PADDING + label_width + RECORD_FIELD_GAP;
    let value_width = width
        .map(|width| width.saturating_sub(value_indent).max(1))
        .unwrap_or_else(|| value.width().max(1));
    for (index, value_line) in wrap_hyperlink_line(value, value_width)
        .into_iter()
        .enumerate()
    {
        let mut line = HyperlinkLine::default();
        if index == 0 {
            let label = line_text(label);
            line.push_span(Span::raw(" "), None);
            line.push_span(
                Span::styled(label.clone(), Style::default().add_modifier(Modifier::BOLD)),
                None,
            );
            line.push_span(
                Span::raw(
                    " ".repeat(
                        label_width.saturating_sub(display_width(&label)) + RECORD_FIELD_GAP,
                    ),
                ),
                None,
            );
        } else {
            line.push_span(Span::raw(" ".repeat(value_indent)), None);
        }
        line.append(value_line);
        out.push(line);
    }
}

fn render_stacked_field(
    out: &mut Vec<HyperlinkLine>,
    label: &HyperlinkLine,
    value: &HyperlinkLine,
    width: Option<usize>,
) {
    let label_width = width
        .map(|width| width.saturating_sub(CELL_PADDING).max(1))
        .unwrap_or_else(|| label.width().max(1));
    for label_line in wrap_hyperlink_line(label, label_width) {
        let mut line = HyperlinkLine::default();
        line.push_span(Span::raw(" "), None);
        let mut label_line = label_line;
        for span in &mut label_line.line.spans {
            span.style = span
                .style
                .patch(Style::default().add_modifier(Modifier::BOLD));
        }
        line.append(label_line);
        out.push(line);
    }
    let value_width = width
        .map(|width| width.saturating_sub(STACKED_VALUE_INDENT).max(1))
        .unwrap_or_else(|| value.width().max(1));
    for value_line in wrap_hyperlink_line(value, value_width) {
        let mut line = HyperlinkLine::default();
        line.push_span(Span::raw(" ".repeat(STACKED_VALUE_INDENT)), None);
        line.append(value_line);
        out.push(line);
    }
}

fn render_pipe_fallback(header: &TableRow, rows: &[TableRow]) -> Vec<HyperlinkLine> {
    std::iter::once(header)
        .chain(rows)
        .map(|row| {
            let text = row
                .cells
                .iter()
                .map(line_text)
                .collect::<Vec<_>>()
                .join(" | ");
            HyperlinkLine::new(Line::from(format!("| {text} |")))
        })
        .collect()
}

fn line_text(line: &HyperlinkLine) -> String {
    line.line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
}

fn longest_token_width(text: &str) -> usize {
    text.split_whitespace()
        .map(display_width)
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminal_hyperlinks::TerminalHyperlink;

    #[test]
    fn width_allocation_preserves_compact_columns_after_token_heavy_columns() {
        let metrics = [
            ColumnMetrics {
                max_width: 100,
                header_token_width: 4,
                body_token_width: 100,
                kind: ColumnKind::TokenHeavy,
            },
            ColumnMetrics {
                max_width: 8,
                header_token_width: 5,
                body_token_width: 8,
                kind: ColumnKind::Compact,
            },
        ];

        assert_eq!(compute_column_widths(&metrics, Some(24)), Some(vec![16, 8]));
    }

    #[test]
    fn wrapped_table_link_fragments_keep_complete_destination() {
        let destination = "https://example.com/a/very/long/path/to/an/artifact";
        let mut url = HyperlinkLine::default();
        url.push_span(destination.into(), Some(destination));
        let table = TableState {
            alignments: vec![Alignment::None, Alignment::None],
            rows: vec![
                TableRow {
                    cells: vec!["Item".into(), "URL".into()],
                    header: true,
                },
                TableRow {
                    cells: vec!["report".into(), url],
                    header: false,
                },
            ],
            current_row: TableRow::default(),
            current_cell: HyperlinkLine::default(),
            in_header: false,
        };

        let rendered = render(table, Some(32));
        let links = rendered
            .iter()
            .flat_map(|line| &line.hyperlinks)
            .collect::<Vec<&TerminalHyperlink>>();
        assert!(links.len() > 1);
        assert!(links.iter().all(|link| link.destination == destination));
        assert!(rendered.iter().all(|line| line.width() <= 32));
    }
}
