#[cfg(test)]
use app_server_protocol::protocol::v2::{ThreadItem, UserInput};
use std::io;

use crate::app_server_session::AppServerSession;

const MAX_TRANSCRIPT_PREVIEW_LINES: usize = 6;

/// A bounded, display-only transcript line used by the resume picker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TranscriptPreviewLine {
    pub(crate) speaker: TranscriptPreviewSpeaker,
    pub(crate) text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TranscriptPreviewSpeaker {
    User,
    Assistant,
}

/// Load a bounded preview from the canonical App Server thread projection.
#[allow(dead_code)]
pub(crate) async fn load_transcript_preview(
    app_server: &AppServerSession,
    thread_id: &str,
) -> io::Result<Vec<TranscriptPreviewLine>> {
    let entries = crate::thread_transcript::load_session_transcript(app_server, thread_id).await?;
    let entries = entries
        .into_iter()
        .filter_map(|entry| {
            let speaker = match entry.kind {
                crate::projection::EntryKind::User => TranscriptPreviewSpeaker::User,
                crate::projection::EntryKind::Assistant => TranscriptPreviewSpeaker::Assistant,
                _ => return None,
            };
            Some((speaker, entry.text))
        })
        .collect();
    preview_from_entries(entries)
}

pub(crate) fn preview_from_entries(
    entries: Vec<(TranscriptPreviewSpeaker, String)>,
) -> io::Result<Vec<TranscriptPreviewLine>> {
    let mut lines = Vec::with_capacity(MAX_TRANSCRIPT_PREVIEW_LINES);
    for (speaker, text) in entries.into_iter().rev() {
        for text in text.lines().rev() {
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            lines.push(TranscriptPreviewLine {
                speaker,
                text: text.to_string(),
            });
            if lines.len() == MAX_TRANSCRIPT_PREVIEW_LINES {
                break;
            }
        }
        if lines.len() == MAX_TRANSCRIPT_PREVIEW_LINES {
            break;
        }
    }

    lines.reverse();
    Ok(lines)
}

#[cfg(test)]
fn append_item_preview_lines(lines: &mut Vec<TranscriptPreviewLine>, item: &ThreadItem) {
    let (speaker, text) = match item {
        ThreadItem::UserMessage { content, .. } => (
            TranscriptPreviewSpeaker::User,
            content
                .iter()
                .filter_map(|input| match input {
                    UserInput::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        ),
        ThreadItem::AgentMessage { text, .. } => {
            (TranscriptPreviewSpeaker::Assistant, text.clone())
        }
        _ => return,
    };

    let remaining = MAX_TRANSCRIPT_PREVIEW_LINES.saturating_sub(lines.len());
    lines.extend(
        text.lines()
            .rev()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .take(remaining)
            .map(|line| TranscriptPreviewLine {
                speaker,
                text: line.to_string(),
            }),
    );
}

#[cfg(test)]
#[path = "resume_picker_transcript_preview_tests.rs"]
mod tests;
