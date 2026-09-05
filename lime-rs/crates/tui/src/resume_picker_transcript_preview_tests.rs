use super::*;
use app_server_protocol::protocol::v2::ThreadItem;

fn user_item(text: &str) -> ThreadItem {
    ThreadItem::UserMessage {
        id: String::from("user"),
        metadata: None,
        client_id: None,
        content: vec![UserInput::Text {
            text: text.to_string(),
            text_elements: Vec::new(),
        }],
    }
}

fn assistant_item(text: &str) -> ThreadItem {
    ThreadItem::AgentMessage {
        id: String::from("assistant"),
        metadata: None,
        text: text.to_string(),
        phase: None,
        memory_citation: None,
        delivery: None,
    }
}

#[test]
fn preview_keeps_newest_lines_and_restores_transcript_order() {
    let mut lines = Vec::new();
    append_item_preview_lines(&mut lines, &assistant_item("third\n\nfourth"));
    append_item_preview_lines(&mut lines, &user_item("first\nsecond"));
    lines.reverse();

    assert_eq!(
        lines,
        vec![
            TranscriptPreviewLine {
                speaker: TranscriptPreviewSpeaker::User,
                text: String::from("first"),
            },
            TranscriptPreviewLine {
                speaker: TranscriptPreviewSpeaker::User,
                text: String::from("second"),
            },
            TranscriptPreviewLine {
                speaker: TranscriptPreviewSpeaker::Assistant,
                text: String::from("third"),
            },
            TranscriptPreviewLine {
                speaker: TranscriptPreviewSpeaker::Assistant,
                text: String::from("fourth"),
            },
        ]
    );
}

#[test]
fn preview_ignores_non_text_inputs_and_blank_lines() {
    let mut lines = Vec::new();
    append_item_preview_lines(
        &mut lines,
        &ThreadItem::UserMessage {
            id: String::from("user"),
            metadata: None,
            client_id: None,
            content: vec![
                UserInput::Image {
                    detail: None,
                    url: String::from("https://example.test/image.png"),
                },
                UserInput::Text {
                    text: String::from("  visible  \n\n"),
                    text_elements: Vec::new(),
                },
            ],
        },
    );

    assert_eq!(
        lines,
        vec![TranscriptPreviewLine {
            speaker: TranscriptPreviewSpeaker::User,
            text: String::from("visible"),
        }]
    );
}

#[test]
fn preview_is_bounded_to_six_lines() {
    let mut lines = Vec::new();
    append_item_preview_lines(
        &mut lines,
        &assistant_item("one\ntwo\nthree\nfour\nfive\nsix\nseven"),
    );

    assert_eq!(lines.len(), MAX_TRANSCRIPT_PREVIEW_LINES);
    assert_eq!(lines[0].text, "seven");
    assert_eq!(lines[5].text, "two");
}

#[test]
fn preview_from_entries_returns_newest_lines_in_transcript_order() {
    let lines = preview_from_entries(vec![
        (TranscriptPreviewSpeaker::User, String::from("first")),
        (
            TranscriptPreviewSpeaker::Assistant,
            String::from("second\nthird"),
        ),
    ])
    .expect("preview");

    assert_eq!(
        lines,
        vec![
            TranscriptPreviewLine {
                speaker: TranscriptPreviewSpeaker::User,
                text: String::from("first"),
            },
            TranscriptPreviewLine {
                speaker: TranscriptPreviewSpeaker::Assistant,
                text: String::from("second"),
            },
            TranscriptPreviewLine {
                speaker: TranscriptPreviewSpeaker::Assistant,
                text: String::from("third"),
            },
        ]
    );
}
