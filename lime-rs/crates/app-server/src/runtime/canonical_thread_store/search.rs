use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Range;

use agent_protocol::{
    AgentInput, MessageContentPart, SortDirection, Thread, ThreadId, ThreadItem, ThreadItemPayload,
};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use pulldown_cmark::{Event, Parser, TagEnd};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use thread_store::{
    SearchTextRange, SearchThreadOccurrencesParams, SearchThreadsParams, StoreCursor,
    StoredThreadOccurrence, StoredThreadSearchResult, ThreadOccurrenceSearchPage, ThreadSearchPage,
    ThreadSearchSortKey, ThreadSearchSourceKind, ThreadStoreError, ThreadStoreResult,
};

use super::{decode_json, encode_cursor_with_inclusive, CursorKind, ProjectionStore};

const SNIPPET_CONTEXT_BEFORE_CHARS: usize = 48;
const SNIPPET_CONTEXT_AFTER_CHARS: usize = 96;
const CODEX_USER_MESSAGE_BEGIN: &str = "## My request for Codex:";

#[cfg(test)]
mod tests;

#[derive(Debug)]
struct Candidate {
    item: ThreadItem,
    item_ordinal: i64,
    turn_ordinal: i64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct OccurrenceSearchCursor {
    thread_id: ThreadId,
    search_term: String,
    next_item_ordinal: i64,
    next_occurrence_index: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ThreadSearchCursor {
    search_term: String,
    archived: bool,
    sort_key: ThreadSearchSortKey,
    source_kinds: Vec<ThreadSearchSourceKind>,
    position: i64,
    thread_id: ThreadId,
}

pub(super) fn search_threads(
    store: &ProjectionStore,
    params: SearchThreadsParams,
) -> ThreadStoreResult<ThreadSearchPage> {
    let search_term = params.search_term.trim();
    if search_term.is_empty() {
        return Err(ThreadStoreError::invalid_request(
            "thread/search requires search_term",
        ));
    }
    if params.page_size == 0 {
        return Err(ThreadStoreError::invalid_request(
            "thread/search requires page_size greater than zero",
        ));
    }

    let cursor = parse_thread_search_cursor(params.cursor.as_ref(), &params, search_term)?;
    let conn = store.open_thread_store()?;
    let mut statement = conn
        .prepare(
            "SELECT thread_json FROM canonical_threads
             WHERE archived = ?1
               AND NOT EXISTS (
                    SELECT 1 FROM canonical_thread_spawn_edges AS edge
                    WHERE edge.child_thread_id = canonical_threads.thread_id
                      AND edge.status = 'pending'
               )",
        )
        .map_err(|error| ThreadStoreError::new(error.to_string()))?;
    let mut threads = statement
        .query_map(params![i64::from(params.archived)], |row| {
            row.get::<_, String>(0)
        })
        .map_err(|error| ThreadStoreError::new(error.to_string()))?
        .map(|row| {
            decode_json::<Thread>(&row.map_err(|error| ThreadStoreError::new(error.to_string()))?)
        })
        .collect::<ThreadStoreResult<Vec<_>>>()?;
    drop(statement);

    let matcher = LiteralMatcher::new(search_term);
    let mut matches = Vec::new();
    for mut thread in threads.drain(..) {
        store.enrich_thread_agent_context(&conn, &mut thread)?;
        if !thread_matches_source_kinds(&thread, &params.source_kinds) {
            continue;
        }
        let Some(snippet) = first_thread_match_snippet(&conn, &thread.thread_id, &matcher)? else {
            continue;
        };
        let position = thread_search_position(&thread, params.sort_key);
        if cursor.as_ref().is_some_and(|cursor| {
            !thread_is_after_cursor(position, &thread.thread_id, cursor, params.sort_direction)
        }) {
            continue;
        }
        matches.push((thread, snippet, position));
    }

    matches.sort_by(|left, right| {
        let ordering =
            compare_thread_search_keys(left.2, &left.0.thread_id, right.2, &right.0.thread_id);
        match params.sort_direction {
            SortDirection::Asc => ordering,
            SortDirection::Desc => ordering.reverse(),
        }
    });
    let has_more = matches.len() > params.page_size;
    matches.truncate(params.page_size);
    let next_cursor = has_more
        .then(|| matches.last())
        .flatten()
        .map(|(thread, _, position)| {
            encode_thread_search_cursor(&params, search_term, *position, thread.thread_id.clone())
        })
        .transpose()?;
    let backwards_cursor = matches
        .first()
        .map(|(thread, _, position)| {
            encode_thread_search_cursor(&params, search_term, *position, thread.thread_id.clone())
        })
        .transpose()?;

    Ok(ThreadSearchPage {
        data: matches
            .into_iter()
            .map(|(thread, snippet, _)| StoredThreadSearchResult { thread, snippet })
            .collect(),
        next_cursor,
        backwards_cursor,
    })
}

fn parse_thread_search_cursor(
    cursor: Option<&StoreCursor>,
    params: &SearchThreadsParams,
    search_term: &str,
) -> ThreadStoreResult<Option<ThreadSearchCursor>> {
    let Some(cursor) = cursor else {
        return Ok(None);
    };
    URL_SAFE_NO_PAD
        .decode(cursor.as_str())
        .ok()
        .and_then(|bytes| serde_json::from_slice::<ThreadSearchCursor>(&bytes).ok())
        .filter(|value| {
            value.search_term == search_term
                && value.archived == params.archived
                && value.sort_key == params.sort_key
                && value.source_kinds == params.source_kinds
        })
        .map(Some)
        .ok_or_else(|| invalid_cursor(Some(cursor)))
}

fn encode_thread_search_cursor(
    params: &SearchThreadsParams,
    search_term: &str,
    position: i64,
    thread_id: ThreadId,
) -> ThreadStoreResult<StoreCursor> {
    let bytes = serde_json::to_vec(&ThreadSearchCursor {
        search_term: search_term.to_string(),
        archived: params.archived,
        sort_key: params.sort_key,
        source_kinds: params.source_kinds.clone(),
        position,
        thread_id,
    })
    .map_err(|error| ThreadStoreError::new(error.to_string()))?;
    StoreCursor::new(URL_SAFE_NO_PAD.encode(bytes)).map_err(ThreadStoreError::new)
}

fn thread_search_position(thread: &Thread, sort_key: ThreadSearchSortKey) -> i64 {
    match sort_key {
        ThreadSearchSortKey::CreatedAt => thread.created_at_ms,
        ThreadSearchSortKey::UpdatedAt => thread.updated_at_ms,
        ThreadSearchSortKey::RecencyAt => thread.recency_at_ms.unwrap_or(thread.updated_at_ms),
    }
}

fn thread_is_after_cursor(
    position: i64,
    thread_id: &ThreadId,
    cursor: &ThreadSearchCursor,
    direction: SortDirection,
) -> bool {
    let ordering =
        compare_thread_search_keys(position, thread_id, cursor.position, &cursor.thread_id);
    match direction {
        SortDirection::Asc => ordering == Ordering::Greater,
        SortDirection::Desc => ordering == Ordering::Less,
    }
}

fn compare_thread_search_keys(
    left_position: i64,
    left_id: &ThreadId,
    right_position: i64,
    right_id: &ThreadId,
) -> Ordering {
    left_position
        .cmp(&right_position)
        .then_with(|| left_id.as_str().cmp(right_id.as_str()))
}

fn first_thread_match_snippet(
    conn: &Connection,
    thread_id: &ThreadId,
    matcher: &LiteralMatcher,
) -> ThreadStoreResult<Option<String>> {
    let mut statement = conn
        .prepare(
            "SELECT item_json FROM canonical_items
             WHERE thread_id = ?1 ORDER BY ordinal ASC, item_id ASC",
        )
        .map_err(|error| ThreadStoreError::new(error.to_string()))?;
    let items = statement
        .query_map(params![thread_id.as_str()], |row| row.get::<_, String>(0))
        .map_err(|error| ThreadStoreError::new(error.to_string()))?;
    for item in items {
        let item: ThreadItem =
            decode_json(&item.map_err(|error| ThreadStoreError::new(error.to_string()))?)?;
        let Some(text) = thread_search_text(&item) else {
            continue;
        };
        let Some(matched) = matcher.find_ranges(&text).into_iter().next() else {
            continue;
        };
        return Ok(thread_search_snippet(&text, matched));
    }
    Ok(None)
}

fn thread_search_text(item: &ThreadItem) -> Option<String> {
    let text = match &item.payload {
        ThreadItemPayload::UserMessage { content, .. } => content
            .iter()
            .filter_map(|input| match input {
                AgentInput::Text { text, .. } => Some(strip_user_message_prefix(text)),
                AgentInput::Image { .. }
                | AgentInput::LocalImage { .. }
                | AgentInput::Skill { .. }
                | AgentInput::Mention { .. } => None,
            })
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" "),
        ThreadItemPayload::AgentMessage {
            text,
            content_parts,
            ..
        } => {
            if text.trim().is_empty() {
                content_parts
                    .iter()
                    .filter_map(|part| match part {
                        MessageContentPart::Text { text } => Some(text.as_str()),
                        MessageContentPart::Media { .. } => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                text.clone()
            }
        }
        _ => return None,
    };
    let text = normalize_preview_text(&text);
    (!text.is_empty()).then_some(text)
}

fn thread_search_snippet(text: &str, matched: Range<usize>) -> Option<String> {
    let snippet_start = char_start_before(text, matched.start, SNIPPET_CONTEXT_BEFORE_CHARS);
    let snippet_end = char_end_after(text, matched.end, SNIPPET_CONTEXT_AFTER_CHARS);
    let excerpt = text[snippet_start..snippet_end].trim();
    if excerpt.is_empty() {
        return None;
    }
    let mut snippet = String::new();
    if snippet_start > 0 {
        snippet.push_str("... ");
    }
    snippet.push_str(excerpt);
    if snippet_end < text.len() {
        snippet.push_str(" ...");
    }
    Some(snippet)
}

fn normalize_preview_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn thread_matches_source_kinds(thread: &Thread, source_kinds: &[ThreadSearchSourceKind]) -> bool {
    if source_kinds.is_empty() {
        return true;
    }
    let actual = thread_source_kind(thread);
    source_kinds.iter().any(|requested| {
        *requested == actual
            || (*requested == ThreadSearchSourceKind::SubAgent
                && matches!(
                    actual,
                    ThreadSearchSourceKind::SubAgent
                        | ThreadSearchSourceKind::SubAgentReview
                        | ThreadSearchSourceKind::SubAgentCompact
                        | ThreadSearchSourceKind::SubAgentThreadSpawn
                        | ThreadSearchSourceKind::SubAgentOther
                ))
    })
}

fn thread_source_kind(thread: &Thread) -> ThreadSearchSourceKind {
    let source = thread
        .metadata
        .as_object()
        .and_then(|metadata| {
            ["sourceKind", "source_kind", "source", "threadSource"]
                .into_iter()
                .find_map(|key| metadata.get(key).and_then(serde_json::Value::as_str))
        })
        .unwrap_or("appServer");
    let normalized = normalize_source(source);
    match normalized.as_str() {
        "cli" => ThreadSearchSourceKind::Cli,
        "vscode" => ThreadSearchSourceKind::VsCode,
        "exec" => ThreadSearchSourceKind::Exec,
        "appserver" | "mcp" => ThreadSearchSourceKind::AppServer,
        "subagentreview" => ThreadSearchSourceKind::SubAgentReview,
        "subagentcompact" => ThreadSearchSourceKind::SubAgentCompact,
        "subagentthreadspawn" => ThreadSearchSourceKind::SubAgentThreadSpawn,
        "subagentother" => ThreadSearchSourceKind::SubAgentOther,
        "subagent" => ThreadSearchSourceKind::SubAgent,
        _ if thread.parent_thread_id.is_some() => match thread
            .agent_role
            .as_deref()
            .map(normalize_source)
            .as_deref()
        {
            Some("review") => ThreadSearchSourceKind::SubAgentReview,
            Some("compact") => ThreadSearchSourceKind::SubAgentCompact,
            Some("threadspawn") => ThreadSearchSourceKind::SubAgentThreadSpawn,
            Some("") | None => ThreadSearchSourceKind::SubAgent,
            Some(_) => ThreadSearchSourceKind::SubAgentOther,
        },
        _ => ThreadSearchSourceKind::Unknown,
    }
}

fn normalize_source(source: &str) -> String {
    source
        .chars()
        .filter(|value| value.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
}

pub(super) fn search_thread_occurrences(
    store: &ProjectionStore,
    params: SearchThreadOccurrencesParams,
) -> ThreadStoreResult<ThreadOccurrenceSearchPage> {
    if params.search_term.trim().is_empty() {
        return Err(ThreadStoreError::invalid_request(
            "thread/searchOccurrences requires search_term",
        ));
    }
    if params.page_size == 0 {
        return Err(ThreadStoreError::invalid_request(
            "thread/searchOccurrences requires page_size greater than zero",
        ));
    }

    let conn = store.open_thread_store()?;
    let exists = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM canonical_threads WHERE thread_id = ?1)",
            params![params.thread_id.as_str()],
            |row| row.get::<_, bool>(0),
        )
        .map_err(|error| ThreadStoreError::new(error.to_string()))?;
    if !exists {
        return Err(ThreadStoreError::thread_not_found(format!(
            "no rollout found for thread id {}",
            params.thread_id
        )));
    }

    let cursor = parse_cursor(
        params.cursor.as_ref(),
        &params.thread_id,
        &params.search_term,
    )?;
    let mut statement = conn
        .prepare(
            "SELECT items.item_json, items.ordinal, turns.ordinal
             FROM canonical_items AS items
             JOIN canonical_turns AS turns
               ON turns.thread_id = items.thread_id
              AND turns.turn_id = items.turn_id
             WHERE items.thread_id = ?1
             ORDER BY items.ordinal ASC, items.item_id ASC",
        )
        .map_err(|error| ThreadStoreError::new(error.to_string()))?;
    let candidates = statement
        .query_map(params![params.thread_id.as_str()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })
        .map_err(|error| ThreadStoreError::new(error.to_string()))?
        .map(|row| {
            let (item_json, item_ordinal, turn_ordinal) =
                row.map_err(|error| ThreadStoreError::new(error.to_string()))?;
            Ok(Candidate {
                item: decode_json(&item_json)?,
                item_ordinal,
                turn_ordinal,
            })
        })
        .collect::<ThreadStoreResult<Vec<_>>>()?;

    if let Some(cursor) = cursor.as_ref() {
        let max_ordinal = candidates
            .last()
            .map_or(0, |candidate| candidate.item_ordinal);
        if cursor.next_item_ordinal < 0 || cursor.next_item_ordinal > max_ordinal {
            return Err(invalid_cursor(params.cursor.as_ref()));
        }
    }

    let mut latest_final_ordinal = HashMap::new();
    for candidate in &candidates {
        if is_final_agent_message(&candidate.item) {
            latest_final_ordinal.insert(candidate.item.turn_id.clone(), candidate.item_ordinal);
        }
    }

    let matcher = LiteralMatcher::new(&params.search_term);
    let mut data = Vec::with_capacity(params.page_size);
    for candidate in candidates {
        let start_occurrence_index = match cursor.as_ref() {
            Some(cursor) if candidate.item_ordinal < cursor.next_item_ordinal => continue,
            Some(cursor) if candidate.item_ordinal == cursor.next_item_ordinal => {
                cursor.next_occurrence_index
            }
            _ => 0,
        };
        let Some(text) = searchable_text(
            &candidate.item,
            latest_final_ordinal.get(&candidate.item.turn_id).copied(),
            candidate.item_ordinal,
        ) else {
            continue;
        };
        let ranges = matcher.find_ranges(text.as_ref());
        let turn_cursor = encode_cursor_with_inclusive(
            CursorKind::Turns,
            candidate.turn_ordinal,
            candidate.item.turn_id.as_str(),
            true,
        )?;
        for (occurrence_index, matched) in
            ranges.into_iter().enumerate().skip(start_occurrence_index)
        {
            if data.len() == params.page_size {
                return Ok(ThreadOccurrenceSearchPage {
                    data,
                    next_cursor: Some(encode_search_cursor(OccurrenceSearchCursor {
                        thread_id: params.thread_id,
                        search_term: params.search_term,
                        next_item_ordinal: candidate.item_ordinal,
                        next_occurrence_index: occurrence_index,
                    })?),
                });
            }
            data.push(occurrence_in_item(
                &candidate.item,
                text.as_ref(),
                matched,
                turn_cursor.clone(),
            ));
        }
    }

    Ok(ThreadOccurrenceSearchPage {
        data,
        next_cursor: None,
    })
}

fn parse_cursor(
    cursor: Option<&StoreCursor>,
    thread_id: &ThreadId,
    search_term: &str,
) -> ThreadStoreResult<Option<OccurrenceSearchCursor>> {
    let Some(cursor) = cursor else {
        return Ok(None);
    };
    let decoded = URL_SAFE_NO_PAD
        .decode(cursor.as_str())
        .ok()
        .and_then(|bytes| serde_json::from_slice::<OccurrenceSearchCursor>(&bytes).ok())
        .filter(|value| {
            &value.thread_id == thread_id
                && value.search_term == search_term
                && value.next_item_ordinal >= 0
        })
        .ok_or_else(|| invalid_cursor(Some(cursor)))?;
    Ok(Some(decoded))
}

fn encode_search_cursor(cursor: OccurrenceSearchCursor) -> ThreadStoreResult<StoreCursor> {
    let bytes =
        serde_json::to_vec(&cursor).map_err(|error| ThreadStoreError::new(error.to_string()))?;
    let encoded = URL_SAFE_NO_PAD.encode(bytes);
    StoreCursor::new(encoded).map_err(ThreadStoreError::new)
}

fn invalid_cursor(cursor: Option<&StoreCursor>) -> ThreadStoreError {
    ThreadStoreError::invalid_request(format!(
        "invalid cursor: {}",
        cursor.map_or("", StoreCursor::as_str)
    ))
}

fn searchable_text(
    item: &ThreadItem,
    latest_final_ordinal: Option<i64>,
    item_ordinal: i64,
) -> Option<Cow<'_, str>> {
    match &item.payload {
        ThreadItemPayload::UserMessage { content, .. } => {
            let mut text_parts = content
                .iter()
                .filter_map(|input| match input {
                    AgentInput::Text { text, .. } => Some(strip_user_message_prefix(text)),
                    AgentInput::Image { .. }
                    | AgentInput::LocalImage { .. }
                    | AgentInput::Skill { .. }
                    | AgentInput::Mention { .. } => None,
                })
                .filter(|text| !text.is_empty())
                .peekable();
            let first = text_parts.next()?;
            match text_parts.next() {
                None => Some(Cow::Borrowed(first)),
                Some(second) => {
                    let mut parts = vec![first, second];
                    parts.extend(text_parts);
                    Some(Cow::Owned(parts.concat()))
                }
            }
        }
        ThreadItemPayload::AgentMessage {
            text,
            content_parts,
            ..
        } if latest_final_ordinal == Some(item_ordinal) => {
            let markdown = if text.is_empty() {
                content_parts
                    .iter()
                    .filter_map(|part| match part {
                        MessageContentPart::Text { text } => Some(text.as_str()),
                        MessageContentPart::Media { .. } => None,
                    })
                    .collect::<String>()
            } else {
                text.clone()
            };
            let text = markdown_to_search_text(&markdown);
            (!text.is_empty()).then_some(Cow::Owned(text))
        }
        _ => None,
    }
}

fn is_final_agent_message(item: &ThreadItem) -> bool {
    let ThreadItemPayload::AgentMessage { phase, .. } = &item.payload else {
        return false;
    };
    phase.as_deref().is_some_and(|phase| {
        let normalized = phase.trim().to_ascii_lowercase().replace('-', "_");
        matches!(normalized.as_str(), "final" | "final_answer")
    })
}

fn strip_user_message_prefix(text: &str) -> &str {
    text.find(CODEX_USER_MESSAGE_BEGIN).map_or_else(
        || text.trim(),
        |index| text[index + CODEX_USER_MESSAGE_BEGIN.len()..].trim(),
    )
}

fn markdown_to_search_text(markdown: &str) -> String {
    let mut text = String::new();
    for event in Parser::new(markdown.trim()) {
        match event {
            Event::Text(value)
            | Event::Code(value)
            | Event::Html(value)
            | Event::InlineHtml(value) => text.push_str(&value),
            Event::SoftBreak | Event::HardBreak | Event::Rule => text.push(' '),
            Event::End(
                TagEnd::Paragraph
                | TagEnd::Heading(_)
                | TagEnd::BlockQuote
                | TagEnd::CodeBlock
                | TagEnd::List(_)
                | TagEnd::Item
                | TagEnd::Table
                | TagEnd::TableHead
                | TagEnd::TableRow
                | TagEnd::TableCell,
            ) => text.push(' '),
            Event::Start(_)
            | Event::End(
                TagEnd::Emphasis
                | TagEnd::Strong
                | TagEnd::Strikethrough
                | TagEnd::Link
                | TagEnd::HtmlBlock
                | TagEnd::FootnoteDefinition
                | TagEnd::Image
                | TagEnd::MetadataBlock(_),
            )
            | Event::FootnoteReference(_)
            | Event::TaskListMarker(_) => {}
        }
    }
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

struct LiteralMatcher {
    lowercase_needle: String,
}

impl LiteralMatcher {
    fn new(needle: &str) -> Self {
        Self {
            lowercase_needle: needle.to_lowercase(),
        }
    }

    fn find_ranges(&self, text: &str) -> Vec<Range<usize>> {
        let lowercase_text = text.to_lowercase();
        let mut spans = Vec::with_capacity(text.chars().count());
        let mut lowercase_start = 0;
        for (original_start, character) in text.char_indices() {
            let lowercase_end =
                lowercase_start + character.to_lowercase().map(char::len_utf8).sum::<usize>();
            spans.push((
                lowercase_start..lowercase_end,
                original_start..original_start + character.len_utf8(),
            ));
            lowercase_start = lowercase_end;
        }

        lowercase_text
            .match_indices(&self.lowercase_needle)
            .filter_map(|(start, matched)| {
                let end = start.saturating_add(matched.len());
                let original_start = spans
                    .iter()
                    .find(|(lowercase, _)| lowercase.contains(&start))?
                    .1
                    .start;
                let original_end = spans
                    .iter()
                    .find(|(lowercase, _)| lowercase.contains(&end.saturating_sub(1)))?
                    .1
                    .end;
                Some(original_start..original_end)
            })
            .collect()
    }
}

fn occurrence_in_item(
    item: &ThreadItem,
    text: &str,
    matched: Range<usize>,
    turn_cursor: StoreCursor,
) -> StoredThreadOccurrence {
    let snippet_start = char_start_before(text, matched.start, SNIPPET_CONTEXT_BEFORE_CHARS);
    let snippet_end = char_end_after(text, matched.end, SNIPPET_CONTEXT_AFTER_CHARS);
    let leading_ellipsis = snippet_start > 0;
    let trailing_ellipsis = snippet_end < text.len();
    let mut snippet = String::new();
    if leading_ellipsis {
        snippet.push_str("... ");
    }
    snippet.push_str(&text[snippet_start..snippet_end]);
    if trailing_ellipsis {
        snippet.push_str(" ...");
    }
    let snippet_match_start =
        u32::from(leading_ellipsis) * 4 + utf16_len(&text[snippet_start..matched.start]);
    let match_len = utf16_len(&text[matched]);

    StoredThreadOccurrence {
        turn_id: item.turn_id.clone(),
        item_id: item.item_id.clone(),
        snippet,
        snippet_match_range: SearchTextRange {
            start: snippet_match_start,
            end: snippet_match_start.saturating_add(match_len),
        },
        turn_cursor,
    }
}

fn utf16_len(text: &str) -> u32 {
    u32::try_from(text.encode_utf16().count()).unwrap_or(u32::MAX)
}

fn char_start_before(text: &str, byte_index: usize, chars_before: usize) -> usize {
    text[..byte_index]
        .char_indices()
        .rev()
        .nth(chars_before)
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn char_end_after(text: &str, byte_index: usize, chars_after: usize) -> usize {
    text[byte_index..]
        .char_indices()
        .nth(chars_after)
        .map(|(offset, _)| byte_index.saturating_add(offset))
        .unwrap_or(text.len())
}
