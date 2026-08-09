use super::patch::patch_declined;
use super::TrackedTool;
use crate::RuntimeEvent;
use agent_protocol::ThreadItem;
use lime_agent::{AgentEvent as RuntimeAgentEvent, AgentToolResult};
use serde::Deserialize;
use serde_json::{json, Value};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use tool_runtime::apply_patch::TURN_DIFF_DELTA_METADATA_KEY;

const DIFF_CONTEXT_LINES: usize = 3;
const DEV_NULL: &str = "/dev/null";

#[derive(Clone, Debug)]
pub(super) struct TurnDiffTracker {
    valid: bool,
    baseline_by_path: HashMap<String, String>,
    current_by_path: HashMap<String, String>,
    origin_by_current_path: HashMap<String, String>,
    unified_diff: Option<String>,
}

impl Default for TurnDiffTracker {
    fn default() -> Self {
        Self {
            valid: true,
            baseline_by_path: HashMap::new(),
            current_by_path: HashMap::new(),
            origin_by_current_path: HashMap::new(),
            unified_diff: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct TurnDiffDelta {
    changes: Vec<TurnDiffChange>,
}

#[derive(Debug, Deserialize)]
#[serde(
    deny_unknown_fields,
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
enum TurnDiffChange {
    Add {
        path: String,
        content: String,
        #[serde(default)]
        overwritten_content: Option<String>,
    },
    Delete {
        path: String,
        content: String,
    },
    Update {
        path: String,
        #[serde(default)]
        move_path: Option<String>,
        old_content: String,
        #[serde(default)]
        overwritten_move_content: Option<String>,
        new_content: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ContentLine<'a> {
    text: &'a str,
    terminated: bool,
}

impl TurnDiffTracker {
    pub(super) fn update_from_result(
        &mut self,
        tool: &TrackedTool,
        result: &AgentToolResult,
    ) -> Option<RuntimeEvent> {
        if tool.patch_id.is_none() {
            return None;
        }

        let had_unified_diff = self.unified_diff.is_some();
        let changed = if result.success {
            match result
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get(TURN_DIFF_DELTA_METADATA_KEY))
            {
                Some(delta) => self.track_delta(delta),
                None => self.invalidate(),
            }
        } else if patch_declined(result) {
            false
        } else {
            self.invalidate()
        };

        if !changed || (!had_unified_diff && self.unified_diff.is_none()) {
            return None;
        }

        Some(RuntimeEvent::new(
            "turn.diff.updated",
            json!({ "diff": self.unified_diff.clone().unwrap_or_default() }),
        ))
    }

    fn track_delta(&mut self, value: &Value) -> bool {
        if !self.valid {
            return false;
        }
        let Ok(delta) = serde_json::from_value::<TurnDiffDelta>(value.clone()) else {
            return self.invalidate();
        };
        if delta.changes.is_empty() {
            return false;
        }

        let mut next = self.clone();
        if delta
            .changes
            .iter()
            .try_for_each(|change| next.apply_change(change))
            .is_err()
        {
            return self.invalidate();
        }
        next.refresh_unified_diff();
        *self = next;
        true
    }

    fn invalidate(&mut self) -> bool {
        if !self.valid {
            return false;
        }
        self.valid = false;
        self.unified_diff = None;
        true
    }

    fn apply_change(&mut self, change: &TurnDiffChange) -> Result<(), ()> {
        match change {
            TurnDiffChange::Add {
                path,
                content,
                overwritten_content,
            } => self.apply_add(path, content, overwritten_content.as_deref()),
            TurnDiffChange::Delete { path, content } => self.apply_delete(path, content),
            TurnDiffChange::Update {
                path,
                move_path,
                old_content,
                overwritten_move_content,
                new_content,
            } => self.apply_update(
                path,
                move_path.as_deref(),
                old_content,
                overwritten_move_content.as_deref(),
                new_content,
            ),
        }
    }

    fn apply_add(
        &mut self,
        path: &str,
        content: &str,
        overwritten_content: Option<&str>,
    ) -> Result<(), ()> {
        let path = normalize_path(path)?;
        self.origin_by_current_path.remove(&path);
        match self.current_by_path.get(&path) {
            Some(current) if overwritten_content == Some(current.as_str()) => {}
            Some(_) => return Err(()),
            None => match overwritten_content {
                Some(previous) if !self.baseline_by_path.contains_key(&path) => {
                    self.baseline_by_path
                        .insert(path.clone(), previous.to_string());
                }
                Some(_) => return Err(()),
                None => {}
            },
        }
        self.current_by_path.insert(path, content.to_string());
        Ok(())
    }

    fn apply_delete(&mut self, path: &str, content: &str) -> Result<(), ()> {
        let path = normalize_path(path)?;
        match self.current_by_path.remove(&path) {
            Some(current) if current == content => {}
            Some(_) => return Err(()),
            None if !self.baseline_by_path.contains_key(&path) => {
                self.baseline_by_path
                    .insert(path.clone(), content.to_string());
            }
            None => return Err(()),
        }
        self.origin_by_current_path.remove(&path);
        Ok(())
    }

    fn apply_update(
        &mut self,
        path: &str,
        move_path: Option<&str>,
        old_content: &str,
        overwritten_move_content: Option<&str>,
        new_content: &str,
    ) -> Result<(), ()> {
        let source_path = normalize_path(path)?;
        match self.current_by_path.get(&source_path) {
            Some(current) if current == old_content => {}
            Some(_) => return Err(()),
            None if !self.baseline_by_path.contains_key(&source_path) => {
                self.baseline_by_path
                    .insert(source_path.clone(), old_content.to_string());
            }
            None => return Err(()),
        }

        let Some(move_path) = move_path else {
            self.current_by_path
                .insert(source_path, new_content.to_string());
            return Ok(());
        };
        let destination_path = normalize_path(move_path)?;
        if destination_path == source_path {
            return Err(());
        }

        match self.current_by_path.get(&destination_path) {
            Some(current) if overwritten_move_content == Some(current.as_str()) => {}
            Some(_) => return Err(()),
            None => match overwritten_move_content {
                Some(previous) if !self.baseline_by_path.contains_key(&destination_path) => {
                    self.baseline_by_path
                        .insert(destination_path.clone(), previous.to_string());
                }
                Some(_) => return Err(()),
                None => {}
            },
        }

        let origin = self
            .origin_by_current_path
            .remove(&source_path)
            .unwrap_or_else(|| source_path.clone());
        self.current_by_path.remove(&source_path);
        self.current_by_path
            .insert(destination_path.clone(), new_content.to_string());
        self.origin_by_current_path.remove(&destination_path);
        if destination_path != origin {
            self.origin_by_current_path.insert(destination_path, origin);
        }
        Ok(())
    }

    fn refresh_unified_diff(&mut self) {
        let rename_pairs = self.rename_pairs();
        let paired_destinations = rename_pairs.values().cloned().collect::<HashSet<_>>();
        let mut paths = self
            .baseline_by_path
            .keys()
            .chain(self.current_by_path.keys())
            .cloned()
            .collect::<Vec<_>>();
        paths.sort();
        paths.dedup();

        let mut handled = HashSet::new();
        let mut unified_diff = String::new();
        for path in paths {
            if !handled.insert(path.clone()) || paired_destinations.contains(&path) {
                continue;
            }
            let (left_path, right_path) = rename_pairs
                .get(&path)
                .map(|destination| (path.as_str(), destination.as_str()))
                .unwrap_or((path.as_str(), path.as_str()));
            handled.insert(right_path.to_string());
            if let Some(diff) = render_diff(
                left_path,
                self.baseline_by_path.get(left_path).map(String::as_str),
                right_path,
                self.current_by_path.get(right_path).map(String::as_str),
            ) {
                unified_diff.push_str(&diff);
                if !unified_diff.ends_with('\n') {
                    unified_diff.push('\n');
                }
            }
        }
        self.unified_diff = (!unified_diff.is_empty()).then_some(unified_diff);
    }

    fn rename_pairs(&self) -> HashMap<String, String> {
        self.origin_by_current_path
            .iter()
            .filter_map(|(destination, origin)| {
                if destination == origin
                    || self.current_by_path.contains_key(origin)
                    || !self.current_by_path.contains_key(destination)
                    || !self.baseline_by_path.contains_key(origin)
                    || self.baseline_by_path.contains_key(destination)
                {
                    None
                } else {
                    Some((origin.clone(), destination.clone()))
                }
            })
            .collect()
    }
}

pub(super) fn runtime_event_for_persistence<'a>(
    event: &'a RuntimeAgentEvent,
) -> Cow<'a, RuntimeAgentEvent> {
    match event {
        RuntimeAgentEvent::ItemStarted { item } if has_internal_delta(item) => {
            Cow::Owned(RuntimeAgentEvent::ItemStarted {
                item: remove_internal_delta(item.clone()),
            })
        }
        RuntimeAgentEvent::ItemUpdated { item } if has_internal_delta(item) => {
            Cow::Owned(RuntimeAgentEvent::ItemUpdated {
                item: remove_internal_delta(item.clone()),
            })
        }
        RuntimeAgentEvent::ItemCompleted { item } if has_internal_delta(item) => {
            Cow::Owned(RuntimeAgentEvent::ItemCompleted {
                item: remove_internal_delta(item.clone()),
            })
        }
        _ => Cow::Borrowed(event),
    }
}

fn has_internal_delta(item: &ThreadItem) -> bool {
    item.metadata
        .as_object()
        .is_some_and(|metadata| metadata.contains_key(TURN_DIFF_DELTA_METADATA_KEY))
}

fn remove_internal_delta(mut item: ThreadItem) -> ThreadItem {
    if let Some(metadata) = item.metadata.as_object_mut() {
        metadata.remove(TURN_DIFF_DELTA_METADATA_KEY);
    }
    item
}

fn normalize_path(path: &str) -> Result<String, ()> {
    let normalized = path.replace('\\', "/");
    (!normalized.trim().is_empty())
        .then_some(normalized)
        .ok_or(())
}

fn render_diff(
    left_path: &str,
    left_content: Option<&str>,
    right_path: &str,
    right_content: Option<&str>,
) -> Option<String> {
    if left_content == right_content && left_path == right_path {
        return None;
    }
    if left_content == right_content {
        return Some(format!(
            "diff --git a/{left_path} b/{right_path}\nsimilarity index 100%\nrename from {left_path}\nrename to {right_path}\n"
        ));
    }

    let mut diff = format!("diff --git a/{left_path} b/{right_path}\n");
    match (left_content, right_content) {
        (None, Some(_)) => diff.push_str("new file mode 100644\n"),
        (Some(_), None) => diff.push_str("deleted file mode 100644\n"),
        (Some(_), Some(_)) => {}
        (None, None) => return None,
    }
    diff.push_str("--- ");
    diff.push_str(
        &left_content
            .map(|_| format!("a/{left_path}"))
            .unwrap_or_else(|| DEV_NULL.to_string()),
    );
    diff.push('\n');
    diff.push_str("+++ ");
    diff.push_str(
        &right_content
            .map(|_| format!("b/{right_path}"))
            .unwrap_or_else(|| DEV_NULL.to_string()),
    );
    diff.push('\n');
    append_coarse_unified_hunk(
        &mut diff,
        left_content.unwrap_or_default(),
        right_content.unwrap_or_default(),
    );
    Some(diff)
}

fn append_coarse_unified_hunk(diff: &mut String, previous: &str, current: &str) {
    let previous_lines = content_lines(previous);
    let current_lines = content_lines(current);
    let mut prefix_len = 0;
    while previous_lines.get(prefix_len) == current_lines.get(prefix_len)
        && prefix_len < previous_lines.len()
        && prefix_len < current_lines.len()
    {
        prefix_len += 1;
    }
    let mut suffix_len = 0;
    while suffix_len + prefix_len < previous_lines.len()
        && suffix_len + prefix_len < current_lines.len()
        && previous_lines[previous_lines.len() - suffix_len - 1]
            == current_lines[current_lines.len() - suffix_len - 1]
    {
        suffix_len += 1;
    }

    let before_context = prefix_len.min(DIFF_CONTEXT_LINES);
    let after_context = suffix_len.min(DIFF_CONTEXT_LINES);
    let previous_change_end = previous_lines.len() - suffix_len;
    let current_change_end = current_lines.len() - suffix_len;
    let previous_start_index = prefix_len - before_context;
    let current_start_index = prefix_len - before_context;
    let previous_count = before_context + (previous_change_end - prefix_len) + after_context;
    let current_count = before_context + (current_change_end - prefix_len) + after_context;

    if previous_count == 0 && current_count == 0 {
        return;
    }
    let previous_start = if previous_count == 0 {
        previous_start_index
    } else {
        previous_start_index + 1
    };
    let current_start = if current_count == 0 {
        current_start_index
    } else {
        current_start_index + 1
    };
    diff.push_str(&format!(
        "@@ -{previous_start},{previous_count} +{current_start},{current_count} @@\n"
    ));

    for line in &previous_lines[previous_start_index..prefix_len] {
        append_diff_line(diff, ' ', *line);
    }
    for line in &previous_lines[prefix_len..previous_change_end] {
        append_diff_line(diff, '-', *line);
    }
    for line in &current_lines[prefix_len..current_change_end] {
        append_diff_line(diff, '+', *line);
    }
    for line in &previous_lines[previous_change_end..previous_change_end + after_context] {
        append_diff_line(diff, ' ', *line);
    }
}

fn content_lines(content: &str) -> Vec<ContentLine<'_>> {
    content
        .split_inclusive('\n')
        .map(|line| match line.strip_suffix('\n') {
            Some(text) => ContentLine {
                text,
                terminated: true,
            },
            None => ContentLine {
                text: line,
                terminated: false,
            },
        })
        .collect()
}

fn append_diff_line(diff: &mut String, prefix: char, line: ContentLine<'_>) {
    diff.push(prefix);
    diff.push_str(line.text);
    diff.push('\n');
    if !line.terminated {
        diff.push_str("\\ No newline at end of file\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_protocol::{ItemId, ItemStatus, SessionId, ThreadId, ThreadItemPayload, TurnId};

    fn patch_tool() -> TrackedTool {
        TrackedTool {
            name: "apply_patch".to_string(),
            arguments: None,
            command_facts: None,
            test_run_id: None,
            patch_id: Some("patch-1".to_string()),
            emitted_output: false,
        }
    }

    fn result(delta: Value) -> AgentToolResult {
        AgentToolResult {
            success: true,
            output: "applied".to_string(),
            error: None,
            structured_content: None,
            images: None,
            metadata: Some(HashMap::from([(
                TURN_DIFF_DELTA_METADATA_KEY.to_string(),
                delta,
            )])),
        }
    }

    #[test]
    fn aggregates_multiple_patch_results_into_net_turn_diff() {
        let mut tracker = TurnDiffTracker::default();
        let first = tracker
            .update_from_result(
                &patch_tool(),
                &result(json!({
                    "changes": [{
                        "type": "add",
                        "path": "src/lib.rs",
                        "content": "first\n"
                    }]
                })),
            )
            .expect("first turn diff");
        assert!(first.payload["diff"].as_str().unwrap().contains("+first"));

        let second = tracker
            .update_from_result(
                &patch_tool(),
                &result(json!({
                    "changes": [{
                        "type": "update",
                        "path": "src/lib.rs",
                        "oldContent": "first\n",
                        "newContent": "second\n"
                    }]
                })),
            )
            .expect("updated turn diff");
        let diff = second.payload["diff"].as_str().unwrap();
        assert!(diff.contains("+second"));
        assert!(!diff.contains("+first\n"));
    }

    #[test]
    fn net_zero_patch_emits_empty_turn_diff() {
        let mut tracker = TurnDiffTracker::default();
        tracker.update_from_result(
            &patch_tool(),
            &result(json!({
                "changes": [{ "type": "add", "path": "a.txt", "content": "one\n" }]
            })),
        );
        let cleared = tracker
            .update_from_result(
                &patch_tool(),
                &result(json!({
                    "changes": [{ "type": "delete", "path": "a.txt", "content": "one\n" }]
                })),
            )
            .expect("cleared turn diff");
        assert_eq!(cleared.payload["diff"], "");
    }

    #[test]
    fn move_preserves_source_and_destination_in_unified_diff() {
        let mut tracker = TurnDiffTracker::default();
        let event = tracker
            .update_from_result(
                &patch_tool(),
                &result(json!({
                    "changes": [{
                        "type": "update",
                        "path": "old.txt",
                        "movePath": "new.txt",
                        "oldContent": "same\n",
                        "newContent": "same\n"
                    }]
                })),
            )
            .expect("move turn diff");
        let diff = event.payload["diff"].as_str().unwrap();
        assert!(diff.contains("rename from old.txt"));
        assert!(diff.contains("rename to new.txt"));
    }

    #[test]
    fn internal_delta_is_removed_before_runtime_event_persistence() {
        let mut item = ThreadItem::new(
            SessionId::new("session-1"),
            ThreadId::new("thread-1"),
            TurnId::new("turn-1"),
            1,
            1,
            ThreadItemPayload::Unknown {
                upstream_type: "test".to_string(),
                field_names: Vec::new(),
            },
        );
        item.item_id = ItemId::new("item-1");
        item.status = ItemStatus::Completed;
        item.metadata = json!({
            "safe": true,
            "turnDiffDelta": {
                "changes": [{ "type": "add", "path": "secret.txt", "content": "full" }]
            }
        });
        let event = RuntimeAgentEvent::ItemCompleted { item };
        let persisted = runtime_event_for_persistence(&event);
        let RuntimeAgentEvent::ItemCompleted { item } = persisted.as_ref() else {
            panic!("item completed event expected");
        };
        assert_eq!(item.metadata["safe"], true);
        assert!(item.metadata.get(TURN_DIFF_DELTA_METADATA_KEY).is_none());
    }
}
