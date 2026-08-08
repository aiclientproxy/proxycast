use app_server_protocol::AgentEvent;
use app_server_protocol::AgentSession;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

const DEFAULT_SCHEMA_VERSION: &str = "article-workspace.v1";
const INLINE_IMAGE_TASK_SLOT_MARKER: &str = "lime:image-task-slot:";
const INLINE_IMAGE_TASK_PLACEHOLDER: &str = "pending-image-task://";
const RESOLVED_IMAGE_URL_PREFIXES: [&str; 5] =
    ["http://", "https://", "file://", "asset://", "data:image/"];

pub(super) fn article_workspace_from_events(
    session: &AgentSession,
    events: &[AgentEvent],
) -> Option<Value> {
    let mut workspace = ArticleWorkspaceBuilder::new(session);
    for event in events {
        for patch in workspace_patches_from_event(event) {
            workspace.apply_patch(event, &patch);
        }
    }
    workspace.into_value()
}

struct ArticleWorkspaceBuilder<'a> {
    session: &'a AgentSession,
    app_id: Option<String>,
    objects: BTreeMap<String, Value>,
    object_order: Vec<String>,
    primary_object_ref: Option<Value>,
    selected_object_ref: Option<Value>,
    edited_draft: Option<Value>,
    layout_state: Option<Value>,
    source_artifacts: Vec<Value>,
    worker_evidence: Vec<Value>,
    worker_evidence_index_by_key: BTreeMap<String, usize>,
    article_generation_task_statuses: BTreeMap<String, String>,
    updated_at: Option<String>,
}

impl<'a> ArticleWorkspaceBuilder<'a> {
    fn new(session: &'a AgentSession) -> Self {
        Self {
            session,
            app_id: Some(session.app_id.clone()),
            objects: BTreeMap::new(),
            object_order: Vec::new(),
            primary_object_ref: None,
            selected_object_ref: None,
            edited_draft: None,
            layout_state: None,
            source_artifacts: Vec::new(),
            worker_evidence: Vec::new(),
            worker_evidence_index_by_key: BTreeMap::new(),
            article_generation_task_statuses: BTreeMap::new(),
            updated_at: None,
        }
    }

    fn apply_patch(&mut self, event: &AgentEvent, patch: &Value) {
        let Some(objects) = patch.get("objects").and_then(Value::as_array) else {
            return;
        };
        if objects.is_empty() {
            return;
        }

        if let Some(app_id) = string_field(patch, &["appId", "app_id"]) {
            self.app_id = Some(app_id);
        }
        if let Some(primary_object_ref) =
            object_ref_field(patch, &["primaryObjectRef", "primary_object_ref"])
        {
            self.primary_object_ref = Some(primary_object_ref);
        }
        if let Some(selected_object_ref) =
            object_ref_field(patch, &["selectedObjectRef", "selected_object_ref"])
        {
            self.selected_object_ref = Some(selected_object_ref);
        }
        if let Some(edited_draft) = edited_draft_from_event(event) {
            if !self.should_reject_edited_draft(&edited_draft) {
                self.edited_draft = Some(edited_draft);
            }
        }
        if let Some(layout_state) = patch
            .get("layoutState")
            .or_else(|| patch.get("layout_state"))
            .filter(|value| value.is_object())
            .cloned()
        {
            self.layout_state = Some(layout_state);
        }

        for object in objects {
            let Some(key) = article_object_key(object) else {
                continue;
            };
            if !self.objects.contains_key(&key) {
                self.object_order.push(key.clone());
            }
            let next_object = self
                .objects
                .get(&key)
                .map(|current| merge_article_object(current, object))
                .unwrap_or_else(|| object.clone());
            self.objects.insert(key, next_object);
        }

        if let Some(source_artifact) = source_artifact_from_event(event) {
            self.source_artifacts.push(source_artifact);
        }
        for worker_evidence in worker_evidence_from_patch(event, patch) {
            self.push_worker_evidence(worker_evidence);
        }
        self.updated_at = Some(event.timestamp.clone());
    }

    fn should_reject_edited_draft(&self, edited_draft: &Value) -> bool {
        let Some(reference) = edited_draft
            .get("objectRef")
            .or_else(|| edited_draft.get("object_ref"))
        else {
            return false;
        };
        let Some(key) = article_object_ref_key(reference) else {
            return false;
        };
        let Some(current) = self.objects.get(&key) else {
            return false;
        };
        should_reject_article_markdown_update(current.get("source"), Some(edited_draft))
    }

    fn into_value(self) -> Option<Value> {
        if self.objects.is_empty() {
            return None;
        }

        let objects = self
            .object_order
            .iter()
            .filter_map(|key| self.objects.get(key).cloned())
            .collect::<Vec<_>>();
        let mut objects = objects;
        apply_article_generation_task_statuses(
            &mut objects,
            &self.article_generation_task_statuses,
        );
        let layout_state = self.layout_state.unwrap_or_else(default_layout_state);
        let mut value = Map::new();
        value.insert("schemaVersion".to_string(), json!(DEFAULT_SCHEMA_VERSION));
        value.insert(
            "appId".to_string(),
            json!(self.app_id.unwrap_or_else(|| self.session.app_id.clone())),
        );
        value.insert("sessionId".to_string(), json!(self.session.session_id));
        if let Some(workspace_id) = self.session.workspace_id.clone() {
            value.insert("workspaceId".to_string(), json!(workspace_id));
        }
        if let Some(primary_object_ref) = self.primary_object_ref {
            value.insert("primaryObjectRef".to_string(), primary_object_ref);
        }
        if let Some(selected_object_ref) = self.selected_object_ref {
            value.insert("selectedObjectRef".to_string(), selected_object_ref);
        }
        if let Some(edited_draft) = self.edited_draft {
            value.insert("editedDraft".to_string(), edited_draft.clone());
            value.insert("edited_draft".to_string(), edited_draft);
        }
        value.insert("objects".to_string(), Value::Array(objects));
        value.insert("objectCount".to_string(), json!(self.objects.len()));
        value.insert("layoutState".to_string(), layout_state);
        value.insert(
            "sourceArtifacts".to_string(),
            Value::Array(self.source_artifacts),
        );
        if !self.worker_evidence.is_empty() {
            value.insert(
                "workerEvidence".to_string(),
                Value::Array(self.worker_evidence),
            );
        }
        if let Some(updated_at) = self.updated_at {
            value.insert("updatedAt".to_string(), json!(updated_at));
        }
        Some(Value::Object(value))
    }

    fn record_article_generation_task_status(&mut self, worker_evidence: &Value) {
        let Some(task_id) = article_generation_task_id_from_worker_evidence(worker_evidence) else {
            return;
        };
        let Some(status) = string_field(worker_evidence, &["status"]) else {
            return;
        };
        if !matches!(status.as_str(), "completed" | "failed") {
            return;
        }
        self.article_generation_task_statuses
            .insert(task_id, status);
    }

    fn push_worker_evidence(&mut self, worker_evidence: Value) {
        let key = worker_evidence_dedupe_key(&worker_evidence)
            .unwrap_or_else(|| format!("worker-evidence:{}", self.worker_evidence.len()));
        if let Some(existing_index) = self.worker_evidence_index_by_key.get(&key).copied() {
            let merged = merge_worker_evidence_value(
                &self.worker_evidence[existing_index],
                &worker_evidence,
            );
            self.worker_evidence[existing_index] = merged;
            return;
        }
        self.record_article_generation_task_status(&worker_evidence);
        if let Some(updated_at) = string_field(&worker_evidence, &["updatedAt", "updated_at"]) {
            self.updated_at = Some(updated_at);
        }
        let next_index = self.worker_evidence.len();
        self.worker_evidence_index_by_key.insert(key, next_index);
        self.worker_evidence.push(worker_evidence);
    }
}

fn workspace_patches_from_event(event: &AgentEvent) -> Vec<Value> {
    let payload = &event.payload;
    let artifact = payload.get("artifact");
    let metadata = payload.get("metadata");
    let artifact_metadata = artifact.and_then(|artifact| artifact.get("metadata"));

    let mut patches = Vec::new();
    for candidate in [
        payload.get("articleWorkspace"),
        payload.get("article_workspace"),
        payload.get("workspacePatch"),
        payload.get("workspace_patch"),
        metadata.and_then(|value| value.get("articleWorkspace")),
        metadata.and_then(|value| value.get("article_workspace")),
        metadata.and_then(|value| value.get("workspacePatch")),
        metadata.and_then(|value| value.get("workspace_patch")),
        artifact.and_then(|value| value.get("articleWorkspace")),
        artifact.and_then(|value| value.get("article_workspace")),
        artifact.and_then(|value| value.get("workspacePatch")),
        artifact.and_then(|value| value.get("workspace_patch")),
        artifact_metadata.and_then(|value| value.get("articleWorkspace")),
        artifact_metadata.and_then(|value| value.get("article_workspace")),
        artifact_metadata.and_then(|value| value.get("workspacePatch")),
        artifact_metadata.and_then(|value| value.get("workspace_patch")),
    ]
    .into_iter()
    .flatten()
    {
        if candidate.get("objects").and_then(Value::as_array).is_some() {
            patches.push(candidate.clone());
        }
    }

    if let Some(content_patch) = artifact_content_patch(artifact) {
        patches.push(content_patch);
    }

    patches
}

fn artifact_content_patch(artifact: Option<&Value>) -> Option<Value> {
    let content = artifact?.get("content")?.as_str()?;
    let value: Value = serde_json::from_str(content).ok()?;
    value.get("objects").and_then(Value::as_array)?;
    Some(value)
}

fn edited_draft_from_event(event: &AgentEvent) -> Option<Value> {
    let payload = &event.payload;
    let artifact = payload.get("artifact");
    let metadata = payload.get("metadata");
    let artifact_metadata = artifact.and_then(|value| value.get("metadata"));
    [Some(payload), metadata, artifact, artifact_metadata]
        .into_iter()
        .flatten()
        .find_map(|value| {
            value
                .get("editedDraft")
                .or_else(|| value.get("edited_draft"))
                .filter(|draft| {
                    draft
                        .get("objectRef")
                        .or_else(|| draft.get("object_ref"))
                        .is_some_and(article_object_ref_is_valid)
                        && string_field(draft, &["markdown"]).is_some()
                })
                .cloned()
        })
}

fn article_object_key(object: &Value) -> Option<String> {
    let reference = object.get("ref").or_else(|| object.get("objectRef"))?;
    article_object_ref_key(reference)
}

fn merge_article_object(current: &Value, next: &Value) -> Value {
    let (Some(current_object), Some(next_object)) = (current.as_object(), next.as_object()) else {
        return next.clone();
    };
    let mut merged = current_object.clone();
    for (key, value) in next_object {
        if key == "source" {
            let current_source = current_object.get("source");
            let mut source = merge_json_object(current_source, value);
            if should_reject_article_markdown_update(current_source, Some(value)) {
                preserve_article_markdown(current_source, &mut source);
            }
            merged.insert(key.clone(), source);
            continue;
        }
        if key == "ref" || key == "objectRef" {
            let reference = merge_json_object(
                current_object
                    .get(key)
                    .or_else(|| current_object.get("ref"))
                    .or_else(|| current_object.get("objectRef")),
                value,
            );
            merged.insert(key.clone(), reference);
            continue;
        }
        merged.insert(key.clone(), value.clone());
    }
    Value::Object(merged)
}

fn merge_json_object(current: Option<&Value>, next: &Value) -> Value {
    let (Some(current_object), Some(next_object)) =
        (current.and_then(Value::as_object), next.as_object())
    else {
        return next.clone();
    };
    let mut merged = current_object.clone();
    for (key, value) in next_object {
        merged.insert(key.clone(), value.clone());
    }
    Value::Object(merged)
}

fn should_reject_article_markdown_update(current: Option<&Value>, next: Option<&Value>) -> bool {
    let Some(current_markdown) = current.and_then(article_markdown) else {
        return false;
    };
    let Some(next_markdown) = next.and_then(article_markdown) else {
        return false;
    };
    contains_inline_image_task_marker(current_markdown)
        && !contains_inline_image_task_marker(next_markdown)
        && !contains_resolved_markdown_image(next_markdown)
}

fn preserve_article_markdown(current: Option<&Value>, next: &mut Value) {
    let (Some(current), Some(next)) = (current.and_then(Value::as_object), next.as_object_mut())
    else {
        return;
    };
    for key in [
        "documentText",
        "document_text",
        "finalMarkdown",
        "final_markdown",
        "updatedAt",
        "updated_at",
        "edited",
    ] {
        if let Some(value) = current.get(key) {
            next.insert(key.to_string(), value.clone());
        }
    }
}

fn article_markdown(value: &Value) -> Option<&str> {
    value
        .get("markdown")
        .or_else(|| value.get("documentText"))
        .or_else(|| value.get("document_text"))
        .or_else(|| value.get("finalMarkdown"))
        .or_else(|| value.get("final_markdown"))
        .and_then(Value::as_str)
}

fn contains_inline_image_task_marker(markdown: &str) -> bool {
    markdown.contains(INLINE_IMAGE_TASK_SLOT_MARKER)
        || markdown.contains(INLINE_IMAGE_TASK_PLACEHOLDER)
}

fn contains_resolved_markdown_image(markdown: &str) -> bool {
    markdown.lines().any(|line| {
        let Some(image_start) = line.find("![") else {
            return false;
        };
        let image = &line[image_start..];
        let Some(url_start) = image.find("](") else {
            return false;
        };
        let url = &image[url_start + 2..];
        !url.starts_with(INLINE_IMAGE_TASK_PLACEHOLDER)
            && RESOLVED_IMAGE_URL_PREFIXES
                .iter()
                .any(|prefix| url.starts_with(prefix))
    })
}

fn article_object_ref_key(reference: &Value) -> Option<String> {
    let app_id = string_field(reference, &["appId", "app_id"])?;
    let kind = string_field(reference, &["kind"])?;
    let id = string_field(reference, &["id"])?;
    let session_id = string_field(reference, &["sessionId", "session_id"])?;
    Some(format!("{app_id}:{session_id}:{kind}:{id}"))
}

fn article_object_task_id(object: &Value) -> Option<String> {
    object
        .get("source")
        .and_then(|source| string_field(source, &["taskId", "task_id"]))
        .or_else(|| {
            object
                .get("ref")
                .or_else(|| object.get("objectRef"))
                .and_then(|reference| string_field(reference, &["sourceTaskId", "source_task_id"]))
        })
}

fn article_generation_task_id_from_worker_evidence(worker_evidence: &Value) -> Option<String> {
    if string_field(worker_evidence, &["taskKind"]).as_deref() != Some("content.article.generate") {
        return None;
    }
    string_field(worker_evidence, &["taskId"])
}

fn apply_article_generation_task_statuses(
    objects: &mut [Value],
    task_statuses: &BTreeMap<String, String>,
) {
    for object in objects {
        let Some(task_id) = article_object_task_id(object) else {
            continue;
        };
        if task_statuses.get(&task_id).map(String::as_str) != Some("failed") {
            continue;
        }
        if article_object_kind(object).as_deref() != Some("articleDraft") {
            continue;
        }
        if let Some(object_map) = object.as_object_mut() {
            object_map.insert("status".to_string(), json!("failed"));
            object_map.insert(
                "summary".to_string(),
                json!("写作失败，文章草稿未达到可交付状态"),
            );
        }
    }
}

fn object_ref_field(value: &Value, keys: &[&str]) -> Option<Value> {
    keys.iter()
        .find_map(|key| value.get(*key))
        .filter(|value| article_object_ref_is_valid(value))
        .cloned()
}

fn article_object_ref_is_valid(value: &Value) -> bool {
    string_field(value, &["appId", "app_id"]).is_some()
        && string_field(value, &["kind"]).is_some()
        && string_field(value, &["id"]).is_some()
        && string_field(value, &["sessionId", "session_id"]).is_some()
}

fn article_object_kind(object: &Value) -> Option<String> {
    string_field(object, &["kind"]).or_else(|| {
        object
            .get("ref")
            .or_else(|| object.get("objectRef"))
            .and_then(|reference| string_field(reference, &["kind"]))
    })
}

fn source_artifact_from_event(event: &AgentEvent) -> Option<Value> {
    let artifact = event.payload.get("artifact").unwrap_or(&event.payload);
    let artifact_ref = string_field(artifact, &["artifactId", "artifact_id", "id"])
        .or_else(|| string_field(artifact, &["artifactRef", "artifact_ref", "path"]))?;
    Some(json!({
        "artifactRef": artifact_ref,
        "eventId": event.event_id,
        "turnId": event.turn_id,
        "kind": string_field(artifact, &["kind", "artifactKind", "artifact_kind"]),
        "title": string_field(artifact, &["title", "artifactTitle", "artifact_title"]),
        "updatedAt": event.timestamp,
    }))
}

fn worker_evidence_from_patch(event: &AgentEvent, patch: &Value) -> Vec<Value> {
    patch
        .get("workerEvidence")
        .or_else(|| patch.get("worker_evidence"))
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .enumerate()
                .filter_map(|(index, item)| worker_evidence_item_from_patch(event, item, index))
                .collect()
        })
        .unwrap_or_default()
}

fn worker_evidence_item_from_patch(
    event: &AgentEvent,
    item: &Value,
    index: usize,
) -> Option<Value> {
    let mut object = item.as_object()?.clone();
    object
        .entry("id".to_string())
        .or_insert_with(|| json!(format!("{}:patchWorkerEvidence:{index}", event.event_id)));
    object
        .entry("eventId".to_string())
        .or_insert_with(|| json!(event.event_id));
    object
        .entry("turnId".to_string())
        .or_insert_with(|| json!(event.turn_id));
    object
        .entry("eventType".to_string())
        .or_insert_with(|| json!(event.event_type));
    object
        .entry("source".to_string())
        .or_insert_with(|| json!("workspace_patch"));
    object
        .entry("updatedAt".to_string())
        .or_insert_with(|| json!(event.timestamp));
    let mut worker_evidence = Value::Object(object);
    sanitize_worker_evidence_for_article_workspace(&mut worker_evidence);
    Some(worker_evidence)
}

fn sanitize_worker_evidence_for_article_workspace(worker_evidence: &mut Value) {
    let Some(object) = worker_evidence.as_object_mut() else {
        return;
    };
    for key in WORKER_EVIDENCE_AUDIT_ONLY_KEYS {
        object.remove(*key);
    }
}

const WORKER_EVIDENCE_AUDIT_ONLY_KEYS: &[&str] = &[
    "workerEntrypoint",
    "worker_entrypoint",
    "inputSummary",
    "input_summary",
    "outputSummary",
    "output_summary",
    "workflowKey",
    "workflow_key",
    "subagents",
    "sub_agents",
    "skillRefs",
    "skill_refs",
    "cliRefs",
    "cli_refs",
    "connectorRefs",
    "connector_refs",
    "hookPolicy",
    "hook_policy",
    "hookRefs",
    "hook_refs",
    "runtimeRegistries",
    "runtime_registries",
    "orchestration",
    "workflowSteps",
    "workflow_steps",
    "hookKey",
    "hook_key",
    "hookEvent",
    "hook_event",
    "hookScope",
    "hook_scope",
    "hookEntrypoint",
    "hook_entrypoint",
    "hookRequired",
    "hook_required",
    "reasonCode",
    "reason_code",
    "resultSummary",
    "result_summary",
];

fn worker_evidence_dedupe_key(worker_evidence: &Value) -> Option<String> {
    let task_id = string_field(worker_evidence, &["taskId", "task_id"])?;
    let turn_id = string_field(worker_evidence, &["turnId", "turn_id"]).unwrap_or_default();
    let status = string_field(worker_evidence, &["status"]).unwrap_or_default();
    let retry_attempt = worker_evidence
        .get("retryAttempt")
        .or_else(|| worker_evidence.get("retry_attempt"))
        .and_then(Value::as_u64)
        .map(|value| value.to_string())
        .unwrap_or_default();
    Some(format!("{turn_id}:{task_id}:{status}:{retry_attempt}"))
}

fn merge_worker_evidence_value(current: &Value, next: &Value) -> Value {
    let (Some(current_object), Some(next_object)) = (current.as_object(), next.as_object()) else {
        return if worker_evidence_value_score(next) > worker_evidence_value_score(current) {
            next.clone()
        } else {
            current.clone()
        };
    };
    let mut merged = current_object.clone();
    for (key, value) in next_object {
        let current_value = merged.get(key);
        if worker_evidence_field_should_replace(current_value, value) {
            merged.insert(key.clone(), value.clone());
        }
    }
    Value::Object(merged)
}

fn worker_evidence_field_should_replace(current: Option<&Value>, next: &Value) -> bool {
    if !worker_evidence_field_is_meaningful(next) {
        return false;
    }
    current
        .map(|value| !worker_evidence_field_is_meaningful(value))
        .unwrap_or(true)
}

fn worker_evidence_field_is_meaningful(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::String(value) => !value.trim().is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        Value::Bool(_) | Value::Number(_) => true,
    }
}

fn worker_evidence_value_score(value: &Value) -> usize {
    value
        .as_object()
        .map(|object| {
            object
                .values()
                .filter(|value| worker_evidence_field_is_meaningful(value))
                .count()
        })
        .unwrap_or_default()
}

fn default_layout_state() -> Value {
    json!({
        "activeTabKind": "articleWorkspace",
        "openTabKinds": ["articleWorkspace"],
        "splitMode": "chat-right-dock",
    })
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(*key))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}
