use app_server_protocol::AgentEvent;
use app_server_protocol::ArtifactContentStatus;
use app_server_protocol::ArtifactSummary;
use serde_json::{json, Map, Value};
use std::collections::HashSet;

const ARTIFACT_DOCUMENT_SCHEMA_VERSION: &str = "artifact_document.v1";

pub(super) fn artifact_summaries_from_event(event: &AgentEvent) -> Vec<ArtifactSummary> {
    let mut seen = HashSet::new();
    let mut summaries = Vec::new();
    for patch in workspace_patches_from_event(event) {
        for object in product_objects_from_patch(&patch) {
            let artifact_ids = artifact_ids_for_object(object);
            if artifact_ids.is_empty() {
                continue;
            }
            let Some(document) =
                artifact_document_from_object(&patch, object, &artifact_ids, event)
            else {
                continue;
            };
            let artifact_ref = artifact_ids
                .first()
                .cloned()
                .unwrap_or_else(|| document_artifact_id(&document));
            if !seen.insert(artifact_ref.clone()) {
                continue;
            }
            let metadata = artifact_summary_metadata(&patch, object, &artifact_ids, &document);
            summaries.push(ArtifactSummary {
                artifact_ref: artifact_ref.clone(),
                event_id: event.event_id.clone(),
                sequence: event.sequence,
                turn_id: event.turn_id.clone(),
                artifact_id: Some(artifact_ref),
                path: Some(artifact_document_path(object, &document)),
                title: string_field(object, &["title", "name"]),
                kind: Some("artifact_document".to_string()),
                status: string_field(&document, &["status"]),
                content: serde_json::to_string_pretty(&document).ok(),
                content_status: ArtifactContentStatus::NotRequested,
                metadata: Some(metadata),
            });
        }
    }
    summaries
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

fn product_objects_from_patch(patch: &Value) -> impl Iterator<Item = &Value> {
    patch
        .get("objects")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|object| object_ref(object).is_some())
}

fn artifact_document_from_object(
    patch: &Value,
    object: &Value,
    artifact_ids: &[String],
    event: &AgentEvent,
) -> Option<Value> {
    let reference = object_ref(object)?;
    let app_id = string_field(patch, &["appId", "app_id"])
        .or_else(|| string_field(reference, &["appId", "app_id"]))?;
    let session_id = string_field(patch, &["sessionId", "session_id"])
        .or_else(|| string_field(reference, &["sessionId", "session_id"]))?;
    let object_kind = string_field(reference, &["kind"])?;
    let object_id = string_field(reference, &["id"])?;
    let title = string_field(object, &["title", "name"]).unwrap_or_else(|| object_kind.clone());
    let status = artifact_document_status(
        string_field(object, &["status"])
            .as_deref()
            .unwrap_or("ready"),
    );
    let layout = product_object_layout(&object_kind);
    let blocks = artifact_document_blocks(object, layout);
    let artifact_id = artifact_document_id(&app_id, &artifact_ids[0]);
    let version_no = object_version_no(reference);
    let version_id = format!("{artifact_id}:v{version_no}");
    let summary = string_field(object, &["summary", "description"])
        .or_else(|| document_text(object).map(|text| truncate_chars(text.as_str(), 160)));
    let source_turn_id = string_field(reference, &["sourceTurnId", "source_turn_id"])
        .or_else(|| event.turn_id.clone());
    let source_task_id =
        string_field(reference, &["sourceTaskId", "source_task_id"]).or_else(|| {
            object
                .get("source")
                .and_then(|source| string_field(source, &["taskId", "task_id"]))
        });
    let sources = artifact_document_sources(object, artifact_ids, source_turn_id.as_deref());

    Some(json!({
        "schemaVersion": ARTIFACT_DOCUMENT_SCHEMA_VERSION,
        "artifactId": artifact_id,
        "workspaceId": string_field(patch, &["workspaceId", "workspace_id"]),
        "threadId": Value::Null,
        "turnId": source_turn_id,
        "kind": artifact_document_kind(&object_kind),
        "title": title,
        "status": status,
        "language": "zh-CN",
        "summary": summary,
        "blocks": blocks,
        "sources": sources,
        "metadata": {
            "generatedBy": "automation",
            "currentVersionId": version_id,
            "currentVersionNo": version_no,
            "versionHistory": [{
                "id": version_id,
                "artifactId": artifact_id,
                "versionNo": version_no,
                "title": title,
                "kind": artifact_document_kind(&object_kind),
                "status": status,
                "summary": summary,
                "createdBy": "automation",
                "createdAt": event.timestamp,
            }],
            "currentVersionDiff": {
                "targetVersionId": version_id,
                "targetVersionNo": version_no,
                "addedCount": blocks.as_array().map(Vec::len).unwrap_or(0),
                "removedCount": 0,
                "updatedCount": 0,
                "movedCount": 0,
                "changedBlocks": changed_blocks(&blocks),
            },
            "sourceRunBinding": {
                "turnId": source_turn_id,
                "taskId": source_task_id,
                "appId": app_id,
                "sessionId": session_id,
            },
            "articleWorkspace": {
                "appId": app_id,
                "sessionId": session_id,
                "workspaceId": string_field(patch, &["workspaceId", "workspace_id"]),
                "objectKind": object_kind,
                "objectId": object_id,
                "artifactIds": artifact_ids,
                "surfaceKind": layout,
                "layout": layout,
            },
        },
    }))
}

fn artifact_summary_metadata(
    patch: &Value,
    object: &Value,
    artifact_ids: &[String],
    document: &Value,
) -> Value {
    let reference = object_ref(object);
    let object_kind = reference
        .and_then(|value| string_field(value, &["kind"]))
        .unwrap_or_else(|| "generic".to_string());
    let layout = product_object_layout(object_kind.as_str());
    json!({
        "openedFrom": "app_server_article_workspace",
        "artifactSchema": ARTIFACT_DOCUMENT_SCHEMA_VERSION,
        "artifactKind": string_field(document, &["kind"]),
        "surfaceKind": layout,
        "layout": layout,
        "artifactDocument": document,
        "artifactTitle": string_field(document, &["title"]),
        "artifactDocumentId": document_artifact_id(document),
        "artifactVersionId": document
            .get("metadata")
            .and_then(|metadata| string_field(metadata, &["currentVersionId", "current_version_id"])),
        "artifactVersionNo": document
            .get("metadata")
            .and_then(|metadata| metadata.get("currentVersionNo").or_else(|| metadata.get("current_version_no")))
            .and_then(Value::as_u64),
        "sessionId": string_field(patch, &["sessionId", "session_id"])
            .or_else(|| reference.and_then(|value| string_field(value, &["sessionId", "session_id"]))),
        "turnId": string_field(document, &["turnId", "turn_id"]),
        "articleWorkspace": {
            "appId": string_field(patch, &["appId", "app_id"])
                .or_else(|| reference.and_then(|value| string_field(value, &["appId", "app_id"]))),
            "sessionId": string_field(patch, &["sessionId", "session_id"])
                .or_else(|| reference.and_then(|value| string_field(value, &["sessionId", "session_id"]))),
            "workspaceId": string_field(patch, &["workspaceId", "workspace_id"]),
            "objectKind": reference.and_then(|value| string_field(value, &["kind"])),
            "objectId": reference.and_then(|value| string_field(value, &["id"])),
            "artifactIds": artifact_ids,
            "surfaceKind": layout,
        },
    })
}

fn artifact_document_blocks(object: &Value, layout: &str) -> Value {
    match layout {
        "imageGrid" => image_blocks(object),
        "storyboard" => storyboard_blocks(object),
        "checklist" => checklist_blocks(object),
        "briefForm" => brief_blocks(object),
        _ => document_blocks(object),
    }
}

fn image_blocks(object: &Value) -> Value {
    let blocks = read_array_from_source(
        object,
        &[
            "images",
            "imageItems",
            "image_items",
            "imageUrls",
            "image_urls",
        ],
    )
    .into_iter()
    .take(12)
    .enumerate()
    .map(|(index, value)| {
        let record = value.as_object();
        let url = value.as_str().map(trimmed_string).flatten().or_else(|| {
            record.and_then(|record| {
                string_from_map(record, &["url", "src", "thumbnailUrl", "thumbnail_url"])
            })
        });
        let title = record
            .and_then(|record| string_from_map(record, &["title", "name", "alt"]))
            .or_else(|| url.clone())
            .unwrap_or_else(|| format!("image-{}", index + 1));
        let prompt = record
            .and_then(|record| string_from_map(record, &["prompt", "imagePrompt", "image_prompt"]));
        let caption = [Some(title.clone()), prompt]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(" - ");
        json!({
            "id": format!("image-{}", index + 1),
            "type": "image",
            "url": url.unwrap_or_default(),
            "alt": record
                .and_then(|record| string_from_map(record, &["alt", "description"]))
                .unwrap_or_else(|| title.clone()),
            "caption": caption,
        })
    })
    .collect::<Vec<_>>();
    Value::Array(blocks)
}

fn storyboard_blocks(object: &Value) -> Value {
    let blocks = read_array_from_source(object, &["shots", "storyboard", "scenes"])
        .into_iter()
        .take(24)
        .enumerate()
        .filter_map(|(index, value)| {
            let title = value
                .as_str()
                .map(trimmed_string)
                .flatten()
                .or_else(|| {
                    value.as_object().and_then(|record| {
                        string_from_map(record, &["title", "name", "scene", "summary"])
                    })
                })
                .unwrap_or_else(|| format!("{}", index + 1));
            if title.is_empty() {
                return None;
            }
            let record = value.as_object();
            let content = [
                Some(format!("### {}. {}", index + 1, title)),
                record.and_then(|record| {
                    string_from_map(record, &["description", "action", "camera", "notes"])
                }),
                record.and_then(|record| {
                    string_from_map(record, &["visualPrompt", "visual_prompt", "prompt"])
                }),
                record.and_then(|record| string_from_map(record, &["duration", "time", "seconds"])),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join("\n\n");
            Some(json!({
                "id": format!("shot-{}", index + 1),
                "type": "rich_text",
                "contentFormat": "markdown",
                "content": content,
                "markdown": content,
            }))
        })
        .collect::<Vec<_>>();
    Value::Array(blocks)
}

fn checklist_blocks(object: &Value) -> Value {
    let items = read_array_from_source(
        object,
        &["items", "checklist", "checklistItems", "checklist_items"],
    )
    .into_iter()
    .enumerate()
    .filter_map(|(index, value)| {
        let record = value.as_object();
        let title = value.as_str().map(trimmed_string).flatten().or_else(|| {
            record.and_then(|record| string_from_map(record, &["title", "label", "name"]))
        })?;
        let notes =
            record.and_then(|record| string_from_map(record, &["notes", "description", "reason"]));
        let status = record.and_then(|record| string_from_map(record, &["status", "state"]));
        let text = [Some(title), notes]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(": ");
        Some(json!({
            "id": record
                .and_then(|record| string_from_map(record, &["id", "key"]))
                .unwrap_or_else(|| format!("item-{}", index + 1)),
            "text": text,
            "state": if status.as_deref() == Some("done") || status.as_deref() == Some("ready") {
                "done"
            } else {
                "todo"
            },
        }))
    })
    .collect::<Vec<_>>();
    if items.is_empty() {
        document_blocks(object)
    } else {
        Value::Array(vec![json!({
            "id": "checklist",
            "type": "checklist",
            "title": string_field(object, &["title", "name"]),
            "items": items,
        })])
    }
}

fn brief_blocks(object: &Value) -> Value {
    let fields = brief_fields(object);
    if fields.is_empty() {
        return document_blocks(object);
    }
    Value::Array(
        fields
            .into_iter()
            .enumerate()
            .map(|(index, (key, label, value))| {
                json!({
                    "id": format!("brief-{}", if key.is_empty() { (index + 1).to_string() } else { sanitize_id(&key) }),
                    "type": "callout",
                    "tone": "neutral",
                    "title": label,
                    "body": value,
                })
            })
            .collect(),
    )
}

fn document_blocks(object: &Value) -> Value {
    let content = document_text(object)
        .or_else(|| string_field(object, &["summary", "description"]))
        .or_else(|| string_field(object, &["title", "name"]))
        .unwrap_or_default();
    Value::Array(vec![json!({
        "id": "body",
        "type": "rich_text",
        "contentFormat": "markdown",
        "content": content,
        "markdown": content,
    })])
}

fn changed_blocks(blocks: &Value) -> Value {
    Value::Array(
        blocks
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(index, block)| {
                json!({
                    "blockId": string_field(block, &["id"]).unwrap_or_else(|| format!("block-{}", index + 1)),
                    "changeType": "added",
                    "afterType": string_field(block, &["type"]),
                    "afterIndex": index,
                    "afterText": block_text(block),
                })
            })
            .collect(),
    )
}

fn block_text(block: &Value) -> String {
    match string_field(block, &["type"]).as_deref() {
        Some("rich_text") => string_field(block, &["markdown", "content"]).unwrap_or_default(),
        Some("callout") => [
            string_field(block, &["title"]),
            string_field(block, &["body"]),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("\n"),
        Some("checklist") => block
            .get("items")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|item| string_field(item, &["text"]))
            .collect::<Vec<_>>()
            .join("\n"),
        Some("image") => [
            string_field(block, &["caption"]),
            string_field(block, &["url"]),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("\n"),
        _ => String::new(),
    }
}

fn artifact_document_sources(
    object: &Value,
    artifact_ids: &[String],
    source_turn_id: Option<&str>,
) -> Value {
    let mut sources = Vec::new();
    let source = object.get("source");
    let task_kind = source.and_then(|source| string_field(source, &["taskKind", "task_kind"]));
    let task_id = source
        .and_then(|source| string_field(source, &["taskId", "task_id"]))
        .or_else(|| {
            object_ref(object)
                .and_then(|reference| string_field(reference, &["sourceTaskId", "source_task_id"]))
        });
    if task_kind.is_some() || task_id.is_some() || source_turn_id.is_some() {
        sources.push(json!({
            "id": "source-task",
            "type": "tool",
            "label": task_kind.clone().or(task_id.clone()).unwrap_or_else(|| "Article Workspace".to_string()),
            "locator": {
                "toolCallId": task_id,
                "turnId": source_turn_id,
            },
            "reliability": "derived",
        }));
    }
    sources.extend(artifact_ids.iter().map(|artifact_id| {
        json!({
            "id": format!("artifact-{}", sanitize_id(artifact_id)),
            "type": "file",
            "label": artifact_id,
            "locator": {
                "path": artifact_id,
            },
            "reliability": "derived",
        })
    }));
    Value::Array(sources)
}

fn artifact_ids_for_object(object: &Value) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut ids = Vec::new();
    if let Some(reference) = object_ref(object) {
        push_string_array(
            &mut ids,
            &mut seen,
            reference
                .get("artifactIds")
                .or_else(|| reference.get("artifact_ids")),
        );
    }
    push_string(
        &mut ids,
        &mut seen,
        string_field(object, &["previewArtifactId", "preview_artifact_id"]),
    );
    if let Some(source) = object.get("source") {
        push_string_array(
            &mut ids,
            &mut seen,
            source
                .get("artifactIds")
                .or_else(|| source.get("artifact_ids")),
        );
    }
    ids
}

fn push_string(ids: &mut Vec<String>, seen: &mut HashSet<String>, value: Option<String>) {
    let Some(value) = value else {
        return;
    };
    if seen.insert(value.clone()) {
        ids.push(value);
    }
}

fn push_string_array(ids: &mut Vec<String>, seen: &mut HashSet<String>, value: Option<&Value>) {
    for item in value.and_then(Value::as_array).into_iter().flatten() {
        push_string(ids, seen, item.as_str().and_then(trimmed_string));
    }
}

fn read_array_from_source(object: &Value, keys: &[&str]) -> Vec<Value> {
    let Some(source) = object.get("source") else {
        return Vec::new();
    };
    keys.iter()
        .find_map(|key| source.get(*key).and_then(Value::as_array))
        .cloned()
        .unwrap_or_default()
}

fn brief_fields(object: &Value) -> Vec<(String, String, String)> {
    let Some(source) = object.get("source") else {
        return Vec::new();
    };
    let fields = read_array_field(source, &["fields", "briefFields", "brief_fields"])
        .into_iter()
        .enumerate()
        .filter_map(|(index, value)| {
            let record = value.as_object()?;
            let key = string_from_map(record, &["key", "id"])
                .unwrap_or_else(|| format!("field-{}", index + 1));
            let label =
                string_from_map(record, &["label", "title", "name"]).unwrap_or_else(|| key.clone());
            let value = string_from_map(record, &["value", "text", "content"])?;
            Some((key, label, value))
        })
        .collect::<Vec<_>>();
    if !fields.is_empty() {
        return fields;
    }
    source
        .get("brief")
        .and_then(Value::as_object)
        .map(|brief| {
            brief
                .iter()
                .filter_map(|(key, value)| {
                    value
                        .as_str()
                        .and_then(trimmed_string)
                        .map(|value| (key.clone(), key.clone(), value))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn read_array_field(value: &Value, keys: &[&str]) -> Vec<Value> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_array))
        .cloned()
        .unwrap_or_default()
}

fn document_text(object: &Value) -> Option<String> {
    object.get("source").and_then(|source| {
        string_field(
            source,
            &[
                "markdown",
                "documentText",
                "document_text",
                "body",
                "content",
                "text",
                "excerpt",
            ],
        )
    })
}

fn product_object_layout(kind: &str) -> &'static str {
    match kind {
        "contentBrief" => "briefForm",
        "articleDraft" | "videoScript" => "document",
        "imageGenerationSet" => "imageGrid",
        "videoStoryboard" => "storyboard",
        "deliveryChecklist" => "checklist",
        _ => "generic",
    }
}

fn artifact_document_kind(kind: &str) -> &'static str {
    match kind {
        "contentBrief" | "imageGenerationSet" | "videoStoryboard" => "brief",
        "deliveryChecklist" => "plan",
        _ => "report",
    }
}

fn artifact_document_status(status: &str) -> &'static str {
    match status {
        "draft" => "draft",
        "generating" => "streaming",
        "failed" => "failed",
        "archived" => "archived",
        _ => "ready",
    }
}

fn object_version_no(reference: &Value) -> u64 {
    string_field(reference, &["version"])
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

fn artifact_document_id(app_id: &str, source_ref: &str) -> String {
    format!(
        "artifact-document:{}:{}",
        sanitize_id(app_id),
        sanitize_id(source_ref)
    )
}

fn document_artifact_id(document: &Value) -> String {
    string_field(document, &["artifactId", "artifact_id"])
        .unwrap_or_else(|| "artifact-document:article-workspace".to_string())
}

fn artifact_document_path(object: &Value, document: &Value) -> String {
    let title = string_field(document, &["title"])
        .or_else(|| string_field(object, &["title", "name"]))
        .unwrap_or_else(|| "article-workspace-artifact".to_string());
    format!(
        ".lime/artifacts/article-workspace/{}.artifact-document.json",
        sanitize_id(&title)
    )
}

fn object_ref(object: &Value) -> Option<&Value> {
    object
        .get("ref")
        .or_else(|| object.get("objectRef"))
        .filter(|reference| {
            string_field(reference, &["appId", "app_id"]).is_some()
                && string_field(reference, &["kind"]).is_some()
                && string_field(reference, &["id"]).is_some()
                && string_field(reference, &["sessionId", "session_id"]).is_some()
        })
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    value
        .as_object()
        .and_then(|record| string_from_map(record, keys))
}

fn string_from_map(record: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        record
            .get(*key)
            .and_then(Value::as_str)
            .and_then(trimmed_string)
    })
}

fn trimmed_string(value: &str) -> Option<String> {
    let normalized = value.trim();
    (!normalized.is_empty()).then(|| normalized.to_string())
}

fn sanitize_id(value: &str) -> String {
    let mut sanitized = value
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | ':' | '-') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    while sanitized.contains("--") {
        sanitized = sanitized.replace("--", "-");
    }
    let sanitized = sanitized
        .trim_matches('-')
        .chars()
        .take(120)
        .collect::<String>();
    if sanitized.is_empty() {
        "article-workspace".to_string()
    } else {
        sanitized
    }
}

fn truncate_chars(value: &str, max: usize) -> String {
    value.chars().take(max).collect()
}
