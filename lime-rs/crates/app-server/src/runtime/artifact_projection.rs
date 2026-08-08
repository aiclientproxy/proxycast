use super::article_workspace_artifact_document_projection;
use super::artifact_document_versions;
use super::output_refs;
use super::string_array_field;
use super::string_field;
use super::StoredSession;
use app_server_protocol::AgentEvent;
use app_server_protocol::ArtifactContentStatus;
use app_server_protocol::ArtifactSummary;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::collections::HashSet;

pub(super) fn paginate_artifact_summaries(
    artifacts: Vec<ArtifactSummary>,
    cursor: Option<String>,
    limit: Option<u32>,
) -> (Vec<ArtifactSummary>, Option<String>) {
    let start = cursor
        .as_deref()
        .and_then(|cursor| cursor.parse::<usize>().ok())
        .unwrap_or(0)
        .min(artifacts.len());
    let Some(limit) = limit
        .filter(|limit| *limit > 0)
        .and_then(|limit| usize::try_from(limit).ok())
    else {
        return (artifacts.into_iter().skip(start).collect(), None);
    };

    let end = start.saturating_add(limit).min(artifacts.len());
    let next_cursor = (end < artifacts.len()).then(|| end.to_string());
    (
        artifacts
            .into_iter()
            .skip(start)
            .take(end.saturating_sub(start))
            .collect(),
        next_cursor,
    )
}

pub(super) fn events_for_turn(events: &[AgentEvent], turn_id: Option<&str>) -> Vec<AgentEvent> {
    events
        .iter()
        .filter(|event| match turn_id {
            Some(turn_id) => event.turn_id.as_deref() == Some(turn_id),
            None => true,
        })
        .cloned()
        .collect()
}

pub(super) fn artifact_summaries_for_turn(
    events: &[AgentEvent],
    turn_id: Option<&str>,
) -> Vec<ArtifactSummary> {
    let mut index_by_ref = HashMap::new();
    let mut summaries = Vec::new();
    for event in events.iter().rev() {
        if let Some(turn_id) = turn_id {
            if event.turn_id.as_deref() != Some(turn_id) {
                continue;
            }
        }
        for mut summary in artifact_summaries_from_event(event) {
            summary.content = None;
            summary.content_status = ArtifactContentStatus::NotRequested;
            upsert_artifact_summary(&mut summaries, &mut index_by_ref, summary);
        }
    }
    summaries
}

pub(super) fn upsert_artifact_summary(
    summaries: &mut Vec<ArtifactSummary>,
    index_by_ref: &mut HashMap<String, usize>,
    summary: ArtifactSummary,
) {
    if let Some(index) = index_by_ref.get(&summary.artifact_ref).copied() {
        artifact_document_versions::merge_artifact_document_version_history(
            &mut summaries[index],
            &summary,
        );
        return;
    }
    index_by_ref.insert(summary.artifact_ref.clone(), summaries.len());
    summaries.push(summary);
}

pub(super) fn stored_artifact_summaries_for_turn(
    stored: &StoredSession,
    turn_id: Option<&str>,
) -> Vec<ArtifactSummary> {
    let mut summaries = artifact_summaries_for_turn(&stored.events, turn_id);
    let mut seen = summaries
        .iter()
        .map(|summary| summary.artifact_ref.clone())
        .collect::<HashSet<_>>();
    for summary in output_refs::output_summaries_for_turn(stored.output_blobs.values(), turn_id) {
        if seen.insert(summary.artifact_ref.clone()) {
            summaries.push(summary);
        }
    }
    summaries
}

pub(super) fn stored_user_visible_artifact_summaries_for_turn(
    stored: &StoredSession,
    turn_id: Option<&str>,
) -> Vec<ArtifactSummary> {
    stored_artifact_summaries_for_turn(stored, turn_id)
        .into_iter()
        .filter(is_user_visible_artifact_summary)
        .collect()
}

fn is_user_visible_artifact_summary(summary: &ArtifactSummary) -> bool {
    !is_workspace_patch_artifact_summary(summary)
}

fn is_workspace_patch_artifact_summary(summary: &ArtifactSummary) -> bool {
    summary.kind.as_deref().is_some_and(is_workspace_patch_kind)
        || summary.path.as_deref().is_some_and(is_workspace_patch_path)
        || summary
            .metadata
            .as_ref()
            .is_some_and(metadata_contains_workspace_patch)
}

fn is_workspace_patch_kind(kind: &str) -> bool {
    kind.trim().replace('-', "_").to_ascii_lowercase() == "workspace_patch"
}

fn is_workspace_patch_path(path: &str) -> bool {
    let normalized = path.trim().replace('\\', "/").to_ascii_lowercase();
    normalized.ends_with("/workspace-patch.json")
}

fn metadata_contains_workspace_patch(metadata: &Value) -> bool {
    let Some(metadata) = metadata.as_object() else {
        return false;
    };
    if metadata.contains_key("workspacePatch") || metadata.contains_key("workspace_patch") {
        return true;
    }
    string_from_metadata(
        metadata,
        &[
            "artifactKind",
            "artifact_kind",
            "outputArtifactKind",
            "output_artifact_kind",
            "kind",
        ],
    )
    .as_deref()
    .is_some_and(is_workspace_patch_kind)
}

pub(super) fn artifact_summaries_from_event(event: &AgentEvent) -> Vec<ArtifactSummary> {
    let mut summaries =
        article_workspace_artifact_document_projection::artifact_summaries_from_event(event);
    if let Some(summary) = artifact_summary_from_event(event) {
        summaries.push(summary);
    }
    summaries.extend(artifact_ref_summaries_from_event(event));
    dedupe_artifact_summaries(summaries)
}

fn artifact_summary_from_event(event: &AgentEvent) -> Option<ArtifactSummary> {
    let payload = &event.payload;
    let artifact = payload.get("artifact").unwrap_or(payload);
    let is_artifact_event = event.event_type.contains("artifact")
        || payload.get("artifact").is_some()
        || string_field(payload, &["artifactRef"]).is_some();
    if !is_artifact_event {
        return None;
    }

    let artifact_id = string_field(artifact, &["artifactId", "artifact_id", "id"])
        .or_else(|| string_field(payload, &["artifactId", "artifact_id"]));
    let path = string_field(artifact, &["filePath", "file_path", "path", "artifactRef"])
        .or_else(|| string_field(payload, &["filePath", "file_path", "path", "artifactRef"]));
    let artifact_ref = artifact_id
        .clone()
        .or_else(|| path.clone())
        .unwrap_or_else(|| event.event_id.clone());
    let metadata = artifact_metadata_with_sidecar_ref(
        artifact
            .get("metadata")
            .cloned()
            .or_else(|| payload.get("metadata").cloned())
            .or_else(|| {
                if payload.get("artifact").is_some() && artifact.is_object() {
                    Some(artifact.clone())
                } else {
                    None
                }
            }),
        artifact,
        payload,
    );

    Some(ArtifactSummary {
        artifact_ref,
        event_id: event.event_id.clone(),
        sequence: event.sequence,
        turn_id: event.turn_id.clone(),
        artifact_id,
        path,
        title: string_field(artifact, &["title", "artifactTitle"])
            .or_else(|| string_field(payload, &["title", "artifactTitle"])),
        kind: string_field(artifact, &["kind", "artifactKind"])
            .or_else(|| string_field(payload, &["kind", "artifactKind"])),
        status: string_field(artifact, &["status", "artifactStatus"])
            .or_else(|| string_field(payload, &["status", "artifactStatus"])),
        content: string_field(artifact, &["content"])
            .or_else(|| string_field(payload, &["content"])),
        content_status: ArtifactContentStatus::NotRequested,
        metadata,
    })
}

fn artifact_metadata_with_sidecar_ref(
    metadata: Option<Value>,
    artifact: &Value,
    payload: &Value,
) -> Option<Value> {
    let sidecar_ref = artifact
        .get("sidecarRef")
        .or_else(|| payload.get("sidecarRef"))
        .cloned();
    let Some(sidecar_ref) = sidecar_ref else {
        return metadata;
    };
    let Some(Value::Object(mut metadata)) = metadata else {
        return Some(json!({
            "sidecarRef": sidecar_ref,
        }));
    };
    metadata
        .entry("sidecarRef".to_string())
        .or_insert(sidecar_ref);
    copy_artifact_content_metadata_if_present(&mut metadata, "contentStatus", artifact, payload);
    copy_artifact_content_metadata_if_present(&mut metadata, "contentBytes", artifact, payload);
    copy_artifact_content_metadata_if_present(&mut metadata, "contentSha256", artifact, payload);
    Some(Value::Object(metadata))
}

fn copy_artifact_content_metadata_if_present(
    metadata: &mut Map<String, Value>,
    key: &str,
    artifact: &Value,
    payload: &Value,
) {
    if metadata.contains_key(key) {
        return;
    }
    if let Some(value) = artifact.get(key).or_else(|| payload.get(key)) {
        metadata.insert(key.to_string(), value.clone());
    }
}

fn artifact_ref_summaries_from_event(event: &AgentEvent) -> Vec<ArtifactSummary> {
    let payload = &event.payload;
    let artifact_refs = string_array_field(
        payload,
        &[
            "artifactRefs",
            "artifact_refs",
            "artifactIds",
            "artifact_ids",
            "artifactRef",
            "artifact_ref",
            "artifactId",
            "artifact_id",
        ],
    );
    if artifact_refs.is_empty() {
        return Vec::new();
    }
    let path = string_field(payload, &["filePath", "file_path", "path", "artifactPath"]);
    let metadata = artifact_ref_metadata_from_payload(payload);
    artifact_refs
        .into_iter()
        .map(|artifact_ref| ArtifactSummary {
            artifact_id: Some(artifact_ref.clone()),
            artifact_ref,
            event_id: event.event_id.clone(),
            sequence: event.sequence,
            turn_id: event.turn_id.clone(),
            path: path.clone(),
            title: string_field(payload, &["title", "artifactTitle"]),
            kind: string_field(payload, &["kind", "artifactKind"]),
            status: string_field(payload, &["status", "artifactStatus"]),
            content: string_field(payload, &["content"]),
            content_status: ArtifactContentStatus::NotRequested,
            metadata: metadata.clone(),
        })
        .collect()
}

fn artifact_ref_metadata_from_payload(payload: &Value) -> Option<Value> {
    let original_metadata = payload.get("metadata").cloned();
    let mut metadata = original_metadata
        .as_ref()
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut normalized = false;

    insert_string_metadata_if_missing(
        &mut metadata,
        "previewText",
        &["previewText", "preview_text", "preview", "summary"],
        payload,
        &mut normalized,
    );
    insert_string_metadata_if_missing(
        &mut metadata,
        "changeKind",
        &["changeKind", "change_kind", "operation"],
        payload,
        &mut normalized,
    );
    insert_string_metadata_if_missing(
        &mut metadata,
        "checkpointRef",
        &[
            "checkpointRef",
            "checkpoint_ref",
            "checkpointId",
            "checkpoint_id",
        ],
        payload,
        &mut normalized,
    );
    insert_string_metadata_if_missing(
        &mut metadata,
        "contentRef",
        &["contentRef", "content_ref"],
        payload,
        &mut normalized,
    );
    insert_string_metadata_if_missing(
        &mut metadata,
        "diffRef",
        &["diffRef", "diff_ref"],
        payload,
        &mut normalized,
    );

    if !metadata_has_any(&metadata, &["file_change", "fileChange"]) {
        if let Some(change) = payload
            .get("file_change")
            .or_else(|| payload.get("fileChange"))
            .or_else(|| payload.get("change"))
            .cloned()
        {
            metadata.insert("file_change".to_string(), change);
            normalized = true;
        }
    }
    if !metadata_has_any(&metadata, &["artifactVersionDiff", "artifact_version_diff"]) {
        if let Some(diff) = payload
            .get("artifactVersionDiff")
            .or_else(|| payload.get("artifact_version_diff"))
            .or_else(|| payload.get("diff"))
            .cloned()
        {
            metadata.insert("artifactVersionDiff".to_string(), diff);
            normalized = true;
        } else if let Some(diff_ref) = string_field(payload, &["diffRef", "diff_ref"]) {
            metadata.insert(
                "artifactVersionDiff".to_string(),
                json!({ "diffRef": diff_ref }),
            );
            normalized = true;
        }
    }

    if metadata.is_empty() {
        original_metadata
    } else if normalized
        || original_metadata
            .as_ref()
            .and_then(Value::as_object)
            .is_some()
    {
        Some(Value::Object(metadata))
    } else {
        original_metadata
    }
}

fn insert_string_metadata_if_missing(
    metadata: &mut Map<String, Value>,
    target_key: &str,
    source_keys: &[&str],
    payload: &Value,
    normalized: &mut bool,
) {
    if metadata.contains_key(target_key) {
        return;
    }
    if let Some(value) =
        string_from_metadata(metadata, source_keys).or_else(|| string_field(payload, source_keys))
    {
        metadata.insert(target_key.to_string(), json!(value));
        *normalized = true;
    }
}

fn metadata_has_any(metadata: &Map<String, Value>, keys: &[&str]) -> bool {
    keys.iter().any(|key| metadata.contains_key(*key))
}

fn string_from_metadata(metadata: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        metadata
            .get(*key)
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn dedupe_artifact_summaries(summaries: Vec<ArtifactSummary>) -> Vec<ArtifactSummary> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for summary in summaries {
        if seen.insert(summary.artifact_ref.clone()) {
            deduped.push(summary);
        }
    }
    deduped
}
