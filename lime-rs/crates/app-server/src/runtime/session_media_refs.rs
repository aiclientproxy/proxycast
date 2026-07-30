use super::sidecar_store::{normalize_sidecar_relative_path, session_scoped_relative_path};
use super::{RuntimeCoreError, StoredSession};
use app_server_protocol::protocol::v2::MediaReadParams;
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub(super) struct RequestedMediaSidecar {
    keys: HashSet<String>,
    relative_path: Option<String>,
}

impl RequestedMediaSidecar {
    pub(super) fn from_params(params: &MediaReadParams) -> Result<Self, RuntimeCoreError> {
        let mut keys = HashSet::new();
        push_key(&mut keys, params.uri.as_deref());
        push_key(&mut keys, params.ref_id.as_deref());
        if let Some(sidecar_ref) = params.sidecar_ref.as_ref() {
            push_key(&mut keys, string_value(sidecar_ref, &["ref"]).as_deref());
            push_key(&mut keys, string_value(sidecar_ref, &["uri"]).as_deref());
        }
        let relative_path = params
            .sidecar_ref
            .as_ref()
            .and_then(|sidecar_ref| string_value(sidecar_ref, &["relativePath", "relative_path"]))
            .and_then(|path| normalize_sidecar_relative_path(path.as_str()).ok());

        if keys.is_empty() && relative_path.is_none() {
            return Err(RuntimeCoreError::Backend(
                "media/read requires uri, ref, or sidecarRef".to_string(),
            ));
        }
        Ok(Self {
            keys,
            relative_path,
        })
    }

    pub(super) fn display_uri(&self) -> String {
        self.keys
            .iter()
            .next()
            .cloned()
            .or_else(|| self.relative_path.clone())
            .unwrap_or_else(|| "sidecar://media/unknown".to_string())
    }
}

#[derive(Debug, Clone)]
pub(super) struct KnownMediaSidecarRef {
    ref_id: Option<String>,
    uri: Option<String>,
    aliases: Vec<String>,
    pub(super) relative_path: String,
    pub(super) sha256: Option<String>,
    pub(super) bytes: Option<u64>,
    pub(super) mime_type: Option<String>,
    pub(super) sidecar_ref: Value,
}

impl KnownMediaSidecarRef {
    pub(super) fn matches(&self, requested: &RequestedMediaSidecar) -> bool {
        self.aliases
            .iter()
            .any(|value| requested.keys.contains(value))
            || requested
                .relative_path
                .as_deref()
                .is_some_and(|value| value == self.relative_path)
    }

    pub(super) fn display_uri(&self, requested: &RequestedMediaSidecar) -> String {
        self.uri
            .clone()
            .or_else(|| self.ref_id.clone())
            .or_else(|| self.aliases.first().cloned())
            .unwrap_or_else(|| requested.display_uri())
    }
}

pub(super) fn known_media_sidecar_refs(stored: &StoredSession) -> Vec<KnownMediaSidecarRef> {
    let mut refs = Vec::new();
    for event in &stored.events {
        collect_media_sidecar_refs_from_value(&event.payload, &mut refs);
    }
    refs
}

pub(super) fn session_scoped_media_relative_path(
    session_id: &str,
    relative_path: &str,
) -> Result<String, RuntimeCoreError> {
    let relative_path =
        normalize_sidecar_relative_path(relative_path).map_err(RuntimeCoreError::Backend)?;
    let session_prefix = session_scoped_relative_path(session_id, "");
    if !relative_path.starts_with(session_prefix.as_str()) {
        return Err(RuntimeCoreError::Backend(
            "agent session media sidecar path is outside the requested session".to_string(),
        ));
    }
    Ok(relative_path)
}

fn collect_media_sidecar_refs_from_value(value: &Value, refs: &mut Vec<KnownMediaSidecarRef>) {
    match value {
        Value::Object(object) => {
            for key in ["sidecarRef", "sidecar_ref"] {
                if let Some(sidecar_ref) = object.get(key) {
                    if let Some(known_ref) = known_media_sidecar_ref(sidecar_ref, value) {
                        refs.push(known_ref);
                    }
                }
            }
            for child in object.values() {
                collect_media_sidecar_refs_from_value(child, refs);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_media_sidecar_refs_from_value(item, refs);
            }
        }
        _ => {}
    }
}

fn known_media_sidecar_ref(sidecar_ref: &Value, context: &Value) -> Option<KnownMediaSidecarRef> {
    if !looks_like_media_sidecar_ref(sidecar_ref, context) {
        return None;
    }
    let relative_path = string_value(sidecar_ref, &["relativePath", "relative_path"])
        .and_then(|path| normalize_sidecar_relative_path(path.as_str()).ok())?;
    let ref_id = string_value(sidecar_ref, &["ref"]);
    let uri = string_value(sidecar_ref, &["uri"]).or_else(|| string_value(context, &["uri"]));
    let aliases = media_sidecar_aliases(sidecar_ref, context, ref_id.as_deref(), uri.as_deref());
    Some(KnownMediaSidecarRef {
        ref_id,
        uri,
        aliases,
        relative_path,
        sha256: string_value(sidecar_ref, &["sha256", "sha"])
            .or_else(|| string_value(context, &["sha256", "sha", "contentSha256"])),
        bytes: u64_value(sidecar_ref, &["bytes", "byteSize", "byte_size"])
            .or_else(|| u64_value(context, &["bytes", "byteSize", "byte_size", "contentBytes"])),
        mime_type: media_mime_type(sidecar_ref).or_else(|| media_mime_type(context)),
        sidecar_ref: sidecar_ref.clone(),
    })
}

fn media_sidecar_aliases(
    sidecar_ref: &Value,
    context: &Value,
    ref_id: Option<&str>,
    uri: Option<&str>,
) -> Vec<String> {
    let mut aliases = Vec::new();
    push_alias(&mut aliases, ref_id);
    push_alias(&mut aliases, uri);
    for key in [
        "ref",
        "refId",
        "ref_id",
        "uri",
        "sourceUri",
        "source_uri",
        "previewUrl",
        "preview_url",
        "artifactRef",
        "artifact_ref",
        "artifactId",
        "artifact_id",
        "id",
        "path",
        "filePath",
        "file_path",
        "contentRef",
        "content_ref",
        "cacheRef",
        "cache_ref",
        "outputRef",
        "output_ref",
    ] {
        push_alias(&mut aliases, string_value(context, &[key]).as_deref());
    }
    for key in ["relativePath", "relative_path"] {
        push_alias(&mut aliases, string_value(sidecar_ref, &[key]).as_deref());
    }
    aliases
}

fn looks_like_media_sidecar_ref(sidecar_ref: &Value, context: &Value) -> bool {
    is_media_kind(sidecar_ref)
        || is_media_kind(context)
        || media_mime_type(sidecar_ref).is_some()
        || media_mime_type(context).is_some()
        || string_value(
            sidecar_ref,
            &["ref", "uri", "relativePath", "relative_path"],
        )
        .or_else(|| string_value(context, &["uri", "sourceUri", "source_uri"]))
        .is_some_and(|value| media_sidecar_uri(value.as_str()))
}

fn is_media_kind(value: &Value) -> bool {
    string_value(
        value,
        &[
            "kind",
            "type",
            "artifactKind",
            "artifact_kind",
            "contentKind",
            "content_kind",
        ],
    )
    .is_some_and(|value| {
        let normalized = value.trim().replace('-', "_").to_ascii_lowercase();
        normalized.contains("media")
            || matches!(
                normalized.as_str(),
                "image"
                    | "audio"
                    | "video"
                    | "image_artifact"
                    | "audio_artifact"
                    | "video_artifact"
            )
    })
}

fn media_mime_type(value: &Value) -> Option<String> {
    string_value(
        value,
        &[
            "mimeType",
            "mime_type",
            "mediaType",
            "media_type",
            "contentType",
            "content_type",
        ],
    )
    .filter(|value| {
        let normalized = value.trim().to_ascii_lowercase();
        normalized.starts_with("image/")
            || normalized.starts_with("audio/")
            || normalized.starts_with("video/")
    })
}

fn media_sidecar_uri(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase();
    normalized.starts_with("sidecar://media/") || normalized.contains("/media/")
}

fn push_key(keys: &mut HashSet<String>, value: Option<&str>) {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return;
    };
    keys.insert(value.to_string());
}

fn push_alias(aliases: &mut Vec<String>, value: Option<&str>) {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return;
    };
    if !aliases.iter().any(|alias| alias == value) {
        aliases.push(value.to_string());
    }
}

fn string_value(value: &Value, keys: &[&str]) -> Option<String> {
    let object = value.as_object()?;
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn u64_value(value: &Value, keys: &[&str]) -> Option<u64> {
    let object = value.as_object()?;
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(|value| {
            value.as_u64().or_else(|| {
                value
                    .as_i64()
                    .filter(|value| *value >= 0)
                    .and_then(|value| u64::try_from(value).ok())
            })
        })
}
