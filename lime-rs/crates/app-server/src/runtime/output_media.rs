use super::image_media::{extension_for_media_type, validate_image_bytes, MAX_IMAGE_MEDIA_BYTES};
use super::sidecar_store::{session_scoped_relative_path, SidecarBytesWriteRequest, SidecarStore};
use super::RuntimeCoreError;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

pub(super) fn persist_local_image_output_payload(
    event_type: &str,
    payload: &mut Value,
    session_id: &str,
    sidecar_store: Option<&SidecarStore>,
) -> Result<(), RuntimeCoreError> {
    if event_type != "item.completed" {
        return Ok(());
    }
    let Some(item) = payload
        .get_mut("item")
        .and_then(Value::as_object_mut)
        .filter(|item| string_field(item, "type") == Some("media"))
    else {
        return Ok(());
    };
    let Some(source) = string_field(item, "uri").map(str::to_string) else {
        return Ok(());
    };
    if is_safe_media_reference(&source) {
        remove_alternate_source_fields(item);
        if let Some(preview) = string_field(item, "preview")
            .filter(|value| is_absolute_local_path(value))
            .map(str::to_string)
        {
            item.insert(
                "preview".to_string(),
                Value::String(display_file_name(&preview).unwrap_or_else(|| "image".to_string())),
            );
        }
        return Ok(());
    }
    if !is_absolute_local_path(&source) {
        return Err(RuntimeCoreError::Backend(
            "image output must use a safe media reference or an absolute local path".to_string(),
        ));
    }
    let declared_media_type = string_field(item, "mime_type")
        .or_else(|| string_field(item, "mimeType"))
        .filter(|value| value.starts_with("image/"))
        .map(str::to_string)
        .ok_or_else(|| {
            RuntimeCoreError::Backend("local image output requires an image MIME type".to_string())
        })?;
    let store = sidecar_store.ok_or_else(|| {
        RuntimeCoreError::Backend(
            "local image output requires an initialized App Server sidecar store".to_string(),
        )
    })?;
    let path = Path::new(&source);
    let metadata = std::fs::metadata(path).map_err(|error| {
        RuntimeCoreError::Backend(format!("read local image output metadata failed: {error}"))
    })?;
    if !metadata.is_file() {
        return Err(RuntimeCoreError::Backend(
            "local image output must reference a regular file".to_string(),
        ));
    }
    if metadata.len() > MAX_IMAGE_MEDIA_BYTES as u64 {
        return Err(RuntimeCoreError::Backend(format!(
            "local image output exceeds {MAX_IMAGE_MEDIA_BYTES} bytes"
        )));
    }
    let bytes = std::fs::read(path).map_err(|error| {
        RuntimeCoreError::Backend(format!("read local image output failed: {error}"))
    })?;
    let declared_for_validation =
        (declared_media_type != "image/*").then_some(declared_media_type.as_str());
    let detected_media_type = validate_image_bytes(&bytes, declared_for_validation)
        .map_err(RuntimeCoreError::Backend)?
        .to_string();
    let digest = hex::encode(Sha256::digest(&bytes));
    let display_name = display_file_name(&source).unwrap_or_else(|| {
        format!(
            "image.{}",
            extension_for_media_type(detected_media_type.as_str())
        )
    });
    let encoded_display_name =
        url::form_urlencoded::byte_serialize(display_name.as_bytes()).collect::<String>();
    let canonical_uri = format!("sidecar://media/output-{digest}/{encoded_display_name}");
    let relative_path = session_scoped_relative_path(
        session_id,
        &format!(
            "media/output-{digest}.{}",
            extension_for_media_type(detected_media_type.as_str())
        ),
    );
    let mut sidecar_ref = store
        .write_bytes(&SidecarBytesWriteRequest {
            session_id: session_id.to_string(),
            kind: "media".to_string(),
            logical_id: format!("output-{digest}"),
            relative_path,
            content: bytes,
        })
        .map_err(RuntimeCoreError::Backend)?;
    sidecar_ref.ref_id = canonical_uri.clone();

    item.insert("uri".to_string(), Value::String(canonical_uri));
    item.insert("mime_type".to_string(), Value::String(detected_media_type));
    remove_alternate_source_fields(item);
    item.insert("preview".to_string(), Value::String(display_name));
    item.insert(
        "sidecarRef".to_string(),
        serde_json::to_value(sidecar_ref).map_err(|error| {
            RuntimeCoreError::Backend(format!("serialize image output sidecar failed: {error}"))
        })?,
    );
    Ok(())
}

fn remove_alternate_source_fields(item: &mut Map<String, Value>) {
    for key in [
        "path",
        "sourcePath",
        "source_path",
        "sourceUri",
        "source_uri",
        "url",
        "previewUrl",
        "preview_url",
    ] {
        item.remove(key);
    }
}

fn string_field<'a>(item: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    item.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn is_safe_media_reference(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase();
    normalized.starts_with("sidecar://")
        || normalized.starts_with("http://")
        || normalized.starts_with("https://")
}

fn is_absolute_local_path(value: &str) -> bool {
    let value = value.trim();
    Path::new(value).is_absolute()
        || value.starts_with("\\\\")
        || value.as_bytes().get(0..3).is_some_and(|prefix| {
            prefix[0].is_ascii_alphabetic()
                && prefix[1] == b':'
                && matches!(prefix[2], b'/' | b'\\')
        })
}

fn display_file_name(value: &str) -> Option<String> {
    value
        .trim()
        .rsplit(['/', '\\'])
        .find(|part| !part.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn local_image_output_is_sidecarized_without_retaining_source_path() {
        let root = tempfile::tempdir().expect("sidecar root");
        let store = SidecarStore::new(root.path()).expect("sidecar store");
        let source = root.path().join("fixture-media-reference.png");
        std::fs::write(&source, b"\x89PNG\r\n\x1a\nfixture").expect("image fixture");
        let mut payload = json!({
            "item": {
                "id": "media-1",
                "type": "media",
                "status": "completed",
                "uri": source,
                "mime_type": "image/png",
                "path": source,
                "source_path": source,
                "url": source,
                "preview": source
            }
        });

        persist_local_image_output_payload(
            "item.completed",
            &mut payload,
            "session-1",
            Some(&store),
        )
        .expect("persist output media");

        let serialized = payload.to_string();
        assert!(!serialized.contains(source.to_string_lossy().as_ref()));
        assert!(payload["item"]["uri"]
            .as_str()
            .is_some_and(|value| value.starts_with("sidecar://media/output-")));
        assert_eq!(
            payload["item"]["preview"].as_str(),
            Some("fixture-media-reference.png")
        );
        assert_eq!(payload["item"]["sidecarRef"]["kind"], "media");
        let expected_sha256 = format!(
            "sha256:{}",
            hex::encode(Sha256::digest(b"\x89PNG\r\n\x1a\nfixture"))
        );
        assert_eq!(
            payload["item"]["sidecarRef"]["sha256"].as_str(),
            Some(expected_sha256.as_str())
        );
        let relative_path = payload["item"]["sidecarRef"]["relativePath"]
            .as_str()
            .expect("relative path");
        assert!(store
            .read_bytes_verified(relative_path, None, MAX_IMAGE_MEDIA_BYTES as u64)
            .expect("read sidecar")
            .is_some());
    }

    #[test]
    fn safe_sidecar_output_is_left_unchanged() {
        let mut payload = json!({
            "item": {
                "type": "media",
                "uri": "sidecar://media/already-safe/image.png",
                "mime_type": "image/png",
                "source_path": "/tmp/private-image.png"
            }
        });

        persist_local_image_output_payload("item.completed", &mut payload, "session-1", None)
            .expect("safe output");

        assert_eq!(
            payload["item"]["uri"],
            "sidecar://media/already-safe/image.png"
        );
        assert!(!payload.to_string().contains("/tmp/private-image.png"));
        assert!(payload["item"].get("sidecarRef").is_none());
    }
}
