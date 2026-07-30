use super::image_media::{extension_for_media_type, validate_image_bytes, MAX_IMAGE_MEDIA_BYTES};
use super::sidecar_store::{
    session_scoped_relative_path, SidecarBytesWriteRequest, SidecarRef, SidecarStore,
};
use agent_protocol::AgentInput;
use agent_runtime::reply_input::{
    RuntimeReplyInput, RuntimeReplyInputImage, RuntimeReplyInputMedia,
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

const INPUT_MEDIA_URI_PREFIX: &str = "sidecar://media/input-";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreparedRuntimeInput {
    pub(super) durable: Vec<AgentInput>,
    pub(super) provider: RuntimeReplyInput,
}

pub(super) fn prepare_runtime_input(
    input: Vec<AgentInput>,
    sidecar_store: Option<&SidecarStore>,
    session_id: &str,
) -> Result<PreparedRuntimeInput, String> {
    let provider = RuntimeReplyInput::try_from_user_parts(input.clone(), |media| {
        resolve_runtime_input_media(media, sidecar_store, session_id)
    })
    .map_err(|error| error.to_string())?;
    let resolved_images = provider.images().cloned().collect::<Vec<_>>();
    let mut resolved_images = resolved_images.into_iter();
    let durable = input
        .into_iter()
        .map(|part| match part {
            AgentInput::Image { .. } | AgentInput::LocalImage { .. } => {
                let image = resolved_images
                    .next()
                    .expect("provider input image count must match canonical input");
                AgentInput::Image {
                    uri: image.uri,
                    detail: image.detail,
                }
            }
            part => part,
        })
        .collect::<Vec<_>>();
    debug_assert!(resolved_images.next().is_none());
    Ok(PreparedRuntimeInput { durable, provider })
}

pub(super) fn resolve_runtime_input_media(
    media: RuntimeReplyInputMedia,
    sidecar_store: Option<&SidecarStore>,
    session_id: &str,
) -> Result<RuntimeReplyInputImage, String> {
    match media {
        RuntimeReplyInputMedia::Image { uri, detail } => {
            let uri = uri.trim();
            if uri.to_ascii_lowercase().starts_with("data:") {
                let store = sidecar_store.ok_or_else(|| {
                    "inline provider media requires an initialized App Server sidecar store"
                        .to_string()
                })?;
                let (bytes, media_type) = decode_image_data_url(uri)?;
                validate_image_bytes(&bytes, Some(&media_type))?;
                let sidecar_ref = persist_image_bytes(&bytes, &media_type, store, session_id)?;
                return Ok(RuntimeReplyInputImage {
                    uri: sidecar_ref.ref_id,
                    media_type: media_type.clone(),
                    provider_data: Some(format!(
                        "data:{media_type};base64,{}",
                        BASE64_STANDARD.encode(bytes)
                    )),
                    detail,
                });
            }
            if uri.to_ascii_lowercase().starts_with("sidecar://") {
                return hydrate_persisted_image(uri, detail, sidecar_store, session_id);
            }
            let parsed = url::Url::parse(uri)
                .map_err(|error| format!("remote provider image URL is invalid: {error}"))?;
            if !matches!(parsed.scheme(), "http" | "https") {
                return Err("remote provider image must use http or https".to_string());
            }
            Ok(RuntimeReplyInputImage {
                uri: parsed.to_string(),
                media_type: remote_image_media_type(&parsed).to_string(),
                provider_data: None,
                detail,
            })
        }
        RuntimeReplyInputMedia::LocalImage { path, detail } => {
            let path = Path::new(path.trim());
            if path.as_os_str().is_empty() {
                return Err("local image path must not be empty".to_string());
            }
            let store = sidecar_store.ok_or_else(|| {
                "local provider image requires an initialized App Server sidecar store".to_string()
            })?;
            let metadata = std::fs::metadata(path)
                .map_err(|error| format!("read local image metadata failed: {error}"))?;
            if !metadata.is_file() {
                return Err("local image path must reference a regular file".to_string());
            }
            if metadata.len() > MAX_IMAGE_MEDIA_BYTES as u64 {
                return Err(format!(
                    "provider image input exceeds {} bytes",
                    MAX_IMAGE_MEDIA_BYTES
                ));
            }
            let bytes =
                std::fs::read(path).map_err(|error| format!("read local image failed: {error}"))?;
            let media_type = validate_image_bytes(&bytes, None)?.to_string();
            let sidecar_ref = persist_image_bytes(&bytes, &media_type, store, session_id)?;
            Ok(RuntimeReplyInputImage {
                uri: sidecar_ref.ref_id,
                media_type: media_type.clone(),
                provider_data: Some(format!(
                    "data:{media_type};base64,{}",
                    BASE64_STANDARD.encode(bytes)
                )),
                detail,
            })
        }
    }
}

fn decode_image_data_url(source: &str) -> Result<(Vec<u8>, String), String> {
    let (metadata, encoded) = source
        .strip_prefix("data:")
        .and_then(|value| value.split_once(','))
        .ok_or_else(|| "image data URL is malformed".to_string())?;
    if !metadata
        .split(';')
        .any(|part| part.eq_ignore_ascii_case("base64"))
    {
        return Err("image data URL must use base64 encoding".to_string());
    }
    let media_type = metadata
        .split(';')
        .next()
        .map(str::trim)
        .filter(|value| {
            matches!(
                *value,
                "image/png" | "image/jpeg" | "image/gif" | "image/webp"
            )
        })
        .ok_or_else(|| "provider image input uses an unsupported media type".to_string())?
        .to_string();
    let bytes = BASE64_STANDARD
        .decode(encoded.trim())
        .map_err(|error| format!("image data URL base64 decode failed: {error}"))?;
    if bytes.is_empty() {
        return Err("provider image input is empty".to_string());
    }
    if bytes.len() > MAX_IMAGE_MEDIA_BYTES {
        return Err(format!(
            "provider image input exceeds {} bytes",
            MAX_IMAGE_MEDIA_BYTES
        ));
    }
    Ok((bytes, media_type))
}

fn persist_image_bytes(
    bytes: &[u8],
    media_type: &str,
    store: &SidecarStore,
    session_id: &str,
) -> Result<SidecarRef, String> {
    let digest = hex::encode(Sha256::digest(bytes));
    let relative_path = session_scoped_relative_path(
        session_id,
        &format!(
            "media/input-{digest}.{}",
            extension_for_media_type(media_type)
        ),
    );
    let mut sidecar_ref = store.write_bytes(&SidecarBytesWriteRequest {
        session_id: session_id.to_string(),
        kind: "media".to_string(),
        logical_id: format!("input-{digest}"),
        relative_path,
        content: bytes.to_vec(),
    })?;
    sidecar_ref.ref_id = canonical_input_media_uri(&digest, media_type);
    Ok(sidecar_ref)
}

fn hydrate_persisted_image(
    uri: &str,
    detail: Option<agent_protocol::ImageDetail>,
    sidecar_store: Option<&SidecarStore>,
    session_id: &str,
) -> Result<RuntimeReplyInputImage, String> {
    let store = sidecar_store.ok_or_else(|| {
        "canonical provider media requires an initialized App Server sidecar store".to_string()
    })?;
    let locator = parse_canonical_input_media_uri(uri)?;
    let relative_path = session_scoped_relative_path(
        session_id,
        &format!("media/input-{}.{}", locator.digest, locator.extension),
    );
    let expected_sha256 = format!("sha256:{}", locator.digest);
    let read = store
        .read_bytes_verified(
            &relative_path,
            Some(&expected_sha256),
            MAX_IMAGE_MEDIA_BYTES as u64,
        )?
        .ok_or_else(|| format!("canonical provider image sidecar is missing: {uri}"))?;
    let detected = validate_image_bytes(&read.bytes, Some(locator.media_type))?;
    Ok(RuntimeReplyInputImage {
        uri: uri.to_string(),
        media_type: detected.to_string(),
        provider_data: Some(format!(
            "data:{detected};base64,{}",
            BASE64_STANDARD.encode(read.bytes)
        )),
        detail,
    })
}

pub(super) fn copy_canonical_input_media_for_fork(
    input: &[AgentInput],
    sidecar_store: Option<&SidecarStore>,
    source_session_id: &str,
    target_session_id: &str,
) -> Result<(), String> {
    for part in input {
        let AgentInput::Image { uri, .. } = part else {
            continue;
        };
        if !uri.starts_with(INPUT_MEDIA_URI_PREFIX) {
            continue;
        }
        copy_canonical_input_media(uri, sidecar_store, source_session_id, target_session_id)?;
    }
    Ok(())
}

fn copy_canonical_input_media(
    uri: &str,
    sidecar_store: Option<&SidecarStore>,
    source_session_id: &str,
    target_session_id: &str,
) -> Result<(), String> {
    let store = sidecar_store.ok_or_else(|| {
        "canonical provider media requires an initialized App Server sidecar store".to_string()
    })?;
    let locator = parse_canonical_input_media_uri(uri)?;
    let file_name = format!("media/input-{}.{}", locator.digest, locator.extension);
    let source_relative_path = session_scoped_relative_path(source_session_id, &file_name);
    let expected_sha256 = format!("sha256:{}", locator.digest);
    let read = store
        .read_bytes_verified(
            &source_relative_path,
            Some(&expected_sha256),
            MAX_IMAGE_MEDIA_BYTES as u64,
        )?
        .ok_or_else(|| format!("canonical provider image sidecar is missing: {uri}"))?;
    validate_image_bytes(&read.bytes, Some(locator.media_type))?;
    store.write_bytes(&SidecarBytesWriteRequest {
        session_id: target_session_id.to_string(),
        kind: "media".to_string(),
        logical_id: format!("input-{}", locator.digest),
        relative_path: session_scoped_relative_path(target_session_id, &file_name),
        content: read.bytes,
    })?;
    Ok(())
}

pub(super) fn attach_input_media_output_refs(
    payload: &mut Value,
    sidecar_store: Option<&SidecarStore>,
    session_id: &str,
) -> Result<(), String> {
    let Some(input) = payload
        .get("input")
        .cloned()
        .and_then(|value| serde_json::from_value::<Vec<AgentInput>>(value).ok())
    else {
        return Ok(());
    };
    let mut output_refs = Vec::new();
    for part in input {
        let AgentInput::Image { uri, .. } = part else {
            continue;
        };
        if !uri.starts_with(INPUT_MEDIA_URI_PREFIX) {
            continue;
        }
        output_refs.push(input_media_output_ref(&uri, sidecar_store, session_id)?);
    }
    let Some(payload) = payload.as_object_mut() else {
        return Ok(());
    };
    if output_refs.is_empty() {
        payload.remove("outputRefs");
    } else {
        payload.insert("outputRefs".to_string(), Value::Array(output_refs));
    }
    Ok(())
}

fn input_media_output_ref(
    uri: &str,
    sidecar_store: Option<&SidecarStore>,
    session_id: &str,
) -> Result<Value, String> {
    let store = sidecar_store.ok_or_else(|| {
        "canonical provider media requires an initialized App Server sidecar store".to_string()
    })?;
    let locator = parse_canonical_input_media_uri(uri)?;
    let relative_path = session_scoped_relative_path(
        session_id,
        &format!("media/input-{}.{}", locator.digest, locator.extension),
    );
    let expected_sha256 = format!("sha256:{}", locator.digest);
    let read = store
        .read_bytes_verified(
            &relative_path,
            Some(&expected_sha256),
            MAX_IMAGE_MEDIA_BYTES as u64,
        )?
        .ok_or_else(|| format!("canonical provider image sidecar is missing: {uri}"))?;
    validate_image_bytes(&read.bytes, Some(locator.media_type))?;
    Ok(json!({
        "ref": uri,
        "kind": "media",
        "relativePath": relative_path,
        "bytes": read.total_bytes,
        "sha256": read.sha256,
        "contentStatus": "available",
        "mimeType": locator.media_type,
    }))
}

fn canonical_input_media_uri(digest: &str, media_type: &str) -> String {
    format!(
        "{INPUT_MEDIA_URI_PREFIX}{digest}.{}",
        extension_for_media_type(media_type)
    )
}

struct CanonicalInputMediaLocator<'a> {
    digest: &'a str,
    extension: &'a str,
    media_type: &'static str,
}

fn parse_canonical_input_media_uri(uri: &str) -> Result<CanonicalInputMediaLocator<'_>, String> {
    let locator = uri.strip_prefix(INPUT_MEDIA_URI_PREFIX).ok_or_else(|| {
        "provider image sidecar URI is not a canonical input reference".to_string()
    })?;
    let (digest, extension) = locator
        .rsplit_once('.')
        .ok_or_else(|| "provider image sidecar URI is missing its media extension".to_string())?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("provider image sidecar URI has an invalid content digest".to_string());
    }
    let media_type = media_type_for_extension(extension).ok_or_else(|| {
        "provider image sidecar URI has an unsupported media extension".to_string()
    })?;
    Ok(CanonicalInputMediaLocator {
        digest,
        extension,
        media_type,
    })
}

fn media_type_for_extension(extension: &str) -> Option<&'static str> {
    match extension {
        "png" => Some("image/png"),
        "jpg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

fn remote_image_media_type(url: &url::Url) -> &'static str {
    let path = url.path().to_ascii_lowercase();
    if path.ends_with(".png") {
        "image/png"
    } else if path.ends_with(".jpg") || path.ends_with(".jpeg") {
        "image/jpeg"
    } else if path.ends_with(".gif") {
        "image/gif"
    } else if path.ends_with(".webp") {
        "image/webp"
    } else {
        "image/*"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_protocol::ImageDetail;

    const PNG_DATA_URL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";

    #[test]
    fn typed_local_image_is_validated_and_sidecarized_for_provider() {
        let root = tempfile::tempdir().expect("sidecar root");
        let store = SidecarStore::new(root.path()).expect("sidecar store");
        let image_path = root.path().join("input.png");
        let (_, encoded) = PNG_DATA_URL.split_once(',').expect("png data URL");
        let bytes = BASE64_STANDARD.decode(encoded).expect("decode fixture");
        std::fs::write(&image_path, bytes).expect("write local image");

        let image = resolve_runtime_input_media(
            RuntimeReplyInputMedia::LocalImage {
                path: image_path.display().to_string(),
                detail: Some(ImageDetail::Original),
            },
            Some(&store),
            "session-1",
        )
        .expect("resolve local image");

        assert!(image.uri.starts_with("sidecar://media/"));
        assert_eq!(image.media_type, "image/png");
        assert_eq!(image.provider_data.as_deref(), Some(PNG_DATA_URL));
        assert_eq!(image.detail, Some(ImageDetail::Original));
        assert!(!image
            .provider_data
            .as_deref()
            .is_some_and(|value| value.contains(&image_path.display().to_string())));
    }

    #[test]
    fn typed_remote_and_inline_images_keep_native_provider_shapes() {
        let remote = resolve_runtime_input_media(
            RuntimeReplyInputMedia::Image {
                uri: "https://example.com/assets/image.webp?version=1".to_string(),
                detail: Some(ImageDetail::High),
            },
            None,
            "session-1",
        )
        .expect("resolve remote image");
        assert_eq!(
            remote.uri,
            "https://example.com/assets/image.webp?version=1"
        );
        assert_eq!(remote.media_type, "image/webp");
        assert_eq!(remote.provider_data, None);
        assert_eq!(remote.detail, Some(ImageDetail::High));

        let root = tempfile::tempdir().expect("sidecar root");
        let store = SidecarStore::new(root.path()).expect("sidecar store");
        let inline = resolve_runtime_input_media(
            RuntimeReplyInputMedia::Image {
                uri: PNG_DATA_URL.to_string(),
                detail: Some(ImageDetail::Low),
            },
            Some(&store),
            "session-1",
        )
        .expect("resolve inline image");
        assert!(inline.uri.starts_with("sidecar://media/"));
        assert_eq!(inline.provider_data.as_deref(), Some(PNG_DATA_URL));
        assert_eq!(inline.detail, Some(ImageDetail::Low));
    }

    #[test]
    fn prepared_input_separates_durable_reference_from_provider_payload() {
        let root = tempfile::tempdir().expect("sidecar root");
        let store = SidecarStore::new(root.path()).expect("sidecar store");
        let prepared = prepare_runtime_input(
            vec![
                AgentInput::text("describe it"),
                AgentInput::Image {
                    uri: PNG_DATA_URL.to_string(),
                    detail: Some(ImageDetail::High),
                },
            ],
            Some(&store),
            "session-1",
        )
        .expect("prepare provider input");

        let durable_json =
            serde_json::to_string(&prepared.durable).expect("serialize durable input");
        assert!(!durable_json.contains("base64,"));
        let durable_uri = match &prepared.durable[1] {
            AgentInput::Image { uri, detail } => {
                assert_eq!(*detail, Some(ImageDetail::High));
                uri.clone()
            }
            other => panic!("expected canonical image, got {other:?}"),
        };
        assert!(durable_uri.starts_with(INPUT_MEDIA_URI_PREFIX));
        assert!(durable_uri.ends_with(".png"));
        let provider_image = prepared.provider.images().next().expect("provider image");
        assert_eq!(provider_image.uri, durable_uri);
        assert_eq!(provider_image.provider_data.as_deref(), Some(PNG_DATA_URL));

        let hydrated = prepare_runtime_input(prepared.durable.clone(), Some(&store), "session-1")
            .expect("hydrate durable provider input");
        assert_eq!(hydrated.durable, prepared.durable);
        assert_eq!(
            hydrated
                .provider
                .images()
                .next()
                .and_then(|image| image.provider_data.as_deref()),
            Some(PNG_DATA_URL)
        );
    }

    #[test]
    fn canonical_input_event_uses_verified_output_refs_without_inline_payload() {
        let root = tempfile::tempdir().expect("sidecar root");
        let store = SidecarStore::new(root.path()).expect("sidecar store");
        let prepared = prepare_runtime_input(
            vec![AgentInput::Image {
                uri: PNG_DATA_URL.to_string(),
                detail: None,
            }],
            Some(&store),
            "session-1",
        )
        .expect("prepare provider input");
        let mut payload = json!({"input": prepared.durable});

        attach_input_media_output_refs(&mut payload, Some(&store), "session-1")
            .expect("attach canonical output refs");

        let encoded = serde_json::to_string(&payload).expect("serialize canonical event");
        assert!(!encoded.contains("base64,"));
        let output_ref = &payload["outputRefs"][0];
        assert!(output_ref["ref"]
            .as_str()
            .is_some_and(|uri| uri.starts_with(INPUT_MEDIA_URI_PREFIX)));
        assert_eq!(output_ref["kind"], "media");
        assert_eq!(output_ref["mimeType"], "image/png");
        assert!(output_ref["bytes"].as_u64().is_some_and(|bytes| bytes > 0));
        assert!(output_ref["sha256"]
            .as_str()
            .is_some_and(|digest| digest.starts_with("sha256:")));
        assert!(output_ref["relativePath"]
            .as_str()
            .is_some_and(|path| path.starts_with("sessions/session-1/media/input-")));
    }

    #[test]
    fn typed_media_rejects_local_leaks_and_mismatched_payloads() {
        let without_store = resolve_runtime_input_media(
            RuntimeReplyInputMedia::LocalImage {
                path: "/workspace/image.png".to_string(),
                detail: None,
            },
            None,
            "session-1",
        )
        .expect_err("local image requires sidecar");
        assert!(without_store.contains("sidecar store"));

        let root = tempfile::tempdir().expect("sidecar root");
        let store = SidecarStore::new(root.path()).expect("sidecar store");
        let mismatched = resolve_runtime_input_media(
            RuntimeReplyInputMedia::Image {
                uri: PNG_DATA_URL.replacen("image/png", "image/jpeg", 1),
                detail: None,
            },
            Some(&store),
            "session-1",
        )
        .expect_err("declared MIME must match image signature");
        assert!(mismatched.contains("media type mismatch"));

        assert!(resolve_runtime_input_media(
            RuntimeReplyInputMedia::Image {
                uri: "file:///workspace/image.png".to_string(),
                detail: None,
            },
            Some(&store),
            "session-1",
        )
        .expect_err("remote image cannot use file scheme")
        .contains("http or https"));
    }
}
