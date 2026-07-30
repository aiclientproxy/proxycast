use super::session_media_refs::{
    known_media_sidecar_refs, session_scoped_media_relative_path, KnownMediaSidecarRef,
    RequestedMediaSidecar,
};
use super::sidecar_store::SidecarReadBytesResult;
use super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::protocol::v2::{MediaReadParams, MediaReadResponse};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};

const DEFAULT_MAX_MEDIA_SIDECAR_BYTES: u64 = 8 * 1024 * 1024;
const MAX_MEDIA_SIDECAR_BYTES: u64 = 32 * 1024 * 1024;

impl RuntimeCore {
    pub fn read_media(
        &self,
        params: MediaReadParams,
    ) -> Result<MediaReadResponse, RuntimeCoreError> {
        self.read_media_with_cancel(params, || false)
    }

    pub(crate) fn read_media_with_cancel(
        &self,
        params: MediaReadParams,
        is_canceled: impl Fn() -> bool,
    ) -> Result<MediaReadResponse, RuntimeCoreError> {
        let resolved = self.resolve_media_read_request(&params)?;
        fail_if_canceled(&is_canceled)?;
        let content = resolved
            .sidecar_store
            .read_bytes_range_verified_with_cancel(
                resolved.relative_path.as_str(),
                resolved.known_ref.sha256.as_deref(),
                resolved.offset,
                resolved.length,
                resolved.max_bytes,
                &is_canceled,
            )
            .map_err(sidecar_read_error)?
            .ok_or_else(|| {
                RuntimeCoreError::Backend(
                    "agent session media sidecar content is not available".to_string(),
                )
            })?;
        fail_if_canceled(&is_canceled)?;
        validate_known_media_size(&resolved.known_ref, content.total_bytes)?;

        Ok(media_read_response(
            &resolved.params,
            &resolved.requested,
            &resolved.known_ref,
            &content,
        ))
    }

    fn resolve_media_read_request(
        &self,
        params: &MediaReadParams,
    ) -> Result<ResolvedMediaRead, RuntimeCoreError> {
        let requested = RequestedMediaSidecar::from_params(params)?;
        let sidecar_store = self.sidecar_store.as_ref().ok_or_else(|| {
            RuntimeCoreError::Backend(
                "media/read requires an initialized sidecar store".to_string(),
            )
        })?;
        let (known_ref, session_id) = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .values()
                .find(|stored| stored.session.thread_id == params.thread_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(params.thread_id.clone()))?;
            let known_ref = known_media_sidecar_refs(stored)
                .into_iter()
                .find(|candidate| candidate.matches(&requested))
                .ok_or_else(|| {
                    RuntimeCoreError::Backend(
                        "agent session media sidecar reference is not available".to_string(),
                    )
                })?;
            (known_ref, stored.session.session_id.clone())
        };
        let relative_path = session_scoped_media_relative_path(
            session_id.as_str(),
            known_ref.relative_path.as_str(),
        )?;
        let max_bytes = params
            .max_bytes
            .unwrap_or(DEFAULT_MAX_MEDIA_SIDECAR_BYTES)
            .min(MAX_MEDIA_SIDECAR_BYTES);
        let offset = params.offset.unwrap_or(0);
        let length = params.length.unwrap_or(max_bytes);
        Ok(ResolvedMediaRead {
            params: params.clone(),
            requested,
            known_ref,
            sidecar_store: sidecar_store.clone(),
            relative_path,
            max_bytes,
            offset,
            length,
        })
    }
}

fn validate_known_media_size(
    known_ref: &KnownMediaSidecarRef,
    actual_bytes: u64,
) -> Result<(), RuntimeCoreError> {
    if let Some(expected_bytes) = known_ref.bytes {
        if expected_bytes != actual_bytes {
            return Err(RuntimeCoreError::Backend(format!(
                "agent session media sidecar size mismatch: expected {expected_bytes}, actual {actual_bytes}"
            )));
        }
    }
    Ok(())
}

fn media_read_response(
    params: &MediaReadParams,
    requested: &RequestedMediaSidecar,
    known_ref: &KnownMediaSidecarRef,
    content: &SidecarReadBytesResult,
) -> MediaReadResponse {
    MediaReadResponse {
        thread_id: params.thread_id.clone(),
        uri: known_ref.display_uri(requested),
        mime_type: known_ref.mime_type.clone(),
        bytes: content.bytes.len() as u64,
        total_bytes: content.total_bytes,
        offset: content.offset,
        length: content.length,
        content_range: format_content_range(content.offset, content.length, content.total_bytes),
        has_more: content.has_more,
        sha256: content.sha256.clone(),
        content_base64: BASE64_STANDARD.encode(&content.bytes),
        sidecar_ref: Some(known_ref.sidecar_ref.clone()),
    }
}

#[derive(Debug, Clone)]
struct ResolvedMediaRead {
    params: MediaReadParams,
    requested: RequestedMediaSidecar,
    known_ref: KnownMediaSidecarRef,
    sidecar_store: std::sync::Arc<super::sidecar_store::SidecarStore>,
    relative_path: String,
    max_bytes: u64,
    offset: u64,
    length: u64,
}

fn fail_if_canceled(is_canceled: &impl Fn() -> bool) -> Result<(), RuntimeCoreError> {
    if is_canceled() {
        Err(RuntimeCoreError::RequestCanceled)
    } else {
        Ok(())
    }
}

fn sidecar_read_error(error: String) -> RuntimeCoreError {
    if error == super::sidecar_store::SIDECAR_READ_CANCELED {
        RuntimeCoreError::RequestCanceled
    } else {
        RuntimeCoreError::Backend(error)
    }
}

fn format_content_range(offset: u64, length: u64, total_bytes: u64) -> String {
    if length == 0 {
        return format!("bytes */{total_bytes}");
    }
    let end = offset.saturating_add(length).saturating_sub(1);
    format!("bytes {offset}-{end}/{total_bytes}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::sidecar_store::{SidecarBytesWriteRequest, SidecarStore};
    use crate::RuntimeEvent;
    use app_server_protocol::AgentSessionStartParams;
    use serde_json::{json, Value};
    use std::sync::Arc;

    fn prepared_core_with_media_ref(
        sidecar_ref_override: Option<Value>,
    ) -> (RuntimeCore, tempfile::TempDir, String) {
        let temp = tempfile::tempdir().expect("tempdir");
        let sidecar_store = Arc::new(SidecarStore::new(temp.path()).expect("sidecar store"));
        let core = RuntimeCore::default().with_sidecar_store(sidecar_store.clone());
        core.start_session(AgentSessionStartParams {
            session_id: Some("sess-media-read".to_string()),
            thread_id: Some("thread-media-read".to_string()),
            app_id: "agent-chat".to_string(),
            workspace_id: Some("default".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("session");
        let sidecar_ref = sidecar_store
            .write_bytes(&SidecarBytesWriteRequest {
                session_id: "sess-media-read".to_string(),
                kind: "media".to_string(),
                logical_id: "fixture-image".to_string(),
                relative_path: "sessions/sess-media-read/media/fixture-image.png".to_string(),
                content: vec![0x89, b'P', b'N', b'G'],
            })
            .expect("write media sidecar");
        let sidecar_ref_value =
            sidecar_ref_override.unwrap_or_else(|| serde_json::to_value(&sidecar_ref).unwrap());
        let ref_id = sidecar_ref.ref_id.clone();
        core.append_runtime_events(
            "sess-media-read",
            "thread-media-read",
            Some("turn-media-read"),
            vec![RuntimeEvent::new(
                "message.delta",
                json!({
                    "itemId": "agent-media-1",
                    "contentPart": {
                        "type": "media",
                        "kind": "image",
                        "reference": {
                            "uri": ref_id,
                            "mime_type": "image/png",
                            "sidecar_ref": sidecar_ref_value
                        }
                    }
                }),
            )],
        )
        .expect("append media event");
        (core, temp, ref_id)
    }

    fn prepared_core_with_artifact_sidecar(
        artifact_kind: &str,
        mime_type: &str,
        content: Vec<u8>,
    ) -> (RuntimeCore, tempfile::TempDir) {
        let temp = tempfile::tempdir().expect("tempdir");
        let sidecar_store = Arc::new(SidecarStore::new(temp.path()).expect("sidecar store"));
        let core = RuntimeCore::default().with_sidecar_store(sidecar_store.clone());
        core.start_session(AgentSessionStartParams {
            session_id: Some("sess-artifact-media-read".to_string()),
            thread_id: Some("thread-artifact-media-read".to_string()),
            app_id: "agent-chat".to_string(),
            workspace_id: Some("default".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("session");
        let sidecar_ref = sidecar_store
            .write_bytes(&SidecarBytesWriteRequest {
                session_id: "sess-artifact-media-read".to_string(),
                kind: "artifact_snapshot".to_string(),
                logical_id: "artifact-image-1".to_string(),
                relative_path:
                    "sessions/sess-artifact-media-read/runtime-artifacts/artifact-image-1.bin"
                        .to_string(),
                content,
            })
            .expect("write artifact sidecar");
        core.append_runtime_events(
            "sess-artifact-media-read",
            "thread-artifact-media-read",
            Some("turn-artifact-media-read"),
            vec![RuntimeEvent::new(
                "artifact.snapshot",
                json!({
                    "artifact": {
                        "artifactId": "artifact://message/image-1",
                        "path": ".lime/artifacts/image-1.bin",
                        "kind": artifact_kind,
                        "mimeType": mime_type,
                        "sidecarRef": sidecar_ref,
                    }
                }),
            )],
        )
        .expect("append artifact event");
        (core, temp)
    }

    #[test]
    fn reads_known_media_sidecar_bytes_with_digest_check() {
        let (core, _temp, ref_id) = prepared_core_with_media_ref(None);

        let response = core
            .read_media(MediaReadParams {
                thread_id: "thread-media-read".to_string(),
                uri: Some(ref_id.clone()),
                ref_id: None,
                sidecar_ref: None,
                max_bytes: Some(1024),
                offset: None,
                length: None,
            })
            .expect("read media");

        assert_eq!(response.thread_id, "thread-media-read");
        assert_eq!(response.uri, ref_id);
        assert_eq!(response.mime_type.as_deref(), Some("image/png"));
        assert_eq!(response.bytes, 4);
        assert_eq!(response.total_bytes, 4);
        assert_eq!(response.offset, 0);
        assert_eq!(response.length, 4);
        assert_eq!(response.content_range, "bytes 0-3/4");
        assert!(!response.has_more);
        assert_eq!(response.content_base64, "iVBORw==");
        assert!(response.sha256.starts_with("sha256:"));
        assert!(response.sidecar_ref.is_some());
    }

    #[test]
    fn reads_known_media_sidecar_range_with_full_digest_check() {
        let (core, _temp, ref_id) = prepared_core_with_media_ref(None);

        let response = core
            .read_media(MediaReadParams {
                thread_id: "thread-media-read".to_string(),
                uri: Some(ref_id.clone()),
                ref_id: None,
                sidecar_ref: None,
                max_bytes: Some(2),
                offset: Some(1),
                length: Some(2),
            })
            .expect("read media range");

        assert_eq!(response.thread_id, "thread-media-read");
        assert_eq!(response.uri, ref_id);
        assert_eq!(response.bytes, 2);
        assert_eq!(response.total_bytes, 4);
        assert_eq!(response.offset, 1);
        assert_eq!(response.length, 2);
        assert_eq!(response.content_range, "bytes 1-2/4");
        assert!(response.has_more);
        assert_eq!(response.content_base64, "UE4=");
        assert!(response.sha256.starts_with("sha256:"));
    }

    #[test]
    fn reads_media_artifact_sidecar_by_artifact_uri_alias() {
        let (core, _temp) =
            prepared_core_with_artifact_sidecar("image", "image/png", vec![0x89, b'P', b'N', b'G']);

        let response = core
            .read_media(MediaReadParams {
                thread_id: "thread-artifact-media-read".to_string(),
                uri: Some("artifact://message/image-1".to_string()),
                ref_id: None,
                sidecar_ref: None,
                max_bytes: Some(1024),
                offset: None,
                length: None,
            })
            .expect("read media artifact");

        assert_eq!(response.thread_id, "thread-artifact-media-read");
        assert_eq!(response.mime_type.as_deref(), Some("image/png"));
        assert_eq!(response.bytes, 4);
        assert_eq!(response.content_base64, "iVBORw==");
        assert_eq!(
            response
                .sidecar_ref
                .as_ref()
                .and_then(|sidecar_ref| sidecar_ref.get("kind"))
                .and_then(Value::as_str),
            Some("artifact_snapshot")
        );
    }

    #[test]
    fn rejects_non_media_artifact_sidecar_alias() {
        let (core, _temp) = prepared_core_with_artifact_sidecar(
            "markdown_report",
            "text/markdown",
            b"# Report".to_vec(),
        );

        let error = core
            .read_media(MediaReadParams {
                thread_id: "thread-artifact-media-read".to_string(),
                uri: Some("artifact://message/image-1".to_string()),
                ref_id: None,
                sidecar_ref: None,
                max_bytes: Some(1024),
                offset: None,
                length: None,
            })
            .expect_err("non-media artifact must not be readable as media");

        assert!(error.to_string().contains("reference is not available"));
    }

    #[test]
    fn rejects_unknown_media_sidecar_ref() {
        let (core, _temp, _ref_id) = prepared_core_with_media_ref(None);

        let error = core
            .read_media(MediaReadParams {
                thread_id: "thread-media-read".to_string(),
                uri: Some("sidecar://media/missing".to_string()),
                ref_id: None,
                sidecar_ref: None,
                max_bytes: Some(1024),
                offset: None,
                length: None,
            })
            .expect_err("unknown ref");

        assert!(error.to_string().contains("reference is not available"));
    }

    #[test]
    fn rejects_media_sidecar_digest_mismatch() {
        let bad_ref = json!({
            "ref": "sidecar://media/bad",
            "kind": "media",
            "relativePath": "sessions/sess-media-read/media/fixture-image.png",
            "bytes": 4,
            "sha256": "sha256:bad"
        });
        let (core, _temp, _ref_id) = prepared_core_with_media_ref(Some(bad_ref.clone()));
        let ref_id = bad_ref
            .get("ref")
            .and_then(|value| value.as_str())
            .unwrap()
            .to_string();

        let error = core
            .read_media(MediaReadParams {
                thread_id: "thread-media-read".to_string(),
                uri: Some(ref_id),
                ref_id: None,
                sidecar_ref: None,
                max_bytes: Some(1024),
                offset: None,
                length: None,
            })
            .expect_err("digest mismatch");

        assert!(error.to_string().contains("校验失败"));
    }
}
