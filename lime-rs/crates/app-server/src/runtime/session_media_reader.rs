use super::session_media_refs::{
    known_media_sidecar_refs, session_scoped_media_relative_path, KnownMediaSidecarRef,
    RequestedMediaSidecar,
};
use super::sidecar_store::{SidecarReadBytesChunk, SidecarReadBytesResult};
use super::timestamp;
use super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::protocol::v2::{MediaReadParams, MediaReadResponse};
use app_server_protocol::AgentEvent;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde_json::{json, Value};

const DEFAULT_MAX_MEDIA_SIDECAR_BYTES: u64 = 8 * 1024 * 1024;
const MAX_MEDIA_SIDECAR_BYTES: u64 = 32 * 1024 * 1024;
const MEDIA_READ_CHUNK_EVENT_TYPE: &str = "media.read.chunk";
const MEDIA_READ_COMPLETED_EVENT_TYPE: &str = "media.read.completed";

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

    pub(crate) fn read_media_streaming_with_cancel(
        &self,
        params: MediaReadParams,
        is_canceled: impl Fn() -> bool,
        on_event: &mut impl FnMut(AgentEvent) -> Result<(), RuntimeCoreError>,
    ) -> Result<MediaReadResponse, RuntimeCoreError> {
        if !params.stream {
            return self.read_media_with_cancel(params, is_canceled);
        }
        let resolved = self.resolve_media_read_request(&params)?;
        fail_if_canceled(&is_canceled)?;
        let stream_id = media_read_stream_id(&resolved.params.thread_id, resolved.offset);
        let mut chunk_index = 0_u64;
        let mut on_chunk = |chunk: SidecarReadBytesChunk| {
            chunk_index += 1;
            let event = media_read_chunk_event(&resolved, &stream_id, chunk_index, &chunk);
            if let Err(error) = on_event(event) {
                tracing::warn!("failed to send media/read streaming chunk: {}", error);
            }
        };
        let content = resolved
            .sidecar_store
            .stream_bytes_range_verified_with_cancel(
                resolved.relative_path.as_str(),
                resolved.known_ref.sha256.as_deref(),
                resolved.offset,
                resolved.length,
                resolved.max_bytes,
                &is_canceled,
                &mut on_chunk,
            )
            .map_err(sidecar_read_error)?
            .ok_or_else(|| {
                RuntimeCoreError::Backend(
                    "agent session media sidecar content is not available".to_string(),
                )
            })?;
        drop(on_chunk);
        fail_if_canceled(&is_canceled)?;
        validate_known_media_size(&resolved.known_ref, content.total_bytes)?;
        let response = media_read_response(
            &resolved.params,
            &resolved.requested,
            &resolved.known_ref,
            &content,
        );
        on_event(media_read_completed_event(
            &resolved,
            &stream_id,
            chunk_index,
            &response,
        ))?;
        Ok(response)
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
        let (known_ref, session_id, base_sequence) = {
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
            (
                known_ref,
                stored.session.session_id.clone(),
                stored.events.len() as u64,
            )
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
            session_id,
            base_sequence,
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

fn media_read_stream_id(session_id: &str, offset: u64) -> String {
    format!(
        "media-read:{}:{}:{}",
        safe_event_id_component(session_id),
        offset,
        safe_event_id_component(&timestamp())
    )
}

fn media_read_chunk_event(
    resolved: &ResolvedMediaRead,
    stream_id: &str,
    chunk_index: u64,
    chunk: &SidecarReadBytesChunk,
) -> AgentEvent {
    media_read_event(
        resolved,
        stream_id,
        chunk_index,
        MEDIA_READ_CHUNK_EVENT_TYPE,
        json!({
            "streamId": stream_id,
            "chunkIndex": chunk_index,
            "done": false,
            "chunk": {
                "threadId": resolved.params.thread_id.clone(),
                "uri": resolved.known_ref.display_uri(&resolved.requested),
                "mimeType": resolved.known_ref.mime_type.clone(),
                "bytes": chunk.bytes.len() as u64,
                "totalBytes": chunk.total_bytes,
                "offset": chunk.offset,
                "length": chunk.length,
                "contentRange": format_content_range(chunk.offset, chunk.length, chunk.total_bytes),
                "hasMore": chunk.has_more,
                "contentBase64": BASE64_STANDARD.encode(&chunk.bytes),
                "sidecarRef": resolved.known_ref.sidecar_ref.clone(),
            },
        }),
    )
}

fn media_read_completed_event(
    resolved: &ResolvedMediaRead,
    stream_id: &str,
    chunk_count: u64,
    response: &MediaReadResponse,
) -> AgentEvent {
    media_read_event(
        resolved,
        stream_id,
        chunk_count.saturating_add(1),
        MEDIA_READ_COMPLETED_EVENT_TYPE,
        json!({
            "streamId": stream_id,
            "chunkCount": chunk_count,
            "done": true,
            "media": {
                "threadId": response.thread_id.clone(),
                "uri": response.uri.clone(),
                "mimeType": response.mime_type.clone(),
                "bytes": response.bytes,
                "totalBytes": response.total_bytes,
                "offset": response.offset,
                "length": response.length,
                "contentRange": response.content_range.clone(),
                "hasMore": response.has_more,
                "sha256": response.sha256.clone(),
                "sidecarRef": response.sidecar_ref.clone(),
            },
        }),
    )
}

fn media_read_event(
    resolved: &ResolvedMediaRead,
    stream_id: &str,
    sequence_offset: u64,
    event_type: &str,
    payload: Value,
) -> AgentEvent {
    AgentEvent {
        event_id: format!(
            "evt_{}_{}",
            safe_event_id_component(stream_id),
            sequence_offset
        ),
        sequence: resolved.base_sequence.saturating_add(sequence_offset),
        session_id: resolved.session_id.clone(),
        thread_id: Some(resolved.params.thread_id.clone()),
        turn_id: None,
        event_type: event_type.to_string(),
        timestamp: timestamp(),
        payload,
    }
}

fn safe_event_id_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
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
    session_id: String,
    base_sequence: u64,
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
    use serde_json::json;
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
                stream: false,
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
                stream: false,
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
    fn streams_known_media_sidecar_chunks_before_completed_event() {
        let (core, _temp, ref_id) = prepared_core_with_media_ref(None);
        let mut events = Vec::new();

        let response = core
            .read_media_streaming_with_cancel(
                MediaReadParams {
                    thread_id: "thread-media-read".to_string(),
                    uri: Some(ref_id.clone()),
                    ref_id: None,
                    sidecar_ref: None,
                    max_bytes: Some(1024),
                    offset: None,
                    length: Some(4),
                    stream: true,
                },
                || false,
                &mut |event| {
                    events.push(event);
                    Ok(())
                },
            )
            .expect("stream media");

        assert_eq!(response.content_base64, "iVBORw==");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, MEDIA_READ_CHUNK_EVENT_TYPE);
        assert_eq!(events[0].session_id, "sess-media-read");
        assert_eq!(events[0].thread_id.as_deref(), Some("thread-media-read"));
        assert_eq!(events[0].payload["done"], false);
        assert_eq!(events[0].payload["chunk"]["contentBase64"], "iVBORw==");
        assert_eq!(events[0].payload["chunk"]["offset"], 0);
        assert_eq!(events[0].payload["chunk"]["length"], 4);
        assert_eq!(events[1].event_type, MEDIA_READ_COMPLETED_EVENT_TYPE);
        assert_eq!(events[1].payload["done"], true);
        assert_eq!(events[1].payload["chunkCount"], 1);
        assert_eq!(events[1].payload["media"]["sha256"], response.sha256);
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
                stream: false,
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
                stream: false,
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
                stream: false,
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
            .and_then(Value::as_str)
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
                stream: false,
            })
            .expect_err("digest mismatch");

        assert!(error.to_string().contains("校验失败"));
    }
}
