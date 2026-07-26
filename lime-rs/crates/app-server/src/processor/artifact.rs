use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{
    ArtifactSnapshot, ArtifactWriteParams, ArtifactWriteResponse, ArtifactWriteSidecar,
};
use app_server_protocol::{error_codes, JsonRpcError};
use serde_json::{json, Map, Value};

impl RequestProcessor {
    pub(super) async fn handle_artifact_write_v2_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ArtifactWriteParams = parse_params(params)?;
        validate_artifact_write(&params)?;
        let session_id = self
            .resolve_persisted_v2_thread_session(&params.thread_id)
            .await?;
        let payload = artifact_snapshot_payload(&params.artifact);
        let events = self
            .runtime
            .append_artifact_snapshot(&session_id, params.turn_id.as_deref(), payload)
            .map_err(to_jsonrpc_error)?;
        let event = events.into_iter().next().ok_or_else(|| {
            JsonRpcError::new(
                error_codes::RUNTIME_ERROR,
                "artifact/write did not persist an artifact snapshot",
            )
        })?;
        let sidecar = artifact_write_sidecar(&event.payload)?;

        dispatch_result(ArtifactWriteResponse {
            thread_id: params.thread_id,
            turn_id: event.turn_id,
            artifact_ref: params.artifact.artifact_ref,
            artifact_document_id: params.artifact.artifact_document_id,
            event_id: event.event_id,
            sequence: event.sequence,
            persisted_at: event.timestamp,
            sidecar,
        })
    }
}

fn validate_artifact_write(params: &ArtifactWriteParams) -> Result<(), JsonRpcError> {
    for (field, value) in [
        ("threadId", params.thread_id.as_str()),
        (
            "artifact.artifactRef",
            params.artifact.artifact_ref.as_str(),
        ),
        ("artifact.kind", params.artifact.kind.as_str()),
        ("artifact.content", params.artifact.content.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(JsonRpcError::new(
                error_codes::INVALID_PARAMS,
                format!("{field} must not be empty"),
            ));
        }
    }
    if params
        .turn_id
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(JsonRpcError::new(
            error_codes::INVALID_PARAMS,
            "turnId must not be empty",
        ));
    }
    Ok(())
}

fn artifact_snapshot_payload(artifact: &ArtifactSnapshot) -> Value {
    let mut value = Map::new();
    value.insert("artifactId".to_string(), json!(artifact.artifact_ref));
    value.insert("artifactRef".to_string(), json!(artifact.artifact_ref));
    value.insert("kind".to_string(), json!(artifact.kind));
    value.insert("content".to_string(), json!(artifact.content));
    insert_optional(
        &mut value,
        "artifactDocumentId",
        &artifact.artifact_document_id,
    );
    insert_optional(&mut value, "filePath", &artifact.path);
    insert_optional(&mut value, "path", &artifact.path);
    insert_optional(&mut value, "title", &artifact.title);
    insert_optional(&mut value, "status", &artifact.status);
    if let Some(metadata) = &artifact.metadata {
        value.insert("metadata".to_string(), metadata.clone());
    }
    json!({ "artifact": value })
}

fn insert_optional(value: &mut Map<String, Value>, key: &str, item: &Option<String>) {
    if let Some(item) = item {
        value.insert(key.to_string(), json!(item));
    }
}

fn artifact_write_sidecar(payload: &Value) -> Result<ArtifactWriteSidecar, JsonRpcError> {
    let sidecar = payload
        .pointer("/artifact/sidecarRef")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            JsonRpcError::new(
                error_codes::RUNTIME_ERROR,
                "artifact/write persisted no sidecar evidence",
            )
        })?;
    Ok(ArtifactWriteSidecar {
        relative_path: required_sidecar_string(sidecar, "relativePath")?,
        bytes: sidecar
            .get("bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                JsonRpcError::new(error_codes::RUNTIME_ERROR, "artifact sidecar bytes missing")
            })?,
        sha256: required_sidecar_string(sidecar, "sha256")?,
        content_status: required_sidecar_string(sidecar, "contentStatus")?,
    })
}

fn required_sidecar_string(
    sidecar: &Map<String, Value>,
    key: &str,
) -> Result<String, JsonRpcError> {
    sidecar
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            JsonRpcError::new(
                error_codes::RUNTIME_ERROR,
                format!("artifact sidecar {key} missing"),
            )
        })
}
