use crate::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolPolicyErrorKind, RuntimeToolTurnContext,
};
use app_server_protocol::MediaTaskArtifactVideoCreateParams;
use serde_json::{Map as JsonMap, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub(crate) struct VideoTaskInput {
    pub project_root_path: String,
    pub prompt: String,
    pub title: Option<String>,
    pub raw_text: Option<String>,
    pub aspect_ratio: Option<String>,
    pub resolution: Option<String>,
    pub duration: Option<u64>,
    pub image_url: Option<String>,
    pub end_image_url: Option<String>,
    pub seed: Option<i64>,
    pub generate_audio: Option<bool>,
    pub camera_fixed: Option<bool>,
    pub provider_id: Option<String>,
    pub model: Option<String>,
    pub session_id: String,
    pub thread_id: String,
    pub turn_id: String,
    pub project_id: Option<String>,
    pub content_id: Option<String>,
    pub entry_source: Option<String>,
    pub modality_contract_key: Option<String>,
    pub modality: Option<String>,
    pub routing_slot: Option<String>,
    pub runtime_contract: Option<Value>,
    pub requested_target: Option<String>,
}

pub fn check_runtime_video_task_permissions(
    params: &Value,
    working_directory: &Path,
    session_id: &str,
    turn_context: Option<&RuntimeToolTurnContext>,
) -> Result<(), RuntimeToolExecutionError> {
    parse_video_task_input(params, working_directory, session_id, turn_context)
        .map(|_| ())
        .map_err(|error| runtime_video_task_permission_error(error.message().to_string()))
}

pub(crate) fn parse_video_task_input(
    params: &Value,
    working_directory: &Path,
    session_id: &str,
    turn_context: Option<&RuntimeToolTurnContext>,
) -> Result<VideoTaskInput, RuntimeToolExecutionError> {
    let project_root_path = resolve_project_root_path(params, working_directory)?;
    let prompt = required_string(params, &["prompt"])?;
    let session_id = required_identity(
        optional_string(params, &["session_id", "sessionId"])
            .or_else(|| turn_context_identity(turn_context, SESSION_ID_KEYS))
            .or_else(|| non_empty_string(session_id)),
        "session_id",
    )?;
    let thread_id = required_identity(
        optional_string(params, &["thread_id", "threadId"])
            .or_else(|| turn_context_identity(turn_context, THREAD_ID_KEYS)),
        "thread_id",
    )?;
    let turn_id = required_identity(
        optional_string(params, &["turn_id", "turnId"])
            .or_else(|| turn_context_identity(turn_context, TURN_ID_KEYS)),
        "turn_id",
    )?;

    Ok(VideoTaskInput {
        project_root_path,
        prompt,
        title: optional_string(params, &["title"]),
        raw_text: optional_string(params, &["raw_text", "rawText"]),
        aspect_ratio: optional_string(params, &["aspect_ratio", "aspectRatio"]),
        resolution: optional_string(params, &["resolution"]),
        duration: optional_positive_u64(params, &["duration"])?,
        image_url: optional_string(params, &["image_url", "imageUrl"]),
        end_image_url: optional_string(params, &["end_image_url", "endImageUrl"]),
        seed: optional_i64(params, &["seed"])?,
        generate_audio: optional_bool(params, &["generate_audio", "generateAudio"])?,
        camera_fixed: optional_bool(params, &["camera_fixed", "cameraFixed"])?,
        provider_id: optional_string(params, &["provider_id", "providerId"]),
        model: optional_string(params, &["model"]),
        session_id,
        thread_id,
        turn_id,
        project_id: optional_string(params, &["project_id", "projectId"]),
        content_id: optional_string(params, &["content_id", "contentId"]),
        entry_source: optional_string(params, &["entry_source", "entrySource"]),
        modality_contract_key: optional_string(
            params,
            &["modality_contract_key", "modalityContractKey"],
        ),
        modality: optional_string(params, &["modality"]),
        routing_slot: optional_string(params, &["routing_slot", "routingSlot"]),
        runtime_contract: params
            .get("runtime_contract")
            .cloned()
            .or_else(|| params.get("runtimeContract").cloned()),
        requested_target: optional_string(params, &["requested_target", "requestedTarget"]),
    })
}

pub(crate) fn build_create_params(input: VideoTaskInput) -> MediaTaskArtifactVideoCreateParams {
    MediaTaskArtifactVideoCreateParams {
        project_root_path: input.project_root_path,
        prompt: input.prompt,
        title: input.title,
        raw_text: input.raw_text,
        aspect_ratio: input.aspect_ratio,
        resolution: input.resolution,
        duration: input.duration,
        image_url: input.image_url,
        end_image_url: input.end_image_url,
        seed: input.seed,
        generate_audio: input.generate_audio,
        camera_fixed: input.camera_fixed,
        provider_id: input.provider_id,
        model: input.model,
        session_id: Some(input.session_id),
        thread_id: Some(input.thread_id),
        turn_id: Some(input.turn_id),
        project_id: input.project_id,
        content_id: input.content_id,
        entry_source: input.entry_source,
        modality_contract_key: input.modality_contract_key,
        modality: input.modality,
        required_capabilities: Vec::new(),
        routing_slot: input.routing_slot,
        runtime_contract: input.runtime_contract,
        requested_target: input.requested_target,
        output_path: None,
    }
}

pub(crate) fn runtime_video_task_error(message: impl Into<String>) -> RuntimeToolExecutionError {
    let message = message.into();
    RuntimeToolExecutionError::new(
        message.clone(),
        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(message)),
    )
}

const SESSION_ID_KEYS: &[&str] = &["session_id", "sessionId"];
const THREAD_ID_KEYS: &[&str] = &["thread_id", "threadId"];
const TURN_ID_KEYS: &[&str] = &["turn_id", "turnId"];
const SCOPE_POINTERS: &[&str] = &["/action_scope", "/actionScope", "/scope"];

fn resolve_project_root_path(
    params: &Value,
    working_directory: &Path,
) -> Result<String, RuntimeToolExecutionError> {
    if let Some(path) = optional_string(params, &["project_root_path", "projectRootPath"]) {
        return validate_absolute_path(&path);
    }
    if !working_directory.is_absolute() {
        return Err(runtime_video_task_error(
            "project_root_path requires an absolute working directory",
        ));
    }
    validate_absolute_path(&working_directory.to_string_lossy())
}

fn validate_absolute_path(path: &str) -> Result<String, RuntimeToolExecutionError> {
    let candidate = PathBuf::from(path);
    if !candidate.is_absolute() {
        return Err(runtime_video_task_error("project_root_path 必须是绝对路径"));
    }
    Ok(candidate.to_string_lossy().to_string())
}

fn required_identity(
    value: Option<String>,
    field: &str,
) -> Result<String, RuntimeToolExecutionError> {
    value.ok_or_else(|| runtime_video_task_error(format!("{field} 不能为空")))
}

fn required_string(params: &Value, keys: &[&str]) -> Result<String, RuntimeToolExecutionError> {
    optional_string(params, keys)
        .ok_or_else(|| runtime_video_task_error(format!("Missing required parameter: {}", keys[0])))
}

fn optional_string(params: &Value, keys: &[&str]) -> Option<String> {
    first_value(params, keys)
        .and_then(Value::as_str)
        .and_then(non_empty_string)
}

fn optional_positive_u64(
    params: &Value,
    keys: &[&str],
) -> Result<Option<u64>, RuntimeToolExecutionError> {
    let Some(value) = first_value(params, keys) else {
        return Ok(None);
    };
    let Some(value) = value.as_u64().filter(|value| *value > 0) else {
        return Err(runtime_video_task_error(format!(
            "{} must be a positive integer",
            keys[0]
        )));
    };
    Ok(Some(value))
}

fn optional_i64(params: &Value, keys: &[&str]) -> Result<Option<i64>, RuntimeToolExecutionError> {
    let Some(value) = first_value(params, keys) else {
        return Ok(None);
    };
    value
        .as_i64()
        .map(Some)
        .ok_or_else(|| runtime_video_task_error(format!("{} must be an integer", keys[0])))
}

fn optional_bool(params: &Value, keys: &[&str]) -> Result<Option<bool>, RuntimeToolExecutionError> {
    let Some(value) = first_value(params, keys) else {
        return Ok(None);
    };
    value
        .as_bool()
        .map(Some)
        .ok_or_else(|| runtime_video_task_error(format!("{} must be a boolean", keys[0])))
}

fn first_value<'a>(params: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| params.get(*key))
}

fn turn_context_identity(
    turn_context: Option<&RuntimeToolTurnContext>,
    keys: &[&str],
) -> Option<String> {
    let metadata = &turn_context?.metadata;
    metadata_string(metadata, keys).or_else(|| metadata_scope_string(metadata, keys))
}

fn metadata_string(metadata: &HashMap<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| metadata.get(*key))
        .and_then(Value::as_str)
        .and_then(non_empty_string)
}

fn metadata_scope_string(metadata: &HashMap<String, Value>, keys: &[&str]) -> Option<String> {
    let value = Value::Object(
        metadata
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<JsonMap<String, Value>>(),
    );
    SCOPE_POINTERS
        .iter()
        .find_map(|pointer| value.pointer(pointer))
        .and_then(|scope| keys.iter().find_map(|key| scope.get(*key)))
        .and_then(Value::as_str)
        .and_then(non_empty_string)
}

fn non_empty_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn runtime_video_task_permission_error(message: impl Into<String>) -> RuntimeToolExecutionError {
    let message = message.into();
    RuntimeToolExecutionError::new(
        message.clone(),
        Some(RuntimeToolPolicyErrorKind::PermissionDenied(message)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn parses_video_params_and_identity_from_turn_context() {
        let workspace = TempDir::new().expect("workspace");
        let turn_context = RuntimeToolTurnContext {
            metadata: HashMap::from([
                ("thread_id".to_string(), json!("thread-video-1")),
                ("turn_id".to_string(), json!("turn-video-1")),
            ]),
            ..RuntimeToolTurnContext::default()
        };

        let input = parse_video_task_input(
            &json!({
                "prompt": "生成一段产品演示视频",
                "duration": 8,
                "generate_audio": true
            }),
            workspace.path(),
            "session-video-1",
            Some(&turn_context),
        )
        .expect("video task input");

        assert_eq!(input.project_root_path, workspace.path().to_string_lossy());
        assert_eq!(input.session_id, "session-video-1");
        assert_eq!(input.thread_id, "thread-video-1");
        assert_eq!(input.turn_id, "turn-video-1");
        assert_eq!(input.duration, Some(8));
        assert_eq!(input.generate_audio, Some(true));
    }

    #[test]
    fn rejects_missing_turn_identity() {
        let workspace = TempDir::new().expect("workspace");
        let error = parse_video_task_input(
            &json!({ "prompt": "生成视频" }),
            workspace.path(),
            "session-video-1",
            None,
        )
        .expect_err("thread identity must be required");

        assert!(error.message().contains("thread_id"));
    }

    #[test]
    fn rejects_invalid_duration_instead_of_dropping_it() {
        let workspace = TempDir::new().expect("workspace");
        let error = parse_video_task_input(
            &json!({
                "prompt": "生成视频",
                "duration": 0,
                "thread_id": "thread-video-1",
                "turn_id": "turn-video-1"
            }),
            workspace.path(),
            "session-video-1",
            None,
        )
        .expect_err("zero duration must fail closed");

        assert!(error.message().contains("duration"));
    }
}
