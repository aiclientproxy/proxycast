use crate::media_runtime_contract::{
    runtime_contract_projection_from_payload, VOICE_GENERATION_CONTRACT_KEY,
    VOICE_GENERATION_ROUTING_SLOT,
};
use crate::media_task_payload::{
    create_audio_payload, create_image_payload, create_transcription_payload, create_video_payload,
    AUDIO_TASK_DEFAULT_MIME_TYPE,
};
use crate::model_task_contract::MediaRouteAssessment;
use crate::runtime::sidecar_store::SidecarStore;
use app_server_protocol::MediaTaskArtifactAudioCompleteParams;
use app_server_protocol::MediaTaskArtifactAudioCreateParams;
use app_server_protocol::MediaTaskArtifactCompletedImageInput;
use app_server_protocol::MediaTaskArtifactImageCompleteParams;
use app_server_protocol::MediaTaskArtifactImageCreateParams;
use app_server_protocol::MediaTaskArtifactListFilters;
use app_server_protocol::MediaTaskArtifactListParams;
use app_server_protocol::MediaTaskArtifactListResponse;
use app_server_protocol::MediaTaskArtifactLookupParams;
use app_server_protocol::MediaTaskArtifactResponse;
use app_server_protocol::MediaTaskArtifactTranscriptionCreateParams;
use app_server_protocol::MediaTaskArtifactVideoCreateParams;
use lime_media_runtime::list_task_outputs;
use lime_media_runtime::load_task_output;
use lime_media_runtime::patch_task_artifact;
use lime_media_runtime::retry_task_artifact;
use lime_media_runtime::update_task_status;
use lime_media_runtime::write_task_artifact;
use lime_media_runtime::MediaTaskOutput;
use lime_media_runtime::MediaTaskType;
use lime_media_runtime::TaskArtifactPatch;
use lime_media_runtime::TaskRelationships;
use lime_media_runtime::TaskWriteOptions;
use lime_media_runtime::DEFAULT_ARTIFACT_ROOT;
use lime_media_runtime::{TaskErrorRecord, TaskProgress};
use serde_json::json;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

mod idempotency;
mod sidecar;

const AUDIO_TASK_COMPLETION_WORKER_ID: &str = "app-server-audio-output-writer";
const IMAGE_TASK_COMPLETION_WORKER_ID: &str = "app-server-image-output-writer";

fn data_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn normalize_required_string(value: &str, field_name: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(format!("{field_name} 不能为空"))
    } else {
        Ok(trimmed.to_string())
    }
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value.and_then(|raw| {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn maybe_json_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(Value::as_str)
        .and_then(|value| normalize_optional_string(Some(value.to_string())))
}

fn maybe_json_u64(value: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(Value::as_u64)
}

fn maybe_json_bool(value: &Value, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(Value::as_bool)
}

fn response_from_output(output: MediaTaskOutput) -> Result<MediaTaskArtifactResponse, String> {
    let value = serde_json::to_value(output).map_err(data_error)?;
    serde_json::from_value(value).map_err(data_error)
}

fn response_vec_from_outputs(
    outputs: Vec<MediaTaskOutput>,
) -> Result<Vec<MediaTaskArtifactResponse>, String> {
    outputs.into_iter().map(response_from_output).collect()
}

fn build_audio_output_summary(
    payload: &Value,
    audio_path: &str,
    mime_type: &str,
    duration_ms: Option<u64>,
    provider_id: Option<&str>,
    model: Option<&str>,
) -> Value {
    json!({
        "kind": "audio_output",
        "status": "completed",
        "audio_path": audio_path,
        "mime_type": mime_type,
        "duration_ms": duration_ms,
        "source_text": maybe_json_string(payload, &["source_text", "sourceText", "prompt"]),
        "voice": maybe_json_string(payload, &["voice"]),
        "voice_style": maybe_json_string(payload, &["voice_style", "voiceStyle"]),
        "target_language": maybe_json_string(payload, &["target_language", "targetLanguage"]),
        "provider_id": provider_id,
        "model": model,
        "modality_contract_key": VOICE_GENERATION_CONTRACT_KEY,
        "modality": "audio",
        "routing_slot": VOICE_GENERATION_ROUTING_SLOT,
    })
}

fn build_audio_generation_result_value(audio_output: &Value) -> Value {
    json!({
        "kind": "audio_generation_result",
        "status": "completed",
        "audio_output": audio_output,
        "outputs": [audio_output],
        "audio_path": audio_output.get("audio_path").cloned().unwrap_or(Value::Null),
        "mime_type": audio_output.get("mime_type").cloned().unwrap_or(Value::Null),
        "duration_ms": audio_output.get("duration_ms").cloned().unwrap_or(Value::Null),
    })
}

fn normalize_completed_image(
    image: MediaTaskArtifactCompletedImageInput,
    fallback_prompt: Option<&str>,
    fallback_provider_id: Option<&str>,
    fallback_model: Option<&str>,
    fallback_size: Option<&str>,
    index: usize,
    sidecar_context: Option<&sidecar::MediaSidecarContext<'_>>,
) -> Result<Value, String> {
    let url = normalize_required_string(&image.url, "images[].url")?;
    let prompt = normalize_optional_string(image.prompt)
        .or_else(|| normalize_optional_string(image.slot_prompt.clone()))
        .or_else(|| fallback_prompt.map(str::to_string));
    let slot_prompt = normalize_optional_string(image.slot_prompt).or_else(|| prompt.clone());
    let provider_id = normalize_optional_string(image.provider_id)
        .or_else(|| fallback_provider_id.map(str::to_string));
    let model =
        normalize_optional_string(image.model).or_else(|| fallback_model.map(str::to_string));
    let size = normalize_optional_string(image.size).or_else(|| fallback_size.map(str::to_string));
    let slot_index = image.slot_index.unwrap_or((index + 1) as u32);

    let mut output = json!({
        "url": url,
        "prompt": prompt,
        "revised_prompt": normalize_optional_string(image.revised_prompt),
        "size": size,
        "provider_id": provider_id,
        "model": model,
        "slot_index": slot_index,
        "slot_id": normalize_optional_string(image.slot_id),
        "slot_prompt": slot_prompt,
    });
    if let Some(context) = sidecar_context {
        sidecar::attach_image_sidecar_ref(&mut output, context, &url, index)?;
    }
    Ok(output)
}

fn image_failure_error_record(failure: Option<&Value>) -> TaskErrorRecord {
    let code = failure
        .and_then(|value| maybe_json_string(value, &["code", "error_code", "errorCode"]))
        .unwrap_or_else(|| "image_generation_failed".to_string());
    let message = failure
        .and_then(|value| {
            value
                .as_str()
                .and_then(|raw| normalize_optional_string(Some(raw.to_string())))
                .or_else(|| maybe_json_string(value, &["message", "error", "reason", "detail"]))
        })
        .unwrap_or_else(|| "图片任务生成失败。".to_string());

    TaskErrorRecord {
        code,
        message,
        retryable: failure
            .and_then(|value| maybe_json_bool(value, &["retryable"]))
            .unwrap_or(false),
        stage: failure.and_then(|value| maybe_json_string(value, &["stage", "phase"])),
        provider_code: failure
            .and_then(|value| maybe_json_string(value, &["provider_code", "providerCode"])),
        occurred_at: failure
            .and_then(|value| maybe_json_string(value, &["occurred_at", "occurredAt"])),
    }
}

fn build_image_task_result_value(
    payload: &Value,
    params: MediaTaskArtifactImageCompleteParams,
    sidecar_store: Option<&SidecarStore>,
    workspace_root: &str,
    task_ref: &str,
) -> Result<(Value, Value, String, usize, usize, Option<TaskErrorRecord>), String> {
    let requested_status = normalize_optional_string(params.status.clone());
    let requested_failed = requested_status.as_deref() == Some("failed");
    if params.images.is_empty() && !requested_failed {
        return Err("images 不能为空".to_string());
    }
    if requested_failed && params.failures.is_empty() {
        return Err("failures 不能为空".to_string());
    }

    let payload_prompt = maybe_json_string(payload, &["prompt"]);
    let payload_size = maybe_json_string(payload, &["size"]);
    let provider_id = normalize_optional_string(params.provider_id)
        .or_else(|| maybe_json_string(payload, &["provider_id", "providerId"]));
    let model =
        normalize_optional_string(params.model).or_else(|| maybe_json_string(payload, &["model"]));
    let executor_mode = normalize_optional_string(params.executor_mode)
        .or_else(|| maybe_json_string(payload, &["executor_mode", "executorMode"]))
        .unwrap_or_else(|| "images_api".to_string());
    let sidecar_context =
        sidecar::image_sidecar_context(sidecar_store, Path::new(workspace_root), task_ref, payload);

    let images: Vec<Value> = params
        .images
        .into_iter()
        .enumerate()
        .map(|(index, image)| {
            normalize_completed_image(
                image,
                payload_prompt.as_deref(),
                provider_id.as_deref(),
                model.as_deref(),
                payload_size.as_deref(),
                index,
                sidecar_context.as_ref(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let image_count = images.len();
    let failure_count = params.failures.len();
    let requested_count = maybe_json_u64(payload, &["count", "requested_count", "requestedCount"])
        .unwrap_or(image_count as u64 + failure_count as u64)
        .max(image_count as u64 + failure_count as u64);
    let requested_status = requested_status.unwrap_or_else(|| {
        if failure_count > 0 && image_count == 0 {
            "failed".to_string()
        } else if failure_count > 0 && image_count < requested_count as usize {
            "partial".to_string()
        } else {
            "succeeded".to_string()
        }
    });
    let normalized_status = match requested_status.as_str() {
        "failed" => "failed".to_string(),
        "partial" => "partial".to_string(),
        _ => "succeeded".to_string(),
    };
    let last_error = if normalized_status == "failed" {
        Some(image_failure_error_record(params.failures.first()))
    } else {
        None
    };
    let response_status = normalized_status.clone();
    let response_id = normalize_optional_string(params.response_id)
        .unwrap_or_else(|| format!("app-server-image-complete-{image_count}"));
    let response = json!({
        "id": response_id,
        "status": response_status,
        "model": model,
        "provider_id": provider_id,
    });
    let responses = if params.responses.is_empty() {
        vec![response.clone()]
    } else {
        params.responses
    };
    let storyboard_slots = images
        .iter()
        .map(|image| {
            json!({
                "slot_index": image.get("slot_index").cloned().unwrap_or(Value::Null),
                "slot_id": image.get("slot_id").cloned().unwrap_or(Value::Null),
                "label": Value::Null,
                "prompt": image
                    .get("slot_prompt")
                    .cloned()
                    .or_else(|| image.get("prompt").cloned())
                    .unwrap_or(Value::Null),
                "shot_type": Value::Null,
            })
        })
        .collect::<Vec<_>>();
    let result = json!({
        "kind": "image_generation_result",
        "status": normalized_status,
        "prompt": payload_prompt,
        "provider_id": provider_id,
        "executor_mode": executor_mode,
        "outer_model": maybe_json_string(payload, &["outer_model", "outerModel"]),
        "model": model,
        "size": payload_size,
        "count": image_count,
        "layout_hint": maybe_json_string(payload, &["layout_hint", "layoutHint"]),
        "requested_count": requested_count,
        "received_count": image_count,
        "images": images,
        "response": response,
        "responses": responses,
        "failures": params.failures,
        "postprocess": Value::Null,
        "storyboard_slots": storyboard_slots,
    });
    let payload_patch = json!({
        "provider_id": result.get("provider_id").cloned().unwrap_or(Value::Null),
        "model": result.get("model").cloned().unwrap_or(Value::Null),
        "executor_mode": result.get("executor_mode").cloned().unwrap_or(Value::Null),
        "received_count": image_count,
        "failure_count": failure_count,
        "image_output": {
            "kind": "image_output",
            "status": normalized_status,
            "image_count": image_count,
            "failure_count": failure_count,
            "provider_id": result.get("provider_id").cloned().unwrap_or(Value::Null),
            "model": result.get("model").cloned().unwrap_or(Value::Null),
            "modality_contract_key": "image_generation",
            "modality": "image",
            "routing_slot": "image_generation_model",
        },
    });
    Ok((
        result,
        payload_patch,
        normalized_status,
        image_count,
        failure_count,
        last_error,
    ))
}

fn task_payload(output: &MediaTaskOutput) -> &Value {
    &output.record.payload
}

fn media_task_contract_key(output: &MediaTaskOutput) -> Option<String> {
    runtime_contract_projection_from_payload(task_payload(output)).contract_key
}

fn media_task_routing_outcome(output: &MediaTaskOutput) -> String {
    if output.normalized_status == "failed" {
        "failed".to_string()
    } else if maybe_json_string(task_payload(output), &["failure_code", "failureCode"]).is_some() {
        "blocked".to_string()
    } else {
        "accepted".to_string()
    }
}

fn media_task_matches_modality_filters(
    output: &MediaTaskOutput,
    contract_key: Option<&str>,
    routing_outcome: Option<&str>,
) -> bool {
    if let Some(contract_key) = contract_key {
        if media_task_contract_key(output).as_deref() != Some(contract_key) {
            return false;
        }
    }
    if let Some(routing_outcome) = routing_outcome {
        if media_task_routing_outcome(output) != routing_outcome {
            return false;
        }
    }
    true
}

fn add_unique(values: &mut Vec<String>, value: Option<String>) {
    if let Some(value) = value {
        if !values.iter().any(|existing| existing == &value) {
            values.push(value);
        }
    }
}

fn count_by(values: impl IntoIterator<Item = String>) -> Vec<Value> {
    let mut counts = BTreeMap::<String, usize>::new();
    for value in values {
        *counts.entry(value).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(status, count)| json!({ "status": status, "count": count }))
        .collect()
}

fn count_routing_outcomes(outputs: &[MediaTaskOutput]) -> Vec<Value> {
    let mut counts = BTreeMap::<String, usize>::new();
    for output in outputs {
        *counts
            .entry(media_task_routing_outcome(output))
            .or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(outcome, count)| json!({ "outcome": outcome, "count": count }))
        .collect()
}

fn build_modality_runtime_contract_index(outputs: &[MediaTaskOutput]) -> Value {
    let mut contract_keys = Vec::new();
    let mut entry_keys = Vec::new();
    let mut thread_ids = Vec::new();
    let mut turn_ids = Vec::new();
    let mut content_ids = Vec::new();
    let mut modalities = Vec::new();
    let mut skill_ids = Vec::new();
    let mut model_ids = Vec::new();
    let mut execution_profile_keys = Vec::new();
    let mut executor_adapter_keys = Vec::new();
    let mut executor_kinds = Vec::new();
    let mut executor_binding_keys = Vec::new();
    let mut limecore_policy_refs = Vec::new();
    let mut snapshots = Vec::new();
    let mut audio_statuses = Vec::new();

    for output in outputs {
        let payload = task_payload(output);
        let runtime_contract = runtime_contract_projection_from_payload(payload);
        let contract_key = media_task_contract_key(output);
        let modality = maybe_json_string(payload, &["modality"]);
        let entry_key = maybe_json_string(
            payload,
            &["entry_key", "entryKey", "entry_source", "entrySource"],
        );
        let thread_id = maybe_json_string(payload, &["thread_id", "threadId"]);
        let turn_id = maybe_json_string(payload, &["turn_id", "turnId"]);
        let content_id = maybe_json_string(payload, &["content_id", "contentId"]);
        let model = maybe_json_string(payload, &["model"]);
        let routing_slot = maybe_json_string(payload, &["routing_slot", "routingSlot"]);
        let execution_profile_key = runtime_contract.execution_profile_key.clone();
        let executor_adapter_key = runtime_contract.executor_adapter_key.clone();
        let executor_kind = runtime_contract.executor_kind.clone();
        let executor_binding_key = runtime_contract.executor_binding_key.clone();
        let policy_refs = runtime_contract.policy_refs.clone();
        let audio_output = payload
            .get("audio_output")
            .or_else(|| payload.get("audioOutput"));
        let audio_output_status =
            audio_output.and_then(|value| maybe_json_string(value, &["status"]));

        add_unique(&mut contract_keys, contract_key.clone());
        add_unique(&mut entry_keys, entry_key.clone());
        add_unique(&mut thread_ids, thread_id.clone());
        add_unique(&mut turn_ids, turn_id.clone());
        add_unique(&mut content_ids, content_id.clone());
        add_unique(&mut modalities, modality.clone());
        add_unique(&mut skill_ids, executor_binding_key.clone());
        add_unique(&mut model_ids, model.clone());
        add_unique(&mut execution_profile_keys, execution_profile_key.clone());
        add_unique(&mut executor_adapter_keys, executor_adapter_key.clone());
        add_unique(&mut executor_kinds, executor_kind.clone());
        add_unique(&mut executor_binding_keys, executor_binding_key.clone());
        for policy_ref in policy_refs.iter().cloned() {
            add_unique(&mut limecore_policy_refs, Some(policy_ref));
        }
        if let Some(status) = audio_output_status.clone() {
            audio_statuses.push(status);
        }

        snapshots.push(json!({
            "task_id": output.task_id,
            "task_type": output.task_type,
            "normalized_status": output.normalized_status,
            "contract_key": contract_key,
            "entry_key": entry_key,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "content_id": content_id,
            "modality": modality,
            "skill_id": executor_binding_key,
            "model_id": model,
            "routing_slot": routing_slot,
            "provider_id": maybe_json_string(payload, &["provider_id", "providerId"]),
            "model": maybe_json_string(payload, &["model"]),
            "execution_profile_key": execution_profile_key,
            "executor_adapter_key": executor_adapter_key,
            "executor_kind": executor_kind,
            "executor_binding_key": executor_binding_key,
            "limecore_policy_refs": policy_refs,
            "routing_event": "model_routing_decision",
            "routing_outcome": media_task_routing_outcome(output),
            "failure_code": maybe_json_string(payload, &["failure_code", "failureCode"]),
            "audio_output_status": audio_output_status,
            "audio_output_path": audio_output.and_then(|value| maybe_json_string(value, &["audio_path", "audioPath"])),
            "audio_output_mime_type": audio_output.and_then(|value| maybe_json_string(value, &["mime_type", "mimeType"])),
            "audio_output_duration_ms": audio_output.and_then(|value| maybe_json_u64(value, &["duration_ms", "durationMs"])),
        }));
    }

    let mut index = serde_json::Map::new();
    index.insert("snapshot_count".to_string(), json!(outputs.len()));
    index.insert("contract_keys".to_string(), json!(contract_keys));
    index.insert("entry_keys".to_string(), json!(entry_keys));
    index.insert("thread_ids".to_string(), json!(thread_ids));
    index.insert("turn_ids".to_string(), json!(turn_ids));
    index.insert("content_ids".to_string(), json!(content_ids));
    index.insert("modalities".to_string(), json!(modalities));
    index.insert("skill_ids".to_string(), json!(skill_ids));
    index.insert("model_ids".to_string(), json!(model_ids));
    index.insert("cost_states".to_string(), json!([]));
    index.insert("limit_states".to_string(), json!([]));
    index.insert("estimated_cost_classes".to_string(), json!([]));
    index.insert("limit_event_kinds".to_string(), json!([]));
    index.insert("quota_low_count".to_string(), json!(0));
    index.insert(
        "execution_profile_keys".to_string(),
        json!(execution_profile_keys),
    );
    index.insert(
        "executor_adapter_keys".to_string(),
        json!(executor_adapter_keys),
    );
    index.insert("executor_kinds".to_string(), json!(executor_kinds));
    index.insert(
        "executor_binding_keys".to_string(),
        json!(executor_binding_keys),
    );
    index.insert(
        "limecore_policy_refs".to_string(),
        json!(limecore_policy_refs),
    );
    index.insert("limecore_policy_snapshot_count".to_string(), json!(0));
    index.insert("limecore_policy_snapshot_statuses".to_string(), json!([]));
    index.insert("limecore_policy_decisions".to_string(), json!([]));
    index.insert("limecore_policy_decision_sources".to_string(), json!([]));
    index.insert("limecore_policy_evaluation_statuses".to_string(), json!([]));
    index.insert(
        "limecore_policy_evaluation_decisions".to_string(),
        json!([]),
    );
    index.insert(
        "limecore_policy_evaluation_decision_sources".to_string(),
        json!([]),
    );
    index.insert(
        "limecore_policy_evaluation_blocking_refs".to_string(),
        json!([]),
    );
    index.insert("limecore_policy_evaluation_ask_refs".to_string(), json!([]));
    index.insert(
        "limecore_policy_evaluation_pending_refs".to_string(),
        json!([]),
    );
    index.insert("limecore_policy_unresolved_refs".to_string(), json!([]));
    index.insert("limecore_policy_missing_inputs".to_string(), json!([]));
    index.insert("limecore_policy_pending_hit_refs".to_string(), json!([]));
    index.insert("limecore_policy_value_hit_count".to_string(), json!(0));
    index.insert(
        "blocked_count".to_string(),
        json!(outputs
            .iter()
            .filter(|output| media_task_routing_outcome(output) == "blocked")
            .count()),
    );
    index.insert(
        "routing_outcomes".to_string(),
        Value::Array(count_routing_outcomes(outputs)),
    );
    index.insert("model_registry_assessment_count".to_string(), json!(0));
    index.insert(
        "audio_output_count".to_string(),
        json!(audio_statuses.len()),
    );
    index.insert(
        "audio_output_statuses".to_string(),
        Value::Array(count_by(audio_statuses)),
    );
    index.insert("audio_output_error_codes".to_string(), json!([]));
    index.insert("transcript_count".to_string(), json!(0));
    index.insert("transcript_statuses".to_string(), json!([]));
    index.insert("transcript_error_codes".to_string(), json!([]));
    index.insert("snapshots".to_string(), json!(snapshots));
    Value::Object(index)
}

pub fn create_image_generation_task_artifact(
    params: MediaTaskArtifactImageCreateParams,
    route_assessment: Option<MediaRouteAssessment>,
) -> Result<MediaTaskArtifactResponse, String> {
    let workspace_root = normalize_required_string(&params.project_root_path, "projectRootPath")?;
    let prompt = normalize_required_string(&params.prompt, "prompt")?;
    let mut params = params;
    params.project_root_path = workspace_root.clone();
    params.prompt = prompt;
    let output = write_task_artifact(
        Path::new(&workspace_root),
        MediaTaskType::ImageGenerate,
        normalize_optional_string(params.title.clone()),
        create_image_payload(&params, route_assessment.as_ref()),
        TaskWriteOptions {
            status: Some("pending_submit".to_string()),
            output_path: None,
            artifact_dir: None,
            idempotency_key: Some(idempotency::build_image_idempotency_key(&params).as_str()),
            relationships: TaskRelationships {
                slot_id: normalize_optional_string(params.slot_id.clone()),
                ..TaskRelationships::default()
            },
        },
    )
    .map_err(data_error)?;
    response_from_output(output)
}

pub fn create_audio_generation_task_artifact(
    params: MediaTaskArtifactAudioCreateParams,
    route_assessment: Option<MediaRouteAssessment>,
) -> Result<MediaTaskArtifactResponse, String> {
    let workspace_root = normalize_required_string(&params.project_root_path, "projectRootPath")?;
    let source_text = normalize_required_string(&params.source_text, "sourceText")?;
    let mut params = params;
    params.project_root_path = workspace_root.clone();
    params.source_text = source_text;
    let output_path = normalize_optional_string(params.output_path.clone());
    let output = write_task_artifact(
        Path::new(&workspace_root),
        MediaTaskType::AudioGenerate,
        normalize_optional_string(params.title.clone()),
        create_audio_payload(&params, route_assessment.as_ref()),
        TaskWriteOptions {
            status: Some("pending_submit".to_string()),
            output_path: output_path.as_deref(),
            artifact_dir: None,
            idempotency_key: Some(idempotency::build_audio_idempotency_key(&params).as_str()),
            relationships: TaskRelationships::default(),
        },
    )
    .map_err(data_error)?;
    response_from_output(output)
}

pub fn create_video_generation_task_artifact(
    params: MediaTaskArtifactVideoCreateParams,
    route_assessment: Option<MediaRouteAssessment>,
) -> Result<MediaTaskArtifactResponse, String> {
    let workspace_root = normalize_required_string(&params.project_root_path, "projectRootPath")?;
    let prompt = normalize_required_string(&params.prompt, "prompt")?;
    let mut params = params;
    params.project_root_path = workspace_root.clone();
    params.prompt = prompt;
    let output_path = normalize_optional_string(params.output_path.clone());
    let output = write_task_artifact(
        Path::new(&workspace_root),
        MediaTaskType::VideoGenerate,
        normalize_optional_string(params.title.clone()),
        create_video_payload(&params, route_assessment.as_ref()),
        TaskWriteOptions {
            status: Some("pending_submit".to_string()),
            output_path: output_path.as_deref(),
            artifact_dir: None,
            idempotency_key: Some(idempotency::build_video_idempotency_key(&params).as_str()),
            relationships: TaskRelationships::default(),
        },
    )
    .map_err(data_error)?;
    response_from_output(output)
}

pub fn create_transcription_task_artifact(
    params: MediaTaskArtifactTranscriptionCreateParams,
    route_assessment: Option<MediaRouteAssessment>,
) -> Result<MediaTaskArtifactResponse, String> {
    let workspace_root = normalize_required_string(&params.project_root_path, "projectRootPath")?;
    if normalize_optional_string(params.source_path.clone()).is_none()
        && normalize_optional_string(params.source_url.clone()).is_none()
    {
        return Err("sourcePath 或 sourceUrl 至少需要提供一个".to_string());
    }
    let mut params = params;
    params.project_root_path = workspace_root.clone();
    params.source_path = normalize_optional_string(params.source_path);
    params.source_url = normalize_optional_string(params.source_url);
    let output_path = normalize_optional_string(params.output_path.clone());
    let output = write_task_artifact(
        Path::new(&workspace_root),
        MediaTaskType::TranscriptionGenerate,
        normalize_optional_string(params.title.clone()),
        create_transcription_payload(&params, route_assessment.as_ref()),
        TaskWriteOptions {
            status: Some("pending_submit".to_string()),
            output_path: output_path.as_deref(),
            artifact_dir: None,
            idempotency_key: Some(
                idempotency::build_transcription_idempotency_key(&params).as_str(),
            ),
            relationships: TaskRelationships::default(),
        },
    )
    .map_err(data_error)?;
    response_from_output(output)
}

pub fn complete_audio_generation_task_artifact(
    params: MediaTaskArtifactAudioCompleteParams,
    sidecar_store: Option<&SidecarStore>,
) -> Result<MediaTaskArtifactResponse, String> {
    let workspace_root = normalize_required_string(&params.project_root_path, "projectRootPath")?;
    let task_ref = normalize_required_string(&params.task_ref, "taskRef")?;
    let audio_path = normalize_required_string(&params.audio_path, "audioPath")?;
    let workspace_root_path = Path::new(&workspace_root);
    let current = load_task_output(workspace_root_path, &task_ref, None).map_err(data_error)?;

    if current.task_type != MediaTaskType::AudioGenerate.as_str() {
        return Err(format!(
            "只能完成 audio_generate 任务，当前任务类型为 {}",
            current.task_type
        ));
    }
    if matches!(current.normalized_status.as_str(), "cancelled" | "failed") {
        return Err(format!(
            "当前音频任务状态为 {}，不能直接写回完成态",
            current.normalized_status
        ));
    }

    let payload = task_payload(&current);
    let mime_type = normalize_optional_string(params.mime_type)
        .or_else(|| maybe_json_string(payload, &["mime_type", "mimeType"]))
        .unwrap_or_else(|| AUDIO_TASK_DEFAULT_MIME_TYPE.to_string());
    let provider_id = normalize_optional_string(params.provider_id)
        .or_else(|| maybe_json_string(payload, &["provider_id", "providerId"]));
    let model =
        normalize_optional_string(params.model).or_else(|| maybe_json_string(payload, &["model"]));
    let duration_ms = params.duration_ms.or_else(|| {
        maybe_json_u64(payload, &["duration_ms", "durationMs"]).or_else(|| {
            payload
                .get("audio_output")
                .or_else(|| payload.get("audioOutput"))
                .and_then(|audio| maybe_json_u64(audio, &["duration_ms", "durationMs"]))
        })
    });
    let audio_output = build_audio_output_summary(
        payload,
        &audio_path,
        &mime_type,
        duration_ms,
        provider_id.as_deref(),
        model.as_deref(),
    );
    let audio_output = sidecar::attach_audio_sidecar_ref(
        audio_output,
        sidecar_store,
        &workspace_root,
        &task_ref,
        payload,
    )?;
    let result = build_audio_generation_result_value(&audio_output);
    let output = patch_task_artifact(
        workspace_root_path,
        &task_ref,
        None,
        TaskArtifactPatch {
            status: Some("succeeded".to_string()),
            payload_patch: Some(json!({
                "audio_path": audio_path,
                "mime_type": mime_type,
                "duration_ms": duration_ms,
                "provider_id": provider_id,
                "model": model,
                "audio_output": audio_output,
            })),
            result: Some(Some(result)),
            last_error: Some(None),
            current_attempt_worker_id: Some(Some(AUDIO_TASK_COMPLETION_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(data_error)?;
    response_from_output(output)
}

pub async fn complete_image_generation_task_artifact(
    params: MediaTaskArtifactImageCompleteParams,
    sidecar_store: Option<&SidecarStore>,
) -> Result<MediaTaskArtifactResponse, String> {
    let workspace_root = normalize_required_string(&params.project_root_path, "projectRootPath")?;
    let task_ref = normalize_required_string(&params.task_ref, "taskRef")?;
    let workspace_root_path = Path::new(&workspace_root);
    let current = load_task_output(workspace_root_path, &task_ref, None).map_err(data_error)?;

    if current.task_type != MediaTaskType::ImageGenerate.as_str() {
        return Err(format!(
            "只能完成 image_generate 任务，当前任务类型为 {}",
            current.task_type
        ));
    }
    if matches!(current.normalized_status.as_str(), "cancelled" | "failed") {
        return Err(format!(
            "当前图片任务状态为 {}，不能直接写回完成态",
            current.normalized_status
        ));
    }

    let payload = task_payload(&current);
    let (mut result, payload_patch, status, image_count, failure_count, last_error) =
        build_image_task_result_value(payload, params, sidecar_store, &workspace_root, &task_ref)?;
    if let Some(sidecar_context) =
        sidecar::image_sidecar_context(sidecar_store, workspace_root_path, &task_ref, payload)
    {
        sidecar::attach_missing_image_sidecar_refs(&mut result, &sidecar_context).await?;
    }
    let progress_message = if status == "failed" {
        last_error
            .as_ref()
            .map(|value| value.message.clone())
            .unwrap_or_else(|| "图片任务生成失败。".to_string())
    } else if status == "partial" {
        format!("图片任务已返回 {image_count} 张，另有 {failure_count} 张失败。")
    } else {
        format!("图片任务已完成，共生成 {image_count} 张。")
    };
    let preview_slots = result
        .get("storyboard_slots")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(index, slot)| {
            let slot_index =
                maybe_json_u64(&slot, &["slot_index", "slotIndex"]).unwrap_or((index + 1) as u64);
            let slot_id = maybe_json_string(&slot, &["slot_id", "slotId"])
                .unwrap_or_else(|| format!("image-slot-{slot_index}"));
            json!({
                "slot_id": slot_id,
                "slot_index": slot_index,
                "label": slot.get("label").cloned().unwrap_or(Value::Null),
                "prompt": slot.get("prompt").cloned().unwrap_or(Value::Null),
                "shot_type": slot.get("shot_type").cloned().unwrap_or(Value::Null),
                "status": "complete",
            })
        })
        .collect::<Vec<_>>();
    let progress: TaskProgress = serde_json::from_value(json!({
        "phase": status,
        "percent": 100,
        "message": progress_message,
        "preview_slots": preview_slots,
    }))
    .map_err(data_error)?;
    let last_error_patch = if status == "failed" {
        Some(Some(
            last_error.unwrap_or_else(|| image_failure_error_record(None)),
        ))
    } else {
        Some(None)
    };
    let output = patch_task_artifact(
        workspace_root_path,
        &task_ref,
        None,
        TaskArtifactPatch {
            status: Some(status),
            payload_patch: Some(payload_patch),
            result: Some(Some(result)),
            last_error: last_error_patch,
            progress: Some(progress),
            current_attempt_worker_id: Some(Some(IMAGE_TASK_COMPLETION_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(data_error)?;
    response_from_output(output)
}

pub(crate) async fn attach_image_result_sidecar_refs(
    workspace_root_path: &Path,
    task_ref: &str,
    output: MediaTaskOutput,
    sidecar_store: Option<&SidecarStore>,
) -> Result<MediaTaskOutput, String> {
    if !matches!(output.normalized_status.as_str(), "partial" | "succeeded") {
        return Ok(output);
    }
    let Some(sidecar_context) = sidecar::image_sidecar_context(
        sidecar_store,
        workspace_root_path,
        task_ref,
        task_payload(&output),
    ) else {
        return Ok(output);
    };
    let Some(mut result) = output.record.result.clone() else {
        return Ok(output);
    };
    let changed = sidecar::attach_missing_image_sidecar_refs(&mut result, &sidecar_context).await?;
    if !changed {
        return Ok(output);
    }

    patch_task_artifact(
        workspace_root_path,
        task_ref,
        None,
        TaskArtifactPatch {
            result: Some(Some(result)),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(data_error)
}

pub fn get_media_task_artifact(
    params: MediaTaskArtifactLookupParams,
) -> Result<MediaTaskArtifactResponse, String> {
    let workspace_root = normalize_required_string(&params.project_root_path, "projectRootPath")?;
    let task_ref = normalize_required_string(&params.task_ref, "taskRef")?;
    let output =
        load_task_output(Path::new(&workspace_root), &task_ref, None).map_err(data_error)?;
    response_from_output(output)
}

pub fn list_media_task_artifacts(
    params: MediaTaskArtifactListParams,
) -> Result<MediaTaskArtifactListResponse, String> {
    let workspace_root = normalize_required_string(&params.project_root_path, "projectRootPath")?;
    let status_filter = normalize_optional_string(params.status);
    let task_family_filter = normalize_optional_string(params.task_family);
    let task_type_filter = normalize_optional_string(params.task_type);
    let modality_contract_key_filter = normalize_optional_string(params.modality_contract_key);
    let routing_outcome_filter = normalize_optional_string(params.routing_outcome);
    let parsed_task_type = task_type_filter
        .as_deref()
        .map(|value| {
            value
                .parse::<MediaTaskType>()
                .map_err(|_| format!("不支持的 taskType: {value}"))
        })
        .transpose()?;
    let mut outputs = list_task_outputs(
        Path::new(&workspace_root),
        None,
        status_filter.as_deref(),
        task_family_filter.as_deref(),
        parsed_task_type,
        None,
    )
    .map_err(data_error)?;
    outputs.retain(|output| {
        media_task_matches_modality_filters(
            output,
            modality_contract_key_filter.as_deref(),
            routing_outcome_filter.as_deref(),
        )
    });
    if let Some(limit) = params.limit {
        outputs.truncate(limit);
    }
    let modality_runtime_contracts = build_modality_runtime_contract_index(&outputs);
    let tasks = response_vec_from_outputs(outputs)?;

    Ok(MediaTaskArtifactListResponse {
        success: true,
        workspace_root: workspace_root.clone(),
        artifact_root: Path::new(&workspace_root)
            .join(DEFAULT_ARTIFACT_ROOT)
            .to_string_lossy()
            .to_string(),
        filters: MediaTaskArtifactListFilters {
            status: status_filter,
            task_family: task_family_filter,
            task_type: parsed_task_type.map(|value| value.as_str().to_string()),
            modality_contract_key: modality_contract_key_filter,
            routing_outcome: routing_outcome_filter,
            limit: params.limit,
        },
        total: tasks.len(),
        modality_runtime_contracts,
        tasks,
    })
}

pub fn cancel_media_task_artifact(
    params: MediaTaskArtifactLookupParams,
) -> Result<MediaTaskArtifactResponse, String> {
    let workspace_root = normalize_required_string(&params.project_root_path, "projectRootPath")?;
    let task_ref = normalize_required_string(&params.task_ref, "taskRef")?;
    let output = update_task_status(Path::new(&workspace_root), &task_ref, None, "cancelled")
        .map_err(data_error)?;
    response_from_output(output)
}

pub(crate) fn retry_media_task_artifact(
    params: MediaTaskArtifactLookupParams,
) -> Result<MediaTaskArtifactResponse, String> {
    let workspace_root = normalize_required_string(&params.project_root_path, "projectRootPath")?;
    let task_ref = normalize_required_string(&params.task_ref, "taskRef")?;
    let output =
        retry_task_artifact(Path::new(&workspace_root), &task_ref, None).map_err(data_error)?;
    response_from_output(output)
}

#[cfg(test)]
mod tests;
