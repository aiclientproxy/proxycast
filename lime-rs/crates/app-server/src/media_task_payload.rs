use crate::media_runtime_contract::{
    runtime_contract_or_default, MediaRuntimeContractKind, AUDIO_TRANSCRIPTION_CONTRACT_KEY,
    AUDIO_TRANSCRIPTION_ROUTING_SLOT, IMAGE_GENERATION_CONTRACT_KEY, IMAGE_GENERATION_ROUTING_SLOT,
    VIDEO_GENERATION_CONTRACT_KEY, VIDEO_GENERATION_ROUTING_SLOT, VOICE_GENERATION_CONTRACT_KEY,
    VOICE_GENERATION_ROUTING_SLOT,
};
use crate::model_task_contract::{
    apply_media_route_assessment_payload, build_model_task_request, model_task_request_value,
    MediaRouteAssessment, ModelTaskRequestInput,
};
use app_server_protocol::{
    MediaTaskArtifactAudioCreateParams, MediaTaskArtifactImageCreateParams,
    MediaTaskArtifactTranscriptionCreateParams, MediaTaskArtifactVideoCreateParams, ModelRefSource,
    ModelTaskKind, ModelTaskRequest, ModelTaskSource,
};
use serde_json::{json, Value};

pub(crate) const AUDIO_TASK_DEFAULT_MIME_TYPE: &str = "audio/mpeg";

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

fn normalize_string_list(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .filter_map(|value| normalize_optional_string(Some(value)))
        .collect()
}

fn image_required_capabilities(params: &MediaTaskArtifactImageCreateParams) -> Vec<String> {
    if params.required_capabilities.is_empty() {
        vec!["image_generation".to_string()]
    } else {
        normalize_string_list(params.required_capabilities.clone())
    }
}

fn image_route_required_capabilities() -> Vec<String> {
    vec!["image_generation".to_string()]
}

fn image_input_modalities(params: &MediaTaskArtifactImageCreateParams) -> Vec<String> {
    let mut modalities = vec!["text".to_string()];
    if params
        .reference_images
        .iter()
        .any(|reference| !reference.trim().is_empty())
    {
        modalities.push("image".to_string());
    }
    modalities
}

pub(crate) fn image_model_task_request(
    params: &MediaTaskArtifactImageCreateParams,
) -> ModelTaskRequest {
    let modality_contract_key = normalize_optional_string(params.modality_contract_key.clone())
        .unwrap_or_else(|| IMAGE_GENERATION_CONTRACT_KEY.to_string());
    let routing_slot = normalize_optional_string(params.routing_slot.clone())
        .unwrap_or_else(|| IMAGE_GENERATION_ROUTING_SLOT.to_string());
    build_model_task_request(ModelTaskRequestInput {
        task_kind: ModelTaskKind::ImageGenerate,
        source: ModelTaskSource::MediaTaskArtifact,
        provider_id: params.provider_id.clone(),
        model_id: params.model.clone(),
        model_ref_source: ModelRefSource::Task,
        modality_contract_key: Some(modality_contract_key),
        routing_slot: Some(routing_slot),
        task_families: vec!["image_generation".to_string()],
        input_modalities: image_input_modalities(params),
        output_modalities: vec!["image".to_string()],
        runtime_features: Vec::new(),
        capabilities: image_route_required_capabilities(),
        session_id: params.session_id.clone(),
        thread_id: params.thread_id.clone(),
        turn_id: params.turn_id.clone(),
        content_id: params.content_id.clone(),
        trace_id: None,
    })
}

fn image_command_run_id(params: &MediaTaskArtifactImageCreateParams) -> String {
    normalize_optional_string(params.turn_id.clone())
        .map(|turn_id| format!("image-command-run-{turn_id}"))
        .or_else(|| {
            normalize_optional_string(params.content_id.clone())
                .map(|content_id| format!("image-command-run-{content_id}"))
        })
        .or_else(|| {
            normalize_optional_string(params.session_id.clone())
                .map(|session_id| format!("image-command-run-{session_id}"))
        })
        .unwrap_or_else(|| "image-command-run-manual".to_string())
}

fn image_command_run_title(params: &MediaTaskArtifactImageCreateParams) -> String {
    normalize_optional_string(params.title.clone()).unwrap_or_else(|| "图片生成".to_string())
}

fn image_command_requested_count(params: &MediaTaskArtifactImageCreateParams) -> u32 {
    let count = params.count.unwrap_or(1).max(1);
    count.max(params.storyboard_slots.len() as u32)
}

fn image_command_branch_snapshot(
    run_id: &str,
    params: &MediaTaskArtifactImageCreateParams,
    index: u32,
) -> Value {
    let slot = params
        .storyboard_slots
        .get(index.saturating_sub(1) as usize);
    let branch_id = slot
        .and_then(|slot| normalize_optional_string(slot.slot_id.clone()))
        .map(|slot_id| format!("{run_id}:branch:{slot_id}"))
        .unwrap_or_else(|| format!("{run_id}:branch:{index}"));
    let title = slot
        .and_then(|slot| normalize_optional_string(slot.label.clone()))
        .unwrap_or_else(|| {
            if image_command_requested_count(params) > 1 {
                format!("图片 {index}")
            } else {
                "图片结果".to_string()
            }
        });
    let prompt = slot
        .and_then(|slot| normalize_optional_string(Some(slot.prompt.clone())))
        .unwrap_or_else(|| params.prompt.clone());

    json!({
        "branch_id": branch_id,
        "branchId": branch_id,
        "title": title,
        "prompt": prompt,
        "slot_id": slot.and_then(|slot| normalize_optional_string(slot.slot_id.clone())),
        "slotId": slot.and_then(|slot| normalize_optional_string(slot.slot_id.clone())),
        "shot_type": slot.and_then(|slot| normalize_optional_string(slot.shot_type.clone())),
        "shotType": slot.and_then(|slot| normalize_optional_string(slot.shot_type.clone())),
        "status": "queued",
    })
}

fn image_command_run_snapshot(
    params: &MediaTaskArtifactImageCreateParams,
    route_assessment: Option<&MediaRouteAssessment>,
) -> Value {
    let run_id = image_command_run_id(params);
    let requested_count = image_command_requested_count(params);
    let route_blocked = route_assessment
        .map(|assessment| assessment.route_failure.is_some())
        .unwrap_or(false);
    let route_status = if route_blocked { "failed" } else { "succeeded" };
    let run_status = if route_blocked { "failed" } else { "queued" };
    let branches = (1..=requested_count)
        .map(|index| image_command_branch_snapshot(&run_id, params, index))
        .collect::<Vec<_>>();

    json!({
        "run_id": run_id,
        "runId": run_id,
        "workflow_key": "image_command_workflow",
        "workflowKey": "image_command_workflow",
        "session_id": params.session_id,
        "sessionId": params.session_id,
        "thread_id": params.thread_id,
        "threadId": params.thread_id,
        "turn_id": params.turn_id,
        "turnId": params.turn_id,
        "title": image_command_run_title(params),
        "summary": params.prompt,
        "requested_count": requested_count,
        "requestedCount": requested_count,
        "status": run_status,
        "steps": [
            {
                "id": "intent",
                "title": "解析图片需求",
                "status": "succeeded"
            },
            {
                "id": "route",
                "title": "确认图片模型",
                "status": route_status
            },
            {
                "id": "create_tasks",
                "title": "创建图片任务",
                "status": if route_blocked { "pending" } else { "succeeded" }
            },
            {
                "id": "generate",
                "title": "生成图片",
                "status": if route_blocked { "pending" } else { "running" }
            },
            {
                "id": "persist_outputs",
                "title": "保存结果",
                "status": "pending"
            }
        ],
        "branches": branches,
        "next_actions": [
            {
                "type": "open_workbench"
            }
        ],
        "nextActions": [
            {
                "type": "open_workbench"
            }
        ]
    })
}

pub(crate) fn create_image_payload(
    params: &MediaTaskArtifactImageCreateParams,
    route_assessment: Option<&MediaRouteAssessment>,
) -> Value {
    let modality_contract_key = normalize_optional_string(params.modality_contract_key.clone())
        .unwrap_or_else(|| IMAGE_GENERATION_CONTRACT_KEY.to_string());
    let modality =
        normalize_optional_string(params.modality.clone()).unwrap_or_else(|| "image".to_string());
    let routing_slot = normalize_optional_string(params.routing_slot.clone())
        .unwrap_or_else(|| IMAGE_GENERATION_ROUTING_SLOT.to_string());
    let required_capabilities = image_required_capabilities(params);
    let model_task_request = model_task_request_value(&image_model_task_request(params));

    let image_command_run = image_command_run_snapshot(params, route_assessment);
    let mut payload = json!({
        "prompt": params.prompt,
        "title_generation_result": params.title_generation_result,
        "persona_context": params.persona_context,
        "presentation": params.presentation,
        "taste_context": params.taste_context,
        "mode": params.mode,
        "raw_text": params.raw_text,
        "layout_hint": params.layout_hint,
        "size": params.size,
        "aspect_ratio": params.aspect_ratio,
        "count": params.count.unwrap_or(1),
        "usage": params.usage,
        "style": params.style,
        "provider_id": params.provider_id,
        "model": params.model,
        "executor_mode": params.executor_mode,
        "outer_model": params.outer_model,
        "session_id": params.session_id,
        "thread_id": params.thread_id,
        "turn_id": params.turn_id,
        "project_id": params.project_id,
        "content_id": params.content_id,
        "entry_source": normalize_optional_string(params.entry_source.clone()).unwrap_or_else(|| "at_image_command".to_string()),
        "modality_contract_key": modality_contract_key,
        "modality": modality,
        "required_capabilities": required_capabilities,
        "routing_slot": routing_slot,
        "model_task_request": model_task_request,
        "modelTaskRequest": model_task_request,
        "runtime_contract": runtime_contract_or_default(params.runtime_contract.as_ref(), MediaRuntimeContractKind::ImageGeneration),
        "requested_target": params.requested_target,
        "slot_id": params.slot_id,
        "anchor_hint": params.anchor_hint,
        "anchor_section_title": params.anchor_section_title,
        "anchor_text": params.anchor_text,
        "target_output_id": params.target_output_id,
        "target_output_ref_id": params.target_output_ref_id,
        "reference_images": params.reference_images,
        "storyboard_slots": params.storyboard_slots,
        "image_command_run": image_command_run,
        "imageCommandRun": image_command_run,
    });
    apply_media_route_assessment_payload(&mut payload, route_assessment);
    payload
}

fn video_required_capabilities(params: &MediaTaskArtifactVideoCreateParams) -> Vec<String> {
    if params.required_capabilities.is_empty() {
        vec!["video_generation".to_string()]
    } else {
        normalize_string_list(params.required_capabilities.clone())
    }
}

pub(crate) fn video_model_task_request(
    params: &MediaTaskArtifactVideoCreateParams,
) -> ModelTaskRequest {
    let modality_contract_key = normalize_optional_string(params.modality_contract_key.clone())
        .unwrap_or_else(|| VIDEO_GENERATION_CONTRACT_KEY.to_string());
    let routing_slot = normalize_optional_string(params.routing_slot.clone())
        .unwrap_or_else(|| VIDEO_GENERATION_ROUTING_SLOT.to_string());
    build_model_task_request(ModelTaskRequestInput {
        task_kind: ModelTaskKind::VideoGenerate,
        source: ModelTaskSource::MediaTaskArtifact,
        provider_id: params.provider_id.clone(),
        model_id: params.model.clone(),
        model_ref_source: ModelRefSource::Task,
        modality_contract_key: Some(modality_contract_key),
        routing_slot: Some(routing_slot),
        task_families: vec!["video_generation".to_string()],
        input_modalities: vec!["text".to_string(), "image".to_string()],
        output_modalities: vec!["video".to_string()],
        runtime_features: Vec::new(),
        capabilities: video_required_capabilities(params),
        session_id: params.session_id.clone(),
        thread_id: params.thread_id.clone(),
        turn_id: params.turn_id.clone(),
        content_id: params.content_id.clone(),
        trace_id: None,
    })
}

pub(crate) fn create_video_payload(
    params: &MediaTaskArtifactVideoCreateParams,
    route_assessment: Option<&MediaRouteAssessment>,
) -> Value {
    let modality_contract_key = normalize_optional_string(params.modality_contract_key.clone())
        .unwrap_or_else(|| VIDEO_GENERATION_CONTRACT_KEY.to_string());
    let modality =
        normalize_optional_string(params.modality.clone()).unwrap_or_else(|| "video".to_string());
    let routing_slot = normalize_optional_string(params.routing_slot.clone())
        .unwrap_or_else(|| VIDEO_GENERATION_ROUTING_SLOT.to_string());
    let required_capabilities = video_required_capabilities(params);
    let model_task_request = model_task_request_value(&video_model_task_request(params));

    let mut payload = json!({
        "prompt": params.prompt,
        "project_root_path": params.project_root_path,
        "raw_text": params.raw_text,
        "aspect_ratio": params.aspect_ratio,
        "resolution": params.resolution,
        "duration": params.duration,
        "image_url": params.image_url,
        "end_image_url": params.end_image_url,
        "seed": params.seed,
        "generate_audio": params.generate_audio,
        "camera_fixed": params.camera_fixed,
        "provider_id": params.provider_id,
        "model": params.model,
        "session_id": params.session_id,
        "thread_id": params.thread_id,
        "turn_id": params.turn_id,
        "project_id": params.project_id,
        "content_id": params.content_id,
        "entry_source": normalize_optional_string(params.entry_source.clone()).unwrap_or_else(|| "video_workspace".to_string()),
        "modality_contract_key": modality_contract_key,
        "modality": modality,
        "required_capabilities": required_capabilities,
        "routing_slot": routing_slot,
        "model_task_request": model_task_request,
        "modelTaskRequest": model_task_request,
        "runtime_contract": runtime_contract_or_default(params.runtime_contract.as_ref(), MediaRuntimeContractKind::VideoGeneration),
        "requested_target": normalize_optional_string(params.requested_target.clone()).unwrap_or_else(|| "video".to_string()),
    });
    apply_media_route_assessment_payload(&mut payload, route_assessment);
    payload
}

fn audio_required_capabilities(params: &MediaTaskArtifactAudioCreateParams) -> Vec<String> {
    if params.required_capabilities.is_empty() {
        vec!["voice_generation".to_string()]
    } else {
        normalize_string_list(params.required_capabilities.clone())
    }
}

pub(crate) fn audio_model_task_request(
    params: &MediaTaskArtifactAudioCreateParams,
) -> ModelTaskRequest {
    let modality_contract_key = normalize_optional_string(params.modality_contract_key.clone())
        .unwrap_or_else(|| VOICE_GENERATION_CONTRACT_KEY.to_string());
    let routing_slot = normalize_optional_string(params.routing_slot.clone())
        .unwrap_or_else(|| VOICE_GENERATION_ROUTING_SLOT.to_string());
    build_model_task_request(ModelTaskRequestInput {
        task_kind: ModelTaskKind::VoiceGenerate,
        source: ModelTaskSource::MediaTaskArtifact,
        provider_id: params.provider_id.clone(),
        model_id: params.model.clone(),
        model_ref_source: ModelRefSource::Task,
        modality_contract_key: Some(modality_contract_key),
        routing_slot: Some(routing_slot),
        task_families: vec!["text_to_speech".to_string()],
        input_modalities: vec!["text".to_string()],
        output_modalities: vec!["audio".to_string()],
        runtime_features: Vec::new(),
        capabilities: audio_required_capabilities(params),
        session_id: params.session_id.clone(),
        thread_id: params.thread_id.clone(),
        turn_id: params.turn_id.clone(),
        content_id: params.content_id.clone(),
        trace_id: None,
    })
}

pub(crate) fn create_audio_payload(
    params: &MediaTaskArtifactAudioCreateParams,
    route_assessment: Option<&MediaRouteAssessment>,
) -> Value {
    let modality_contract_key = normalize_optional_string(params.modality_contract_key.clone())
        .unwrap_or_else(|| VOICE_GENERATION_CONTRACT_KEY.to_string());
    let modality =
        normalize_optional_string(params.modality.clone()).unwrap_or_else(|| "audio".to_string());
    let routing_slot = normalize_optional_string(params.routing_slot.clone())
        .unwrap_or_else(|| VOICE_GENERATION_ROUTING_SLOT.to_string());
    let required_capabilities = audio_required_capabilities(params);
    let source_text = params.source_text.trim();
    let mime_type = normalize_optional_string(params.mime_type.clone())
        .unwrap_or_else(|| AUDIO_TASK_DEFAULT_MIME_TYPE.to_string());
    let model_task_request = model_task_request_value(&audio_model_task_request(params));

    let mut payload = json!({
        "prompt": source_text,
        "source_text": source_text,
        "raw_text": params.raw_text,
        "voice": params.voice,
        "voice_style": params.voice_style,
        "target_language": params.target_language,
        "mime_type": mime_type,
        "audio_path": params.audio_path,
        "output_path": params.output_path,
        "duration_ms": params.duration_ms,
        "provider_id": params.provider_id,
        "model": params.model,
        "session_id": params.session_id,
        "thread_id": params.thread_id,
        "turn_id": params.turn_id,
        "project_id": params.project_id,
        "content_id": params.content_id,
        "entry_source": normalize_optional_string(params.entry_source.clone()).unwrap_or_else(|| "at_voice_command".to_string()),
        "modality_contract_key": modality_contract_key,
        "modality": modality,
        "required_capabilities": required_capabilities,
        "routing_slot": routing_slot,
        "model_task_request": model_task_request,
        "modelTaskRequest": model_task_request,
        "runtime_contract": runtime_contract_or_default(params.runtime_contract.as_ref(), MediaRuntimeContractKind::VoiceGeneration),
        "requested_target": normalize_optional_string(params.requested_target.clone()).unwrap_or_else(|| "voice".to_string()),
        "audio_output": {
            "kind": "audio_output",
            "status": "pending",
            "audio_path": params.audio_path,
            "mime_type": mime_type,
            "duration_ms": params.duration_ms,
            "source_text": source_text,
            "voice": params.voice,
            "voice_style": params.voice_style,
            "target_language": params.target_language,
        }
    });
    apply_media_route_assessment_payload(&mut payload, route_assessment);
    payload
}

fn transcription_required_capabilities(
    params: &MediaTaskArtifactTranscriptionCreateParams,
) -> Vec<String> {
    if params.required_capabilities.is_empty() {
        vec!["audio_transcription".to_string()]
    } else {
        normalize_string_list(params.required_capabilities.clone())
    }
}

pub(crate) fn transcription_model_task_request(
    params: &MediaTaskArtifactTranscriptionCreateParams,
) -> ModelTaskRequest {
    let modality_contract_key = normalize_optional_string(params.modality_contract_key.clone())
        .unwrap_or_else(|| AUDIO_TRANSCRIPTION_CONTRACT_KEY.to_string());
    let routing_slot = normalize_optional_string(params.routing_slot.clone())
        .unwrap_or_else(|| AUDIO_TRANSCRIPTION_ROUTING_SLOT.to_string());
    build_model_task_request(ModelTaskRequestInput {
        task_kind: ModelTaskKind::TranscriptionGenerate,
        source: ModelTaskSource::MediaTaskArtifact,
        provider_id: params.provider_id.clone(),
        model_id: params.model.clone(),
        model_ref_source: ModelRefSource::Task,
        modality_contract_key: Some(modality_contract_key),
        routing_slot: Some(routing_slot),
        task_families: vec!["speech_to_text".to_string()],
        input_modalities: vec!["audio".to_string()],
        output_modalities: vec!["text".to_string()],
        runtime_features: Vec::new(),
        capabilities: transcription_required_capabilities(params),
        session_id: params.session_id.clone(),
        thread_id: params.thread_id.clone(),
        turn_id: params.turn_id.clone(),
        content_id: params.content_id.clone(),
        trace_id: None,
    })
}

pub(crate) fn create_transcription_payload(
    params: &MediaTaskArtifactTranscriptionCreateParams,
    route_assessment: Option<&MediaRouteAssessment>,
) -> Value {
    let modality_contract_key = normalize_optional_string(params.modality_contract_key.clone())
        .unwrap_or_else(|| AUDIO_TRANSCRIPTION_CONTRACT_KEY.to_string());
    let modality =
        normalize_optional_string(params.modality.clone()).unwrap_or_else(|| "audio".to_string());
    let routing_slot = normalize_optional_string(params.routing_slot.clone())
        .unwrap_or_else(|| AUDIO_TRANSCRIPTION_ROUTING_SLOT.to_string());
    let required_capabilities = transcription_required_capabilities(params);
    let model_task_request = model_task_request_value(&transcription_model_task_request(params));
    let output_format = normalize_optional_string(params.output_format.clone())
        .unwrap_or_else(|| "txt".to_string());
    let mut payload = json!({
        "prompt": params.prompt,
        "raw_text": params.raw_text,
        "source_path": params.source_path,
        "source_url": params.source_url,
        "language": params.language,
        "output_format": output_format,
        "speaker_labels": params.speaker_labels,
        "timestamps": params.timestamps,
        "output_path": params.output_path,
        "provider_id": params.provider_id,
        "model": params.model,
        "session_id": params.session_id,
        "thread_id": params.thread_id,
        "turn_id": params.turn_id,
        "project_id": params.project_id,
        "content_id": params.content_id,
        "entry_source": normalize_optional_string(params.entry_source.clone())
            .unwrap_or_else(|| "at_transcription_command".to_string()),
        "modality_contract_key": modality_contract_key,
        "modality": modality,
        "required_capabilities": required_capabilities,
        "routing_slot": routing_slot,
        "model_task_request": model_task_request,
        "modelTaskRequest": model_task_request,
        "runtime_contract": runtime_contract_or_default(
            params.runtime_contract.as_ref(),
            MediaRuntimeContractKind::AudioTranscription,
        ),
        "requested_target": normalize_optional_string(params.requested_target.clone())
            .unwrap_or_else(|| "transcript".to_string()),
        "transcript": {
            "kind": "transcript",
            "status": "pending",
            "source_path": params.source_path,
            "source_url": params.source_url,
            "language": params.language,
            "output_format": output_format,
            "speaker_labels": params.speaker_labels,
            "timestamps": params.timestamps,
            "provider_id": params.provider_id,
            "model": params.model,
        }
    });
    apply_media_route_assessment_payload(&mut payload, route_assessment);
    payload
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_route_assembly::{
        resolved_route_from_task_with_credential, ModelRouteSelection,
    };
    use app_server_protocol::ModelRefSource;

    #[test]
    fn image_payload_contains_typed_model_task_request() {
        let payload = create_image_payload(
            &MediaTaskArtifactImageCreateParams {
                project_root_path: "/tmp/project".to_string(),
                prompt: "生成封面".to_string(),
                provider_id: Some("openai".to_string()),
                model: Some("gpt-image-2".to_string()),
                session_id: Some("session-1".to_string()),
                thread_id: Some("thread-1".to_string()),
                turn_id: Some("turn-1".to_string()),
                content_id: Some("content-1".to_string()),
                ..MediaTaskArtifactImageCreateParams::default()
            },
            None,
        );

        assert_eq!(
            payload["model_task_request"]["taskKind"].as_str(),
            Some("image_generate")
        );
        assert_eq!(
            payload["model_task_request"]["modelRef"]["providerId"].as_str(),
            Some("openai")
        );
        assert_eq!(
            payload["model_task_request"]["modelRef"]["modelId"].as_str(),
            Some("gpt-image-2")
        );
        assert_eq!(
            payload["model_task_request"]["routingSlot"].as_str(),
            Some(IMAGE_GENERATION_ROUTING_SLOT)
        );
        assert_eq!(
            payload["model_task_request"]["requirements"]["outputModalities"][0].as_str(),
            Some("image")
        );
        assert_eq!(
            payload["model_task_request"]["requirements"]["inputModalities"],
            json!(["text"])
        );
    }

    #[test]
    fn image_task_requires_image_input_only_when_reference_is_present() {
        let text_to_image = image_model_task_request(&MediaTaskArtifactImageCreateParams {
            prompt: "生成封面".to_string(),
            reference_images: vec!["  ".to_string()],
            ..MediaTaskArtifactImageCreateParams::default()
        });
        let image_to_image = image_model_task_request(&MediaTaskArtifactImageCreateParams {
            prompt: "改成夜景".to_string(),
            reference_images: vec!["https://cdn.example.test/source.png".to_string()],
            ..MediaTaskArtifactImageCreateParams::default()
        });

        assert_eq!(
            text_to_image.requirements.input_modalities,
            vec!["text".to_string()]
        );
        assert_eq!(
            image_to_image.requirements.input_modalities,
            vec!["text".to_string(), "image".to_string()]
        );
    }

    #[test]
    fn image_task_lowers_workflow_capabilities_to_image_provider_requirements() {
        let params = MediaTaskArtifactImageCreateParams {
            prompt: "生成封面".to_string(),
            required_capabilities: vec![
                "text_generation".to_string(),
                "image_generation".to_string(),
                "vision_input".to_string(),
            ],
            ..MediaTaskArtifactImageCreateParams::default()
        };

        let payload = create_image_payload(&params, None);
        let request = image_model_task_request(&params);

        assert_eq!(
            payload["required_capabilities"],
            json!(["text_generation", "image_generation", "vision_input"])
        );
        assert_eq!(
            request.requirements.capabilities,
            vec!["image_generation".to_string()]
        );
    }

    #[test]
    fn image_reference_reports_exact_image_input_capability_gap() {
        let params = MediaTaskArtifactImageCreateParams {
            prompt: "改成夜景".to_string(),
            reference_images: vec!["https://cdn.example.test/source.png".to_string()],
            ..MediaTaskArtifactImageCreateParams::default()
        };
        let snapshot =
            crate::model_task_contract::capability_snapshot_from_model_capabilities(&json!({
                "capabilities": {
                    "vision": false,
                    "streaming": true
                },
                "taskFamilies": ["image_generation"],
                "inputModalities": ["text"],
                "outputModalities": ["image"],
                "runtimeFeatures": ["streaming", "images_api"]
            }));

        let assessment =
            MediaRouteAssessment::from_snapshot(&image_model_task_request(&params), snapshot);

        let failure = assessment
            .route_failure
            .as_ref()
            .expect("image input capability gap");
        assert_eq!(failure.reason_code, "capability_gap");
        assert_eq!(
            failure.capability_gap.as_deref(),
            Some("input_modality:image")
        );
    }

    #[test]
    fn image_payload_contains_command_run_snapshot_with_branches() {
        let payload = create_image_payload(
            &MediaTaskArtifactImageCreateParams {
                project_root_path: "/tmp/project".to_string(),
                prompt: "生成两张青柠主图".to_string(),
                title: Some("青柠主图".to_string()),
                count: Some(2),
                provider_id: Some("openai".to_string()),
                model: Some("gpt-image-2".to_string()),
                session_id: Some("session-1".to_string()),
                thread_id: Some("thread-1".to_string()),
                turn_id: Some("turn-1".to_string()),
                storyboard_slots: vec![
                    app_server_protocol::ImageStoryboardSlotInput {
                        prompt: "白底青柠主图".to_string(),
                        slot_id: Some("white-bg".to_string()),
                        label: Some("白底主图".to_string()),
                        shot_type: Some("product_main".to_string()),
                    },
                    app_server_protocol::ImageStoryboardSlotInput {
                        prompt: "浅灰背景青柠主图".to_string(),
                        slot_id: Some("gray-bg".to_string()),
                        label: Some("浅灰主图".to_string()),
                        shot_type: Some("product_main".to_string()),
                    },
                ],
                ..MediaTaskArtifactImageCreateParams::default()
            },
            None,
        );

        let run = &payload["image_command_run"];
        assert_eq!(run["run_id"].as_str(), Some("image-command-run-turn-1"));
        assert_eq!(run["workflow_key"].as_str(), Some("image_command_workflow"));
        assert_eq!(run["title"].as_str(), Some("青柠主图"));
        assert_eq!(run["requested_count"].as_u64(), Some(2));
        assert_eq!(run["status"].as_str(), Some("queued"));
        assert_eq!(run["steps"][0]["id"].as_str(), Some("intent"));
        assert_eq!(run["steps"][2]["status"].as_str(), Some("succeeded"));
        assert_eq!(run["branches"].as_array().map(Vec::len), Some(2));
        assert_eq!(
            run["branches"][0]["branch_id"].as_str(),
            Some("image-command-run-turn-1:branch:white-bg")
        );
        assert_eq!(run["branches"][0]["title"].as_str(), Some("白底主图"));
        assert_eq!(
            run["branches"][1]["prompt"].as_str(),
            Some("浅灰背景青柠主图")
        );
        assert_eq!(payload["imageCommandRun"], payload["image_command_run"]);
    }

    #[test]
    fn image_payload_marks_route_failure_when_assessment_blocks() {
        let params = MediaTaskArtifactImageCreateParams {
            project_root_path: "/tmp/project".to_string(),
            prompt: "生成封面".to_string(),
            provider_id: Some("openai".to_string()),
            model: Some("text-only".to_string()),
            ..MediaTaskArtifactImageCreateParams::default()
        };
        let snapshot =
            crate::model_task_contract::capability_snapshot_from_model_capabilities(&json!({
                "capabilities": {
                    "vision": false,
                    "streaming": true
                },
                "taskFamilies": ["chat"],
                "inputModalities": ["text"],
                "outputModalities": ["text"],
                "runtimeFeatures": ["streaming"]
            }));
        let assessment =
            MediaRouteAssessment::from_snapshot(&image_model_task_request(&params), snapshot);
        let payload = create_image_payload(&params, Some(&assessment));

        assert_eq!(payload["failure_code"].as_str(), Some("capability_gap"));
        assert_eq!(payload["failureCode"].as_str(), Some("capability_gap"));
        assert_eq!(
            payload["route_failure"]["category"].as_str(),
            Some("capability_gap")
        );
        assert_eq!(
            payload["route_failure"]["capabilityGap"].as_str(),
            Some("task_family:image_generation")
        );
        assert_eq!(
            payload["model_route_assessment"]["status"].as_str(),
            Some("blocked")
        );
    }

    #[test]
    fn image_payload_contains_resolved_route_when_assessment_resolves_route() {
        let params = MediaTaskArtifactImageCreateParams {
            project_root_path: "/tmp/project".to_string(),
            prompt: "生成封面".to_string(),
            provider_id: Some("openai".to_string()),
            model: Some("gpt-image-2".to_string()),
            ..MediaTaskArtifactImageCreateParams::default()
        };
        let task_request = image_model_task_request(&params);
        let route = resolved_route_from_task_with_credential(
            &task_request,
            ModelRouteSelection {
                provider_id: "openai",
                model_id: "gpt-image-2",
                model_ref_source: ModelRefSource::Task,
                reasoning_effort: None,
            },
            &json!({
                "providerReadiness": {
                    "ready": true,
                    "status": "ready"
                },
                "routingMode": "task_route",
                "decisionSource": "media_task_artifact",
                "decisionReason": "explicit_task_model",
                "serviceModelSlot": IMAGE_GENERATION_ROUTING_SLOT,
                "modelRegistry": {
                    "source": "api",
                    "reasonCode": "matched_media_task_model",
                    "modelCapabilities": {
                        "provenance": "provider_explicit",
                        "capabilities": {
                            "vision": false,
                            "streaming": false
                        },
                        "taskFamilies": ["image_generation"],
                        "inputModalities": ["text"],
                        "outputModalities": ["image"],
                        "runtimeFeatures": ["images_api"]
                    }
                }
            }),
            None,
            None,
            None,
        );
        let assessment =
            MediaRouteAssessment::from_snapshot(&task_request, route.capability_snapshot.clone())
                .with_resolved_route(route);

        let payload = create_image_payload(&params, Some(&assessment));

        assert_eq!(
            payload["resolved_route"]["modelRef"]["source"].as_str(),
            Some("task")
        );
        assert_eq!(
            payload["resolvedRoute"]["modelRef"]["routingSlot"].as_str(),
            Some(IMAGE_GENERATION_ROUTING_SLOT)
        );
        assert_eq!(
            payload["model_route_assessment"]["resolvedRoute"]["decision"]["routingMode"].as_str(),
            Some("task_route")
        );
        assert_eq!(
            payload["resolved_route"]["protocol"].as_str(),
            Some("openai_images")
        );
        assert_eq!(
            payload["model_route_execution"]["executor"]["kind"].as_str(),
            Some("media_task_worker")
        );
        assert_eq!(
            payload["model_route_execution"]["executor"]["bindingKey"].as_str(),
            Some("mediaTaskArtifact/image/create")
        );
        assert_eq!(
            payload["model_route_execution"]["credentialResolver"]["owner"].as_str(),
            Some("media_task_worker")
        );
        assert_eq!(
            payload["model_route_execution"]["credentialResolver"]["secretMaterialStatus"].as_str(),
            Some("not_embedded")
        );
        assert_eq!(
            payload["model_route_assessment"]["routeExecution"]["route"]["providerId"].as_str(),
            Some("openai")
        );
        assert!(payload["route_failure"].is_null());
    }

    #[test]
    fn audio_payload_keeps_task_request_without_executable_route() {
        let payload = create_audio_payload(
            &MediaTaskArtifactAudioCreateParams {
                project_root_path: "/tmp/project".to_string(),
                source_text: "朗读这段文字".to_string(),
                provider_id: Some("voice-provider".to_string()),
                model: Some("voice-model".to_string()),
                session_id: Some("session-1".to_string()),
                thread_id: Some("thread-1".to_string()),
                turn_id: Some("turn-1".to_string()),
                content_id: Some("content-1".to_string()),
                ..MediaTaskArtifactAudioCreateParams::default()
            },
            None,
        );

        assert_eq!(
            payload["model_task_request"]["taskKind"].as_str(),
            Some("voice_generate")
        );
        assert_eq!(
            payload["model_task_request"]["modelRef"]["providerId"].as_str(),
            Some("voice-provider")
        );
        assert_eq!(
            payload["model_task_request"]["routingSlot"].as_str(),
            Some(VOICE_GENERATION_ROUTING_SLOT)
        );
        assert_eq!(
            payload["model_task_request"]["requirements"]["outputModalities"][0].as_str(),
            Some("audio")
        );
        assert!(payload.get("resolved_route").is_none());
        assert!(payload.get("resolvedRoute").is_none());
        assert!(payload.get("model_route_execution").is_none());
        assert!(payload.get("modelRouteExecution").is_none());
    }
}
