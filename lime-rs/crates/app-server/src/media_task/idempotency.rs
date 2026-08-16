use app_server_protocol::{
    MediaTaskArtifactAudioCreateParams, MediaTaskArtifactImageCreateParams,
    MediaTaskArtifactTranscriptionCreateParams, MediaTaskArtifactVideoCreateParams,
};
use serde_json::{json, Value};

pub(super) fn build_image_idempotency_key(params: &MediaTaskArtifactImageCreateParams) -> String {
    let seed = json!({
        "kind": "image",
        "projectRootPath": params.project_root_path,
        "prompt": params.prompt,
        "mode": params.mode,
        "size": params.size,
        "aspectRatio": params.aspect_ratio,
        "count": params.count,
        "style": params.style,
        "providerId": params.provider_id,
        "model": params.model,
        "threadId": params.thread_id,
        "turnId": params.turn_id,
        "contentId": params.content_id,
        "targetOutputId": params.target_output_id,
        "targetOutputRefId": params.target_output_ref_id,
        "slotId": params.slot_id,
        "referenceImages": params.reference_images,
        "storyboardSlots": params.storyboard_slots,
    });
    format!("app-server:media:image:{:x}", sha256_json(&seed))
}

pub(super) fn build_audio_idempotency_key(params: &MediaTaskArtifactAudioCreateParams) -> String {
    let seed = json!({
        "kind": "audio",
        "projectRootPath": params.project_root_path,
        "sourceText": params.source_text,
        "voice": params.voice,
        "voiceStyle": params.voice_style,
        "targetLanguage": params.target_language,
        "providerId": params.provider_id,
        "model": params.model,
        "threadId": params.thread_id,
        "turnId": params.turn_id,
        "contentId": params.content_id,
        "outputPath": params.output_path,
    });
    format!("app-server:media:audio:{:x}", sha256_json(&seed))
}

pub(super) fn build_video_idempotency_key(params: &MediaTaskArtifactVideoCreateParams) -> String {
    let seed = json!({
        "kind": "video",
        "projectRootPath": params.project_root_path,
        "prompt": params.prompt,
        "providerId": params.provider_id,
        "model": params.model,
        "threadId": params.thread_id,
        "turnId": params.turn_id,
        "contentId": params.content_id,
        "aspectRatio": params.aspect_ratio,
        "resolution": params.resolution,
        "duration": params.duration,
        "imageUrl": params.image_url,
        "endImageUrl": params.end_image_url,
        "seed": params.seed,
        "generateAudio": params.generate_audio,
        "cameraFixed": params.camera_fixed,
        "outputPath": params.output_path,
    });
    format!("app-server:media:video:{:x}", sha256_json(&seed))
}

pub(super) fn build_transcription_idempotency_key(
    params: &MediaTaskArtifactTranscriptionCreateParams,
) -> String {
    let seed = json!({
        "kind": "transcription",
        "projectRootPath": params.project_root_path,
        "prompt": params.prompt,
        "sourcePath": params.source_path,
        "sourceUrl": params.source_url,
        "language": params.language,
        "outputFormat": params.output_format,
        "speakerLabels": params.speaker_labels,
        "timestamps": params.timestamps,
        "providerId": params.provider_id,
        "model": params.model,
        "sessionId": params.session_id,
        "threadId": params.thread_id,
        "turnId": params.turn_id,
        "contentId": params.content_id,
        "outputPath": params.output_path,
    });
    format!("app-server:media:transcription:{:x}", sha256_json(&seed))
}

fn sha256_json(value: &Value) -> sha2::digest::Output<sha2::Sha256> {
    use sha2::Digest;
    sha2::Sha256::digest(serde_json::to_string(value).unwrap_or_default().as_bytes())
}
