use super::*;

#[test]
fn write_media_task_artifact_uses_default_task_root() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let output = write_media_task_artifact(
        temp_dir.path(),
        MediaTaskType::ImageGenerate,
        Some("配图".to_string()),
        serde_json::json!({ "prompt": "未来城市插图" }),
        None,
        None,
        None,
    )
    .expect("write media task");

    assert!(output.path.starts_with(".lime/tasks/image_generate/"));
    assert_eq!(output.task_type, "image_generate");
    assert_eq!(output.task_family, "image");
    assert_eq!(output.status, "pending_submit");
    assert_eq!(output.normalized_status, "pending");
    assert_eq!(output.attempt_count, 1);
    assert!(output.current_attempt_id.is_some());
    assert_eq!(output.record.attempts.len(), 1);
    assert_eq!(
        output.ui_hints.render_mode.as_deref(),
        Some("media_placeholder_card")
    );
    assert!(temp_dir.path().join(&output.path).exists());
}

#[test]
fn task_artifact_mutations_keep_single_parseable_json_file() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let output = write_task_artifact(
        temp_dir.path(),
        TaskType::ImageGenerate,
        Some("配图".to_string()),
        serde_json::json!({ "prompt": "未来城市插图" }),
        TaskWriteOptions::default(),
    )
    .expect("write media task");

    update_task_status(temp_dir.path(), &output.task_id, None, "running")
        .expect("update task status");
    let patched = patch_task_artifact(
        temp_dir.path(),
        &output.task_id,
        None,
        TaskArtifactPatch {
            status: Some("succeeded".to_string()),
            result: Some(Some(serde_json::json!({
                "artifacts": [
                    {
                        "kind": "image",
                        "path": "images/result.png"
                    }
                ]
            }))),
            ..TaskArtifactPatch::default()
        },
    )
    .expect("patch task result");

    let task_path = temp_dir.path().join(&patched.path);
    let content = std::fs::read_to_string(&task_path).expect("read task json");
    let parsed: TaskArtifactRecord = serde_json::from_str(&content).expect("parse task json");
    assert_eq!(parsed.normalized_status, "succeeded");
    assert_eq!(
        parsed
            .result
            .as_ref()
            .and_then(|value| value.pointer("/artifacts/0/path")),
        Some(&serde_json::json!("images/result.png"))
    );

    let parent = task_path.parent().expect("task parent should exist");
    let entries: Vec<_> = std::fs::read_dir(parent)
        .expect("read task dir")
        .map(|entry| {
            entry
                .expect("read task dir entry")
                .file_name()
                .to_string_lossy()
                .to_string()
        })
        .collect();
    assert_eq!(
        entries,
        vec![task_path
            .file_name()
            .expect("task file name")
            .to_string_lossy()
            .to_string()]
    );
}

#[test]
fn payload_patch_merges_nested_objects_without_dropping_existing_fields() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let output = write_task_artifact(
        temp_dir.path(),
        TaskType::AudioGenerate,
        None,
        serde_json::json!({
            "audio_output": {
                "kind": "audio_output",
                "audio_path": ".lime/media/audio/task.mp3",
                "mime_type": "audio/mpeg"
            }
        }),
        TaskWriteOptions::default(),
    )
    .expect("write audio task");

    patch_task_artifact(
        temp_dir.path(),
        &output.task_id,
        None,
        TaskArtifactPatch {
            payload_patch: Some(serde_json::json!({
                "audio_output": {
                    "status": "completed"
                }
            })),
            ..TaskArtifactPatch::default()
        },
    )
    .expect("patch nested payload");

    let patched = load_task_output(temp_dir.path(), &output.task_id, None).expect("load task");
    assert_eq!(
        patched.record.payload.pointer("/audio_output/status"),
        Some(&serde_json::json!("completed"))
    );
    assert_eq!(
        patched.record.payload.pointer("/audio_output/audio_path"),
        Some(&serde_json::json!(".lime/media/audio/task.mp3"))
    );
    assert_eq!(
        patched.record.payload.pointer("/audio_output/mime_type"),
        Some(&serde_json::json!("audio/mpeg"))
    );
}

#[test]
fn write_media_task_artifact_rejects_parent_dir_escape() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let error = write_media_task_artifact(
        temp_dir.path(),
        MediaTaskType::CoverGenerate,
        None,
        serde_json::json!({ "prompt": "封面" }),
        None,
        Some("../escape.json"),
        None,
    )
    .expect_err("should reject unsafe path");

    assert!(matches!(error, MediaRuntimeError::InvalidParams(_)));
}

#[test]
fn write_media_task_artifact_supports_custom_artifact_dir() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let output = write_media_task_artifact(
        temp_dir.path(),
        MediaTaskType::VideoGenerate,
        None,
        serde_json::json!({ "prompt": "短视频" }),
        Some("queued".to_string()),
        None,
        Some("custom/tasks"),
    )
    .expect("write media task");

    assert!(output.path.starts_with("custom/tasks/video_generate/"));
    assert_eq!(output.status, "queued");
    assert_eq!(output.normalized_status, "queued");
    assert_eq!(output.record.attempts.len(), 1);
    assert!(output.record.current_attempt_id.is_some());
}

#[test]
fn write_task_artifact_supports_transcription_generate() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let output = write_task_artifact(
        temp_dir.path(),
        TaskType::TranscriptionGenerate,
        None,
        serde_json::json!({
            "source_path": "/tmp/interview.wav",
            "output_format": "srt"
        }),
        TaskWriteOptions::default(),
    )
    .expect("write transcription task");

    assert!(output
        .path
        .starts_with(".lime/tasks/transcription_generate/"));
    assert_eq!(output.task_type, "transcription_generate");
    assert_eq!(output.task_family, "document");
    assert_eq!(output.status, "pending_submit");
    assert_eq!(
        output.ui_hints.open_action.as_deref(),
        Some("open_task_panel")
    );
}

#[test]
fn write_task_artifact_supports_audio_generate() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let output = write_task_artifact(
        temp_dir.path(),
        TaskType::AudioGenerate,
        Some("配音".to_string()),
        serde_json::json!({
            "source_text": "这是需要配音的文案",
            "voice": "warm_narrator",
            "audio_output": {
                "status": "pending",
                "mime_type": "audio/mpeg"
            }
        }),
        TaskWriteOptions::default(),
    )
    .expect("write audio task");

    assert!(output.path.starts_with(".lime/tasks/audio_generate/"));
    assert_eq!(output.task_type, "audio_generate");
    assert_eq!(output.task_family, "audio");
    assert_eq!(output.status, "pending_submit");
    assert_eq!(output.normalized_status, "pending");
    assert_eq!(
        output.ui_hints.open_action.as_deref(),
        Some("open_audio_player")
    );
    assert_eq!(
        output.record.payload.pointer("/audio_output/status"),
        Some(&serde_json::json!("pending"))
    );
}

#[test]
fn write_task_artifact_reuses_idempotent_record() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let first = write_task_artifact(
        temp_dir.path(),
        TaskType::BroadcastGenerate,
        Some("播客".to_string()),
        serde_json::json!({ "content": "demo" }),
        TaskWriteOptions {
            idempotency_key: Some("broadcast-1"),
            ..TaskWriteOptions::default()
        },
    )
    .expect("write first");

    let second = write_task_artifact(
        temp_dir.path(),
        TaskType::BroadcastGenerate,
        Some("播客".to_string()),
        serde_json::json!({ "content": "demo" }),
        TaskWriteOptions {
            idempotency_key: Some("broadcast-1"),
            ..TaskWriteOptions::default()
        },
    )
    .expect("write second");

    assert_eq!(first.task_id, second.task_id);
    assert!(second.reused_existing);
}

#[test]
fn write_task_artifact_does_not_reuse_cancelled_idempotent_record() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let cancelled = write_task_artifact(
        temp_dir.path(),
        TaskType::BroadcastGenerate,
        Some("播客".to_string()),
        serde_json::json!({ "content": "demo" }),
        TaskWriteOptions {
            status: Some("cancelled".to_string()),
            idempotency_key: Some("broadcast-1"),
            ..TaskWriteOptions::default()
        },
    )
    .expect("write cancelled");

    let created_again = write_task_artifact(
        temp_dir.path(),
        TaskType::BroadcastGenerate,
        Some("播客".to_string()),
        serde_json::json!({ "content": "demo" }),
        TaskWriteOptions {
            idempotency_key: Some("broadcast-1"),
            ..TaskWriteOptions::default()
        },
    )
    .expect("write again");

    assert_ne!(cancelled.task_id, created_again.task_id);
    assert!(!created_again.reused_existing);
    assert_eq!(created_again.normalized_status, "pending");
}

#[test]
fn list_task_outputs_filters_by_normalized_status() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let _pending = write_task_artifact(
        temp_dir.path(),
        TaskType::Typesetting,
        None,
        serde_json::json!({ "content": "demo" }),
        TaskWriteOptions::default(),
    )
    .expect("write pending");
    let failed = write_task_artifact(
        temp_dir.path(),
        TaskType::UrlParse,
        None,
        serde_json::json!({ "url": "https://example.com" }),
        TaskWriteOptions {
            status: Some("failed".to_string()),
            ..TaskWriteOptions::default()
        },
    )
    .expect("write failed");

    let items = list_task_outputs(temp_dir.path(), None, Some("failed"), None, None, Some(10))
        .expect("list tasks");

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].task_id, failed.task_id);
}

#[test]
fn retry_task_artifact_appends_attempt_to_same_task() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let failed = write_task_artifact(
        temp_dir.path(),
        TaskType::ModalResourceSearch,
        None,
        serde_json::json!({ "query": "城市" }),
        TaskWriteOptions {
            status: Some("failed".to_string()),
            idempotency_key: Some("search-1"),
            ..TaskWriteOptions::default()
        },
    )
    .expect("write failed");

    let retried = retry_task_artifact(temp_dir.path(), &failed.task_id, None).expect("retry task");

    assert_eq!(failed.task_id, retried.task_id);
    assert_ne!(failed.current_attempt_id, retried.current_attempt_id);
    assert_eq!(retried.status, "pending_submit");
    assert_eq!(retried.record.retry_count, 1);
    assert_eq!(retried.record.source_task_id, None);
    assert_eq!(retried.record.idempotency_key.as_deref(), Some("search-1"));
    assert_eq!(retried.record.attempts.len(), 2);
    assert_eq!(
        retried.record.relationships.derived_from_attempt_id,
        failed.current_attempt_id
    );
}

#[test]
fn load_task_output_upgrades_legacy_error_and_attempt_history() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let legacy_path = temp_dir
        .path()
        .join(".lime/tasks/image_generate/legacy-task.json");
    std::fs::create_dir_all(
        legacy_path
            .parent()
            .expect("legacy task parent should exist"),
    )
    .expect("create legacy task dir");
    std::fs::write(
        &legacy_path,
        serde_json::json!({
            "task_id": "legacy-task",
            "task_type": "image_generate",
            "payload": {
                "prompt": "未来实验室"
            },
            "status": "failed",
            "created_at": "2026-04-03T00:00:00Z",
            "retry_count": 2,
            "last_error": "provider timeout"
        })
        .to_string(),
    )
    .expect("write legacy task");

    let output = load_task_output(temp_dir.path(), "legacy-task", None).expect("load legacy task");

    assert_eq!(output.task_family, "image");
    assert_eq!(output.attempt_count, 1);
    assert_eq!(output.record.retry_count, 0);
    assert_eq!(output.record.attempts.len(), 1);
    assert_eq!(
        output
            .record
            .last_error
            .as_ref()
            .map(|value| value.code.as_str()),
        Some("legacy_error")
    );
    assert_eq!(
        output
            .record
            .attempts
            .first()
            .and_then(|attempt| attempt.error.as_ref())
            .map(|value| value.message.as_str()),
        Some("provider timeout")
    );
}

#[test]
fn parse_media_task_output_accepts_serialized_success_payload() {
    let payload = MediaTaskOutput {
        success: true,
        task_id: "task-1".to_string(),
        task_type: "image_generate".to_string(),
        task_family: "image".to_string(),
        status: "pending_submit".to_string(),
        normalized_status: "pending".to_string(),
        current_attempt_id: Some("attempt-1".to_string()),
        attempt_count: 1,
        last_error: None,
        progress: TaskProgress::default(),
        ui_hints: TaskUiHints::default(),
        path: ".lime/tasks/image_generate/demo.json".to_string(),
        absolute_path: "/tmp/demo.json".to_string(),
        artifact_path: ".lime/tasks/image_generate/demo.json".to_string(),
        absolute_artifact_path: "/tmp/demo.json".to_string(),
        reused_existing: false,
        idempotency_key: None,
        record: MediaTaskArtifactRecord {
            task_id: "task-1".to_string(),
            task_type: "image_generate".to_string(),
            task_family: "image".to_string(),
            title: None,
            summary: Some("image_generate 任务".to_string()),
            payload: serde_json::json!({ "prompt": "demo" }),
            status: "pending_submit".to_string(),
            normalized_status: "pending".to_string(),
            created_at: "2026-04-03T00:00:00Z".to_string(),
            updated_at: None,
            submitted_at: None,
            started_at: None,
            completed_at: None,
            cancelled_at: None,
            idempotency_key: None,
            retry_count: 0,
            source_task_id: None,
            result: None,
            last_error: None,
            current_attempt_id: Some("attempt-1".to_string()),
            attempts: vec![TaskAttemptRecord {
                attempt_id: "attempt-1".to_string(),
                attempt_index: 1,
                status: "pending_submit".to_string(),
                input_snapshot: serde_json::json!({ "prompt": "demo" }),
                ..TaskAttemptRecord::default()
            }],
            relationships: TaskRelationships::default(),
            progress: TaskProgress::default(),
            ui_hints: TaskUiHints::default(),
        },
    };
    let serialized = serde_json::to_string(&payload).expect("serialize");

    let parsed = parse_media_task_output(&serialized).expect("parse success payload");
    assert_eq!(parsed.task_id, "task-1");
    assert_eq!(
        parsed.artifact_paths(),
        vec![".lime/tasks/image_generate/demo.json".to_string()]
    );
}
