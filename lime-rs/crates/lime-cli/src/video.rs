use std::path::Path;

use lime_core::config::load_config;
use lime_media_runtime::{
    execute_video_generation_task as execute_video_task_runtime, MediaRuntimeError,
    TaskArtifactPatch, TaskErrorRecord, TaskProgress, TaskType, VideoGenerationRunnerConfig,
    VIDEO_TASK_RUNNER_WORKER_ID,
};
use serde_json::{json, Value};

use super::{
    read_env_port, read_non_empty_env, resolve_workspace_root, task_write_options,
    SharedTaskWriteArgs, VideoGenerateArgs,
};

pub(super) fn create_video_task(args: VideoGenerateArgs) -> Result<Value, MediaRuntimeError> {
    generate_video_task(args)
}

pub(super) fn generate_video_task(args: VideoGenerateArgs) -> Result<Value, MediaRuntimeError> {
    let runner_config = match resolve_cli_video_generation_runner_config() {
        Ok(config) => config,
        Err(error_message) => {
            let workspace_root = resolve_workspace_root(args.output.workspace.clone())?;
            let created = create_video_task_artifact(&workspace_root, &args)?;
            return mark_cli_video_task_failed(&workspace_root, &created.task_id, error_message);
        }
    };
    generate_video_task_with_runner_config(args, runner_config)
}

fn generate_video_task_with_runner_config(
    args: VideoGenerateArgs,
    runner_config: VideoGenerationRunnerConfig,
) -> Result<Value, MediaRuntimeError> {
    let workspace_root = resolve_workspace_root(args.output.workspace.clone())?;
    let created = create_video_task_artifact(&workspace_root, &args)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| MediaRuntimeError::Io(format!("初始化视频任务运行时失败: {error}")))?;
    let output = runtime.block_on(execute_video_task_runtime(
        &workspace_root,
        &created.task_id,
        &runner_config,
    ))?;
    Ok(json!(output))
}

fn create_video_task_artifact(
    workspace_root: &Path,
    args: &VideoGenerateArgs,
) -> Result<lime_media_runtime::MediaTaskOutput, MediaRuntimeError> {
    lime_media_runtime::write_task_artifact(
        workspace_root,
        TaskType::VideoGenerate,
        args.title.clone(),
        build_video_task_payload(args),
        task_write_options(&args.output),
    )
}

fn build_video_task_payload(args: &VideoGenerateArgs) -> Value {
    json!({
        "prompt": args.prompt,
        "projectId": args.project_id,
        "providerId": args.provider_id,
        "model": args.model,
        "aspectRatio": args.aspect_ratio,
        "resolution": args.resolution,
        "duration": args.duration,
        "imageUrl": args.image_url,
        "endImageUrl": args.end_image_url,
        "seed": args.seed,
        "generateAudio": args.generate_audio,
        "cameraFixed": args.camera_fixed,
    })
}

fn resolve_cli_video_generation_runner_config() -> Result<VideoGenerationRunnerConfig, String> {
    let endpoint_override = read_non_empty_env("LIME_MEDIA_VIDEO_ENDPOINT");
    let api_key_override = read_non_empty_env("LIME_MEDIA_VIDEO_API_KEY")
        .or_else(|| read_non_empty_env("LIME_SERVER_API_KEY"));
    let host_override = read_non_empty_env("LIME_SERVER_HOST");
    let port_override = read_env_port("LIME_SERVER_PORT")?;

    let (loaded_config, config_load_error) = match load_config() {
        Ok(config) => (Some(config), None),
        Err(error) => (None, Some(error.to_string())),
    };

    let host = host_override
        .or_else(|| {
            loaded_config
                .as_ref()
                .map(|config| config.server.host.clone())
        })
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port = port_override
        .or_else(|| loaded_config.as_ref().map(|config| config.server.port))
        .unwrap_or(9000);
    let api_key = api_key_override
        .or_else(|| {
            loaded_config
                .as_ref()
                .map(|config| config.server.api_key.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .ok_or_else(|| match config_load_error {
            Some(error) => format!("Lime 本地视频服务未配置 API Key，且加载本地配置失败: {error}"),
            None => "Lime 本地视频服务未配置 API Key".to_string(),
        })?;

    Ok(VideoGenerationRunnerConfig::fal(
        endpoint_override.unwrap_or_else(|| build_video_generation_endpoint(&host, port)),
        api_key,
        "Authorization".to_string(),
        Some("Bearer".to_string()),
    ))
}

fn normalize_video_generation_service_host(host: &str) -> String {
    let trimmed = host.trim();
    if trimmed.is_empty() || trimmed == "0.0.0.0" || trimmed == "::" {
        return "127.0.0.1".to_string();
    }
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        return trimmed.to_string();
    }
    if trimmed.contains(':') {
        return format!("[{trimmed}]");
    }
    trimmed.to_string()
}

fn build_video_generation_endpoint(host: &str, port: u16) -> String {
    format!(
        "http://{}:{port}/v1/videos/generations",
        normalize_video_generation_service_host(host)
    )
}

fn build_video_task_progress(
    phase: &str,
    message: impl Into<String>,
    percent: Option<u32>,
) -> TaskProgress {
    TaskProgress {
        phase: Some(phase.to_string()),
        percent,
        message: Some(message.into()),
        preview_slots: Vec::new(),
    }
}

fn build_video_task_error(
    code: &str,
    message: impl Into<String>,
    retryable: bool,
    stage: &str,
) -> TaskErrorRecord {
    TaskErrorRecord {
        code: code.to_string(),
        message: message.into(),
        retryable,
        stage: Some(stage.to_string()),
        provider_code: None,
        occurred_at: None,
    }
}

fn mark_cli_video_task_failed(
    workspace_root: &Path,
    task_id: &str,
    message: impl Into<String>,
) -> Result<Value, MediaRuntimeError> {
    let task_error =
        build_video_task_error("video_worker_unavailable", message, false, "bootstrap");
    let output = lime_media_runtime::patch_task_artifact(
        workspace_root,
        task_id,
        None,
        TaskArtifactPatch {
            status: Some("failed".to_string()),
            last_error: Some(Some(task_error.clone())),
            progress: Some(build_video_task_progress(
                "failed",
                task_error.message.clone(),
                None,
            )),
            current_attempt_worker_id: Some(Some(VIDEO_TASK_RUNNER_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )?;
    Ok(json!(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn build_video_generation_endpoint_normalizes_localhost() {
        assert_eq!(
            build_video_generation_endpoint("0.0.0.0", 9000),
            "http://127.0.0.1:9000/v1/videos/generations"
        );
        assert_eq!(
            build_video_generation_endpoint("::1", 9000),
            "http://[::1]:9000/v1/videos/generations"
        );
    }

    #[test]
    fn generate_video_task_with_runner_config_executes_created_task() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let captured_request = Arc::new(Mutex::new(String::new()));
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind video api");
        let address = listener.local_addr().expect("resolve video api address");
        let captured_request_for_server = Arc::clone(&captured_request);
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept request");
            let mut buffer = [0_u8; 8192];
            let read = stream.read(&mut buffer).expect("read request");
            let request = String::from_utf8_lossy(&buffer[..read]).to_string();
            *captured_request_for_server
                .lock()
                .expect("lock captured request") = request;
            let body = r#"{"data":[{"url":"https://cdn.example.test/video.mp4","duration":6}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
        });

        let output = generate_video_task_with_runner_config(
            VideoGenerateArgs {
                prompt: "生成一段青柠实验室短视频".to_string(),
                title: Some("短视频".to_string()),
                project_id: Some("project-1".to_string()),
                provider_id: Some("veo-provider".to_string()),
                model: Some("veo-3".to_string()),
                aspect_ratio: Some("16:9".to_string()),
                resolution: Some("1080p".to_string()),
                duration: Some(6),
                image_url: Some("https://example.test/start.png".to_string()),
                end_image_url: None,
                seed: Some(42),
                generate_audio: Some(true),
                camera_fixed: Some(false),
                output: SharedTaskWriteArgs {
                    workspace: Some(temp_dir.path().to_path_buf()),
                    output: None,
                    artifact_dir: None,
                    idempotency_key: None,
                    json: true,
                },
            },
            VideoGenerationRunnerConfig::fal(
                format!("http://{address}/v1/videos/generations"),
                "test-key".to_string(),
                "Authorization".to_string(),
                Some("Bearer".to_string()),
            ),
        )
        .expect("generate video task");

        server.join().expect("join video api");
        assert_eq!(output["normalized_status"], "succeeded");
        assert_eq!(
            output["record"]["result"]["video"]["url"],
            "https://cdn.example.test/video.mp4"
        );
        assert_eq!(
            output["record"]["attempts"]
                .as_array()
                .and_then(|attempts| attempts.last())
                .and_then(|attempt| attempt.get("worker_id"))
                .and_then(Value::as_str),
            Some(VIDEO_TASK_RUNNER_WORKER_ID)
        );

        let request = captured_request.lock().expect("lock request").clone();
        assert!(request.starts_with("POST /v1/videos/generations HTTP/1.1"));
        assert!(request.contains("authorization: Bearer test-key"));
        assert!(request.contains("x-provider-id: veo-provider"));
        assert!(request.contains("\"prompt\":\"生成一段青柠实验室短视频\""));
        assert!(request.contains("\"generate_audio\":true"));
    }
}
