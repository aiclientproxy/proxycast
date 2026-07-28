use super::*;

#[tokio::test]
async fn execute_video_generation_task_should_advance_task_file_to_succeeded() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let captured_auth = Arc::new(Mutex::new(None::<String>));
    let captured_provider_id = Arc::new(Mutex::new(None::<String>));
    let captured_body = Arc::new(Mutex::new(None::<Value>));
    let captured_updates = Arc::new(Mutex::new(Vec::<String>::new()));
    let created = write_task_artifact(
        temp_dir.path(),
        TaskType::VideoGenerate,
        Some("短视频".to_string()),
        json!({
            "prompt": "生成一段青柠实验室短视频",
            "provider_id": "veo-provider",
            "model": "veo-3",
            "aspect_ratio": "16:9",
            "resolution": "1080p",
            "duration": 8,
            "image_url": "https://example.test/start.png",
            "end_image_url": "https://example.test/end.png",
            "seed": 42,
            "generate_audio": true,
            "camera_fixed": false
        }),
        TaskWriteOptions {
            status: Some("pending_submit".to_string()),
            ..TaskWriteOptions::default()
        },
    )
    .expect("create video task");

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind video api");
    let address = listener.local_addr().expect("resolve address");
    let captured_auth_for_server = Arc::clone(&captured_auth);
    let captured_provider_id_for_server = Arc::clone(&captured_provider_id);
    let captured_body_for_server = Arc::clone(&captured_body);
    let server = tokio::spawn(async move {
        let app = Router::new().route(
            "/v1/videos/generations",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let captured_auth = Arc::clone(&captured_auth_for_server);
                let captured_provider_id = Arc::clone(&captured_provider_id_for_server);
                let captured_body = Arc::clone(&captured_body_for_server);
                async move {
                    *captured_auth.lock().expect("lock auth") = headers
                        .get("authorization")
                        .and_then(|value| value.to_str().ok())
                        .map(ToOwned::to_owned);
                    *captured_provider_id.lock().expect("lock provider id") = headers
                        .get("x-provider-id")
                        .and_then(|value| value.to_str().ok())
                        .map(ToOwned::to_owned);
                    *captured_body.lock().expect("lock body") = Some(body);
                    (
                        StatusCode::OK,
                        Json(json!({
                            "id": "video-job-1",
                            "data": [
                                {
                                    "id": "generated-video-1",
                                    "url": "https://cdn.example.test/generated.mp4",
                                    "mime_type": "video/mp4",
                                    "duration": 8
                                }
                            ]
                        })),
                    )
                }
            }),
        );
        axum::serve(listener, app).await.expect("serve video api");
    });

    let updates_for_hook = Arc::clone(&captured_updates);
    let result = execute_video_generation_task_with_hook(
        temp_dir.path(),
        &created.task_id,
        &VideoGenerationRunnerConfig::fal(
            format!("http://{address}/v1/videos/generations"),
            "test-key".to_string(),
            "Authorization".to_string(),
            Some("Key".to_string()),
        ),
        move |output| {
            updates_for_hook
                .lock()
                .expect("lock updates")
                .push(output.normalized_status.clone());
        },
    )
    .await
    .expect("execute video task");

    assert_eq!(result.normalized_status, "succeeded");
    assert_eq!(
        result
            .record
            .result
            .as_ref()
            .and_then(|value| value.pointer("/video/url"))
            .and_then(Value::as_str),
        Some("https://cdn.example.test/generated.mp4")
    );
    assert_eq!(
        result
            .record
            .attempts
            .last()
            .and_then(|attempt| attempt.worker_id.as_deref()),
        Some(VIDEO_TASK_RUNNER_WORKER_ID)
    );
    assert_eq!(
        captured_auth.lock().expect("lock auth").as_deref(),
        Some("Key test-key")
    );
    assert_eq!(
        captured_provider_id
            .lock()
            .expect("lock provider id")
            .as_deref(),
        Some("veo-provider")
    );
    let body = captured_body
        .lock()
        .expect("lock body")
        .clone()
        .expect("captured body");
    assert_eq!(
        body.pointer("/prompt"),
        Some(&json!("生成一段青柠实验室短视频"))
    );
    assert_eq!(body.pointer("/model"), Some(&json!("veo-3")));
    assert_eq!(body.pointer("/aspect_ratio"), Some(&json!("16:9")));
    assert_eq!(body.pointer("/duration"), Some(&json!(8)));
    assert_eq!(body.pointer("/generate_audio"), Some(&json!(true)));
    assert_eq!(body.pointer("/camera_fixed"), Some(&json!(false)));
    assert_eq!(body.pointer("/user"), Some(&json!(created.task_id.clone())));
    assert_eq!(
        captured_updates.lock().expect("lock updates").as_slice(),
        ["queued", "running", "succeeded"]
    );
    assert_eq!(
        result
            .record
            .payload
            .pointer("/llm_events/1/type")
            .and_then(Value::as_str),
        Some("turn.completed")
    );
    assert_eq!(
        result
            .record
            .payload
            .pointer("/provider_diagnostics/taskFamily")
            .and_then(Value::as_str),
        Some("video_generation")
    );
    assert_eq!(
        result
            .record
            .payload
            .pointer("/provider_diagnostics/modelId")
            .and_then(Value::as_str),
        None
    );
    assert_eq!(
        result
            .record
            .payload
            .pointer("/provider_diagnostics/transport")
            .and_then(Value::as_str),
        Some("provider_http")
    );

    server.abort();
}

#[tokio::test]
async fn xai_video_task_persists_request_id_and_polls_to_done() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let captured_auth = Arc::new(Mutex::new(Vec::<String>::new()));
    let captured_body = Arc::new(Mutex::new(None::<Value>));
    let created = write_task_artifact(
        temp_dir.path(),
        TaskType::VideoGenerate,
        Some("Grok 视频".to_string()),
        json!({
            "prompt": "让青柠切片在白色背景上缓慢旋转",
            "provider_id": "xai",
            "model": "grok-imagine-video",
            "aspect_ratio": "16:9",
            "resolution": "720p",
            "duration": 6,
            "image_url": "https://example.test/lime.png"
        }),
        TaskWriteOptions {
            status: Some("pending_submit".to_string()),
            ..TaskWriteOptions::default()
        },
    )
    .expect("create xAI video task");

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind xAI video API");
    let address = listener.local_addr().expect("resolve xAI address");
    let start_auth = Arc::clone(&captured_auth);
    let start_body = Arc::clone(&captured_body);
    let poll_auth = Arc::clone(&captured_auth);
    let server = tokio::spawn(async move {
        let app = Router::new()
            .route(
                "/v1/videos/generations",
                post(move |headers: HeaderMap, Json(body): Json<Value>| {
                    let captured_auth = Arc::clone(&start_auth);
                    let captured_body = Arc::clone(&start_body);
                    async move {
                        captured_auth.lock().expect("start auth").push(
                            headers
                                .get("authorization")
                                .and_then(|value| value.to_str().ok())
                                .unwrap_or_default()
                                .to_string(),
                        );
                        *captured_body.lock().expect("start body") = Some(body);
                        (
                            StatusCode::OK,
                            Json(json!({ "request_id": "xai-request-1" })),
                        )
                    }
                }),
            )
            .route(
                "/v1/videos/xai-request-1",
                get(move |headers: HeaderMap| {
                    let captured_auth = Arc::clone(&poll_auth);
                    async move {
                        captured_auth.lock().expect("poll auth").push(
                            headers
                                .get("authorization")
                                .and_then(|value| value.to_str().ok())
                                .unwrap_or_default()
                                .to_string(),
                        );
                        (
                            StatusCode::OK,
                            Json(json!({
                                "status": "done",
                                "video": {
                                    "url": "https://cdn.example.test/xai-video.mp4"
                                }
                            })),
                        )
                    }
                }),
            );
        axum::serve(listener, app)
            .await
            .expect("serve xAI video API");
    });

    let mut config = VideoGenerationRunnerConfig::xai(
        format!("http://{address}/v1/videos/generations"),
        "xai-key".to_string(),
        "Authorization".to_string(),
        Some("Bearer".to_string()),
    );
    config.poll_interval = Duration::from_millis(1);
    config.overall_timeout = Duration::from_secs(1);
    let result = execute_video_generation_task(temp_dir.path(), &created.task_id, &config)
        .await
        .expect("execute xAI video task");

    assert_eq!(result.normalized_status, "succeeded");
    assert_eq!(
        result.record.payload.pointer("/provider_task/request_id"),
        Some(&json!("xai-request-1"))
    );
    assert_eq!(
        result
            .record
            .result
            .as_ref()
            .and_then(|value| value.pointer("/provider_request_id")),
        Some(&json!("xai-request-1"))
    );
    assert_eq!(
        result
            .record
            .result
            .as_ref()
            .and_then(|value| value.pointer("/video/url")),
        Some(&json!("https://cdn.example.test/xai-video.mp4"))
    );
    assert_eq!(
        captured_auth.lock().expect("captured auth").as_slice(),
        ["Bearer xai-key", "Bearer xai-key"]
    );
    let body = captured_body
        .lock()
        .expect("captured body")
        .clone()
        .expect("xAI start body");
    assert_eq!(body["model"], "grok-imagine-video");
    assert_eq!(body["image"]["url"], "https://example.test/lime.png");
    assert_eq!(body["duration"], 6);

    server.abort();
}

#[tokio::test]
async fn xai_video_task_resumes_polling_without_restarting_generation() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let start_count = Arc::new(AtomicUsize::new(0));
    let poll_count = Arc::new(AtomicUsize::new(0));
    let created = write_task_artifact(
        temp_dir.path(),
        TaskType::VideoGenerate,
        Some("恢复 Grok 视频".to_string()),
        json!({
            "prompt": "恢复既有视频任务",
            "provider_id": "xai",
            "model": "grok-imagine-video",
            "provider_task": {
                "protocol": "xai_video",
                "request_id": "xai-resume-1",
                "status": "pending"
            }
        }),
        TaskWriteOptions {
            status: Some("running".to_string()),
            ..TaskWriteOptions::default()
        },
    )
    .expect("create resumable xAI task");

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind resume video API");
    let address = listener.local_addr().expect("resolve resume address");
    let starts = Arc::clone(&start_count);
    let polls = Arc::clone(&poll_count);
    let server = tokio::spawn(async move {
        let app = Router::new()
            .route(
                "/v1/videos/generations",
                post(move || {
                    let starts = Arc::clone(&starts);
                    async move {
                        starts.fetch_add(1, Ordering::SeqCst);
                        (StatusCode::OK, Json(json!({ "request_id": "unexpected" })))
                    }
                }),
            )
            .route(
                "/v1/videos/xai-resume-1",
                get(move || {
                    let polls = Arc::clone(&polls);
                    async move {
                        polls.fetch_add(1, Ordering::SeqCst);
                        (
                            StatusCode::OK,
                            Json(json!({
                                "status": "done",
                                "video": { "url": "https://cdn.example.test/resumed.mp4" }
                            })),
                        )
                    }
                }),
            );
        axum::serve(listener, app)
            .await
            .expect("serve resume video API");
    });

    let mut config = VideoGenerationRunnerConfig::xai(
        format!("http://{address}/v1/videos/generations"),
        "xai-key".to_string(),
        "Authorization".to_string(),
        Some("Bearer".to_string()),
    );
    config.poll_interval = Duration::from_millis(1);
    config.overall_timeout = Duration::from_secs(1);
    let result = execute_video_generation_task(temp_dir.path(), &created.task_id, &config)
        .await
        .expect("resume xAI video task");

    assert_eq!(result.normalized_status, "succeeded");
    assert_eq!(start_count.load(Ordering::SeqCst), 0);
    assert_eq!(poll_count.load(Ordering::SeqCst), 1);
    assert_eq!(
        result
            .record
            .result
            .as_ref()
            .and_then(|value| value.pointer("/video/url")),
        Some(&json!("https://cdn.example.test/resumed.mp4"))
    );

    server.abort();
}
