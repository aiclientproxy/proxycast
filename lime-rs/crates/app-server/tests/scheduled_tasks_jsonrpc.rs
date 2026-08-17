use std::sync::{Arc, Mutex};

use app_server::{
    ActionRespondRequest, AppServer, CancelExecutionRequest, ExecutionBackend, ExecutionRequest,
    LocalAppDataSource, MockBackend, ProjectionStore, RuntimeCore, RuntimeCoreError, RuntimeEvent,
    RuntimeEventSink,
};
use app_server_protocol::{
    protocol::v2::ThreadSettings, AgentSessionReadParams, AgentSessionStartParams, JsonRpcMessage,
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_SCHEDULED_TASK_CHANGED,
    METHOD_SCHEDULED_TASK_CREATE, METHOD_SCHEDULED_TASK_DELETE, METHOD_SCHEDULED_TASK_ENABLED_SET,
    METHOD_SCHEDULED_TASK_LIST, METHOD_SCHEDULED_TASK_READ, METHOD_SCHEDULED_TASK_RUN_LIST,
    METHOD_SCHEDULED_TASK_RUN_START, METHOD_SCHEDULED_TASK_RUN_UPDATED,
    METHOD_SCHEDULED_TASK_SCHEDULE_PREVIEW, METHOD_SCHEDULED_TASK_UPDATE, PROTOCOL_VERSION,
};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::Utc;
use lime_core::database::dao::api_key_provider::{
    ApiKeyEntry, ApiKeyProvider, ApiKeyProviderDao, ApiProviderType, ProviderGroup,
};
use lime_core::database::schema::create_tables;
use lime_core::models::model_registry::{
    ModelCapabilities, ModelModality, ModelRuntimeFeature, ModelTaskFamily,
    ProviderModelCapability, ProviderModelConfig,
};
use lime_services::api_key_provider_service::ApiKeyProviderService;
use rusqlite::Connection;
use serde_json::{json, Value};
use tempfile::TempDir;

struct ScheduledTaskAppServer {
    _temp: TempDir,
    runtime: RuntimeCore,
    server: AppServer,
}

async fn scheduled_task_app_server() -> ScheduledTaskAppServer {
    scheduled_task_app_server_with_backend(
        Arc::new(MockBackend),
        &[("scheduled-test-provider", "gpt-5")],
    )
    .await
}

async fn scheduled_task_app_server_with_backend(
    backend: Arc<dyn ExecutionBackend>,
    providers: &[(&str, &str)],
) -> ScheduledTaskAppServer {
    let temp = TempDir::new().expect("create scheduled task fixture temp dir");
    let connection = Connection::open_in_memory().expect("open scheduled task product db");
    create_tables(&connection).expect("create scheduled task product schema");
    for (index, (provider, model)) in providers.iter().enumerate() {
        insert_chat_provider(&connection, provider, model, index as i32);
    }
    let app_data_source = LocalAppDataSource::initialize_with_roots(
        Arc::new(Mutex::new(connection)),
        temp.path(),
        temp.path().join("app-server"),
    )
    .await
    .expect("scheduled task app data source");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("scheduled task projection store"),
    );
    let runtime = RuntimeCore::with_backend(backend)
        .with_app_data_source(Arc::new(app_data_source))
        .with_projection_store(projection_store);

    ScheduledTaskAppServer {
        _temp: temp,
        runtime: runtime.clone(),
        server: AppServer::with_runtime(runtime),
    }
}

#[derive(Default)]
struct RouteAwareBackend {
    preflight_routes: Mutex<Vec<(String, String)>>,
    turn_routes: Mutex<Vec<(String, String)>>,
}

impl RouteAwareBackend {
    fn preflight_routes(&self) -> Vec<(String, String)> {
        self.preflight_routes
            .lock()
            .expect("route preflight mutex poisoned")
            .clone()
    }

    fn turn_routes(&self) -> Vec<(String, String)> {
        self.turn_routes
            .lock()
            .expect("route turn mutex poisoned")
            .clone()
    }
}

#[async_trait::async_trait]
impl ExecutionBackend for RouteAwareBackend {
    fn requires_provider_selection(&self) -> bool {
        true
    }

    async fn preflight_thread_settings(
        &self,
        _session: &app_server_protocol::AgentSession,
        settings: &ThreadSettings,
    ) -> Result<(), RuntimeCoreError> {
        self.preflight_routes
            .lock()
            .expect("route preflight mutex poisoned")
            .push((settings.model_provider.clone(), settings.model.clone()));
        Ok(())
    }

    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let provider = request.provider_preference().ok_or_else(|| {
            RuntimeCoreError::Backend("route-aware backend requires provider".to_string())
        })?;
        let model = request.model_preference().ok_or_else(|| {
            RuntimeCoreError::Backend("route-aware backend requires model".to_string())
        })?;
        self.turn_routes
            .lock()
            .expect("route turn mutex poisoned")
            .push((provider.to_string(), model.to_string()));
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(RuntimeEvent::new("turn.completed", json!({})))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

#[tokio::test]
async fn scheduled_task_crud_preview_run_and_history_use_public_jsonrpc() {
    let app = scheduled_task_app_server().await;
    initialize(&app.server).await;

    let preview = request_ok(
        &app.server,
        2,
        METHOD_SCHEDULED_TASK_SCHEDULE_PREVIEW,
        json!({
            "schedule": {
                "type": "weekly",
                "days": ["MO", "FR"],
                "time": "08:30",
                "timezone": "Asia/Shanghai"
            }
        }),
    )
    .await;
    assert_eq!(
        preview["result"]["nextRunAt"]
            .as_array()
            .expect("scheduled preview occurrences")
            .len(),
        5
    );
    assert_eq!(preview["result"]["warnings"], json!([]));

    let created = create_task(&app.server, 3, "每日项目简报").await;
    let task_id = created["result"]["task"]["id"]
        .as_str()
        .expect("created scheduled task id")
        .to_string();
    let revision = created["result"]["task"]["updatedAt"]
        .as_str()
        .expect("created scheduled task revision")
        .to_string();
    assert_eq!(created["result"]["task"]["enabled"], true);
    assert_eq!(
        created["result"]["task"]["execution"]["threadMode"],
        "new_thread"
    );
    assert_eq!(
        created["result"]["task"]["execution"]["requestMetadata"]["harness"]["service_skill"]["id"],
        "daily-brief"
    );

    let read = request_ok(
        &app.server,
        4,
        METHOD_SCHEDULED_TASK_READ,
        json!({"id": task_id}),
    )
    .await;
    assert_eq!(read["result"]["task"]["title"], "每日项目简报");
    assert_eq!(
        read["result"]["task"]["execution"]["requestMetadata"]["harness"]["service_skill"]["id"],
        "daily-brief"
    );

    let listed = request_ok(
        &app.server,
        5,
        METHOD_SCHEDULED_TASK_LIST,
        json!({"query": "项目", "enabled": true, "limit": 20}),
    )
    .await;
    assert_eq!(listed["result"]["items"].as_array().unwrap().len(), 1);
    assert_eq!(listed["result"]["items"][0]["id"], task_id);

    let updated = request_ok(
        &app.server,
        6,
        METHOD_SCHEDULED_TASK_UPDATE,
        json!({
            "id": task_id,
            "task": {
                "prompt": "整理项目进展、阻塞项和下一步行动",
                "notificationPolicy": "all_runs",
                "revision": revision
            }
        }),
    )
    .await;
    assert_eq!(
        updated["result"]["task"]["prompt"],
        "整理项目进展、阻塞项和下一步行动"
    );
    assert_eq!(updated["result"]["task"]["notificationPolicy"], "all_runs");

    let paused = request_ok(
        &app.server,
        7,
        METHOD_SCHEDULED_TASK_ENABLED_SET,
        json!({"id": task_id, "enabled": false}),
    )
    .await;
    assert_eq!(paused["result"]["task"]["enabled"], false);
    assert_eq!(paused["result"]["task"]["nextRunAt"], Value::Null);

    let resumed = request_ok(
        &app.server,
        8,
        METHOD_SCHEDULED_TASK_ENABLED_SET,
        json!({"id": task_id, "enabled": true}),
    )
    .await;
    assert_eq!(resumed["result"]["task"]["enabled"], true);

    let started = request_ok(
        &app.server,
        9,
        METHOD_SCHEDULED_TASK_RUN_START,
        json!({"id": task_id}),
    )
    .await;
    let run_id = started["result"]["run"]["id"]
        .as_str()
        .expect("started run id")
        .to_string();
    let session_id = started["result"]["run"]["sessionId"]
        .as_str()
        .expect("started run session id")
        .to_string();
    assert!(session_id.starts_with("scheduled-session-"));
    assert_eq!(started["result"]["run"]["taskId"], task_id);
    assert!(started["result"]["run"]["startedAt"].is_string());

    let runs = request_ok(
        &app.server,
        10,
        METHOD_SCHEDULED_TASK_RUN_LIST,
        json!({"taskId": task_id, "limit": 20}),
    )
    .await;
    assert_eq!(runs["result"]["runs"].as_array().unwrap().len(), 1);
    assert_eq!(runs["result"]["runs"][0]["id"], run_id);
    assert_eq!(runs["result"]["runs"][0]["sessionId"], session_id);

    let read_after_run = request_ok(
        &app.server,
        11,
        METHOD_SCHEDULED_TASK_READ,
        json!({"id": task_id}),
    )
    .await;
    assert_eq!(
        read_after_run["result"]["task"]["lastRunSummary"]["id"],
        run_id
    );
    assert_eq!(
        read_after_run["result"]["task"]["lastRunSummary"]["sessionId"],
        session_id
    );
    let listed_after_run = request_ok(
        &app.server,
        12,
        METHOD_SCHEDULED_TASK_LIST,
        json!({"query": "项目"}),
    )
    .await;
    assert_eq!(
        listed_after_run["result"]["items"][0]["lastRun"]["id"],
        run_id
    );
    assert_eq!(listed_after_run["result"]["items"][0]["attention"], false);

    let deletable = create_task(&app.server, 13, "待删除任务").await;
    let deletable_id = deletable["result"]["task"]["id"]
        .as_str()
        .expect("deletable task id")
        .to_string();
    let deleted = request_ok(
        &app.server,
        14,
        METHOD_SCHEDULED_TASK_DELETE,
        json!({"id": deletable_id}),
    )
    .await;
    assert_eq!(deleted["result"]["deleted"], true);
    let missing = request_ok(
        &app.server,
        15,
        METHOD_SCHEDULED_TASK_READ,
        json!({"id": deletable_id}),
    )
    .await;
    assert_eq!(missing["result"]["task"], Value::Null);
}

#[tokio::test]
async fn scheduled_task_create_rejects_non_object_request_metadata_over_public_jsonrpc() {
    let app = scheduled_task_app_server().await;
    initialize(&app.server).await;

    let response = request_raw(
        &app.server,
        2,
        METHOD_SCHEDULED_TASK_CREATE,
        json!({
            "task": {
                "title": "非法 metadata 任务",
                "prompt": "不应创建",
                "schedule": {
                    "type": "daily",
                    "time": "09:00",
                    "timezone": "Asia/Shanghai"
                },
                "execution": {
                    "threadMode": "new_thread",
                    "projectId": "project-alpha",
                    "requestMetadata": "legacy"
                }
            }
        }),
    )
    .await;

    assert!(response.get("error").is_some());
    assert!(response.to_string().contains("requestMetadata 必须为对象"));
}

#[tokio::test]
async fn scheduled_task_mutations_and_terminal_run_publish_typed_notifications() {
    let app = scheduled_task_app_server().await;
    let mut outbound = app.server.subscribe_outbound_messages();
    initialize(&app.server).await;

    let created = create_task(&app.server, 2, "通知闭环").await;
    let task_id = created["result"]["task"]["id"]
        .as_str()
        .expect("created scheduled task id")
        .to_string();
    let revision = created["result"]["task"]["updatedAt"]
        .as_str()
        .expect("created scheduled task revision");
    let created_notification =
        next_outbound_notification(&mut outbound, METHOD_SCHEDULED_TASK_CHANGED).await;
    assert_eq!(created_notification["params"]["taskId"], task_id);
    assert_eq!(created_notification["params"]["change"], "created");

    request_ok(
        &app.server,
        3,
        METHOD_SCHEDULED_TASK_UPDATE,
        json!({
            "id": task_id,
            "task": {"title": "通知闭环更新", "revision": revision}
        }),
    )
    .await;
    let updated_notification =
        next_outbound_notification(&mut outbound, METHOD_SCHEDULED_TASK_CHANGED).await;
    assert_eq!(updated_notification["params"]["change"], "updated");

    request_ok(
        &app.server,
        4,
        METHOD_SCHEDULED_TASK_ENABLED_SET,
        json!({"id": task_id, "enabled": false}),
    )
    .await;
    let enabled_notification =
        next_outbound_notification(&mut outbound, METHOD_SCHEDULED_TASK_CHANGED).await;
    assert_eq!(enabled_notification["params"]["change"], "enabled");

    let started = request_ok(
        &app.server,
        5,
        METHOD_SCHEDULED_TASK_RUN_START,
        json!({"id": task_id}),
    )
    .await;
    let run_id = started["result"]["run"]["id"]
        .as_str()
        .expect("scheduled task run id")
        .to_string();
    let session_id = started["result"]["run"]["sessionId"]
        .as_str()
        .expect("scheduled task session id")
        .to_string();
    let turn_id = started["result"]["run"]["turnId"]
        .as_str()
        .expect("scheduled task turn id")
        .to_string();
    app.server
        .append_external_runtime_events(
            &session_id,
            Some(&turn_id),
            vec![RuntimeEvent::new(
                "turn.completed",
                json!({"source": "scheduled-task-jsonrpc-test"}),
            )],
        )
        .await
        .expect("append canonical scheduled task terminal event");
    let run_notification =
        next_outbound_notification(&mut outbound, METHOD_SCHEDULED_TASK_RUN_UPDATED).await;
    assert_eq!(run_notification["params"]["taskId"], task_id);
    assert_eq!(run_notification["params"]["runId"], run_id);
    assert_eq!(run_notification["params"]["status"], "success");
    assert_eq!(run_notification["params"]["attention"], false);
    assert_eq!(run_notification["params"]["notificationPolicy"], "failures");
    assert_eq!(run_notification["params"]["title"], "通知闭环更新");

    request_ok(
        &app.server,
        6,
        METHOD_SCHEDULED_TASK_DELETE,
        json!({"id": task_id}),
    )
    .await;
    let deleted_notification =
        next_outbound_notification(&mut outbound, METHOD_SCHEDULED_TASK_CHANGED).await;
    assert_eq!(deleted_notification["params"]["change"], "deleted");

    let history = request_ok(
        &app.server,
        7,
        METHOD_SCHEDULED_TASK_RUN_LIST,
        json!({"taskId": task_id, "limit": 10}),
    )
    .await;
    assert_eq!(history["result"]["runs"][0]["id"], run_id);
    assert_eq!(history["result"]["runs"][0]["status"], "success");
    let deleted_read = request_ok(
        &app.server,
        8,
        METHOD_SCHEDULED_TASK_READ,
        json!({"id": task_id}),
    )
    .await;
    assert_eq!(deleted_read["result"]["task"], Value::Null);
}

#[tokio::test]
async fn scheduled_task_new_thread_resolves_inherited_and_explicit_model_routes() {
    let backend = Arc::new(RouteAwareBackend::default());
    let app = scheduled_task_app_server_with_backend(
        backend.clone(),
        &[("scheduled-route-provider", "scheduled-route-model")],
    )
    .await;
    initialize(&app.server).await;

    let inherited = create_task_with_model(&app.server, 2, "继承默认模型", None).await;
    let inherited_task_id = inherited["result"]["task"]["id"]
        .as_str()
        .expect("inherited task id");
    let inherited_run = request_ok(
        &app.server,
        3,
        METHOD_SCHEDULED_TASK_RUN_START,
        json!({"id": inherited_task_id}),
    )
    .await;
    let inherited_session_id = inherited_run["result"]["run"]["sessionId"]
        .as_str()
        .expect("inherited run session id");
    let inherited_session = app
        .runtime
        .read_session(AgentSessionReadParams {
            session_id: inherited_session_id.to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read inherited scheduled task session");
    let inherited_metadata = inherited_session
        .session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref())
        .expect("inherited scheduled task route metadata");
    assert_eq!(
        inherited_metadata["providerSelector"],
        "scheduled-route-provider"
    );
    assert_eq!(
        inherited_metadata["providerName"],
        "scheduled-route-provider"
    );
    assert_eq!(inherited_metadata["modelName"], "scheduled-route-model");

    let route_selector = format!(
        "route:{}.{}",
        URL_SAFE_NO_PAD.encode("scheduled-route-provider"),
        URL_SAFE_NO_PAD.encode("scheduled-route-model")
    );
    let explicit = create_task_with_model(&app.server, 4, "显式模型", Some(&route_selector)).await;
    let explicit_task_id = explicit["result"]["task"]["id"]
        .as_str()
        .expect("explicit task id");
    request_ok(
        &app.server,
        5,
        METHOD_SCHEDULED_TASK_RUN_START,
        json!({"id": explicit_task_id}),
    )
    .await;

    let expected_routes = vec![
        (
            "scheduled-route-provider".to_string(),
            "scheduled-route-model".to_string(),
        ),
        (
            "scheduled-route-provider".to_string(),
            "scheduled-route-model".to_string(),
        ),
    ];
    let preflight_routes = backend.preflight_routes();
    assert_eq!(preflight_routes.len(), 4);
    assert!(preflight_routes
        .iter()
        .all(|route| route == &expected_routes[0]));
    assert_eq!(backend.turn_routes(), expected_routes);
}

#[tokio::test]
async fn scheduled_task_new_thread_fails_closed_without_an_executable_model_route() {
    let backend = Arc::new(RouteAwareBackend::default());
    let app = scheduled_task_app_server_with_backend(backend.clone(), &[]).await;
    initialize(&app.server).await;
    let created = create_task_with_model(&app.server, 2, "缺少模型路由", None).await;
    let task_id = created["result"]["task"]["id"]
        .as_str()
        .expect("route-less task id");
    let mut outbound = app.server.subscribe_outbound_messages();

    let failed = request_raw(
        &app.server,
        3,
        METHOD_SCHEDULED_TASK_RUN_START,
        json!({"id": task_id}),
    )
    .await;
    assert_eq!(
        failed.pointer("/error/data/reasonCode"),
        Some(&json!("model_catalog_has_no_executable_selection"))
    );
    assert!(backend.preflight_routes().is_empty());
    assert!(backend.turn_routes().is_empty());

    let runs = request_ok(
        &app.server,
        4,
        METHOD_SCHEDULED_TASK_RUN_LIST,
        json!({"taskId": task_id, "limit": 20}),
    )
    .await;
    assert_eq!(runs["result"]["runs"][0]["status"], "error");
    let notification =
        next_outbound_notification(&mut outbound, METHOD_SCHEDULED_TASK_RUN_UPDATED).await;
    assert_eq!(notification["params"]["taskId"], task_id);
    assert_eq!(notification["params"]["status"], "error");
    assert_eq!(notification["params"]["attention"], true);
    assert_eq!(notification["params"]["notificationPolicy"], "failures");
}

#[tokio::test]
async fn scheduled_task_continue_thread_uses_canonical_session_identity() {
    let app = scheduled_task_app_server().await;
    initialize(&app.server).await;
    let canonical_session_id = "canonical-session-1";
    let canonical_thread_id = "canonical-thread-1";
    app.runtime
        .start_session(AgentSessionStartParams {
            session_id: Some(canonical_session_id.to_string()),
            thread_id: Some(canonical_thread_id.to_string()),
            app_id: "scheduled-task-source".to_string(),
            workspace_id: Some("project-alpha".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("create canonical source thread");

    let created = request_ok(
        &app.server,
        2,
        METHOD_SCHEDULED_TASK_CREATE,
        json!({
            "task": {
                "title": "延续项目对话",
                "prompt": "继续整理项目进展",
                "schedule": {
                    "type": "daily",
                    "time": "08:30",
                    "timezone": "Asia/Shanghai"
                },
                "execution": {
                    "threadMode": "continue_thread",
                    "sourceThreadId": canonical_thread_id,
                    "projectId": "project-alpha"
                },
                "enabled": true,
                "notificationPolicy": "failures",
                "overlapPolicy": "skip_if_running"
            }
        }),
    )
    .await;
    let task_id = created["result"]["task"]["id"]
        .as_str()
        .expect("continued scheduled task id");

    let started = request_ok(
        &app.server,
        3,
        METHOD_SCHEDULED_TASK_RUN_START,
        json!({"id": task_id}),
    )
    .await;
    assert_eq!(started["result"]["run"]["sessionId"], canonical_session_id);
    assert_eq!(started["result"]["run"]["threadId"], canonical_thread_id);
}

#[tokio::test]
async fn paused_manual_run_is_allowed_and_does_not_shift_schedule_anchor() {
    let app = scheduled_task_app_server().await;
    initialize(&app.server).await;
    let created = create_task(&app.server, 2, "手动运行合同").await;
    let task_id = created["result"]["task"]["id"]
        .as_str()
        .expect("created scheduled task id")
        .to_string();

    let paused = request_ok(
        &app.server,
        3,
        METHOD_SCHEDULED_TASK_ENABLED_SET,
        json!({"id": task_id, "enabled": false}),
    )
    .await;
    assert_eq!(paused["result"]["task"]["enabled"], false);
    assert_eq!(paused["result"]["task"]["nextRunAt"], Value::Null);
    let paused_run = request_ok(
        &app.server,
        4,
        METHOD_SCHEDULED_TASK_RUN_START,
        json!({"id": task_id}),
    )
    .await;
    assert!(paused_run["result"]["run"]["id"].is_string());
    let read_paused = request_ok(
        &app.server,
        5,
        METHOD_SCHEDULED_TASK_READ,
        json!({"id": task_id}),
    )
    .await;
    assert_eq!(read_paused["result"]["task"]["enabled"], false);
    assert_eq!(read_paused["result"]["task"]["nextRunAt"], Value::Null);

    let read_running_paused = request_ok(
        &app.server,
        6,
        METHOD_SCHEDULED_TASK_READ,
        json!({"id": task_id}),
    )
    .await;
    assert_eq!(read_running_paused["result"]["task"]["enabled"], false);
    assert_eq!(
        read_running_paused["result"]["task"]["nextRunAt"],
        Value::Null
    );

    let anchor_app = scheduled_task_app_server().await;
    initialize(&anchor_app.server).await;
    let anchor_created = create_task(&anchor_app.server, 2, "手动运行锚点").await;
    let anchor_task_id = anchor_created["result"]["task"]["id"]
        .as_str()
        .expect("anchor task id")
        .to_string();
    let scheduled_next_run = anchor_created["result"]["task"]["nextRunAt"]
        .as_str()
        .expect("anchor task next run")
        .to_string();
    request_ok(
        &anchor_app.server,
        3,
        METHOD_SCHEDULED_TASK_RUN_START,
        json!({"id": anchor_task_id}),
    )
    .await;
    let read_after_run = request_ok(
        &anchor_app.server,
        4,
        METHOD_SCHEDULED_TASK_READ,
        json!({"id": anchor_task_id}),
    )
    .await;
    assert_eq!(
        read_after_run["result"]["task"]["nextRunAt"],
        scheduled_next_run
    );
}

async fn create_task(server: &AppServer, id: u64, title: &str) -> Value {
    create_task_with_model(server, id, title, Some("gpt-5")).await
}

async fn create_task_with_model(
    server: &AppServer,
    id: u64,
    title: &str,
    model_id: Option<&str>,
) -> Value {
    let mut execution = json!({
        "threadMode": "new_thread",
        "projectId": "project-alpha",
        "cwd": "/tmp/project-alpha",
        "reasoningEffort": "medium",
        "requestMetadata": {
            "harness": {
                "service_skill": { "id": "daily-brief" }
            }
        }
    });
    if let Some(model_id) = model_id {
        execution["modelId"] = json!(model_id);
    }
    request_ok(
        server,
        id,
        METHOD_SCHEDULED_TASK_CREATE,
        json!({
            "task": {
                "title": title,
                "prompt": "整理今天的重要进展",
                "schedule": {
                    "type": "weekdays",
                    "time": "08:30",
                    "timezone": "Asia/Shanghai"
                },
                "execution": execution,
                "enabled": true,
                "notificationPolicy": "failures",
                "overlapPolicy": "skip_if_running"
            }
        }),
    )
    .await
}

fn insert_chat_provider(conn: &Connection, provider_id: &str, model_id: &str, sort_order: i32) {
    let now = Utc::now();
    ApiKeyProviderDao::insert_provider(
        conn,
        &ApiKeyProvider {
            id: provider_id.to_string(),
            name: provider_id.to_string(),
            provider_type: ApiProviderType::Openai,
            api_host: "https://fixture.invalid/v1".to_string(),
            is_system: false,
            group: ProviderGroup::Custom,
            enabled: true,
            sort_order,
            api_version: None,
            project: None,
            location: None,
            region: None,
            models: vec![ProviderModelConfig {
                id: model_id.to_string(),
                display_name: None,
                capability: Some(ProviderModelCapability {
                    task_families: vec![ModelTaskFamily::Chat],
                    input_modalities: vec![ModelModality::Text],
                    output_modalities: vec![ModelModality::Text],
                    runtime_features: vec![
                        ModelRuntimeFeature::Streaming,
                        ModelRuntimeFeature::ToolCalling,
                    ],
                    capabilities: ModelCapabilities {
                        tools: true,
                        streaming: true,
                        function_calling: true,
                        ..ModelCapabilities::default()
                    },
                }),
            }],
            prompt_cache_mode: None,
            created_at: now,
            updated_at: now,
        },
    )
    .expect("insert scheduled task chat provider");
    ApiKeyProviderDao::insert_api_key(
        conn,
        &ApiKeyEntry {
            id: format!("{provider_id}-key"),
            provider_id: provider_id.to_string(),
            api_key_encrypted: ApiKeyProviderService::new().encrypt_api_key("fixture-key"),
            alias: None,
            enabled: true,
            usage_count: 0,
            error_count: 0,
            last_used_at: None,
            created_at: now,
        },
    )
    .expect("insert scheduled task chat provider key");
}

async fn initialize(server: &AppServer) {
    let response = request_ok(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "scheduled-tasks-jsonrpc-test",
                "version": "1.0.0"
            }
        }),
    )
    .await;
    assert_eq!(
        response.pointer("/result/serverInfo/protocolVersion"),
        Some(&json!(PROTOCOL_VERSION))
    );
    let lines = server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "method": METHOD_INITIALIZED,
                "params": {}
            })
            .to_string(),
        )
        .await
        .expect("handle initialized notification");
    assert!(lines.is_empty());
}

async fn next_outbound_notification(
    outbound: &mut tokio::sync::broadcast::Receiver<JsonRpcMessage>,
    method: &str,
) -> Value {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let message = outbound
                .recv()
                .await
                .expect("receive App Server notification");
            let value = serde_json::to_value(message).expect("serialize App Server notification");
            if value.get("method").and_then(Value::as_str) == Some(method) {
                return value;
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {method}"))
}

async fn request_ok(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let response = request_raw(server, id, method, params).await;
    assert!(
        response.get("error").is_none(),
        "{method} request failed: {response:#}"
    );
    response
}

async fn request_raw(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params
            })
            .to_string(),
        )
        .await
        .expect("handle scheduled task JSON-RPC request")
        .iter()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("{method} should return response id {id}"))
}
