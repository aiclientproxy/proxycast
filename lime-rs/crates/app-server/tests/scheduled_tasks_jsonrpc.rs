use std::sync::{Arc, Mutex};

use app_server::{AppServer, LocalAppDataSource, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::{
    AgentSessionStartParams, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_SCHEDULED_TASK_CREATE,
    METHOD_SCHEDULED_TASK_DELETE, METHOD_SCHEDULED_TASK_ENABLED_SET, METHOD_SCHEDULED_TASK_LIST,
    METHOD_SCHEDULED_TASK_READ, METHOD_SCHEDULED_TASK_RUN_LIST, METHOD_SCHEDULED_TASK_RUN_START,
    METHOD_SCHEDULED_TASK_SCHEDULE_PREVIEW, METHOD_SCHEDULED_TASK_UPDATE, PROTOCOL_VERSION,
};
use lime_core::database::schema::create_tables;
use rusqlite::Connection;
use serde_json::{json, Value};
use tempfile::TempDir;

struct ScheduledTaskAppServer {
    _temp: TempDir,
    runtime: RuntimeCore,
    server: AppServer,
}

async fn scheduled_task_app_server() -> ScheduledTaskAppServer {
    let temp = TempDir::new().expect("create scheduled task fixture temp dir");
    let connection = Connection::open_in_memory().expect("open scheduled task product db");
    create_tables(&connection).expect("create scheduled task product schema");
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
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_app_data_source(Arc::new(app_data_source))
        .with_projection_store(projection_store);

    ScheduledTaskAppServer {
        _temp: temp,
        runtime: runtime.clone(),
        server: AppServer::with_runtime(runtime),
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

    let read = request_ok(
        &app.server,
        4,
        METHOD_SCHEDULED_TASK_READ,
        json!({"id": task_id}),
    )
    .await;
    assert_eq!(read["result"]["task"]["title"], "每日项目简报");

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
                "execution": {
                    "threadMode": "new_thread",
                    "projectId": "project-alpha",
                    "cwd": "/tmp/project-alpha",
                    "modelId": "gpt-5",
                    "reasoningEffort": "medium"
                },
                "enabled": true,
                "notificationPolicy": "failures",
                "overlapPolicy": "skip_if_running"
            }
        }),
    )
    .await
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
