use super::*;
use lime_core::database::dao::agent_run::AgentRunStatus;
use lime_core::database::schema::create_tables;
use rusqlite::Connection;
use std::sync::{Arc, Mutex};

fn test_db() -> DbConnection {
    let conn = Connection::open_in_memory().expect("open scheduled task test database");
    create_tables(&conn).expect("create scheduled task test schema");
    Arc::new(Mutex::new(conn))
}

fn create_request(
    thread_mode: ScheduledTaskThreadMode,
    source_thread_id: Option<&str>,
) -> ScheduledTaskCreateParams {
    ScheduledTaskCreateParams {
        task: ScheduledTaskCreateRequest {
            title: "每日简报".to_string(),
            prompt: "整理今天的重要进展".to_string(),
            schedule: ScheduledTaskSchedule::Weekdays {
                time: "08:30".to_string(),
                timezone: "Asia/Shanghai".to_string(),
            },
            execution: ScheduledTaskExecution {
                thread_mode,
                source_thread_id: source_thread_id.map(ToOwned::to_owned),
                project_id: Some("project-1".to_string()),
                cwd: Some("/tmp/project-1".to_string()),
                model_id: Some("gpt-5".to_string()),
                reasoning_effort: Some("medium".to_string()),
                approval_policy: Some(json!("on-request")),
                sandbox_policy: Some(json!("workspace-write")),
            },
            enabled: true,
            notification_policy: Some(ScheduledTaskNotificationPolicy::Failures),
            overlap_policy: Some(ScheduledTaskOverlapPolicy::SkipIfRunning),
        },
    }
}

fn insert_legacy_automation_job(db: &DbConnection, source: &AutomationJob) {
    let mut legacy = source.clone();
    legacy.id = "legacy-automation-job".to_string();
    legacy.name = "旧浏览器任务".to_string();
    legacy.schedule = TaskSchedule::At {
        at: "2026-08-13T08:00:00Z".to_string(),
    };
    legacy.payload = json!({
        "kind": "browser_session",
        "prompt": "运行旧浏览器任务"
    });
    let conn = database::lock_db(db).expect("lock scheduled task test database");
    AutomationJobDao::create(&conn, &legacy).expect("insert legacy automation job");
}

fn insert_run(db: &DbConnection, task_id: &str, status: AgentRunStatus) {
    let run = AgentRun {
        id: "scheduled-run-1".to_string(),
        source: "automation".to_string(),
        source_ref: Some(task_id.to_string()),
        session_id: Some("canonical-session-1".to_string()),
        status,
        started_at: "2026-08-13T08:00:00Z".to_string(),
        finished_at: Some("2026-08-13T08:00:05Z".to_string()),
        duration_ms: Some(5_000),
        error_code: Some("provider_error".to_string()),
        error_message: Some("模型服务暂不可用".to_string()),
        metadata: Some(
            json!({
                "threadId": "canonical-thread-1",
                "turnId": "turn-1"
            })
            .to_string(),
        ),
        created_at: "2026-08-13T08:00:00Z".to_string(),
        updated_at: "2026-08-13T08:00:05Z".to_string(),
    };
    let conn = database::lock_db(db).expect("lock scheduled task test database");
    AgentRunDao::create_run(&conn, &run).expect("insert scheduled task run");
}

#[test]
fn scheduled_task_create_read_list_and_update_preserve_lineage() {
    let db = test_db();
    let created = create_scheduled_task(
        &db,
        create_request(ScheduledTaskThreadMode::ContinueThread, Some("thread-1")),
    )
    .expect("create scheduled task")
    .task;

    let read = read_scheduled_task(
        &db,
        ScheduledTaskIdParams {
            id: created.id.clone(),
        },
    )
    .expect("read scheduled task")
    .task
    .expect("scheduled task exists");
    assert_eq!(read.execution.source_thread_id.as_deref(), Some("thread-1"));
    {
        let conn = database::lock_db(&db).expect("lock scheduled task test database");
        let job = AutomationJobDao::get(&conn, &read.id)
            .expect("read scheduled task job")
            .expect("scheduled task job exists");
        assert!(job.payload.get("session_id").is_none());
        assert!(job.payload.get("thread_id").is_none());
    }

    let listed = list_scheduled_tasks(
        &db,
        ScheduledTaskListParams {
            query: Some("简报".to_string()),
            enabled: Some(true),
            ..ScheduledTaskListParams::default()
        },
    )
    .expect("list scheduled tasks");
    assert_eq!(listed.items.len(), 1);

    let updated = update_scheduled_task(
        &db,
        ScheduledTaskUpdateParams {
            id: created.id,
            task: ScheduledTaskUpdateRequest {
                prompt: Some("整理今天的重要进展并给出风险".to_string()),
                revision: Some(read.updated_at),
                ..ScheduledTaskUpdateRequest {
                    title: None,
                    prompt: None,
                    schedule: None,
                    execution: None,
                    enabled: None,
                    notification_policy: None,
                    overlap_policy: None,
                    revision: None,
                }
            },
        },
    )
    .expect("update scheduled task")
    .task;
    assert_eq!(
        updated.execution.source_thread_id.as_deref(),
        Some("thread-1")
    );
    assert_eq!(updated.prompt, "整理今天的重要进展并给出风险");
}

#[test]
fn scheduled_task_update_rejects_stale_revision() {
    let db = test_db();
    let created = create_scheduled_task(
        &db,
        create_request(ScheduledTaskThreadMode::NewThread, None),
    )
    .expect("create scheduled task")
    .task;
    let error = update_scheduled_task(
        &db,
        ScheduledTaskUpdateParams {
            id: created.id,
            task: ScheduledTaskUpdateRequest {
                title: Some("过期更新".to_string()),
                revision: Some("stale-revision".to_string()),
                ..ScheduledTaskUpdateRequest {
                    title: None,
                    prompt: None,
                    schedule: None,
                    execution: None,
                    enabled: None,
                    notification_policy: None,
                    overlap_policy: None,
                    revision: None,
                }
            },
        },
    )
    .expect_err("stale revision must fail");
    assert!(error.to_string().contains("其他窗口"));
}

#[test]
fn scheduled_task_boundary_excludes_and_rejects_legacy_automation_jobs() {
    let db = test_db();
    let created = create_scheduled_task(
        &db,
        create_request(ScheduledTaskThreadMode::NewThread, None),
    )
    .expect("create scheduled task")
    .task;
    let scheduled_job = {
        let conn = database::lock_db(&db).expect("lock scheduled task test database");
        AutomationJobDao::get(&conn, &created.id)
            .expect("read scheduled task job")
            .expect("scheduled task job exists")
    };
    insert_legacy_automation_job(&db, &scheduled_job);

    let listed = list_scheduled_tasks(&db, ScheduledTaskListParams::default())
        .expect("list scheduled tasks");
    assert_eq!(listed.items.len(), 1);
    assert_eq!(listed.items[0].id, created.id);

    let read_error = read_scheduled_task(
        &db,
        ScheduledTaskIdParams {
            id: "legacy-automation-job".to_string(),
        },
    )
    .expect_err("legacy automation read must fail closed");
    assert!(read_error.to_string().contains("已安排任务不存在"));

    let update_error = update_scheduled_task(
        &db,
        ScheduledTaskUpdateParams {
            id: "legacy-automation-job".to_string(),
            task: ScheduledTaskUpdateRequest {
                title: Some("不应更新".to_string()),
                prompt: None,
                schedule: None,
                execution: None,
                enabled: None,
                notification_policy: None,
                overlap_policy: None,
                revision: None,
            },
        },
    )
    .expect_err("legacy automation update must fail closed");
    assert!(update_error.to_string().contains("已安排任务不存在"));

    let delete_error = delete_scheduled_task(
        &db,
        ScheduledTaskIdParams {
            id: "legacy-automation-job".to_string(),
        },
    )
    .expect_err("legacy automation delete must fail closed");
    assert!(delete_error.to_string().contains("已安排任务不存在"));

    let runs_error = list_scheduled_task_runs(
        &db,
        ScheduledTaskRunListParams {
            task_id: "legacy-automation-job".to_string(),
            limit: None,
        },
    )
    .expect_err("legacy automation run list must fail closed");
    assert!(runs_error.to_string().contains("已安排任务不存在"));

    let run_error = super::super::start_scheduled_task_run_record(
        &db,
        "legacy-automation-job".to_string(),
        None,
    )
    .expect_err("legacy automation run must fail closed");
    assert!(run_error.to_string().contains("已安排任务不存在"));

    let conn = database::lock_db(&db).expect("lock scheduled task test database");
    assert!(AutomationJobDao::get(&conn, "legacy-automation-job")
        .expect("read retained legacy automation job")
        .is_some());
}

#[test]
fn scheduled_task_read_missing_job_remains_empty() {
    let db = test_db();
    let read = read_scheduled_task(
        &db,
        ScheduledTaskIdParams {
            id: "missing-task".to_string(),
        },
    )
    .expect("read missing scheduled task");
    assert!(read.task.is_none());
}

#[test]
fn scheduled_task_read_models_project_latest_failed_run_and_attention() {
    let db = test_db();
    let created = create_scheduled_task(
        &db,
        create_request(ScheduledTaskThreadMode::NewThread, None),
    )
    .expect("create scheduled task")
    .task;
    insert_run(&db, &created.id, AgentRunStatus::Error);

    let read = read_scheduled_task(
        &db,
        ScheduledTaskIdParams {
            id: created.id.clone(),
        },
    )
    .expect("read scheduled task")
    .task
    .expect("scheduled task exists");
    let last_run = read.last_run_summary.expect("latest run summary");
    assert_eq!(last_run.id, "scheduled-run-1");
    assert_eq!(last_run.status, "error");
    assert_eq!(last_run.session_id.as_deref(), Some("canonical-session-1"));
    assert_eq!(last_run.thread_id.as_deref(), Some("canonical-thread-1"));
    assert_eq!(last_run.turn_id.as_deref(), Some("turn-1"));
    assert_eq!(last_run.error.as_deref(), Some("模型服务暂不可用"));

    let listed = list_scheduled_tasks(&db, ScheduledTaskListParams::default())
        .expect("list scheduled tasks");
    assert_eq!(listed.items.len(), 1);
    assert!(listed.items[0].attention);
    assert_eq!(
        listed.items[0].last_run.as_ref().map(|run| run.id.as_str()),
        Some("scheduled-run-1")
    );
}

#[test]
fn scheduled_task_read_models_project_missed_run_as_attention() {
    let db = test_db();
    let created = create_scheduled_task(
        &db,
        create_request(ScheduledTaskThreadMode::NewThread, None),
    )
    .expect("create scheduled task")
    .task;
    insert_run(&db, &created.id, AgentRunStatus::Missed);

    let listed = list_scheduled_tasks(&db, ScheduledTaskListParams::default())
        .expect("list scheduled tasks");
    assert_eq!(listed.items.len(), 1);
    assert!(listed.items[0].attention);
    assert_eq!(
        listed.items[0]
            .last_run
            .as_ref()
            .map(|run| run.status.as_str()),
        Some("missed")
    );
}

#[test]
fn new_thread_storage_does_not_freeze_run_lineage() {
    let db = test_db();
    let created = create_scheduled_task(
        &db,
        create_request(ScheduledTaskThreadMode::NewThread, None),
    )
    .expect("create scheduled task")
    .task;
    let conn = database::lock_db(&db).expect("lock scheduled task test database");
    let job = AutomationJobDao::get(&conn, &created.id)
        .expect("read automation job")
        .expect("automation job exists");
    assert_eq!(
        job.payload.get("thread_mode").and_then(Value::as_str),
        Some("new_thread")
    );
    assert!(job.payload.get("session_id").is_none());
    assert!(job.payload.get("thread_id").is_none());
}

#[test]
fn schedule_preview_returns_five_occurrences() {
    let preview = preview_scheduled_task_schedule(ScheduledTaskSchedulePreviewParams {
        schedule: ScheduledTaskSchedule::Weekly {
            days: vec![ScheduledTaskWeekday::MO, ScheduledTaskWeekday::FR],
            time: "08:30".to_string(),
            timezone: "Asia/Shanghai".to_string(),
        },
    })
    .expect("preview scheduled task");
    assert_eq!(preview.next_run_at.len(), 5);
}

#[test]
fn hourly_schedule_keeps_optional_weekdays_and_rejects_empty_days() {
    let schedule = lower_scheduled_schedule(&ScheduledTaskSchedule::Hourly {
        interval_hours: 2,
        days: Some(vec![
            ScheduledTaskWeekday::FR,
            ScheduledTaskWeekday::MO,
            ScheduledTaskWeekday::FR,
        ]),
        minute: 15,
        timezone: "Asia/Shanghai".to_string(),
    })
    .expect("lower hourly weekdays");
    assert_eq!(
        schedule,
        TaskSchedule::Cron {
            expr: "15 */2 * * 1,5".to_string(),
            tz: Some("Asia/Shanghai".to_string()),
        }
    );

    let error = lower_scheduled_schedule(&ScheduledTaskSchedule::Hourly {
        interval_hours: 2,
        days: Some(Vec::new()),
        minute: 15,
        timezone: "Asia/Shanghai".to_string(),
    })
    .expect_err("empty hourly days must fail");
    assert!(error.to_string().contains("days"));
}
