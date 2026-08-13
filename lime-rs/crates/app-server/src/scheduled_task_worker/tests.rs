use super::*;
use crate::automation_execution::AutomationRunFinish;
use crate::{
    ActionRespondRequest, CancelExecutionRequest, ExecutionBackend, ExecutionRequest,
    LocalAppDataSource, ProjectionStore, RuntimeEvent, RuntimeEventSink,
};
use app_server_protocol::AgentSessionStartParams;
use lime_core::config::{AutomationExecutionMode, DeliveryConfig, TaskSchedule};
use lime_core::database::dao::automation_job::{AutomationJob, AutomationJobDao};
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

#[derive(Default)]
struct RecordingCompletedBackend {
    requests: Mutex<Vec<ExecutionRequest>>,
}

#[async_trait::async_trait]
impl ExecutionBackend for RecordingCompletedBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.requests
            .lock()
            .expect("scheduled task worker request mutex poisoned")
            .push(request);
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

struct WorkerFixture {
    _temp: TempDir,
    db: DbConnection,
    runtime: RuntimeCore,
    backend: Arc<RecordingCompletedBackend>,
}

async fn worker_fixture() -> WorkerFixture {
    let temp = tempfile::tempdir().expect("create scheduled task worker temp dir");
    let db = database::init_database_at_path(temp.path().join("product.sqlite"))
        .expect("initialize scheduled task worker database");
    let app_data_source = LocalAppDataSource::initialize_with_roots(
        db.clone(),
        temp.path(),
        temp.path().join("app-server"),
    )
    .await
    .expect("initialize scheduled task worker app data source");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("initialize scheduled task worker projection store"),
    );
    let backend = Arc::new(RecordingCompletedBackend::default());
    let runtime = RuntimeCore::with_backend(backend.clone())
        .with_app_data_source(Arc::new(app_data_source))
        .with_projection_store(projection_store);
    WorkerFixture {
        _temp: temp,
        db,
        runtime,
        backend,
    }
}

fn insert_job(
    db: &DbConnection,
    id: &str,
    due_at: &str,
    thread_mode: &str,
    source_thread_id: Option<&str>,
    scheduled_task: bool,
) {
    let now = Utc::now().to_rfc3339();
    let mut payload = json!({
        "kind": "agent_turn",
        "prompt": "整理项目进展",
        "thread_mode": thread_mode,
        "notification_policy": "failures",
        "overlap_policy": "skip_if_running"
    });
    if scheduled_task {
        payload["scheduled_task_schedule"] = json!({
            "type": "daily",
            "time": "08:30",
            "timezone": "Asia/Shanghai"
        });
    }
    if let Some(source_thread_id) = source_thread_id {
        payload["source_thread_id"] = json!(source_thread_id);
    }
    let conn = database::lock_db(db).expect("lock scheduled task worker database");
    AutomationJobDao::create(
        &conn,
        &AutomationJob {
            id: id.to_string(),
            name: id.to_string(),
            description: None,
            enabled: true,
            workspace_id: "workspace-1".to_string(),
            execution_mode: AutomationExecutionMode::Skill,
            schedule: TaskSchedule::Every { every_secs: 3_600 },
            payload,
            delivery: DeliveryConfig::default(),
            timeout_secs: None,
            max_retries: 3,
            next_run_at: Some(due_at.to_string()),
            last_status: None,
            last_error: None,
            last_run_at: None,
            last_finished_at: None,
            running_started_at: None,
            consecutive_failures: 0,
            last_retry_count: 0,
            auto_disabled_until: None,
            last_delivery: None,
            created_at: now.clone(),
            updated_at: now,
        },
    )
    .expect("insert scheduled task worker job");
}

#[tokio::test]
async fn due_scan_ignores_legacy_automation_and_executes_scheduled_task_once() {
    let fixture = worker_fixture().await;
    let now = Utc::now();
    let due_at = (now - chrono::Duration::minutes(1)).to_rfc3339();
    insert_job(
        &fixture.db,
        "legacy-job",
        &due_at,
        "new_thread",
        None,
        false,
    );
    insert_job(
        &fixture.db,
        "scheduled-task",
        &due_at,
        "new_thread",
        None,
        true,
    );

    let claims = claim_due_scheduled_tasks(&fixture.db, &now.to_rfc3339(), 32)
        .expect("claim due scheduled tasks");
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0].job.id, "scheduled-task");
    let run_id = claims[0].run_id.clone();
    execute_claimed_scheduled_task(
        &fixture.db,
        &fixture.runtime,
        claims.into_iter().next().unwrap(),
    )
    .await
    .expect("execute claimed scheduled task");

    let conn = database::lock_db(&fixture.db).expect("lock scheduled task worker database");
    let run = AgentRunDao::get_run(&conn, &run_id)
        .expect("read scheduled task run")
        .expect("scheduled task run exists");
    assert_eq!(run.status, AgentRunStatus::Success);
    let metadata: Value = serde_json::from_str(run.metadata.as_deref().unwrap()).unwrap();
    assert_eq!(metadata["trigger"], "schedule");
    assert_eq!(metadata["scheduledFor"], due_at);
    let task = AutomationJobDao::get(&conn, "scheduled-task")
        .expect("read scheduled task")
        .expect("scheduled task exists");
    assert!(task.running_started_at.is_none());
    assert_eq!(task.last_status.as_deref(), Some("success"));
    drop(conn);

    let duplicate =
        claim_due_scheduled_tasks(&fixture.db, &now.to_rfc3339(), 32).expect("repeat due scan");
    assert!(duplicate.is_empty());
    let requests = fixture
        .backend
        .requests
        .lock()
        .expect("scheduled task worker request mutex poisoned");
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0]
            .session
            .business_object_ref
            .as_ref()
            .unwrap()
            .metadata
            .as_ref()
            .unwrap()["runId"],
        run_id
    );
}

#[tokio::test]
async fn restart_recovery_marks_stale_run_once_and_keeps_claimed_window() {
    let fixture = worker_fixture().await;
    let now = Utc::now();
    let due_at = (now - chrono::Duration::minutes(1)).to_rfc3339();
    insert_job(
        &fixture.db,
        "restart-task",
        &due_at,
        "new_thread",
        None,
        true,
    );
    let claim = claim_due_scheduled_tasks(&fixture.db, &now.to_rfc3339(), 32)
        .expect("claim restart task")
        .into_iter()
        .next()
        .expect("restart task claim");
    {
        let mut conn = database::lock_db(&fixture.db).expect("lock restart task database");
        assert!(AutomationWindowClaimDao::mark_started(
            &mut conn,
            &claim,
            &(now + chrono::Duration::seconds(1)).to_rfc3339(),
            "restart-session",
            &json!({
                "claimedAt": claim.claimed_at,
                "threadId": "restart-thread",
                "turnId": claim.run_id,
            })
            .to_string(),
        )
        .expect("mark restart task started"));
    }

    let recovered_at = (now + chrono::Duration::minutes(5)).to_rfc3339();
    let recoveries =
        recover_stale_scheduled_task_runs(&fixture.db, &fixture.runtime, &recovered_at)
            .await
            .expect("recover restart task");
    assert_eq!(recoveries.len(), 1);
    assert_eq!(
        recoveries[0].kind,
        lime_scheduler::AutomationRunRecoveryKind::Interrupted
    );

    let conn = database::lock_db(&fixture.db).expect("read recovered restart task");
    let run = AgentRunDao::get_run(&conn, &claim.run_id)
        .expect("read recovered restart run")
        .expect("recovered restart run exists");
    assert_eq!(run.status, AgentRunStatus::Error);
    assert_eq!(run.error_code.as_deref(), Some("scheduled_run_interrupted"));
    let task = AutomationJobDao::get(&conn, "restart-task")
        .expect("read recovered restart task")
        .expect("recovered restart task exists");
    assert!(task.running_started_at.is_none());
    assert!(task.next_run_at.is_some());
    drop(conn);

    let repeat = recover_stale_scheduled_task_runs(
        &fixture.db,
        &fixture.runtime,
        &(now + chrono::Duration::minutes(6)).to_rfc3339(),
    )
    .await
    .expect("repeat restart recovery");
    assert!(repeat.is_empty());
    assert!(
        claim_due_scheduled_tasks(&fixture.db, &now.to_rfc3339(), 32)
            .expect("scan after restart recovery")
            .is_empty()
    );
}

#[test]
fn clock_rollback_does_not_reclaim_an_already_claimed_window() {
    let temp = tempfile::tempdir().expect("create clock rollback task temp dir");
    let db = database::init_database_at_path(temp.path().join("product.sqlite"))
        .expect("initialize clock rollback task database");
    let now = Utc::now();
    let due_at = (now - chrono::Duration::minutes(1)).to_rfc3339();
    insert_job(&db, "clock-task", &due_at, "new_thread", None, true);

    let claims =
        claim_due_scheduled_tasks(&db, &now.to_rfc3339(), 32).expect("claim before clock rollback");
    assert_eq!(claims.len(), 1);
    let claimed_run_id = claims[0].run_id.clone();
    let earlier = (now - chrono::Duration::hours(2)).to_rfc3339();
    let repeated = claim_due_scheduled_tasks(&db, &earlier, 32).expect("scan after clock rollback");
    assert!(repeated.is_empty());
    let conn = database::lock_db(&db).expect("read clock rollback task");
    let runs = AgentRunDao::list_runs_by_source_ref(&conn, "automation", "clock-task", 10)
        .expect("read clock rollback runs");
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].id, claimed_run_id);
}

#[tokio::test]
async fn restart_recovery_ignores_legacy_automation_running_state() {
    let fixture = worker_fixture().await;
    let now = Utc::now();
    let due_at = (now - chrono::Duration::minutes(1)).to_rfc3339();
    insert_job(
        &fixture.db,
        "legacy-running-task",
        &due_at,
        "new_thread",
        None,
        false,
    );
    {
        let conn = database::lock_db(&fixture.db).expect("lock legacy running task database");
        conn.execute(
            "UPDATE automation_jobs
             SET running_started_at = ?1, last_status = 'running'
             WHERE id = ?2",
            rusqlite::params![now.to_rfc3339(), "legacy-running-task"],
        )
        .expect("mark legacy automation running");
    }

    let recoveries = recover_stale_scheduled_task_runs(
        &fixture.db,
        &fixture.runtime,
        &(now + chrono::Duration::minutes(1)).to_rfc3339(),
    )
    .await
    .expect("recover scheduled task runs");
    assert!(recoveries.is_empty());
    let conn = database::lock_db(&fixture.db).expect("read legacy running task database");
    let job = AutomationJobDao::get(&conn, "legacy-running-task")
        .expect("read legacy running task")
        .expect("legacy running task exists");
    assert_eq!(job.last_status.as_deref(), Some("running"));
    assert!(job.running_started_at.is_some());
}

#[tokio::test]
async fn restart_recovery_recomputes_schedule_after_running_task_edit() {
    let fixture = worker_fixture().await;
    let now = Utc::now();
    let due_at = (now - chrono::Duration::minutes(1)).to_rfc3339();
    insert_job(
        &fixture.db,
        "edited-running-task",
        &due_at,
        "new_thread",
        None,
        true,
    );
    let claim = claim_due_scheduled_tasks(&fixture.db, &now.to_rfc3339(), 32)
        .expect("claim edited running task")
        .into_iter()
        .next()
        .expect("edited running task claim");
    {
        let mut conn = database::lock_db(&fixture.db).expect("lock edited running task database");
        assert!(AutomationWindowClaimDao::mark_started(
            &mut conn,
            &claim,
            &(now + chrono::Duration::seconds(1)).to_rfc3339(),
            "edited-running-session",
            &json!({
                "claimedAt": claim.claimed_at,
                "taskRevision": claim.task_revision,
            })
            .to_string(),
        )
        .expect("mark edited running task started"));
        conn.execute(
            "UPDATE automation_jobs
             SET schedule_json = ?1, next_run_at = NULL, updated_at = ?2
             WHERE id = ?3",
            rusqlite::params![
                serde_json::to_string(&TaskSchedule::Every { every_secs: 7_200 }).unwrap(),
                (now + chrono::Duration::minutes(2)).to_rfc3339(),
                "edited-running-task",
            ],
        )
        .expect("edit task while running");
    }

    let recovered_at = now + chrono::Duration::minutes(5);
    let recoveries = recover_stale_scheduled_task_runs(
        &fixture.db,
        &fixture.runtime,
        &recovered_at.to_rfc3339(),
    )
    .await
    .expect("recover edited running task");
    assert_eq!(recoveries.len(), 1);
    let conn = database::lock_db(&fixture.db).expect("read edited recovered task");
    let job = AutomationJobDao::get(&conn, "edited-running-task")
        .expect("read edited recovered task")
        .expect("edited recovered task exists");
    assert!(job.running_started_at.is_none());
    let expected_next_run_at = (recovered_at + chrono::Duration::hours(2)).to_rfc3339();
    assert_eq!(
        job.next_run_at.as_deref(),
        Some(expected_next_run_at.as_str())
    );
}

#[test]
fn recent_backlog_claim_projects_catch_up_trigger() {
    let fixture = tempfile::tempdir().expect("create catch-up task temp dir");
    let db = database::init_database_at_path(fixture.path().join("product.sqlite"))
        .expect("initialize catch-up task database");
    let now = Utc::now();
    insert_job(
        &db,
        "catch-up-task",
        &(now - chrono::Duration::hours(2)).to_rfc3339(),
        "new_thread",
        None,
        true,
    );
    let claims =
        claim_due_scheduled_tasks(&db, &now.to_rfc3339(), 32).expect("claim recent backlog");
    let claim = claims.into_iter().next().expect("catch-up claim");
    assert!(claim.catch_up);
    let conn = database::lock_db(&db).expect("read catch-up task database");
    let run = AgentRunDao::get_run(&conn, &claim.run_id)
        .expect("read catch-up run")
        .expect("catch-up run exists");
    let metadata: Value = serde_json::from_str(run.metadata.as_deref().unwrap())
        .expect("parse catch-up run metadata");
    assert_eq!(metadata["trigger"], "catch_up");
}

#[tokio::test]
async fn one_shot_due_task_executes_with_null_next_run() {
    let fixture = worker_fixture().await;
    let now = Utc::now();
    let due_at = (now - chrono::Duration::minutes(1)).to_rfc3339();
    insert_job(
        &fixture.db,
        "one-shot-task",
        &due_at,
        "new_thread",
        None,
        true,
    );
    {
        let conn = database::lock_db(&fixture.db).expect("lock one-shot task database");
        conn.execute(
            "UPDATE automation_jobs SET schedule_json = ?1 WHERE id = ?2",
            rusqlite::params![
                serde_json::to_string(&TaskSchedule::At { at: due_at.clone() }).unwrap(),
                "one-shot-task"
            ],
        )
        .expect("set one-shot schedule");
    }

    let claim = claim_due_scheduled_tasks(&fixture.db, &now.to_rfc3339(), 32)
        .expect("claim one-shot task")
        .into_iter()
        .next()
        .expect("one-shot task claim");
    assert!(claim.next_run_at.is_none());
    execute_claimed_scheduled_task(&fixture.db, &fixture.runtime, claim.clone())
        .await
        .expect("execute one-shot task");

    let conn = database::lock_db(&fixture.db).expect("read one-shot task database");
    let job = AutomationJobDao::get(&conn, "one-shot-task")
        .expect("read one-shot task")
        .expect("one-shot task exists");
    assert_eq!(job.last_status.as_deref(), Some("success"));
    assert!(job.running_started_at.is_none());
    assert!(job.next_run_at.is_none());
    let run = AgentRunDao::get_run(&conn, &claim.run_id)
        .expect("read one-shot run")
        .expect("one-shot run exists");
    assert_eq!(run.status, AgentRunStatus::Success);
}

#[tokio::test]
async fn task_change_after_claim_cancels_run_without_starting_turn() {
    let fixture = worker_fixture().await;
    let now = Utc::now();
    let due_at = (now - chrono::Duration::minutes(1)).to_rfc3339();
    insert_job(
        &fixture.db,
        "scheduled-task",
        &due_at,
        "new_thread",
        None,
        true,
    );
    let claim = claim_due_scheduled_tasks(&fixture.db, &now.to_rfc3339(), 32)
        .expect("claim due scheduled task")
        .into_iter()
        .next()
        .expect("scheduled task claim");
    {
        let conn = database::lock_db(&fixture.db).expect("lock scheduled task worker database");
        conn.execute(
            "UPDATE automation_jobs SET enabled = 0, updated_at = ?1 WHERE id = ?2",
            rusqlite::params![
                (now + chrono::Duration::seconds(1)).to_rfc3339(),
                "scheduled-task"
            ],
        )
        .expect("pause task after claim");
    }

    execute_claimed_scheduled_task(&fixture.db, &fixture.runtime, claim.clone())
        .await
        .expect("invalidate changed scheduled task claim");

    let conn = database::lock_db(&fixture.db).expect("lock scheduled task worker database");
    let run = AgentRunDao::get_run(&conn, &claim.run_id)
        .expect("read invalidated scheduled task run")
        .expect("invalidated scheduled task run exists");
    assert_eq!(run.status, AgentRunStatus::Canceled);
    assert!(fixture
        .backend
        .requests
        .lock()
        .expect("scheduled task worker request mutex poisoned")
        .is_empty());
}

#[tokio::test]
async fn continue_thread_uses_canonical_identity_for_claimed_run() {
    let fixture = worker_fixture().await;
    let canonical_session_id = "canonical-session";
    let canonical_thread_id = "canonical-thread";
    fixture
        .runtime
        .start_session(AgentSessionStartParams {
            session_id: Some(canonical_session_id.to_string()),
            thread_id: Some(canonical_thread_id.to_string()),
            app_id: "scheduled-task-source".to_string(),
            workspace_id: Some("workspace-1".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("start canonical source thread");
    let now = Utc::now();
    let due_at = (now - chrono::Duration::minutes(1)).to_rfc3339();
    insert_job(
        &fixture.db,
        "continued-task",
        &due_at,
        "continue_thread",
        Some(canonical_thread_id),
        true,
    );
    let claim = claim_due_scheduled_tasks(&fixture.db, &now.to_rfc3339(), 32)
        .expect("claim continued scheduled task")
        .into_iter()
        .next()
        .expect("continued scheduled task claim");

    execute_claimed_scheduled_task(&fixture.db, &fixture.runtime, claim.clone())
        .await
        .expect("execute continued scheduled task");

    let requests = fixture
        .backend
        .requests
        .lock()
        .expect("scheduled task worker request mutex poisoned");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].session.session_id, canonical_session_id);
    assert_eq!(requests[0].session.thread_id, canonical_thread_id);
    let conn = database::lock_db(&fixture.db).expect("lock scheduled task worker database");
    let run = AgentRunDao::get_run(&conn, &claim.run_id)
        .expect("read continued scheduled task run")
        .expect("continued scheduled task run exists");
    assert_eq!(run.session_id.as_deref(), Some(canonical_session_id));
}

#[test]
fn reconcile_due_window_marks_recent_backlog_as_catch_up() {
    let now = Utc::now();
    let due_at = (now - chrono::Duration::hours(2)).to_rfc3339();
    let job = sample_reconcile_job(&due_at, 3_600);
    let window = reconcile_due_window(&job, &now.to_rfc3339()).expect("reconcile recent backlog");
    assert!(window.catch_up);
    assert_eq!(window.skipped_window_count, 2);
    assert_eq!(window.scheduled_for, now.to_rfc3339());
    assert!(window.next_run_at.is_some());
}

#[test]
fn reconcile_due_window_marks_old_backlog_as_missed() {
    let now = Utc::now();
    let due_at = (now - chrono::Duration::hours(26)).to_rfc3339();
    let job = sample_reconcile_job(&due_at, 48 * 3_600);
    let window = reconcile_due_window(&job, &now.to_rfc3339()).expect("reconcile old backlog");
    assert!(!window.catch_up);
    assert_eq!(window.skipped_window_count, 0);
    assert!(window.next_run_at.is_some());
}

#[test]
fn due_scan_records_old_window_as_missed_without_claiming_a_turn() {
    let temp = tempfile::tempdir().expect("create missed task temp dir");
    let db = database::init_database_at_path(temp.path().join("product.sqlite"))
        .expect("initialize missed task database");
    let now = Utc::now();
    insert_job(
        &db,
        "missed-task",
        &(now - chrono::Duration::hours(26)).to_rfc3339(),
        "new_thread",
        None,
        true,
    );
    {
        let conn = database::lock_db(&db).expect("lock missed task database");
        conn.execute(
            "UPDATE automation_jobs SET schedule_json = ?1 WHERE id = ?2",
            rusqlite::params![
                serde_json::to_string(&TaskSchedule::At {
                    at: (now - chrono::Duration::hours(26)).to_rfc3339()
                })
                .unwrap(),
                "missed-task"
            ],
        )
        .expect("set one-shot old schedule");
    }
    let claims = claim_due_scheduled_tasks(&db, &now.to_rfc3339(), 32).expect("reconcile old task");
    assert!(claims.is_empty());
    let conn = database::lock_db(&db).expect("read missed task database");
    let task = AutomationJobDao::get(&conn, "missed-task")
        .expect("read missed task")
        .expect("missed task exists");
    assert_eq!(task.last_status.as_deref(), Some("missed"));
    let runs = AgentRunDao::list_runs_by_source_ref(&conn, "automation", "missed-task", 10)
        .expect("read missed task history");
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].status, AgentRunStatus::Missed);
    assert_eq!(runs[0].error_code.as_deref(), Some("scheduled_run_missed"));
}

#[tokio::test]
async fn overlap_scan_records_missed_and_terminal_preserves_advanced_window() {
    let fixture = worker_fixture().await;
    let now = Utc::now();
    let due_at = (now - chrono::Duration::minutes(1)).to_rfc3339();
    insert_job(
        &fixture.db,
        "overlap-task",
        &due_at,
        "new_thread",
        None,
        true,
    );
    let claim = claim_due_scheduled_tasks(&fixture.db, &now.to_rfc3339(), 32)
        .expect("claim overlap task")
        .into_iter()
        .next()
        .expect("overlap task claim");
    {
        let mut conn = database::lock_db(&fixture.db).expect("lock overlap task database");
        assert!(AutomationWindowClaimDao::mark_started(
            &mut conn,
            &claim,
            &(now + chrono::Duration::seconds(1)).to_rfc3339(),
            "overlap-session",
            "{}",
        )
        .expect("mark overlap task started"));
    }

    let overlap_scan_at = now + chrono::Duration::hours(1);
    let overlap_claims = claim_due_scheduled_tasks(&fixture.db, &overlap_scan_at.to_rfc3339(), 32)
        .expect("scan overlapping window");
    assert!(overlap_claims.is_empty());
    let advanced_next_run = {
        let conn = database::lock_db(&fixture.db).expect("read overlapping task database");
        let job = AutomationJobDao::get(&conn, "overlap-task")
            .expect("read overlap task")
            .expect("overlap task exists");
        assert_eq!(job.last_status.as_deref(), Some("running"));
        assert_eq!(
            job.running_started_at.as_deref(),
            Some(claim.claimed_at.as_str())
        );
        let runs = AgentRunDao::list_runs_by_source_ref(&conn, "automation", "overlap-task", 10)
            .expect("read overlap task runs");
        assert!(runs.iter().any(|run| {
            run.status == AgentRunStatus::Missed
                && run.error_code.as_deref() == Some("scheduled_run_overlap")
        }));
        job.next_run_at.expect("overlap scan advances next run")
    };

    fixture
        .runtime
        .finish_automation_job_run(AutomationRunFinish {
            job: claim.job.clone(),
            run_id: claim.run_id.clone(),
            status: AgentRunStatus::Success,
            finished_at: (now + chrono::Duration::hours(1) + chrono::Duration::minutes(1))
                .to_rfc3339(),
            duration_ms: Some(3_660_000),
            error_code: None,
            error_message: None,
            metadata: json!({"trigger": "schedule"}),
            ownership_started_at: Some(claim.claimed_at.clone()),
            task_revision: Some(claim.task_revision.clone()),
        })
        .await
        .expect("finish overlapping task run");

    let conn = database::lock_db(&fixture.db).expect("read finished overlap task database");
    let job = AutomationJobDao::get(&conn, "overlap-task")
        .expect("read finished overlap task")
        .expect("finished overlap task exists");
    assert_eq!(job.last_status.as_deref(), Some("success"));
    assert!(job.running_started_at.is_none());
    assert_eq!(job.next_run_at.as_deref(), Some(advanced_next_run.as_str()));
}

fn sample_reconcile_job(next_run_at: &str, every_secs: u64) -> AutomationJob {
    let now = Utc::now().to_rfc3339();
    AutomationJob {
        id: "reconcile-task".to_string(),
        name: "reconcile-task".to_string(),
        description: None,
        enabled: true,
        workspace_id: "workspace-1".to_string(),
        execution_mode: AutomationExecutionMode::Skill,
        schedule: TaskSchedule::Every { every_secs },
        payload: json!({
            "kind": "agent_turn",
            "prompt": "整理项目进展",
            "thread_mode": "new_thread",
            "scheduled_task_schedule": {"type": "daily", "time": "08:30", "timezone": "Asia/Shanghai"},
            "notification_policy": "failures",
            "overlap_policy": "skip_if_running"
        }),
        delivery: DeliveryConfig::default(),
        timeout_secs: None,
        max_retries: 3,
        next_run_at: Some(next_run_at.to_string()),
        last_status: None,
        last_error: None,
        last_run_at: None,
        last_finished_at: None,
        running_started_at: None,
        consecutive_failures: 0,
        last_retry_count: 0,
        auto_disabled_until: None,
        last_delivery: None,
        created_at: now.clone(),
        updated_at: now,
    }
}
