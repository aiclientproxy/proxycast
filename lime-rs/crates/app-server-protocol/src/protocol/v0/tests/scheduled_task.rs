use super::super::*;
use serde_json::json;

#[test]
fn scheduled_task_schedule_uses_codex_weekday_wire_values() {
    let schedule = ScheduledTaskSchedule::Weekly {
        days: vec![ScheduledTaskWeekday::MO, ScheduledTaskWeekday::FR],
        time: "08:30".to_string(),
        timezone: "Asia/Shanghai".to_string(),
    };

    assert_eq!(
        serde_json::to_value(schedule).expect("serialize schedule"),
        json!({
            "type": "weekly",
            "days": ["MO", "FR"],
            "time": "08:30",
            "timezone": "Asia/Shanghai"
        })
    );
}

#[test]
fn scheduled_task_policies_use_closed_wire_values() {
    assert_eq!(
        serde_json::to_value(ScheduledTaskThreadMode::ContinueThread)
            .expect("serialize thread mode"),
        json!("continue_thread")
    );
    assert_eq!(
        serde_json::to_value(ScheduledTaskNotificationPolicy::Failures)
            .expect("serialize notification policy"),
        json!("failures")
    );
    assert_eq!(
        serde_json::to_value(ScheduledTaskOverlapPolicy::SkipIfRunning)
            .expect("serialize overlap policy"),
        json!("skip_if_running")
    );
    assert!(serde_json::from_value::<ScheduledTaskThreadMode>(json!("unknown")).is_err());
}

#[test]
fn scheduled_task_methods_are_current_request_methods() {
    let methods = [
        METHOD_SCHEDULED_TASK_LIST,
        METHOD_SCHEDULED_TASK_READ,
        METHOD_SCHEDULED_TASK_CREATE,
        METHOD_SCHEDULED_TASK_UPDATE,
        METHOD_SCHEDULED_TASK_DELETE,
        METHOD_SCHEDULED_TASK_ENABLED_SET,
        METHOD_SCHEDULED_TASK_RUN_START,
        METHOD_SCHEDULED_TASK_RUN_LIST,
        METHOD_SCHEDULED_TASK_SCHEDULE_PREVIEW,
    ];

    for method in methods {
        assert!(is_app_server_request_method(method), "missing {method}");
        assert!(APP_SERVER_METHODS
            .iter()
            .any(|spec| { spec.method == method && spec.kind == AppServerMethodKind::Request }));
    }
}
