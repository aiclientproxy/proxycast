pub(crate) const WORKFLOW_RUN_STARTED: &str = "workflow.run.started";
pub(crate) const WORKFLOW_RUN_RESUMING: &str = "workflow.run.resuming";
pub(crate) const WORKFLOW_RUN_RETRYING: &str = "workflow.run.retrying";
pub(crate) const WORKFLOW_RUN_COMPLETED: &str = "workflow.run.completed";
pub(crate) const WORKFLOW_RUN_FAILED: &str = "workflow.run.failed";
pub(crate) const WORKFLOW_RUN_CANCELED: &str = "workflow.run.canceled";

pub(crate) const WORKFLOW_STEP_STARTED: &str = "workflow.step.started";
pub(crate) const WORKFLOW_STEP_RESUMING: &str = "workflow.step.resuming";
pub(crate) const WORKFLOW_STEP_PROGRESS: &str = "workflow.step.progress";
pub(crate) const WORKFLOW_STEP_RETRYING: &str = "workflow.step.retrying";
pub(crate) const WORKFLOW_STEP_COMPLETED: &str = "workflow.step.completed";
pub(crate) const WORKFLOW_STEP_FAILED: &str = "workflow.step.failed";
pub(crate) const WORKFLOW_STEP_CANCELED: &str = "workflow.step.canceled";

pub(crate) fn workflow_run_event_is_terminal(event_type: &str) -> bool {
    matches!(
        event_type,
        WORKFLOW_RUN_COMPLETED | WORKFLOW_RUN_FAILED | WORKFLOW_RUN_CANCELED
    )
}

pub(crate) fn workflow_step_event_is_terminal(event_type: &str) -> bool {
    matches!(
        event_type,
        WORKFLOW_STEP_COMPLETED | WORKFLOW_STEP_FAILED | WORKFLOW_STEP_CANCELED
    )
}
