use super::super::event_store::append_workflow_audit_runtime_events;
use super::super::read_model;
use super::super::{new_id, timestamp};
use super::events::{
    WORKFLOW_RUN_CANCELED, WORKFLOW_RUN_FAILED, WORKFLOW_RUN_RETRYING, WORKFLOW_STEP_CANCELED,
    WORKFLOW_STEP_FAILED, WORKFLOW_STEP_PROGRESS, WORKFLOW_STEP_RETRYING,
};
use super::read_model::{WorkflowReadModel, WorkflowRunReadModel, WorkflowStepReadModel};
use super::status::WorkflowStatus;
use crate::runtime::TurnStartRequest;
use crate::{RuntimeCore, RuntimeCoreError, RuntimeEvent, RuntimeHostContext};
use app_server_protocol::{
    AgentSessionActionRespondParams, AgentSessionActionScope, AgentSessionActionType,
    AgentSessionApprovalDecision, AgentSessionReadParams, RuntimeOptions, WorkflowCancelParams,
    WorkflowCancelResponse, WorkflowRespondParams, WorkflowRespondResponse, WorkflowRetryParams,
    WorkflowRetryResponse,
};
use serde_json::{json, Map, Value};

impl RuntimeCore {
    pub async fn cancel_workflow_current(
        &self,
        params: WorkflowCancelParams,
    ) -> Result<WorkflowCancelResponse, RuntimeCoreError> {
        let target = WorkflowControlTarget::from_cancel_params(params)?;
        let context = self
            .load_session_current(AgentSessionReadParams {
                session_id: target.session_id.clone(),
                history_limit: None,
                history_offset: None,
                history_before_message_id: None,
            })
            .await?;
        let workflow_read_model = read_model::workflow_read_model_from_stored_session(
            &context.stored,
            &context.workflow_audit_events,
        );
        let Some(run) = workflow_read_model
            .workflow_runs
            .iter()
            .find(|run| run.workflow_run_id == target.workflow_run_id)
            .cloned()
        else {
            return Err(RuntimeCoreError::Backend(format!(
                "workflow run not found: {}",
                target.workflow_run_id
            )));
        };

        if run.status.is_terminal() {
            return workflow_cancel_response(
                context.stored.session.session_id,
                &workflow_read_model,
            );
        }

        let Some(event_log_writer) = self.event_log_writer.as_deref() else {
            return Err(RuntimeCoreError::Backend(
                "workflow/cancel requires workflow audit log writer".to_string(),
            ));
        };

        let steps = steps_to_cancel(&workflow_read_model, &target)?;
        let canceled_at = timestamp();
        let mut events = steps
            .iter()
            .map(|step| {
                RuntimeEvent::new(
                    WORKFLOW_STEP_CANCELED,
                    canceled_step_payload(step, &run, &target, &canceled_at),
                )
            })
            .collect::<Vec<_>>();
        events.push(RuntimeEvent::new(
            WORKFLOW_RUN_CANCELED,
            canceled_run_payload(&run, &target, &canceled_at),
        ));

        append_workflow_audit_runtime_events(
            Some(event_log_writer),
            &context.stored.session.session_id,
            &context.stored.session.thread_id,
            run.turn_id.as_deref(),
            events,
        )?;

        let workflow_audit_events =
            self.read_workflow_audit_events_for_session(&context.stored.session.session_id)?;
        let workflow_read_model = read_model::workflow_read_model_from_stored_session(
            &context.stored,
            &workflow_audit_events,
        );
        workflow_cancel_response(context.stored.session.session_id, &workflow_read_model)
    }

    pub async fn retry_workflow_current(
        &self,
        params: WorkflowRetryParams,
        host: RuntimeHostContext,
    ) -> Result<WorkflowRetryResponse, RuntimeCoreError> {
        let target = WorkflowControlTarget::from_retry_params(params)?;
        let context = self
            .load_session_current(AgentSessionReadParams {
                session_id: target.session_id.clone(),
                history_limit: None,
                history_offset: None,
                history_before_message_id: None,
            })
            .await?;
        let workflow_read_model = read_model::workflow_read_model_from_stored_session(
            &context.stored,
            &context.workflow_audit_events,
        );
        let Some(run) = workflow_read_model
            .workflow_runs
            .iter()
            .find(|run| run.workflow_run_id == target.workflow_run_id)
            .cloned()
        else {
            return Err(RuntimeCoreError::Backend(format!(
                "workflow run not found: {}",
                target.workflow_run_id
            )));
        };

        if retry_target_already_running(&workflow_read_model, &target, &run) {
            return workflow_retry_response(
                context.stored.session.session_id,
                &workflow_read_model,
                None,
            );
        }

        let Some(event_log_writer) = self.event_log_writer.as_deref() else {
            return Err(RuntimeCoreError::Backend(
                "workflow/retry requires workflow audit log writer".to_string(),
            ));
        };

        let steps = steps_to_retry(&workflow_read_model, &target, &run)?;
        let retried_at = timestamp();
        let reschedule =
            workflow_retry_reschedule_plan(&context.stored, &run, &target, &retried_at)?;
        let mut events = steps
            .iter()
            .map(|step| {
                RuntimeEvent::new(
                    WORKFLOW_STEP_RETRYING,
                    retrying_step_payload(step, &run, &target, &reschedule, &retried_at),
                )
            })
            .collect::<Vec<_>>();
        events.push(RuntimeEvent::new(
            WORKFLOW_RUN_RETRYING,
            retrying_run_payload(&run, &target, &reschedule, &retried_at),
        ));

        append_workflow_audit_runtime_events(
            Some(event_log_writer),
            &context.stored.session.session_id,
            &context.stored.session.thread_id,
            run.turn_id.as_deref(),
            events,
        )?;

        let start_result = self
            .start_turn_inner(
                reschedule.start_params(&context.stored.session.session_id),
                host,
                None,
                false,
                false,
                super::super::turn_start::TurnStartInputKind::User,
            )
            .await;
        if let Err(error) = start_result {
            let failed_at = timestamp();
            append_workflow_audit_runtime_events(
                Some(event_log_writer),
                &context.stored.session.session_id,
                &context.stored.session.thread_id,
                run.turn_id.as_deref(),
                workflow_retry_reschedule_failed_events(
                    &steps,
                    &run,
                    &target,
                    &reschedule,
                    &failed_at,
                    &error,
                ),
            )?;
            return Err(error);
        }

        let workflow_audit_events =
            self.read_workflow_audit_events_for_session(&context.stored.session.session_id)?;
        let workflow_read_model = read_model::workflow_read_model_from_stored_session(
            &context.stored,
            &workflow_audit_events,
        );
        workflow_retry_response(
            context.stored.session.session_id,
            &workflow_read_model,
            Some(reschedule.rescheduled_turn_id),
        )
    }

    pub async fn respond_workflow_current(
        &self,
        params: WorkflowRespondParams,
        host: RuntimeHostContext,
    ) -> Result<WorkflowRespondResponse, RuntimeCoreError> {
        let target = WorkflowRespondTarget::from_params(params)?;
        let context = self
            .load_session_current(AgentSessionReadParams {
                session_id: target.session_id.clone(),
                history_limit: None,
                history_offset: None,
                history_before_message_id: None,
            })
            .await?;
        let workflow_read_model = read_model::workflow_read_model_from_stored_session(
            &context.stored,
            &context.workflow_audit_events,
        );
        let Some(run) = workflow_read_model
            .workflow_runs
            .iter()
            .find(|run| run.workflow_run_id == target.workflow_run_id)
            .cloned()
        else {
            return Err(RuntimeCoreError::Backend(format!(
                "workflow run not found: {}",
                target.workflow_run_id
            )));
        };
        if run.status.is_terminal() {
            return Err(RuntimeCoreError::Backend(format!(
                "terminal workflow run cannot accept response: {}",
                run.workflow_run_id
            )));
        }

        let step = step_to_respond(&workflow_read_model, &target)?;
        let request_id = target
            .request_id
            .clone()
            .or_else(|| step.request_id.clone())
            .ok_or_else(|| {
                RuntimeCoreError::Backend(
                    "workflow/respond requires requestId from params or waiting step".to_string(),
                )
            })?;
        let pending_action =
            read_model::pending_action_from_stored_session(&context.stored, &request_id);
        let action_type = target
            .action_type
            .or_else(|| {
                step.agent_action_type
                    .as_deref()
                    .and_then(agent_action_type_from_str)
            })
            .or_else(|| pending_action.as_ref().map(|action| action.action_type))
            .ok_or_else(|| {
                RuntimeCoreError::Backend(
                    "workflow/respond requires actionType from params, waiting step, or action replay"
                        .to_string(),
                )
            })?;
        let action_scope = pending_action
            .and_then(|action| action.scope)
            .or_else(|| workflow_action_scope(&context.stored.session.thread_id, &run, &target));
        let decision = match action_type {
            AgentSessionActionType::ToolConfirmation => {
                if target.confirmed {
                    Some(AgentSessionApprovalDecision::AllowOnce)
                } else {
                    Some(AgentSessionApprovalDecision::Decline)
                }
            }
            AgentSessionActionType::AskUser | AgentSessionActionType::Elicitation => None,
        };

        let Some(event_log_writer) = self.event_log_writer.as_deref() else {
            return Err(RuntimeCoreError::Backend(
                "workflow/respond requires workflow audit log writer".to_string(),
            ));
        };

        self.respond_action(
            AgentSessionActionRespondParams {
                session_id: context.stored.session.session_id.clone(),
                request_id: request_id.clone(),
                action_type,
                decision,
                confirmed: Some(target.confirmed),
                response: target.response_text(),
                user_data: target.response_user_data(),
                metadata: Some(json!({
                    "source": "workflow/respond",
                    "workflowResume": {
                        "workflowRunId": target.workflow_run_id.clone(),
                        "workflowKey": run.workflow_key.clone(),
                        "stepId": step.step_id.clone(),
                    },
                })),
                event_name: None,
                action_scope,
            },
            host,
        )
        .await?;

        let responded_at = timestamp();
        append_workflow_audit_runtime_events(
            Some(event_log_writer),
            &context.stored.session.session_id,
            &context.stored.session.thread_id,
            run.turn_id.as_deref(),
            vec![RuntimeEvent::new(
                WORKFLOW_STEP_PROGRESS,
                responded_step_payload(&step, &run, &request_id, &target, &responded_at),
            )],
        )?;

        let workflow_audit_events =
            self.read_workflow_audit_events_for_session(&context.stored.session.session_id)?;
        let workflow_read_model = read_model::workflow_read_model_from_stored_session(
            &context.stored,
            &workflow_audit_events,
        );
        workflow_respond_response(context.stored.session.session_id, &workflow_read_model)
    }
}

#[derive(Debug, Clone)]
struct WorkflowControlTarget {
    session_id: String,
    workflow_run_id: String,
    step_id: Option<String>,
    reason_code: String,
    reason: Option<String>,
}

#[derive(Debug, Clone)]
struct WorkflowRespondTarget {
    session_id: String,
    workflow_run_id: String,
    step_id: Option<String>,
    request_id: Option<String>,
    action_type: Option<AgentSessionActionType>,
    confirmed: bool,
    response: Option<Value>,
}

#[derive(Debug, Clone)]
struct WorkflowRetryReschedulePlan {
    source_turn_id: String,
    rescheduled_turn_id: String,
    input: Vec<agent_protocol::AgentInput>,
    runtime_options: RuntimeOptions,
}

impl WorkflowControlTarget {
    fn from_cancel_params(params: WorkflowCancelParams) -> Result<Self, RuntimeCoreError> {
        Ok(Self {
            session_id: required_field(params.session_id, "sessionId", "workflow/cancel")?,
            workflow_run_id: required_field(
                params.workflow_run_id,
                "workflowRunId",
                "workflow/cancel",
            )?,
            step_id: optional_field(params.step_id),
            reason_code: optional_field(params.reason_code)
                .unwrap_or_else(|| "workflow_canceled".to_string()),
            reason: optional_field(params.reason),
        })
    }

    fn from_retry_params(params: WorkflowRetryParams) -> Result<Self, RuntimeCoreError> {
        Ok(Self {
            session_id: required_field(params.session_id, "sessionId", "workflow/retry")?,
            workflow_run_id: required_field(
                params.workflow_run_id,
                "workflowRunId",
                "workflow/retry",
            )?,
            step_id: optional_field(params.step_id),
            reason_code: optional_field(params.reason_code)
                .unwrap_or_else(|| "workflow_retry_requested".to_string()),
            reason: optional_field(params.reason),
        })
    }
}

impl WorkflowRetryReschedulePlan {
    fn start_params(&self, session_id: &str) -> TurnStartRequest {
        TurnStartRequest {
            session_id: session_id.to_string(),
            turn_id: Some(self.rescheduled_turn_id.clone()),
            input: self.input.clone(),
            runtime_options: Some(self.runtime_options.clone()),
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        }
    }
}

impl WorkflowRespondTarget {
    fn from_params(params: WorkflowRespondParams) -> Result<Self, RuntimeCoreError> {
        Ok(Self {
            session_id: required_field(params.session_id, "sessionId", "workflow/respond")?,
            workflow_run_id: required_field(
                params.workflow_run_id,
                "workflowRunId",
                "workflow/respond",
            )?,
            step_id: optional_field(params.step_id),
            request_id: optional_field(params.request_id),
            action_type: params.action_type,
            confirmed: params.confirmed.unwrap_or(true),
            response: params.response,
        })
    }

    fn response_text(&self) -> Option<String> {
        match self.response.as_ref()? {
            Value::String(value) => Some(value.clone()),
            Value::Object(object) => object
                .get("response")
                .or_else(|| object.get("answer"))
                .and_then(Value::as_str)
                .map(ToString::to_string),
            _ => None,
        }
    }

    fn response_user_data(&self) -> Option<Value> {
        match self.response.as_ref()? {
            Value::String(_) => None,
            value => Some(value.clone()),
        }
    }
}

fn required_field(
    value: String,
    field_name: &str,
    method: &str,
) -> Result<String, RuntimeCoreError> {
    let value = value.trim().to_string();
    if value.is_empty() {
        return Err(RuntimeCoreError::Backend(format!(
            "{field_name} is required for {method}"
        )));
    }
    Ok(value)
}

fn optional_field(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn steps_to_cancel(
    workflow_read_model: &WorkflowReadModel,
    target: &WorkflowControlTarget,
) -> Result<Vec<WorkflowStepReadModel>, RuntimeCoreError> {
    if let Some(step_id) = target.step_id.as_deref() {
        let Some(step) = workflow_read_model
            .workflow_steps
            .iter()
            .find(|step| step.workflow_run_id == target.workflow_run_id && step.step_id == step_id)
        else {
            return Err(RuntimeCoreError::Backend(format!(
                "workflow step not found: {step_id}"
            )));
        };
        if step.status.is_terminal() {
            return Ok(Vec::new());
        }
        return Ok(vec![step.clone()]);
    }

    let mut steps = workflow_read_model
        .workflow_steps
        .iter()
        .filter(|step| step.workflow_run_id == target.workflow_run_id)
        .filter(|step| !step.status.is_terminal())
        .cloned()
        .collect::<Vec<_>>();
    steps.sort_by(|left, right| {
        left.index
            .cmp(&right.index)
            .then_with(|| left.updated_at.cmp(&right.updated_at))
            .then_with(|| left.step_id.cmp(&right.step_id))
    });
    Ok(steps)
}

fn retry_target_already_running(
    workflow_read_model: &WorkflowReadModel,
    target: &WorkflowControlTarget,
    run: &WorkflowRunReadModel,
) -> bool {
    if let Some(step_id) = target.step_id.as_deref() {
        return workflow_read_model.workflow_steps.iter().any(|step| {
            step.workflow_run_id == target.workflow_run_id
                && step.step_id == step_id
                && step.status == WorkflowStatus::Retrying
        });
    }
    run.status == WorkflowStatus::Retrying
}

fn steps_to_retry(
    workflow_read_model: &WorkflowReadModel,
    target: &WorkflowControlTarget,
    run: &WorkflowRunReadModel,
) -> Result<Vec<WorkflowStepReadModel>, RuntimeCoreError> {
    if let Some(step_id) = target.step_id.as_deref() {
        let Some(step) = workflow_read_model
            .workflow_steps
            .iter()
            .find(|step| step.workflow_run_id == target.workflow_run_id && step.step_id == step_id)
        else {
            return Err(RuntimeCoreError::Backend(format!(
                "workflow step not found: {step_id}"
            )));
        };
        ensure_step_retryable(step)?;
        return Ok(vec![step.clone()]);
    }

    let mut steps = workflow_read_model
        .workflow_steps
        .iter()
        .filter(|step| step.workflow_run_id == target.workflow_run_id)
        .filter(|step| workflow_status_can_retry(step.status))
        .cloned()
        .collect::<Vec<_>>();
    steps.sort_by(|left, right| {
        left.index
            .cmp(&right.index)
            .then_with(|| left.updated_at.cmp(&right.updated_at))
            .then_with(|| left.step_id.cmp(&right.step_id))
    });
    if !steps.is_empty() {
        return Ok(steps);
    }

    if workflow_status_can_retry(run.status) {
        return Ok(Vec::new());
    }
    if run.status == WorkflowStatus::Completed {
        return Err(RuntimeCoreError::Backend(format!(
            "completed workflow run is not retryable: {}",
            run.workflow_run_id
        )));
    }
    Err(RuntimeCoreError::Backend(format!(
        "workflow run is not retryable until a failed, canceled, or skipped run/step exists: {}",
        run.workflow_run_id
    )))
}

fn step_to_respond(
    workflow_read_model: &WorkflowReadModel,
    target: &WorkflowRespondTarget,
) -> Result<WorkflowStepReadModel, RuntimeCoreError> {
    let mut waiting_steps = workflow_read_model
        .workflow_steps
        .iter()
        .filter(|step| step.workflow_run_id == target.workflow_run_id)
        .filter(|step| step.status == WorkflowStatus::Waiting)
        .filter(|step| {
            target
                .step_id
                .as_deref()
                .is_none_or(|step_id| step.step_id == step_id)
        })
        .cloned()
        .collect::<Vec<_>>();
    waiting_steps.sort_by(|left, right| {
        left.index
            .cmp(&right.index)
            .then_with(|| left.updated_at.cmp(&right.updated_at))
            .then_with(|| left.step_id.cmp(&right.step_id))
    });

    if target.step_id.is_some() && waiting_steps.is_empty() {
        return Err(RuntimeCoreError::Backend(format!(
            "workflow step is not waiting for response: {}",
            target.step_id.as_deref().unwrap_or_default()
        )));
    }
    if waiting_steps.len() > 1 {
        return Err(RuntimeCoreError::Backend(
            "workflow/respond requires stepId when multiple waiting steps exist".to_string(),
        ));
    }
    waiting_steps.into_iter().next().ok_or_else(|| {
        RuntimeCoreError::Backend(format!(
            "workflow run has no waiting step to respond: {}",
            target.workflow_run_id
        ))
    })
}

fn ensure_step_retryable(step: &WorkflowStepReadModel) -> Result<(), RuntimeCoreError> {
    if workflow_status_can_retry(step.status) {
        return Ok(());
    }
    if step.status == WorkflowStatus::Completed {
        return Err(RuntimeCoreError::Backend(format!(
            "completed workflow step is not retryable: {}",
            step.step_id
        )));
    }
    Err(RuntimeCoreError::Backend(format!(
        "workflow step is not retryable until it reaches failed, canceled, or skipped: {}",
        step.step_id
    )))
}

fn workflow_status_can_retry(status: WorkflowStatus) -> bool {
    matches!(
        status,
        WorkflowStatus::Failed | WorkflowStatus::Canceled | WorkflowStatus::Skipped
    )
}

fn workflow_retry_reschedule_plan(
    stored: &crate::runtime::StoredSession,
    run: &WorkflowRunReadModel,
    target: &WorkflowControlTarget,
    retried_at: &str,
) -> Result<WorkflowRetryReschedulePlan, RuntimeCoreError> {
    let source_turn_id = run.turn_id.clone().ok_or_else(|| {
        RuntimeCoreError::Backend(format!(
            "workflow/retry requires source turnId to reschedule executor: {}",
            run.workflow_run_id
        ))
    })?;
    let input = stored
        .turn_inputs
        .get(&source_turn_id)
        .cloned()
        .ok_or_else(|| {
            RuntimeCoreError::Backend(format!(
                "workflow/retry cannot reschedule missing source turn input: {source_turn_id}"
            ))
        })?;
    let runtime_options = stored.turn_runtime_options.get(&source_turn_id).cloned();
    let rescheduled_turn_id = new_id("turn");
    Ok(WorkflowRetryReschedulePlan {
        source_turn_id,
        rescheduled_turn_id,
        input,
        runtime_options: retry_runtime_options(runtime_options, target, run, retried_at),
    })
}

fn retry_runtime_options(
    runtime_options: Option<RuntimeOptions>,
    target: &WorkflowControlTarget,
    run: &WorkflowRunReadModel,
    retried_at: &str,
) -> RuntimeOptions {
    let mut runtime_options = runtime_options.unwrap_or_default();
    let mut metadata = runtime_options
        .runtime_metadata_mut()
        .take()
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}));
    let retry = json!({
        "source": "workflow/retry",
        "workflowRunId": target.workflow_run_id.clone(),
        "stepId": target.step_id.clone(),
        "reasonCode": target.reason_code.clone(),
        "reason": target.reason.clone(),
        "sourceTurnId": run.turn_id.clone(),
        "retriedAt": retried_at,
    });
    if let Some(object) = metadata.as_object_mut() {
        object.insert("workflowRetry".to_string(), retry.clone());
        object.insert("workflow_retry".to_string(), retry);
    }
    *runtime_options.runtime_metadata_mut() = Some(metadata);
    runtime_options
}

fn agent_action_type_from_str(value: &str) -> Option<AgentSessionActionType> {
    match value.trim() {
        "tool_confirmation" => Some(AgentSessionActionType::ToolConfirmation),
        "ask_user" => Some(AgentSessionActionType::AskUser),
        "elicitation" => Some(AgentSessionActionType::Elicitation),
        _ => None,
    }
}

fn workflow_action_scope(
    thread_id: &str,
    run: &WorkflowRunReadModel,
    target: &WorkflowRespondTarget,
) -> Option<AgentSessionActionScope> {
    let scope = AgentSessionActionScope {
        session_id: Some(target.session_id.clone()),
        thread_id: Some(thread_id.to_string()),
        turn_id: run.turn_id.clone(),
    };
    if scope.session_id.is_none() && scope.thread_id.is_none() && scope.turn_id.is_none() {
        None
    } else {
        Some(scope)
    }
}

fn canceled_run_payload(
    run: &WorkflowRunReadModel,
    target: &WorkflowControlTarget,
    canceled_at: &str,
) -> Value {
    let mut object = Map::new();
    object.insert(
        "workflowRunId".to_string(),
        json!(run.workflow_run_id.clone()),
    );
    insert_opt(&mut object, "workflowKey", run.workflow_key.clone());
    insert_opt(&mut object, "workflowTitle", run.title.clone());
    insert_opt(&mut object, "taskId", run.task_id.clone());
    insert_opt(&mut object, "turnId", run.turn_id.clone());
    insert_opt(&mut object, "appId", run.app_id.clone());
    insert_opt(&mut object, "sourceKind", run.source_kind.clone());
    object.insert("status".to_string(), json!("canceled"));
    object.insert("updatedAt".to_string(), json!(canceled_at));
    object.insert("finishedAt".to_string(), json!(canceled_at));
    object.insert(
        "cancellation".to_string(),
        cancellation_payload(target, canceled_at),
    );
    object.insert(
        "metadata".to_string(),
        json!({
            "pluginWorkflow": {
                "status": "canceled"
            }
        }),
    );
    Value::Object(object)
}

fn canceled_step_payload(
    step: &WorkflowStepReadModel,
    run: &WorkflowRunReadModel,
    target: &WorkflowControlTarget,
    canceled_at: &str,
) -> Value {
    let mut object = Map::new();
    object.insert(
        "workflowRunId".to_string(),
        json!(step.workflow_run_id.clone()),
    );
    insert_opt(&mut object, "workflowKey", run.workflow_key.clone());
    object.insert("stepId".to_string(), json!(step.step_id.clone()));
    object.insert("stepTitle".to_string(), json!(step.title.clone()));
    insert_opt(&mut object, "stepKind", step.kind.clone());
    if let Some(index) = step.index {
        object.insert("stepIndex".to_string(), json!(index));
    }
    if let Some(step_count) = step.step_count {
        object.insert("stepCount".to_string(), json!(step_count));
    }
    object.insert("attempt".to_string(), json!(step.attempt));
    object.insert("status".to_string(), json!("canceled"));
    object.insert("updatedAt".to_string(), json!(canceled_at));
    object.insert("finishedAt".to_string(), json!(canceled_at));
    object.insert(
        "cancellation".to_string(),
        cancellation_payload(target, canceled_at),
    );
    Value::Object(object)
}

fn retrying_run_payload(
    run: &WorkflowRunReadModel,
    target: &WorkflowControlTarget,
    reschedule: &WorkflowRetryReschedulePlan,
    retried_at: &str,
) -> Value {
    let mut object = Map::new();
    object.insert(
        "workflowRunId".to_string(),
        json!(run.workflow_run_id.clone()),
    );
    insert_opt(&mut object, "workflowKey", run.workflow_key.clone());
    insert_opt(&mut object, "workflowTitle", run.title.clone());
    insert_opt(&mut object, "taskId", run.task_id.clone());
    insert_opt(&mut object, "turnId", run.turn_id.clone());
    insert_opt(&mut object, "appId", run.app_id.clone());
    insert_opt(&mut object, "sourceKind", run.source_kind.clone());
    object.insert("status".to_string(), json!("retrying"));
    object.insert("updatedAt".to_string(), json!(retried_at));
    object.insert(
        "retry".to_string(),
        retry_payload(target, reschedule, retried_at),
    );
    object.insert(
        "metadata".to_string(),
        json!({
            "pluginWorkflow": {
                "status": "retrying",
                "sourceTurnId": reschedule.source_turn_id,
                "rescheduledTurnId": reschedule.rescheduled_turn_id
            }
        }),
    );
    Value::Object(object)
}

fn retrying_step_payload(
    step: &WorkflowStepReadModel,
    run: &WorkflowRunReadModel,
    target: &WorkflowControlTarget,
    reschedule: &WorkflowRetryReschedulePlan,
    retried_at: &str,
) -> Value {
    let mut object = Map::new();
    object.insert(
        "workflowRunId".to_string(),
        json!(step.workflow_run_id.clone()),
    );
    insert_opt(&mut object, "workflowKey", run.workflow_key.clone());
    object.insert("stepId".to_string(), json!(step.step_id.clone()));
    object.insert("stepTitle".to_string(), json!(step.title.clone()));
    insert_opt(&mut object, "stepKind", step.kind.clone());
    if let Some(index) = step.index {
        object.insert("stepIndex".to_string(), json!(index));
    }
    if let Some(step_count) = step.step_count {
        object.insert("stepCount".to_string(), json!(step_count));
    }
    object.insert("previousAttempt".to_string(), json!(step.attempt));
    object.insert("attempt".to_string(), json!(step.attempt + 1));
    object.insert("status".to_string(), json!("retrying"));
    object.insert("updatedAt".to_string(), json!(retried_at));
    object.insert(
        "retry".to_string(),
        retry_payload(target, reschedule, retried_at),
    );
    Value::Object(object)
}

fn responded_step_payload(
    step: &WorkflowStepReadModel,
    run: &WorkflowRunReadModel,
    request_id: &str,
    target: &WorkflowRespondTarget,
    responded_at: &str,
) -> Value {
    let mut object = Map::new();
    object.insert(
        "workflowRunId".to_string(),
        json!(step.workflow_run_id.clone()),
    );
    insert_opt(&mut object, "workflowKey", run.workflow_key.clone());
    object.insert("stepId".to_string(), json!(step.step_id.clone()));
    object.insert("stepTitle".to_string(), json!(step.title.clone()));
    insert_opt(&mut object, "stepKind", step.kind.clone());
    if let Some(index) = step.index {
        object.insert("stepIndex".to_string(), json!(index));
    }
    if let Some(step_count) = step.step_count {
        object.insert("stepCount".to_string(), json!(step_count));
    }
    object.insert("attempt".to_string(), json!(step.attempt));
    object.insert("status".to_string(), json!("running"));
    object.insert("updatedAt".to_string(), json!(responded_at));
    object.insert("requestId".to_string(), json!(request_id));
    if let Some(action_type) = target.action_type.or_else(|| {
        step.agent_action_type
            .as_deref()
            .and_then(agent_action_type_from_str)
    }) {
        object.insert("actionType".to_string(), json!(action_type));
    }
    object.insert(
        "response".to_string(),
        response_payload(target, request_id, responded_at),
    );
    Value::Object(object)
}

fn cancellation_payload(target: &WorkflowControlTarget, canceled_at: &str) -> Value {
    let mut object = Map::new();
    object.insert("source".to_string(), json!("workflow/cancel"));
    object.insert("reasonCode".to_string(), json!(target.reason_code.clone()));
    if let Some(reason) = target.reason.clone() {
        object.insert("reason".to_string(), json!(reason));
    }
    object.insert("canceledAt".to_string(), json!(canceled_at));
    Value::Object(object)
}

fn retry_payload(
    target: &WorkflowControlTarget,
    reschedule: &WorkflowRetryReschedulePlan,
    retried_at: &str,
) -> Value {
    let mut object = Map::new();
    object.insert("source".to_string(), json!("workflow/retry"));
    object.insert("reasonCode".to_string(), json!(target.reason_code.clone()));
    if let Some(reason) = target.reason.clone() {
        object.insert("reason".to_string(), json!(reason));
    }
    if let Some(step_id) = target.step_id.clone() {
        object.insert("stepId".to_string(), json!(step_id));
    }
    object.insert(
        "sourceTurnId".to_string(),
        json!(reschedule.source_turn_id.clone()),
    );
    object.insert(
        "rescheduledTurnId".to_string(),
        json!(reschedule.rescheduled_turn_id.clone()),
    );
    object.insert("retriedAt".to_string(), json!(retried_at));
    Value::Object(object)
}

fn response_payload(target: &WorkflowRespondTarget, request_id: &str, responded_at: &str) -> Value {
    let mut object = Map::new();
    object.insert("source".to_string(), json!("workflow/respond"));
    object.insert("requestId".to_string(), json!(request_id));
    object.insert("confirmed".to_string(), json!(target.confirmed));
    if let Some(response) = target.response.clone() {
        object.insert("payload".to_string(), response);
    }
    object.insert("respondedAt".to_string(), json!(responded_at));
    Value::Object(object)
}

fn workflow_retry_reschedule_failed_events(
    steps: &[WorkflowStepReadModel],
    run: &WorkflowRunReadModel,
    target: &WorkflowControlTarget,
    reschedule: &WorkflowRetryReschedulePlan,
    failed_at: &str,
    error: &RuntimeCoreError,
) -> Vec<RuntimeEvent> {
    let failure = json!({
        "source": "workflow/retry",
        "reasonCode": "workflow_retry_reschedule_failed",
        "message": error.to_string(),
        "retry": retry_payload(target, reschedule, failed_at),
    });
    let mut events = steps
        .iter()
        .map(|step| {
            RuntimeEvent::new(
                WORKFLOW_STEP_FAILED,
                retry_reschedule_failed_step_payload(
                    step, run, target, reschedule, failed_at, &failure,
                ),
            )
        })
        .collect::<Vec<_>>();
    events.push(RuntimeEvent::new(
        WORKFLOW_RUN_FAILED,
        retry_reschedule_failed_run_payload(run, target, reschedule, failed_at, &failure),
    ));
    events
}

fn retry_reschedule_failed_run_payload(
    run: &WorkflowRunReadModel,
    target: &WorkflowControlTarget,
    reschedule: &WorkflowRetryReschedulePlan,
    failed_at: &str,
    failure: &Value,
) -> Value {
    let mut object = Map::new();
    object.insert(
        "workflowRunId".to_string(),
        json!(run.workflow_run_id.clone()),
    );
    insert_opt(&mut object, "workflowKey", run.workflow_key.clone());
    insert_opt(&mut object, "workflowTitle", run.title.clone());
    insert_opt(&mut object, "taskId", run.task_id.clone());
    insert_opt(&mut object, "turnId", run.turn_id.clone());
    insert_opt(&mut object, "appId", run.app_id.clone());
    insert_opt(&mut object, "sourceKind", run.source_kind.clone());
    object.insert("status".to_string(), json!("failed"));
    object.insert("updatedAt".to_string(), json!(failed_at));
    object.insert("finishedAt".to_string(), json!(failed_at));
    object.insert("failure".to_string(), failure.clone());
    object.insert(
        "retry".to_string(),
        retry_payload(target, reschedule, failed_at),
    );
    Value::Object(object)
}

fn retry_reschedule_failed_step_payload(
    step: &WorkflowStepReadModel,
    run: &WorkflowRunReadModel,
    target: &WorkflowControlTarget,
    reschedule: &WorkflowRetryReschedulePlan,
    failed_at: &str,
    failure: &Value,
) -> Value {
    let mut object = Map::new();
    object.insert(
        "workflowRunId".to_string(),
        json!(step.workflow_run_id.clone()),
    );
    insert_opt(&mut object, "workflowKey", run.workflow_key.clone());
    object.insert("stepId".to_string(), json!(step.step_id.clone()));
    object.insert("stepTitle".to_string(), json!(step.title.clone()));
    insert_opt(&mut object, "stepKind", step.kind.clone());
    if let Some(index) = step.index {
        object.insert("stepIndex".to_string(), json!(index));
    }
    if let Some(step_count) = step.step_count {
        object.insert("stepCount".to_string(), json!(step_count));
    }
    object.insert("attempt".to_string(), json!(step.attempt + 1));
    object.insert("status".to_string(), json!("failed"));
    object.insert("updatedAt".to_string(), json!(failed_at));
    object.insert("finishedAt".to_string(), json!(failed_at));
    object.insert("failure".to_string(), failure.clone());
    object.insert(
        "retry".to_string(),
        retry_payload(target, reschedule, failed_at),
    );
    Value::Object(object)
}

fn insert_opt(object: &mut Map<String, Value>, key: &str, value: Option<String>) {
    if let Some(value) = value {
        object.insert(key.to_string(), json!(value));
    }
}

fn workflow_cancel_response(
    session_id: String,
    workflow_read_model: &WorkflowReadModel,
) -> Result<WorkflowCancelResponse, RuntimeCoreError> {
    let workflow = workflow_value(workflow_read_model)?;
    let workflow_runs = array_field(&workflow, "workflowRuns");
    let workflow_steps = array_field(&workflow, "workflowSteps");
    Ok(WorkflowCancelResponse {
        session_id,
        workflow,
        workflow_runs,
        workflow_steps,
    })
}

fn workflow_retry_response(
    session_id: String,
    workflow_read_model: &WorkflowReadModel,
    rescheduled_turn_id: Option<String>,
) -> Result<WorkflowRetryResponse, RuntimeCoreError> {
    let workflow = workflow_value(workflow_read_model)?;
    let workflow_runs = array_field(&workflow, "workflowRuns");
    let workflow_steps = array_field(&workflow, "workflowSteps");
    Ok(WorkflowRetryResponse {
        session_id,
        workflow,
        workflow_runs,
        workflow_steps,
        rescheduled_turn_id,
    })
}

fn workflow_respond_response(
    session_id: String,
    workflow_read_model: &WorkflowReadModel,
) -> Result<WorkflowRespondResponse, RuntimeCoreError> {
    let workflow = workflow_value(workflow_read_model)?;
    let workflow_runs = array_field(&workflow, "workflowRuns");
    let workflow_steps = array_field(&workflow, "workflowSteps");
    Ok(WorkflowRespondResponse {
        session_id,
        workflow,
        workflow_runs,
        workflow_steps,
    })
}

fn workflow_value(workflow_read_model: &WorkflowReadModel) -> Result<Value, RuntimeCoreError> {
    serde_json::to_value(workflow_read_model).map_err(|error| {
        RuntimeCoreError::Backend(format!("failed to serialize workflow read model: {error}"))
    })
}

fn array_field(value: &Value, key: &str) -> Vec<Value> {
    value
        .get(key)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}
