use super::event_store::{
    append_runtime_events_to_state, append_runtime_events_to_state_with_message_lifecycle,
    append_workflow_audit_runtime_events, CanonicalMessageLifecycleState,
};
use super::status::{agent_turn_blocks_queue_resume, agent_turn_is_active, agent_turn_is_terminal};
use super::turn_start::{
    user_input_text, validate_user_input, TurnStartInputKind, TurnStartRequest,
};
use super::workflow::cancel::workflow_cancel_events_from_audit_records;
use super::workflow::events::{WORKFLOW_RUN_RESUMING, WORKFLOW_STEP_RESUMING};
use super::*;
use agent_protocol::{ItemStatus, ThreadId, ThreadItem, ThreadItemPayload, ThreadTurnsView};
use agent_runtime::session_loop::{
    RuntimeSessionClosureTask, RuntimeSessionHandle, RuntimeSessionInput,
    RuntimeSessionInputHandle, RuntimeSessionLoopError, RuntimeSessionSubmitResult,
    RuntimeSessionTaskFailure, RuntimeSessionTaskOutcome, RuntimeSessionUserInputResult,
};
use app_server_protocol::*;
use futures::future::BoxFuture;
use serde_json::json;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use thread_store::ReadThreadParams;
use tokio::sync::{mpsc, oneshot, Notify};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Default)]
pub(in crate::runtime) struct CollectingRuntimeEventSink {
    events: Vec<RuntimeEvent>,
    preappended_events: Vec<AgentEvent>,
}

impl CollectingRuntimeEventSink {
    fn emitted_count(&self) -> usize {
        self.events.len() + self.preappended_events.len()
    }

    fn has_turn_terminal_event(&self) -> bool {
        self.events
            .iter()
            .any(|event| runtime_event_is_turn_terminal(&event.event_type))
    }

    fn into_events(self) -> Vec<RuntimeEvent> {
        self.events
    }

    fn into_parts(self) -> (Vec<RuntimeEvent>, Vec<AgentEvent>) {
        (self.events, self.preappended_events)
    }

    fn take_events(&mut self) -> Vec<RuntimeEvent> {
        std::mem::take(&mut self.events)
    }

    fn ensure_turn_accepted_event(&mut self) {
        if self
            .events
            .iter()
            .any(|event| event.event_type == "turn.accepted")
            || !self.events.iter().any(|event| {
                matches!(
                    event.event_type.as_str(),
                    "turn.started" | "turn.completed" | "turn.failed" | "turn.canceled"
                )
            })
        {
            return;
        }
        self.events.insert(
            0,
            RuntimeEvent::new(
                "turn.accepted",
                json!({
                    "backend": "app_server",
                    "source": "turn/start",
                }),
            ),
        );
    }

    fn emit_failure(&mut self, error: &RuntimeCoreError) -> Result<(), RuntimeCoreError> {
        self.emit(RuntimeEvent::new(
            "turn.failed",
            json!({
                "message": error.to_string(),
                "reason": error.turn_failure_reason(),
            }),
        ))
    }
}

fn runtime_event_is_turn_terminal(event_type: &str) -> bool {
    matches!(
        event_type,
        "turn.completed" | "turn.failed" | "turn.canceled"
    )
}

fn app_server_action_resolved_event(request: &ActionRespondRequest) -> RuntimeEvent {
    let mut payload = json!({
        "backend": "runtime_core",
        "source": "runtime_preflight",
        "requestId": request.request_id,
        "actionId": request.request_id,
        "actionType": request.action_type,
        "confirmed": request.confirmed,
        "response": request.response,
        "userData": request.user_data,
        "scope": request.action_scope,
    });
    if let Some(decision) = request.decision {
        if let Some(object) = payload.as_object_mut() {
            object.insert("decision".to_string(), json!(decision.as_str()));
            object.insert("decisionScope".to_string(), json!(decision.scope()));
        }
    }
    RuntimeEvent::new("action.resolved", payload)
}

fn app_server_action_canceled_event(request: &ActionRespondRequest) -> RuntimeEvent {
    RuntimeEvent::new(
        "action.canceled",
        json!({
            "backend": "runtime_core",
            "source": "runtime_preflight",
            "requestId": request.request_id,
            "actionId": request.request_id,
            "actionType": request.action_type,
            "confirmed": false,
            "decision": AgentSessionApprovalDecision::Cancel.as_str(),
            "scope": request.action_scope,
        }),
    )
}

fn turn_interrupt_action_canceled_event(
    request_id: &str,
    identity: &pending_action_descriptor::PendingActionIdentity,
) -> RuntimeEvent {
    RuntimeEvent::new(
        "action.canceled",
        json!({
            "backend": "runtime_core",
            "source": "turn/interrupt",
            "requestId": request_id,
            "actionId": request_id,
            "actionType": identity.action_type,
            "reason": "turn_interrupted",
            "scope": identity.scope,
        }),
    )
}

fn active_tool_item_for_turn(
    stored: &StoredSession,
    turn_id: &str,
    tool_call_id: &str,
) -> Option<ThreadItem> {
    stored.events.iter().rev().find_map(|event| {
        if !matches!(event.event_type.as_str(), "item.started" | "item.updated") {
            return None;
        }
        let item = serde_json::from_value::<ThreadItem>(event.payload.get("item")?.clone()).ok()?;
        if item.turn_id.as_str() != turn_id || item.status.is_terminal() {
            return None;
        }
        match &item.payload {
            ThreadItemPayload::Tool { call_id, .. } if call_id == tool_call_id => Some(item),
            _ => None,
        }
    })
}

fn turn_interrupt_tool_completed_event(mut item: ThreadItem) -> RuntimeEvent {
    let completed_at_ms = chrono::Utc::now()
        .timestamp_millis()
        .max(item.updated_at_ms);
    item.status = ItemStatus::Cancelled;
    item.updated_at_ms = completed_at_ms;
    item.completed_at_ms = Some(completed_at_ms);
    RuntimeEvent::new(
        "item.completed",
        json!({
            "item": item,
            "reason": "turn_interrupted",
        }),
    )
}

impl RuntimeEventSink for CollectingRuntimeEventSink {
    fn emit(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        self.events.push(event);
        Ok(())
    }

    fn emit_preappended(&mut self, event: AgentEvent) -> Result<(), RuntimeCoreError> {
        self.preappended_events.push(event);
        Ok(())
    }
}

enum SessionLoopEvent {
    Runtime(RuntimeEvent),
    Transient(RuntimeEvent),
    Preappended(AgentEvent),
}

struct ChannelRuntimeEventSink {
    sender: mpsc::UnboundedSender<SessionLoopEvent>,
}

struct SubmittedRuntimeSessionTurn {
    session: RuntimeSessionHandle,
    session_id: String,
    turn_id: String,
    event_receiver: mpsc::UnboundedReceiver<SessionLoopEvent>,
    completion: oneshot::Receiver<Result<RuntimeSessionTaskOutcome, RuntimeSessionTaskFailure>>,
    driver_completion: Arc<RuntimeTurnDriverCompletion>,
    driver_completions: RuntimeTurnDriverCompletions,
}

impl Drop for SubmittedRuntimeSessionTurn {
    fn drop(&mut self) {
        self.driver_completion.complete();
        self.driver_completions
            .remove(&self.session_id, &self.turn_id, &self.driver_completion);
    }
}

#[derive(Clone, Default)]
pub(in crate::runtime) struct RuntimeTurnDriverCompletions {
    inner: Arc<Mutex<HashMap<(String, String), Arc<RuntimeTurnDriverCompletion>>>>,
}

impl RuntimeTurnDriverCompletions {
    fn register(&self, session_id: &str, turn_id: &str) -> Arc<RuntimeTurnDriverCompletion> {
        let completion = Arc::new(RuntimeTurnDriverCompletion::default());
        self.inner
            .lock()
            .expect("runtime turn driver completions mutex poisoned")
            .insert(
                (session_id.to_string(), turn_id.to_string()),
                Arc::clone(&completion),
            );
        completion
    }

    fn get(&self, session_id: &str, turn_id: &str) -> Option<Arc<RuntimeTurnDriverCompletion>> {
        self.inner
            .lock()
            .expect("runtime turn driver completions mutex poisoned")
            .get(&(session_id.to_string(), turn_id.to_string()))
            .cloned()
    }

    fn remove(&self, session_id: &str, turn_id: &str, expected: &Arc<RuntimeTurnDriverCompletion>) {
        let mut completions = self
            .inner
            .lock()
            .expect("runtime turn driver completions mutex poisoned");
        let key = (session_id.to_string(), turn_id.to_string());
        if completions
            .get(&key)
            .is_some_and(|completion| Arc::ptr_eq(completion, expected))
        {
            completions.remove(&key);
        }
    }
}

#[derive(Default)]
struct RuntimeTurnDriverCompletion {
    completed: AtomicBool,
    notify: Notify,
}

impl RuntimeTurnDriverCompletion {
    async fn wait(&self) {
        loop {
            let notified = self.notify.notified();
            if self.completed.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    fn complete(&self) {
        if !self.completed.swap(true, Ordering::AcqRel) {
            self.notify.notify_waiters();
        }
    }
}

impl RuntimeEventSink for ChannelRuntimeEventSink {
    fn emit(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        self.sender
            .send(SessionLoopEvent::Runtime(event))
            .map_err(|_| {
                RuntimeCoreError::Backend("runtime session event receiver closed".to_string())
            })
    }

    fn emit_transient(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        self.sender
            .send(SessionLoopEvent::Transient(event))
            .map_err(|_| {
                RuntimeCoreError::Backend("runtime session event receiver closed".to_string())
            })
    }
}

fn workflow_resume_audit_events_from_action_response(
    params: &AgentSessionActionRespondParams,
    decision: Option<AgentSessionApprovalDecision>,
    confirmed: bool,
) -> Vec<RuntimeEvent> {
    let Some(metadata) = params.metadata.as_ref() else {
        return Vec::new();
    };
    let Some(binding) = workflow_resume_binding_from_action_metadata(metadata) else {
        return Vec::new();
    };
    let decision_text = decision
        .map(|decision| decision.as_str())
        .unwrap_or_else(|| if confirmed { "approved" } else { "rejected" });
    let decision_scope = decision.map(|decision| decision.scope()).unwrap_or("once");
    let status_decision = if confirmed { "approved" } else { "rejected" };
    let base_payload = json!({
        "workflowRunId": binding.workflow_run_id,
        "workflowKey": binding.workflow_key,
        "stepId": binding.step_id,
        "actionId": params.request_id,
        "decision": decision_text,
        "decisionScope": decision_scope,
        "statusDecision": status_decision,
        "status": "resuming",
        "source": "agentSession/action/respond",
    });
    vec![
        RuntimeEvent::new(WORKFLOW_STEP_RESUMING, base_payload.clone()),
        RuntimeEvent::new(WORKFLOW_RUN_RESUMING, base_payload),
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkflowResumeActionBinding {
    workflow_run_id: String,
    workflow_key: String,
    step_id: String,
}

fn workflow_resume_binding_from_action_metadata(
    metadata: &Value,
) -> Option<WorkflowResumeActionBinding> {
    for candidate in workflow_resume_action_metadata_candidates(metadata) {
        let Some(workflow_run_id) = string_field(
            candidate,
            &["workflowRunId", "workflow_run_id", "runId", "run_id"],
        ) else {
            continue;
        };
        let Some(workflow_key) = string_field(
            candidate,
            &["workflowKey", "workflow_key", "key", "workflow"],
        ) else {
            continue;
        };
        let Some(step_id) = string_field(candidate, &["stepId", "step_id", "id"]) else {
            continue;
        };
        return Some(WorkflowResumeActionBinding {
            workflow_run_id,
            workflow_key,
            step_id,
        });
    }
    None
}

fn workflow_resume_action_metadata_candidates(metadata: &Value) -> Vec<&Value> {
    let mut candidates = vec![metadata];
    for key in [
        "workflowResume",
        "workflow_resume",
        "workflowResumeLifecycle",
        "workflow_resume_lifecycle",
        "workerLifecycle",
        "worker_lifecycle",
        "pluginWorkflow",
        "plugin_workflow",
    ] {
        if let Some(candidate) = metadata.get(key) {
            candidates.push(candidate);
        }
    }
    candidates
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

enum RuntimeEventTarget<'a> {
    Callback(&'a mut RuntimeEventCallback<'a>),
    Hub(RuntimeEventHub),
}

impl RuntimeEventTarget<'_> {
    fn publish(&mut self, event: AgentEvent) -> Result<(), RuntimeCoreError> {
        match self {
            Self::Callback(callback) => callback(event),
            Self::Hub(_) if event.event_type == "turn.accepted" => Ok(()),
            Self::Hub(hub) => {
                hub.publish(event);
                Ok(())
            }
        }
    }
}

pub(in crate::runtime) struct AppendingRuntimeEventSink<'a> {
    state: Arc<Mutex<RuntimeCoreState>>,
    file_checkpoint_snapshot_store: Arc<dyn FileCheckpointSnapshotStore>,
    output_snapshot_store: Arc<dyn output_refs::OutputSnapshotStore>,
    sidecar_store: Option<Arc<SidecarStore>>,
    event_log_writer: Option<Arc<EventLogWriter>>,
    trace_event_writer: Option<Arc<TraceEventWriter>>,
    projection_store: Option<Arc<ProjectionStore>>,
    session_loops: agent_runtime::session_loop::RuntimeSessionRegistry,
    session_id: String,
    thread_id: String,
    turn_id: String,
    target: RuntimeEventTarget<'a>,
    events: Vec<AgentEvent>,
    deferred_hub_events: Vec<AgentEvent>,
    message_lifecycle: CanonicalMessageLifecycleState,
}

impl<'a> AppendingRuntimeEventSink<'a> {
    fn new(
        state: Arc<Mutex<RuntimeCoreState>>,
        file_checkpoint_snapshot_store: Arc<dyn FileCheckpointSnapshotStore>,
        output_snapshot_store: Arc<dyn output_refs::OutputSnapshotStore>,
        sidecar_store: Option<Arc<SidecarStore>>,
        event_log_writer: Option<Arc<EventLogWriter>>,
        trace_event_writer: Option<Arc<TraceEventWriter>>,
        projection_store: Option<Arc<ProjectionStore>>,
        session_loops: agent_runtime::session_loop::RuntimeSessionRegistry,
        session_id: String,
        thread_id: String,
        turn_id: String,
        callback: &'a mut RuntimeEventCallback<'a>,
    ) -> Self {
        let message_lifecycle = state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .get(&session_id)
            .map(|stored| CanonicalMessageLifecycleState::from_events(&stored.events, &turn_id))
            .unwrap_or_default();
        Self {
            state,
            file_checkpoint_snapshot_store,
            output_snapshot_store,
            sidecar_store,
            event_log_writer,
            trace_event_writer,
            projection_store,
            session_loops,
            session_id,
            thread_id,
            turn_id,
            target: RuntimeEventTarget::Callback(callback),
            events: Vec::new(),
            deferred_hub_events: Vec::new(),
            message_lifecycle,
        }
    }

    fn with_hub(
        state: Arc<Mutex<RuntimeCoreState>>,
        file_checkpoint_snapshot_store: Arc<dyn FileCheckpointSnapshotStore>,
        output_snapshot_store: Arc<dyn output_refs::OutputSnapshotStore>,
        sidecar_store: Option<Arc<SidecarStore>>,
        event_log_writer: Option<Arc<EventLogWriter>>,
        trace_event_writer: Option<Arc<TraceEventWriter>>,
        projection_store: Option<Arc<ProjectionStore>>,
        session_loops: agent_runtime::session_loop::RuntimeSessionRegistry,
        session_id: String,
        thread_id: String,
        turn_id: String,
        hub: RuntimeEventHub,
    ) -> AppendingRuntimeEventSink<'static> {
        let message_lifecycle = state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .get(&session_id)
            .map(|stored| CanonicalMessageLifecycleState::from_events(&stored.events, &turn_id))
            .unwrap_or_default();
        AppendingRuntimeEventSink {
            state,
            file_checkpoint_snapshot_store,
            output_snapshot_store,
            sidecar_store,
            event_log_writer,
            trace_event_writer,
            projection_store,
            session_loops,
            session_id,
            thread_id,
            turn_id,
            target: RuntimeEventTarget::Hub(hub),
            events: Vec::new(),
            deferred_hub_events: Vec::new(),
            message_lifecycle,
        }
    }

    fn defer_hub_events(&mut self, events: Vec<AgentEvent>) {
        self.deferred_hub_events = events;
    }

    fn flush_deferred_hub_events(&mut self) -> Result<(), RuntimeCoreError> {
        for event in std::mem::take(&mut self.deferred_hub_events) {
            self.target.publish(event)?;
        }
        Ok(())
    }

    fn emitted_count(&self) -> usize {
        self.events.len()
    }

    fn into_events(self) -> Vec<AgentEvent> {
        self.events
    }

    fn emit_failure(&mut self, error: &RuntimeCoreError) -> Result<(), RuntimeCoreError> {
        self.emit(RuntimeEvent::new(
            "turn.failed",
            json!({
                "message": error.to_string(),
                "reason": error.turn_failure_reason(),
            }),
        ))
    }
}

impl RuntimeEventSink for AppendingRuntimeEventSink<'_> {
    fn emit(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        if self.is_duplicate_accepted_event(&event) {
            return Ok(());
        }
        let mut next_message_lifecycle = self.message_lifecycle.clone();
        let mut events = append_runtime_events_to_state_with_message_lifecycle(
            &self.state,
            self.file_checkpoint_snapshot_store.as_ref(),
            self.output_snapshot_store.as_ref(),
            self.sidecar_store.as_deref(),
            self.event_log_writer.as_deref(),
            self.trace_event_writer.as_deref(),
            self.projection_store.as_deref(),
            Some(&self.session_loops),
            &self.session_id,
            &self.thread_id,
            &self.turn_id,
            vec![event],
            &mut next_message_lifecycle,
        )?;
        self.message_lifecycle = next_message_lifecycle;
        for event in events.drain(..) {
            let is_turn_started = event.event_type == "turn.started";
            self.target.publish(event.clone())?;
            self.events.push(event);
            if is_turn_started {
                self.flush_deferred_hub_events()?;
            }
        }
        Ok(())
    }

    fn emit_preappended(&mut self, event: AgentEvent) -> Result<(), RuntimeCoreError> {
        let is_turn_started = event.event_type == "turn.started";
        self.target.publish(event.clone())?;
        self.events.push(event);
        if is_turn_started {
            self.flush_deferred_hub_events()?;
        }
        Ok(())
    }

    fn emit_transient(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        self.target.publish(AgentEvent {
            event_id: format!("transient-{}", Uuid::new_v4()),
            sequence: 0,
            session_id: self.session_id.clone(),
            thread_id: Some(self.thread_id.clone()),
            turn_id: Some(self.turn_id.clone()),
            event_type: event.event_type,
            timestamp: super::value_fields::timestamp(),
            payload: event.payload,
        })
    }
}

impl AppendingRuntimeEventSink<'_> {
    fn is_duplicate_accepted_event(&self, event: &RuntimeEvent) -> bool {
        event.event_type == "turn.accepted"
            && self
                .events
                .iter()
                .any(|existing| existing.event_type == "turn.accepted")
    }
}

impl RuntimeCore {
    pub async fn start_turn<P>(
        &self,
        params: P,
        host: RuntimeHostContext,
    ) -> Result<RuntimeCoreOutput<AgentSessionTurnStartResponse>, RuntimeCoreError>
    where
        P: Into<TurnStartRequest>,
    {
        self.start_turn_inner(
            params.into(),
            host,
            None,
            true,
            false,
            TurnStartInputKind::User,
        )
        .await
    }

    /// Admit a v2 turn and return its canonical in-progress identity before the
    /// backend completion. The session actor owns the long-running task.
    pub async fn start_turn_admitted<P>(
        &self,
        params: P,
        host: RuntimeHostContext,
    ) -> Result<RuntimeCoreOutput<AgentSessionTurnStartResponse>, RuntimeCoreError>
    where
        P: Into<TurnStartRequest>,
    {
        self.start_turn_inner(
            params.into(),
            host,
            None,
            true,
            true,
            TurnStartInputKind::User,
        )
        .await
    }

    /// Adds user input to the expected active regular turn without creating a
    /// replacement turn. The session loop owns the atomic active-turn check.
    pub async fn steer_turn(
        &self,
        thread_id: &str,
        expected_turn_id: &str,
        input: Vec<agent_protocol::AgentInput>,
        client_user_message_id: Option<String>,
    ) -> Result<RuntimeCoreOutput<String>, RuntimeCoreError> {
        let thread_id = thread_id.trim();
        let expected_turn_id = expected_turn_id.trim();
        if thread_id.is_empty() {
            return Err(RuntimeCoreError::InvalidRequest(
                "turn/steer requires threadId".to_string(),
            ));
        }
        if expected_turn_id.is_empty() {
            return Err(RuntimeCoreError::InvalidRequest(
                "turn/steer requires expectedTurnId".to_string(),
            ));
        }
        validate_user_input(&input).map_err(RuntimeCoreError::InvalidRequest)?;

        let session = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .values()
                .find(|stored| stored.session.thread_id == thread_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(thread_id.to_string()))?;
            let active_turn = stored
                .turns
                .iter()
                .find(|turn| agent_turn_is_active(turn.status))
                .ok_or_else(|| {
                    RuntimeCoreError::InvalidRequest("no active turn to steer".to_string())
                })?;
            if active_turn.turn_id != expected_turn_id {
                return Err(RuntimeCoreError::InvalidRequest(format!(
                    "expected active turn id `{expected_turn_id}` but found `{}`",
                    active_turn.turn_id
                )));
            }
            stored.session.clone()
        };

        let prepared_input = super::input_media::prepare_runtime_input(
            input,
            self.sidecar_store.as_deref(),
            &session.session_id,
        )
        .map_err(RuntimeCoreError::Backend)?;
        let session_loop = self.session_loops.get_or_create(&session.session_id).await;
        let active_turn_id = session_loop
            .steer_for_turn_id_with_metadata(
                Some(expected_turn_id),
                vec![RuntimeSessionInput::User(prepared_input.provider)],
                client_user_message_id.clone(),
                None,
            )
            .await
            .map_err(|error| steer_runtime_error(error, expected_turn_id))?;

        let input_text = user_input_text(&prepared_input.durable);
        let explicit_item_id = client_user_message_id
            .as_deref()
            .map(|client_id| format!("steer-{client_id}"))
            .unwrap_or_else(|| format!("steer-{}", Uuid::new_v4()));
        let mut payload = json!({
            "itemId": explicit_item_id,
            "role": "user",
            "visibility": "user_visible",
            "source": "turn/steer",
            "input": prepared_input.durable,
            "content": {
                "kind": "inline_text",
                "text": input_text,
            },
        });
        if let Some(client_id) = client_user_message_id
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
        {
            payload["clientId"] = Value::String(client_id);
        }
        let events = self.append_runtime_events(
            &session.session_id,
            &session.thread_id,
            Some(&active_turn_id),
            vec![RuntimeEvent::new(
                super::turn_input_events::TURN_INPUT_EVENT_TYPE,
                payload,
            )],
        )?;

        Ok(RuntimeCoreOutput {
            response: active_turn_id,
            events,
        })
    }

    #[cfg(test)]
    pub(crate) async fn start_turn_with_event_callback<P>(
        &self,
        params: P,
        host: RuntimeHostContext,
        event_callback: &mut RuntimeEventCallback<'_>,
    ) -> Result<RuntimeCoreOutput<AgentSessionTurnStartResponse>, RuntimeCoreError>
    where
        P: Into<TurnStartRequest>,
    {
        self.start_turn_inner(
            params.into(),
            host,
            Some(event_callback),
            true,
            false,
            TurnStartInputKind::User,
        )
        .await
    }

    pub(in crate::runtime) async fn start_turn_inner(
        &self,
        mut params: TurnStartRequest,
        host: RuntimeHostContext,
        mut event_callback: Option<&mut RuntimeEventCallback<'_>>,
        enable_goal_continuation: bool,
        return_after_admission: bool,
        input_kind: TurnStartInputKind,
    ) -> Result<RuntimeCoreOutput<AgentSessionTurnStartResponse>, RuntimeCoreError> {
        self.ensure_current_session_hydrated(&params.session_id)
            .await?;
        validate_user_input(&params.input).map_err(RuntimeCoreError::InvalidRequest)?;

        let thread_id = self.session_snapshot(&params.session_id)?.0.thread_id;
        if let Some(settings) = self
            .reconcile_thread_model_selection_for_turn(&thread_id, params.runtime_options.as_ref())
            .await?
        {
            apply_reconciled_thread_model_selection(&mut params.runtime_options, &settings);
            self.event_hub.publish(AgentEvent {
                event_id: new_id("event"),
                sequence: 0,
                session_id: params.session_id.clone(),
                thread_id: Some(thread_id.clone()),
                turn_id: None,
                event_type: "thread.settings.updated".to_string(),
                timestamp: timestamp(),
                payload: json!({ "threadSettings": settings }),
            });
        }

        if let Some(defaults) = self.session_runtime_defaults(&params.session_id)? {
            params.runtime_options =
                Some(super::session_runtime_defaults::merge_with_request_options(
                    defaults,
                    params.runtime_options.take(),
                ));
        }
        if !input_kind.is_agent_only() {
            if let Some(output) = self
                .steer_existing_actor_turn(&params, event_callback.as_deref_mut())
                .await?
            {
                return Ok(output);
            }
        }
        let mut pre_turn_events = self
            .maybe_auto_compact_before_turn(&params.session_id, params.runtime_options.as_ref())
            .await?;
        if let Some(callback) = event_callback.as_deref_mut() {
            for event in pre_turn_events.iter().cloned() {
                callback(event)?;
            }
        }
        self.prepare_memory_prompt_context(&mut params).await;
        self.prepare_session_compaction_prompt_context(&mut params);
        let session_approval_cache_entry = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let session_workspace_id = state
                .sessions
                .get(&params.session_id)
                .and_then(|stored| stored.session.workspace_id.as_deref());
            super::approval_cache::apply_hint_to_turn_start(
                &state.session_approval_cache,
                &mut params,
                session_workspace_id,
            )
        };

        if let Some(capability_id) = params
            .runtime_options
            .as_ref()
            .and_then(|options| options.capability_id.as_deref())
        {
            let capability_context = self.capability_list_context(CapabilityListParams {
                session_id: Some(params.session_id.clone()),
                ..CapabilityListParams::default()
            })?;
            self.capability_source
                .prepare_turn_capabilities(&capability_context, params.runtime_options.as_ref());
            self.ensure_capability_allowed_with_context(&capability_context, capability_id)?;
        }

        let prepared_input = super::input_media::prepare_runtime_input(
            params.input,
            self.sidecar_store.as_deref(),
            &params.session_id,
        )
        .map_err(RuntimeCoreError::Backend)?;
        params.input = prepared_input.durable;

        let queued_turn = {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get_mut(&params.session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(params.session_id.clone()))?;
            let turn_id = optional_id_or_new(params.turn_id.clone(), "turn");
            let active_turn_id = stored
                .turns
                .iter()
                .find(|turn| {
                    if params.skip_pre_submit_resume {
                        agent_turn_blocks_queue_resume(turn.status)
                    } else {
                        agent_turn_is_active(turn.status)
                    }
                })
                .map(|turn| turn.turn_id.clone());
            if params.queue_if_busy && active_turn_id.is_some() {
                if let Some(existing_turn) = stored
                    .turns
                    .iter()
                    .find(|turn| {
                        (agent_turn_is_active(turn.status)
                            || matches!(turn.status, AgentTurnStatus::Queued))
                            && stored
                                .turn_inputs
                                .get(&turn.turn_id)
                                .is_some_and(|input| input == &params.input)
                    })
                    .cloned()
                {
                    Some((stored.session.clone(), existing_turn, false))
                } else {
                    let turn = AgentTurn {
                        turn_id,
                        session_id: stored.session.session_id.clone(),
                        thread_id: stored.session.thread_id.clone(),
                        status: AgentTurnStatus::Queued,
                        started_at: Some(timestamp()),
                        completed_at: None,
                    };
                    stored.session.status = AgentSessionStatus::Running;
                    stored.session.updated_at = timestamp();
                    stored
                        .turn_inputs
                        .insert(turn.turn_id.clone(), params.input.clone());
                    if let Some(runtime_options) = params.runtime_options.clone() {
                        stored
                            .turn_runtime_options
                            .insert(turn.turn_id.clone(), runtime_options);
                    }
                    stored.turns.push(turn.clone());
                    Some((stored.session.clone(), turn, true))
                }
            } else {
                if let Some(active_turn_id) = active_turn_id {
                    return Err(RuntimeCoreError::TurnAlreadyActive(active_turn_id));
                }
                None
            }
        };
        if let Some((session, turn, append_queue_event)) = queued_turn {
            if !append_queue_event {
                return Ok(RuntimeCoreOutput {
                    response: AgentSessionTurnStartResponse { turn },
                    events: Vec::new(),
                });
            }
            let queued_turn_intent =
                super::queued_turn_intent::snapshot_value(params.runtime_options.as_ref())
                    .map_err(RuntimeCoreError::Backend)?;
            let events = self.append_runtime_events(
                &session.session_id,
                &session.thread_id,
                Some(&turn.turn_id),
                vec![RuntimeEvent::new(
                    "queue.added",
                    json!({
                        "source": "turn/start",
                        "queuedTurnId": params
                            .runtime_options
                            .as_ref()
                            .and_then(|options| options.queued_turn_id.clone())
                            .unwrap_or_else(|| turn.turn_id.clone()),
                        "queuedTurnIntent": queued_turn_intent,
                    }),
                )],
            )?;
            return Ok(RuntimeCoreOutput {
                response: AgentSessionTurnStartResponse { turn },
                events,
            });
        }

        let mut provider_input = prepared_input.provider;
        provider_input.agent_only = input_kind.is_agent_only();
        let agent_only_input_event = input_kind
            .is_agent_only()
            .then(|| super::turn_input_events::runtime_event_for_goal_continuation(&params.input))
            .flatten();
        self.prepare_media_prompt_context_from_provider_input(&mut params, &provider_input);

        let (session, previous_session, turn) = {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get_mut(&params.session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(params.session_id.clone()))?;
            let turn_id = optional_id_or_new(params.turn_id.clone(), "turn");
            let previous_session = stored.session.clone();

            let turn = AgentTurn {
                turn_id,
                session_id: stored.session.session_id.clone(),
                thread_id: stored.session.thread_id.clone(),
                status: AgentTurnStatus::Accepted,
                started_at: Some(timestamp()),
                completed_at: None,
            };

            stored.session.status = AgentSessionStatus::Running;
            stored.session.updated_at = timestamp();
            stored
                .turn_inputs
                .insert(turn.turn_id.clone(), params.input.clone());
            if let Some(runtime_options) = params.runtime_options.clone() {
                stored
                    .turn_runtime_options
                    .insert(turn.turn_id.clone(), runtime_options);
            }
            stored.turns.push(turn.clone());

            (stored.session.clone(), previous_session, turn)
        };

        let first_sampling_turn = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(&session.session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session.session_id.clone()))?;
            super::provider_history::provider_turn_history_excluding_current_turn_input(
                stored,
                self.sidecar_store.as_deref(),
                &turn.turn_id,
            )
        }?
        .is_empty();

        let runtime_options = params.runtime_options.clone();
        let request_host = host.clone();
        let agent_control_host = host.clone();
        let forked_from_thread_id = self
            .projection_store
            .as_deref()
            .map(|store| {
                store.read_thread_sync(ReadThreadParams {
                    thread_id: ThreadId::new(session.thread_id.clone()),
                    include_archived: true,
                    turns_view: ThreadTurnsView::NotLoaded,
                })
            })
            .transpose()
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?
            .flatten()
            .and_then(|thread| thread.forked_from_id.map(|id| id.to_string()));
        let mut request = ExecutionRequest {
            host,
            session: session.clone(),
            turn: turn.clone(),
            forked_from_thread_id,
            input: provider_input,
            event_name: params
                .runtime_options
                .as_ref()
                .and_then(|options| options.event_name.clone()),
            expected_output: runtime_options
                .as_ref()
                .and_then(|options| options.expected_output.clone()),
            structured_output: runtime_options
                .as_ref()
                .and_then(|options| options.structured_output.clone()),
            output_schema: runtime_options
                .as_ref()
                .and_then(|options| options.output_schema.clone()),
            queued_turn_id: params
                .runtime_options
                .as_ref()
                .and_then(|options| options.queued_turn_id.clone()),
            runtime_options: params.runtime_options,
            queue_if_busy: params.queue_if_busy,
            skip_pre_submit_resume: params.skip_pre_submit_resume,
            agent_control_gateway: None,
        };
        let effective_runtime_options = self
            .backend
            .effective_turn_runtime_options(&request, first_sampling_turn);
        request.runtime_options = effective_runtime_options.clone();
        let prepared_runtime_options = self
            .backend
            .prepare_turn_runtime_options(&request, first_sampling_turn)
            .await;
        let prepared_runtime_options = match prepared_runtime_options {
            Ok(options) => options,
            Err(error) => {
                self.rollback_started_turn(
                    &session.session_id,
                    &turn.turn_id,
                    previous_session.clone(),
                );
                return Err(error);
            }
        };
        let agent_control_tool_options = if self.projection_store.is_some() {
            Some(
                super::agent_control_gateway::spawn_tool_options::load(
                    self,
                    prepared_runtime_options.as_ref(),
                )
                .await,
            )
        } else {
            None
        };
        request.runtime_options = prepared_runtime_options.clone();
        if prepared_runtime_options.is_none() && self.backend.requires_provider_selection() {
            let error = RuntimeCoreError::pending_route_for_session(
                session.session_id.clone(),
                request.runtime_options.as_ref(),
            );
            self.rollback_started_turn(
                &session.session_id,
                &turn.turn_id,
                previous_session.clone(),
            );
            return Err(error);
        }
        {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get_mut(&session.session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session.session_id.clone()))?;
            match prepared_runtime_options {
                Some(options) => {
                    stored
                        .turn_runtime_options
                        .insert(turn.turn_id.clone(), options);
                }
                None => {
                    stored.turn_runtime_options.remove(&turn.turn_id);
                }
            }
        }
        request.agent_control_gateway = self.projection_store.as_ref().map(|_| {
            self.agent_control_gateway_for_turn(&session, &turn, agent_control_host)
                .with_tool_options(agent_control_tool_options.unwrap_or_default())
        });

        let mailbox_delivery = match if self.backend.requires_provider_selection() {
            // The provider-backed backend drains QueueOnly at the first post-step mailbox
            // boundary through the session loader. Non-provider backends do not expose that
            // boundary, so their durable mailbox must be materialized before execution.
            self.deliver_pending_agent_trigger_mailbox_for_turn(&session, &turn)
                .await
        } else {
            self.deliver_pending_agent_mailbox_for_turn(&session, &turn)
                .await
        } {
            Ok(delivery) => delivery,
            Err(error) => {
                self.rollback_started_turn(
                    &session.session_id,
                    &turn.turn_id,
                    previous_session.clone(),
                );
                return Err(error);
            }
        };
        if let Some(callback) = event_callback.as_deref_mut() {
            for event in mailbox_delivery.events.iter().cloned() {
                callback(event)?;
            }
        }
        pre_turn_events.extend(mailbox_delivery.events);
        let provider_history = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(&session.session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session.session_id.clone()))?;
            super::provider_history::provider_turn_history_excluding_current_turn_input(
                stored,
                self.sidecar_store.as_deref(),
                &turn.turn_id,
            )
        }?;

        if return_after_admission {
            let submitted = match self
                .submit_backend_via_session_loop(request.clone(), provider_history.clone())
                .await
            {
                Ok(submitted) => submitted,
                Err(error) => {
                    self.rollback_started_turn(
                        &session.session_id,
                        &turn.turn_id,
                        previous_session,
                    );
                    return Err(error);
                }
            };
            let mut sink = AppendingRuntimeEventSink::with_hub(
                self.state.clone(),
                self.file_checkpoint_snapshot_store.clone(),
                self.output_snapshot_store.clone(),
                self.sidecar_store.clone(),
                self.event_log_writer.clone(),
                self.trace_event_writer.clone(),
                self.projection_store.clone(),
                self.session_loops.clone(),
                session.session_id.clone(),
                session.thread_id.clone(),
                turn.turn_id.clone(),
                self.event_hub.clone(),
            );
            if let Some(event) = agent_only_input_event.clone() {
                if let Err(error) = sink.emit(event) {
                    let _ = submitted
                        .session
                        .interrupt_for_turn(Some(&turn.turn_id))
                        .await;
                    self.rollback_started_turn(
                        &session.session_id,
                        &turn.turn_id,
                        previous_session,
                    );
                    return Err(error);
                }
            }
            if let Err(error) = sink.emit(RuntimeEvent::new(
                "turn.accepted",
                json!({
                    "backend": "app_server",
                    "source": "turn/start",
                }),
            )) {
                let _ = submitted
                    .session
                    .interrupt_for_turn(Some(&turn.turn_id))
                    .await;
                self.rollback_started_turn(&session.session_id, &turn.turn_id, previous_session);
                return Err(error);
            }
            if let Some(event) = super::expert_role_switch::runtime_event_from_request_metadata(
                request.runtime_metadata(),
            ) {
                if let Err(error) = sink.emit(event) {
                    let _ = submitted
                        .session
                        .interrupt_for_turn(Some(&turn.turn_id))
                        .await;
                    self.rollback_started_turn(
                        &session.session_id,
                        &turn.turn_id,
                        previous_session,
                    );
                    return Err(error);
                }
            }
            if let Some(entry) = session_approval_cache_entry.as_ref() {
                if let Err(error) = sink.emit(RuntimeEvent::new(
                    "approval.session_cache.hit",
                    json!({
                        "backend": "runtime_core",
                        "decision": entry.decision.as_str(),
                        "decisionScope": entry.decision.scope(),
                        "sourceRequestId": &entry.request_id,
                        "key": super::approval_cache::entry_key_metadata(entry),
                    }),
                )) {
                    let _ = submitted
                        .session
                        .interrupt_for_turn(Some(&turn.turn_id))
                        .await;
                    self.rollback_started_turn(
                        &session.session_id,
                        &turn.turn_id,
                        previous_session,
                    );
                    return Err(error);
                }
            }
            let admission_events = sink.events.clone();
            sink.defer_hub_events(pre_turn_events.clone());

            let runtime = self.clone();
            let session_id = session.session_id.clone();
            let turn_id = turn.turn_id.clone();
            let previous_session_for_task = previous_session.clone();
            let goal_continuation_host = request_host.clone();
            let pending_work_host = request_host.clone();
            let pending_work_runtime_options = request.runtime_options.clone();
            tokio::spawn(async move {
                tokio::task::yield_now().await;
                let mut sink = sink;
                let drive_result = runtime
                    .drive_backend_to_completion(submitted, &mut sink)
                    .await;
                let drive_result = match drive_result {
                    Ok(()) => runtime.ensure_admitted_turn_terminal_after_task(
                        &session_id,
                        &turn_id,
                        &mut sink,
                    ),
                    Err(error) => Err(error),
                };
                if let Err(error) = drive_result {
                    if sink.emitted_count() > 0 {
                        let _ = sink.emit_failure(&error);
                        runtime.wake_pending_session_work_if_turn_terminal(
                            &session_id,
                            &turn_id,
                            pending_work_host.clone(),
                            pending_work_runtime_options.clone(),
                        );
                    } else {
                        runtime.rollback_started_turn(
                            &session_id,
                            &turn_id,
                            previous_session_for_task.clone(),
                        );
                    }
                    return;
                }
                runtime
                    .schedule_pending_agent_mailbox_triggers(
                        session_id.clone(),
                        pending_work_host,
                        pending_work_runtime_options,
                    )
                    .await;
                let continuation_runtime = runtime.clone();
                let continuation_hub = runtime.event_hub.clone();
                let continuation_session_id = session_id.clone();
                let runtime_handle = tokio::runtime::Handle::current();
                let _ = tokio::task::spawn_blocking(move || {
                    runtime_handle.block_on(
                        continuation_runtime.maybe_continue_thread_goal_if_idle_with_hub(
                            &continuation_session_id,
                            goal_continuation_host,
                            continuation_hub,
                        ),
                    );
                })
                .await;
            });
            let mut events = pre_turn_events;
            events.extend(admission_events);
            return Ok(RuntimeCoreOutput {
                response: AgentSessionTurnStartResponse { turn },
                events,
            });
        }

        let backend_events = if let Some(event_callback) = event_callback {
            let mut sink = AppendingRuntimeEventSink::new(
                self.state.clone(),
                self.file_checkpoint_snapshot_store.clone(),
                self.output_snapshot_store.clone(),
                self.sidecar_store.clone(),
                self.event_log_writer.clone(),
                self.trace_event_writer.clone(),
                self.projection_store.clone(),
                self.session_loops.clone(),
                session.session_id.clone(),
                session.thread_id.clone(),
                turn.turn_id.clone(),
                event_callback,
            );
            if let Some(event) = agent_only_input_event.clone() {
                sink.emit(event)?;
            }
            sink.emit(RuntimeEvent::new(
                "turn.accepted",
                json!({
                    "backend": "app_server",
                    "source": "turn/start",
                }),
            ))?;
            if let Some(event) = super::expert_role_switch::runtime_event_from_request_metadata(
                request.runtime_metadata(),
            ) {
                sink.emit(event)?;
            }
            if let Some(entry) = session_approval_cache_entry.as_ref() {
                sink.emit(RuntimeEvent::new(
                    "approval.session_cache.hit",
                    json!({
                        "backend": "runtime_core",
                        "decision": entry.decision.as_str(),
                        "decisionScope": entry.decision.scope(),
                        "sourceRequestId": &entry.request_id,
                        "key": super::approval_cache::entry_key_metadata(entry),
                    }),
                ))?;
            }
            if let Err(error) = self
                .execute_backend_via_session_loop(
                    request.clone(),
                    provider_history.clone(),
                    &mut sink,
                )
                .await
            {
                if sink.emitted_count() > 0 {
                    sink.emit_failure(&error)?;
                    self.wake_pending_session_work_if_turn_terminal(
                        &session.session_id,
                        &turn.turn_id,
                        request_host.clone(),
                        request.runtime_options.clone(),
                    );
                } else {
                    self.rollback_started_turn(
                        &session.session_id,
                        &turn.turn_id,
                        previous_session,
                    );
                }
                return Err(error);
            }
            let events = sink.into_events();
            events
        } else {
            let mut sink = CollectingRuntimeEventSink::default();
            if let Some(event) = agent_only_input_event {
                sink.emit(event)?;
            }
            if let Some(event) = super::expert_role_switch::runtime_event_from_request_metadata(
                request.runtime_metadata(),
            ) {
                sink.emit(event)?;
            }
            if let Some(entry) = session_approval_cache_entry.as_ref() {
                sink.emit(RuntimeEvent::new(
                    "approval.session_cache.hit",
                    json!({
                        "backend": "runtime_core",
                        "decision": entry.decision.as_str(),
                        "decisionScope": entry.decision.scope(),
                        "sourceRequestId": &entry.request_id,
                        "key": super::approval_cache::entry_key_metadata(entry),
                    }),
                ))?;
            }
            if let Err(error) = self
                .execute_backend_via_session_loop(
                    request.clone(),
                    provider_history.clone(),
                    &mut sink,
                )
                .await
            {
                if sink.emitted_count() > 0 {
                    sink.ensure_turn_accepted_event();
                    sink.emit_failure(&error)?;
                    let (runtime_events, _) = sink.into_parts();
                    if let Err(append_error) = self.append_runtime_events(
                        &session.session_id,
                        &session.thread_id,
                        Some(&turn.turn_id),
                        runtime_events,
                    ) {
                        self.rollback_started_turn(
                            &session.session_id,
                            &turn.turn_id,
                            previous_session,
                        );
                        return Err(append_error);
                    }
                    self.wake_pending_session_work_if_turn_terminal(
                        &session.session_id,
                        &turn.turn_id,
                        request_host.clone(),
                        request.runtime_options.clone(),
                    );
                } else {
                    self.rollback_started_turn(
                        &session.session_id,
                        &turn.turn_id,
                        previous_session,
                    );
                }
                return Err(error);
            }
            sink.ensure_turn_accepted_event();
            let (runtime_events, preappended_events) = sink.into_parts();
            match self.append_runtime_events(
                &session.session_id,
                &session.thread_id,
                Some(&turn.turn_id),
                runtime_events,
            ) {
                Ok(mut backend_events) => {
                    backend_events.extend(preappended_events);
                    backend_events
                }
                Err(error) => {
                    self.rollback_started_turn(
                        &session.session_id,
                        &turn.turn_id,
                        previous_session,
                    );
                    return Err(error);
                }
            }
        };
        let mut events = pre_turn_events;
        events.extend(backend_events);
        let response_turn = self
            .stored_turn(&session.session_id, &turn.turn_id)?
            .unwrap_or(turn);
        if input_kind.runs_idle_scheduler() && agent_turn_is_terminal(response_turn.status) {
            self.schedule_pending_agent_mailbox_triggers(
                session.session_id.clone(),
                request_host.clone(),
                request.runtime_options.clone(),
            )
            .await;
        }
        if enable_goal_continuation && response_turn.status == AgentTurnStatus::Completed {
            self.maybe_continue_thread_goal_if_idle(&session.session_id, request_host)
                .await;
        }

        Ok(RuntimeCoreOutput {
            response: AgentSessionTurnStartResponse {
                turn: response_turn,
            },
            events,
        })
    }

    async fn execute_backend_request(
        &self,
        request: ExecutionRequest,
        provider_history: super::provider_history::ProviderTurnHistory,
        pending_input: Option<RuntimeSessionInputHandle>,
        cancellation_token: Option<CancellationToken>,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.backend
            .start_turn_with_provider_history_and_session_input(
                request,
                provider_history,
                pending_input,
                cancellation_token,
                sink,
            )
            .await
    }

    fn wake_pending_session_work_if_turn_terminal(
        &self,
        session_id: &str,
        turn_id: &str,
        host: RuntimeHostContext,
        runtime_options: Option<RuntimeOptions>,
    ) {
        let terminal = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            state
                .sessions
                .get(session_id)
                .and_then(|stored| stored.turns.iter().find(|turn| turn.turn_id == turn_id))
                .is_some_and(|turn| agent_turn_is_terminal(turn.status))
        };
        if terminal {
            self.wake_pending_session_work(session_id.to_string(), host, runtime_options);
        }
    }

    async fn steer_existing_actor_turn(
        &self,
        params: &TurnStartRequest,
        mut event_callback: Option<&mut RuntimeEventCallback<'_>>,
    ) -> Result<Option<RuntimeCoreOutput<AgentSessionTurnStartResponse>>, RuntimeCoreError> {
        if params.queue_if_busy {
            return Ok(None);
        }
        let session = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let Some(stored) = state.sessions.get(&params.session_id) else {
                return Err(RuntimeCoreError::SessionNotFound(params.session_id.clone()));
            };
            stored.session.clone()
        };

        let prepared_input = super::input_media::prepare_runtime_input(
            params.input.clone(),
            self.sidecar_store.as_deref(),
            &params.session_id,
        )
        .map_err(RuntimeCoreError::Backend)?;
        let (client_user_message_id, trace) = super::session_submission::metadata(
            params
                .runtime_options
                .as_ref()
                .and_then(RuntimeOptions::runtime_metadata),
        );
        let Some(session_loop) = self.session_loops.get_existing(&session.session_id).await else {
            return Ok(None);
        };
        let requested_turn_id = params.turn_id.as_deref().filter(|id| !id.trim().is_empty());
        let active_turn_id = match session_loop
            .steer_for_turn_id_with_metadata(
                requested_turn_id,
                vec![RuntimeSessionInput::User(prepared_input.provider)],
                client_user_message_id,
                trace,
            )
            .await
        {
            Ok(active_turn_id) => active_turn_id,
            Err(RuntimeSessionLoopError::InvalidTask(_)) => return Ok(None),
            Err(RuntimeSessionLoopError::Closed) => {
                return Err(RuntimeCoreError::Backend(
                    "runtime session submission loop is closed".to_string(),
                ));
            }
            Err(RuntimeSessionLoopError::OperationFailed(message)) => {
                return Err(RuntimeCoreError::Backend(message));
            }
        };
        let turn = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let Some(turn) = state
                .sessions
                .get(&session.session_id)
                .and_then(|stored| {
                    stored.turns.iter().find(|turn| {
                        turn.turn_id == active_turn_id && agent_turn_is_active(turn.status)
                    })
                })
                .cloned()
            else {
                return Err(RuntimeCoreError::Backend(
                    "session actor accepted steer without an active canonical turn".to_string(),
                ));
            };
            turn
        };
        let events = self.append_runtime_events(
            &session.session_id,
            &session.thread_id,
            Some(&active_turn_id),
            vec![RuntimeEvent::new(
                super::turn_input_events::TURN_INPUT_EVENT_TYPE,
                json!({
                    "role": "user",
                    "visibility": "user_visible",
                    "source": "session_steer",
                    "input": prepared_input.durable,
                    "content": {
                        "kind": "inline_text",
                        "text": user_input_text(&params.input),
                    },
                }),
            )],
        )?;
        if let Some(callback) = event_callback.as_deref_mut() {
            for event in events.iter().cloned() {
                callback(event)?;
            }
        }
        Ok(Some(RuntimeCoreOutput {
            response: AgentSessionTurnStartResponse { turn },
            events,
        }))
    }

    async fn submit_backend_via_session_loop(
        &self,
        request: ExecutionRequest,
        provider_history: super::provider_history::ProviderTurnHistory,
    ) -> Result<SubmittedRuntimeSessionTurn, RuntimeCoreError> {
        let session_id = request.session.session_id.clone();
        let turn_id = request.turn.turn_id.clone();
        let queue_if_busy = request.queue_if_busy;
        let user_input = request.input.clone();
        let (client_user_message_id, trace) =
            super::session_submission::metadata(request.runtime_metadata());
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let runtime = self.clone();
        let mailbox_loader_runtime = runtime.clone();
        let mailbox_loader_session = request.session.clone();
        let mailbox_loader_turn = request.turn.clone();
        {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            state
                .sessions
                .get(&session_id)
                .and_then(|stored| stored.turn_inputs.get(&turn_id))
                .cloned()
                .ok_or_else(|| {
                    RuntimeCoreError::Backend(
                        "runtime session submission requires durable turn input".to_string(),
                    )
                })?;
        }
        let mailbox_loader_events = event_sender.clone();
        let mailbox_loader =
            move || -> BoxFuture<'static, Result<Vec<RuntimeSessionInput>, String>> {
                let runtime = mailbox_loader_runtime.clone();
                let session = mailbox_loader_session.clone();
                let turn = mailbox_loader_turn.clone();
                let event_sender = mailbox_loader_events.clone();
                Box::pin(async move {
                    let delivery = runtime
                        .deliver_pending_agent_mailbox_for_turn(&session, &turn)
                        .await
                        .map_err(|error| error.to_string())?;
                    for event in delivery.events.iter().cloned() {
                        event_sender
                            .send(SessionLoopEvent::Preappended(event))
                            .map_err(|_| "runtime session event receiver closed".to_string())?;
                    }
                    Ok(delivery
                        .consumed_messages()
                        .iter()
                        .map(super::inter_agent_input::from_mailbox_message)
                        .map(RuntimeSessionInput::InterAgent)
                        .collect::<Vec<_>>())
                })
            };
        let task = Arc::new(
            RuntimeSessionClosureTask::new(
                turn_id.clone(),
                Vec::new(),
                move |context, _input, _cancellation_token| {
                    let runtime = runtime.clone();
                    let request = request.clone();
                    let provider_history = provider_history.clone();
                    let event_sender = event_sender.clone();
                    Box::pin(async move {
                        let mut channel_sink = ChannelRuntimeEventSink {
                            sender: event_sender,
                        };
                        let cancellation_token = _cancellation_token.clone();
                        runtime
                            .execute_backend_request(
                                request,
                                provider_history,
                                Some(context.input_handle()),
                                Some(cancellation_token),
                                &mut channel_sink,
                            )
                            .await
                            .map_err(|error| RuntimeSessionTaskFailure {
                                message: error.to_string(),
                                reason_code: Some(error.turn_failure_reason().to_string()),
                            })
                    })
                },
            )
            .with_mailbox_loader(mailbox_loader)
            .with_abort(|_context| Box::pin(async {})),
        );
        let session = self.session_loops.get_or_create(&session_id).await;
        let submission = match session
            .submit_user_input_with_metadata(
                task,
                vec![RuntimeSessionInput::User(user_input)],
                queue_if_busy,
                client_user_message_id,
                trace,
            )
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?
        {
            RuntimeSessionUserInputResult::Submitted(submission) => submission,
            RuntimeSessionUserInputResult::Steered {
                turn_id: active_turn_id,
                ..
            } => {
                return Err(RuntimeCoreError::TurnAlreadyActive(active_turn_id));
            }
        };
        if matches!(submission.result, RuntimeSessionSubmitResult::Busy) {
            return Err(RuntimeCoreError::TurnAlreadyActive(turn_id));
        }
        let driver_completion = self.turn_driver_completions.register(&session_id, &turn_id);
        Ok(SubmittedRuntimeSessionTurn {
            session,
            session_id,
            turn_id,
            event_receiver,
            completion: submission.completion,
            driver_completion,
            driver_completions: self.turn_driver_completions.clone(),
        })
    }

    async fn drive_backend_to_completion(
        &self,
        mut submission: SubmittedRuntimeSessionTurn,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let session = submission.session.clone();
        let turn_id = submission.turn_id.clone();
        loop {
            tokio::select! {
                event = submission.event_receiver.recv() => {
                    if let Some(event) = event {
                        let result = match event {
                            SessionLoopEvent::Runtime(event) => sink.emit(event),
                            SessionLoopEvent::Transient(event) => sink.emit_transient(event),
                            SessionLoopEvent::Preappended(event) => sink.emit_preappended(event),
                        };
                        if let Err(error) = result {
                            let _ = session.interrupt_for_turn(Some(&turn_id)).await;
                            return Err(error);
                        }
                    }
                }
                result = &mut submission.completion => {
                    while let Ok(event) = submission.event_receiver.try_recv() {
                        match event {
                            SessionLoopEvent::Runtime(event) => sink.emit(event)?,
                            SessionLoopEvent::Transient(event) => sink.emit_transient(event)?,
                            SessionLoopEvent::Preappended(event) => sink.emit_preappended(event)?,
                        }
                    }
                    let outcome = result.map_err(|_| {
                        RuntimeCoreError::Backend("runtime session task completion channel closed".to_string())
                    })?;
                    match outcome {
                        Ok(RuntimeSessionTaskOutcome::Completed) => return Ok(()),
                        Ok(RuntimeSessionTaskOutcome::Shutdown) => {
                            sink.emit(RuntimeEvent::new(
                                "turn.canceled",
                                json!({
                                    "reason": "session_task_shutdown",
                                    "source": "session_loop",
                                }),
                            ))?;
                            return Ok(());
                        }
                        Ok(
                            outcome @ (RuntimeSessionTaskOutcome::Interrupted
                            | RuntimeSessionTaskOutcome::Replaced),
                        ) => {
                            let reason = match outcome {
                                RuntimeSessionTaskOutcome::Interrupted => {
                                    "session_task_interrupted"
                                }
                                RuntimeSessionTaskOutcome::Replaced => "session_task_replaced",
                                RuntimeSessionTaskOutcome::Completed
                                | RuntimeSessionTaskOutcome::Shutdown => unreachable!(),
                            };
                            sink.emit(RuntimeEvent::new(
                                "turn.canceled",
                                json!({
                                    "reason": reason,
                                    "source": "session_loop",
                                }),
                            ))?;
                            return Ok(());
                        }
                        Err(error) => {
                            return Err(runtime_error_from_session_task_failure(error));
                        }
                    }
                }
            }
        }
    }

    fn ensure_admitted_turn_terminal_after_task(
        &self,
        session_id: &str,
        turn_id: &str,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let turn = self
            .stored_turn(session_id, turn_id)?
            .ok_or_else(|| RuntimeCoreError::TurnNotActive(turn_id.to_string()))?;
        if agent_turn_is_terminal(turn.status) {
            return Ok(());
        }
        sink.emit(RuntimeEvent::new(
            "turn.completed",
            json!({
                "backend": "runtime_core",
                "reason": "session_task_completed",
                "source": "session_loop",
            }),
        ))
    }

    async fn execute_backend_via_session_loop(
        &self,
        request: ExecutionRequest,
        provider_history: super::provider_history::ProviderTurnHistory,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let submission = self
            .submit_backend_via_session_loop(request, provider_history)
            .await?;
        self.drive_backend_to_completion(submission, sink).await
    }

    /// Validate the same active-turn boundary used by `turn/interrupt` before
    /// the App Server detaches any typed reverse request route.
    pub fn ensure_turn_interruptible(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Result<(), RuntimeCoreError> {
        let state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let stored = state
            .sessions
            .get(session_id)
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
        let turn = stored
            .turns
            .iter()
            .find(|turn| turn.turn_id == turn_id)
            .ok_or_else(|| RuntimeCoreError::TurnNotActive(turn_id.to_string()))?;
        if !agent_turn_is_active(turn.status) {
            return Err(RuntimeCoreError::TurnNotActive(turn_id.to_string()));
        }
        Ok(())
    }

    pub async fn cancel_turn(
        &self,
        params: AgentSessionTurnCancelParams,
        host: RuntimeHostContext,
    ) -> Result<RuntimeCoreOutput<AgentSessionTurnCancelResponse>, RuntimeCoreError> {
        let (session, turn_snapshot, pending_action, pending_tool_item) = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(&params.session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(params.session_id.clone()))?;
            let turn = stored
                .turns
                .iter()
                .find(|turn| turn.turn_id == params.turn_id)
                .ok_or_else(|| RuntimeCoreError::TurnNotActive(params.turn_id.clone()))?;
            if !agent_turn_is_active(turn.status) {
                return Err(RuntimeCoreError::TurnNotActive(params.turn_id.clone()));
            }

            let pending_action =
                pending_action_descriptor::pending_identity_for_turn(stored, &params.turn_id);
            let pending_tool_item =
                pending_action_descriptor::pending_tool_id_for_turn(stored, &params.turn_id)
                    .and_then(|tool_call_id| {
                        active_tool_item_for_turn(stored, &params.turn_id, &tool_call_id)
                    });
            (
                stored.session.clone(),
                turn.clone(),
                pending_action,
                pending_tool_item,
            )
        };

        let mut pre_cancel_events = if let Some((request_id, identity)) = pending_action.as_ref() {
            self.append_runtime_events(
                &session.session_id,
                &session.thread_id,
                Some(&turn_snapshot.turn_id),
                vec![turn_interrupt_action_canceled_event(request_id, identity)],
            )?
        } else {
            Vec::new()
        };
        if let Some(item) = pending_tool_item {
            pre_cancel_events.extend(self.append_runtime_events(
                &session.session_id,
                &session.thread_id,
                Some(&turn_snapshot.turn_id),
                vec![turn_interrupt_tool_completed_event(item)],
            )?);
        }

        let driver_completion = self
            .turn_driver_completions
            .get(&session.session_id, &params.turn_id);
        let mut interrupted_by_session_loop = false;
        if agent_turn_is_active(turn_snapshot.status) {
            if let Some(session_loop) = self.session_loops.get_existing(&session.session_id).await {
                interrupted_by_session_loop = session_loop
                    .interrupt_for_turn(Some(&params.turn_id))
                    .await
                    .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
            }
        }
        let events = if interrupted_by_session_loop {
            driver_completion
                .ok_or_else(|| {
                    RuntimeCoreError::Backend(
                        "active runtime turn is missing its driver completion".to_string(),
                    )
                })?
                .wait()
                .await;
            // The driver has already appended and published turn.canceled after draining
            // provider/tool events; returning it here would publish the same event twice.
            pre_cancel_events
        } else {
            let mut events = pre_cancel_events;
            events.extend(self.append_runtime_events(
                &session.session_id,
                &session.thread_id,
                Some(&turn_snapshot.turn_id),
                vec![RuntimeEvent::new(
                    "turn.canceled",
                    json!({
                        "source": "turn/interrupt",
                        "backend": "runtime_core",
                    }),
                )],
            )?);
            events
        };
        {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            super::approval_cache::remove_session(
                &mut state.session_approval_cache,
                &session.session_id,
            );
        }
        if let Some(event_log_writer) = self.event_log_writer.as_deref() {
            let workflow_audit_records = event_log_writer
                .read_session_workflow_audit_events(&session.session_id)
                .map_err(RuntimeCoreError::Backend)?;
            append_workflow_audit_runtime_events(
                Some(event_log_writer),
                &session.session_id,
                &session.thread_id,
                Some(&turn_snapshot.turn_id),
                workflow_cancel_events_from_audit_records(
                    &workflow_audit_records,
                    &turn_snapshot.turn_id,
                ),
            )?;
        }

        let pending_work_runtime_options = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            state
                .sessions
                .get(&params.session_id)
                .and_then(|stored| stored.turn_runtime_options.get(&params.turn_id))
                .cloned()
        };
        let pending_work_host = host.clone();
        if agent_turn_is_active(turn_snapshot.status) {
            let backend = self.backend.clone();
            let backend_turn_snapshot = turn_snapshot.clone();
            tokio::spawn(async move {
                let mut sink = CollectingRuntimeEventSink::default();
                let _ = backend
                    .cancel_turn(
                        CancelExecutionRequest {
                            host,
                            session,
                            turn: backend_turn_snapshot,
                        },
                        &mut sink,
                    )
                    .await;
            });
        }
        if !interrupted_by_session_loop {
            self.wake_pending_session_work(
                params.session_id,
                pending_work_host,
                pending_work_runtime_options,
            );
        }

        Ok(RuntimeCoreOutput {
            response: AgentSessionTurnCancelResponse {},
            events,
        })
    }

    pub async fn respond_action(
        &self,
        params: AgentSessionActionRespondParams,
        host: RuntimeHostContext,
    ) -> Result<RuntimeCoreOutput<AgentSessionActionRespondResponse>, RuntimeCoreError> {
        self.ensure_current_session_hydrated(&params.session_id)
            .await?;
        let request_id = params.request_id.clone();
        let (
            session,
            turn_snapshot,
            pending_action_identity,
            pending_action_descriptor,
            pending_tool_item,
        ) = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(&params.session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(params.session_id.clone()))?;
            let pending_action_identity =
                pending_action_descriptor::identity_from_stored_session(stored, &request_id)
                    .ok_or_else(|| RuntimeCoreError::ActionResponse {
                        code: "action_not_found".to_string(),
                        request_id: request_id.clone(),
                    })?;
            let turn_id = pending_action_identity
                .scope
                .turn_id
                .as_deref()
                .expect("pending action identity always has complete scope");
            let turn = stored
                .turns
                .iter()
                .find(|turn| turn.turn_id == turn_id)
                .expect("pending action identity always references a waiting turn")
                .clone();
            let pending_action_descriptor =
                pending_action_descriptor::from_stored_session(stored, &request_id);
            let pending_tool_item = pending_action_descriptor::pending_tool_id_for_turn(
                stored, turn_id,
            )
            .and_then(|tool_call_id| active_tool_item_for_turn(stored, turn_id, &tool_call_id));
            (
                stored.session.clone(),
                Some(turn),
                pending_action_identity,
                pending_action_descriptor,
                pending_tool_item,
            )
        };
        if params.action_type != pending_action_identity.action_type {
            return Err(RuntimeCoreError::ActionResponse {
                code: "action_type_mismatch".to_string(),
                request_id,
            });
        }
        match params.action_scope.as_ref() {
            None => {
                return Err(RuntimeCoreError::ActionResponse {
                    code: "action_scope_missing".to_string(),
                    request_id,
                });
            }
            Some(scope) if scope != &pending_action_identity.scope => {
                return Err(RuntimeCoreError::ActionResponse {
                    code: "action_scope_mismatch".to_string(),
                    request_id,
                });
            }
            Some(_) => {}
        }
        let decision = match pending_action_identity.action_type {
            AgentSessionActionType::ToolConfirmation => Some(params.decision.ok_or_else(|| {
                RuntimeCoreError::Backend(
                    "tool_confirmation action/respond requires decision".to_string(),
                )
            })?),
            AgentSessionActionType::AskUser | AgentSessionActionType::Elicitation => {
                if params.decision.is_some() {
                    return Err(RuntimeCoreError::Backend(
                        "approval decision is only valid for tool_confirmation action/respond"
                            .to_string(),
                    ));
                }
                None
            }
        };
        let confirmed = decision
            .map(AgentSessionApprovalDecision::confirmed)
            .unwrap_or_else(|| params.confirmed.unwrap_or(false));
        let workflow_resume_audit_events =
            workflow_resume_audit_events_from_action_response(&params, decision, confirmed);
        let (cancel_denied_permission_action, app_server_owned_permission_action) = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(&params.session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(params.session_id.clone()))?;
            if let Some(decision) = decision {
                super::approval_decision_contract::validate_tool_confirmation_decision(
                    stored,
                    &request_id,
                    decision,
                )?;
            }
            let cancel_denied_permission_action =
                super::permission_state_projection::should_cancel_denied_permission_action(
                    stored,
                    &request_id,
                    decision.is_some_and(AgentSessionApprovalDecision::is_cancel),
                );
            let app_server_owned_permission_action =
                super::permission_state_projection::is_runtime_preflight_action(
                    stored,
                    &request_id,
                );
            (
                cancel_denied_permission_action,
                app_server_owned_permission_action,
            )
        };

        let session_approval_cache_entry = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(&params.session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(params.session_id.clone()))?;
            super::approval_cache::entry_from_action_response(
                stored,
                &request_id,
                decision,
                params.action_scope.clone(),
                super::timestamp(),
            )
        };
        if decision == Some(AgentSessionApprovalDecision::AllowForSession)
            && session_approval_cache_entry.is_none()
        {
            return Err(RuntimeCoreError::Backend(format!(
                "allow_for_session requires session approval cache owner for tool_confirmation request '{}'",
                request_id
            )));
        }

        let mut sink = CollectingRuntimeEventSink::default();
        let mut precommitted_events = Vec::new();
        let action_response = ActionRespondRequest {
            host: host.clone(),
            session: session.clone(),
            turn: turn_snapshot.clone(),
            request_id: request_id.clone(),
            action_type: params.action_type,
            decision,
            confirmed,
            response: params.response,
            user_data: params.user_data,
            metadata: params.metadata,
            event_name: params.event_name,
            action_scope: params.action_scope,
            pending_action_descriptor,
        };
        if app_server_owned_permission_action {
            if decision.is_some_and(AgentSessionApprovalDecision::is_cancel) {
                sink.emit(app_server_action_canceled_event(&action_response))?;
            } else {
                sink.emit(app_server_action_resolved_event(&action_response))?;
            }
        } else {
            self.backend
                .respond_action(action_response.clone(), &mut sink)
                .await?;
            if self.backend.has_live_session_responses() {
                let backend_events = sink.take_events();
                precommitted_events.extend(self.append_runtime_events(
                    &session.session_id,
                    &session.thread_id,
                    turn_snapshot.as_ref().map(|turn| turn.turn_id.as_str()),
                    backend_events,
                )?);
                if decision.is_some_and(AgentSessionApprovalDecision::is_cancel) {
                    if let Some(item) = pending_tool_item {
                        precommitted_events.extend(self.append_runtime_events(
                            &session.session_id,
                            &session.thread_id,
                            turn_snapshot.as_ref().map(|turn| turn.turn_id.as_str()),
                            vec![turn_interrupt_tool_completed_event(item)],
                        )?);
                    }
                }
                if !decision.is_some_and(AgentSessionApprovalDecision::is_cancel) {
                    self.dispatch_live_action_response(&action_response).await?;
                }
            }
        }
        let backend_cancel_requested = decision
            .is_some_and(AgentSessionApprovalDecision::is_cancel)
            && turn_snapshot
                .as_ref()
                .is_some_and(|turn| agent_turn_is_active(turn.status));
        let driver_completion = turn_snapshot.as_ref().and_then(|turn| {
            self.turn_driver_completions
                .get(&session.session_id, &turn.turn_id)
        });
        let mut interrupted_by_session_loop = false;
        if backend_cancel_requested
            && self.backend.has_live_session_responses()
            && !sink.has_turn_terminal_event()
        {
            if let (Some(turn), Some(session_loop)) = (
                turn_snapshot.as_ref(),
                self.session_loops.get_existing(&session.session_id).await,
            ) {
                interrupted_by_session_loop = session_loop
                    .interrupt_for_turn(Some(&turn.turn_id))
                    .await
                    .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
            }
        }
        if interrupted_by_session_loop {
            driver_completion
                .ok_or_else(|| {
                    RuntimeCoreError::Backend(
                        "active runtime turn is missing its driver completion".to_string(),
                    )
                })?
                .wait()
                .await;
        }
        if backend_cancel_requested
            && !interrupted_by_session_loop
            && !sink.has_turn_terminal_event()
        {
            self.backend
                .cancel_turn(
                    CancelExecutionRequest {
                        host,
                        session: session.clone(),
                        turn: turn_snapshot
                            .clone()
                            .expect("active turn snapshot must exist for approval cancel"),
                    },
                    &mut sink,
                )
                .await?;
        }
        if cancel_denied_permission_action && !backend_cancel_requested {
            sink.emit(RuntimeEvent::new(
                "turn.canceled",
                json!({
                    "backend": "runtime",
                    "reason": "permission_denied",
                    "requestId": request_id,
                }),
            ))?;
        }
        let mut events = self.append_runtime_events(
            &session.session_id,
            &session.thread_id,
            turn_snapshot.as_ref().map(|turn| turn.turn_id.as_str()),
            sink.into_events(),
        )?;
        precommitted_events.append(&mut events);
        let events = precommitted_events;
        append_workflow_audit_runtime_events(
            self.event_log_writer.as_deref(),
            &session.session_id,
            &session.thread_id,
            turn_snapshot.as_ref().map(|turn| turn.turn_id.as_str()),
            workflow_resume_audit_events,
        )?;
        if let Some(entry) = session_approval_cache_entry {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            super::approval_cache::insert_entry(
                &mut state.session_approval_cache,
                &params.session_id,
                entry,
            );
        }
        if decision.is_some_and(AgentSessionApprovalDecision::is_cancel) {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            super::approval_cache::remove_session(
                &mut state.session_approval_cache,
                &params.session_id,
            );
        }

        Ok(RuntimeCoreOutput {
            response: AgentSessionActionRespondResponse {},
            events,
        })
    }

    pub async fn respond_permission(
        &self,
        request: PermissionRespondRequest,
    ) -> Result<(), RuntimeCoreError> {
        for (field, value) in [
            ("session_id", request.session_id.as_str()),
            ("thread_id", request.thread_id.as_str()),
            ("turn_id", request.turn_id.as_str()),
            ("request_id", request.request_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(RuntimeCoreError::Backend(format!(
                    "permission response requires canonical {field}"
                )));
            }
        }
        let session = self
            .session_loops
            .get_existing(&request.session_id)
            .await
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(request.session_id.clone()))?;
        self.backend.resolve_permission_action(&request).await?;
        let response = serde_json::to_value(&request.response)
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
        session
            .respond_permission(Some(&request.turn_id), request.request_id, response)
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))
    }

    pub async fn respond_dynamic_tool(
        &self,
        request: crate::DynamicToolRespondRequest,
    ) -> Result<(), RuntimeCoreError> {
        for (field, value) in [
            ("session_id", request.session_id.as_str()),
            ("thread_id", request.thread_id.as_str()),
            ("turn_id", request.turn_id.as_str()),
            ("call_id", request.call_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(RuntimeCoreError::Backend(format!(
                    "dynamic tool response requires canonical {field}"
                )));
            }
        }
        {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(&request.session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(request.session_id.clone()))?;
            if stored.session.thread_id != request.thread_id {
                return Err(RuntimeCoreError::Backend(
                    "dynamic tool response thread identity mismatch".to_string(),
                ));
            }
            if !stored
                .turns
                .iter()
                .any(|turn| turn.turn_id == request.turn_id)
            {
                return Err(RuntimeCoreError::Backend(
                    "dynamic tool response turn identity mismatch".to_string(),
                ));
            }
        }
        let session = self
            .session_loops
            .get_existing(&request.session_id)
            .await
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(request.session_id.clone()))?;
        let response = serde_json::to_value(request.response)
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
        session
            .respond_dynamic_tool(Some(&request.turn_id), request.call_id, response)
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))
    }

    async fn dispatch_live_action_response(
        &self,
        request: &ActionRespondRequest,
    ) -> Result<(), RuntimeCoreError> {
        let turn_id = request.turn.as_ref().map(|turn| turn.turn_id.as_str());
        let response = request
            .user_data
            .clone()
            .or_else(|| {
                request
                    .response
                    .as_ref()
                    .map(|response| json!({ "answer": response }))
            })
            .unwrap_or_else(|| json!({ "confirmed": request.confirmed }));
        let session = self
            .session_loops
            .get_or_create(&request.session.session_id)
            .await;
        let result = match request.action_type {
            AgentSessionActionType::ToolConfirmation => {
                session
                    .approve(turn_id, &request.request_id, response)
                    .await
            }
            AgentSessionActionType::AskUser => {
                session
                    .answer_user_input(turn_id, &request.request_id, response)
                    .await
            }
            AgentSessionActionType::Elicitation => {
                session
                    .resolve_mcp_elicitation(turn_id, &request.request_id, response)
                    .await
            }
        };
        result.map_err(|error| {
            tracing::warn!(
                session_id = %request.session.session_id,
                request_id = %request.request_id,
                error = %error,
                "failed to route action response to the active session task"
            );
            RuntimeCoreError::ActionResponse {
                code: "action_response_not_pending".to_string(),
                request_id: request.request_id.clone(),
            }
        })
    }

    pub fn events_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
        let state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let stored = state
            .sessions
            .get(session_id)
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
        Ok(stored.events.clone())
    }

    pub fn append_external_runtime_events(
        &self,
        session_id: &str,
        turn_id: Option<&str>,
        runtime_events: Vec<RuntimeEvent>,
    ) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
        self.event_appender()
            .append_external_runtime_events(session_id, turn_id, runtime_events)
    }

    pub(crate) fn append_artifact_snapshot(
        &self,
        session_id: &str,
        turn_id: Option<&str>,
        payload: Value,
    ) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
        self.event_appender()
            .append_artifact_snapshot(session_id, turn_id, payload)
    }

    pub(in crate::runtime) fn append_runtime_events(
        &self,
        session_id: &str,
        thread_id: &str,
        turn_id: Option<&str>,
        runtime_events: Vec<RuntimeEvent>,
    ) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
        append_runtime_events_to_state(
            &self.state,
            self.file_checkpoint_snapshot_store.as_ref(),
            self.output_snapshot_store.as_ref(),
            self.sidecar_store.as_deref(),
            self.event_log_writer.as_deref(),
            self.trace_event_writer.as_deref(),
            self.projection_store.as_deref(),
            Some(&self.session_loops),
            session_id,
            thread_id,
            turn_id,
            runtime_events,
        )
    }
}

fn apply_reconciled_thread_model_selection(
    runtime_options: &mut Option<RuntimeOptions>,
    settings: &app_server_protocol::protocol::v2::ThreadSettings,
) {
    let request = runtime_options
        .get_or_insert_with(RuntimeOptions::default)
        .runtime_request_mut();
    request.provider_config = None;
    request.provider_preference = Some(settings.model_provider.clone());
    request.model_preference = Some(settings.model.clone());
    request.reasoning_effort = settings.effort.clone();
    request.service_tier = settings.service_tier.clone();
    request.collaboration_mode = Some(settings.collaboration_mode.clone());
    if let Some(metadata) = request.metadata.as_mut().and_then(Value::as_object_mut) {
        metadata.remove("agentControlRoute");
    }
}

fn runtime_error_from_session_task_failure(error: RuntimeSessionTaskFailure) -> RuntimeCoreError {
    if error.reason_code.as_deref() == Some("usage_limit_exceeded") {
        RuntimeCoreError::UsageLimitExceeded(error.message)
    } else {
        RuntimeCoreError::Backend(error.message)
    }
}

fn steer_runtime_error(error: RuntimeSessionLoopError, expected_turn_id: &str) -> RuntimeCoreError {
    match error {
        RuntimeSessionLoopError::InvalidTask(message)
            if message.contains("does not accept steer input") =>
        {
            RuntimeCoreError::InvalidRequest(format!(
                "turn {expected_turn_id} does not accept steering"
            ))
        }
        RuntimeSessionLoopError::InvalidTask(message) if message.contains("has no active turn") => {
            RuntimeCoreError::InvalidRequest("no active turn to steer".to_string())
        }
        RuntimeSessionLoopError::InvalidTask(message)
            if message.contains("no longer active") || message.contains("already finishing") =>
        {
            RuntimeCoreError::InvalidRequest(format!(
                "expected active turn id `{expected_turn_id}` is no longer active"
            ))
        }
        RuntimeSessionLoopError::InvalidTask(message) => RuntimeCoreError::InvalidRequest(message),
        RuntimeSessionLoopError::Closed => {
            RuntimeCoreError::Backend("runtime session submission loop is closed".to_string())
        }
        RuntimeSessionLoopError::OperationFailed(message) => RuntimeCoreError::Backend(message),
    }
}
