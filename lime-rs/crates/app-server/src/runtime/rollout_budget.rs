use super::error::RuntimeCoreError;
use super::event_store::append_runtime_events_to_state;
use super::{
    EventLogWriter, FileCheckpointSnapshotStore, OutputSnapshotStore, ProjectionStore,
    RuntimeCoreState, RuntimeEvent, SidecarStore, TraceEventWriter,
};
use app_server_protocol::AgentEvent;
use lime_agent::AgentTokenUsage;
use lime_core::config::RolloutBudgetConfig;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

pub(crate) const ROLLOUT_BUDGET_REMINDER_EVENT_TYPE: &str = "rollout_budget.reminder";

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RolloutBudgetReminder {
    pub remaining_tokens: i64,
    pub reminder_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AttemptKey {
    root_thread_id: String,
    turn_id: String,
    thread_id: String,
    route_attempt: u32,
    attempt: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ReminderKey {
    root_thread_id: String,
    thread_id: String,
    window_id: String,
}

#[derive(Debug, Default)]
struct BudgetState {
    weighted_tokens_used: HashMap<String, f64>,
    exhausted_roots: HashSet<String>,
    attempts: HashMap<AttemptKey, f64>,
    delivered: HashMap<ReminderKey, usize>,
    hydrated_roots: HashSet<String>,
}

/// One shared budget owner for every root-thread tree.
#[derive(Debug, Clone)]
pub(crate) struct RolloutBudget {
    enabled: bool,
    config: Arc<RolloutBudgetConfig>,
    state: Arc<Mutex<BudgetState>>,
}

struct SamplingReminderSource {
    state: Arc<Mutex<RuntimeCoreState>>,
    file_checkpoint_snapshot_store: Arc<dyn FileCheckpointSnapshotStore>,
    output_snapshot_store: Arc<dyn OutputSnapshotStore>,
    sidecar_store: Option<Arc<SidecarStore>>,
    event_log_writer: Option<Arc<EventLogWriter>>,
    trace_event_writer: Option<Arc<TraceEventWriter>>,
    projection_store: Option<Arc<ProjectionStore>>,
    budget: Arc<RolloutBudget>,
    root_thread_id: String,
    session_id: String,
    thread_id: String,
    turn_id: String,
    route_deliveries: Mutex<HashSet<(usize, String)>>,
}

pub(crate) struct SamplingReminderSourceOptions {
    pub state: Arc<Mutex<RuntimeCoreState>>,
    pub file_checkpoint_snapshot_store: Arc<dyn FileCheckpointSnapshotStore>,
    pub output_snapshot_store: Arc<dyn OutputSnapshotStore>,
    pub sidecar_store: Option<Arc<SidecarStore>>,
    pub event_log_writer: Option<Arc<EventLogWriter>>,
    pub trace_event_writer: Option<Arc<TraceEventWriter>>,
    pub projection_store: Option<Arc<ProjectionStore>>,
    pub budget: Arc<RolloutBudget>,
    pub root_thread_id: String,
    pub session_id: String,
    pub thread_id: String,
    pub turn_id: String,
}

pub(crate) fn sampling_reminder_source(
    options: SamplingReminderSourceOptions,
) -> agent_runtime::session_config::RolloutBudgetReminderSourceHandle {
    agent_runtime::session_config::RolloutBudgetReminderSourceHandle::new(Arc::new(
        SamplingReminderSource {
            state: options.state,
            file_checkpoint_snapshot_store: options.file_checkpoint_snapshot_store,
            output_snapshot_store: options.output_snapshot_store,
            sidecar_store: options.sidecar_store,
            event_log_writer: options.event_log_writer,
            trace_event_writer: options.trace_event_writer,
            projection_store: options.projection_store,
            budget: options.budget,
            root_thread_id: options.root_thread_id,
            session_id: options.session_id,
            thread_id: options.thread_id,
            turn_id: options.turn_id,
            route_deliveries: Mutex::new(HashSet::new()),
        },
    ))
}

impl agent_runtime::session_config::RolloutBudgetReminderSource for SamplingReminderSource {
    fn next_reminder(
        &self,
        route_attempt: usize,
    ) -> Result<Option<agent_runtime::session_config::RolloutBudgetReminder>, String> {
        let window_id = {
            let state = self
                .state
                .lock()
                .map_err(|_| "runtime core state mutex poisoned".to_string())?;
            let stored = state
                .sessions
                .get(&self.session_id)
                .ok_or_else(|| format!("session not found: {}", self.session_id))?;
            window_id(&self.thread_id, &stored.events)
        };
        let reminder =
            self.budget
                .pending_reminder(&self.root_thread_id, &self.thread_id, &window_id);
        if reminder.is_none() {
            let existing_reminder = {
                let state = self
                    .state
                    .lock()
                    .map_err(|_| "runtime core state mutex poisoned".to_string())?;
                let stored = state
                    .sessions
                    .get(&self.session_id)
                    .ok_or_else(|| format!("session not found: {}", self.session_id))?;
                stored
                    .events
                    .iter()
                    .rev()
                    .find(|event| {
                        event.turn_id.as_deref() == Some(self.turn_id.as_str())
                            && event.event_type == ROLLOUT_BUDGET_REMINDER_EVENT_TYPE
                            && event
                                .payload
                                .get("windowId")
                                .and_then(serde_json::Value::as_str)
                                == Some(window_id.as_str())
                    })
                    .cloned()
            };
            let Some(event) = existing_reminder else {
                return Ok(None);
            };
            if !self
                .route_deliveries
                .lock()
                .map_err(|_| "rollout reminder delivery mutex poisoned".to_string())?
                .insert((route_attempt, event.event_id.clone()))
            {
                return Ok(None);
            }
            return Ok(Some(agent_runtime::session_config::RolloutBudgetReminder {
                remaining_tokens: event
                    .payload
                    .get("remainingTokens")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or_default(),
                reminder_index: event
                    .payload
                    .get("reminderIndex")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or_default() as usize,
                window_id,
                durable_event_id: event.event_id,
            }));
        }
        let reminder = reminder.expect("pending reminder checked above");
        let text = format!(
            "<rollout_budget>\nYou have {} weighted tokens left in the shared session token budget.\n</rollout_budget>",
            reminder.remaining_tokens
        );
        let events = append_runtime_events_to_state(
            &self.state,
            self.file_checkpoint_snapshot_store.as_ref(),
            self.output_snapshot_store.as_ref(),
            self.sidecar_store.as_deref(),
            self.event_log_writer.as_deref(),
            self.trace_event_writer.as_deref(),
            self.projection_store.as_deref(),
            None,
            &self.session_id,
            &self.thread_id,
            Some(&self.turn_id),
            vec![RuntimeEvent::new(
                ROLLOUT_BUDGET_REMINDER_EVENT_TYPE,
                serde_json::json!({
                    "remainingTokens": reminder.remaining_tokens,
                    "reminderIndex": reminder.reminder_index,
                    "windowId": window_id,
                    "routeAttempt": route_attempt,
                    "text": text,
                }),
            )],
        )
        .map_err(|error| error.to_string())?;
        let durable_event_id = events
            .iter()
            .find(|event| event.event_type == ROLLOUT_BUDGET_REMINDER_EVENT_TYPE)
            .map(|event| event.event_id.clone())
            .ok_or_else(|| "rollout budget reminder append omitted durable event".to_string())?;
        self.budget.mark_reminder_delivered(
            &self.root_thread_id,
            &self.thread_id,
            &window_id,
            &reminder,
        );
        self.route_deliveries
            .lock()
            .map_err(|_| "rollout reminder delivery mutex poisoned".to_string())?
            .insert((route_attempt, durable_event_id.clone()));
        Ok(Some(agent_runtime::session_config::RolloutBudgetReminder {
            remaining_tokens: reminder.remaining_tokens,
            reminder_index: reminder.reminder_index,
            window_id,
            durable_event_id,
        }))
    }
}

impl RolloutBudget {
    pub(crate) fn new(config: Option<RolloutBudgetConfig>) -> Result<Self, RuntimeCoreError> {
        let enabled = config.is_some();
        let config = config.unwrap_or_else(|| RolloutBudgetConfig {
            limit_tokens: 1,
            reminder_at_remaining_tokens: Vec::new(),
            sampling_token_weight: 1.0,
            prefill_token_weight: 1.0,
        });
        config
            .validate()
            .map_err(RuntimeCoreError::InvalidRolloutBudgetConfig)?;
        Ok(Self {
            enabled,
            config: Arc::new(config),
            state: Arc::new(Mutex::new(BudgetState::default())),
        })
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn check_admission(&self, root_thread_id: &str) -> Result<(), RuntimeCoreError> {
        if self
            .state
            .lock()
            .expect("rollout budget mutex poisoned")
            .exhausted_roots
            .contains(root_thread_id)
        {
            Err(RuntimeCoreError::RolloutBudgetExhausted)
        } else {
            Ok(())
        }
    }

    pub(crate) fn needs_hydration(&self, root_thread_id: &str) -> bool {
        self.enabled()
            && !self
                .state
                .lock()
                .expect("rollout budget mutex poisoned")
                .hydrated_roots
                .contains(root_thread_id)
    }

    pub(crate) fn restore_root<'a>(
        &self,
        root_thread_id: &str,
        histories: impl IntoIterator<Item = (&'a str, &'a [AgentEvent])>,
    ) -> Result<(), RuntimeCoreError> {
        if !self.enabled() {
            return Ok(());
        }
        for (thread_id, events) in histories {
            for event in events {
                match event.event_type.as_str() {
                    "provider.usage" => {
                        let Some(usage) = event.payload.get("usage") else {
                            continue;
                        };
                        let usage = serde_json::from_value::<AgentTokenUsage>(usage.clone())
                            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
                        let attempt = event
                            .payload
                            .get("attempt")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or_default() as u32;
                        let route_attempt = event
                            .payload
                            .get("routeAttempt")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(1) as u32;
                        self.record_usage(
                            root_thread_id,
                            thread_id,
                            event.turn_id.as_deref().unwrap_or_default(),
                            route_attempt,
                            attempt,
                            &usage,
                        )?;
                    }
                    ROLLOUT_BUDGET_REMINDER_EVENT_TYPE => {
                        let Some(window_id) = event
                            .payload
                            .get("windowId")
                            .and_then(serde_json::Value::as_str)
                        else {
                            continue;
                        };
                        let Some(reminder_index) = event
                            .payload
                            .get("reminderIndex")
                            .and_then(serde_json::Value::as_u64)
                            .and_then(|value| usize::try_from(value).ok())
                        else {
                            continue;
                        };
                        self.mark_reminder_delivered(
                            root_thread_id,
                            thread_id,
                            window_id,
                            &RolloutBudgetReminder {
                                remaining_tokens: event
                                    .payload
                                    .get("remainingTokens")
                                    .and_then(serde_json::Value::as_i64)
                                    .unwrap_or_default(),
                                reminder_index,
                            },
                        );
                    }
                    _ => {}
                }
            }
        }
        self.state
            .lock()
            .expect("rollout budget mutex poisoned")
            .hydrated_roots
            .insert(root_thread_id.to_string());
        Ok(())
    }

    pub(crate) fn record_usage(
        &self,
        root_thread_id: &str,
        thread_id: &str,
        turn_id: &str,
        route_attempt: u32,
        attempt: u32,
        usage: &AgentTokenUsage,
    ) -> Result<bool, RuntimeCoreError> {
        if !self.enabled() {
            return Ok(false);
        }
        let units = usage_units(usage, &self.config)?;
        let mut state = self.state.lock().expect("rollout budget mutex poisoned");
        let key = AttemptKey {
            root_thread_id: root_thread_id.to_string(),
            thread_id: thread_id.to_string(),
            turn_id: turn_id.to_string(),
            route_attempt,
            attempt,
        };
        let previous = state.attempts.insert(key, units).unwrap_or_default();
        let weighted_tokens_used = state
            .weighted_tokens_used
            .entry(root_thread_id.to_string())
            .or_default();
        *weighted_tokens_used += units - previous;
        let exhausted = *weighted_tokens_used >= self.config.limit_tokens as f64;
        if exhausted {
            state.exhausted_roots.insert(root_thread_id.to_string());
        }
        Ok(exhausted)
    }

    pub(crate) fn pending_reminder(
        &self,
        root_thread_id: &str,
        thread_id: &str,
        window_id: &str,
    ) -> Option<RolloutBudgetReminder> {
        if !self.enabled() {
            return None;
        }
        let state = self.state.lock().expect("rollout budget mutex poisoned");
        let remaining = (self.config.limit_tokens as f64
            - state
                .weighted_tokens_used
                .get(root_thread_id)
                .copied()
                .unwrap_or_default())
        .max(0.0)
        .floor() as i64;
        let index = self
            .config
            .reminder_at_remaining_tokens
            .iter()
            .filter(|threshold| remaining <= **threshold)
            .count();
        if index == 0 {
            return None;
        }
        let key = ReminderKey {
            root_thread_id: root_thread_id.to_string(),
            thread_id: thread_id.to_string(),
            window_id: window_id.to_string(),
        };
        if state
            .delivered
            .get(&key)
            .is_some_and(|delivered| *delivered >= index)
        {
            return None;
        }
        Some(RolloutBudgetReminder {
            remaining_tokens: remaining,
            reminder_index: index,
        })
    }

    pub(crate) fn mark_reminder_delivered(
        &self,
        root_thread_id: &str,
        thread_id: &str,
        window_id: &str,
        reminder: &RolloutBudgetReminder,
    ) {
        self.state
            .lock()
            .expect("rollout budget mutex poisoned")
            .delivered
            .insert(
                ReminderKey {
                    root_thread_id: root_thread_id.to_string(),
                    thread_id: thread_id.to_string(),
                    window_id: window_id.to_string(),
                },
                reminder.reminder_index,
            );
    }

    #[cfg(test)]
    fn rearm(&self, root_thread_id: &str, thread_id: &str, window_id: &str) {
        self.state
            .lock()
            .expect("rollout budget mutex poisoned")
            .delivered
            .remove(&ReminderKey {
                root_thread_id: root_thread_id.to_string(),
                thread_id: thread_id.to_string(),
                window_id: window_id.to_string(),
            });
    }
}

pub(crate) fn window_id(thread_id: &str, events: &[AgentEvent]) -> String {
    let mut window_id = thread_id.to_string();
    for event in events {
        match event.event_type.as_str() {
            "context.compaction.completed" => {
                window_id = event
                    .payload
                    .get("artifact")
                    .filter(|value| value.is_object())
                    .and_then(|artifact| artifact.get("windowId"))
                    .or_else(|| event.payload.get("windowId"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .unwrap_or(&event.event_id)
                    .to_string();
            }
            "turn.rollback" | "history.rollback" | "thread.rollback" => {
                window_id = event.event_id.clone();
            }
            _ => {}
        }
    }
    window_id
}

fn usage_units(
    usage: &AgentTokenUsage,
    config: &RolloutBudgetConfig,
) -> Result<f64, RuntimeCoreError> {
    if let Some(units) = usage.codex_rollout_budget_units.as_ref() {
        let units = units
            .as_f64()
            .ok_or_else(|| RuntimeCoreError::InvalidRolloutBudgetUnits)?;
        if !units.is_finite() || units < 0.0 {
            return Err(RuntimeCoreError::InvalidRolloutBudgetUnits);
        }
        return Ok(units);
    }
    let non_cached = usage
        .input_tokens
        .saturating_sub(usage.cached_input_tokens.unwrap_or_default())
        .saturating_sub(usage.cache_creation_input_tokens.unwrap_or_default());
    Ok(
        f64::from(usage.output_tokens) * config.sampling_token_weight
            + f64::from(non_cached) * config.prefill_token_weight,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> RolloutBudgetConfig {
        RolloutBudgetConfig {
            limit_tokens: 100,
            reminder_at_remaining_tokens: vec![75, 50, 25],
            sampling_token_weight: 1.0,
            prefill_token_weight: 1.0,
        }
    }

    fn usage(input: u32, output: u32) -> AgentTokenUsage {
        AgentTokenUsage {
            input_tokens: input,
            output_tokens: output,
            cached_input_tokens: None,
            cache_creation_input_tokens: None,
            codex_rollout_budget_units: None,
        }
    }

    fn event(
        event_id: &str,
        thread_id: &str,
        turn_id: &str,
        event_type: &str,
        payload: serde_json::Value,
    ) -> AgentEvent {
        AgentEvent {
            event_id: event_id.to_string(),
            sequence: 1,
            session_id: format!("session-{thread_id}"),
            thread_id: Some(thread_id.to_string()),
            turn_id: Some(turn_id.to_string()),
            event_type: event_type.to_string(),
            timestamp: "2026-08-14T00:00:00Z".to_string(),
            payload,
        }
    }

    #[test]
    fn reminder_is_absent_before_the_first_threshold() {
        let budget = RolloutBudget::new(Some(config())).expect("valid config");

        assert!(budget.pending_reminder("tree", "root", "window").is_none());
    }

    #[test]
    fn attempt_snapshots_only_charge_delta() {
        let budget = RolloutBudget::new(Some(config())).expect("valid config");
        assert!(!budget
            .record_usage("tree", "root", "turn", 1, 1, &usage(10, 10))
            .unwrap());
        assert!(!budget
            .record_usage("tree", "root", "turn", 1, 1, &usage(10, 15))
            .unwrap());
        assert_eq!(
            budget.pending_reminder("tree", "root", "window"),
            Some(RolloutBudgetReminder {
                remaining_tokens: 75,
                reminder_index: 1
            })
        );
    }

    #[test]
    fn root_and_child_share_usage_while_other_roots_are_isolated() {
        let budget = RolloutBudget::new(Some(config())).expect("valid config");
        budget
            .record_usage("tree-a", "root-a", "turn-root", 1, 1, &usage(10, 10))
            .unwrap();
        budget
            .record_usage("tree-a", "child-a", "turn-child", 1, 1, &usage(5, 5))
            .unwrap();

        assert_eq!(
            budget.pending_reminder("tree-a", "root-a", "window-a"),
            Some(RolloutBudgetReminder {
                remaining_tokens: 70,
                reminder_index: 1,
            })
        );
        assert!(budget
            .pending_reminder("tree-b", "root-b", "window-b")
            .is_none());
    }

    #[test]
    fn reroute_attempts_keep_provider_snapshots_isolated() {
        let budget = RolloutBudget::new(Some(config())).expect("valid config");
        budget
            .record_usage("tree", "root", "turn", 1, 1, &usage(10, 20))
            .unwrap();
        budget
            .record_usage("tree", "root", "turn", 2, 1, &usage(5, 15))
            .unwrap();
        budget
            .record_usage("tree", "root", "turn", 2, 1, &usage(5, 20))
            .unwrap();

        assert_eq!(
            budget.pending_reminder("tree", "root", "window"),
            Some(RolloutBudgetReminder {
                remaining_tokens: 45,
                reminder_index: 2,
            })
        );
    }

    #[test]
    fn provider_units_are_validated_and_override_tokens() {
        let budget = RolloutBudget::new(Some(config())).expect("valid config");
        let mut value = usage(1, 1);
        value.codex_rollout_budget_units = Some(json!(40.5).as_number().unwrap().clone());
        budget
            .record_usage("tree", "root", "turn", 1, 1, &value)
            .unwrap();
        assert_eq!(
            budget
                .pending_reminder("tree", "root", "window")
                .unwrap()
                .remaining_tokens,
            59
        );
        value.codex_rollout_budget_units = Some(json!(-1).as_number().unwrap().clone());
        assert!(matches!(
            budget.record_usage("tree", "root", "turn", 1, 2, &value),
            Err(RuntimeCoreError::InvalidRolloutBudgetUnits)
        ));
    }

    #[test]
    fn reminders_rearm_per_window_without_refund() {
        let budget = RolloutBudget::new(Some(config())).expect("valid config");
        budget
            .record_usage("tree", "root", "turn", 1, 1, &usage(10, 20))
            .unwrap();
        let reminder = budget.pending_reminder("tree", "root", "a").unwrap();
        budget.mark_reminder_delivered("tree", "root", "a", &reminder);
        assert!(budget.pending_reminder("tree", "root", "a").is_none());
        assert!(budget.pending_reminder("tree", "root", "b").is_some());
        budget.rearm("tree", "root", "a");
        assert!(budget.pending_reminder("tree", "root", "a").is_some());
    }

    #[test]
    fn compaction_and_rollback_create_new_reminder_windows() {
        let mut events = Vec::new();
        assert_eq!(window_id("thread", &events), "thread");

        events.push(event(
            "compact-1",
            "thread",
            "turn-1",
            "context.compaction.completed",
            json!({ "artifact": { "windowId": "window-2" } }),
        ));
        assert_eq!(window_id("thread", &events), "window-2");

        events.push(event(
            "rollback-1",
            "thread",
            "turn-2",
            "history.rollback",
            json!({ "rollbackToSequence": 1 }),
        ));
        assert_eq!(window_id("thread", &events), "rollback-1");
    }

    #[test]
    fn restore_root_recovers_usage_delivery_and_exhaustion() {
        let budget = RolloutBudget::new(Some(config())).expect("valid config");
        let events = vec![
            event(
                "usage-1",
                "child",
                "turn-1",
                "provider.usage",
                json!({
                    "attempt": 1,
                    "routeAttempt": 2,
                    "usage": usage(30, 20),
                }),
            ),
            event(
                "reminder-1",
                "child",
                "turn-1",
                ROLLOUT_BUDGET_REMINDER_EVENT_TYPE,
                json!({
                    "remainingTokens": 50,
                    "reminderIndex": 2,
                    "windowId": "window-1",
                }),
            ),
        ];

        budget
            .restore_root("tree", [("child", events.as_slice())])
            .unwrap();

        assert!(!budget.needs_hydration("tree"));
        assert!(budget
            .pending_reminder("tree", "child", "window-1")
            .is_none());
        assert_eq!(
            budget.pending_reminder("tree", "child", "window-2"),
            Some(RolloutBudgetReminder {
                remaining_tokens: 50,
                reminder_index: 2,
            })
        );

        assert!(budget
            .record_usage("tree", "child", "turn-2", 1, 1, &usage(25, 25))
            .unwrap());
        assert!(matches!(
            budget.check_admission("tree"),
            Err(RuntimeCoreError::RolloutBudgetExhausted)
        ));
        assert!(budget.check_admission("other-tree").is_ok());
    }
}
