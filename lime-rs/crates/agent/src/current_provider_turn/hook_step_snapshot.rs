use crate::protocol::AgentEvent;
use agent_protocol::turn_context::TurnContextOverride;
use agent_runtime::provider_turn::{
    RuntimeHookSnapshotFuture, RuntimeHookSnapshotSource, RuntimeHookSnapshotSourceHandle,
    RuntimeHookStepSnapshot,
};
use agent_runtime::session_loop::RuntimeSessionInputHandle;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tool_runtime::hook_discovery::{discover_hooks, HookDiscoveryInput};
use tool_runtime::hook_runtime::{
    AsyncHookResultSink, CommandHookReporter, HookLifecycleEmitter, HookLifecycleEvent,
    HookLifecyclePhase,
};

pub(super) fn current_hook_step_snapshot_source(
    _turn_context: Option<TurnContextOverride>,
    working_directory: PathBuf,
    event_sender: mpsc::UnboundedSender<AgentEvent>,
    pending_input: Option<RuntimeSessionInputHandle>,
    cancel_token: Option<CancellationToken>,
) -> Option<RuntimeHookSnapshotSourceHandle> {
    let codex_home = lime_core::app_paths::resolve_codex_home_dir()?;
    Some(RuntimeHookSnapshotSourceHandle::new(Arc::new(
        CurrentHookStepSnapshotSource {
            input: HookDiscoveryInput {
                codex_home,
                cwd: working_directory,
                plugins: Vec::new(),
            },
            emitter: Arc::new(AgentHookLifecycleEmitter {
                event_sender: event_sender.clone(),
            }),
            event_sender,
            pending_input,
            cancel_token,
        },
    )))
}

struct CurrentHookStepSnapshotSource {
    input: HookDiscoveryInput,
    emitter: Arc<dyn HookLifecycleEmitter>,
    event_sender: mpsc::UnboundedSender<AgentEvent>,
    pending_input: Option<RuntimeSessionInputHandle>,
    cancel_token: Option<CancellationToken>,
}

impl RuntimeHookSnapshotSource for CurrentHookStepSnapshotSource {
    fn capture(&self) -> RuntimeHookSnapshotFuture<'_> {
        Box::pin(async move {
            let report = discover_hooks(&self.input);
            if !report.errors.is_empty() {
                return Err(report
                    .errors
                    .iter()
                    .map(|error| format!("{}: {}", error.path.display(), error.message))
                    .collect::<Vec<_>>()
                    .join("; "));
            }
            let hooks = report.executable_hooks();
            let snapshots = hooks.iter().map(|hook| hook.snapshot.clone()).collect();
            let reporter = CommandHookReporter::new(hooks, self.emitter.clone());
            let reporter = match self.pending_input.clone() {
                Some(pending_input) => {
                    reporter.with_async_result_sink(Arc::new(AgentAsyncHookResultSink {
                        pending_input,
                        cancel_token: self.cancel_token.clone(),
                        event_sender: self.event_sender.clone(),
                    }))
                }
                None => reporter,
            };
            Ok(RuntimeHookStepSnapshot {
                hooks: snapshots,
                reporter: Arc::new(reporter),
            })
        })
    }
}

struct AgentAsyncHookResultSink {
    pending_input: RuntimeSessionInputHandle,
    cancel_token: Option<CancellationToken>,
    event_sender: mpsc::UnboundedSender<AgentEvent>,
}

impl AsyncHookResultSink for AgentAsyncHookResultSink {
    fn submit(&self, run: agent_protocol::hook::HookRunSummary) {
        if self
            .cancel_token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return;
        }
        let contexts = run
            .entries
            .iter()
            .filter(|entry| entry.kind == agent_protocol::hook::HookOutputEntryKind::Context)
            .map(|entry| entry.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>();
        for entry in run
            .entries
            .iter()
            .filter(|entry| entry.kind != agent_protocol::hook::HookOutputEntryKind::Context)
        {
            let _ = self.event_sender.send(AgentEvent::Warning {
                code: Some("async_hook_result".to_string()),
                message: entry.text.clone(),
            });
        }
        if contexts.is_empty() {
            return;
        }
        let text = format!(
            "<hook_additional_context>\n{}\n</hook_additional_context>",
            contexts.join("\n")
        );
        let pending_input = self.pending_input.clone();
        let cancel_token = self.cancel_token.clone();
        tokio::spawn(async move {
            if cancel_token
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
            {
                return;
            }
            let _ = pending_input.inject_developer_input(text).await;
        });
    }
}

struct AgentHookLifecycleEmitter {
    event_sender: mpsc::UnboundedSender<AgentEvent>,
}

impl HookLifecycleEmitter for AgentHookLifecycleEmitter {
    fn emit(&self, event: HookLifecycleEvent) {
        let event = match event.phase {
            HookLifecyclePhase::Started => AgentEvent::HookStarted {
                turn_id: event.turn_id,
                run: event.run,
            },
            HookLifecyclePhase::Completed => AgentEvent::HookCompleted {
                turn_id: event.turn_id,
                run: event.run,
            },
        };
        let _ = self.event_sender.send(event);
    }
}
