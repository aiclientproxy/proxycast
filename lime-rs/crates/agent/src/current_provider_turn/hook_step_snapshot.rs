use crate::protocol::AgentEvent;
use agent_protocol::turn_context::TurnContextOverride;
use agent_runtime::provider_turn::{
    RuntimeHookSnapshotFuture, RuntimeHookSnapshotSource, RuntimeHookSnapshotSourceHandle,
    RuntimeHookStepSnapshot,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tool_runtime::hook_discovery::{
    discover_hooks, plugin_sources_from_activations, HookDiscoveryInput,
};
use tool_runtime::hook_runtime::{
    CommandHookReporter, HookLifecycleEmitter, HookLifecycleEvent, HookLifecyclePhase,
};

pub(super) fn current_hook_step_snapshot_source(
    turn_context: Option<TurnContextOverride>,
    working_directory: PathBuf,
    event_sender: mpsc::UnboundedSender<AgentEvent>,
) -> Option<RuntimeHookSnapshotSourceHandle> {
    let codex_home = lime_core::app_paths::resolve_codex_home_dir()?;
    Some(RuntimeHookSnapshotSourceHandle::new(Arc::new(
        CurrentHookStepSnapshotSource {
            input: HookDiscoveryInput {
                codex_home,
                cwd: working_directory,
                plugins: plugin_sources_from_activations(&plugin_activations(
                    turn_context.as_ref(),
                )),
            },
            emitter: Arc::new(AgentHookLifecycleEmitter { event_sender }),
        },
    )))
}

struct CurrentHookStepSnapshotSource {
    input: HookDiscoveryInput,
    emitter: Arc<dyn HookLifecycleEmitter>,
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
            Ok(RuntimeHookStepSnapshot {
                hooks: snapshots,
                reporter: Arc::new(CommandHookReporter::new(hooks, self.emitter.clone())),
            })
        })
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

fn plugin_activations(turn_context: Option<&TurnContextOverride>) -> Vec<serde_json::Value> {
    turn_context
        .and_then(|context| context.metadata.get("plugin_activations"))
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn plugin_hook_sources_only_use_absolute_activated_packages() {
        let context = TurnContextOverride {
            metadata: std::collections::HashMap::from([(
                "plugin_activations".to_string(),
                json!([
                    {"pluginId":"docs", "packageSourceUri":"/tmp/docs"},
                    {"pluginId":"relative", "packageSourceUri":"plugins/relative"},
                    {"pluginId":"docs", "packageSourceUri":"/tmp/duplicate"}
                ]),
            )]),
            ..TurnContextOverride::default()
        };

        assert_eq!(
            plugin_sources_from_activations(&plugin_activations(Some(&context))),
            vec![tool_runtime::hook_discovery::HookPluginSource {
                plugin_id: "docs".to_string(),
                package_root: PathBuf::from("/tmp/docs"),
            }]
        );
    }
}
