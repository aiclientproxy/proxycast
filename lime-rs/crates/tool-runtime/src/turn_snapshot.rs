use crate::tool_definition::{RuntimeToolDefinition, RuntimeToolExposure};
pub use agent_protocol::hook::{
    HookEventName as RuntimeHookEventName, HookExecutionMode as RuntimeHookExecutionMode,
    HookHandlerType as RuntimeHookHandlerType, HookScope as RuntimeHookScope,
    HookSnapshot as RuntimeHookSnapshot, HookSource as RuntimeHookSource,
    HookTrustStatus as RuntimeHookTrustStatus,
};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RuntimeToolIdentity {
    pub namespace: Option<String>,
    pub name: String,
}

impl RuntimeToolIdentity {
    pub fn plain(name: impl Into<String>) -> Self {
        Self {
            namespace: None,
            name: name.into(),
        }
    }

    pub fn namespaced(namespace: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            namespace: Some(namespace.into()),
            name: name.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeToolSnapshot {
    pub identity: RuntimeToolIdentity,
    pub definition: RuntimeToolDefinition,
    pub exposure: RuntimeToolExposure,
    pub supports_parallel: bool,
    /// Final model-visible selection for this sampling step.
    ///
    /// Exposure alone is insufficient because code mode, feature gates, and hosted
    /// specs can change the tool surface before the step snapshot is captured.
    pub model_visible: bool,
}

impl RuntimeToolSnapshot {
    pub fn new(
        identity: RuntimeToolIdentity,
        definition: RuntimeToolDefinition,
        exposure: RuntimeToolExposure,
        supports_parallel: bool,
        model_visible: bool,
    ) -> Self {
        Self {
            identity,
            definition,
            exposure,
            supports_parallel,
            model_visible,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeTurnSnapshot {
    tools: Vec<RuntimeToolSnapshot>,
    hooks: Vec<RuntimeHookSnapshot>,
}

#[derive(Deserialize)]
struct RuntimeTurnSnapshotWire {
    tools: Vec<RuntimeToolSnapshot>,
    hooks: Vec<RuntimeHookSnapshot>,
}

impl<'de> Deserialize<'de> for RuntimeTurnSnapshot {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RuntimeTurnSnapshotWire::deserialize(deserializer)?;
        Self::try_new(wire.tools, wire.hooks).map_err(|error| {
            serde::de::Error::custom(format!("invalid runtime turn snapshot: {error:?}"))
        })
    }
}

impl RuntimeTurnSnapshot {
    pub fn try_new(
        tools: Vec<RuntimeToolSnapshot>,
        mut hooks: Vec<RuntimeHookSnapshot>,
    ) -> Result<Self, RuntimeTurnSnapshotError> {
        validate_tool_snapshots(&tools)?;
        validate_hook_snapshots(&hooks)?;
        hooks.sort_by_key(|hook| hook.display_order);
        Ok(Self { tools, hooks })
    }

    pub fn tools(&self) -> &[RuntimeToolSnapshot] {
        &self.tools
    }

    pub fn hooks(&self) -> &[RuntimeHookSnapshot] {
        &self.hooks
    }

    pub fn tool(&self, identity: &RuntimeToolIdentity) -> Option<&RuntimeToolSnapshot> {
        self.tools.iter().find(|tool| tool.identity == *identity)
    }

    pub fn model_visible_tools(&self) -> impl Iterator<Item = &RuntimeToolSnapshot> {
        self.tools.iter().filter(|tool| tool.model_visible)
    }

    pub fn deferred_tools(&self) -> impl Iterator<Item = &RuntimeToolSnapshot> {
        self.tools
            .iter()
            .filter(|tool| tool.exposure == RuntimeToolExposure::Deferred)
    }

    pub fn hooks_for(
        &self,
        event_name: RuntimeHookEventName,
    ) -> impl Iterator<Item = &RuntimeHookSnapshot> {
        self.hooks
            .iter()
            .filter(move |hook| hook.event_name == event_name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeTurnSnapshotError {
    EmptyToolName,
    EmptyToolNamespace,
    ToolDefinitionNameMismatch(RuntimeToolIdentity),
    DuplicateToolIdentity(RuntimeToolIdentity),
    HiddenToolCannotSupportParallel(RuntimeToolIdentity),
    EmptyHookKey,
    UnsupportedHookHandlerType {
        key: String,
        handler_type: RuntimeHookHandlerType,
    },
    UnsupportedHookExecutionMode {
        key: String,
        execution_mode: RuntimeHookExecutionMode,
    },
    DuplicateHookKey(String),
    DuplicateHookDisplayOrder(i64),
    RelativeHookSourcePath(PathBuf),
}

fn validate_tool_snapshots(tools: &[RuntimeToolSnapshot]) -> Result<(), RuntimeTurnSnapshotError> {
    let mut identities = HashSet::new();
    for tool in tools {
        if tool.identity.name.trim().is_empty() {
            return Err(RuntimeTurnSnapshotError::EmptyToolName);
        }
        if tool
            .identity
            .namespace
            .as_deref()
            .is_some_and(|namespace| namespace.trim().is_empty())
        {
            return Err(RuntimeTurnSnapshotError::EmptyToolNamespace);
        }
        if tool.identity.name != tool.definition.name {
            return Err(RuntimeTurnSnapshotError::ToolDefinitionNameMismatch(
                tool.identity.clone(),
            ));
        }
        if tool.exposure == RuntimeToolExposure::Hidden && tool.supports_parallel {
            return Err(RuntimeTurnSnapshotError::HiddenToolCannotSupportParallel(
                tool.identity.clone(),
            ));
        }
        if !identities.insert(tool.identity.clone()) {
            return Err(RuntimeTurnSnapshotError::DuplicateToolIdentity(
                tool.identity.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_hook_snapshots(hooks: &[RuntimeHookSnapshot]) -> Result<(), RuntimeTurnSnapshotError> {
    let mut keys = HashSet::new();
    let mut display_orders = HashSet::new();
    for hook in hooks {
        if hook.key.trim().is_empty() {
            return Err(RuntimeTurnSnapshotError::EmptyHookKey);
        }
        if hook.handler_type != RuntimeHookHandlerType::Command {
            return Err(RuntimeTurnSnapshotError::UnsupportedHookHandlerType {
                key: hook.key.clone(),
                handler_type: hook.handler_type,
            });
        }
        if hook.execution_mode != RuntimeHookExecutionMode::Sync {
            return Err(RuntimeTurnSnapshotError::UnsupportedHookExecutionMode {
                key: hook.key.clone(),
                execution_mode: hook.execution_mode,
            });
        }
        if !hook.source_path.is_absolute() {
            return Err(RuntimeTurnSnapshotError::RelativeHookSourcePath(
                hook.source_path.clone(),
            ));
        }
        if !keys.insert(hook.key.clone()) {
            return Err(RuntimeTurnSnapshotError::DuplicateHookKey(hook.key.clone()));
        }
        if !display_orders.insert(hook.display_order) {
            return Err(RuntimeTurnSnapshotError::DuplicateHookDisplayOrder(
                hook.display_order,
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(
        identity: RuntimeToolIdentity,
        exposure: RuntimeToolExposure,
        supports_parallel: bool,
        model_visible: bool,
    ) -> RuntimeToolSnapshot {
        RuntimeToolSnapshot::new(
            identity.clone(),
            RuntimeToolDefinition::new(
                &identity.name,
                format!("{} description", identity.name),
                json!({ "type": "object" }),
            ),
            exposure,
            supports_parallel,
            model_visible,
        )
    }

    fn hook(
        key: &str,
        event_name: RuntimeHookEventName,
        display_order: i64,
    ) -> RuntimeHookSnapshot {
        RuntimeHookSnapshot {
            key: key.to_string(),
            event_name,
            handler_type: RuntimeHookHandlerType::Command,
            execution_mode: RuntimeHookExecutionMode::Sync,
            matcher: Some("*".to_string()),
            command: Some("true".to_string()),
            timeout_sec: 30,
            status_message: None,
            additional_context_limit: None,
            source_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".codex/hooks.json"),
            source: RuntimeHookSource::Project,
            plugin_id: None,
            display_order,
            enabled: true,
            is_managed: false,
            current_hash: "sha256:test".to_string(),
            trust_status: RuntimeHookTrustStatus::Trusted,
        }
    }

    #[test]
    fn turn_snapshot_separates_direct_deferred_and_hidden_tools() {
        let snapshot = RuntimeTurnSnapshot::try_new(
            vec![
                tool(
                    RuntimeToolIdentity::plain("direct"),
                    RuntimeToolExposure::Direct,
                    false,
                    true,
                ),
                tool(
                    RuntimeToolIdentity::plain("model_only"),
                    RuntimeToolExposure::DirectModelOnly,
                    false,
                    true,
                ),
                tool(
                    RuntimeToolIdentity::plain("deferred"),
                    RuntimeToolExposure::Deferred,
                    false,
                    false,
                ),
                tool(
                    RuntimeToolIdentity::plain("hidden"),
                    RuntimeToolExposure::Hidden,
                    false,
                    false,
                ),
            ],
            Vec::new(),
        )
        .expect("valid turn snapshot");

        assert_eq!(
            snapshot
                .model_visible_tools()
                .map(|tool| tool.definition.name.as_str())
                .collect::<Vec<_>>(),
            vec!["direct", "model_only"]
        );
        assert_eq!(
            snapshot
                .deferred_tools()
                .map(|tool| tool.definition.name.as_str())
                .collect::<Vec<_>>(),
            vec!["deferred"]
        );
        assert!(snapshot
            .tool(&RuntimeToolIdentity::plain("hidden"))
            .is_some());
        assert!(snapshot
            .tool(&RuntimeToolIdentity::plain("Hidden"))
            .is_none());
    }

    #[test]
    fn turn_snapshot_keeps_case_sensitive_namespaced_tool_identities() {
        let snapshot = RuntimeTurnSnapshot::try_new(
            vec![
                tool(
                    RuntimeToolIdentity::plain("Read"),
                    RuntimeToolExposure::Direct,
                    false,
                    true,
                ),
                tool(
                    RuntimeToolIdentity::plain("read"),
                    RuntimeToolExposure::Deferred,
                    false,
                    false,
                ),
                tool(
                    RuntimeToolIdentity::namespaced("mcp__docs__", "Read"),
                    RuntimeToolExposure::Direct,
                    false,
                    true,
                ),
            ],
            Vec::new(),
        )
        .expect("Codex tool identities are case-sensitive and namespaced");

        assert_eq!(snapshot.tools().len(), 3);
    }

    #[test]
    fn turn_snapshot_uses_explicit_step_visibility_and_orders_hooks_globally() {
        let snapshot = RuntimeTurnSnapshot::try_new(
            vec![
                tool(
                    RuntimeToolIdentity::plain("nested"),
                    RuntimeToolExposure::Direct,
                    false,
                    false,
                ),
                tool(
                    RuntimeToolIdentity::plain("model_only"),
                    RuntimeToolExposure::DirectModelOnly,
                    false,
                    true,
                ),
            ],
            vec![
                hook(
                    "project:pre_tool_use:0:1",
                    RuntimeHookEventName::PreToolUse,
                    20,
                ),
                hook(
                    "project:permission_request:0:0",
                    RuntimeHookEventName::PermissionRequest,
                    10,
                ),
                hook(
                    "project:pre_tool_use:0:0",
                    RuntimeHookEventName::PreToolUse,
                    5,
                ),
            ],
        )
        .expect("valid hook snapshot");

        assert_eq!(
            snapshot
                .model_visible_tools()
                .map(|tool| tool.identity.name.as_str())
                .collect::<Vec<_>>(),
            vec!["model_only"]
        );
        assert_eq!(
            snapshot
                .hooks_for(RuntimeHookEventName::PreToolUse)
                .map(|hook| hook.key.as_str())
                .collect::<Vec<_>>(),
            vec!["project:pre_tool_use:0:0", "project:pre_tool_use:0:1"]
        );
        assert_eq!(snapshot.hooks()[0].scope(), RuntimeHookScope::Turn);
    }

    #[test]
    fn turn_snapshot_rejects_invalid_tool_and_hook_contracts() {
        let duplicate_identity = RuntimeToolIdentity::namespaced("mcp__docs__", "read");
        let error = RuntimeTurnSnapshot::try_new(
            vec![
                tool(
                    duplicate_identity.clone(),
                    RuntimeToolExposure::Direct,
                    false,
                    true,
                ),
                tool(
                    duplicate_identity.clone(),
                    RuntimeToolExposure::Deferred,
                    false,
                    false,
                ),
            ],
            Vec::new(),
        )
        .expect_err("exact namespaced identities must be unique");

        assert_eq!(
            error,
            RuntimeTurnSnapshotError::DuplicateToolIdentity(duplicate_identity)
        );

        let error = RuntimeTurnSnapshot::try_new(
            vec![tool(
                RuntimeToolIdentity::plain("hidden"),
                RuntimeToolExposure::Hidden,
                true,
                false,
            )],
            Vec::new(),
        )
        .expect_err("hidden tools must not advertise parallel execution");

        assert_eq!(
            error,
            RuntimeTurnSnapshotError::HiddenToolCannotSupportParallel(RuntimeToolIdentity::plain(
                "hidden"
            ))
        );

        let error = RuntimeTurnSnapshot::try_new(
            Vec::new(),
            vec![
                hook("project:stop:0:0", RuntimeHookEventName::Stop, 5),
                hook("project:stop:0:0", RuntimeHookEventName::Stop, 6),
            ],
        )
        .expect_err("duplicate hook keys must fail closed");

        assert_eq!(
            error,
            RuntimeTurnSnapshotError::DuplicateHookKey("project:stop:0:0".to_string())
        );
    }

    #[test]
    fn turn_snapshot_rejects_unexecutable_hook_kinds() {
        for handler_type in [
            RuntimeHookHandlerType::Prompt,
            RuntimeHookHandlerType::Agent,
        ] {
            let mut unsupported = hook(
                "project:user_prompt_submit:0:0",
                RuntimeHookEventName::UserPromptSubmit,
                1,
            );
            unsupported.handler_type = handler_type;

            assert_eq!(
                RuntimeTurnSnapshot::try_new(Vec::new(), vec![unsupported])
                    .expect_err("unsupported hook handler must fail closed"),
                RuntimeTurnSnapshotError::UnsupportedHookHandlerType {
                    key: "project:user_prompt_submit:0:0".to_string(),
                    handler_type,
                }
            );
        }

        let mut asynchronous = hook(
            "project:session_end:0:0",
            RuntimeHookEventName::SessionEnd,
            1,
        );
        asynchronous.execution_mode = RuntimeHookExecutionMode::Async;
        assert_eq!(
            RuntimeTurnSnapshot::try_new(Vec::new(), vec![asynchronous])
                .expect_err("async hook snapshot must fail closed"),
            RuntimeTurnSnapshotError::UnsupportedHookExecutionMode {
                key: "project:session_end:0:0".to_string(),
                execution_mode: RuntimeHookExecutionMode::Async,
            }
        );
    }

    #[test]
    fn turn_snapshot_round_trips_tool_and_hook_contracts() {
        let snapshot = RuntimeTurnSnapshot::try_new(
            vec![tool(
                RuntimeToolIdentity::plain("parallel"),
                RuntimeToolExposure::Direct,
                true,
                true,
            )],
            vec![hook("project:stop:0:0", RuntimeHookEventName::Stop, 10)],
        )
        .expect("valid turn snapshot");

        let encoded = serde_json::to_value(&snapshot).expect("serialize snapshot");
        assert_eq!(encoded["tools"][0]["supports_parallel"], json!(true));
        assert_eq!(encoded["hooks"][0]["event_name"], json!("stop"));

        let decoded: RuntimeTurnSnapshot =
            serde_json::from_value(encoded).expect("deserialize snapshot");
        assert_eq!(decoded, snapshot);

        let invalid = json!({
            "tools": [{
                "identity": { "namespace": null, "name": "hidden" },
                "definition": {
                    "name": "hidden",
                    "description": "hidden description",
                    "input_schema": { "type": "object" }
                },
                "exposure": "hidden",
                "supports_parallel": true,
                "model_visible": false
            }],
            "hooks": []
        });
        serde_json::from_value::<RuntimeTurnSnapshot>(invalid)
            .expect_err("deserialization must preserve snapshot validation");
    }
}
