use crate::{AppDataSource, ExecutionRequest};
use agent_runtime::reply_input::RuntimeReplyInputPart;
use lime_agent::AgentSessionConfig;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::Arc;

const APP_MENTION_PREFIX: &str = "app://";
const PLUGIN_MENTION_PREFIX: &str = "plugin://";
const PLUGIN_ACTIVATION_SCHEMA_VERSION: &str = "plugin-activation/v2";
const PLUGIN_RUNTIME_CAPABILITY_SCHEMA_VERSION: &str = "plugin-runtime-capabilities/v2";
const LEGACY_APP_RUNTIME_CAPABILITY_SCHEMA_VERSION: &str = "plugin-runtime-capabilities/v0.1";
const MENTION_SELECTION_TURN_METADATA_KEY: &str = "mention_selection";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct MentionSelection {
    apps: Vec<ResolvedAppMention>,
    plugins: Vec<ResolvedPluginMention>,
    plugin_filter_requested: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedAppMention {
    id: String,
    plugin_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedPluginMention {
    id: String,
    runtime_capabilities: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PluginActivationRecord {
    id: String,
    runtime_capabilities: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InstalledAppRecord {
    plugin_id: String,
    runtime_capabilities: Value,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct MentionRequests {
    app_ids: Vec<String>,
    plugin_ids: Vec<String>,
}

pub(super) async fn resolve_mentions(
    request: &ExecutionRequest,
    app_data_source: Option<Arc<dyn AppDataSource>>,
    plugin_activations: &[Value],
) -> MentionSelection {
    let requests = mention_requests(request);
    if requests.app_ids.is_empty() && requests.plugin_ids.is_empty() {
        return MentionSelection::default();
    }

    let installed_app_states = if requests.app_ids.is_empty() {
        Vec::new()
    } else if let Some(app_data_source) = app_data_source {
        match app_data_source.list_plugin_installed().await {
            Ok(installed) => installed.states,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "App installed registry 不可用，app:// Mention 按 fail-closed 处理"
                );
                Vec::new()
            }
        }
    } else {
        tracing::warn!(
            app_count = requests.app_ids.len(),
            "忽略无法由 app installed registry 校验的 app:// Mention"
        );
        Vec::new()
    };

    if !requests.plugin_ids.is_empty() && plugin_activations.is_empty() {
        tracing::warn!(
            plugin_count = requests.plugin_ids.len(),
            "Plugin v2 activation snapshot 为空，plugin:// Mention 按 fail-closed 处理"
        );
    }

    resolve_from_sources(requests, plugin_activations, &installed_app_states)
}

impl MentionSelection {
    pub(super) fn plugin_activations_for_turn(&self, activations: &[Value]) -> Vec<Value> {
        if !self.plugin_filter_requested {
            return activations.to_vec();
        }
        let selected_ids = self
            .plugins
            .iter()
            .map(|plugin| plugin.id.as_str())
            .collect::<HashSet<_>>();
        activations
            .iter()
            .filter(|activation| {
                string_at(activation, "/pluginId")
                    .is_some_and(|id| selected_ids.contains(id.as_str()))
            })
            .cloned()
            .collect()
    }

    pub(super) fn apply_to_session_config(&self, session_config: &mut AgentSessionConfig) {
        if self.apps.is_empty() && self.plugins.is_empty() {
            return;
        }

        let turn_context = session_config
            .turn_context
            .get_or_insert_with(Default::default);
        turn_context.metadata.insert(
            MENTION_SELECTION_TURN_METADATA_KEY.to_string(),
            json!({
                "schemaVersion": 1,
                "apps": self.apps.iter().map(|app| json!({
                    "id": app.id,
                    "pluginId": app.plugin_id,
                })).collect::<Vec<_>>(),
                "plugins": self.plugins.iter().map(|plugin| json!({
                    "id": plugin.id,
                    "runtimeCapabilities": plugin.runtime_capabilities,
                })).collect::<Vec<_>>(),
            }),
        );
    }
}

fn mention_requests(request: &ExecutionRequest) -> MentionRequests {
    let mut requests = MentionRequests::default();
    let mut seen_apps = HashSet::new();
    let mut seen_plugins = HashSet::new();
    for part in &request.input.parts {
        let RuntimeReplyInputPart::Mention { path, .. } = part else {
            continue;
        };
        let path = path.trim();
        if let Some(id) = mention_id(path, APP_MENTION_PREFIX) {
            if seen_apps.insert(id.to_string()) {
                requests.app_ids.push(id.to_string());
            }
        } else if let Some(id) = mention_id(path, PLUGIN_MENTION_PREFIX) {
            if seen_plugins.insert(id.to_string()) {
                requests.plugin_ids.push(id.to_string());
            }
        }
    }
    requests
}

fn mention_id<'a>(path: &'a str, prefix: &str) -> Option<&'a str> {
    path.strip_prefix(prefix).filter(|id| !id.is_empty())
}

fn resolve_from_sources(
    requests: MentionRequests,
    plugin_activations: &[Value],
    installed_app_states: &[Value],
) -> MentionSelection {
    let plugin_records = plugin_activations
        .iter()
        .filter_map(plugin_activation_record)
        .collect::<Vec<_>>();
    let app_records = installed_app_states
        .iter()
        .filter_map(installed_app_record)
        .collect::<Vec<_>>();
    let plugins = requests
        .plugin_ids
        .iter()
        .filter_map(|id| resolve_plugin_mention(id, &plugin_records))
        .collect::<Vec<_>>();
    let apps = requests
        .app_ids
        .iter()
        .filter_map(|id| resolve_app_mention(id, &app_records))
        .collect::<Vec<_>>();

    let unresolved_plugin_count = requests.plugin_ids.len().saturating_sub(plugins.len());
    let unresolved_app_count = requests.app_ids.len().saturating_sub(apps.len());
    if unresolved_plugin_count > 0 || unresolved_app_count > 0 {
        tracing::warn!(
            unresolved_plugin_count,
            unresolved_app_count,
            "部分结构化 Mention 未通过 current registry 校验"
        );
    }

    MentionSelection {
        apps,
        plugins,
        plugin_filter_requested: !requests.plugin_ids.is_empty(),
    }
}

fn plugin_activation_record(activation: &Value) -> Option<PluginActivationRecord> {
    if string_at(activation, "/schemaVersion").as_deref() != Some(PLUGIN_ACTIVATION_SCHEMA_VERSION)
    {
        return None;
    }
    let id = string_at(activation, "/pluginId")?;
    let runtime_capabilities = activation.get("runtimeCapabilities")?.clone();
    if string_at(&runtime_capabilities, "/schemaVersion").as_deref()
        != Some(PLUGIN_RUNTIME_CAPABILITY_SCHEMA_VERSION)
        || string_at(&runtime_capabilities, "/pluginId").as_deref() != Some(id.as_str())
    {
        return None;
    }
    Some(PluginActivationRecord {
        id,
        runtime_capabilities,
    })
}

fn installed_app_record(state: &Value) -> Option<InstalledAppRecord> {
    if state.get("disabled").and_then(Value::as_bool) != Some(false) {
        return None;
    }
    let plugin_id = string_at(state, "/appId")?;
    let mut runtime_capabilities = state.pointer("/manifest/runtimeCapabilities")?.clone();
    if string_at(&runtime_capabilities, "/schemaVersion").as_deref()
        != Some(LEGACY_APP_RUNTIME_CAPABILITY_SCHEMA_VERSION)
        || string_at(&runtime_capabilities, "/pluginId").as_deref() != Some(plugin_id.as_str())
    {
        return None;
    }
    if let Some(package_source_uri) = string_at(state, "/identity/sourceUri") {
        runtime_capabilities
            .as_object_mut()?
            .entry("packageSourceUri".to_string())
            .or_insert(Value::String(package_source_uri));
    }
    Some(InstalledAppRecord {
        plugin_id,
        runtime_capabilities,
    })
}

fn resolve_plugin_mention(
    id: &str,
    records: &[PluginActivationRecord],
) -> Option<ResolvedPluginMention> {
    let mut matches = records.iter().filter(|record| record.id == id);
    let record = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    Some(ResolvedPluginMention {
        id: record.id.clone(),
        runtime_capabilities: record.runtime_capabilities.clone(),
    })
}

fn resolve_app_mention(id: &str, records: &[InstalledAppRecord]) -> Option<ResolvedAppMention> {
    let mut matches = records.iter().filter(|record| {
        record
            .runtime_capabilities
            .get("tools")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .any(|tool| {
                string_at(tool, "/key").as_deref() == Some(id)
                    && string_at(tool, "/provider")
                        .is_some_and(|provider| provider.starts_with("connector:"))
            })
    });
    let record = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    Some(ResolvedAppMention {
        id: id.to_string(),
        plugin_id: record.plugin_id.clone(),
    })
}

fn string_at(value: &Value, pointer: &str) -> Option<String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_backend::tests::request_for_test;
    use agent_runtime::reply_input::{RuntimeReplyInput, RuntimeReplyInputPart};

    #[test]
    fn mention_requests_preserve_order_and_deduplicate_supported_paths() {
        let mut request = request_for_test("hello", None, None);
        request.input = RuntimeReplyInput::from_parts(vec![
            mention("Docs", "app://docs"),
            mention("Creator", "plugin://creator"),
            mention("Docs again", "app://docs"),
            mention("Empty", "plugin://"),
            mention("Skill", "skill://writer"),
        ]);

        assert_eq!(
            mention_requests(&request),
            MentionRequests {
                app_ids: vec!["docs".to_string()],
                plugin_ids: vec!["creator".to_string()],
            }
        );
    }

    #[test]
    fn plugin_mentions_only_resolve_from_v2_activation_snapshot() {
        let requests = MentionRequests {
            app_ids: vec![
                "docs".to_string(),
                "shared".to_string(),
                "missing".to_string(),
            ],
            plugin_ids: vec![
                "creator".to_string(),
                "legacy-only".to_string(),
                "drifted".to_string(),
                "unmarked".to_string(),
                "duplicate".to_string(),
            ],
        };
        let mut unmarked = plugin_activation("unmarked", "unmarked");
        unmarked["runtimeCapabilities"]
            .as_object_mut()
            .expect("runtime capabilities object")
            .remove("schemaVersion");
        let activations = vec![
            plugin_activation("creator", "creator"),
            plugin_activation("drifted", "different-id"),
            unmarked,
            plugin_activation("duplicate", "duplicate"),
            plugin_activation("duplicate", "duplicate"),
        ];
        let states = vec![
            installed_state("creator", false, "creator", &["docs", "shared"]),
            installed_state("other", false, "other", &["shared"]),
            installed_state("legacy-only", false, "legacy-only", &["legacy-app"]),
        ];

        let selection = resolve_from_sources(requests, &activations, &states);
        let selected_activations = selection.plugin_activations_for_turn(&activations);

        assert_eq!(
            selection.apps,
            vec![ResolvedAppMention {
                id: "docs".to_string(),
                plugin_id: "creator".to_string(),
            }]
        );
        assert_eq!(selection.plugins.len(), 1);
        assert_eq!(selection.plugins[0].id, "creator");
        assert_eq!(selected_activations.len(), 1);
        assert_eq!(selected_activations[0]["pluginId"], "creator");
        assert_eq!(
            selection.plugins[0].runtime_capabilities["packageSourceUri"],
            "/plugins/creator"
        );
    }

    #[test]
    fn unresolved_plugin_mention_does_not_fall_back_to_all_activations() {
        let activations = vec![plugin_activation("creator", "creator")];
        let selection = resolve_from_sources(
            MentionRequests {
                app_ids: Vec::new(),
                plugin_ids: vec!["missing".to_string()],
            },
            &activations,
            &[],
        );

        assert!(selection.plugins.is_empty());
        assert!(selection
            .plugin_activations_for_turn(&activations)
            .is_empty());
        assert_eq!(
            MentionSelection::default().plugin_activations_for_turn(&activations),
            activations
        );
    }

    #[test]
    fn resolved_selection_adds_internal_context_without_raw_mentions() {
        let selection = resolve_from_sources(
            MentionRequests {
                app_ids: vec!["docs".to_string()],
                plugin_ids: vec!["creator".to_string()],
            },
            &[plugin_activation("creator", "creator")],
            &[installed_state("creator", false, "creator", &["docs"])],
        );
        let mut session_config = lime_agent::AgentSessionConfig {
            id: "session-1".to_string(),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            forked_from_thread_id: None,
            schedule_id: None,
            max_turns: None,
            provider_token_budget: None,
            system_prompt: Some("base".to_string()),
            system_prompt_override: Some(true),
            include_context_trace: Some(true),
            turn_context: None,
        };

        selection.apply_to_session_config(&mut session_config);
        selection.apply_to_session_config(&mut session_config);

        let turn_context = session_config.turn_context.as_ref().expect("turn context");
        let metadata = &turn_context.metadata[MENTION_SELECTION_TURN_METADATA_KEY];
        assert_eq!(metadata["apps"][0]["id"], "docs");
        assert_eq!(metadata["plugins"][0]["id"], "creator");
        let prompt = session_config.system_prompt.expect("system prompt");
        assert_eq!(prompt, "base");
        assert!(!prompt.contains("plugin://creator"));
        assert!(!prompt.contains("app://docs"));
    }

    fn mention(name: &str, path: &str) -> RuntimeReplyInputPart {
        RuntimeReplyInputPart::Mention {
            name: name.to_string(),
            path: path.to_string(),
        }
    }

    fn installed_state(
        app_id: &str,
        disabled: bool,
        capability_plugin_id: &str,
        connectors: &[&str],
    ) -> Value {
        json!({
            "appId": app_id,
            "disabled": disabled,
            "identity": {
                "sourceUri": format!("/plugins/{app_id}"),
            },
            "manifest": {
                "runtimeCapabilities": {
                    "schemaVersion": LEGACY_APP_RUNTIME_CAPABILITY_SCHEMA_VERSION,
                    "pluginId": capability_plugin_id,
                    "skills": [],
                    "tools": connectors.iter().map(|id| json!({
                        "key": id,
                        "provider": "connector:api",
                    })).collect::<Vec<_>>(),
                    "mcpBindings": [],
                    "workflowBindings": [],
                }
            }
        })
    }

    fn plugin_activation(plugin_id: &str, capability_plugin_id: &str) -> Value {
        json!({
            "schemaVersion": PLUGIN_ACTIVATION_SCHEMA_VERSION,
            "pluginId": plugin_id,
            "version": "1.0.0",
            "contentDigest": format!("sha256:{plugin_id}"),
            "marketplaceId": "test-marketplace",
            "packageSourceUri": format!("/plugins/{plugin_id}"),
            "runtimeCapabilities": {
                "schemaVersion": PLUGIN_RUNTIME_CAPABILITY_SCHEMA_VERSION,
                "pluginId": capability_plugin_id,
                "version": "1.0.0",
                "packageSourceUri": format!("/plugins/{plugin_id}"),
                "skills": [],
                "mcpBindings": [],
            }
        })
    }
}
