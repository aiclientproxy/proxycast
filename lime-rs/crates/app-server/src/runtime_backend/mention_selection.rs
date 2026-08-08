use crate::runtime::PluginTurnSnapshot;
use crate::{AppDataSource, ExecutionRequest};
use agent_runtime::reply_input::RuntimeReplyInputPart;
use lime_agent::AgentSessionConfig;
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;

const APP_MENTION_PREFIX: &str = "app://";
const PLUGIN_MENTION_PREFIX: &str = "plugin://";
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedPluginMention {
    config_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InstalledAppRecord {
    id: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct MentionRequests {
    app_ids: Vec<String>,
    plugin_ids: Vec<String>,
}

pub(super) async fn resolve_mentions(
    request: &ExecutionRequest,
    app_data_source: Option<Arc<dyn AppDataSource>>,
    plugin_snapshots: &[PluginTurnSnapshot],
) -> MentionSelection {
    let requests = mention_requests(request);
    if requests.app_ids.is_empty() && requests.plugin_ids.is_empty() {
        return MentionSelection::default();
    }

    let installed_app_states = if requests.app_ids.is_empty() {
        Vec::new()
    } else if let Some(app_data_source) = app_data_source {
        match app_data_source
            .list_installed_apps(Default::default())
            .await
        {
            Ok(installed) => installed.apps,
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

    if !requests.plugin_ids.is_empty() && plugin_snapshots.is_empty() {
        tracing::warn!(
            plugin_count = requests.plugin_ids.len(),
            "Plugin snapshot 为空，plugin:// Mention 按 fail-closed 处理"
        );
    }

    resolve_from_sources(requests, plugin_snapshots, &installed_app_states)
}

impl MentionSelection {
    pub(super) fn plugin_snapshots_for_turn(
        &self,
        snapshots: &[PluginTurnSnapshot],
    ) -> Vec<PluginTurnSnapshot> {
        if !self.plugin_filter_requested {
            return snapshots.to_vec();
        }
        let selected_ids = self
            .plugins
            .iter()
            .map(|plugin| plugin.config_name.as_str())
            .collect::<HashSet<_>>();
        snapshots
            .iter()
            .filter(|snapshot| selected_ids.contains(snapshot.config_name.as_str()))
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
                "apps": self.apps.iter().map(|app| json!({"id": app.id})).collect::<Vec<_>>(),
                "plugins": self.plugins.iter().map(|plugin| json!({"configName": plugin.config_name})).collect::<Vec<_>>(),
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
    plugin_snapshots: &[PluginTurnSnapshot],
    installed_apps: &[app_server_protocol::protocol::v2::InstalledApp],
) -> MentionSelection {
    let app_records = installed_apps
        .iter()
        .filter_map(installed_app_record)
        .collect::<Vec<_>>();
    let plugins = requests
        .plugin_ids
        .iter()
        .filter_map(|id| resolve_plugin_mention(id, plugin_snapshots))
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

fn installed_app_record(
    app: &app_server_protocol::protocol::v2::InstalledApp,
) -> Option<InstalledAppRecord> {
    if !app.enabled {
        return None;
    }
    Some(InstalledAppRecord { id: app.id.clone() })
}

fn resolve_plugin_mention(
    id: &str,
    records: &[PluginTurnSnapshot],
) -> Option<ResolvedPluginMention> {
    let mut matches = records.iter().filter(|record| record.config_name == id);
    let record = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    Some(ResolvedPluginMention {
        config_name: record.config_name.clone(),
    })
}

fn resolve_app_mention(id: &str, records: &[InstalledAppRecord]) -> Option<ResolvedAppMention> {
    let mut matches = records.iter().filter(|record| record.id == id);
    let _record = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    Some(ResolvedAppMention { id: id.to_string() })
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
            mention("Creator", "plugin://creator@test-marketplace"),
            mention("Docs again", "app://docs"),
            mention("Empty", "plugin://"),
            mention("Skill", "skill://writer"),
        ]);

        assert_eq!(
            mention_requests(&request),
            MentionRequests {
                app_ids: vec!["docs".to_string()],
                plugin_ids: vec!["creator@test-marketplace".to_string()],
            }
        );
    }

    #[test]
    fn plugin_mentions_only_resolve_from_current_snapshot() {
        let requests = MentionRequests {
            app_ids: vec![
                "docs".to_string(),
                "shared".to_string(),
                "missing".to_string(),
            ],
            plugin_ids: vec![
                "creator@test-marketplace".to_string(),
                "duplicate@test-marketplace".to_string(),
            ],
        };
        let snapshots = vec![
            plugin_snapshot("creator"),
            plugin_snapshot("duplicate"),
            plugin_snapshot("duplicate"),
        ];
        let states = vec![
            installed_app("docs", "creator", true),
            installed_app("shared", "other", true),
            installed_app("legacy-app", "legacy-only", true),
        ];

        let selection = resolve_from_sources(requests, &snapshots, &states);
        let selected_snapshots = selection.plugin_snapshots_for_turn(&snapshots);

        assert_eq!(
            selection.apps,
            vec![
                ResolvedAppMention {
                    id: "docs".to_string()
                },
                ResolvedAppMention {
                    id: "shared".to_string()
                },
            ]
        );
        assert_eq!(selection.plugins.len(), 1);
        assert_eq!(selection.plugins[0].config_name, "creator@test-marketplace");
        assert_eq!(selected_snapshots.len(), 1);
        assert_eq!(selected_snapshots[0].id, "creator");
    }

    #[test]
    fn unresolved_plugin_mention_does_not_fall_back_to_all_snapshots() {
        let snapshots = vec![plugin_snapshot("creator")];
        let selection = resolve_from_sources(
            MentionRequests {
                app_ids: Vec::new(),
                plugin_ids: vec!["missing@test-marketplace".to_string()],
            },
            &snapshots,
            &[],
        );

        assert!(selection.plugins.is_empty());
        assert!(selection.plugin_snapshots_for_turn(&snapshots).is_empty());
        assert_eq!(
            MentionSelection::default().plugin_snapshots_for_turn(&snapshots),
            snapshots
        );
    }

    #[test]
    fn resolved_selection_adds_internal_context_without_raw_mentions() {
        let selection = resolve_from_sources(
            MentionRequests {
                app_ids: vec!["docs".to_string()],
                plugin_ids: vec!["creator@test-marketplace".to_string()],
            },
            &[plugin_snapshot("creator")],
            &[installed_app("docs", "creator", true)],
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
        assert_eq!(
            metadata["plugins"][0]["configName"],
            "creator@test-marketplace"
        );
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

    fn installed_app(
        app_id: &str,
        _plugin_id: &str,
        enabled: bool,
    ) -> app_server_protocol::protocol::v2::InstalledApp {
        app_server_protocol::protocol::v2::InstalledApp {
            id: app_id.to_string(),
            runtime_name: Some(app_id.to_string()),
            enabled,
            callable: true,
        }
    }

    fn plugin_snapshot(plugin_id: &str) -> PluginTurnSnapshot {
        PluginTurnSnapshot {
            id: plugin_id.to_string(),
            config_name: format!("{plugin_id}@test-marketplace"),
            display_name: plugin_id.to_string(),
            package_root: format!("/plugins/{plugin_id}").into(),
            skill_names: Vec::new(),
            mcp_server_names: Vec::new(),
        }
    }
}
