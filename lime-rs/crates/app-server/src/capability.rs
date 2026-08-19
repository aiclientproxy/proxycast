use app_server_protocol::CapabilityDescriptor;
use app_server_protocol::CapabilityListParams;
use app_server_protocol::RuntimeOptions;
use app_server_protocol::METHOD_THREAD_READ;
use app_server_protocol::METHOD_THREAD_START;
use app_server_protocol::METHOD_TURN_INTERRUPT;
use app_server_protocol::METHOD_TURN_START;
use lime_agent::agent_tools::catalog::{
    tool_catalog_entries_for_surface, ToolLifecycle, WorkspaceToolSurface,
};
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilityListContext {
    pub app_id: Option<String>,
    pub workspace_id: Option<String>,
    pub session_id: Option<String>,
}

impl From<CapabilityListParams> for CapabilityListContext {
    fn from(params: CapabilityListParams) -> Self {
        Self {
            app_id: params.app_id,
            workspace_id: params.workspace_id,
            session_id: params.session_id,
        }
    }
}

pub trait CapabilitySource: Send + Sync {
    fn list_capabilities(&self, context: &CapabilityListContext) -> Vec<CapabilityDescriptor>;

    fn prepare_turn_capabilities(
        &self,
        _context: &CapabilityListContext,
        _runtime_options: Option<&RuntimeOptions>,
    ) {
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityInventoryRecord {
    pub descriptor: CapabilityDescriptor,
    pub app_ids: Vec<String>,
    pub workspace_ids: Vec<String>,
    pub session_ids: Vec<String>,
}

impl CapabilityInventoryRecord {
    pub fn new(descriptor: CapabilityDescriptor) -> Self {
        Self {
            descriptor,
            app_ids: Vec::new(),
            workspace_ids: Vec::new(),
            session_ids: Vec::new(),
        }
    }

    pub fn agent_session() -> Self {
        Self::new(agent_session_descriptor())
    }

    pub fn executable_agent_turn(
        id: impl Into<String>,
        title: impl Into<String>,
        description: Option<String>,
    ) -> Self {
        Self::new(CapabilityDescriptor {
            id: id.into(),
            title: title.into(),
            description,
            methods: vec![METHOD_TURN_START.to_string()],
        })
    }

    pub fn for_apps(mut self, app_ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.app_ids = app_ids.into_iter().map(Into::into).collect();
        self
    }

    pub fn for_workspaces(
        mut self,
        workspace_ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.workspace_ids = workspace_ids.into_iter().map(Into::into).collect();
        self
    }

    pub fn for_sessions(
        mut self,
        session_ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.session_ids = session_ids.into_iter().map(Into::into).collect();
        self
    }

    fn is_visible_to(&self, context: &CapabilityListContext) -> bool {
        scope_matches(&self.app_ids, context.app_id.as_deref())
            && scope_matches(&self.workspace_ids, context.workspace_id.as_deref())
            && scope_matches(&self.session_ids, context.session_id.as_deref())
    }
}

pub fn capability_descriptor_allows_agent_turn_start(descriptor: &CapabilityDescriptor) -> bool {
    descriptor
        .methods
        .iter()
        .any(|method| method == METHOD_TURN_START)
}

#[derive(Debug, Clone)]
pub struct CapabilityInventorySource {
    records: Vec<CapabilityInventoryRecord>,
}

impl CapabilityInventorySource {
    pub fn new(records: Vec<CapabilityInventoryRecord>) -> Self {
        Self { records }
    }

    pub fn default_agent_session() -> Self {
        Self::new(vec![CapabilityInventoryRecord::agent_session()])
    }

    pub fn default_current_surface() -> Self {
        let mut records = vec![CapabilityInventoryRecord::agent_session()];
        for tool in tool_catalog_entries_for_surface(WorkspaceToolSurface::workbench())
            .into_iter()
            .filter(|tool| tool.lifecycle == ToolLifecycle::Current)
        {
            records.push(current_tool_capability_record(tool.name));
        }
        Self::new(records)
    }
}

impl Default for CapabilityInventorySource {
    fn default() -> Self {
        Self::default_current_surface()
    }
}

impl CapabilitySource for CapabilityInventorySource {
    fn list_capabilities(&self, context: &CapabilityListContext) -> Vec<CapabilityDescriptor> {
        let mut capabilities = Vec::new();
        for record in self
            .records
            .iter()
            .filter(|record| record.is_visible_to(context))
        {
            if let Some(index) = capabilities
                .iter()
                .position(|capability: &CapabilityDescriptor| capability.id == record.descriptor.id)
            {
                capabilities[index] = record.descriptor.clone();
            } else {
                capabilities.push(record.descriptor.clone());
            }
        }
        capabilities
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppPolicyManifest {
    #[serde(default)]
    pub capabilities: Vec<AppPolicyCapability>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppPolicyCapability {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub methods: Vec<String>,
    #[serde(default)]
    pub app_ids: Vec<String>,
    #[serde(default)]
    pub workspace_ids: Vec<String>,
    #[serde(default)]
    pub session_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AppPolicyManifestError {
    #[error("app policy capability at index {index} is missing id")]
    MissingId { index: usize },
    #[error("app policy capability at index {index} is missing title")]
    MissingTitle { index: usize },
    #[error("app policy capability at index {index} is missing methods")]
    MissingMethods { index: usize },
}

impl TryFrom<AppPolicyManifest> for CapabilityInventorySource {
    type Error = AppPolicyManifestError;

    fn try_from(manifest: AppPolicyManifest) -> Result<Self, Self::Error> {
        let mut records = Vec::with_capacity(manifest.capabilities.len());
        for (index, capability) in manifest.capabilities.into_iter().enumerate() {
            let id = capability.id.trim();
            if id.is_empty() {
                return Err(AppPolicyManifestError::MissingId { index });
            }

            let title = capability.title.trim();
            if title.is_empty() {
                return Err(AppPolicyManifestError::MissingTitle { index });
            }

            if capability.methods.is_empty() {
                return Err(AppPolicyManifestError::MissingMethods { index });
            }

            records.push(CapabilityInventoryRecord {
                descriptor: CapabilityDescriptor {
                    id: id.to_string(),
                    title: title.to_string(),
                    description: capability.description,
                    methods: capability.methods,
                },
                app_ids: capability.app_ids,
                workspace_ids: capability.workspace_ids,
                session_ids: capability.session_ids,
            });
        }

        Ok(CapabilityInventorySource::new(records))
    }
}

pub fn capability_source_from_app_policy_json(
    json: &str,
) -> Result<CapabilityInventorySource, AppPolicyLoadError> {
    let manifest: AppPolicyManifest = serde_json::from_str(json)?;
    Ok(CapabilityInventorySource::try_from(manifest)?)
}

#[derive(Debug, Error)]
pub enum AppPolicyLoadError {
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Manifest(#[from] AppPolicyManifestError),
}

fn agent_session_descriptor() -> CapabilityDescriptor {
    CapabilityDescriptor {
        id: "agent.session".to_string(),
        title: "Agent Session".to_string(),
        description: Some("Start, read, and cancel App Server agent sessions.".to_string()),
        methods: vec![
            METHOD_THREAD_START.to_string(),
            METHOD_THREAD_READ.to_string(),
            METHOD_TURN_START.to_string(),
            METHOD_TURN_INTERRUPT.to_string(),
        ],
    }
}

fn current_tool_capability_record(tool_name: &str) -> CapabilityInventoryRecord {
    CapabilityInventoryRecord::executable_agent_turn(
        format!("tool.{tool_name}"),
        tool_name.to_string(),
        Some("Current App Server tool catalog capability.".to_string()),
    )
}

fn scope_matches(allowed: &[String], requested: Option<&str>) -> bool {
    allowed.is_empty()
        || requested.is_some_and(|requested| allowed.iter().any(|value| value == requested))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executable_inventory_record_declares_turn_start_method() {
        let record = CapabilityInventoryRecord::executable_agent_turn(
            "session.draft.write",
            "Session Draft Write",
            Some("Session-scoped executable capability.".to_string()),
        )
        .for_apps(["content-studio"])
        .for_workspaces(["workspace-main"])
        .for_sessions(["sess_allowed"]);

        assert_eq!(record.descriptor.id, "session.draft.write");
        assert_eq!(record.descriptor.title, "Session Draft Write");
        assert_eq!(
            record.descriptor.description.as_deref(),
            Some("Session-scoped executable capability.")
        );
        assert_eq!(
            record.descriptor.methods,
            vec![METHOD_TURN_START.to_string()]
        );
        assert!(capability_descriptor_allows_agent_turn_start(
            &record.descriptor
        ));
        assert_eq!(record.app_ids, vec!["content-studio".to_string()]);
        assert_eq!(record.workspace_ids, vec!["workspace-main".to_string()]);
        assert_eq!(record.session_ids, vec!["sess_allowed".to_string()]);
    }

    #[test]
    fn default_inventory_source_exposes_current_tool_capabilities() {
        let source = CapabilityInventorySource::default();
        let capabilities = source.list_capabilities(&CapabilityListContext::default());
        let ids = capabilities
            .iter()
            .map(|capability| capability.id.as_str())
            .collect::<Vec<_>>();

        assert!(ids.contains(&"agent.session"));
        assert!(ids.contains(&"tool.WebFetch"));
        assert!(ids.contains(&"tool.WebSearch"));

        let web_search = capabilities
            .iter()
            .find(|capability| capability.id == "tool.WebSearch")
            .expect("WebSearch capability");
        assert_eq!(web_search.title, "WebSearch");
        assert_eq!(web_search.methods, vec![METHOD_TURN_START.to_string()]);
    }

    #[test]
    fn inventory_source_filters_by_app_and_workspace_scope() {
        let source = CapabilityInventorySource::new(vec![
            CapabilityInventoryRecord::new(CapabilityDescriptor {
                id: "global".to_string(),
                title: "Global".to_string(),
                description: None,
                methods: vec!["global/method".to_string()],
            }),
            CapabilityInventoryRecord::new(CapabilityDescriptor {
                id: "content".to_string(),
                title: "Content".to_string(),
                description: None,
                methods: vec!["content/method".to_string()],
            })
            .for_apps(["content-studio"])
            .for_workspaces(["workspace-main"]),
            CapabilityInventoryRecord::new(CapabilityDescriptor {
                id: "other".to_string(),
                title: "Other".to_string(),
                description: None,
                methods: vec!["other/method".to_string()],
            })
            .for_apps(["other-app"]),
        ]);

        let capabilities = source.list_capabilities(&CapabilityListContext {
            app_id: Some("content-studio".to_string()),
            workspace_id: Some("workspace-main".to_string()),
            session_id: None,
        });

        let ids: Vec<&str> = capabilities
            .iter()
            .map(|capability| capability.id.as_str())
            .collect();
        assert_eq!(ids, vec!["global", "content"]);
    }

    #[test]
    fn scoped_inventory_record_is_hidden_when_scope_is_missing() {
        let source = CapabilityInventorySource::new(vec![CapabilityInventoryRecord::new(
            CapabilityDescriptor {
                id: "content".to_string(),
                title: "Content".to_string(),
                description: None,
                methods: vec!["content/method".to_string()],
            },
        )
        .for_apps(["content-studio"])]);

        let capabilities = source.list_capabilities(&CapabilityListContext::default());

        assert!(capabilities.is_empty());
    }

    #[test]
    fn inventory_source_filters_by_session_scope() {
        let source = CapabilityInventorySource::new(vec![
            CapabilityInventoryRecord::new(CapabilityDescriptor {
                id: "session.allowed".to_string(),
                title: "Session Allowed".to_string(),
                description: None,
                methods: vec!["turn/start".to_string()],
            })
            .for_sessions(["sess_allowed"]),
            CapabilityInventoryRecord::new(CapabilityDescriptor {
                id: "workspace.global".to_string(),
                title: "Workspace Global".to_string(),
                description: None,
                methods: vec!["capability/list".to_string()],
            }),
        ]);

        let matched = source.list_capabilities(&CapabilityListContext {
            app_id: None,
            workspace_id: None,
            session_id: Some("sess_allowed".to_string()),
        });
        let missing = source.list_capabilities(&CapabilityListContext::default());
        let other = source.list_capabilities(&CapabilityListContext {
            app_id: None,
            workspace_id: None,
            session_id: Some("sess_other".to_string()),
        });

        let matched_ids: Vec<&str> = matched
            .iter()
            .map(|capability| capability.id.as_str())
            .collect();
        assert_eq!(matched_ids, vec!["session.allowed", "workspace.global"]);

        let missing_ids: Vec<&str> = missing
            .iter()
            .map(|capability| capability.id.as_str())
            .collect();
        assert_eq!(missing_ids, vec!["workspace.global"]);

        let other_ids: Vec<&str> = other
            .iter()
            .map(|capability| capability.id.as_str())
            .collect();
        assert_eq!(other_ids, vec!["workspace.global"]);
    }

    #[test]
    fn inventory_source_prefers_later_visible_descriptor_for_duplicate_id() {
        let source = CapabilityInventorySource::new(vec![
            CapabilityInventoryRecord::new(CapabilityDescriptor {
                id: "workspace_skill.report".to_string(),
                title: "Report Readiness".to_string(),
                description: None,
                methods: vec!["capability/list".to_string()],
            })
            .for_workspaces(["workspace-main"]),
            CapabilityInventoryRecord::executable_agent_turn(
                "workspace_skill.report",
                "Report Executable",
                None,
            )
            .for_workspaces(["workspace-main"])
            .for_sessions(["sess_allowed"]),
        ]);

        let workspace_only = source.list_capabilities(&CapabilityListContext {
            app_id: None,
            workspace_id: Some("workspace-main".to_string()),
            session_id: None,
        });
        assert_eq!(workspace_only.len(), 1);
        assert_eq!(workspace_only[0].title, "Report Readiness");
        assert_eq!(workspace_only[0].methods, vec!["capability/list"]);

        let session_scoped = source.list_capabilities(&CapabilityListContext {
            app_id: None,
            workspace_id: Some("workspace-main".to_string()),
            session_id: Some("sess_allowed".to_string()),
        });
        assert_eq!(session_scoped.len(), 1);
        assert_eq!(session_scoped[0].title, "Report Executable");
        assert_eq!(
            session_scoped[0].methods,
            vec![METHOD_TURN_START.to_string()]
        );
    }

    #[test]
    fn app_policy_manifest_builds_scoped_capability_source() {
        let source = capability_source_from_app_policy_json(
            r#"{
              "capabilities": [
                {
                  "id": "content.draft.generate",
                  "title": "Generate Draft",
                  "description": "Generate a content draft.",
                  "methods": ["turn/start"],
                  "appIds": ["content-studio"],
                  "workspaceIds": ["workspace-main"],
                  "sessionIds": ["sess_allowed"]
                }
              ]
            }"#,
        )
        .expect("policy source");

        let matched = source.list_capabilities(&CapabilityListContext {
            app_id: Some("content-studio".to_string()),
            workspace_id: Some("workspace-main".to_string()),
            session_id: Some("sess_allowed".to_string()),
        });
        let unmatched = source.list_capabilities(&CapabilityListContext {
            app_id: Some("content-studio".to_string()),
            workspace_id: Some("workspace-main".to_string()),
            session_id: Some("sess_other".to_string()),
        });

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].id, "content.draft.generate");
        assert_eq!(matched[0].title, "Generate Draft");
        assert_eq!(
            matched[0].description.as_deref(),
            Some("Generate a content draft.")
        );
        assert_eq!(matched[0].methods, vec![METHOD_TURN_START.to_string()]);
        assert!(unmatched.is_empty());
    }

    #[test]
    fn app_policy_manifest_rejects_incomplete_capabilities() {
        let error = capability_source_from_app_policy_json(
            r#"{
              "capabilities": [
                {
                  "id": "content.draft.generate",
                  "title": "Generate Draft",
                  "methods": []
                }
              ]
            }"#,
        )
        .expect_err("policy error");

        assert!(matches!(
            error,
            AppPolicyLoadError::Manifest(AppPolicyManifestError::MissingMethods { index: 0 })
        ));
    }
}
