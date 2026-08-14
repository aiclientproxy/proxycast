use async_trait::async_trait;
use lime_agent::AgentRuntimeState;
use lime_skills::{
    discover_orchestrator_skills, OrchestratorSkillDiscovery, OrchestratorSkillResource,
    OrchestratorSkillResourceContent, OrchestratorSkillResourceGateway,
    OrchestratorSkillResourcePage, APPS_MCP_SERVER_NAME,
};
use serde_json::Value;

pub(super) fn skills_enabled(config: Option<&Value>) -> bool {
    feature_enabled(config, "skills")
}

#[cfg(test)]
pub(super) fn mcp_enabled(config: Option<&Value>) -> bool {
    feature_enabled(config, "mcp")
}

fn feature_enabled(config: Option<&Value>, feature: &str) -> bool {
    let Some(config) = config else {
        return true;
    };
    if config.pointer("/orchestrator/loadError").is_some() {
        return false;
    }
    config
        .pointer(&format!("/orchestrator/{feature}/enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

pub(super) async fn discover_for_turn(
    state: &AgentRuntimeState,
    session_id: &str,
    thread_id: &str,
    config: Option<&Value>,
) -> OrchestratorSkillDiscovery {
    if !skills_enabled(config) {
        return OrchestratorSkillDiscovery::default();
    }
    let has_server = state
        .has_mcp_server(session_id, thread_id, APPS_MCP_SERVER_NAME)
        .await
        .unwrap_or(false);
    if !has_server {
        return OrchestratorSkillDiscovery::default();
    }
    let gateway = SessionResourceGateway {
        state: state.clone(),
        session_id: session_id.to_string(),
        thread_id: thread_id.to_string(),
    };
    match discover_orchestrator_skills(&gateway).await {
        Ok(discovery) => discovery,
        Err(error) => OrchestratorSkillDiscovery {
            skills: Vec::new(),
            warnings: vec![error],
        },
    }
}

struct SessionResourceGateway {
    state: AgentRuntimeState,
    session_id: String,
    thread_id: String,
}

#[async_trait]
impl OrchestratorSkillResourceGateway for SessionResourceGateway {
    async fn list_resource_page(
        &self,
        server: &str,
        cursor: Option<String>,
    ) -> Result<OrchestratorSkillResourcePage, String> {
        let page = self
            .state
            .list_mcp_resource_page(&self.session_id, &self.thread_id, server, cursor)
            .await?;
        Ok(OrchestratorSkillResourcePage {
            resources: page
                .resources
                .into_iter()
                .map(|resource| OrchestratorSkillResource {
                    uri: resource.uri,
                    description: resource.description,
                    mime_type: resource.mime_type,
                    meta: resource.meta,
                })
                .collect(),
            next_cursor: page.next_cursor,
        })
    }

    async fn read_resource(
        &self,
        server: &str,
        uri: &str,
    ) -> Result<Vec<OrchestratorSkillResourceContent>, String> {
        let content = self
            .state
            .read_mcp_resource(&self.session_id, &self.thread_id, server, uri)
            .await?;
        Ok(match (content.text, content.blob) {
            (Some(text), None) => vec![OrchestratorSkillResourceContent::Text {
                uri: content.uri,
                text,
            }],
            (None, Some(_)) => vec![OrchestratorSkillResourceContent::Blob { uri: content.uri }],
            (None, None) => Vec::new(),
            (Some(_), Some(_)) => {
                return Err("MCP resource response 同时包含 text 与 blob".to_string());
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn orchestrator_feature_defaults_enabled_and_load_error_fails_closed() {
        assert!(skills_enabled(None));
        assert!(mcp_enabled(None));
        let disabled = json!({"orchestrator": {"skills": {"enabled": false}}});
        assert!(!skills_enabled(Some(&disabled)));
        assert!(mcp_enabled(Some(&disabled)));
        let load_error = json!({"orchestrator": {"loadError": "invalid config"}});
        assert!(!skills_enabled(Some(&load_error)));
        assert!(!mcp_enabled(Some(&load_error)));
    }
}
