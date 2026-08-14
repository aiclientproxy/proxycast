use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use url::Url;

use crate::{
    AgentSkillAuthority, AgentSkillDependencies, AgentSkillInterface, AgentSkillMetadata,
    AgentSkillPolicy, AgentSkillScope, AgentSkillSource,
};

pub const APPS_MCP_SERVER_NAME: &str = "codex_apps";
const ORCHESTRATOR_SKILL_MIME_TYPE: &str = "mcp/skill";
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(10);
const READ_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_RESOURCE_PAGES: usize = 10;
const MAX_VISIBLE_SKILLS: usize = 100;
const MAX_HIDDEN_SKILLS: usize = 1_000;
const MAX_SKILL_NAME_CHARS: usize = 64;
const MAX_QUALIFIED_SKILL_NAME_CHARS: usize = 128;
const MAX_PACKAGE_URI_CHARS: usize = 1_024;
const MAX_RESOURCE_URI_CHARS: usize = 2_048;
const MAX_RESOURCE_CONTENT_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct OrchestratorSkillResource {
    pub uri: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
    pub meta: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrchestratorSkillResourcePage {
    pub resources: Vec<OrchestratorSkillResource>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrchestratorSkillResourceContent {
    Text { uri: String, text: String },
    Blob { uri: String },
}

#[async_trait]
pub trait OrchestratorSkillResourceGateway: Send + Sync {
    async fn list_resource_page(
        &self,
        server: &str,
        cursor: Option<String>,
    ) -> Result<OrchestratorSkillResourcePage, String>;

    async fn read_resource(
        &self,
        server: &str,
        uri: &str,
    ) -> Result<Vec<OrchestratorSkillResourceContent>, String>;
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct OrchestratorSkillDiscovery {
    pub skills: Vec<AgentSkillMetadata>,
    pub warnings: Vec<String>,
}

pub async fn discover_orchestrator_skills(
    gateway: &dyn OrchestratorSkillResourceGateway,
) -> Result<OrchestratorSkillDiscovery, String> {
    let deadline = tokio::time::Instant::now() + DISCOVERY_TIMEOUT;
    let mut discovery = OrchestratorSkillDiscovery::default();
    let mut cursor = None;
    let mut seen_cursors = HashSet::new();
    let mut visible = 0usize;
    let mut hidden = 0usize;
    let mut skipped = 0usize;
    let mut truncated = false;
    let mut completed_pages = 0usize;

    for _ in 0..MAX_RESOURCE_PAGES {
        let page = tokio::time::timeout_at(
            deadline,
            gateway.list_resource_page(APPS_MCP_SERVER_NAME, cursor.clone()),
        )
        .await
        .map_err(|_| format!("Orchestrator Skill discovery 在 {DISCOVERY_TIMEOUT:?} 后超时"))?;
        let page = match page {
            Ok(page) => page,
            Err(error) if completed_pages == 0 => return Err(error),
            Err(error) => {
                discovery.warnings.push(format!(
                    "Orchestrator Skill discovery 在 {completed_pages} 页后停止: {error}"
                ));
                cursor = None;
                break;
            }
        };
        completed_pages += 1;

        for resource in page.resources {
            if resource.mime_type.as_deref() != Some(ORCHESTRATOR_SKILL_MIME_TYPE) {
                continue;
            }
            let Some(skill) = skill_from_resource(resource) else {
                skipped += 1;
                continue;
            };
            if skill.policy.allow_implicit_invocation {
                if visible >= MAX_VISIBLE_SKILLS {
                    truncated = true;
                    continue;
                }
                visible += 1;
            } else {
                if hidden >= MAX_HIDDEN_SKILLS {
                    truncated = true;
                    continue;
                }
                hidden += 1;
            }
            discovery.skills.push(skill);
        }

        let Some(next_cursor) = page.next_cursor else {
            cursor = None;
            break;
        };
        if !seen_cursors.insert(next_cursor.clone()) {
            discovery
                .warnings
                .push("Orchestrator Skill resource pagination 返回了重复 cursor".to_string());
            cursor = None;
            break;
        }
        cursor = Some(next_cursor);
    }

    if cursor.is_some() || truncated {
        discovery.warnings.push(format!(
            "Orchestrator Skill discovery 已按 {MAX_VISIBLE_SKILLS} 个可见 Skill、{MAX_HIDDEN_SKILLS} 个隐藏 Skill 或 {MAX_RESOURCE_PAGES} 页上限截断"
        ));
    }
    if skipped > 0 {
        discovery.warnings.push(format!(
            "跳过了 {skipped} 个格式无效的 Orchestrator Skill resource"
        ));
    }
    Ok(discovery)
}

pub async fn read_orchestrator_skill_resource(
    gateway: &dyn OrchestratorSkillResourceGateway,
    package_uri: &str,
    resource_uri: &str,
) -> Result<String, String> {
    if !resource_belongs_to_orchestrator_package(package_uri, resource_uri) {
        return Err("Orchestrator Skill resource 不属于声明的 package".to_string());
    }
    let contents = tokio::time::timeout(
        READ_TIMEOUT,
        gateway.read_resource(APPS_MCP_SERVER_NAME, resource_uri),
    )
    .await
    .map_err(|_| format!("Orchestrator Skill resource read 在 {READ_TIMEOUT:?} 后超时"))??;
    let text = contents.into_iter().find_map(|content| match content {
        OrchestratorSkillResourceContent::Text { uri, text } if uri == resource_uri => Some(text),
        OrchestratorSkillResourceContent::Text { .. }
        | OrchestratorSkillResourceContent::Blob { .. } => None,
    });
    let text = text.ok_or_else(|| {
        format!("Orchestrator Skill resource {resource_uri} 未返回 matching text content")
    })?;
    if text.len() > MAX_RESOURCE_CONTENT_BYTES {
        return Err(format!(
            "Orchestrator Skill resource {resource_uri} 超过 {MAX_RESOURCE_CONTENT_BYTES} byte 上限"
        ));
    }
    Ok(text)
}

fn skill_from_resource(resource: OrchestratorSkillResource) -> Option<AgentSkillMetadata> {
    let package_uri = validated_skill_uri(&resource.uri, MAX_PACKAGE_URI_CHARS)?;
    package_uri.strip_prefix("skill://")?.split_once('/')?;
    let meta = resource.meta.as_ref()?.as_object()?;
    let allow_implicit_invocation = meta
        .get("allow_implicit_invocation")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let skill_name = normalized_label(meta.get("skill_name")?.as_str()?, MAX_SKILL_NAME_CHARS)?;
    let user_owned = meta.get("source").and_then(Value::as_str) == Some("user");
    let name = if user_owned {
        skill_name
    } else {
        let plugin_name =
            normalized_label(meta.get("plugin_name")?.as_str()?, MAX_SKILL_NAME_CHARS)?;
        let qualified_name = format!("{plugin_name}:{skill_name}");
        (qualified_name.chars().count() <= MAX_QUALIFIED_SKILL_NAME_CHARS)
            .then_some(qualified_name)?
    };
    let description = normalized_description(resource.description.as_deref().unwrap_or_default())?;
    let resource_uri = format!("{}/SKILL.md", package_uri.trim_end_matches('/'));
    validated_skill_uri(&resource_uri, MAX_RESOURCE_URI_CHARS)?;

    Some(AgentSkillMetadata {
        skill_id: format!("orchestrator:{}", name.to_ascii_lowercase()),
        name: name.clone(),
        description,
        scope: AgentSkillScope::Orchestrator,
        source: AgentSkillSource::Orchestrator,
        authority: AgentSkillAuthority::Orchestrator,
        enabled: true,
        interface: AgentSkillInterface {
            display_name: name,
            execution_mode: "mcp_resource".to_string(),
            provider: Some(APPS_MCP_SERVER_NAME.to_string()),
            model: None,
            argument_hint: None,
        },
        dependencies: AgentSkillDependencies::default(),
        policy: AgentSkillPolicy {
            allow_implicit_invocation,
            when_to_use: None,
        },
        capabilities: Vec::new(),
        directory: PathBuf::from(package_uri),
        skill_file_path: PathBuf::from(resource_uri),
    })
}

fn validated_skill_uri(uri: &str, max_chars: usize) -> Option<&str> {
    validated_skill_url(uri, max_chars).map(|_| uri)
}

fn validated_skill_url(uri: &str, max_chars: usize) -> Option<Url> {
    if uri.chars().count() > max_chars
        || uri
            .chars()
            .any(|ch| ch.is_control() || ch.is_whitespace() || matches!(ch, '<' | '>'))
    {
        return None;
    }
    let url = Url::parse(uri).ok()?;
    let path_is_valid = url.path_segments().is_some_and(|segments| {
        let segments = segments.collect::<Vec<_>>();
        !segments.is_empty() && segments.iter().all(|segment| !segment.is_empty())
    });
    (url.scheme() == "skill"
        && url.as_str() == uri
        && url.host_str().is_some_and(|host| !host.is_empty())
        && url.username().is_empty()
        && url.password().is_none()
        && url.port().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && path_is_valid)
        .then_some(url)
}

pub fn resource_belongs_to_orchestrator_package(package: &str, resource: &str) -> bool {
    let Some(package) = validated_skill_url(package, MAX_PACKAGE_URI_CHARS) else {
        return false;
    };
    let Some(resource) = validated_skill_url(resource, MAX_RESOURCE_URI_CHARS) else {
        return false;
    };
    let Some(package_segments) = package.path_segments() else {
        return false;
    };
    let Some(resource_segments) = resource.path_segments() else {
        return false;
    };
    let package_segments = package_segments.collect::<Vec<_>>();
    let resource_segments = resource_segments.collect::<Vec<_>>();
    package.scheme() == resource.scheme()
        && package.host_str() == resource.host_str()
        && resource_segments.len() > package_segments.len()
        && resource_segments.starts_with(&package_segments)
}

fn normalized_label(value: &str, max_chars: usize) -> Option<String> {
    let value = normalized_single_line(value, max_chars)?;
    (!value.is_empty() && !value.chars().any(|ch| matches!(ch, '&' | '<' | '>'))).then_some(value)
}

fn normalized_description(value: &str) -> Option<String> {
    let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if value.chars().any(char::is_control) {
        return None;
    }
    Some(
        value
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;"),
    )
}

fn normalized_single_line(value: &str, max_chars: usize) -> Option<String> {
    let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
    (value.chars().count() <= max_chars && !value.chars().any(char::is_control)).then_some(value)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use tokio::sync::Mutex;

    use super::*;

    struct TestGateway {
        pages: Mutex<VecDeque<OrchestratorSkillResourcePage>>,
        contents: Vec<OrchestratorSkillResourceContent>,
    }

    #[async_trait]
    impl OrchestratorSkillResourceGateway for TestGateway {
        async fn list_resource_page(
            &self,
            server: &str,
            _cursor: Option<String>,
        ) -> Result<OrchestratorSkillResourcePage, String> {
            assert_eq!(server, APPS_MCP_SERVER_NAME);
            self.pages
                .lock()
                .await
                .pop_front()
                .ok_or_else(|| "no fixture page".to_string())
        }

        async fn read_resource(
            &self,
            server: &str,
            _uri: &str,
        ) -> Result<Vec<OrchestratorSkillResourceContent>, String> {
            assert_eq!(server, APPS_MCP_SERVER_NAME);
            Ok(self.contents.clone())
        }
    }

    fn skill_resource(uri: &str) -> OrchestratorSkillResource {
        OrchestratorSkillResource {
            uri: uri.to_string(),
            description: Some("Create release notes".to_string()),
            mime_type: Some(ORCHESTRATOR_SKILL_MIME_TYPE.to_string()),
            meta: Some(serde_json::json!({
                "skill_name": "release-notes",
                "plugin_name": "delivery",
                "source": "plugin",
                "allow_implicit_invocation": true,
                "skill_id": "skill-1"
            })),
        }
    }

    #[tokio::test]
    async fn discovery_builds_orchestrator_identity_and_resource_locator() {
        let gateway = TestGateway {
            pages: Mutex::new(VecDeque::from([OrchestratorSkillResourcePage {
                resources: vec![skill_resource("skill://delivery/release-notes")],
                next_cursor: None,
            }])),
            contents: Vec::new(),
        };

        let discovery = discover_orchestrator_skills(&gateway)
            .await
            .expect("discover skill");

        assert!(discovery.warnings.is_empty());
        assert_eq!(discovery.skills.len(), 1);
        let skill = &discovery.skills[0];
        assert_eq!(skill.name, "delivery:release-notes");
        assert_eq!(skill.scope, AgentSkillScope::Orchestrator);
        assert_eq!(skill.source, AgentSkillSource::Orchestrator);
        assert_eq!(skill.authority, AgentSkillAuthority::Orchestrator);
        assert_eq!(
            skill.skill_file_path.to_str(),
            Some("skill://delivery/release-notes/SKILL.md")
        );
    }

    #[tokio::test]
    async fn discovery_stops_on_duplicate_cursor() {
        let gateway = TestGateway {
            pages: Mutex::new(VecDeque::from([
                OrchestratorSkillResourcePage {
                    resources: Vec::new(),
                    next_cursor: Some("same".to_string()),
                },
                OrchestratorSkillResourcePage {
                    resources: Vec::new(),
                    next_cursor: Some("same".to_string()),
                },
            ])),
            contents: Vec::new(),
        };

        let discovery = discover_orchestrator_skills(&gateway)
            .await
            .expect("duplicate cursor returns partial catalog");

        assert!(discovery
            .warnings
            .iter()
            .any(|warning| warning.contains("重复 cursor")));
    }

    #[tokio::test]
    async fn read_requires_package_owned_matching_text() {
        let resource_uri = "skill://delivery/release-notes/SKILL.md";
        let gateway = TestGateway {
            pages: Mutex::new(VecDeque::new()),
            contents: vec![OrchestratorSkillResourceContent::Text {
                uri: resource_uri.to_string(),
                text: "# Release notes".to_string(),
            }],
        };

        let text = read_orchestrator_skill_resource(
            &gateway,
            "skill://delivery/release-notes",
            resource_uri,
        )
        .await
        .expect("read matching resource");
        assert_eq!(text, "# Release notes");

        let error =
            read_orchestrator_skill_resource(&gateway, "skill://delivery/other", resource_uri)
                .await
                .expect_err("cross-package read must fail");
        assert!(error.contains("不属于"));
    }
}
