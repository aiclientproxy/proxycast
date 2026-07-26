use crate::MultiAgentMode;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const WORLD_STATE_TURN_METADATA_KEY: &str = "world_state";
pub const WORLD_STATE_SOURCE: &str = "app_server_world_state";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeWorldState {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<RuntimeWorldEnvironment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permissions: Option<RuntimeWorldPermissions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collaboration: Option<RuntimeWorldMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multi_agent: Option<MultiAgentMode>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instruction_sections: Vec<RuntimeWorldInstructionSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeWorldEnvironment {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeWorldPermissions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approval_policy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sandbox_policy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeWorldMode {
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeWorldInstructionSection {
    pub id: String,
    pub body: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl RuntimeWorldState {
    pub fn from_cwd(cwd: &Path) -> Self {
        Self {
            environment: Some(RuntimeWorldEnvironment {
                cwd: Some(cwd.to_string_lossy().into_owned()),
                ..RuntimeWorldEnvironment::default()
            }),
            ..RuntimeWorldState::default()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.environment.is_none()
            && self.permissions.is_none()
            && self.collaboration.is_none()
            && self.multi_agent.is_none()
            && self.instruction_sections.is_empty()
    }

    pub fn render_environment_context(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }

        let mut rendered = String::from("<environment_context>");
        if let Some(environment) = &self.environment {
            render_environment(&mut rendered, environment);
        }
        if let Some(permissions) = &self.permissions {
            render_permissions(&mut rendered, permissions);
        }
        if let Some(collaboration) = &self.collaboration {
            render_mode(&mut rendered, "collaboration", collaboration);
        }
        if let Some(multi_agent) = &self.multi_agent {
            render_multi_agent_mode(&mut rendered, multi_agent);
        }
        for section in &self.instruction_sections {
            render_instruction_section(&mut rendered, section);
        }
        rendered.push_str("\n</environment_context>");
        Some(rendered)
    }
}

fn render_environment(rendered: &mut String, environment: &RuntimeWorldEnvironment) {
    push_text_node(rendered, "cwd", environment.cwd.as_deref());
    push_text_node(
        rendered,
        "project_root",
        environment.project_root.as_deref(),
    );
    push_text_node(
        rendered,
        "workspace_id",
        environment.workspace_id.as_deref(),
    );
    push_text_node(rendered, "thread_id", environment.thread_id.as_deref());
    push_text_node(rendered, "turn_id", environment.turn_id.as_deref());
    if environment.provider.is_some()
        || environment.model.is_some()
        || environment.reasoning_effort.is_some()
    {
        rendered.push_str("\n  <model");
        push_attr(rendered, "provider", environment.provider.as_deref());
        push_attr(rendered, "name", environment.model.as_deref());
        push_attr(
            rendered,
            "reasoning_effort",
            environment.reasoning_effort.as_deref(),
        );
        rendered.push_str(" />");
    }
}

fn render_permissions(rendered: &mut String, permissions: &RuntimeWorldPermissions) {
    if permissions.approval_policy.is_none()
        && permissions.sandbox_policy.is_none()
        && permissions.web_search.is_none()
    {
        return;
    }
    rendered.push_str("\n  <permissions");
    push_attr(
        rendered,
        "approval_policy",
        permissions.approval_policy.as_deref(),
    );
    push_attr(
        rendered,
        "sandbox_policy",
        permissions.sandbox_policy.as_deref(),
    );
    if let Some(web_search) = permissions.web_search {
        push_attr(
            rendered,
            "web_search",
            Some(if web_search { "enabled" } else { "disabled" }),
        );
    }
    rendered.push_str(" />");
}

fn render_mode(rendered: &mut String, tag: &str, mode: &RuntimeWorldMode) {
    if mode.mode.trim().is_empty() {
        return;
    }
    rendered.push_str("\n  <");
    rendered.push_str(tag);
    push_attr(rendered, "mode", Some(mode.mode.as_str()));
    push_attr(rendered, "source", mode.source.as_deref());
    rendered.push_str(" />");
}

fn render_multi_agent_mode(rendered: &mut String, mode: &MultiAgentMode) {
    const EXPLICIT_REQUEST_ONLY: &str = "Any earlier instruction enabling proactive multi-agent delegation no longer applies. Do not spawn sub-agents unless the user or applicable AGENTS.md/skill instructions explicitly ask for sub-agents, delegation, or parallel agent work.";
    const PROACTIVE: &str = "Proactive multi-agent delegation is active. Any earlier instruction requiring an explicit user request before spawning sub-agents no longer applies. Use sub-agents when parallel work would materially improve speed or quality. This mode remains active until a later multi-agent mode developer message changes it.";

    let body = match mode {
        MultiAgentMode::Custom(body) => body.as_str(),
        MultiAgentMode::ExplicitRequestOnly => EXPLICIT_REQUEST_ONLY,
        MultiAgentMode::Proactive => PROACTIVE,
    };
    if body.trim().is_empty() {
        return;
    }
    rendered.push_str("\n  <multi_agent_mode>");
    push_xml_escaped_text(rendered, body);
    rendered.push_str("</multi_agent_mode>");
}

fn render_instruction_section(rendered: &mut String, section: &RuntimeWorldInstructionSection) {
    if section.id.trim().is_empty() || section.body.trim().is_empty() {
        return;
    }
    rendered.push_str("\n  <instructions");
    push_attr(rendered, "id", Some(section.id.as_str()));
    push_attr(rendered, "source", section.source.as_deref());
    rendered.push('>');
    push_xml_escaped_text(rendered, &section.body);
    rendered.push_str("</instructions>");
}

fn push_text_node(rendered: &mut String, tag: &str, value: Option<&str>) {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return;
    };
    rendered.push_str("\n  <");
    rendered.push_str(tag);
    rendered.push('>');
    push_xml_escaped_text(rendered, value);
    rendered.push_str("</");
    rendered.push_str(tag);
    rendered.push('>');
}

fn push_attr(rendered: &mut String, name: &str, value: Option<&str>) {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return;
    };
    rendered.push(' ');
    rendered.push_str(name);
    rendered.push_str("=\"");
    push_xml_escaped_text(rendered, value);
    rendered.push('"');
}

fn push_xml_escaped_text(rendered: &mut String, value: &str) {
    for character in value.chars() {
        match character {
            '&' => rendered.push_str("&amp;"),
            '<' => rendered.push_str("&lt;"),
            '>' => rendered.push_str("&gt;"),
            '"' => rendered.push_str("&quot;"),
            '\'' => rendered.push_str("&apos;"),
            _ => rendered.push(character),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_environment_context_from_typed_sections() {
        let state = RuntimeWorldState {
            environment: Some(RuntimeWorldEnvironment {
                cwd: Some("/tmp/repo & app".to_string()),
                project_root: Some("/tmp/repo".to_string()),
                workspace_id: Some("workspace-1".to_string()),
                thread_id: Some("thread-1".to_string()),
                turn_id: Some("turn-1".to_string()),
                provider: Some("anthropic".to_string()),
                model: Some("claude".to_string()),
                reasoning_effort: Some("high".to_string()),
            }),
            permissions: Some(RuntimeWorldPermissions {
                approval_policy: Some("on-request".to_string()),
                sandbox_policy: Some("workspace-write".to_string()),
                web_search: Some(true),
            }),
            collaboration: Some(RuntimeWorldMode {
                mode: "solo".to_string(),
                source: Some("request".to_string()),
            }),
            multi_agent: Some(MultiAgentMode::Custom(
                "Delegate <independent> work".to_string(),
            )),
            instruction_sections: vec![RuntimeWorldInstructionSection {
                id: "agents_md".to_string(),
                body: "Use <boring> code".to_string(),
                source: Some("AGENTS.md".to_string()),
            }],
            source: Some(WORLD_STATE_SOURCE.to_string()),
        };

        assert_eq!(
            state.render_environment_context().as_deref(),
            Some(
                "<environment_context>\n  <cwd>/tmp/repo &amp; app</cwd>\n  <project_root>/tmp/repo</project_root>\n  <workspace_id>workspace-1</workspace_id>\n  <thread_id>thread-1</thread_id>\n  <turn_id>turn-1</turn_id>\n  <model provider=\"anthropic\" name=\"claude\" reasoning_effort=\"high\" />\n  <permissions approval_policy=\"on-request\" sandbox_policy=\"workspace-write\" web_search=\"enabled\" />\n  <collaboration mode=\"solo\" source=\"request\" />\n  <multi_agent_mode>Delegate &lt;independent&gt; work</multi_agent_mode>\n  <instructions id=\"agents_md\" source=\"AGENTS.md\">Use &lt;boring&gt; code</instructions>\n</environment_context>"
            )
        );
    }

    #[test]
    fn omits_empty_world_state() {
        assert!(RuntimeWorldState::default()
            .render_environment_context()
            .is_none());
    }

    #[test]
    fn cwd_only_state_does_not_claim_app_server_provenance() {
        let state = RuntimeWorldState::from_cwd(Path::new("/tmp/workspace"));

        assert_eq!(state.source, None);
        assert_eq!(
            state.render_environment_context().as_deref(),
            Some("<environment_context>\n  <cwd>/tmp/workspace</cwd>\n</environment_context>")
        );
    }
}
