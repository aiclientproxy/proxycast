//! Agent Skills 轻量 metadata 的 prompt 渲染。

use crate::agent_body::AgentSkillBody;
use crate::agent_snapshot::{AgentSkillMetadata, AgentSkillSnapshot};

pub const DEFAULT_AGENT_SKILL_RENDER_CHAR_BUDGET: usize = 4_000;
pub const DEFAULT_AGENT_SKILL_BODY_RENDER_CHAR_BUDGET: usize = 12_000;

const AGENT_SKILLS_PROMPT_MARKER: &str = "<skills_instructions>";
const AGENT_SKILLS_PROMPT_END_MARKER: &str = "</skills_instructions>";
const AGENT_SKILL_BODY_PROMPT_MARKER: &str = "<selected_skill_instructions>";
const AGENT_SKILL_BODY_PROMPT_END_MARKER: &str = "</selected_skill_instructions>";
const DESCRIPTION_MAX_CHARS: usize = 180;
const WHEN_TO_USE_MAX_CHARS: usize = 160;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentSkillRenderOptions {
    pub char_budget: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentSkillBodyRenderOptions {
    pub char_budget: usize,
}

impl Default for AgentSkillRenderOptions {
    fn default() -> Self {
        Self {
            char_budget: DEFAULT_AGENT_SKILL_RENDER_CHAR_BUDGET,
        }
    }
}

impl Default for AgentSkillBodyRenderOptions {
    fn default() -> Self {
        Self {
            char_budget: DEFAULT_AGENT_SKILL_BODY_RENDER_CHAR_BUDGET,
        }
    }
}

pub fn render_available_agent_skills(
    snapshot: &AgentSkillSnapshot,
    options: AgentSkillRenderOptions,
) -> Option<String> {
    if snapshot.skills.is_empty() || options.char_budget == 0 {
        return None;
    }

    let mut prompt = String::from(
        "<skills_instructions>\n## 可用 Agent Skills\n这些条目只是轻量 metadata。需要实际使用本地 skill 时，必须先读取对应 `SKILL.md`；scope=orchestrator 的 `skill://` locator 必须通过 `read_mcp_resource` 从 server=`codex_apps` 读取该 URI，不能作为本地文件访问。不要因为这里列出 skill 就默认调用 SkillTool、扩大工具权限或声称已经执行。\n",
    );
    if char_len(&prompt) >= options.char_budget {
        return None;
    }

    let mut rendered_count = 0usize;
    for skill in &snapshot.skills {
        let line = render_skill_line(skill);
        if try_push_with_budget(&mut prompt, &line, options.char_budget) {
            rendered_count += 1;
            continue;
        }

        let remaining = options
            .char_budget
            .saturating_sub(char_len(&prompt))
            .saturating_sub(char_len(AGENT_SKILLS_PROMPT_END_MARKER))
            .saturating_sub(1);
        if remaining > 24 {
            let compact = render_compact_skill_line(skill, remaining);
            if !compact.is_empty() {
                let _ = try_push_with_budget(&mut prompt, &compact, options.char_budget);
                rendered_count += 1;
            }
        }
        break;
    }

    if rendered_count == 0 {
        return None;
    }

    let omitted = snapshot.skills.len().saturating_sub(rendered_count);
    if omitted > 0 {
        let notice = format!("- 另有 {omitted} 个 skill 因预算限制未列出。\n");
        let _ = try_push_with_budget(&mut prompt, &notice, options.char_budget);
    }
    if !prompt.ends_with('\n') {
        prompt.push('\n');
    }
    if char_len(&prompt) + char_len(AGENT_SKILLS_PROMPT_END_MARKER) <= options.char_budget {
        prompt.push_str(AGENT_SKILLS_PROMPT_END_MARKER);
    }

    Some(prompt)
}

pub fn render_selected_agent_skill_bodies(
    bodies: &[AgentSkillBody],
    options: AgentSkillBodyRenderOptions,
) -> Option<String> {
    if bodies.is_empty() || options.char_budget == 0 {
        return None;
    }

    let mut prompt = String::from(
        "<selected_skill_instructions>\n## 已按用户显式选择加载的 Agent Skill\n以下内容来自被用户显式点名或引用的 `SKILL.md`。请按这些指令处理当前任务；但不要因此默认调用 SkillTool、扩大工具权限或声称已经执行外部动作。\n",
    );
    if char_len(&prompt) >= options.char_budget {
        return None;
    }

    let mut rendered_count = 0usize;
    for body in bodies {
        let header = format!(
            "### `{}` scope={} path=`{}`\n",
            body.locator.name,
            body.locator.scope.as_label(),
            body.locator.skill_file_path.display()
        );
        let remaining = options
            .char_budget
            .saturating_sub(char_len(&prompt))
            .saturating_sub(char_len(AGENT_SKILL_BODY_PROMPT_END_MARKER))
            .saturating_sub(1);
        if remaining <= char_len(&header) + 8 {
            break;
        }
        let body_budget = remaining.saturating_sub(char_len(&header));
        let markdown =
            truncate_to_char_budget(&render_skill_body_with_references(body), body_budget);
        let section = format!("{header}{}\n", markdown.trim());
        if try_push_with_budget(&mut prompt, &section, options.char_budget) {
            rendered_count += 1;
        } else {
            break;
        }
    }

    if rendered_count == 0 {
        return None;
    }
    let omitted = bodies.len().saturating_sub(rendered_count);
    if omitted > 0 {
        let notice = format!("- 另有 {omitted} 个显式选择的 skill 因预算限制未注入正文。\n");
        let _ = try_push_with_budget(&mut prompt, &notice, options.char_budget);
    }
    if !prompt.ends_with('\n') {
        prompt.push('\n');
    }
    if char_len(&prompt) + char_len(AGENT_SKILL_BODY_PROMPT_END_MARKER) <= options.char_budget {
        prompt.push_str(AGENT_SKILL_BODY_PROMPT_END_MARKER);
    }

    Some(prompt)
}

fn render_skill_body_with_references(body: &AgentSkillBody) -> String {
    let mut rendered = body.markdown_content.trim().to_string();
    if body.references.is_empty() {
        return rendered;
    }
    rendered.push_str("\n\n## Loaded references\n");
    for reference in &body.references {
        rendered.push_str(&format!(
            "\n### `{}`\n{}\n",
            reference.relative_path,
            reference.content.trim()
        ));
    }
    rendered
}

fn render_skill_line(skill: &AgentSkillMetadata) -> String {
    let mut parts = Vec::new();
    let display_name = if skill.interface.display_name == skill.name {
        String::new()
    } else {
        format!(
            " display=`{}`",
            normalize_inline_text(&skill.interface.display_name)
        )
    };
    parts.push(format!(
        "- `{}`{} scope={} path=`{}`",
        skill.name,
        display_name,
        skill.scope.as_label(),
        skill.skill_file_path.display()
    ));

    let description = truncate_text(&skill.description, DESCRIPTION_MAX_CHARS);
    if !description.is_empty() {
        parts.push(format!("desc={description}"));
    }
    if let Some(when_to_use) = skill.policy.when_to_use.as_ref() {
        let when_to_use = truncate_text(when_to_use, WHEN_TO_USE_MAX_CHARS);
        if !when_to_use.is_empty() {
            parts.push(format!("when={when_to_use}"));
        }
    }
    if let Some(argument_hint) = skill.interface.argument_hint.as_ref() {
        let argument_hint = truncate_text(argument_hint, 80);
        if !argument_hint.is_empty() {
            parts.push(format!("args={argument_hint}"));
        }
    }
    if !skill.capabilities.is_empty() {
        parts.push(format!("declared_tools={}", skill.capabilities.join(",")));
    }

    format!("{}。\n", parts.join("; "))
}

fn render_compact_skill_line(skill: &AgentSkillMetadata, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let base = format!(
        "- `{}` scope={} path=`{}`。\n",
        skill.name,
        skill.scope.as_label(),
        skill.skill_file_path.display()
    );
    truncate_to_char_budget(&base, max_chars)
}

fn try_push_with_budget(target: &mut String, value: &str, char_budget: usize) -> bool {
    if char_len(target) + char_len(value) > char_budget {
        return false;
    }
    target.push_str(value);
    true
}

fn truncate_text(value: &str, max_chars: usize) -> String {
    let value = normalize_inline_text(value);
    truncate_to_char_budget(&value, max_chars)
}

fn truncate_to_char_budget(value: &str, max_chars: usize) -> String {
    if char_len(value) <= max_chars {
        return value.to_string();
    }
    if max_chars <= 1 {
        return "…".chars().take(max_chars).collect();
    }
    let mut truncated: String = value.chars().take(max_chars - 1).collect();
    truncated.push('…');
    truncated
}

fn normalize_inline_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn char_len(value: &str) -> usize {
    value.chars().count()
}

pub fn contains_agent_skills_prompt(system_prompt: &str) -> bool {
    system_prompt.contains(AGENT_SKILLS_PROMPT_MARKER)
}

pub fn contains_selected_agent_skill_body_prompt(system_prompt: &str) -> bool {
    system_prompt.contains(AGENT_SKILL_BODY_PROMPT_MARKER)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AgentSkillAuthority, AgentSkillBodyLocator, AgentSkillDependencies, AgentSkillInterface,
        AgentSkillMetadata, AgentSkillPolicy, AgentSkillScope, AgentSkillSource,
        AgentSkillToolDependency,
    };
    use std::path::PathBuf;

    #[test]
    fn render_returns_none_for_empty_snapshot() {
        assert!(render_available_agent_skills(
            &AgentSkillSnapshot::default(),
            AgentSkillRenderOptions::default()
        )
        .is_none());
    }

    #[test]
    fn render_includes_metadata_and_not_body() {
        let snapshot = AgentSkillSnapshot {
            roots: Vec::new(),
            skills: vec![metadata(
                "research",
                "Research",
                "Search public sources and cite them.",
                Some("Use when the answer needs current source-backed research."),
            )],
        };

        let prompt =
            render_available_agent_skills(&snapshot, AgentSkillRenderOptions::default()).unwrap();

        assert!(prompt.contains("<skills_instructions>"));
        assert!(prompt.contains("## 可用 Agent Skills"));
        assert!(prompt.contains("`research`"));
        assert!(prompt.contains("Search public sources"));
        assert!(prompt.contains("path=`/tmp/skills/research/SKILL.md`"));
        assert!(prompt.contains("必须先读取对应 `SKILL.md`"));
        assert!(!prompt.contains("Full body should not be rendered"));
    }

    #[test]
    fn render_respects_budget_and_reports_omitted_skills() {
        let snapshot = AgentSkillSnapshot {
            roots: Vec::new(),
            skills: vec![
                metadata("alpha", "Alpha", "A".repeat(500).as_str(), None),
                metadata("beta", "Beta", "B".repeat(500).as_str(), None),
                metadata("gamma", "Gamma", "C".repeat(500).as_str(), None),
            ],
        };

        let prompt =
            render_available_agent_skills(&snapshot, AgentSkillRenderOptions { char_budget: 360 })
                .expect("prompt");

        assert!(prompt.chars().count() <= 360);
        assert!(prompt.contains("`alpha`"));
        assert!(prompt.contains("因预算限制未列出"));
    }

    #[test]
    fn render_selected_bodies_includes_skill_markdown() {
        let body = AgentSkillBody {
            locator: AgentSkillBodyLocator {
                name: "writer".to_string(),
                scope: AgentSkillScope::Project,
                directory: PathBuf::from("/tmp/skills/writer"),
                skill_file_path: PathBuf::from("/tmp/skills/writer/SKILL.md"),
            },
            markdown_content: "# Writer\n\nUse direct language.".to_string(),
            references: vec![crate::agent_body::AgentSkillReferenceBody {
                relative_path: "references/style.md".to_string(),
                content: "# Style\n\nUse concise paragraphs.".to_string(),
            }],
        };

        let prompt =
            render_selected_agent_skill_bodies(&[body], AgentSkillBodyRenderOptions::default())
                .expect("prompt");

        assert!(prompt.contains("<selected_skill_instructions>"));
        assert!(prompt.contains("`writer`"));
        assert!(prompt.contains("# Writer"));
        assert!(prompt.contains("Use direct language."));
        assert!(prompt.contains("references/style.md"));
        assert!(prompt.contains("Use concise paragraphs."));
        assert!(prompt.contains("不要因此默认调用 SkillTool"));
    }

    fn metadata(
        name: &str,
        display_name: &str,
        description: &str,
        when_to_use: Option<&str>,
    ) -> AgentSkillMetadata {
        AgentSkillMetadata {
            skill_id: format!("project:{name}"),
            name: name.to_string(),
            description: description.to_string(),
            scope: AgentSkillScope::Project,
            source: AgentSkillSource::Project,
            authority: AgentSkillAuthority::Workspace,
            enabled: true,
            interface: AgentSkillInterface {
                display_name: display_name.to_string(),
                execution_mode: "prompt".to_string(),
                provider: None,
                model: None,
                argument_hint: None,
            },
            dependencies: AgentSkillDependencies {
                tools: vec![AgentSkillToolDependency {
                    dependency_type: "runtime_tool".to_string(),
                    value: "Read".to_string(),
                    required: true,
                }],
            },
            policy: AgentSkillPolicy {
                allow_implicit_invocation: true,
                when_to_use: when_to_use.map(ToString::to_string),
            },
            capabilities: vec!["Read".to_string()],
            directory: PathBuf::from(format!("/tmp/skills/{name}")),
            skill_file_path: PathBuf::from(format!("/tmp/skills/{name}/SKILL.md")),
        }
    }
}
