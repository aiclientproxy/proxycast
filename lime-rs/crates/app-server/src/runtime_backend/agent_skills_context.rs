use std::path::Path;

use crate::runtime::PluginTurnSnapshot;
use lime_core::config::SkillConfig;
use lime_skills::{
    agent_skill_roots_for_workspace, build_agent_skill_snapshot_from_roots,
    contains_agent_skills_prompt, evaluate_agent_skill_selection_bodies,
    render_available_agent_skills, reorder_agent_skill_snapshot_for_query,
    select_agent_skills_by_name_candidates, select_explicit_agent_skills,
    select_implicit_agent_skills, AgentSkillMetadata, AgentSkillRenderOptions, AgentSkillSelection,
    AgentSkillSelectionEvaluation, AgentSkillSelectionTrigger, AgentSkillSnapshot,
    DEFAULT_AGENT_SKILL_BODY_TOKEN_BUDGET,
};
use serde_json::Value;

pub(super) struct AgentSkillsTurnContext {
    pub system_prompt: Option<String>,
    pub snapshot: Option<AgentSkillSnapshot>,
}

pub(super) fn agent_skills_context_for_turn_with_plugins_and_orchestrator(
    system_prompt: Option<String>,
    user_input: &str,
    metadata_values: &[&Value],
    working_dir: Option<&Path>,
    project_root: Option<&Path>,
    plugin_snapshots: &[PluginTurnSnapshot],
    orchestrator_skills: &[AgentSkillMetadata],
) -> AgentSkillsTurnContext {
    if should_suppress_agent_skills_for_metadata(metadata_values) {
        return AgentSkillsTurnContext {
            system_prompt,
            snapshot: None,
        };
    }

    let snapshot = build_agent_skill_snapshot_for_turn_with_plugins_and_orchestrator(
        working_dir,
        project_root,
        metadata_values,
        plugin_snapshots,
        orchestrator_skills,
    );
    let system_prompt = append_expert_skill_hints(system_prompt, metadata_values, &snapshot);

    if system_prompt
        .as_deref()
        .is_some_and(contains_agent_skills_prompt)
    {
        return AgentSkillsTurnContext {
            system_prompt,
            snapshot: Some(snapshot),
        };
    }

    let render_snapshot =
        reorder_agent_skill_snapshot_for_query(&snapshot, user_input, Default::default());
    let Some(context) =
        render_available_agent_skills(&render_snapshot, AgentSkillRenderOptions::default())
    else {
        return AgentSkillsTurnContext {
            system_prompt,
            snapshot: Some(snapshot),
        };
    };

    AgentSkillsTurnContext {
        system_prompt: append_context_block(system_prompt, context),
        snapshot: Some(snapshot),
    }
}

pub(super) fn selected_agent_skill_names_for_turn(
    user_input: &str,
    metadata_values: &[&Value],
    working_dir: Option<&Path>,
    project_root: Option<&Path>,
) -> Vec<String> {
    selected_agent_skill_names_for_turn_with_plugins(
        user_input,
        metadata_values,
        working_dir,
        project_root,
        &[],
    )
}

pub(super) fn selected_agent_skill_names_for_turn_with_plugins(
    user_input: &str,
    metadata_values: &[&Value],
    working_dir: Option<&Path>,
    project_root: Option<&Path>,
    plugin_snapshots: &[PluginTurnSnapshot],
) -> Vec<String> {
    if should_suppress_agent_skills_for_metadata(metadata_values) {
        return Vec::new();
    }

    let snapshot = build_agent_skill_snapshot_for_turn_with_plugins(
        working_dir,
        project_root,
        metadata_values,
        plugin_snapshots,
    );
    selected_agent_skill_selections_with_plugins(
        user_input,
        metadata_values,
        &snapshot,
        plugin_snapshots,
    )
    .into_iter()
    .map(|selection| selection.locator.name)
    .collect()
}

pub(super) fn build_agent_skill_snapshot_for_turn_with_plugins(
    working_dir: Option<&Path>,
    project_root: Option<&Path>,
    metadata_values: &[&Value],
    plugin_snapshots: &[PluginTurnSnapshot],
) -> AgentSkillSnapshot {
    build_agent_skill_snapshot_for_turn_with_plugins_and_orchestrator(
        working_dir,
        project_root,
        metadata_values,
        plugin_snapshots,
        &[],
    )
}

pub(super) fn build_agent_skill_snapshot_for_turn_with_plugins_and_orchestrator(
    working_dir: Option<&Path>,
    project_root: Option<&Path>,
    metadata_values: &[&Value],
    plugin_snapshots: &[PluginTurnSnapshot],
    orchestrator_skills: &[AgentSkillMetadata],
) -> AgentSkillSnapshot {
    let mut roots = agent_skill_roots_for_workspace(working_dir, project_root);
    roots.extend(
        plugin_snapshots
            .iter()
            .map(|snapshot| lime_skills::AgentSkillRoot {
                path: snapshot.package_root.join("skills"),
                scope: lime_skills::AgentSkillScope::App,
            }),
    );
    let mut snapshot = build_agent_skill_snapshot_from_roots(roots);
    for skill in orchestrator_skills {
        if snapshot.skills.iter().any(|existing| {
            existing.authority == skill.authority
                && existing.skill_file_path == skill.skill_file_path
        }) {
            continue;
        }
        snapshot.skills.push(skill.clone());
    }
    let config = metadata_values
        .iter()
        .filter_map(|metadata| metadata.get("skills"))
        .filter_map(|skills| skills.get("config"))
        .filter_map(|config| serde_json::from_value::<Vec<SkillConfig>>(config.clone()).ok())
        .flatten()
        .collect::<Vec<_>>();
    lime_skills::apply_agent_skill_config(&mut snapshot, &config);
    snapshot
}

pub(super) fn selected_agent_skill_body_evaluations(
    selections: &[AgentSkillSelection],
    snapshot: &AgentSkillSnapshot,
) -> Vec<AgentSkillSelectionEvaluation> {
    evaluate_agent_skill_selection_bodies(
        selections,
        snapshot,
        DEFAULT_AGENT_SKILL_BODY_TOKEN_BUDGET,
    )
}

pub(super) fn selected_agent_skill_selections_with_plugins(
    user_input: &str,
    metadata_values: &[&Value],
    snapshot: &AgentSkillSnapshot,
    _plugin_snapshots: &[PluginTurnSnapshot],
) -> Vec<AgentSkillSelection> {
    if should_suppress_agent_skills_for_metadata(metadata_values) {
        return Vec::new();
    }

    let mut selections = select_catalog_bound_agent_skills(metadata_values, snapshot);
    selections.extend(select_expert_bound_agent_skills(metadata_values, snapshot));
    selections.extend(select_explicit_agent_skills(user_input, snapshot));
    let selections = dedupe_agent_skill_selections(selections);
    if !selections.is_empty() {
        return selections;
    }
    select_implicit_agent_skills(user_input, snapshot)
}

pub(super) fn selected_agent_skill_body_selections_for_prompt_with_plugins(
    user_input: &str,
    metadata_values: &[&Value],
    snapshot: &AgentSkillSnapshot,
    _plugin_snapshots: &[PluginTurnSnapshot],
) -> Vec<AgentSkillSelection> {
    if should_suppress_agent_skills_for_metadata(metadata_values) {
        return Vec::new();
    }

    let mut selections = select_catalog_bound_agent_skills(metadata_values, snapshot);
    selections.extend(select_explicit_agent_skills(user_input, snapshot));
    let selections = dedupe_agent_skill_selections(selections);
    if !selections.is_empty() {
        return selections;
    }
    if has_expert_skill_refs(metadata_values) || has_workspace_skill_runtime_enable(metadata_values)
    {
        return Vec::new();
    }
    select_implicit_agent_skills(user_input, snapshot)
}

fn select_catalog_bound_agent_skills(
    metadata_values: &[&Value],
    snapshot: &AgentSkillSnapshot,
) -> Vec<AgentSkillSelection> {
    select_agent_skills_by_name_candidates(
        catalog_bound_skill_candidates(metadata_values),
        snapshot,
        AgentSkillSelectionTrigger::CatalogBinding,
    )
}

fn select_expert_bound_agent_skills(
    metadata_values: &[&Value],
    snapshot: &AgentSkillSnapshot,
) -> Vec<AgentSkillSelection> {
    select_agent_skills_by_name_candidates(
        expert_bound_skill_candidates(metadata_values),
        snapshot,
        AgentSkillSelectionTrigger::ExpertBinding,
    )
}

fn dedupe_agent_skill_selections(selections: Vec<AgentSkillSelection>) -> Vec<AgentSkillSelection> {
    let mut deduped = Vec::new();
    for selection in selections {
        if deduped.iter().any(|existing: &AgentSkillSelection| {
            existing.locator.skill_file_path == selection.locator.skill_file_path
        }) {
            continue;
        }
        deduped.push(selection);
    }
    deduped
}

fn catalog_bound_skill_candidates(metadata_values: &[&Value]) -> Vec<String> {
    let mut candidates = Vec::new();
    for metadata in metadata_values {
        if let Some(service_scene_run) = service_scene_run_value(metadata) {
            collect_skill_locator_candidates(&mut candidates, service_scene_run);
            push_string_candidate(
                &mut candidates,
                service_scene_run
                    .get("skill_key")
                    .or_else(|| service_scene_run.get("skillKey")),
            );
            push_string_candidate(
                &mut candidates,
                service_scene_run
                    .get("linked_skill_id")
                    .or_else(|| service_scene_run.get("linkedSkillId")),
            );
            push_string_candidate(
                &mut candidates,
                service_scene_run
                    .get("skill_id")
                    .or_else(|| service_scene_run.get("skillId")),
            );
        }
        collect_model_skill_launch_candidates(&mut candidates, metadata);
    }
    candidates
}

fn expert_bound_skill_candidates(metadata_values: &[&Value]) -> Vec<String> {
    let mut candidates = Vec::new();
    for skill_ref in expert_skill_refs(metadata_values) {
        if skill_ref.trim().starts_with("service-skill:") {
            continue;
        }
        for candidate in expert_ref_name_candidates(&skill_ref) {
            push_unique_string(&mut candidates, &candidate);
        }
    }
    candidates
}

fn has_expert_skill_refs(metadata_values: &[&Value]) -> bool {
    !expert_skill_refs(metadata_values).is_empty()
}

fn has_workspace_skill_runtime_enable(metadata_values: &[&Value]) -> bool {
    metadata_values.iter().any(|metadata| {
        metadata
            .pointer("/harness/workspace_skill_runtime_enable")
            .or_else(|| metadata.pointer("/harness/workspaceSkillRuntimeEnable"))
            .or_else(|| metadata.get("workspace_skill_runtime_enable"))
            .or_else(|| metadata.get("workspaceSkillRuntimeEnable"))
            .is_some()
    })
}

fn append_expert_skill_hints(
    system_prompt: Option<String>,
    metadata_values: &[&Value],
    snapshot: &AgentSkillSnapshot,
) -> Option<String> {
    let refs = expert_skill_refs(metadata_values);
    if refs.is_empty() {
        return system_prompt;
    }

    let mut lines = Vec::new();
    for skill_ref in refs {
        let Some(skill) = find_skill_for_expert_ref(snapshot, &skill_ref) else {
            lines.push(format!(
                "- `{skill_ref}`: 未在当前 Agent Skills snapshot 中匹配到可读 `SKILL.md`。"
            ));
            continue;
        };
        lines.push(format!(
            "- `{skill_ref}` -> `{}` scope={} path=`{}`",
            skill.name,
            skill.scope.as_label(),
            skill.skill_file_path.display()
        ));
    }
    if lines.is_empty() {
        return system_prompt;
    }

    append_context_block(
        system_prompt,
        format!(
            "<expert_skill_refs>\n## 专家绑定的 Agent Skill 候选\n这些 skillRefs 来自当前专家 metadata；它们只是候选，不是已执行的技能。需要使用候选 skill 时，首个相关工具调用必须先调用 `skill_search` 搜索并确认候选，再调用 `Skill` 读取 `SKILL.md` 并执行；不要因为候选存在就跳过 selector、授权或声称外部动作已经执行。\n{}\n</expert_skill_refs>",
            lines.join("\n")
        ),
    )
}

fn expert_skill_refs(metadata_values: &[&Value]) -> Vec<String> {
    let mut refs = Vec::new();
    for metadata in metadata_values {
        collect_string_array_candidates(&mut refs, metadata.pointer("/harness/expert/skill_refs"));
        collect_string_array_candidates(&mut refs, metadata.pointer("/harness/expert/skillRefs"));
        collect_string_array_candidates(&mut refs, metadata.pointer("/expert/skillRefs"));
        collect_string_array_candidates(&mut refs, metadata.pointer("/expert/skill_refs"));
    }
    refs
}

fn collect_string_array_candidates(candidates: &mut Vec<String>, value: Option<&Value>) {
    let Some(values) = value.and_then(Value::as_array) else {
        return;
    };
    for value in values {
        push_string_candidate(candidates, Some(value));
    }
}

fn find_skill_for_expert_ref<'a>(
    snapshot: &'a AgentSkillSnapshot,
    skill_ref: &str,
) -> Option<&'a lime_skills::AgentSkillMetadata> {
    if skill_ref.trim().starts_with("service-skill:") {
        return None;
    }
    let candidates = expert_ref_name_candidates(skill_ref);
    snapshot.skills.iter().find(|skill| {
        candidates.iter().any(|candidate| {
            skill.name.eq_ignore_ascii_case(candidate)
                || skill
                    .directory
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|directory| directory.eq_ignore_ascii_case(candidate))
                || skill
                    .skill_file_path
                    .to_string_lossy()
                    .eq_ignore_ascii_case(candidate)
        })
    })
}

fn expert_ref_name_candidates(skill_ref: &str) -> Vec<String> {
    let trimmed = skill_ref.trim();
    let candidate = trimmed
        .strip_prefix("skill:")
        .or_else(|| trimmed.strip_prefix("workspace_skill:"))
        .or_else(|| trimmed.strip_prefix("service-skill:"))
        .unwrap_or(trimmed)
        .trim();
    let candidate = candidate.split('@').next().unwrap_or(candidate).trim();
    let mut candidates = Vec::new();
    push_unique_string(&mut candidates, candidate);
    if let Some(stripped) = candidate.strip_prefix("project:") {
        push_unique_string(&mut candidates, stripped);
    }
    candidates
}

fn collect_skill_locator_candidates(candidates: &mut Vec<String>, service_scene_run: &Value) {
    let Some(locator) = service_scene_run
        .get("skill_locator")
        .or_else(|| service_scene_run.get("skillLocator"))
    else {
        return;
    };
    push_string_candidate(
        candidates,
        locator
            .get("name")
            .or_else(|| locator.get("skill_name"))
            .or_else(|| locator.get("skillName")),
    );
    push_string_candidate(
        candidates,
        locator
            .get("directory")
            .or_else(|| locator.get("skill_directory"))
            .or_else(|| locator.get("skillDirectory")),
    );
    push_string_candidate(
        candidates,
        locator
            .get("skill_file_path")
            .or_else(|| locator.get("skillFilePath")),
    );
}

fn service_scene_run_value(metadata: &Value) -> Option<&Value> {
    metadata
        .pointer("/harness/service_scene_launch/service_scene_run")
        .or_else(|| metadata.pointer("/harness/serviceSceneLaunch/serviceSceneRun"))
        .or_else(|| metadata.pointer("/service_scene_launch/service_scene_run"))
        .or_else(|| metadata.pointer("/serviceSceneLaunch/serviceSceneRun"))
}

fn collect_model_skill_launch_candidates(candidates: &mut Vec<String>, metadata: &Value) {
    for launch in model_skill_launch_values(metadata) {
        push_string_candidate(
            candidates,
            launch.get("skill_name").or_else(|| launch.get("skillName")),
        );
    }
}

fn model_skill_launch_values(metadata: &Value) -> Vec<&Value> {
    let mut values = Vec::new();
    if let Some(harness) = metadata.get("harness").and_then(Value::as_object) {
        for (key, value) in harness {
            if looks_like_model_skill_launch_key(key) && !is_retired_model_skill_launch_key(key) {
                values.push(value);
            }
        }
    }
    if let Some(object) = metadata.as_object() {
        for (key, value) in object {
            if looks_like_model_skill_launch_key(key) && !is_retired_model_skill_launch_key(key) {
                values.push(value);
            }
        }
    }
    values
}

fn looks_like_model_skill_launch_key(key: &str) -> bool {
    let normalized = key.trim();
    normalized.ends_with("_skill_launch") || normalized.ends_with("SkillLaunch")
}

fn is_retired_model_skill_launch_key(key: &str) -> bool {
    matches!(key.trim(), "image_skill_launch" | "imageSkillLaunch")
}

fn should_suppress_agent_skills_for_metadata(metadata_values: &[&Value]) -> bool {
    metadata_values.iter().any(|metadata| {
        has_current_image_command_intent(metadata) || has_retired_image_skill_launch(metadata)
    })
}

fn has_current_image_command_intent(metadata: &Value) -> bool {
    [
        "/harness/image_command_intent",
        "/harness/imageCommandIntent",
        "/image_command_intent",
        "/imageCommandIntent",
    ]
    .iter()
    .any(|pointer| metadata.pointer(pointer).is_some())
}

fn has_retired_image_skill_launch(metadata: &Value) -> bool {
    [
        "/harness/image_skill_launch",
        "/harness/imageSkillLaunch",
        "/image_skill_launch",
        "/imageSkillLaunch",
    ]
    .iter()
    .any(|pointer| metadata.pointer(pointer).is_some())
}

fn push_string_candidate(candidates: &mut Vec<String>, value: Option<&Value>) {
    let Some(candidate) = value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|candidate| !candidate.is_empty())
    else {
        return;
    };
    if !candidates.iter().any(|existing| existing == candidate) {
        candidates.push(candidate.to_string());
    }
}

fn push_unique_string(values: &mut Vec<String>, value: &str) {
    let value = value.trim();
    if value.is_empty() {
        return;
    }
    if !values
        .iter()
        .any(|existing| existing.eq_ignore_ascii_case(value))
    {
        values.push(value.to_string());
    }
}

fn append_context_block(system_prompt: Option<String>, context: String) -> Option<String> {
    let mut prompt = system_prompt.unwrap_or_default();
    if !prompt.trim().is_empty() {
        prompt.push_str("\n\n");
    }
    prompt.push_str(&context);
    Some(prompt)
}

#[cfg(test)]
fn append_agent_skills_context_to_system_prompt(
    system_prompt: Option<String>,
    user_input: &str,
    metadata_values: &[&Value],
    working_dir: Option<&Path>,
    project_root: Option<&Path>,
) -> Option<String> {
    agent_skills_context_for_turn_with_plugins_and_orchestrator(
        system_prompt,
        user_input,
        metadata_values,
        working_dir,
        project_root,
        &[],
        &[],
    )
    .system_prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn appends_project_skill_metadata_without_enabling_skilltool() {
        let workspace = TempDir::new().expect("workspace");
        write_skill(
            &workspace,
            "research",
            "Research",
            "Use source-backed research.",
        );

        let prompt = append_agent_skills_context_to_system_prompt(
            Some("base".to_string()),
            "hello",
            &[],
            Some(workspace.path()),
            Some(workspace.path()),
        )
        .expect("prompt");

        assert!(prompt.starts_with("base"));
        assert!(prompt.contains("## 可用 Agent Skills"));
        assert!(prompt.contains("`research`"));
        assert!(prompt.contains("必须先读取对应 `SKILL.md`"));
        assert!(!prompt.contains("Full body should not be rendered"));
        assert!(!prompt.contains("allow_model_skills"));
    }

    #[test]
    fn explicit_selection_does_not_inject_skill_body_into_system_prompt() {
        let workspace = TempDir::new().expect("workspace");
        write_skill(&workspace, "writer", "Writer", "Write clearly.");

        let prompt = append_agent_skills_context_to_system_prompt(
            Some("base".to_string()),
            "请用 $writer 改写这段话",
            &[],
            Some(workspace.path()),
            Some(workspace.path()),
        )
        .expect("prompt");

        assert!(prompt.contains("`writer`"));
        assert!(!prompt.contains("<selected_skill_instructions>"));
        assert!(!prompt.contains("# Body"));
        assert!(!prompt.contains("Full body should not be rendered."));
        assert!(prompt.contains("## 可用 Agent Skills"));
        assert!(!prompt.contains("allow_model_skills"));
    }

    #[test]
    fn body_evaluation_uses_the_real_skill_body_budget_before_prompt_injection() {
        let workspace = TempDir::new().expect("workspace");
        write_skill(&workspace, "writer", "Writer", "Write clearly.");
        let snapshot = build_agent_skill_snapshot_for_turn_with_plugins(
            Some(workspace.path()),
            Some(workspace.path()),
            &[],
            &[],
        );
        let selections = selected_agent_skill_body_selections_for_prompt_with_plugins(
            "$writer",
            &[],
            &snapshot,
            &[],
        );

        let evaluations = selected_agent_skill_body_evaluations(&selections, &snapshot);

        assert_eq!(evaluations.len(), 1);
        assert_eq!(evaluations[0].skill_id, "project:writer");
        assert_eq!(evaluations[0].body_budget.max_visible_tokens, 3_000);
        assert!(evaluations[0].body_budget.estimated_tokens.is_some());
    }

    #[test]
    fn skips_duplicate_skills_prompt() {
        let existing = "<skills_instructions>\n## Skills\n</skills_instructions>";

        let prompt = append_agent_skills_context_to_system_prompt(
            Some(existing.to_string()),
            "hello",
            &[],
            None,
            None,
        )
        .expect("prompt");

        assert_eq!(prompt, existing);
    }

    #[test]
    fn catalog_binding_keeps_body_out_of_system_prompt() {
        let workspace = TempDir::new().expect("workspace");
        write_skill(&workspace, "writer", "Writer", "Write clearly.");
        let metadata = serde_json::json!({
            "harness": {
                "service_scene_launch": {
                    "service_scene_run": {
                        "skill_locator": {
                            "source": "catalog",
                            "name": "local:writer"
                        },
                        "skill_id": "service-skill-writer",
                        "linked_skill_id": "service-skill-writer"
                    }
                }
            }
        });

        let prompt = append_agent_skills_context_to_system_prompt(
            Some("base".to_string()),
            "帮我处理这段话",
            &[&metadata],
            Some(workspace.path()),
            Some(workspace.path()),
        )
        .expect("prompt");

        assert!(prompt.contains("`writer`"));
        assert!(!prompt.contains("<selected_skill_instructions>"));
        assert!(!prompt.contains("# Body"));
        assert!(prompt.contains("## 可用 Agent Skills"));
    }

    #[test]
    fn retired_image_skill_launch_does_not_enter_agent_skills() {
        let workspace = TempDir::new().expect("workspace");
        let skill_dir = workspace.path().join(".agents/skills/image-router");
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: image-router
description: Generate image task.
allowed_tools: lime_create_image_generation_task
---

# Image Generate

Call lime_create_image_generation_task directly.
"#,
        )
        .expect("skill");
        let metadata = serde_json::json!({
            "harness": {
                "image_skill_launch": {
                    "skill_name": "image-router",
                    "kind": "image_task",
                    "image_task": {
                        "prompt": "广州夏天",
                        "modality_contract_key": "image_generation"
                    }
                }
            }
        });

        let prompt = append_agent_skills_context_to_system_prompt(
            Some("base".to_string()),
            "@配图 画一张广州夏天的图",
            &[&metadata],
            Some(workspace.path()),
            Some(workspace.path()),
        )
        .expect("prompt");
        assert_eq!(prompt, "base");
        assert!(!prompt.contains("<selected_skill_instructions>"));
        assert!(!prompt.contains("## 可用 Agent Skills"));
        assert!(!prompt.contains("Call lime_create_image_generation_task directly."));

        let names = selected_agent_skill_names_for_turn(
            "$image-router",
            &[&metadata],
            Some(workspace.path()),
            Some(workspace.path()),
        );
        assert!(names.is_empty());
    }

    #[test]
    fn image_command_intent_does_not_enter_agent_skills() {
        let workspace = TempDir::new().expect("workspace");
        write_skill(
            &workspace,
            "image_generate",
            "image_generate",
            "Generate image task.",
        );
        let metadata = serde_json::json!({
            "harness": {
                "image_command_intent": {
                    "kind": "image_task",
                    "image_task": {
                        "prompt": "广州夏天",
                        "modality_contract_key": "image_generation"
                    }
                }
            }
        });

        let prompt = append_agent_skills_context_to_system_prompt(
            Some("base".to_string()),
            "$image_generate 画一张广州夏天的图",
            &[&metadata],
            Some(workspace.path()),
            Some(workspace.path()),
        )
        .expect("prompt");
        assert_eq!(prompt, "base");

        let names = selected_agent_skill_names_for_turn(
            "$image_generate 画一张广州夏天的图",
            &[&metadata],
            Some(workspace.path()),
            Some(workspace.path()),
        );
        assert!(names.is_empty());
    }

    #[test]
    fn appends_expert_skill_refs_as_selector_candidates_without_body() {
        let workspace = TempDir::new().expect("workspace");
        write_skill(&workspace, "writer", "Writer", "Write clearly.");
        let metadata = serde_json::json!({
            "harness": {
                "expert": {
                    "skill_refs": [
                        "skill:writer",
                        "service-skill:daily-trend-briefing",
                        "skill:writer"
                    ]
                }
            }
        });

        let prompt = append_agent_skills_context_to_system_prompt(
            Some("base".to_string()),
            "帮我处理这段话",
            &[&metadata],
            Some(workspace.path()),
            Some(workspace.path()),
        )
        .expect("prompt");

        assert!(prompt.contains("<expert_skill_refs>"));
        assert!(prompt.contains("`skill:writer` -> `writer`"));
        assert!(
            prompt.contains("`service-skill:daily-trend-briefing`: 未在当前 Agent Skills snapshot")
        );
        assert!(prompt.contains("首个相关工具调用必须先调用 `skill_search`"));
        assert!(!prompt.contains("<selected_skill_instructions>"));
        assert!(!prompt.contains("# Body"));

        let names = selected_agent_skill_names_for_turn(
            "帮我处理这段话",
            &[&metadata],
            Some(workspace.path()),
            Some(workspace.path()),
        );
        assert_eq!(names, vec!["writer".to_string()]);
    }

    #[test]
    fn workspace_runtime_enable_should_not_implicitly_inject_skill_body() {
        let workspace = TempDir::new().expect("workspace");
        let skill_dir = workspace.path().join(".agents/skills/capability-report");
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: Capability Report\ndescription: Capability report live review.\nallowed_tools: Read\n---\n\n# Body\n",
        )
        .expect("skill");
        let metadata = serde_json::json!({
            "harness": {
                "workspace_skill_runtime_enable": {
                    "source": "manual_session_enable",
                    "approval": "manual",
                    "workspace_root": workspace.path().to_string_lossy(),
                    "bindings": [
                        {
                            "directory": "capability-report",
                            "registered_skill_directory": workspace
                                .path()
                                .join(".agents/skills/capability-report")
                                .to_string_lossy()
                        }
                    ]
                },
                "expert": {
                    "skill_refs": ["skill:capability-report"]
                }
            }
        });

        let prompt = append_agent_skills_context_to_system_prompt(
            Some("base".to_string()),
            "请先搜索 capability-report 后再执行",
            &[&metadata],
            Some(workspace.path()),
            Some(workspace.path()),
        )
        .expect("prompt");

        assert!(prompt.contains("<expert_skill_refs>"));
        assert!(prompt.contains("`skill:capability-report` -> `capability-report`"));
        assert!(!prompt.contains("<selected_skill_instructions>"));
        assert!(!prompt.contains("# Body"));
    }

    #[test]
    fn plugin_package_skill_path_enters_snapshot_catalog_without_body() {
        let package = TempDir::new().expect("package");
        let skill_dir = package.path().join("skills").join("article-writing");
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: article-writing
description: Draft article copy.
---

# Package Article Writing

Use package workflow rules.
"#,
        )
        .expect("skill file");
        let plugin_snapshot = PluginTurnSnapshot {
            id: "content-factory-app".to_string(),
            config_name: "content-factory-app@test".to_string(),
            display_name: "Content Factory".to_string(),
            package_root: package.path().to_path_buf(),
            skill_names: vec!["article-writing".to_string()],
            mcp_server_names: Vec::new(),
        };

        let context = agent_skills_context_for_turn_with_plugins_and_orchestrator(
            Some("base".to_string()),
            "写一篇文章",
            &[],
            None,
            None,
            &[plugin_snapshot],
            &[],
        );
        let prompt = context.system_prompt.expect("prompt");
        let snapshot = context.snapshot.expect("snapshot");
        let canonical_skill_path =
            std::fs::canonicalize(skill_dir.join("SKILL.md")).expect("canonical skill path");

        assert!(snapshot.skills.iter().any(|skill| {
            skill.name == "article-writing" && skill.skill_file_path == canonical_skill_path
        }));
        assert!(!prompt.contains("<selected_skill_instructions>"));
        assert!(!prompt.contains("# Package Article Writing"));
        assert!(!prompt.contains("Use package workflow rules."));
    }

    #[test]
    fn orchestrator_skill_enters_turn_snapshot_once_without_local_body_read() {
        let skill: AgentSkillMetadata = serde_json::from_value(serde_json::json!({
            "skill_id": "orchestrator:delivery:release-notes",
            "name": "delivery:release-notes",
            "description": "Draft source-backed release notes.",
            "scope": "orchestrator",
            "source": "orchestrator",
            "authority": "orchestrator",
            "enabled": true,
            "interface": {
                "display_name": "delivery:release-notes",
                "execution_mode": "mcp_resource",
                "provider": "codex_apps",
                "model": null,
                "argument_hint": null
            },
            "dependencies": { "tools": [] },
            "policy": {
                "allow_implicit_invocation": true,
                "when_to_use": null
            },
            "capabilities": [],
            "directory": "skill://delivery/release-notes",
            "skill_file_path": "skill://delivery/release-notes/SKILL.md"
        }))
        .expect("orchestrator skill metadata");

        let context = agent_skills_context_for_turn_with_plugins_and_orchestrator(
            Some("base".to_string()),
            "draft release notes",
            &[],
            None,
            None,
            &[],
            &[skill.clone(), skill],
        );
        let prompt = context.system_prompt.expect("prompt");
        let snapshot = context.snapshot.expect("snapshot");
        let matching = snapshot
            .skills
            .iter()
            .filter(|skill| skill.skill_id == "orchestrator:delivery:release-notes")
            .collect::<Vec<_>>();

        assert_eq!(matching.len(), 1);
        assert_eq!(
            matching[0].skill_file_path,
            Path::new("skill://delivery/release-notes/SKILL.md")
        );
        assert!(prompt.contains("`delivery:release-notes`"));
        assert!(!prompt.contains("<selected_skill_instructions>"));
    }

    #[test]
    fn catalog_bound_selection_ignores_unknown_skill_metadata() {
        let workspace = TempDir::new().expect("workspace");
        write_skill(&workspace, "writer", "Writer", "Write clearly.");
        let metadata = serde_json::json!({
            "harness": {
                "service_scene_launch": {
                    "service_scene_run": {
                        "skill_key": "missing"
                    }
                }
            }
        });

        let names = selected_agent_skill_names_for_turn(
            "",
            &[&metadata],
            Some(workspace.path()),
            Some(workspace.path()),
        );

        assert!(names.is_empty());
    }

    #[test]
    fn implicit_selection_keeps_body_out_of_system_prompt() {
        let workspace = TempDir::new().expect("workspace");
        write_skill(
            &workspace,
            "fact-check-lab",
            "Fact Check Lab",
            "联网信息检索与事实核验",
        );
        write_skill(&workspace, "writer", "Writer", "写作润色");

        let prompt = append_agent_skills_context_to_system_prompt(
            Some("base".to_string()),
            "帮我做最新事实核验和趋势调研",
            &[],
            Some(workspace.path()),
            Some(workspace.path()),
        )
        .expect("prompt");

        assert!(prompt.contains("`fact-check-lab`"));
        assert!(!prompt.contains("<selected_skill_instructions>"));
        assert!(!prompt.contains("# Body"));
        assert!(prompt.contains("## 可用 Agent Skills"));
    }

    #[test]
    fn implicit_selection_enables_selected_skill_name_for_turn() {
        let workspace = TempDir::new().expect("workspace");
        write_skill(
            &workspace,
            "fact-check-lab",
            "Fact Check Lab",
            "联网信息检索与事实核验",
        );
        write_skill(&workspace, "writer", "Writer", "写作润色");

        let names = selected_agent_skill_names_for_turn(
            "帮我做最新事实核验和趋势调研",
            &[],
            Some(workspace.path()),
            Some(workspace.path()),
        );

        assert_eq!(names, vec!["fact-check-lab".to_string()]);
    }

    fn write_skill(workspace: &TempDir, name: &str, display_name: &str, description: &str) {
        let skill_dir = workspace.path().join(".agents/skills").join(name);
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!(
                "---\nname: {display_name}\ndescription: {description}\nmetadata:\n  lime_when_to_use: {description}\n---\n\n# Body\n\nFull body should not be rendered."
            ),
        )
        .expect("skill");
    }
}
