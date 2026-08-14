use app_server_protocol::{
    SkillAuthority, SkillDetail, SkillDetailDependencies, SkillDetailInterface, SkillDetailScope,
    SkillDetailToolDependency, SkillLocator, SkillPolicy, SkillReadParams, SkillReadResponse,
    SkillSource, SkillSummary, SkillWorkflowStep,
};
use lime_skills::{
    build_agent_skill_snapshot, load_skill_from_file, AgentSkillAuthority, AgentSkillMetadata,
    AgentSkillScope, AgentSkillSnapshot, AgentSkillSource, LoadedSkillDefinition,
};
pub(crate) fn read_skill(params: SkillReadParams) -> Result<SkillReadResponse, String> {
    let snapshot = build_agent_skill_snapshot();
    read_skill_from_snapshot(&snapshot, params)
}

fn read_skill_from_snapshot(
    snapshot: &AgentSkillSnapshot,
    params: SkillReadParams,
) -> Result<SkillReadResponse, String> {
    let skill_id = params.skill_id.trim();
    if skill_id.is_empty() {
        return Err("Skill id is required".to_string());
    }
    let metadata = snapshot
        .skills
        .iter()
        .find(|metadata| metadata.skill_id == skill_id)
        .ok_or_else(|| format!("Skill 不存在: {skill_id}"))?;
    if !metadata.enabled {
        return Err(format!("skill '{skill_id}' disabled model invocation"));
    }
    let skill = load_skill_from_file(&metadata.name, &metadata.skill_file_path)?;
    if !skill.standard_compliance.validation_errors.is_empty() {
        return Err(format!(
            "skill '{}' failed standard validation: {}",
            skill_id,
            skill.standard_compliance.validation_errors.join("; ")
        ));
    }
    if skill.disable_model_invocation {
        return Err(format!("skill '{skill_id}' disabled model invocation"));
    }
    Ok(SkillReadResponse {
        skill: skill_to_detail(metadata, skill),
    })
}

fn agent_skill_to_summary(skill: &AgentSkillMetadata) -> SkillSummary {
    SkillSummary {
        skill_id: skill.skill_id.clone(),
        name: skill.name.clone(),
        description: skill.description.clone(),
        scope: skill_scope(skill.scope),
        source: skill_source(skill.source),
        authority: skill_authority(skill.authority),
        enabled: skill.enabled,
        interface: SkillDetailInterface {
            display_name: skill.interface.display_name.clone(),
            execution_mode: skill.interface.execution_mode.clone(),
            provider: skill.interface.provider.clone(),
            model: skill.interface.model.clone(),
            argument_hint: skill.interface.argument_hint.clone(),
        },
        dependencies: SkillDetailDependencies {
            tools: skill
                .dependencies
                .tools
                .iter()
                .map(|dependency| SkillDetailToolDependency {
                    dependency_type: dependency.dependency_type.clone(),
                    value: dependency.value.clone(),
                    required: dependency.required,
                })
                .collect(),
        },
        policy: SkillPolicy {
            allow_implicit_invocation: skill.policy.allow_implicit_invocation,
            when_to_use: skill.policy.when_to_use.clone(),
        },
        capabilities: skill.capabilities.clone(),
        locator: SkillLocator {
            directory: skill.directory.to_string_lossy().into_owned(),
            skill_file_path: skill.skill_file_path.to_string_lossy().into_owned(),
        },
    }
}

fn skill_to_detail(metadata: &AgentSkillMetadata, skill: LoadedSkillDefinition) -> SkillDetail {
    SkillDetail {
        metadata: agent_skill_to_summary(metadata),
        markdown_content: skill.markdown_content,
        workflow_steps: skill
            .workflow_steps
            .into_iter()
            .map(|step| SkillWorkflowStep {
                id: step.id,
                name: step.name,
                dependencies: Vec::new(),
            })
            .collect(),
    }
}

fn skill_scope(scope: AgentSkillScope) -> SkillDetailScope {
    match scope {
        AgentSkillScope::Project => SkillDetailScope::Project,
        AgentSkillScope::User => SkillDetailScope::User,
        AgentSkillScope::App => SkillDetailScope::App,
        AgentSkillScope::Orchestrator => SkillDetailScope::App,
        AgentSkillScope::Other => SkillDetailScope::Other,
    }
}

fn skill_source(source: AgentSkillSource) -> SkillSource {
    match source {
        AgentSkillSource::Project => SkillSource::Project,
        AgentSkillSource::User => SkillSource::User,
        AgentSkillSource::App => SkillSource::App,
        AgentSkillSource::Orchestrator => SkillSource::Orchestrator,
        AgentSkillSource::Other => SkillSource::Other,
    }
}

fn skill_authority(authority: AgentSkillAuthority) -> SkillAuthority {
    match authority {
        AgentSkillAuthority::Workspace => SkillAuthority::Workspace,
        AgentSkillAuthority::User => SkillAuthority::User,
        AgentSkillAuthority::Application => SkillAuthority::Application,
        AgentSkillAuthority::Orchestrator => SkillAuthority::Orchestrator,
        AgentSkillAuthority::External => SkillAuthority::External,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lime_skills::{build_agent_skill_snapshot_from_roots, AgentSkillRoot, AgentSkillScope};
    use tempfile::TempDir;

    #[test]
    fn read_skill_uses_exact_stable_id_locator_when_names_overlap() {
        let temp = TempDir::new().expect("temp dir");
        let project_root = temp.path().join("project-skills");
        let user_root = temp.path().join("user-skills");
        write_skill(&project_root, "writer", "Project body");
        write_skill(&user_root, "writer", "User body");
        let snapshot = build_agent_skill_snapshot_from_roots([
            AgentSkillRoot {
                path: project_root,
                scope: AgentSkillScope::Project,
            },
            AgentSkillRoot {
                path: user_root,
                scope: AgentSkillScope::User,
            },
        ]);

        let response = read_skill_from_snapshot(
            &snapshot,
            SkillReadParams {
                skill_id: "user:writer".to_string(),
            },
        )
        .expect("read user-scoped writer");

        assert_eq!(response.skill.metadata.skill_id, "user:writer");
        assert!(response.skill.markdown_content.contains("User body"));
        assert!(!response.skill.markdown_content.contains("Project body"));
    }

    #[test]
    fn read_skill_rejects_name_only_reference() {
        let temp = TempDir::new().expect("temp dir");
        let root = temp.path().join("project-skills");
        write_skill(&root, "writer", "Project body");
        let snapshot = build_agent_skill_snapshot_from_roots([AgentSkillRoot {
            path: root,
            scope: AgentSkillScope::Project,
        }]);

        let error = read_skill_from_snapshot(
            &snapshot,
            SkillReadParams {
                skill_id: "writer".to_string(),
            },
        )
        .expect_err("name-only reference must fail closed");

        assert_eq!(error, "Skill 不存在: writer");
    }

    #[test]
    fn read_uses_first_provider_precedence_for_same_stable_id() {
        let temp = TempDir::new().expect("temp dir");
        let first_root = temp.path().join("first-project-provider");
        let second_root = temp.path().join("second-project-provider");
        write_skill(&first_root, "writer", "First provider body");
        write_skill(&second_root, "writer", "Second provider body");
        let snapshot = build_agent_skill_snapshot_from_roots([
            AgentSkillRoot {
                path: first_root,
                scope: AgentSkillScope::Project,
            },
            AgentSkillRoot {
                path: second_root,
                scope: AgentSkillScope::Project,
            },
        ]);

        let read = read_skill_from_snapshot(
            &snapshot,
            SkillReadParams {
                skill_id: "project:writer".to_string(),
            },
        )
        .expect("read winning provider");

        assert!(read
            .skill
            .metadata
            .locator
            .skill_file_path
            .contains("first-project-provider"));
        assert!(read.skill.markdown_content.contains("First provider body"));
        assert!(!read.skill.markdown_content.contains("Second provider body"));
    }

    #[test]
    fn read_skill_rejects_blank_missing_and_disabled_ids() {
        let temp = TempDir::new().expect("temp dir");
        let root = temp.path().join("project-skills");
        write_disabled_skill(&root, "disabled");
        let snapshot = build_agent_skill_snapshot_from_roots([AgentSkillRoot {
            path: root,
            scope: AgentSkillScope::Project,
        }]);

        let blank = read_skill_from_snapshot(
            &snapshot,
            SkillReadParams {
                skill_id: "   ".to_string(),
            },
        )
        .expect_err("blank id must fail closed");
        let missing = read_skill_from_snapshot(
            &snapshot,
            SkillReadParams {
                skill_id: "project:missing".to_string(),
            },
        )
        .expect_err("missing id must fail closed");
        let disabled = read_skill_from_snapshot(
            &snapshot,
            SkillReadParams {
                skill_id: "project:disabled".to_string(),
            },
        )
        .expect_err("disabled id must fail closed");

        assert_eq!(blank, "Skill id is required");
        assert_eq!(missing, "Skill 不存在: project:missing");
        assert_eq!(
            disabled,
            "skill 'project:disabled' disabled model invocation"
        );
    }

    fn write_skill(root: &std::path::Path, name: &str, body: &str) {
        let skill_dir = root.join(name);
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: Test skill.\n---\n\n# {body}\n"),
        )
        .expect("skill file");
    }

    fn write_disabled_skill(root: &std::path::Path, name: &str) {
        let skill_dir = root.join(name);
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!(
                "---\nname: {name}\ndescription: Disabled skill.\ndisable-model-invocation: true\n---\n\n# Disabled\n"
            ),
        )
        .expect("skill file");
    }
}
