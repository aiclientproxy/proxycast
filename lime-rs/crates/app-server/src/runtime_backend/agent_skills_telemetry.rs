use crate::runtime::PluginTurnSnapshot;
use crate::{ExecutionRequest, RuntimeEvent};
use lime_skills::{AgentSkillBodyBudgetDecisionKind, AgentSkillSelectionEvaluation};
use serde_json::{json, Value};

pub(super) fn runtime_status_events_for_agent_skills(
    request: &ExecutionRequest,
    plugin_snapshots: &[PluginTurnSnapshot],
) -> Vec<RuntimeEvent> {
    let host_request = super::request_context::runtime_request_from_request(request);
    let workspace_scope =
        super::request_context::request_workspace_scope(request, host_request.as_ref());
    let metadata_values = super::skill_runtime_enable::request_metadata_values(request);
    let input_text = request.input.concat_text();
    let snapshot = super::agent_skills_context::build_agent_skill_snapshot_for_turn_with_plugins(
        workspace_scope.working_dir.as_deref(),
        workspace_scope.project_root.as_deref(),
        &metadata_values,
        plugin_snapshots,
    );
    let selections = super::agent_skills_context::selected_agent_skill_selections_with_plugins(
        &input_text,
        &metadata_values,
        &snapshot,
        plugin_snapshots,
    );
    let body_selections =
        super::agent_skills_context::selected_agent_skill_body_selections_for_prompt_with_plugins(
            &input_text,
            &metadata_values,
            &snapshot,
            plugin_snapshots,
        );
    let runtime_enable_sources =
        super::skill_runtime_enable::workspace_skill_runtime_enable_sources(request);
    let mut events = Vec::new();
    if !body_selections.is_empty() {
        let evaluations = super::agent_skills_context::selected_agent_skill_body_evaluations(
            &body_selections,
            &snapshot,
        );
        events.extend(skill_body_read_events(&evaluations));
    }
    if !runtime_enable_sources.is_empty() || !selections.is_empty() {
        events.push(skill_gate_decision_event(
            runtime_enable_sources
                .iter()
                .map(|source| source.skill_name.clone())
                .collect(),
            selections
                .iter()
                .map(|selection| selection.locator.name.clone())
                .collect(),
            !runtime_enable_sources.is_empty(),
        ));
    }
    events
}

fn skill_body_read_events(evaluations: &[AgentSkillSelectionEvaluation]) -> Vec<RuntimeEvent> {
    evaluations
        .iter()
        .map(|evaluation| match evaluation.decision {
            AgentSkillBodyBudgetDecisionKind::Allow => runtime_status_event(
                "context",
                "Skill instructions loaded",
                format!(
                    "Loaded SKILL.md for `{}`.",
                    evaluation.selection.locator.name
                ),
                json!({
                    "skillRuntime": {
                        "event": "skill_body_read",
                        "skillId": evaluation.skill_id,
                        "skillName": evaluation.selection.locator.name,
                        "trigger": evaluation.selection.trigger,
                        "reason": evaluation.reason,
                        "source": evaluation.source.map(|source| source.as_label()),
                        "authority": evaluation.authority.map(|authority| authority.as_label()),
                        "requiredCapabilities": evaluation.required_capabilities,
                        "missingCapabilities": evaluation.missing_capabilities,
                        "skillFilePath": evaluation.selection.locator.skill_file_path,
                        "status": "loaded",
                        "bodyTokenBudget": evaluation.body_budget,
                    }
                }),
            ),
            AgentSkillBodyBudgetDecisionKind::Omitted => runtime_status_event(
                "context",
                "Skill instructions omitted",
                format!(
                    "SKILL.md for `{}` exceeded the token budget.",
                    evaluation.selection.locator.name
                ),
                json!({
                    "skillRuntime": {
                        "event": "skill_body_read",
                        "skillId": evaluation.skill_id,
                        "skillName": evaluation.selection.locator.name,
                        "trigger": evaluation.selection.trigger,
                        "reason": evaluation.reason,
                        "source": evaluation.source.map(|source| source.as_label()),
                        "authority": evaluation.authority.map(|authority| authority.as_label()),
                        "requiredCapabilities": evaluation.required_capabilities,
                        "missingCapabilities": evaluation.missing_capabilities,
                        "skillFilePath": evaluation.selection.locator.skill_file_path,
                        "status": "omitted",
                        "bodyTokenBudget": evaluation.body_budget,
                    }
                }),
            ),
            AgentSkillBodyBudgetDecisionKind::Deny => runtime_status_event(
                "failed",
                "Skill instructions failed",
                format!(
                    "Failed to load SKILL.md for `{}`.",
                    evaluation.selection.locator.name
                ),
                json!({
                    "skillRuntime": {
                        "event": "skill_body_read",
                        "skillId": evaluation.skill_id,
                        "skillName": evaluation.selection.locator.name,
                        "trigger": evaluation.selection.trigger,
                        "reason": evaluation.reason,
                        "source": evaluation.source.map(|source| source.as_label()),
                        "authority": evaluation.authority.map(|authority| authority.as_label()),
                        "requiredCapabilities": evaluation.required_capabilities,
                        "missingCapabilities": evaluation.missing_capabilities,
                        "skillFilePath": evaluation.selection.locator.skill_file_path,
                        "status": "failed",
                        "bodyTokenBudget": evaluation.body_budget,
                        "error": evaluation.error,
                    }
                }),
            ),
        })
        .collect()
}

fn skill_gate_decision_event(
    source_allowlist: Vec<String>,
    selected_skills: Vec<String>,
    workspace_runtime_enable: bool,
) -> RuntimeEvent {
    let mode = if workspace_runtime_enable {
        "workspace_runtime_enable"
    } else {
        "selected_skills"
    };
    runtime_status_event(
        "permission_review",
        "Skill gate decision",
        if workspace_runtime_enable {
            "Workspace skill runtime enable applied.".to_string()
        } else {
            "Selected skills were allowed for this turn.".to_string()
        },
        json!({
            "skillRuntime": {
                "event": "skill_gate_decision",
                "mode": mode,
                "workspaceRuntimeEnable": workspace_runtime_enable,
                "sourceAllowlist": source_allowlist,
                "selectedSkills": selected_skills,
            }
        }),
    )
}

fn runtime_status_event(
    phase: &str,
    title: impl Into<String>,
    detail: impl Into<String>,
    metadata: Value,
) -> RuntimeEvent {
    RuntimeEvent::new(
        "runtime.status",
        json!({
            "status": {
                "phase": phase,
                "title": title.into(),
                "detail": detail.into(),
                "metadata": metadata,
            }
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn emits_body_read_and_gate_decision_for_explicit_skill_selection() {
        let workspace = TempDir::new().expect("workspace");
        write_agent_skill(&workspace, "writer");
        let request = super::super::tests::request_for_test(
            "请用 $writer 改写这段话",
            None,
            Some(json!({
                "workspaceRoot": workspace.path().to_string_lossy().to_string(),
            })),
        );

        let events = runtime_status_events_for_agent_skills(&request, &[]);

        assert_eq!(events.len(), 2);
        assert!(events
            .iter()
            .all(|event| event.event_type == "runtime.status"));
        assert_eq!(
            events[0].payload["status"]["metadata"]["skillRuntime"]["event"],
            json!("skill_body_read")
        );
        assert_eq!(
            events[0].payload["status"]["metadata"]["skillRuntime"]["skillName"],
            json!("writer")
        );
        assert_eq!(
            events[0].payload["status"]["metadata"]["skillRuntime"]["skillId"],
            json!("project:writer")
        );
        assert_eq!(
            events[0].payload["status"]["metadata"]["skillRuntime"]["source"],
            json!("project")
        );
        assert_eq!(
            events[0].payload["status"]["metadata"]["skillRuntime"]["authority"],
            json!("workspace")
        );
        assert_eq!(
            events[0].payload["status"]["metadata"]["skillRuntime"]["bodyTokenBudget"]["decision"],
            json!("allow")
        );
        assert_eq!(
            events[1].payload["status"]["metadata"]["skillRuntime"]["event"],
            json!("skill_gate_decision")
        );
        assert_eq!(
            events[1].payload["status"]["metadata"]["skillRuntime"]["mode"],
            json!("selected_skills")
        );
    }

    #[test]
    fn emits_catalog_bound_skill_telemetry_from_service_scene_metadata() {
        let workspace = TempDir::new().expect("workspace");
        write_agent_skill(&workspace, "writer");
        let request = super::super::tests::request_for_test(
            "帮我处理这段话",
            None,
            Some(json!({
                "workspaceRoot": workspace.path().to_string_lossy().to_string(),
                "harness": {
                    "service_scene_launch": {
                        "service_scene_run": {
                            "skill_key": "local:writer"
                        }
                    }
                }
            })),
        );

        let events = runtime_status_events_for_agent_skills(&request, &[]);

        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0].payload["status"]["metadata"]["skillRuntime"]["event"],
            json!("skill_body_read")
        );
        assert_eq!(
            events[0].payload["status"]["metadata"]["skillRuntime"]["trigger"],
            json!("catalog_binding")
        );
        assert_eq!(
            events[1].payload["status"]["metadata"]["skillRuntime"]["selectedSkills"],
            json!(["writer"])
        );
    }

    #[test]
    fn image_command_intent_emits_no_agent_skill_telemetry() {
        let workspace = TempDir::new().expect("workspace");
        write_agent_skill(&workspace, "image_generate");
        let request = super::super::tests::request_for_test(
            "$image_generate 画一张广州夏天的图",
            None,
            Some(json!({
                "workspaceRoot": workspace.path().to_string_lossy().to_string(),
                "harness": {
                    "image_command_intent": {
                        "kind": "image_task",
                        "image_task": {
                            "prompt": "广州夏天",
                            "modality_contract_key": "image_generation"
                        }
                    }
                }
            })),
        );

        let events = runtime_status_events_for_agent_skills(&request, &[]);

        assert!(events.is_empty());
    }

    #[test]
    fn retired_image_skill_launch_emits_no_agent_skill_telemetry() {
        let workspace = TempDir::new().expect("workspace");
        write_agent_skill(&workspace, "image_generate");
        let request = super::super::tests::request_for_test(
            "$image_generate 画一张广州夏天的图",
            None,
            Some(json!({
                "workspaceRoot": workspace.path().to_string_lossy().to_string(),
                "harness": {
                    "image_skill_launch": {
                        "skill_name": "image_generate",
                        "kind": "image_task",
                        "image_task": {
                            "prompt": "广州夏天",
                            "modality_contract_key": "image_generation"
                        }
                    }
                }
            })),
        );

        let events = runtime_status_events_for_agent_skills(&request, &[]);

        assert!(events.is_empty());
    }

    #[test]
    fn emits_expert_bound_skill_telemetry_from_expert_metadata() {
        let workspace = TempDir::new().expect("workspace");
        write_agent_skill(&workspace, "writer");
        let request = super::super::tests::request_for_test(
            "帮我处理这段话",
            None,
            Some(json!({
                "workspaceRoot": workspace.path().to_string_lossy().to_string(),
                "harness": {
                    "expert": {
                        "skill_refs": ["skill:writer"]
                    }
                }
            })),
        );

        let events = runtime_status_events_for_agent_skills(&request, &[]);

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].payload["status"]["metadata"]["skillRuntime"]["event"],
            json!("skill_gate_decision")
        );
        assert_eq!(
            events[0].payload["status"]["metadata"]["skillRuntime"]["selectedSkills"],
            json!(["writer"])
        );
    }

    #[test]
    fn unknown_service_scene_metadata_emits_no_skill_telemetry() {
        let workspace = TempDir::new().expect("workspace");
        write_agent_skill(&workspace, "writer");
        let request = super::super::tests::request_for_test(
            "帮我处理这段话",
            None,
            Some(json!({
                "workspaceRoot": workspace.path().to_string_lossy().to_string(),
                "harness": {
                    "service_scene_launch": {
                        "service_scene_run": {
                            "skill_key": "missing"
                        }
                    }
                }
            })),
        );

        let events = runtime_status_events_for_agent_skills(&request, &[]);

        assert!(events.is_empty());
    }

    fn write_agent_skill(workspace: &TempDir, name: &str) {
        let skill_dir = workspace.path().join(".agents").join("skills").join(name);
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: Test skill.\n---\n\n# {name}\n"),
        )
        .expect("skill");
    }
}
