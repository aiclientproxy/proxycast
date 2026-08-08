use super::*;

#[test]
fn session_config_forwards_skill_snapshot_without_system_body() {
    let workspace = TempDir::new().expect("workspace");
    let skill_dir = workspace.path().join(".agents/skills/writer");
    std::fs::create_dir_all(&skill_dir).expect("skill dir");
    std::fs::write(
        skill_dir.join("SKILL.md"),
        r#"---
name: Writer
description: Write clearly.
---

# Writer Skill

Use concise language.
"#,
    )
    .expect("skill file");
    let mut request = request_for_test(
        "请用 $writer 改写这段话",
        None,
        Some(json!({
            "harness": {
                "workspace_root": workspace.path().to_string_lossy(),
                "cwd": workspace.path().to_string_lossy()
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());
    let host_request = runtime_request_from_request(&request);
    let scope = session_scope_from_request(&request).expect("session scope");
    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let policy = request_tool_policy_from_request(host_request.as_ref());

    let config = session_config_from_request(
        &request,
        host_request.as_ref(),
        &scope,
        &selection,
        &policy,
        None,
    );

    let prompt = config.system_prompt.expect("system prompt");
    assert!(prompt.contains("## 可用 Agent Skills"));
    assert!(!prompt.contains("<selected_skill_instructions>"));
    assert!(!prompt.contains("# Writer Skill"));
    assert!(!prompt.contains("Use concise language."));
    assert!(!prompt.contains("allow_model_skills"));
    let turn_context = config.turn_context.expect("turn context");
    let snapshot = turn_context
        .metadata
        .get(lime_skills::SKILL_SNAPSHOT_TURN_METADATA_KEY)
        .and_then(|value| {
            serde_json::from_value::<lime_skills::AgentSkillSnapshot>(value.clone()).ok()
        })
        .expect("skill snapshot");
    assert!(snapshot.skills.iter().any(|skill| skill.name == "writer"));
}

#[test]
fn session_config_keeps_selected_skill_allowed_tools_inside_agent_skill_runtime() {
    let workspace = TempDir::new().expect("workspace");
    let skill_dir = workspace.path().join(".agents/skills/writer");
    std::fs::create_dir_all(&skill_dir).expect("skill dir");
    std::fs::write(
        skill_dir.join("SKILL.md"),
        r#"---
name: Writer
description: Write clearly.
allowed_tools: Read Write
---

# Writer Skill

Use concise language.
"#,
    )
    .expect("skill file");
    let mut request = request_for_test(
        "帮我处理这段话",
        None,
        Some(json!({
            "harness": {
                "workspace_root": workspace.path().to_string_lossy(),
                "cwd": workspace.path().to_string_lossy(),
                "service_scene_launch": {
                    "service_scene_run": {
                        "skill_key": "local:writer"
                    }
                }
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());
    let host_request = runtime_request_from_request(&request);
    let scope = session_scope_from_request(&request).expect("session scope");
    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let policy = request_tool_policy_from_request(host_request.as_ref());

    let config = session_config_from_request(
        &request,
        host_request.as_ref(),
        &scope,
        &selection,
        &policy,
        None,
    );

    let prompt = config.system_prompt.expect("system prompt");
    assert!(prompt.contains("## 可用 Agent Skills"));
    assert!(!prompt.contains("Use concise language."));
    let turn_context = config.turn_context.expect("turn context");
    assert!(
        turn_context.metadata.get("tool_scope").is_none(),
        "App Server must not duplicate Agent Skill allowed-tools as a main-turn scope"
    );
    assert!(turn_context
        .metadata
        .contains_key(lime_skills::SKILL_SNAPSHOT_TURN_METADATA_KEY));
}

#[test]
fn session_config_does_not_project_expert_runtime_enable_allowed_tools_to_turn_scope() {
    let workspace = TempDir::new().expect("workspace");
    let skill_dir = workspace.path().join(".agents/skills/capability-report");
    std::fs::create_dir_all(&skill_dir).expect("skill dir");
    std::fs::write(
        skill_dir.join("SKILL.md"),
        r#"---
name: Capability Report
description: Review local capability.
allowed_tools: Read
---

# Capability Report

Use evidence.
"#,
    )
    .expect("skill file");
    let workspace_root = workspace.path().to_string_lossy().to_string();
    let mut request = request_for_test(
        "请先搜索 capability-report 后再执行",
        None,
        Some(json!({
            "harness": {
                "workspace_root": workspace_root,
                "cwd": workspace.path().to_string_lossy(),
                "expert": {
                    "skill_refs": ["skill:capability-report"]
                },
                "workspace_skill_runtime_enable": {
                    "source": "manual_session_enable",
                    "approval": "manual",
                    "workspace_root": workspace.path().to_string_lossy(),
                    "bindings": [
                        {
                            "directory": "capability-report",
                            "registered_skill_directory": skill_dir.to_string_lossy(),
                            "source_draft_id": "capdraft-1",
                            "source_verification_report_id": "capver-1"
                        }
                    ]
                }
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());
    let host_request = runtime_request_from_request(&request);
    let scope = session_scope_from_request(&request).expect("session scope");
    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let policy = request_tool_policy_from_request(host_request.as_ref());

    let config = session_config_from_request(
        &request,
        host_request.as_ref(),
        &scope,
        &selection,
        &policy,
        None,
    );

    let prompt = config.system_prompt.expect("system prompt");
    assert!(prompt.contains("<expert_skill_refs>"));
    assert!(!prompt.contains("<selected_skill_instructions>"));
    let turn_context = config.turn_context.expect("turn context");
    assert!(!turn_context.metadata.contains_key("tool_scope"));
}

#[test]
fn session_config_does_not_project_tool_scope_for_unknown_skill_metadata() {
    let workspace = TempDir::new().expect("workspace");
    let skill_dir = workspace.path().join(".agents/skills/writer");
    std::fs::create_dir_all(&skill_dir).expect("skill dir");
    std::fs::write(
        skill_dir.join("SKILL.md"),
        r#"---
name: Writer
description: Write clearly.
allowed_tools: Read Write
---

# Writer Skill

Use concise language.
"#,
    )
    .expect("skill file");
    let mut request = request_for_test(
        "帮我处理这段话",
        None,
        Some(json!({
            "harness": {
                "workspace_root": workspace.path().to_string_lossy(),
                "cwd": workspace.path().to_string_lossy(),
                "service_scene_launch": {
                    "service_scene_run": {
                        "skill_key": "missing"
                    }
                }
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());
    let host_request = runtime_request_from_request(&request);
    let scope = session_scope_from_request(&request).expect("session scope");
    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let policy = request_tool_policy_from_request(host_request.as_ref());

    let config = session_config_from_request(
        &request,
        host_request.as_ref(),
        &scope,
        &selection,
        &policy,
        None,
    );

    let turn_context = config.turn_context.expect("turn context");
    assert!(!turn_context.metadata.contains_key("tool_scope"));
}

#[test]
fn session_config_does_not_append_skill_body_without_explicit_selection() {
    let workspace = TempDir::new().expect("workspace");
    let skill_dir = workspace.path().join(".agents/skills/writer");
    std::fs::create_dir_all(&skill_dir).expect("skill dir");
    std::fs::write(
        skill_dir.join("SKILL.md"),
        r#"---
name: Writer
description: Write clearly.
---

# Writer Skill

Use concise language.
"#,
    )
    .expect("skill file");
    let mut request = request_for_test(
        "帮我改写这段话",
        None,
        Some(json!({
            "harness": {
                "workspace_root": workspace.path().to_string_lossy(),
                "cwd": workspace.path().to_string_lossy()
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());
    let host_request = runtime_request_from_request(&request);
    let scope = session_scope_from_request(&request).expect("session scope");
    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let policy = request_tool_policy_from_request(host_request.as_ref());

    let config = session_config_from_request(
        &request,
        host_request.as_ref(),
        &scope,
        &selection,
        &policy,
        None,
    );

    let prompt = config.system_prompt.expect("system prompt");
    assert!(prompt.contains("## 可用 Agent Skills"));
    assert!(prompt.contains("`writer`"));
    assert!(!prompt.contains("<selected_skill_instructions>"));
    assert!(!prompt.contains("# Writer Skill"));
    assert!(!prompt.contains("Use concise language."));
}
