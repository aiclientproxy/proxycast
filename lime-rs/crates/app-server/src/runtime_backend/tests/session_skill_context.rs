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
fn session_config_loads_enabled_plugin_v2_skills_and_freezes_identity() {
    let workspace = TempDir::new().expect("workspace");
    let plugin_root = workspace.path().join("plugins/browser");
    let skill_dir = plugin_root.join("skills/control-browser");
    std::fs::create_dir_all(&skill_dir).expect("plugin skill dir");
    std::fs::write(
        skill_dir.join("SKILL.md"),
        r#"---
name: Control Browser
description: Inspect pages through the managed browser.
---

# Control Browser

Use the managed browser session.
"#,
    )
    .expect("plugin skill file");
    let mut request = request_for_test("检查当前页面", None, None);
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());
    let host_request = runtime_request_from_request(&request);
    let scope = session_scope_from_request(&request).expect("session scope");
    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let policy = request_tool_policy_from_request(host_request.as_ref());
    let activations = vec![json!({
        "schemaVersion": "plugin-activation/v2",
        "pluginId": "browser",
        "version": "1.0.0",
        "contentDigest": "sha256:fixture",
        "marketplaceId": "openai-bundled",
        "packageSourceUri": plugin_root.to_string_lossy(),
        "runtimeCapabilities": {
            "schemaVersion": "plugin-runtime-capabilities/v2",
            "pluginId": "browser",
            "version": "1.0.0",
            "packageSourceUri": plugin_root.to_string_lossy(),
            "skills": [{
                "id": "control-browser",
                "title": "Control Browser",
                "path": "./skills/",
                "activation": "available",
                "required": false,
                "promptInjectionPolicy": {
                    "mode": "available",
                    "source": "plugin-v2-installed"
                }
            }],
            "mcpBindings": []
        }
    })];

    let config = session_config_from_request_with_plugin_activations(
        &request,
        host_request.as_ref(),
        &scope,
        &selection,
        &policy,
        None,
        &activations,
    );

    let prompt = config.system_prompt.expect("system prompt");
    assert!(prompt.contains("<plugin_runtime_capabilities>"));
    assert!(prompt.contains("plugin_id: browser"));
    let turn_context = config.turn_context.expect("turn context");
    assert_eq!(
        turn_context.metadata["plugin_activation"]["pluginId"],
        "browser"
    );
    assert_eq!(
        turn_context.metadata["plugin_activations"],
        Value::Array(activations)
    );
    let snapshot = turn_context
        .metadata
        .get(lime_skills::SKILL_SNAPSHOT_TURN_METADATA_KEY)
        .and_then(|value| {
            serde_json::from_value::<lime_skills::AgentSkillSnapshot>(value.clone()).ok()
        })
        .expect("skill snapshot");
    assert!(snapshot
        .skills
        .iter()
        .any(|skill| skill.name.eq_ignore_ascii_case("control-browser")));
}

#[test]
fn session_config_appends_plugin_activation_metadata_to_system_prompt() {
    let mut request = request_for_test(
        "@创作工作台 写一篇公众号文章",
        None,
        Some(json!({
            "harness": {
                "plugin_activation": {
                    "source": "plugin_explicit_mention",
                    "trigger": "@创作工作台",
                    "body": "写一篇公众号文章",
                    "session_id": "session-1",
                    "plugin_id": "creator-workbench",
                    "active_entry_key": "creator",
                    "selected_object_ref": {
                        "plugin_id": "creator-workbench",
                        "object_kind": "articleDraft",
                        "object_id": "pending"
                    },
                    "opened_tabs": ["articleWorkspace"],
                    "context_source": "user"
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
    assert!(prompt.contains("<plugin_activation_context>"));
    assert!(prompt.contains("source: plugin_explicit_mention"));
    assert!(prompt.contains("trigger: @创作工作台"));
    assert!(prompt.contains("plugin_id: creator-workbench"));
    assert!(prompt.contains("active_entry_key: creator"));
    assert!(prompt.contains("object_kind: articleDraft"));
    assert!(prompt.contains("opened_tabs: articleWorkspace"));
    assert!(prompt.contains("Do not infer or switch plugins from natural language"));
    assert!(!prompt.contains("allow_model_skills"));
}

#[test]
fn session_config_appends_plugin_runtime_capabilities_to_system_prompt() {
    let workspace = TempDir::new().expect("workspace");
    let skill_dir = workspace.path().join(".agents/skills/article-writing");
    std::fs::create_dir_all(&skill_dir).expect("skill dir");
    std::fs::write(
        skill_dir.join("SKILL.md"),
        r#"---
name: Article Writing
description: Draft article copy.
---

# Article Writing

Use article workflow rules.
"#,
    )
    .expect("skill file");
    let mut request = request_for_test(
        "@创作工作台 写一篇公众号文章",
        None,
        Some(json!({
            "harness": {
                "workspace_root": workspace.path().to_string_lossy(),
                "cwd": workspace.path().to_string_lossy(),
                "plugin_activation": {
                    "source": "plugin_explicit_mention",
                    "trigger": "@创作工作台",
                    "body": "写一篇公众号文章",
                    "session_id": "session-1",
                    "plugin_id": "content-factory-app",
                    "workflow_key": "content-article",
                    "runtime_capabilities": {
                        "pluginId": "content-factory-app",
                        "skills": [
                            {
                                "id": "article-writing",
                                "title": "Article Writing",
                                "required": true,
                                "promptInjectionPolicy": {
                                    "mode": "workflow_scoped",
                                    "source": "runtimeCapabilities.skills"
                                }
                            }
                        ],
                        "mcpBindings": [
                            {
                                "serverId": "browser",
                                "toolKey": "browser/search",
                                "provider": "mcp",
                                "required": true
                            }
                        ],
                        "workflowBindings": [
                            {
                                "workflowKey": "content-article",
                                "skillIds": ["article-writing"]
                            }
                        ]
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
    assert!(prompt.contains("<plugin_runtime_capabilities>"));
    assert!(prompt.contains("plugin_id: content-factory-app"));
    assert!(prompt.contains("id=article-writing"));
    assert!(prompt.contains("server_id=browser"));
    assert!(!prompt.contains("<selected_skill_instructions>"));
    assert!(!prompt.contains("Use article workflow rules."));
    let turn_context = config.turn_context.expect("turn context");
    let snapshot = turn_context
        .metadata
        .get(lime_skills::SKILL_SNAPSHOT_TURN_METADATA_KEY)
        .and_then(|value| {
            serde_json::from_value::<lime_skills::AgentSkillSnapshot>(value.clone()).ok()
        })
        .expect("skill snapshot");
    assert!(snapshot
        .skills
        .iter()
        .any(|skill| skill.name.eq_ignore_ascii_case("article-writing")));
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
