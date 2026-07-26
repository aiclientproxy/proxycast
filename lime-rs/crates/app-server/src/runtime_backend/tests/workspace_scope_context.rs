use super::*;
use app_server_protocol::{RuntimeRequest, RuntimeSearchMode};

#[test]
fn request_working_dir_uses_typed_runtime_request_absolute_directory() {
    let workspace = TempDir::new().expect("create workspace");
    let request = request_for_test(
        "hello",
        Some(RuntimeRequest {
            working_dir: Some(workspace.path().to_string_lossy().into_owned()),
            ..RuntimeRequest::default()
        }),
        None,
    );
    let host_request = runtime_request_from_request(&request);

    let working_dir = request_workspace_scope(&request, host_request.as_ref())
        .working_dir
        .expect("working dir");

    assert_eq!(working_dir, workspace.path());
}

#[test]
fn request_working_dir_rejects_relative_directory() {
    let request = request_for_test(
        "hello",
        Some(RuntimeRequest {
            working_dir: Some("relative-workspace".to_string()),
            ..RuntimeRequest::default()
        }),
        None,
    );
    let host_request = runtime_request_from_request(&request);

    assert!(request_workspace_scope(&request, host_request.as_ref())
        .working_dir
        .is_none());
}

#[test]
fn request_workspace_scope_keeps_project_root_and_working_dir_distinct() {
    let workspace = TempDir::new().expect("create workspace");
    let repo = workspace.path().join("repo");
    let nested = repo.join("apps").join("writer");
    std::fs::create_dir_all(&nested).expect("create nested");
    let request = request_for_test(
        "hello",
        Some(RuntimeRequest {
            project_root: Some(repo.to_string_lossy().into_owned()),
            working_dir: Some(nested.to_string_lossy().into_owned()),
            ..RuntimeRequest::default()
        }),
        None,
    );
    let host_request = runtime_request_from_request(&request);

    let scope = request_workspace_scope(&request, host_request.as_ref());

    assert_eq!(scope.working_dir.as_deref(), Some(nested.as_path()));
    assert_eq!(scope.project_root.as_deref(), Some(repo.as_path()));
}

#[test]
fn turn_context_projects_typed_world_state_without_unowned_sections() {
    let workspace = TempDir::new().expect("create workspace");
    let repo = workspace.path().join("repo");
    let nested = repo.join("apps").join("writer");
    std::fs::create_dir_all(&nested).expect("create nested workspace");
    let request = request_for_test(
        "plan the change",
        Some(RuntimeRequest {
            provider_preference: Some("openai".to_string()),
            model_preference: Some("gpt-4.1".to_string()),
            collaboration_mode: Some(agent_protocol::CollaborationMode {
                mode: agent_protocol::ModeKind::Plan,
                settings: agent_protocol::CollaborationModeSettings {
                    model: "gpt-4.1".to_string(),
                    reasoning_effort: Some("high".to_string()),
                    developer_instructions: None,
                },
            }),
            reasoning_effort: Some("high".to_string()),
            approval_policy: Some("on-request".to_string()),
            sandbox_policy: Some("workspace-write".to_string()),
            project_root: Some(repo.to_string_lossy().into_owned()),
            working_dir: Some(nested.to_string_lossy().into_owned()),
            web_search: Some(true),
            search_mode: Some(RuntimeSearchMode::Required),
            ..RuntimeRequest::default()
        }),
        None,
    );
    let host_request = runtime_request_from_request(&request).expect("host request");
    let scope = session_scope_from_request(&request).expect("session scope");
    let selection = selection_from_explicit_preferences(&request).expect("model selection");

    let turn_context =
        turn_context_from_request(&request, Some(&host_request), &scope, &selection, None)
            .expect("turn context");
    let world_state = turn_context
        .metadata
        .get(agent_protocol::world_state::WORLD_STATE_TURN_METADATA_KEY)
        .expect("world state metadata");

    assert_eq!(
        world_state
            .pointer("/environment/cwd")
            .and_then(Value::as_str),
        Some(nested.to_string_lossy().as_ref()),
    );
    assert_eq!(
        world_state
            .pointer("/environment/projectRoot")
            .and_then(Value::as_str),
        Some(repo.to_string_lossy().as_ref()),
    );
    assert_eq!(world_state["environment"]["workspaceId"], "workspace-main");
    assert_eq!(world_state["environment"]["threadId"], "thread-1");
    assert_eq!(world_state["environment"]["turnId"], "turn-1");
    assert_eq!(world_state["environment"]["provider"], "openai");
    assert_eq!(world_state["environment"]["model"], "gpt-4.1");
    assert_eq!(world_state["environment"]["reasoningEffort"], "high");
    assert_eq!(world_state["permissions"]["approvalPolicy"], "on-request");
    assert_eq!(
        world_state["permissions"]["sandboxPolicy"],
        "workspace-write"
    );
    assert_eq!(world_state["permissions"]["webSearch"], true);
    assert_eq!(world_state["collaboration"]["mode"], "plan");
    assert_eq!(world_state["collaboration"]["source"], "runtime_request");
    assert_eq!(world_state["multiAgent"], "explicitRequestOnly");
    assert_eq!(world_state["source"], "app_server_world_state");
    assert!(world_state.get("instructionSections").is_none());
}

#[test]
fn turn_context_derives_proactive_multi_agent_mode_from_ultra_effort() {
    let workspace = TempDir::new().expect("create workspace");
    let request = request_for_test(
        "delegate independent work",
        Some(RuntimeRequest {
            provider_preference: Some("openai".to_string()),
            model_preference: Some("gpt-5.4".to_string()),
            reasoning_effort: Some("ultra".to_string()),
            working_dir: Some(workspace.path().to_string_lossy().into_owned()),
            ..RuntimeRequest::default()
        }),
        None,
    );
    let host_request = runtime_request_from_request(&request).expect("host request");
    let scope = session_scope_from_request(&request).expect("session scope");
    let selection = selection_from_explicit_preferences(&request).expect("model selection");

    let turn_context =
        turn_context_from_request(&request, Some(&host_request), &scope, &selection, None)
            .expect("turn context");
    let world_state = turn_context
        .metadata
        .get(agent_protocol::world_state::WORLD_STATE_TURN_METADATA_KEY)
        .expect("world state metadata");

    assert_eq!(world_state["multiAgent"], "proactive");
}

#[test]
fn request_workspace_scope_falls_back_to_typed_project_root_when_working_dir_missing() {
    let workspace = TempDir::new().expect("create workspace");
    let request = request_for_test(
        "hello",
        Some(RuntimeRequest {
            workspace_root: Some(workspace.path().to_string_lossy().into_owned()),
            ..RuntimeRequest::default()
        }),
        None,
    );
    let host_request = runtime_request_from_request(&request);
    let scope = request_workspace_scope(&request, host_request.as_ref());

    assert_eq!(scope.working_dir.as_deref(), Some(workspace.path()));
    assert_eq!(scope.project_root.as_deref(), Some(workspace.path()));
}

#[test]
fn session_config_merges_turn_prompt_runtime_agents_and_tool_policy() {
    let workspace = TempDir::new().expect("create workspace");
    let runtime_agents_path = workspace.path().join(".lime").join("AGENTS.md");
    std::fs::create_dir_all(runtime_agents_path.parent().expect("runtime agents parent"))
        .expect("create runtime agents parent");
    std::fs::write(&runtime_agents_path, "- 工作区动态指令").expect("write runtime agents");
    let request = request_for_test(
        "需要联网核实最新信息",
        Some(RuntimeRequest {
            system_prompt: Some("请求级系统提示".to_string()),
            working_dir: Some(workspace.path().to_string_lossy().into_owned()),
            web_search: Some(true),
            search_mode: Some(RuntimeSearchMode::Required),
            ..RuntimeRequest::default()
        }),
        None,
    );
    let host_request = runtime_request_from_request(&request);
    let scope = session_scope_from_request(&request).expect("scope");
    let selection = RuntimeModelSelection {
        provider: "openai".to_string(),
        model: "gpt-4.1".to_string(),
        source: "test",
        reasoning_effort: Some("high".to_string()),
    };
    let policy = request_tool_policy_from_request(host_request.as_ref());

    let config = session_config_from_request(
        &request,
        host_request.as_ref(),
        &scope,
        &selection,
        &policy,
        None,
    );
    let system_prompt = config.system_prompt.expect("system prompt");

    assert!(system_prompt.contains("请求级系统提示"));
    assert!(system_prompt.contains("【Lime Runtime AGENTS 指令】"));
    assert!(system_prompt.contains("工作区动态指令"));
    assert!(system_prompt.contains("【请求级工具策略】"));
}

#[test]
fn session_config_merges_hierarchical_runtime_agents_layers() {
    let workspace = TempDir::new().expect("create workspace");
    let repo = workspace.path().join("repo");
    let nested = repo.join("apps").join("writer");
    std::fs::create_dir_all(nested.join(".lime")).expect("create nested runtime agents dir");
    std::fs::create_dir_all(repo.join(".lime")).expect("create root runtime agents dir");
    std::fs::write(repo.join(".git"), "").expect("write project marker");
    std::fs::write(repo.join(".lime").join("AGENTS.md"), "- 根共享规则")
        .expect("write root shared runtime agents");
    std::fs::write(nested.join(".lime").join("AGENTS.md"), "- 子目录共享规则")
        .expect("write nested shared runtime agents");
    let request = request_for_test(
        "请按项目规则处理",
        Some(RuntimeRequest {
            system_prompt: Some("请求级系统提示".to_string()),
            working_dir: Some(nested.to_string_lossy().into_owned()),
            ..RuntimeRequest::default()
        }),
        None,
    );
    let host_request = runtime_request_from_request(&request);
    let scope = session_scope_from_request(&request).expect("scope");
    let selection = RuntimeModelSelection {
        provider: "openai".to_string(),
        model: "gpt-4.1".to_string(),
        source: "test",
        reasoning_effort: None,
    };
    let policy = request_tool_policy_from_request(host_request.as_ref());

    let config = session_config_from_request(
        &request,
        host_request.as_ref(),
        &scope,
        &selection,
        &policy,
        None,
    );
    let system_prompt = config.system_prompt.expect("system prompt");
    let root_shared = system_prompt.find("根共享规则").expect("root shared");
    let nested_shared = system_prompt.find("子目录共享规则").expect("nested shared");

    assert!(system_prompt.contains("请求级系统提示"));
    assert!(root_shared < nested_shared);
}

#[test]
fn session_config_uses_explicit_project_root_for_runtime_agents_boundary() {
    let workspace = TempDir::new().expect("create workspace");
    let parent = workspace.path().join("parent");
    let repo = parent.join("repo");
    let nested = repo.join("apps").join("writer");
    std::fs::create_dir_all(parent.join(".lime")).expect("create parent runtime agents dir");
    std::fs::create_dir_all(repo.join(".lime")).expect("create root runtime agents dir");
    std::fs::create_dir_all(nested.join(".lime")).expect("create nested runtime agents dir");
    std::fs::write(
        parent.join(".lime").join("AGENTS.md"),
        "- 父目录规则不应出现",
    )
    .expect("write parent runtime agents");
    std::fs::write(repo.join(".lime").join("AGENTS.md"), "- 显式根规则")
        .expect("write root runtime agents");
    std::fs::write(
        nested.join(".lime").join("AGENTS.override.md"),
        "- 子目录覆盖规则",
    )
    .expect("write nested override runtime agents");
    let request = request_for_test(
        "请按项目规则处理",
        Some(RuntimeRequest {
            project_root: Some(repo.to_string_lossy().into_owned()),
            system_prompt: Some("请求级系统提示".to_string()),
            working_dir: Some(nested.to_string_lossy().into_owned()),
            ..RuntimeRequest::default()
        }),
        None,
    );
    let host_request = runtime_request_from_request(&request);
    let scope = session_scope_from_request(&request).expect("scope");
    let selection = RuntimeModelSelection {
        provider: "openai".to_string(),
        model: "gpt-4.1".to_string(),
        source: "test",
        reasoning_effort: None,
    };
    let policy = request_tool_policy_from_request(host_request.as_ref());

    let config = session_config_from_request(
        &request,
        host_request.as_ref(),
        &scope,
        &selection,
        &policy,
        None,
    );
    let system_prompt = config.system_prompt.expect("system prompt");
    let root_rule = system_prompt.find("显式根规则").expect("root rule");
    let nested_override = system_prompt
        .find("子目录覆盖规则")
        .expect("nested override rule");
    let turn_context = config.turn_context.expect("turn context");
    let runtime_metadata = turn_context
        .metadata
        .get("app_server_runtime_backend")
        .expect("runtime metadata");
    let nested_string = nested.to_string_lossy().to_string();
    let repo_string = repo.to_string_lossy().to_string();

    assert!(system_prompt.contains("# AGENTS.md instructions"));
    assert!(system_prompt.contains("<INSTRUCTIONS>"));
    assert!(root_rule < nested_override);
    assert!(!system_prompt.contains("父目录规则不应出现"));
    assert_eq!(turn_context.cwd.as_deref(), Some(nested.as_path()));
    assert_eq!(
        runtime_metadata["workingDir"].as_str(),
        Some(nested_string.as_str()),
    );
    assert_eq!(
        runtime_metadata["projectRoot"].as_str(),
        Some(repo_string.as_str()),
    );
}

#[test]
fn typed_runtime_request_reasoning_and_thinking_are_preserved() {
    let request = request_for_test(
        "hello",
        Some(RuntimeRequest {
            reasoning_effort: Some("high".to_string()),
            thinking_enabled: Some(true),
            ..RuntimeRequest::default()
        }),
        None,
    );
    let host_request = runtime_request_from_request(&request).expect("host request");

    assert_eq!(
        host_reasoning_effort(&host_request).as_deref(),
        Some("high")
    );
    assert_eq!(host_thinking_enabled(&host_request), Some(true));

    let scope = session_scope_from_request(&request).expect("scope");
    let selection = RuntimeModelSelection {
        provider: "openai".to_string(),
        model: "gpt-4.1".to_string(),
        source: "test",
        reasoning_effort: Some("high".to_string()),
    };
    let turn_context =
        turn_context_from_request(&request, Some(&host_request), &scope, &selection, None)
            .expect("turn context");
    let runtime_metadata = turn_context
        .metadata
        .get("app_server_runtime_backend")
        .expect("runtime metadata");

    assert_eq!(runtime_metadata["thinkingEnabled"], true);
}
