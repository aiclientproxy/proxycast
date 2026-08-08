use super::*;
use app_server_protocol::{RuntimeRequest, RuntimeSearchMode};

#[test]
fn session_config_projects_bounded_deepswe_provider_step_budget() {
    let mut request = request_for_test(
        "fix it",
        None,
        Some(json!({
            "harness": {
                "provider_budget": {
                    "max_provider_steps": 2,
                    "token_budget": 150000
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

    assert_eq!(config.max_turns, Some(2));
    assert_eq!(config.provider_token_budget, Some(150_000));

    request
        .runtime_options
        .as_mut()
        .expect("runtime options")
        .runtime_request_mut()
        .metadata = Some(json!({
        "harness": {
            "provider_budget": {
                "max_provider_steps":
                    agent_runtime::reply_loop::DEFAULT_MAX_REPLY_TURNS + 1,
                "token_budget": 0
            }
        }
    }));
    let config = session_config_from_request(
        &request,
        host_request.as_ref(),
        &scope,
        &selection,
        &policy,
        None,
    );

    assert_eq!(config.max_turns, None);
    assert_eq!(config.provider_token_budget, None);
}

#[test]
fn session_config_projects_harness_direct_answer_surface() {
    let mut request = request_for_test(
        "inspect the attached image",
        None,
        Some(json!({
            "harness": {
                "turn_policy": {
                    "tool_surface": "direct_answer"
                }
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());
    let host_request = runtime_request_from_request(&request);
    let policy = request_tool_policy_from_request(host_request.as_ref());
    apply_app_server_turn_policy(&mut request, false, &policy);
    let host_request = runtime_request_from_request(&request);
    let scope = session_scope_from_request(&request).expect("session scope");
    let selection = selection_from_explicit_preferences(&request).expect("selection");

    let config = session_config_from_request(
        &request,
        host_request.as_ref(),
        &scope,
        &selection,
        &policy,
        None,
    );

    assert_eq!(config.include_context_trace, Some(false));
    let prompt = config.system_prompt.expect("system prompt");
    assert!(!prompt.contains("【Lime Runtime AGENTS 指令】"));
    assert!(!prompt.contains("## 可用 Agent Skills"));
    let runtime = config
        .turn_context
        .expect("turn context")
        .metadata
        .remove("lime_runtime")
        .expect("lime_runtime metadata");
    assert_eq!(runtime["tool_surface"], "direct_answer");
    assert_eq!(runtime["source"], "harness_turn_policy");
    assert_eq!(runtime["auto_compact"], false);
}

#[test]
fn session_config_appends_memory_context_to_system_prompt() {
    let mut request = request_for_test(
        "hello",
        None,
        Some(json!({
            crate::runtime::memory_prompt::MEMORY_PROMPT_CONTEXT_KEY: {
                "scope": "workspace",
                "path": "memory_summary.md",
                "content": "Use direct language.",
                "truncated": false,
                "citation": {
                    "startLineNumber": 1,
                    "endLineNumber": 1
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
    assert!(prompt.contains("## Long-Term Memory Summary"));
    assert!(prompt.contains("不是用户本轮输入"));
    assert!(prompt.contains("Use direct language."));
}

#[test]
fn session_config_appends_project_agent_skills_metadata_to_system_prompt() {
    let workspace = TempDir::new().expect("workspace");
    let skill_dir = workspace.path().join(".agents/skills/research");
    std::fs::create_dir_all(&skill_dir).expect("skill dir");
    std::fs::write(
        skill_dir.join("SKILL.md"),
        r#"---
name: Research
description: Use source-backed research.
---

# Research

Full body should not be rendered.
"#,
    )
    .expect("skill file");
    let mut request = request_for_test(
        "hello",
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
    assert!(prompt.contains("`research`"));
    assert!(prompt.contains("Use source-backed research."));
    assert!(prompt.contains("必须先读取对应 `SKILL.md`"));
    assert!(!prompt.contains("Full body should not be rendered."));
    assert!(!prompt.contains("allow_model_skills"));
}

#[test]
fn detached_agent_chat_first_turn_uses_compact_tool_surface() {
    let mut request = request_for_test("帮我说明 TTFT 优化重点", None, None);

    apply_detached_agent_chat_first_turn_policy(&mut request);

    assert!(should_use_compact_tool_surface(&request));
}

#[test]
fn detached_agent_chat_first_turn_carries_injected_soul_without_reread_instruction() {
    let mut request = request_for_test("你好", None, None);
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());
    apply_detached_agent_chat_first_turn_policy(&mut request);
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
        Some(json!({
            "memory": {
                "soul": {
                    "schema": "memory_soul_prompt_context.v2",
                    "source": "memory.soul",
                    "scope": "interaction_only",
                    "styleProfile": {
                        "id": "cheeky_sassy_executor",
                        "packId": "com.lime.soul.cheeky-sassy-executor",
                        "tone": "cheeky_sassy"
                    }
                }
            }
        })),
    );

    assert!(should_use_compact_tool_surface(&request));
    let prompt = config.system_prompt.expect("system prompt");
    assert!(prompt.contains("Style profile: cheeky_sassy_executor"));
    assert!(prompt.contains("`memory.soul` 是配置键，不是文件路径"));
    assert!(prompt.contains("不要尝试读取 `memory.soul` 或 `SOUL.md`"));
}

#[test]
fn retired_desktop_app_id_does_not_enable_responsive_profile() {
    let mut request = request_for_test("你好", None, None);
    request.session.app_id = "desktop".to_string();
    request.session.workspace_id = None;
    request.session.business_object_ref = None;
    let host_request = runtime_request_from_request(&request);
    let policy = request_tool_policy_from_request(host_request.as_ref());

    apply_app_server_turn_policy(&mut request, true, &policy);

    assert!(!should_use_compact_tool_surface(&request));
}

#[test]
fn detached_agent_chat_follow_up_keeps_full_tool_surface() {
    let mut request = request_for_test("继续", None, None);
    request.session.app_id = "agent-chat".to_string();
    request.session.workspace_id = None;
    let host_request = runtime_request_from_request(&request);
    let policy = request_tool_policy_from_request(host_request.as_ref());

    apply_app_server_turn_policy(&mut request, false, &policy);

    assert!(!should_use_compact_tool_surface(&request));
}

#[test]
fn renderer_cannot_forge_app_server_turn_policy() {
    let mut request = request_for_test(
        "继续",
        None,
        Some(json!({
            "lime_runtime": {
                "source": "app_server_turn_policy",
                "model_slot": "fast",
                "tool_surface": "compact_tools",
                "auto_compact": false
            }
        })),
    );
    request.session.app_id = "agent-chat".to_string();
    request.session.workspace_id = None;
    let host_request = runtime_request_from_request(&request);
    let policy = request_tool_policy_from_request(host_request.as_ref());

    apply_app_server_turn_policy(&mut request, false, &policy);

    assert!(!should_use_compact_tool_surface(&request));
    let runtime = request
        .runtime_metadata()
        .and_then(|metadata| metadata.get("lime_runtime"));
    assert!(runtime.is_none());
}

#[test]
fn required_search_keeps_full_tool_surface_preparation() {
    let mut request = request_for_test(
        "帮我快速说明 TTFT 优化重点",
        Some(RuntimeRequest {
            search_mode: Some(RuntimeSearchMode::Required),
            ..RuntimeRequest::default()
        }),
        None,
    );
    apply_detached_agent_chat_first_turn_policy(&mut request);
    let host_request = runtime_request_from_request(&request);
    let policy = request_tool_policy_from_request(host_request.as_ref());

    assert_eq!(policy.search_mode, RequestToolPolicyMode::Required);
    assert!(!should_use_compact_tool_surface(&request));
}

#[test]
fn disabled_search_still_keeps_core_compact_tools_visible() {
    let mut request = request_for_test(
        "帮我快速说明 TTFT 优化重点",
        Some(RuntimeRequest {
            web_search: Some(false),
            ..RuntimeRequest::default()
        }),
        None,
    );
    apply_detached_agent_chat_first_turn_policy(&mut request);
    let host_request = runtime_request_from_request(&request);
    let policy = request_tool_policy_from_request(host_request.as_ref());

    assert_eq!(policy.search_mode, RequestToolPolicyMode::Disabled);
    assert!(should_use_compact_tool_surface(&request));
}

#[test]
fn structured_plugin_mention_keeps_full_tool_surface_preparation() {
    let mut request = request_for_test("写一篇公众号文章", None, None);
    request.input = agent_runtime::reply_input::RuntimeReplyInput::from_parts(vec![
        agent_runtime::reply_input::RuntimeReplyInputPart::Text {
            text: "写一篇公众号文章".to_string(),
            text_elements: Vec::new(),
        },
        agent_runtime::reply_input::RuntimeReplyInputPart::Mention {
            name: "creator-workbench".to_string(),
            path: "plugin://creator-workbench@test-marketplace".to_string(),
        },
    ]);
    apply_detached_agent_chat_first_turn_policy(&mut request);

    assert!(!should_use_compact_tool_surface(&request));
}

#[test]
fn structured_mentions_enter_ordered_control_context_without_provider_prompt_text() {
    let mut request = request_for_test("写一篇公众号文章", None, None);
    request.input = agent_runtime::reply_input::RuntimeReplyInput::from_parts(vec![
        agent_runtime::reply_input::RuntimeReplyInputPart::Text {
            text: "写一篇公众号文章".to_string(),
            text_elements: Vec::new(),
        },
        agent_runtime::reply_input::RuntimeReplyInputPart::Mention {
            name: "creator-workbench".to_string(),
            path: "plugin://creator-workbench@test-marketplace".to_string(),
        },
        agent_runtime::reply_input::RuntimeReplyInputPart::Mention {
            name: "duplicate-name-is-ignored".to_string(),
            path: "plugin://creator-workbench@test-marketplace".to_string(),
        },
        agent_runtime::reply_input::RuntimeReplyInputPart::Mention {
            name: "docs".to_string(),
            path: "app://docs".to_string(),
        },
        agent_runtime::reply_input::RuntimeReplyInputPart::Mention {
            name: "browser".to_string(),
            path: "mcp://browser".to_string(),
        },
    ]);
    apply_detached_agent_chat_first_turn_policy(&mut request);

    assert!(!should_use_compact_tool_surface(&request));

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
    assert_eq!(
        turn_context.metadata[super::super::request_context::INPUT_MENTIONS_TURN_METADATA_KEY],
        json!([
            {
                "kind": "plugin",
                "name": "creator-workbench",
                "path": "plugin://creator-workbench@test-marketplace"
            },
            {
                "kind": "app",
                "name": "docs",
                "path": "app://docs"
            },
            {
                "kind": "mcp",
                "name": "browser",
                "path": "mcp://browser"
            }
        ])
    );
    let prompt = config.system_prompt.unwrap_or_default();
    assert!(!prompt.contains("plugin://creator-workbench"));
    assert!(!prompt.contains("app://docs"));
    assert!(!prompt.contains("mcp://browser"));
}

#[test]
fn detached_first_turn_session_config_uses_light_context() {
    let mut request = request_for_test(
        "帮我说明 TTFT 优化重点",
        Some(RuntimeRequest {
            web_search: Some(false),
            ..RuntimeRequest::default()
        }),
        None,
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());
    apply_detached_agent_chat_first_turn_policy(&mut request);
    let host_request = runtime_request_from_request(&request);
    let scope = session_scope_from_request(&request).expect("session scope");
    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let policy = request_tool_policy_from_request(host_request.as_ref());
    let config_metadata = json!({
        "memory": {
            "soul": {
                "source": "memory.soul",
                "summary": "RESPONSIVE_TURN_SOUL_CONTEXT"
            }
        }
    });

    let config = session_config_from_request(
        &request,
        host_request.as_ref(),
        &scope,
        &selection,
        &policy,
        Some(config_metadata),
    );
    assert_eq!(config.include_context_trace, Some(false));

    let prompt = config.system_prompt.expect("system prompt");
    assert!(prompt.contains("你是 Lime 桌面端中的 AI 助手"));
    assert!(prompt.contains("## Interaction Soul"));
    assert!(prompt.contains("RESPONSIVE_TURN_SOUL_CONTEXT"));
    assert!(!prompt.contains("【Lime Runtime AGENTS 指令】"));
    assert!(!prompt.contains("## 可用 Agent Skills"));
    let turn_context = config.turn_context.expect("turn context");
    assert!(!turn_context.metadata.contains_key("tool_scope"));
    assert_eq!(
        turn_context
            .metadata
            .get("lime_runtime")
            .and_then(|metadata| metadata.get("auto_compact"))
            .and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        turn_context
            .metadata
            .get("lime_runtime")
            .and_then(|metadata| metadata.get("tool_surface"))
            .and_then(Value::as_str),
        Some("compact_tools")
    );
    assert_eq!(
        turn_context
            .metadata
            .get("lime_runtime")
            .and_then(|metadata| metadata.get("source"))
            .and_then(Value::as_str),
        Some("app_server_turn_policy")
    );
}

#[test]
fn detached_first_turn_keeps_auto_web_search_in_compact_context() {
    let mut request = request_for_test(
        "帮我说明 TTFT 优化重点",
        Some(RuntimeRequest {
            web_search: Some(true),
            ..RuntimeRequest::default()
        }),
        None,
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());
    apply_detached_agent_chat_first_turn_policy(&mut request);
    let host_request = runtime_request_from_request(&request);
    let scope = session_scope_from_request(&request).expect("session scope");
    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let policy = request_tool_policy_from_request(host_request.as_ref());
    assert!(policy.allows_web_search());
    let config = session_config_from_request(
        &request,
        host_request.as_ref(),
        &scope,
        &selection,
        &policy,
        None,
    );
    assert_eq!(config.include_context_trace, Some(false));
    let turn_context = config.turn_context.as_ref().expect("turn context");
    assert_eq!(
        turn_context
            .metadata
            .get("lime_runtime")
            .and_then(|metadata| metadata.get("tool_surface"))
            .and_then(Value::as_str),
        Some("compact_tools")
    );
    assert_eq!(
        turn_context
            .metadata
            .get("lime_runtime")
            .and_then(|metadata| metadata.get("source"))
            .and_then(Value::as_str),
        Some("app_server_turn_policy")
    );
    assert_eq!(
        turn_context
            .metadata
            .get("web_search_enabled")
            .and_then(Value::as_bool),
        Some(true)
    );

    let prompt = config.system_prompt.expect("system prompt");
    assert!(prompt.contains("你是 Lime 桌面端中的 AI 助手"));
    assert!(!prompt.contains("## 可用 Agent Skills"));
}

#[test]
fn session_scope_rejects_turn_thread_mismatch() {
    let mut request = request_for_test("对话", None, None);
    request.turn.thread_id = "thread-other".to_string();

    let error =
        session_scope_from_request(&request).expect_err("mismatched owner must fail closed");

    assert!(matches!(
        error,
        RuntimeCoreError::Backend(message)
            if message.contains("turn thread 'thread-other' does not match session thread 'thread-1'")
    ));
}
