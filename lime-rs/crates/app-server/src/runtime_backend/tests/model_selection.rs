use super::*;

fn reasoning_capability_snapshot(levels: &[&str]) -> app_server_protocol::CapabilitySnapshot {
    app_server_protocol::CapabilitySnapshot {
        runtime_features: vec!["streaming".to_string(), "reasoning".to_string()],
        capabilities: app_server_protocol::ModelCapabilitiesInfo {
            tools: true,
            streaming: true,
            reasoning: true,
            reasoning_effort: Some(app_server_protocol::ModelReasoningEffortSupportInfo {
                supported: true,
                levels: levels.iter().map(|level| (*level).to_string()).collect(),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn thread_settings(
    provider: &str,
    model: &str,
    effort: Option<&str>,
) -> app_server_protocol::protocol::v2::ThreadSettings {
    app_server_protocol::protocol::v2::ThreadSettings {
        cwd: "/tmp/model-switch".to_string(),
        approval_policy: serde_json::Value::Null,
        approvals_reviewer: serde_json::Value::Null,
        sandbox_policy: serde_json::Value::Null,
        active_permission_profile: None,
        model: model.to_string(),
        model_provider: provider.to_string(),
        service_tier: None,
        effort: effort.map(ToString::to_string),
        summary: None,
        collaboration_mode: agent_protocol::CollaborationMode {
            mode: agent_protocol::ModeKind::Default,
            settings: agent_protocol::CollaborationModeSettings {
                model: model.to_string(),
                reasoning_effort: effort.map(ToString::to_string),
                developer_instructions: None,
            },
        },
        personality: None,
        tool_preferences: None,
    }
}

fn configured_model_switch_backend(model: &str) -> (RuntimeBackend, String) {
    let connection = rusqlite::Connection::open_in_memory().expect("open database");
    lime_core::database::schema::create_tables(&connection).expect("create schema");
    let db = std::sync::Arc::new(std::sync::Mutex::new(connection));
    let backend = RuntimeBackend::with_db(db.clone());
    let provider = backend
        .api_key_provider_service
        .add_custom_provider(
            &db,
            "Model Switch Fixture".to_string(),
            lime_core::database::dao::api_key_provider::ApiProviderType::Openai,
            "https://example.invalid/v1".to_string(),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("custom provider");
    backend
        .api_key_provider_service
        .add_api_key(&db, &provider.id, "fixture-key", None, true)
        .expect("provider key");
    backend
        .api_key_provider_service
        .update_provider(
            &db,
            &provider.id,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(vec![model.to_string()]),
        )
        .expect("declared model");
    (backend, provider.id)
}

#[tokio::test]
async fn model_switch_preflight_requires_ready_catalog_model_and_supported_effort() {
    let (backend, provider_id) = configured_model_switch_backend("switch-model");
    let session = request_for_test("preflight", None, None).session;

    backend
        .preflight_thread_settings_route(
            &session,
            &thread_settings(&provider_id, "switch-model", None),
        )
        .await
        .expect("declared model is executable");

    let missing_model = backend
        .preflight_thread_settings_route(
            &session,
            &thread_settings(&provider_id, "missing-model", None),
        )
        .await
        .expect_err("unknown model must fail before settings persistence");
    assert!(matches!(
        missing_model,
        RuntimeCoreError::PendingRoute { reason_code, .. }
            if reason_code == "model_registry_metadata_missing"
    ));

    let unsupported_effort = backend
        .preflight_thread_settings_route(
            &session,
            &thread_settings(&provider_id, "switch-model", Some("ultra")),
        )
        .await
        .expect_err("unsupported effort must fail before settings persistence");
    assert!(matches!(
        unsupported_effort,
        RuntimeCoreError::RouteRejected {
            reason_code,
            category: app_server_protocol::RouteFailureCategory::CapabilityGap,
            ..
        } if reason_code == "reasoning_effort_unsupported"
    ));
}

#[tokio::test]
async fn prepared_route_pins_the_round_robin_credential_for_execution() {
    let connection = rusqlite::Connection::open_in_memory().expect("open database");
    lime_core::database::schema::create_tables(&connection).expect("create schema");
    let db = std::sync::Arc::new(std::sync::Mutex::new(connection));
    let backend = RuntimeBackend::with_db(db.clone());
    let provider = backend
        .api_key_provider_service
        .add_custom_provider(
            &db,
            "Route Pin Fixture".to_string(),
            lime_core::database::dao::api_key_provider::ApiProviderType::Openai,
            "https://example.invalid/v1".to_string(),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("custom provider");
    let first_key = backend
        .api_key_provider_service
        .add_api_key(
            &db,
            &provider.id,
            "first-key",
            Some("first".to_string()),
            true,
        )
        .expect("first key");
    let second_key = backend
        .api_key_provider_service
        .add_api_key(
            &db,
            &provider.id,
            "second-key",
            Some("second".to_string()),
            false,
        )
        .expect("second key");
    backend
        .api_key_provider_service
        .update_provider(
            &db,
            &provider.id,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(vec!["route-pin-model".to_string()]),
        )
        .expect("declared model");
    assert_eq!(
        backend
            .api_key_provider_service
            .get_provider(&db, &provider.id)
            .expect("read provider")
            .expect("configured provider")
            .api_keys
            .iter()
            .filter(|key| key.enabled)
            .count(),
        2
    );

    let advanced_ref = backend
        .api_key_provider_service
        .select_credential_for_provider(&db, &provider.id, Some(&provider.id), None)
        .await
        .expect("advance round robin")
        .expect("first credential")
        .uuid;
    let next_ref = backend
        .api_key_provider_service
        .select_credential_for_provider(&db, &provider.id, Some(&provider.id), None)
        .await
        .expect("read next round-robin credential")
        .expect("second credential")
        .uuid;
    assert_ne!(next_ref, advanced_ref);

    let mut request = request_for_test("hello", None, None);
    let runtime_request = request
        .runtime_options
        .as_mut()
        .expect("runtime options")
        .runtime_request_mut();
    runtime_request.provider_preference = Some(provider.id.clone());
    runtime_request.model_preference = Some("route-pin-model".to_string());

    let prepared = backend
        .prepare_turn_route(&request, true)
        .await
        .expect("prepare route")
        .expect("prepared options");
    let first_ref = lime_core::models::runtime_api_key_credential_uuid(&first_key.id);
    let second_ref = lime_core::models::runtime_api_key_credential_uuid(&second_key.id);
    let prepared_ref = prepared
        .runtime_request
        .as_ref()
        .and_then(|runtime_request| runtime_request.metadata.as_ref())
        .and_then(|metadata| metadata.pointer("/agentControlRoute/credentialRef"))
        .and_then(Value::as_str)
        .map(ToString::to_string);
    assert!(
        matches!(prepared_ref.as_deref(), Some(value) if value == first_ref || value == second_ref)
    );
    assert_eq!(prepared_ref.as_deref(), Some(advanced_ref.as_str()));
    assert_ne!(prepared_ref.as_deref(), Some(next_ref.as_str()));

    request.runtime_options = Some(prepared);
    let execution_route = backend
        .resolve_turn_route(&request)
        .await
        .expect("resolve prepared route");
    assert_eq!(
        execution_route
            .resolution
            .resolved_route
            .auth
            .credential_ref
            .as_deref(),
        prepared_ref.as_deref()
    );
}

#[test]
fn explicit_runtime_preferences_win() {
    let mut request = request_for_test("hello", None, None);
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("deepseek".to_string());
    options.runtime_request_mut().model_preference = Some("deepseek-chat".to_string());

    let selection = selection_from_explicit_preferences(&request).expect("selection");
    assert_eq!(
        selection,
        RuntimeModelSelection {
            provider: "deepseek".to_string(),
            model: "deepseek-chat".to_string(),
            source: "runtime_request",
            reasoning_effort: None,
        }
    );
}

#[test]
fn missing_runtime_selection_returns_typed_pending_route() {
    let request = request_for_test("hello", None, None);

    let error = resolve_runtime_model_selection(&request).expect_err("selection must stay pending");

    assert!(matches!(
        error,
        RuntimeCoreError::PendingRoute {
            session_id,
            provider: None,
            model: None,
            reason_code,
        } if session_id == "session-1" && reason_code == "provider_and_model_missing"
    ));
}

#[test]
fn app_server_turn_policy_can_select_configured_responsive_slot() {
    let mut request = request_for_test(
        "只回答一个字：好",
        None,
        Some(json!({
            "harness": {
                "model_slots": {
                    "fast": {
                        "provider": "responsive-provider",
                        "model": "fast-chat",
                        "source": "service_models.responsive_chat",
                        "reason": "service_model_preference"
                    }
                }
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("deepseek".to_string());
    options.runtime_request_mut().model_preference = Some("deepseek-chat".to_string());
    apply_detached_agent_chat_first_turn_policy(&mut request);

    let selection = resolve_runtime_model_selection(&request).expect("responsive selection");

    assert_eq!(
        selection,
        RuntimeModelSelection {
            provider: "responsive-provider".to_string(),
            model: "fast-chat".to_string(),
            source: "profile_model_slot",
            reasoning_effort: None,
        }
    );
}

#[test]
fn runtime_model_selection_prefers_responsive_slot_for_detached_first_turn() {
    let mut request = request_for_test(
        "只回答一个字：好",
        None,
        Some(json!({
            "harness": {
                "model_slots": {
                    "fast": {
                        "provider": "responsive-provider",
                        "model": "fast-chat",
                        "source": "service_models.responsive_chat",
                        "reason": "service_model_preference"
                    }
                }
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("deepseek".to_string());
    options.runtime_request_mut().model_preference = Some("deepseek-chat".to_string());
    apply_detached_agent_chat_first_turn_policy(&mut request);

    let selection = resolve_runtime_model_selection(&request).expect("selection");

    assert_eq!(selection.provider, "responsive-provider");
    assert_eq!(selection.model, "fast-chat");
    assert_eq!(selection.source, "profile_model_slot");
    assert_eq!(selection.reasoning_effort, None);

    let scope = session_scope_from_request(&request).expect("scope");
    let turn_context =
        turn_context_from_request(&request, None, &scope, &selection, None).expect("turn context");
    assert_eq!(turn_context.effort, None);
}

#[test]
fn effective_reasoning_selection_drops_effort_missing_from_snapshot() {
    let selection = RuntimeModelSelection {
        provider: "openai-compatible".to_string(),
        model: "gpt-4o-mini".to_string(),
        source: "runtime_options",
        reasoning_effort: Some("minimal".to_string()),
    };

    let snapshot = reasoning_capability_snapshot(&["low", "high"]);
    let effective_selection = selection_with_capability_reasoning(&selection, &snapshot);

    assert_eq!(effective_selection.reasoning_effort, None);
}

#[test]
fn effective_reasoning_selection_keeps_explicit_minimal_wire_value() {
    let selection = RuntimeModelSelection {
        provider: "openai".to_string(),
        model: "gpt-codex".to_string(),
        source: "runtime_options",
        reasoning_effort: Some("minimal".to_string()),
    };

    let snapshot = reasoning_capability_snapshot(&["minimal", "low", "high"]);
    let effective_selection = selection_with_capability_reasoning(&selection, &snapshot);

    assert_eq!(
        effective_selection.reasoning_effort.as_deref(),
        Some("minimal")
    );
}

#[test]
fn responsive_policy_without_configured_slot_keeps_explicit_preferences() {
    let mut request = request_for_test("只回答一个字：好", None, None);
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("deepseek".to_string());
    options.runtime_request_mut().model_preference = Some("deepseek-chat".to_string());
    apply_detached_agent_chat_first_turn_policy(&mut request);

    let selection = resolve_runtime_model_selection(&request).expect("selection");
    assert_eq!(selection.provider, "deepseek");
    assert_eq!(selection.model, "deepseek-chat");
}

#[test]
fn host_provider_config_without_direct_credentials_stays_database_backed() {
    let request = request_for_test(
        "hello",
        Some(app_server_protocol::RuntimeRequest {
            provider_config: Some(app_server_protocol::RuntimeProviderConfig {
                provider_id: Some("database-openai".to_string()),
                provider_name: Some("openai".to_string()),
                model_name: Some("gpt-4.1".to_string()),
                ..app_server_protocol::RuntimeProviderConfig::default()
            }),
            provider_preference: Some("database-openai".to_string()),
            model_preference: Some("gpt-4.1".to_string()),
            ..app_server_protocol::RuntimeRequest::default()
        }),
        None,
    );
    let host_request = runtime_request_from_request(&request);
    let selection = selection_from_host_provider_config(&request).expect("selection");

    let direct_config = direct_provider_config_from_request(
        host_request.as_ref(),
        &selection,
        selection.reasoning_effort.clone(),
    );

    assert!(direct_config.is_none());
    assert_eq!(selection.provider, "database-openai");
    assert_eq!(selection.model, "gpt-4.1");
    assert_eq!(selection.source, "runtime_request_provider_config");
}

#[test]
fn imported_session_source_model_is_not_session_default_route() {
    let request = imported_request_with_session_metadata(json!({
        "providerName": "openai",
        "modelName": "gpt-5.5",
        "sourceClient": "codex",
        "importedContinuation": {
            "modelProvider": "openai",
            "model": "gpt-5.5"
        }
    }));

    assert!(selection_from_session_default(&request).is_none());
}

#[test]
fn imported_session_current_provider_selector_remains_session_default_route() {
    let request = imported_request_with_session_metadata(json!({
        "providerSelector": "custom-current-provider",
        "providerName": "openai",
        "modelName": "gpt-5.5",
        "sourceClient": "codex",
        "importedContinuation": {
            "modelProvider": "openai",
            "model": "gpt-5.5"
        }
    }));

    let selection = selection_from_session_default(&request).expect("selection");

    assert_eq!(selection.provider, "custom-current-provider");
    assert_eq!(selection.model, "gpt-5.5");
    assert_eq!(selection.source, "session_default");
}

#[test]
fn direct_host_provider_config_allows_localhost_fixture_without_database_provider() {
    let request = request_for_test(
        "hello",
        Some(app_server_protocol::RuntimeRequest {
            provider_config: Some(app_server_protocol::RuntimeProviderConfig {
                provider_id: Some("fixture-openai".to_string()),
                provider_name: Some("openai".to_string()),
                model_name: Some("lime-fixture-chat".to_string()),
                api_key: Some("fixture-key".to_string()),
                base_url: Some("http://127.0.0.1:56599".to_string()),
                tool_call_strategy: Some(app_server_protocol::RuntimeToolCallStrategy::Native),
                supports_websockets: Some(true),
                ..app_server_protocol::RuntimeProviderConfig::default()
            }),
            provider_preference: Some("fixture-openai".to_string()),
            model_preference: Some("lime-fixture-chat".to_string()),
            reasoning_effort: Some("high".to_string()),
            ..app_server_protocol::RuntimeRequest::default()
        }),
        None,
    );
    let host_request = runtime_request_from_request(&request);
    let selection = selection_from_host_provider_config(&request).expect("selection");

    let direct_config = direct_provider_config_from_request(
        host_request.as_ref(),
        &selection,
        selection.reasoning_effort.clone(),
    )
    .expect("direct provider config");

    assert_eq!(direct_config.provider_name, "openai");
    assert_eq!(
        direct_config.provider_selector.as_deref(),
        Some("fixture-openai")
    );
    assert_eq!(direct_config.model_name, "lime-fixture-chat");
    assert_eq!(direct_config.api_key.as_deref(), Some("fixture-key"));
    assert_eq!(
        direct_config.base_url.as_deref(),
        Some("http://127.0.0.1:56599")
    );
    assert_eq!(direct_config.reasoning_effort.as_deref(), Some("high"));
    assert!(!direct_config.toolshim);
    assert!(direct_config.supports_websockets);
}

#[tokio::test]
async fn prepared_direct_route_persists_non_secret_provider_defaults() {
    let connection = rusqlite::Connection::open_in_memory().expect("open database");
    lime_core::database::schema::create_tables(&connection).expect("create schema");
    let db = std::sync::Arc::new(std::sync::Mutex::new(connection));
    let backend = RuntimeBackend::with_db(db);
    let request = request_for_test(
        "hello",
        Some(app_server_protocol::RuntimeRequest {
            provider_preference: Some("fixture-openai".to_string()),
            model_preference: Some("gpt-5.4".to_string()),
            reasoning_effort: Some("high".to_string()),
            provider_config: Some(app_server_protocol::RuntimeProviderConfig {
                provider_id: Some("fixture-openai".to_string()),
                provider_name: Some("openai".to_string()),
                model_name: Some("gpt-5.4".to_string()),
                api_key: Some("fixture-secret".to_string()),
                base_url: Some("http://127.0.0.1:56599".to_string()),
                tool_call_strategy: Some(app_server_protocol::RuntimeToolCallStrategy::ToolShim),
                toolshim_model: Some("fixture-toolshim".to_string()),
                model_capabilities: Some(json!({
                    "capabilities": {
                        "tools": true,
                        "streaming": true,
                        "functionCalling": true
                    },
                    "taskFamilies": ["chat"],
                    "inputModalities": ["text"],
                    "outputModalities": ["text"],
                    "runtimeFeatures": ["streaming", "tool_calling"]
                })),
                supports_websockets: Some(true),
            }),
            ..app_server_protocol::RuntimeRequest::default()
        }),
        None,
    );

    let prepared = backend
        .prepare_turn_route(&request, true)
        .await
        .expect("prepare direct route")
        .expect("direct route options");
    let provider_config = prepared
        .runtime_request
        .as_ref()
        .and_then(|request| request.metadata.as_ref())
        .and_then(|metadata| metadata.pointer("/agentControlRoute/providerConfig"))
        .expect("durable provider config");

    assert_eq!(provider_config["providerId"], "fixture-openai");
    assert_eq!(provider_config["modelName"], "gpt-5.4");
    assert_eq!(provider_config["reasoningEffort"], Value::Null);
    assert_eq!(provider_config["toolshim"], true);
    assert_eq!(provider_config["toolshimModel"], "fixture-toolshim");
    assert_eq!(provider_config["supportsWebsockets"], true);
    assert!(provider_config.get("apiKey").is_none());
    assert!(provider_config.get("baseUrl").is_none());
}

#[test]
fn typed_runtime_reasoning_flows_to_selection_and_turn_context() {
    let request = request_for_test(
        "hello",
        Some(app_server_protocol::RuntimeRequest {
            provider_preference: Some("openai".to_string()),
            model_preference: Some("gpt-4.1".to_string()),
            reasoning_effort: Some("medium".to_string()),
            ..app_server_protocol::RuntimeRequest::default()
        }),
        None,
    );

    let selection = selection_from_explicit_preferences(&request).expect("selection");
    assert_eq!(selection.reasoning_effort.as_deref(), Some("medium"));

    let scope = session_scope_from_request(&request).expect("scope");
    let turn_context =
        turn_context_from_request(&request, None, &scope, &selection, None).expect("turn context");

    assert_eq!(turn_context.effort.as_deref(), Some("medium"));
}

#[test]
fn model_request_policy_reasoning_default_flows_to_selection_and_turn_context() {
    let mut request = request_for_test(
        "hello",
        None,
        Some(json!({
            "harness": {
                "model_request_policy": {
                    "reasoning_policy": {
                        "supports_reasoning_summary_parameter": false,
                        "default_reasoning_level": "high"
                    }
                }
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());

    let selection = selection_from_explicit_preferences(&request).expect("selection");
    assert_eq!(selection.reasoning_effort.as_deref(), Some("high"));

    let scope = session_scope_from_request(&request).expect("scope");
    let turn_context =
        turn_context_from_request(&request, None, &scope, &selection, None).expect("turn context");

    assert_eq!(turn_context.effort.as_deref(), Some("high"));
}

#[test]
fn explicit_reasoning_effort_wins_over_model_request_policy_default() {
    let mut request = request_for_test(
        "hello",
        None,
        Some(json!({
            "harness": {
                "reasoning_effort": "low",
                "model_request_policy": {
                    "reasoning_policy": {
                        "supports_reasoning_summary_parameter": false,
                        "default_reasoning_level": "high"
                    }
                }
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());

    let selection = selection_from_explicit_preferences(&request).expect("selection");

    assert_eq!(selection.reasoning_effort.as_deref(), Some("low"));
}

#[test]
fn model_request_policy_context_flows_to_lime_runtime_metadata() {
    let mut request = request_for_test(
        "hello",
        None,
        Some(json!({
            "harness": {
                "model_request_policy": {
                    "context_policy": {
                        "context_window": 100000,
                        "max_context_window": 400000,
                        "auto_compact_token_limit": 95000,
                        "effective_context_window_percent": 50
                    }
                }
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());

    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let scope = session_scope_from_request(&request).expect("scope");
    let turn_context =
        turn_context_from_request(&request, None, &scope, &selection, None).expect("turn context");
    let runtime = turn_context
        .metadata
        .get("lime_runtime")
        .expect("lime_runtime metadata");

    assert_eq!(
        runtime
            .pointer("/context_policy/source")
            .and_then(Value::as_str),
        Some("model_request_policy")
    );
    assert_eq!(
        runtime
            .pointer("/context_policy/resolved_context_window")
            .and_then(Value::as_i64),
        Some(100_000)
    );
    assert_eq!(
        runtime
            .pointer("/context_policy/model_context_window")
            .and_then(Value::as_i64),
        Some(50_000)
    );
    assert_eq!(
        runtime
            .pointer("/context_policy/auto_compact_token_limit")
            .and_then(Value::as_i64),
        Some(90_000)
    );
    assert_eq!(
        runtime.get("model_context_window").and_then(Value::as_i64),
        Some(50_000)
    );
    assert_eq!(
        runtime
            .get("auto_compact_token_limit")
            .and_then(Value::as_i64),
        Some(90_000)
    );
}

#[test]
fn app_server_turn_policy_keeps_context_policy_and_auto_compact_override() {
    let mut request = request_for_test(
        "只回答一个字：好",
        None,
        Some(json!({
            "harness": {
                "model_request_policy": {
                    "context_policy": {
                        "resolved_context_window": 200000,
                        "model_context_window": 120000,
                        "auto_compact_token_limit": 150000
                    }
                }
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());
    apply_detached_agent_chat_first_turn_policy(&mut request);

    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let scope = session_scope_from_request(&request).expect("scope");
    let turn_context =
        turn_context_from_request(&request, None, &scope, &selection, None).expect("turn context");
    let runtime = turn_context
        .metadata
        .get("lime_runtime")
        .expect("lime_runtime metadata");

    assert_eq!(
        runtime.get("auto_compact").and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        runtime.get("tool_surface").and_then(Value::as_str),
        Some("compact_tools")
    );
    assert_eq!(
        runtime
            .pointer("/context_policy/source")
            .and_then(Value::as_str),
        Some("model_request_policy")
    );
    assert_eq!(
        runtime.get("model_context_window").and_then(Value::as_i64),
        Some(120_000)
    );
    assert_eq!(
        runtime
            .get("auto_compact_token_limit")
            .and_then(Value::as_i64),
        Some(150_000)
    );
}

#[test]
fn valid_w3c_trace_context_flows_to_turn_context_metadata() {
    let mut request = request_for_test(
        "hello",
        None,
        Some(json!({
            "agentUiPerformanceTrace": {
                "traceId": "trace-local",
                "w3cTraceContext": {
                    "traceparent": "00-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA-BBBBBBBBBBBBBBBB-01",
                    "tracestate": " vendor=value "
                }
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());

    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let scope = session_scope_from_request(&request).expect("scope");
    let turn_context =
        turn_context_from_request(&request, None, &scope, &selection, None).expect("turn context");
    let w3c = turn_context
        .metadata
        .get("w3c_trace_context")
        .expect("w3c trace context");

    assert_eq!(
        w3c["traceparent"],
        "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"
    );
    assert_eq!(w3c["tracestate"], "vendor=value");
}

#[test]
fn invalid_w3c_trace_context_is_not_forwarded_to_turn_context() {
    let mut request = request_for_test(
        "hello",
        None,
        Some(json!({
            "agentUiPerformanceTrace": {
                "traceId": "trace-local",
                "w3cTraceContext": {
                    "traceparent": "not-a-traceparent",
                    "tracestate": "vendor=value"
                }
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());

    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let scope = session_scope_from_request(&request).expect("scope");
    let turn_context =
        turn_context_from_request(&request, None, &scope, &selection, None).expect("turn context");

    assert!(!turn_context.metadata.contains_key("w3c_trace_context"));
}

#[test]
fn runtime_options_expected_output_schema_flows_to_turn_context() {
    let output_schema = json!({
        "type": "object",
        "properties": {
            "items": {
                "type": "array"
            }
        },
        "required": ["items"]
    });
    let mut request = request_for_test("hello", None, None);
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());
    options.expected_output = Some(json!({
        "artifactKind": "content_batch",
        "outputFormat": {
            "type": "json_schema",
            "schema": output_schema.clone()
        }
    }));
    request.expected_output = options.expected_output.clone();

    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let scope = session_scope_from_request(&request).expect("scope");
    let turn_context =
        turn_context_from_request(&request, None, &scope, &selection, None).expect("turn context");

    assert_eq!(turn_context.output_schema, Some(output_schema));
    assert_eq!(
        turn_context.output_schema_source,
        Some(TurnOutputSchemaSource::Turn)
    );
}

#[test]
fn typed_runtime_reasoning_is_used_without_metadata_aliases() {
    let request = request_for_test(
        "hello",
        Some(app_server_protocol::RuntimeRequest {
            provider_preference: Some("openai".to_string()),
            model_preference: Some("gpt-4.1-mini".to_string()),
            reasoning_effort: Some("low".to_string()),
            ..app_server_protocol::RuntimeRequest::default()
        }),
        None,
    );

    let selection = selection_from_explicit_preferences(&request).expect("selection");
    assert_eq!(selection.reasoning_effort.as_deref(), Some("low"));

    let scope = session_scope_from_request(&request).expect("scope");
    let turn_context =
        turn_context_from_request(&request, None, &scope, &selection, None).expect("turn context");
    assert_eq!(turn_context.effort.as_deref(), Some("low"));
}

#[test]
fn typed_plan_mode_flows_to_turn_context_collaboration_mode() {
    let mut request = request_for_test(
        "先给我一个修复计划，不要直接改代码",
        Some(app_server_protocol::RuntimeRequest {
            collaboration_mode: Some(agent_protocol::CollaborationMode {
                mode: agent_protocol::ModeKind::Plan,
                settings: agent_protocol::CollaborationModeSettings {
                    model: "gpt-4.1".to_string(),
                    reasoning_effort: None,
                    developer_instructions: None,
                },
            }),
            ..app_server_protocol::RuntimeRequest::default()
        }),
        None,
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());

    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let scope = session_scope_from_request(&request).expect("scope");
    let turn_context =
        turn_context_from_request(&request, None, &scope, &selection, None).expect("turn context");

    assert_eq!(
        turn_context
            .collaboration_mode
            .as_ref()
            .map(|mode| mode.mode),
        Some(agent_protocol::ModeKind::Plan)
    );
}

#[test]
fn injected_tool_execution_config_flows_to_turn_context_metadata() {
    let mut request = request_for_test("hello", None, None);
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1-mini".to_string());
    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let scope = session_scope_from_request(&request).expect("scope");

    let turn_context = turn_context_from_request(
        &request,
        None,
        &scope,
        &selection,
        Some(json!({
            "agent": {
                "toolExecution": {
                    "toolOverrides": {
                        "bash": {
                            "warningPolicy": "none"
                        }
                    }
                }
            }
        })),
    )
    .expect("turn context");

    assert_eq!(
        turn_context
            .metadata
            .get("config")
            .and_then(|value| value.pointer("/agent/toolExecution/toolOverrides/bash/warningPolicy"))
            .and_then(Value::as_str),
        Some("none")
    );
}

#[test]
fn injected_workspace_sandbox_config_flows_to_turn_context_metadata() {
    let mut request = request_for_test("hello", None, None);
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1-mini".to_string());
    let selection = selection_from_explicit_preferences(&request).expect("selection");
    let scope = session_scope_from_request(&request).expect("scope");

    let turn_context = turn_context_from_request(
        &request,
        None,
        &scope,
        &selection,
        Some(json!({
            "agent": {
                "workspaceSandbox": {
                    "enabled": true,
                    "strict": true,
                    "notifyOnFallback": false
                }
            }
        })),
    )
    .expect("turn context");

    assert_eq!(
        turn_context
            .metadata
            .get("config")
            .and_then(|value| value.pointer("/agent/workspaceSandbox/enabled"))
            .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        turn_context
            .metadata
            .get("config")
            .and_then(|value| value.pointer("/agent/workspaceSandbox/strict"))
            .and_then(Value::as_bool),
        Some(true)
    );
}

#[test]
fn top_level_request_metadata_reasoning_is_used_when_runtime_metadata_omits_it() {
    let mut request = request_for_test("hello", None, Some(json!({ "trace": "runtime-only" })));
    request
        .runtime_options
        .get_or_insert_default()
        .runtime_request_mut()
        .metadata = Some(json!({
        "harness": {
            "modelReasoningEffort": "high"
        }
    }));
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("openai".to_string());
    options.runtime_request_mut().model_preference = Some("gpt-4.1".to_string());

    let selection = selection_from_explicit_preferences(&request).expect("selection");

    assert_eq!(selection.reasoning_effort.as_deref(), Some("high"));
}

#[test]
fn runtime_request_provider_config_and_reasoning_are_typed() {
    let request = request_for_test(
        "hello",
        Some(app_server_protocol::RuntimeRequest {
            reasoning_effort: Some("high".to_string()),
            provider_config: Some(app_server_protocol::RuntimeProviderConfig {
                provider_id: Some("turn-openai".to_string()),
                provider_name: Some("openai".to_string()),
                model_name: Some("turn-model".to_string()),
                api_key: Some("turn-key".to_string()),
                base_url: Some("http://127.0.0.1:56599".to_string()),
                tool_call_strategy: Some(app_server_protocol::RuntimeToolCallStrategy::ToolShim),
                toolshim_model: Some("turn-toolshim-model".to_string()),
                ..app_server_protocol::RuntimeProviderConfig::default()
            }),
            ..app_server_protocol::RuntimeRequest::default()
        }),
        None,
    );
    let host_request = runtime_request_from_request(&request);
    let selection = selection_from_host_provider_config(&request).expect("selection");
    let direct_config = direct_provider_config_from_request(
        host_request.as_ref(),
        &selection,
        selection.reasoning_effort.clone(),
    )
    .expect("direct provider config");

    assert_eq!(selection.provider, "turn-openai");
    assert_eq!(selection.model, "turn-model");
    assert_eq!(selection.reasoning_effort.as_deref(), Some("high"));
    assert_eq!(
        direct_config.provider_selector.as_deref(),
        Some("turn-openai")
    );
    assert_eq!(direct_config.model_name, "turn-model");
    assert_eq!(direct_config.api_key.as_deref(), Some("turn-key"));
    assert_eq!(
        direct_config.base_url.as_deref(),
        Some("http://127.0.0.1:56599")
    );
    assert_eq!(direct_config.reasoning_effort.as_deref(), Some("high"));
    assert!(direct_config.toolshim);
    assert_eq!(
        direct_config.toolshim_model.as_deref(),
        Some("turn-toolshim-model")
    );
}

#[test]
fn session_default_provider_model_is_used_after_frontend_compaction() {
    let request = request_with_session_metadata(json!({
        "providerSelector": "openai-compatible",
        "modelName": "gpt-4.1-mini"
    }));

    let selection = selection_from_session_default(&request).expect("selection");

    assert_eq!(selection.provider, "openai-compatible");
    assert_eq!(selection.model, "gpt-4.1-mini");
    assert_eq!(selection.source, "session_default");
}

#[test]
fn turn_model_override_uses_durable_session_provider() {
    let mut request = request_with_session_metadata(json!({
        "providerSelector": "openai-compatible",
        "modelName": "gpt-4.1-mini"
    }));
    request
        .runtime_options
        .get_or_insert_default()
        .runtime_request_mut()
        .model_preference = Some("gpt-5.4".to_string());

    let selection = resolve_runtime_model_selection(&request).expect("selection");

    assert_eq!(selection.provider, "openai-compatible");
    assert_eq!(selection.model, "gpt-5.4");
    assert_eq!(selection.source, "runtime_request_with_session_default");
}

#[test]
fn partial_turn_route_without_complete_session_default_fails_closed() {
    let mut request = request_for_test(
        "hello",
        Some(app_server_protocol::RuntimeRequest {
            model_preference: Some("gpt-5.4".to_string()),
            ..app_server_protocol::RuntimeRequest::default()
        }),
        None,
    );
    request.session.business_object_ref = Some(BusinessObjectRef {
        kind: "agent.thread".to_string(),
        id: "thread-1".to_string(),
        title: None,
        uri: None,
        metadata: Some(json!({ "modelName": "gpt-4.1-mini" })),
    });

    let error = resolve_runtime_model_selection(&request).expect_err("incomplete route");

    assert!(matches!(
        error,
        RuntimeCoreError::PendingRoute {
            provider: None,
            model: Some(model),
            reason_code,
            ..
        } if model == "gpt-5.4" && reason_code == "provider_missing"
    ));
}

#[test]
fn incomplete_session_default_is_not_a_runtime_selection() {
    let missing_model = request_with_session_metadata(json!({
        "providerSelector": "openai-compatible"
    }));
    let missing_provider = request_with_session_metadata(json!({
        "modelName": "gpt-4.1-mini"
    }));

    assert!(selection_from_session_default(&missing_model).is_none());
    assert!(selection_from_session_default(&missing_provider).is_none());
}

#[test]
fn session_extension_data_provider_routing_is_used_as_session_default() {
    let request = request_with_session_metadata(json!({
        "model": "claude-sonnet-4",
        "extensionData": {
            "lime_provider_routing.v0": {
                "providerSelector": "lime-hub"
            }
        }
    }));

    let selection = selection_from_session_default(&request).expect("selection");

    assert_eq!(selection.provider, "lime-hub");
    assert_eq!(selection.model, "claude-sonnet-4");
    assert_eq!(selection.source, "session_default");
}

#[test]
fn effective_child_options_pin_session_default_without_copying_session_metadata() {
    let request = request_with_session_metadata(json!({
        "providerSelector": "session-provider",
        "modelName": "session-model",
        "unrelatedLegacyMetadata": "must-not-be-copied"
    }));

    let options = effective_runtime_options_for_turn(&request, false).expect("effective options");
    let runtime_request = options.runtime_request.expect("effective runtime request");

    assert_eq!(
        runtime_request.provider_preference.as_deref(),
        Some("session-provider")
    );
    assert_eq!(
        runtime_request.model_preference.as_deref(),
        Some("session-model")
    );
    assert_eq!(
        runtime_request.workspace_id.as_deref(),
        Some("workspace-main")
    );
    assert!(runtime_request
        .metadata
        .as_ref()
        .is_none_or(|metadata| { metadata.get("unrelatedLegacyMetadata").is_none() }));
}

#[test]
fn effective_child_options_pin_profile_route_reasoning_policy_and_workspace() {
    let request = request_for_test(
        "delegate",
        Some(app_server_protocol::RuntimeRequest {
            reasoning_effort: Some("minimal".to_string()),
            working_dir: Some("/tmp/lime-profile-workspace/subdir".to_string()),
            project_root: Some("/tmp/lime-profile-workspace".to_string()),
            web_search: Some(true),
            search_mode: Some(app_server_protocol::RuntimeSearchMode::Required),
            ..app_server_protocol::RuntimeRequest::default()
        }),
        Some(json!({
            "harness": {
                "model_slots": {
                    "coding": {
                        "provider": "profile-provider",
                        "model": "gpt-codex"
                    }
                }
            }
        })),
    );

    let options = effective_runtime_options_for_turn(&request, false).expect("effective options");
    let runtime_request = options.runtime_request.expect("effective runtime request");

    assert_eq!(
        runtime_request.provider_preference.as_deref(),
        Some("profile-provider")
    );
    assert_eq!(
        runtime_request.model_preference.as_deref(),
        Some("gpt-codex")
    );
    assert_eq!(runtime_request.reasoning_effort.as_deref(), Some("minimal"));
    assert_eq!(
        runtime_request.working_dir.as_deref(),
        Some("/tmp/lime-profile-workspace/subdir")
    );
    assert_eq!(
        runtime_request.workspace_root.as_deref(),
        Some("/tmp/lime-profile-workspace")
    );
    assert_eq!(
        runtime_request.project_root.as_deref(),
        Some("/tmp/lime-profile-workspace")
    );
    assert_eq!(runtime_request.web_search, Some(true));
    assert_eq!(
        runtime_request.search_mode,
        Some(app_server_protocol::RuntimeSearchMode::Required)
    );
}
