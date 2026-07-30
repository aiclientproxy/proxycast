use super::*;
use crate::{CapabilitySnapshot, JsonRpcNotification, ModelCapabilitiesInfo};
use schemars::schema_for;
use serde_json::json;

#[test]
fn v2_display_items_match_codex_tagged_nullable_wire() {
    let command = json!({
        "type": "commandExecution",
        "id": "command-1",
        "command": "cargo test",
        "cwd": "/workspace",
        "source": "agent",
        "status": "completed",
        "commandActions": [],
        "exitCode": 0,
        "durationMs": 12
    });
    let decoded: ThreadItem = serde_json::from_value(command.clone()).expect("command item");
    assert_eq!(
        serde_json::to_value(decoded).expect("command wire"),
        command
    );

    let mcp = json!({
        "type": "mcpToolCall",
        "id": "mcp-1",
        "server": "docs",
        "tool": "search",
        "status": "completed",
        "arguments": {"query": "ThreadItem"},
        "result": {
            "content": [{"type": "text", "text": "found"}],
            "structuredContent": null,
            "_meta": null
        },
        "error": null,
        "durationMs": 12
    });
    let decoded: ThreadItem = serde_json::from_value(mcp.clone()).expect("MCP item");
    assert_eq!(serde_json::to_value(decoded).expect("MCP wire"), mcp);

    for content_item in [
        json!({"type": "inputText", "text": "ok"}),
        json!({"type": "inputImage", "imageUrl": "https://example.test/a.png"}),
        json!({"type": "inputAudio", "audioUrl": "https://example.test/a.wav"}),
    ] {
        let decoded: DynamicToolCallOutputContentItem =
            serde_json::from_value(content_item.clone()).expect("dynamic content item");
        assert_eq!(serde_json::to_value(decoded).unwrap(), content_item);
    }
}

#[test]
fn v2_agent_message_phase_uses_the_canonical_enum_owner() {
    let item = json!({
        "type": "agentMessage",
        "id": "message-1",
        "text": "done",
        "phase": "final_answer"
    });
    let decoded: ThreadItem = serde_json::from_value(item.clone()).expect("agent message");
    assert_eq!(serde_json::to_value(decoded).unwrap(), item);
}

#[test]
fn thread_start_uses_v2_camel_case_fields() {
    let params = ThreadStartParams {
        model: Some("gpt-5.4".to_string()),
        model_provider: Some("openai".to_string()),
        runtime_workspace_roots: Some(vec!["/workspace".to_string()]),
        session_start_source: Some(ThreadStartSource::Startup),
        ..ThreadStartParams::default()
    };

    assert_eq!(
        serde_json::to_value(params).expect("serialize thread/start params"),
        json!({
            "model": "gpt-5.4",
            "modelProvider": "openai",
            "runtimeWorkspaceRoots": ["/workspace"],
            "sessionStartSource": "startup"
        })
    );
}

#[test]
fn thread_start_dynamic_tools_use_typed_function_and_namespace_shapes() {
    let params: ThreadStartParams = serde_json::from_value(json!({
        "dynamicTools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Lookup a record",
                "inputSchema": {"type": "object"}
            },
            {
                "type": "namespace",
                "name": "docs",
                "description": "Documentation tools",
                "tools": [{
                    "type": "function",
                    "name": "search",
                    "description": "Search documentation",
                    "inputSchema": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}}
                    },
                    "deferLoading": true
                }]
            }
        ]
    }))
    .expect("typed dynamic tools");

    let tools = params.dynamic_tools.expect("dynamic tools");
    assert!(matches!(tools[0], DynamicToolSpec::Function(_)));
    let DynamicToolSpec::Namespace(namespace) = &tools[1] else {
        panic!("second dynamic tool must be a namespace");
    };
    assert_eq!(namespace.name, "docs");
    let DynamicToolNamespaceTool::Function(function) = &namespace.tools[0];
    assert_eq!(function.name, "search");
    assert!(function.defer_loading);
}

#[test]
fn thread_start_dynamic_tools_reject_untyped_values() {
    let error = serde_json::from_value::<ThreadStartParams>(json!({
        "dynamicTools": [{
            "name": "lookup",
            "description": "Lookup a record",
            "inputSchema": {"type": "object"}
        }]
    }))
    .expect_err("dynamic tool type tag is required");

    assert!(error.to_string().contains("type"));
}

#[test]
fn thread_elicitation_requests_round_trip_exact_codex_shape() {
    let thread_id = "019f9b19-17a2-78b2-84d7-ce881fcf0617";
    let requests = [
        json!({
            "id": 41,
            "method": "thread/increment_elicitation",
            "params": {"threadId": thread_id}
        }),
        json!({
            "id": 42,
            "method": "thread/decrement_elicitation",
            "params": {"threadId": thread_id}
        }),
    ];
    for expected in requests {
        let request: ClientRequest =
            serde_json::from_value(expected.clone()).expect("decode elicitation request");
        assert_eq!(
            serde_json::to_value(request).expect("encode elicitation request"),
            expected
        );
    }
    assert_eq!(
        Method::parse(METHOD_THREAD_INCREMENT_ELICITATION),
        Some(Method::ThreadIncrementElicitation)
    );
    assert_eq!(
        Method::parse(METHOD_THREAD_DECREMENT_ELICITATION),
        Some(Method::ThreadDecrementElicitation)
    );
    assert_eq!(
        serde_json::to_value(ThreadIncrementElicitationResponse {
            count: 2,
            paused: true,
        })
        .expect("encode increment response"),
        json!({"count": 2, "paused": true})
    );
    assert_eq!(
        serde_json::to_value(ThreadDecrementElicitationResponse {
            count: 0,
            paused: false,
        })
        .expect("encode decrement response"),
        json!({"count": 0, "paused": false})
    );
}

#[test]
fn thread_guardian_approval_round_trips_opaque_event_shape() {
    let expected = json!({
        "id": 43,
        "method": "thread/approveGuardianDeniedAction",
        "params": {
            "threadId": "019f9b19-17a2-78b2-84d7-ce881fcf0617",
            "event": {
                "id": "guardian-review-1",
                "status": "denied",
                "action": {
                    "type": "command",
                    "source": "shell",
                    "command": "git status --short",
                    "cwd": "/workspace"
                }
            }
        }
    });
    let request: ClientRequest =
        serde_json::from_value(expected.clone()).expect("decode Guardian approval request");

    assert_eq!(
        serde_json::to_value(request).expect("encode Guardian approval request"),
        expected
    );
    assert_eq!(
        Method::parse(METHOD_THREAD_APPROVE_GUARDIAN_DENIED_ACTION),
        Some(Method::ThreadApproveGuardianDeniedAction)
    );
    assert_eq!(
        serde_json::to_value(ThreadApproveGuardianDeniedActionResponse {})
            .expect("encode Guardian approval response"),
        json!({})
    );
}

#[test]
fn thread_inject_items_round_trips_raw_response_items() {
    let expected = json!({
        "id": 44,
        "method": "thread/inject_items",
        "params": {
            "threadId": "019f9b19-17a2-78b2-84d7-ce881fcf0617",
            "items": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "injected context"}],
                "provider_extension": {"keep": true}
            }]
        }
    });
    let request: ClientRequest =
        serde_json::from_value(expected.clone()).expect("decode thread/inject_items request");

    assert_eq!(
        serde_json::to_value(request).expect("encode thread/inject_items request"),
        expected
    );
    assert_eq!(
        Method::parse(METHOD_THREAD_INJECT_ITEMS),
        Some(Method::ThreadInjectItems)
    );
    assert_eq!(
        serde_json::to_value(ThreadInjectItemsResponse {})
            .expect("encode thread/inject_items response"),
        json!({})
    );
}

#[test]
fn artifact_write_round_trips_typed_snapshot_shape() {
    let expected = json!({
        "id": 4,
        "method": "artifact/write",
        "params": {
            "threadId": "thread_1",
            "turnId": "turn_2",
            "artifact": {
                "artifactRef": "artifact_doc_3",
                "artifactDocumentId": "doc_3",
                "path": "drafts/article.json",
                "title": "Draft",
                "kind": "artifact_document",
                "status": "ready",
                "content": "{\"schemaVersion\":\"artifact-document/v1\"}",
                "metadata": {"versionNo": 2}
            }
        }
    });

    let request: ClientRequest =
        serde_json::from_value(expected.clone()).expect("decode artifact/write request");
    assert_eq!(request.method(), Method::ArtifactWrite);
    assert_eq!(
        serde_json::to_value(request).expect("encode artifact/write request"),
        expected
    );
    assert_eq!(
        Method::parse(METHOD_ARTIFACT_WRITE),
        Some(Method::ArtifactWrite)
    );
    assert!(METHODS.contains(&METHOD_ARTIFACT_WRITE));
}

#[test]
fn media_read_round_trips_thread_scoped_shape() {
    let expected = json!({
        "id": 5,
        "method": "media/read",
        "params": {
            "threadId": "thread_1",
            "uri": "sidecar://media/image_1",
            "maxBytes": 1024,
            "offset": 0,
            "length": 512
        }
    });
    let request: ClientRequest =
        serde_json::from_value(expected.clone()).expect("decode media/read request");

    assert_eq!(request.method(), Method::MediaRead);
    assert_eq!(
        serde_json::to_value(request).expect("encode media/read request"),
        expected
    );
    assert_eq!(Method::parse(METHOD_MEDIA_READ), Some(Method::MediaRead));
    assert!(METHODS.contains(&METHOD_MEDIA_READ));
}

#[test]
fn model_list_keeps_codex_fields_with_multimodel_extensions() {
    let expected = json!({
        "id": 5,
        "method": "model/list",
        "params": {
            "cursor": "20",
            "limit": 10,
            "includeHidden": true
        }
    });
    let request: ClientRequest =
        serde_json::from_value(expected.clone()).expect("decode model/list request");
    assert_eq!(request.method(), Method::ModelList);
    assert_eq!(
        serde_json::to_value(request).expect("encode model/list request"),
        expected
    );
    assert_eq!(Method::parse(METHOD_MODEL_LIST), Some(Method::ModelList));
    assert!(METHODS.contains(&METHOD_MODEL_LIST));

    let response = ModelListResponse {
        data: vec![Model {
            id: "route:b3BlbmFp.Z3B0LTU".to_string(),
            provider_id: "openai".to_string(),
            model: "gpt-5".to_string(),
            upgrade: None,
            upgrade_info: None,
            availability_nux: None,
            display_name: "GPT-5".to_string(),
            description: "Coding model".to_string(),
            hidden: false,
            supported_reasoning_efforts: vec![ReasoningEffortOption {
                reasoning_effort: "high".to_string(),
                description: "High".to_string(),
            }],
            default_reasoning_effort: "high".to_string(),
            input_modalities: vec![InputModality::Text, InputModality::Image],
            capability_snapshot: CapabilitySnapshot {
                task_families: vec!["chat".to_string(), "reasoning".to_string()],
                input_modalities: vec!["text".to_string(), "image".to_string()],
                output_modalities: vec!["text".to_string()],
                runtime_features: vec!["streaming".to_string(), "tool_calling".to_string()],
                capabilities: ModelCapabilitiesInfo {
                    vision: true,
                    tools: true,
                    streaming: true,
                    reasoning: true,
                    ..ModelCapabilitiesInfo::default()
                },
                source: Some("provider_explicit".to_string()),
                reason_code: None,
            },
            context_window: Some(400_000),
            max_output_tokens: Some(128_000),
            supports_personality: false,
            additional_speed_tiers: Vec::new(),
            service_tiers: Vec::new(),
            default_service_tier: None,
            is_default: false,
        }],
        next_cursor: None,
    };
    assert_eq!(
        serde_json::to_value(response).expect("encode model/list response"),
        json!({
            "data": [{
                "id": "route:b3BlbmFp.Z3B0LTU",
                "providerId": "openai",
                "model": "gpt-5",
                "upgrade": null,
                "upgradeInfo": null,
                "availabilityNux": null,
                "displayName": "GPT-5",
                "description": "Coding model",
                "hidden": false,
                "supportedReasoningEfforts": [{
                    "reasoningEffort": "high",
                    "description": "High"
                }],
                "defaultReasoningEffort": "high",
                "inputModalities": ["text", "image"],
                "capabilitySnapshot": {
                    "taskFamilies": ["chat", "reasoning"],
                    "inputModalities": ["text", "image"],
                    "outputModalities": ["text"],
                    "runtimeFeatures": ["streaming", "tool_calling"],
                    "capabilities": {
                        "vision": true,
                        "tools": true,
                        "streaming": true,
                        "jsonMode": false,
                        "functionCalling": false,
                        "reasoning": true
                    },
                    "source": "provider_explicit"
                },
                "contextWindow": 400000,
                "maxOutputTokens": 128000,
                "supportsPersonality": false,
                "additionalSpeedTiers": [],
                "serviceTiers": [],
                "defaultServiceTier": null,
                "isDefault": false
            }],
            "nextCursor": null
        })
    );
}

#[test]
fn model_list_updated_notification_round_trips_typed_generation() {
    let notification = ServerNotification::ModelListUpdated(ModelListUpdatedNotification {
        generation: 17,
        provider_id: Some("openai".to_string()),
    });
    let jsonrpc: JsonRpcNotification = notification.clone().into();

    assert_eq!(jsonrpc.method, METHOD_MODEL_LIST_UPDATED);
    assert_eq!(
        jsonrpc.params,
        Some(json!({
            "generation": 17,
            "providerId": "openai"
        }))
    );
    assert_eq!(
        ServerNotification::try_from(jsonrpc).expect("typed model list update"),
        notification
    );
    assert!(NOTIFICATION_METHODS.contains(&METHOD_MODEL_LIST_UPDATED));
}

#[test]
fn config_warning_notification_round_trips_codex_shape() {
    let expected = json!({
        "method": "configWarning",
        "params": {
            "summary": "Invalid configuration; using defaults.",
            "details": "failed to parse config.toml",
            "path": "/tmp/config.toml",
            "range": {
                "start": { "line": 2, "column": 5 },
                "end": { "line": 2, "column": 12 }
            }
        }
    });

    let notification: ServerNotification =
        serde_json::from_value(expected.clone()).expect("decode config warning notification");
    assert_eq!(notification.method(), METHOD_CONFIG_WARNING);
    let raw: crate::JsonRpcNotification = notification.clone().into();
    assert_eq!(raw.method, METHOD_CONFIG_WARNING);
    assert_eq!(
        ServerNotification::try_from(raw).expect("decode JSON-RPC config warning notification"),
        notification
    );
    assert_eq!(
        serde_json::to_value(notification).expect("encode config warning notification"),
        expected
    );
    assert!(NOTIFICATION_METHODS.contains(&METHOD_CONFIG_WARNING));

    assert_eq!(
        serde_json::to_value(ConfigWarningNotification {
            summary: "Using defaults.".to_string(),
            details: None,
            path: None,
            range: None,
        })
        .expect("encode nullable config warning details"),
        json!({
            "summary": "Using defaults.",
            "details": null
        })
    );
}

#[test]
fn warning_notification_round_trips_thread_scoped_lime_shape() {
    let expected = json!({
        "method": "warning",
        "params": {
            "threadId": "thread-1",
            "message": "技能不可用，已继续执行。",
            "code": "skill_not_available"
        }
    });

    let notification: ServerNotification =
        serde_json::from_value(expected.clone()).expect("decode warning notification");
    assert_eq!(notification.method(), METHOD_WARNING);
    let raw: crate::JsonRpcNotification = notification.clone().into();
    assert_eq!(raw.method, METHOD_WARNING);
    assert_eq!(
        ServerNotification::try_from(raw).expect("decode JSON-RPC warning notification"),
        notification
    );
    assert_eq!(
        serde_json::to_value(notification).expect("encode warning notification"),
        expected
    );
    assert!(NOTIFICATION_METHODS.contains(&METHOD_WARNING));

    assert_eq!(
        serde_json::to_value(WarningNotification {
            thread_id: None,
            message: "全局提醒".to_string(),
            code: None,
        })
        .expect("encode global warning notification"),
        json!({
            "threadId": null,
            "message": "全局提醒"
        })
    );
    assert!(serde_json::from_value::<ServerNotification>(json!({
        "method": "warning",
        "params": { "threadId": "thread-1" }
    }))
    .is_err());
    assert!(serde_json::from_value::<ServerNotification>(json!({
        "method": "warning",
        "params": {
            "threadId": "thread-1",
            "message": "warning",
            "code": 42
        }
    }))
    .is_err());
}

#[test]
fn thread_search_occurrences_round_trips_exact_codex_shape() {
    let expected = json!({
        "id": 6,
        "method": "thread/searchOccurrences",
        "params": {
            "threadId": "019f9b19-17a2-78b2-84d7-ce881fcf0617",
            "searchTerm": "needle",
            "cursor": null,
            "limit": 50
        }
    });
    let request: ClientRequest =
        serde_json::from_value(expected.clone()).expect("decode thread/searchOccurrences request");
    assert_eq!(request.method(), Method::ThreadSearchOccurrences);
    assert_eq!(
        serde_json::to_value(request).expect("encode thread/searchOccurrences request"),
        expected
    );
    assert_eq!(
        Method::parse(METHOD_THREAD_SEARCH_OCCURRENCES),
        Some(Method::ThreadSearchOccurrences)
    );
    assert!(METHODS.contains(&METHOD_THREAD_SEARCH_OCCURRENCES));

    let response = ThreadSearchOccurrencesResponse {
        data: vec![ThreadSearchOccurrence {
            turn_id: "turn_1".to_string(),
            item_id: "item_1".to_string(),
            snippet: "The needle is here.".to_string(),
            snippet_match_range: ThreadSearchTextRange { start: 4, end: 10 },
            turn_cursor: "opaque-inclusive-turn-cursor".to_string(),
        }],
        next_cursor: None,
    };
    assert_eq!(
        serde_json::to_value(response).expect("encode thread/searchOccurrences response"),
        json!({
            "data": [{
                "turnId": "turn_1",
                "itemId": "item_1",
                "snippet": "The needle is here.",
                "snippetMatchRange": {"start": 4, "end": 10},
                "turnCursor": "opaque-inclusive-turn-cursor"
            }],
            "nextCursor": null
        })
    );
}

#[test]
fn thread_search_round_trips_exact_codex_request_shape() {
    let expected = json!({
        "id": 7,
        "method": "thread/search",
        "params": {
            "cursor": "opaque-thread-cursor",
            "limit": 25,
            "sortKey": "updated_at",
            "sortDirection": "asc",
            "sourceKinds": ["appServer", "subAgentReview"],
            "archived": true,
            "searchTerm": "needle"
        }
    });
    let request: ClientRequest =
        serde_json::from_value(expected.clone()).expect("decode thread/search request");
    assert_eq!(request.method(), Method::ThreadSearch);
    assert_eq!(
        serde_json::to_value(request).expect("encode thread/search request"),
        expected
    );
    assert_eq!(
        Method::parse(METHOD_THREAD_SEARCH),
        Some(Method::ThreadSearch)
    );
    assert!(METHODS.contains(&METHOD_THREAD_SEARCH));
}

#[test]
fn thread_background_terminals_round_trip_exact_codex_shapes() {
    let thread_id = "019f9b19-17a2-78b2-84d7-ce881fcf0617";
    let requests = [
        json!({
            "id": 8,
            "method": "thread/backgroundTerminals/clean",
            "params": {"threadId": thread_id}
        }),
        json!({
            "id": 9,
            "method": "thread/backgroundTerminals/list",
            "params": {"threadId": thread_id, "cursor": null, "limit": 25}
        }),
        json!({
            "id": 10,
            "method": "thread/backgroundTerminals/terminate",
            "params": {"threadId": thread_id, "processId": "42"}
        }),
    ];
    for expected in requests {
        let request: ClientRequest = serde_json::from_value(expected.clone())
            .expect("decode thread/backgroundTerminals request");
        assert_eq!(
            serde_json::to_value(request).expect("encode thread/backgroundTerminals request"),
            expected
        );
    }
    assert_eq!(
        Method::parse(METHOD_THREAD_BACKGROUND_TERMINALS_CLEAN),
        Some(Method::ThreadBackgroundTerminalsClean)
    );
    assert_eq!(
        Method::parse(METHOD_THREAD_BACKGROUND_TERMINALS_LIST),
        Some(Method::ThreadBackgroundTerminalsList)
    );
    assert_eq!(
        Method::parse(METHOD_THREAD_BACKGROUND_TERMINALS_TERMINATE),
        Some(Method::ThreadBackgroundTerminalsTerminate)
    );

    let response = ThreadBackgroundTerminalsListResponse {
        data: vec![ThreadBackgroundTerminal {
            item_id: "item_456".to_string(),
            process_id: "42".to_string(),
            command: "python3 -m http.server".to_string(),
            cwd: "/workspace".to_string(),
            os_pid: None,
            cpu_percent: None,
            rss_kb: None,
        }],
        next_cursor: None,
    };
    assert_eq!(
        serde_json::to_value(response).expect("encode background terminal list response"),
        json!({
            "data": [{
                "itemId": "item_456",
                "processId": "42",
                "command": "python3 -m http.server",
                "cwd": "/workspace",
                "osPid": null,
                "cpuPercent": null,
                "rssKb": null
            }],
            "nextCursor": null
        })
    );
}

#[test]
fn thread_fork_round_trips_codex_goal_deferral_fields() {
    let expected = json!({
        "id": 2,
        "method": "thread/fork",
        "params": {
            "threadId": "thread_1",
            "lastTurnId": "turn_2",
            "excludeTurns": true,
            "deferGoalContinuation": true
        }
    });
    let request: ClientRequest =
        serde_json::from_value(expected.clone()).expect("decode thread/fork request");
    assert_eq!(request.method(), Method::ThreadFork);
    assert_eq!(
        serde_json::to_value(request).expect("encode thread/fork request"),
        expected
    );
    assert_eq!(Method::parse(METHOD_THREAD_FORK), Some(Method::ThreadFork));
    assert!(METHODS.contains(&METHOD_THREAD_FORK));
}

#[test]
fn thread_compact_start_round_trips_exact_codex_shape() {
    let expected = json!({
        "id": 3,
        "method": "thread/compact/start",
        "params": {
            "threadId": "thread_1"
        }
    });
    let request: ClientRequest =
        serde_json::from_value(expected.clone()).expect("decode thread/compact/start request");
    assert_eq!(request.method(), Method::ThreadCompactStart);
    assert_eq!(
        serde_json::to_value(request).expect("encode thread/compact/start request"),
        expected
    );
    assert_eq!(
        Method::parse(METHOD_THREAD_COMPACT_START),
        Some(Method::ThreadCompactStart)
    );
    assert!(METHODS.contains(&METHOD_THREAD_COMPACT_START));
    assert_eq!(
        serde_json::to_value(ThreadCompactStartResponse {}).expect("encode empty response"),
        json!({})
    );
}

#[test]
fn thread_name_set_round_trips_exact_codex_shape() {
    let expected = json!({
        "id": 4,
        "method": "thread/name/set",
        "params": {
            "threadId": "thread_1",
            "name": "Renamed thread"
        }
    });
    let request: ClientRequest =
        serde_json::from_value(expected.clone()).expect("decode thread/name/set request");
    assert_eq!(request.method(), Method::ThreadSetName);
    assert_eq!(
        serde_json::to_value(request).expect("encode thread/name/set request"),
        expected
    );
    assert_eq!(
        Method::parse(METHOD_THREAD_NAME_SET),
        Some(Method::ThreadSetName)
    );
    assert!(METHODS.contains(&METHOD_THREAD_NAME_SET));
    assert_eq!(
        serde_json::to_value(ThreadSetNameResponse {}).expect("encode empty response"),
        json!({})
    );
}

#[test]
fn thread_metadata_update_round_trips_exact_codex_shape() {
    let expected = json!({
        "id": 5,
        "method": "thread/metadata/update",
        "params": {
            "threadId": "019bf4f0-5080-7000-8000-000000000001",
            "gitInfo": {
                "sha": "abc123",
                "branch": null,
                "originUrl": "https://example.test/repo.git"
            },
            "isPinned": true
        }
    });
    let request: ClientRequest =
        serde_json::from_value(expected.clone()).expect("decode thread/metadata/update request");
    assert_eq!(request.method(), Method::ThreadMetadataUpdate);
    assert_eq!(
        serde_json::to_value(request).expect("encode thread/metadata/update request"),
        expected
    );
    assert_eq!(
        Method::parse(METHOD_THREAD_METADATA_UPDATE),
        Some(Method::ThreadMetadataUpdate)
    );
    assert!(METHODS.contains(&METHOD_THREAD_METADATA_UPDATE));
}

#[test]
fn thread_metadata_git_patch_distinguishes_omission_null_and_value() {
    let params: ThreadMetadataUpdateParams = serde_json::from_value(json!({
        "threadId": "019bf4f0-5080-7000-8000-000000000001",
        "gitInfo": {
            "sha": null,
            "branch": "main"
        }
    }))
    .expect("decode git metadata patch");
    let git_info = params.git_info.expect("gitInfo");
    assert_eq!(git_info.sha, Some(None));
    assert_eq!(git_info.branch, Some(Some("main".to_string())));
    assert_eq!(git_info.origin_url, None);
}

#[test]
fn thread_loaded_list_round_trips_exact_codex_shape() {
    let expected = json!({
        "id": 5,
        "method": "thread/loaded/list",
        "params": {
            "cursor": "019bf4f0-5080-7000-8000-000000000001",
            "limit": 20
        }
    });
    let request: ClientRequest =
        serde_json::from_value(expected.clone()).expect("decode thread/loaded/list request");
    assert_eq!(request.method(), Method::ThreadLoadedList);
    assert_eq!(
        serde_json::to_value(request).expect("encode thread/loaded/list request"),
        expected
    );
    assert_eq!(
        Method::parse(METHOD_THREAD_LOADED_LIST),
        Some(Method::ThreadLoadedList)
    );
    assert!(METHODS.contains(&METHOD_THREAD_LOADED_LIST));
}

#[test]
fn thread_unsubscribe_round_trips_exact_codex_shape() {
    let expected = json!({
        "id": 6,
        "method": "thread/unsubscribe",
        "params": {
            "threadId": "019bf4f0-5080-7000-8000-000000000001"
        }
    });
    let request: ClientRequest =
        serde_json::from_value(expected.clone()).expect("decode thread/unsubscribe request");
    assert_eq!(request.method(), Method::ThreadUnsubscribe);
    assert_eq!(
        serde_json::to_value(request).expect("encode thread/unsubscribe request"),
        expected
    );
    assert_eq!(
        Method::parse(METHOD_THREAD_UNSUBSCRIBE),
        Some(Method::ThreadUnsubscribe)
    );
    assert!(METHODS.contains(&METHOD_THREAD_UNSUBSCRIBE));
    assert_eq!(
        serde_json::to_value(ThreadUnsubscribeResponse {
            status: ThreadUnsubscribeStatus::NotLoaded,
        })
        .expect("encode thread/unsubscribe response"),
        json!({"status": "notLoaded"})
    );
}

#[test]
fn thread_closed_notification_round_trips_exact_codex_shape() {
    let expected = json!({
        "method": "thread/closed",
        "params": {
            "threadId": "019bf4f0-5080-7000-8000-000000000001"
        }
    });

    let notification: ServerNotification =
        serde_json::from_value(expected.clone()).expect("decode thread/closed notification");
    assert_eq!(notification.method(), METHOD_THREAD_CLOSED);
    let raw: crate::JsonRpcNotification = notification.clone().into();
    assert_eq!(raw.method, METHOD_THREAD_CLOSED);
    assert_eq!(
        ServerNotification::try_from(raw).expect("decode JSON-RPC thread/closed notification"),
        notification
    );
    assert_eq!(
        serde_json::to_value(notification).expect("encode thread/closed notification"),
        expected
    );
    assert!(NOTIFICATION_METHODS.contains(&METHOD_THREAD_CLOSED));
}

#[test]
fn thread_resume_round_trips_exclude_turns_and_initial_page() {
    let value = json!({
        "threadId": "thread_1",
        "modelProvider": "openai",
        "excludeTurns": true,
        "initialTurnsPage": {
            "limit": 20,
            "sortDirection": "desc",
            "itemsView": "summary"
        }
    });

    let params: ThreadResumeParams =
        serde_json::from_value(value.clone()).expect("deserialize thread/resume params");
    assert!(params.exclude_turns);
    assert_eq!(params.thread_id, "thread_1");
    assert_eq!(
        serde_json::to_value(params).expect("serialize thread/resume params"),
        value
    );
}

#[test]
fn thread_token_usage_notification_round_trips_codex_shape() {
    let expected = json!({
        "method": "thread/tokenUsage/updated",
        "params": {
            "threadId": "thread_1",
            "turnId": "turn_2",
            "tokenUsage": {
                "total": {
                    "totalTokens": 120,
                    "inputTokens": 90,
                    "cachedInputTokens": 30,
                    "cacheWriteInputTokens": 12,
                    "outputTokens": 30,
                    "reasoningOutputTokens": 10
                },
                "last": {
                    "totalTokens": 60,
                    "inputTokens": 45,
                    "cachedInputTokens": 15,
                    "cacheWriteInputTokens": 6,
                    "outputTokens": 15,
                    "reasoningOutputTokens": 5
                },
                "modelContextWindow": 128000
            }
        }
    });

    let notification: ServerNotification =
        serde_json::from_value(expected.clone()).expect("decode token usage notification");
    assert_eq!(notification.method(), "thread/tokenUsage/updated");
    let jsonrpc_notification: crate::JsonRpcNotification = notification.clone().into();
    assert_eq!(jsonrpc_notification.method, "thread/tokenUsage/updated");
    assert_eq!(
        ServerNotification::try_from(jsonrpc_notification)
            .expect("decode JSON-RPC token usage notification"),
        notification
    );
    assert_eq!(
        serde_json::to_value(notification).expect("encode token usage notification"),
        expected
    );
}

#[test]
fn model_safety_buffering_notification_round_trips_codex_shape() {
    let expected = json!({
        "method": "model/safetyBuffering/updated",
        "params": {
            "threadId": "thread_1",
            "turnId": "turn_2",
            "model": "gpt-5-codex",
            "useCases": ["policy"],
            "reasons": ["buffering"],
            "showBufferingUi": true,
            "fasterModel": "gpt-5-mini"
        }
    });

    let notification: ServerNotification =
        serde_json::from_value(expected.clone()).expect("decode safety buffering notification");
    assert_eq!(notification.method(), METHOD_MODEL_SAFETY_BUFFERING_UPDATED);
    let raw: crate::JsonRpcNotification = notification.clone().into();
    assert_eq!(raw.method, METHOD_MODEL_SAFETY_BUFFERING_UPDATED);
    assert_eq!(
        ServerNotification::try_from(raw).expect("decode JSON-RPC safety buffering notification"),
        notification
    );
    assert_eq!(
        serde_json::to_value(notification).expect("encode safety buffering notification"),
        expected
    );
    assert!(NOTIFICATION_METHODS.contains(&METHOD_MODEL_SAFETY_BUFFERING_UPDATED));
}

#[test]
fn model_rerouted_notification_round_trips_codex_shape() {
    let expected = json!({
        "method": "model/rerouted",
        "params": {
            "threadId": "thread_1",
            "turnId": "turn_2",
            "fromModel": "gpt-5-codex",
            "toModel": "gpt-5.1-codex",
            "reason": "highRiskCyberActivity"
        }
    });

    let notification: ServerNotification =
        serde_json::from_value(expected.clone()).expect("decode model rerouted notification");
    assert_eq!(notification.method(), METHOD_MODEL_REROUTED);
    let raw: crate::JsonRpcNotification = notification.clone().into();
    assert_eq!(raw.method, METHOD_MODEL_REROUTED);
    assert_eq!(
        ServerNotification::try_from(raw).expect("decode JSON-RPC model rerouted notification"),
        notification
    );
    assert_eq!(
        serde_json::to_value(notification).expect("encode model rerouted notification"),
        expected
    );
    assert!(NOTIFICATION_METHODS.contains(&METHOD_MODEL_REROUTED));
}

#[test]
fn model_verification_notification_round_trips_codex_shape() {
    let expected = json!({
        "method": "model/verification",
        "params": {
            "threadId": "thread_1",
            "turnId": "turn_2",
            "verifications": ["trustedAccessForCyber"]
        }
    });

    let notification: ServerNotification =
        serde_json::from_value(expected.clone()).expect("decode model verification notification");
    assert_eq!(notification.method(), METHOD_MODEL_VERIFICATION);
    let raw: crate::JsonRpcNotification = notification.clone().into();
    assert_eq!(raw.method, METHOD_MODEL_VERIFICATION);
    assert_eq!(
        ServerNotification::try_from(raw).expect("decode JSON-RPC model verification notification"),
        notification
    );
    assert_eq!(
        serde_json::to_value(notification).expect("encode model verification notification"),
        expected
    );
    assert!(NOTIFICATION_METHODS.contains(&METHOD_MODEL_VERIFICATION));
}

#[test]
fn thread_status_changed_notification_round_trips_codex_shape() {
    let expected = json!({
        "method": "thread/status/changed",
        "params": {
            "threadId": "thread_1",
            "status": {
                "type": "active",
                "activeFlags": ["waitingOnApproval", "waitingOnUserInput"]
            }
        }
    });

    let notification: ServerNotification =
        serde_json::from_value(expected.clone()).expect("decode thread status notification");
    assert_eq!(notification.method(), METHOD_THREAD_STATUS_CHANGED);
    let raw: crate::JsonRpcNotification = notification.clone().into();
    assert_eq!(raw.method, METHOD_THREAD_STATUS_CHANGED);
    assert_eq!(
        ServerNotification::try_from(raw).expect("decode JSON-RPC thread status notification"),
        notification
    );
    assert_eq!(
        serde_json::to_value(notification).expect("encode thread status notification"),
        expected
    );
    assert!(NOTIFICATION_METHODS.contains(&METHOD_THREAD_STATUS_CHANGED));
}

#[test]
fn plan_delta_notification_round_trips_codex_shape() {
    let expected = json!({
        "method": "item/plan/delta",
        "params": {
            "threadId": "thread_1",
            "turnId": "turn_2",
            "itemId": "plan_3",
            "delta": "- [ ] verify typed plan delta"
        }
    });

    let notification: ServerNotification =
        serde_json::from_value(expected.clone()).expect("decode plan delta notification");
    assert_eq!(notification.method(), METHOD_PLAN_DELTA);
    let raw: crate::JsonRpcNotification = notification.clone().into();
    assert_eq!(raw.method, METHOD_PLAN_DELTA);
    assert_eq!(
        ServerNotification::try_from(raw).expect("decode JSON-RPC plan delta notification"),
        notification
    );
    assert_eq!(
        serde_json::to_value(notification).expect("encode plan delta notification"),
        expected
    );
    assert!(NOTIFICATION_METHODS.contains(&METHOD_PLAN_DELTA));
}

#[test]
fn command_execution_output_delta_notification_round_trips_codex_shape() {
    let expected = json!({
        "method": "item/commandExecution/outputDelta",
        "params": {
            "threadId": "thread_1",
            "turnId": "turn_2",
            "itemId": "command_3",
            "delta": "stdout\n"
        }
    });

    let notification: ServerNotification =
        serde_json::from_value(expected.clone()).expect("decode command output delta");
    assert_eq!(notification.method(), METHOD_COMMAND_EXECUTION_OUTPUT_DELTA);
    let raw: crate::JsonRpcNotification = notification.clone().into();
    assert_eq!(raw.method, METHOD_COMMAND_EXECUTION_OUTPUT_DELTA);
    assert_eq!(
        ServerNotification::try_from(raw).expect("decode JSON-RPC command output delta"),
        notification
    );
    assert_eq!(
        serde_json::to_value(notification).expect("encode command output delta"),
        expected
    );
    assert!(NOTIFICATION_METHODS.contains(&METHOD_COMMAND_EXECUTION_OUTPUT_DELTA));
}

#[test]
fn file_change_patch_updated_notification_round_trips_codex_shape() {
    let expected = json!({
        "method": "item/fileChange/patchUpdated",
        "params": {
            "threadId": "thread_1",
            "turnId": "turn_2",
            "itemId": "patch_3",
            "changes": [
                {
                    "path": "src/lib.rs",
                    "kind": { "type": "update", "move_path": "src/main.rs" },
                    "diff": "-old\n+new"
                }
            ]
        }
    });

    let notification: ServerNotification =
        serde_json::from_value(expected.clone()).expect("decode file change patch update");
    assert_eq!(notification.method(), METHOD_FILE_CHANGE_PATCH_UPDATED);
    let raw: crate::JsonRpcNotification = notification.clone().into();
    assert_eq!(raw.method, METHOD_FILE_CHANGE_PATCH_UPDATED);
    assert_eq!(
        ServerNotification::try_from(raw).expect("decode JSON-RPC file change patch update"),
        notification
    );
    assert_eq!(
        serde_json::to_value(notification).expect("encode file change patch update"),
        expected
    );
    assert!(NOTIFICATION_METHODS.contains(&METHOD_FILE_CHANGE_PATCH_UPDATED));
}

#[test]
fn mcp_tool_call_progress_notification_round_trips_codex_shape() {
    let expected = json!({
        "method": "item/mcpToolCall/progress",
        "params": {
            "threadId": "thread_1",
            "turnId": "turn_2",
            "itemId": "item_mcp-call-3",
            "message": "正在检索文档"
        }
    });

    let notification: ServerNotification =
        serde_json::from_value(expected.clone()).expect("decode MCP tool call progress");
    assert_eq!(notification.method(), METHOD_MCP_TOOL_CALL_PROGRESS);
    let raw: crate::JsonRpcNotification = notification.clone().into();
    assert_eq!(raw.method, METHOD_MCP_TOOL_CALL_PROGRESS);
    assert_eq!(
        ServerNotification::try_from(raw).expect("decode JSON-RPC MCP tool call progress"),
        notification
    );
    assert_eq!(
        serde_json::to_value(notification).expect("encode MCP tool call progress"),
        expected
    );
    assert!(NOTIFICATION_METHODS.contains(&METHOD_MCP_TOOL_CALL_PROGRESS));
}

#[test]
fn server_request_resolved_notification_round_trips_codex_shape() {
    let notification =
        ServerNotification::ServerRequestResolved(ServerRequestResolvedNotification {
            thread_id: "thread-1".to_string(),
            request_id: crate::RequestId::String("app-server-request:boot:7".to_string()),
        });

    let raw: crate::JsonRpcNotification = notification.clone().into();
    assert_eq!(raw.method, METHOD_SERVER_REQUEST_RESOLVED);
    assert_eq!(
        raw.params.as_ref(),
        Some(&json!({
            "threadId": "thread-1",
            "requestId": "app-server-request:boot:7"
        }))
    );
    assert_eq!(
        ServerNotification::try_from(raw).expect("decode resolved notification"),
        notification
    );
}

#[test]
fn patch_change_kind_uses_codex_tagged_wire_shape() {
    assert_eq!(
        serde_json::to_value(PatchChangeKind::Add).expect("serialize add"),
        json!({"type": "add"})
    );
    assert_eq!(
        serde_json::to_value(PatchChangeKind::Update {
            move_path: Some("target.txt".to_string()),
        })
        .expect("serialize move update"),
        json!({"type": "update", "move_path": "target.txt"})
    );
}

#[test]
fn thread_goal_set_preserves_nullable_patch_fields() {
    let omitted: ThreadGoalSetParams = serde_json::from_value(json!({
        "threadId": "thread_1"
    }))
    .expect("decode omitted goal patch");
    assert_eq!(omitted.objective, None);
    assert_eq!(omitted.status, None);
    assert_eq!(omitted.token_budget, None);

    let cleared: ThreadGoalSetParams = serde_json::from_value(json!({
        "threadId": "thread_1",
        "objective": null,
        "status": null,
        "tokenBudget": null
    }))
    .expect("decode cleared goal patch");
    assert_eq!(cleared.objective, None);
    assert_eq!(cleared.status, None);
    assert_eq!(cleared.token_budget, Some(None));

    let selected: ThreadGoalSetParams = serde_json::from_value(json!({
        "threadId": "thread_1",
        "objective": "ship it",
        "status": "active",
        "tokenBudget": 100
    }))
    .expect("decode selected goal patch");
    assert_eq!(selected.objective.as_deref(), Some("ship it"));
    assert_eq!(selected.status, Some(ThreadGoalStatus::Active));
    assert_eq!(selected.token_budget, Some(Some(100)));
}

#[test]
fn thread_goal_methods_and_notifications_round_trip() {
    let set: ClientRequest = serde_json::from_value(json!({
        "id": 1,
        "method": "thread/goal/set",
        "params": {"threadId": "thread_1", "objective": "ship it"}
    }))
    .expect("decode goal set");
    assert_eq!(set.method(), Method::ThreadGoalSet);

    let get: ClientRequest = serde_json::from_value(json!({
        "id": 2,
        "method": "thread/goal/get",
        "params": {"threadId": "thread_1"}
    }))
    .expect("decode goal get");
    assert_eq!(get.method(), Method::ThreadGoalGet);

    let clear: ClientRequest = serde_json::from_value(json!({
        "id": 3,
        "method": "thread/goal/clear",
        "params": {"threadId": "thread_1"}
    }))
    .expect("decode goal clear");
    assert_eq!(clear.method(), Method::ThreadGoalClear);

    let updated = json!({
        "method": "thread/goal/updated",
        "params": {
            "threadId": "thread_1",
            "turnId": null,
            "goal": {
                "threadId": "thread_1",
                "objective": "ship it",
                "status": "active",
                "tokenBudget": null,
                "tokensUsed": 0,
                "timeUsedSeconds": 0,
                "createdAt": 1,
                "updatedAt": 1
            }
        }
    });
    let notification: ServerNotification =
        serde_json::from_value(updated.clone()).expect("decode goal update");
    assert_eq!(notification.method(), "thread/goal/updated");
    assert_eq!(
        serde_json::to_value(notification).expect("encode goal update"),
        updated
    );

    let cleared = json!({
        "method": "thread/goal/cleared",
        "params": {"threadId": "thread_1"}
    });
    let notification: ServerNotification =
        serde_json::from_value(cleared.clone()).expect("decode goal clear notification");
    assert_eq!(notification.method(), "thread/goal/cleared");
    assert_eq!(
        serde_json::to_value(notification).expect("encode goal clear"),
        cleared
    );
}

#[test]
fn thread_archive_contract_matches_v2_shapes() {
    let archive: ClientRequest = serde_json::from_value(json!({
        "id": 7,
        "method": "thread/archive",
        "params": {"threadId": "thread_1"}
    }))
    .expect("deserialize thread/archive request");
    assert_eq!(archive.method(), Method::ThreadArchive);
    assert_eq!(
        serde_json::to_value(ThreadArchiveResponse {}).expect("serialize archive response"),
        json!({})
    );

    let delete: ClientRequest = serde_json::from_value(json!({
        "id": 8,
        "method": "thread/delete",
        "params": {"threadId": "thread_1"}
    }))
    .expect("deserialize thread/delete request");
    assert_eq!(delete.method(), Method::ThreadDelete);
    assert_eq!(
        serde_json::to_value(ThreadDeleteResponse {}).expect("serialize delete response"),
        json!({})
    );

    let unarchive: ClientRequest = serde_json::from_value(json!({
        "id": 9,
        "method": "thread/unarchive",
        "params": {"threadId": "thread_1"}
    }))
    .expect("deserialize thread/unarchive request");
    assert_eq!(unarchive.method(), Method::ThreadUnarchive);

    for expected in [
        json!({
            "method": "thread/archived",
            "params": {"threadId": "thread_1"}
        }),
        json!({
            "method": "thread/deleted",
            "params": {"threadId": "thread_1"}
        }),
        json!({
            "method": "thread/unarchived",
            "params": {"threadId": "thread_1"}
        }),
    ] {
        let notification: ServerNotification = serde_json::from_value(expected.clone())
            .expect("deserialize archive lifecycle notification");
        assert_eq!(notification.method(), expected["method"]);
        assert_eq!(
            serde_json::to_value(notification).expect("serialize archive lifecycle notification"),
            expected
        );
    }
}

#[test]
fn turn_start_and_steer_preserve_canonical_metadata_fields() {
    let start = TurnStartParams {
        thread_id: "thread_1".to_string(),
        client_user_message_id: Some("msg_1".to_string()),
        input: vec![UserInput::Text {
            text: "hello".to_string(),
            text_elements: vec![],
        }],
        responsesapi_client_metadata: Some([(String::from("source"), String::from("gui"))].into()),
        additional_context: Some(
            [(
                String::from("doc"),
                AdditionalContextEntry {
                    value: "untrusted excerpt".to_string(),
                    kind: AdditionalContextKind::Untrusted,
                },
            )]
            .into(),
        ),
        ..TurnStartParams::default()
    };
    let value = serde_json::to_value(start).expect("serialize turn/start params");
    assert_eq!(value["threadId"], "thread_1");
    assert_eq!(value["clientUserMessageId"], "msg_1");
    assert_eq!(value["responsesapiClientMetadata"]["source"], "gui");
    assert_eq!(value["additionalContext"]["doc"]["kind"], "untrusted");

    let steer: TurnSteerParams = serde_json::from_value(json!({
        "threadId": "thread_1",
        "input": [{"type": "text", "text": "continue"}],
        "expectedTurnId": "turn_1"
    }))
    .expect("deserialize turn/steer params");
    assert_eq!(steer.expected_turn_id, "turn_1");
}

#[test]
fn multi_agent_mode_uses_typed_codex_wire_and_rejects_arbitrary_json() {
    let params: TurnStartParams = serde_json::from_value(json!({
        "threadId": "thread_1",
        "input": [{"type": "text", "text": "delegate work"}],
        "multiAgentMode": "proactive"
    }))
    .expect("typed proactive mode");
    assert_eq!(
        params.multi_agent_mode,
        Some(agent_protocol::MultiAgentMode::Proactive)
    );
    assert_eq!(
        serde_json::to_value(params).expect("serialize params")["multiAgentMode"],
        "proactive"
    );

    let error = serde_json::from_value::<TurnStartParams>(json!({
        "threadId": "thread_1",
        "input": [{"type": "text", "text": "delegate work"}],
        "multiAgentMode": {"enabled": true}
    }))
    .expect_err("arbitrary multi-agent JSON must fail closed");
    assert!(error.to_string().contains("unknown variant"));
}

#[test]
fn turn_interrupt_wire_shape_is_canonical() {
    let value = serde_json::to_value(TurnInterruptParams {
        thread_id: "thread_1".to_string(),
        turn_id: "turn_1".to_string(),
    })
    .expect("serialize turn/interrupt params");

    assert_eq!(value, json!({"threadId": "thread_1", "turnId": "turn_1"}));
}

#[test]
fn v2_method_registry_round_trips_wire_names() {
    for method in METHODS {
        let parsed = Method::parse(method).expect("registered method");
        assert_eq!(parsed.as_str(), *method);
        let wire = serde_json::to_value(parsed).expect("serialize method");
        assert_eq!(wire, json!(method));
    }
    assert_eq!(
        SERVER_REQUEST_METHODS,
        &[
            METHOD_CURRENT_TIME_READ,
            METHOD_MCP_SERVER_ELICITATION_REQUEST,
            METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
            METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
            METHOD_ITEM_PERMISSIONS_REQUEST_APPROVAL,
            METHOD_ITEM_TOOL_CALL,
            METHOD_ITEM_TOOL_REQUEST_USER_INPUT,
        ]
    );
    assert_eq!(Method::parse("agentSession/turn/start"), None);
}

#[test]
fn typed_thread_items_round_trip_v2_variant_tags() {
    let cases = [
        json!({
            "type": "userMessage",
            "id": "item_user",
            "content": [{"type": "text", "text": "hello"}]
        }),
        json!({
            "type": "hookPrompt",
            "id": "item_hook",
            "fragments": [{"text": "policy", "hookRunId": "hook_1"}]
        }),
        json!({
            "type": "commandExecution",
            "id": "item_command",
            "command": "ls",
            "cwd": "/workspace",
            "source": "unifiedExecStartup",
            "status": "completed",
            "commandActions": [{"type": "unknown", "command": "ls"}],
            "exitCode": 0,
            "durationMs": 4
        }),
        json!({
            "type": "dynamicToolCall",
            "id": "item_dynamic",
            "tool": "search",
            "arguments": {"query": "runtime"},
            "status": "completed",
            "success": true
        }),
        json!({
            "type": "contextCompaction",
            "id": "item_compaction"
        }),
    ];

    for value in cases {
        let item: ThreadItem = serde_json::from_value(value.clone()).expect("typed item");
        assert_eq!(serde_json::to_value(item).expect("serialized item"), value);
    }
}

#[test]
fn initialize_capabilities_preserve_connection_notification_opt_out() {
    let value = serde_json::to_value(InitializeCapabilities {
        experimental_api: true,
        request_attestation: true,
        mcp_server_openai_form_elicitation: true,
        opt_out_notification_methods: Some(vec![
            "thread/started".to_string(),
            "item/agentMessage/delta".to_string(),
        ]),
    })
    .expect("serialize initialize capabilities");

    assert_eq!(
        value,
        json!({
            "experimentalApi": true,
            "requestAttestation": true,
            "mcpServerOpenaiFormElicitation": true,
            "optOutNotificationMethods": [
                "thread/started",
                "item/agentMessage/delta"
            ]
        })
    );
}

#[test]
fn typed_v2_request_and_standard_response_round_trip() {
    let request = ClientRequest::TurnSteer {
        id: crate::RequestId::String("req_1".to_string()),
        params: TurnSteerParams {
            thread_id: "thread_1".to_string(),
            input: vec![UserInput::Text {
                text: "continue".to_string(),
                text_elements: vec![],
            }],
            expected_turn_id: "turn_1".to_string(),
            ..TurnSteerParams::default()
        },
    };
    let request_value = serde_json::to_value(&request).expect("serialize typed request");
    assert_eq!(request_value["method"], "turn/steer");
    assert_eq!(request_value["id"], "req_1");
    let decoded_request: ClientRequest =
        serde_json::from_value(request_value).expect("deserialize typed request");
    assert_eq!(decoded_request.method(), Method::TurnSteer);

    let response = ClientResponsePayload::TurnSteer(TurnSteerResponse {
        turn_id: "turn_1".to_string(),
    })
    .into_response(crate::RequestId::String("req_1".to_string()))
    .expect("lower typed response");
    let response_value = serde_json::to_value(&response).expect("serialize JSON-RPC response");
    assert_eq!(
        response_value,
        json!({"id": "req_1", "result": {"turnId": "turn_1"}})
    );
    let decoded_response: ClientResponse =
        serde_json::from_value(response_value).expect("deserialize typed response");
    assert_eq!(
        decoded_response.id,
        crate::RequestId::String("req_1".to_string())
    );
    assert_eq!(decoded_response.result["turnId"], "turn_1");
}

#[test]
fn typed_v2_server_envelopes_fail_closed_for_unknown_methods() {
    let current_time_value = json!({
        "id": "current-time-1",
        "method": "currentTime/read",
        "params": { "threadId": "thread_1" }
    });
    let current_time: ServerRequest =
        serde_json::from_value(current_time_value.clone()).expect("decode current-time request");
    assert_eq!(current_time.method(), METHOD_CURRENT_TIME_READ);
    let current_time_jsonrpc: crate::JsonRpcRequest = current_time.clone().into();
    assert_eq!(
        ServerRequest::try_from(current_time_jsonrpc)
            .expect("decode JSON-RPC current-time request"),
        current_time
    );
    assert_eq!(
        serde_json::to_value(current_time).expect("encode current-time request"),
        current_time_value
    );
    assert_eq!(
        serde_json::to_value(CurrentTimeReadResponse {
            current_time_at: 1_783_860_000,
        })
        .expect("encode current-time response"),
        json!({ "currentTimeAt": 1_783_860_000_i64 })
    );

    let permissions_value = json!({
        "id": "permissions-1",
        "method": "item/permissions/requestApproval",
        "params": {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "itemId": "item-1",
            "environmentId": "environment-1",
            "startedAtMs": 1_783_860_000_123_i64,
            "cwd": "/tmp/workspace",
            "reason": "Allow generated files",
            "permissions": {
                "network": { "enabled": true },
                "fileSystem": {
                    "read": null,
                    "write": null,
                    "globScanMaxDepth": 2,
                    "entries": [
                        {
                            "path": {
                                "type": "special",
                                "value": { "kind": "project_roots", "subpath": null }
                            },
                            "access": "write"
                        },
                        {
                            "path": { "type": "glob_pattern", "pattern": "**/*.env" },
                            "access": "deny"
                        }
                    ]
                }
            }
        }
    });
    let permissions: ServerRequest =
        serde_json::from_value(permissions_value.clone()).expect("decode permissions request");
    assert_eq!(
        permissions.method(),
        METHOD_ITEM_PERMISSIONS_REQUEST_APPROVAL
    );
    let permissions_jsonrpc: crate::JsonRpcRequest = permissions.clone().into();
    assert_eq!(
        ServerRequest::try_from(permissions_jsonrpc).expect("decode JSON-RPC permissions request"),
        permissions
    );
    assert_eq!(
        serde_json::to_value(permissions).expect("encode permissions request"),
        permissions_value
    );

    let default_response: PermissionsRequestApprovalResponse = serde_json::from_value(json!({
        "permissions": {}
    }))
    .expect("decode default permission response");
    assert_eq!(default_response.scope, PermissionGrantScope::Turn);
    assert_eq!(default_response.strict_auto_review, None);
    assert_eq!(
        serde_json::to_value(PermissionsRequestApprovalResponse {
            permissions: GrantedPermissionProfile::default(),
            scope: PermissionGrantScope::Session,
            strict_auto_review: Some(true),
        })
        .expect("encode permission response"),
        json!({
            "permissions": {},
            "scope": "session",
            "strictAutoReview": true
        })
    );
    assert!(serde_json::from_value::<RequestPermissionProfile>(json!({
        "network": null,
        "fileSystem": null,
        "macos": {}
    }))
    .is_err());

    let dynamic_tool_call_value = json!({
        "id": "dynamic-tool-call-1",
        "method": "item/tool/call",
        "params": {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "callId": "call-1",
            "namespace": "workspace",
            "tool": "render",
            "arguments": { "format": "png" }
        }
    });
    let dynamic_tool_call: ServerRequest = serde_json::from_value(dynamic_tool_call_value.clone())
        .expect("decode dynamic tool call request");
    assert_eq!(dynamic_tool_call.method(), METHOD_ITEM_TOOL_CALL);
    let dynamic_tool_call_jsonrpc: crate::JsonRpcRequest = dynamic_tool_call.clone().into();
    assert_eq!(
        ServerRequest::try_from(dynamic_tool_call_jsonrpc)
            .expect("decode JSON-RPC dynamic tool call request"),
        dynamic_tool_call
    );
    assert_eq!(
        serde_json::to_value(dynamic_tool_call).expect("encode dynamic tool call request"),
        dynamic_tool_call_value
    );
    assert_eq!(
        serde_json::to_value(DynamicToolCallResponse {
            content_items: vec![
                DynamicToolCallOutputContentItem::InputText {
                    text: "done".to_string(),
                },
                DynamicToolCallOutputContentItem::InputImage {
                    image_url: "data:image/png;base64,AA==".to_string(),
                },
                DynamicToolCallOutputContentItem::InputAudio {
                    audio_url: "data:audio/wav;base64,AA==".to_string(),
                },
            ],
            success: true,
        })
        .expect("encode dynamic tool call response"),
        json!({
            "contentItems": [
                { "type": "inputText", "text": "done" },
                { "type": "inputImage", "imageUrl": "data:image/png;base64,AA==" },
                { "type": "inputAudio", "audioUrl": "data:audio/wav;base64,AA==" }
            ],
            "success": true
        })
    );

    let request_value = json!({
        "id": 7,
        "method": "mcpServer/elicitation/request",
        "params": {
            "threadId": "thread_1",
            "turnId": "turn_1",
            "serverName": "form-server",
            "mode": "form",
            "_meta": null,
            "message": "Choose a value",
            "requestedSchema": {
                "type": "object",
                "properties": {"confirmed": {"type": "boolean"}},
                "required": ["confirmed"]
            }
        }
    });
    let request: ServerRequest =
        serde_json::from_value(request_value.clone()).expect("decode typed server request");
    assert_eq!(request.method(), METHOD_MCP_SERVER_ELICITATION_REQUEST);
    let jsonrpc_request: crate::JsonRpcRequest = request.clone().into();
    assert_eq!(
        ServerRequest::try_from(jsonrpc_request).expect("decode JSON-RPC server request"),
        request
    );
    assert_eq!(
        serde_json::to_value(request).expect("encode typed server request"),
        request_value
    );
    let approval_value = json!({
        "id": "approval-1",
        "method": "item/commandExecution/requestApproval",
        "params": {
            "threadId": "thread_1",
            "turnId": "turn_1",
            "itemId": "item_command",
            "startedAtMs": 1783860000000_i64,
            "approvalId": "approval-1",
            "command": "npm test",
            "availableDecisions": ["accept", "acceptForSession", "decline", "cancel"]
        }
    });
    let approval: ServerRequest =
        serde_json::from_value(approval_value.clone()).expect("decode approval request");
    assert_eq!(
        approval.method(),
        METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL
    );
    assert_eq!(
        serde_json::to_value(approval.clone()).expect("encode approval request"),
        approval_value
    );
    let approval_jsonrpc: crate::JsonRpcRequest = approval.clone().into();
    assert_eq!(
        ServerRequest::try_from(approval_jsonrpc).expect("round trip approval request"),
        approval
    );

    let file_approval_value = json!({
        "id": "file-approval-1",
        "method": "item/fileChange/requestApproval",
        "params": {
            "threadId": "thread_1",
            "turnId": "turn_1",
            "itemId": "item_file_change",
            "startedAtMs": 1783860000000_i64,
            "reason": "需要修改文件",
            "grantRoot": "/workspace"
        }
    });
    let file_approval: ServerRequest =
        serde_json::from_value(file_approval_value.clone()).expect("decode file approval request");
    assert_eq!(
        file_approval.method(),
        METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL
    );
    assert_eq!(
        serde_json::to_value(file_approval.clone()).expect("encode file approval request"),
        file_approval_value
    );
    assert_eq!(
        ServerRequest::try_from(crate::JsonRpcRequest::from(file_approval.clone()))
            .expect("round trip file approval request"),
        file_approval
    );

    let user_input_value = json!({
        "id": 8,
        "method": "item/tool/requestUserInput",
        "params": {
            "threadId": "thread_1",
            "turnId": "turn_1",
            "itemId": "item_request_user_input",
            "questions": [{
                "id": "mode",
                "header": "模式",
                "question": "请选择执行模式",
                "isOther": false,
                "isSecret": false,
                "options": [
                    {"label": "自动执行", "description": "直接继续"},
                    {"label": "确认后执行", "description": "再次确认"}
                ]
            }],
            "autoResolutionMs": null
        }
    });
    let user_input: ServerRequest =
        serde_json::from_value(user_input_value.clone()).expect("decode user input request");
    assert_eq!(user_input.method(), METHOD_ITEM_TOOL_REQUEST_USER_INPUT);
    assert_eq!(
        serde_json::to_value(user_input.clone()).expect("encode user input request"),
        user_input_value
    );
    assert_eq!(
        ServerRequest::try_from(crate::JsonRpcRequest::from(user_input.clone()))
            .expect("round trip user input request"),
        user_input
    );

    let notification = ServerNotification::TurnCompleted(TurnCompletedNotification {
        thread_id: "thread_1".to_string(),
        turn: Turn {
            id: "turn_1".to_string(),
            items: vec![],
            items_view: TurnItemsView::Full,
            status: TurnStatus::Completed,
            error: None,
            started_at: None,
            completed_at: None,
            duration_ms: None,
        },
    });
    let notification_value = serde_json::to_value(&notification).expect("encode notification");
    assert_eq!(notification_value["method"], "turn/completed");
    let jsonrpc_notification: crate::JsonRpcNotification = notification.clone().into();
    assert_eq!(jsonrpc_notification.method, "turn/completed");
    assert_eq!(
        ServerNotification::try_from(jsonrpc_notification).expect("decode JSON-RPC notification"),
        notification
    );
    let decoded_notification: ServerNotification =
        serde_json::from_value(notification_value).expect("decode notification");
    assert_eq!(decoded_notification.method(), "turn/completed");

    assert!(serde_json::from_value::<ServerNotification>(json!({
        "method": "future/notification",
        "params": {}
    }))
    .is_err());
}

#[test]
fn lifecycle_notifications_round_trip_only_the_v2_shapes() {
    let thread = json!({
        "id": "thread_1",
        "sessionId": "session_1",
        "preview": "",
        "ephemeral": false,
        "isPinned": false,
        "historyMode": "legacy",
        "modelProvider": "openai",
        "createdAt": 10,
        "updatedAt": 11,
        "cwd": "/workspace",
        "cliVersion": "1.0.0",
        "source": "appServer",
        "turns": []
    });
    let turn = json!({
        "id": "turn_1",
        "items": [],
        "itemsView": "full",
        "status": "inProgress"
    });
    let item = json!({
        "type": "agentMessage",
        "id": "item_1",
        "text": "hello"
    });
    let cases = [
        json!({
            "method": "thread/started",
            "params": {"thread": thread}
        }),
        json!({
            "method": "thread/archived",
            "params": {"threadId": "thread_1"}
        }),
        json!({
            "method": "thread/deleted",
            "params": {"threadId": "thread_1"}
        }),
        json!({
            "method": "thread/unarchived",
            "params": {"threadId": "thread_1"}
        }),
        json!({
            "method": "turn/started",
            "params": {"threadId": "thread_1", "turn": turn}
        }),
        json!({
            "method": "turn/completed",
            "params": {
                "threadId": "thread_1",
                "turn": {
                    "id": "turn_1",
                    "items": [],
                    "itemsView": "full",
                    "status": "completed"
                }
            }
        }),
        json!({
            "method": "item/started",
            "params": {
                "item": item,
                "threadId": "thread_1",
                "turnId": "turn_1",
                "startedAtMs": 12
            }
        }),
        json!({
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "agentMessage",
                    "id": "item_1",
                    "text": "hello"
                },
                "threadId": "thread_1",
                "turnId": "turn_1",
                "completedAtMs": 13
            }
        }),
        json!({
            "method": "item/agentMessage/delta",
            "params": {
                "threadId": "thread_1",
                "turnId": "turn_1",
                "itemId": "item_1",
                "delta": "hello"
            }
        }),
        json!({
            "method": "item/reasoning/summaryTextDelta",
            "params": {
                "threadId": "thread_1",
                "turnId": "turn_1",
                "itemId": "reasoning_1",
                "delta": "先分析",
                "summaryIndex": 0
            }
        }),
        json!({
            "method": "item/reasoning/summaryPartAdded",
            "params": {
                "threadId": "thread_1",
                "turnId": "turn_1",
                "itemId": "reasoning_1",
                "summaryIndex": 1
            }
        }),
        json!({
            "method": "item/reasoning/textDelta",
            "params": {
                "threadId": "thread_1",
                "turnId": "turn_1",
                "itemId": "reasoning_1",
                "delta": "raw reasoning",
                "contentIndex": 0
            }
        }),
    ];

    for expected in cases {
        let notification: ServerNotification =
            serde_json::from_value(expected.clone()).expect("decode v2 lifecycle notification");
        assert_eq!(notification.method(), expected["method"]);
        assert_eq!(
            serde_json::to_value(notification).expect("encode v2 lifecycle notification"),
            expected
        );
    }

    for retired in [
        json!({
            "method": "turn/started",
            "params": {
                "sessionId": "session_1",
                "threadId": "thread_1",
                "turnId": "turn_1",
                "status": "running"
            }
        }),
        json!({
            "method": "item/started",
            "params": {
                "sessionId": "session_1",
                "threadId": "thread_1",
                "turnId": "turn_1",
                "itemId": "item_1",
                "status": "running"
            }
        }),
    ] {
        assert!(
            serde_json::from_value::<ServerNotification>(retired).is_err(),
            "retired agentSession lifecycle payload must fail closed"
        );
    }
}

#[test]
fn typed_v2_envelope_schema_names_are_stable() {
    let schemas = [
        (
            "ClientRequest",
            serde_json::to_value(schema_for!(ClientRequest)).unwrap(),
        ),
        (
            "ClientResponse",
            serde_json::to_value(schema_for!(ClientResponse)).unwrap(),
        ),
        (
            "ServerRequest",
            serde_json::to_value(schema_for!(ServerRequest)).unwrap(),
        ),
        (
            "ServerNotification",
            serde_json::to_value(schema_for!(ServerNotification)).unwrap(),
        ),
    ];

    for (name, schema) in schemas {
        assert_eq!(schema["title"], name);
        assert!(V2_ENVELOPE_SCHEMA_TYPE_NAMES.contains(&name));
        assert!(V2_SCHEMA_TYPE_NAMES.contains(&name));
    }

    let response_schema = serde_json::to_value(schema_for!(ClientResponse)).unwrap();
    assert_eq!(
        response_schema["properties"]["id"]["$ref"],
        "#/$defs/RequestId"
    );
    assert!(response_schema["properties"].get("result").is_some());
    assert!(response_schema["properties"].get("method").is_none());

    let notification_schema = serde_json::to_value(schema_for!(ServerNotification)).unwrap();
    let methods = notification_schema["oneOf"]
        .as_array()
        .expect("server notification variants")
        .iter()
        .filter_map(|variant| variant["properties"]["method"]["const"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        methods,
        [
            "configWarning",
            "warning",
            "thread/started",
            "thread/archived",
            "thread/deleted",
            "thread/unarchived",
            "thread/closed",
            "thread/name/updated",
            "thread/status/changed",
            "turn/started",
            "turn/completed",
            "item/started",
            "item/completed",
            "item/agentMessage/delta",
            "item/commandExecution/outputDelta",
            "item/fileChange/patchUpdated",
            "item/plan/delta",
            "item/mcpToolCall/progress",
            "item/reasoning/summaryTextDelta",
            "item/reasoning/summaryPartAdded",
            "item/reasoning/textDelta",
            "model/rerouted",
            "model/list/updated",
            "model/verification",
            "model/safetyBuffering/updated",
            "thread/settings/updated",
            "thread/tokenUsage/updated",
            "thread/goal/updated",
            "thread/goal/cleared",
            "serverRequest/resolved",
        ]
    );
}
