use super::*;

#[tokio::test]
async fn read_session_projects_runtime_events_into_thread_read_tool_calls() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_tool_read".to_string()),
        thread_id: Some("thread_tool_read".to_string()),
        app_id: "desktop".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.session".to_string(),
            id: "sess_tool_read".to_string(),
            title: Some("Tool Read".to_string()),
            uri: None,
            metadata: Some(json!({
                "executionStrategy": "react"
            })),
        }),
        locale: None,
    })
    .expect("session");

    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_tool_read".to_string(),
                turn_id: Some("turn_tool_read".to_string()),
                input: AgentInput {
                    text: "整理今天的国际新闻".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("turn")
        .response
        .turn;
    core.append_external_runtime_events(
        "sess_tool_read",
        Some(&turn.turn_id),
        vec![
            RuntimeEvent::new(
                "item.started",
                canonical_tool_item_payload(
                    "sess_tool_read",
                    "thread_tool_read",
                    &turn.turn_id,
                    1,
                    "inProgress",
                    "fetch-call-1",
                    "WebFetch",
                    json!({"url": "https://example.com"}),
                    None,
                    json!({}),
                ),
            ),
            RuntimeEvent::new(
                "item.completed",
                canonical_tool_item_payload(
                    "sess_tool_read",
                    "thread_tool_read",
                    &turn.turn_id,
                    2,
                    "completed",
                    "fetch-call-1",
                    "WebFetch",
                    json!({}),
                    Some(json!({"text": "fetched https://example.com"})),
                    json!({}),
                ),
            ),
            RuntimeEvent::new(
                "item.started",
                canonical_tool_item_payload(
                    "sess_tool_read",
                    "thread_tool_read",
                    &turn.turn_id,
                    3,
                    "inProgress",
                    "search-call-1",
                    "WebSearch",
                    json!({"query": "international news"}),
                    None,
                    json!({}),
                ),
            ),
            RuntimeEvent::new(
                "item.completed",
                canonical_tool_item_payload(
                    "sess_tool_read",
                    "thread_tool_read",
                    &turn.turn_id,
                    4,
                    "completed",
                    "search-call-1",
                    "WebSearch",
                    json!({}),
                    Some(json!({"text": "search results"})),
                    json!({}),
                ),
            ),
            RuntimeEvent::new("turn.completed", json!({})),
        ],
    )
    .expect("append canonical tool events");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_tool_read".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");
    assert_eq!(detail["execution_strategy"], "react");
    assert_eq!(detail["thread_read"]["status"], "completed");
    assert_eq!(detail["thread_read"]["execution_strategy"], "react");
    let tool_calls = detail["thread_read"]["tool_calls"]
        .as_array()
        .expect("tool calls");
    assert_eq!(tool_calls.len(), 2);
    let web_fetch = tool_calls
        .iter()
        .find(|call| call["tool_name"] == "WebFetch")
        .expect("WebFetch call");
    assert_eq!(web_fetch["status"], "completed");
    assert_eq!(web_fetch["success"], true);
    assert_eq!(web_fetch["output_preview"], "fetched https://example.com");

    let web_search = tool_calls
        .iter()
        .find(|call| call["tool_name"] == "WebSearch")
        .expect("WebSearch call");
    assert_eq!(web_search["id"], "search-call-1");
    assert_eq!(web_search["status"], "completed");
    assert_eq!(web_search["success"], true);
    assert_eq!(web_search["output_preview"], "search results");
}

#[tokio::test]
async fn read_session_merges_tool_started_arguments_into_completed_tool_calls() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_tool_arguments".to_string()),
        thread_id: Some("thread_tool_arguments".to_string()),
        app_id: "desktop".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_tool_arguments".to_string(),
                turn_id: Some("turn_tool_arguments".to_string()),
                input: AgentInput {
                    text: "打开导入文件".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("turn")
        .response
        .turn;

    core.append_external_runtime_events(
        "sess_tool_arguments",
        Some(&turn.turn_id),
        vec![
            RuntimeEvent::new(
                "item.started",
                canonical_tool_item_payload(
                    "sess_tool_arguments",
                    "thread_tool_arguments",
                    &turn.turn_id,
                    1,
                    "inProgress",
                    "call_read_md",
                    "read_file",
                    json!({"path": "/workspace/docs/imported-preview.md"}),
                    None,
                    json!({}),
                ),
            ),
            RuntimeEvent::new(
                "item.completed",
                canonical_tool_item_payload(
                    "sess_tool_arguments",
                    "thread_tool_arguments",
                    &turn.turn_id,
                    2,
                    "completed",
                    "call_read_md",
                    "read_file",
                    json!({}),
                    Some(json!({"text": "导入会话 Markdown 预览内容"})),
                    json!({}),
                ),
            ),
            RuntimeEvent::new(
                "item.started",
                canonical_tool_item_payload(
                    "sess_tool_arguments",
                    "thread_tool_arguments",
                    &turn.turn_id,
                    3,
                    "inProgress",
                    "call_mcp_docs",
                    "mcp__docs__search_docs",
                    json!({"query": "mcp structured content"}),
                    None,
                    json!({}),
                ),
            ),
            RuntimeEvent::new(
                "item.completed",
                canonical_tool_item_payload(
                    "sess_tool_arguments",
                    "thread_tool_arguments",
                    &turn.turn_id,
                    4,
                    "completed",
                    "call_mcp_docs",
                    "mcp__docs__search_docs",
                    json!({}),
                    Some(json!({
                        "text": "docs found",
                        "structuredContent": {"answer": "ok", "ids": ["doc-1"]}
                    })),
                    json!({}),
                ),
            ),
            RuntimeEvent::new("turn.completed", json!({})),
        ],
    )
    .expect("append tool events");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_tool_arguments".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");
    let tool_calls = detail["thread_read"]["tool_calls"]
        .as_array()
        .expect("tool calls");
    let read_file = tool_calls
        .iter()
        .find(|call| call["id"] == "call_read_md")
        .expect("read_file call");

    assert_eq!(read_file["tool_name"], "read_file");
    assert_eq!(read_file["status"], "completed");
    assert_eq!(read_file["success"], true);
    assert_eq!(
        tool_argument_value(&read_file["arguments"], "path"),
        Some("/workspace/docs/imported-preview.md")
    );
    assert_eq!(read_file["output_preview"], "导入会话 Markdown 预览内容");
    let mcp_docs = tool_calls
        .iter()
        .find(|call| call["id"] == "call_mcp_docs")
        .expect("mcp docs call");
    assert_eq!(mcp_docs["tool_name"], "mcp__docs__search_docs");
    assert_eq!(mcp_docs["status"], "completed");
    assert_eq!(mcp_docs["success"], true);
    assert_eq!(mcp_docs["structured_content"]["answer"], "ok");
    assert_eq!(mcp_docs["structured_content"]["ids"][0], "doc-1");
}

#[tokio::test]
async fn read_session_preserves_image_tool_result_metadata_for_history_restore() {
    let (core, turn) = start_read_model_test_turn(
        "sess_image_tool_read",
        "thread_image_tool_read",
        "turn_image_tool_read",
    )
    .await;

    let response_output = json!({
        "task_id": "task-image-1",
        "task_type": "image_generate",
        "task_family": "image",
        "status": "pending",
        "normalized_status": "pending",
        "artifact_path": ".lime/tasks/image_generate/task-image-1.json",
        "record": {
            "task_id": "task-image-1",
            "task_type": "image_generate",
            "task_family": "image",
            "status": "pending",
            "normalized_status": "pending",
            "payload": {
                "prompt": "画一张深圳夏天的图",
                "provider_id": "agnes",
                "model": "agnes-image-21-flash"
            }
        }
    })
    .to_string();

    core.append_external_runtime_events(
        "sess_image_tool_read",
        Some(&turn.turn_id),
        vec![
            RuntimeEvent::new(
                "item.started",
                canonical_tool_item_payload(
                    "sess_image_tool_read",
                    "thread_image_tool_read",
                    &turn.turn_id,
                    1,
                    "inProgress",
                    "image-tool-1",
                    "lime_create_image_generation_task",
                    json!({
                        "prompt": "画一张深圳夏天的图",
                        "provider_id": "agnes",
                        "model": "agnes-image-21-flash"
                    }),
                    None,
                    json!({}),
                ),
            ),
            RuntimeEvent::new(
                "image_task.created",
                json!({
                    "taskId": "task-image-1",
                    "task_id": "task-image-1",
                    "artifactPath": ".lime/tasks/image_generate/task-image-1.json",
                    "artifact_path": ".lime/tasks/image_generate/task-image-1.json"
                }),
            ),
            RuntimeEvent::new(
                "item.completed",
                canonical_tool_item_payload(
                    "sess_image_tool_read",
                    "thread_image_tool_read",
                    &turn.turn_id,
                    3,
                    "completed",
                    "image-tool-1",
                    "lime_create_image_generation_task",
                    json!({}),
                    Some(json!({"text": response_output})),
                    json!({
                        "task_id": "task-image-1",
                        "task_type": "image_generate",
                        "task_family": "image",
                        "status": "pending",
                        "normalized_status": "pending",
                        "artifact_path": ".lime/tasks/image_generate/task-image-1.json",
                        "record": {
                            "task_id": "task-image-1",
                            "task_type": "image_generate",
                            "task_family": "image",
                            "status": "pending",
                            "normalized_status": "pending",
                            "payload": {
                                "prompt": "画一张深圳夏天的图",
                                "provider_id": "agnes",
                                "model": "agnes-image-21-flash"
                            }
                        }
                    }),
                ),
            ),
            RuntimeEvent::new("turn.completed", json!({})),
        ],
    )
    .expect("append image tool events");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_image_tool_read".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");
    let tool_call = detail["thread_read"]["tool_calls"]
        .as_array()
        .expect("tool calls")
        .iter()
        .find(|call| call["id"] == "image-tool-1")
        .expect("image tool call");
    assert_eq!(tool_call["status"], "completed");
    assert_eq!(tool_call["success"], true);
    assert_eq!(tool_call["output"], response_output);
    assert_eq!(tool_call["metadata"]["task_id"], "task-image-1");
    assert_eq!(tool_call["metadata"]["task_type"], "image_generate");
    assert_eq!(tool_call["metadata"]["task_family"], "image");
    assert_eq!(
        tool_call["metadata"]["record"]["payload"]["model"],
        "agnes-image-21-flash"
    );

    let item = detail["items"]
        .as_array()
        .expect("items")
        .iter()
        .find(|item| item["id"] == "image-tool-1")
        .expect("image synthetic item");
    assert_eq!(item["type"], "tool_call");
    assert_eq!(item["metadata"]["task_id"], "task-image-1");
    assert_eq!(item["output"], response_output);
}

#[tokio::test]
async fn read_session_projects_turn_completed_usage_into_read_model_turns() {
    let (core, turn) = start_read_model_test_turn(
        "sess_turn_usage_read",
        "thread_turn_usage_read",
        "turn_usage_read",
    )
    .await;

    core.append_external_runtime_events(
        "sess_turn_usage_read",
        Some(&turn.turn_id),
        vec![RuntimeEvent::new(
            "turn.completed",
            json!({
                "usage": {
                    "input_tokens": 31_000,
                    "output_tokens": 119,
                    "cached_input_tokens": 0
                }
            }),
        )],
    )
    .expect("append usage terminal event");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_turn_usage_read".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");

    assert_eq!(
        detail["turns"][0]["usage"]["input_tokens"].as_u64(),
        Some(31_000)
    );
    assert_eq!(
        detail["thread_read"]["turns"][0]["usage"]["output_tokens"].as_u64(),
        Some(119)
    );
    assert_eq!(
        detail["thread_read"]["diagnostics"]["latest_turn_usage"]["cached_input_tokens"].as_u64(),
        Some(0)
    );
    assert_eq!(
        detail["thread_read"]["runtime_summary"]["latestTurnUsage"]["input_tokens"].as_u64(),
        Some(31_000)
    );
}

#[tokio::test]
async fn read_session_projects_workflow_audit_turn_completed_usage_into_read_model_turns() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let core = RuntimeCore::default().with_event_log_writer(event_log_writer.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_audit_turn_usage_read".to_string()),
        thread_id: Some("thread_audit_turn_usage_read".to_string()),
        app_id: "desktop".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_audit_turn_usage_read".to_string(),
                turn_id: Some("turn_audit_usage_read".to_string()),
                input: AgentInput {
                    text: "@配图 画一张深圳夏天的图".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("turn")
        .response
        .turn;

    crate::runtime::event_store::append_workflow_audit_runtime_events(
        Some(event_log_writer.as_ref()),
        "sess_audit_turn_usage_read",
        "thread_audit_turn_usage_read",
        Some(&turn.turn_id),
        vec![RuntimeEvent::new(
            "turn.completed",
            json!({
                "usage": {
                    "input_tokens": 1175,
                    "output_tokens": 112
                }
            }),
        )],
    )
    .expect("append workflow audit usage");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_audit_turn_usage_read".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");

    assert_eq!(
        detail["turns"][0]["usage"]["input_tokens"].as_u64(),
        Some(1175)
    );
    assert_eq!(
        detail["thread_read"]["turns"][0]["usage"]["output_tokens"].as_u64(),
        Some(112)
    );
    assert_eq!(
        detail["thread_read"]["diagnostics"]["latest_turn_usage"]["input_tokens"].as_u64(),
        Some(1175)
    );
    assert_eq!(
        detail["thread_read"]["runtime_summary"]["latestTurnUsage"]["output_tokens"].as_u64(),
        Some(112)
    );
}

#[tokio::test]
async fn read_session_preserves_workspace_patch_host_tool_metadata_on_thread_items() {
    let (core, turn) = start_read_model_test_turn(
        "sess_legacy_tool_metadata",
        "thread_legacy_tool_metadata",
        "turn_legacy_tool_metadata",
    )
    .await;

    core.append_external_runtime_events(
        "sess_legacy_tool_metadata",
        Some(&turn.turn_id),
        vec![
            RuntimeEvent::new(
                "item.started",
                canonical_tool_item_payload(
                    "sess_legacy_tool_metadata",
                    "thread_legacy_tool_metadata",
                    &turn.turn_id,
                    1,
                    "inProgress",
                    "workspace-patch-host-tool-websearch-1",
                    "WebSearch",
                    json!({"query": "golang 学习路径"}),
                    None,
                    json!({
                        "source": "workspace_patch_host_tool_requests",
                        "workflowKey": "content_article_workflow",
                        "workflow_key": "content_article_workflow"
                    }),
                ),
            ),
            RuntimeEvent::new(
                "item.completed",
                canonical_tool_item_payload(
                    "sess_legacy_tool_metadata",
                    "thread_legacy_tool_metadata",
                    &turn.turn_id,
                    2,
                    "completed",
                    "workspace-patch-host-tool-websearch-1",
                    "WebSearch",
                    json!({}),
                    Some(json!({"text": "找到 3 条资料"})),
                    json!({
                        "source": "workspace_patch_host_tool_requests",
                        "workflowKey": "content_article_workflow",
                        "workflow_key": "content_article_workflow"
                    }),
                ),
            ),
            RuntimeEvent::new("turn.completed", json!({})),
        ],
    )
    .expect("append legacy tool events");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_legacy_tool_metadata".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");
    let item = detail["items"]
        .as_array()
        .expect("items")
        .iter()
        .find(|item| item["id"] == "workspace-patch-host-tool-websearch-1")
        .expect("workspace patch host tool item");

    assert_eq!(item["type"], "web_search");
    assert_eq!(
        item["metadata"]["source"],
        "workspace_patch_host_tool_requests"
    );
    assert_eq!(item["metadata"]["workflowKey"], "content_article_workflow");
    assert_eq!(item["metadata"]["workflow_key"], "content_article_workflow");
}

#[allow(clippy::too_many_arguments)]
fn canonical_tool_item_payload(
    session_id: &str,
    thread_id: &str,
    turn_id: &str,
    sequence: u64,
    status: &str,
    call_id: &str,
    tool_name: &str,
    arguments: serde_json::Value,
    output: Option<serde_json::Value>,
    metadata: serde_json::Value,
) -> serde_json::Value {
    let arguments = match arguments {
        serde_json::Value::Object(arguments) => arguments
            .into_iter()
            .map(|(name, value)| {
                json!({
                    "name": name,
                    "value": value.as_str().map(str::to_string).unwrap_or_else(|| value.to_string())
                })
            })
            .collect::<Vec<_>>(),
        serde_json::Value::Array(arguments) => arguments,
        serde_json::Value::Null => Vec::new(),
        value => vec![json!({"name": "value", "value": value.to_string()})],
    };
    let terminal = matches!(status, "completed" | "failed" | "interrupted" | "cancelled");
    json!({
        "item": {
            "sessionId": session_id,
            "threadId": thread_id,
            "turnId": turn_id,
            "itemId": format!("item_{call_id}"),
            "sequence": sequence,
            "ordinal": sequence,
            "createdAtMs": 1_784_000_000_000_i64,
            "updatedAtMs": 1_784_000_000_000_i64 + sequence as i64,
            "completedAtMs": terminal.then_some(1_784_000_000_000_i64 + sequence as i64),
            "kind": "tool",
            "status": status,
            "payload": {
                "type": "tool",
                "call_id": call_id,
                "name": tool_name,
                "arguments": arguments,
                "output": output
            },
            "metadata": metadata
        }
    })
}

fn tool_argument_value<'a>(arguments: &'a serde_json::Value, name: &str) -> Option<&'a str> {
    arguments.as_array()?.iter().find_map(|argument| {
        (argument.get("name").and_then(serde_json::Value::as_str) == Some(name))
            .then(|| argument.get("value").and_then(serde_json::Value::as_str))
            .flatten()
    })
}
