use super::super::bridge_tool_timeout;
use super::super::tools::{
    app_tool_authority_from_meta, normalize_tool_input_schema, tool_result_output_schema,
};
use super::common::*;
use crate::manager::McpClientManager;
use crate::types::{McpContent, McpError, McpToolDefinition};
use std::time::Duration;

#[test]
fn app_tool_authority_uses_exact_connector_link_and_account_requirement() {
    let meta = rmcp::model::Meta(
        serde_json::json!({
            "connector_id": " calendar ",
            "link_id": "link-calendar",
            "_codex_apps": { "requires_explicit_link_id": true }
        })
        .as_object()
        .expect("tool metadata")
        .clone(),
    );

    let authority = app_tool_authority_from_meta(Some(&meta));
    assert_eq!(authority.connector_id.as_deref(), Some("calendar"));
    assert_eq!(authority.link_id.as_deref(), Some("link-calendar"));
    assert!(authority.requires_explicit_link_id);
}

#[test]
fn bridge_tool_timeout_preserves_explicit_and_startup_fallback_values() {
    let mut config = create_test_config();
    config.startup_timeout = 120;
    config.tool_timeout = Some(15);
    assert_eq!(bridge_tool_timeout(&config), Duration::from_secs(15));

    config.tool_timeout = None;
    assert_eq!(bridge_tool_timeout(&config), Duration::from_secs(120));
}

#[test]
fn test_extract_tool_metadata_from_schema_extension() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {},
        "x-lime": {
            "deferred_loading": true,
            "always_visible": false,
            "allowed_callers": ["assistant", "code_execution"],
            "input_examples": [{"q": "rust"}],
            "tags": ["search", "docs"]
        }
    });
    let meta = McpClientManager::extract_tool_metadata("docs_search", &schema);
    assert_eq!(meta.deferred_loading, Some(true));
    assert_eq!(meta.always_visible, Some(false));
    assert_eq!(
        meta.allowed_callers.unwrap_or_default(),
        vec!["assistant".to_string(), "code_execution".to_string()]
    );
    assert_eq!(meta.input_examples.len(), 1);
    assert_eq!(
        meta.tags.unwrap_or_default(),
        vec!["search".to_string(), "docs".to_string()]
    );
}

#[test]
fn test_normalize_tool_input_schema_adds_missing_properties() {
    let schema = normalize_tool_input_schema(serde_json::json!({
        "type": "object"
    }));

    assert_eq!(schema["properties"], serde_json::json!({}));
}

#[test]
fn test_normalize_tool_input_schema_replaces_null_properties() {
    let schema = normalize_tool_input_schema(serde_json::json!({
        "type": "object",
        "properties": null
    }));

    assert_eq!(schema["properties"], serde_json::json!({}));
}

#[test]
fn test_tool_result_output_schema_wraps_structured_content_schema() {
    let structured_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "answer": { "type": "string" }
        },
        "required": ["answer"]
    });
    let schema = tool_result_output_schema(structured_schema.clone());

    assert_eq!(schema["type"], "object");
    assert_eq!(schema["required"], serde_json::json!(["content"]));
    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(schema["properties"]["structuredContent"], structured_schema);
    assert_eq!(schema["properties"]["content"]["type"], "array");
    assert_eq!(schema["properties"]["isError"]["type"], "boolean");
    assert_eq!(schema["properties"]["_meta"]["type"], "object");
}

// ========================================================================
// 服务器生命周期测试
// ========================================================================

#[test]
fn test_apply_runtime_tool_names_prefixes_all_tools() {
    let tools = vec![
        create_test_tool("tool1", "Tool 1", "server1"),
        create_test_tool("tool2", "Tool 2", "server2"),
    ];

    let resolved = McpClientManager::apply_runtime_tool_names(tools);

    assert_eq!(resolved.len(), 2);
    assert!(resolved.iter().any(|t| t.name == "mcp__server1__tool1"));
    assert!(resolved.iter().any(|t| t.name == "mcp__server2__tool2"));
}

#[test]
fn test_apply_runtime_tool_names_keeps_distinct_servers_for_same_tool() {
    let tools = vec![
        create_test_tool("read_file", "Read file from server1", "server1"),
        create_test_tool("read_file", "Read file from server2", "server2"),
        create_test_tool("unique_tool", "Unique tool", "server1"),
    ];

    let resolved = McpClientManager::apply_runtime_tool_names(tools);

    assert_eq!(resolved.len(), 3);
    assert!(resolved.iter().any(|t| t.name == "mcp__server1__read_file"));
    assert!(resolved.iter().any(|t| t.name == "mcp__server2__read_file"));
    assert!(resolved
        .iter()
        .any(|t| t.name == "mcp__server1__unique_tool"));
}

#[test]
fn test_apply_runtime_tool_names_supports_multiple_same_name_tools() {
    let tools = vec![
        create_test_tool("tool_a", "Tool A from server1", "server1"),
        create_test_tool("tool_a", "Tool A from server2", "server2"),
        create_test_tool("tool_a", "Tool A from server3", "server3"),
    ];

    let resolved = McpClientManager::apply_runtime_tool_names(tools);

    assert_eq!(resolved.len(), 3);
    assert!(resolved.iter().any(|t| t.name == "mcp__server1__tool_a"));
    assert!(resolved.iter().any(|t| t.name == "mcp__server2__tool_a"));
    assert!(resolved.iter().any(|t| t.name == "mcp__server3__tool_a"));
}

#[test]
fn test_apply_runtime_tool_names_empty_list() {
    let tools: Vec<McpToolDefinition> = vec![];
    let resolved = McpClientManager::apply_runtime_tool_names(tools);
    assert!(resolved.is_empty());
}

#[test]
fn test_apply_default_loading_policy_auto_defers_large_server_tools() {
    let tools = (0..7)
        .map(|index| McpToolDefinition {
            name: format!("tool_{index}"),
            description: format!("tool {index}"),
            input_schema: serde_json::json!({}),
            output_schema: None,
            server_name: "large-server".to_string(),
            deferred_loading: None,
            always_visible: if index == 0 { Some(true) } else { None },
            allowed_callers: None,
            input_examples: None,
            tags: None,
        })
        .collect::<Vec<_>>();

    let resolved = McpClientManager::apply_default_loading_policy(tools);
    assert_eq!(resolved.len(), 7);
    assert_eq!(resolved[0].deferred_loading, Some(false));
    assert!(resolved
        .iter()
        .skip(1)
        .all(|tool| tool.deferred_loading == Some(true)));
}

#[test]
fn test_apply_runtime_tool_names_preserves_metadata_fields() {
    let mut tool_a = create_test_tool("search", "Search docs", "server1");
    tool_a.deferred_loading = Some(true);
    tool_a.always_visible = Some(true);
    tool_a.allowed_callers = Some(vec!["assistant".to_string()]);
    tool_a.tags = Some(vec!["docs".to_string()]);
    tool_a.input_examples = Some(vec![serde_json::json!({ "query": "rust" })]);

    let mut tool_b = create_test_tool("search", "Search issues", "server2");
    tool_b.deferred_loading = Some(false);
    tool_b.always_visible = Some(false);
    tool_b.allowed_callers = Some(vec!["code_execution".to_string()]);
    tool_b.tags = Some(vec!["issues".to_string()]);
    tool_b.input_examples = Some(vec![serde_json::json!({ "query": "bug" })]);

    let resolved = McpClientManager::apply_runtime_tool_names(vec![tool_a, tool_b]);
    let server1 = resolved
        .iter()
        .find(|tool| tool.name == "mcp__server1__search")
        .expect("server1 tool should be renamed");
    let server2 = resolved
        .iter()
        .find(|tool| tool.name == "mcp__server2__search")
        .expect("server2 tool should be renamed");

    assert_eq!(server1.deferred_loading, Some(true));
    assert_eq!(server1.always_visible, Some(true));
    assert_eq!(server1.allowed_callers, Some(vec!["assistant".to_string()]));
    assert_eq!(server1.tags, Some(vec!["docs".to_string()]));
    assert_eq!(
        server1.input_examples,
        Some(vec![serde_json::json!({ "query": "rust" })])
    );

    assert_eq!(server2.deferred_loading, Some(false));
    assert_eq!(server2.always_visible, Some(false));
    assert_eq!(
        server2.allowed_callers,
        Some(vec!["code_execution".to_string()])
    );
    assert_eq!(server2.tags, Some(vec!["issues".to_string()]));
    assert_eq!(
        server2.input_examples,
        Some(vec![serde_json::json!({ "query": "bug" })])
    );
}

#[test]
fn test_apply_default_loading_policy_respects_threshold_boundary_and_explicit_values() {
    let threshold_tools = (0..crate::tool_policy::TEST_AUTO_DEFER_TOOL_COUNT_THRESHOLD)
        .map(|index| McpToolDefinition {
            name: format!("threshold_{index}"),
            description: format!("threshold {index}"),
            input_schema: serde_json::json!({}),
            output_schema: None,
            server_name: "threshold-server".to_string(),
            deferred_loading: None,
            always_visible: None,
            allowed_callers: None,
            input_examples: None,
            tags: None,
        })
        .collect::<Vec<_>>();

    let mut large_tools = (0..5)
        .map(|index| McpToolDefinition {
            name: format!("auto_{index}"),
            description: format!("auto {index}"),
            input_schema: serde_json::json!({}),
            output_schema: None,
            server_name: "large-server".to_string(),
            deferred_loading: None,
            always_visible: None,
            allowed_callers: None,
            input_examples: None,
            tags: None,
        })
        .collect::<Vec<_>>();

    large_tools.push(McpToolDefinition {
        name: "explicit_false".to_string(),
        description: "explicit false".to_string(),
        input_schema: serde_json::json!({}),
        output_schema: None,
        server_name: "large-server".to_string(),
        deferred_loading: Some(false),
        always_visible: None,
        allowed_callers: None,
        input_examples: None,
        tags: None,
    });
    large_tools.push(McpToolDefinition {
        name: "explicit_true".to_string(),
        description: "explicit true".to_string(),
        input_schema: serde_json::json!({}),
        output_schema: None,
        server_name: "large-server".to_string(),
        deferred_loading: Some(true),
        always_visible: None,
        allowed_callers: None,
        input_examples: None,
        tags: None,
    });

    let mut all_tools = threshold_tools;
    all_tools.extend(large_tools);

    let resolved = McpClientManager::apply_default_loading_policy(all_tools);

    assert!(resolved
        .iter()
        .filter(|tool| tool.server_name == "threshold-server")
        .all(|tool| tool.deferred_loading == Some(false)));

    let explicit_false = resolved
        .iter()
        .find(|tool| tool.name == "explicit_false")
        .expect("explicit false tool should exist");
    assert_eq!(explicit_false.deferred_loading, Some(false));

    let explicit_true = resolved
        .iter()
        .find(|tool| tool.name == "explicit_true")
        .expect("explicit true tool should exist");
    assert_eq!(explicit_true.deferred_loading, Some(true));

    assert!(resolved
        .iter()
        .filter(|tool| tool.server_name == "large-server" && tool.name.starts_with("auto_"))
        .all(|tool| tool.deferred_loading == Some(true)));
}

// ========================================================================
// 工具列表缓存测试（Task 4.3）
// ========================================================================

#[tokio::test]
async fn test_list_tools_returns_cached_when_valid() {
    let manager = McpClientManager::new(None);

    // 预先设置缓存
    let cached_tools = vec![create_test_tool(
        "mcp__cached_server__cached_tool",
        "Cached tool",
        "cached_server",
    )];
    manager.update_tool_cache(cached_tools.clone()).await;

    // 调用 list_tools 应该返回缓存的工具
    let result = manager.list_tools().await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "mcp__cached_server__cached_tool");
}

#[tokio::test]
async fn test_list_tools_for_context_filters_deferred_and_caller() {
    let manager = McpClientManager::new(None);
    manager
        .update_tool_cache(vec![
            create_test_tool("mcp__s1__always_tool", "always", "s1"),
            McpToolDefinition {
                name: "mcp__s1__hidden_tool".to_string(),
                description: "hidden".to_string(),
                input_schema: serde_json::json!({}),
                output_schema: None,
                server_name: "s1".to_string(),
                deferred_loading: Some(true),
                always_visible: Some(false),
                allowed_callers: Some(vec!["code_execution".to_string()]),
                input_examples: None,
                tags: None,
            },
            McpToolDefinition {
                name: "mcp__s1__visible_deferred".to_string(),
                description: "visible deferred".to_string(),
                input_schema: serde_json::json!({}),
                output_schema: None,
                server_name: "s1".to_string(),
                deferred_loading: Some(true),
                always_visible: Some(true),
                allowed_callers: Some(vec!["assistant".to_string()]),
                input_examples: None,
                tags: None,
            },
        ])
        .await;

    let assistant_tools = manager
        .list_tools_for_context(Some("assistant"), false)
        .await
        .unwrap();
    assert!(assistant_tools
        .iter()
        .any(|t| t.name == "mcp__s1__always_tool"));
    assert!(assistant_tools
        .iter()
        .any(|t| t.name == "mcp__s1__visible_deferred"));
    assert!(!assistant_tools
        .iter()
        .any(|t| t.name == "mcp__s1__hidden_tool"));

    let code_exec_tools = manager
        .list_tools_for_context(Some("code_execution"), true)
        .await
        .unwrap();
    assert!(code_exec_tools
        .iter()
        .any(|t| t.name == "mcp__s1__hidden_tool"));
}

#[tokio::test]
async fn test_search_tools_prioritizes_exact_match() {
    let manager = McpClientManager::new(None);
    manager
        .update_tool_cache(vec![
            McpToolDefinition {
                name: "mcp__s1__weather".to_string(),
                description: "Get weather".to_string(),
                input_schema: serde_json::json!({}),
                output_schema: None,
                server_name: "s1".to_string(),
                deferred_loading: Some(true),
                always_visible: Some(false),
                allowed_callers: None,
                input_examples: None,
                tags: Some(vec!["forecast".to_string()]),
            },
            McpToolDefinition {
                name: "mcp__s1__get_weather".to_string(),
                description: "weather by city".to_string(),
                input_schema: serde_json::json!({}),
                output_schema: None,
                server_name: "s1".to_string(),
                deferred_loading: Some(true),
                always_visible: Some(false),
                allowed_callers: None,
                input_examples: None,
                tags: Some(vec!["weather".to_string()]),
            },
        ])
        .await;

    let tools = manager
        .search_tools("weather", 5, Some("assistant"))
        .await
        .unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].name, "mcp__s1__weather");
}

#[tokio::test]
async fn test_search_tools_empty_query_prioritizes_always_visible_then_name() {
    let manager = McpClientManager::new(None);
    manager
        .update_tool_cache(vec![
            McpToolDefinition {
                name: "mcp__s1__alpha".to_string(),
                description: "alpha".to_string(),
                input_schema: serde_json::json!({}),
                output_schema: None,
                server_name: "s1".to_string(),
                deferred_loading: Some(false),
                always_visible: Some(false),
                allowed_callers: None,
                input_examples: None,
                tags: None,
            },
            McpToolDefinition {
                name: "mcp__s1__zeta".to_string(),
                description: "zeta".to_string(),
                input_schema: serde_json::json!({}),
                output_schema: None,
                server_name: "s1".to_string(),
                deferred_loading: Some(true),
                always_visible: Some(true),
                allowed_callers: None,
                input_examples: None,
                tags: None,
            },
            McpToolDefinition {
                name: "mcp__s1__beta".to_string(),
                description: "beta".to_string(),
                input_schema: serde_json::json!({}),
                output_schema: None,
                server_name: "s1".to_string(),
                deferred_loading: Some(false),
                always_visible: Some(true),
                allowed_callers: None,
                input_examples: None,
                tags: None,
            },
        ])
        .await;

    let tools = manager
        .search_tools("", 2, Some("assistant"))
        .await
        .expect("empty query search should succeed");

    assert_eq!(tools.len(), 2);
    assert_eq!(
        tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        vec!["mcp__s1__beta", "mcp__s1__zeta"]
    );
}

#[tokio::test]
async fn test_call_tool_with_caller_rejects_unauthorized_caller() {
    let manager = McpClientManager::new(None);
    manager
        .update_tool_cache(vec![McpToolDefinition {
            name: "mcp__s1__restricted".to_string(),
            description: "Restricted tool".to_string(),
            input_schema: serde_json::json!({}),
            output_schema: None,
            server_name: "s1".to_string(),
            deferred_loading: None,
            always_visible: None,
            allowed_callers: Some(vec!["code_execution".to_string()]),
            input_examples: None,
            tags: None,
        }])
        .await;

    let result = manager
        .call_tool_with_caller(
            "mcp__s1__restricted",
            serde_json::json!({}),
            Some("assistant"),
        )
        .await;
    assert!(result.is_err());
    match result {
        Err(McpError::ToolCallFailed(message)) => {
            assert!(message.contains("无权调用"));
        }
        _ => panic!("Expected ToolCallFailed"),
    }
}

#[tokio::test]
async fn test_list_tools_returns_empty_when_no_servers() {
    let manager = McpClientManager::new(None);

    // 没有运行的服务器时，应该返回空列表
    let result = manager.list_tools().await.unwrap();
    assert!(result.is_empty());
}

// ========================================================================
// 工具调用测试（Task 4.3）
// ========================================================================

#[tokio::test]
async fn test_call_tool_not_found() {
    let manager = McpClientManager::new(None);

    // 调用不存在的工具
    let result = manager
        .call_tool("mcp__missing__nonexistent_tool", serde_json::json!({}))
        .await;

    // 应该返回 ToolNotFound 错误
    assert!(result.is_err());
    match result {
        Err(McpError::ToolNotFound(name)) => {
            assert_eq!(name, "mcp__missing__nonexistent_tool");
        }
        _ => panic!("Expected ToolNotFound error"),
    }
}

#[tokio::test]
async fn test_call_tool_invalid_arguments() {
    let manager = McpClientManager::new(None);

    // 添加一个客户端
    let client = create_test_client("test-server");
    manager
        .add_client("test-server".to_string(), client)
        .await
        .unwrap();

    // 使用非对象参数调用工具
    let result = manager
        .call_tool("mcp__test-server__some_tool", serde_json::json!("invalid"))
        .await;

    // 应该返回错误（参数必须是对象或 null）
    assert!(result.is_err());
}

// ========================================================================
// 内容转换测试（Task 4.3）
// ========================================================================

#[test]
fn test_convert_content_text() {
    let content = rmcp::model::Content::text("Hello, World!");
    let mcp_content = McpClientManager::convert_content(content);

    match mcp_content {
        McpContent::Text { text } => {
            assert_eq!(text, "Hello, World!");
        }
        _ => panic!("Expected Text content"),
    }
}

#[test]
fn test_convert_call_tool_result_preserves_structured_content() {
    let result = rmcp::model::CallToolResult {
        content: vec![rmcp::model::Content::text("structured result")],
        structured_content: Some(serde_json::json!({
            "results": [
                { "title": "MCP current" }
            ]
        })),
        is_error: Some(false),
        meta: None,
    };

    let mcp_result = McpClientManager::convert_call_tool_result(result);

    assert_eq!(
        mcp_result.structured_content,
        Some(serde_json::json!({
            "results": [
                { "title": "MCP current" }
            ]
        }))
    );
    assert!(!mcp_result.is_error);
    assert!(matches!(
        mcp_result.content.as_slice(),
        [McpContent::Text { text }] if text == "structured result"
    ));
}

#[test]
fn test_convert_content_image() {
    let content = rmcp::model::Content::image("base64data", "image/png");
    let mcp_content = McpClientManager::convert_content(content);

    match mcp_content {
        McpContent::Image { data, mime_type } => {
            assert_eq!(data, "base64data");
            assert_eq!(mime_type, "image/png");
        }
        _ => panic!("Expected Image content"),
    }
}

// ========================================================================
// 提示词管理测试（Task 4.4）
// ========================================================================
