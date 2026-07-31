//! mcp request processor tests.

use super::super::*;
use super::tests_support::initialize_processor;
use app_server_protocol::{
    ClientCapabilities, JsonRpcMessage, RequestId, METHOD_INITIALIZE, METHOD_INITIALIZED,
    METHOD_MCP_PROMPT_GET, METHOD_MCP_PROMPT_LIST, METHOD_MCP_RESOURCE_LIST,
    METHOD_MCP_RESOURCE_READ, METHOD_MCP_RESOURCE_SUBSCRIBE, METHOD_MCP_RESOURCE_UNSUBSCRIBE,
    METHOD_MCP_SERVER_CREATE, METHOD_MCP_SERVER_DELETE, METHOD_MCP_SERVER_ENABLED_SET,
    METHOD_MCP_SERVER_IMPORT_FROM_APP, METHOD_MCP_SERVER_LIST, METHOD_MCP_SERVER_OAUTH_LOGIN,
    METHOD_MCP_SERVER_START, METHOD_MCP_SERVER_STATUS_LIST, METHOD_MCP_SERVER_STOP,
    METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE, METHOD_MCP_SERVER_UPDATE, METHOD_MCP_TOOL_CALL,
    METHOD_MCP_TOOL_CALL_WITH_CALLER, METHOD_MCP_TOOL_LIST,
};
use async_trait::async_trait;
use serde_json::json;

#[derive(Default)]
struct OAuthTestDataSource {
    login: std::sync::Mutex<Option<lime_mcp::McpOAuthLoginHandle>>,
}

impl crate::SessionAppDataSource for OAuthTestDataSource {}
impl crate::WorkspaceAppDataSource for OAuthTestDataSource {}
impl crate::SkillAppDataSource for OAuthTestDataSource {}
impl crate::WorkspaceSkillBindingAppDataSource for OAuthTestDataSource {}
impl crate::GatewayAppDataSource for OAuthTestDataSource {}
impl crate::MediaAppDataSource for OAuthTestDataSource {}
impl crate::VoiceAppDataSource for OAuthTestDataSource {}
impl crate::PluginDataSource for OAuthTestDataSource {}
impl crate::KnowledgeAppDataSource for OAuthTestDataSource {}
impl crate::AutomationOverviewAppDataSource for OAuthTestDataSource {}
impl crate::AutomationManagementAppDataSource for OAuthTestDataSource {}
impl crate::MemoryAppDataSource for OAuthTestDataSource {}
impl crate::DiagnosticsAppDataSource for OAuthTestDataSource {}
impl crate::UsageStatsAppDataSource for OAuthTestDataSource {}
impl crate::ModelProviderAppDataSource for OAuthTestDataSource {}
impl crate::ConnectAppDataSource for OAuthTestDataSource {}
impl crate::RightSurfaceAppDataSource for OAuthTestDataSource {}

#[async_trait]
impl crate::McpAppDataSource for OAuthTestDataSource {
    async fn login_mcp_server_oauth(
        &self,
        _params: app_server_protocol::McpServerOauthLoginParams,
    ) -> Result<lime_mcp::McpOAuthLoginHandle, RuntimeCoreError> {
        self.login
            .lock()
            .expect("OAuth test login mutex poisoned")
            .take()
            .ok_or_else(|| RuntimeCoreError::Backend("OAuth test login missing".to_string()))
    }
}

fn oauth_test_data_source() -> (
    Arc<OAuthTestDataSource>,
    tokio::sync::oneshot::Sender<Result<(), String>>,
) {
    let (completion_tx, completion_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
    let handle = lime_mcp::McpOAuthLoginHandle::new(
        "https://auth.example/authorize",
        "pending",
        async move {
            completion_rx
                .await
                .map_err(|error| lime_mcp::McpError::ProtocolError(error.to_string()))?
                .map_err(lime_mcp::McpError::ConfigError)
        },
    );
    (
        Arc::new(OAuthTestDataSource {
            login: std::sync::Mutex::new(Some(handle)),
        }),
        completion_tx,
    )
}

async fn start_oauth_test() -> (
    RequestProcessor,
    tokio::sync::oneshot::Sender<Result<(), String>>,
    tokio::sync::mpsc::UnboundedReceiver<app_server_protocol::protocol::v2::ServerNotification>,
) {
    let (data_source, completion_tx) = oauth_test_data_source();
    let runtime = RuntimeCore::default().with_app_data_source(data_source);
    let (notification_tx, notification_rx) = tokio::sync::mpsc::unbounded_channel();
    let hook: ServerNotificationHook = Arc::new(move |notification| {
        let notification_tx = notification_tx.clone();
        Box::pin(async move {
            let _ = notification_tx.send(notification);
        })
    });
    let processor = RequestProcessor::new(runtime).with_server_notification_hook(hook);
    initialize_processor(&processor).await;
    (processor, completion_tx, notification_rx)
}

#[tokio::test]
async fn mcp_list_methods_require_initialized_and_return_current_empty_state() {
    let processor = RequestProcessor::new(RuntimeCore::default());
    let blocked = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(1),
            METHOD_MCP_TOOL_LIST,
            Some(json!({})),
        ))
        .await
        .expect("blocked response");
    assert!(matches!(
        &blocked[0],
        JsonRpcMessage::Error(error) if error.error.code == error_codes::NOT_INITIALIZED
    ));

    processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(2),
            METHOD_INITIALIZE,
            Some(
                serde_json::to_value(InitializeParams {
                    client_info: ClientInfo {
                        name: "test-client".to_string(),
                        title: None,
                        version: None,
                    },
                    capabilities: ClientCapabilities::default(),
                })
                .expect("initialize params"),
            ),
        ))
        .await
        .expect("initialize");
    processor.handle_notification(JsonRpcNotification::new(
        METHOD_INITIALIZED,
        Some(json!({})),
    ));

    let cases = [
        (RequestId::Integer(3), METHOD_MCP_SERVER_LIST, "servers"),
        (
            RequestId::Integer(4),
            METHOD_MCP_SERVER_STATUS_LIST,
            "servers",
        ),
        (RequestId::Integer(5), METHOD_MCP_TOOL_LIST, "tools"),
        (RequestId::Integer(6), METHOD_MCP_PROMPT_LIST, "prompts"),
        (RequestId::Integer(7), METHOD_MCP_RESOURCE_LIST, "resources"),
    ];

    for (id, method, field) in cases {
        let messages = processor
            .handle_request(JsonRpcRequest::new(id, method, Some(json!({}))))
            .await
            .expect("mcp list response");

        match &messages[0] {
            JsonRpcMessage::Response(response) => {
                assert_eq!(response.result[field], json!([]));
            }
            other => panic!("expected response, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn mcp_runtime_methods_require_initialized_and_fail_closed_without_manager() {
    let processor = RequestProcessor::new(RuntimeCore::default());
    let blocked = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(1),
            METHOD_MCP_TOOL_CALL,
            Some(json!({
                "toolName": "mcp__docs__search",
                "arguments": {},
            })),
        ))
        .await
        .expect("blocked response");
    assert!(matches!(
        &blocked[0],
        JsonRpcMessage::Error(error) if error.error.code == error_codes::NOT_INITIALIZED
    ));

    processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(2),
            METHOD_INITIALIZE,
            Some(
                serde_json::to_value(InitializeParams {
                    client_info: ClientInfo {
                        name: "test-client".to_string(),
                        title: None,
                        version: None,
                    },
                    capabilities: ClientCapabilities::default(),
                })
                .expect("initialize params"),
            ),
        ))
        .await
        .expect("initialize");
    processor.handle_notification(JsonRpcNotification::new(
        METHOD_INITIALIZED,
        Some(json!({})),
    ));

    let cases = [
        (
            RequestId::Integer(3),
            METHOD_MCP_SERVER_CREATE,
            json!({
                "server": {
                    "id": "server-1",
                    "name": "docs",
                    "server_config": { "command": "node" },
                    "enabled_lime": true,
                    "enabled_claude": false,
                    "enabled_codex": true,
                    "enabled_gemini": false,
                }
            }),
        ),
        (
            RequestId::Integer(4),
            METHOD_MCP_SERVER_UPDATE,
            json!({
                "server": {
                    "id": "server-1",
                    "name": "docs",
                    "server_config": { "command": "node" },
                    "enabled_lime": true,
                    "enabled_claude": false,
                    "enabled_codex": true,
                    "enabled_gemini": false,
                }
            }),
        ),
        (
            RequestId::Integer(5),
            METHOD_MCP_SERVER_DELETE,
            json!({ "id": "server-1" }),
        ),
        (
            RequestId::Integer(6),
            METHOD_MCP_SERVER_ENABLED_SET,
            json!({ "id": "server-1", "appType": "codex", "enabled": true }),
        ),
        (
            RequestId::Integer(7),
            METHOD_MCP_SERVER_IMPORT_FROM_APP,
            json!({ "appType": "codex" }),
        ),
        (
            RequestId::Integer(8),
            METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE,
            json!({}),
        ),
        (
            RequestId::Integer(9),
            METHOD_MCP_SERVER_START,
            json!({ "name": "docs" }),
        ),
        (
            RequestId::Integer(10),
            METHOD_MCP_SERVER_STOP,
            json!({ "name": "docs" }),
        ),
        (
            RequestId::Integer(11),
            METHOD_MCP_TOOL_CALL,
            json!({ "toolName": "mcp__docs__search", "arguments": {} }),
        ),
        (
            RequestId::Integer(12),
            METHOD_MCP_TOOL_CALL_WITH_CALLER,
            json!({
                "toolName": "mcp__docs__search",
                "arguments": {},
                "caller": "assistant",
            }),
        ),
        (
            RequestId::Integer(13),
            METHOD_MCP_PROMPT_GET,
            json!({ "server": "docs", "name": "docs_prompt", "arguments": {} }),
        ),
        (
            RequestId::Integer(14),
            METHOD_MCP_RESOURCE_READ,
            json!({ "server": "docs", "uri": "docs://readme" }),
        ),
        (
            RequestId::Integer(15),
            METHOD_MCP_RESOURCE_SUBSCRIBE,
            json!({ "server": "docs", "uri": "docs://readme" }),
        ),
        (
            RequestId::Integer(16),
            METHOD_MCP_RESOURCE_UNSUBSCRIBE,
            json!({ "server": "docs", "uri": "docs://readme" }),
        ),
    ];

    for (id, method, params) in cases {
        let messages = processor
            .handle_request(JsonRpcRequest::new(id, method, Some(params)))
            .await
            .expect("mcp runtime response");

        match &messages[0] {
            JsonRpcMessage::Error(error) => {
                assert_eq!(error.error.code, error_codes::RUNTIME_ERROR);
            }
            other => panic!("expected runtime error, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn mcp_oauth_login_returns_before_typed_success_notification() {
    let (processor, completion_tx, mut notification_rx) = start_oauth_test().await;
    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(30),
            METHOD_MCP_SERVER_OAUTH_LOGIN,
            Some(json!({ "name": "remote-docs" })),
        ))
        .await
        .expect("OAuth login response");

    let [JsonRpcMessage::Response(response)] = messages.as_slice() else {
        panic!("expected OAuth login response, got {messages:?}");
    };
    assert_eq!(
        response.result["authorizationUrl"],
        "https://auth.example/authorize"
    );
    assert_eq!(response.result["state"], "pending");

    completion_tx.send(Ok(())).expect("complete OAuth login");
    let notification =
        tokio::time::timeout(std::time::Duration::from_secs(1), notification_rx.recv())
            .await
            .expect("OAuth success notification timeout")
            .expect("OAuth success notification channel closed");
    assert_eq!(
        notification,
        app_server_protocol::protocol::v2::ServerNotification::McpServerOauthLoginCompleted(
            app_server_protocol::protocol::v2::McpServerOauthLoginCompletedNotification {
                name: "remote-docs".to_string(),
                thread_id: None,
                success: true,
                error: None,
            }
        )
    );
}

#[tokio::test]
async fn mcp_oauth_login_publishes_typed_failure_notification() {
    let (processor, completion_tx, mut notification_rx) = start_oauth_test().await;
    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(31),
            METHOD_MCP_SERVER_OAUTH_LOGIN,
            Some(json!({ "name": "remote-docs" })),
        ))
        .await
        .expect("OAuth login response");
    assert!(matches!(messages.as_slice(), [JsonRpcMessage::Response(_)]));

    completion_tx
        .send(Err("scope rejected".to_string()))
        .expect("fail OAuth login");
    let notification =
        tokio::time::timeout(std::time::Duration::from_secs(1), notification_rx.recv())
            .await
            .expect("OAuth failure notification timeout")
            .expect("OAuth failure notification channel closed");
    let app_server_protocol::protocol::v2::ServerNotification::McpServerOauthLoginCompleted(
        notification,
    ) = notification
    else {
        panic!("expected OAuth completion notification");
    };
    assert_eq!(notification.name, "remote-docs");
    assert_eq!(notification.thread_id, None);
    assert!(!notification.success);
    assert!(notification
        .error
        .as_deref()
        .is_some_and(|error| error.contains("scope rejected")));
}
