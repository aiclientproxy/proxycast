//! mcp request processor tests.

use super::super::*;
use super::tests_support::initialize_processor;
use app_server_protocol::protocol::v2::METHOD_MCP_SERVER_TOOL_CALL;
use app_server_protocol::{
    ClientCapabilities, JsonRpcMessage, RequestId, METHOD_INITIALIZE, METHOD_INITIALIZED,
    METHOD_MCP_PROMPT_GET, METHOD_MCP_PROMPT_LIST, METHOD_MCP_RESOURCE_LIST,
    METHOD_MCP_RESOURCE_SUBSCRIBE, METHOD_MCP_RESOURCE_UNSUBSCRIBE, METHOD_MCP_SERVER_CREATE,
    METHOD_MCP_SERVER_DELETE, METHOD_MCP_SERVER_ENABLED_SET, METHOD_MCP_SERVER_IMPORT_FROM_APP,
    METHOD_MCP_SERVER_LIST, METHOD_MCP_SERVER_OAUTH_LOGIN, METHOD_MCP_SERVER_START,
    METHOD_MCP_SERVER_STATUS_LIST, METHOD_MCP_SERVER_STOP, METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE,
    METHOD_MCP_SERVER_UPDATE, METHOD_MCP_TOOL_LIST,
};
use async_trait::async_trait;
use serde_json::json;

#[derive(Default)]
struct McpNotificationTestDataSource {
    login: std::sync::Mutex<Option<lime_mcp::McpOAuthLoginHandle>>,
    start_result: std::sync::Mutex<Option<Result<(), String>>>,
}

impl crate::SessionAppDataSource for McpNotificationTestDataSource {}
impl crate::WorkspaceAppDataSource for McpNotificationTestDataSource {}
impl crate::SkillAppDataSource for McpNotificationTestDataSource {}
impl crate::WorkspaceSkillBindingAppDataSource for McpNotificationTestDataSource {}
impl crate::GatewayAppDataSource for McpNotificationTestDataSource {}
impl crate::MediaAppDataSource for McpNotificationTestDataSource {}
impl crate::VoiceAppDataSource for McpNotificationTestDataSource {}
impl crate::PluginDataSource for McpNotificationTestDataSource {}
impl crate::KnowledgeAppDataSource for McpNotificationTestDataSource {}
impl crate::AutomationOverviewAppDataSource for McpNotificationTestDataSource {}
impl crate::AutomationManagementAppDataSource for McpNotificationTestDataSource {}
impl crate::MemoryAppDataSource for McpNotificationTestDataSource {}
impl crate::DiagnosticsAppDataSource for McpNotificationTestDataSource {}
impl crate::UsageStatsAppDataSource for McpNotificationTestDataSource {}
impl crate::ModelProviderAppDataSource for McpNotificationTestDataSource {}
impl crate::ConnectAppDataSource for McpNotificationTestDataSource {}
impl crate::RightSurfaceAppDataSource for McpNotificationTestDataSource {}

#[async_trait]
impl crate::McpAppDataSource for McpNotificationTestDataSource {
    async fn start_mcp_server(
        &self,
        _params: app_server_protocol::McpServerStartParams,
    ) -> Result<app_server_protocol::McpServerLifecycleResponse, RuntimeCoreError> {
        self.start_result
            .lock()
            .expect("MCP test start mutex poisoned")
            .take()
            .ok_or_else(|| RuntimeCoreError::Backend("MCP test start result missing".to_string()))?
            .map(|()| app_server_protocol::McpServerLifecycleResponse::default())
            .map_err(RuntimeCoreError::Backend)
    }

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
    Arc<McpNotificationTestDataSource>,
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
        Arc::new(McpNotificationTestDataSource {
            login: std::sync::Mutex::new(Some(handle)),
            start_result: std::sync::Mutex::new(None),
        }),
        completion_tx,
    )
}

async fn start_mcp_notification_test(
    start_result: Result<(), String>,
) -> (
    RequestProcessor,
    tokio::sync::mpsc::UnboundedReceiver<app_server_protocol::protocol::v2::ServerNotification>,
) {
    let data_source = Arc::new(McpNotificationTestDataSource {
        login: std::sync::Mutex::new(None),
        start_result: std::sync::Mutex::new(Some(start_result)),
    });
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
    (processor, notification_rx)
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
            METHOD_MCP_SERVER_TOOL_CALL,
            Some(json!({
                "threadId": "thread-missing",
                "server": "docs",
                "tool": "search",
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
            RequestId::Integer(13),
            METHOD_MCP_PROMPT_GET,
            json!({ "server": "docs", "name": "docs_prompt", "arguments": {} }),
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
async fn mcp_start_publishes_starting_then_ready_before_success_response() {
    let (processor, mut notification_rx) = start_mcp_notification_test(Ok(())).await;
    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(28),
            METHOD_MCP_SERVER_START,
            Some(json!({ "name": "remote-docs" })),
        ))
        .await
        .expect("MCP start response");
    assert!(matches!(messages.as_slice(), [JsonRpcMessage::Response(_)]));

    let starting = notification_rx.recv().await.expect("starting notification");
    let ready = notification_rx.recv().await.expect("ready notification");
    assert_eq!(
        starting,
        app_server_protocol::protocol::v2::ServerNotification::McpServerStatusUpdated(
            app_server_protocol::protocol::v2::McpServerStatusUpdatedNotification {
                thread_id: None,
                name: "remote-docs".to_string(),
                status: app_server_protocol::protocol::v2::McpServerStartupState::Starting,
                error: None,
                failure_reason: None,
            }
        )
    );
    assert_eq!(
        ready,
        app_server_protocol::protocol::v2::ServerNotification::McpServerStatusUpdated(
            app_server_protocol::protocol::v2::McpServerStatusUpdatedNotification {
                thread_id: None,
                name: "remote-docs".to_string(),
                status: app_server_protocol::protocol::v2::McpServerStartupState::Ready,
                error: None,
                failure_reason: None,
            }
        )
    );
    assert!(notification_rx.try_recv().is_err());
}

#[tokio::test]
async fn mcp_start_publishes_starting_then_failed_before_error_response() {
    let (processor, mut notification_rx) =
        start_mcp_notification_test(Err("handshake rejected".to_string())).await;
    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(29),
            METHOD_MCP_SERVER_START,
            Some(json!({ "name": "remote-docs" })),
        ))
        .await
        .expect("MCP start error response");
    assert!(matches!(
        messages.as_slice(),
        [JsonRpcMessage::Error(error)] if error.error.code == error_codes::RUNTIME_ERROR
    ));

    let starting = notification_rx.recv().await.expect("starting notification");
    let failed = notification_rx.recv().await.expect("failed notification");
    let app_server_protocol::protocol::v2::ServerNotification::McpServerStatusUpdated(starting) =
        starting
    else {
        panic!("expected starting status notification");
    };
    assert_eq!(
        starting.status,
        app_server_protocol::protocol::v2::McpServerStartupState::Starting
    );
    let app_server_protocol::protocol::v2::ServerNotification::McpServerStatusUpdated(failed) =
        failed
    else {
        panic!("expected failed status notification");
    };
    assert_eq!(
        failed.status,
        app_server_protocol::protocol::v2::McpServerStartupState::Failed
    );
    assert_eq!(failed.thread_id, None);
    assert_eq!(failed.name, "remote-docs");
    assert!(failed
        .error
        .as_deref()
        .is_some_and(|error| error.contains("handshake rejected")));
    assert_eq!(failed.failure_reason, None);
    assert!(notification_rx.try_recv().is_err());
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
