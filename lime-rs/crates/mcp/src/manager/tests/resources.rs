use crate::manager::McpClientManager;
use crate::types::McpError;
use rmcp::{
    model::{
        AnnotateAble, ListResourcesResult, ListToolsResult, PaginatedRequestParam, RawResource,
        ReadResourceRequestParam, ReadResourceResult, ResourceContents, ServerCapabilities,
        ServerInfo, SubscribeRequestParam, UnsubscribeRequestParam,
    },
    service::RequestContext,
    RoleServer, ServerHandler, ServiceExt,
};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
struct ExactTargetResourceServer {
    name: &'static str,
    subscriptions: Arc<Mutex<Vec<String>>>,
    unsubscriptions: Arc<Mutex<Vec<String>>>,
    read_meta: Arc<Mutex<Vec<serde_json::Value>>>,
}

impl ServerHandler for ExactTargetResourceServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .enable_resources_subscribe()
                .build(),
            ..Default::default()
        }
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, rmcp::ErrorData> {
        let tool = serde_json::from_value(serde_json::json!({
            "name": "calendar_search",
            "description": "Search calendar events",
            "inputSchema": {"type": "object"},
            "_meta": {
                "connector_id": "calendar",
                "connector_name": "Calendar",
                "link_id": "link-calendar",
                "ui": {"resourceUri": "ui://calendar/event"},
                "_codex_apps": {
                    "resource_uri": "/calendar/link-calendar/calendar_search",
                    "requires_explicit_link_id": true
                }
            }
        }))
        .map_err(|error| rmcp::ErrorData::internal_error(error.to_string(), None))?;
        Ok(ListToolsResult::with_all_items(vec![tool]))
    }

    async fn list_resources(
        &self,
        request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, rmcp::ErrorData> {
        let page = request.and_then(|request| request.cursor);
        let (suffix, next_cursor) = match page.as_deref() {
            None => ("first", Some("next".to_string())),
            Some("next") => ("second", None),
            Some(_) => return Ok(ListResourcesResult::default()),
        };
        Ok(ListResourcesResult {
            meta: None,
            next_cursor,
            resources: vec![RawResource::new(
                format!("skill://{}/{suffix}", self.name),
                suffix.to_string(),
            )
            .no_annotation()],
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParam,
        context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, rmcp::ErrorData> {
        self.read_meta
            .lock()
            .await
            .push(serde_json::Value::Object(context.meta.0));
        Ok(ReadResourceResult {
            contents: vec![ResourceContents::text(self.name, request.uri)],
        })
    }

    async fn subscribe(
        &self,
        request: SubscribeRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<(), rmcp::ErrorData> {
        self.subscriptions.lock().await.push(request.uri);
        Ok(())
    }

    async fn unsubscribe(
        &self,
        request: UnsubscribeRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<(), rmcp::ErrorData> {
        self.unsubscriptions.lock().await.push(request.uri);
        Ok(())
    }
}

async fn add_exact_target_server(
    manager: &McpClientManager,
    name: &'static str,
) -> (
    Arc<Mutex<Vec<String>>>,
    Arc<Mutex<Vec<String>>>,
    Arc<Mutex<Vec<serde_json::Value>>>,
    tokio::task::JoinHandle<()>,
) {
    add_exact_target_server_with_disabled_tools(manager, name, Vec::new()).await
}

async fn add_exact_target_server_with_disabled_tools(
    manager: &McpClientManager,
    name: &'static str,
    disabled_tools: Vec<String>,
) -> (
    Arc<Mutex<Vec<String>>>,
    Arc<Mutex<Vec<String>>>,
    Arc<Mutex<Vec<serde_json::Value>>>,
    tokio::task::JoinHandle<()>,
) {
    let subscriptions = Arc::new(Mutex::new(Vec::new()));
    let unsubscriptions = Arc::new(Mutex::new(Vec::new()));
    let read_meta = Arc::new(Mutex::new(Vec::new()));
    let server = ExactTargetResourceServer {
        name,
        subscriptions: subscriptions.clone(),
        unsubscriptions: unsubscriptions.clone(),
        read_meta: read_meta.clone(),
    };
    let (server_transport, client_transport) = tokio::io::duplex(4096);
    let server_handle = tokio::spawn(async move {
        let service = server
            .serve(server_transport)
            .await
            .expect("start exact target server");
        service
            .waiting()
            .await
            .expect("wait for exact target server");
    });
    let client = crate::client_service::LimeMcpClientService::new(name.to_string(), None)
        .serve(client_transport)
        .await
        .expect("start exact target client");
    let mut wrapper = super::common::create_test_client(name);
    wrapper.config.disabled_tools = disabled_tools;
    wrapper.set_running_service(client);
    manager
        .add_client(name.to_string(), wrapper)
        .await
        .expect("add exact target client");
    (subscriptions, unsubscriptions, read_meta, server_handle)
}

#[tokio::test]
async fn test_list_resources_returns_empty_when_no_servers() {
    let manager = McpClientManager::new(None);

    // 没有运行的服务器时，应该返回空列表
    let result = manager.list_resources().await.unwrap();
    assert!(result.is_empty());
}

#[tokio::test]
async fn test_list_resource_templates_returns_empty_when_no_servers() {
    let manager = McpClientManager::new(None);

    // 没有运行的服务器时，应该返回空列表
    let result = manager.list_resource_templates().await.unwrap();
    assert!(result.is_empty());
}

#[tokio::test]
async fn list_resource_page_preserves_exact_server_cursor() {
    let manager = McpClientManager::new(None);
    let (_, _, _, server) = add_exact_target_server(&manager, "server-a").await;

    let first = manager
        .list_resource_page("server-a", None)
        .await
        .expect("first page");
    assert_eq!(first.resources[0].uri, "skill://server-a/first");
    assert_eq!(first.next_cursor.as_deref(), Some("next"));

    let second = manager
        .list_resource_page("server-a", first.next_cursor)
        .await
        .expect("second page");
    assert_eq!(second.resources[0].uri, "skill://server-a/second");
    assert!(second.next_cursor.is_none());

    manager
        .stop_server("server-a")
        .await
        .expect("stop server-a");
    server.await.expect("join server-a");
}

#[tokio::test]
async fn test_read_resource_not_found() {
    let manager = McpClientManager::new(None);

    // 读取不存在的资源
    let result = manager
        .read_resource("missing-server", "file:///nonexistent/resource")
        .await;

    // 应该返回错误
    assert!(result.is_err());
    match result {
        Err(McpError::ServerNotRunning(name)) => {
            assert_eq!(name, "missing-server");
        }
        _ => panic!("Expected ServerNotRunning error"),
    }
}

#[tokio::test]
async fn test_subscribe_resource_not_found() {
    let manager = McpClientManager::new(None);

    let result = manager
        .subscribe_resource("missing-server", "file:///nonexistent/resource")
        .await;

    assert!(matches!(result, Err(McpError::ServerNotRunning(_))));
}

#[tokio::test]
async fn test_unsubscribe_resource_not_found() {
    let manager = McpClientManager::new(None);

    let result = manager
        .unsubscribe_resource("missing-server", "file:///nonexistent/resource")
        .await;

    assert!(matches!(result, Err(McpError::ServerNotRunning(_))));
}

#[tokio::test]
async fn resource_operations_reject_empty_exact_target_before_dispatch() {
    let manager = McpClientManager::new(None);

    assert!(matches!(
        manager.read_resource(" ", "docs://readme").await,
        Err(McpError::ConfigError(_))
    ));
    assert!(matches!(
        manager.subscribe_resource("docs", " ").await,
        Err(McpError::ConfigError(_))
    ));
    assert!(matches!(
        manager.unsubscribe_resource(" ", " ").await,
        Err(McpError::ConfigError(_))
    ));
}

#[tokio::test]
async fn resource_operations_route_same_uri_to_exact_server() {
    let manager = McpClientManager::new(None);
    let (subscriptions_a, unsubscriptions_a, _, server_a) =
        add_exact_target_server(&manager, "server-a").await;
    let (subscriptions_b, unsubscriptions_b, _, server_b) =
        add_exact_target_server(&manager, "server-b").await;
    let uri = "docs://shared";

    let content = manager
        .read_resource("server-b", uri)
        .await
        .expect("read exact server resource");
    manager
        .subscribe_resource("server-b", uri)
        .await
        .expect("subscribe exact server resource");
    manager
        .unsubscribe_resource("server-b", uri)
        .await
        .expect("unsubscribe exact server resource");

    assert_eq!(content.text.as_deref(), Some("server-b"));
    assert!(subscriptions_a.lock().await.is_empty());
    assert!(unsubscriptions_a.lock().await.is_empty());
    assert_eq!(subscriptions_b.lock().await.as_slice(), [uri]);
    assert_eq!(unsubscriptions_b.lock().await.as_slice(), [uri]);

    manager
        .stop_server("server-a")
        .await
        .expect("stop server-a");
    manager
        .stop_server("server-b")
        .await
        .expect("stop server-b");
    server_a.await.expect("join server-a");
    server_b.await.expect("join server-b");
}

#[tokio::test]
async fn resource_read_preserves_request_meta_on_the_mcp_wire() {
    let manager = McpClientManager::new(None);
    let (_, _, read_meta, server) = add_exact_target_server(&manager, "server-a").await;
    let expected = serde_json::json!({
        "x-codex-turn-metadata": {
            "mcp_request_meta": {
                "selected_connector_ids": ["calendar"],
            },
        },
    });

    manager
        .read_resource_with_meta("server-a", "ui://calendar/event", Some(expected.clone()))
        .await
        .expect("read resource with request metadata");

    let read_meta = read_meta.lock().await;
    assert_eq!(
        read_meta[0].pointer("/x-codex-turn-metadata/mcp_request_meta/selected_connector_ids/0"),
        expected.pointer("/x-codex-turn-metadata/mcp_request_meta/selected_connector_ids/0")
    );
    manager
        .stop_server("server-a")
        .await
        .expect("stop server-a");
    server.await.expect("join server-a");
}

#[tokio::test]
async fn origin_resource_read_revalidates_current_tool_authority_and_scopes_wire_meta() {
    let manager = McpClientManager::new(None);
    let (_, _, read_meta, server) = add_exact_target_server(&manager, "codex_apps").await;
    let origin = crate::types::McpResourceOrigin {
        call_id: "item-calendar".to_string(),
        turn_id: Some("turn-1".to_string()),
        tool: "calendar_search".to_string(),
        connector_id: "calendar".to_string(),
        link_id: Some("link-calendar".to_string()),
        uri: "ui://calendar/event".to_string(),
        ambiguous_account: false,
    };

    manager
        .read_resource_for_origin("codex_apps", "thread-1", &origin)
        .await
        .expect("origin-scoped resource read");
    let observed = read_meta.lock().await;
    assert_eq!(observed[0]["threadId"], "thread-1");
    assert_eq!(
        observed[0].pointer("/x-codex-turn-metadata/mcp_request_meta/selected_connector_ids/0"),
        Some(&serde_json::json!("calendar"))
    );
    assert_eq!(
        observed[0].pointer("/x-codex-turn-metadata/mcp_request_meta/link_id"),
        Some(&serde_json::json!("link-calendar"))
    );
    drop(observed);

    let mut stale = origin.clone();
    stale.link_id = Some("link-other".to_string());
    let error = manager
        .read_resource_for_origin("codex_apps", "thread-1", &stale)
        .await
        .expect_err("stale account authority must fail closed");
    assert!(error.to_string().contains("link does not match"));
    assert_eq!(read_meta.lock().await.len(), 1);

    let mut stale_connector = origin.clone();
    stale_connector.connector_id = "mail".to_string();
    let error = manager
        .read_resource_for_origin("codex_apps", "thread-1", &stale_connector)
        .await
        .expect_err("stale connector authority must fail closed");
    assert!(error.to_string().contains("connector does not match"));

    let mut ambiguous = origin.clone();
    ambiguous.ambiguous_account = true;
    let error = manager
        .read_resource_for_origin("codex_apps", "thread-1", &ambiguous)
        .await
        .expect_err("ambiguous account must fail closed");
    assert!(error.to_string().contains("ambiguous account"));

    let error = manager
        .read_resource_for_origin("docs", "thread-1", &origin)
        .await
        .expect_err("non-app origin must fail closed");
    assert!(error.to_string().contains("only valid for codex_apps"));
    assert_eq!(read_meta.lock().await.len(), 1);

    manager
        .stop_server("codex_apps")
        .await
        .expect("stop codex_apps");
    server.await.expect("join codex_apps");
}

#[tokio::test]
async fn origin_resource_read_rejects_tools_disabled_in_current_catalog() {
    let manager = McpClientManager::new(None);
    let (_, _, read_meta, server) = add_exact_target_server_with_disabled_tools(
        &manager,
        "codex_apps",
        vec!["calendar_search".to_string()],
    )
    .await;
    let origin = crate::types::McpResourceOrigin {
        call_id: "item-calendar".to_string(),
        turn_id: Some("turn-1".to_string()),
        tool: "calendar_search".to_string(),
        connector_id: "calendar".to_string(),
        link_id: Some("link-calendar".to_string()),
        uri: "ui://calendar/event".to_string(),
        ambiguous_account: false,
    };

    let error = manager
        .read_resource_for_origin("codex_apps", "thread-1", &origin)
        .await
        .expect_err("disabled current tool must fail closed");
    assert!(error.to_string().contains("calendar_search"));
    assert!(read_meta.lock().await.is_empty());

    manager
        .stop_server("codex_apps")
        .await
        .expect("stop codex_apps");
    server.await.expect("join codex_apps");
}

#[test]
fn test_convert_resource_to_definition() {
    use rmcp::model::{AnnotateAble, RawResource};

    // 创建一个 rmcp Resource
    let raw_resource = RawResource {
        uri: "file:///test/resource.txt".to_string(),
        name: "resource.txt".to_string(),
        title: Some("Test Resource".to_string()),
        description: Some("A test resource".to_string()),
        mime_type: Some("text/plain".to_string()),
        size: Some(1024),
        icons: None,
        meta: None,
    };
    let resource = raw_resource.no_annotation();

    // 转换为 McpResourceDefinition
    let definition =
        McpClientManager::convert_resource_to_definition(resource, "test_server".to_string());

    // 验证转换结果
    assert_eq!(definition.uri, "file:///test/resource.txt");
    assert_eq!(definition.name, "resource.txt");
    assert_eq!(definition.description, Some("A test resource".to_string()));
    assert_eq!(definition.mime_type, Some("text/plain".to_string()));
    assert_eq!(definition.server_name, "test_server");
}

#[test]
fn test_convert_resource_to_definition_minimal() {
    use rmcp::model::{AnnotateAble, RawResource};

    // 创建一个最小的 rmcp Resource（只有必需字段）
    let raw_resource =
        RawResource::new("file:///minimal.txt".to_string(), "minimal.txt".to_string());
    let resource = raw_resource.no_annotation();

    // 转换为 McpResourceDefinition
    let definition =
        McpClientManager::convert_resource_to_definition(resource, "server1".to_string());

    // 验证转换结果
    assert_eq!(definition.uri, "file:///minimal.txt");
    assert_eq!(definition.name, "minimal.txt");
    assert!(definition.description.is_none());
    assert!(definition.mime_type.is_none());
    assert_eq!(definition.server_name, "server1");
}

#[test]
fn test_convert_resource_template_to_definition() {
    use rmcp::model::{AnnotateAble, RawResourceTemplate};

    let raw_template = RawResourceTemplate {
        uri_template: "file:///{path}".to_string(),
        name: "workspace-file".to_string(),
        title: Some("Workspace File".to_string()),
        description: Some("Read a workspace file by path".to_string()),
        mime_type: Some("text/plain".to_string()),
    };
    let template = raw_template.no_annotation();

    let definition =
        McpClientManager::convert_resource_template_to_definition(template, "docs".to_string());

    assert_eq!(definition.uri_template, "file:///{path}");
    assert_eq!(definition.name, "workspace-file");
    assert_eq!(definition.title, Some("Workspace File".to_string()));
    assert_eq!(
        definition.description,
        Some("Read a workspace file by path".to_string())
    );
    assert_eq!(definition.mime_type, Some("text/plain".to_string()));
    assert_eq!(definition.server_name, "docs");
}

#[test]
fn test_convert_read_resource_result_text() {
    // 创建文本资源内容
    let result = rmcp::model::ReadResourceResult {
        contents: vec![rmcp::model::ResourceContents::text(
            "Hello, World!",
            "file:///test.txt",
        )],
    };

    // 转换为 McpResourceContent
    let mcp_content = McpClientManager::convert_read_resource_result("file:///test.txt", result);

    // 验证转换结果
    assert_eq!(mcp_content.uri, "file:///test.txt");
    assert_eq!(mcp_content.mime_type, Some("text".to_string()));
    assert_eq!(mcp_content.text, Some("Hello, World!".to_string()));
    assert!(mcp_content.blob.is_none());
    assert!(mcp_content.meta.is_none());
}

#[test]
fn test_convert_read_resource_result_preserves_meta() {
    let mut meta = rmcp::model::Meta::new();
    meta.0.insert(
        "ui".to_string(),
        serde_json::json!({
            "csp": {
                "connectDomains": ["https://api.example.com"],
                "resourceDomains": ["https://cdn.example.com"]
            }
        }),
    );
    let result = rmcp::model::ReadResourceResult {
        contents: vec![rmcp::model::ResourceContents::TextResourceContents {
            uri: "ui://plugin/report.html".to_string(),
            mime_type: Some("text/html".to_string()),
            text: "<main>report</main>".to_string(),
            meta: Some(meta),
        }],
    };

    let content = McpClientManager::convert_read_resource_result("ui://plugin/report.html", result);

    assert_eq!(
        content.meta,
        Some(serde_json::json!({
            "ui": {
                "csp": {
                    "connectDomains": ["https://api.example.com"],
                    "resourceDomains": ["https://cdn.example.com"]
                }
            }
        }))
    );
}

#[test]
fn test_convert_read_resource_result_blob() {
    // 创建二进制资源内容
    let result = rmcp::model::ReadResourceResult {
        contents: vec![rmcp::model::ResourceContents::BlobResourceContents {
            uri: "file:///test.bin".to_string(),
            mime_type: Some("application/octet-stream".to_string()),
            blob: "base64encodeddata".to_string(),
            meta: None,
        }],
    };

    // 转换为 McpResourceContent
    let mcp_content = McpClientManager::convert_read_resource_result("file:///test.bin", result);

    // 验证转换结果
    assert_eq!(mcp_content.uri, "file:///test.bin");
    assert_eq!(
        mcp_content.mime_type,
        Some("application/octet-stream".to_string())
    );
    assert!(mcp_content.text.is_none());
    assert_eq!(mcp_content.blob, Some("base64encodeddata".to_string()));
    assert!(mcp_content.meta.is_none());
}

#[test]
fn test_convert_read_resource_result_empty() {
    // 创建空的资源结果
    let result = rmcp::model::ReadResourceResult { contents: vec![] };

    // 转换为 McpResourceContent
    let mcp_content = McpClientManager::convert_read_resource_result("file:///empty.txt", result);

    // 验证转换结果（应该返回空内容）
    assert_eq!(mcp_content.uri, "file:///empty.txt");
    assert!(mcp_content.mime_type.is_none());
    assert!(mcp_content.text.is_none());
    assert!(mcp_content.blob.is_none());
    assert!(mcp_content.meta.is_none());
}
