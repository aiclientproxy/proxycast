use std::sync::Arc;

use app_server::{
    AppServer, AutomationManagementAppDataSource, AutomationOverviewAppDataSource,
    ConnectAppDataSource, DiagnosticsAppDataSource, GatewayAppDataSource, KnowledgeAppDataSource,
    McpAppDataSource, MediaAppDataSource, MemoryAppDataSource, MockBackend,
    ModelProviderAppDataSource, PluginDataSource, RightSurfaceAppDataSource, RuntimeCore,
    RuntimeCoreError, SessionAppDataSource, SkillAppDataSource, UsageStatsAppDataSource,
    VoiceAppDataSource, WorkspaceAppDataSource, WorkspaceSkillBindingAppDataSource,
};
use app_server_protocol::protocol::v2::{
    ModelProviderCapabilitiesReadResponse, METHOD_MODEL_PROVIDER_CAPABILITIES_READ,
};
use app_server_protocol::{METHOD_INITIALIZE, METHOD_INITIALIZED, PROTOCOL_VERSION};
use async_trait::async_trait;
use serde_json::{json, Value};

struct CapabilityDataSource;

impl SessionAppDataSource for CapabilityDataSource {}
impl WorkspaceAppDataSource for CapabilityDataSource {}
impl SkillAppDataSource for CapabilityDataSource {}
impl WorkspaceSkillBindingAppDataSource for CapabilityDataSource {}
impl GatewayAppDataSource for CapabilityDataSource {}
impl MediaAppDataSource for CapabilityDataSource {}
impl VoiceAppDataSource for CapabilityDataSource {}
impl PluginDataSource for CapabilityDataSource {}
impl KnowledgeAppDataSource for CapabilityDataSource {}
impl AutomationOverviewAppDataSource for CapabilityDataSource {}
impl McpAppDataSource for CapabilityDataSource {}
impl AutomationManagementAppDataSource for CapabilityDataSource {}
impl MemoryAppDataSource for CapabilityDataSource {}
impl DiagnosticsAppDataSource for CapabilityDataSource {}
impl UsageStatsAppDataSource for CapabilityDataSource {}
impl ConnectAppDataSource for CapabilityDataSource {}
impl RightSurfaceAppDataSource for CapabilityDataSource {}

#[async_trait]
impl ModelProviderAppDataSource for CapabilityDataSource {
    async fn read_model_provider_capabilities(
        &self,
    ) -> Result<ModelProviderCapabilitiesReadResponse, RuntimeCoreError> {
        Ok(ModelProviderCapabilitiesReadResponse {
            namespace_tools: true,
            image_generation: false,
            web_search: false,
        })
    }
}

#[tokio::test]
async fn model_provider_capabilities_read_uses_the_v2_public_jsonrpc_contract() {
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_app_data_source(Arc::new(CapabilityDataSource));
    let server = AppServer::with_runtime(runtime);
    initialize_server(&server).await;

    let response = request(
        &server,
        2,
        METHOD_MODEL_PROVIDER_CAPABILITIES_READ,
        json!({}),
    )
    .await;

    assert_eq!(
        response.get("result"),
        Some(&json!({
            "namespaceTools": true,
            "imageGeneration": false,
            "webSearch": false
        }))
    );
}

async fn initialize_server(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "model-provider-capabilities-jsonrpc-test",
                "version": "1.0.0"
            }
        }),
    )
    .await;
    assert_eq!(
        response.pointer("/result/serverInfo/protocolVersion"),
        Some(&json!(PROTOCOL_VERSION))
    );
    let lines = server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "method": METHOD_INITIALIZED,
                "params": {}
            })
            .to_string(),
        )
        .await
        .expect("handle initialized notification");
    assert!(lines.is_empty());
}

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let lines = server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params
            })
            .to_string(),
        )
        .await
        .expect("handle JSON-RPC request");
    assert_eq!(lines.len(), 1, "{method} should return one response");
    let response: Value = serde_json::from_str(&lines[0]).expect("decode JSON-RPC response");
    assert_eq!(response.get("id"), Some(&json!(id)));
    if let Some(error) = response.get("error") {
        panic!("{method} failed: {error}");
    }
    response
}
