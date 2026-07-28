use crate::credential_bridge::{
    create_configured_reply_provider, ConfiguredReplyProvider, CredentialBridge,
};
use crate::protocol::AgentActionRequiredScope;
mod mcp_runtime;
#[cfg(test)]
mod mcp_runtime_tests;
use agent_runtime::action_required::{
    ActionRequiredError, ActionTerminalStatus, PendingActionDescriptor, PendingActionRestoreOutcome,
};
use lime_core::database::DbConnection;
use lime_mcp::{ElicitationRequestRouter, McpRuntimeServerSpec};
pub(crate) use mcp_runtime::McpThreadRuntime;
use model_provider::current_client::{
    CurrentProviderError, CurrentProviderHealthRegistry, CurrentProviderHealthSnapshot,
};
use model_provider::runtime_provider::RuntimeProviderConfig;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use tool_runtime::gateway_dispatch_execution::RuntimeGatewayToolExecutionRegistry;
use tool_runtime::tool_definition::RuntimeToolDefinition;

pub struct AgentRuntimeState {
    initialized: Arc<AtomicBool>,
    cancel_tokens: Arc<RwLock<HashMap<String, CancellationToken>>>,
    credential_bridge: CredentialBridge,
    provider_health: CurrentProviderHealthRegistry,
    providers: Arc<RwLock<HashMap<String, ConfiguredReplyProvider>>>,
    native_tool_definitions: Arc<RwLock<HashMap<String, RuntimeToolDefinition>>>,
    gateway_tools: RuntimeGatewayToolExecutionRegistry,
    mcp_runtimes: Arc<RwLock<HashMap<String, Arc<McpThreadRuntime>>>>,
    mcp_runtime_lifecycle: Arc<Mutex<()>>,
    action_required: Arc<agent_runtime::action_required::ActionRequiredState>,
    live_execution_gateway:
        Arc<RwLock<Option<Arc<dyn crate::live_execution_process::LiveExecutionProcessGateway>>>>,
}

impl Clone for AgentRuntimeState {
    fn clone(&self) -> Self {
        Self {
            initialized: Arc::clone(&self.initialized),
            cancel_tokens: Arc::clone(&self.cancel_tokens),
            credential_bridge: CredentialBridge::new(),
            provider_health: self.provider_health.clone(),
            providers: Arc::clone(&self.providers),
            native_tool_definitions: Arc::clone(&self.native_tool_definitions),
            gateway_tools: self.gateway_tools.clone(),
            mcp_runtimes: Arc::clone(&self.mcp_runtimes),
            mcp_runtime_lifecycle: Arc::clone(&self.mcp_runtime_lifecycle),
            action_required: Arc::clone(&self.action_required),
            live_execution_gateway: Arc::clone(&self.live_execution_gateway),
        }
    }
}

impl Default for AgentRuntimeState {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentRuntimeState {
    pub fn new() -> Self {
        Self {
            initialized: Arc::new(AtomicBool::new(false)),
            cancel_tokens: Arc::new(RwLock::new(HashMap::new())),
            credential_bridge: CredentialBridge::new(),
            provider_health: CurrentProviderHealthRegistry::new(),
            providers: Arc::new(RwLock::new(HashMap::new())),
            native_tool_definitions: Arc::new(RwLock::new(HashMap::new())),
            gateway_tools: RuntimeGatewayToolExecutionRegistry::default(),
            mcp_runtimes: Arc::new(RwLock::new(HashMap::new())),
            mcp_runtime_lifecycle: Arc::new(Mutex::new(())),
            action_required: Arc::new(
                agent_runtime::action_required::ActionRequiredState::default(),
            ),
            live_execution_gateway: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn init_agent_with_db(&self, _db: &DbConnection) -> Result<(), String> {
        if !self.initialized.swap(true, Ordering::AcqRel) {
            let definitions = crate::native_tools::current_native_tool_definitions();
            self.reset_native_tool_definitions(definitions).await;
        }
        Ok(())
    }

    pub(crate) fn credential_bridge(&self) -> &CredentialBridge {
        &self.credential_bridge
    }

    pub(crate) async fn install_provider_for_session(
        &self,
        session_id: &str,
        config: &RuntimeProviderConfig,
    ) -> Result<ConfiguredReplyProvider, CurrentProviderError> {
        let mut providers = self.providers.write().await;
        if let Some(provider) = providers.get(session_id) {
            if provider.client().config() == config {
                return Ok(provider.clone());
            }
        }
        let provider = create_configured_reply_provider(config, &self.provider_health)?;
        providers.insert(session_id.to_string(), provider.clone());
        Ok(provider)
    }

    pub(crate) async fn provider_for_session(
        &self,
        session_id: &str,
    ) -> Option<ConfiguredReplyProvider> {
        self.providers.read().await.get(session_id).cloned()
    }

    pub fn provider_health_snapshot(
        &self,
        config: &RuntimeProviderConfig,
    ) -> Option<CurrentProviderHealthSnapshot> {
        self.provider_health.snapshot_for(config)
    }

    pub async fn close_provider_session(&self, session_id: &str) {
        self.providers.write().await.remove(session_id);
    }

    pub(crate) fn gateway_tools(&self) -> &RuntimeGatewayToolExecutionRegistry {
        &self.gateway_tools
    }

    pub(crate) async fn mcp_runtime(
        &self,
        session_id: &str,
        thread_id: &str,
    ) -> Result<Arc<McpThreadRuntime>, String> {
        let runtimes = self.mcp_runtimes.read().await;
        let runtime = runtimes
            .get(session_id)
            .cloned()
            .ok_or_else(|| format!("MCP runtime is not initialized for session '{session_id}'"))?;
        if runtime.thread_id() != thread_id {
            return Err(format!(
                "MCP runtime thread mismatch for session '{session_id}': expected '{}', got '{thread_id}'",
                runtime.thread_id()
            ));
        }
        Ok(runtime)
    }

    pub(crate) fn action_required_state(
        &self,
    ) -> Arc<agent_runtime::action_required::ActionRequiredState> {
        Arc::clone(&self.action_required)
    }

    pub async fn pending_action_descriptors(&self) -> Vec<PendingActionDescriptor> {
        self.action_required.pending_action_descriptors().await
    }

    pub async fn restore_pending_action_descriptors(
        &self,
        descriptors: impl IntoIterator<Item = PendingActionDescriptor>,
    ) -> Vec<PendingActionRestoreOutcome> {
        self.action_required
            .restore_pending_actions(descriptors)
            .await
    }

    pub async fn contains_pending_action(&self, request_id: &str) -> bool {
        self.action_required.contains_action(request_id).await
    }

    pub async fn terminal_action_status(&self, request_id: &str) -> Option<ActionTerminalStatus> {
        self.action_required.terminal_status(request_id).await
    }

    async fn reset_native_tool_definitions(&self, definitions: Vec<RuntimeToolDefinition>) {
        let mut current = self.native_tool_definitions.write().await;
        current.clear();
        current.extend(
            definitions
                .into_iter()
                .map(|definition| (definition.name.clone(), definition)),
        );
    }

    pub async fn contains_native_tool(&self, tool_name: &str) -> bool {
        self.native_tool_definitions
            .read()
            .await
            .contains_key(tool_name)
    }

    pub(crate) async fn native_tool_definitions_snapshot(&self) -> Vec<RuntimeToolDefinition> {
        let mut definitions = self
            .native_tool_definitions
            .read()
            .await
            .values()
            .cloned()
            .collect::<Vec<_>>();
        definitions.sort_by(|left, right| left.name.cmp(&right.name));
        definitions
    }

    pub(crate) async fn register_native_tool(
        &self,
        registration: crate::native_tools::NativeRegistration,
    ) -> Result<(), String> {
        let definition = registration.definition();
        self.gateway_tools
            .register(registration.into_gateway_execution());
        self.native_tool_definitions
            .write()
            .await
            .insert(definition.name.clone(), definition);
        Ok(())
    }

    pub async fn register_memory_store_tools(
        &self,
        gateway: Arc<dyn tool_runtime::memory_store::MemoryStoreGateway>,
    ) -> Result<(), String> {
        for registration in crate::native_tools::create_memory_tools(gateway) {
            self.register_native_tool(registration).await?;
        }
        Ok(())
    }

    pub async fn register_image_task_tools(
        &self,
        gateway: Arc<dyn tool_runtime::image_task::ImageTaskGateway>,
    ) -> Result<(), String> {
        for registration in crate::native_tools::create_image_tools(gateway) {
            self.register_native_tool(registration).await?;
        }
        Ok(())
    }

    pub async fn register_tool_search_tools(
        &self,
        gateway: Arc<dyn tool_runtime::tool_search::ToolSearchGateway>,
    ) -> Result<(), String> {
        for registration in crate::native_tools::create_tool_search_tools(gateway) {
            self.register_native_tool(registration).await?;
        }
        Ok(())
    }

    pub async fn register_mcp_resource_tools(
        &self,
        gateway: Arc<dyn tool_runtime::mcp_resource::McpResourceGateway>,
    ) -> Result<(), String> {
        for registration in crate::native_tools::create_mcp_resource_tools(gateway) {
            self.register_native_tool(registration).await?;
        }
        Ok(())
    }

    pub async fn install_live_execution_process_gateway(
        &self,
        gateway: Arc<dyn crate::live_execution_process::LiveExecutionProcessGateway>,
    ) -> Result<(), String> {
        *self.live_execution_gateway.write().await = Some(gateway);
        Ok(())
    }

    pub(crate) async fn live_execution_process_gateway(
        &self,
    ) -> Option<Arc<dyn crate::live_execution_process::LiveExecutionProcessGateway>> {
        self.live_execution_gateway.read().await.clone()
    }

    pub async fn create_cancel_token(&self, session_id: &str) -> CancellationToken {
        let token = CancellationToken::new();
        self.cancel_tokens
            .write()
            .await
            .insert(session_id.to_string(), token.clone());
        token
    }

    pub async fn cancel_session(&self, session_id: &str) -> bool {
        let canceled = if let Some(token) = self.cancel_tokens.read().await.get(session_id) {
            token.cancel();
            true
        } else {
            false
        };
        if let Some(scope) =
            AgentActionRequiredScope::from_parts(Some(session_id.to_string()), None, None)
        {
            self.action_required.cancel_for_scope(&scope).await;
        }
        canceled
    }

    pub async fn remove_cancel_token(&self, session_id: &str) {
        self.cancel_tokens.write().await.remove(session_id);
    }

    pub async fn submit_elicitation_response(
        &self,
        session_id: &str,
        request_id: &str,
        _user_data: serde_json::Value,
        action_scope: Option<AgentActionRequiredScope>,
    ) -> Result<(), ActionRequiredError> {
        let scope = require_action_scope(session_id, request_id, action_scope)?;
        self.action_required
            .resolve_action(request_id, Some(&scope))
            .await
    }

    pub async fn ensure_action_resumable(
        &self,
        session_id: &str,
        request_id: &str,
        action_scope: Option<AgentActionRequiredScope>,
    ) -> Result<(), ActionRequiredError> {
        let scope = require_action_scope(session_id, request_id, action_scope)?;
        self.action_required
            .ensure_resumable(request_id, Some(&scope))
            .await
    }

    pub async fn cancel_action(
        &self,
        session_id: &str,
        request_id: &str,
        action_scope: Option<AgentActionRequiredScope>,
    ) -> Result<(), ActionRequiredError> {
        let scope = require_action_scope(session_id, request_id, action_scope)?;
        self.action_required
            .cancel_action(request_id, Some(&scope))
            .await
    }

    pub async fn confirm_tool_action(
        &self,
        session_id: &str,
        request_id: &str,
        _confirmed: bool,
        action_scope: Option<AgentActionRequiredScope>,
    ) -> Result<(), ActionRequiredError> {
        let scope = require_action_scope(session_id, request_id, action_scope)?;
        self.action_required
            .resolve_action(request_id, Some(&scope))
            .await
    }

    pub async fn ensure_mcp_runtime(
        &self,
        session_id: String,
        thread_id: String,
        elicitation_router: ElicitationRequestRouter,
        server_specs: Vec<McpRuntimeServerSpec>,
    ) -> Result<(), String> {
        self.ensure_mcp_runtime_generation(session_id, thread_id, elicitation_router, server_specs)
            .await
            .map(|_| ())
    }

    async fn ensure_mcp_runtime_generation(
        &self,
        session_id: String,
        thread_id: String,
        elicitation_router: ElicitationRequestRouter,
        server_specs: Vec<McpRuntimeServerSpec>,
    ) -> Result<Arc<McpThreadRuntime>, String> {
        if session_id.trim().is_empty() || thread_id.trim().is_empty() {
            return Err("MCP runtime requires canonical session and thread identity".to_string());
        }
        let _lifecycle = self.mcp_runtime_lifecycle.lock().await;
        if let Some(runtime) = self.mcp_runtimes.read().await.get(&session_id).cloned() {
            if runtime.thread_id() != thread_id {
                return Err(format!(
                    "MCP runtime thread mismatch for session '{session_id}'"
                ));
            }
            if runtime.server_specs() == server_specs {
                return Ok(runtime);
            }
        }

        let runtime = Arc::new(McpThreadRuntime::new(
            session_id.clone(),
            thread_id.clone(),
            elicitation_router,
            server_specs.clone(),
        ));
        runtime.start().await?;
        let mut runtimes = self.mcp_runtimes.write().await;
        if let Some(existing) = runtimes.get(&session_id).cloned() {
            if existing.thread_id() != runtime.thread_id() {
                return Err(format!(
                    "MCP runtime thread mismatch for session '{session_id}'"
                ));
            }
        }
        runtimes.insert(session_id, Arc::clone(&runtime));
        Ok(runtime)
    }

    pub async fn close_mcp_runtime(&self, session_id: &str, thread_id: &str) {
        let _lifecycle = self.mcp_runtime_lifecycle.lock().await;
        let runtime = self.mcp_runtimes.write().await.remove(session_id);
        let Some(runtime) = runtime else {
            return;
        };
        if runtime.thread_id() != thread_id {
            self.mcp_runtimes
                .write()
                .await
                .insert(session_id.to_string(), runtime);
            tracing::warn!(
                session_id,
                thread_id,
                "refusing to close MCP runtime with mismatched thread"
            );
            return;
        }
        runtime.shutdown().await;
    }

    #[cfg(test)]
    pub(crate) async fn mcp_runtime_count(&self) -> usize {
        self.mcp_runtimes.read().await.len()
    }

    pub async fn clear_mcp_runtimes(&self) {
        let _lifecycle = self.mcp_runtime_lifecycle.lock().await;
        let runtimes = std::mem::take(&mut *self.mcp_runtimes.write().await);
        for runtime in runtimes.into_values() {
            runtime.shutdown().await;
        }
    }

    pub async fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }
}

fn require_action_scope(
    session_id: &str,
    request_id: &str,
    scope: Option<AgentActionRequiredScope>,
) -> Result<AgentActionRequiredScope, ActionRequiredError> {
    let scope = scope.ok_or_else(|| ActionRequiredError::ScopeMissing(request_id.to_string()))?;
    let complete = [
        scope.session_id.as_deref(),
        scope.thread_id.as_deref(),
        scope.turn_id.as_deref(),
    ]
    .into_iter()
    .all(|field| field.is_some_and(|value| !value.trim().is_empty()));
    if !complete || scope.session_id.as_deref() != Some(session_id) {
        return Err(ActionRequiredError::ScopeMismatch(request_id.to_string()));
    }
    Ok(scope)
}

#[cfg(test)]
mod provider_session_tests {
    use super::AgentRuntimeState;
    use model_provider::current_client::{
        CurrentProviderContent, CurrentProviderMessage, CurrentProviderRequest,
    };
    use model_provider::runtime_provider::{RuntimeProviderConfig, RuntimeProviderProtocol};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn provider_config(model_name: &str) -> RuntimeProviderConfig {
        RuntimeProviderConfig {
            provider_name: "openai".to_string(),
            provider_selector: Some("openai".to_string()),
            model_name: model_name.to_string(),
            api_key: Some("test-key".to_string()),
            auth: model_provider::runtime_provider::RuntimeProviderAuth::ApiKey,
            base_url: Some("https://api.openai.com/v1".to_string()),
            api_version: None,
            credential_uuid: "credential-1".to_string(),
            reasoning_effort: None,
            service_tier: None,
            protocol: Some(RuntimeProviderProtocol::Responses),
            supports_websockets: true,
            toolshim: false,
            toolshim_model: None,
        }
    }

    #[tokio::test]
    async fn provider_clients_are_reused_only_within_matching_session_route() {
        let state = AgentRuntimeState::new();
        let config = provider_config("gpt-5.4");

        let first = state
            .install_provider_for_session("session-a", &config)
            .await
            .expect("first provider");
        let second = state
            .install_provider_for_session("session-a", &config)
            .await
            .expect("reused provider");
        let other_session = state
            .install_provider_for_session("session-b", &config)
            .await
            .expect("other session provider");

        assert!(Arc::ptr_eq(&first.client(), &second.client()));
        assert!(!Arc::ptr_eq(&first.client(), &other_session.client()));

        state.close_provider_session("session-a").await;
        assert!(state.provider_for_session("session-a").await.is_none());
        assert!(state.provider_for_session("session-b").await.is_some());
    }

    #[tokio::test]
    async fn provider_health_snapshot_uses_the_cloned_runtime_registry() {
        let state = AgentRuntimeState::new();
        let observer = state.clone();
        let route = provider_config("gpt-5.4");
        let mut different_route = route.clone();
        different_route.model_name = "gpt-5.5".to_string();

        assert!(observer.provider_health_snapshot(&route).is_none());
        state
            .install_provider_for_session("session-a", &route)
            .await
            .expect("install provider");

        let snapshot = observer
            .provider_health_snapshot(&route)
            .expect("shared route health snapshot");
        assert_eq!(
            snapshot.state,
            model_provider::current_client::CurrentProviderHealthState::Closed
        );
        assert_eq!(snapshot.window_sample_count, Some(0));
        assert_eq!(snapshot.window_failure_count, Some(0));
        assert!(observer
            .provider_health_snapshot(&different_route)
            .is_none());
    }

    #[tokio::test]
    async fn provider_route_change_replaces_only_that_session_client() {
        let state = AgentRuntimeState::new();
        let first = state
            .install_provider_for_session("session-a", &provider_config("gpt-5.4"))
            .await
            .expect("first provider");
        let replacement = state
            .install_provider_for_session("session-a", &provider_config("gpt-5.5"))
            .await
            .expect("replacement provider");

        assert!(!Arc::ptr_eq(&first.client(), &replacement.client()));
        assert_eq!(replacement.client().config().model_name, "gpt-5.5");
    }

    #[tokio::test]
    async fn route_health_survives_session_route_switch_without_cross_route_blocking() {
        let (base_url, requests, server) = spawn_health_fixture().await;
        let state = AgentRuntimeState::new();
        let mut route = provider_config("gpt-5.4");
        route.base_url = Some(base_url);
        route.protocol = Some(RuntimeProviderProtocol::ChatCompletions);
        route.supports_websockets = false;

        let first = state
            .install_provider_for_session("session-a", &route)
            .await
            .expect("first provider");
        for _ in 0..10 {
            let error = match first.client().stream(text_request()).await {
                Err(error) => error,
                Ok(_) => panic!("transient route failure should be surfaced"),
            };
            assert!(error.message.contains("429"));
        }
        let opened_snapshot = state
            .provider_health_snapshot(&route)
            .expect("opened route snapshot");
        assert_eq!(
            opened_snapshot.state,
            model_provider::current_client::CurrentProviderHealthState::Open
        );
        assert!(opened_snapshot.retry_after.is_some());

        let mut alternate_route = route.clone();
        alternate_route.model_name = "gpt-5.5".to_string();
        let alternate = state
            .install_provider_for_session("session-a", &alternate_route)
            .await
            .expect("alternate provider");
        let alternate_stream = alternate
            .client()
            .stream(text_request())
            .await
            .expect("different model route must not inherit open circuit");
        drop(alternate_stream);

        let restored = state
            .install_provider_for_session("session-a", &route)
            .await
            .expect("restored provider");
        let error = match restored.client().stream(text_request()).await {
            Err(error) => error,
            Ok(_) => panic!("restored route must reuse its open health entry"),
        };
        assert!(error.message.contains("health circuit is open"));

        server.await.expect("health fixture");
        assert_eq!(requests.load(Ordering::SeqCst), 11);
    }

    fn text_request() -> CurrentProviderRequest {
        CurrentProviderRequest::new(vec![CurrentProviderMessage::user(vec![
            CurrentProviderContent::Text("health probe".to_string()),
        ])])
    }

    async fn spawn_health_fixture() -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind health fixture");
        let address = listener.local_addr().expect("health fixture address");
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = Arc::clone(&requests);
        let server = tokio::spawn(async move {
            for index in 0..11 {
                let (mut stream, _) = listener.accept().await.expect("accept health request");
                request_count.fetch_add(1, Ordering::SeqCst);
                let mut headers = [0_u8; 4096];
                let _ = stream.read(&mut headers).await;
                let response = if index < 10 {
                    "HTTP/1.1 429 Too Many Requests\r\nContent-Length: 4\r\n\r\nrate"
                } else {
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 0\r\n\r\n"
                };
                stream
                    .write_all(response.as_bytes())
                    .await
                    .expect("write health response");
                stream.shutdown().await.expect("close health response");
            }
        });
        (format!("http://{address}"), requests, server)
    }
}
