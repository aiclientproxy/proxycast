use crate::{credential_bridge::ConfiguredReplyProvider, AgentRuntimeState};
use agent_runtime::turn_executor::TurnProviderConfiguration;
use app_server_protocol::ProtocolKind;
use lime_core::database::DbConnection;
use model_provider::runtime_provider::{
    RuntimeProviderAuth, RuntimeProviderConfig, RuntimeProviderProtocol,
};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionProviderConfig {
    pub provider_name: String,
    pub provider_selector: Option<String>,
    pub model_name: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub api_version: Option<String>,
    pub credential_uuid: Option<String>,
    pub reasoning_effort: Option<String>,
    pub service_tier: Option<String>,
    pub route_protocol: Option<ProtocolKind>,
    pub toolshim: bool,
    pub toolshim_model: Option<String>,
    pub model_capabilities: Option<Value>,
    pub supports_websockets: bool,
}

struct ProviderConfigurationRequest<'a> {
    pub db: &'a DbConnection,
    pub session_id: &'a str,
    pub provider: &'a str,
    pub model: &'a str,
    pub reasoning_effort: Option<String>,
    pub service_tier: Option<String>,
    pub route_protocol: Option<ProtocolKind>,
    pub credential_ref: Option<&'a str>,
    pub direct_provider_config: Option<SessionProviderConfig>,
    pub auth: RuntimeProviderAuth,
    pub api_version: Option<String>,
}

pub(crate) struct ConfiguredSessionProvider {
    config: SessionProviderConfig,
    provider: ConfiguredReplyProvider,
}

impl ConfiguredSessionProvider {
    pub(crate) fn into_config(self) -> SessionProviderConfig {
        self.config
    }

    pub(crate) fn provider(&self) -> ConfiguredReplyProvider {
        self.provider.clone()
    }
}

#[derive(Debug, Clone)]
pub struct ModelRouteProviderConfiguration {
    pub turn_provider: TurnProviderConfiguration,
    pub service_tier: Option<String>,
    pub api_version: Option<String>,
    pub route_protocol: Option<ProtocolKind>,
    pub credential_ref: Option<String>,
    pub direct_provider_config: Option<SessionProviderConfig>,
    pub auth: RuntimeProviderAuth,
}

async fn configure_provider_for_session(
    agent_state: &AgentRuntimeState,
    request: ProviderConfigurationRequest<'_>,
) -> Result<ConfiguredSessionProvider, String> {
    let direct_route_protocol = request
        .direct_provider_config
        .as_ref()
        .and_then(|config| config.route_protocol.as_ref());
    ensure_supported_route_protocol(request.route_protocol.as_ref().or(direct_route_protocol))?;
    agent_state.init_agent_with_db(request.db).await?;

    if request.auth == RuntimeProviderAuth::OemManaged {
        return Err("OEM-managed authentication has no current model-provider adapter".to_string());
    }

    if let Some(mut config) = request.direct_provider_config {
        if let Some(credential_ref) = request.credential_ref {
            if config.credential_uuid.as_deref() != Some(credential_ref) {
                return Err(
                    "direct provider config does not match the resolved credential ref".to_string(),
                );
            }
        }
        config.route_protocol = request.route_protocol.or(config.route_protocol);
        config.service_tier = request.service_tier.or(config.service_tier);
        config.api_version = request.api_version.or(config.api_version);
        let runtime_config =
            session_provider_config_to_runtime_provider_config(&config, request.auth);
        let provider =
            install_provider_for_session(agent_state, request.session_id, &runtime_config).await?;
        return Ok(ConfiguredSessionProvider { config, provider });
    }

    match request.auth {
        RuntimeProviderAuth::NoAuth => {
            return Err("no-auth provider route is missing its resolved direct config".to_string());
        }
        RuntimeProviderAuth::OemManaged => unreachable!("rejected before provider selection"),
        RuntimeProviderAuth::ApiKey => {}
    }

    let mut runtime_config = agent_state
        .credential_bridge()
        .select_and_configure(
            request.db,
            request.provider,
            request.model,
            request.credential_ref,
        )
        .await
        .map_err(|error| format!("从 API Key Provider 选择凭证失败: {error}"))?;
    runtime_config.reasoning_effort = request.reasoning_effort;
    runtime_config.service_tier = request.service_tier;
    runtime_config.api_version = request.api_version.or(runtime_config.api_version);
    runtime_config.protocol = runtime_provider_protocol_from_route_protocol(
        &runtime_config.provider_name,
        request.route_protocol,
    );

    let provider =
        install_provider_for_session(agent_state, request.session_id, &runtime_config).await?;
    if let Err(error) = agent_state
        .credential_bridge()
        .record_usage(request.db, &runtime_config.credential_uuid)
    {
        tracing::warn!("[AgentRuntime] 记录凭证使用失败: {}", error);
    }

    let config = SessionProviderConfig {
        provider_name: runtime_config.provider_name,
        provider_selector: runtime_config.provider_selector,
        model_name: runtime_config.model_name,
        api_key: runtime_config.api_key,
        base_url: runtime_config.base_url,
        api_version: runtime_config.api_version,
        credential_uuid: Some(runtime_config.credential_uuid),
        reasoning_effort: runtime_config.reasoning_effort,
        service_tier: runtime_config.service_tier,
        route_protocol: route_protocol_from_runtime_provider_protocol(runtime_config.protocol),
        toolshim: runtime_config.toolshim,
        toolshim_model: runtime_config.toolshim_model,
        model_capabilities: None,
        supports_websockets: runtime_config.supports_websockets,
    };
    Ok(ConfiguredSessionProvider { config, provider })
}

async fn install_provider_for_session(
    agent_state: &AgentRuntimeState,
    session_id: &str,
    runtime_config: &RuntimeProviderConfig,
) -> Result<ConfiguredReplyProvider, String> {
    agent_state
        .install_provider_for_session(session_id, runtime_config)
        .await
        .map_err(|error| format!("创建 Provider 失败: {error}"))
}

pub(crate) async fn configure_model_route_provider_for_session_with_provider_and_credential_ref(
    agent_state: &AgentRuntimeState,
    db: &DbConnection,
    session_id: &str,
    configuration: ModelRouteProviderConfiguration,
    credential_ref: Option<&str>,
) -> Result<ConfiguredSessionProvider, String> {
    configure_provider_for_session(
        agent_state,
        ProviderConfigurationRequest {
            db,
            session_id,
            provider: configuration.turn_provider.route.provider.as_str(),
            model: configuration.turn_provider.route.model.as_str(),
            reasoning_effort: configuration.turn_provider.reasoning_effort,
            service_tier: configuration.service_tier,
            route_protocol: configuration.route_protocol,
            credential_ref,
            direct_provider_config: configuration.direct_provider_config,
            auth: configuration.auth,
            api_version: configuration.api_version,
        },
    )
    .await
}

pub fn route_protocol_from_session_provider_config(
    config: &SessionProviderConfig,
) -> Option<ProtocolKind> {
    config.route_protocol.clone()
}

pub fn supports_provider_type(provider_type: &str) -> bool {
    RuntimeProviderProtocol::from_provider_type(provider_type).is_some()
}

pub fn supports_direct_route(provider_name: &str, protocol: &ProtocolKind) -> bool {
    RuntimeProviderProtocol::from_direct_route(provider_name, protocol).is_some()
}

fn ensure_supported_route_protocol(protocol: Option<&ProtocolKind>) -> Result<(), String> {
    let Some(protocol) = protocol else {
        return Err(
            "provider route is missing an explicit protocol; resolve the model route before provider admission"
                .to_string(),
        );
    };
    if RuntimeProviderProtocol::from_route_protocol(protocol).is_some() {
        return Ok(());
    }
    Err(format!(
        "unsupported provider protocol {protocol:?}; no current model-provider wire adapter is registered"
    ))
}

fn runtime_provider_protocol_from_route_protocol(
    provider_name: &str,
    protocol: Option<ProtocolKind>,
) -> Option<RuntimeProviderProtocol> {
    protocol
        .as_ref()
        .and_then(|protocol| RuntimeProviderProtocol::from_direct_route(provider_name, protocol))
}

fn route_protocol_from_runtime_provider_protocol(
    protocol: Option<RuntimeProviderProtocol>,
) -> Option<ProtocolKind> {
    if protocol == Some(RuntimeProviderProtocol::VertexGemini) {
        return Some(ProtocolKind::VertexGemini);
    }
    route_protocol_from_model_provider_protocol(
        protocol.map(RuntimeProviderProtocol::to_model_provider_protocol),
    )
}

fn session_provider_config_to_runtime_provider_config(
    config: &SessionProviderConfig,
    auth: RuntimeProviderAuth,
) -> RuntimeProviderConfig {
    RuntimeProviderConfig {
        provider_name: config.provider_name.clone(),
        provider_selector: config.provider_selector.clone(),
        model_name: config.model_name.clone(),
        api_key: match auth {
            RuntimeProviderAuth::ApiKey => config.api_key.clone(),
            RuntimeProviderAuth::NoAuth | RuntimeProviderAuth::OemManaged => None,
        },
        auth,
        base_url: config.base_url.clone(),
        api_version: config.api_version.clone(),
        credential_uuid: config.credential_uuid.clone().unwrap_or_default(),
        reasoning_effort: config.reasoning_effort.clone(),
        service_tier: config.service_tier.clone(),
        protocol: runtime_provider_protocol_from_route_protocol(
            &config.provider_name,
            config.route_protocol.clone(),
        ),
        supports_websockets: config.supports_websockets,
        toolshim: config.toolshim,
        toolshim_model: config.toolshim_model.clone(),
    }
}

fn route_protocol_from_model_provider_protocol(
    protocol: Option<model_provider::ModelProviderProtocol>,
) -> Option<ProtocolKind> {
    match protocol {
        Some(model_provider::ModelProviderProtocol::Responses) => {
            Some(ProtocolKind::OpenaiResponses)
        }
        Some(model_provider::ModelProviderProtocol::ChatCompletions) => {
            Some(ProtocolKind::OpenaiChat)
        }
        Some(model_provider::ModelProviderProtocol::AnthropicMessages) => {
            Some(ProtocolKind::AnthropicMessages)
        }
        Some(model_provider::ModelProviderProtocol::GeminiGenerateContent) => {
            Some(ProtocolKind::GeminiGenerateContent)
        }
        Some(model_provider::ModelProviderProtocol::Custom(_)) | None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lime_core::database::dao::api_key_provider::ApiProviderType;
    use lime_core::database::schema;
    use lime_core::models::runtime_api_key_credential_uuid;
    use lime_services::api_key_provider_service::ApiKeyProviderService;
    use rusqlite::Connection;
    use std::sync::{Arc, Mutex};

    fn test_db() -> DbConnection {
        let connection =
            Connection::open_in_memory().expect("open provider configuration database");
        schema::create_tables(&connection).expect("create provider configuration schema");
        Arc::new(Mutex::new(connection))
    }

    fn model_route_configuration(
        provider: impl Into<String>,
        model: impl Into<String>,
        reasoning_effort: Option<String>,
        route_protocol: Option<ProtocolKind>,
    ) -> ModelRouteProviderConfiguration {
        ModelRouteProviderConfiguration {
            turn_provider: TurnProviderConfiguration::from_model_selection(
                provider,
                model,
                reasoning_effort,
            ),
            service_tier: None,
            api_version: None,
            route_protocol,
            credential_ref: None,
            direct_provider_config: None,
            auth: RuntimeProviderAuth::ApiKey,
        }
    }

    #[test]
    fn model_provider_protocol_is_projected_to_runtime_adapter_protocol() {
        assert_eq!(
            runtime_provider_protocol_from_route_protocol(
                "openai",
                Some(ProtocolKind::OpenaiResponses),
            ),
            Some(RuntimeProviderProtocol::Responses)
        );
        assert_eq!(
            runtime_provider_protocol_from_route_protocol(
                "codex",
                Some(ProtocolKind::CodexResponses),
            ),
            Some(RuntimeProviderProtocol::Responses)
        );
        assert_eq!(
            runtime_provider_protocol_from_route_protocol("openai", Some(ProtocolKind::OpenaiChat)),
            Some(RuntimeProviderProtocol::ChatCompletions)
        );
        assert_eq!(
            runtime_provider_protocol_from_route_protocol(
                "anthropic",
                Some(ProtocolKind::AnthropicMessages),
            ),
            Some(RuntimeProviderProtocol::AnthropicMessages)
        );
        assert_eq!(
            runtime_provider_protocol_from_route_protocol(
                "google",
                Some(ProtocolKind::GeminiGenerateContent),
            ),
            Some(RuntimeProviderProtocol::GeminiGenerateContent)
        );
        assert_eq!(
            runtime_provider_protocol_from_route_protocol(
                "azure",
                Some(ProtocolKind::OpenaiResponses),
            ),
            Some(RuntimeProviderProtocol::AzureResponses)
        );
        assert_eq!(
            runtime_provider_protocol_from_route_protocol(
                "gcpvertexai",
                Some(ProtocolKind::VertexGemini),
            ),
            Some(RuntimeProviderProtocol::VertexGemini)
        );
    }

    #[test]
    fn non_openai_route_protocols_do_not_invent_runtime_adapter_protocol() {
        for protocol in [
            ProtocolKind::OpenaiImages,
            ProtocolKind::Fal,
            ProtocolKind::XaiVideo,
            ProtocolKind::BedrockConverse,
            ProtocolKind::Unknown,
        ] {
            assert_eq!(
                runtime_provider_protocol_from_route_protocol("openai", Some(protocol.clone())),
                None,
                "{protocol:?} must stay route metadata until a matching runtime adapter exists"
            );
        }
    }

    #[test]
    fn unsupported_route_protocols_fail_closed_before_provider_selection() {
        for protocol in [
            ProtocolKind::OpenaiImages,
            ProtocolKind::Fal,
            ProtocolKind::XaiVideo,
            ProtocolKind::BedrockConverse,
            ProtocolKind::Unknown,
        ] {
            let error = ensure_supported_route_protocol(Some(&protocol))
                .expect_err("unsupported protocol must fail closed");
            assert!(error.contains("unsupported provider protocol"));
            assert!(error.contains("no current model-provider wire adapter"));
        }
    }

    #[test]
    fn missing_route_protocol_fails_closed_before_provider_selection() {
        let error = ensure_supported_route_protocol(None)
            .expect_err("missing route protocol must fail before provider selection");

        assert!(error.contains("missing an explicit protocol"));
        assert!(error.contains("resolve the model route"));
    }

    #[test]
    fn direct_provider_config_preserves_non_openai_route_metadata_only() {
        let config = SessionProviderConfig {
            provider_name: "anthropic".to_string(),
            provider_selector: Some("anthropic".to_string()),
            model_name: "claude-sonnet-4-5".to_string(),
            api_key: Some("sk-test".to_string()),
            base_url: Some("https://api.anthropic.com".to_string()),
            api_version: None,
            credential_uuid: Some("credential-anthropic".to_string()),
            reasoning_effort: Some("medium".to_string()),
            service_tier: None,
            route_protocol: Some(ProtocolKind::AnthropicMessages),
            toolshim: false,
            toolshim_model: None,
            model_capabilities: None,
            supports_websockets: false,
        };

        let runtime_config = session_provider_config_to_runtime_provider_config(
            &config,
            RuntimeProviderAuth::ApiKey,
        );

        assert_eq!(runtime_config.provider_name, "anthropic");
        assert_eq!(runtime_config.model_name, "claude-sonnet-4-5");
        assert_eq!(
            runtime_config.protocol,
            Some(RuntimeProviderProtocol::AnthropicMessages)
        );
        assert_eq!(
            route_protocol_from_session_provider_config(&config),
            Some(ProtocolKind::AnthropicMessages)
        );
    }

    #[test]
    fn azure_session_config_projects_to_dedicated_runtime_adapter() {
        let config = SessionProviderConfig {
            provider_name: "azure".to_string(),
            provider_selector: Some("azure-openai".to_string()),
            model_name: "gpt-5.4".to_string(),
            api_key: Some("azure-test-key".to_string()),
            base_url: Some("https://resource.openai.azure.com".to_string()),
            api_version: Some("2025-04-01-preview".to_string()),
            credential_uuid: Some("runtime-api-key:azure-key".to_string()),
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            route_protocol: Some(ProtocolKind::OpenaiResponses),
            toolshim: false,
            toolshim_model: None,
            model_capabilities: None,
            supports_websockets: true,
        };

        let runtime_config = session_provider_config_to_runtime_provider_config(
            &config,
            RuntimeProviderAuth::ApiKey,
        );

        assert_eq!(
            runtime_config.protocol,
            Some(RuntimeProviderProtocol::AzureResponses)
        );
        assert_eq!(runtime_config.api_version, config.api_version);
        assert_eq!(runtime_config.base_url, config.base_url);
        assert_eq!(runtime_config.auth, RuntimeProviderAuth::ApiKey);
    }

    #[test]
    fn provider_config_protocol_projects_to_route_protocol() {
        let mut config = SessionProviderConfig {
            provider_name: "openai".to_string(),
            provider_selector: Some("openai".to_string()),
            model_name: "gpt-4.1".to_string(),
            api_key: Some("must-be-discarded".to_string()),
            base_url: None,
            api_version: None,
            credential_uuid: None,
            reasoning_effort: None,
            service_tier: None,
            route_protocol: Some(ProtocolKind::OpenaiResponses),
            toolshim: false,
            toolshim_model: None,
            model_capabilities: None,
            supports_websockets: true,
        };
        assert_eq!(
            route_protocol_from_session_provider_config(&config),
            Some(ProtocolKind::OpenaiResponses)
        );
        assert!(config.supports_websockets);
        assert!(
            session_provider_config_to_runtime_provider_config(
                &config,
                RuntimeProviderAuth::ApiKey,
            )
            .supports_websockets
        );

        config.route_protocol = Some(ProtocolKind::OpenaiChat);
        assert_eq!(
            route_protocol_from_session_provider_config(&config),
            Some(ProtocolKind::OpenaiChat)
        );
    }

    #[test]
    fn no_auth_direct_config_projects_without_synthetic_credential() {
        let config = SessionProviderConfig {
            provider_name: "ollama".to_string(),
            provider_selector: Some("ollama".to_string()),
            model_name: "qwen3:14b".to_string(),
            api_key: None,
            base_url: Some("http://127.0.0.1:11434".to_string()),
            api_version: None,
            credential_uuid: None,
            reasoning_effort: None,
            service_tier: None,
            route_protocol: Some(ProtocolKind::OpenaiResponses),
            toolshim: false,
            toolshim_model: None,
            model_capabilities: None,
            supports_websockets: false,
        };

        let runtime_config = session_provider_config_to_runtime_provider_config(
            &config,
            RuntimeProviderAuth::NoAuth,
        );

        assert_eq!(runtime_config.auth, RuntimeProviderAuth::NoAuth);
        assert_eq!(runtime_config.provider_name, "ollama");
        assert_eq!(
            runtime_config.protocol,
            Some(RuntimeProviderProtocol::Responses)
        );
        assert!(runtime_config.api_key.is_none());
        assert!(runtime_config.credential_uuid.is_empty());
    }

    #[tokio::test]
    async fn model_route_configuration_honors_exact_credential_ref() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Configured Exact Provider".to_string(),
                ApiProviderType::Openai,
                "https://configured.example/v1".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create configured provider");
        service
            .add_api_key(&db, &provider.id, "sk-configured-first-key", None, false)
            .expect("add first configured key");
        let exact_key = service
            .add_api_key(&db, &provider.id, "sk-configured-exact-key", None, false)
            .expect("add exact configured key");
        let credential_ref = runtime_api_key_credential_uuid(&exact_key.id);
        let runtime = AgentRuntimeState::new();
        let configuration = model_route_configuration(
            provider.id,
            "fixture-model",
            Some("high".to_string()),
            Some(ProtocolKind::OpenaiChat),
        );

        let configured =
            configure_model_route_provider_for_session_with_provider_and_credential_ref(
                &runtime,
                &db,
                "exact-session",
                configuration,
                Some(&credential_ref),
            )
            .await
            .expect("configure exact model route")
            .into_config();

        assert_eq!(
            configured.credential_uuid.as_deref(),
            Some(credential_ref.as_str())
        );
        assert_eq!(
            configured.api_key.as_deref(),
            Some("sk-configured-exact-key")
        );
        assert_eq!(configured.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(configured.route_protocol, Some(ProtocolKind::OpenaiChat));
    }

    #[tokio::test]
    async fn model_route_configuration_does_not_fallback_from_missing_exact_ref() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Configured No Fallback Provider".to_string(),
                ApiProviderType::Openai,
                "https://configured.example/v1".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create configured provider");
        service
            .add_api_key(
                &db,
                &provider.id,
                "sk-configured-available-key",
                None,
                false,
            )
            .expect("add configured key");
        let missing_ref = runtime_api_key_credential_uuid("missing-key");
        let runtime = AgentRuntimeState::new();
        let configuration = model_route_configuration(
            provider.id,
            "fixture-model",
            None,
            Some(ProtocolKind::OpenaiChat),
        );

        let error =
            match configure_model_route_provider_for_session_with_provider_and_credential_ref(
                &runtime,
                &db,
                "missing-exact-session",
                configuration,
                Some(&missing_ref),
            )
            .await
            {
                Ok(_) => panic!("missing exact ref must not fall back"),
                Err(error) => error,
            };
        assert!(error.contains("指定的 durable"));
    }
}
