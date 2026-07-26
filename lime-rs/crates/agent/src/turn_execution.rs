use crate::current_provider_turn::stream_current_provider_turn;
use crate::protocol::AgentEvent as RuntimeAgentEvent;
use crate::provider_configuration::{
    configure_model_route_provider_for_session_with_provider_and_credential_ref,
    ModelRouteProviderConfiguration, SessionProviderConfig,
};
use crate::request_tool_policy::{ReplyAttemptError, RequestToolPolicy, StreamReplyExecution};
use crate::runtime_state::AgentRuntimeState;
use crate::AgentSessionConfig;
use agent_runtime::reply_input::RuntimeReplyInput;
use agent_runtime::session_loop::RuntimeSessionInputHandle;
use lime_core::database::DbConnection;
use model_provider::current_client::CurrentProviderMessage;
use model_provider::ModelProviderProtocol;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;
use tool_runtime::agent_control::AgentControlGatewayHandle;

const APP_SERVER_RUNTIME_BACKEND_METADATA_KEY: &str = "app_server_runtime_backend";

pub struct AgentTurnExecutionRequest<'a> {
    pub session_id: &'a str,
    pub input: RuntimeReplyInput,
    pub initial_messages: Vec<CurrentProviderMessage>,
    pub session_config: AgentSessionConfig,
    pub request_tool_policy: &'a RequestToolPolicy,
    pub provider_configuration: Option<AgentTurnProviderConfiguration<'a>>,
    pub agent_control_gateway: Option<AgentControlGatewayHandle>,
    pub pending_input: Option<RuntimeSessionInputHandle>,
    pub cancellation_token: Option<CancellationToken>,
}

pub struct AgentTurnProviderConfiguration<'a> {
    pub db: &'a DbConnection,
    pub session_id: &'a str,
    pub route_configuration: ModelRouteProviderConfiguration,
    pub credential_ref: Option<&'a str>,
}

pub struct AgentTurnExecution {
    pub stream: StreamReplyExecution,
    pub provider_config: Option<SessionProviderConfig>,
}

pub async fn run_agent_turn_with_policy<F>(
    agent_state: &AgentRuntimeState,
    request: AgentTurnExecutionRequest<'_>,
    on_event: F,
) -> Result<AgentTurnExecution, ReplyAttemptError>
where
    F: FnMut(&RuntimeAgentEvent) + Send,
{
    let route_thinking_default = request
        .provider_configuration
        .as_ref()
        .and_then(|configuration| route_thinking_default(&configuration.route_configuration));
    let configured_provider = if let Some(provider_configuration) = request.provider_configuration {
        Some(
            configure_model_route_provider_for_session_with_provider_and_credential_ref(
                agent_state,
                provider_configuration.db,
                provider_configuration.session_id,
                provider_configuration.route_configuration,
                provider_configuration.credential_ref,
            )
            .await
            .map_err(|message| ReplyAttemptError::new(message, false))?,
        )
    } else {
        None
    };
    let owns_cancel_token = request.cancellation_token.is_none();
    let cancel_token = match request.cancellation_token {
        Some(token) => token,
        None => agent_state.create_cancel_token(request.session_id).await,
    };
    let working_directory = request
        .session_config
        .turn_context
        .as_ref()
        .and_then(|context| context.cwd.clone());
    let mut session_config = request.session_config;
    apply_route_thinking_default(&mut session_config, route_thinking_default);
    let provider = match configured_provider.as_ref() {
        Some(configured_provider) => Ok(configured_provider.provider()),
        None => agent_state
            .provider_for_session(request.session_id)
            .await
            .ok_or_else(|| ReplyAttemptError::new("Provider is not configured", false)),
    };
    let provider = match provider {
        Ok(provider) => provider,
        Err(error) => {
            if owns_cancel_token {
                agent_state.remove_cancel_token(request.session_id).await;
            }
            return Err(error);
        }
    };
    let execution = stream_current_provider_turn(
        agent_state,
        provider,
        request.input,
        request.initial_messages,
        working_directory.as_deref(),
        session_config,
        Some(cancel_token),
        request.pending_input,
        request.request_tool_policy,
        request.agent_control_gateway,
        on_event,
    )
    .await;
    if owns_cancel_token {
        agent_state.remove_cancel_token(request.session_id).await;
    }
    let stream = execution?;
    Ok(AgentTurnExecution {
        stream,
        provider_config: configured_provider.map(|configured| configured.into_config()),
    })
}

fn route_thinking_default(configuration: &ModelRouteProviderConfiguration) -> Option<bool> {
    let route = &configuration.turn_provider.route;
    if route.protocol != ModelProviderProtocol::ChatCompletions {
        return None;
    }
    (!route
        .capabilities
        .iter()
        .any(|capability| capability.eq_ignore_ascii_case("reasoning")))
    .then_some(false)
}

fn apply_route_thinking_default(
    session_config: &mut AgentSessionConfig,
    route_default: Option<bool>,
) {
    let Some(route_default) = route_default else {
        return;
    };
    let Some(context) = session_config.turn_context.as_mut() else {
        return;
    };
    let metadata = context
        .metadata
        .entry(APP_SERVER_RUNTIME_BACKEND_METADATA_KEY.to_string())
        .or_insert_with(|| json!({}));
    if !metadata.is_object() {
        *metadata = json!({});
    }
    let object = metadata.as_object_mut().expect("runtime backend metadata");
    if object
        .get("thinkingEnabled")
        .or_else(|| object.get("thinking_enabled"))
        .and_then(Value::as_bool)
        .is_none()
    {
        object.insert("thinkingEnabled".to_string(), json!(route_default));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_protocol::ModelId;
    use agent_runtime::turn_executor::TurnProviderConfiguration;
    use model_provider::ModelRoute;

    fn route_configuration(
        protocol: ModelProviderProtocol,
        capabilities: Vec<&str>,
    ) -> ModelRouteProviderConfiguration {
        ModelRouteProviderConfiguration {
            turn_provider: TurnProviderConfiguration {
                route: ModelRoute {
                    provider: "provider-1".to_string(),
                    model: ModelId::new("model-1"),
                    protocol,
                    capabilities: capabilities.into_iter().map(str::to_string).collect(),
                    metadata: json!({}),
                },
                reasoning_effort: None,
            },
            service_tier: None,
            route_protocol: None,
            credential_ref: None,
            direct_provider_config: None,
        }
    }

    fn session_config(thinking_enabled: Option<bool>) -> AgentSessionConfig {
        let mut context = agent_protocol::turn_context::TurnContextOverride::default();
        if let Some(thinking_enabled) = thinking_enabled {
            context.metadata.insert(
                APP_SERVER_RUNTIME_BACKEND_METADATA_KEY.to_string(),
                json!({ "thinkingEnabled": thinking_enabled }),
            );
        }
        agent_runtime::session_config::SessionConfigBuilder::new("session-1")
            .turn_context(context)
            .build()
    }

    #[test]
    fn non_reasoning_chat_route_disables_provider_thinking_by_default() {
        let configuration = route_configuration(
            ModelProviderProtocol::ChatCompletions,
            vec!["streaming", "tools"],
        );
        let route_default = route_thinking_default(&configuration);
        let mut config = session_config(None);

        apply_route_thinking_default(&mut config, route_default);

        assert_eq!(
            config
                .turn_context
                .as_ref()
                .and_then(|context| context
                    .metadata
                    .get(APP_SERVER_RUNTIME_BACKEND_METADATA_KEY))
                .and_then(|metadata| metadata.get("thinkingEnabled")),
            Some(&json!(false))
        );
    }

    #[test]
    fn reasoning_capability_and_explicit_turn_control_are_preserved() {
        let reasoning_route = route_configuration(
            ModelProviderProtocol::ChatCompletions,
            vec!["streaming", "reasoning"],
        );
        assert_eq!(route_thinking_default(&reasoning_route), None);

        let plain_route = route_configuration(ModelProviderProtocol::ChatCompletions, vec![]);
        let mut config = session_config(Some(true));
        apply_route_thinking_default(&mut config, route_thinking_default(&plain_route));
        assert_eq!(
            config
                .turn_context
                .as_ref()
                .and_then(|context| context
                    .metadata
                    .get(APP_SERVER_RUNTIME_BACKEND_METADATA_KEY))
                .and_then(|metadata| metadata.get("thinkingEnabled")),
            Some(&json!(true))
        );

        let responses_route = route_configuration(ModelProviderProtocol::Responses, vec![]);
        assert_eq!(route_thinking_default(&responses_route), None);
    }
}
