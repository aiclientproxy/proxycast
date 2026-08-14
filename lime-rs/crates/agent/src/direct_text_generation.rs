use crate::lime_session_repository::LimeSessionRepository;
use crate::provider_configuration::{
    configure_model_route_provider_for_session_with_provider_and_credential_ref,
    ModelRouteProviderConfiguration, SessionProviderConfig,
};
use crate::request_tool_policy::{
    resolve_request_tool_policy_with_mode,
    stream_runtime_reply_with_configured_provider_for_direct_generation,
    stream_runtime_reply_with_policy, RequestToolPolicyMode,
};
use crate::turn_context_configuration::AgentTurnContext;
use crate::{AgentEvent, AgentRuntimeState, AgentTokenUsage, SessionConfigBuilder};
use agent_protocol::SessionId;
use lime_core::database::DbConnection;
use thread_store::session_repository::{SessionDetail, SessionRepository};

#[derive(Debug, Clone)]
pub struct DirectTextGenerationRequest {
    pub session_id: String,
    pub thread_id: String,
    pub turn_id: String,
    pub system_prompt: String,
    pub user_prompt: String,
    pub turn_context: Option<AgentTurnContext>,
    pub provider_configuration: Option<ModelRouteProviderConfiguration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectTextGenerationResult {
    pub text: String,
    pub usage: Option<AgentTokenUsage>,
    pub provider_config: Option<SessionProviderConfig>,
}

pub async fn run_direct_text_generation_with_db(
    agent_state: &AgentRuntimeState,
    request: DirectTextGenerationRequest,
    db: &DbConnection,
) -> Result<DirectTextGenerationResult, String> {
    let DirectTextGenerationRequest {
        session_id,
        thread_id,
        turn_id,
        system_prompt,
        user_prompt,
        turn_context,
        provider_configuration,
    } = request;
    let configured_provider = match provider_configuration {
        Some(provider_configuration) => {
            let credential_ref = provider_configuration.credential_ref.clone();
            Some(
                configure_model_route_provider_for_session_with_provider_and_credential_ref(
                    agent_state,
                    db,
                    &session_id,
                    provider_configuration,
                    credential_ref.as_deref(),
                )
                .await?,
            )
        }
        None => None,
    };
    let request_tool_policy =
        resolve_request_tool_policy_with_mode(Some(false), Some(RequestToolPolicyMode::Disabled));
    let mut session_config = SessionConfigBuilder::new(session_id.clone())
        .thread_id(thread_id)
        .turn_id(turn_id)
        .system_prompt(system_prompt)
        .include_context_trace(false);
    if let Some(turn_context) = turn_context {
        session_config = session_config.turn_context(turn_context);
    }
    let session_config = session_config.build();
    let mut text = String::new();
    let mut usage: Option<AgentTokenUsage> = None;
    let execution = match configured_provider.as_ref() {
        Some(configured_provider) => {
            stream_runtime_reply_with_configured_provider_for_direct_generation(
                agent_state,
                &user_prompt,
                None,
                session_config,
                None,
                &request_tool_policy,
                configured_provider,
                |event| collect_model_text(event, &mut text, &mut usage),
            )
            .await
        }
        None => {
            stream_runtime_reply_with_policy(
                agent_state,
                &user_prompt,
                None,
                session_config,
                None,
                &request_tool_policy,
                None,
                |event| collect_model_text(event, &mut text, &mut usage),
            )
            .await
        }
    };
    execution.map_err(|error| error.message)?;
    if usage.is_none() {
        usage = resolve_session_usage_from_repository(db, &session_id);
        match usage.as_ref() {
            Some(usage) => tracing::info!(
                session_id = %session_id,
                source = "session_repository",
                input_tokens = usage.input_tokens,
                output_tokens = usage.output_tokens,
                "[AgentRuntime] direct text generation usage recovered from persisted session stats"
            ),
            None => tracing::info!(
                session_id = %session_id,
                "[AgentRuntime] direct text generation completed without usage stats"
            ),
        }
    }
    Ok(DirectTextGenerationResult {
        text,
        usage,
        provider_config: configured_provider.map(|configured| configured.into_config()),
    })
}

fn resolve_session_usage_from_repository(
    db: &DbConnection,
    session_id: &str,
) -> Option<AgentTokenUsage> {
    let repository = LimeSessionRepository::new(db.clone());
    let session = repository
        .get_session(&SessionId::new(session_id))
        .ok()
        .flatten()?;
    resolve_usage_from_session_detail(&session)
}

fn resolve_usage_from_session_detail(session: &SessionDetail) -> Option<AgentTokenUsage> {
    crate::session_usage_projection::project_token_usage(
        session.input_tokens,
        session.output_tokens,
        session.cached_input_tokens,
        session.cache_creation_input_tokens,
    )
}

fn collect_model_text(
    event: &AgentEvent,
    output: &mut String,
    usage: &mut Option<AgentTokenUsage>,
) {
    match event {
        AgentEvent::TextDelta { text, .. } => output.push_str(text),
        AgentEvent::TextDeltaBatch { text, .. } => output.push_str(text),
        AgentEvent::Done { usage: event_usage } => {
            if event_usage.is_some() {
                *usage = event_usage.clone();
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_model_text_retains_text_and_final_usage() {
        let mut text = String::new();
        let mut usage = None;

        collect_model_text(
            &AgentEvent::TextDelta {
                item_id: "text-1".to_string(),
                text: "好啊，".to_string(),
            },
            &mut text,
            &mut usage,
        );
        collect_model_text(
            &AgentEvent::Done {
                usage: Some(AgentTokenUsage {
                    input_tokens: 31_000,
                    output_tokens: 0,
                    cached_input_tokens: Some(1_000),
                    cache_creation_input_tokens: None,
                    codex_rollout_budget_units: None,
                }),
            },
            &mut text,
            &mut usage,
        );

        assert_eq!(text, "好啊，");
        assert_eq!(
            usage,
            Some(AgentTokenUsage {
                input_tokens: 31_000,
                output_tokens: 0,
                cached_input_tokens: Some(1_000),
                cache_creation_input_tokens: None,
                codex_rollout_budget_units: None,
            })
        );
    }

    #[test]
    fn resolve_usage_from_session_detail_reads_repository_token_stats() {
        let session = SessionDetail {
            metadata: thread_store::session_repository::SessionMetadata {
                id: SessionId::new("session-1"),
                title: "测试会话".to_string(),
                model: "gpt-5.2".to_string(),
                session_type: "chat".to_string(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                working_dir: None,
                user_set_name: false,
                provider_name: None,
                extension_data: serde_json::Value::Null,
            },
            total_tokens: Some(43),
            input_tokens: Some(31),
            output_tokens: Some(12),
            cached_input_tokens: Some(7),
            cache_creation_input_tokens: None,
            accumulated_total_tokens: None,
            accumulated_input_tokens: None,
            accumulated_output_tokens: None,
            message_count: 0,
            schedule_id: None,
            recipe_json: None,
            user_recipe_values_json: None,
            model_config_json: None,
        };

        assert_eq!(
            resolve_usage_from_session_detail(&session),
            Some(AgentTokenUsage {
                input_tokens: 31,
                output_tokens: 12,
                cached_input_tokens: Some(7),
                cache_creation_input_tokens: None,
                codex_rollout_budget_units: None,
            })
        );
    }
}
