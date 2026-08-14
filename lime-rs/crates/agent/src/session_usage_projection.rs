use crate::protocol::AgentTokenUsage;
use agent_runtime::session_execution::{
    project_session_execution_runtime_usage, SessionExecutionRuntimeUsageSource,
};

pub(crate) fn project_token_usage(
    input_tokens: Option<i32>,
    output_tokens: Option<i32>,
    cached_input_tokens: Option<i32>,
    cache_creation_input_tokens: Option<i32>,
) -> Option<AgentTokenUsage> {
    project_token_usage_source(SessionExecutionRuntimeUsageSource {
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_creation_input_tokens,
    })
}

pub(crate) fn project_token_usage_source(
    source: SessionExecutionRuntimeUsageSource,
) -> Option<AgentTokenUsage> {
    project_session_execution_runtime_usage(source).map(|usage| AgentTokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        cache_creation_input_tokens: usage.cache_creation_input_tokens,
        codex_rollout_budget_units: None,
    })
}

#[cfg(test)]
mod tests {
    use super::project_token_usage;
    use crate::AgentTokenUsage;

    #[test]
    fn project_token_usage_ignores_negative_optional_cache_values() {
        assert_eq!(
            project_token_usage(Some(31_000), Some(0), Some(-1), Some(512)),
            Some(AgentTokenUsage {
                input_tokens: 31_000,
                output_tokens: 0,
                cached_input_tokens: None,
                cache_creation_input_tokens: Some(512),
                codex_rollout_budget_units: None,
            })
        );
    }

    #[test]
    fn project_token_usage_requires_non_negative_input_and_output() {
        assert_eq!(project_token_usage(Some(31_000), None, None, None), None);
        assert_eq!(project_token_usage(Some(-1), Some(0), None, None), None);
    }
}
