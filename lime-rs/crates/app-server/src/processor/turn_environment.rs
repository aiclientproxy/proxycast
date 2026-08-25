use super::{to_jsonrpc_error, RequestProcessor};
use app_server_protocol::protocol::v2::TurnEnvironmentParams;
use app_server_protocol::JsonRpcError;

impl RequestProcessor {
    pub(super) async fn resolve_turn_environment_selections(
        &self,
        thread_id: &str,
        requested: Option<Vec<TurnEnvironmentParams>>,
    ) -> Result<Option<Vec<TurnEnvironmentParams>>, JsonRpcError> {
        let selections = match requested {
            Some(selections) => Some(selections),
            None => {
                let thread = self
                    .runtime
                    .read_thread(agent_protocol::thread::ThreadReadParams {
                        thread_id: agent_protocol::ThreadId::new(thread_id),
                        turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
                    })
                    .await
                    .map_err(to_jsonrpc_error)?;
                persisted_environment_selections(&thread.thread.metadata)?
            }
        };
        self.normalize_environment_selections(selections).await
    }
}

pub(super) fn persisted_environment_selections(
    metadata: &serde_json::Value,
) -> Result<Option<Vec<TurnEnvironmentParams>>, JsonRpcError> {
    let Some(value) = metadata.get("environments") else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    serde_json::from_value(value.clone())
        .map(Some)
        .map_err(|error| {
            JsonRpcError::new(
                app_server_protocol::error_codes::INVALID_REQUEST,
                format!("persisted thread environments are invalid: {error}"),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_omitted_and_explicit_empty_environment_states() {
        assert_eq!(
            persisted_environment_selections(&serde_json::json!({})).expect("omitted state"),
            None
        );
        assert_eq!(
            persisted_environment_selections(&serde_json::json!({"environments": []}))
                .expect("explicit empty state"),
            Some(Vec::new())
        );
    }
}
