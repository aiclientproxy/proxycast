//! Experimental feature catalog and process-wide enablement.

use super::{dispatch_result, parse_params, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{
    ExperimentalFeature, ExperimentalFeatureEnablementSetParams,
    ExperimentalFeatureEnablementSetResponse, ExperimentalFeatureListParams,
    ExperimentalFeatureListResponse, ExperimentalFeatureStage,
};
use app_server_protocol::{error_codes, JsonRpcError};
use lime_core::config::load_config;
use lime_core::config::save_config;

const WEBMCP_FEATURE: &str = "webmcp";

fn config_error(error: impl std::fmt::Display) -> JsonRpcError {
    JsonRpcError::new(
        error_codes::RUNTIME_ERROR,
        format!("failed to access Lime config: {error}"),
    )
}

impl RequestProcessor {
    pub(super) async fn handle_experimental_feature_list_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ExperimentalFeatureListParams = parse_params(params)?;
        if let Some(thread_id) = params.thread_id.as_deref() {
            self.resolve_loaded_v2_thread_session(thread_id)?;
        }

        let enabled = load_config()
            .map(|config| config.experimental.webmcp.enabled)
            .map_err(|error| config_error(error))?;
        let data = vec![ExperimentalFeature {
            name: WEBMCP_FEATURE.to_string(),
            stage: ExperimentalFeatureStage::UnderDevelopment,
            display_name: Some("WebMCP".to_string()),
            description: Some(
                "Expose a controlled WebMCP surface to supported web apps.".to_string(),
            ),
            announcement: None,
            enabled,
            default_enabled: false,
        }];
        let start = parse_cursor(params.cursor.as_deref(), data.len())?;
        let limit = params.limit.unwrap_or(data.len() as u32).max(1) as usize;
        let end = start.saturating_add(limit).min(data.len());
        dispatch_result(ExperimentalFeatureListResponse {
            data: data[start..end].to_vec(),
            next_cursor: (end < data.len()).then(|| end.to_string()),
        })
    }

    pub(super) async fn handle_experimental_feature_enablement_set_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let ExperimentalFeatureEnablementSetParams { mut enablement } = parse_params(params)?;
        enablement.retain(|name, _| name == WEBMCP_FEATURE);
        if enablement.is_empty() {
            return dispatch_result(ExperimentalFeatureEnablementSetResponse { enablement });
        }

        let _guard = super::config::config_lock()?;
        let mut config = load_config().map_err(config_error)?;
        if let Some(enabled) = enablement.get(WEBMCP_FEATURE).copied() {
            config.experimental.webmcp.enabled = enabled;
        }
        save_config(&config).map_err(config_error)?;
        dispatch_result(ExperimentalFeatureEnablementSetResponse { enablement })
    }
}

fn parse_cursor(cursor: Option<&str>, total: usize) -> Result<usize, JsonRpcError> {
    let Some(cursor) = cursor else {
        return Ok(0);
    };
    let start = cursor.parse::<usize>().map_err(|_| {
        JsonRpcError::new(
            error_codes::INVALID_PARAMS,
            format!("invalid cursor: {cursor}"),
        )
    })?;
    if start > total {
        return Err(JsonRpcError::new(
            error_codes::INVALID_PARAMS,
            format!("cursor {start} exceeds total feature flags {total}"),
        ));
    }
    Ok(start)
}

#[cfg(test)]
mod tests {
    use super::parse_cursor;

    #[test]
    fn cursor_is_bounded_and_numeric() {
        assert_eq!(parse_cursor(None, 1).unwrap(), 0);
        assert_eq!(parse_cursor(Some("1"), 1).unwrap(), 1);
        assert!(parse_cursor(Some("nope"), 1).is_err());
        assert!(parse_cursor(Some("2"), 1).is_err());
    }
}
