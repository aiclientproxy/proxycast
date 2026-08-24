//! model domain handlers for the App Server processor.

use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::ServerNotification as V2ServerNotification;
use app_server_protocol::{
    JsonRpcError, ModelListParams, ModelListUpdatedNotification, ModelProviderAliasReadParams,
    ModelProviderCapabilitiesReadResponse, ModelProviderConfigExportParams,
    ModelProviderConfigImportParams, ModelProviderCreateParams, ModelProviderDeleteParams,
    ModelProviderFetchModelsParams, ModelProviderFetchModelsResponse, ModelProviderKeyCreateParams,
    ModelProviderKeyDeleteParams, ModelProviderKeyUpdateParams, ModelProviderReadParams,
    ModelProviderSortOrdersUpdateParams, ModelProviderTestChatParams,
    ModelProviderTestConnectionParams, ModelProviderUiStateReadParams,
    ModelProviderUiStateWriteParams, ModelProviderUpdateParams, METHOD_MODEL_LIST,
    METHOD_MODEL_PREFERENCES_LIST, METHOD_MODEL_PROVIDER_ALIAS_LIST,
    METHOD_MODEL_PROVIDER_ALIAS_READ, METHOD_MODEL_PROVIDER_CAPABILITIES_READ,
    METHOD_MODEL_PROVIDER_CATALOG_LIST, METHOD_MODEL_PROVIDER_CONFIG_EXPORT,
    METHOD_MODEL_PROVIDER_CONFIG_IMPORT, METHOD_MODEL_PROVIDER_CREATE,
    METHOD_MODEL_PROVIDER_DELETE, METHOD_MODEL_PROVIDER_FETCH_MODELS,
    METHOD_MODEL_PROVIDER_KEY_CREATE, METHOD_MODEL_PROVIDER_KEY_DELETE,
    METHOD_MODEL_PROVIDER_KEY_UPDATE, METHOD_MODEL_PROVIDER_LIST, METHOD_MODEL_PROVIDER_READ,
    METHOD_MODEL_PROVIDER_SORT_ORDERS_UPDATE, METHOD_MODEL_PROVIDER_TEST_CHAT,
    METHOD_MODEL_PROVIDER_TEST_CONNECTION, METHOD_MODEL_PROVIDER_UI_STATE_READ,
    METHOD_MODEL_PROVIDER_UI_STATE_WRITE, METHOD_MODEL_PROVIDER_UPDATE,
    METHOD_MODEL_SYNC_STATE_READ,
};
use futures::future::BoxFuture;
use futures::FutureExt;
use model_provider::provider_capabilities::ProviderCapabilities;
use std::time::Duration;

const MODEL_CATALOG_RETRY_MAX_ATTEMPTS: u32 = 5;
const MODEL_CATALOG_RETRY_BASE_DELAY: Duration = Duration::from_secs(5);
const MODEL_CATALOG_RETRY_MAX_DELAY: Duration = Duration::from_secs(60);

impl RequestProcessor {
    pub(super) fn dispatch_model_request<'a>(
        &'a self,
        method: &str,
        params: &mut Option<serde_json::Value>,
    ) -> Option<BoxFuture<'a, Result<RpcDispatch, JsonRpcError>>> {
        let request = match method {
            METHOD_MODEL_LIST => self.handle_model_list_impl(params.take()).boxed(),
            METHOD_MODEL_PROVIDER_CAPABILITIES_READ => self
                .handle_model_provider_capabilities_read_impl(params.take())
                .boxed(),
            METHOD_MODEL_PREFERENCES_LIST => self.handle_model_preferences_list_impl().boxed(),
            METHOD_MODEL_SYNC_STATE_READ => self.handle_model_sync_state_read_impl().boxed(),
            METHOD_MODEL_PROVIDER_LIST => self.handle_model_provider_list_impl().boxed(),
            METHOD_MODEL_PROVIDER_CATALOG_LIST => {
                self.handle_model_provider_catalog_list_impl().boxed()
            }
            METHOD_MODEL_PROVIDER_READ => {
                self.handle_model_provider_read_impl(params.take()).boxed()
            }
            METHOD_MODEL_PROVIDER_CREATE => self
                .handle_model_provider_create_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_UPDATE => self
                .handle_model_provider_update_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_DELETE => self
                .handle_model_provider_delete_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_SORT_ORDERS_UPDATE => self
                .handle_model_provider_sort_orders_update_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_CONFIG_EXPORT => self
                .handle_model_provider_config_export_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_CONFIG_IMPORT => self
                .handle_model_provider_config_import_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_TEST_CONNECTION => self
                .handle_model_provider_test_connection_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_TEST_CHAT => self
                .handle_model_provider_test_chat_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_FETCH_MODELS => self
                .handle_model_provider_fetch_models_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_KEY_CREATE => self
                .handle_model_provider_key_create_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_KEY_UPDATE => self
                .handle_model_provider_key_update_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_KEY_DELETE => self
                .handle_model_provider_key_delete_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_UI_STATE_READ => self
                .handle_model_provider_ui_state_read_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_UI_STATE_WRITE => self
                .handle_model_provider_ui_state_write_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_ALIAS_READ => self
                .handle_model_provider_alias_read_impl(params.take())
                .boxed(),
            METHOD_MODEL_PROVIDER_ALIAS_LIST => {
                self.handle_model_provider_alias_list_impl().boxed()
            }
            _ => return None,
        };
        Some(request)
    }

    pub(super) async fn handle_model_list_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelListParams = parse_params(params)?;
        let response = self
            .runtime
            .list_models(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_capabilities_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let _: app_server_protocol::ModelProviderCapabilitiesReadParams = parse_params(params)?;
        let (provider, base_url) = self
            .runtime
            .current_model_provider_route()
            .map_err(to_jsonrpc_error)?;
        let capabilities =
            ProviderCapabilities::from_provider_route(&provider, base_url.as_deref())
                .unwrap_or(ProviderCapabilities::NONE);
        dispatch_result(ModelProviderCapabilitiesReadResponse {
            namespace_tools: capabilities.namespace_tools,
            image_generation: capabilities.image_generation,
            web_search: capabilities.web_search,
        })
    }

    pub(super) async fn handle_model_preferences_list_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_model_preferences()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_sync_state_read_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .read_model_sync_state()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_list_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_model_providers()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_catalog_list_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_model_provider_catalog()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderReadParams = parse_params(params)?;
        let response = self
            .runtime
            .read_model_provider(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_create_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderCreateParams = parse_params(params)?;
        let response = self
            .runtime
            .create_model_provider(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_model_list_updated(Some(response.provider.id.clone()))
            .await;
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderUpdateParams = parse_params(params)?;
        let response = self
            .runtime
            .update_model_provider(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_model_list_updated(Some(response.provider.id.clone()))
            .await;
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_delete_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderDeleteParams = parse_params(params)?;
        let response = self
            .runtime
            .delete_model_provider(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_model_list_updated(None).await;
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_sort_orders_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderSortOrdersUpdateParams = parse_params(params)?;
        let response = self
            .runtime
            .update_model_provider_sort_orders(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_model_list_updated(None).await;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_config_export_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderConfigExportParams = parse_params(params)?;
        let response = self
            .runtime
            .export_model_provider_config(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_config_import_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderConfigImportParams = parse_params(params)?;
        let response = self
            .runtime
            .import_model_provider_config(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_model_list_updated(None).await;
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_test_connection_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderTestConnectionParams = parse_params(params)?;
        let response = self
            .runtime
            .test_model_provider_connection(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_test_chat_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderTestChatParams = parse_params(params)?;
        let response = self
            .runtime
            .test_model_provider_chat(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_fetch_models_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderFetchModelsParams = parse_params(params)?;
        let provider_id = params.provider_id.clone();
        let response = self
            .runtime
            .fetch_model_provider_models(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_model_list_updated(Some(provider_id)).await;
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_key_create_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderKeyCreateParams = parse_params(params)?;
        let response = self
            .runtime
            .create_model_provider_key(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.refresh_model_catalog_after_credential_change(&response.key.provider_id)
            .await;
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_key_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderKeyUpdateParams = parse_params(params)?;
        let catalog_changed = params.enabled.is_some();
        let response = self
            .runtime
            .update_model_provider_key(params)
            .await
            .map_err(to_jsonrpc_error)?;
        if catalog_changed && response.key.enabled {
            self.refresh_model_catalog_after_credential_change(&response.key.provider_id)
                .await;
        } else if catalog_changed {
            self.publish_model_list_updated(Some(response.key.provider_id.clone()))
                .await;
        }
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_key_delete_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderKeyDeleteParams = parse_params(params)?;
        let provider_id = self
            .runtime
            .list_model_providers()
            .await
            .ok()
            .and_then(|response| {
                response.providers.into_iter().find_map(|provider| {
                    provider
                        .api_keys
                        .iter()
                        .any(|key| key.id == params.key_id)
                        .then_some(provider.id)
                })
            });
        let response = self
            .runtime
            .delete_model_provider_key(params)
            .await
            .map_err(to_jsonrpc_error)?;
        if let Some(provider_id) = provider_id {
            self.refresh_model_catalog_after_credential_change(&provider_id)
                .await;
        } else {
            self.publish_model_list_updated(None).await;
        }
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_ui_state_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderUiStateReadParams = parse_params(params)?;
        let response = self
            .runtime
            .read_model_provider_ui_state(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_ui_state_write_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderUiStateWriteParams = parse_params(params)?;
        let response = self
            .runtime
            .write_model_provider_ui_state(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_alias_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderAliasReadParams = parse_params(params)?;
        let response = self
            .runtime
            .read_model_provider_alias(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_alias_list_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_model_provider_aliases()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    async fn refresh_model_catalog_after_credential_change(&self, provider_id: &str) {
        let result = self
            .runtime
            .refresh_model_provider_catalog(provider_id)
            .await;
        let should_retry = match &result {
            Ok(response) => model_catalog_response_is_retryable(response),
            Err(_) => false,
        };
        match result {
            Ok(response) if response.source == "Api" => {}
            Ok(response) => tracing::warn!(
                provider_id,
                error = response
                    .error
                    .as_deref()
                    .unwrap_or("model catalog refresh failed"),
                "credential changed but provider model catalog refresh did not succeed"
            ),
            Err(error) => tracing::warn!(
                provider_id,
                error = %error,
                "credential changed but provider model catalog refresh could not run"
            ),
        }
        self.publish_model_list_updated(Some(provider_id.to_string()))
            .await;
        if should_retry {
            self.schedule_model_catalog_retry(provider_id).await;
        }
    }

    async fn schedule_model_catalog_retry(&self, provider_id: &str) {
        match self
            .runtime
            .has_model_provider_last_success(provider_id)
            .await
        {
            Ok(true) => return,
            Ok(false) => {}
            Err(error) => {
                tracing::warn!(
                    provider_id,
                    error = %error,
                    "model catalog retry skipped because last-success state could not be read"
                );
                return;
            }
        }
        let Some(permit) = self
            .runtime
            .acquire_model_provider_catalog_retry(provider_id)
            .await
        else {
            tracing::debug!(provider_id, "model catalog retry already in flight");
            return;
        };

        let processor = self.clone();
        let provider_id = provider_id.to_string();
        tokio::spawn(async move {
            let _permit = permit;
            for attempt in 1..=MODEL_CATALOG_RETRY_MAX_ATTEMPTS {
                match processor
                    .runtime
                    .has_model_provider_last_success(&provider_id)
                    .await
                {
                    Ok(true) => return,
                    Ok(false) => {}
                    Err(error) => {
                        tracing::warn!(
                            provider_id,
                            error = %error,
                            "model catalog retry stopped because last-success state could not be read"
                        );
                        return;
                    }
                }

                let response = match processor
                    .runtime
                    .refresh_model_provider_catalog(&provider_id)
                    .await
                {
                    Ok(response) if response.source == "Api" => {
                        tracing::info!(provider_id, attempt, "model catalog retry succeeded");
                        processor
                            .publish_model_list_updated(Some(provider_id.clone()))
                            .await;
                        processor
                            .runtime
                            .schedule_pending_route_recovery(processor.runtime_host_context());
                        return;
                    }
                    Ok(response) => response,
                    Err(error) => {
                        tracing::warn!(
                            provider_id,
                            attempt,
                            error = %error,
                            "model catalog retry stopped after runtime failure"
                        );
                        return;
                    }
                };

                if !model_catalog_response_is_retryable(&response)
                    || attempt == MODEL_CATALOG_RETRY_MAX_ATTEMPTS
                {
                    tracing::warn!(
                        provider_id,
                        attempt,
                        error_kind = response.error_kind.as_deref().unwrap_or("unknown"),
                        "model catalog retry stopped without a successful catalog"
                    );
                    return;
                }

                let delay = model_catalog_retry_delay(attempt);
                tracing::warn!(
                    provider_id,
                    attempt,
                    max_attempts = MODEL_CATALOG_RETRY_MAX_ATTEMPTS,
                    delay_ms = delay.as_millis(),
                    "model catalog retry scheduled"
                );
                tokio::time::sleep(delay).await;
            }
        });
    }

    async fn publish_model_list_updated(&self, provider_id: Option<String>) {
        let generation = match self.runtime.model_catalog_generation().await {
            Ok(generation) => generation,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "model catalog changed but its committed generation could not be read"
                );
                return;
            }
        };
        self.publish_server_notification(V2ServerNotification::ModelListUpdated(
            ModelListUpdatedNotification {
                generation,
                provider_id,
            },
        ))
        .await;
    }
}

fn model_catalog_response_is_retryable(response: &ModelProviderFetchModelsResponse) -> bool {
    match response.error_kind.as_deref() {
        Some("network" | "invalid_response") => true,
        Some("other") => response.request_url.is_some(),
        _ => false,
    }
}

fn model_catalog_retry_delay(attempt: u32) -> Duration {
    let exponent = attempt.saturating_sub(1).min(31);
    MODEL_CATALOG_RETRY_BASE_DELAY
        .saturating_mul(2_u32.saturating_pow(exponent))
        .min(MODEL_CATALOG_RETRY_MAX_DELAY)
}

#[cfg(test)]
mod retry_tests {
    use super::*;

    #[test]
    fn catalog_retry_classification_is_transient_only() {
        let response =
            |kind: Option<&str>, request_url: Option<&str>| ModelProviderFetchModelsResponse {
                source: "Error".to_string(),
                error_kind: kind.map(str::to_string),
                request_url: request_url.map(str::to_string),
                ..ModelProviderFetchModelsResponse::default()
            };

        assert!(model_catalog_response_is_retryable(&response(
            Some("network"),
            None
        )));
        assert!(model_catalog_response_is_retryable(&response(
            Some("invalid_response"),
            None
        )));
        assert!(model_catalog_response_is_retryable(&response(
            Some("other"),
            Some("https://api.example.com/v1/models")
        )));
        for kind in ["not_found", "unauthorized", "forbidden", "other"] {
            assert!(!model_catalog_response_is_retryable(&response(
                Some(kind),
                None
            )));
        }
    }

    #[test]
    fn catalog_retry_delay_is_exponential_and_bounded() {
        assert_eq!(model_catalog_retry_delay(1), Duration::from_secs(5));
        assert_eq!(model_catalog_retry_delay(2), Duration::from_secs(10));
        assert_eq!(model_catalog_retry_delay(3), Duration::from_secs(20));
        assert_eq!(model_catalog_retry_delay(5), Duration::from_secs(60));
        assert_eq!(model_catalog_retry_delay(99), Duration::from_secs(60));
    }
}

#[cfg(test)]
mod capability_route_tests {
    use super::*;
    use crate::runtime::configured_provider_base_url;

    #[test]
    fn default_openai_route_uses_openai_base_url_and_hosted_capabilities() {
        let mut config = lime_core::config::Config::default();
        config.default_provider = "openai-response".into();
        config.providers.openai.base_url = Some("https://api.openai.com/v1".into());

        let provider = config.default_provider.trim();
        let base_url = configured_provider_base_url(&config, provider);
        assert_eq!(base_url, Some("https://api.openai.com/v1"));
        assert_eq!(
            ProviderCapabilities::from_provider_route(provider, base_url),
            Some(ProviderCapabilities {
                namespace_tools: false,
                custom_tools: true,
                image_generation: true,
                web_search: true,
            })
        );
    }

    #[test]
    fn openai_compatible_route_fails_closed() {
        let mut config = lime_core::config::Config::default();
        config.providers.openai.base_url = Some("https://gateway.example/v1".into());

        let capabilities = ProviderCapabilities::from_provider_route(
            config.default_provider.trim(),
            configured_provider_base_url(&config, config.default_provider.trim()),
        );
        assert_eq!(capabilities, Some(ProviderCapabilities::NONE));
    }

    #[test]
    fn unsupported_provider_route_fails_closed() {
        let config = lime_core::config::Config {
            default_provider: "amazon-bedrock".into(),
            ..lime_core::config::Config::default()
        };

        assert_eq!(
            configured_provider_base_url(&config, "amazon-bedrock"),
            None
        );
        assert_eq!(
            ProviderCapabilities::from_provider_route("amazon-bedrock", None),
            None
        );
    }
}
