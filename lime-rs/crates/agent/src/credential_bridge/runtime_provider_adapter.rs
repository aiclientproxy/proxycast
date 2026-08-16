use model_provider::current_client::{
    CurrentProviderClient, CurrentProviderError, CurrentProviderHealthRegistry,
    CurrentProviderSession,
};
use model_provider::provider_stream::RuntimeReplyProviderHandle;
use model_provider::runtime_provider::RuntimeProviderConfig;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct ConfiguredReplyProvider {
    client: Arc<CurrentProviderClient>,
    runtime_handle: RuntimeReplyProviderHandle,
}

impl ConfiguredReplyProvider {
    pub(crate) fn client(&self) -> Arc<CurrentProviderClient> {
        Arc::clone(&self.client)
    }

    pub(crate) fn session(&self) -> CurrentProviderSession {
        CurrentProviderSession::new(Arc::clone(&self.client))
    }

    pub(crate) fn runtime_handle(&self) -> &RuntimeReplyProviderHandle {
        &self.runtime_handle
    }
}

pub(crate) fn create_configured_reply_provider(
    config: &RuntimeProviderConfig,
    health_registry: &CurrentProviderHealthRegistry,
) -> Result<ConfiguredReplyProvider, CurrentProviderError> {
    let client = CurrentProviderClient::new_with_health_registry(config.clone(), health_registry)?;
    let runtime_handle = client.runtime_handle();
    Ok(ConfiguredReplyProvider {
        client: Arc::new(client),
        runtime_handle,
    })
}
