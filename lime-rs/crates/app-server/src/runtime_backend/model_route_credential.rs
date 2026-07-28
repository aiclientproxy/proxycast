use lime_agent::SessionProviderConfig;
use lime_core::database::dao::api_key_provider::ProviderWithKeys;
use lime_core::database::DbConnection;
use lime_core::models::RuntimeProviderCredential;
use lime_services::api_key_provider_service::ApiKeyProviderService;
use lime_services::model_registry_service::{ModelRegistryService, ProviderModelCacheAccess};

pub(crate) struct RouteCredential {
    runtime_credential: Option<RuntimeProviderCredential>,
}

impl RouteCredential {
    pub(crate) fn unavailable() -> Self {
        Self {
            runtime_credential: None,
        }
    }

    pub(crate) fn runtime_credential(&self) -> Option<&RuntimeProviderCredential> {
        self.runtime_credential.as_ref()
    }

    pub(crate) fn credential_ref(&self) -> Option<&str> {
        self.runtime_credential
            .as_ref()
            .map(|credential| credential.uuid.as_str())
    }

    pub(crate) fn model_cache_access<'a>(
        &'a self,
        provider: Option<&ProviderWithKeys>,
    ) -> ProviderModelCacheAccess<'a> {
        match self.runtime_credential.as_ref() {
            Some(credential) => ProviderModelCacheAccess::Credential(credential),
            None if provider.is_some_and(provider_is_keyless) => ProviderModelCacheAccess::Keyless,
            None => ProviderModelCacheAccess::Unavailable,
        }
    }
}

pub(crate) async fn resolve_route_credential(
    db: &DbConnection,
    service: &ApiKeyProviderService,
    provider_id: &str,
    provider: Option<&ProviderWithKeys>,
    direct_provider_config: Option<&SessionProviderConfig>,
    preferred_credential_ref: Option<&str>,
    excluded_credential_refs: &[String],
) -> Result<RouteCredential, String> {
    if direct_provider_config.is_some() || provider.is_some_and(provider_is_keyless) {
        return Ok(RouteCredential::unavailable());
    }

    let preferred_credential_ref = preferred_credential_ref.filter(|credential_ref| {
        !excluded_credential_refs
            .iter()
            .any(|excluded| excluded == credential_ref)
    });
    let preferred_credential_ref = match preferred_credential_ref {
        Some(credential_ref) if service.runtime_credential_is_cooling(credential_ref)? => None,
        preferred => preferred,
    };
    let runtime_credential = match preferred_credential_ref {
        Some(credential_ref) => service
            .select_runtime_credential_by_ref(db, provider_id, credential_ref)?
            .ok_or_else(|| "resolved_credential_unavailable".to_string())?,
        None => {
            let excluded_credential_refs = excluded_credential_refs
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            let selected = service
                .select_credential_for_provider_excluding(
                    db,
                    provider_id,
                    Some(provider_id),
                    None,
                    &excluded_credential_refs,
                )
                .await?;
            let Some(selected) = selected else {
                return match provider {
                    Some(_) => Err("resolved_credential_unavailable".to_string()),
                    None => Ok(RouteCredential::unavailable()),
                };
            };
            service
                .select_runtime_credential_by_ref(db, provider_id, &selected.uuid)?
                .ok_or_else(|| "resolved_credential_provider_mismatch".to_string())?
        }
    };

    Ok(RouteCredential {
        runtime_credential: Some(runtime_credential),
    })
}

fn provider_is_keyless(provider: &ProviderWithKeys) -> bool {
    !ModelRegistryService::requires_api_key_for_runtime(
        &provider.provider.id,
        &provider.provider.api_host,
        provider.provider.effective_provider_type(),
    )
}
