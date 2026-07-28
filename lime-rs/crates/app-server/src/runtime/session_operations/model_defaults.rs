use super::super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::protocol::v2::ThreadSettingsUpdateParams;

impl RuntimeCore {
    pub(super) async fn apply_target_model_defaults(
        &self,
        params: &mut ThreadSettingsUpdateParams,
    ) -> Result<(), RuntimeCoreError> {
        if params.service_tier.is_some() || !self.backend.requires_provider_selection() {
            return Ok(());
        }
        let target_model_update = params.model.as_deref().or_else(|| {
            params
                .collaboration_mode
                .as_ref()
                .map(|mode| mode.settings.model.as_str())
        });
        if target_model_update.is_none() && params.model_provider.is_none() {
            return Ok(());
        }
        let (_, current, _) = self
            .loaded_thread_settings(&params.thread_id)?
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(params.thread_id.clone()))?;
        let target_model = target_model_update.unwrap_or(&current.model).to_string();
        let target_provider = params
            .model_provider
            .clone()
            .unwrap_or_else(|| current.model_provider.clone());
        if target_model == current.model && target_provider == current.model_provider {
            return Ok(());
        }
        let catalogs = self.model_catalog(Some(&target_provider)).await?;
        if let Some(default_service_tier) =
            super::super::model_providers::catalog_model_default_service_tier(
                &catalogs,
                &target_provider,
                &target_model,
            )
        {
            params.service_tier = Some(default_service_tier);
        }
        Ok(())
    }
}
