mod boundary;
pub(crate) mod locale_copy;
mod prompt_context;
mod style_pack_install;
mod style_pack_paths;
mod style_pack_registry;
mod style_pack_store;
mod style_profile;

use super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::{
    SoulStylePackInstallParams, SoulStylePackInstallResponse,
    SoulStylePackInstallStatus as ProtocolStylePackInstallStatus, SoulStylePackListEntry,
    SoulStylePackListParams, SoulStylePackListResponse, SoulStylePackMutableStatus,
    SoulStylePackStatusSetParams, SoulStylePackStatusSetResponse, SoulStylePackUninstallParams,
    SoulStylePackUninstallResponse,
};

pub(crate) const MEMORY_SOUL_PROMPT_CONTEXT_KEY: &str = "memory_soul_prompt_context";

pub(crate) use prompt_context::{
    memory_soul_prompt_context_from_config, soul_packet_from_metadata,
};
pub use style_pack_paths::install_style_pack_data_root;

impl RuntimeCore {
    pub(crate) fn install_soul_style_pack(
        &self,
        params: SoulStylePackInstallParams,
    ) -> Result<SoulStylePackInstallResponse, RuntimeCoreError> {
        let root = style_pack_paths::style_pack_data_root().map_err(RuntimeCoreError::Backend)?;
        let record = style_pack_store::install_style_pack_from_sources(
            &root,
            style_pack_store::StylePackInstallSources {
                manifest_source: params.manifest_source,
                locale_sources: params.locale_sources,
                enable_after_install: params.enable_after_install,
            },
        )
        .map_err(RuntimeCoreError::Backend)?;
        Ok(SoulStylePackInstallResponse {
            pack_id: record.pack_id,
            profile_ids: record.profile_ids,
            status: protocol_install_status(record.status),
        })
    }

    pub(crate) fn list_soul_style_packs(
        &self,
        _params: SoulStylePackListParams,
    ) -> Result<SoulStylePackListResponse, RuntimeCoreError> {
        let packs = style_pack_registry::list_installed_style_packs()
            .map_err(RuntimeCoreError::Backend)?
            .into_iter()
            .map(|entry| SoulStylePackListEntry {
                pack_id: entry.pack_id,
                source: entry.source,
                status: protocol_install_status(entry.status),
                profile_ids: entry.profile_ids,
                manifest_source: entry.manifest_source,
                locale_sources: entry.locale_sources,
                updated_at: entry.updated_at,
                integrity_digest: entry.integrity_digest,
            })
            .collect();
        Ok(SoulStylePackListResponse { packs })
    }

    pub(crate) fn set_soul_style_pack_status(
        &self,
        params: SoulStylePackStatusSetParams,
    ) -> Result<SoulStylePackStatusSetResponse, RuntimeCoreError> {
        let root = style_pack_paths::style_pack_data_root().map_err(RuntimeCoreError::Backend)?;
        let next = mutable_install_status(params.status);
        style_pack_store::set_style_pack_status_from_root(&root, &params.pack_id, next)
            .map_err(RuntimeCoreError::Backend)?;
        Ok(SoulStylePackStatusSetResponse {
            pack_id: params.pack_id,
            status: protocol_install_status(next),
        })
    }

    pub(crate) fn uninstall_soul_style_pack(
        &self,
        params: SoulStylePackUninstallParams,
    ) -> Result<SoulStylePackUninstallResponse, RuntimeCoreError> {
        let root = style_pack_paths::style_pack_data_root().map_err(RuntimeCoreError::Backend)?;
        style_pack_store::uninstall_style_pack_from_root(&root, &params.pack_id)
            .map_err(RuntimeCoreError::Backend)?;
        Ok(SoulStylePackUninstallResponse {
            pack_id: params.pack_id,
            status: ProtocolStylePackInstallStatus::Uninstalled,
        })
    }
}

fn mutable_install_status(
    status: SoulStylePackMutableStatus,
) -> style_pack_install::StylePackInstallStatus {
    match status {
        SoulStylePackMutableStatus::Enabled => style_pack_install::StylePackInstallStatus::Enabled,
        SoulStylePackMutableStatus::Disabled => {
            style_pack_install::StylePackInstallStatus::Disabled
        }
    }
}

fn protocol_install_status(
    status: style_pack_install::StylePackInstallStatus,
) -> ProtocolStylePackInstallStatus {
    match status {
        style_pack_install::StylePackInstallStatus::Discovered => {
            ProtocolStylePackInstallStatus::Discovered
        }
        style_pack_install::StylePackInstallStatus::Downloading => {
            ProtocolStylePackInstallStatus::Downloading
        }
        style_pack_install::StylePackInstallStatus::Validating => {
            ProtocolStylePackInstallStatus::Validating
        }
        style_pack_install::StylePackInstallStatus::Installing => {
            ProtocolStylePackInstallStatus::Installing
        }
        style_pack_install::StylePackInstallStatus::Installed => {
            ProtocolStylePackInstallStatus::Installed
        }
        style_pack_install::StylePackInstallStatus::Enabled => {
            ProtocolStylePackInstallStatus::Enabled
        }
        style_pack_install::StylePackInstallStatus::Disabled => {
            ProtocolStylePackInstallStatus::Disabled
        }
        style_pack_install::StylePackInstallStatus::Failed => {
            ProtocolStylePackInstallStatus::Failed
        }
        style_pack_install::StylePackInstallStatus::Uninstalled => {
            ProtocolStylePackInstallStatus::Uninstalled
        }
    }
}
