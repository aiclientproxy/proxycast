use super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::protocol::v2::{
    HookErrorInfo, HookMetadata, HooksListEntry, HooksListParams, HooksListResponse,
};
use std::path::PathBuf;
use tool_runtime::hook_discovery::{
    discover_hooks, plugin_sources_from_activations, HookDiscoveryInput,
};

impl RuntimeCore {
    pub async fn list_hooks(
        &self,
        params: HooksListParams,
    ) -> Result<HooksListResponse, RuntimeCoreError> {
        let codex_home = lime_core::app_paths::resolve_codex_home_dir()
            .ok_or_else(|| RuntimeCoreError::Backend("cannot resolve CODEX_HOME".to_string()))?;
        let activations = self
            .app_data_source
            .list_plugin_catalog_activations()
            .await?;
        let plugins = plugin_sources_from_activations(&activations);
        let cwds = if params.cwds.is_empty() {
            vec![std::env::current_dir().map_err(|error| {
                RuntimeCoreError::Backend(format!("cannot resolve current directory: {error}"))
            })?]
        } else {
            params.cwds
        };

        let data = cwds
            .into_iter()
            .map(|cwd| list_hooks_for_cwd(codex_home.clone(), cwd, plugins.clone()))
            .collect();
        Ok(HooksListResponse { data })
    }
}

fn list_hooks_for_cwd(
    codex_home: PathBuf,
    cwd: PathBuf,
    plugins: Vec<tool_runtime::hook_discovery::HookPluginSource>,
) -> HooksListEntry {
    let report = discover_hooks(&HookDiscoveryInput {
        codex_home,
        cwd: cwd.clone(),
        plugins,
    });
    HooksListEntry {
        cwd,
        hooks: report
            .hooks
            .iter()
            .map(|hook| HookMetadata::from(&hook.snapshot))
            .collect(),
        warnings: report.warnings,
        errors: report
            .errors
            .into_iter()
            .map(|error| HookErrorInfo {
                path: error.path,
                message: error.message,
            })
            .collect(),
    }
}
