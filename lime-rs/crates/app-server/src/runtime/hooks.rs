use super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::protocol::v2::{
    HookErrorInfo, HookMetadata, HooksListEntry, HooksListParams, HooksListResponse,
};
use std::path::PathBuf;
use tool_runtime::hook_discovery::{discover_hooks, HookDiscoveryInput};

impl RuntimeCore {
    pub async fn list_hooks(
        &self,
        params: HooksListParams,
    ) -> Result<HooksListResponse, RuntimeCoreError> {
        let codex_home = lime_core::app_paths::resolve_codex_home_dir()
            .ok_or_else(|| RuntimeCoreError::Backend("cannot resolve CODEX_HOME".to_string()))?;
        let cwds = if params.cwds.is_empty() {
            vec![std::env::current_dir().map_err(|error| {
                RuntimeCoreError::Backend(format!("cannot resolve current directory: {error}"))
            })?]
        } else {
            params.cwds
        };

        let data = cwds
            .into_iter()
            .map(|cwd| list_hooks_for_cwd(codex_home.clone(), cwd))
            .collect();
        Ok(HooksListResponse { data })
    }
}

fn list_hooks_for_cwd(codex_home: PathBuf, cwd: PathBuf) -> HooksListEntry {
    let report = discover_hooks(&HookDiscoveryInput {
        codex_home,
        cwd: cwd.clone(),
        plugins: Vec::new(),
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
