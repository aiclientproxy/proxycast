use super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::protocol::v2::{
    SkillDependencies as V2SkillDependencies, SkillErrorInfo, SkillInterface as V2SkillInterface,
    SkillMetadata, SkillScope as V2SkillScope, SkillToolDependency as V2SkillToolDependency,
    SkillsConfigWriteParams, SkillsConfigWriteResponse, SkillsExtraRootsSetParams,
    SkillsExtraRootsSetResponse, SkillsListEntry, SkillsListParams, SkillsListResponse,
};
use app_server_protocol::*;
use lime_core::config::{ConfigManager, SkillConfig};
use lime_skills::{
    agent_skill_roots_for_workspace, apply_agent_skill_config,
    build_agent_skill_snapshot_from_roots, load_skill_summary_report_from_directory,
    AgentSkillMetadata, AgentSkillScope,
};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

impl RuntimeCore {
    pub async fn list_skills(
        &self,
        params: SkillsListParams,
    ) -> Result<SkillsListResponse, RuntimeCoreError> {
        if params.force_reload {
            lime_skills::invalidate_agent_skill_snapshot_cache();
        }
        let cwds = if params.cwds.is_empty() {
            vec![std::env::current_dir().map_err(|error| {
                RuntimeCoreError::Backend(format!("cannot resolve current directory: {error}"))
            })?]
        } else {
            params.cwds
        };
        let config = load_skill_config()?;

        Ok(SkillsListResponse {
            data: cwds
                .into_iter()
                .map(|cwd| list_skills_for_cwd(cwd, &config))
                .collect(),
        })
    }

    pub async fn set_extra_skill_roots(
        &self,
        params: SkillsExtraRootsSetParams,
    ) -> Result<SkillsExtraRootsSetResponse, RuntimeCoreError> {
        lime_skills::set_runtime_extra_skill_roots(params.extra_roots)
            .map_err(RuntimeCoreError::InvalidRequest)?;
        Ok(SkillsExtraRootsSetResponse {})
    }

    pub async fn write_skill_config(
        &self,
        params: SkillsConfigWriteParams,
    ) -> Result<SkillsConfigWriteResponse, RuntimeCoreError> {
        let selector = SkillConfigSelector::from_params(&params)?;
        let _guard = skill_config_lock()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let config_path = ConfigManager::default_config_path();
        let mut manager = ConfigManager::load(&config_path).map_err(|error| {
            RuntimeCoreError::Backend(format!("failed to load skill settings: {error}"))
        })?;
        update_skill_config(manager.config_mut(), &selector, params.enabled);
        manager.save().map_err(|error| {
            RuntimeCoreError::Backend(format!("failed to update skill settings: {error}"))
        })?;
        lime_skills::invalidate_agent_skill_snapshot_cache();
        Ok(SkillsConfigWriteResponse {
            effective_enabled: params.enabled,
        })
    }

    pub async fn read_skill(
        &self,
        params: SkillReadParams,
    ) -> Result<SkillReadResponse, RuntimeCoreError> {
        self.app_data_source.read_skill(params).await
    }

    pub async fn list_management_skills(
        &self,
        params: SkillManagementListParams,
    ) -> Result<SkillManagementListResponse, RuntimeCoreError> {
        self.app_data_source.list_management_skills(params).await
    }

    pub async fn install_management_skill(
        &self,
        params: SkillManagementInstallParams,
    ) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        self.app_data_source.install_management_skill(params).await
    }

    pub async fn uninstall_management_skill(
        &self,
        params: SkillManagementUninstallParams,
    ) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        self.app_data_source
            .uninstall_management_skill(params)
            .await
    }

    pub async fn list_skill_repositories(
        &self,
    ) -> Result<SkillRepositoryListResponse, RuntimeCoreError> {
        self.app_data_source.list_skill_repositories().await
    }

    pub async fn save_skill_repository(
        &self,
        params: SkillRepositorySaveParams,
    ) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        self.app_data_source.save_skill_repository(params).await
    }

    pub async fn delete_skill_repository(
        &self,
        params: SkillRepositoryDeleteParams,
    ) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        self.app_data_source.delete_skill_repository(params).await
    }

    pub async fn refresh_skill_cache(
        &self,
    ) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        self.app_data_source.refresh_skill_cache().await
    }

    pub async fn list_installed_skill_directories(
        &self,
    ) -> Result<SkillInstalledDirectoriesListResponse, RuntimeCoreError> {
        self.app_data_source
            .list_installed_skill_directories()
            .await
    }

    pub async fn inspect_local_skill(
        &self,
        params: SkillLocalInspectParams,
    ) -> Result<SkillLocalInspectResponse, RuntimeCoreError> {
        self.app_data_source.inspect_local_skill(params).await
    }

    pub async fn inspect_local_skill_detail(
        &self,
        params: SkillLocalDetailInspectParams,
    ) -> Result<SkillLocalDetailInspectResponse, RuntimeCoreError> {
        self.app_data_source
            .inspect_local_skill_detail(params)
            .await
    }

    pub async fn create_skill_scaffold(
        &self,
        params: SkillScaffoldCreateParams,
    ) -> Result<SkillScaffoldCreateResponse, RuntimeCoreError> {
        self.app_data_source.create_skill_scaffold(params).await
    }

    pub async fn import_local_skill(
        &self,
        params: SkillLocalImportParams,
    ) -> Result<SkillLocalImportResponse, RuntimeCoreError> {
        self.app_data_source.import_local_skill(params).await
    }

    pub async fn rename_local_skill(
        &self,
        params: SkillLocalRenameParams,
    ) -> Result<SkillLocalRenameResponse, RuntimeCoreError> {
        self.app_data_source.rename_local_skill(params).await
    }

    pub async fn inspect_remote_skill(
        &self,
        params: SkillRemoteInspectParams,
    ) -> Result<SkillRemoteInspectResponse, RuntimeCoreError> {
        self.app_data_source.inspect_remote_skill(params).await
    }

    pub async fn inspect_local_skill_package(
        &self,
        params: SkillPackageLocalInspectParams,
    ) -> Result<SkillPackageLocalInspectResponse, RuntimeCoreError> {
        self.app_data_source
            .inspect_local_skill_package(params)
            .await
    }

    pub async fn install_local_skill_package(
        &self,
        params: SkillPackageLocalInstallParams,
    ) -> Result<SkillPackageLocalInstallResponse, RuntimeCoreError> {
        self.app_data_source
            .install_local_skill_package(params)
            .await
    }

    pub async fn replace_local_skill_package(
        &self,
        params: SkillPackageLocalReplaceParams,
    ) -> Result<SkillPackageLocalReplaceResponse, RuntimeCoreError> {
        self.app_data_source
            .replace_local_skill_package(params)
            .await
    }

    pub async fn export_local_skill_package(
        &self,
        params: SkillPackageExportParams,
    ) -> Result<SkillPackageExportResponse, RuntimeCoreError> {
        self.app_data_source
            .export_local_skill_package(params)
            .await
    }

    pub async fn install_marketplace_skill(
        &self,
        params: SkillMarketplaceInstallParams,
    ) -> Result<SkillMarketplaceInstallResponse, RuntimeCoreError> {
        self.app_data_source.install_marketplace_skill(params).await
    }

    pub async fn install_skill_from_download_url(
        &self,
        params: SkillDownloadInstallParams,
    ) -> Result<SkillDownloadInstallResponse, RuntimeCoreError> {
        self.app_data_source
            .install_skill_from_download_url(params)
            .await
    }

    pub async fn list_workspace_skill_bindings(
        &self,
        params: WorkspaceSkillBindingsListParams,
    ) -> Result<WorkspaceSkillBindingsListResponse, RuntimeCoreError> {
        self.app_data_source
            .list_workspace_skill_bindings(params)
            .await
    }

    pub async fn list_workspace_registered_skills(
        &self,
        params: WorkspaceRegisteredSkillsListParams,
    ) -> Result<WorkspaceRegisteredSkillsListResponse, RuntimeCoreError> {
        self.app_data_source
            .list_workspace_registered_skills(params)
            .await
    }
}

fn list_skills_for_cwd(cwd: PathBuf, config: &[SkillConfig]) -> SkillsListEntry {
    let roots = agent_skill_roots_for_workspace(Some(&cwd), Some(&cwd));
    let errors = roots
        .iter()
        .flat_map(|root| load_skill_summary_report_from_directory(&root.path).errors)
        .map(|error| SkillErrorInfo {
            path: error.path,
            message: error.message,
        })
        .collect();
    let mut snapshot = build_agent_skill_snapshot_from_roots(roots);
    apply_agent_skill_config(&mut snapshot, config);
    let mut seen = HashSet::new();
    let skills = snapshot
        .skills
        .iter()
        .filter(|skill| seen.insert(skill.stable_id().to_string()))
        .map(skill_metadata)
        .collect();

    SkillsListEntry {
        cwd,
        skills,
        errors,
    }
}

fn skill_config_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn load_skill_config() -> Result<Vec<SkillConfig>, RuntimeCoreError> {
    let _guard = skill_config_lock()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let path = ConfigManager::default_config_path();
    ConfigManager::load(&path)
        .map(|manager| manager.config().skills.config.clone())
        .map_err(|error| {
            RuntimeCoreError::Backend(format!("failed to load skill settings: {error}"))
        })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SkillConfigSelector {
    Path(PathBuf),
    Name(String),
}

impl SkillConfigSelector {
    fn from_params(params: &SkillsConfigWriteParams) -> Result<Self, RuntimeCoreError> {
        match (params.path.as_ref(), params.name.as_deref()) {
            (Some(path), None) if path.is_absolute() => Ok(Self::Path(normalize_path(path))),
            (None, Some(name)) if !name.trim().is_empty() => {
                Ok(Self::Name(name.trim().to_string()))
            }
            _ => Err(RuntimeCoreError::InvalidRequest(
                "skills/config/write requires exactly one of path or name".to_string(),
            )),
        }
    }

    fn matches(&self, entry: &SkillConfig) -> bool {
        match (self, entry.path.as_ref(), entry.name.as_deref()) {
            (Self::Path(expected), Some(path), None) => normalize_path(path) == *expected,
            (Self::Name(expected), None, Some(name)) => name.trim() == expected,
            _ => false,
        }
    }

    fn entry(&self) -> SkillConfig {
        match self {
            Self::Path(path) => SkillConfig {
                path: Some(path.clone()),
                name: None,
                enabled: false,
            },
            Self::Name(name) => SkillConfig {
                path: None,
                name: Some(name.clone()),
                enabled: false,
            },
        }
    }
}

fn update_skill_config(
    config: &mut lime_core::config::Config,
    selector: &SkillConfigSelector,
    enabled: bool,
) {
    if enabled {
        config
            .skills
            .config
            .retain(|entry| !selector.matches(entry));
        return;
    }

    let mut matched = false;
    config.skills.config.retain_mut(|entry| {
        if !selector.matches(entry) {
            return true;
        }
        if matched {
            return false;
        }
        *entry = selector.entry();
        matched = true;
        true
    });
    if !matched {
        config.skills.config.push(selector.entry());
    }
}

fn normalize_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn skill_metadata(skill: &AgentSkillMetadata) -> SkillMetadata {
    SkillMetadata {
        name: skill.name.clone(),
        description: skill.description.clone(),
        short_description: None,
        interface: Some(V2SkillInterface {
            display_name: Some(skill.interface.display_name.clone()),
            short_description: None,
            icon_small: None,
            icon_large: None,
            icon_small_url: None,
            icon_large_url: None,
            brand_color: None,
            default_prompt: None,
        }),
        dependencies: Some(V2SkillDependencies {
            tools: skill
                .dependencies
                .tools
                .iter()
                .map(|dependency| V2SkillToolDependency {
                    dependency_type: dependency.dependency_type.clone(),
                    value: dependency.value.clone(),
                    description: None,
                    transport: None,
                    command: None,
                    url: None,
                })
                .collect(),
        }),
        path: skill.skill_file_path.clone(),
        scope: skill_scope(skill.scope),
        enabled: skill.enabled,
    }
}

fn skill_scope(scope: AgentSkillScope) -> V2SkillScope {
    match scope {
        AgentSkillScope::Project => V2SkillScope::Repo,
        AgentSkillScope::User => V2SkillScope::User,
        AgentSkillScope::App => V2SkillScope::System,
        AgentSkillScope::Other => V2SkillScope::Admin,
    }
}

#[cfg(test)]
mod skills_list_tests {
    use super::*;

    #[test]
    fn agent_scopes_lower_to_codex_scopes() {
        assert_eq!(skill_scope(AgentSkillScope::Project), V2SkillScope::Repo);
        assert_eq!(skill_scope(AgentSkillScope::User), V2SkillScope::User);
        assert_eq!(skill_scope(AgentSkillScope::App), V2SkillScope::System);
        assert_eq!(skill_scope(AgentSkillScope::Other), V2SkillScope::Admin);
    }
}
