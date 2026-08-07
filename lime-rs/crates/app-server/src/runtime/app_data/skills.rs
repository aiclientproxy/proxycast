use super::unavailable;
use super::NoopAppDataSource;
use super::RuntimeCoreError;
use app_server_protocol::*;
use async_trait::async_trait;
use serde_json::json;

#[async_trait]
pub trait SkillAppDataSource: Send + Sync {
    async fn read_skill(
        &self,
        _params: SkillReadParams,
    ) -> Result<SkillReadResponse, RuntimeCoreError> {
        Err(RuntimeCoreError::Backend("skill not found".to_string()))
    }

    async fn list_management_skills(
        &self,
        _params: SkillManagementListParams,
    ) -> Result<SkillManagementListResponse, RuntimeCoreError> {
        Err(unavailable("skillManagement/list"))
    }

    async fn install_management_skill(
        &self,
        _params: SkillManagementInstallParams,
    ) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        Err(unavailable("skillManagement/install"))
    }

    async fn uninstall_management_skill(
        &self,
        _params: SkillManagementUninstallParams,
    ) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        Err(unavailable("skillManagement/uninstall"))
    }

    async fn list_skill_repositories(
        &self,
    ) -> Result<SkillRepositoryListResponse, RuntimeCoreError> {
        Err(unavailable("skillRepository/list"))
    }

    async fn save_skill_repository(
        &self,
        _params: SkillRepositorySaveParams,
    ) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        Err(unavailable("skillRepository/save"))
    }

    async fn delete_skill_repository(
        &self,
        _params: SkillRepositoryDeleteParams,
    ) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        Err(unavailable("skillRepository/delete"))
    }

    async fn refresh_skill_cache(&self) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        Err(unavailable("skillCache/refresh"))
    }

    async fn list_installed_skill_directories(
        &self,
    ) -> Result<SkillInstalledDirectoriesListResponse, RuntimeCoreError> {
        Err(unavailable("skillInstalledDirectories/list"))
    }

    async fn inspect_local_skill(
        &self,
        _params: SkillLocalInspectParams,
    ) -> Result<SkillLocalInspectResponse, RuntimeCoreError> {
        Err(unavailable("skillLocal/inspect"))
    }

    async fn inspect_local_skill_detail(
        &self,
        _params: SkillLocalDetailInspectParams,
    ) -> Result<SkillLocalDetailInspectResponse, RuntimeCoreError> {
        Err(unavailable("skillLocal/detail/inspect"))
    }

    async fn create_skill_scaffold(
        &self,
        _params: SkillScaffoldCreateParams,
    ) -> Result<SkillScaffoldCreateResponse, RuntimeCoreError> {
        Err(unavailable("skillLocal/scaffold/create"))
    }

    async fn import_local_skill(
        &self,
        _params: SkillLocalImportParams,
    ) -> Result<SkillLocalImportResponse, RuntimeCoreError> {
        Err(unavailable("skillLocal/import"))
    }

    async fn rename_local_skill(
        &self,
        _params: SkillLocalRenameParams,
    ) -> Result<SkillLocalRenameResponse, RuntimeCoreError> {
        Err(unavailable("skillLocal/rename"))
    }

    async fn inspect_remote_skill(
        &self,
        _params: SkillRemoteInspectParams,
    ) -> Result<SkillRemoteInspectResponse, RuntimeCoreError> {
        Err(unavailable("skillRemote/inspect"))
    }

    async fn inspect_local_skill_package(
        &self,
        _params: SkillPackageLocalInspectParams,
    ) -> Result<SkillPackageLocalInspectResponse, RuntimeCoreError> {
        Err(unavailable("skillPackage/local/inspect"))
    }

    async fn install_local_skill_package(
        &self,
        _params: SkillPackageLocalInstallParams,
    ) -> Result<SkillPackageLocalInstallResponse, RuntimeCoreError> {
        Err(unavailable("skillPackage/local/install"))
    }

    async fn replace_local_skill_package(
        &self,
        _params: SkillPackageLocalReplaceParams,
    ) -> Result<SkillPackageLocalReplaceResponse, RuntimeCoreError> {
        Err(unavailable("skillPackage/local/replace"))
    }

    async fn export_local_skill_package(
        &self,
        _params: SkillPackageExportParams,
    ) -> Result<SkillPackageExportResponse, RuntimeCoreError> {
        Err(unavailable("skillPackage/export"))
    }

    async fn install_marketplace_skill(
        &self,
        _params: SkillMarketplaceInstallParams,
    ) -> Result<SkillMarketplaceInstallResponse, RuntimeCoreError> {
        Err(unavailable("skillMarketplace/install"))
    }

    async fn install_skill_from_download_url(
        &self,
        _params: SkillDownloadInstallParams,
    ) -> Result<SkillDownloadInstallResponse, RuntimeCoreError> {
        Err(unavailable("skillPackage/download/install"))
    }
}

#[async_trait]
pub trait WorkspaceSkillBindingAppDataSource: Send + Sync {
    async fn list_workspace_skill_bindings(
        &self,
        _params: WorkspaceSkillBindingsListParams,
    ) -> Result<WorkspaceSkillBindingsListResponse, RuntimeCoreError> {
        Ok(WorkspaceSkillBindingsListResponse {
            bindings: json!({
                "request": {
                    "workspace_root": "",
                    "caller": "assistant",
                    "surface": {
                        "workbench": false,
                        "browser_assist": false
                    }
                },
                "warnings": [],
                "counts": {
                    "registered_total": 0,
                    "ready_for_manual_enable_total": 0,
                    "blocked_total": 0,
                    "query_loop_visible_total": 0,
                    "tool_runtime_visible_total": 0,
                    "launch_enabled_total": 0
                },
                "bindings": []
            }),
        })
    }

    async fn list_workspace_registered_skills(
        &self,
        _params: WorkspaceRegisteredSkillsListParams,
    ) -> Result<WorkspaceRegisteredSkillsListResponse, RuntimeCoreError> {
        Ok(WorkspaceRegisteredSkillsListResponse::default())
    }
}

impl SkillAppDataSource for NoopAppDataSource {}
impl WorkspaceSkillBindingAppDataSource for NoopAppDataSource {}
