use super::super::*;
use async_trait::async_trait;

#[async_trait]
impl SkillAppDataSource for LocalAppDataSource {
    async fn read_skill(
        &self,
        params: SkillReadParams,
    ) -> Result<SkillReadResponse, RuntimeCoreError> {
        skills::catalog::read_skill(params).map_err(data_error)
    }

    async fn list_management_skills(
        &self,
        params: SkillManagementListParams,
    ) -> Result<SkillManagementListResponse, RuntimeCoreError> {
        skills::management::list_management_skills(self.db.clone(), params)
            .await
            .map_err(data_error)
    }

    async fn install_management_skill(
        &self,
        params: SkillManagementInstallParams,
    ) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        skills::management::install_management_skill(self.db.clone(), params)
            .await
            .map_err(data_error)
    }

    async fn uninstall_management_skill(
        &self,
        params: SkillManagementUninstallParams,
    ) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        skills::management::uninstall_management_skill(self.db.clone(), params).map_err(data_error)
    }

    async fn list_skill_repositories(
        &self,
    ) -> Result<SkillRepositoryListResponse, RuntimeCoreError> {
        skills::management::list_skill_repositories(self.db.clone()).map_err(data_error)
    }

    async fn save_skill_repository(
        &self,
        params: SkillRepositorySaveParams,
    ) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        skills::management::save_skill_repository(self.db.clone(), params).map_err(data_error)
    }

    async fn delete_skill_repository(
        &self,
        params: SkillRepositoryDeleteParams,
    ) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        skills::management::delete_skill_repository(self.db.clone(), params).map_err(data_error)
    }

    async fn refresh_skill_cache(&self) -> Result<SkillManagementWriteResponse, RuntimeCoreError> {
        SkillService::new()
            .map_err(|error| data_error(error.to_string()))?
            .refresh_cache();
        Ok(SkillManagementWriteResponse { success: true })
    }

    async fn list_installed_skill_directories(
        &self,
    ) -> Result<SkillInstalledDirectoriesListResponse, RuntimeCoreError> {
        skills::local::list_installed_skill_directories().map_err(data_error)
    }

    async fn inspect_local_skill(
        &self,
        params: SkillLocalInspectParams,
    ) -> Result<SkillLocalInspectResponse, RuntimeCoreError> {
        skills::local::inspect_local_skill(params).map_err(data_error)
    }

    async fn inspect_local_skill_detail(
        &self,
        params: SkillLocalDetailInspectParams,
    ) -> Result<SkillLocalDetailInspectResponse, RuntimeCoreError> {
        skills::package::inspect_local_skill_detail(params).map_err(data_error)
    }

    async fn create_skill_scaffold(
        &self,
        params: SkillScaffoldCreateParams,
    ) -> Result<SkillScaffoldCreateResponse, RuntimeCoreError> {
        skills::local::create_skill_scaffold(params).map_err(data_error)
    }

    async fn import_local_skill(
        &self,
        params: SkillLocalImportParams,
    ) -> Result<SkillLocalImportResponse, RuntimeCoreError> {
        skills::local::import_local_skill(params).map_err(data_error)
    }

    async fn rename_local_skill(
        &self,
        params: SkillLocalRenameParams,
    ) -> Result<SkillLocalRenameResponse, RuntimeCoreError> {
        skills::local::rename_local_skill(params).map_err(data_error)
    }

    async fn inspect_remote_skill(
        &self,
        params: SkillRemoteInspectParams,
    ) -> Result<SkillRemoteInspectResponse, RuntimeCoreError> {
        skills::local::inspect_remote_skill(params)
            .await
            .map_err(data_error)
    }

    async fn inspect_local_skill_package(
        &self,
        params: SkillPackageLocalInspectParams,
    ) -> Result<SkillPackageLocalInspectResponse, RuntimeCoreError> {
        skills::package::inspect_local_skill_package(params).map_err(data_error)
    }

    async fn install_local_skill_package(
        &self,
        params: SkillPackageLocalInstallParams,
    ) -> Result<SkillPackageLocalInstallResponse, RuntimeCoreError> {
        skills::package::install_local_skill_package(params).map_err(data_error)
    }

    async fn replace_local_skill_package(
        &self,
        params: SkillPackageLocalReplaceParams,
    ) -> Result<SkillPackageLocalReplaceResponse, RuntimeCoreError> {
        skills::package::replace_local_skill_package(params).map_err(data_error)
    }

    async fn export_local_skill_package(
        &self,
        params: SkillPackageExportParams,
    ) -> Result<SkillPackageExportResponse, RuntimeCoreError> {
        skills::package::export_local_skill_package(params).map_err(data_error)
    }

    async fn install_marketplace_skill(
        &self,
        params: SkillMarketplaceInstallParams,
    ) -> Result<SkillMarketplaceInstallResponse, RuntimeCoreError> {
        skills::marketplace::install_marketplace_skill(params).map_err(data_error)
    }

    async fn install_skill_from_download_url(
        &self,
        params: SkillDownloadInstallParams,
    ) -> Result<SkillDownloadInstallResponse, RuntimeCoreError> {
        skills::package::install_skill_from_download_url(params)
            .await
            .map_err(data_error)
    }
}
