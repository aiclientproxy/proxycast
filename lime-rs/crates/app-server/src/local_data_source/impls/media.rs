use super::super::*;
use async_trait::async_trait;

#[async_trait]
impl MediaAppDataSource for LocalAppDataSource {
    async fn create_image_media_task_artifact(
        &self,
        params: MediaTaskArtifactImageCreateParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        let normalized = media_tasks::normalize_image_create_params_for_task_submission(params)
            .map_err(data_error)?;
        let params = media_tasks::resolve_image_provider_for_task_submission(
            &self.db,
            &self.api_key_provider_service,
            normalized,
        )
        .map_err(data_error)?;
        let route_assessment = media_tasks::assess_image_route(
            &self.db,
            &self.api_key_provider_service,
            &self.model_registry_service,
            &params,
        )
        .await
        .map_err(data_error)?;
        let response = media_tasks::create_image_media_task_artifact(params, route_assessment)
            .map_err(data_error)?;
        let _ = crate::media_task_worker::spawn_image_task_worker_for_created_task(
            &response,
            crate::media_task_worker::MediaTaskWorkerContext::new(self.db.clone())
                .with_sidecar_store(self.sidecar_store.clone()),
        );
        Ok(response)
    }

    async fn create_audio_media_task_artifact(
        &self,
        params: MediaTaskArtifactAudioCreateParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        let route_assessment = media_tasks::assess_audio_route(
            &self.db,
            &self.api_key_provider_service,
            &self.model_registry_service,
            &params,
        )
        .await
        .map_err(data_error)?;
        let response = media_tasks::create_audio_media_task_artifact(params, route_assessment)
            .map_err(data_error)?;
        let _ = crate::media_task_worker::spawn_audio_task_worker_for_created_task(
            &response,
            crate::media_task_worker::MediaTaskWorkerContext::new(self.db.clone())
                .with_sidecar_store(self.sidecar_store.clone()),
        );
        Ok(response)
    }

    async fn create_transcription_media_task_artifact(
        &self,
        params: MediaTaskArtifactTranscriptionCreateParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        let route_assessment = media_tasks::assess_transcription_route(
            &self.db,
            &self.api_key_provider_service,
            &self.model_registry_service,
            &params,
        )
        .await
        .map_err(data_error)?;
        let response =
            media_tasks::create_transcription_media_task_artifact(params, route_assessment)
                .map_err(data_error)?;
        let _ = crate::media_task_worker::spawn_transcription_task_worker_for_created_task(
            &response,
            crate::media_task_worker::MediaTaskWorkerContext::new(self.db.clone()),
        );
        Ok(response)
    }

    async fn create_video_media_task_artifact(
        &self,
        params: MediaTaskArtifactVideoCreateParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        let route_assessment = media_tasks::assess_video_route(
            &self.db,
            &self.api_key_provider_service,
            &self.model_registry_service,
            &params,
        )
        .await
        .map_err(data_error)?;
        let response = media_tasks::create_video_media_task_artifact(params, route_assessment)
            .map_err(data_error)?;
        let _ = crate::media_task_worker::spawn_video_task_worker_for_created_task(
            &response,
            crate::media_task_worker::MediaTaskWorkerContext::new(self.db.clone()),
        );
        Ok(response)
    }

    async fn complete_audio_media_task_artifact(
        &self,
        params: MediaTaskArtifactAudioCompleteParams,
        sidecar_store: Option<std::sync::Arc<crate::runtime::SidecarStore>>,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        media_tasks::complete_audio_media_task_artifact(params, sidecar_store.as_deref())
            .map_err(data_error)
    }

    async fn complete_image_media_task_artifact(
        &self,
        params: MediaTaskArtifactImageCompleteParams,
        sidecar_store: Option<std::sync::Arc<crate::runtime::SidecarStore>>,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        media_tasks::complete_image_media_task_artifact(params, sidecar_store.as_deref())
            .await
            .map_err(data_error)
    }

    async fn get_media_task_artifact(
        &self,
        params: MediaTaskArtifactLookupParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        media_tasks::get_media_task_artifact(params).map_err(data_error)
    }

    async fn list_media_task_artifacts(
        &self,
        params: MediaTaskArtifactListParams,
    ) -> Result<MediaTaskArtifactListResponse, RuntimeCoreError> {
        media_tasks::list_media_task_artifacts(params).map_err(data_error)
    }

    async fn cancel_media_task_artifact(
        &self,
        params: MediaTaskArtifactLookupParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        media_tasks::cancel_media_task_artifact(params).map_err(data_error)
    }

    async fn get_gallery_material(
        &self,
        params: GalleryMaterialLookupParams,
    ) -> Result<GalleryMaterialResponse, RuntimeCoreError> {
        gallery_materials::get_gallery_material(&self.db, params).map_err(data_error)
    }

    async fn create_gallery_material_metadata(
        &self,
        params: GalleryMaterialMetadataCreateParams,
    ) -> Result<GalleryMaterialMetadataResponse, RuntimeCoreError> {
        gallery_materials::create_gallery_material_metadata(&self.db, params).map_err(data_error)
    }

    async fn get_gallery_material_metadata(
        &self,
        params: GalleryMaterialLookupParams,
    ) -> Result<GalleryMaterialMetadataResponse, RuntimeCoreError> {
        gallery_materials::get_gallery_material_metadata(&self.db, params).map_err(data_error)
    }

    async fn update_gallery_material_metadata(
        &self,
        params: GalleryMaterialMetadataUpdateParams,
    ) -> Result<GalleryMaterialMetadataResponse, RuntimeCoreError> {
        gallery_materials::update_gallery_material_metadata(&self.db, params).map_err(data_error)
    }

    async fn delete_gallery_material_metadata(
        &self,
        params: GalleryMaterialLookupParams,
    ) -> Result<GalleryMaterialDeleteResponse, RuntimeCoreError> {
        gallery_materials::delete_gallery_material_metadata(&self.db, params).map_err(data_error)
    }

    async fn list_gallery_materials_by_image_category(
        &self,
        params: GalleryMaterialFilterParams,
    ) -> Result<GalleryMaterialListResponse, RuntimeCoreError> {
        gallery_materials::list_gallery_materials_by_image_category(&self.db, params)
            .map_err(data_error)
    }

    async fn list_gallery_materials_by_layout_category(
        &self,
        params: GalleryMaterialFilterParams,
    ) -> Result<GalleryMaterialListResponse, RuntimeCoreError> {
        gallery_materials::list_gallery_materials_by_layout_category(&self.db, params)
            .map_err(data_error)
    }

    async fn list_gallery_materials_by_mood(
        &self,
        params: GalleryMaterialFilterParams,
    ) -> Result<GalleryMaterialListResponse, RuntimeCoreError> {
        gallery_materials::list_gallery_materials_by_mood(&self.db, params).map_err(data_error)
    }

    async fn list_project_materials(
        &self,
        params: ProjectMaterialListParams,
    ) -> Result<ProjectMaterialListResponse, RuntimeCoreError> {
        project_materials::list_project_materials(&self.db, params).map_err(data_error)
    }

    async fn get_project_material(
        &self,
        params: ProjectMaterialLookupParams,
    ) -> Result<ProjectMaterialResponse, RuntimeCoreError> {
        project_materials::get_project_material(&self.db, params).map_err(data_error)
    }

    async fn count_project_materials(
        &self,
        params: ProjectMaterialListParams,
    ) -> Result<ProjectMaterialCountResponse, RuntimeCoreError> {
        project_materials::count_project_materials(&self.db, params).map_err(data_error)
    }

    async fn upload_project_material(
        &self,
        params: ProjectMaterialUploadParams,
    ) -> Result<ProjectMaterialResponse, RuntimeCoreError> {
        project_materials::upload_project_material(&self.db, params).map_err(data_error)
    }

    async fn import_project_material_from_url(
        &self,
        params: ProjectMaterialImportFromUrlParams,
    ) -> Result<ProjectMaterialResponse, RuntimeCoreError> {
        project_materials::import_project_material_from_url(&self.db, params)
            .await
            .map_err(data_error)
    }

    async fn update_project_material(
        &self,
        params: ProjectMaterialUpdateParams,
    ) -> Result<ProjectMaterialResponse, RuntimeCoreError> {
        project_materials::update_project_material(&self.db, params).map_err(data_error)
    }

    async fn delete_project_material(
        &self,
        params: ProjectMaterialLookupParams,
    ) -> Result<ProjectMaterialDeleteResponse, RuntimeCoreError> {
        project_materials::delete_project_material(&self.db, params).map_err(data_error)
    }

    async fn read_project_material_content(
        &self,
        params: ProjectMaterialLookupParams,
    ) -> Result<ProjectMaterialContentResponse, RuntimeCoreError> {
        project_materials::read_project_material_content(&self.db, params).map_err(data_error)
    }
}
