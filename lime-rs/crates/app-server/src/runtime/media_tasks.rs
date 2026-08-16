use super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::*;

impl RuntimeCore {
    pub async fn create_image_media_task_artifact(
        &self,
        params: MediaTaskArtifactImageCreateParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        self.app_data_source
            .create_image_media_task_artifact(params)
            .await
    }

    pub async fn create_audio_media_task_artifact(
        &self,
        params: MediaTaskArtifactAudioCreateParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        self.app_data_source
            .create_audio_media_task_artifact(params)
            .await
    }

    pub async fn create_transcription_media_task_artifact(
        &self,
        params: MediaTaskArtifactTranscriptionCreateParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        self.app_data_source
            .create_transcription_media_task_artifact(params)
            .await
    }

    pub async fn create_video_media_task_artifact(
        &self,
        params: MediaTaskArtifactVideoCreateParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        self.app_data_source
            .create_video_media_task_artifact(params)
            .await
    }

    pub async fn complete_audio_media_task_artifact(
        &self,
        params: MediaTaskArtifactAudioCompleteParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        self.app_data_source
            .complete_audio_media_task_artifact(params, self.sidecar_store.clone())
            .await
    }

    pub async fn complete_image_media_task_artifact(
        &self,
        params: MediaTaskArtifactImageCompleteParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        self.app_data_source
            .complete_image_media_task_artifact(params, self.sidecar_store.clone())
            .await
    }

    pub async fn get_media_task_artifact(
        &self,
        params: MediaTaskArtifactLookupParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        self.app_data_source.get_media_task_artifact(params).await
    }

    pub async fn list_media_task_artifacts(
        &self,
        params: MediaTaskArtifactListParams,
    ) -> Result<MediaTaskArtifactListResponse, RuntimeCoreError> {
        self.app_data_source.list_media_task_artifacts(params).await
    }

    pub async fn cancel_media_task_artifact(
        &self,
        params: MediaTaskArtifactLookupParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        self.app_data_source
            .cancel_media_task_artifact(params)
            .await
    }

    pub async fn get_gallery_material(
        &self,
        params: GalleryMaterialLookupParams,
    ) -> Result<GalleryMaterialResponse, RuntimeCoreError> {
        self.app_data_source.get_gallery_material(params).await
    }

    pub async fn create_gallery_material_metadata(
        &self,
        params: GalleryMaterialMetadataCreateParams,
    ) -> Result<GalleryMaterialMetadataResponse, RuntimeCoreError> {
        self.app_data_source
            .create_gallery_material_metadata(params)
            .await
    }

    pub async fn get_gallery_material_metadata(
        &self,
        params: GalleryMaterialLookupParams,
    ) -> Result<GalleryMaterialMetadataResponse, RuntimeCoreError> {
        self.app_data_source
            .get_gallery_material_metadata(params)
            .await
    }

    pub async fn update_gallery_material_metadata(
        &self,
        params: GalleryMaterialMetadataUpdateParams,
    ) -> Result<GalleryMaterialMetadataResponse, RuntimeCoreError> {
        self.app_data_source
            .update_gallery_material_metadata(params)
            .await
    }

    pub async fn delete_gallery_material_metadata(
        &self,
        params: GalleryMaterialLookupParams,
    ) -> Result<GalleryMaterialDeleteResponse, RuntimeCoreError> {
        self.app_data_source
            .delete_gallery_material_metadata(params)
            .await
    }

    pub async fn list_gallery_materials_by_image_category(
        &self,
        params: GalleryMaterialFilterParams,
    ) -> Result<GalleryMaterialListResponse, RuntimeCoreError> {
        self.app_data_source
            .list_gallery_materials_by_image_category(params)
            .await
    }

    pub async fn list_gallery_materials_by_layout_category(
        &self,
        params: GalleryMaterialFilterParams,
    ) -> Result<GalleryMaterialListResponse, RuntimeCoreError> {
        self.app_data_source
            .list_gallery_materials_by_layout_category(params)
            .await
    }

    pub async fn list_gallery_materials_by_mood(
        &self,
        params: GalleryMaterialFilterParams,
    ) -> Result<GalleryMaterialListResponse, RuntimeCoreError> {
        self.app_data_source
            .list_gallery_materials_by_mood(params)
            .await
    }

    pub async fn list_project_materials(
        &self,
        params: ProjectMaterialListParams,
    ) -> Result<ProjectMaterialListResponse, RuntimeCoreError> {
        self.app_data_source.list_project_materials(params).await
    }

    pub async fn get_project_material(
        &self,
        params: ProjectMaterialLookupParams,
    ) -> Result<ProjectMaterialResponse, RuntimeCoreError> {
        self.app_data_source.get_project_material(params).await
    }

    pub async fn count_project_materials(
        &self,
        params: ProjectMaterialListParams,
    ) -> Result<ProjectMaterialCountResponse, RuntimeCoreError> {
        self.app_data_source.count_project_materials(params).await
    }

    pub async fn upload_project_material(
        &self,
        params: ProjectMaterialUploadParams,
    ) -> Result<ProjectMaterialResponse, RuntimeCoreError> {
        self.app_data_source.upload_project_material(params).await
    }

    pub async fn import_project_material_from_url(
        &self,
        params: ProjectMaterialImportFromUrlParams,
    ) -> Result<ProjectMaterialResponse, RuntimeCoreError> {
        self.app_data_source
            .import_project_material_from_url(params)
            .await
    }

    pub async fn update_project_material(
        &self,
        params: ProjectMaterialUpdateParams,
    ) -> Result<ProjectMaterialResponse, RuntimeCoreError> {
        self.app_data_source.update_project_material(params).await
    }

    pub async fn delete_project_material(
        &self,
        params: ProjectMaterialLookupParams,
    ) -> Result<ProjectMaterialDeleteResponse, RuntimeCoreError> {
        self.app_data_source.delete_project_material(params).await
    }

    pub async fn read_project_material_content(
        &self,
        params: ProjectMaterialLookupParams,
    ) -> Result<ProjectMaterialContentResponse, RuntimeCoreError> {
        self.app_data_source
            .read_project_material_content(params)
            .await
    }
}
