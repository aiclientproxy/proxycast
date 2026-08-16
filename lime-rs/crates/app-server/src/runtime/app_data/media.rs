use super::unavailable;
use super::NoopAppDataSource;
use super::RuntimeCoreError;
use crate::runtime::SidecarStore;
use app_server_protocol::*;
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
pub trait MediaAppDataSource: Send + Sync {
    async fn create_image_media_task_artifact(
        &self,
        _params: MediaTaskArtifactImageCreateParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        Err(unavailable("mediaTaskArtifact/image/create"))
    }

    async fn create_audio_media_task_artifact(
        &self,
        _params: MediaTaskArtifactAudioCreateParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        Err(unavailable("mediaTaskArtifact/audio/create"))
    }

    async fn create_transcription_media_task_artifact(
        &self,
        _params: MediaTaskArtifactTranscriptionCreateParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        Err(unavailable("mediaTaskArtifact/transcription/create"))
    }

    async fn create_video_media_task_artifact(
        &self,
        _params: MediaTaskArtifactVideoCreateParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        Err(unavailable("mediaTaskArtifact/video/create"))
    }

    async fn complete_audio_media_task_artifact(
        &self,
        _params: MediaTaskArtifactAudioCompleteParams,
        _sidecar_store: Option<Arc<SidecarStore>>,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        Err(unavailable("mediaTaskArtifact/audio/complete"))
    }

    async fn complete_image_media_task_artifact(
        &self,
        _params: MediaTaskArtifactImageCompleteParams,
        _sidecar_store: Option<Arc<SidecarStore>>,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        Err(unavailable("mediaTaskArtifact/image/complete"))
    }

    async fn get_media_task_artifact(
        &self,
        _params: MediaTaskArtifactLookupParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        Err(unavailable("mediaTaskArtifact/get"))
    }

    async fn list_media_task_artifacts(
        &self,
        _params: MediaTaskArtifactListParams,
    ) -> Result<MediaTaskArtifactListResponse, RuntimeCoreError> {
        Err(unavailable("mediaTaskArtifact/list"))
    }

    async fn cancel_media_task_artifact(
        &self,
        _params: MediaTaskArtifactLookupParams,
    ) -> Result<MediaTaskArtifactResponse, RuntimeCoreError> {
        Err(unavailable("mediaTaskArtifact/cancel"))
    }

    async fn get_gallery_material(
        &self,
        _params: GalleryMaterialLookupParams,
    ) -> Result<GalleryMaterialResponse, RuntimeCoreError> {
        Err(unavailable("galleryMaterial/get"))
    }

    async fn create_gallery_material_metadata(
        &self,
        _params: GalleryMaterialMetadataCreateParams,
    ) -> Result<GalleryMaterialMetadataResponse, RuntimeCoreError> {
        Err(unavailable("galleryMaterialMetadata/create"))
    }

    async fn get_gallery_material_metadata(
        &self,
        _params: GalleryMaterialLookupParams,
    ) -> Result<GalleryMaterialMetadataResponse, RuntimeCoreError> {
        Err(unavailable("galleryMaterialMetadata/get"))
    }

    async fn update_gallery_material_metadata(
        &self,
        _params: GalleryMaterialMetadataUpdateParams,
    ) -> Result<GalleryMaterialMetadataResponse, RuntimeCoreError> {
        Err(unavailable("galleryMaterialMetadata/update"))
    }

    async fn delete_gallery_material_metadata(
        &self,
        _params: GalleryMaterialLookupParams,
    ) -> Result<GalleryMaterialDeleteResponse, RuntimeCoreError> {
        Err(unavailable("galleryMaterialMetadata/delete"))
    }

    async fn list_gallery_materials_by_image_category(
        &self,
        _params: GalleryMaterialFilterParams,
    ) -> Result<GalleryMaterialListResponse, RuntimeCoreError> {
        Err(unavailable("galleryMaterial/listByImageCategory"))
    }

    async fn list_gallery_materials_by_layout_category(
        &self,
        _params: GalleryMaterialFilterParams,
    ) -> Result<GalleryMaterialListResponse, RuntimeCoreError> {
        Err(unavailable("galleryMaterial/listByLayoutCategory"))
    }

    async fn list_gallery_materials_by_mood(
        &self,
        _params: GalleryMaterialFilterParams,
    ) -> Result<GalleryMaterialListResponse, RuntimeCoreError> {
        Err(unavailable("galleryMaterial/listByMood"))
    }

    async fn list_project_materials(
        &self,
        _params: ProjectMaterialListParams,
    ) -> Result<ProjectMaterialListResponse, RuntimeCoreError> {
        Err(unavailable("projectMaterial/list"))
    }

    async fn get_project_material(
        &self,
        _params: ProjectMaterialLookupParams,
    ) -> Result<ProjectMaterialResponse, RuntimeCoreError> {
        Err(unavailable("projectMaterial/get"))
    }

    async fn count_project_materials(
        &self,
        _params: ProjectMaterialListParams,
    ) -> Result<ProjectMaterialCountResponse, RuntimeCoreError> {
        Err(unavailable("projectMaterial/count"))
    }

    async fn upload_project_material(
        &self,
        _params: ProjectMaterialUploadParams,
    ) -> Result<ProjectMaterialResponse, RuntimeCoreError> {
        Err(unavailable("projectMaterial/upload"))
    }

    async fn import_project_material_from_url(
        &self,
        _params: ProjectMaterialImportFromUrlParams,
    ) -> Result<ProjectMaterialResponse, RuntimeCoreError> {
        Err(unavailable("projectMaterial/importFromUrl"))
    }

    async fn update_project_material(
        &self,
        _params: ProjectMaterialUpdateParams,
    ) -> Result<ProjectMaterialResponse, RuntimeCoreError> {
        Err(unavailable("projectMaterial/update"))
    }

    async fn delete_project_material(
        &self,
        _params: ProjectMaterialLookupParams,
    ) -> Result<ProjectMaterialDeleteResponse, RuntimeCoreError> {
        Err(unavailable("projectMaterial/delete"))
    }

    async fn read_project_material_content(
        &self,
        _params: ProjectMaterialLookupParams,
    ) -> Result<ProjectMaterialContentResponse, RuntimeCoreError> {
        Err(unavailable("projectMaterial/content"))
    }
}

impl MediaAppDataSource for NoopAppDataSource {}
