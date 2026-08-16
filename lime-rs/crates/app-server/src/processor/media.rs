//! media domain handlers for the App Server processor.

use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::{
    JsonRpcError, MediaTaskArtifactAudioCompleteParams, MediaTaskArtifactAudioCreateParams,
    MediaTaskArtifactImageCompleteParams, MediaTaskArtifactImageCreateParams,
    MediaTaskArtifactListParams, MediaTaskArtifactLookupParams,
    MediaTaskArtifactTranscriptionCreateParams, MediaTaskArtifactVideoCreateParams,
};

impl RequestProcessor {
    pub(super) async fn handle_media_task_artifact_image_create_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MediaTaskArtifactImageCreateParams = parse_params(params)?;
        let response = self
            .runtime
            .create_image_media_task_artifact(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_media_task_artifact_audio_create_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MediaTaskArtifactAudioCreateParams = parse_params(params)?;
        let response = self
            .runtime
            .create_audio_media_task_artifact(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_media_task_artifact_video_create_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MediaTaskArtifactVideoCreateParams = parse_params(params)?;
        let response = self
            .runtime
            .create_video_media_task_artifact(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_media_task_artifact_transcription_create_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MediaTaskArtifactTranscriptionCreateParams = parse_params(params)?;
        let response = self
            .runtime
            .create_transcription_media_task_artifact(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_media_task_artifact_audio_complete_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MediaTaskArtifactAudioCompleteParams = parse_params(params)?;
        let response = self
            .runtime
            .complete_audio_media_task_artifact(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_media_task_artifact_image_complete_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MediaTaskArtifactImageCompleteParams = parse_params(params)?;
        let response = self
            .runtime
            .complete_image_media_task_artifact(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_media_task_artifact_get_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MediaTaskArtifactLookupParams = parse_params(params)?;
        let response = self
            .runtime
            .get_media_task_artifact(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_media_task_artifact_list_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MediaTaskArtifactListParams = parse_params(params)?;
        let response = self
            .runtime
            .list_media_task_artifacts(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_media_task_artifact_cancel_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MediaTaskArtifactLookupParams = parse_params(params)?;
        let response = self
            .runtime
            .cancel_media_task_artifact(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }
}
