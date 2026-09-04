use crate::AppDataSource;
use app_server_protocol::{MediaTaskArtifactResponse, MediaTaskArtifactVideoCreateParams};
use async_trait::async_trait;
use std::sync::Arc;
use tool_runtime::video_task::VideoTaskGateway;

pub(crate) fn video_task_gateway(
    app_data_source: Arc<dyn AppDataSource>,
) -> Arc<dyn VideoTaskGateway> {
    Arc::new(AppServerVideoTaskGateway { app_data_source })
}

struct AppServerVideoTaskGateway {
    app_data_source: Arc<dyn AppDataSource>,
}

#[async_trait]
impl VideoTaskGateway for AppServerVideoTaskGateway {
    async fn create_video_media_task_artifact(
        &self,
        params: MediaTaskArtifactVideoCreateParams,
    ) -> Result<MediaTaskArtifactResponse, String> {
        self.app_data_source
            .create_video_media_task_artifact(params)
            .await
            .map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoopAppDataSource;

    #[tokio::test]
    async fn video_gateway_delegates_to_media_data_source() {
        let workspace = tempfile::tempdir().expect("workspace");
        let error = video_task_gateway(Arc::new(NoopAppDataSource))
            .create_video_media_task_artifact(MediaTaskArtifactVideoCreateParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                prompt: "生成视频".to_string(),
                ..MediaTaskArtifactVideoCreateParams::default()
            })
            .await
            .expect_err("noop media data source should fail closed");

        assert!(error.contains("mediaTaskArtifact/video/create"));
    }
}
