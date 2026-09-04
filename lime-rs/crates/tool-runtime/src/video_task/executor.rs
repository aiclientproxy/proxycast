use super::definition::VIDEO_TASK_TOOL_NAME;
use super::params::{build_create_params, parse_video_task_input, runtime_video_task_error};
use crate::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionFuture, RuntimeToolExecutionRequest,
    RuntimeToolExecutionResult, RuntimeToolExecutor, RuntimeToolExecutorHandle,
};
use app_server_protocol::{MediaTaskArtifactResponse, MediaTaskArtifactVideoCreateParams};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

#[async_trait]
pub trait VideoTaskGateway: Send + Sync {
    async fn create_video_media_task_artifact(
        &self,
        params: MediaTaskArtifactVideoCreateParams,
    ) -> Result<MediaTaskArtifactResponse, String>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct VideoTaskToolResultProjection {
    pub output: Option<String>,
    pub metadata: HashMap<String, Value>,
}

pub struct RuntimeVideoTaskExecutor {
    gateway: Arc<dyn VideoTaskGateway>,
}

impl RuntimeVideoTaskExecutor {
    pub fn new(gateway: Arc<dyn VideoTaskGateway>) -> Self {
        Self { gateway }
    }

    pub fn handle(gateway: Arc<dyn VideoTaskGateway>) -> RuntimeToolExecutorHandle {
        RuntimeToolExecutorHandle::new(Arc::new(Self::new(gateway)))
    }

    async fn execute_video_task(
        &self,
        request: RuntimeToolExecutionRequest<'_>,
    ) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
        if request.tool_name != VIDEO_TASK_TOOL_NAME {
            return Err(runtime_video_task_error(format!(
                "video task executor cannot run tool '{}'",
                request.tool_name
            )));
        }
        if request
            .context
            .cancel_token()
            .is_some_and(|token| token.is_cancelled())
        {
            return Err(runtime_video_task_error("video task cancelled"));
        }

        let input = parse_video_task_input(
            request.params,
            request.context.working_directory(),
            request.context.session_id(),
            request.turn_context,
        )?;
        let response = self
            .gateway
            .create_video_media_task_artifact(build_create_params(input))
            .await
            .map_err(runtime_video_task_error)?;
        let projection = video_task_tool_result_projection(response);
        Ok(RuntimeToolExecutionResult::new(
            true,
            projection.output.unwrap_or_default(),
            None,
            projection.metadata,
        ))
    }
}

impl RuntimeToolExecutor for RuntimeVideoTaskExecutor {
    fn execute<'a>(
        &'a self,
        request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move { self.execute_video_task(request).await })
    }
}

pub fn runtime_video_task_executor_handle(
    gateway: Arc<dyn VideoTaskGateway>,
) -> RuntimeToolExecutorHandle {
    RuntimeVideoTaskExecutor::handle(gateway)
}

pub fn video_task_tool_result_projection(
    response: MediaTaskArtifactResponse,
) -> VideoTaskToolResultProjection {
    let text = serde_json::to_string_pretty(&response).unwrap_or_else(|_| "{}".to_string());
    let mut metadata = HashMap::from([
        ("task_id".to_string(), json!(response.task_id)),
        ("task_type".to_string(), json!(response.task_type)),
        ("task_family".to_string(), json!(response.task_family)),
        ("status".to_string(), json!(response.status)),
        (
            "normalized_status".to_string(),
            json!(response.normalized_status),
        ),
        ("path".to_string(), json!(response.path)),
        ("artifact_path".to_string(), json!(response.artifact_path)),
        (
            "reused_existing".to_string(),
            json!(response.reused_existing),
        ),
        ("record".to_string(), json!(response.record)),
    ]);
    if let Some(idempotency_key) = response.idempotency_key {
        metadata.insert("idempotency_key".to_string(), json!(idempotency_key));
    }
    VideoTaskToolResultProjection {
        output: Some(text),
        metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_executor::{RuntimeToolExecutionContext, RuntimeToolExecutionContextInput};
    use agent_protocol::turn_context::TurnContextOverride;
    use serde_json::json;
    use tempfile::TempDir;
    use tokio::sync::Mutex;

    #[derive(Default)]
    struct VideoToolTestGateway {
        last_params: Mutex<Option<MediaTaskArtifactVideoCreateParams>>,
    }

    #[async_trait]
    impl VideoTaskGateway for VideoToolTestGateway {
        async fn create_video_media_task_artifact(
            &self,
            params: MediaTaskArtifactVideoCreateParams,
        ) -> Result<MediaTaskArtifactResponse, String> {
            *self.last_params.lock().await = Some(params.clone());
            Ok(MediaTaskArtifactResponse {
                success: true,
                task_id: "task-video-1".to_string(),
                task_type: "video_generate".to_string(),
                task_family: "video".to_string(),
                status: "pending_submit".to_string(),
                normalized_status: "pending".to_string(),
                artifact_path: ".lime/tasks/video_generate/task-video-1.json".to_string(),
                record: json!({
                    "task_type": "video_generate",
                    "payload": {
                        "provider_id": params.provider_id,
                        "model": params.model,
                        "duration": params.duration,
                        "generate_audio": params.generate_audio
                    }
                }),
                ..MediaTaskArtifactResponse::default()
            })
        }
    }

    #[tokio::test]
    async fn video_tool_builds_standard_video_task_request() {
        let workspace = TempDir::new().expect("workspace");
        let gateway = Arc::new(VideoToolTestGateway::default());
        let executor = RuntimeVideoTaskExecutor::new(gateway.clone());
        let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: workspace.path().to_path_buf(),
            session_id: "session-video-1".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        });
        let turn_context = TurnContextOverride {
            metadata: HashMap::from([
                ("thread_id".to_string(), json!("thread-video-1")),
                ("turn_id".to_string(), json!("turn-video-1")),
            ]),
            ..TurnContextOverride::default()
        };
        let result = executor
            .execute_video_task(RuntimeToolExecutionRequest {
                tool_name: VIDEO_TASK_TOOL_NAME,
                params: &json!({
                    "prompt": "生成一段产品演示视频",
                    "provider_id": "xai",
                    "model": "grok-imagine-video",
                    "duration": 8,
                    "generate_audio": true
                }),
                context: &context,
                turn_context: Some(&turn_context),
            })
            .await
            .expect("video tool should call gateway");

        assert_eq!(
            result.metadata.get("task_type"),
            Some(&json!("video_generate"))
        );
        assert_eq!(result.metadata.get("task_family"), Some(&json!("video")));
        assert_eq!(
            result.metadata.get("normalized_status"),
            Some(&json!("pending"))
        );

        let params = gateway
            .last_params
            .lock()
            .await
            .clone()
            .expect("gateway params");
        assert_eq!(params.project_root_path, workspace.path().to_string_lossy());
        assert_eq!(params.session_id.as_deref(), Some("session-video-1"));
        assert_eq!(params.thread_id.as_deref(), Some("thread-video-1"));
        assert_eq!(params.turn_id.as_deref(), Some("turn-video-1"));
        assert_eq!(params.duration, Some(8));
        assert_eq!(params.generate_audio, Some(true));
    }
}
