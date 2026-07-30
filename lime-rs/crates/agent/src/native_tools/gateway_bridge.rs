use crate::native_tools::NativeRegistration;
use std::sync::Arc;
use tool_runtime::current_time::CurrentTimeGateway;
use tool_runtime::gateway_dispatch_execution::RuntimeGatewayToolExecutionRegistration;
use tool_runtime::image_task::ImageTaskGateway;
use tool_runtime::mcp_resource::McpResourceGateway;
use tool_runtime::memory_store::MemoryStoreGateway;
use tool_runtime::native_dispatch::NativeDispatch;
use tool_runtime::tool_executor::RuntimeToolExecutorHandle;
use tool_runtime::tool_search::ToolSearchGateway;

pub(crate) fn create_memory_tools(gateway: Arc<dyn MemoryStoreGateway>) -> Vec<NativeRegistration> {
    create_gateway_tools(
        NativeDispatch::builder()
            .with_memory_store_gateway(gateway)
            .build(),
    )
}

pub(crate) fn create_current_time_tools(
    gateway: Arc<dyn CurrentTimeGateway>,
) -> Vec<NativeRegistration> {
    create_gateway_tools(
        NativeDispatch::builder()
            .with_current_time_gateway(gateway)
            .build(),
    )
}

pub(crate) fn create_image_tools(gateway: Arc<dyn ImageTaskGateway>) -> Vec<NativeRegistration> {
    create_gateway_tools(
        NativeDispatch::builder()
            .with_image_task_gateway(gateway)
            .build(),
    )
}

pub(crate) fn create_tool_search_tools(
    gateway: Arc<dyn ToolSearchGateway>,
) -> Vec<NativeRegistration> {
    create_gateway_tools(
        NativeDispatch::builder()
            .with_tool_search_gateway(gateway)
            .build(),
    )
}

pub(crate) fn create_mcp_resource_tools(
    gateway: Arc<dyn McpResourceGateway>,
) -> Vec<NativeRegistration> {
    create_gateway_tools(
        NativeDispatch::builder()
            .with_mcp_resource_gateway(gateway)
            .build(),
    )
}

fn create_gateway_tools(dispatch: NativeDispatch) -> Vec<NativeRegistration> {
    let surfaces = dispatch.surfaces();
    let executor = RuntimeToolExecutorHandle::new(Arc::new(dispatch));

    surfaces
        .into_iter()
        .map(|surface| {
            let execution_registration = RuntimeGatewayToolExecutionRegistration::new(
                surface.definition(),
                executor.clone(),
                surface.aliases(),
            );
            NativeRegistration::gateway(execution_registration)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::{
        McpToolListResponse, McpToolSearchParams, MediaTaskArtifactImageCreateParams,
        MediaTaskArtifactResponse, MemoryStoreAddNoteParams, MemoryStoreAddNoteResponse,
        MemoryStoreCitation, MemoryStoreListParams, MemoryStoreListResponse, MemoryStoreReadParams,
        MemoryStoreReadResponse, MemoryStoreSearchParams, MemoryStoreSearchResponse,
    };
    use async_trait::async_trait;
    use serde_json::json;
    use tempfile::{tempdir, TempDir};
    use tokio::sync::Mutex;
    use tool_runtime::gateway_dispatch_execution::{
        execute_runtime_gateway_dispatch_tool, RuntimeGatewayDispatchToolRequest,
        RuntimeGatewayToolExecutionRegistry,
    };

    fn install_gateway_registrations(
        registrations: Vec<NativeRegistration>,
    ) -> RuntimeGatewayToolExecutionRegistry {
        let registry = RuntimeGatewayToolExecutionRegistry::default();
        for registration in registrations {
            registry.register(registration.into_gateway_execution());
        }
        registry
    }

    #[derive(Default)]
    struct FakeMemoryStoreGateway;

    #[async_trait]
    impl MemoryStoreGateway for FakeMemoryStoreGateway {
        async fn list_memory_store(
            &self,
            params: MemoryStoreListParams,
        ) -> Result<MemoryStoreListResponse, String> {
            Ok(MemoryStoreListResponse {
                root_scope: params.root.scope,
                path: params.path.unwrap_or_default(),
                entries: Vec::new(),
                truncated: false,
                next_cursor: None,
            })
        }

        async fn read_memory_store(
            &self,
            params: MemoryStoreReadParams,
        ) -> Result<MemoryStoreReadResponse, String> {
            Ok(MemoryStoreReadResponse {
                path: params.path,
                start_line_number: 1,
                content: "remember this".to_string(),
                truncated: false,
                citation: MemoryStoreCitation {
                    path: "MEMORY.md".to_string(),
                    start_line_number: 1,
                    end_line_number: 1,
                },
            })
        }

        async fn search_memory_store(
            &self,
            _params: MemoryStoreSearchParams,
        ) -> Result<MemoryStoreSearchResponse, String> {
            Ok(MemoryStoreSearchResponse {
                hits: Vec::new(),
                truncated: false,
                next_cursor: None,
            })
        }

        async fn add_memory_store_note(
            &self,
            _params: MemoryStoreAddNoteParams,
        ) -> Result<MemoryStoreAddNoteResponse, String> {
            Ok(MemoryStoreAddNoteResponse {
                path: "extensions/ad_hoc/notes/note.md".to_string(),
                citation: MemoryStoreCitation {
                    path: "extensions/ad_hoc/notes/note.md".to_string(),
                    start_line_number: 1,
                    end_line_number: 1,
                },
            })
        }
    }

    #[derive(Default)]
    struct ImageToolTestGateway {
        last_params: Mutex<Option<MediaTaskArtifactImageCreateParams>>,
    }

    #[async_trait]
    impl ImageTaskGateway for ImageToolTestGateway {
        async fn create_image_media_task_artifact(
            &self,
            params: MediaTaskArtifactImageCreateParams,
        ) -> Result<MediaTaskArtifactResponse, String> {
            *self.last_params.lock().await = Some(params.clone());
            Ok(MediaTaskArtifactResponse {
                success: true,
                task_id: "task-image-1".to_string(),
                task_type: "image_generate".to_string(),
                task_family: "image".to_string(),
                status: "pending".to_string(),
                normalized_status: "pending".to_string(),
                artifact_path: ".lime/tasks/image_generate/task-image-1.json".to_string(),
                record: json!({
                    "task_type": "image_generate",
                    "payload": {
                        "provider_id": params.provider_id,
                        "model": params.model,
                        "executor_mode": params.executor_mode,
                        "entry_source": params.entry_source,
                        "modality_contract_key": params.modality_contract_key,
                        "routing_slot": params.routing_slot,
                        "usage": params.usage,
                        "slot_id": params.slot_id,
                        "anchor_section_title": params.anchor_section_title,
                        "anchor_text": params.anchor_text,
                        "model_task_request": {
                            "taskKind": "image_generate"
                        },
                        "runtime_contract": params.runtime_contract,
                    }
                }),
                ..MediaTaskArtifactResponse::default()
            })
        }
    }

    #[derive(Default)]
    struct FakeToolSearchGateway {
        last_params: Mutex<Option<McpToolSearchParams>>,
    }

    #[async_trait]
    impl ToolSearchGateway for FakeToolSearchGateway {
        async fn search_tools(
            &self,
            params: McpToolSearchParams,
        ) -> Result<McpToolListResponse, String> {
            *self.last_params.lock().await = Some(params);
            Ok(McpToolListResponse {
                tools: vec![json!({
                    "name": "mcp__browser__click",
                    "description": "Click an element",
                    "source": "extension",
                    "status": "deferred"
                })],
            })
        }
    }

    #[tokio::test]
    async fn memory_read_delegates_to_runtime_permissions_and_executor() {
        let dir = tempdir().expect("tempdir");
        let registry =
            install_gateway_registrations(create_memory_tools(Arc::new(FakeMemoryStoreGateway)));
        let denied_params = json!({ "path": "../MEMORY.md" });
        let denied = execute_runtime_gateway_dispatch_tool(
            &registry,
            RuntimeGatewayDispatchToolRequest {
                tool_name: tool_runtime::memory_store::MEMORY_READ_TOOL_NAME,
                params: &denied_params,
                working_directory: dir.path().to_path_buf(),
                session_id: "test-session".to_string(),
                cancel_token: None,
                turn_context: None,
            },
        )
        .await
        .expect("memory_read registration");
        let params = json!({ "path": "MEMORY.md" });
        let result = execute_runtime_gateway_dispatch_tool(
            &registry,
            RuntimeGatewayDispatchToolRequest {
                tool_name: tool_runtime::memory_store::MEMORY_READ_TOOL_NAME,
                params: &params,
                working_directory: dir.path().to_path_buf(),
                session_id: "test-session".to_string(),
                cancel_token: None,
                turn_context: None,
            },
        )
        .await
        .expect("memory_read registration")
        .expect("memory_read result");

        assert!(denied.is_err());
        assert_eq!(result.is_error, Some(false));
        assert_eq!(
            result.content[0].as_text().map(|text| text.text.as_str()),
            Some("remember this")
        );
        assert_eq!(
            result
                .structured_content
                .as_ref()
                .and_then(|metadata| metadata.get("operation")),
            Some(&json!("read"))
        );
    }

    #[tokio::test]
    async fn image_task_delegates_to_runtime_permissions_and_executor() {
        let workspace = TempDir::new().expect("workspace");
        let gateway = Arc::new(ImageToolTestGateway::default());
        let registry = install_gateway_registrations(create_image_tools(gateway.clone()));
        let denied_params = json!({
            "project_root_path": workspace.path().to_string_lossy(),
            "prompt": "生成分镜",
            "layout_hint": "storyboard_3x3",
            "thread_id": "thread-image-1",
            "turn_id": "turn-image-1"
        });
        let denied = execute_runtime_gateway_dispatch_tool(
            &registry,
            RuntimeGatewayDispatchToolRequest {
                tool_name: tool_runtime::image_task::IMAGE_TASK_TOOL_NAME,
                params: &denied_params,
                working_directory: workspace.path().to_path_buf(),
                session_id: "test-session".to_string(),
                cancel_token: None,
                turn_context: None,
            },
        )
        .await
        .expect("image registration");
        let params = json!({
            "project_root_path": workspace.path().to_string_lossy(),
            "prompt": "生成一张青柠实验室封面",
            "provider_id": "openai",
            "model": "gpt-image-2",
            "executor_mode": "images_api",
            "thread_id": "thread-image-1",
            "turn_id": "turn-image-1",
            "content_id": "content-image-1",
            "entry_source": "at_image_command",
            "modality_contract_key": "image_generation",
            "modality": "image",
            "routing_slot": "image_generation_model",
            "runtime_contract": {
                "contract_key": "image_generation",
                "routing_slot": "image_generation_model"
            },
            "usage": "document-inline",
            "slot_id": "document-image-slot-1",
            "anchor_section_title": "产品愿景",
            "anchor_text": "给这一段生成配图"
        });
        let result = execute_runtime_gateway_dispatch_tool(
            &registry,
            RuntimeGatewayDispatchToolRequest {
                tool_name: tool_runtime::image_task::IMAGE_TASK_TOOL_NAME,
                params: &params,
                working_directory: workspace.path().to_path_buf(),
                session_id: "test-session".to_string(),
                cancel_token: None,
                turn_context: None,
            },
        )
        .await
        .expect("image registration")
        .expect("image result");

        assert!(denied.is_err());
        assert_eq!(result.is_error, Some(false));
        assert_eq!(
            result
                .structured_content
                .as_ref()
                .and_then(|metadata| metadata.get("task_type")),
            Some(&json!("image_generate"))
        );
        let params = gateway
            .last_params
            .lock()
            .await
            .clone()
            .expect("gateway params");
        assert_eq!(params.session_id.as_deref(), Some("test-session"));
        assert_eq!(params.thread_id.as_deref(), Some("thread-image-1"));
        assert_eq!(params.turn_id.as_deref(), Some("turn-image-1"));
    }

    #[tokio::test]
    async fn tool_search_delegates_to_runtime_executor() {
        let dir = tempdir().expect("tempdir");
        let gateway = Arc::new(FakeToolSearchGateway::default());
        let registry = install_gateway_registrations(create_tool_search_tools(gateway.clone()));
        let params = json!({ "query": "browser click", "max_results": 3 });
        let result = execute_runtime_gateway_dispatch_tool(
            &registry,
            RuntimeGatewayDispatchToolRequest {
                tool_name: tool_runtime::tool_search::TOOL_SEARCH_TOOL_NAME,
                params: &params,
                working_directory: dir.path().to_path_buf(),
                session_id: "test-session".to_string(),
                cancel_token: None,
                turn_context: None,
            },
        )
        .await
        .expect("tool_search registration")
        .expect("tool_search result");

        assert_eq!(result.is_error, Some(false));
        assert_eq!(
            result
                .structured_content
                .as_ref()
                .and_then(|metadata| metadata.get("tool_search_result_count")),
            Some(&json!(1))
        );
        let output: serde_json::Value = serde_json::from_str(
            result.content[0]
                .as_text()
                .map(|text| text.text.as_str())
                .expect("tool_search text"),
        )
        .expect("tool_search output");
        assert_eq!(output["matches"], json!(["mcp__browser__click"]));

        let params = gateway
            .last_params
            .lock()
            .await
            .clone()
            .expect("gateway params");
        assert_eq!(params.query, "browser click");
        assert_eq!(params.limit, 3);
        assert_eq!(params.caller.as_deref(), Some("tool_search"));
    }
}
