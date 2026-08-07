use crate::apply_patch::{apply_patch_tool_definition, runtime_apply_patch_executor_handle};
use crate::current_time::{
    current_time_tool_definition, runtime_current_time_executor_handle, CurrentTimeGateway,
    CLOCK_CURRENT_TIME_TOOL_NAME,
};
use crate::image_task::{
    image_task_tool_definition, runtime_image_task_executor_handle, ImageTaskGateway,
};
use crate::mcp_resource::{
    list_mcp_resources_tool_definition, read_mcp_resource_tool_definition,
    runtime_mcp_resource_executor_handle, McpResourceGateway, LIST_MCP_RESOURCES_LOOKUP_ALIASES,
    READ_MCP_RESOURCE_LOOKUP_ALIASES,
};
use crate::memory_store::{
    memory_store_tool_definitions, runtime_memory_store_executor_handle, MemoryStoreGateway,
};
use crate::skill_search::{runtime_skill_search_executor_handle, skill_search_tool_definition};
use crate::sleep::{runtime_sleep_executor_handle, sleep_tool_definition, CLOCK_SLEEP_TOOL_NAME};
use crate::tool_definition::RuntimeToolDefinition;
use crate::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionFuture, RuntimeToolExecutionRequest,
    RuntimeToolExecutor, RuntimeToolExecutorHandle, RuntimeToolPolicyErrorKind,
};
use crate::tool_search::{
    runtime_tool_search_executor_handle, tool_search_definition, ToolSearchGateway,
};
use crate::update_plan::{
    runtime_plan_update_executor_handle, update_plan_definition, UPDATE_PLAN_LEGACY_ALIASES,
};
use crate::view_image::{
    runtime_view_image_executor_handle, view_image_tool_definition, VIEW_IMAGE_LEGACY_ALIASES,
};
use crate::web_fetch::{runtime_web_fetch_executor_handle, web_fetch_tool_definition};
use crate::web_search::{runtime_web_search_executor_handle, web_search_tool_definition};
use crate::workspace_dispatch::{
    runtime_workspace_dispatch_definitions, runtime_workspace_dispatch_handle,
};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

pub const TOOL_SEARCH_LOOKUP_ALIASES: &[&str] =
    &["ToolSearch", "ToolSearchTool", "mcp__system__tool_search"];

#[derive(Clone)]
pub struct NativeDispatch {
    entries: Arc<Vec<NativeDispatchEntry>>,
    lookup: Arc<HashMap<String, usize>>,
}

#[derive(Clone)]
struct NativeDispatchEntry {
    definition: RuntimeToolDefinition,
    executor: RuntimeToolExecutorHandle,
    aliases: &'static [&'static str],
}

#[derive(Default)]
pub struct NativeDispatchBuilder {
    entries: Vec<NativeDispatchEntry>,
    lookup: HashMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeNativeDispatchSurface {
    definition: RuntimeToolDefinition,
    aliases: &'static [&'static str],
}

impl RuntimeNativeDispatchSurface {
    pub fn new(definition: RuntimeToolDefinition, aliases: &'static [&'static str]) -> Self {
        Self {
            definition,
            aliases,
        }
    }

    pub fn definition(&self) -> RuntimeToolDefinition {
        self.definition.clone()
    }

    pub fn aliases(&self) -> &'static [&'static str] {
        self.aliases
    }
}

impl NativeDispatchBuilder {
    pub fn with_standard_tools(self) -> Self {
        let builder = self
            .register(
                apply_patch_tool_definition(),
                runtime_apply_patch_executor_handle(),
                &["ApplyPatchTool"],
            )
            .register(
                skill_search_tool_definition(),
                runtime_skill_search_executor_handle(),
                &["SkillSearchTool"],
            )
            .register(
                sleep_tool_definition(),
                runtime_sleep_executor_handle(),
                &[CLOCK_SLEEP_TOOL_NAME],
            )
            .register(
                view_image_tool_definition(),
                runtime_view_image_executor_handle(),
                VIEW_IMAGE_LEGACY_ALIASES,
            )
            .register(
                update_plan_definition(),
                runtime_plan_update_executor_handle(),
                UPDATE_PLAN_LEGACY_ALIASES,
            )
            .register(
                web_fetch_tool_definition(),
                runtime_web_fetch_executor_handle(),
                &["WebFetchTool", "mcp__system__web_fetch"],
            )
            .register(
                web_search_tool_definition(),
                runtime_web_search_executor_handle(),
                &["WebSearchTool", "mcp__system__web_search"],
            );

        runtime_workspace_dispatch_definitions()
            .into_iter()
            .fold(builder, |builder, definition| {
                builder.register(definition, runtime_workspace_dispatch_handle(), &[])
            })
    }

    pub fn with_memory_store_gateway(self, gateway: Arc<dyn MemoryStoreGateway>) -> Self {
        let executor = runtime_memory_store_executor_handle(gateway);
        memory_store_tool_definitions()
            .into_iter()
            .fold(self, |builder, definition| {
                builder.register(definition, executor.clone(), &[])
            })
    }

    pub fn with_image_task_gateway(self, gateway: Arc<dyn ImageTaskGateway>) -> Self {
        self.register(
            image_task_tool_definition(),
            runtime_image_task_executor_handle(gateway),
            &[],
        )
    }

    pub fn with_current_time_gateway(self, gateway: Arc<dyn CurrentTimeGateway>) -> Self {
        self.register(
            current_time_tool_definition(),
            runtime_current_time_executor_handle(gateway),
            &[CLOCK_CURRENT_TIME_TOOL_NAME],
        )
    }

    pub fn with_tool_search_gateway(self, gateway: Arc<dyn ToolSearchGateway>) -> Self {
        self.register(
            tool_search_definition(),
            runtime_tool_search_executor_handle(gateway),
            TOOL_SEARCH_LOOKUP_ALIASES,
        )
    }

    pub fn with_mcp_resource_gateway(self, gateway: Arc<dyn McpResourceGateway>) -> Self {
        let executor = runtime_mcp_resource_executor_handle(gateway);
        self.register(
            list_mcp_resources_tool_definition(),
            executor.clone(),
            LIST_MCP_RESOURCES_LOOKUP_ALIASES,
        )
        .register(
            read_mcp_resource_tool_definition(),
            executor,
            READ_MCP_RESOURCE_LOOKUP_ALIASES,
        )
    }

    pub fn register(
        mut self,
        definition: RuntimeToolDefinition,
        executor: RuntimeToolExecutorHandle,
        aliases: &'static [&'static str],
    ) -> Self {
        let index = self.entries.len();
        register_lookup_name(&mut self.lookup, &definition.name, index);
        for alias in aliases {
            register_lookup_name(&mut self.lookup, alias, index);
        }
        self.entries.push(NativeDispatchEntry {
            definition,
            executor,
            aliases,
        });
        self
    }

    pub fn build(self) -> NativeDispatch {
        NativeDispatch {
            entries: Arc::new(self.entries),
            lookup: Arc::new(self.lookup),
        }
    }
}

impl NativeDispatch {
    pub fn builder() -> NativeDispatchBuilder {
        NativeDispatchBuilder::default()
    }

    pub fn contains(&self, tool_name: &str) -> bool {
        self.lookup.contains_key(tool_name)
    }

    pub fn canonical_name(&self, tool_name: &str) -> Option<&str> {
        self.entry_for(tool_name)
            .map(|entry| entry.definition.name.as_str())
    }

    pub fn definitions(&self) -> Vec<RuntimeToolDefinition> {
        self.entries
            .iter()
            .map(|entry| entry.definition.clone())
            .collect()
    }

    pub fn surfaces(&self) -> Vec<RuntimeNativeDispatchSurface> {
        self.entries
            .iter()
            .map(|entry| RuntimeNativeDispatchSurface::new(entry.definition.clone(), entry.aliases))
            .collect()
    }

    fn entry_for(&self, tool_name: &str) -> Option<&NativeDispatchEntry> {
        self.lookup
            .get(tool_name)
            .and_then(|index| self.entries.get(*index))
    }
}

impl RuntimeToolExecutor for NativeDispatch {
    fn execute<'a>(
        &'a self,
        request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move {
            let entry = self.entry_for(request.tool_name).ok_or_else(|| {
                RuntimeToolExecutionError::new(
                    format!("unsupported native tool: {}", request.tool_name),
                    Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                        "unsupported_native_tool".to_string(),
                    )),
                )
            })?;
            entry
                .executor
                .execute(RuntimeToolExecutionRequest {
                    tool_name: entry.definition.name.as_str(),
                    params: request.params,
                    context: request.context,
                    turn_context: request.turn_context,
                })
                .await
        })
    }
}

fn register_lookup_name(lookup: &mut HashMap<String, usize>, name: &str, index: usize) {
    let replaced = lookup.insert(name.to_string(), index);
    debug_assert!(
        replaced.is_none(),
        "duplicate native dispatch tool name: {name}"
    );
}

fn build_runtime_native_dispatch() -> NativeDispatch {
    NativeDispatch::builder().with_standard_tools().build()
}

pub fn runtime_native_dispatch() -> NativeDispatch {
    static DISPATCH: OnceLock<NativeDispatch> = OnceLock::new();
    DISPATCH.get_or_init(build_runtime_native_dispatch).clone()
}

pub fn runtime_native_dispatch_handle() -> RuntimeToolExecutorHandle {
    static HANDLE: OnceLock<RuntimeToolExecutorHandle> = OnceLock::new();
    HANDLE
        .get_or_init(|| RuntimeToolExecutorHandle::new(Arc::new(runtime_native_dispatch())))
        .clone()
}

pub fn runtime_native_dispatch_definitions() -> Vec<RuntimeToolDefinition> {
    runtime_native_dispatch().definitions()
}

pub fn runtime_native_dispatch_tool_names() -> Vec<String> {
    runtime_native_dispatch()
        .definitions()
        .into_iter()
        .map(|definition| definition.name)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apply_patch::APPLY_PATCH_TOOL_NAME;
    use crate::skill_search::SKILL_SEARCH_TOOL_NAME;
    use crate::sleep::SLEEP_TOOL_NAME;
    use crate::tool_executor::{RuntimeToolExecutionContext, RuntimeToolExecutionContextInput};
    use crate::update_plan::UPDATE_PLAN_NAME;
    use crate::view_image::VIEW_IMAGE_TOOL_NAME;
    use crate::web_fetch::WEB_FETCH_TOOL_NAME;
    use crate::web_search::WEB_SEARCH_TOOL_NAME;
    use app_server_protocol::{
        McpResourceListResponse, McpToolListResponse, McpToolSearchParams,
        MediaTaskArtifactImageCreateParams, MediaTaskArtifactResponse, MemoryStoreAddNoteParams,
        MemoryStoreAddNoteResponse, MemoryStoreCitation, MemoryStoreListParams,
        MemoryStoreListResponse, MemoryStoreReadParams, MemoryStoreReadResponse,
        MemoryStoreSearchParams, MemoryStoreSearchResponse,
    };
    use async_trait::async_trait;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn context() -> RuntimeToolExecutionContext {
        RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: PathBuf::from("."),
            session_id: "dispatch-test-session".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        })
    }

    #[derive(Default)]
    struct DefinitionOnlyMemoryGateway;

    #[async_trait]
    impl MemoryStoreGateway for DefinitionOnlyMemoryGateway {
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
    struct DefinitionOnlyImageGateway;

    #[async_trait]
    impl ImageTaskGateway for DefinitionOnlyImageGateway {
        async fn create_image_media_task_artifact(
            &self,
            params: MediaTaskArtifactImageCreateParams,
        ) -> Result<MediaTaskArtifactResponse, String> {
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
                    }
                }),
                ..MediaTaskArtifactResponse::default()
            })
        }
    }

    #[derive(Default)]
    struct DefinitionOnlyToolSearchGateway;

    #[async_trait]
    impl ToolSearchGateway for DefinitionOnlyToolSearchGateway {
        async fn search_tools(
            &self,
            _params: McpToolSearchParams,
        ) -> Result<McpToolListResponse, String> {
            Ok(McpToolListResponse { tools: Vec::new() })
        }
    }

    #[derive(Default)]
    struct DefinitionOnlyMcpResourceGateway;

    #[async_trait]
    impl McpResourceGateway for DefinitionOnlyMcpResourceGateway {
        async fn list_mcp_resources(&self) -> Result<McpResourceListResponse, String> {
            Ok(McpResourceListResponse::default())
        }

        async fn read_mcp_resource(
            &self,
            _params: app_server_protocol::protocol::v2::McpServerResourceReadParams,
        ) -> Result<app_server_protocol::protocol::v2::McpServerResourceReadResponse, String>
        {
            Ok(
                app_server_protocol::protocol::v2::McpServerResourceReadResponse {
                    contents: Vec::new(),
                },
            )
        }
    }

    #[test]
    fn runtime_native_dispatch_lists_current_tools() {
        let names = runtime_native_dispatch_tool_names();

        assert_eq!(
            names,
            vec![
                APPLY_PATCH_TOOL_NAME,
                SKILL_SEARCH_TOOL_NAME,
                SLEEP_TOOL_NAME,
                VIEW_IMAGE_TOOL_NAME,
                UPDATE_PLAN_NAME,
                WEB_FETCH_TOOL_NAME,
                WEB_SEARCH_TOOL_NAME,
                "Read",
                "Glob",
                "Grep",
            ]
        );
    }

    #[test]
    fn runtime_native_dispatch_canonicalizes_lookup_only_aliases() {
        let dispatch = runtime_native_dispatch();

        assert_eq!(dispatch.canonical_name("clock.sleep"), Some("sleep"));
        assert_eq!(dispatch.canonical_name("ViewImageTool"), Some("view_image"));
        assert_eq!(
            dispatch.canonical_name("UpdatePlanTool"),
            Some("update_plan")
        );
        assert_eq!(dispatch.canonical_name("WebSearchTool"), Some("WebSearch"));
        assert_eq!(dispatch.canonical_name("missing"), None);
    }

    #[test]
    fn native_dispatch_builder_registers_gateway_tools() {
        let dispatch = NativeDispatch::builder()
            .with_memory_store_gateway(Arc::new(DefinitionOnlyMemoryGateway))
            .with_image_task_gateway(Arc::new(DefinitionOnlyImageGateway))
            .with_tool_search_gateway(Arc::new(DefinitionOnlyToolSearchGateway))
            .with_mcp_resource_gateway(Arc::new(DefinitionOnlyMcpResourceGateway))
            .build();

        assert_eq!(
            dispatch
                .definitions()
                .into_iter()
                .map(|tool| tool.name)
                .collect::<Vec<_>>(),
            vec![
                crate::memory_store::MEMORY_LIST_TOOL_NAME,
                crate::memory_store::MEMORY_READ_TOOL_NAME,
                crate::memory_store::MEMORY_SEARCH_TOOL_NAME,
                crate::memory_store::MEMORY_ADD_NOTE_TOOL_NAME,
                crate::image_task::IMAGE_TASK_TOOL_NAME,
                crate::tool_search::TOOL_SEARCH_TOOL_NAME,
                crate::mcp_resource::LIST_MCP_RESOURCES_TOOL_NAME,
                crate::mcp_resource::READ_MCP_RESOURCE_TOOL_NAME,
            ]
        );
        assert_eq!(
            dispatch.canonical_name(crate::memory_store::MEMORY_READ_TOOL_NAME),
            Some(crate::memory_store::MEMORY_READ_TOOL_NAME)
        );
        assert_eq!(
            dispatch.canonical_name(crate::image_task::IMAGE_TASK_TOOL_NAME),
            Some(crate::image_task::IMAGE_TASK_TOOL_NAME)
        );
        assert_eq!(
            dispatch.canonical_name("ToolSearch"),
            Some(crate::tool_search::TOOL_SEARCH_TOOL_NAME)
        );
        assert_eq!(
            dispatch.canonical_name("ToolSearchTool"),
            Some(crate::tool_search::TOOL_SEARCH_TOOL_NAME)
        );
        assert_eq!(
            dispatch.canonical_name("ListMcpResourcesTool"),
            Some(crate::mcp_resource::LIST_MCP_RESOURCES_TOOL_NAME)
        );
        assert_eq!(
            dispatch.canonical_name("ReadMcpResourceTool"),
            Some(crate::mcp_resource::READ_MCP_RESOURCE_TOOL_NAME)
        );
        assert_eq!(
            dispatch
                .surfaces()
                .into_iter()
                .map(|surface| {
                    (
                        surface.definition().name,
                        surface.aliases().iter().copied().collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>(),
            vec![
                (
                    crate::memory_store::MEMORY_LIST_TOOL_NAME.to_string(),
                    vec![]
                ),
                (
                    crate::memory_store::MEMORY_READ_TOOL_NAME.to_string(),
                    vec![]
                ),
                (
                    crate::memory_store::MEMORY_SEARCH_TOOL_NAME.to_string(),
                    vec![]
                ),
                (
                    crate::memory_store::MEMORY_ADD_NOTE_TOOL_NAME.to_string(),
                    vec![]
                ),
                (crate::image_task::IMAGE_TASK_TOOL_NAME.to_string(), vec![]),
                (
                    crate::tool_search::TOOL_SEARCH_TOOL_NAME.to_string(),
                    TOOL_SEARCH_LOOKUP_ALIASES.to_vec()
                ),
                (
                    crate::mcp_resource::LIST_MCP_RESOURCES_TOOL_NAME.to_string(),
                    crate::mcp_resource::LIST_MCP_RESOURCES_LOOKUP_ALIASES.to_vec()
                ),
                (
                    crate::mcp_resource::READ_MCP_RESOURCE_TOOL_NAME.to_string(),
                    crate::mcp_resource::READ_MCP_RESOURCE_LOOKUP_ALIASES.to_vec()
                ),
            ]
        );
    }

    #[tokio::test]
    async fn runtime_native_dispatch_executes_canonical_tool() {
        let params = json!({
            "explanation": "dispatch",
            "plan": [
                { "step": "wire dispatcher", "status": "in_progress" }
            ]
        });
        let context = context();

        let result = runtime_native_dispatch_handle()
            .execute(RuntimeToolExecutionRequest {
                tool_name: "UpdatePlanTool",
                params: &params,
                context: &context,
                turn_context: None,
            })
            .await
            .expect("dispatcher should execute canonical current tool");

        assert!(result.success);
        assert_eq!(result.output, "Plan updated");
        assert_eq!(
            result.metadata.get("tool_family"),
            Some(&json!("update_plan"))
        );
    }

    #[tokio::test]
    async fn runtime_native_dispatch_rejects_unknown_tool() {
        let params = json!({});
        let context = context();

        let error = runtime_native_dispatch_handle()
            .execute(RuntimeToolExecutionRequest {
                tool_name: "ConfigTool",
                params: &params,
                context: &context,
                turn_context: None,
            })
            .await
            .expect_err("unknown tools should fail closed");

        assert!(error.message().contains("unsupported native tool"));
    }
}
