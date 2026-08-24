//! 工作区工具的 current dispatch。
//!
//! `Read`、`Glob` 与 `Grep` 已经是 tool-runtime 的 current 实现；本模块只把它们
//! 统一到 `RuntimeToolExecutor`，让 provider turn loop 不需要经过 Agent tool registry。

use crate::file_read_execution::{
    execute_runtime_file_read_tool, file_read_canonical_tool_name, file_read_tool_definition,
    RuntimeFileReadRequest,
};
use crate::file_search_execution::{
    execute_runtime_file_search_tool, file_search_canonical_tool_name,
    file_search_tool_definitions, RuntimeFileSearchRequest,
};
use crate::tool_definition::RuntimeToolDefinition;
use crate::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionFuture, RuntimeToolExecutionRequest,
    RuntimeToolExecutionResult, RuntimeToolExecutor, RuntimeToolExecutorHandle,
    RuntimeToolPolicyErrorKind,
};
use rmcp::model::{CallToolResult, Content};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

#[derive(Debug, Default)]
pub struct RuntimeWorkspaceDispatch;

impl RuntimeWorkspaceDispatch {
    async fn execute_workspace_tool(
        &self,
        request: RuntimeToolExecutionRequest<'_>,
    ) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
        let context = request.context;
        let result = if file_read_canonical_tool_name(request.tool_name).is_some() {
            execute_runtime_file_read_tool(RuntimeFileReadRequest {
                tool_name: request.tool_name,
                params: request.params,
                working_directory: context.working_directory().clone(),
                cancel_token: context.cancel_token().cloned(),
                environment_id: context.environment_id().map(ToString::to_string),
                filesystem_gateway: context.filesystem_gateway().cloned(),
                sandbox_policy: request
                    .turn_context
                    .and_then(|turn| turn.sandbox_policy.clone()),
            })
            .await
        } else if file_search_canonical_tool_name(request.tool_name).is_some() {
            execute_runtime_file_search_tool(RuntimeFileSearchRequest {
                tool_name: request.tool_name,
                params: request.params,
                working_directory: context.working_directory().clone(),
                cancel_token: context.cancel_token().cloned(),
                environment_id: context.environment_id().map(ToString::to_string),
                filesystem_gateway: context.filesystem_gateway().cloned(),
                sandbox_policy: request
                    .turn_context
                    .and_then(|turn| turn.sandbox_policy.clone()),
            })
            .await
        } else {
            None
        };

        let result = result.ok_or_else(|| unsupported_workspace_tool(request.tool_name))?;
        result
            .map(call_tool_result_to_runtime_result)
            .map_err(|error| runtime_workspace_error(error.message))
    }
}

impl RuntimeToolExecutor for RuntimeWorkspaceDispatch {
    fn execute<'a>(
        &'a self,
        request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move { self.execute_workspace_tool(request).await })
    }
}

pub fn runtime_workspace_dispatch_handle() -> RuntimeToolExecutorHandle {
    static HANDLE: OnceLock<RuntimeToolExecutorHandle> = OnceLock::new();
    HANDLE
        .get_or_init(|| RuntimeToolExecutorHandle::new(Arc::new(RuntimeWorkspaceDispatch)))
        .clone()
}

pub fn runtime_workspace_dispatch_definitions() -> Vec<RuntimeToolDefinition> {
    let mut definitions = vec![file_read_tool_definition()];
    definitions.extend(file_search_tool_definitions());
    definitions
}

fn unsupported_workspace_tool(tool_name: &str) -> RuntimeToolExecutionError {
    RuntimeToolExecutionError::new(
        format!("unsupported workspace tool: {tool_name}"),
        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
            "unsupported_workspace_tool".to_string(),
        )),
    )
}

fn runtime_workspace_error(message: impl Into<String>) -> RuntimeToolExecutionError {
    RuntimeToolExecutionError::new(
        message.into(),
        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
            "workspace_tool_failed".to_string(),
        )),
    )
}

fn call_tool_result_to_runtime_result(result: CallToolResult) -> RuntimeToolExecutionResult {
    let output = result
        .content
        .iter()
        .filter_map(content_text)
        .collect::<Vec<_>>()
        .join("\n");
    let metadata = result
        .structured_content
        .and_then(|value| value.as_object().cloned())
        .map(|value| value.into_iter().collect::<HashMap<_, _>>())
        .unwrap_or_default();
    let success = !result.is_error.unwrap_or(false);
    RuntimeToolExecutionResult::new(
        success,
        output.clone(),
        (!success).then_some(output),
        metadata,
    )
}

fn content_text(content: &Content) -> Option<String> {
    content.as_text().map(|text| text.text.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_executor::{RuntimeToolExecutionContext, RuntimeToolExecutionContextInput};
    use serde_json::json;

    #[test]
    fn definitions_expose_current_workspace_tool_surface() {
        let names = runtime_workspace_dispatch_definitions()
            .into_iter()
            .map(|definition| definition.name)
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["Read", "Glob", "Grep"]);
    }

    #[tokio::test]
    async fn dispatch_executes_read_without_agent_adapter() {
        let directory = tempfile::tempdir().expect("tempdir");
        std::fs::write(directory.path().join("note.txt"), "hello current runtime")
            .expect("write fixture");
        let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: directory.path().to_path_buf(),
            session_id: "session-1".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        });
        let result = runtime_workspace_dispatch_handle()
            .execute(RuntimeToolExecutionRequest {
                tool_name: "Read",
                params: &json!({ "path": "note.txt" }),
                context: &context,
                turn_context: None,
            })
            .await
            .expect("read result");

        assert!(result.success);
        assert!(result.output.contains("hello current runtime"));
    }

    #[test]
    fn call_tool_result_preserves_error_and_structured_metadata() {
        let result = call_tool_result_to_runtime_result(CallToolResult {
            content: vec![Content::text("denied")],
            structured_content: Some(json!({ "reason": "policy" })),
            is_error: Some(true),
            meta: None,
        });
        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("denied"));
        assert_eq!(result.metadata.get("reason"), Some(&json!("policy")));
    }
}
