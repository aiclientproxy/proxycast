use crate::native_dispatch::{runtime_native_dispatch, runtime_native_dispatch_handle};
use crate::native_overlay::{
    check_runtime_native_tool_permissions, runtime_native_tool_overlay_for_dispatch_name,
    RuntimeNativePermissionDecision,
};
use crate::tool_executor::{
    turn_context_has_tool_approval, RuntimeToolExecutionContext, RuntimeToolExecutionContextInput,
    RuntimeToolExecutionError, RuntimeToolExecutionRequest, RuntimeToolExecutionResult,
    RuntimeToolPolicyErrorKind, RuntimeToolTurnContext,
};
use crate::tool_result_projection::{
    runtime_tool_result_to_call_tool_result, RuntimeToolResultParts,
    TOOL_HANDLER_EXECUTED_METADATA_KEY,
};
use rmcp::model::{CallToolResult, ErrorCode, ErrorData};
use serde_json::Value;
use std::path::PathBuf;
use tokio_util::sync::CancellationToken;

pub struct RuntimeNativeDispatchToolRequest<'a> {
    pub tool_name: &'a str,
    pub params: &'a Value,
    pub working_directory: PathBuf,
    pub session_id: String,
    pub cancel_token: Option<CancellationToken>,
    pub turn_context: Option<&'a RuntimeToolTurnContext>,
    pub attempt: Option<crate::execution_orchestrator::RuntimeToolExecutionAttempt>,
}

pub type RuntimeNativeDispatchToolResult = Result<CallToolResult, ErrorData>;

fn runtime_native_dispatch_error(message: impl Into<String>, handler_executed: bool) -> ErrorData {
    ErrorData::new(
        ErrorCode::INTERNAL_ERROR,
        message.into(),
        Some(Value::Object(
            [(
                TOOL_HANDLER_EXECUTED_METADATA_KEY.to_string(),
                Value::Bool(handler_executed),
            )]
            .into_iter()
            .collect(),
        )),
    )
}

pub async fn execute_runtime_native_dispatch_tool(
    request: RuntimeNativeDispatchToolRequest<'_>,
) -> Option<RuntimeNativeDispatchToolResult> {
    execute_runtime_native_dispatch_tool_typed(request)
        .await
        .map(|result| match result {
            Ok(result) => Ok(runtime_tool_result_to_call_tool_result(
                RuntimeToolResultParts {
                    success: result.success,
                    output: Some(result.output),
                    error: result.error,
                    metadata: result.metadata,
                },
            )),
            Err(error) => Err(runtime_native_dispatch_error(
                error.message().to_string(),
                error.handler_executed(),
            )),
        })
}

pub async fn execute_runtime_native_dispatch_tool_typed(
    request: RuntimeNativeDispatchToolRequest<'_>,
) -> Option<Result<RuntimeToolExecutionResult, RuntimeToolExecutionError>> {
    let dispatch = runtime_native_dispatch();
    let canonical_tool_name = dispatch.canonical_name(request.tool_name)?.to_string();

    let attempt_manages_apply_patch = request.attempt.is_some()
        && canonical_tool_name == crate::apply_patch::APPLY_PATCH_TOOL_NAME;
    if !attempt_manages_apply_patch {
        if let Some(overlay) = runtime_native_tool_overlay_for_dispatch_name(request.tool_name) {
            match check_runtime_native_tool_permissions(
                overlay,
                request.params,
                &request.working_directory,
                request.turn_context,
            ) {
                RuntimeNativePermissionDecision::Allow => {}
                RuntimeNativePermissionDecision::Deny(message) => {
                    return Some(Err(RuntimeToolExecutionError::new(
                        message,
                        Some(RuntimeToolPolicyErrorKind::PermissionDenied(
                            "native_tool_permission_denied".to_string(),
                        )),
                    )
                    .before_handler()));
                }
                RuntimeNativePermissionDecision::Ask(message) => {
                    if !turn_context_has_tool_approval(request.turn_context) {
                        return Some(Err(RuntimeToolExecutionError::new(
                            message,
                            Some(RuntimeToolPolicyErrorKind::PermissionDenied(
                                "native_tool_approval_required".to_string(),
                            )),
                        )
                        .before_handler()));
                    }
                }
            }
        }
    }

    if request
        .cancel_token
        .as_ref()
        .is_some_and(CancellationToken::is_cancelled)
    {
        return Some(Err(RuntimeToolExecutionError::new(
            "Tool execution cancelled",
            Some(RuntimeToolPolicyErrorKind::Canceled(
                "native_dispatch_cancelled".to_string(),
            )),
        )
        .before_handler()));
    }

    let mut runtime_context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
        working_directory: request.working_directory,
        session_id: request.session_id,
        cancel_token: request.cancel_token,
        workspace_sandbox: None,
    });
    if let Some(attempt) = request.attempt {
        runtime_context = runtime_context
            .with_tool_identity(attempt.identity().clone())
            .with_execution_attempt(attempt);
    }
    Some(
        runtime_native_dispatch_handle()
            .execute(RuntimeToolExecutionRequest {
                tool_name: &canonical_tool_name,
                params: request.params,
                context: &runtime_context,
                turn_context: request.turn_context,
            })
            .await,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sleep::SLEEP_TOOL_NAME;
    use serde_json::json;

    #[tokio::test]
    async fn unknown_tool_returns_none_for_registry_fallback() {
        let result = execute_runtime_native_dispatch_tool(RuntimeNativeDispatchToolRequest {
            tool_name: "missing_tool",
            params: &json!({ "command": "echo hi" }),
            working_directory: PathBuf::from("."),
            session_id: "session-native-dispatch-1".to_string(),
            cancel_token: None,
            turn_context: None,
            attempt: None,
        })
        .await;

        assert!(result.is_none());
    }

    #[tokio::test]
    async fn workspace_tool_executes_through_current_dispatch() {
        let directory = tempfile::tempdir().expect("tempdir");
        std::fs::write(directory.path().join("current.txt"), "current-dispatch")
            .expect("write fixture");
        let result = execute_runtime_native_dispatch_tool(RuntimeNativeDispatchToolRequest {
            tool_name: "Read",
            params: &json!({ "path": "current.txt" }),
            working_directory: directory.path().to_path_buf(),
            session_id: "session-native-dispatch-workspace".to_string(),
            cancel_token: None,
            turn_context: None,
            attempt: None,
        })
        .await
        .expect("workspace tool is dispatch-backed")
        .expect("workspace tool execution");

        assert_eq!(result.is_error, Some(false));
        assert!(serde_json::to_string(&result)
            .expect("serialize result")
            .contains("current-dispatch"));
    }

    #[tokio::test]
    async fn cancelled_request_fails_before_executor_runs() {
        let cancel_token = CancellationToken::new();
        cancel_token.cancel();

        let result = execute_runtime_native_dispatch_tool(RuntimeNativeDispatchToolRequest {
            tool_name: SLEEP_TOOL_NAME,
            params: &json!({ "duration_ms": 60_000 }),
            working_directory: PathBuf::from("."),
            session_id: "session-native-dispatch-2".to_string(),
            cancel_token: Some(cancel_token),
            turn_context: None,
            attempt: None,
        })
        .await
        .expect("sleep is dispatch-backed");

        let Err(error) = result else {
            panic!("cancelled dispatch should fail before executing sleep");
        };
        assert!(error.message.contains("cancelled"));
        assert_eq!(
            error
                .data
                .as_ref()
                .and_then(|data| data.get(TOOL_HANDLER_EXECUTED_METADATA_KEY)),
            Some(&Value::Bool(false))
        );
    }
}
