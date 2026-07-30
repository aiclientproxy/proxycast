use crate::tool_definition::RuntimeToolDefinition;
use crate::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionFuture, RuntimeToolExecutionRequest,
    RuntimeToolExecutionResult, RuntimeToolExecutor, RuntimeToolExecutorHandle,
    RuntimeToolPolicyErrorKind,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

pub const CURRENT_TIME_TOOL_NAME: &str = "curr_time";
pub const CLOCK_CURRENT_TIME_TOOL_NAME: &str = "clock.curr_time";

#[async_trait]
pub trait CurrentTimeGateway: Send + Sync {
    async fn read_current_time(&self, thread_id: &str) -> Result<i64, String>;
}

pub fn current_time_tool_definition() -> RuntimeToolDefinition {
    RuntimeToolDefinition::new(
        CURRENT_TIME_TOOL_NAME,
        "Return the current time in UTC.",
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }),
    )
}

pub fn runtime_current_time_executor_handle(
    gateway: Arc<dyn CurrentTimeGateway>,
) -> RuntimeToolExecutorHandle {
    RuntimeToolExecutorHandle::new(Arc::new(RuntimeCurrentTimeExecutor { gateway }))
}

struct RuntimeCurrentTimeExecutor {
    gateway: Arc<dyn CurrentTimeGateway>,
}

impl RuntimeToolExecutor for RuntimeCurrentTimeExecutor {
    fn execute<'a>(
        &'a self,
        request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move {
            if request.tool_name != CURRENT_TIME_TOOL_NAME
                && request.tool_name != CLOCK_CURRENT_TIME_TOOL_NAME
            {
                return Err(current_time_error(format!(
                    "current-time executor cannot run tool '{}'",
                    request.tool_name
                )));
            }
            serde_json::from_value::<CurrentTimeInput>(request.params.clone()).map_err(
                |error| current_time_error(format!("curr_time parameters are invalid: {error}")),
            )?;
            let thread_id = current_thread_id(request.turn_context)?;
            let current_time_at = self
                .gateway
                .read_current_time(thread_id)
                .await
                .map_err(current_time_error)?;
            let current_time = DateTime::<Utc>::from_timestamp(current_time_at, 0)
                .ok_or_else(|| current_time_error("current time is outside the supported range"))?;
            let formatted = current_time.format("%Y-%m-%d %H:%M:%S UTC").to_string();

            Ok(RuntimeToolExecutionResult::new(
                true,
                formatted.clone(),
                None,
                HashMap::from([
                    ("tool_family".to_string(), json!("clock")),
                    ("current_time".to_string(), json!(formatted)),
                    ("current_time_at".to_string(), json!(current_time_at)),
                ]),
            ))
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CurrentTimeInput {}

fn current_thread_id(
    turn_context: Option<&crate::tool_executor::RuntimeToolTurnContext>,
) -> Result<&str, RuntimeToolExecutionError> {
    turn_context
        .and_then(|context| context.metadata.get("app_server_runtime_backend"))
        .and_then(|metadata| metadata.get("threadId"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|thread_id| !thread_id.is_empty())
        .ok_or_else(|| current_time_error("curr_time requires canonical thread identity"))
}

fn current_time_error(message: impl Into<String>) -> RuntimeToolExecutionError {
    let message = message.into();
    RuntimeToolExecutionError::new(
        message.clone(),
        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(message)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_executor::{RuntimeToolExecutionContext, RuntimeToolExecutionContextInput};
    use agent_protocol::turn_context::TurnContextOverride;
    use std::path::PathBuf;

    struct FixedCurrentTime;

    #[async_trait]
    impl CurrentTimeGateway for FixedCurrentTime {
        async fn read_current_time(&self, thread_id: &str) -> Result<i64, String> {
            assert_eq!(thread_id, "thread-clock");
            Ok(1_783_860_000)
        }
    }

    fn context() -> RuntimeToolExecutionContext {
        RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: PathBuf::from("/tmp/workspace"),
            session_id: "session-clock".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        })
    }

    fn turn_context() -> TurnContextOverride {
        let mut context = TurnContextOverride::default();
        context.metadata.insert(
            "app_server_runtime_backend".to_string(),
            json!({ "threadId": "thread-clock" }),
        );
        context
    }

    #[tokio::test]
    async fn current_time_tool_reads_the_host_gateway() {
        let context = context();
        let turn_context = turn_context();
        let result = runtime_current_time_executor_handle(Arc::new(FixedCurrentTime))
            .execute(RuntimeToolExecutionRequest {
                tool_name: CURRENT_TIME_TOOL_NAME,
                params: &json!({}),
                context: &context,
                turn_context: Some(&turn_context),
            })
            .await
            .expect("current time");

        assert!(result.success);
        assert_eq!(result.output, "2026-07-12 12:40:00 UTC");
        assert_eq!(
            result.metadata.get("current_time_at"),
            Some(&json!(1_783_860_000_i64))
        );
    }

    #[tokio::test]
    async fn current_time_tool_fails_closed_without_thread_identity() {
        let context = context();
        let error = runtime_current_time_executor_handle(Arc::new(FixedCurrentTime))
            .execute(RuntimeToolExecutionRequest {
                tool_name: CURRENT_TIME_TOOL_NAME,
                params: &json!({}),
                context: &context,
                turn_context: None,
            })
            .await
            .expect_err("missing thread identity");

        assert!(error.message().contains("canonical thread identity"));
    }
}
