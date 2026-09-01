use crate::tool_definition::RuntimeToolDefinition;
use crate::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionFuture, RuntimeToolExecutionRequest,
    RuntimeToolExecutionResult, RuntimeToolExecutor, RuntimeToolExecutorHandle,
    RuntimeToolPolicyErrorKind, RuntimeToolTurnContext,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

pub const UPDATE_PLAN_NAME: &str = "update_plan";
pub const UPDATE_PLAN_LEGACY_ALIASES: &[&str] =
    &["UpdatePlan", "UpdatePlanTool", "update_plan_tool"];

/// Resolves the per-turn checklist exposure switch.
///
/// The absence of a switch preserves Lime's existing product default. Callers can
/// explicitly disable the surface through trusted runtime/config metadata, matching
/// Codex's opt-in semantics without allowing arbitrary prompt text to change tools.
pub fn update_plan_enabled_from_metadata(metadata: Option<&serde_json::Value>) -> bool {
    let Some(metadata) = metadata else {
        return true;
    };
    [
        "/harness/update_plan_enabled",
        "/harness/updatePlanEnabled",
        "/harness/features/update_plan/enabled",
        "/harness/features/updatePlan/enabled",
        "/runtime_request/update_plan_enabled",
        "/runtime_request/updatePlanEnabled",
        "/runtime_request/features/update_plan/enabled",
        "/runtime_request/features/updatePlan/enabled",
        "/config/agent/toolExecution/update_plan_enabled",
        "/config/agent/toolExecution/updatePlanEnabled",
        "/config/experimental/update_plan/enabled",
        "/config/experimental/updatePlan/enabled",
    ]
    .into_iter()
    .find_map(|pointer| {
        metadata
            .pointer(pointer)
            .and_then(serde_json::Value::as_bool)
    })
    .unwrap_or(true)
}

#[cfg(test)]
mod feature_tests {
    use super::update_plan_enabled_from_metadata;
    use serde_json::json;

    #[test]
    fn explicit_disable_is_honored_across_camel_and_snake_paths() {
        assert!(!update_plan_enabled_from_metadata(Some(&json!({
            "harness": {"updatePlanEnabled": false}
        }))));
        assert!(!update_plan_enabled_from_metadata(Some(&json!({
            "runtime_request": {"features": {"update_plan": {"enabled": false}}}
        }))));
    }

    #[test]
    fn missing_switch_keeps_lime_default() {
        assert!(update_plan_enabled_from_metadata(None));
        assert!(update_plan_enabled_from_metadata(Some(
            &json!({"harness": {}})
        )));
    }
}
pub const PLAN_UPDATED_MESSAGE: &str = "Plan updated";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanStepStatus {
    Pending,
    InProgress,
    Completed,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlanStep {
    pub step: String,
    pub status: PlanStepStatus,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlanUpdate {
    #[serde(default)]
    pub explanation: Option<String>,
    pub plan: Vec<PlanStep>,
}

#[derive(Debug, Default)]
pub struct RuntimePlanUpdateExecutor;

impl RuntimePlanUpdateExecutor {
    pub fn new() -> Self {
        Self
    }

    pub fn handle() -> RuntimeToolExecutorHandle {
        RuntimeToolExecutorHandle::new(Arc::new(Self::new()))
    }

    async fn execute_plan_update(
        &self,
        request: RuntimeToolExecutionRequest<'_>,
    ) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
        if request.tool_name != UPDATE_PLAN_NAME {
            return Err(plan_update_error(format!(
                "update_plan executor cannot run tool '{}'",
                request.tool_name
            )));
        }

        check_plan_mode(request.turn_context)?;
        let update = parse_plan_update(request.params.clone())?;
        validate_plan_update(&update)?;
        let metadata = HashMap::from([
            ("tool_family".to_string(), json!("update_plan")),
            ("explanation".to_string(), json!(update.explanation)),
            ("plan".to_string(), json!(update.plan)),
        ]);

        Ok(RuntimeToolExecutionResult::new(
            true,
            PLAN_UPDATED_MESSAGE.to_string(),
            None,
            metadata,
        ))
    }
}

impl RuntimeToolExecutor for RuntimePlanUpdateExecutor {
    fn execute<'a>(
        &'a self,
        request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move { self.execute_plan_update(request).await })
    }
}

pub fn runtime_plan_update_executor_handle() -> RuntimeToolExecutorHandle {
    static HANDLE: OnceLock<RuntimeToolExecutorHandle> = OnceLock::new();
    HANDLE
        .get_or_init(RuntimePlanUpdateExecutor::handle)
        .clone()
}

pub fn update_plan_definition() -> RuntimeToolDefinition {
    RuntimeToolDefinition::new(
        UPDATE_PLAN_NAME,
        r#"Updates the task plan.
Provide an optional explanation and a list of plan items, each with a step and status.
At most one step can be in_progress at a time."#,
        update_plan_input_schema(),
    )
}

pub fn update_plan_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "description": "Optional explanation for this plan update."
            },
            "plan": {
                "type": "array",
                "description": "The list of steps",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {
                            "type": "string",
                            "description": "Task step text."
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "Step status."
                        }
                    },
                    "required": ["step", "status"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["plan"],
        "additionalProperties": false
    })
}

pub fn check_plan_update_permissions(params: &Value) -> Result<(), RuntimeToolExecutionError> {
    let update = parse_plan_update(params.clone())?;
    validate_plan_update(&update)
}

fn parse_plan_update(params: Value) -> Result<PlanUpdate, RuntimeToolExecutionError> {
    serde_json::from_value(params)
        .map_err(|error| plan_update_error(format!("update_plan 参数无效: {error}")))
}

fn validate_plan_update(update: &PlanUpdate) -> Result<(), RuntimeToolExecutionError> {
    let in_progress_count = update
        .plan
        .iter()
        .filter(|item| item.status == PlanStepStatus::InProgress)
        .count();

    if in_progress_count > 1 {
        return Err(plan_update_error(
            "update_plan 最多只能有一个 in_progress 步骤",
        ));
    }

    for item in &update.plan {
        if item.step.trim().is_empty() {
            return Err(plan_update_error("update_plan.plan[].step 不能为空"));
        }
    }

    Ok(())
}

fn check_plan_mode(
    turn_context: Option<&RuntimeToolTurnContext>,
) -> Result<(), RuntimeToolExecutionError> {
    if turn_context
        .and_then(|context| context.collaboration_mode.as_ref())
        .is_some_and(|mode| mode.mode == agent_protocol::ModeKind::Plan)
    {
        return Err(plan_update_error(
            "update_plan is a TODO/checklist tool and is not allowed in Plan mode",
        ));
    }

    Ok(())
}

fn plan_update_error(message: impl Into<String>) -> RuntimeToolExecutionError {
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
    use serde_json::json;
    use std::path::PathBuf;

    fn context() -> RuntimeToolExecutionContext {
        RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: PathBuf::from("/tmp/workspace"),
            session_id: "session-plan-1".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        })
    }

    #[test]
    fn update_plan_definition_matches_codex_shape() {
        let definition = update_plan_definition();

        assert_eq!(definition.name, UPDATE_PLAN_NAME);
        assert!(definition.description.contains("At most one step"));
        assert_eq!(
            definition.input_schema["properties"]["plan"]["items"]["properties"]["status"]["enum"],
            json!(["pending", "in_progress", "completed"])
        );
        assert_eq!(
            definition.input_schema["additionalProperties"],
            json!(false)
        );
    }

    #[test]
    fn plan_update_permissions_reject_multiple_in_progress_steps() {
        let result = check_plan_update_permissions(&json!({
            "plan": [
                { "step": "读现状", "status": "in_progress" },
                { "step": "实现迁移", "status": "in_progress" }
            ]
        }));

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn plan_update_executor_returns_codex_ack_and_metadata() {
        let runtime_context = context();
        let result = runtime_plan_update_executor_handle()
            .execute(RuntimeToolExecutionRequest {
                tool_name: UPDATE_PLAN_NAME,
                params: &json!({
                    "explanation": "继续实现",
                    "plan": [
                        { "step": "读现状", "status": "completed" },
                        { "step": "实现迁移", "status": "in_progress" }
                    ]
                }),
                context: &runtime_context,
                turn_context: None,
            })
            .await
            .expect("update_plan execution should succeed");

        assert!(result.success);
        assert_eq!(result.output, PLAN_UPDATED_MESSAGE);
        assert_eq!(
            result.metadata.get("tool_family"),
            Some(&json!("update_plan"))
        );
        assert_eq!(result.metadata.get("explanation"), Some(&json!("继续实现")));
        assert_eq!(result.metadata["plan"][1]["status"], json!("in_progress"));
    }

    #[tokio::test]
    async fn plan_update_executor_rejects_plan_mode() {
        let runtime_context = context();
        let turn_context = TurnContextOverride {
            collaboration_mode: Some(agent_protocol::CollaborationMode {
                mode: agent_protocol::ModeKind::Plan,
                settings: agent_protocol::CollaborationModeSettings {
                    model: "gpt-5-codex".to_string(),
                    reasoning_effort: None,
                    developer_instructions: None,
                },
            }),
            ..TurnContextOverride::default()
        };
        let error = runtime_plan_update_executor_handle()
            .execute(RuntimeToolExecutionRequest {
                tool_name: UPDATE_PLAN_NAME,
                params: &json!({
                    "plan": [
                        { "step": "写计划", "status": "in_progress" }
                    ]
                }),
                context: &runtime_context,
                turn_context: Some(&turn_context),
            })
            .await
            .expect_err("update_plan should be rejected in Plan mode");

        assert!(error.message().contains("not allowed in Plan mode"));
    }

    #[test]
    fn plan_update_rejects_non_codex_status_aliases() {
        let result = check_plan_update_permissions(&json!({
            "plan": [
                { "step": "处理别名", "status": "inProgress" }
            ]
        }));

        assert!(result.is_err());
    }
}
