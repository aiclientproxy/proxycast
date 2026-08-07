//! Hook 门控的工具执行装饰器。
//!
//! 把 [`crate::hook_lifecycle`] 的裁决接到真实工具执行：`PreToolUse` 可阻断或改写参数，
//! `PostToolUse` 在执行后求值。装饰器不改写任何既有 `RuntimeToolExecutor` 实现，
//! 也不自己决定 Hook 语义 —— 裁决只来自 `evaluate_hook_event`。
//!
//! 边界：
//! - 被阻断的调用不得进入 handler，错误必须标记为 `before_handler`，避免把未执行说成已执行。
//! - `PostToolUse` 阻断不能把已成功的执行改写成 handler 未执行。
//! - Hook 决策失败（fail closed）与显式 block 走同一条阻断路径，不静默放行。

use crate::hook_lifecycle::{
    evaluate_hook_event_async, RuntimeHookEvaluation, RuntimeHookEventContext,
    RuntimeHookHandlerReport, RuntimeHookReportFuture, RuntimeHookReporter,
};
use crate::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionFuture, RuntimeToolExecutionRequest,
    RuntimeToolExecutor, RuntimeToolPolicyErrorKind,
};
use crate::turn_snapshot::{RuntimeHookEventName, RuntimeHookSnapshot, RuntimeTurnSnapshot};
use serde_json::Value;
use std::sync::Arc;

/// 固定回报，用于测试与确定性回归。
pub struct FixedHookReporter {
    report: Option<RuntimeHookHandlerReport>,
}

impl FixedHookReporter {
    pub fn new(report: Option<RuntimeHookHandlerReport>) -> Self {
        Self { report }
    }
}

impl RuntimeHookReporter for FixedHookReporter {
    fn report<'a>(
        &'a self,
        _hook: &'a RuntimeHookSnapshot,
        _event_name: RuntimeHookEventName,
        _context: &'a RuntimeHookEventContext,
    ) -> RuntimeHookReportFuture<'a> {
        Box::pin(async move { self.report.clone() })
    }
}

/// 用 Hook 门控包装一个既有执行器。
pub struct HookGatedToolExecutor {
    inner: Arc<dyn RuntimeToolExecutor>,
    snapshot: Arc<RuntimeTurnSnapshot>,
    reporter: Arc<dyn RuntimeHookReporter>,
}

impl HookGatedToolExecutor {
    pub fn new(
        inner: Arc<dyn RuntimeToolExecutor>,
        snapshot: Arc<RuntimeTurnSnapshot>,
        reporter: Arc<dyn RuntimeHookReporter>,
    ) -> Self {
        Self {
            inner,
            snapshot,
            reporter,
        }
    }

    fn event_context(request: &RuntimeToolExecutionRequest<'_>) -> RuntimeHookEventContext {
        RuntimeHookEventContext {
            session_id: Some(request.context.session_id().to_string()),
            turn_id: request
                .context
                .tool_identity()
                .map(|identity| identity.turn_id().to_string()),
            tool_call_id: request
                .context
                .tool_identity()
                .map(|identity| identity.call_id().to_string()),
            working_directory: request.context.working_directory().clone(),
            tool_name: Some(request.tool_name.to_string()),
            tool_arguments: Some(request.params.to_string()),
            tool_output: None,
            content: None,
        }
    }

    async fn evaluate(
        &self,
        event_name: RuntimeHookEventName,
        context: &RuntimeHookEventContext,
    ) -> RuntimeHookEvaluation {
        evaluate_hook_event_async(&self.snapshot, event_name, context, self.reporter.as_ref()).await
    }
}

fn blocked_error(
    event_name: RuntimeHookEventName,
    evaluation: &RuntimeHookEvaluation,
) -> RuntimeToolExecutionError {
    let reason = evaluation
        .blocking_reason()
        .unwrap_or_else(|| "hook blocked the tool call".to_string());
    let label = match event_name {
        RuntimeHookEventName::PreToolUse => "pre_tool_use",
        RuntimeHookEventName::PostToolUse => "post_tool_use",
        _ => "hook",
    };
    RuntimeToolExecutionError::new(
        format!("{label} hook blocked this call: {reason}"),
        Some(RuntimeToolPolicyErrorKind::PermissionDenied(reason)),
    )
}

impl RuntimeToolExecutor for HookGatedToolExecutor {
    fn execute<'a>(
        &'a self,
        request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move {
            let context = Self::event_context(&request);
            let pre = self
                .evaluate(RuntimeHookEventName::PreToolUse, &context)
                .await;
            if pre.is_blocked() {
                // 未进入 handler：必须标记 before_handler，否则上层会误判已执行。
                return Err(blocked_error(RuntimeHookEventName::PreToolUse, &pre).before_handler());
            }

            // rewrite 必须是合法 JSON 才能替换参数；非法改写按 fail closed 阻断，
            // 不静默沿用原参数。
            let rewritten = match pre.rewritten_arguments() {
                Some(arguments) => {
                    match serde_json::from_str::<Value>(&arguments) {
                        Ok(value) => Some(value),
                        Err(error) => {
                            return Err(RuntimeToolExecutionError::new(
                            format!("pre_tool_use hook returned invalid rewritten arguments: {error}"),
                            Some(RuntimeToolPolicyErrorKind::SafetyCheckFailed(
                                "invalid_hook_rewrite".to_string(),
                            )),
                        )
                        .before_handler());
                        }
                    }
                }
                None => None,
            };

            let effective_request = RuntimeToolExecutionRequest {
                tool_name: request.tool_name,
                params: rewritten.as_ref().unwrap_or(request.params),
                context: request.context,
                turn_context: request.turn_context,
            };
            let result = self.inner.execute(effective_request).await?;

            let post_context = RuntimeHookEventContext {
                tool_output: Some(
                    serde_json::json!({
                        "success": result.success,
                        "output": result.output,
                        "structuredContent": result.structured_content,
                        "error": result.error,
                    })
                    .to_string(),
                ),
                ..context
            };
            let post = self
                .evaluate(RuntimeHookEventName::PostToolUse, &post_context)
                .await;
            if post.is_blocked() {
                // handler 已经执行过，不能声明 before_handler。
                return Err(blocked_error(RuntimeHookEventName::PostToolUse, &post));
            }
            Ok(result)
        })
    }
}

/// 便捷构造：直接返回 trait object handle 使用的 `Arc`。
pub fn hook_gated_executor(
    inner: Arc<dyn RuntimeToolExecutor>,
    snapshot: Arc<RuntimeTurnSnapshot>,
    reporter: Arc<dyn RuntimeHookReporter>,
) -> Arc<dyn RuntimeToolExecutor> {
    Arc::new(HookGatedToolExecutor::new(inner, snapshot, reporter))
}

/// 由 `PreToolUse` 的 allow 决策收集的模型可见注入上下文。
pub async fn pre_tool_use_injected_context(
    snapshot: &RuntimeTurnSnapshot,
    context: &RuntimeHookEventContext,
    reporter: &dyn RuntimeHookReporter,
) -> Vec<String> {
    evaluate_hook_event_async(
        snapshot,
        RuntimeHookEventName::PreToolUse,
        context,
        reporter,
    )
    .await
    .injected_context()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_definition::{RuntimeToolDefinition, RuntimeToolExposure};
    use crate::tool_executor::{
        RuntimeToolExecutionContext, RuntimeToolExecutionContextInput, RuntimeToolExecutionResult,
    };
    use crate::turn_snapshot::{
        RuntimeHookExecutionMode, RuntimeHookHandlerType, RuntimeHookSnapshot, RuntimeHookSource,
        RuntimeHookTrustStatus, RuntimeToolIdentity, RuntimeToolSnapshot,
    };
    use std::path::PathBuf;
    use std::sync::Mutex;

    struct RecordingExecutor {
        calls: Mutex<Vec<Value>>,
    }

    impl RecordingExecutor {
        fn new() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<Value> {
            self.calls.lock().expect("calls lock").clone()
        }
    }

    impl RuntimeToolExecutor for RecordingExecutor {
        fn execute<'a>(
            &'a self,
            request: RuntimeToolExecutionRequest<'a>,
        ) -> RuntimeToolExecutionFuture<'a> {
            Box::pin(async move {
                self.calls
                    .lock()
                    .expect("calls lock")
                    .push(request.params.clone());
                Ok(RuntimeToolExecutionResult::new(
                    true,
                    "ok".to_string(),
                    None,
                    std::collections::HashMap::new(),
                ))
            })
        }
    }

    fn hook(event_name: RuntimeHookEventName) -> RuntimeHookSnapshot {
        RuntimeHookSnapshot {
            key: format!("project:{event_name:?}:0"),
            event_name,
            handler_type: RuntimeHookHandlerType::Command,
            execution_mode: RuntimeHookExecutionMode::Sync,
            matcher: None,
            command: Some("true".to_string()),
            timeout_sec: 5,
            status_message: None,
            additional_context_limit: None,
            source_path: PathBuf::from("/tmp/hooks.json"),
            source: RuntimeHookSource::Project,
            plugin_id: None,
            display_order: 0,
            enabled: true,
            is_managed: false,
            current_hash: "sha256:test".to_string(),
            trust_status: RuntimeHookTrustStatus::Trusted,
        }
    }

    fn snapshot(hooks: Vec<RuntimeHookSnapshot>) -> Arc<RuntimeTurnSnapshot> {
        let tool = RuntimeToolSnapshot::new(
            RuntimeToolIdentity::plain("shell"),
            RuntimeToolDefinition::new("shell", "run", serde_json::json!({"type": "object"})),
            RuntimeToolExposure::Direct,
            false,
            true,
        );
        Arc::new(RuntimeTurnSnapshot::try_new(vec![tool], hooks).expect("snapshot"))
    }

    async fn run(
        hooks: Vec<RuntimeHookSnapshot>,
        report: Option<RuntimeHookHandlerReport>,
        params: Value,
    ) -> (
        Arc<RecordingExecutor>,
        Result<RuntimeToolExecutionResult, RuntimeToolExecutionError>,
    ) {
        let inner = Arc::new(RecordingExecutor::new());
        let gated = HookGatedToolExecutor::new(
            inner.clone(),
            snapshot(hooks),
            Arc::new(FixedHookReporter::new(report)),
        );
        let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: PathBuf::from("/tmp"),
            session_id: "session-1".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        });
        let result = gated
            .execute(RuntimeToolExecutionRequest {
                tool_name: "shell",
                params: &params,
                context: &context,
                turn_context: None,
            })
            .await;
        (inner, result)
    }

    #[tokio::test]
    async fn pre_tool_use_block_prevents_handler_execution() {
        let (inner, result) = run(
            vec![hook(RuntimeHookEventName::PreToolUse)],
            Some(RuntimeHookHandlerReport::Block {
                reason: "denied".to_string(),
            }),
            serde_json::json!({"cmd": "rm -rf /"}),
        )
        .await;

        let error = result.expect_err("must block");
        assert!(error.message().contains("pre_tool_use hook blocked"));
        assert!(!error.handler_executed());
        assert!(inner.calls().is_empty());
    }

    #[tokio::test]
    async fn missing_hook_decision_blocks_before_handler() {
        let (inner, result) = run(
            vec![hook(RuntimeHookEventName::PreToolUse)],
            None,
            serde_json::json!({}),
        )
        .await;

        assert!(result.is_err());
        assert!(!result.unwrap_err().handler_executed());
        assert!(inner.calls().is_empty());
    }

    #[tokio::test]
    async fn pre_tool_use_rewrite_replaces_arguments_before_execution() {
        let (inner, result) = run(
            vec![hook(RuntimeHookEventName::PreToolUse)],
            Some(RuntimeHookHandlerReport::Rewrite {
                arguments: "{\"cmd\":\"ls\"}".to_string(),
            }),
            serde_json::json!({"cmd": "rm -rf /"}),
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(inner.calls(), vec![serde_json::json!({"cmd": "ls"})]);
    }

    #[tokio::test]
    async fn invalid_rewrite_fails_closed_without_using_original_arguments() {
        let (inner, result) = run(
            vec![hook(RuntimeHookEventName::PreToolUse)],
            Some(RuntimeHookHandlerReport::Rewrite {
                arguments: "not json".to_string(),
            }),
            serde_json::json!({"cmd": "ls"}),
        )
        .await;

        let error = result.expect_err("invalid rewrite must fail closed");
        assert!(error.message().contains("invalid rewritten arguments"));
        assert!(!error.handler_executed());
        assert!(inner.calls().is_empty());
    }

    #[tokio::test]
    async fn post_tool_use_block_keeps_handler_executed_flag() {
        let (inner, result) = run(
            vec![hook(RuntimeHookEventName::PostToolUse)],
            Some(RuntimeHookHandlerReport::Block {
                reason: "audit failed".to_string(),
            }),
            serde_json::json!({}),
        )
        .await;

        let error = result.expect_err("post hook must block");
        assert!(error.message().contains("post_tool_use hook blocked"));
        // handler 已执行，不能声明未执行。
        assert!(error.handler_executed());
        assert_eq!(inner.calls().len(), 1);
    }

    #[tokio::test]
    async fn no_hooks_passes_through_untouched() {
        let (inner, result) = run(Vec::new(), None, serde_json::json!({"cmd": "ls"})).await;

        assert!(result.is_ok());
        assert_eq!(inner.calls(), vec![serde_json::json!({"cmd": "ls"})]);
    }

    #[tokio::test]
    async fn unrelated_event_hooks_do_not_gate_execution() {
        let (inner, result) = run(
            vec![hook(RuntimeHookEventName::SessionEnd)],
            Some(RuntimeHookHandlerReport::Block {
                reason: "should not apply".to_string(),
            }),
            serde_json::json!({}),
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(inner.calls().len(), 1);
    }

    #[tokio::test]
    async fn injected_context_is_collected_from_allow_decisions() {
        let snapshot = snapshot(vec![hook(RuntimeHookEventName::PreToolUse)]);
        let reporter = FixedHookReporter::new(Some(RuntimeHookHandlerReport::Allow {
            additional_context: Some("review the diff".to_string()),
        }));
        let context = RuntimeHookEventContext {
            tool_name: Some("shell".to_string()),
            tool_arguments: Some("{}".to_string()),
            content: None,
            ..RuntimeHookEventContext::default()
        };

        assert_eq!(
            pre_tool_use_injected_context(&snapshot, &context, &reporter).await,
            vec!["review the diff".to_string()]
        );
    }
}
