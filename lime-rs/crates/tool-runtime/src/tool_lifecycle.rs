use crate::tool_call::{ToolCall, ToolEnvironment};
use crate::tool_result_projection::NormalizedToolOutput;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

pub type ToolLifecycleEmissionFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolLifecyclePhase {
    Started,
    Completed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolLifecycleEvent {
    pub turn_id: String,
    pub call_id: String,
    pub tool_name: String,
    pub arguments: Value,
    pub provider_metadata: Value,
    pub environments: Vec<ToolEnvironment>,
    pub phase: ToolLifecyclePhase,
    pub output: Option<NormalizedToolOutput>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutputDeltaEvent {
    pub turn_id: String,
    pub call_id: String,
    pub tool_name: String,
    pub delta: String,
    pub output_kind: Option<String>,
    pub metadata: HashMap<String, Value>,
}

impl ToolLifecycleEvent {
    pub fn started(call: &ToolCall) -> Self {
        Self::from_call(call, ToolLifecyclePhase::Started, None)
    }

    pub fn completed(call: &ToolCall, output: NormalizedToolOutput) -> Self {
        Self::from_call(call, ToolLifecyclePhase::Completed, Some(output))
    }

    fn from_call(
        call: &ToolCall,
        phase: ToolLifecyclePhase,
        output: Option<NormalizedToolOutput>,
    ) -> Self {
        Self {
            turn_id: call.turn_id().to_string(),
            call_id: call.call_id().to_string(),
            tool_name: call.tool_name().to_string(),
            arguments: call.arguments().clone(),
            provider_metadata: call.provider_metadata().clone(),
            environments: call.environments().to_vec(),
            phase,
            output,
        }
    }
}

/// Host capability that publishes canonical tool lifecycle events.
pub trait ToolLifecycleEmitter: Send + Sync {
    fn emit<'a>(&'a self, event: ToolLifecycleEvent) -> ToolLifecycleEmissionFuture<'a>;

    fn emit_output_delta<'a>(
        &'a self,
        _event: ToolOutputDeltaEvent,
    ) -> ToolLifecycleEmissionFuture<'a> {
        Box::pin(async {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_call::{ToolCall, ToolEnvironment};
    use crate::tool_definition::{RuntimeToolDefinition, RuntimeToolExposure};
    use crate::tool_executor::{
        RuntimeToolExecutionContext, RuntimeToolExecutionContextInput, RuntimeToolExecutionFuture,
        RuntimeToolExecutionRequest, RuntimeToolExecutionResult, RuntimeToolExecutor,
        RuntimeToolExecutorHandle,
    };
    use crate::tool_io::{
        ToolIoPayloadStats, ToolOutputReference, ToolOutputTruncation, ToolOutputTruncationReason,
    };
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use tokio_util::sync::CancellationToken;

    #[derive(Default)]
    struct RecordingEmitter {
        events: Mutex<Vec<ToolLifecycleEvent>>,
    }

    impl RecordingEmitter {
        fn events(&self) -> Vec<ToolLifecycleEvent> {
            self.events.lock().expect("recording emitter lock").clone()
        }
    }

    impl ToolLifecycleEmitter for RecordingEmitter {
        fn emit<'a>(&'a self, event: ToolLifecycleEvent) -> ToolLifecycleEmissionFuture<'a> {
            Box::pin(async move {
                self.events
                    .lock()
                    .expect("recording emitter lock")
                    .push(event);
            })
        }
    }

    struct StructuredExecutor;

    struct PendingExecutor;

    fn structured_result(text: impl Into<String>) -> RuntimeToolExecutionResult {
        RuntimeToolExecutionResult::new(true, text.into(), None, HashMap::new())
            .with_structured_content(serde_json::json!({ "rows": 3 }))
            .with_truncation(ToolOutputTruncation::new(
                ToolOutputTruncationReason::PayloadOffloaded,
                ToolIoPayloadStats {
                    chars: 12_000,
                    bytes: 12_000,
                    tokens: 3_000,
                },
            ))
            .with_sidecar_reference(ToolOutputReference::new(
                "sidecar://tool-output-1",
                Some("preview".to_string()),
            ))
    }

    impl RuntimeToolExecutor for StructuredExecutor {
        fn execute<'a>(
            &'a self,
            _request: RuntimeToolExecutionRequest<'a>,
        ) -> RuntimeToolExecutionFuture<'a> {
            Box::pin(async move { Ok(structured_result("preview")) })
        }

        fn execute_call<'a>(
            &'a self,
            call: &'a ToolCall,
            _context: &'a RuntimeToolExecutionContext,
            _turn_context: Option<&'a crate::tool_executor::RuntimeToolTurnContext>,
        ) -> RuntimeToolExecutionFuture<'a> {
            Box::pin(async move {
                let environment_id = call
                    .environments()
                    .first()
                    .map(|environment| environment.environment_id.as_str())
                    .unwrap_or("none");
                Ok(structured_result(format!(
                    "{}@{environment_id}",
                    call.call_id()
                )))
            })
        }
    }

    impl RuntimeToolExecutor for PendingExecutor {
        fn execute<'a>(
            &'a self,
            _request: RuntimeToolExecutionRequest<'a>,
        ) -> RuntimeToolExecutionFuture<'a> {
            Box::pin(std::future::pending())
        }
    }

    #[tokio::test]
    async fn canonical_tool_contract_binds_spec_executor_and_lifecycle() {
        let emitter = Arc::new(RecordingEmitter::default());
        let call = ToolCall::new(
            "turn-1",
            "call-1",
            "inspect",
            serde_json::json!({ "path": "README.md" }),
            vec![ToolEnvironment::new(
                "local",
                PathBuf::from("/tmp/workspace"),
            )],
            emitter.clone(),
        );
        let runtime = RuntimeToolExecutorHandle::new(Arc::new(StructuredExecutor)).bind(
            RuntimeToolDefinition::new(
                "inspect",
                "Inspect one path",
                serde_json::json!({ "type": "object" }),
            ),
            RuntimeToolExposure::Deferred,
        );
        let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: PathBuf::from("/tmp/workspace"),
            session_id: "session-1".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        });

        let output = runtime.execute_call(&call, &context, None).await;

        assert_eq!(runtime.definition().name, "inspect");
        assert_eq!(runtime.exposure(), RuntimeToolExposure::Deferred);
        assert!(output.success);
        assert_eq!(output.text, "call-1@local");
        assert_eq!(
            output.structured_content,
            Some(serde_json::json!({ "rows": 3 }))
        );
        assert_eq!(
            output
                .truncation
                .as_ref()
                .map(|truncation| truncation.reason),
            Some(ToolOutputTruncationReason::PayloadOffloaded)
        );
        assert_eq!(
            output
                .sidecar_reference
                .as_ref()
                .map(|sidecar| sidecar.reference.as_str()),
            Some("sidecar://tool-output-1")
        );
        assert_eq!(
            output
                .metadata
                .get(crate::tool_result_projection::TOOL_HANDLER_EXECUTED_METADATA_KEY),
            Some(&Value::Bool(true))
        );

        let events = emitter.events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].phase, ToolLifecyclePhase::Started);
        assert_eq!(events[0].turn_id, "turn-1");
        assert_eq!(events[0].call_id, "call-1");
        assert_eq!(events[0].environments[0].environment_id, "local");
        assert_eq!(events[1].phase, ToolLifecyclePhase::Completed);
        assert_eq!(events[1].output, Some(output));
    }

    #[tokio::test]
    async fn canonical_tool_contract_completes_name_mismatch_as_failure() {
        let emitter = Arc::new(RecordingEmitter::default());
        let call = ToolCall::new(
            "turn-2",
            "call-2",
            "other",
            serde_json::json!({}),
            Vec::new(),
            emitter.clone(),
        );
        let runtime = RuntimeToolExecutorHandle::new(Arc::new(StructuredExecutor)).bind(
            RuntimeToolDefinition::new(
                "inspect",
                "Inspect one path",
                serde_json::json!({ "type": "object" }),
            ),
            RuntimeToolExposure::Hidden,
        );
        let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: PathBuf::from("/tmp/workspace"),
            session_id: "session-2".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        });

        let output = runtime.execute_call(&call, &context, None).await;

        assert!(!output.success);
        assert!(output.text.contains("does not match bound runtime"));
        assert_eq!(
            output
                .metadata
                .get(crate::tool_result_projection::TOOL_HANDLER_EXECUTED_METADATA_KEY),
            Some(&Value::Bool(false))
        );
        let events = emitter.events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].phase, ToolLifecyclePhase::Started);
        assert_eq!(events[1].phase, ToolLifecyclePhase::Completed);
        assert_eq!(events[1].output, Some(output));
    }

    #[tokio::test]
    async fn canonical_tool_contract_projects_inflight_cancel_as_aborted() {
        let emitter = Arc::new(RecordingEmitter::default());
        let call = ToolCall::new(
            "turn-cancel",
            "call-cancel",
            "wait",
            serde_json::json!({}),
            Vec::new(),
            emitter.clone(),
        );
        let runtime = RuntimeToolExecutorHandle::new(Arc::new(PendingExecutor)).bind(
            RuntimeToolDefinition::new(
                "wait",
                "Wait until cancellation",
                serde_json::json!({ "type": "object" }),
            ),
            RuntimeToolExposure::Direct,
        );
        let cancel_token = CancellationToken::new();
        let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: PathBuf::from("/tmp/workspace"),
            session_id: "session-cancel".to_string(),
            cancel_token: Some(cancel_token.clone()),
            workspace_sandbox: None,
        });
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            cancel_token.cancel();
        });

        let output = runtime.execute_call(&call, &context, None).await;

        assert!(!output.success);
        assert_eq!(output.error, None);
        assert_eq!(
            output
                .metadata
                .get(crate::tool_result_projection::TOOL_OUTCOME_METADATA_KEY),
            Some(&Value::String(
                crate::tool_result_projection::TOOL_OUTCOME_ABORTED.to_string()
            ))
        );
        let events = emitter.events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].phase, ToolLifecyclePhase::Completed);
        assert_eq!(events[1].output, Some(output));
    }
}
