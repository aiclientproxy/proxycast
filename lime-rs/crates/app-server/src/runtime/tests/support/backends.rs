use super::super::*;

use std::sync::atomic::{AtomicUsize, Ordering};

pub(in crate::runtime::tests) struct CompletedBackend;

#[async_trait]
impl ExecutionBackend for CompletedBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(RuntimeEvent::new(
            "message.delta",
            json!({ "text": "你好！有什么可以帮你的吗？" }),
        ))?;
        sink.emit(RuntimeEvent::new("turn.completed", json!({})))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

pub(in crate::runtime::tests) struct ProviderTraceBackend;

#[async_trait]
impl ExecutionBackend for ProviderTraceBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(RuntimeEvent::new(
            "provider.request.started",
            json!({
                "stage": "request_started",
                "provider": "openai",
                "model": "gpt-4.1",
                "attempt": 1,
                "elapsed_ms": 0,
                "status": "running"
            }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "provider.first_event.received",
            json!({
                "stage": "first_event_received",
                "provider": "openai",
                "model": "gpt-4.1",
                "attempt": 1,
                "elapsed_ms": 1200,
                "provider_request_id": "req-provider-1",
                "provider_request_id_header": "x-request-id",
                "status": "running"
            }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "provider.first_text_delta.received",
            json!({
                "stage": "first_text_delta_received",
                "provider": "openai",
                "model": "gpt-4.1",
                "attempt": 1,
                "elapsed_ms": 1500,
                "text_chars": 4,
                "provider_request_id": "req-provider-1",
                "provider_request_id_header": "x-request-id",
                "status": "running"
            }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "message.delta",
            json!({ "text": "你好！有什么可以帮你的吗？" }),
        ))?;
        sink.emit(RuntimeEvent::new("turn.completed", json!({})))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

pub(in crate::runtime::tests) struct CodingLifecycleBackend;

#[async_trait]
impl ExecutionBackend for CodingLifecycleBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new(
            "file.changed",
            json!({
                "path": "src/App.tsx",
                "artifactId": "artifact_app_tsx"
            }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "patch.started",
            json!({ "patchId": "patch_app_tsx" }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "patch.applied",
            json!({ "patchId": "patch_app_tsx" }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "command.started",
            json!({
                "commandId": "cmd_test",
                "command": "npm test"
            }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "command.output",
            json!({
                "commandId": "cmd_test",
                "outputRef": "output://cmd_test"
            }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "command.exited",
            json!({
                "commandId": "cmd_test",
                "exitCode": 0
            }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "test.started",
            json!({ "testRunId": "test_unit" }),
        ))?;
        sink.emit(RuntimeEvent::new(
            "test.completed",
            json!({
                "testRunId": "test_unit",
                "result": "passed"
            }),
        ))?;
        sink.emit(RuntimeEvent::new("turn.completed", json!({})))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

pub(in crate::runtime::tests) struct InvalidCodingPayloadBackend;

#[async_trait]
impl ExecutionBackend for InvalidCodingPayloadBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new(
            "file.changed",
            json!({ "path": "src/App.tsx" }),
        ))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

pub(in crate::runtime::tests) struct PartialFailureBackend;

#[async_trait]
impl ExecutionBackend for PartialFailureBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        Err(RuntimeCoreError::Backend(
            "provider stream timed out after 60s".to_string(),
        ))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

pub(in crate::runtime::tests) struct UsageLimitFailureBackend;

#[async_trait]
impl ExecutionBackend for UsageLimitFailureBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        Err(RuntimeCoreError::UsageLimitExceeded(
            "provider quota exhausted".to_string(),
        ))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

pub(in crate::runtime::tests) struct FailBeforeEmitBackend {
    pub(in crate::runtime::tests) start_count: AtomicUsize,
}

#[async_trait]
impl ExecutionBackend for FailBeforeEmitBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        if self.start_count.fetch_add(1, Ordering::SeqCst) == 0 {
            return sink.emit(RuntimeEvent::new("turn.accepted", json!({})));
        }
        Err(RuntimeCoreError::Backend(
            "backend unavailable before turn start".to_string(),
        ))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

pub(in crate::runtime::tests) struct HangingCancelBackend {
    pub(in crate::runtime::tests) cancel_count: AtomicUsize,
}

#[async_trait]
impl ExecutionBackend for HangingCancelBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.cancel_count.fetch_add(1, Ordering::SeqCst);
        std::future::pending::<()>().await;
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

pub(in crate::runtime::tests) struct RunningCountingBackend {
    pub(in crate::runtime::tests) start_count: AtomicUsize,
}

#[async_trait]
impl ExecutionBackend for RunningCountingBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.start_count.fetch_add(1, Ordering::SeqCst);
        sink.emit(RuntimeEvent::new("turn.started", json!({})))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

pub(in crate::runtime::tests) struct TurnCompletedRecordingBackend {
    pub(in crate::runtime::tests) requests: Mutex<Vec<ExecutionRequest>>,
}

#[async_trait]
impl ExecutionBackend for TurnCompletedRecordingBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.requests
            .lock()
            .expect("test backend requests mutex poisoned")
            .push(request);
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(RuntimeEvent::new("turn.completed", json!({})))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

#[derive(Default)]
pub(in crate::runtime::tests) struct RecordingBackend {
    pub(in crate::runtime::tests) requests: Mutex<Vec<ExecutionRequest>>,
}

#[async_trait]
impl ExecutionBackend for RecordingBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.requests
            .lock()
            .expect("test backend requests mutex poisoned")
            .push(request);
        sink.emit(RuntimeEvent::new("turn.accepted", json!({})))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}
