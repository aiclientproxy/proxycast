//! Cell identity helpers for the process connection driver.
//!
//! Public cell identity and callback projection for process reconnections.

use code_mode_protocol::{
    RuntimeCodeModeCellId, RuntimeCodeModeFuture, RuntimeCodeModeNestedToolCall,
    RuntimeCodeModeResponse, RuntimeCodeModeSessionDelegate, RuntimeCodeModeStartedCell,
    RuntimeCodeModeWaitOutcome,
};
use serde_json::Value;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub(crate) fn public_cell_id(
    generation: u64,
    cell_id: &RuntimeCodeModeCellId,
) -> RuntimeCodeModeCellId {
    if generation == 1 {
        RuntimeCodeModeCellId::new(cell_id.as_str())
    } else {
        RuntimeCodeModeCellId::new(format!("g{generation}:{}", cell_id.as_str()))
    }
}

pub(crate) fn remote_cell_id(
    generation: u64,
    cell_id: &RuntimeCodeModeCellId,
) -> Result<RuntimeCodeModeCellId, String> {
    if generation == 1 {
        return Ok(RuntimeCodeModeCellId::new(cell_id.as_str()));
    }
    let prefix = format!("g{generation}:");
    let Some(value) = cell_id.as_str().strip_prefix(&prefix) else {
        return Err(format!(
            "cell {cell_id} belongs to a stale code-mode host generation"
        ));
    };
    if value.is_empty() {
        return Err("code-mode cell ID omitted its remote identifier".to_string());
    }
    Ok(RuntimeCodeModeCellId::new(value))
}

pub(crate) fn public_started_cell(
    generation: u64,
    started: RuntimeCodeModeStartedCell,
) -> RuntimeCodeModeStartedCell {
    let cell_id = public_cell_id(generation, &started.cell_id);
    RuntimeCodeModeStartedCell::new(
        cell_id,
        Box::pin(async move {
            started
                .initial_response()
                .await
                .map(|response| public_response(generation, response))
        }),
    )
}

pub(crate) fn public_wait_outcome(
    generation: u64,
    outcome: RuntimeCodeModeWaitOutcome,
) -> RuntimeCodeModeWaitOutcome {
    match outcome {
        RuntimeCodeModeWaitOutcome::LiveCell(response) => {
            RuntimeCodeModeWaitOutcome::LiveCell(public_response(generation, response))
        }
        RuntimeCodeModeWaitOutcome::MissingCell(response) => {
            RuntimeCodeModeWaitOutcome::MissingCell(public_response(generation, response))
        }
    }
}

fn public_response(generation: u64, response: RuntimeCodeModeResponse) -> RuntimeCodeModeResponse {
    match response {
        RuntimeCodeModeResponse::Yielded {
            cell_id,
            content_items,
            code_mode_host_duration,
        } => RuntimeCodeModeResponse::Yielded {
            cell_id: public_cell_id(generation, &cell_id),
            content_items,
            code_mode_host_duration,
        },
        RuntimeCodeModeResponse::Terminated {
            cell_id,
            content_items,
            code_mode_host_duration,
        } => RuntimeCodeModeResponse::Terminated {
            cell_id: public_cell_id(generation, &cell_id),
            content_items,
            code_mode_host_duration,
        },
        RuntimeCodeModeResponse::Result {
            cell_id,
            content_items,
            error_text,
            code_mode_host_duration,
        } => RuntimeCodeModeResponse::Result {
            cell_id: public_cell_id(generation, &cell_id),
            content_items,
            error_text,
            code_mode_host_duration,
        },
    }
}

pub(crate) struct GenerationDelegate {
    pub(crate) delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    pub(crate) generation: u64,
}

impl RuntimeCodeModeSessionDelegate for GenerationDelegate {
    fn invoke_tool<'a>(
        &'a self,
        mut invocation: RuntimeCodeModeNestedToolCall,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, Value> {
        invocation.cell_id = public_cell_id(self.generation, &invocation.cell_id);
        self.delegate.invoke_tool(invocation, cancellation_token)
    }

    fn notify<'a>(
        &'a self,
        tool_call_id: String,
        cell_id: RuntimeCodeModeCellId,
        text: String,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, ()> {
        self.delegate.notify(
            tool_call_id,
            public_cell_id(self.generation, &cell_id),
            text,
            cancellation_token,
        )
    }

    fn cell_closed(&self, cell_id: &RuntimeCodeModeCellId) {
        self.delegate
            .cell_closed(&public_cell_id(self.generation, cell_id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;

    #[test]
    fn first_generation_keeps_plain_ids_and_reconnects_are_scoped() {
        assert_eq!(
            public_cell_id(1, &RuntimeCodeModeCellId::new("7")).as_str(),
            "7"
        );
        assert_eq!(
            public_cell_id(2, &RuntimeCodeModeCellId::new("7")).as_str(),
            "g2:7"
        );
        assert_eq!(
            remote_cell_id(1, &RuntimeCodeModeCellId::new("g2:7"))
                .expect("opaque first generation ID")
                .as_str(),
            "g2:7"
        );
        assert_eq!(
            remote_cell_id(1, &RuntimeCodeModeCellId::new("7"))
                .expect("first generation ID")
                .as_str(),
            "7"
        );
        assert_eq!(
            remote_cell_id(2, &RuntimeCodeModeCellId::new("g2:7"))
                .expect("current generation ID")
                .as_str(),
            "7"
        );
        assert!(remote_cell_id(2, &RuntimeCodeModeCellId::new("g1:7")).is_err());
        assert!(remote_cell_id(2, &RuntimeCodeModeCellId::new("7")).is_err());
    }

    #[tokio::test]
    async fn initial_and_wait_responses_are_projected_to_the_same_generation() {
        let response = RuntimeCodeModeResponse::Result {
            cell_id: RuntimeCodeModeCellId::new("7"),
            content_items: Vec::new(),
            error_text: None,
            code_mode_host_duration: None,
        };
        let started = RuntimeCodeModeStartedCell::new(
            RuntimeCodeModeCellId::new("7"),
            Box::pin(async move { Ok(response) }),
        );
        let started = public_started_cell(3, started);
        assert_eq!(started.cell_id.as_str(), "g3:7");
        assert_eq!(
            started
                .initial_response()
                .await
                .expect("initial response")
                .cell_id()
                .as_str(),
            "g3:7"
        );

        let outcome = public_wait_outcome(
            3,
            RuntimeCodeModeWaitOutcome::LiveCell(RuntimeCodeModeResponse::Yielded {
                cell_id: RuntimeCodeModeCellId::new("8"),
                content_items: Vec::new(),
                code_mode_host_duration: None,
            }),
        );
        assert_eq!(outcome.into_response().cell_id().as_str(), "g3:8");
    }

    #[tokio::test]
    async fn remapping_preserves_host_duration_on_initial_and_wait_paths() {
        let duration = Some(Duration::from_millis(12));
        let response = RuntimeCodeModeResponse::Yielded {
            cell_id: RuntimeCodeModeCellId::new("7"),
            content_items: Vec::new(),
            code_mode_host_duration: duration,
        };
        let started = RuntimeCodeModeStartedCell::new(
            RuntimeCodeModeCellId::new("7"),
            Box::pin({
                let response = response.clone();
                async move { Ok(response) }
            }),
        );
        let initial = public_started_cell(2, started)
            .initial_response()
            .await
            .expect("initial response");
        assert_eq!(initial.cell_id().as_str(), "g2:7");
        assert_eq!(initial.code_mode_host_duration(), duration);

        let waited = public_wait_outcome(2, RuntimeCodeModeWaitOutcome::LiveCell(response));
        let waited = waited.into_response();
        assert_eq!(waited.cell_id().as_str(), "g2:7");
        assert_eq!(waited.code_mode_host_duration(), duration);
    }

    #[derive(Default)]
    struct RecordingDelegate {
        invocations: Mutex<Vec<RuntimeCodeModeNestedToolCall>>,
        notifications: Mutex<Vec<RuntimeCodeModeCellId>>,
        closed: Mutex<Vec<RuntimeCodeModeCellId>>,
    }

    impl RuntimeCodeModeSessionDelegate for RecordingDelegate {
        fn invoke_tool<'a>(
            &'a self,
            invocation: RuntimeCodeModeNestedToolCall,
            _cancellation_token: CancellationToken,
        ) -> RuntimeCodeModeFuture<'a, Value> {
            Box::pin(async move {
                self.invocations
                    .lock()
                    .expect("invocations")
                    .push(invocation);
                Ok(Value::Null)
            })
        }

        fn notify<'a>(
            &'a self,
            _tool_call_id: String,
            cell_id: RuntimeCodeModeCellId,
            _text: String,
            _cancellation_token: CancellationToken,
        ) -> RuntimeCodeModeFuture<'a, ()> {
            Box::pin(async move {
                self.notifications
                    .lock()
                    .expect("notifications")
                    .push(cell_id);
                Ok(())
            })
        }

        fn cell_closed(&self, cell_id: &RuntimeCodeModeCellId) {
            self.closed
                .lock()
                .expect("closed cells")
                .push(cell_id.clone());
        }
    }

    #[tokio::test]
    async fn delegate_callbacks_are_projected_to_the_current_generation() {
        let delegate = Arc::new(RecordingDelegate::default());
        let generation = GenerationDelegate {
            delegate: delegate.clone(),
            generation: 4,
        };
        generation
            .invoke_tool(
                RuntimeCodeModeNestedToolCall {
                    cell_id: RuntimeCodeModeCellId::new("9"),
                    runtime_tool_call_id: "call".to_string(),
                    tool_name: "tool".to_string(),
                    kind: Default::default(),
                    input: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect("invoke callback");
        generation
            .notify(
                "call".to_string(),
                RuntimeCodeModeCellId::new("9"),
                "working".to_string(),
                CancellationToken::new(),
            )
            .await
            .expect("notify callback");
        generation.cell_closed(&RuntimeCodeModeCellId::new("9"));

        assert_eq!(
            delegate.invocations.lock().expect("invocations")[0]
                .cell_id
                .as_str(),
            "g4:9"
        );
        assert_eq!(
            delegate.notifications.lock().expect("notifications")[0].as_str(),
            "g4:9"
        );
        assert_eq!(
            delegate.closed.lock().expect("closed cells")[0].as_str(),
            "g4:9"
        );
    }
}
