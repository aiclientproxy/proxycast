//! Generation-scoped cell identifiers and delegate projection.

use code_mode_protocol::{
    RuntimeCodeModeCellId, RuntimeCodeModeNestedToolCall, RuntimeCodeModeResponse,
    RuntimeCodeModeSessionDelegate, RuntimeCodeModeStartedCell, RuntimeCodeModeWaitOutcome,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

const SEPARATOR: char = ':';

pub(super) fn next_execution_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

pub(super) fn next_wait_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

pub(super) fn public_started_cell(
    generation: u64,
    started: RuntimeCodeModeStartedCell,
) -> RuntimeCodeModeStartedCell {
    let public_cell_id = public_cell_id(generation, &started.cell_id);
    RuntimeCodeModeStartedCell::new(
        public_cell_id,
        Box::pin(async move {
            started
                .initial_response()
                .await
                .map(|response| public_response(generation, response))
        }),
    )
}

pub(super) fn remote_cell_id(
    generation: u64,
    cell_id: &RuntimeCodeModeCellId,
) -> Result<RuntimeCodeModeCellId, String> {
    let value = cell_id.as_str();
    let (prefix, remote) = value
        .split_once(SEPARATOR)
        .ok_or_else(|| "code-mode cell ID is missing its session generation".to_string())?;
    let actual = prefix
        .strip_prefix('g')
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| "code-mode cell ID has an invalid session generation".to_string())?;
    if actual != generation {
        return Err("code-mode cell belongs to a retired gRPC session generation".to_string());
    }
    if remote.trim().is_empty() {
        return Err("code-mode cell ID omitted its remote identifier".to_string());
    }
    Ok(RuntimeCodeModeCellId::new(remote))
}

pub(super) fn public_wait_outcome(
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

fn public_cell_id(generation: u64, remote: &RuntimeCodeModeCellId) -> RuntimeCodeModeCellId {
    RuntimeCodeModeCellId::new(format!("g{generation}{SEPARATOR}{}", remote.as_str()))
}

pub(super) struct GenerationDelegate {
    pub(super) delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    pub(super) generation: u64,
}

impl RuntimeCodeModeSessionDelegate for GenerationDelegate {
    fn invoke_tool<'a>(
        &'a self,
        mut invocation: RuntimeCodeModeNestedToolCall,
        cancellation_token: CancellationToken,
    ) -> code_mode_protocol::RuntimeCodeModeFuture<'a, serde_json::Value> {
        invocation.cell_id = public_cell_id(self.generation, &invocation.cell_id);
        self.delegate.invoke_tool(invocation, cancellation_token)
    }

    fn notify<'a>(
        &'a self,
        tool_call_id: String,
        cell_id: RuntimeCodeModeCellId,
        text: String,
        cancellation_token: CancellationToken,
    ) -> code_mode_protocol::RuntimeCodeModeFuture<'a, ()> {
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

    #[test]
    fn remote_cell_id_rejects_retired_generation() {
        let cell = RuntimeCodeModeCellId::new("g4:12");
        assert_eq!(remote_cell_id(4, &cell).unwrap().as_str(), "12");
        assert!(remote_cell_id(5, &cell).is_err());
    }
}
