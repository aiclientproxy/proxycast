//! gRPC operation identifiers and request policy.

use super::{conversion, GrpcCodeModeSession};
use code_mode_protocol::grpc::{self as proto, code_mode_host_client::CodeModeHostClient};
use code_mode_protocol::{
    RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest, RuntimeCodeModeStartedCell,
    RuntimeCodeModeWaitOutcome, RuntimeCodeModeWaitRequest,
};
use std::sync::atomic::Ordering;

pub(super) async fn execute(
    session: &GrpcCodeModeSession,
    request: RuntimeCodeModeExecuteRequest,
) -> Result<RuntimeCodeModeStartedCell, String> {
    if matches!(
        super::state::lifecycle(&session.closed),
        super::state::SessionState::Closed
    ) {
        return Err("code-mode session is closed".to_string());
    }
    let execution_id = super::generation::next_execution_id();
    let mut enabled_tools = Vec::with_capacity(request.enabled_tools.len());
    for tool in request.enabled_tools {
        enabled_tools.push(proto::ToolDefinition {
            name: tool.global_name,
            tool_name: Some(proto::ToolName {
                name: tool.identity.name,
                namespace: tool.identity.namespace,
            }),
            description: tool.definition.description,
            kind: match tool.kind {
                code_mode_protocol::CodeModeToolKind::Function => proto::ToolKind::Function as i32,
                code_mode_protocol::CodeModeToolKind::Freeform => proto::ToolKind::Freeform as i32,
            },
            input_schema_json: Some(
                serde_json::to_vec(&tool.definition.input_schema)
                    .map_err(|error| error.to_string())?,
            ),
            output_schema_json: None,
        });
    }
    let cancellation = request.cancellation_token.clone();
    let mut client = CodeModeHostClient::new(session.channel.clone());
    let execute = client.execute(proto::ExecuteRequest {
        session_id: session.session_id.clone(),
        execution_id,
        tool_call_id: request.tool_call_id,
        source: request.source,
        enabled_tools,
        yield_time_ms: request.yield_time_ms,
        max_output_tokens: request
            .max_output_tokens
            .and_then(|value| u64::try_from(value).ok()),
    });
    let response = if let Some(cancellation) = cancellation.clone() {
        tokio::select! {
            response = execute => response,
            _ = cancellation.cancelled() => return Err("code-mode gRPC execute cancelled".to_string()),
        }
    } else {
        execute.await
    }
    .map_err(|error| format!("code-mode gRPC execute failed: {error}"));
    let response = match response {
        Ok(response) => response,
        Err(error) => {
            session.mark_closed().await;
            return Err(error);
        }
    };
    let mut stream = response.into_inner();
    let started = stream
        .message()
        .await
        .map_err(|error| format!("failed to read code-mode execution start: {error}"))?
        .ok_or_else(|| "code-mode gRPC execute returned no start event".to_string())?;
    let cell_id = match started.event {
        Some(proto::execute_event::Event::Started(started)) => {
            RuntimeCodeModeCellId::new(started.cell_id)
        }
        _ => return Err("code-mode gRPC execute returned an invalid start event".to_string()),
    };
    let initial_cell_id = cell_id.clone();
    let channel = session.channel.clone();
    let session_id = session.session_id.clone();
    let initial = Box::pin(async move {
        let read_outcome = async {
            let outcome = super::completion::read_outcome(&mut stream).await?;
            conversion::response_from_outcome(outcome)
        };
        if let Some(cancellation) = cancellation {
            tokio::select! {
                result = read_outcome => result,
                _ = cancellation.cancelled() => {
                    let _ = CodeModeHostClient::new(channel).terminate(proto::TerminateRequest {
                        session_id,
                        cell_id: initial_cell_id.to_string(),
                    }).await;
                    Err("code-mode gRPC execute cancelled".to_string())
                }
            }
        } else {
            read_outcome.await
        }
    });
    Ok(RuntimeCodeModeStartedCell::new(cell_id, initial))
}

pub(super) async fn wait(
    session: &GrpcCodeModeSession,
    request: RuntimeCodeModeWaitRequest,
) -> Result<RuntimeCodeModeWaitOutcome, String> {
    let response = CodeModeHostClient::new(session.channel.clone())
        .wait(proto::WaitRequest {
            session_id: session.session_id.clone(),
            cell_id: request.cell_id.to_string(),
            wait_id: super::generation::next_wait_id(),
            yield_time_ms: request.yield_time_ms,
        })
        .await
        .map_err(|error| format!("code-mode gRPC wait failed: {error}"));
    let response = match response {
        Ok(response) => response.into_inner(),
        Err(error) => {
            session.mark_closed().await;
            return Err(error);
        }
    };
    let state = response
        .state
        .ok_or_else(|| "code-mode gRPC wait returned no state".to_string())?;
    match state {
        proto::wait_response::State::LiveCell(outcome) => {
            conversion::response_from_outcome(outcome).map(RuntimeCodeModeWaitOutcome::LiveCell)
        }
        proto::wait_response::State::MissingCell(outcome) => {
            conversion::response_from_outcome(outcome).map(RuntimeCodeModeWaitOutcome::MissingCell)
        }
    }
}

pub(super) async fn terminate(
    session: &GrpcCodeModeSession,
    cell_id: RuntimeCodeModeCellId,
) -> Result<RuntimeCodeModeWaitOutcome, String> {
    let response = CodeModeHostClient::new(session.channel.clone())
        .terminate(proto::TerminateRequest {
            session_id: session.session_id.clone(),
            cell_id: cell_id.to_string(),
        })
        .await
        .map_err(|error| format!("code-mode gRPC terminate failed: {error}"));
    let response = match response {
        Ok(response) => response.into_inner(),
        Err(error) => {
            session.mark_closed().await;
            return Err(error);
        }
    };
    let outcome = response
        .state
        .ok_or_else(|| "code-mode gRPC terminate returned no state".to_string())?;
    match outcome {
        proto::wait_response::State::LiveCell(outcome) => {
            conversion::response_from_outcome(outcome).map(RuntimeCodeModeWaitOutcome::LiveCell)
        }
        proto::wait_response::State::MissingCell(outcome) => {
            conversion::response_from_outcome(outcome).map(RuntimeCodeModeWaitOutcome::MissingCell)
        }
    }
}

pub(super) async fn shutdown(session: &GrpcCodeModeSession) -> Result<(), String> {
    if session.closed.swap(true, Ordering::AcqRel) {
        return Ok(());
    }
    super::deadline::request(
        "code-mode gRPC close session",
        std::time::Duration::ZERO,
        CodeModeHostClient::new(session.channel.clone()).close_session(
            proto::CloseSessionRequest {
                session_id: session.session_id.clone(),
            },
        ),
    )
    .await?;
    Ok(())
}
