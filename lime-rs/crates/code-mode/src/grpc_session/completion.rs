//! Execution completion helpers.

use code_mode_protocol::grpc as proto;

pub(super) async fn read_outcome(
    stream: &mut tonic::Streaming<proto::ExecuteEvent>,
) -> Result<proto::ExecutionOutcome, String> {
    let event = stream
        .message()
        .await
        .map_err(|error| format!("failed to read code-mode execution outcome: {error}"))?
        .ok_or_else(|| "code-mode gRPC execute returned no outcome".to_string())?;
    decode_outcome_event(event)
}

pub(super) fn decode_outcome_event(
    event: proto::ExecuteEvent,
) -> Result<proto::ExecutionOutcome, String> {
    match event.event {
        Some(proto::execute_event::Event::Outcome(outcome)) => Ok(outcome),
        _ => Err("code-mode gRPC execute returned an invalid outcome".to_string()),
    }
}
