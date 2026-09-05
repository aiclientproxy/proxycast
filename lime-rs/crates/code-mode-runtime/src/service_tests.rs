use super::*;
use code_mode_protocol::{RuntimeCodeModeSession, RuntimeCodeModeSessionDelegate};
use serde_json::Value;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

struct NoopDelegate;

impl RuntimeCodeModeSessionDelegate for NoopDelegate {
    fn invoke_tool<'a>(
        &'a self,
        _invocation: RuntimeCodeModeNestedToolCall,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, Value> {
        Box::pin(async move {
            cancellation_token.cancelled().await;
            Err("nested tools are disabled in this test".to_string())
        })
    }

    fn notify<'a>(
        &'a self,
        _tool_call_id: String,
        _cell_id: RuntimeCodeModeCellId,
        _text: String,
        _cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

    fn cell_closed(&self, _cell_id: &RuntimeCodeModeCellId) {}
}

fn request(source: &str) -> RuntimeCodeModeExecuteRequest {
    RuntimeCodeModeExecuteRequest {
        tool_call_id: "service-test".to_string(),
        source: source.to_string(),
        enabled_tools: Vec::new(),
        yield_time_ms: Some(1_000),
        max_output_tokens: None,
        cancellation_token: None,
    }
}

#[tokio::test]
async fn in_process_session_executes_javascript_and_returns_result() {
    let session = InProcessCodeModeSession::with_delegate(Arc::new(NoopDelegate));
    let started = session
        .execute(request("text(40 + 2);"))
        .await
        .expect("start cell");
    let response = started.initial_response().await.expect("cell response");
    assert!(matches!(
        response,
        RuntimeCodeModeResponse::Result { content_items, error_text: None, .. }
            if content_items == vec![FunctionCallOutputContentItem::InputText { text: "42".to_string() }]
    ));
}

#[tokio::test]
async fn in_process_session_reports_missing_cell_as_terminal_result() {
    let session = InProcessCodeModeSession::with_delegate(Arc::new(NoopDelegate));
    let outcome = session
        .wait(RuntimeCodeModeWaitRequest {
            cell_id: RuntimeCodeModeCellId::new("missing"),
            yield_time_ms: 1,
        })
        .await
        .expect("missing cell outcome");
    assert!(matches!(
        outcome,
        RuntimeCodeModeWaitOutcome::MissingCell(RuntimeCodeModeResponse::Result {
            error_text: Some(_),
            ..
        })
    ));
}
