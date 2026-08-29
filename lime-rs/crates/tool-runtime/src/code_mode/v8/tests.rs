use super::*;
use crate::code_mode::{RuntimeCodeModeTool, RuntimeCodeModeWaitRequest};
use crate::tool_definition::RuntimeToolDefinition;
use crate::turn_snapshot::RuntimeToolIdentity;
use serde_json::{json, Value};
use std::sync::Mutex;

#[derive(Default)]
struct RecordingDelegate {
    calls: Mutex<Vec<RuntimeCodeModeNestedToolCall>>,
    notifications: Mutex<Vec<String>>,
}

impl RuntimeCodeModeSessionDelegate for RecordingDelegate {
    fn invoke_tool<'a>(
        &'a self,
        invocation: RuntimeCodeModeNestedToolCall,
        _cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, Value> {
        Box::pin(async move {
            self.calls.lock().expect("calls").push(invocation);
            Ok(json!({ "answer": 42 }))
        })
    }

    fn notify<'a>(
        &'a self,
        _tool_call_id: String,
        _cell_id: RuntimeCodeModeCellId,
        text: String,
        _cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, ()> {
        Box::pin(async move {
            self.notifications.lock().expect("notifications").push(text);
            Ok(())
        })
    }

    fn cell_closed(&self, _cell_id: &RuntimeCodeModeCellId) {}
}

fn request(source: &str) -> RuntimeCodeModeExecuteRequest {
    RuntimeCodeModeExecuteRequest {
        tool_call_id: "call-1".to_string(),
        source: source.to_string(),
        enabled_tools: Vec::new(),
        yield_time_ms: Some(1_000),
        max_output_tokens: None,
        cancellation_token: None,
    }
}

async fn session(
    delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
) -> RuntimeCodeModeSessionHandle {
    V8CodeModeSessionProvider
        .create_session(delegate)
        .await
        .expect("session")
}

#[test]
fn linked_v8_has_sandbox_enabled() {
    unsafe extern "C" {
        fn v8__V8__IsSandboxEnabled() -> bool;
    }
    assert!(unsafe { v8__V8__IsSandboxEnabled() });
}

#[tokio::test]
async fn executes_text_in_a_real_v8_isolate() {
    let session = session(Arc::new(RecordingDelegate::default())).await;
    let response = session
        .execute(request("text(40 + 2);"))
        .await
        .expect("execute")
        .initial_response()
        .await
        .expect("response");
    assert_eq!(
        response,
        RuntimeCodeModeResponse::Result {
            cell_id: RuntimeCodeModeCellId::new("1"),
            output: "42".to_string(),
            error_text: None,
        }
    );
}

#[tokio::test]
async fn shares_only_explicitly_stored_values_between_cells() {
    let session = session(Arc::new(RecordingDelegate::default())).await;
    session
        .execute(request("store('answer', 42);"))
        .await
        .expect("store execute")
        .initial_response()
        .await
        .expect("store response");
    let response = session
        .execute(request("text(load('answer'));"))
        .await
        .expect("load execute")
        .initial_response()
        .await
        .expect("load response");
    assert_eq!(response.cell_id().as_str(), "2");
    assert!(matches!(
        response,
        RuntimeCodeModeResponse::Result { output, error_text: None, .. } if output == "42"
    ));
}

#[tokio::test]
async fn routes_nested_tools_and_notifications_through_the_delegate() {
    let delegate = Arc::new(RecordingDelegate::default());
    let session = session(delegate.clone()).await;
    let mut request = request(
        "const result = await tools.lookup({ value: 41 }); notify('working'); text(result.answer);",
    );
    request.enabled_tools.push(RuntimeCodeModeTool {
        identity: RuntimeToolIdentity::plain("lookup"),
        definition: RuntimeToolDefinition::new(
            "lookup",
            "Returns an answer.",
            json!({ "type": "object" }),
        ),
        code_name: "lookup".to_string(),
        global_name: "lookup".to_string(),
    });
    let response = session
        .execute(request)
        .await
        .expect("execute")
        .initial_response()
        .await
        .expect("response");
    assert!(matches!(
        response,
        RuntimeCodeModeResponse::Result { output, error_text: None, .. } if output == "42"
    ));
    let calls = delegate.calls.lock().expect("calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tool_name, "lookup");
    assert_eq!(calls[0].input, Some(json!({ "value": 41 })));
    assert_eq!(
        delegate
            .notifications
            .lock()
            .expect("notifications")
            .as_slice(),
        ["working"]
    );
}

#[tokio::test]
async fn routes_sequential_nested_tools_in_the_same_cell() {
    let delegate = Arc::new(RecordingDelegate::default());
    let session = session(delegate.clone()).await;
    let mut request = request(
        "const first = await tools.lookup({ value: 1 }); const second = await tools.lookup({ value: 2 }); text(first.answer + second.answer);",
    );
    request.enabled_tools.push(RuntimeCodeModeTool {
        identity: RuntimeToolIdentity::plain("lookup"),
        definition: RuntimeToolDefinition::new(
            "lookup",
            "Returns an answer.",
            json!({ "type": "object" }),
        ),
        code_name: "lookup".to_string(),
        global_name: "lookup".to_string(),
    });

    let response = tokio::time::timeout(
        Duration::from_secs(1),
        session
            .execute(request)
            .await
            .expect("execute")
            .initial_response(),
    )
    .await
    .expect("sequential nested tools must not block")
    .expect("response");

    assert!(matches!(
        response,
        RuntimeCodeModeResponse::Result { output, error_text: None, .. } if output == "84"
    ));
    let calls = delegate.calls.lock().expect("calls");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].input, Some(json!({ "value": 1 })));
    assert_eq!(calls[1].input, Some(json!({ "value": 2 })));
}

#[tokio::test]
async fn yields_waits_and_terminates_cells() {
    let session = session(Arc::new(RecordingDelegate::default())).await;
    let mut delayed =
        request("await new Promise(resolve => setTimeout(resolve, 25)); text('completed');");
    delayed.yield_time_ms = Some(1);
    let yielded = session
        .execute(delayed)
        .await
        .expect("execute")
        .initial_response()
        .await
        .expect("yield");
    assert!(matches!(yielded, RuntimeCodeModeResponse::Yielded { .. }));
    let waited = session
        .wait(RuntimeCodeModeWaitRequest {
            cell_id: RuntimeCodeModeCellId::new("1"),
            yield_time_ms: 1_000,
        })
        .await
        .expect("wait")
        .into_response();
    assert!(matches!(
        waited,
        RuntimeCodeModeResponse::Result { output, error_text: None, .. } if output == "completed"
    ));

    let mut infinite = request("while (true) {}");
    infinite.yield_time_ms = Some(1);
    let running = session.execute(infinite).await.expect("infinite execute");
    let cell_id = running.cell_id.clone();
    assert!(matches!(
        running.initial_response().await.expect("infinite yield"),
        RuntimeCodeModeResponse::Yielded { .. }
    ));
    assert!(matches!(
        session
            .terminate(cell_id)
            .await
            .expect("terminate")
            .into_response(),
        RuntimeCodeModeResponse::Terminated { .. }
    ));
}
