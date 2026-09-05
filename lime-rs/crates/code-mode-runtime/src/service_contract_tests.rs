use super::{create_cell_request, runtime_response};
use crate::session_runtime::{CellEvent, ToolKind};
use code_mode_protocol::{
    CodeModeToolKind, RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest,
    RuntimeCodeModeResponse, RuntimeCodeModeTool, RuntimeToolDefinition, RuntimeToolIdentity,
};
use serde_json::json;

#[test]
fn execute_contract_preserves_freeform_tool_kind() {
    let request = RuntimeCodeModeExecuteRequest {
        tool_call_id: "call-1".to_string(),
        source: "text(1);".to_string(),
        enabled_tools: vec![RuntimeCodeModeTool {
            identity: RuntimeToolIdentity::plain("shell"),
            definition: RuntimeToolDefinition::new("shell", "run shell", json!({})),
            kind: CodeModeToolKind::Freeform,
            code_name: "shell".to_string(),
            global_name: "shell".to_string(),
        }],
        yield_time_ms: Some(1),
        max_output_tokens: None,
        cancellation_token: None,
    };
    let cell = create_cell_request(request);
    assert_eq!(cell.enabled_tools[0].kind, ToolKind::Freeform);
}

#[test]
fn runtime_response_rejects_pending_frontier() {
    let result = runtime_response(
        RuntimeCodeModeCellId::new("1"),
        CellEvent::Pending {
            content_items: Vec::new(),
            pending_tool_call_ids: vec!["tool-1".to_string()],
        },
    );
    assert!(result.is_err());
}

#[test]
fn runtime_response_maps_terminal_result() {
    let result = runtime_response(
        RuntimeCodeModeCellId::new("1"),
        CellEvent::Completed {
            content_items: Vec::new(),
            error_text: None,
        },
    )
    .expect("terminal result");
    assert!(matches!(
        result,
        RuntimeCodeModeResponse::Result {
            error_text: None,
            ..
        }
    ));
}
