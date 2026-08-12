use super::*;
use crate::tool_definition::RuntimeToolExposure;
use serde_json::json;
use std::sync::Mutex;

fn tool(name: &str, exposure: RuntimeToolExposure) -> RuntimeToolSnapshot {
    RuntimeToolSnapshot::new(
        RuntimeToolIdentity::plain(name),
        RuntimeToolDefinition::new(name, format!("{name} description"), json!({})),
        exposure,
        false,
        false,
    )
}

fn tool_names(tools: &[RuntimeToolSnapshot]) -> Vec<&str> {
    tools
        .iter()
        .map(|tool| tool.definition.name.as_str())
        .collect()
}

fn nested_tool_names(tools: &[RuntimeCodeModeTool]) -> Vec<&str> {
    tools.iter().map(|tool| tool.code_name.as_str()).collect()
}

#[test]
fn exec_source_parser_accepts_supported_pragma_fields() {
    let parsed = parse_code_mode_exec_source(
        "// @exec: {\"yield_time_ms\":250,\"max_output_tokens\":512}\ntext(42);",
    )
    .expect("supported pragma");

    assert_eq!(parsed.code, "text(42);");
    assert_eq!(parsed.yield_time_ms, Some(250));
    assert_eq!(parsed.max_output_tokens, Some(512));
}

#[test]
fn exec_source_parser_fails_closed_for_invalid_pragmas() {
    for source in [
        "// @exec: {\"future_field\":true}\ntext(42);",
        "// @exec: {\"yield_time_ms\":9007199254740992}\ntext(42);",
        "// @exec: {\"max_output_tokens\":9007199254740992}\ntext(42);",
        "// @exec: {\"yield_time_ms\":1}\n",
        "   ",
    ] {
        assert!(
            parse_code_mode_exec_source(source).is_err(),
            "invalid pragma source must fail: {source:?}"
        );
    }
}

struct RecordingSession {
    operations: Mutex<Vec<String>>,
}

impl RecordingSession {
    fn new() -> Self {
        Self {
            operations: Mutex::new(Vec::new()),
        }
    }

    fn operations(&self) -> Vec<String> {
        self.operations.lock().expect("recorded operations").clone()
    }
}

impl RuntimeCodeModeSession for RecordingSession {
    fn execute<'a>(
        &'a self,
        request: RuntimeCodeModeExecuteRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeStartedCell> {
        Box::pin(async move {
            self.operations
                .lock()
                .expect("record execute")
                .push(format!(
                    "execute:{}:{}",
                    request.tool_call_id, request.source
                ));
            Ok(RuntimeCodeModeStartedCell::new(
                RuntimeCodeModeCellId::new("cell-execute"),
                Box::pin(async {
                    Ok(RuntimeCodeModeResponse::Result {
                        cell_id: RuntimeCodeModeCellId::new("cell-execute"),
                        output: "done".to_string(),
                        error_text: None,
                    })
                }),
            ))
        })
    }

    fn wait<'a>(
        &'a self,
        request: RuntimeCodeModeWaitRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            self.operations.lock().expect("record wait").push(format!(
                "wait:{}:{}",
                request.cell_id, request.yield_time_ms
            ));
            Ok(RuntimeCodeModeWaitOutcome::LiveCell(
                RuntimeCodeModeResponse::Yielded {
                    cell_id: request.cell_id,
                    output: "pending".to_string(),
                },
            ))
        })
    }

    fn terminate<'a>(
        &'a self,
        cell_id: RuntimeCodeModeCellId,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            self.operations
                .lock()
                .expect("record terminate")
                .push(format!("terminate:{cell_id}"));
            Ok(RuntimeCodeModeWaitOutcome::LiveCell(
                RuntimeCodeModeResponse::Terminated {
                    cell_id,
                    output: String::new(),
                },
            ))
        })
    }

    fn shutdown(&self) -> RuntimeCodeModeFuture<'_, ()> {
        Box::pin(async move {
            self.operations
                .lock()
                .expect("record shutdown")
                .push("shutdown".to_string());
            Ok(())
        })
    }
}

struct RecordingProvider {
    session: RuntimeCodeModeSessionHandle,
    creates: Mutex<usize>,
}

impl RuntimeCodeModeSessionProvider for RecordingProvider {
    fn create_session<'a>(
        &'a self,
        _delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    ) -> RuntimeCodeModeSessionProviderFuture<'a> {
        Box::pin(async move {
            *self.creates.lock().expect("record create") += 1;
            Ok(self.session.clone())
        })
    }
}

#[test]
fn public_exec_and_wait_contract_matches_codex_names_grammar_and_defaults() {
    assert_eq!(CODE_MODE_EXEC_TOOL_NAME, "exec");
    assert_eq!(CODE_MODE_WAIT_TOOL_NAME, "wait");
    assert!(CODE_MODE_EXEC_FREEFORM_GRAMMAR.contains("start: pragma_source | plain_source"));
    assert!(CODE_MODE_EXEC_FREEFORM_GRAMMAR.contains("PRAGMA_LINE:"));
    assert_eq!(DEFAULT_CODE_MODE_EXEC_YIELD_TIME_MS, 10_000);
    assert_eq!(DEFAULT_CODE_MODE_WAIT_YIELD_TIME_MS, 10_000);
    assert_eq!(DEFAULT_CODE_MODE_MAX_OUTPUT_TOKENS, 10_000);

    let wait = code_mode_wait_tool_definition();
    assert_eq!(wait.name, CODE_MODE_WAIT_TOOL_NAME);
    assert_eq!(wait.input_schema["required"], json!(["cell_id"]));
    assert_eq!(
        wait.input_schema["properties"]["terminate"]["type"],
        "boolean"
    );
}

#[test]
fn runtime_responses_preserve_terminal_state_and_model_visible_status() {
    let yielded = RuntimeCodeModeResponse::Yielded {
        cell_id: RuntimeCodeModeCellId::new("7"),
        output: "partial".to_string(),
    };
    assert!(!yielded.is_terminal());
    assert_eq!(yielded.cell_id().as_str(), "7");
    assert_eq!(
        yielded.into_tool_result(),
        RuntimeCodeModeToolResult {
            cell_id: RuntimeCodeModeCellId::new("7"),
            success: true,
            output: "Script running with cell ID 7\nOutput:\npartial".to_string(),
            error: None,
        }
    );

    let terminated = RuntimeCodeModeResponse::Terminated {
        cell_id: RuntimeCodeModeCellId::new("8"),
        output: "last output".to_string(),
    };
    assert!(terminated.is_terminal());
    assert_eq!(
        terminated.into_tool_result(),
        RuntimeCodeModeToolResult {
            cell_id: RuntimeCodeModeCellId::new("8"),
            success: true,
            output: "Script terminated\nOutput:\nlast output".to_string(),
            error: None,
        }
    );

    let failed = RuntimeCodeModeResponse::Result {
        cell_id: RuntimeCodeModeCellId::new("9"),
        output: "before failure".to_string(),
        error_text: Some("boom".to_string()),
    };
    assert!(failed.is_terminal());
    assert_eq!(
        failed.into_tool_result(),
        RuntimeCodeModeToolResult {
            cell_id: RuntimeCodeModeCellId::new("9"),
            success: false,
            output: "Script failed\nOutput:\nbefore failure\nScript error:\nboom".to_string(),
            error: Some("boom".to_string()),
        }
    );
}

#[tokio::test]
async fn session_handle_delegates_execute_wait_terminate_and_shutdown() {
    let session = Arc::new(RecordingSession::new());
    let handle = RuntimeCodeModeSessionHandle::new(session.clone());

    let started = handle
        .execute(RuntimeCodeModeExecuteRequest {
            tool_call_id: "call-1".to_string(),
            source: "return 1".to_string(),
            enabled_tools: Vec::new(),
            yield_time_ms: None,
            max_output_tokens: None,
            cancellation_token: None,
        })
        .await
        .expect("execute starts a cell");
    assert_eq!(started.cell_id.as_str(), "cell-execute");
    assert!(started
        .initial_response()
        .await
        .expect("initial response")
        .is_terminal());

    let waited = handle
        .wait(RuntimeCodeModeWaitRequest {
            cell_id: RuntimeCodeModeCellId::new("cell-execute"),
            yield_time_ms: 42,
        })
        .await
        .expect("wait delegates");
    assert!(matches!(
        waited,
        RuntimeCodeModeWaitOutcome::LiveCell(RuntimeCodeModeResponse::Yielded { .. })
    ));

    let terminated = handle
        .terminate(RuntimeCodeModeCellId::new("cell-execute"))
        .await
        .expect("terminate delegates");
    assert!(matches!(
        terminated,
        RuntimeCodeModeWaitOutcome::LiveCell(RuntimeCodeModeResponse::Terminated { .. })
    ));
    handle.shutdown().await.expect("shutdown delegates");
    assert_eq!(
        session.operations(),
        vec![
            "execute:call-1:return 1",
            "wait:cell-execute:42",
            "terminate:cell-execute",
            "shutdown",
        ]
    );
}

#[tokio::test]
async fn session_provider_accepts_default_limits_and_rejects_unsupported_limits() {
    let session = RuntimeCodeModeSessionHandle::new(Arc::new(RecordingSession::new()));
    let provider = RecordingProvider {
        session,
        creates: Mutex::new(0),
    };
    let delegate: Arc<dyn RuntimeCodeModeSessionDelegate> =
        Arc::new(NoopRuntimeCodeModeSessionDelegate);

    provider
        .create_session_with_limits(delegate.clone(), RuntimeCodeModeSessionLimits::default())
        .await
        .expect("default limits delegate to create_session");
    assert_eq!(*provider.creates.lock().expect("create count"), 1);

    let error = provider
        .create_session_with_limits(
            delegate,
            RuntimeCodeModeSessionLimits {
                max_yield_time_ms: Some(1_000),
                max_heap_size_bytes: None,
            },
        )
        .await
        .expect_err("non-default limits require provider support");
    assert_eq!(
        error,
        "code mode session provider does not support resource limits"
    );
    assert_eq!(*provider.creates.lock().expect("create count"), 1);
}

#[test]
fn exposure_plan_matches_direct_code_mode_and_code_mode_only_surfaces() {
    let tools = vec![
        tool("direct", RuntimeToolExposure::Direct),
        tool("deferred", RuntimeToolExposure::Deferred),
        tool(
            "deferred_model_only",
            RuntimeToolExposure::DeferredModelOnly,
        ),
        tool("direct_model_only", RuntimeToolExposure::DirectModelOnly),
        tool("nested_only", RuntimeToolExposure::CodeModeOnly),
        tool("hidden", RuntimeToolExposure::Hidden),
    ];

    let direct = plan_runtime_code_mode_tools(&tools, RuntimeToolMode::Direct, false, false)
        .expect("direct mode does not require code mode");
    assert_eq!(
        tool_names(&direct.model_visible_tools),
        vec!["direct", "direct_model_only"]
    );
    assert_eq!(
        tool_names(&direct.searchable_tools),
        vec!["deferred", "deferred_model_only"]
    );
    assert!(direct.nested_tools.is_empty());

    let code_mode = plan_runtime_code_mode_tools(&tools, RuntimeToolMode::CodeMode, true, false)
        .expect("available code mode");
    assert_eq!(
        tool_names(&code_mode.model_visible_tools),
        vec!["direct", "direct_model_only"]
    );
    assert_eq!(
        nested_tool_names(&code_mode.nested_tools),
        vec!["direct", "deferred", "nested_only"]
    );

    let code_mode_only =
        plan_runtime_code_mode_tools(&tools, RuntimeToolMode::CodeModeOnly, true, false)
            .expect("available code-mode-only surface");
    assert_eq!(
        tool_names(&code_mode_only.model_visible_tools),
        vec!["direct_model_only"]
    );
    assert_eq!(
        nested_tool_names(&code_mode_only.nested_tools),
        vec!["direct", "deferred", "nested_only"]
    );
}

#[test]
fn unavailable_code_mode_falls_back_only_when_explicitly_allowed() {
    let fallback = resolve_runtime_tool_mode(RuntimeToolMode::CodeMode, false, false)
        .expect("regular code mode may use direct fallback");
    assert_eq!(fallback.effective, RuntimeToolMode::Direct);
    assert!(fallback.used_direct_fallback);

    assert_eq!(
        resolve_runtime_tool_mode(RuntimeToolMode::CodeMode, false, true),
        Err(RuntimeToolModeResolutionError::CodeModeUnavailable {
            requested: RuntimeToolMode::CodeMode,
        })
    );
    assert_eq!(
        resolve_runtime_tool_mode(RuntimeToolMode::CodeModeOnly, false, false),
        Err(RuntimeToolModeResolutionError::CodeModeUnavailable {
            requested: RuntimeToolMode::CodeModeOnly,
        })
    );
}

#[test]
fn nested_names_follow_codex_namespace_and_collision_rules() {
    assert_eq!(
        code_mode_name_for_identity(&RuntimeToolIdentity::namespaced("mcp", "lookup")),
        "mcp__lookup"
    );
    assert_eq!(
        code_mode_name_for_identity(&RuntimeToolIdentity::namespaced("mcp__", "lookup")),
        "mcp__lookup"
    );
    assert_eq!(
        normalize_code_mode_identifier("hidden-dynamic-tool"),
        "hidden_dynamic_tool"
    );
    assert_eq!(normalize_code_mode_identifier("9 invalid"), "__invalid");
    assert_eq!(normalize_code_mode_identifier(""), "_");

    let tools = vec![
        tool("hidden-dynamic-tool", RuntimeToolExposure::CodeModeOnly),
        tool("hidden dynamic tool", RuntimeToolExposure::CodeModeOnly),
    ];
    let plan = plan_runtime_code_mode_tools(&tools, RuntimeToolMode::CodeModeOnly, true, false)
        .expect("available code mode");
    assert_eq!(
        nested_tool_names(&plan.nested_tools),
        vec!["hidden-dynamic-tool"]
    );
    assert_eq!(
        nested_tool_names(&plan.shadowed_nested_tools),
        vec!["hidden dynamic tool"]
    );

    assert_eq!(
        plan_runtime_code_mode_tools(
            &[tool(
                CODE_MODE_EXEC_TOOL_NAME,
                RuntimeToolExposure::DirectModelOnly,
            )],
            RuntimeToolMode::CodeMode,
            true,
            false,
        ),
        Err(RuntimeToolModeResolutionError::ReservedToolNameCollision {
            tool_name: CODE_MODE_EXEC_TOOL_NAME.to_string(),
        })
    );
}
