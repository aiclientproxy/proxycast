//! Code Mode 的协议、工具描述和 session contract。
//!
//! 该 crate 不依赖任何执行实现，保证 host、runtime、facade 与 Agent Runtime
//! 可以围绕同一份类型建立单向依赖。

pub mod description;
pub mod grpc;
pub mod host;
pub mod json_schema_types;
pub mod response;
pub mod runtime;
pub mod session;

pub use description::{build_exec_tool_description, is_code_mode_nested_tool};
pub use json_schema_types::render_json_schema_to_typescript;
pub use response::{FunctionCallOutputContentItem, ImageDetail, DEFAULT_IMAGE_DETAIL};

pub const CODE_MODE_EXEC_TOOL_NAME: &str = "exec";
pub const CODE_MODE_WAIT_TOOL_NAME: &str = "wait";
pub const DEFAULT_CODE_MODE_EXEC_YIELD_TIME_MS: u64 = 10_000;
pub const DEFAULT_CODE_MODE_WAIT_YIELD_TIME_MS: u64 = 10_000;
pub const DEFAULT_CODE_MODE_MAX_OUTPUT_TOKENS: usize = 10_000;
pub const CODE_MODE_EXEC_PRAGMA_PREFIX: &str = "// @exec:";
pub const CODE_MODE_EXEC_FREEFORM_GRAMMAR: &str = r#"
start: pragma_source | plain_source
pragma_source: PRAGMA_LINE NEWLINE SOURCE
plain_source: SOURCE

PRAGMA_LINE: /[ \t]*\/\/ @exec:[^\r\n]*/
NEWLINE: /\r?\n/
SOURCE: /[\s\S]+/
"#;

// Public protocol modules follow Codex's owner boundaries.
pub use description::*;
pub use runtime::*;
pub use session::*;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(name: &str, exposure: RuntimeToolExposure) -> RuntimeToolSnapshot {
        RuntimeToolSnapshot::new(
            RuntimeToolIdentity::plain(name),
            RuntimeToolDefinition::new(name, format!("{name} description"), json!({})),
            exposure,
            false,
            false,
        )
    }

    #[test]
    fn exec_parser_accepts_codex_pragma_and_rejects_unknown_fields() {
        let parsed = parse_code_mode_exec_source(
            "// @exec: {\"yield_time_ms\":250,\"max_output_tokens\":512}\ntext(42);",
        )
        .expect("valid exec pragma");
        assert_eq!(parsed.code, "text(42);");
        assert_eq!(parsed.yield_time_ms, Some(250));
        assert_eq!(parsed.max_output_tokens, Some(512));

        assert!(parse_code_mode_exec_source("// @exec: {\"future\":true}\ntext(42);").is_err());
        assert!(parse_code_mode_exec_source("   ").is_err());
    }

    #[test]
    fn tool_planner_keeps_direct_and_nested_surfaces_distinct() {
        let plan = plan_runtime_code_mode_tools(
            &[
                tool("read", RuntimeToolExposure::Direct),
                tool("search", RuntimeToolExposure::Deferred),
                tool("hidden", RuntimeToolExposure::Hidden),
            ],
            RuntimeToolMode::CodeMode,
            true,
            false,
        )
        .expect("valid code mode plan");
        assert_eq!(plan.model_visible_tools.len(), 1);
        assert_eq!(plan.searchable_tools.len(), 1);
        assert_eq!(plan.nested_tools.len(), 2);
    }

    #[test]
    fn code_mode_response_preserves_terminal_state_and_truncates_output() {
        let response = RuntimeCodeModeResponse::Result {
            cell_id: RuntimeCodeModeCellId::new("1"),
            content_items: vec![FunctionCallOutputContentItem::InputText {
                text: "123456".to_string(),
            }],
            error_text: None,
            code_mode_host_duration: None,
        };
        assert!(response.is_terminal());
        let result = response.into_tool_result_with_max_tokens(1);
        assert!(result.output.contains("output truncated"));
    }
}
