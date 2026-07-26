use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{
    ThreadBackgroundTerminal, ThreadBackgroundTerminalsCleanParams,
    ThreadBackgroundTerminalsCleanResponse, ThreadBackgroundTerminalsListParams,
    ThreadBackgroundTerminalsListResponse, ThreadBackgroundTerminalsTerminateParams,
    ThreadBackgroundTerminalsTerminateResponse,
};
use app_server_protocol::{error_codes, JsonRpcError};
use uuid::Uuid;

impl RequestProcessor {
    pub(super) async fn handle_thread_background_terminals_clean_v2(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadBackgroundTerminalsCleanParams = parse_params(params)?;
        let thread_id = canonical_thread_id(&params.thread_id)?;
        self.runtime
            .clean_background_terminals(&thread_id)
            .map_err(to_jsonrpc_error)?;
        dispatch_result(ThreadBackgroundTerminalsCleanResponse {})
    }

    pub(super) async fn handle_thread_background_terminals_list_v2(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadBackgroundTerminalsListParams = parse_params(params)?;
        let thread_id = canonical_thread_id(&params.thread_id)?;
        let terminals = self
            .runtime
            .list_background_terminals(&thread_id)
            .map_err(to_jsonrpc_error)?;
        let (data, next_cursor) =
            paginate_background_terminals(&terminals, params.cursor, params.limit)?;
        dispatch_result(ThreadBackgroundTerminalsListResponse { data, next_cursor })
    }

    pub(super) async fn handle_thread_background_terminals_terminate_v2(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadBackgroundTerminalsTerminateParams = parse_params(params)?;
        let thread_id = canonical_thread_id(&params.thread_id)?;
        let process_id = params.process_id.parse::<u64>().map_err(|error| {
            invalid_request(format!("invalid background terminal process id: {error}"))
        })?;
        let terminated = self
            .runtime
            .terminate_background_terminal(&thread_id, process_id)
            .map_err(to_jsonrpc_error)?;
        dispatch_result(ThreadBackgroundTerminalsTerminateResponse { terminated })
    }
}

fn canonical_thread_id(thread_id: &str) -> Result<String, JsonRpcError> {
    let thread_id = thread_id.trim();
    if thread_id.is_empty() {
        return Err(invalid_request("thread id must not be empty"));
    }
    Uuid::parse_str(thread_id)
        .map_err(|error| invalid_request(format!("invalid thread id: {error}")))?;
    Ok(thread_id.to_string())
}

fn paginate_background_terminals(
    terminals: &[ThreadBackgroundTerminal],
    cursor: Option<String>,
    limit: Option<u32>,
) -> Result<(Vec<ThreadBackgroundTerminal>, Option<String>), JsonRpcError> {
    let start = match cursor {
        Some(cursor) => {
            let cursor = cursor
                .parse::<u64>()
                .map_err(|error| invalid_request(format!("invalid cursor: {error}")))?;
            terminals
                .iter()
                .position(|terminal| {
                    terminal
                        .process_id
                        .parse::<u64>()
                        .is_ok_and(|process_id| process_id > cursor)
                })
                .unwrap_or(terminals.len())
        }
        None => 0,
    };
    let effective_limit = limit
        .map(|limit| usize::try_from(limit.max(1)).unwrap_or(usize::MAX))
        .unwrap_or(terminals.len());
    let end = start.saturating_add(effective_limit).min(terminals.len());
    let next_cursor = (end < terminals.len()).then(|| terminals[end - 1].process_id.clone());
    Ok((terminals[start..end].to_vec(), next_cursor))
}

fn invalid_request(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(error_codes::INVALID_REQUEST, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terminal(process_id: &str) -> ThreadBackgroundTerminal {
        ThreadBackgroundTerminal {
            item_id: format!("item-{process_id}"),
            process_id: process_id.to_string(),
            command: format!("command-{process_id}"),
            cwd: "/tmp".to_string(),
            os_pid: None,
            cpu_percent: None,
            rss_kb: None,
        }
    }

    #[test]
    fn paginates_with_process_id_cursor_after_anchor_disappears() {
        let terminals = vec![terminal("1"), terminal("2"), terminal("3"), terminal("4")];
        let (first, cursor) = paginate_background_terminals(&terminals, None, Some(2))
            .expect("first background terminal page");
        assert_eq!(first, vec![terminal("1"), terminal("2")]);
        assert_eq!(cursor.as_deref(), Some("2"));

        let without_anchor = vec![terminal("1"), terminal("3"), terminal("4")];
        let (second, cursor) = paginate_background_terminals(&without_anchor, cursor, Some(2))
            .expect("second background terminal page");
        assert_eq!(second, vec![terminal("3"), terminal("4")]);
        assert_eq!(cursor, None);
    }

    #[test]
    fn rejects_non_numeric_cursor_and_clamps_zero_limit() {
        let terminals = vec![terminal("1"), terminal("2")];
        assert!(
            paginate_background_terminals(&terminals, Some("opaque".to_string()), Some(1)).is_err()
        );
        let (data, cursor) = paginate_background_terminals(&terminals, None, Some(0))
            .expect("zero limit is clamped");
        assert_eq!(data, vec![terminal("1")]);
        assert_eq!(cursor.as_deref(), Some("1"));
    }
}
