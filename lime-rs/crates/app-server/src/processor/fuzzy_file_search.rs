use super::{dispatch_result, parse_params, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::FuzzyFileSearchParams;
use app_server_protocol::JsonRpcError;
use serde_json::Value;

impl RequestProcessor {
    pub(super) async fn handle_fuzzy_file_search_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: FuzzyFileSearchParams = parse_params(params)?;
        dispatch_result(self.fuzzy_file_search.search(params).await?)
    }
}
