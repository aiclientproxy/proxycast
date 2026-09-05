use super::super::RequestProcessor;
use super::tests_support::initialize_processor;
use crate::RuntimeCore;
use app_server_protocol::protocol::v2::METHOD_COMMAND_EXEC;
use app_server_protocol::{error_codes, JsonRpcMessage, JsonRpcRequest, RequestId};
use serde_json::json;

#[tokio::test]
async fn command_exec_rejects_client_granted_permissions_at_jsonrpc_boundary() {
    let processor = RequestProcessor::new(RuntimeCore::default());
    initialize_processor(&processor).await;

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(2),
            METHOD_COMMAND_EXEC,
            Some(json!({
                "command": ["printf", "ok"],
                "grantedPermissions": {"network": {"enabled": true}}
            })),
        ))
        .await
        .expect("command/exec response");

    let [JsonRpcMessage::Error(error)] = messages.as_slice() else {
        panic!("expected command/exec invalid params response: {messages:?}");
    };
    assert_eq!(error.error.code, error_codes::INVALID_PARAMS);
    assert!(error.error.message.contains("managed by permissionProfile"));
}
