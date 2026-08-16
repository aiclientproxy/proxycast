//! Codex attestation capability boundary tests.

use super::super::*;
use app_server_protocol::error_codes;
use app_server_protocol::{JsonRpcMessage, JsonRpcRequest, RequestId, METHOD_INITIALIZE};
use serde_json::json;

#[tokio::test]
async fn initialize_rejects_unimplemented_codex_attestation_capability() {
    let processor = RequestProcessor::new(RuntimeCore::default());
    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(1),
            METHOD_INITIALIZE,
            Some(json!({
                "clientInfo": {
                    "name": "codex-compatible-client",
                    "version": "0.1.0"
                },
                "capabilities": {
                    "requestAttestation": true
                }
            })),
        ))
        .await
        .expect("initialize response");

    let [JsonRpcMessage::Error(error)] = messages.as_slice() else {
        panic!("expected unsupported capability error, got {messages:?}");
    };
    assert_eq!(error.error.code, error_codes::INVALID_PARAMS);
    assert_eq!(
        error.error.message,
        "capabilities.requestAttestation is unsupported: Lime has no Codex Desktop Host attestation producer"
    );
}

#[tokio::test]
async fn initialize_rejects_malformed_codex_attestation_capability() {
    let processor = RequestProcessor::new(RuntimeCore::default());
    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(2),
            METHOD_INITIALIZE,
            Some(json!({
                "clientInfo": {
                    "name": "codex-compatible-client",
                    "version": "0.1.0"
                },
                "capabilities": {
                    "requestAttestation": "true"
                }
            })),
        ))
        .await
        .expect("initialize response");

    let [JsonRpcMessage::Error(error)] = messages.as_slice() else {
        panic!("expected malformed capability error, got {messages:?}");
    };
    assert_eq!(error.error.code, error_codes::INVALID_PARAMS);
    assert_eq!(
        error.error.message,
        "capabilities.requestAttestation must be a boolean"
    );
}
