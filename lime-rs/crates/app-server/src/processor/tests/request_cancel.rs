use super::tests_support::initialize_processor;
use crate::processor::RequestProcessor;
use crate::runtime::sidecar_store::{SidecarBytesWriteRequest, SidecarStore};
use crate::RuntimeCore;
use crate::RuntimeEvent;
use app_server_protocol::error_codes;
use app_server_protocol::protocol::v2::METHOD_MEDIA_READ;
use app_server_protocol::AgentSessionStartParams;
use app_server_protocol::JsonRpcMessage;
use app_server_protocol::JsonRpcNotification;
use app_server_protocol::JsonRpcRequest;
use app_server_protocol::RequestId;
use app_server_protocol::METHOD_AGENT_SESSION_EVENT;
use app_server_protocol::METHOD_CANCEL_REQUEST;
use app_server_protocol::METHOD_THREAD_LIST;
use app_server_protocol::METHOD_THREAD_RESUME;
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn cancel_request_notification_fails_matching_request_id_closed() {
    let processor = RequestProcessor::new(RuntimeCore::default());
    initialize_processor(&processor).await;

    processor.handle_notification(JsonRpcNotification::new(
        METHOD_CANCEL_REQUEST,
        Some(json!({ "id": 7 })),
    ));

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(7),
            METHOD_THREAD_LIST,
            Some(json!({})),
        ))
        .await
        .expect("request");

    let [JsonRpcMessage::Error(error)] = messages.as_slice() else {
        panic!("expected canceled request error, got {messages:?}");
    };
    assert_eq!(error.id, RequestId::Integer(7));
    assert_eq!(error.error.code, error_codes::REQUEST_CANCELLED);
    assert_eq!(error.error.message, "request canceled");
}

#[tokio::test]
async fn canceled_v2_thread_resume_wins_over_initialization_and_clears_state() {
    let processor = RequestProcessor::new(RuntimeCore::default());
    let request_id = RequestId::Integer(8);

    processor.handle_notification(JsonRpcNotification::new(
        METHOD_CANCEL_REQUEST,
        Some(json!({ "id": 8 })),
    ));

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            request_id.clone(),
            METHOD_THREAD_RESUME,
            Some(json!({ "threadId": "thread-resume" })),
        ))
        .await
        .expect("canceled resume request");
    let [JsonRpcMessage::Error(error)] = messages.as_slice() else {
        panic!("expected canceled resume error, got {messages:?}");
    };
    assert_eq!(error.error.code, error_codes::REQUEST_CANCELLED);

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            request_id,
            METHOD_THREAD_RESUME,
            Some(json!({ "threadId": "thread-resume" })),
        ))
        .await
        .expect("resume request after cancel cleanup");
    let [JsonRpcMessage::Error(error)] = messages.as_slice() else {
        panic!("expected not initialized resume error, got {messages:?}");
    };
    assert_eq!(error.error.code, error_codes::NOT_INITIALIZED);
}

#[tokio::test]
async fn v2_thread_resume_requires_initialization_before_runtime_lookup() {
    let processor = RequestProcessor::new(RuntimeCore::default());

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(9),
            METHOD_THREAD_RESUME,
            Some(json!({ "threadId": "thread-resume" })),
        ))
        .await
        .expect("uninitialized resume request");
    let [JsonRpcMessage::Error(error)] = messages.as_slice() else {
        panic!("expected not initialized resume error, got {messages:?}");
    };
    assert_eq!(error.error.code, error_codes::NOT_INITIALIZED);
}

#[tokio::test]
async fn media_read_streaming_request_emits_chunk_notifications() {
    let temp = tempfile::tempdir().expect("sidecar tempdir");
    let sidecar_store = Arc::new(SidecarStore::new(temp.path()).expect("sidecar store"));
    let runtime = RuntimeCore::default().with_sidecar_store(sidecar_store.clone());
    runtime
        .start_session(AgentSessionStartParams {
            session_id: Some("sess-media-stream".to_string()),
            thread_id: Some("thread-media-stream".to_string()),
            app_id: "agent-chat".to_string(),
            workspace_id: Some("default".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("session");
    let sidecar_ref = sidecar_store
        .write_bytes(&SidecarBytesWriteRequest {
            session_id: "sess-media-stream".to_string(),
            kind: "media".to_string(),
            logical_id: "fixture-image".to_string(),
            relative_path: "sessions/sess-media-stream/media/fixture-image.png".to_string(),
            content: vec![0x89, b'P', b'N', b'G'],
        })
        .expect("write media sidecar");
    runtime
        .append_external_runtime_events(
            "sess-media-stream",
            None,
            vec![RuntimeEvent::new(
                "message.delta",
                json!({
                    "contentPart": {
                        "type": "media",
                        "reference": {
                            "uri": sidecar_ref.ref_id,
                            "sidecarRef": sidecar_ref
                        }
                    }
                }),
            )],
        )
        .expect("append media ref");
    let processor = RequestProcessor::new(runtime);
    initialize_processor(&processor).await;
    let mut streamed = Vec::new();

    let messages = processor
        .handle_request_streaming(
            JsonRpcRequest::new(
                RequestId::Integer(10),
                METHOD_MEDIA_READ,
                Some(json!({
                    "threadId": "thread-media-stream",
                    "uri": sidecar_ref.ref_id,
                    "maxBytes": 1024,
                    "length": 4,
                    "stream": true
                })),
            ),
            &mut |message| streamed.push(message),
        )
        .await
        .expect("request");

    let [JsonRpcMessage::Response(response)] = messages.as_slice() else {
        panic!("expected final media read response, got {messages:?}");
    };
    assert_eq!(response.id, RequestId::Integer(10));
    assert_eq!(response.result["contentBase64"], "iVBORw==");
    assert_eq!(streamed.len(), 2);
    for message in &streamed {
        let JsonRpcMessage::Notification(notification) = message else {
            panic!("expected streaming notification, got {message:?}");
        };
        assert_eq!(notification.method, METHOD_AGENT_SESSION_EVENT);
    }
    let first_event = &streamed[0];
    let JsonRpcMessage::Notification(first_notification) = first_event else {
        panic!("expected first notification");
    };
    let first_params = first_notification.params.as_ref().expect("params");
    assert_eq!(first_params["event"]["type"], "media.read.chunk");
    assert_eq!(
        first_params["event"]["payload"]["chunk"]["contentBase64"],
        "iVBORw=="
    );
    let second_event = &streamed[1];
    let JsonRpcMessage::Notification(second_notification) = second_event else {
        panic!("expected second notification");
    };
    let second_params = second_notification.params.as_ref().expect("params");
    assert_eq!(second_params["event"]["type"], "media.read.completed");
    assert_eq!(second_params["event"]["payload"]["done"], true);
}

#[tokio::test]
async fn cancel_request_notification_cancels_media_read_before_sidecar_io() {
    let temp = tempfile::tempdir().expect("sidecar tempdir");
    let sidecar_store = Arc::new(SidecarStore::new(temp.path()).expect("sidecar store"));
    let runtime = RuntimeCore::default().with_sidecar_store(sidecar_store.clone());
    runtime
        .start_session(AgentSessionStartParams {
            session_id: Some("sess-media-cancel".to_string()),
            thread_id: Some("thread-media-cancel".to_string()),
            app_id: "agent-chat".to_string(),
            workspace_id: Some("default".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("session");
    let sidecar_ref = sidecar_store
        .write_bytes(&SidecarBytesWriteRequest {
            session_id: "sess-media-cancel".to_string(),
            kind: "media".to_string(),
            logical_id: "fixture-image".to_string(),
            relative_path: "sessions/sess-media-cancel/media/fixture-image.png".to_string(),
            content: vec![0x89, b'P', b'N', b'G'],
        })
        .expect("write media sidecar");
    runtime
        .append_external_runtime_events(
            "sess-media-cancel",
            None,
            vec![RuntimeEvent::new(
                "message.delta",
                json!({
                    "contentPart": {
                        "type": "media",
                        "reference": {
                            "uri": sidecar_ref.ref_id,
                            "sidecarRef": sidecar_ref
                        }
                    }
                }),
            )],
        )
        .expect("append media ref");
    let processor = RequestProcessor::new(runtime);
    initialize_processor(&processor).await;

    processor.handle_notification(JsonRpcNotification::new(
        METHOD_CANCEL_REQUEST,
        Some(json!({ "id": 9 })),
    ));

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(9),
            METHOD_MEDIA_READ,
            Some(json!({
                "threadId": "thread-media-cancel",
                "uri": sidecar_ref.ref_id,
                "maxBytes": 1024
            })),
        ))
        .await
        .expect("request");

    let [JsonRpcMessage::Error(error)] = messages.as_slice() else {
        panic!("expected canceled media read error, got {messages:?}");
    };
    assert_eq!(error.id, RequestId::Integer(9));
    assert_eq!(error.error.code, error_codes::REQUEST_CANCELLED);
    assert_eq!(error.error.message, "request canceled");
}
