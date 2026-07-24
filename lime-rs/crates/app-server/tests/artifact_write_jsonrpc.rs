use std::sync::Arc;
use std::time::Duration;

use app_server::{
    run_json_lines, AgentInput, AgentSessionStartParams, AgentSessionTurnStartParams, AppServer,
    ProjectionStore, RuntimeCore, RuntimeEvent, RuntimeHostContext, SidecarStore,
};
use app_server_protocol::protocol::v2::METHOD_ARTIFACT_WRITE;
use app_server_protocol::{
    error_codes, METHOD_ARTIFACT_READ, METHOD_INITIALIZE, METHOD_INITIALIZED,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::time::timeout;

#[tokio::test]
async fn artifact_write_public_jsonrpc_persists_and_reads_typed_content_without_legacy_wrapper() {
    let temp = TempDir::new().expect("artifact/write transport fixture temp dir");
    let runtime = RuntimeCore::default()
        .with_sidecar_store(Arc::new(
            SidecarStore::new(temp.path()).expect("artifact sidecar store"),
        ))
        .with_projection_store(Arc::new(
            ProjectionStore::initialize(temp.path().join("projection.sqlite"))
                .expect("artifact projection store"),
        ));
    runtime
        .start_session(AgentSessionStartParams {
            session_id: Some("sess_artifact_transport".to_string()),
            thread_id: Some("thread_artifact_transport".to_string()),
            app_id: "content-studio".to_string(),
            workspace_id: Some("workspace-main".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("artifact session");
    let turn = runtime
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_artifact_transport".to_string(),
                turn_id: Some("turn_artifact_transport".to_string()),
                input: AgentInput {
                    text: "draft article".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("artifact turn")
        .response
        .turn;
    runtime
        .append_external_runtime_events(
            "sess_artifact_transport",
            Some(&turn.turn_id),
            vec![RuntimeEvent::new("turn.completed", json!({}))],
        )
        .expect("complete artifact turn");

    let server = AppServer::with_runtime(runtime);
    let (mut input_client, input_server) = tokio::io::duplex(16 * 1024);
    let (output_server, output_client) = tokio::io::duplex(16 * 1024);
    let runner = tokio::spawn(run_json_lines(server, input_server, output_server));
    let mut output_lines = BufReader::new(output_client).lines();

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": METHOD_INITIALIZE,
            "params": {
                "clientInfo": {
                    "name": "artifact-write-jsonrpc-test",
                    "version": "1.0.0"
                }
            }
        }),
    )
    .await;
    assert_response_ok(&read_response(&mut output_lines, 1).await, "initialize");
    write_message(
        &mut input_client,
        json!({ "jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {} }),
    )
    .await;

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": METHOD_ARTIFACT_WRITE,
            "params": {
                "threadId": "thread_artifact_transport",
                "turnId": "turn_artifact_transport",
                "artifact": {
                    "artifactRef": "artifact-document-transport",
                    "artifactDocumentId": "document-transport",
                    "path": "drafts/article.json",
                    "title": "Draft",
                    "kind": "artifact_document",
                    "status": "ready",
                    "content": "{\"schemaVersion\":\"artifact-document/v1\"}",
                    "metadata": {"versionNo": 1}
                }
            }
        }),
    )
    .await;
    let (write, messages_before_write_response) =
        read_response_with_preceding_messages(&mut output_lines, 2).await;
    assert_response_ok(&write, "artifact/write");
    assert!(
        messages_before_write_response
            .iter()
            .all(|message| message.get("method") != Some(&json!("agentSession/event"))),
        "artifact/write emitted a legacy agentSession/event wrapper: {messages_before_write_response:#?}"
    );
    assert_eq!(
        write.pointer("/result/threadId"),
        Some(&json!("thread_artifact_transport"))
    );
    assert_eq!(
        write.pointer("/result/artifactRef"),
        Some(&json!("artifact-document-transport"))
    );
    assert_eq!(
        write.pointer("/result/sidecar/contentStatus"),
        Some(&json!("available"))
    );
    assert!(write
        .pointer("/result/sidecar/bytes")
        .and_then(Value::as_u64)
        .is_some_and(|bytes| bytes > 0));

    assert!(
        timeout(Duration::from_millis(100), output_lines.next_line())
            .await
            .is_err(),
        "artifact/write emitted an unexpected notification after its response"
    );

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": METHOD_ARTIFACT_READ,
            "params": {
                "sessionId": "sess_artifact_transport",
                "artifactRef": "artifact-document-transport",
                "includeContent": true
            }
        }),
    )
    .await;
    let read = read_response(&mut output_lines, 3).await;
    assert_response_ok(&read, "artifact/read");
    assert_eq!(
        read.pointer("/result/artifacts/0/content"),
        Some(&json!("{\"schemaVersion\":\"artifact-document/v1\"}"))
    );
    assert_eq!(
        read.pointer("/result/artifacts/0/contentStatus"),
        Some(&json!("available"))
    );

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "agentSession/runtimeEvents/append",
            "params": {
                "sessionId": "sess_artifact_transport",
                "runtimeEvents": []
            }
        }),
    )
    .await;
    let retired = read_response(&mut output_lines, 4).await;
    assert_eq!(
        retired.pointer("/error/code"),
        Some(&json!(error_codes::METHOD_NOT_FOUND))
    );
    assert!(retired.get("result").is_none());

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "agentSession/action/replay",
            "params": {
                "sessionId": "sess_artifact_transport",
                "requestId": "request-retired"
            }
        }),
    )
    .await;
    let retired_replay = read_response(&mut output_lines, 5).await;
    assert_eq!(
        retired_replay.pointer("/error/code"),
        Some(&json!(error_codes::METHOD_NOT_FOUND))
    );
    assert!(retired_replay.get("result").is_none());

    drop(input_client);
    timeout(Duration::from_secs(2), runner)
        .await
        .expect("JSON lines runner should stop after input closes")
        .expect("JSON lines runner task")
        .expect("JSON lines runner result");
}

async fn write_message(client: &mut DuplexStream, message: Value) {
    client
        .write_all(format!("{message}\n").as_bytes())
        .await
        .expect("write JSON-RPC message");
}

async fn read_response(lines: &mut Lines<BufReader<DuplexStream>>, id: u64) -> Value {
    read_response_with_preceding_messages(lines, id).await.0
}

async fn read_response_with_preceding_messages(
    lines: &mut Lines<BufReader<DuplexStream>>,
    id: u64,
) -> (Value, Vec<Value>) {
    let mut preceding = Vec::new();
    loop {
        let line = timeout(Duration::from_secs(2), lines.next_line())
            .await
            .expect("timed out waiting for JSON-RPC response")
            .expect("read JSON-RPC response")
            .expect("JSON-RPC output closed");
        let message: Value = serde_json::from_str(&line).expect("decode JSON-RPC response");
        if message.get("id").and_then(Value::as_u64) == Some(id) {
            return (message, preceding);
        }
        preceding.push(message);
    }
}

fn assert_response_ok(response: &Value, label: &str) {
    assert!(
        response.get("error").is_none(),
        "{label} returned an error: {response:#?}"
    );
    assert!(
        response.get("result").is_some(),
        "{label} returned no result: {response:#?}"
    );
}
