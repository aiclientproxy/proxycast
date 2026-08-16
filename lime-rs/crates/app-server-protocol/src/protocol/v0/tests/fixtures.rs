use super::super::*;
use crate::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, RequestId};
use serde_json::json;

#[test]
fn initialize_request_matches_protocol_fixture_shape() {
    let value = serde_json::to_value(JsonRpcRequest::new(
        RequestId::Integer(1),
        METHOD_INITIALIZE,
        Some(
            serde_json::to_value(InitializeParams {
                client_info: ClientInfo {
                    name: "desktop-client".to_string(),
                    title: Some("Desktop Client".to_string()),
                    version: Some("1.58.0".to_string()),
                },
                capabilities: ClientCapabilities {
                    event_methods: vec![METHOD_AGENT_SESSION_EVENT.to_string()],
                    experimental: false,
                    opt_out_notification_methods: None,
                },
            })
            .expect("serialize params"),
        ),
    ))
    .expect("serialize request");

    assert_eq!(
        value,
        json!({
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {
                    "name": "desktop-client",
                    "title": "Desktop Client",
                    "version": "1.58.0"
                },
                "capabilities": {
                    "eventMethods": ["agentSession/event"],
                    "experimental": false
                }
            }
        })
    );
}

#[test]
fn initialize_response_matches_protocol_fixture_shape() {
    let value = serde_json::to_value(
        JsonRpcResponse::new(
            RequestId::Integer(1),
            InitializeResponse {
                server_info: ServerInfo {
                    name: SERVER_NAME.to_string(),
                    version: "1.58.0".to_string(),
                    protocol_version: PROTOCOL_VERSION.to_string(),
                },
                platform: PlatformInfo {
                    family: "desktop".to_string(),
                    os: "macos".to_string(),
                },
                capabilities: ServerCapabilities {
                    agent_session: true,
                    capability_discovery: true,
                    artifact: true,
                    workspace: true,
                },
            },
        )
        .expect("create response"),
    )
    .expect("serialize response");

    assert_eq!(
        value,
        json!({
            "id": 1,
            "result": {
                "serverInfo": {
                    "name": "app-server",
                    "version": "1.58.0",
                    "protocolVersion": "appserver.v0"
                },
                "platform": {
                    "family": "desktop",
                    "os": "macos"
                },
                "capabilities": {
                    "agentSession": true,
                    "capabilityDiscovery": true,
                    "artifact": true,
                    "workspace": true
                }
            }
        })
    );
}

#[test]
fn capability_list_request_matches_protocol_fixture_shape() {
    let value = serde_json::to_value(JsonRpcRequest::new(
        RequestId::Integer(2),
        METHOD_CAPABILITY_LIST,
        Some(
            serde_json::to_value(CapabilityListParams {
                app_id: Some("content-studio".to_string()),
                workspace_id: Some("workspace-main".to_string()),
                session_id: Some("sess_1".to_string()),
                cursor: Some("2".to_string()),
                limit: Some(25),
            })
            .expect("serialize params"),
        ),
    ))
    .expect("serialize request");

    assert_eq!(
        value,
        json!({
            "id": 2,
            "method": "capability/list",
            "params": {
                "appId": "content-studio",
                "workspaceId": "workspace-main",
                "sessionId": "sess_1",
                "cursor": "2",
                "limit": 25
            }
        })
    );
}

#[test]
fn thread_start_request_matches_protocol_fixture_shape() {
    let value = serde_json::to_value(JsonRpcRequest::new(
        RequestId::String("req-start".to_string()),
        METHOD_THREAD_START,
        Some(
            serde_json::to_value(AgentSessionStartParams {
                session_id: Some("sess_1".to_string()),
                thread_id: Some("thread_1".to_string()),
                app_id: "writer".to_string(),
                workspace_id: Some("workspace_1".to_string()),
                business_object_ref: Some(BusinessObjectRef {
                    kind: "document".to_string(),
                    id: "doc_1".to_string(),
                    title: Some("Draft".to_string()),
                    uri: Some("file:///draft.md".to_string()),
                    metadata: Some(json!({ "source": "fixture" })),
                }),
                locale: Some("zh-CN".to_string()),
            })
            .expect("serialize params"),
        ),
    ))
    .expect("serialize request");

    assert_eq!(
        value,
        json!({
            "id": "req-start",
            "method": "thread/start",
            "params": {
                "sessionId": "sess_1",
                "threadId": "thread_1",
                "appId": "writer",
                "workspaceId": "workspace_1",
                "businessObjectRef": {
                    "kind": "document",
                    "id": "doc_1",
                    "title": "Draft",
                    "uri": "file:///draft.md",
                    "metadata": {
                        "source": "fixture"
                    }
                },
                "locale": "zh-CN"
            }
        })
    );
}

#[test]
fn artifact_read_request_matches_protocol_fixture_shape() {
    let value = serde_json::to_value(JsonRpcRequest::new(
        RequestId::Integer(6),
        METHOD_ARTIFACT_READ,
        Some(
            serde_json::to_value(ArtifactReadParams {
                session_id: "sess_1".to_string(),
                turn_id: Some("turn_1".to_string()),
                artifact_ref: Some("artifact-document:req-1".to_string()),
                include_content: Some(true),
                cursor: Some("2".to_string()),
                limit: Some(10),
            })
            .expect("serialize params"),
        ),
    ))
    .expect("serialize request");

    assert_eq!(
        value,
        json!({
            "id": 6,
            "method": "artifact/read",
            "params": {
                "sessionId": "sess_1",
                "turnId": "turn_1",
                "artifactRef": "artifact-document:req-1",
                "includeContent": true,
                "cursor": "2",
                "limit": 10
            }
        })
    );
}

#[test]
fn artifact_summary_content_status_matches_protocol_fixture_shape() {
    let value = serde_json::to_value(ArtifactSummary {
        artifact_ref: "artifact-document:req-1".to_string(),
        event_id: "evt-artifact-1".to_string(),
        sequence: 7,
        turn_id: Some("turn_1".to_string()),
        artifact_id: Some("req-1".to_string()),
        path: Some(".lime/artifacts/report.md".to_string()),
        title: Some("Report".to_string()),
        kind: Some("document".to_string()),
        status: Some("ready".to_string()),
        content: Some("# Report".to_string()),
        content_status: ArtifactContentStatus::Available,
        metadata: Some(json!({ "version": 2 })),
    })
    .expect("serialize artifact summary");

    assert_eq!(
        value,
        json!({
            "artifactRef": "artifact-document:req-1",
            "eventId": "evt-artifact-1",
            "sequence": 7,
            "turnId": "turn_1",
            "artifactId": "req-1",
            "path": ".lime/artifacts/report.md",
            "title": "Report",
            "kind": "document",
            "status": "ready",
            "content": "# Report",
            "contentStatus": "available",
            "metadata": {
                "version": 2
            }
        })
    );
}

#[test]
fn turn_start_request_matches_protocol_fixture_shape() {
    let value = serde_json::to_value(JsonRpcRequest::new(
        RequestId::Integer(2),
        METHOD_TURN_START,
        Some(
            serde_json::to_value(AgentSessionTurnStartParams {
                session_id: "sess_1".to_string(),
                turn_id: Some("turn_1".to_string()),
                input: AgentInput {
                    text: "hello".to_string(),
                    attachments: vec![AgentAttachment {
                        kind: "file".to_string(),
                        uri: Some("file:///draft.md".to_string()),
                        metadata: Some(json!({ "mimeType": "text/markdown" })),
                    }],
                },
                runtime_options: Some(RuntimeOptions {
                    capability_id: Some("draft.write".to_string()),
                    stream: true,
                    event_name: Some("plugin_runtime:app:task".to_string()),
                    queued_turn_id: Some("queued-turn-1".to_string()),
                    runtime_request: Some(RuntimeRequest {
                        provider_preference: Some("deepseek".to_string()),
                        model_preference: Some("deepseek-v4-flash".to_string()),
                        metadata: Some(json!({ "taskId": "task-1" })),
                        ..RuntimeRequest::default()
                    }),
                    ..RuntimeOptions::default()
                }),
                queue_if_busy: true,
                skip_pre_submit_resume: true,
            })
            .expect("serialize params"),
        ),
    ))
    .expect("serialize request");

    assert_eq!(
        value,
        json!({
            "id": 2,
            "method": "turn/start",
            "params": {
                "sessionId": "sess_1",
                "turnId": "turn_1",
                "input": {
                    "text": "hello",
                    "attachments": [{
                        "kind": "file",
                        "uri": "file:///draft.md",
                        "metadata": {
                            "mimeType": "text/markdown"
                        }
                    }]
                },
                "runtimeOptions": {
                    "capabilityId": "draft.write",
                    "stream": true,
                    "eventName": "plugin_runtime:app:task",
                    "queuedTurnId": "queued-turn-1",
                    "runtimeRequest": {
                        "providerPreference": "deepseek",
                        "modelPreference": "deepseek-v4-flash",
                        "metadata": {
                            "taskId": "task-1"
                        }
                    }
                },
                "queueIfBusy": true,
                "skipPreSubmitResume": true
            }
        })
    );
}

#[test]
fn turn_interrupt_request_matches_protocol_fixture_shape() {
    let value = serde_json::to_value(JsonRpcRequest::new(
        RequestId::Integer(3),
        METHOD_TURN_INTERRUPT,
        Some(
            serde_json::to_value(AgentSessionTurnCancelParams {
                session_id: "sess_1".to_string(),
                turn_id: "turn_1".to_string(),
            })
            .expect("serialize params"),
        ),
    ))
    .expect("serialize request");

    assert_eq!(
        value,
        json!({
            "id": 3,
            "method": "turn/interrupt",
            "params": {
                "sessionId": "sess_1",
                "turnId": "turn_1"
            }
        })
    );
}

#[test]
fn agent_session_action_respond_request_matches_protocol_fixture_shape() {
    let value = serde_json::to_value(JsonRpcRequest::new(
        RequestId::Integer(5),
        METHOD_AGENT_SESSION_ACTION_RESPOND,
        Some(
            serde_json::to_value(AgentSessionActionRespondParams {
                session_id: "sess_1".to_string(),
                request_id: "req_confirm_1".to_string(),
                action_type: AgentSessionActionType::ToolConfirmation,
                decision: Some(AgentSessionApprovalDecision::AllowOnce),
                confirmed: None,
                response: Some("allow".to_string()),
                user_data: Some(json!({ "choice": "allow" })),
                metadata: Some(json!({ "source": "content-studio" })),
                event_name: Some("plugin_runtime:app:task".to_string()),
                action_scope: Some(AgentSessionActionScope {
                    session_id: Some("sess_1".to_string()),
                    thread_id: Some("thread_1".to_string()),
                    turn_id: Some("turn_1".to_string()),
                }),
            })
            .expect("serialize params"),
        ),
    ))
    .expect("serialize request");

    assert_eq!(
        value,
        json!({
            "id": 5,
            "method": "agentSession/action/respond",
            "params": {
                "sessionId": "sess_1",
                "requestId": "req_confirm_1",
                "actionType": "tool_confirmation",
                "decision": "allow_once",
                "response": "allow",
                "userData": {
                    "choice": "allow"
                },
                "metadata": {
                    "source": "content-studio"
                },
                "eventName": "plugin_runtime:app:task",
                "actionScope": {
                    "sessionId": "sess_1",
                    "threadId": "thread_1",
                    "turnId": "turn_1"
                }
            }
        })
    );
}

#[test]
fn agent_session_event_notification_matches_protocol_fixture_shape() {
    let value = serde_json::to_value(JsonRpcNotification::new(
        METHOD_AGENT_SESSION_EVENT,
        Some(
            serde_json::to_value(AgentSessionEventParams {
                event: AgentEvent {
                    event_id: "evt_1".to_string(),
                    sequence: 1,
                    session_id: "sess_1".to_string(),
                    thread_id: Some("thread_1".to_string()),
                    turn_id: Some("turn_1".to_string()),
                    event_type: "turn.started".to_string(),
                    timestamp: "2026-06-04T00:00:00Z".to_string(),
                    payload: json!({
                        "status": "running",
                        "delta": {
                            "text": "hello"
                        },
                        "turn": {
                            "sessionId": "sess_1",
                            "threadId": "thread_1",
                            "turnId": "turn_1",
                            "status": "inProgress",
                            "createdAtMs": 100,
                            "updatedAtMs": 120
                        }
                    }),
                },
            })
            .expect("serialize params"),
        ),
    ))
    .expect("serialize notification");

    assert_eq!(
        value,
        json!({
            "method": "agentSession/event",
            "params": {
                "event": {
                    "eventId": "evt_1",
                    "sequence": 1,
                    "sessionId": "sess_1",
                    "threadId": "thread_1",
                    "turnId": "turn_1",
                    "type": "turn.started",
                    "timestamp": "2026-06-04T00:00:00Z",
                    "payload": {
                        "status": "running",
                        "delta": {
                            "text": "hello"
                        },
                        "turn": {
                            "sessionId": "sess_1",
                            "threadId": "thread_1",
                            "turnId": "turn_1",
                            "status": "inProgress",
                            "createdAtMs": 100,
                            "updatedAtMs": 120
                        }
                    }
                }
            }
        })
    );
}
