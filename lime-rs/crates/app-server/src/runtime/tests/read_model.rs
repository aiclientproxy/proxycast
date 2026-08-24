use super::support::*;
use super::*;
use async_trait::async_trait;
use std::sync::Arc;

struct RoutingDecisionReadModelBackend;

#[async_trait]
impl ExecutionBackend for RoutingDecisionReadModelBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new(
            "routing.decision.made",
            json!({
                "backend": "runtime",
                "routingDecision": {
                    "routingMode": "profile_slot",
                    "decisionSource": "profile_model_slot",
                    "decisionReason": "profile_slot_selected",
                    "settingsSource": "workspace_profile",
                    "serviceModelSlot": "coding",
                    "selectedProvider": "custom-coding",
                    "selectedModel": "coder-large",
                    "requestedProvider": "custom-coding",
                    "requestedModel": "coder-large",
                    "fallbackChain": []
                },
                "modelSlot": {
                    "serviceModelSlot": "coding",
                    "slots": [
                        {
                            "slot": "coding",
                            "provider": "custom-coding",
                            "model": "coder-large"
                        },
                        {
                            "slot": "review",
                            "provider": "custom-review",
                            "model": "review-small"
                        }
                    ]
                },
                "providerReadiness": {
                    "ready": true,
                    "status": "ready",
                    "source": "provider_store",
                    "enabledKeyCount": 1
                },
                "modelRegistry": {
                    "source": "provider_declared_model",
                    "status": "matched",
                    "reasonCode": "matched_provider_models",
                    "modelCapabilities": {
                        "provenance": "inferred_hint",
                        "capabilities": {
                            "tools": true,
                            "streaming": true,
                            "reasoning": true
                        },
                        "taskFamilies": ["chat", "reasoning"],
                        "runtimeFeatures": ["streaming", "tool_calling", "reasoning"]
                    },
                    "modelAlias": {
                        "canonicalModelId": "coder-large",
                        "providerModelId": "coder-large",
                        "aliasSource": "local"
                    },
                    "reasoning": {
                        "supported": true
                    }
                },
                "modelTaskRequest": {
                    "taskKind": "chat",
                    "source": "agent_turn",
                    "modelRef": {
                        "providerId": "custom-coding",
                        "modelId": "coder-large",
                        "routingSlot": "coding",
                        "source": "profile_slot"
                    },
                    "modalityContractKey": "chat",
                    "routingSlot": "coding",
                    "requirements": {
                        "taskFamilies": ["chat"],
                        "inputModalities": ["text"],
                        "outputModalities": ["text"],
                        "runtimeFeatures": ["streaming"],
                        "capabilities": ["tools", "streaming"]
                    },
                    "sessionId": "sess_routing_read",
                    "threadId": "thread_routing_read",
                    "turnId": "turn_routing_read"
                },
                "resolvedRoute": {
                    "modelRef": {
                        "providerId": "custom-coding",
                        "modelId": "coder-large",
                        "routingSlot": "coding",
                        "source": "profile_slot"
                    },
                    "protocol": "openai_chat",
                    "endpoint": {
                        "kind": "openai_compatible",
                        "baseUrl": "https://coding.example.com/v1"
                    },
                    "auth": {
                        "kind": "api_key_ref",
                        "providerId": "custom-coding",
                        "credentialRef": "runtime-api-key-key-1"
                    },
                    "transport": "http",
                    "framing": "sse",
                    "defaults": {},
                    "capabilitySnapshot": {
                        "taskFamilies": ["chat", "reasoning"],
                        "inputModalities": ["text"],
                        "outputModalities": ["text"],
                        "runtimeFeatures": ["streaming", "tool_calling", "reasoning"],
                        "capabilities": {
                            "vision": false,
                            "tools": true,
                            "streaming": true,
                            "jsonMode": false,
                            "functionCalling": false,
                            "reasoning": true
                        },
                        "source": "provider_declared_model",
                        "reasonCode": "matched_provider_models"
                    },
                    "decision": {
                        "routingMode": "profile_slot",
                        "decisionSource": "profile_model_slot",
                        "decisionReason": "profile_slot_selected",
                        "settingsSource": "workspace_profile",
                        "serviceModelSlot": "coding",
                        "fallbackChain": [],
                        "candidateCount": 1
                    }
                }
            }),
        ))?;
        sink.emit(RuntimeEvent::new("turn.completed", json!({})))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

async fn start_read_model_test_turn(
    session_id: &str,
    thread_id: &str,
    turn_id: &str,
) -> (RuntimeCore, AgentTurn) {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some(session_id.to_string()),
        thread_id: Some(thread_id.to_string()),
        app_id: "desktop".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: session_id.to_string(),
                turn_id: Some(turn_id.to_string()),
                input: AgentInput {
                    text: "测试工具生命周期".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("turn")
        .response
        .turn;
    (core, turn)
}

#[tokio::test]
async fn load_session_current_enriches_completed_media_task_store_results() {
    let workspace = tempfile::TempDir::new().expect("workspace root");
    let workspace_root = workspace.path().to_string_lossy().to_string();
    let session_id = "sess-media-task-read-model";
    let thread_id = "thread-media-task-read-model";
    let turn_id = "turn-media-task-read-model";
    let timestamp = "2026-07-07T00:00:00.000Z";
    let session = AgentSession {
        session_id: session_id.to_string(),
        thread_id: thread_id.to_string(),
        app_id: "agent-runtime".to_string(),
        workspace_id: Some("workspace-media".to_string()),
        business_object_ref: Some(BusinessObjectRef {
            kind: "project".to_string(),
            id: "workspace-media".to_string(),
            title: None,
            uri: None,
            metadata: Some(json!({
                "workingDir": workspace_root,
            })),
        }),
        status: AgentSessionStatus::Completed,
        created_at: timestamp.to_string(),
        updated_at: timestamp.to_string(),
    };
    let media_task = media_task_artifact_response(json!({
        "task_id": "task-image-read-model",
        "task_type": "image_generate",
        "payload": {
            "sessionId": session_id,
            "threadId": thread_id,
            "turnId": turn_id,
            "prompt": "生成一张青柠封面图"
        },
        "status": "succeeded",
        "normalized_status": "succeeded",
        "created_at": timestamp,
        "updated_at": "2026-07-07T00:00:01.000Z",
        "completed_at": "2026-07-07T00:00:02.000Z",
        "result": {
            "images": [{
                "url": "data:image/png;base64,AAECAw==",
                "caption": "青柠封面图",
                "sidecarRef": {
                    "ref": "sidecar://media/image-read-model",
                    "kind": "media",
                    "relativePath": "sessions/sess-media-task-read-model/media/image-read-model.png",
                    "bytes": 4,
                    "sha256": "sha256:read-model",
                    "contentStatus": "available",
                    "uri": "sidecar://media/image-read-model",
                    "mimeType": "image/png"
                }
            }]
        }
    }));
    let data_source =
        Arc::new(TestSessionDataSource::new().with_media_task_artifacts(vec![media_task]));
    let core = RuntimeCore::default().with_app_data_source(data_source.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some(session.session_id.clone()),
        thread_id: Some(session.thread_id.clone()),
        app_id: session.app_id.clone(),
        workspace_id: session.workspace_id.clone(),
        business_object_ref: session.business_object_ref.clone(),
        locale: None,
    })
    .expect("session");

    let loaded = core
        .load_session_current(AgentSessionReadParams {
            session_id: session.session_id.clone(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("session read");
    let detail = loaded.response.detail.expect("read detail");

    assert_eq!(data_source.media_task_list_requests().len(), 1);
    let item = detail
        .get("items")
        .and_then(serde_json::Value::as_array)
        .and_then(|items| {
            items.iter().find(|item| {
                item.get("id").and_then(serde_json::Value::as_str)
                    == Some("media-task-result:turn-media-task-read-model:task-image-read-model")
            })
        })
        .expect("media task synthetic item");
    assert_eq!(item["type"], "agent_message");
    assert_eq!(item["status"], "completed");
    assert_eq!(item["contentParts"][0]["type"], "media");
    assert_eq!(item["contentParts"][0]["kind"], "image");
    assert_eq!(
        item["contentParts"][0]["reference"]["uri"],
        "sidecar://media/image-read-model"
    );
    assert_eq!(
        item["contentParts"][0]["source"],
        "media_task_store_owner_facts"
    );
    assert_eq!(item["metadata"]["source"], "media_task_store_owner_facts");
    assert_eq!(item["metadata"]["task_id"], "task-image-read-model");
}

fn media_task_artifact_response(record: serde_json::Value) -> MediaTaskArtifactResponse {
    MediaTaskArtifactResponse {
        success: true,
        task_id: "task-image-read-model".to_string(),
        task_type: "image_generate".to_string(),
        task_family: "media".to_string(),
        status: "succeeded".to_string(),
        normalized_status: "succeeded".to_string(),
        current_attempt_id: None,
        path: String::new(),
        absolute_path: String::new(),
        artifact_path: ".lime/tasks/image_generate/task-image-read-model.json".to_string(),
        absolute_artifact_path: String::new(),
        reused_existing: false,
        idempotency_key: None,
        record,
    }
}

mod article_workspace_artifact_documents;
mod article_workspace_worker_evidence;
mod artifacts;
mod imports_items;
mod messages_diagnostics;
mod role_switch;
mod tool_calls;
mod workflow;
