use super::tool_process_metadata::SoulStyleMetadata;
use super::{
    backend_error, current_agent_runtime_config_metadata, direct_provider_config_from_request,
    initialize_runtime_database, model_route_contract, model_route_resolver, request_context,
    RuntimeBackend,
};
use crate::runtime::memory_prompt::append_soul_context_to_system_prompt;
use crate::{ExecutionRequest, RuntimeCoreError};
use lime_agent::{
    host_managed_generation_session_id, run_host_managed_generation,
    write_host_managed_generation_status, HostManagedGenerationPlan,
    HostManagedGenerationRunRequest,
};
use serde_json::{json, Value};

pub(super) async fn prepare_plugin_worker_request(
    runtime_backend: &RuntimeBackend,
    request: &ExecutionRequest,
    worker_request: &mut Value,
) -> Result<(), RuntimeCoreError> {
    let Some(plan) = HostManagedGenerationPlan::from_worker_request(worker_request) else {
        return Ok(());
    };
    let config_metadata = current_agent_runtime_config_metadata();
    let runtime_metadata = request.runtime_metadata();
    let soul_style = host_generation_soul_style(config_metadata.as_ref(), runtime_metadata);
    if plan.requests.is_empty() {
        write_host_managed_generation_status(
            worker_request,
            host_generation_status_payload(HostGenerationStatusPayload {
                status: "skipped",
                soul_style: soul_style.as_ref(),
                ..HostGenerationStatusPayload::default()
            }),
        );
        return Ok(());
    }

    match generate_outputs(
        runtime_backend,
        request,
        worker_request,
        &plan,
        config_metadata.as_ref(),
        runtime_metadata,
    )
    .await
    {
        Ok(generated) => {
            write_host_managed_generation_status(
                worker_request,
                host_generation_status_payload(HostGenerationStatusPayload {
                    status: "completed",
                    provider: Some(generated.provider.as_str()),
                    model: Some(generated.model.as_str()),
                    outputs: Some(generated.outputs.as_slice()),
                    soul_style: soul_style.as_ref(),
                    ..HostGenerationStatusPayload::default()
                }),
            );
        }
        Err(error) => {
            let message = error.to_string();
            write_host_managed_generation_status(
                worker_request,
                host_generation_status_payload(HostGenerationStatusPayload {
                    status: "unavailable",
                    reason_code: Some("host_generation_unavailable"),
                    message: Some(message.as_str()),
                    soul_style: soul_style.as_ref(),
                    ..HostGenerationStatusPayload::default()
                }),
            );
        }
    }

    Ok(())
}

struct GeneratedOutputs {
    provider: String,
    model: String,
    outputs: Vec<Value>,
}

async fn generate_outputs(
    runtime_backend: &RuntimeBackend,
    request: &ExecutionRequest,
    worker_request: &Value,
    plan: &HostManagedGenerationPlan,
    config_metadata: Option<&Value>,
    runtime_metadata: Option<&Value>,
) -> Result<GeneratedOutputs, RuntimeCoreError> {
    let db = initialize_runtime_database(runtime_backend.db.as_ref())?;
    runtime_backend.ensure_agent_initialized(&db).await?;
    let requested_selection = request_context::resolve_runtime_model_selection(request)?;
    let host_request = request_context::runtime_request_from_request(request);
    let direct_provider_config = direct_provider_config_from_request(
        host_request.as_ref(),
        &requested_selection,
        requested_selection.reasoning_effort.clone(),
    );
    let route_resolution = model_route_resolver::resolve_chat_model_route(
        &db,
        &runtime_backend.api_key_provider_service,
        request,
        &requested_selection,
        direct_provider_config.as_ref(),
    )
    .await
    .map_err(backend_error)?;
    let selection = route_resolution.selection.clone();
    if let Some(route_failure) = route_resolution.resolved_route.failure.as_ref() {
        return Err(RuntimeCoreError::Backend(format!(
            "host managed generation route unavailable: {}",
            route_failure.reason_code
        )));
    }

    let generation_session_id = host_managed_generation_session_id(worker_request);
    let system_prompt_context =
        append_soul_context_to_system_prompt(None, config_metadata, runtime_metadata);
    let generated = run_host_managed_generation(
        &runtime_backend.agent_state,
        HostManagedGenerationRunRequest {
            db: &db,
            generation_session_id,
            worker_request,
            plan,
            system_prompt_context,
            provider_configuration: Some(
                model_route_contract::provider_configuration_from_runtime(
                    &selection,
                    &route_resolution.resolved_route,
                    direct_provider_config,
                    request_context::service_tier_from_request(request),
                ),
            ),
        },
    )
    .await
    .map_err(RuntimeCoreError::Backend)?;
    let provider_config = generated.provider_config.ok_or_else(|| {
        RuntimeCoreError::Backend("host managed generation provider was not configured".to_string())
    })?;
    let provider = provider_config
        .provider_selector
        .clone()
        .unwrap_or_else(|| selection.provider.clone());
    let model = provider_config.model_name.clone();

    Ok(GeneratedOutputs {
        provider,
        model,
        outputs: generated.outputs,
    })
}

#[derive(Default)]
struct HostGenerationStatusPayload<'a> {
    status: &'a str,
    provider: Option<&'a str>,
    model: Option<&'a str>,
    outputs: Option<&'a [Value]>,
    reason_code: Option<&'a str>,
    message: Option<&'a str>,
    soul_style: Option<&'a SoulStyleMetadata>,
}

fn host_generation_soul_style(
    config_metadata: Option<&Value>,
    runtime_metadata: Option<&Value>,
) -> Option<SoulStyleMetadata> {
    SoulStyleMetadata::from_config_metadata(config_metadata)
        .or_else(|| SoulStyleMetadata::from_config_metadata(runtime_metadata))
}

fn host_generation_status_payload(input: HostGenerationStatusPayload<'_>) -> Value {
    let mut payload = serde_json::Map::new();
    payload.insert("status".to_string(), json!(input.status));
    insert_optional_string(&mut payload, "provider", input.provider);
    insert_optional_string(&mut payload, "model", input.model);
    insert_optional_string(&mut payload, "reasonCode", input.reason_code);
    insert_optional_string(&mut payload, "message", input.message);
    if let Some(outputs) = input.outputs {
        payload.insert("outputs".to_string(), Value::Array(outputs.to_vec()));
    }

    let lifecycle = host_generation_lifecycle(input.status, input.soul_style);
    payload.insert(
        "presentation".to_string(),
        host_generation_presentation(
            input.status,
            input.provider,
            input.model,
            input.outputs,
            &lifecycle,
        ),
    );
    payload.insert(
        "generationBriefBoundary".to_string(),
        host_generation_brief_boundary(),
    );
    payload.insert(
        "host_managed_generation_facts".to_string(),
        host_generation_facts(input.status, input.provider, input.model, input.outputs),
    );
    payload.insert(
        "soul_lifecycle".to_string(),
        Value::Object(lifecycle.clone()),
    );
    payload.insert(
        "soul_surface".to_string(),
        Value::String("plugin_host_managed_generation".to_string()),
    );
    if let Some(phase) = lifecycle.get("phase").and_then(Value::as_str) {
        payload.insert("soul_phase".to_string(), Value::String(phase.to_string()));
    }
    if let Some(style_level) = lifecycle.get("styleLevel").and_then(Value::as_str) {
        payload.insert(
            "style_level".to_string(),
            Value::String(style_level.to_string()),
        );
    }
    if let Some(risk_level) = lifecycle.get("riskLevel").and_then(Value::as_str) {
        payload.insert(
            "risk_level".to_string(),
            Value::String(risk_level.to_string()),
        );
    }
    if let Some(soul_style) = input.soul_style {
        soul_style.insert_top_level_fields(&mut payload);
    }

    Value::Object(payload)
}

fn host_generation_lifecycle(
    status: &str,
    soul_style: Option<&SoulStyleMetadata>,
) -> serde_json::Map<String, Value> {
    let (phase, style_level) = match status {
        "completed" => ("after_artifact", "L2"),
        "unavailable" | "failed" => ("after_artifact_failure", "L2"),
        _ => ("artifact_generation_progress", "L1"),
    };
    let mut lifecycle = serde_json::Map::from_iter([
        (
            "surface".to_string(),
            Value::String("plugin_host_managed_generation".to_string()),
        ),
        ("phase".to_string(), Value::String(phase.to_string())),
        ("status".to_string(), Value::String(status.to_string())),
        (
            "styleLevel".to_string(),
            Value::String(style_level.to_string()),
        ),
        ("riskLevel".to_string(), Value::String("normal".to_string())),
    ]);
    if let Some(soul_style) = soul_style {
        soul_style.insert_lifecycle_fields(&mut lifecycle);
    }
    lifecycle
}

fn host_generation_presentation(
    status: &str,
    provider: Option<&str>,
    model: Option<&str>,
    outputs: Option<&[Value]>,
    lifecycle: &serde_json::Map<String, Value>,
) -> Value {
    let status_key = match status {
        "completed" => "completed",
        "unavailable" | "failed" => "unavailable",
        "skipped" => "skipped",
        _ => "requested",
    };
    let output_count = outputs.map(|items| items.len()).unwrap_or(0);
    json!({
        "schemaVersion": "lime.plugin.host_managed_generation.presentation.v1",
        "surface": "plugin_host_managed_generation",
        "status": status,
        "styleLevel": lifecycle.get("styleLevel").and_then(Value::as_str).unwrap_or("L1"),
        "riskLevel": lifecycle.get("riskLevel").and_then(Value::as_str).unwrap_or("normal"),
        "soulSurface": lifecycle.get("surface").and_then(Value::as_str).unwrap_or("plugin_host_managed_generation"),
        "soulPhase": lifecycle.get("phase").and_then(Value::as_str).unwrap_or("artifact_generation_progress"),
        "titleKey": format!("plugin.apps.runtime.agentRun.hostManagedGeneration.{status_key}.title"),
        "messageKey": format!("plugin.apps.runtime.agentRun.hostManagedGeneration.{status_key}.message"),
        "values": {
            "provider": provider.unwrap_or(""),
            "model": model.unwrap_or(""),
            "outputCount": output_count,
        },
        "toneVariant": lifecycle.get("toneVariant").and_then(Value::as_str).unwrap_or("neutral"),
        "profileId": lifecycle.get("profileId").and_then(Value::as_str),
        "packId": lifecycle.get("packId").and_then(Value::as_str),
    })
}

fn host_generation_brief_boundary() -> Value {
    json!({
        "schemaVersion": "lime.plugin.host_managed_generation.boundary.v1",
        "artifactBodyStyleLevel": "L3",
        "formalArtifactVoiceSource": "generation_brief_only",
        "productSoulDefault": "interaction_only",
        "rules": [
            "Process narration may follow Interaction Soul at L1/L2.",
            "Generated artifact body is L3 and must use explicit Generation Brief or plugin-declared voice.",
            "Product Soul must not rewrite plugin artifact body by default."
        ],
    })
}

fn host_generation_facts(
    status: &str,
    provider: Option<&str>,
    model: Option<&str>,
    outputs: Option<&[Value]>,
) -> Value {
    let output_refs = outputs
        .unwrap_or_default()
        .iter()
        .map(|output| {
            json!({
                "id": output.get("id").and_then(Value::as_str),
                "kind": output.get("kind").and_then(Value::as_str),
                "targetObjectKind": output.get("targetObjectKind").and_then(Value::as_str),
                "outputField": output.get("outputField").and_then(Value::as_str),
                "contentType": output.get("contentType").and_then(Value::as_str),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "source": "app_server_runtime_backend",
        "surface": "plugin_host_managed_generation",
        "status": status,
        "provider": provider,
        "model": model,
        "outputCount": output_refs.len(),
        "outputs": output_refs,
        "artifactBodyStyleLevel": "L3",
        "formalArtifactVoiceSource": "generation_brief_only",
        "productSoulDefault": "interaction_only",
    })
}

fn insert_optional_string(
    target: &mut serde_json::Map<String, Value>,
    key: &str,
    value: Option<&str>,
) {
    if let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) {
        target.insert(key.to_string(), Value::String(value.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::{
        AgentSession, AgentSessionStatus, AgentTurn, AgentTurnStatus, RuntimeOptions,
    };
    use lime_agent::{HOST_MANAGED_GENERATION_SCHEMA, HOST_MANAGED_GENERATION_SOURCE};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn generation_plan_reads_host_managed_generation_requests() {
        let worker_request = json!({
            "runtime": {
                "hostManagedGeneration": {
                    "enabled": true,
                    "systemPrompt": "生成中文 Markdown",
                    "requests": [
                        {
                            "id": "article-draft-document",
                            "kind": "markdown_document",
                            "targetObjectKind": "articleDraft",
                            "outputField": "documentText",
                            "instructions": "写一篇文章"
                        }
                    ]
                }
            }
        });

        let plan = HostManagedGenerationPlan::from_worker_request(&worker_request).expect("plan");
        assert_eq!(plan.system_prompt.as_deref(), Some("生成中文 Markdown"));
        assert_eq!(plan.requests.len(), 1);
        assert_eq!(plan.requests[0].id, "article-draft-document");
        assert_eq!(plan.requests[0].kind.as_deref(), Some("markdown_document"));
        assert_eq!(
            plan.requests[0].target_object_kind.as_deref(),
            Some("articleDraft")
        );
        assert_eq!(
            plan.requests[0].output_field.as_deref(),
            Some("documentText")
        );
    }

    #[test]
    fn write_generation_status_mirrors_payload_into_runtime_result() {
        let mut worker_request = json!({
            "runtime": {
                "hostManagedGeneration": {
                    "enabled": true
                }
            }
        });

        write_host_managed_generation_status(
            &mut worker_request,
            json!({
                "status": "unavailable",
                "reasonCode": "host_generation_unavailable",
                "message": "no configured provider"
            }),
        );

        assert_eq!(
            worker_request["hostManagedGeneration"]["schemaVersion"],
            HOST_MANAGED_GENERATION_SCHEMA
        );
        assert_eq!(
            worker_request["hostManagedGeneration"]["source"],
            HOST_MANAGED_GENERATION_SOURCE
        );
        assert_eq!(
            worker_request["hostManagedGeneration"]["status"],
            "unavailable"
        );
        assert_eq!(
            worker_request["runtime"]["hostManagedGenerationResult"],
            worker_request["hostManagedGeneration"]
        );
    }

    #[test]
    fn host_generation_system_prompt_context_inherits_soul_from_config_metadata() {
        let config_metadata = json!({
            "memory": {
                "soul": {
                    "schema": "memory_soul_prompt_context.v2",
                    "source": "memory.soul",
                    "scope": "interaction_only",
                    "styleProfile": {
                        "id": "cheeky_sassy_executor",
                        "packId": "com.lime.soul.cheeky-sassy-executor",
                        "tone": "cheeky_sassy",
                        "intensity": "low",
                        "allowedMoves": ["Apply this tone to greetings, opening turns, self-introductions, chat replies, and tool progress."],
                        "forbiddenMoves": ["Do not invent tool results."],
                        "seriousModeFallback": "calm_professional_partner"
                    },
                    "styleBoundary": {
                        "formalArtifactVoiceSource": "generation_brief_only",
                        "fidelityRules": ["Formal artifacts must use explicit Generation Brief voice, not Product Soul."]
                    }
                }
            }
        });

        let prompt_context =
            append_soul_context_to_system_prompt(None, Some(&config_metadata), None)
                .expect("prompt context");

        assert!(prompt_context.contains("## Interaction Soul"));
        assert!(prompt_context.contains("Style profile: cheeky_sassy_executor"));
        assert!(prompt_context.contains("Style pack: com.lime.soul.cheeky-sassy-executor"));
        assert!(!prompt_context.contains("Style pack: com.lime.builtin.default"));
        assert!(prompt_context.contains("Formal artifact voice source: generation_brief_only"));
        assert!(
            prompt_context.contains("Formal artifacts must use explicit Generation Brief voice")
        );
    }

    #[tokio::test]
    async fn prepare_worker_request_injects_host_managed_generation_from_fixture_provider() {
        let generated_markdown = "# 宿主离线生成标题\n\n这里是 localhost fixture 生成的正文。";
        let base_url = start_openai_compatible_fixture(generated_markdown).await;
        let backend = RuntimeBackend::new();
        let request = generation_request_for_test(&base_url);
        let mut worker_request = json!({
            "schemaVersion": "content-factory.worker-request.v1",
            "appId": "content-factory-app",
            "sessionId": "session-host-generation",
            "turnId": "turn-host-generation",
            "taskId": "task-host-generation",
            "taskKind": "content.article.generate",
            "prompt": "写一篇关于内容工厂插件化写作的文章",
            "runtime": {
                "hostManagedGeneration": {
                    "enabled": true,
                    "systemPrompt": "生成可直接进入文章编辑器的 Markdown 正文。",
                    "requests": [
                        {
                            "id": "article-draft-document",
                            "kind": "markdown_document",
                            "targetObjectKind": "articleDraft",
                            "outputField": "documentText",
                            "instructions": "只输出文章正文。"
                        }
                    ]
                }
            }
        });

        prepare_plugin_worker_request(&backend, &request, &mut worker_request)
            .await
            .expect("prepare worker request");

        assert_eq!(
            worker_request["hostManagedGeneration"]["status"], "completed",
            "host generation failed: {:?}",
            worker_request["hostManagedGeneration"]
        );
        assert_eq!(
            worker_request["hostManagedGeneration"]["provider"],
            "fixture-openai"
        );
        assert_eq!(
            worker_request["hostManagedGeneration"]["model"],
            "lime-fixture-chat"
        );
        assert_eq!(
            worker_request["hostManagedGeneration"]["outputs"][0]["id"],
            "article-draft-document"
        );
        assert_eq!(
            worker_request["hostManagedGeneration"]["outputs"][0]["targetObjectKind"],
            "articleDraft"
        );
        assert_eq!(
            worker_request["hostManagedGeneration"]["outputs"][0]["outputField"],
            "documentText"
        );
        assert_eq!(
            worker_request["hostManagedGeneration"]["outputs"][0]["content"],
            generated_markdown
        );
        assert_eq!(
            worker_request["hostManagedGeneration"]["presentation"]["surface"],
            "plugin_host_managed_generation"
        );
        assert_eq!(
            worker_request["hostManagedGeneration"]["presentation"]["styleLevel"],
            "L2"
        );
        assert_eq!(
            worker_request["hostManagedGeneration"]["presentation"]["messageKey"],
            "plugin.apps.runtime.agentRun.hostManagedGeneration.completed.message"
        );
        assert_eq!(
            worker_request["hostManagedGeneration"]["generationBriefBoundary"]
                ["formalArtifactVoiceSource"],
            "generation_brief_only"
        );
        assert_eq!(
            worker_request["hostManagedGeneration"]["host_managed_generation_facts"]
                ["artifactBodyStyleLevel"],
            "L3"
        );
        assert_eq!(
            worker_request["runtime"]["hostManagedGenerationResult"],
            worker_request["hostManagedGeneration"]
        );
    }

    #[test]
    fn host_generation_status_payload_marks_process_and_formal_artifact_boundary() {
        let soul_style = SoulStyleMetadata {
            profile_id: Some("cheeky_sassy_executor".to_string()),
            pack_id: Some("com.lime.soul.cheeky-sassy-executor".to_string()),
            tone_variant: Some("cheeky_sassy".to_string()),
        };

        let payload = host_generation_status_payload(HostGenerationStatusPayload {
            status: "completed",
            provider: Some("fixture-openai"),
            model: Some("lime-fixture-chat"),
            outputs: Some(&[json!({
                "id": "article-draft-document",
                "kind": "markdown_document",
                "targetObjectKind": "articleDraft",
                "outputField": "documentText",
                "contentType": "text/markdown",
                "content": "# 正文"
            })]),
            soul_style: Some(&soul_style),
            ..HostGenerationStatusPayload::default()
        });

        assert_eq!(payload["presentation"]["styleLevel"], "L2");
        assert_eq!(payload["presentation"]["toneVariant"], "cheeky_sassy");
        assert_eq!(
            payload["presentation"]["packId"],
            "com.lime.soul.cheeky-sassy-executor"
        );
        assert_eq!(payload["soul_lifecycle"]["phase"], "after_artifact");
        assert_eq!(
            payload["host_managed_generation_facts"]["formalArtifactVoiceSource"],
            "generation_brief_only"
        );
        assert_eq!(
            payload["generationBriefBoundary"]["productSoulDefault"],
            "interaction_only"
        );
    }

    async fn start_openai_compatible_fixture(content: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fixture provider");
        let addr = listener.local_addr().expect("fixture address");
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept fixture request");
            let mut buffer = vec![0_u8; 16 * 1024];
            let _ = socket
                .read(&mut buffer)
                .await
                .expect("read fixture request");
            let body = fixture_sse_body(content);
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream; charset=utf-8\r\ncache-control: no-cache\r\nconnection: close\r\ncontent-length: {}\r\n\r\n{}",
                body.as_bytes().len(),
                body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write fixture response");
            let _ = socket.shutdown().await;
        });
        format!("http://{}", addr)
    }

    fn fixture_sse_body(content: &str) -> String {
        let first_chunk = json!({
            "id": "chatcmpl-host-managed-generation-fixture",
            "object": "chat.completion.chunk",
            "created": 1_770_000_000,
            "model": "lime-fixture-chat",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": null
                }
            ]
        });
        let final_chunk = json!({
            "id": "chatcmpl-host-managed-generation-fixture",
            "object": "chat.completion.chunk",
            "created": 1_770_000_000,
            "model": "lime-fixture-chat",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2
            }
        });
        format!("data: {first_chunk}\n\ndata: {final_chunk}\n\ndata: [DONE]\n\n")
    }

    fn generation_request_for_test(base_url: &str) -> ExecutionRequest {
        ExecutionRequest {
            host: crate::RuntimeHostContext::default(),
            session: AgentSession {
                session_id: "session-host-generation".to_string(),
                thread_id: "thread-host-generation".to_string(),
                app_id: "content-factory-app".to_string(),
                workspace_id: Some("workspace-main".to_string()),
                business_object_ref: None,
                status: AgentSessionStatus::Running,
                created_at: "2026-07-03T00:00:00.000Z".to_string(),
                updated_at: "2026-07-03T00:00:00.000Z".to_string(),
            },
            turn: AgentTurn {
                turn_id: "turn-host-generation".to_string(),
                session_id: "session-host-generation".to_string(),
                thread_id: "thread-host-generation".to_string(),
                status: AgentTurnStatus::Accepted,
                started_at: None,
                completed_at: None,
            },
            forked_from_thread_id: None,
            input: agent_runtime::reply_input::RuntimeReplyInput::text(
                "写一篇关于内容工厂插件化写作的文章",
            ),
            runtime_options: Some(RuntimeOptions {
                stream: true,
                runtime_request: Some(app_server_protocol::RuntimeRequest {
                    provider_config: Some(app_server_protocol::RuntimeProviderConfig {
                        provider_id: Some("fixture-openai".to_string()),
                        provider_name: Some("openai".to_string()),
                        model_name: Some("lime-fixture-chat".to_string()),
                        api_key: Some("fixture-key".to_string()),
                        base_url: Some(base_url.to_string()),
                        model_capabilities: Some(json!({
                            "capabilities": {
                                "tools": true,
                                "streaming": true,
                                "functionCalling": true
                            },
                            "taskFamilies": ["chat"],
                            "inputModalities": ["text"],
                            "outputModalities": ["text"],
                            "runtimeFeatures": ["streaming", "tool_calling"]
                        })),
                        tool_call_strategy: Some(
                            app_server_protocol::RuntimeToolCallStrategy::Native,
                        ),
                        ..app_server_protocol::RuntimeProviderConfig::default()
                    }),
                    provider_preference: Some("fixture-openai".to_string()),
                    model_preference: Some("lime-fixture-chat".to_string()),
                    ..app_server_protocol::RuntimeRequest::default()
                }),
                ..RuntimeOptions::default()
            }),
            expected_output: None,
            structured_output: None,
            output_schema: None,
            event_name: None,
            queued_turn_id: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
            agent_control_gateway: None,
        }
    }
}
