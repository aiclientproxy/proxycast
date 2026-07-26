use super::launch_gate::validate_worker_cloud_release_signature;
use super::*;
use crate::runtime::timestamp;
use app_server_protocol::AgentSession;
use app_server_protocol::AgentSessionStatus;
use app_server_protocol::AgentTurn;
use app_server_protocol::AgentTurnStatus;

#[test]
fn worker_delta_workspace_snapshot_passes_through_without_resplitting() {
    let document_text = "# 草稿\n\n第一段用于验证 worker 已经按段落输出 partial snapshot，App Server 不应再拿当前 partial 做比例切片。\n\n第二段用于验证同一个 artifact 继续增长，sequence 由 worker 负责维护。";
    let event = RuntimeEvent::new(
        "artifact.snapshot",
        json!({
            "artifact": {
                "artifactId": "task-article-1:workspace-patch",
                "status": "streaming",
                "metadata": {
                    "complete": false,
                    "writePhase": "streaming",
                    "contentStatus": "streaming",
                    "streamSource": "worker_delta",
                    "streamSequence": 7,
                    "contentFactoryWorkspacePatch": {
                        "objects": [
                            {
                                "ref": {
                                    "kind": "articleDraft"
                                },
                                "source": {
                                    "documentText": document_text,
                                    "finalMarkdown": document_text
                                }
                            }
                        ]
                    }
                }
            }
        }),
    );

    let events = worker_progress_events_for_sink(event, None).expect("progress events");

    assert_eq!(events.len(), 1);
    let artifact = &events[0].payload["artifact"];
    assert_eq!(artifact["metadata"]["streamSource"], "worker_delta");
    assert_eq!(artifact["metadata"]["streamSequence"], 7);
    assert_eq!(
        artifact["metadata"]["contentFactoryWorkspacePatch"]["objects"][0]["source"]
            ["documentText"],
        document_text
    );
    assert_eq!(
        artifact["filePath"],
        ".lime/artifacts/content-factory/workspace-patch.json"
    );
}

#[test]
fn complete_workspace_snapshot_passes_through_without_fake_streaming() {
    let document_text = "# 草稿\n\n第一段是 worker 最终正文。\n\n第二段也是最终正文。";
    let event = RuntimeEvent::new(
        "artifact.snapshot",
        json!({
            "artifact": {
                "artifactId": "task-article-1:workspace-patch",
                "status": "ready",
                "metadata": {
                    "complete": true,
                    "contentFactoryWorkspacePatch": {
                        "objects": [
                            {
                                "ref": {
                                    "kind": "articleDraft"
                                },
                                "source": {
                                    "documentText": document_text,
                                    "finalMarkdown": document_text
                                }
                            }
                        ]
                    }
                }
            }
        }),
    );

    let events = worker_progress_events_for_sink(event, None).expect("progress events");

    assert_eq!(events.len(), 1);
    let artifact = &events[0].payload["artifact"];
    assert_eq!(artifact["metadata"]["complete"], true);
    assert!(artifact["metadata"].get("streamSequence").is_none());
    assert_eq!(
        artifact["metadata"]["contentFactoryWorkspacePatch"]["objects"][0]["source"]
            ["documentText"],
        document_text
    );
    assert_eq!(
        artifact["filePath"],
        ".lime/artifacts/content-factory/workspace-patch.json"
    );
}

#[test]
fn workflow_worker_progress_without_context_fails_closed() {
    let error = worker_progress_events_for_sink(
        RuntimeEvent::new(
            "workflow.tool.completed",
            json!({
                "stepId": "research",
                "toolName": "WebSearch"
            }),
        ),
        None,
    )
    .expect_err("workflow progress requires plugin workflow context");

    assert!(error
        .to_string()
        .contains("without plugin workflow context"));
}

#[tokio::test]
async fn rejects_worker_turn_when_content_factory_is_not_installed() {
    let request = execution_request(json!({
        "plugin": {
            "source": "right_surface_article_workspace",
            "app_id": WORKER_APP_ID,
            "article_workspace_action": {
                "key": "regenerate",
                "task_kind": "content.image.generate",
                "output_artifact_kind": WORKSPACE_PATCH_KIND,
                "prompt": "重新生成配图"
            }
        },
        "right_surface": {
            "surface_kind": "articleWorkspace"
        }
    }));
    let mut sink = TestRuntimeEventSink::default();

    let handled = RuntimeCore::default()
        .maybe_run_plugin_worker_turn(&request, &mut sink)
        .await
        .expect("worker dispatch check");

    assert!(handled);
    assert_eq!(
        sink.events
            .iter()
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        vec!["turn.accepted", "runtime.error", "turn.failed"]
    );
}

#[test]
fn extracts_content_factory_article_workspace_worker_turn() {
    let request = execution_request(json!({
        "plugin": {
            "source": "right_surface_article_workspace",
            "app_id": WORKER_APP_ID,
            "session_id": "session-content-factory",
            "workspace_id": "workspace-main",
            "article_workspace_action": {
                "key": "regenerate",
                "task_kind": "content.image.generate",
                "output_artifact_kind": WORKSPACE_PATCH_KIND,
                "prompt": "重新生成配图",
                "object": {
                    "app_id": WORKER_APP_ID,
                    "kind": "imageGenerationSet",
                    "id": "image-set-1",
                    "session_id": "session-content-factory",
                    "artifact_ids": ["artifact-image-set-1"]
                }
            }
        },
        "right_surface": {
            "surface_kind": "articleWorkspace"
        }
    }));

    let worker_turn = PaneActionWorkerTurn::from_execution_request(&request).expect("worker turn");

    assert_eq!(worker_turn.app_id, WORKER_APP_ID);
    assert_eq!(worker_turn.action_key.as_deref(), Some("regenerate"));
    assert_eq!(worker_turn.task_kind, "content.image.generate");
    assert_eq!(worker_turn.workspace_id.as_deref(), Some("workspace-main"));
    assert_eq!(
        worker_turn.surface_kind.as_deref(),
        Some("articleWorkspace")
    );
    assert_eq!(worker_turn.pane_kind.as_deref(), Some("imageGenerationSet"));
    assert_eq!(
        worker_turn.output_artifact_kind.as_deref(),
        Some(WORKSPACE_PATCH_KIND)
    );
    assert_eq!(
        worker_turn.source_artifact_ids,
        vec!["artifact-image-set-1"]
    );
    assert_eq!(
        worker_turn.source_object_ref.unwrap()["kind"].as_str(),
        Some("imageGenerationSet")
    );
}

#[test]
fn canonicalizes_legacy_creator_workspace_patch_kind_for_content_factory_action() {
    let request = execution_request(json!({
        "plugin": {
            "source": "right_surface_article_workspace",
            "app_id": WORKER_APP_ID,
            "session_id": "session-content-factory",
            "workspace_id": "workspace-main",
            "article_workspace_action": {
                "key": "regenerate",
                "task_kind": "content.image.generate",
                "output_artifact_kind": LEGACY_CREATOR_WORKSPACE_PATCH_KIND,
                "prompt": "重新生成配图",
                "object": {
                    "app_id": WORKER_APP_ID,
                    "kind": "imageGenerationSet",
                    "id": "image-set-1",
                    "session_id": "session-content-factory",
                    "artifact_ids": ["artifact-image-set-1"]
                }
            }
        },
        "right_surface": {
            "surface_kind": "articleWorkspace"
        }
    }));

    let worker_turn = PaneActionWorkerTurn::from_execution_request(&request).expect("worker turn");

    assert_eq!(worker_turn.app_id, WORKER_APP_ID);
    assert_eq!(
        worker_turn.output_artifact_kind.as_deref(),
        Some(WORKSPACE_PATCH_KIND)
    );
    assert!(validate_worker_turn_runtime_contract(
        &worker_turn,
        &app_server_protocol::PluginTaskRuntimeContract {
            output_artifact_kind: Some(WORKSPACE_PATCH_KIND.to_string()),
            ..Default::default()
        },
    )
    .is_ok());
}

#[test]
fn article_workspace_worker_request_resolves_manifest_workflow_defaults() {
    let request = execution_request(json!({
        "plugin": {
            "source": "right_surface_article_workspace",
            "app_id": WORKER_APP_ID,
            "workspace_id": "workspace-main",
            "article_workspace_action": {
                "key": "write_article",
                "intent": "write_article",
                "risk": "write",
                "task_kind": "content.article.generate",
                "output_artifact_kind": WORKSPACE_PATCH_KIND,
                "prompt": "写一篇关于内容工厂插件编排的文章",
                "object": {
                    "app_id": WORKER_APP_ID,
                    "kind": "articleDraft",
                    "id": "article-1",
                    "session_id": "session-content-factory"
                }
            }
        },
        "right_surface": {
            "surface_kind": "articleWorkspace"
        }
    }));
    let installed_state = json!({
        "manifest": {
            "agentRuntime": {
                "workflows": [
                    {
                        "key": "content_article_workflow",
                        "taskKind": "content.article.generate",
                        "cliRefs": ["content-factory"],
                        "connectorRefs": ["lime-knowledge", "web-research"],
                        "hookPolicy": {
                            "prompt": ["prompt-submit"],
                            "task": ["task-complete"]
                        },
                        "steps": [
                            {
                                "id": "draft",
                                "subagent": "article-writer",
                                "skillRefs": ["article-writing"]
                            },
                            {
                                "id": "image-plan",
                                "subagent": "image-planner",
                                "skillRefs": ["article-image-plan"]
                            }
                        ]
                    }
                ]
            }
        }
    });

    let worker_turn = PaneActionWorkerTurn::from_execution_request(&request).expect("worker turn");
    let worker_request = worker_turn.worker_request(
        request.session.session_id.as_str(),
        request.turn.turn_id.as_str(),
        Some("./src/runtime/content-factory-worker.mjs"),
        &installed_state,
    );

    assert_eq!(worker_request["workflowKey"], "content_article_workflow");
    assert_eq!(worker_request["hookPolicy"]["prompt"][0], "prompt-submit");
    assert_eq!(worker_request["cliRefs"][0], "content-factory");
    assert_eq!(worker_request["connectorRefs"][1], "web-research");
    assert!(worker_request["subagents"]
        .as_array()
        .expect("subagents")
        .iter()
        .any(|value| value == "article-writer"));
    assert_eq!(
        worker_request["skillRefs"]
            .as_array()
            .expect("plugin workflow skill refs")
            .len(),
        2
    );
    assert!(worker_request["skillRefs"]
        .as_array()
        .expect("plugin workflow skill refs")
        .iter()
        .any(|value| value == "article-image-plan"));
    assert_eq!(
        worker_request["orchestration"]
            .as_array()
            .expect("plugin workflow steps")
            .len(),
        2
    );
}

#[test]
fn ignores_content_factory_plugin_activation_for_agent_turn() {
    let request = execution_request(json!({
        "harness": {
            "plugin_activation": {
                "source": "plugin_explicit_mention",
                "trigger": "@内容工厂",
                "body": "写一篇公众号文章",
                "session_id": "session-content-factory",
                "plugin_id": WORKER_APP_ID,
                "active_plugin_id": WORKER_APP_ID,
                "active_entry_key": "content_factory",
                "intent_key": "content_article_generate",
                "task_kind": "content.article.generate",
                "output_artifact_kind": WORKSPACE_PATCH_KIND,
                "right_surface": "articleWorkspace",
                "expected_objects": ["articleDraft"],
                "selected_object_ref": {
                    "plugin_id": WORKER_APP_ID,
                    "object_kind": "articleDraft",
                    "object_id": "pending"
                },
                "opened_tabs": ["articleWorkspace"],
                "context_source": "user"
            }
        }
    }));

    assert!(
        PaneActionWorkerTurn::from_execution_request(&request).is_none(),
        "plugin activation is Agent context, not a worker turn"
    );
}

#[test]
fn ignores_generic_plugin_activation_for_agent_turn() {
    let request = execution_request(json!({
        "harness": {
            "plugin_activation": {
                "source": "plugin_explicit_mention",
                "trigger": "@其他插件",
                "body": "写文章",
                "session_id": "session-other",
                "plugin_id": "other-plugin",
                "active_entry_key": "other",
                "output_artifact_kind": "other.workspace_patch"
            }
        }
    }));

    assert!(
        PaneActionWorkerTurn::from_execution_request(&request).is_none(),
        "plugin activation should not bypass the normal Agent backend"
    );
}

#[test]
fn extracts_content_factory_custom_pane_action_worker_turn() {
    let request = execution_request(json!({
        "plugin": {
            "source": PANE_ACTION_SOURCE,
            "app_id": WORKER_APP_ID,
            "session_id": "session-content-factory",
            "workspace_id": "workspace-main",
            "pane_action": {
                "key": "regenerate",
                "intent": "regenerate",
                "risk": "write",
                "task_kind": "content.image.generate",
                "output_artifact_kind": WORKSPACE_PATCH_KIND,
                "prompt": "重新生成配图",
                "surface_kind": "appSurface",
                "pane_kind": "imageGrid",
                "source_artifact_ids": ["artifact-image-set-1", "artifact-image-set-1"],
                "object": {
                    "app_id": WORKER_APP_ID,
                    "kind": "imageGenerationSet",
                    "id": "image-set-1",
                    "session_id": "session-content-factory"
                }
            }
        },
        "right_surface": {
            "surface_kind": "appSurface",
            "pane_kind": "imageGrid"
        }
    }));

    let worker_turn = PaneActionWorkerTurn::from_execution_request(&request).expect("worker turn");
    let worker_request = worker_turn.worker_request(
        request.session.session_id.as_str(),
        request.turn.turn_id.as_str(),
        Some("./src/runtime/content-factory-worker.mjs"),
        &json!({}),
    );

    assert_eq!(worker_turn.app_id, WORKER_APP_ID);
    assert_eq!(worker_turn.source, PANE_ACTION_SOURCE);
    assert_eq!(worker_turn.action_key.as_deref(), Some("regenerate"));
    assert_eq!(worker_turn.action_intent.as_deref(), Some("regenerate"));
    assert_eq!(worker_turn.action_risk.as_deref(), Some("write"));
    assert_eq!(worker_turn.surface_kind.as_deref(), Some("appSurface"));
    assert_eq!(worker_turn.pane_kind.as_deref(), Some("imageGrid"));
    assert_eq!(
        worker_turn.output_artifact_kind.as_deref(),
        Some(WORKSPACE_PATCH_KIND)
    );
    assert_eq!(
        worker_turn.source_artifact_ids,
        vec!["artifact-image-set-1"]
    );
    assert_eq!(worker_request["surfaceKind"], "appSurface");
    assert_eq!(worker_request["paneKind"], "imageGrid");
    assert_eq!(
        worker_request["sourceArtifactIds"][0],
        "artifact-image-set-1"
    );
    assert_eq!(worker_request["outputArtifactKind"], WORKSPACE_PATCH_KIND);
    assert_eq!(
        worker_request["expectedOutput"]["artifactKind"],
        WORKSPACE_PATCH_KIND
    );
    assert_eq!(
        worker_request["runtime"]["outputArtifactKind"],
        WORKSPACE_PATCH_KIND
    );
}

#[test]
fn defers_pane_action_output_artifact_kind_validation_to_runtime_contract() {
    let request = execution_request(json!({
        "plugin": {
            "source": PANE_ACTION_SOURCE,
            "app_id": WORKER_APP_ID,
            "pane_action": {
                "key": "regenerate",
                "task_kind": "content.image.generate",
                "output_artifact_kind": "other.workspace_patch",
                "prompt": "重新生成配图",
                "surface_kind": "appSurface",
                "pane_kind": "imageGrid"
            }
        },
        "right_surface": {
            "surface_kind": "appSurface",
            "pane_kind": "imageGrid"
        }
    }));

    let worker_turn = PaneActionWorkerTurn::from_execution_request(&request).expect("worker turn");
    assert_eq!(
        worker_turn.output_artifact_kind.as_deref(),
        Some("other.workspace_patch")
    );

    let error = validate_worker_turn_runtime_contract(
        &worker_turn,
        &app_server_protocol::PluginTaskRuntimeContract {
            output_artifact_kind: Some(WORKSPACE_PATCH_KIND.to_string()),
            ..Default::default()
        },
    )
    .expect_err("output kind mismatch should fail closed");

    assert!(error
        .to_string()
        .contains("unsupported by runtime contract"));
}

#[test]
fn extracts_generic_plugin_pane_action_worker_turn() {
    let request = execution_request(json!({
        "plugin": {
            "source": PANE_ACTION_SOURCE,
            "app_id": "creator-pack",
            "pane_action": {
                "key": "regenerate",
                "task_kind": "creator.generate",
                "output_artifact_kind": "creator.workspace_patch",
                "prompt": "重新生成内容",
                "surface_kind": "appSurface",
                "pane_kind": "creatorCanvas"
            }
        },
        "right_surface": {
            "surface_kind": "appSurface",
            "pane_kind": "creatorCanvas"
        }
    }));

    let worker_turn = PaneActionWorkerTurn::from_execution_request(&request).expect("worker turn");
    assert_eq!(worker_turn.app_id, "creator-pack");
    assert_eq!(
        worker_turn.output_artifact_kind.as_deref(),
        Some("creator.workspace_patch")
    );

    let worker_request = worker_turn.worker_request(
        request.session.session_id.as_str(),
        request.turn.turn_id.as_str(),
        Some("./worker.mjs"),
        &json!({}),
    );
    assert_eq!(
        worker_request["expectedOutput"],
        json!({ "artifactKind": "creator.workspace_patch" })
    );
}

#[test]
fn classifies_worker_failures_for_retry_projection() {
    let timeout = classify_worker_failure("Plugin worker timed out after 100ms");
    assert_eq!(timeout.error_code, "PLUGIN_WORKER_TIMEOUT");
    assert_eq!(timeout.category, "timeout");
    assert!(timeout.retryable);
    assert_eq!(timeout.retry_advice, "retry_same_action");
    assert_eq!(timeout.retry_max_attempts, 1);

    let blocker = classify_worker_failure("Plugin worker runtime has blockers: key");
    assert_eq!(blocker.error_code, "PLUGIN_WORKER_BLOCKED");
    assert_eq!(blocker.category, "configuration");
    assert!(!blocker.retryable);
    assert_eq!(blocker.retry_advice, "resolve_runtime_blocker");

    let unsupported =
        classify_worker_failure("Plugin worker direct provider access is unsupported.");
    assert_eq!(unsupported.error_code, "PLUGIN_WORKER_CONTRACT_UNSUPPORTED");
    assert_eq!(unsupported.category, "configuration");
    assert!(!unsupported.retryable);
    assert_eq!(unsupported.retry_advice, "fix_runtime_contract");

    let host_generation =
        classify_worker_failure("Plugin worker did not complete: HOST_MANAGED_GENERATION_REQUIRED");
    assert_eq!(
        host_generation.error_code,
        "PLUGIN_WORKER_HOST_GENERATION_UNAVAILABLE"
    );
    assert_eq!(host_generation.category, "host_generation_unavailable");
    assert!(!host_generation.retryable);
    assert_eq!(host_generation.retry_advice, "configure_host_generation");

    let retryable = classify_worker_failure("Plugin worker did not complete: WORKER_RETRYABLE");
    assert_eq!(retryable.error_code, "PLUGIN_WORKER_RETRYABLE_FAILURE");
    assert_eq!(retryable.category, "worker_retryable");
    assert!(retryable.retryable);
    assert_eq!(retryable.retry_advice, "retry_same_action");
    assert_eq!(retryable.retry_max_attempts, 1);
    assert!(retryable.should_retry());
    assert!(!retryable.with_retry_attempt(1).should_retry());

    let invalid_output =
        classify_worker_failure("failed to decode Plugin worker response: expected value");
    assert_eq!(invalid_output.error_code, "PLUGIN_WORKER_OUTPUT_INVALID");
    assert_eq!(invalid_output.category, "worker_output");
    assert!(!invalid_output.retryable);
    assert_eq!(invalid_output.retry_advice, "inspect_worker_output");
}

#[test]
fn accepts_verified_cloud_release_signature_evidence_for_worker() {
    let installed_state = json!({
        "schemaVersion": "plugin.installed-state.v1",
        "appId": "content-factory-app",
        "identity": {
            "appId": "content-factory-app",
            "sourceKind": "cloud_release",
            "sourceUri": "https://lime.local/plugins/content-factory-app/releases/0.3.0/package.zip",
            "packageHash": "sha256:test-package",
            "manifestHash": "sha256:test-manifest"
        },
        "setup": {
            "cloudReleaseEvidence": {
                "status": "ready",
                "signaturePolicy": "required",
                "signatureVerificationStatus": "verified",
                "packageHashMatched": true,
                "manifestHashMatched": true,
                "packageVerificationStatus": "verified"
            }
        }
    });

    validate_worker_cloud_release_signature(&installed_state)
        .expect("verified evidence should pass");
}

#[test]
fn rejects_optional_seeded_cloud_release_signature_warning_for_worker() {
    let installed_state = json!({
        "schemaVersion": "plugin.installed-state.v1",
        "appId": "content-factory-app",
        "identity": {
            "appId": "content-factory-app",
            "sourceKind": "cloud_release",
            "sourceUri": "https://seeded.local/plugins/content-factory-app/2.0.0.lapp",
            "packageHash": "sha256:test-package",
            "manifestHash": "sha256:test-manifest"
        },
        "setup": {
            "cloudReleaseEvidence": {
                "status": "warning",
                "signaturePolicy": "optional",
                "signatureVerificationStatus": "not_configured",
                "packageHashMatched": true,
                "manifestHashMatched": true,
                "packageVerificationStatus": "verified"
            }
        }
    });

    let error = validate_worker_cloud_release_signature(&installed_state)
        .expect_err("optional seeded signature warning must not launch worker");
    let message = error.to_string();
    assert!(message.contains("signature policy is not required"));
    assert!(message.contains("signature is not verified"));
    assert!(message.contains("release evidence is not ready"));
}

#[test]
fn rejects_cloud_release_worker_without_release_evidence() {
    let installed_state = json!({
        "schemaVersion": "plugin.installed-state.v1",
        "appId": "content-factory-app",
        "identity": {
            "appId": "content-factory-app",
            "sourceKind": "cloud_release",
            "sourceUri": "https://seeded.local/plugins/content-factory-app/2.0.0.lapp",
            "packageHash": "sha256:test-package",
            "manifestHash": "sha256:test-manifest"
        },
        "setup": {}
    });

    let error = validate_worker_cloud_release_signature(&installed_state)
        .expect_err("missing evidence should fail closed");
    assert!(error.to_string().contains("missing cloud release evidence"));
}

#[test]
fn rejects_cloud_release_worker_without_verified_signature_evidence() {
    let installed_state = json!({
        "schemaVersion": "plugin.installed-state.v1",
        "appId": "content-factory-app",
        "identity": {
            "appId": "content-factory-app",
            "sourceKind": "cloud_release",
            "sourceUri": "https://lime.local/plugins/content-factory-app/releases/0.3.0/package.zip",
            "packageHash": "sha256:test-package",
            "manifestHash": "sha256:test-manifest"
        },
        "setup": {
            "cloudReleaseEvidence": {
                "status": "blocked",
                "signaturePolicy": "required",
                "signatureVerificationStatus": "declared",
                "packageHashMatched": true,
                "manifestHashMatched": true,
                "packageVerificationStatus": "verified"
            }
        }
    });

    let error = validate_worker_cloud_release_signature(&installed_state)
        .expect_err("unverified evidence should fail");
    let failure = classify_worker_failure(error.to_string().as_str());

    assert_eq!(
        failure.error_code,
        "PLUGIN_WORKER_PACKAGE_SIGNATURE_UNVERIFIED"
    );
    assert_eq!(failure.category, "configuration");
    assert_eq!(failure.retry_advice, "reinstall_verified_package");
    assert!(!failure.retryable);
}

fn execution_request(metadata: Value) -> ExecutionRequest {
    ExecutionRequest {
        host: super::super::RuntimeHostContext::default(),
        session: AgentSession {
            session_id: "session-content-factory".to_string(),
            thread_id: "thread-content-factory".to_string(),
            app_id: WORKER_APP_ID.to_string(),
            workspace_id: Some("workspace-main".to_string()),
            business_object_ref: None,
            status: AgentSessionStatus::Running,
            created_at: timestamp(),
            updated_at: timestamp(),
        },
        turn: AgentTurn {
            turn_id: "turn-action-1".to_string(),
            session_id: "session-content-factory".to_string(),
            thread_id: "thread-content-factory".to_string(),
            status: AgentTurnStatus::Accepted,
            started_at: Some(timestamp()),
            completed_at: None,
        },
        forked_from_thread_id: None,
        input: agent_runtime::reply_input::RuntimeReplyInput::text("重新生成配图"),
        runtime_options: Some(app_server_protocol::RuntimeOptions {
            runtime_request: Some(app_server_protocol::RuntimeRequest {
                metadata: Some(metadata),
                ..app_server_protocol::RuntimeRequest::default()
            }),
            ..app_server_protocol::RuntimeOptions::default()
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

#[derive(Default)]
struct TestRuntimeEventSink {
    events: Vec<RuntimeEvent>,
}

impl RuntimeEventSink for TestRuntimeEventSink {
    fn emit(&mut self, event: RuntimeEvent) -> Result<(), RuntimeCoreError> {
        self.events.push(event);
        Ok(())
    }
}
