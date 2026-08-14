use super::*;
use crate::{ExecutionRequest, RuntimeHostContext};
use app_server_protocol::{
    AgentSession, AgentSessionStatus, AgentTurn, AgentTurnStatus, RuntimeOptions,
    RuntimeProviderConfig, RuntimeRequest,
};
use serde_json::{json, Value};

fn request_for_presentation_test(
    runtime_request: Option<RuntimeRequest>,
    metadata: Option<Value>,
) -> ExecutionRequest {
    let runtime_request = match (runtime_request, metadata) {
        (Some(mut runtime_request), Some(metadata)) => {
            runtime_request.metadata = Some(metadata);
            Some(runtime_request)
        }
        (Some(runtime_request), None) => Some(runtime_request),
        (None, Some(metadata)) => Some(RuntimeRequest {
            metadata: Some(metadata),
            ..RuntimeRequest::default()
        }),
        (None, None) => None,
    };
    ExecutionRequest {
        host: RuntimeHostContext::default(),
        session: AgentSession {
            session_id: "session-1".to_string(),
            thread_id: "thread-1".to_string(),
            app_id: "content-studio".to_string(),
            workspace_id: Some("workspace-main".to_string()),
            business_object_ref: None,
            status: AgentSessionStatus::Running,
            created_at: "2026-07-03T00:00:00.000Z".to_string(),
            updated_at: "2026-07-03T00:00:00.000Z".to_string(),
        },
        turn: AgentTurn {
            turn_id: "turn-1".to_string(),
            session_id: "session-1".to_string(),
            thread_id: "thread-1".to_string(),
            status: AgentTurnStatus::Accepted,
            started_at: None,
            completed_at: None,
        },
        forked_from_thread_id: None,
        input: agent_runtime::reply_input::RuntimeReplyInput::text("@配图 画一张深圳夏天的图"),
        runtime_options: Some(RuntimeOptions {
            stream: true,
            runtime_request,
            ..RuntimeOptions::default()
        }),
        event_name: None,
        expected_output: None,
        structured_output: None,
        output_schema: None,
        queued_turn_id: None,
        queue_if_busy: false,
        skip_pre_submit_resume: false,
        agent_control_gateway: None,
        rollout_budget_reminder_source: None,
    }
}

#[test]
fn parses_and_normalizes_model_generated_presentation() {
    let presentation = parse_generated_presentation(
        r#"{"assistant_intro":"好啊，我来处理这张深圳夏天的画面。","completion_caption":"搞定，深圳夏天的阳光和城市感都放进去了。\n还想更清爽或更电影感，可以继续调。"}"#,
        "openai",
        "gpt-4.1",
        PresentationLanguage::ChineseSimplified,
    )
    .expect("presentation");

    assert_eq!(
        presentation.assistant_intro.as_deref(),
        Some("好啊，我来处理这张深圳夏天的画面。")
    );
    assert_eq!(
        presentation.payload["result_captions"]["complete"].as_str(),
        Some("搞定，深圳夏天的阳光和城市感都放进去了。\n还想更清爽或更电影感，可以继续调。")
    );
}

#[test]
fn generated_presentation_payload_carries_soul_surface_contract() {
    let presentation = parse_generated_presentation(
        r#"{"assistant_intro":"好啊，我来画。","completion_caption":"完成了，可以继续调。"}"#,
        "openai",
        "gpt-4.1",
        PresentationLanguage::ChineseSimplified,
    )
    .expect("presentation");

    assert_eq!(
        presentation.payload["schemaVersion"],
        "lime.image_generation.presentation.v1"
    );
    assert_eq!(presentation.payload["surface"], "image_generation");
    assert_eq!(
        presentation.payload["styleLevels"]["title"]["styleLevel"],
        "L0"
    );
    assert_eq!(
        presentation.payload["styleLevels"]["runningStatus"]["styleLevel"],
        "L1"
    );
    assert_eq!(
        presentation.payload["styleLevels"]["assistantIntro"]["styleLevel"],
        "L2"
    );
    assert_eq!(
        presentation.payload["styleLevels"]["completionCaption"]["styleLevel"],
        "L2"
    );
    assert_eq!(
        presentation.payload["styleLevels"]["mediaArtifact"]["styleLevel"],
        "L3"
    );
    assert_eq!(
        presentation.payload["generationBriefBoundary"]["formalArtifactVoiceSource"],
        "generation_brief_only"
    );
    assert_eq!(
        presentation.payload["generationBriefBoundary"]["productSoulDefault"],
        "interaction_only"
    );
    assert_eq!(
        presentation.payload["image_generation_presentation_facts"]["mediaArtifactStyleLevel"],
        "L3"
    );
}

#[test]
fn generated_presentation_payload_keeps_style_pack_metadata() {
    let soul_style = SoulStyleMetadata {
        profile_id: Some("cheeky_sassy_executor".to_string()),
        pack_id: Some("com.lime.soul.cheeky-sassy-executor".to_string()),
        tone_variant: Some("cheeky_sassy".to_string()),
    };
    let presentation = parse_generated_presentation_with_soul_style(
        r#"{"assistant_intro":"好啊，我来画。","completion_caption":"完成了，可以继续调。"}"#,
        "openai",
        "gpt-4.1",
        PresentationLanguage::ChineseSimplified,
        Some(&soul_style),
    )
    .expect("presentation");

    assert_eq!(
        presentation.payload["soul_lifecycle"]["profileId"],
        "cheeky_sassy_executor"
    );
    assert_eq!(
        presentation.payload["soul_lifecycle"]["packId"],
        "com.lime.soul.cheeky-sassy-executor"
    );
    assert_eq!(
        presentation.payload["soul_lifecycle"]["toneVariant"],
        "cheeky_sassy"
    );
    assert_eq!(presentation.payload["profile_id"], "cheeky_sassy_executor");
    assert_eq!(
        presentation.payload["pack_id"],
        "com.lime.soul.cheeky-sassy-executor"
    );
    assert_eq!(presentation.payload["tone_variant"], "cheeky_sassy");
}

#[test]
fn rejects_internal_or_branded_visible_copy() {
    let raw = format!(
        r#"{{"assistant_intro":"{} 马上写入 JSONL。","completion_caption":"workflow 已完成"}}"#,
        concat!("R", "ibbi")
    );
    assert!(parse_generated_presentation(
        &raw,
        "openai",
        "gpt-4.1",
        PresentationLanguage::ChineseSimplified,
    )
    .is_none());
}

#[test]
fn merges_generated_fields_without_dropping_contract() {
    let generated = parse_generated_presentation(
        r#"{"assistant_intro":"好啊，我来画。","completion_caption":"完成了，可以继续调。"}"#,
        "openai",
        "gpt-4.1",
        PresentationLanguage::ChineseSimplified,
    )
    .expect("presentation");

    let merged = merge_generated_presentation(
        Some(json!({
            "version": "lime-image-chat-v1",
            "assistant_intro_request": {
                "source": "model_generated_before_tool"
            }
        })),
        &generated,
    )
    .expect("merged");

    assert_eq!(merged["version"], "lime-image-chat-v1");
    assert_eq!(merged["assistant_intro"], "好啊，我来画。");
}

#[test]
fn generated_payload_excludes_workflow_audit_facts() {
    let presentation = parse_generated_presentation(
        r#"{
            "planning_summary": "用明亮的城市光线和夏日空气感来组织画面。",
            "assistant_intro": "好啊，我来画这张深圳夏天的照片。",
            "completion_caption": "完成了，画面保留了深圳夏天的通透感。",
            "workflowRunId": "image-command-run-turn-1",
            "requestId": "approval-1",
            "redaction": { "policy": "workflow_audit_metadata_only" }
        }"#,
        "openai",
        "gpt-4.1",
        PresentationLanguage::ChineseSimplified,
    )
    .expect("presentation");

    assert_eq!(
        presentation.payload["planning_summary"].as_str(),
        Some("用明亮的城市光线和夏日空气感来组织画面。")
    );
    for key in [
        "workflowRunId",
        "workflow_run_id",
        "requestId",
        "request_id",
        "redaction",
    ] {
        assert!(
            presentation.payload.get(key).is_none(),
            "generated presentation payload must not expose audit field {key}"
        );
    }
}

#[test]
fn rejects_english_presentation_for_chinese_request() {
    assert!(parse_generated_presentation(
        r#"{"assistant_intro":"Sure, let's generate the Shenzhen summer photo.","completion_caption":"Done, the Shenzhen summer photo is ready."}"#,
        "openai",
        "gpt-4.1",
        PresentationLanguage::ChineseSimplified,
    )
    .is_none());
}

#[test]
fn allows_model_labels_inside_chinese_presentation() {
    let presentation = parse_generated_presentation(
        r#"{"assistant_intro":"好啊，用 Agnes Image 2.1 Flash 给你生成这张深圳夏天照片。","completion_caption":"搞定，深圳夏天的城市感和明亮空气已经出来了。"}"#,
        "openai",
        "gpt-4.1",
        PresentationLanguage::ChineseSimplified,
    )
    .expect("presentation");

    assert_eq!(
        presentation.assistant_intro.as_deref(),
        Some("好啊，用 Agnes Image 2.1 Flash 给你生成这张深圳夏天照片。")
    );
    assert_eq!(
        presentation.completion_caption.as_deref(),
        Some("搞定，深圳夏天的城市感和明亮空气已经出来了。")
    );
}

#[test]
fn presentation_prompt_carries_language_contract() {
    assert_eq!(
        detect_presentation_language("用 Agnes 生成一张深圳夏天照片"),
        PresentationLanguage::ChineseSimplified
    );
    let system_prompt = presentation_system_prompt();
    assert!(system_prompt.contains("Simplified Chinese"));
    assert!(system_prompt.contains("Sure"));
    assert!(system_prompt.contains("Done"));
}

#[test]
fn presentation_selection_prefers_text_slot_over_image_runtime_preference() {
    let mut request = request_for_presentation_test(
        None,
        Some(json!({
            "harness": {
                "modelSlots": {
                    "fast": {
                        "provider": "openai",
                        "model": "gpt-4.1-mini"
                    },
                    "base": {
                        "provider": "anthropic",
                        "model": "claude-sonnet-4"
                    }
                }
            }
        })),
    );
    let options = request.runtime_options.as_mut().expect("runtime options");
    options.runtime_request_mut().provider_preference = Some("agnes".to_string());
    options.runtime_request_mut().model_preference = Some("agnes-image-2.1-flash".to_string());

    let selection = resolve_presentation_model_selection(&request).expect("selection");

    assert_eq!(selection.provider, "openai");
    assert_eq!(selection.model, "gpt-4.1-mini");
    assert_eq!(selection.source, "profile_model_slot");
}

#[test]
fn presentation_selection_rejects_image_only_runtime_request_config() {
    let request = request_for_presentation_test(
        Some(RuntimeRequest {
            provider_config: Some(RuntimeProviderConfig {
                provider_id: Some("agnes".to_string()),
                provider_name: Some("openai".to_string()),
                model_name: Some("agnes-image-2.1-flash".to_string()),
                api_key: Some("sk-test".to_string()),
                base_url: Some("https://apihub.agnes-ai.com/v1".to_string()),
                ..RuntimeProviderConfig::default()
            }),
            provider_preference: Some("agnes".to_string()),
            model_preference: Some("agnes-image-2.1-flash".to_string()),
            ..RuntimeRequest::default()
        }),
        None,
    );

    let error = resolve_presentation_model_selection(&request).expect_err("image only");

    assert!(error
        .to_string()
        .contains("presentation_text_model_unavailable"));
}

#[test]
fn presentation_selection_keeps_agnes_text_runtime_request_model_without_image_word() {
    let request = request_for_presentation_test(
        Some(RuntimeRequest {
            provider_config: Some(RuntimeProviderConfig {
                provider_id: Some("custom-agnes-provider".to_string()),
                provider_name: Some("openai".to_string()),
                model_name: Some("agnes-2.0-flash".to_string()),
                api_key: Some("sk-test".to_string()),
                base_url: Some("https://apihub.agnes-ai.com/v1".to_string()),
                ..RuntimeProviderConfig::default()
            }),
            provider_preference: Some("custom-agnes-provider".to_string()),
            model_preference: Some("agnes-2.0-flash".to_string()),
            ..RuntimeRequest::default()
        }),
        None,
    );

    let selection = resolve_presentation_model_selection(&request).expect("selection");

    assert_eq!(selection.provider, "custom-agnes-provider");
    assert_eq!(selection.model, "agnes-2.0-flash");
}

#[test]
fn presentation_selection_skips_image_fast_slot_and_uses_base_slot() {
    let request = request_for_presentation_test(
        None,
        Some(json!({
            "harness": {
                "modelSlots": {
                    "fast": {
                        "provider": "agnes",
                        "model": "agnes-image-2.1-flash"
                    },
                    "base": {
                        "provider": "openai",
                        "model": "gpt-4.1-mini"
                    }
                }
            }
        })),
    );

    let selection = resolve_presentation_model_selection(&request).expect("selection");

    assert_eq!(selection.provider, "openai");
    assert_eq!(selection.model, "gpt-4.1-mini");
}

#[test]
fn presentation_selection_allows_text_runtime_request_direct_config() {
    let request = request_for_presentation_test(
        Some(RuntimeRequest {
            provider_config: Some(RuntimeProviderConfig {
                provider_id: Some("fixture-openai".to_string()),
                provider_name: Some("openai".to_string()),
                model_name: Some("lime-fixture-chat".to_string()),
                api_key: Some("sk-test".to_string()),
                base_url: Some("http://127.0.0.1:56599".to_string()),
                ..RuntimeProviderConfig::default()
            }),
            provider_preference: Some("fixture-openai".to_string()),
            model_preference: Some("lime-fixture-chat".to_string()),
            ..RuntimeRequest::default()
        }),
        None,
    );

    let selection = resolve_presentation_model_selection(&request).expect("selection");

    assert_eq!(selection.provider, "fixture-openai");
    assert_eq!(selection.model, "lime-fixture-chat");
    assert_eq!(selection.source, "runtime_request_provider_config");
}
