use super::presentation_soul::insert_image_generation_soul_payload_fields;
use super::{workflow_run_id, ImageCommandIntent};
use crate::runtime::memory_prompt::append_soul_context_to_system_prompt;
use crate::runtime_backend::{
    backend_error, current_agent_runtime_config_metadata, direct_provider_config_from_request,
    initialize_runtime_database, model_route_contract, model_route_resolver,
    request_context::{self, RuntimeModelSelection},
    tool_process_metadata::SoulStyleMetadata,
};
use crate::{ExecutionRequest, RuntimeCoreError};
use lime_agent::{
    insert_agent_turn_metadata, run_direct_text_generation_with_db,
    set_agent_turn_user_visible_input_text, AgentTokenUsage, DirectTextGenerationRequest,
};
use serde_json::{json, Map, Value};

const PRESENTATION_SCHEMA_VERSION: &str = "image_task_presentation.v1";
const PRESENTATION_SOURCE: &str = "model_generated";
const PRESENTATION_SOURCE_METADATA: &str = "metadata_provided";
const MAX_PLANNING_SUMMARY_CHARS: usize = 160;
const MAX_INTRO_CHARS: usize = 180;
const MAX_CAPTION_CHARS: usize = 220;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PresentationLanguage {
    ChineseSimplified,
    Japanese,
    Korean,
    English,
    Unknown,
}

impl PresentationLanguage {
    fn code(self) -> &'static str {
        match self {
            Self::ChineseSimplified => "zh-CN",
            Self::Japanese => "ja-JP",
            Self::Korean => "ko-KR",
            Self::English => "en-US",
            Self::Unknown => "same_as_user_request",
        }
    }

    fn rule(self) -> &'static str {
        match self {
            Self::ChineseSimplified => {
                "Output assistant_intro and completion_caption in Simplified Chinese. Do not start with English phrases like Sure, Done, or is ready."
            }
            Self::Japanese => {
                "Output assistant_intro and completion_caption in Japanese. Do not switch to English."
            }
            Self::Korean => {
                "Output assistant_intro and completion_caption in Korean. Do not switch to English."
            }
            Self::English => "Output assistant_intro and completion_caption in natural English.",
            Self::Unknown => {
                "Output assistant_intro and completion_caption in the same language as the user request."
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct GeneratedImageTaskPresentation {
    pub(super) planning_summary: Option<String>,
    pub(super) assistant_intro: Option<String>,
    pub(super) completion_caption: Option<String>,
    pub(super) payload: Value,
    pub(super) usage: Option<AgentTokenUsage>,
}

pub(super) async fn generate_image_task_presentation(
    runtime_backend: &super::super::RuntimeBackend,
    request: &ExecutionRequest,
    intent: &ImageCommandIntent,
) -> Result<Option<GeneratedImageTaskPresentation>, RuntimeCoreError> {
    let db = initialize_runtime_database(runtime_backend.db.as_ref())?;
    runtime_backend.ensure_agent_initialized(&db).await?;
    let requested_selection = resolve_presentation_model_selection(request)?;
    let host_request = request_context::runtime_request_from_request(request);
    let host_selection = request_context::selection_from_host_provider_config(request);
    let host_direct_provider_config = host_selection.as_ref().and_then(|selection| {
        direct_provider_config_from_request(
            host_request.as_ref(),
            selection,
            selection.reasoning_effort.clone(),
        )
    });
    let direct_provider_config = if host_selection
        .as_ref()
        .is_some_and(|selection| same_provider_model(selection, &requested_selection))
    {
        host_direct_provider_config
    } else {
        if host_direct_provider_config.is_some() {
            if let Some(selection) = host_selection.as_ref() {
                tracing::info!(
                    session_id = %intent.scope.session_id,
                    thread_id = %intent.scope.thread_id,
                    turn_id = %intent.scope.turn_id,
                    workflow_run_id = %workflow_run_id(&intent.scope),
                    host_provider = %selection.provider,
                    host_model = %selection.model,
                    selected_provider = %requested_selection.provider,
                    selected_model = %requested_selection.model,
                    reason_code = "presentation_direct_config_skipped_for_non_text_selection",
                    "[RuntimeBackend] ImageCommandWorkflow presentation skipped host direct provider config"
                );
            }
        }
        None
    };
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
            "presentation_text_route_unavailable: {}",
            route_failure.reason_code
        )));
    }

    let presentation_session_id = presentation_session_id(intent);
    let provider_configuration = model_route_contract::provider_configuration_from_runtime(
        &selection,
        &route_resolution.resolved_route,
        direct_provider_config,
        request_context::service_tier_from_request(request),
    );

    let config_metadata = current_agent_runtime_config_metadata();
    let mut turn_context = request_context::turn_context_from_request(
        request,
        host_request.as_ref(),
        &intent.scope,
        &selection,
        config_metadata.clone(),
    )
    .unwrap_or_default();
    set_agent_turn_user_visible_input_text(
        &mut turn_context,
        intent
            .raw_text
            .clone()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| Some(intent.prompt.clone())),
    );
    insert_agent_turn_metadata(
        &mut turn_context,
        "lime_runtime".to_string(),
        json!({
            "auto_compact": false,
            "tool_surface": "direct_answer",
            "source": "image_command_presentation",
        }),
    );
    insert_agent_turn_metadata(
        &mut turn_context,
        "image_command_presentation".to_string(),
        json!({
            "schema": PRESENTATION_SCHEMA_VERSION,
            "workflow_run_id": workflow_run_id(&intent.scope),
            "mode": intent.mode,
            "entry_source": intent.entry_source,
            "provider": selection.provider,
            "model": selection.model,
        }),
    );

    let runtime_metadata = request.runtime_metadata();
    let soul_style = image_generation_soul_style(config_metadata.as_ref(), runtime_metadata);
    let system_prompt = append_soul_context_to_system_prompt(
        Some(presentation_system_prompt()),
        config_metadata.as_ref(),
        runtime_metadata,
    )
    .unwrap_or_else(presentation_system_prompt);
    let presentation_language =
        detect_presentation_language(intent.raw_text.as_deref().unwrap_or(&intent.prompt));
    tracing::info!(
        session_id = %intent.scope.session_id,
        thread_id = %intent.scope.thread_id,
        turn_id = %intent.scope.turn_id,
        workflow_run_id = %workflow_run_id(&intent.scope),
        provider = %selection.provider,
        model = %selection.model,
        language = presentation_language.code(),
        "[RuntimeBackend] ImageCommandWorkflow presentation generation started"
    );
    let generated = run_direct_text_generation_with_db(
        &runtime_backend.agent_state,
        DirectTextGenerationRequest {
            session_id: presentation_session_id,
            thread_id: format!("{}:image-presentation", intent.scope.thread_id),
            turn_id: format!("{}:image-presentation", intent.scope.turn_id),
            system_prompt,
            user_prompt: presentation_user_prompt(intent, presentation_language),
            turn_context: Some(turn_context),
            provider_configuration: Some(provider_configuration),
        },
        &db,
    )
    .await
    .map_err(RuntimeCoreError::Backend)?;
    let generated_provider = generated
        .provider_config
        .as_ref()
        .and_then(|config| config.provider_selector.as_deref())
        .unwrap_or(&selection.provider);
    let generated_model = generated
        .provider_config
        .as_ref()
        .map(|config| config.model_name.as_str())
        .unwrap_or(&selection.model);

    let raw_text_len = generated.text.chars().count();
    let parsed = parse_generated_presentation_with_soul_style(
        &generated.text,
        generated_provider,
        generated_model,
        presentation_language,
        soul_style.as_ref(),
    )
    .map(|mut presentation| {
        presentation.usage = generated.usage.clone();
        presentation
    });
    if parsed.is_some() {
        tracing::info!(
            session_id = %intent.scope.session_id,
            thread_id = %intent.scope.thread_id,
            turn_id = %intent.scope.turn_id,
            workflow_run_id = %workflow_run_id(&intent.scope),
            provider = %selection.provider,
            model = %selection.model,
            raw_text_chars = raw_text_len,
            "[RuntimeBackend] ImageCommandWorkflow presentation generation parsed"
        );
    } else {
        tracing::warn!(
            session_id = %intent.scope.session_id,
            thread_id = %intent.scope.thread_id,
            turn_id = %intent.scope.turn_id,
            workflow_run_id = %workflow_run_id(&intent.scope),
            provider = %selection.provider,
            model = %selection.model,
            raw_text_chars = raw_text_len,
            parse_reason = presentation_parse_failure_reason(&generated.text, presentation_language),
            output_preview = %redacted_model_output_preview(&generated.text),
            "[RuntimeBackend] ImageCommandWorkflow presentation generation produced unusable output"
        );
    }

    Ok(parsed)
}

pub(super) fn normalize_existing_presentation(
    intent: &ImageCommandIntent,
) -> Option<GeneratedImageTaskPresentation> {
    let existing = intent.presentation.as_ref()?;
    let presentation_language =
        detect_presentation_language(intent.raw_text.as_deref().unwrap_or(&intent.prompt));
    let provider = string_field(existing, &["provider", "providerId", "provider_id"])
        .or_else(|| intent.provider_id.clone());
    let model = string_field(existing, &["model", "modelName", "model_name"])
        .or_else(|| intent.model.clone());
    let soul_style = existing
        .as_object()
        .and_then(SoulStyleMetadata::from_payload_object);
    normalize_presentation_value(
        existing,
        provider.as_deref(),
        model.as_deref(),
        presentation_language,
        PRESENTATION_SOURCE_METADATA,
        soul_style.as_ref(),
    )
}

fn image_generation_soul_style(
    config_metadata: Option<&Value>,
    runtime_metadata: Option<&Value>,
) -> Option<SoulStyleMetadata> {
    SoulStyleMetadata::from_config_metadata(config_metadata)
        .or_else(|| SoulStyleMetadata::from_config_metadata(runtime_metadata))
}

fn resolve_presentation_model_selection(
    request: &ExecutionRequest,
) -> Result<RuntimeModelSelection, RuntimeCoreError> {
    for selection in [
        presentation_text_selection_from_profile_model_slot(request),
        request_context::selection_from_session_default(request),
        request_context::selection_from_host_provider_config(request),
        request_context::selection_from_explicit_preferences(request),
    ]
    .into_iter()
    .flatten()
    {
        if !selection_looks_image_generation_only(&selection) {
            return Ok(selection);
        }
    }

    Err(RuntimeCoreError::Backend(
        "presentation_text_model_unavailable: no configured text-capable provider/model selection"
            .to_string(),
    ))
}

fn presentation_text_selection_from_profile_model_slot(
    request: &ExecutionRequest,
) -> Option<RuntimeModelSelection> {
    let reasoning_effort = request_context::reasoning_effort_from_request(request);
    metadata_candidates(request)
        .into_iter()
        .find_map(|metadata| {
            ["fast", "base", "coding", "local"]
                .into_iter()
                .filter_map(|slot| profile_slot_value(metadata, slot))
                .find_map(|slot| {
                    let selection = RuntimeModelSelection {
                        provider: string_field_from_value(
                            slot,
                            &[
                                "provider",
                                "providerId",
                                "provider_id",
                                "providerPreference",
                                "provider_preference",
                                "selectedProvider",
                                "selected_provider",
                            ],
                        )?,
                        model: string_field_from_value(
                            slot,
                            &[
                                "model",
                                "modelName",
                                "model_name",
                                "modelPreference",
                                "model_preference",
                                "selectedModel",
                                "selected_model",
                            ],
                        )?,
                        source: "profile_model_slot",
                        reasoning_effort: reasoning_effort.clone(),
                    };
                    (!selection_looks_image_generation_only(&selection)).then_some(selection)
                })
        })
}

fn metadata_candidates(request: &ExecutionRequest) -> Vec<&Value> {
    request.runtime_metadata().into_iter().collect()
}

fn profile_slot_value<'a>(metadata: &'a Value, slot: &str) -> Option<&'a Value> {
    let container = [
        "/harness/coding_model_slots",
        "/harness/codingModelSlots",
        "/harness/model_slots",
        "/harness/modelSlots",
        "/coding_model_slots",
        "/codingModelSlots",
        "/model_slots",
        "/modelSlots",
        "/coding_profile/model_slots",
        "/codingProfile/modelSlots",
    ]
    .iter()
    .find_map(|pointer| metadata.pointer(pointer))?;

    match container {
        Value::Object(object) => object.get(slot),
        Value::Array(items) => items.iter().find(|item| {
            string_field_from_value(
                item,
                &[
                    "slot",
                    "id",
                    "name",
                    "serviceModelSlot",
                    "service_model_slot",
                ],
            )
            .as_deref()
                == Some(slot)
        }),
        _ => None,
    }
}

fn string_field_from_value(value: &Value, keys: &[&str]) -> Option<String> {
    let object = value.as_object()?;
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn same_provider_model(left: &RuntimeModelSelection, right: &RuntimeModelSelection) -> bool {
    left.provider == right.provider && left.model == right.model
}

fn selection_looks_image_generation_only(selection: &RuntimeModelSelection) -> bool {
    let provider = selection.provider.to_ascii_lowercase();
    let model = selection.model.to_ascii_lowercase();
    provider == "fal"
        || provider.contains("fal-ai")
        || model.contains("agnes-image")
        || model.contains("gpt-image")
        || model.contains("dall-e")
        || model.contains("dalle")
        || model.contains("imagen")
        || model.contains("nano-banana")
        || model.contains("banana")
        || model.contains("image-")
        || model.contains("-image")
        || model.contains("qwen-image")
        || model.contains("glm-image")
        || model.contains("flux")
        || model.contains("seedream")
        || model.contains("kontext")
        || model.contains("recraft")
        || model.contains("ideogram")
        || model.contains("sdxl")
        || model.contains("sd3")
        || model.contains("stable-diffusion")
        || model.contains("text-to-image")
        || model.contains("picture")
        || model.contains("drawing")
        || model.contains("midjourney")
        || model.contains("wan2")
        || model.contains("kolors")
}

pub(super) fn merge_generated_presentation(
    existing: Option<Value>,
    generated: &GeneratedImageTaskPresentation,
) -> Option<Value> {
    let mut merged = existing
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    if let Some(generated_object) = generated.payload.as_object() {
        for (key, value) in generated_object {
            merged.insert(key.clone(), value.clone());
        }
    }
    Some(Value::Object(merged))
}

#[cfg(test)]
fn parse_generated_presentation(
    raw_text: &str,
    provider: &str,
    model: &str,
    expected_language: PresentationLanguage,
) -> Option<GeneratedImageTaskPresentation> {
    parse_generated_presentation_with_soul_style(raw_text, provider, model, expected_language, None)
}

fn parse_generated_presentation_with_soul_style(
    raw_text: &str,
    provider: &str,
    model: &str,
    expected_language: PresentationLanguage,
    soul_style: Option<&SoulStyleMetadata>,
) -> Option<GeneratedImageTaskPresentation> {
    let value = parse_json_object(raw_text)?;
    normalize_presentation_value(
        &value,
        Some(provider),
        Some(model),
        expected_language,
        PRESENTATION_SOURCE,
        soul_style,
    )
}

fn normalize_presentation_value(
    value: &Value,
    provider: Option<&str>,
    model: Option<&str>,
    expected_language: PresentationLanguage,
    source: &str,
    soul_style: Option<&SoulStyleMetadata>,
) -> Option<GeneratedImageTaskPresentation> {
    let assistant_intro = sanitize_user_visible_copy(
        string_field(&value, &["assistant_intro", "assistantIntro", "intro"]).as_deref(),
        MAX_INTRO_CHARS,
        expected_language,
    );
    let planning_summary = sanitize_user_visible_copy(
        string_field(
            &value,
            &[
                "planning_summary",
                "planningSummary",
                "process_summary",
                "processSummary",
            ],
        )
        .as_deref(),
        MAX_PLANNING_SUMMARY_CHARS,
        expected_language,
    );
    let result_captions = value.get("result_captions").and_then(Value::as_object);
    let completion_caption = sanitize_user_visible_copy(
        string_field(
            &value,
            &["completion_caption", "completionCaption", "complete"],
        )
        .or_else(|| {
            result_captions.and_then(|captions| {
                string_field_from_map(captions, &["complete", "completion_caption"])
            })
        })
        .as_deref(),
        MAX_CAPTION_CHARS,
        expected_language,
    );
    if planning_summary.is_none() && assistant_intro.is_none() && completion_caption.is_none() {
        return None;
    }

    let mut payload = Map::new();
    payload.insert("schema".to_string(), json!(PRESENTATION_SCHEMA_VERSION));
    payload.insert("source".to_string(), json!(source));
    if let Some(provider) = provider {
        payload.insert("provider".to_string(), json!(provider));
    }
    if let Some(model) = model {
        payload.insert("model".to_string(), json!(model));
    }
    payload.insert("language".to_string(), json!(expected_language.code()));
    insert_image_generation_soul_payload_fields(
        &mut payload,
        source,
        provider,
        model,
        expected_language.code(),
        soul_style,
    );
    if let Some(planning_summary) = planning_summary.as_ref() {
        payload.insert("planning_summary".to_string(), json!(planning_summary));
        payload.insert("planningSummary".to_string(), json!(planning_summary));
    }
    if let Some(assistant_intro) = assistant_intro.as_ref() {
        payload.insert("assistant_intro".to_string(), json!(assistant_intro));
        payload.insert("assistantIntro".to_string(), json!(assistant_intro));
    }
    if let Some(completion_caption) = completion_caption.as_ref() {
        payload.insert("completion_caption".to_string(), json!(completion_caption));
        payload.insert("completionCaption".to_string(), json!(completion_caption));
        payload.insert(
            "result_captions".to_string(),
            json!({
                "complete": completion_caption,
                "partial": completion_caption,
            }),
        );
    }

    Some(GeneratedImageTaskPresentation {
        planning_summary,
        assistant_intro,
        completion_caption,
        payload: Value::Object(payload),
        usage: None,
    })
}

fn parse_json_object(raw_text: &str) -> Option<Value> {
    let trimmed = raw_text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        if value.is_object() {
            return Some(value);
        }
    }
    let without_fence = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|value| value.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(trimmed);
    if let Ok(value) = serde_json::from_str::<Value>(without_fence) {
        if value.is_object() {
            return Some(value);
        }
    }
    let start = without_fence.find('{')?;
    let end = without_fence.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str::<Value>(&without_fence[start..=end])
        .ok()
        .filter(|value| value.is_object())
}

fn presentation_parse_failure_reason(
    raw_text: &str,
    expected_language: PresentationLanguage,
) -> &'static str {
    if raw_text.trim().is_empty() {
        return "empty_model_output";
    }
    let Some(value) = parse_json_object(raw_text) else {
        return "invalid_json_object";
    };
    let result_captions = value.get("result_captions").and_then(Value::as_object);
    let assistant_intro = string_field(&value, &["assistant_intro", "assistantIntro", "intro"]);
    let planning_summary = string_field(
        &value,
        &[
            "planning_summary",
            "planningSummary",
            "process_summary",
            "processSummary",
        ],
    );
    let completion_caption = string_field(
        &value,
        &["completion_caption", "completionCaption", "complete"],
    )
    .or_else(|| {
        result_captions.and_then(|captions| {
            string_field_from_map(captions, &["complete", "completion_caption"])
        })
    });
    if planning_summary.is_none() && assistant_intro.is_none() && completion_caption.is_none() {
        return "missing_visible_fields";
    }
    let planning_ok = sanitize_user_visible_copy(
        planning_summary.as_deref(),
        MAX_PLANNING_SUMMARY_CHARS,
        expected_language,
    )
    .is_some();
    let intro_ok = sanitize_user_visible_copy(
        assistant_intro.as_deref(),
        MAX_INTRO_CHARS,
        expected_language,
    )
    .is_some();
    let caption_ok = sanitize_user_visible_copy(
        completion_caption.as_deref(),
        MAX_CAPTION_CHARS,
        expected_language,
    )
    .is_some();
    if !planning_ok && !intro_ok && !caption_ok {
        return "visible_copy_rejected";
    }
    "unknown"
}

fn redacted_model_output_preview(raw_text: &str) -> String {
    if raw_text.trim().is_empty() {
        return String::new();
    }
    if contains_forbidden_visible_copy(raw_text) {
        return "[redacted]".to_string();
    }
    raw_text
        .replace('\r', " ")
        .replace('\n', " ")
        .chars()
        .take(180)
        .collect::<String>()
        .trim()
        .to_string()
}

fn sanitize_user_visible_copy(
    value: Option<&str>,
    max_chars: usize,
    expected_language: PresentationLanguage,
) -> Option<String> {
    let normalized = value?
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if normalized.is_empty()
        || contains_forbidden_visible_copy(&normalized)
        || !visible_copy_matches_expected_language(&normalized, expected_language)
    {
        return None;
    }
    if normalized.chars().count() <= max_chars {
        return Some(normalized);
    }
    Some(
        normalized
            .chars()
            .take(max_chars)
            .collect::<String>()
            .trim()
            .to_string(),
    )
}

fn contains_forbidden_visible_copy(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase();
    [
        concat!("r", "ibbi"),
        "workflow",
        "imagecommandworkflow",
        "task_id",
        "task id",
        ".lime",
        "json",
        "jsonl",
        "tool parameter",
        "tool parameters",
        "工具参数",
        "任务 id",
        "任务ID",
        "任务文件",
        "工作流",
    ]
    .iter()
    .any(|term| normalized.contains(&term.to_ascii_lowercase()))
}

fn detect_presentation_language(value: &str) -> PresentationLanguage {
    if value.chars().any(is_hangul) {
        return PresentationLanguage::Korean;
    }
    if value.chars().any(is_japanese_kana) {
        return PresentationLanguage::Japanese;
    }
    if value.chars().any(is_cjk_unified) {
        return PresentationLanguage::ChineseSimplified;
    }
    if value.chars().any(|ch| ch.is_ascii_alphabetic()) {
        return PresentationLanguage::English;
    }
    PresentationLanguage::Unknown
}

fn visible_copy_matches_expected_language(value: &str, expected: PresentationLanguage) -> bool {
    match expected {
        PresentationLanguage::ChineseSimplified => value.chars().any(is_cjk_unified),
        PresentationLanguage::Japanese => value.chars().any(is_japanese_kana),
        PresentationLanguage::Korean => value.chars().any(is_hangul),
        PresentationLanguage::English | PresentationLanguage::Unknown => true,
    }
}

fn is_cjk_unified(ch: char) -> bool {
    matches!(
        ch,
        '\u{4E00}'..='\u{9FFF}'
            | '\u{3400}'..='\u{4DBF}'
            | '\u{20000}'..='\u{2A6DF}'
            | '\u{2A700}'..='\u{2B73F}'
            | '\u{2B740}'..='\u{2B81F}'
            | '\u{2B820}'..='\u{2CEAF}'
    )
}

fn is_japanese_kana(ch: char) -> bool {
    matches!(ch, '\u{3040}'..='\u{30FF}' | '\u{31F0}'..='\u{31FF}')
}

fn is_hangul(ch: char) -> bool {
    matches!(ch, '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}' | '\u{3130}'..='\u{318F}')
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn string_field_from_map(map: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| map.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn presentation_session_id(intent: &ImageCommandIntent) -> String {
    format!(
        "{}:image-presentation:{}",
        intent.scope.session_id, intent.scope.turn_id
    )
}

fn presentation_user_prompt(
    intent: &ImageCommandIntent,
    presentation_language: PresentationLanguage,
) -> String {
    json!({
        "task": "Generate user-visible copy for one image generation turn.",
        "user_request": intent.raw_text.as_deref().unwrap_or(&intent.prompt),
        "image_prompt": intent.prompt,
        "mode": intent.mode.as_deref().unwrap_or("generate"),
        "model_label_hint": intent.model,
        "output_language": presentation_language.code(),
        "requested_target": intent.requested_target,
        "rules": [
            "Return only valid JSON.",
            "Write in the same language as the user request.",
            presentation_language.rule(),
            "planning_summary should be one short, user-visible process summary about composition, mood, and constraints. It must not expose hidden chain-of-thought.",
            "assistant_intro should follow the active Interaction Soul when present, stay brief, and naturally acknowledge the request before generation.",
            "completion_caption should follow the active Interaction Soul when present, describe the result as if the image has completed, invite lightweight iteration, and avoid sounding templated.",
            "Do not mention workflow, task id, files, JSON, tools, internal paths, or runtime details.",
            "Do not mention branded assistant names."
        ],
        "schema": {
            "planning_summary": "string",
            "assistant_intro": "string",
            "completion_caption": "string"
        }
    })
    .to_string()
}

fn presentation_system_prompt() -> String {
    String::from(
        "You write concise, natural user-visible copy for Lime image generation turns.\n\
Return only JSON that matches the requested schema.\n\
The copy must feel contextual and human, not like a reusable template.\n\
If an Interaction Soul section is present, apply it to assistant_intro and completion_caption while keeping planning_summary factual and bounded.\n\
planning_summary is a brief visible process summary, not hidden chain-of-thought. Summarize what visual direction you will use without revealing internal workflow.\n\
Detect the user's request language and keep both fields in that language. For Chinese requests, use Simplified Chinese and never use English openers such as Sure, Done, or is ready.\n\
Never reveal internal workflow, task ids, tool names, files, JSON/JSONL, audit details, or runtime implementation.\n\
Never use branded assistant names.\n\
Do not include markdown fences.",
    )
}

#[cfg(test)]
mod tests;
