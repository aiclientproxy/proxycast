use super::boundary::default_prompt_boundary;
use super::style_profile::resolve_style_profile;
use crate::runtime::context_packet::ContextPacket;
use lime_core::config::MemorySoulConfig;
use serde_json::{json, Map, Value};

const SOUL_CONTEXT_VERSION: &str = "memory_soul_prompt_context.v2";
const SOUL_TEXT_MAX_CHARS: usize = 600;
const SOUL_SHORT_TEXT_MAX_CHARS: usize = 160;
const SOUL_LIST_ITEM_MAX_CHARS: usize = 120;
const SOUL_LIST_MAX_ITEMS: usize = 8;
const SOUL_PACKET_MAX_TOKENS: usize = 1400;

pub(crate) fn memory_soul_prompt_context_from_config(
    soul: Option<&MemorySoulConfig>,
) -> Option<Value> {
    let soul = soul?;
    if !soul.enabled {
        return None;
    }

    let name = normalize_text(soul.name.as_deref(), 80);
    let summary = normalize_text(soul.summary.as_deref(), SOUL_TEXT_MAX_CHARS);
    let tone = normalize_list(&soul.tone);
    let communication_style = normalize_list(&soul.communication_style);
    let explanation_depth =
        normalize_text(soul.explanation_depth.as_deref(), SOUL_SHORT_TEXT_MAX_CHARS);
    let challenge_style =
        normalize_text(soul.challenge_style.as_deref(), SOUL_SHORT_TEXT_MAX_CHARS);
    let avoid = normalize_list(&soul.avoid);
    let style_profile = resolve_style_profile(soul.style_profile_id.as_deref());
    let boundary = default_prompt_boundary();

    if name.is_none()
        && summary.is_none()
        && tone.is_empty()
        && communication_style.is_empty()
        && explanation_depth.is_none()
        && challenge_style.is_none()
        && avoid.is_empty()
        && soul.style_profile_id.is_none()
    {
        return None;
    }

    let mut context = Map::new();
    context.insert("schema".to_string(), json!(SOUL_CONTEXT_VERSION));
    context.insert("source".to_string(), json!("memory.soul"));
    context.insert("scope".to_string(), json!("interaction_only"));
    context.insert(
        "formalArtifactVoiceSource".to_string(),
        json!(boundary.formal_artifact_voice_source),
    );
    context.insert("styleProfile".to_string(), style_profile.as_context_value());
    context.insert("styleBoundary".to_string(), boundary.as_context_value());
    insert_optional_string(&mut context, "name", name);
    insert_optional_string(&mut context, "summary", summary);
    insert_non_empty_list(&mut context, "tone", tone);
    insert_non_empty_list(&mut context, "communicationStyle", communication_style);
    insert_optional_string(&mut context, "explanationDepth", explanation_depth);
    insert_optional_string(&mut context, "challengeStyle", challenge_style);
    insert_non_empty_list(&mut context, "avoid", avoid);

    Some(Value::Object(context))
}

pub(crate) fn soul_packet_from_metadata(
    config_metadata: Option<&Value>,
    runtime_metadata: Option<&Value>,
) -> Option<ContextPacket> {
    let config_metadata = config_metadata?;
    let value = config_metadata
        .pointer("/memory/soul")
        .or_else(|| config_metadata.get(super::MEMORY_SOUL_PROMPT_CONTEXT_KEY))?;
    let source = value
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or("memory.soul");
    if source != "memory.soul" {
        return None;
    }
    let scope = value
        .get("scope")
        .and_then(Value::as_str)
        .filter(|scope| !scope.trim().is_empty())
        .unwrap_or("interaction_only");
    let name = value.get("name").and_then(Value::as_str);
    let summary = value.get("summary").and_then(Value::as_str);
    let schema = value.get("schema").and_then(Value::as_str);
    let style_profile = value.get("styleProfile");
    let style_boundary = value.get("styleBoundary");
    let tone = string_array(value.get("tone"));
    let communication_style = string_array(value.get("communicationStyle"));
    let explanation_depth = value.get("explanationDepth").and_then(Value::as_str);
    let challenge_style = value.get("challengeStyle").and_then(Value::as_str);
    let avoid = string_array(value.get("avoid"));

    if [name, summary, explanation_depth, challenge_style]
        .into_iter()
        .flatten()
        .all(|item| item.trim().is_empty())
        && style_profile.is_none()
        && tone.is_empty()
        && communication_style.is_empty()
        && avoid.is_empty()
    {
        return None;
    }

    let mut lines = Vec::new();
    if let Some(schema) = non_empty_str(schema) {
        lines.push(format!("- Schema: {schema}"));
    }
    if let Some(name) = non_empty_str(name) {
        lines.push(format!("- Name: {name}"));
    }
    if let Some(summary) = non_empty_str(summary) {
        lines.push(format!("- Summary: {summary}"));
    }
    lines.push(
        "- Turn relevance: Answer simple greetings directly in one or two short sentences. Do not infer an unstated task, continue unrelated work, or summarize a project from style or persona context."
            .to_string(),
    );
    append_style_boundary_lines(style_boundary, &mut lines);
    append_persona_context_lines(runtime_metadata, &mut lines);
    append_style_profile_lines(style_profile, &mut lines);
    if !tone.is_empty() {
        lines.push(format!("- Tone: {}", tone.join(", ")));
    }
    if !communication_style.is_empty() {
        lines.push("- Communication style:".to_string());
        lines.extend(
            communication_style
                .into_iter()
                .map(|item| format!("  - {item}")),
        );
    }
    if let Some(explanation_depth) = non_empty_str(explanation_depth) {
        lines.push(format!("- Explanation depth: {explanation_depth}"));
    }
    if let Some(challenge_style) = non_empty_str(challenge_style) {
        lines.push(format!("- Challenge style: {challenge_style}"));
    }
    if !avoid.is_empty() {
        lines.push("- Avoid:".to_string());
        lines.extend(avoid.into_iter().map(|item| format!("  - {item}")));
    }

    let mut metadata = Map::new();
    metadata.insert("scope".to_string(), json!(scope));
    Some(ContextPacket::interaction_soul(
        lines.join("\n"),
        SOUL_PACKET_MAX_TOKENS,
        metadata,
    ))
}

fn append_persona_context_lines(runtime_metadata: Option<&Value>, lines: &mut Vec<String>) {
    let Some(persona_context) = persona_context_value(runtime_metadata) else {
        return;
    };

    let persona_packs = persona_pack_lines(persona_context);
    if persona_packs.is_empty() {
        return;
    }

    lines.push("- Persona knowledge packs (context only):".to_string());
    lines.extend(persona_packs.into_iter().map(|item| format!("  - {item}")));
    append_named_list(
        lines,
        "Persona context boundaries",
        string_array(persona_context.get("boundaries")),
    );

    let contract = persona_context
        .get("style_profile_contract")
        .or_else(|| persona_context.get("styleProfileContract"));
    if let Some(contract) = contract {
        let inherits_global_soul = contract
            .get("inherits_global_soul")
            .or_else(|| contract.get("inheritsGlobalSoul"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let writes_back_to_global_soul = contract
            .get("writes_back_to_global_soul")
            .or_else(|| contract.get("writesBackToGlobalSoul"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let formal_artifact_voice_source = contract
            .get("formal_artifact_voice_source")
            .or_else(|| contract.get("formalArtifactVoiceSource"))
            .and_then(Value::as_str)
            .unwrap_or("generation_brief_only");
        lines.push(format!(
            "- Persona context contract: inherits_global_soul={inherits_global_soul}; writes_back_to_global_soul={writes_back_to_global_soul}; formal_artifact_voice_source={formal_artifact_voice_source}."
        ));
    }
}

fn persona_context_value(metadata: Option<&Value>) -> Option<&Value> {
    let metadata = metadata?;
    [
        "/persona_context",
        "/personaContext",
        "/harness/persona_context",
        "/harness/personaContext",
        "/knowledge_pack/persona_context",
        "/knowledgePack/personaContext",
    ]
    .into_iter()
    .find_map(|pointer| metadata.pointer(pointer))
}

fn persona_pack_lines(persona_context: &Value) -> Vec<String> {
    persona_context
        .get("packs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|pack| {
            let name = non_empty_str(pack.get("name").and_then(Value::as_str))?;
            let activation =
                non_empty_str(pack.get("activation").and_then(Value::as_str)).unwrap_or("explicit");
            let role =
                non_empty_str(pack.get("role").and_then(Value::as_str)).unwrap_or("companion");
            Some(format!("{name} (activation: {activation}, role: {role})"))
        })
        .collect()
}

fn append_style_boundary_lines(style_boundary: Option<&Value>, lines: &mut Vec<String>) {
    let Some(style_boundary) = style_boundary else {
        return;
    };

    if let Some(source) = non_empty_str(
        style_boundary
            .get("formalArtifactVoiceSource")
            .and_then(Value::as_str),
    ) {
        lines.push(format!("- Formal artifact voice source: {source}"));
    }
    append_named_list(
        lines,
        "Serious mode triggers",
        string_array(style_boundary.get("seriousModeTriggers")),
    );
    append_named_list(
        lines,
        "Style fidelity rules",
        string_array(style_boundary.get("fidelityRules")),
    );
}

fn append_style_profile_lines(style_profile: Option<&Value>, lines: &mut Vec<String>) {
    let Some(style_profile) = style_profile else {
        return;
    };

    if let Some(id) = non_empty_str(style_profile.get("id").and_then(Value::as_str)) {
        lines.push(format!("- Style profile: {id}"));
    }
    if let Some(pack_id) = non_empty_str(style_profile.get("packId").and_then(Value::as_str)) {
        lines.push(format!("- Style pack: {pack_id}"));
    }
    if let Some(tone) = non_empty_str(style_profile.get("tone").and_then(Value::as_str)) {
        lines.push(format!("- Style tone: {tone}"));
    }
    append_named_list(
        lines,
        "Response contract",
        string_array(style_profile.get("responseContract")),
    );
    append_named_list(
        lines,
        "Voice primitives",
        string_array(style_profile.get("voicePrimitives")),
    );
    append_named_list(
        lines,
        "Surface contracts",
        string_array(style_profile.get("surfaceContracts")),
    );
    append_named_list(
        lines,
        "Allowed style moves",
        string_array(style_profile.get("allowedMoves")),
    );
    append_named_list(
        lines,
        "Forbidden style moves",
        string_array(style_profile.get("forbiddenMoves")),
    );
    append_named_list(
        lines,
        "Anti-repetition rules",
        string_array(style_profile.get("antiRepetitionRules")),
    );
    if style_profile.get("fewShotAnchors").is_some() {
        lines.push(
            "- Few-shot anchors: examples are design-time fixtures and are intentionally omitted from the runtime prompt; never reuse a fixed opener."
                .to_string(),
        );
    }
    if let Some(fallback) = non_empty_str(
        style_profile
            .get("seriousModeFallback")
            .and_then(Value::as_str),
    ) {
        lines.push(format!(
            "- Serious/high-risk fallback: {fallback}; use it for permission, deletion, production API, medical, legal, financial, or safety-sensitive turns."
        ));
    }
    if let Some(risk_fallback) = style_profile.get("riskFallback") {
        let fallback = non_empty_str(risk_fallback.get("profileId").and_then(Value::as_str));
        let triggers = string_array(risk_fallback.get("triggers"));
        if let Some(fallback) = fallback {
            lines.push(format!("- Risk fallback profile: {fallback}"));
        }
        append_named_list(lines, "Risk fallback triggers", triggers);
    }
}

fn append_named_list(lines: &mut Vec<String>, label: &str, items: Vec<String>) {
    if items.is_empty() {
        return;
    }
    lines.push(format!("- {label}:"));
    lines.extend(items.into_iter().map(|item| format!("  - {item}")));
}

fn normalize_text(value: Option<&str>, max_chars: usize) -> Option<String> {
    let normalized = value?.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        None
    } else {
        Some(normalized.chars().take(max_chars).collect())
    }
}

fn normalize_list(items: &[String]) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut result = Vec::new();
    for item in items {
        let Some(normalized) = normalize_text(Some(item), SOUL_LIST_ITEM_MAX_CHARS) else {
            continue;
        };
        if seen.insert(normalized.clone()) {
            result.push(normalized);
        }
        if result.len() >= SOUL_LIST_MAX_ITEMS {
            break;
        }
    }
    result
}

fn insert_optional_string(context: &mut Map<String, Value>, key: &str, value: Option<String>) {
    if let Some(value) = value {
        context.insert(key.to_string(), json!(value));
    }
}

fn insert_non_empty_list(context: &mut Map<String, Value>, key: &str, values: Vec<String>) {
    if !values.is_empty() {
        context.insert(key.to_string(), json!(values));
    }
}

fn string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .filter_map(|item| normalize_text(Some(item), SOUL_LIST_ITEM_MAX_CHARS))
                .collect()
        })
        .unwrap_or_default()
}

fn non_empty_str(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::context_packet::assemble_context_packets;

    #[test]
    fn runtime_soul_prompt_excludes_few_shot_example_text() {
        let soul = MemorySoulConfig {
            enabled: true,
            style_profile_id: Some("cheeky_sassy_executor".to_string()),
            ..MemorySoulConfig::default()
        };
        let context = memory_soul_prompt_context_from_config(Some(&soul)).expect("soul context");
        let metadata = json!({
            "memory": {
                "soul": context
            }
        });
        let packet = soul_packet_from_metadata(Some(&metadata), None).expect("soul packet");
        let prompt = assemble_context_packets(vec![packet])
            .rendered
            .expect("rendered soul prompt");

        assert!(prompt.contains("Answer simple greetings directly"));
        assert!(prompt.contains("Treat style profile ids, style pack ids, schemas"));
        assert!(prompt.contains("do not repeat or imitate prior assistant wording"));
        assert!(prompt.contains("必须静默应用"));
        assert!(prompt.contains("examples are design-time fixtures"));
        assert!(!prompt.contains("资料回来了"));
        assert!(!prompt.contains("我先把证据抓回来"));
    }
}
