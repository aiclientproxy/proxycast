use agent_runtime::reply_input::{RuntimeReplyInput, RuntimeReplyInputPart};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub(super) const UNSUPPORTED_HISTORY_IMAGE_PLACEHOLDER: &str =
    "image content omitted because you do not support image input";

pub(super) struct StructuredInputContext {
    pub(super) messages: Vec<model_provider::current_client::CurrentProviderMessage>,
    pub(super) warnings: Vec<StructuredInputWarning>,
}

pub(super) struct StructuredInputWarning {
    pub(super) code: &'static str,
    pub(super) message: String,
}

pub(super) fn user_message(
    input: &RuntimeReplyInput,
) -> Option<model_provider::current_client::CurrentProviderMessage> {
    use model_provider::current_client::{CurrentProviderContent, CurrentProviderMessage};

    let content = input
        .parts
        .iter()
        .filter_map(|part| match part {
            RuntimeReplyInputPart::Text { text, .. } => {
                (!text.is_empty()).then_some(CurrentProviderContent::Text(text.clone()))
            }
            RuntimeReplyInputPart::Image(image) => Some(CurrentProviderContent::Image {
                uri: image.uri.clone(),
                media_type: image.media_type.clone(),
                provider_data: image.provider_data.clone(),
                detail: image.detail,
            }),
            RuntimeReplyInputPart::Skill { .. } | RuntimeReplyInputPart::Mention { .. } => None,
        })
        .collect::<Vec<_>>();
    (!content.is_empty()).then(|| CurrentProviderMessage::user(content))
}

pub(super) fn structured_input_context(
    input: &RuntimeReplyInput,
    skill_snapshot: Option<&lime_skills::AgentSkillSnapshot>,
) -> StructuredInputContext {
    let mut messages = Vec::new();
    let mut warnings = Vec::new();
    let mut seen_paths = HashSet::new();
    let mut blocked_fallback_names = HashSet::new();
    for part in &input.parts {
        match part {
            RuntimeReplyInputPart::Skill { name, path } => {
                blocked_fallback_names.insert(name.to_ascii_lowercase());
                let Some(skill_snapshot) = skill_snapshot else {
                    push_skill_not_available_warning(name, path, &mut warnings);
                    continue;
                };
                let Some(selection) = lime_skills::select_structured_agent_skill(
                    name,
                    Path::new(path),
                    skill_snapshot,
                ) else {
                    push_skill_not_available_warning(name, path, &mut warnings);
                    continue;
                };
                push_skill_context(
                    name,
                    selection,
                    &mut seen_paths,
                    &mut messages,
                    &mut warnings,
                );
            }
            RuntimeReplyInputPart::Mention { name, path } => {
                if is_skill_mention_path(path) {
                    blocked_fallback_names.insert(name.to_ascii_lowercase());
                    let Some(skill_snapshot) = skill_snapshot else {
                        push_skill_not_available_warning(name, path, &mut warnings);
                        continue;
                    };
                    let Some(selection) =
                        lime_skills::select_mentioned_agent_skill(name, path, skill_snapshot)
                    else {
                        push_skill_not_available_warning(name, path, &mut warnings);
                        continue;
                    };
                    push_skill_context(
                        name,
                        selection,
                        &mut seen_paths,
                        &mut messages,
                        &mut warnings,
                    );
                } else if !is_control_mention_path(path) {
                    tracing::warn!(
                        mention_name = %name,
                        mention_path = %path,
                        "跳过尚未接通 current resolver 的结构化 Mention"
                    );
                    warnings.push(StructuredInputWarning {
                        code: "mention_not_available",
                        message: format!("当前无法解析结构化 Mention `{name}`。"),
                    });
                }
            }
            RuntimeReplyInputPart::Text { .. } | RuntimeReplyInputPart::Image(_) => {}
        }
    }
    if let Some(skill_snapshot) = skill_snapshot {
        for selection in
            lime_skills::select_explicit_agent_skills(&input.concat_text(), skill_snapshot)
        {
            if blocked_fallback_names.contains(&selection.locator.name.to_ascii_lowercase()) {
                continue;
            }
            let name = selection.locator.name.clone();
            push_skill_context(
                &name,
                selection,
                &mut seen_paths,
                &mut messages,
                &mut warnings,
            );
        }
    }
    StructuredInputContext { messages, warnings }
}

pub(super) fn skill_snapshot_from_turn_context(
    turn_context: Option<&agent_protocol::turn_context::TurnContextOverride>,
) -> Option<lime_skills::AgentSkillSnapshot> {
    let value = turn_context?
        .metadata
        .get(lime_skills::SKILL_SNAPSHOT_TURN_METADATA_KEY)?;
    match serde_json::from_value(value.clone()) {
        Ok(snapshot) => Some(snapshot),
        Err(error) => {
            tracing::warn!(error = %error, "忽略无法解析的 current Skill snapshot");
            None
        }
    }
}

pub(super) fn prepare_image_inputs_for_model(
    input: &RuntimeReplyInput,
    initial_messages: &mut [model_provider::current_client::CurrentProviderMessage],
    supports_image_input: bool,
) -> Result<(), crate::request_tool_policy::ReplyAttemptError> {
    use model_provider::current_client::CurrentProviderContent;

    if supports_image_input {
        return Ok(());
    }
    if input.has_images() {
        return Err(crate::request_tool_policy::ReplyAttemptError::new(
            "当前选中模型的 input_modality_policy 不支持图片输入，已拒绝把 image 内容发送到 provider；请切换支持 image 的模型或移除图片。",
            false,
        ));
    }

    for content in initial_messages
        .iter_mut()
        .flat_map(|message| &mut message.content)
    {
        if matches!(content, CurrentProviderContent::Image { .. }) {
            *content =
                CurrentProviderContent::Text(UNSUPPORTED_HISTORY_IMAGE_PLACEHOLDER.to_string());
        }
    }
    Ok(())
}

fn push_skill_context(
    requested_name: &str,
    selection: lime_skills::AgentSkillSelection,
    seen_paths: &mut HashSet<PathBuf>,
    messages: &mut Vec<model_provider::current_client::CurrentProviderMessage>,
    warnings: &mut Vec<StructuredInputWarning>,
) {
    use model_provider::current_client::{CurrentProviderContent, CurrentProviderMessage};

    if let Some(resource_uri) = selection
        .locator
        .skill_file_path
        .to_str()
        .filter(|path| path.starts_with("skill://"))
    {
        if !seen_paths.insert(selection.locator.skill_file_path.clone()) {
            return;
        }
        messages.push(CurrentProviderMessage::user(vec![
            CurrentProviderContent::Text(format!(
                "<skill_resource>\n<name>{requested_name}</name>\n<server>{}</server>\n<uri>{resource_uri}</uri>\n在使用该 Skill 前，必须调用 read_mcp_resource 读取上述 server 与 URI；这是 Orchestrator resource，不是本地文件。\n</skill_resource>",
                lime_skills::APPS_MCP_SERVER_NAME
            )),
        ]));
        return;
    }

    let instructions = match lime_skills::read_agent_skill_instructions(&selection.locator) {
        Ok(instructions) => instructions,
        Err(error) => {
            tracing::warn!(skill_name = %requested_name, error = %error, "跳过无法加载的结构化 Skill");
            warnings.push(StructuredInputWarning {
                code: "skill_load_failed",
                message: format!("无法加载所选 Skill `{requested_name}`：{error}"),
            });
            return;
        }
    };
    if !seen_paths.insert(instructions.path.clone()) {
        return;
    }
    messages.push(CurrentProviderMessage::user(vec![
        CurrentProviderContent::Text(format!(
            "<skill>\n<name>{}</name>\n<path>{}</path>\n{}\n</skill>",
            instructions.name,
            instructions.path.display(),
            instructions.contents
        )),
    ]));
}

fn push_skill_not_available_warning(
    name: &str,
    path: &str,
    warnings: &mut Vec<StructuredInputWarning>,
) {
    tracing::warn!(skill_name = %name, skill_path = %path, "跳过未出现在 current snapshot 的结构化 Skill");
    warnings.push(StructuredInputWarning {
        code: "skill_not_available",
        message: format!("无法加载所选 Skill `{name}`：它不在当前可用 Skills 中。"),
    });
}

fn is_skill_mention_path(path: &str) -> bool {
    path.starts_with("skill://")
        || path
            .rsplit(['/', '\\'])
            .next()
            .is_some_and(|name| name.eq_ignore_ascii_case("SKILL.md"))
}

fn is_control_mention_path(path: &str) -> bool {
    ["app://", "plugin://", "mcp://"]
        .iter()
        .any(|prefix| path.starts_with(prefix))
}
