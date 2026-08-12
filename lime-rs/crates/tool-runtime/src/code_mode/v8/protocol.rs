#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ToolName {
    pub name: String,
    pub namespace: Option<String>,
}

#[derive(Clone, Debug)]
pub(super) struct ToolDefinition {
    pub name: String,
    pub tool_name: ToolName,
    pub description: String,
}

#[derive(Clone, Debug)]
pub(super) struct EnabledToolMetadata {
    pub tool_name: ToolName,
    pub global_name: String,
    pub description: String,
}

pub(super) fn enabled_tool_metadata(definition: &ToolDefinition) -> EnabledToolMetadata {
    EnabledToolMetadata {
        tool_name: definition.tool_name.clone(),
        global_name: definition.name.clone(),
        description: definition.description.clone(),
    }
}

pub(super) struct ExecuteRequest {
    pub tool_call_id: String,
    pub enabled_tools: Vec<ToolDefinition>,
    pub source: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ImageDetail {
    Auto,
    Low,
    High,
    Original,
}

pub(super) const DEFAULT_IMAGE_DETAIL: ImageDetail = ImageDetail::Auto;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum FunctionCallOutputContentItem {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        detail: Option<ImageDetail>,
    },
    InputAudio {
        audio_url: String,
    },
}
