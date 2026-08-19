use serde::{Deserialize, Serialize};

pub const TOOL_SEARCH_TOOL_NAME: &str = "tool_search";
pub const SKILL_SEARCH_TOOL_NAME: &str = "skill_search";
pub const UPDATE_PLAN_TOOL_NAME: &str = "update_plan";
pub const LIST_MCP_RESOURCES_TOOL_NAME: &str = "list_mcp_resources";
pub const READ_MCP_RESOURCE_TOOL_NAME: &str = "read_mcp_resource";
pub const SOCIAL_IMAGE_TOOL_NAME: &str = "social_generate_cover_image";
pub const LIME_CREATE_VIDEO_TASK_TOOL_NAME: &str = "lime_create_video_generation_task";
pub const LIME_CREATE_AUDIO_TASK_TOOL_NAME: &str = "lime_create_audio_generation_task";
pub const LIME_CREATE_TRANSCRIPTION_TASK_TOOL_NAME: &str = "lime_create_transcription_task";
pub const LIME_CREATE_BROADCAST_TASK_TOOL_NAME: &str = "lime_create_broadcast_generation_task";
pub const LIME_CREATE_COVER_TASK_TOOL_NAME: &str = "lime_create_cover_generation_task";
pub const LIME_CREATE_RESOURCE_SEARCH_TASK_TOOL_NAME: &str =
    "lime_create_modal_resource_search_task";
pub const LIME_SEARCH_WEB_IMAGES_TOOL_NAME: &str = "lime_search_web_images";
pub const LIME_CREATE_IMAGE_TASK_TOOL_NAME: &str = "lime_create_image_generation_task";
pub const LIME_CREATE_URL_PARSE_TASK_TOOL_NAME: &str = "lime_create_url_parse_task";
pub const LIME_CREATE_TYPESETTING_TASK_TOOL_NAME: &str = "lime_create_typesetting_task";
pub const LIME_RUN_SERVICE_SKILL_TOOL_NAME: &str = "lime_run_service_skill";
pub const VIEW_IMAGE_TOOL_NAME: &str = "view_image";
pub const APPLY_PATCH_TOOL_NAME: &str = "apply_patch";
pub const MEMORY_LIST_TOOL_NAME: &str = "memory_list";
pub const MEMORY_READ_TOOL_NAME: &str = "memory_read";
pub const MEMORY_SEARCH_TOOL_NAME: &str = "memory_search";
pub const MEMORY_ADD_NOTE_TOOL_NAME: &str = "memory_add_note";
pub const SLEEP_TOOL_NAME: &str = "sleep";
pub const CURRENT_TIME_TOOL_NAME: &str = "curr_time";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSurfaceProfile {
    Core,
    #[serde(rename = "workbench")]
    Workbench,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCapability {
    Planning,
    Delegation,
    WebSearch,
    SkillExecution,
    SessionControl,
    ContentCreation,
    WorkspaceIo,
    Memory,
    Execution,
    Vision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolLifecycle {
    Current,
    Compat,
    Deprecated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSourceKind {
    RuntimeBuiltin,
    LimeInjected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolPermissionPlane {
    SessionAllowlist,
    ParameterRestricted,
    CallerFiltered,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ToolCatalogEntry {
    pub name: &'static str,
    pub profiles: &'static [ToolSurfaceProfile],
    pub capabilities: &'static [ToolCapability],
    pub lifecycle: ToolLifecycle,
    pub source: ToolSourceKind,
    pub permission_plane: ToolPermissionPlane,
    pub workspace_default_allow: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkspaceToolSurface {
    pub workbench: bool,
}

impl WorkspaceToolSurface {
    pub const fn core() -> Self {
        Self { workbench: false }
    }

    pub const fn workbench() -> Self {
        Self { workbench: true }
    }

    pub const fn includes_profile(self, profile: ToolSurfaceProfile) -> bool {
        match profile {
            ToolSurfaceProfile::Core => true,
            ToolSurfaceProfile::Workbench => self.workbench,
        }
    }
}

const CORE_PROFILES: &[ToolSurfaceProfile] = &[ToolSurfaceProfile::Core];
const WORKBENCH_PROFILES: &[ToolSurfaceProfile] = &[ToolSurfaceProfile::Workbench];

const PLAN_CAP: &[ToolCapability] = &[ToolCapability::Planning];
const SEARCH_CAP: &[ToolCapability] = &[ToolCapability::WebSearch];
const SKILL_CAP: &[ToolCapability] = &[ToolCapability::SkillExecution];
const CONTENT_CAP: &[ToolCapability] = &[ToolCapability::ContentCreation];
const SESSION_CAP: &[ToolCapability] = &[ToolCapability::SessionControl];
const WORKSPACE_IO_CAP: &[ToolCapability] = &[ToolCapability::WorkspaceIo];
const MEMORY_CAP: &[ToolCapability] = &[ToolCapability::Memory];
const EXECUTION_CAP: &[ToolCapability] = &[ToolCapability::Execution];

static NATIVE_TOOL_CATALOG: &[ToolCatalogEntry] = &[
    ToolCatalogEntry {
        name: "Read",
        profiles: CORE_PROFILES,
        capabilities: WORKSPACE_IO_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::RuntimeBuiltin,
        permission_plane: ToolPermissionPlane::ParameterRestricted,
        workspace_default_allow: false,
    },
    ToolCatalogEntry {
        name: APPLY_PATCH_TOOL_NAME,
        profiles: CORE_PROFILES,
        capabilities: WORKSPACE_IO_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::ParameterRestricted,
        workspace_default_allow: false,
    },
    ToolCatalogEntry {
        name: "Glob",
        profiles: CORE_PROFILES,
        capabilities: WORKSPACE_IO_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::RuntimeBuiltin,
        permission_plane: ToolPermissionPlane::ParameterRestricted,
        workspace_default_allow: false,
    },
    ToolCatalogEntry {
        name: "Grep",
        profiles: CORE_PROFILES,
        capabilities: WORKSPACE_IO_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::RuntimeBuiltin,
        permission_plane: ToolPermissionPlane::ParameterRestricted,
        workspace_default_allow: false,
    },
    ToolCatalogEntry {
        name: "exec_command",
        profiles: CORE_PROFILES,
        capabilities: EXECUTION_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::RuntimeBuiltin,
        permission_plane: ToolPermissionPlane::ParameterRestricted,
        workspace_default_allow: false,
    },
    ToolCatalogEntry {
        name: "Skill",
        profiles: CORE_PROFILES,
        capabilities: SKILL_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::RuntimeBuiltin,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: UPDATE_PLAN_TOOL_NAME,
        profiles: CORE_PROFILES,
        capabilities: PLAN_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::RuntimeBuiltin,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: VIEW_IMAGE_TOOL_NAME,
        profiles: CORE_PROFILES,
        capabilities: WORKSPACE_IO_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::RuntimeBuiltin,
        permission_plane: ToolPermissionPlane::ParameterRestricted,
        workspace_default_allow: false,
    },
    ToolCatalogEntry {
        name: "WebFetch",
        profiles: CORE_PROFILES,
        capabilities: SEARCH_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::RuntimeBuiltin,
        permission_plane: ToolPermissionPlane::ParameterRestricted,
        workspace_default_allow: false,
    },
    ToolCatalogEntry {
        name: "WebSearch",
        profiles: CORE_PROFILES,
        capabilities: SEARCH_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::RuntimeBuiltin,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: "request_user_input",
        profiles: CORE_PROFILES,
        capabilities: PLAN_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::RuntimeBuiltin,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: SLEEP_TOOL_NAME,
        profiles: CORE_PROFILES,
        capabilities: PLAN_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: CURRENT_TIME_TOOL_NAME,
        profiles: CORE_PROFILES,
        capabilities: PLAN_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: "StructuredOutput",
        profiles: CORE_PROFILES,
        capabilities: SESSION_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::RuntimeBuiltin,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: false,
    },
    ToolCatalogEntry {
        name: "write_stdin",
        profiles: CORE_PROFILES,
        capabilities: EXECUTION_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::RuntimeBuiltin,
        permission_plane: ToolPermissionPlane::ParameterRestricted,
        workspace_default_allow: false,
    },
    ToolCatalogEntry {
        name: TOOL_SEARCH_TOOL_NAME,
        profiles: CORE_PROFILES,
        capabilities: SEARCH_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: SKILL_SEARCH_TOOL_NAME,
        profiles: CORE_PROFILES,
        capabilities: SKILL_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: LIST_MCP_RESOURCES_TOOL_NAME,
        profiles: CORE_PROFILES,
        capabilities: SEARCH_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: READ_MCP_RESOURCE_TOOL_NAME,
        profiles: CORE_PROFILES,
        capabilities: SEARCH_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: MEMORY_LIST_TOOL_NAME,
        profiles: CORE_PROFILES,
        capabilities: MEMORY_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: MEMORY_READ_TOOL_NAME,
        profiles: CORE_PROFILES,
        capabilities: MEMORY_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: MEMORY_SEARCH_TOOL_NAME,
        profiles: CORE_PROFILES,
        capabilities: MEMORY_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: MEMORY_ADD_NOTE_TOOL_NAME,
        profiles: CORE_PROFILES,
        capabilities: MEMORY_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: SOCIAL_IMAGE_TOOL_NAME,
        profiles: WORKBENCH_PROFILES,
        capabilities: CONTENT_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: LIME_CREATE_VIDEO_TASK_TOOL_NAME,
        profiles: WORKBENCH_PROFILES,
        capabilities: CONTENT_CAP,
        lifecycle: ToolLifecycle::Deprecated,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: false,
    },
    ToolCatalogEntry {
        name: LIME_CREATE_AUDIO_TASK_TOOL_NAME,
        profiles: WORKBENCH_PROFILES,
        capabilities: CONTENT_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: LIME_CREATE_TRANSCRIPTION_TASK_TOOL_NAME,
        profiles: WORKBENCH_PROFILES,
        capabilities: CONTENT_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: LIME_CREATE_BROADCAST_TASK_TOOL_NAME,
        profiles: WORKBENCH_PROFILES,
        capabilities: CONTENT_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: LIME_CREATE_COVER_TASK_TOOL_NAME,
        profiles: WORKBENCH_PROFILES,
        capabilities: CONTENT_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: LIME_CREATE_RESOURCE_SEARCH_TASK_TOOL_NAME,
        profiles: WORKBENCH_PROFILES,
        capabilities: CONTENT_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: LIME_SEARCH_WEB_IMAGES_TOOL_NAME,
        profiles: WORKBENCH_PROFILES,
        capabilities: SEARCH_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: LIME_CREATE_IMAGE_TASK_TOOL_NAME,
        profiles: WORKBENCH_PROFILES,
        capabilities: CONTENT_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: LIME_CREATE_URL_PARSE_TASK_TOOL_NAME,
        profiles: WORKBENCH_PROFILES,
        capabilities: CONTENT_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: LIME_CREATE_TYPESETTING_TASK_TOOL_NAME,
        profiles: WORKBENCH_PROFILES,
        capabilities: CONTENT_CAP,
        lifecycle: ToolLifecycle::Current,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
    ToolCatalogEntry {
        name: LIME_RUN_SERVICE_SKILL_TOOL_NAME,
        profiles: WORKBENCH_PROFILES,
        capabilities: EXECUTION_CAP,
        lifecycle: ToolLifecycle::Compat,
        source: ToolSourceKind::LimeInjected,
        permission_plane: ToolPermissionPlane::SessionAllowlist,
        workspace_default_allow: true,
    },
];

pub fn native_tool_catalog() -> &'static [ToolCatalogEntry] {
    NATIVE_TOOL_CATALOG
}

fn normalize_tool_catalog_alias(tool_name: &str) -> &str {
    match tool_catalog_reference_lookup_key(tool_name).as_str() {
        "requestuserinput" | "requestuserinputtool" => "request_user_input",
        "clocksleep" | "clock.sleep" | "sleep" => SLEEP_TOOL_NAME,
        "clockcurrtime" | "clock.curr_time" | "currtime" | "curr_time" => CURRENT_TIME_TOOL_NAME,
        "filereadtool" | "readfiletool" | "readfile" | "developerread" | "mcpsystemreadfile" => {
            "Read"
        }
        "applypatch" | "applypatchtool" => APPLY_PATCH_TOOL_NAME,
        "globtool" | "mcpsystemglob" => "Glob",
        "greptool" | "mcpsystemgrep" => "Grep",
        "listmcpresources" | "listmcpresourcestool" => LIST_MCP_RESOURCES_TOOL_NAME,
        "readmcpresource" | "readmcpresourcetool" => READ_MCP_RESOURCE_TOOL_NAME,
        "memorylist" | "memorylisttool" => MEMORY_LIST_TOOL_NAME,
        "memoryread" | "memoryreadtool" => MEMORY_READ_TOOL_NAME,
        "memorysearch" | "memorysearchtool" => MEMORY_SEARCH_TOOL_NAME,
        "memoryaddnote" | "memoryaddnotetool" => MEMORY_ADD_NOTE_TOOL_NAME,
        "skilltool" => "Skill",
        "syntheticoutputtool" => "StructuredOutput",
        "toolsearchtool" | "toolsearch" | "mcpsystemtoolsearch" => TOOL_SEARCH_TOOL_NAME,
        "skillsearchtool" | "skillsearch" | "skillssearch" => SKILL_SEARCH_TOOL_NAME,
        "updateplan" | "updateplantool" | "updateplan_tool" | "update_plan" => {
            UPDATE_PLAN_TOOL_NAME
        }
        "webfetchtool" | "webfetch" | "mcpsystemwebfetch" => "WebFetch",
        "websearchtool" | "websearch" | "mcpsystemwebsearch" => "WebSearch",
        "viewimage" | "viewimagetool" => VIEW_IMAGE_TOOL_NAME,
        _ => tool_name.trim(),
    }
}

fn tool_catalog_reference_lookup_key(tool_name: &str) -> String {
    tool_name
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

fn tool_catalog_lookup_key(tool_name: &str) -> String {
    normalize_tool_catalog_alias(tool_name)
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

pub fn tool_catalog_names_match(left: &str, right: &str) -> bool {
    let left_key = tool_catalog_lookup_key(left);
    let right_key = tool_catalog_lookup_key(right);
    !left_key.is_empty() && left_key == right_key
}

pub fn tool_catalog_entry(tool_name: &str) -> Option<&'static ToolCatalogEntry> {
    let requested_name = tool_name.trim();
    let canonical_name = normalize_tool_catalog_alias(requested_name);
    if let Some(entry) = native_tool_catalog()
        .iter()
        .find(|entry| entry.name == canonical_name)
    {
        return Some(entry);
    }
    let normalized_key = tool_catalog_lookup_key(canonical_name);
    native_tool_catalog()
        .iter()
        .filter(|entry| {
            if entry.name.ends_with("__") {
                requested_name.starts_with(entry.name)
                    || (!normalized_key.is_empty()
                        && normalized_key.starts_with(&tool_catalog_lookup_key(entry.name)))
            } else {
                tool_catalog_names_match(entry.name, canonical_name)
            }
        })
        .max_by_key(|entry| entry.name.len())
}

pub fn tool_catalog_entries_for_surface(
    surface: WorkspaceToolSurface,
) -> Vec<&'static ToolCatalogEntry> {
    native_tool_catalog()
        .iter()
        .filter(|entry| {
            entry
                .profiles
                .iter()
                .any(|profile| surface.includes_profile(*profile))
        })
        .collect()
}

pub fn workspace_default_allowed_tool_names(surface: WorkspaceToolSurface) -> Vec<&'static str> {
    let mut names = tool_catalog_entries_for_surface(surface)
        .into_iter()
        .filter(|entry| entry.workspace_default_allow)
        .filter(|entry| entry.lifecycle == ToolLifecycle::Current)
        .filter(|entry| !entry.name.ends_with("__"))
        .map(|entry| entry.name)
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    names
}

pub fn workspace_allowed_tool_names(surface: WorkspaceToolSurface) -> Vec<&'static str> {
    workspace_default_allowed_tool_names(surface)
}

pub fn workbench_tool_names() -> Vec<&'static str> {
    tool_catalog_entries_for_surface(WorkspaceToolSurface::workbench())
        .into_iter()
        .filter(|entry| entry.profiles.contains(&ToolSurfaceProfile::Workbench))
        .map(|entry| entry.name)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn test_workspace_default_allowed_tool_names_excludes_parameter_restricted_tools() {
        let names = workspace_default_allowed_tool_names(WorkspaceToolSurface::core());
        assert!(!names.contains(&"Agent"));
        assert!(!names.contains(&"SendMessage"));
        assert!(!names.contains(&"TeamCreate"));
        assert!(!names.contains(&"TeamDelete"));
        assert!(!names.contains(&"ListPeers"));
        assert!(names.contains(&"WebSearch"));
        assert!(names.contains(&MEMORY_LIST_TOOL_NAME));
        assert!(names.contains(&MEMORY_READ_TOOL_NAME));
        assert!(names.contains(&MEMORY_SEARCH_TOOL_NAME));
        assert!(names.contains(&MEMORY_ADD_NOTE_TOOL_NAME));
        assert!(!names.contains(&"Read"));
        assert!(!names.contains(&VIEW_IMAGE_TOOL_NAME));
        assert!(!names.contains(&"Bash"));
        assert!(!names.contains(&SOCIAL_IMAGE_TOOL_NAME));
    }

    #[test]
    fn test_tool_catalog_entry_rejects_retired_team_aliases() {
        for retired_name in [
            "Agent",
            "AgentTool",
            "SendMessage",
            "SendMessageTool",
            "TeamCreate",
            "TeamCreateTool",
            "TeamDelete",
            "TeamDeleteTool",
            "ListPeers",
            "ListPeersTool",
            "send_input",
            "subagent_task",
        ] {
            assert!(
                tool_catalog_entry(retired_name).is_none(),
                "retired Team tool must not resolve through the current catalog: {retired_name}"
            );
        }
        assert!(tool_catalog_entry("spawn_agent").is_none());
        assert!(tool_catalog_entry("brief").is_none());
        assert!(tool_catalog_entry("BriefTool").is_none());
        assert!(tool_catalog_entry("SendUserMessage").is_none());
        assert!(tool_catalog_entry("ask").is_none());
        assert!(tool_catalog_entry("AskUserQuestionTool").is_none());
        assert_eq!(
            tool_catalog_entry("request_user_input")
                .expect("Codex request_user_input should be current")
                .name,
            "request_user_input"
        );
        assert_eq!(
            tool_catalog_entry("clock.sleep")
                .expect("Codex clock.sleep should resolve to current sleep")
                .name,
            SLEEP_TOOL_NAME
        );
        assert_eq!(
            tool_catalog_entry("sleep")
                .expect("Codex sleep should be current")
                .name,
            SLEEP_TOOL_NAME
        );
        assert!(tool_catalog_entry("remote_trigger").is_none());
    }

    #[test]
    fn test_tool_catalog_entry_normalizes_reference_js_tool_names_to_current_surface() {
        let cases = [
            ("request_user_input", "request_user_input"),
            ("RequestUserInputTool", "request_user_input"),
            ("clock.sleep", SLEEP_TOOL_NAME),
            ("sleep", SLEEP_TOOL_NAME),
            ("ApplyPatchTool", APPLY_PATCH_TOOL_NAME),
            ("apply_patch", APPLY_PATCH_TOOL_NAME),
            ("FileReadTool", "Read"),
            ("read_file", "Read"),
            ("developer__read", "Read"),
            ("mcp__system__read_file", "Read"),
            ("GlobTool", "Glob"),
            ("mcp__system__glob", "Glob"),
            ("GrepTool", "Grep"),
            ("mcp__system__grep", "Grep"),
            ("ListMcpResourcesTool", LIST_MCP_RESOURCES_TOOL_NAME),
            ("MemoryListTool", MEMORY_LIST_TOOL_NAME),
            ("memory_list", MEMORY_LIST_TOOL_NAME),
            ("MemoryReadTool", MEMORY_READ_TOOL_NAME),
            ("memory_read", MEMORY_READ_TOOL_NAME),
            ("MemorySearchTool", MEMORY_SEARCH_TOOL_NAME),
            ("memory_search", MEMORY_SEARCH_TOOL_NAME),
            ("MemoryAddNoteTool", MEMORY_ADD_NOTE_TOOL_NAME),
            ("memory_add_note", MEMORY_ADD_NOTE_TOOL_NAME),
            ("ReadMcpResourceTool", READ_MCP_RESOURCE_TOOL_NAME),
            ("SkillTool", "Skill"),
            ("SyntheticOutputTool", "StructuredOutput"),
            ("update_plan", UPDATE_PLAN_TOOL_NAME),
            ("UpdatePlan", UPDATE_PLAN_TOOL_NAME),
            ("UpdatePlanTool", UPDATE_PLAN_TOOL_NAME),
            ("ToolSearch", TOOL_SEARCH_TOOL_NAME),
            ("ToolSearchTool", TOOL_SEARCH_TOOL_NAME),
            ("mcp__system__tool_search", TOOL_SEARCH_TOOL_NAME),
            ("WebFetchTool", "WebFetch"),
            ("web_fetch", "WebFetch"),
            ("mcp__system__web_fetch", "WebFetch"),
            ("WebSearchTool", "WebSearch"),
            ("web_search", "WebSearch"),
            ("mcp__system__web_search", "WebSearch"),
            ("ViewImageTool", VIEW_IMAGE_TOOL_NAME),
        ];

        for (input, expected) in cases {
            assert_eq!(
                tool_catalog_entry(input)
                    .unwrap_or_else(|| panic!("reference tool '{input}' should normalize"))
                    .name,
                expected
            );
        }
        assert_eq!(
            tool_catalog_entry("exec_command").map(|entry| entry.name),
            Some("exec_command")
        );
        assert_eq!(
            tool_catalog_entry("write_stdin").map(|entry| entry.name),
            Some("write_stdin")
        );
        for retired_shell_name in [
            "Bash",
            "BashTool",
            "PowerShell",
            "PowerShellTool",
            "shell_command",
            "developer__shell",
            "mcp__system__shell",
            "local_shell_call",
        ] {
            assert!(
                tool_catalog_entry(retired_shell_name).is_none(),
                "retired shell surface must not resolve: {retired_shell_name}"
            );
        }
        for deleted_tool_name in [
            "ConfigTool",
            "EnterWorktreeTool",
            "ExitWorktreeTool",
            "NotebookEditTool",
            "RemoteTriggerTool",
            "ScheduleCronTool",
            "CronCreateTool",
            "CronListTool",
            "CronDeleteTool",
            "SleepTool",
            "Edit",
            "FileEditTool",
            "edit_file",
            "developer__text_editor",
            "mcp__system__edit_file",
            "Write",
            "FileWriteTool",
            "write_file",
            "create_file",
            "mcp__system__write_file",
            "TaskCreateTool",
            "TaskGetTool",
            "TaskListTool",
            "TaskOutputTool",
            "TaskStopTool",
            "TaskUpdateTool",
            "WorkflowTool",
        ] {
            assert!(
                tool_catalog_entry(deleted_tool_name).is_none(),
                "deleted Agent tool alias should not resolve: {deleted_tool_name}"
            );
        }
    }

    #[test]
    fn test_tool_catalog_entry_leaves_intentional_reference_exceptions_unmapped() {
        for name in ["MCPTool", "McpAuthTool", "REPLTool"] {
            assert!(
                tool_catalog_entry(name).is_none(),
                "reference exception '{name}' should stay outside current catalog"
            );
        }
    }

    #[test]
    fn test_workspace_default_allowed_tool_names_includes_workbench_surface() {
        let names = workspace_default_allowed_tool_names(WorkspaceToolSurface::workbench());
        assert!(names.contains(&SOCIAL_IMAGE_TOOL_NAME));
        assert!(!names.contains(&LIME_CREATE_VIDEO_TASK_TOOL_NAME));
        assert!(names.contains(&LIME_CREATE_AUDIO_TASK_TOOL_NAME));
        assert!(names.contains(&LIME_CREATE_TRANSCRIPTION_TASK_TOOL_NAME));
    }

    #[test]
    fn test_tool_catalog_entries_for_surface_counts_and_lifecycle_boundaries() {
        let core = tool_catalog_entries_for_surface(WorkspaceToolSurface::core());
        let workbench_increment = native_tool_catalog()
            .iter()
            .filter(|entry| entry.profiles.contains(&ToolSurfaceProfile::Workbench))
            .count();
        assert_eq!(
            core.iter()
                .filter(|entry| entry.lifecycle == ToolLifecycle::Current)
                .count(),
            core.len()
        );
        assert!(core.iter().any(|entry| entry.name == VIEW_IMAGE_TOOL_NAME));
        assert!(core
            .iter()
            .any(|entry| entry.name == SKILL_SEARCH_TOOL_NAME));
        let sleep = core
            .iter()
            .find(|entry| entry.name == SLEEP_TOOL_NAME)
            .expect("sleep should stay in current catalog");
        assert_eq!(sleep.lifecycle, ToolLifecycle::Current);
        assert_eq!(
            sleep.permission_plane,
            ToolPermissionPlane::SessionAllowlist
        );
        assert!(sleep.workspace_default_allow);
        let apply_patch = core
            .iter()
            .find(|entry| entry.name == APPLY_PATCH_TOOL_NAME)
            .expect("apply_patch should stay in current catalog");
        assert_eq!(apply_patch.lifecycle, ToolLifecycle::Current);
        assert_eq!(
            apply_patch.permission_plane,
            ToolPermissionPlane::ParameterRestricted
        );
        assert!(!apply_patch.workspace_default_allow);
        assert_eq!(
            core.iter()
                .filter(|entry| entry.lifecycle == ToolLifecycle::Compat)
                .count(),
            0
        );
        assert!(core
            .iter()
            .all(|entry| !entry.profiles.contains(&ToolSurfaceProfile::Workbench)));
        let workbench = tool_catalog_entries_for_surface(WorkspaceToolSurface::workbench());
        assert_eq!(workbench.len(), core.len() + workbench_increment);
        assert!(workbench
            .iter()
            .any(|entry| entry.name == SOCIAL_IMAGE_TOOL_NAME));
    }

    #[test]
    fn test_workbench_tool_names_only_returns_workbench_increment() {
        let names = workbench_tool_names().into_iter().collect::<BTreeSet<_>>();
        assert_eq!(names.len(), 12);
        assert!(names.contains(SOCIAL_IMAGE_TOOL_NAME));
        assert!(names.contains(LIME_CREATE_VIDEO_TASK_TOOL_NAME));
        assert_eq!(
            tool_catalog_entry(LIME_CREATE_VIDEO_TASK_TOOL_NAME)
                .expect("retired video task tool should stay cataloged for guard visibility")
                .lifecycle,
            ToolLifecycle::Deprecated
        );
        assert!(names.contains(LIME_CREATE_AUDIO_TASK_TOOL_NAME));
        assert!(names.contains(LIME_CREATE_TRANSCRIPTION_TASK_TOOL_NAME));
        assert!(names.contains(LIME_RUN_SERVICE_SKILL_TOOL_NAME));
        assert!(names.contains(LIME_SEARCH_WEB_IMAGES_TOOL_NAME));
        assert!(!names.contains(TOOL_SEARCH_TOOL_NAME));
    }

    #[test]
    fn test_workspace_default_allowed_tool_names_for_workbench() {
        let surface = WorkspaceToolSurface::workbench();
        let names = workspace_default_allowed_tool_names(surface);
        let expected_len = tool_catalog_entries_for_surface(surface)
            .into_iter()
            .filter(|entry| entry.workspace_default_allow)
            .filter(|entry| entry.lifecycle == ToolLifecycle::Current)
            .filter(|entry| !entry.name.ends_with("__"))
            .map(|entry| entry.name)
            .collect::<BTreeSet<_>>()
            .len();
        assert_eq!(names.len(), expected_len);
        assert!(names.contains(&SOCIAL_IMAGE_TOOL_NAME));
        assert!(names.contains(&TOOL_SEARCH_TOOL_NAME));
        assert!(names.contains(&LIST_MCP_RESOURCES_TOOL_NAME));
        assert!(names.contains(&READ_MCP_RESOURCE_TOOL_NAME));
        assert!(!names.contains(&"TeamCreate"));
        assert!(!names.contains(&"TeamDelete"));
        assert!(names.contains(&LIME_CREATE_TRANSCRIPTION_TASK_TOOL_NAME));
        assert!(names.contains(&LIME_CREATE_AUDIO_TASK_TOOL_NAME));
        assert!(!names.contains(&LIME_CREATE_VIDEO_TASK_TOOL_NAME));
        assert!(names.contains(&LIME_SEARCH_WEB_IMAGES_TOOL_NAME));
        assert!(!names.contains(&LIME_RUN_SERVICE_SKILL_TOOL_NAME));
    }
}
