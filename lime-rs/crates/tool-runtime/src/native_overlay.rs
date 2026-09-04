use crate::apply_patch::{apply_patch_tool_definition, check_runtime_apply_patch_permissions};
use crate::image_task::{check_runtime_image_task_permissions, IMAGE_TASK_TOOL_NAME};
use crate::mcp_resource::{
    check_runtime_mcp_resource_permissions, LIST_MCP_RESOURCES_TOOL_NAME,
    READ_MCP_RESOURCE_TOOL_NAME,
};
use crate::memory_store::{
    check_runtime_memory_store_permissions, MEMORY_ADD_NOTE_TOOL_NAME, MEMORY_LIST_TOOL_NAME,
    MEMORY_READ_TOOL_NAME, MEMORY_SEARCH_TOOL_NAME,
};
use crate::request_user_input::REQUEST_USER_INPUT_TOOL_NAME;
use crate::skill_gate::skill_tool_definition;
use crate::skill_search::{check_runtime_skill_search_permissions, skill_search_tool_definition};
use crate::sleep::{check_runtime_sleep_permissions, sleep_tool_definition, CLOCK_SLEEP_TOOL_NAME};
use crate::tool_definition::RuntimeToolDefinition;
use crate::tool_executor::{RuntimeToolExecutionError, RuntimeToolTurnContext};
use crate::tool_search::{check_runtime_tool_search_permissions, TOOL_SEARCH_TOOL_NAME};
use crate::update_plan::{
    check_plan_update_permissions, update_plan_definition, UPDATE_PLAN_LEGACY_ALIASES,
};
use crate::video_task::{check_runtime_video_task_permissions, VIDEO_TASK_TOOL_NAME};
use crate::view_image::{
    check_runtime_view_image_permissions, view_image_tool_definition, VIEW_IMAGE_LEGACY_ALIASES,
};
use crate::web_fetch::{is_preapproved_web_fetch_host, web_fetch_tool_definition, WebFetchInput};
use crate::web_search::web_search_tool_definition;
use serde_json::Value;
use std::path::Path;
use url::Url;

/// Lime-owned native tool overlay installed on top of the temporary Agent registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeNativeToolOverlay {
    ViewImage,
    ApplyPatch,
    SkillSearch,
    Skill,
    Sleep,
    UpdatePlan,
    WebFetch,
    WebSearch,
}

impl RuntimeNativeToolOverlay {
    pub const fn name(self) -> &'static str {
        match self {
            Self::ViewImage => "view_image",
            Self::ApplyPatch => "apply_patch",
            Self::SkillSearch => "skill_search",
            Self::Skill => "Skill",
            Self::Sleep => "sleep",
            Self::UpdatePlan => "update_plan",
            Self::WebFetch => "WebFetch",
            Self::WebSearch => "WebSearch",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeNativeToolRegistrationOwner {
    NativeDispatch,
    SkillGate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeNativeToolTurnContextSource {
    None,
    AgentTurn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeNativeToolRegistration {
    tool: RuntimeNativeToolOverlay,
    name: &'static str,
    owner: RuntimeNativeToolRegistrationOwner,
    turn_context_source: RuntimeNativeToolTurnContextSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimeNativeToolInstallStep {
    registration: RuntimeNativeToolRegistration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeNativeToolSurface {
    definition: RuntimeToolDefinition,
    aliases: &'static [&'static str],
    max_retries: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeNativePermissionDecision {
    Allow,
    Deny(String),
    Ask(String),
}

impl RuntimeNativePermissionDecision {
    pub fn allow() -> Self {
        Self::Allow
    }

    pub fn deny(message: impl Into<String>) -> Self {
        Self::Deny(message.into())
    }

    pub fn ask(message: impl Into<String>) -> Self {
        Self::Ask(message.into())
    }

    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow)
    }

    pub fn is_denied(&self) -> bool {
        matches!(self, Self::Deny(_))
    }

    pub fn is_ask(&self) -> bool {
        matches!(self, Self::Ask(_))
    }
}

impl RuntimeNativeToolRegistration {
    pub const fn new(tool: RuntimeNativeToolOverlay) -> Self {
        Self {
            tool,
            name: tool.name(),
            owner: tool.registration_owner(),
            turn_context_source: tool.turn_context_source(),
        }
    }

    pub const fn tool(self) -> RuntimeNativeToolOverlay {
        self.tool
    }

    pub const fn name(self) -> &'static str {
        self.name
    }

    pub const fn owner(self) -> RuntimeNativeToolRegistrationOwner {
        self.owner
    }

    pub const fn turn_context_source(self) -> RuntimeNativeToolTurnContextSource {
        self.turn_context_source
    }
}

impl RuntimeNativeToolInstallStep {
    pub const fn new(registration: RuntimeNativeToolRegistration) -> Self {
        Self { registration }
    }

    pub const fn registration(self) -> RuntimeNativeToolRegistration {
        self.registration
    }

    pub const fn tool(self) -> RuntimeNativeToolOverlay {
        self.registration.tool()
    }

    pub const fn name(self) -> &'static str {
        self.registration.name()
    }

    pub const fn owner(self) -> RuntimeNativeToolRegistrationOwner {
        self.registration.owner()
    }

    pub const fn turn_context_source(self) -> RuntimeNativeToolTurnContextSource {
        self.registration.turn_context_source()
    }

    pub const fn registers_agent_tool(self) -> bool {
        matches!(
            self.registration.owner(),
            RuntimeNativeToolRegistrationOwner::NativeDispatch
        )
    }
}

impl RuntimeNativeToolSurface {
    pub fn new(
        definition: RuntimeToolDefinition,
        aliases: &'static [&'static str],
        max_retries: Option<u32>,
    ) -> Self {
        Self {
            definition,
            aliases,
            max_retries,
        }
    }

    pub fn name(&self) -> &str {
        &self.definition.name
    }

    pub fn description(&self) -> &str {
        &self.definition.description
    }

    pub fn input_schema(&self) -> serde_json::Value {
        self.definition.input_schema.clone()
    }

    pub fn definition(&self) -> RuntimeToolDefinition {
        self.definition.clone()
    }

    pub fn aliases(&self) -> &'static [&'static str] {
        self.aliases
    }

    pub fn max_retries(&self) -> Option<u32> {
        self.max_retries
    }
}

impl RuntimeNativeToolOverlay {
    pub const fn registration_owner(self) -> RuntimeNativeToolRegistrationOwner {
        match self {
            Self::Skill => RuntimeNativeToolRegistrationOwner::SkillGate,
            Self::ViewImage
            | Self::ApplyPatch
            | Self::SkillSearch
            | Self::Sleep
            | Self::UpdatePlan
            | Self::WebFetch
            | Self::WebSearch => RuntimeNativeToolRegistrationOwner::NativeDispatch,
        }
    }

    pub const fn turn_context_source(self) -> RuntimeNativeToolTurnContextSource {
        match self {
            Self::SkillSearch | Self::UpdatePlan | Self::WebFetch | Self::WebSearch => {
                RuntimeNativeToolTurnContextSource::AgentTurn
            }
            Self::ViewImage | Self::ApplyPatch | Self::Skill | Self::Sleep => {
                RuntimeNativeToolTurnContextSource::None
            }
        }
    }
}

const APPLY_PATCH_LEGACY_ALIASES: &[&str] = &["ApplyPatchTool"];
const SKILL_SEARCH_LEGACY_ALIASES: &[&str] = &["SkillSearchTool"];
const SLEEP_LEGACY_ALIASES: &[&str] = &[CLOCK_SLEEP_TOOL_NAME];
const NO_LEGACY_ALIASES: &[&str] = &[];

const RUNTIME_NATIVE_TOOL_OVERLAY: &[RuntimeNativeToolOverlay] = &[
    RuntimeNativeToolOverlay::ViewImage,
    RuntimeNativeToolOverlay::ApplyPatch,
    RuntimeNativeToolOverlay::SkillSearch,
    RuntimeNativeToolOverlay::Skill,
    RuntimeNativeToolOverlay::Sleep,
    RuntimeNativeToolOverlay::UpdatePlan,
    RuntimeNativeToolOverlay::WebFetch,
    RuntimeNativeToolOverlay::WebSearch,
];

const RUNTIME_NATIVE_TOOL_OVERLAY_REGISTRATIONS: &[RuntimeNativeToolRegistration] = &[
    RuntimeNativeToolRegistration::new(RuntimeNativeToolOverlay::ViewImage),
    RuntimeNativeToolRegistration::new(RuntimeNativeToolOverlay::ApplyPatch),
    RuntimeNativeToolRegistration::new(RuntimeNativeToolOverlay::SkillSearch),
    RuntimeNativeToolRegistration::new(RuntimeNativeToolOverlay::Skill),
    RuntimeNativeToolRegistration::new(RuntimeNativeToolOverlay::Sleep),
    RuntimeNativeToolRegistration::new(RuntimeNativeToolOverlay::UpdatePlan),
    RuntimeNativeToolRegistration::new(RuntimeNativeToolOverlay::WebFetch),
    RuntimeNativeToolRegistration::new(RuntimeNativeToolOverlay::WebSearch),
];

const RUNTIME_NATIVE_TOOL_INSTALL_PLAN: &[RuntimeNativeToolInstallStep] = &[
    RuntimeNativeToolInstallStep::new(RuntimeNativeToolRegistration::new(
        RuntimeNativeToolOverlay::ViewImage,
    )),
    RuntimeNativeToolInstallStep::new(RuntimeNativeToolRegistration::new(
        RuntimeNativeToolOverlay::ApplyPatch,
    )),
    RuntimeNativeToolInstallStep::new(RuntimeNativeToolRegistration::new(
        RuntimeNativeToolOverlay::SkillSearch,
    )),
    RuntimeNativeToolInstallStep::new(RuntimeNativeToolRegistration::new(
        RuntimeNativeToolOverlay::Skill,
    )),
    RuntimeNativeToolInstallStep::new(RuntimeNativeToolRegistration::new(
        RuntimeNativeToolOverlay::Sleep,
    )),
    RuntimeNativeToolInstallStep::new(RuntimeNativeToolRegistration::new(
        RuntimeNativeToolOverlay::UpdatePlan,
    )),
    RuntimeNativeToolInstallStep::new(RuntimeNativeToolRegistration::new(
        RuntimeNativeToolOverlay::WebFetch,
    )),
    RuntimeNativeToolInstallStep::new(RuntimeNativeToolRegistration::new(
        RuntimeNativeToolOverlay::WebSearch,
    )),
];

const RUNTIME_NATIVE_TOOL_OVERLAY_NAMES: &[&str] = &[
    "view_image",
    "apply_patch",
    "skill_search",
    "Skill",
    "sleep",
    "update_plan",
    "WebFetch",
    "WebSearch",
];

pub fn runtime_native_tool_overlay_tools() -> &'static [RuntimeNativeToolOverlay] {
    RUNTIME_NATIVE_TOOL_OVERLAY
}

pub fn runtime_native_tool_overlay_registrations() -> &'static [RuntimeNativeToolRegistration] {
    RUNTIME_NATIVE_TOOL_OVERLAY_REGISTRATIONS
}

pub fn runtime_native_tool_install_plan() -> &'static [RuntimeNativeToolInstallStep] {
    RUNTIME_NATIVE_TOOL_INSTALL_PLAN
}

pub fn runtime_native_tool_overlay_tool_names() -> &'static [&'static str] {
    RUNTIME_NATIVE_TOOL_OVERLAY_NAMES
}

pub fn runtime_native_tool_definition(tool: RuntimeNativeToolOverlay) -> RuntimeToolDefinition {
    match runtime_native_tool_surface(tool) {
        Some(surface) => surface.definition(),
        None => match tool {
            RuntimeNativeToolOverlay::Skill => skill_tool_definition(),
            RuntimeNativeToolOverlay::ViewImage
            | RuntimeNativeToolOverlay::ApplyPatch
            | RuntimeNativeToolOverlay::SkillSearch
            | RuntimeNativeToolOverlay::Sleep
            | RuntimeNativeToolOverlay::UpdatePlan
            | RuntimeNativeToolOverlay::WebFetch
            | RuntimeNativeToolOverlay::WebSearch => {
                unreachable!("native dispatch tool should expose runtime surface")
            }
        },
    }
}

pub fn runtime_native_tool_install_definitions() -> Vec<RuntimeToolDefinition> {
    runtime_native_tool_install_plan()
        .iter()
        .map(|step| runtime_native_tool_definition(step.tool()))
        .collect()
}

pub fn runtime_native_tool_surface(
    tool: RuntimeNativeToolOverlay,
) -> Option<RuntimeNativeToolSurface> {
    match tool {
        RuntimeNativeToolOverlay::ViewImage => Some(RuntimeNativeToolSurface::new(
            view_image_tool_definition(),
            VIEW_IMAGE_LEGACY_ALIASES,
            Some(0),
        )),
        RuntimeNativeToolOverlay::ApplyPatch => Some(RuntimeNativeToolSurface::new(
            apply_patch_tool_definition(),
            APPLY_PATCH_LEGACY_ALIASES,
            Some(0),
        )),
        RuntimeNativeToolOverlay::SkillSearch => Some(RuntimeNativeToolSurface::new(
            skill_search_tool_definition(),
            SKILL_SEARCH_LEGACY_ALIASES,
            None,
        )),
        RuntimeNativeToolOverlay::Skill => None,
        RuntimeNativeToolOverlay::Sleep => Some(RuntimeNativeToolSurface::new(
            sleep_tool_definition(),
            SLEEP_LEGACY_ALIASES,
            Some(0),
        )),
        RuntimeNativeToolOverlay::UpdatePlan => Some(RuntimeNativeToolSurface::new(
            update_plan_definition(),
            UPDATE_PLAN_LEGACY_ALIASES,
            Some(0),
        )),
        RuntimeNativeToolOverlay::WebFetch => Some(RuntimeNativeToolSurface::new(
            web_fetch_tool_definition(),
            NO_LEGACY_ALIASES,
            Some(0),
        )),
        RuntimeNativeToolOverlay::WebSearch => Some(RuntimeNativeToolSurface::new(
            web_search_tool_definition(),
            NO_LEGACY_ALIASES,
            Some(0),
        )),
    }
}

pub fn runtime_native_tool_overlay_for_dispatch_name(
    tool_name: &str,
) -> Option<RuntimeNativeToolOverlay> {
    let canonical_name = crate::native_dispatch::runtime_native_dispatch()
        .canonical_name(tool_name)?
        .to_string();
    runtime_native_tool_overlay_registrations()
        .iter()
        .find(|registration| {
            registration.owner() == RuntimeNativeToolRegistrationOwner::NativeDispatch
                && registration.name().eq_ignore_ascii_case(&canonical_name)
        })
        .map(|registration| registration.tool())
}

pub fn check_runtime_native_tool_permissions(
    tool: RuntimeNativeToolOverlay,
    params: &Value,
    working_directory: &Path,
    turn_context: Option<&RuntimeToolTurnContext>,
) -> RuntimeNativePermissionDecision {
    match tool {
        RuntimeNativeToolOverlay::ViewImage => permission_from_runtime_result(
            check_runtime_view_image_permissions(params, working_directory),
        ),
        RuntimeNativeToolOverlay::ApplyPatch => permission_from_runtime_result(
            check_runtime_apply_patch_permissions(params, working_directory),
        ),
        RuntimeNativeToolOverlay::SkillSearch => {
            permission_from_runtime_result(check_runtime_skill_search_permissions(params))
        }
        RuntimeNativeToolOverlay::Sleep => {
            permission_from_runtime_result(check_runtime_sleep_permissions(params))
        }
        RuntimeNativeToolOverlay::UpdatePlan => {
            permission_from_runtime_result(check_plan_update_permissions(params))
        }
        RuntimeNativeToolOverlay::WebFetch => {
            check_runtime_web_fetch_permissions(params, turn_context)
        }
        RuntimeNativeToolOverlay::WebSearch => check_runtime_web_search_permissions(turn_context),
        RuntimeNativeToolOverlay::Skill => RuntimeNativePermissionDecision::deny(
            "Skill gate permission is handled by the Skill runtime adapter",
        ),
    }
}

pub fn check_runtime_gateway_tool_permissions(
    tool_name: &str,
    params: &Value,
    working_directory: &Path,
    session_id: &str,
    turn_context: Option<&RuntimeToolTurnContext>,
) -> RuntimeNativePermissionDecision {
    match tool_name {
        MEMORY_LIST_TOOL_NAME
        | MEMORY_READ_TOOL_NAME
        | MEMORY_SEARCH_TOOL_NAME
        | MEMORY_ADD_NOTE_TOOL_NAME => permission_from_runtime_result(
            check_runtime_memory_store_permissions(tool_name, params, working_directory),
        ),
        IMAGE_TASK_TOOL_NAME => {
            permission_from_runtime_result(check_runtime_image_task_permissions(
                params,
                working_directory,
                session_id,
                turn_context,
            ))
        }
        VIDEO_TASK_TOOL_NAME => {
            permission_from_runtime_result(check_runtime_video_task_permissions(
                params,
                working_directory,
                session_id,
                turn_context,
            ))
        }
        TOOL_SEARCH_TOOL_NAME => {
            permission_from_runtime_result(check_runtime_tool_search_permissions())
        }
        LIST_MCP_RESOURCES_TOOL_NAME | READ_MCP_RESOURCE_TOOL_NAME => {
            permission_from_runtime_result(check_runtime_mcp_resource_permissions())
        }
        _ => RuntimeNativePermissionDecision::deny(format!(
            "unsupported gateway native tool: {tool_name}"
        )),
    }
}

fn permission_from_runtime_result(
    result: Result<(), RuntimeToolExecutionError>,
) -> RuntimeNativePermissionDecision {
    match result {
        Ok(()) => RuntimeNativePermissionDecision::allow(),
        Err(error) => RuntimeNativePermissionDecision::deny(error.message().to_string()),
    }
}

fn check_runtime_web_fetch_permissions(
    params: &Value,
    turn_context: Option<&RuntimeToolTurnContext>,
) -> RuntimeNativePermissionDecision {
    if turn_context_allows_web_tools_without_confirmation(turn_context) {
        return RuntimeNativePermissionDecision::allow();
    }

    let parsed_url = serde_json::from_value::<WebFetchInput>(params.clone())
        .ok()
        .and_then(|input| Url::parse(&input.url).ok());

    if let Some(url) = parsed_url.as_ref() {
        if let Some(hostname) = url.host_str() {
            if is_preapproved_web_fetch_host(hostname, url.path()) {
                return RuntimeNativePermissionDecision::allow();
            }
        }
    }

    match parsed_url.and_then(|url| url.host_str().map(str::to_string)) {
        Some(hostname) => RuntimeNativePermissionDecision::ask(format!(
            "WebFetch 将访问远程站点 {hostname}，请确认后继续。"
        )),
        None => RuntimeNativePermissionDecision::ask("WebFetch 将访问远程 URL，请确认后继续。"),
    }
}

fn check_runtime_web_search_permissions(
    turn_context: Option<&RuntimeToolTurnContext>,
) -> RuntimeNativePermissionDecision {
    if turn_context_allows_web_tools_without_confirmation(turn_context) {
        return RuntimeNativePermissionDecision::allow();
    }

    RuntimeNativePermissionDecision::ask("WebSearch 将联网搜索最新信息，请确认后继续。")
}

fn turn_context_metadata_bool(
    turn_context: Option<&RuntimeToolTurnContext>,
    keys: &[&str],
) -> bool {
    turn_context
        .and_then(|context| {
            keys.iter()
                .find_map(|key| context.metadata.get(*key))
                .and_then(serde_json::Value::as_bool)
        })
        .unwrap_or(false)
}

fn turn_context_approval_policy_is_never(turn_context: Option<&RuntimeToolTurnContext>) -> bool {
    turn_context
        .and_then(|context| context.approval_policy.as_deref())
        .map(str::trim)
        .is_some_and(|policy| policy.eq_ignore_ascii_case("never"))
}

fn turn_context_allows_web_tools_without_confirmation(
    turn_context: Option<&RuntimeToolTurnContext>,
) -> bool {
    turn_context_metadata_bool(turn_context, &["web_search_enabled", "webSearchEnabled"])
        || turn_context_approval_policy_is_never(turn_context)
}

/// Allowlist for tools that may still be registered through the Agent registry.
///
/// The list is owned here so App Server / GUI inventory and the migration guard have
/// one current policy source while registry-backed tools are migrated to their domain owners.
const RUNTIME_NATIVE_TOOL_REGISTRATION_ALLOWLIST: &[&str] = &[
    "exec_command",
    "write_stdin",
    "Read",
    "view_image",
    "Glob",
    "Grep",
    REQUEST_USER_INPUT_TOOL_NAME,
    "Skill",
    "apply_patch",
    "skill_search",
    "sleep",
    "update_plan",
    "WebFetch",
    "WebSearch",
    TOOL_SEARCH_TOOL_NAME,
    "memory_list",
    "memory_read",
    "memory_search",
    "memory_add_note",
    "lime_create_image_generation_task",
    "list_mcp_resources",
    "read_mcp_resource",
];

pub fn runtime_native_tool_registration_allowlist() -> &'static [&'static str] {
    RUNTIME_NATIVE_TOOL_REGISTRATION_ALLOWLIST
}

pub fn runtime_native_tool_registration_is_allowed(tool_name: &str) -> bool {
    let tool_name = tool_name.trim();
    !tool_name.is_empty()
        && RUNTIME_NATIVE_TOOL_REGISTRATION_ALLOWLIST
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(tool_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::{HashMap, HashSet};
    use tempfile::tempdir;

    #[test]
    fn runtime_native_tool_overlay_names_are_current_contract() {
        assert_eq!(
            runtime_native_tool_overlay_tool_names(),
            &[
                "view_image",
                "apply_patch",
                "skill_search",
                "Skill",
                "sleep",
                "update_plan",
                "WebFetch",
                "WebSearch"
            ]
        );
    }

    #[test]
    fn runtime_native_tool_overlay_name_slice_matches_tools() {
        let names = runtime_native_tool_overlay_tools()
            .iter()
            .map(|tool| tool.name())
            .collect::<Vec<_>>();

        assert_eq!(names, runtime_native_tool_overlay_tool_names());
    }

    #[test]
    fn runtime_native_tool_overlay_registrations_match_tools() {
        let tools = runtime_native_tool_overlay_tools().to_vec();
        let registration_tools = runtime_native_tool_overlay_registrations()
            .iter()
            .map(|registration| registration.tool())
            .collect::<Vec<_>>();
        let registration_names = runtime_native_tool_overlay_registrations()
            .iter()
            .map(|registration| registration.name())
            .collect::<Vec<_>>();

        assert_eq!(registration_tools, tools);
        assert_eq!(registration_names, runtime_native_tool_overlay_tool_names());
    }

    #[test]
    fn runtime_native_tool_install_plan_matches_registration_contract() {
        let registrations = runtime_native_tool_overlay_registrations()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let plan_registrations = runtime_native_tool_install_plan()
            .iter()
            .map(|step| step.registration())
            .collect::<Vec<_>>();
        let plan_names = runtime_native_tool_install_plan()
            .iter()
            .map(|step| step.name())
            .collect::<Vec<_>>();

        assert_eq!(plan_registrations, registrations);
        assert_eq!(plan_names, runtime_native_tool_overlay_tool_names());
        assert_eq!(
            runtime_native_tool_install_plan()
                .iter()
                .filter(|step| matches!(
                    step.turn_context_source(),
                    RuntimeNativeToolTurnContextSource::AgentTurn
                ))
                .map(|step| step.name())
                .collect::<Vec<_>>(),
            vec!["skill_search", "update_plan", "WebFetch", "WebSearch"]
        );
        assert_eq!(
            runtime_native_tool_install_definitions()
                .into_iter()
                .map(|definition| definition.name)
                .collect::<Vec<_>>(),
            plan_names
        );
    }

    #[test]
    fn runtime_native_tool_overlay_dispatch_plan_matches_current_dispatch() {
        let dispatch_names = crate::native_dispatch::runtime_native_dispatch_tool_names();

        for step in runtime_native_tool_install_plan() {
            match step.owner() {
                RuntimeNativeToolRegistrationOwner::NativeDispatch => {
                    assert!(
                        dispatch_names.iter().any(|name| name == step.name()),
                        "{} must be backed by tool-runtime native dispatch",
                        step.name()
                    );
                }
                RuntimeNativeToolRegistrationOwner::SkillGate => {
                    assert_eq!(step.name(), "Skill");
                    assert!(
                        !step.registers_agent_tool(),
                        "Skill must stay definition-only; execution is owned by the current live hook"
                    );
                }
            }
        }
    }

    #[test]
    fn runtime_native_tool_surface_matches_registration_plan() {
        for registration in runtime_native_tool_overlay_registrations() {
            match registration.owner() {
                RuntimeNativeToolRegistrationOwner::NativeDispatch => {
                    let surface = runtime_native_tool_surface(registration.tool())
                        .expect("dispatch-backed registration must have a current surface");
                    assert_eq!(surface.name(), registration.name());
                    assert!(!surface.description().trim().is_empty());
                    assert!(surface.input_schema().is_object());
                }
                RuntimeNativeToolRegistrationOwner::SkillGate => {
                    assert!(
                        runtime_native_tool_surface(registration.tool()).is_none(),
                        "Skill gate overlay must not pretend to be backed by native dispatch"
                    );
                }
            }
        }
    }

    #[test]
    fn runtime_native_tool_overlay_lookup_follows_dispatch_aliases() {
        assert_eq!(
            runtime_native_tool_overlay_for_dispatch_name("UpdatePlanTool"),
            Some(RuntimeNativeToolOverlay::UpdatePlan)
        );
        assert_eq!(
            runtime_native_tool_overlay_for_dispatch_name("clock.sleep"),
            Some(RuntimeNativeToolOverlay::Sleep)
        );
        assert_eq!(
            runtime_native_tool_overlay_for_dispatch_name("mcp__system__web_search"),
            Some(RuntimeNativeToolOverlay::WebSearch)
        );
        assert_eq!(runtime_native_tool_overlay_for_dispatch_name("Skill"), None);
        assert_eq!(runtime_native_tool_overlay_for_dispatch_name("Write"), None);
    }

    #[test]
    fn runtime_native_tool_surface_keeps_wrapper_alias_and_retry_contract() {
        assert_eq!(
            runtime_native_tool_surface(RuntimeNativeToolOverlay::ViewImage)
                .expect("view_image surface")
                .aliases(),
            &["ViewImage", "ViewImageTool"]
        );
        assert_eq!(
            runtime_native_tool_surface(RuntimeNativeToolOverlay::Sleep)
                .expect("sleep surface")
                .aliases(),
            &["clock.sleep"]
        );
        assert_eq!(
            runtime_native_tool_surface(RuntimeNativeToolOverlay::SkillSearch)
                .expect("skill_search surface")
                .max_retries(),
            None
        );
        assert_eq!(
            runtime_native_tool_surface(RuntimeNativeToolOverlay::ApplyPatch)
                .expect("apply_patch surface")
                .max_retries(),
            Some(0)
        );
    }

    #[test]
    fn runtime_native_tool_overlay_names_are_unique() {
        let names = runtime_native_tool_overlay_tool_names();
        let unique = names.iter().copied().collect::<HashSet<_>>();
        assert_eq!(unique.len(), names.len());
    }

    #[test]
    fn runtime_native_tool_registration_allowlist_is_codex_first_contract() {
        let names = runtime_native_tool_registration_allowlist();
        let unique = names.iter().copied().collect::<HashSet<_>>();
        assert_eq!(unique.len(), names.len());
        assert!(names.contains(&"exec_command"));
        assert!(names.contains(&"write_stdin"));
        assert!(!names.contains(&"Bash"));
        assert!(!names.contains(&"PowerShell"));
        assert!(names.contains(&"Read"));
        assert!(names.contains(&"view_image"));
        assert!(names.contains(&REQUEST_USER_INPUT_TOOL_NAME));
        assert!(names.contains(&"apply_patch"));
        assert!(names.contains(&"sleep"));
        assert!(names.contains(&"update_plan"));
        assert!(!names.contains(&"Ask"));
        assert!(!names.contains(&"LSP"));
        assert!(!names.contains(&"LSPTool"));
        assert!(!names.contains(&"Write"));
        assert!(!names.contains(&"Edit"));
        assert!(!names.contains(&"TaskCreate"));
        assert!(!names.contains(&"TaskList"));
        assert!(!names.contains(&"TaskGet"));
        assert!(!names.contains(&"TaskUpdate"));
        assert!(!names.contains(&"TaskOutput"));
        assert!(!names.contains(&"TaskStop"));
        assert!(!names.contains(&"ToolSearch"));
        assert!(!names.contains(&"ToolSearchTool"));
        assert!(!names.contains(&"UpdatePlan"));
        assert!(!names.contains(&"ViewImage"));
        assert!(!names.contains(&"EnterPlanMode"));
        assert!(!names.contains(&"ExitPlanMode"));
        assert!(!names.contains(&"NotebookEdit"));
        assert!(!names.contains(&"EnterWorktree"));
        assert!(!names.contains(&"Workflow"));
        assert!(!names.contains(&"Config"));
        assert!(!names.contains(&"Sleep"));
        assert!(!names.contains(&"SleepTool"));
        assert!(!names.contains(&"Cron"));
        assert!(!names.contains(&"RemoteTrigger"));
        assert!(!names.contains(&"Agent"));
        assert!(!names.contains(&"SendMessage"));
        assert!(!names.contains(&"TeamCreate"));
        assert!(!names.contains(&"TeamDelete"));
        assert!(!names.contains(&"ListPeers"));
    }

    #[test]
    fn runtime_native_tool_registration_policy_matches_allowlist() {
        assert!(runtime_native_tool_registration_is_allowed("memory_list"));
        assert!(runtime_native_tool_registration_is_allowed(" Memory_List "));
        assert!(runtime_native_tool_registration_is_allowed("WebSearch"));
        assert!(runtime_native_tool_registration_is_allowed("tool_search"));
        assert!(!runtime_native_tool_registration_is_allowed(""));
        assert!(!runtime_native_tool_registration_is_allowed("Write"));
        assert!(!runtime_native_tool_registration_is_allowed(
            "RuntimeApprovalResume"
        ));
    }

    #[test]
    fn runtime_native_permissions_validate_stateless_current_rules() {
        let dir = tempdir().expect("tempdir");
        let denied_patch = check_runtime_native_tool_permissions(
            RuntimeNativeToolOverlay::ApplyPatch,
            &json!({
                "patch": "*** Begin Patch\n*** Add File: ../outside.md\n+blocked\n*** End Patch"
            }),
            dir.path(),
            None,
        );
        let allowed_patch = check_runtime_native_tool_permissions(
            RuntimeNativeToolOverlay::ApplyPatch,
            &json!({
                "patch": "*** Begin Patch\n*** Add File: notes/current.md\n+ok\n*** End Patch"
            }),
            dir.path(),
            None,
        );
        let invalid_sleep = check_runtime_native_tool_permissions(
            RuntimeNativeToolOverlay::Sleep,
            &json!({ "seconds": 1 }),
            dir.path(),
            None,
        );
        let invalid_plan = check_runtime_native_tool_permissions(
            RuntimeNativeToolOverlay::UpdatePlan,
            &json!({
                "plan": [
                    { "step": "第一步", "status": "in_progress" },
                    { "step": "第二步", "status": "in_progress" }
                ]
            }),
            dir.path(),
            None,
        );

        assert!(denied_patch.is_denied());
        assert!(allowed_patch.is_allowed());
        assert!(invalid_sleep.is_denied());
        assert!(invalid_plan.is_denied());
    }

    #[test]
    fn runtime_native_permissions_own_web_confirmation_policy() {
        let dir = tempdir().expect("tempdir");
        let ask_fetch = check_runtime_native_tool_permissions(
            RuntimeNativeToolOverlay::WebFetch,
            &json!({
                "url": "https://example.com/docs",
                "prompt": "总结内容"
            }),
            dir.path(),
            None,
        );
        let allowed_fetch = check_runtime_native_tool_permissions(
            RuntimeNativeToolOverlay::WebFetch,
            &json!({
                "url": "https://react.dev/reference/react/useEffect",
                "prompt": "总结内容"
            }),
            dir.path(),
            None,
        );
        let never_policy = RuntimeToolTurnContext {
            approval_policy: Some("never".to_string()),
            ..RuntimeToolTurnContext::default()
        };
        let metadata_policy = RuntimeToolTurnContext {
            metadata: HashMap::from([("web_search_enabled".to_string(), json!(true))]),
            ..RuntimeToolTurnContext::default()
        };
        let allowed_search_by_policy = check_runtime_native_tool_permissions(
            RuntimeNativeToolOverlay::WebSearch,
            &json!({ "query": "latest ai news" }),
            dir.path(),
            Some(&never_policy),
        );
        let allowed_fetch_by_metadata = check_runtime_native_tool_permissions(
            RuntimeNativeToolOverlay::WebFetch,
            &json!({
                "url": "https://example.com/docs",
                "prompt": "总结内容"
            }),
            dir.path(),
            Some(&metadata_policy),
        );

        assert!(ask_fetch.is_ask());
        assert_eq!(
            ask_fetch,
            RuntimeNativePermissionDecision::Ask(
                "WebFetch 将访问远程站点 example.com，请确认后继续。".to_string()
            )
        );
        assert!(allowed_fetch.is_allowed());
        assert!(allowed_search_by_policy.is_allowed());
        assert!(allowed_fetch_by_metadata.is_allowed());
    }
}
