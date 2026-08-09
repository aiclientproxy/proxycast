use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const METHOD_THREAD_START: &str = "thread/start";
pub const METHOD_THREAD_FORK: &str = "thread/fork";
pub const METHOD_THREAD_RESUME: &str = "thread/resume";
pub const METHOD_THREAD_READ: &str = "thread/read";
pub const METHOD_THREAD_LIST: &str = "thread/list";
pub const METHOD_THREAD_SECTION_MOVE: &str = "thread/section/move";
pub const METHOD_THREAD_SECTION_LIST: &str = "threadSection/list";
pub const METHOD_THREAD_SECTION_CREATE: &str = "threadSection/create";
pub const METHOD_THREAD_SECTION_UPDATE: &str = "threadSection/update";
pub const METHOD_THREAD_SECTION_DELETE: &str = "threadSection/delete";
pub const METHOD_THREAD_LOADED_LIST: &str = "thread/loaded/list";
pub const METHOD_THREAD_UNSUBSCRIBE: &str = "thread/unsubscribe";
pub const METHOD_THREAD_INCREMENT_ELICITATION: &str = "thread/increment_elicitation";
pub const METHOD_THREAD_DECREMENT_ELICITATION: &str = "thread/decrement_elicitation";
pub const METHOD_THREAD_ARCHIVE: &str = "thread/archive";
pub const METHOD_THREAD_DELETE: &str = "thread/delete";
pub const METHOD_THREAD_UNARCHIVE: &str = "thread/unarchive";
pub const METHOD_THREAD_NAME_SET: &str = "thread/name/set";
pub const METHOD_THREAD_METADATA_UPDATE: &str = "thread/metadata/update";
pub const METHOD_THREAD_COMPACT_START: &str = "thread/compact/start";
pub const METHOD_THREAD_TURNS_LIST: &str = "thread/turns/list";
pub const METHOD_THREAD_ITEMS_LIST: &str = "thread/items/list";
pub const METHOD_THREAD_INJECT_ITEMS: &str = "thread/inject_items";
pub const METHOD_THREAD_SEARCH: &str = "thread/search";
pub const METHOD_THREAD_SEARCH_OCCURRENCES: &str = "thread/searchOccurrences";
pub const METHOD_THREAD_SETTINGS_UPDATE: &str = "thread/settings/update";
pub const METHOD_THREAD_MEMORY_MODE_SET: &str = "thread/memoryMode/set";
pub const METHOD_MEMORY_RESET: &str = "memory/reset";
pub const METHOD_THREAD_SHELL_COMMAND: &str = "thread/shellCommand";
pub const METHOD_THREAD_APPROVE_GUARDIAN_DENIED_ACTION: &str = "thread/approveGuardianDeniedAction";
pub const METHOD_THREAD_BACKGROUND_TERMINALS_CLEAN: &str = "thread/backgroundTerminals/clean";
pub const METHOD_THREAD_BACKGROUND_TERMINALS_LIST: &str = "thread/backgroundTerminals/list";
pub const METHOD_THREAD_BACKGROUND_TERMINALS_TERMINATE: &str =
    "thread/backgroundTerminals/terminate";
pub const METHOD_THREAD_GOAL_SET: &str = "thread/goal/set";
pub const METHOD_THREAD_GOAL_GET: &str = "thread/goal/get";
pub const METHOD_THREAD_GOAL_CLEAR: &str = "thread/goal/clear";
pub const METHOD_ARTIFACT_WRITE: &str = "artifact/write";
pub const METHOD_MEDIA_READ: &str = "media/read";
pub const METHOD_MCP_SERVER_RESOURCE_READ: &str = "mcpServer/resource/read";
pub const METHOD_MCP_SERVER_TOOL_CALL: &str = "mcpServer/tool/call";
pub const METHOD_MODEL_LIST: &str = "model/list";
pub const METHOD_APP_READ: &str = "app/read";
pub const METHOD_APP_LIST: &str = "app/list";
pub const METHOD_APP_INSTALLED: &str = "app/installed";
pub const METHOD_HOOKS_LIST: &str = "hooks/list";
pub const METHOD_SKILLS_LIST: &str = "skills/list";
pub const METHOD_SKILLS_EXTRA_ROOTS_SET: &str = "skills/extraRoots/set";
pub const METHOD_SKILLS_CONFIG_WRITE: &str = "skills/config/write";
pub const METHOD_PLUGIN_LIST: &str = "plugin/list";
pub const METHOD_PLUGIN_SEARCH: &str = "plugin/search";
pub const METHOD_PLUGIN_READ: &str = "plugin/read";
pub const METHOD_PLUGIN_INSTALL: &str = "plugin/install";
pub const METHOD_PLUGIN_UNINSTALL: &str = "plugin/uninstall";
pub const METHOD_PLUGIN_INSTALLED: &str = "plugin/installed";
pub const METHOD_PLUGIN_ENABLED_SET: &str = "plugin/enabled/set";
pub const METHOD_MODEL_LIST_UPDATED: &str = "model/list/updated";
pub const METHOD_APP_LIST_UPDATED: &str = "app/list/updated";
pub const METHOD_HOOK_STARTED: &str = "hook/started";
pub const METHOD_HOOK_COMPLETED: &str = "hook/completed";
pub const METHOD_TURN_START: &str = "turn/start";
pub const METHOD_TURN_STEER: &str = "turn/steer";
pub const METHOD_TURN_INTERRUPT: &str = "turn/interrupt";
pub const METHOD_REVIEW_START: &str = "review/start";
pub const METHOD_FS_READ_FILE: &str = "fs/readFile";
pub const METHOD_FS_WRITE_FILE: &str = "fs/writeFile";
pub const METHOD_FS_CREATE_DIRECTORY: &str = "fs/createDirectory";
pub const METHOD_FS_GET_METADATA: &str = "fs/getMetadata";
pub const METHOD_FS_READ_DIRECTORY: &str = "fs/readDirectory";
pub const METHOD_FS_REMOVE: &str = "fs/remove";
pub const METHOD_FS_COPY: &str = "fs/copy";
pub const METHOD_FS_WATCH: &str = "fs/watch";
pub const METHOD_FS_UNWATCH: &str = "fs/unwatch";
pub const METHOD_PROCESS_SPAWN: &str = "process/spawn";
pub const METHOD_PROCESS_WRITE_STDIN: &str = "process/writeStdin";
pub const METHOD_PROCESS_RESIZE_PTY: &str = "process/resizePty";
pub const METHOD_PROCESS_KILL: &str = "process/kill";
pub const METHOD_PROCESS_OUTPUT_DELTA: &str = "process/outputDelta";
pub const METHOD_PROCESS_EXITED: &str = "process/exited";
pub const METHOD_COMMAND_EXEC: &str = "command/exec";
pub const METHOD_COMMAND_EXEC_WRITE: &str = "command/exec/write";
pub const METHOD_COMMAND_EXEC_RESIZE: &str = "command/exec/resize";
pub const METHOD_COMMAND_EXEC_TERMINATE: &str = "command/exec/terminate";
pub const METHOD_COMMAND_EXEC_OUTPUT_DELTA: &str = "command/exec/outputDelta";
pub const METHOD_FS_CHANGED: &str = "fs/changed";
pub const METHOD_THREAD_STARTED: &str = "thread/started";
pub const METHOD_THREAD_ARCHIVED: &str = "thread/archived";
pub const METHOD_THREAD_DELETED: &str = "thread/deleted";
pub const METHOD_THREAD_UNARCHIVED: &str = "thread/unarchived";
pub const METHOD_THREAD_CLOSED: &str = "thread/closed";
pub const METHOD_THREAD_NAME_UPDATED: &str = "thread/name/updated";
pub const METHOD_THREAD_STATUS_CHANGED: &str = "thread/status/changed";
pub const METHOD_TURN_STARTED: &str = "turn/started";
pub const METHOD_TURN_COMPLETED: &str = "turn/completed";
pub const METHOD_TURN_DIFF_UPDATED: &str = "turn/diff/updated";
pub const METHOD_TURN_MODERATION_METADATA: &str = "turn/moderationMetadata";
pub const METHOD_TURN_PLAN_UPDATED: &str = "turn/plan/updated";
pub const METHOD_ITEM_STARTED: &str = "item/started";
pub const METHOD_ITEM_COMPLETED: &str = "item/completed";
pub const METHOD_ITEM_AUTO_APPROVAL_REVIEW_STARTED: &str = "item/autoApprovalReview/started";
pub const METHOD_ITEM_AUTO_APPROVAL_REVIEW_COMPLETED: &str = "item/autoApprovalReview/completed";
pub const METHOD_AGENT_MESSAGE_DELTA: &str = "item/agentMessage/delta";
pub const METHOD_COMMAND_EXECUTION_OUTPUT_DELTA: &str = "item/commandExecution/outputDelta";
pub const METHOD_COMMAND_EXECUTION_TERMINAL_INTERACTION: &str =
    "item/commandExecution/terminalInteraction";
pub const METHOD_FILE_CHANGE_PATCH_UPDATED: &str = "item/fileChange/patchUpdated";
pub const METHOD_PLAN_DELTA: &str = "item/plan/delta";
pub const METHOD_MCP_TOOL_CALL_PROGRESS: &str = "item/mcpToolCall/progress";
pub const METHOD_REASONING_SUMMARY_TEXT_DELTA: &str = "item/reasoning/summaryTextDelta";
pub const METHOD_REASONING_SUMMARY_PART_ADDED: &str = "item/reasoning/summaryPartAdded";
pub const METHOD_REASONING_TEXT_DELTA: &str = "item/reasoning/textDelta";
pub const METHOD_MODEL_REROUTED: &str = "model/rerouted";
pub const METHOD_MODEL_VERIFICATION: &str = "model/verification";
pub const METHOD_MODEL_SAFETY_BUFFERING_UPDATED: &str = "model/safetyBuffering/updated";
pub const METHOD_THREAD_SETTINGS_UPDATED: &str = "thread/settings/updated";
pub const METHOD_THREAD_TOKEN_USAGE_UPDATED: &str = "thread/tokenUsage/updated";
pub const METHOD_THREAD_GOAL_UPDATED: &str = "thread/goal/updated";
pub const METHOD_THREAD_GOAL_CLEARED: &str = "thread/goal/cleared";
pub const METHOD_SERVER_REQUEST_RESOLVED: &str = "serverRequest/resolved";
pub const METHOD_CONFIG_WARNING: &str = "configWarning";
pub const METHOD_WARNING: &str = "warning";
pub const METHOD_GUARDIAN_WARNING: &str = "guardianWarning";
pub const METHOD_ERROR: &str = "error";
pub const METHOD_SKILLS_CHANGED: &str = "skills/changed";
pub const METHOD_MCP_SERVER_OAUTH_LOGIN_COMPLETED: &str = "mcpServer/oauthLogin/completed";
pub const METHOD_MCP_SERVER_STARTUP_STATUS_UPDATED: &str = "mcpServer/startupStatus/updated";
pub const METHOD_CURRENT_TIME_READ: &str = "currentTime/read";
pub const METHOD_MCP_SERVER_ELICITATION_REQUEST: &str = "mcpServer/elicitation/request";
pub const METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL: &str =
    "item/commandExecution/requestApproval";
pub const METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL: &str = "item/fileChange/requestApproval";
pub const METHOD_ITEM_PERMISSIONS_REQUEST_APPROVAL: &str = "item/permissions/requestApproval";
pub const METHOD_ITEM_TOOL_CALL: &str = "item/tool/call";
pub const METHOD_ITEM_TOOL_REQUEST_USER_INPUT: &str = "item/tool/requestUserInput";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum Method {
    #[serde(rename = "thread/start")]
    ThreadStart,
    #[serde(rename = "thread/fork")]
    ThreadFork,
    #[serde(rename = "thread/resume")]
    ThreadResume,
    #[serde(rename = "thread/read")]
    ThreadRead,
    #[serde(rename = "thread/list")]
    ThreadList,
    #[serde(rename = "thread/section/move")]
    ThreadSectionMove,
    #[serde(rename = "threadSection/list")]
    ThreadSectionList,
    #[serde(rename = "threadSection/create")]
    ThreadSectionCreate,
    #[serde(rename = "threadSection/update")]
    ThreadSectionUpdate,
    #[serde(rename = "threadSection/delete")]
    ThreadSectionDelete,
    #[serde(rename = "thread/loaded/list")]
    ThreadLoadedList,
    #[serde(rename = "thread/unsubscribe")]
    ThreadUnsubscribe,
    #[serde(rename = "thread/increment_elicitation")]
    ThreadIncrementElicitation,
    #[serde(rename = "thread/decrement_elicitation")]
    ThreadDecrementElicitation,
    #[serde(rename = "thread/archive")]
    ThreadArchive,
    #[serde(rename = "thread/delete")]
    ThreadDelete,
    #[serde(rename = "thread/unarchive")]
    ThreadUnarchive,
    #[serde(rename = "thread/name/set")]
    ThreadSetName,
    #[serde(rename = "thread/metadata/update")]
    ThreadMetadataUpdate,
    #[serde(rename = "thread/compact/start")]
    ThreadCompactStart,
    #[serde(rename = "thread/turns/list")]
    ThreadTurnsList,
    #[serde(rename = "thread/items/list")]
    ThreadItemsList,
    #[serde(rename = "thread/inject_items")]
    ThreadInjectItems,
    #[serde(rename = "thread/search")]
    ThreadSearch,
    #[serde(rename = "thread/searchOccurrences")]
    ThreadSearchOccurrences,
    #[serde(rename = "thread/settings/update")]
    ThreadSettingsUpdate,
    #[serde(rename = "thread/memoryMode/set")]
    ThreadMemoryModeSet,
    #[serde(rename = "memory/reset")]
    MemoryReset,
    #[serde(rename = "thread/shellCommand")]
    ThreadShellCommand,
    #[serde(rename = "thread/approveGuardianDeniedAction")]
    ThreadApproveGuardianDeniedAction,
    #[serde(rename = "thread/backgroundTerminals/clean")]
    ThreadBackgroundTerminalsClean,
    #[serde(rename = "thread/backgroundTerminals/list")]
    ThreadBackgroundTerminalsList,
    #[serde(rename = "thread/backgroundTerminals/terminate")]
    ThreadBackgroundTerminalsTerminate,
    #[serde(rename = "thread/goal/set")]
    ThreadGoalSet,
    #[serde(rename = "thread/goal/get")]
    ThreadGoalGet,
    #[serde(rename = "thread/goal/clear")]
    ThreadGoalClear,
    #[serde(rename = "artifact/write")]
    ArtifactWrite,
    #[serde(rename = "media/read")]
    MediaRead,
    #[serde(rename = "mcpServer/resource/read")]
    McpServerResourceRead,
    #[serde(rename = "mcpServer/tool/call")]
    McpServerToolCall,
    #[serde(rename = "model/list")]
    ModelList,
    #[serde(rename = "app/read")]
    AppRead,
    #[serde(rename = "app/list")]
    AppList,
    #[serde(rename = "app/installed")]
    AppInstalled,
    #[serde(rename = "hooks/list")]
    HooksList,
    #[serde(rename = "skills/list")]
    SkillsList,
    #[serde(rename = "skills/extraRoots/set")]
    SkillsExtraRootsSet,
    #[serde(rename = "skills/config/write")]
    SkillsConfigWrite,
    #[serde(rename = "plugin/list")]
    PluginList,
    #[serde(rename = "plugin/search")]
    PluginSearch,
    #[serde(rename = "plugin/read")]
    PluginRead,
    #[serde(rename = "plugin/install")]
    PluginInstall,
    #[serde(rename = "plugin/uninstall")]
    PluginUninstall,
    #[serde(rename = "plugin/installed")]
    PluginInstalled,
    #[serde(rename = "plugin/enabled/set")]
    PluginEnabledSet,
    #[serde(rename = "turn/start")]
    TurnStart,
    #[serde(rename = "turn/steer")]
    TurnSteer,
    #[serde(rename = "turn/interrupt")]
    TurnInterrupt,
    #[serde(rename = "review/start")]
    ReviewStart,
    #[serde(rename = "fs/readFile")]
    FsReadFile,
    #[serde(rename = "fs/writeFile")]
    FsWriteFile,
    #[serde(rename = "fs/createDirectory")]
    FsCreateDirectory,
    #[serde(rename = "fs/getMetadata")]
    FsGetMetadata,
    #[serde(rename = "fs/readDirectory")]
    FsReadDirectory,
    #[serde(rename = "fs/remove")]
    FsRemove,
    #[serde(rename = "fs/copy")]
    FsCopy,
    #[serde(rename = "fs/watch")]
    FsWatch,
    #[serde(rename = "fs/unwatch")]
    FsUnwatch,
    #[serde(rename = "process/spawn")]
    ProcessSpawn,
    #[serde(rename = "process/writeStdin")]
    ProcessWriteStdin,
    #[serde(rename = "process/resizePty")]
    ProcessResizePty,
    #[serde(rename = "process/kill")]
    ProcessKill,
    #[serde(rename = "command/exec")]
    CommandExec,
    #[serde(rename = "command/exec/write")]
    CommandExecWrite,
    #[serde(rename = "command/exec/resize")]
    CommandExecResize,
    #[serde(rename = "command/exec/terminate")]
    CommandExecTerminate,
}

impl Method {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ThreadStart => METHOD_THREAD_START,
            Self::ThreadFork => METHOD_THREAD_FORK,
            Self::ThreadResume => METHOD_THREAD_RESUME,
            Self::ThreadRead => METHOD_THREAD_READ,
            Self::ThreadList => METHOD_THREAD_LIST,
            Self::ThreadSectionMove => METHOD_THREAD_SECTION_MOVE,
            Self::ThreadSectionList => METHOD_THREAD_SECTION_LIST,
            Self::ThreadSectionCreate => METHOD_THREAD_SECTION_CREATE,
            Self::ThreadSectionUpdate => METHOD_THREAD_SECTION_UPDATE,
            Self::ThreadSectionDelete => METHOD_THREAD_SECTION_DELETE,
            Self::ThreadLoadedList => METHOD_THREAD_LOADED_LIST,
            Self::ThreadUnsubscribe => METHOD_THREAD_UNSUBSCRIBE,
            Self::ThreadIncrementElicitation => METHOD_THREAD_INCREMENT_ELICITATION,
            Self::ThreadDecrementElicitation => METHOD_THREAD_DECREMENT_ELICITATION,
            Self::ThreadArchive => METHOD_THREAD_ARCHIVE,
            Self::ThreadDelete => METHOD_THREAD_DELETE,
            Self::ThreadUnarchive => METHOD_THREAD_UNARCHIVE,
            Self::ThreadSetName => METHOD_THREAD_NAME_SET,
            Self::ThreadMetadataUpdate => METHOD_THREAD_METADATA_UPDATE,
            Self::ThreadCompactStart => METHOD_THREAD_COMPACT_START,
            Self::ThreadTurnsList => METHOD_THREAD_TURNS_LIST,
            Self::ThreadItemsList => METHOD_THREAD_ITEMS_LIST,
            Self::ThreadInjectItems => METHOD_THREAD_INJECT_ITEMS,
            Self::ThreadSearch => METHOD_THREAD_SEARCH,
            Self::ThreadSearchOccurrences => METHOD_THREAD_SEARCH_OCCURRENCES,
            Self::ThreadSettingsUpdate => METHOD_THREAD_SETTINGS_UPDATE,
            Self::ThreadMemoryModeSet => METHOD_THREAD_MEMORY_MODE_SET,
            Self::MemoryReset => METHOD_MEMORY_RESET,
            Self::ThreadShellCommand => METHOD_THREAD_SHELL_COMMAND,
            Self::ThreadApproveGuardianDeniedAction => METHOD_THREAD_APPROVE_GUARDIAN_DENIED_ACTION,
            Self::ThreadBackgroundTerminalsClean => METHOD_THREAD_BACKGROUND_TERMINALS_CLEAN,
            Self::ThreadBackgroundTerminalsList => METHOD_THREAD_BACKGROUND_TERMINALS_LIST,
            Self::ThreadBackgroundTerminalsTerminate => {
                METHOD_THREAD_BACKGROUND_TERMINALS_TERMINATE
            }
            Self::ThreadGoalSet => METHOD_THREAD_GOAL_SET,
            Self::ThreadGoalGet => METHOD_THREAD_GOAL_GET,
            Self::ThreadGoalClear => METHOD_THREAD_GOAL_CLEAR,
            Self::ArtifactWrite => METHOD_ARTIFACT_WRITE,
            Self::MediaRead => METHOD_MEDIA_READ,
            Self::McpServerResourceRead => METHOD_MCP_SERVER_RESOURCE_READ,
            Self::McpServerToolCall => METHOD_MCP_SERVER_TOOL_CALL,
            Self::ModelList => METHOD_MODEL_LIST,
            Self::AppRead => METHOD_APP_READ,
            Self::AppList => METHOD_APP_LIST,
            Self::AppInstalled => METHOD_APP_INSTALLED,
            Self::HooksList => METHOD_HOOKS_LIST,
            Self::SkillsList => METHOD_SKILLS_LIST,
            Self::SkillsExtraRootsSet => METHOD_SKILLS_EXTRA_ROOTS_SET,
            Self::SkillsConfigWrite => METHOD_SKILLS_CONFIG_WRITE,
            Self::PluginList => METHOD_PLUGIN_LIST,
            Self::PluginSearch => METHOD_PLUGIN_SEARCH,
            Self::PluginRead => METHOD_PLUGIN_READ,
            Self::PluginInstall => METHOD_PLUGIN_INSTALL,
            Self::PluginUninstall => METHOD_PLUGIN_UNINSTALL,
            Self::PluginInstalled => METHOD_PLUGIN_INSTALLED,
            Self::PluginEnabledSet => METHOD_PLUGIN_ENABLED_SET,
            Self::TurnStart => METHOD_TURN_START,
            Self::TurnSteer => METHOD_TURN_STEER,
            Self::TurnInterrupt => METHOD_TURN_INTERRUPT,
            Self::ReviewStart => METHOD_REVIEW_START,
            Self::FsReadFile => METHOD_FS_READ_FILE,
            Self::FsWriteFile => METHOD_FS_WRITE_FILE,
            Self::FsCreateDirectory => METHOD_FS_CREATE_DIRECTORY,
            Self::FsGetMetadata => METHOD_FS_GET_METADATA,
            Self::FsReadDirectory => METHOD_FS_READ_DIRECTORY,
            Self::FsRemove => METHOD_FS_REMOVE,
            Self::FsCopy => METHOD_FS_COPY,
            Self::FsWatch => METHOD_FS_WATCH,
            Self::FsUnwatch => METHOD_FS_UNWATCH,
            Self::ProcessSpawn => METHOD_PROCESS_SPAWN,
            Self::ProcessWriteStdin => METHOD_PROCESS_WRITE_STDIN,
            Self::ProcessResizePty => METHOD_PROCESS_RESIZE_PTY,
            Self::ProcessKill => METHOD_PROCESS_KILL,
            Self::CommandExec => METHOD_COMMAND_EXEC,
            Self::CommandExecWrite => METHOD_COMMAND_EXEC_WRITE,
            Self::CommandExecResize => METHOD_COMMAND_EXEC_RESIZE,
            Self::CommandExecTerminate => METHOD_COMMAND_EXEC_TERMINATE,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            METHOD_THREAD_START => Some(Self::ThreadStart),
            METHOD_THREAD_FORK => Some(Self::ThreadFork),
            METHOD_THREAD_RESUME => Some(Self::ThreadResume),
            METHOD_THREAD_READ => Some(Self::ThreadRead),
            METHOD_THREAD_LIST => Some(Self::ThreadList),
            METHOD_THREAD_SECTION_MOVE => Some(Self::ThreadSectionMove),
            METHOD_THREAD_SECTION_LIST => Some(Self::ThreadSectionList),
            METHOD_THREAD_SECTION_CREATE => Some(Self::ThreadSectionCreate),
            METHOD_THREAD_SECTION_UPDATE => Some(Self::ThreadSectionUpdate),
            METHOD_THREAD_SECTION_DELETE => Some(Self::ThreadSectionDelete),
            METHOD_THREAD_LOADED_LIST => Some(Self::ThreadLoadedList),
            METHOD_THREAD_UNSUBSCRIBE => Some(Self::ThreadUnsubscribe),
            METHOD_THREAD_INCREMENT_ELICITATION => Some(Self::ThreadIncrementElicitation),
            METHOD_THREAD_DECREMENT_ELICITATION => Some(Self::ThreadDecrementElicitation),
            METHOD_THREAD_ARCHIVE => Some(Self::ThreadArchive),
            METHOD_THREAD_DELETE => Some(Self::ThreadDelete),
            METHOD_THREAD_UNARCHIVE => Some(Self::ThreadUnarchive),
            METHOD_THREAD_NAME_SET => Some(Self::ThreadSetName),
            METHOD_THREAD_METADATA_UPDATE => Some(Self::ThreadMetadataUpdate),
            METHOD_THREAD_COMPACT_START => Some(Self::ThreadCompactStart),
            METHOD_THREAD_TURNS_LIST => Some(Self::ThreadTurnsList),
            METHOD_THREAD_ITEMS_LIST => Some(Self::ThreadItemsList),
            METHOD_THREAD_INJECT_ITEMS => Some(Self::ThreadInjectItems),
            METHOD_THREAD_SEARCH => Some(Self::ThreadSearch),
            METHOD_THREAD_SEARCH_OCCURRENCES => Some(Self::ThreadSearchOccurrences),
            METHOD_THREAD_SETTINGS_UPDATE => Some(Self::ThreadSettingsUpdate),
            METHOD_THREAD_MEMORY_MODE_SET => Some(Self::ThreadMemoryModeSet),
            METHOD_MEMORY_RESET => Some(Self::MemoryReset),
            METHOD_THREAD_SHELL_COMMAND => Some(Self::ThreadShellCommand),
            METHOD_THREAD_APPROVE_GUARDIAN_DENIED_ACTION => {
                Some(Self::ThreadApproveGuardianDeniedAction)
            }
            METHOD_THREAD_BACKGROUND_TERMINALS_CLEAN => Some(Self::ThreadBackgroundTerminalsClean),
            METHOD_THREAD_BACKGROUND_TERMINALS_LIST => Some(Self::ThreadBackgroundTerminalsList),
            METHOD_THREAD_BACKGROUND_TERMINALS_TERMINATE => {
                Some(Self::ThreadBackgroundTerminalsTerminate)
            }
            METHOD_THREAD_GOAL_SET => Some(Self::ThreadGoalSet),
            METHOD_THREAD_GOAL_GET => Some(Self::ThreadGoalGet),
            METHOD_THREAD_GOAL_CLEAR => Some(Self::ThreadGoalClear),
            METHOD_ARTIFACT_WRITE => Some(Self::ArtifactWrite),
            METHOD_MEDIA_READ => Some(Self::MediaRead),
            METHOD_MCP_SERVER_RESOURCE_READ => Some(Self::McpServerResourceRead),
            METHOD_MCP_SERVER_TOOL_CALL => Some(Self::McpServerToolCall),
            METHOD_MODEL_LIST => Some(Self::ModelList),
            METHOD_APP_READ => Some(Self::AppRead),
            METHOD_APP_LIST => Some(Self::AppList),
            METHOD_APP_INSTALLED => Some(Self::AppInstalled),
            METHOD_HOOKS_LIST => Some(Self::HooksList),
            METHOD_SKILLS_LIST => Some(Self::SkillsList),
            METHOD_SKILLS_EXTRA_ROOTS_SET => Some(Self::SkillsExtraRootsSet),
            METHOD_SKILLS_CONFIG_WRITE => Some(Self::SkillsConfigWrite),
            METHOD_PLUGIN_LIST => Some(Self::PluginList),
            METHOD_PLUGIN_SEARCH => Some(Self::PluginSearch),
            METHOD_PLUGIN_READ => Some(Self::PluginRead),
            METHOD_PLUGIN_INSTALL => Some(Self::PluginInstall),
            METHOD_PLUGIN_UNINSTALL => Some(Self::PluginUninstall),
            METHOD_PLUGIN_INSTALLED => Some(Self::PluginInstalled),
            METHOD_PLUGIN_ENABLED_SET => Some(Self::PluginEnabledSet),
            METHOD_TURN_START => Some(Self::TurnStart),
            METHOD_TURN_STEER => Some(Self::TurnSteer),
            METHOD_TURN_INTERRUPT => Some(Self::TurnInterrupt),
            METHOD_REVIEW_START => Some(Self::ReviewStart),
            METHOD_FS_READ_FILE => Some(Self::FsReadFile),
            METHOD_FS_WRITE_FILE => Some(Self::FsWriteFile),
            METHOD_FS_CREATE_DIRECTORY => Some(Self::FsCreateDirectory),
            METHOD_FS_GET_METADATA => Some(Self::FsGetMetadata),
            METHOD_FS_READ_DIRECTORY => Some(Self::FsReadDirectory),
            METHOD_FS_REMOVE => Some(Self::FsRemove),
            METHOD_FS_COPY => Some(Self::FsCopy),
            METHOD_FS_WATCH => Some(Self::FsWatch),
            METHOD_FS_UNWATCH => Some(Self::FsUnwatch),
            METHOD_PROCESS_SPAWN => Some(Self::ProcessSpawn),
            METHOD_PROCESS_WRITE_STDIN => Some(Self::ProcessWriteStdin),
            METHOD_PROCESS_RESIZE_PTY => Some(Self::ProcessResizePty),
            METHOD_PROCESS_KILL => Some(Self::ProcessKill),
            METHOD_COMMAND_EXEC => Some(Self::CommandExec),
            METHOD_COMMAND_EXEC_WRITE => Some(Self::CommandExecWrite),
            METHOD_COMMAND_EXEC_RESIZE => Some(Self::CommandExecResize),
            METHOD_COMMAND_EXEC_TERMINATE => Some(Self::CommandExecTerminate),
            _ => None,
        }
    }
}

pub const METHODS: &[&str] = &[
    METHOD_THREAD_START,
    METHOD_THREAD_FORK,
    METHOD_THREAD_RESUME,
    METHOD_THREAD_READ,
    METHOD_THREAD_LIST,
    METHOD_THREAD_SECTION_MOVE,
    METHOD_THREAD_SECTION_LIST,
    METHOD_THREAD_SECTION_CREATE,
    METHOD_THREAD_SECTION_UPDATE,
    METHOD_THREAD_SECTION_DELETE,
    METHOD_THREAD_LOADED_LIST,
    METHOD_THREAD_UNSUBSCRIBE,
    METHOD_THREAD_INCREMENT_ELICITATION,
    METHOD_THREAD_DECREMENT_ELICITATION,
    METHOD_THREAD_ARCHIVE,
    METHOD_THREAD_DELETE,
    METHOD_THREAD_UNARCHIVE,
    METHOD_THREAD_NAME_SET,
    METHOD_THREAD_METADATA_UPDATE,
    METHOD_THREAD_COMPACT_START,
    METHOD_THREAD_TURNS_LIST,
    METHOD_THREAD_ITEMS_LIST,
    METHOD_THREAD_INJECT_ITEMS,
    METHOD_THREAD_SEARCH,
    METHOD_THREAD_SEARCH_OCCURRENCES,
    METHOD_THREAD_SETTINGS_UPDATE,
    METHOD_THREAD_MEMORY_MODE_SET,
    METHOD_MEMORY_RESET,
    METHOD_THREAD_SHELL_COMMAND,
    METHOD_THREAD_APPROVE_GUARDIAN_DENIED_ACTION,
    METHOD_THREAD_BACKGROUND_TERMINALS_CLEAN,
    METHOD_THREAD_BACKGROUND_TERMINALS_LIST,
    METHOD_THREAD_BACKGROUND_TERMINALS_TERMINATE,
    METHOD_THREAD_GOAL_SET,
    METHOD_THREAD_GOAL_GET,
    METHOD_THREAD_GOAL_CLEAR,
    METHOD_ARTIFACT_WRITE,
    METHOD_MEDIA_READ,
    METHOD_MCP_SERVER_RESOURCE_READ,
    METHOD_MCP_SERVER_TOOL_CALL,
    METHOD_MODEL_LIST,
    METHOD_APP_READ,
    METHOD_APP_LIST,
    METHOD_APP_INSTALLED,
    METHOD_HOOKS_LIST,
    METHOD_SKILLS_LIST,
    METHOD_SKILLS_EXTRA_ROOTS_SET,
    METHOD_SKILLS_CONFIG_WRITE,
    METHOD_PLUGIN_LIST,
    METHOD_PLUGIN_SEARCH,
    METHOD_PLUGIN_READ,
    METHOD_PLUGIN_INSTALL,
    METHOD_PLUGIN_UNINSTALL,
    METHOD_PLUGIN_INSTALLED,
    METHOD_PLUGIN_ENABLED_SET,
    METHOD_TURN_START,
    METHOD_TURN_STEER,
    METHOD_TURN_INTERRUPT,
    METHOD_REVIEW_START,
    METHOD_FS_READ_FILE,
    METHOD_FS_WRITE_FILE,
    METHOD_FS_CREATE_DIRECTORY,
    METHOD_FS_GET_METADATA,
    METHOD_FS_READ_DIRECTORY,
    METHOD_FS_REMOVE,
    METHOD_FS_COPY,
    METHOD_FS_WATCH,
    METHOD_FS_UNWATCH,
    METHOD_PROCESS_SPAWN,
    METHOD_PROCESS_WRITE_STDIN,
    METHOD_PROCESS_RESIZE_PTY,
    METHOD_PROCESS_KILL,
    METHOD_COMMAND_EXEC,
    METHOD_COMMAND_EXEC_WRITE,
    METHOD_COMMAND_EXEC_RESIZE,
    METHOD_COMMAND_EXEC_TERMINATE,
];

pub const NOTIFICATION_METHODS: &[&str] = &[
    METHOD_CONFIG_WARNING,
    METHOD_WARNING,
    METHOD_GUARDIAN_WARNING,
    METHOD_ERROR,
    METHOD_SKILLS_CHANGED,
    METHOD_MCP_SERVER_OAUTH_LOGIN_COMPLETED,
    METHOD_MCP_SERVER_STARTUP_STATUS_UPDATED,
    METHOD_HOOK_STARTED,
    METHOD_HOOK_COMPLETED,
    METHOD_THREAD_STARTED,
    METHOD_THREAD_ARCHIVED,
    METHOD_THREAD_DELETED,
    METHOD_THREAD_UNARCHIVED,
    METHOD_THREAD_CLOSED,
    METHOD_THREAD_NAME_UPDATED,
    METHOD_THREAD_STATUS_CHANGED,
    METHOD_TURN_STARTED,
    METHOD_TURN_COMPLETED,
    METHOD_TURN_DIFF_UPDATED,
    METHOD_TURN_PLAN_UPDATED,
    METHOD_ITEM_STARTED,
    METHOD_ITEM_COMPLETED,
    METHOD_ITEM_AUTO_APPROVAL_REVIEW_STARTED,
    METHOD_ITEM_AUTO_APPROVAL_REVIEW_COMPLETED,
    METHOD_AGENT_MESSAGE_DELTA,
    METHOD_COMMAND_EXECUTION_OUTPUT_DELTA,
    METHOD_COMMAND_EXECUTION_TERMINAL_INTERACTION,
    METHOD_FILE_CHANGE_PATCH_UPDATED,
    METHOD_PLAN_DELTA,
    METHOD_MCP_TOOL_CALL_PROGRESS,
    METHOD_REASONING_SUMMARY_TEXT_DELTA,
    METHOD_REASONING_SUMMARY_PART_ADDED,
    METHOD_REASONING_TEXT_DELTA,
    METHOD_MODEL_REROUTED,
    METHOD_MODEL_LIST_UPDATED,
    METHOD_APP_LIST_UPDATED,
    METHOD_MODEL_VERIFICATION,
    METHOD_TURN_MODERATION_METADATA,
    METHOD_MODEL_SAFETY_BUFFERING_UPDATED,
    METHOD_THREAD_SETTINGS_UPDATED,
    METHOD_THREAD_TOKEN_USAGE_UPDATED,
    METHOD_THREAD_GOAL_UPDATED,
    METHOD_THREAD_GOAL_CLEARED,
    METHOD_SERVER_REQUEST_RESOLVED,
    METHOD_PROCESS_OUTPUT_DELTA,
    METHOD_PROCESS_EXITED,
    METHOD_COMMAND_EXEC_OUTPUT_DELTA,
    METHOD_FS_CHANGED,
];

pub const SERVER_REQUEST_METHODS: &[&str] = &[
    METHOD_CURRENT_TIME_READ,
    METHOD_MCP_SERVER_ELICITATION_REQUEST,
    METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
    METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
    METHOD_ITEM_PERMISSIONS_REQUEST_APPROVAL,
    METHOD_ITEM_TOOL_CALL,
    METHOD_ITEM_TOOL_REQUEST_USER_INPUT,
];
