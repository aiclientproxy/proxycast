use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const METHOD_THREAD_START: &str = "thread/start";
pub const METHOD_THREAD_FORK: &str = "thread/fork";
pub const METHOD_THREAD_RESUME: &str = "thread/resume";
pub const METHOD_THREAD_READ: &str = "thread/read";
pub const METHOD_THREAD_LIST: &str = "thread/list";
pub const METHOD_THREAD_ARCHIVE: &str = "thread/archive";
pub const METHOD_THREAD_DELETE: &str = "thread/delete";
pub const METHOD_THREAD_UNARCHIVE: &str = "thread/unarchive";
pub const METHOD_THREAD_TURNS_LIST: &str = "thread/turns/list";
pub const METHOD_THREAD_ITEMS_LIST: &str = "thread/items/list";
pub const METHOD_THREAD_SETTINGS_UPDATE: &str = "thread/settings/update";
pub const METHOD_THREAD_MEMORY_MODE_SET: &str = "thread/memoryMode/set";
pub const METHOD_THREAD_SHELL_COMMAND: &str = "thread/shellCommand";
pub const METHOD_THREAD_GOAL_SET: &str = "thread/goal/set";
pub const METHOD_THREAD_GOAL_GET: &str = "thread/goal/get";
pub const METHOD_THREAD_GOAL_CLEAR: &str = "thread/goal/clear";
pub const METHOD_ARTIFACT_WRITE: &str = "artifact/write";
pub const METHOD_TURN_START: &str = "turn/start";
pub const METHOD_TURN_STEER: &str = "turn/steer";
pub const METHOD_TURN_INTERRUPT: &str = "turn/interrupt";
pub const METHOD_THREAD_STARTED: &str = "thread/started";
pub const METHOD_THREAD_ARCHIVED: &str = "thread/archived";
pub const METHOD_THREAD_DELETED: &str = "thread/deleted";
pub const METHOD_THREAD_UNARCHIVED: &str = "thread/unarchived";
pub const METHOD_TURN_STARTED: &str = "turn/started";
pub const METHOD_TURN_COMPLETED: &str = "turn/completed";
pub const METHOD_ITEM_STARTED: &str = "item/started";
pub const METHOD_ITEM_COMPLETED: &str = "item/completed";
pub const METHOD_AGENT_MESSAGE_DELTA: &str = "item/agentMessage/delta";
pub const METHOD_COMMAND_EXECUTION_OUTPUT_DELTA: &str = "item/commandExecution/outputDelta";
pub const METHOD_FILE_CHANGE_PATCH_UPDATED: &str = "item/fileChange/patchUpdated";
pub const METHOD_PLAN_DELTA: &str = "item/plan/delta";
pub const METHOD_MCP_TOOL_CALL_PROGRESS: &str = "item/mcpToolCall/progress";
pub const METHOD_REASONING_SUMMARY_TEXT_DELTA: &str = "item/reasoning/summaryTextDelta";
pub const METHOD_REASONING_SUMMARY_PART_ADDED: &str = "item/reasoning/summaryPartAdded";
pub const METHOD_REASONING_TEXT_DELTA: &str = "item/reasoning/textDelta";
pub const METHOD_THREAD_SETTINGS_UPDATED: &str = "thread/settings/updated";
pub const METHOD_THREAD_TOKEN_USAGE_UPDATED: &str = "thread/tokenUsage/updated";
pub const METHOD_THREAD_GOAL_UPDATED: &str = "thread/goal/updated";
pub const METHOD_THREAD_GOAL_CLEARED: &str = "thread/goal/cleared";
pub const METHOD_SERVER_REQUEST_RESOLVED: &str = "serverRequest/resolved";
pub const METHOD_MCP_SERVER_ELICITATION_REQUEST: &str = "mcpServer/elicitation/request";
pub const METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL: &str =
    "item/commandExecution/requestApproval";
pub const METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL: &str = "item/fileChange/requestApproval";
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
    #[serde(rename = "thread/archive")]
    ThreadArchive,
    #[serde(rename = "thread/delete")]
    ThreadDelete,
    #[serde(rename = "thread/unarchive")]
    ThreadUnarchive,
    #[serde(rename = "thread/turns/list")]
    ThreadTurnsList,
    #[serde(rename = "thread/items/list")]
    ThreadItemsList,
    #[serde(rename = "thread/settings/update")]
    ThreadSettingsUpdate,
    #[serde(rename = "thread/memoryMode/set")]
    ThreadMemoryModeSet,
    #[serde(rename = "thread/shellCommand")]
    ThreadShellCommand,
    #[serde(rename = "thread/goal/set")]
    ThreadGoalSet,
    #[serde(rename = "thread/goal/get")]
    ThreadGoalGet,
    #[serde(rename = "thread/goal/clear")]
    ThreadGoalClear,
    #[serde(rename = "artifact/write")]
    ArtifactWrite,
    #[serde(rename = "turn/start")]
    TurnStart,
    #[serde(rename = "turn/steer")]
    TurnSteer,
    #[serde(rename = "turn/interrupt")]
    TurnInterrupt,
}

impl Method {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ThreadStart => METHOD_THREAD_START,
            Self::ThreadFork => METHOD_THREAD_FORK,
            Self::ThreadResume => METHOD_THREAD_RESUME,
            Self::ThreadRead => METHOD_THREAD_READ,
            Self::ThreadList => METHOD_THREAD_LIST,
            Self::ThreadArchive => METHOD_THREAD_ARCHIVE,
            Self::ThreadDelete => METHOD_THREAD_DELETE,
            Self::ThreadUnarchive => METHOD_THREAD_UNARCHIVE,
            Self::ThreadTurnsList => METHOD_THREAD_TURNS_LIST,
            Self::ThreadItemsList => METHOD_THREAD_ITEMS_LIST,
            Self::ThreadSettingsUpdate => METHOD_THREAD_SETTINGS_UPDATE,
            Self::ThreadMemoryModeSet => METHOD_THREAD_MEMORY_MODE_SET,
            Self::ThreadShellCommand => METHOD_THREAD_SHELL_COMMAND,
            Self::ThreadGoalSet => METHOD_THREAD_GOAL_SET,
            Self::ThreadGoalGet => METHOD_THREAD_GOAL_GET,
            Self::ThreadGoalClear => METHOD_THREAD_GOAL_CLEAR,
            Self::ArtifactWrite => METHOD_ARTIFACT_WRITE,
            Self::TurnStart => METHOD_TURN_START,
            Self::TurnSteer => METHOD_TURN_STEER,
            Self::TurnInterrupt => METHOD_TURN_INTERRUPT,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            METHOD_THREAD_START => Some(Self::ThreadStart),
            METHOD_THREAD_FORK => Some(Self::ThreadFork),
            METHOD_THREAD_RESUME => Some(Self::ThreadResume),
            METHOD_THREAD_READ => Some(Self::ThreadRead),
            METHOD_THREAD_LIST => Some(Self::ThreadList),
            METHOD_THREAD_ARCHIVE => Some(Self::ThreadArchive),
            METHOD_THREAD_DELETE => Some(Self::ThreadDelete),
            METHOD_THREAD_UNARCHIVE => Some(Self::ThreadUnarchive),
            METHOD_THREAD_TURNS_LIST => Some(Self::ThreadTurnsList),
            METHOD_THREAD_ITEMS_LIST => Some(Self::ThreadItemsList),
            METHOD_THREAD_SETTINGS_UPDATE => Some(Self::ThreadSettingsUpdate),
            METHOD_THREAD_MEMORY_MODE_SET => Some(Self::ThreadMemoryModeSet),
            METHOD_THREAD_SHELL_COMMAND => Some(Self::ThreadShellCommand),
            METHOD_THREAD_GOAL_SET => Some(Self::ThreadGoalSet),
            METHOD_THREAD_GOAL_GET => Some(Self::ThreadGoalGet),
            METHOD_THREAD_GOAL_CLEAR => Some(Self::ThreadGoalClear),
            METHOD_ARTIFACT_WRITE => Some(Self::ArtifactWrite),
            METHOD_TURN_START => Some(Self::TurnStart),
            METHOD_TURN_STEER => Some(Self::TurnSteer),
            METHOD_TURN_INTERRUPT => Some(Self::TurnInterrupt),
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
    METHOD_THREAD_ARCHIVE,
    METHOD_THREAD_DELETE,
    METHOD_THREAD_UNARCHIVE,
    METHOD_THREAD_TURNS_LIST,
    METHOD_THREAD_ITEMS_LIST,
    METHOD_THREAD_SETTINGS_UPDATE,
    METHOD_THREAD_MEMORY_MODE_SET,
    METHOD_THREAD_SHELL_COMMAND,
    METHOD_THREAD_GOAL_SET,
    METHOD_THREAD_GOAL_GET,
    METHOD_THREAD_GOAL_CLEAR,
    METHOD_ARTIFACT_WRITE,
    METHOD_TURN_START,
    METHOD_TURN_STEER,
    METHOD_TURN_INTERRUPT,
];

pub const NOTIFICATION_METHODS: &[&str] = &[
    METHOD_THREAD_STARTED,
    METHOD_THREAD_ARCHIVED,
    METHOD_THREAD_DELETED,
    METHOD_THREAD_UNARCHIVED,
    METHOD_TURN_STARTED,
    METHOD_TURN_COMPLETED,
    METHOD_ITEM_STARTED,
    METHOD_ITEM_COMPLETED,
    METHOD_AGENT_MESSAGE_DELTA,
    METHOD_COMMAND_EXECUTION_OUTPUT_DELTA,
    METHOD_FILE_CHANGE_PATCH_UPDATED,
    METHOD_PLAN_DELTA,
    METHOD_MCP_TOOL_CALL_PROGRESS,
    METHOD_REASONING_SUMMARY_TEXT_DELTA,
    METHOD_REASONING_SUMMARY_PART_ADDED,
    METHOD_REASONING_TEXT_DELTA,
    METHOD_THREAD_SETTINGS_UPDATED,
    METHOD_THREAD_TOKEN_USAGE_UPDATED,
    METHOD_THREAD_GOAL_UPDATED,
    METHOD_THREAD_GOAL_CLEARED,
    METHOD_SERVER_REQUEST_RESOLVED,
];

pub const SERVER_REQUEST_METHODS: &[&str] = &[
    METHOD_MCP_SERVER_ELICITATION_REQUEST,
    METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
    METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
    METHOD_ITEM_TOOL_REQUEST_USER_INPUT,
];
