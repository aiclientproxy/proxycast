use super::{
    AgentMessageDeltaNotification, AppListUpdatedNotification, AppsInstalledParams,
    AppsInstalledResponse, AppsListParams, AppsListResponse, AppsReadParams, AppsReadResponse,
    ArtifactWriteParams, ArtifactWriteResponse, CommandExecOutputDeltaNotification,
    CommandExecParams, CommandExecResizeParams, CommandExecResizeResponse, CommandExecResponse,
    CommandExecTerminateParams, CommandExecTerminateResponse, CommandExecWriteParams,
    CommandExecWriteResponse, CommandExecutionOutputDeltaNotification,
    CommandExecutionRequestApprovalParams, CommandExecutionTerminalInteractionNotification,
    ConfigWarningNotification, CurrentTimeReadParams, DynamicToolCallParams, ErrorNotification,
    FileChangePatchUpdatedNotification, FileChangeRequestApprovalParams, FsChangedNotification,
    FsCopyParams, FsCopyResponse, FsCreateDirectoryParams, FsCreateDirectoryResponse,
    FsGetMetadataParams, FsGetMetadataResponse, FsReadDirectoryParams, FsReadDirectoryResponse,
    FsReadFileParams, FsReadFileResponse, FsRemoveParams, FsRemoveResponse, FsUnwatchParams,
    FsUnwatchResponse, FsWatchParams, FsWatchResponse, FsWriteFileParams, FsWriteFileResponse,
    GuardianWarningNotification, HookCompletedNotification, HookStartedNotification,
    HooksListParams, HooksListResponse, ItemCompletedNotification,
    ItemGuardianApprovalReviewCompletedNotification, ItemGuardianApprovalReviewStartedNotification,
    ItemStartedNotification, McpServerElicitationRequestParams,
    McpServerOauthLoginCompletedNotification, McpServerResourceReadParams,
    McpServerResourceReadResponse, McpServerStatusUpdatedNotification, McpServerToolCallParams,
    McpServerToolCallResponse, McpToolCallProgressNotification, MediaReadParams, MediaReadResponse,
    MemoryResetResponse, Method, ModelListParams, ModelListUpdatedNotification,
    ModelReroutedNotification, ModelSafetyBufferingUpdatedNotification,
    ModelVerificationNotification, PermissionsRequestApprovalParams, PlanDeltaNotification,
    PluginCatalogEnabledSetParams, PluginCatalogInstallParams, PluginCatalogInstalledParams,
    PluginCatalogListParams, PluginCatalogReadParams, PluginCatalogUninstallParams,
    PluginSearchParams, PluginSearchResponse, ProcessExitedNotification, ProcessKillParams,
    ProcessKillResponse, ProcessOutputDeltaNotification, ProcessResizePtyParams,
    ProcessResizePtyResponse, ProcessSpawnParams, ProcessSpawnResponse, ProcessWriteStdinParams,
    ProcessWriteStdinResponse, ReasoningSummaryPartAddedNotification,
    ReasoningSummaryTextDeltaNotification, ReasoningTextDeltaNotification, ReviewStartParams,
    ReviewStartResponse, ServerRequestResolvedNotification, SkillsChangedNotification,
    SkillsConfigWriteParams, SkillsConfigWriteResponse, SkillsExtraRootsSetParams,
    SkillsExtraRootsSetResponse, SkillsListParams, SkillsListResponse,
    ThreadApproveGuardianDeniedActionParams, ThreadApproveGuardianDeniedActionResponse,
    ThreadArchiveParams, ThreadArchiveResponse, ThreadArchivedNotification,
    ThreadBackgroundTerminalsCleanParams, ThreadBackgroundTerminalsCleanResponse,
    ThreadBackgroundTerminalsListParams, ThreadBackgroundTerminalsListResponse,
    ThreadBackgroundTerminalsTerminateParams, ThreadBackgroundTerminalsTerminateResponse,
    ThreadClosedNotification, ThreadCompactStartParams, ThreadCompactStartResponse,
    ThreadDecrementElicitationParams, ThreadDecrementElicitationResponse, ThreadDeleteParams,
    ThreadDeleteResponse, ThreadDeletedNotification, ThreadForkParams, ThreadForkResponse,
    ThreadGoalClearParams, ThreadGoalClearResponse, ThreadGoalClearedNotification,
    ThreadGoalGetParams, ThreadGoalGetResponse, ThreadGoalSetParams, ThreadGoalSetResponse,
    ThreadGoalUpdatedNotification, ThreadIncrementElicitationParams,
    ThreadIncrementElicitationResponse, ThreadInjectItemsParams, ThreadInjectItemsResponse,
    ThreadItemsListParams, ThreadItemsListResponse, ThreadListParams, ThreadListResponse,
    ThreadLoadedListParams, ThreadLoadedListResponse, ThreadMemoryModeSetParams,
    ThreadMemoryModeSetResponse, ThreadMetadataUpdateParams, ThreadMetadataUpdateResponse,
    ThreadNameUpdatedNotification, ThreadReadParams, ThreadReadResponse, ThreadResumeParams,
    ThreadResumeResponse, ThreadSearchOccurrencesParams, ThreadSearchOccurrencesResponse,
    ThreadSearchParams, ThreadSearchResponse, ThreadSectionCreateParams,
    ThreadSectionCreateResponse, ThreadSectionDeleteParams, ThreadSectionDeleteResponse,
    ThreadSectionListParams, ThreadSectionListResponse, ThreadSectionMoveParams,
    ThreadSectionMoveResponse, ThreadSectionUpdateParams, ThreadSectionUpdateResponse,
    ThreadSetNameParams, ThreadSetNameResponse, ThreadSettingsUpdateParams,
    ThreadSettingsUpdateResponse, ThreadSettingsUpdatedNotification, ThreadShellCommandParams,
    ThreadShellCommandResponse, ThreadStartParams, ThreadStartResponse, ThreadStartedNotification,
    ThreadStatusChangedNotification, ThreadTokenUsageUpdatedNotification, ThreadTurnsListParams,
    ThreadTurnsListResponse, ThreadUnarchiveParams, ThreadUnarchiveResponse,
    ThreadUnarchivedNotification, ThreadUnsubscribeParams, ThreadUnsubscribeResponse,
    ToolRequestUserInputParams, TurnCompletedNotification, TurnDiffUpdatedNotification,
    TurnInterruptParams, TurnInterruptResponse, TurnModerationMetadataNotification,
    TurnPlanUpdatedNotification, TurnStartParams, TurnStartResponse, TurnStartedNotification,
    TurnSteerParams, TurnSteerResponse, WarningNotification, METHOD_APP_LIST_UPDATED,
    METHOD_COMMAND_EXECUTION_OUTPUT_DELTA, METHOD_COMMAND_EXECUTION_TERMINAL_INTERACTION,
    METHOD_COMMAND_EXEC_OUTPUT_DELTA, METHOD_CONFIG_WARNING, METHOD_CURRENT_TIME_READ,
    METHOD_ERROR, METHOD_FILE_CHANGE_PATCH_UPDATED, METHOD_FS_CHANGED, METHOD_GUARDIAN_WARNING,
    METHOD_HOOK_COMPLETED, METHOD_HOOK_STARTED, METHOD_ITEM_AUTO_APPROVAL_REVIEW_COMPLETED,
    METHOD_ITEM_AUTO_APPROVAL_REVIEW_STARTED, METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
    METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL, METHOD_ITEM_PERMISSIONS_REQUEST_APPROVAL,
    METHOD_ITEM_TOOL_CALL, METHOD_ITEM_TOOL_REQUEST_USER_INPUT,
    METHOD_MCP_SERVER_ELICITATION_REQUEST, METHOD_MCP_SERVER_OAUTH_LOGIN_COMPLETED,
    METHOD_MCP_SERVER_STARTUP_STATUS_UPDATED, METHOD_MCP_TOOL_CALL_PROGRESS,
    METHOD_MODEL_LIST_UPDATED, METHOD_MODEL_REROUTED, METHOD_MODEL_SAFETY_BUFFERING_UPDATED,
    METHOD_MODEL_VERIFICATION, METHOD_PLAN_DELTA, METHOD_PROCESS_EXITED,
    METHOD_PROCESS_OUTPUT_DELTA, METHOD_REASONING_SUMMARY_PART_ADDED,
    METHOD_REASONING_SUMMARY_TEXT_DELTA, METHOD_REASONING_TEXT_DELTA,
    METHOD_SERVER_REQUEST_RESOLVED, METHOD_SKILLS_CHANGED, METHOD_THREAD_CLOSED,
    METHOD_THREAD_GOAL_CLEARED, METHOD_THREAD_GOAL_UPDATED, METHOD_THREAD_NAME_UPDATED,
    METHOD_THREAD_STATUS_CHANGED, METHOD_THREAD_TOKEN_USAGE_UPDATED, METHOD_TURN_DIFF_UPDATED,
    METHOD_TURN_MODERATION_METADATA, METHOD_TURN_PLAN_UPDATED, METHOD_WARNING,
};
use crate::{JsonRpcNotification, JsonRpcRequest, RequestId};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Typed v2 envelope names. The central schema registry can adopt this list
/// once the v2 request/notification catalog is wired into the public dispatch.
pub const V2_ENVELOPE_SCHEMA_TYPE_NAMES: &[&str] = &[
    "ClientRequest",
    "ClientResponse",
    "ServerRequest",
    "ServerNotification",
];

/// Requests sent by a v2 client. Unknown methods fail closed during decode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "method")]
pub enum ClientRequest {
    #[serde(rename = "thread/start")]
    ThreadStart {
        id: RequestId,
        params: ThreadStartParams,
    },
    #[serde(rename = "thread/fork")]
    ThreadFork {
        id: RequestId,
        params: ThreadForkParams,
    },
    #[serde(rename = "thread/resume")]
    ThreadResume {
        id: RequestId,
        params: ThreadResumeParams,
    },
    #[serde(rename = "thread/read")]
    ThreadRead {
        id: RequestId,
        params: ThreadReadParams,
    },
    #[serde(rename = "thread/list")]
    ThreadList {
        id: RequestId,
        params: ThreadListParams,
    },
    #[serde(rename = "thread/section/move")]
    ThreadSectionMove {
        id: RequestId,
        params: ThreadSectionMoveParams,
    },
    #[serde(rename = "threadSection/list")]
    ThreadSectionList {
        id: RequestId,
        params: ThreadSectionListParams,
    },
    #[serde(rename = "threadSection/create")]
    ThreadSectionCreate {
        id: RequestId,
        params: ThreadSectionCreateParams,
    },
    #[serde(rename = "threadSection/update")]
    ThreadSectionUpdate {
        id: RequestId,
        params: ThreadSectionUpdateParams,
    },
    #[serde(rename = "threadSection/delete")]
    ThreadSectionDelete {
        id: RequestId,
        params: ThreadSectionDeleteParams,
    },
    #[serde(rename = "thread/loaded/list")]
    ThreadLoadedList {
        id: RequestId,
        params: ThreadLoadedListParams,
    },
    #[serde(rename = "thread/unsubscribe")]
    ThreadUnsubscribe {
        id: RequestId,
        params: ThreadUnsubscribeParams,
    },
    #[serde(rename = "thread/increment_elicitation")]
    ThreadIncrementElicitation {
        id: RequestId,
        params: ThreadIncrementElicitationParams,
    },
    #[serde(rename = "thread/decrement_elicitation")]
    ThreadDecrementElicitation {
        id: RequestId,
        params: ThreadDecrementElicitationParams,
    },
    #[serde(rename = "thread/archive")]
    ThreadArchive {
        id: RequestId,
        params: ThreadArchiveParams,
    },
    #[serde(rename = "thread/delete")]
    ThreadDelete {
        id: RequestId,
        params: ThreadDeleteParams,
    },
    #[serde(rename = "thread/unarchive")]
    ThreadUnarchive {
        id: RequestId,
        params: ThreadUnarchiveParams,
    },
    #[serde(rename = "thread/name/set")]
    ThreadSetName {
        id: RequestId,
        params: ThreadSetNameParams,
    },
    #[serde(rename = "thread/metadata/update")]
    ThreadMetadataUpdate {
        id: RequestId,
        params: ThreadMetadataUpdateParams,
    },
    #[serde(rename = "thread/compact/start")]
    ThreadCompactStart {
        id: RequestId,
        params: ThreadCompactStartParams,
    },
    #[serde(rename = "thread/turns/list")]
    ThreadTurnsList {
        id: RequestId,
        params: ThreadTurnsListParams,
    },
    #[serde(rename = "thread/items/list")]
    ThreadItemsList {
        id: RequestId,
        params: ThreadItemsListParams,
    },
    #[serde(rename = "thread/inject_items")]
    ThreadInjectItems {
        id: RequestId,
        params: ThreadInjectItemsParams,
    },
    #[serde(rename = "thread/search")]
    ThreadSearch {
        id: RequestId,
        params: ThreadSearchParams,
    },
    #[serde(rename = "thread/searchOccurrences")]
    ThreadSearchOccurrences {
        id: RequestId,
        params: ThreadSearchOccurrencesParams,
    },
    #[serde(rename = "thread/settings/update")]
    ThreadSettingsUpdate {
        id: RequestId,
        params: ThreadSettingsUpdateParams,
    },
    #[serde(rename = "thread/memoryMode/set")]
    ThreadMemoryModeSet {
        id: RequestId,
        params: ThreadMemoryModeSetParams,
    },
    #[serde(rename = "memory/reset")]
    MemoryReset { id: RequestId },
    #[serde(rename = "thread/shellCommand")]
    ThreadShellCommand {
        id: RequestId,
        params: ThreadShellCommandParams,
    },
    #[serde(rename = "thread/approveGuardianDeniedAction")]
    ThreadApproveGuardianDeniedAction {
        id: RequestId,
        params: ThreadApproveGuardianDeniedActionParams,
    },
    #[serde(rename = "thread/backgroundTerminals/clean")]
    ThreadBackgroundTerminalsClean {
        id: RequestId,
        params: ThreadBackgroundTerminalsCleanParams,
    },
    #[serde(rename = "thread/backgroundTerminals/list")]
    ThreadBackgroundTerminalsList {
        id: RequestId,
        params: ThreadBackgroundTerminalsListParams,
    },
    #[serde(rename = "thread/backgroundTerminals/terminate")]
    ThreadBackgroundTerminalsTerminate {
        id: RequestId,
        params: ThreadBackgroundTerminalsTerminateParams,
    },
    #[serde(rename = "thread/goal/set")]
    ThreadGoalSet {
        id: RequestId,
        params: ThreadGoalSetParams,
    },
    #[serde(rename = "thread/goal/get")]
    ThreadGoalGet {
        id: RequestId,
        params: ThreadGoalGetParams,
    },
    #[serde(rename = "thread/goal/clear")]
    ThreadGoalClear {
        id: RequestId,
        params: ThreadGoalClearParams,
    },
    #[serde(rename = "artifact/write")]
    ArtifactWrite {
        id: RequestId,
        params: ArtifactWriteParams,
    },
    #[serde(rename = "media/read")]
    MediaRead {
        id: RequestId,
        params: MediaReadParams,
    },
    #[serde(rename = "mcpServer/resource/read")]
    McpServerResourceRead {
        id: RequestId,
        params: McpServerResourceReadParams,
    },
    #[serde(rename = "mcpServer/tool/call")]
    McpServerToolCall {
        id: RequestId,
        params: McpServerToolCallParams,
    },
    #[serde(rename = "model/list")]
    ModelList {
        id: RequestId,
        params: ModelListParams,
    },
    #[serde(rename = "app/read")]
    AppRead {
        id: RequestId,
        params: AppsReadParams,
    },
    #[serde(rename = "app/list")]
    AppList {
        id: RequestId,
        params: AppsListParams,
    },
    #[serde(rename = "app/installed")]
    AppInstalled {
        id: RequestId,
        params: AppsInstalledParams,
    },
    #[serde(rename = "hooks/list")]
    HooksList {
        id: RequestId,
        params: HooksListParams,
    },
    #[serde(rename = "skills/list")]
    SkillsList {
        id: RequestId,
        params: SkillsListParams,
    },
    #[serde(rename = "skills/extraRoots/set")]
    SkillsExtraRootsSet {
        id: RequestId,
        params: SkillsExtraRootsSetParams,
    },
    #[serde(rename = "skills/config/write")]
    SkillsConfigWrite {
        id: RequestId,
        params: SkillsConfigWriteParams,
    },
    #[serde(rename = "plugin/list")]
    PluginList {
        id: RequestId,
        params: PluginCatalogListParams,
    },
    #[serde(rename = "plugin/search")]
    PluginSearch {
        id: RequestId,
        params: PluginSearchParams,
    },
    #[serde(rename = "plugin/read")]
    PluginRead {
        id: RequestId,
        params: PluginCatalogReadParams,
    },
    #[serde(rename = "plugin/install")]
    PluginInstall {
        id: RequestId,
        params: PluginCatalogInstallParams,
    },
    #[serde(rename = "plugin/uninstall")]
    PluginUninstall {
        id: RequestId,
        params: PluginCatalogUninstallParams,
    },
    #[serde(rename = "plugin/installed")]
    PluginInstalled {
        id: RequestId,
        params: PluginCatalogInstalledParams,
    },
    #[serde(rename = "plugin/enabled/set")]
    PluginEnabledSet {
        id: RequestId,
        params: PluginCatalogEnabledSetParams,
    },
    #[serde(rename = "turn/start")]
    TurnStart {
        id: RequestId,
        params: TurnStartParams,
    },
    #[serde(rename = "turn/steer")]
    TurnSteer {
        id: RequestId,
        params: TurnSteerParams,
    },
    #[serde(rename = "turn/interrupt")]
    TurnInterrupt {
        id: RequestId,
        params: TurnInterruptParams,
    },
    #[serde(rename = "review/start")]
    ReviewStart {
        id: RequestId,
        params: ReviewStartParams,
    },
    #[serde(rename = "fs/readFile")]
    FsReadFile {
        id: RequestId,
        params: FsReadFileParams,
    },
    #[serde(rename = "fs/writeFile")]
    FsWriteFile {
        id: RequestId,
        params: FsWriteFileParams,
    },
    #[serde(rename = "fs/createDirectory")]
    FsCreateDirectory {
        id: RequestId,
        params: FsCreateDirectoryParams,
    },
    #[serde(rename = "fs/getMetadata")]
    FsGetMetadata {
        id: RequestId,
        params: FsGetMetadataParams,
    },
    #[serde(rename = "fs/readDirectory")]
    FsReadDirectory {
        id: RequestId,
        params: FsReadDirectoryParams,
    },
    #[serde(rename = "fs/remove")]
    FsRemove {
        id: RequestId,
        params: FsRemoveParams,
    },
    #[serde(rename = "fs/copy")]
    FsCopy { id: RequestId, params: FsCopyParams },
    #[serde(rename = "fs/watch")]
    FsWatch {
        id: RequestId,
        params: FsWatchParams,
    },
    #[serde(rename = "fs/unwatch")]
    FsUnwatch {
        id: RequestId,
        params: FsUnwatchParams,
    },
    #[serde(rename = "process/spawn")]
    ProcessSpawn {
        id: RequestId,
        params: ProcessSpawnParams,
    },
    #[serde(rename = "process/writeStdin")]
    ProcessWriteStdin {
        id: RequestId,
        params: ProcessWriteStdinParams,
    },
    #[serde(rename = "process/resizePty")]
    ProcessResizePty {
        id: RequestId,
        params: ProcessResizePtyParams,
    },
    #[serde(rename = "process/kill")]
    ProcessKill {
        id: RequestId,
        params: ProcessKillParams,
    },
    #[serde(rename = "command/exec")]
    CommandExec {
        id: RequestId,
        params: CommandExecParams,
    },
    #[serde(rename = "command/exec/write")]
    CommandExecWrite {
        id: RequestId,
        params: CommandExecWriteParams,
    },
    #[serde(rename = "command/exec/resize")]
    CommandExecResize {
        id: RequestId,
        params: CommandExecResizeParams,
    },
    #[serde(rename = "command/exec/terminate")]
    CommandExecTerminate {
        id: RequestId,
        params: CommandExecTerminateParams,
    },
}

impl ClientRequest {
    pub fn id(&self) -> &RequestId {
        match self {
            Self::ThreadStart { id, .. }
            | Self::ThreadFork { id, .. }
            | Self::ThreadResume { id, .. }
            | Self::ThreadRead { id, .. }
            | Self::ThreadList { id, .. }
            | Self::ThreadSectionMove { id, .. }
            | Self::ThreadSectionList { id, .. }
            | Self::ThreadSectionCreate { id, .. }
            | Self::ThreadSectionUpdate { id, .. }
            | Self::ThreadSectionDelete { id, .. }
            | Self::ThreadLoadedList { id, .. }
            | Self::ThreadUnsubscribe { id, .. }
            | Self::ThreadIncrementElicitation { id, .. }
            | Self::ThreadDecrementElicitation { id, .. }
            | Self::ThreadArchive { id, .. }
            | Self::ThreadDelete { id, .. }
            | Self::ThreadUnarchive { id, .. }
            | Self::ThreadSetName { id, .. }
            | Self::ThreadMetadataUpdate { id, .. }
            | Self::ThreadCompactStart { id, .. }
            | Self::ThreadTurnsList { id, .. }
            | Self::ThreadItemsList { id, .. }
            | Self::ThreadInjectItems { id, .. }
            | Self::ThreadSearch { id, .. }
            | Self::ThreadSearchOccurrences { id, .. }
            | Self::ThreadSettingsUpdate { id, .. }
            | Self::ThreadMemoryModeSet { id, .. }
            | Self::MemoryReset { id }
            | Self::ThreadShellCommand { id, .. }
            | Self::ThreadApproveGuardianDeniedAction { id, .. }
            | Self::ThreadBackgroundTerminalsClean { id, .. }
            | Self::ThreadBackgroundTerminalsList { id, .. }
            | Self::ThreadBackgroundTerminalsTerminate { id, .. }
            | Self::ThreadGoalSet { id, .. }
            | Self::ThreadGoalGet { id, .. }
            | Self::ThreadGoalClear { id, .. }
            | Self::ArtifactWrite { id, .. }
            | Self::MediaRead { id, .. }
            | Self::McpServerResourceRead { id, .. }
            | Self::McpServerToolCall { id, .. }
            | Self::ModelList { id, .. }
            | Self::AppRead { id, .. }
            | Self::AppList { id, .. }
            | Self::AppInstalled { id, .. }
            | Self::HooksList { id, .. }
            | Self::SkillsList { id, .. }
            | Self::SkillsExtraRootsSet { id, .. }
            | Self::SkillsConfigWrite { id, .. }
            | Self::PluginList { id, .. }
            | Self::PluginSearch { id, .. }
            | Self::PluginRead { id, .. }
            | Self::PluginInstall { id, .. }
            | Self::PluginUninstall { id, .. }
            | Self::PluginInstalled { id, .. }
            | Self::PluginEnabledSet { id, .. }
            | Self::TurnStart { id, .. }
            | Self::TurnSteer { id, .. }
            | Self::TurnInterrupt { id, .. }
            | Self::ReviewStart { id, .. }
            | Self::FsReadFile { id, .. }
            | Self::FsWriteFile { id, .. }
            | Self::FsCreateDirectory { id, .. }
            | Self::FsGetMetadata { id, .. }
            | Self::FsReadDirectory { id, .. }
            | Self::FsRemove { id, .. }
            | Self::FsCopy { id, .. }
            | Self::FsWatch { id, .. }
            | Self::FsUnwatch { id, .. }
            | Self::ProcessSpawn { id, .. }
            | Self::ProcessWriteStdin { id, .. }
            | Self::ProcessResizePty { id, .. }
            | Self::ProcessKill { id, .. } => id,
            Self::CommandExec { id, .. }
            | Self::CommandExecWrite { id, .. }
            | Self::CommandExecResize { id, .. }
            | Self::CommandExecTerminate { id, .. } => id,
        }
    }

    pub fn method(&self) -> Method {
        match self {
            Self::ThreadStart { .. } => Method::ThreadStart,
            Self::ThreadFork { .. } => Method::ThreadFork,
            Self::ThreadResume { .. } => Method::ThreadResume,
            Self::ThreadRead { .. } => Method::ThreadRead,
            Self::ThreadList { .. } => Method::ThreadList,
            Self::ThreadSectionMove { .. } => Method::ThreadSectionMove,
            Self::ThreadSectionList { .. } => Method::ThreadSectionList,
            Self::ThreadSectionCreate { .. } => Method::ThreadSectionCreate,
            Self::ThreadSectionUpdate { .. } => Method::ThreadSectionUpdate,
            Self::ThreadSectionDelete { .. } => Method::ThreadSectionDelete,
            Self::ThreadLoadedList { .. } => Method::ThreadLoadedList,
            Self::ThreadUnsubscribe { .. } => Method::ThreadUnsubscribe,
            Self::ThreadIncrementElicitation { .. } => Method::ThreadIncrementElicitation,
            Self::ThreadDecrementElicitation { .. } => Method::ThreadDecrementElicitation,
            Self::ThreadArchive { .. } => Method::ThreadArchive,
            Self::ThreadDelete { .. } => Method::ThreadDelete,
            Self::ThreadUnarchive { .. } => Method::ThreadUnarchive,
            Self::ThreadSetName { .. } => Method::ThreadSetName,
            Self::ThreadMetadataUpdate { .. } => Method::ThreadMetadataUpdate,
            Self::ThreadCompactStart { .. } => Method::ThreadCompactStart,
            Self::ThreadTurnsList { .. } => Method::ThreadTurnsList,
            Self::ThreadItemsList { .. } => Method::ThreadItemsList,
            Self::ThreadInjectItems { .. } => Method::ThreadInjectItems,
            Self::ThreadSearch { .. } => Method::ThreadSearch,
            Self::ThreadSearchOccurrences { .. } => Method::ThreadSearchOccurrences,
            Self::ThreadSettingsUpdate { .. } => Method::ThreadSettingsUpdate,
            Self::ThreadMemoryModeSet { .. } => Method::ThreadMemoryModeSet,
            Self::MemoryReset { .. } => Method::MemoryReset,
            Self::ThreadShellCommand { .. } => Method::ThreadShellCommand,
            Self::ThreadApproveGuardianDeniedAction { .. } => {
                Method::ThreadApproveGuardianDeniedAction
            }
            Self::ThreadBackgroundTerminalsClean { .. } => Method::ThreadBackgroundTerminalsClean,
            Self::ThreadBackgroundTerminalsList { .. } => Method::ThreadBackgroundTerminalsList,
            Self::ThreadBackgroundTerminalsTerminate { .. } => {
                Method::ThreadBackgroundTerminalsTerminate
            }
            Self::ThreadGoalSet { .. } => Method::ThreadGoalSet,
            Self::ThreadGoalGet { .. } => Method::ThreadGoalGet,
            Self::ThreadGoalClear { .. } => Method::ThreadGoalClear,
            Self::ArtifactWrite { .. } => Method::ArtifactWrite,
            Self::MediaRead { .. } => Method::MediaRead,
            Self::McpServerResourceRead { .. } => Method::McpServerResourceRead,
            Self::McpServerToolCall { .. } => Method::McpServerToolCall,
            Self::ModelList { .. } => Method::ModelList,
            Self::AppRead { .. } => Method::AppRead,
            Self::AppList { .. } => Method::AppList,
            Self::AppInstalled { .. } => Method::AppInstalled,
            Self::HooksList { .. } => Method::HooksList,
            Self::SkillsList { .. } => Method::SkillsList,
            Self::SkillsExtraRootsSet { .. } => Method::SkillsExtraRootsSet,
            Self::SkillsConfigWrite { .. } => Method::SkillsConfigWrite,
            Self::PluginList { .. } => Method::PluginList,
            Self::PluginSearch { .. } => Method::PluginSearch,
            Self::PluginRead { .. } => Method::PluginRead,
            Self::PluginInstall { .. } => Method::PluginInstall,
            Self::PluginUninstall { .. } => Method::PluginUninstall,
            Self::PluginInstalled { .. } => Method::PluginInstalled,
            Self::PluginEnabledSet { .. } => Method::PluginEnabledSet,
            Self::TurnStart { .. } => Method::TurnStart,
            Self::TurnSteer { .. } => Method::TurnSteer,
            Self::TurnInterrupt { .. } => Method::TurnInterrupt,
            Self::ReviewStart { .. } => Method::ReviewStart,
            Self::FsReadFile { .. } => Method::FsReadFile,
            Self::FsWriteFile { .. } => Method::FsWriteFile,
            Self::FsCreateDirectory { .. } => Method::FsCreateDirectory,
            Self::FsGetMetadata { .. } => Method::FsGetMetadata,
            Self::FsReadDirectory { .. } => Method::FsReadDirectory,
            Self::FsRemove { .. } => Method::FsRemove,
            Self::FsCopy { .. } => Method::FsCopy,
            Self::FsWatch { .. } => Method::FsWatch,
            Self::FsUnwatch { .. } => Method::FsUnwatch,
            Self::ProcessSpawn { .. } => Method::ProcessSpawn,
            Self::ProcessWriteStdin { .. } => Method::ProcessWriteStdin,
            Self::ProcessResizePty { .. } => Method::ProcessResizePty,
            Self::ProcessKill { .. } => Method::ProcessKill,
            Self::CommandExec { .. } => Method::CommandExec,
            Self::CommandExecWrite { .. } => Method::CommandExecWrite,
            Self::CommandExecResize { .. } => Method::CommandExecResize,
            Self::CommandExecTerminate { .. } => Method::CommandExecTerminate,
        }
    }
}

/// Successful JSON-RPC response. The method is intentionally absent from the
/// wire response; JSON-RPC correlates it with the request id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ClientResponse {
    pub id: RequestId,
    pub result: Value,
}

/// Typed response payloads used by callers that still know the originating
/// request method. They lower into the standard [`ClientResponse`] envelope
/// without leaking a non-standard `method` field onto the wire.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientResponsePayload {
    ThreadStart(ThreadStartResponse),
    ThreadFork(ThreadForkResponse),
    ThreadResume(ThreadResumeResponse),
    ThreadRead(ThreadReadResponse),
    ThreadList(ThreadListResponse),
    ThreadSectionMove(ThreadSectionMoveResponse),
    ThreadSectionList(ThreadSectionListResponse),
    ThreadSectionCreate(ThreadSectionCreateResponse),
    ThreadSectionUpdate(ThreadSectionUpdateResponse),
    ThreadSectionDelete(ThreadSectionDeleteResponse),
    ThreadLoadedList(ThreadLoadedListResponse),
    ThreadUnsubscribe(ThreadUnsubscribeResponse),
    ThreadIncrementElicitation(ThreadIncrementElicitationResponse),
    ThreadDecrementElicitation(ThreadDecrementElicitationResponse),
    ThreadArchive(ThreadArchiveResponse),
    ThreadDelete(ThreadDeleteResponse),
    ThreadUnarchive(ThreadUnarchiveResponse),
    ThreadSetName(ThreadSetNameResponse),
    ThreadMetadataUpdate(ThreadMetadataUpdateResponse),
    ThreadCompactStart(ThreadCompactStartResponse),
    ThreadTurnsList(ThreadTurnsListResponse),
    ThreadItemsList(ThreadItemsListResponse),
    ThreadInjectItems(ThreadInjectItemsResponse),
    ThreadSearch(ThreadSearchResponse),
    ThreadSearchOccurrences(ThreadSearchOccurrencesResponse),
    ThreadSettingsUpdate(ThreadSettingsUpdateResponse),
    ThreadMemoryModeSet(ThreadMemoryModeSetResponse),
    MemoryReset(MemoryResetResponse),
    ThreadShellCommand(ThreadShellCommandResponse),
    ThreadApproveGuardianDeniedAction(ThreadApproveGuardianDeniedActionResponse),
    ThreadBackgroundTerminalsClean(ThreadBackgroundTerminalsCleanResponse),
    ThreadBackgroundTerminalsList(ThreadBackgroundTerminalsListResponse),
    ThreadBackgroundTerminalsTerminate(ThreadBackgroundTerminalsTerminateResponse),
    ThreadGoalSet(ThreadGoalSetResponse),
    ThreadGoalGet(ThreadGoalGetResponse),
    ThreadGoalClear(ThreadGoalClearResponse),
    ArtifactWrite(ArtifactWriteResponse),
    MediaRead(MediaReadResponse),
    McpServerResourceRead(McpServerResourceReadResponse),
    McpServerToolCall(McpServerToolCallResponse),
    AppRead(AppsReadResponse),
    AppList(AppsListResponse),
    AppInstalled(AppsInstalledResponse),
    HooksList(HooksListResponse),
    SkillsList(SkillsListResponse),
    SkillsExtraRootsSet(SkillsExtraRootsSetResponse),
    SkillsConfigWrite(SkillsConfigWriteResponse),
    PluginSearch(PluginSearchResponse),
    TurnStart(TurnStartResponse),
    TurnSteer(TurnSteerResponse),
    TurnInterrupt(TurnInterruptResponse),
    ReviewStart(ReviewStartResponse),
    FsReadFile(FsReadFileResponse),
    FsWriteFile(FsWriteFileResponse),
    FsCreateDirectory(FsCreateDirectoryResponse),
    FsGetMetadata(FsGetMetadataResponse),
    FsReadDirectory(FsReadDirectoryResponse),
    FsRemove(FsRemoveResponse),
    FsCopy(FsCopyResponse),
    FsWatch(FsWatchResponse),
    FsUnwatch(FsUnwatchResponse),
    ProcessSpawn(ProcessSpawnResponse),
    ProcessWriteStdin(ProcessWriteStdinResponse),
    ProcessResizePty(ProcessResizePtyResponse),
    ProcessKill(ProcessKillResponse),
    CommandExec(CommandExecResponse),
    CommandExecWrite(CommandExecWriteResponse),
    CommandExecResize(CommandExecResizeResponse),
    CommandExecTerminate(CommandExecTerminateResponse),
}

impl ClientResponsePayload {
    pub fn method(&self) -> Method {
        match self {
            Self::ThreadStart(_) => Method::ThreadStart,
            Self::ThreadFork(_) => Method::ThreadFork,
            Self::ThreadResume(_) => Method::ThreadResume,
            Self::ThreadRead(_) => Method::ThreadRead,
            Self::ThreadList(_) => Method::ThreadList,
            Self::ThreadSectionMove(_) => Method::ThreadSectionMove,
            Self::ThreadSectionList(_) => Method::ThreadSectionList,
            Self::ThreadSectionCreate(_) => Method::ThreadSectionCreate,
            Self::ThreadSectionUpdate(_) => Method::ThreadSectionUpdate,
            Self::ThreadSectionDelete(_) => Method::ThreadSectionDelete,
            Self::ThreadLoadedList(_) => Method::ThreadLoadedList,
            Self::ThreadUnsubscribe(_) => Method::ThreadUnsubscribe,
            Self::ThreadIncrementElicitation(_) => Method::ThreadIncrementElicitation,
            Self::ThreadDecrementElicitation(_) => Method::ThreadDecrementElicitation,
            Self::ThreadArchive(_) => Method::ThreadArchive,
            Self::ThreadDelete(_) => Method::ThreadDelete,
            Self::ThreadUnarchive(_) => Method::ThreadUnarchive,
            Self::ThreadSetName(_) => Method::ThreadSetName,
            Self::ThreadMetadataUpdate(_) => Method::ThreadMetadataUpdate,
            Self::ThreadCompactStart(_) => Method::ThreadCompactStart,
            Self::ThreadTurnsList(_) => Method::ThreadTurnsList,
            Self::ThreadItemsList(_) => Method::ThreadItemsList,
            Self::ThreadInjectItems(_) => Method::ThreadInjectItems,
            Self::ThreadSearch(_) => Method::ThreadSearch,
            Self::ThreadSearchOccurrences(_) => Method::ThreadSearchOccurrences,
            Self::ThreadSettingsUpdate(_) => Method::ThreadSettingsUpdate,
            Self::ThreadMemoryModeSet(_) => Method::ThreadMemoryModeSet,
            Self::MemoryReset(_) => Method::MemoryReset,
            Self::ThreadShellCommand(_) => Method::ThreadShellCommand,
            Self::ThreadApproveGuardianDeniedAction(_) => Method::ThreadApproveGuardianDeniedAction,
            Self::ThreadBackgroundTerminalsClean(_) => Method::ThreadBackgroundTerminalsClean,
            Self::ThreadBackgroundTerminalsList(_) => Method::ThreadBackgroundTerminalsList,
            Self::ThreadBackgroundTerminalsTerminate(_) => {
                Method::ThreadBackgroundTerminalsTerminate
            }
            Self::ThreadGoalSet(_) => Method::ThreadGoalSet,
            Self::ThreadGoalGet(_) => Method::ThreadGoalGet,
            Self::ThreadGoalClear(_) => Method::ThreadGoalClear,
            Self::ArtifactWrite(_) => Method::ArtifactWrite,
            Self::MediaRead(_) => Method::MediaRead,
            Self::McpServerResourceRead(_) => Method::McpServerResourceRead,
            Self::McpServerToolCall(_) => Method::McpServerToolCall,
            Self::AppRead(_) => Method::AppRead,
            Self::AppList(_) => Method::AppList,
            Self::AppInstalled(_) => Method::AppInstalled,
            Self::HooksList(_) => Method::HooksList,
            Self::SkillsList(_) => Method::SkillsList,
            Self::SkillsExtraRootsSet(_) => Method::SkillsExtraRootsSet,
            Self::SkillsConfigWrite(_) => Method::SkillsConfigWrite,
            Self::PluginSearch(_) => Method::PluginSearch,
            Self::TurnStart(_) => Method::TurnStart,
            Self::TurnSteer(_) => Method::TurnSteer,
            Self::TurnInterrupt(_) => Method::TurnInterrupt,
            Self::ReviewStart(_) => Method::ReviewStart,
            Self::FsReadFile(_) => Method::FsReadFile,
            Self::FsWriteFile(_) => Method::FsWriteFile,
            Self::FsCreateDirectory(_) => Method::FsCreateDirectory,
            Self::FsGetMetadata(_) => Method::FsGetMetadata,
            Self::FsReadDirectory(_) => Method::FsReadDirectory,
            Self::FsRemove(_) => Method::FsRemove,
            Self::FsCopy(_) => Method::FsCopy,
            Self::FsWatch(_) => Method::FsWatch,
            Self::FsUnwatch(_) => Method::FsUnwatch,
            Self::ProcessSpawn(_) => Method::ProcessSpawn,
            Self::ProcessWriteStdin(_) => Method::ProcessWriteStdin,
            Self::ProcessResizePty(_) => Method::ProcessResizePty,
            Self::ProcessKill(_) => Method::ProcessKill,
            Self::CommandExec(_) => Method::CommandExec,
            Self::CommandExecWrite(_) => Method::CommandExecWrite,
            Self::CommandExecResize(_) => Method::CommandExecResize,
            Self::CommandExecTerminate(_) => Method::CommandExecTerminate,
        }
    }

    pub fn into_response(self, id: RequestId) -> Result<ClientResponse, serde_json::Error> {
        let result = match self {
            Self::ThreadStart(response) => serde_json::to_value(response)?,
            Self::ThreadFork(response) => serde_json::to_value(response)?,
            Self::ThreadResume(response) => serde_json::to_value(response)?,
            Self::ThreadRead(response) => serde_json::to_value(response)?,
            Self::ThreadList(response) => serde_json::to_value(response)?,
            Self::ThreadSectionMove(response) => serde_json::to_value(response)?,
            Self::ThreadSectionList(response) => serde_json::to_value(response)?,
            Self::ThreadSectionCreate(response) => serde_json::to_value(response)?,
            Self::ThreadSectionUpdate(response) => serde_json::to_value(response)?,
            Self::ThreadSectionDelete(response) => serde_json::to_value(response)?,
            Self::ThreadLoadedList(response) => serde_json::to_value(response)?,
            Self::ThreadUnsubscribe(response) => serde_json::to_value(response)?,
            Self::ThreadIncrementElicitation(response) => serde_json::to_value(response)?,
            Self::ThreadDecrementElicitation(response) => serde_json::to_value(response)?,
            Self::ThreadArchive(response) => serde_json::to_value(response)?,
            Self::ThreadDelete(response) => serde_json::to_value(response)?,
            Self::ThreadUnarchive(response) => serde_json::to_value(response)?,
            Self::ThreadSetName(response) => serde_json::to_value(response)?,
            Self::ThreadMetadataUpdate(response) => serde_json::to_value(response)?,
            Self::ThreadCompactStart(response) => serde_json::to_value(response)?,
            Self::ThreadTurnsList(response) => serde_json::to_value(response)?,
            Self::ThreadItemsList(response) => serde_json::to_value(response)?,
            Self::ThreadInjectItems(response) => serde_json::to_value(response)?,
            Self::ThreadSearch(response) => serde_json::to_value(response)?,
            Self::ThreadSearchOccurrences(response) => serde_json::to_value(response)?,
            Self::ThreadSettingsUpdate(response) => serde_json::to_value(response)?,
            Self::ThreadMemoryModeSet(response) => serde_json::to_value(response)?,
            Self::MemoryReset(response) => serde_json::to_value(response)?,
            Self::ThreadShellCommand(response) => serde_json::to_value(response)?,
            Self::ThreadApproveGuardianDeniedAction(response) => serde_json::to_value(response)?,
            Self::ThreadBackgroundTerminalsClean(response) => serde_json::to_value(response)?,
            Self::ThreadBackgroundTerminalsList(response) => serde_json::to_value(response)?,
            Self::ThreadBackgroundTerminalsTerminate(response) => serde_json::to_value(response)?,
            Self::ThreadGoalSet(response) => serde_json::to_value(response)?,
            Self::ThreadGoalGet(response) => serde_json::to_value(response)?,
            Self::ThreadGoalClear(response) => serde_json::to_value(response)?,
            Self::ArtifactWrite(response) => serde_json::to_value(response)?,
            Self::MediaRead(response) => serde_json::to_value(response)?,
            Self::McpServerResourceRead(response) => serde_json::to_value(response)?,
            Self::McpServerToolCall(response) => serde_json::to_value(response)?,
            Self::AppRead(response) => serde_json::to_value(response)?,
            Self::AppList(response) => serde_json::to_value(response)?,
            Self::AppInstalled(response) => serde_json::to_value(response)?,
            Self::HooksList(response) => serde_json::to_value(response)?,
            Self::SkillsList(response) => serde_json::to_value(response)?,
            Self::SkillsExtraRootsSet(response) => serde_json::to_value(response)?,
            Self::SkillsConfigWrite(response) => serde_json::to_value(response)?,
            Self::PluginSearch(response) => serde_json::to_value(response)?,
            Self::TurnStart(response) => serde_json::to_value(response)?,
            Self::TurnSteer(response) => serde_json::to_value(response)?,
            Self::TurnInterrupt(response) => serde_json::to_value(response)?,
            Self::ReviewStart(response) => serde_json::to_value(response)?,
            Self::FsReadFile(response) => serde_json::to_value(response)?,
            Self::FsWriteFile(response) => serde_json::to_value(response)?,
            Self::FsCreateDirectory(response) => serde_json::to_value(response)?,
            Self::FsGetMetadata(response) => serde_json::to_value(response)?,
            Self::FsReadDirectory(response) => serde_json::to_value(response)?,
            Self::FsRemove(response) => serde_json::to_value(response)?,
            Self::FsCopy(response) => serde_json::to_value(response)?,
            Self::FsWatch(response) => serde_json::to_value(response)?,
            Self::FsUnwatch(response) => serde_json::to_value(response)?,
            Self::ProcessSpawn(response) => serde_json::to_value(response)?,
            Self::ProcessWriteStdin(response) => serde_json::to_value(response)?,
            Self::ProcessResizePty(response) => serde_json::to_value(response)?,
            Self::ProcessKill(response) => serde_json::to_value(response)?,
            Self::CommandExec(response) => serde_json::to_value(response)?,
            Self::CommandExecWrite(response) => serde_json::to_value(response)?,
            Self::CommandExecResize(response) => serde_json::to_value(response)?,
            Self::CommandExecTerminate(response) => serde_json::to_value(response)?,
        };
        Ok(ClientResponse { id, result })
    }
}

/// Requests initiated by the server and sent to a v2 client. Unknown methods
/// fail closed until their typed params and response contracts are added.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "method")]
pub enum ServerRequest {
    #[serde(rename = "currentTime/read")]
    CurrentTimeRead {
        id: RequestId,
        params: CurrentTimeReadParams,
    },
    #[serde(rename = "mcpServer/elicitation/request")]
    McpServerElicitationRequest {
        id: RequestId,
        params: McpServerElicitationRequestParams,
    },
    #[serde(rename = "item/commandExecution/requestApproval")]
    ItemCommandExecutionRequestApproval {
        id: RequestId,
        params: CommandExecutionRequestApprovalParams,
    },
    #[serde(rename = "item/fileChange/requestApproval")]
    ItemFileChangeRequestApproval {
        id: RequestId,
        params: FileChangeRequestApprovalParams,
    },
    #[serde(rename = "item/permissions/requestApproval")]
    ItemPermissionsRequestApproval {
        id: RequestId,
        params: PermissionsRequestApprovalParams,
    },
    #[serde(rename = "item/tool/call")]
    DynamicToolCall {
        id: RequestId,
        params: DynamicToolCallParams,
    },
    #[serde(rename = "item/tool/requestUserInput")]
    ItemToolRequestUserInput {
        id: RequestId,
        params: ToolRequestUserInputParams,
    },
}

impl ServerRequest {
    pub fn id(&self) -> &RequestId {
        match self {
            Self::CurrentTimeRead { id, .. } | Self::McpServerElicitationRequest { id, .. } => id,
            Self::ItemCommandExecutionRequestApproval { id, .. } => id,
            Self::ItemFileChangeRequestApproval { id, .. } => id,
            Self::ItemPermissionsRequestApproval { id, .. } => id,
            Self::DynamicToolCall { id, .. } => id,
            Self::ItemToolRequestUserInput { id, .. } => id,
        }
    }

    pub fn method(&self) -> &'static str {
        match self {
            Self::CurrentTimeRead { .. } => METHOD_CURRENT_TIME_READ,
            Self::McpServerElicitationRequest { .. } => METHOD_MCP_SERVER_ELICITATION_REQUEST,
            Self::ItemCommandExecutionRequestApproval { .. } => {
                METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL
            }
            Self::ItemFileChangeRequestApproval { .. } => METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
            Self::ItemPermissionsRequestApproval { .. } => METHOD_ITEM_PERMISSIONS_REQUEST_APPROVAL,
            Self::DynamicToolCall { .. } => METHOD_ITEM_TOOL_CALL,
            Self::ItemToolRequestUserInput { .. } => METHOD_ITEM_TOOL_REQUEST_USER_INPUT,
        }
    }
}

impl TryFrom<JsonRpcRequest> for ServerRequest {
    type Error = String;

    fn try_from(request: JsonRpcRequest) -> Result<Self, Self::Error> {
        let params = request.params.unwrap_or_else(|| serde_json::json!({}));
        match request.method.as_str() {
            METHOD_CURRENT_TIME_READ => serde_json::from_value(params)
                .map(|params| Self::CurrentTimeRead {
                    id: request.id,
                    params,
                })
                .map_err(|error| error.to_string()),
            METHOD_MCP_SERVER_ELICITATION_REQUEST => serde_json::from_value(params)
                .map(|params| Self::McpServerElicitationRequest {
                    id: request.id,
                    params,
                })
                .map_err(|error| error.to_string()),
            METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL => serde_json::from_value(params)
                .map(|params| Self::ItemCommandExecutionRequestApproval {
                    id: request.id,
                    params,
                })
                .map_err(|error| error.to_string()),
            METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL => serde_json::from_value(params)
                .map(|params| Self::ItemFileChangeRequestApproval {
                    id: request.id,
                    params,
                })
                .map_err(|error| error.to_string()),
            METHOD_ITEM_PERMISSIONS_REQUEST_APPROVAL => serde_json::from_value(params)
                .map(|params| Self::ItemPermissionsRequestApproval {
                    id: request.id,
                    params,
                })
                .map_err(|error| error.to_string()),
            METHOD_ITEM_TOOL_CALL => serde_json::from_value(params)
                .map(|params| Self::DynamicToolCall {
                    id: request.id,
                    params,
                })
                .map_err(|error| error.to_string()),
            METHOD_ITEM_TOOL_REQUEST_USER_INPUT => serde_json::from_value(params)
                .map(|params| Self::ItemToolRequestUserInput {
                    id: request.id,
                    params,
                })
                .map_err(|error| error.to_string()),
            method => Err(format!("unknown v2 server request method: {method}")),
        }
    }
}

impl From<ServerRequest> for JsonRpcRequest {
    fn from(request: ServerRequest) -> Self {
        match request {
            ServerRequest::CurrentTimeRead { id, params } => JsonRpcRequest::new(
                id,
                METHOD_CURRENT_TIME_READ,
                Some(serde_json::to_value(params).expect("serialize v2 app-server request")),
            ),
            ServerRequest::McpServerElicitationRequest { id, params } => JsonRpcRequest::new(
                id,
                METHOD_MCP_SERVER_ELICITATION_REQUEST,
                Some(serde_json::to_value(params).expect("serialize v2 app-server request")),
            ),
            ServerRequest::ItemCommandExecutionRequestApproval { id, params } => {
                JsonRpcRequest::new(
                    id,
                    METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
                    Some(serde_json::to_value(params).expect("serialize v2 app-server request")),
                )
            }
            ServerRequest::ItemFileChangeRequestApproval { id, params } => JsonRpcRequest::new(
                id,
                METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
                Some(serde_json::to_value(params).expect("serialize v2 app-server request")),
            ),
            ServerRequest::ItemPermissionsRequestApproval { id, params } => JsonRpcRequest::new(
                id,
                METHOD_ITEM_PERMISSIONS_REQUEST_APPROVAL,
                Some(serde_json::to_value(params).expect("serialize v2 app-server request")),
            ),
            ServerRequest::DynamicToolCall { id, params } => JsonRpcRequest::new(
                id,
                METHOD_ITEM_TOOL_CALL,
                Some(serde_json::to_value(params).expect("serialize v2 app-server request")),
            ),
            ServerRequest::ItemToolRequestUserInput { id, params } => JsonRpcRequest::new(
                id,
                METHOD_ITEM_TOOL_REQUEST_USER_INPUT,
                Some(serde_json::to_value(params).expect("serialize v2 app-server request")),
            ),
        }
    }
}

/// Notifications emitted by the current v2 skeleton. Unknown methods fail
/// closed until their typed payloads are added to the v2 catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "method", content = "params")]
pub enum ServerNotification {
    #[serde(rename = "configWarning")]
    ConfigWarning(ConfigWarningNotification),
    #[serde(rename = "warning")]
    Warning(WarningNotification),
    #[serde(rename = "guardianWarning")]
    GuardianWarning(GuardianWarningNotification),
    #[serde(rename = "error")]
    Error(ErrorNotification),
    #[serde(rename = "skills/changed")]
    SkillsChanged(SkillsChangedNotification),
    #[serde(rename = "mcpServer/oauthLogin/completed")]
    McpServerOauthLoginCompleted(McpServerOauthLoginCompletedNotification),
    #[serde(rename = "mcpServer/startupStatus/updated")]
    McpServerStatusUpdated(McpServerStatusUpdatedNotification),
    #[serde(rename = "app/list/updated")]
    AppListUpdated(AppListUpdatedNotification),
    #[serde(rename = "hook/started")]
    HookStarted(HookStartedNotification),
    #[serde(rename = "hook/completed")]
    HookCompleted(HookCompletedNotification),
    #[serde(rename = "thread/started")]
    ThreadStarted(ThreadStartedNotification),
    #[serde(rename = "thread/archived")]
    ThreadArchived(ThreadArchivedNotification),
    #[serde(rename = "thread/deleted")]
    ThreadDeleted(ThreadDeletedNotification),
    #[serde(rename = "thread/unarchived")]
    ThreadUnarchived(ThreadUnarchivedNotification),
    #[serde(rename = "thread/closed")]
    ThreadClosed(ThreadClosedNotification),
    #[serde(rename = "thread/name/updated")]
    ThreadNameUpdated(ThreadNameUpdatedNotification),
    #[serde(rename = "thread/status/changed")]
    ThreadStatusChanged(ThreadStatusChangedNotification),
    #[serde(rename = "turn/started")]
    TurnStarted(TurnStartedNotification),
    #[serde(rename = "turn/completed")]
    TurnCompleted(TurnCompletedNotification),
    #[serde(rename = "turn/diff/updated")]
    TurnDiffUpdated(TurnDiffUpdatedNotification),
    #[serde(rename = "turn/plan/updated")]
    TurnPlanUpdated(TurnPlanUpdatedNotification),
    #[serde(rename = "item/started")]
    ItemStarted(ItemStartedNotification),
    #[serde(rename = "item/completed")]
    ItemCompleted(ItemCompletedNotification),
    #[serde(rename = "item/autoApprovalReview/started")]
    ItemAutoApprovalReviewStarted(ItemGuardianApprovalReviewStartedNotification),
    #[serde(rename = "item/autoApprovalReview/completed")]
    ItemAutoApprovalReviewCompleted(ItemGuardianApprovalReviewCompletedNotification),
    #[serde(rename = "item/agentMessage/delta")]
    AgentMessageDelta(AgentMessageDeltaNotification),
    #[serde(rename = "item/commandExecution/outputDelta")]
    CommandExecutionOutputDelta(CommandExecutionOutputDeltaNotification),
    #[serde(rename = "item/commandExecution/terminalInteraction")]
    CommandExecutionTerminalInteraction(CommandExecutionTerminalInteractionNotification),
    #[serde(rename = "item/fileChange/patchUpdated")]
    FileChangePatchUpdated(FileChangePatchUpdatedNotification),
    #[serde(rename = "item/plan/delta")]
    PlanDelta(PlanDeltaNotification),
    #[serde(rename = "item/mcpToolCall/progress")]
    McpToolCallProgress(McpToolCallProgressNotification),
    #[serde(rename = "item/reasoning/summaryTextDelta")]
    ReasoningSummaryTextDelta(ReasoningSummaryTextDeltaNotification),
    #[serde(rename = "item/reasoning/summaryPartAdded")]
    ReasoningSummaryPartAdded(ReasoningSummaryPartAddedNotification),
    #[serde(rename = "item/reasoning/textDelta")]
    ReasoningTextDelta(ReasoningTextDeltaNotification),
    #[serde(rename = "model/rerouted")]
    ModelRerouted(ModelReroutedNotification),
    #[serde(rename = "model/list/updated")]
    ModelListUpdated(ModelListUpdatedNotification),
    #[serde(rename = "model/verification")]
    ModelVerification(ModelVerificationNotification),
    #[serde(rename = "turn/moderationMetadata")]
    TurnModerationMetadata(TurnModerationMetadataNotification),
    #[serde(rename = "model/safetyBuffering/updated")]
    ModelSafetyBufferingUpdated(ModelSafetyBufferingUpdatedNotification),
    #[serde(rename = "fs/changed")]
    FsChanged(FsChangedNotification),
    #[serde(rename = "process/outputDelta")]
    ProcessOutputDelta(ProcessOutputDeltaNotification),
    #[serde(rename = "process/exited")]
    ProcessExited(ProcessExitedNotification),
    #[serde(rename = "command/exec/outputDelta")]
    CommandExecOutputDelta(CommandExecOutputDeltaNotification),
    #[serde(rename = "thread/settings/updated")]
    ThreadSettingsUpdated(ThreadSettingsUpdatedNotification),
    #[serde(rename = "thread/tokenUsage/updated")]
    ThreadTokenUsageUpdated(ThreadTokenUsageUpdatedNotification),
    #[serde(rename = "thread/goal/updated")]
    ThreadGoalUpdated(ThreadGoalUpdatedNotification),
    #[serde(rename = "thread/goal/cleared")]
    ThreadGoalCleared(ThreadGoalClearedNotification),
    #[serde(rename = "serverRequest/resolved")]
    ServerRequestResolved(ServerRequestResolvedNotification),
}

impl ServerNotification {
    pub fn method(&self) -> &'static str {
        match self {
            Self::ConfigWarning(_) => METHOD_CONFIG_WARNING,
            Self::Warning(_) => METHOD_WARNING,
            Self::GuardianWarning(_) => METHOD_GUARDIAN_WARNING,
            Self::Error(_) => METHOD_ERROR,
            Self::SkillsChanged(_) => METHOD_SKILLS_CHANGED,
            Self::McpServerOauthLoginCompleted(_) => METHOD_MCP_SERVER_OAUTH_LOGIN_COMPLETED,
            Self::McpServerStatusUpdated(_) => METHOD_MCP_SERVER_STARTUP_STATUS_UPDATED,
            Self::AppListUpdated(_) => METHOD_APP_LIST_UPDATED,
            Self::HookStarted(_) => METHOD_HOOK_STARTED,
            Self::HookCompleted(_) => METHOD_HOOK_COMPLETED,
            Self::ThreadStarted(_) => "thread/started",
            Self::ThreadArchived(_) => "thread/archived",
            Self::ThreadDeleted(_) => "thread/deleted",
            Self::ThreadUnarchived(_) => "thread/unarchived",
            Self::ThreadClosed(_) => METHOD_THREAD_CLOSED,
            Self::ThreadNameUpdated(_) => METHOD_THREAD_NAME_UPDATED,
            Self::ThreadStatusChanged(_) => METHOD_THREAD_STATUS_CHANGED,
            Self::TurnStarted(_) => "turn/started",
            Self::TurnCompleted(_) => "turn/completed",
            Self::TurnDiffUpdated(_) => METHOD_TURN_DIFF_UPDATED,
            Self::TurnPlanUpdated(_) => METHOD_TURN_PLAN_UPDATED,
            Self::ItemStarted(_) => "item/started",
            Self::ItemCompleted(_) => "item/completed",
            Self::ItemAutoApprovalReviewStarted(_) => METHOD_ITEM_AUTO_APPROVAL_REVIEW_STARTED,
            Self::ItemAutoApprovalReviewCompleted(_) => METHOD_ITEM_AUTO_APPROVAL_REVIEW_COMPLETED,
            Self::AgentMessageDelta(_) => "item/agentMessage/delta",
            Self::CommandExecutionOutputDelta(_) => METHOD_COMMAND_EXECUTION_OUTPUT_DELTA,
            Self::CommandExecutionTerminalInteraction(_) => {
                METHOD_COMMAND_EXECUTION_TERMINAL_INTERACTION
            }
            Self::FileChangePatchUpdated(_) => METHOD_FILE_CHANGE_PATCH_UPDATED,
            Self::PlanDelta(_) => METHOD_PLAN_DELTA,
            Self::McpToolCallProgress(_) => METHOD_MCP_TOOL_CALL_PROGRESS,
            Self::ReasoningSummaryTextDelta(_) => METHOD_REASONING_SUMMARY_TEXT_DELTA,
            Self::ReasoningSummaryPartAdded(_) => METHOD_REASONING_SUMMARY_PART_ADDED,
            Self::ReasoningTextDelta(_) => METHOD_REASONING_TEXT_DELTA,
            Self::ModelRerouted(_) => METHOD_MODEL_REROUTED,
            Self::ModelListUpdated(_) => METHOD_MODEL_LIST_UPDATED,
            Self::ModelVerification(_) => METHOD_MODEL_VERIFICATION,
            Self::TurnModerationMetadata(_) => METHOD_TURN_MODERATION_METADATA,
            Self::ModelSafetyBufferingUpdated(_) => METHOD_MODEL_SAFETY_BUFFERING_UPDATED,
            Self::FsChanged(_) => METHOD_FS_CHANGED,
            Self::ProcessOutputDelta(_) => METHOD_PROCESS_OUTPUT_DELTA,
            Self::ProcessExited(_) => METHOD_PROCESS_EXITED,
            Self::CommandExecOutputDelta(_) => METHOD_COMMAND_EXEC_OUTPUT_DELTA,
            Self::ThreadSettingsUpdated(_) => "thread/settings/updated",
            Self::ThreadTokenUsageUpdated(_) => METHOD_THREAD_TOKEN_USAGE_UPDATED,
            Self::ThreadGoalUpdated(_) => METHOD_THREAD_GOAL_UPDATED,
            Self::ThreadGoalCleared(_) => METHOD_THREAD_GOAL_CLEARED,
            Self::ServerRequestResolved(_) => METHOD_SERVER_REQUEST_RESOLVED,
        }
    }
}

impl TryFrom<JsonRpcNotification> for ServerNotification {
    type Error = String;

    fn try_from(notification: JsonRpcNotification) -> Result<Self, String> {
        let params = notification.params.unwrap_or_else(|| serde_json::json!({}));
        match notification.method.as_str() {
            METHOD_CONFIG_WARNING => serde_json::from_value(params)
                .map(Self::ConfigWarning)
                .map_err(|error| error.to_string()),
            METHOD_WARNING => serde_json::from_value(params)
                .map(Self::Warning)
                .map_err(|error| error.to_string()),
            METHOD_GUARDIAN_WARNING => serde_json::from_value(params)
                .map(Self::GuardianWarning)
                .map_err(|error| error.to_string()),
            METHOD_ERROR => serde_json::from_value(params)
                .map(Self::Error)
                .map_err(|error| error.to_string()),
            METHOD_SKILLS_CHANGED => serde_json::from_value(params)
                .map(Self::SkillsChanged)
                .map_err(|error| error.to_string()),
            METHOD_MCP_SERVER_OAUTH_LOGIN_COMPLETED => serde_json::from_value(params)
                .map(Self::McpServerOauthLoginCompleted)
                .map_err(|error| error.to_string()),
            METHOD_MCP_SERVER_STARTUP_STATUS_UPDATED => serde_json::from_value(params)
                .map(Self::McpServerStatusUpdated)
                .map_err(|error| error.to_string()),
            METHOD_APP_LIST_UPDATED => serde_json::from_value(params)
                .map(Self::AppListUpdated)
                .map_err(|error| error.to_string()),
            METHOD_HOOK_STARTED => serde_json::from_value(params)
                .map(Self::HookStarted)
                .map_err(|error| error.to_string()),
            METHOD_HOOK_COMPLETED => serde_json::from_value(params)
                .map(Self::HookCompleted)
                .map_err(|error| error.to_string()),
            "thread/started" => serde_json::from_value(params)
                .map(Self::ThreadStarted)
                .map_err(|error| error.to_string()),
            "thread/archived" => serde_json::from_value(params)
                .map(Self::ThreadArchived)
                .map_err(|error| error.to_string()),
            "thread/deleted" => serde_json::from_value(params)
                .map(Self::ThreadDeleted)
                .map_err(|error| error.to_string()),
            "thread/unarchived" => serde_json::from_value(params)
                .map(Self::ThreadUnarchived)
                .map_err(|error| error.to_string()),
            METHOD_THREAD_CLOSED => serde_json::from_value(params)
                .map(Self::ThreadClosed)
                .map_err(|error| error.to_string()),
            METHOD_THREAD_NAME_UPDATED => serde_json::from_value(params)
                .map(Self::ThreadNameUpdated)
                .map_err(|error| error.to_string()),
            METHOD_THREAD_STATUS_CHANGED => serde_json::from_value(params)
                .map(Self::ThreadStatusChanged)
                .map_err(|error| error.to_string()),
            "turn/started" => serde_json::from_value(params)
                .map(Self::TurnStarted)
                .map_err(|error| error.to_string()),
            "turn/completed" => serde_json::from_value(params)
                .map(Self::TurnCompleted)
                .map_err(|error| error.to_string()),
            METHOD_TURN_DIFF_UPDATED => serde_json::from_value(params)
                .map(Self::TurnDiffUpdated)
                .map_err(|error| error.to_string()),
            METHOD_TURN_PLAN_UPDATED => serde_json::from_value(params)
                .map(Self::TurnPlanUpdated)
                .map_err(|error| error.to_string()),
            "item/started" => serde_json::from_value(params)
                .map(Self::ItemStarted)
                .map_err(|error| error.to_string()),
            "item/completed" => serde_json::from_value(params)
                .map(Self::ItemCompleted)
                .map_err(|error| error.to_string()),
            METHOD_ITEM_AUTO_APPROVAL_REVIEW_STARTED => serde_json::from_value(params)
                .map(Self::ItemAutoApprovalReviewStarted)
                .map_err(|error| error.to_string()),
            METHOD_ITEM_AUTO_APPROVAL_REVIEW_COMPLETED => serde_json::from_value(params)
                .map(Self::ItemAutoApprovalReviewCompleted)
                .map_err(|error| error.to_string()),
            "item/agentMessage/delta" => serde_json::from_value(params)
                .map(Self::AgentMessageDelta)
                .map_err(|error| error.to_string()),
            METHOD_COMMAND_EXECUTION_OUTPUT_DELTA => serde_json::from_value(params)
                .map(Self::CommandExecutionOutputDelta)
                .map_err(|error| error.to_string()),
            METHOD_COMMAND_EXECUTION_TERMINAL_INTERACTION => serde_json::from_value(params)
                .map(Self::CommandExecutionTerminalInteraction)
                .map_err(|error| error.to_string()),
            METHOD_FILE_CHANGE_PATCH_UPDATED => serde_json::from_value(params)
                .map(Self::FileChangePatchUpdated)
                .map_err(|error| error.to_string()),
            METHOD_PLAN_DELTA => serde_json::from_value(params)
                .map(Self::PlanDelta)
                .map_err(|error| error.to_string()),
            METHOD_MCP_TOOL_CALL_PROGRESS => serde_json::from_value(params)
                .map(Self::McpToolCallProgress)
                .map_err(|error| error.to_string()),
            METHOD_REASONING_SUMMARY_TEXT_DELTA => serde_json::from_value(params)
                .map(Self::ReasoningSummaryTextDelta)
                .map_err(|error| error.to_string()),
            METHOD_REASONING_SUMMARY_PART_ADDED => serde_json::from_value(params)
                .map(Self::ReasoningSummaryPartAdded)
                .map_err(|error| error.to_string()),
            METHOD_REASONING_TEXT_DELTA => serde_json::from_value(params)
                .map(Self::ReasoningTextDelta)
                .map_err(|error| error.to_string()),
            METHOD_MODEL_REROUTED => serde_json::from_value(params)
                .map(Self::ModelRerouted)
                .map_err(|error| error.to_string()),
            METHOD_MODEL_LIST_UPDATED => serde_json::from_value(params)
                .map(Self::ModelListUpdated)
                .map_err(|error| error.to_string()),
            METHOD_MODEL_VERIFICATION => serde_json::from_value(params)
                .map(Self::ModelVerification)
                .map_err(|error| error.to_string()),
            METHOD_TURN_MODERATION_METADATA => serde_json::from_value(params)
                .map(Self::TurnModerationMetadata)
                .map_err(|error| error.to_string()),
            METHOD_MODEL_SAFETY_BUFFERING_UPDATED => serde_json::from_value(params)
                .map(Self::ModelSafetyBufferingUpdated)
                .map_err(|error| error.to_string()),
            METHOD_FS_CHANGED => serde_json::from_value(params)
                .map(Self::FsChanged)
                .map_err(|error| error.to_string()),
            METHOD_PROCESS_OUTPUT_DELTA => serde_json::from_value(params)
                .map(Self::ProcessOutputDelta)
                .map_err(|error| error.to_string()),
            METHOD_PROCESS_EXITED => serde_json::from_value(params)
                .map(Self::ProcessExited)
                .map_err(|error| error.to_string()),
            METHOD_COMMAND_EXEC_OUTPUT_DELTA => serde_json::from_value(params)
                .map(Self::CommandExecOutputDelta)
                .map_err(|error| error.to_string()),
            "thread/settings/updated" => serde_json::from_value(params)
                .map(Self::ThreadSettingsUpdated)
                .map_err(|error| error.to_string()),
            METHOD_THREAD_TOKEN_USAGE_UPDATED => serde_json::from_value(params)
                .map(Self::ThreadTokenUsageUpdated)
                .map_err(|error| error.to_string()),
            METHOD_THREAD_GOAL_UPDATED => serde_json::from_value(params)
                .map(Self::ThreadGoalUpdated)
                .map_err(|error| error.to_string()),
            METHOD_THREAD_GOAL_CLEARED => serde_json::from_value(params)
                .map(Self::ThreadGoalCleared)
                .map_err(|error| error.to_string()),
            METHOD_SERVER_REQUEST_RESOLVED => serde_json::from_value(params)
                .map(Self::ServerRequestResolved)
                .map_err(|error| error.to_string()),
            method => Err(format!("unknown v2 notification method: {method}")),
        }
    }
}

impl From<ServerNotification> for JsonRpcNotification {
    fn from(notification: ServerNotification) -> Self {
        match notification {
            ServerNotification::ConfigWarning(params) => {
                jsonrpc_notification(METHOD_CONFIG_WARNING, params)
            }
            ServerNotification::Warning(params) => jsonrpc_notification(METHOD_WARNING, params),
            ServerNotification::GuardianWarning(params) => {
                jsonrpc_notification(METHOD_GUARDIAN_WARNING, params)
            }
            ServerNotification::Error(params) => jsonrpc_notification(METHOD_ERROR, params),
            ServerNotification::SkillsChanged(params) => {
                jsonrpc_notification(METHOD_SKILLS_CHANGED, params)
            }
            ServerNotification::McpServerOauthLoginCompleted(params) => {
                jsonrpc_notification(METHOD_MCP_SERVER_OAUTH_LOGIN_COMPLETED, params)
            }
            ServerNotification::McpServerStatusUpdated(params) => {
                jsonrpc_notification(METHOD_MCP_SERVER_STARTUP_STATUS_UPDATED, params)
            }
            ServerNotification::AppListUpdated(params) => {
                jsonrpc_notification(METHOD_APP_LIST_UPDATED, params)
            }
            ServerNotification::HookStarted(params) => {
                jsonrpc_notification(METHOD_HOOK_STARTED, params)
            }
            ServerNotification::HookCompleted(params) => {
                jsonrpc_notification(METHOD_HOOK_COMPLETED, params)
            }
            ServerNotification::ThreadStarted(params) => {
                jsonrpc_notification("thread/started", params)
            }
            ServerNotification::ThreadArchived(params) => {
                jsonrpc_notification("thread/archived", params)
            }
            ServerNotification::ThreadDeleted(params) => {
                jsonrpc_notification("thread/deleted", params)
            }
            ServerNotification::ThreadUnarchived(params) => {
                jsonrpc_notification("thread/unarchived", params)
            }
            ServerNotification::ThreadClosed(params) => {
                jsonrpc_notification(METHOD_THREAD_CLOSED, params)
            }
            ServerNotification::ThreadNameUpdated(params) => {
                jsonrpc_notification(METHOD_THREAD_NAME_UPDATED, params)
            }
            ServerNotification::ThreadStatusChanged(params) => {
                jsonrpc_notification(METHOD_THREAD_STATUS_CHANGED, params)
            }
            ServerNotification::TurnStarted(params) => jsonrpc_notification("turn/started", params),
            ServerNotification::TurnCompleted(params) => {
                jsonrpc_notification("turn/completed", params)
            }
            ServerNotification::TurnDiffUpdated(params) => {
                jsonrpc_notification(METHOD_TURN_DIFF_UPDATED, params)
            }
            ServerNotification::TurnPlanUpdated(params) => {
                jsonrpc_notification(METHOD_TURN_PLAN_UPDATED, params)
            }
            ServerNotification::ItemStarted(params) => jsonrpc_notification("item/started", params),
            ServerNotification::ItemCompleted(params) => {
                jsonrpc_notification("item/completed", params)
            }
            ServerNotification::ItemAutoApprovalReviewStarted(params) => {
                jsonrpc_notification(METHOD_ITEM_AUTO_APPROVAL_REVIEW_STARTED, params)
            }
            ServerNotification::ItemAutoApprovalReviewCompleted(params) => {
                jsonrpc_notification(METHOD_ITEM_AUTO_APPROVAL_REVIEW_COMPLETED, params)
            }
            ServerNotification::AgentMessageDelta(params) => {
                jsonrpc_notification("item/agentMessage/delta", params)
            }
            ServerNotification::CommandExecutionOutputDelta(params) => {
                jsonrpc_notification(METHOD_COMMAND_EXECUTION_OUTPUT_DELTA, params)
            }
            ServerNotification::CommandExecutionTerminalInteraction(params) => {
                jsonrpc_notification(METHOD_COMMAND_EXECUTION_TERMINAL_INTERACTION, params)
            }
            ServerNotification::FileChangePatchUpdated(params) => {
                jsonrpc_notification(METHOD_FILE_CHANGE_PATCH_UPDATED, params)
            }
            ServerNotification::PlanDelta(params) => {
                jsonrpc_notification(METHOD_PLAN_DELTA, params)
            }
            ServerNotification::McpToolCallProgress(params) => {
                jsonrpc_notification(METHOD_MCP_TOOL_CALL_PROGRESS, params)
            }
            ServerNotification::ReasoningSummaryTextDelta(params) => {
                jsonrpc_notification(METHOD_REASONING_SUMMARY_TEXT_DELTA, params)
            }
            ServerNotification::ReasoningSummaryPartAdded(params) => {
                jsonrpc_notification(METHOD_REASONING_SUMMARY_PART_ADDED, params)
            }
            ServerNotification::ReasoningTextDelta(params) => {
                jsonrpc_notification(METHOD_REASONING_TEXT_DELTA, params)
            }
            ServerNotification::ModelRerouted(params) => {
                jsonrpc_notification(METHOD_MODEL_REROUTED, params)
            }
            ServerNotification::ModelListUpdated(params) => {
                jsonrpc_notification(METHOD_MODEL_LIST_UPDATED, params)
            }
            ServerNotification::ModelVerification(params) => {
                jsonrpc_notification(METHOD_MODEL_VERIFICATION, params)
            }
            ServerNotification::TurnModerationMetadata(params) => {
                jsonrpc_notification(METHOD_TURN_MODERATION_METADATA, params)
            }
            ServerNotification::ModelSafetyBufferingUpdated(params) => {
                jsonrpc_notification(METHOD_MODEL_SAFETY_BUFFERING_UPDATED, params)
            }
            ServerNotification::FsChanged(params) => {
                jsonrpc_notification(METHOD_FS_CHANGED, params)
            }
            ServerNotification::ProcessOutputDelta(params) => {
                jsonrpc_notification(METHOD_PROCESS_OUTPUT_DELTA, params)
            }
            ServerNotification::ProcessExited(params) => {
                jsonrpc_notification(METHOD_PROCESS_EXITED, params)
            }
            ServerNotification::CommandExecOutputDelta(params) => {
                jsonrpc_notification(METHOD_COMMAND_EXEC_OUTPUT_DELTA, params)
            }
            ServerNotification::ThreadSettingsUpdated(params) => {
                jsonrpc_notification("thread/settings/updated", params)
            }
            ServerNotification::ThreadTokenUsageUpdated(params) => {
                jsonrpc_notification(METHOD_THREAD_TOKEN_USAGE_UPDATED, params)
            }
            ServerNotification::ThreadGoalUpdated(params) => {
                jsonrpc_notification(METHOD_THREAD_GOAL_UPDATED, params)
            }
            ServerNotification::ThreadGoalCleared(params) => {
                jsonrpc_notification(METHOD_THREAD_GOAL_CLEARED, params)
            }
            ServerNotification::ServerRequestResolved(params) => {
                jsonrpc_notification(METHOD_SERVER_REQUEST_RESOLVED, params)
            }
        }
    }
}

fn jsonrpc_notification(method: &'static str, params: impl Serialize) -> JsonRpcNotification {
    JsonRpcNotification::new(
        method,
        Some(serde_json::to_value(params).expect("serialize v2 app-server notification")),
    )
}
