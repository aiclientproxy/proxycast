use super::{
    AgentMessageDeltaNotification, ArtifactWriteParams, ArtifactWriteResponse,
    CommandExecutionOutputDeltaNotification, CommandExecutionRequestApprovalParams,
    FileChangePatchUpdatedNotification, FileChangeRequestApprovalParams, ItemCompletedNotification,
    ItemStartedNotification, McpServerElicitationRequestParams, McpToolCallProgressNotification,
    Method, PlanDeltaNotification, ReasoningSummaryPartAddedNotification,
    ReasoningSummaryTextDeltaNotification, ReasoningTextDeltaNotification,
    ServerRequestResolvedNotification, ThreadArchiveParams, ThreadArchiveResponse,
    ThreadArchivedNotification, ThreadDeleteParams, ThreadDeleteResponse,
    ThreadDeletedNotification, ThreadForkParams, ThreadForkResponse, ThreadGoalClearParams,
    ThreadGoalClearResponse, ThreadGoalClearedNotification, ThreadGoalGetParams,
    ThreadGoalGetResponse, ThreadGoalSetParams, ThreadGoalSetResponse,
    ThreadGoalUpdatedNotification, ThreadItemsListParams, ThreadItemsListResponse,
    ThreadListParams, ThreadListResponse, ThreadMemoryModeSetParams, ThreadMemoryModeSetResponse,
    ThreadReadParams, ThreadReadResponse, ThreadResumeParams, ThreadResumeResponse,
    ThreadSettingsUpdateParams, ThreadSettingsUpdateResponse, ThreadSettingsUpdatedNotification,
    ThreadShellCommandParams, ThreadShellCommandResponse, ThreadStartParams, ThreadStartResponse,
    ThreadStartedNotification, ThreadTokenUsageUpdatedNotification, ThreadTurnsListParams,
    ThreadTurnsListResponse, ThreadUnarchiveParams, ThreadUnarchiveResponse,
    ThreadUnarchivedNotification, ToolRequestUserInputParams, TurnCompletedNotification,
    TurnInterruptParams, TurnInterruptResponse, TurnStartParams, TurnStartResponse,
    TurnStartedNotification, TurnSteerParams, TurnSteerResponse,
    METHOD_COMMAND_EXECUTION_OUTPUT_DELTA, METHOD_FILE_CHANGE_PATCH_UPDATED,
    METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL, METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
    METHOD_ITEM_TOOL_REQUEST_USER_INPUT, METHOD_MCP_SERVER_ELICITATION_REQUEST,
    METHOD_MCP_TOOL_CALL_PROGRESS, METHOD_PLAN_DELTA, METHOD_REASONING_SUMMARY_PART_ADDED,
    METHOD_REASONING_SUMMARY_TEXT_DELTA, METHOD_REASONING_TEXT_DELTA,
    METHOD_SERVER_REQUEST_RESOLVED, METHOD_THREAD_GOAL_CLEARED, METHOD_THREAD_GOAL_UPDATED,
    METHOD_THREAD_TOKEN_USAGE_UPDATED,
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
    #[serde(rename = "thread/shellCommand")]
    ThreadShellCommand {
        id: RequestId,
        params: ThreadShellCommandParams,
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
}

impl ClientRequest {
    pub fn id(&self) -> &RequestId {
        match self {
            Self::ThreadStart { id, .. }
            | Self::ThreadFork { id, .. }
            | Self::ThreadResume { id, .. }
            | Self::ThreadRead { id, .. }
            | Self::ThreadList { id, .. }
            | Self::ThreadArchive { id, .. }
            | Self::ThreadDelete { id, .. }
            | Self::ThreadUnarchive { id, .. }
            | Self::ThreadTurnsList { id, .. }
            | Self::ThreadItemsList { id, .. }
            | Self::ThreadSettingsUpdate { id, .. }
            | Self::ThreadMemoryModeSet { id, .. }
            | Self::ThreadShellCommand { id, .. }
            | Self::ThreadGoalSet { id, .. }
            | Self::ThreadGoalGet { id, .. }
            | Self::ThreadGoalClear { id, .. }
            | Self::ArtifactWrite { id, .. }
            | Self::TurnStart { id, .. }
            | Self::TurnSteer { id, .. }
            | Self::TurnInterrupt { id, .. } => id,
        }
    }

    pub fn method(&self) -> Method {
        match self {
            Self::ThreadStart { .. } => Method::ThreadStart,
            Self::ThreadFork { .. } => Method::ThreadFork,
            Self::ThreadResume { .. } => Method::ThreadResume,
            Self::ThreadRead { .. } => Method::ThreadRead,
            Self::ThreadList { .. } => Method::ThreadList,
            Self::ThreadArchive { .. } => Method::ThreadArchive,
            Self::ThreadDelete { .. } => Method::ThreadDelete,
            Self::ThreadUnarchive { .. } => Method::ThreadUnarchive,
            Self::ThreadTurnsList { .. } => Method::ThreadTurnsList,
            Self::ThreadItemsList { .. } => Method::ThreadItemsList,
            Self::ThreadSettingsUpdate { .. } => Method::ThreadSettingsUpdate,
            Self::ThreadMemoryModeSet { .. } => Method::ThreadMemoryModeSet,
            Self::ThreadShellCommand { .. } => Method::ThreadShellCommand,
            Self::ThreadGoalSet { .. } => Method::ThreadGoalSet,
            Self::ThreadGoalGet { .. } => Method::ThreadGoalGet,
            Self::ThreadGoalClear { .. } => Method::ThreadGoalClear,
            Self::ArtifactWrite { .. } => Method::ArtifactWrite,
            Self::TurnStart { .. } => Method::TurnStart,
            Self::TurnSteer { .. } => Method::TurnSteer,
            Self::TurnInterrupt { .. } => Method::TurnInterrupt,
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
    ThreadArchive(ThreadArchiveResponse),
    ThreadDelete(ThreadDeleteResponse),
    ThreadUnarchive(ThreadUnarchiveResponse),
    ThreadTurnsList(ThreadTurnsListResponse),
    ThreadItemsList(ThreadItemsListResponse),
    ThreadSettingsUpdate(ThreadSettingsUpdateResponse),
    ThreadMemoryModeSet(ThreadMemoryModeSetResponse),
    ThreadShellCommand(ThreadShellCommandResponse),
    ThreadGoalSet(ThreadGoalSetResponse),
    ThreadGoalGet(ThreadGoalGetResponse),
    ThreadGoalClear(ThreadGoalClearResponse),
    ArtifactWrite(ArtifactWriteResponse),
    TurnStart(TurnStartResponse),
    TurnSteer(TurnSteerResponse),
    TurnInterrupt(TurnInterruptResponse),
}

impl ClientResponsePayload {
    pub fn method(&self) -> Method {
        match self {
            Self::ThreadStart(_) => Method::ThreadStart,
            Self::ThreadFork(_) => Method::ThreadFork,
            Self::ThreadResume(_) => Method::ThreadResume,
            Self::ThreadRead(_) => Method::ThreadRead,
            Self::ThreadList(_) => Method::ThreadList,
            Self::ThreadArchive(_) => Method::ThreadArchive,
            Self::ThreadDelete(_) => Method::ThreadDelete,
            Self::ThreadUnarchive(_) => Method::ThreadUnarchive,
            Self::ThreadTurnsList(_) => Method::ThreadTurnsList,
            Self::ThreadItemsList(_) => Method::ThreadItemsList,
            Self::ThreadSettingsUpdate(_) => Method::ThreadSettingsUpdate,
            Self::ThreadMemoryModeSet(_) => Method::ThreadMemoryModeSet,
            Self::ThreadShellCommand(_) => Method::ThreadShellCommand,
            Self::ThreadGoalSet(_) => Method::ThreadGoalSet,
            Self::ThreadGoalGet(_) => Method::ThreadGoalGet,
            Self::ThreadGoalClear(_) => Method::ThreadGoalClear,
            Self::ArtifactWrite(_) => Method::ArtifactWrite,
            Self::TurnStart(_) => Method::TurnStart,
            Self::TurnSteer(_) => Method::TurnSteer,
            Self::TurnInterrupt(_) => Method::TurnInterrupt,
        }
    }

    pub fn into_response(self, id: RequestId) -> Result<ClientResponse, serde_json::Error> {
        let result = match self {
            Self::ThreadStart(response) => serde_json::to_value(response)?,
            Self::ThreadFork(response) => serde_json::to_value(response)?,
            Self::ThreadResume(response) => serde_json::to_value(response)?,
            Self::ThreadRead(response) => serde_json::to_value(response)?,
            Self::ThreadList(response) => serde_json::to_value(response)?,
            Self::ThreadArchive(response) => serde_json::to_value(response)?,
            Self::ThreadDelete(response) => serde_json::to_value(response)?,
            Self::ThreadUnarchive(response) => serde_json::to_value(response)?,
            Self::ThreadTurnsList(response) => serde_json::to_value(response)?,
            Self::ThreadItemsList(response) => serde_json::to_value(response)?,
            Self::ThreadSettingsUpdate(response) => serde_json::to_value(response)?,
            Self::ThreadMemoryModeSet(response) => serde_json::to_value(response)?,
            Self::ThreadShellCommand(response) => serde_json::to_value(response)?,
            Self::ThreadGoalSet(response) => serde_json::to_value(response)?,
            Self::ThreadGoalGet(response) => serde_json::to_value(response)?,
            Self::ThreadGoalClear(response) => serde_json::to_value(response)?,
            Self::ArtifactWrite(response) => serde_json::to_value(response)?,
            Self::TurnStart(response) => serde_json::to_value(response)?,
            Self::TurnSteer(response) => serde_json::to_value(response)?,
            Self::TurnInterrupt(response) => serde_json::to_value(response)?,
        };
        Ok(ClientResponse { id, result })
    }
}

/// Requests initiated by the server and sent to a v2 client. Unknown methods
/// fail closed until their typed params and response contracts are added.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "method")]
pub enum ServerRequest {
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
    #[serde(rename = "item/tool/requestUserInput")]
    ItemToolRequestUserInput {
        id: RequestId,
        params: ToolRequestUserInputParams,
    },
}

impl ServerRequest {
    pub fn id(&self) -> &RequestId {
        match self {
            Self::McpServerElicitationRequest { id, .. } => id,
            Self::ItemCommandExecutionRequestApproval { id, .. } => id,
            Self::ItemFileChangeRequestApproval { id, .. } => id,
            Self::ItemToolRequestUserInput { id, .. } => id,
        }
    }

    pub fn method(&self) -> &'static str {
        match self {
            Self::McpServerElicitationRequest { .. } => METHOD_MCP_SERVER_ELICITATION_REQUEST,
            Self::ItemCommandExecutionRequestApproval { .. } => {
                METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL
            }
            Self::ItemFileChangeRequestApproval { .. } => METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
            Self::ItemToolRequestUserInput { .. } => METHOD_ITEM_TOOL_REQUEST_USER_INPUT,
        }
    }
}

impl TryFrom<JsonRpcRequest> for ServerRequest {
    type Error = String;

    fn try_from(request: JsonRpcRequest) -> Result<Self, Self::Error> {
        let params = request.params.unwrap_or_else(|| serde_json::json!({}));
        match request.method.as_str() {
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
    #[serde(rename = "thread/started")]
    ThreadStarted(ThreadStartedNotification),
    #[serde(rename = "thread/archived")]
    ThreadArchived(ThreadArchivedNotification),
    #[serde(rename = "thread/deleted")]
    ThreadDeleted(ThreadDeletedNotification),
    #[serde(rename = "thread/unarchived")]
    ThreadUnarchived(ThreadUnarchivedNotification),
    #[serde(rename = "turn/started")]
    TurnStarted(TurnStartedNotification),
    #[serde(rename = "turn/completed")]
    TurnCompleted(TurnCompletedNotification),
    #[serde(rename = "item/started")]
    ItemStarted(ItemStartedNotification),
    #[serde(rename = "item/completed")]
    ItemCompleted(ItemCompletedNotification),
    #[serde(rename = "item/agentMessage/delta")]
    AgentMessageDelta(AgentMessageDeltaNotification),
    #[serde(rename = "item/commandExecution/outputDelta")]
    CommandExecutionOutputDelta(CommandExecutionOutputDeltaNotification),
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
            Self::ThreadStarted(_) => "thread/started",
            Self::ThreadArchived(_) => "thread/archived",
            Self::ThreadDeleted(_) => "thread/deleted",
            Self::ThreadUnarchived(_) => "thread/unarchived",
            Self::TurnStarted(_) => "turn/started",
            Self::TurnCompleted(_) => "turn/completed",
            Self::ItemStarted(_) => "item/started",
            Self::ItemCompleted(_) => "item/completed",
            Self::AgentMessageDelta(_) => "item/agentMessage/delta",
            Self::CommandExecutionOutputDelta(_) => METHOD_COMMAND_EXECUTION_OUTPUT_DELTA,
            Self::FileChangePatchUpdated(_) => METHOD_FILE_CHANGE_PATCH_UPDATED,
            Self::PlanDelta(_) => METHOD_PLAN_DELTA,
            Self::McpToolCallProgress(_) => METHOD_MCP_TOOL_CALL_PROGRESS,
            Self::ReasoningSummaryTextDelta(_) => METHOD_REASONING_SUMMARY_TEXT_DELTA,
            Self::ReasoningSummaryPartAdded(_) => METHOD_REASONING_SUMMARY_PART_ADDED,
            Self::ReasoningTextDelta(_) => METHOD_REASONING_TEXT_DELTA,
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

    fn try_from(notification: JsonRpcNotification) -> Result<Self, Self::Error> {
        let params = notification.params.unwrap_or_else(|| serde_json::json!({}));
        match notification.method.as_str() {
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
            "turn/started" => serde_json::from_value(params)
                .map(Self::TurnStarted)
                .map_err(|error| error.to_string()),
            "turn/completed" => serde_json::from_value(params)
                .map(Self::TurnCompleted)
                .map_err(|error| error.to_string()),
            "item/started" => serde_json::from_value(params)
                .map(Self::ItemStarted)
                .map_err(|error| error.to_string()),
            "item/completed" => serde_json::from_value(params)
                .map(Self::ItemCompleted)
                .map_err(|error| error.to_string()),
            "item/agentMessage/delta" => serde_json::from_value(params)
                .map(Self::AgentMessageDelta)
                .map_err(|error| error.to_string()),
            METHOD_COMMAND_EXECUTION_OUTPUT_DELTA => serde_json::from_value(params)
                .map(Self::CommandExecutionOutputDelta)
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
            ServerNotification::TurnStarted(params) => jsonrpc_notification("turn/started", params),
            ServerNotification::TurnCompleted(params) => {
                jsonrpc_notification("turn/completed", params)
            }
            ServerNotification::ItemStarted(params) => jsonrpc_notification("item/started", params),
            ServerNotification::ItemCompleted(params) => {
                jsonrpc_notification("item/completed", params)
            }
            ServerNotification::AgentMessageDelta(params) => {
                jsonrpc_notification("item/agentMessage/delta", params)
            }
            ServerNotification::CommandExecutionOutputDelta(params) => {
                jsonrpc_notification(METHOD_COMMAND_EXECUTION_OUTPUT_DELTA, params)
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
