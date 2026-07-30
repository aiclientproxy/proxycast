export interface AgentRuntimeCreateSessionOptions {
  runStartHooks?: boolean;
  workingDir?: string | null;
  metadata?: Record<string, unknown>;
}

export const AGENT_RUNTIME_DEFAULT_HISTORY_LIMIT = 40;

export interface AgentRuntimeInterruptTurnRequest {
  session_id: string;
  turn_id?: string;
  event_name?: string;
}

export interface AgentRuntimeCompactSessionRequest {
  session_id: string;
  event_name: string;
}

export interface AgentRuntimeCapabilityManifestRequest {
  app_id?: string;
  workspace_id?: string;
  session_id?: string;
  cursor?: string;
  limit?: number;
}

export interface AgentRuntimeGetSessionOptions {
  resumeSessionStartHooks?: boolean;
  /**
   * 前端诊断来源，只进入客户端日志和性能指标，不透传到 App Server。
   */
  source?: string;
  /**
   * 限制每个 canonical Turn/Item owner 返回的 entry 数量；传 0 表示读到 EOF。
   */
  historyLimit?: number;
  /**
   * `thread/items/list` 返回的 opaque next cursor。`null` 表示 Item 已到 EOF。
   */
  historyItemCursor?: string | null;
  /**
   * `thread/turns/list` 返回的 opaque next cursor。`null` 表示 Turn 已到 EOF。
   */
  historyTurnCursor?: string | null;
}

export interface AgentRuntimeReplayRequestRequest {
  session_id: string;
  request_id: string;
}

export interface AgentRuntimeListFileCheckpointsRequest {
  session_id: string;
}

export interface AgentRuntimeGetFileCheckpointRequest {
  session_id: string;
  checkpoint_id: string;
}

export interface AgentRuntimeDiffFileCheckpointRequest {
  session_id: string;
  checkpoint_id: string;
}

export interface AgentRuntimeRestoreFileCheckpointRequest {
  session_id: string;
  checkpoint_id: string;
  confirm_restore: boolean;
  create_backup?: boolean;
}

export interface AgentRuntimeRespondActionRequest {
  session_id: string;
  request_id: string;
  action_type: "tool_confirmation" | "ask_user" | "elicitation";
  confirmed?: boolean;
  decision?: "allow_once" | "allow_for_session" | "decline" | "cancel";
  response?: string;
  user_data?: unknown;
  metadata?: Record<string, unknown>;
  event_name?: string;
  action_scope?: {
    session_id?: string;
    thread_id?: string;
    turn_id?: string;
  };
}

export interface AgentRuntimeReplayedActionRequiredView {
  type: "action_required";
  request_id: string;
  action_type: "tool_confirmation" | "ask_user" | "elicitation";
  tool_name?: string;
  arguments?: Record<string, unknown>;
  prompt?: string;
  questions?: unknown;
  requested_schema?: Record<string, unknown>;
  available_decisions?: Array<
    "allow_once" | "allow_for_session" | "decline" | "cancel"
  >;
  scope?: {
    session_id?: string;
    thread_id?: string;
    turn_id?: string;
  };
}
