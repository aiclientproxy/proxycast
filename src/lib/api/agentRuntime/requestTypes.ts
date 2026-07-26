export interface AgentRuntimeCreateSessionOptions {
  runStartHooks?: boolean;
  workingDir?: string | null;
  metadata?: Record<string, unknown>;
}

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
   * 限制返回的历史窗口数量；传 0 表示请求完整历史。
   */
  historyLimit?: number;
  /**
   * 从最新历史向前跳过的消息数量，用于加载更早历史分页。
   */
  historyOffset?: number;
  /**
   * 稳定游标：读取指定后端消息 ID 之前的更早历史，优先于 offset。
   */
  historyBeforeMessageId?: number;
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
