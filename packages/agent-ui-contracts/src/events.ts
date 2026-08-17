export type AgentUiOwner =
  | "runtime"
  | "model"
  | "tool"
  | "action"
  | "artifact"
  | "evidence"
  | "context"
  | "policy"
  | "task"
  | "session"
  | "diagnostics"
  | "ui_projection"
  | "unknown"
  | "agent";

export type AgentUiScope =
  | "application"
  | "workspace"
  | "session"
  | "thread"
  | "run"
  | "turn"
  | "message"
  | "part"
  | "task"
  | "agent"
  | "tool_call"
  | "action_request"
  | "artifact"
  | "evidence"
  | "unknown";

export type AgentUiPhase =
  | "draft"
  | "submitted"
  | "accepted"
  | "routing"
  | "preparing"
  | "planning"
  | "reasoning"
  | "acting"
  | "waiting"
  | "producing"
  | "reconciling"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted"
  | "archived"
  | "hydrating"
  | "unknown"
  | "reviewing";

export type AgentUiSurface =
  | "composer"
  | "conversation"
  | "inline_process"
  | "runtime_status"
  | "tool_ui"
  | "hitl"
  | "task_capsule"
  | "artifact_workspace"
  | "timeline_evidence"
  | "session_tabs"
  | "diagnostics"
  | "custom"
  | "unknown"
  | "team_roster"
  | "work_board"
  | "delegation_graph"
  | "handoff_lane"
  | "worker_notifications"
  | "review_lane"
  | "teammate_transcript"
  | "remote_teammate"
  | "team_policy";

export type AgentUiPersistence =
  | "ephemeral_live"
  | "transcript"
  | "snapshot"
  | "archive"
  | "artifact_store"
  | "evidence_pack"
  | "diagnostics_log"
  | "ui_local"
  | "unknown";

export type AgentUiControl =
  | "send"
  | "queue"
  | "steer"
  | "interrupt"
  | "approve"
  | "reject"
  | "answer"
  | "edit"
  | "retry"
  | "rollback"
  | "remove"
  | "export"
  | "open_detail"
  | "none"
  | "unknown"
  | "delegate"
  | "assign"
  | "continue_agent"
  | "wait"
  | "stop"
  | "close"
  | "request_review";

export const AGENT_UI_EVENT_CLASSES = [
  "session.opened",
  "session.hydrated",
  "run.started",
  "run.status",
  "run.finished",
  "run.failed",
  "run.canceled",
  "plan.delta",
  "plan.final",
  "text.delta",
  "text.final",
  "reasoning.delta",
  "reasoning.summary",
  "tool.started",
  "tool.args",
  "tool.args.delta",
  "tool.progress",
  "tool.output.delta",
  "tool.result",
  "tool.failed",
  "tool.changed",
  "action.required",
  "action.resolved",
  "queue.changed",
  "task.changed",
  "agent.changed",
  "context.changed",
  "context.compaction.started",
  "context.compaction.completed",
  "permission.changed",
  "artifact.created",
  "artifact.updated",
  "artifact.preview.ready",
  "artifact.diff.ready",
  "artifact.failed",
  "artifact.changed",
  "evidence.changed",
  "state.snapshot",
  "state.delta",
  "messages.snapshot",
  "diagnostic.changed",
  "metric.changed",
  "agent.spawned",
  "agent.completed",
  "agent.handoff",
  "worker.notification",
  "review.requested",
  "review.completed",
] as const;

export type AgentUiEventClass = (typeof AGENT_UI_EVENT_CLASSES)[number];

export type AgentUiRuntimeEntity =
  | "agent_turn"
  | "subagent_turn"
  | "external_task"
  | "work_item"
  | "unknown";

export type AgentUiTopology =
  | "solo_run"
  | "coordinator_team"
  | "parallel_workers"
  | "specialist_handoff"
  | "review_team"
  | "human_agent_board"
  | "remote_teammate"
  | "unknown";

export type AgentUiRuntimeStatus =
  | "idle"
  | "queued"
  | "submitted"
  | "accepted"
  | "preparing"
  | "running"
  | "waiting"
  | "needs_input"
  | "plan_ready"
  | "completed"
  | "failed"
  | "aborted"
  | "cancelled"
  | "closed"
  | "not_found"
  | "unknown";

export type AgentUiProjectionSourceType =
  | "action_required"
  | "action_resolved"
  | "artifact_snapshot"
  | "evidence_projection"
  | "hydration_projection"
  | "performance_metric"
  | "queue_added"
  | "remote_task_projection"
  | "runtime_status"
  | "team_control_projection"
  | "text_delta"
  | "thinking_delta"
  | "tool_end"
  | "tool_input_delta"
  | "tool_output_delta"
  | "tool_progress"
  | "tool_start"
  | string;

export interface AgentUiProjectionRefs {
  artifactIds?: string[];
  artifactPaths?: string[];
  contextSourceIds?: string[];
  teamMemoryKeys?: string[];
  diagnosticKeys?: string[];
  rawEventRef?: string;
}

export interface AgentUiProjectionEvent {
  type: AgentUiEventClass;
  sourceType: AgentUiProjectionSourceType;
  sequence?: number;
  timestamp?: string;
  sessionId?: string;
  threadId?: string;
  runId?: string;
  turnId?: string;
  messageId?: string;
  partId?: string;
  taskId?: string;
  toolCallId?: string;
  actionId?: string;
  artifactId?: string;
  evidenceId?: string;
  agentId?: string;
  diagnosticId?: string;
  owner: AgentUiOwner;
  scope: AgentUiScope;
  phase: AgentUiPhase;
  surface?: AgentUiSurface;
  persistence?: AgentUiPersistence;
  control?: AgentUiControl;
  parentThreadId?: string;
  agentName?: string;
  teamName?: string;
  teamId?: string;
  agentRole?: string;
  agentSource?: string;
  workerNotificationId?: string;
  remoteTaskId?: string;
  transcriptRef?: string;
  topology?: AgentUiTopology;
  runtimeEntity?: AgentUiRuntimeEntity;
  runtimeStatus?: AgentUiRuntimeStatus;
  latestTurnStatus?: AgentUiRuntimeStatus;
  teamPhase?: string;
  teamParallelBudget?: number;
  teamActiveCount?: number;
  teamQueuedCount?: number;
  queuedTurnCount?: number;
  queueReason?: string;
  providerConcurrencyGroup?: string;
  providerParallelBudget?: number;
  retryableOverload?: boolean;
  workItemId?: string;
  reviewId?: string;
  handoffId?: string;
  workerUsage?: Record<string, unknown> | null;
  teamPolicy?: Record<string, unknown> | null;
  payload?: Record<string, unknown>;
  refs?: AgentUiProjectionRefs;
  rawEventRef?: string;
}

export interface AgentUiProjectionContext {
  sequence?: number;
  timestamp?: string;
  sessionId?: string | null;
  threadId?: string | null;
  runId?: string | null;
  turnId?: string | null;
  messageId?: string | null;
  taskId?: string | null;
  runtimeEntity?: AgentUiRuntimeEntity | null;
}
