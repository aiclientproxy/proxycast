import type {
  AgentUiControl,
  AgentUiEventClass,
  AgentUiProjectionEvent,
  AgentUiSurface,
} from "./agentUiEventProjection";
import {
  summarizeAgentUiSubagentsSurfaceLanes as summarizeAgentUiSubagentsSurfaceLanesBase,
  summarizeAgentUiSubagentsSurfaces as summarizeAgentUiSubagentsSurfacesBase,
} from "@limecloud/agent-runtime-projection";
import type {
  AgentUiSubagentsSurfaceLaneSummary,
  AgentUiSubagentsSurfaceSummary,
} from "@limecloud/agent-runtime-projection";

export {
  AGENT_UI_ACTION_EVENT_TYPES,
  AGENT_UI_ARTIFACT_EVENT_TYPES,
  AGENT_UI_DIAGNOSTIC_EVENT_TYPES,
  AGENT_UI_EVIDENCE_EVENT_TYPES,
  AGENT_UI_NOTABLE_EVENT_TYPES,
  AGENT_UI_TASK_EVENT_TYPES,
  AGENT_UI_SUBAGENTS_EVENT_TYPES,
  AGENT_UI_SUBAGENTS_SURFACE_DEFINITIONS,
  AGENT_UI_SUBAGENTS_SURFACE_LANES,
  AGENT_UI_SUBAGENTS_SURFACES,
  EMPTY_AGENT_UI_PROJECTION_SUMMARY,
  EMPTY_AGENT_UI_SUBAGENTS_PROJECTION_SUMMARY,
  findLatestAgentUiProjectionEventForArtifact,
  summarizeAgentUiProjectionEvents,
  summarizeAgentUiSubagentsProjectionEvents,
} from "@limecloud/agent-runtime-projection";
export type {
  AgentUiProjectionSummary,
  AgentUiSubagentsProjectionSummary,
  AgentUiSubagentsSurfaceDefinition,
  AgentUiSubagentsSurfaceLaneDefinition,
  AgentUiSubagentsSurfaceLaneSummary,
  AgentUiSubagentsSurfaceSummary,
} from "@limecloud/agent-runtime-projection";

export type AgentUiProjectionTranslation = (
  key: string,
  options?: Record<string, unknown>,
) => string;

export const AGENT_UI_EVENT_LABELS: Partial<Record<AgentUiEventClass, string>> =
  {
    "action.required": "Action 等待",
    "action.resolved": "Action 已处理",
    "agent.changed": "Agent 状态",
    "agent.completed": "Agent 完成",
    "agent.handoff": "Agent 交接",
    "agent.spawned": "Agent 已创建",
    "artifact.preview.ready": "Artifact 预览",
    "artifact.updated": "Artifact 更新",
    "context.changed": "Context 更新",
    "context.compaction.completed": "Context 压缩完成",
    "context.compaction.started": "Context 压缩开始",
    "diagnostic.changed": "Diagnostics",
    "evidence.changed": "Evidence",
    "metric.changed": "Metric",
    "permission.changed": "Permission",
    "queue.changed": "Queue",
    "run.failed": "Run 失败",
    "run.finished": "Run 完成",
    "run.started": "Run 开始",
    "run.status": "Run 状态",
    "task.changed": "Task",
    "tool.failed": "Tool 失败",
    "tool.output.delta": "Tool 输出",
    "tool.progress": "Tool 进度",
    "tool.result": "Tool 结果",
    "tool.started": "Tool 开始",
    "worker.notification": "Worker 通知",
    "review.requested": "Review 请求",
    "review.completed": "Review 完成",
  };

const AGENT_UI_EVENT_LABEL_KEYS: Partial<Record<AgentUiEventClass, string>> =
  Object.fromEntries(
    Object.keys(AGENT_UI_EVENT_LABELS).map((type) => [
      type,
      `agentChat.agentUiProjection.eventType.${type}`,
    ]),
  ) as Partial<Record<AgentUiEventClass, string>>;

const AGENT_UI_SOURCE_TYPE_LABELS: Record<string, string> = {
  action_required: "等待操作",
  artifact_snapshot: "产物快照",
  evidence_projection: "证据记录",
  item_completed: "历史正文",
  performance_metric: "性能指标",
  patch: "代码变更",
  queue_added: "队列更新",
  remote_task_projection: "远端任务",
  runtime_status: "运行状态",
  team_control_projection: "Subagents 控制",
  tool_call: "工具调用",
  tool_start: "工具开始",
};

const AGENT_UI_SOURCE_TYPE_LABEL_KEYS: Record<string, string> =
  Object.fromEntries(
    Object.keys(AGENT_UI_SOURCE_TYPE_LABELS).map((sourceType) => [
      sourceType,
      `agentChat.agentUiProjection.sourceType.${sourceType}`,
    ]),
  );

const AGENT_UI_PHASE_LABELS: Record<string, string> = {
  accepted: "已接受",
  acting: "执行中",
  archived: "已归档",
  cancelled: "已取消",
  completed: "已完成",
  failed: "失败",
  hydrating: "恢复中",
  interrupted: "已中断",
  planning: "规划中",
  preparing: "准备中",
  producing: "生成中",
  reasoning: "思考中",
  reconciling: "对齐中",
  reviewing: "评审中",
  routing: "路由中",
  submitted: "已提交",
  unknown: "未知",
  waiting: "等待中",
};

const AGENT_UI_PHASE_LABEL_KEYS: Record<string, string> = Object.fromEntries(
  Object.keys(AGENT_UI_PHASE_LABELS).map((phase) => [
    phase,
    `agentChat.agentUiProjection.phase.${phase}`,
  ]),
);

export const AGENT_UI_CONTROL_LABELS: Partial<Record<AgentUiControl, string>> =
  {
    answer: "补充输入",
    approve: "批准",
    assign: "指派",
    close: "关闭",
    continue_agent: "继续",
    delegate: "委派",
    edit: "编辑",
    export: "导出",
    interrupt: "中断",
    none: "无控制",
    open_detail: "打开详情",
    queue: "加入队列",
    reject: "拒绝",
    remove: "移除",
    request_review: "请求审核",
    retry: "重试",
    rollback: "回滚",
    send: "发送",
    steer: "调整方向",
    stop: "停止",
    unknown: "未知控制",
    wait: "等待",
  };

const AGENT_UI_CONTROL_LABEL_KEYS: Partial<Record<AgentUiControl, string>> =
  Object.fromEntries(
    Object.keys(AGENT_UI_CONTROL_LABELS).map((control) => [
      control,
      `agentChat.agentUiProjection.control.${control}`,
    ]),
  ) as Partial<Record<AgentUiControl, string>>;

function translateAgentUiProjectionLabel(
  t: AgentUiProjectionTranslation | undefined,
  key: string | undefined,
  fallback: string,
): string {
  if (!t || !key) {
    return fallback;
  }
  return t(key, { defaultValue: fallback });
}

export function formatAgentUiProjectionEventType(
  type: AgentUiEventClass,
  t?: AgentUiProjectionTranslation,
): string {
  return translateAgentUiProjectionLabel(
    t,
    AGENT_UI_EVENT_LABEL_KEYS[type],
    AGENT_UI_EVENT_LABELS[type] || type,
  );
}

export function formatAgentUiProjectionSourceType(
  sourceType?: string | null,
  t?: AgentUiProjectionTranslation,
): string {
  const normalized = sourceType?.trim();
  if (!normalized) {
    return translateAgentUiProjectionLabel(
      t,
      "agentChat.agentUiProjection.sourceType.unknown",
      "事件来源",
    );
  }
  return translateAgentUiProjectionLabel(
    t,
    AGENT_UI_SOURCE_TYPE_LABEL_KEYS[normalized],
    AGENT_UI_SOURCE_TYPE_LABELS[normalized] ?? "事件来源",
  );
}

export function formatAgentUiProjectionPhase(
  phase: string,
  t?: AgentUiProjectionTranslation,
): string {
  return translateAgentUiProjectionLabel(
    t,
    AGENT_UI_PHASE_LABEL_KEYS[phase],
    AGENT_UI_PHASE_LABELS[phase] ?? phase,
  );
}

export function formatAgentUiProjectionControl(
  control: AgentUiControl,
  t?: AgentUiProjectionTranslation,
): string {
  return translateAgentUiProjectionLabel(
    t,
    AGENT_UI_CONTROL_LABEL_KEYS[control],
    AGENT_UI_CONTROL_LABELS[control] ?? control,
  );
}

export function formatAgentUiProjectionSurfaceLabel(
  surface: AgentUiSurface | string | null | undefined,
  t?: AgentUiProjectionTranslation,
  fallback = "工作区",
): string {
  const normalized = surface?.trim();
  if (!normalized) {
    return fallback;
  }
  return translateAgentUiProjectionLabel(
    t,
    `agentChat.agentUiProjection.surface.${normalized}.label`,
    fallback,
  );
}

export function formatAgentUiProjectionSurfaceDescription(
  surface: AgentUiSurface | string | null | undefined,
  t?: AgentUiProjectionTranslation,
  fallback = "",
): string {
  const normalized = surface?.trim();
  if (!normalized) {
    return fallback;
  }
  return translateAgentUiProjectionLabel(
    t,
    `agentChat.agentUiProjection.surface.${normalized}.description`,
    fallback,
  );
}

function isTeamReassignmentEvent(event: AgentUiProjectionEvent): boolean {
  return (
    event.surface === "work_board" &&
    readAgentUiProjectionPayloadValue(event, "taskEvent") ===
      "team_reassignment"
  );
}

function formatTeamReassignmentPrimaryDetail(
  event: AgentUiProjectionEvent,
): string {
  const target = event.workItemId ?? event.taskId ?? event.agentId;
  return target ? `重新指派：${target}` : "重新指派";
}

function formatTeamReassignmentAuxiliaryDetail(
  event: AgentUiProjectionEvent,
): string | null {
  const previousAssigneeId = readAgentUiProjectionPayloadValue(
    event,
    "previousAssigneeId",
  );
  const nextAssigneeId = readAgentUiProjectionPayloadValue(
    event,
    "nextAssigneeId",
  );
  const reassignmentReason = readAgentUiProjectionPayloadValue(
    event,
    "reassignmentReason",
  );
  const route =
    previousAssigneeId && nextAssigneeId
      ? `${previousAssigneeId} → ${nextAssigneeId}`
      : (nextAssigneeId ?? previousAssigneeId);
  const detailParts = [
    route ? `负责人：${route}` : null,
    reassignmentReason ? `原因：${reassignmentReason}` : null,
  ].filter(Boolean);

  return detailParts.length > 0 ? detailParts.join(" / ") : null;
}

function formatHandoffLaneAuxiliaryDetail(
  event: AgentUiProjectionEvent,
): string | null {
  const status = readAgentUiProjectionPayloadValue(event, "status");
  const from = readAgentUiProjectionPayloadValue(event, "from");
  const to = readAgentUiProjectionPayloadValue(event, "to");
  const resumeTarget = readAgentUiProjectionPayloadValue(event, "resumeTarget");
  const contextBoundary = readAgentUiProjectionPayloadValue(
    event,
    "contextBoundary",
  );
  const route = from && to ? `${from} → ${to}` : (to ?? from);
  const detailParts = [
    status ? `状态：${status}` : null,
    route ? `交接：${route}` : null,
    resumeTarget ? `恢复：${resumeTarget}` : null,
    contextBoundary ? `边界：${contextBoundary}` : null,
  ].filter(Boolean);

  return detailParts.length > 0 ? detailParts.join(" / ") : null;
}

function formatPlanApprovalAuxiliaryDetail(
  event: AgentUiProjectionEvent,
): string | null {
  if (
    readAgentUiProjectionPayloadValue(event, "actionType") !== "plan_approval"
  ) {
    return null;
  }

  const decisionKind = readAgentUiProjectionPayloadValue(event, "decisionKind");
  const from = readAgentUiProjectionPayloadValue(event, "from");
  const targetSessionId = readAgentUiProjectionPayloadValue(
    event,
    "targetSessionId",
  );
  const deliveryTarget = readAgentUiProjectionPayloadValue(
    event,
    "deliveryTarget",
  );
  const permissionMode = readAgentUiProjectionPayloadValue(
    event,
    "permissionMode",
  );
  const approved = readAgentUiProjectionPayloadValue(event, "approved");
  const awaitingLeaderApproval = readAgentUiProjectionPayloadValue(
    event,
    "awaitingLeaderApproval",
  );
  const planFilePath = readAgentUiProjectionPayloadValue(event, "planFilePath");
  const planFile = readAgentUiProjectionPayloadValue(event, "planFile");
  const planId = readAgentUiProjectionPayloadValue(event, "planId");
  const detailParts = [
    decisionKind ? `决策：${decisionKind}` : null,
    from ? `请求方：${from}` : null,
    targetSessionId ? `目标：${targetSessionId}` : null,
    deliveryTarget ? `投递：${deliveryTarget}` : null,
    permissionMode ? `权限：${permissionMode}` : null,
    approved ? `结果：${approved === "true" ? "已批准" : "已拒绝"}` : null,
    awaitingLeaderApproval === "true" ? "等待 leader 审批" : null,
    planFilePath ? `计划：${planFilePath}` : null,
    planFile ? `计划：${planFile}` : null,
    planId ? `Plan：${planId}` : null,
  ].filter(Boolean);

  return detailParts.length > 0 ? detailParts.join(" / ") : null;
}

export function readAgentUiProjectionPayloadText(
  event: AgentUiProjectionEvent,
): string | null {
  const payload = event.payload;
  if (!payload) {
    return null;
  }

  const candidates = [
    "summaryPreview",
    "status",
    "verdict",
    "taskEvent",
    "decisionKind",
    "actionType",
    "toolName",
    "kind",
    "reason",
    "phase",
  ];

  for (const key of candidates) {
    const value = payload[key];
    if (
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    ) {
      return `${value}`;
    }
  }

  return null;
}

export function formatAgentUiProjectionEventDetail(
  event: AgentUiProjectionEvent,
): string {
  if (isTeamReassignmentEvent(event)) {
    return formatTeamReassignmentPrimaryDetail(event);
  }

  return (
    readAgentUiProjectionPayloadText(event) ||
    event.actionId ||
    event.taskId ||
    event.artifactId ||
    event.evidenceId ||
    event.toolCallId ||
    event.sourceType
  );
}

function readAgentUiProjectionPayloadValue(
  event: AgentUiProjectionEvent,
  key: string,
): string | null {
  const value = event.payload?.[key];
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    const normalized = `${value}`.trim();
    return normalized || null;
  }

  return null;
}

function readAgentUiProjectionPayloadStringList(
  event: AgentUiProjectionEvent,
  key: string,
): string[] {
  const value = event.payload?.[key];
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => (typeof item === "string" ? item.trim() : ""))
    .filter(Boolean);
}

function formatPreviewListItem(label: string, values: string[]): string | null {
  const firstValue = values[0];
  if (!firstValue) {
    return null;
  }
  const suffix = values.length > 1 ? ` +${values.length - 1}` : "";
  return `${label}：${firstValue}${suffix}`;
}

export function formatAgentUiProjectionEventAuxiliaryDetail(
  event: AgentUiProjectionEvent,
): string | null {
  if (isTeamReassignmentEvent(event)) {
    return formatTeamReassignmentAuxiliaryDetail(event);
  }

  if (event.surface === "handoff_lane" || event.type === "agent.handoff") {
    return formatHandoffLaneAuxiliaryDetail(event);
  }

  if (event.type === "action.required" || event.type === "action.resolved") {
    return formatPlanApprovalAuxiliaryDetail(event);
  }

  if (
    event.surface !== "review_lane" &&
    event.type !== "review.requested" &&
    event.type !== "review.completed"
  ) {
    return null;
  }

  const decisionStatus = readAgentUiProjectionPayloadValue(
    event,
    "decisionStatus",
  );
  const reviewer = readAgentUiProjectionPayloadValue(event, "reviewer");
  const riskLevel = readAgentUiProjectionPayloadValue(event, "riskLevel");
  const checklistCount = readAgentUiProjectionPayloadValue(
    event,
    "checklistCount",
  );
  const followupActionCount = readAgentUiProjectionPayloadValue(
    event,
    "followupActionCount",
  );
  const regressionRequirementCount = readAgentUiProjectionPayloadValue(
    event,
    "regressionRequirementCount",
  );
  const requestedFixes = readAgentUiProjectionPayloadStringList(
    event,
    "requestedFixes",
  );
  const regressionRequirements = readAgentUiProjectionPayloadStringList(
    event,
    "regressionRequirements",
  );
  const detailParts = [
    decisionStatus ? `决策：${decisionStatus}` : null,
    reviewer ? `审核人：${reviewer}` : null,
    riskLevel ? `风险：${riskLevel}` : null,
    checklistCount ? `清单 ${checklistCount}` : null,
    followupActionCount ? `后续 ${followupActionCount}` : null,
    regressionRequirementCount ? `回归 ${regressionRequirementCount}` : null,
    formatPreviewListItem("修复", requestedFixes),
    formatPreviewListItem("回归项", regressionRequirements),
  ].filter(Boolean);

  return detailParts.length > 0 ? detailParts.join(" / ") : null;
}

export function summarizeAgentUiSubagentsSurfaceLanes(
  events: AgentUiProjectionEvent[],
  options: { t?: AgentUiProjectionTranslation } = {},
): AgentUiSubagentsSurfaceLaneSummary[] {
  return summarizeAgentUiSubagentsSurfaceLanesBase(events).map((lane) => ({
    ...lane,
    label: translateAgentUiProjectionLabel(
      options.t,
      `agentChat.agentUiProjection.lane.${lane.id}.label`,
      lane.label,
    ),
    description: translateAgentUiProjectionLabel(
      options.t,
      `agentChat.agentUiProjection.lane.${lane.id}.description`,
      lane.description,
    ),
  }));
}

export function summarizeAgentUiSubagentsSurfaces(
  events: AgentUiProjectionEvent[],
  options: {
    latestLimit?: number;
    t?: AgentUiProjectionTranslation;
  } = {},
): AgentUiSubagentsSurfaceSummary[] {
  return summarizeAgentUiSubagentsSurfacesBase(events, {
    latestLimit: options.latestLimit,
  }).map((surface) => ({
    ...surface,
    label: formatAgentUiProjectionSurfaceLabel(
      surface.surface,
      options.t,
      surface.label,
    ),
    description: formatAgentUiProjectionSurfaceDescription(
      surface.surface,
      options.t,
      surface.description,
    ),
  }));
}
