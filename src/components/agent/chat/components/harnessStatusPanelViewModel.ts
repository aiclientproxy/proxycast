import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import type { StepStatus } from "@/lib/workspace/workbenchContract";
import type { ActionRequired } from "../types";
import type { HarnessSessionState } from "../utils/harnessState";
import { isRuntimeStatusDiagnosticsOnly } from "../utils/turnSummaryPresentation";
import {
  normalizeToolNameKey,
  resolveToolDisplayLabel,
} from "../utils/toolDisplayInfo";
import {
  buildWorkflowSummaryText,
  getWorkflowStatusLabel,
} from "../utils/workflowStepPresentation";
import {
  pickCommandFromArguments,
  pickPathFromArguments,
} from "./harnessStatusPanelTextParsing";

export * from "./harnessFileReviewViewModel";
export * from "./harnessOutputSignalViewModel";
export * from "./harnessToolInventoryViewModel";
export {
  createUrlPattern,
  findFirstUrl,
  isLikelyFilePath,
  normalizeUrlCandidate,
  pickCommandFromArguments,
  pickPathFromArguments,
  splitTextIntoSegments,
} from "./harnessStatusPanelTextParsing";
export type {
  NormalizedUrlCandidate,
  TextSegment,
} from "./harnessStatusPanelTextParsing";

export function formatIsoDateTime(value?: string): string {
  if (!value) {
    return "未知";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return parsed.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatSize(value?: number): string | null {
  if (!value || value <= 0) {
    return null;
  }

  if (value >= 1024 * 1024) {
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  }
  if (value >= 1024) {
    return `${Math.round(value / 1024)} KB`;
  }
  return `${value} B`;
}

export type HarnessStatusBadgeVariant =
  | "default"
  | "secondary"
  | "destructive"
  | "outline";

export interface ChildSubagentSessionSummary {
  total: number;
  running: number;
  queued: number;
  active: number;
  interrupted: number;
  settled: number;
  failed: number;
}

export type ApprovalRiskKind = "file_change" | "command" | "input" | "default";

export interface RuntimeTaskPresentation {
  title: string;
  summaryText: string;
  phaseLabel: string;
  statusLabel: string;
  progressLabel: string;
  stepStatus: StepStatus;
  checkpoints: string[];
}

export interface HarnessRuntimeFactSummary {
  threadId: string;
  status: string | null;
  turnCount: number;
  itemCount: number;
  pendingRequestCount: number;
  artifactCount: number;
  evidenceRefCount: number;
  decisionReason: string | null;
  fallbackChain: string[];
  oemPolicy: NonNullable<AgentRuntimeThreadReadModel["oem_policy"]> | null;
}

export function isFileMutationApproval(item: ActionRequired): boolean {
  const normalizedToolName = normalizeToolNameKey(item.toolName || "");
  return [
    "write",
    "writefile",
    "edit",
    "editfile",
    "multiedit",
    "createfile",
    "delete",
    "remove",
    "move",
    "patch",
    "applypatch",
  ].some((keyword) => normalizedToolName.includes(keyword));
}

export function resolveApprovalRiskKind(
  item: ActionRequired,
): ApprovalRiskKind {
  if (pickCommandFromArguments(item.arguments)) {
    return "command";
  }
  if (pickPathFromArguments(item.arguments) && isFileMutationApproval(item)) {
    return "file_change";
  }
  if (item.actionType === "ask_user" || item.actionType === "elicitation") {
    return "input";
  }
  return "default";
}

export function resolveApprovalActionLabelKey(item: ActionRequired): string {
  switch (item.actionType) {
    case "ask_user":
      return "agentChat.harness.approvals.action.askUser";
    case "elicitation":
      return "agentChat.harness.approvals.action.elicitation";
    case "tool_confirmation":
    default:
      return "agentChat.harness.approvals.action.tool";
  }
}

export function describeApproval(item: ActionRequired): string | undefined {
  const hints: string[] = [];

  if (item.toolName?.trim()) {
    hints.push(resolveFriendlyToolLabel(item.toolName) || item.toolName.trim());
  }

  const path = pickPathFromArguments(item.arguments);
  if (path) {
    hints.push(path);
  }

  const command = pickCommandFromArguments(item.arguments);
  if (command) {
    hints.push(command);
  }

  return hints.length > 0 ? hints.join(" · ") : undefined;
}

export function joinDisplayParts(
  parts: Array<string | null | undefined>,
): string | undefined {
  const normalized = parts
    .map((part) => part?.trim())
    .filter((part): part is string => Boolean(part));

  return normalized.length > 0 ? normalized.join(" · ") : undefined;
}

const APPROVAL_RISK_LABEL_KEY_BY_KIND: Record<ApprovalRiskKind, string> = {
  file_change: "agentChat.harness.approvals.risk.file_change",
  command: "agentChat.harness.approvals.risk.command",
  input: "agentChat.harness.approvals.risk.input",
  default: "agentChat.harness.approvals.risk.default",
};

export function resolveApprovalRiskLabelKey(kind: ApprovalRiskKind): string {
  return APPROVAL_RISK_LABEL_KEY_BY_KIND[kind] || kind;
}

export function resolveFriendlyToolLabel(value?: string): string | null {
  const normalized = value?.trim();
  if (!normalized) {
    return null;
  }

  if (normalizeToolNameKey(normalized) === "turnsummary") {
    return "当前任务摘要";
  }

  return resolveToolDisplayLabel(normalized);
}

export function formatRuntimePhaseLabel(
  runtimeStatus: HarnessSessionState["runtimeStatus"],
): string {
  if (!runtimeStatus) {
    return "空闲";
  }

  switch (runtimeStatus.phase) {
    case "preparing":
      return "准备中";
    case "routing":
      return "处理中";
    case "context":
      return "整理信息";
    case "cancelled":
      return "已取消";
    case "failed":
      return "需要处理";
    default:
      return runtimeStatus.phase;
  }
}

export function resolveRuntimeStepStatus(
  runtimeStatus: NonNullable<HarnessSessionState["runtimeStatus"]>,
): StepStatus {
  if (runtimeStatus.phase === "failed") {
    return "error";
  }
  if (runtimeStatus.phase === "cancelled") {
    return "skipped";
  }
  return "active";
}

export function resolveRuntimeStatusLabel(
  runtimeStatus: NonNullable<HarnessSessionState["runtimeStatus"]>,
): string {
  if (runtimeStatus.phase === "cancelled") {
    return "已取消";
  }
  return getWorkflowStatusLabel(resolveRuntimeStepStatus(runtimeStatus));
}

export function buildRuntimeSummaryText(
  runtimeStatus: NonNullable<HarnessSessionState["runtimeStatus"]>,
): string {
  const detail = runtimeStatus.detail?.trim();
  if (detail) {
    return detail;
  }
  if (runtimeStatus.phase === "cancelled") {
    return "当前流程已取消，可重新发起新的任务继续。";
  }
  return buildWorkflowSummaryText({
    leadingStep: {
      status: resolveRuntimeStepStatus(runtimeStatus),
    },
    remainingCount: 1,
    emptyLabel: "当前流程已完成",
  });
}

export function formatRuntimeProgressLabel(
  runtimeStatus: NonNullable<HarnessSessionState["runtimeStatus"]>,
  checkpoints: string[],
): string {
  if (checkpoints.length > 0) {
    return `已记录 ${checkpoints.length} 个任务节点`;
  }
  if (runtimeStatus.phase === "failed") {
    return "等待处理异常后重试";
  }
  if (runtimeStatus.phase === "cancelled") {
    return "当前流程已取消";
  }
  return "等待更多执行进展";
}

export function buildRuntimeTaskPresentation(
  runtimeStatus: HarnessSessionState["runtimeStatus"],
): RuntimeTaskPresentation | null {
  if (!runtimeStatus || isRuntimeStatusDiagnosticsOnly(runtimeStatus)) {
    return null;
  }

  const checkpoints = (runtimeStatus.checkpoints ?? [])
    .map((item) => item.trim())
    .filter((item) => item.length > 0);

  return {
    title: runtimeStatus.title?.trim() || "正在整理当前任务",
    summaryText: buildRuntimeSummaryText(runtimeStatus),
    phaseLabel: formatRuntimePhaseLabel(runtimeStatus),
    statusLabel: resolveRuntimeStatusLabel(runtimeStatus),
    progressLabel: formatRuntimeProgressLabel(runtimeStatus, checkpoints),
    stepStatus: resolveRuntimeStepStatus(runtimeStatus),
    checkpoints,
  };
}

export function buildRuntimeFactSummary(
  threadRead?: AgentRuntimeThreadReadModel | null,
): HarnessRuntimeFactSummary | null {
  const threadId = threadRead?.thread_id?.trim();
  if (!threadId) {
    return null;
  }

  const runtimeSummary = threadRead?.runtime_summary;
  const decisionReason =
    threadRead?.decision_reason ?? runtimeSummary?.decisionReason ?? null;
  const fallbackChain = Array.isArray(threadRead?.fallback_chain)
    ? threadRead?.fallback_chain || []
    : Array.isArray(runtimeSummary?.fallbackChain)
      ? runtimeSummary?.fallbackChain || []
      : [];
  const oemPolicy = threadRead?.oem_policy ?? null;

  return {
    threadId,
    status: threadRead?.status ?? null,
    turnCount: Array.isArray(threadRead?.turns) ? threadRead.turns.length : 0,
    itemCount: Array.isArray(threadRead?.thread_items)
      ? threadRead.thread_items.length
      : 0,
    pendingRequestCount: Array.isArray(threadRead?.pending_requests)
      ? threadRead.pending_requests.length
      : 0,
    artifactCount: Array.isArray(threadRead?.artifacts)
      ? threadRead.artifacts.length
      : 0,
    evidenceRefCount: Array.isArray(threadRead?.evidence_summary?.evidence_refs)
      ? threadRead.evidence_summary.evidence_refs.length
      : 0,
    decisionReason,
    fallbackChain,
    oemPolicy,
  };
}
