import type { AgentRuntimeFileCheckpointThreadSummary } from "@/lib/api/agentRuntime/sessionTypes";
import type { AgentThreadItem } from "@/lib/api/agentProtocolCoreTypes";
import type { CodingWorkbenchView } from "@limecloud/agent-runtime-projection";
import type { CanvasWorkbenchHeaderView } from "../components/CanvasWorkbenchLayout";
import type {
  CanvasWorkbenchChangeItem,
  CanvasWorkbenchChangeView,
} from "../components/canvas-workbench/changes/CanvasWorkbenchChangesPanelViewModel";
import { extractFileNameFromPath } from "./workspacePath";

export function resolvePathLeaf(value?: string | null): string {
  const normalized = (value || "").trim().replace(/\\/g, "/");
  if (!normalized) {
    return "";
  }
  const segments = normalized.split("/").filter(Boolean);
  return segments.at(-1) || normalized;
}

export function buildWorkspaceHeaderView({
  projectRootPath,
  workspacePathMissing,
  workspaceHealthError,
}: {
  projectRootPath?: string | null;
  workspacePathMissing?: boolean | null;
  workspaceHealthError?: boolean | null;
}): CanvasWorkbenchHeaderView {
  const workspaceRootLabel = resolvePathLeaf(projectRootPath) || "未绑定";
  const hasProjectRootPath = Boolean(projectRootPath?.trim());
  const workspaceBindingValue = workspacePathMissing
    ? "路径缺失"
    : workspaceHealthError
      ? "状态异常"
      : hasProjectRootPath
        ? "已连接"
        : "未绑定";

  return {
    eyebrow: "Project Workspace",
    tabLabel: "文件",
    tabBadge:
      workspacePathMissing || workspaceHealthError
        ? workspaceBindingValue
        : hasProjectRootPath
          ? workspaceRootLabel
          : undefined,
    tabBadgeTone:
      workspacePathMissing || workspaceHealthError
        ? "rose"
        : hasProjectRootPath
          ? "sky"
          : undefined,
    title: hasProjectRootPath ? "项目工作区文件" : "当前没有可浏览的项目文件",
    subtitle: hasProjectRootPath
      ? projectRootPath || ""
      : "绑定工作区目录后，这里会显示真实文件树。",
    badges: [
      {
        key: "workspace-root",
        label: hasProjectRootPath ? workspaceRootLabel : "未绑定工作区",
        tone: hasProjectRootPath ? "accent" : "default",
      },
      ...(workspacePathMissing
        ? [
            {
              key: "workspace-missing",
              label: "路径缺失",
              tone: "default" as const,
            },
          ]
        : workspaceHealthError
          ? [
              {
                key: "workspace-health-error",
                label: "状态异常",
                tone: "default" as const,
              },
            ]
          : []),
    ],
    summaryStats: [
      {
        key: "workspace-root",
        label: "工作区",
        value: workspaceRootLabel,
        detail:
          projectRootPath?.trim() || "绑定工作区后，这里会展示真实文件树。",
        tone: hasProjectRootPath ? "accent" : "default",
      },
      {
        key: "workspace-binding",
        label: "目录状态",
        value: workspaceBindingValue,
        detail: workspacePathMissing
          ? "当前工作区路径缺失，需重新选择目录。"
          : workspaceHealthError
            ? "当前工作区状态异常，建议先修复后再继续浏览。"
            : hasProjectRootPath
              ? "画布会直接读取项目里的真实文件。"
              : "尚未绑定工作区目录。",
        tone:
          workspacePathMissing || workspaceHealthError ? "default" : "success",
      },
    ],
    panelCopy: {
      unavailableText: "当前工作区路径不可用，暂时无法浏览项目文件。",
      emptyText: "当前会话没有绑定可浏览的工作区目录。",
      sectionEyebrow: "项目目录",
      loadingText: "正在加载目录...",
      emptyDirectoryText: "暂无目录内容。",
    },
  };
}

function statusForCanvasChange(
  status: string,
): CanvasWorkbenchChangeItem["status"] {
  if (status === "running" || status === "pending") return "in_progress";
  if (status === "failed") return "failed";
  return "completed";
}

export function buildCanvasWorkbenchChangeViewFromCodingProjection({
  codingView,
  fileCheckpointSummary,
  onOpenFile,
  threadItems,
  reviewThreadId,
  currentTurnId,
  turnDiff,
  canInterrupt,
  onStartReview,
}: {
  codingView: CodingWorkbenchView;
  fileCheckpointSummary?: AgentRuntimeFileCheckpointThreadSummary | null;
  onOpenFile?: (path: string) => void | Promise<void>;
  threadItems?: readonly AgentThreadItem[];
  reviewThreadId?: string | null;
  currentTurnId?: string | null;
  turnDiff?: string;
  canInterrupt?: boolean;
  onStartReview?: () => Promise<unknown>;
}): CanvasWorkbenchChangeView | null {
  const items = codingView.changes.map(
    (change): CanvasWorkbenchChangeItem => ({
      id: change.id,
      path: change.path,
      displayName: change.destinationPath
        ? `${extractFileNameFromPath(change.path)} -> ${extractFileNameFromPath(
            change.destinationPath,
          )}`
        : extractFileNameFromPath(change.path),
      source: "runtime",
      status: statusForCanvasChange(change.status),
      changeKind: change.changeKind,
      preview: change.preview,
      currentContent: change.preview ?? null,
      previousContent: null,
      checkpointPath: change.checkpointRef || null,
      checkpointLabel: change.checkpointRef ? "snapshot" : null,
    }),
  );
  if (
    items.length === 0 &&
    !(fileCheckpointSummary?.count ?? 0) &&
    turnDiff === undefined
  ) {
    return null;
  }
  const latestCheckpoint = fileCheckpointSummary?.latest_checkpoint || null;
  return {
    items,
    checkpointCount: fileCheckpointSummary?.count ?? 0,
    latestCheckpointPath:
      latestCheckpoint?.snapshot_path || latestCheckpoint?.path || null,
    onOpenFile,
    threadItems,
    reviewThreadId,
    currentTurnId,
    turnDiff,
    canInterrupt,
    onStartReview,
  };
}
