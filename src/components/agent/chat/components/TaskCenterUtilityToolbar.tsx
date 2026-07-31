import React from "react";
import type { ThreadGoal } from "@limecloud/app-server-client";
import {
  ChevronDown,
  CircleDot,
  Code2,
  FileText,
  FolderOpen,
  GitBranch,
  GitCommitHorizontal,
  Globe2,
  Monitor,
  PanelRightClose,
  PanelRightOpen,
  PanelRight,
  SlidersHorizontal,
  SquareTerminal,
  UserRound,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  openProjectPathWithTool,
  type ProjectPathOpenTool,
} from "@/lib/api/fileSystem";
import {
  readProjectGitStatus,
  type ProjectGitStatus,
} from "@/lib/api/projectGit";
import { cn } from "@/lib/utils";
import { agentText } from "./harnessPanelText";
import type { AgentSessionExecutionRuntime } from "@/lib/api/agentExecutionRuntime";
import type {
  AgentRuntimeThreadReadModel,
  AgentTodoItem,
} from "@/lib/api/agentRuntime/sessionTypes";
import type { CanonicalChildThreadSummary } from "../projection/canonicalChildThreadSummary";
import type {
  ActionRequired,
  AgentThreadItem,
  ConfirmResponse,
  Message,
} from "../types";
import {
  buildGeneralWorkbenchTaskRailProjection,
  type GeneralWorkbenchTaskRailContextInput,
} from "./generalWorkbenchTaskRailViewModel";
import {
  buildGeneralWorkbenchRunControlSurfaceProjection,
  type GeneralWorkbenchRunControlEnvironmentInput,
  type GeneralWorkbenchRunControlSplitLaneInput,
} from "./generalWorkbenchRunControlSurfaceViewModel";
import type { SidebarActivityLog } from "../hooks/useThemeContextWorkspace";
import {
  calculateWorkflowProgressPercent,
  countCompletedWorkflowSteps,
  type GeneralWorkbenchWorkflowStepInput,
} from "./generalWorkbenchWorkflowPanelViewModel";
import {
  buildGeneralWorkbenchActivityLogGroups,
  buildGeneralWorkbenchCreationTaskGroups,
  type GeneralWorkbenchCreationTaskEvent,
} from "./generalWorkbenchWorkflowData";
import { TaskCenterTaskRail } from "./TaskCenterTaskRail";
import { hydrateAgentPlanState } from "../utils/planState";
import type { WorkspaceRightSurfaceLauncherProjection } from "../workspace/right-surface";
import { buildWorkspaceTaskRailRuntimeContext } from "../workspace/useWorkspaceTaskRailRuntime";

interface TaskCenterUtilityToolbarProps {
  projectRootPath?: string | null;
  taskRail?: {
    sessionId?: string | null;
    workflowSteps: GeneralWorkbenchWorkflowStepInput[];
    messages: Message[];
    activityLogs?: SidebarActivityLog[];
    creationTaskEvents?: GeneralWorkbenchCreationTaskEvent[];
    pendingActions?: readonly ActionRequired[];
    submittedActionsInFlight?: readonly ActionRequired[];
    threadItems?: readonly AgentThreadItem[];
    todoItems?: readonly AgentTodoItem[];
    threadGoal?: ThreadGoal | null;
    threadRead?: AgentRuntimeThreadReadModel | null;
    executionRuntime?: AgentSessionExecutionRuntime | null;
    canonicalChildren?: CanonicalChildThreadSummary[];
    context?: GeneralWorkbenchTaskRailContextInput;
    providerType?: string | null;
    model?: string | null;
    accessMode?: GeneralWorkbenchTaskRailContextInput["accessMode"];
    reasoningEffort?: string | null;
    workspaceRootPath?: string | null;
    onOpenOutput?: (path: string) => void | Promise<void>;
    onRespondToAction?: (response: ConfirmResponse) => void | Promise<void>;
  };
  placement?: "task-strip" | "workbench-header";
  showCanvasToggle: boolean;
  isCanvasOpen: boolean;
  onToggleCanvas?: () => void;
  showHarnessToggle: boolean;
  harnessPanelVisible: boolean;
  onToggleHarnessPanel?: () => void;
  showExpertInfoToggle?: boolean;
  expertInfoPanelVisible?: boolean;
  onToggleExpertInfoPanel?: () => void;
  harnessPendingCount: number;
  harnessAttentionLevel: "idle" | "active" | "warning";
  harnessToggleLabel: string;
  shellPanelOpen: boolean;
  onToggleShellPanel?: () => void;
  onToggleBrowserPanel?: () => void;
  onToggleFilesPanel?: () => void;
  onToggleTracePanel?: () => void;
  onToggleObjectCanvasPanel?: () => void;
  rightSurfaceLaunchers?: readonly WorkspaceRightSurfaceLauncherProjection[];
}

const taskCenterToolButtonClassName =
  "inline-flex h-7 shrink-0 items-center justify-center whitespace-nowrap rounded-[12px] border border-[color:var(--lime-chrome-border)] bg-[color:var(--lime-surface)] px-2 leading-none text-[color:var(--lime-chrome-text)] shadow-none transition-[background-color,color] hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-text-strong)]";

const taskCenterIconOnlyButtonClassName =
  "inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-[12px] border border-transparent bg-transparent leading-none text-[color:var(--lime-chrome-muted)] shadow-none transition-[background-color,color] hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-chrome-text)] disabled:cursor-not-allowed disabled:opacity-50";

const taskCenterToolGroupClassName =
  "inline-flex max-w-full shrink-0 flex-wrap items-center gap-1 overflow-visible";

function VisualStudioCodeIcon({ className }: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "inline-flex h-4 w-4 items-center justify-center rounded-[5px] bg-[#f5fbff] text-[#0a84ff]",
        className,
      )}
    >
      <Code2 className="h-3 w-3" strokeWidth={2.3} />
    </span>
  );
}

function extractErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function useProjectGitStatus(rootPath?: string | null) {
  const normalizedRootPath = rootPath?.trim() || null;
  const [status, setStatus] = React.useState<ProjectGitStatus | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!normalizedRootPath) {
      setStatus(null);
      setLoading(false);
      setError(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    void readProjectGitStatus(normalizedRootPath)
      .then((nextStatus) => {
        if (cancelled) return;
        setStatus(nextStatus);
      })
      .catch((readError) => {
        if (cancelled) return;
        setStatus(null);
        setError(extractErrorMessage(readError));
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [normalizedRootPath]);

  return { status, loading, error };
}

export function TaskCenterUtilityToolbar({
  projectRootPath,
  taskRail,
  placement = "task-strip",
  showCanvasToggle,
  isCanvasOpen,
  onToggleCanvas,
  showHarnessToggle,
  harnessPanelVisible,
  onToggleHarnessPanel,
  showExpertInfoToggle = false,
  expertInfoPanelVisible = false,
  onToggleExpertInfoPanel,
  harnessPendingCount,
  harnessAttentionLevel,
  harnessToggleLabel,
  shellPanelOpen,
  onToggleShellPanel,
  onToggleBrowserPanel,
  onToggleFilesPanel,
  onToggleTracePanel,
  onToggleObjectCanvasPanel,
  rightSurfaceLaunchers,
}: TaskCenterUtilityToolbarProps) {
  const { t } = useTranslation("agent");
  const normalizedProjectRootPath = projectRootPath?.trim() || null;
  const [environmentVisited, setEnvironmentVisited] = React.useState(false);
  const [environmentOpen, setEnvironmentOpen] = React.useState(false);
  const autoRevealedPlanKeyRef = React.useRef<string | null>(null);
  const { status, loading, error } = useProjectGitStatus(
    environmentVisited ? normalizedProjectRootPath : null,
  );
  const shouldRenderHarnessToggle =
    showHarnessToggle || Boolean(onToggleHarnessPanel);
  const isWorkbenchHeaderPlacement = placement === "workbench-header";
  const rightSurfaceLauncherByKind = React.useMemo(
    () =>
      new Map(
        (rightSurfaceLaunchers ?? []).map((projection) => [
          projection.kind,
          projection,
        ]),
      ),
    [rightSurfaceLaunchers],
  );
  const workbenchLauncher = rightSurfaceLauncherByKind.get("workbench");
  const expertInfoLauncher = rightSurfaceLauncherByKind.get("expertInfo");
  const shellLauncher = rightSurfaceLauncherByKind.get("shell");
  const harnessLauncher = rightSurfaceLauncherByKind.get("harness");
  const filesLauncher = rightSurfaceLauncherByKind.get("files");
  const browserLauncher = rightSurfaceLauncherByKind.get("browser");
  const traceLauncher = rightSurfaceLauncherByKind.get("trace");
  const articleEditorLauncher =
    rightSurfaceLauncherByKind.get("articleWorkspace");
  const objectCanvasLauncher = rightSurfaceLauncherByKind.get("objectCanvas");
  const articleEditorHasVisibleSignal = Boolean(
    articleEditorLauncher?.active ||
    (articleEditorLauncher?.pendingCount ?? 0) > 0 ||
    (!objectCanvasLauncher && articleEditorLauncher),
  );
  const objectProfileLauncher = articleEditorHasVisibleSignal
    ? articleEditorLauncher
    : (objectCanvasLauncher ?? articleEditorLauncher);
  const objectProfilePendingCount =
    (articleEditorLauncher?.pendingCount ?? 0) +
    (objectCanvasLauncher?.pendingCount ?? 0);
  const objectProfileActive = Boolean(
    articleEditorLauncher?.active || objectCanvasLauncher?.active,
  );
  const objectProfileDisabled = Boolean(objectProfileLauncher?.disabled);
  const objectProfileLabelKey = articleEditorHasVisibleSignal
    ? "agentChat.navbar.articleWorkspace"
    : "agentChat.navbar.objectCanvas";
  const objectProfileDefaultLabel = articleEditorHasVisibleSignal
    ? "文章编辑器"
    : "对象画布";
  const openObjectProfileLabelKey = articleEditorHasVisibleSignal
    ? "agentChat.navbar.openArticleWorkspace"
    : "agentChat.navbar.openObjectCanvas";
  const closeObjectProfileLabelKey = articleEditorHasVisibleSignal
    ? "agentChat.navbar.closeArticleWorkspace"
    : "agentChat.navbar.closeObjectCanvas";
  const shouldRenderObjectCanvasToggle =
    Boolean(onToggleObjectCanvasPanel) &&
    Boolean(objectProfileLauncher) &&
    (!objectProfileLauncher?.disabled ||
      objectProfileActive ||
      objectProfilePendingCount > 0);
  const shouldRenderFilesToggle =
    Boolean(onToggleFilesPanel) &&
    Boolean(filesLauncher) &&
    (!filesLauncher?.disabled ||
      Boolean(filesLauncher?.active) ||
      (filesLauncher?.pendingCount ?? 0) > 0);
  const shouldRenderTraceToggle =
    Boolean(onToggleTracePanel) &&
    Boolean(traceLauncher) &&
    (!traceLauncher?.disabled ||
      Boolean(traceLauncher?.active) ||
      (traceLauncher?.pendingCount ?? 0) > 0);
  const shouldRenderBrowserToggle =
    Boolean(onToggleBrowserPanel) &&
    Boolean(browserLauncher) &&
    (!browserLauncher?.disabled ||
      Boolean(browserLauncher?.active) ||
      (browserLauncher?.pendingCount ?? 0) > 0);
  const shouldRenderPanelToolGroup =
    shouldRenderHarnessToggle ||
    showExpertInfoToggle ||
    showCanvasToggle ||
    shouldRenderObjectCanvasToggle ||
    shouldRenderTraceToggle ||
    shouldRenderBrowserToggle ||
    shouldRenderFilesToggle;
  const effectiveCanvasOpen = workbenchLauncher?.active ?? isCanvasOpen;
  const workbenchPendingCount = workbenchLauncher?.pendingCount ?? 0;
  const effectiveShellPanelOpen = shellLauncher?.active ?? shellPanelOpen;
  const shellPendingCount = shellLauncher?.pendingCount ?? 0;
  const effectiveObjectCanvasPanelOpen = objectProfileActive;
  const objectCanvasPendingCount = objectProfilePendingCount;
  const effectiveFilesPanelOpen = Boolean(filesLauncher?.active);
  const filesPendingCount = filesLauncher?.pendingCount ?? 0;
  const effectiveBrowserPanelOpen = Boolean(browserLauncher?.active);
  const browserPendingCount = browserLauncher?.pendingCount ?? 0;
  const effectiveTracePanelOpen = traceLauncher?.active ?? false;
  const tracePendingCount = traceLauncher?.pendingCount ?? 0;
  const effectiveExpertInfoPanelVisible =
    expertInfoLauncher?.active ?? expertInfoPanelVisible;
  const expertInfoPendingCount = expertInfoLauncher?.pendingCount ?? 0;
  const effectiveHarnessPanelVisible =
    harnessPanelVisible || Boolean(harnessLauncher?.active);
  const effectiveHarnessPendingCount = Math.max(
    harnessPendingCount,
    harnessLauncher?.pendingCount ?? 0,
  );
  const expertInfoToggleLabel = agentText(
    effectiveExpertInfoPanelVisible
      ? "agentChat.navbar.closeExpertInfo"
      : "agentChat.navbar.openExpertInfo",
    effectiveExpertInfoPanelVisible ? "关闭专家信息" : "打开专家信息",
  );

  const handleOpenTool = React.useCallback(
    async (tool: ProjectPathOpenTool) => {
      if (!normalizedProjectRootPath) {
        toast.error(
          agentText(
            "agentChat.navbar.appSwitcher.toast.noProjectRoot",
            "当前项目缺少本地目录",
          ),
        );
        return;
      }

      try {
        await openProjectPathWithTool(normalizedProjectRootPath, tool);
      } catch (openError) {
        toast.error(
          agentText(
            "agentChat.navbar.appSwitcher.toast.openFailed",
            "打开项目失败：{{message}}",
            { message: extractErrorMessage(openError) },
          ),
        );
      }
    },
    [normalizedProjectRootPath],
  );

  const branchLabel =
    status?.currentBranch?.trim() ||
    agentText("agentChat.navbar.environment.branchFallback", "无分支");
  const changeCount = status?.hasGitRepository
    ? status.uncommittedFileCount
    : 0;
  const environmentStatusLabel = loading
    ? agentText("agentChat.navbar.environment.loading", "读取中")
    : error
      ? agentText("agentChat.navbar.environment.failed", "读取失败")
      : !normalizedProjectRootPath
        ? agentText(
            "agentChat.navbar.environment.noProjectRoot",
            "未选择项目目录",
          )
        : status?.hasGitRepository
          ? agentText(
              "agentChat.navbar.environment.uncommittedFiles",
              "{{count}} 个文件",
              { count: changeCount },
            )
          : agentText("agentChat.navbar.environment.noGit", "非 Git 项目");
  const taskRailTranslate = React.useCallback(
    (key: string, options?: Record<string, unknown>) =>
      (
        t as (nextKey: string, nextOptions?: Record<string, unknown>) => unknown
      )(key, options),
    [t],
  );
  const taskRailProjection = React.useMemo(() => {
    if (!environmentOpen || !taskRail) {
      return null;
    }
    const completedSteps = countCompletedWorkflowSteps(taskRail.workflowSteps);
    const taskRailContext =
      taskRail.context ??
      buildWorkspaceTaskRailRuntimeContext({
        providerType: taskRail.providerType,
        model: taskRail.model,
        accessMode: taskRail.accessMode,
        reasoningEffort: taskRail.reasoningEffort,
        workspaceRootPath: taskRail.workspaceRootPath ?? null,
        threadGoal: taskRail.threadGoal,
        threadRead: taskRail.threadRead,
        threadItems: taskRail.threadItems,
        canonicalChildren: taskRail.canonicalChildren,
      });
    return buildGeneralWorkbenchTaskRailProjection({
      workflowSteps: taskRail.workflowSteps,
      completedSteps,
      progressPercent: calculateWorkflowProgressPercent({
        completedSteps,
        totalSteps: taskRail.workflowSteps.length,
      }),
      messages: taskRail.messages,
      groupedActivityLogs: buildGeneralWorkbenchActivityLogGroups(
        taskRail.activityLogs ?? [],
      ),
      groupedCreationTaskEvents: buildGeneralWorkbenchCreationTaskGroups(
        taskRail.creationTaskEvents ?? [],
      ),
      pendingActions: taskRail.pendingActions,
      submittedActionsInFlight: taskRail.submittedActionsInFlight,
      threadItems: taskRail.threadItems,
      todoItems: taskRail.todoItems,
      threadGoal: taskRail.threadGoal,
      threadRead: taskRail.threadRead,
      canonicalChildren: taskRail.canonicalChildren,
      context: taskRailContext,
      t: taskRailTranslate,
    });
  }, [environmentOpen, taskRail, taskRailTranslate]);
  React.useEffect(() => {
    if (environmentOpen || !taskRail) {
      return;
    }
    const todoPlanKey = (taskRail.todoItems ?? [])
      .filter((item) => item.content.trim().length > 0)
      .map((item) => `${item.content.trim()}\u0000${item.status ?? ""}`)
      .join("\u0001");
    const revisionId = taskRail.threadItems?.length
      ? hydrateAgentPlanState({ threadItems: taskRail.threadItems }).revisionId
      : null;
    const planKey = todoPlanKey
      ? `todo:${todoPlanKey}`
      : revisionId
        ? `revision:${revisionId}`
        : null;
    if (!planKey || autoRevealedPlanKeyRef.current === planKey) {
      return;
    }
    autoRevealedPlanKeyRef.current = planKey;
    setEnvironmentOpen(true);
    setEnvironmentVisited(true);
  }, [environmentOpen, taskRail?.threadItems, taskRail?.todoItems]);
  const runControlSurfaceProjection = React.useMemo(() => {
    if (!taskRailProjection) {
      return null;
    }

    const environment: GeneralWorkbenchRunControlEnvironmentInput = {
      modeLabel: normalizedProjectRootPath
        ? agentText("agentChat.navbar.environment.local", "本地")
        : null,
      branchLabel: status?.currentBranch?.trim() || null,
      gitStatusLabel: status
        ? agentText(
            "agentChat.navbar.environment.uncommittedFiles",
            "{{count}} 个文件",
            {
              count: status.hasGitRepository ? status.uncommittedFileCount : 0,
            },
          )
        : null,
    };
    const splitLane: GeneralWorkbenchRunControlSplitLaneInput = {
      state: shellPanelOpen
        ? "open"
        : showCanvasToggle
          ? "available"
          : "unavailable",
    };

    return buildGeneralWorkbenchRunControlSurfaceProjection({
      contextItems: taskRailProjection.contextItems,
      planItems: taskRailProjection.planItems,
      planRevision: taskRailProjection.planRevision,
      planOverflowCount: taskRailProjection.planOverflowCount,
      activityItems: taskRailProjection.activityItems,
      activityOverflowCount: taskRailProjection.activityOverflowCount,
      approvalItems: taskRailProjection.approvalItems,
      approvalOverflowCount: taskRailProjection.approvalOverflowCount,
      outputItems: taskRailProjection.outputItems.slice(0, 4),
      outputOverflowCount: taskRailProjection.outputOverflowCount,
      threadRead: taskRail?.threadRead,
      environment,
      splitLane,
      t: taskRailTranslate,
    });
  }, [
    normalizedProjectRootPath,
    shellPanelOpen,
    showCanvasToggle,
    status,
    taskRail,
    taskRailProjection,
    taskRailTranslate,
  ]);

  return (
    <div
      className={cn(
        "ml-auto flex min-w-0 shrink flex-wrap items-center justify-end gap-x-2 gap-y-1 overflow-visible",
        isWorkbenchHeaderPlacement ? "min-h-8" : "min-h-9 pb-1",
      )}
      data-testid="task-center-utility-toolbar"
      data-placement={placement}
    >
      <div
        className={taskCenterToolGroupClassName}
        data-testid="task-center-tool-group-app"
      >
        <Popover>
          <PopoverTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className={cn(taskCenterToolButtonClassName, "gap-1.5")}
              aria-label={agentText(
                "agentChat.navbar.appSwitcher.open",
                "打开应用切换",
              )}
              title={agentText(
                "agentChat.navbar.appSwitcher.open",
                "打开应用切换",
              )}
              data-testid="task-center-app-switcher-trigger"
            >
              <VisualStudioCodeIcon />
              <span>
                {agentText("agentChat.navbar.appSwitcher.open", "打开位置")}
              </span>
              <ChevronDown className="h-3 w-3" />
            </Button>
          </PopoverTrigger>
          <PopoverContent
            align="end"
            sideOffset={8}
            className="w-[min(11rem,calc(100vw-1rem))] rounded-2xl border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] p-2 text-[color:var(--lime-text)] shadow-xl shadow-slate-950/10"
            data-testid="task-center-app-switcher-popover"
          >
            <AppSwitcherAction
              icon={<VisualStudioCodeIcon className="h-3.5 w-3.5" />}
              label={agentText(
                "agentChat.navbar.appSwitcher.vscode",
                "VS Code",
              )}
              disabled={!normalizedProjectRootPath}
              onClick={() => void handleOpenTool("vscode")}
            />
            <AppSwitcherAction
              icon={<SquareTerminal className="h-3.5 w-3.5 text-slate-500" />}
              label={agentText("agentChat.navbar.appSwitcher.cursor", "Cursor")}
              disabled={!normalizedProjectRootPath}
              onClick={() => void handleOpenTool("cursor")}
            />
            <AppSwitcherAction
              icon={<FolderOpen className="h-3.5 w-3.5 text-sky-500" />}
              label={agentText("agentChat.navbar.appSwitcher.finder", "Finder")}
              disabled={!normalizedProjectRootPath}
              onClick={() => void handleOpenTool("finder")}
            />
            <AppSwitcherAction
              icon={<SquareTerminal className="h-3.5 w-3.5 text-slate-500" />}
              label={agentText(
                "agentChat.navbar.appSwitcher.terminal",
                "Terminal",
              )}
              disabled={!normalizedProjectRootPath}
              onClick={() => void handleOpenTool("terminal")}
            />
          </PopoverContent>
        </Popover>
      </div>

      <div
        className={taskCenterToolGroupClassName}
        data-testid="task-center-tool-group-environment"
      >
        <Popover
          open={environmentOpen}
          onOpenChange={(open) => {
            setEnvironmentOpen(open);
            if (open) {
              setEnvironmentVisited(true);
            }
          }}
        >
          <PopoverTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className={taskCenterIconOnlyButtonClassName}
              aria-label={agentText(
                "agentChat.navbar.environment.open",
                "打开环境信息",
              )}
              title={agentText(
                "agentChat.navbar.environment.open",
                "打开环境信息",
              )}
              data-testid="task-center-environment-trigger"
            >
              <SlidersHorizontal className="h-4 w-4" />
            </Button>
          </PopoverTrigger>
          <PopoverContent
            align="end"
            sideOffset={8}
            className="w-[min(30rem,calc(100vw-1rem))] rounded-3xl border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] p-4 text-[color:var(--lime-text)] shadow-xl shadow-slate-950/10"
            data-testid="task-center-environment-popover"
          >
            <div className="flex min-w-0 flex-wrap items-center justify-between gap-x-3 gap-y-1">
              <span className="min-w-0 text-xs font-medium text-[color:var(--lime-text-muted)]">
                {agentText("agentChat.navbar.environment.title", "环境信息")}
              </span>
              <span className="min-w-0 max-w-full truncate text-[11px] text-[color:var(--lime-text-muted)]">
                {environmentStatusLabel}
              </span>
            </div>
            <div className="mt-3 space-y-2 text-sm">
              <div className="flex min-w-0 flex-wrap items-center justify-between gap-x-3 gap-y-1">
                <span className="flex min-w-0 items-center gap-2">
                  <GitCommitHorizontal className="h-4 w-4" />
                  {agentText("agentChat.navbar.environment.changes", "变更")}
                </span>
                <span className="min-w-0 truncate text-xs text-[color:var(--lime-text-muted)]">
                  {status?.hasGitRepository
                    ? agentText(
                        "agentChat.navbar.environment.uncommittedFiles",
                        "{{count}} 个文件",
                        { count: changeCount },
                      )
                    : environmentStatusLabel}
                </span>
              </div>
              <div className="flex min-w-0 items-center gap-2">
                <Monitor className="h-4 w-4" />
                <span className="min-w-0 truncate">
                  {agentText("agentChat.navbar.environment.local", "本地")}
                </span>
              </div>
              <div className="flex min-w-0 items-center gap-2">
                <GitBranch className="h-4 w-4" />
                <span className="min-w-0 truncate">{branchLabel}</span>
              </div>
              <button
                type="button"
                className="flex w-full cursor-not-allowed items-center gap-2 rounded-xl py-1 text-left text-[color:var(--lime-text-muted)] opacity-70"
                disabled
                title={agentText(
                  "agentChat.navbar.environment.submitUnavailable",
                  "提交和推送需要后续接入 Git 写操作",
                )}
              >
                <CircleDot className="h-4 w-4" />
                <span>
                  {agentText(
                    "agentChat.navbar.environment.submit",
                    "提交或推送",
                  )}
                </span>
              </button>
            </div>
            {taskRailProjection ? (
              <TaskCenterTaskRail
                projection={taskRailProjection}
                runControlSurfaceProjection={runControlSurfaceProjection}
                onOpenOutput={taskRail?.onOpenOutput}
                onRespondToAction={taskRail?.onRespondToAction}
                t={taskRailTranslate}
              />
            ) : null}
          </PopoverContent>
        </Popover>
      </div>

      {shouldRenderPanelToolGroup ? (
        <div
          className={taskCenterToolGroupClassName}
          data-testid="task-center-tool-group-panels"
        >
          {shouldRenderHarnessToggle ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className={cn(
                taskCenterIconOnlyButtonClassName,
                "relative",
                effectiveHarnessPanelVisible &&
                  "bg-[color:var(--lime-chrome-tab-active-surface)] text-[color:var(--lime-text)]",
                harnessAttentionLevel === "warning" &&
                  !effectiveHarnessPanelVisible &&
                  "bg-[color:var(--lime-warning-soft)] text-[color:var(--lime-warning)] hover:bg-[color:var(--lime-warning-soft)] hover:text-[color:var(--lime-warning)]",
              )}
              disabled={harnessLauncher?.disabled}
              onClick={onToggleHarnessPanel}
              aria-label={
                effectiveHarnessPanelVisible
                  ? agentText(
                      "agentChat.navbar.closeHarness",
                      "关闭{{label}}",
                      {
                        label: harnessToggleLabel,
                      },
                    )
                  : agentText("agentChat.navbar.openHarness", "打开{{label}}", {
                      label: harnessToggleLabel,
                    })
              }
              aria-expanded={effectiveHarnessPanelVisible}
              title={harnessToggleLabel}
              data-testid="task-center-harness-toggle"
            >
              <Code2 className="h-4 w-4" />
              {effectiveHarnessPendingCount > 0 ? (
                <span className="absolute -right-1 -top-1 rounded-full border border-[color:var(--lime-surface-border-strong)] bg-[color:var(--lime-surface)] px-1 text-[9px] font-medium leading-4 text-[color:var(--lime-brand-strong)]">
                  {effectiveHarnessPendingCount > 99
                    ? "99+"
                    : effectiveHarnessPendingCount}
                </span>
              ) : null}
            </Button>
          ) : null}

          {showExpertInfoToggle ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className={cn(
                taskCenterIconOnlyButtonClassName,
                "relative",
                effectiveExpertInfoPanelVisible &&
                  "bg-[color:var(--lime-chrome-tab-active-surface)] text-[color:var(--lime-text)]",
              )}
              disabled={expertInfoLauncher?.disabled}
              onClick={onToggleExpertInfoPanel}
              aria-label={expertInfoToggleLabel}
              aria-expanded={effectiveExpertInfoPanelVisible}
              title={expertInfoToggleLabel}
              data-testid="task-center-expert-info-toggle"
            >
              <UserRound className="h-4 w-4" />
              {expertInfoPendingCount > 0 ? (
                <span className="absolute -right-1 -top-1 rounded-full border border-[color:var(--lime-surface-border-strong)] bg-[color:var(--lime-surface)] px-1 text-[9px] font-medium leading-4 text-[color:var(--lime-brand-strong)]">
                  {expertInfoPendingCount > 99 ? "99+" : expertInfoPendingCount}
                </span>
              ) : null}
            </Button>
          ) : null}

          {shouldRenderObjectCanvasToggle ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className={cn(
                taskCenterIconOnlyButtonClassName,
                "relative",
                effectiveObjectCanvasPanelOpen &&
                  "bg-[color:var(--lime-chrome-tab-active-surface)] text-[color:var(--lime-text)]",
              )}
              disabled={objectProfileDisabled}
              onClick={onToggleObjectCanvasPanel}
              aria-label={agentText(
                effectiveObjectCanvasPanelOpen
                  ? closeObjectProfileLabelKey
                  : openObjectProfileLabelKey,
                `${effectiveObjectCanvasPanelOpen ? "关闭" : "打开"}${objectProfileDefaultLabel}`,
              )}
              aria-expanded={effectiveObjectCanvasPanelOpen}
              title={agentText(
                objectProfileLabelKey,
                objectProfileDefaultLabel,
              )}
              data-testid="task-center-object-canvas-toggle"
            >
              <Monitor className="h-4 w-4" />
              {objectCanvasPendingCount > 0 ? (
                <span className="absolute -right-1 -top-1 rounded-full border border-[color:var(--lime-surface-border-strong)] bg-[color:var(--lime-surface)] px-1 text-[9px] font-medium leading-4 text-[color:var(--lime-brand-strong)]">
                  {objectCanvasPendingCount > 99
                    ? "99+"
                    : objectCanvasPendingCount}
                </span>
              ) : null}
            </Button>
          ) : null}

          {shouldRenderBrowserToggle ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className={cn(
                taskCenterIconOnlyButtonClassName,
                "relative",
                effectiveBrowserPanelOpen &&
                  "bg-[color:var(--lime-chrome-tab-active-surface)] text-[color:var(--lime-text)]",
              )}
              disabled={browserLauncher?.disabled}
              onClick={onToggleBrowserPanel}
              aria-label={agentText(
                effectiveBrowserPanelOpen
                  ? "agentChat.navbar.closeBrowser"
                  : "agentChat.navbar.openBrowser",
                effectiveBrowserPanelOpen ? "关闭浏览器" : "打开浏览器",
              )}
              aria-expanded={effectiveBrowserPanelOpen}
              title={agentText("agentChat.navbar.browser", "浏览器")}
              data-testid="task-center-browser-toggle"
            >
              <Globe2 className="h-4 w-4" />
              {browserPendingCount > 0 ? (
                <span className="absolute -right-1 -top-1 rounded-full border border-[color:var(--lime-surface-border-strong)] bg-[color:var(--lime-surface)] px-1 text-[9px] font-medium leading-4 text-[color:var(--lime-brand-strong)]">
                  {browserPendingCount > 99 ? "99+" : browserPendingCount}
                </span>
              ) : null}
            </Button>
          ) : null}

          {shouldRenderTraceToggle ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className={cn(
                taskCenterIconOnlyButtonClassName,
                "relative",
                effectiveTracePanelOpen &&
                  "bg-[color:var(--lime-chrome-tab-active-surface)] text-[color:var(--lime-text)]",
              )}
              disabled={traceLauncher?.disabled}
              onClick={onToggleTracePanel}
              aria-label={agentText(
                effectiveTracePanelOpen
                  ? "agentChat.navbar.closeTrace"
                  : "agentChat.navbar.openTrace",
                effectiveTracePanelOpen ? "关闭 Trace" : "打开 Trace",
              )}
              aria-expanded={effectiveTracePanelOpen}
              title={agentText("agentChat.navbar.trace", "Trace")}
              data-testid="task-center-trace-toggle"
            >
              <PanelRight className="h-4 w-4" />
              {tracePendingCount > 0 ? (
                <span className="absolute -right-1 -top-1 rounded-full border border-[color:var(--lime-surface-border-strong)] bg-[color:var(--lime-surface)] px-1 text-[9px] font-medium leading-4 text-[color:var(--lime-brand-strong)]">
                  {tracePendingCount > 99 ? "99+" : tracePendingCount}
                </span>
              ) : null}
            </Button>
          ) : null}

          {shouldRenderFilesToggle ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className={cn(
                taskCenterIconOnlyButtonClassName,
                "relative",
                effectiveFilesPanelOpen &&
                  "bg-[color:var(--lime-chrome-tab-active-surface)] text-[color:var(--lime-text)]",
              )}
              disabled={filesLauncher?.disabled}
              onClick={onToggleFilesPanel}
              aria-label={agentText(
                "agentChat.fileChangesSummary.openFile",
                "打开文件",
              )}
              aria-expanded={effectiveFilesPanelOpen}
              title={agentText("agentChat.canvasWorkbench.tabs.files", "文件")}
              data-testid="task-center-files-toggle"
            >
              <FileText className="h-4 w-4" />
              {filesPendingCount > 0 ? (
                <span className="absolute -right-1 -top-1 rounded-full border border-[color:var(--lime-surface-border-strong)] bg-[color:var(--lime-surface)] px-1 text-[9px] font-medium leading-4 text-[color:var(--lime-brand-strong)]">
                  {filesPendingCount > 99 ? "99+" : filesPendingCount}
                </span>
              ) : null}
            </Button>
          ) : null}

          {showCanvasToggle ? (
            <>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className={cn(
                  taskCenterIconOnlyButtonClassName,
                  "relative",
                  effectiveShellPanelOpen &&
                    "bg-[color:var(--lime-chrome-tab-active-surface)] text-[color:var(--lime-text)]",
                )}
                disabled={shellLauncher?.disabled ?? !normalizedProjectRootPath}
                aria-label={agentText(
                  "agentChat.navbar.openShell",
                  "打开 Shell",
                )}
                aria-expanded={effectiveShellPanelOpen}
                title={agentText("agentChat.navbar.openShell", "打开 Shell")}
                data-testid="task-center-shell-toggle"
                onClick={onToggleShellPanel}
              >
                <SquareTerminal className="h-4 w-4" />
                {shellPendingCount > 0 ? (
                  <span className="absolute -right-1 -top-1 rounded-full border border-[color:var(--lime-surface-border-strong)] bg-[color:var(--lime-surface)] px-1 text-[9px] font-medium leading-4 text-[color:var(--lime-brand-strong)]">
                    {shellPendingCount > 99 ? "99+" : shellPendingCount}
                  </span>
                ) : null}
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className={cn(
                  taskCenterIconOnlyButtonClassName,
                  "relative",
                  effectiveCanvasOpen &&
                    "bg-[color:var(--lime-chrome-tab-active-surface)] text-[color:var(--lime-text)]",
                )}
                disabled={workbenchLauncher?.disabled}
                onClick={onToggleCanvas}
                aria-label={agentText(
                  effectiveCanvasOpen
                    ? "agentChat.navbar.closeWorkbench"
                    : "agentChat.navbar.openWorkbench",
                  effectiveCanvasOpen ? "关闭工作台" : "打开工作台",
                )}
                title={agentText(
                  effectiveCanvasOpen
                    ? "agentChat.navbar.closeWorkbench"
                    : "agentChat.navbar.openWorkbench",
                  effectiveCanvasOpen ? "关闭工作台" : "打开工作台",
                )}
                data-testid="task-center-workbench-toggle"
              >
                {effectiveCanvasOpen ? (
                  <PanelRightClose className="h-4 w-4" />
                ) : (
                  <PanelRightOpen className="h-4 w-4" />
                )}
                {workbenchPendingCount > 0 ? (
                  <span className="absolute -right-1 -top-1 rounded-full border border-[color:var(--lime-surface-border-strong)] bg-[color:var(--lime-surface)] px-1 text-[9px] font-medium leading-4 text-[color:var(--lime-brand-strong)]">
                    {workbenchPendingCount > 99 ? "99+" : workbenchPendingCount}
                  </span>
                ) : null}
              </Button>
            </>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function AppSwitcherAction({
  icon,
  label,
  disabled,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="flex w-full items-center gap-2 rounded-xl px-2.5 py-2 text-left text-xs transition hover:bg-[color:var(--lime-surface-hover)] disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:bg-transparent"
      disabled={disabled}
      onClick={onClick}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}
