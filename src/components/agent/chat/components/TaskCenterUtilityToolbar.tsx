import React from "react";
import type { ThreadGoal } from "@limecloud/app-server-client";
import {
  ChevronDown,
  Code2,
  FileText,
  Globe2,
  PanelRightClose,
  PanelRightOpen,
  PanelRight,
  Activity,
  SlidersHorizontal,
  SquareTerminal,
  UserRound,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Popover, PopoverTrigger } from "@/components/ui/popover";
import { openProjectPathWithTool } from "@/lib/api/fileSystem";
import { openExternalUrlWithSystemBrowser } from "@/lib/api/externalUrl";
import {
  checkoutProjectGitBranch,
  createProjectGitWorktree,
  createProjectGitBranch,
  readProjectGitDiff,
  readProjectGitStatus,
  type ProjectGitStatus,
} from "@/lib/api/projectGit";
import { ensureProjectWorkspace } from "@/lib/api/project";
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
import type { GeneralWorkbenchTaskRailContextInput } from "./generalWorkbenchTaskRailViewModel";
import type { SidebarActivityLog } from "../hooks/useThemeContextWorkspace";
import type { GeneralWorkbenchWorkflowStepInput } from "./generalWorkbenchWorkflowPanelViewModel";
import type { GeneralWorkbenchCreationTaskEvent } from "./generalWorkbenchWorkflowData";
import { TaskCenterEnvironmentPanel } from "./TaskCenterEnvironmentPanel";
import { useThreadEnvironmentLifecycleStatus } from "../hooks/useThreadEnvironmentLifecycleStatus";
import { TaskCenterLocationPanel } from "./TaskCenterLocationPanel";
import { markProjectOpened } from "../hooks/agentProjectStorage";
import { hydrateAgentPlanState } from "../utils/planState";
import type { WorkspaceRightSurfaceLauncherProjection } from "../workspace/right-surface";

interface TaskCenterUtilityToolbarProps {
  projectRootPath?: string | null;
  onProjectChange?: (projectId: string | null) => void;
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
  onToggleActivityPanel?: () => void;
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

function resolveWorktreeName(path: string): string {
  const normalized = path.replace(/[\\/]+$/, "");
  return normalized.split(/[\\/]/).at(-1)?.trim() || "";
}

function summarizeGitPatch(patch: string): {
  additions: number;
  deletions: number;
} {
  let additions = 0;
  let deletions = 0;
  for (const line of patch.split("\n")) {
    if (line.startsWith("+++") || line.startsWith("---")) {
      continue;
    }
    if (line.startsWith("+")) {
      additions += 1;
    } else if (line.startsWith("-")) {
      deletions += 1;
    }
  }
  return { additions, deletions };
}

function useProjectGitStatus(rootPath?: string | null) {
  const normalizedRootPath = rootPath?.trim() || null;
  const [status, setStatus] = React.useState<ProjectGitStatus | null>(null);
  const [changeSummary, setChangeSummary] = React.useState<{
    additions: number;
    deletions: number;
  } | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const requestIdRef = React.useRef(0);

  const refresh = React.useCallback(async () => {
    if (!normalizedRootPath) {
      setStatus(null);
      setChangeSummary(null);
      setLoading(false);
      setError(null);
      return;
    }

    const requestId = ++requestIdRef.current;
    setLoading(true);
    setError(null);
    setChangeSummary(null);
    try {
      const nextStatus = await readProjectGitStatus(normalizedRootPath);
      if (requestId === requestIdRef.current) {
        setStatus(nextStatus);
      }
    } catch (readError) {
      if (requestId === requestIdRef.current) {
        setStatus(null);
        setError(extractErrorMessage(readError));
      }
    }
    try {
      const diff = await readProjectGitDiff(normalizedRootPath);
      if (requestId === requestIdRef.current) {
        setChangeSummary(summarizeGitPatch(diff.patch));
      }
    } catch {
      if (requestId === requestIdRef.current) {
        setChangeSummary(null);
      }
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }, [normalizedRootPath]);

  React.useEffect(() => {
    void refresh();
    return () => {
      requestIdRef.current += 1;
    };
  }, [refresh]);

  return { status, changeSummary, loading, error, refresh };
}

export function TaskCenterUtilityToolbar({
  projectRootPath,
  onProjectChange,
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
  onToggleActivityPanel,
  rightSurfaceLaunchers,
}: TaskCenterUtilityToolbarProps) {
  const { t } = useTranslation("agent");
  const normalizedProjectRootPath = projectRootPath?.trim() || null;
  const [environmentVisited, setEnvironmentVisited] = React.useState(false);
  const [environmentOpen, setEnvironmentOpen] = React.useState(false);
  const environmentLifecycle = useThreadEnvironmentLifecycleStatus({
    threadId: taskRail?.threadRead?.thread_id ?? taskRail?.sessionId,
    threadRead: taskRail?.threadRead,
  });
  const autoRevealedPlanKeyRef = React.useRef<string | null>(null);
  const { status, changeSummary, loading, error, refresh } =
    useProjectGitStatus(environmentVisited ? normalizedProjectRootPath : null);
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
  const activityLauncher = rightSurfaceLauncherByKind.get("activity");
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
  const shouldRenderActivityToggle =
    Boolean(onToggleActivityPanel) &&
    Boolean(activityLauncher) &&
    (!activityLauncher?.disabled ||
      Boolean(activityLauncher?.active) ||
      (activityLauncher?.pendingCount ?? 0) > 0);
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
    shouldRenderTraceToggle ||
    shouldRenderActivityToggle ||
    shouldRenderBrowserToggle ||
    shouldRenderFilesToggle;
  const effectiveCanvasOpen = workbenchLauncher?.active ?? isCanvasOpen;
  const workbenchPendingCount = workbenchLauncher?.pendingCount ?? 0;
  const effectiveShellPanelOpen = shellLauncher?.active ?? shellPanelOpen;
  const shellPendingCount = shellLauncher?.pendingCount ?? 0;
  const effectiveFilesPanelOpen = Boolean(filesLauncher?.active);
  const filesPendingCount = filesLauncher?.pendingCount ?? 0;
  const effectiveBrowserPanelOpen = Boolean(browserLauncher?.active);
  const browserPendingCount = browserLauncher?.pendingCount ?? 0;
  const effectiveTracePanelOpen = traceLauncher?.active ?? false;
  const tracePendingCount = traceLauncher?.pendingCount ?? 0;
  const effectiveActivityPanelOpen = Boolean(activityLauncher?.active);
  const activityPendingCount = activityLauncher?.pendingCount ?? 0;
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

  const handleOpenLocalLocation = React.useCallback(async () => {
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
      await openProjectPathWithTool(normalizedProjectRootPath, "finder");
    } catch (openError) {
      toast.error(
        agentText(
          "agentChat.navbar.appSwitcher.toast.openFailed",
          "打开项目失败：{{message}}",
          { message: extractErrorMessage(openError) },
        ),
      );
    }
  }, [normalizedProjectRootPath]);

  const handleOpenCodexWeb = React.useCallback(async () => {
    try {
      await openExternalUrlWithSystemBrowser("https://chatgpt.com/codex");
    } catch (openError) {
      toast.error(
        agentText(
          "agentChat.navbar.appSwitcher.toast.codexWebFailed",
          "打开 Codex web 失败：{{message}}",
          { message: extractErrorMessage(openError) },
        ),
      );
    }
  }, []);

  const handleCreateWorktree = React.useCallback(async () => {
    if (!normalizedProjectRootPath || !onProjectChange) {
      return;
    }
    try {
      const worktree = await createProjectGitWorktree(
        normalizedProjectRootPath,
        undefined,
        status?.currentBranch ?? undefined,
      );
      const project = await ensureProjectWorkspace({
        name: resolveWorktreeName(worktree.worktreePath) || "Worktree",
        rootPath: worktree.worktreePath,
        workspaceType: "general",
      });
      markProjectOpened(project.id);
      onProjectChange(project.id);
      toast.success(
        agentText(
          "agentChat.navbar.appSwitcher.worktreeCreated",
          "工作树已创建",
        ),
      );
    } catch (createError) {
      toast.error(
        agentText(
          "agentChat.navbar.appSwitcher.worktreeCreateFailed",
          "创建工作树失败：{{message}}",
          { message: extractErrorMessage(createError) },
        ),
      );
    }
  }, [normalizedProjectRootPath, onProjectChange, status?.currentBranch]);

  const branchLabel =
    status?.currentBranch?.trim() ||
    agentText("agentChat.navbar.environment.branchFallback", "无分支");
  const changeCount = status?.hasGitRepository
    ? status.uncommittedFileCount
    : 0;
  const handleCheckoutBranch = React.useCallback(
    async (branch: string) => {
      if (!normalizedProjectRootPath || !status?.hasGitRepository) {
        return;
      }
      try {
        await checkoutProjectGitBranch(normalizedProjectRootPath, branch);
        await refresh();
      } catch (checkoutError) {
        toast.error(
          agentText(
            "agentChat.navbar.environment.branchSwitchFailed",
            "切换分支失败：{{message}}",
            { message: extractErrorMessage(checkoutError) },
          ),
        );
      }
    },
    [normalizedProjectRootPath, refresh, status?.hasGitRepository],
  );
  const handleCreateBranch = React.useCallback(
    async (branch: string) => {
      if (!normalizedProjectRootPath || !status?.hasGitRepository) {
        return;
      }
      try {
        await createProjectGitBranch(normalizedProjectRootPath, branch);
        await refresh();
      } catch (createError) {
        toast.error(
          agentText(
            "agentChat.navbar.environment.branchCreateFailed",
            "创建分支失败：{{message}}",
            { message: extractErrorMessage(createError) },
          ),
        );
      }
    },
    [normalizedProjectRootPath, refresh, status?.hasGitRepository],
  );
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
  React.useEffect(() => {
    if (environmentOpen) {
      return;
    }
    const todoPlanKey = (taskRail?.todoItems ?? [])
      .filter((item) => item.content.trim().length > 0)
      .map((item) => `${item.content.trim()}\u0000${item.status ?? ""}`)
      .join("\u0001");
    const revisionId = taskRail?.threadItems?.length
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
          <TaskCenterLocationPanel
            canOpenLocal={Boolean(normalizedProjectRootPath)}
            onOpenLocal={() => void handleOpenLocalLocation()}
            onOpenCodexWeb={() => void handleOpenCodexWeb()}
            canCreateWorktree={Boolean(
              normalizedProjectRootPath && onProjectChange,
            )}
            onCreateWorktree={() => void handleCreateWorktree()}
            translate={taskRailTranslate}
          />
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
              className={cn(taskCenterIconOnlyButtonClassName, "relative")}
              aria-label={agentText(
                "agentChat.navbar.environment.open",
                "打开环境信息",
              )}
              title={agentText(
                "agentChat.navbar.environment.open",
                "打开环境信息",
              )}
              data-testid="task-center-environment-trigger"
              data-environment-lifecycle={
                environmentLifecycle.some(
                  (environment) => environment.status === "disconnected",
                )
                  ? "disconnected"
                  : environmentLifecycle.some(
                        (environment) => environment.status === "pending",
                      )
                    ? "pending"
                    : environmentLifecycle.length > 0
                      ? "connected"
                      : "local"
              }
            >
              <SlidersHorizontal className="h-4 w-4" />
              {environmentLifecycle.length > 0 ? (
                <span
                  aria-hidden="true"
                  className={cn(
                    "absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full ring-2 ring-[color:var(--lime-surface)]",
                    environmentLifecycle.some(
                      (environment) => environment.status === "disconnected",
                    )
                      ? "bg-rose-500"
                      : environmentLifecycle.some(
                            (environment) => environment.status === "pending",
                          )
                        ? "bg-amber-500"
                        : "bg-emerald-500",
                  )}
                />
              ) : null}
            </Button>
          </PopoverTrigger>
          <TaskCenterEnvironmentPanel
            normalizedProjectRootPath={normalizedProjectRootPath}
            status={status}
            environmentStatusLabel={environmentStatusLabel}
            lifecycleStatuses={environmentLifecycle}
            changeSummary={changeSummary}
            branchLabel={branchLabel}
            changeCount={changeCount}
            onCheckoutBranch={handleCheckoutBranch}
            onCreateBranch={handleCreateBranch}
            translate={taskRailTranslate}
          />
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

          {shouldRenderActivityToggle ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className={cn(
                taskCenterIconOnlyButtonClassName,
                "relative",
                effectiveActivityPanelOpen &&
                  "bg-[color:var(--lime-chrome-tab-active-surface)] text-[color:var(--lime-text)]",
              )}
              disabled={activityLauncher?.disabled}
              onClick={onToggleActivityPanel}
              aria-label={agentText(
                effectiveActivityPanelOpen
                  ? "agentChat.navbar.closeActivity"
                  : "agentChat.navbar.openActivity",
                effectiveActivityPanelOpen ? "关闭活动" : "打开活动",
              )}
              aria-expanded={effectiveActivityPanelOpen}
              title={agentText("agentChat.navbar.activity", "活动")}
              data-testid="task-center-activity-toggle"
            >
              <Activity className="h-4 w-4" />
              {activityPendingCount > 0 ? (
                <span className="absolute -right-1 -top-1 rounded-full border border-[color:var(--lime-surface-border-strong)] bg-[color:var(--lime-surface)] px-1 text-[9px] font-medium leading-4 text-[color:var(--lime-brand-strong)]">
                  {activityPendingCount > 99 ? "99+" : activityPendingCount}
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
