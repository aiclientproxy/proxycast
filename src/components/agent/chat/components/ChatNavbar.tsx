import React from "react";
import {
  Box,
  ChevronDown,
  FolderOpen,
  Home,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  Settings,
  Sparkles,
  X,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { ProjectSelector } from "@/components/projects/ProjectSelector";
import { cn } from "@/lib/utils";
import { ThreadProjectSelector } from "./ThreadProjectSelector";
import {
  buildFileNameTabTooltip,
  resolveFileNameTabLabel,
} from "../utils/tabFileDisplay";
import { InputbarProjectContextBar } from "./Inputbar/components/InputbarProjectContextBar";
import { buildInputbarCoreCopy } from "./Inputbar/components/inputbarCoreCopy";
import { Navbar } from "../styles";
import {
  TASK_CENTER_CHROME_ACTIVE_TAB,
  TASK_CENTER_CHROME_RAIL_SURFACE,
} from "../workspace/taskCenterChromeTokens";

interface ChatNavbarProps {
  isRunning: boolean;
  chrome?: "full" | "workspace-compact";
  collapseChrome?: boolean;
  contextVariant?: "default" | "task-center";
  entryContextLabel?: string;
  entryContextHint?: string;
  onToggleFullscreen: () => void;
  onBackToProjectManagement?: () => void;
  onBackToResources?: () => void;
  onToggleSettings?: () => void;
  onBackHome?: () => void;
  showCanvasToggle?: boolean;
  isCanvasOpen?: boolean;
  onToggleCanvas?: () => void;
  threadId?: string | null;
  projectId?: string | null;
  workspaceRootPath?: string | null;
  openedProjects?: ChatNavbarOpenedProject[];
  onProjectChange?: (projectId: string | null) => void;
  onCloseProject?: (projectId: string) => void;
  workspaceType?: string;
  deferWorkspaceListLoad?: boolean;
  showHarnessToggle?: boolean;
  harnessPanelVisible?: boolean;
  onToggleHarnessPanel?: () => void;
  harnessPendingCount?: number;
  harnessAttentionLevel?: "idle" | "active" | "warning";
  harnessToggleLabel?: string;
  showContextCompactionAction?: boolean;
  contextCompactionRunning?: boolean;
  onCompactContext?: () => void;
}

interface ChatNavbarOpenedProject {
  id: string;
  name: string;
  rootPath?: string | null;
}

const toolbarGroupClassName =
  "flex max-w-full flex-nowrap items-center overflow-hidden whitespace-nowrap rounded-[20px] border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface-subtle)] p-1.5 shadow-sm shadow-slate-950/5 backdrop-blur-sm";

const toolbarDividerClassName =
  "mx-1.5 h-6 w-px shrink-0 bg-[color:var(--lime-surface-border)]";

const toolbarEmbeddedButtonClassName =
  "h-9 shrink-0 whitespace-nowrap rounded-2xl border border-transparent px-3.5 text-xs shadow-none";

const toolbarGhostIconButtonClassName =
  "h-9 w-9 shrink-0 rounded-2xl text-[color:var(--lime-text-muted)] hover:bg-[color:var(--lime-surface-hover)] hover:text-[color:var(--lime-text)]";

const toolbarTextButtonClassName =
  "gap-1.5 text-[color:var(--lime-text)] hover:bg-[color:var(--lime-surface)] hover:text-[color:var(--lime-text-strong)]";

const taskCenterTopRailClassName =
  "relative flex h-[42px] w-full items-end overflow-visible bg-[color:var(--lime-chrome-rail)] px-4 pt-1";

const taskCenterWorkspaceTabClassName =
  "relative z-20 flex h-9 min-w-[148px] max-w-[224px] items-center rounded-t-[18px] rounded-b-none border border-b-0 border-[color:var(--lime-chrome-border)] bg-[color:var(--lime-chrome-tab-active-surface)] px-2 text-sm font-medium text-[color:var(--lime-chrome-text)] shadow-[inset_0_1px_0_rgba(255,255,255,0.72)] dark:border-white/10 dark:bg-slate-900 dark:text-slate-300";

const taskCenterWorkspaceInactiveTabShellClassName =
  "group relative z-10 ml-1 flex h-8 min-w-[108px] max-w-[184px] items-center rounded-t-[15px] border border-b-0 border-transparent bg-transparent text-[color:var(--lime-chrome-muted)] transition hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-chrome-text)]";

const taskCenterWorkspaceInactiveTabButtonClassName =
  "flex h-full min-w-0 flex-1 items-center gap-1.5 px-3 pb-0.5 text-left text-xs font-medium";

const taskCenterWorkspaceTabCloseButtonClassName =
  "mr-1 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[color:var(--lime-chrome-muted)] opacity-80 transition hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-chrome-text)] focus-visible:opacity-100 group-hover:opacity-100";

const taskCenterWorkspaceTabCurveClassName =
  "pointer-events-none absolute bottom-0 h-[18px] w-[18px] bg-transparent";

function normalizeProjectId(value?: string | null): string | null {
  const normalized = value?.trim();
  return normalized ? normalized : null;
}

function resolveProjectDisplayName(project: ChatNavbarOpenedProject): string {
  return resolveFileNameTabLabel(
    project.rootPath?.trim() || project.name.trim() || project.id,
    project.id,
  );
}

function resolveProjectTooltip(project: ChatNavbarOpenedProject): string {
  return buildFileNameTabTooltip({
    label: resolveProjectDisplayName(project),
    source: project.rootPath?.trim() || project.name.trim() || project.id,
    fallback: project.id,
  });
}

function buildOpenedProjectTabs(
  openedProjects: ChatNavbarOpenedProject[],
): ChatNavbarOpenedProject[] {
  const seen = new Set<string>();
  return openedProjects.filter((project) => {
    const projectId = normalizeProjectId(project.id);
    if (!projectId || seen.has(projectId)) {
      return false;
    }
    seen.add(projectId);
    return true;
  });
}

export const ChatNavbar: React.FC<ChatNavbarProps> = ({
  isRunning: _isRunning,
  chrome = "full",
  collapseChrome = false,
  contextVariant = "default",
  entryContextLabel,
  entryContextHint,
  onToggleFullscreen: _onToggleFullscreen,
  onBackToProjectManagement,
  onBackToResources,
  onToggleSettings,
  onBackHome,
  showCanvasToggle = false,
  isCanvasOpen = false,
  onToggleCanvas,
  threadId = null,
  projectId = null,
  workspaceRootPath = null,
  openedProjects = [],
  onProjectChange,
  onCloseProject,
  workspaceType,
  deferWorkspaceListLoad,
  showHarnessToggle = false,
  harnessPanelVisible = false,
  onToggleHarnessPanel,
  harnessPendingCount = 0,
  harnessAttentionLevel = "idle",
  harnessToggleLabel = "Harness",
  showContextCompactionAction = false,
  contextCompactionRunning = false,
  onCompactContext,
}) => {
  const { t } = useTranslation("agent");
  const navText = (
    key: string,
    defaultValue: string,
    options?: Record<string, unknown>,
  ) =>
    String(
      t(
        key as never,
        {
          defaultValue,
          ...options,
        } as never,
      ),
    );
  const inputbarCoreCopy = React.useMemo(
    () =>
      buildInputbarCoreCopy((key, values) =>
        String(t(key as never, (values ?? {}) as never)),
      ),
    [t],
  );
  const [workspaceSelectorOpen, setWorkspaceSelectorOpen] =
    React.useState(false);
  const isTaskCenterChrome = contextVariant === "task-center";
  const isWorkspaceCompact = chrome === "workspace-compact";
  const effectiveCollapseChrome = collapseChrome && !isTaskCenterChrome;
  const groupClassName = cn(
    toolbarGroupClassName,
    (isWorkspaceCompact || effectiveCollapseChrome) && "rounded-[18px] p-1",
    effectiveCollapseChrome &&
      "border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] shadow-sm shadow-slate-950/5 backdrop-blur-0",
  );
  const dividerClassName = cn(
    toolbarDividerClassName,
    (isWorkspaceCompact || effectiveCollapseChrome) && "mx-1 h-5",
  );
  const embeddedButtonClassName = cn(
    toolbarEmbeddedButtonClassName,
    (isWorkspaceCompact || effectiveCollapseChrome) &&
      "h-8 rounded-[18px] px-3",
  );
  const ghostIconButtonClassName = cn(
    toolbarGhostIconButtonClassName,
    (isWorkspaceCompact || effectiveCollapseChrome) && "h-8 w-8 rounded-[18px]",
  );
  const showStatusTools = showHarnessToggle || showContextCompactionAction;
  const showNavigationTools =
    !effectiveCollapseChrome &&
    !isWorkspaceCompact &&
    (Boolean(onBackHome) ||
      Boolean(onBackToResources) ||
      Boolean(onBackToProjectManagement));
  const showWorkspaceTools = !effectiveCollapseChrome && showCanvasToggle;
  const showProjectSelector = !isWorkspaceCompact && !isTaskCenterChrome;
  const showCompactSettingsButton =
    isWorkspaceCompact && !isTaskCenterChrome && Boolean(onToggleSettings);
  const compactProjectSelectorClassName =
    isWorkspaceCompact || effectiveCollapseChrome
      ? "min-w-[184px] max-w-[248px]"
      : "min-w-[196px] max-w-[280px]";
  const showEntryContext = Boolean(entryContextLabel);
  const shouldDeferWorkspaceListLoad =
    deferWorkspaceListLoad ?? isTaskCenterChrome;
  const normalizedProjectId = normalizeProjectId(projectId);
  const openedProjectTabs = React.useMemo(
    () => buildOpenedProjectTabs(openedProjects),
    [openedProjects],
  );
  const openedProjectPickerItems = React.useMemo(
    () =>
      openedProjectTabs.map((project) => ({
        ...project,
        name: resolveProjectDisplayName(project),
      })),
    [openedProjectTabs],
  );
  const orderedOpenedProjectTabs = React.useMemo(() => {
    if (!normalizedProjectId) {
      return [{ id: "", name: "" }, ...openedProjectTabs];
    }
    const hasCurrentProject = openedProjectTabs.some(
      (project) => normalizeProjectId(project.id) === normalizedProjectId,
    );
    return hasCurrentProject ? openedProjectTabs : [{ id: "", name: "" }];
  }, [normalizedProjectId, openedProjectTabs]);
  const canCloseProjectTabs = Boolean(
    onCloseProject && orderedOpenedProjectTabs.length > 1,
  );
  const activeWorkspace = orderedOpenedProjectTabs.find(
    (project) => normalizeProjectId(project.id) === normalizedProjectId,
  );
  const currentWorkspaceName = activeWorkspace?.name || null;

  if (isTaskCenterChrome) {
    return (
      <Navbar
        $compact
        $collapsed={false}
        $taskCenter
        data-testid="task-center-workspace-bar"
        style={{
          padding: 0,
          gap: 0,
          alignItems: "stretch",
          overflow: "visible",
          zIndex: 8,
        }}
      >
        <div
          className={taskCenterTopRailClassName}
          style={{ background: TASK_CENTER_CHROME_RAIL_SURFACE }}
        >
          <div className="flex min-w-0 items-center">
            {orderedOpenedProjectTabs.map((project, index) => {
              const isActiveProject = normalizedProjectId
                ? normalizeProjectId(project.id) === normalizedProjectId
                : project.id === "";
              const projectName = resolveProjectDisplayName(project);
              const projectTooltip = resolveProjectTooltip(project);
              const closeProjectLabel = navText(
                "agentChat.navbar.closeWorkspaceTab",
                "关闭{{label}}",
                { label: projectName },
              );
              if (isActiveProject) {
                const canCloseActiveProject = Boolean(
                  project.id && canCloseProjectTabs,
                );
                return (
                  <div
                    key={project.id || "__workspace-selector__"}
                    className={cn(
                      taskCenterWorkspaceTabClassName,
                      index > 0 && "ml-1",
                    )}
                    data-testid="task-center-workspace-shell"
                    data-project-id={project.id}
                  >
                    <span
                      aria-hidden="true"
                      className={cn(
                        taskCenterWorkspaceTabCurveClassName,
                        "-left-4",
                      )}
                      style={{
                        borderBottomRightRadius: 18,
                        boxShadow: `5px 5px 0 5px ${TASK_CENTER_CHROME_ACTIVE_TAB}`,
                      }}
                    />
                    <span
                      aria-hidden="true"
                      className={cn(
                        taskCenterWorkspaceTabCurveClassName,
                        "-right-4",
                      )}
                      style={{
                        borderBottomLeftRadius: 18,
                        boxShadow: `-5px 5px 0 5px ${TASK_CENTER_CHROME_ACTIVE_TAB}`,
                      }}
                    />
                    <InputbarProjectContextBar
                      projectId={project.id || projectId}
                      openedProjects={openedProjectPickerItems}
                      onProjectChange={onProjectChange}
                      projectOpen={workspaceSelectorOpen}
                      onProjectOpenChange={setWorkspaceSelectorOpen}
                      showModeControls={false}
                      copy={inputbarCoreCopy.projectContext}
                      className="min-w-0 flex-1 flex-nowrap"
                      projectTriggerClassName={cn(
                        "h-8 w-full max-w-none justify-start rounded-t-[16px] rounded-b-none bg-transparent px-2.5 pb-0.5 text-sm text-[color:var(--lime-chrome-text)] hover:bg-transparent hover:text-[color:var(--lime-chrome-text)] hover:shadow-none",
                        canCloseActiveProject
                          ? "max-w-[188px]"
                          : "max-w-[224px]",
                      )}
                      projectTriggerContentClassName="text-left"
                      projectMenuAlign="start"
                      projectMenuSide="bottom"
                      projectMenuSideOffset={8}
                    />
                    {canCloseActiveProject ? (
                      <button
                        type="button"
                        className={taskCenterWorkspaceTabCloseButtonClassName}
                        aria-label={closeProjectLabel}
                        title={closeProjectLabel}
                        data-testid={`task-center-opened-project-close-${project.id}`}
                        onClick={() => onCloseProject?.(project.id)}
                      >
                        <X className="h-3 w-3" />
                      </button>
                    ) : null}
                  </div>
                );
              }
              return (
                <div
                  key={project.id}
                  className={taskCenterWorkspaceInactiveTabShellClassName}
                  data-project-id={project.id}
                  title={projectTooltip}
                >
                  <button
                    type="button"
                    className={taskCenterWorkspaceInactiveTabButtonClassName}
                    title={projectTooltip}
                    data-testid="task-center-opened-project-tab"
                    data-project-id={project.id}
                    onClick={() => onProjectChange?.(project.id)}
                  >
                    <FolderOpen className="h-3.5 w-3.5 shrink-0" />
                    <span className="min-w-0 truncate">{projectName}</span>
                  </button>
                  {onCloseProject ? (
                    <button
                      type="button"
                      className={taskCenterWorkspaceTabCloseButtonClassName}
                      aria-label={closeProjectLabel}
                      title={closeProjectLabel}
                      data-testid={`task-center-opened-project-close-${project.id}`}
                      onClick={() => onCloseProject(project.id)}
                    >
                      <X className="h-3 w-3" />
                    </button>
                  ) : null}
                </div>
              );
            })}
            <div className="relative ml-2 flex h-9 items-center pb-1">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-8 w-8 rounded-[14px] bg-transparent text-[color:var(--lime-chrome-muted)] shadow-none hover:bg-[color:var(--lime-chrome-tab-hover)] hover:text-[color:var(--lime-chrome-text)] dark:text-slate-300 dark:hover:text-white"
                onClick={() => {
                  setWorkspaceSelectorOpen((current) => !current);
                }}
                aria-label={
                  workspaceSelectorOpen
                    ? navText(
                        "agentChat.navbar.collapseWorkspaceMenu",
                        "收起工作区菜单",
                      )
                    : navText(
                        "agentChat.navbar.expandWorkspaceMenu",
                        "展开工作区菜单",
                      )
                }
                aria-expanded={workspaceSelectorOpen}
                title={
                  workspaceSelectorOpen
                    ? navText(
                        "agentChat.navbar.collapseWorkspaceMenu",
                        "收起工作区菜单",
                      )
                    : navText(
                        "agentChat.navbar.expandWorkspaceMenu",
                        "展开工作区菜单",
                      )
                }
                data-testid="task-center-workspace-menu-trigger"
              >
                <Plus size={17} strokeWidth={1.7} />
              </Button>
            </div>
          </div>
          <div className="ml-auto h-9 shrink-0" aria-hidden="true" />
        </div>
      </Navbar>
    );
  }

  return (
    <Navbar $compact={isWorkspaceCompact} $collapsed={effectiveCollapseChrome}>
      <div
        className="flex min-w-0 items-center gap-2 overflow-hidden"
        data-testid="chat-navbar-leading-tools"
      >
        {showNavigationTools ? (
          <div className={groupClassName}>
            {onBackHome && (
              <Button
                variant="ghost"
                size="icon"
                className={ghostIconButtonClassName}
                onClick={onBackHome}
                title={navText("agentChat.navbar.backHome", "返回新建任务")}
                aria-label={navText(
                  "agentChat.navbar.backHome",
                  "返回新建任务",
                )}
              >
                <Home size={18} />
              </Button>
            )}
            {onBackHome && (onBackToResources || onBackToProjectManagement) ? (
              <div className={dividerClassName} aria-hidden="true" />
            ) : null}
            {onBackToResources && (
              <Button
                variant="ghost"
                size="sm"
                className={cn(
                  embeddedButtonClassName,
                  toolbarTextButtonClassName,
                )}
                onClick={onBackToResources}
              >
                <FolderOpen size={16} className="mr-0.5" />
                {navText("agentChat.navbar.backResources", "返回资源")}
              </Button>
            )}
            {onBackToResources && onBackToProjectManagement ? (
              <div className={dividerClassName} aria-hidden="true" />
            ) : null}
            {onBackToProjectManagement && (
              <Button
                variant="ghost"
                size="sm"
                className={cn(
                  embeddedButtonClassName,
                  toolbarTextButtonClassName,
                )}
                onClick={onBackToProjectManagement}
              >
                {navText("agentChat.navbar.projectManagement", "项目管理")}
              </Button>
            )}
          </div>
        ) : null}

        {showWorkspaceTools ? (
          <div className={groupClassName}>
            {showCanvasToggle ? (
              <Button
                variant="ghost"
                size="icon"
                className={ghostIconButtonClassName}
                onClick={onToggleCanvas}
                aria-label={
                  isCanvasOpen
                    ? navText("agentChat.navbar.collapseCanvas", "折叠画布")
                    : navText("agentChat.navbar.expandCanvas", "展开画布")
                }
                title={
                  isCanvasOpen
                    ? navText("agentChat.navbar.collapseCanvas", "折叠画布")
                    : navText("agentChat.navbar.expandCanvas", "展开画布")
                }
              >
                {isCanvasOpen ? (
                  <PanelRightClose size={18} />
                ) : (
                  <PanelRightOpen size={18} />
                )}
              </Button>
            ) : null}
          </div>
        ) : null}

        {showEntryContext ? (
          <div
            className={cn(
              "ml-1 min-w-0",
              isWorkspaceCompact ? "max-w-[180px]" : "max-w-[320px]",
            )}
          >
            <div className="flex min-w-0 flex-col gap-1">
              <span className="inline-flex w-fit items-center rounded-full border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface-muted)] px-3 py-1 text-[11px] font-medium text-[color:var(--lime-text-muted)]">
                {entryContextLabel}
              </span>
              {!isWorkspaceCompact && entryContextHint ? (
                <p className="truncate text-xs text-slate-500">
                  {entryContextHint}
                </p>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>

      <div className="flex-1" />

      <div
        className="flex min-w-0 shrink-0 flex-nowrap items-center gap-2 overflow-hidden whitespace-nowrap"
        data-testid="chat-navbar-trailing-tools"
      >
        {threadId ? (
          <ThreadProjectSelector
            threadId={threadId}
            workspaceName={currentWorkspaceName}
            workspaceRootPath={workspaceRootPath}
          />
        ) : null}
        {showProjectSelector ? (
          <div className={groupClassName}>
            <ProjectSelector
              value={projectId}
              onChange={(nextProjectId) => onProjectChange?.(nextProjectId)}
              onOpenProjectContents={
                onBackToResources ? () => onBackToResources() : undefined
              }
              workspaceType={workspaceType}
              placeholder={navText(
                "agentChat.navbar.projectPlaceholder",
                "选择项目",
              )}
              dropdownSide="bottom"
              dropdownAlign="end"
              enableManagement={workspaceType === "general"}
              density="compact"
              chrome="embedded"
              deferProjectListLoad={shouldDeferWorkspaceListLoad}
              skipDefaultWorkspaceReadyCheck={shouldDeferWorkspaceListLoad}
              className={compactProjectSelectorClassName}
            />
            {onToggleSettings ? (
              <>
                <div className={dividerClassName} aria-hidden="true" />
                <Button
                  variant="ghost"
                  size="icon"
                  className={ghostIconButtonClassName}
                  onClick={onToggleSettings}
                  aria-label={navText(
                    "agentChat.navbar.openSettings",
                    "打开设置",
                  )}
                  title={navText("agentChat.navbar.openSettings", "打开设置")}
                >
                  <Settings size={18} />
                </Button>
              </>
            ) : null}
          </div>
        ) : null}

        {showCompactSettingsButton ? (
          <div className={groupClassName}>
            <Button
              variant="ghost"
              size="icon"
              className={ghostIconButtonClassName}
              onClick={onToggleSettings}
              aria-label={navText("agentChat.navbar.openSettings", "打开设置")}
              title={navText("agentChat.navbar.openSettings", "打开设置")}
            >
              <Settings size={18} />
            </Button>
          </div>
        ) : null}

        {showStatusTools ? (
          <div className={groupClassName}>
            {showContextCompactionAction ? (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className={cn(
                  embeddedButtonClassName,
                  toolbarTextButtonClassName,
                )}
                onClick={onCompactContext}
                disabled={contextCompactionRunning}
                aria-label={
                  contextCompactionRunning
                    ? navText(
                        "agentChat.navbar.compactContextRunning",
                        "正在压缩上下文",
                      )
                    : navText("agentChat.navbar.compactContext", "压缩上下文")
                }
                title={
                  contextCompactionRunning
                    ? navText(
                        "agentChat.navbar.compactContextRunning",
                        "正在压缩上下文",
                      )
                    : navText("agentChat.navbar.compactContext", "压缩上下文")
                }
              >
                <Box size={14} />
                <span>
                  {contextCompactionRunning
                    ? navText(
                        "agentChat.navbar.compactContextShortRunning",
                        "压缩中...",
                      )
                    : navText("agentChat.navbar.compactContext", "压缩上下文")}
                </span>
              </Button>
            ) : null}

            {showContextCompactionAction && showHarnessToggle ? (
              <div className={dividerClassName} aria-hidden="true" />
            ) : null}

            {showHarnessToggle ? (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className={cn(
                  embeddedButtonClassName,
                  toolbarTextButtonClassName,
                  harnessPanelVisible &&
                    "bg-[color:var(--lime-surface-hover)] text-[color:var(--lime-text)]",
                  harnessAttentionLevel === "warning" &&
                    !harnessPanelVisible &&
                    "border-[color:var(--lime-warning-border)] bg-[color:var(--lime-warning-soft)] text-[color:var(--lime-warning)] hover:bg-[color:var(--lime-warning-soft)] hover:text-[color:var(--lime-warning)]",
                )}
                onClick={onToggleHarnessPanel}
                aria-label={
                  harnessPanelVisible
                    ? navText(
                        "agentChat.navbar.closeHarness",
                        "关闭{{label}}",
                        { label: harnessToggleLabel },
                      )
                    : navText("agentChat.navbar.openHarness", "打开{{label}}", {
                        label: harnessToggleLabel,
                      })
                }
                aria-expanded={harnessPanelVisible}
                title={harnessToggleLabel}
              >
                <Sparkles size={14} />
                <span>{harnessToggleLabel}</span>
                {harnessPendingCount > 0 ? (
                  <span className="rounded-full border border-[color:var(--lime-surface-border-strong)] bg-[color:var(--lime-surface)] px-1.5 py-0.5 text-[10px] font-medium leading-none text-[color:var(--lime-brand-strong)] shadow-sm shadow-slate-950/10">
                    {harnessPendingCount > 99 ? "99+" : harnessPendingCount}
                  </span>
                ) : null}
                <ChevronDown
                  className={cn(
                    "h-3.5 w-3.5 transition-transform",
                    harnessPanelVisible && "rotate-180",
                  )}
                />
              </Button>
            ) : null}
          </div>
        ) : null}
      </div>
    </Navbar>
  );
};
