import React, { useEffect, useMemo, useState } from "react";
import {
  Check,
  ChevronDown,
  ChevronRight,
  FolderOpen,
  FolderPlus,
  FolderX,
  GitBranch,
  Monitor,
  Plus,
  Search,
} from "lucide-react";
import { toast } from "sonner";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { requestChatHostOpenPath } from "@/components/agent/chat/host/chatHostCapabilities";
import { cn } from "@/lib/utils";
import {
  ensureProjectWorkspace,
  extractErrorMessage,
  getProject,
  getWorkspaceProjectsRoot,
  resolveProjectRootPath,
} from "@/lib/api/project";
import {
  checkoutProjectGitBranch,
  createProjectGitBranch,
  createProjectGitWorktree,
  readProjectGitStatus,
  type ProjectGitStatus,
} from "@/lib/api/projectGit";
import { markProjectOpened } from "../../../hooks/agentProjectStorage";

export interface InputbarOpenedProject {
  id: string;
  name: string;
  rootPath?: string | null;
}

export interface InputbarProjectContextCopy {
  projectAria: string;
  searchPlaceholder: string;
  openedLabel: string;
  currentBadge: string;
  noProjectLabel: string;
  noProjectDescription: string;
  noOpenedProjects: string;
  addNewProject: string;
  createBlankProject: string;
  useExistingFolder: string;
  selectFolderDialogTitle: string;
  newProjectNameFallback: string;
  projectCreated: string;
  projectOpened: string;
  projectCreateFailed: string;
  projectOpenFailed: string;
  noProjectAction: string;
  modeLabel: string;
  localMode: string;
  modeMenuTitle: string;
  localProcessing: string;
  newWorktree: string;
  worktreeCreated: string;
  worktreeCreateFailed: string;
  branchLabel: string;
  branchFallback: string;
  branchSearchPlaceholder: string;
  branchCreateAction: string;
  branchCreateNamedAction: (branch: string) => string;
  branchSwitched: string;
  branchSwitchFailed: string;
  branchCreated: string;
  branchCreateFailed: string;
  uncommittedFiles: (count: number) => string;
}

interface InputbarProjectContextBarProps {
  projectId?: string | null;
  openedProjects?: InputbarOpenedProject[];
  onProjectChange?: (projectId: string | null) => void;
  modeLabel?: string;
  branchLabel?: string;
  projectOpen?: boolean;
  onProjectOpenChange?: (open: boolean) => void;
  showModeControls?: boolean;
  className?: string;
  projectTriggerClassName?: string;
  projectTriggerContentClassName?: string;
  projectMenuAlign?: "start" | "center" | "end";
  projectMenuSide?: "top" | "right" | "bottom" | "left";
  projectMenuSideOffset?: number;
  copy: InputbarProjectContextCopy;
}

type ProjectActionState = "blank" | "folder" | null;

type GitActionState = "checkout" | "createBranch" | "worktree" | null;

function normalizeProjectId(projectId?: string | null): string {
  return projectId?.trim() ?? "";
}

function isUuidLike(value?: string | null): boolean {
  const normalized = value?.trim();
  if (!normalized) {
    return false;
  }
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
    normalized,
  );
}

function isDisplayableProject(project?: InputbarOpenedProject | null) {
  if (!project) {
    return false;
  }
  const projectId = normalizeProjectId(project.id);
  const projectName = project.name.trim();
  const projectRootPath = project.rootPath?.trim();
  if (!projectId || !projectName) {
    return false;
  }
  if (projectRootPath) {
    return true;
  }
  return !(isUuidLike(projectId) && projectName === projectId);
}

function resolveDirectoryName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop()?.trim() || path.trim();
}

function formatProjectPath(path?: string | null): string | null {
  const normalized = path?.trim();
  if (!normalized) {
    return null;
  }

  const parts = normalized.split(/[\\/]+/).filter(Boolean);
  if (parts.length <= 2) {
    return normalized;
  }

  return `.../${parts.slice(-2).join("/")}`;
}

export function InputbarProjectContextBar({
  projectId,
  openedProjects,
  onProjectChange,
  modeLabel,
  branchLabel,
  projectOpen,
  onProjectOpenChange,
  showModeControls = true,
  className,
  projectTriggerClassName,
  projectTriggerContentClassName,
  projectMenuAlign = "start",
  projectMenuSide = "bottom",
  projectMenuSideOffset = 8,
  copy,
}: InputbarProjectContextBarProps) {
  const normalizedProjectId = normalizeProjectId(projectId);
  const [resolvedProject, setResolvedProject] =
    useState<InputbarOpenedProject | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [uncontrolledProjectOpen, setUncontrolledProjectOpen] = useState(false);
  const [submenuOpen, setSubmenuOpen] = useState(false);
  const [modePopoverOpen, setModePopoverOpen] = useState(false);
  const [branchPopoverOpen, setBranchPopoverOpen] = useState(false);
  const [branchSearchQuery, setBranchSearchQuery] = useState("");
  const [actionState, setActionState] = useState<ProjectActionState>(null);
  const [gitActionState, setGitActionState] = useState<GitActionState>(null);
  const [gitRepositoryState, setGitRepositoryState] =
    useState<ProjectGitStatus | null>(null);
  const projectPopoverOpen = projectOpen ?? uncontrolledProjectOpen;
  const handleProjectPopoverOpenChange = React.useCallback(
    (nextOpen: boolean) => {
      if (projectOpen === undefined) {
        setUncontrolledProjectOpen(nextOpen);
      }
      onProjectOpenChange?.(nextOpen);
    },
    [onProjectOpenChange, projectOpen],
  );

  const openedProjectsContainCurrent = Boolean(
    normalizedProjectId &&
    openedProjects?.some(
      (project) => normalizeProjectId(project.id) === normalizedProjectId,
    ),
  );

  useEffect(() => {
    if (!normalizedProjectId || openedProjectsContainCurrent) {
      setResolvedProject(null);
      return;
    }

    let cancelled = false;
    void getProject(normalizedProjectId)
      .then((project) => {
        if (cancelled) {
          return;
        }
        const projectSummary = project
          ? {
              id: project.id,
              name: project.name,
              rootPath: project.rootPath,
            }
          : null;
        setResolvedProject(
          isDisplayableProject(projectSummary) ? projectSummary : null,
        );
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        setResolvedProject(null);
      });

    return () => {
      cancelled = true;
    };
  }, [normalizedProjectId, openedProjectsContainCurrent]);

  const visibleOpenedProjects = useMemo(() => {
    const source: InputbarOpenedProject[] = [
      ...(openedProjects ?? []),
      ...(resolvedProject ? [resolvedProject] : []),
    ];
    const seen = new Set<string>();
    return source.filter((project) => {
      const id = normalizeProjectId(project.id);
      if (!id || seen.has(id) || !isDisplayableProject(project)) {
        return false;
      }
      seen.add(id);
      return true;
    });
  }, [openedProjects, resolvedProject]);
  const activeProject =
    visibleOpenedProjects.find(
      (project) => normalizeProjectId(project.id) === normalizedProjectId,
    ) ?? null;

  const activeProjectRootPath = activeProject?.rootPath?.trim() || "";

  useEffect(() => {
    if (!showModeControls || !activeProjectRootPath) {
      setGitRepositoryState(null);
      return;
    }

    let cancelled = false;
    setGitRepositoryState(null);
    void readProjectGitStatus(activeProjectRootPath)
      .then((state) => {
        if (!cancelled) {
          setGitRepositoryState(state);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setGitRepositoryState(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeProjectRootPath, showModeControls]);

  const normalizedSearch = searchQuery.trim().toLowerCase();
  const filteredOpenedProjects = normalizedSearch
    ? visibleOpenedProjects.filter(
        (project) =>
          project.name.toLowerCase().includes(normalizedSearch) ||
          project.id.toLowerCase().includes(normalizedSearch),
      )
    : visibleOpenedProjects;
  const resolvedModeLabel = modeLabel?.trim() || copy.localMode;
  const hasGitRepository = Boolean(gitRepositoryState?.hasGitRepository);
  const resolvedBranchLabel = hasGitRepository
    ? branchLabel?.trim() ||
      gitRepositoryState?.currentBranch ||
      copy.branchFallback
    : "";
  const branchOptions = useMemo(() => {
    if (!hasGitRepository) {
      return [];
    }
    const branches = [...(gitRepositoryState?.branches ?? [])];
    const currentBranch = gitRepositoryState?.currentBranch?.trim();
    if (currentBranch && !branches.includes(currentBranch)) {
      branches.unshift(currentBranch);
    }
    return branches;
  }, [gitRepositoryState, hasGitRepository]);
  const filteredBranchOptions = branchSearchQuery.trim()
    ? branchOptions.filter((branch) =>
        branch.toLowerCase().includes(branchSearchQuery.trim().toLowerCase()),
      )
    : branchOptions;
  const normalizedBranchCreateName = branchSearchQuery.trim();
  const canCreateBranch =
    hasGitRepository &&
    normalizedBranchCreateName.length > 0 &&
    !branchOptions.includes(normalizedBranchCreateName);

  const openEnsuredProject = (project: InputbarOpenedProject) => {
    markProjectOpened(project.id);
    onProjectChange?.(project.id);
    handleProjectPopoverOpenChange(false);
  };

  const handleCreateBlankProject = async () => {
    if (actionState) {
      return;
    }
    setActionState("blank");
    try {
      const projectsRoot = await getWorkspaceProjectsRoot();
      const name = copy.newProjectNameFallback.trim() || "Untitled Project";
      const rootPath = await resolveProjectRootPath(name, projectsRoot);
      const project = await ensureProjectWorkspace({
        name,
        rootPath,
        workspaceType: "general",
      });
      openEnsuredProject(project);
      toast.success(copy.projectCreated);
    } catch (error) {
      toast.error(`${copy.projectCreateFailed}: ${extractErrorMessage(error)}`);
    } finally {
      setActionState(null);
    }
  };

  const handleUseExistingFolder = async () => {
    if (actionState) {
      return;
    }
    setActionState("folder");
    try {
      const selectedPath = await requestChatHostOpenPath({
        directory: true,
        multiple: false,
        title: copy.selectFolderDialogTitle,
      });
      if (typeof selectedPath !== "string" || !selectedPath.trim()) {
        return;
      }

      const rootPath = selectedPath.trim();
      const project = await ensureProjectWorkspace({
        name:
          resolveDirectoryName(rootPath) ||
          copy.newProjectNameFallback.trim() ||
          "Untitled Project",
        rootPath,
        workspaceType: "general",
      });
      openEnsuredProject(project);
      toast.success(copy.projectOpened);
    } catch (error) {
      toast.error(`${copy.projectOpenFailed}: ${extractErrorMessage(error)}`);
    } finally {
      setActionState(null);
    }
  };

  const handleCreateWorktree = async () => {
    if (!activeProjectRootPath || gitActionState) {
      return;
    }
    setGitActionState("worktree");
    try {
      const worktree = await createProjectGitWorktree(
        activeProjectRootPath,
        undefined,
        gitRepositoryState?.currentBranch ?? undefined,
      );
      const project = await ensureProjectWorkspace({
        name:
          resolveDirectoryName(worktree.worktreePath) ||
          copy.newProjectNameFallback.trim() ||
          "Untitled Project",
        rootPath: worktree.worktreePath,
        workspaceType: "general",
      });
      setGitRepositoryState(worktree.status);
      openEnsuredProject(project);
      setModePopoverOpen(false);
      toast.success(copy.worktreeCreated);
    } catch (error) {
      toast.error(
        `${copy.worktreeCreateFailed}: ${extractErrorMessage(error)}`,
      );
    } finally {
      setGitActionState(null);
    }
  };

  const handleCheckoutBranch = async (branch: string) => {
    if (!activeProjectRootPath || gitActionState) {
      return;
    }
    if (branch === gitRepositoryState?.currentBranch) {
      setBranchPopoverOpen(false);
      return;
    }
    setGitActionState("checkout");
    try {
      const status = await checkoutProjectGitBranch(
        activeProjectRootPath,
        branch,
      );
      setGitRepositoryState(status);
      setBranchPopoverOpen(false);
      toast.success(copy.branchSwitched);
    } catch (error) {
      toast.error(`${copy.branchSwitchFailed}: ${extractErrorMessage(error)}`);
    } finally {
      setGitActionState(null);
    }
  };

  const handleCreateBranch = async () => {
    if (!activeProjectRootPath || !canCreateBranch || gitActionState) {
      return;
    }
    setGitActionState("createBranch");
    try {
      const status = await createProjectGitBranch(
        activeProjectRootPath,
        normalizedBranchCreateName,
      );
      setGitRepositoryState(status);
      setBranchSearchQuery("");
      setBranchPopoverOpen(false);
      toast.success(copy.branchCreated);
    } catch (error) {
      toast.error(`${copy.branchCreateFailed}: ${extractErrorMessage(error)}`);
    } finally {
      setGitActionState(null);
    }
  };

  return (
    <div
      className={cn(
        "inline-flex max-w-full min-w-0 flex-wrap items-center gap-1.5 text-slate-600",
        className,
      )}
      data-testid="inputbar-project-context-bar"
    >
      <Popover
        open={projectPopoverOpen}
        onOpenChange={handleProjectPopoverOpenChange}
      >
        <PopoverTrigger asChild>
          <button
            type="button"
            className={cn(
              "inline-flex h-7 min-w-0 max-w-[180px] items-center gap-1.5 rounded-lg px-2 text-xs font-medium transition",
              "text-slate-700 hover:bg-white hover:text-slate-950 hover:shadow-sm hover:shadow-slate-950/5",
              projectTriggerClassName,
            )}
            aria-label={copy.projectAria}
            title={activeProject?.name || copy.noProjectLabel}
            data-testid="inputbar-project-context-project-trigger"
          >
            <FolderOpen className="h-3.5 w-3.5 shrink-0 text-emerald-700" />
            <span className={cn("truncate", projectTriggerContentClassName)}>
              {activeProject?.name || copy.noProjectLabel}
            </span>
            <ChevronDown className="h-3 w-3 shrink-0 text-slate-400" />
          </button>
        </PopoverTrigger>
        <PopoverContent
          align={projectMenuAlign}
          side={projectMenuSide}
          sideOffset={projectMenuSideOffset}
          className="z-[70] w-[292px] overflow-visible rounded-xl border border-slate-200 bg-white p-1.5 text-slate-900 shadow-[0_24px_60px_-30px_rgba(15,23,42,0.52)]"
          data-testid="inputbar-project-context-menu"
          onCloseAutoFocus={(event) => event.preventDefault()}
        >
          <div className="flex h-8 items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-2">
            <Search className="h-3.5 w-3.5 text-slate-400" />
            <input
              className="min-w-0 flex-1 bg-transparent text-[13px] font-medium text-slate-900 outline-none placeholder:text-slate-400"
              value={searchQuery}
              placeholder={copy.searchPlaceholder}
              onChange={(event) => setSearchQuery(event.target.value)}
            />
          </div>

          <div className="px-2 pb-1 pt-2 text-[11px] font-semibold text-slate-500">
            {copy.openedLabel}
          </div>

          <div className="max-h-[210px] overflow-auto">
            {filteredOpenedProjects.length > 0 ? (
              filteredOpenedProjects.map((project) => {
                const isActive =
                  normalizeProjectId(project.id) === normalizedProjectId;
                const pathPreview = formatProjectPath(project.rootPath);
                return (
                  <button
                    key={project.id}
                    type="button"
                    className={cn(
                      "flex h-9 w-full min-w-0 items-center gap-2 rounded-lg px-2 text-left transition",
                      isActive
                        ? "bg-slate-100 text-slate-950"
                        : "text-slate-700 hover:bg-slate-50 hover:text-slate-950",
                    )}
                    data-testid="inputbar-project-context-opened-project"
                    onClick={() => openEnsuredProject(project)}
                  >
                    <FolderOpen
                      className={cn(
                        "h-4 w-4 shrink-0",
                        isActive ? "text-emerald-700" : "text-slate-500",
                      )}
                    />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[13px] font-medium">
                        {project.name}
                      </span>
                      {pathPreview ? (
                        <span className="block truncate text-[10.5px] font-medium text-slate-400">
                          {pathPreview}
                        </span>
                      ) : null}
                    </span>
                    {isActive ? (
                      <Check
                        aria-label={copy.currentBadge}
                        className="h-3.5 w-3.5 shrink-0 text-emerald-700"
                      />
                    ) : null}
                  </button>
                );
              })
            ) : (
              <div className="rounded-lg px-2 py-3 text-[13px] text-slate-500">
                {visibleOpenedProjects.length > 0
                  ? copy.noOpenedProjects
                  : copy.noProjectDescription}
              </div>
            )}
          </div>

          <div className="my-1 h-px bg-slate-100" />

          <div
            className="relative"
            onMouseEnter={() => setSubmenuOpen(true)}
            onMouseLeave={() => setSubmenuOpen(false)}
          >
            <button
              type="button"
              className="flex h-9 w-full items-center gap-2 rounded-lg px-2 text-left text-[13px] font-medium text-slate-700 transition hover:bg-slate-50 hover:text-slate-950 focus:bg-slate-50 focus:outline-none"
              data-testid="inputbar-project-context-add-project"
              onFocus={() => setSubmenuOpen(true)}
            >
              <Plus className="h-4 w-4 text-slate-500" />
              <span className="min-w-0 flex-1 truncate">
                {copy.addNewProject}
              </span>
              <ChevronRight className="h-3.5 w-3.5 text-slate-400" />
            </button>
            {submenuOpen ? (
              // Keep the hit area flush with the parent while preserving the visual gap.
              <div
                className="absolute left-full top-0 z-[80] w-[194px] pl-2"
                data-testid="inputbar-project-context-add-submenu"
              >
                <div
                  className="rounded-xl border border-slate-200 bg-white p-1.5 shadow-[0_24px_60px_-30px_rgba(15,23,42,0.52)]"
                  data-testid="inputbar-project-context-add-submenu-panel"
                >
                  <button
                    type="button"
                    className="flex h-9 w-full items-center gap-2 rounded-lg px-2 text-left text-[13px] font-medium text-slate-700 transition hover:bg-slate-50 hover:text-slate-950"
                    data-testid="inputbar-project-context-create-blank"
                    disabled={Boolean(actionState)}
                    onClick={() => void handleCreateBlankProject()}
                  >
                    <FolderPlus className="h-4 w-4 text-emerald-700" />
                    <span className="truncate">{copy.createBlankProject}</span>
                  </button>
                  <button
                    type="button"
                    className="flex h-9 w-full items-center gap-2 rounded-lg px-2 text-left text-[13px] font-medium text-slate-700 transition hover:bg-slate-50 hover:text-slate-950"
                    data-testid="inputbar-project-context-use-existing"
                    disabled={Boolean(actionState)}
                    onClick={() => void handleUseExistingFolder()}
                  >
                    <FolderOpen className="h-4 w-4 text-sky-700" />
                    <span className="truncate">{copy.useExistingFolder}</span>
                  </button>
                </div>
              </div>
            ) : null}
          </div>

          <button
            type="button"
            className="flex h-9 w-full items-center gap-2 rounded-lg px-2 text-left text-[13px] font-medium text-slate-700 transition hover:bg-slate-50 hover:text-slate-950"
            data-testid="inputbar-project-context-no-project"
            onClick={() => {
              onProjectChange?.(null);
              handleProjectPopoverOpenChange(false);
            }}
          >
            <FolderX className="h-4 w-4 text-slate-500" />
            <span className="min-w-0 flex-1 truncate">
              {copy.noProjectAction}
            </span>
            {!normalizedProjectId ? (
              <Check
                aria-label={copy.currentBadge}
                className="h-3.5 w-3.5 text-emerald-700"
              />
            ) : null}
          </button>
        </PopoverContent>
      </Popover>

      {showModeControls ? (
        <>
          <span className="h-4 w-px bg-slate-200" aria-hidden />
          {hasGitRepository ? (
            <Popover open={modePopoverOpen} onOpenChange={setModePopoverOpen}>
              <PopoverTrigger asChild>
                <button
                  type="button"
                  className="inline-flex h-7 min-w-0 max-w-[180px] items-center gap-1.5 rounded-lg px-2 text-xs font-medium text-slate-600 transition hover:bg-white hover:text-slate-950"
                  title={`${copy.modeLabel}: ${resolvedModeLabel}`}
                  data-testid="inputbar-project-context-mode"
                >
                  <Monitor className="h-3.5 w-3.5 shrink-0 text-sky-700" />
                  <span className="truncate">{resolvedModeLabel}</span>
                  <ChevronDown className="h-3 w-3 shrink-0 text-slate-400" />
                </button>
              </PopoverTrigger>
              <PopoverContent
                align="start"
                side="bottom"
                sideOffset={8}
                className="z-[70] w-[216px] rounded-xl border border-slate-200 bg-white p-1.5 text-slate-900 shadow-[0_24px_60px_-30px_rgba(15,23,42,0.52)]"
                data-testid="inputbar-project-context-mode-menu"
                onCloseAutoFocus={(event) => event.preventDefault()}
              >
                <div className="px-2 pb-1 pt-1 text-[11px] font-semibold text-slate-400">
                  {copy.modeMenuTitle}
                </div>
                <button
                  type="button"
                  className="flex h-8 w-full items-center gap-2 rounded-lg px-2 text-left text-[13px] font-medium text-slate-800 transition hover:bg-slate-50"
                  onClick={() => setModePopoverOpen(false)}
                >
                  <Monitor className="h-3.5 w-3.5 text-slate-500" />
                  <span className="min-w-0 flex-1 truncate">
                    {copy.localProcessing}
                  </span>
                  <Check className="h-3.5 w-3.5 text-slate-600" />
                </button>
                <button
                  type="button"
                  className="flex h-8 w-full items-center gap-2 rounded-lg px-2 text-left text-[13px] font-medium text-slate-700 transition hover:bg-slate-50 hover:text-slate-950"
                  disabled={gitActionState === "worktree"}
                  onClick={() => void handleCreateWorktree()}
                >
                  <FolderPlus className="h-3.5 w-3.5 text-slate-500" />
                  <span className="min-w-0 flex-1 truncate">
                    {copy.newWorktree}
                  </span>
                </button>
              </PopoverContent>
            </Popover>
          ) : (
            <span
              className="inline-flex h-7 min-w-0 max-w-[180px] items-center gap-1.5 rounded-lg px-2 text-xs font-medium text-slate-600"
              title={`${copy.modeLabel}: ${resolvedModeLabel}`}
              data-testid="inputbar-project-context-mode"
            >
              <Monitor className="h-3.5 w-3.5 shrink-0 text-sky-700" />
              <span className="truncate">{resolvedModeLabel}</span>
            </span>
          )}
          {hasGitRepository ? (
            <Popover
              open={branchPopoverOpen}
              onOpenChange={setBranchPopoverOpen}
            >
              <PopoverTrigger asChild>
                <button
                  type="button"
                  className="inline-flex h-7 min-w-0 max-w-[180px] items-center gap-1.5 rounded-lg px-2 text-xs font-medium text-slate-600 transition hover:bg-white hover:text-slate-950"
                  title={`${copy.branchLabel}: ${resolvedBranchLabel}`}
                  data-testid="inputbar-project-context-branch"
                >
                  <GitBranch className="h-3.5 w-3.5 shrink-0 text-slate-500" />
                  <span className="truncate">{resolvedBranchLabel}</span>
                  <ChevronDown className="h-3 w-3 shrink-0 text-slate-400" />
                </button>
              </PopoverTrigger>
              <PopoverContent
                align="start"
                side="bottom"
                sideOffset={8}
                className="z-[70] w-[300px] rounded-xl border border-slate-200 bg-white p-1.5 text-slate-900 shadow-[0_24px_60px_-30px_rgba(15,23,42,0.52)]"
                data-testid="inputbar-project-context-branch-menu"
                onCloseAutoFocus={(event) => event.preventDefault()}
              >
                <div className="flex h-8 items-center gap-2 rounded-lg border border-slate-200 bg-white px-2">
                  <Search className="h-3.5 w-3.5 text-slate-400" />
                  <input
                    className="min-w-0 flex-1 bg-transparent text-[13px] font-medium text-slate-900 outline-none placeholder:text-slate-400"
                    value={branchSearchQuery}
                    placeholder={copy.branchSearchPlaceholder}
                    onChange={(event) =>
                      setBranchSearchQuery(event.target.value)
                    }
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && canCreateBranch) {
                        event.preventDefault();
                        void handleCreateBranch();
                      }
                    }}
                  />
                </div>
                <div className="px-2 pb-1 pt-2 text-[11px] font-semibold text-slate-500">
                  {gitRepositoryState?.uncommittedFileCount
                    ? copy.uncommittedFiles(
                        gitRepositoryState.uncommittedFileCount,
                      )
                    : copy.branchLabel}
                </div>
                <div className="max-h-[190px] overflow-auto pb-1">
                  {filteredBranchOptions.map((branch) => (
                    <button
                      key={branch}
                      type="button"
                      className="flex h-10 w-full min-w-0 items-center gap-2 rounded-lg px-2 text-left text-[13px] font-medium text-slate-800 transition hover:bg-slate-50"
                      disabled={Boolean(gitActionState)}
                      onClick={() => void handleCheckoutBranch(branch)}
                    >
                      <GitBranch className="h-4 w-4 shrink-0 text-slate-500" />
                      <span className="min-w-0 flex-1 truncate">{branch}</span>
                      {branch === resolvedBranchLabel ? (
                        <Check className="h-3.5 w-3.5 shrink-0 text-slate-600" />
                      ) : null}
                    </button>
                  ))}
                </div>
                <div className="my-1 h-px bg-slate-100" />
                <button
                  type="button"
                  className={cn(
                    "flex h-9 w-full items-center gap-2 rounded-lg px-2 text-left text-[13px] font-medium transition",
                    canCreateBranch
                      ? "text-slate-700 hover:bg-slate-50 hover:text-slate-950"
                      : "text-slate-400",
                  )}
                  aria-disabled={!canCreateBranch}
                  disabled={!canCreateBranch || Boolean(gitActionState)}
                  onClick={() => void handleCreateBranch()}
                >
                  <Plus className="h-4 w-4 text-slate-400" />
                  <span className="min-w-0 flex-1 truncate">
                    {canCreateBranch
                      ? copy.branchCreateNamedAction(normalizedBranchCreateName)
                      : copy.branchCreateAction}
                  </span>
                </button>
              </PopoverContent>
            </Popover>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
