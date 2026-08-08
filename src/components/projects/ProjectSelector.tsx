/**
 * @file ProjectSelector.tsx
 * @description 项目选择器组件，用于在聊天入口和侧边栏选择项目
 * @module components/projects/ProjectSelector
 * @requirements 4.1, 4.2, 4.3, 4.5
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Check,
  ChevronDown,
  Eye,
  FolderIcon,
  FolderOpen,
  Pencil,
  Plus,
  Search,
  Trash2,
} from "lucide-react";
import { open as openDialog } from "@/lib/desktop-host/dialog";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useProjects } from "@/hooks/useProjects";
import { revealPathInFinder } from "@/lib/api/fileSystem";
import type { ProjectType } from "@/lib/api/project";
import {
  getDefaultProject,
  getProject,
  getProjectByRootPath,
  USER_PROJECT_TYPES,
} from "@/lib/api/project";
import { toProjectView } from "@/lib/projectView";
import type { Project } from "@/types/project";
import { cn } from "@/lib/utils";
import { scheduleMinimumDelayIdleTask } from "@/lib/utils/scheduleMinimumDelayIdleTask";
import { CreateProjectDialog } from "./CreateProjectDialog";
import {
  canDeleteProject,
  canRenameProject,
  getAvailableProjects,
  resolveProjectDeletionFallback,
  resolveSelectedProject,
} from "./projectSelectorUtils";

const PASSIVE_DEFERRED_PROJECT_SUMMARY_LOAD_MS = 12_000;

export interface ProjectSelectorProps {
  /** 当前选中的项目 ID */
  value: string | null;
  /** 选择变化回调 */
  onChange: (projectId: string) => void;
  /** 受控展开状态 */
  open?: boolean;
  /** 展开状态变化回调 */
  onOpenChange?: (open: boolean) => void;
  /** 是否仅作被动展示，不响应点击 */
  passiveTrigger?: boolean;
  /** 按主题类型筛选（可选，不传则显示所有项目） */
  workspaceType?: string;
  /** 占位符文本 */
  placeholder?: string;
  /** 是否禁用 */
  disabled?: boolean;
  /** 自定义类名 */
  className?: string;
  /** 下拉方向 */
  dropdownSide?: "top" | "bottom";
  /** 下拉对齐 */
  dropdownAlign?: "start" | "end";
  /** 是否启用项目管理 */
  enableManagement?: boolean;
  /** 显示密度 */
  density?: "default" | "compact";
  /** 外观表面 */
  chrome?: "default" | "embedded" | "workspace-tab";
  /** 是否跳过默认项目目录健康检查 */
  skipDefaultWorkspaceReadyCheck?: boolean;
  /** 是否延后完整项目列表加载到展开时 */
  deferProjectListLoad?: boolean;
  /** 是否在未选择时自动回落到默认项目 */
  autoSelectFallback?: boolean;
  /** 打开当前项目/工作区内容视图 */
  onOpenProjectContents?: (projectId: string) => void;
}

function detectDesktopPlatform(): "macos" | "windows" | "linux" | "unknown" {
  const platform =
    typeof navigator !== "undefined" ? navigator.platform.toLowerCase() : "";
  const userAgent =
    typeof navigator !== "undefined" ? navigator.userAgent.toLowerCase() : "";
  const source = `${platform} ${userAgent}`;

  if (source.includes("mac")) return "macos";
  if (source.includes("win")) return "windows";
  if (source.includes("linux")) return "linux";
  return "unknown";
}

function isProjectSelectableForWorkspace(
  project: Project | null | undefined,
  workspaceType?: string,
): project is Project {
  if (!project || project.isArchived) {
    return false;
  }

  if (!workspaceType) {
    return true;
  }

  return project.isDefault || project.workspaceType === workspaceType;
}

function resolveDefaultProjectType(workspaceType?: string): ProjectType {
  if (
    workspaceType &&
    USER_PROJECT_TYPES.includes(
      workspaceType as (typeof USER_PROJECT_TYPES)[number],
    )
  ) {
    return workspaceType as ProjectType;
  }

  return "general";
}

interface ProjectSelectorTextCopy {
  defaultProjectBadge: string;
  noDirectory: string;
  pendingProject: string;
  workspaceTypeLabels: Record<string, string>;
}

function formatProjectPathPreview(
  rootPath: string | undefined,
  noDirectoryLabel: string,
): string {
  if (!rootPath) {
    return noDirectoryLabel;
  }

  const segments = rootPath.split(/[\\/]+/).filter(Boolean);
  if (segments.length <= 1) {
    return rootPath;
  }

  if (segments.length === 2) {
    return segments.join("/");
  }

  return `…/${segments.slice(-2).join("/")}`;
}

function getDirectoryName(path: string): string {
  return (
    path
      .split(/[\\/]+/)
      .filter(Boolean)
      .pop() || path
  );
}

function getWorkspaceTypeLabel(
  workspaceTypeLabels: Record<string, string>,
  workspaceType: string,
): string {
  return workspaceTypeLabels[workspaceType] ?? workspaceType;
}

function getProjectMetaText(
  project: Project | null | undefined,
  copy: ProjectSelectorTextCopy,
): string {
  if (!project) {
    return copy.pendingProject;
  }

  const meta = [
    getWorkspaceTypeLabel(copy.workspaceTypeLabels, project.workspaceType),
  ];
  if (project.isDefault) {
    meta.unshift(copy.defaultProjectBadge);
  }

  return meta.join(" · ");
}

function getProjectSummaryText(
  project: Project | null | undefined,
  copy: ProjectSelectorTextCopy,
): string | null {
  if (!project) {
    return null;
  }

  return `${project.name} · ${getProjectMetaText(project, copy)} · ${formatProjectPathPreview(project.rootPath, copy.noDirectory)}`;
}

/**
 * 项目选择器组件
 *
 * 通用对话默认会启用搜索和轻量项目管理。
 */
export function ProjectSelector({
  value,
  onChange,
  open: openProp,
  onOpenChange,
  passiveTrigger = false,
  workspaceType,
  placeholder,
  disabled = false,
  className,
  dropdownSide = "top",
  dropdownAlign = "start",
  enableManagement = false,
  density = "default",
  chrome = "default",
  skipDefaultWorkspaceReadyCheck = false,
  deferProjectListLoad = false,
  autoSelectFallback = true,
  onOpenProjectContents,
}: ProjectSelectorProps) {
  const { t } = useTranslation("common");
  const {
    projects,
    generalProjects,
    defaultProject,
    loading,
    refresh,
    create,
    rename,
    remove,
    getOrCreateDefault,
  } = useProjects({
    skipDefaultWorkspaceReadyCheck,
    autoLoad: !deferProjectListLoad,
  });
  const [internalOpen, setInternalOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [renameTargetId, setRenameTargetId] = useState<string | null>(null);
  const [renameName, setRenameName] = useState("");
  const [isRenaming, setIsRenaming] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isRevealingPath, setIsRevealingPath] = useState(false);
  const [isOpeningExistingFolder, setIsOpeningExistingFolder] = useState(false);
  const [summaryProject, setSummaryProject] = useState<Project | null>(null);
  const [projectListHydrating, setProjectListHydrating] = useState(false);
  const [hasLoadedProjectList, setHasLoadedProjectList] =
    useState(!deferProjectListLoad);
  const projectSummaryLoadDelayMs =
    deferProjectListLoad && passiveTrigger
      ? PASSIVE_DEFERRED_PROJECT_SUMMARY_LOAD_MS
      : 0;
  const compact = density === "compact";
  const embedded = chrome === "embedded";
  const workspaceTab = chrome === "workspace-tab";
  const compactLikeTrigger = compact || embedded || workspaceTab;
  const compactPanel = compact || workspaceTab;
  const open = openProp ?? internalOpen;
  const entityLabel = workspaceTab
    ? t("common.projectSelector.entity.workspace", {})
    : t("common.projectSelector.entity.project", {});
  const createEntityLabel = workspaceTab
    ? t("common.projectSelector.action.createWorkspace", {})
    : t("common.projectSelector.action.createProject", {});
  const managementTitle = workspaceTab
    ? t("common.projectSelector.management.title.workspace", {})
    : t("common.projectSelector.management.title.project", {});
  const managementDescription = workspaceTab
    ? t("common.projectSelector.management.description.workspace", {})
    : t("common.projectSelector.management.description.project", {});
  const revealPathLabel = t(
    `common.projectSelector.action.revealPath.${detectDesktopPlatform()}`,
    {
      defaultValue: t("common.projectSelector.action.revealPath.default", {}),
    },
  );
  const resolvedPlaceholder =
    placeholder ??
    (workspaceTab
      ? t("common.projectSelector.placeholder.workspace", {})
      : t("common.projectSelector.placeholder.project", {}));
  const projectSelectorCopy = useMemo<ProjectSelectorTextCopy>(
    () => ({
      defaultProjectBadge: t("common.projectSelector.meta.default", {}),
      noDirectory: t("common.projectSelector.path.notSet", {}),
      pendingProject: t("common.projectSelector.meta.pending", {}),
      workspaceTypeLabels: {
        persistent: t("common.projectSelector.workspaceType.persistent", {}),
        temporary: t("common.projectSelector.workspaceType.temporary", {}),
        blog: t("common.projectSelector.workspaceType.blog", {}),
        general: t("common.projectSelector.workspaceType.general", {}),
      },
    }),
    [t],
  );

  const projectSource =
    workspaceType === "general" ? generalProjects : projects;

  const availableProjects = useMemo(
    () => getAvailableProjects(projectSource, workspaceType),
    [projectSource, workspaceType],
  );

  const selectedProject = useMemo(() => {
    if (!deferProjectListLoad || hasLoadedProjectList) {
      return resolveSelectedProject(
        availableProjects,
        value,
        autoSelectFallback ? defaultProject : null,
      );
    }

    return summaryProject;
  }, [
    availableProjects,
    autoSelectFallback,
    defaultProject,
    deferProjectListLoad,
    hasLoadedProjectList,
    summaryProject,
    value,
  ]);

  const renameTarget = useMemo(
    () =>
      renameTargetId
        ? availableProjects.find((project) => project.id === renameTargetId) ||
          null
        : null,
    [availableProjects, renameTargetId],
  );

  const deleteTarget = useMemo(
    () =>
      deleteTargetId
        ? availableProjects.find((project) => project.id === deleteTargetId) ||
          null
        : null,
    [availableProjects, deleteTargetId],
  );

  const filteredProjects = useMemo(() => {
    const normalizedQuery = searchQuery.trim().toLowerCase();
    if (!normalizedQuery) {
      return availableProjects;
    }

    return availableProjects.filter(
      (project) =>
        project.name.toLowerCase().includes(normalizedQuery) ||
        project.tags.some((tag) => tag.toLowerCase().includes(normalizedQuery)),
    );
  }, [availableProjects, searchQuery]);

  const defaultProjectType = useMemo(
    () => resolveDefaultProjectType(workspaceType),
    [workspaceType],
  );

  const loadDeferredProjectList = useCallback(() => {
    if (
      !deferProjectListLoad ||
      hasLoadedProjectList ||
      loading ||
      projectListHydrating
    ) {
      return;
    }

    setProjectListHydrating(true);
    void refresh()
      .then(() => {
        setHasLoadedProjectList(true);
      })
      .finally(() => {
        setProjectListHydrating(false);
      });
  }, [
    deferProjectListLoad,
    hasLoadedProjectList,
    loading,
    projectListHydrating,
    refresh,
  ]);

  useEffect(() => {
    if (open) {
      return;
    }

    setSearchQuery("");
  }, [open]);

  useEffect(() => {
    if (!open) {
      return;
    }

    loadDeferredProjectList();
  }, [loadDeferredProjectList, open]);

  useEffect(() => {
    if (!deferProjectListLoad) {
      setHasLoadedProjectList(true);
      setSummaryProject(null);
      setProjectListHydrating(false);
      return;
    }

    let cancelled = false;

    const loadProjectSummary = async () => {
      try {
        const selectedProjectSummaryRaw = value
          ? await getProject(value)
          : null;
        const selectedProjectSummary = selectedProjectSummaryRaw
          ? toProjectView(selectedProjectSummaryRaw)
          : null;
        const resolvedSelectedProject = isProjectSelectableForWorkspace(
          selectedProjectSummary,
          workspaceType,
        )
          ? selectedProjectSummary
          : null;
        const fallbackDefaultProjectRaw =
          autoSelectFallback && !resolvedSelectedProject
            ? await getDefaultProject()
            : null;
        const fallbackDefaultProject = fallbackDefaultProjectRaw
          ? toProjectView(fallbackDefaultProjectRaw)
          : null;
        const resolvedProject = resolvedSelectedProject
          ? resolvedSelectedProject
          : isProjectSelectableForWorkspace(
                fallbackDefaultProject,
                workspaceType,
              )
            ? fallbackDefaultProject
            : null;

        if (cancelled) {
          return;
        }

        setSummaryProject(resolvedProject);
        if (autoSelectFallback && !value && resolvedProject?.id) {
          onChange(resolvedProject.id);
        }
      } catch (error) {
        if (cancelled) {
          return;
        }

        console.error("加载项目摘要失败:", error);
        setSummaryProject(null);
      }
    };

    if (projectSummaryLoadDelayMs > 0) {
      const cancelSummaryLoad = scheduleMinimumDelayIdleTask(
        loadProjectSummary,
        {
          minimumDelayMs: projectSummaryLoadDelayMs,
          idleTimeoutMs: 1_500,
        },
      );
      return () => {
        cancelled = true;
        cancelSummaryLoad();
      };
    }

    void loadProjectSummary();

    return () => {
      cancelled = true;
    };
  }, [
    deferProjectListLoad,
    onChange,
    autoSelectFallback,
    projectSummaryLoadDelayMs,
    value,
    workspaceType,
  ]);

  useEffect(() => {
    if (deferProjectListLoad && !hasLoadedProjectList) {
      return;
    }

    if (loading) {
      return;
    }

    if (!autoSelectFallback) {
      return;
    }

    if (value && availableProjects.some((project) => project.id === value)) {
      return;
    }

    let cancelled = false;

    const ensureSelection = async () => {
      const fallbackProjectId = resolveProjectDeletionFallback(
        availableProjects,
        defaultProject,
        null,
      );

      if (fallbackProjectId) {
        if (!cancelled && fallbackProjectId !== value) {
          onChange(fallbackProjectId);
        }
        return;
      }

      try {
        const createdDefault = await getOrCreateDefault();
        if (!cancelled && createdDefault.id && createdDefault.id !== value) {
          onChange(createdDefault.id);
        }
      } catch (error) {
        console.error("创建默认项目失败:", error);
      }
    };

    void ensureSelection();

    return () => {
      cancelled = true;
    };
  }, [
    availableProjects,
    autoSelectFallback,
    defaultProject,
    deferProjectListLoad,
    getOrCreateDefault,
    hasLoadedProjectList,
    loading,
    onChange,
    value,
  ]);

  const handleOpenChange = (nextOpen: boolean) => {
    if (openProp === undefined) {
      setInternalOpen(nextOpen);
    }
    onOpenChange?.(nextOpen);

    if (nextOpen) {
      loadDeferredProjectList();
    }
  };

  const handleSelect = (projectId: string) => {
    if (projectId !== value) {
      onChange(projectId);
    }
    handleOpenChange(false);
  };

  const handleOpenExistingFolder = async () => {
    setIsOpeningExistingFolder(true);
    try {
      const selectedPath = await openDialog({
        directory: true,
        multiple: false,
      });
      if (typeof selectedPath !== "string" || !selectedPath.trim()) {
        return;
      }

      const rootPath = selectedPath.trim();
      const existingProject = await getProjectByRootPath(rootPath);
      if (existingProject) {
        onChange(existingProject.id);
        handleOpenChange(false);
        return;
      }

      const project = await create({
        name: getDirectoryName(rootPath),
        rootPath,
        workspaceType: defaultProjectType,
      });
      onChange(project.id);
      handleOpenChange(false);
    } catch (error) {
      toast.error(
        t("common.projectSelector.toast.openExistingFolderFailed", {
          message: error instanceof Error ? error.message : String(error),
        }),
      );
    } finally {
      setIsOpeningExistingFolder(false);
    }
  };

  const handleCreateProject = async (
    name: string,
    type: ProjectType,
    rootPath: string,
  ) => {
    const project = await create({
      name,
      rootPath,
      workspaceType: type,
    });
    onChange(project.id);
    handleOpenChange(false);
    toast.success(
      t("common.projectSelector.toast.created", {
        entity: entityLabel,
      }),
    );
  };

  const handleOpenRename = () => {
    const currentSelectedProject = selectedProject;
    if (!currentSelectedProject || !canRenameProject(currentSelectedProject)) {
      return;
    }

    setRenameTargetId(currentSelectedProject.id);
    setRenameName(currentSelectedProject.name);
    handleOpenChange(false);
    setRenameDialogOpen(true);
  };

  const handleConfirmRename = async () => {
    if (!renameTarget || !canRenameProject(renameTarget)) {
      return;
    }

    const nextName = renameName.trim();
    if (!nextName) {
      toast.error(
        t("common.projectSelector.toast.nameRequired", {
          entity: entityLabel,
        }),
      );
      return;
    }

    setIsRenaming(true);
    try {
      await rename(renameTarget.id, nextName);
      setRenameDialogOpen(false);
      setRenameTargetId(null);
      toast.success(
        t("common.projectSelector.toast.renamed", {
          entity: entityLabel,
        }),
      );
    } catch (error) {
      toast.error(
        t("common.projectSelector.toast.renameFailed", {
          message: error instanceof Error ? error.message : String(error),
        }),
      );
    } finally {
      setIsRenaming(false);
    }
  };

  const handleOpenDelete = (projectOverride?: Project) => {
    const currentSelectedProject = projectOverride ?? selectedProject;
    if (!currentSelectedProject || !canDeleteProject(currentSelectedProject)) {
      return;
    }

    setDeleteTargetId(currentSelectedProject.id);
    handleOpenChange(false);
    setDeleteDialogOpen(true);
  };

  const handleOpenSelectedContents = () => {
    if (!selectedProject || !onOpenProjectContents) {
      return;
    }

    if (selectedProject.id !== value) {
      onChange(selectedProject.id);
    }
    handleOpenChange(false);
    onOpenProjectContents(selectedProject.id);
  };

  const handleRevealSelectedPath = async () => {
    const rootPath = selectedProject?.rootPath?.trim();
    if (!rootPath) {
      toast.error(t("common.projectSelector.toast.pathMissing", {}));
      return;
    }

    setIsRevealingPath(true);
    try {
      await revealPathInFinder(rootPath);
    } catch (error) {
      toast.error(
        t("common.projectSelector.toast.revealPathFailed", {
          message: error instanceof Error ? error.message : String(error),
        }),
      );
    } finally {
      setIsRevealingPath(false);
    }
  };

  const handleConfirmDelete = async () => {
    if (!deleteTarget || !canDeleteProject(deleteTarget)) {
      return;
    }

    setIsDeleting(true);
    try {
      const deletedProjectId = deleteTarget.id;
      await remove(deletedProjectId);

      if (value === deletedProjectId) {
        const fallbackProjectId =
          resolveProjectDeletionFallback(
            availableProjects,
            defaultProject,
            deletedProjectId,
          ) || (await getOrCreateDefault()).id;

        if (fallbackProjectId) {
          onChange(fallbackProjectId);
        }
      }

      setDeleteDialogOpen(false);
      setDeleteTargetId(null);
      toast.success(
        t("common.projectSelector.toast.removed", {
          entity: entityLabel,
        }),
      );
    } catch (error) {
      toast.error(
        t("common.projectSelector.toast.removeFailed", {
          message: error instanceof Error ? error.message : String(error),
        }),
      );
    } finally {
      setIsDeleting(false);
    }
  };

  const managementHint =
    selectedProject && !canRenameProject(selectedProject)
      ? t("common.projectSelector.management.defaultLocked", {
          entity: entityLabel,
        })
      : null;
  const canRevealSelectedProjectPath = Boolean(
    selectedProject?.rootPath?.trim(),
  );
  const projectSummaryText = getProjectSummaryText(
    selectedProject,
    projectSelectorCopy,
  );
  const workspaceCurrentProject =
    workspaceTab && value
      ? filteredProjects.find((project) => project.id === value) ??
        (summaryProject?.id === value ? summaryProject : null) ??
        (defaultProject?.id === value ? defaultProject : null)
      : null;
  const visualSelectedProject = workspaceTab
    ? workspaceCurrentProject ??
      selectedProject ??
      (autoSelectFallback ? defaultProject : null) ??
      filteredProjects[0] ??
      null
    : selectedProject;
  const selectedProjectId = visualSelectedProject?.id ?? value ?? null;
  const workspaceMenuProjects =
    workspaceTab && visualSelectedProject
      ? [
          visualSelectedProject,
          ...filteredProjects.filter(
            (project) => project.id !== visualSelectedProject.id,
          ),
        ]
      : filteredProjects;
  const popoverWidthClass = workspaceTab
    ? "w-[360px]"
    : compactPanel
      ? "w-[392px]"
      : "w-[420px]";
  const headerPaddingClass = compactPanel ? "px-4 py-3" : "px-4 py-4";
  const bodyPaddingClass = compactPanel ? "px-4 py-3" : "px-4 py-4";
  const managementPaddingClass = compactPanel ? "px-4 py-3" : "px-4 py-4";
  const displayLoading = loading || projectListHydrating;

  return (
    <>
      <Popover open={open} onOpenChange={handleOpenChange}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            size="sm"
            className={cn(
              workspaceTab
                ? "h-7 min-w-[128px] max-w-[196px] justify-start gap-2 rounded-none border-transparent bg-transparent px-0.5 py-0 text-left shadow-none transition-colors hover:bg-transparent focus-visible:ring-0 dark:hover:bg-transparent"
                : embedded
                  ? "h-10 min-w-[180px] max-w-[280px] justify-between gap-2 rounded-full border-transparent bg-transparent px-1.5 py-0 text-left shadow-none hover:bg-slate-50/80 focus-visible:ring-1 focus-visible:ring-slate-300 focus-visible:ring-offset-0"
                  : compact
                    ? "h-10 min-w-[180px] max-w-[280px] justify-between gap-2 rounded-full border-slate-200/80 bg-white/94 px-2.5 py-0 text-left shadow-sm shadow-slate-950/5 transition-[border-color,box-shadow,background-color] hover:border-slate-300/80 hover:bg-white"
                    : "h-11 min-w-[220px] max-w-[320px] justify-between gap-3 rounded-2xl border-slate-200/80 bg-white/92 px-3 py-0 text-left shadow-sm shadow-slate-950/5 transition-[border-color,box-shadow,background-color] hover:border-slate-300/80 hover:bg-white hover:shadow-md hover:shadow-slate-950/8",
              className,
              passiveTrigger &&
                "pointer-events-none cursor-default select-none focus-visible:ring-0",
            )}
            disabled={disabled || displayLoading}
            tabIndex={passiveTrigger ? -1 : undefined}
            title={
              visualSelectedProject
                ? `${visualSelectedProject.name}\n${visualSelectedProject.rootPath}`
                : resolvedPlaceholder
            }
          >
            <span
              className={cn(
                "flex min-w-0 flex-1 items-center",
                compactLikeTrigger ? "gap-2.5" : "gap-3",
              )}
            >
              {workspaceTab ? (
                <>
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center text-[color:var(--lime-chrome-text)] dark:text-slate-100">
                    <FolderIcon className="h-4 w-4 shrink-0" />
                  </span>
                  <span className="min-w-0 flex flex-1 items-center gap-1.5">
                    <span className="truncate text-[12px] font-semibold leading-none text-[color:var(--lime-chrome-text)] dark:text-slate-100">
                      {visualSelectedProject?.name || resolvedPlaceholder}
                    </span>
                  </span>
                </>
              ) : (
                <>
                  <span
                    className={cn(
                      "flex h-8 w-8 shrink-0 items-center justify-center text-slate-600",
                      "rounded-[14px] border border-slate-200/80 bg-slate-50",
                    )}
                  >
                    {selectedProject?.icon ? (
                      <span
                        className={compactLikeTrigger ? "text-sm" : "text-base"}
                      >
                        {selectedProject.icon}
                      </span>
                    ) : (
                      <FolderIcon className="h-4 w-4 shrink-0 text-muted-foreground" />
                    )}
                  </span>
                  {compactLikeTrigger ? (
                    <span className="min-w-0 flex flex-1 items-center gap-2">
                      <span className="truncate text-sm font-semibold text-slate-900">
                        {selectedProject?.name || resolvedPlaceholder}
                      </span>
                      {selectedProject?.isDefault ? (
                        <span className="shrink-0 rounded-full border border-amber-200/80 bg-amber-50 px-2 py-0.5 text-[10px] font-medium leading-none text-amber-700">
                          {t("common.projectSelector.badge.default", {})}
                        </span>
                      ) : null}
                    </span>
                  ) : (
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center gap-2">
                        <span className="truncate text-sm font-semibold text-slate-900">
                          {selectedProject?.name || resolvedPlaceholder}
                        </span>
                        {selectedProject?.isDefault ? (
                          <span className="shrink-0 rounded-full border border-amber-200/80 bg-amber-50 px-2 py-0.5 text-[10px] font-medium leading-none text-amber-700">
                            {t("common.projectSelector.badge.default", {})}
                          </span>
                        ) : null}
                      </span>
                    </span>
                  )}
                </>
              )}
            </span>
            {!workspaceTab ? (
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-100/90 text-slate-500">
                <ChevronDown
                  className={cn(
                    "h-3.5 w-3.5 transition-transform",
                    open && "rotate-180",
                  )}
                />
              </span>
            ) : null}
          </Button>
        </PopoverTrigger>
        <PopoverContent
          side={dropdownSide}
          align={dropdownAlign}
          className={cn(
            popoverWidthClass,
            workspaceTab
              ? "overflow-hidden rounded-[28px] border border-slate-200 bg-white p-0 shadow-2xl shadow-slate-950/12"
              : "overflow-hidden rounded-[28px] border border-[color:var(--lime-surface-border)] bg-[image:var(--lime-home-card-surface-strong)] p-0 shadow-lg shadow-slate-950/10",
          )}
        >
          {workspaceTab ? (
            <div className="p-3">
              <ScrollArea
                className="max-h-[308px] overflow-y-auto pr-1"
                data-testid="workspace-selector-scroll"
              >
                <div className="space-y-1">
                  {displayLoading ? (
                    <div className="rounded-[22px] px-4 py-8 text-center text-sm text-slate-500">
                      {t("common.loading", {})}
                    </div>
                  ) : workspaceMenuProjects.length === 0 ? (
                    <div className="rounded-[22px] px-4 py-8 text-center text-sm text-slate-500">
                      {t("common.projectSelector.empty", {})}
                    </div>
                  ) : (
                    workspaceMenuProjects.map((project) => {
                      const isSelected = project.id === selectedProjectId;
                      return (
                        <div
                          key={project.id}
                          data-testid={`workspace-selector-row-${project.id}`}
                          data-selected={isSelected ? "true" : "false"}
                          className={cn(
                            "group flex w-full items-center gap-3 rounded-[18px] px-3 py-3 text-left transition-colors",
                            isSelected
                              ? "bg-slate-100 text-slate-900"
                              : "text-slate-800 hover:bg-slate-50",
                          )}
                        >
                          <button
                            type="button"
                            onClick={() => handleSelect(project.id)}
                            className="flex min-w-0 flex-1 items-center gap-3 text-left"
                            aria-current={isSelected ? "page" : undefined}
                          >
                            <FolderIcon className="h-5 w-5 shrink-0 text-slate-500" />
                            <span className="min-w-0 flex-1">
                              <span className="block truncate text-[15px] font-semibold leading-5">
                                {project.name}
                              </span>
                              <span
                                className="mt-1 block truncate text-[13px] leading-5 text-slate-500"
                                title={project.rootPath}
                              >
                                {project.rootPath ||
                                  projectSelectorCopy.noDirectory}
                              </span>
                            </span>
                          </button>
                          <span className="flex h-8 w-8 shrink-0 items-center justify-center">
                            {isSelected ? (
                              <Check className="h-4 w-4 text-slate-500" />
                            ) : enableManagement &&
                              canDeleteProject(project) ? (
                              <button
                                type="button"
                                className="flex h-8 w-8 items-center justify-center rounded-full text-slate-500 opacity-0 transition-[background-color,color,opacity] hover:bg-white hover:text-rose-600 focus-visible:opacity-100 group-hover:opacity-100"
                                aria-label={t(
                                  "common.projectSelector.action.removeEntity",
                                  {
                                    entity: entityLabel,
                                  },
                                )}
                                onClick={(event) => {
                                  event.stopPropagation();
                                  handleOpenDelete(project);
                                }}
                              >
                                <Trash2 className="h-4 w-4" />
                              </button>
                            ) : null}
                          </span>
                        </div>
                      );
                    })
                  )}
                </div>
              </ScrollArea>
              {enableManagement ? (
                <div className="mt-3 border-t border-slate-100 pt-3">
                  <button
                    type="button"
                    className="flex h-11 w-full items-center gap-3 rounded-[16px] px-3 text-left text-[14px] font-medium text-slate-700 transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
                    onClick={() => void handleOpenExistingFolder()}
                    disabled={isOpeningExistingFolder}
                  >
                    <FolderOpen className="h-5 w-5 shrink-0 text-slate-500" />
                    <span>
                      {t(
                        "common.projectSelector.action.openExistingFolder",
                        {},
                      )}
                    </span>
                  </button>
                  <button
                    type="button"
                    className="flex h-11 w-full items-center gap-3 rounded-[16px] px-3 text-left text-[14px] font-medium text-slate-700 transition-colors hover:bg-slate-50"
                    onClick={() => {
                      handleOpenChange(false);
                      setCreateDialogOpen(true);
                    }}
                  >
                    <Plus className="h-5 w-5 shrink-0 text-slate-500" />
                    <span>{createEntityLabel}</span>
                  </button>
                </div>
              ) : null}
            </div>
          ) : (
            <div className="flex flex-col">
              <div
                className={cn(
                  "relative border-b border-white/80",
                  headerPaddingClass,
                )}
              >
                <div className="pointer-events-none absolute -left-10 top-[-24px] h-24 w-24 rounded-full bg-[color:var(--lime-home-glow-secondary)] blur-3xl" />
                <div className="pointer-events-none absolute right-[-20px] top-0 h-20 w-20 rounded-full bg-[color:var(--lime-home-glow-primary)] blur-3xl" />
                <div
                  className={cn(
                    "relative",
                    compact ? "space-y-2.5" : "space-y-3",
                  )}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-slate-900">
                        {t("common.projectSelector.header.title", {
                          entity: entityLabel,
                        })}
                      </div>
                      <div className="mt-1 text-xs leading-5 text-slate-500">
                        {t("common.projectSelector.header.description", {
                          entity: entityLabel,
                        })}
                      </div>
                    </div>
                    <Badge
                      variant="outline"
                      className="border-slate-200/80 bg-white/85 text-slate-600"
                    >
                      {t("common.projectSelector.header.count", {
                        count: filteredProjects.length,
                        entity: entityLabel,
                      })}
                    </Badge>
                  </div>

                  {!compactPanel && projectSummaryText ? (
                    <div className="rounded-[18px] border border-white/90 bg-white/85 px-3 py-2 text-[11px] leading-5 text-slate-600 shadow-sm">
                      <span className="font-medium text-slate-800">
                        {t("common.projectSelector.current.label", {
                          entity: entityLabel,
                        })}
                      </span>
                      <span>{projectSummaryText}</span>
                    </div>
                  ) : null}

                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                    <Input
                      value={searchQuery}
                      onChange={(event) => setSearchQuery(event.target.value)}
                      placeholder={t(
                        "common.projectSelector.search.placeholder",
                        {
                          entity: entityLabel,
                        },
                      )}
                      className={cn(
                        compactPanel ? "h-9" : "h-10",
                        "border-slate-200/80 bg-white/85 pl-9 focus-visible:border-slate-300 focus-visible:ring-1 focus-visible:ring-slate-300 focus-visible:ring-offset-0",
                      )}
                    />
                  </div>
                </div>
              </div>

              <div className={bodyPaddingClass}>
                <ScrollArea
                  className={cn(
                    compactPanel ? "max-h-[280px]" : "max-h-[320px]",
                  )}
                >
                  <div
                    className={cn(
                      "pr-2",
                      compactPanel ? "space-y-2" : "space-y-3",
                    )}
                  >
                    {displayLoading ? (
                      <div className="rounded-[22px] border border-dashed border-slate-300/80 bg-white/80 px-4 py-8 text-center text-sm text-slate-500">
                        {t("common.loading", {})}
                      </div>
                    ) : filteredProjects.length === 0 ? (
                      <div className="rounded-[22px] border border-dashed border-slate-300/80 bg-white/80 px-4 py-8 text-center text-sm text-slate-500">
                        {t("common.projectSelector.empty", {})}
                      </div>
                    ) : (
                      filteredProjects.map((project) => {
                        const isSelected = project.id === selectedProject?.id;
                        return (
                          <button
                            key={project.id}
                            type="button"
                            onClick={() => handleSelect(project.id)}
                            className={cn(
                              compact
                                ? "flex w-full items-center gap-2.5 rounded-[18px] border px-3 py-2.5 text-left transition"
                                : "flex w-full items-center gap-3 rounded-[22px] border px-4 py-3 text-left transition",
                              isSelected
                                ? "border-slate-300 bg-slate-50/85"
                                : "border-slate-200/80 bg-white/85 hover:border-slate-300 hover:bg-white",
                            )}
                          >
                            <div
                              className={cn(
                                "shrink-0 items-center justify-center border border-slate-200/80 bg-slate-50 text-slate-600",
                                compact
                                  ? "flex h-9 w-9 rounded-[14px]"
                                  : "flex h-11 w-11 rounded-[16px]",
                              )}
                            >
                              {project.icon ? (
                                <span
                                  className={compact ? "text-sm" : "text-base"}
                                >
                                  {project.icon}
                                </span>
                              ) : (
                                <FolderIcon className="h-4 w-4 shrink-0 text-muted-foreground" />
                              )}
                            </div>
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-2">
                                <span
                                  className={cn(
                                    "truncate text-slate-900",
                                    compact
                                      ? "text-sm font-semibold"
                                      : "font-medium",
                                  )}
                                >
                                  {project.name}
                                </span>
                                {project.isDefault ? (
                                  <Badge
                                    variant="outline"
                                    className="border-amber-200/80 bg-amber-50 text-[10px] font-medium text-amber-700"
                                  >
                                    {t(
                                      "common.projectSelector.badge.default",
                                      {},
                                    )}
                                  </Badge>
                                ) : null}
                                <Badge
                                  variant="outline"
                                  className="border-slate-200/80 bg-white/80 text-[10px] text-slate-600"
                                >
                                  {getWorkspaceTypeLabel(
                                    projectSelectorCopy.workspaceTypeLabels,
                                    project.workspaceType,
                                  )}
                                </Badge>
                              </div>
                              <div
                                className="mt-1 truncate text-xs text-muted-foreground"
                                title={project.rootPath}
                              >
                                {compact
                                  ? formatProjectPathPreview(
                                      project.rootPath,
                                      projectSelectorCopy.noDirectory,
                                    )
                                  : project.rootPath}
                              </div>
                              {project.tags.length > 0 ? (
                                <div
                                  className={cn(
                                    "flex flex-wrap gap-1.5",
                                    compact ? "mt-1.5" : "mt-2",
                                  )}
                                >
                                  {project.tags.slice(0, 2).map((tag) => (
                                    <Badge
                                      key={tag}
                                      variant="outline"
                                      className="border-slate-200/80 bg-white/70 text-[10px] text-slate-500"
                                    >
                                      {tag}
                                    </Badge>
                                  ))}
                                </div>
                              ) : null}
                            </div>
                            {isSelected ? (
                              <Check className="h-4 w-4 shrink-0 text-slate-700" />
                            ) : null}
                          </button>
                        );
                      })
                    )}
                  </div>
                </ScrollArea>
              </div>

              {enableManagement ? (
                <div
                  className={cn(
                    "border-t border-white/80",
                    managementPaddingClass,
                  )}
                >
                  <div className={cn(compact ? "mb-2.5" : "mb-3")}>
                    <div className="text-sm font-semibold text-slate-900">
                      {managementTitle}
                    </div>
                    <div className="mt-1 text-xs leading-5 text-slate-500">
                      {managementDescription}
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      className={cn(
                        compact ? "h-8 px-3 text-xs" : "h-9",
                        "gap-1.5 rounded-full border border-[color:var(--lime-surface-border-strong)] bg-[image:var(--lime-primary-gradient)] text-white shadow-sm shadow-slate-950/10 hover:opacity-95",
                      )}
                      onClick={() => {
                        handleOpenChange(false);
                        setCreateDialogOpen(true);
                      }}
                    >
                      <Plus className="h-3.5 w-3.5" />
                      {createEntityLabel}
                    </Button>
                    {onOpenProjectContents ? (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className={cn(
                          compact ? "h-8 px-3 text-xs" : "h-9",
                          "gap-1.5 rounded-full border-slate-200/80 bg-white",
                        )}
                        onClick={handleOpenSelectedContents}
                        disabled={!selectedProject}
                      >
                        <Eye className="h-3.5 w-3.5" />
                        {t("common.projectSelector.action.viewContents", {})}
                      </Button>
                    ) : null}
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className={cn(
                        compact ? "h-8 px-3 text-xs" : "h-9",
                        "gap-1.5 rounded-full border-slate-200/80 bg-white",
                      )}
                      onClick={() => void handleRevealSelectedPath()}
                      disabled={
                        !canRevealSelectedProjectPath || isRevealingPath
                      }
                    >
                      <FolderOpen className="h-3.5 w-3.5" />
                      {revealPathLabel}
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className={cn(
                        compact ? "h-8 px-3 text-xs" : "h-9",
                        "gap-1.5 rounded-full border-slate-200/80 bg-white",
                      )}
                      onClick={handleOpenRename}
                      disabled={!canRenameProject(selectedProject)}
                    >
                      <Pencil className="h-3.5 w-3.5" />
                      {t("common.projectSelector.action.rename", {})}
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className={cn(
                        compact ? "h-8 px-3 text-xs" : "h-9",
                        "gap-1.5 rounded-full border-rose-200/80 bg-rose-50/80 text-destructive hover:text-destructive",
                      )}
                      onClick={() => handleOpenDelete()}
                      disabled={!canDeleteProject(selectedProject)}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      {t("common.projectSelector.action.remove", {})}
                    </Button>
                  </div>
                  {managementHint ? (
                    <p className="mt-3 text-xs text-slate-500">
                      {managementHint}
                    </p>
                  ) : null}
                </div>
              ) : null}
            </div>
          )}
        </PopoverContent>
      </Popover>

      <CreateProjectDialog
        open={createDialogOpen}
        onOpenChange={setCreateDialogOpen}
        onSubmit={handleCreateProject}
        defaultType={defaultProjectType}
        allowedTypes={enableManagement ? [defaultProjectType] : undefined}
      />

      <Dialog
        open={renameDialogOpen}
        onOpenChange={(nextOpen) => {
          if (!isRenaming) {
            setRenameDialogOpen(nextOpen);
            if (!nextOpen) {
              setRenameTargetId(null);
            }
          }
        }}
      >
        <DialogContent className="sm:max-w-[460px] overflow-hidden border-slate-200/80 bg-[linear-gradient(180deg,rgba(248,250,252,0.96)_0%,rgba(255,255,255,0.98)_100%)] p-0">
          <DialogHeader className="border-b border-white/80 px-6 py-5">
            <DialogTitle>
              {t("common.projectSelector.rename.title", {
                entity: entityLabel,
              })}
            </DialogTitle>
            <DialogDescription>
              {t("common.projectSelector.rename.description", {
                entity: entityLabel,
              })}
            </DialogDescription>
          </DialogHeader>
          <div className="px-6 py-5">
            <Input
              value={renameName}
              onChange={(event) => setRenameName(event.target.value)}
              placeholder={t("common.projectSelector.rename.placeholder", {})}
              autoFocus
            />
          </div>
          <DialogFooter className="border-t border-white/80 px-6 py-4">
            <Button
              type="button"
              variant="outline"
              className="border-slate-200/80 bg-white"
              onClick={() => {
                if (!isRenaming) {
                  setRenameDialogOpen(false);
                  setRenameTargetId(null);
                }
              }}
              disabled={isRenaming}
            >
              {t("common.cancel", {})}
            </Button>
            <Button
              type="button"
              onClick={() => void handleConfirmRename()}
              disabled={isRenaming}
            >
              {isRenaming
                ? t("common.projectSelector.action.saving", {})
                : t("common.save", {})}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={deleteDialogOpen}
        onOpenChange={(nextOpen) => {
          if (!isDeleting) {
            setDeleteDialogOpen(nextOpen);
            if (!nextOpen) {
              setDeleteTargetId(null);
            }
          }
        }}
      >
        <DialogContent className="sm:max-w-[480px] overflow-hidden border-rose-200/80 bg-[linear-gradient(180deg,rgba(255,241,242,0.96)_0%,rgba(255,255,255,0.98)_100%)] p-0">
          <DialogHeader className="border-b border-white/80 px-6 py-5">
            <DialogTitle className="text-destructive">
              {t("common.projectSelector.remove.title", {
                entity: entityLabel,
              })}
            </DialogTitle>
            <DialogDescription>
              {t("common.projectSelector.remove.description", {
                entity: entityLabel,
                name: deleteTarget ? `「${deleteTarget.name}」` : "",
              })}
            </DialogDescription>
          </DialogHeader>
          <div className="px-6 py-5">
            <div className="rounded-[22px] border border-destructive/20 bg-destructive/5 px-4 py-4 text-sm">
              <p className="font-medium text-destructive">
                {t("common.projectSelector.remove.dangerTitle", {})}
              </p>
              <p className="mt-1 text-muted-foreground">
                {t("common.projectSelector.remove.dangerDescription", {})}
              </p>
            </div>
          </div>
          <DialogFooter className="border-t border-white/80 px-6 py-4">
            <Button
              type="button"
              variant="outline"
              className="border-slate-200/80 bg-white"
              onClick={() => {
                if (!isDeleting) {
                  setDeleteDialogOpen(false);
                  setDeleteTargetId(null);
                }
              }}
              disabled={isDeleting}
            >
              {t("common.cancel", {})}
            </Button>
            <Button
              type="button"
              variant="destructive"
              onClick={() => void handleConfirmDelete()}
              disabled={isDeleting}
            >
              {isDeleting
                ? t("common.projectSelector.action.deleting", {})
                : t("common.projectSelector.action.removeEntity", {
                    entity: entityLabel,
                  })}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
