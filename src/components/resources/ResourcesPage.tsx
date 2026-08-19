import { useCallback, useEffect, useMemo, useState } from "react";
import { open } from "@/lib/desktop-host/dialog";
import { useTranslation } from "react-i18next";
import {
  ArrowRight,
  ArrowUp,
  File,
  FileText,
  Folder,
  Image as ImageIcon,
  Library,
  MoreHorizontal,
  Music2,
  Pencil,
  RefreshCw,
  Search,
  Trash2,
  Video,
  type LucideIcon,
} from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { WorkbenchInfoTip } from "@/components/media/WorkbenchInfoTip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useProjects } from "@/hooks/useProjects";
import { openResourceManager } from "@/features/resource-manager/openResourceManager";
import { inferResourceManagerKind } from "@/features/resource-manager/resourceManagerSession";
import type { ResourceManagerItemInput } from "@/features/resource-manager/types";
import { listMaterials } from "@/lib/api/materials";
import {
  getStoredResourceProjectId,
  onResourceProjectChange,
  setStoredResourceProjectId,
} from "@/lib/resourceProjectSelection";
import { buildHomeAgentParams } from "@/lib/workspace/navigation";
import { CanvasBreadcrumbHeader } from "@/components/workspace/canvas/shared/CanvasBreadcrumbHeader";
import { cn } from "@/lib/utils";
import type { Page, PageParams, ResourcesPageParams } from "@/types/page";
import { ResourcesImageWorkbench } from "./ResourcesImageWorkbench";
import {
  canNavigateResourceFolderUp,
  getCategoryCounts,
  getResourceCollectionSummary,
  getCategoryScopedResources,
  getCurrentFolder,
  getFolderBreadcrumbs,
  getFolderScopedResources,
  matchResourceCategory,
  type ResourceSortDirection,
  type ResourceSortField,
  type ResourceViewCategory,
} from "./services/resourceQueries";
import { fetchDocumentDetail } from "./services/resourceAdapter";
import type { ResourceItem } from "./services/types";
import { useResourcesStore } from "./store";

interface ResourcesPageProps {
  onNavigate?: (page: Page, params?: PageParams) => void;
  pageParams?: ResourcesPageParams;
}

const resourceCategoryItems: Array<{
  key: ResourceViewCategory;
  icon: typeof FileText;
}> = [
  { key: "all", icon: Library },
  { key: "document", icon: FileText },
  { key: "image", icon: ImageIcon },
  { key: "audio", icon: Music2 },
  { key: "video", icon: Video },
];

const resourceCategoryIconMap: Record<ResourceViewCategory, LucideIcon> = {
  all: Library,
  document: FileText,
  image: ImageIcon,
  audio: Music2,
  video: Video,
};

const RESOURCE_PAGE_SIZE = 20;

const formatTime = (timestamp: number): string => {
  return new Date(timestamp).toLocaleString("zh-CN", {
    hour12: false,
  });
};

const getKindIcon = (item: ResourceItem) => {
  if (item.kind === "folder") return Folder;
  if (item.kind === "document") return FileText;
  return File;
};

function buildResourceManagerItemInput(
  item: ResourceItem,
  content?: string | null,
  options?: {
    resourceCategory?: ResourceViewCategory;
  },
): ResourceManagerItemInput {
  const title = item.name || "未命名资源";
  const mimeType = item.mimeType || null;
  const filePath = item.filePath || null;
  const resourceFolderId = item.parentId?.trim() || null;
  const resourceCategory = options?.resourceCategory ?? null;
  const resolvedKind = inferResourceManagerKind({
    src: filePath,
    filePath,
    title,
    mimeType,
    content,
  });

  return {
    id: item.id,
    kind: resolvedKind,
    src: filePath,
    filePath,
    title,
    description: item.description ?? null,
    content: content ?? null,
    mimeType,
    size: item.size ?? null,
    metadata: {
      sourceType: item.sourceType,
      mimeType,
      projectId: item.projectId,
      size: item.size ?? null,
      sourcePage: "resources",
      resourceFolderId,
      resourceCategory,
    },
    sourceContext: {
      kind: "project_resource",
      projectId: item.projectId,
      contentId: item.id,
      sourcePage: "resources",
      resourceFolderId,
      resourceCategory,
    },
  };
}

export function ResourcesPage({ onNavigate, pageParams }: ResourcesPageProps) {
  const { t, i18n } = useTranslation("workspace");
  const {
    projects,
    defaultProject,
    loading: projectsLoading,
    error: projectError,
  } = useProjects();

  const projectId = useResourcesStore((state) => state.projectId);
  const items = useResourcesStore((state) => state.items);
  const loading = useResourcesStore((state) => state.loading);
  const saving = useResourcesStore((state) => state.saving);
  const error = useResourcesStore((state) => state.error);
  const currentFolderId = useResourcesStore((state) => state.currentFolderId);
  const searchQuery = useResourcesStore((state) => state.searchQuery);
  const sortField = useResourcesStore((state) => state.sortField);
  const sortDirection = useResourcesStore((state) => state.sortDirection);
  const setProjectId = useResourcesStore((state) => state.setProjectId);
  const loadResources = useResourcesStore((state) => state.loadResources);
  const refresh = useResourcesStore((state) => state.refresh);
  const setCurrentFolderId = useResourcesStore(
    (state) => state.setCurrentFolderId,
  );
  const setSearchQuery = useResourcesStore((state) => state.setSearchQuery);
  const setSortField = useResourcesStore((state) => state.setSortField);
  const setSortDirection = useResourcesStore((state) => state.setSortDirection);
  const uploadFile = useResourcesStore((state) => state.uploadFile);
  const renameById = useResourcesStore((state) => state.renameById);
  const deleteById = useResourcesStore((state) => state.deleteById);
  const moveToRoot = useResourcesStore((state) => state.moveToRoot);

  const [viewCategory, setViewCategory] = useState<ResourceViewCategory>("all");
  const [currentPage, setCurrentPage] = useState(1);
  const [crossProjectMediaHint, setCrossProjectMediaHint] = useState<{
    projectId: string;
    projectName: string;
    count: number;
    category: "image" | "audio" | "video";
  } | null>(null);
  const focusProjectId = pageParams?.projectId?.trim() || null;
  const focusContentId = pageParams?.contentId?.trim() || null;
  const focusIntentId = pageParams?.focusIntentId?.trim() || null;
  const focusResourceTitle = pageParams?.focusResourceTitle?.trim() || null;
  const focusResourceFolderId = pageParams?.resourceFolderId?.trim() || null;
  const focusResourceCategory = pageParams?.resourceCategory ?? null;

  const availableProjects = useMemo(
    () => projects.filter((project) => !project.isArchived),
    [projects],
  );

  const selectedProject = useMemo(
    () => availableProjects.find((project) => project.id === projectId) ?? null,
    [availableProjects, projectId],
  );

  const categoryCounts = useMemo(() => getCategoryCounts(items), [items]);

  const currentFolder = useMemo(
    () => getCurrentFolder(items, currentFolderId),
    [items, currentFolderId],
  );

  const focusedItem = useMemo(() => {
    if (!focusContentId) {
      return null;
    }

    return items.find((item) => item.id === focusContentId) ?? null;
  }, [focusContentId, items]);

  const breadcrumbs = useMemo(
    () => getFolderBreadcrumbs(items, currentFolderId),
    [items, currentFolderId],
  );

  const canNavigateUp = useMemo(
    () => canNavigateResourceFolderUp(currentFolderId),
    [currentFolderId],
  );

  const folderDisplayItems = useMemo(
    () =>
      getFolderScopedResources(
        items,
        currentFolderId,
        searchQuery,
        sortField,
        sortDirection,
      ),
    [items, currentFolderId, searchQuery, sortField, sortDirection],
  );

  const isFolderMode = viewCategory === "all";

  const displayItems = useMemo(() => {
    if (isFolderMode) {
      return folderDisplayItems;
    }

    return getCategoryScopedResources(
      items,
      viewCategory,
      searchQuery,
      sortField,
      sortDirection,
    );
  }, [
    isFolderMode,
    items,
    searchQuery,
    sortDirection,
    sortField,
    viewCategory,
    folderDisplayItems,
  ]);

  const totalPages = useMemo(
    () => Math.max(1, Math.ceil(displayItems.length / RESOURCE_PAGE_SIZE)),
    [displayItems.length],
  );

  const safeCurrentPage = Math.min(Math.max(currentPage, 1), totalPages);

  const pagedDisplayItems = useMemo(() => {
    const startIndex = (safeCurrentPage - 1) * RESOURCE_PAGE_SIZE;
    return displayItems.slice(startIndex, startIndex + RESOURCE_PAGE_SIZE);
  }, [displayItems, safeCurrentPage]);

  const pageRangeStart =
    displayItems.length === 0
      ? 0
      : (safeCurrentPage - 1) * RESOURCE_PAGE_SIZE + 1;

  const pageRangeEnd =
    displayItems.length === 0
      ? 0
      : Math.min(safeCurrentPage * RESOURCE_PAGE_SIZE, displayItems.length);

  useEffect(() => {
    if (projectId || projectsLoading) return;

    const storedProjectId = getStoredResourceProjectId({ includeLegacy: true });
    if (
      storedProjectId &&
      availableProjects.some((project) => project.id === storedProjectId)
    ) {
      setProjectId(storedProjectId);
      return;
    }

    const preferredProject =
      (defaultProject && !defaultProject.isArchived ? defaultProject : null) ??
      availableProjects[0];
    if (!preferredProject) return;

    setProjectId(preferredProject.id);
  }, [
    availableProjects,
    defaultProject,
    projectId,
    projectsLoading,
    setProjectId,
  ]);

  useEffect(() => {
    if (
      !focusProjectId ||
      focusProjectId === projectId ||
      !availableProjects.some((project) => project.id === focusProjectId)
    ) {
      return;
    }

    setProjectId(focusProjectId);
  }, [availableProjects, focusProjectId, projectId, setProjectId]);

  useEffect(() => {
    setStoredResourceProjectId(projectId, {
      source: "resources",
      emitEvent: true,
    });
  }, [projectId]);

  useEffect(() => {
    return onResourceProjectChange((detail) => {
      if (
        detail.source !== "image-gen-save" &&
        detail.source !== "general-chat"
      ) {
        return;
      }

      if (!detail.projectId || detail.projectId === projectId) {
        return;
      }

      if (
        !availableProjects.some((project) => project.id === detail.projectId)
      ) {
        return;
      }

      setProjectId(detail.projectId);
    });
  }, [availableProjects, projectId, setProjectId]);

  useEffect(() => {
    if (!projectId) return;
    void loadResources();
  }, [projectId, loadResources]);

  useEffect(() => {
    setCurrentPage(1);
  }, [
    currentFolderId,
    projectId,
    searchQuery,
    sortDirection,
    sortField,
    viewCategory,
  ]);

  useEffect(() => {
    if (currentPage !== safeCurrentPage) {
      setCurrentPage(safeCurrentPage);
    }
  }, [currentPage, safeCurrentPage]);

  useEffect(() => {
    if (!focusIntentId || !focusedItem) {
      return;
    }

    const nextCategory =
      focusResourceCategory &&
      matchResourceCategory(focusedItem, focusResourceCategory)
        ? focusResourceCategory
        : "all";
    setViewCategory(nextCategory);
    setCurrentFolderId(focusResourceFolderId ?? focusedItem.parentId ?? null);
    if (searchQuery.trim()) {
      setSearchQuery("");
    }
  }, [
    focusIntentId,
    focusedItem,
    focusResourceCategory,
    focusResourceFolderId,
    searchQuery,
    setCurrentFolderId,
    setSearchQuery,
  ]);

  useEffect(() => {
    if (!focusIntentId || !focusContentId) {
      return;
    }

    const focusIndex = displayItems.findIndex(
      (item) => item.id === focusContentId,
    );
    if (focusIndex < 0) {
      return;
    }

    setCurrentPage(Math.floor(focusIndex / RESOURCE_PAGE_SIZE) + 1);
  }, [displayItems, focusContentId, focusIntentId]);

  useEffect(() => {
    if (
      loading ||
      !projectId ||
      (viewCategory !== "image" &&
        viewCategory !== "audio" &&
        viewCategory !== "video")
    ) {
      setCrossProjectMediaHint(null);
      return;
    }

    if (categoryCounts[viewCategory] > 0) {
      setCrossProjectMediaHint(null);
      return;
    }

    const candidateProjects = availableProjects.filter(
      (project) => project.id !== projectId,
    );
    if (candidateProjects.length === 0) {
      setCrossProjectMediaHint(null);
      return;
    }

    let disposed = false;
    void (async () => {
      const results = await Promise.all(
        candidateProjects.map(async (project) => {
          try {
            const materials = await listMaterials(project.id, {
              type: viewCategory,
            });
            return {
              projectId: project.id,
              projectName: project.name,
              count: materials.length,
            };
          } catch {
            return {
              projectId: project.id,
              projectName: project.name,
              count: 0,
            };
          }
        }),
      );

      if (disposed) {
        return;
      }

      const matched = results.find((item) => item.count > 0);
      if (!matched) {
        setCrossProjectMediaHint(null);
        return;
      }

      setCrossProjectMediaHint({
        ...matched,
        category: viewCategory,
      });
    })();

    return () => {
      disposed = true;
    };
  }, [availableProjects, categoryCounts, loading, projectId, viewCategory]);

  const handleUploadImage = useCallback(async () => {
    if (!projectId) return;

    const selected = await open({
      directory: false,
      multiple: false,
      title: "选择本地图片",
      filters: [
        {
          name: "图片",
          extensions: [
            "jpg",
            "jpeg",
            "png",
            "webp",
            "gif",
            "bmp",
            "svg",
            "ico",
            "heic",
          ],
        },
      ],
    });
    if (!selected || Array.isArray(selected)) return;

    await uploadFile(selected);
  }, [projectId, uploadFile]);

  const handleRename = useCallback(
    async (item: ResourceItem) => {
      const name = window.prompt("请输入新名称", item.name);
      if (!name?.trim() || name.trim() === item.name) return;
      await renameById(item.id, name.trim());
    },
    [renameById],
  );

  const handleDelete = useCallback(
    async (item: ResourceItem) => {
      const confirmed = window.confirm(
        `确定删除「${item.name}」吗？该操作无法撤销。`,
      );
      if (!confirmed) return;
      await deleteById(item.id);
    },
    [deleteById],
  );

  const getKindLabel = useCallback(
    (kind: ResourceItem["kind"]) => t(`workspace.resources.kind.${kind}`),
    [t],
  );
  const getSourceLabel = useCallback(
    (sourceType: ResourceItem["sourceType"]) =>
      t(`workspace.resources.source.${sourceType}`),
    [t],
  );
  const getCategoryLabel = useCallback(
    (category: ResourceViewCategory) =>
      t(`workspace.resources.category.${category}`),
    [t],
  );
  const getSortFieldLabel = useCallback(
    (field: ResourceSortField) => t(`workspace.resources.sort.${field}`),
    [t],
  );
  const getSortDirectionLabel = useCallback(
    (direction: ResourceSortDirection) =>
      t(`workspace.resources.sortDirection.${direction}`),
    [t],
  );

  const handleOpenFile = useCallback(
    async (item: ResourceItem) => {
      if (!item.filePath) {
        toast.error("该文件缺少本地路径，无法打开资源管理器");
        return;
      }

      const candidates = displayItems.filter(
        (candidate) => candidate.kind === "file" && candidate.filePath,
      );
      const sourceItems = candidates.length > 0 ? candidates : [item];
      const initialIndex = Math.max(
        0,
        sourceItems.findIndex((candidate) => candidate.id === item.id),
      );
      const sessionId = await openResourceManager({
        sourceLabel: getCategoryLabel(viewCategory),
        initialIndex,
        items: sourceItems.map((candidate) =>
          buildResourceManagerItemInput(candidate, null, {
            resourceCategory: viewCategory,
          }),
        ),
      });

      if (!sessionId) {
        toast.error("该文件缺少可预览地址，无法打开资源管理器");
      }
    },
    [displayItems, getCategoryLabel, viewCategory],
  );

  const handleOpenDocument = useCallback(
    async (item: ResourceItem) => {
      try {
        const documentCandidates = displayItems.filter(
          (candidate) => candidate.kind === "document",
        );
        const sourceItems = documentCandidates.some(
          (candidate) => candidate.id === item.id,
        )
          ? documentCandidates
          : [item, ...documentCandidates];

        const resolvedItems = (
          await Promise.all(
            sourceItems.map(async (candidate) => {
              try {
                const detail = await fetchDocumentDetail(candidate.id);
                if (!detail) {
                  if (candidate.id === item.id) {
                    throw new Error("文档不存在或已被删除");
                  }
                  return null;
                }
                if (!detail.body) {
                  return null;
                }

                return buildResourceManagerItemInput(
                  {
                    ...candidate,
                    name: detail.title || candidate.name,
                    mimeType: candidate.mimeType || "text/markdown",
                    size: candidate.size ?? detail.word_count,
                  },
                  detail.body,
                  {
                    resourceCategory: viewCategory,
                  },
                );
              } catch (detailError) {
                if (candidate.id === item.id) {
                  throw detailError;
                }
                return null;
              }
            }),
          )
        ).filter(
          (candidate): candidate is ResourceManagerItemInput =>
            candidate !== null,
        );

        const initialIndex = resolvedItems.findIndex(
          (candidate) => candidate.id === item.id,
        );
        if (initialIndex < 0) {
          toast.error("文档内容为空，无法打开资源管理器");
          return;
        }

        const sessionId = await openResourceManager({
          sourceLabel: t("workspace.resources.title"),
          initialIndex,
          items: resolvedItems,
        });
        if (!sessionId) {
          toast.error("文档内容为空，无法打开资源管理器");
        }
      } catch (detailError) {
        toast.error(
          detailError instanceof Error
            ? detailError.message
            : String(detailError),
        );
      }
    },
    [displayItems, t, viewCategory],
  );

  const handleOpenResource = useCallback(
    async (item: ResourceItem) => {
      if (item.kind === "folder") {
        setCurrentFolderId(item.id);
        return;
      }
      if (item.kind === "document") {
        await handleOpenDocument(item);
        return;
      }
      await handleOpenFile(item);
    },
    [handleOpenDocument, handleOpenFile, setCurrentFolderId],
  );

  const handleNavigateUp = useCallback(() => {
    if (!canNavigateUp) return;
    setCurrentFolderId(currentFolder?.parentId ?? null);
  }, [canNavigateUp, currentFolder?.parentId, setCurrentFolderId]);

  const handleBackHome = useCallback(() => {
    onNavigate?.("agent", buildHomeAgentParams());
  }, [onNavigate]);

  const activeCategoryLabel = getCategoryLabel(viewCategory);
  const ActiveCategoryIcon = resourceCategoryIconMap[viewCategory];

  const projectSummary = useMemo(
    () => getResourceCollectionSummary(items),
    [items],
  );

  const defaultScopeStatusDescription = useMemo(() => {
    if (!projectId) {
      return t("workspace.resources.scope.noProject");
    }
    if (!isFolderMode) {
      return t("workspace.resources.scope.categoryDescription", {
        label: activeCategoryLabel,
      });
    }
    if (currentFolder && breadcrumbs.length > 0) {
      return t("workspace.resources.scope.path", {
        path: breadcrumbs.map((folder) => folder.name).join(" / "),
      });
    }
    return t("workspace.resources.scope.rootDescription");
  }, [
    activeCategoryLabel,
    breadcrumbs,
    currentFolder,
    isFolderMode,
    projectId,
    t,
  ]);

  const scopeModeLabel = isFolderMode
    ? t("workspace.resources.scope.folderMode")
    : t("workspace.resources.scope.categoryMode", {
        label: activeCategoryLabel,
      });

  const scopeStatusDescription = useMemo(() => {
    if (!crossProjectMediaHint) {
      return defaultScopeStatusDescription;
    }

    const categoryLabel = getCategoryLabel(crossProjectMediaHint.category);
    return t("workspace.resources.scope.crossProjectHint", {
      category: categoryLabel,
      count: crossProjectMediaHint.count,
      projectName: crossProjectMediaHint.projectName,
    });
  }, [crossProjectMediaHint, defaultScopeStatusDescription, getCategoryLabel, t]);

  const projectSummaryLabel = useMemo(() => {
    if (!projectId) {
      return null;
    }

    const latestUpdateLabel =
      projectSummary.latestUpdatedAt === null
        ? t("workspace.resources.summary.noUpdates")
        : new Date(projectSummary.latestUpdatedAt).toLocaleString(
            i18n.language,
            { hour12: false },
          );

    return t("workspace.resources.summary.project", {
      contentItemCount: projectSummary.contentItemCount,
      folderCount: projectSummary.folderCount,
      latestUpdateLabel,
    });
  }, [i18n.language, projectId, projectSummary, t]);

  const showEmptyState = projectId && !loading && displayItems.length === 0;
  const focusStatusTitle =
    focusResourceTitle || focusedItem?.name || focusContentId || null;
  const showFocusStatus = Boolean(focusIntentId && focusStatusTitle);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-[linear-gradient(180deg,rgba(248,250,252,1)_0%,rgba(244,249,248,0.96)_52%,rgba(248,250,252,1)_100%)]">
      <div className="border-b border-slate-200/70 bg-white">
        <div className="mx-auto w-full max-w-[1480px] px-4 py-3 lg:px-6">
          <CanvasBreadcrumbHeader
            label={t("workspace.resources.title")}
            onBackHome={handleBackHome}
          />
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        <div className="mx-auto flex min-h-full w-full max-w-[1480px] flex-col gap-6 px-4 py-5 lg:px-6 lg:py-6">
          <section className="rounded-[26px] border border-slate-200/80 bg-white p-5 shadow-sm shadow-slate-950/5">
            <div className="flex flex-col gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
                  {t("workspace.resources.title")}
                </h1>
                <WorkbenchInfoTip
                  ariaLabel={t("workspace.resources.hero.tipAria")}
                  content={t("workspace.resources.hero.tip")}
                  tone="mint"
                />
                <span
                  className={cn(
                    "inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium",
                    saving
                      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                      : "border-slate-200 bg-slate-50 text-slate-600",
                  )}
                >
                  {saving && (
                    <RefreshCw className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  )}
                  {saving
                    ? t("workspace.resources.status.syncing")
                    : t("workspace.resources.status.ready")}
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Badge
                  variant="outline"
                  className="rounded-full border-slate-200 bg-slate-50 px-3 py-1 text-slate-700"
                >
                  {selectedProject?.name ??
                    t("workspace.resources.project.unselected")}
                </Badge>
                {projectSummaryLabel && (
                  <Badge
                    variant="outline"
                    className="rounded-full border-slate-200 bg-white px-3 py-1 text-slate-600"
                  >
                    {projectSummaryLabel}
                  </Badge>
                )}
                {showFocusStatus ? (
                  <Badge
                    variant="outline"
                    className={cn(
                      "rounded-full px-3 py-1",
                      focusedItem
                        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                        : "border-amber-200 bg-amber-50 text-amber-700",
                    )}
                    data-testid="resources-focus-status"
                  >
                    {t(
                      focusedItem
                        ? "workspace.resources.focus.located"
                        : "workspace.resources.focus.locating",
                      { title: focusStatusTitle },
                    )}
                  </Badge>
                ) : null}
                {searchQuery.trim() ? (
                  <Badge
                    variant="outline"
                    className="rounded-full border-sky-200 bg-sky-50 px-3 py-1 text-sky-700"
                  >
                    {t("workspace.resources.search.badge", {
                      query: searchQuery.trim(),
                    })}
                  </Badge>
                ) : (
                  <Badge
                    variant="outline"
                    className="rounded-full border-slate-200 bg-white px-3 py-1 text-slate-500"
                  >
                    {t("workspace.resources.sort.status", {
                      direction: getSortDirectionLabel(sortDirection),
                      field: getSortFieldLabel(sortField),
                    })}
                  </Badge>
                )}
              </div>
            </div>
          </section>

          <section
            className="rounded-[26px] border border-sky-200/80 bg-[linear-gradient(135deg,rgba(239,246,255,0.96)_0%,rgba(255,255,255,0.98)_100%)] p-5 shadow-sm shadow-slate-950/5"
            data-testid="resources-migration-callout"
          >
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div className="space-y-1.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center rounded-full border border-sky-200 bg-white px-2.5 py-1 text-[11px] font-medium text-sky-700">
                    {t("workspace.resources.callout.badge")}
                  </span>
                  <h2 className="text-base font-semibold text-slate-900">
                    {t("workspace.resources.callout.title")}
                  </h2>
                </div>
                <p className="text-sm leading-6 text-slate-600">
                  {t("workspace.resources.callout.description")}
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  type="button"
                  variant="outline"
                  className="rounded-2xl border-slate-200 bg-white text-slate-700 hover:bg-slate-100 hover:text-slate-900"
                  onClick={handleBackHome}
                >
                  {t("workspace.resources.action.backToGenerate")}
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </div>
          </section>

          <div className="grid gap-6 xl:grid-cols-[290px_minmax(0,1fr)]">
            <aside className="space-y-4">
              <section className="rounded-[26px] border border-slate-200/80 bg-white/90 p-4 shadow-sm shadow-slate-950/5">
                <div className="px-1">
                  <div className="flex flex-wrap items-center gap-2 text-sm font-semibold text-slate-900">
                    <span>{t("workspace.resources.category.title")}</span>
                    <WorkbenchInfoTip
                      ariaLabel={t("workspace.resources.category.tipAria")}
                      content={t("workspace.resources.category.tip")}
                      tone="slate"
                    />
                  </div>
                </div>

                <div className="mt-4 space-y-2">
                  {resourceCategoryItems.map(({ key, icon: Icon }) => {
                    const label = getCategoryLabel(key);
                    return (
                    <button
                      key={key}
                      type="button"
                      className={cn(
                        "flex w-full items-center justify-between rounded-2xl border border-transparent px-3 py-3 text-left text-sm transition hover:border-slate-200 hover:bg-slate-50",
                        viewCategory === key &&
                          "border-emerald-200 bg-[linear-gradient(135deg,rgba(240,253,250,0.98)_0%,rgba(236,253,245,0.96)_52%,rgba(224,242,254,0.95)_100%)] text-slate-800 shadow-sm shadow-emerald-950/10 hover:opacity-95",
                      )}
                      onClick={() => setViewCategory(key)}
                    >
                      <span className="flex items-center gap-3">
                        <span
                          className={cn(
                            "flex h-10 w-10 items-center justify-center rounded-2xl border",
                            viewCategory === key
                              ? "border-emerald-200 bg-white/90 text-emerald-700"
                              : "border-slate-200 bg-slate-50 text-slate-600",
                          )}
                        >
                          <Icon className="h-4 w-4" />
                        </span>
                        <span>
                          <span className="block font-medium">{label}</span>
                          <span
                            className={cn(
                              "block text-xs",
                              viewCategory === key
                                ? "text-slate-600"
                                : "text-slate-500",
                            )}
                          >
                            {key === "all"
                              ? t("workspace.resources.category.allHint")
                              : t("workspace.resources.category.kindHint", {
                                  label,
                                })}
                          </span>
                        </span>
                      </span>
                      <span
                        className={cn(
                          "rounded-full px-2.5 py-1 text-xs font-medium",
                          viewCategory === key
                            ? "border border-emerald-200 bg-white/90 text-emerald-700"
                            : "bg-slate-100 text-slate-600",
                        )}
                      >
                        {categoryCounts[key]}
                      </span>
                    </button>
                    );
                  })}
                </div>
              </section>

              <section className="rounded-[26px] border border-slate-200/80 bg-white/90 p-4 shadow-sm shadow-slate-950/5">
                <div className="px-1">
                  <div className="flex flex-wrap items-center gap-2 text-sm font-semibold text-slate-900">
                    <span>{t("workspace.resources.projectSwitcher.title")}</span>
                    <WorkbenchInfoTip
                      ariaLabel={t(
                        "workspace.resources.projectSwitcher.tipAria",
                      )}
                      content={t("workspace.resources.projectSwitcher.tip")}
                      tone="slate"
                    />
                  </div>
                  <p className="mt-1 text-xs leading-5 text-slate-500">
                    {t("workspace.resources.projectSwitcher.description")}
                  </p>
                </div>

                <ScrollArea className="mt-4 h-[388px] pr-1">
                  <div className="space-y-2">
                    {projectsLoading ? (
                      <div className="rounded-2xl border border-dashed border-slate-200 px-3 py-4 text-sm text-slate-500">
                        {t("workspace.resources.project.loading")}
                      </div>
                    ) : availableProjects.length === 0 ? (
                      <div className="rounded-2xl border border-dashed border-slate-200 px-3 py-4 text-sm text-slate-500">
                        {t("workspace.resources.project.empty")}
                      </div>
                    ) : (
                      availableProjects.map((project) => (
                        <button
                          key={project.id}
                          type="button"
                          className={cn(
                            "w-full rounded-2xl border border-transparent bg-white/80 px-3 py-3 text-left transition hover:border-slate-200 hover:bg-slate-50",
                            project.id === projectId &&
                              "border-slate-300 bg-slate-50 shadow-sm",
                          )}
                          onClick={() => setProjectId(project.id)}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <div className="truncate font-medium text-slate-900">
                                {project.name}
                              </div>
                              <div className="mt-1 text-xs text-slate-500">
                                {project.id.slice(0, 8)}
                              </div>
                            </div>
                            {project.id === projectId && (
                              <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-1 text-[11px] font-medium text-emerald-700">
                                {t("workspace.resources.project.current")}
                              </span>
                            )}
                          </div>
                        </button>
                      ))
                    )}
                  </div>
                </ScrollArea>
              </section>
            </aside>

            <section className="min-w-0 space-y-4">
              <div className="rounded-[26px] border border-slate-200/80 bg-white/90 p-5 shadow-sm shadow-slate-950/5">
                <div className="flex flex-col gap-3 xl:flex-row xl:items-center">
                  <div className="relative min-w-0 flex-1">
                    <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                    <Input
                      value={searchQuery}
                      onChange={(event) => setSearchQuery(event.target.value)}
                      placeholder={
                        isFolderMode
                          ? t("workspace.resources.search.folderPlaceholder")
                          : t("workspace.resources.search.categoryPlaceholder")
                      }
                      className="h-11 rounded-xl border-slate-200 bg-slate-50/80 pl-10 shadow-none focus-visible:ring-slate-300"
                    />
                  </div>

                  <div className="flex flex-wrap items-center gap-3 xl:flex-nowrap">
                    <Select
                      value={sortField}
                      onValueChange={(value) =>
                        setSortField(value as ResourceSortField)
                      }
                    >
                      <SelectTrigger className="h-11 w-[160px] rounded-xl border-slate-200 bg-white text-slate-700 shadow-none">
                        <span>{getSortFieldLabel(sortField)}</span>
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="updatedAt">
                          {t("workspace.resources.sort.updatedAt")}
                        </SelectItem>
                        <SelectItem value="createdAt">
                          {t("workspace.resources.sort.createdAt")}
                        </SelectItem>
                        <SelectItem value="name">
                          {t("workspace.resources.sort.name")}
                        </SelectItem>
                      </SelectContent>
                    </Select>

                    <Button
                      variant="outline"
                      className="h-11 rounded-xl border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                      onClick={() =>
                        setSortDirection(
                          sortDirection === "asc" ? "desc" : "asc",
                        )
                      }
                    >
                      {getSortDirectionLabel(sortDirection)}
                    </Button>

                    <Button
                      variant="outline"
                      className="rounded-xl border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                      onClick={refresh}
                      disabled={!projectId || loading}
                    >
                      <RefreshCw
                        className={cn(
                          "mr-2 h-4 w-4",
                          loading && "animate-spin",
                        )}
                      />
                      {t("workspace.resources.action.refresh")}
                    </Button>
                    <Button
                      variant="outline"
                      className="rounded-xl border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                      onClick={handleNavigateUp}
                      disabled={!projectId || !isFolderMode || !canNavigateUp}
                    >
                      <ArrowUp className="mr-2 h-4 w-4" />
                      {t("workspace.resources.action.backFolder")}
                    </Button>
                  </div>
                </div>

                <div
                  className={cn(
                    "mt-4 rounded-2xl border px-4 py-3",
                    crossProjectMediaHint
                      ? "border-amber-300/70 bg-amber-50/90"
                      : "border-slate-200/80 bg-slate-50/80",
                  )}
                >
                  <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                    <div className="min-w-0">
                      <div
                        className={cn(
                          "flex flex-wrap items-center gap-2 text-sm font-medium",
                          crossProjectMediaHint
                            ? "text-amber-900"
                            : "text-slate-700",
                        )}
                      >
                        <ActiveCategoryIcon
                          className={cn(
                            "h-4 w-4",
                            crossProjectMediaHint
                              ? "text-amber-700"
                              : "text-emerald-600",
                          )}
                        />
                        <span>{t("workspace.resources.scope.title")}</span>
                        <Badge
                          variant="outline"
                          className={cn(
                            "rounded-full bg-white px-3 py-1",
                            crossProjectMediaHint
                              ? "border-amber-200 text-amber-800"
                              : "border-slate-200 text-slate-600",
                          )}
                        >
                          {scopeModeLabel}
                        </Badge>
                      </div>
                      <p
                        className={cn(
                          "mt-1 text-sm leading-6",
                          crossProjectMediaHint
                            ? "text-amber-900"
                            : "text-slate-500",
                        )}
                      >
                        {scopeStatusDescription}
                      </p>
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                      {crossProjectMediaHint ? (
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          className="rounded-xl border-amber-300 bg-white text-amber-900 hover:bg-amber-100"
                          onClick={() =>
                            setProjectId(crossProjectMediaHint.projectId)
                          }
                        >
                          {t("workspace.resources.scope.switchProject")}
                        </Button>
                      ) : isFolderMode ? (
                        <>
                          <button
                            className={cn(
                              "rounded-full border px-3 py-1.5 text-sm transition hover:border-slate-300 hover:bg-slate-50",
                              currentFolderId === null
                                ? "border-slate-300 bg-slate-100 text-slate-900"
                                : "border-slate-200 bg-white text-slate-600",
                            )}
                            onClick={() => setCurrentFolderId(null)}
                            type="button"
                          >
                            {t("workspace.resources.scope.root")}
                          </button>
                          {breadcrumbs.map((folder) => (
                            <button
                              key={folder.id}
                              className={cn(
                                "rounded-full border px-3 py-1.5 text-sm transition hover:border-slate-300 hover:bg-slate-50",
                                currentFolderId === folder.id
                                  ? "border-slate-300 bg-slate-100 text-slate-900"
                                  : "border-slate-200 bg-white text-slate-600",
                              )}
                              onClick={() => setCurrentFolderId(folder.id)}
                              type="button"
                            >
                              / {folder.name}
                            </button>
                          ))}
                        </>
                      ) : (
                        <span className="rounded-full border border-sky-200 bg-white px-3 py-1.5 text-sm text-sky-700">
                          {t("workspace.resources.scope.categoryView", {
                            label: activeCategoryLabel,
                          })}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {(error || projectError) && (
                <div className="rounded-[22px] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 shadow-sm">
                  {error || projectError}
                </div>
              )}

              {viewCategory === "image" && (
                <ResourcesImageWorkbench
                  projectId={projectId}
                  onNavigate={onNavigate}
                  onUploadImage={handleUploadImage}
                />
              )}

              <div className="overflow-hidden rounded-[26px] border border-slate-200/80 bg-white/95 shadow-sm shadow-slate-950/5">
                <div className="border-b border-slate-200/80 px-5 py-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="text-lg font-semibold tracking-tight text-slate-900">
                      {t("workspace.resources.list.title")}
                    </h3>
                    <WorkbenchInfoTip
                      ariaLabel={t("workspace.resources.list.tipAria")}
                      content={t("workspace.resources.list.tip")}
                      tone="slate"
                    />
                  </div>
                </div>

                <div className="p-5">
                  {!projectId ? (
                    <div className="rounded-[24px] border border-dashed border-slate-300 bg-slate-50/80 p-10 text-center text-sm text-slate-500">
                      {t("workspace.resources.empty.selectProject")}
                    </div>
                  ) : loading ? (
                    <div className="rounded-[24px] border border-dashed border-slate-300 bg-slate-50/80 p-10 text-center text-sm text-slate-500">
                      {t("workspace.resources.empty.loading")}
                    </div>
                  ) : showEmptyState ? (
                    <div className="flex min-h-[320px] flex-col items-center justify-center rounded-[24px] border border-dashed border-slate-300 bg-[linear-gradient(135deg,rgba(248,250,252,0.98)_0%,rgba(243,249,247,0.96)_100%)] px-6 py-12 text-center">
                      <div className="flex h-16 w-16 items-center justify-center rounded-[20px] border border-white/90 bg-white/90 text-slate-700 shadow-sm">
                        <FileText className="h-7 w-7" />
                      </div>
                      <h3 className="mt-6 text-2xl font-semibold tracking-tight text-slate-900">
                        {t("workspace.resources.empty.title")}
                      </h3>
                      <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">
                        {t("workspace.resources.empty.description")}
                      </p>
                    </div>
                  ) : (
                    <div className="overflow-x-auto rounded-[22px] border border-slate-200/80">
                      <Table>
                        <TableHeader className="bg-slate-50/90">
                          <TableRow className="hover:bg-slate-50/90">
                            <TableHead className="text-slate-500">
                              {t("workspace.resources.table.name")}
                            </TableHead>
                            <TableHead className="w-[120px] text-slate-500">
                              {t("workspace.resources.table.type")}
                            </TableHead>
                            <TableHead className="w-[120px] text-slate-500">
                              {t("workspace.resources.table.source")}
                            </TableHead>
                            <TableHead className="w-[220px] text-slate-500">
                              {t("workspace.resources.table.updatedAt")}
                            </TableHead>
                            <TableHead className="w-[80px] text-right text-slate-500">
                              {t("workspace.resources.table.actions")}
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {pagedDisplayItems.map((item) => {
                            const Icon = getKindIcon(item);
                            return (
                              <TableRow
                                key={item.id}
                                data-testid={
                                  item.id === focusContentId
                                    ? "resources-focused-row"
                                    : undefined
                                }
                                className={cn(
                                  "hover:bg-slate-50/70",
                                  item.id === focusContentId &&
                                    "bg-emerald-50/70 ring-1 ring-inset ring-emerald-200/80",
                                )}
                              >
                                <TableCell>
                                  <button
                                    type="button"
                                    className="flex max-w-[580px] items-center gap-3 text-left text-slate-800 transition hover:text-slate-950"
                                    onClick={() => {
                                      if (item.kind === "folder") {
                                        setCurrentFolderId(item.id);
                                        return;
                                      }
                                      void handleOpenResource(item);
                                    }}
                                  >
                                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-slate-200 bg-slate-50 text-slate-600">
                                      <Icon className="h-4 w-4" />
                                    </span>
                                    <span className="min-w-0">
                                      <span className="block truncate font-medium">
                                        {item.name}
                                      </span>
                                      <span className="mt-0.5 block text-xs text-slate-500">
                                        {item.kind === "folder"
                                          ? t(
                                              "workspace.resources.item.folderDescription",
                                            )
                                          : item.description?.trim() ||
                                            t(
                                              "workspace.resources.item.fileDescription",
                                            )}
                                      </span>
                                    </span>
                                  </button>
                                </TableCell>
                                <TableCell>
                                  <Badge
                                    variant={
                                      item.kind === "folder"
                                        ? "default"
                                        : "outline"
                                    }
                                    className={cn(
                                      "rounded-full",
                                      item.kind === "folder" &&
                                        "border border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-50",
                                    )}
                                  >
                                    {getKindLabel(item.kind)}
                                  </Badge>
                                </TableCell>
                                <TableCell>
                                  <Badge
                                    variant="secondary"
                                    className="rounded-full bg-slate-100 text-slate-700"
                                  >
                                    {getSourceLabel(item.sourceType)}
                                  </Badge>
                                </TableCell>
                                <TableCell className="text-slate-500">
                                  {formatTime(item.updatedAt)}
                                </TableCell>
                                <TableCell className="text-right">
                                  <DropdownMenu>
                                    <DropdownMenuTrigger asChild>
                                      <Button
                                        size="icon"
                                        variant="ghost"
                                        className="rounded-xl text-slate-500 hover:bg-slate-100 hover:text-slate-900"
                                      >
                                        <MoreHorizontal className="h-4 w-4" />
                                      </Button>
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent align="end">
                                      <DropdownMenuItem
                                        onClick={() => {
                                          void handleOpenResource(item);
                                        }}
                                      >
                                        {item.kind === "folder"
                                          ? t(
                                              "workspace.resources.action.openFolder",
                                            )
                                          : t("workspace.resources.action.open")}
                                      </DropdownMenuItem>
                                      <DropdownMenuItem
                                        onClick={() => {
                                          void handleRename(item);
                                        }}
                                      >
                                        <Pencil className="mr-2 h-4 w-4" />
                                        {t("workspace.resources.action.rename")}
                                      </DropdownMenuItem>
                                      {item.sourceType === "content" &&
                                        item.parentId && (
                                          <DropdownMenuItem
                                            onClick={() => {
                                              void moveToRoot(item.id);
                                            }}
                                          >
                                            <ArrowUp className="mr-2 h-4 w-4" />
                                            {t(
                                              "workspace.resources.action.moveRoot",
                                            )}
                                          </DropdownMenuItem>
                                        )}
                                      <DropdownMenuItem
                                        className="text-destructive"
                                        onClick={() => {
                                          void handleDelete(item);
                                        }}
                                      >
                                        <Trash2 className="mr-2 h-4 w-4" />
                                        {t("workspace.resources.action.delete")}
                                      </DropdownMenuItem>
                                    </DropdownMenuContent>
                                  </DropdownMenu>
                                </TableCell>
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>

                      <div className="flex flex-col gap-3 border-t border-slate-200/80 bg-slate-50/70 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
                        <div className="flex flex-wrap items-center gap-2 text-sm text-slate-500">
                          <span>
                            {t("workspace.resources.pagination.range", {
                              end: pageRangeEnd,
                              start: pageRangeStart,
                              total: displayItems.length,
                            })}
                          </span>
                          <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-600">
                            {t("workspace.resources.pagination.pageSize", {
                              count: RESOURCE_PAGE_SIZE,
                            })}
                          </span>
                        </div>

                        <div className="flex items-center gap-2">
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            className="rounded-xl border-slate-200 bg-white text-slate-700 hover:bg-slate-100"
                            disabled={safeCurrentPage === 1}
                            onClick={() =>
                              setCurrentPage((previous) =>
                                Math.max(1, previous - 1),
                              )
                            }
                          >
                            {t("workspace.resources.pagination.previous")}
                          </Button>
                          <span className="min-w-[88px] text-center text-sm text-slate-600">
                            {t("workspace.resources.pagination.current", {
                              current: safeCurrentPage,
                              total: totalPages,
                            })}
                          </span>
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            className="rounded-xl border-slate-200 bg-white text-slate-700 hover:bg-slate-100"
                            disabled={safeCurrentPage === totalPages}
                            onClick={() =>
                              setCurrentPage((previous) =>
                                Math.min(totalPages, previous + 1),
                              )
                            }
                          >
                            {t("workspace.resources.pagination.next")}
                          </Button>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
