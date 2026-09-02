import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FileQuestion } from "lucide-react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import {
  openPathWithDefaultApp,
  revealPathInFinder,
} from "@/lib/api/fileSystem";
import { openExternalUrlWithSystemBrowser } from "@/lib/api/externalUrl";
import { readFilePreview } from "@/lib/api/fileBrowser";
import { hasDesktopHostEventCapability } from "@/lib/desktop-runtime";
import { DATA_RESOURCE_PREVIEW_MAX_SIZE } from "./DataResourceRenderer";
import { useImageResourceViewControls } from "./imageResourceViewControls";
import { getResourcePreviewTarget } from "./resourceFormatCatalog";
import { ResourceManagerInspector } from "./ResourceManagerInspector";
import { ResourceManagerPreviewPane } from "./ResourceManagerPreviewPane";
import {
  buildInspectorRows,
  buildMetadataChips,
  buildSourceContextRows,
  getChatLocationActionLabel,
  getEffectiveSourceContext,
  getItemTitle,
  getProjectResourceActionLabel,
  hasChatLocationContext,
  hasProjectResourceContext,
  itemMatchesResourceSearch,
  normalizeSearchValue,
  RESOURCE_KIND_FILTERS,
  type ResourceManagerKindFilter,
} from "./resourceManagerPresentation";
import { ResourceManagerSidebar } from "./ResourceManagerSidebar";
import { ResourceManagerToolbar } from "./ResourceManagerToolbar";
import { TEXT_RESOURCE_PREVIEW_MAX_SIZE } from "./TextResourceRenderer";
import { RESOURCE_MANAGER_SESSION_EVENT } from "./openResourceManager";
import {
  writeResourceManagerNavigationIntent,
  type ResourceManagerNavigationIntentAction,
} from "./resourceManagerIntents";
import {
  clampResourceManagerIndex,
  readResourceManagerSession,
} from "./resourceManagerSession";
import type { ResourceManagerItem, ResourceManagerSession } from "./types";

function readSessionIdFromLocation(): string | null {
  if (typeof window === "undefined") {
    return null;
  }

  return new URLSearchParams(window.location.search).get("session");
}

function closeCurrentResourceManagerWindow(): void {
  window.close();
}

function sanitizeFileName(value: string): string {
  return (
    value
      .trim()
      .replace(/[\\/:*?"<>|]/g, "-")
      .replace(/\s+/g, " ")
      .slice(0, 80) || "lime-resource"
  );
}

function getResourceExtension(item: ResourceManagerItem): string {
  const source = item.filePath || item.src || item.title || "";
  const cleanSource = source.split(/[?#]/)[0] ?? "";
  const match = cleanSource.match(/\.([a-zA-Z0-9]{2,6})$/);
  if (match?.[1]) return match[1].toLowerCase();
  if (item.kind === "video") return "mp4";
  if (item.kind === "audio") return "mp3";
  if (item.kind === "pdf") return "pdf";
  if (item.kind === "markdown") return "md";
  if (item.kind === "text") return "txt";
  if (item.kind === "data") return "json";
  if (item.kind === "archive") return "zip";
  return "bin";
}

function downloadResourceItem(item: ResourceManagerItem): void {
  const title = sanitizeFileName(getItemTitle(item));
  const extension = getResourceExtension(item);
  const link = document.createElement("a");

  if (
    item.content &&
    (item.kind === "text" || item.kind === "markdown" || item.kind === "data")
  ) {
    const blob = new Blob([item.content], {
      type:
        item.kind === "markdown"
          ? "text/markdown;charset=utf-8"
          : item.kind === "data"
            ? item.mimeType || "application/json;charset=utf-8"
            : "text/plain;charset=utf-8",
    });
    const objectUrl = URL.createObjectURL(blob);
    link.href = objectUrl;
    link.download = `${title}.${extension}`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
    return;
  }

  if (!item.src) {
    toast.info("当前资源没有可下载地址");
    return;
  }

  link.href = item.src;
  link.download = `${title}.${extension}`;
  link.rel = "noopener noreferrer";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

async function copyTextToClipboard(value: string): Promise<void> {
  if (!navigator.clipboard?.writeText) {
    throw new Error("clipboard_write_text_unavailable");
  }
  await navigator.clipboard.writeText(value);
}

async function copyImageToClipboard(item: ResourceManagerItem): Promise<void> {
  const clipboard = navigator.clipboard as Clipboard & {
    write?: (items: ClipboardItem[]) => Promise<void>;
  };
  const ClipboardItemCtor = (
    window as Window & { ClipboardItem?: typeof ClipboardItem }
  ).ClipboardItem;

  if (!item.src) {
    throw new Error("image_src_missing");
  }

  if (!clipboard.write || !ClipboardItemCtor) {
    await copyTextToClipboard(item.filePath || item.src);
    toast.info("当前环境不支持复制图片，已复制图片地址");
    return;
  }

  const response = await fetch(item.src);
  const blob = await response.blob();
  const mimeType = blob.type || "image/png";
  await clipboard.write([new ClipboardItemCtor({ [mimeType]: blob })]);
}

function shouldIgnoreViewerShortcut(event: KeyboardEvent): boolean {
  const target = event.target as HTMLElement | null;
  return Boolean(
    target?.tagName === "INPUT" ||
    target?.tagName === "TEXTAREA" ||
    target?.isContentEditable,
  );
}

function getReadablePreviewLimit(item: ResourceManagerItem): number {
  return item.kind === "data"
    ? DATA_RESOURCE_PREVIEW_MAX_SIZE
    : TEXT_RESOURCE_PREVIEW_MAX_SIZE;
}

const EMPTY_RESOURCE_MANAGER_ITEMS: ResourceManagerItem[] = [];

export function ResourceManagerPage() {
  const { t } = useTranslation("workspace");
  const [session, setSession] = useState<ResourceManagerSession | null>(() =>
    readResourceManagerSession(readSessionIdFromLocation()),
  );
  const [activeIndex, setActiveIndex] = useState(
    () => session?.initialIndex ?? 0,
  );
  const [isFullscreen, setIsFullscreen] = useState(() =>
    Boolean(document.fullscreenElement),
  );
  const [showResourceList, setShowResourceList] = useState(true);
  const [showInspector, setShowInspector] = useState(false);
  const [resourceSearchQuery, setResourceSearchQuery] = useState("");
  const [previewSearchQuery, setPreviewSearchQuery] = useState("");
  const [previewSearchMatchCount, setPreviewSearchMatchCount] = useState(0);
  const [previewSearchActiveIndex, setPreviewSearchActiveIndex] = useState(0);
  const [markdownViewMode, setMarkdownViewMode] = useState<
    "preview" | "source"
  >("preview");
  const [dataViewMode, setDataViewMode] = useState<"formatted" | "raw">(
    "formatted",
  );
  const [resourceKindFilter, setResourceKindFilter] =
    useState<ResourceManagerKindFilter>("all");
  const [menuOpen, setMenuOpen] = useState(false);
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const previewSearchInputRef = useRef<HTMLInputElement | null>(null);

  const items = session?.items ?? EMPTY_RESOURCE_MANAGER_ITEMS;
  const activeItem = items[activeIndex] ?? null;
  const activeSourceContext = getEffectiveSourceContext(activeItem, session);
  const hasMultipleItems = items.length > 1;
  const canLocateSource = hasChatLocationContext(activeSourceContext);
  const canOpenProjectResource = hasProjectResourceContext(activeSourceContext);
  const canOpenOrigin = Boolean(activeSourceContext?.originUrl);
  const canContinueImageTask = Boolean(
    activeItem?.kind === "image" && activeSourceContext?.taskId,
  );
  const canCopyLinkFromToolbar = Boolean(
    activeItem &&
    (activeItem.kind !== "image" ||
      getResourcePreviewTarget(activeItem) !== "webview") &&
    activeItem.kind !== "text" &&
    activeItem.kind !== "markdown" &&
    (activeItem.kind !== "data" ||
      getResourcePreviewTarget(activeItem) !== "data"),
  );
  const canSearchActivePreview = Boolean(
    activeItem &&
    (activeItem.kind === "text" ||
      activeItem.kind === "markdown" ||
      (activeItem.kind === "data" &&
        getResourcePreviewTarget(activeItem) === "data")),
  );
  const imageControls = useImageResourceViewControls({
    itemKey:
      activeItem?.kind === "image"
        ? `${activeItem.id}:${activeItem.src ?? ""}`
        : null,
    enabled: activeItem?.kind === "image",
  });
  const metadataChips = useMemo(
    () => (activeItem ? buildMetadataChips(activeItem) : []),
    [activeItem],
  );
  const normalizedResourceSearchQuery = useMemo(
    () => normalizeSearchValue(resourceSearchQuery),
    [resourceSearchQuery],
  );
  const visibleResourceEntries = useMemo(
    () =>
      items
        .map((item, index) => ({ item, index }))
        .filter(({ item }) =>
          itemMatchesResourceSearch({
            item,
            query: normalizedResourceSearchQuery,
            kindFilter: resourceKindFilter,
          }),
        ),
    [items, normalizedResourceSearchQuery, resourceKindFilter],
  );
  const availableKindFilters = useMemo(
    () =>
      RESOURCE_KIND_FILTERS.filter(
        (kind) => kind === "all" || items.some((item) => item.kind === kind),
      ),
    [items],
  );
  const inspectorRows = useMemo(
    () => (activeItem ? buildInspectorRows(activeItem) : []),
    [activeItem],
  );
  const sourceContextRows = useMemo(
    () => buildSourceContextRows(activeSourceContext),
    [activeSourceContext],
  );

  const selectIndex = useCallback(
    (index: number) => {
      setActiveIndex(clampResourceManagerIndex(index, items.length));
    },
    [items.length],
  );

  const showPrevious = useCallback(() => {
    if (!items.length) return;
    selectIndex((activeIndex - 1 + items.length) % items.length);
  }, [activeIndex, items.length, selectIndex]);

  const showNext = useCallback(() => {
    if (!items.length) return;
    selectIndex((activeIndex + 1) % items.length);
  }, [activeIndex, items.length, selectIndex]);

  const handlePreviewSearchQueryChange = useCallback((query: string) => {
    setPreviewSearchQuery(query);
    setPreviewSearchActiveIndex(0);
  }, []);

  const handlePreviewSearchMatchCountChange = useCallback(
    (matchCount: number) => {
      setPreviewSearchMatchCount(matchCount);
      setPreviewSearchActiveIndex((current) =>
        matchCount <= 0 ? 0 : Math.min(current, matchCount - 1),
      );
    },
    [],
  );

  const showPreviousPreviewSearchMatch = useCallback(() => {
    setPreviewSearchActiveIndex((current) =>
      previewSearchMatchCount <= 0
        ? 0
        : (current - 1 + previewSearchMatchCount) % previewSearchMatchCount,
    );
  }, [previewSearchMatchCount]);

  const showNextPreviewSearchMatch = useCallback(() => {
    setPreviewSearchActiveIndex((current) =>
      previewSearchMatchCount <= 0
        ? 0
        : (current + 1) % previewSearchMatchCount,
    );
  }, [previewSearchMatchCount]);

  const handleClearResourceFilters = useCallback(() => {
    setResourceSearchQuery("");
    setResourceKindFilter("all");
  }, []);

  useEffect(() => {
    setPreviewSearchQuery("");
    setPreviewSearchMatchCount(0);
    setPreviewSearchActiveIndex(0);
    setMarkdownViewMode("preview");
    setDataViewMode("formatted");
  }, [activeItem?.id]);

  useEffect(() => {
    if (!items.length || visibleResourceEntries.length === 0) {
      return;
    }

    if (visibleResourceEntries.some((entry) => entry.index === activeIndex)) {
      return;
    }

    selectIndex(visibleResourceEntries[0]?.index ?? 0);
  }, [activeIndex, items.length, selectIndex, visibleResourceEntries]);

  useEffect(() => {
    if (!hasDesktopHostEventCapability()) {
      return;
    }

    let unlisten: (() => void) | null = null;
    void import("@/lib/desktop-host/event")
      .then(({ listen }) =>
        listen<{ sessionId?: string }>(
          RESOURCE_MANAGER_SESSION_EVENT,
          (event) => {
            const nextSession = readResourceManagerSession(
              event.payload?.sessionId,
            );
            if (!nextSession) return;
            setSession(nextSession);
            setActiveIndex(nextSession.initialIndex);
          },
        ),
      )
      .then((cleanup) => {
        unlisten = cleanup;
      })
      .catch((error) => {
        console.warn("[资源管理器] 监听会话切换失败:", error);
      });

    return () => {
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () =>
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  const handleOpenWithDefaultApp = useCallback(async () => {
    const filePath = activeItem?.filePath?.trim();
    if (!filePath) {
      toast.info("仅本地文件支持用系统应用打开");
      return;
    }

    try {
      await openPathWithDefaultApp(filePath);
    } catch (error) {
      console.warn("[资源管理器] 用系统应用打开失败:", error);
      toast.error("系统应用打开失败，请稍后重试");
    }
  }, [activeItem?.filePath]);

  const handleRevealPath = useCallback(async () => {
    const filePath = activeItem?.filePath?.trim();
    if (!filePath) {
      toast.info("仅本地文件支持定位");
      return;
    }

    try {
      await revealPathInFinder(filePath);
    } catch (error) {
      console.warn("[资源管理器] 定位文件失败:", error);
      toast.error("定位文件失败，请稍后重试");
    }
  }, [activeItem?.filePath]);

  const handleCopyLink = useCallback(async () => {
    if (!activeItem) return;
    const target =
      activeItem.filePath || activeItem.src || getItemTitle(activeItem);
    try {
      await copyTextToClipboard(target);
      toast.success(activeItem.filePath ? "已复制资源路径" : "已复制资源地址");
    } catch (error) {
      console.warn("[资源管理器] 复制资源地址失败:", error);
      toast.error("复制失败，请检查系统剪贴板权限");
    }
  }, [activeItem]);

  const handleCopyImage = useCallback(async () => {
    if (!activeItem || activeItem.kind !== "image") {
      toast.info("当前资源暂不支持复制为图片");
      return;
    }

    try {
      await copyImageToClipboard(activeItem);
      toast.success("已复制图片");
    } catch (error) {
      console.warn("[资源管理器] 复制图片失败:", error);
      try {
        await copyTextToClipboard(activeItem.filePath || activeItem.src || "");
        toast.info("复制图片失败，已复制图片地址");
      } catch {
        toast.error("复制图片失败，已保留原图地址可复制");
      }
    }
  }, [activeItem]);

  const handleCopyContent = useCallback(async () => {
    if (
      !activeItem ||
      (activeItem.kind !== "text" &&
        activeItem.kind !== "markdown" &&
        activeItem.kind !== "data")
    ) {
      toast.info("当前资源没有可复制的内容");
      return;
    }

    try {
      let content = activeItem.content ?? null;
      if (content === null && activeItem.filePath) {
        const preview = await readFilePreview(
          activeItem.filePath,
          getReadablePreviewLimit(activeItem),
        );
        if (preview.error) {
          toast.error(`读取内容失败：${preview.error}`);
          return;
        }
        if (preview.isBinary) {
          toast.info("该文件被识别为二进制内容，不能复制为文本");
          return;
        }
        content = preview.content ?? "";
      }

      if (!content) {
        toast.info("当前没有可复制的内容");
        return;
      }

      await copyTextToClipboard(content);
      toast.success("已复制内容");
    } catch (error) {
      console.warn("[资源管理器] 复制文本内容失败:", error);
      toast.error("复制失败，请检查系统剪贴板权限");
    }
  }, [activeItem]);

  const handleDownloadActive = useCallback(() => {
    if (!activeItem) return;
    downloadResourceItem(activeItem);
  }, [activeItem]);

  const handleToggleFullscreen = useCallback(async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
        return;
      }
      if (!document.documentElement.requestFullscreen) {
        toast.info("当前环境不支持全屏查看");
        return;
      }
      await document.documentElement.requestFullscreen();
    } catch (error) {
      console.warn("[资源管理器] 切换全屏失败:", error);
      toast.error("全屏切换失败，请检查系统窗口权限");
    }
  }, []);

  const handlePrint = useCallback(() => {
    window.print();
    setMenuOpen(false);
  }, []);

  const handleOpenOrigin = useCallback(async () => {
    const originUrl = activeSourceContext?.originUrl?.trim();
    if (!originUrl) {
      toast.info("当前资源没有原文链接");
      setMenuOpen(false);
      return;
    }

    try {
      const parsed = new URL(originUrl);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        toast.info("仅支持打开 http/https 原文链接");
        setMenuOpen(false);
        return;
      }

      await openExternalUrlWithSystemBrowser(parsed.toString());
    } catch (error) {
      console.warn("[资源管理器] 打开原文链接失败:", error);
      toast.error("原文链接无法打开");
    } finally {
      setMenuOpen(false);
    }
  }, [activeSourceContext?.originUrl]);

  const handleWriteNavigationIntent = useCallback(
    (action: ResourceManagerNavigationIntentAction, successMessage: string) => {
      if (!activeItem) return;

      const intent = writeResourceManagerNavigationIntent({
        action,
        item: activeItem,
        sourceContext: activeSourceContext,
      });
      if (!intent) {
        toast.info("当前资源缺少可回跳的业务上下文");
        setMenuOpen(false);
        return;
      }

      toast.success(successMessage);
      setMenuOpen(false);
    },
    [activeItem, activeSourceContext],
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (shouldIgnoreViewerShortcut(event)) return;

      const key = event.key.toLowerCase();
      if ((event.metaKey || event.ctrlKey) && key === "c") {
        event.preventDefault();
        if (activeItem?.kind === "image") {
          void handleCopyImage();
        } else if (
          activeItem?.kind === "text" ||
          activeItem?.kind === "markdown" ||
          activeItem?.kind === "data"
        ) {
          void handleCopyContent();
        } else {
          void handleCopyLink();
        }
        return;
      }
      if ((event.metaKey || event.ctrlKey) && key === "f") {
        event.preventDefault();
        const targetInput = canSearchActivePreview
          ? previewSearchInputRef.current
          : searchInputRef.current;
        targetInput?.focus();
        targetInput?.select();
        return;
      }
      if (event.key === "/") {
        event.preventDefault();
        searchInputRef.current?.focus();
        searchInputRef.current?.select();
        return;
      }
      if (key === "i") {
        event.preventDefault();
        setShowInspector((current) => !current);
        return;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        void closeCurrentResourceManagerWindow();
        return;
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        showPrevious();
        return;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        showNext();
        return;
      }
      if (event.key === "Home") {
        event.preventDefault();
        selectIndex(0);
        return;
      }
      if (event.key === "End") {
        event.preventDefault();
        selectIndex(items.length - 1);
        return;
      }
      if (key === "o") {
        event.preventDefault();
        void handleOpenWithDefaultApp();
        return;
      }
      if (key === "l") {
        event.preventDefault();
        void handleRevealPath();
        return;
      }
      if (key === "d") {
        event.preventDefault();
        handleDownloadActive();
        return;
      }
      if (key === "g") {
        event.preventDefault();
        setShowResourceList((current) => !current);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    activeItem,
    canSearchActivePreview,
    handleCopyContent,
    handleCopyImage,
    handleCopyLink,
    handleDownloadActive,
    handleOpenWithDefaultApp,
    handleRevealPath,
    items.length,
    selectIndex,
    showNext,
    showPrevious,
  ]);

  if (!session || !activeItem) {
    return (
      <main className="flex h-screen w-screen items-center justify-center bg-[#f5f6f8] px-6 text-center text-slate-500">
        <div className="max-w-sm rounded-[24px] border border-slate-200 bg-white p-8 shadow-sm shadow-slate-950/5">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl border border-slate-200 bg-slate-50 text-slate-600">
            <FileQuestion className="h-7 w-7" />
          </div>
          <h1 className="mt-5 text-lg font-semibold text-slate-950">
            {t("workspace.resourceManager.empty.title")}
          </h1>
          <p className="mt-2 text-sm leading-6 text-slate-500">
            {t("workspace.resourceManager.empty.description")}
          </p>
        </div>
      </main>
    );
  }

  return (
    <main className="flex h-screen w-screen flex-col overflow-hidden bg-[#f5f6f8] text-slate-900">
      <ResourceManagerToolbar
        activeItem={activeItem}
        activeIndex={activeIndex}
        itemCount={items.length}
        hasMultipleItems={hasMultipleItems}
        showResourceList={showResourceList}
        showInspector={showInspector}
        isFullscreen={isFullscreen}
        menuOpen={menuOpen}
        imageControls={imageControls}
        canCopyLinkFromToolbar={canCopyLinkFromToolbar}
        canSearchActivePreview={canSearchActivePreview}
        canLocateSource={canLocateSource}
        canOpenProjectResource={canOpenProjectResource}
        canOpenOrigin={canOpenOrigin}
        canContinueImageTask={canContinueImageTask}
        chatLocationActionLabel={getChatLocationActionLabel(
          activeSourceContext,
        )}
        projectResourceActionLabel={getProjectResourceActionLabel(
          activeSourceContext,
        )}
        previewSearchQuery={previewSearchQuery}
        previewSearchMatchCount={previewSearchMatchCount}
        previewSearchActiveIndex={previewSearchActiveIndex}
        previewSearchInputRef={previewSearchInputRef}
        markdownViewMode={markdownViewMode}
        dataViewMode={dataViewMode}
        onPrevious={showPrevious}
        onNext={showNext}
        onToggleResourceList={() => setShowResourceList((current) => !current)}
        onCopyImage={handleCopyImage}
        onCopyLink={handleCopyLink}
        onCopyContent={handleCopyContent}
        onDownload={handleDownloadActive}
        onRevealPath={handleRevealPath}
        onOpenWithDefaultApp={handleOpenWithDefaultApp}
        onToggleInspector={() => setShowInspector((current) => !current)}
        onToggleFullscreen={handleToggleFullscreen}
        onToggleMenu={() => setMenuOpen((current) => !current)}
        onClose={closeCurrentResourceManagerWindow}
        onPrint={handlePrint}
        onOpenOrigin={handleOpenOrigin}
        onWriteNavigationIntent={handleWriteNavigationIntent}
        onPreviewSearchQueryChange={handlePreviewSearchQueryChange}
        onPreviousPreviewSearchMatch={showPreviousPreviewSearchMatch}
        onNextPreviewSearchMatch={showNextPreviewSearchMatch}
        onToggleMarkdownViewMode={() =>
          setMarkdownViewMode((current) =>
            current === "preview" ? "source" : "preview",
          )
        }
        onToggleDataViewMode={() =>
          setDataViewMode((current) =>
            current === "formatted" ? "raw" : "formatted",
          )
        }
      />

      <section className="flex min-h-0 flex-1 overflow-hidden bg-[#f5f6f8]">
        {showResourceList ? (
          <ResourceManagerSidebar
            sourceLabel={session.sourceLabel}
            items={items}
            visibleEntries={visibleResourceEntries}
            activeIndex={activeIndex}
            searchQuery={resourceSearchQuery}
            kindFilter={resourceKindFilter}
            availableKindFilters={availableKindFilters}
            searchInputRef={searchInputRef}
            onSearchQueryChange={setResourceSearchQuery}
            onKindFilterChange={setResourceKindFilter}
            onSelectIndex={selectIndex}
            onClearFilters={handleClearResourceFilters}
          />
        ) : null}

        <div className="flex min-w-0 flex-1">
          <ResourceManagerPreviewPane
            item={activeItem}
            imageControls={imageControls}
            metadataChips={metadataChips}
            hasMultipleItems={hasMultipleItems}
            previewSearchQuery={previewSearchQuery}
            previewSearchActiveIndex={previewSearchActiveIndex}
            markdownViewMode={markdownViewMode}
            dataViewMode={dataViewMode}
            onPrevious={showPrevious}
            onNext={showNext}
            onSearchMatchCountChange={handlePreviewSearchMatchCountChange}
          />

          {showInspector ? (
            <ResourceManagerInspector
              item={activeItem}
              inspectorRows={inspectorRows}
              sourceContextRows={sourceContextRows}
            />
          ) : null}
        </div>
      </section>
    </main>
  );
}
