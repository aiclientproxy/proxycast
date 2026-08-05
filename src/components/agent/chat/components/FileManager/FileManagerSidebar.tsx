import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  AlertTriangle,
  AppWindow,
  ChevronLeft,
  Copy,
  Download,
  ExternalLink,
  Eye,
  EyeOff,
  FileText,
  Folder,
  Home,
  List,
  Monitor,
  Package,
  Pin,
  PlusCircle,
  RefreshCw,
  X,
  type LucideIcon,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import {
  getFileIconDataUrl,
  getFileManagerLocations,
  listDirectory,
  type FileEntry,
  type FileManagerLocation,
} from "@/lib/api/fileBrowser";
import { getKnowledgeUnsupportedSourceMessage } from "@/features/knowledge/import/knowledgeSourceSupport";
import {
  openPathWithDefaultApp,
  revealPathInFinder,
} from "@/lib/api/fileSystem";
import { cn } from "@/lib/utils";
import type { MessagePathReference } from "../../types";
import {
  clearRememberedPathReferencesForDrag,
  createPathReference,
  PATH_REFERENCE_DRAG_MIME,
  rememberPathReferencesForDrag,
  serializePathReferencesForDrag,
} from "../../utils/pathReferences";
import {
  compareFileManagerEntries,
  formatEntryModifiedTime,
  formatFileSize,
} from "./fileManagerDisplay";
import {
  asPinnedLocation,
  buildContextMenuActionDescriptors,
  isApplicationEntry,
  isSkillPackageEntry,
  resolveContextMenuPosition,
  type FileManagerActionLabels,
  type FileManagerContextMenuAction,
} from "./fileManagerSidebarViewModel";

const PINNED_LOCATIONS_STORAGE_KEY = "lime.file-manager.pinned-locations";
const MAX_ICON_PREFETCH_ENTRIES = 72;
const ICON_PREFETCH_CONCURRENCY = 2;

type ViewMode = "list" | "grid";

interface FileManagerSidebarProps {
  onClose: () => void;
  onAddPathReferences: (references: MessagePathReference[]) => void;
  onImportAsKnowledge?: (reference: MessagePathReference) => void;
  onOpenFileInWorkspace?: (entry: FileEntry) => void;
  onInstallSkillPackage?: (entry: FileEntry) => void;
  initialDirectory?: string | null;
}

interface ContextMenuState {
  x: number;
  y: number;
  entry: FileEntry;
}

const CONTEXT_MENU_ICON_BY_ACTION: Record<
  FileManagerContextMenuAction,
  LucideIcon
> = {
  open: ExternalLink,
  reveal: Folder,
  add: PlusCircle,
  "preview-workspace": AppWindow,
  "install-skill-package": Package,
  "import-knowledge": FileText,
  "copy-path": Copy,
  "copy-name": FileText,
  pin: Pin,
  refresh: RefreshCw,
};

function loadPinnedLocations(): FileManagerLocation[] {
  if (typeof window === "undefined") {
    return [];
  }
  try {
    const parsed = JSON.parse(
      window.localStorage.getItem(PINNED_LOCATIONS_STORAGE_KEY) || "[]",
    ) as unknown;
    return Array.isArray(parsed)
      ? parsed
          .map(asPinnedLocation)
          .filter((item): item is FileManagerLocation => Boolean(item))
      : [];
  } catch {
    return [];
  }
}

function savePinnedLocations(locations: FileManagerLocation[]): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(
    PINNED_LOCATIONS_STORAGE_KEY,
    JSON.stringify(locations),
  );
}

function getLocationIcon(kind: string): LucideIcon {
  switch (kind) {
    case "home":
      return Home;
    case "desktop":
      return Monitor;
    case "downloads":
      return Download;
    case "applications":
      return AppWindow;
    case "documents":
      return FileText;
    default:
      return Folder;
  }
}

function EntryIcon({
  icon: Icon,
  className,
}: {
  icon: LucideIcon;
  className: string;
}) {
  return <Icon className={className} aria-hidden strokeWidth={2.2} />;
}

function createReferenceFromEntry(
  entry: FileEntry,
): MessagePathReference | null {
  return createPathReference({
    path: entry.path,
    name: entry.name,
    isDir: entry.isDir,
    size: entry.size,
    mimeType: entry.mimeType,
    source: "file_manager",
  });
}

async function copyText(
  value: string,
  successMessage: string,
  errorMessage: string,
): Promise<void> {
  try {
    await navigator.clipboard.writeText(value);
    toast.success(successMessage);
  } catch {
    toast.error(errorMessage);
  }
}

export const FileManagerSidebar: React.FC<FileManagerSidebarProps> = ({
  onClose,
  onAddPathReferences,
  onImportAsKnowledge,
  onOpenFileInWorkspace,
  onInstallSkillPackage,
  initialDirectory,
}) => {
  const { i18n, t } = useTranslation("agent");
  const locale = i18n.language || "zh-CN";
  const normalizedInitialDirectory = initialDirectory?.trim() ?? "";
  const currentProjectLabel = t("agentChat.fileManager.currentProject");
  const [locations, setLocations] = useState<FileManagerLocation[]>([]);
  const [pinnedLocations, setPinnedLocations] = useState<FileManagerLocation[]>(
    () => loadPinnedLocations(),
  );
  const initialPinnedLocationsRef = useRef(pinnedLocations);
  const [activePath, setActivePath] = useState<string>("");
  const [activeLocationKind, setActiveLocationKind] = useState<string>("");
  const [parentPath, setParentPath] = useState<string | null>(null);
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("list");
  const [showHiddenDirectories, setShowHiddenDirectories] = useState(false);
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const iconDataUrlCacheRef = useRef<Map<string, string>>(new Map());
  const entriesRef = useRef<FileEntry[]>([]);
  const actionLabels = useMemo<FileManagerActionLabels>(
    () => ({
      open: t("agentChat.fileManager.action.open"),
      reveal: t("agentChat.fileManager.action.reveal"),
      addToChat: t("agentChat.fileManager.action.addToChat"),
      preview: t("agentChat.fileManager.action.preview"),
      importKnowledge: t("agentChat.fileManager.action.importKnowledge"),
      importKnowledgeTitle: t(
        "agentChat.fileManager.action.importKnowledgeTitle",
      ),
      copyPath: t("agentChat.fileManager.action.copyPath"),
      copyName: t("agentChat.fileManager.action.copyName"),
      pin: t("agentChat.fileManager.action.pin"),
      refresh: t("agentChat.fileManager.refresh"),
      installPackage: t("agentChat.fileManager.action.installPackage"),
      installPackageTitle: t(
        "agentChat.fileManager.action.installPackageTitle",
      ),
    }),
    [t],
  );

  useEffect(() => {
    entriesRef.current = entries;
  }, [entries]);

  const entryPathSignature = useMemo(
    () => entries.map((entry) => entry.path).join("\u0000"),
    [entries],
  );

  useEffect(() => {
    let cancelled = false;
    void getFileManagerLocations()
      .then((result) => {
        if (cancelled) {
          return;
        }
        setLocations(result);
        const first = normalizedInitialDirectory
          ? {
              id: "project-root",
              label: currentProjectLabel,
              path: normalizedInitialDirectory,
              kind: "project",
            }
          : result[0] || initialPinnedLocationsRef.current[0];
        if (first) {
          setActivePath(first.path);
          setActiveLocationKind(first.kind);
          setViewMode(first.kind === "applications" ? "grid" : "list");
        }
      })
      .catch((loadError) => {
        if (cancelled) {
          return;
        }
        setError(
          loadError instanceof Error ? loadError.message : String(loadError),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [currentProjectLabel, normalizedInitialDirectory]);

  const loadActiveDirectory = useCallback(async () => {
    if (!activePath.trim()) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const listing = await listDirectory(activePath);
      setEntries(listing.entries || []);
      setParentPath(listing.parentPath || null);
      if (listing.error) {
        setError(listing.error);
      }
    } catch (loadError) {
      setEntries([]);
      setParentPath(null);
      setError(
        loadError instanceof Error ? loadError.message : String(loadError),
      );
    } finally {
      setLoading(false);
    }
  }, [activePath]);

  useEffect(() => {
    void loadActiveDirectory();
  }, [loadActiveDirectory]);

  useEffect(() => {
    const currentEntries = entriesRef.current;
    if (currentEntries.length === 0) {
      return;
    }

    let cancelled = false;
    const iconCache = iconDataUrlCacheRef.current;
    const pendingEntries = currentEntries
      .filter((entry) => !entry.iconDataUrl)
      .slice(0, MAX_ICON_PREFETCH_ENTRIES);
    const cachedUpdates = new Map<string, string>();
    const requestEntries: FileEntry[] = [];

    for (const entry of pendingEntries) {
      if (!iconCache.has(entry.path)) {
        requestEntries.push(entry);
        continue;
      }
      cachedUpdates.set(entry.path, iconCache.get(entry.path)!);
    }

    if (cachedUpdates.size > 0) {
      setEntries((current) =>
        current.map((entry) => {
          const iconDataUrl = cachedUpdates.get(entry.path);
          return iconDataUrl && !entry.iconDataUrl
            ? { ...entry, iconDataUrl }
            : entry;
        }),
      );
    }

    if (requestEntries.length === 0) {
      return;
    }

    const loadIcons = async () => {
      for (
        let offset = 0;
        offset < requestEntries.length && !cancelled;
        offset += ICON_PREFETCH_CONCURRENCY
      ) {
        const batch = requestEntries.slice(
          offset,
          offset + ICON_PREFETCH_CONCURRENCY,
        );
        const resolved = await Promise.all(
          batch.map(async (entry) => {
            try {
              const iconDataUrl = await getFileIconDataUrl(entry.path);
              return { path: entry.path, iconDataUrl: iconDataUrl || null };
            } catch {
              return { path: entry.path, iconDataUrl: null };
            }
          }),
        );

        if (cancelled) {
          return;
        }

        const updates = new Map<string, string>();
        for (const item of resolved) {
          if (item.iconDataUrl) {
            iconCache.set(item.path, item.iconDataUrl);
            updates.set(item.path, item.iconDataUrl);
          }
        }

        if (updates.size > 0) {
          setEntries((current) =>
            current.map((entry) => {
              const iconDataUrl = updates.get(entry.path);
              return iconDataUrl && !entry.iconDataUrl
                ? { ...entry, iconDataUrl }
                : entry;
            }),
          );
        }
      }
    };

    void loadIcons();
    return () => {
      cancelled = true;
    };
  }, [activePath, entryPathSignature]);

  useEffect(() => {
    if (!contextMenu) {
      return;
    }
    const closeMenu = () => setContextMenu(null);
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeMenu();
      }
    };
    window.addEventListener("mousedown", closeMenu);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("mousedown", closeMenu);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [contextMenu]);

  const allLocations = useMemo(() => {
    const byPath = new Map<string, FileManagerLocation>();
    if (normalizedInitialDirectory) {
      byPath.set(normalizedInitialDirectory, {
        id: "project-root",
        label: currentProjectLabel,
        path: normalizedInitialDirectory,
        kind: "project",
      });
    }
    for (const location of locations) {
      byPath.set(location.path, location);
    }
    for (const location of pinnedLocations) {
      byPath.set(location.path, location);
    }
    return Array.from(byPath.values());
  }, [
    currentProjectLabel,
    locations,
    normalizedInitialDirectory,
    pinnedLocations,
  ]);

  const activeTitle = useMemo(() => {
    return (
      allLocations.find((location) => location.path === activePath)?.label ||
      activePath.split(/[\\/]/).filter(Boolean).at(-1) ||
      t("agentChat.fileManager.title")
    );
  }, [activePath, allLocations, t]);

  const visibleEntries = useMemo(
    () =>
      showHiddenDirectories
        ? entries
        : entries.filter(
            (entry) => !entry.isDir || !entry.name.startsWith("."),
          ),
    [entries, showHiddenDirectories],
  );

  const sortedEntries = useMemo(
    () =>
      visibleEntries
        .slice()
        .sort((left, right) => compareFileManagerEntries(left, right, locale)),
    [locale, visibleEntries],
  );

  const handleSelectLocation = useCallback((location: FileManagerLocation) => {
    setActivePath(location.path);
    setActiveLocationKind(location.kind);
    setViewMode(location.kind === "applications" ? "grid" : "list");
  }, []);

  const handleOpenEntry = useCallback(
    (entry: FileEntry) => {
      const isApplication = isApplicationEntry(entry, activeLocationKind);
      if (entry.isDir && !isApplication) {
        setActivePath(entry.path);
        setActiveLocationKind("");
        return;
      }
      void openPathWithDefaultApp(entry.path).catch((openError) => {
        toast.error(
          t("agentChat.fileManager.toast.openFailed", {
            message:
              openError instanceof Error
                ? openError.message
                : String(openError),
          }),
        );
      });
    },
    [activeLocationKind, t],
  );

  const handleAddEntry = useCallback(
    (entry: FileEntry) => {
      const reference = createReferenceFromEntry(entry);
      if (!reference) {
        return;
      }
      onAddPathReferences([reference]);
      toast.success(
        t("agentChat.fileManager.toast.addedToChat", {
          name: reference.name,
        }),
      );
    },
    [onAddPathReferences, t],
  );

  const handleEntryPrimaryAction = useCallback(
    (entry: FileEntry) => {
      const isApplication = isApplicationEntry(entry, activeLocationKind);
      if (entry.isDir || isApplication) {
        handleOpenEntry(entry);
        return;
      }

      if (isSkillPackageEntry(entry) && onInstallSkillPackage) {
        onInstallSkillPackage(entry);
        return;
      }

      handleAddEntry(entry);
    },
    [
      activeLocationKind,
      handleAddEntry,
      handleOpenEntry,
      onInstallSkillPackage,
    ],
  );

  const handleImportEntryAsKnowledge = useCallback(
    (entry: FileEntry) => {
      const unsupportedMessage = getKnowledgeUnsupportedSourceMessage(entry);
      if (unsupportedMessage) {
        toast.info(unsupportedMessage);
        return;
      }
      const reference = createReferenceFromEntry(entry);
      if (!reference) {
        return;
      }
      onImportAsKnowledge?.(reference);
    },
    [onImportAsKnowledge],
  );

  const handleOpenEntryInWorkspace = useCallback(
    (entry: FileEntry) => {
      const isApplication = isApplicationEntry(entry, activeLocationKind);
      if (entry.isDir || isApplication) {
        handleOpenEntry(entry);
        return;
      }

      onOpenFileInWorkspace?.(entry);
    },
    [activeLocationKind, handleOpenEntry, onOpenFileInWorkspace],
  );

  const handlePinEntry = useCallback(
    (entry: FileEntry) => {
      if (!entry.isDir) {
        toast.info(t("agentChat.fileManager.toast.folderOnlyPin"));
        return;
      }
      const nextLocation: FileManagerLocation = {
        id: `pinned:${entry.path}`,
        label: entry.name,
        path: entry.path,
        kind: "pinned",
      };
      setPinnedLocations((current) => {
        const next = [
          ...current.filter((location) => location.path !== nextLocation.path),
          nextLocation,
        ];
        savePinnedLocations(next);
        return next;
      });
      toast.success(
        t("agentChat.fileManager.toast.pinned", { name: entry.name }),
      );
    },
    [t],
  );

  const handleContextAction = useCallback(
    (action: string, entry: FileEntry) => {
      setContextMenu(null);
      switch (action) {
        case "open":
          handleOpenEntry(entry);
          break;
        case "reveal":
          void revealPathInFinder(entry.path).catch((revealError) => {
            toast.error(
              t("agentChat.fileManager.toast.revealFailed", {
                message:
                  revealError instanceof Error
                    ? revealError.message
                    : String(revealError),
              }),
            );
          });
          break;
        case "copy-path":
          void copyText(
            entry.path,
            t("agentChat.fileManager.toast.copiedPath"),
            t("agentChat.fileManager.toast.copyFailed"),
          );
          break;
        case "copy-name":
          void copyText(
            entry.name,
            t("agentChat.fileManager.toast.copiedName"),
            t("agentChat.fileManager.toast.copyFailed"),
          );
          break;
        case "add":
          handleAddEntry(entry);
          break;
        case "preview-workspace":
          handleOpenEntryInWorkspace(entry);
          break;
        case "install-skill-package":
          onInstallSkillPackage?.(entry);
          break;
        case "import-knowledge":
          handleImportEntryAsKnowledge(entry);
          break;
        case "pin":
          handlePinEntry(entry);
          break;
        case "refresh":
          void loadActiveDirectory();
          break;
      }
    },
    [
      handleAddEntry,
      handleImportEntryAsKnowledge,
      handleOpenEntry,
      handleOpenEntryInWorkspace,
      handlePinEntry,
      loadActiveDirectory,
      onInstallSkillPackage,
      t,
    ],
  );

  const handleDragStart = useCallback(
    (event: React.DragEvent, entry: FileEntry) => {
      const reference = createReferenceFromEntry(entry);
      if (!reference) {
        event.preventDefault();
        return;
      }
      rememberPathReferencesForDrag([reference]);
      event.dataTransfer.effectAllowed = "copy";
      event.dataTransfer.setData(
        PATH_REFERENCE_DRAG_MIME,
        serializePathReferencesForDrag([reference]),
      );
      event.dataTransfer.setData("text/plain", reference.path);
    },
    [],
  );

  const handleDragEnd = useCallback(() => {
    clearRememberedPathReferencesForDrag(1000);
  }, []);

  const renderEntry = (entry: FileEntry) => {
    const isApplication = isApplicationEntry(entry, activeLocationKind);
    const isSkillPackage = isSkillPackageEntry(entry);
    const Icon =
      isApplication || isSkillPackage
        ? Package
        : entry.isDir
          ? Folder
          : FileText;
    const iconKind = isApplication
      ? "application"
      : isSkillPackage
        ? "skill-package"
        : entry.isDir
          ? "folder"
          : "file";
    const hasNativeIcon = Boolean(entry.iconDataUrl);
    const canInstallSkillPackage = Boolean(
      onInstallSkillPackage && isSkillPackage,
    );
    const handleEntryKeyDown = (event: React.KeyboardEvent) => {
      if (event.key !== "Enter" && event.key !== " ") {
        return;
      }
      event.preventDefault();
      handleEntryPrimaryAction(entry);
    };

    return (
      <div
        key={entry.path}
        role="button"
        tabIndex={0}
        draggable
        data-testid="file-manager-entry"
        data-entry-kind={
          isApplication ? "application" : entry.isDir ? "directory" : "file"
        }
        data-file-path={entry.path}
        aria-label={entry.name}
        onClick={() => handleEntryPrimaryAction(entry)}
        onKeyDown={handleEntryKeyDown}
        onDragStart={(event) => handleDragStart(event, entry)}
        onDragEnd={handleDragEnd}
        onContextMenu={(event) => {
          event.preventDefault();
          const sidebar = event.currentTarget.closest(
            '[data-testid="file-manager-sidebar"]',
          );
          const position =
            typeof window === "undefined"
              ? { x: event.clientX, y: event.clientY }
              : resolveContextMenuPosition({
                  clientX: event.clientX,
                  clientY: event.clientY,
                  sidebarRect:
                    sidebar instanceof HTMLElement
                      ? sidebar.getBoundingClientRect()
                      : null,
                  viewportWidth: window.innerWidth,
                  viewportHeight: window.innerHeight,
                });
          setContextMenu({ ...position, entry });
        }}
        className={cn(
          "group w-full border border-transparent text-left transition focus:outline-none focus:ring-2 focus:ring-sky-200",
          viewMode === "grid"
            ? "flex min-h-[104px] flex-col items-center justify-center gap-2 rounded-[10px] px-3 py-3 text-center hover:border-slate-200 hover:bg-slate-50"
            : "grid min-h-[34px] grid-cols-[minmax(0,1fr)_94px_66px] items-center gap-3 border-b-slate-100 px-3 py-1.5 hover:bg-sky-50/70",
        )}
        title={
          entry.isDir
            ? t("agentChat.fileManager.entryTitle.openFolder")
            : isApplication
              ? t("agentChat.fileManager.entryTitle.openApplication")
              : isSkillPackage && canInstallSkillPackage
                ? t("agentChat.fileManager.action.installPackageTitle")
                : t("agentChat.fileManager.entryTitle.addToChat")
        }
      >
        <span
          className={cn(
            "flex min-w-0",
            viewMode === "grid"
              ? "w-full flex-col items-center gap-2"
              : "items-center gap-2.5",
          )}
        >
          <span
            data-testid="file-manager-entry-icon"
            data-icon-kind={iconKind}
            data-icon-source={hasNativeIcon ? "native" : "fallback"}
            className={cn(
              "inline-flex shrink-0 items-center justify-center",
              viewMode === "grid"
                ? "h-11 w-11 rounded-[10px] border shadow-sm shadow-slate-950/5"
                : "h-5 w-5",
              hasNativeIcon
                ? "border-transparent bg-transparent text-slate-700 shadow-none"
                : isApplication
                  ? "border-sky-100 bg-sky-50 text-sky-700"
                  : entry.isDir
                    ? "border-amber-100 bg-amber-50 text-amber-700"
                    : "border-slate-200 bg-slate-50 text-slate-600",
            )}
          >
            {entry.iconDataUrl ? (
              <img
                data-testid="file-manager-entry-native-icon"
                src={entry.iconDataUrl}
                alt=""
                draggable={false}
                className={cn(
                  "h-full w-full object-contain",
                  viewMode === "grid" ? "rounded-[10px]" : "rounded-[4px]",
                )}
              />
            ) : (
              <EntryIcon
                icon={Icon}
                className={viewMode === "grid" ? "h-5 w-5" : "h-4 w-4"}
              />
            )}
          </span>
          <span
            data-testid="file-manager-entry-label"
            className={cn("min-w-0", viewMode === "grid" ? "w-full" : "flex-1")}
          >
            <span className="block truncate text-[13px] font-medium text-slate-800">
              {entry.name}
            </span>
            {viewMode === "grid" ? (
              <span className="mt-0.5 block truncate text-[11px] text-slate-500">
                {entry.isDir
                  ? formatEntryModifiedTime(entry.modifiedAt, locale)
                  : [
                      formatEntryModifiedTime(entry.modifiedAt, locale),
                      formatFileSize(entry.size),
                    ]
                      .filter(Boolean)
                      .join(" · ")}
              </span>
            ) : null}
          </span>
        </span>
        {viewMode === "list" ? (
          <>
            <span className="truncate text-[12px] text-slate-500">
              {formatEntryModifiedTime(entry.modifiedAt, locale)}
            </span>
            <span className="truncate text-right text-[12px] text-slate-500">
              {entry.isDir ? "-" : formatFileSize(entry.size, "-")}
            </span>
          </>
        ) : null}
      </div>
    );
  };

  return (
    <aside
      className="flex h-full w-[34vw] min-w-[460px] max-w-[720px] shrink-0 overflow-hidden rounded-[12px] border border-[color:var(--lime-surface-border)] bg-white shadow-sm shadow-slate-950/5"
      data-testid="file-manager-sidebar"
      data-lime-no-window-drag
    >
      <div
        className="flex w-[148px] shrink-0 flex-col gap-1 border-r border-slate-200 bg-slate-50 px-2 py-2"
        data-testid="file-manager-location-rail"
      >
        <div className="px-2 py-1 text-[11px] font-medium text-slate-500">
          {t("agentChat.fileManager.locations")}
        </div>
        {allLocations.map((location) => {
          const Icon = getLocationIcon(location.kind);
          const active = location.path === activePath;
          return (
            <button
              key={`${location.id}:${location.path}`}
              type="button"
              className={cn(
                "flex h-8 w-full min-w-0 items-center gap-2 rounded-[7px] px-2 text-left text-[13px] text-slate-600 transition",
                active
                  ? "bg-sky-100 text-sky-900"
                  : "bg-transparent hover:bg-white hover:text-slate-900",
              )}
              title={location.label}
              aria-label={location.label}
              onClick={() => handleSelectLocation(location)}
            >
              <EntryIcon icon={Icon} className="h-4 w-4 shrink-0" />
              <span className="min-w-0 truncate">{location.label}</span>
            </button>
          );
        })}
      </div>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-11 shrink-0 items-center gap-1.5 border-b border-slate-200 bg-slate-50 px-2">
          <div className="flex items-center gap-1">
            <button
              type="button"
              className="inline-flex h-8 w-8 items-center justify-center rounded-[7px] text-slate-500 transition hover:bg-white hover:text-slate-900 disabled:opacity-40"
              aria-label={t("agentChat.fileManager.goParent")}
              title={t("agentChat.fileManager.goParent")}
              disabled={!parentPath}
              onClick={() => parentPath && setActivePath(parentPath)}
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button
              type="button"
              className="inline-flex h-8 w-8 items-center justify-center rounded-[7px] text-slate-500 transition hover:bg-white hover:text-slate-900"
              aria-label={t("agentChat.fileManager.refresh")}
              title={t("agentChat.fileManager.refresh")}
              onClick={() => void loadActiveDirectory()}
            >
              <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
            </button>
          </div>
          <button
            type="button"
            data-testid="file-manager-toggle-hidden-directories"
            className={cn(
              "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[7px] text-slate-500 transition hover:bg-white hover:text-slate-900",
              showHiddenDirectories && "bg-white text-sky-700",
            )}
            aria-label={t(
              showHiddenDirectories
                ? "agentChat.fileManager.hideHiddenDirectories"
                : "agentChat.fileManager.showHiddenDirectories",
            )}
            aria-pressed={showHiddenDirectories}
            title={t(
              showHiddenDirectories
                ? "agentChat.fileManager.hideHiddenDirectories"
                : "agentChat.fileManager.showHiddenDirectories",
            )}
            onClick={() => setShowHiddenDirectories((current) => !current)}
          >
            {showHiddenDirectories ? (
              <EyeOff className="h-4 w-4" />
            ) : (
              <Eye className="h-4 w-4" />
            )}
          </button>
          <div
            className="flex min-w-0 flex-1 items-center gap-2 rounded-[7px] border border-slate-200 bg-white px-2.5 py-1.5"
            title={activePath ? t("agentChat.fileManager.currentFolder") : ""}
          >
            <Folder className="h-4 w-4 shrink-0 text-slate-400" />
            <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-slate-800">
              {activePath
                ? activeTitle
                : t("agentChat.fileManager.preparingLocation")}
            </span>
            <span className="shrink-0 text-[11px] text-slate-500">
              {activePath ? t("agentChat.fileManager.localLocation") : ""}
            </span>
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              className="inline-flex h-8 w-8 items-center justify-center rounded-[7px] text-slate-500 transition hover:bg-white hover:text-slate-900"
              aria-label={t("agentChat.fileManager.toggleView")}
              title={
                viewMode === "list"
                  ? t("agentChat.fileManager.gridView")
                  : t("agentChat.fileManager.listView")
              }
              onClick={() =>
                setViewMode((current) => (current === "list" ? "grid" : "list"))
              }
            >
              <List className="h-4 w-4" />
            </button>
            <button
              type="button"
              className="inline-flex h-8 w-8 items-center justify-center rounded-[7px] text-slate-500 transition hover:bg-rose-50 hover:text-rose-600"
              aria-label={t("agentChat.fileManager.close")}
              title={t("agentChat.fileManager.close")}
              onClick={onClose}
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {error ? (
          <div className="m-3 flex items-start gap-2 rounded-[16px] border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span className="min-w-0 break-words">{error}</span>
          </div>
        ) : null}

        {viewMode === "list" ? (
          <div
            data-testid="file-manager-list-header"
            className="grid h-8 shrink-0 grid-cols-[minmax(0,1fr)_94px_66px] items-center gap-3 border-b border-slate-200 bg-white px-3 text-[11px] font-medium text-slate-500"
          >
            <span>{t("agentChat.fileManager.column.name")}</span>
            <span>{t("agentChat.fileManager.column.modified")}</span>
            <span className="text-right">
              {t("agentChat.fileManager.column.size")}
            </span>
          </div>
        ) : null}

        <div className="min-h-0 flex-1 overflow-y-auto bg-white">
          {!loading && sortedEntries.length === 0 ? (
            <div className="m-3 flex h-full min-h-[220px] flex-col items-center justify-center rounded-[10px] border border-dashed border-slate-200 bg-slate-50 px-4 text-center text-sm text-slate-500">
              <Folder className="mb-2 h-8 w-8 text-slate-300" />
              {t("agentChat.fileManager.emptyDirectory")}
            </div>
          ) : null}

          <div
            className={cn(
              viewMode === "grid" ? "grid grid-cols-2 gap-2 p-3" : "",
            )}
          >
            {sortedEntries.map(renderEntry)}
          </div>
        </div>
      </div>

      {contextMenu ? (
        <div
          data-testid="file-manager-context-menu"
          className="fixed z-[120] w-52 max-w-[calc(100vw-16px)] overflow-hidden rounded-[16px] border border-slate-200 bg-white p-1.5 text-sm text-slate-700 shadow-xl shadow-slate-950/12"
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onMouseDown={(event) => event.stopPropagation()}
        >
          {buildContextMenuActionDescriptors({
            entry: contextMenu.entry,
            knowledgeImportEnabled: Boolean(onImportAsKnowledge),
            workspacePreviewEnabled: Boolean(onOpenFileInWorkspace),
            skillPackageInstallEnabled: Boolean(onInstallSkillPackage),
            labels: actionLabels,
            knowledgeUnsupportedMessage: getKnowledgeUnsupportedSourceMessage(
              contextMenu.entry,
            ),
          }).map(({ action, label, disabled, title }) => {
            const MenuIcon = CONTEXT_MENU_ICON_BY_ACTION[action];
            return (
              <button
                key={action}
                type="button"
                disabled={disabled}
                title={title}
                className={cn(
                  "flex w-full items-center gap-2 rounded-[12px] px-3 py-2 text-left transition",
                  disabled
                    ? "cursor-not-allowed text-slate-400"
                    : "hover:bg-amber-50 hover:text-amber-800",
                )}
                onClick={() =>
                  !disabled && handleContextAction(action, contextMenu.entry)
                }
              >
                <MenuIcon className="h-4 w-4 shrink-0" />
                <span className="min-w-0 truncate">{label}</span>
              </button>
            );
          })}
        </div>
      ) : null}
    </aside>
  );
};
