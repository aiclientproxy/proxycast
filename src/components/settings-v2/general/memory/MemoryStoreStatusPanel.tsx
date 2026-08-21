import { useCallback, useEffect, useState } from "react";
import type { TFunction } from "i18next";
import { Database } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  addMemoryStoreNote,
  consolidateMemoryStore,
  getMemoryStoreHealth,
  listMemoryStore,
  listMemoryStoreReviewNotes,
  readMemoryStore,
  rebuildMemoryStoreIndex,
  resolveMemoryStoreReviewNote,
  resetMemory,
  type MemoryStoreHealthResponse,
  type MemoryStoreReviewNote,
} from "@/lib/api/memoryStore";
import { getDefaultProject } from "@/lib/api/project";
import { cn } from "@/lib/utils";
import { RolloutCandidatesPanel } from "./RolloutCandidatesPanel";
import {
  isRolloutCandidateEntry,
  parseRolloutCandidateMarkdown,
  type RolloutCandidateSummary,
} from "./rolloutCandidates";

interface MemoryStoreStatusPanelProps {
  vectorSearchEnabled: boolean;
  memoryStatusDescriptionKey: string;
  showMessage: (message: string) => void;
}

function memoryPanelT(
  t: TFunction<"settings">,
  key: string,
  values: Record<string, string | number | boolean> = {},
): string {
  const translate = t as unknown as (
    key: string,
    values?: Record<string, string | number | boolean>,
  ) => string;
  return String(translate(key, values));
}

function formatStorageSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function healthRootScope(
  t: TFunction<"settings">,
  health: MemoryStoreHealthResponse | null,
): string {
  if (!health) {
    return memoryPanelT(t, "settings.memory.store.statusUnknown");
  }
  return health.rootScope === "workspace"
    ? memoryPanelT(t, "settings.memory.store.scope.workspace")
    : memoryPanelT(t, "settings.memory.store.scope.global");
}

export function MemoryStoreStatusPanel({
  vectorSearchEnabled,
  memoryStatusDescriptionKey,
  showMessage,
}: MemoryStoreStatusPanelProps) {
  const { t } = useTranslation("settings");
  const [health, setHealth] = useState<MemoryStoreHealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [consolidating, setConsolidating] = useState(false);
  const [rebuildingIndex, setRebuildingIndex] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [noteContent, setNoteContent] = useState("");
  const [savingNote, setSavingNote] = useState(false);
  const [reviewNotes, setReviewNotes] = useState<MemoryStoreReviewNote[]>([]);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [resolvingReviewPath, setResolvingReviewPath] = useState<string | null>(
    null,
  );
  const [rolloutCandidates, setRolloutCandidates] = useState<
    RolloutCandidateSummary[]
  >([]);
  const [rolloutLoading, setRolloutLoading] = useState(false);
  const [rolloutLoadFailed, setRolloutLoadFailed] = useState(false);
  const [rolloutWorkspaceName, setRolloutWorkspaceName] = useState<
    string | null
  >(null);
  const [rolloutWorkspaceRoot, setRolloutWorkspaceRoot] = useState<
    string | null
  >(null);
  const [rolloutConsolidating, setRolloutConsolidating] = useState(false);

  const refreshHealth = useCallback(
    async (showSuccess = false) => {
      setHealthLoading(true);
      try {
        const response = await getMemoryStoreHealth({ scope: "global" });
        setHealth(response);
        if (showSuccess) {
          showMessage(
            memoryPanelT(t, "settings.memory.store.message.healthRefreshed"),
          );
        }
      } catch (error) {
        console.error("加载记忆文件状态失败:", error);
        showMessage(
          memoryPanelT(t, "settings.memory.store.message.healthFailed"),
        );
      } finally {
        setHealthLoading(false);
      }
    },
    [showMessage, t],
  );

  const refreshReviewNotes = useCallback(
    async (showSuccess = false) => {
      setReviewLoading(true);
      try {
        const response = await listMemoryStoreReviewNotes({
          scope: "global",
          maxResults: 20,
        });
        setReviewNotes(response.notes ?? []);
        if (showSuccess) {
          showMessage(
            memoryPanelT(t, "settings.memory.store.message.reviewRefreshed"),
          );
        }
      } catch (error) {
        console.error("加载待审阅记忆笔记失败:", error);
        showMessage(
          memoryPanelT(t, "settings.memory.store.message.reviewLoadFailed"),
        );
      } finally {
        setReviewLoading(false);
      }
    },
    [showMessage, t],
  );

  const refreshRolloutCandidates = useCallback(
    async (showSuccess = false) => {
      setRolloutLoading(true);
      setRolloutLoadFailed(false);
      try {
        const workspace = await getDefaultProject();
        const workspaceRoot = workspace?.rootPath?.trim();
        setRolloutWorkspaceName(workspace?.name?.trim() || null);
        setRolloutWorkspaceRoot(workspaceRoot || null);
        if (!workspaceRoot) {
          setRolloutCandidates([]);
          return;
        }
        const response = await listMemoryStore({
          scope: "workspace",
          workspaceRoot,
          path: "rollout_summaries",
          maxResults: 20,
        });
        const entries = (response.entries ?? [])
          .filter(isRolloutCandidateEntry)
          .sort((left, right) => right.modifiedAt - left.modifiedAt)
          .slice(0, 5);
        const candidates = await Promise.all(
          entries.map(async (entry) => {
            const read = await readMemoryStore({
              scope: "workspace",
              workspaceRoot,
              path: entry.path,
              maxLines: 120,
              maxTokens: 2048,
            });
            return parseRolloutCandidateMarkdown(
              read.path,
              read.content,
              read.truncated,
            );
          }),
        );
        setRolloutCandidates(candidates);
        if (showSuccess) {
          showMessage(
            memoryPanelT(t, "settings.memory.store.message.rolloutRefreshed"),
          );
        }
      } catch (error) {
        console.error("加载运行摘要候选失败:", error);
        setRolloutCandidates([]);
        setRolloutWorkspaceRoot(null);
        setRolloutLoadFailed(true);
        if (showSuccess) {
          showMessage(
            memoryPanelT(t, "settings.memory.store.message.rolloutLoadFailed"),
          );
        }
      } finally {
        setRolloutLoading(false);
      }
    },
    [showMessage, t],
  );

  useEffect(() => {
    void refreshHealth(false);
    void refreshReviewNotes(false);
    void refreshRolloutCandidates(false);
  }, [refreshHealth, refreshReviewNotes, refreshRolloutCandidates]);

  const refreshMemoryStorePanels = useCallback(async () => {
    await refreshHealth(false);
    await refreshReviewNotes(false);
    await refreshRolloutCandidates(false);
  }, [refreshHealth, refreshReviewNotes, refreshRolloutCandidates]);

  const handleReset = async () => {
    const confirmed = window.confirm(
      memoryPanelT(t, "settings.memory.store.resetConfirm"),
    );
    if (!confirmed) {
      return;
    }
    setResetting(true);
    try {
      await resetMemory();
      await refreshMemoryStorePanels();
      showMessage(memoryPanelT(t, "settings.memory.store.message.resetDone"));
    } catch (error) {
      console.error("重置记忆文件失败:", error);
      showMessage(memoryPanelT(t, "settings.memory.store.message.resetFailed"));
    } finally {
      setResetting(false);
    }
  };

  const handleRebuildIndex = async () => {
    setRebuildingIndex(true);
    try {
      const response = await rebuildMemoryStoreIndex({ scope: "global" });
      await refreshHealth(false);
      showMessage(
        memoryPanelT(t, "settings.memory.store.message.indexRebuilt", {
          files: response.sourceFileCount,
        }),
      );
    } catch (error) {
      console.error("重建记忆索引失败:", error);
      showMessage(
        memoryPanelT(t, "settings.memory.store.message.indexRebuildFailed"),
      );
    } finally {
      setRebuildingIndex(false);
    }
  };

  const handleConsolidate = async () => {
    setConsolidating(true);
    try {
      const response = await consolidateMemoryStore({ scope: "global" });
      await refreshMemoryStorePanels();
      showMessage(
        memoryPanelT(t, "settings.memory.store.message.consolidated", {
          processed: response.processedNotes,
          skipped: response.skippedNotes,
        }),
      );
    } catch (error) {
      console.error("整理记忆笔记失败:", error);
      showMessage(
        memoryPanelT(t, "settings.memory.store.message.consolidateFailed"),
      );
    } finally {
      setConsolidating(false);
    }
  };

  const handleConsolidateRolloutCandidates = async () => {
    const workspaceRoot = rolloutWorkspaceRoot?.trim();
    if (!workspaceRoot) {
      showMessage(
        memoryPanelT(
          t,
          "settings.memory.store.message.rolloutWorkspaceMissing",
        ),
      );
      return;
    }
    setRolloutConsolidating(true);
    try {
      const response = await consolidateMemoryStore({
        scope: "workspace",
        workspaceRoot,
      });
      await refreshMemoryStorePanels();
      showMessage(
        memoryPanelT(t, "settings.memory.store.message.rolloutConsolidated", {
          processed: response.processedNotes,
          skipped: response.skippedNotes,
        }),
      );
    } catch (error) {
      console.error("整理运行摘要候选失败:", error);
      showMessage(
        memoryPanelT(
          t,
          "settings.memory.store.message.rolloutConsolidateFailed",
        ),
      );
    } finally {
      setRolloutConsolidating(false);
    }
  };

  const handleAddNote = async () => {
    const content = noteContent.trim();
    if (!content) {
      showMessage(memoryPanelT(t, "settings.memory.store.message.noteEmpty"));
      return;
    }
    setSavingNote(true);
    try {
      await addMemoryStoreNote({
        scope: "global",
        title: memoryPanelT(t, "settings.memory.store.noteDefaultTitle"),
        content,
      });
      setNoteContent("");
      await refreshHealth(false);
      showMessage(memoryPanelT(t, "settings.memory.store.message.noteSaved"));
    } catch (error) {
      console.error("保存记忆修正失败:", error);
      showMessage(memoryPanelT(t, "settings.memory.store.message.noteFailed"));
    } finally {
      setSavingNote(false);
    }
  };

  const handleResolveReviewNote = async (
    note: MemoryStoreReviewNote,
    action: "accept" | "reject",
  ) => {
    setResolvingReviewPath(note.path);
    try {
      await resolveMemoryStoreReviewNote({
        scope: "global",
        path: note.path,
        action,
      });
      await refreshMemoryStorePanels();
      showMessage(
        memoryPanelT(
          t,
          action === "accept"
            ? "settings.memory.store.message.reviewAccepted"
            : "settings.memory.store.message.reviewRejected",
        ),
      );
    } catch (error) {
      console.error("处理待审阅记忆笔记失败:", error);
      showMessage(
        memoryPanelT(t, "settings.memory.store.message.reviewResolveFailed"),
      );
    } finally {
      setResolvingReviewPath(null);
    }
  };

  const summaryStatus =
    health?.summaryExists && health.memoryExists
      ? memoryPanelT(t, "settings.memory.store.summaryReady")
      : memoryPanelT(t, "settings.memory.store.summaryMissing");

  return (
    <section
      className="rounded-md border border-slate-200/90 bg-white p-5 shadow-sm shadow-slate-950/5"
      data-testid="settings-memory-store-panel"
    >
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-emerald-200 bg-emerald-50 text-emerald-700">
            <Database className="h-5 w-5" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-semibold text-slate-950">
              {memoryPanelT(t, "settings.memory.everyday.title")}
            </p>
            <p className="mt-1 text-sm leading-6 text-slate-500">
              {memoryPanelT(t, "settings.memory.everyday.description")}
            </p>
          </div>
        </div>
        <div className="flex shrink-0 flex-wrap gap-2">
          <button
            type="button"
            onClick={() => void refreshHealth(true)}
            data-testid="settings-memory-health-refresh"
            disabled={
              healthLoading ||
              reviewLoading ||
              rolloutLoading ||
              consolidating ||
              rolloutConsolidating ||
              rebuildingIndex ||
              resetting ||
              Boolean(resolvingReviewPath)
            }
            className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 transition hover:border-slate-300 hover:text-slate-950 disabled:opacity-60"
          >
            {healthLoading
              ? memoryPanelT(t, "settings.memory.store.loading")
              : memoryPanelT(t, "settings.memory.action.refresh")}
          </button>
          <button
            type="button"
            onClick={handleConsolidate}
            data-testid="settings-memory-consolidate"
            disabled={
              consolidating ||
              rolloutConsolidating ||
              rebuildingIndex ||
              resetting ||
              Boolean(resolvingReviewPath)
            }
            className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 transition hover:border-slate-300 hover:text-slate-950 disabled:opacity-60"
          >
            {consolidating
              ? memoryPanelT(t, "settings.memory.store.consolidating")
              : memoryPanelT(t, "settings.memory.store.consolidate")}
          </button>
          <button
            type="button"
            onClick={handleRebuildIndex}
            data-testid="settings-memory-index-rebuild"
            disabled={
              consolidating ||
              rolloutConsolidating ||
              rebuildingIndex ||
              resetting ||
              Boolean(resolvingReviewPath)
            }
            className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 transition hover:border-slate-300 hover:text-slate-950 disabled:opacity-60"
          >
            {rebuildingIndex
              ? memoryPanelT(t, "settings.memory.store.indexRebuilding")
              : memoryPanelT(t, "settings.memory.store.indexRebuild")}
          </button>
          <button
            type="button"
            onClick={handleReset}
            data-testid="settings-memory-reset"
            disabled={
              consolidating ||
              rolloutConsolidating ||
              rebuildingIndex ||
              resetting ||
              Boolean(resolvingReviewPath)
            }
            className="rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-medium text-rose-700 transition hover:border-rose-300 hover:bg-rose-100 disabled:opacity-60"
          >
            {resetting
              ? memoryPanelT(t, "settings.memory.store.resetting")
              : memoryPanelT(t, "settings.memory.store.reset")}
          </button>
        </div>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        <div className="rounded-md border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs font-medium text-slate-500">
            {memoryPanelT(t, "settings.memory.embedding.status.vectorSearch")}
          </p>
          <p
            className={cn(
              "mt-2 text-base font-semibold",
              vectorSearchEnabled ? "text-emerald-700" : "text-slate-500",
            )}
          >
            {vectorSearchEnabled
              ? memoryPanelT(t, "settings.memory.embedding.status.enabled")
              : memoryPanelT(t, "settings.memory.embedding.status.disabled")}
          </p>
        </div>
        <div className="rounded-md border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs font-medium text-slate-500">
            {memoryPanelT(t, "settings.memory.embedding.status.config")}
          </p>
          <p className="mt-2 text-base font-semibold text-slate-950">
            {vectorSearchEnabled
              ? memoryPanelT(t, "settings.memory.embedding.status.configured")
              : memoryPanelT(
                  t,
                  "settings.memory.embedding.status.fullTextOnly",
                )}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {memoryPanelT(t, memoryStatusDescriptionKey)}
          </p>
        </div>
        <div className="rounded-md border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs font-medium text-slate-500">
            {memoryPanelT(t, "settings.memory.store.files")}
          </p>
          <p className="mt-2 text-base font-semibold text-slate-950">
            {memoryPanelT(t, "settings.memory.store.filesValue", {
              count: health?.fileCount ?? 0,
              size: formatStorageSize(health?.totalBytes ?? 0),
            })}
          </p>
        </div>
        <div className="rounded-md border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs font-medium text-slate-500">
            {memoryPanelT(t, "settings.memory.store.notes")}
          </p>
          <p className="mt-2 text-base font-semibold text-slate-950">
            {memoryPanelT(t, "settings.memory.store.notesValue", {
              count: health?.notesCount ?? 0,
            })}
          </p>
        </div>
      </div>

      <div className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <p className="text-xs font-medium text-slate-500">
              {memoryPanelT(t, "settings.memory.store.summary")}
            </p>
            <p className="mt-1 text-sm font-semibold text-slate-950">
              {summaryStatus}
            </p>
          </div>
          <div>
            <p className="text-xs font-medium text-slate-500">
              {memoryPanelT(t, "settings.memory.store.scope")}
            </p>
            <p className="mt-1 text-sm font-semibold text-slate-950">
              {healthRootScope(t, health)}
            </p>
          </div>
        </div>
        <p className="mt-3 truncate rounded-md bg-white px-3 py-2 text-xs text-slate-500 ring-1 ring-slate-200">
          {health?.rootPath ??
            memoryPanelT(t, "settings.memory.store.pathUnavailable")}
        </p>
      </div>

      <div className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-xs font-medium text-slate-500">
              {memoryPanelT(t, "settings.memory.store.noteTitle")}
            </p>
            <p className="mt-1 text-sm text-slate-600">
              {memoryPanelT(t, "settings.memory.store.noteDescription")}
            </p>
          </div>
          <button
            type="button"
            onClick={handleAddNote}
            disabled={savingNote}
            data-testid="settings-memory-note-save"
            className="rounded-md bg-slate-950 px-3 py-2 text-xs font-medium text-white transition hover:bg-slate-800 disabled:opacity-60"
          >
            {savingNote
              ? memoryPanelT(t, "settings.memory.store.noteSaving")
              : memoryPanelT(t, "settings.memory.store.noteSave")}
          </button>
        </div>
        <textarea
          value={noteContent}
          onChange={(event) => setNoteContent(event.target.value)}
          data-testid="settings-memory-note-textarea"
          placeholder={memoryPanelT(t, "settings.memory.store.notePlaceholder")}
          className="mt-3 min-h-24 w-full resize-y rounded-md border border-slate-200 bg-white px-3 py-2 text-sm leading-6 text-slate-800 outline-none transition placeholder:text-slate-400 focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100"
        />
      </div>

      <RolloutCandidatesPanel
        t={t}
        candidates={rolloutCandidates}
        loading={rolloutLoading}
        loadFailed={rolloutLoadFailed}
        workspaceName={rolloutWorkspaceName}
        consolidating={rolloutConsolidating}
        onRefresh={() => void refreshRolloutCandidates(true)}
        onConsolidate={() => void handleConsolidateRolloutCandidates()}
      />

      <div className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-xs font-medium text-slate-500">
              {memoryPanelT(t, "settings.memory.store.reviewTitle")}
            </p>
            <p className="mt-1 text-sm text-slate-600">
              {reviewLoading
                ? memoryPanelT(t, "settings.memory.store.reviewLoading")
                : memoryPanelT(t, "settings.memory.store.reviewCount", {
                    count: reviewNotes.length,
                  })}
            </p>
          </div>
          <button
            type="button"
            onClick={() => void refreshReviewNotes(true)}
            disabled={reviewLoading || Boolean(resolvingReviewPath)}
            data-testid="settings-memory-review-refresh"
            className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 transition hover:border-slate-300 hover:text-slate-950 disabled:opacity-60"
          >
            {reviewLoading
              ? memoryPanelT(t, "settings.memory.store.reviewRefreshing")
              : memoryPanelT(t, "settings.memory.store.reviewRefresh")}
          </button>
        </div>
        {reviewNotes.length === 0 ? (
          <p className="mt-3 rounded-md bg-white px-3 py-2 text-sm text-slate-500 ring-1 ring-slate-200">
            {memoryPanelT(t, "settings.memory.store.reviewEmpty")}
          </p>
        ) : (
          <div className="mt-3 space-y-2">
            {reviewNotes.map((note) => {
              const resolving = resolvingReviewPath === note.path;
              return (
                <div
                  key={note.path}
                  className="rounded-md border border-slate-200 bg-white p-3"
                >
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-medium text-slate-500">
                        {note.path}
                      </p>
                      <p className="mt-1 text-sm leading-6 text-slate-800">
                        {note.preview}
                      </p>
                    </div>
                    <div className="flex shrink-0 gap-2">
                      <button
                        type="button"
                        onClick={() =>
                          void handleResolveReviewNote(note, "accept")
                        }
                        disabled={Boolean(resolvingReviewPath)}
                        className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-medium text-emerald-700 transition hover:border-emerald-300 hover:bg-emerald-100 disabled:opacity-60"
                      >
                        {resolving
                          ? memoryPanelT(
                              t,
                              "settings.memory.store.reviewResolving",
                            )
                          : memoryPanelT(
                              t,
                              "settings.memory.store.reviewAccept",
                            )}
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          void handleResolveReviewNote(note, "reject")
                        }
                        disabled={Boolean(resolvingReviewPath)}
                        className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 transition hover:border-slate-300 hover:text-slate-950 disabled:opacity-60"
                      >
                        {memoryPanelT(t, "settings.memory.store.reviewReject")}
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
