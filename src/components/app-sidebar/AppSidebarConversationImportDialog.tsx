import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  FileInput,
  Loader2,
  X,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { Modal } from "@/components/Modal";
import {
  commitConversationImportThread,
  previewConversationImportThread,
  readConversationImportJob,
  scanConversationImportSource,
  waitForConversationImportJob,
  type ConversationImportJob,
  type ConversationImportThreadCommitResponse,
  type ConversationImportThreadPreviewResponse,
  type ImportedThreadSummary,
} from "@/lib/api/conversationImport";
import { open as openDesktopDialog } from "@/lib/desktop-host/dialog";
import { formatNumber } from "@/i18n/format";
import {
  DEFAULT_CONVERSATION_IMPORT_SOURCE_CLIENT,
  buildImportThreadGroups,
  buildImportPreviewMetaText,
  buildSourceProvenanceLabels,
  filterImportThreadsByArchiveStatus,
  formatImportOptionalDate,
  firstImportableThread,
  initialImportSelection,
  isSelectableImportThread,
  isImportingThread,
  isImportedThread,
  normalizeOptional,
  resolveImportSourceClientLabel,
  resolveImportThreadTitle,
  resolveImportWarningText,
  selectedImportThreads,
  truncateImportPreviewText,
  type ImportThreadArchiveFilter,
  type ImportThreadGroupMode,
} from "./conversationImportDialogViewModel";
import { AppSidebarConversationImportThreadList } from "./AppSidebarConversationImportThreadList";
import { AppSidebarConversationImportProgress } from "./AppSidebarConversationImportProgress";

const SCAN_LIMIT = 40;
const PREVIEW_LIMIT = 12;

interface AppSidebarConversationImportDialogProps {
  isOpen: boolean;
  workspaceId?: string | null;
  projectPath?: string | null;
  projectName?: string | null;
  onClose: () => void;
  onImported: (responses: ConversationImportThreadCommitResponse[]) => void;
}

type ImportStage = "idle" | "scanning" | "previewing" | "committing";

export function AppSidebarConversationImportDialog({
  isOpen,
  workspaceId,
  projectPath,
  projectName,
  onClose,
  onImported,
}: AppSidebarConversationImportDialogProps) {
  const { t, i18n } = useTranslation("navigation");
  const [stage, setStage] = useState<ImportStage>("idle");
  const [sourceRootInput, setSourceRootInput] = useState("");
  const [threads, setThreads] = useState<ImportedThreadSummary[]>([]);
  const [selectedThreadId, setSelectedThreadId] = useState<string | null>(null);
  const [checkedThreadIds, setCheckedThreadIds] = useState<Set<string>>(
    () => new Set(),
  );
  const [groupMode, setGroupMode] = useState<ImportThreadGroupMode>("day");
  const [archiveFilter, setArchiveFilter] =
    useState<ImportThreadArchiveFilter>("all");
  const [selectingSourceRoot, setSelectingSourceRoot] = useState(false);
  const [preview, setPreview] =
    useState<ConversationImportThreadPreviewResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sourceMessage, setSourceMessage] = useState<string | null>(null);
  const [activeImportJob, setActiveImportJob] =
    useState<ConversationImportJob | null>(null);
  const [activeImportIndex, setActiveImportIndex] = useState(0);
  const importWaitAbortRef = useRef<AbortController | null>(null);

  const sourceRoot = normalizeOptional(sourceRootInput);
  const sourceRootRef = useRef<string | undefined>(sourceRoot);
  const selectedThread = useMemo(
    () =>
      threads.find((thread) => thread.sourceThreadId === selectedThreadId) ??
      null,
    [selectedThreadId, threads],
  );
  const loading = stage === "scanning" || stage === "previewing";
  const committing = stage === "committing";
  const threadTitle = resolveImportThreadTitle(
    preview?.thread ?? selectedThread,
    t,
  );
  const commitThreads = useMemo(
    () => selectedImportThreads(threads, checkedThreadIds),
    [checkedThreadIds, threads],
  );
  const visibleThreads = useMemo(
    () => filterImportThreadsByArchiveStatus(threads, archiveFilter),
    [archiveFilter, threads],
  );
  const selectableThreads = useMemo(
    () => visibleThreads.filter(isSelectableImportThread),
    [visibleThreads],
  );
  const importGroups = useMemo(
    () => buildImportThreadGroups(visibleThreads, groupMode, i18n.language, t),
    [groupMode, i18n.language, t, visibleThreads],
  );
  const checkedCount = commitThreads.length;
  const checkedImportedCount = commitThreads.filter(isImportedThread).length;
  const checkedImportingCount = commitThreads.filter(isImportingThread).length;
  const allSelectableChecked =
    selectableThreads.length > 0 &&
    selectableThreads.every((thread) =>
      checkedThreadIds.has(thread.sourceThreadId),
    );
  const dryRun = preview?.summary.dryRun;
  const fidelity = preview?.summary.fidelity;
  const targetLabel =
    normalizeOptional(projectName) ||
    normalizeOptional(workspaceId) ||
    t(
      "navigation.sidebar.importDialog.target.standalone",
      "Standalone conversation",
    );
  const selectedUpdatedAt = formatImportOptionalDate(
    preview?.thread.updatedAt ?? selectedThread?.updatedAt,
    i18n.language,
  );
  const previewMetaText = buildImportPreviewMetaText(selectedUpdatedAt, t);

  const loadPreview = useCallback(
    async (thread: ImportedThreadSummary, nextSourceRoot?: string) => {
      setStage("previewing");
      setError(null);
      const result = await previewConversationImportThread({
        sourceClient: DEFAULT_CONVERSATION_IMPORT_SOURCE_CLIENT,
        sourceRoot: nextSourceRoot,
        sourceThreadId: thread.sourceThreadId,
        sourcePath: thread.sourcePath,
        limit: PREVIEW_LIMIT,
      });
      setPreview(result);
    },
    [],
  );

  const loadThreads = useCallback(
    async (nextSourceRoot?: string) => {
      setStage("scanning");
      setError(null);
      setSourceMessage(null);
      setPreview(null);
      try {
        let cursor: string | undefined;
        let resolvedSourceRoot = nextSourceRoot;
        const nextThreads: ImportedThreadSummary[] = [];
        do {
          const result = await scanConversationImportSource({
            sourceClient: DEFAULT_CONVERSATION_IMPORT_SOURCE_CLIENT,
            sourceRoot: resolvedSourceRoot,
            projectPath: normalizeOptional(projectPath),
            includeArchived: true,
            limit: SCAN_LIMIT,
            ...(cursor ? { cursor } : {}),
          });
          resolvedSourceRoot = result.source.sourceRoot ?? resolvedSourceRoot;
          setSourceMessage(result.source.message ?? null);
          nextThreads.push(...result.threads);
          cursor = result.nextCursor ?? undefined;
        } while (cursor);
        setThreads(nextThreads);
        const nextSelected = firstImportableThread(nextThreads);
        setSelectedThreadId(nextSelected?.sourceThreadId ?? null);
        setCheckedThreadIds(initialImportSelection(nextThreads));
        if (nextSelected) {
          await loadPreview(nextSelected, resolvedSourceRoot);
        }
      } catch (scanError) {
        setThreads([]);
        setSelectedThreadId(null);
        setCheckedThreadIds(new Set());
        setSourceMessage(null);
        setError(
          scanError instanceof Error && scanError.message.trim()
            ? scanError.message.trim()
            : t(
                "navigation.sidebar.importDialog.error.scan",
                "Failed to read local history",
              ),
        );
      } finally {
        setStage("idle");
      }
    },
    [loadPreview, projectPath, t],
  );

  useEffect(() => {
    sourceRootRef.current = sourceRoot;
  }, [sourceRoot]);

  useEffect(() => {
    if (!isOpen) {
      importWaitAbortRef.current?.abort();
      importWaitAbortRef.current = null;
      setStage("idle");
      setError(null);
      setSourceMessage(null);
      setPreview(null);
      setActiveImportJob(null);
      setActiveImportIndex(0);
      return;
    }
    void loadThreads(sourceRootRef.current);
  }, [isOpen, loadThreads]);

  const handleDialogClose = useCallback(() => {
    importWaitAbortRef.current?.abort();
    importWaitAbortRef.current = null;
    onClose();
  }, [onClose]);

  const handleSelectThread = useCallback(
    async (thread: ImportedThreadSummary) => {
      if (committing || loading) {
        return;
      }
      setSelectedThreadId(thread.sourceThreadId);
      try {
        await loadPreview(thread, preview?.source.sourceRoot ?? sourceRoot);
      } catch (previewError) {
        setPreview(null);
        setError(
          previewError instanceof Error && previewError.message.trim()
            ? previewError.message.trim()
            : t(
                "navigation.sidebar.importDialog.error.preview",
                "Failed to read conversation preview",
              ),
        );
      } finally {
        setStage("idle");
      }
    },
    [
      committing,
      loadPreview,
      loading,
      preview?.source.sourceRoot,
      sourceRoot,
      t,
    ],
  );

  const handleToggleThread = useCallback(
    (thread: ImportedThreadSummary) => {
      if (committing || loading || !isSelectableImportThread(thread)) {
        return;
      }
      setCheckedThreadIds((current) => {
        const next = new Set(current);
        if (next.has(thread.sourceThreadId)) {
          next.delete(thread.sourceThreadId);
        } else {
          next.add(thread.sourceThreadId);
        }
        return next;
      });
      if (selectedThreadId !== thread.sourceThreadId) {
        void handleSelectThread(thread);
      }
    },
    [committing, handleSelectThread, loading, selectedThreadId],
  );

  const handleToggleAll = useCallback(() => {
    if (committing || loading) {
      return;
    }
    setCheckedThreadIds(() => {
      if (allSelectableChecked) {
        return new Set();
      }
      return new Set(selectableThreads.map((thread) => thread.sourceThreadId));
    });
  }, [allSelectableChecked, committing, loading, selectableThreads]);

  const handleToggleGroup = useCallback(
    (groupThreads: ImportedThreadSummary[]) => {
      if (committing || loading) {
        return;
      }
      const selectableGroupThreads = groupThreads.filter(
        isSelectableImportThread,
      );
      const groupChecked =
        selectableGroupThreads.length > 0 &&
        selectableGroupThreads.every((thread) =>
          checkedThreadIds.has(thread.sourceThreadId),
        );
      setCheckedThreadIds((current) => {
        const next = new Set(current);
        for (const thread of selectableGroupThreads) {
          if (groupChecked) {
            next.delete(thread.sourceThreadId);
          } else {
            next.add(thread.sourceThreadId);
          }
        }
        return next;
      });
    },
    [checkedThreadIds, committing, loading],
  );

  const handlePickSourceRoot = useCallback(async () => {
    if (committing || loading || selectingSourceRoot) {
      return;
    }
    setSelectingSourceRoot(true);
    setError(null);
    try {
      const result = await openDesktopDialog({
        title: t(
          "navigation.sidebar.importDialog.sourceRoot.dialogTitle",
          "Select local history data directory",
        ),
        directory: true,
        multiple: false,
      });
      const nextSourceRoot = normalizeOptional(result);
      if (!nextSourceRoot) {
        return;
      }
      sourceRootRef.current = nextSourceRoot;
      setSourceRootInput(nextSourceRoot);
      await loadThreads(nextSourceRoot);
    } catch (pickError) {
      setError(
        pickError instanceof Error && pickError.message.trim()
          ? pickError.message.trim()
          : t(
              "navigation.sidebar.importDialog.error.chooseDirectory",
              "Failed to choose local history directory",
            ),
      );
    } finally {
      setSelectingSourceRoot(false);
    }
  }, [committing, loadThreads, loading, selectingSourceRoot, t]);

  const handleCommit = useCallback(async () => {
    if (commitThreads.length === 0) {
      return;
    }

    const waitAbortController = new AbortController();
    importWaitAbortRef.current?.abort();
    importWaitAbortRef.current = waitAbortController;
    setStage("committing");
    setError(null);
    try {
      const resolvedSourceRoot = preview?.source.sourceRoot ?? sourceRoot;
      const startedJobs: ConversationImportJob[] = [];
      const results: ConversationImportThreadCommitResponse[] = [];
      for (const [index, thread] of commitThreads.entries()) {
        setActiveImportIndex(index + 1);
        const activeJobId = isImportingThread(thread)
          ? normalizeOptional(thread.importJobId)
          : undefined;
        if (activeJobId) {
          const resumed = await readConversationImportJob({
            jobId: activeJobId,
          });
          setActiveImportJob(resumed.job);
          startedJobs.push(resumed.job);
          continue;
        }
        const started = await commitConversationImportThread({
          sourceClient: DEFAULT_CONVERSATION_IMPORT_SOURCE_CLIENT,
          sourceRoot: resolvedSourceRoot,
          sourceThreadId: thread.sourceThreadId,
          sourcePath: thread.sourcePath,
          workspaceId: normalizeOptional(workspaceId),
          confirmed: true,
          ...(isImportedThread(thread) ? { replaceExisting: true } : {}),
        });
        setActiveImportJob(started.job);
        startedJobs.push(started.job);
      }
      for (const [index, job] of startedJobs.entries()) {
        setActiveImportIndex(index + 1);
        setActiveImportJob(job);
        const result = await waitForConversationImportJob(job, {
          signal: waitAbortController.signal,
          onProgress: setActiveImportJob,
        });
        results.push(result);
      }
      onImported(results);
    } catch (commitError) {
      if (
        commitError instanceof DOMException &&
        commitError.name === "AbortError"
      ) {
        return;
      }
      setError(
        commitError instanceof Error && commitError.message.trim()
          ? commitError.message.trim()
          : t(
              "navigation.sidebar.importDialog.error.commit",
              "Failed to import local history",
            ),
      );
    } finally {
      if (importWaitAbortRef.current === waitAbortController) {
        importWaitAbortRef.current = null;
      }
      setStage("idle");
    }
  }, [
    onImported,
    commitThreads,
    preview?.source.sourceRoot,
    sourceRoot,
    t,
    workspaceId,
  ]);

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleDialogClose}
      className="p-0"
      maxWidth="max-w-[920px]"
      showCloseButton={false}
      closeOnOverlayClick
    >
      <div
        className="relative flex max-h-[calc(100vh-4rem)] min-h-[540px] flex-col overflow-hidden bg-white text-slate-900"
        data-testid="app-sidebar-conversation-import-dialog"
      >
        <button
          type="button"
          aria-label={t(
            "navigation.sidebar.importDialog.close",
            "Close import dialog",
          )}
          className="absolute right-4 top-4 z-10 inline-flex h-9 w-9 items-center justify-center rounded-lg border border-transparent text-slate-500 transition hover:border-slate-200 hover:bg-slate-50 hover:text-slate-900 disabled:cursor-not-allowed disabled:opacity-50"
          onClick={handleDialogClose}
          data-testid="app-sidebar-conversation-import-close"
        >
          <X className="h-4 w-4" />
        </button>

        <header className="flex items-center justify-between gap-4 border-b border-slate-200 bg-slate-50 px-6 py-4 pr-16">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <FileInput className="h-4 w-4 text-emerald-700" />
              <h2 className="text-lg font-semibold text-slate-950">
                {t(
                  "navigation.sidebar.importDialog.title",
                  "Import Conversation",
                )}
              </h2>
            </div>
            <p className="mt-1 truncate text-sm text-slate-500">
              {targetLabel}
            </p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700">
            {t(
              "navigation.sidebar.importDialog.selection.selectedCount",
              "Selected {{count}}",
              {
                count: checkedCount,
              },
            )}
          </div>
        </header>

        <div className="grid min-h-0 flex-1 grid-cols-[minmax(300px,380px)_minmax(0,1fr)]">
          <AppSidebarConversationImportThreadList
            allSelectableChecked={allSelectableChecked}
            archiveFilter={archiveFilter}
            checkedThreadIds={checkedThreadIds}
            committing={committing}
            groupMode={groupMode}
            importGroups={importGroups}
            loading={loading}
            locale={i18n.language}
            emptyMessage={sourceMessage}
            selectableThreadCount={selectableThreads.length}
            selectedThreadId={selectedThreadId}
            selectingSourceRoot={selectingSourceRoot}
            sourceRoot={sourceRoot}
            sourceRootInput={sourceRootInput}
            t={t}
            visibleThreads={visibleThreads}
            onArchiveFilterChange={setArchiveFilter}
            onGroupModeChange={setGroupMode}
            onPickSourceRoot={handlePickSourceRoot}
            onRefresh={(nextSourceRoot) => void loadThreads(nextSourceRoot)}
            onSelectThread={handleSelectThread}
            onSourceRootInputChange={setSourceRootInput}
            onToggleAll={handleToggleAll}
            onToggleGroup={handleToggleGroup}
            onToggleThread={handleToggleThread}
          />

          <main className="flex min-h-0 flex-col">
            <div className="grid grid-cols-3 gap-3 border-b border-slate-200 p-4">
              <div className="rounded-lg border border-slate-200 bg-white p-3">
                <span className="text-xs font-semibold text-slate-500">
                  {t("navigation.sidebar.importDialog.meta.source", "Source")}
                </span>
                <strong className="mt-1 block text-sm text-slate-950">
                  {resolveImportSourceClientLabel(t)}
                </strong>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-3">
                <span className="text-xs font-semibold text-slate-500">
                  {t(
                    "navigation.sidebar.importDialog.meta.selected",
                    "Selected",
                  )}
                </span>
                <strong className="mt-1 block truncate text-sm text-slate-950">
                  {formatNumber(checkedCount, { locale: i18n.language })}
                </strong>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-3">
                <span className="text-xs font-semibold text-slate-500">
                  {t(
                    "navigation.sidebar.importDialog.meta.reimport",
                    "Re-import",
                  )}
                </span>
                <strong className="mt-1 block text-sm text-slate-950">
                  {formatNumber(checkedImportedCount, {
                    locale: i18n.language,
                  })}
                </strong>
              </div>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto p-5">
              {error ? (
                <div
                  role="alert"
                  className="mb-4 flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700"
                >
                  <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
                  <span>{error}</span>
                </div>
              ) : null}

              {preview ? (
                <div className="space-y-4">
                  <section className="rounded-xl border border-slate-200 bg-white p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <h3 className="truncate text-lg font-semibold text-slate-950">
                          {threadTitle}
                        </h3>
                        <p className="mt-2 text-sm leading-6 text-slate-500">
                          {previewMetaText}
                        </p>
                      </div>
                      {preview.thread.importStatus === "imported" ? (
                        <span className="inline-flex flex-shrink-0 items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700">
                          <CheckCircle2 className="h-3.5 w-3.5" />
                          {t(
                            "navigation.sidebar.importDialog.status.imported",
                            "Imported",
                          )}
                        </span>
                      ) : null}
                      {isImportingThread(preview.thread) ? (
                        <span className="inline-flex flex-shrink-0 items-center gap-1 rounded-full border border-sky-200 bg-sky-50 px-2.5 py-1 text-xs font-semibold text-sky-700">
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          {t(
                            "navigation.sidebar.importDialog.status.importing",
                            "Importing",
                          )}
                        </span>
                      ) : null}
                    </div>

                    <div className="mt-4 grid grid-cols-4 gap-2">
                      {[
                        [
                          t(
                            "navigation.sidebar.importDialog.summary.messages",
                            "Messages",
                          ),
                          dryRun?.willImportMessages ??
                            preview.summary.messageCount,
                        ],
                        [
                          t(
                            "navigation.sidebar.importDialog.summary.turns",
                            "Turns",
                          ),
                          dryRun?.willImportTurns ?? 0,
                        ],
                        [
                          t(
                            "navigation.sidebar.importDialog.summary.attachments",
                            "Attachments",
                          ),
                          dryRun?.willImportAttachments ?? 0,
                        ],
                        [
                          t(
                            "navigation.sidebar.importDialog.summary.timeline",
                            "Timeline",
                          ),
                          dryRun?.willImportTimelineItems ??
                            preview.summary.messageCount +
                              preview.summary.rolloutEventItems,
                        ],
                      ].map(([label, value]) => (
                        <div
                          key={String(label)}
                          className="rounded-lg bg-slate-50 p-3"
                        >
                          <span className="block text-xs font-semibold text-slate-500">
                            {label}
                          </span>
                          <strong className="mt-1 block text-base text-slate-950">
                            {formatNumber(Number(value), {
                              locale: i18n.language,
                            })}
                          </strong>
                        </div>
                      ))}
                    </div>
                    {fidelity ? (
                      <p className="mt-3 text-xs leading-5 text-slate-500">
                        {t(
                          "navigation.sidebar.importDialog.fidelity.compact",
                          "Tools, commands, patches, approvals, and reasoning records will be preserved.",
                        )}
                      </p>
                    ) : null}
                  </section>

                  {preview.summary.warnings.length > 0 ? (
                    <section className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
                      <div className="mb-2 flex items-center gap-2 font-semibold">
                        <AlertTriangle className="h-4 w-4" />
                        {t(
                          "navigation.sidebar.importDialog.warnings.title",
                          "Import notes",
                        )}
                      </div>
                      <ul className="space-y-1">
                        {preview.summary.warnings.map((warning) => (
                          <li key={warning}>
                            {resolveImportWarningText(warning, t)}
                          </li>
                        ))}
                      </ul>
                    </section>
                  ) : null}

                  <section className="space-y-3">
                    <h4 className="text-sm font-semibold text-slate-700">
                      {t(
                        "navigation.sidebar.importDialog.messages.title",
                        "Message preview",
                      )}
                    </h4>
                    <div className="space-y-2">
                      {preview.messages.map((message, index) => {
                        const provenanceLabels = buildSourceProvenanceLabels(
                          message.provenance,
                          t,
                        );
                        return (
                          <article
                            key={`${message.role}-${index}`}
                            className="rounded-xl border border-slate-200 bg-white p-4"
                          >
                            <div className="mb-2 flex items-center justify-between gap-3">
                              <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-600">
                                {message.role === "assistant"
                                  ? t(
                                      "navigation.sidebar.importDialog.role.assistant",
                                      "Assistant",
                                    )
                                  : t(
                                      "navigation.sidebar.importDialog.role.user",
                                      "User",
                                    )}
                              </span>
                              <span className="flex flex-wrap justify-end gap-2">
                                {message.truncated ? (
                                  <span className="text-xs font-medium text-amber-700">
                                    {t(
                                      "navigation.sidebar.importDialog.messages.truncated",
                                      "Truncated",
                                    )}
                                  </span>
                                ) : null}
                                {(message.attachments ?? []).length > 0 ? (
                                  <span className="text-xs font-medium text-emerald-700">
                                    {t(
                                      "navigation.sidebar.importDialog.messages.attachments",
                                      "Attachments {{count}}",
                                      {
                                        count: (message.attachments ?? [])
                                          .length,
                                      },
                                    )}
                                  </span>
                                ) : null}
                              </span>
                            </div>
                            {provenanceLabels.length > 0 ? (
                              <div className="mb-2 flex flex-wrap gap-2 text-[11px] font-medium text-slate-500">
                                {provenanceLabels.map((label) => (
                                  <span key={label}>{label}</span>
                                ))}
                              </div>
                            ) : null}
                            <p className="whitespace-pre-wrap text-sm leading-6 text-slate-700">
                              {truncateImportPreviewText(message.text)}
                            </p>
                          </article>
                        );
                      })}
                    </div>
                  </section>
                </div>
              ) : (
                <div className="flex min-h-[360px] items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50 text-sm font-medium text-slate-500">
                  {loading ? (
                    <span className="inline-flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      {t(
                        "navigation.sidebar.importDialog.preview.loading",
                        "Generating preview",
                      )}
                    </span>
                  ) : (
                    t(
                      "navigation.sidebar.importDialog.preview.empty",
                      "Select a local history conversation to preview",
                    )
                  )}
                </div>
              )}
            </div>

            <footer className="border-t border-slate-200 bg-slate-50">
              {committing && activeImportJob ? (
                <AppSidebarConversationImportProgress
                  job={activeImportJob}
                  currentThread={activeImportIndex}
                  totalThreads={commitThreads.length}
                />
              ) : null}
              <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-4">
                <p className="max-w-xl text-xs leading-5 text-slate-500">
                  {committing
                    ? t(
                        "navigation.sidebar.importDialog.progress.background",
                        "Import continues in the background.",
                      )
                    : checkedImportedCount > 0
                      ? t(
                          "navigation.sidebar.importDialog.confirmNotice.replace",
                          "Already imported conversations will be cleared and imported again.",
                        )
                      : t(
                          "navigation.sidebar.importDialog.confirmNotice",
                          "Only checked conversations will be imported.",
                        )}
                </p>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    className="inline-flex h-10 items-center justify-center rounded-lg border border-slate-200 bg-white px-4 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
                    onClick={handleDialogClose}
                  >
                    {committing
                      ? t(
                          "navigation.sidebar.importDialog.action.close",
                          "Close",
                        )
                      : t(
                          "navigation.sidebar.importDialog.action.cancel",
                          "Cancel",
                        )}
                  </button>
                  <button
                    type="button"
                    className="inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-slate-900 bg-slate-900 px-4 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
                    disabled={
                      commitThreads.length === 0 || loading || committing
                    }
                    onClick={() => void handleCommit()}
                    data-testid="app-sidebar-conversation-import-confirm"
                  >
                    {committing ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : null}
                    {committing
                      ? t(
                          "navigation.sidebar.importDialog.action.importing",
                          "Importing",
                        )
                      : checkedImportedCount > 0
                        ? t(
                            "navigation.sidebar.importDialog.action.replace",
                            "Re-import",
                          )
                        : checkedImportingCount > 0
                          ? t(
                              "navigation.sidebar.importDialog.action.monitor",
                              "View progress",
                            )
                          : t(
                              "navigation.sidebar.importDialog.action.confirm",
                              "Import Conversation",
                            )}
                  </button>
                </div>
              </div>
            </footer>
          </main>
        </div>
      </div>
    </Modal>
  );
}
