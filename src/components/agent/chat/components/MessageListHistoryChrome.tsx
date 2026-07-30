import React from "react";
import { Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { formatNumber } from "@/i18n/format";
import { LIME_BRAND_LOGO_SRC, LIME_BRAND_NAME } from "@/lib/branding";

function formatContentLength(value: number): string {
  return formatNumber(value);
}

interface PersistedHistoryWindowProps {
  loadedEntries: number;
  hasMore: boolean;
  isLoadingFull?: boolean;
  error?: string | null;
  onLoadFullHistory?: () => void | Promise<void>;
}

export function PersistedHistoryWindow({
  loadedEntries,
  hasMore,
  isLoadingFull = false,
  error = null,
  onLoadFullHistory,
}: PersistedHistoryWindowProps) {
  const { t } = useTranslation("agent");

  return (
    <div
      data-testid="message-list-persisted-history-window"
      className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200/80 bg-slate-50/80 px-4 py-3 text-sm text-slate-600"
    >
      <div className="min-w-0 flex-1">
        {t("agentChat.messageList.history.persistedEntrySummary", {
          loaded: formatContentLength(loadedEntries),
        })}
        {error ? <span className="ml-2 text-red-600">{error}</span> : null}
      </div>
      <button
        type="button"
        data-testid="message-list-load-full-history"
        className="shrink-0 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
        disabled={!hasMore || isLoadingFull || !onLoadFullHistory}
        onClick={() => {
          void onLoadFullHistory?.();
        }}
      >
        {isLoadingFull
          ? t("agentChat.messageList.history.loadingMore")
          : t("agentChat.messageList.history.loadMore")}
      </button>
    </div>
  );
}

interface HistoryWindowProps {
  hiddenHistoryCount: number;
  isRestoredHistoryWindow: boolean;
  renderedEntriesCount: number;
  onExpandAllHistory: () => void;
}

export function HistoryWindow({
  hiddenHistoryCount,
  isRestoredHistoryWindow,
  renderedEntriesCount,
  onExpandAllHistory,
}: HistoryWindowProps) {
  const { t } = useTranslation("agent");

  return (
    <div
      data-testid="message-list-history-window"
      data-hidden-history-count={hiddenHistoryCount}
      data-rendered-entries-count={renderedEntriesCount}
      data-restored-history-window={isRestoredHistoryWindow ? "true" : "false"}
      className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200/80 bg-slate-50/80 px-4 py-3 text-sm text-slate-600"
    >
      <div className="min-w-0 flex-1">
        {t(
          isRestoredHistoryWindow
            ? "agentChat.messageList.history.windowEntrySummaryRestored"
            : "agentChat.messageList.history.windowEntrySummaryDeferred",
          {
            loaded: formatContentLength(renderedEntriesCount),
            hidden: formatContentLength(hiddenHistoryCount),
          },
        )}
      </div>
      <button
        type="button"
        data-testid="message-list-expand-history"
        className="shrink-0 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-100"
        onClick={onExpandAllHistory}
      >
        {t("agentChat.messageList.history.expandEarlierEntries")}
      </button>
    </div>
  );
}

export function RestoringSessionEmptyState() {
  const { t } = useTranslation("agent");

  return (
    <div
      className="flex h-64 flex-col items-center justify-center gap-3 text-muted-foreground"
      data-testid="message-list-restoring-session"
      role="status"
      aria-live="polite"
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full border border-border/70 bg-background/80 shadow-sm">
        <Loader2 className="h-5 w-5 animate-spin" />
      </div>
      <div className="space-y-1 text-center">
        <p className="text-lg font-medium text-foreground">
          {t("agentChat.messageList.restoring.title")}
        </p>
        <p className="text-sm text-muted-foreground">
          {t("agentChat.messageList.restoring.description")}
        </p>
      </div>
    </div>
  );
}

export function TaskCenterEmptyState() {
  const { t } = useTranslation("agent");

  return (
    <div className="flex min-h-[18rem] items-center justify-center px-6 py-8">
      <section
        data-testid="message-list-empty-task-center"
        className="w-full max-w-[560px] text-left"
      >
        <div className="flex items-start gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-[10px] border border-slate-200/80 bg-slate-50/80">
            <img
              src={LIME_BRAND_LOGO_SRC}
              alt={LIME_BRAND_NAME}
              className="h-5 w-5 opacity-80"
            />
          </div>
          <div className="min-w-0 flex-1">
            <div className="space-y-2">
              <div className="text-xs font-medium text-slate-500">
                {t("agentChat.messageList.taskCenterEmpty.badge")}
              </div>
              <h2 className="text-xl font-semibold text-slate-900">
                {t("agentChat.messageList.taskCenterEmpty.title")}
              </h2>
              <p className="max-w-[34rem] text-sm leading-6 text-slate-600">
                {t("agentChat.messageList.taskCenterEmpty.description")}
              </p>
              <p className="text-sm leading-6 text-slate-500">
                {t("agentChat.messageList.taskCenterEmpty.helper")}
              </p>
            </div>
          </div>
        </div>

        <div className="mt-5 grid gap-1.5 pl-12 text-xs text-slate-500">
          <span>{t("agentChat.messageList.taskCenterEmpty.chip.pending")}</span>
          <span>
            {t("agentChat.messageList.taskCenterEmpty.chip.organized")}
          </span>
          <span>{t("agentChat.messageList.taskCenterEmpty.chip.restore")}</span>
        </div>
      </section>
    </div>
  );
}
