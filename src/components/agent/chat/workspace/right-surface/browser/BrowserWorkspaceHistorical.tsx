import { History, Lock } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { BrowserTabHistoricalProjection } from "@/lib/api/browserTab";

export function BrowserWorkspaceHistorical({
  projection,
}: {
  projection: BrowserTabHistoricalProjection;
}) {
  const { t } = useTranslation("agent");
  return (
    <section
      className="flex h-full min-h-0 flex-col overflow-hidden bg-[color:var(--lime-surface)]"
      data-testid="browser-workspace-historical"
      data-browser-historical="true"
      data-browser-session-id={projection.browserSessionId}
      data-browser-tab-id={projection.tabId}
      data-browser-thread-id={projection.threadId}
      data-browser-page-revision={projection.pageRevision}
      data-browser-active-turn-id=""
      data-browser-control-owner="released"
      data-browser-web-contents-id=""
    >
      <div className="flex h-9 shrink-0 items-center gap-2 border-b border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface-muted)] px-3 text-xs text-[color:var(--lime-text-muted)]">
        <History className="h-3.5 w-3.5 shrink-0" />
        <span className="truncate font-medium text-[color:var(--lime-text-strong)]">
          {projection.title}
        </span>
        <span className="ml-auto shrink-0">
          {t("agentChat.browserWorkspace.historicalRevision", {
            revision: projection.pageRevision,
          })}
        </span>
      </div>
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-[color:var(--lime-surface-border)] px-3">
        <Lock className="h-3.5 w-3.5 shrink-0 text-[color:var(--lime-text-muted)]" />
        <div className="min-w-0 flex-1 truncate text-xs text-[color:var(--lime-text-muted)]">
          {projection.url}
        </div>
        <span className="shrink-0 rounded-md border border-[color:var(--lime-surface-border)] px-2 py-1 text-[11px] text-[color:var(--lime-text-muted)]">
          {t("agentChat.browserWorkspace.historicalBadge")}
        </span>
      </div>
      <div
        className="flex min-h-0 flex-1 items-center justify-center bg-[color:var(--lime-surface-muted)] p-6"
        data-browser-workspace-status="historical"
        role="status"
      >
        <div className="max-w-[420px] text-center text-xs leading-5 text-[color:var(--lime-text-muted)]">
          <History className="mx-auto mb-2 h-5 w-5" />
          <div className="font-medium text-[color:var(--lime-text-strong)]">
            {t("agentChat.browserWorkspace.historicalTitle")}
          </div>
          <div className="mt-1">
            {t("agentChat.browserWorkspace.historicalBody")}
          </div>
          {projection.mark ? (
            <div className="mt-2 text-[11px]">
              {t("agentChat.browserWorkspace.historicalMark", {
                mark: projection.mark,
              })}
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}
