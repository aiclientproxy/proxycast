import { Download, Globe2, ShieldAlert } from "lucide-react";
import type {
  BrowserTabDownloadEvent,
  BrowserTabPermissionRequestEvent,
} from "@/lib/api/browserTab";

type Translate = (key: string, options?: Record<string, unknown>) => string;

export interface BrowserWorkspaceError {
  body: string;
  source: "host" | "load";
  title: string;
}

export function BrowserWorkspaceHostUnavailable({ t }: { t: Translate }) {
  return (
    <div
      className="absolute inset-0 flex items-center justify-center p-6"
      role="alert"
      data-testid="browser-workspace-host-unavailable"
      data-browser-workspace-status="host-unavailable"
    >
      <div className="max-w-[380px] text-center">
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-md border border-amber-200 bg-amber-50 text-amber-700">
          <Globe2 className="h-5 w-5" />
        </div>
        <div className="text-sm font-semibold text-[color:var(--lime-text-strong)]">
          {t("agentChat.browserWorkspace.hostUnavailableTitle")}
        </div>
        <div className="mt-1 text-xs leading-5 text-[color:var(--lime-text-muted)]">
          {t("agentChat.browserWorkspace.hostUnavailableBody")}
        </div>
      </div>
    </div>
  );
}

export function BrowserWorkspaceLoading({ t }: { t: Translate }) {
  return (
    <div
      className="pointer-events-none absolute inset-0 flex items-center justify-center p-6"
      aria-live="polite"
      role="status"
      data-testid="browser-workspace-loading"
      data-browser-workspace-status="loading"
    >
      <div className="text-center text-xs text-[color:var(--lime-text-muted)]">
        <Globe2 className="mx-auto mb-2 h-5 w-5" />
        {t("agentChat.browserWorkspace.loading")}
      </div>
    </div>
  );
}

export function BrowserWorkspaceErrorBanner({
  error,
}: {
  error: BrowserWorkspaceError;
}) {
  return (
    <div
      className="shrink-0 border-b border-rose-200 bg-rose-50 px-3 py-2 text-xs leading-5 text-rose-900"
      role="alert"
      data-testid="browser-workspace-error"
      data-browser-workspace-status={`${error.source}-error`}
      data-browser-error-source={error.source}
    >
      <div className="font-medium">{error.title}</div>
      <div className="text-rose-800/90">{error.body}</div>
    </div>
  );
}

export function BrowserWorkspacePermissionBanner({
  permission,
  t,
}: {
  permission: BrowserTabPermissionRequestEvent;
  t: Translate;
}) {
  const source =
    permission.requestingUrl || permission.embeddingOrigin || permission.url;
  return (
    <div
      className="shrink-0 border-b border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-950"
      role="alert"
      data-testid="browser-workspace-permission"
      data-browser-workspace-status={`permission-${permission.decision}`}
    >
      <div className="flex items-start gap-2">
        <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-amber-700" />
        <div className="min-w-0">
          <div className="font-medium">
            {t(
              permission.decision === "pending"
                ? "agentChat.browserWorkspace.permissionPendingTitle"
                : "agentChat.browserWorkspace.permissionBlockedTitle",
              {
              permission: permission.permission,
              },
            )}
          </div>
          <div className="break-words text-amber-900/85">
            {t(
              permission.decision === "pending"
                ? "agentChat.browserWorkspace.permissionPendingBody"
                : "agentChat.browserWorkspace.permissionBlockedBody",
              { source },
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function BrowserWorkspaceDownloadShelf({
  download,
  t,
}: {
  download: BrowserTabDownloadEvent;
  t: Translate;
}) {
  const percent = download.totalBytes
    ? Math.max(
        0,
        Math.min(
          100,
          Math.round((download.receivedBytes / download.totalBytes) * 100),
        ),
      )
    : 0;
  const terminalKey =
    download.state === "completed"
      ? "downloadComplete"
      : download.state === "cancelled"
        ? "downloadCancelled"
        : download.state === "interrupted"
          ? "downloadInterrupted"
          : "downloadProgress";
  return (
    <div
      className="pointer-events-none shrink-0 border-b border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] px-3 py-2 text-xs text-[color:var(--lime-text)]"
      aria-live="polite"
      role="status"
      data-testid="browser-workspace-download"
      data-browser-workspace-status={`download-${download.state}`}
    >
      <div className="flex min-w-0 items-center gap-2">
        <Download className="h-4 w-4 shrink-0 text-[color:var(--lime-text-muted)]" />
        <span className="truncate font-medium">
          {t(`agentChat.browserWorkspace.${terminalKey}`, {
            filename: download.filename,
            percent,
          })}
        </span>
      </div>
      {download.state === "started" || download.state === "progressing" ? (
        <div className="mt-2 h-1 overflow-hidden rounded-full bg-[color:var(--lime-surface-muted)]">
          <div
            className="h-full bg-emerald-500 transition-[width] duration-150"
            style={{
              width: `${Math.max(download.totalBytes ? 6 : 14, percent)}%`,
            }}
          />
        </div>
      ) : null}
    </div>
  );
}
