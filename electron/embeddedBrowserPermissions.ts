import type { Session, WebContents } from "electron";

type HostEventEmitter = (event: string, payload?: unknown) => void;

export type EmbeddedBrowserPermissionDecision = "blocked" | "pending";

export interface EmbeddedBrowserPermissionRequestEvent {
  viewId: string;
  requestId: string;
  permission: string;
  url: string;
  requestingUrl: string | null;
  embeddingOrigin: string | null;
  decision: EmbeddedBrowserPermissionDecision;
}

export interface EmbeddedBrowserPermissionEntry {
  viewId: string;
  webContents: WebContents;
}

interface EmbeddedBrowserPermissionController {
  findEntryByWebContents(
    webContents: WebContents,
  ): EmbeddedBrowserPermissionEntry | null;
  deferPermissionRequest?: boolean;
  onPendingPermission?: (
    requestId: string,
    viewId: string,
    callback: (allowed: boolean) => void,
  ) => void;
}

export function installEmbeddedBrowserPermissionHandling(
  browserSession: Session,
  controller: EmbeddedBrowserPermissionController,
  emit: HostEventEmitter,
): void {
  browserSession.setPermissionRequestHandler(
    (webContents, permission, callback, details) => {
      const entry = controller.findEntryByWebContents(webContents);
      if (!entry) {
        callback(false);
        return;
      }
      const requestId = createPermissionRequestId();
      const deferred = controller.deferPermissionRequest === true;
      if (deferred) {
        controller.onPendingPermission?.(requestId, entry.viewId, callback);
      } else {
        callback(false);
      }
      emit("embedded-browser-view-permission-request", {
        viewId: entry.viewId,
        requestId,
        permission: normalizePermission(permission),
        url: normalizeUrl(webContents.getURL()) ?? "",
        requestingUrl: normalizeUrl(details.requestingUrl),
        embeddingOrigin: normalizeUrl(readEmbeddingOrigin(details)),
        decision: deferred ? "pending" : "blocked",
      } satisfies EmbeddedBrowserPermissionRequestEvent);
    },
  );
}

function createPermissionRequestId(): string {
  return `embedded-permission-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 10)}`;
}

function normalizePermission(value: string): string {
  return value.trim() || "unknown";
}

function readEmbeddingOrigin(details: object): string | undefined {
  if (!("embeddingOrigin" in details)) {
    return undefined;
  }
  const value = details.embeddingOrigin;
  return typeof value === "string" ? value : undefined;
}

function normalizeUrl(value: string | undefined): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
