import {
  app,
  session,
  WebContentsView,
  type BrowserWindow,
  type Rectangle,
} from "./electronRuntime";
import path from "node:path";
import { mkdirSync } from "node:fs";
import { installEmbeddedBrowserContextMenu } from "./embeddedBrowserContextMenu";
import { installEmbeddedBrowserDownloadHandling } from "./embeddedBrowserDownloads";
import { installEmbeddedBrowserPermissionHandling } from "./embeddedBrowserPermissions";
import {
  buildEmbeddedBrowserHtmlDataUrl,
  readEmbeddedBrowserHtmlPayload,
} from "./embeddedBrowserHtml";
import type {
  Session as ElectronSession,
  WebContents as ElectronWebContents,
} from "electron";

type HostArgs = Record<string, unknown> | null | undefined;
type HostEventEmitter = (event: string, payload?: unknown) => void;
type EmbeddedBrowserDestroyReason =
  | "renderer-cleanup"
  | "renderer-navigation"
  | "window-closed"
  | "webcontents-destroyed"
  | "host-dispose";

export const EMBEDDED_BROWSER_COMMANDS = [
  "embedded_browser_view_mount",
  "embedded_browser_view_set_bounds",
  "embedded_browser_view_navigate",
  "embedded_browser_view_load_html",
  "embedded_browser_view_reload",
  "embedded_browser_view_stop",
  "embedded_browser_view_find_in_page",
  "embedded_browser_view_stop_find_in_page",
  "embedded_browser_view_set_zoom",
  "embedded_browser_view_go_back",
  "embedded_browser_view_go_forward",
  "embedded_browser_view_destroy",
] as const;

export type EmbeddedBrowserCommand = (typeof EMBEDDED_BROWSER_COMMANDS)[number];

export interface EmbeddedBrowserViewState {
  viewId: string;
  url: string;
  title: string;
  faviconUrl: string | null;
  canGoBack: boolean;
  canGoForward: boolean;
  isLoading: boolean;
  loadProgress: number;
  zoomFactor: number;
  find: EmbeddedBrowserFindState;
}

export type EmbeddedBrowserLoadFailureCategory =
  | "dns"
  | "tls"
  | "blocked"
  | "aborted"
  | "load_failed";

export interface EmbeddedBrowserFindState {
  text: string;
  activeMatchOrdinal: number;
  matches: number;
  finalUpdate: boolean;
}

interface EmbeddedBrowserEntry {
  viewId: string;
  view: WebContentsView;
  window: BrowserWindow;
  closeListener: () => void;
  rendererNavigationListener: (
    event: unknown,
    url: string,
    isInPlace: boolean,
    isMainFrame: boolean,
  ) => void;
  webContentsDestroyedListener: () => void;
  webContentsGoneListener: () => void;
  leaseIds: Set<string>;
  pendingUrl?: string;
  sourceUri?: string;
  navigationToken: number;
  faviconUrl: string | null;
  loadProgress: number;
  find: EmbeddedBrowserFindState;
  findRequestId: number | null;
}

type PendingPermission = {
  viewId: string;
  callback: (allowed: boolean) => void;
};

export interface EmbeddedBrowserNativeView {
  state: EmbeddedBrowserViewState;
  view: WebContentsView;
  window: BrowserWindow;
}

const EMBEDDED_BROWSER_PARTITION = "persist:embedded-browser";
const EMBEDDED_BROWSER_ACCEPT_LANGUAGES = "zh-CN,zh,en-US,en";

export function isEmbeddedBrowserCommand(
  command: string,
): command is EmbeddedBrowserCommand {
  return EMBEDDED_BROWSER_COMMANDS.includes(command as EmbeddedBrowserCommand);
}

export class ElectronEmbeddedBrowserHost {
  #entries = new Map<string, EmbeddedBrowserEntry>();
  #downloadPaths = new Map<string, string>();
  #pendingPermissions = new Map<string, PendingPermission>();
  #deferPermissions = false;
  #emit: HostEventEmitter;
  #sessionHandlersInstalled = false;
  readonly #downloadDirectory: string;

  constructor(
    emit: HostEventEmitter = () => undefined,
    options: { deferPermissions?: boolean } = {},
  ) {
    this.#emit = emit;
    this.#deferPermissions = options.deferPermissions === true;
    const userDataRoot =
      typeof app.getPath === "function"
        ? app.getPath("userData")
        : process.env.TMPDIR || process.cwd();
    this.#downloadDirectory = path.join(userDataRoot, "browser-downloads");
    mkdirSync(this.#downloadDirectory, { recursive: true });
  }

  async invoke(
    window: BrowserWindow | null,
    command: EmbeddedBrowserCommand,
    args?: HostArgs,
  ): Promise<unknown> {
    if (command === "embedded_browser_view_destroy") {
      return this.#destroy(args, "renderer-cleanup");
    }

    if (!window || window.isDestroyed()) {
      throw new Error("当前窗口不可用，无法显示内嵌浏览器。");
    }

    switch (command) {
      case "embedded_browser_view_mount":
        return await this.#mount(window, args);
      case "embedded_browser_view_set_bounds":
        return this.#setBounds(window, args);
      case "embedded_browser_view_navigate":
        return await this.#navigate(window, args);
      case "embedded_browser_view_load_html":
        return await this.#loadHtml(window, args);
      case "embedded_browser_view_reload":
        return this.#reload(window, args);
      case "embedded_browser_view_stop":
        return this.#stop(window, args);
      case "embedded_browser_view_find_in_page":
        return this.#findInPage(window, args);
      case "embedded_browser_view_stop_find_in_page":
        return this.#stopFindInPage(window, args);
      case "embedded_browser_view_set_zoom":
        return this.#setZoom(window, args);
      case "embedded_browser_view_go_back":
        return this.#goBack(window, args);
      case "embedded_browser_view_go_forward":
        return this.#goForward(window, args);
      default:
        return {};
    }
  }

  dispose(): void {
    for (const viewId of [...this.#entries.keys()]) {
      this.#destroy({ viewId }, "host-dispose");
    }
  }

  resolveNativeView(viewId: string): EmbeddedBrowserNativeView | null {
    const entry = this.#entries.get(viewId);
    if (
      !entry ||
      entry.window.isDestroyed() ||
      entry.view.webContents.isDestroyed()
    ) {
      return null;
    }
    return {
      state: readState(entry),
      view: entry.view,
      window: entry.window,
    };
  }

  /** 仅供 Browser Host 以受控 downloadId 解析完成下载，不暴露给 Renderer。 */
  resolveDownloadPath(downloadId: string): string | null {
    return this.#downloadPaths.get(downloadId) ?? null;
  }

  resolvePermission(
    requestId: string,
    allowed: boolean,
    expectedViewId?: string,
  ): boolean {
    const pending = this.#pendingPermissions.get(requestId);
    if (!pending || (expectedViewId && pending.viewId !== expectedViewId)) {
      return false;
    }
    this.#pendingPermissions.delete(requestId);
    pending.callback(allowed);
    return true;
  }

  clearPendingPermission(requestId: string, expectedViewId?: string): boolean {
    const pending = this.#pendingPermissions.get(requestId);
    if (!pending || (expectedViewId && pending.viewId !== expectedViewId)) {
      return false;
    }
    this.#pendingPermissions.delete(requestId);
    pending.callback(false);
    return true;
  }

  clearPendingPermissions(viewId?: string): void {
    for (const [requestId, pending] of this.#pendingPermissions) {
      if (viewId && pending.viewId !== viewId) {
        continue;
      }
      this.#pendingPermissions.delete(requestId);
      pending.callback(false);
    }
  }

  pendingPermissionViewId(requestId: string): string | null {
    return this.#pendingPermissions.get(requestId)?.viewId ?? null;
  }

  async #mount(
    window: BrowserWindow,
    args?: HostArgs,
  ): Promise<EmbeddedBrowserViewState> {
    const viewId = readViewId(args);
    const url = readOptionalUrl(args);
    const bounds = readOptionalBounds(args);
    const visible = readOptionalBoolean(args, "visible") ?? true;
    const entry = this.#ensureEntry(window, viewId);
    const leaseId = readString(args, "leaseId");
    if (leaseId) {
      entry.leaseIds.add(leaseId);
    }

    if (bounds) {
      applyBounds(entry.view, bounds, visible);
    } else {
      entry.view.setVisible(visible);
    }

    if (url) {
      this.#startNavigation(entry, url);
    }

    return this.#emitState(entry);
  }

  #setBounds(window: BrowserWindow, args?: HostArgs): EmbeddedBrowserViewState {
    const entry = this.#ensureEntry(window, readViewId(args));
    const bounds = readBounds(args);
    const visible = readOptionalBoolean(args, "visible") ?? true;
    applyBounds(entry.view, bounds, visible);
    return this.#emitState(entry);
  }

  async #navigate(
    window: BrowserWindow,
    args?: HostArgs,
  ): Promise<EmbeddedBrowserViewState> {
    const entry = this.#ensureEntry(window, readViewId(args));
    this.#startNavigation(entry, readUrl(args));
    return this.#emitState(entry);
  }

  async #loadHtml(
    window: BrowserWindow,
    args?: HostArgs,
  ): Promise<EmbeddedBrowserViewState> {
    const entry = this.#ensureEntry(window, readViewId(args));
    const payload = readEmbeddedBrowserHtmlPayload(args);
    entry.sourceUri = payload.sourceUri;
    this.#startNavigation(
      entry,
      payload.sourceUri,
      buildEmbeddedBrowserHtmlDataUrl(payload),
    );
    return this.#emitState(entry);
  }

  #reload(window: BrowserWindow, args?: HostArgs): EmbeddedBrowserViewState {
    const entry = this.#ensureEntry(window, readViewId(args));
    entry.loadProgress = 0.1;
    entry.view.webContents.reload();
    return this.#emitState(entry);
  }

  #stop(window: BrowserWindow, args?: HostArgs): EmbeddedBrowserViewState {
    const entry = this.#ensureEntry(window, readViewId(args));
    entry.pendingUrl = undefined;
    if (entry.view.webContents.isLoading()) {
      entry.view.webContents.stop();
    }
    entry.loadProgress = 1;
    return this.#emitState(entry);
  }

  #findInPage(
    window: BrowserWindow,
    args?: HostArgs,
  ): EmbeddedBrowserViewState {
    const entry = this.#ensureEntry(window, readViewId(args));
    const text = readString(args, "text") ?? "";
    if (!text) {
      return this.#stopFindInPage(window, args);
    }

    entry.find = {
      text,
      activeMatchOrdinal: 0,
      matches: 0,
      finalUpdate: false,
    };
    entry.findRequestId = entry.view.webContents.findInPage(text, {
      forward: readOptionalBoolean(args, "forward") ?? true,
      findNext: readOptionalBoolean(args, "findNext") ?? false,
    });
    return this.#emitState(entry);
  }

  #stopFindInPage(
    window: BrowserWindow,
    args?: HostArgs,
  ): EmbeddedBrowserViewState {
    const entry = this.#ensureEntry(window, readViewId(args));
    entry.findRequestId = null;
    entry.find = createEmptyFindState();
    entry.view.webContents.stopFindInPage(readStopFindAction(args));
    return this.#emitState(entry);
  }

  #setZoom(window: BrowserWindow, args?: HostArgs): EmbeddedBrowserViewState {
    const entry = this.#ensureEntry(window, readViewId(args));
    const zoomFactor = readZoomFactor(args);
    entry.view.webContents.setZoomFactor(zoomFactor);
    return this.#emitState(entry);
  }

  #goBack(window: BrowserWindow, args?: HostArgs): EmbeddedBrowserViewState {
    const entry = this.#ensureEntry(window, readViewId(args));
    if (entry.view.webContents.navigationHistory.canGoBack()) {
      entry.view.webContents.navigationHistory.goBack();
    }
    return this.#emitState(entry);
  }

  #goForward(window: BrowserWindow, args?: HostArgs): EmbeddedBrowserViewState {
    const entry = this.#ensureEntry(window, readViewId(args));
    if (entry.view.webContents.navigationHistory.canGoForward()) {
      entry.view.webContents.navigationHistory.goForward();
    }
    return this.#emitState(entry);
  }

  #destroy(
    args: HostArgs,
    reason: EmbeddedBrowserDestroyReason,
  ): Record<string, never> {
    const viewId = readViewId(args);
    const entry = this.#entries.get(viewId);
    if (!entry) {
      return {};
    }
    const leaseId = readString(args, "leaseId");
    if (leaseId) {
      if (!entry.leaseIds.delete(leaseId) || entry.leaseIds.size > 0) {
        return {};
      }
    }

    this.#entries.delete(viewId);
    for (const [requestId, pending] of this.#pendingPermissions) {
      if (pending.viewId === viewId) {
        pending.callback(false);
        this.#pendingPermissions.delete(requestId);
      }
    }
    detachEntryFromWindow(entry);
    closeEntryView(entry);
    this.#emit("embedded-browser-view-destroyed", {
      viewId,
      reason,
      requestedLeaseId: leaseId,
      remainingLeaseCount: entry.leaseIds.size,
    });
    return {};
  }

  #ensureEntry(window: BrowserWindow, viewId: string): EmbeddedBrowserEntry {
    const existing = this.#entries.get(viewId);
    if (existing) {
      if (existing.window !== window) {
        detachEntryFromWindow(existing);
        existing.window = window;
        existing.closeListener = () =>
          this.#destroy({ viewId }, "window-closed");
        existing.rendererNavigationListener =
          this.#createRendererNavigationListener(viewId);
        attachEntryToWindow(existing, window);
        return existing;
      }
      window.contentView.addChildView(existing.view);
      return existing;
    }

    const { embeddedSession, userAgent: embeddedBrowserUserAgent } =
      configureEmbeddedBrowserSession();
    this.#installSessionHandlers(embeddedSession);
    const view = new WebContentsView({
      webPreferences: {
        contextIsolation: true,
        javascript: true,
        nodeIntegration: false,
        partition: EMBEDDED_BROWSER_PARTITION,
        sandbox: true,
        webSecurity: true,
      },
    });
    view.setBackgroundColor("#ffffff");
    view.webContents.setUserAgent(embeddedBrowserUserAgent);
    const entry: EmbeddedBrowserEntry = {
      viewId,
      view,
      window,
      closeListener: () => this.#destroy({ viewId }, "window-closed"),
      rendererNavigationListener:
        this.#createRendererNavigationListener(viewId),
      webContentsDestroyedListener: () =>
        this.#destroy({ viewId }, "webcontents-destroyed"),
      webContentsGoneListener: () =>
        this.#destroy({ viewId }, "webcontents-destroyed"),
      leaseIds: new Set<string>(),
      navigationToken: 0,
      faviconUrl: null,
      loadProgress: 1,
      find: createEmptyFindState(),
      findRequestId: null,
    };
    view.webContents.setWindowOpenHandler(({ url }) => {
      if (entry.sourceUri) {
        return { action: "deny" };
      }
      const normalizedUrl = normalizeHttpUrl(url);
      if (normalizedUrl) {
        this.#startNavigation(entry, normalizedUrl);
        this.#emitState(entry);
      }
      return { action: "deny" };
    });
    view.webContents.on("will-navigate", (event, url) => {
      if (entry.sourceUri && normalizeHttpUrl(url)) {
        event.preventDefault();
      }
    });
    installEmbeddedBrowserContextMenu({
      view,
      window,
      navigate: (url) => this.#startNavigation(entry, url),
      emitState: () => {
        this.#emitState(entry);
      },
    });
    attachEntryToWindow(entry, window);
    this.#entries.set(viewId, entry);
    view.webContents.on("did-navigate", (_event, url) => {
      this.#clearPendingNavigation(entry, url);
      entry.loadProgress = Math.max(entry.loadProgress, 0.7);
      this.#emitState(entry);
    });
    view.webContents.on("did-navigate-in-page", (_event, url) => {
      this.#clearPendingNavigation(entry, url);
      this.#emitState(entry);
    });
    view.webContents.on("did-start-loading", () => {
      entry.loadProgress = 0.35;
      this.#emitState(entry);
    });
    view.webContents.on("did-stop-loading", () => {
      this.#clearPendingNavigation(entry, entry.view.webContents.getURL());
      entry.loadProgress = 1;
      this.#emitState(entry);
    });
    view.webContents.on("page-title-updated", () => this.#emitState(entry));
    view.webContents.on("page-favicon-updated", (_event, favicons) => {
      entry.faviconUrl = Array.isArray(favicons)
        ? (favicons.find((item) => typeof item === "string" && item.trim()) ??
          null)
        : null;
      this.#emitState(entry);
    });
    view.webContents.on("found-in-page", (_event, result) => {
      const nextFind = readFindResult(result, entry.find.text);
      if (entry.findRequestId !== null && nextFind.requestId !== null) {
        if (nextFind.requestId !== entry.findRequestId) {
          return;
        }
      }
      entry.find = {
        text: nextFind.text,
        activeMatchOrdinal: nextFind.activeMatchOrdinal,
        matches: nextFind.matches,
        finalUpdate: nextFind.finalUpdate,
      };
      this.#emitState(entry);
    });
    view.webContents.on(
      "did-fail-load",
      (_event, errorCode, errorDescription, validatedUrl) => {
        if (!this.#isCurrentNavigationEvent(entry, validatedUrl)) {
          return;
        }
        this.#clearPendingNavigation(entry, validatedUrl);
        entry.loadProgress = 1;
        this.#emitLoadFailed(
          entry,
          errorCode,
          errorDescription,
          typeof validatedUrl === "string" ? validatedUrl : undefined,
        );
      },
    );
    return entry;
  }

  #createRendererNavigationListener(
    viewId: string,
  ): EmbeddedBrowserEntry["rendererNavigationListener"] {
    return (_event, _url, isInPlace, isMainFrame) => {
      if (isMainFrame && !isInPlace) {
        this.#destroy({ viewId }, "renderer-navigation");
      }
    };
  }

  #installSessionHandlers(embeddedSession: ElectronSession): void {
    if (this.#sessionHandlersInstalled) {
      return;
    }
    this.#sessionHandlersInstalled = true;
    const controller = {
      findEntryByWebContents: (webContents: ElectronWebContents) =>
        this.#findEntryByWebContents(webContents),
      onCompletedDownload: (downloadId: string, savePath: string) => {
        this.#downloadPaths.set(downloadId, savePath);
      },
      resolveSavePath: (filename: string) =>
        path.join(this.#downloadDirectory, sanitizeDownloadFilename(filename)),
      deferPermissionRequest: this.#deferPermissions,
      onPendingPermission: (
        requestId: string,
        viewId: string,
        callback: (allowed: boolean) => void,
      ) => {
        this.#pendingPermissions.set(requestId, {
          viewId,
          callback,
        });
      },
    };
    installEmbeddedBrowserDownloadHandling(
      embeddedSession,
      controller,
      this.#emit,
    );
    installEmbeddedBrowserPermissionHandling(
      embeddedSession,
      controller,
      this.#emit,
    );
  }

  #findEntryByWebContents(webContents: ElectronWebContents) {
    for (const entry of this.#entries.values()) {
      if (entry.view.webContents === webContents) {
        return {
          viewId: entry.viewId,
          webContents: entry.view.webContents,
        };
      }
    }
    return null;
  }

  #startNavigation(
    entry: EmbeddedBrowserEntry,
    url: string,
    loadUrl = url,
  ): void {
    if (loadUrl === url) {
      entry.sourceUri = undefined;
    }
    entry.pendingUrl = url;
    entry.loadProgress = 0.1;
    const navigationToken = entry.navigationToken + 1;
    entry.navigationToken = navigationToken;

    let loadPromise: Promise<void>;
    try {
      loadPromise = entry.view.webContents.loadURL(loadUrl);
    } catch (error) {
      this.#clearPendingNavigation(entry);
      this.#emitLoadFailed(
        entry,
        null,
        error instanceof Error ? error.message : String(error),
        url,
      );
      return;
    }

    void loadPromise
      .then(() => {
        if (
          this.#entries.get(entry.viewId) !== entry ||
          entry.navigationToken !== navigationToken ||
          entry.pendingUrl !== url
        ) {
          return;
        }
        this.#clearPendingNavigation(entry);
        entry.loadProgress = 1;
        this.#emitState(entry);
      })
      .catch((error) => {
        if (
          this.#entries.get(entry.viewId) !== entry ||
          entry.navigationToken !== navigationToken ||
          entry.pendingUrl !== url
        ) {
          return;
        }
        this.#clearPendingNavigation(entry);
        entry.loadProgress = 1;
        this.#emitLoadFailed(
          entry,
          null,
          error instanceof Error ? error.message : String(error),
          url,
        );
      });
  }

  #clearPendingNavigation(entry: EmbeddedBrowserEntry, url?: unknown): void {
    if (!this.#isCurrentNavigationEvent(entry, url)) {
      return;
    }
    entry.pendingUrl = undefined;
  }

  #isCurrentNavigationEvent(
    entry: EmbeddedBrowserEntry,
    url?: unknown,
  ): boolean {
    return !(
      typeof url === "string" &&
      entry.pendingUrl &&
      url !== entry.pendingUrl
    );
  }

  #emitLoadFailed(
    entry: EmbeddedBrowserEntry,
    errorCode: number | null,
    errorDescription: string,
    url?: string,
  ): void {
    this.#emit("embedded-browser-view-load-failed", {
      ...readState(entry),
      ...(url ? { url } : {}),
      errorCode,
      errorDescription,
      failureCategory: classifyLoadFailure(errorCode, errorDescription),
    });
  }

  #emitState(entry: EmbeddedBrowserEntry): EmbeddedBrowserViewState {
    const state = readState(entry);
    this.#emit("embedded-browser-view-state", state);
    return state;
  }
}

function configureEmbeddedBrowserSession(): {
  embeddedSession: ElectronSession;
  userAgent: string;
} {
  const embeddedSession = session.fromPartition(EMBEDDED_BROWSER_PARTITION, {
    cache: true,
  });
  const userAgent = normalizeEmbeddedBrowserUserAgent(
    embeddedSession.getUserAgent(),
  );
  embeddedSession.setUserAgent(userAgent, EMBEDDED_BROWSER_ACCEPT_LANGUAGES);
  return { embeddedSession, userAgent };
}

function normalizeEmbeddedBrowserUserAgent(value: string): string {
  const normalized = value
    .replace(/\sElectron\/[^\s]+/gi, "")
    .replace(/\sLime\/[^\s]+/gi, "")
    .replace(/\s+/g, " ")
    .trim();
  return normalized || value;
}

function readState(entry: EmbeddedBrowserEntry): EmbeddedBrowserViewState {
  const webContents = entry.view.webContents;
  return {
    viewId: entry.viewId,
    url: entry.pendingUrl || entry.sourceUri || webContents.getURL(),
    title: webContents.getTitle(),
    faviconUrl: entry.faviconUrl,
    canGoBack: webContents.navigationHistory.canGoBack(),
    canGoForward: webContents.navigationHistory.canGoForward(),
    isLoading: webContents.isLoading() || Boolean(entry.pendingUrl),
    loadProgress: normalizeLoadProgress(entry),
    zoomFactor: normalizeZoomFactor(webContents.getZoomFactor()),
    find: entry.find,
  };
}

function createEmptyFindState(): EmbeddedBrowserFindState {
  return {
    text: "",
    activeMatchOrdinal: 0,
    matches: 0,
    finalUpdate: true,
  };
}

function readFindResult(
  value: unknown,
  fallbackText: string,
): EmbeddedBrowserFindState & { requestId: number | null } {
  const record = readRecord(value);
  return {
    text: fallbackText,
    requestId: readInteger(record?.requestId),
    activeMatchOrdinal: Math.max(
      0,
      readInteger(record?.activeMatchOrdinal) ?? 0,
    ),
    matches: Math.max(0, readInteger(record?.matches) ?? 0),
    finalUpdate:
      typeof record?.finalUpdate === "boolean" ? record.finalUpdate : false,
  };
}

function normalizeLoadProgress(entry: EmbeddedBrowserEntry): number {
  const loading =
    entry.view.webContents.isLoading() || Boolean(entry.pendingUrl);
  if (!loading) {
    return 1;
  }
  return clampNumber(entry.loadProgress, 0.05, 0.95);
}

function normalizeZoomFactor(value: number): number {
  return Math.round(clampNumber(value, 0.5, 3) * 100) / 100;
}

function readViewId(args?: HostArgs): string {
  const value = readString(args, "viewId");
  if (!value) {
    throw new Error("embedded browser viewId 不能为空。");
  }
  return value;
}

function readUrl(args?: HostArgs): string {
  const value = readOptionalUrl(args);
  if (!value) {
    throw new Error("embedded browser url 不能为空。");
  }
  return value;
}

function readOptionalUrl(args?: HostArgs): string | null {
  const value = readString(args, "url");
  if (!value) {
    return null;
  }
  const normalized = normalizeHttpUrl(value);
  if (!normalized) {
    throw new Error("内嵌浏览器只支持 http/https 地址。");
  }
  return normalized;
}

function readBounds(args?: HostArgs): Rectangle {
  const bounds = readOptionalBounds(args);
  if (!bounds) {
    throw new Error("embedded browser bounds 不能为空。");
  }
  return bounds;
}

function readOptionalBounds(args?: HostArgs): Rectangle | null {
  const source = readRecord(args)?.bounds;
  if (!source || typeof source !== "object" || Array.isArray(source)) {
    return null;
  }
  const record = source as Record<string, unknown>;
  const x = readInteger(record.x);
  const y = readInteger(record.y);
  const width = readInteger(record.width);
  const height = readInteger(record.height);
  if (x === null || y === null || width === null || height === null) {
    throw new Error("embedded browser bounds 必须包含整数 x/y/width/height。");
  }
  return {
    x: Math.max(0, x),
    y: Math.max(0, y),
    width: Math.max(0, width),
    height: Math.max(0, height),
  };
}

function applyBounds(
  view: WebContentsView,
  bounds: Rectangle,
  visible: boolean,
): void {
  view.setBounds(bounds);
  view.setVisible(visible && bounds.width > 0 && bounds.height > 0);
}

function attachEntryToWindow(
  entry: EmbeddedBrowserEntry,
  window: BrowserWindow,
): void {
  window.on("closed", entry.closeListener);
  window.webContents.on(
    "did-start-navigation",
    entry.rendererNavigationListener,
  );
  entry.view.webContents.on("destroyed", entry.webContentsDestroyedListener);
  entry.view.webContents.on(
    "render-process-gone",
    entry.webContentsGoneListener,
  );
  window.contentView.addChildView(entry.view);
}

function detachEntryFromWindow(entry: EmbeddedBrowserEntry): void {
  try {
    if (!entry.window.isDestroyed()) {
      entry.window.off("closed", entry.closeListener);
      entry.window.webContents.off(
        "did-start-navigation",
        entry.rendererNavigationListener,
      );
      entry.window.contentView.removeChildView(entry.view);
    }
    entry.view.webContents.off("destroyed", entry.webContentsDestroyedListener);
    entry.view.webContents.off(
      "render-process-gone",
      entry.webContentsGoneListener,
    );
  } catch (error) {
    ignoreElectronDestroyedObject(error);
  }
}

function closeEntryView(entry: EmbeddedBrowserEntry): void {
  try {
    if (!entry.view.webContents.isDestroyed()) {
      entry.view.webContents.close();
    }
  } catch (error) {
    ignoreElectronDestroyedObject(error);
  }
}

function ignoreElectronDestroyedObject(error: unknown): void {
  if (
    error instanceof Error &&
    /Object has been destroyed/i.test(error.message)
  ) {
    return;
  }
  throw error;
}

function readOptionalBoolean(args: HostArgs, key: string): boolean | null {
  const value = readRecord(args)?.[key];
  return typeof value === "boolean" ? value : null;
}

function readString(args: HostArgs, key: string): string | null {
  const value = readRecord(args)?.[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readRecord(args: unknown): Record<string, unknown> | null {
  return args && typeof args === "object" && !Array.isArray(args)
    ? (args as Record<string, unknown>)
    : null;
}

function readInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.round(value)
    : null;
}

function readNumber(args: HostArgs, key: string): number | null {
  const value = readRecord(args)?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readZoomFactor(args: HostArgs): number {
  const value = readNumber(args, "zoomFactor");
  if (value === null) {
    throw new Error("embedded browser zoomFactor 必须是有效数字。");
  }
  return normalizeZoomFactor(value);
}

function readStopFindAction(
  args: HostArgs,
): "clearSelection" | "keepSelection" | "activateSelection" {
  const value = readString(args, "action");
  if (
    value === "clearSelection" ||
    value === "keepSelection" ||
    value === "activateSelection"
  ) {
    return value;
  }
  return "clearSelection";
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function normalizeHttpUrl(value: string): string | null {
  try {
    const url = new URL(value.trim());
    return url.protocol === "http:" || url.protocol === "https:"
      ? url.href
      : null;
  } catch {
    return null;
  }
}

function sanitizeDownloadFilename(value: string): string {
  const normalized = value.trim().replace(/[^a-zA-Z0-9._-]+/g, "_");
  return normalized || "download";
}

function classifyLoadFailure(
  errorCode: number | null,
  errorDescription: string,
): EmbeddedBrowserLoadFailureCategory {
  const description = errorDescription.toUpperCase();
  if (errorCode === -3 || description.includes("ABORTED")) {
    return "aborted";
  }
  if (
    errorCode === -105 ||
    errorCode === -137 ||
    description.includes("NAME_NOT_RESOLVED") ||
    description.includes("DNS") ||
    description.includes("HOST_RESOLVER")
  ) {
    return "dns";
  }
  if (
    (typeof errorCode === "number" && errorCode <= -200 && errorCode >= -299) ||
    description.includes("CERT") ||
    description.includes("SSL") ||
    description.includes("TLS")
  ) {
    return "tls";
  }
  if (
    errorCode === -20 ||
    errorCode === -27 ||
    errorCode === -30 ||
    description.includes("BLOCKED") ||
    description.includes("CSP") ||
    description.includes("POLICY")
  ) {
    return "blocked";
  }
  return "load_failed";
}
