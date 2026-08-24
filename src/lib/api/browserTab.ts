import { safeInvoke, safeListen } from "@/lib/dev-bridge";
import {
  getElectronHostBridge,
  isElectronDevBridgeFallbackAvailable,
  isElectronHostCommandAvailable,
} from "@/lib/electron-host";
import type {
  EmbeddedBrowserBounds,
  EmbeddedBrowserDownloadEvent,
  EmbeddedBrowserPermissionRequestEvent,
  EmbeddedBrowserViewLoadFailedEvent,
  EmbeddedBrowserViewState,
} from "./embeddedBrowser";

export type BrowserTabOrigin = "agent" | "user";
export type BrowserTabControlOwner =
  | "agent"
  | "human_takeover"
  | "released"
  | "user";
export type BrowserTabMark = "deliverable" | "handoff";

export interface BrowserTabState extends EmbeddedBrowserViewState {
  activeTurnId: string | null;
  browserSessionId: string;
  controlOwner: BrowserTabControlOwner;
  humanReason: string | null;
  mark: BrowserTabMark | null;
  origin: BrowserTabOrigin;
  ownerWebContentsId: number;
  pageRevision: number;
  selected: boolean;
  tabId: string;
  threadId: string;
  webContentsId: number;
  windowId: number;
}

/**
 * Historical Browser projection is deliberately not a BrowserTabState. It is
 * a read-only replay fact and must never be passed to the Electron host.
 */
export interface BrowserTabHistoricalProjection {
  browserSessionId: string;
  tabId: string;
  threadId: string;
  url: string;
  title: string;
  pageRevision: number;
  mark: BrowserTabMark | null;
  origin: BrowserTabOrigin;
  selected: boolean;
  snapshotId: string | null;
  replayedAt: string | null;
  readOnly: true;
}

export interface BrowserTabHistoricalSource {
  mode: "historical" | "replay";
  browserSessionId: string;
  tabId: string;
  threadId: string;
  url: string;
  title?: string | null;
  pageRevision?: number | null;
  mark?: BrowserTabMark | null;
  origin?: BrowserTabOrigin | null;
  selected?: boolean | null;
  snapshotId?: string | null;
  replayedAt?: string | null;
}

export function createBrowserTabHistoricalProjection(
  source: BrowserTabHistoricalSource,
): BrowserTabHistoricalProjection | null {
  const browserSessionId = source.browserSessionId.trim();
  const tabId = source.tabId.trim();
  const threadId = source.threadId.trim();
  const url = source.url.trim();
  if (!browserSessionId || !tabId || !threadId || !url) {
    return null;
  }
  const pageRevision = source.pageRevision ?? 0;
  if (!Number.isInteger(pageRevision) || pageRevision < 0) {
    return null;
  }
  return {
    browserSessionId,
    tabId,
    threadId,
    url,
    title: source.title?.trim() || url,
    pageRevision,
    mark:
      source.mark === "deliverable" || source.mark === "handoff"
        ? source.mark
        : null,
    origin: source.origin === "user" ? "user" : "agent",
    selected: source.selected !== false,
    snapshotId: source.snapshotId?.trim() || null,
    replayedAt: source.replayedAt?.trim() || null,
    readOnly: true,
  };
}

export function readBrowserTabHistoricalProjection(
  value: unknown,
): BrowserTabHistoricalProjection | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }
  const mode = record.mode;
  if (mode !== "historical" && mode !== "replay") {
    return null;
  }
  const source = {
    mode,
    browserSessionId: record.browserSessionId,
    tabId: record.tabId,
    threadId: record.threadId,
    url: record.url,
    title: record.title,
    pageRevision: record.pageRevision,
    mark: record.mark,
    origin: record.origin,
    selected: record.selected,
    snapshotId: record.snapshotId,
    replayedAt: record.replayedAt,
  };
  if (
    typeof source.browserSessionId !== "string" ||
    typeof source.tabId !== "string" ||
    typeof source.threadId !== "string" ||
    typeof source.url !== "string"
  ) {
    return null;
  }
  return createBrowserTabHistoricalProjection(
    source as BrowserTabHistoricalSource,
  );
}

export interface BrowserTabMountParams extends Record<string, unknown> {
  browserSessionId: string;
  bounds: EmbeddedBrowserBounds;
  selected: boolean;
  tabId: string;
  threadId: string;
  url: string;
  visible: boolean;
}

export interface BrowserTabEventIdentity {
  browserSessionId: string;
  ownerWebContentsId: number;
  tabId: string;
  threadId: string;
  webContentsId: number | null;
  windowId: number;
}

export type BrowserTabLoadFailedEvent = EmbeddedBrowserViewLoadFailedEvent &
  BrowserTabEventIdentity;
export type BrowserTabDownloadEvent = EmbeddedBrowserDownloadEvent &
  BrowserTabEventIdentity & {
    artifactRef?: string;
    artifactFilename?: string;
    artifactMimeType?: string | null;
    artifactCreatedAt?: string;
    artifactPersistedAt?: string;
    artifactSidecarPath?: string;
    artifactContentStatus?: string;
    artifactStatus?: "failed";
    artifactError?: string;
  };
export type BrowserTabPermissionRequestEvent =
  EmbeddedBrowserPermissionRequestEvent & BrowserTabEventIdentity;

export interface BrowserTabClosedEvent {
  browserSessionId: string;
  reason: string;
  tabId: string;
  threadId: string;
  viewId: string;
}

const BROWSER_TAB_REQUIRED_COMMANDS = [
  "browser_tab_mount",
  "browser_tab_set_bounds",
  "browser_tab_navigate",
  "browser_tab_reload",
  "browser_tab_stop",
  "browser_tab_find_in_page",
  "browser_tab_stop_find_in_page",
  "browser_tab_set_zoom",
  "browser_tab_go_back",
  "browser_tab_go_forward",
  "browser_tab_select",
  "browser_tab_close",
] as const;

export function isBrowserTabHostAvailable(): boolean {
  return (
    Boolean(getElectronHostBridge()) &&
    !isElectronDevBridgeFallbackAvailable() &&
    BROWSER_TAB_REQUIRED_COMMANDS.every((command) =>
      isElectronHostCommandAvailable(command),
    )
  );
}

async function invokeBrowserTab(
  command: string,
  params: Record<string, unknown>,
): Promise<BrowserTabState> {
  const result = await safeInvoke<unknown>(command, params);
  assertBrowserTabState(result);
  return result;
}

export async function mountBrowserTab(
  params: BrowserTabMountParams,
): Promise<BrowserTabState> {
  return await invokeBrowserTab("browser_tab_mount", params);
}

export async function setBrowserTabBounds(params: {
  tabId: string;
  bounds: EmbeddedBrowserBounds;
  visible: boolean;
}): Promise<BrowserTabState> {
  return await invokeBrowserTab("browser_tab_set_bounds", params);
}

export async function navigateBrowserTab(
  tabId: string,
  url: string,
): Promise<BrowserTabState> {
  return await invokeBrowserTab("browser_tab_navigate", { tabId, url });
}

export async function reloadBrowserTab(
  tabId: string,
): Promise<BrowserTabState> {
  return await invokeBrowserTab("browser_tab_reload", { tabId });
}

export async function stopBrowserTab(tabId: string): Promise<BrowserTabState> {
  return await invokeBrowserTab("browser_tab_stop", { tabId });
}

export async function findInBrowserTab(params: {
  tabId: string;
  text: string;
  forward?: boolean;
  findNext?: boolean;
}): Promise<BrowserTabState> {
  return await invokeBrowserTab("browser_tab_find_in_page", params);
}

export async function stopFindInBrowserTab(
  tabId: string,
): Promise<BrowserTabState> {
  return await invokeBrowserTab("browser_tab_stop_find_in_page", { tabId });
}

export async function setBrowserTabZoom(
  tabId: string,
  zoomFactor: number,
): Promise<BrowserTabState> {
  return await invokeBrowserTab("browser_tab_set_zoom", {
    tabId,
    zoomFactor,
  });
}

export async function goBackBrowserTab(
  tabId: string,
): Promise<BrowserTabState> {
  return await invokeBrowserTab("browser_tab_go_back", { tabId });
}

export async function goForwardBrowserTab(
  tabId: string,
): Promise<BrowserTabState> {
  return await invokeBrowserTab("browser_tab_go_forward", { tabId });
}

export async function selectBrowserTab(params: {
  tabId: string;
  bounds: EmbeddedBrowserBounds;
}): Promise<BrowserTabState> {
  return await invokeBrowserTab("browser_tab_select", params);
}

export async function closeBrowserTab(tabId: string): Promise<void> {
  await safeInvoke("browser_tab_close", { tabId });
}

export function listenBrowserTabState(
  handler: (state: BrowserTabState) => void,
): Promise<() => void> {
  return listenValidated("browser-tab-state", assertBrowserTabState, handler);
}

export function listenBrowserTabClosed(
  handler: (event: BrowserTabClosedEvent) => void,
): Promise<() => void> {
  return listenValidated(
    "browser-tab-closed",
    assertBrowserTabClosedEvent,
    handler,
  );
}

export function listenBrowserTabLoadFailed(
  handler: (event: BrowserTabLoadFailedEvent) => void,
): Promise<() => void> {
  return listenValidated(
    "browser-tab-load-failed",
    assertBrowserTabLoadFailedEvent,
    handler,
  );
}

export function listenBrowserTabDownload(
  handler: (event: BrowserTabDownloadEvent) => void,
): Promise<() => void> {
  return listenValidated(
    "browser-tab-download",
    assertBrowserTabDownloadEvent,
    handler,
  );
}

export function listenBrowserTabPermissionRequest(
  handler: (event: BrowserTabPermissionRequestEvent) => void,
): Promise<() => void> {
  return listenValidated(
    "browser-tab-permission-request",
    assertBrowserTabPermissionRequestEvent,
    handler,
  );
}

function listenValidated<T>(
  event: string,
  assertValue: (value: unknown) => asserts value is T,
  handler: (value: T) => void,
): Promise<() => void> {
  return safeListen<T>(event, (message) => {
    assertValue(message.payload);
    handler(message.payload);
  });
}

function assertBrowserTabState(
  value: unknown,
): asserts value is BrowserTabState {
  const record = asRecord(value);
  if (
    !record ||
    typeof record.viewId !== "string" ||
    typeof record.url !== "string" ||
    typeof record.title !== "string" ||
    typeof record.canGoBack !== "boolean" ||
    typeof record.canGoForward !== "boolean" ||
    typeof record.isLoading !== "boolean" ||
    typeof record.browserSessionId !== "string" ||
    typeof record.controlOwner !== "string" ||
    typeof record.origin !== "string" ||
    typeof record.ownerWebContentsId !== "number" ||
    typeof record.pageRevision !== "number" ||
    typeof record.selected !== "boolean" ||
    typeof record.tabId !== "string" ||
    typeof record.threadId !== "string" ||
    typeof record.webContentsId !== "number" ||
    typeof record.windowId !== "number"
  ) {
    throw new Error("Browser tab 未返回有效状态。");
  }
}

function assertBrowserTabClosedEvent(
  value: unknown,
): asserts value is BrowserTabClosedEvent {
  const record = asRecord(value);
  if (
    !record ||
    typeof record.browserSessionId !== "string" ||
    typeof record.reason !== "string" ||
    typeof record.tabId !== "string" ||
    typeof record.threadId !== "string" ||
    typeof record.viewId !== "string"
  ) {
    throw new Error("Browser tab 关闭事件字段不完整。");
  }
}

function assertBrowserTabLoadFailedEvent(
  value: unknown,
): asserts value is BrowserTabLoadFailedEvent {
  assertBrowserTabEventIdentity(value);
  const record = asRecord(value);
  if (
    !record ||
    typeof record.errorDescription !== "string" ||
    typeof record.failureCategory !== "string"
  ) {
    throw new Error("Browser tab 加载失败事件字段不完整。");
  }
}

function assertBrowserTabDownloadEvent(
  value: unknown,
): asserts value is BrowserTabDownloadEvent {
  assertBrowserTabEventIdentity(value);
  const record = asRecord(value);
  if (
    !record ||
    typeof record.downloadId !== "string" ||
    typeof record.filename !== "string" ||
    typeof record.state !== "string"
  ) {
    throw new Error("Browser tab 下载事件字段不完整。");
  }
  if (
    record.artifactRef !== undefined &&
    typeof record.artifactRef !== "string"
  ) {
    throw new Error("Browser tab artifact ref 字段不完整。");
  }
  for (const key of [
    "artifactFilename",
    "artifactCreatedAt",
    "artifactPersistedAt",
    "artifactSidecarPath",
    "artifactContentStatus",
    "artifactError",
  ]) {
    if (record[key] !== undefined && typeof record[key] !== "string") {
      throw new Error("Browser tab artifact 元数据字段不完整。");
    }
  }
  if (
    record.artifactStatus !== undefined &&
    record.artifactStatus !== "failed"
  ) {
    throw new Error("Browser tab artifact 状态字段不完整。");
  }
}

function assertBrowserTabPermissionRequestEvent(
  value: unknown,
): asserts value is BrowserTabPermissionRequestEvent {
  assertBrowserTabEventIdentity(value);
  const record = asRecord(value);
  if (
    !record ||
    typeof record.requestId !== "string" ||
    typeof record.permission !== "string" ||
    typeof record.decision !== "string"
  ) {
    throw new Error("Browser tab 权限事件字段不完整。");
  }
}

function assertBrowserTabEventIdentity(
  value: unknown,
): asserts value is BrowserTabEventIdentity {
  const record = asRecord(value);
  if (
    !record ||
    typeof record.browserSessionId !== "string" ||
    typeof record.ownerWebContentsId !== "number" ||
    typeof record.tabId !== "string" ||
    typeof record.threadId !== "string" ||
    !(
      typeof record.webContentsId === "number" || record.webContentsId === null
    ) ||
    typeof record.windowId !== "number"
  ) {
    throw new Error("Browser tab 事件身份字段不完整。");
  }
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}
