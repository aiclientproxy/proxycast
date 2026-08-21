import { randomUUID } from "node:crypto";
import process from "node:process";
import { isDeepStrictEqual } from "node:util";
import type { WebContents } from "electron";
import type { BrowserWindow, Rectangle } from "./electronRuntime";
import {
  browserNodeCenter,
  describeBrowserNode,
  observeBrowserPage,
} from "./browserTabObservation";
import {
  observeBrowserTabUserControl,
  type BrowserTabUserControlObserver,
} from "./browserTabUserControl";
import {
  ElectronEmbeddedBrowserHost,
  type EmbeddedBrowserViewState,
} from "./embeddedBrowserHost";

type HostArgs = Record<string, unknown> | null | undefined;
type HostEventEmitter = (event: string, payload?: unknown) => void;
type BrowserTurnInterruptHandler = (
  threadId: string,
  turnId: string,
) => Promise<void>;

export const BROWSER_TAB_COMMANDS = [
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

export type BrowserTabCommand = (typeof BROWSER_TAB_COMMANDS)[number];
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

export interface BrowserToolCall {
  approvalToken?: string;
  arguments: Record<string, unknown>;
  callId: string;
  ownerWebContentsId: number;
  phase: "preflight" | "approvedExecute";
  threadId: string;
  tool: string;
  turnId: string;
}

export interface BrowserToolApproval {
  actionKind: string;
  approvalToken: string;
  backendNodeId?: number;
  browserSessionId: string;
  reason: string;
  riskClass: string;
  snapshotId: string;
  tabId: string;
  viewId: string;
  webContentsId: number;
}

export interface BrowserToolResult {
  approval?: BrowserToolApproval;
  data?: unknown;
  imageBase64?: string;
  state?: BrowserTabState;
  status: "approval_required" | "completed" | "human_takeover";
}

interface PendingBrowserApproval {
  actionKind: string;
  approvalToken: string;
  arguments: Record<string, unknown>;
  backendNodeId: number | null;
  browserSessionId: string;
  callId: string;
  ownerWebContentsId: number;
  snapshotId: string;
  tabId: string;
  targetDescription: string | null;
  threadId: string;
  tool: string;
  turnId: string;
  viewId: string;
  webContentsId: number;
  windowId: number;
}

interface BrowserRoute {
  activeTurnId: string | null;
  browserSessionId: string;
  controlOwner: BrowserTabControlOwner;
  humanReason: string | null;
  mark: BrowserTabMark | null;
  origin: BrowserTabOrigin;
  ownerWebContentsId: number;
  pageRevision: number;
  latestSnapshotId: string | null;
  latestSnapshotNodeIds: Set<number>;
  lastPageStateKey: string | null;
  selected: boolean;
  tabId: string;
  threadId: string;
  viewId: string;
  windowId: number;
}

interface NativeUserControlObserver {
  observer: BrowserTabUserControlObserver;
  webContentsId: number;
}

const NAVIGATION_TIMEOUT_MS = 30_000;
const DANGEROUS_ACTION_PATTERN =
  /\b(delete|remove|submit|publish|purchase|pay|checkout|authorize|login|sign in|send)\b|删除|移除|提交|发布|购买|支付|授权|登录|发送/i;

export function isBrowserTabCommand(
  command: string,
): command is BrowserTabCommand {
  return BROWSER_TAB_COMMANDS.includes(command as BrowserTabCommand);
}

export class ElectronBrowserTabHost {
  readonly #embeddedHost: ElectronEmbeddedBrowserHost;
  readonly #emit: HostEventEmitter;
  readonly #routesByTabId = new Map<string, BrowserRoute>();
  readonly #tabIdsByViewId = new Map<string, string>();
  readonly #pendingApprovals = new Map<string, PendingBrowserApproval>();
  readonly #userControlObserversByViewId = new Map<
    string,
    NativeUserControlObserver
  >();
  #interruptTurn: BrowserTurnInterruptHandler | null = null;

  constructor(
    embeddedHost: ElectronEmbeddedBrowserHost,
    emit: HostEventEmitter = () => undefined,
  ) {
    this.#embeddedHost = embeddedHost;
    this.#emit = emit;
  }

  setTurnInterruptHandler(handler: BrowserTurnInterruptHandler | null): void {
    this.#interruptTurn = handler;
  }

  async invoke(
    window: BrowserWindow | null,
    command: BrowserTabCommand,
    args?: HostArgs,
  ): Promise<unknown> {
    if (command === "browser_tab_close") {
      return this.#closeFromRenderer(window, args);
    }
    if (!window || window.isDestroyed()) {
      throw new Error("Browser tab owner window is unavailable");
    }
    switch (command) {
      case "browser_tab_mount":
        return await this.#mount(window, args);
      case "browser_tab_set_bounds":
        return await this.#invokeForRenderer(
          window,
          args,
          "embedded_browser_view_set_bounds",
        );
      case "browser_tab_navigate":
        return await this.#invokeForRenderer(
          window,
          args,
          "embedded_browser_view_navigate",
          true,
        );
      case "browser_tab_reload":
        return await this.#invokeForRenderer(
          window,
          args,
          "embedded_browser_view_reload",
          true,
        );
      case "browser_tab_stop":
        return await this.#invokeForRenderer(
          window,
          args,
          "embedded_browser_view_stop",
          true,
        );
      case "browser_tab_find_in_page":
        return await this.#invokeForRenderer(
          window,
          args,
          "embedded_browser_view_find_in_page",
        );
      case "browser_tab_stop_find_in_page":
        return await this.#invokeForRenderer(
          window,
          args,
          "embedded_browser_view_stop_find_in_page",
        );
      case "browser_tab_set_zoom":
        return await this.#invokeForRenderer(
          window,
          args,
          "embedded_browser_view_set_zoom",
        );
      case "browser_tab_go_back":
        return await this.#invokeForRenderer(
          window,
          args,
          "embedded_browser_view_go_back",
          true,
        );
      case "browser_tab_go_forward":
        return await this.#invokeForRenderer(
          window,
          args,
          "embedded_browser_view_go_forward",
          true,
        );
      case "browser_tab_select":
        return await this.#select(window, args);
    }
  }

  async executeTool(call: BrowserToolCall): Promise<BrowserToolResult> {
    if (call.phase === "approvedExecute") {
      return await this.#executeApprovedTool(call);
    }
    if (call.phase !== "preflight" || call.approvalToken !== undefined) {
      throw new Error("Browser tool preflight phase is invalid");
    }
    if (call.tool === "openTabs") {
      return {
        status: "completed",
        data: this.#routesForThread(call.threadId, call.ownerWebContentsId).map(
          (route) => this.#readState(route),
        ),
      };
    }
    if (call.tool === "newTab") {
      const state = await this.#newAgentTab(call);
      return { status: "completed", state };
    }

    const route = this.#resolveToolRoute(call);
    if (call.tool === "claimTab") {
      if (route.activeTurnId !== null && route.activeTurnId !== call.turnId) {
        throw new Error("Browser tab is controlled by another active turn");
      }
      this.#assertClaimSnapshot(route, call.arguments);
      route.activeTurnId = call.turnId;
      route.controlOwner = "agent";
      route.humanReason = null;
      return { status: "completed", state: this.#emitState(route) };
    }
    if (call.tool === "releaseTab") {
      this.#assertAgentControl(route, call.turnId);
      this.#detachDebugger(route);
      route.activeTurnId = null;
      route.controlOwner = "released";
      route.humanReason = null;
      return { status: "completed", state: this.#emitState(route) };
    }

    this.#assertAgentControl(route, call.turnId);
    if (call.tool === "markHandoff" || call.tool === "markDeliverable") {
      route.mark = call.tool === "markHandoff" ? "handoff" : "deliverable";
      return { status: "completed", state: this.#emitState(route) };
    }

    const native = this.#nativeRoute(route);
    await this.#attachDebugger(route);
    switch (call.tool) {
      case "goto": {
        this.#invalidateSnapshot(route);
        const url = readHttpUrl(call.arguments.url);
        const navigation = waitForNavigation(native.view.webContents);
        await native.view.webContents.debugger.sendCommand("Page.navigate", {
          url,
        });
        await navigation;
        return { status: "completed", state: this.#emitState(route) };
      }
      case "observe": {
        const snapshotId = randomUUID();
        const observation = await observeBrowserPage(native.view.webContents, {
          pageRevision: route.pageRevision,
          snapshotId,
        });
        if (route.pageRevision !== observation.pageRevision) {
          throw new Error("Browser page changed while it was being observed");
        }
        route.latestSnapshotId = snapshotId;
        route.latestSnapshotNodeIds = new Set(
          observation.nodes
            .map((node) =>
              typeof node.backendNodeId === "number"
                ? node.backendNodeId
                : null,
            )
            .filter((node): node is number => node !== null),
        );
        return {
          status: "completed",
          state: this.#emitState(route),
          data: observation,
        };
      }
      case "screenshot": {
        const response = asRecord(
          await native.view.webContents.debugger.sendCommand(
            "Page.captureScreenshot",
            { format: "png", fromSurface: true },
          ),
        );
        const imageBase64 = readStringValue(response?.data);
        if (!imageBase64) {
          throw new Error("Browser screenshot did not return image data");
        }
        return {
          status: "completed",
          state: this.#emitState(route),
          imageBase64,
        };
      }
      case "click":
        this.#assertSnapshot(route, call.arguments, true);
        return await this.#click(route, call);
      case "fill":
        this.#assertSnapshot(route, call.arguments, true);
        return await this.#fill(route, call);
      case "press":
        this.#assertSnapshot(route, call.arguments, false);
        return await this.#press(route, call);
      default:
        throw new Error(`Unsupported Browser tool: ${call.tool}`);
    }
  }

  /** 仅供真实 Electron E2E 读取 pending approval，生产 IPC 不暴露 token。 */
  pendingApprovalForE2e(
    threadId: string,
    turnId: string,
  ): PendingBrowserApproval | null {
    for (const approval of this.#pendingApprovals.values()) {
      if (approval.threadId === threadId && approval.turnId === turnId) {
        return structuredClone(approval);
      }
    }
    return null;
  }

  pendingApprovalsForE2e(): PendingBrowserApproval[] {
    return [...this.#pendingApprovals.values()].map((approval) =>
      structuredClone(approval),
    );
  }

  observeEmbeddedEvent(event: string, payload: unknown): void {
    const record = asRecord(payload);
    const viewId = readString(record, "viewId");
    const tabId = viewId ? this.#tabIdsByViewId.get(viewId) : null;
    const route = tabId ? this.#routesByTabId.get(tabId) : null;
    if (!route || !record) {
      return;
    }
    if (event === "embedded-browser-view-state") {
      assertEmbeddedState(record);
      const nextPageStateKey = browserPageStateKey(record);
      if (
        route.lastPageStateKey !== null &&
        route.lastPageStateKey !== nextPageStateKey
      ) {
        this.#invalidateSnapshot(route);
      }
      this.#emitState(route, record);
      return;
    }
    if (event === "embedded-browser-view-destroyed") {
      this.#removeRoute(route);
      this.#emitClosed(
        route,
        readString(record, "reason") ?? "native-destroyed",
      );
      return;
    }
    const mappedEvent = BROWSER_EMBEDDED_EVENT_MAP[event];
    if (mappedEvent) {
      const native = this.#embeddedHost.resolveNativeView(route.viewId);
      const webContentsId = native?.view.webContents.id ?? null;
      this.#emit(mappedEvent, {
        ...record,
        browserSessionId: route.browserSessionId,
        ownerWebContentsId: route.ownerWebContentsId,
        tabId: route.tabId,
        threadId: route.threadId,
        webContentsId,
        windowId: route.windowId,
      });
    }
  }

  turnEnded(threadId: string, turnId: string): void {
    this.#clearPendingApprovals(
      (approval) =>
        approval.threadId === threadId && approval.turnId === turnId,
    );
    for (const route of [...this.#routesByTabId.values()]) {
      if (route.threadId !== threadId || route.activeTurnId !== turnId) {
        continue;
      }
      this.#detachDebugger(route);
      route.activeTurnId = null;
      route.controlOwner = "released";
      route.humanReason = null;
      const retain = route.origin === "user" || route.mark !== null;
      route.mark = null;
      if (retain) {
        this.#emitState(route);
      } else {
        this.#close(route, "turn-ended");
      }
    }
  }

  connectionLost(reason = "app-server-disconnected"): void {
    this.#pendingApprovals.clear();
    for (const route of [...this.#routesByTabId.values()]) {
      if (
        route.activeTurnId === null &&
        route.controlOwner !== "agent" &&
        route.controlOwner !== "human_takeover"
      ) {
        continue;
      }
      this.#detachDebugger(route);
      route.activeTurnId = null;
      route.controlOwner = "released";
      route.humanReason = null;
      const retain = route.origin === "user" || route.mark !== null;
      route.mark = null;
      if (retain && this.#embeddedHost.resolveNativeView(route.viewId)) {
        this.#emitState(route);
      } else {
        this.#close(route, reason);
      }
    }
  }

  dispose(): void {
    for (const route of [...this.#routesByTabId.values()]) {
      this.#close(route, "host-dispose");
    }
  }

  async #mount(
    window: BrowserWindow,
    args?: HostArgs,
  ): Promise<BrowserTabState> {
    const tabId = readRequiredString(args, "tabId");
    const existing = this.#routesByTabId.get(tabId);
    const identity = readMountIdentity(
      args,
      window,
      existing?.viewId ?? `browser_view_${randomUUID()}`,
    );
    if (existing) {
      assertSameRoute(existing, identity);
    } else {
      const viewOwner = this.#tabIdsByViewId.get(identity.viewId);
      if (viewOwner && viewOwner !== identity.tabId) {
        throw new Error("Browser viewId is already owned by another tab");
      }
      this.#routesByTabId.set(identity.tabId, identity);
      this.#tabIdsByViewId.set(identity.viewId, identity.tabId);
    }
    const route = existing ?? identity;
    // A remount only restores the existing native view; reapplying the initial
    // URL here would silently undo user navigation after Right Surface restore.
    const mountArgs = existing
      ? { ...asRecord(args), url: undefined }
      : asRecord(args);
    const state = await this.#embeddedHost.invoke(
      window,
      "embedded_browser_view_mount",
      {
        ...mountArgs,
        viewId: route.viewId,
        visible: route.selected,
      },
    );
    assertEmbeddedState(state);
    this.#observeNativeUserControl(route);
    if (!existing && route.selected) {
      return await this.#select(window, {
        ...asRecord(args),
        tabId: route.tabId,
      });
    }
    return this.#emitState(route, state);
  }

  async #invokeForRenderer(
    window: BrowserWindow,
    args: HostArgs,
    command:
      | "embedded_browser_view_find_in_page"
      | "embedded_browser_view_go_back"
      | "embedded_browser_view_go_forward"
      | "embedded_browser_view_navigate"
      | "embedded_browser_view_reload"
      | "embedded_browser_view_set_bounds"
      | "embedded_browser_view_set_zoom"
      | "embedded_browser_view_stop"
      | "embedded_browser_view_stop_find_in_page",
    userControl = false,
  ): Promise<BrowserTabState> {
    const route = this.#routeForRenderer(window, args);
    if (userControl) {
      this.#detachDebugger(route);
      this.#invalidateSnapshot(route);
      route.activeTurnId = null;
      route.controlOwner = "user";
      route.humanReason = null;
    }
    const state = await this.#embeddedHost.invoke(window, command, {
      ...asRecord(args),
      viewId: route.viewId,
    });
    assertEmbeddedState(state);
    return this.#emitState(route, state);
  }

  async #select(
    window: BrowserWindow,
    args?: HostArgs,
  ): Promise<BrowserTabState> {
    const selected = this.#routeForRenderer(window, args);
    for (const route of this.#routesForThread(
      selected.threadId,
      selected.ownerWebContentsId,
    )) {
      route.selected = route.tabId === selected.tabId;
      const native = this.#nativeRoute(route);
      const bounds = readBounds(args) ?? native.view.getBounds();
      await this.#embeddedHost.invoke(
        native.window,
        "embedded_browser_view_set_bounds",
        { viewId: route.viewId, bounds, visible: route.selected },
      );
      this.#emitState(route);
    }
    return this.#readState(selected);
  }

  #closeFromRenderer(
    window: BrowserWindow | null,
    args?: HostArgs,
  ): Record<string, never> {
    const route = readString(args, "tabId")
      ? this.#routesByTabId.get(readRequiredString(args, "tabId"))
      : null;
    if (!route) {
      return {};
    }
    if (!window || route.ownerWebContentsId !== window.webContents.id) {
      throw new Error("Browser tab owner window mismatch");
    }
    this.#close(route, "renderer-cleanup");
    return {};
  }

  async #newAgentTab(call: BrowserToolCall): Promise<BrowserTabState> {
    const base = this.#routesForThread(
      call.threadId,
      call.ownerWebContentsId,
    ).find((route) => route.selected);
    if (!base) {
      throw new Error("Browser newTab requires a visible tab in this thread");
    }
    const native = this.#nativeRoute(base);
    const tabId = call.callId;
    const route: BrowserRoute = {
      activeTurnId: call.turnId,
      browserSessionId: base.browserSessionId,
      controlOwner: "agent",
      humanReason: null,
      mark: null,
      origin: "agent",
      ownerWebContentsId: base.ownerWebContentsId,
      pageRevision: base.pageRevision,
      latestSnapshotId: null,
      latestSnapshotNodeIds: new Set(),
      lastPageStateKey: null,
      selected: true,
      tabId,
      threadId: call.threadId,
      viewId: `browser_view_${randomUUID()}`,
      windowId: base.windowId,
    };
    for (const item of this.#routesForThread(
      call.threadId,
      call.ownerWebContentsId,
    )) {
      item.selected = false;
      const itemNative = this.#nativeRoute(item);
      await this.#embeddedHost.invoke(
        itemNative.window,
        "embedded_browser_view_set_bounds",
        {
          viewId: item.viewId,
          bounds: itemNative.view.getBounds(),
          visible: false,
        },
      );
      this.#emitState(item);
    }
    this.#routesByTabId.set(tabId, route);
    this.#tabIdsByViewId.set(route.viewId, tabId);
    const state = await this.#embeddedHost.invoke(
      native.window,
      "embedded_browser_view_mount",
      {
        viewId: route.viewId,
        url: readHttpUrl(call.arguments.url),
        bounds: native.view.getBounds(),
        visible: true,
      },
    );
    assertEmbeddedState(state);
    this.#observeNativeUserControl(route);
    return this.#emitState(route, state);
  }

  #resolveToolRoute(call: BrowserToolCall): BrowserRoute {
    const tabId = readString(call.arguments, "tabId");
    const candidates = this.#routesForThread(
      call.threadId,
      call.ownerWebContentsId,
    );
    const route = tabId
      ? candidates.find((item) => item.tabId === tabId)
      : candidates.find((item) => item.selected);
    if (!route) {
      throw new Error("Browser tool route is stale or unavailable");
    }
    return route;
  }

  #routeForRenderer(window: BrowserWindow, args: HostArgs): BrowserRoute {
    const tabId = readRequiredString(args, "tabId");
    const route = this.#routesByTabId.get(tabId);
    if (
      !route ||
      route.ownerWebContentsId !== window.webContents.id ||
      route.windowId !== window.id
    ) {
      throw new Error("Browser tab route does not belong to this window");
    }
    return route;
  }

  #routesForThread(
    threadId: string,
    ownerWebContentsId: number,
  ): BrowserRoute[] {
    return [...this.#routesByTabId.values()].filter(
      (route) =>
        route.threadId === threadId &&
        route.ownerWebContentsId === ownerWebContentsId,
    );
  }

  #assertAgentControl(route: BrowserRoute, turnId: string): void {
    if (route.controlOwner !== "agent" || route.activeTurnId !== turnId) {
      throw new Error("Browser tab must be claimed by the active turn");
    }
  }

  #assertClaimSnapshot(
    route: BrowserRoute,
    args: Record<string, unknown>,
  ): void {
    const title = readRequiredString(args, "title");
    const url = readRequiredString(args, "url");
    const pageRevision = readNonNegativeInteger(args, "pageRevision");
    const state = this.#readState(route);
    if (
      state.title !== title ||
      state.url !== url ||
      route.pageRevision !== pageRevision
    ) {
      throw new Error("Browser claim snapshot is stale");
    }
  }

  #assertSnapshot(
    route: BrowserRoute,
    args: Record<string, unknown>,
    requireNode: boolean,
  ): void {
    const snapshotId = readRequiredString(args, "snapshotId");
    if (route.latestSnapshotId !== snapshotId) {
      throw new Error(
        "Browser page snapshot is stale or target is unavailable",
      );
    }
    if (requireNode) {
      const backendNodeId = readPositiveInteger(args.backendNodeId);
      if (!route.latestSnapshotNodeIds.has(backendNodeId)) {
        throw new Error(
          "Browser page snapshot is stale or target is unavailable",
        );
      }
    }
  }

  #invalidateSnapshot(route: BrowserRoute): void {
    this.#clearPendingApprovals((approval) => approval.tabId === route.tabId);
    route.pageRevision += 1;
    route.latestSnapshotId = null;
    route.latestSnapshotNodeIds.clear();
  }

  #observeNativeUserControl(route: BrowserRoute): void {
    const webContents = this.#nativeRoute(route).view.webContents;
    const existing = this.#userControlObserversByViewId.get(route.viewId);
    if (existing?.webContentsId === webContents.id) {
      return;
    }
    existing?.observer.dispose();
    const observer = observeBrowserTabUserControl(webContents, () => {
      this.#takeUserControl(route);
    });
    this.#userControlObserversByViewId.set(route.viewId, {
      observer,
      webContentsId: webContents.id,
    });
  }

  #takeUserControl(route: BrowserRoute): void {
    if (
      this.#routesByTabId.get(route.tabId) !== route ||
      (route.controlOwner !== "agent" &&
        route.controlOwner !== "human_takeover")
    ) {
      return;
    }
    const activeTurnId = route.activeTurnId;
    this.#detachDebugger(route);
    this.#invalidateSnapshot(route);
    route.activeTurnId = null;
    route.controlOwner = "user";
    route.humanReason = null;
    this.#emitState(route);
    if (activeTurnId && this.#interruptTurn) {
      void this.#interruptTurn(route.threadId, activeTurnId).catch((error) => {
        console.warn(
          "[browser-tab-host] failed to interrupt Agent turn after native user control",
          error,
        );
      });
    }
  }

  async #runAgentInput<T>(
    route: BrowserRoute,
    action: () => Promise<T>,
  ): Promise<T> {
    this.#observeNativeUserControl(route);
    const observer = this.#userControlObserversByViewId.get(route.viewId);
    if (!observer) {
      throw new Error("Browser native input observer is unavailable");
    }
    return await observer.observer.runAgentInput(action);
  }

  async #attachDebugger(route: BrowserRoute): Promise<void> {
    const debuggerApi = this.#nativeRoute(route).view.webContents.debugger;
    if (!debuggerApi.isAttached()) {
      debuggerApi.attach("1.3");
    }
    await debuggerApi.sendCommand("Page.enable");
    await debuggerApi.sendCommand("DOM.enable");
  }

  #detachDebugger(route: BrowserRoute): void {
    const native = this.#embeddedHost.resolveNativeView(route.viewId);
    if (!native) {
      return;
    }
    const debuggerApi = native.view.webContents.debugger;
    if (debuggerApi.isAttached()) {
      debuggerApi.detach();
    }
  }

  async #click(
    route: BrowserRoute,
    call: BrowserToolCall,
    expectedDescription: string | null = null,
  ): Promise<BrowserToolResult> {
    const args = call.arguments;
    const backendNodeId = readPositiveInteger(args.backendNodeId);
    const native = this.#nativeRoute(route);
    const description = await describeBrowserNode(
      native.view.webContents,
      backendNodeId,
    );
    if (expectedDescription !== null && description !== expectedDescription) {
      throw new Error("Browser approved click target changed after preflight");
    }
    if (
      expectedDescription === null &&
      DANGEROUS_ACTION_PATTERN.test(description)
    ) {
      return this.#requireApproval(
        route,
        call,
        `Sensitive click target: ${description}`,
        "high_impact_click",
        description,
      );
    }
    const point = await browserNodeCenter(
      native.view.webContents,
      backendNodeId,
    );
    await this.#runAgentInput(route, async () => {
      await native.view.webContents.debugger.sendCommand(
        "Input.dispatchMouseEvent",
        {
          type: "mousePressed",
          x: point.x,
          y: point.y,
          button: "left",
          clickCount: 1,
        },
      );
      await native.view.webContents.debugger.sendCommand(
        "Input.dispatchMouseEvent",
        {
          type: "mouseReleased",
          x: point.x,
          y: point.y,
          button: "left",
          clickCount: 1,
        },
      );
    });
    this.#invalidateSnapshot(route);
    return { status: "completed", state: this.#emitState(route) };
  }

  async #fill(
    route: BrowserRoute,
    call: BrowserToolCall,
    expectedDescription: string | null = null,
  ): Promise<BrowserToolResult> {
    const args = call.arguments;
    const backendNodeId = readPositiveInteger(args.backendNodeId);
    const text = readRequiredString(args, "text");
    const native = this.#nativeRoute(route);
    const description = await describeBrowserNode(
      native.view.webContents,
      backendNodeId,
    );
    if (expectedDescription !== null && description !== expectedDescription) {
      throw new Error("Browser approved fill target changed after preflight");
    }
    if (
      expectedDescription === null &&
      /password|token|secret|密码|令牌|密钥/i.test(description)
    ) {
      return this.#requireApproval(
        route,
        call,
        "Sensitive input requires approval",
        "sensitive_input",
        description,
      );
    }
    const debuggerApi = native.view.webContents.debugger;
    await this.#runAgentInput(route, async () => {
      await debuggerApi.sendCommand("DOM.focus", { backendNodeId });
      await debuggerApi.sendCommand("Input.dispatchKeyEvent", {
        type: "keyDown",
        key: "a",
        code: "KeyA",
        modifiers: process.platform === "darwin" ? 4 : 2,
      });
      await debuggerApi.sendCommand("Input.dispatchKeyEvent", {
        type: "keyUp",
        key: "a",
        code: "KeyA",
        modifiers: process.platform === "darwin" ? 4 : 2,
      });
      await debuggerApi.sendCommand("Input.insertText", { text });
    });
    this.#invalidateSnapshot(route);
    return { status: "completed", state: this.#emitState(route) };
  }

  async #press(
    route: BrowserRoute,
    call: BrowserToolCall,
    approved = false,
  ): Promise<BrowserToolResult> {
    const args = call.arguments;
    const key = readRequiredString(args, "key");
    if (!approved && /^(Enter|NumpadEnter)$/i.test(key)) {
      return this.#requireApproval(
        route,
        call,
        "Enter may submit the current form",
        "form_submission",
      );
    }
    const debuggerApi = this.#nativeRoute(route).view.webContents.debugger;
    await this.#runAgentInput(route, async () => {
      await debuggerApi.sendCommand("Input.dispatchKeyEvent", {
        type: "keyDown",
        key,
      });
      await debuggerApi.sendCommand("Input.dispatchKeyEvent", {
        type: "keyUp",
        key,
      });
    });
    this.#invalidateSnapshot(route);
    return { status: "completed", state: this.#emitState(route) };
  }

  #requireApproval(
    route: BrowserRoute,
    call: BrowserToolCall,
    reason: string,
    riskClass: string,
    targetDescription: string | null = null,
  ): BrowserToolResult {
    const callKey = browserToolCallKey(call);
    if (this.#pendingApprovals.has(callKey)) {
      throw new Error("Browser action approval is already pending");
    }
    const native = this.#nativeRoute(route);
    const approvalToken = randomUUID();
    const snapshotId = readRequiredString(call.arguments, "snapshotId");
    const backendNodeId =
      call.tool === "click" || call.tool === "fill"
        ? readPositiveInteger(call.arguments.backendNodeId)
        : null;
    const approval: PendingBrowserApproval = {
      actionKind: call.tool,
      approvalToken,
      arguments: structuredClone(call.arguments),
      backendNodeId,
      browserSessionId: route.browserSessionId,
      callId: call.callId,
      ownerWebContentsId: call.ownerWebContentsId,
      snapshotId,
      tabId: route.tabId,
      targetDescription,
      threadId: call.threadId,
      tool: call.tool,
      turnId: call.turnId,
      viewId: route.viewId,
      webContentsId: native.view.webContents.id,
      windowId: route.windowId,
    };
    this.#pendingApprovals.set(callKey, approval);
    return {
      status: "approval_required",
      state: this.#emitState(route),
      approval: {
        actionKind: approval.actionKind,
        approvalToken,
        ...(backendNodeId === null ? {} : { backendNodeId }),
        browserSessionId: approval.browserSessionId,
        reason,
        riskClass,
        snapshotId,
        tabId: approval.tabId,
        viewId: approval.viewId,
        webContentsId: approval.webContentsId,
      },
    };
  }

  async #executeApprovedTool(
    call: BrowserToolCall,
  ): Promise<BrowserToolResult> {
    const callKey = browserToolCallKey(call);
    const pending = this.#pendingApprovals.get(callKey);
    this.#pendingApprovals.delete(callKey);
    if (!pending || call.approvalToken !== pending.approvalToken) {
      throw new Error("Browser action approval token is stale or invalid");
    }
    if (
      call.tool !== pending.tool ||
      call.threadId !== pending.threadId ||
      call.turnId !== pending.turnId ||
      call.callId !== pending.callId ||
      call.ownerWebContentsId !== pending.ownerWebContentsId ||
      !isDeepStrictEqual(call.arguments, pending.arguments)
    ) {
      throw new Error("Browser approved action identity mismatch");
    }
    const route = this.#resolveToolRoute(call);
    this.#assertAgentControl(route, call.turnId);
    const native = this.#nativeRoute(route);
    if (
      route.browserSessionId !== pending.browserSessionId ||
      route.tabId !== pending.tabId ||
      route.viewId !== pending.viewId ||
      route.windowId !== pending.windowId ||
      native.view.webContents.id !== pending.webContentsId
    ) {
      throw new Error("Browser approved action native route mismatch");
    }
    this.#assertSnapshot(
      route,
      call.arguments,
      call.tool === "click" || call.tool === "fill",
    );
    if (
      readRequiredString(call.arguments, "snapshotId") !== pending.snapshotId
    ) {
      throw new Error("Browser approved action snapshot mismatch");
    }
    if (
      pending.backendNodeId !== null &&
      readPositiveInteger(call.arguments.backendNodeId) !==
        pending.backendNodeId
    ) {
      throw new Error("Browser approved action target mismatch");
    }
    switch (call.tool) {
      case "click":
        return await this.#click(route, call, pending.targetDescription);
      case "fill":
        return await this.#fill(route, call, pending.targetDescription);
      case "press":
        return await this.#press(route, call, true);
      default:
        throw new Error("Browser tool does not support approved execution");
    }
  }

  #clearPendingApprovals(
    predicate: (approval: PendingBrowserApproval) => boolean,
  ): void {
    for (const [key, approval] of this.#pendingApprovals) {
      if (predicate(approval)) {
        this.#pendingApprovals.delete(key);
      }
    }
  }

  #nativeRoute(route: BrowserRoute) {
    const native = this.#embeddedHost.resolveNativeView(route.viewId);
    if (
      !native ||
      native.window.id !== route.windowId ||
      native.window.webContents.id !== route.ownerWebContentsId
    ) {
      throw new Error("Browser native route is stale or owner mismatched");
    }
    return native;
  }

  #readState(route: BrowserRoute): BrowserTabState {
    const native = this.#nativeRoute(route);
    return enrichState(route, native.state, native.view.webContents.id);
  }

  #emitState(
    route: BrowserRoute,
    state?: EmbeddedBrowserViewState,
  ): BrowserTabState {
    const native = this.#nativeRoute(route);
    const next = enrichState(
      route,
      state ?? native.state,
      native.view.webContents.id,
    );
    route.lastPageStateKey = browserPageStateKey(next);
    this.#emit("browser-tab-state", next);
    return next;
  }

  #close(route: BrowserRoute, reason: string): void {
    this.#detachDebugger(route);
    const wasSelected = route.selected;
    this.#removeRoute(route);
    void this.#embeddedHost.invoke(null, "embedded_browser_view_destroy", {
      viewId: route.viewId,
    });
    this.#emitClosed(route, reason);
    if (wasSelected) {
      const fallback = this.#routesForThread(
        route.threadId,
        route.ownerWebContentsId,
      )[0];
      if (fallback) {
        fallback.selected = true;
        const native = this.#nativeRoute(fallback);
        void this.#embeddedHost.invoke(
          native.window,
          "embedded_browser_view_set_bounds",
          {
            viewId: fallback.viewId,
            bounds: native.view.getBounds(),
            visible: true,
          },
        );
        this.#emitState(fallback);
      }
    }
  }

  #removeRoute(route: BrowserRoute): void {
    this.#clearPendingApprovals((approval) => approval.tabId === route.tabId);
    this.#userControlObserversByViewId.get(route.viewId)?.observer.dispose();
    this.#userControlObserversByViewId.delete(route.viewId);
    this.#routesByTabId.delete(route.tabId);
    this.#tabIdsByViewId.delete(route.viewId);
  }

  #emitClosed(route: BrowserRoute, reason: string): void {
    this.#emit("browser-tab-closed", {
      browserSessionId: route.browserSessionId,
      reason,
      tabId: route.tabId,
      threadId: route.threadId,
      viewId: route.viewId,
    });
  }
}

function browserToolCallKey(call: BrowserToolCall): string {
  return [call.threadId, call.turnId, call.callId].join("\u0000");
}

const BROWSER_EMBEDDED_EVENT_MAP: Record<string, string> = {
  "embedded-browser-view-download": "browser-tab-download",
  "embedded-browser-view-load-failed": "browser-tab-load-failed",
  "embedded-browser-view-permission-request": "browser-tab-permission-request",
};

function readMountIdentity(
  args: HostArgs,
  window: BrowserWindow,
  viewId: string,
): BrowserRoute {
  const requestedOrigin = readString(args, "origin");
  if (requestedOrigin && requestedOrigin !== "user") {
    throw new Error("Renderer may only mount user Browser tabs");
  }
  const origin = "user";
  return {
    activeTurnId: null,
    browserSessionId: readRequiredString(args, "browserSessionId"),
    controlOwner: origin === "user" ? "user" : "released",
    humanReason: null,
    mark: null,
    origin,
    ownerWebContentsId: window.webContents.id,
    pageRevision: 0,
    latestSnapshotId: null,
    latestSnapshotNodeIds: new Set(),
    lastPageStateKey: null,
    selected: readOptionalBoolean(args, "selected") ?? true,
    tabId: readRequiredString(args, "tabId"),
    threadId: readRequiredString(args, "threadId"),
    viewId,
    windowId: window.id,
  };
}

function assertSameRoute(route: BrowserRoute, next: BrowserRoute): void {
  for (const key of [
    "browserSessionId",
    "ownerWebContentsId",
    "tabId",
    "threadId",
    "viewId",
    "windowId",
  ] as const) {
    if (route[key] !== next[key]) {
      throw new Error(`Browser route identity mismatch: ${key}`);
    }
  }
}

function enrichState(
  route: BrowserRoute,
  state: EmbeddedBrowserViewState,
  webContentsId: number,
): BrowserTabState {
  return {
    ...state,
    activeTurnId: route.activeTurnId,
    browserSessionId: route.browserSessionId,
    controlOwner: route.controlOwner,
    humanReason: route.humanReason,
    mark: route.mark,
    origin: route.origin,
    ownerWebContentsId: route.ownerWebContentsId,
    pageRevision: route.pageRevision,
    selected: route.selected,
    tabId: route.tabId,
    threadId: route.threadId,
    webContentsId,
    windowId: route.windowId,
  };
}

function browserPageStateKey(value: unknown): string {
  const record = asRecord(value);
  return JSON.stringify([
    record?.url ?? "",
    record?.title ?? "",
    record?.isLoading === true,
  ]);
}

function waitForNavigation(webContents: WebContents): Promise<void> {
  return new Promise((resolve, reject) => {
    const finish = () => {
      clearTimeout(timeout);
      webContents.off("did-stop-loading", finish);
      webContents.off("did-fail-load", fail);
      resolve();
    };
    const fail = (
      _event: unknown,
      errorCode: number,
      errorDescription: string,
    ) => {
      clearTimeout(timeout);
      webContents.off("did-stop-loading", finish);
      webContents.off("did-fail-load", fail);
      reject(
        new Error(
          `Browser navigation failed (${errorCode}): ${errorDescription}`,
        ),
      );
    };
    const timeout = setTimeout(() => {
      webContents.off("did-stop-loading", finish);
      webContents.off("did-fail-load", fail);
      reject(new Error("Browser navigation timed out"));
    }, NAVIGATION_TIMEOUT_MS);
    webContents.once("did-stop-loading", finish);
    webContents.once("did-fail-load", fail);
  });
}

function assertEmbeddedState(
  value: unknown,
): asserts value is EmbeddedBrowserViewState {
  const record = asRecord(value);
  if (
    !record ||
    typeof record.viewId !== "string" ||
    typeof record.url !== "string" ||
    typeof record.title !== "string"
  ) {
    throw new Error("Browser native host returned invalid state");
  }
}

function readRequiredString(value: unknown, key: string): string {
  const result = readString(value, key);
  if (!result) {
    throw new Error(`Browser ${key} must be a non-empty string`);
  }
  return result;
}

function readString(value: unknown, key: string): string | null {
  return readStringValue(asRecord(value)?.[key]);
}

function readStringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readPositiveInteger(value: unknown): number {
  if (!Number.isInteger(value) || Number(value) <= 0) {
    throw new Error("Browser backendNodeId must be a positive integer");
  }
  return Number(value);
}

function readNonNegativeInteger(value: unknown, key: string): number {
  const candidate = asRecord(value)?.[key];
  if (!Number.isInteger(candidate) || Number(candidate) < 0) {
    throw new Error(`Browser ${key} must be a non-negative integer`);
  }
  return Number(candidate);
}

function readOptionalBoolean(value: unknown, key: string): boolean | null {
  const candidate = asRecord(value)?.[key];
  return typeof candidate === "boolean" ? candidate : null;
}

function readBounds(value: unknown): Rectangle | null {
  const record = asRecord(asRecord(value)?.bounds);
  const x = record?.x;
  const y = record?.y;
  const width = record?.width;
  const height = record?.height;
  if ([x, y, width, height].every(Number.isFinite)) {
    return {
      x: Math.max(0, Math.round(Number(x))),
      y: Math.max(0, Math.round(Number(y))),
      width: Math.max(0, Math.round(Number(width))),
      height: Math.max(0, Math.round(Number(height))),
    };
  }
  return null;
}

function readHttpUrl(value: unknown): string {
  const url = readOptionalHttpUrl(value);
  if (!url) {
    throw new Error("Browser url must use http or https");
  }
  return url;
}

function readOptionalHttpUrl(value: unknown): string | null {
  const raw = readStringValue(value);
  if (!raw) {
    return null;
  }
  try {
    const url = new URL(raw);
    return url.protocol === "http:" || url.protocol === "https:"
      ? url.href
      : null;
  } catch {
    return null;
  }
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}
