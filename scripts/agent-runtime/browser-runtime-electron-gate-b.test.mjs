import { describe, expect, it } from "vitest";

import {
  buildAssertions,
  extractBrowserObservation,
  extractBrowserState,
  extractFinalBrowserState,
  isReclaimedBrowserWorkspaceState,
  parseArgs,
} from "./browser-runtime-electron-gate-b.mjs";
import { buildApprovalAssertions } from "./browser-runtime-electron-gate-b-approval.mjs";
import { buildCancelAssertions } from "./browser-runtime-electron-gate-b-cancel.mjs";
import { buildDisconnectAssertions } from "./browser-runtime-electron-gate-b-disconnect.mjs";
import { buildDownloadAssertions } from "./browser-runtime-electron-gate-b-download.mjs";
import { buildPermissionAssertions } from "./browser-runtime-electron-gate-b-permission.mjs";
import { buildUserControlAssertions } from "./browser-runtime-electron-gate-b-user-control.mjs";
import { buildWindowCloseAssertions } from "./browser-runtime-electron-gate-b-window-close.mjs";

describe("Browser Electron Gate B", () => {
  it("解析独立 fixture 参数并拒绝过短超时", () => {
    expect(
      parseArgs([
        "--output",
        ".lime/qc/browser.json",
        "--timeout-ms",
        "60000",
        "--interval-ms",
        "250",
        "--keep-temp",
      ]),
    ).toMatchObject({
      output: expect.stringContaining(".lime/qc/browser.json"),
      timeoutMs: 60000,
      intervalMs: 250,
      keepTemp: true,
      scenario: "lifecycle",
    });
    expect(() => parseArgs(["--timeout-ms", "1000"])).toThrow(
      "--timeout-ms 必须 >= 30000",
    );
    expect(() => parseArgs(["--interval-ms", "10"])).toThrow(
      "--interval-ms 必须 >= 100",
    );
    expect(parseArgs(["--scenario", "cancel"]).output).toContain(
      "browser-runtime-electron-gate-b-cancel-summary.json",
    );
    expect(parseArgs(["--scenario", "projection"]).output).toContain(
      "browser-runtime-gate-a-summary.json",
    );
    expect(parseArgs(["--scenario", "approval"]).output).toContain(
      "browser-runtime-electron-gate-b-approval-summary.json",
    );
    expect(parseArgs(["--scenario", "user-control"]).output).toContain(
      "browser-runtime-electron-gate-b-user-control-summary.json",
    );
    expect(parseArgs(["--scenario", "window-close"]).output).toContain(
      "browser-runtime-electron-gate-b-window-close-summary.json",
    );
    expect(parseArgs(["--scenario", "disconnect"]).output).toContain(
      "browser-runtime-electron-gate-b-disconnect-summary.json",
    );
    expect(parseArgs(["--scenario", "permission"]).output).toContain(
      "browser-runtime-electron-gate-b-permission-summary.json",
    );
    expect(parseArgs(["--scenario", "download"]).output).toContain(
      "browser-runtime-electron-gate-b-download-summary.json",
    );
    expect(() => parseArgs(["--scenario", "unknown"])).toThrow(
      "--scenario 必须是 projection、lifecycle、approval、user-control、cancel、window-close、disconnect、permission、download 或 artifact",
    );
  });

  it("审批只允许单次授权，批准执行一次且拒绝不再变更页面", () => {
    const decisions = ["allow_once", "decline", "cancel"];
    const params = {
      activeTurnId: "turn-1",
      agentControlled: {
        tabId: "tab-1",
        webContentsId: 42,
      },
      approvedMutation: {
        state: {
          activeTurnId: "turn-1",
          tabId: "tab-1",
          webContentsId: 42,
        },
      },
      consoleErrors: [],
      debuggerAfterTerminal: { attached: false },
      debuggerBeforeApproval: { attached: true },
      declinedMutationFailure: "Browser approval was declined",
      firstDecision: { clicked: true, decision: "allow_once" },
      firstPrompt: {
        decisions,
        interactionId: "approval-1",
        summary: "Sensitive click target: Delete account",
      },
      finalMarker: "BROWSER_RUNTIME_GATE_B_DONE",
      finalText: "BROWSER_RUNTIME_GATE_B_DONE",
      initial: {
        observation: { snapshotId: "snapshot-1" },
        target: { backendNodeId: 7, name: "Delete account" },
      },
      invokeDiagnostics: {
        invokeErrorCount: 0,
        mockFallbackHitCount: 0,
      },
      mutationCountAfterApproval: 1,
      mutationCountAfterDecline: 1,
      pageErrors: [],
      released: {
        activeTurnId: null,
        controlOwner: "released",
        webContentsId: 42,
      },
      secondDecision: { clicked: true, decision: "decline" },
      secondObservation: {
        observation: { snapshotId: "snapshot-2" },
        state: { webContentsId: 42 },
      },
      secondPrompt: {
        decisions,
        interactionId: "approval-2",
      },
      terminal: { turn: { id: "turn-1", status: "completed" } },
    };

    expect(Object.values(buildApprovalAssertions(params)).every(Boolean)).toBe(
      true,
    );
    expect(
      buildApprovalAssertions({
        ...params,
        firstPrompt: {
          ...params.firstPrompt,
          decisions: [...decisions, "allow_for_session"],
        },
      }).browserApprovalIsOnceOnly,
    ).toBe(false);
    expect(
      buildApprovalAssertions({
        ...params,
        mutationCountAfterApproval: 2,
      }).approvedMutationExecutedOnce,
    ).toBe(false);
    expect(
      buildApprovalAssertions({
        ...params,
        mutationCountAfterDecline: 2,
      }).declinedWithoutMutation,
    ).toBe(false);
  });

  it("native 用户输入撤销 Agent 控制、snapshot 和旧审批 token", () => {
    const params = {
      activeTurnId: "turn-1",
      agentControlled: {
        activeTurnId: "turn-1",
        controlOwner: "agent",
        pageRevision: 4,
        webContentsId: 42,
      },
      consoleErrors: [],
      debuggerAfterTerminal: { attached: false },
      debuggerAfterUserInput: { attached: false },
      debuggerBeforeUserInput: { attached: true },
      finalMarker: "BROWSER_RUNTIME_GATE_B_DONE",
      finalText: "BROWSER_RUNTIME_GATE_B_DONE",
      initial: {
        observation: { pageRevision: 4, snapshotId: "snapshot-1" },
      },
      invokeDiagnostics: {
        invokeErrorCount: 0,
        mockFallbackHitCount: 0,
      },
      mutationCountAfterStaleApproval: 1,
      mutationCountAfterUserInput: 1,
      pageErrors: [],
      staleApprovalDecision: {
        decision: { clicked: true, decision: "allow_once" },
        prompt: {
          summary: "Sensitive click target: Delete account",
        },
      },
      staleApprovalFailure: "approval token is stale or invalid",
      terminal: { turn: { id: "turn-1", status: "interrupted" } },
      userControlState: {
        activeTurnId: null,
        controlOwner: "user",
        pageRevision: 5,
        webContentsId: 42,
      },
      userInput: { clicked: true, webContentsId: 42 },
    };

    expect(
      Object.values(buildUserControlAssertions(params)).every(Boolean),
    ).toBe(true);
    expect(
      buildUserControlAssertions({
        ...params,
        mutationCountAfterStaleApproval: 2,
      }).staleApprovalDidNotReplayMutation,
    ).toBe(false);
    expect(
      buildUserControlAssertions({
        ...params,
        userControlState: {
          ...params.userControlState,
          controlOwner: "agent",
        },
      }).userInputRevokedAgentControl,
    ).toBe(false);
  });

  it("要求真实下载事件、canonical tab identity、无路径泄露和 terminal 同时成立", () => {
    const downloadEvent = (state) => ({
      event: "browser-tab-download",
      payload: {
        browserSessionId: "browser-session-1",
        downloadId: "download-1",
        filename: "browser-gate-b.txt",
        ownerWebContentsId: 41,
        state,
        tabId: "tab-1",
        threadId: "thread-1",
        viewId: "view-1",
        webContentsId: 42,
        windowId: 7,
      },
    });
    const assertions = buildDownloadAssertions({
      agentControlled: {
        sessionId: "browser-session-1",
        tabId: "tab-1",
        viewId: "view-1",
        webContentsId: 42,
      },
      cancelEvidence: {
        diagnostics: { consoleErrors: [], pageErrors: [] },
        failedAssertions: [],
        invoke: { mockFallbackHitCount: 0 },
        status: "pass",
      },
      events: [downloadEvent("started"), downloadEvent("cancelled")],
      gui: {
        banner: {
          bottom: 160,
          height: 40,
          text: "browser-gate-b.txt 下载已取消",
          visible: true,
          width: 600,
          x: 100,
          y: 120,
        },
        viewport: { height: 500, width: 600, x: 100, y: 160 },
        workspace: {
          sessionId: "browser-session-1",
          tabId: "tab-1",
          threadId: "thread-1",
          viewId: "view-1",
          webContentsId: 42,
        },
      },
      initial: {
        state: {
          browserSessionId: "browser-session-1",
          ownerWebContentsId: 41,
          threadId: "thread-1",
          windowId: 7,
        },
      },
      native: {
        bounds: { height: 500, width: 600, x: 100, y: 160 },
      },
      trigger: { filename: "browser-gate-b.txt", triggered: true },
    });
    expect(Object.entries(assertions).filter(([, passed]) => !passed)).toEqual(
      [],
    );
  });

  it("要求真实权限拒绝、canonical tab identity、GUI/native 非遮挡和 terminal 同时成立", () => {
    const assertions = buildPermissionAssertions({
      agentControlled: {
        sessionId: "browser-session-1",
        tabId: "tab-1",
        viewId: "view-1",
        webContentsId: 42,
      },
      cancelEvidence: {
        diagnostics: { consoleErrors: [], pageErrors: [] },
        failedAssertions: [],
        invoke: { mockFallbackHitCount: 0 },
        status: "pass",
      },
      event: {
        event: "browser-tab-permission-request",
        payload: {
          browserSessionId: "browser-session-1",
          decision: "blocked",
          ownerWebContentsId: 41,
          permission: "geolocation",
          requestingUrl: "https://example.com/page",
          tabId: "tab-1",
          threadId: "thread-1",
          url: "https://example.com/page",
          viewId: "view-1",
          webContentsId: 42,
          windowId: 7,
        },
      },
      gui: {
        banner: {
          bottom: 160,
          height: 40,
          text: "已阻止 geolocation 权限",
          visible: true,
          width: 600,
          x: 100,
          y: 120,
        },
        viewport: { height: 500, width: 600, x: 100, y: 160 },
        workspace: {
          sessionId: "browser-session-1",
          tabId: "tab-1",
          threadId: "thread-1",
          viewId: "view-1",
          webContentsId: 42,
        },
      },
      initial: {
        state: {
          browserSessionId: "browser-session-1",
          ownerWebContentsId: 41,
          threadId: "thread-1",
          windowId: 7,
        },
      },
      native: {
        bounds: { height: 500, width: 600, x: 100, y: 160 },
      },
      permissionResult: {
        code: 1,
        message: "User denied Geolocation",
        origin: "https://example.com",
        outcome: "blocked",
        permissionAfter: "denied",
        url: "https://example.com/page",
      },
    });
    expect(Object.entries(assertions).filter(([, passed]) => !passed)).toEqual(
      [],
    );
  });

  it("要求真实 interrupt、read model、provider 取消和 Browser cleanup 同时成立", () => {
    const assertions = buildCancelAssertions({
      activeTurnId: "turn-1",
      agentControlled: {
        activeTurnId: "turn-1",
        controlOwner: "agent",
        sessionId: "browser-session-1",
        tabId: "tab-1",
        webContentsId: 42,
      },
      consoleErrors: [],
      debuggerAfterInterrupt: {
        attached: false,
        exists: true,
        webContentsId: 42,
      },
      debuggerBeforeInterrupt: {
        attached: true,
        exists: true,
        webContentsId: 42,
      },
      finalMarker: "BROWSER_RUNTIME_GATE_B_DONE",
      guiAfterInterrupt: { bodyText: "", webContentsId: 42 },
      identity: { threadId: "thread-1" },
      interruptCall: {
        method: "turn/interrupt",
        status: "success",
        threadId: "thread-1",
        transport: "electron-ipc",
        turnId: "turn-1",
      },
      invokeDiagnostics: {
        browserNavigateCount: 0,
        invokeErrorCount: 0,
        mockFallbackHitCount: 0,
      },
      pageErrors: [],
      providerAfterInterrupt: { requestCount: 4 },
      providerBeforeInterrupt: {
        requestCount: 4,
        unfinishedResponseCloseCount: 0,
      },
      providerCancellation: {
        event: "response-close",
        responseFinished: false,
      },
      released: {
        activeTurnId: null,
        controlOwner: "released",
        sessionId: "browser-session-1",
        tabId: "tab-1",
        webContentsId: 42,
      },
      terminalStatus: "interrupted",
      terminalTurnId: "turn-1",
    });
    expect(Object.values(assertions).every(Boolean)).toBe(true);
  });

  it("要求真实 BrowserWindow close 与嵌入 WebContents 同步销毁", () => {
    const assertions = buildWindowCloseAssertions({
      agentControlled: {
        activeTurnId: "turn-1",
        controlOwner: "agent",
        ownerWebContentsId: 41,
        tabId: "tab-1",
        threadId: "thread-1",
        viewId: "view-1",
        webContentsId: 42,
        windowId: 7,
      },
      closeRequest: {
        exists: true,
        trigger: "browser-window-destroy",
        windowId: 7,
      },
      consoleErrors: [],
      debuggerBeforeClose: {
        attached: true,
        exists: true,
        webContentsId: 42,
      },
      guiBeforeTurn: {
        activeSurface: "browser",
        panelVisible: true,
      },
      identity: { sessionId: "session-1", threadId: "thread-1" },
      nativeAfterClose: {
        browserWebContentsDestroyed: true,
        browserWebContentsExists: false,
        windowDestroyed: true,
        windowExists: false,
        routeClosedEvent: {
          event: "browser-tab-closed",
          payload: {
            reason: "window-closed",
            tabId: "tab-1",
            threadId: "thread-1",
            viewId: "view-1",
          },
        },
      },
      pageClosed: true,
      pageErrors: [],
      providerRequestsBeforeClose: { after: 4, before: 4 },
    });
    expect(Object.values(assertions).every(Boolean)).toBe(true);
  });

  it("要求真实 App Server sidecar 断连释放 Browser 控制", () => {
    const assertions = buildDisconnectAssertions({
      activeTurnId: "turn-1",
      agentControlled: {
        activeTurnId: "turn-1",
        controlOwner: "agent",
        sessionId: "browser-session-1",
        tabId: "tab-1",
        webContentsId: 42,
      },
      consoleErrors: [],
      debuggerAfterDisconnect: {
        attached: false,
        exists: true,
        webContentsId: 42,
      },
      debuggerBeforeDisconnect: {
        attached: true,
        exists: true,
        webContentsId: 42,
      },
      finalMarker: "BROWSER_RUNTIME_GATE_B_DONE",
      guiAfterDisconnect: { bodyText: "", webContentsId: 42 },
      invokeDiagnostics: {
        browserNavigateCount: 0,
        mockFallbackHitCount: 0,
      },
      pageErrors: [],
      providerAfterDisconnect: { requestCount: 4 },
      providerBeforeDisconnect: {
        requestCount: 4,
        unfinishedResponseCloseCount: 0,
      },
      providerCancellation: {
        event: "response-close",
        responseFinished: false,
      },
      released: {
        activeTurnId: null,
        controlOwner: "released",
        sessionId: "browser-session-1",
        tabId: "tab-1",
        webContentsId: 42,
      },
      termination: {
        available: true,
        pid: 4321,
        requested: true,
        signal: "SIGTERM",
      },
    });
    expect(Object.values(assertions).every(Boolean)).toBe(true);
  });

  it("从动态工具输出中提取 canonical Browser state", () => {
    expect(
      extractBrowserState({
        contentItems: [
          {
            type: "inputText",
            text: JSON.stringify({
              data: [
                {
                  browserSessionId: "browser-session-1",
                  tabId: "tab-1",
                  url: "https://example.com",
                  title: "Example Domain",
                  pageRevision: 0,
                  webContentsId: 42,
                },
              ],
            }),
          },
        ],
      }),
    ).toMatchObject({ tabId: "tab-1", webContentsId: 42 });
    expect(
      extractBrowserState({
        browserSessionId: "browser-session-1",
        tabId: "tab-1",
        url: "https://example.com",
        title: "Example Domain",
      }),
    ).toMatchObject({ tabId: "tab-1", browserSessionId: "browser-session-1" });
  });

  it("从 observe 输出中提取 snapshot identity", () => {
    expect(
      extractBrowserObservation({
        contentItems: [
          {
            type: "inputText",
            text: JSON.stringify({
              data: {
                pageRevision: 3,
                snapshotId: "snapshot-3",
                nodes: [],
              },
            }),
          },
        ],
      }),
    ).toMatchObject({ pageRevision: 3, snapshotId: "snapshot-3" });
  });

  it("从最终 assistant marker 中提取 Agent observation identity", () => {
    expect(
      extractFinalBrowserState(
        'before\nBROWSER_RUNTIME_GATE_B_DONE:{"browserSessionId":"browser-session-1","tabId":"tab-1","webContentsId":42}\nafter',
      ),
    ).toEqual({
      browserSessionId: "browser-session-1",
      tabId: "tab-1",
      webContentsId: 42,
    });
    expect(
      extractFinalBrowserState("BROWSER_RUNTIME_GATE_B_DONE:invalid"),
    ).toBeNull();
  });

  it("重新 claim 对齐 Agent observe 的 canonical revision", () => {
    const state = {
      activeTurnId: "turn-1",
      controlOwner: "agent",
      pageRevision: 8,
      webContentsId: 42,
    };
    const params = {
      activeTurnId: "turn-1",
      initialWebContentsId: 42,
      recovered: { observation: { pageRevision: 8 } },
      state,
    };

    expect(isReclaimedBrowserWorkspaceState(params)).toBe(true);
    expect(
      isReclaimedBrowserWorkspaceState({
        ...params,
        recovered: { observation: { pageRevision: 7 } },
      }),
    ).toBe(false);
  });

  it("要求 GUI、Electron IPC 和动态工具 round trip 同时成立", () => {
    const assertions = buildAssertions({
      gui: {
        panelVisible: true,
        activeSurface: "browser",
        sessionId: "browser-session-1",
        tabId: "tab-1",
        threadId: "thread-1",
        webContentsId: 42,
      },
      initial: {
        observation: { pageRevision: 0, snapshotId: "snapshot-before" },
        state: {
          activeTurnId: "turn-1",
          browserSessionId: "browser-session-1",
          tabId: "tab-1",
          threadId: "thread-1",
          webContentsId: 42,
        },
      },
      latestAfterUserNavigation: {
        activeTurnId: null,
        controlOwner: "user",
        pageRevision: 1,
        sessionId: "browser-session-1",
        tabId: "tab-1",
        threadId: "thread-1",
        webContentsId: 42,
      },
      recovered: {
        observation: { pageRevision: 2, snapshotId: "snapshot-after" },
        state: {
          activeTurnId: "turn-1",
          browserSessionId: "browser-session-1",
          controlOwner: "agent",
          pageRevision: 2,
          tabId: "tab-1",
          threadId: "thread-1",
          webContentsId: 42,
        },
      },
      released: {
        activeTurnId: null,
        controlOwner: "released",
        sessionId: "browser-session-1",
        tabId: "tab-1",
        webContentsId: 42,
      },
      staleMutationFailure: "stale_snapshot_rejected",
      debuggerBeforeTerminal: {
        attached: true,
        exists: true,
        webContentsId: 42,
      },
      debuggerAfterTerminal: {
        attached: false,
        exists: true,
        webContentsId: 42,
      },
      destroyed: {
        tabId: null,
        webContentsId: null,
      },
      identity: { threadId: "thread-1" },
      turnId: "turn-1",
      trace: [
        {
          command: "browser_tab_mount",
          status: "error",
          transport: "electron-ipc",
          args_preview: { tabId: "tab-1" },
        },
        {
          command: "browser_tab_mount",
          status: "success",
          transport: "electron-ipc",
          args_preview: { tabId: "tab-1" },
        },
        {
          command: "browser_tab_navigate",
          status: "success",
          transport: "electron-ipc",
          args_preview: { tabId: "tab-1" },
        },
      ],
      providerRequests: [
        {
          path: "/v1/chat/completions",
          body: {
            tools: [{ function: { name: "browser__openTabs" } }],
            messages: [{ role: "tool", tool_call_id: "call-1" }],
          },
        },
        {
          path: "/v1/chat/completions",
          body: { messages: [{ role: "tool", tool_call_id: "call-2" }] },
        },
        {
          path: "/v1/chat/completions",
          body: { messages: [{ role: "tool", tool_call_id: "call-3" }] },
        },
        {
          path: "/v1/chat/completions",
          body: { messages: [{ role: "tool", tool_call_id: "call-4" }] },
        },
        {
          path: "/v1/chat/completions",
          body: { messages: [{ role: "tool", tool_call_id: "call-5" }] },
        },
        {
          path: "/v1/chat/completions",
          body: { messages: [{ role: "tool", tool_call_id: "call-6" }] },
        },
        {
          path: "/v1/chat/completions",
          body: { messages: [{ role: "tool", tool_call_id: "call-7" }] },
        },
        {
          path: "/v1/chat/completions",
          body: { messages: [{ role: "tool", tool_call_id: "call-8" }] },
        },
      ],
      finalText:
        'BROWSER_RUNTIME_GATE_B_DONE:{"activeTurnId":"turn-1","browserSessionId":"browser-session-1","ownerWebContentsId":7,"staleMutationRejected":true,"tabId":"tab-1","threadId":"thread-1","viewId":"view-1","webContentsId":42,"windowId":3}',
    });
    expect(Object.values(assertions).every(Boolean)).toBe(true);
  });
});
