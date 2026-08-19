import { describe, expect, it } from "vitest";

import {
  buildAssertions,
  extractBrowserObservation,
  extractBrowserState,
  extractFinalBrowserState,
  parseArgs,
} from "./browser-runtime-electron-gate-b.mjs";

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
    });
    expect(() => parseArgs(["--timeout-ms", "1000"])).toThrow(
      "--timeout-ms 必须 >= 30000",
    );
    expect(() => parseArgs(["--interval-ms", "10"])).toThrow(
      "--interval-ms 必须 >= 100",
    );
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
    expect(extractFinalBrowserState("BROWSER_RUNTIME_GATE_B_DONE:invalid")).toBeNull();
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
        observation: { pageRevision: 1, snapshotId: "snapshot-after" },
        state: {
          activeTurnId: "turn-1",
          browserSessionId: "browser-session-1",
          controlOwner: "agent",
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
