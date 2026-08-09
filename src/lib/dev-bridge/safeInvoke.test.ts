import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  listenViaHttpEvent: vi.fn(),
  hasDevBridgeEventListenerCapability: vi.fn(),
  invokeViaHttp: vi.fn(),
  isDevBridgeAvailable: vi.fn(),
  resolveBridgeRequestTimeoutMs: vi.fn(() => 1800),
  normalizeDevBridgeError: vi.fn((cmd: string, error: unknown) => {
    if (error instanceof Error) {
      return new Error(`[${cmd}] ${error.message}`);
    }
    return new Error(`[${cmd}] ${String(error)}`);
  }),
}));

vi.mock("./http-client", () => ({
  hasDevBridgeEventListenerCapability:
    mocks.hasDevBridgeEventListenerCapability,
  invokeViaHttp: mocks.invokeViaHttp,
  isDevBridgeAvailable: mocks.isDevBridgeAvailable,
  listenViaHttpEvent: mocks.listenViaHttpEvent,
  normalizeDevBridgeError: mocks.normalizeDevBridgeError,
  resolveBridgeRequestTimeoutMs: mocks.resolveBridgeRequestTimeoutMs,
}));

vi.mock("./mockPriorityCommands", () => ({
  shouldDisallowMockEventFallbackInBrowser: vi.fn(() => false),
}));

import {
  clearInvokeErrorBuffer,
  clearInvokeTraceBuffer,
  getInvokeErrorBuffer,
  getInvokeTraceBuffer,
  hasNativeDesktopHostEventSupport,
  safeEmit,
  safeInvoke,
  safeListen,
} from "./safeInvoke";
import { shouldDisallowMockEventFallbackInBrowser } from "./mockPriorityCommands";

function clearElectronBridge(): void {
  delete (window as any).electronAPI;
  delete (window as any).__LIME_ELECTRON__;
}

describe("safeInvoke", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.isDevBridgeAvailable.mockReturnValue(true);
    mocks.hasDevBridgeEventListenerCapability.mockReturnValue(false);
    mocks.resolveBridgeRequestTimeoutMs.mockReturnValue(1800);
    window.localStorage.clear();
    clearInvokeErrorBuffer();
    clearInvokeTraceBuffer();
    clearElectronBridge();
  });

  it("无 Electron host 时浏览器开发模式走 HTTP bridge", async () => {
    mocks.invokeViaHttp.mockResolvedValueOnce({ ok: true });

    const result = await safeInvoke("workspace_list");

    expect(result).toEqual({ ok: true });
    expect(mocks.invokeViaHttp).toHaveBeenCalledWith(
      "workspace_list",
      undefined,
    );
    expect(getInvokeTraceBuffer()).toEqual([
      expect.objectContaining({
        command: "workspace_list",
        transport: "http-bridge",
        status: "success",
      }),
    ]);
  });

  it("HTTP bridge 失败时直接抛出规范化错误，不回退 renderer mock", async () => {
    mocks.invokeViaHttp.mockRejectedValueOnce(new Error("Failed to fetch"));

    await expect(safeInvoke("workspace_list")).rejects.toThrow(
      "[workspace_list] Failed to fetch",
    );

    expect(mocks.normalizeDevBridgeError).toHaveBeenCalledWith(
      "workspace_list",
      expect.any(Error),
    );
    expect(getInvokeErrorBuffer()).toEqual([
      expect.objectContaining({
        command: "workspace_list",
        transport: "http-bridge",
      }),
    ]);
    expect(getInvokeTraceBuffer()).toEqual([
      expect.objectContaining({
        command: "workspace_list",
        transport: "http-bridge",
        status: "error",
      }),
    ]);
  });

  it("无 Electron host 且无 HTTP bridge 时 fail-closed", async () => {
    mocks.isDevBridgeAvailable.mockReturnValue(false);

    await expect(safeInvoke("workspace_list")).rejects.toThrow(
      'Desktop Host IPC 不可用，命令 "workspace_list" 无法进入 App Server JSON-RPC 主链',
    );

    expect(mocks.invokeViaHttp).not.toHaveBeenCalled();
    expect(getInvokeTraceBuffer()).toEqual([
      expect.objectContaining({
        command: "workspace_list",
        transport: "unavailable",
        status: "error",
      }),
    ]);
  });

  it("Electron host 声明支持命令时走 electron-ipc", async () => {
    const invoke = vi.fn().mockResolvedValueOnce({ lines: [] });
    (window as any).electronAPI = {
      supportsCommand: (command: string) =>
        command === "app_server_handle_json_lines",
      invoke,
      listen: vi.fn(),
      emit: vi.fn(),
    };

    await expect(
      safeInvoke("app_server_handle_json_lines", {
        request: { lines: [] },
      }),
    ).resolves.toEqual({ lines: [] });

    expect(invoke).toHaveBeenCalledWith("app_server_handle_json_lines", {
      request: { lines: [] },
    });
    expect(mocks.invokeViaHttp).not.toHaveBeenCalled();
    expect(getInvokeTraceBuffer()).toEqual([
      expect.objectContaining({
        command: "app_server_handle_json_lines",
        transport: "electron-ipc",
        status: "success",
      }),
    ]);
  });

  it("Electron host 未声明支持命令时直接失败，不绕到 HTTP bridge", async () => {
    const invoke = vi.fn();
    (window as any).electronAPI = {
      supportsCommand: () => false,
      invoke,
      listen: vi.fn(),
      emit: vi.fn(),
    };

    await expect(safeInvoke("agent_runtime_list_sessions")).rejects.toThrow(
      'Desktop Host 尚未支持命令 "agent_runtime_list_sessions"',
    );

    expect(invoke).not.toHaveBeenCalled();
    expect(mocks.invokeViaHttp).not.toHaveBeenCalled();
    expect(getInvokeTraceBuffer()).toEqual([
      expect.objectContaining({
        command: "agent_runtime_list_sessions",
        transport: "electron-ipc",
        status: "error",
      }),
    ]);
  });

  it("Electron invoke 失败时记录 electron-ipc 错误", async () => {
    (window as any).electronAPI = {
      supportsCommand: () => true,
      invoke: vi.fn().mockRejectedValueOnce(new Error("backend failed")),
      listen: vi.fn(),
      emit: vi.fn(),
    };

    await expect(safeInvoke("workspace_list")).rejects.toThrow(
      "backend failed",
    );

    expect(getInvokeErrorBuffer()).toEqual([
      expect.objectContaining({
        command: "workspace_list",
        transport: "electron-ipc",
        error: "backend failed",
      }),
    ]);
  });

  it("Electron invoke 长时间不返回时按命令超时 fail-closed", async () => {
    vi.useFakeTimers();
    try {
      mocks.resolveBridgeRequestTimeoutMs.mockReturnValueOnce(30000);
      const invoke = vi.fn(() => new Promise<never>(() => {}));
      (window as any).electronAPI = {
        supportsCommand: () => true,
        invoke,
        listen: vi.fn(),
        emit: vi.fn(),
      };

      const invokePromise = safeInvoke("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: 1,
              method: "thread/list",
              params: { limit: 10 },
            }),
          ],
        },
      });
      const timeoutExpectation = expect(invokePromise).rejects.toThrow(
        'Desktop Host IPC 命令 "app_server_handle_json_lines" 在 30000ms 内未返回',
      );

      await vi.advanceTimersByTimeAsync(30000);
      await timeoutExpectation;

      expect(invoke).toHaveBeenCalledWith("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: 1,
              method: "thread/list",
              params: { limit: 10 },
            }),
          ],
        },
      });
      expect(getInvokeErrorBuffer()).toEqual([
        expect.objectContaining({
          command: "app_server_handle_json_lines",
          transport: "electron-ipc",
          error: expect.stringContaining("30000ms"),
        }),
      ]);
      expect(getInvokeTraceBuffer()).toEqual([
        expect.objectContaining({
          command: "app_server_handle_json_lines",
          transport: "electron-ipc",
          status: "error",
          error: expect.stringContaining("30000ms"),
        }),
      ]);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("safeListen / safeEmit", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.isDevBridgeAvailable.mockReturnValue(false);
    mocks.hasDevBridgeEventListenerCapability.mockReturnValue(false);
    clearElectronBridge();
  });

  it("Electron event listen 可用时返回幂等 unlisten", async () => {
    const unlisten = vi.fn();
    const listen = vi.fn().mockResolvedValueOnce(unlisten);
    (window as any).electronAPI = {
      invoke: vi.fn(),
      listen,
      emit: vi.fn(),
    };

    const safeUnlisten = await safeListen("config-changed", vi.fn());

    safeUnlisten();
    safeUnlisten();

    expect(listen).toHaveBeenCalledWith("config-changed", expect.any(Function));
    expect(unlisten).toHaveBeenCalledTimes(1);
  });

  it("Electron 事件桥调用失败时真相事件必须失败", async () => {
    vi.mocked(shouldDisallowMockEventFallbackInBrowser).mockReturnValueOnce(
      true,
    );
    (window as any).electronAPI = {
      invoke: vi.fn(),
      listen: vi.fn().mockRejectedValueOnce(new Error("event bridge failed")),
      emit: vi.fn(),
    };

    await expect(safeListen("agent_stream_session-1", vi.fn())).rejects.toThrow(
      '事件 "agent_stream_session-1" 监听失败',
    );
  });

  it("Electron 事件桥调用失败时非真相事件返回空清理函数", async () => {
    const consoleWarnSpy = vi
      .spyOn(console, "warn")
      .mockImplementation(() => {});
    (window as any).electronAPI = {
      invoke: vi.fn(),
      listen: vi.fn().mockRejectedValueOnce(new Error("event bridge failed")),
      emit: vi.fn(),
    };

    try {
      const unlisten = await safeListen("config-changed", vi.fn());
      expect(typeof unlisten).toBe("function");
      unlisten();
    } finally {
      consoleWarnSpy.mockRestore();
    }
  });

  it("无 Electron 事件桥但 HTTP event bridge 可用时走 HTTP 事件通道", async () => {
    const unlisten = vi.fn();
    mocks.hasDevBridgeEventListenerCapability.mockReturnValue(true);
    mocks.listenViaHttpEvent.mockResolvedValueOnce(unlisten);

    const safeUnlisten = await safeListen("config-changed", vi.fn());

    safeUnlisten();
    safeUnlisten();

    expect(mocks.listenViaHttpEvent).toHaveBeenCalledWith(
      "config-changed",
      expect.any(Function),
    );
    expect(unlisten).toHaveBeenCalledTimes(1);
  });

  it("devBridgeFallback Electron bridge 下事件监听应走 HTTP 事件通道", async () => {
    const unlisten = vi.fn();
    const listen = vi.fn();
    mocks.hasDevBridgeEventListenerCapability.mockReturnValue(true);
    mocks.listenViaHttpEvent.mockResolvedValueOnce(unlisten);
    (window as any).electronAPI = {
      devBridgeFallback: true,
      invoke: vi.fn(),
      listen,
      emit: vi.fn(),
    };

    const safeUnlisten = await safeListen("config-changed", vi.fn());

    safeUnlisten();

    expect(listen).not.toHaveBeenCalled();
    expect(mocks.listenViaHttpEvent).toHaveBeenCalledWith(
      "config-changed",
      expect.any(Function),
    );
    expect(unlisten).toHaveBeenCalledTimes(1);
  });

  it("HTTP event bridge 失败时直接抛错，不回退 mock event", async () => {
    mocks.hasDevBridgeEventListenerCapability.mockReturnValue(true);
    mocks.listenViaHttpEvent.mockRejectedValueOnce(
      new Error("connection failed"),
    );

    await expect(safeListen("config-changed", vi.fn())).rejects.toThrow(
      '事件 "config-changed" 监听失败',
    );
  });

  it("无任何事件桥时 fail-closed", async () => {
    await expect(safeListen("config-changed", vi.fn())).rejects.toThrow(
      '事件 "config-changed" 监听失败',
    );
  });

  it("hasNativeDesktopHostEventSupport 只认 Electron 事件桥", () => {
    expect(hasNativeDesktopHostEventSupport()).toBe(false);

    (window as any).electronAPI = {
      invoke: vi.fn(),
      listen: vi.fn(),
      emit: vi.fn(),
    };

    expect(hasNativeDesktopHostEventSupport()).toBe(true);
  });

  it("safeEmit 只通过 Electron host 发送事件", async () => {
    const emit = vi.fn().mockResolvedValueOnce(undefined);
    (window as any).electronAPI = {
      invoke: vi.fn(),
      listen: vi.fn(),
      emit,
    };

    await safeEmit("config-changed", { ok: true });

    expect(emit).toHaveBeenCalledWith("config-changed", { ok: true });
  });

  it("safeEmit 无 Electron host 时 fail-closed", async () => {
    await expect(safeEmit("config-changed")).rejects.toThrow(
      'Desktop Host IPC 不可用，事件 "config-changed" 无法发送',
    );
  });
});
