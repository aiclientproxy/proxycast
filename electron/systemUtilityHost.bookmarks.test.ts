import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const {
  invokeNativeHostMock,
  disposeNativeHostMock,
  onEventNativeHostMock,
  nativeEventListeners,
  emitNativeEventMock,
} = vi.hoisted(() => ({
  invokeNativeHostMock: vi.fn(),
  disposeNativeHostMock: vi.fn(),
  nativeEventListeners: [] as Array<
    (event: { event: string; payload?: unknown }) => void
  >,
  emitNativeEventMock: vi.fn(),
  onEventNativeHostMock: vi.fn(
    (listener: (event: { event: string; payload?: unknown }) => void) => {
      nativeEventListeners.push(listener);
      return () => undefined;
    },
  ),
}));

vi.mock("./electronRuntime", () => ({
  app: {
    getVersion: () => "1.0.0-test",
  },
  shell: {
    openExternal: vi.fn(),
  },
}));

vi.mock("./macosNativeHost", () => ({
  MacOSNativeHostClient: class {
    invoke = invokeNativeHostMock;
    dispose = disposeNativeHostMock;
    onEvent = onEventNativeHostMock;
  },
  NativeHostError: class NativeHostError extends Error {
    code: string;
    data: unknown;

    constructor(code: string, message: string, data?: unknown) {
      super(message);
      this.code = code;
      this.data = data;
    }
  },
}));

import { SystemUtilityHost } from "./systemUtilityHost";

const originalPlatform = process.platform;
const tempRoots: string[] = [];

function createHost(
  appDataRoot: string,
  emit?: (event: string, payload?: unknown) => void,
): SystemUtilityHost {
  return new SystemUtilityHost({
    appDataRoot,
    readConfig: async () => ({}),
    emit,
  });
}

beforeEach(() => {
  Object.defineProperty(process, "platform", {
    configurable: true,
    value: "darwin",
  });
  invokeNativeHostMock.mockReset();
  disposeNativeHostMock.mockReset();
  emitNativeEventMock.mockReset();
  nativeEventListeners.splice(0);
});

afterEach(() => {
  Object.defineProperty(process, "platform", {
    configurable: true,
    value: originalPlatform,
  });
  while (tempRoots.length > 0) {
    rmSync(tempRoots.pop()!, { recursive: true, force: true });
  }
});

describe("SystemUtilityHost security-scoped bookmarks", () => {
  it("将 native host unsolicited event 转发到 Electron Host emitter", () => {
    const root = mkdtempSync(path.join(tmpdir(), "lime-bookmark-host-"));
    tempRoots.push(root);
    const host = createHost(root, emitNativeEventMock);

    expect(nativeEventListeners).toHaveLength(1);
    nativeEventListeners[0]?.({
      event: "hidTopology.changed",
      payload: { devices: [] },
    });
    expect(emitNativeEventMock).toHaveBeenCalledWith("hidTopology.changed", {
      devices: [],
    });
    host.dispose();
  });

  it("按稳定 ID 持久化 bookmark.create 结果", async () => {
    const root = mkdtempSync(path.join(tmpdir(), "lime-bookmark-host-"));
    tempRoots.push(root);
    invokeNativeHostMock.mockResolvedValueOnce({
      path: "/Users/coso/Documents/project",
      bookmark: "base64-bookmark",
    });
    const host = createHost(root);

    await expect(
      host.invokeMacOSNativeHost({
        method: "bookmark.create",
        params: {
          path: "/Users/coso/Documents/project",
          persistId: "workspace-main",
        },
      }),
    ).resolves.toMatchObject({
      bookmarkId: "workspace-main",
      persisted: true,
    });

    const recordPath = path.join(
      root,
      "macos/security-scoped-bookmarks/workspace-main.json",
    );
    expect(JSON.parse(readFileSync(recordPath, "utf8"))).toEqual({
      schemaVersion: 1,
      bookmarkId: "workspace-main",
      bookmark: "base64-bookmark",
      path: "/Users/coso/Documents/project",
    });
    expect(invokeNativeHostMock).toHaveBeenCalledWith({
      method: "bookmark.create",
      params: { path: "/Users/coso/Documents/project" },
    });
    host.dispose();
  });

  it("冷启动时可按 ID resolve/start 已持久化 bookmark", async () => {
    const root = mkdtempSync(path.join(tmpdir(), "lime-bookmark-host-"));
    tempRoots.push(root);
    const host = createHost(root);
    invokeNativeHostMock.mockResolvedValueOnce({
      path: "/Users/coso/Documents/project",
      bookmark: "base64-bookmark",
    });
    await host.invokeMacOSNativeHost({
      method: "bookmark.create",
      params: { path: "/Users/coso/Documents/project", persistId: "workspace" },
    });
    invokeNativeHostMock.mockReset();
    invokeNativeHostMock.mockResolvedValueOnce({
      path: "/Users/coso/Documents/project",
      isStale: false,
    });

    await expect(
      createHost(root).invokeMacOSNativeHost({
        method: "bookmark.resolve",
        params: { bookmarkId: "workspace" },
      }),
    ).resolves.toMatchObject({ isStale: false });
    expect(invokeNativeHostMock).toHaveBeenCalledWith({
      method: "bookmark.resolve",
      params: { bookmark: "base64-bookmark" },
    });
  });

  it("revoke 会删除持久化记录且不伪造 helper 调用", async () => {
    const root = mkdtempSync(path.join(tmpdir(), "lime-bookmark-host-"));
    tempRoots.push(root);
    const host = createHost(root);
    invokeNativeHostMock.mockResolvedValueOnce({ bookmark: "bookmark" });
    await host.invokeMacOSNativeHost({
      method: "bookmark.create",
      params: { path: "/tmp/project", persistId: "workspace" },
    });

    await expect(
      host.invokeMacOSNativeHost({
        method: "bookmark.revoke",
        params: { bookmarkId: "workspace" },
      }),
    ).resolves.toEqual({ bookmarkId: "workspace", revoked: true });
    expect(
      existsSync(
        path.join(root, "macos/security-scoped-bookmarks/workspace.json"),
      ),
    ).toBe(false);
    expect(invokeNativeHostMock).toHaveBeenCalledTimes(1);
  });

  it("按稳定 ID start/stop 管理活动 token，revoke 先停止访问再删除记录", async () => {
    const root = mkdtempSync(path.join(tmpdir(), "lime-bookmark-host-"));
    tempRoots.push(root);
    const host = createHost(root);
    invokeNativeHostMock
      .mockResolvedValueOnce({ bookmark: "bookmark" })
      .mockResolvedValueOnce({
        token: "token-1",
        path: "/tmp/project",
        isStale: false,
        started: true,
      })
      .mockResolvedValueOnce({ token: "token-1", stopped: true })
      .mockResolvedValueOnce({
        token: "token-2",
        path: "/tmp/project",
        isStale: false,
        started: true,
      })
      .mockResolvedValueOnce({ token: "token-2", stopped: true });

    await host.invokeMacOSNativeHost({
      method: "bookmark.create",
      params: { path: "/tmp/project", persistId: "workspace" },
    });
    await expect(
      host.invokeMacOSNativeHost({
        method: "bookmark.start",
        params: { bookmarkId: "workspace" },
      }),
    ).resolves.toMatchObject({ bookmarkId: "workspace", token: "token-1" });
    await expect(
      host.invokeMacOSNativeHost({
        method: "bookmark.stop",
        params: { bookmarkId: "workspace" },
      }),
    ).resolves.toEqual({ token: "token-1", stopped: true });
    await host.invokeMacOSNativeHost({
      method: "bookmark.start",
      params: { bookmarkId: "workspace" },
    });
    await expect(
      host.invokeMacOSNativeHost({
        method: "bookmark.revoke",
        params: { bookmarkId: "workspace" },
      }),
    ).resolves.toEqual({ bookmarkId: "workspace", revoked: true });

    expect(invokeNativeHostMock).toHaveBeenNthCalledWith(2, {
      method: "bookmark.start",
      params: { bookmark: "bookmark" },
    });
    expect(invokeNativeHostMock).toHaveBeenNthCalledWith(3, {
      method: "bookmark.stop",
      params: { token: "token-1" },
    });
    expect(invokeNativeHostMock).toHaveBeenNthCalledWith(5, {
      method: "bookmark.stop",
      params: { token: "token-2" },
    });
    expect(
      existsSync(
        path.join(root, "macos/security-scoped-bookmarks/workspace.json"),
      ),
    ).toBe(false);
    host.dispose();
  });

  it("拒绝 bookmark ID 路径穿越和不存在的冷启动记录", async () => {
    const root = mkdtempSync(path.join(tmpdir(), "lime-bookmark-host-"));
    tempRoots.push(root);
    const host = createHost(root);

    await expect(
      host.invokeMacOSNativeHost({
        method: "bookmark.revoke",
        params: { bookmarkId: "../outside" },
      }),
    ).rejects.toThrow(/Bookmark id/);
    await expect(
      host.invokeMacOSNativeHost({
        method: "bookmark.start",
        params: { bookmarkId: "missing" },
      }),
    ).rejects.toMatchObject({ code: "bookmark_unavailable" });
    expect(invokeNativeHostMock).not.toHaveBeenCalled();
  });
});
