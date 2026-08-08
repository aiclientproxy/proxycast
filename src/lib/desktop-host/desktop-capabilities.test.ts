import { beforeEach, describe, expect, it, vi } from "vitest";
import { getCurrent, getUrls, onOpenUrl } from "./deep-link";
import {
  isRegistered,
  register,
  triggerShortcut,
  unregister,
} from "./global-shortcut";
import { open as openShell } from "./shell";

function clearElectronBridge(): void {
  delete (window as any).electronAPI;
}

describe("desktop-host desktop capabilities", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.unstubAllEnvs();
    clearElectronBridge();
  });

  it("Shell 能力委托 Electron bridge", async () => {
    const shellOpen = vi.fn().mockResolvedValue(undefined);
    (window as any).electronAPI = {
      invoke: vi.fn(),
      shell: { open: shellOpen },
    };

    await openShell("https://example.com");

    expect(shellOpen).toHaveBeenCalledWith("https://example.com", undefined);
  });

  it("生产环境无 Electron shell bridge 时 fail-closed", async () => {
    vi.stubEnv("MODE", "production");
    vi.stubEnv("VITEST", "");

    try {
      await expect(openShell("https://example.com")).rejects.toThrow(
        "shell.open 只能在测试环境使用",
      );
    } finally {
      vi.unstubAllEnvs();
    }
  });

  it("Deep Link 能力委托 Electron bridge", async () => {
    const unlisten = vi.fn();
    const onOpenUrlBridge = vi.fn((handler: (urls: string[]) => void) => {
      handler(["lime://callback"]);
      return unlisten;
    });
    (window as any).electronAPI = {
      invoke: vi.fn(),
      deepLink: {
        onOpenUrl: onOpenUrlBridge,
        getUrls: vi.fn().mockResolvedValue(["lime://callback"]),
        getCurrent: vi.fn().mockResolvedValue(["lime://current"]),
      },
    };
    const handler = vi.fn();

    const resultUnlisten = await onOpenUrl(handler);

    await expect(getUrls()).resolves.toEqual(["lime://callback"]);
    await expect(getCurrent()).resolves.toEqual(["lime://current"]);
    expect(handler).toHaveBeenCalledWith(["lime://callback"]);
    resultUnlisten();
    expect(unlisten).toHaveBeenCalledTimes(1);
  });

  it("生产环境无 Electron deep-link bridge 时 fail-closed", async () => {
    vi.stubEnv("MODE", "production");
    vi.stubEnv("VITEST", "");

    try {
      await expect(getUrls()).rejects.toThrow(
        "deepLink.getUrls 只能在测试环境使用",
      );
      await expect(onOpenUrl(vi.fn())).rejects.toThrow(
        "deepLink.onOpenUrl 只能在测试环境使用",
      );
    } finally {
      vi.unstubAllEnvs();
    }
  });

  it("测试环境允许全局快捷键内存夹具", async () => {
    const handler = vi.fn();

    await register("CommandOrControl+K", handler);
    await expect(isRegistered("CommandOrControl+K")).resolves.toBe(true);
    triggerShortcut("CommandOrControl+K");
    await unregister("CommandOrControl+K");

    expect(handler).toHaveBeenCalledTimes(1);
    await expect(isRegistered("CommandOrControl+K")).resolves.toBe(false);
  });

  it("生产环境无 Electron globalShortcut bridge 时 fail-closed", async () => {
    vi.stubEnv("MODE", "production");
    vi.stubEnv("VITEST", "");

    try {
      await expect(register("CommandOrControl+K", vi.fn())).rejects.toThrow(
        "globalShortcut.register 只能在测试环境使用",
      );
      await expect(isRegistered("CommandOrControl+K")).rejects.toThrow(
        "globalShortcut.isRegistered 只能在测试环境使用",
      );
      expect(() => triggerShortcut("CommandOrControl+K")).toThrow(
        "globalShortcut.triggerShortcut 只能在测试环境使用",
      );
    } finally {
      vi.unstubAllEnvs();
    }
  });
});
