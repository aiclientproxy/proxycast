import { beforeEach, describe, expect, it, vi } from "vitest";
import { MockWindow, getCurrentWindow } from "./window";

function clearElectronBridge(): void {
  delete (window as any).electronAPI;
}

describe("desktop-host/window", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.unstubAllEnvs();
    clearElectronBridge();
  });

  it("Electron window bridge 可用时委托真实 bridge", async () => {
    const bridgeWindow = {
      show: vi.fn().mockResolvedValue(undefined),
      hide: vi.fn().mockResolvedValue(undefined),
      close: vi.fn().mockResolvedValue(undefined),
      minimize: vi.fn().mockResolvedValue(undefined),
      maximize: vi.fn().mockResolvedValue(undefined),
      unmaximize: vi.fn().mockResolvedValue(undefined),
      center: vi.fn().mockResolvedValue(undefined),
      setFocus: vi.fn().mockResolvedValue(undefined),
      startDragging: vi.fn().mockResolvedValue(undefined),
      setTitle: vi.fn().mockResolvedValue(undefined),
      setSize: vi.fn().mockResolvedValue(undefined),
      setPosition: vi.fn().mockResolvedValue(undefined),
      isVisible: vi.fn().mockResolvedValue(true),
      isMaximized: vi.fn().mockResolvedValue(false),
      isFullscreen: vi.fn().mockResolvedValue(false),
      isDecorated: vi.fn().mockResolvedValue(true),
      isResizable: vi.fn().mockResolvedValue(true),
    };
    (window as any).electronAPI = {
      invoke: vi.fn(),
      listen: vi.fn(),
      emit: vi.fn(),
      window: bridgeWindow,
    };

    const appWindow = getCurrentWindow();
    await appWindow.show();
    await appWindow.resize(1024, 768);
    await appWindow.setPosition(10, 20);

    await expect(appWindow.isVisible()).resolves.toBe(true);

    expect(bridgeWindow.show).toHaveBeenCalledTimes(1);
    expect(bridgeWindow.setSize).toHaveBeenCalledWith(1024, 768);
    expect(bridgeWindow.setPosition).toHaveBeenCalledWith(10, 20);
    expect(bridgeWindow.isVisible).toHaveBeenCalledTimes(1);
  });

  it("测试环境允许窗口内存夹具", async () => {
    const appWindow = new MockWindow("fixture", { visible: false });

    await appWindow.show();
    await appWindow.maximize();
    await appWindow.setTitle("Fixture");

    await expect(appWindow.isVisible()).resolves.toBe(true);
    await expect(appWindow.isMaximized()).resolves.toBe(true);
    expect(appWindow.options.title).toBe("Fixture");
  });

  it("非测试环境无 Electron window bridge 时 fail-closed", async () => {
    vi.stubEnv("MODE", "production");
    vi.stubEnv("VITEST", "");

    try {
      const appWindow = getCurrentWindow();
      await expect(appWindow.show()).rejects.toThrow(
        "window.show 只能在测试环境使用",
      );
      await expect(appWindow.startDragging()).rejects.toThrow(
        "window.startDragging 只能在测试环境使用",
      );
      await expect(appWindow.isVisible()).rejects.toThrow(
        "window.isVisible 只能在测试环境使用",
      );
    } finally {
      vi.unstubAllEnvs();
    }
  });
});
