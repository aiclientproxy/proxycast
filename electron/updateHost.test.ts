import { beforeEach, describe, expect, it, vi } from "vitest";

const {
  appState,
  autoUpdaterEmit,
  autoUpdaterListeners,
  checkForUpdatesMock,
  onAutoUpdater,
  onceAutoUpdater,
  quitAndInstallMock,
  removeAutoUpdaterListener,
  setFeedURLMock,
} = vi.hoisted(() => {
  const listeners = new Map<string, Array<(...args: unknown[]) => void>>();
  const removeListener = (
    event: string,
    listener: (...args: unknown[]) => void,
  ) => {
    const current = listeners.get(event) ?? [];
    listeners.set(
      event,
      current.filter((candidate) => candidate !== listener),
    );
  };
  const on = (event: string, listener: (...args: unknown[]) => void) => {
    const current = listeners.get(event) ?? [];
    current.push(listener);
    listeners.set(event, current);
  };
  const once = (event: string, listener: (...args: unknown[]) => void) => {
    const wrapped = (...args: unknown[]) => {
      removeListener(event, wrapped);
      listener(...args);
    };
    on(event, wrapped);
  };
  const emit = (event: string, ...args: unknown[]) => {
    const current = listeners.get(event) ?? [];
    for (const listener of [...current]) {
      listener(...args);
    }
  };
  return {
    appState: {
      isPackaged: false,
    },
    autoUpdaterEmit: emit,
    autoUpdaterListeners: listeners,
    checkForUpdatesMock: vi.fn(),
    onAutoUpdater: on,
    onceAutoUpdater: once,
    quitAndInstallMock: vi.fn(),
    removeAutoUpdaterListener: removeListener,
    setFeedURLMock: vi.fn(),
  };
});

vi.mock("./electronRuntime", () => ({
  app: {
    getVersion: () => "1.60.0",
    get isPackaged() {
      return appState.isPackaged;
    },
  },
  autoUpdater: {
    checkForUpdates: checkForUpdatesMock,
    on: onAutoUpdater,
    once: onceAutoUpdater,
    quitAndInstall: quitAndInstallMock,
    removeListener: removeAutoUpdaterListener,
    setFeedURL: setFeedURLMock,
  },
}));

import { ElectronUpdateHost } from "./updateHost";

function emitDownloadedUpdate(releaseName: string | undefined): void {
  autoUpdaterEmit(
    "update-downloaded",
    {},
    "release notes",
    releaseName,
    new Date("2026-07-17T00:00:00.000Z"),
    "https://updates.limecloud.com/Lime-update.zip",
  );
}

describe("ElectronUpdateHost", () => {
  beforeEach(() => {
    appState.isPackaged = false;
    delete process.env.LIME_ELECTRON_ENABLE_DEV_UPDATER;
    delete process.env.LIME_ELECTRON_E2E;
    delete process.env.LIME_ELECTRON_SMOKE;
    delete process.env.VITE_DEV_SERVER_URL;
    autoUpdaterListeners.clear();
    checkForUpdatesMock.mockReset();
    quitAndInstallMock.mockReset();
    setFeedURLMock.mockReset();
  });

  it("开发 renderer 模式下即使宿主是 .app 也不启用 updater", async () => {
    appState.isPackaged = true;
    process.env.VITE_DEV_SERVER_URL = "http://127.0.0.1:1420";
    const host = new ElectronUpdateHost(vi.fn());

    await expect(host.invoke("check_for_updates")).resolves.toEqual({
      current: "1.60.0",
      hasUpdate: false,
      error: "Electron updater is only enabled for packaged builds.",
    });

    expect(checkForUpdatesMock).not.toHaveBeenCalled();
    expect(setFeedURLMock).not.toHaveBeenCalled();
  });

  it("updater 不可用时自动检查更新应返回本地版本", async () => {
    const host = new ElectronUpdateHost(vi.fn());

    await expect(
      host.invoke("check_for_updates", { automatic: true }),
    ).resolves.toEqual(
      expect.objectContaining({
        current: "1.60.0",
        hasUpdate: false,
        error: null,
      }),
    );

    expect(checkForUpdatesMock).not.toHaveBeenCalled();
    expect(setFeedURLMock).not.toHaveBeenCalled();
  });

  it("Electron GUI smoke 自动化会话不应触发真实 updater", async () => {
    appState.isPackaged = true;
    process.env.LIME_ELECTRON_SMOKE = "1";
    process.env.LIME_ELECTRON_E2E = "1";
    const host = new ElectronUpdateHost(vi.fn());

    await expect(
      host.invoke("check_for_updates", { automatic: true }),
    ).resolves.toEqual(
      expect.objectContaining({
        current: "1.60.0",
        hasUpdate: false,
        error: null,
      }),
    );

    expect(checkForUpdatesMock).not.toHaveBeenCalled();
    expect(setFeedURLMock).not.toHaveBeenCalled();
  });

  it("本地 Lime-dev.app 不应启用真实 updater", async () => {
    const originalExecPath = process.execPath;
    Object.defineProperty(process, "execPath", {
      configurable: true,
      value:
        "/Users/example/project/.lime/electron-dev-host/Lime-dev.app/Contents/MacOS/Lime",
    });
    try {
      appState.isPackaged = true;
      const host = new ElectronUpdateHost(vi.fn());

      await expect(host.invoke("check_for_updates")).resolves.toEqual({
        current: "1.60.0",
        hasUpdate: false,
        error: "Electron updater is only enabled for packaged builds.",
      });

      expect(checkForUpdatesMock).not.toHaveBeenCalled();
      expect(setFeedURLMock).not.toHaveBeenCalled();
    } finally {
      Object.defineProperty(process, "execPath", {
        configurable: true,
        value: originalExecPath,
      });
    }
  });

  it("open_update_window 应把按钮锚点矩形传给更新窗口控制器", async () => {
    const open = vi.fn();
    const host = new ElectronUpdateHost(vi.fn(), {
      open,
      close: vi.fn(),
    });

    await expect(
      host.invoke("open_update_window", {
        anchorRect: { x: 18, y: 816, width: 30, height: 30 },
      }),
    ).resolves.toBeNull();

    expect(open).toHaveBeenCalledWith(
      expect.objectContaining({
        current: "1.60.0",
        anchorRect: { x: 18, y: 816, width: 30, height: 30 },
      }),
    );
  });

  it("自动下载进行中启动安装会话不应再次检查并重复下载", async () => {
    vi.useFakeTimers();
    try {
      appState.isPackaged = true;
      checkForUpdatesMock.mockImplementation(() => {
        autoUpdaterEmit("update-available");
      });
      const host = new ElectronUpdateHost(vi.fn());

      const checkSession = host.invoke("check_for_updates");
      await Promise.resolve();
      expect(checkForUpdatesMock).toHaveBeenCalledTimes(1);
      await expect(host.invoke("get_update_install_session")).resolves.toEqual(
        expect.objectContaining({ stage: "downloading" }),
      );

      const installSession = host.invoke("start_update_install_session");
      emitDownloadedUpdate("1.61.0");

      await expect(checkSession).resolves.toEqual(
        expect.objectContaining({ hasUpdate: true, latest: "1.61.0" }),
      );

      await expect(installSession).resolves.toEqual(
        expect.objectContaining({
          stage: "restarting",
          latestVersion: "1.61.0",
        }),
      );
      expect(checkForUpdatesMock).toHaveBeenCalledTimes(1);
      expect(quitAndInstallMock).not.toHaveBeenCalled();
      await vi.advanceTimersByTimeAsync(250);
      expect(quitAndInstallMock).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("多个自动检查在下载完成前只应触发一次 Electron 检查", async () => {
    appState.isPackaged = true;
    checkForUpdatesMock.mockImplementation(() => {
      autoUpdaterEmit("update-available");
    });
    const host = new ElectronUpdateHost(vi.fn());

    const firstCheck = host.invoke("check_for_updates", { automatic: true });
    const secondCheck = host.invoke("check_for_updates", { automatic: true });
    emitDownloadedUpdate("1.61.0");

    await expect(Promise.all([firstCheck, secondCheck])).resolves.toEqual([
      expect.objectContaining({ hasUpdate: true }),
      expect.objectContaining({ hasUpdate: true }),
    ]);

    expect(checkForUpdatesMock).toHaveBeenCalledTimes(1);
  });

  it("多个安装请求只应触发一次重启安装", async () => {
    vi.useFakeTimers();
    try {
      appState.isPackaged = true;
      checkForUpdatesMock.mockImplementation(() => {
        autoUpdaterEmit("update-available");
      });
      const host = new ElectronUpdateHost(vi.fn());

      const checkSession = host.invoke("check_for_updates");
      const firstInstall = host.invoke("start_update_install_session");
      const secondInstall = host.invoke("start_update_install_session");
      emitDownloadedUpdate("1.61.0");

      await expect(
        Promise.all([checkSession, firstInstall, secondInstall]),
      ).resolves.toEqual([
        expect.objectContaining({ hasUpdate: true }),
        expect.objectContaining({ stage: "restarting" }),
        expect.objectContaining({ stage: "restarting" }),
      ]);
      await vi.advanceTimersByTimeAsync(250);
      expect(quitAndInstallMock).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it.each(["1.60.0", "v1.60.0", "1.59.0"])(
    "候选版本 %s 不高于当前版本时不得进入安装",
    async (candidateVersion) => {
      vi.useFakeTimers();
      try {
        appState.isPackaged = true;
        checkForUpdatesMock.mockImplementation(() => {
          autoUpdaterEmit("update-available");
        });
        const host = new ElectronUpdateHost(vi.fn());

        const checkSession = host.invoke("check_for_updates");
        const installSession = host.invoke("start_update_install_session");
        emitDownloadedUpdate(candidateVersion);

        await expect(checkSession).resolves.toEqual(
          expect.objectContaining({
            hasUpdate: false,
            latest: candidateVersion.replace(/^v/i, ""),
          }),
        );
        await expect(installSession).resolves.toEqual(
          expect.objectContaining({ stage: "up_to_date" }),
        );
        await vi.advanceTimersByTimeAsync(250);
        expect(quitAndInstallMock).not.toHaveBeenCalled();
      } finally {
        vi.useRealTimers();
      }
    },
  );

  it.each([undefined, "latest"])(
    "候选版本 %s 无效时应终止更新且不得安装",
    async (candidateVersion) => {
      vi.useFakeTimers();
      try {
        appState.isPackaged = true;
        checkForUpdatesMock.mockImplementation(() => {
          autoUpdaterEmit("update-available");
        });
        const host = new ElectronUpdateHost(vi.fn());

        const checkSession = host.invoke("check_for_updates");
        const installSession = host.invoke("start_update_install_session");
        emitDownloadedUpdate(candidateVersion);

        await expect(checkSession).resolves.toEqual(
          expect.objectContaining({
            hasUpdate: false,
            error: "更新清单未提供有效的候选版本",
          }),
        );
        await expect(installSession).resolves.toEqual(
          expect.objectContaining({
            stage: "failed",
            latestVersion: null,
          }),
        );
        await vi.advanceTimersByTimeAsync(250);
        expect(quitAndInstallMock).not.toHaveBeenCalled();
      } finally {
        vi.useRealTimers();
      }
    },
  );
});
