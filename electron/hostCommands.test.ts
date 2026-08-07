/* global Buffer, process */
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { createServer } from "node:http";
import os from "node:os";
import path from "node:path";
import { parse as parseYaml } from "yaml";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ElectronHostCommands } from "./hostCommands";
import type { ElectronAppServerHost } from "./appServerHost";
import { SystemUtilityHost } from "./systemUtilityHost";
import { VoiceModelHost } from "./voiceModelHost";

const {
  pluginShellHostGetUiRuntimeStatusMock,
  pluginShellHostLaunchShellMock,
  pluginShellHostSelectDirectoryMock,
  pluginShellHostStartUiRuntimeMock,
  pluginShellHostStopUiRuntimeMock,
  pluginRuntimeTaskHostCancelTaskMock,
  pluginRuntimeTaskHostGetTaskMock,
  pluginRuntimeTaskHostStartTaskMock,
  pluginRuntimeTaskHostSubmitHostResponseMock,
  browserWindowCtorMock,
  browserWindowGetAllWindowsMock,
  fileShellHostGetFileIconDataUrlMock,
  fileShellHostGetFileManagerLocationsMock,
  fileShellHostGetHomeDirMock,
  fileShellHostOpenFilePreviewWindowMock,
  fileShellHostOpenWithDefaultAppMock,
  fileShellHostRevealInFinderMock,
  getFileIconMock,
  getPathMock,
  openExternalMock,
  openPathMock,
  showDesktopNotificationMock,
  showOpenDialogMock,
  showItemInFolderMock,
  openProjectPathWithLocalToolMock,
  projectShellHostDisposeForShutdownMock,
  projectShellHostKillSessionMock,
  projectShellHostResizeSessionMock,
  projectShellHostRunCommandMock,
  projectShellHostStartSessionMock,
  projectShellHostWriteSessionMock,
  systemUtilityHostGetBrowserBackendPolicyMock,
  systemUtilityHostGetBrowserBackendsStatusMock,
  systemUtilityHostGetBrowserConnectorInstallStatusMock,
  systemUtilityHostGetBrowserConnectorSettingsMock,
  systemUtilityHostGetChromeBridgeEndpointInfoMock,
  systemUtilityHostGetChromeBridgeStatusMock,
  systemUtilityHostGetChromeProfileSessionsMock,
  systemUtilityHostGetEnvironmentPreviewMock,
  systemUtilityHostOpenExternalUrlMock,
  systemUtilityHostOpenSystemSettingsUrlMock,
  voiceModelHostDeleteMock,
  voiceModelHostDownloadMock,
  voiceModelHostGetInstallStateMock,
  voiceModelHostListCatalogMock,
  webContentsViewCtorMock,
} = vi.hoisted(() => {
  const loadUrlMock = vi.fn();
  const contentViewAddChildViewMock = vi.fn();
  const contentViewRemoveChildViewMock = vi.fn();
  const showWindowMock = vi.fn();
  const focusWindowMock = vi.fn();
  const browserWindowCtorMock = vi.fn(() => ({
    contentView: {
      addChildView: contentViewAddChildViewMock,
      removeChildView: contentViewRemoveChildViewMock,
    },
    focus: focusWindowMock,
    isDestroyed: () => false,
    loadURL: loadUrlMock,
    off: vi.fn(),
    on: vi.fn(),
    once: vi.fn((event: string, callback: () => void) => {
      if (event === "ready-to-show") {
        callback();
      }
    }),
    show: showWindowMock,
    webContents: {
      getURL: () => "",
    },
  }));
  const webContentsLoadUrlMock = vi.fn();
  const webContentsReloadMock = vi.fn();
  const webContentsDestroyMock = vi.fn();
  const webContentsViewCtorMock = vi.fn(() => ({
    setBackgroundColor: vi.fn(),
    setBounds: vi.fn(),
    setVisible: vi.fn(),
    webContents: {
      destroy: webContentsDestroyMock,
      getTitle: () => "Example",
      getURL: () => "https://example.com/",
      isDestroyed: () => false,
      isLoading: () => false,
      loadURL: webContentsLoadUrlMock,
      navigationHistory: {
        canGoBack: () => false,
        canGoForward: () => false,
        goBack: vi.fn(),
        goForward: vi.fn(),
      },
      on: vi.fn(),
      reload: webContentsReloadMock,
      setWindowOpenHandler: vi.fn(),
    },
  }));
  const browserWindowGetAllWindowsMock = vi.fn((): MockBrowserWindow[] => []);
  return {
    browserWindowCtorMock,
    browserWindowGetAllWindowsMock,
    pluginRuntimeTaskHostCancelTaskMock: vi.fn(),
    pluginRuntimeTaskHostGetTaskMock: vi.fn(),
    pluginRuntimeTaskHostStartTaskMock: vi.fn(),
    pluginRuntimeTaskHostSubmitHostResponseMock: vi.fn(),
    pluginShellHostGetUiRuntimeStatusMock: vi.fn(),
    pluginShellHostLaunchShellMock: vi.fn(),
    pluginShellHostSelectDirectoryMock: vi.fn(),
    pluginShellHostStartUiRuntimeMock: vi.fn(),
    pluginShellHostStopUiRuntimeMock: vi.fn(),
    contentViewAddChildViewMock,
    contentViewRemoveChildViewMock,
    fileShellHostGetFileIconDataUrlMock: vi.fn(),
    fileShellHostGetFileManagerLocationsMock: vi.fn(),
    fileShellHostGetHomeDirMock: vi.fn(),
    fileShellHostOpenFilePreviewWindowMock: vi.fn(),
    fileShellHostOpenWithDefaultAppMock: vi.fn(),
    fileShellHostRevealInFinderMock: vi.fn(),
    getFileIconMock: vi.fn(),
    getPathMock: vi.fn((_name: string) => os.tmpdir()),
    openExternalMock: vi.fn(),
    openPathMock: vi.fn(),
    showDesktopNotificationMock: vi.fn(() => ({ status: "sent" })),
    showOpenDialogMock: vi.fn(),
    showItemInFolderMock: vi.fn(),
    openProjectPathWithLocalToolMock: vi.fn(),
    projectShellHostDisposeForShutdownMock: vi.fn(),
    projectShellHostKillSessionMock: vi.fn(),
    projectShellHostResizeSessionMock: vi.fn(),
    projectShellHostRunCommandMock: vi.fn(),
    projectShellHostStartSessionMock: vi.fn(),
    projectShellHostWriteSessionMock: vi.fn(),
    systemUtilityHostGetBrowserBackendPolicyMock: vi.fn(),
    systemUtilityHostGetBrowserBackendsStatusMock: vi.fn(),
    systemUtilityHostGetBrowserConnectorInstallStatusMock: vi.fn(),
    systemUtilityHostGetBrowserConnectorSettingsMock: vi.fn(),
    systemUtilityHostGetChromeBridgeEndpointInfoMock: vi.fn(),
    systemUtilityHostGetChromeBridgeStatusMock: vi.fn(),
    systemUtilityHostGetChromeProfileSessionsMock: vi.fn(),
    systemUtilityHostGetEnvironmentPreviewMock: vi.fn(),
    systemUtilityHostOpenExternalUrlMock: vi.fn(),
    systemUtilityHostOpenSystemSettingsUrlMock: vi.fn(),
    voiceModelHostDeleteMock: vi.fn(),
    voiceModelHostDownloadMock: vi.fn(),
    voiceModelHostGetInstallStateMock: vi.fn(),
    voiceModelHostListCatalogMock: vi.fn(),
    webContentsDestroyMock,
    webContentsLoadUrlMock,
    webContentsReloadMock,
    webContentsViewCtorMock,
  };
});
const tempDirs: string[] = [];
const TEST_REMOTE_PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/lwcdVwAAAABJRU5ErkJggg==";
type MockBrowserWindow = {
  focus: ReturnType<typeof vi.fn>;
  show: ReturnType<typeof vi.fn>;
  webContents: {
    getURL: () => string;
  };
};
type AppServerRequestMock = (
  method: string,
  params?: unknown,
) => Promise<unknown>;

vi.mock("./electronRuntime", () => ({
  app: {
    getFileIcon: getFileIconMock,
    getAppPath: () => os.tmpdir(),
    getName: () => "Lime",
    getPath: getPathMock,
    getVersion: () => "0.0.0-test",
  },
  BrowserWindow: Object.assign(browserWindowCtorMock, {
    getAllWindows: browserWindowGetAllWindowsMock,
  }),
  dialog: {
    showOpenDialog: showOpenDialogMock,
  },
  shell: {
    openExternal: openExternalMock,
    openPath: openPathMock,
    showItemInFolder: showItemInFolderMock,
  },
  WebContentsView: webContentsViewCtorMock,
}));

vi.mock("./desktopNotificationHost", () => ({
  showDesktopNotification: showDesktopNotificationMock,
}));

vi.mock("./pluginShellHost", () => ({
  PluginShellHost: vi.fn(() => ({
    getUiRuntimeStatus: pluginShellHostGetUiRuntimeStatusMock,
    launchShell: pluginShellHostLaunchShellMock,
    selectDirectory: pluginShellHostSelectDirectoryMock,
    startUiRuntime: pluginShellHostStartUiRuntimeMock,
    stopUiRuntime: pluginShellHostStopUiRuntimeMock,
  })),
}));

vi.mock("./pluginRuntimeTaskHost", () => ({
  PluginRuntimeTaskHost: vi.fn(() => ({
    cancelTask: pluginRuntimeTaskHostCancelTaskMock,
    getTask: pluginRuntimeTaskHostGetTaskMock,
    startTask: pluginRuntimeTaskHostStartTaskMock,
    submitHostResponse: pluginRuntimeTaskHostSubmitHostResponseMock,
  })),
}));

vi.mock("./fileShellHost", () => ({
  FileShellHost: vi.fn(() => ({
    getFileIconDataUrl: fileShellHostGetFileIconDataUrlMock,
    getFileManagerLocations: fileShellHostGetFileManagerLocationsMock,
    getHomeDir: fileShellHostGetHomeDirMock,
    openFilePreviewWindow: fileShellHostOpenFilePreviewWindowMock,
    openWithDefaultApp: fileShellHostOpenWithDefaultAppMock,
    revealInFinder: fileShellHostRevealInFinderMock,
  })),
}));

vi.mock("./projectShellHost", () => ({
  ProjectShellHost: vi.fn(() => ({
    disposeForShutdown: projectShellHostDisposeForShutdownMock,
    killSession: projectShellHostKillSessionMock,
    resizeSession: projectShellHostResizeSessionMock,
    runCommand: projectShellHostRunCommandMock,
    startSession: projectShellHostStartSessionMock,
    writeSession: projectShellHostWriteSessionMock,
  })),
}));

vi.mock("./systemUtilityHost", () => ({
  SystemUtilityHost: vi.fn(() => ({
    getBrowserBackendPolicy: systemUtilityHostGetBrowserBackendPolicyMock,
    getBrowserBackendsStatus: systemUtilityHostGetBrowserBackendsStatusMock,
    getBrowserConnectorInstallStatus:
      systemUtilityHostGetBrowserConnectorInstallStatusMock,
    getBrowserConnectorSettings:
      systemUtilityHostGetBrowserConnectorSettingsMock,
    getChromeBridgeEndpointInfo:
      systemUtilityHostGetChromeBridgeEndpointInfoMock,
    getChromeBridgeStatus: systemUtilityHostGetChromeBridgeStatusMock,
    getChromeProfileSessions: systemUtilityHostGetChromeProfileSessionsMock,
    getEnvironmentPreview: systemUtilityHostGetEnvironmentPreviewMock,
    openExternalUrl: systemUtilityHostOpenExternalUrlMock,
    openSystemSettingsUrl: systemUtilityHostOpenSystemSettingsUrlMock,
  })),
}));

vi.mock("./voiceModelHost", () => ({
  VoiceModelHost: vi.fn(() => ({
    delete: voiceModelHostDeleteMock,
    download: voiceModelHostDownloadMock,
    getInstallState: voiceModelHostGetInstallStateMock,
    listCatalog: voiceModelHostListCatalogMock,
  })),
}));

vi.mock("./projectToolsHost", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./projectToolsHost")>();
  return {
    ...actual,
    openProjectPathWithLocalTool: openProjectPathWithLocalToolMock,
  };
});

function createHost(
  userDataDir: string,
  emit: (event: string, payload?: unknown) => void = () => undefined,
  request: AppServerRequestMock = async () => {
    throw new Error("App Server should not be called");
  },
  appDataRoot = userDataDir,
) {
  const appServerHost = {
    request,
  } as unknown as ElectronAppServerHost;
  return new ElectronHostCommands(
    appServerHost,
    userDataDir,
    emit,
    appDataRoot,
  );
}

async function createTempUserDataDir() {
  const dir = await mkdtemp(path.join(os.tmpdir(), "lime-host-commands-"));
  tempDirs.push(dir);
  return dir;
}

async function withRemotePngServer<T>(
  run: (url: string) => Promise<T>,
): Promise<T> {
  const pngBytes = Buffer.from(TEST_REMOTE_PNG_BASE64, "base64");
  const server = createServer((request, response) => {
    if (request.url === "/hero.png") {
      response.writeHead(200, { "content-type": "image/png" });
      response.end(pngBytes);
      return;
    }
    response.writeHead(404);
    response.end();
  });
  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    server.close();
    throw new Error("无法启动远程图片测试服务");
  }
  try {
    return await run(`http://127.0.0.1:${address.port}/hero.png`);
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()));
    });
  }
}

function buildPluginShellDescriptor(): Record<string, unknown> {
  return {
    descriptorVersion: 1,
    appId: "content-factory-app",
    packageHash: "package-fnv1a-current",
    manifestHash: "manifest-fnv1a-current",
    installMode: "standalone",
    runtimeProfile: {
      shellKind: "app_shell",
      installMode: "standalone",
    },
    isolation: {
      packageMount: "read-only",
      secrets: "refs-only",
      sideEffects: "runtime-broker",
      evidence: "runtime-provenance",
    },
    entry: {
      entryKey: "dashboard",
    },
    branding: {
      name: "Content Factory",
      windowTitle: "Content Factory",
    },
  };
}

afterEach(async () => {
  vi.useRealTimers();
  vi.clearAllMocks();
  browserWindowGetAllWindowsMock.mockReturnValue([]);
  getPathMock.mockImplementation(() => os.tmpdir());
  showDesktopNotificationMock.mockReturnValue({ status: "sent" });
  showOpenDialogMock.mockResolvedValue({ canceled: true, filePaths: [] });
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir) {
      await rm(dir, { recursive: true, force: true });
    }
  }
});

describe("ElectronHostCommands retired file browser facade", () => {
  it.each(["list_dir", "read_file_preview_cmd"])(
    "%s 不再作为 Electron Host compat facade 暴露",
    async (command) => {
      const userDataDir = await createTempUserDataDir();
      const host = createHost(userDataDir);

      await expect(
        host.invoke(command, { path: "/workspace" }),
      ).rejects.toThrow(`Electron host command is not implemented: ${command}`);
    },
  );
});

describe("ElectronHostCommands retired automation facade", () => {
  it.each([
    "get_automation_scheduler_config",
    "get_automation_status",
    "get_automation_health",
    "get_automation_jobs",
  ])("%s 不再作为 Electron Host compat facade 暴露", async (command) => {
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    await expect(host.invoke(command, {})).rejects.toThrow(
      `Electron host command is not implemented: ${command}`,
    );
  });
});

describe("ElectronHostCommands retired API Key Provider facade", () => {
  it.each([
    "get_api_key_providers",
    "get_system_provider_catalog",
    "get_provider_ui_state",
    "set_provider_ui_state",
    "fetch_provider_models_auto",
  ])("%s 不再作为 Electron Host provider facade 暴露", async (command) => {
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    await expect(host.invoke(command, {})).rejects.toThrow(
      `Electron host command is not implemented: ${command}`,
    );
  });
});

describe("ElectronHostCommands frontend debug logging", () => {
  it("report_frontend_debug_log 通过 Host 日志通道写入并返回 null", async () => {
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    await expect(
      host.invoke("report_frontend_debug_log", {
        report: {
          level: "debug",
          message: "AgentChatPage.loadData.start",
        },
      }),
    ).resolves.toBeNull();

    expect(logSpy).toHaveBeenCalledWith(
      "[electron-renderer:debug] AgentChatPage.loadData.start",
    );
  });

  it("report_frontend_debug_log 忽略已关闭 stdout 管道的 EPIPE", async () => {
    vi.spyOn(console, "log").mockImplementation(() => {
      const error = new Error("write EPIPE") as Error & { code: string };
      error.code = "EPIPE";
      throw error;
    });
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    await expect(
      host.invoke("report_frontend_debug_log", {
        report: {
          level: "info",
          message: "renderer debug after parent pipe closed",
        },
      }),
    ).resolves.toBeNull();
  });

  it("report_frontend_debug_log 不吞掉非管道类日志错误", async () => {
    vi.spyOn(console, "log").mockImplementation(() => {
      throw new Error("unexpected console failure");
    });
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    await expect(
      host.invoke("report_frontend_debug_log", {
        report: {
          message: "renderer debug",
        },
      }),
    ).rejects.toThrow("unexpected console failure");
  });

  it("report_frontend_debug_log 处理 stdout 异步 EPIPE 事件时不触发 uncaughtException", async () => {
    const uncaughtExceptionSpy = vi.fn();
    process.once("uncaughtException", uncaughtExceptionSpy);

    process.stdout.emit(
      "error",
      Object.assign(new Error("write EPIPE"), {
        code: "EPIPE",
      }),
    );
    await Promise.resolve();

    process.removeListener("uncaughtException", uncaughtExceptionSpy);
    expect(uncaughtExceptionSpy).not.toHaveBeenCalled();
  });

  it("report_frontend_crash 忽略已关闭 stderr 管道的 EPIPE", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {
      const error = new Error("write EPIPE") as Error & { code: string };
      error.code = "EPIPE";
      throw error;
    });
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    await expect(
      host.invoke("report_frontend_crash", {
        report: {
          message: "renderer crashed after parent pipe closed",
        },
      }),
    ).resolves.toEqual({ success: true });
  });
});

describe("ElectronHostCommands local file shell facade", () => {
  it("save_exported_document 通过 Electron Host 写入用户选择的本地路径", async () => {
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);
    const targetPath = path.join(userDataDir, "exports", "report.md");

    await expect(
      host.invoke("save_exported_document", {
        filePath: targetPath,
        content: "# Report\n\n正文",
      }),
    ).resolves.toBeNull();

    await expect(readFile(targetPath, "utf8")).resolves.toBe(
      "# Report\n\n正文",
    );
    const directoryStats = await stat(path.dirname(targetPath));
    expect(directoryStats.isDirectory()).toBe(true);
  });

  it("save_exported_document 允许写入空内容但拒绝缺失 content", async () => {
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);
    const targetPath = path.join(userDataDir, "exports", "empty.md");

    await expect(
      host.invoke("save_exported_document", {
        filePath: targetPath,
        content: "",
      }),
    ).resolves.toBeNull();
    await expect(readFile(targetPath, "utf8")).resolves.toBe("");

    await expect(
      host.invoke("save_exported_document", {
        filePath: targetPath,
      }),
    ).rejects.toThrow("Missing required string field: content");
  });

  it("save_layered_design_project_export 通过 Electron Host 写入工程目录", async () => {
    const projectRootPath = await createTempUserDataDir();
    const host = createHost(projectRootPath);

    const result = await host.invoke("save_layered_design_project_export", {
      request: {
        projectRootPath,
        documentId: "doc-1",
        title: "Design Test",
        directoryName: "Design Test.layered-design",
        files: [
          {
            relativePath: "design.json",
            mimeType: "application/json",
            encoding: "utf8",
            content: '{"layers":[]}',
          },
          {
            relativePath: "export-manifest.json",
            mimeType: "application/json",
            encoding: "utf8",
            content: '{"assets":[]}',
          },
          {
            relativePath: "psd-like-manifest.json",
            mimeType: "application/json",
            encoding: "utf8",
            content: '{"groups":[]}',
          },
          {
            relativePath: "preview.png",
            mimeType: "image/png",
            encoding: "base64",
            content: Buffer.from("preview").toString("base64"),
          },
          {
            relativePath: "assets/subject.png",
            mimeType: "image/png",
            encoding: "base64",
            content: Buffer.from("asset").toString("base64"),
          },
        ],
      },
    });

    expect(result).toEqual(
      expect.objectContaining({
        projectRootPath,
        exportDirectoryRelativePath:
          ".lime/layered-designs/design-test.layered-design",
        assetCount: 1,
        fileCount: 5,
        remoteReferenceAssetCount: 0,
        cachedRemoteAssetCount: 0,
        uncachedRemoteAssetCount: 0,
      }),
    );
    const exportResult = result as { designPath: string; manifestPath: string };
    await expect(readFile(exportResult.designPath, "utf8")).resolves.toBe(
      '{"layers":[]}',
    );
    await expect(readFile(exportResult.manifestPath, "utf8")).resolves.toBe(
      '{"assets":[]}',
    );
  });

  it("read_layered_design_project_export 读回已保存的工程文档", async () => {
    const projectRootPath = await createTempUserDataDir();
    const host = createHost(projectRootPath);
    await host.invoke("save_layered_design_project_export", {
      request: {
        projectRootPath,
        documentId: "doc-1",
        title: "Design Test",
        directoryName: "Design Test.layered-design",
        files: [
          {
            relativePath: "design.json",
            mimeType: "application/json",
            encoding: "utf8",
            content: '{"layers":[{"id":"hero"}]}',
          },
          {
            relativePath: "export-manifest.json",
            mimeType: "application/json",
            encoding: "utf8",
            content: '{"assets":[]}',
          },
        ],
      },
    });

    await expect(
      host.invoke("read_layered_design_project_export", {
        request: {
          projectRootPath,
          exportDirectoryRelativePath:
            ".lime/layered-designs/design-test.layered-design",
        },
      }),
    ).resolves.toEqual(
      expect.objectContaining({
        projectRootPath,
        exportDirectoryRelativePath:
          ".lime/layered-designs/design-test.layered-design",
        designJson: '{"layers":[{"id":"hero"}]}',
        manifestJson: '{"assets":[]}',
        assetCount: 0,
        fileCount: 2,
      }),
    );
  });

  it("save_layered_design_project_export 缓存远程资产并在读回时水合 design.json", async () => {
    await withRemotePngServer(async (remoteAssetUrl) => {
      const projectRootPath = await createTempUserDataDir();
      const host = createHost(projectRootPath);

      const result = await host.invoke("save_layered_design_project_export", {
        request: {
          projectRootPath,
          documentId: "remote-design",
          title: "Remote Design",
          directoryName: "Remote Design.layered-design",
          files: [
            {
              relativePath: "design.json",
              mimeType: "application/json",
              encoding: "utf8",
              content: JSON.stringify({
                id: "remote-design",
                assets: [
                  {
                    id: "remote-asset",
                    kind: "subject",
                    src: remoteAssetUrl,
                    width: 512,
                    height: 512,
                  },
                ],
              }),
            },
            {
              relativePath: "export-manifest.json",
              mimeType: "application/json",
              encoding: "utf8",
              content: JSON.stringify({
                assets: [
                  {
                    id: "remote-asset",
                    kind: "subject",
                    source: "reference",
                    originalSrc: remoteAssetUrl,
                  },
                ],
              }),
            },
            {
              relativePath: "psd-like-manifest.json",
              mimeType: "application/json",
              encoding: "utf8",
              content: JSON.stringify({
                layers: [
                  {
                    id: "remote-layer",
                    asset: {
                      id: "remote-asset",
                      source: "reference",
                      originalSrc: remoteAssetUrl,
                    },
                  },
                ],
              }),
            },
          ],
        },
      });

      expect(result).toEqual(
        expect.objectContaining({
          remoteReferenceAssetCount: 1,
          cachedRemoteAssetCount: 1,
          uncachedRemoteAssetCount: 0,
          assetCount: 1,
          fileCount: 4,
        }),
      );
      const exportResult = result as {
        exportDirectoryPath: string;
        manifestPath: string;
      };
      const manifest = JSON.parse(
        await readFile(exportResult.manifestPath, "utf8"),
      ) as { assets: Array<Record<string, unknown>> };
      expect(manifest.assets[0]).toMatchObject({
        source: "file",
        filename: "assets/remote-asset.png",
        originalSrc: remoteAssetUrl,
      });
      await expect(
        readFile(
          path.join(
            exportResult.exportDirectoryPath,
            "assets/remote-asset.png",
          ),
        ),
      ).resolves.toEqual(Buffer.from(TEST_REMOTE_PNG_BASE64, "base64"));

      await expect(
        host.invoke("read_layered_design_project_export", {
          request: {
            projectRootPath,
            exportDirectoryRelativePath:
              ".lime/layered-designs/remote-design.layered-design",
          },
        }),
      ).resolves.toEqual(
        expect.objectContaining({
          designJson: expect.stringContaining("data:image/png;base64,"),
          psdLikeManifestJson: expect.stringContaining(
            '"filename": "assets/remote-asset.png"',
          ),
        }),
      );
    });
  });

  it("save_layered_design_project_export 拒绝目录穿越", async () => {
    const projectRootPath = await createTempUserDataDir();
    const host = createHost(projectRootPath);

    await expect(
      host.invoke("save_layered_design_project_export", {
        request: {
          projectRootPath,
          documentId: "doc-1",
          title: "Design Test",
          files: [
            {
              relativePath: "../design.json",
              mimeType: "application/json",
              encoding: "utf8",
              content: "{}",
            },
            {
              relativePath: "export-manifest.json",
              mimeType: "application/json",
              encoding: "utf8",
              content: "{}",
            },
          ],
        },
      }),
    ).rejects.toThrow("导出文件路径不能包含目录穿越或根路径");
  });

  it("Layered Design extraction current 命令返回合法 unsupported fallback", async () => {
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    await expect(
      host.invoke("recognize_layered_design_text", {
        request: {
          imageSrc: "data:image/png;base64,ZmFrZQ==",
          width: 640,
          height: 180,
        },
      }),
    ).resolves.toEqual({
      supported: false,
      engine: "electron_host_unsupported",
      blocks: [],
      message: "Electron Host 尚未接入 native OCR provider",
    });

    await expect(
      host.invoke("analyze_layered_design_flat_image", {
        request: {
          image: {
            src: "data:image/png;base64,ZmFrZQ==",
            width: 900,
            height: 1400,
            mimeType: "image/png",
          },
        },
      }),
    ).resolves.toEqual({
      supported: false,
      engine: "electron_host_unsupported",
      message: "Electron Host 尚未接入 native structured analyzer provider",
    });
  });

  it("Plugin shell 命令应只分发到 PluginShellHost", async () => {
    pluginShellHostSelectDirectoryMock.mockResolvedValueOnce({
      path: "/tmp/plugin",
      cancelled: false,
    });
    pluginShellHostLaunchShellMock.mockResolvedValueOnce({
      status: "launched",
      devShell: true,
      blockerCodes: [],
      launchedAt: "2026-05-15T00:00:00.000Z",
    });
    pluginShellHostStartUiRuntimeMock.mockResolvedValueOnce({
      appId: "content-factory-app",
      status: "running",
    });
    pluginShellHostGetUiRuntimeStatusMock.mockResolvedValueOnce({
      appId: "content-factory-app",
      status: "running",
    });
    pluginShellHostStopUiRuntimeMock.mockResolvedValueOnce({
      appId: "content-factory-app",
      status: "stopped",
    });
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    const selectArgs = { request: { title: "选择应用目录" } };
    await expect(
      host.invoke("plugin_select_directory", selectArgs),
    ).resolves.toEqual({
      path: "/tmp/plugin",
      cancelled: false,
    });

    const launchArgs = {
      request: {
        descriptor: buildPluginShellDescriptor(),
      },
    };
    await expect(
      host.invoke("plugin_launch_shell", launchArgs),
    ).resolves.toMatchObject({
      status: "launched",
      devShell: true,
    });

    const runtimeArgs = {
      request: {
        appId: "content-factory-app",
        entryKey: "dashboard",
      },
    };
    await expect(
      host.invoke("plugin_start_ui_runtime", runtimeArgs),
    ).resolves.toMatchObject({ status: "running" });
    await expect(
      host.invoke("plugin_get_ui_runtime_status", runtimeArgs),
    ).resolves.toMatchObject({ status: "running" });
    await expect(
      host.invoke("plugin_stop_ui_runtime", runtimeArgs),
    ).resolves.toMatchObject({ status: "stopped" });

    expect(pluginShellHostSelectDirectoryMock).toHaveBeenCalledWith(selectArgs);
    expect(pluginShellHostLaunchShellMock).toHaveBeenCalledWith(launchArgs);
    expect(pluginShellHostStartUiRuntimeMock).toHaveBeenCalledWith(runtimeArgs);
    expect(pluginShellHostGetUiRuntimeStatusMock).toHaveBeenCalledWith(
      runtimeArgs,
    );
    expect(pluginShellHostStopUiRuntimeMock).toHaveBeenCalledWith(runtimeArgs);
  });

  it("File Shell 命令应只分发到 FileShellHost", async () => {
    fileShellHostRevealInFinderMock.mockReturnValueOnce({});
    fileShellHostOpenWithDefaultAppMock.mockResolvedValueOnce({});
    fileShellHostOpenFilePreviewWindowMock.mockResolvedValueOnce({
      opened: true,
      reused: false,
      url: "file:///tmp/demo.html",
      title: "Demo",
    });
    fileShellHostGetFileIconDataUrlMock.mockResolvedValueOnce(
      "data:image/png;base64,abc",
    );
    fileShellHostGetHomeDirMock.mockReturnValueOnce("/Users/demo");
    fileShellHostGetFileManagerLocationsMock.mockResolvedValueOnce([
      {
        id: "home",
        label: "个人",
        path: "/Users/demo",
        kind: "home",
      },
    ]);
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    const revealArgs = { path: "/tmp/demo.txt" };
    await expect(host.invoke("reveal_in_finder", revealArgs)).resolves.toEqual(
      {},
    );

    const openArgs = { path: "/tmp/demo.txt" };
    await expect(
      host.invoke("open_with_default_app", openArgs),
    ).resolves.toEqual({});

    const previewArgs = {
      path: "/tmp/demo.html",
      title: "Demo",
    };
    await expect(
      host.invoke("open_file_preview_window", previewArgs),
    ).resolves.toEqual({
      opened: true,
      reused: false,
      url: "file:///tmp/demo.html",
      title: "Demo",
    });

    const resourceManagerResult = await host.invoke(
      "open_resource_manager_window",
      {
        sessionId: "resource-session-1",
      },
    );
    expect(resourceManagerResult).toEqual(
      expect.objectContaining({
        opened: true,
        reused: false,
      }),
    );
    expect(String((resourceManagerResult as { url: string }).url)).toContain(
      "lime_window=resource-manager",
    );
    expect(String((resourceManagerResult as { url: string }).url)).toContain(
      "session=resource-session-1",
    );
    expect(browserWindowCtorMock).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Lime 资源管理器",
        width: 1240,
        show: false,
      }),
    );

    await expect(
      host.invoke("get_file_icon_data_url", { path: "/Applications/Lime.app" }),
    ).resolves.toBe("data:image/png;base64,abc");

    await expect(host.invoke("get_home_dir")).resolves.toBe("/Users/demo");
    await expect(host.invoke("get_file_manager_locations")).resolves.toEqual([
      {
        id: "home",
        label: "个人",
        path: "/Users/demo",
        kind: "home",
      },
    ]);

    expect(fileShellHostRevealInFinderMock).toHaveBeenCalledWith(revealArgs);
    expect(fileShellHostOpenWithDefaultAppMock).toHaveBeenCalledWith(openArgs);
    expect(fileShellHostOpenFilePreviewWindowMock).toHaveBeenCalledWith(
      previewArgs,
    );
    expect(fileShellHostGetFileIconDataUrlMock).toHaveBeenCalledWith({
      path: "/Applications/Lime.app",
    });
    expect(fileShellHostGetHomeDirMock).toHaveBeenCalledOnce();
    expect(fileShellHostGetFileManagerLocationsMock).toHaveBeenCalledOnce();
  });

  it("open_project_path_with_tool 应按工具类型走 Electron shell 或本地工具封装", async () => {
    openPathMock.mockResolvedValueOnce("");
    openProjectPathWithLocalToolMock.mockResolvedValue(undefined);
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    await expect(
      host.invoke("open_project_path_with_tool", {
        rootPath: "/tmp/project",
        tool: "finder",
      }),
    ).resolves.toEqual({});
    await expect(
      host.invoke("open_project_path_with_tool", {
        rootPath: "/tmp/project",
        tool: "terminal",
      }),
    ).resolves.toEqual({});

    expect(openPathMock).toHaveBeenCalledWith("/tmp/project");
    expect(openProjectPathWithLocalToolMock).toHaveBeenCalledWith(
      "/tmp/project",
      "terminal",
    );
  });

  it("项目 Shell 命令应只分发到 ProjectShellHost", async () => {
    projectShellHostRunCommandMock.mockResolvedValueOnce({
      command: "pwd",
      exitCode: 0,
    });
    projectShellHostStartSessionMock.mockResolvedValueOnce({
      sessionId: "project-shell-1",
      tty: true,
    });
    projectShellHostWriteSessionMock.mockResolvedValueOnce({});
    projectShellHostResizeSessionMock.mockResolvedValueOnce({});
    projectShellHostKillSessionMock.mockResolvedValueOnce({});
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    const runArgs = {
      rootPath: "/tmp/project",
      command: " pwd ",
      timeoutMs: 10,
    };
    await expect(
      host.invoke("run_project_shell_command", runArgs),
    ).resolves.toMatchObject({ command: "pwd", exitCode: 0 });

    const startArgs = {
      rootPath: "/tmp/project",
      cols: 120,
      rows: 14,
    };
    await expect(
      host.invoke("project_shell_session_start", startArgs),
    ).resolves.toMatchObject({
      sessionId: "project-shell-1",
      tty: true,
    });
    const writeArgs = {
      sessionId: "project-shell-1",
      data: "ls\r",
    };
    await expect(
      host.invoke("project_shell_session_write", writeArgs),
    ).resolves.toEqual({});
    const resizeArgs = {
      sessionId: "project-shell-1",
      cols: 100,
      rows: 20,
    };
    await expect(
      host.invoke("project_shell_session_resize", resizeArgs),
    ).resolves.toEqual({});
    const killArgs = {
      sessionId: "project-shell-1",
    };
    await expect(
      host.invoke("project_shell_session_kill", killArgs),
    ).resolves.toEqual({});

    expect(projectShellHostRunCommandMock).toHaveBeenCalledWith(runArgs);
    expect(projectShellHostStartSessionMock).toHaveBeenCalledWith(startArgs);
    expect(projectShellHostWriteSessionMock).toHaveBeenCalledWith(writeArgs);
    expect(projectShellHostResizeSessionMock).toHaveBeenCalledWith(resizeArgs);
    expect(projectShellHostKillSessionMock).toHaveBeenCalledWith(killArgs);
  });

  it("disposeProjectShellSessionsForShutdown 应委托 ProjectShellHost", async () => {
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    host.disposeProjectShellSessionsForShutdown();

    expect(projectShellHostDisposeForShutdownMock).toHaveBeenCalledOnce();
  });
});

describe("ElectronHostCommands app config persistence", () => {
  it("Host 配置保留 userData，机器资产显式接收 AppDataRoot", async () => {
    const userDataDir = await createTempUserDataDir();
    const appDataRoot = path.join(userDataDir, "machine-data");

    createHost(
      userDataDir,
      () => undefined,
      async () => {
        throw new Error("App Server should not be called");
      },
      appDataRoot,
    );

    expect(SystemUtilityHost).toHaveBeenCalledWith({
      appDataRoot,
      readConfig: expect.any(Function),
    });
    expect(VoiceModelHost).toHaveBeenCalledWith(
      appDataRoot,
      expect.any(Function),
    );
  });

  it("save_config 应只写入 App Server current config.yaml", async () => {
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    await expect(
      host.invoke("save_config", {
        config: {
          default_provider: "anthropic",
          workspace_preferences: {
            media_defaults: {
              image: {
                preferredProviderId: "relay-openai",
                preferredModelId: "gpt-images-2",
                allowFallback: true,
              },
            },
          },
        },
      }),
    ).resolves.toBeNull();

    const yamlConfig = parseYaml(
      await readFile(path.join(userDataDir, "config.yaml"), "utf8"),
    ) as Record<string, unknown>;

    expect(yamlConfig).toMatchObject({
      workspace_preferences: {
        media_defaults: {
          image: {
            preferredProviderId: "relay-openai",
            preferredModelId: "gpt-images-2",
            allowFallback: true,
          },
        },
      },
    });
    await expect(stat(path.join(userDataDir, "config.json"))).rejects.toThrow();
  });

  it("save_config 写入微信模型配置时不应生成重复 YAML 键", async () => {
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    await expect(
      host.invoke("save_config", {
        config: {
          channels: {
            wechat: {
              enabled: false,
              default_model: "lime-hub/claude-sonnet-4-6",
              dm_policy: "pairing",
              group_policy: "allowlist",
              streaming: "off",
              reply_to_mode: "off",
            },
          },
        },
      }),
    ).resolves.toBeNull();

    const yamlContent = await readFile(
      path.join(userDataDir, "config.yaml"),
      "utf8",
    );
    const parsedConfig = parseYaml(yamlContent) as Record<string, unknown>;

    expect(parsedConfig).toMatchObject({
      channels: {
        wechat: {
          default_model: "lime-hub/claude-sonnet-4-6",
          streaming: "off",
          reply_to_mode: "off",
        },
      },
    });
    expect(yamlContent.match(/^\s{4}streaming:/gm)).toHaveLength(1);
    expect(yamlContent.match(/^\s{4}reply_to_mode:/gm)).toHaveLength(1);
  });

  it("get_config 应读取 App Server current config.yaml", async () => {
    const userDataDir = await createTempUserDataDir();
    await mkdir(userDataDir, { recursive: true });
    await writeFile(
      path.join(userDataDir, "config.yaml"),
      [
        "default_provider: yaml-provider",
        "workspace_preferences:",
        "  media_defaults:",
        "    image:",
        "      preferredProviderId: relay-openai",
        "      preferredModelId: gpt-images-2",
      ].join("\n"),
      "utf8",
    );

    await expect(
      createHost(userDataDir).invoke("get_config"),
    ).resolves.toMatchObject({
      default_provider: "yaml-provider",
      workspace_preferences: {
        media_defaults: {
          image: {
            preferredProviderId: "relay-openai",
            preferredModelId: "gpt-images-2",
          },
        },
      },
    });
  });

  it("get_config 不再读取旧 config.json", async () => {
    const userDataDir = await createTempUserDataDir();
    await mkdir(userDataDir, { recursive: true });
    await writeFile(
      path.join(userDataDir, "config.json"),
      JSON.stringify({ default_provider: "legacy-json-provider" }, null, 2),
      "utf8",
    );

    await expect(
      createHost(userDataDir).invoke("get_config"),
    ).resolves.toMatchObject({
      default_provider: "openai",
    });
  });
});

describe("ElectronHostCommands experimental config", () => {
  it("默认读取关闭的 WebMCP 预留配置", async () => {
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    await expect(host.invoke("get_experimental_config")).resolves.toEqual({
      webmcp: { enabled: false },
    });
  });

  it("保存实验配置时合并完整配置并保留未知实验字段", async () => {
    const userDataDir = await createTempUserDataDir();
    await mkdir(userDataDir, { recursive: true });
    await createHost(userDataDir).invoke("save_config", {
      config: {
        default_provider: "anthropic",
        experimental: {
          webmcp: { enabled: false },
          update_check: { enabled: true },
        },
      },
    });

    const host = createHost(userDataDir);
    await expect(
      host.invoke("save_experimental_config", {
        experimentalConfig: {
          webmcp: { enabled: true },
          update_check: { enabled: true },
        },
      }),
    ).resolves.toBeNull();

    await expect(host.invoke("get_experimental_config")).resolves.toEqual({
      webmcp: { enabled: true },
      update_check: { enabled: true },
    });
    const savedConfig = parseYaml(
      await readFile(path.join(userDataDir, "config.yaml"), "utf8"),
    ) as Record<string, unknown>;
    expect(savedConfig.default_provider).toBe("anthropic");
  });
});

describe("ElectronHostCommands retired MCP legacy facade", () => {
  it.each([
    "get_mcp_servers",
    "mcp_list_servers_with_status",
    "mcp_list_tools",
    "mcp_list_prompts",
    "mcp_list_resources",
  ])(
    "%s 已从 Electron Host 退场，生产只能走 App Server MCP current API",
    async (command) => {
      const userDataDir = await createTempUserDataDir();
      const request = vi.fn();
      const host = createHost(userDataDir, () => undefined, request);

      await expect(host.invoke(command)).rejects.toThrow(
        `Electron host command is not implemented: ${command}`,
      );
      expect(request).not.toHaveBeenCalled();
    },
  );
});

describe("ElectronHostCommands retired Knowledge legacy facade", () => {
  it.each([
    "knowledge_list_packs",
    "knowledge_get_pack",
    "knowledge_import_source",
    "knowledge_compile_pack",
    "knowledge_set_default_pack",
    "knowledge_update_pack_status",
    "knowledge_resolve_context",
    "knowledge_validate_context_run",
  ])(
    "%s 已从 Electron Host 退场，生产只能走 App Server JSONL current",
    async (command) => {
      const userDataDir = await createTempUserDataDir();
      const request = vi.fn();
      const host = createHost(userDataDir, () => undefined, request);

      await expect(
        host.invoke(command, {
          request: {
            workingDir: "/workspace/project",
            name: "sample-product",
          },
        }),
      ).rejects.toThrow(`Electron host command is not implemented: ${command}`);
      expect(request).not.toHaveBeenCalled();
    },
  );
});

describe("ElectronHostCommands model provider current source", () => {
  it("get_default_provider 应忽略旧配置值并返回 App Server 当前已配置 Provider", async () => {
    const userDataDir = await createTempUserDataDir();
    await createHost(userDataDir).invoke("save_config", {
      config: { default_provider: "retired-provider" },
    });
    const request = vi.fn(async (method: string, _params?: unknown) => {
      if (method === "modelProvider/list") {
        return {
          providers: [
            {
              id: "retired-provider",
              name: "Retired Provider",
              enabled: true,
              api_key_count: 0,
            },
            {
              id: "lime-hub",
              name: "Lime Hub",
              enabled: true,
              api_key_count: 1,
            },
          ],
        };
      }
      throw new Error(`unexpected App Server method: ${method}`);
    });
    const host = createHost(userDataDir, () => undefined, request);

    await expect(host.invoke("get_default_provider")).resolves.toBe("lime-hub");
    expect(request).toHaveBeenCalledWith("modelProvider/list", {});
  });

  it("get_runtime_provider_selection 应按 opaque route 选择当前 Provider 模型", async () => {
    const userDataDir = await createTempUserDataDir();
    await createHost(userDataDir).invoke("save_config", {
      config: { default_provider: "retired-provider" },
    });
    const request = vi.fn(async (method: string) => {
      if (method === "modelProvider/list") {
        return {
          providers: [
            {
              id: "retired-provider",
              name: "Retired Provider",
              enabled: true,
              api_key_count: 0,
            },
            {
              id: "lime-hub",
              name: "Lime Hub",
              enabled: true,
              api_key_count: 1,
            },
          ],
        };
      }
      if (method === "model/list") {
        return {
          data: [
            {
              id: "route:ZGVlcHNlZWs.ZGVlcHNlZWstdjQtcHJv",
              model: "deepseek-v4-pro",
              upgrade: null,
              upgradeInfo: null,
              availabilityNux: null,
              displayName: "DeepSeek V4 Pro",
              description: "",
              hidden: false,
              supportedReasoningEfforts: [],
              defaultReasoningEffort: "none",
              inputModalities: ["text"],
              supportsPersonality: false,
              additionalSpeedTiers: [],
              serviceTiers: [],
              defaultServiceTier: null,
              isDefault: false,
            },
            {
              id: "route:bGltZS1odWI.Z3B0LTUuNi1zb2w",
              model: "gpt-5.6-sol",
              upgrade: null,
              upgradeInfo: null,
              availabilityNux: null,
              displayName: "GPT-5.6 Sol",
              description: "",
              hidden: false,
              supportedReasoningEfforts: [],
              defaultReasoningEffort: "none",
              inputModalities: ["text"],
              supportsPersonality: false,
              additionalSpeedTiers: [],
              serviceTiers: [],
              defaultServiceTier: null,
              isDefault: false,
            },
          ],
          nextCursor: null,
        };
      }
      throw new Error(`unexpected App Server method: ${method}`);
    });
    const host = createHost(userDataDir, () => undefined, request);

    await expect(
      host.invoke("get_runtime_provider_selection"),
    ).resolves.toEqual({
      provider_configured: true,
      provider_name: "Lime Hub",
      provider_selector: "lime-hub",
      model_name: "gpt-5.6-sol",
    });
    expect(request).toHaveBeenCalledWith("modelProvider/list", {});
    expect(request).toHaveBeenCalledWith("model/list", {
      includeHidden: true,
    });
  });
});

describe("ElectronHostCommands Plugin runtime dispatcher", () => {
  it("plugin_runtime_* 命令应只分发到 PluginRuntimeTaskHost", async () => {
    pluginRuntimeTaskHostStartTaskMock.mockResolvedValueOnce({
      status: "accepted",
    });
    pluginRuntimeTaskHostGetTaskMock.mockResolvedValueOnce({
      status: "thread_read_available",
    });
    pluginRuntimeTaskHostCancelTaskMock.mockResolvedValueOnce({
      status: "cancelled",
    });
    pluginRuntimeTaskHostSubmitHostResponseMock.mockResolvedValueOnce({
      status: "submitted",
    });
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    const startArgs = { request: { taskId: "task-start" } };
    const getArgs = { request: { taskId: "task-get" } };
    const cancelArgs = { request: { taskId: "task-cancel" } };
    const responseArgs = { request: { taskId: "task-response" } };

    await expect(
      host.invoke("plugin_runtime_start_task", startArgs),
    ).resolves.toEqual({ status: "accepted" });
    await expect(
      host.invoke("plugin_runtime_get_task", getArgs),
    ).resolves.toEqual({ status: "thread_read_available" });
    await expect(
      host.invoke("plugin_runtime_cancel_task", cancelArgs),
    ).resolves.toEqual({ status: "cancelled" });
    await expect(
      host.invoke("plugin_runtime_submit_host_response", responseArgs),
    ).resolves.toEqual({ status: "submitted" });

    expect(pluginRuntimeTaskHostStartTaskMock).toHaveBeenCalledWith(startArgs);
    expect(pluginRuntimeTaskHostGetTaskMock).toHaveBeenCalledWith(getArgs);
    expect(pluginRuntimeTaskHostCancelTaskMock).toHaveBeenCalledWith(
      cancelArgs,
    );
    expect(pluginRuntimeTaskHostSubmitHostResponseMock).toHaveBeenCalledWith(
      responseArgs,
    );
  });
});

describe("ElectronHostCommands system utilities", () => {
  it("系统工具命令应只分发到 SystemUtilityHost", async () => {
    systemUtilityHostOpenExternalUrlMock.mockResolvedValueOnce({});
    systemUtilityHostOpenSystemSettingsUrlMock.mockResolvedValueOnce({});
    systemUtilityHostGetEnvironmentPreviewMock.mockResolvedValueOnce({
      entries: [],
    });
    systemUtilityHostGetBrowserConnectorSettingsMock.mockReturnValueOnce({
      enabled: true,
    });
    systemUtilityHostGetBrowserConnectorInstallStatusMock.mockReturnValueOnce({
      status: "not_installed",
    });
    systemUtilityHostGetChromeProfileSessionsMock.mockReturnValueOnce([]);
    systemUtilityHostGetChromeBridgeEndpointInfoMock.mockReturnValueOnce({
      server_running: false,
    });
    systemUtilityHostGetChromeBridgeStatusMock.mockReturnValueOnce({
      observer_count: 0,
    });
    systemUtilityHostGetBrowserBackendPolicyMock.mockReturnValueOnce({
      priority: ["lime_extension_bridge", "cdp_direct"],
    });
    systemUtilityHostGetBrowserBackendsStatusMock.mockReturnValueOnce({
      backends: [],
    });
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);

    const externalArgs = { url: "https://user.limeai.run/login" };
    await expect(
      host.invoke("open_external_url", externalArgs),
    ).resolves.toEqual({});

    const settingsArgs = { url: "ms-settings:clipboard" };
    await expect(
      host.invoke("open_system_settings_url", settingsArgs),
    ).resolves.toEqual({});

    await expect(host.invoke("get_environment_preview")).resolves.toEqual({
      entries: [],
    });
    await expect(
      host.invoke("get_browser_connector_settings_cmd"),
    ).resolves.toEqual({ enabled: true });
    await expect(
      host.invoke("get_browser_connector_install_status_cmd"),
    ).resolves.toEqual({ status: "not_installed" });
    await expect(host.invoke("get_chrome_profile_sessions")).resolves.toEqual(
      [],
    );
    await expect(
      host.invoke("get_chrome_bridge_endpoint_info"),
    ).resolves.toEqual({ server_running: false });
    await expect(host.invoke("get_chrome_bridge_status")).resolves.toEqual({
      observer_count: 0,
    });
    await expect(host.invoke("get_browser_backend_policy")).resolves.toEqual({
      priority: ["lime_extension_bridge", "cdp_direct"],
    });
    await expect(host.invoke("get_browser_backends_status")).resolves.toEqual({
      backends: [],
    });

    expect(systemUtilityHostOpenExternalUrlMock).toHaveBeenCalledWith(
      externalArgs,
    );
    expect(systemUtilityHostOpenSystemSettingsUrlMock).toHaveBeenCalledWith(
      settingsArgs,
    );
    expect(systemUtilityHostGetEnvironmentPreviewMock).toHaveBeenCalledOnce();
    expect(
      systemUtilityHostGetBrowserConnectorSettingsMock,
    ).toHaveBeenCalledOnce();
    expect(
      systemUtilityHostGetBrowserConnectorInstallStatusMock,
    ).toHaveBeenCalledOnce();
    expect(
      systemUtilityHostGetChromeProfileSessionsMock,
    ).toHaveBeenCalledOnce();
    expect(
      systemUtilityHostGetChromeBridgeEndpointInfoMock,
    ).toHaveBeenCalledOnce();
    expect(systemUtilityHostGetChromeBridgeStatusMock).toHaveBeenCalledOnce();
    expect(systemUtilityHostGetBrowserBackendPolicyMock).toHaveBeenCalledOnce();
    expect(
      systemUtilityHostGetBrowserBackendsStatusMock,
    ).toHaveBeenCalledOnce();
  });

  it("voice_models_* 命令应只分发到 VoiceModelHost", async () => {
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);
    const state = {
      model_id: "sensevoice-small-int8-2024-07-17",
      installed: false,
    };
    const stateArgs = { modelId: "sensevoice-small-int8-2024-07-17" };
    const downloadArgs = {
      modelId: "sensevoice-small-int8-2024-07-17",
      catalogEntry: { id: "sensevoice-small-int8-2024-07-17" },
    };
    const deleteArgs = { modelId: "sensevoice-small-int8-2024-07-17" };
    voiceModelHostListCatalogMock.mockReturnValue([{ id: "voice-model" }]);
    voiceModelHostGetInstallStateMock.mockResolvedValue(state);
    voiceModelHostDownloadMock.mockResolvedValue({ state });
    voiceModelHostDeleteMock.mockResolvedValue(state);

    await expect(host.invoke("voice_models_list_catalog")).resolves.toEqual([
      { id: "voice-model" },
    ]);
    await expect(
      host.invoke("voice_models_get_install_state", stateArgs),
    ).resolves.toBe(state);
    await expect(
      host.invoke("voice_models_download", downloadArgs),
    ).resolves.toEqual({ state });
    await expect(host.invoke("voice_models_delete", deleteArgs)).resolves.toBe(
      state,
    );

    expect(voiceModelHostListCatalogMock).toHaveBeenCalledOnce();
    expect(voiceModelHostGetInstallStateMock).toHaveBeenCalledWith(stateArgs);
    expect(voiceModelHostDownloadMock).toHaveBeenCalledWith(downloadArgs);
    expect(voiceModelHostDeleteMock).toHaveBeenCalledWith(deleteArgs);
  });

  it("通过 Electron Host dispatcher 分发 summary-only 桌面通知", async () => {
    const userDataDir = await createTempUserDataDir();
    const host = createHost(userDataDir);
    const args = {
      request: {
        body: " Lime local output · +520 ms ",
        silent: true,
        tag: "claw-trace-regression-alert-123",
        title: " Regression alert: Critical ",
      },
    };

    await expect(
      host.invoke("show_desktop_notification", args),
    ).resolves.toEqual({ status: "sent" });

    expect(showDesktopNotificationMock).toHaveBeenCalledOnce();
    expect(showDesktopNotificationMock).toHaveBeenCalledWith(args);
  });

  it("启动真实 OAuth 本机回调桥并把回调事件广播到 renderer", async () => {
    const userDataDir = await createTempUserDataDir();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    const host = createHost(userDataDir, (event, payload) => {
      emitted.push({ event, payload });
    });

    const response = (await host.invoke(
      "start_oem_cloud_oauth_callback_bridge",
    )) as { callbackUrl: string };
    expect(response.callbackUrl).toMatch(
      /^http:\/\/127\.0\.0\.1:\d+\/oauth\/callback$/,
    );

    const callbackUrl = new URL(response.callbackUrl);
    callbackUrl.searchParams.set("tenant_id", "tenant-1");
    callbackUrl.searchParams.set("token", "token-1");
    callbackUrl.searchParams.set("next", "/dashboard");
    callbackUrl.searchParams.set("device_code", "device-1");
    callbackUrl.searchParams.set("status", "ok");

    const callbackResponse = await fetch(callbackUrl);
    expect(callbackResponse.status).toBe(200);
    await expect(callbackResponse.text()).resolves.toContain(
      "Lime 登录结果已返回",
    );

    expect(emitted).toEqual([
      {
        event: "oem-cloud-oauth-callback",
        payload: {
          sourcePath: "/oauth/callback",
          tenantId: "tenant-1",
          token: "token-1",
          next: "/dashboard",
          error: null,
          deviceCode: "device-1",
          status: "ok",
        },
      },
    ]);

    await expect(fetch(callbackUrl)).rejects.toThrow();
  });
});
