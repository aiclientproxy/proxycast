import { mkdirSync, mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import {
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  openSettings,
} from "../mcp-config-fixture-smoke.mjs";
import {
  ensureDefaultWorkspace,
  initializeAppServer,
} from "../../agent-runtime/claw-chat-current-fixture-rpc.mjs";

const APP_SERVER_COMMAND = "app_server_handle_json_lines";
const NATIVE_HOST_COMMAND = "macos_native_host_invoke";
const APPLICATION_ID = "com.limecloud.lime";
const HELPER_ID = "macos-native-host";
const PROTOCOL_VERSION = 1;

async function invokeNativeHostFromPage(page, method, params = {}) {
  return await page.evaluate(
    async ({ command, method: nativeMethod, params: nativeParams }) => {
      const invoke = window.electronAPI?.invoke;
      if (typeof invoke !== "function") {
        throw new Error("Electron preload invoke bridge is unavailable");
      }
      return await invoke(command, {
        request: { method: nativeMethod, params: nativeParams },
      });
    },
    { command: NATIVE_HOST_COMMAND, method, params },
  );
}

function readJsonArray(raw) {
  try {
    const parsed = JSON.parse(raw || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function permissionStatus(capabilities, key) {
  const value = capabilities?.[key];
  return {
    status: typeof value?.status === "string" ? value.status : "unavailable",
    reason: typeof value?.reason === "string" ? value.reason : null,
  };
}

export async function runMacOSNativeHostElectronGateB(options) {
  const runtimeEnv = createTempRuntimeEnv();
  const errors = { console: [], page: [] };
  const requestLog = [];
  const nativeMethods = [];
  const checks = [];
  let handle = null;
  let bookmarkRoot = null;
  try {
    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv: { APP_SERVER_BIN: "" },
      consoleErrors: errors.console,
      pageErrors: errors.page,
      backendMode: "unavailable",
    });
    const { page, rendererSnapshot } = handle;
    if (
      rendererSnapshot.electron !== true ||
      rendererSnapshot.hasInvokeBridge !== true ||
      rendererSnapshot.supportsAppServer !== true
    ) {
      throw new Error(
        "Electron renderer/preload App Server bridge is not ready",
      );
    }
    checks.push({ name: "electron.renderer-preload", status: "passed" });

    await initializeAppServer(page, requestLog);
    const workspace = await ensureDefaultWorkspace(page, requestLog);
    if (!workspace.workspaceId) {
      throw new Error(
        "workspace/default/ensure did not return workspace identity",
      );
    }
    checks.push({
      name: "app-server.json-rpc",
      status: "passed",
      command: APP_SERVER_COMMAND,
      methods: requestLog.map((entry) => entry.method),
    });

    const invokeNative = async (method, params = {}) => {
      nativeMethods.push(method);
      return await invokeNativeHostFromPage(page, method, params);
    };
    const capabilities = await invokeNative("capabilities.read");
    if (
      capabilities?.protocolVersion !== PROTOCOL_VERSION ||
      capabilities.helperId !== HELPER_ID ||
      capabilities.platform !== "darwin" ||
      capabilities.applicationId !== `${APPLICATION_ID}.native-host`
    ) {
      throw new Error("Electron IPC native helper identity handshake failed");
    }
    const windows = await invokeNative("window.read");
    if (!Array.isArray(windows?.windows)) {
      throw new Error("Electron IPC window.read did not return a window list");
    }
    const displays = await invokeNative("display.read");
    if (!Array.isArray(displays?.displays) || displays.displays.length === 0) {
      throw new Error("Electron IPC display.read did not return a display");
    }
    const permissionMethods = [
      ["accessibility", "accessibility.read"],
      ["inputMonitoring", "inputMonitoring.read"],
      ["screenCapture", "screenCapture.read"],
    ];
    const permissions = {};
    for (const [key, method] of permissionMethods) {
      const result = await invokeNative(method);
      permissions[key] = permissionStatus({ [key]: result }, key);
      if (options.strictPermissions && result?.status !== "ready") {
        throw new Error(`Electron IPC ${key} permission is not ready`);
      }
    }

    bookmarkRoot = mkdtempSync(
      path.join(tmpdir(), "lime-macos-native-electron-gate-b-"),
    );
    const created = await invokeNative("bookmark.create", {
      path: bookmarkRoot,
    });
    if (typeof created?.bookmark !== "string" || !created.bookmark) {
      throw new Error("Electron IPC bookmark.create returned no bookmark");
    }
    const resolved = await invokeNative("bookmark.resolve", {
      bookmark: created.bookmark,
    });
    if (
      resolved?.isStale !== false ||
      realpathSync(resolved.path) !== realpathSync(bookmarkRoot)
    ) {
      throw new Error(
        "Electron IPC bookmark.resolve returned an unexpected path",
      );
    }
    const started = await invokeNative("bookmark.start", {
      bookmark: created.bookmark,
    });
    if (started?.started !== true || typeof started.token !== "string") {
      throw new Error("Electron IPC bookmark.start did not return a token");
    }
    const stopped = await invokeNative("bookmark.stop", {
      token: started.token,
    });
    if (stopped?.stopped !== true) {
      throw new Error("Electron IPC bookmark.stop did not stop the token");
    }
    checks.push({
      name: "native-host.electron-ipc",
      status: "passed",
      command: NATIVE_HOST_COMMAND,
      methods: [...nativeMethods],
      windowCount: windows.windows.length,
      displayCount: displays.displays.length,
    });

    // 触发生产设置页的 safeInvoke，确保 App Server 证据来自真实 renderer 主链。
    await openSettings(page, options);
    const gui = await page.evaluate(() => ({
      url: window.location.href,
      electron: window.__LIME_ELECTRON__ === true,
      appSidebarVisible: Boolean(
        document.querySelector('[data-testid="app-sidebar"]'),
      ),
      startupVisible: Boolean(
        document.querySelector("[data-lime-startup-shell]"),
      ),
      settingsHeaderVisible: Boolean(
        document.querySelector('[data-testid="settings-top-header"]'),
      ),
      traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
    }));
    const trace = readJsonArray(gui.traceRaw);
    const appServerTrace = trace.filter(
      (entry) =>
        entry?.command === APP_SERVER_COMMAND &&
        entry?.transport === "electron-ipc" &&
        entry?.status === "success",
    );
    const invokeErrors = readJsonArray(gui.errorRaw);
    if (appServerTrace.length === 0) {
      throw new Error(
        "Electron IPC trace did not record app_server_handle_json_lines",
      );
    }
    if (gui.startupVisible || !gui.settingsHeaderVisible) {
      throw new Error(
        `Electron GUI did not reach the visible current shell: ${JSON.stringify(
          {
            appSidebarVisible: gui.appSidebarVisible,
            startupVisible: gui.startupVisible,
            settingsHeaderVisible: gui.settingsHeaderVisible,
            url: gui.url,
          },
        )}`,
      );
    }
    if (
      errors.console.length > 0 ||
      errors.page.length > 0 ||
      invokeErrors.length > 0
    ) {
      throw new Error(
        "Electron native Gate B observed console, page, or invoke errors",
      );
    }
    const screenshotPath = path.join(
      options.evidenceDir,
      "electron-gate-b.png",
    );
    mkdirSync(options.evidenceDir, { recursive: true });
    await page.screenshot({ path: screenshotPath, fullPage: true });
    checks.push({
      name: "gui.visible-state",
      status: "passed",
      screenshot: path.relative(process.cwd(), screenshotPath),
    });
    return {
      result: "passed",
      renderer: rendererSnapshot,
      appServer: {
        command: APP_SERVER_COMMAND,
        transport: "electron-ipc",
        traceCount: appServerTrace.length,
        methods: requestLog.map((entry) => entry.method),
        workspaceId: workspace.workspaceId,
      },
      nativeHost: {
        command: NATIVE_HOST_COMMAND,
        transport: "electron-ipc",
        methods: [...nativeMethods],
        capabilities: {
          protocolVersion: capabilities.protocolVersion,
          helperId: capabilities.helperId,
          platform: capabilities.platform,
          applicationId: capabilities.applicationId,
        },
        windowCount: windows.windows.length,
        displayCount: displays.displays.length,
        permissions,
      },
      gui: {
        url: gui.url,
        appSidebarVisible: gui.appSidebarVisible,
        startupVisible: gui.startupVisible,
        settingsHeaderVisible: gui.settingsHeaderVisible,
        screenshot: path.relative(process.cwd(), screenshotPath),
      },
      trace: trace
        .filter((entry) => entry?.transport === "electron-ipc")
        .map((entry) => ({
          command: entry.command,
          transport: entry.transport,
          status: entry.status,
        })),
      errors: {
        console: errors.console,
        page: errors.page,
        invoke: invokeErrors,
      },
      checks,
    };
  } finally {
    await closeElectronFixture(handle);
    if (bookmarkRoot) {
      rmSync(bookmarkRoot, { recursive: true, force: true });
    }
    rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
  }
}
