/* global Electron, process */
import {
  IPC_DEEP_LINK_GET_CURRENT_CHANNEL,
  IPC_DEEP_LINK_GET_URLS_CHANNEL,
  IPC_DIALOG_OPEN_CHANNEL,
  IPC_DIALOG_SAVE_CHANNEL,
  IPC_EMIT_CHANNEL,
  IPC_GLOBAL_SHORTCUT_COMMAND_CHANNEL,
  IPC_INVOKE_CHANNEL,
  IPC_SHELL_OPEN_CHANNEL,
  IPC_WINDOW_COMMAND_CHANNEL,
  isElectronAppServerCommand,
  isElectronHostCommand,
  isElectronUpdateCommand,
  type ElectronInvokeResponse,
} from "./ipcChannels";
import {
  LIME_HOST_DATA_DIR_NAME,
  resolveCurrentDesktopStorageRoots,
} from "./appDataPaths";
import { ElectronAppServerHost } from "./appServerHost";
import {
  ElectronEmbeddedBrowserHost,
  isEmbeddedBrowserCommand,
} from "./embeddedBrowserHost";
import { ElectronDevHttpBridge } from "./devHttpBridge";
import { ElectronHostCommands } from "./hostCommands";
import {
  buildMainWindowChromeOptions,
  buildMainWindowStartupHtml,
  buildMainWindowStartupOptions,
} from "./mainWindowOptions";
import {
  isMainWindowRendererLoadInterruption,
  isNavigationAbortError,
  isWindowLifecycleLoadAbort,
} from "./mainWindowLoadErrors";
import { installMainWindowMediaPermissionHandler } from "./mainWindowMediaPermissions";
import { ElectronUpdateHost } from "./updateHost";
import { createElectronSmokeRunner } from "./smokeChecks";
import { resolveWindowsSquirrelStartupPlan } from "./windowsSquirrelStartup";
import {
  buildUpdateNotificationWindowBounds,
  type RectangleLike,
} from "./updateNotificationWindowPosition";
import { buildUpdateNotificationWindowUrl } from "./updateNotificationWindowUrl";
import {
  app,
  BrowserWindow,
  dialog,
  globalShortcut,
  ipcMain,
  Menu,
  nativeImage,
  net,
  screen,
  session,
  type MenuItemConstructorOptions,
  type OpenDialogOptions,
  protocol,
  shell,
  type SaveDialogOptions,
  Tray,
  type IpcMainInvokeEvent,
} from "./electronRuntime";
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const APP_NAME = "Lime";
const APP_BUNDLE_IDENTIFIER = "com.limecloud.lime";
const APP_ICON_SOURCE = "lime-rs/icons/icon.png";
const APP_ICON_PACKAGED_NAME = "icon.png";
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isWindowsSquirrelStartup = handleWindowsSquirrelStartup();
configureElectronUserDataPath();
const desktopStorageRoots = resolveCurrentDesktopStorageRoots(
  app.getPath("userData"),
);
configureElectronSessionDataPath(desktopStorageRoots.hostSessionData);
const appServerHost = new ElectronAppServerHost();
const hostCommands = new ElectronHostCommands(
  appServerHost,
  app.getPath("userData"),
  broadcast,
  desktopStorageRoots.appDataRoot,
);
const embeddedBrowserHost = new ElectronEmbeddedBrowserHost(broadcast);
const updateHost = new ElectronUpdateHost(broadcast, {
  open: openUpdateNotificationWindow,
  close: closeUpdateNotificationWindow,
});
let devHttpBridge: ElectronDevHttpBridge | null = null;
const pendingDeepLinks: string[] = [];
const pendingSkillPackageOpenPaths: string[] = [];
const SKILL_PACKAGE_OPEN_EVENT = "skill-package://open";
const TRAY_MODEL_SELECTED_EVENT = "tray-model-selected";
const STARTUP_SCREEN_VISIBLE_TIMEOUT_MS = 900;
const ELECTRON_SMOKE_SHUTDOWN_TIMEOUT_MS = 10_000;
const UPDATE_NOTIFICATION_ANCHOR_SELECTOR =
  '[data-testid="app-sidebar-update-button"]';
const UPDATE_NOTIFICATION_WINDOW_SIZE = {
  width: 232,
  height: 128,
};

let mainWindow: BrowserWindow | null = null;
let updateNotificationWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let isQuitting = false;
let trayModelShortcutsState: TrayModelShortcutsState | null = null;

protocol.registerSchemesAsPrivileged([
  {
    scheme: "asset",
    privileges: {
      standard: true,
      secure: true,
      stream: true,
    },
  },
]);

app.setName(APP_NAME);

interface TrayQuickModelItem {
  provider_type: string;
  provider_label: string;
  model: string;
}

interface TrayQuickModelGroup {
  provider_type: string;
  provider_label: string;
  models: TrayQuickModelItem[];
}

interface TrayModelShortcutsState {
  currentModelProviderType: string;
  currentModelProviderLabel: string;
  currentModel: string;
  currentThemeLabel: string;
  quickModelGroups: TrayQuickModelGroup[];
}

function createMainWindow(): BrowserWindow {
  const window = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 980,
    minHeight: 680,
    title: APP_NAME,
    ...buildMainWindowStartupOptions(),
    ...buildMainWindowChromeOptions(),
    icon: resolveDesktopAsset(APP_ICON_SOURCE, APP_ICON_PACKAGED_NAME),
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      preload: path.resolve(__dirname, "../preload/preload.cjs"),
    },
  });

  const devServerUrl = process.env.VITE_DEV_SERVER_URL?.trim();
  installMainWindowNavigationGuard(window, devServerUrl);
  installDevRendererContextMenu(window, devServerUrl);
  installDevRendererShortcuts(window, devServerUrl);
  const smokeRunner = isElectronSmokeMode()
    ? createElectronSmokeRunner({
        window,
        appServerHost,
        appVersion: app.getVersion(),
      })
    : null;
  void showStartupScreenBeforeRenderer(window, devServerUrl).catch((error) => {
    if (
      isMainWindowRendererLoadInterruption(error, mainWindowLoadState(window))
    ) {
      return;
    }
    console.error(
      `[electron-host] main window renderer load failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  });

  if (smokeRunner) {
    const runSmokeAfterRendererLoad = () => {
      const loadedUrl = window.webContents.getURL();
      if (smokeRunner.isStartupUrl(loadedUrl)) {
        return;
      }
      window.webContents.off("did-finish-load", runSmokeAfterRendererLoad);
      console.log("[electron-smoke] renderer loaded");
      void smokeRunner
        .run()
        .then(() => {
          void exitElectronSmoke(0);
        })
        .catch((error) => {
          console.error(
            `[electron-smoke] failed: ${
              error instanceof Error ? error.message : String(error)
            }`,
          );
          void exitElectronSmoke(1);
        });
    };
    window.webContents.on("did-finish-load", runSmokeAfterRendererLoad);
    window.webContents.on(
      "did-fail-load",
      (_event, code, description, _url, isMainFrame) => {
        if (code === -3 || isMainFrame === false) {
          return;
        }
        console.error(
          `[electron-smoke] renderer failed: ${code} ${description}`,
        );
        void smokeRunner
          .recordFailure("renderer-load")
          .catch(() => undefined)
          .finally(() => exitElectronSmoke(1));
      },
    );
  }

  window.on("closed", () => {
    if (mainWindow === window) {
      mainWindow = null;
    }
  });
  window.on("close", (event) => {
    if (isQuitting || process.env.LIME_ELECTRON_SMOKE === "1") {
      return;
    }
    event.preventDefault();
    window.hide();
  });

  mainWindow = window;
  return window;
}

function registerLocalAssetProtocol(): void {
  if (protocol.isProtocolHandled("asset")) {
    return;
  }

  protocol.handle("asset", (request) => {
    const filePath = decodeLocalAssetFilePath(request.url);
    if (!filePath || !existsSync(filePath)) {
      return new Response("asset not found", { status: 404 });
    }

    return net.fetch(pathToFileURL(filePath).toString());
  });
}

function decodeLocalAssetFilePath(requestUrl: string): string | null {
  try {
    const url = new URL(requestUrl);
    const encodedPath = url.host ? `${url.host}${url.pathname}` : url.pathname;
    const filePath = decodeURIComponent(encodedPath);
    return isAbsoluteLocalAssetPath(filePath) ? filePath : null;
  } catch {
    return null;
  }
}

function isAbsoluteLocalAssetPath(filePath: string): boolean {
  return (
    filePath.startsWith("/") ||
    /^[a-zA-Z]:[\\/]/.test(filePath) ||
    filePath.startsWith("\\\\")
  );
}

async function showStartupScreenBeforeRenderer(
  window: BrowserWindow,
  devServerUrl?: string,
): Promise<void> {
  await loadMainWindowStartupScreen(window);
  await waitForMainWindowStartupScreenVisible(window);
  showMainWindowDuringStartup(window);
  await loadMainWindowRenderer(window, devServerUrl);
}

function showMainWindowDuringStartup(window: BrowserWindow): void {
  if (window.isDestroyed() || !shouldShowMainWindowDuringStartup()) {
    return;
  }

  if (isElectronSmokeMode()) {
    window.showInactive();
    return;
  }

  window.show();
}

function shouldShowMainWindowDuringStartup(): boolean {
  return (
    !isElectronSmokeMode() || process.env.LIME_ELECTRON_SMOKE_VISIBLE === "1"
  );
}

function isElectronSmokeMode(): boolean {
  return process.env.LIME_ELECTRON_SMOKE === "1";
}

async function loadMainWindowStartupScreen(
  window: BrowserWindow,
): Promise<void> {
  const iconDataUrl = resolveStartupIconDataUrl();
  const startupHtml = buildMainWindowStartupHtml({
    appName: APP_NAME,
    iconDataUrl,
    locale: app.getLocale(),
  });

  try {
    await window.loadURL(writeMainWindowStartupHtmlFile(startupHtml));
  } catch (error) {
    console.warn(
      `[electron-host] startup screen failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

function writeMainWindowStartupHtmlFile(html: string): string {
  const startupDir = path.join(app.getPath("userData"), "startup");
  mkdirSync(startupDir, { recursive: true });
  const startupPath = path.join(startupDir, "main-window-startup.html");
  writeFileSync(startupPath, html, "utf8");
  return pathToFileURL(startupPath).toString();
}

async function waitForMainWindowStartupScreenVisible(
  window: BrowserWindow,
): Promise<void> {
  if (window.isDestroyed()) {
    return;
  }

  try {
    await Promise.race([
      window.webContents.executeJavaScript(
        `new Promise((resolve) => {
          const finish = () => resolve(true);
          const waitForPaint = () => {
            requestAnimationFrame(() => requestAnimationFrame(finish));
          };
          const logo = document.querySelector("[data-lime-startup-logo]");
          if (!logo || logo.tagName !== "IMG") {
            waitForPaint();
            return;
          }
          if (logo.complete && logo.naturalWidth > 0) {
            waitForPaint();
            return;
          }
          if (typeof logo.decode === "function") {
            logo.decode().catch(() => undefined).then(waitForPaint);
            return;
          }
          logo.addEventListener("load", waitForPaint, { once: true });
          logo.addEventListener("error", waitForPaint, { once: true });
        })`,
      ),
      new Promise((resolve) => {
        setTimeout(resolve, STARTUP_SCREEN_VISIBLE_TIMEOUT_MS);
      }),
    ]);
  } catch (error) {
    if (!window.isDestroyed()) {
      console.warn(
        `[electron-host] startup screen visibility wait failed: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
  }
}

async function loadMainWindowRenderer(
  window: BrowserWindow,
  devServerUrl?: string,
): Promise<void> {
  if (window.isDestroyed()) {
    return;
  }

  if (devServerUrl) {
    await clearDevRendererCache();
    if (!window.isDestroyed()) {
      await loadMainWindowUrl(window, withNativeStartupFlag(devServerUrl));
    }
    if (
      !window.isDestroyed() &&
      process.env.LIME_ELECTRON_OPEN_DEVTOOLS === "1"
    ) {
      window.webContents.openDevTools({ mode: "detach" });
    }
    return;
  }

  await loadMainWindowUrl(
    window,
    withNativeStartupFlag(
      pathToFileURL(
        path.resolve(app.getAppPath(), "dist/index.html"),
      ).toString(),
    ),
  );
}

async function loadMainWindowUrl(
  window: BrowserWindow,
  url: string,
): Promise<void> {
  try {
    await window.loadURL(url);
  } catch (error) {
    if (
      isNavigationAbortError(error) ||
      isWindowLifecycleLoadAbort(error, mainWindowLoadState(window))
    ) {
      return;
    }
    throw error;
  }
}

function mainWindowLoadState(
  window: Pick<BrowserWindow, "isDestroyed" | "webContents">,
) {
  return {
    appQuitting: isQuitting,
    windowDestroyed: window.isDestroyed(),
    webContentsDestroyed: window.webContents.isDestroyed(),
  };
}

function withNativeStartupFlag(targetUrl: string): string {
  const url = new URL(targetUrl);
  url.searchParams.set("nativeStartup", "1");
  return url.toString();
}

async function clearDevRendererCache(): Promise<void> {
  if (process.env.LIME_ELECTRON_CLEAR_RENDERER_CACHE === "0") {
    return;
  }

  try {
    await session.defaultSession.clearCache();
  } catch (error) {
    console.warn(
      `[electron-host] failed to clear dev renderer cache: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

async function exitElectronSmoke(exitCode: number): Promise<void> {
  process.exitCode = exitCode;
  isQuitting = true;
  globalShortcut.unregisterAll();
  tray?.destroy();
  tray = null;
  devHttpBridge?.stop();
  devHttpBridge = null;
  let shutdownTimer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      appServerHost.stop(),
      new Promise<never>((_, reject) => {
        shutdownTimer = setTimeout(
          () =>
            reject(
              new Error(
                `app-server stop timed out after ${ELECTRON_SMOKE_SHUTDOWN_TIMEOUT_MS}ms`,
              ),
            ),
          ELECTRON_SMOKE_SHUTDOWN_TIMEOUT_MS,
        );
      }),
    ]);
  } catch (error) {
    console.warn(
      `[electron-smoke] app-server stop failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  } finally {
    if (shutdownTimer) {
      clearTimeout(shutdownTimer);
    }
    app.exit(exitCode);
  }
}

function installMainWindowNavigationGuard(
  window: BrowserWindow,
  devServerUrl?: string,
): void {
  const allowedOrigin = devServerUrl ? new URL(devServerUrl).origin : null;

  window.webContents.on("will-navigate", (event, targetUrl) => {
    if (isAllowedMainWindowNavigation(targetUrl, allowedOrigin)) {
      return;
    }
    event.preventDefault();
    void shell.openExternal(targetUrl);
  });

  window.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//i.test(url)) {
      void shell.openExternal(url);
      return { action: "deny" };
    }
    return { action: "deny" };
  });
}

function installDevRendererContextMenu(
  window: BrowserWindow,
  devServerUrl?: string,
): void {
  if (!devServerUrl || process.env.LIME_ELECTRON_DEV_CONTEXT_MENU === "0") {
    return;
  }

  window.webContents.on("context-menu", (_event, params) => {
    const template: MenuItemConstructorOptions[] = [];

    if (params.isEditable) {
      template.push(
        { role: "undo" },
        { role: "redo" },
        { type: "separator" },
        { role: "cut" },
        { role: "copy" },
        { role: "paste" },
        { role: "selectAll" },
        { type: "separator" },
      );
    } else if (params.selectionText.trim()) {
      template.push({ role: "copy" }, { type: "separator" });
    }

    template.push(
      { role: "reload" },
      { role: "forceReload" },
      { type: "separator" },
      { role: "toggleDevTools" },
      {
        label: "检查元素",
        click: () => {
          if (!window.isDestroyed()) {
            window.webContents.inspectElement(params.x, params.y);
          }
        },
      },
    );

    Menu.buildFromTemplate(template).popup({
      window,
      frame: params.frame ?? undefined,
      x: params.x,
      y: params.y,
      sourceType: params.menuSourceType,
    });
  });
}

function installDevRendererShortcuts(
  window: BrowserWindow,
  devServerUrl?: string,
): void {
  if (!devServerUrl || process.env.LIME_ELECTRON_DEV_SHORTCUTS === "0") {
    return;
  }

  window.webContents.on("before-input-event", (event, input) => {
    if (input.type !== "keyDown") {
      return;
    }

    const key = input.key.toLowerCase();
    const commandOrControl =
      process.platform === "darwin" ? input.meta : input.control;
    const reloadRequested = key === "f5" || (commandOrControl && key === "r");
    const devToolsRequested =
      key === "f12" ||
      (commandOrControl && input.shift && key === "i") ||
      (process.platform === "darwin" && input.meta && input.alt && key === "i");

    if (reloadRequested) {
      event.preventDefault();
      if (input.shift) {
        window.webContents.reloadIgnoringCache();
      } else {
        window.webContents.reload();
      }
      return;
    }

    if (devToolsRequested) {
      event.preventDefault();
      if (window.webContents.isDevToolsOpened()) {
        window.webContents.closeDevTools();
      } else {
        window.webContents.openDevTools({ mode: "detach" });
      }
    }
  });
}

function isAllowedMainWindowNavigation(
  targetUrl: string,
  allowedOrigin: string | null,
): boolean {
  try {
    const url = new URL(targetUrl);
    if (url.protocol === "file:") {
      return true;
    }
    return allowedOrigin !== null && url.origin === allowedOrigin;
  } catch {
    return false;
  }
}

function createTray(): Tray | null {
  const image = loadTrayImage();
  if (image.isEmpty()) {
    console.warn("[electron-host] tray icon missing, skip tray setup");
    return null;
  }

  const nextTray = new Tray(image);
  nextTray.setToolTip(APP_NAME);
  nextTray.setContextMenu(buildTrayMenu());
  nextTray.on("click", () => {
    showMainWindow();
  });
  return nextTray;
}

function buildTrayMenu(): Menu {
  const template: MenuItemConstructorOptions[] = [
    {
      label: `显示 ${APP_NAME}`,
      click: () => showMainWindow(),
    },
    {
      label: "隐藏窗口",
      click: () => mainWindow?.hide(),
    },
    { type: "separator" },
    {
      label: "退出",
      click: () => {
        isQuitting = true;
        app.quit();
      },
    },
  ];

  const shortcutItems = buildTrayModelShortcutMenuItems();
  if (shortcutItems.length > 0) {
    template.splice(2, 0, { type: "separator" }, ...shortcutItems);
  }

  return Menu.buildFromTemplate(template);
}

function buildTrayModelShortcutMenuItems(): MenuItemConstructorOptions[] {
  const state = trayModelShortcutsState;
  if (!state || !state.currentModel.trim()) {
    return [];
  }

  const currentLabel = [
    state.currentModelProviderLabel || state.currentModelProviderType,
    state.currentModel,
  ]
    .filter(Boolean)
    .join(" / ");
  const items: MenuItemConstructorOptions[] = [
    {
      label: currentLabel,
      enabled: false,
    },
  ];

  if (state.currentThemeLabel) {
    items.push({
      label: state.currentThemeLabel,
      enabled: false,
    });
  }

  const groups = state.quickModelGroups.filter(
    (group) => group.provider_type && group.models.length > 0,
  );
  if (groups.length === 0) {
    return items;
  }

  items.push({ type: "separator" });
  for (const group of groups) {
    items.push({
      label: group.provider_label || group.provider_type,
      submenu: group.models.map((item) => ({
        label: item.model,
        type: "checkbox",
        checked:
          item.provider_type === state.currentModelProviderType &&
          item.model === state.currentModel,
        click: () => {
          broadcast(TRAY_MODEL_SELECTED_EVENT, {
            providerType: item.provider_type,
            model: item.model,
          });
        },
      })),
    });
  }

  return items;
}

function showMainWindow(): void {
  if (!mainWindow) {
    createMainWindow();
    return;
  }
  mainWindow.show();
  if (mainWindow.isMinimized()) {
    mainWindow.restore();
  }
  mainWindow.focus();
}

async function openUpdateNotificationWindow(
  params?: {
    current?: string | null;
    latest?: string | null;
    downloadUrl?: string | null;
    anchorRect?: RectangleLike | null;
  } | null,
): Promise<void> {
  const url = buildUpdateNotificationWindowUrl({
    appPath: app.getAppPath(),
    devServerUrl: process.env.VITE_DEV_SERVER_URL?.trim(),
    current: params?.current,
    latest: params?.latest,
    downloadUrl: params?.downloadUrl,
  });

  if (updateNotificationWindow && !updateNotificationWindow.isDestroyed()) {
    await loadMainWindowUrl(updateNotificationWindow, url);
    await positionUpdateNotificationWindow(
      updateNotificationWindow,
      params?.anchorRect,
    );
    updateNotificationWindow.show();
    updateNotificationWindow.focus();
    return;
  }

  updateNotificationWindow = new BrowserWindow({
    width: UPDATE_NOTIFICATION_WINDOW_SIZE.width,
    height: UPDATE_NOTIFICATION_WINDOW_SIZE.height,
    minWidth: 220,
    minHeight: 112,
    maxWidth: 320,
    maxHeight: 180,
    title: "Lime Update",
    frame: false,
    transparent: true,
    hasShadow: false,
    resizable: false,
    movable: true,
    show: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    backgroundColor: "#00000000",
    icon: resolveDesktopAsset(APP_ICON_SOURCE, APP_ICON_PACKAGED_NAME),
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      preload: path.resolve(__dirname, "../preload/preload.cjs"),
    },
  });

  updateNotificationWindow.on("closed", () => {
    updateNotificationWindow = null;
  });
  updateNotificationWindow.once("ready-to-show", () => {
    updateNotificationWindow?.show();
    updateNotificationWindow?.focus();
  });
  updateNotificationWindow.webContents.setWindowOpenHandler(
    ({ url: target }) => {
      if (/^https?:\/\//i.test(target)) {
        void shell.openExternal(target);
      }
      return { action: "deny" };
    },
  );

  await positionUpdateNotificationWindow(
    updateNotificationWindow,
    params?.anchorRect,
  );
  await loadMainWindowUrl(updateNotificationWindow, url);
}

async function closeUpdateNotificationWindow(): Promise<void> {
  if (!updateNotificationWindow || updateNotificationWindow.isDestroyed()) {
    updateNotificationWindow = null;
    return;
  }
  updateNotificationWindow.close();
}

async function positionUpdateNotificationWindow(
  window: BrowserWindow,
  preferredAnchorRect?: RectangleLike | null,
): Promise<void> {
  if (window.isDestroyed()) {
    return;
  }

  const contentBounds = getUpdateNotificationReferenceContentBounds();
  const anchorRect =
    preferredAnchorRect ?? (await readUpdateNotificationAnchorRect());
  const display = screen.getDisplayMatching(contentBounds);
  window.setBounds(
    buildUpdateNotificationWindowBounds({
      anchorRect,
      contentBounds,
      updateWindowSize: UPDATE_NOTIFICATION_WINDOW_SIZE,
      workArea: display.workArea,
    }),
  );
}

function getUpdateNotificationReferenceContentBounds(): RectangleLike {
  const referenceWindow =
    mainWindow && !mainWindow.isDestroyed() ? mainWindow : null;
  if (referenceWindow) {
    return referenceWindow.getContentBounds();
  }
  return screen.getPrimaryDisplay().workArea;
}

async function readUpdateNotificationAnchorRect(): Promise<RectangleLike | null> {
  const referenceWindow =
    mainWindow && !mainWindow.isDestroyed() ? mainWindow : null;
  if (!referenceWindow) {
    return null;
  }

  try {
    return (await referenceWindow.webContents.executeJavaScript(
      `(() => {
        const element =
          document.querySelector('[data-update-notification-anchor="true"]') ||
          document.querySelector(${JSON.stringify(
            UPDATE_NOTIFICATION_ANCHOR_SELECTOR,
          )});
        if (!element) {
          return null;
        }
        const rect = element.getBoundingClientRect();
        if (!rect.width || !rect.height) {
          return null;
        }
        return {
          x: rect.x,
          y: rect.y,
          width: rect.width,
          height: rect.height,
        };
      })()`,
      true,
    )) as RectangleLike | null;
  } catch (error) {
    console.warn(
      `[electron-host] update notification anchor unavailable: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    return null;
  }
}

function loadTrayImage(): Electron.NativeImage {
  const preferred =
    process.platform === "darwin"
      ? "lime-rs/icons/tray/trayTemplate.png"
      : "lime-rs/icons/tray/tray-running.png";
  const image = nativeImage.createFromPath(
    resolveDesktopAsset(preferred, path.basename(preferred)),
  );
  if (process.platform === "darwin") {
    image.setTemplateImage(true);
  }
  return image;
}

function configureElectronUserDataPath(): void {
  const e2eUserDataDir = process.env.ELECTRON_E2E_USER_DATA_DIR?.trim();
  if (e2eUserDataDir) {
    if (process.env.LIME_ELECTRON_E2E !== "1") {
      console.warn(
        "[electron-host] ELECTRON_E2E_USER_DATA_DIR is ignored outside E2E mode",
      );
    } else {
      const resolvedUserDataDir = path.resolve(e2eUserDataDir);
      mkdirSync(resolvedUserDataDir, { recursive: true });
      app.setPath("userData", resolvedUserDataDir);
      return;
    }
  }

  // Agent durable data 由独立 resolver 管理；这里仅固定 host profile/config 根，
  // 并保留 macOS 既有的小写目录。
  const stableUserDataDir = path.join(
    app.getPath("appData"),
    LIME_HOST_DATA_DIR_NAME,
  );
  if (
    path.resolve(app.getPath("userData")) !== path.resolve(stableUserDataDir)
  ) {
    mkdirSync(stableUserDataDir, { recursive: true });
    app.setPath("userData", stableUserDataDir);
  }
}

/**
 * 固定 Chromium cookies/storage/network state 与 cache 的唯一根。
 * macOS 与 host profile 同根，行为不变；Windows 从 roaming `%APPDATA%`
 * 移到 `<AppDataRoot>\host-session`，旧 roaming 会话数据不迁移。
 */
function configureElectronSessionDataPath(hostSessionData: string): void {
  const resolved = path.resolve(hostSessionData);
  if (path.resolve(app.getPath("sessionData")) === resolved) {
    return;
  }
  mkdirSync(resolved, { recursive: true });
  app.setPath("sessionData", resolved);
}

function broadcast(event: string, payload?: unknown): void {
  for (const window of BrowserWindow.getAllWindows()) {
    window.webContents.send(`evt:${event}`, { event, payload });
  }
  devHttpBridge?.broadcast(event, payload);
}

function currentWindow(event: IpcMainInvokeEvent): BrowserWindow | null {
  return BrowserWindow.fromWebContents(event.sender) ?? mainWindow;
}

function registerIpcHandlers(): void {
  ipcMain.handle(
    IPC_INVOKE_CHANNEL,
    async (event, command: string, args?: Record<string, unknown>) => {
      try {
        const result = await handleHostInvoke(event, command, args);
        return { ok: true, result } satisfies ElectronInvokeResponse;
      } catch (error) {
        return {
          ok: false,
          error: {
            message: error instanceof Error ? error.message : String(error),
          },
        } satisfies ElectronInvokeResponse;
      }
    },
  );

  ipcMain.on(IPC_EMIT_CHANNEL, (_event, event: string, payload?: unknown) => {
    broadcast(event, payload);
  });

  ipcMain.handle(IPC_DIALOG_OPEN_CHANNEL, async (_event, options) => {
    const normalizedOptions = normalizeOpenDialogOptions(options);
    const result = await dialog.showOpenDialog(normalizedOptions.options);
    if (result.canceled) {
      return null;
    }
    return normalizedOptions.multipleSelection
      ? result.filePaths
      : (result.filePaths[0] ?? null);
  });

  ipcMain.handle(IPC_DIALOG_SAVE_CHANNEL, async (_event, options) => {
    const result = await dialog.showSaveDialog({
      ...normalizeSaveDialogOptions(options),
    });
    return result.canceled ? null : (result.filePath ?? null);
  });

  ipcMain.handle(IPC_SHELL_OPEN_CHANNEL, async (_event, target: string) => {
    if (/^https?:\/\//i.test(target)) {
      await shell.openExternal(target);
      return;
    }
    await shell.openPath(target);
  });

  ipcMain.handle(
    IPC_WINDOW_COMMAND_CHANNEL,
    async (event, command: string, ...args: unknown[]) => {
      const window = currentWindow(event);
      if (!window) {
        return null;
      }
      return handleWindowCommand(window, command, args);
    },
  );

  ipcMain.handle(
    IPC_GLOBAL_SHORTCUT_COMMAND_CHANNEL,
    async (_event, command: string, shortcut?: string) => {
      return handleGlobalShortcutCommand(command, shortcut);
    },
  );

  ipcMain.handle(IPC_DEEP_LINK_GET_URLS_CHANNEL, async () => pendingDeepLinks);
  ipcMain.handle(IPC_DEEP_LINK_GET_CURRENT_CHANNEL, async () =>
    pendingDeepLinks.length > 0 ? pendingDeepLinks : null,
  );
}

async function handleHostInvoke(
  event: IpcMainInvokeEvent | null,
  command: string,
  args?: Record<string, unknown>,
): Promise<unknown> {
  if (!isElectronHostCommand(command)) {
    throw new Error(`Electron host command is not supported: ${command}`);
  }
  const request = args?.request;
  if (isElectronAppServerCommand(command)) {
    if (command === "app_server_handle_json_lines") {
      return await appServerHost.handleJsonLines(
        typeof request === "object" && request !== null
          ? (request as { lines: string[] })
          : { lines: [] },
      );
    }
    return await appServerHost.drainEvents(
      typeof request === "object" && request !== null
        ? (request as { includeRecent?: boolean; limit?: number })
        : {},
    );
  }
  if (command === "sync_tray_model_shortcuts") {
    syncTrayModelShortcuts(args);
    return null;
  }
  if (command === "take_pending_skill_package_open_requests") {
    return takePendingSkillPackageOpenRequests();
  }
  if (isEmbeddedBrowserCommand(command)) {
    return await embeddedBrowserHost.invoke(
      event ? currentWindow(event) : null,
      command,
      args,
    );
  }
  if (isElectronUpdateCommand(command)) {
    return await updateHost.invoke(command, args);
  }
  return await hostCommands.invoke(command, args);
}

type HostOpenDialogOptions = {
  title?: string;
  defaultPath?: string;
  filters?: OpenDialogOptions["filters"];
  multiple?: boolean;
  directory?: boolean;
  recursive?: boolean;
};

function normalizeOpenDialogOptions(options: unknown): {
  options: OpenDialogOptions;
  multipleSelection: boolean;
} {
  const input =
    options && typeof options === "object"
      ? (options as HostOpenDialogOptions)
      : {};
  const properties: OpenDialogOptions["properties"] = [];
  if (input.directory) {
    properties.push("openDirectory");
  } else {
    properties.push("openFile");
  }
  if (input.multiple) {
    properties.push("multiSelections");
  }
  if (input.recursive) {
    properties.push("treatPackageAsDirectory");
  }

  return {
    options: {
      title: input.title,
      defaultPath: input.defaultPath,
      filters: input.filters,
      properties,
    },
    multipleSelection: Boolean(input.multiple),
  };
}

function normalizeSaveDialogOptions(options: unknown): SaveDialogOptions {
  return options && typeof options === "object"
    ? (options as SaveDialogOptions)
    : {};
}

function handleWindowCommand(
  window: BrowserWindow,
  command: string,
  args: unknown[],
): unknown {
  switch (command) {
    case "show":
      window.show();
      return null;
    case "hide":
      window.hide();
      return null;
    case "close":
      window.close();
      return null;
    case "minimize":
      window.minimize();
      return null;
    case "maximize":
      window.maximize();
      return null;
    case "unmaximize":
      window.unmaximize();
      return null;
    case "center":
      window.center();
      return null;
    case "setFocus":
      window.focus();
      return null;
    case "startDragging":
      return null;
    case "setTitle":
      window.setTitle(String(args[0] ?? ""));
      return null;
    case "setSize":
      window.setSize(Number(args[0]), Number(args[1]));
      return null;
    case "setPosition":
      window.setPosition(Number(args[0]), Number(args[1]));
      return null;
    case "isVisible":
      return window.isVisible();
    case "isMaximized":
      return window.isMaximized();
    case "isFullscreen":
      return window.isFullScreen();
    case "isDecorated":
      return !window.isFullScreen();
    case "isResizable":
      return window.isResizable();
    default:
      throw new Error(`Unsupported window command: ${command}`);
  }
}

function handleGlobalShortcutCommand(
  command: string,
  shortcut?: string,
): boolean | null {
  switch (command) {
    case "register":
      if (!shortcut) {
        throw new Error("Shortcut is required");
      }
      return globalShortcut.register(shortcut, () => {
        broadcast("global-shortcut", { shortcut });
      });
    case "unregister":
      if (shortcut) {
        globalShortcut.unregister(shortcut);
      }
      return null;
    case "unregisterAll":
      globalShortcut.unregisterAll();
      return null;
    case "isRegistered":
      return shortcut ? globalShortcut.isRegistered(shortcut) : false;
    default:
      throw new Error(`Unsupported global shortcut command: ${command}`);
  }
}

function syncTrayModelShortcuts(args?: Record<string, unknown>): void {
  trayModelShortcutsState = normalizeTrayModelShortcutsState(args);
  tray?.setContextMenu(buildTrayMenu());
}

function normalizeTrayModelShortcutsState(
  args?: Record<string, unknown>,
): TrayModelShortcutsState {
  const groups = Array.isArray(args?.quickModelGroups)
    ? args.quickModelGroups
    : [];

  return {
    currentModelProviderType: normalizeString(args?.currentModelProviderType),
    currentModelProviderLabel: normalizeString(args?.currentModelProviderLabel),
    currentModel: normalizeString(args?.currentModel),
    currentThemeLabel: normalizeString(args?.currentThemeLabel),
    quickModelGroups: groups
      .map(normalizeTrayQuickModelGroup)
      .filter((group): group is TrayQuickModelGroup => group !== null),
  };
}

function normalizeTrayQuickModelGroup(
  value: unknown,
): TrayQuickModelGroup | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const input = value as Record<string, unknown>;
  const providerType = normalizeString(input.provider_type);
  const providerLabel = normalizeString(input.provider_label) || providerType;
  const models = Array.isArray(input.models) ? input.models : [];
  const normalizedModels = models
    .map((item) =>
      normalizeTrayQuickModelItem(item, providerType, providerLabel),
    )
    .filter((item): item is TrayQuickModelItem => item !== null);

  if (!providerType || normalizedModels.length === 0) {
    return null;
  }

  return {
    provider_type: providerType,
    provider_label: providerLabel,
    models: normalizedModels,
  };
}

function normalizeTrayQuickModelItem(
  value: unknown,
  fallbackProviderType: string,
  fallbackProviderLabel: string,
): TrayQuickModelItem | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const input = value as Record<string, unknown>;
  const providerType =
    normalizeString(input.provider_type) || fallbackProviderType;
  const model = normalizeString(input.model);
  if (!providerType || !model) {
    return null;
  }

  return {
    provider_type: providerType,
    provider_label:
      normalizeString(input.provider_label) ||
      fallbackProviderLabel ||
      providerType,
    model,
  };
}

function normalizeString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function handleDeepLink(url: string): void {
  pendingDeepLinks.push(url);
  broadcast("deep-link-open-url", [url]);
}

function normalizeDeepLinkUrl(value: string): string | null {
  const trimmed = value.trim().replace(/^["']|["']$/g, "");
  if (!trimmed.startsWith("lime://")) {
    return null;
  }
  try {
    const url = new URL(trimmed);
    return url.protocol === "lime:" ? trimmed : null;
  } catch {
    return null;
  }
}

function collectDeepLinkUrls(values: string[]): string[] {
  const urls: string[] = [];
  for (const value of values) {
    const normalized = normalizeDeepLinkUrl(value);
    if (normalized && !urls.includes(normalized)) {
      urls.push(normalized);
    }
  }
  return urls;
}

function recordDeepLinkUrls(urls: string[]): void {
  for (const url of urls) {
    handleDeepLink(url);
  }
}

function collectSkillPackageOpenPaths(values: string[]): string[] {
  const paths: string[] = [];
  for (const value of values) {
    const normalized = normalizeSkillPackageOpenPath(value);
    if (normalized && !paths.includes(normalized)) {
      paths.push(normalized);
    }
  }
  return paths;
}

function normalizeSkillPackageOpenPath(value: string): string | null {
  const trimmed = value.trim().replace(/^["']|["']$/g, "");
  if (!trimmed) {
    return null;
  }

  let resolved = trimmed;
  try {
    const url = new URL(trimmed);
    if (url.protocol === "file:") {
      resolved = decodeURIComponent(url.pathname);
      if (process.platform === "win32" && url.hostname) {
        resolved = `\\\\${url.hostname}${resolved.replace(/\//g, "\\")}`;
      }
      if (process.platform === "win32" && /^\/[A-Za-z]:/.test(resolved)) {
        resolved = resolved.slice(1);
      }
    } else if (!/^[A-Za-z]:$/.test(url.protocol)) {
      return null;
    }
  } catch {
    // 普通文件路径。
  }

  const extension = path.extname(resolved).toLowerCase();
  if (extension !== ".skill" && extension !== ".skills") {
    return null;
  }
  return resolved;
}

function recordSkillPackageOpenPaths(paths: string[]): void {
  const newPaths = paths.filter(
    (pathValue) => !pendingSkillPackageOpenPaths.includes(pathValue),
  );
  if (newPaths.length === 0) {
    return;
  }

  pendingSkillPackageOpenPaths.push(...newPaths);
  if (app.isReady()) {
    showMainWindow();
    broadcast(SKILL_PACKAGE_OPEN_EVENT, newPaths);
  }
}

function takePendingSkillPackageOpenRequests(): string[] {
  return pendingSkillPackageOpenPaths.splice(0);
}

app.setAsDefaultProtocolClient("lime");
app.on("open-url", (event, url) => {
  event.preventDefault();
  handleDeepLink(url);
});
app.on("open-file", (event, filePath) => {
  event.preventDefault();
  recordSkillPackageOpenPaths(collectSkillPackageOpenPaths([filePath]));
});
recordDeepLinkUrls(collectDeepLinkUrls(process.argv));
recordSkillPackageOpenPaths(collectSkillPackageOpenPaths(process.argv));

const singleInstanceLock =
  !isWindowsSquirrelStartup &&
  (process.env.LIME_ELECTRON_E2E === "1" ||
    process.env.LIME_ELECTRON_SMOKE === "1" ||
    app.requestSingleInstanceLock());
if (isWindowsSquirrelStartup) {
  // Windows Squirrel installer events are handled before the normal app boot path.
} else if (!singleInstanceLock) {
  app.quit();
} else {
  app.on("second-instance", (_event, argv) => {
    showMainWindow();
    recordDeepLinkUrls(collectDeepLinkUrls(argv));
    recordSkillPackageOpenPaths(collectSkillPackageOpenPaths(argv));
  });

  app.whenReady().then(() => {
    configureApplicationIdentity();
    registerLocalAssetProtocol();
    installMainWindowMediaPermissionHandler({
      session: session.defaultSession,
      getMainWindow: () => mainWindow,
    });
    registerIpcHandlers();
    devHttpBridge = startDevHttpBridge();
    tray = createTray();
    void warmupAppServer();
    createMainWindow();
  });
}

app.on("window-all-closed", () => {
  if (isQuitting || process.env.LIME_ELECTRON_SMOKE === "1") {
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createMainWindow();
  }
});

app.on("before-quit", () => {
  isQuitting = true;
  globalShortcut.unregisterAll();
  tray?.destroy();
  tray = null;
  devHttpBridge?.stop();
  devHttpBridge = null;
  void appServerHost.stop();
});

function handleWindowsSquirrelStartup(): boolean {
  const plan = resolveWindowsSquirrelStartupPlan({
    argv: process.argv,
    execPath: process.execPath,
    platform: process.platform,
  });
  if (!plan) {
    return false;
  }
  if (plan.action === "quit") {
    app.quit();
    return true;
  }

  try {
    spawn(plan.updateExecutable, plan.args, { detached: true }).once(
      "error",
      (error) => {
        console.error(
          `[electron-host] Squirrel Update.exe failed: ${error.message}`,
        );
      },
    );
    // Squirrel gives lifecycle handlers a short deadline. Update.exe is detached,
    // so the host should not wait for its process lifetime before quitting.
    setTimeout(() => app.quit(), 1_000);
  } catch (error) {
    console.error(
      `[electron-host] unable to start Squirrel Update.exe: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    app.quit();
  }
  return true;
}

async function warmupAppServer(): Promise<void> {
  try {
    const response = await appServerHost.warmup();
    console.log(
      `[electron-host] app-server ready protocol=${response.serverInfo.protocolVersion} version=${response.serverInfo.version}`,
    );
  } catch (error) {
    console.error(
      `[electron-host] app-server warmup failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

function resolveDesktopAsset(
  sourceRelativePath: string,
  packagedName: string,
): string {
  const packagedPath = path.join(
    process.resourcesPath,
    "desktop-assets",
    packagedName,
  );
  if (existsSync(packagedPath)) {
    return packagedPath;
  }

  const builtAssetPath = path.resolve(
    app.getAppPath(),
    "dist-electron/desktop-assets",
    packagedName,
  );
  if (existsSync(builtAssetPath)) {
    return builtAssetPath;
  }

  const appPath = app.getAppPath();
  const repoPath = path.resolve(appPath, sourceRelativePath);
  if (existsSync(repoPath)) {
    return repoPath;
  }

  return path.resolve(process.cwd(), sourceRelativePath);
}

function resolveStartupIconDataUrl(): string | null {
  const iconPath = resolveDesktopAsset(APP_ICON_SOURCE, APP_ICON_PACKAGED_NAME);
  try {
    const icon = readFileSync(iconPath);
    return `data:image/png;base64,${icon.toString("base64")}`;
  } catch {
    return null;
  }
}

function configureApplicationIdentity(): void {
  app.setName(APP_NAME);
  if (process.platform === "win32") {
    app.setAppUserModelId(APP_BUNDLE_IDENTIFIER);
  }

  const appIconPath = resolveDesktopAsset(
    APP_ICON_SOURCE,
    APP_ICON_PACKAGED_NAME,
  );
  const appIcon = nativeImage.createFromPath(appIconPath);
  if (process.platform === "darwin") {
    if (!appIcon.isEmpty()) {
      app.dock?.setIcon(appIcon);
    }
    app.setAboutPanelOptions({
      applicationName: APP_NAME,
      applicationVersion: app.getVersion(),
      copyright: "Copyright © Lime",
      iconPath: appIconPath,
    });
  }
}

function startDevHttpBridge(): ElectronDevHttpBridge | null {
  const devServerUrl = process.env.VITE_DEV_SERVER_URL?.trim();
  if (!devServerUrl || process.env.LIME_ELECTRON_DEV_HTTP_BRIDGE === "0") {
    return null;
  }

  const bridge = new ElectronDevHttpBridge({
    invoke: (command, args) => handleHostInvoke(null, command, args),
    host: process.env.LIME_ELECTRON_DEV_HTTP_BRIDGE_HOST?.trim() || undefined,
    port: parseDevHttpBridgePort(
      process.env.LIME_ELECTRON_DEV_HTTP_BRIDGE_PORT,
    ),
  });
  bridge.start();
  return bridge;
}

function parseDevHttpBridgePort(value: string | undefined): number | undefined {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 && parsed < 65536
    ? parsed
    : undefined;
}
