import os from "node:os";
import path from "node:path";

export const LIME_COMPANY_DATA_DIR_NAME = "LimeCloud";
export const LIME_APP_DATA_DIR_NAME = "lime";
export const LIME_HOST_DATA_DIR_NAME = "lime";
export const LIME_AGENT_ROOT_DIR_NAME = "app-server";
export const LIME_HOST_SESSION_DIR_NAME = "host-session";
export const WINDOWS_SQUIRREL_INSTALL_DIR_NAME = "lime";

export type AppDataRootOptions = {
  platform: NodeJS.Platform | string;
  hostUserData: string;
  appDataRootOverride?: string;
  localAppData?: string;
  home?: string;
};

export type AgentRootOptions = AppDataRootOptions & {
  agentRootOverride?: string;
};

export type DesktopStorageRootOptions = AgentRootOptions & {
  e2eMode: boolean;
  e2eUserDataDir?: string;
};

export type DesktopStorageRoots = {
  appDataRoot: string;
  agentRoot: string;
  hostSessionData: string;
};

/** Windows roaming data 只保存 host profile，不能成为 durable Agent root。 */
export function resolveAppDataRoot(options: AppDataRootOptions): string {
  const pathApi = pathForPlatform(options.platform);
  const override = nonEmptyPath(options.appDataRootOverride);
  if (override) {
    return assertOutsideWindowsInstallRoot(pathApi.resolve(override), options);
  }

  if (options.platform !== "win32") {
    return pathApi.resolve(options.hostUserData);
  }

  const localAppData = nonEmptyPath(options.localAppData);
  if (localAppData) {
    return assertOutsideWindowsInstallRoot(
      pathApi.resolve(
        localAppData,
        LIME_COMPANY_DATA_DIR_NAME,
        LIME_APP_DATA_DIR_NAME,
      ),
      options,
    );
  }

  const home = nonEmptyPath(options.home);
  if (home) {
    return assertOutsideWindowsInstallRoot(
      pathApi.resolve(
        home,
        "AppData",
        "Local",
        LIME_COMPANY_DATA_DIR_NAME,
        LIME_APP_DATA_DIR_NAME,
      ),
      options,
    );
  }

  throw new Error(
    "无法解析 Windows AppDataRoot：LOCALAPPDATA 或 Electron home 路径缺失",
  );
}

export function resolveAgentRoot(options: AgentRootOptions): string {
  const pathApi = pathForPlatform(options.platform);
  const override = nonEmptyPath(options.agentRootOverride);
  if (override) {
    return assertOutsideWindowsInstallRoot(pathApi.resolve(override), options);
  }

  return pathApi.join(resolveAppDataRoot(options), LIME_AGENT_ROOT_DIR_NAME);
}

/**
 * Chromium cookies/storage/network state 与 cache 的唯一根。
 * macOS 与 host profile 物理同根；Windows 必须离开 roaming `%APPDATA%`，
 * 落在 `<AppDataRoot>\host-session`。
 */
export function resolveHostSessionData(options: AppDataRootOptions): string {
  const pathApi = pathForPlatform(options.platform);
  if (options.platform !== "win32") {
    return pathApi.resolve(options.hostUserData);
  }

  return pathApi.join(
    resolveAppDataRoot(options),
    LIME_HOST_SESSION_DIR_NAME,
  );
}

export function resolveDesktopStorageRoots(
  options: DesktopStorageRootOptions,
): DesktopStorageRoots {
  const pathApi = pathForPlatform(options.platform);
  const e2eRoot = nonEmptyPath(options.e2eUserDataDir);
  if (options.e2eMode && !e2eRoot) {
    throw new Error(
      "E2E 模式缺少 ELECTRON_E2E_USER_DATA_DIR，拒绝解析真实数据根",
    );
  }

  const appDataRoot = resolveAppDataRoot({
    ...options,
    appDataRootOverride: options.e2eMode
      ? e2eRoot
      : options.appDataRootOverride,
  });
  const agentRoot = options.e2eMode
    ? pathApi.join(appDataRoot, LIME_AGENT_ROOT_DIR_NAME)
    : resolveAgentRoot({
        ...options,
        appDataRootOverride: appDataRoot,
      });
  // E2E 下 host profile 就是隔离根，不能让 sessionData 回到真实 userData。
  const hostSessionData = resolveHostSessionData({
    ...options,
    hostUserData: options.e2eMode ? appDataRoot : options.hostUserData,
    appDataRootOverride: appDataRoot,
  });

  return { appDataRoot, agentRoot, hostSessionData };
}

export function resolveCurrentDesktopStorageRoots(
  hostUserData: string,
): DesktopStorageRoots {
  return resolveDesktopStorageRoots({
    platform: process.platform,
    hostUserData,
    localAppData: process.env.LOCALAPPDATA,
    home: os.homedir(),
    e2eMode: process.env.LIME_ELECTRON_E2E === "1",
    e2eUserDataDir: process.env.ELECTRON_E2E_USER_DATA_DIR,
    agentRootOverride: process.env.LIME_AGENT_RUNTIME_ROOT,
  });
}

function pathForPlatform(platform: NodeJS.Platform | string): typeof path {
  return platform === "win32" ? path.win32 : path.posix;
}

function nonEmptyPath(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function assertOutsideWindowsInstallRoot(
  candidate: string,
  options: AppDataRootOptions,
): string {
  if (options.platform !== "win32") {
    return candidate;
  }

  const home = nonEmptyPath(options.home);
  const localAppData =
    nonEmptyPath(options.localAppData) ??
    (home ? path.win32.join(home, "AppData", "Local") : undefined);
  if (!localAppData) {
    return candidate;
  }

  const installRoot = path.win32.resolve(
    localAppData,
    WINDOWS_SQUIRREL_INSTALL_DIR_NAME,
  );
  if (isSameOrDescendantWindowsPath(candidate, installRoot)) {
    throw new Error(`Windows 数据根不能位于 Squirrel 安装根 ${installRoot}`);
  }
  return candidate;
}

function isSameOrDescendantWindowsPath(
  candidate: string,
  root: string,
): boolean {
  const normalizedCandidate = path.win32.resolve(candidate).toLowerCase();
  const normalizedRoot = path.win32.resolve(root).toLowerCase();
  return (
    normalizedCandidate === normalizedRoot ||
    normalizedCandidate.startsWith(`${normalizedRoot}\\`)
  );
}
