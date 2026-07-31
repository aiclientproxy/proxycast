import { app, autoUpdater } from "./electronRuntime";
import {
  assessUpdateCandidate,
  normalizeUpdateVersion,
  type UpdateCandidateAssessment,
} from "./updateVersion";

type UpdateInstallStage =
  | "idle"
  | "checking"
  | "downloading"
  | "installing"
  | "restarting"
  | "completed"
  | "failed"
  | "up_to_date";

type UpdateInstallSession = {
  sessionId: string;
  stage: UpdateInstallStage;
  currentVersion: string;
  latestVersion?: string | null;
  downloadUrl?: string | null;
  downloadedBytes: number;
  totalBytes?: number | null;
  percent: number;
  message: string;
  error?: string | null;
  startedAt: number;
  updatedAt: number;
  completedAt?: number | null;
  canCloseWindow: boolean;
  isActive: boolean;
};

type VersionInfo = {
  current: string;
  latest?: string | null;
  hasUpdate: boolean;
  downloadUrl?: string | null;
  releaseNotes?: string | null;
  releaseNotesUrl?: string | null;
  pubDate?: string | null;
  error?: string | null;
};

type CheckForUpdatesOptions = {
  automatic?: boolean;
};

type UpdateNotificationAnchorRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type UpdateWindowOpenParams = VersionInfo & {
  anchorRect?: UpdateNotificationAnchorRect | null;
};

type DownloadUpdateResult = {
  success: boolean;
  message: string;
  filePath?: string | null;
};

type UpdateInfo = {
  releaseDate?: string;
  releaseName?: string;
  releaseNotes?: string;
  updateURL?: string;
  version?: string;
};

type Broadcast = (event: string, payload?: unknown) => void;
type UpdateWindowController = {
  open: (params?: UpdateWindowOpenParams | null) => Promise<void> | void;
  close: () => Promise<void> | void;
};

const UPDATE_INSTALL_SESSION_EVENT = "app-update://session";
const MAC_UPDATE_MANIFEST_NAME = "RELEASES.json";
const DEFAULT_UPDATE_BASE_URL = "https://updates.limecloud.com";
const DEV_MAC_APP_PATH_SEGMENT = "/Lime-dev.app/Contents/MacOS/";

export class ElectronUpdateHost {
  readonly #broadcast: Broadcast;
  readonly #windowController: UpdateWindowController;
  #configured = false;
  #checkPromise: Promise<UpdateInfo | null> | null = null;
  #session: UpdateInstallSession;
  #latestInfo: UpdateInfo | null = null;

  constructor(
    broadcast: Broadcast,
    windowController: UpdateWindowController = {
      open: () => undefined,
      close: () => undefined,
    },
  ) {
    this.#broadcast = broadcast;
    this.#windowController = windowController;
    this.#session = this.#idleSession();
  }

  async invoke(
    command: string,
    args?: Record<string, unknown>,
  ): Promise<unknown> {
    switch (command) {
      case "check_for_updates":
        return await this.checkForUpdates(parseCheckForUpdatesOptions(args));
      case "download_update":
        return await this.downloadUpdate();
      case "start_update_install_session":
        return await this.startInstallSession();
      case "get_update_install_session":
        return this.#session;
      case "get_update_check_settings":
        return this.#updateSettings();
      case "set_update_check_settings":
        return null;
      case "get_update_notification_metrics":
        return this.#updateMetrics();
      case "dismiss_update_notification":
      case "remind_update_later":
        return Math.floor(Date.now() / 1000) + 24 * 3600;
      case "record_update_notification_action":
      case "skip_update_version":
        return null;
      case "close_update_window":
        await this.#windowController.close();
        return null;
      case "open_update_window":
      case "test_update_window":
        await this.#windowController.open({
          ...this.#currentVersionInfo(),
          anchorRect: parseUpdateNotificationAnchorRect(args?.anchorRect),
        });
        return null;
      default:
        throw new Error(
          `Electron update command is not implemented: ${command}`,
        );
    }
  }

  async checkForUpdates(
    options: CheckForUpdatesOptions = {},
  ): Promise<VersionInfo> {
    if (!this.#canUseUpdater()) {
      if (options.automatic) {
        return this.#currentVersionInfo();
      }
      return {
        current: app.getVersion(),
        hasUpdate: false,
        error: "Electron updater is only enabled for packaged builds.",
      };
    }
    this.#configure();
    if (this.#hasConfirmedUpdate()) {
      return this.#currentVersionInfo();
    }

    if (!this.#checkPromise) {
      this.#setSession({
        stage: "checking",
        latestVersion: null,
        downloadUrl: null,
        error: null,
        message: "正在检查更新",
        isActive: true,
        completedAt: null,
      });
    }
    try {
      const info = await this.#checkForUpdatesOnce();
      this.#latestInfo = info;
      const assessment = this.#assessUpdateInfo(info);
      if (assessment.status === "invalid") {
        throw new Error("更新清单未提供有效的候选版本");
      }
      const hasUpdate = assessment.status === "newer";
      this.#setSession({
        stage: hasUpdate ? "completed" : "up_to_date",
        latestVersion: assessment.version,
        downloadUrl: hasUpdate ? this.#downloadUrl(info) : null,
        percent: hasUpdate ? 1 : 0,
        error: null,
        message: hasUpdate ? "更新已下载，准备安装" : "当前已是最新版本",
        isActive: false,
        completedAt: Date.now(),
      });
      return this.#versionInfo(info);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.#setSession({
        stage: "failed",
        error: message,
        message,
        isActive: false,
        completedAt: Date.now(),
      });
      return {
        current: app.getVersion(),
        hasUpdate: false,
        error: message,
      };
    }
  }

  async downloadUpdate(): Promise<DownloadUpdateResult> {
    if (!this.#canUseUpdater()) {
      return {
        success: false,
        message:
          "Electron updater is only enabled for packaged macOS and Windows builds.",
      };
    }
    this.#configure();

    if (this.#session.stage === "completed") {
      return {
        success: true,
        message: "更新已下载，准备安装",
        filePath: null,
      };
    }
    if (this.#isInstallingUpdate()) {
      return {
        success: true,
        message: "更新安装已开始",
        filePath: null,
      };
    }

    const versionInfo = await this.checkForUpdates();
    if (versionInfo.hasUpdate) {
      return {
        success: true,
        message: "更新已下载，准备安装",
        filePath: null,
      };
    }
    return {
      success: false,
      message: versionInfo.error || "当前已是最新版本",
      filePath: null,
    };
  }

  async startInstallSession(): Promise<UpdateInstallSession> {
    if (!this.#canUseUpdater()) {
      this.#setSession({
        stage: "failed",
        error:
          "Electron updater is only enabled for packaged macOS and Windows builds.",
        message:
          "Electron updater is only enabled for packaged macOS and Windows builds.",
        isActive: false,
        completedAt: Date.now(),
      });
      return this.#session;
    }
    this.#configure();

    try {
      if (this.#isInstallingUpdate()) {
        return this.#session;
      }
      if (this.#session.stage !== "completed") {
        if (!this.#checkPromise) {
          this.#setSession({
            stage: "checking",
            latestVersion: null,
            downloadUrl: null,
            error: null,
            message: "正在检查并下载更新",
            isActive: true,
            completedAt: null,
          });
        }
        const info = await this.#checkForUpdatesOnce();
        this.#latestInfo = info;
        const assessment = this.#assessUpdateInfo(info);
        if (assessment.status === "invalid") {
          throw new Error("更新清单未提供有效的候选版本");
        }
        if (assessment.status !== "newer") {
          this.#setSession({
            stage: "up_to_date",
            latestVersion: assessment.version,
            downloadUrl: null,
            percent: 0,
            error: null,
            message: "当前已是最新版本",
            isActive: false,
            completedAt: Date.now(),
          });
          return this.#session;
        }
      }
      if (this.#isInstallingUpdate()) {
        return this.#session;
      }
      const installAssessment = this.#assessUpdateInfo(this.#latestInfo);
      if (installAssessment.status !== "newer") {
        throw new Error("更新版本未通过安装前校验");
      }
      this.#setSession({
        stage: "restarting",
        latestVersion: installAssessment.version,
        error: null,
        message: "正在重启并安装更新",
        isActive: true,
        canCloseWindow: false,
      });
      setTimeout(() => autoUpdater.quitAndInstall(), 250);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.#setSession({
        stage: "failed",
        error: message,
        message,
        isActive: false,
        completedAt: Date.now(),
      });
    }
    return this.#session;
  }

  #configure(): void {
    if (this.#configured) {
      return;
    }
    this.#configured = true;
    if (!isAutoUpdaterSupportedPlatform(process.platform)) {
      return;
    }
    autoUpdater.setFeedURL({
      serverType: process.platform === "darwin" ? "json" : "default",
      url: updateFeedUrlForPlatform(
        runtimeUpdateFeedUrl(process.platform, process.arch),
        process.platform,
      ),
    });
    autoUpdater.on("update-available", () => {
      this.#latestInfo = null;
      this.#setSession({
        stage: "downloading",
        latestVersion: null,
        downloadUrl: null,
        error: null,
        message: "正在下载并验证更新",
        isActive: true,
        completedAt: null,
      });
    });
    autoUpdater.on("update-not-available", () => {
      this.#latestInfo = null;
      this.#setSession({
        stage: "up_to_date",
        latestVersion: null,
        downloadUrl: null,
        percent: 0,
        error: null,
        message: "当前已是最新版本",
        isActive: false,
        completedAt: Date.now(),
      });
    });
    autoUpdater.on(
      "update-downloaded",
      (_event, releaseNotes, releaseName, releaseDate, updateURL) => {
        const info = buildUpdateInfo(
          releaseNotes,
          releaseName,
          releaseDate,
          updateURL,
        );
        this.#applyDownloadedUpdate(info);
      },
    );
    autoUpdater.on("error", (error) => {
      const message = error instanceof Error ? error.message : String(error);
      this.#setSession({
        stage: "failed",
        error: message,
        message,
        isActive: false,
        completedAt: Date.now(),
      });
    });
  }

  #canUseUpdater(): boolean {
    if (!isAutoUpdaterSupportedPlatform(process.platform)) {
      return false;
    }
    if (process.env.LIME_ELECTRON_ENABLE_DEV_UPDATER === "1") {
      return true;
    }
    if (isElectronAutomationSession()) {
      return false;
    }
    if (process.env.VITE_DEV_SERVER_URL) {
      return false;
    }
    if (isDevelopmentPackagedApp(process.execPath, process.platform)) {
      return false;
    }
    return app.isPackaged;
  }

  #versionInfo(info: UpdateInfo | null): VersionInfo {
    const assessment = this.#assessUpdateInfo(info);
    const hasUpdate = assessment.status === "newer";
    return {
      current: app.getVersion(),
      latest: assessment.version,
      hasUpdate,
      downloadUrl: hasUpdate ? this.#downloadUrl(info) : null,
      releaseNotes:
        typeof info?.releaseNotes === "string" ? info.releaseNotes : null,
      releaseNotesUrl: null,
      pubDate: info?.releaseDate ?? null,
      error: null,
    };
  }

  #currentVersionInfo(): VersionInfo {
    return this.#versionInfo(this.#latestInfo);
  }

  #downloadUrl(info: UpdateInfo | null): string | null {
    return info?.updateURL ?? null;
  }

  #hasConfirmedUpdate(): boolean {
    return this.#session.stage === "completed" || this.#isInstallingUpdate();
  }

  #isInstallingUpdate(): boolean {
    return (
      this.#session.stage === "installing" ||
      this.#session.stage === "restarting"
    );
  }

  async #checkForUpdatesOnce(): Promise<UpdateInfo | null> {
    if (this.#checkPromise) {
      return await this.#checkPromise;
    }
    const promise = new Promise<UpdateInfo | null>((resolve, reject) => {
      const cleanup = () => {
        autoUpdater.removeListener("update-downloaded", onDownloaded);
        autoUpdater.removeListener("update-not-available", onNotAvailable);
        autoUpdater.removeListener("error", onError);
      };
      const onDownloaded = (
        _event: Electron.Event,
        releaseNotes: string,
        releaseName: string,
        releaseDate: Date,
        updateURL: string,
      ) => {
        cleanup();
        const info = buildUpdateInfo(
          releaseNotes,
          releaseName,
          releaseDate,
          updateURL,
        );
        if (this.#assessUpdateInfo(info).status === "invalid") {
          reject(new Error("更新清单未提供有效的候选版本"));
          return;
        }
        resolve(info);
      };
      const onNotAvailable = () => {
        cleanup();
        resolve(null);
      };
      const onError = (error: Error) => {
        cleanup();
        reject(error);
      };
      autoUpdater.once("update-downloaded", onDownloaded);
      autoUpdater.once("update-not-available", onNotAvailable);
      autoUpdater.once("error", onError);
      try {
        autoUpdater.checkForUpdates();
      } catch (error) {
        cleanup();
        reject(error);
      }
    }).finally(() => {
      this.#checkPromise = null;
    });
    this.#checkPromise = promise;
    return await promise;
  }

  #assessUpdateInfo(info: UpdateInfo | null): UpdateCandidateAssessment {
    if (!info) {
      return { status: "not_newer", version: null };
    }
    return assessUpdateCandidate(info.version, app.getVersion());
  }

  #applyDownloadedUpdate(info: UpdateInfo): void {
    const assessment = this.#assessUpdateInfo(info);
    if (assessment.status === "invalid") {
      this.#latestInfo = null;
      this.#setSession({
        stage: "failed",
        latestVersion: null,
        downloadUrl: null,
        percent: 0,
        error: "更新清单未提供有效的候选版本",
        message: "更新清单未提供有效的候选版本",
        isActive: false,
        completedAt: Date.now(),
      });
      return;
    }

    this.#latestInfo = info;
    const hasUpdate = assessment.status === "newer";
    this.#setSession({
      stage: hasUpdate ? "completed" : "up_to_date",
      latestVersion: assessment.version,
      downloadUrl: hasUpdate ? this.#downloadUrl(info) : null,
      percent: hasUpdate ? 1 : 0,
      error: null,
      message: hasUpdate ? "更新已下载，准备安装" : "当前已是最新版本",
      isActive: false,
      completedAt: Date.now(),
    });
  }

  #setSession(update: Partial<UpdateInstallSession>): void {
    this.#session = {
      ...this.#session,
      ...update,
      updatedAt: Date.now(),
    };
    this.#broadcast(UPDATE_INSTALL_SESSION_EVENT, this.#session);
  }

  #idleSession(): UpdateInstallSession {
    const now = Date.now();
    return {
      sessionId: `electron-update-${now}`,
      stage: "idle",
      currentVersion: app.getVersion(),
      latestVersion: null,
      downloadUrl: null,
      downloadedBytes: 0,
      totalBytes: null,
      percent: 0,
      message: "尚未开始更新",
      error: null,
      startedAt: now,
      updatedAt: now,
      completedAt: null,
      canCloseWindow: true,
      isActive: false,
    };
  }

  #updateSettings() {
    return {
      enabled: true,
      check_interval_hours: 24,
      show_notification: true,
      last_check_timestamp: 0,
      skipped_version: null,
      remind_later_until: null,
    };
  }

  #updateMetrics() {
    return {
      shown_count: 0,
      update_now_count: 0,
      remind_later_count: 0,
      skip_version_count: 0,
      dismiss_count: 0,
      update_now_rate: 0,
      remind_later_rate: 0,
      skip_version_rate: 0,
      dismiss_rate: 0,
    };
  }
}

function parseCheckForUpdatesOptions(
  value: Record<string, unknown> | undefined,
): CheckForUpdatesOptions {
  return {
    automatic: value?.automatic === true,
  };
}

function parseUpdateNotificationAnchorRect(
  value: unknown,
): UpdateNotificationAnchorRect | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const input = value as Record<string, unknown>;
  const x = Number(input.x);
  const y = Number(input.y);
  const width = Number(input.width);
  const height = Number(input.height);
  if (
    !Number.isFinite(x) ||
    !Number.isFinite(y) ||
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0
  ) {
    return null;
  }

  return { x, y, width, height };
}

function normalizeUpdateDate(
  value: Date | string | undefined,
): string | undefined {
  if (value instanceof Date) {
    return value.toISOString();
  }
  return typeof value === "string" ? value : undefined;
}

function buildUpdateInfo(
  releaseNotes: string | undefined,
  releaseName: string | undefined,
  releaseDate: Date | string | undefined,
  updateURL: string | undefined,
): UpdateInfo {
  return {
    releaseDate: normalizeUpdateDate(releaseDate),
    releaseName,
    releaseNotes: typeof releaseNotes === "string" ? releaseNotes : undefined,
    updateURL,
    version: normalizeUpdateVersion(releaseName) ?? undefined,
  };
}

function isAutoUpdaterSupportedPlatform(platform: NodeJS.Platform): boolean {
  return platform === "darwin" || platform === "win32";
}

function isDevelopmentPackagedApp(
  execPath: string,
  platform: NodeJS.Platform,
): boolean {
  if (platform !== "darwin") {
    return false;
  }
  return execPath.replace(/\\/g, "/").includes(DEV_MAC_APP_PATH_SEGMENT);
}

function isElectronAutomationSession(): boolean {
  return (
    process.env.LIME_ELECTRON_SMOKE === "1" ||
    process.env.LIME_ELECTRON_E2E === "1"
  );
}

function runtimeUpdateFeedUrl(
  platform: NodeJS.Platform,
  arch: NodeJS.Architecture,
): string {
  const explicitFeedUrl = process.env.LIME_ELECTRON_UPDATES_URL?.trim();
  if (explicitFeedUrl) {
    return explicitFeedUrl.replace(/\/+$/, "");
  }
  const baseUrl = (process.env.LIME_UPDATES_BASE_URL || DEFAULT_UPDATE_BASE_URL)
    .trim()
    .replace(/\/+$/, "");
  return `${baseUrl}/lime/stable/${updateFeedLabel(platform, arch)}`;
}

function updateFeedLabel(
  platform: NodeJS.Platform,
  arch: NodeJS.Architecture,
): string {
  if (platform === "darwin") {
    return arch === "arm64" ? "darwin-arm64" : "darwin-x64";
  }
  if (platform === "win32") {
    return arch === "arm64" ? "win32-arm64" : "win32-x64";
  }
  return `${platform}-${arch}`;
}

function updateFeedUrlForPlatform(
  feedUrl: string,
  platform: NodeJS.Platform,
): string {
  const normalized = feedUrl.trim().replace(/\/+$/, "");
  if (platform !== "darwin") {
    return normalized;
  }
  if (normalized.endsWith(`/${MAC_UPDATE_MANIFEST_NAME}`)) {
    return normalized;
  }
  return `${normalized}/${MAC_UPDATE_MANIFEST_NAME}`;
}
