/**
 * @file update-notification.tsx
 * @description 更新提醒独立窗口页面
 *
 * 独立于主应用的更新提醒悬浮窗口，采用轻量 toast 形态展示更新操作。
 *
 * input: URL 参数（current, latest, download_url）
 * output: 更新提醒 UI
 * pos: pages 层，独立 Desktop Host 窗口
 */

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useState,
  type MouseEvent,
} from "react";
import { getCurrentWindow } from "@/lib/desktop-host/window";
import { openExternalUrlWithSystemBrowser } from "@/lib/api/externalUrl";
import {
  closeUpdateWindow,
  dismissUpdateNotification,
  getUpdateInstallSession,
  isUpdateInstallSessionActive,
  listenUpdateInstallSession,
  recordUpdateNotificationAction,
  remindUpdateLater,
  startUpdateInstallSession,
  type UpdateInstallSession,
} from "@/lib/api/appUpdate";
import { Bell, Download, RefreshCw, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import "./update-notification.css";

interface UpdateParams {
  currentVersion: string;
  latestVersion: string;
  downloadUrl: string;
}

function getUpdateParamsFromUrl(): UpdateParams {
  const params = new URLSearchParams(window.location.search);
  return {
    currentVersion: params.get("current") || "",
    latestVersion: params.get("latest") || "",
    downloadUrl: params.get("download_url") || "",
  };
}

function clampProgressPercent(percent: number | null | undefined): number {
  if (!Number.isFinite(percent)) {
    return 0;
  }
  return Math.min(100, Math.max(0, Math.round((percent ?? 0) * 100)));
}

export function UpdateNotificationPage() {
  const { t } = useTranslation("common");
  const [params, setParams] = useState<UpdateParams>({
    currentVersion: "",
    latestVersion: "",
    downloadUrl: "",
  });
  const [installSession, setInstallSession] =
    useState<UpdateInstallSession | null>(null);
  const [visible, setVisible] = useState(false);
  const [closing, setClosing] = useState(false);
  const installActive = isUpdateInstallSessionActive(installSession);
  const hasInstallSession =
    installSession !== null && installSession.stage !== "idle";
  const progressPercent = clampProgressPercent(installSession?.percent);
  const latestVersion =
    installSession?.latestVersion || params.latestVersion || "";
  const currentVersion =
    installSession?.currentVersion || params.currentVersion || "";
  const downloadUrl = installSession?.downloadUrl || params.downloadUrl || "";
  const closeLabel = t("common.updateNotification.action.close");
  const hideLabel = t("common.updateNotification.action.hide");
  const failed = installSession?.stage === "failed";
  const upToDate = installSession?.stage === "up_to_date";
  const sessionProgressLabel = (() => {
    if (!installSession) {
      return "";
    }

    switch (installSession.stage) {
      case "checking":
        return t("common.updateNotification.progress.checking");
      case "downloading":
        return installSession.totalBytes
          ? t("common.updateNotification.progress.downloading", {
              percent: `${progressPercent}%`,
            })
          : t("common.updateNotification.progress.downloadingUnknown");
      case "installing":
        return t("common.updateNotification.progress.installing");
      case "restarting":
        return t("common.updateNotification.progress.restarting");
      case "failed":
        return t("common.updateNotification.progress.failed");
      case "up_to_date":
        return t("common.updateNotification.progress.upToDate");
      case "completed":
        return t("common.updateNotification.progress.completed");
      case "idle":
      default:
        return "";
    }
  })();
  const sessionActionLabel = (() => {
    if (!installSession) {
      return t("common.updateNotification.action.updateNow");
    }

    switch (installSession.stage) {
      case "checking":
        return t("common.updateNotification.progress.checking");
      case "downloading":
        return t("common.updateNotification.action.downloading");
      case "installing":
        return t("common.updateNotification.action.installing");
      case "restarting":
        return t("common.updateNotification.action.restarting");
      case "failed":
        return t("common.updateNotification.action.retry");
      case "completed":
        return t("common.updateNotification.action.restartToUpdate");
      default:
        return t("common.updateNotification.action.updateNow");
    }
  })();
  const notificationTitle = failed
    ? t("common.updateNotification.progress.failed")
    : upToDate
      ? t("common.updateNotification.progress.upToDate")
      : latestVersion
        ? t("common.updateNotification.version.new", {
            version: latestVersion,
          })
        : sessionProgressLabel ||
          t("common.updateNotification.progress.checking");

  useLayoutEffect(() => {
    const previousWindow = document.documentElement.dataset.limeWindow;
    document.documentElement.dataset.limeWindow = "update-notification";

    return () => {
      if (previousWindow === undefined) {
        delete document.documentElement.dataset.limeWindow;
        return;
      }
      document.documentElement.dataset.limeWindow = previousWindow;
    };
  }, []);

  useEffect(() => {
    setParams(getUpdateParamsFromUrl());
    const timer = window.setTimeout(() => setVisible(true), 10);

    let disposed = false;
    void getUpdateInstallSession()
      .then((session) => {
        if (!disposed && session.stage !== "idle") {
          setInstallSession(session);
        }
      })
      .catch((error) => {
        console.error("读取更新安装会话失败:", error);
      });

    const unlistenPromise = listenUpdateInstallSession((session) => {
      setInstallSession(session);
    });

    return () => {
      disposed = true;
      window.clearTimeout(timer);
      void unlistenPromise
        .then((unlisten) => unlisten())
        .catch((error) => {
          console.error("取消更新安装会话监听失败:", error);
        });
    };
  }, []);

  // 直接关闭窗口（无动画）
  const closeWindow = useCallback(async () => {
    try {
      await closeUpdateWindow();
    } catch (err) {
      console.error("关闭窗口失败:", err);
      // 备用方案：直接关闭
      await getCurrentWindow().close();
    }
  }, []);

  // 带动画关闭
  const closeWithAnimation = useCallback(async () => {
    if (closing) return;
    setClosing(true);
    await new Promise((resolve) => window.setTimeout(resolve, 160));
    await closeWindow();
  }, [closing, closeWindow]);

  // 关闭并应用退避策略
  const handleDismiss = useCallback(async () => {
    if (installActive) {
      await closeWithAnimation();
      return;
    }

    try {
      await dismissUpdateNotification(latestVersion || null);
    } catch (error) {
      console.error("记录关闭提醒失败:", error);
    }
    await closeWithAnimation();
  }, [installActive, latestVersion, closeWithAnimation]);

  // ESC 关闭窗口
  useEffect(() => {
    const handleKeyDown = async (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        await handleDismiss();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleDismiss]);

  // 开始拖动窗口
  const handleStartDrag = useCallback(async (e: MouseEvent) => {
    if (e.button !== 0) return;
    try {
      await getCurrentWindow().startDragging();
    } catch (err) {
      console.error("拖动窗口失败:", err);
    }
  }, []);

  // 立即更新
  const handleDownload = async () => {
    try {
      await recordUpdateNotificationAction("update_now");
    } catch (error) {
      console.error("记录立即更新行为失败:", error);
    }

    try {
      const session = await startUpdateInstallSession();
      setInstallSession(session);
    } catch (error) {
      console.error("启动更新安装失败:", error);
      if (downloadUrl) {
        try {
          await openExternalUrlWithSystemBrowser(downloadUrl);
        } catch (openError) {
          console.error("打开更新下载链接失败:", openError);
        }
      }
    }
  };

  // 稍后提醒
  const handleLater = async (hours: number) => {
    if (installActive) return;

    try {
      await remindUpdateLater(hours);
    } catch (error) {
      console.error("设置稍后提醒失败:", error);
    }
    await closeWithAnimation();
  };

  return (
    <div className="update-container">
      <div
        className={`update-toast ${visible ? "is-visible" : ""} ${
          closing ? "is-closing" : ""
        }`}
        onMouseDown={handleStartDrag}
      >
        <div className="update-toast-icon" aria-hidden>
          <Bell size={14} />
        </div>

        <div className="update-toast-main">
          <div className="update-toast-top">
            <div className="update-toast-message">
              {notificationTitle}
              {currentVersion && !failed && !upToDate && latestVersion ? (
                <span className="update-toast-sub">
                  {t("common.updateNotification.version.current", {
                    version: currentVersion,
                  })}
                </span>
              ) : null}
            </div>

            <div
              className="update-toast-actions"
              onMouseDown={(e) => e.stopPropagation()}
            >
              {installActive ? (
                <>
                  <button
                    onClick={handleDismiss}
                    className="update-btn update-btn-ghost"
                  >
                    {hideLabel}
                  </button>
                  <button
                    disabled
                    className="update-btn update-btn-primary"
                    aria-live="polite"
                  >
                    <RefreshCw size={14} className="animate-spin" />
                    {sessionActionLabel}
                  </button>
                </>
              ) : (
                <>
                  {!failed && !upToDate ? (
                    <button
                      onClick={() => handleLater(24)}
                      className="update-btn update-btn-ghost"
                    >
                      {t("common.updateNotification.action.laterOneDay")}
                    </button>
                  ) : null}
                  <button
                    onClick={handleDismiss}
                    className="update-btn update-btn-icon"
                    title={closeLabel}
                    aria-label={closeLabel}
                  >
                    <X size={13} />
                  </button>
                  {!upToDate ? (
                    <button
                      onClick={handleDownload}
                      className="update-btn update-btn-primary"
                    >
                      {failed ? (
                        <RefreshCw size={14} />
                      ) : (
                        <Download size={14} />
                      )}
                      {sessionActionLabel}
                    </button>
                  ) : null}
                </>
              )}
            </div>
          </div>

          {hasInstallSession ? (
            <div className="update-session-row" aria-live="polite">
              <span className="update-session-text">
                {sessionProgressLabel}
              </span>
              {installSession?.stage === "downloading" ||
              installSession?.stage === "installing" ||
              installSession?.stage === "restarting" ? (
                <>
                  <div
                    className="update-progress"
                    role="progressbar"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={progressPercent}
                  >
                    <div
                      className="update-progress-bar"
                      style={{ width: `${progressPercent}%` }}
                    />
                  </div>
                  {installSession?.totalBytes ? (
                    <span className="update-progress-value">
                      {progressPercent}%
                    </span>
                  ) : null}
                </>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
