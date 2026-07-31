import { useEffect, useMemo, useState } from "react";
import { CheckCircle2, ExternalLink, RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  checkForUpdates,
  getUpdateInstallSession,
  isUpdateInstallSessionActive,
  listenUpdateInstallSession,
  startUpdateInstallSession,
  type UpdateInstallSession,
  type VersionInfo,
} from "@/lib/api/appUpdate";
import { LIME_BRAND_LOGO_SRC, LIME_BRAND_NAME } from "@/lib/branding";
import {
  interceptHttpExternalLinkClick,
  resolveHttpExternalHref,
} from "@/lib/markdown/externalLinks";
import { getRuntimeAppVersion } from "@/lib/appVersion";
import { cn } from "@/lib/utils";

const FALLBACK_RELEASES_URL = "https://github.com/limecloud/lime/releases";
const PRIMARY_ACTION_BUTTON_CLASS =
  "inline-flex items-center gap-2 rounded-full border border-slate-900 bg-slate-950 px-4 py-2 text-sm font-medium text-white shadow-sm shadow-slate-950/10 transition hover:bg-slate-800 disabled:opacity-50";
const SECONDARY_ACTION_BUTTON_CLASS =
  "inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-300 hover:bg-slate-50 disabled:opacity-50";

function normalizeAboutVersion(value?: string | null): string | null {
  const trimmed = value?.trim();
  if (!trimmed || trimmed.toLowerCase() === "unknown") {
    return null;
  }
  return trimmed;
}

export function AboutSection() {
  const { t } = useTranslation("settings");
  const runtimeAppVersion = useMemo(
    () => normalizeAboutVersion(getRuntimeAppVersion()),
    [],
  );
  const [versionInfo, setVersionInfo] = useState<VersionInfo>({
    current: runtimeAppVersion ?? "",
    latest: undefined,
    hasUpdate: false,
    downloadUrl: undefined,
    error: undefined,
  });
  const [checking, setChecking] = useState(false);
  const [installSession, setInstallSession] =
    useState<UpdateInstallSession | null>(null);
  const [installStartFailed, setInstallStartFailed] = useState(false);
  const manualDownloadUrl =
    versionInfo.releaseNotesUrl || FALLBACK_RELEASES_URL;
  const manualDownloadRel = resolveHttpExternalHref(manualDownloadUrl)
    ? "noreferrer noopener"
    : undefined;
  const isWindows =
    typeof navigator !== "undefined" && navigator.userAgent.includes("Windows");
  const installingUpdate = isUpdateInstallSessionActive(installSession);
  const rawInstallProgressPercent = installSession?.percent ?? 0;
  const installProgressPercent = Number.isFinite(rawInstallProgressPercent)
    ? Math.min(100, Math.max(0, Math.round(rawInstallProgressPercent * 100)))
    : 0;
  const hasInstallProgress =
    installSession !== null && installSession.stage !== "idle";
  const installFailed =
    installStartFailed || installSession?.stage === "failed";
  const confirmedSessionUpdate = Boolean(
    installSession?.stage === "completed" && installSession.latestVersion,
  );
  const updateAvailable = versionInfo.hasUpdate || confirmedSessionUpdate;
  const latestAvailableVersion = confirmedSessionUpdate
    ? installSession?.latestVersion
    : versionInfo.latest;
  const installStatusLabel = useMemo(() => {
    if (!installSession) {
      return "";
    }

    switch (installSession.stage) {
      case "checking":
        return t("settings.about.update.progress.checking");
      case "downloading":
        return installSession.totalBytes
          ? t("settings.about.update.progress.downloading", {
              percent: `${installProgressPercent}%`,
            })
          : t("settings.about.update.progress.downloadingUnknown");
      case "installing":
        return t("settings.about.update.progress.installing");
      case "restarting":
        return t("settings.about.update.progress.restarting");
      case "failed":
        return t("settings.about.update.progress.failed");
      case "up_to_date":
        return t("settings.about.update.progress.upToDate");
      case "completed":
        return t("settings.about.update.progress.completed");
      case "idle":
      default:
        return "";
    }
  }, [installProgressPercent, installSession, t]);
  const installActionLabel = useMemo(() => {
    if (!installSession) {
      return versionInfo.hasUpdate
        ? t("settings.about.action.restartToUpdate")
        : t("settings.about.action.download");
    }

    switch (installSession.stage) {
      case "checking":
        return t("settings.about.update.progress.checking");
      case "downloading":
        return t("settings.about.action.downloading");
      case "installing":
        return t("settings.about.action.installing");
      case "restarting":
        return t("settings.about.action.restarting");
      case "failed":
        return t("settings.about.action.retryDownload");
      case "completed":
        return t("settings.about.action.restartToUpdate");
      default:
        return t("settings.about.action.download");
    }
  }, [installSession, t, versionInfo.hasUpdate]);

  useEffect(() => {
    const loadCurrentVersion = async () => {
      try {
        const result = await checkForUpdates({ automatic: true });
        setVersionInfo({
          ...result,
          current:
            normalizeAboutVersion(result.current) ?? runtimeAppVersion ?? "",
          downloadUrl: result.downloadUrl || FALLBACK_RELEASES_URL,
        });
      } catch (error) {
        console.error("Failed to load version:", error);
        setVersionInfo((prev) => ({
          ...prev,
          current:
            normalizeAboutVersion(prev.current) ?? runtimeAppVersion ?? "",
          error: t("settings.about.update.errorCheck"),
          downloadUrl: prev.downloadUrl || FALLBACK_RELEASES_URL,
        }));
      }
    };

    void loadCurrentVersion();
  }, [runtimeAppVersion, t]);

  useEffect(() => {
    let disposed = false;
    void getUpdateInstallSession()
      .then((session) => {
        if (!disposed && session.stage !== "idle") {
          setInstallSession(session);
        }
      })
      .catch((error) => {
        console.error("Failed to read update install session:", error);
      });

    const unlistenPromise = listenUpdateInstallSession((session) => {
      setInstallSession(session);
      setInstallStartFailed(false);
    });

    return () => {
      disposed = true;
      void unlistenPromise
        .then((unlisten) => unlisten())
        .catch((error) => {
          console.error("Failed to remove update install listener:", error);
        });
    };
  }, []);

  const handleCheckUpdate = async () => {
    setChecking(true);
    setInstallStartFailed(false);
    try {
      const result = await checkForUpdates();
      setVersionInfo({
        ...result,
        current:
          normalizeAboutVersion(result.current) ?? runtimeAppVersion ?? "",
        downloadUrl: result.downloadUrl || FALLBACK_RELEASES_URL,
      });
    } catch (error) {
      console.error("Failed to check for updates:", error);
      setVersionInfo((prev) => ({
        ...prev,
        current: normalizeAboutVersion(prev.current) ?? runtimeAppVersion ?? "",
        error: t("settings.about.update.errorCheck"),
        downloadUrl: prev.downloadUrl || FALLBACK_RELEASES_URL,
      }));
    } finally {
      setChecking(false);
    }
  };

  const handleDownloadUpdate = async () => {
    setInstallStartFailed(false);
    try {
      const session = await startUpdateInstallSession();
      setInstallSession(session);
    } catch (error) {
      console.error("Failed to start update install session:", error);
      setInstallStartFailed(true);
    }
  };

  const currentVersion =
    normalizeAboutVersion(versionInfo.current) ??
    normalizeAboutVersion(installSession?.currentVersion) ??
    runtimeAppVersion;
  const versionLabel = currentVersion
    ? t("settings.about.version.label", {
        version: currentVersion,
        build: currentVersion,
      })
    : t("settings.about.version.loadingLabel", {
        loading: t("settings.about.version.loading"),
        defaultValue: "Version {{loading}}",
      });

  const updateStatus = useMemo(() => {
    if (installingUpdate) {
      return {
        label:
          installStatusLabel || t("settings.about.update.progress.checking"),
        className: "border-sky-200 bg-sky-50 text-sky-700",
      };
    }

    if (installFailed) {
      return {
        label: t("settings.about.update.progress.failed"),
        className: "border-rose-200 bg-rose-50 text-rose-700",
      };
    }

    if (updateAvailable) {
      return {
        label: t("settings.about.status.updateAvailable", {
          version: latestAvailableVersion ?? "",
        }),
        className: "border-emerald-200 bg-emerald-50 text-emerald-700",
      };
    }

    if (versionInfo.error) {
      return {
        label: t("settings.about.update.errorCheck"),
        className: "border-amber-200 bg-amber-50 text-amber-700",
      };
    }

    if (versionInfo.latest) {
      return {
        label: t("settings.about.status.latest"),
        className: "border-slate-200 bg-slate-100 text-slate-600",
      };
    }

    return {
      label: t("settings.about.status.manualCheck"),
      className: "border-sky-200 bg-sky-50 text-sky-700",
    };
  }, [
    installFailed,
    installStatusLabel,
    installingUpdate,
    latestAvailableVersion,
    t,
    updateAvailable,
    versionInfo.error,
    versionInfo.latest,
  ]);

  return (
    <div className="pb-8">
      <section className="mx-auto max-w-[560px] rounded-[28px] border border-slate-200/80 bg-white px-6 py-9 text-center shadow-sm shadow-slate-950/5">
        <div className="mx-auto flex h-24 w-24 items-center justify-center rounded-[24px] border border-slate-200 bg-slate-50 shadow-sm shadow-slate-950/5">
          <img
            src={LIME_BRAND_LOGO_SRC}
            alt={LIME_BRAND_NAME}
            className="h-16 w-16 object-contain"
          />
        </div>

        <h2 className="mt-6 text-[28px] font-semibold tracking-tight text-slate-950">
          {LIME_BRAND_NAME}
        </h2>
        <p className="mt-3 text-base text-slate-700">{versionLabel}</p>
        <p className="mt-2 text-sm text-slate-500">
          {t("settings.about.copyright", {
            brand: LIME_BRAND_NAME,
          })}
        </p>

        <div className="mt-5 flex justify-center">
          <span
            className={cn(
              "rounded-full border px-3 py-1.5 text-xs font-medium",
              updateStatus.className,
            )}
          >
            {updateStatus.label}
          </span>
        </div>

        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          <button
            type="button"
            onClick={() => void handleCheckUpdate()}
            disabled={checking || installingUpdate}
            className={SECONDARY_ACTION_BUTTON_CLASS}
          >
            <RefreshCw className={cn("h-4 w-4", checking && "animate-spin")} />
            {t("settings.about.action.check")}
          </button>

          {updateAvailable ? (
            <>
              <button
                type="button"
                onClick={() => void handleDownloadUpdate()}
                disabled={installingUpdate}
                className={PRIMARY_ACTION_BUTTON_CLASS}
              >
                <RefreshCw
                  className={cn("h-4 w-4", installingUpdate && "animate-spin")}
                />
                {installActionLabel}
              </button>
              <a
                href={manualDownloadUrl}
                rel={manualDownloadRel}
                onAuxClick={(event) => {
                  interceptHttpExternalLinkClick(event, manualDownloadUrl);
                }}
                onClick={(event) => {
                  interceptHttpExternalLinkClick(event, manualDownloadUrl);
                }}
                className={SECONDARY_ACTION_BUTTON_CLASS}
              >
                <ExternalLink className="h-4 w-4" />
                {t("settings.about.action.webDownload")}
              </a>
            </>
          ) : null}
        </div>

        {isWindows ? (
          <p className="mx-auto mt-4 max-w-[420px] text-xs leading-5 text-slate-500">
            {t("settings.about.windowsSetupNotice")}
          </p>
        ) : null}

        {hasInstallProgress || installFailed ? (
          <div
            className={cn(
              "mt-5 rounded-[20px] border p-4 text-left text-sm shadow-sm shadow-slate-950/5",
              installFailed
                ? "border-rose-200 bg-rose-50 text-rose-700"
                : installSession?.stage === "restarting" ||
                    installSession?.stage === "completed" ||
                    installSession?.stage === "up_to_date"
                  ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                  : "border-sky-200 bg-sky-50 text-sky-700",
            )}
          >
            <div className="flex items-start gap-2">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
              <div className="min-w-0 flex-1">
                <p>
                  {installFailed
                    ? t("settings.about.update.errorDownload")
                    : installStatusLabel ||
                      t("settings.about.update.progress.checking")}
                </p>
                {!installFailed &&
                (installSession?.stage === "downloading" ||
                  installSession?.stage === "installing" ||
                  installSession?.stage === "restarting") ? (
                  <div className="mt-3 flex items-center gap-3">
                    <div
                      className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/70"
                      role="progressbar"
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={installProgressPercent}
                    >
                      <div
                        className="h-full rounded-full bg-slate-950 transition-[width]"
                        style={{ width: `${installProgressPercent}%` }}
                      />
                    </div>
                    {installSession?.totalBytes ? (
                      <span className="w-10 text-right text-xs tabular-nums">
                        {installProgressPercent}%
                      </span>
                    ) : null}
                  </div>
                ) : null}
                {installFailed ? (
                  <a
                    href={manualDownloadUrl}
                    rel={manualDownloadRel}
                    onAuxClick={(event) => {
                      interceptHttpExternalLinkClick(event, manualDownloadUrl);
                    }}
                    onClick={(event) => {
                      interceptHttpExternalLinkClick(event, manualDownloadUrl);
                    }}
                    className="mt-2 inline-flex items-center gap-1 underline hover:no-underline"
                  >
                    <ExternalLink className="h-3 w-3" />
                    {t("settings.about.action.openWebDownload")}
                  </a>
                ) : null}
              </div>
            </div>
          </div>
        ) : null}
      </section>
    </div>
  );
}
