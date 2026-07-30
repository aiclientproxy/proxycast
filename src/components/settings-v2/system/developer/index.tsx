import {
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useTranslation } from "react-i18next";
import {
  AlertCircle,
  Bug,
  DatabaseZap,
  Code2,
  Globe,
  ScrollText,
  ShieldAlert,
  Sparkles,
  Trash2,
  type LucideIcon,
} from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { useComponentDebug } from "@/contexts/ComponentDebugContext";
import { getConfig, saveConfig, type Config } from "@/lib/api/appConfig";
import { summarizeAgentUiPerformanceMetrics } from "@/lib/agentUiPerformanceMetrics";
import { getLogs, getPersistedLogsTail } from "@/lib/api/logs";
import {
  getLogStorageDiagnostics,
  getServerDiagnostics,
  getWindowsStartupDiagnostics,
} from "@/lib/api/serverRuntime";
import {
  buildCrashDiagnosticPayload,
  clearCrashDiagnosticHistory,
  collectRuntimeSnapshotForDiagnostic,
  collectGeneralWorkbenchDocumentStateForDiagnostic,
  CLEAR_CRASH_DIAGNOSTIC_HISTORY_CONFIRM_TEXT,
  copyCrashDiagnosticJsonToClipboard,
  copyCrashDiagnosticToClipboard,
  exportCrashDiagnosticToJson,
  isClipboardPermissionDeniedError,
  normalizeCrashReportingConfig,
  openCrashDiagnosticDownloadDirectory,
} from "@/lib/crashDiagnostic";
import { cn } from "@/lib/utils";
import {
  isClawTraceEnabled,
  isWorkspaceHarnessEnabled,
  normalizeDeveloperConfig,
} from "@/lib/developerFeatures";
import {
  DANGER_BUTTON_CLASS_NAME,
  SECONDARY_BUTTON_CLASS_NAME,
} from "./shared";
import { ClawTraceSettingsPanel } from "./ClawTraceSettingsPanel";

const ClipboardPermissionGuideCard = lazy(() =>
  import("../shared/ClipboardPermissionGuideCard").then((module) => ({
    default: module.ClipboardPermissionGuideCard,
  })),
);
const WorkspaceRepairHistoryCard = lazy(() =>
  import("../shared/WorkspaceRepairHistoryCard").then((module) => ({
    default: module.WorkspaceRepairHistoryCard,
  })),
);
const ServiceSkillCatalogTools = lazy(() =>
  import("./ServiceSkillCatalogTools").then((module) => ({
    default: module.ServiceSkillCatalogTools,
  })),
);
const SiteAdapterCatalogTools = lazy(() =>
  import("./SiteAdapterCatalogTools").then((module) => ({
    default: module.SiteAdapterCatalogTools,
  })),
);

interface SurfacePanelProps {
  icon: LucideIcon;
  title: string;
  description?: string;
  aside?: ReactNode;
  children: ReactNode;
}

function SurfacePanel({
  icon: Icon,
  title,
  description,
  aside,
  children,
}: SurfacePanelProps) {
  return (
    <article className="rounded-[26px] border border-slate-200/80 bg-white p-5 shadow-sm shadow-slate-950/5">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
            <Icon className="h-4 w-4 text-sky-600" />
            {title}
          </div>
          {description ? (
            <p className="text-sm leading-6 text-slate-500">{description}</p>
          ) : null}
        </div>
        {aside ? (
          <div className="flex flex-wrap items-center gap-2">{aside}</div>
        ) : null}
      </div>

      <div className="mt-5">{children}</div>
    </article>
  );
}

function StatusPill({
  active,
  activeLabel,
  inactiveLabel,
}: {
  active: boolean;
  activeLabel: string;
  inactiveLabel: string;
}) {
  return (
    <span
      className={cn(
        "rounded-full border px-3 py-1 text-xs font-medium",
        active
          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
          : "border-slate-200 bg-slate-100 text-slate-500",
      )}
    >
      {active ? activeLabel : inactiveLabel}
    </span>
  );
}

function DeferredPanelFallback({ label }: { label: string }) {
  const { t } = useTranslation("settings");

  return (
    <div className="rounded-[20px] border border-dashed border-slate-300 bg-slate-50 p-4 text-sm leading-6 text-slate-500">
      {t("settings.developer.deferred.loading", {
        label,
      })}
    </div>
  );
}

function CompactSwitchRow({
  title,
  description,
  checked,
  disabled,
  ariaLabel,
  onCheckedChange,
}: {
  title: string;
  description: string;
  checked: boolean;
  disabled?: boolean;
  ariaLabel: string;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <div className="flex flex-col gap-4 rounded-[22px] border border-slate-200/80 bg-slate-50 p-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="space-y-1">
        <p className="text-sm font-semibold text-slate-900">{title}</p>
        <p className="text-sm leading-6 text-slate-500">{description}</p>
      </div>
      <Switch
        aria-label={ariaLabel}
        checked={checked}
        disabled={disabled}
        onCheckedChange={(value) => onCheckedChange(Boolean(value))}
      />
    </div>
  );
}

function AdvancedDetails({
  icon: Icon,
  title,
  description,
  children,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
  children: ReactNode;
}) {
  const { t } = useTranslation("settings");

  return (
    <details className="group rounded-[26px] border border-slate-200/80 bg-white p-5 shadow-sm shadow-slate-950/5">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 [&::-webkit-details-marker]:hidden">
        <span className="flex min-w-0 items-center gap-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-2xl border border-sky-100 bg-sky-50 text-sky-700">
            <Icon className="h-4 w-4" />
          </span>
          <span className="min-w-0">
            <span className="block text-sm font-semibold text-slate-900">
              {title}
            </span>
            <span className="block truncate text-sm text-slate-500">
              {description}
            </span>
          </span>
        </span>
        <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-500 transition group-open:bg-slate-950 group-open:text-white">
          {t("settings.developer.details.expand")}
        </span>
      </summary>
      <div className="mt-4 space-y-4">{children}</div>
    </details>
  );
}

interface DeveloperSettingsProps {
  embedded?: boolean;
}

type DeveloperMessage = {
  type: "success" | "error";
  text: string;
};

export function DeveloperSettings({
  embedded = false,
}: DeveloperSettingsProps = {}) {
  const { t } = useTranslation("settings");
  const { enabled, setEnabled } = useComponentDebug();
  const [appConfig, setAppConfig] = useState<Config | null>(null);
  const [diagnosticBusy, setDiagnosticBusy] = useState(false);
  const [workspaceHarnessSaving, setWorkspaceHarnessSaving] = useState(false);
  const [message, setMessage] = useState<DeveloperMessage | null>(null);
  const [showClipboardGuide, setShowClipboardGuide] = useState(false);
  const messageTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearMessage = useCallback(() => {
    if (messageTimerRef.current !== null) {
      clearTimeout(messageTimerRef.current);
      messageTimerRef.current = null;
    }
    setMessage(null);
  }, []);

  const showTransientMessage = useCallback((nextMessage: DeveloperMessage) => {
    if (messageTimerRef.current !== null) {
      clearTimeout(messageTimerRef.current);
    }
    setMessage(nextMessage);
    messageTimerRef.current = setTimeout(() => {
      setMessage(null);
      messageTimerRef.current = null;
    }, 2500);
  }, []);

  useEffect(
    () => () => {
      if (messageTimerRef.current !== null) {
        clearTimeout(messageTimerRef.current);
      }
    },
    [],
  );

  const loadAppConfig = useCallback(async () => {
    try {
      const config = await getConfig();
      setAppConfig(config);
      return config;
    } catch (error) {
      console.error("加载开发者配置失败:", error);
      setAppConfig(null);
      return null;
    }
  }, []);

  useEffect(() => {
    void loadAppConfig();
  }, [loadAppConfig]);

  const buildDiagnosticPayload = useCallback(async () => {
    const configPromise = getConfig();
    const runtimeSnapshotPromise = configPromise.then((config) =>
      collectRuntimeSnapshotForDiagnostic(config),
    );
    const [
      config,
      logs,
      persistedLogs,
      generalWorkbenchDocumentState,
      serverDiagnostics,
      logStorageDiagnostics,
      windowsStartupDiagnostics,
      runtimeSnapshotResult,
    ] = await Promise.all([
      configPromise,
      getLogs(),
      getPersistedLogsTail(200),
      collectGeneralWorkbenchDocumentStateForDiagnostic(),
      getServerDiagnostics().catch(() => null),
      getLogStorageDiagnostics().catch(() => null),
      getWindowsStartupDiagnostics().catch(() => null),
      runtimeSnapshotPromise,
    ]);
    return buildCrashDiagnosticPayload({
      crashConfig: normalizeCrashReportingConfig(config.crash_reporting),
      logs,
      persistedLogTail: persistedLogs,
      agentUiPerformanceSnapshot: summarizeAgentUiPerformanceMetrics(),
      collectionNotes: runtimeSnapshotResult.collectionNotes,
      generalWorkbenchDocumentState,
      serverDiagnostics,
      logStorageDiagnostics,
      windowsStartupDiagnostics,
      runtimeSnapshot: runtimeSnapshotResult.runtimeSnapshot,
      appVersion: import.meta.env.VITE_APP_VERSION,
      platform: navigator.platform,
      userAgent: navigator.userAgent,
    });
  }, []);

  const handleCopyDiagnostic = useCallback(async () => {
    setDiagnosticBusy(true);
    clearMessage();
    setShowClipboardGuide(false);
    try {
      const payload = await buildDiagnosticPayload();
      await copyCrashDiagnosticToClipboard(payload);
      showTransientMessage({
        type: "success",
        text: t("settings.developer.message.diagnosticCopied"),
      });
    } catch (error) {
      console.error("复制诊断信息失败:", error);
      const isPermissionDenied = isClipboardPermissionDeniedError(error);
      setShowClipboardGuide(isPermissionDenied);
      setMessage({
        type: "error",
        text:
          error instanceof Error
            ? error.message
            : t("settings.developer.message.copyDiagnosticFailed"),
      });
    } finally {
      setDiagnosticBusy(false);
    }
  }, [buildDiagnosticPayload, clearMessage, showTransientMessage, t]);

  const handleCopyDiagnosticJson = useCallback(async () => {
    setDiagnosticBusy(true);
    clearMessage();
    setShowClipboardGuide(false);
    try {
      const payload = await buildDiagnosticPayload();
      await copyCrashDiagnosticJsonToClipboard(payload);
      showTransientMessage({
        type: "success",
        text: t("settings.developer.message.diagnosticJsonCopied"),
      });
    } catch (error) {
      console.error("复制纯 JSON 失败:", error);
      const isPermissionDenied = isClipboardPermissionDeniedError(error);
      setShowClipboardGuide(isPermissionDenied);
      setMessage({
        type: "error",
        text:
          error instanceof Error
            ? error.message
            : t("settings.developer.message.copyDiagnosticJsonFailed"),
      });
    } finally {
      setDiagnosticBusy(false);
    }
  }, [buildDiagnosticPayload, clearMessage, showTransientMessage, t]);

  const handleExportDiagnostic = useCallback(async () => {
    setDiagnosticBusy(true);
    clearMessage();
    setShowClipboardGuide(false);
    try {
      const payload = await buildDiagnosticPayload();
      const result = exportCrashDiagnosticToJson(payload, {
        sceneTag: "settings-developer",
      });
      try {
        await openCrashDiagnosticDownloadDirectory();
      } catch {
        // 导出已经成功，打开目录失败不覆盖导出反馈。
      }
      showTransientMessage({
        type: "success",
        text: t("settings.developer.message.diagnosticExported", {
          fileName: result.fileName,
          location: result.locationHint,
        }),
      });
    } catch (error) {
      console.error("导出诊断信息失败:", error);
      setMessage({
        type: "error",
        text:
          error instanceof Error
            ? error.message
            : t("settings.developer.message.exportDiagnosticFailed"),
      });
    } finally {
      setDiagnosticBusy(false);
    }
  }, [buildDiagnosticPayload, clearMessage, showTransientMessage, t]);

  const handleOpenDownloadDirectory = useCallback(async () => {
    setDiagnosticBusy(true);
    clearMessage();
    try {
      await openCrashDiagnosticDownloadDirectory();
    } catch (error) {
      console.error("打开下载目录失败:", error);
      setMessage({
        type: "error",
        text:
          error instanceof Error
            ? error.message
            : t("settings.developer.message.openDownloadDirectoryFailed"),
      });
    } finally {
      setDiagnosticBusy(false);
    }
  }, [clearMessage, t]);

  const handleClearDiagnosticHistory = useCallback(async () => {
    const confirmed =
      typeof window === "undefined" ||
      window.confirm(CLEAR_CRASH_DIAGNOSTIC_HISTORY_CONFIRM_TEXT);
    if (!confirmed) {
      return;
    }

    setDiagnosticBusy(true);
    clearMessage();
    setShowClipboardGuide(false);
    try {
      await clearCrashDiagnosticHistory();
      showTransientMessage({
        type: "success",
        text: t("settings.developer.message.diagnosticHistoryCleared"),
      });
    } catch (error) {
      console.error("清空旧诊断信息失败:", error);
      setMessage({
        type: "error",
        text:
          error instanceof Error
            ? error.message
            : t("settings.developer.message.clearDiagnosticHistoryFailed"),
      });
    } finally {
      setDiagnosticBusy(false);
    }
  }, [clearMessage, showTransientMessage, t]);

  const handleWorkspaceHarnessEnabledChange = useCallback(
    async (nextEnabled: boolean) => {
      setWorkspaceHarnessSaving(true);
      clearMessage();
      try {
        const latestConfig = appConfig ?? (await getConfig());
        const nextConfig: Config = {
          ...latestConfig,
          developer: {
            ...normalizeDeveloperConfig(latestConfig.developer),
            workspace_harness_enabled: nextEnabled,
          },
        };
        await saveConfig(nextConfig);
        setAppConfig(nextConfig);
        showTransientMessage({
          type: "success",
          text: nextEnabled
            ? t("settings.developer.message.workspaceHarnessEnabled")
            : t("settings.developer.message.workspaceHarnessDisabled"),
        });
      } catch (error) {
        console.error("保存处理工作台开关失败:", error);
        setMessage({
          type: "error",
          text:
            error instanceof Error
              ? error.message
              : t("settings.developer.message.saveWorkspaceHarnessFailed"),
        });
      } finally {
        setWorkspaceHarnessSaving(false);
      }
    },
    [appConfig, clearMessage, showTransientMessage, t],
  );

  const workspaceHarnessEnabled = isWorkspaceHarnessEnabled(appConfig);
  const clawTraceEnabled = isClawTraceEnabled(appConfig);

  return (
    <div className={cn("space-y-5", embedded ? "pb-0" : "pb-8")}>
      {message ? (
        <div
          className={cn(
            "flex items-center gap-2 rounded-[20px] border px-4 py-3 text-sm shadow-sm shadow-slate-950/5",
            message.type === "success"
              ? "border-emerald-200 bg-emerald-50 text-emerald-700"
              : "border-rose-200 bg-rose-50 text-rose-700",
          )}
        >
          <AlertCircle className="h-4 w-4" />
          <span>{message.text}</span>
        </div>
      ) : null}

      <section className="rounded-[26px] border border-slate-200/80 bg-white px-5 py-4 shadow-sm shadow-slate-950/5">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
          <div className="space-y-1.5">
            {embedded ? (
              <h2 className="text-[20px] font-semibold tracking-tight text-slate-900">
                {t("settings.developer.title.embedded")}
              </h2>
            ) : (
              <h1 className="text-[24px] font-semibold tracking-tight text-slate-900">
                {t("settings.developer.title")}
              </h1>
            )}
            <p className="text-sm text-slate-500">
              {t("settings.developer.description")}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 xl:justify-end">
            <StatusPill
              active={workspaceHarnessEnabled}
              activeLabel={t("settings.developer.status.workspaceHarness.on")}
              inactiveLabel={t(
                "settings.developer.status.workspaceHarness.off",
              )}
            />
            <StatusPill
              active={clawTraceEnabled}
              activeLabel={t("settings.developer.status.clawTrace.on")}
              inactiveLabel={t("settings.developer.status.clawTrace.off")}
            />
            <StatusPill
              active={enabled}
              activeLabel={t("settings.developer.status.componentDebug.on")}
              inactiveLabel={t("settings.developer.status.componentDebug.off")}
            />
            {diagnosticBusy ? (
              <span className="rounded-full border border-sky-200 bg-sky-50 px-3 py-1 text-xs font-medium text-sky-700">
                {t("settings.developer.status.diagnosticBusy")}
              </span>
            ) : null}
          </div>
        </div>
      </section>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,0.92fr)_minmax(360px,1.08fr)]">
        <div className="space-y-5">
          <SurfacePanel
            icon={Sparkles}
            title={t("settings.developer.debugSwitch.title")}
            description={t("settings.developer.debugSwitch.description")}
          >
            <div className="space-y-3">
              <CompactSwitchRow
                title={t(
                  "settings.developer.debugSwitch.workspaceHarness.title",
                )}
                description={
                  workspaceHarnessSaving
                    ? t("settings.developer.action.saving")
                    : t(
                        "settings.developer.debugSwitch.workspaceHarness.description",
                      )
                }
                checked={workspaceHarnessEnabled}
                disabled={workspaceHarnessSaving}
                ariaLabel={t(
                  "settings.developer.debugSwitch.workspaceHarness.aria",
                )}
                onCheckedChange={(checked) => {
                  void handleWorkspaceHarnessEnabledChange(checked);
                }}
              />
              <ClawTraceSettingsPanel
                appConfig={appConfig}
                onConfigSaved={setAppConfig}
                onMessage={showTransientMessage}
              />
              <CompactSwitchRow
                title={t("settings.developer.debugSwitch.component.title")}
                description={t(
                  "settings.developer.debugSwitch.component.description",
                )}
                checked={enabled}
                ariaLabel={t("settings.developer.debugSwitch.component.aria")}
                onCheckedChange={setEnabled}
              />
            </div>
          </SurfacePanel>

          <SurfacePanel
            icon={Bug}
            title={t("settings.developer.diagnostic.title")}
            description={t("settings.developer.diagnostic.description")}
          >
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => void handleClearDiagnosticHistory()}
                disabled={diagnosticBusy}
                className={DANGER_BUTTON_CLASS_NAME}
              >
                <Trash2 className="h-4 w-4" />
                {t("settings.developer.diagnostic.action.clearHistory")}
              </button>
              <button
                type="button"
                onClick={() => void handleCopyDiagnostic()}
                disabled={diagnosticBusy}
                className={SECONDARY_BUTTON_CLASS_NAME}
              >
                <Bug className="h-4 w-4" />
                {t("settings.developer.diagnostic.action.copy")}
              </button>
              <button
                type="button"
                onClick={() => void handleCopyDiagnosticJson()}
                disabled={diagnosticBusy}
                className={SECONDARY_BUTTON_CLASS_NAME}
              >
                <Code2 className="h-4 w-4" />
                {t("settings.developer.diagnostic.action.copyJson")}
              </button>
              <button
                type="button"
                onClick={() => void handleExportDiagnostic()}
                disabled={diagnosticBusy}
                className={SECONDARY_BUTTON_CLASS_NAME}
              >
                <ScrollText className="h-4 w-4" />
                {t("settings.developer.diagnostic.action.exportJson")}
              </button>
              <button
                type="button"
                onClick={() => void handleOpenDownloadDirectory()}
                disabled={diagnosticBusy}
                className={SECONDARY_BUTTON_CLASS_NAME}
              >
                <Sparkles className="h-4 w-4" />
                {t(
                  "settings.developer.diagnostic.action.openDownloadDirectory",
                )}
              </button>
            </div>

            {showClipboardGuide ? (
              <div className="mt-4">
                <Suspense
                  fallback={
                    <DeferredPanelFallback
                      label={t("settings.developer.deferred.clipboardGuide")}
                    />
                  }
                >
                  <ClipboardPermissionGuideCard />
                </Suspense>
              </div>
            ) : null}
          </SurfacePanel>
        </div>

        <div className="space-y-5">
          <AdvancedDetails
            icon={DatabaseZap}
            title={t("settings.developer.serviceSkill.title")}
            description={t("settings.developer.serviceSkill.description")}
          >
            <Suspense
              fallback={
                <DeferredPanelFallback
                  label={t("settings.developer.serviceSkill.title")}
                />
              }
            >
              <ServiceSkillCatalogTools />
            </Suspense>
          </AdvancedDetails>

          <AdvancedDetails
            icon={Globe}
            title={t("settings.developer.siteAdapter.title")}
            description={t("settings.developer.siteAdapter.description")}
          >
            <Suspense
              fallback={
                <DeferredPanelFallback
                  label={t("settings.developer.siteAdapter.title")}
                />
              }
            >
              <SiteAdapterCatalogTools />
            </Suspense>
          </AdvancedDetails>

          <AdvancedDetails
            icon={ShieldAlert}
            title={t("settings.developer.workspaceRepair.title")}
            description={t("settings.developer.workspaceRepair.description")}
          >
            <Suspense
              fallback={
                <DeferredPanelFallback
                  label={t("settings.developer.workspaceRepair.title")}
                />
              }
            >
              <WorkspaceRepairHistoryCard
                className="rounded-[22px] border-slate-200/80 bg-white p-4"
                description={t(
                  "settings.developer.workspaceRepair.cardDescription",
                )}
              />
            </Suspense>
          </AdvancedDetails>
        </div>
      </div>
    </div>
  );
}
