/**
 * @file ExperimentalSettings.tsx
 * @description 实验室设置页面 - 管理实验性功能的开关和配置
 * @module components/settings-v2/system/experimental
 *
 * 需求: 6.1, 6.2, 6.3, 6.5 - 实验室标签页，截图对话功能开关，快捷键设置，权限警告
 */

import {
  Suspense,
  lazy,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";
import {
  Globe,
  AlertCircle,
  AlertTriangle,
  RefreshCw,
  Bug,
  Wrench,
  ShieldAlert,
  Sparkles,
  FolderOpen,
  type LucideIcon,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";
import {
  getConfig,
  saveConfig,
  type Config,
  type CrashReportingConfig,
  type ToolCallingConfig,
} from "@/lib/api/appConfig";
import {
  DEFAULT_EXPERIMENTAL_FEATURES,
  getExperimentalConfig,
  saveExperimentalConfig,
  type ExperimentalFeatures,
} from "@/lib/api/experimentalFeatures";
import { getLogs, getPersistedLogsTail } from "@/lib/api/logs";
import {
  getLogStorageDiagnostics,
  getServerDiagnostics,
  getWindowsStartupDiagnostics,
} from "@/lib/api/serverRuntime";
import { getDesktopHostDiagnostics } from "@/lib/api/desktopHostDiagnostics";
import { applyCrashReportingSettings } from "@/lib/crashReporting";
import {
  buildCrashDiagnosticPayload,
  collectRuntimeSnapshotForDiagnostic,
  collectGeneralWorkbenchDocumentStateForDiagnostic,
  copyCrashDiagnosticJsonToClipboard,
  copyCrashDiagnosticToClipboard,
  DEFAULT_CRASH_REPORTING_CONFIG,
  exportCrashDiagnosticToJson,
  isClipboardPermissionDeniedError,
  normalizeCrashReportingConfig,
  openCrashDiagnosticDownloadDirectory,
} from "@/lib/crashDiagnostic";
import { Switch } from "@/components/ui/switch";
import {
  DEFAULT_TOOL_CALLING_CONFIG,
  normalizeToolCallingConfig,
} from "./tool-calling-config";

const UpdateCheckSettings = lazy(() =>
  import("./UpdateCheckSettings").then((module) => ({
    default: module.UpdateCheckSettings,
  })),
);
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

// ============================================================
// 组件
// ============================================================

interface SurfacePanelProps {
  icon: LucideIcon;
  title: string;
  description?: string;
  children: ReactNode;
  aside?: ReactNode;
}

function SurfacePanel({
  icon: Icon,
  title,
  description,
  children,
  aside,
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
          {t("settings.experimental.details.expand")}
        </span>
      </summary>
      <div className="mt-4 space-y-4">{children}</div>
    </details>
  );
}

const SECONDARY_BUTTON_CLASS_NAME =
  "inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-600 transition hover:border-slate-300 hover:text-slate-900 disabled:cursor-not-allowed disabled:opacity-50";
const PRIMARY_BUTTON_CLASS_NAME =
  "inline-flex items-center gap-2 rounded-full border border-slate-950 bg-slate-950 px-4 py-2 text-sm font-medium text-white shadow-sm shadow-slate-950/15 transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50";
const FIELD_CLASS_NAME =
  "w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm shadow-slate-950/5 outline-none focus:border-slate-300 focus:ring-2 focus:ring-slate-200";

function DeferredPanelFallback({ label }: { label: string }) {
  const { t } = useTranslation("settings");

  return (
    <div className="rounded-[20px] border border-dashed border-slate-300 bg-slate-50 p-4 text-sm leading-6 text-slate-500">
      {t("settings.experimental.deferred.loading", {
        label,
      })}
    </div>
  );
}

interface ExperimentalSettingsProps {
  embedded?: boolean;
}

export function ExperimentalSettings({
  embedded = false,
}: ExperimentalSettingsProps = {}) {
  const { t } = useTranslation("settings");
  // 状态
  const [config, setConfig] = useState<ExperimentalFeatures | null>(null);
  const [toolCallingConfig, setToolCallingConfig] = useState<ToolCallingConfig>(
    DEFAULT_TOOL_CALLING_CONFIG,
  );
  const [crashConfig, setCrashConfig] = useState<CrashReportingConfig>(
    DEFAULT_CRASH_REPORTING_CONFIG,
  );
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [diagnosticBusy, setDiagnosticBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);
  const [showClipboardGuide, setShowClipboardGuide] = useState(false);

  // 加载配置
  const loadConfig = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [experimentalConfig, fullConfig] = await Promise.all([
        getExperimentalConfig(),
        getConfig(),
      ]);
      setConfig(experimentalConfig);
      setToolCallingConfig(normalizeToolCallingConfig(fullConfig.tool_calling));
      setCrashConfig(normalizeCrashReportingConfig(fullConfig.crash_reporting));
    } catch (err) {
      console.error("加载实验室配置失败:", err);
      setError(
        err instanceof Error
          ? err.message
          : t("settings.experimental.message.loadFailed"),
      );
      setConfig(DEFAULT_EXPERIMENTAL_FEATURES);
      setCrashConfig(DEFAULT_CRASH_REPORTING_CONFIG);
      setToolCallingConfig(DEFAULT_TOOL_CALLING_CONFIG);
    } finally {
      setLoading(false);
    }
  }, [t]);

  // 初始加载
  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  const handleToggleWebMcp = useCallback(async () => {
    if (!config) return;

    const newEnabled = !(config.webmcp?.enabled ?? false);
    const newConfig: ExperimentalFeatures = {
      ...config,
      webmcp: {
        enabled: newEnabled,
      },
    };

    setSaving(true);
    setMessage(null);

    try {
      await saveExperimentalConfig(newConfig);
      setConfig(newConfig);
      setMessage({
        type: "success",
        text: newEnabled
          ? t("settings.experimental.message.webMcpEnabled")
          : t("settings.experimental.message.webMcpDisabled"),
      });
      setTimeout(() => setMessage(null), 2000);
    } catch (err) {
      console.error("保存 WebMCP 配置失败:", err);
      setMessage({
        type: "error",
        text:
          err instanceof Error
            ? err.message
            : t("settings.experimental.message.saveFailed"),
      });
    } finally {
      setSaving(false);
    }
  }, [config, t]);

  const persistToolCallingConfig = useCallback(
    async (next: ToolCallingConfig, successText: string) => {
      setSaving(true);
      setMessage(null);
      try {
        const latestConfig = await getConfig();
        const updatedConfig: Config = {
          ...latestConfig,
          tool_calling: next,
        };
        await saveConfig(updatedConfig);
        setToolCallingConfig(next);
        setMessage({ type: "success", text: successText });
        setTimeout(() => setMessage(null), 2000);
      } catch (err) {
        console.error("保存 Tool Calling 配置失败:", err);
        setMessage({
          type: "error",
          text:
            err instanceof Error
              ? err.message
              : t("settings.experimental.message.toolCallingSaveFailed"),
        });
      } finally {
        setSaving(false);
      }
    },
    [t],
  );

  const handleToggleToolCallingEnabled = useCallback(() => {
    const next = {
      ...toolCallingConfig,
      enabled: !toolCallingConfig.enabled,
    };
    void persistToolCallingConfig(
      next,
      next.enabled
        ? t("settings.experimental.message.toolCallingEnabled")
        : t("settings.experimental.message.toolCallingDisabled"),
    );
  }, [persistToolCallingConfig, toolCallingConfig, t]);

  const handleToggleDynamicFiltering = useCallback(() => {
    const next = {
      ...toolCallingConfig,
      dynamic_filtering: !toolCallingConfig.dynamic_filtering,
    };
    void persistToolCallingConfig(
      next,
      next.dynamic_filtering
        ? t("settings.experimental.message.dynamicFilteringEnabled")
        : t("settings.experimental.message.dynamicFilteringDisabled"),
    );
  }, [persistToolCallingConfig, toolCallingConfig, t]);

  const handleToggleNativeInputExamples = useCallback(() => {
    const next = {
      ...toolCallingConfig,
      native_input_examples: !toolCallingConfig.native_input_examples,
    };
    void persistToolCallingConfig(
      next,
      next.native_input_examples
        ? t("settings.experimental.message.nativeInputExamplesEnabled")
        : t("settings.experimental.message.nativeInputExamplesDisabled"),
    );
  }, [persistToolCallingConfig, toolCallingConfig, t]);

  const persistCrashConfig = useCallback(
    async (next: CrashReportingConfig) => {
      setSaving(true);
      setMessage(null);
      try {
        const latestConfig = await getConfig();
        const normalized = normalizeCrashReportingConfig(next);
        const updatedConfig: Config = {
          ...latestConfig,
          crash_reporting: normalized,
        };
        await saveConfig(updatedConfig);
        await applyCrashReportingSettings(normalized);
        setCrashConfig(normalized);
        setMessage({
          type: "success",
          text: t("settings.experimental.message.crashReportingSaved"),
        });
        setTimeout(() => setMessage(null), 2000);
      } catch (err) {
        console.error("保存崩溃上报配置失败:", err);
        setMessage({
          type: "error",
          text:
            err instanceof Error
              ? err.message
              : t("settings.experimental.message.crashReportingSaveFailed"),
        });
      } finally {
        setSaving(false);
      }
    },
    [t],
  );

  const handleCrashEnabledToggle = useCallback(() => {
    const nextConfig = {
      ...crashConfig,
      enabled: !crashConfig.enabled,
    };
    void persistCrashConfig(nextConfig);
  }, [crashConfig, persistCrashConfig]);

  const handleCrashFieldChange = useCallback(
    (
      field: keyof CrashReportingConfig,
      value: string | boolean | number | null,
    ) => {
      setCrashConfig((previous) => ({
        ...previous,
        [field]: value,
      }));
    },
    [],
  );

  const handleSaveCrashConfig = useCallback(() => {
    void persistCrashConfig(crashConfig);
  }, [crashConfig, persistCrashConfig]);

  const buildDiagnosticPayload = useCallback(async () => {
    const [
      logs,
      persistedLogs,
      generalWorkbenchDocumentState,
      serverDiagnostics,
      logStorageDiagnostics,
      windowsStartupDiagnostics,
      desktopHostDiagnostics,
      runtimeSnapshotResult,
    ] = await Promise.all([
      getLogs(),
      getPersistedLogsTail(200),
      collectGeneralWorkbenchDocumentStateForDiagnostic(),
      getServerDiagnostics().catch(() => null),
      getLogStorageDiagnostics().catch(() => null),
      getWindowsStartupDiagnostics().catch(() => null),
      getDesktopHostDiagnostics().catch(() => null),
      collectRuntimeSnapshotForDiagnostic(),
    ]);
    return buildCrashDiagnosticPayload({
      crashConfig,
      logs,
      persistedLogTail: persistedLogs,
      collectionNotes: runtimeSnapshotResult.collectionNotes,
      generalWorkbenchDocumentState,
      serverDiagnostics,
      logStorageDiagnostics,
      windowsStartupDiagnostics,
      desktopHostDiagnostics,
      runtimeSnapshot: runtimeSnapshotResult.runtimeSnapshot,
      appVersion: import.meta.env.VITE_APP_VERSION,
      platform: navigator.platform,
      userAgent: navigator.userAgent,
    });
  }, [crashConfig]);

  const copyCrashDiagnostic = useCallback(async () => {
    setDiagnosticBusy(true);
    setMessage(null);
    setShowClipboardGuide(false);
    try {
      const payload = await buildDiagnosticPayload();
      await copyCrashDiagnosticToClipboard(payload);
      setMessage({
        type: "success",
        text: t("settings.experimental.message.diagnosticCopied"),
      });
      setTimeout(() => setMessage(null), 2500);
    } catch (err) {
      console.error("复制诊断信息失败:", err);
      const isPermissionDenied = isClipboardPermissionDeniedError(err);
      setShowClipboardGuide(isPermissionDenied);
      setMessage({
        type: "error",
        text:
          err instanceof Error
            ? err.message
            : t("settings.experimental.message.diagnosticCopyFailed"),
      });
    } finally {
      setDiagnosticBusy(false);
    }
  }, [buildDiagnosticPayload, t]);

  const copyCrashDiagnosticJson = useCallback(async () => {
    setDiagnosticBusy(true);
    setMessage(null);
    setShowClipboardGuide(false);
    try {
      const payload = await buildDiagnosticPayload();
      await copyCrashDiagnosticJsonToClipboard(payload);
      setMessage({
        type: "success",
        text: t("settings.experimental.message.diagnosticJsonCopied"),
      });
      setTimeout(() => setMessage(null), 2500);
    } catch (err) {
      console.error("复制纯 JSON 失败:", err);
      const isPermissionDenied = isClipboardPermissionDeniedError(err);
      setShowClipboardGuide(isPermissionDenied);
      setMessage({
        type: "error",
        text:
          err instanceof Error
            ? err.message
            : t("settings.experimental.message.diagnosticJsonCopyFailed"),
      });
    } finally {
      setDiagnosticBusy(false);
    }
  }, [buildDiagnosticPayload, t]);

  const exportCrashDiagnostic = useCallback(async () => {
    setDiagnosticBusy(true);
    setMessage(null);
    setShowClipboardGuide(false);
    try {
      const payload = await buildDiagnosticPayload();
      const result = exportCrashDiagnosticToJson(payload, {
        sceneTag: "settings-experimental",
      });
      try {
        await openCrashDiagnosticDownloadDirectory();
      } catch {
        // 导出已经成功，打开目录失败不覆盖导出反馈。
      }
      setMessage({
        type: "success",
        text: t("settings.experimental.message.diagnosticExported", {
          fileName: result.fileName,
          location: result.locationHint,
        }),
      });
      setTimeout(() => setMessage(null), 2500);
    } catch (err) {
      console.error("导出诊断信息失败:", err);
      setMessage({
        type: "error",
        text:
          err instanceof Error
            ? err.message
            : t("settings.experimental.message.diagnosticExportFailed"),
      });
    } finally {
      setDiagnosticBusy(false);
    }
  }, [buildDiagnosticPayload, t]);

  const openCrashDownloadDirectory = useCallback(async () => {
    setDiagnosticBusy(true);
    setMessage(null);
    setShowClipboardGuide(false);
    try {
      await openCrashDiagnosticDownloadDirectory();
    } catch (err) {
      console.error("打开下载目录失败:", err);
      setMessage({
        type: "error",
        text:
          err instanceof Error
            ? err.message
            : t("settings.experimental.message.downloadDirectoryOpenFailed"),
      });
    } finally {
      setDiagnosticBusy(false);
    }
  }, [t]);

  if (loading) {
    return (
      <div className={cn("space-y-6", embedded ? "pb-0" : "pb-8")}>
        <div className="h-[228px] animate-pulse rounded-[30px] border border-slate-200/80 bg-[linear-gradient(135deg,rgba(244,251,248,0.98)_0%,rgba(248,250,252,0.98)_45%,rgba(241,246,255,0.96)_100%)]" />
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.18fr)_minmax(320px,0.82fr)]">
          <div className="h-[320px] animate-pulse rounded-[26px] border border-slate-200/80 bg-white" />
          <div className="h-[320px] animate-pulse rounded-[26px] border border-slate-200/80 bg-white" />
        </div>
      </div>
    );
  }

  if (error && !config) {
    return (
      <div className="rounded-[26px] border border-rose-200 bg-rose-50/80 p-5 shadow-sm shadow-slate-950/5">
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 text-rose-600" />
          <div>
            <p className="text-sm font-semibold text-rose-700">
              {t("settings.experimental.error.loadTitle")}
            </p>
            <p className="mt-1 text-sm leading-6 text-rose-600">{error}</p>
            <button
              type="button"
              onClick={() => void loadConfig()}
              className="mt-3 inline-flex items-center gap-2 rounded-full border border-rose-200 bg-white px-4 py-2 text-sm font-medium text-rose-700 transition hover:bg-rose-100/70"
            >
              <RefreshCw className="h-4 w-4" />
              {t("settings.experimental.action.retry")}
            </button>
          </div>
        </div>
      </div>
    );
  }

  const busyLabel = saving
    ? t("settings.experimental.status.saving")
    : diagnosticBusy
      ? t("settings.experimental.status.diagnosticBusy")
      : null;

  return (
    <div className={cn("space-y-5", embedded ? "pb-0" : "pb-8")}>
      {message && (
        <div
          className={cn(
            "flex items-center gap-2 rounded-[20px] border px-4 py-3 text-sm shadow-sm shadow-slate-950/5",
            message.type === "success"
              ? "border-emerald-200 bg-emerald-50/90 text-emerald-700"
              : "border-rose-200 bg-rose-50/90 text-rose-700",
          )}
        >
          <AlertCircle className="h-4 w-4" />
          {message.text}
        </div>
      )}

      <section className="rounded-[26px] border border-slate-200/80 bg-white px-5 py-4 shadow-sm shadow-slate-950/5">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
          <div className="space-y-1.5">
            {embedded ? (
              <h2 className="text-[20px] font-semibold tracking-tight text-slate-900">
                {t("settings.experimental.title")}
              </h2>
            ) : (
              <h1 className="text-[24px] font-semibold tracking-tight text-slate-900">
                {t("settings.experimental.title")}
              </h1>
            )}
            <p className="text-sm text-slate-500">
              {t("settings.experimental.description")}
            </p>
          </div>

          {busyLabel ? (
            <div className="flex flex-wrap items-center gap-2 xl:justify-end">
              <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-500">
                {busyLabel}
              </span>
            </div>
          ) : null}
        </div>
      </section>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,0.96fr)_minmax(360px,1.04fr)]">
        <div className="space-y-5">
          <SurfacePanel
            icon={Wrench}
            title={t("settings.experimental.toolCalling.title")}
            description={t("settings.experimental.toolCalling.description")}
          >
            <div className="space-y-3">
              <CompactSwitchRow
                title={t("settings.experimental.toolCalling.enabled.title")}
                description={t(
                  "settings.experimental.toolCalling.enabled.description",
                )}
                checked={toolCallingConfig.enabled}
                onCheckedChange={handleToggleToolCallingEnabled}
                disabled={saving}
                ariaLabel={t("settings.experimental.toolCalling.enabled.aria")}
              />

              <div className="grid gap-3 md:grid-cols-2">
                <CompactSwitchRow
                  title={t(
                    "settings.experimental.toolCalling.dynamicFiltering.title",
                  )}
                  description={t(
                    "settings.experimental.toolCalling.dynamicFiltering.description",
                  )}
                  checked={toolCallingConfig.dynamic_filtering}
                  onCheckedChange={handleToggleDynamicFiltering}
                  disabled={saving || !toolCallingConfig.enabled}
                  ariaLabel={t(
                    "settings.experimental.toolCalling.dynamicFiltering.aria",
                  )}
                />

                <CompactSwitchRow
                  title={t(
                    "settings.experimental.toolCalling.nativeInputExamples.title",
                  )}
                  description={t(
                    "settings.experimental.toolCalling.nativeInputExamples.description",
                  )}
                  checked={toolCallingConfig.native_input_examples}
                  onCheckedChange={handleToggleNativeInputExamples}
                  disabled={saving || !toolCallingConfig.enabled}
                  ariaLabel={t(
                    "settings.experimental.toolCalling.nativeInputExamples.aria",
                  )}
                />
              </div>
            </div>
          </SurfacePanel>
        </div>

        <div className="space-y-5">
          <AdvancedDetails
            icon={Globe}
            title={t("settings.experimental.webMcp.title")}
            description={t("settings.experimental.webMcp.description")}
          >
            <CompactSwitchRow
              title={t("settings.experimental.webMcp.enabled.title")}
              description={t(
                "settings.experimental.webMcp.enabled.description",
              )}
              checked={config?.webmcp?.enabled ?? false}
              onCheckedChange={handleToggleWebMcp}
              disabled={saving}
              ariaLabel={t("settings.experimental.webMcp.enabled.aria")}
            />
          </AdvancedDetails>

          <AdvancedDetails
            icon={Bug}
            title={t("settings.experimental.crashReporting.title")}
            description={t("settings.experimental.crashReporting.description")}
          >
            <div className="space-y-3">
              <CompactSwitchRow
                title={t("settings.experimental.crashReporting.enabled.title")}
                description={t(
                  "settings.experimental.crashReporting.enabled.description",
                )}
                checked={Boolean(crashConfig.enabled)}
                onCheckedChange={handleCrashEnabledToggle}
                disabled={saving}
                ariaLabel={t(
                  "settings.experimental.crashReporting.enabled.aria",
                )}
              />

              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2 md:col-span-2">
                  <label className="text-sm font-medium text-slate-700">
                    {t("settings.experimental.crashReporting.dsn.label")}
                  </label>
                  <input
                    value={crashConfig.dsn ?? ""}
                    onChange={(event) =>
                      handleCrashFieldChange("dsn", event.target.value || null)
                    }
                    disabled={saving}
                    placeholder={t(
                      "settings.experimental.crashReporting.dsn.placeholder",
                    )}
                    className={FIELD_CLASS_NAME}
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-700">
                    {t(
                      "settings.experimental.crashReporting.environment.label",
                    )}
                  </label>
                  <input
                    value={crashConfig.environment ?? "production"}
                    onChange={(event) =>
                      handleCrashFieldChange("environment", event.target.value)
                    }
                    disabled={saving}
                    className={FIELD_CLASS_NAME}
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-700">
                    {t("settings.experimental.crashReporting.sampleRate.label")}
                  </label>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.1}
                    value={Number(crashConfig.sample_rate ?? 1)}
                    onChange={(event) =>
                      handleCrashFieldChange(
                        "sample_rate",
                        Number(event.target.value || 1),
                      )
                    }
                    disabled={saving}
                    className={FIELD_CLASS_NAME}
                  />
                </div>
              </div>

              <CompactSwitchRow
                title={t("settings.experimental.crashReporting.sendPii.title")}
                description={t(
                  "settings.experimental.crashReporting.sendPii.description",
                )}
                checked={Boolean(crashConfig.send_pii)}
                onCheckedChange={(checked) =>
                  handleCrashFieldChange("send_pii", checked)
                }
                disabled={saving}
                ariaLabel={t(
                  "settings.experimental.crashReporting.sendPii.aria",
                )}
              />

              <div className="flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={() => void copyCrashDiagnostic()}
                  disabled={saving || diagnosticBusy}
                  className={SECONDARY_BUTTON_CLASS_NAME}
                >
                  <Bug className="h-4 w-4" />
                  {t(
                    "settings.experimental.crashReporting.action.copyDiagnostic",
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => void copyCrashDiagnosticJson()}
                  disabled={saving || diagnosticBusy}
                  className={SECONDARY_BUTTON_CLASS_NAME}
                >
                  <Sparkles className="h-4 w-4" />
                  {t("settings.experimental.crashReporting.action.copyJson")}
                </button>
                <button
                  type="button"
                  onClick={() => void exportCrashDiagnostic()}
                  disabled={saving || diagnosticBusy}
                  className={SECONDARY_BUTTON_CLASS_NAME}
                >
                  <FolderOpen className="h-4 w-4" />
                  {t("settings.experimental.crashReporting.action.exportJson")}
                </button>
                <button
                  type="button"
                  onClick={() => void openCrashDownloadDirectory()}
                  disabled={saving || diagnosticBusy}
                  className={SECONDARY_BUTTON_CLASS_NAME}
                >
                  <FolderOpen className="h-4 w-4" />
                  {t(
                    "settings.experimental.crashReporting.action.openDownloadDirectory",
                  )}
                </button>
                <button
                  type="button"
                  onClick={handleSaveCrashConfig}
                  disabled={saving || diagnosticBusy}
                  className={PRIMARY_BUTTON_CLASS_NAME}
                >
                  {t("settings.experimental.crashReporting.action.save")}
                </button>
              </div>
            </div>
          </AdvancedDetails>

          {showClipboardGuide ? (
            <SurfacePanel
              icon={ShieldAlert}
              title={t("settings.experimental.clipboardGuide.title")}
              description={t(
                "settings.experimental.clipboardGuide.description",
              )}
            >
              <Suspense
                fallback={
                  <DeferredPanelFallback
                    label={t(
                      "settings.experimental.clipboardGuide.fallbackLabel",
                    )}
                  />
                }
              >
                <ClipboardPermissionGuideCard />
              </Suspense>
            </SurfacePanel>
          ) : null}

          <AdvancedDetails
            icon={RefreshCw}
            title={t("settings.experimental.update.title")}
            description={t("settings.experimental.update.description")}
          >
            <Suspense
              fallback={
                <DeferredPanelFallback
                  label={t("settings.experimental.update.fallbackLabel")}
                />
              }
            >
              <UpdateCheckSettings />
            </Suspense>
          </AdvancedDetails>

          <AdvancedDetails
            icon={ShieldAlert}
            title={t("settings.experimental.workspaceRepair.title")}
            description={t("settings.experimental.workspaceRepair.description")}
          >
            <Suspense
              fallback={
                <DeferredPanelFallback
                  label={t(
                    "settings.experimental.workspaceRepair.fallbackLabel",
                  )}
                />
              }
            >
              <WorkspaceRepairHistoryCard
                className="rounded-[22px] border-slate-200/80 bg-white p-4"
                description={t(
                  "settings.experimental.workspaceRepair.cardDescription",
                )}
              />
            </Suspense>
          </AdvancedDetails>
        </div>
      </div>
    </div>
  );
}
