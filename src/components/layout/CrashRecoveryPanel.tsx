import { useCallback, useMemo, useState } from "react";
import {
  AlertTriangle,
  Download,
  FolderOpen,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { getConfig } from "@/lib/api/appConfig";
import { getLogs, getPersistedLogsTail } from "@/lib/api/logs";
import {
  buildCrashDiagnosticPayload,
  clearCrashDiagnosticHistory,
  collectGeneralWorkbenchDocumentStateForDiagnostic,
  copyCrashDiagnosticJsonToClipboard,
  copyCrashDiagnosticToClipboard,
  exportCrashDiagnosticToJson,
  isClipboardPermissionDeniedError,
  normalizeCrashReportingConfig,
  openCrashDiagnosticDownloadDirectory,
} from "@/lib/crashDiagnostic";
import { getProjectByRootPath, updateProject } from "@/lib/api/project";
import { cn } from "@/lib/utils";
import { ClipboardPermissionGuideCard } from "@/components/settings-v2/system/shared/ClipboardPermissionGuideCard";
import { open as openDialog } from "@/lib/desktop-host/dialog";
import { notifyProjectRuntimeAgentsGuide } from "@/components/workspace/services/runtimeAgentsGuideService";
import {
  buildCrashRecoveryReloadUrl,
  isModuleImportFailureErrorMessage,
} from "./CrashRecoveryPanel.helpers";
import { useTranslation } from "react-i18next";

interface CrashRecoveryPanelProps {
  error: Error | null;
  componentStack: string;
  onRetry: () => void;
}

const VITE_CACHE_DIR = "node_modules/.vite";
const VITE_ELECTRON_CACHE_DIR = "node_modules/.vite-electron";

export function CrashRecoveryPanel({
  error,
  componentStack,
  onRetry,
}: CrashRecoveryPanelProps) {
  const { t } = useTranslation("errors");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);
  const [showClipboardGuide, setShowClipboardGuide] = useState(false);
  const errorMessage = error?.message ?? "";
  const isModuleImportFailure = useMemo(
    () => isModuleImportFailureErrorMessage(errorMessage),
    [errorMessage],
  );

  const sceneTag =
    errorMessage.includes("Workspace 路径不存在") ||
    errorMessage.includes("Workspace 路径存在但不是目录")
      ? "workspace-path-missing"
      : "crash-recovery";

  // 从错误消息中提取旧的 workspace 路径
  const oldWorkspacePath = useMemo(() => {
    const match = errorMessage.match(
      /Workspace 路径(?:不存在，且自动创建失败|存在但不是目录): (.+?)。/,
    );
    return match?.[1] ?? null;
  }, [errorMessage]);

  const stackPreview = useMemo(() => {
    const raw = componentStack.trim();
    if (!raw) {
      return "";
    }
    return raw
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .slice(0, 8)
      .join(" | ");
  }, [componentStack]);

  const buildPayload = useCallback(async () => {
    const notes: string[] = [];
    if (error?.message) {
      notes.push(`boundary_error: ${error.message}`);
    }
    if (error?.stack) {
      notes.push(
        `boundary_error_stack: ${error.stack.split("\n").slice(0, 5).join(" | ")}`,
      );
    }
    if (stackPreview) {
      notes.push(`boundary_component_stack: ${stackPreview}`);
    }

    const [config, logs, persistedLogs, generalWorkbenchDocumentState] =
      await Promise.all([
        getConfig().catch(() => {
          notes.push("get_config_failed");
          return null;
        }),
        getLogs().catch(() => {
          notes.push("get_logs_failed");
          return [];
        }),
        getPersistedLogsTail(250).catch(() => {
          notes.push("get_persisted_logs_tail_failed");
          return [];
        }),
        collectGeneralWorkbenchDocumentStateForDiagnostic().catch(() => {
          notes.push("get_general_workbench_document_state_failed");
          return null;
        }),
      ]);

    return buildCrashDiagnosticPayload({
      crashConfig: normalizeCrashReportingConfig(config?.crash_reporting),
      logs,
      persistedLogTail: persistedLogs,
      generalWorkbenchDocumentState,
      appVersion: import.meta.env.VITE_APP_VERSION,
      platform: navigator.platform,
      userAgent: navigator.userAgent,
      maxCrashLogs: 40,
      maxPersistedLogs: 250,
      collectionNotes: notes,
    });
  }, [error, stackPreview]);

  const runAction = useCallback(
    async (
      action: (
        payload: Awaited<ReturnType<typeof buildPayload>>,
      ) => Promise<void> | void,
      successText: string,
    ) => {
      setBusy(true);
      setMessage(null);
      setShowClipboardGuide(false);
      try {
        const payload = await buildPayload();
        await action(payload);
        setMessage({ type: "success", text: successText });
      } catch (actionError) {
        const denied = isClipboardPermissionDeniedError(actionError);
        setShowClipboardGuide(denied);
        setMessage({
          type: "error",
          text:
            actionError instanceof Error
              ? actionError.message
              : t("errors.crashRecovery.message.generateDiagnosticFailed"),
        });
      } finally {
        setBusy(false);
      }
    },
    [buildPayload, t],
  );

  const handleCopyTemplate = useCallback(() => {
    void runAction(
      (payload) => copyCrashDiagnosticToClipboard(payload),
      t("errors.crashRecovery.message.diagnosticCopied"),
    );
  }, [runAction, t]);

  const handleCopyJson = useCallback(() => {
    void runAction(
      (payload) => copyCrashDiagnosticJsonToClipboard(payload),
      t("errors.crashRecovery.message.diagnosticJsonCopied"),
    );
  }, [runAction, t]);

  const handleExportJson = useCallback(() => {
    void runAction(async (payload) => {
      exportCrashDiagnosticToJson(payload, { sceneTag });
      try {
        await openCrashDiagnosticDownloadDirectory();
      } catch {
        // 导出结果已经落盘，打开目录失败不覆盖导出成功反馈。
      }
    }, t("errors.crashRecovery.message.diagnosticExportedShort"));
  }, [sceneTag, runAction, t]);

  const handleSelectNewDirectory = useCallback(async () => {
    if (!oldWorkspacePath) {
      setMessage({
        type: "error",
        text: t("errors.crashRecovery.message.workspacePathExtractFailed"),
      });
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const newPath = await openDialog({ directory: true, multiple: false });
      if (!newPath) {
        return;
      }
      const workspace = await getProjectByRootPath(oldWorkspacePath);
      if (!workspace) {
        setMessage({
          type: "error",
          text: t("errors.crashRecovery.message.workspaceNotFound", {
            path: oldWorkspacePath,
          }),
        });
        return;
      }
      await updateProject(workspace.id, { rootPath: newPath });
      notifyProjectRuntimeAgentsGuide(
        {
          id: workspace.id,
          rootPath: newPath,
        },
        {
          successMessage: t(
            "errors.crashRecovery.message.workspacePathUpdatedShort",
          ),
          showSuccessWhenGuideAlreadySeen: false,
        },
      );
      setMessage({
        type: "success",
        text: t("errors.crashRecovery.message.workspacePathUpdated", {
          path: newPath,
        }),
      });
      onRetry();
    } catch (err) {
      setMessage({
        type: "error",
        text:
          err instanceof Error
            ? err.message
            : t("errors.crashRecovery.message.workspacePathUpdateFailed"),
      });
    } finally {
      setBusy(false);
    }
  }, [oldWorkspacePath, onRetry, t]);

  const handleOpenDownloadDirectory = useCallback(async () => {
    setBusy(true);
    setMessage(null);
    setShowClipboardGuide(false);
    try {
      await openCrashDiagnosticDownloadDirectory();
    } catch (actionError) {
      setMessage({
        type: "error",
        text:
          actionError instanceof Error
            ? actionError.message
            : t("errors.crashRecovery.message.generateDiagnosticFailed"),
      });
    } finally {
      setBusy(false);
    }
  }, [t]);

  const handleClearDiagnosticHistory = useCallback(async () => {
    const confirmed =
      typeof window === "undefined" ||
      window.confirm(t("errors.crashRecovery.confirm.clearDiagnosticHistory"));
    if (!confirmed) {
      return;
    }

    setBusy(true);
    setMessage(null);
    setShowClipboardGuide(false);
    try {
      await clearCrashDiagnosticHistory();
      setMessage({
        type: "success",
        text: t("errors.crashRecovery.message.diagnosticHistoryCleared"),
      });
    } catch (err) {
      setMessage({
        type: "error",
        text:
          err instanceof Error
            ? err.message
            : t("errors.crashRecovery.message.clearDiagnosticHistoryFailed"),
      });
    } finally {
      setBusy(false);
    }
  }, [t]);

  const handleForceResourceReload = useCallback(() => {
    if (typeof window === "undefined") {
      return;
    }

    setBusy(true);
    setMessage(null);
    setShowClipboardGuide(false);

    try {
      const reloadUrl = buildCrashRecoveryReloadUrl(
        window.location.href,
        `${Date.now()}`,
      );
      window.location.replace(reloadUrl);
    } catch (err) {
      setBusy(false);
      setMessage({
        type: "error",
        text:
          err instanceof Error
            ? err.message
            : t("errors.crashRecovery.message.resourceReloadFailed"),
      });
    }
  }, [t]);

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-background px-6">
      <div className="w-full max-w-2xl rounded-xl border bg-card p-6 shadow-sm">
        <div className="mb-4 flex items-start gap-3">
          <div className="rounded-lg bg-rose-500/10 p-2">
            <AlertTriangle className="h-5 w-5 text-rose-500" />
          </div>
          <div className="min-w-0">
            <h2 className="text-lg font-semibold">
              {t("errors.crashRecovery.title")}
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {t("errors.crashRecovery.descriptionPrefix")}
              {isModuleImportFailure
                ? t("errors.crashRecovery.descriptionAction.forceReload")
                : t("errors.crashRecovery.descriptionAction.retry")}
              {t("errors.crashRecovery.descriptionSuffix")}
            </p>
          </div>
        </div>

        {error?.message && (
          <div className="mb-4 rounded-md border border-rose-200 bg-rose-50/70 px-3 py-2 text-xs text-rose-700 dark:border-rose-900/40 dark:bg-rose-950/20 dark:text-rose-300">
            {t("errors.crashRecovery.latestError", {
              message: error.message,
            })}
          </div>
        )}

        {isModuleImportFailure && (
          <div className="mb-4 rounded-md border border-amber-200 bg-amber-50/80 px-3 py-2 text-xs text-amber-800 dark:border-amber-900/40 dark:bg-amber-950/20 dark:text-amber-200">
            {t("errors.crashRecovery.moduleImportFailure.prefix")}
            <code className="mx-1 rounded bg-slate-100 px-1 py-0.5 text-slate-700 dark:bg-white/10 dark:text-slate-100">
              {VITE_CACHE_DIR}
            </code>
            {t("errors.crashRecovery.moduleImportFailure.middle")}
            <code className="mx-1 rounded bg-slate-100 px-1 py-0.5 text-slate-700 dark:bg-white/10 dark:text-slate-100">
              {VITE_ELECTRON_CACHE_DIR}
            </code>
            {t("errors.crashRecovery.moduleImportFailure.suffix")}
          </div>
        )}

        {message && (
          <div
            className={cn(
              "mb-4 rounded-md px-3 py-2 text-sm",
              message.type === "success"
                ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                : "bg-destructive/10 text-destructive",
            )}
          >
            {message.text}
          </div>
        )}

        {showClipboardGuide && (
          <ClipboardPermissionGuideCard className="mb-4" />
        )}

        <div className="flex flex-wrap items-center gap-2">
          {sceneTag === "workspace-path-missing" ? (
            <button
              type="button"
              onClick={() => void handleSelectNewDirectory()}
              disabled={busy}
              className={cn(
                "inline-flex items-center gap-1 rounded-md border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs text-emerald-700 transition-colors dark:border-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-300",
                busy && "cursor-not-allowed opacity-50",
              )}
            >
              <FolderOpen className="h-3.5 w-3.5" />
              {t("errors.crashRecovery.action.selectDirectory")}
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => void handleClearDiagnosticHistory()}
            disabled={busy}
            className={cn(
              "inline-flex items-center gap-1 rounded-md border border-rose-200 bg-rose-50 px-3 py-1.5 text-xs text-rose-700 transition-colors dark:border-rose-900/40 dark:bg-rose-950/20 dark:text-rose-300",
              busy && "cursor-not-allowed opacity-50",
            )}
          >
            <Trash2 className="h-3.5 w-3.5" />
            {t("errors.crashRecovery.action.clearDiagnosticHistory")}
          </button>
          <button
            type="button"
            onClick={handleCopyTemplate}
            disabled={busy}
            className={cn(
              "rounded-md border px-3 py-1.5 text-xs transition-colors",
              busy && "cursor-not-allowed opacity-50",
            )}
          >
            {t("errors.crashRecovery.action.copyDiagnostic")}
          </button>
          <button
            type="button"
            onClick={handleCopyJson}
            disabled={busy}
            className={cn(
              "rounded-md border px-3 py-1.5 text-xs transition-colors",
              busy && "cursor-not-allowed opacity-50",
            )}
          >
            {t("errors.crashRecovery.action.copyJson")}
          </button>
          <button
            type="button"
            onClick={handleExportJson}
            disabled={busy}
            className={cn(
              "inline-flex items-center gap-1 rounded-md border px-3 py-1.5 text-xs transition-colors",
              busy && "cursor-not-allowed opacity-50",
            )}
          >
            <Download className="h-3.5 w-3.5" />
            {t("errors.crashRecovery.action.exportJson")}
          </button>
          <button
            type="button"
            onClick={handleOpenDownloadDirectory}
            disabled={busy}
            className={cn(
              "rounded-md border px-3 py-1.5 text-xs transition-colors",
              busy && "cursor-not-allowed opacity-50",
            )}
          >
            {t("errors.crashRecovery.action.openDownloadDirectory")}
          </button>
          {isModuleImportFailure ? (
            <button
              type="button"
              onClick={handleForceResourceReload}
              disabled={busy}
              className={cn(
                "inline-flex items-center gap-1 rounded-md border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs text-emerald-700 transition-colors dark:border-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-300",
                busy && "cursor-not-allowed opacity-50",
              )}
            >
              <RefreshCw className="h-3.5 w-3.5" />
              {t("errors.crashRecovery.action.forceReload")}
            </button>
          ) : null}
          <button
            type="button"
            onClick={onRetry}
            disabled={busy}
            className={cn(
              "inline-flex items-center gap-1 rounded-md border px-3 py-1.5 text-xs transition-colors",
              busy && "cursor-not-allowed opacity-50",
            )}
          >
            <RefreshCw className="h-3.5 w-3.5" />
            {isModuleImportFailure
              ? t("errors.crashRecovery.action.retryOnly")
              : t("errors.crashRecovery.action.retry")}
          </button>
        </div>
      </div>
    </div>
  );
}
