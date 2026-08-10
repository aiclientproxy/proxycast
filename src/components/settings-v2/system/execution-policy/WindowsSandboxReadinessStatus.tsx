import { useCallback, useEffect, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  CircleDashed,
  RefreshCw,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  readWindowsSandboxReadiness,
  type WindowsSandboxReadiness,
} from "@/lib/api/windowsSandbox";
import { cn } from "@/lib/utils";
import { isWindowsDesktopPlatform } from "./windowsSandboxPlatform";

type ReadinessState = WindowsSandboxReadiness | "checking" | "error";

const STATUS_STYLES: Record<ReadinessState, string> = {
  checking: "border-slate-200 bg-slate-100 text-slate-600",
  ready: "border-emerald-200 bg-emerald-50 text-emerald-700",
  notConfigured: "border-slate-200 bg-slate-100 text-slate-600",
  updateRequired: "border-amber-200 bg-amber-50 text-amber-800",
  error: "border-rose-200 bg-rose-50 text-rose-700",
};

const STATUS_ICONS: Record<ReadinessState, typeof CheckCircle2> = {
  checking: CircleDashed,
  ready: CheckCircle2,
  notConfigured: CircleDashed,
  updateRequired: AlertCircle,
  error: AlertCircle,
};

export function WindowsSandboxReadinessStatus() {
  const { t } = useTranslation("settings");
  const [status, setStatus] = useState<ReadinessState>("checking");

  const refresh = useCallback(async () => {
    setStatus("checking");
    try {
      setStatus(await readWindowsSandboxReadiness());
    } catch {
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    if (isWindowsDesktopPlatform()) {
      void refresh();
    }
  }, [refresh]);

  if (!isWindowsDesktopPlatform()) {
    return null;
  }

  const StatusIcon = STATUS_ICONS[status];

  return (
    <div
      className="mt-4 flex min-h-[76px] items-start justify-between gap-4 border-t border-slate-200 pt-4"
      data-status={status}
      data-testid="windows-sandbox-readiness"
    >
      <div className="min-w-0 space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium text-slate-800">
            {t("settings.executionPolicy.windowsSandbox.title")}
          </span>
          <span
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium",
              STATUS_STYLES[status],
            )}
          >
            <StatusIcon
              className={cn("h-3.5 w-3.5", {
                "animate-spin": status === "checking",
              })}
            />
            {t(`settings.executionPolicy.windowsSandbox.status.${status}`)}
          </span>
        </div>
        <p className="text-xs leading-5 text-slate-500">
          {t(`settings.executionPolicy.windowsSandbox.detail.${status}`)}
        </p>
      </div>
      <button
        type="button"
        className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-500 transition hover:border-slate-300 hover:text-slate-900 disabled:cursor-wait disabled:opacity-50"
        aria-label={t("settings.executionPolicy.windowsSandbox.action.refresh")}
        title={t("settings.executionPolicy.windowsSandbox.action.refresh")}
        disabled={status === "checking"}
        onClick={() => void refresh()}
      >
        <RefreshCw className="h-4 w-4" />
      </button>
    </div>
  );
}
