import {
  ArrowUpRight,
  CircleAlert,
  Clock3,
  LoaderCircle,
  TriangleAlert,
} from "lucide-react";
import type { TFunction } from "i18next";
import { Button } from "@/components/ui/button";
import type { ScheduledTaskRunSummary } from "@/lib/api/scheduledTasks";
import {
  formatScheduledTaskRunDuration,
  formatScheduledTaskTime,
  scheduledTaskPresentationCopy,
} from "./scheduledTaskPresentation";

interface ScheduledTaskRunHistoryProps {
  runs: ScheduledTaskRunSummary[];
  loading: boolean;
  locale: string;
  t: TFunction<"workspace">;
  onOpenRun: (run: ScheduledTaskRunSummary) => void;
}

export function ScheduledTaskRunHistory({
  runs,
  loading,
  locale,
  t,
  onOpenRun,
}: ScheduledTaskRunHistoryProps) {
  const copy = scheduledTaskPresentationCopy(t);
  return (
    <section className="border-t border-slate-200 pt-6">
      <div className="mb-3 flex items-center justify-between gap-4">
        <h3 className="text-sm font-semibold text-slate-900">
          {t("scheduledTasks.history.title")}
        </h3>
        <span className="text-xs text-slate-500">
          {t("scheduledTasks.history.count", { count: runs.length })}
        </span>
      </div>
      {loading ? (
        <div className="flex h-24 items-center justify-center text-slate-500">
          <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
          {t("scheduledTasks.history.loading")}
        </div>
      ) : runs.length ? (
        <div className="divide-y divide-slate-100 border-y border-slate-200">
          {runs.map((run) => {
            const duration = formatScheduledTaskRunDuration(run);
            const canOpen = Boolean(run.sessionId);
            return (
              <div key={run.id} className="flex min-h-16 items-center gap-3 py-3">
                {run.status === "running" || run.status === "queued" ? (
                  <Clock3 className="h-4 w-4 shrink-0 text-sky-600" />
                ) : run.status === "failed" || run.status === "error" ? (
                  <CircleAlert className="h-4 w-4 shrink-0 text-rose-600" />
                ) : run.status === "missed" ? (
                  <TriangleAlert className="h-4 w-4 shrink-0 text-amber-600" />
                ) : (
                  <span className="h-2.5 w-2.5 shrink-0 rounded-full bg-emerald-500" />
                )}
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-slate-900">
                    {copy.status[run.status] ?? run.status}
                  </p>
                  <p className="mt-0.5 truncate text-xs text-slate-500">
                    {formatScheduledTaskTime(run.startedAt, locale)}
                    {duration === null
                      ? ""
                      : ` · ${t("scheduledTasks.history.duration", { seconds: duration })}`}
                  </p>
                  {run.error ? (
                    <p className="mt-1 line-clamp-2 text-xs text-rose-700">{run.error}</p>
                  ) : null}
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  disabled={!canOpen}
                  aria-label={t("scheduledTasks.history.open")}
                  title={
                    canOpen
                      ? t("scheduledTasks.history.open")
                      : t("scheduledTasks.history.openUnavailable")
                  }
                  onClick={() => onOpenRun(run)}
                >
                  <ArrowUpRight className="h-4 w-4" />
                </Button>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="flex min-h-24 items-center justify-center border-y border-slate-200 text-sm text-slate-500">
          {t("scheduledTasks.history.empty")}
        </div>
      )}
    </section>
  );
}
