import { ArrowLeft, CircleAlert, LoaderCircle, MoreHorizontal, Pause, Pencil, Play, Trash2, X } from "lucide-react";
import type { TFunction } from "i18next";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import type { ScheduledTask, ScheduledTaskRunSummary } from "@/lib/api/scheduledTasks";
import { describeScheduledTaskSchedule, formatScheduledTaskTime, scheduledTaskPresentationCopy, scheduledTaskStatusLabel } from "./scheduledTaskPresentation";
import { ScheduledTaskRunHistory } from "./ScheduledTaskRunHistory";

interface ScheduledTaskDetailsProps {
  task: ScheduledTask;
  runs: ScheduledTaskRunSummary[];
  loadingRuns: boolean;
  busyAction: string | null;
  locale: string;
  t: TFunction<"workspace">;
  onBack: () => void;
  onClose: () => void;
  onEdit: () => void;
  onToggleEnabled: () => void;
  onRun: () => void;
  onDelete: () => void;
  onOpenRun: (run: ScheduledTaskRunSummary) => void;
}

export function ScheduledTaskDetails({ task, runs, loadingRuns, busyAction, locale, t, onBack, onClose, onEdit, onToggleEnabled, onRun, onDelete, onOpenRun }: ScheduledTaskDetailsProps) {
  const copy = scheduledTaskPresentationCopy(t);
  const currentRun = runs.find((run) => run.status === "running" || run.status === "queued") ?? task.lastRunSummary;
  const status = scheduledTaskStatusLabel(currentRun, task.enabled, Boolean(currentRun?.error), copy);
  const running = currentRun?.status === "running" || currentRun?.status === "queued";
  return (
    <div className="flex h-full min-h-0 flex-col bg-white">
      <header className="flex min-h-16 items-center gap-3 border-b border-slate-200 px-4 py-3 sm:px-7">
        <Button variant="ghost" size="icon" className="md:hidden" aria-label={t("scheduledTasks.action.back")} title={t("scheduledTasks.action.back")} onClick={onBack}><ArrowLeft className="h-5 w-5" /></Button>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className={`h-2 w-2 shrink-0 rounded-full ${task.enabled ? "bg-emerald-500" : "bg-slate-300"}`} />
            <span className="text-xs font-medium text-slate-500">{status}</span>
          </div>
          <h2 className="mt-0.5 truncate text-lg font-semibold text-slate-950" title={task.title}>{task.title}</h2>
        </div>
        {running && currentRun?.sessionId ? (
          <Button className="bg-slate-900 hover:bg-slate-800" onClick={() => onOpenRun(currentRun)}>{t("scheduledTasks.action.openRun")}</Button>
        ) : (
          <Button variant="outline" disabled={busyAction !== null} onClick={onToggleEnabled}>
            {busyAction === "toggle" ? <LoaderCircle className="mr-2 h-4 w-4 animate-spin" /> : task.enabled ? <Pause className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
            {task.enabled ? t("scheduledTasks.action.pause") : t("scheduledTasks.action.resume")}
          </Button>
        )}
        <DropdownMenu>
          <DropdownMenuTrigger asChild><Button variant="ghost" size="icon" aria-label={t("scheduledTasks.action.more")} title={t("scheduledTasks.action.more")}><MoreHorizontal className="h-5 w-5" /></Button></DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-44 bg-white">
            <DropdownMenuItem onClick={onRun}><Play className="h-4 w-4" />{t("scheduledTasks.action.runNow")}</DropdownMenuItem>
            <DropdownMenuItem onClick={onEdit}><Pencil className="h-4 w-4" />{t("scheduledTasks.action.edit")}</DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem className="text-rose-700" onClick={onDelete}><Trash2 className="h-4 w-4" />{t("scheduledTasks.action.delete")}</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
        <Button variant="ghost" size="icon" aria-label={t("scheduledTasks.action.close")} title={t("scheduledTasks.action.close")} onClick={onClose}><X className="h-5 w-5" /></Button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-6 sm:px-8">
        <div className="mx-auto max-w-[900px]">
          {currentRun?.error ? (
            <div className="mb-6 flex gap-3 border-l-2 border-amber-400 bg-amber-50 px-4 py-3 text-sm text-amber-950"><CircleAlert className="mt-0.5 h-4 w-4 shrink-0" /><div><p className="font-semibold">{t("scheduledTasks.details.attentionTitle")}</p><p className="mt-1 text-xs leading-5">{currentRun.error}</p></div></div>
          ) : null}
          <p className="whitespace-pre-wrap text-sm leading-6 text-slate-700">{task.prompt}</p>
          <DetailSection title={t("scheduledTasks.details.execution")}>
            <DetailRow label={t("scheduledTasks.details.threadMode")} value={t(task.execution.threadMode === "new_thread" ? "scheduledTasks.editor.threadMode.new" : "scheduledTasks.editor.threadMode.continue")} />
            <DetailRow label={t("scheduledTasks.details.project")} value={task.execution.projectId || t("scheduledTasks.value.notSet")} />
            <DetailRow label={t("scheduledTasks.details.model")} value={task.execution.modelId || t("scheduledTasks.editor.value.inherit")} />
            <DetailRow label={t("scheduledTasks.details.reasoning")} value={task.execution.reasoningEffort || t("scheduledTasks.editor.value.inherit")} />
          </DetailSection>
          <DetailSection title={t("scheduledTasks.details.schedule")}>
            <DetailRow label={t("scheduledTasks.details.repeat")} value={describeScheduledTaskSchedule(task.schedule, copy, locale)} />
            <DetailRow label={t("scheduledTasks.details.timezone")} value={task.schedule.timezone} />
            <DetailRow label={t("scheduledTasks.details.nextRun")} value={task.enabled ? formatScheduledTaskTime(task.nextRunAt, locale) : t("scheduledTasks.status.paused")} />
            <DetailRow label={t("scheduledTasks.details.notification")} value={t(`scheduledTasks.notification.${task.notificationPolicy === "all_runs" ? "all" : task.notificationPolicy}`)} />
          </DetailSection>
          <ScheduledTaskRunHistory runs={runs} loading={loadingRuns} locale={locale} t={t} onOpenRun={onOpenRun} />
        </div>
      </div>
    </div>
  );
}

function DetailSection({ title, children }: { title: string; children: ReactNode }) { return <section className="mt-8"><h3 className="mb-2 text-sm font-semibold text-slate-900">{title}</h3><div className="divide-y divide-slate-100 border-y border-slate-200">{children}</div></section>; }
function DetailRow({ label, value }: { label: string; value: string }) { return <div className="grid min-h-12 grid-cols-[minmax(110px,0.35fr)_minmax(0,1fr)] items-center gap-4 py-2 text-sm"><span className="text-slate-500">{label}</span><span className="min-w-0 break-words text-right font-medium text-slate-900">{value}</span></div>; }
