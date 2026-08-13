import { AlertCircle, CalendarClock, Search, X } from "lucide-react";
import type { TFunction } from "i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { ScheduledTaskSummary } from "@/lib/api/scheduledTasks";
import {
  describeScheduledTaskSchedule,
  scheduledTaskPresentationCopy,
  scheduledTaskStatusLabel,
} from "./scheduledTaskPresentation";
import type { ScheduledTaskFilter } from "./scheduledTaskViewModel";

interface ScheduledTaskListProps {
  tasks: ScheduledTaskSummary[];
  selectedId: string | null;
  query: string;
  filter: ScheduledTaskFilter;
  locale: string;
  loading: boolean;
  t: TFunction<"workspace">;
  onQueryChange: (query: string) => void;
  onFilterChange: (filter: ScheduledTaskFilter) => void;
  onSelect: (id: string) => void;
  onCreate: () => void;
}

const FILTERS: ScheduledTaskFilter[] = ["all", "enabled", "paused"];

export function ScheduledTaskList({
  tasks,
  selectedId,
  query,
  filter,
  locale,
  loading,
  t,
  onQueryChange,
  onFilterChange,
  onSelect,
  onCreate,
}: ScheduledTaskListProps) {
  const copy = scheduledTaskPresentationCopy(t);
  return (
    <aside className="flex min-h-0 w-full shrink-0 flex-col border-b border-slate-200 bg-white md:w-[340px] md:border-b-0 md:border-r xl:w-[390px]">
      <div className="border-b border-slate-200 px-4 py-4">
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <Input
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder={t("scheduledTasks.search.placeholder")}
            aria-label={t("scheduledTasks.search.aria")}
            className="border-slate-200 bg-slate-50 pl-9 pr-9 focus-visible:bg-white"
          />
          {query ? (
            <button
              type="button"
              className="absolute right-2 top-1/2 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-md text-slate-500 hover:bg-slate-200"
              aria-label={t("scheduledTasks.search.clear")}
              title={t("scheduledTasks.search.clear")}
              onClick={() => onQueryChange("")}
            >
              <X className="h-4 w-4" />
            </button>
          ) : null}
        </div>
        <div
          className="mt-3 grid grid-cols-3 rounded-md bg-slate-100 p-1"
          role="group"
          aria-label={t("scheduledTasks.filter.aria")}
        >
          {FILTERS.map((value) => (
            <button
              key={value}
              type="button"
              className={cn(
                "h-8 rounded px-2 text-xs font-medium text-slate-600 transition-colors",
                filter === value && "bg-white text-slate-950 shadow-sm",
              )}
              aria-pressed={filter === value}
              onClick={() => onFilterChange(value)}
            >
              {t(`scheduledTasks.filter.${value}`)}
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {loading ? (
          <div className="space-y-2 p-2" aria-label={t("scheduledTasks.loading")}>
            {[0, 1, 2].map((item) => (
              <div key={item} className="h-20 animate-pulse rounded-md bg-slate-100" />
            ))}
          </div>
        ) : tasks.length ? (
          <div className="space-y-1">
            {tasks.map((task) => {
              const selected = task.id === selectedId;
              const status = scheduledTaskStatusLabel(
                task.lastRun,
                task.enabled,
                task.attention,
                copy,
              );
              return (
                <button
                  key={task.id}
                  type="button"
                  className={cn(
                    "w-full rounded-md border border-transparent px-3 py-3 text-left transition-colors hover:bg-slate-50",
                    selected && "border-emerald-200 bg-emerald-50",
                  )}
                  aria-current={selected ? "true" : undefined}
                  onClick={() => onSelect(task.id)}
                >
                  <span className="flex items-start gap-3">
                    <span
                      className={cn(
                        "mt-1.5 h-2 w-2 shrink-0 rounded-full bg-slate-300",
                        task.enabled && !task.attention && "bg-emerald-500",
                        task.attention && "bg-amber-500",
                      )}
                    />
                    <span className="min-w-0 flex-1">
                      <span className="flex min-w-0 items-center gap-2">
                        <span className="truncate text-sm font-semibold text-slate-900" title={task.title}>
                          {task.title}
                        </span>
                        {task.attention ? (
                          <AlertCircle className="h-4 w-4 shrink-0 text-amber-600" />
                        ) : null}
                      </span>
                      <span className="mt-1 block truncate text-xs text-slate-500">
                        {describeScheduledTaskSchedule(task.schedule, copy, locale)}
                      </span>
                      <span className="mt-1 block text-xs font-medium text-slate-600">
                        {status}
                      </span>
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        ) : (
          <div className="flex min-h-64 flex-col items-center justify-center px-6 text-center">
            <CalendarClock className="h-8 w-8 text-slate-400" />
            <p className="mt-3 text-sm font-semibold text-slate-900">
              {query || filter !== "all"
                ? t("scheduledTasks.empty.filtered.title")
                : t("scheduledTasks.empty.title")}
            </p>
            <p className="mt-1 text-xs leading-5 text-slate-500">
              {query || filter !== "all"
                ? t("scheduledTasks.empty.filtered.description")
                : t("scheduledTasks.empty.description")}
            </p>
            {query || filter !== "all" ? (
              <Button
                variant="outline"
                size="sm"
                className="mt-4"
                onClick={() => {
                  onQueryChange("");
                  onFilterChange("all");
                }}
              >
                {t("scheduledTasks.action.clearFilters")}
              </Button>
            ) : (
              <Button size="sm" className="mt-4 bg-slate-900 hover:bg-slate-800" onClick={onCreate}>
                {t("scheduledTasks.action.create")}
              </Button>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
