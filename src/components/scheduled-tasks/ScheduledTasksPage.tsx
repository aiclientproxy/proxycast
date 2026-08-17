import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bot,
  CalendarClock,
  LoaderCircle,
  PenLine,
  Plus,
  RefreshCw,
} from "lucide-react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { isAppServerBridgeAvailable } from "@/lib/api/appServerBridgeAvailability";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  scheduledTasksApi,
  subscribeScheduledTaskNotifications,
  type ScheduledTask,
  type ScheduledTaskRunSummary,
  type ScheduledTaskSchedule,
  type ScheduledTaskSummary,
} from "@/lib/api/scheduledTasks";
import type { Page, PageParams, ScheduledTasksPageParams } from "@/types/page";
import { ScheduledTaskDetails } from "./ScheduledTaskDetails";
import { ScheduledTaskEditor } from "./ScheduledTaskEditor";
import { ScheduledTaskList } from "./ScheduledTaskList";
import {
  buildScheduledTaskCreateRequest,
  buildScheduledTaskUpdateRequest,
  defaultScheduledTaskForm,
  filterScheduledTasks,
  scheduledTaskToForm,
  validateScheduledTaskForm,
  type ScheduledTaskFilter,
  type ScheduledTaskFormErrors,
  type ScheduledTaskFormState,
} from "./scheduledTaskViewModel";

interface ScheduledTasksPageProps {
  onNavigate?: (page: Page, params?: PageParams) => void;
  pageParams?: ScheduledTasksPageParams;
}

type EditorMode = "create" | "edit" | null;

export function ScheduledTasksPage({
  onNavigate,
  pageParams,
}: ScheduledTasksPageProps) {
  const { t, i18n } = useTranslation("workspace");
  const [tasks, setTasks] = useState<ScheduledTaskSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(
    pageParams?.selectedTaskId ?? null,
  );
  const [selectedTask, setSelectedTask] = useState<ScheduledTask | null>(null);
  const selectedIdRef = useRef(selectedId);
  const [runs, setRuns] = useState<ScheduledTaskRunSummary[]>([]);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<ScheduledTaskFilter>("all");
  const [loading, setLoading] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [editorMode, setEditorMode] = useState<EditorMode>(null);
  const [form, setForm] = useState<ScheduledTaskFormState>(() =>
    defaultScheduledTaskForm(),
  );
  const [formErrors, setFormErrors] = useState<ScheduledTaskFormErrors>({});
  const [preview, setPreview] = useState<string[]>([]);
  const [previewLoading, setPreviewLoading] = useState(false);

  const filteredTasks = useMemo(
    () => filterScheduledTasks(tasks, query, filter),
    [filter, query, tasks],
  );

  const loadTasks = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const response = await scheduledTasksApi.list({ limit: 200 });
      setTasks(response.items);
      setSelectedId((current) => {
        if (current && response.items.some((item) => item.id === current)) {
          return current;
        }
        return null;
      });
    } catch (error) {
      setLoadError(errorMessage(error));
    } finally {
      setLoading(false);
    }
  }, []);

  const loadTask = useCallback(
    async (id: string) => {
      setLoadingDetail(true);
      setLoadingRuns(true);
      try {
        const [task, history] = await Promise.all([
          scheduledTasksApi.read(id),
          scheduledTasksApi.listRuns(id),
        ]);
        if (!task) {
          throw new Error(t("scheduledTasks.error.notFound"));
        }
        setSelectedTask(task);
        setRuns(history);
      } catch (error) {
        toast.error(
          t("scheduledTasks.error.detail", { message: errorMessage(error) }),
        );
        setSelectedTask(null);
        setRuns([]);
      } finally {
        setLoadingDetail(false);
        setLoadingRuns(false);
      }
    },
    [t],
  );

  useEffect(() => {
    void loadTasks();
  }, [loadTasks]);

  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);

  useEffect(() => {
    if (!selectedId) {
      setSelectedTask(null);
      setRuns([]);
      return;
    }
    void loadTask(selectedId);
  }, [loadTask, selectedId]);

  useEffect(() => {
    let disposed = false;
    let refreshQueued = false;
    const detailTaskIds = new Set<string>();

    const queueRefresh = (taskId: string, includeDetail: boolean) => {
      if (includeDetail) {
        detailTaskIds.add(taskId);
      }
      if (refreshQueued) {
        return;
      }
      refreshQueued = true;
      queueMicrotask(() => {
        refreshQueued = false;
        if (disposed) {
          return;
        }
        const selectedTaskId = selectedIdRef.current;
        const shouldLoadDetail = Boolean(
          selectedTaskId && detailTaskIds.has(selectedTaskId),
        );
        detailTaskIds.clear();
        void Promise.all([
          loadTasks(),
          ...(selectedTaskId && shouldLoadDetail
            ? [loadTask(selectedTaskId)]
            : []),
        ]);
      });
    };

    const unsubscribe = subscribeScheduledTaskNotifications(
      {
        onChanged: ({ change, taskId }) => {
          if (change === "deleted" && selectedIdRef.current === taskId) {
            selectedIdRef.current = null;
            setSelectedId(null);
            setSelectedTask(null);
            setRuns([]);
            queueRefresh(taskId, false);
            return;
          }
          queueRefresh(taskId, selectedIdRef.current === taskId);
        },
        onRunUpdated: ({ taskId }) => {
          queueRefresh(taskId, selectedIdRef.current === taskId);
        },
      },
      { isBridgeAvailable: isAppServerBridgeAvailable },
    );
    return () => {
      disposed = true;
      unsubscribe();
    };
  }, [loadTask, loadTasks]);

  const startCreate = useCallback(
    (preset?: "daily" | "weekly" | "monitor") => {
      const next = defaultScheduledTaskForm();
      if (preset === "daily") {
        next.title = t("scheduledTasks.template.daily.title");
        next.prompt = t("scheduledTasks.template.daily.prompt");
        next.scheduleType = "weekdays";
        next.time = "08:30";
      } else if (preset === "weekly") {
        next.title = t("scheduledTasks.template.weekly.title");
        next.prompt = t("scheduledTasks.template.weekly.prompt");
        next.scheduleType = "weekly";
        next.days = ["FR"];
        next.time = "16:00";
      } else if (preset === "monitor") {
        next.title = t("scheduledTasks.template.monitor.title");
        next.prompt = t("scheduledTasks.template.monitor.prompt");
        next.scheduleType = "hourly";
        next.intervalHours = 4;
        next.time = "00:00";
      }
      next.projectId = pageParams?.projectId ?? "";
      next.sourceThreadId = pageParams?.threadId ?? "";
      setForm(next);
      setFormErrors({});
      setPreview([]);
      setEditorMode("create");
    },
    [pageParams?.projectId, pageParams?.threadId, t],
  );

  const startEdit = useCallback(() => {
    if (!selectedTask) return;
    setForm(scheduledTaskToForm(selectedTask));
    setFormErrors({});
    setPreview([]);
    setEditorMode("edit");
  }, [selectedTask]);

  const createWithLime = useCallback(() => {
    onNavigate?.("agent", {
      agentEntry: "claw",
      projectId: pageParams?.projectId,
      initialUserPrompt: t("scheduledTasks.createWithLime.prompt"),
      initialSessionName: t("scheduledTasks.createWithLime.sessionName"),
      entryBannerMessage: t("scheduledTasks.createWithLime.banner"),
      autoRunInitialPromptOnMount: false,
    });
  }, [onNavigate, pageParams?.projectId, t]);

  const saveTask = useCallback(async () => {
    const errors = validateScheduledTaskForm(form);
    setFormErrors(errors);
    if (Object.keys(errors).length) {
      toast.error(t("scheduledTasks.editor.validation.fix"));
      return;
    }
    setBusyAction("save");
    try {
      const task =
        editorMode === "edit" && selectedTask
          ? await scheduledTasksApi.update(
              selectedTask.id,
              buildScheduledTaskUpdateRequest(form, selectedTask.updatedAt),
            )
          : await scheduledTasksApi.create(
              buildScheduledTaskCreateRequest(form),
            );
      setEditorMode(null);
      setSelectedId(task.id);
      setSelectedTask(task);
      toast.success(
        t(
          editorMode === "edit"
            ? "scheduledTasks.toast.updated"
            : "scheduledTasks.toast.created",
        ),
      );
      await loadTasks();
      await loadTask(task.id);
    } catch (error) {
      toast.error(
        t("scheduledTasks.error.save", { message: errorMessage(error) }),
      );
    } finally {
      setBusyAction(null);
    }
  }, [editorMode, form, loadTask, loadTasks, selectedTask, t]);

  const previewSchedule = useCallback(
    async (schedule: ScheduledTaskSchedule) => {
      setPreviewLoading(true);
      try {
        const result = await scheduledTasksApi.previewSchedule(schedule);
        setPreview(result.nextRunAt);
      } catch (error) {
        toast.error(
          t("scheduledTasks.error.preview", { message: errorMessage(error) }),
        );
      } finally {
        setPreviewLoading(false);
      }
    },
    [t],
  );

  const toggleEnabled = useCallback(async () => {
    if (!selectedTask) return;
    setBusyAction("toggle");
    try {
      const task = await scheduledTasksApi.setEnabled(
        selectedTask.id,
        !selectedTask.enabled,
      );
      setSelectedTask(task);
      await loadTasks();
    } catch (error) {
      toast.error(
        t("scheduledTasks.error.toggle", { message: errorMessage(error) }),
      );
    } finally {
      setBusyAction(null);
    }
  }, [loadTasks, selectedTask, t]);

  const runNow = useCallback(async () => {
    if (!selectedTask) return;
    const taskId = selectedTask.id;
    setBusyAction("run");
    try {
      await scheduledTasksApi.startRun(taskId);
      toast.success(t("scheduledTasks.toast.runStarted"));
    } catch (error) {
      toast.error(
        t("scheduledTasks.error.run", { message: errorMessage(error) }),
      );
    } finally {
      await Promise.all([loadTask(taskId), loadTasks()]);
      setBusyAction(null);
    }
  }, [loadTask, loadTasks, selectedTask, t]);

  const removeTask = useCallback(async () => {
    const confirmationKey = hasActiveRun(selectedTask, runs)
      ? "scheduledTasks.confirm.deleteRunning"
      : "scheduledTasks.confirm.delete";
    if (
      !selectedTask ||
      !window.confirm(t(confirmationKey, { title: selectedTask.title }))
    ) {
      return;
    }
    setBusyAction("delete");
    try {
      await scheduledTasksApi.remove(selectedTask.id);
      setSelectedId(null);
      setSelectedTask(null);
      toast.success(t("scheduledTasks.toast.deleted"));
      await loadTasks();
    } catch (error) {
      toast.error(
        t("scheduledTasks.error.delete", { message: errorMessage(error) }),
      );
    } finally {
      setBusyAction(null);
    }
  }, [loadTasks, runs, selectedTask, t]);

  const openRun = useCallback(
    (run: ScheduledTaskRunSummary) => {
      if (!run.sessionId) return;
      onNavigate?.("agent", {
        agentEntry: "claw",
        projectId: selectedTask?.execution.projectId ?? undefined,
        initialSessionId: run.sessionId,
        initialSessionName: selectedTask?.title,
      });
    },
    [onNavigate, selectedTask],
  );

  return (
    <div className="lime-workbench-theme-scope flex h-full min-h-0 flex-1 flex-col bg-slate-50">
      <header className="flex min-h-[72px] items-center justify-between gap-4 border-b border-slate-200 bg-white px-5 py-3 sm:px-7">
        <div className="min-w-0">
          <h1 className="truncate text-xl font-semibold text-slate-950">
            {t("scheduledTasks.title")}
          </h1>
          <p className="mt-0.5 truncate text-sm text-slate-500">
            {t("scheduledTasks.subtitle")}
          </p>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button className="shrink-0 bg-slate-900 hover:bg-slate-800">
              <Plus className="mr-2 h-4 w-4" />
              {t("scheduledTasks.action.create")}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56 bg-white">
            <DropdownMenuItem onClick={createWithLime}>
              <Bot className="h-4 w-4" />
              {t("scheduledTasks.action.createWithLime")}
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => startCreate()}>
              <PenLine className="h-4 w-4" />
              {t("scheduledTasks.action.manual")}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </header>

      {loadError ? (
        <div className="flex flex-1 flex-col items-center justify-center p-8 text-center">
          <CalendarClock className="h-9 w-9 text-rose-500" />
          <h2 className="mt-4 text-base font-semibold text-slate-950">
            {t("scheduledTasks.error.loadTitle")}
          </h2>
          <p className="mt-1 max-w-md text-sm text-slate-600">{loadError}</p>
          <Button
            variant="outline"
            className="mt-5"
            onClick={() => void loadTasks()}
          >
            <RefreshCw className="mr-2 h-4 w-4" />
            {t("scheduledTasks.action.retry")}
          </Button>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col md:flex-row">
          <div className={selectedId || editorMode ? "hidden md:flex" : "flex"}>
            <ScheduledTaskList
              tasks={filteredTasks}
              selectedId={selectedId}
              query={query}
              filter={filter}
              locale={i18n.language}
              loading={loading}
              t={t}
              onQueryChange={setQuery}
              onFilterChange={setFilter}
              onSelect={(id) => {
                setEditorMode(null);
                setSelectedId(id);
              }}
              onCreate={() => startCreate()}
            />
          </div>
          <main
            className={
              selectedId || editorMode
                ? "min-h-0 flex-1"
                : "hidden min-h-0 flex-1 md:block"
            }
          >
            {editorMode ? (
              <ScheduledTaskEditor
                mode={editorMode}
                form={form}
                errors={formErrors}
                preview={preview}
                previewLoading={previewLoading}
                saving={busyAction === "save"}
                locale={i18n.language}
                t={t}
                onChange={setForm}
                onPreview={previewSchedule}
                onSave={() => void saveTask()}
                onCancel={() => setEditorMode(null)}
              />
            ) : loadingDetail && selectedId ? (
              <div className="flex h-full items-center justify-center text-sm text-slate-500">
                <LoaderCircle className="mr-2 h-5 w-5 animate-spin" />
                {t("scheduledTasks.details.loading")}
              </div>
            ) : selectedTask ? (
              <ScheduledTaskDetails
                task={selectedTask}
                runs={runs}
                loadingRuns={loadingRuns}
                busyAction={busyAction}
                locale={i18n.language}
                t={t}
                onBack={() => setSelectedId(null)}
                onClose={() => setSelectedId(null)}
                onEdit={startEdit}
                onToggleEnabled={() => void toggleEnabled()}
                onRun={() => void runNow()}
                onDelete={() => void removeTask()}
                onOpenRun={openRun}
              />
            ) : (
              <EmptyWorkbench t={t} onCreate={startCreate} />
            )}
          </main>
        </div>
      )}
    </div>
  );
}

function EmptyWorkbench({
  t,
  onCreate,
}: {
  t: TFunction<"workspace">;
  onCreate: (preset?: "daily" | "weekly" | "monitor") => void;
}) {
  return (
    <div className="flex h-full min-h-[420px] items-center justify-center overflow-y-auto px-6 py-10">
      <div className="w-full max-w-xl text-center">
        <CalendarClock className="mx-auto h-10 w-10 text-slate-400" />
        <h2 className="mt-4 text-lg font-semibold text-slate-950">
          {t("scheduledTasks.empty.workbenchTitle")}
        </h2>
        <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-600">
          {t("scheduledTasks.empty.workbenchDescription")}
        </p>
        <div className="mt-7 divide-y divide-slate-200 border-y border-slate-200 text-left">
          {(["daily", "weekly", "monitor"] as const).map((preset) => (
            <button
              key={preset}
              type="button"
              className="flex w-full items-center justify-between gap-4 px-2 py-3 text-sm hover:bg-slate-100"
              onClick={() => onCreate(preset)}
            >
              <span>
                <span className="block font-medium text-slate-900">
                  {t(`scheduledTasks.template.${preset}.title`)}
                </span>
                <span className="mt-0.5 block text-xs text-slate-500">
                  {t(`scheduledTasks.template.${preset}.summary`)}
                </span>
              </span>
              <Plus className="h-4 w-4 shrink-0 text-slate-500" />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function hasActiveRun(
  task: ScheduledTask | null,
  runs: ScheduledTaskRunSummary[],
): boolean {
  return [task?.lastRunSummary, ...runs].some(
    (run) => run?.status === "queued" || run?.status === "running",
  );
}
