import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  capabilityDraftsApi,
  type WorkspaceRegisteredSkillRecord,
} from "@/lib/api/capabilityDrafts";
import { listWorkspaceSkillBindings } from "@/lib/api/agentRuntime/inventoryClient";
import type { AgentRuntimeCompletionAuditSummary } from "@/lib/api/agentRuntime/evidenceTypes";
import type { AgentRuntimeWorkspaceSkillBinding } from "@/lib/api/agentRuntime/toolInventoryTypes";
import {
  scheduledTasksApi,
  type ScheduledTask,
} from "@/lib/api/scheduledTasks";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { isWorkspaceSkillScheduledTaskForDirectory } from "../workspaceSkillAgentAutomationDraft";
import { WorkspaceRegisteredSkillCard } from "./WorkspaceRegisteredSkillCard";

interface WorkspaceRegisteredSkillsPanelProps {
  workspaceRoot?: string | null;
  projectPending?: boolean;
  projectError?: string | null;
  refreshSignal?: number;
  workspaceId?: string | null;
  onEnableRuntime?: (binding: AgentRuntimeWorkspaceSkillBinding) => void;
  onCreateManagedAutomationDraft?: (
    binding: AgentRuntimeWorkspaceSkillBinding,
  ) => void;
  completionAuditSummariesByDirectory?: Record<
    string,
    AgentRuntimeCompletionAuditSummary | undefined
  >;
  hideWhenEmpty?: boolean;
  className?: string;
}

function sortRegisteredSkills(
  skills: WorkspaceRegisteredSkillRecord[],
): WorkspaceRegisteredSkillRecord[] {
  return [...skills].sort((left, right) =>
    right.registration.registeredAt.localeCompare(
      left.registration.registeredAt,
    ),
  );
}

async function loadWorkspaceRegisteredState(workspaceRoot: string) {
  const [nextSkills, bindingSnapshot, scheduledTasks] = await Promise.all([
    capabilityDraftsApi.listRegisteredSkills({ workspaceRoot }),
    listWorkspaceSkillBindings({
      workspaceRoot,
      caller: "assistant",
      workbench: true,
    }),
    scheduledTasksApi
      .listDetailed({ limit: 200 })
      .catch(() => [] as ScheduledTask[]),
  ]);

  return {
    skills: nextSkills,
    bindings: Array.isArray(bindingSnapshot.bindings)
      ? bindingSnapshot.bindings
      : [],
    scheduledTasks,
  };
}

export function WorkspaceRegisteredSkillsPanel({
  workspaceRoot,
  projectPending = false,
  projectError,
  refreshSignal = 0,
  workspaceId,
  onEnableRuntime,
  onCreateManagedAutomationDraft,
  completionAuditSummariesByDirectory,
  hideWhenEmpty = false,
  className,
}: WorkspaceRegisteredSkillsPanelProps) {
  const { t } = useTranslation("agent");
  const [skills, setSkills] = useState<WorkspaceRegisteredSkillRecord[]>([]);
  const [bindings, setBindings] = useState<AgentRuntimeWorkspaceSkillBinding[]>(
    [],
  );
  const [scheduledTasks, setScheduledTasks] = useState<ScheduledTask[]>([]);
  const [managedAutomationUpdatingJobId, setManagedAutomationUpdatingJobId] =
    useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const normalizedWorkspaceRoot = workspaceRoot?.trim() || null;

  const loadRegisteredSkills = useCallback(async () => {
    if (!normalizedWorkspaceRoot) {
      setSkills([]);
      setBindings([]);
      setScheduledTasks([]);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const nextState = await loadWorkspaceRegisteredState(
        normalizedWorkspaceRoot,
      );
      setSkills(nextState.skills);
      setBindings(nextState.bindings);
      setScheduledTasks(nextState.scheduledTasks);
    } catch (loadError) {
      setSkills([]);
      setBindings([]);
      setScheduledTasks([]);
      setError(String(loadError));
    } finally {
      setLoading(false);
    }
  }, [normalizedWorkspaceRoot]);

  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      if (!normalizedWorkspaceRoot) {
        setSkills([]);
        setBindings([]);
        setScheduledTasks([]);
        setError(null);
        return;
      }

      setLoading(true);
      setError(null);
      try {
        const nextState = await loadWorkspaceRegisteredState(
          normalizedWorkspaceRoot,
        );
        if (!cancelled) {
          setSkills(nextState.skills);
          setBindings(nextState.bindings);
          setScheduledTasks(nextState.scheduledTasks);
        }
      } catch (loadError) {
        if (!cancelled) {
          setSkills([]);
          setBindings([]);
          setScheduledTasks([]);
          setError(String(loadError));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void run();

    return () => {
      cancelled = true;
    };
  }, [normalizedWorkspaceRoot, refreshSignal]);

  const visibleSkills = useMemo(
    () => sortRegisteredSkills(skills).slice(0, 4),
    [skills],
  );
  const bindingByDirectory = useMemo(() => {
    const next = new Map<string, AgentRuntimeWorkspaceSkillBinding>();
    bindings.forEach((binding) => {
      if (binding.directory) {
        next.set(binding.directory, binding);
      }
    });
    return next;
  }, [bindings]);
  const managedScheduledTasksByDirectory = useMemo(() => {
    const next = new Map<string, ScheduledTask[]>();
    for (const skill of skills) {
      next.set(
        skill.directory,
        scheduledTasks.filter(
          (task) =>
            (!workspaceId || task.execution.projectId === workspaceId) &&
            isWorkspaceSkillScheduledTaskForDirectory(task, skill.directory),
        ),
      );
    }
    return next;
  }, [scheduledTasks, skills, workspaceId]);
  const handleToggleManagedScheduledTask = useCallback(
    async (task: ScheduledTask, enabled: boolean) => {
      setManagedAutomationUpdatingJobId(task.id);
      setError(null);
      try {
        const updatedTask = await scheduledTasksApi.setEnabled(
          task.id,
          enabled,
        );
        setScheduledTasks((previousTasks) =>
          previousTasks.map((item) =>
            item.id === updatedTask.id ? updatedTask : item,
          ),
        );
      } catch (toggleError) {
        setError(String(toggleError));
      } finally {
        setManagedAutomationUpdatingJobId(null);
      }
    },
    [],
  );
  const effectiveError = projectError || error;
  const isBusy = projectPending || loading;
  if (
    hideWhenEmpty &&
    (!normalizedWorkspaceRoot ||
      isBusy ||
      (!effectiveError && visibleSkills.length === 0))
  ) {
    return null;
  }

  return (
    <section
      className={cn(
        "rounded-[28px] border border-sky-200/80 bg-white p-5 shadow-sm shadow-sky-950/5",
        className,
      )}
      data-testid="workspace-registered-skills-panel"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-sky-200 bg-sky-50 px-2.5 py-1 text-[11px] font-medium text-sky-700">
              {t("capabilityDraft.registeredPanel.badge", "已保存")}
            </span>
            <h2 className="text-[15px] font-semibold text-slate-900">
              {t("capabilityDraft.registeredPanel.title", "已保存技能")}
            </h2>
          </div>
          <p className="text-[11px] leading-5 text-slate-500">
            {t(
              "capabilityDraft.registeredPanel.description",
              "这里是已经检查并保存到当前项目的技能。可以先试用，也可以设置定时运行。",
            )}
          </p>
        </div>
        {normalizedWorkspaceRoot ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="rounded-2xl px-3 text-slate-600 hover:bg-slate-50 hover:text-slate-900"
            onClick={() => void loadRegisteredSkills()}
            disabled={isBusy}
            data-testid="workspace-registered-skills-refresh"
          >
            <RefreshCw
              className={cn("mr-1.5 h-3.5 w-3.5", isBusy && "animate-spin")}
            />
            {t("capabilityDraft.registeredPanel.action.refresh", "刷新")}
          </Button>
        ) : null}
      </div>

      {!normalizedWorkspaceRoot ? (
        <div className="mt-4 rounded-[22px] border border-dashed border-sky-200 bg-sky-50/60 px-4 py-5 text-sm leading-6 text-sky-800">
          {t(
            "capabilityDraft.registeredPanel.empty.missingProject",
            "选择或进入一个项目后，才能查看这个项目里保存的技能。",
          )}
        </div>
      ) : effectiveError ? (
        <div className="mt-4 rounded-[22px] border border-rose-200 bg-rose-50 px-4 py-5 text-sm leading-6 text-rose-700">
          <div>
            {t(
              "capabilityDraft.registeredPanel.empty.error",
              "已保存技能暂时没读到，请稍后重试。",
            )}
          </div>
          <details className="mt-2 text-[11px] text-rose-600">
            <summary className="cursor-pointer">
              {t(
                "capabilityDraft.registeredPanel.technicalDetails",
                "技术详情",
              )}
            </summary>
            <div className="mt-1 break-words">{effectiveError}</div>
          </details>
        </div>
      ) : isBusy ? (
        <div className="mt-4 rounded-[22px] border border-slate-200 bg-slate-50 px-4 py-5 text-sm leading-6 text-slate-500">
          {t(
            "capabilityDraft.registeredPanel.empty.loading",
            "正在读取已保存技能...",
          )}
        </div>
      ) : visibleSkills.length === 0 ? (
        <div className="mt-4 rounded-[22px] border border-dashed border-sky-200 bg-sky-50/60 px-4 py-5 text-sm leading-6 text-sky-800">
          {t(
            "capabilityDraft.registeredPanel.empty.noSkills",
            "当前项目还没有已保存技能。先在“正在制作”里检查并保存一个。",
          )}
        </div>
      ) : (
        <div className="mt-4 space-y-3">
          {visibleSkills.map((skill) => (
            <WorkspaceRegisteredSkillCard
              key={skill.key || skill.directory}
              skill={skill}
              binding={bindingByDirectory.get(skill.directory)}
              managedScheduledTasks={
                managedScheduledTasksByDirectory.get(skill.directory) ?? []
              }
              managedAutomationUpdatingJobId={managedAutomationUpdatingJobId}
              completionAuditSummary={
                completionAuditSummariesByDirectory?.[skill.directory]
              }
              onToggleManagedScheduledTask={handleToggleManagedScheduledTask}
              onEnableRuntime={onEnableRuntime}
              onCreateManagedAutomationDraft={onCreateManagedAutomationDraft}
            />
          ))}
        </div>
      )}
    </section>
  );
}
