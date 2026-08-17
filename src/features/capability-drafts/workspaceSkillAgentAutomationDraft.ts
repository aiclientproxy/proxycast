import type { ScheduledTaskFormState } from "@/components/scheduled-tasks/scheduledTaskViewModel";
import type { AgentRuntimeWorkspaceSkillBinding } from "@/lib/api/agentRuntime/toolInventoryTypes";
import type {
  ScheduledTask,
  ScheduledTaskSchedule,
} from "@/lib/api/scheduledTasks";

const DEFAULT_CRON_TIMEZONE = "Asia/Shanghai";

export interface WorkspaceSkillManagedAutomationPresentation {
  statusLabel: string;
  scheduleLabel: string;
  lastRunLabel: string;
  jobId?: string;
  jobName?: string;
  enabled?: boolean;
}

export interface WorkspaceSkillManagedAutomationPresentationCopy {
  lastRunNone?: string;
  lastRunValueNone?: string;
  notCreatedSchedule?: string;
  notCreatedStatus?: string;
  notRunStatus?: string;
  stateEnabled?: string;
  statePaused?: string;
  unknownSchedule?: string;
  formatEverySchedule?: (seconds: number) => string;
  formatAtSchedule?: (at: string) => string;
  formatCronSchedule?: (expr: string, timezone?: string | null) => string;
  formatLastRun?: (lastRun: string, error?: string | null) => string;
  formatSchedule?: (schedule: string, nextRun?: string | null) => string;
  formatStatus?: (state: string, lastStatus: string) => string;
}

export interface WorkspaceSkillManagedAutomationInitialValuesCopy {
  descriptionPausedByDefault?: string;
  descriptionSource?: string;
  formatDescriptionProvenance?: (
    sourceDraftId: string,
    sourceVerificationReportId: string,
  ) => string;
  formatDescriptionSkill?: (skillName: string) => string;
  formatName?: (displayName: string) => string;
  formatPromptIntro?: (displayName: string, skillName: string) => string;
  promptNeedsInput?: string;
  promptReadRunbook?: string;
  promptResultEvidence?: string;
}

function normalizeText(value?: string | null): string {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeStringArray(value?: string[] | null): string[] {
  return Array.isArray(value)
    ? value.map((item) => item.trim()).filter(Boolean)
    : [];
}

function resolveSourceDraftId(
  binding: AgentRuntimeWorkspaceSkillBinding,
): string {
  return normalizeText(
    binding.registration.source_draft_id ?? binding.registration.sourceDraftId,
  );
}

function resolveSourceVerificationReportId(
  binding: AgentRuntimeWorkspaceSkillBinding,
): string {
  return normalizeText(
    binding.registration.source_verification_report_id ??
      binding.registration.sourceVerificationReportId,
  );
}

function buildSkillName(binding: AgentRuntimeWorkspaceSkillBinding): string {
  return `project:${binding.directory}`;
}

function buildDisplayName(binding: AgentRuntimeWorkspaceSkillBinding): string {
  return normalizeText(binding.name) || binding.directory;
}

function buildPermissionSummary(
  binding: AgentRuntimeWorkspaceSkillBinding,
): string[] {
  return normalizeStringArray(
    binding.permission_summary ?? binding.registration.permission_summary,
  );
}

export function canBuildWorkspaceSkillAgentAutomationDraft(
  binding?: AgentRuntimeWorkspaceSkillBinding | null,
): binding is AgentRuntimeWorkspaceSkillBinding {
  if (!binding || binding.binding_status !== "ready_for_manual_enable") {
    return false;
  }
  return Boolean(
    normalizeText(binding.directory) &&
    normalizeText(binding.registered_skill_directory) &&
    resolveSourceDraftId(binding) &&
    resolveSourceVerificationReportId(binding),
  );
}

export function buildWorkspaceSkillAgentAutomationRequestMetadata(input: {
  binding: AgentRuntimeWorkspaceSkillBinding;
  workspaceRoot: string;
  copy?: WorkspaceSkillManagedAutomationInitialValuesCopy;
}): Record<string, unknown> | null {
  const { binding } = input;
  const workspaceRoot = normalizeText(input.workspaceRoot);
  if (!workspaceRoot || !canBuildWorkspaceSkillAgentAutomationDraft(binding)) {
    return null;
  }

  const skillName = buildSkillName(binding);
  const displayName = buildDisplayName(binding);
  const permissionSummary = buildPermissionSummary(binding);
  const sourceDraftId = resolveSourceDraftId(binding);
  const sourceVerificationReportId = resolveSourceVerificationReportId(binding);
  return {
    harness: {
      theme: "general",
      session_mode: "general_workbench",
      run_title: displayName,
      agent_envelope: {
        source: "skill_forge_p4_agent_envelope",
        state: "automation_draft",
        skill: skillName,
        directory: binding.directory,
        registered_skill_directory: binding.registered_skill_directory,
        source_draft_id: sourceDraftId,
        source_verification_report_id: sourceVerificationReportId,
        authorization_scope: "scheduled_run_session",
      },
      workspace_skill_runtime_enable: {
        source: "manual_session_enable",
        approval: "manual",
        workspace_root: workspaceRoot,
        bindings: [
          {
            directory: binding.directory,
            skill: skillName,
            registered_skill_directory: binding.registered_skill_directory,
            source_draft_id: sourceDraftId,
            source_verification_report_id: sourceVerificationReportId,
            permission_summary: permissionSummary,
          },
        ],
      },
    },
  };
}

function buildAutomationPrompt(
  binding: AgentRuntimeWorkspaceSkillBinding,
  copy?: WorkspaceSkillManagedAutomationInitialValuesCopy,
): string {
  const displayName = buildDisplayName(binding);
  const skillName = buildSkillName(binding);
  return [
    copy?.formatPromptIntro?.(displayName, skillName) ??
      `请按当前 Workspace Agent envelope 草案运行 Skill「${displayName}」（${skillName}）。`,
    copy?.promptReadRunbook ??
      "先读取 Skill 的 Runbook、权限说明和输入约束，再执行任务。",
    copy?.promptNeedsInput ??
      "如果执行缺少必要输入或外部写权限，请返回 needs_input / blocked 的原因，不要绕过确认。",
    copy?.promptResultEvidence ??
      "完成后输出结果摘要，并保留可进入 evidence pack 的产物与关键步骤。",
  ].join("\n");
}

export interface WorkspaceSkillScheduledTaskDraft {
  form: ScheduledTaskFormState;
  requestMetadata: Record<string, unknown>;
}

export function buildWorkspaceSkillAgentAutomationInitialValues(input: {
  binding: AgentRuntimeWorkspaceSkillBinding;
  workspaceRoot: string;
  workspaceId: string;
  copy?: WorkspaceSkillManagedAutomationInitialValuesCopy;
}): WorkspaceSkillScheduledTaskDraft | null {
  const workspaceId = normalizeText(input.workspaceId);
  const requestMetadata = buildWorkspaceSkillAgentAutomationRequestMetadata({
    binding: input.binding,
    workspaceRoot: input.workspaceRoot,
    copy: input.copy,
  });
  if (!workspaceId || !requestMetadata) {
    return null;
  }

  const displayName = buildDisplayName(input.binding);
  const copy = input.copy;

  return {
    form: {
      title:
        copy?.formatName?.(displayName) ?? `${displayName}｜Managed Agent 草案`,
      prompt: buildAutomationPrompt(input.binding, copy),
      enabled: false,
      scheduleType: "daily",
      intervalHours: 1,
      days: [],
      time: "09:00",
      timezone: DEFAULT_CRON_TIMEZONE,
      threadMode: "new_thread",
      sourceThreadId: "",
      projectId: workspaceId,
      cwd: "",
      modelId: "",
      reasoningEffort: "",
      notificationPolicy: "failures",
    },
    requestMetadata,
  };
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}

function readNestedRecord(
  source: Record<string, unknown> | undefined,
  key: string,
): Record<string, unknown> | undefined {
  return asRecord(source?.[key]);
}

function describeSchedule(
  schedule: ScheduledTaskSchedule,
  copy?: WorkspaceSkillManagedAutomationPresentationCopy,
): string {
  switch (schedule.type) {
    case "hourly":
      return `每 ${schedule.intervalHours} 小时`;
    case "daily":
      return `每天 ${schedule.time}`;
    case "weekdays":
      return `工作日 ${schedule.time}`;
    case "weekly":
      return `每周 ${schedule.days.join(",")} ${schedule.time}`;
    default:
      return copy?.unknownSchedule ?? "未知调度";
  }
}

export function isWorkspaceSkillScheduledTaskForDirectory(
  task: ScheduledTask,
  directory: string,
): boolean {
  const normalizedDirectory = normalizeText(directory);
  if (!normalizedDirectory) {
    return false;
  }

  const requestMetadata = asRecord(task.execution.requestMetadata);
  const harness = readNestedRecord(requestMetadata, "harness");
  const agentEnvelope =
    readNestedRecord(harness, "agent_envelope") ??
    readNestedRecord(harness, "agentEnvelope");
  const envelopeDirectory = normalizeText(
    agentEnvelope?.directory as string | undefined,
  );
  const envelopeSkill = normalizeText(
    agentEnvelope?.skill as string | undefined,
  );

  return (
    envelopeDirectory === normalizedDirectory ||
    envelopeSkill === `project:${normalizedDirectory}`
  );
}

export function buildWorkspaceSkillManagedAutomationPresentation(
  tasks: readonly ScheduledTask[],
  copy?: WorkspaceSkillManagedAutomationPresentationCopy,
): WorkspaceSkillManagedAutomationPresentation {
  const [task] = tasks;
  if (!task) {
    return {
      statusLabel: copy?.notCreatedStatus ?? "Managed Job：未创建",
      scheduleLabel:
        copy?.notCreatedSchedule ?? "Schedule：等待创建 automation job 草案。",
      lastRunLabel: copy?.lastRunNone ?? "最近运行：暂无",
    };
  }

  const stateLabel = task.enabled
    ? (copy?.stateEnabled ?? "已启用")
    : (copy?.statePaused ?? "草案暂停");
  const lastStatus =
    task.lastRunSummary?.status ?? copy?.notRunStatus ?? "尚未运行";
  const lastRun =
    task.lastRunSummary?.startedAt ?? copy?.lastRunValueNone ?? "暂无";
  const schedule = describeSchedule(task.schedule, copy);
  return {
    jobId: task.id,
    jobName: task.title,
    enabled: task.enabled,
    statusLabel:
      copy?.formatStatus?.(stateLabel, lastStatus) ??
      `Managed Job：${stateLabel} · ${lastStatus}`,
    scheduleLabel:
      copy?.formatSchedule?.(schedule, task.nextRunAt) ??
      `Schedule：${schedule}${task.nextRunAt ? ` · 下次 ${task.nextRunAt}` : ""}`,
    lastRunLabel:
      copy?.formatLastRun?.(lastRun, task.lastRunSummary?.error) ??
      `最近运行：${lastRun}${
        task.lastRunSummary?.error ? ` · ${task.lastRunSummary.error}` : ""
      }`,
  };
}
