import { useCallback, useMemo } from "react";
import { CheckCircle2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { WorkspaceRegisteredSkillRecord } from "@/lib/api/capabilityDrafts";
import type { AgentRuntimeCompletionAuditSummary } from "@/lib/api/agentRuntime/evidenceTypes";
import type { AgentRuntimeWorkspaceSkillBinding } from "@/lib/api/agentRuntime/toolInventoryTypes";
import type { ScheduledTask } from "@/lib/api/scheduledTasks";
import { WorkspaceRegisteredSkillRegistrationDetails } from "./WorkspaceRegisteredSkillRegistrationDetails";
import { Button } from "@/components/ui/button";
import { formatNumber } from "@/i18n/format";
import { cn } from "@/lib/utils";
import {
  buildAgentEnvelopeDraftPresentation,
  type AgentEnvelopeDraftCompletionAuditLabelParts,
  type AgentEnvelopeDraftPresentationCopy,
} from "../agentEnvelopeDraftPresentation";
import {
  buildWorkspaceSkillManagedAutomationPresentation,
  canBuildWorkspaceSkillAgentAutomationDraft,
  type WorkspaceSkillManagedAutomationPresentationCopy,
} from "../workspaceSkillAgentAutomationDraft";

function summarizePermissionForUser(
  skill: WorkspaceRegisteredSkillRecord,
  copy: {
    defaultReadonly: string;
    readonly: string;
    draftWrite: string;
    localCommand: string;
  },
) {
  if (skill.permissionSummary.length === 0) {
    return copy.defaultReadonly;
  }

  const labels = new Set<string>();
  for (const item of skill.permissionSummary) {
    const normalized = item.toLowerCase();
    if (normalized.includes("level 0") || item.includes("只读")) {
      labels.add(copy.readonly);
      continue;
    }
    if (
      normalized.includes("level 1") ||
      normalized.includes("draft") ||
      normalized.includes("write") ||
      item.includes("草案") ||
      item.includes("写入")
    ) {
      labels.add(copy.draftWrite);
      continue;
    }
    if (
      normalized.includes("cli") ||
      normalized.includes("command") ||
      item.includes("命令")
    ) {
      labels.add(copy.localCommand);
      continue;
    }
    labels.add(item.replace(/^Level\s+\d+\s*/i, "").trim() || item);
  }

  return Array.from(labels).slice(0, 2).join(" / ");
}

function getUserScheduleLabel(
  task: ScheduledTask | undefined,
  copy: {
    enabled: string;
    notCreated: string;
    paused: string;
  },
) {
  if (!task) {
    return copy.notCreated;
  }
  return task.enabled ? copy.enabled : copy.paused;
}

function getUserResultLabel(
  task: ScheduledTask | undefined,
  completionAuditSummary: AgentRuntimeCompletionAuditSummary | undefined,
  copy: {
    auditBlocked: string;
    auditPassed: string;
    noJob: string;
    noRun: string;
    readyToCheck: string;
    running: string;
  },
) {
  if (completionAuditSummary?.decision === "completed") {
    return copy.auditPassed;
  }
  if (completionAuditSummary) {
    return copy.auditBlocked;
  }
  if (!task) {
    return copy.noJob;
  }
  if (task.lastRunSummary?.status === "running") {
    return copy.running;
  }
  if (task.lastRunSummary?.status === "success") {
    return copy.readyToCheck;
  }
  return copy.noRun;
}

export function WorkspaceRegisteredSkillCard({
  skill,
  binding,
  managedScheduledTasks,
  managedAutomationUpdatingJobId,
  completionAuditSummary,
  onToggleManagedScheduledTask,
  onEnableRuntime,
  onCreateManagedAutomationDraft,
}: {
  skill: WorkspaceRegisteredSkillRecord;
  binding?: AgentRuntimeWorkspaceSkillBinding;
  managedScheduledTasks: ScheduledTask[];
  managedAutomationUpdatingJobId?: string | null;
  completionAuditSummary?: AgentRuntimeCompletionAuditSummary;
  onToggleManagedScheduledTask?: (
    task: ScheduledTask,
    enabled: boolean,
  ) => void;
  onEnableRuntime?: (binding: AgentRuntimeWorkspaceSkillBinding) => void;
  onCreateManagedAutomationDraft?: (
    binding: AgentRuntimeWorkspaceSkillBinding,
  ) => void;
}) {
  const { i18n, t } = useTranslation("agent");
  const locale = i18n.language;
  const buildCompletionAuditLabel = useCallback(
    (parts: AgentEnvelopeDraftCompletionAuditLabelParts) => {
      const controlledGetLabel =
        parts.controlledGetExecutedCount > 0 ||
        parts.controlledGetArtifactCount > 0
          ? t(
              "capabilityDraft.registeredPanel.agentEnvelope.evidence.controlledGet.executed",
              {
                defaultValue: "，受控 GET {{executed}}/{{artifacts}} executed",
                artifacts: formatNumber(parts.controlledGetArtifactCount, {
                  locale,
                }),
                executed: formatNumber(parts.controlledGetExecutedCount, {
                  locale,
                }),
              },
            )
          : parts.controlledGetRequired
            ? t(
                "capabilityDraft.registeredPanel.agentEnvelope.evidence.controlledGet.requiredMissing",
                "，受控 GET required 0/0 executed",
              )
            : "";
      const blockingLabel =
        parts.blockingReasons.length > 0
          ? t(
              "capabilityDraft.registeredPanel.agentEnvelope.evidence.blocking",
              {
                defaultValue: "，阻塞：{{reasons}}",
                reasons: parts.blockingReasons.slice(0, 2).join(" / "),
              },
            )
          : "";
      const suffix = parts.missingControlledGetRequirement
        ? t(
            "capabilityDraft.registeredPanel.agentEnvelope.evidence.suffix.missingControlledGet",
            "；缺受控 GET evidence，不能固化为 Agent",
          )
        : parts.decision === "completed"
          ? ""
          : t(
              "capabilityDraft.registeredPanel.agentEnvelope.evidence.suffix.notCompleted",
              "；未 completed，不能固化为 Agent",
            );

      return t(
        "capabilityDraft.registeredPanel.agentEnvelope.evidence.completionAudit",
        {
          defaultValue:
            "Evidence：completion audit {{decision}}，owner {{successfulOwnerRunCount}}/{{ownerRunCount}}，ToolCall {{toolCallCount}}，artifact {{artifactCount}}{{controlledGetLabel}}{{blockingLabel}}{{suffix}}。",
          artifactCount: formatNumber(parts.artifactCount, { locale }),
          blockingLabel,
          controlledGetLabel,
          decision: parts.decision,
          ownerRunCount: formatNumber(parts.ownerRunCount, { locale }),
          successfulOwnerRunCount: formatNumber(parts.successfulOwnerRunCount, {
            locale,
          }),
          suffix,
          toolCallCount: formatNumber(parts.workspaceSkillToolCallCount, {
            locale,
          }),
        },
      );
    },
    [locale, t],
  );
  const envelopeCopy = useMemo<AgentEnvelopeDraftPresentationCopy>(
    () => ({
      actionBlocked: t(
        "capabilityDraft.registeredPanel.agentEnvelope.action.blocked",
        "先处理问题",
      ),
      actionDraft: t(
        "capabilityDraft.registeredPanel.agentEnvelope.action.draft",
        "保存成项目助手",
      ),
      actionManualEnable: t(
        "capabilityDraft.registeredPanel.agentEnvelope.action.manualEnable",
        "先试用一次",
      ),
      agentCardPending: t(
        "capabilityDraft.registeredPanel.agentEnvelope.agentCard.pending",
        "项目助手：试用通过后再保存。",
      ),
      blockedReasonFallback: t(
        "capabilityDraft.registeredPanel.agentEnvelope.blockedReasonFallback",
        "当前还有问题需要处理",
      ),
      description: t(
        "capabilityDraft.registeredPanel.agentEnvelope.description",
        "试用结果没问题后，可以把它保存成当前项目里的助手。",
      ),
      discoveryPending: t(
        "capabilityDraft.registeredPanel.agentEnvelope.discovery.pending",
        "团队可见性：保存成项目助手后再开放。",
      ),
      evidenceMissing: t(
        "capabilityDraft.registeredPanel.agentEnvelope.evidence.missing",
        "最近结果：还没有试用结果。",
      ),
      evidenceSourceMetadataOnly: t(
        "capabilityDraft.registeredPanel.agentEnvelope.evidence.sourceMetadataOnly",
        "最近结果：已记录这次试用来源，等待检查。",
      ),
      formatAgentCardReady: (directory) =>
        t("capabilityDraft.registeredPanel.agentEnvelope.agentCard.ready", {
          defaultValue: "项目助手：已准备好保存（{{directory}}）。",
          directory,
        }),
      formatCompletionAuditEvidenceLabel: buildCompletionAuditLabel,
      formatCompletedEvidencePack: (packId) =>
        packId
          ? t(
              "capabilityDraft.registeredPanel.agentEnvelope.evidence.completedPack",
              {
                defaultValue: "最近结果：已通过检查（{{packId}}）。",
                packId,
              },
            )
          : t(
              "capabilityDraft.registeredPanel.agentEnvelope.evidence.completedPackFallback",
              "最近结果：已通过检查。",
            ),
      formatDiscoveryReady: (registeredSkillDirectory) =>
        t("capabilityDraft.registeredPanel.agentEnvelope.discovery.ready", {
          defaultValue: "团队可见性：当前项目成员可以使用这条技能。",
          directory: registeredSkillDirectory,
        }),
      formatMemoryWithReport: (reportId) =>
        t("capabilityDraft.registeredPanel.agentEnvelope.memory.withReport", {
          defaultValue: "记录：已保留检查结果，后续会继续积累使用反馈。",
          reportId,
        }),
      formatPendingEvidencePack: (packId) =>
        t(
          "capabilityDraft.registeredPanel.agentEnvelope.evidence.pendingPack",
          {
            defaultValue: "最近结果：已生成，等待检查后才能保存成项目助手。",
            packId,
          },
        ),
      formatPermissionWithSummary: (summary) =>
        t(
          "capabilityDraft.registeredPanel.agentEnvelope.permission.withSummary",
          {
            defaultValue: "权限：{{summary}}。",
            summary,
          },
        ),
      formatRunbook: (name) =>
        t("capabilityDraft.registeredPanel.agentEnvelope.runbook", {
          defaultValue: "使用方式：{{name}}",
          name,
        }),
      memoryPending: t(
        "capabilityDraft.registeredPanel.agentEnvelope.memory.pending",
        "记录：等待首次试用后记录偏好和修正。",
      ),
      permissionEmpty: t(
        "capabilityDraft.registeredPanel.agentEnvelope.permission.empty",
        "权限：默认需要手动确认。",
      ),
      schedule: t(
        "capabilityDraft.registeredPanel.agentEnvelope.schedule",
        "定时运行：尚未设置。",
      ),
      sharingPending: t(
        "capabilityDraft.registeredPanel.agentEnvelope.sharing.pending",
        "共享：先只对你可见。",
      ),
      sharingReady: t(
        "capabilityDraft.registeredPanel.agentEnvelope.sharing.ready",
        "共享：可在当前项目内共享。",
      ),
      statusLabels: {
        blocked: t(
          "capabilityDraft.registeredPanel.agentEnvelope.status.blocked",
          "需要处理",
        ),
        evidence_ready: t(
          "capabilityDraft.registeredPanel.agentEnvelope.status.evidenceReady",
          "可保存",
        ),
        manual_enable_required: t(
          "capabilityDraft.registeredPanel.agentEnvelope.status.manualEnableRequired",
          "待试用",
        ),
        source_metadata_ready: t(
          "capabilityDraft.registeredPanel.agentEnvelope.status.sourceMetadataReady",
          "待检查",
        ),
      },
      widgetPending: t(
        "capabilityDraft.registeredPanel.agentEnvelope.widget.pending",
        "结果：试用后展示最近状态和下一步。",
      ),
      widgetReady: t(
        "capabilityDraft.registeredPanel.agentEnvelope.widget.ready",
        "结果：展示最近状态、产物和下一步动作。",
      ),
    }),
    [buildCompletionAuditLabel, t],
  );
  const managedAutomationCopy =
    useMemo<WorkspaceSkillManagedAutomationPresentationCopy>(
      () => ({
        formatAtSchedule: (at) =>
          t("capabilityDraft.registeredPanel.managedJob.schedule.at", {
            defaultValue: "一次性 {{at}}",
            at,
          }),
        formatCronSchedule: (expr, timezone) =>
          t("capabilityDraft.registeredPanel.managedJob.schedule.cron", {
            defaultValue: "Cron {{expr}}{{timezone}}",
            expr,
            timezone: timezone
              ? t(
                  "capabilityDraft.registeredPanel.managedJob.schedule.timezone",
                  {
                    defaultValue: " · {{timezone}}",
                    timezone,
                  },
                )
              : "",
          }),
        formatEverySchedule: (seconds) =>
          t("capabilityDraft.registeredPanel.managedJob.schedule.every", {
            defaultValue: "每 {{seconds}} 秒",
            seconds: formatNumber(seconds, { locale }),
          }),
        formatLastRun: (lastRun, error) =>
          t("capabilityDraft.registeredPanel.managedJob.lastRun.withValue", {
            defaultValue: "最近运行：{{lastRun}}{{error}}",
            error: error
              ? t("capabilityDraft.registeredPanel.managedJob.lastRun.error", {
                  defaultValue: " · {{error}}",
                  error,
                })
              : "",
            lastRun,
          }),
        formatSchedule: (schedule, nextRun) =>
          t("capabilityDraft.registeredPanel.managedJob.schedule.withValue", {
            defaultValue: "Schedule：{{schedule}}{{nextRun}}",
            nextRun: nextRun
              ? t(
                  "capabilityDraft.registeredPanel.managedJob.schedule.nextRun",
                  {
                    defaultValue: " · 下次 {{nextRun}}",
                    nextRun,
                  },
                )
              : "",
            schedule,
          }),
        formatStatus: (state, lastStatus) =>
          t("capabilityDraft.registeredPanel.managedJob.status.withValue", {
            defaultValue: "Managed Job：{{state}} · {{lastStatus}}",
            lastStatus,
            state,
          }),
        lastRunNone: t(
          "capabilityDraft.registeredPanel.managedJob.lastRun.none",
          "最近运行：暂无",
        ),
        lastRunValueNone: t(
          "capabilityDraft.registeredPanel.managedJob.lastRun.valueNone",
          "暂无",
        ),
        notCreatedSchedule: t(
          "capabilityDraft.registeredPanel.managedJob.schedule.notCreated",
          "Schedule：等待创建 automation job 草案。",
        ),
        notCreatedStatus: t(
          "capabilityDraft.registeredPanel.managedJob.status.notCreated",
          "Managed Job：未创建",
        ),
        notRunStatus: t(
          "capabilityDraft.registeredPanel.managedJob.status.notRun",
          "尚未运行",
        ),
        stateEnabled: t(
          "capabilityDraft.registeredPanel.managedJob.state.enabled",
          "已启用",
        ),
        statePaused: t(
          "capabilityDraft.registeredPanel.managedJob.state.paused",
          "草案暂停",
        ),
        unknownSchedule: t(
          "capabilityDraft.registeredPanel.managedJob.schedule.unknown",
          "未知调度",
        ),
      }),
      [locale, t],
    );
  const bindingBlocked = binding?.binding_status === "blocked";
  const runtimeEnableReady =
    binding?.binding_status === "ready_for_manual_enable";
  const envelopeDraft = buildAgentEnvelopeDraftPresentation({
    skill,
    binding,
    completionAuditSummary,
    copy: envelopeCopy,
  });
  const canCreateManagedAutomationDraft =
    canBuildWorkspaceSkillAgentAutomationDraft(binding);
  const canCreateAgentEnvelopeDraft =
    envelopeDraft.actionEnabled &&
    canCreateManagedAutomationDraft &&
    Boolean(onCreateManagedAutomationDraft);
  const managedAutomationPresentation =
    buildWorkspaceSkillManagedAutomationPresentation(
      managedScheduledTasks,
      managedAutomationCopy,
    );
  const [managedScheduledTask] = managedScheduledTasks;
  const userPermissionSummary = summarizePermissionForUser(skill, {
    defaultReadonly: t(
      "capabilityDraft.registeredPanel.user.permission.defaultReadonly",
      "默认只读取信息",
    ),
    draftWrite: t(
      "capabilityDraft.registeredPanel.user.permission.draftWrite",
      "可写入草案",
    ),
    localCommand: t(
      "capabilityDraft.registeredPanel.user.permission.localCommand",
      "可运行本地命令",
    ),
    readonly: t(
      "capabilityDraft.registeredPanel.user.permission.readonly",
      "只读取信息",
    ),
  });
  const userScheduleLabel = getUserScheduleLabel(managedScheduledTask, {
    enabled: t(
      "capabilityDraft.registeredPanel.user.schedule.enabled",
      "已开启",
    ),
    notCreated: t(
      "capabilityDraft.registeredPanel.user.schedule.notCreated",
      "未设置",
    ),
    paused: t("capabilityDraft.registeredPanel.user.schedule.paused", "已暂停"),
  });
  const userResultLabel = getUserResultLabel(
    managedScheduledTask,
    completionAuditSummary,
    {
      auditBlocked: t(
        "capabilityDraft.registeredPanel.user.result.auditBlocked",
        "最近结果需要处理",
      ),
      auditPassed: t(
        "capabilityDraft.registeredPanel.user.result.auditPassed",
        "最近结果已通过检查",
      ),
      noJob: t(
        "capabilityDraft.registeredPanel.user.result.noJob",
        "还没有定时运行",
      ),
      noRun: t(
        "capabilityDraft.registeredPanel.user.result.noRun",
        "还没有运行记录",
      ),
      readyToCheck: t(
        "capabilityDraft.registeredPanel.user.result.readyToCheck",
        "可以检查最近结果",
      ),
      running: t(
        "capabilityDraft.registeredPanel.user.result.running",
        "正在运行",
      ),
    },
  );
  const managedAutomationUpdating =
    managedScheduledTask?.id === managedAutomationUpdatingJobId;

  return (
    <article className="rounded-[22px] border border-slate-200 bg-slate-50 px-4 py-3.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="inline-flex items-center rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[11px] font-medium text-emerald-700">
          <CheckCircle2 className="mr-1 h-3 w-3" />
          {t("capabilityDraft.registeredPanel.card.badge.registered", "已注册")}
        </span>
        <span
          className={cn(
            "rounded-full border bg-white px-2.5 py-1 text-[11px] font-medium",
            bindingBlocked
              ? "border-amber-200 text-amber-700"
              : "border-sky-200 text-sky-700",
          )}
        >
          {bindingBlocked
            ? t(
                "capabilityDraft.registeredPanel.card.binding.blocked",
                "需要处理",
              )
            : t(
                "capabilityDraft.registeredPanel.card.binding.candidate",
                "可试用",
              )}
        </span>
      </div>
      <div className="mt-2.5 space-y-1.5">
        <h3 className="text-sm font-semibold text-slate-900">
          {skill.name || skill.directory}
        </h3>
        <p className="line-clamp-2 text-[12px] leading-5 text-slate-600">
          {skill.description ||
            t(
              "capabilityDraft.registeredPanel.card.descriptionFallback",
              "已注册为当前 Workspace 的本地 Skill 包。",
            )}
        </p>
      </div>
      <div className="mt-3 grid gap-2 sm:grid-cols-3">
        <div className="rounded-2xl border border-emerald-100 bg-white px-3 py-2">
          <div className="text-[11px] text-slate-500">
            {t("capabilityDraft.registeredPanel.user.step.saved", "已保存")}
          </div>
          <div className="mt-1 text-[12px] font-semibold text-emerald-700">
            {t(
              "capabilityDraft.registeredPanel.user.step.savedValue",
              "可在当前项目使用",
            )}
          </div>
        </div>
        <div className="rounded-2xl border border-sky-100 bg-white px-3 py-2">
          <div className="text-[11px] text-slate-500">
            {t("capabilityDraft.registeredPanel.user.step.permission", "权限")}
          </div>
          <div className="mt-1 text-[12px] font-semibold text-slate-800">
            {userPermissionSummary}
          </div>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white px-3 py-2">
          <div className="text-[11px] text-slate-500">
            {t(
              "capabilityDraft.registeredPanel.user.step.schedule",
              "定时运行",
            )}
          </div>
          <div className="mt-1 text-[12px] font-semibold text-slate-800">
            {userScheduleLabel}
          </div>
        </div>
      </div>
      <p className="mt-3 rounded-2xl border border-sky-100 bg-sky-50 px-3 py-2 text-[12px] leading-5 text-sky-800">
        {t(
          "capabilityDraft.registeredPanel.user.nextStep",
          "建议先试用一次。确认结果没问题后，再设置定时运行或保存成项目助手。",
        )}
      </p>
      <WorkspaceRegisteredSkillRegistrationDetails
        skill={skill}
        binding={binding}
      />
      <div className="mt-3 rounded-2xl border border-cyan-100 bg-white px-3 py-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-[11px] font-semibold text-cyan-800">
            {t(
              "capabilityDraft.registeredPanel.agentEnvelope.title",
              "项目助手",
            )}
          </span>
          <span className="rounded-full border border-cyan-200 bg-cyan-50 px-2 py-0.5 text-[10px] font-medium text-cyan-700">
            {envelopeDraft.statusLabel}
          </span>
        </div>
        <p className="mt-1.5 text-[11px] leading-5 text-slate-600">
          {envelopeDraft.description}
        </p>
        <div className="mt-3 grid gap-2 sm:grid-cols-3">
          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2">
            <div className="text-[11px] text-slate-500">
              {t("capabilityDraft.registeredPanel.user.assistant.try", "试用")}
            </div>
            <div className="mt-1 text-[12px] font-semibold text-slate-800">
              {runtimeEnableReady
                ? t(
                    "capabilityDraft.registeredPanel.user.assistant.tryReady",
                    "可以试用一次",
                  )
                : t(
                    "capabilityDraft.registeredPanel.user.assistant.tryBlocked",
                    "暂不可试用",
                  )}
            </div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2">
            <div className="text-[11px] text-slate-500">
              {t(
                "capabilityDraft.registeredPanel.user.assistant.schedule",
                "定时运行",
              )}
            </div>
            <div className="mt-1 text-[12px] font-semibold text-slate-800">
              {userScheduleLabel}
            </div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2">
            <div className="text-[11px] text-slate-500">
              {t(
                "capabilityDraft.registeredPanel.user.assistant.result",
                "最近结果",
              )}
            </div>
            <div className="mt-1 text-[12px] font-semibold text-slate-800">
              {userResultLabel}
            </div>
          </div>
        </div>
        <details className="mt-3 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] leading-5 text-slate-500">
          <summary className="cursor-pointer select-none text-[12px] font-medium text-slate-700">
            {t(
              "capabilityDraft.registeredPanel.agentEnvelope.details",
              "项目助手详情",
            )}
          </summary>
          <div className="mt-2 grid gap-1">
            <span>{envelopeDraft.agentCardLabel}</span>
            <span>{envelopeDraft.sharingLabel}</span>
            <span>{envelopeDraft.sharingDiscoveryLabel}</span>
            <span>{envelopeDraft.runbookLabel}</span>
            <span>{envelopeDraft.memoryLabel}</span>
            <span>{envelopeDraft.widgetLabel}</span>
            <span>{envelopeDraft.permissionLabel}</span>
            <span>{envelopeDraft.scheduleLabel}</span>
            <span>{envelopeDraft.evidenceLabel}</span>
            <span>{managedAutomationPresentation.statusLabel}</span>
            <span>{managedAutomationPresentation.scheduleLabel}</span>
            <span>{managedAutomationPresentation.lastRunLabel}</span>
          </div>
        </details>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {onEnableRuntime && binding ? (
            <Button
              type="button"
              size="sm"
              className="h-8 rounded-2xl bg-slate-900 px-3 text-[12px] text-white hover:bg-slate-800"
              disabled={!runtimeEnableReady}
              onClick={() => onEnableRuntime(binding)}
              data-testid="workspace-registered-skill-enable-runtime"
            >
              {t(
                "capabilityDraft.registeredPanel.action.enableRuntime",
                "试用一次",
              )}
            </Button>
          ) : null}
          {onCreateManagedAutomationDraft && binding ? (
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-8 rounded-2xl border-slate-200 bg-white px-3 text-[12px] text-slate-700 hover:bg-slate-50"
              disabled={!canCreateManagedAutomationDraft}
              onClick={() => onCreateManagedAutomationDraft(binding)}
              data-testid="workspace-registered-agent-managed-automation"
            >
              {t(
                "capabilityDraft.registeredPanel.action.createManagedJobDraft",
                "设置定时运行",
              )}
            </Button>
          ) : null}
          {managedScheduledTask && onToggleManagedScheduledTask ? (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              className="h-8 rounded-2xl px-3 text-[12px] text-slate-600 hover:bg-slate-50"
              disabled={managedAutomationUpdating}
              onClick={() =>
                onToggleManagedScheduledTask(
                  managedScheduledTask,
                  !managedScheduledTask.enabled,
                )
              }
              data-testid="workspace-registered-agent-managed-automation-toggle"
            >
              {managedScheduledTask.enabled
                ? t(
                    "capabilityDraft.registeredPanel.action.pauseManagedJob",
                    "暂停定时运行",
                  )
                : t(
                    "capabilityDraft.registeredPanel.action.resumeManagedJob",
                    "开启定时运行",
                  )}
            </Button>
          ) : null}
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className="h-8 rounded-2xl px-3 text-[12px] text-cyan-700 hover:bg-cyan-50 disabled:text-slate-400"
            disabled={!canCreateAgentEnvelopeDraft}
            onClick={() => {
              if (binding && canCreateAgentEnvelopeDraft) {
                onCreateManagedAutomationDraft?.(binding);
              }
            }}
            data-testid="workspace-registered-agent-envelope-action"
          >
            {envelopeDraft.actionLabel}
          </Button>
        </div>
      </div>
    </article>
  );
}
