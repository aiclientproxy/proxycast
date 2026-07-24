import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  ArrowRight,
  Bell,
  Bot,
  FileText,
  History,
  Pencil,
  Play,
  Plus,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { WorkbenchInfoTip } from "@/components/media/WorkbenchInfoTip";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  AutomationHealthResult,
  AutomationJobRecord,
  AutomationSchedulerConfig,
  AutomationStatus,
  createAutomationJob,
  deleteAutomationJob,
  getAutomationHealth,
  getAutomationJobs,
  getAutomationRunHistory,
  getAutomationSchedulerConfig,
  getAutomationStatus,
  runAutomationJobNow,
  updateAutomationJob,
  updateAutomationSchedulerConfig,
} from "@/lib/api/automation";
import type { Project } from "@/lib/api/project";
import { listProjects } from "@/lib/api/project";
import type { AgentRun } from "@/lib/api/executionRun";
import { LatestRunStatusBadge } from "@/components/execution/LatestRunStatusBadge";
import { AutomationHealthPanel } from "./AutomationHealthPanel";
import { AutomationJobDetailsDialog } from "./AutomationJobDetailsDialog";
import { AutomationJobFocusStrip } from "./AutomationJobFocusStrip";
import { AutomationOverviewFocusCard } from "./AutomationOverviewFocusCard";
import {
  AutomationJobDialog,
  AutomationJobDialogSubmit,
  type AutomationJobDialogInitialValues,
} from "./AutomationJobDialog";
import type { AutomationThreadLineage } from "./automationThreadLineage";
import {
  type AutomationServiceSkillContextCopy,
  resolveServiceSkillAutomationContext,
  type AutomationServiceSkillContext,
} from "./serviceSkillContext";
import {
  recordAutomationJobMutationAgentUiProjection,
  recordAutomationJobsRefreshAgentUiProjection,
  recordAutomationRunHistoryAgentUiProjection,
  recordAutomationStatusRefreshAgentUiProjection,
} from "./automationAgentUiProjection";
import {
  type AutomationPresentationCopy,
  describeSchedule,
  describeServiceSkillSlotPreview,
  describeServiceSkillTaskLine,
  executionModeLabel,
  formatTime,
  isLegacyBrowserAutomation,
  statusDetailPrefix,
  statusDetailToneClass,
  statusLabel,
  statusVariant,
} from "./automationPresentation";
import type { AutomationAccessModeCopy } from "./automationAccessMode";
import { resolveLegacySceneAppAutomationContext } from "./legacySceneAppContext";
import type { AutomationWorkspaceTab } from "@/types/page";

const AUTOMATION_CORE_LOAD_TIMEOUT_MS = 8000;
const AUTOMATION_AUXILIARY_LOAD_TIMEOUT_MS = 5000;

type SettingsTranslate = (
  key: string,
  values?: Record<string, unknown>,
) => string;

function buildAutomationServiceSkillContextCopy(
  t: SettingsTranslate,
): AutomationServiceSkillContextCopy {
  return {
    defaultTitle: t("settings.automation.tasks.list.badge.serviceSkill"),
    unknownLabel: t("settings.automation.serviceSkill.unknown"),
    runnerInstant: t("settings.automation.serviceSkill.runner.instant"),
    runnerScheduled: t("settings.automation.serviceSkill.runner.scheduled"),
    runnerManaged: t("settings.automation.serviceSkill.runner.managed"),
    executionLocationClient: t(
      "settings.automation.serviceSkill.executionLocation.client",
    ),
    sourceCloudCatalog: t(
      "settings.automation.serviceSkill.source.cloudCatalog",
    ),
    sourceLocalCustom: t("settings.automation.serviceSkill.source.localCustom"),
    slotFallbackLabel: (index) =>
      t("settings.automation.serviceSkill.slotFallback", {
        index: index + 1,
      }),
  };
}

function buildAutomationPresentationCopy(
  t: SettingsTranslate,
  serviceSkillContextCopy: AutomationServiceSkillContextCopy,
): AutomationPresentationCopy {
  const accessModeCopy: AutomationAccessModeCopy = {
    readOnly: t("settings.automation.jobDialog.accessMode.readOnly"),
    current: t("settings.automation.jobDialog.accessMode.current"),
    fullAccess: t("settings.automation.jobDialog.accessMode.fullAccess"),
    policyReadOnly: t(
      "settings.automation.jobDialog.accessMode.policy.readOnly",
    ),
    policyCurrent: t("settings.automation.jobDialog.accessMode.policy.current"),
    policyFullAccess: t(
      "settings.automation.jobDialog.accessMode.policy.fullAccess",
    ),
  };

  return {
    legacyBrowserAutomationNotice: t(
      "settings.automation.details.legacy.message",
    ),
    legacyBrowserAutomationStatus: t(
      "settings.automation.details.status.offline",
    ),
    scheduleHours: (count) =>
      t("settings.automation.details.schedule.hours", {
        count,
      }),
    scheduleMinutes: (count) =>
      t("settings.automation.details.schedule.minutes", {
        count,
      }),
    scheduleSeconds: (count) =>
      t("settings.automation.details.schedule.seconds", {
        count,
      }),
    scheduleCron: (expr) =>
      t("settings.automation.details.schedule.cron", {
        expr,
      }),
    scheduleAt: (time) =>
      t("settings.automation.details.schedule.at", {
        time,
      }),
    executionModeIntelligent: t(
      "settings.automation.jobDialog.executionMode.intelligent",
    ),
    executionModeSkill: t("settings.automation.jobDialog.executionMode.skill"),
    executionModeLogOnly: t(
      "settings.automation.jobDialog.executionMode.logOnly",
    ),
    payloadBrowserSession: t(
      "settings.automation.details.payload.browserSession",
    ),
    payloadAgentTurn: t("settings.automation.details.payload.agentTurn"),
    legacyPayloadProfile: (profile) =>
      t("settings.automation.details.legacy.payload.profile", {
        profile,
      }),
    legacyPayloadEnvironment: (environment) =>
      t("settings.automation.details.legacy.payload.environment", {
        environment,
      }),
    legacyPayloadUrl: (url) =>
      t("settings.automation.details.legacy.payload.url", {
        url,
      }),
    legacyPayloadTargetId: (targetId) =>
      t("settings.automation.details.legacy.payload.targetId", {
        targetId,
      }),
    legacyPayloadWindow: (status) =>
      t("settings.automation.details.legacy.payload.window", {
        status,
      }),
    legacyPayloadWindowOpen: t(
      "settings.automation.details.legacy.payload.windowOpen",
    ),
    legacyPayloadWindowClosed: t(
      "settings.automation.details.legacy.payload.windowClosed",
    ),
    legacyPayloadStreamMode: (streamMode) =>
      t("settings.automation.details.legacy.payload.streamMode", {
        streamMode,
      }),
    statusQueued: t("settings.automation.details.status.queued"),
    statusSuccess: t("settings.automation.details.status.success"),
    statusRunning: t("settings.automation.details.status.running"),
    statusWaitingForHuman: t(
      "settings.automation.details.status.waitingForHuman",
    ),
    statusHumanControlling: t(
      "settings.automation.details.status.humanControlling",
    ),
    statusAgentResuming: t("settings.automation.details.status.agentResuming"),
    statusError: t("settings.automation.details.status.error"),
    statusTimeout: t("settings.automation.details.status.timeout"),
    statusPending: t("settings.automation.details.status.pending"),
    statusDetailBlocking: t(
      "settings.automation.details.statusDetail.blocking",
    ),
    statusDetailResume: t("settings.automation.details.statusDetail.resume"),
    statusDetailLastError: t(
      "settings.automation.details.statusDetail.lastError",
    ),
    statusDetailRunning: t("settings.automation.details.statusDetail.running"),
    deliveryModeAnnounce: t(
      "settings.automation.details.delivery.mode.announce",
    ),
    deliveryModeNone: t("settings.automation.jobDialog.delivery.mode.none"),
    deliveryChannelLocalFile: t(
      "settings.automation.details.delivery.channel.localFile",
    ),
    outputSchemaJson: t("settings.automation.details.delivery.schema.json"),
    outputSchemaTable: t("settings.automation.details.delivery.schema.table"),
    outputSchemaCsv: t("settings.automation.details.delivery.schema.csv"),
    outputSchemaLinks: t("settings.automation.details.delivery.schema.links"),
    outputSchemaText: t("settings.automation.details.delivery.schema.text"),
    outputFormatJson: t("settings.automation.details.delivery.format.json"),
    outputFormatText: t("settings.automation.details.delivery.format.text"),
    serviceSkillTaskLine: (title) =>
      t("settings.automation.details.serviceSkill.taskLine", {
        title,
      }),
    serviceSkillMoreItems: (count) =>
      t("settings.automation.details.serviceSkill.moreItems", {
        count,
      }),
    serviceSkillContextCopy,
    accessModeCopy,
  };
}

function isAutomationJobAtRisk(
  job: AutomationJobRecord,
  riskyJobMessageMap: Map<string, string>,
): boolean {
  if (riskyJobMessageMap.has(job.id) || Boolean(job.auto_disabled_until)) {
    return true;
  }

  return [
    "error",
    "timeout",
    "waiting_for_human",
    "human_controlling",
  ].includes(job.last_status ?? "");
}

function resolveAutomationJobSortTime(job: AutomationJobRecord): number {
  const candidates = [
    job.last_run_at,
    job.last_finished_at,
    job.updated_at,
    job.created_at,
    job.next_run_at,
  ];

  for (const value of candidates) {
    if (!value) {
      continue;
    }

    const timestamp = Date.parse(value);
    if (Number.isFinite(timestamp)) {
      return timestamp;
    }
  }

  return 0;
}

type AutomationWorkspaceTemplate = {
  id: string;
  tag: string;
  name: string;
  description: string;
  detail: string;
  actionLabel: string;
  icon: typeof Bot;
  initialValues?: AutomationJobDialogInitialValues | null;
};

function createWorkspaceTemplates(
  t: SettingsTranslate,
): AutomationWorkspaceTemplate[] {
  return [
    {
      id: "daily-brief",
      tag: t("settings.automation.tasks.template.dailyBrief.tag"),
      name: t("settings.automation.tasks.template.dailyBrief.name"),
      description: t(
        "settings.automation.tasks.template.dailyBrief.description",
      ),
      detail: t("settings.automation.tasks.template.dailyBrief.detail"),
      actionLabel: t("settings.automation.tasks.template.dailyBrief.action"),
      icon: Bell,
      initialValues: {
        name: t("settings.automation.tasks.template.dailyBrief.initial.name"),
        description: t(
          "settings.automation.tasks.template.dailyBrief.initial.description",
        ),
        payload_kind: "agent_turn",
        schedule_kind: "cron",
        cron_expr: "0 9 * * *",
        cron_tz: "Asia/Shanghai",
        prompt: t(
          "settings.automation.tasks.template.dailyBrief.initial.prompt",
        ),
        delivery_mode: "none",
      },
    },
    {
      id: "structured-delivery",
      tag: t("settings.automation.tasks.template.structuredDelivery.tag"),
      name: t("settings.automation.tasks.template.structuredDelivery.name"),
      description: t(
        "settings.automation.tasks.template.structuredDelivery.description",
      ),
      detail: t("settings.automation.tasks.template.structuredDelivery.detail"),
      actionLabel: t(
        "settings.automation.tasks.template.structuredDelivery.action",
      ),
      icon: FileText,
      initialValues: {
        name: t(
          "settings.automation.tasks.template.structuredDelivery.initial.name",
        ),
        description: t(
          "settings.automation.tasks.template.structuredDelivery.initial.description",
        ),
        payload_kind: "agent_turn",
        schedule_kind: "every",
        every_secs: "3600",
        prompt: t(
          "settings.automation.tasks.template.structuredDelivery.initial.prompt",
        ),
        delivery_mode: "announce",
        delivery_channel: "local_file",
        delivery_output_schema: "json",
        delivery_output_format: "json",
        best_effort: true,
      },
    },
    {
      id: "blank",
      tag: t("settings.automation.tasks.template.blank.tag"),
      name: t("settings.automation.tasks.template.blank.name"),
      description: t("settings.automation.tasks.template.blank.description"),
      detail: t("settings.automation.tasks.template.blank.detail"),
      actionLabel: t("settings.automation.tasks.template.blank.action"),
      icon: Plus,
    },
  ];
}

export type AutomationSettingsMode = "full" | "workspace" | "settings";

interface AutomationSettingsProps {
  mode?: AutomationSettingsMode;
  initialSelectedJobId?: string;
  initialWorkspaceTab?: AutomationWorkspaceTab;
  threadLineage?: AutomationThreadLineage | null;
  onOpenSettings?: () => void;
  onOpenWorkspace?: () => void;
}

function resolveAutomationLoadErrorMessage(
  error: unknown,
  fallback: string,
): string {
  if (error instanceof Error && error.message.trim()) {
    return error.message.trim();
  }
  if (typeof error === "string" && error.trim()) {
    return error.trim();
  }
  return fallback;
}

function withAutomationLoadTimeout<T>(
  promise: Promise<T>,
  timeoutMessage: string,
  timeoutMs: number,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timeoutId = window.setTimeout(() => {
      reject(new Error(timeoutMessage));
    }, timeoutMs);

    promise.then(resolve, reject).finally(() => {
      window.clearTimeout(timeoutId);
    });
  });
}

export function AutomationSettings({
  mode = "full",
  initialSelectedJobId,
  initialWorkspaceTab,
  threadLineage,
  onOpenSettings,
  onOpenWorkspace,
}: AutomationSettingsProps) {
  const { i18n, t: rawT } = useTranslation("settings");
  const t = rawT as SettingsTranslate;
  const serviceSkillContextCopy = useMemo(
    () => buildAutomationServiceSkillContextCopy(t),
    [t],
  );
  const automationPresentationCopy = useMemo(
    () => buildAutomationPresentationCopy(t, serviceSkillContextCopy),
    [serviceSkillContextCopy, t],
  );
  const legacyBrowserAutomationNotice =
    automationPresentationCopy.legacyBrowserAutomationNotice;
  const legacyBrowserAutomationStatus =
    automationPresentationCopy.legacyBrowserAutomationStatus;
  const serviceSkillExecutionCompatLabel = t(
    "settings.automation.tasks.list.badge.serviceSkillLegacyCompat",
  );
  const workspaceTemplates = useMemo(() => createWorkspaceTemplates(t), [t]);
  const workspaceOnly = mode === "workspace";
  const settingsOnly = mode === "settings";
  const showWorkspacePanels = !settingsOnly;
  const showSchedulerEditor = !workspaceOnly;
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [schedulerConfig, setSchedulerConfig] =
    useState<AutomationSchedulerConfig | null>(null);
  const [status, setStatus] = useState<AutomationStatus | null>(null);
  const [jobs, setJobs] = useState<AutomationJobRecord[]>([]);
  const [health, setHealth] = useState<AutomationHealthResult | null>(null);
  const [workspaces, setWorkspaces] = useState<Project[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [jobRuns, setJobRuns] = useState<AgentRun[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [schedulerSaving, setSchedulerSaving] = useState(false);
  const [jobSaving, setJobSaving] = useState(false);
  const [runningJobId, setRunningJobId] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogMode, setDialogMode] = useState<"create" | "edit">("create");
  const [dialogInitialValues, setDialogInitialValues] =
    useState<AutomationJobDialogInitialValues | null>(null);
  const [workspaceTab, setWorkspaceTab] = useState<AutomationWorkspaceTab>(
    initialWorkspaceTab ?? "tasks",
  );
  const [detailDialogOpen, setDetailDialogOpen] = useState(false);
  const autoOpenedInitialJobIdRef = useRef<string | null>(null);
  const isMountedRef = useRef(true);
  const refreshRequestIdRef = useRef(0);
  const historyRequestIdRef = useRef(0);
  const schedulerConfigRef = useRef<AutomationSchedulerConfig | null>(null);
  const statusRef = useRef<AutomationStatus | null>(null);
  const jobsRef = useRef<AutomationJobRecord[]>([]);

  useEffect(() => {
    isMountedRef.current = true;

    return () => {
      isMountedRef.current = false;
      refreshRequestIdRef.current += 1;
      historyRequestIdRef.current += 1;
    };
  }, []);

  useEffect(() => {
    schedulerConfigRef.current = schedulerConfig;
  }, [schedulerConfig]);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  useEffect(() => {
    jobsRef.current = jobs;
  }, [jobs, serviceSkillContextCopy]);

  const selectedJob = useMemo(
    () => jobs.find((job) => job.id === selectedJobId) ?? null,
    [jobs, selectedJobId],
  );
  const serviceSkillContextByJobId = useMemo(() => {
    const mapping = new Map<string, AutomationServiceSkillContext>();
    jobs.forEach((job) => {
      const context = resolveServiceSkillAutomationContext(
        job.payload,
        serviceSkillContextCopy,
      );
      if (context) {
        mapping.set(job.id, context);
      }
    });
    return mapping;
  }, [jobs, serviceSkillContextCopy]);
  const selectedServiceSkillContext = useMemo(
    () =>
      selectedJobId
        ? (serviceSkillContextByJobId.get(selectedJobId) ?? null)
        : null,
    [selectedJobId, serviceSkillContextByJobId],
  );
  const riskyJobMessageMap = useMemo(() => {
    const mapping = new Map<string, string>();
    health?.risky_jobs.forEach((job) => {
      if (job.detail_message?.trim()) {
        mapping.set(job.job_id, job.detail_message.trim());
      }
    });
    return mapping;
  }, [health]);
  const sceneAppAutomationContextByJobId = useMemo(() => {
    const mapping = new Map<
      string,
      NonNullable<ReturnType<typeof resolveLegacySceneAppAutomationContext>>
    >();
    jobs.forEach((job) => {
      const context = resolveLegacySceneAppAutomationContext(job.payload);
      if (context) {
        mapping.set(job.id, context);
      }
    });
    return mapping;
  }, [jobs]);
  const overviewFocusJob = useMemo(() => {
    if (selectedJob && sceneAppAutomationContextByJobId.has(selectedJob.id)) {
      return selectedJob;
    }

    const candidates = jobs.filter((job) =>
      sceneAppAutomationContextByJobId.has(job.id),
    );
    if (candidates.length === 0) {
      return null;
    }

    return (
      [...candidates].sort((left, right) => {
        const leftRisky = isAutomationJobAtRisk(left, riskyJobMessageMap);
        const rightRisky = isAutomationJobAtRisk(right, riskyJobMessageMap);
        if (leftRisky !== rightRisky) {
          return leftRisky ? -1 : 1;
        }
        if (left.enabled !== right.enabled) {
          return left.enabled ? -1 : 1;
        }
        return (
          resolveAutomationJobSortTime(right) -
          resolveAutomationJobSortTime(left)
        );
      })[0] ?? null
    );
  }, [jobs, riskyJobMessageMap, sceneAppAutomationContextByJobId, selectedJob]);
  const retiredSceneAppAutomationMessage = t(
    "settings.automation.details.sceneApp.retired.description",
  );
  const selectedRetiredSceneAppMessage =
    selectedJob && sceneAppAutomationContextByJobId.has(selectedJob.id)
      ? retiredSceneAppAutomationMessage
      : null;
  const overviewRetiredSceneAppMessage =
    overviewFocusJob &&
    sceneAppAutomationContextByJobId.has(overviewFocusJob.id)
      ? retiredSceneAppAutomationMessage
      : null;
  const legacyBrowserJobCount = useMemo(
    () => jobs.filter((job) => isLegacyBrowserAutomation(job)).length,
    [jobs],
  );

  const workspaceNameMap = useMemo(() => {
    const mapping = new Map<string, string>();
    workspaces.forEach((workspace) => {
      mapping.set(workspace.id, workspace.name);
    });
    return mapping;
  }, [workspaces]);
  const refreshAll = useCallback(
    async (silent: boolean = false) => {
      const requestId = refreshRequestIdRef.current + 1;
      refreshRequestIdRef.current = requestId;
      const hasVisibleContent = schedulerConfigRef.current !== null;
      const isCurrentRequest = () =>
        isMountedRef.current && refreshRequestIdRef.current === requestId;

      if (!silent) {
        setLoading(true);
      }
      try {
        const buildTimeoutMessage = (label: string, timeoutMs: number) =>
          t("settings.automation.main.load.timeout", {
            label,
            timeoutMs,
          });
        const schedulerConfigLoadLabel = t(
          "settings.automation.main.load.label.schedulerConfig",
        );
        const statusLoadLabel = t("settings.automation.main.load.label.status");
        const jobsLoadLabel = t("settings.automation.main.load.label.jobs");
        const [schedulerConfigResult, statusResult, jobsResult] =
          await Promise.allSettled([
            withAutomationLoadTimeout(
              getAutomationSchedulerConfig(),
              buildTimeoutMessage(
                schedulerConfigLoadLabel,
                AUTOMATION_CORE_LOAD_TIMEOUT_MS,
              ),
              AUTOMATION_CORE_LOAD_TIMEOUT_MS,
            ),
            withAutomationLoadTimeout(
              getAutomationStatus(),
              buildTimeoutMessage(
                statusLoadLabel,
                AUTOMATION_CORE_LOAD_TIMEOUT_MS,
              ),
              AUTOMATION_CORE_LOAD_TIMEOUT_MS,
            ),
            withAutomationLoadTimeout(
              getAutomationJobs(),
              buildTimeoutMessage(
                jobsLoadLabel,
                AUTOMATION_CORE_LOAD_TIMEOUT_MS,
              ),
              AUTOMATION_CORE_LOAD_TIMEOUT_MS,
            ),
          ]);

        const coreErrors: string[] = [];

        const nextSchedulerConfig =
          schedulerConfigResult.status === "fulfilled"
            ? schedulerConfigResult.value
            : schedulerConfigRef.current;
        const nextJobs =
          jobsResult.status === "fulfilled"
            ? jobsResult.value
            : jobsRef.current;

        if (schedulerConfigResult.status === "fulfilled") {
          setSchedulerConfig(schedulerConfigResult.value);
        } else {
          coreErrors.push(
            resolveAutomationLoadErrorMessage(
              schedulerConfigResult.reason,
              t("settings.automation.main.load.error.schedulerConfig"),
            ),
          );
        }

        if (statusResult.status === "fulfilled") {
          setStatus(statusResult.value);
        } else {
          coreErrors.push(
            resolveAutomationLoadErrorMessage(
              statusResult.reason,
              t("settings.automation.main.load.error.status"),
            ),
          );
        }

        if (jobsResult.status === "fulfilled") {
          setJobs(jobsResult.value);
          recordAutomationJobsRefreshAgentUiProjection(jobsResult.value);
        } else {
          coreErrors.push(
            resolveAutomationLoadErrorMessage(
              jobsResult.reason,
              t("settings.automation.main.load.error.jobs"),
            ),
          );
        }

        if (statusResult.status === "fulfilled") {
          recordAutomationStatusRefreshAgentUiProjection(
            statusResult.value,
            nextJobs,
          );
        }

        if (!nextSchedulerConfig) {
          throw new Error(
            coreErrors.join(
              t("settings.automation.main.load.errorSeparator"),
            ) || t("settings.automation.main.load.error.schedulerConfig"),
          );
        }

        if (!isCurrentRequest()) {
          return;
        }

        setLoadError(null);
        setSelectedJobId((current) => {
          if (!showWorkspacePanels) {
            return null;
          }
          if (
            initialSelectedJobId &&
            nextJobs.some((job) => job.id === initialSelectedJobId)
          ) {
            return initialSelectedJobId;
          }
          if (current && nextJobs.some((job) => job.id === current)) {
            return current;
          }
          return null;
        });
        if (
          showWorkspacePanels &&
          initialSelectedJobId &&
          nextJobs.some((job) => job.id === initialSelectedJobId) &&
          autoOpenedInitialJobIdRef.current !== initialSelectedJobId
        ) {
          setDetailDialogOpen(true);
          autoOpenedInitialJobIdRef.current = initialSelectedJobId;
        }
        if (coreErrors.length > 0) {
          toast.error(
            t("settings.automation.main.toast.partialLoad", {
              details: coreErrors.join(
                t("settings.automation.main.load.errorSeparator"),
              ),
            }),
          );
        }

        const workspacesLoadLabel = t(
          "settings.automation.main.load.label.workspaces",
        );
        const healthLoadLabel = t("settings.automation.main.load.label.health");
        void Promise.allSettled([
          withAutomationLoadTimeout(
            listProjects(),
            buildTimeoutMessage(
              workspacesLoadLabel,
              AUTOMATION_AUXILIARY_LOAD_TIMEOUT_MS,
            ),
            AUTOMATION_AUXILIARY_LOAD_TIMEOUT_MS,
          ),
          withAutomationLoadTimeout(
            getAutomationHealth({
              top_limit: Math.max(6, nextJobs.length),
            }),
            buildTimeoutMessage(
              healthLoadLabel,
              AUTOMATION_AUXILIARY_LOAD_TIMEOUT_MS,
            ),
            AUTOMATION_AUXILIARY_LOAD_TIMEOUT_MS,
          ),
        ]).then(([workspacesSettled, healthSettled]) => {
          if (!isCurrentRequest()) {
            return;
          }

          const auxiliaryErrors: string[] = [];

          if (workspacesSettled?.status === "fulfilled") {
            setWorkspaces(workspacesSettled.value);
          } else if (workspacesSettled) {
            auxiliaryErrors.push(
              resolveAutomationLoadErrorMessage(
                workspacesSettled.reason,
                t("settings.automation.main.load.error.workspaces"),
              ),
            );
          }

          if (healthSettled?.status === "fulfilled") {
            setHealth(healthSettled.value);
          } else if (healthSettled) {
            auxiliaryErrors.push(
              resolveAutomationLoadErrorMessage(
                healthSettled.reason,
                t("settings.automation.main.load.error.health"),
              ),
            );
          }

          if (auxiliaryErrors.length > 0) {
            toast.error(
              t("settings.automation.main.toast.partialLoad", {
                details: auxiliaryErrors.join(
                  t("settings.automation.main.load.errorSeparator"),
                ),
              }),
            );
          }
        });
      } catch (error) {
        if (!isCurrentRequest()) {
          return;
        }

        const message = resolveAutomationLoadErrorMessage(
          error,
          t("settings.automation.main.load.error.page"),
        );
        if (!hasVisibleContent) {
          setLoadError(message);
        }
        toast.error(
          t("settings.automation.main.toast.loadFailed", {
            message,
          }),
        );
      } finally {
        if (!silent && isCurrentRequest()) {
          setLoading(false);
        }
      }
    },
    [initialSelectedJobId, showWorkspacePanels, t],
  );

  const refreshHistory = useCallback(
    async (jobId: string) => {
      const requestId = historyRequestIdRef.current + 1;
      historyRequestIdRef.current = requestId;
      const isCurrentRequest = () =>
        isMountedRef.current && historyRequestIdRef.current === requestId;

      setHistoryLoading(true);
      try {
        const runs = await getAutomationRunHistory(jobId, 15);
        if (!isCurrentRequest()) {
          return;
        }
        setJobRuns(runs);
        const job = jobsRef.current.find((item) => item.id === jobId);
        if (job) {
          recordAutomationRunHistoryAgentUiProjection(job, runs);
        }
      } catch (error) {
        if (!isCurrentRequest()) {
          return;
        }
        toast.error(
          t("settings.automation.history.toast.loadFailed", {
            message: error instanceof Error ? error.message : String(error),
          }),
        );
        setJobRuns([]);
      } finally {
        if (isCurrentRequest()) {
          setHistoryLoading(false);
        }
      }
    },
    [t],
  );

  useEffect(() => {
    void refreshAll();
  }, [refreshAll]);

  useEffect(() => {
    if (!showWorkspacePanels || !initialSelectedJobId) {
      return;
    }

    setSelectedJobId((current) => {
      if (current === initialSelectedJobId) {
        return current;
      }
      if (jobs.length === 0) {
        return initialSelectedJobId;
      }
      if (jobs.some((job) => job.id === initialSelectedJobId)) {
        return initialSelectedJobId;
      }
      return current;
    });
  }, [initialSelectedJobId, jobs, showWorkspacePanels]);

  useEffect(() => {
    if (!initialSelectedJobId) {
      autoOpenedInitialJobIdRef.current = null;
      return;
    }
    if (!showWorkspacePanels) {
      return;
    }
    if (!jobs.some((job) => job.id === initialSelectedJobId)) {
      return;
    }
    if (autoOpenedInitialJobIdRef.current === initialSelectedJobId) {
      return;
    }
    setSelectedJobId(initialSelectedJobId);
    setDetailDialogOpen(true);
    autoOpenedInitialJobIdRef.current = initialSelectedJobId;
  }, [initialSelectedJobId, jobs, showWorkspacePanels]);

  useEffect(() => {
    if (!showWorkspacePanels || !initialWorkspaceTab) {
      return;
    }

    setWorkspaceTab(initialWorkspaceTab);
  }, [initialWorkspaceTab, showWorkspacePanels]);

  useEffect(() => {
    if (!showWorkspacePanels) {
      setDetailDialogOpen(false);
      setJobRuns([]);
      return;
    }
    if (!selectedJobId || !detailDialogOpen) {
      setJobRuns([]);
      return;
    }
    void refreshHistory(selectedJobId);
  }, [detailDialogOpen, refreshHistory, selectedJobId, showWorkspacePanels]);

  useEffect(() => {
    if (detailDialogOpen && !selectedJob) {
      setDetailDialogOpen(false);
    }
  }, [detailDialogOpen, selectedJob]);

  async function handleSaveScheduler() {
    if (!schedulerConfig) {
      return;
    }

    setSchedulerSaving(true);
    try {
      await updateAutomationSchedulerConfig({
        ...schedulerConfig,
        poll_interval_secs: Math.max(5, schedulerConfig.poll_interval_secs),
      });
      toast.success(t("settings.automation.scheduler.toast.saved"));
      await refreshAll(true);
    } catch (error) {
      toast.error(
        t("settings.automation.scheduler.toast.saveFailed", {
          message: error instanceof Error ? error.message : String(error),
        }),
      );
    } finally {
      setSchedulerSaving(false);
    }
  }

  async function handleSubmitJob(payload: AutomationJobDialogSubmit) {
    setJobSaving(true);
    try {
      const result =
        payload.mode === "create"
          ? await createAutomationJob(payload.request)
          : await updateAutomationJob(payload.id, payload.request);
      recordAutomationJobMutationAgentUiProjection(
        result,
        payload.mode === "create" ? "created" : "updated",
      );

      toast.success(
        payload.mode === "create"
          ? t("settings.automation.tasks.toast.created")
          : t("settings.automation.tasks.toast.updated"),
      );
      setDialogOpen(false);
      setDialogInitialValues(null);
      await refreshAll(true);
      setSelectedJobId(result.id);
      setDetailDialogOpen(true);
      await refreshHistory(result.id);
    } catch (error) {
      toast.error(
        t("settings.automation.tasks.toast.saveFailed", {
          message: error instanceof Error ? error.message : String(error),
        }),
      );
      throw error;
    } finally {
      setJobSaving(false);
    }
  }

  async function handleDeleteJob(job: AutomationJobRecord) {
    if (
      !window.confirm(
        t("settings.automation.tasks.confirm.delete", {
          name: job.name,
        }),
      )
    ) {
      return;
    }

    try {
      await deleteAutomationJob(job.id);
      recordAutomationJobMutationAgentUiProjection(job, "deleted");
      toast.success(t("settings.automation.tasks.toast.deleted"));
      await refreshAll(true);
    } catch (error) {
      toast.error(
        t("settings.automation.tasks.toast.deleteFailed", {
          message: error instanceof Error ? error.message : String(error),
        }),
      );
    }
  }

  async function handleRunNow(job: AutomationJobRecord) {
    if (isLegacyBrowserAutomation(job)) {
      toast.error(legacyBrowserAutomationNotice);
      return;
    }

    setRunningJobId(job.id);
    try {
      const result = await runAutomationJobNow(job.id);
      toast.success(
        t("settings.automation.tasks.toast.runCompleted", {
          success: result.success_count,
          failed: result.failed_count,
          timeout: result.timeout_count,
        }),
      );
      await refreshAll(true);
      await refreshHistory(job.id);
    } catch (error) {
      toast.error(
        t("settings.automation.tasks.toast.runFailed", {
          message: error instanceof Error ? error.message : String(error),
        }),
      );
    } finally {
      setRunningJobId(null);
    }
  }

  function openCreateDialog(
    initialValues?: AutomationJobDialogInitialValues | null,
  ) {
    if (!threadLineage?.sessionId?.trim() || !threadLineage.threadId?.trim()) {
      toast.error(
        t("settings.automation.jobDialog.validation.threadLineageRequired"),
      );
      return;
    }
    setDialogMode("create");
    setDialogInitialValues(initialValues ?? null);
    setDialogOpen(true);
  }

  function openEditDialog(job: AutomationJobRecord) {
    setSelectedJobId(job.id);
    setDialogMode("edit");
    setDialogInitialValues(null);
    setDialogOpen(true);
  }

  function openJobDetails(jobId: string) {
    setSelectedJobId(jobId);
    setDetailDialogOpen(true);
  }

  function handleDialogOpenChange(open: boolean) {
    setDialogOpen(open);
    if (!open) {
      setDialogInitialValues(null);
    }
  }

  const heroTitle = settingsOnly
    ? t("settings.automation.main.hero.title.settings")
    : workspaceOnly
      ? t("settings.automation.main.hero.title.workspace")
      : t("settings.automation.main.hero.title.full");
  const heroDescription = settingsOnly
    ? t("settings.automation.main.hero.description.settings")
    : workspaceOnly
      ? t("settings.automation.main.hero.description.workspace")
      : t("settings.automation.main.hero.description.full");
  const headerSummary = settingsOnly
    ? t("settings.automation.main.hero.summary.settings")
    : workspaceOnly
      ? t("settings.automation.main.hero.summary.workspace")
      : t("settings.automation.main.hero.summary.full");

  if (loading && !schedulerConfig) {
    return (
      <div className="flex h-40 items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  if (!schedulerConfig) {
    return (
      <Card className="rounded-[28px] border-slate-200/80 bg-white shadow-sm shadow-slate-950/5">
        <CardContent className="flex min-h-40 flex-col items-center justify-center gap-3 py-10 text-center">
          <div className="text-lg font-semibold text-slate-900">
            {t("settings.automation.main.load.errorTitle")}
          </div>
          <p className="max-w-[520px] text-sm leading-6 text-slate-500">
            {loadError ?? t("settings.automation.main.load.errorDescription")}
          </p>
          <Button onClick={() => void refreshAll()}>
            {t("settings.automation.main.action.reload")}
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="lime-workbench-theme-scope space-y-6 pb-8">
      <section className="rounded-[28px] border border-slate-200/80 bg-white px-5 py-4 shadow-sm shadow-slate-950/5">
        <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center gap-2">
              {!settingsOnly ? (
                <Badge
                  variant="outline"
                  className="rounded-full border-sky-200 bg-sky-50 px-3 py-1 text-xs font-medium text-sky-700"
                >
                  {t("settings.automation.main.badge.systemEntry")}
                </Badge>
              ) : null}
              <h1 className="text-[24px] font-semibold tracking-tight text-slate-900">
                {heroTitle}
              </h1>
              <WorkbenchInfoTip
                ariaLabel={t("settings.automation.main.hero.tipAria")}
                content={heroDescription}
                tone="mint"
              />
            </div>
            <p className="text-sm text-slate-500">{headerSummary}</p>
          </div>

          <div className="flex flex-wrap items-center gap-2 xl:justify-end">
            {showWorkspacePanels ? (
              <Button variant="default" onClick={() => openCreateDialog()}>
                <Plus className="mr-2 h-4 w-4" />
                {t("settings.automation.main.action.create")}
              </Button>
            ) : null}
            {workspaceOnly && onOpenSettings ? (
              <Button variant="outline" onClick={onOpenSettings}>
                {t("settings.automation.main.action.openSettings")}
              </Button>
            ) : null}
            {settingsOnly && onOpenWorkspace ? (
              <Button variant="outline" onClick={onOpenWorkspace}>
                {t("settings.automation.main.action.openWorkspace")}
              </Button>
            ) : null}
            <Button
              variant={showWorkspacePanels ? "outline" : "default"}
              onClick={() => void refreshAll(true)}
              disabled={loading}
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              {t("settings.automation.main.action.refresh")}
            </Button>
          </div>
        </div>

        <div className="mt-4 flex flex-col gap-4 rounded-[22px] border border-slate-200/80 bg-slate-50/60 p-4">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`rounded-full border px-3 py-1.5 text-xs font-medium ${
                schedulerConfig.enabled
                  ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                  : "border-slate-200 bg-white text-slate-600"
              }`}
            >
              {t("settings.automation.main.summary.scheduler", {
                status: schedulerConfig.enabled
                  ? t("settings.automation.main.status.enabled")
                  : t("settings.automation.main.status.disabled"),
              })}
            </span>
            <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600">
              {t("settings.automation.main.summary.jobCount", {
                count: jobs.length,
              })}
            </span>
            <span
              className={`rounded-full border px-3 py-1.5 text-xs font-medium ${
                (health?.risky_jobs.length ?? 0) > 0
                  ? "border-amber-200 bg-amber-50 text-amber-700"
                  : "border-slate-200 bg-white text-slate-600"
              }`}
            >
              {t("settings.automation.main.summary.riskCount", {
                count: health?.risky_jobs.length ?? 0,
              })}
            </span>
            <span
              className={`rounded-full border px-3 py-1.5 text-xs font-medium ${
                status?.running
                  ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                  : "border-slate-200 bg-white text-slate-600"
              }`}
            >
              {t("settings.automation.main.summary.polling", {
                status: status?.running
                  ? t("settings.automation.main.polling.running")
                  : t("settings.automation.main.polling.paused"),
              })}
            </span>
            {legacyBrowserJobCount > 0 ? (
              <span className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-700">
                {t("settings.automation.main.summary.legacyBrowserCount", {
                  count: legacyBrowserJobCount,
                })}
              </span>
            ) : null}
          </div>

          {showWorkspacePanels ? (
            <Tabs
              value={workspaceTab}
              onValueChange={(value) =>
                setWorkspaceTab(value as AutomationWorkspaceTab)
              }
            >
              <TabsList className="grid h-auto w-full max-w-[420px] grid-cols-2 rounded-[20px] border border-slate-200 bg-white p-1 shadow-sm shadow-slate-950/5">
                <TabsTrigger
                  value="tasks"
                  data-testid="automation-tab-tasks"
                  className="rounded-[14px] px-4 py-3"
                >
                  {t("settings.automation.main.tab.tasks")}
                </TabsTrigger>
                <TabsTrigger
                  value="overview"
                  data-testid="automation-tab-overview"
                  className="rounded-[14px] px-4 py-3"
                >
                  {t("settings.automation.main.tab.overview")}
                </TabsTrigger>
              </TabsList>
            </Tabs>
          ) : null}
        </div>
      </section>

      {showSchedulerEditor ? (
        <Card className="rounded-[28px] border-slate-200/80 bg-white shadow-sm shadow-slate-950/5">
          <CardHeader className="pb-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <CardTitle className="text-xl text-slate-900">
                    {t("settings.automation.scheduler.title")}
                  </CardTitle>
                  <WorkbenchInfoTip
                    ariaLabel={t("settings.automation.scheduler.tipAria")}
                    content={t("settings.automation.scheduler.description")}
                    tone="slate"
                  />
                </div>
              </div>
              <Badge variant={schedulerConfig.enabled ? "default" : "outline"}>
                {schedulerConfig.enabled
                  ? t("settings.automation.main.status.enabled")
                  : t("settings.automation.main.status.disabled")}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              <div className="rounded-[22px] border border-slate-200/80 bg-slate-50/70 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-medium text-slate-900">
                      {t("settings.automation.scheduler.enable.label")}
                    </div>
                    <div className="mt-1 text-xs text-slate-500">
                      {t("settings.automation.scheduler.enable.description")}
                    </div>
                  </div>
                  <Switch
                    checked={schedulerConfig.enabled}
                    onCheckedChange={(checked) =>
                      setSchedulerConfig((current) =>
                        current ? { ...current, enabled: checked } : current,
                      )
                    }
                  />
                </div>
              </div>

              <div className="rounded-[22px] border border-slate-200/80 bg-slate-50/70 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-medium text-slate-900">
                      {t("settings.automation.scheduler.history.label")}
                    </div>
                    <div className="mt-1 text-xs text-slate-500">
                      {t("settings.automation.scheduler.history.description")}
                    </div>
                  </div>
                  <Switch
                    checked={schedulerConfig.enable_history}
                    onCheckedChange={(checked) =>
                      setSchedulerConfig((current) =>
                        current
                          ? { ...current, enable_history: checked }
                          : current,
                      )
                    }
                  />
                </div>
              </div>

              <div className="rounded-[22px] border border-slate-200/80 bg-slate-50/70 p-4">
                <div className="text-sm font-medium text-slate-900">
                  {t("settings.automation.scheduler.pollInterval.label")}
                </div>
                <Input
                  className="mt-3"
                  type="number"
                  min={5}
                  value={schedulerConfig.poll_interval_secs}
                  onChange={(event) =>
                    setSchedulerConfig((current) =>
                      current
                        ? {
                            ...current,
                            poll_interval_secs: Number(event.target.value) || 5,
                          }
                        : current,
                    )
                  }
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <Button
                onClick={() => void handleSaveScheduler()}
                disabled={schedulerSaving}
              >
                {schedulerSaving
                  ? t("settings.automation.scheduler.action.saving")
                  : t("settings.automation.scheduler.action.save")}
              </Button>
              <div className="text-sm text-slate-500">
                {t("settings.automation.scheduler.pollWindow", {
                  last: formatTime(status?.last_polled_at, i18n.language),
                  next: formatTime(status?.next_poll_at, i18n.language),
                })}
              </div>
            </div>
          </CardContent>
        </Card>
      ) : null}

      {showWorkspacePanels ? (
        <Tabs
          value={workspaceTab}
          onValueChange={(value) =>
            setWorkspaceTab(value as AutomationWorkspaceTab)
          }
          className="space-y-0"
        >
          <TabsContent value="tasks" className="mt-0 space-y-6">
            <Card className="rounded-[28px] border-slate-200/80 bg-white shadow-sm shadow-slate-950/5">
              <CardHeader className="pb-4">
                <div className="flex flex-wrap items-end justify-between gap-4">
                  <div>
                    <div className="text-xs font-medium tracking-[0.14em] text-slate-500">
                      {t("settings.automation.tasks.kicker")}
                    </div>
                    <div className="mt-2 flex flex-wrap items-center gap-2">
                      <CardTitle className="text-xl text-slate-900">
                        {t("settings.automation.tasks.title")}
                      </CardTitle>
                      <WorkbenchInfoTip
                        ariaLabel={t("settings.automation.tasks.tipAria")}
                        content={t("settings.automation.tasks.description")}
                        tone="slate"
                      />
                    </div>
                  </div>
                  <Button variant="outline" onClick={() => openCreateDialog()}>
                    <Plus className="mr-2 h-4 w-4" />
                    {t("settings.automation.tasks.action.blankStart")}
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {legacyBrowserJobCount > 0 ? (
                  <div className="mb-4 rounded-[22px] border border-amber-200 bg-amber-50 px-4 py-4 text-sm leading-6 text-amber-800">
                    {t("settings.automation.tasks.legacyBrowserNotice", {
                      count: legacyBrowserJobCount,
                    })}
                  </div>
                ) : null}
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                  {workspaceTemplates.map((template) => {
                    const Icon = template.icon;
                    return (
                      <div
                        key={template.id}
                        className="rounded-[24px] border border-slate-200/80 bg-slate-50/70 p-5"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-[14px] bg-white text-slate-700 shadow-sm shadow-slate-950/5">
                            <Icon className="h-5 w-5" />
                          </div>
                          <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-500">
                            {template.tag}
                          </span>
                        </div>
                        <div className="mt-4 text-base font-semibold text-slate-900">
                          <span className="inline-flex items-center gap-2">
                            {template.name}
                            <WorkbenchInfoTip
                              ariaLabel={t(
                                "settings.automation.tasks.template.tipAria",
                                {
                                  name: template.name,
                                },
                              )}
                              content={template.detail}
                              tone="slate"
                            />
                          </span>
                        </div>
                        <p className="mt-2 text-sm leading-6 text-slate-600">
                          {template.description}
                        </p>
                        <Button
                          data-testid={`automation-template-${template.id}`}
                          variant="ghost"
                          className="mt-5 w-full justify-between rounded-[16px] border border-slate-200 bg-white text-slate-700 hover:bg-slate-100"
                          onClick={() =>
                            openCreateDialog(template.initialValues)
                          }
                        >
                          {template.actionLabel}
                          <ArrowRight className="h-4 w-4" />
                        </Button>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            <div className="space-y-6">
              <Card className="rounded-[28px] border-slate-200/80 bg-white shadow-sm shadow-slate-950/5">
                <CardHeader className="pb-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <CardTitle className="text-xl text-slate-900">
                          {t("settings.automation.tasks.list.title")}
                        </CardTitle>
                        <WorkbenchInfoTip
                          ariaLabel={t(
                            "settings.automation.tasks.list.tipAria",
                          )}
                          content={t(
                            "settings.automation.tasks.list.description",
                          )}
                          tone="slate"
                        />
                      </div>
                    </div>
                    <Badge variant="outline">
                      {t("settings.automation.tasks.list.count", {
                        count: jobs.length,
                      })}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  {jobs.length ? (
                    <Table className="min-w-[1120px]">
                      <TableHeader>
                        <TableRow>
                          <TableHead className="min-w-[320px]">
                            {t("settings.automation.tasks.list.column.flow")}
                          </TableHead>
                          <TableHead className="min-w-[140px]">
                            {t(
                              "settings.automation.tasks.list.column.workspace",
                            )}
                          </TableHead>
                          <TableHead className="min-w-[150px]">
                            {t(
                              "settings.automation.tasks.list.column.schedule",
                            )}
                          </TableHead>
                          <TableHead className="min-w-[110px]">
                            {t("settings.automation.tasks.list.column.mode")}
                          </TableHead>
                          <TableHead className="min-w-[210px]">
                            {t("settings.automation.tasks.list.column.status")}
                          </TableHead>
                          <TableHead className="min-w-[150px]">
                            {t("settings.automation.tasks.list.column.lastRun")}
                          </TableHead>
                          <TableHead className="min-w-[240px] text-right">
                            {t("settings.automation.tasks.list.column.actions")}
                          </TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {jobs.map((job) => {
                          const jobDetailMessage = riskyJobMessageMap.get(
                            job.id,
                          );
                          const serviceSkillContext =
                            serviceSkillContextByJobId.get(job.id) ?? null;
                          const serviceSkillTaskLine = serviceSkillContext
                            ? describeServiceSkillTaskLine(
                                serviceSkillContext,
                                automationPresentationCopy,
                              )
                            : null;
                          const serviceSkillSlotPreview = serviceSkillContext
                            ? describeServiceSkillSlotPreview(
                                serviceSkillContext,
                                2,
                                automationPresentationCopy,
                              )
                            : null;
                          const legacyBrowserJob =
                            isLegacyBrowserAutomation(job);
                          const isOverviewFocusRow =
                            overviewFocusJob?.id === job.id;
                          return (
                            <TableRow
                              key={job.id}
                              data-testid={`automation-job-row-${job.id}`}
                              className={
                                isOverviewFocusRow
                                  ? "cursor-pointer bg-sky-50/70"
                                  : selectedJobId === job.id
                                    ? "cursor-pointer bg-slate-50"
                                    : "cursor-pointer"
                              }
                              onClick={() => openJobDetails(job.id)}
                            >
                              <TableCell className="align-top">
                                <div className="space-y-1">
                                  <div className="font-medium text-slate-900">
                                    {job.name}
                                  </div>
                                  <div className="max-w-[320px] text-xs leading-5 text-slate-500">
                                    {job.description ||
                                      t(
                                        "settings.automation.tasks.list.descriptionFallback",
                                      )}
                                  </div>
                                  {serviceSkillContext ? (
                                    <div
                                      data-testid={`automation-job-service-skill-summary-${job.id}`}
                                      className="space-y-1.5 pt-1"
                                    >
                                      <div className="flex flex-wrap gap-2">
                                        <Badge
                                          variant="outline"
                                          className="border-sky-200 bg-sky-50 text-sky-700"
                                        >
                                          {t(
                                            "settings.automation.tasks.list.badge.serviceSkill",
                                          )}
                                        </Badge>
                                        <Badge variant="outline">
                                          {serviceSkillContext.runnerLabel}
                                        </Badge>
                                        <Badge variant="outline">
                                          {
                                            serviceSkillContext.executionLocationLabel
                                          }
                                        </Badge>
                                        {serviceSkillContext.executionLocationLegacyCompat ? (
                                          <Badge variant="outline">
                                            {serviceSkillExecutionCompatLabel}
                                          </Badge>
                                        ) : null}
                                        <Badge variant="outline">
                                          {serviceSkillContext.sourceLabel}
                                        </Badge>
                                      </div>
                                      {serviceSkillTaskLine ? (
                                        <div className="max-w-[360px] text-xs leading-5 text-slate-600">
                                          {serviceSkillTaskLine}
                                        </div>
                                      ) : null}
                                      {serviceSkillSlotPreview ? (
                                        <div className="max-w-[360px] text-xs leading-5 text-slate-500">
                                          {t(
                                            "settings.automation.tasks.list.serviceSkillSlotPreview",
                                            {
                                              summary: serviceSkillSlotPreview,
                                            },
                                          )}
                                        </div>
                                      ) : null}
                                    </div>
                                  ) : null}
                                  {isOverviewFocusRow ? (
                                    <AutomationJobFocusStrip
                                      jobId={job.id}
                                      retiredMessage={
                                        overviewRetiredSceneAppMessage
                                      }
                                    />
                                  ) : null}
                                </div>
                              </TableCell>
                              <TableCell className="align-top text-sm text-slate-500">
                                {workspaceNameMap.get(job.workspace_id) ??
                                  job.workspace_id}
                              </TableCell>
                              <TableCell className="align-top text-sm text-slate-500">
                                {describeSchedule(
                                  job,
                                  automationPresentationCopy,
                                  i18n.language,
                                )}
                              </TableCell>
                              <TableCell className="align-top text-sm text-slate-500">
                                {executionModeLabel(
                                  job.execution_mode,
                                  automationPresentationCopy,
                                )}
                              </TableCell>
                              <TableCell className="align-top">
                                <div className="space-y-2">
                                  <div className="flex flex-wrap gap-2">
                                    <Badge
                                      variant={statusVariant(job.last_status)}
                                    >
                                      {statusLabel(
                                        job.last_status,
                                        automationPresentationCopy,
                                      )}
                                    </Badge>
                                    {legacyBrowserJob ? (
                                      <Badge variant="outline">
                                        {legacyBrowserAutomationStatus}
                                      </Badge>
                                    ) : null}
                                    {!job.enabled ? (
                                      <Badge variant="outline">
                                        {t(
                                          "settings.automation.tasks.list.badge.disabled",
                                        )}
                                      </Badge>
                                    ) : null}
                                    {isOverviewFocusRow ? (
                                      <Badge
                                        variant="outline"
                                        className="border-sky-200 bg-sky-50 text-sky-700"
                                      >
                                        {t("settings.automation.focus.label")}
                                      </Badge>
                                    ) : null}
                                    {job.auto_disabled_until ? (
                                      <Badge variant="destructive">
                                        {t(
                                          "settings.automation.tasks.list.badge.cooldown",
                                        )}
                                      </Badge>
                                    ) : null}
                                  </div>
                                  {jobDetailMessage ? (
                                    <div
                                      className={`max-w-[260px] text-xs leading-5 ${statusDetailToneClass(
                                        job.last_status,
                                      )}`}
                                    >
                                      {statusDetailPrefix(
                                        job.last_status,
                                        automationPresentationCopy,
                                      )}
                                      : {jobDetailMessage}
                                    </div>
                                  ) : null}
                                </div>
                              </TableCell>
                              <TableCell className="align-top text-sm text-slate-500">
                                <div
                                  data-testid={`automation-job-run-window-${job.id}`}
                                  className="space-y-1"
                                >
                                  <div>
                                    {t(
                                      "settings.automation.tasks.list.nextRun",
                                      {
                                        time: formatTime(
                                          job.next_run_at,
                                          i18n.language,
                                        ),
                                      },
                                    )}
                                  </div>
                                  <div className="text-xs text-slate-400">
                                    {t(
                                      "settings.automation.tasks.list.recentRun",
                                      {
                                        time: formatTime(
                                          job.last_run_at,
                                          i18n.language,
                                        ),
                                      },
                                    )}
                                  </div>
                                </div>
                              </TableCell>
                              <TableCell className="align-top">
                                <div className="flex justify-end gap-2">
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      void handleRunNow(job);
                                    }}
                                    disabled={
                                      runningJobId === job.id ||
                                      legacyBrowserJob
                                    }
                                  >
                                    <Play className="mr-1 h-4 w-4" />
                                    {legacyBrowserJob
                                      ? legacyBrowserAutomationStatus
                                      : runningJobId === job.id
                                        ? t(
                                            "settings.automation.tasks.list.action.running",
                                          )
                                        : t(
                                            "settings.automation.tasks.list.action.run",
                                          )}
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      openEditDialog(job);
                                    }}
                                  >
                                    <Pencil className="h-4 w-4" />
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    className="gap-1.5 text-slate-600"
                                    data-testid={`automation-job-open-details-${job.id}`}
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      openJobDetails(job.id);
                                    }}
                                  >
                                    <History className="h-4 w-4" />
                                    {t(
                                      "settings.automation.tasks.list.action.details",
                                    )}
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      void handleDeleteJob(job);
                                    }}
                                  >
                                    <Trash2 className="h-4 w-4" />
                                  </Button>
                                </div>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  ) : (
                    <div className="rounded-[22px] border border-dashed border-slate-200 bg-slate-50 p-10 text-center">
                      <div className="text-base font-medium text-slate-900">
                        {t("settings.automation.tasks.empty.title")}
                      </div>
                      <p className="mt-2 text-sm text-slate-500">
                        {t("settings.automation.tasks.empty.description")}
                      </p>
                      <Button
                        className="mt-5"
                        onClick={() => openCreateDialog()}
                      >
                        <Plus className="mr-2 h-4 w-4" />
                        {t("settings.automation.tasks.empty.action")}
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="overview" className="mt-0 space-y-6">
            <Card className="rounded-[28px] border-slate-200/80 bg-white shadow-sm shadow-slate-950/5">
              <CardHeader className="pb-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <CardTitle className="text-xl text-slate-900">
                        {t("settings.automation.overview.title")}
                      </CardTitle>
                      <WorkbenchInfoTip
                        ariaLabel={t("settings.automation.overview.tipAria")}
                        content={t("settings.automation.overview.tipContent")}
                        tone="slate"
                      />
                    </div>
                  </div>
                  <LatestRunStatusBadge
                    source="automation"
                    label={t(
                      "settings.automation.overview.latestRunStatusLabel",
                    )}
                  />
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                  <div className="rounded-[22px] border border-slate-200/80 bg-slate-50/70 p-4">
                    <div className="text-sm font-medium text-slate-900">
                      {t("settings.automation.overview.metric.scheduler")}
                    </div>
                    <div className="mt-3 text-2xl font-semibold tracking-tight text-slate-900">
                      {status?.running
                        ? t("settings.automation.overview.status.running")
                        : t("settings.automation.overview.status.stopped")}
                    </div>
                  </div>
                  <div className="rounded-[22px] border border-slate-200/80 bg-slate-50/70 p-4">
                    <div className="text-sm font-medium text-slate-900">
                      {t("settings.automation.overview.metric.lastPoll")}
                    </div>
                    <div className="mt-3 text-base font-semibold text-slate-900">
                      {formatTime(status?.last_polled_at, i18n.language)}
                    </div>
                  </div>
                  <div className="rounded-[22px] border border-slate-200/80 bg-slate-50/70 p-4">
                    <div className="text-sm font-medium text-slate-900">
                      {t("settings.automation.overview.metric.nextPoll")}
                    </div>
                    <div className="mt-3 text-base font-semibold text-slate-900">
                      {formatTime(status?.next_poll_at, i18n.language)}
                    </div>
                  </div>
                  <div className="rounded-[22px] border border-slate-200/80 bg-slate-50/70 p-4">
                    <div className="text-sm font-medium text-slate-900">
                      {t("settings.automation.overview.metric.active")}
                    </div>
                    <div className="mt-3 text-base font-semibold text-slate-900">
                      {status?.active_job_name ??
                        t("settings.automation.overview.active.empty")}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <AutomationOverviewFocusCard
              job={overviewFocusJob}
              workspaceName={
                overviewFocusJob
                  ? (workspaceNameMap.get(overviewFocusJob.workspace_id) ??
                    null)
                  : null
              }
              retiredMessage={overviewRetiredSceneAppMessage}
              onOpenJobDetails={
                overviewFocusJob
                  ? () => openJobDetails(overviewFocusJob.id)
                  : undefined
              }
            />

            <AutomationHealthPanel health={health} status={status} />
          </TabsContent>
        </Tabs>
      ) : null}

      <AutomationJobDialog
        open={dialogOpen}
        mode={dialogMode}
        job={dialogMode === "edit" ? selectedJob : null}
        workspaces={workspaces}
        initialValues={dialogMode === "create" ? dialogInitialValues : null}
        threadLineage={threadLineage}
        saving={jobSaving}
        onOpenChange={handleDialogOpenChange}
        onSubmit={handleSubmitJob}
      />
      <AutomationJobDetailsDialog
        open={detailDialogOpen}
        onOpenChange={setDetailDialogOpen}
        job={selectedJob}
        workspaceName={
          selectedJob
            ? (workspaceNameMap.get(selectedJob.workspace_id) ?? null)
            : null
        }
        serviceSkillContext={selectedServiceSkillContext}
        jobRuns={jobRuns}
        historyLoading={historyLoading}
        retiredSceneAppMessage={selectedRetiredSceneAppMessage}
        onRefreshHistory={refreshHistory}
      />
    </div>
  );
}
