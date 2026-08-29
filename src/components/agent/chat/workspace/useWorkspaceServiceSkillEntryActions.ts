import { useCallback, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { scheduledTasksApi } from "@/lib/api/scheduledTasks";
import { createContent } from "@/lib/api/project";
import type { ScheduledTaskFormState } from "@/components/scheduled-tasks/scheduledTaskViewModel";
import type {
  A2UIFormData,
  A2UIResponse,
} from "@/components/workspace/a2ui/types";
import type { Page, PageParams } from "@/types/page";
import type { ChatToolPreferences } from "../utils/chatToolPreferences";
import type { CreationMode } from "../components/types";
import type { PendingA2UISource } from "../types";
import { normalizeProjectId } from "../utils/topicProjectResolution";
import {
  resolveWorkspaceEntry,
  type WorkspaceEntryPayload,
} from "../workspaceEntry";
import {
  composeServiceSkillPrompt,
  validateServiceSkillSlotValues,
} from "../service-skills/promptComposer";
import { supportsServiceSkillLocalAutomation } from "../service-skills/automationDraft";
import { recordServiceSkillAutomationLink } from "../service-skills/automationLinkStorage";
import {
  buildServiceSkillLaunchA2UIResponse,
  readServiceSkillLaunchSlotValuesFromFormData,
} from "../service-skills/serviceSkillLaunchA2UI";
import { buildServiceSkillWorkspaceSeed } from "../service-skills/workspaceLaunch";
import type { CreationReplayMetadata } from "../utils/creationReplayMetadata";
import type {
  RecordServiceSkillUsageInput,
  ServiceSkillHomeItem,
  ServiceSkillSlotValues,
} from "../service-skills/types";
import {
  buildServiceSkillAutomationSetupState,
  buildServiceSkillScheduledTaskCreateRequest,
  buildServiceSkillSelectionPlan,
  getWorkspaceServiceSkillErrorMessage,
  normalizeWorkspaceServiceSkillOptionalText,
  type PendingServiceSkillAutomationLaunch,
  type PendingServiceSkillLaunchInputState,
  resolveServiceSkillLaunchUserInput,
  type ServiceSkillSelectionOptions,
  shouldCreateServiceSkillAutomationContent,
} from "./workspaceServiceSkillEntryActionsViewModel";

interface ServiceSkillLaunchOptions {
  launchUserInput?: string | null;
}

interface UseWorkspaceServiceSkillEntryActionsParams {
  activeTheme: string;
  creationMode: CreationMode;
  projectId?: string | null;
  contentId?: string | null;
  sessionId?: string | null;
  threadId?: string | null;
  ensureSessionForThreadLineage?: (options?: {
    skipSessionRestore?: boolean;
    skipSessionStartHooks?: boolean;
  }) => Promise<string | null>;
  input: string;
  chatToolPreferences: ChatToolPreferences;
  providerType?: string;
  model?: string;
  creationReplay?: CreationReplayMetadata;
  onNavigate?: (page: Page, params?: PageParams) => void;
  recordServiceSkillUsage: (input: RecordServiceSkillUsageInput) => void;
}

export function useWorkspaceServiceSkillEntryActions({
  activeTheme,
  creationMode,
  projectId,
  contentId,
  sessionId,
  threadId,
  ensureSessionForThreadLineage,
  input,
  chatToolPreferences,
  providerType,
  model,
  creationReplay,
  onNavigate,
  recordServiceSkillUsage,
}: UseWorkspaceServiceSkillEntryActionsParams) {
  const { t } = useTranslation("workspace");
  const [automationDialogOpen, setAutomationDialogOpen] = useState(false);
  const [automationDialogInitialValues, setAutomationDialogInitialValues] =
    useState<ScheduledTaskFormState | null>(null);
  const [automationJobSaving, setAutomationJobSaving] = useState(false);
  const [pendingServiceSkillAutomation, setPendingServiceSkillAutomation] =
    useState<PendingServiceSkillAutomationLaunch | null>(null);
  const [pendingServiceSkillLaunchInput, setPendingServiceSkillLaunchInput] =
    useState<PendingServiceSkillLaunchInputState | null>(null);
  const serviceSkillLaunchRequestCountRef = useRef(0);

  const currentProjectId = normalizeProjectId(projectId);
  const currentContentId = contentId?.trim() || null;
  const resolveAutomationThreadLineage = useCallback(async () => {
    const currentSessionId = sessionId?.trim();
    const currentThreadId = (threadId ?? sessionId)?.trim();
    const currentLineage =
      currentSessionId && currentThreadId
        ? { sessionId: currentSessionId, threadId: currentThreadId }
        : null;
    if (currentLineage) {
      return currentLineage;
    }

    const ensuredSessionId =
      (await ensureSessionForThreadLineage?.())?.trim() || null;
    if (!ensuredSessionId) {
      return null;
    }

    return { sessionId: ensuredSessionId, threadId: ensuredSessionId };
  }, [ensureSessionForThreadLineage, sessionId, threadId]);

  const navigateToServiceSkillWorkspace = useCallback(
    (payload: WorkspaceEntryPayload): boolean => {
      const resolved = resolveWorkspaceEntry({
        projectId: payload.projectId ?? currentProjectId,
        activeTheme,
        creationMode,
        defaultToolPreferences: chatToolPreferences,
        payload,
      });

      if (!resolved.ok) {
        if (resolved.reason === "missing_project") {
          toast.error("缺少项目工作区，请先选择项目后再启动技能。");
          return false;
        }
        toast.error("技能缺少可执行内容，请先补齐参数后重试。");
        return false;
      }

      if (!onNavigate) {
        toast.error("当前入口暂不支持切换技能工作区，请从桌面主界面重试。");
        return false;
      }

      onNavigate("agent", resolved.navigationParams);
      return true;
    },
    [
      activeTheme,
      chatToolPreferences,
      creationMode,
      currentProjectId,
      onNavigate,
    ],
  );

  const createServiceSkillSeededContent = useCallback(
    async (
      skill: ServiceSkillHomeItem,
      targetProjectId?: string | null,
      options?: {
        body?: string;
        metadata?: Record<string, unknown>;
      },
    ) => {
      const normalizedProjectId = normalizeProjectId(
        targetProjectId ?? currentProjectId,
      );
      const seed = buildServiceSkillWorkspaceSeed(
        skill,
        skill.themeTarget ?? activeTheme,
      );

      if (!normalizedProjectId || !seed) {
        return null;
      }

      const mergedMetadata = {
        ...(seed.metadata ?? {}),
        ...(options?.metadata ?? {}),
      };

      return createContent({
        project_id: normalizedProjectId,
        title: seed.title,
        content_type: seed.contentType,
        body: options?.body ?? "",
        metadata:
          Object.keys(mergedMetadata).length > 0 ? mergedMetadata : undefined,
      });
    },
    [activeTheme, currentProjectId],
  );

  const prepareServiceSkillWorkspacePayload = useCallback(
    async (
      skill: ServiceSkillHomeItem,
      prompt: string,
      options?: {
        contentId?: string | null;
        projectId?: string | null;
      },
    ): Promise<WorkspaceEntryPayload> => {
      const normalizedProjectId = normalizeProjectId(
        options?.projectId ?? currentProjectId,
      );
      const existingContentId =
        options?.contentId?.trim() || currentContentId || undefined;
      const seed = buildServiceSkillWorkspaceSeed(
        skill,
        skill.themeTarget ?? activeTheme,
      );

      if (existingContentId) {
        return {
          prompt,
          contentId: existingContentId,
          themeOverride: skill.themeTarget,
          initialRequestMetadata: seed?.requestMetadata,
          autoRunInitialPromptOnMount: true,
        };
      }

      if (!normalizedProjectId || !seed) {
        return {
          prompt,
          themeOverride: skill.themeTarget,
          initialRequestMetadata: seed?.requestMetadata,
          autoRunInitialPromptOnMount: true,
        };
      }

      const created = await createServiceSkillSeededContent(
        skill,
        normalizedProjectId,
      );

      if (!created) {
        return {
          prompt,
          themeOverride: skill.themeTarget,
          initialRequestMetadata: seed.requestMetadata,
          autoRunInitialPromptOnMount: true,
        };
      }

      return {
        prompt,
        contentId: created.id,
        themeOverride: skill.themeTarget,
        initialRequestMetadata: seed.requestMetadata,
        autoRunInitialPromptOnMount: true,
      };
    },
    [
      activeTheme,
      createServiceSkillSeededContent,
      currentContentId,
      currentProjectId,
    ],
  );

  const handleServiceSkillLaunch = useCallback(
    async (
      skill: ServiceSkillHomeItem,
      slotValues: ServiceSkillSlotValues,
      options?: ServiceSkillLaunchOptions,
    ): Promise<boolean> => {
      const persistedLaunchUserInput =
        options && "launchUserInput" in options
          ? normalizeWorkspaceServiceSkillOptionalText(options.launchUserInput)
          : undefined;

      const launchUserInput = resolveServiceSkillLaunchUserInput(
        input,
        options,
      );
      const prompt = composeServiceSkillPrompt({
        skill,
        slotValues,
        userInput: launchUserInput,
      });

      if (skill.runnerType !== "instant") {
        toast.info("这轮会先回到生成起第一版，后面再按设定继续带回来。");
      }

      let workspacePayload: WorkspaceEntryPayload;
      try {
        workspacePayload = await prepareServiceSkillWorkspacePayload(
          skill,
          prompt,
        );
      } catch (error) {
        toast.error(
          `准备技能工作区失败：${getWorkspaceServiceSkillErrorMessage(error)}`,
        );
        return false;
      }

      const entered = navigateToServiceSkillWorkspace(workspacePayload);
      if (!entered) {
        return false;
      }

      recordServiceSkillUsage({
        skillId: skill.id,
        runnerType: skill.runnerType,
        slotValues,
        ...(persistedLaunchUserInput
          ? {
              launchUserInput: persistedLaunchUserInput,
            }
          : {}),
      });
      setPendingServiceSkillLaunchInput(null);
      return true;
    },
    [
      input,
      navigateToServiceSkillWorkspace,
      setPendingServiceSkillLaunchInput,
      prepareServiceSkillWorkspacePayload,
      recordServiceSkillUsage,
    ],
  );

  const pendingServiceSkillLaunchForm = useMemo<A2UIResponse | null>(() => {
    if (!pendingServiceSkillLaunchInput) {
      return null;
    }

    return buildServiceSkillLaunchA2UIResponse(
      pendingServiceSkillLaunchInput.skill,
      {
        initialSlotValues: pendingServiceSkillLaunchInput.initialSlotValues,
        prefillHint: pendingServiceSkillLaunchInput.prefillHint,
        submitLabel: "继续当前结果",
        responseKey: pendingServiceSkillLaunchInput.requestKey,
      },
    );
  }, [pendingServiceSkillLaunchInput]);

  const pendingServiceSkillLaunchSource =
    useMemo<PendingA2UISource | null>(() => {
      if (!pendingServiceSkillLaunchInput) {
        return null;
      }

      return {
        kind: "service_skill",
        skillId: pendingServiceSkillLaunchInput.skill.id,
        requestKey: pendingServiceSkillLaunchInput.requestKey,
        messageId: undefined,
      };
    }, [pendingServiceSkillLaunchInput]);

  const handlePendingServiceSkillLaunchSubmit = useCallback(
    async (formData: A2UIFormData): Promise<boolean> => {
      if (!pendingServiceSkillLaunchInput) {
        return false;
      }

      const slotValues = readServiceSkillLaunchSlotValuesFromFormData(
        pendingServiceSkillLaunchInput.skill,
        formData,
      );
      const validation = validateServiceSkillSlotValues(
        pendingServiceSkillLaunchInput.skill,
        slotValues,
      );
      if (!validation.valid) {
        toast.info(
          `还差${validation.missing.map((slot) => slot.label).join("、")}，补齐后再继续。`,
        );
        return false;
      }

      const launched = await handleServiceSkillLaunch(
        pendingServiceSkillLaunchInput.skill,
        slotValues,
        {
          launchUserInput: pendingServiceSkillLaunchInput.launchUserInput,
        },
      );
      if (launched) {
        setPendingServiceSkillLaunchInput(null);
      }
      return launched;
    },
    [handleServiceSkillLaunch, pendingServiceSkillLaunchInput],
  );

  const clearPendingServiceSkillLaunch = useCallback(() => {
    setPendingServiceSkillLaunchInput(null);
  }, []);

  const handleServiceSkillSelect = useCallback(
    (skill: ServiceSkillHomeItem, options?: ServiceSkillSelectionOptions) => {
      const selectionPlan = buildServiceSkillSelectionPlan({
        skill,
        options,
        creationReplay,
        nextRequestCount: serviceSkillLaunchRequestCountRef.current + 1,
      });

      if (selectionPlan.kind === "launch") {
        void handleServiceSkillLaunch(skill, selectionPlan.slotValues, {
          launchUserInput: selectionPlan.launchUserInput,
        });
        return;
      }

      serviceSkillLaunchRequestCountRef.current += 1;
      setPendingServiceSkillLaunchInput(selectionPlan.pendingInput);
    },
    [creationReplay, handleServiceSkillLaunch],
  );

  const handleServiceSkillAutomationSetup = useCallback(
    async (skill: ServiceSkillHomeItem, slotValues: ServiceSkillSlotValues) => {
      if (!supportsServiceSkillLocalAutomation(skill)) {
        await handleServiceSkillLaunch(skill, slotValues);
        return;
      }

      if (!currentProjectId) {
        toast.error("缺少项目工作区，请先选择项目后再创建本地自动化任务。");
        return;
      }

      try {
        const automationThreadLineage = await resolveAutomationThreadLineage();
        if (!automationThreadLineage) {
          toast.error(
            t("scheduledTasks.editor.validation.threadLineageRequired"),
          );
          return;
        }

        const setupState = buildServiceSkillAutomationSetupState({
          skill,
          slotValues,
          input,
          workspaceId: currentProjectId,
          threadLineage: automationThreadLineage,
          modelSelection: {
            providerId: providerType ?? "",
            modelId: model ?? "",
          },
        });

        setAutomationDialogInitialValues(setupState.dialogInitialValues);
        setPendingServiceSkillAutomation(setupState.pendingAutomation);
        setPendingServiceSkillLaunchInput(null);
        setAutomationDialogOpen(true);
      } catch (error) {
        toast.error(
          `准备本地自动化任务失败：${getWorkspaceServiceSkillErrorMessage(
            error,
          )}`,
        );
      }
    },
    [
      currentProjectId,
      handleServiceSkillLaunch,
      input,
      model,
      providerType,
      resolveAutomationThreadLineage,
      setPendingServiceSkillLaunchInput,
      t,
    ],
  );

  const handleAutomationDialogOpenChange = useCallback((open: boolean) => {
    setAutomationDialogOpen(open);
    if (!open) {
      setAutomationDialogInitialValues(null);
      setPendingServiceSkillAutomation(null);
    }
  }, []);

  const handleAutomationDialogSubmit = useCallback(
    async (form: ScheduledTaskFormState) => {
      setAutomationJobSaving(true);
      try {
        const pendingLaunch = pendingServiceSkillAutomation;
        let automationContentId = currentContentId;

        if (
          shouldCreateServiceSkillAutomationContent({
            pendingAutomation: pendingLaunch,
            contentId: automationContentId,
          }) &&
          pendingLaunch
        ) {
          const createdContent = await createServiceSkillSeededContent(
            pendingLaunch.skill,
            form.projectId,
          );
          automationContentId = createdContent?.id ?? null;
        }
        const scheduledTaskRequest =
          buildServiceSkillScheduledTaskCreateRequest({
            pendingAutomation: pendingLaunch,
            form,
            contentId: automationContentId,
          });

        const createdTask =
          await scheduledTasksApi.create(scheduledTaskRequest);
        toast.success(`本地定时任务已创建：${createdTask.title}`);

        setAutomationDialogOpen(false);
        setAutomationDialogInitialValues(null);
        setPendingServiceSkillAutomation(null);

        if (!pendingLaunch) {
          return;
        }

        recordServiceSkillAutomationLink({
          skillId: pendingLaunch.usage.skillId,
          jobId: createdTask.id,
          jobName: createdTask.title,
        });
        recordServiceSkillUsage(pendingLaunch.usage);

        let workspacePayload: WorkspaceEntryPayload;
        try {
          workspacePayload = await prepareServiceSkillWorkspacePayload(
            pendingLaunch.skill,
            pendingLaunch.prompt,
            {
              contentId: automationContentId,
              projectId: form.projectId,
            },
          );
        } catch (error) {
          toast.error(
            `自动化任务已创建，但准备工作区失败：${getWorkspaceServiceSkillErrorMessage(
              error,
            )}`,
          );
          return;
        }

        const entered = navigateToServiceSkillWorkspace(workspacePayload);
        if (!entered) {
          toast.error("自动化已创建，但没能回到生成，请稍后手动打开。");
        }
      } catch (error) {
        toast.error(
          `创建本地自动化任务失败：${getWorkspaceServiceSkillErrorMessage(
            error,
          )}`,
        );
        throw error;
      } finally {
        setAutomationJobSaving(false);
      }
    },
    [
      createServiceSkillSeededContent,
      currentContentId,
      navigateToServiceSkillWorkspace,
      pendingServiceSkillAutomation,
      prepareServiceSkillWorkspacePayload,
      recordServiceSkillUsage,
    ],
  );

  return {
    pendingServiceSkillLaunchForm,
    pendingServiceSkillLaunchSource,
    automationDialogOpen,
    automationDialogInitialValues,
    automationThreadLineage:
      pendingServiceSkillAutomation?.threadLineage ?? null,
    automationJobSaving,
    handleServiceSkillSelect,
    handlePendingServiceSkillLaunchSubmit,
    clearPendingServiceSkillLaunch,
    handleServiceSkillLaunch,
    handleServiceSkillAutomationSetup,
    handleAutomationDialogOpenChange,
    handleAutomationDialogSubmit,
  };
}
