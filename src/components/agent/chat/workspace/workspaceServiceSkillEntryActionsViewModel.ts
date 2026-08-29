import type { ScheduledTaskCreateRequest } from "@/lib/api/scheduledTasks";
import {
  buildScheduledTaskCreateRequest,
  type ScheduledTaskFormState,
} from "@/components/scheduled-tasks/scheduledTaskViewModel";
import {
  composeServiceSkillPrompt,
  createDefaultServiceSkillSlotValues,
  validateServiceSkillSlotValues,
} from "../service-skills/promptComposer";
import {
  buildServiceSkillAutomationAgentTurnPayloadContext,
  buildServiceSkillScheduledTaskInitialForm,
} from "../service-skills/automationDraft";
import { resolveServiceSkillLaunchPrefill } from "../service-skills/serviceSkillLaunchPrefill";
import type {
  RecordServiceSkillUsageInput,
  ServiceSkillHomeItem,
  ServiceSkillSlotValues,
} from "../service-skills/types";
import type { CreationReplayMetadata } from "../utils/creationReplayMetadata";

export interface ServiceSkillLaunchUserInputOptions {
  launchUserInput?: string | null;
}

export interface ServiceSkillSelectionOptions {
  requestKey?: number | string;
  initialSlotValues?: ServiceSkillSlotValues;
  prefillHint?: string;
  launchUserInput?: string | null;
}

export interface PendingServiceSkillLaunchInputState {
  requestKey: string;
  skill: ServiceSkillHomeItem;
  initialSlotValues: ServiceSkillSlotValues;
  prefillHint?: string;
  launchUserInput?: string;
}

export interface PendingServiceSkillAutomationLaunch {
  skill: ServiceSkillHomeItem;
  prompt: string;
  slotValues: ServiceSkillSlotValues;
  userInput?: string;
  threadLineage: ServiceSkillAutomationThreadLineage;
  usage: RecordServiceSkillUsageInput;
}

export interface ServiceSkillAutomationThreadLineage {
  sessionId: string;
  threadId: string;
}

export interface ServiceSkillAutomationSetupState {
  dialogInitialValues: ScheduledTaskFormState;
  pendingAutomation: PendingServiceSkillAutomationLaunch;
}

export type ServiceSkillSelectionPlan =
  | {
      kind: "launch";
      slotValues: ServiceSkillSlotValues;
      launchUserInput?: string;
    }
  | {
      kind: "pending";
      pendingInput: PendingServiceSkillLaunchInputState;
    };

export interface BuildServiceSkillSelectionPlanInput {
  skill: ServiceSkillHomeItem;
  options?: ServiceSkillSelectionOptions;
  creationReplay?: CreationReplayMetadata;
  nextRequestCount: number;
}

export interface BuildServiceSkillAutomationSetupStateInput {
  skill: ServiceSkillHomeItem;
  slotValues: ServiceSkillSlotValues;
  input: string;
  workspaceId: string;
  threadLineage: ServiceSkillAutomationThreadLineage;
  modelSelection?: {
    providerId: string;
    modelId: string;
  };
}

export interface ShouldCreateServiceSkillAutomationContentInput {
  pendingAutomation?: PendingServiceSkillAutomationLaunch | null;
  contentId?: string | null;
}

export interface BuildServiceSkillAutomationSubmitRequestInput {
  pendingAutomation?: PendingServiceSkillAutomationLaunch | null;
  form: ScheduledTaskFormState;
  contentId?: string | null;
}

export function buildServiceSkillScheduledTaskCreateRequest(
  input: BuildServiceSkillAutomationSubmitRequestInput,
): ScheduledTaskCreateRequest {
  const request = buildScheduledTaskCreateRequest(input.form);
  if (!input.pendingAutomation) {
    return request;
  }
  const context = buildServiceSkillAutomationAgentTurnPayloadContext({
    skill: input.pendingAutomation.skill,
    slotValues: input.pendingAutomation.slotValues,
    userInput: input.pendingAutomation.userInput,
    contentId: input.contentId,
  });
  return {
    ...request,
    execution: {
      ...request.execution,
      requestMetadata: context.request_metadata ?? null,
    },
  };
}

export function getWorkspaceServiceSkillErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }
  return "请稍后重试";
}

export function normalizeWorkspaceServiceSkillOptionalText(
  value?: string | null,
): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }

  const normalized = value.trim();
  return normalized ? normalized : undefined;
}

export function resolveServiceSkillLaunchUserInput(
  currentInput: string,
  options?: ServiceSkillLaunchUserInputOptions,
): string | undefined {
  if (options && "launchUserInput" in options) {
    return normalizeWorkspaceServiceSkillOptionalText(options.launchUserInput);
  }

  return normalizeWorkspaceServiceSkillOptionalText(currentInput);
}

export function buildServiceSkillSelectionPlan({
  skill,
  options,
  creationReplay,
  nextRequestCount,
}: BuildServiceSkillSelectionPlanInput): ServiceSkillSelectionPlan {
  const replayPrefill = resolveServiceSkillLaunchPrefill({
    skill,
    creationReplay,
  });
  const launchUserInput =
    normalizeWorkspaceServiceSkillOptionalText(options?.launchUserInput) ??
    replayPrefill?.launchUserInput;
  const initialSlotValues = {
    ...createDefaultServiceSkillSlotValues(skill),
    ...(replayPrefill?.slotValues || {}),
    ...(options?.initialSlotValues || {}),
  };
  const validation = validateServiceSkillSlotValues(skill, initialSlotValues);

  if (skill.slotSchema.length === 0 || validation.valid) {
    return {
      kind: "launch",
      slotValues: initialSlotValues,
      launchUserInput,
    };
  }

  return {
    kind: "pending",
    pendingInput: {
      requestKey:
        options?.requestKey === undefined
          ? `${skill.id}:${nextRequestCount}`
          : `${skill.id}:${options.requestKey}`,
      skill,
      initialSlotValues,
      prefillHint: options?.prefillHint ?? replayPrefill?.hint,
      launchUserInput,
    },
  };
}

export function buildServiceSkillAutomationSetupState({
  skill,
  slotValues,
  input,
  workspaceId,
  threadLineage,
  modelSelection,
}: BuildServiceSkillAutomationSetupStateInput): ServiceSkillAutomationSetupState {
  const userInput = normalizeWorkspaceServiceSkillOptionalText(input);
  const prompt = composeServiceSkillPrompt({
    skill,
    slotValues,
    userInput,
  });

  return {
    dialogInitialValues: {
      ...buildServiceSkillScheduledTaskInitialForm({
        skill,
        slotValues,
        userInput,
        workspaceId,
      }),
      sourceThreadId: threadLineage.threadId,
      modelProviderId: modelSelection?.providerId.trim() || "",
      modelId: modelSelection?.modelId.trim() || "",
    },
    pendingAutomation: {
      skill,
      prompt,
      slotValues,
      userInput,
      threadLineage,
      usage: {
        skillId: skill.id,
        runnerType: skill.runnerType,
        slotValues,
      },
    },
  };
}

export function shouldCreateServiceSkillAutomationContent({
  pendingAutomation,
  contentId,
}: ShouldCreateServiceSkillAutomationContentInput): boolean {
  return Boolean(pendingAutomation && !contentId);
}
