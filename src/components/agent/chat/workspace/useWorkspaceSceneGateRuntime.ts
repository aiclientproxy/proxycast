import { useCallback, useMemo, useState } from "react";
import { toast } from "sonner";
import { listProjects, type Project } from "@/lib/api/project";
import type { A2UIFormData, A2UIResponse } from "@/components/workspace/a2ui/types";
import type { PendingA2UISource } from "../types";
import {
  getSlashEntryUsageMap,
  getSlashEntryUsageRecordKey,
  recordSlashEntryUsage,
} from "../skill-selection/slashEntryUsage";
import { resolveServiceSkillLaunchPrefillCopy } from "../service-skills/serviceSkillLaunchPrefill";
import type { ServiceSkillHomeItem } from "../service-skills/types";
import { buildCreationReplaySlotPrefill } from "../service-skills/creationReplaySlotPrefill";
import type { CreationReplayMetadata } from "../utils/creationReplayMetadata";
import {
  buildRuntimeSceneGateA2UIForm,
  formatRuntimeSceneGateValidationMessage,
  readRuntimeSceneGateSubmission,
  resolveRuntimeSceneGateCopy,
  type RuntimeSceneGatePrefill,
  type RuntimeSceneGateRequest,
} from "./sceneSkillGate";
import {
  buildServiceSceneLaunchRequestMetadata,
  parseRuntimeSceneCommand,
  resolveSceneSlotValues,
  RuntimeSceneLaunchValidationError,
  resolveRuntimeSceneLaunchRequest,
} from "./serviceSkillSceneLaunch";

interface PendingSceneGateState {
  request: RuntimeSceneGateRequest;
  projects: Project[];
  prefill?: RuntimeSceneGatePrefill;
}

interface ResumeSceneGateInput {
  rawText: string;
  requestMetadata: Record<string, unknown>;
}

interface UseWorkspaceSceneGateRuntimeParams {
  serviceSkills: ServiceSkillHomeItem[];
  projectId?: string | null;
  contentId?: string | null;
  creationReplay?: CreationReplayMetadata;
  applyProjectSelection?: (projectId?: string | null) => void;
  resumeSceneGate: (input: ResumeSceneGateInput) => Promise<boolean>;
}

const SOURCE_SERVICE_SKILL_LAUNCH_PREFILL_COPY =
  resolveServiceSkillLaunchPrefillCopy();
const SOURCE_RUNTIME_SCENE_GATE_COPY = resolveRuntimeSceneGateCopy();

function normalizeOptionalText(value?: string | null): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }

  const normalized = value.trim();
  return normalized ? normalized : undefined;
}

function buildSceneGateReplayText(params: {
  rawText: string;
  slotValues: Record<string, string>;
}): string | undefined {
  const userInput = normalizeOptionalText(
    parseRuntimeSceneCommand(params.rawText)?.userInput,
  );
  if (userInput) {
    return userInput;
  }

  const fallbackValues = Object.values(params.slotValues)
    .map((value) => normalizeOptionalText(value))
    .filter((value): value is string => Boolean(value));

  return fallbackValues.length > 0 ? fallbackValues.join(" ") : undefined;
}

function compactSceneGateSlotValues(
  slotValues: Record<string, string> | undefined,
): Record<string, string> | undefined {
  if (!slotValues) {
    return undefined;
  }

  const nextValues = Object.fromEntries(
    Object.entries(slotValues)
      .map(([key, value]) => [key, normalizeOptionalText(value)])
      .filter((entry): entry is [string, string] => Boolean(entry[1])),
  );

  return Object.keys(nextValues).length > 0 ? nextValues : undefined;
}

function resolveRecentSceneGateReplayPrefill(params: {
  request: RuntimeSceneGateRequest;
  skill: ServiceSkillHomeItem;
}): RuntimeSceneGatePrefill | undefined {
  const recentRecord = getSlashEntryUsageMap().get(
    getSlashEntryUsageRecordKey("scene", params.request.sceneKey),
  );
  const replayText = normalizeOptionalText(recentRecord?.replayText);
  if (!replayText) {
    return undefined;
  }

  const slotValues = compactSceneGateSlotValues(
    resolveSceneSlotValues({
      skill: params.skill,
      userInput: replayText,
    }).resolvedSlotValues,
  );
  if (!slotValues) {
    return undefined;
  }

  return {
    slotValues,
    hint: SOURCE_SERVICE_SKILL_LAUNCH_PREFILL_COPY.formatRecentSceneHint(
      params.request.commandPrefix,
    ),
  };
}

export function resolveSceneGatePrefill(params: {
  request: RuntimeSceneGateRequest;
  serviceSkills: ServiceSkillHomeItem[];
  creationReplay?: CreationReplayMetadata;
  projectId?: string | null;
}): RuntimeSceneGatePrefill | undefined {
  const matchedSkill = params.serviceSkills.find((skill) => {
    const normalizedSkillId = skill.id.trim();
    const normalizedSkillKey = skill.skillKey?.trim();
    return (
      normalizedSkillId === params.request.skillId ||
      normalizedSkillKey === params.request.skillId ||
      normalizedSkillKey === params.request.sceneKey
    );
  });

  const recentScenePrefill = matchedSkill
    ? resolveRecentSceneGateReplayPrefill({
        request: params.request,
        skill: matchedSkill,
      })
    : undefined;
  const replayPrefill = matchedSkill
    ? buildCreationReplaySlotPrefill(matchedSkill, params.creationReplay)
    : null;
  const preferredProjectId =
    normalizeOptionalText(params.projectId) ||
    normalizeOptionalText(params.creationReplay?.source.project_id);

  if (!recentScenePrefill && !replayPrefill && !preferredProjectId) {
    return undefined;
  }

  const slotValues = compactSceneGateSlotValues({
    ...(recentScenePrefill?.slotValues || {}),
    ...(replayPrefill?.slotValues || {}),
  });

  return {
    slotValues,
    hint: replayPrefill?.hint ?? recentScenePrefill?.hint,
    projectId: preferredProjectId,
  };
}

export function useWorkspaceSceneGateRuntime({
  serviceSkills,
  projectId,
  contentId,
  creationReplay,
  applyProjectSelection,
  resumeSceneGate,
}: UseWorkspaceSceneGateRuntimeParams): {
  pendingSceneGateForm: A2UIResponse | null;
  pendingSceneGateSource: PendingA2UISource | null;
  openRuntimeSceneGate: (request: RuntimeSceneGateRequest) => Promise<void>;
  handleSceneGateSubmit: (formData: A2UIFormData) => Promise<boolean>;
  clearRuntimeSceneGate: () => void;
} {
  const [pendingSceneGate, setPendingSceneGate] =
    useState<PendingSceneGateState | null>(null);

  const clearRuntimeSceneGate = useCallback(() => {
    setPendingSceneGate(null);
  }, []);

  const openRuntimeSceneGate = useCallback(
    async (request: RuntimeSceneGateRequest) => {
      let projects: Project[] = [];
      if (request.fields.some((field) => field.kind === "project")) {
        try {
          projects = await listProjects();
        } catch {
          projects = [];
        }
      }

      setPendingSceneGate({
        request,
        projects,
        prefill: resolveSceneGatePrefill({
          request,
          serviceSkills,
          creationReplay,
          projectId,
        }),
      });
    },
    [creationReplay, projectId, serviceSkills],
  );

  const pendingSceneGateForm = useMemo(() => {
    if (!pendingSceneGate) {
      return null;
    }

    return buildRuntimeSceneGateA2UIForm({
      request: pendingSceneGate.request,
      projects: pendingSceneGate.projects,
      prefill: pendingSceneGate.prefill,
    });
  }, [pendingSceneGate]);

  const pendingSceneGateSource = useMemo<PendingA2UISource | null>(() => {
    if (!pendingSceneGate) {
      return null;
    }

    return {
      kind: "scene_gate",
      gateKey: pendingSceneGate.request.gateKey,
      sceneKey: pendingSceneGate.request.sceneKey,
      messageId: undefined,
    };
  }, [pendingSceneGate]);

  const handleSceneGateSubmit = useCallback(
    async (formData: A2UIFormData) => {
      if (!pendingSceneGate) {
        return false;
      }

      const submission = readRuntimeSceneGateSubmission({
        request: pendingSceneGate.request,
        formData,
        prefill: pendingSceneGate.prefill,
      });
      if (submission.missingFieldLabels.length > 0) {
        toast.info(
          formatRuntimeSceneGateValidationMessage(pendingSceneGate.request),
        );
        return false;
      }

      try {
        const nextProjectId = submission.projectId ?? projectId;
        const sceneLaunchRequest = await resolveRuntimeSceneLaunchRequest({
          rawText: pendingSceneGate.request.rawText,
          serviceSkills,
          projectId,
          projectIdOverride: nextProjectId,
          contentId,
          slotValueOverrides: submission.slotValues,
        });

        if (!sceneLaunchRequest) {
          toast.error(SOURCE_RUNTIME_SCENE_GATE_COPY.unavailableMessage);
          return false;
        }

        if (submission.projectId) {
          applyProjectSelection?.(submission.projectId);
        }

        const started = await resumeSceneGate({
          rawText: pendingSceneGate.request.rawText,
          requestMetadata: buildServiceSceneLaunchRequestMetadata(
            undefined,
            sceneLaunchRequest.requestContext,
          ),
        });
        if (!started) {
          return false;
        }

        recordSlashEntryUsage({
          kind: "scene",
          entryId: sceneLaunchRequest.sceneEntry.sceneKey,
          replayText: buildSceneGateReplayText({
            rawText: pendingSceneGate.request.rawText,
            slotValues: submission.slotValues,
          }),
        });
        setPendingSceneGate(null);
        return true;
      } catch (error) {
        if (
          error instanceof RuntimeSceneLaunchValidationError &&
          error.gateRequest
        ) {
          toast.info(
            formatRuntimeSceneGateValidationMessage(error.gateRequest),
          );
          await openRuntimeSceneGate(error.gateRequest);
          return false;
        }

        const message =
          error instanceof Error && error.message.trim()
            ? SOURCE_RUNTIME_SCENE_GATE_COPY.formatLaunchFailed(error.message)
            : SOURCE_RUNTIME_SCENE_GATE_COPY.launchFailedFallback;
        toast.error(message);
        return false;
      }
    },
    [
      applyProjectSelection,
      contentId,
      openRuntimeSceneGate,
      pendingSceneGate,
      projectId,
      resumeSceneGate,
      serviceSkills,
    ],
  );

  return {
    pendingSceneGateForm,
    pendingSceneGateSource,
    openRuntimeSceneGate,
    handleSceneGateSubmit,
    clearRuntimeSceneGate,
  };
}
