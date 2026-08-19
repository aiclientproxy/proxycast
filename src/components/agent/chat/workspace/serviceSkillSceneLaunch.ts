import {
  getSkillCatalog,
  listSkillCatalogSceneEntries,
  type SkillCatalogSceneEntry,
} from "@/lib/api/skillCatalog";
import {
  listServiceSkills,
  type ServiceSkillItem,
} from "@/lib/api/serviceSkills";
import {
  matchesRuntimeSceneEntry,
  resolveRuntimeSceneSkillFromEntry,
} from "../service-skills/runtimeSceneBinding";
import type { ServiceSkillSlotValues } from "../service-skills/types";
import { composeServiceSkillPrompt } from "../service-skills/promptComposer";
import {
  buildRuntimeSceneGateRequest,
  formatRuntimeSceneGateValidationMessage,
  type RuntimeSceneGateRequest,
} from "./sceneSkillGate";

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }

  return value as Record<string, unknown>;
}

function normalizeOptionalText(value?: string | null): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }

  const normalized = value.trim();
  return normalized ? normalized : undefined;
}

interface ParsedRuntimeSceneCommand {
  sceneKey: string;
  userInput: string;
}

type ServiceSceneLaunchRequestContext = Record<string, unknown>;

interface ServiceSceneLaunchRequest {
  skill: ServiceSkillItem;
  sceneEntry: SkillCatalogSceneEntry;
  requestContext: ServiceSceneLaunchRequestContext;
  dispatchText?: string;
}

export class RuntimeSceneLaunchValidationError extends Error {
  gateRequest?: RuntimeSceneGateRequest;

  constructor(
    message: string,
    options?: { gateRequest?: RuntimeSceneGateRequest },
  ) {
    super(message);
    this.name = "RuntimeSceneLaunchValidationError";
    this.gateRequest = options?.gateRequest;
  }
}

export function parseRuntimeSceneCommand(
  rawText: string,
): ParsedRuntimeSceneCommand | null {
  const sceneMatch = rawText.trim().match(/^\/([^\s]+)\s*([\s\S]*)$/);
  if (!sceneMatch) {
    return null;
  }

  const [, sceneKey, userInput] = sceneMatch;
  return {
    sceneKey,
    userInput: userInput?.trim() || "",
  };
}

async function resolveRuntimeSceneSkillWithCatalogFallback(
  serviceSkills: ServiceSkillItem[],
  entry: SkillCatalogSceneEntry,
): Promise<ServiceSkillItem | null> {
  const matchedSkill = resolveRuntimeSceneSkillFromEntry(serviceSkills, entry);
  if (matchedSkill) {
    return matchedSkill;
  }

  try {
    return resolveRuntimeSceneSkillFromEntry(await listServiceSkills(), entry);
  } catch {
    return null;
  }
}

export { matchesRuntimeSceneEntry };

function extractFirstUrl(value: string): string | undefined {
  const match = value.match(/https?:\/\/[^\s<>"')）]+/i);
  return normalizeOptionalText(match?.[0]);
}

export function resolveSceneSlotValues(params: {
  skill: ServiceSkillItem;
  userInput: string;
  slotValueOverrides?: ServiceSkillSlotValues;
}): {
  resolvedSlotValues: ServiceSkillSlotValues;
  missingRequiredSlots: ServiceSkillItem["slotSchema"];
} {
  const { skill, userInput } = params;
  const normalizedUserInput = userInput.trim();

  const resolvedSlotValues: ServiceSkillSlotValues = {
    ...(params.slotValueOverrides || {}),
  };
  const singleUrlSlot =
    skill.slotSchema.filter((slot) => slot.type === "url").length === 1
      ? skill.slotSchema.find((slot) => slot.type === "url")
      : undefined;
  if (singleUrlSlot && !resolvedSlotValues[singleUrlSlot.key]) {
    const matchedUrl = extractFirstUrl(normalizedUserInput);
    if (matchedUrl) {
      resolvedSlotValues[singleUrlSlot.key] = matchedUrl;
    }
  }

  const unresolvedRequiredSlots = skill.slotSchema.filter(
    (slot) => slot.required && !resolvedSlotValues[slot.key],
  );
  if (unresolvedRequiredSlots.length === 1 && normalizedUserInput) {
    const [slot] = unresolvedRequiredSlots;
    if (slot && slot.type !== "url" && !resolvedSlotValues[slot.key]) {
      resolvedSlotValues[slot.key] = normalizedUserInput;
    }
  }

  const missingRequiredSlots = skill.slotSchema.filter(
    (slot) => slot.required && !resolvedSlotValues[slot.key],
  );

  return {
    resolvedSlotValues,
    missingRequiredSlots,
  };
}

function normalizeLocalServiceSceneExecutionKind(
  value?: string | null,
): "agent_turn" | "native_skill" | "automation_job" {
  if (value === "native_skill") {
    return "native_skill";
  }

  if (value === "automation_job") {
    return "automation_job";
  }

  return "agent_turn";
}

function buildServiceSceneLaunchRequestContext(params: {
  rawText: string;
  parsedCommand: ParsedRuntimeSceneCommand;
  sceneEntry: SkillCatalogSceneEntry;
  skill: ServiceSkillItem;
  slotValues: ServiceSkillSlotValues;
  projectId?: string | null;
  contentId?: string | null;
}): Record<string, unknown> {
  const { rawText, parsedCommand, sceneEntry, skill, projectId, contentId } =
    params;
  const slotValues =
    Object.keys(params.slotValues).length > 0 ? params.slotValues : undefined;

  return {
    kind: "local_service_skill",
    service_scene_run: {
      raw_text: rawText,
      user_input:
        normalizeOptionalText(parsedCommand.userInput) ??
        normalizeOptionalText(rawText) ??
        "",
      entry_id: sceneEntry.id,
      scene_key: sceneEntry.sceneKey,
      command_prefix: sceneEntry.commandPrefix,
      linked_skill_id: sceneEntry.linkedSkillId ?? skill.id,
      skill_locator:
        sceneEntry.skillLocator ?? {
          source: "catalog",
          name: skill.skillKey ?? sceneEntry.linkedSkillId ?? skill.id,
        },
      skill_id: skill.id,
      skill_key: skill.skillKey,
      skill_title: skill.title,
      skill_summary: skill.summary,
      runner_type: skill.runnerType,
      execution_kind: normalizeLocalServiceSceneExecutionKind(
        sceneEntry.executionKind ?? skill.defaultExecutorBinding,
      ),
      execution_location: "client_default",
      project_id: projectId ?? undefined,
      content_id: contentId ?? undefined,
      entry_source: "slash_scene_command",
      render_contract: sceneEntry.renderContract ?? undefined,
      slot_values: slotValues,
    },
  };
}

export function buildServiceSceneLaunchRequestMetadata(
  existingMetadata: Record<string, unknown> | undefined,
  requestContext: ServiceSceneLaunchRequestContext,
): Record<string, unknown> {
  const serviceSceneRun = asRecord(requestContext.service_scene_run);
  const existingHarness = asRecord(existingMetadata?.harness);

  return {
    ...(existingMetadata || {}),
    harness: {
      ...(existingHarness || {}),
      service_scene_launch: {
        kind:
          typeof requestContext.kind === "string"
            ? requestContext.kind
            : "local_service_skill",
        ...(serviceSceneRun
          ? {
              service_scene_run: serviceSceneRun,
            }
          : {
              request_context: requestContext,
            }),
      },
    },
  };
}

export async function resolveRuntimeSceneLaunchRequest(params: {
  rawText: string;
  serviceSkills: ServiceSkillItem[];
  projectId?: string | null;
  projectIdOverride?: string | null;
  contentId?: string | null;
  slotValueOverrides?: ServiceSkillSlotValues;
}): Promise<ServiceSceneLaunchRequest | null> {
  const parsedSceneCommand = parseRuntimeSceneCommand(params.rawText);
  if (!parsedSceneCommand) {
    return null;
  }

  const catalog = await getSkillCatalog();
  const sceneEntry = listSkillCatalogSceneEntries(catalog).find((entry) =>
    matchesRuntimeSceneEntry(entry, parsedSceneCommand.sceneKey),
  );
  if (!sceneEntry) {
    return null;
  }

  const matchedSkill = await resolveRuntimeSceneSkillWithCatalogFallback(
    params.serviceSkills,
    sceneEntry,
  );
  if (!matchedSkill) {
    return null;
  }

  const slotResolution = resolveSceneSlotValues({
    skill: matchedSkill,
    userInput: parsedSceneCommand.userInput,
    slotValueOverrides: params.slotValueOverrides,
  });
  const requireProject = Boolean(
    matchedSkill.readinessRequirements?.requiresProject &&
      !normalizeOptionalText(params.projectIdOverride ?? params.projectId),
  );
  const gateRequest = buildRuntimeSceneGateRequest({
    rawText: params.rawText,
    sceneEntry,
    skill: matchedSkill,
    missingSlots: slotResolution.missingRequiredSlots,
    requireProject,
  });
  if (gateRequest) {
    throw new RuntimeSceneLaunchValidationError(
      formatRuntimeSceneGateValidationMessage(gateRequest),
      { gateRequest },
    );
  }

  const requestContext = buildServiceSceneLaunchRequestContext({
    rawText: params.rawText,
    parsedCommand: parsedSceneCommand,
    sceneEntry,
    skill: matchedSkill,
    slotValues: slotResolution.resolvedSlotValues,
    projectId: params.projectIdOverride ?? params.projectId,
    contentId: params.contentId,
  });
  const dispatchText = composeServiceSkillPrompt({
    skill: matchedSkill,
    slotValues: slotResolution.resolvedSlotValues,
    userInput: normalizeOptionalText(parsedSceneCommand.userInput),
  });

  return {
    skill: matchedSkill,
    sceneEntry,
    requestContext,
    dispatchText,
  };
}
