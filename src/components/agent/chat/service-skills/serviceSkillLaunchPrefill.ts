import {
  getSlashEntryUsageMap,
  getSlashEntryUsageRecordKey,
} from "../skill-selection/slashEntryUsage";
import { resolveSceneSlotValues } from "../workspace/serviceSkillSceneLaunch";
import type { CreationReplayMetadata } from "../utils/creationReplayMetadata";
import { formatNumber } from "@/i18n/format";
import { agentZhCNResource as agentSourceResource } from "@/i18n/agentResources";
import {
  buildCreationReplaySlotPrefill,
  resolveCreationReplaySlotPrefillCopy,
  type CreationReplaySlotPrefillCopy,
  type ResolvedCreationReplaySlotPrefillCopy,
} from "./creationReplaySlotPrefill";
import { getServiceSkillUsageMap } from "./storage";
import type { ServiceSkillHomeItem, ServiceSkillSlotValues } from "./types";

export interface ServiceSkillLaunchPrefillResult {
  slotValues?: ServiceSkillSlotValues;
  launchUserInput?: string;
  hint?: string;
}

export interface ServiceSkillLaunchPrefillCopy {
  creationReplay?: CreationReplaySlotPrefillCopy;
  filledPrefix?: string;
  extraPrefix?: string;
  itemSeparator?: string;
  segmentSeparator?: string;
  formatFilledItems?: (visibleItems: string[], totalCount: number) => string;
  formatRecentServiceHint?: (skillTitle: string) => string;
  formatRecentSceneHint?: (sceneTitle: string) => string;
}

export interface ResolvedServiceSkillLaunchPrefillCopy {
  creationReplay: ResolvedCreationReplaySlotPrefillCopy;
  filledPrefix: string;
  extraPrefix: string;
  itemSeparator: string;
  segmentSeparator: string;
  formatFilledItems: (visibleItems: string[], totalCount: number) => string;
  formatRecentServiceHint: (skillTitle: string) => string;
  formatRecentSceneHint: (sceneTitle: string) => string;
}

type AgentSourceResourceKey = keyof typeof agentSourceResource;

function interpolateServiceSkillPrefillSourceTemplate(
  template: string,
  values?: Record<string, number | string>,
): string {
  return template.replace(/\{\{\s*(\w+)\s*\}\}/g, (match, name) => {
    const value = values?.[name];
    return value == null ? match : String(value);
  });
}

function translateServiceSkillPrefillSourceKey(
  key: string,
  values?: Record<string, number | string>,
): string {
  const template = agentSourceResource[key as AgentSourceResourceKey] ?? key;
  return interpolateServiceSkillPrefillSourceTemplate(template, values);
}

const SOURCE_SERVICE_SKILL_LAUNCH_PREFILL_COPY: Omit<
  ResolvedServiceSkillLaunchPrefillCopy,
  "creationReplay"
> = {
  filledPrefix: translateServiceSkillPrefillSourceKey(
    "skills.workspace.serviceSkill.prefill.filledPrefix",
  ),
  extraPrefix: translateServiceSkillPrefillSourceKey(
    "skills.workspace.serviceSkill.prefill.extraPrefix",
  ),
  itemSeparator: translateServiceSkillPrefillSourceKey(
    "skills.workspace.serviceSkill.prefill.itemSeparator",
  ),
  segmentSeparator: translateServiceSkillPrefillSourceKey(
    "skills.workspace.serviceSkill.prefill.segmentSeparator",
  ),
  formatFilledItems: (visibleItems, totalCount) =>
    formatSourceFilledItems(
      visibleItems,
      totalCount,
      translateServiceSkillPrefillSourceKey(
        "skills.workspace.serviceSkill.prefill.itemSeparator",
      ),
    ),
  formatRecentServiceHint: (skillTitle) =>
    translateServiceSkillPrefillSourceKey(
      "skills.workspace.serviceSkill.prefill.recentServiceHint",
      { title: skillTitle },
    ),
  formatRecentSceneHint: (sceneTitle) =>
    translateServiceSkillPrefillSourceKey(
      "skills.workspace.serviceSkill.prefill.recentSceneHint",
      { title: sceneTitle },
    ),
};

function formatSourceFilledItems(
  visibleItems: string[],
  totalCount: number,
  itemSeparator: string,
): string {
  const items = visibleItems.join(itemSeparator);
  if (visibleItems.length >= totalCount) {
    return items;
  }

  return translateServiceSkillPrefillSourceKey(
    "skills.workspace.serviceSkill.prefill.filledWithMore",
    {
      items,
      remaining: formatNumber(totalCount - visibleItems.length, {
        locale: "zh-CN",
      }),
      total: formatNumber(totalCount, { locale: "zh-CN" }),
    },
  );
}

export function resolveServiceSkillLaunchPrefillCopy(
  copy?: ServiceSkillLaunchPrefillCopy,
): ResolvedServiceSkillLaunchPrefillCopy {
  const itemSeparator =
    copy?.itemSeparator ??
    SOURCE_SERVICE_SKILL_LAUNCH_PREFILL_COPY.itemSeparator;
  return {
    ...SOURCE_SERVICE_SKILL_LAUNCH_PREFILL_COPY,
    ...(copy ?? {}),
    itemSeparator,
    creationReplay: resolveCreationReplaySlotPrefillCopy(copy?.creationReplay),
    formatFilledItems:
      copy?.formatFilledItems ??
      ((visibleItems, totalCount) =>
        formatSourceFilledItems(visibleItems, totalCount, itemSeparator)),
  };
}

function summarizePrefillValue(value: string, maxLength = 32): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }

  return `${normalized.slice(0, maxLength).trimEnd()}...`;
}

function normalizeOptionalText(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }

  const normalized = value.trim();
  return normalized ? normalized : undefined;
}

function compactSlotValues(
  slotValues: ServiceSkillSlotValues | undefined,
): ServiceSkillSlotValues | undefined {
  if (!slotValues) {
    return undefined;
  }

  const nextValues = Object.fromEntries(
    Object.entries(slotValues)
      .map(([key, value]) => [key.trim(), normalizeOptionalText(value)])
      .filter((entry): entry is [string, string] =>
        Boolean(entry[0] && entry[1]),
      ),
  );

  return Object.keys(nextValues).length > 0 ? nextValues : undefined;
}

export function buildServiceSkillLaunchPrefillSummary(params: {
  skill: Pick<ServiceSkillHomeItem, "slotSchema">;
  slotValues?: ServiceSkillSlotValues;
  launchUserInput?: string;
  limit?: number;
  copy?: ServiceSkillLaunchPrefillCopy;
}): string {
  const limit = params.limit ?? 2;
  const copy = resolveServiceSkillLaunchPrefillCopy(params.copy);
  const summaryItems = params.skill.slotSchema
    .map((slot) => {
      const value = compactSlotValues(params.slotValues)?.[slot.key];
      if (!value) {
        return null;
      }

      return `${slot.label}=${summarizePrefillValue(value)}`;
    })
    .filter((item): item is string => Boolean(item));
  const launchUserInput = normalizeOptionalText(params.launchUserInput);
  const hasDuplicatedLaunchUserInput = launchUserInput
    ? Object.values(compactSlotValues(params.slotValues) ?? {}).some(
        (value) => value === launchUserInput,
      )
    : false;

  if (summaryItems.length === 0 && !launchUserInput) {
    return "";
  }

  const segments: string[] = [];

  if (summaryItems.length > 0) {
    const visibleItems = summaryItems.slice(0, limit);
    segments.push(
      `${copy.filledPrefix}${copy.formatFilledItems(
        visibleItems,
        summaryItems.length,
      )}`,
    );
  }

  if (launchUserInput && !hasDuplicatedLaunchUserInput) {
    segments.push(
      `${copy.extraPrefix}${summarizePrefillValue(launchUserInput, 40)}`,
    );
  }

  return segments.join(copy.segmentSeparator);
}

function resolveRecentServiceSkillPrefill(
  skill: ServiceSkillHomeItem,
  copy: ServiceSkillLaunchPrefillCopy = {},
): ServiceSkillLaunchPrefillResult | undefined {
  const resolvedCopy = resolveServiceSkillLaunchPrefillCopy(copy);
  const recentUsage = getServiceSkillUsageMap().get(skill.id);
  const slotValues = compactSlotValues(recentUsage?.slotValues);
  const launchUserInput = normalizeOptionalText(recentUsage?.launchUserInput);
  if (!slotValues && !launchUserInput) {
    return undefined;
  }

  return {
    ...(slotValues ? { slotValues } : {}),
    ...(launchUserInput ? { launchUserInput } : {}),
    hint: resolvedCopy.formatRecentServiceHint(skill.title),
  };
}

function resolveRecentScenePrefill(
  skill: ServiceSkillHomeItem,
  copy: ServiceSkillLaunchPrefillCopy = {},
): ServiceSkillLaunchPrefillResult | undefined {
  const resolvedCopy = resolveServiceSkillLaunchPrefillCopy(copy);
  const sceneKey = normalizeOptionalText(skill.sceneBinding?.sceneKey);
  if (!sceneKey) {
    return undefined;
  }

  const recentSceneUsage = getSlashEntryUsageMap().get(
    getSlashEntryUsageRecordKey("scene", sceneKey),
  );
  const replayText = normalizeOptionalText(recentSceneUsage?.replayText);
  if (!replayText) {
    return undefined;
  }

  const slotValues = compactSlotValues(
    resolveSceneSlotValues({
      skill,
      userInput: replayText,
    }).resolvedSlotValues,
  );
  if (!slotValues) {
    return undefined;
  }

  return {
    slotValues,
    hint: resolvedCopy.formatRecentSceneHint(
      skill.sceneBinding?.commandPrefix || skill.title,
    ),
  };
}

export function resolveServiceSkillLaunchPrefill(params: {
  skill: ServiceSkillHomeItem | null;
  creationReplay?: CreationReplayMetadata;
  copy?: ServiceSkillLaunchPrefillCopy;
}): ServiceSkillLaunchPrefillResult | null {
  const { skill, creationReplay } = params;
  const copy = resolveServiceSkillLaunchPrefillCopy(params.copy);
  if (!skill) {
    return null;
  }

  const recentServicePrefill = resolveRecentServiceSkillPrefill(skill, copy);
  const recentScenePrefill = recentServicePrefill
    ? undefined
    : resolveRecentScenePrefill(skill, copy);
  const creationReplayPrefill = buildCreationReplaySlotPrefill(
    skill,
    creationReplay,
    copy.creationReplay,
  );
  const slotValues = compactSlotValues({
    ...(recentServicePrefill?.slotValues ||
      recentScenePrefill?.slotValues ||
      {}),
    ...(creationReplayPrefill?.slotValues || {}),
  });

  if (!slotValues && !creationReplayPrefill?.hint) {
    return null;
  }

  return {
    slotValues,
    ...(recentServicePrefill?.launchUserInput
      ? { launchUserInput: recentServicePrefill.launchUserInput }
      : {}),
    hint:
      creationReplayPrefill?.hint ??
      recentServicePrefill?.hint ??
      recentScenePrefill?.hint,
  };
}
