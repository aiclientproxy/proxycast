import { resolveServiceSkillEntryDescription } from "./entryAdapter";
import { formatNumber } from "@/i18n/format";
import { agentZhCNResource as agentSourceResource } from "@/i18n/agentResources";
import type {
  ServiceSkillHomeItem,
  ServiceSkillItem,
  ServiceSkillRunnerType,
  ServiceSkillTone,
  ServiceSkillType,
} from "./types";

const RUNNER_TONES: Record<ServiceSkillRunnerType, ServiceSkillTone> = {
  instant: "emerald",
  scheduled: "sky",
  managed: "amber",
};

export interface ServiceSkillPresentationCopy {
  runnerLabels?: Partial<Record<ServiceSkillRunnerType, string>>;
  runnerDescriptions?: Partial<Record<ServiceSkillRunnerType, string>>;
  actionLabels?: Partial<Record<ServiceSkillRunnerType, string>>;
  typeLabels?: Partial<Record<ServiceSkillType, string>>;
  fallbackRequiredInputs?: string;
  requiredPrefix?: string;
  outputPrefix?: string;
  requiredSlotActionLabel?: string;
  automationActionLabel?: string;
  outputProjectResource?: string;
  outputCurrentContent?: string;
  outputScheduled?: string;
  outputManaged?: string;
  outputDefault?: string;
  dependencyRequiresModel?: string;
  dependencyRequiresBrowser?: string;
  dependencyRequiresProject?: string;
  formatDependencyRequiresSkillKey?: (skillKey: string) => string;
  formatFactItems?: (visibleItems: string[], totalCount: number) => string;
}

interface ServiceSkillPresentationOptions {
  copy?: ServiceSkillPresentationCopy;
}

interface BuildServiceSkillCapabilityDescriptionOptions {
  includeSummary?: boolean;
  includeRequiredInputs?: boolean;
  includeOutputHint?: boolean;
  requiredInputsLimit?: number;
  copy?: ServiceSkillPresentationCopy;
}

type SummarizeServiceSkillRequiredInputsOptions =
  | number
  | {
      limit?: number;
      copy?: ServiceSkillPresentationCopy;
    };

type AgentSourceResourceKey = keyof typeof agentSourceResource;

function interpolateServiceSkillSourceTemplate(
  template: string,
  values?: Record<string, number | string>,
): string {
  return template.replace(/\{\{\s*(\w+)\s*\}\}/g, (match, name) => {
    const value = values?.[name];
    return value == null ? match : String(value);
  });
}

function translateServiceSkillSourceKey(
  key: string,
  values?: Record<string, number | string>,
): string {
  const template = agentSourceResource[key as AgentSourceResourceKey] ?? key;
  return interpolateServiceSkillSourceTemplate(template, values);
}

const SOURCE_SERVICE_SKILL_PRESENTATION_COPY: ServiceSkillPresentationCopy = {
  runnerLabels: {
    instant: translateServiceSkillSourceKey(
      "skills.workspace.serviceSkill.runner.instant.label",
    ),
    managed: translateServiceSkillSourceKey(
      "skills.workspace.serviceSkill.runner.managed.label",
    ),
    scheduled: translateServiceSkillSourceKey(
      "skills.workspace.serviceSkill.runner.scheduled.label",
    ),
  },
  runnerDescriptions: {
    instant: translateServiceSkillSourceKey(
      "skills.workspace.serviceSkill.runner.instant.description",
    ),
    managed: translateServiceSkillSourceKey(
      "skills.workspace.serviceSkill.runner.managed.description",
    ),
    scheduled: translateServiceSkillSourceKey(
      "skills.workspace.serviceSkill.runner.scheduled.description",
    ),
  },
  actionLabels: {
    instant: translateServiceSkillSourceKey(
      "skills.workspace.serviceSkill.action.instant",
    ),
    managed: translateServiceSkillSourceKey(
      "skills.workspace.serviceSkill.action.managed",
    ),
    scheduled: translateServiceSkillSourceKey(
      "skills.workspace.serviceSkill.action.scheduled",
    ),
  },
  typeLabels: {
    prompt: translateServiceSkillSourceKey(
      "skills.workspace.serviceSkill.type.prompt",
    ),
    service: translateServiceSkillSourceKey(
      "skills.workspace.serviceSkill.type.service",
    ),
  },
  fallbackRequiredInputs: translateServiceSkillSourceKey(
    "skills.workspace.serviceSkill.requiredInputs.empty",
  ),
  requiredPrefix: translateServiceSkillSourceKey(
    "skills.workspace.serviceSkill.requiredPrefix",
  ),
  outputPrefix: translateServiceSkillSourceKey(
    "skills.workspace.serviceSkill.outputPrefix",
  ),
  requiredSlotActionLabel: translateServiceSkillSourceKey(
    "skills.workspace.serviceSkill.action.requiredSlot",
  ),
  automationActionLabel: translateServiceSkillSourceKey(
    "skills.workspace.serviceSkill.action.automation",
  ),
  outputProjectResource: translateServiceSkillSourceKey(
    "skills.workspace.serviceSkill.output.projectResource",
  ),
  outputCurrentContent: translateServiceSkillSourceKey(
    "skills.workspace.serviceSkill.output.currentContent",
  ),
  outputScheduled: translateServiceSkillSourceKey(
    "skills.workspace.serviceSkill.output.scheduled",
  ),
  outputManaged: translateServiceSkillSourceKey(
    "skills.workspace.serviceSkill.output.managed",
  ),
  outputDefault: translateServiceSkillSourceKey(
    "skills.workspace.serviceSkill.output.default",
  ),
  dependencyRequiresModel: translateServiceSkillSourceKey(
    "skills.workspace.serviceSkill.dependency.model",
  ),
  dependencyRequiresBrowser: translateServiceSkillSourceKey(
    "skills.workspace.serviceSkill.dependency.browser",
  ),
  dependencyRequiresProject: translateServiceSkillSourceKey(
    "skills.workspace.serviceSkill.dependency.project",
  ),
  formatDependencyRequiresSkillKey: (skillKey) =>
    translateServiceSkillSourceKey(
      "skills.workspace.serviceSkill.dependency.skillKey",
      { skillKey },
    ),
  formatFactItems: (visibleItems, totalCount) => {
    const itemSeparator = translateServiceSkillSourceKey(
      "skills.workspace.serviceSkill.factItems.separator",
    );
    const items = visibleItems.join(itemSeparator);
    if (visibleItems.length >= totalCount) {
      return items;
    }

    return translateServiceSkillSourceKey(
      "skills.workspace.serviceSkill.factItems.withMore",
      {
        items,
        remaining: formatNumber(totalCount - visibleItems.length, {
          locale: "zh-CN",
        }),
        total: formatNumber(totalCount, { locale: "zh-CN" }),
      },
    );
  },
};

function resolveServiceSkillPresentationCopy(
  copy?: ServiceSkillPresentationCopy,
): ServiceSkillPresentationCopy {
  return {
    ...SOURCE_SERVICE_SKILL_PRESENTATION_COPY,
    ...(copy ?? {}),
    actionLabels: {
      ...SOURCE_SERVICE_SKILL_PRESENTATION_COPY.actionLabels,
      ...(copy?.actionLabels ?? {}),
    },
    runnerDescriptions: {
      ...SOURCE_SERVICE_SKILL_PRESENTATION_COPY.runnerDescriptions,
      ...(copy?.runnerDescriptions ?? {}),
    },
    runnerLabels: {
      ...SOURCE_SERVICE_SKILL_PRESENTATION_COPY.runnerLabels,
      ...(copy?.runnerLabels ?? {}),
    },
    typeLabels: {
      ...SOURCE_SERVICE_SKILL_PRESENTATION_COPY.typeLabels,
      ...(copy?.typeLabels ?? {}),
    },
  };
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(
    new Set(
      values.map((value) => value.trim()).filter((value) => value.length > 0),
    ),
  );
}

function hasRequiredSlots(item: Pick<ServiceSkillItem, "slotSchema">): boolean {
  return item.slotSchema.some((slot) => slot.required);
}

function summarizeServiceSkillFactItems(
  items: string[],
  limit = 2,
  copy: ServiceSkillPresentationCopy = {},
): string {
  const normalizedItems = uniqueStrings(items);
  if (normalizedItems.length === 0) {
    return "";
  }

  const visibleItems =
    normalizedItems.length <= limit
      ? normalizedItems
      : normalizedItems.slice(0, limit);
  const formatFactItems =
    resolveServiceSkillPresentationCopy(copy).formatFactItems;

  return formatFactItems?.(visibleItems, normalizedItems.length) ?? "";
}

function readServiceSkillBundleMetadata(
  item: Pick<ServiceSkillItem, "skillBundle">,
  key: string,
): string | null {
  const value = item.skillBundle?.metadata?.[key];
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function resolveServiceSkillTypeFromBundle(
  item: Pick<ServiceSkillItem, "skillBundle">,
): ServiceSkillType | null {
  const skillType = readServiceSkillBundleMetadata(item, "Lime_skill_type");
  if (
    skillType === "service" ||
    skillType === "prompt"
  ) {
    return skillType;
  }
  return null;
}

export function resolveServiceSkillType(
  item: Pick<
    ServiceSkillItem,
    | "skillType"
    | "defaultExecutorBinding"
    | "skillBundle"
  >,
): ServiceSkillType {
  if (item.skillType) {
    return item.skillType;
  }

  const skillTypeFromBundle = resolveServiceSkillTypeFromBundle(item);
  if (skillTypeFromBundle) {
    return skillTypeFromBundle;
  }

  return "service";
}

export function getServiceSkillTypeLabel(
  item: ServiceSkillItem,
  options: ServiceSkillPresentationOptions = {},
): string {
  const copy = resolveServiceSkillPresentationCopy(options.copy);
  const type = resolveServiceSkillType(item);
  return (
    copy.typeLabels?.[type] ?? `skills.workspace.serviceSkill.type.${type}`
  );
}

function normalizeRequiredInputsOptions(
  options: SummarizeServiceSkillRequiredInputsOptions = 2,
): { limit: number; copy: ServiceSkillPresentationCopy } {
  if (typeof options === "number") {
    return {
      limit: options,
      copy: resolveServiceSkillPresentationCopy(),
    };
  }

  return {
    limit: options.limit ?? 2,
    copy: resolveServiceSkillPresentationCopy(options.copy),
  };
}

export function summarizeServiceSkillRequiredInputs(
  item: Pick<ServiceSkillItem, "slotSchema">,
  options: SummarizeServiceSkillRequiredInputsOptions = 2,
): string {
  const { limit, copy } = normalizeRequiredInputsOptions(options);
  const requiredInputLabels = item.slotSchema
    .filter((slot) => slot.required)
    .map((slot) => slot.label);

  if (requiredInputLabels.length === 0) {
    return (
      copy.fallbackRequiredInputs ??
      "skills.workspace.serviceSkill.requiredInputs.empty"
    );
  }

  return summarizeServiceSkillFactItems(requiredInputLabels, limit, copy);
}

export function buildServiceSkillCapabilityDescription(
  item: Pick<
    ServiceSkillItem,
    "entryHint" | "summary" | "slotSchema" | "outputHint"
  >,
  options: BuildServiceSkillCapabilityDescriptionOptions = {},
): string {
  const segments: string[] = [];
  const copy = resolveServiceSkillPresentationCopy(options.copy);

  if (options.includeSummary ?? true) {
    segments.push(resolveServiceSkillEntryDescription(item));
  }

  if (options.includeRequiredInputs ?? true) {
    segments.push(
      `${
        copy.requiredPrefix ?? "skills.workspace.serviceSkill.requiredPrefix"
      }${summarizeServiceSkillRequiredInputs(item, {
        copy,
        limit: options.requiredInputsLimit,
      })}`,
    );
  }

  if (options.includeOutputHint ?? true) {
    segments.push(
      `${copy.outputPrefix ?? "skills.workspace.serviceSkill.outputPrefix"}${item.outputHint.trim()}`,
    );
  }

  return segments.join(" · ");
}

export function getServiceSkillRunnerLabel(
  item: ServiceSkillItem,
  options: ServiceSkillPresentationOptions = {},
): string {
  const copy = resolveServiceSkillPresentationCopy(options.copy);
  return (
    copy.runnerLabels?.[item.runnerType] ??
    `skills.workspace.serviceSkill.runner.${item.runnerType}.label`
  );
}

export function getServiceSkillRunnerTone(
  item: Pick<ServiceSkillItem, "runnerType">,
): ServiceSkillTone {
  return RUNNER_TONES[item.runnerType];
}

export function getServiceSkillRunnerDescription(
  item: ServiceSkillItem,
  options: ServiceSkillPresentationOptions = {},
): string {
  const copy = resolveServiceSkillPresentationCopy(options.copy);
  return (
    copy.runnerDescriptions?.[item.runnerType] ??
    `skills.workspace.serviceSkill.runner.${item.runnerType}.description`
  );
}

export function getServiceSkillActionLabel(
  item: ServiceSkillItem,
  options: ServiceSkillPresentationOptions = {},
): string {
  const copy = resolveServiceSkillPresentationCopy(options.copy);
  if (hasRequiredSlots(item)) {
    return (
      copy.requiredSlotActionLabel ??
      "skills.workspace.serviceSkill.action.requiredSlot"
    );
  }

  return (
    copy.actionLabels?.[item.runnerType] ??
    `skills.workspace.serviceSkill.action.${item.runnerType}`
  );
}

export function getServiceSkillOutputDestination(
  item: ServiceSkillItem,
  options: ServiceSkillPresentationOptions = {},
): string {
  const copy = resolveServiceSkillPresentationCopy(options.copy);
  if (item.outputDestination?.trim()) {
    return item.outputDestination.trim();
  }

  const outputDestinationFromBundle = readServiceSkillBundleMetadata(
    item,
    "Lime_output_destination",
  );
  if (outputDestinationFromBundle) {
    return outputDestinationFromBundle;
  }

  if (item.runnerType === "scheduled") {
    return (
      copy.outputScheduled ?? "skills.workspace.serviceSkill.output.scheduled"
    );
  }

  if (item.runnerType === "managed") {
    return copy.outputManaged ?? "skills.workspace.serviceSkill.output.managed";
  }

  return copy.outputDefault ?? "skills.workspace.serviceSkill.output.default";
}

export function listServiceSkillDependencies(
  item: ServiceSkillItem,
  options: ServiceSkillPresentationOptions = {},
): string[] {
  const requirements: string[] = [];
  const copy = resolveServiceSkillPresentationCopy(options.copy);

  if (item.readinessRequirements?.requiresModel) {
    requirements.push(
      copy.dependencyRequiresModel ??
        "skills.workspace.serviceSkill.dependency.model",
    );
  }
  if (item.readinessRequirements?.requiresBrowser) {
    requirements.push(
      copy.dependencyRequiresBrowser ??
        "skills.workspace.serviceSkill.dependency.browser",
    );
  }
  if (item.readinessRequirements?.requiresProject) {
    requirements.push(
      copy.dependencyRequiresProject ??
        "skills.workspace.serviceSkill.dependency.project",
    );
  }
  if (item.readinessRequirements?.requiresSkillKey) {
    requirements.push(
      copy.formatDependencyRequiresSkillKey?.(
        item.readinessRequirements.requiresSkillKey,
      ) ??
        `skills.workspace.serviceSkill.dependency.skillKey:${item.readinessRequirements.requiresSkillKey}`,
    );
  }

  return uniqueStrings([...(item.setupRequirements ?? []), ...requirements]);
}

export function getServiceSkillPrimaryActionLabel(
  skill: ServiceSkillHomeItem,
  canCreateAutomation: boolean,
  options: ServiceSkillPresentationOptions = {},
): string {
  const copy = resolveServiceSkillPresentationCopy(options.copy);
  if (canCreateAutomation) {
    return (
      copy.automationActionLabel ??
      copy.actionLabels?.scheduled ??
      "skills.workspace.serviceSkill.action.automation"
    );
  }
  return getServiceSkillActionLabel(skill, { copy });
}
