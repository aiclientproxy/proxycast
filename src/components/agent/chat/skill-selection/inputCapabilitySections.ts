import type { Character } from "@/lib/api/projectMemory";
import type { Skill } from "@/lib/api/skills";
import { buildServiceSkillRecommendationBuckets } from "@/components/agent/chat/service-skills/recommendedServiceSkills";
import { buildServiceSkillCapabilityDescription } from "@/components/agent/chat/service-skills/skillPresentation";
import {
  buildServiceSkillLaunchPrefillSummary,
  resolveServiceSkillLaunchPrefill,
} from "@/components/agent/chat/service-skills/serviceSkillLaunchPrefill";
import type {
  ServiceSkillGroup,
  ServiceSkillHomeItem,
  ServiceSkillSlotValues,
} from "@/components/agent/chat/service-skills/types";
import { resolveMentionCommandPrefillReplayText } from "@/components/agent/chat/utils/mentionCommandReplayText";
import type { SlashCommandDefinition } from "../commands";
import type {
  BuiltinInputCommand,
  RuntimeSceneSlashCommand,
} from "./builtinCommands";
import {
  getMentionEntryUsageMap,
  getMentionEntryUsageRecordKey,
} from "./mentionEntryUsage";
import {
  getSlashEntryUsageMap,
  getSlashEntryUsageRecordKey,
  type SlashEntryUsageKind,
} from "./slashEntryUsage";
import {
  buildCuratedTaskRecentUsageDescription,
  buildCuratedTaskCapabilityDescription,
  filterCuratedTaskTemplates,
  listFeaturedHomeCuratedTaskTemplates,
  listCuratedTaskTemplates,
  resolveCuratedTaskTemplateLaunchPrefill,
  type CuratedTaskInputValues,
  type CuratedTaskTemplateCopy,
  type CuratedTaskTemplateItem,
} from "../utils/curatedTaskTemplates";
import { listCuratedTaskRecommendationSignals } from "../utils/curatedTaskRecommendationSignals";
import {
  buildInstalledSkillCapabilityDescription,
  type InstalledSkillPresentationCopy,
} from "@/components/skills/installedSkillPresentation";
import {
  extractCuratedTaskReferenceMemoryIds,
  mergeCuratedTaskReferenceEntries,
  normalizeCuratedTaskReferenceMemoryIds,
  type CuratedTaskReferenceEntry,
} from "../utils/curatedTaskReferenceSelection";
import { buildCuratedTaskResultBaselineSnapshot } from "../utils/curatedTaskResultBaseline";
import { buildReviewFeedbackProjection } from "../utils/reviewFeedbackProjection";
import type {
  InputbarPluginCapability,
  InputbarPluginSkillCapability,
} from "../components/Inputbar/pluginInputCapability";
import { buildMentionPluginCapabilityItems } from "./pluginCapabilitySections";

const FEATURED_SERVICE_SKILL_LIMIT = 4;
const RECENT_REPLAY_TEXT_PREVIEW_LIMIT = 48;

type InputCapabilityIcon =
  | "blocks"
  | "command"
  | "image-plus"
  | "sparkles"
  | "user"
  | "zap";

type InputCapabilityBase = {
  key: string;
  title: string;
  description: string;
  icon: InputCapabilityIcon;
  iconClassName: string;
  kindLabel?: string;
};

export type InputCapabilityDescriptor =
  | (InputCapabilityBase & {
      kind: "builtin_command";
      command: BuiltinInputCommand;
      replayText?: string;
    })
  | (InputCapabilityBase & {
      kind: "service_skill";
      skill: ServiceSkillHomeItem;
    })
  | (InputCapabilityBase & {
      kind: "slash_command";
      command: SlashCommandDefinition;
      replayText?: string;
    })
  | (InputCapabilityBase & {
      kind: "scene_command";
      command: RuntimeSceneSlashCommand;
      replayText?: string;
    })
  | (InputCapabilityBase & {
      kind: "curated_task";
      task: CuratedTaskTemplateItem;
      launchInputValues?: CuratedTaskInputValues;
      referenceMemoryIds?: string[];
      referenceEntries?: CuratedTaskReferenceEntry[];
      launcherPrefillHint?: string;
    })
  | (InputCapabilityBase & {
      kind: "plugin";
      plugin: InputbarPluginCapability;
      skill?: InputbarPluginSkillCapability;
      disabled?: boolean;
    })
  | (InputCapabilityBase & {
      kind: "character";
      character: Character;
    })
  | (InputCapabilityBase & {
      kind: "installed_skill";
      skill: Skill;
      replayText?: string;
    })
  | (InputCapabilityBase & {
      kind: "available_skill";
      skill: Skill;
    });

export interface InputCapabilitySection {
  key: string;
  heading: string;
  items: InputCapabilityDescriptor[];
  banner?: {
    badge?: string;
    title: string;
    summary: string;
    footnote?: string;
    actionLabel?: string;
    actionItemKey?: string;
  };
}

interface MentionServiceSkillGroup {
  key: string;
  title: string;
  sort: number;
  skills: ServiceSkillHomeItem[];
}

interface RecentSlashEntry {
  key: string;
  kind: SlashEntryUsageKind | "curated_task";
  kindLabel?: string;
  title: string;
  description: string;
  usedAt: number;
  commandPrefix?: string;
  replayText?: string;
  taskId?: string;
}

interface RecentMentionEntry {
  key: string;
  kind: "builtin_command" | "service_skill";
  kindLabel?: string;
  title: string;
  description: string;
  usedAt: number;
  replayText?: string;
  commandKey?: string;
  commandPrefix?: string;
  skillId?: string;
}

type InputCommandCapabilityGroupKey =
  | "search_read"
  | "generate_expression"
  | "media_transform"
  | "preview_publish"
  | "browser_execution"
  | "other";

type SlashCommandSectionGroupKey =
  | "workspace_action"
  | "prompt_action"
  | "status_help";

interface InputCommandSectionMetaBase {
  key: string;
  icon: InputCapabilityIcon;
  iconClassName: string;
  order: number;
}

interface InputCommandSectionMeta extends InputCommandSectionMetaBase {
  heading: string;
  kindLabel: string;
}

interface InputCapabilitySectionMetaCopy {
  heading: string;
  kindLabel: string;
}

export interface InputCapabilitySectionsCopy {
  baseline: {
    formatBaseline: (title: string) => string;
    formatDestinationHighlight: (value: string) => string;
    formatOperatingActionHighlight: (value: string) => string;
    formatStatusHighlight: (value: string) => string;
  };
  headings: {
    availableSkills: string;
    characters: string;
    featuredServiceSkills: string;
    plugins: string;
    installedSkills: string;
    installedSkillsEmpty: string;
    recentContinuations: string;
    recentContinuationsEmpty: string;
    recentMention: string;
    recentOperations: string;
    resultTemplates: string;
    resultTemplatesEmpty: string;
    unsupported: string;
  };
  inputCommandGroups: Record<
    InputCommandCapabilityGroupKey,
    InputCapabilitySectionMetaCopy
  >;
  installedSkillPresentation: InstalledSkillPresentationCopy;
  mentionRegistry: {
    badge: string;
    charactersFootnote: string;
    installedSkillsFootnote: string;
    serviceSkillsFootnote: string;
    summaryDefault: string;
    summaryWithRecent: string;
    titleDefault: string;
    titleWithRecent: string;
  };
  reviewBanner: {
    formatActionLabel: (title: string) => string;
    formatFootnote: (titles: string[]) => string;
    formatTitle: (title: string) => string;
    titleSeparator: string;
  };
  slashCommandGroups: Record<
    SlashCommandSectionGroupKey,
    InputCapabilitySectionMetaCopy
  >;
  formatRecentInputDescription: (preview: string) => string;
}

export type InputCapabilitySectionsCopyTranslator = (
  key: string,
  values?: Record<string, number | string>,
) => string;

export function buildInputCapabilitySectionsCopy(
  translate: InputCapabilitySectionsCopyTranslator,
): InputCapabilitySectionsCopy {
  return {
    baseline: {
      formatBaseline: (title) =>
        translate("inputCapabilities.baseline.title", { title }),
      formatDestinationHighlight: (value) =>
        translate("inputCapabilities.baseline.destination", { value }),
      formatOperatingActionHighlight: (value) =>
        translate("inputCapabilities.baseline.operatingAction", { value }),
      formatStatusHighlight: (value) =>
        translate("inputCapabilities.baseline.status", { value }),
    },
    formatRecentInputDescription: (preview) =>
      translate("inputCapabilities.recentInput", { preview }),
    headings: {
      availableSkills: translate("inputCapabilities.heading.availableSkills"),
      plugins: translate("inputCapabilities.heading.plugins"),
      characters: translate("inputCapabilities.heading.characters"),
      featuredServiceSkills: translate(
        "inputCapabilities.heading.featuredServiceSkills",
      ),
      installedSkills: translate("inputCapabilities.heading.installedSkills"),
      installedSkillsEmpty: translate(
        "inputCapabilities.heading.installedSkillsEmpty",
      ),
      recentContinuations: translate(
        "inputCapabilities.heading.recentContinuations",
      ),
      recentContinuationsEmpty: translate(
        "inputCapabilities.heading.recentContinuationsEmpty",
      ),
      recentMention: translate("inputCapabilities.heading.recentMention"),
      recentOperations: translate("inputCapabilities.heading.recentOperations"),
      resultTemplates: translate("inputCapabilities.heading.resultTemplates"),
      resultTemplatesEmpty: translate(
        "inputCapabilities.heading.resultTemplatesEmpty",
      ),
      unsupported: translate("inputCapabilities.heading.unsupported"),
    },
    inputCommandGroups: {
      browser_execution: {
        heading: translate("inputCapabilities.inputGroup.browserExecution"),
        kindLabel: translate("inputCapabilities.inputGroup.browserExecution"),
      },
      generate_expression: {
        heading: translate("inputCapabilities.inputGroup.generateExpression"),
        kindLabel: translate("inputCapabilities.inputGroup.generateExpression"),
      },
      media_transform: {
        heading: translate("inputCapabilities.inputGroup.mediaTransform"),
        kindLabel: translate("inputCapabilities.inputGroup.mediaTransform"),
      },
      other: {
        heading: translate("inputCapabilities.inputGroup.other"),
        kindLabel: translate("inputCapabilities.inputGroup.other"),
      },
      preview_publish: {
        heading: translate("inputCapabilities.inputGroup.previewPublish"),
        kindLabel: translate("inputCapabilities.inputGroup.previewPublish"),
      },
      search_read: {
        heading: translate("inputCapabilities.inputGroup.searchRead"),
        kindLabel: translate("inputCapabilities.inputGroup.searchRead"),
      },
    },
    installedSkillPresentation: {
      defaultPromise: translate(
        "skills.workspace.installedSkill.defaultPromise",
      ),
      fallbackRequiredInputs: translate(
        "skills.workspace.installedSkill.fallbackRequiredInputs",
      ),
      fallbackOutputHint: translate(
        "skills.workspace.installedSkill.fallbackOutputHint",
      ),
      requiredPrefix: translate(
        "skills.workspace.installedSkill.requiredPrefix",
      ),
      outputPrefix: translate("skills.workspace.installedSkill.outputPrefix"),
    },
    mentionRegistry: {
      badge: translate("inputCapabilities.mentionRegistry.badge"),
      charactersFootnote: translate(
        "inputCapabilities.mentionRegistry.charactersFootnote",
      ),
      installedSkillsFootnote: translate(
        "inputCapabilities.mentionRegistry.installedSkillsFootnote",
      ),
      serviceSkillsFootnote: translate(
        "inputCapabilities.mentionRegistry.serviceSkillsFootnote",
      ),
      summaryDefault: translate(
        "inputCapabilities.mentionRegistry.summaryDefault",
      ),
      summaryWithRecent: translate(
        "inputCapabilities.mentionRegistry.summaryWithRecent",
      ),
      titleDefault: translate("inputCapabilities.mentionRegistry.titleDefault"),
      titleWithRecent: translate(
        "inputCapabilities.mentionRegistry.titleWithRecent",
      ),
    },
    reviewBanner: {
      formatActionLabel: (title) =>
        translate("inputCapabilities.review.action", { title }),
      formatFootnote: (titles) =>
        translate("inputCapabilities.review.footnote", {
          titles: titles.join(
            translate("inputCapabilities.review.titleSeparator"),
          ),
        }),
      formatTitle: (title) =>
        translate("inputCapabilities.review.title", { title }),
      titleSeparator: translate("inputCapabilities.review.titleSeparator"),
    },
    slashCommandGroups: {
      prompt_action: {
        heading: translate("inputCapabilities.slashGroup.promptAction"),
        kindLabel: translate("inputCapabilities.slashGroup.promptAction"),
      },
      status_help: {
        heading: translate("inputCapabilities.slashGroup.statusHelp"),
        kindLabel: translate("inputCapabilities.slashGroup.statusHelp"),
      },
      workspace_action: {
        heading: translate("inputCapabilities.slashGroup.workspaceAction"),
        kindLabel: translate("inputCapabilities.slashGroup.workspaceAction"),
      },
    },
  };
}

const INPUT_COMMAND_SECTION_META: Record<
  InputCommandCapabilityGroupKey,
  InputCommandSectionMetaBase
> = {
  search_read: {
    key: "search-read",
    icon: "command",
    iconClassName: "mr-2 h-4 w-4 text-sky-600",
    order: 10,
  },
  generate_expression: {
    key: "generate-expression",
    icon: "image-plus",
    iconClassName: "mr-2 h-4 w-4 text-amber-600",
    order: 20,
  },
  media_transform: {
    key: "media-transform",
    icon: "sparkles",
    iconClassName: "mr-2 h-4 w-4 text-cyan-600",
    order: 30,
  },
  preview_publish: {
    key: "preview-publish",
    icon: "zap",
    iconClassName: "mr-2 h-4 w-4 text-rose-600",
    order: 40,
  },
  browser_execution: {
    key: "browser-execution",
    icon: "command",
    iconClassName: "mr-2 h-4 w-4 text-slate-600",
    order: 50,
  },
  other: {
    key: "other-capabilities",
    icon: "command",
    iconClassName: "mr-2 h-4 w-4 text-primary",
    order: 90,
  },
};

const INPUT_COMMAND_GROUP_BY_KEY: Record<
  string,
  InputCommandCapabilityGroupKey
> = {
  modal_resource_search: "search_read",
  research: "search_read",
  deep_search: "search_read",
  research_report: "search_read",
  competitor_research: "search_read",
  site_search: "search_read",
  read_pdf: "search_read",
  file_read_runtime: "search_read",
  knowledge_pack: "search_read",
  knowledge_settle: "search_read",
  summary: "search_read",
  translation: "search_read",
  analysis: "search_read",
  logo_decomposition: "search_read",
  web_scrape: "search_read",
  webpage_read: "search_read",
  url_parse: "search_read",
  image_generate: "generate_expression",
  image_generate_nanobanana_pro: "generate_expression",
  image_storyboard: "generate_expression",
  cover_generate: "generate_expression",
  poster_generate: "generate_expression",
  video_generate: "generate_expression",
  presentation_generate: "generate_expression",
  form_generate: "generate_expression",
  webpage_generate: "generate_expression",
  broadcast_generate: "generate_expression",
  writing_runtime: "generate_expression",
  image_edit: "media_transform",
  image_variation: "media_transform",
  voice_runtime: "media_transform",
  growth_runtime: "other",
  transcription_generate: "media_transform",
  typesetting: "media_transform",
  channel_preview_runtime: "preview_publish",
  upload_runtime: "preview_publish",
  publish_runtime: "preview_publish",
  publish_compliance: "preview_publish",
  browser_runtime: "browser_execution",
  code_runtime: "browser_execution",
};

const SLASH_COMMAND_SECTION_META: Record<
  SlashCommandSectionGroupKey,
  InputCommandSectionMetaBase
> = {
  workspace_action: {
    key: "workspace-action",
    icon: "command",
    iconClassName: "mr-2 h-4 w-4 text-emerald-600",
    order: 10,
  },
  prompt_action: {
    key: "prompt-action",
    icon: "sparkles",
    iconClassName: "mr-2 h-4 w-4 text-amber-600",
    order: 20,
  },
  status_help: {
    key: "status-help",
    icon: "zap",
    iconClassName: "mr-2 h-4 w-4 text-slate-600",
    order: 30,
  },
};

interface BuildInputCapabilitySectionsParams {
  mode: "mention" | "slash";
  mentionQuery: string;
  builtinCommands: BuiltinInputCommand[];
  slashCommands: SlashCommandDefinition[];
  sceneCommands: RuntimeSceneSlashCommand[];
  mentionServiceSkills: ServiceSkillHomeItem[];
  pluginSuggestions?: readonly InputbarPluginCapability[];
  serviceSkillGroups?: ServiceSkillGroup[];
  filteredCharacters: Character[];
  installedSkills: Skill[];
  availableSkills: Skill[];
  projectId?: string | null;
  sessionId?: string | null;
  referenceEntries?: CuratedTaskReferenceEntry[];
  curatedTaskTemplateCopy?: CuratedTaskTemplateCopy;
  inputCapabilityCopy: InputCapabilitySectionsCopy;
}

function compareRecentSlashEntries(
  left: RecentSlashEntry,
  right: RecentSlashEntry,
): number {
  if (left.usedAt !== right.usedAt) {
    return right.usedAt - left.usedAt;
  }
  return (left.commandPrefix ?? left.title).localeCompare(
    right.commandPrefix ?? right.title,
    "zh-CN",
  );
}

function compareRecentMentionEntries(
  left: RecentMentionEntry,
  right: RecentMentionEntry,
): number {
  if (left.usedAt !== right.usedAt) {
    return right.usedAt - left.usedAt;
  }
  return left.title.localeCompare(right.title, "zh-CN");
}

function resolveDisplayTitleFromCommandLike(item: {
  label?: string;
  commandPrefix: string;
}): string {
  const label = item.label?.trim();
  return label && label !== item.commandPrefix ? label : item.commandPrefix;
}

function compareSlashCommandsForEmptyQuery(
  left: SlashCommandDefinition,
  right: SlashCommandDefinition,
): number {
  const emptyQueryOrder: Record<string, number> = {
    new: 10,
    clear: 20,
    compact: 30,
  };

  return (
    (emptyQueryOrder[left.key] ?? 999) - (emptyQueryOrder[right.key] ?? 999)
  );
}

function resolveBuiltinCommandPrefillReplayText(params: {
  command: BuiltinInputCommand;
  replayText?: string;
  slotValues?: ServiceSkillSlotValues;
}): string | undefined {
  return resolveMentionCommandPrefillReplayText({
    commandKey: params.command.key,
    replayText: params.replayText,
    slotValues: params.slotValues,
  });
}

function resolveRecentBuiltinCommandDescription(
  command: BuiltinInputCommand,
  replayText: string | undefined,
  copy: InputCapabilitySectionsCopy,
): string {
  const normalizedReplayText = replayText?.replace(/\s+/g, " ").trim();
  if (normalizedReplayText) {
    const preview =
      normalizedReplayText.length <= RECENT_REPLAY_TEXT_PREVIEW_LIMIT
        ? normalizedReplayText
        : `${normalizedReplayText
            .slice(0, RECENT_REPLAY_TEXT_PREVIEW_LIMIT)
            .trimEnd()}...`;
    return copy.formatRecentInputDescription(preview);
  }

  if (command.description?.trim()) {
    return `${command.label} · ${command.description}`;
  }
  return command.label;
}

function resolveRecentSlashSkillDescription(
  skill: Skill,
  copy: InstalledSkillPresentationCopy,
): string {
  const description = buildInstalledSkillCapabilityDescription(skill, { copy });
  return description ? `${skill.name} · ${description}` : skill.name;
}

function resolveRecentSlashEntryDescription(
  params: {
    replayText?: string;
    fallbackDescription?: string;
    fallbackTitle: string;
  },
  copy: InputCapabilitySectionsCopy,
): string {
  const normalizedReplayText = params.replayText?.replace(/\s+/g, " ").trim();
  if (normalizedReplayText) {
    const preview =
      normalizedReplayText.length <= RECENT_REPLAY_TEXT_PREVIEW_LIMIT
        ? normalizedReplayText
        : `${normalizedReplayText
            .slice(0, RECENT_REPLAY_TEXT_PREVIEW_LIMIT)
            .trimEnd()}...`;
    return copy.formatRecentInputDescription(preview);
  }

  const fallbackDescription = params.fallbackDescription?.trim();
  if (fallbackDescription) {
    return fallbackDescription;
  }

  return params.fallbackTitle;
}

function resolveInputCommandSectionMeta(
  command: Pick<BuiltinInputCommand, "key">,
  copy: InputCapabilitySectionsCopy,
): InputCommandSectionMeta {
  const groupKey = command.key.startsWith("image_model_")
    ? "generate_expression"
    : (INPUT_COMMAND_GROUP_BY_KEY[command.key] ?? "other");
  const meta = INPUT_COMMAND_SECTION_META[groupKey];
  const metaCopy = copy.inputCommandGroups[groupKey];

  return {
    ...meta,
    heading: metaCopy.heading,
    kindLabel: metaCopy.kindLabel,
  };
}

function resolveSlashCommandSectionMeta(
  command: Pick<SlashCommandDefinition, "kind">,
  copy: InputCapabilitySectionsCopy,
): InputCommandSectionMeta {
  const groupKey =
    command.kind === "local_action"
      ? "workspace_action"
      : command.kind === "prompt_action"
        ? "prompt_action"
        : "status_help";
  const metaCopy = copy.slashCommandGroups[groupKey];
  const resolve = (
    meta: InputCommandSectionMetaBase,
  ): InputCommandSectionMeta => ({
    ...meta,
    heading: metaCopy.heading,
    kindLabel: metaCopy.kindLabel,
  });

  switch (command.kind) {
    case "local_action":
      return resolve(SLASH_COMMAND_SECTION_META.workspace_action);
    case "prompt_action":
      return resolve(SLASH_COMMAND_SECTION_META.prompt_action);
    case "info":
    default:
      return resolve(SLASH_COMMAND_SECTION_META.status_help);
  }
}

function groupItemsBySectionMeta<T>(
  items: T[],
  resolveMeta: (item: T) => InputCommandSectionMeta,
): Array<{ meta: InputCommandSectionMeta; items: T[] }> {
  const groups = new Map<
    string,
    { meta: InputCommandSectionMeta; items: T[] }
  >();

  for (const item of items) {
    const meta = resolveMeta(item);
    const current = groups.get(meta.key);
    if (current) {
      current.items.push(item);
      continue;
    }

    groups.set(meta.key, {
      meta,
      items: [item],
    });
  }

  return Array.from(groups.values()).sort((left, right) => {
    if (left.meta.order !== right.meta.order) {
      return left.meta.order - right.meta.order;
    }
    return left.meta.heading.localeCompare(right.meta.heading, "zh-CN");
  });
}

function truncateSectionBannerText(value: string, maxLength = 96): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }

  return `${normalized.slice(0, maxLength).trimEnd()}...`;
}

function buildMentionRegistryBanner(params: {
  hasRecent: boolean;
  hasServiceSkills: boolean;
  hasInstalledSkills: boolean;
  hasAvailableSkills: boolean;
  hasCharacters: boolean;
  copy: InputCapabilitySectionsCopy;
}): InputCapabilitySection["banner"] {
  const footnotes: string[] = [];
  const copy = params.copy.mentionRegistry;

  if (params.hasServiceSkills) {
    footnotes.push(copy.serviceSkillsFootnote);
  }

  if (params.hasInstalledSkills || params.hasAvailableSkills) {
    footnotes.push(copy.installedSkillsFootnote);
  }

  if (params.hasCharacters) {
    footnotes.push(copy.charactersFootnote);
  }

  return {
    badge: copy.badge,
    title: params.hasRecent ? copy.titleWithRecent : copy.titleDefault,
    summary: truncateSectionBannerText(
      params.hasRecent ? copy.summaryWithRecent : copy.summaryDefault,
    ),
    ...(footnotes.length > 0
      ? {
          footnote: truncateSectionBannerText(footnotes.join(" "), 72),
        }
      : {}),
  };
}

function resolveCuratedTaskLaunchContext(params: {
  task: CuratedTaskTemplateItem;
  referenceEntries?: CuratedTaskReferenceEntry[];
}) {
  const launchPrefill = resolveCuratedTaskTemplateLaunchPrefill(params.task);
  const mergedReferenceEntries = mergeCuratedTaskReferenceEntries([
    ...(params.referenceEntries ?? []),
    ...(launchPrefill?.referenceEntries ?? []),
  ]);
  const mergedReferenceMemoryIds =
    normalizeCuratedTaskReferenceMemoryIds([
      ...(params.referenceEntries
        ? (extractCuratedTaskReferenceMemoryIds(params.referenceEntries) ?? [])
        : []),
      ...(launchPrefill?.referenceMemoryIds ?? []),
      ...(extractCuratedTaskReferenceMemoryIds(mergedReferenceEntries) ?? []),
    ]) ?? [];

  return {
    launchPrefill,
    mergedReferenceEntries,
    mergedReferenceMemoryIds,
  };
}

function buildCuratedTaskResultBaselineSummary(params: {
  task: CuratedTaskTemplateItem;
  referenceEntries?: CuratedTaskReferenceEntry[];
  copy: InputCapabilitySectionsCopy;
}): string | null {
  const snapshot = buildCuratedTaskResultBaselineSnapshot({
    referenceEntries: params.referenceEntries,
    taskId: params.task.id,
  });
  if (!snapshot) {
    return null;
  }

  const highlights = [
    snapshot.statusLabel
      ? params.copy.baseline.formatStatusHighlight(snapshot.statusLabel)
      : null,
    snapshot.destinationsLabel
      ? params.copy.baseline.formatDestinationHighlight(
          snapshot.destinationsLabel,
        )
      : snapshot.operatingAction
        ? params.copy.baseline.formatOperatingActionHighlight(
            snapshot.operatingAction,
          )
        : null,
  ].filter((item): item is string => Boolean(item));

  return [
    params.copy.baseline.formatBaseline(snapshot.sourceTitle),
    ...highlights,
  ]
    .filter((item) => item.trim().length > 0)
    .join(" · ");
}

function buildCuratedTaskSlashDescription(params: {
  task: CuratedTaskTemplateItem;
  reasonSummary?: string;
  referenceEntries?: CuratedTaskReferenceEntry[];
  fallbackDescription?: string;
  copy: InputCapabilitySectionsCopy;
}): string {
  const resultBaselineSummary = buildCuratedTaskResultBaselineSummary({
    task: params.task,
    referenceEntries: params.referenceEntries,
    copy: params.copy,
  });

  return [
    resultBaselineSummary,
    params.reasonSummary,
    params.fallbackDescription,
  ]
    .filter((value): value is string =>
      Boolean(value && value.trim().length > 0),
    )
    .join(" · ");
}

const SERVICE_SKILL_GROUP_META: Record<
  string,
  { title: string; sort: number }
> = {
  github: { title: "GitHub", sort: 10 },
  zhihu: { title: "知乎", sort: 20 },
  "linux-do": { title: "Linux.do", sort: 30 },
  bilibili: { title: "Bilibili", sort: 40 },
  "36kr": { title: "36Kr", sort: 50 },
  smzdm: { title: "什么值得买", sort: 60 },
  "yahoo-finance": { title: "Yahoo Finance", sort: 70 },
  general: { title: "通用技能", sort: 90 },
};

function resolveServiceSkillGroupKey(skill: ServiceSkillHomeItem): string {
  const normalized = skill.groupKey?.trim();
  return normalized ? normalized : "general";
}

function resolveServiceSkillGroupTitle(groupKey: string): string {
  return SERVICE_SKILL_GROUP_META[groupKey]?.title ?? groupKey;
}

function resolveServiceSkillGroupSort(groupKey: string): number {
  return SERVICE_SKILL_GROUP_META[groupKey]?.sort ?? 80;
}

function groupMentionServiceSkills(
  skills: ServiceSkillHomeItem[],
  serviceSkillGroups: ServiceSkillGroup[] = [],
): MentionServiceSkillGroup[] {
  const serviceSkillGroupMap = new Map(
    serviceSkillGroups.map((group) => [group.key, group] as const),
  );
  const groups = new Map<string, MentionServiceSkillGroup>();

  for (const skill of skills) {
    const groupKey = resolveServiceSkillGroupKey(skill);
    const groupMeta = serviceSkillGroupMap.get(groupKey);
    const current = groups.get(groupKey);
    if (current) {
      current.skills.push(skill);
      continue;
    }

    groups.set(groupKey, {
      key: groupKey,
      title: groupMeta?.title ?? resolveServiceSkillGroupTitle(groupKey),
      sort: groupMeta?.sort ?? resolveServiceSkillGroupSort(groupKey),
      skills: [skill],
    });
  }

  return Array.from(groups.values()).sort((left, right) => {
    if (left.sort !== right.sort) {
      return left.sort - right.sort;
    }
    return left.title.localeCompare(right.title, "zh-CN");
  });
}

export function buildInputCapabilitySections(
  params: BuildInputCapabilitySectionsParams,
): InputCapabilitySection[] {
  const isEmptyQuery = params.mentionQuery.trim().length === 0;

  if (params.mode === "slash") {
    return buildSlashCapabilitySections(params, isEmptyQuery);
  }

  return buildMentionCapabilitySections(params, isEmptyQuery);
}

function buildMentionCapabilitySections(
  params: BuildInputCapabilitySectionsParams,
  isEmptyQuery: boolean,
): InputCapabilitySection[] {
  const mentionUsageMap = getMentionEntryUsageMap();
  const serviceSkillRecommendationBuckets = isEmptyQuery
    ? buildServiceSkillRecommendationBuckets(params.mentionServiceSkills, {
        featuredLimit: FEATURED_SERVICE_SKILL_LIMIT,
        surface: "mention",
      })
    : {
        recentSkills: [],
        featuredSkills: [],
        remainingSkills: [],
      };
  const visibleRecentServiceSkills =
    serviceSkillRecommendationBuckets.recentSkills;
  const visibleRecentMentionEntries: RecentMentionEntry[] = [];

  if (isEmptyQuery) {
    for (const command of params.builtinCommands) {
      const recentRecord = mentionUsageMap.get(
        getMentionEntryUsageRecordKey("builtin_command", command.key),
      );
      if (!recentRecord) {
        continue;
      }

      const resolvedReplayText = resolveBuiltinCommandPrefillReplayText({
        command,
        replayText: recentRecord.replayText,
        slotValues: recentRecord.slotValues,
      });

      visibleRecentMentionEntries.push({
        key: `builtin-command:${command.key}`,
        kind: "builtin_command",
        title: command.commandPrefix,
        description: resolveRecentBuiltinCommandDescription(
          command,
          resolvedReplayText,
          params.inputCapabilityCopy,
        ),
        usedAt: recentRecord.usedAt,
        replayText: resolvedReplayText,
        commandKey: command.key,
        commandPrefix: command.commandPrefix,
      });
    }

    for (const skill of visibleRecentServiceSkills) {
      if (!skill.recentUsedAt) {
        continue;
      }

      const recentPrefill = resolveServiceSkillLaunchPrefill({
        skill,
      });
      visibleRecentMentionEntries.push({
        key: `service-skill:${skill.id}`,
        kind: "service_skill",
        title: skill.title,
        description: [
          buildServiceSkillLaunchPrefillSummary({
            skill,
            slotValues: recentPrefill?.slotValues,
            launchUserInput: recentPrefill?.launchUserInput,
          }),
          buildServiceSkillCapabilityDescription(skill),
        ]
          .filter((segment) => segment.length > 0)
          .join(" · "),
        usedAt: skill.recentUsedAt,
        skillId: skill.id,
      });
    }
  }

  visibleRecentMentionEntries.sort(compareRecentMentionEntries);

  const visibleBuiltinCommands = params.builtinCommands;
  const visiblePluginItems = buildMentionPluginCapabilityItems({
    plugins: params.pluginSuggestions ?? [],
    query: params.mentionQuery,
  });
  const visibleFeaturedServiceSkills =
    serviceSkillRecommendationBuckets.featuredSkills;
  const visibleServiceSkillGroups = groupMentionServiceSkills(
    isEmptyQuery
      ? serviceSkillRecommendationBuckets.remainingSkills
      : params.mentionServiceSkills,
    params.serviceSkillGroups,
  );
  const mentionRegistryBanner =
    isEmptyQuery &&
    (visibleBuiltinCommands.length > 0 ||
      visibleRecentMentionEntries.length > 0)
      ? buildMentionRegistryBanner({
          hasRecent: visibleRecentMentionEntries.length > 0,
          hasServiceSkills:
            visibleFeaturedServiceSkills.length > 0 ||
            visibleServiceSkillGroups.length > 0,
          hasInstalledSkills: params.installedSkills.length > 0,
          hasAvailableSkills: params.availableSkills.length > 0,
          hasCharacters: params.filteredCharacters.length > 0,
          copy: params.inputCapabilityCopy,
        })
      : undefined;
  let didAttachMentionRegistryBanner = false;
  const attachMentionRegistryBanner = (
    section: InputCapabilitySection,
  ): InputCapabilitySection => {
    if (!mentionRegistryBanner || didAttachMentionRegistryBanner) {
      return section;
    }

    didAttachMentionRegistryBanner = true;
    return {
      ...section,
      banner: mentionRegistryBanner,
    };
  };

  const sections: InputCapabilitySection[] = [];

  for (const group of groupItemsBySectionMeta(
    visibleBuiltinCommands,
    (command) =>
      resolveInputCommandSectionMeta(command, params.inputCapabilityCopy),
  )) {
    sections.push(
      attachMentionRegistryBanner({
        key: `builtin-commands:${group.meta.key}`,
        heading: group.meta.heading,
        items: group.items.map((command) => {
          const recentRecord = isEmptyQuery
            ? undefined
            : mentionUsageMap.get(
                getMentionEntryUsageRecordKey("builtin_command", command.key),
              );
          const resolvedReplayText = resolveBuiltinCommandPrefillReplayText({
            command,
            replayText: recentRecord?.replayText,
            slotValues: recentRecord?.slotValues,
          });

          return {
            key: command.key,
            kind: "builtin_command" as const,
            title: command.commandPrefix,
            description: resolveRecentBuiltinCommandDescription(
              command,
              resolvedReplayText,
              params.inputCapabilityCopy,
            ),
            icon: group.meta.icon,
            iconClassName: group.meta.iconClassName,
            command,
            replayText: resolvedReplayText,
          };
        }),
      }),
    );
  }

  if (visiblePluginItems.length > 0) {
    sections.push({
      key: "plugins",
      heading: params.inputCapabilityCopy.headings.plugins,
      items: visiblePluginItems.map((item) => ({
        key: item.key,
        kind: "plugin" as const,
        title: item.title,
        description: item.description,
        icon: "blocks" as const,
        iconClassName: item.disabled
          ? "mr-2 h-4 w-4 text-slate-400"
          : "mr-2 h-4 w-4 text-emerald-600",
        plugin: item.plugin,
        skill: item.skill,
        disabled: item.disabled,
      })),
    });
  }

  if (visibleRecentMentionEntries.length > 0) {
    sections.push(
      attachMentionRegistryBanner({
        key: "recent-mention",
        heading: params.inputCapabilityCopy.headings.recentMention,
        items: visibleRecentMentionEntries.flatMap<InputCapabilityDescriptor>(
          (entry) => {
            if (entry.kind === "builtin_command") {
              const command = params.builtinCommands.find(
                (item) => item.key === entry.commandKey,
              );
              const meta = command
                ? resolveInputCommandSectionMeta(
                    command,
                    params.inputCapabilityCopy,
                  )
                : null;
              return command
                ? [
                    {
                      key: entry.key,
                      kind: "builtin_command" as const,
                      title: entry.commandPrefix || command.commandPrefix,
                      description: entry.description,
                      icon: meta?.icon ?? "command",
                      iconClassName:
                        meta?.iconClassName ?? "mr-2 h-4 w-4 text-sky-600",
                      kindLabel: entry.kindLabel,
                      command,
                      replayText: entry.replayText,
                    },
                  ]
                : [];
            }

            const skill = params.mentionServiceSkills.find(
              (item) => item.id === entry.skillId,
            );
            return skill
              ? [
                  {
                    key: entry.key,
                    kind: "service_skill" as const,
                    title: entry.title,
                    description: entry.description,
                    icon: "sparkles" as const,
                    iconClassName: "mr-2 h-4 w-4 text-emerald-600",
                    skill,
                  },
                ]
              : [];
          },
        ),
      }),
    );
  }

  if (visibleFeaturedServiceSkills.length > 0) {
    sections.push({
      key: "featured-service-skills",
      heading: params.inputCapabilityCopy.headings.featuredServiceSkills,
      items: visibleFeaturedServiceSkills.map((skill) => ({
        key: `featured-${skill.id}`,
        kind: "service_skill" as const,
        title: skill.title,
        description: buildServiceSkillCapabilityDescription(skill),
        icon: "sparkles" as const,
        iconClassName: "mr-2 h-4 w-4 text-sky-600",
        skill,
      })),
    });
  }

  for (const group of visibleServiceSkillGroups) {
    sections.push({
      key: `service-skill-group:${group.key}`,
      heading: group.title,
      items: group.skills.map((skill) => ({
        key: skill.id,
        kind: "service_skill" as const,
        title: skill.title,
        description: buildServiceSkillCapabilityDescription(skill),
        icon: "sparkles" as const,
        iconClassName: "mr-2 h-4 w-4 text-emerald-600",
        skill,
      })),
    });
  }

  if (params.installedSkills.length > 0) {
    sections.push({
      key: "installed-skills",
      heading: params.inputCapabilityCopy.headings.installedSkills,
      items: params.installedSkills.map((skill) => ({
        key: skill.directory,
        kind: "installed_skill" as const,
        title: skill.name,
        description: buildInstalledSkillCapabilityDescription(skill, {
          copy: params.inputCapabilityCopy.installedSkillPresentation,
        }),
        icon: "zap" as const,
        iconClassName: "mr-2 h-4 w-4 text-primary",
        skill,
      })),
    });
  }

  if (params.availableSkills.length > 0) {
    sections.push({
      key: "available-skills",
      heading: params.inputCapabilityCopy.headings.availableSkills,
      items: params.availableSkills.map((skill) => ({
        key: skill.directory,
        kind: "available_skill" as const,
        title: skill.name,
        description: skill.description?.trim() || skill.name,
        icon: "zap" as const,
        iconClassName: "mr-2 h-4 w-4",
        skill,
      })),
    });
  }

  if (params.filteredCharacters.length > 0) {
    sections.push({
      key: "characters",
      heading: params.inputCapabilityCopy.headings.characters,
      items: params.filteredCharacters.map((character) => ({
        key: character.id,
        kind: "character" as const,
        title: character.name,
        description: character.description?.trim() || character.name,
        icon: "user" as const,
        iconClassName: "mr-2 h-4 w-4",
        character,
      })),
    });
  }

  return sections.filter((section) => section.items.length > 0);
}

function buildSlashCapabilitySections(
  params: BuildInputCapabilitySectionsParams,
  isEmptyQuery: boolean,
): InputCapabilitySection[] {
  const filteredCuratedTaskTemplates = filterCuratedTaskTemplates(
    params.mentionQuery,
    listCuratedTaskTemplates(params.curatedTaskTemplateCopy),
  );
  const featuredCuratedTaskTemplates = listFeaturedHomeCuratedTaskTemplates(
    filteredCuratedTaskTemplates,
    {
      copy: params.curatedTaskTemplateCopy,
      projectId: params.projectId,
      sessionId: params.sessionId,
      referenceEntries: params.referenceEntries,
      limit: filteredCuratedTaskTemplates.length,
    },
  );
  const curatedTaskTemplates = featuredCuratedTaskTemplates.map(
    (item) => item.template,
  );
  const featuredCuratedTaskTemplateMap = new Map(
    featuredCuratedTaskTemplates.map(
      (item) => [item.template.id, item] as const,
    ),
  );
  const latestReviewSignal = listCuratedTaskRecommendationSignals({
    projectId: params.projectId,
    sessionId: params.sessionId,
  })
    .filter((signal) => signal.source === "review_feedback")
    .sort((left, right) => right.createdAt - left.createdAt)[0];
  const allSupportedSlashCommands = params.slashCommands.filter(
    (command) => command.support === "supported",
  );
  const slashUsageMap = getSlashEntryUsageMap();
  const visibleRecentSlashEntries: RecentSlashEntry[] = [];

  if (isEmptyQuery) {
    for (const command of allSupportedSlashCommands) {
      const recentRecord = slashUsageMap.get(
        getSlashEntryUsageRecordKey("command", command.key),
      );
      if (!recentRecord) {
        continue;
      }

      visibleRecentSlashEntries.push({
        key: `command:${command.key}`,
        kind: "command",
        kindLabel: command.commandPrefix,
        commandPrefix: command.commandPrefix,
        title: command.label,
        description: resolveRecentSlashEntryDescription(
          {
            replayText: recentRecord.replayText,
            fallbackDescription: command.description,
            fallbackTitle: command.label,
          },
          params.inputCapabilityCopy,
        ),
        usedAt: recentRecord.usedAt,
        replayText: recentRecord.replayText,
      });
    }

    for (const command of params.sceneCommands) {
      const recentRecord = slashUsageMap.get(
        getSlashEntryUsageRecordKey("scene", command.key),
      );
      if (!recentRecord) {
        continue;
      }

      visibleRecentSlashEntries.push({
        key: `scene:${command.key}`,
        kind: "scene",
        commandPrefix: command.commandPrefix,
        title: command.label,
        description: resolveRecentSlashEntryDescription(
          {
            replayText: recentRecord.replayText,
            fallbackDescription: command.description,
            fallbackTitle: command.label,
          },
          params.inputCapabilityCopy,
        ),
        usedAt: recentRecord.usedAt,
        replayText: recentRecord.replayText,
      });
    }

    for (const skill of params.installedSkills) {
      const recentRecord = slashUsageMap.get(
        getSlashEntryUsageRecordKey("skill", skill.key),
      );
      if (!recentRecord) {
        continue;
      }

      visibleRecentSlashEntries.push({
        key: `skill:${skill.key}`,
        kind: "skill",
        commandPrefix: `/${skill.key}`,
        title: skill.name,
        description: resolveRecentSlashEntryDescription(
          {
            replayText: recentRecord.replayText,
            fallbackDescription: resolveRecentSlashSkillDescription(
              skill,
              params.inputCapabilityCopy.installedSkillPresentation,
            ),
            fallbackTitle: skill.name,
          },
          params.inputCapabilityCopy,
        ),
        usedAt: recentRecord.usedAt,
        replayText: recentRecord.replayText,
      });
    }

    for (const template of curatedTaskTemplates) {
      if (typeof template.recentUsedAt !== "number") {
        continue;
      }

      const launchPrefill = resolveCuratedTaskTemplateLaunchPrefill(template);
      visibleRecentSlashEntries.push({
        key: `curated-task:${template.id}`,
        kind: "curated_task",
        title: template.title,
        description: [
          buildCuratedTaskRecentUsageDescription({
            task: template,
            prefill: launchPrefill,
          }),
          resolveRecentSlashEntryDescription(
            {
              fallbackDescription: buildCuratedTaskCapabilityDescription(
                template,
                {
                  includeSummary: false,
                  includeResultDestination: true,
                  includeFollowUpActions: true,
                  followUpLimit: 1,
                },
              ),
              fallbackTitle: template.title,
            },
            params.inputCapabilityCopy,
          ),
        ]
          .filter((segment) => segment.length > 0)
          .join(" · "),
        usedAt: template.recentUsedAt,
        taskId: template.id,
      });
    }
  }

  visibleRecentSlashEntries.sort(compareRecentSlashEntries);

  const recentSlashCommandKeys = new Set(
    visibleRecentSlashEntries
      .filter((entry) => entry.kind === "command")
      .map((entry) => entry.commandPrefix),
  );
  const recentSlashSceneKeys = new Set(
    visibleRecentSlashEntries
      .filter((entry) => entry.kind === "scene")
      .map((entry) => entry.commandPrefix),
  );
  const recentSlashSkillKeys = new Set(
    visibleRecentSlashEntries
      .filter((entry) => entry.kind === "skill")
      .map((entry) => entry.commandPrefix),
  );
  const recentCuratedTaskIds = new Set(
    visibleRecentSlashEntries
      .filter((entry) => entry.kind === "curated_task")
      .map((entry) => entry.taskId)
      .filter((entry): entry is string => Boolean(entry)),
  );

  const visibleSupportedSlashCommands = (
    isEmptyQuery
      ? allSupportedSlashCommands.filter(
          (command) => command.kind === "local_action",
        )
      : allSupportedSlashCommands
  )
    .filter((command) => !recentSlashCommandKeys.has(command.commandPrefix))
    .sort((left, right) =>
      isEmptyQuery ? compareSlashCommandsForEmptyQuery(left, right) : 0,
    );
  const visibleUnsupportedSlashCommands = !isEmptyQuery
    ? params.slashCommands.filter(
        (command) => command.support === "unsupported",
      )
    : [];
  const visibleSceneCommands = params.sceneCommands.filter(
    (command) => !recentSlashSceneKeys.has(command.commandPrefix),
  );
  const visibleInstalledSkills = isEmptyQuery
    ? params.installedSkills.filter(
        (skill) => !recentSlashSkillKeys.has(`/${skill.key}`),
      )
    : params.installedSkills;
  const visibleCuratedTaskTemplates = isEmptyQuery
    ? curatedTaskTemplates.filter(
        (template) => !recentCuratedTaskIds.has(template.id),
      )
    : curatedTaskTemplates;
  const reviewReasonLabel =
    params.curatedTaskTemplateCopy?.recommendation?.recentReviewReasonLabel ??
    "";
  const highlightedReviewTemplates = visibleCuratedTaskTemplates
    .filter(
      (task) =>
        featuredCuratedTaskTemplateMap.get(task.id)?.reasonLabel ===
        reviewReasonLabel,
    )
    .slice(0, 2);

  const sections: InputCapabilitySection[] = [];

  const buildRecentSlashCapabilityItems = (
    entries: RecentSlashEntry[],
  ): InputCapabilityDescriptor[] =>
    entries.flatMap<InputCapabilityDescriptor>((entry) => {
      if (entry.kind === "command") {
        const command = allSupportedSlashCommands.find(
          (item) => item.commandPrefix === entry.commandPrefix,
        );
        const meta = command
          ? resolveSlashCommandSectionMeta(command, params.inputCapabilityCopy)
          : null;
        return command
          ? [
              {
                key: entry.key,
                kind: "slash_command" as const,
                title: entry.title,
                description: entry.description,
                icon: meta?.icon ?? "command",
                iconClassName:
                  meta?.iconClassName ?? "mr-2 h-4 w-4 text-emerald-600",
                kindLabel:
                  entry.kindLabel ??
                  entry.commandPrefix ??
                  command.commandPrefix,
                command,
                replayText: entry.replayText,
              },
            ]
          : [];
      }

      if (entry.kind === "scene") {
        const command = params.sceneCommands.find(
          (item) => item.commandPrefix === entry.commandPrefix,
        );
        return command
          ? [
              {
                key: entry.key,
                kind: "scene_command" as const,
                title: entry.title,
                description: entry.description,
                icon: "zap" as const,
                iconClassName: "mr-2 h-4 w-4 text-sky-600",
                command,
                replayText: entry.replayText,
              },
            ]
          : [];
      }

      if (entry.kind === "curated_task") {
        const task = curatedTaskTemplates.find(
          (item) => item.id === entry.taskId,
        );
        if (!task) {
          return [];
        }
        const launchContext = resolveCuratedTaskLaunchContext({
          task,
          referenceEntries: params.referenceEntries,
        });
        return [
          {
            key: entry.key,
            kind: "curated_task" as const,
            title: task.title,
            description: buildCuratedTaskSlashDescription({
              task,
              referenceEntries: launchContext.mergedReferenceEntries,
              fallbackDescription: entry.description,
              copy: params.inputCapabilityCopy,
            }),
            icon: "sparkles" as const,
            iconClassName: "mr-2 h-4 w-4 text-amber-600",
            task,
            launchInputValues: launchContext.launchPrefill?.inputValues,
            referenceMemoryIds: launchContext.mergedReferenceMemoryIds,
            referenceEntries: launchContext.mergedReferenceEntries,
            launcherPrefillHint: launchContext.launchPrefill?.hint,
          },
        ];
      }

      const skill = params.installedSkills.find(
        (item) => `/${item.key}` === entry.commandPrefix,
      );
      return skill
        ? [
            {
              key: entry.key,
              kind: "installed_skill" as const,
              title: skill.name,
              description: entry.description,
              icon: "zap" as const,
              iconClassName: "mr-2 h-4 w-4 text-primary",
              skill,
              replayText: entry.replayText,
            },
          ]
        : [];
    });

  const visibleRecentContinuationEntries = isEmptyQuery
    ? visibleRecentSlashEntries.filter((entry) => entry.kind !== "command")
    : visibleRecentSlashEntries;
  const visibleRecentCommandEntries = isEmptyQuery
    ? visibleRecentSlashEntries.filter((entry) => entry.kind === "command")
    : [];

  if (visibleUnsupportedSlashCommands.length > 0) {
    sections.push({
      key: "unsupported-slash-commands",
      heading: params.inputCapabilityCopy.headings.unsupported,
      items: visibleUnsupportedSlashCommands.map((command) => ({
        key: command.key,
        kind: "slash_command" as const,
        title: command.commandPrefix,
        description: command.description,
        icon: "command" as const,
        iconClassName: "mr-2 h-4 w-4 text-muted-foreground",
        command,
      })),
    });
  }

  const visibleResultTemplateItems: InputCapabilityDescriptor[] = [
    ...visibleSceneCommands.map((command) => ({
      key: command.entryId ?? command.key,
      kind: "scene_command" as const,
      title: resolveDisplayTitleFromCommandLike(command),
      description: command.description,
      icon: "zap" as const,
      iconClassName: "mr-2 h-4 w-4 text-sky-600",
      command,
    })),
    ...visibleCuratedTaskTemplates.map((task) => {
      const launchContext = resolveCuratedTaskLaunchContext({
        task,
        referenceEntries: params.referenceEntries,
      });
      return {
        key: task.id,
        kind: "curated_task" as const,
        title: task.title,
        description: buildCuratedTaskSlashDescription({
          task,
          reasonSummary: featuredCuratedTaskTemplateMap.get(task.id)
            ?.reasonSummary,
          referenceEntries: launchContext.mergedReferenceEntries,
          fallbackDescription: buildCuratedTaskCapabilityDescription(task, {
            includeResultDestination: true,
          }),
          copy: params.inputCapabilityCopy,
        }),
        icon: "sparkles" as const,
        iconClassName: "mr-2 h-4 w-4 text-amber-600",
        task,
        launchInputValues: launchContext.launchPrefill?.inputValues,
        referenceMemoryIds: launchContext.mergedReferenceMemoryIds,
        referenceEntries: launchContext.mergedReferenceEntries,
        launcherPrefillHint: launchContext.launchPrefill?.hint,
      };
    }),
  ];

  const resultTemplatesSection: InputCapabilitySection | null =
    visibleResultTemplateItems.length > 0
      ? {
          key: "result-templates",
          heading: isEmptyQuery
            ? params.inputCapabilityCopy.headings.resultTemplatesEmpty
            : params.inputCapabilityCopy.headings.resultTemplates,
          items: visibleResultTemplateItems,
          ...(latestReviewSignal && highlightedReviewTemplates.length > 0
            ? {
                banner: (() => {
                  const projection = buildReviewFeedbackProjection({
                    signal: latestReviewSignal,
                  });
                  const primarySuggestedItem =
                    (projection?.suggestedTasks[0]
                      ? visibleResultTemplateItems.find(
                          (item) =>
                            item.key === projection.suggestedTasks[0]?.taskId,
                        )
                      : null) ??
                    visibleResultTemplateItems.find(
                      (item) => item.key === highlightedReviewTemplates[0]?.id,
                    ) ??
                    null;

                  return {
                    title: params.inputCapabilityCopy.reviewBanner.formatTitle(
                      latestReviewSignal.title,
                    ),
                    summary: truncateSectionBannerText(
                      [
                        latestReviewSignal.summary,
                        projection?.suggestionText ?? "",
                      ]
                        .filter((segment) => segment.trim().length > 0)
                        .join(" "),
                    ),
                    footnote:
                      params.inputCapabilityCopy.reviewBanner.formatFootnote(
                        highlightedReviewTemplates.map((task) => task.title),
                      ),
                    actionLabel: primarySuggestedItem
                      ? params.inputCapabilityCopy.reviewBanner.formatActionLabel(
                          primarySuggestedItem.title,
                        )
                      : undefined,
                    actionItemKey: primarySuggestedItem?.key,
                  };
                })(),
              }
            : {}),
        }
      : null;

  const installedSkillsSection: InputCapabilitySection | null =
    visibleInstalledSkills.length > 0
      ? {
          key: "installed-skills",
          heading: isEmptyQuery
            ? params.inputCapabilityCopy.headings.installedSkillsEmpty
            : params.inputCapabilityCopy.headings.installedSkills,
          items: visibleInstalledSkills.map((skill) => ({
            key: skill.directory,
            kind: "installed_skill" as const,
            title: skill.name,
            description: buildInstalledSkillCapabilityDescription(skill, {
              copy: params.inputCapabilityCopy.installedSkillPresentation,
            }),
            icon: "zap" as const,
            iconClassName: "mr-2 h-4 w-4 text-primary",
            skill,
          })),
        }
      : null;

  if (isEmptyQuery && resultTemplatesSection) {
    sections.push(resultTemplatesSection);
  }

  if (visibleRecentContinuationEntries.length > 0) {
    sections.push({
      key: "recent-slash-continuations",
      heading: isEmptyQuery
        ? params.inputCapabilityCopy.headings.recentContinuationsEmpty
        : params.inputCapabilityCopy.headings.recentContinuations,
      items: buildRecentSlashCapabilityItems(visibleRecentContinuationEntries),
    });
  }

  if (isEmptyQuery && installedSkillsSection) {
    sections.push(installedSkillsSection);
  }

  for (const group of groupItemsBySectionMeta(
    visibleSupportedSlashCommands,
    (command) =>
      resolveSlashCommandSectionMeta(command, params.inputCapabilityCopy),
  )) {
    sections.push({
      key: `supported-slash-commands:${group.meta.key}`,
      heading: group.meta.heading,
      items: group.items.map((command) => {
        const recentRecord = slashUsageMap.get(
          getSlashEntryUsageRecordKey("command", command.key),
        );
        return {
          key: command.key,
          kind: "slash_command" as const,
          title: resolveDisplayTitleFromCommandLike(command),
          description: resolveRecentSlashEntryDescription(
            {
              replayText: recentRecord?.replayText,
              fallbackDescription: command.description,
              fallbackTitle: command.label,
            },
            params.inputCapabilityCopy,
          ),
          icon: group.meta.icon,
          iconClassName: group.meta.iconClassName,
          kindLabel: command.commandPrefix,
          command,
          replayText: recentRecord?.replayText,
        };
      }),
    });
  }

  if (visibleRecentCommandEntries.length > 0) {
    sections.push({
      key: "recent-slash-operations",
      heading: params.inputCapabilityCopy.headings.recentOperations,
      items: buildRecentSlashCapabilityItems(visibleRecentCommandEntries),
    });
  }

  if (!isEmptyQuery && installedSkillsSection) {
    sections.push(installedSkillsSection);
  }

  if (!isEmptyQuery && resultTemplatesSection) {
    sections.push(resultTemplatesSection);
  }

  if (params.availableSkills.length > 0) {
    sections.push({
      key: "available-skills",
      heading: params.inputCapabilityCopy.headings.availableSkills,
      items: params.availableSkills.map((skill) => ({
        key: skill.directory,
        kind: "available_skill" as const,
        title: skill.name,
        description: skill.description?.trim() || skill.name,
        icon: "zap" as const,
        iconClassName: "mr-2 h-4 w-4",
        skill,
      })),
    });
  }

  return sections.filter((section) => section.items.length > 0);
}
