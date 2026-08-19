import { getRuntimeAppVersion } from "@/lib/appVersion";
import { resolveBaseSetupCatalogPayload } from "@/lib/base-setup/bootstrap";
import { compileCommandCatalogProjection } from "@/lib/base-setup/compat/commandCatalogProjection";
import { compileSceneCatalogProjection } from "@/lib/base-setup/compat/sceneCatalogProjection";
import { createSeededCommandCatalogBaseSetupPackage } from "@/lib/base-setup/seededCommandPackage";
import {
  clearStoredBaseSetupPackageSnapshot,
  saveStoredBaseSetupPackageSnapshot,
  type StoredBaseSetupPackageSnapshot,
} from "@/lib/base-setup/storage";
import {
  hasOemCloudSession,
  resolveOemCloudRuntimeContext,
} from "./oemCloudRuntime";
import {
  getSeededServiceSkillCatalog,
  isServiceSkillBundleSummary,
  parseServiceSkillCatalog,
  type ServiceSkillCatalog,
  type ServiceSkillExecutorBinding,
  type ServiceSkillBundleSummary,
  type ServiceSkillItem,
  type ServiceSkillSurfaceScope,
} from "./serviceSkills";

export type SkillCatalogExecutionKind =
  | "native_skill"
  | "agent_turn"
  | "automation_job";

// legacy compat only：若远端或旧包仍返回 cloud_scene，解析层会先正规化为 agent_turn。
export type SkillCatalogCompatExecutionKind = "cloud_scene";
type SkillCatalogAnyExecutionKind =
  | SkillCatalogExecutionKind
  | SkillCatalogCompatExecutionKind;

export interface SkillCatalogExecution {
  kind: SkillCatalogExecutionKind;
}

export type SkillCatalogEntryKind = "skill" | "command" | "scene";

export type SkillCatalogRenderResultKind =
  | "text"
  | "tool_timeline"
  | "image_gallery"
  | "artifact"
  | "form"
  | "table_report";

export type SkillCatalogRenderDetailKind =
  | "json"
  | "task_detail"
  | "artifact_detail"
  | "media_detail"
  | "scene_detail";

export interface SkillCatalogRenderContract {
  resultKind: SkillCatalogRenderResultKind;
  detailKind: SkillCatalogRenderDetailKind;
  supportsStreaming?: boolean;
  supportsTimeline?: boolean;
}

export interface SkillCatalogSceneTemplate {
  id: string;
  title: string;
  description?: string;
  prompt: string;
}

export type SkillCatalogHomePresentationSlot =
  | "input_suggestion"
  | "starter_chip"
  | "guide_card";

export interface SkillCatalogHomePresentation {
  slot: SkillCatalogHomePresentationSlot;
  label?: string;
  title?: string;
  summary?: string;
  iconToken?: string;
  order?: number;
  groupKey?: string;
  prompt?: string;
}

export type SkillCatalogCommandTriggerMode = "mention" | "slash";

export interface SkillCatalogCommandTrigger {
  mode: SkillCatalogCommandTriggerMode;
  prefix: string;
}

export interface SkillCatalogCommandIntentConfirmation {
  id: string;
  ruleKey: string;
  confirmationKey: string;
  systemPromptKey?: string;
}

export interface SkillCatalogSkillLocator {
  source: "project" | "user" | "app" | "other" | "catalog";
  name: string;
  directory?: string;
  skillFilePath?: string;
}

export interface SkillCatalogSkillEntry {
  id: string;
  kind: "skill";
  title: string;
  summary: string;
  skillId: string;
  skillLocator?: SkillCatalogSkillLocator;
  groupKey: string;
  aliases?: string[];
  surfaceScopes?: ServiceSkillSurfaceScope[];
  homePresentation?: SkillCatalogHomePresentation;
  execution: SkillCatalogExecution;
  renderContract?: SkillCatalogRenderContract;
  skillBundle?: ServiceSkillBundleSummary;
}

export interface SkillCatalogCommandEntry {
  id: string;
  kind: "command";
  title: string;
  summary: string;
  commandKey: string;
  aliases?: string[];
  surfaceScopes?: ServiceSkillSurfaceScope[];
  homePresentation?: SkillCatalogHomePresentation;
  triggers: SkillCatalogCommandTrigger[];
  binding?: {
    skillId?: string;
    skillLocator?: SkillCatalogSkillLocator;
    executionKind?:
      | SkillCatalogExecutionKind
      | "task_queue"
      | "server_api"
      | "cli";
    requestDefaults?: Record<string, string>;
    intentConfirmation?: SkillCatalogCommandIntentConfirmation;
  };
  renderContract?: SkillCatalogRenderContract;
}

export interface SkillCatalogSceneEntry {
  id: string;
  kind: "scene";
  title: string;
  summary: string;
  sceneKey: string;
  commandPrefix: string;
  linkedEntryId?: string;
  aliases?: string[];
  surfaceScopes?: ServiceSkillSurfaceScope[];
  homePresentation?: SkillCatalogHomePresentation;
  linkedSkillId?: string;
  skillLocator?: SkillCatalogSkillLocator;
  executionKind?: SkillCatalogExecutionKind | "scene";
  requestDefaults?: Record<string, string>;
  placeholder?: string;
  templates?: SkillCatalogSceneTemplate[];
  renderContract?: SkillCatalogRenderContract;
}

export type SkillCatalogEntry =
  | SkillCatalogSkillEntry
  | SkillCatalogCommandEntry
  | SkillCatalogSceneEntry;

export interface SkillCatalogGroup {
  key: string;
  title: string;
  summary: string;
  entryHint?: string;
  themeTarget?: string;
  sort: number;
  itemCount: number;
}

export interface SkillCatalogItem extends ServiceSkillItem {
  groupKey: string;
  execution: SkillCatalogExecution;
}

export interface SkillCatalog {
  version: string;
  tenantId: string;
  syncedAt: string;
  groups: SkillCatalogGroup[];
  items: SkillCatalogItem[];
  entries: SkillCatalogEntry[];
}

export type SkillCatalogChangeSource =
  | "seeded_fallback"
  | "bootstrap_sync"
  | "manual_override"
  | "cache_clear";

interface SkillCatalogResponseEnvelope {
  code?: number;
  message?: string;
  data?: unknown;
}

interface NormalizedSkillCatalogInput {
  catalog: SkillCatalog;
  baseSetupSnapshot: StoredBaseSetupPackageSnapshot | null;
}

const SKILL_CATALOG_STORAGE_KEY = "lime:skill-catalog:v1";
const LOCAL_MODEL_BOUND_IMAGE_COMMAND_STORAGE_KEY =
  "lime:local-direct-image-commands:v1";
export const SKILL_CATALOG_CHANGED_EVENT = "lime:skill-catalog-changed";
const SEEDED_SKILL_GROUP_PRESETS = [
  {
    key: "github",
    title: "GitHub",
    summary: "围绕仓库与 Issue 的只读研究技能，直接复用真实登录态抓线索。",
    entryHint:
      "先选一个 GitHub 技能，再补关键词或仓库名，结果会直接回流到当前工作区。",
    themeTarget: "general",
    sort: 10,
  },
  {
    key: "zhihu",
    title: "知乎",
    summary: "围绕热榜与内容检索的只读研究技能，适合快速做选题与观点线索扫描。",
    entryHint: "从热榜或关键词入口开始，先抓一轮线索，再回到 Claw 继续整理。",
    themeTarget: "general",
    sort: 20,
  },
  {
    key: "linux-do",
    title: "Linux.do",
    summary: "围绕社区分类与热门讨论的只读研究技能，适合跟踪开发者社区动态。",
    entryHint: "先确定是看分类还是看热门，再直接在真实社区页面采集结果。",
    themeTarget: "general",
    sort: 30,
  },
  {
    key: "bilibili",
    title: "Bilibili",
    summary:
      "围绕视频检索的只读站点技能，适合快速抓视频线索并回流到当前工作区。",
    entryHint: "先给检索词，再直接复用当前浏览器页面做一轮视频线索采集。",
    themeTarget: "general",
    sort: 40,
  },
  {
    key: "36kr",
    title: "36Kr",
    summary: "围绕快讯和资讯流的只读站点技能，适合快速收集行业动态和新闻线索。",
    entryHint: "先确定主题范围，再直接采集快讯结果回到 Claw 继续整理。",
    themeTarget: "general",
    sort: 50,
  },
  {
    key: "smzdm",
    title: "什么值得买",
    summary:
      "围绕消费和商品检索的只读站点技能，适合快速抓价格、优惠与选品线索。",
    entryHint: "输入商品关键词后直接采集结果，再回到 Claw 做整理和对比。",
    themeTarget: "general",
    sort: 60,
  },
  {
    key: "yahoo-finance",
    title: "Yahoo Finance",
    summary: "围绕股票与行情摘要的只读站点技能，适合快速拉一轮金融研究线索。",
    entryHint: "输入股票代码后直接抓取行情摘要，再在工作区继续分析。",
    themeTarget: "general",
    sort: 70,
  },
  {
    key: "general",
    title: "通用技能",
    summary:
      "保留现有写作、调研、趋势选题与增长跟踪能力，作为站点组之外的创作技能入口。",
    entryHint:
      "如果任务不依赖站点登录态，直接从这里选一个创作技能进入工作模式。",
    themeTarget: "general",
    sort: 90,
  },
] as const;
const SEEDED_SKILL_CATALOG_VERSION =
  "client-seed-skill-catalog-2026-04-08-creation-copy";
const SEEDED_COMMAND_ENTRIES = compileCommandCatalogProjection(
  createSeededCommandCatalogBaseSetupPackage(),
  getSeededServiceSkillCatalog().items,
);
const SITE_GROUP_TITLE_OVERRIDES: Record<string, string> = {
  github: "GitHub",
  zhihu: "知乎",
  "linux-do": "Linux.do",
  bilibili: "Bilibili",
  "36kr": "36Kr",
  smzdm: "什么值得买",
  "yahoo-finance": "Yahoo Finance",
  general: "通用技能",
};

function cloneJsonValue<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function cloneSkillCatalog(catalog: SkillCatalog): SkillCatalog {
  return {
    ...catalog,
    groups: catalog.groups.map((group) => ({ ...group })),
    items: catalog.items.map((item) => ({
      ...cloneJsonValue(item),
      execution: { ...item.execution },
    })),
    entries: (catalog.entries || []).map((entry) => cloneJsonValue(entry)),
  };
}

function normalizeCommandTrigger(value: string): string | null {
  const normalized = value.trim();
  if (!normalized) {
    return null;
  }
  return normalized.startsWith("@") ? normalized : `@${normalized}`;
}

function normalizeCommandToken(value: string): string {
  return value.trim().toLowerCase();
}

function hashCommandText(value: string): string {
  let hash = 5381;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 33) ^ value.charCodeAt(index);
  }
  return (hash >>> 0).toString(36);
}

function slugifyCommandSegment(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/^@+/, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function buildModelBoundImageCommandSlug(input: {
  trigger: string;
  providerId: string;
  modelId: string;
}): string {
  const triggerSlug = slugifyCommandSegment(input.trigger);
  if (triggerSlug) {
    return triggerSlug;
  }

  const modelSlug = slugifyCommandSegment(input.modelId);
  if (modelSlug) {
    return modelSlug;
  }

  return `command_${hashCommandText(
    `${input.trigger}:${input.providerId}:${input.modelId}`,
  )}`;
}

function buildModelBoundImageCommandEntrySource(slug: string): string {
  return `at_${slug}_model_command`;
}

function readCommandRequestDefaults(
  entry: SkillCatalogCommandEntry,
): Record<string, string> {
  return entry.binding?.requestDefaults ?? {};
}

export function isModelBoundImageCommandEntry(
  entry: SkillCatalogCommandEntry,
): boolean {
  const defaults = readCommandRequestDefaults(entry);
  return (
    (defaults.imageWorkbench === "true" ||
      defaults.image_workbench === "true") &&
    (defaults.modelBoundImageTask === "true" ||
      defaults.model_bound_image_task === "true" ||
      defaults.directImageTask === "true" ||
      defaults.directTask === "true" ||
      defaults.direct_task === "true" ||
      defaults.direct_image_task === "true") &&
    Boolean((defaults.providerId ?? defaults.provider_id)?.trim()) &&
    Boolean((defaults.model ?? defaults.modelId ?? defaults.model_id)?.trim())
  );
}

export interface ModelBoundImageCommandBindingInput {
  trigger: string;
  providerId: string;
  modelId: string;
  title?: string;
  summary?: string;
  executorMode?: "images_api" | "responses_image_generation";
}

export function buildModelBoundImageCommandEntry(
  input: ModelBoundImageCommandBindingInput,
): SkillCatalogCommandEntry {
  const trigger = normalizeCommandTrigger(input.trigger);
  const providerId = input.providerId.trim();
  const modelId = input.modelId.trim();
  if (!trigger || !providerId || !modelId) {
    throw new Error("invalid image model command binding");
  }

  const slug = buildModelBoundImageCommandSlug({
    trigger,
    providerId,
    modelId,
  });
  const requestDefaults: Record<string, string> = {
    imageWorkbench: "true",
    modelBoundImageTask: "true",
    entrySource: buildModelBoundImageCommandEntrySource(slug),
    providerId,
    model: modelId,
    bindingSource: "local_provider_settings",
    bindingSourceScope: "client_overlay",
  };
  if (input.executorMode) {
    requestDefaults.executorMode = input.executorMode;
  }

  const title = input.title?.trim() || trigger.replace(/^@+/, "");
  return {
    id: `command:image-model:${slug}`,
    kind: "command",
    title,
    summary: input.summary?.trim() || `使用 ${modelId} 创建标准图片任务。`,
    commandKey: `image_model_${slug}`,
    aliases: [title, modelId],
    surfaceScopes: ["mention", "workspace"],
    triggers: [
      {
        mode: "mention",
        prefix: trigger,
      },
    ],
    binding: {
      skillId: "image_generate",
      skillLocator: createCatalogSkillLocator({ id: "image_generate" }),
      executionKind: "task_queue",
      requestDefaults,
    },
    renderContract: {
      resultKind: "image_gallery",
      detailKind: "media_detail",
      supportsStreaming: true,
      supportsTimeline: true,
    },
  };
}

function normalizeText(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim();
  return normalized.length > 0 ? normalized : null;
}

export function createCatalogSkillLocator(input: {
  id?: string | null;
  skillKey?: string | null;
}): SkillCatalogSkillLocator | undefined {
  const name = normalizeText(input.skillKey) ?? normalizeText(input.id);
  if (!name) {
    return undefined;
  }
  return {
    source: "catalog",
    name,
  };
}

function skillLocatorFromSkillId(
  skillId?: string | null,
): SkillCatalogSkillLocator | undefined {
  return createCatalogSkillLocator({ id: skillId });
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function resolveSkillCatalogExecutionKind(
  binding: ServiceSkillExecutorBinding,
): SkillCatalogExecutionKind {
  switch (binding) {
    case "native_skill":
      return "native_skill";
    case "automation_job":
      return "automation_job";
    default:
      return "agent_turn";
  }
}

function normalizeCompatSkillCatalogExecutionKind(
  kind: SkillCatalogAnyExecutionKind,
): SkillCatalogExecutionKind {
  return kind === "cloud_scene" ? "agent_turn" : kind;
}

function isLegacyCompatSceneSourceItem(
  item: Pick<ServiceSkillItem, "sceneBinding">,
): boolean {
  return Boolean(item.sceneBinding);
}

function parseCommandBindingExecutionKind(
  value: unknown,
): SkillCatalogExecutionKind | "task_queue" | "server_api" | "cli" | undefined {
  const normalized = normalizeText(value);
  if (
    normalized === "native_skill" ||
    normalized === "agent_turn" ||
    normalized === "automation_job" ||
    normalized === "cloud_scene" ||
    normalized === "task_queue" ||
    normalized === "server_api" ||
    normalized === "cli"
  ) {
    return normalized === "cloud_scene" ? "agent_turn" : normalized;
  }
  return undefined;
}

function parseCommandIntentConfirmation(
  value: unknown,
): SkillCatalogCommandIntentConfirmation | undefined {
  if (!isPlainRecord(value)) {
    return undefined;
  }

  const id = normalizeText(value.id);
  const ruleKey = normalizeText(value.ruleKey) ?? normalizeText(value.rule_key);
  const confirmationKey =
    normalizeText(value.confirmationKey) ??
    normalizeText(value.confirmation_key);
  if (!id || !ruleKey || !confirmationKey) {
    return undefined;
  }

  const systemPromptKey =
    normalizeText(value.systemPromptKey) ??
    normalizeText(value.system_prompt_key);

  return {
    id,
    ruleKey,
    confirmationKey,
    ...(systemPromptKey ? { systemPromptKey } : {}),
  };
}

function parseSceneExecutionKind(
  value: unknown,
): SkillCatalogSceneEntry["executionKind"] | undefined {
  const normalized = normalizeText(value);
  if (
    normalized === "native_skill" ||
    normalized === "agent_turn" ||
    normalized === "automation_job" ||
    normalized === "cloud_scene" ||
    normalized === "scene"
  ) {
    return normalized === "cloud_scene" ? "agent_turn" : normalized;
  }
  return undefined;
}

function normalizeSearchAliases(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const aliases = value
    .map((item) => normalizeText(item))
    .filter((item): item is string => Boolean(item));

  return aliases.length > 0 ? aliases : undefined;
}

function normalizeStringRecord(
  value: unknown,
): Record<string, string> | undefined {
  if (!isPlainRecord(value)) {
    return undefined;
  }

  const result: Record<string, string> = {};
  for (const [key, item] of Object.entries(value)) {
    const normalized = normalizeText(item);
    if (normalized) {
      result[key] = normalized;
    }
  }

  return Object.keys(result).length > 0 ? result : undefined;
}

function normalizeSurfaceScopes(
  value: unknown,
): ServiceSkillSurfaceScope[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const scopes = value.filter(
    (item): item is ServiceSkillSurfaceScope =>
      item === "home" || item === "mention" || item === "workspace",
  );

  return scopes.length > 0 ? scopes : undefined;
}

function parseRenderContract(
  value: unknown,
): SkillCatalogRenderContract | undefined {
  if (!isPlainRecord(value)) {
    return undefined;
  }

  const resultKind =
    normalizeText(value.resultKind) ?? normalizeText(value.result_kind);
  const detailKind =
    normalizeText(value.detailKind) ?? normalizeText(value.detail_kind);
  if (
    resultKind !== "text" &&
    resultKind !== "tool_timeline" &&
    resultKind !== "image_gallery" &&
    resultKind !== "artifact" &&
    resultKind !== "form" &&
    resultKind !== "table_report"
  ) {
    return undefined;
  }

  if (
    detailKind !== "json" &&
    detailKind !== "task_detail" &&
    detailKind !== "artifact_detail" &&
    detailKind !== "media_detail" &&
    detailKind !== "scene_detail"
  ) {
    return undefined;
  }

  return {
    resultKind,
    detailKind,
    supportsStreaming:
      typeof value.supportsStreaming === "boolean"
        ? value.supportsStreaming
        : typeof value.supports_streaming === "boolean"
          ? value.supports_streaming
          : undefined,
    supportsTimeline:
      typeof value.supportsTimeline === "boolean"
        ? value.supportsTimeline
        : typeof value.supports_timeline === "boolean"
          ? value.supports_timeline
          : undefined,
  };
}

function buildSkillEntryFromCatalogItem(
  item: SkillCatalogItem,
): SkillCatalogSkillEntry {
  return {
    id: `skill:${item.id}`,
    kind: "skill",
    title: item.title,
    summary: item.summary,
    skillId: item.id,
    skillLocator: createCatalogSkillLocator(item),
    groupKey: item.groupKey,
    aliases: item.aliases,
    surfaceScopes: item.surfaceScopes,
    execution: cloneJsonValue(item.execution),
    skillBundle: item.skillBundle
      ? cloneJsonValue(item.skillBundle)
      : undefined,
    renderContract:
      item.defaultArtifactKind === "table_report"
        ? {
            resultKind: "table_report",
            detailKind: "artifact_detail",
            supportsStreaming: true,
          }
        : item.defaultArtifactKind
          ? {
              resultKind: "artifact",
              detailKind: "artifact_detail",
              supportsStreaming: true,
            }
          : {
              resultKind: "text",
              detailKind: "json",
              supportsStreaming: true,
            },
  };
}

function buildSceneEntryFromCatalogItem(
  item: SkillCatalogItem,
): SkillCatalogSceneEntry | null {
  const explicitSceneBinding = item.sceneBinding;
  if (!isLegacyCompatSceneSourceItem(item)) {
    return null;
  }

  const sceneKey =
    normalizeText(explicitSceneBinding?.sceneKey) ??
    normalizeText(item.skillKey) ??
    normalizeText(item.id);
  if (!sceneKey) {
    return null;
  }

  return {
    id: `scene:${sceneKey}`,
    kind: "scene",
    title: normalizeText(explicitSceneBinding?.title) ?? item.title,
    summary: normalizeText(explicitSceneBinding?.summary) ?? item.summary,
    sceneKey,
    commandPrefix:
      normalizeText(explicitSceneBinding?.commandPrefix) ?? `/${sceneKey}`,
    aliases:
      explicitSceneBinding?.aliases && explicitSceneBinding.aliases.length > 0
        ? explicitSceneBinding.aliases
        : item.aliases,
    surfaceScopes: item.surfaceScopes,
    linkedSkillId: item.id,
    skillLocator: createCatalogSkillLocator(item),
    executionKind: normalizeCompatSkillCatalogExecutionKind(
      item.execution.kind,
    ),
    renderContract: {
      resultKind: "tool_timeline",
      detailKind: "scene_detail",
      supportsStreaming: true,
      supportsTimeline: true,
    },
  };
}

function buildSeededCommandEntries(): SkillCatalogCommandEntry[] {
  return SEEDED_COMMAND_ENTRIES.map((entry) => cloneJsonValue(entry));
}

function buildCatalogEntryMergeKeys(entry: SkillCatalogEntry): string[] {
  const keys = [`id:${entry.id}`];

  if (entry.kind === "command") {
    keys.push(`command:${entry.commandKey.trim().toLowerCase()}`);
    entry.triggers.forEach((trigger) => {
      keys.push(`command-trigger:${trigger.prefix.trim().toLowerCase()}`);
    });
  } else if (entry.kind === "scene") {
    keys.push(`scene:${entry.sceneKey.trim().toLowerCase()}`);
  } else if (entry.kind === "skill") {
    keys.push(`skill:${entry.skillId.trim().toLowerCase()}`);
  }

  return keys;
}

function buildSceneEntriesFromCatalogItems(
  items: SkillCatalogItem[],
): SkillCatalogSceneEntry[] {
  return items
    .map((item) => buildSceneEntryFromCatalogItem(item))
    .filter((entry): entry is SkillCatalogSceneEntry => Boolean(entry));
}

function buildSkillEntriesFromCatalogItems(
  items: SkillCatalogItem[],
): SkillCatalogSkillEntry[] {
  return items.map((item) => buildSkillEntryFromCatalogItem(item));
}

function mergeSeededCatalogEntries(
  entries: SkillCatalogEntry[],
  sceneSourceItems: SkillCatalogItem[],
): SkillCatalogEntry[] {
  const merged: SkillCatalogEntry[] = [];
  const seenKeys = new Set<string>();

  [
    ...entries,
    ...buildSeededCommandEntries(),
    ...buildSceneEntriesFromCatalogItems(sceneSourceItems),
  ].forEach((entry) => {
    const mergeKeys = buildCatalogEntryMergeKeys(entry);
    if (mergeKeys.some((key) => seenKeys.has(key))) {
      return;
    }
    mergeKeys.forEach((key) => {
      seenKeys.add(key);
    });
    merged.push(entry);
  });

  return merged;
}

function buildLegacyCatalogEntries(
  items: SkillCatalogItem[],
  sceneSourceItems: SkillCatalogItem[] = items,
): SkillCatalogEntry[] {
  const entries: SkillCatalogEntry[] = [];
  const seenIds = new Set<string>();

  buildSeededCommandEntries().forEach((entry) => {
    if (seenIds.has(entry.id)) {
      return;
    }
    seenIds.add(entry.id);
    entries.push(entry);
  });

  items.forEach((item) => {
    const skillEntry = buildSkillEntryFromCatalogItem(item);
    if (!seenIds.has(skillEntry.id)) {
      seenIds.add(skillEntry.id);
      entries.push(skillEntry);
    }
  });

  buildSceneEntriesFromCatalogItems(sceneSourceItems).forEach((entry) => {
    if (seenIds.has(entry.id)) {
      return;
    }
    seenIds.add(entry.id);
    entries.push(entry);
  });

  return entries;
}

function resolveSeededSkillGroupKey(item: ServiceSkillItem): string {
  void item;
  return "general";
}

function normalizeCategoryGroupKey(value?: string): string | null {
  const normalized = normalizeText(value);
  if (!normalized) {
    return null;
  }

  return normalized.replace(/\s+/g, "-").toLowerCase();
}

function resolveSkillCatalogGroupKey(item: ServiceSkillItem): string {
  return (
    normalizeCategoryGroupKey(item.category) ?? resolveSeededSkillGroupKey(item)
  );
}

function buildSkillCatalogItemFromServiceSkillItem(
  item: ServiceSkillItem,
  groupKeyOverride?: string,
): SkillCatalogItem {
  const clonedItem = cloneJsonValue(item);

  return {
    ...clonedItem,
    groupKey: groupKeyOverride ?? resolveSkillCatalogGroupKey(clonedItem),
    execution: {
      kind: resolveSkillCatalogExecutionKind(clonedItem.defaultExecutorBinding),
    },
  };
}

function buildRawSkillCatalogFromServiceSkillCatalog(
  catalog: ServiceSkillCatalog,
  explicitEntries: SkillCatalogEntry[] = [],
): SkillCatalog {
  const items = catalog.items.map((item) =>
    buildSkillCatalogItemFromServiceSkillItem(item),
  );

  return {
    version: catalog.version,
    tenantId: catalog.tenantId,
    syncedAt: catalog.syncedAt,
    groups: mergeCatalogGroups([], items),
    items,
    entries: [
      ...buildSkillEntriesFromCatalogItems(items),
      ...explicitEntries.map((entry) => cloneJsonValue(entry)),
    ],
  };
}

function buildRawSeededSkillCatalog(): SkillCatalog {
  const seeded = getSeededServiceSkillCatalog();
  const items: SkillCatalogItem[] = seeded.items.map((item) =>
    buildSkillCatalogItemFromServiceSkillItem(
      item,
      resolveSeededSkillGroupKey(item),
    ),
  );

  const itemCountByGroup = items.reduce<Record<string, number>>((acc, item) => {
    acc[item.groupKey] = (acc[item.groupKey] ?? 0) + 1;
    return acc;
  }, {});

  const groups = SEEDED_SKILL_GROUP_PRESETS.filter(
    (preset) => itemCountByGroup[preset.key] > 0,
  ).map((preset) => ({
    ...preset,
    itemCount: itemCountByGroup[preset.key] ?? 0,
  }));

  return {
    version: SEEDED_SKILL_CATALOG_VERSION,
    tenantId: seeded.tenantId,
    syncedAt: seeded.syncedAt,
    groups,
    items,
    entries: buildLegacyCatalogEntries(items),
  };
}

const RAW_SEEDED_SKILL_CATALOG = buildRawSeededSkillCatalog();
const SEEDED_SKILL_CATALOG = normalizeSkillCatalog(RAW_SEEDED_SKILL_CATALOG);

function titleCaseSegment(value: string): string {
  if (!value) {
    return value;
  }
  return value[0]!.toUpperCase() + value.slice(1);
}

function resolveGroupTitle(groupKey: string): string {
  const normalized = groupKey.trim().toLowerCase();
  if (SITE_GROUP_TITLE_OVERRIDES[normalized]) {
    return SITE_GROUP_TITLE_OVERRIDES[normalized]!;
  }
  return normalized.split(/[-_]/).map(titleCaseSegment).join(" ");
}

function resolveKnownGroupPreset(groupKey: string) {
  return SEEDED_SKILL_GROUP_PRESETS.find((preset) => preset.key === groupKey);
}

function buildFallbackGroupPreset(
  groupKey: string,
): Omit<SkillCatalogGroup, "itemCount"> {
  const title = resolveGroupTitle(groupKey);
  return {
    key: groupKey,
    title,
    summary: `围绕 ${title} 的只读站点技能入口，适合直接复用真实页面上下文采集结果。`,
    entryHint: `先进入 ${title} 技能组，再选择具体技能项开始采集。`,
    themeTarget: "general",
    sort: 80,
  };
}

function mergeCatalogGroups(
  currentGroups: SkillCatalogGroup[],
  items: SkillCatalogItem[],
): SkillCatalogGroup[] {
  const itemCountByGroup = items.reduce<Record<string, number>>((acc, item) => {
    acc[item.groupKey] = (acc[item.groupKey] ?? 0) + 1;
    return acc;
  }, {});
  const groupsByKey = new Map<string, SkillCatalogGroup>();

  for (const group of currentGroups) {
    groupsByKey.set(group.key, {
      ...group,
      itemCount: itemCountByGroup[group.key] ?? 0,
    });
  }

  for (const groupKey of Object.keys(itemCountByGroup)) {
    if (groupsByKey.has(groupKey)) {
      continue;
    }

    const preset =
      resolveKnownGroupPreset(groupKey) ?? buildFallbackGroupPreset(groupKey);
    groupsByKey.set(groupKey, {
      ...preset,
      itemCount: itemCountByGroup[groupKey] ?? 0,
    });
  }

  return Array.from(groupsByKey.values())
    .filter((group) => group.itemCount > 0)
    .sort((left, right) => left.sort - right.sort);
}

function shouldExposeSkillCatalogItem(item: SkillCatalogItem): boolean {
  void item;
  return true;
}

function normalizeSkillCatalog(catalog: SkillCatalog): SkillCatalog {
  const filteredItems = catalog.items.filter(shouldExposeSkillCatalogItem);
  const normalizedGroups = mergeCatalogGroups(catalog.groups, filteredItems);
  const normalizedEntries = mergeSeededCatalogEntries(
    catalog.entries.length > 0
      ? catalog.entries.filter((entry) => {
          if (entry.kind === "skill") {
            return filteredItems.some((item) => entry.skillId === item.id);
          }
          if (entry.kind === "scene") {
            const linkedSkillId = entry.linkedSkillId?.trim();
            if (!linkedSkillId) {
              return true;
            }
            if (catalog.items.some((item) => item.id === linkedSkillId)) {
              return true;
            }
            return entry.executionKind === "scene";
          }
          return true;
        })
      : buildLegacyCatalogEntries(filteredItems, catalog.items),
    catalog.items,
  );

  return cloneSkillCatalog({
    ...catalog,
    groups: normalizedGroups,
    items: filteredItems,
    entries: normalizedEntries,
  });
}

function parseSkillCatalogExecution(
  value: unknown,
): SkillCatalogExecution | null {
  if (!isPlainRecord(value)) {
    return null;
  }

  const kind = normalizeText(value.kind);
  if (
    kind !== "native_skill" &&
    kind !== "agent_turn" &&
    kind !== "automation_job" &&
    kind !== "cloud_scene"
  ) {
    return null;
  }

  return {
    kind: normalizeCompatSkillCatalogExecutionKind(kind),
  };
}

function parseSkillCatalogItem(value: unknown): SkillCatalogItem | null {
  if (!isPlainRecord(value)) {
    return null;
  }

  const groupKey = normalizeText(value.groupKey);
  if (!groupKey) {
    return null;
  }
  const execution = parseSkillCatalogExecution(value.execution);
  if (!execution) {
    return null;
  }

  const serviceSkillCandidate = {
    ...value,
  };
  delete (serviceSkillCandidate as Record<string, unknown>).groupKey;
  delete (serviceSkillCandidate as Record<string, unknown>).execution;

  const parsed = parseServiceSkillCatalog({
    version: "__internal__",
    tenantId: "__internal__",
    syncedAt: "__internal__",
    items: [serviceSkillCandidate],
  });
  const item = parsed?.items[0];
  if (!item) {
    return null;
  }

  return {
    ...item,
    groupKey,
    execution,
  };
}

function parseSkillCatalogGroup(value: unknown): SkillCatalogGroup | null {
  if (!isPlainRecord(value)) {
    return null;
  }

  const key = normalizeText(value.key);
  const title = normalizeText(value.title);
  const summary = normalizeText(value.summary);
  if (!key || !title || !summary || typeof value.sort !== "number") {
    return null;
  }

  return {
    key,
    title,
    summary,
    entryHint: normalizeText(value.entryHint) ?? undefined,
    themeTarget: normalizeText(value.themeTarget) ?? undefined,
    sort: value.sort,
    itemCount:
      typeof value.itemCount === "number" && Number.isFinite(value.itemCount)
        ? value.itemCount
        : 0,
  };
}

function parseSkillCatalogCommandTrigger(
  value: unknown,
): SkillCatalogCommandTrigger | null {
  if (!isPlainRecord(value)) {
    return null;
  }

  const mode = normalizeText(value.mode);
  const prefix = normalizeText(value.prefix);
  if (!prefix || (mode !== "mention" && mode !== "slash")) {
    return null;
  }

  return {
    mode,
    prefix,
  };
}

function parseSkillCatalogSceneTemplate(
  value: unknown,
): SkillCatalogSceneTemplate | null {
  if (!isPlainRecord(value)) {
    return null;
  }

  const id = normalizeText(value.id);
  const title = normalizeText(value.title);
  const prompt = normalizeText(value.prompt);
  if (!id || !title || !prompt) {
    return null;
  }

  return {
    id,
    title,
    description: normalizeText(value.description) ?? undefined,
    prompt,
  };
}

function parseSkillCatalogSceneTemplates(
  value: unknown,
): SkillCatalogSceneTemplate[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const templates = value
    .map((item) => parseSkillCatalogSceneTemplate(item))
    .filter((item): item is SkillCatalogSceneTemplate => Boolean(item));
  return templates.length > 0 ? templates : undefined;
}

function parseSkillCatalogHomePresentation(
  value: unknown,
): SkillCatalogHomePresentation | undefined {
  if (!isPlainRecord(value)) {
    return undefined;
  }

  const slot = normalizeText(value.slot);
  if (
    slot !== "input_suggestion" &&
    slot !== "starter_chip" &&
    slot !== "guide_card"
  ) {
    return undefined;
  }

  const order =
    typeof value.order === "number" && Number.isFinite(value.order)
      ? value.order
      : undefined;

  return {
    slot,
    label: normalizeText(value.label) ?? undefined,
    title: normalizeText(value.title) ?? undefined,
    summary: normalizeText(value.summary) ?? undefined,
    iconToken: normalizeText(value.iconToken) ?? undefined,
    order,
    groupKey: normalizeText(value.groupKey) ?? undefined,
    prompt: normalizeText(value.prompt) ?? undefined,
  };
}

function parseSkillCatalogSkillLocator(
  value: unknown,
): SkillCatalogSkillLocator | undefined {
  if (!isPlainRecord(value)) {
    return undefined;
  }

  const source =
    normalizeText(value.source) ?? normalizeText(value.scope) ?? "catalog";
  if (
    source !== "project" &&
    source !== "user" &&
    source !== "app" &&
    source !== "other" &&
    source !== "catalog"
  ) {
    return undefined;
  }

  const name =
    normalizeText(value.name) ??
    normalizeText(value.skillName) ??
    normalizeText(value.skill_name);
  if (!name) {
    return undefined;
  }

  return {
    source,
    name,
    directory:
      normalizeText(value.directory) ??
      normalizeText(value.skillDirectory) ??
      normalizeText(value.skill_directory) ??
      undefined,
    skillFilePath:
      normalizeText(value.skillFilePath) ??
      normalizeText(value.skill_file_path) ??
      undefined,
  };
}

function parseSkillCatalogEntry(value: unknown): SkillCatalogEntry | null {
  if (!isPlainRecord(value)) {
    return null;
  }

  const id = normalizeText(value.id);
  const kind = normalizeText(value.kind);
  const title = normalizeText(value.title);
  const summary = normalizeText(value.summary);
  if (!id || !kind || !title || !summary) {
    return null;
  }
  const homePresentation = parseSkillCatalogHomePresentation(
    value.homePresentation,
  );

  if (kind === "command") {
    const commandKey =
      normalizeText(value.commandKey) ?? normalizeText(value.command_key);
    const triggers = Array.isArray(value.triggers)
      ? value.triggers
          .map((trigger) => parseSkillCatalogCommandTrigger(trigger))
          .filter((trigger): trigger is SkillCatalogCommandTrigger =>
            Boolean(trigger),
          )
      : [];

    if (!commandKey || (triggers.length === 0 && !homePresentation)) {
      return null;
    }

    const binding = isPlainRecord(value.binding)
      ? (() => {
          const skillId =
            normalizeText(value.binding.skillId) ??
            normalizeText(value.binding.skill_id) ??
            undefined;
          const requestDefaults =
            normalizeStringRecord(value.binding.requestDefaults) ??
            normalizeStringRecord(value.binding.request_defaults);
          const intentConfirmation = parseCommandIntentConfirmation(
            value.binding.intentConfirmation ??
              value.binding.intent_confirmation,
          );
          return {
            skillId,
            skillLocator: parseSkillCatalogSkillLocator(
              value.binding.skillLocator ?? value.binding.skill_locator,
            ) ?? skillLocatorFromSkillId(skillId),
            executionKind: parseCommandBindingExecutionKind(
              value.binding.executionKind ?? value.binding.execution_kind,
            ),
            ...(requestDefaults ? { requestDefaults } : {}),
            ...(intentConfirmation ? { intentConfirmation } : {}),
          };
        })()
      : undefined;

    return {
      id,
      kind: "command",
      title,
      summary,
      commandKey,
      aliases: normalizeSearchAliases(value.aliases),
      surfaceScopes: normalizeSurfaceScopes(value.surfaceScopes),
      homePresentation,
      triggers,
      binding,
      renderContract: parseRenderContract(
        value.renderContract ?? value.render_contract,
      ),
    };
  }

  if (kind === "scene") {
    const sceneKey = normalizeText(value.sceneKey);
    const commandPrefix = normalizeText(value.commandPrefix);
    if (!sceneKey || !commandPrefix) {
      return null;
    }
    const requestDefaults =
      normalizeStringRecord(value.requestDefaults) ??
      normalizeStringRecord(value.request_defaults);
    const linkedSkillId = normalizeText(value.linkedSkillId) ?? undefined;

    return {
      id,
      kind: "scene",
      title,
      summary,
      sceneKey,
      commandPrefix,
      linkedEntryId: normalizeText(value.linkedEntryId) ?? undefined,
      aliases: normalizeSearchAliases(value.aliases),
      surfaceScopes: normalizeSurfaceScopes(value.surfaceScopes),
      homePresentation,
      linkedSkillId,
      skillLocator: parseSkillCatalogSkillLocator(
        value.skillLocator ?? value.skill_locator,
      ) ?? skillLocatorFromSkillId(linkedSkillId),
      executionKind: parseSceneExecutionKind(value.executionKind),
      ...(requestDefaults ? { requestDefaults } : {}),
      placeholder: normalizeText(value.placeholder) ?? undefined,
      templates: parseSkillCatalogSceneTemplates(value.templates),
      renderContract: parseRenderContract(value.renderContract),
    };
  }

  if (kind === "skill") {
    const skillId = normalizeText(value.skillId);
    const groupKey = normalizeText(value.groupKey);
    const execution = parseSkillCatalogExecution(value.execution);
    if (!skillId || !groupKey || !execution) {
      return null;
    }

    return {
      id,
      kind: "skill",
      title,
      summary,
      skillId,
      skillLocator:
        parseSkillCatalogSkillLocator(
          value.skillLocator ?? value.skill_locator,
        ) ?? skillLocatorFromSkillId(skillId),
      groupKey,
      aliases: normalizeSearchAliases(value.aliases),
      surfaceScopes: normalizeSurfaceScopes(value.surfaceScopes),
      homePresentation,
      execution,
      skillBundle: isServiceSkillBundleSummary(value.skillBundle)
        ? cloneJsonValue(value.skillBundle)
        : undefined,
      renderContract: parseRenderContract(value.renderContract),
    };
  }

  return null;
}

function readStoredLocalModelBoundImageCommandEntries(): SkillCatalogCommandEntry[] {
  if (typeof window === "undefined") {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(
      LOCAL_MODEL_BOUND_IMAGE_COMMAND_STORAGE_KEY,
    );
    if (!raw) {
      return [];
    }

    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed
      .map((item) => parseSkillCatalogEntry(item))
      .filter(
        (entry): entry is SkillCatalogCommandEntry =>
          entry?.kind === "command" && isModelBoundImageCommandEntry(entry),
      );
  } catch {
    return [];
  }
}

function writeStoredLocalModelBoundImageCommandEntries(
  entries: SkillCatalogCommandEntry[],
): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.setItem(
      LOCAL_MODEL_BOUND_IMAGE_COMMAND_STORAGE_KEY,
      JSON.stringify(entries),
    );
  } catch {
    // ignore local overlay persistence errors
  }
}

function buildCommandEntryTriggerKeys(
  entry: SkillCatalogCommandEntry,
): string[] {
  return entry.triggers.map((trigger) => normalizeCommandToken(trigger.prefix));
}

function shouldReplaceCommandEntry(
  existing: SkillCatalogCommandEntry,
  incoming: SkillCatalogCommandEntry,
): boolean {
  if (existing.id === incoming.id) {
    return true;
  }
  if (
    normalizeCommandToken(existing.commandKey) ===
    normalizeCommandToken(incoming.commandKey)
  ) {
    return true;
  }

  const incomingTriggers = new Set(buildCommandEntryTriggerKeys(incoming));
  if (
    buildCommandEntryTriggerKeys(existing).some((trigger) =>
      incomingTriggers.has(trigger),
    )
  ) {
    return true;
  }

  const existingDefaults = readCommandRequestDefaults(existing);
  const incomingDefaults = readCommandRequestDefaults(incoming);
  return (
    normalizeCommandToken(
      existingDefaults.providerId ?? existingDefaults.provider_id ?? "",
    ) ===
      normalizeCommandToken(
        incomingDefaults.providerId ?? incomingDefaults.provider_id ?? "",
      ) &&
    normalizeCommandToken(
      existingDefaults.model ??
        existingDefaults.modelId ??
        existingDefaults.model_id ??
        "",
    ) ===
      normalizeCommandToken(
        incomingDefaults.model ??
          incomingDefaults.modelId ??
          incomingDefaults.model_id ??
          "",
      )
  );
}

function mergeLocalModelBoundImageCommandEntries(
  catalog: SkillCatalog,
): SkillCatalog {
  const localEntries = readStoredLocalModelBoundImageCommandEntries();
  if (localEntries.length === 0) {
    return catalog;
  }

  return normalizeSkillCatalog({
    ...catalog,
    entries: [
      ...catalog.entries.filter(
        (entry) =>
          entry.kind !== "command" ||
          !localEntries.some((localEntry) =>
            shouldReplaceCommandEntry(entry, localEntry),
          ),
      ),
      ...localEntries,
    ],
  });
}

export function listLocalModelBoundImageCommandEntries(): SkillCatalogCommandEntry[] {
  return readStoredLocalModelBoundImageCommandEntries().map((entry) =>
    cloneJsonValue(entry),
  );
}

export function upsertLocalModelBoundImageCommandBinding(
  input: ModelBoundImageCommandBindingInput,
): SkillCatalogCommandEntry {
  const entry = buildModelBoundImageCommandEntry(input);
  const nextEntries = [
    ...readStoredLocalModelBoundImageCommandEntries().filter(
      (existing) => !shouldReplaceCommandEntry(existing, entry),
    ),
    entry,
  ];
  writeStoredLocalModelBoundImageCommandEntries(nextEntries);
  emitSkillCatalogChanged("manual_override");
  return cloneJsonValue(entry);
}

export function findModelBoundImageCommandEntryForModel(
  catalog: SkillCatalog,
  providerId: string,
  modelId: string,
): SkillCatalogCommandEntry | null {
  const normalizedProviderId = normalizeCommandToken(providerId);
  const normalizedModelId = normalizeCommandToken(modelId);
  if (!normalizedProviderId || !normalizedModelId) {
    return null;
  }

  return (
    listSkillCatalogCommandEntries(catalog).find((entry) => {
      if (!isModelBoundImageCommandEntry(entry)) {
        return false;
      }

      const defaults = readCommandRequestDefaults(entry);
      return (
        normalizeCommandToken(
          defaults.providerId ?? defaults.provider_id ?? "",
        ) === normalizedProviderId &&
        normalizeCommandToken(
          defaults.model ?? defaults.modelId ?? defaults.model_id ?? "",
        ) === normalizedModelId
      );
    }) ?? null
  );
}

function mergeSeededLocalCustomSkillCatalogItems(
  catalog: SkillCatalog,
): SkillCatalog {
  const localCustomItems = RAW_SEEDED_SKILL_CATALOG.items.filter(
    (item) => item.source === "local_custom",
  );
  if (localCustomItems.length === 0) {
    return catalog;
  }

  const localCustomIds = new Set(localCustomItems.map((item) => item.id));
  return {
    ...catalog,
    items: [
      ...catalog.items.filter((item) => !localCustomIds.has(item.id)),
      ...localCustomItems.map((item) => cloneJsonValue(item)),
    ],
  };
}

function parseRawSkillCatalog(value: unknown): SkillCatalog | null {
  if (!isPlainRecord(value)) {
    return null;
  }

  const version = normalizeText(value.version);
  const tenantId = normalizeText(value.tenantId);
  const syncedAt = normalizeText(value.syncedAt);
  if (!version || !tenantId || !syncedAt) {
    return null;
  }
  if (!Array.isArray(value.groups) || !Array.isArray(value.items)) {
    return null;
  }

  const groups: SkillCatalogGroup[] = [];
  for (const item of value.groups) {
    const parsed = parseSkillCatalogGroup(item);
    if (!parsed) {
      return null;
    }
    groups.push(parsed);
  }

  const items: SkillCatalogItem[] = [];
  for (const item of value.items) {
    const parsed = parseSkillCatalogItem(item);
    if (!parsed) {
      return null;
    }
    items.push(parsed);
  }

  const entriesSource = Array.isArray(value.entries) ? value.entries : [];
  const entries: SkillCatalogEntry[] = [];
  for (const entry of entriesSource) {
    const parsed = parseSkillCatalogEntry(entry);
    if (!parsed) {
      return null;
    }
    entries.push(parsed);
  }

  return normalizeSkillCatalog(
    mergeSeededLocalCustomSkillCatalogItems({
      version,
      tenantId,
      syncedAt,
      groups,
      items,
      entries,
    }),
  );
}

function normalizeSkillCatalogInput(
  value: unknown,
): NormalizedSkillCatalogInput | null {
  const directCatalog = parseRawSkillCatalog(value);
  if (directCatalog) {
    return {
      catalog: directCatalog,
      baseSetupSnapshot: null,
    };
  }

  const runtime = resolveOemCloudRuntimeContext();
  const compiledFromBaseSetup = resolveBaseSetupCatalogPayload(value, {
    appVersion: getRuntimeAppVersion(),
    tenantId: runtime?.tenantId ?? "base-setup",
    seededFallbackAvailable: true,
  });
  if (!compiledFromBaseSetup) {
    return null;
  }

  const explicitSceneEntries = compileSceneCatalogProjection(
    compiledFromBaseSetup.package,
    compiledFromBaseSetup.catalog.items,
  );
  const explicitCommandEntries = compileCommandCatalogProjection(
    compiledFromBaseSetup.package,
    compiledFromBaseSetup.catalog.items,
  );

  return {
    catalog: normalizeSkillCatalog(
      mergeSeededLocalCustomSkillCatalogItems(
        buildRawSkillCatalogFromServiceSkillCatalog(
          compiledFromBaseSetup.catalog,
          [...explicitCommandEntries, ...explicitSceneEntries],
        ),
      ),
    ),
    baseSetupSnapshot: compiledFromBaseSetup.snapshot,
  };
}

function isNormalizedSkillCatalogInput(
  value: unknown,
): value is NormalizedSkillCatalogInput {
  if (!value || typeof value !== "object") {
    return false;
  }

  const record = value as Record<string, unknown>;
  if (!("catalog" in record) || !parseRawSkillCatalog(record.catalog)) {
    return false;
  }

  return record.baseSetupSnapshot === null || "baseSetupSnapshot" in record;
}

export function parseSkillCatalog(value: unknown): SkillCatalog | null {
  return normalizeSkillCatalogInput(value)?.catalog ?? null;
}

function persistSkillCatalog(catalog: SkillCatalog): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.setItem(
      SKILL_CATALOG_STORAGE_KEY,
      JSON.stringify(catalog),
    );
  } catch {
    // ignore local cache errors
  }
}

function readCachedSkillCatalog(): SkillCatalog | null {
  if (typeof window === "undefined") {
    return null;
  }

  try {
    const raw = window.localStorage.getItem(SKILL_CATALOG_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    return parseSkillCatalog(JSON.parse(raw));
  } catch {
    return null;
  }
}

function isSameSkillCatalog(left: SkillCatalog, right: SkillCatalog): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function parseCatalogSyncedAt(value: string): number {
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function compareCatalogVersion(left: string, right: string): number {
  return left.localeCompare(right, undefined, {
    numeric: true,
    sensitivity: "base",
  });
}

function isSeededCatalogCompatibleWithActiveTenant(
  catalog: SkillCatalog,
): boolean {
  if (isSeededSkillCatalog(catalog)) {
    return true;
  }

  const runtime = resolveOemCloudRuntimeContext();
  if (!runtime) {
    return true;
  }

  return catalog.tenantId === runtime.tenantId;
}

function shouldRefreshSeededSkillCatalog(
  cached: SkillCatalog,
  seeded: SkillCatalog,
): boolean {
  if (cached.tenantId !== seeded.tenantId) {
    return false;
  }

  for (const seededGroup of seeded.groups) {
    const cachedGroup = cached.groups.find(
      (group) => group.key === seededGroup.key,
    );
    if (
      !cachedGroup ||
      JSON.stringify(cachedGroup) !== JSON.stringify(seededGroup)
    ) {
      return true;
    }
  }

  for (const seededItem of seeded.items) {
    const cachedItem = cached.items.find((item) => item.id === seededItem.id);
    if (
      !cachedItem ||
      JSON.stringify(cachedItem) !== JSON.stringify(seededItem)
    ) {
      return true;
    }
  }

  for (const seededEntry of seeded.entries) {
    const cachedEntry = cached.entries.find(
      (entry) => entry.id === seededEntry.id,
    );
    if (
      !cachedEntry ||
      JSON.stringify(cachedEntry) !== JSON.stringify(seededEntry)
    ) {
      return true;
    }
  }

  return false;
}

function shouldIgnoreServerSyncedCatalog(
  current: SkillCatalog | null,
  incoming: SkillCatalog,
): boolean {
  const runtime = resolveOemCloudRuntimeContext();
  if (runtime && incoming.tenantId !== runtime.tenantId) {
    return true;
  }

  if (!current || current.tenantId !== incoming.tenantId) {
    return false;
  }

  const currentSyncedAt = parseCatalogSyncedAt(current.syncedAt);
  const incomingSyncedAt = parseCatalogSyncedAt(incoming.syncedAt);

  if (currentSyncedAt > 0 && incomingSyncedAt > 0) {
    if (incomingSyncedAt < currentSyncedAt) {
      return true;
    }
    if (incomingSyncedAt > currentSyncedAt) {
      return false;
    }
  }

  return compareCatalogVersion(incoming.version, current.version) < 0;
}

function emitSkillCatalogChanged(source: SkillCatalogChangeSource): void {
  if (typeof window === "undefined") {
    return;
  }

  window.dispatchEvent(
    new CustomEvent<{ source: SkillCatalogChangeSource; timestamp: number }>(
      SKILL_CATALOG_CHANGED_EVENT,
      {
        detail: {
          source,
          timestamp: Date.now(),
        },
      },
    ),
  );
}

export function getSeededSkillCatalog(): SkillCatalog {
  return cloneSkillCatalog(SEEDED_SKILL_CATALOG);
}

export function getCurrentSkillCatalogSnapshot(): SkillCatalog {
  const seeded = getSeededSkillCatalog();
  const cached = readCachedSkillCatalog();

  if (!cached) {
    return mergeLocalModelBoundImageCommandEntries(seeded);
  }

  if (!isSeededCatalogCompatibleWithActiveTenant(cached)) {
    return mergeLocalModelBoundImageCommandEntries(seeded);
  }

  if (shouldRefreshSeededSkillCatalog(cached, seeded)) {
    return mergeLocalModelBoundImageCommandEntries(seeded);
  }

  return mergeLocalModelBoundImageCommandEntries(cached);
}

export function listSkillCatalogEntries(
  catalog: SkillCatalog,
): SkillCatalogEntry[] {
  return (catalog.entries || []).map((entry) => cloneJsonValue(entry));
}

export function listSkillCatalogCommandEntries(
  catalog: SkillCatalog,
): SkillCatalogCommandEntry[] {
  return listSkillCatalogEntries(catalog).filter(
    (entry): entry is SkillCatalogCommandEntry => entry.kind === "command",
  );
}

export function listSkillCatalogSceneEntries(
  catalog: SkillCatalog,
): SkillCatalogSceneEntry[] {
  return listSkillCatalogEntries(catalog).filter(
    (entry): entry is SkillCatalogSceneEntry => entry.kind === "scene",
  );
}

export function listSkillCatalogSkillEntries(
  catalog: SkillCatalog,
): SkillCatalogSkillEntry[] {
  return listSkillCatalogEntries(catalog).filter(
    (entry): entry is SkillCatalogSkillEntry => entry.kind === "skill",
  );
}

export function isSeededSkillCatalog(catalog: SkillCatalog): boolean {
  return (
    catalog.tenantId === SEEDED_SKILL_CATALOG.tenantId &&
    catalog.version === SEEDED_SKILL_CATALOG.version
  );
}

export function saveSkillCatalog(
  catalog: unknown,
  source: Exclude<
    SkillCatalogChangeSource,
    "seeded_fallback" | "cache_clear"
  > = "manual_override",
): SkillCatalog {
  const normalizedInput = normalizeSkillCatalogInput(catalog);
  if (!normalizedInput) {
    throw new Error("invalid skill catalog");
  }

  const current = readCachedSkillCatalog();
  if (current && isSameSkillCatalog(current, normalizedInput.catalog)) {
    persistSkillCatalog(normalizedInput.catalog);
    if (normalizedInput.baseSetupSnapshot) {
      saveStoredBaseSetupPackageSnapshot(normalizedInput.baseSetupSnapshot);
    } else {
      clearStoredBaseSetupPackageSnapshot();
    }
    return mergeLocalModelBoundImageCommandEntries(normalizedInput.catalog);
  }

  persistSkillCatalog(normalizedInput.catalog);
  if (normalizedInput.baseSetupSnapshot) {
    saveStoredBaseSetupPackageSnapshot(normalizedInput.baseSetupSnapshot);
  } else {
    clearStoredBaseSetupPackageSnapshot();
  }
  emitSkillCatalogChanged(source);
  return mergeLocalModelBoundImageCommandEntries(normalizedInput.catalog);
}

export function applyServerSyncedSkillCatalog(
  catalog: unknown,
  source: "bootstrap_sync",
): SkillCatalog {
  const normalizedInput = isNormalizedSkillCatalogInput(catalog)
    ? catalog
    : normalizeSkillCatalogInput(catalog);
  if (!normalizedInput) {
    return mergeLocalModelBoundImageCommandEntries(getSeededSkillCatalog());
  }

  const current = readCachedSkillCatalog();
  if (shouldIgnoreServerSyncedCatalog(current, normalizedInput.catalog)) {
    return current && isSeededCatalogCompatibleWithActiveTenant(current)
      ? mergeLocalModelBoundImageCommandEntries(current)
      : mergeLocalModelBoundImageCommandEntries(getSeededSkillCatalog());
  }

  if (current && isSameSkillCatalog(current, normalizedInput.catalog)) {
    persistSkillCatalog(normalizedInput.catalog);
    if (normalizedInput.baseSetupSnapshot) {
      saveStoredBaseSetupPackageSnapshot(normalizedInput.baseSetupSnapshot);
    } else {
      clearStoredBaseSetupPackageSnapshot();
    }
    return mergeLocalModelBoundImageCommandEntries(normalizedInput.catalog);
  }

  persistSkillCatalog(normalizedInput.catalog);
  if (normalizedInput.baseSetupSnapshot) {
    saveStoredBaseSetupPackageSnapshot(normalizedInput.baseSetupSnapshot);
  } else {
    clearStoredBaseSetupPackageSnapshot();
  }
  emitSkillCatalogChanged(source);
  return mergeLocalModelBoundImageCommandEntries(normalizedInput.catalog);
}

export function clearSkillCatalogCache(): void {
  if (typeof window !== "undefined") {
    try {
      window.localStorage.removeItem(SKILL_CATALOG_STORAGE_KEY);
    } catch {
      // ignore local cache errors
    }
  }

  clearStoredBaseSetupPackageSnapshot();
  emitSkillCatalogChanged("cache_clear");
}

export function subscribeSkillCatalogChanged(
  callback: (source: SkillCatalogChangeSource) => void,
): () => void {
  if (typeof window === "undefined") {
    return () => {};
  }

  const customEventHandler = (event: Event) => {
    const customEvent = event as CustomEvent<{
      source?: SkillCatalogChangeSource;
    }>;
    const source = customEvent.detail?.source;
    if (source) {
      callback(source);
    }
  };

  const storageHandler = (event: StorageEvent) => {
    if (
      event.key !== SKILL_CATALOG_STORAGE_KEY &&
      event.key !== LOCAL_MODEL_BOUND_IMAGE_COMMAND_STORAGE_KEY
    ) {
      return;
    }
    callback(event.newValue ? "manual_override" : "cache_clear");
  };

  window.addEventListener(SKILL_CATALOG_CHANGED_EVENT, customEventHandler);
  window.addEventListener("storage", storageHandler);

  return () => {
    window.removeEventListener(SKILL_CATALOG_CHANGED_EVENT, customEventHandler);
    window.removeEventListener("storage", storageHandler);
  };
}

async function requestRemoteSkillCatalog(): Promise<NormalizedSkillCatalogInput> {
  const runtime = resolveOemCloudRuntimeContext();
  if (!runtime) {
    throw new Error("缺少品牌云端配置，请先注入 base_url 与 tenant_id。");
  }
  if (!hasOemCloudSession(runtime)) {
    throw new Error("缺少品牌云端 Session Token，请先完成登录或注入会话。");
  }

  const response = await fetch(
    `${runtime.controlPlaneBaseUrl}/v1/public/tenants/${encodeURIComponent(runtime.tenantId)}/client/skills`,
    {
      method: "GET",
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${runtime.sessionToken}`,
      },
    },
  );

  let payload: SkillCatalogResponseEnvelope | null = null;
  try {
    payload = (await response.json()) as SkillCatalogResponseEnvelope;
  } catch {
    payload = null;
  }

  if (!response.ok) {
    throw new Error(
      payload?.message?.trim() || `请求失败 (${response.status})`,
    );
  }

  const normalizedInput = normalizeSkillCatalogInput(payload?.data);
  if (!normalizedInput) {
    throw new Error(payload?.message?.trim() || "服务端返回的技能目录格式非法");
  }

  return normalizedInput;
}

export async function getSkillCatalog(): Promise<SkillCatalog> {
  const seeded = getSeededSkillCatalog();
  const cached = readCachedSkillCatalog();
  if (cached) {
    if (!isSeededCatalogCompatibleWithActiveTenant(cached)) {
      return mergeLocalModelBoundImageCommandEntries(seeded);
    }

    if (shouldRefreshSeededSkillCatalog(cached, seeded)) {
      persistSkillCatalog(seeded);
      clearStoredBaseSetupPackageSnapshot();
      return mergeLocalModelBoundImageCommandEntries(seeded);
    }

    return mergeLocalModelBoundImageCommandEntries(cached);
  }

  persistSkillCatalog(seeded);
  clearStoredBaseSetupPackageSnapshot();
  return mergeLocalModelBoundImageCommandEntries(seeded);
}

export async function refreshSkillCatalogFromRemote(): Promise<SkillCatalog | null> {
  const runtime = resolveOemCloudRuntimeContext();
  if (!runtime || !hasOemCloudSession(runtime)) {
    return null;
  }

  const catalog = await requestRemoteSkillCatalog();
  return applyServerSyncedSkillCatalog(catalog, "bootstrap_sync");
}
