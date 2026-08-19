import {
  detectBrowserTaskRequirement,
  type BrowserTaskRequirementMatch,
} from "../utils/browserTaskRequirement";
import type {
  GeneralWorkbenchRunState,
  GeneralWorkbenchRunTerminalItem,
  GeneralWorkbenchRunTodoItem,
} from "@/lib/api/executionRun";
import {
  buildHarnessRequestMetadata,
  extractExistingHarnessMetadata,
  type HarnessOemRoutingRequestMetadata,
  type HarnessTenantFeatureFlagsRequestMetadata,
} from "../utils/harnessRequestMetadata";
import type { WorkspaceSkillRuntimeEnableInput } from "../utils/workspaceSkillBindingsMetadata";
import {
  hasGenerationBriefMetadata,
  mergeGenerationBriefIntoArtifactMetadata,
  mergeRequestMetadataWithArtifact,
  mergeSoulArtifactVoiceDiagnostics,
} from "../utils/artifactGenerationBriefMetadata";
import type { AgentRuntimeWorkspaceSkillBinding } from "@/lib/api/agentRuntime/toolInventoryTypes";
import type { HandleSendOptions } from "../hooks/handleSendTypes";
import type { InputCapabilitySendRoute } from "../skill-selection/inputCapabilitySelection";
import type { ChatToolPreferences } from "../utils/chatToolPreferences";
import type { BrowserTaskRequirement, Message, MessageImage } from "../types";
import type { Character } from "@/lib/api/projectMemory";
import type { ThemeType } from "@/lib/workspace/workbenchContract";
import { resolveOemCloudRuntimeContext } from "@/lib/api/oemCloudRuntime";
import type {
  OemCloudBootstrapResponse,
  OemCloudFeatureFlags,
} from "@/lib/api/oemCloudControlPlane";
import { getOemCloudBootstrapSnapshot } from "@/lib/oemCloudSession";

const OEM_CLOUD_FEATURE_FLAG_KEYS = [
  "oauthLoginEnabled",
  "emailCodeLoginEnabled",
  "passwordLoginEnabled",
  "profileEditable",
  "hubTokensEnabled",
  "billingEnabled",
  "referralEnabled",
  "gatewayEnabled",
] as const satisfies readonly (keyof OemCloudFeatureFlags)[];

export interface InitialDispatchPreviewSnapshot {
  key: string;
  prompt?: string;
  images: MessageImage[];
}

export function buildInitialDispatchKey(
  prompt?: string,
  images?: MessageImage[],
): string | null {
  const normalizedPrompt = (prompt || "").trim();
  const normalizedImages = images || [];

  if (!normalizedPrompt && normalizedImages.length === 0) {
    return null;
  }

  const imageSignature = normalizedImages
    .map(
      (image, index) =>
        `${index}:${image.mediaType}:${image.data.length}:${image.data.slice(0, 16)}`,
    )
    .join("|");

  return `${normalizedPrompt}::${imageSignature}`;
}

export interface SubmissionPreviewSnapshot {
  key: string;
  prompt: string;
  displayContent?: string;
  inputCapabilityRoute?: InputCapabilitySendRoute;
  images: MessageImage[];
  createdAt: number;
}

export interface ContextWorkspaceSummary {
  enabled: boolean;
  activeContextPrompt: string;
  prepareActiveContextPrompt: () => Promise<string>;
}

export interface GeneralWorkbenchEntryPromptState {
  kind: "initial_prompt" | "resume";
  signature: string;
  title: string;
  description: string;
  actionLabel: string;
  prompt: string;
}

interface BuildGeneralWorkbenchSendBoundaryStateOptions {
  isThemeWorkbench: boolean;
  contentId?: string;
  initialDispatchKey: string | null;
  consumedInitialPromptKey: string | null;
  initialUserImages?: MessageImage[];
  mappedTheme: string;
  sourceText: string;
  sendOptions?: HandleSendOptions;
}

export interface GeneralWorkbenchSendBoundaryState {
  sourceText: string;
  browserRequirementMatch: BrowserTaskRequirementMatch | null;
  shouldConsumePendingGeneralWorkbenchInitialPrompt: boolean;
  shouldDismissGeneralWorkbenchEntryPrompt: boolean;
}

type PreparedActiveContextPromptResult =
  | {
      ok: true;
      value: string;
    }
  | {
      ok: false;
      error: unknown;
    };

function resolveGeneralWorkbenchGateLabel(
  gateKey?: GeneralWorkbenchRunTodoItem["gate_key"],
): string | null {
  switch (gateKey) {
    case "topic_select":
      return "选题确认";
    case "write_mode":
      return "写作推进";
    case "publish_confirm":
      return "发布确认";
    case null:
    case undefined:
    default:
      return null;
  }
}

function resolveGeneralWorkbenchPendingRunCandidate(
  state: GeneralWorkbenchRunState | null,
): GeneralWorkbenchRunTodoItem | GeneralWorkbenchRunTerminalItem | null {
  if (!state) {
    return null;
  }

  const activeQueueItem = (state.queue_items || []).find((item) =>
    ["queued", "running", "error", "timeout"].includes(item.status),
  );
  if (activeQueueItem) {
    return activeQueueItem;
  }

  if (
    state.latest_terminal &&
    ["queued", "running", "error", "timeout"].includes(
      state.latest_terminal.status,
    )
  ) {
    return state.latest_terminal;
  }

  return null;
}

export function buildGeneralWorkbenchResumePromptFromRunState(
  state: GeneralWorkbenchRunState | null,
): GeneralWorkbenchEntryPromptState | null {
  const pendingRun = resolveGeneralWorkbenchPendingRunCandidate(state);
  if (!pendingRun) {
    return null;
  }

  const runTitle = pendingRun.title?.trim() || "最近一次创作任务";
  const gateLabel = resolveGeneralWorkbenchGateLabel(pendingRun.gate_key);
  const stageSuffix = gateLabel ? `，当前停留在“${gateLabel}”附近` : "";

  return {
    kind: "resume",
    signature: `run:${pendingRun.run_id}:${pendingRun.status}:${pendingRun.started_at}:${"finished_at" in pendingRun ? pendingRun.finished_at || "" : ""}`,
    title: "发现上次未完成任务",
    description: `最近一次任务“${runTitle}”尚未完成${stageSuffix}。`,
    actionLabel: "继续上次生成",
    prompt: `请基于当前文稿与最近一次未完成的运行继续推进。任务标题：${runTitle}。${gateLabel ? `优先衔接“${gateLabel}”阶段。` : ""}不要从头开始，先概括已有进度，再继续执行。`,
  };
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }

  return value as Record<string, unknown>;
}

function readMetadataText(
  metadata: Record<string, unknown> | undefined,
  key: string,
): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : undefined;
}

function readExistingBrowserRequirement(
  metadata: Record<string, unknown> | undefined,
): BrowserTaskRequirement | undefined {
  const requirement = readMetadataText(metadata, "browser_requirement");
  if (
    requirement === "optional" ||
    requirement === "required" ||
    requirement === "required_with_user_step"
  ) {
    return requirement;
  }
  return undefined;
}

function omitLegacyAccessModeFromHarnessMetadata(
  metadata: Record<string, unknown> | undefined,
): Record<string, unknown> | undefined {
  if (!metadata) {
    return metadata;
  }

  const nextMetadata = { ...metadata };
  delete nextMetadata.access_mode;
  delete nextMetadata.accessMode;
  return nextMetadata;
}

function readExistingOemRouting(
  metadata: Record<string, unknown> | undefined,
): HarnessOemRoutingRequestMetadata | undefined {
  const rawRouting =
    asRecord(metadata?.oem_routing) ?? asRecord(metadata?.oemRouting);
  const tenantId =
    readMetadataText(rawRouting, "tenant_id") ||
    readMetadataText(rawRouting, "tenantId");
  if (!tenantId) {
    return undefined;
  }

  const fallbackToLocalAllowed =
    typeof rawRouting?.fallback_to_local_allowed === "boolean"
      ? rawRouting.fallback_to_local_allowed
      : typeof rawRouting?.fallbackToLocalAllowed === "boolean"
        ? rawRouting.fallbackToLocalAllowed
        : undefined;
  const canInvoke =
    typeof rawRouting?.can_invoke === "boolean"
      ? rawRouting.can_invoke
      : typeof rawRouting?.canInvoke === "boolean"
        ? rawRouting.canInvoke
        : undefined;

  return {
    tenant_id: tenantId,
    provider_source:
      readMetadataText(rawRouting, "provider_source") ||
      readMetadataText(rawRouting, "providerSource"),
    provider_key:
      readMetadataText(rawRouting, "provider_key") ||
      readMetadataText(rawRouting, "providerKey"),
    default_model:
      readMetadataText(rawRouting, "default_model") ||
      readMetadataText(rawRouting, "defaultModel"),
    config_mode:
      readMetadataText(rawRouting, "config_mode") ||
      readMetadataText(rawRouting, "configMode"),
    offer_state:
      readMetadataText(rawRouting, "offer_state") ||
      readMetadataText(rawRouting, "offerState"),
    quota_status:
      readMetadataText(rawRouting, "quota_status") ||
      readMetadataText(rawRouting, "quotaStatus"),
    fallback_to_local_allowed: fallbackToLocalAllowed,
    can_invoke: canInvoke,
  };
}

function readBooleanFlagMap(value: unknown): Record<string, boolean> {
  const source = asRecord(value);
  if (!source) {
    return {};
  }

  return Object.entries(source).reduce<Record<string, boolean>>(
    (flags, [key, flagValue]) => {
      if (typeof flagValue === "boolean") {
        flags[key] = flagValue;
      }
      return flags;
    },
    {},
  );
}

function readExistingTenantFeatureFlags(
  metadata: Record<string, unknown> | undefined,
): HarnessTenantFeatureFlagsRequestMetadata | undefined {
  const rawFlags =
    asRecord(metadata?.tenant_feature_flags) ??
    asRecord(metadata?.tenantFeatureFlags);
  const tenantId =
    readMetadataText(rawFlags, "tenant_id") ||
    readMetadataText(rawFlags, "tenantId");
  if (!tenantId) {
    return undefined;
  }

  const flags = readBooleanFlagMap(
    rawFlags?.flags ?? rawFlags?.features ?? rawFlags?.featureFlags,
  );
  if (Object.keys(flags).length === 0) {
    return undefined;
  }

  return {
    tenant_id: tenantId,
    source:
      readMetadataText(rawFlags, "source") ||
      readMetadataText(rawFlags, "value_source") ||
      readMetadataText(rawFlags, "valueSource"),
    flags,
  };
}

function buildOemRoutingRequestMetadata():
  | HarnessOemRoutingRequestMetadata
  | undefined {
  const runtime = resolveOemCloudRuntimeContext();
  const snapshot = getOemCloudBootstrapSnapshot<OemCloudBootstrapResponse>();
  const preference = snapshot?.providerPreference;
  if (!runtime || !preference?.providerKey) {
    return undefined;
  }

  const matchedOffer = snapshot?.providerOffersSummary?.find(
    (offer) => offer.providerKey === preference.providerKey,
  );
  const providerSource = preference.providerSource || matchedOffer?.source;
  if (providerSource !== "oem_cloud") {
    return undefined;
  }

  return {
    tenant_id: runtime.tenantId,
    provider_source: providerSource,
    provider_key: preference.providerKey,
    default_model: preference.defaultModel || matchedOffer?.defaultModel,
    config_mode: matchedOffer?.configMode,
    offer_state: matchedOffer?.state,
    quota_status: matchedOffer?.quotaStatus,
    fallback_to_local_allowed: matchedOffer?.fallbackToLocalAllowed,
    can_invoke: matchedOffer?.canInvoke,
  };
}

function normalizeProviderSelection(value?: string | null): string {
  return value?.trim().toLowerCase() || "";
}

function providerMatchesOemRouting(
  providerType: string | null | undefined,
  oemRouting: HarnessOemRoutingRequestMetadata | undefined,
): boolean {
  const normalizedProvider = normalizeProviderSelection(providerType);
  const normalizedOemProvider = normalizeProviderSelection(
    oemRouting?.provider_key,
  );
  if (!normalizedProvider || !normalizedOemProvider) {
    return true;
  }

  return normalizedProvider === normalizedOemProvider;
}

function buildTenantFeatureFlagsRequestMetadata():
  | HarnessTenantFeatureFlagsRequestMetadata
  | undefined {
  const runtime = resolveOemCloudRuntimeContext();
  const snapshot = getOemCloudBootstrapSnapshot<OemCloudBootstrapResponse>();
  const features = snapshot?.features;
  if (!runtime || !features) {
    return undefined;
  }

  const flags = OEM_CLOUD_FEATURE_FLAG_KEYS.reduce<Record<string, boolean>>(
    (nextFlags, key) => {
      const value = features[key];
      if (typeof value === "boolean") {
        nextFlags[key] = value;
      }
      return nextFlags;
    },
    {},
  );
  if (Object.keys(flags).length === 0) {
    return undefined;
  }

  return {
    tenant_id: runtime.tenantId,
    source: "oem_cloud_bootstrap",
    flags,
  };
}

export function buildGeneralWorkbenchSendBoundaryState({
  isThemeWorkbench,
  contentId,
  initialDispatchKey,
  consumedInitialPromptKey,
  initialUserImages,
  mappedTheme,
  sourceText,
  sendOptions,
}: BuildGeneralWorkbenchSendBoundaryStateOptions): GeneralWorkbenchSendBoundaryState {
  const shouldConsumePendingGeneralWorkbenchInitialPrompt =
    isThemeWorkbench &&
    Boolean(contentId) &&
    Boolean(initialDispatchKey) &&
    consumedInitialPromptKey !== initialDispatchKey &&
    (initialUserImages || []).length === 0 &&
    !sendOptions?.purpose;
  const shouldDismissGeneralWorkbenchEntryPrompt =
    isThemeWorkbench && !sendOptions?.purpose;

  const browserRequirementMatch =
    mappedTheme === "general" && !sendOptions?.purpose
      ? detectBrowserTaskRequirement(sourceText)
      : null;

  return {
    sourceText,
    browserRequirementMatch,
    shouldConsumePendingGeneralWorkbenchInitialPrompt,
    shouldDismissGeneralWorkbenchEntryPrompt,
  };
}

interface BuildWorkspaceSendTextOptions {
  sourceText: string;
  contextWorkspace: ContextWorkspaceSummary;
  mentionedCharacters: Character[];
  sendOptions?: HandleSendOptions;
  preparedActiveContextPrompt?: Promise<PreparedActiveContextPromptResult> | null;
}

interface BuildWorkspaceRequestMetadataOptions {
  workspaceRequestMetadataBase?: Record<string, unknown>;
  sendOptions?: HandleSendOptions;
  savedSoulArtifactVoiceGenerationBrief?: Record<string, unknown> | null;
  soulArtifactVoiceEnabledForTurn?: boolean;
  currentProviderType?: string | null;
  effectiveToolPreferences: ChatToolPreferences;
  mappedTheme: ThemeType;
  isThemeWorkbench: boolean;
  currentGateKey: string;
  themeWorkbenchActiveQueueTitle?: string;
  contentId?: string | null;
  browserRequirementMatch?: {
    requirement: "optional" | "required" | "required_with_user_step";
    reason: string;
    launchUrl: string;
  } | null;
  workspaceSkillBindings?: AgentRuntimeWorkspaceSkillBinding[] | null;
  workspaceSkillRuntimeEnable?: WorkspaceSkillRuntimeEnableInput | null;
  agentResponseLanguage?: string | null;
}

function applyActiveContextPrompt(
  text: string,
  activeContextPrompt: string,
): string {
  if (!activeContextPrompt.trim()) {
    return text;
  }

  const slashCommandMatch = text.match(/^\/([a-zA-Z0-9_-]+)(?:\s+([\s\S]*))?$/);
  if (slashCommandMatch) {
    const [, skillName, skillArgs] = slashCommandMatch;
    const mergedArgs = [activeContextPrompt, skillArgs.trim()]
      .filter((part) => part.length > 0)
      .join("\n\n");
    return `/${skillName} ${mergedArgs}`.trim();
  }

  return `${activeContextPrompt}\n\n${text}`;
}

function applyMentionedCharacterContext(
  text: string,
  mentionedCharacters: Character[],
): string {
  if (mentionedCharacters.length === 0) {
    return text;
  }

  const characterContext = mentionedCharacters
    .map((char) => {
      let context = `角色：${char.name}`;
      if (char.description) context += `\n简介：${char.description}`;
      if (char.personality) context += `\n性格：${char.personality}`;
      if (char.background) context += `\n背景：${char.background}`;
      return context;
    })
    .join("\n\n");

  return `[角色上下文]\n${characterContext}\n\n[用户输入]\n${text}`;
}

export async function buildWorkspaceSendText(
  options: BuildWorkspaceSendTextOptions,
): Promise<string> {
  const {
    sourceText,
    contextWorkspace,
    mentionedCharacters,
    preparedActiveContextPrompt: preparedActiveContextPromptPromise,
  } = options;

  let text = sourceText;
  let activeContextPrompt = contextWorkspace.enabled
    ? contextWorkspace.activeContextPrompt.trim()
    : "";
  if (!activeContextPrompt && contextWorkspace.enabled) {
    if (preparedActiveContextPromptPromise) {
      const result = await preparedActiveContextPromptPromise;
      if (!result.ok) {
        throw result.error;
      }
      activeContextPrompt = result.value.trim();
    } else {
      activeContextPrompt = (
        await contextWorkspace.prepareActiveContextPrompt()
      ).trim();
    }
  } else if (activeContextPrompt) {
    // 发送路径优先使用现成上下文快照，后台继续补齐正文缓存以优化后续轮次。
    void contextWorkspace.prepareActiveContextPrompt().catch(() => undefined);
  }

  if (activeContextPrompt) {
    text = applyActiveContextPrompt(text, activeContextPrompt);
  }

  text = applyMentionedCharacterContext(text, mentionedCharacters);
  return text;
}

export function hasServiceSkillLaunchRequestMetadata(
  requestMetadata?: Record<string, unknown>,
): boolean {
  const harness = extractExistingHarnessMetadata(requestMetadata);
  if (!harness) {
    return false;
  }

  const siteLaunch =
    asRecord(harness.service_skill_launch) ??
    asRecord(harness.serviceSkillLaunch);
  if (siteLaunch) {
    const adapterName = siteLaunch.adapter_name ?? siteLaunch.adapterName;
    if (typeof adapterName === "string" && adapterName.trim().length > 0) {
      return true;
    }
  }

  const sceneLaunch =
    asRecord(harness.service_scene_launch) ??
    asRecord(harness.serviceSceneLaunch);
  if (!sceneLaunch) {
    return false;
  }

  const serviceSceneRun =
    asRecord(sceneLaunch.service_scene_run) ??
    asRecord(sceneLaunch.serviceSceneRun) ??
    asRecord(sceneLaunch.request_context) ??
    asRecord(sceneLaunch.requestContext);
  if (!serviceSceneRun) {
    return false;
  }

  const skillId =
    serviceSceneRun.skill_id ??
    serviceSceneRun.skillId ??
    serviceSceneRun.linked_skill_id ??
    serviceSceneRun.linkedSkillId;
  return typeof skillId === "string" && skillId.trim().length > 0;
}

export function serviceSkillLaunchRequiresProject(
  requestMetadata?: Record<string, unknown>,
): boolean {
  const harness = extractExistingHarnessMetadata(requestMetadata);
  if (!harness) {
    return false;
  }

  const siteLaunch =
    asRecord(harness.service_skill_launch) ??
    asRecord(harness.serviceSkillLaunch);
  if (siteLaunch) {
    const projectId = siteLaunch.project_id ?? siteLaunch.projectId;
    return typeof projectId !== "string" || projectId.trim().length === 0;
  }

  const sceneLaunch =
    asRecord(harness.service_scene_launch) ??
    asRecord(harness.serviceSceneLaunch);
  const serviceSceneRun =
    asRecord(sceneLaunch?.service_scene_run) ??
    asRecord(sceneLaunch?.serviceSceneRun) ??
    asRecord(sceneLaunch?.request_context) ??
    asRecord(sceneLaunch?.requestContext);
  if (!serviceSceneRun) {
    return false;
  }

  const hasSkill =
    typeof serviceSceneRun.skill_id === "string" ||
    typeof serviceSceneRun.skillId === "string" ||
    typeof serviceSceneRun.linked_skill_id === "string" ||
    typeof serviceSceneRun.linkedSkillId === "string";
  if (!hasSkill) {
    return false;
  }

  const projectId = serviceSceneRun.project_id ?? serviceSceneRun.projectId;
  return typeof projectId !== "string" || projectId.trim().length === 0;
}

export function hasModelSkillLaunchRequestMetadata(
  requestMetadata?: Record<string, unknown>,
): boolean {
  const harness = extractExistingHarnessMetadata(requestMetadata);
  if (!harness) {
    return false;
  }

  const launchKeys = [
    "image_command_intent",
    "imageCommandIntent",
    "cover_skill_launch",
    "coverSkillLaunch",
    "video_skill_launch",
    "videoSkillLaunch",
    "broadcast_skill_launch",
    "broadcastSkillLaunch",
    "resource_search_skill_launch",
    "resourceSearchSkillLaunch",
    "research_skill_launch",
    "researchSkillLaunch",
    "report_skill_launch",
    "reportSkillLaunch",
    "deep_search_skill_launch",
    "deepSearchSkillLaunch",
    "site_search_skill_launch",
    "siteSearchSkillLaunch",
    "pdf_read_skill_launch",
    "pdfReadSkillLaunch",
    "summary_skill_launch",
    "summarySkillLaunch",
    "translation_skill_launch",
    "translationSkillLaunch",
    "analysis_skill_launch",
    "analysisSkillLaunch",
    "transcription_skill_launch",
    "transcriptionSkillLaunch",
    "url_parse_skill_launch",
    "urlParseSkillLaunch",
    "typesetting_skill_launch",
    "typesettingSkillLaunch",
    "presentation_skill_launch",
    "presentationSkillLaunch",
    "form_skill_launch",
    "formSkillLaunch",
    "webpage_skill_launch",
    "webpageSkillLaunch",
    "service_scene_launch",
    "serviceSceneLaunch",
  ] as const;

  return launchKeys.some((key) => {
    const launch = asRecord(harness[key]);
    return Boolean(launch && Object.keys(launch).length > 0);
  });
}

export function buildWorkspaceRequestMetadata(
  options: BuildWorkspaceRequestMetadataOptions,
): Record<string, unknown> {
  const {
    workspaceRequestMetadataBase,
    sendOptions,
    savedSoulArtifactVoiceGenerationBrief,
    soulArtifactVoiceEnabledForTurn,
    currentProviderType,
    effectiveToolPreferences,
    mappedTheme,
    isThemeWorkbench,
    currentGateKey,
    themeWorkbenchActiveQueueTitle,
    contentId,
    browserRequirementMatch,
    workspaceSkillBindings,
    workspaceSkillRuntimeEnable,
    agentResponseLanguage,
  } = options;

  const existingHarnessMetadata = omitLegacyAccessModeFromHarnessMetadata(
    extractExistingHarnessMetadata({
      ...(workspaceRequestMetadataBase || {}),
      ...(sendOptions?.requestMetadata || {}),
    }),
  );
  const resolvedBrowserRequirement =
    browserRequirementMatch?.requirement ||
    readExistingBrowserRequirement(existingHarnessMetadata);
  const resolvedBrowserRequirementReason =
    browserRequirementMatch?.reason ||
    readMetadataText(existingHarnessMetadata, "browser_requirement_reason");
  const resolvedBrowserLaunchUrl =
    browserRequirementMatch?.launchUrl ||
    readMetadataText(existingHarnessMetadata, "browser_launch_url");
  const resolvedOemRouting =
    buildOemRoutingRequestMetadata() ||
    readExistingOemRouting(existingHarnessMetadata);
  const resolvedTenantFeatureFlags =
    buildTenantFeatureFlagsRequestMetadata() ||
    readExistingTenantFeatureFlags(existingHarnessMetadata);
  const oemRoutingMatchesCurrentProvider = providerMatchesOemRouting(
    currentProviderType,
    resolvedOemRouting,
  );
  const harnessMetadata = buildHarnessRequestMetadata({
    base: existingHarnessMetadata,
    theme: mappedTheme,
    turnPurpose: sendOptions?.purpose,
    preferences: {
      task: effectiveToolPreferences.task,
      subagent: effectiveToolPreferences.subagent,
    },
    sessionMode: isThemeWorkbench ? "general_workbench" : "default",
    gateKey: isThemeWorkbench ? currentGateKey : undefined,
    runTitle: themeWorkbenchActiveQueueTitle?.trim() || undefined,
    contentId: contentId || undefined,
    browserRequirement: resolvedBrowserRequirement,
    browserRequirementReason: resolvedBrowserRequirementReason,
    browserLaunchUrl: resolvedBrowserLaunchUrl,
    workspaceSkillBindings,
    workspaceSkillRuntimeEnable,
    agentResponseLanguage,
    oemRouting: oemRoutingMatchesCurrentProvider
      ? resolvedOemRouting
      : undefined,
    tenantFeatureFlags: resolvedTenantFeatureFlags,
  });
  if (!oemRoutingMatchesCurrentProvider) {
    delete harnessMetadata.oem_routing;
    delete harnessMetadata.oemRouting;
  }

  const hasExplicitGenerationBrief =
    hasGenerationBriefMetadata(workspaceRequestMetadataBase) ||
    hasGenerationBriefMetadata(sendOptions?.requestMetadata);
  const shouldApplySavedSoulArtifactVoice =
    soulArtifactVoiceEnabledForTurn !== false &&
    !hasExplicitGenerationBrief &&
    Boolean(savedSoulArtifactVoiceGenerationBrief);
  const baseRequestMetadata = shouldApplySavedSoulArtifactVoice
    ? mergeGenerationBriefIntoArtifactMetadata(
        workspaceRequestMetadataBase,
        savedSoulArtifactVoiceGenerationBrief,
      )
    : mergeGenerationBriefIntoArtifactMetadata(workspaceRequestMetadataBase);
  const requestMetadata = mergeSoulArtifactVoiceDiagnostics(
    mergeGenerationBriefIntoArtifactMetadata(
      mergeRequestMetadataWithArtifact(
        baseRequestMetadata,
        sendOptions?.requestMetadata,
      ),
    ),
    {
      savedGenerationBrief: savedSoulArtifactVoiceGenerationBrief,
      savedVoiceEnabledForTurn: soulArtifactVoiceEnabledForTurn,
      hasExplicitGenerationBrief,
    },
  );

  return {
    ...requestMetadata,
    harness: harnessMetadata,
  };
}

interface CreateSubmissionPreviewSnapshotOptions {
  key: string;
  prompt: string;
  images: MessageImage[];
  displayContent?: string;
  inputCapabilityRoute?: InputCapabilitySendRoute;
}

export function createSubmissionPreviewSnapshot(
  options: CreateSubmissionPreviewSnapshotOptions,
): SubmissionPreviewSnapshot {
  const { key, prompt, images, displayContent, inputCapabilityRoute } = options;

  return {
    key,
    prompt,
    displayContent,
    inputCapabilityRoute,
    images,
    createdAt: Date.now(),
  };
}

export function buildSubmissionPreviewMessages(
  snapshot: SubmissionPreviewSnapshot,
): Message[] {
  const timestamp = new Date(snapshot.createdAt);

  return [
    {
      id: `submission-preview:${snapshot.key}:user`,
      role: "user",
      content: snapshot.displayContent ?? snapshot.prompt,
      images: snapshot.images.length > 0 ? snapshot.images : undefined,
      timestamp,
      inputCapabilityRoute: snapshot.inputCapabilityRoute,
    },
    {
      id: `submission-preview:${snapshot.key}:assistant`,
      role: "assistant",
      content: "",
      timestamp: new Date(timestamp.getTime() + 1),
      isThinking: true,
      contentParts: [],
      runtimeStatus: {
        phase: "preparing",
        title: "正在准备回复",
        detail: "已收到请求，正在准备会话和执行环境。",
        checkpoints: ["请求已接收", "等待会话就绪", "准备开始输出"],
      },
    },
  ];
}

export function buildInitialDispatchPreviewMessages(
  snapshot: InitialDispatchPreviewSnapshot,
): Message[] {
  const timestamp = new Date();
  const normalizedPrompt = snapshot.prompt?.trim() || "";

  return [
    {
      id: `initial-dispatch-preview:${snapshot.key}:user`,
      role: "user",
      content: normalizedPrompt,
      images: snapshot.images.length > 0 ? snapshot.images : undefined,
      timestamp,
    },
    {
      id: `initial-dispatch-preview:${snapshot.key}:assistant`,
      role: "assistant",
      content: "任务已进入处理队列…",
      timestamp: new Date(timestamp.getTime() + 1),
      isThinking: true,
      contentParts: [],
      runtimeStatus: {
        phase: "preparing",
        title: "正在准备回复",
        detail: "已收到请求，正在等待会话接管。",
        checkpoints: ["请求已接收", "等待会话就绪", "准备开始输出"],
      },
    },
  ];
}
