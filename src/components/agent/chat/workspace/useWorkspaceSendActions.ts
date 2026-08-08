import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import type { Dispatch, SetStateAction } from "react";
import type { RuntimeSearchMode } from "@limecloud/app-server-client";
import type { AutoContinueRequestPayload } from "@/lib/api/agentRuntime/sessionTypes";
import { getOrCreateDefaultProject } from "@/lib/api/project";
import type { AgentRuntimeWorkspaceSkillBinding } from "@/lib/api/agentRuntime/toolInventoryTypes";
import { normalizeExecutionStrategyToReact } from "@/lib/api/agentRuntime/executionStrategyCompat";
import type { ServiceModelsConfig } from "@/lib/api/appConfigTypes";
import { logAgentDebug } from "@/lib/agentDebug";
import { recordAgentUiPerformanceMetric } from "@/lib/agentUiPerformanceMetrics";
import { readGlobalMediaGenerationDefaults } from "@/hooks/useGlobalMediaGenerationDefaults";
import type { MediaGenerationDefaults } from "@/lib/mediaGeneration";
import {
  mergeServiceModelPrompt,
  resolveServiceModelExecutionPreference,
} from "@/lib/serviceModels";
import { parseAnalysisWorkbenchCommand } from "../utils/analysisWorkbenchCommand";
import { parseBrowserWorkbenchCommand } from "../utils/browserWorkbenchCommand";
import { parseBroadcastWorkbenchCommand } from "../utils/broadcastWorkbenchCommand";
import { parseChannelPreviewWorkbenchCommand } from "../utils/channelPreviewWorkbenchCommand";
import { parseComplianceWorkbenchCommand } from "../utils/complianceWorkbenchCommand";
import { parseCompetitorWorkbenchCommand } from "../utils/competitorWorkbenchCommand";
import { parseCoverWorkbenchCommand } from "../utils/coverWorkbenchCommand";
import { parseDeepSearchWorkbenchCommand } from "../utils/deepSearchWorkbenchCommand";
import { parseFileReadWorkbenchCommand } from "../utils/fileReadWorkbenchCommand";
import { parseFormWorkbenchCommand } from "../utils/formWorkbenchCommand";
import { parseGrowthWorkbenchCommand } from "../utils/growthWorkbenchCommand";
import { parseImageWorkbenchCommand } from "../utils/imageWorkbenchCommand";
import { parseLogoDecompositionWorkbenchCommand } from "../utils/logoDecompositionWorkbenchCommand";
import { parsePdfWorkbenchCommand } from "../utils/pdfWorkbenchCommand";
import { parsePosterWorkbenchCommand } from "../utils/posterWorkbenchCommand";
import { parsePresentationWorkbenchCommand } from "../utils/presentationWorkbenchCommand";
import { parsePublishWorkbenchCommand } from "../utils/publishWorkbenchCommand";
import { parseReportWorkbenchCommand } from "../utils/reportWorkbenchCommand";
import { parseResourceSearchWorkbenchCommand } from "../utils/resourceSearchWorkbenchCommand";
import { parseSearchWorkbenchCommand } from "../utils/searchWorkbenchCommand";
import { parseSiteSearchWorkbenchCommand } from "../utils/siteSearchWorkbenchCommand";
import { parseSummaryWorkbenchCommand } from "../utils/summaryWorkbenchCommand";
import { parseTranslationWorkbenchCommand } from "../utils/translationWorkbenchCommand";
import { resolveMentionCommandMergedPrefillReplayText } from "../utils/mentionCommandReplayText";
import {
  parseMentionCommand,
  resolveMentionCommandPrefixMatch,
} from "../utils/mentionCommandPrefixMatch";
import { parseTranscriptionWorkbenchCommand } from "../utils/transcriptionWorkbenchCommand";
import { parseTypesettingWorkbenchCommand } from "../utils/typesettingWorkbenchCommand";
import { parseUploadWorkbenchCommand } from "../utils/uploadWorkbenchCommand";
import {
  isUrlParseReadTrigger,
  isUrlParseScrapeTrigger,
  parseUrlParseWorkbenchCommand,
} from "../utils/urlParseWorkbenchCommand";
import { parseVideoWorkbenchCommand } from "../utils/videoWorkbenchCommand";
import { resolveAgentRuntimeSubmitErrorMessage } from "../utils/agentRuntimeErrorPresentation";
import { parseVoiceWorkbenchCommand } from "../utils/voiceWorkbenchCommand";
import { parseWritingWorkbenchCommand } from "../utils/writingWorkbenchCommand";
import { parseWebpageWorkbenchCommand } from "../utils/webpageWorkbenchCommand";
import { resolvePlainInputIntentConfirmation } from "../utils/plainInputIntentConfirmation";
import { detectBrowserTaskRequirement } from "../utils/browserTaskRequirement";
import { isTeamRuntimeRecommendation } from "../utils/contextualRecommendations";
import {
  matchAutoLaunchSiteSkillFromText,
  type AutoMatchedSiteSkill,
} from "../service-skills/autoMatchSiteSkill";
import {
  saveChatToolPreferences,
  type ChatToolPreferences,
} from "../utils/chatToolPreferences";
import type { HandleSendOptions } from "../hooks/handleSendTypes";
import { extractAgentUiPerformanceTraceMetadata } from "../hooks/agentStreamPerformanceMetrics";
import type { SendMessageFn } from "../hooks/agentChatShared";
import { normalizeExecutionStrategy } from "../hooks/agentChatCoreUtils";
import type {
  BrowserAssistSessionState,
  Message,
  MessageImage,
} from "../types";
import type { AgentAccessMode } from "../hooks/agentChatStorage";
import {
  buildSubmissionPreviewMessages,
  type GeneralWorkbenchSendBoundaryState,
  type InitialDispatchPreviewSnapshot,
  createSubmissionPreviewSnapshot,
  type SubmissionPreviewSnapshot,
  buildWorkspaceRequestMetadata,
  buildWorkspaceSendText,
  hasModelSkillLaunchRequestMetadata,
  hasServiceSkillLaunchRequestMetadata,
  serviceSkillLaunchRequiresProject,
  primeBrowserAssistBeforeSend,
  type ContextWorkspaceSummary,
  type EnsureBrowserAssistCanvasOptions,
} from "./workspaceSendHelpers";
import type { Character } from "@/lib/api/projectMemory";
import type { ThemeType } from "@/lib/workspace/workbenchContract";
import type {
  ServiceSkillHomeItem,
  ServiceSkillSlotValues,
} from "../service-skills/types";
import type { ImageWorkbenchCommandRequest } from "./imageCommandIntent";
import type { WorkspaceSkillRuntimeEnableInput } from "../utils/workspaceSkillBindingsMetadata";
import {
  attachSessionIdToRequestContext,
  attachSessionIdToScopedRequestContext,
  buildAnalysisSkillLaunchRequestContext,
  buildBroadcastSkillLaunchRequestContext,
  buildFormSkillLaunchRequestContext,
  buildPresentationSkillLaunchRequestContext,
  buildResourceSearchSkillLaunchRequestContext,
  buildSkillLaunchRequestMetadata,
  buildSummarySkillLaunchRequestContext,
  buildTranscriptionSkillLaunchRequestContext,
  buildTranslationSkillLaunchRequestContext,
  buildTypesettingSkillLaunchRequestContext,
  buildUrlParseSkillLaunchRequestContext,
  buildWebpageSkillLaunchRequestContext,
  extractBoundSessionRequestContext,
  type PendingCommandSessionBinding,
} from "./workspaceModelSkillLaunchRequestContext";
import { buildBrowserControlLaunchRequestMetadata } from "./browserControlLaunch";
import {
  buildServiceSceneLaunchRequestMetadata,
  parseRuntimeSceneCommand,
  RuntimeSceneLaunchValidationError,
  resolveRuntimeSceneLaunchRequest,
} from "./serviceSkillSceneLaunch";
import {
  resolveInputCapabilityDispatchContext,
  type CompletedInputCapabilitySlashUsage,
} from "./inputCapabilityRouting";
import type { RuntimeSceneGateRequest } from "./sceneSkillGate";
import {
  getMentionEntryUsageMap,
  getMentionEntryUsageRecordKey,
  recordMentionEntryUsage,
} from "../skill-selection/mentionEntryUsage";
import { useRuntimeMentionCommandCatalog } from "../skill-selection/runtimeInputCapabilityCatalog";
import { recordServiceSkillUsage } from "../service-skills/storage";
import { recordSlashEntryUsage } from "../skill-selection/slashEntryUsage";
import { CONTENT_POST_SKILL_KEY } from "../utils/contentPostSkill";
import { parseSkillInstallPromptInstruction } from "@/lib/skills/skillInstallPrompt";
import {
  mergeSummaryCommandRecentDefaults,
  mergeTranslationCommandRecentDefaults,
  mergeAnalysisCommandRecentDefaults,
  mergeComplianceCommandRecentDefaults,
  mergeTypesettingCommandRecentDefaults,
  mergePresentationCommandRecentDefaults,
  mergeFormCommandRecentDefaults,
  mergeWebpageCommandRecentDefaults,
  mergePublishLikeCommandRecentDefaults,
} from "./commands/commandRecentDefaults";
import {
  buildPublishDispatchBody,
  buildChannelPreviewDispatchBody,
  buildUploadDispatchBody,
  buildWritingDispatchBody,
} from "./commands/dispatchBodyBuilders";
import { asRecord } from "./commands/skillSlotUtils";
import { waitForNextPaint } from "./commands/sendHelpers";
import {
  isImageGenerationPlainInputIntent,
  isLikelyPlainImageGenerationRequest,
} from "./commands/intentHelpers";
import {
  resolveServiceModelSendOverrides,
  shouldRefreshServiceModelsBeforeSend,
  withConfiguredModelSlots,
} from "./commands/serviceModelHelpers";
import { shouldSkipBrowserAssistPrimeForPlainFirstTurn } from "./commands/browserAssistHelpers";
import { buildImageWorkbenchAssistantDraft } from "./commands/imageWorkbenchHelpers";
import { resolveSkillInstallPromptConfirmation } from "./commands/skillInstallHelpers";
import {
  buildFileReadSkillLaunchRequestContext,
  buildVideoSkillLaunchRequestContext,
  buildCoverSkillLaunchRequestContext,
  buildResearchSkillLaunchRequestContext,
  buildDeepSearchSkillLaunchRequestContext,
  buildReportSkillLaunchRequestContext,
  buildCompetitorSkillLaunchRequestContext,
  buildSiteSearchSkillLaunchRequestContext,
  buildPdfReadSkillLaunchRequestContext,
} from "./commands/skillLaunchContextBuilders";
import {
  resolveGrowthSkillLaunchRequestContext,
  resolveVoiceSkillLaunchRequestContext,
} from "./commands/skillLaunchResolvers";
import {
  resolveMentionCommandUsageSlotValues,
  resolveImageMentionCommandKey,
  normalizeMentionCommandReplayText,
  resolveMentionCommandReplayText,
  resolveBareMentionCommandPrefillSourceText,
} from "./commands/mentionCommandUtils";
import { resolveMentionCommandUsage } from "./commands/mentionCommandUtils";

type CurrentExecutionStrategy = "react";

export type WorkspaceHandleSend = (
  images?: MessageImage[],
  webSearch?: boolean,
  thinking?: boolean,
  textOverride?: string,
  sendExecutionStrategy?: CurrentExecutionStrategy,
  autoContinuePayload?: AutoContinueRequestPayload,
  sendOptions?: HandleSendOptions,
) => Promise<boolean>;
type SetStringState = (value: string) => void;
type ParsedImageWorkbenchCommand = NonNullable<
  ReturnType<typeof parseImageWorkbenchCommand>
>;
type CompletedMentionUsage = {
  skillId: string;
  runnerType: ServiceSkillHomeItem["runnerType"];
  slotValues?: ServiceSkillSlotValues;
  launchUserInput?: string;
};
type CompletedMentionCommandUsage = {
  entryId: string;
  replayText?: string;
  slotValues?: ServiceSkillSlotValues;
};
// normalizeServiceSkillUsageSlotValue 已提取到 ./commands/skillSlotUtils.ts

// MENTION_USAGE_REQUEST_FIELDS + resolve*Mention* 函数组已提取到 ./commands/mentionCommandUtils.ts

interface UseWorkspaceSendActionsParams {
  // 命令 recent defaults 合并函数已提取到 ./commands/commandRecentDefaults.ts
  input: string;
  setInput: SetStringState;
  mentionedCharacters: Character[];
  setMentionedCharacters: Dispatch<SetStateAction<Character[]>>;
  chatToolPreferences: ChatToolPreferences;
  setChatToolPreferences: Dispatch<SetStateAction<ChatToolPreferences>>;
  serviceSkills: ServiceSkillHomeItem[];
  activeTheme: string;
  mappedTheme: ThemeType;
  isThemeWorkbench: boolean;
  contextWorkspace: ContextWorkspaceSummary;
  projectId?: string | null;
  projectRootPath?: string | null;
  sessionId?: string | null;
  executionStrategy: CurrentExecutionStrategy;
  accessMode?: AgentAccessMode;
  providerType?: string | null;
  workspaceSkillBindings?: AgentRuntimeWorkspaceSkillBinding[] | null;
  workspaceSkillRuntimeEnable?: WorkspaceSkillRuntimeEnableInput | null;
  currentGateKey: string;
  themeWorkbenchActiveQueueTitle?: string;
  contentId?: string | null;
  browserAssistProfileKey?: string | null;
  browserAssistPreferredBackend?:
    | "current"
    | "lime_extension_bridge"
    | "cdp_direct"
    | null;
  browserAssistAutoLaunch?: boolean | null;
  browserAssistSessionState?: BrowserAssistSessionState | null;
  workspaceRequestMetadataBase?: Record<string, unknown>;
  savedSoulArtifactVoiceGenerationBrief?: Record<string, unknown> | null;
  soulArtifactVoiceEnabledForTurn?: boolean;
  serviceModels?: ServiceModelsConfig;
  agentResponseLanguage?: string | null;
  resolveServiceModelsBeforeSend?: () => Promise<{
    serviceModels?: ServiceModelsConfig;
    agentResponseLanguage?: string | null;
  }>;
  messages: Message[];
  setChatMessages: Dispatch<SetStateAction<Message[]>>;
  bootstrapDispatchPreview?: InitialDispatchPreviewSnapshot | null;
  sendMessage: SendMessageFn;
  resolveSendBoundary: (input: {
    sourceText: string;
    sendOptions?: HandleSendOptions;
  }) => GeneralWorkbenchSendBoundaryState;
  finalizeAfterSendSuccess: (
    boundary: GeneralWorkbenchSendBoundaryState,
  ) => void;
  rollbackAfterSendFailure: (
    boundary: GeneralWorkbenchSendBoundaryState,
  ) => void;
  ensureBrowserAssistCanvas: (
    target: string,
    options?: EnsureBrowserAssistCanvasOptions,
  ) => Promise<boolean>;
  handleAutoLaunchMatchedSiteSkill: (
    match: AutoMatchedSiteSkill<ServiceSkillHomeItem>,
  ) => Promise<void>;
  openRuntimeSceneGate?: (
    request: RuntimeSceneGateRequest,
  ) => Promise<void> | void;
  ensureSessionForCommandMetadata?: (options?: {
    targetSessionId?: string;
    skipSessionRestore?: boolean;
    skipSessionStartHooks?: boolean;
  }) => Promise<string | null>;
  prepareImageWorkbenchSkillSend?: () => boolean | Promise<boolean>;
  resolveImageWorkbenchCommandRequest: (input: {
    rawText: string;
    parsedCommand: ParsedImageWorkbenchCommand;
    images: MessageImage[];
    sessionIdOverride?: string | null;
    entrySource?: string;
    projectId?: string | null;
    projectRootPath?: string | null;
  }) => ImageWorkbenchCommandRequest | null;
}

interface WorkspaceResolvedSendState {
  sourceText: string;
  dispatchText: string;
  sendBoundary: GeneralWorkbenchSendBoundaryState;
  browserRequirementForSend: GeneralWorkbenchSendBoundaryState["browserRequirementMatch"];
  effectiveToolPreferences: ChatToolPreferences;
  effectiveWebSearch?: boolean;
  effectiveSearchMode?: RuntimeSearchMode;
  submissionPreviewKey: string | null;
}

interface WorkspaceSendPlan extends WorkspaceResolvedSendState {
  text: string;
  images: MessageImage[];
  hasContextWorkspace: boolean;
  sendExecutionStrategy?: CurrentExecutionStrategy;
  autoContinuePayload?: AutoContinueRequestPayload;
  sendOptions?: HandleSendOptions;
  completedMentionCommandUsage: CompletedMentionCommandUsage | null;
  completedMentionUsage: CompletedMentionUsage | null;
  completedSlashUsage?: CompletedInputCapabilitySlashUsage | null;
}

async function resolveImageCommandProjectContext(params: {
  projectId?: string | null;
  projectRootPath?: string | null;
}): Promise<{ projectId: string | null; projectRootPath: string | null }> {
  const projectId = params.projectId?.trim() || null;
  const projectRootPath = params.projectRootPath?.trim() || null;
  if (projectRootPath) {
    return { projectId, projectRootPath };
  }
  if (projectId) {
    return { projectId, projectRootPath: null };
  }

  const defaultProject = await getOrCreateDefaultProject();
  return {
    projectId: defaultProject.id?.trim() || null,
    projectRootPath: defaultProject.rootPath?.trim() || null,
  };
}

interface WorkspaceLocalConfirmationPlan {
  sourceText: string;
  images: MessageImage[];
  sendBoundary: GeneralWorkbenchSendBoundaryState;
  submissionPreviewKey: string | null;
  confirmation: string;
}

type WorkspaceSendResolution =
  | {
      kind: "done";
      result: boolean;
    }
  | {
      kind: "local_confirmation";
      plan: WorkspaceLocalConfirmationPlan;
    }
  | {
      kind: "ready";
      plan: WorkspaceSendPlan;
    };

type AgentWorkspaceTranslator = (
  key: string,
  options?: Record<string, unknown>,
) => string;

export function useWorkspaceSendActions({
  input,
  setInput,
  mentionedCharacters,
  setMentionedCharacters,
  chatToolPreferences,
  setChatToolPreferences,
  serviceSkills,
  activeTheme,
  mappedTheme,
  isThemeWorkbench,
  contextWorkspace,
  projectId,
  projectRootPath,
  sessionId: _sessionId,
  executionStrategy,
  accessMode: _accessMode,
  providerType,
  workspaceSkillBindings,
  workspaceSkillRuntimeEnable,
  currentGateKey,
  themeWorkbenchActiveQueueTitle,
  contentId,
  browserAssistProfileKey,
  browserAssistPreferredBackend,
  browserAssistAutoLaunch,
  browserAssistSessionState,
  workspaceRequestMetadataBase,
  savedSoulArtifactVoiceGenerationBrief,
  soulArtifactVoiceEnabledForTurn,
  serviceModels,
  agentResponseLanguage,
  resolveServiceModelsBeforeSend,
  messages,
  setChatMessages,
  sendMessage,
  resolveSendBoundary,
  finalizeAfterSendSuccess,
  rollbackAfterSendFailure,
  ensureBrowserAssistCanvas,
  handleAutoLaunchMatchedSiteSkill,
  openRuntimeSceneGate,
  ensureSessionForCommandMetadata,
  prepareImageWorkbenchSkillSend,
  resolveImageWorkbenchCommandRequest,
}: UseWorkspaceSendActionsParams) {
  const { t } = useTranslation("agent");
  const messagesCount = messages.length;
  const [submissionPreview, setSubmissionPreview] =
    useState<SubmissionPreviewSnapshot | null>(null);
  const [isPreparingSend, setIsPreparingSend] = useState(false);
  const isPreparingSendRef = useRef(false);
  const translateAgentWorkspace = useCallback<AgentWorkspaceTranslator>(
    (key, options) => {
      const translate = t as unknown as (
        key: string,
        options?: Record<string, unknown>,
      ) => string;
      return String(translate(key, options));
    },
    [t],
  );
  const {
    mentionAgentTurnRouteMap,
    mentionCommandSkillIdMap,
    mentionCommandPrefixKeyMap,
  } = useRuntimeMentionCommandCatalog();
  const resourcePromptRewritePreference =
    serviceModels?.resource_prompt_rewrite;
  const submissionPreviewMessages = useMemo(
    () =>
      messagesCount === 0 && submissionPreview
        ? buildSubmissionPreviewMessages(submissionPreview)
        : [],
    [messagesCount, submissionPreview],
  );
  const displayMessages = useMemo(() => {
    if (submissionPreviewMessages.length > 0) {
      return submissionPreviewMessages;
    }

    return messages;
  }, [messages, submissionPreviewMessages]);

  const resolveSendExecutionPlan = useCallback(
    async (
      images?: MessageImage[],
      _webSearch?: boolean,
      _thinking?: boolean,
      textOverride?: string,
      sendExecutionStrategy?: CurrentExecutionStrategy,
      autoContinuePayload?: AutoContinueRequestPayload,
      sendOptions?: HandleSendOptions,
    ): Promise<WorkspaceSendResolution> => {
      const planStartedAt = Date.now();
      const inputCapabilityDispatch = resolveInputCapabilityDispatchContext({
        sourceText: textOverride ?? input,
        capabilityRoute: sendOptions?.capabilityRoute,
        displayContent: sendOptions?.displayContent,
      });
      let effectiveSendExecutionStrategy = normalizeExecutionStrategy(
        sendExecutionStrategy ?? executionStrategy,
      );
      let sourceText = inputCapabilityDispatch.sourceText;
      logAgentDebug("WorkspaceSend", "plan.start", {
        hasAutoContinue: Boolean(autoContinuePayload?.enabled),
        hasPurpose: Boolean(sendOptions?.purpose),
        imagesCount: images?.length ?? 0,
        messagesCount,
        sourceTextLength: sourceText.trim().length,
      });
      if (!sourceText.trim() && (!images || images.length === 0)) {
        logAgentDebug("WorkspaceSend", "plan.empty", {
          durationMs: Date.now() - planStartedAt,
        });
        return { kind: "done", result: false };
      }
      let effectiveImages = images || [];
      const sendBoundary = resolveSendBoundary({
        sourceText,
        sendOptions,
      });
      sourceText = sendBoundary.sourceText;
      sourceText =
        resolveBareMentionCommandPrefillSourceText(
          sourceText,
          mentionCommandPrefixKeyMap,
        ) || sourceText;
      const mentionUsageMap = getMentionEntryUsageMap();
      type MergeableMentionParsedCommand = Parameters<
        typeof resolveMentionCommandMergedPrefillReplayText
      >[0]["parsedCommand"];
      const maybeApplyMentionCommandRecentDefaults = <
        T extends MergeableMentionParsedCommand & {
          rawText: string;
        },
      >(params: {
        rawText: string;
        commandKey: string;
        parsedCommand: T;
        reparse: (rawText: string) => T | null;
      }): { rawText: string; parsedCommand: T } => {
        const recentRecord = mentionUsageMap.get(
          getMentionEntryUsageRecordKey("builtin_command", params.commandKey),
        );
        if (!recentRecord?.slotValues) {
          return {
            rawText: params.rawText,
            parsedCommand: params.parsedCommand,
          };
        }

        const nextReplayText = resolveMentionCommandMergedPrefillReplayText({
          commandKey: params.commandKey,
          parsedCommand: params.parsedCommand,
          slotValues: recentRecord.slotValues,
        });
        const currentReplayText = resolveMentionCommandReplayText(
          params.parsedCommand,
          params.commandKey,
        );
        if (!nextReplayText || nextReplayText === currentReplayText) {
          return {
            rawText: params.rawText,
            parsedCommand: params.parsedCommand,
          };
        }

        const commandPrefixMatch = resolveMentionCommandPrefixMatch(
          params.rawText,
          mentionCommandPrefixKeyMap,
        );
        if (!commandPrefixMatch) {
          return {
            rawText: params.rawText,
            parsedCommand: params.parsedCommand,
          };
        }

        const nextRawText = `${commandPrefixMatch.commandPrefix} ${nextReplayText}`;
        const reparsed = params.reparse(nextRawText);
        if (!reparsed) {
          return {
            rawText: params.rawText,
            parsedCommand: params.parsedCommand,
          };
        }

        return {
          rawText: nextRawText,
          parsedCommand: reparsed,
        };
      };
      let dispatchText = sourceText;

      let effectiveToolPreferences =
        sendOptions?.toolPreferencesOverride ?? chatToolPreferences;
      const { browserRequirementMatch } = sendBoundary;
      let browserRequirementForSend = browserRequirementMatch;
      const mergedLaunchRequestMetadata = {
        ...(workspaceRequestMetadataBase || {}),
        ...(sendOptions?.requestMetadata || {}),
      };
      let hasBoundSkillLaunch =
        hasServiceSkillLaunchRequestMetadata(mergedLaunchRequestMetadata) ||
        hasModelSkillLaunchRequestMetadata(mergedLaunchRequestMetadata);
      let effectiveWebSearch =
        browserRequirementMatch &&
        browserRequirementMatch.requirement !== "optional"
          ? false
          : undefined;
      let effectiveSearchMode =
        browserRequirementMatch &&
        browserRequirementMatch.requirement !== "optional"
          ? "disabled"
          : sendOptions?.searchMode;

      const {
        activeContextPrompt,
        enabled: contextWorkspaceEnabled,
        prepareActiveContextPrompt,
      } = contextWorkspace;
      const shouldAttachContextWorkspace =
        Boolean(activeContextPrompt.trim()) ||
        (contextWorkspaceEnabled && isThemeWorkbench);
      const effectiveContextWorkspace = {
        ...contextWorkspace,
        enabled: shouldAttachContextWorkspace,
      };
      const preparedActiveContextPrompt =
        effectiveContextWorkspace.enabled && !activeContextPrompt.trim()
          ? prepareActiveContextPrompt().then(
              (value) => ({
                ok: true as const,
                value,
              }),
              (error) => ({
                ok: false as const,
                error,
              }),
            )
          : null;

      let commandSessionId: string | null | undefined;
      let commandSessionPromise: Promise<string | null> | null = null;
      let pendingCommandSessionBinding: PendingCommandSessionBinding | null =
        extractBoundSessionRequestContext(mergedLaunchRequestMetadata);
      let pendingCommandSessionBindingMode: "blocking" | "best_effort" =
        "blocking";
      let completedMentionCommandUsage:
        | WorkspaceSendPlan["completedMentionCommandUsage"]
        | null = null;
      let completedMentionUsage: WorkspaceSendPlan["completedMentionUsage"] =
        null;
      let completedSlashUsage: WorkspaceSendPlan["completedSlashUsage"] =
        inputCapabilityDispatch.completedSlashUsage;
      sendOptions = inputCapabilityDispatch.capabilityRoute
        ? {
            ...(sendOptions || {}),
            capabilityRoute: inputCapabilityDispatch.capabilityRoute,
          }
        : sendOptions;
      let submissionPreviewKey: string | null = null;
      const ensureSubmissionPreview = (previewImages = effectiveImages) => {
        if (submissionPreviewKey) {
          return submissionPreviewKey;
        }

        submissionPreviewKey = crypto.randomUUID();
        setSubmissionPreview(
          createSubmissionPreviewSnapshot({
            key: submissionPreviewKey,
            prompt: sourceText,
            displayContent: sendOptions?.displayContent,
            inputCapabilityRoute: sendOptions?.capabilityRoute,
            images: previewImages,
          }),
        );
        return submissionPreviewKey;
      };
      const clearSubmissionPreview = () => {
        if (!submissionPreviewKey) {
          return;
        }
        setSubmissionPreview(null);
      };
      if (sendOptions?.skipWorkspaceCommandRouting === true) {
        let text: string;
        try {
          text = await buildWorkspaceSendText({
            sourceText: dispatchText,
            contextWorkspace: effectiveContextWorkspace,
            mentionedCharacters,
            sendOptions,
            preparedActiveContextPrompt,
          });
        } catch (error) {
          clearSubmissionPreview();
          throw error;
        }
        const performanceTrace = extractAgentUiPerformanceTraceMetadata(
          sendOptions?.requestMetadata,
        );
        if (performanceTrace?.sessionId || performanceTrace?.requestId) {
          recordAgentUiPerformanceMetric("workspaceSend.plan.ready", {
            durationMs: Date.now() - planStartedAt,
            hasPendingSessionBinding: false,
            primedSessionId: null,
            requestId: performanceTrace.requestId ?? null,
            sessionId: performanceTrace.sessionId ?? null,
            skippedWorkspaceCommandRouting: true,
            source: performanceTrace.source ?? "workspace-send",
            workspaceId: performanceTrace.workspaceId ?? null,
          });
        }
        logAgentDebug("WorkspaceSend", "plan.ready", {
          durationMs: Date.now() - planStartedAt,
          hasPendingSessionBinding: false,
          skippedWorkspaceCommandRouting: true,
          sourceTextLength: sourceText.trim().length,
        });
        return {
          kind: "ready",
          plan: {
            sourceText,
            dispatchText,
            text,
            images: effectiveImages,
            hasContextWorkspace: shouldAttachContextWorkspace,
            sendBoundary,
            browserRequirementForSend,
            effectiveToolPreferences,
            effectiveWebSearch,
            effectiveSearchMode,
            submissionPreviewKey: null,
            sendExecutionStrategy: effectiveSendExecutionStrategy,
            autoContinuePayload,
            sendOptions,
            completedMentionCommandUsage: null,
            completedMentionUsage: null,
            completedSlashUsage,
          },
        };
      }
      const resolveCommandSessionEnsureOptions = () => ({
        targetSessionId: sendOptions?.targetSessionId?.trim() || undefined,
        skipSessionRestore: sendOptions?.skipSessionRestore === true,
        skipSessionStartHooks: sendOptions?.skipSessionStartHooks === true,
      });
      const primeCommandSessionId = (
        reason = "unspecified",
        ensureOptions = resolveCommandSessionEnsureOptions(),
      ) => {
        if (commandSessionId !== undefined) {
          logAgentDebug("WorkspaceSend", "primeSession.reuseResolved", {
            reason,
            sessionId: commandSessionId,
            skipSessionRestore: ensureOptions.skipSessionRestore,
            skipSessionStartHooks: ensureOptions.skipSessionStartHooks,
            targetSessionId: ensureOptions.targetSessionId ?? null,
          });
          return Promise.resolve(commandSessionId);
        }
        if (!commandSessionPromise) {
          const primeStartedAt = Date.now();
          logAgentDebug("WorkspaceSend", "primeSession.start", {
            reason,
            skipSessionRestore: ensureOptions.skipSessionRestore,
            skipSessionStartHooks: ensureOptions.skipSessionStartHooks,
            targetSessionId: ensureOptions.targetSessionId ?? null,
          });
          commandSessionPromise = (async () => {
            try {
              const resolvedSessionId =
                await ensureSessionForCommandMetadata?.(ensureOptions);
              commandSessionId = resolvedSessionId?.trim() || null;
              logAgentDebug("WorkspaceSend", "primeSession.done", {
                durationMs: Date.now() - primeStartedAt,
                reason,
                sessionId: commandSessionId,
                skipSessionRestore: ensureOptions.skipSessionRestore,
                skipSessionStartHooks: ensureOptions.skipSessionStartHooks,
                targetSessionId: ensureOptions.targetSessionId ?? null,
              });
              return commandSessionId;
            } catch (error) {
              logAgentDebug(
                "WorkspaceSend",
                "primeSession.error",
                {
                  durationMs: Date.now() - primeStartedAt,
                  error,
                  reason,
                  skipSessionRestore: ensureOptions.skipSessionRestore,
                  skipSessionStartHooks: ensureOptions.skipSessionStartHooks,
                  targetSessionId: ensureOptions.targetSessionId ?? null,
                },
                { level: "error" },
              );
              throw error;
            }
          })();
        } else {
          logAgentDebug("WorkspaceSend", "primeSession.reusePending", {
            reason,
          });
        }
        return commandSessionPromise;
      };
      const ensureCommandSessionId = async () => {
        return primeCommandSessionId(
          "await_binding",
          resolveCommandSessionEnsureOptions(),
        );
      };
      const markCompletedMentionCommand = (
        commandKey: string,
        replayText?: string,
      ) => {
        completedMentionCommandUsage = {
          entryId: commandKey,
          replayText: normalizeMentionCommandReplayText(replayText),
          slotValues: resolveMentionCommandUsageSlotValues(
            sendOptions?.requestMetadata,
          ),
        };
        completedMentionUsage = resolveMentionCommandUsage({
          commandKey,
          serviceSkills,
          requestMetadata: sendOptions?.requestMetadata,
          mentionCommandSkillIdMap,
        });
      };

      const skillInstallPromptInstruction =
        !sendOptions?.purpose && !hasBoundSkillLaunch && sourceText.trim()
          ? parseSkillInstallPromptInstruction(sourceText)
          : null;
      if (skillInstallPromptInstruction) {
        const confirmation = await resolveSkillInstallPromptConfirmation(
          skillInstallPromptInstruction,
          translateAgentWorkspace,
        );
        return {
          kind: "local_confirmation",
          plan: {
            sourceText,
            images: effectiveImages,
            sendBoundary,
            submissionPreviewKey,
            confirmation,
          },
        };
      }

      const explicitImageWorkbenchCommand =
        !sendOptions?.purpose && !hasBoundSkillLaunch && sourceText.trim()
          ? parseImageWorkbenchCommand(sourceText)
          : null;
      const plainImageIntent =
        !sendOptions?.purpose &&
        !hasBoundSkillLaunch &&
        !explicitImageWorkbenchCommand
          ? resolvePlainInputIntentConfirmation(sourceText)
          : null;
      const parsedPlainImageWorkbenchCommand =
        (plainImageIntent &&
          isImageGenerationPlainInputIntent(plainImageIntent)) ||
        (!plainImageIntent &&
          isLikelyPlainImageGenerationRequest(sourceText.trim()))
          ? parseImageWorkbenchCommand(`@配图 ${sourceText.trim()}`)
          : null;
      const parsedImageWorkbenchCommand =
        explicitImageWorkbenchCommand ?? parsedPlainImageWorkbenchCommand;
      if (parsedImageWorkbenchCommand) {
        const imageCommandProjectContext =
          await resolveImageCommandProjectContext({
            projectId,
            projectRootPath,
          });
        if (!imageCommandProjectContext.projectRootPath) {
          clearSubmissionPreview();
          toast.error("默认项目目录未就绪，暂时无法创建图片任务");
          return { kind: "done", result: false };
        }
        if (prepareImageWorkbenchSkillSend) {
          const prepared = await prepareImageWorkbenchSkillSend();
          if (!prepared) {
            clearSubmissionPreview();
            return { kind: "done", result: false };
          }
        }
        const imageDispatchText =
          parsedPlainImageWorkbenchCommand?.rawText || sourceText;
        const skillRequest = resolveImageWorkbenchCommandRequest({
          rawText: imageDispatchText,
          parsedCommand: parsedImageWorkbenchCommand,
          images: effectiveImages,
          sessionIdOverride: commandSessionId,
          entrySource: parsedImageWorkbenchCommand.entrySource,
          projectId: imageCommandProjectContext.projectId,
          projectRootPath: imageCommandProjectContext.projectRootPath,
        });
        if (!skillRequest) {
          clearSubmissionPreview();
          return { kind: "done", result: false };
        }
        effectiveImages =
          skillRequest.images.length > 0
            ? skillRequest.images
            : effectiveImages;
        const mentionCommandKey = resolveImageMentionCommandKey(
          parsedImageWorkbenchCommand,
        );
        pendingCommandSessionBinding = {
          kind: "request_context",
          requestContext: skillRequest.requestContext,
          requestContextKey: "image_task",
        };
        pendingCommandSessionBindingMode = parsedPlainImageWorkbenchCommand
          ? "best_effort"
          : "blocking";
        ensureSubmissionPreview(effectiveImages);
        void primeCommandSessionId(
          "image_command_intent",
          resolveCommandSessionEnsureOptions(),
        ).catch(() => undefined);
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "image",
            sendOptions?.requestMetadata,
            skillRequest.requestContext,
          ),
        };
        if (mentionCommandKey) {
          markCompletedMentionCommand(
            mentionCommandKey,
            resolveMentionCommandReplayText(
              parsedImageWorkbenchCommand,
              mentionCommandKey,
            ),
          );
        }
        dispatchText = imageDispatchText;
        hasBoundSkillLaunch = true;
      }

      const parsedPosterWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand
          ? parsePosterWorkbenchCommand(sourceText)
          : null;
      if (parsedPosterWorkbenchCommand) {
        const imageCommandProjectContext =
          await resolveImageCommandProjectContext({
            projectId,
            projectRootPath,
          });
        if (!imageCommandProjectContext.projectRootPath) {
          clearSubmissionPreview();
          toast.error("默认项目目录未就绪，暂时无法创建图片任务");
          return { kind: "done", result: false };
        }
        if (prepareImageWorkbenchSkillSend) {
          const prepared = await prepareImageWorkbenchSkillSend();
          if (!prepared) {
            clearSubmissionPreview();
            return { kind: "done", result: false };
          }
        }
        const skillRequest = resolveImageWorkbenchCommandRequest({
          rawText: sourceText,
          parsedCommand: {
            rawText: parsedPosterWorkbenchCommand.rawText,
            commandKey: "poster_generate",
            trigger: "@配图",
            body: parsedPosterWorkbenchCommand.body,
            mode: "generate",
            prompt: parsedPosterWorkbenchCommand.prompt,
            count: 1,
            size: parsedPosterWorkbenchCommand.size,
            aspectRatio: parsedPosterWorkbenchCommand.aspectRatio,
            targetRef: undefined,
          },
          images: effectiveImages,
          sessionIdOverride: commandSessionId,
          entrySource: "at_poster_command",
          projectId: imageCommandProjectContext.projectId,
          projectRootPath: imageCommandProjectContext.projectRootPath,
        });
        if (!skillRequest) {
          clearSubmissionPreview();
          return { kind: "done", result: false };
        }
        effectiveImages =
          skillRequest.images.length > 0
            ? skillRequest.images
            : effectiveImages;
        pendingCommandSessionBinding = {
          kind: "request_context",
          requestContext: skillRequest.requestContext,
          requestContextKey: "image_task",
        };
        ensureSubmissionPreview(effectiveImages);
        void primeCommandSessionId(
          "poster_skill_launch",
          resolveCommandSessionEnsureOptions(),
        ).catch(() => undefined);
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "image",
            sendOptions?.requestMetadata,
            skillRequest.requestContext,
          ),
        };
        markCompletedMentionCommand(
          "poster_generate",
          resolveMentionCommandReplayText(
            parsedPosterWorkbenchCommand,
            "poster_generate",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedCoverWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedPosterWorkbenchCommand
          ? parseCoverWorkbenchCommand(sourceText)
          : null;
      if (parsedCoverWorkbenchCommand) {
        const requestContext = buildCoverSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: parsedCoverWorkbenchCommand,
          projectId,
          contentId,
        });
        if (!requestContext) {
          clearSubmissionPreview();
          return { kind: "done", result: false };
        }
        pendingCommandSessionBinding = {
          kind: "request_context",
          requestContext,
          requestContextKey: "cover_task",
        };
        ensureSubmissionPreview();
        void primeCommandSessionId(
          "cover_skill_launch",
          resolveCommandSessionEnsureOptions(),
        ).catch(() => undefined);
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "cover",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "cover_generate",
          resolveMentionCommandReplayText(
            parsedCoverWorkbenchCommand,
            "cover_generate",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedVideoWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand
          ? parseVideoWorkbenchCommand(sourceText)
          : null;
      if (parsedVideoWorkbenchCommand) {
        const requestContext = buildVideoSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: parsedVideoWorkbenchCommand,
          projectId,
          contentId,
        });
        if (!requestContext) {
          clearSubmissionPreview();
          return { kind: "done", result: false };
        }
        pendingCommandSessionBinding = {
          kind: "request_context",
          requestContext,
          requestContextKey: "video_task",
        };
        ensureSubmissionPreview();
        void primeCommandSessionId(
          "video_skill_launch",
          resolveCommandSessionEnsureOptions(),
        ).catch(() => undefined);
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "video",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "video_generate",
          resolveMentionCommandReplayText(
            parsedVideoWorkbenchCommand,
            "video_generate",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedBroadcastWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand
          ? parseBroadcastWorkbenchCommand(sourceText)
          : null;
      if (parsedBroadcastWorkbenchCommand) {
        const requestContext = buildBroadcastSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: parsedBroadcastWorkbenchCommand,
          projectId,
          contentId,
        });
        pendingCommandSessionBinding = {
          kind: "request_context",
          requestContext,
          requestContextKey: "broadcast_task",
        };
        ensureSubmissionPreview();
        void primeCommandSessionId(
          "broadcast_skill_launch",
          resolveCommandSessionEnsureOptions(),
        ).catch(() => undefined);
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "broadcast",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "broadcast_generate",
          resolveMentionCommandReplayText(
            parsedBroadcastWorkbenchCommand,
            "broadcast_generate",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedResourceSearchWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand
          ? parseResourceSearchWorkbenchCommand(sourceText)
          : null;
      if (parsedResourceSearchWorkbenchCommand) {
        const resourceRewritePreference =
          resolveServiceModelExecutionPreference(
            resourcePromptRewritePreference,
          );
        const requestContext = buildResourceSearchSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: parsedResourceSearchWorkbenchCommand,
          projectId,
          contentId,
          promptOverride: mergeServiceModelPrompt(
            resourceRewritePreference.customPrompt,
            parsedResourceSearchWorkbenchCommand.prompt,
          ),
        });
        pendingCommandSessionBinding = {
          kind: "request_context",
          requestContext,
          requestContextKey: "resource_search_task",
        };
        ensureSubmissionPreview();
        void primeCommandSessionId(
          "resource_search_skill_launch",
          resolveCommandSessionEnsureOptions(),
        ).catch(() => undefined);
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "resourceSearch",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "modal_resource_search",
          resolveMentionCommandReplayText(
            parsedResourceSearchWorkbenchCommand,
            "modal_resource_search",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedTranscriptionWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand
          ? parseTranscriptionWorkbenchCommand(sourceText)
          : null;
      if (parsedTranscriptionWorkbenchCommand) {
        const requestContext = buildTranscriptionSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: parsedTranscriptionWorkbenchCommand,
          projectId,
          contentId,
        });
        pendingCommandSessionBinding = {
          kind: "request_context",
          requestContext,
          requestContextKey: "transcription_task",
        };
        ensureSubmissionPreview();
        void primeCommandSessionId(
          "transcription_skill_launch",
          resolveCommandSessionEnsureOptions(),
        ).catch(() => undefined);
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "transcription",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "transcription_generate",
          resolveMentionCommandReplayText(
            parsedTranscriptionWorkbenchCommand,
            "transcription_generate",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedSearchWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand
          ? parseSearchWorkbenchCommand(sourceText)
          : null;
      if (parsedSearchWorkbenchCommand) {
        const prefilledSearchCommand = maybeApplyMentionCommandRecentDefaults({
          rawText: sourceText,
          commandKey: "research",
          parsedCommand: parsedSearchWorkbenchCommand,
          reparse: parseSearchWorkbenchCommand,
        });
        sourceText = prefilledSearchCommand.rawText;
        dispatchText = sourceText;
        const requestContext = buildResearchSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: prefilledSearchCommand.parsedCommand,
          projectId,
          contentId,
        });
        if (!requestContext) {
          return { kind: "done", result: false };
        }
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "research",
            sendOptions?.requestMetadata,
            requestContext,
          ),
          explicitToolPreferences: true,
        };
        effectiveWebSearch = true;
        effectiveSearchMode = sendOptions.searchMode ?? "required";
        markCompletedMentionCommand(
          "research",
          resolveMentionCommandReplayText(
            prefilledSearchCommand.parsedCommand,
            "research",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedReportWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand
          ? parseReportWorkbenchCommand(sourceText)
          : null;
      if (parsedReportWorkbenchCommand) {
        const prefilledReportCommand = maybeApplyMentionCommandRecentDefaults({
          rawText: sourceText,
          commandKey: "research_report",
          parsedCommand: parsedReportWorkbenchCommand,
          reparse: parseReportWorkbenchCommand,
        });
        sourceText = prefilledReportCommand.rawText;
        dispatchText = sourceText;
        const requestContext = buildReportSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: prefilledReportCommand.parsedCommand,
          projectId,
          contentId,
        });
        if (!requestContext) {
          return { kind: "done", result: false };
        }
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "report",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "research_report",
          resolveMentionCommandReplayText(
            prefilledReportCommand.parsedCommand,
            "research_report",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedCompetitorWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand
          ? parseCompetitorWorkbenchCommand(sourceText)
          : null;
      if (parsedCompetitorWorkbenchCommand) {
        const prefilledCompetitorCommand =
          maybeApplyMentionCommandRecentDefaults({
            rawText: sourceText,
            commandKey: "competitor_research",
            parsedCommand: parsedCompetitorWorkbenchCommand,
            reparse: parseCompetitorWorkbenchCommand,
          });
        sourceText = prefilledCompetitorCommand.rawText;
        dispatchText = sourceText;
        const requestContext = buildCompetitorSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: prefilledCompetitorCommand.parsedCommand,
          projectId,
          contentId,
        });
        if (!requestContext) {
          return { kind: "done", result: false };
        }
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "report",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "competitor_research",
          resolveMentionCommandReplayText(
            prefilledCompetitorCommand.parsedCommand,
            "competitor_research",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedDeepSearchWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedCompetitorWorkbenchCommand
          ? parseDeepSearchWorkbenchCommand(sourceText)
          : null;
      if (parsedDeepSearchWorkbenchCommand) {
        const prefilledDeepSearchCommand =
          maybeApplyMentionCommandRecentDefaults({
            rawText: sourceText,
            commandKey: "deep_search",
            parsedCommand: parsedDeepSearchWorkbenchCommand,
            reparse: parseDeepSearchWorkbenchCommand,
          });
        sourceText = prefilledDeepSearchCommand.rawText;
        dispatchText = sourceText;
        const requestContext = buildDeepSearchSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: prefilledDeepSearchCommand.parsedCommand,
          projectId,
          contentId,
        });
        if (!requestContext) {
          return { kind: "done", result: false };
        }
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "deepSearch",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "deep_search",
          resolveMentionCommandReplayText(
            prefilledDeepSearchCommand.parsedCommand,
            "deep_search",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedSiteSearchWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand
          ? parseSiteSearchWorkbenchCommand(sourceText)
          : null;
      if (parsedSiteSearchWorkbenchCommand) {
        const prefilledSiteSearchCommand =
          maybeApplyMentionCommandRecentDefaults({
            rawText: sourceText,
            commandKey: "site_search",
            parsedCommand: parsedSiteSearchWorkbenchCommand,
            reparse: parseSiteSearchWorkbenchCommand,
          });
        sourceText = prefilledSiteSearchCommand.rawText;
        dispatchText = sourceText;
        const requestContext = buildSiteSearchSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: prefilledSiteSearchCommand.parsedCommand,
          projectId,
          contentId,
        });
        if (!requestContext) {
          return { kind: "done", result: false };
        }
        effectiveWebSearch = false;
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "siteSearch",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "site_search",
          resolveMentionCommandReplayText(
            prefilledSiteSearchCommand.parsedCommand,
            "site_search",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedPdfWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand
          ? parsePdfWorkbenchCommand(sourceText)
          : null;
      if (parsedPdfWorkbenchCommand) {
        const prefilledPdfCommand = maybeApplyMentionCommandRecentDefaults({
          rawText: sourceText,
          commandKey: "read_pdf",
          parsedCommand: parsedPdfWorkbenchCommand,
          reparse: parsePdfWorkbenchCommand,
        });
        sourceText = prefilledPdfCommand.rawText;
        dispatchText = sourceText;
        const requestContext = buildPdfReadSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: prefilledPdfCommand.parsedCommand,
          projectId,
          contentId,
        });
        if (!requestContext) {
          return { kind: "done", result: false };
        }
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "pdfRead",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "read_pdf",
          resolveMentionCommandReplayText(
            prefilledPdfCommand.parsedCommand,
            "read_pdf",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedFileReadWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand
          ? parseFileReadWorkbenchCommand(sourceText)
          : null;
      if (parsedFileReadWorkbenchCommand) {
        const prefilledFileReadCommand = maybeApplyMentionCommandRecentDefaults(
          {
            rawText: sourceText,
            commandKey: "file_read_runtime",
            parsedCommand: parsedFileReadWorkbenchCommand,
            reparse: parseFileReadWorkbenchCommand,
          },
        );
        sourceText = prefilledFileReadCommand.rawText;
        dispatchText = sourceText;
        const requestContext = buildFileReadSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: prefilledFileReadCommand.parsedCommand,
          projectId,
          contentId,
        });
        if (!requestContext) {
          return { kind: "done", result: false };
        }
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "summary",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "file_read_runtime",
          resolveMentionCommandReplayText(
            prefilledFileReadCommand.parsedCommand,
            "file_read_runtime",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedSummaryWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedFileReadWorkbenchCommand
          ? parseSummaryWorkbenchCommand(sourceText)
          : null;
      if (parsedSummaryWorkbenchCommand) {
        const mergedSummaryCommand = mergeSummaryCommandRecentDefaults({
          parsedCommand: parsedSummaryWorkbenchCommand,
          slotValues: mentionUsageMap.get(
            getMentionEntryUsageRecordKey("builtin_command", "summary"),
          )?.slotValues,
        });
        const requestContext = buildSummarySkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: mergedSummaryCommand,
          projectId,
          contentId,
        });
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "summary",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "summary",
          resolveMentionCommandReplayText(mergedSummaryCommand, "summary"),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedTranslationWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedSummaryWorkbenchCommand
          ? parseTranslationWorkbenchCommand(sourceText)
          : null;
      if (parsedTranslationWorkbenchCommand) {
        const mergedTranslationCommand = mergeTranslationCommandRecentDefaults({
          parsedCommand: parsedTranslationWorkbenchCommand,
          slotValues: mentionUsageMap.get(
            getMentionEntryUsageRecordKey("builtin_command", "translation"),
          )?.slotValues,
        });
        const requestContext = buildTranslationSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: mergedTranslationCommand,
          projectId,
          contentId,
        });
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "translation",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "translation",
          resolveMentionCommandReplayText(
            mergedTranslationCommand,
            "translation",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedComplianceWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedSummaryWorkbenchCommand &&
        !parsedTranslationWorkbenchCommand
          ? parseComplianceWorkbenchCommand(sourceText)
          : null;
      if (parsedComplianceWorkbenchCommand) {
        const mergedComplianceCommand = mergeComplianceCommandRecentDefaults({
          parsedCommand: parsedComplianceWorkbenchCommand,
          slotValues: mentionUsageMap.get(
            getMentionEntryUsageRecordKey(
              "builtin_command",
              "publish_compliance",
            ),
          )?.slotValues,
        });
        const requestContext = buildAnalysisSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: mergedComplianceCommand,
          projectId,
          contentId,
          entrySource: "at_publish_compliance_command",
        });
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "analysis",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "publish_compliance",
          resolveMentionCommandReplayText(
            mergedComplianceCommand,
            "publish_compliance",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedLogoDecompositionWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedSummaryWorkbenchCommand &&
        !parsedTranslationWorkbenchCommand &&
        !parsedComplianceWorkbenchCommand
          ? parseLogoDecompositionWorkbenchCommand(sourceText)
          : null;
      if (parsedLogoDecompositionWorkbenchCommand) {
        const mergedLogoDecompositionCommand =
          mergeAnalysisCommandRecentDefaults({
            parsedCommand: parsedLogoDecompositionWorkbenchCommand,
            slotValues: mentionUsageMap.get(
              getMentionEntryUsageRecordKey(
                "builtin_command",
                "logo_decomposition",
              ),
            )?.slotValues,
          });
        const requestContext = buildAnalysisSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: mergedLogoDecompositionCommand,
          projectId,
          contentId,
          entrySource: "at_logo_decomposition_command",
        });
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "analysis",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "logo_decomposition",
          resolveMentionCommandReplayText(
            mergedLogoDecompositionCommand,
            "logo_decomposition",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedAnalysisWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedSummaryWorkbenchCommand &&
        !parsedTranslationWorkbenchCommand &&
        !parsedComplianceWorkbenchCommand &&
        !parsedLogoDecompositionWorkbenchCommand
          ? parseAnalysisWorkbenchCommand(sourceText)
          : null;
      if (parsedAnalysisWorkbenchCommand) {
        const mergedAnalysisCommand = mergeAnalysisCommandRecentDefaults({
          parsedCommand: parsedAnalysisWorkbenchCommand,
          slotValues: mentionUsageMap.get(
            getMentionEntryUsageRecordKey("builtin_command", "analysis"),
          )?.slotValues,
        });
        const requestContext = buildAnalysisSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: mergedAnalysisCommand,
          projectId,
          contentId,
        });
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "analysis",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "analysis",
          resolveMentionCommandReplayText(mergedAnalysisCommand, "analysis"),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedUrlParseWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedSummaryWorkbenchCommand &&
        !parsedTranslationWorkbenchCommand &&
        !parsedComplianceWorkbenchCommand
          ? parseUrlParseWorkbenchCommand(sourceText)
          : null;
      if (parsedUrlParseWorkbenchCommand) {
        const requestContext = buildUrlParseSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: parsedUrlParseWorkbenchCommand,
          projectId,
          contentId,
        });
        pendingCommandSessionBinding = {
          kind: "request_context",
          requestContext,
          requestContextKey: "url_parse_task",
        };
        ensureSubmissionPreview();
        void primeCommandSessionId().catch(() => undefined);
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "urlParse",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          isUrlParseScrapeTrigger(parsedUrlParseWorkbenchCommand.trigger)
            ? "web_scrape"
            : isUrlParseReadTrigger(parsedUrlParseWorkbenchCommand.trigger)
              ? "webpage_read"
              : "url_parse",
          resolveMentionCommandReplayText(
            parsedUrlParseWorkbenchCommand,
            isUrlParseScrapeTrigger(parsedUrlParseWorkbenchCommand.trigger)
              ? "web_scrape"
              : isUrlParseReadTrigger(parsedUrlParseWorkbenchCommand.trigger)
                ? "webpage_read"
                : "url_parse",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedTypesettingWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedSummaryWorkbenchCommand &&
        !parsedTranslationWorkbenchCommand &&
        !parsedUrlParseWorkbenchCommand
          ? parseTypesettingWorkbenchCommand(sourceText)
          : null;
      if (parsedTypesettingWorkbenchCommand) {
        const mergedTypesettingCommand = mergeTypesettingCommandRecentDefaults({
          parsedCommand: parsedTypesettingWorkbenchCommand,
          slotValues: mentionUsageMap.get(
            getMentionEntryUsageRecordKey("builtin_command", "typesetting"),
          )?.slotValues,
        });
        const requestContext = buildTypesettingSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: mergedTypesettingCommand,
          projectId,
          contentId,
        });
        pendingCommandSessionBinding = {
          kind: "request_context",
          requestContext,
          requestContextKey: "typesetting_task",
        };
        ensureSubmissionPreview();
        void primeCommandSessionId().catch(() => undefined);
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "typesetting",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "typesetting",
          resolveMentionCommandReplayText(
            mergedTypesettingCommand,
            "typesetting",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedPresentationWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedSummaryWorkbenchCommand &&
        !parsedTranslationWorkbenchCommand &&
        !parsedUrlParseWorkbenchCommand &&
        !parsedTypesettingWorkbenchCommand
          ? parsePresentationWorkbenchCommand(sourceText)
          : null;
      if (parsedPresentationWorkbenchCommand) {
        const mergedPresentationCommand =
          mergePresentationCommandRecentDefaults({
            parsedCommand: parsedPresentationWorkbenchCommand,
            slotValues: mentionUsageMap.get(
              getMentionEntryUsageRecordKey(
                "builtin_command",
                "presentation_generate",
              ),
            )?.slotValues,
          });
        const requestContext = buildPresentationSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: mergedPresentationCommand,
          projectId,
          contentId,
        });
        ensureSubmissionPreview();
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "presentation",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "presentation_generate",
          resolveMentionCommandReplayText(
            mergedPresentationCommand,
            "presentation_generate",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedFormWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedSummaryWorkbenchCommand &&
        !parsedTranslationWorkbenchCommand &&
        !parsedUrlParseWorkbenchCommand &&
        !parsedTypesettingWorkbenchCommand &&
        !parsedPresentationWorkbenchCommand
          ? parseFormWorkbenchCommand(sourceText)
          : null;
      if (parsedFormWorkbenchCommand) {
        const mergedFormCommand = mergeFormCommandRecentDefaults({
          parsedCommand: parsedFormWorkbenchCommand,
          slotValues: mentionUsageMap.get(
            getMentionEntryUsageRecordKey("builtin_command", "form_generate"),
          )?.slotValues,
        });
        const requestContext = buildFormSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: mergedFormCommand,
          projectId,
          contentId,
        });
        ensureSubmissionPreview();
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "form",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "form_generate",
          resolveMentionCommandReplayText(mergedFormCommand, "form_generate"),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedWebpageWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedSummaryWorkbenchCommand &&
        !parsedTranslationWorkbenchCommand &&
        !parsedUrlParseWorkbenchCommand &&
        !parsedTypesettingWorkbenchCommand &&
        !parsedPresentationWorkbenchCommand &&
        !parsedFormWorkbenchCommand
          ? parseWebpageWorkbenchCommand(sourceText)
          : null;
      if (parsedWebpageWorkbenchCommand) {
        const mergedWebpageCommand = mergeWebpageCommandRecentDefaults({
          parsedCommand: parsedWebpageWorkbenchCommand,
          slotValues: mentionUsageMap.get(
            getMentionEntryUsageRecordKey(
              "builtin_command",
              "webpage_generate",
            ),
          )?.slotValues,
        });
        const requestContext = buildWebpageSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: mergedWebpageCommand,
          projectId,
          contentId,
        });
        ensureSubmissionPreview();
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildSkillLaunchRequestMetadata(
            "webpage",
            sendOptions?.requestMetadata,
            requestContext,
          ),
        };
        markCompletedMentionCommand(
          "webpage_generate",
          resolveMentionCommandReplayText(
            mergedWebpageCommand,
            "webpage_generate",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedAgentTurnMentionShortcut =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedSummaryWorkbenchCommand &&
        !parsedTranslationWorkbenchCommand &&
        !parsedUrlParseWorkbenchCommand &&
        !parsedTypesettingWorkbenchCommand &&
        !parsedPresentationWorkbenchCommand &&
        !parsedFormWorkbenchCommand &&
        !parsedWebpageWorkbenchCommand
          ? parseMentionCommand(sourceText, mentionCommandPrefixKeyMap)
          : null;
      const agentTurnMentionRoute = parsedAgentTurnMentionShortcut
        ? mentionAgentTurnRouteMap.get(
            parsedAgentTurnMentionShortcut.commandKey,
          )
        : undefined;
      if (parsedAgentTurnMentionShortcut && agentTurnMentionRoute) {
        effectiveSendExecutionStrategy = normalizeExecutionStrategy(
          agentTurnMentionRoute.executionStrategy ??
            effectiveSendExecutionStrategy,
        );
        ensureSubmissionPreview();
        markCompletedMentionCommand(
          parsedAgentTurnMentionShortcut.commandKey,
          resolveMentionCommandReplayText(
            parsedAgentTurnMentionShortcut,
            parsedAgentTurnMentionShortcut.commandKey,
          ),
        );
      }

      const parsedWritingWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedSummaryWorkbenchCommand &&
        !parsedTranslationWorkbenchCommand &&
        !parsedUrlParseWorkbenchCommand &&
        !parsedTypesettingWorkbenchCommand &&
        !parsedPresentationWorkbenchCommand &&
        !parsedFormWorkbenchCommand &&
        !parsedWebpageWorkbenchCommand &&
        !agentTurnMentionRoute
          ? parseWritingWorkbenchCommand(sourceText)
          : null;
      if (parsedWritingWorkbenchCommand) {
        const mergedWritingCommand = mergePublishLikeCommandRecentDefaults({
          parsedCommand: parsedWritingWorkbenchCommand,
          slotValues: mentionUsageMap.get(
            getMentionEntryUsageRecordKey("builtin_command", "writing_runtime"),
          )?.slotValues,
        });
        const existingHarnessMetadata =
          asRecord(sendOptions?.requestMetadata?.harness) || {};
        const dispatchBody = buildWritingDispatchBody({
          prompt: mergedWritingCommand.prompt || mergedWritingCommand.body,
          platformLabel: mergedWritingCommand.platformLabel,
          draftKind: mergedWritingCommand.draftKind,
        });
        dispatchText = `/${CONTENT_POST_SKILL_KEY}${
          dispatchBody ? ` ${dispatchBody}` : ""
        }`;
        ensureSubmissionPreview();
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: {
            ...(sendOptions?.requestMetadata || {}),
            harness: {
              ...existingHarnessMetadata,
              publish_command: {
                prompt:
                  mergedWritingCommand.prompt || mergedWritingCommand.body,
                content: mergedWritingCommand.body,
                platform_type: mergedWritingCommand.platformType || undefined,
                platform_label: mergedWritingCommand.platformLabel || undefined,
                entry_source: "at_writing_command",
              },
            },
          },
        };
        markCompletedMentionCommand(
          "writing_runtime",
          resolveMentionCommandReplayText(
            mergedWritingCommand,
            "writing_runtime",
          ),
        );
      }

      const parsedChannelPreviewWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedSummaryWorkbenchCommand &&
        !parsedTranslationWorkbenchCommand &&
        !parsedUrlParseWorkbenchCommand &&
        !parsedTypesettingWorkbenchCommand &&
        !parsedPresentationWorkbenchCommand &&
        !parsedFormWorkbenchCommand &&
        !parsedWebpageWorkbenchCommand &&
        !agentTurnMentionRoute &&
        !parsedWritingWorkbenchCommand
          ? parseChannelPreviewWorkbenchCommand(sourceText)
          : null;
      if (parsedChannelPreviewWorkbenchCommand) {
        const mergedChannelPreviewCommand =
          mergePublishLikeCommandRecentDefaults({
            parsedCommand: parsedChannelPreviewWorkbenchCommand,
            slotValues: mentionUsageMap.get(
              getMentionEntryUsageRecordKey(
                "builtin_command",
                "channel_preview_runtime",
              ),
            )?.slotValues,
          });
        const existingHarnessMetadata =
          asRecord(sendOptions?.requestMetadata?.harness) || {};
        const dispatchBody = buildChannelPreviewDispatchBody({
          prompt:
            mergedChannelPreviewCommand.prompt ||
            mergedChannelPreviewCommand.body,
          platformLabel: mergedChannelPreviewCommand.platformLabel,
        });
        dispatchText = `/${CONTENT_POST_SKILL_KEY}${
          dispatchBody ? ` ${dispatchBody}` : ""
        }`;
        ensureSubmissionPreview();
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: {
            ...(sendOptions?.requestMetadata || {}),
            harness: {
              ...existingHarnessMetadata,
              publish_command: {
                prompt:
                  mergedChannelPreviewCommand.prompt ||
                  mergedChannelPreviewCommand.body,
                content: mergedChannelPreviewCommand.body,
                platform_type:
                  mergedChannelPreviewCommand.platformType || undefined,
                platform_label:
                  mergedChannelPreviewCommand.platformLabel || undefined,
                intent: "preview",
                entry_source: "at_channel_preview_command",
              },
            },
          },
        };
        markCompletedMentionCommand(
          "channel_preview_runtime",
          resolveMentionCommandReplayText(
            mergedChannelPreviewCommand,
            "channel_preview_runtime",
          ),
        );
      }

      const parsedUploadWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedSummaryWorkbenchCommand &&
        !parsedTranslationWorkbenchCommand &&
        !parsedUrlParseWorkbenchCommand &&
        !parsedTypesettingWorkbenchCommand &&
        !parsedPresentationWorkbenchCommand &&
        !parsedFormWorkbenchCommand &&
        !parsedWebpageWorkbenchCommand &&
        !agentTurnMentionRoute &&
        !parsedWritingWorkbenchCommand &&
        !parsedChannelPreviewWorkbenchCommand
          ? parseUploadWorkbenchCommand(sourceText)
          : null;
      if (parsedUploadWorkbenchCommand) {
        const mergedUploadCommand = mergePublishLikeCommandRecentDefaults({
          parsedCommand: parsedUploadWorkbenchCommand,
          slotValues: mentionUsageMap.get(
            getMentionEntryUsageRecordKey("builtin_command", "upload_runtime"),
          )?.slotValues,
        });
        const existingHarnessMetadata =
          asRecord(sendOptions?.requestMetadata?.harness) || {};
        const dispatchBody = buildUploadDispatchBody({
          prompt: mergedUploadCommand.prompt || mergedUploadCommand.body,
          platformLabel: mergedUploadCommand.platformLabel,
        });
        const uploadBrowserRequirementMatch = detectBrowserTaskRequirement(
          dispatchBody ||
            mergedUploadCommand.body ||
            mergedUploadCommand.prompt,
        );
        dispatchText = `/${CONTENT_POST_SKILL_KEY}${
          dispatchBody ? ` ${dispatchBody}` : ""
        }`;
        ensureSubmissionPreview();
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: {
            ...(sendOptions?.requestMetadata || {}),
            harness: {
              ...existingHarnessMetadata,
              ...(uploadBrowserRequirementMatch
                ? {
                    browser_requirement:
                      uploadBrowserRequirementMatch.requirement,
                    browser_requirement_reason:
                      uploadBrowserRequirementMatch.reason,
                    browser_launch_url: uploadBrowserRequirementMatch.launchUrl,
                  }
                : {}),
              publish_command: {
                prompt: mergedUploadCommand.prompt || mergedUploadCommand.body,
                content: mergedUploadCommand.body,
                platform_type: mergedUploadCommand.platformType || undefined,
                platform_label: mergedUploadCommand.platformLabel || undefined,
                intent: "upload",
                entry_source: "at_upload_command",
              },
            },
          },
        };
        markCompletedMentionCommand(
          "upload_runtime",
          resolveMentionCommandReplayText(
            mergedUploadCommand,
            "upload_runtime",
          ),
        );
      }

      const parsedPublishWorkbenchCommand =
        !sendOptions?.purpose &&
        sourceText.trim() &&
        !parsedImageWorkbenchCommand &&
        !parsedCoverWorkbenchCommand &&
        !parsedVideoWorkbenchCommand &&
        !parsedBroadcastWorkbenchCommand &&
        !parsedResourceSearchWorkbenchCommand &&
        !parsedTranscriptionWorkbenchCommand &&
        !parsedSearchWorkbenchCommand &&
        !parsedReportWorkbenchCommand &&
        !parsedDeepSearchWorkbenchCommand &&
        !parsedSiteSearchWorkbenchCommand &&
        !parsedPdfWorkbenchCommand &&
        !parsedSummaryWorkbenchCommand &&
        !parsedTranslationWorkbenchCommand &&
        !parsedUrlParseWorkbenchCommand &&
        !parsedTypesettingWorkbenchCommand &&
        !parsedPresentationWorkbenchCommand &&
        !parsedFormWorkbenchCommand &&
        !parsedWebpageWorkbenchCommand &&
        !agentTurnMentionRoute &&
        !parsedWritingWorkbenchCommand &&
        !parsedChannelPreviewWorkbenchCommand &&
        !parsedUploadWorkbenchCommand
          ? parsePublishWorkbenchCommand(sourceText)
          : null;
      if (parsedPublishWorkbenchCommand) {
        const mergedPublishCommand = mergePublishLikeCommandRecentDefaults({
          parsedCommand: parsedPublishWorkbenchCommand,
          slotValues: mentionUsageMap.get(
            getMentionEntryUsageRecordKey("builtin_command", "publish_runtime"),
          )?.slotValues,
        });
        const existingHarnessMetadata =
          asRecord(sendOptions?.requestMetadata?.harness) || {};
        const nextBody = buildPublishDispatchBody({
          prompt: mergedPublishCommand.prompt || mergedPublishCommand.body,
          platformLabel: mergedPublishCommand.platformLabel,
        });
        const publishBrowserRequirementMatch = detectBrowserTaskRequirement(
          nextBody || mergedPublishCommand.body || mergedPublishCommand.prompt,
        );
        dispatchText = `/${CONTENT_POST_SKILL_KEY}${nextBody ? ` ${nextBody}` : ""}`;
        ensureSubmissionPreview();
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: {
            ...(sendOptions?.requestMetadata || {}),
            harness: {
              ...existingHarnessMetadata,
              ...(publishBrowserRequirementMatch
                ? {
                    browser_requirement:
                      publishBrowserRequirementMatch.requirement,
                    browser_requirement_reason:
                      publishBrowserRequirementMatch.reason,
                    browser_launch_url:
                      publishBrowserRequirementMatch.launchUrl,
                  }
                : {}),
              publish_command: {
                prompt:
                  mergedPublishCommand.prompt || mergedPublishCommand.body,
                content: mergedPublishCommand.body,
                platform_type: mergedPublishCommand.platformType || undefined,
                platform_label: mergedPublishCommand.platformLabel || undefined,
                entry_source: "at_publish_command",
              },
            },
          },
        };
        markCompletedMentionCommand(
          "publish_runtime",
          resolveMentionCommandReplayText(
            mergedPublishCommand,
            "publish_runtime",
          ),
        );
      }

      const parsedVoiceWorkbenchCommand =
        !sendOptions?.purpose && sourceText.trim()
          ? parseVoiceWorkbenchCommand(sourceText)
          : null;
      if (parsedVoiceWorkbenchCommand) {
        let mediaDefaults: MediaGenerationDefaults = {};
        try {
          mediaDefaults = await readGlobalMediaGenerationDefaults();
        } catch (error) {
          console.error("加载全局媒体默认设置失败:", error);
        }
        const voiceSkillLaunch = await resolveVoiceSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: parsedVoiceWorkbenchCommand,
          serviceSkills,
          projectId,
          contentId,
          voicePreference: mediaDefaults.voice,
        });
        if (!voiceSkillLaunch) {
          clearSubmissionPreview();
          return { kind: "done", result: false };
        }

        ensureSubmissionPreview();
        dispatchText = voiceSkillLaunch.dispatchText;
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildServiceSceneLaunchRequestMetadata(
            sendOptions?.requestMetadata,
            voiceSkillLaunch.requestContext,
          ),
        };
        markCompletedMentionCommand(
          "voice_runtime",
          resolveMentionCommandReplayText(
            parsedVoiceWorkbenchCommand,
            "voice_runtime",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedGrowthWorkbenchCommand =
        !sendOptions?.purpose && sourceText.trim()
          ? parseGrowthWorkbenchCommand(sourceText)
          : null;
      if (parsedGrowthWorkbenchCommand) {
        const growthSkillLaunch = await resolveGrowthSkillLaunchRequestContext({
          rawText: sourceText,
          parsedCommand: parsedGrowthWorkbenchCommand,
          serviceSkills,
          projectId,
          contentId,
        });
        if (!growthSkillLaunch) {
          clearSubmissionPreview();
          return { kind: "done", result: false };
        }

        ensureSubmissionPreview();
        dispatchText = growthSkillLaunch.dispatchText;
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildServiceSceneLaunchRequestMetadata(
            sendOptions?.requestMetadata,
            growthSkillLaunch.requestContext,
          ),
        };
        markCompletedMentionCommand(
          "growth_runtime",
          resolveMentionCommandReplayText(
            parsedGrowthWorkbenchCommand,
            "growth_runtime",
          ),
        );
        hasBoundSkillLaunch = true;
      }

      const parsedBrowserWorkbenchCommand =
        !sendOptions?.purpose && sourceText.trim()
          ? parseBrowserWorkbenchCommand(sourceText)
          : null;
      if (parsedBrowserWorkbenchCommand) {
        effectiveWebSearch = false;
        effectiveSearchMode = "disabled";
        browserRequirementForSend = {
          requirement: parsedBrowserWorkbenchCommand.browserRequirement,
          reason: parsedBrowserWorkbenchCommand.browserRequirementReason,
          launchUrl: parsedBrowserWorkbenchCommand.launchUrl,
        };
        ensureSubmissionPreview();
        sendOptions = {
          ...(sendOptions || {}),
          requestMetadata: buildBrowserControlLaunchRequestMetadata(
            sendOptions?.requestMetadata,
            parsedBrowserWorkbenchCommand,
            browserAssistSessionState,
          ),
        };
        markCompletedMentionCommand(
          "browser_runtime",
          resolveMentionCommandReplayText(
            parsedBrowserWorkbenchCommand,
            "browser_runtime",
          ),
        );
      }

      if (
        !sendOptions?.purpose &&
        !sendOptions?.skipSceneCommandRouting &&
        sourceText.trim().startsWith("/")
      ) {
        ensureSubmissionPreview();
        let sceneLaunchRequest = null;
        try {
          sceneLaunchRequest = await resolveRuntimeSceneLaunchRequest({
            rawText: sourceText,
            serviceSkills,
            projectId,
            contentId,
          });
        } catch (error) {
          if (error instanceof RuntimeSceneLaunchValidationError) {
            if (error.gateRequest && openRuntimeSceneGate) {
              await openRuntimeSceneGate(error.gateRequest);
              clearSubmissionPreview();
              return { kind: "done", result: false };
            }
            toast.error(error.message);
            clearSubmissionPreview();
            return { kind: "done", result: false };
          }
          throw error;
        }
        if (sceneLaunchRequest) {
          const sceneRequestDefaults =
            sceneLaunchRequest.sceneEntry.requestDefaults ?? {};
          const sceneExecutionStrategy = normalizeExecutionStrategyToReact(
            sceneRequestDefaults.executionStrategy ??
              sceneRequestDefaults.execution_strategy,
          );
          if (sceneExecutionStrategy) {
            effectiveSendExecutionStrategy = normalizeExecutionStrategy(
              sceneExecutionStrategy,
            );
          }
          if (sceneLaunchRequest.dispatchText) {
            dispatchText = sceneLaunchRequest.dispatchText;
          }
          sendOptions = {
            ...(sendOptions || {}),
            requestMetadata: buildServiceSceneLaunchRequestMetadata(
              sendOptions?.requestMetadata,
              sceneLaunchRequest.requestContext,
            ),
          };
          hasBoundSkillLaunch = true;
          completedSlashUsage = {
            kind: "scene",
            entryId: sceneLaunchRequest.sceneEntry.sceneKey,
            replayText:
              completedSlashUsage?.replayText ??
              parseRuntimeSceneCommand(sourceText)?.userInput,
          };
        }
      }

      const hasMatchedWorkspaceMentionCommandWithoutAgentTurnRoute = Boolean(
        parsedImageWorkbenchCommand ||
        parsedPosterWorkbenchCommand ||
        parsedCoverWorkbenchCommand ||
        parsedVideoWorkbenchCommand ||
        parsedBroadcastWorkbenchCommand ||
        parsedResourceSearchWorkbenchCommand ||
        parsedTranscriptionWorkbenchCommand ||
        parsedSearchWorkbenchCommand ||
        parsedReportWorkbenchCommand ||
        parsedCompetitorWorkbenchCommand ||
        parsedDeepSearchWorkbenchCommand ||
        parsedSiteSearchWorkbenchCommand ||
        parsedPdfWorkbenchCommand ||
        parsedFileReadWorkbenchCommand ||
        parsedSummaryWorkbenchCommand ||
        parsedTranslationWorkbenchCommand ||
        parsedComplianceWorkbenchCommand ||
        parsedLogoDecompositionWorkbenchCommand ||
        parsedAnalysisWorkbenchCommand ||
        parsedUrlParseWorkbenchCommand ||
        parsedTypesettingWorkbenchCommand ||
        parsedPresentationWorkbenchCommand ||
        parsedFormWorkbenchCommand ||
        parsedWebpageWorkbenchCommand ||
        parsedWritingWorkbenchCommand ||
        parsedChannelPreviewWorkbenchCommand ||
        parsedUploadWorkbenchCommand ||
        parsedPublishWorkbenchCommand ||
        parsedVoiceWorkbenchCommand ||
        parsedGrowthWorkbenchCommand ||
        parsedBrowserWorkbenchCommand,
      );
      const trimmedSourceText = sourceText.trim();
      if (
        activeTheme === "general" &&
        !sendOptions?.purpose &&
        !hasBoundSkillLaunch &&
        !images?.length &&
        trimmedSourceText &&
        !trimmedSourceText.startsWith("/") &&
        !trimmedSourceText.startsWith("@")
      ) {
        const matchedSiteSkill = matchAutoLaunchSiteSkillFromText({
          inputText: trimmedSourceText,
          serviceSkills,
        });
        if (matchedSiteSkill) {
          clearSubmissionPreview();
          await handleAutoLaunchMatchedSiteSkill(matchedSiteSkill);
          return { kind: "done", result: true };
        }
      }

      const mergedRequestMetadataAfterLaunch = {
        ...(workspaceRequestMetadataBase || {}),
        ...(sendOptions?.requestMetadata || {}),
      };
      if (
        !projectId &&
        serviceSkillLaunchRequiresProject(mergedRequestMetadataAfterLaunch)
      ) {
        sendOptions?.observer?.onError?.("请先选择项目后再开始对话");
        toast.error("请先选择项目后再开始对话");
        clearSubmissionPreview();
        return { kind: "done", result: false };
      }

      const shouldPrimeSessionForInitialConversationSend =
        !sendOptions?.purpose &&
        !hasBoundSkillLaunch &&
        messagesCount === 0 &&
        sendOptions?.skipSessionRestore !== true &&
        sendOptions?.skipSessionStartHooks !== true &&
        Boolean(ensureSessionForCommandMetadata);

      const shouldSkipBrowserAssistPrime =
        shouldSkipBrowserAssistPrimeForPlainFirstTurn({
          activeTheme,
          browserRequirementMatch: browserRequirementForSend,
          hasBoundSkillLaunch,
          imagesCount: images?.length ?? 0,
          messagesCount,
          sendOptions,
          sourceText,
        });

      if (!hasBoundSkillLaunch && !shouldSkipBrowserAssistPrime) {
        primeBrowserAssistBeforeSend({
          activeTheme,
          sourceText,
          browserRequirementMatch: browserRequirementForSend,
          ensureBrowserAssistCanvas,
        });
      }

      let text: string;
      try {
        const resolvedSubmissionPreviewKey = ensureSubmissionPreview();
        if (messagesCount === 0) {
          const previewStartedAt = Date.now();
          void waitForNextPaint().then(() => {
            logAgentDebug("WorkspaceSend", "initialPreview.paintDone", {
              durationMs: Date.now() - previewStartedAt,
              messagesCount,
            });
          });
        }
        if (shouldPrimeSessionForInitialConversationSend) {
          void primeCommandSessionId(
            "initial_conversation_send",
            resolveCommandSessionEnsureOptions(),
          ).catch(() => undefined);
        } else if (
          !sendOptions?.purpose &&
          !hasBoundSkillLaunch &&
          messagesCount === 0 &&
          Boolean(ensureSessionForCommandMetadata)
        ) {
          logAgentDebug("WorkspaceSend", "primeSession.skipInitial", {
            reason: "initial_conversation_send",
            skipSessionRestore: sendOptions?.skipSessionRestore === true,
            skipSessionStartHooks: sendOptions?.skipSessionStartHooks === true,
          });
        }
        text = await buildWorkspaceSendText({
          sourceText: dispatchText,
          contextWorkspace: effectiveContextWorkspace,
          mentionedCharacters,
          sendOptions,
          preparedActiveContextPrompt,
        });
        if (pendingCommandSessionBinding) {
          const resolvedSessionId =
            pendingCommandSessionBindingMode === "blocking"
              ? await ensureCommandSessionId()
              : commandSessionId;
          if (
            resolvedSessionId !== undefined ||
            pendingCommandSessionBindingMode === "best_effort"
          ) {
            if (pendingCommandSessionBinding.kind === "request_context") {
              attachSessionIdToRequestContext(
                pendingCommandSessionBinding.requestContext,
                pendingCommandSessionBinding.requestContextKey,
                resolvedSessionId,
              );
            } else {
              attachSessionIdToScopedRequestContext(
                pendingCommandSessionBinding.scopedRequestContext,
                resolvedSessionId,
              );
            }
          }
        }
        submissionPreviewKey = resolvedSubmissionPreviewKey;
        const performanceTrace = extractAgentUiPerformanceTraceMetadata(
          sendOptions?.requestMetadata,
        );
        if (performanceTrace?.sessionId || performanceTrace?.requestId) {
          recordAgentUiPerformanceMetric("workspaceSend.plan.ready", {
            durationMs: Date.now() - planStartedAt,
            hasPendingSessionBinding: Boolean(pendingCommandSessionBinding),
            primedSessionId: commandSessionId ?? null,
            requestId: performanceTrace.requestId ?? null,
            sessionId: performanceTrace.sessionId ?? null,
            source: performanceTrace.source ?? "workspace-send",
            workspaceId: performanceTrace.workspaceId ?? null,
          });
        }
        logAgentDebug("WorkspaceSend", "plan.ready", {
          durationMs: Date.now() - planStartedAt,
          hasPendingSessionBinding: Boolean(pendingCommandSessionBinding),
          primedSessionId: commandSessionId ?? null,
          sourceTextLength: sourceText.trim().length,
        });
      } catch (error) {
        clearSubmissionPreview();
        throw error;
      }

      return {
        kind: "ready",
        plan: {
          sourceText,
          dispatchText,
          text,
          images: effectiveImages,
          hasContextWorkspace: shouldAttachContextWorkspace,
          sendBoundary,
          browserRequirementForSend,
          effectiveToolPreferences,
          effectiveWebSearch,
          effectiveSearchMode,
          submissionPreviewKey,
          sendExecutionStrategy: effectiveSendExecutionStrategy,
          autoContinuePayload,
          sendOptions,
          completedMentionCommandUsage,
          completedMentionUsage,
          completedSlashUsage,
        },
      };
    },
    [
      activeTheme,
      browserAssistSessionState,
      chatToolPreferences,
      contentId,
      contextWorkspace,
      ensureSessionForCommandMetadata,
      ensureBrowserAssistCanvas,
      executionStrategy,
      handleAutoLaunchMatchedSiteSkill,
      isThemeWorkbench,
      resolveImageWorkbenchCommandRequest,
      input,
      messagesCount,
      mentionedCharacters,
      openRuntimeSceneGate,
      prepareImageWorkbenchSkillSend,
      projectId,
      projectRootPath,
      resolveSendBoundary,
      resourcePromptRewritePreference,
      serviceSkills,
      translateAgentWorkspace,
      mentionAgentTurnRouteMap,
      mentionCommandPrefixKeyMap,
      mentionCommandSkillIdMap,
      workspaceRequestMetadataBase,
    ],
  );

  const executeLocalConfirmationPlan = useCallback(
    async (plan: WorkspaceLocalConfirmationPlan): Promise<boolean> => {
      const { sourceText, images, sendBoundary, submissionPreviewKey } = plan;
      setInput("");
      setMentionedCharacters([]);
      setChatMessages((previous) => {
        const timestamp = new Date();
        return [
          ...previous,
          {
            id: crypto.randomUUID(),
            role: "user",
            content: sourceText,
            images: images.length > 0 ? images : undefined,
            timestamp,
          },
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: plan.confirmation,
            timestamp: new Date(),
          },
        ];
      });

      finalizeAfterSendSuccess(sendBoundary);
      setSubmissionPreview((current) =>
        current?.key === submissionPreviewKey ? null : current,
      );
      return true;
    },
    [
      finalizeAfterSendSuccess,
      setChatMessages,
      setInput,
      setMentionedCharacters,
    ],
  );

  const executeSendPlan = useCallback(
    async (plan: WorkspaceSendPlan): Promise<boolean> => {
      const {
        sourceText,
        dispatchText,
        text,
        images,
        sendBoundary,
        browserRequirementForSend,
        effectiveWebSearch,
        effectiveSearchMode,
        submissionPreviewKey,
        sendExecutionStrategy,
        autoContinuePayload,
        sendOptions,
        completedMentionCommandUsage,
        completedMentionUsage,
      } = plan;

      const executeStartedAt = Date.now();
      logAgentDebug("WorkspaceSend", "execute.start", {
        imagesCount: images.length,
        messagesCount,
        sourceTextLength: sourceText.trim().length,
      });
      const effectiveToolPreferences = plan.effectiveToolPreferences;
      setInput("");
      setMentionedCharacters([]);

      try {
        let nextRequestMetadata = buildWorkspaceRequestMetadata({
          workspaceRequestMetadataBase,
          savedSoulArtifactVoiceGenerationBrief,
          soulArtifactVoiceEnabledForTurn,
          sendOptions: {
            ...(sendOptions || {}),
            toolPreferencesOverride: effectiveToolPreferences,
          },
          currentProviderType: providerType,
          effectiveToolPreferences,
          mappedTheme,
          isThemeWorkbench,
          currentGateKey,
          themeWorkbenchActiveQueueTitle,
          contentId,
          browserRequirementMatch: browserRequirementForSend,
          browserAssistProfileKey,
          browserAssistPreferredBackend,
          browserAssistAutoLaunch,
          workspaceSkillBindings,
          workspaceSkillRuntimeEnable,
          agentResponseLanguage,
        });
        const shouldRefreshServiceModels =
          Boolean(resolveServiceModelsBeforeSend) &&
          shouldRefreshServiceModelsBeforeSend({
            requestMetadata: nextRequestMetadata,
            purpose: sendOptions?.purpose,
          });
        const serviceModelsForSend = shouldRefreshServiceModels
          ? await resolveServiceModelsBeforeSend?.()
          : null;
        const effectiveServiceModels =
          serviceModelsForSend?.serviceModels ?? serviceModels;
        const effectiveAgentResponseLanguage =
          serviceModelsForSend?.agentResponseLanguage ?? agentResponseLanguage;
        if (shouldRefreshServiceModels) {
          nextRequestMetadata = buildWorkspaceRequestMetadata({
            workspaceRequestMetadataBase,
            savedSoulArtifactVoiceGenerationBrief,
            soulArtifactVoiceEnabledForTurn,
            sendOptions: {
              ...(sendOptions || {}),
              toolPreferencesOverride: effectiveToolPreferences,
            },
            currentProviderType: providerType,
            effectiveToolPreferences,
            mappedTheme,
            isThemeWorkbench,
            currentGateKey,
            themeWorkbenchActiveQueueTitle,
            contentId,
            browserRequirementMatch: browserRequirementForSend,
            browserAssistProfileKey,
            browserAssistPreferredBackend,
            browserAssistAutoLaunch,
            workspaceSkillBindings,
            workspaceSkillRuntimeEnable,
            agentResponseLanguage: effectiveAgentResponseLanguage,
          });
        }
        const serviceModelSendOverrides = resolveServiceModelSendOverrides({
          requestMetadata: nextRequestMetadata,
          purpose: sendOptions?.purpose,
          serviceModels: effectiveServiceModels,
        });
        const nextAssistantDraft =
          sendOptions?.assistantDraft ??
          buildImageWorkbenchAssistantDraft(nextRequestMetadata);
        const nextSendOptions: HandleSendOptions = {
          ...(sendOptions || {}),
          displayContent:
            dispatchText !== sourceText
              ? (sendOptions?.displayContent ?? sourceText)
              : sendOptions?.displayContent,
          requestMetadata: withConfiguredModelSlots(
            nextRequestMetadata,
            effectiveServiceModels,
          ),
          ...(effectiveSearchMode ? { searchMode: effectiveSearchMode } : {}),
          providerOverride:
            sendOptions?.providerOverride ??
            serviceModelSendOverrides.providerOverride,
          modelOverride:
            sendOptions?.modelOverride ??
            serviceModelSendOverrides.modelOverride,
          systemPromptOverride: sendOptions?.systemPromptOverride,
          assistantDraft: nextAssistantDraft,
        };

        logAgentDebug("WorkspaceSend", "sendMessage.start", {
          durationMs: Date.now() - executeStartedAt,
          responsiveModelSlotConfigured: Boolean(
            effectiveServiceModels?.responsive_chat?.enabled !== false &&
            effectiveServiceModels?.responsive_chat?.preferredProviderId &&
            effectiveServiceModels?.responsive_chat?.preferredModelId,
          ),
          modelOverride: nextSendOptions.modelOverride ?? null,
          providerOverride: nextSendOptions.providerOverride ?? null,
        });
        await sendMessage(
          text,
          images,
          effectiveWebSearch,
          undefined,
          false,
          sendExecutionStrategy,
          undefined,
          autoContinuePayload,
          nextSendOptions,
        );
        logAgentDebug("WorkspaceSend", "sendMessage.done", {
          durationMs: Date.now() - executeStartedAt,
        });

        if (completedMentionCommandUsage) {
          recordMentionEntryUsage({
            kind: "builtin_command",
            entryId: completedMentionCommandUsage.entryId,
            replayText: completedMentionCommandUsage.replayText,
            slotValues: completedMentionCommandUsage.slotValues,
          });
        }

        if (completedMentionUsage) {
          recordServiceSkillUsage(completedMentionUsage);
        }

        if (plan.completedSlashUsage) {
          recordSlashEntryUsage(plan.completedSlashUsage);
        }

        finalizeAfterSendSuccess(sendBoundary);
        return true;
      } catch (error) {
        rollbackAfterSendFailure(sendBoundary);
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        sendOptions?.observer?.onError?.(errorMessage);
        console.error("[AgentChat] 发送消息失败:", error);
        toast.error(resolveAgentRuntimeSubmitErrorMessage(errorMessage));
        setInput(sourceText);
        return false;
      } finally {
        setSubmissionPreview((current) =>
          current?.key === submissionPreviewKey ? null : current,
        );
      }
    },
    [
      agentResponseLanguage,
      browserAssistAutoLaunch,
      browserAssistPreferredBackend,
      browserAssistProfileKey,
      contentId,
      currentGateKey,
      finalizeAfterSendSuccess,
      isThemeWorkbench,
      mappedTheme,
      messagesCount,
      providerType,
      rollbackAfterSendFailure,
      serviceModels,
      resolveServiceModelsBeforeSend,
      workspaceSkillBindings,
      workspaceSkillRuntimeEnable,
      sendMessage,
      setInput,
      setMentionedCharacters,
      themeWorkbenchActiveQueueTitle,
      workspaceRequestMetadataBase,
      savedSoulArtifactVoiceGenerationBrief,
      soulArtifactVoiceEnabledForTurn,
    ],
  );

  const handleSend = useCallback<WorkspaceHandleSend>(
    async (
      images,
      webSearch,
      thinking,
      textOverride,
      sendExecutionStrategy,
      autoContinuePayload,
      sendOptions,
    ) => {
      if (isPreparingSendRef.current) {
        return false;
      }

      isPreparingSendRef.current = true;
      setIsPreparingSend(true);

      try {
        const resolution = await resolveSendExecutionPlan(
          images,
          webSearch,
          thinking,
          textOverride,
          sendExecutionStrategy,
          autoContinuePayload,
          sendOptions,
        );
        if (resolution.kind === "done") {
          return resolution.result;
        }
        if (resolution.kind === "local_confirmation") {
          return executeLocalConfirmationPlan(resolution.plan);
        }
        return executeSendPlan(resolution.plan);
      } finally {
        isPreparingSendRef.current = false;
        setIsPreparingSend(false);
      }
    },
    [executeLocalConfirmationPlan, executeSendPlan, resolveSendExecutionPlan],
  );

  const handleRecommendationClick = useCallback(
    (shortLabel: string, fullPrompt: string) => {
      setInput(fullPrompt);

      if (
        activeTheme !== "general" ||
        !isTeamRuntimeRecommendation(shortLabel, fullPrompt)
      ) {
        return;
      }

      const nextToolPreferences = chatToolPreferences.subagent
        ? chatToolPreferences
        : {
            ...chatToolPreferences,
            subagent: true,
          };

      if (!chatToolPreferences.subagent) {
        setChatToolPreferences(nextToolPreferences);
      }
      saveChatToolPreferences(nextToolPreferences, activeTheme);
      void handleSend(
        [],
        undefined,
        undefined,
        fullPrompt,
        executionStrategy,
        undefined,
        {
          toolPreferencesOverride: nextToolPreferences,
        },
      );
    },
    [
      activeTheme,
      chatToolPreferences,
      executionStrategy,
      handleSend,
      setChatToolPreferences,
      setInput,
    ],
  );

  const handleSendRef = useRef(handleSend);

  useEffect(() => {
    handleSendRef.current = handleSend;
  }, [handleSend]);

  return {
    handleSend,
    handleRecommendationClick,
    handleSendRef,
    isPreparingSend,
    displayMessages,
  };
}
