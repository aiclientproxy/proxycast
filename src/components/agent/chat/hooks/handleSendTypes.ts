import type {
  AgentInputMention,
  AssistantDraftState,
  SlashSkillRequest,
  ThreadGoalInput,
} from "./agentChatShared";
import type { ModeKind, RuntimeSearchMode } from "@limecloud/app-server-client";
import type { ChatToolPreferences } from "../utils/chatToolPreferences";
import type { InputCapabilitySendRoute } from "../skill-selection/inputCapabilitySelection";
import type { InterruptedInputDraftSnapshot } from "./agentStreamInputRestoreTypes";
import type {
  ComposerDraftSnapshot,
  ComposerIntent,
  ComposerSubmitTarget,
} from "@/components/input-kit";

export interface HandleSendObserver {
  onComplete?: (content: string) => void;
  onError?: (message: string) => void;
}

export interface HandleSendOptions {
  skipThemeSkillPrefix?: boolean;
  skipSceneCommandRouting?: boolean;
  skipWorkspaceCommandRouting?: boolean;
  purpose?: "content_review" | "text_stylize" | "style_rewrite" | "style_audit";
  observer?: HandleSendObserver;
  requestMetadata?: Record<string, unknown>;
  collaborationMode?: ModeKind;
  threadGoal?: ThreadGoalInput;
  toolPreferencesOverride?: ChatToolPreferences;
  displayContent?: string;
  skillRequest?: SlashSkillRequest;
  capabilityRoute?: InputCapabilitySendRoute;
  inputRestoreDraft?: InterruptedInputDraftSnapshot;
  inputMentions?: readonly AgentInputMention[];
  providerOverride?: string;
  modelOverride?: string;
  reasoningEffort?: string;
  systemPromptOverride?: string;
  searchMode?: RuntimeSearchMode;
  explicitToolPreferences?: boolean;
  assistantDraft?: AssistantDraftState;
  targetSessionId?: string;
  skipSessionRestore?: boolean;
  skipSessionStartHooks?: boolean;
  composerIntent?: ComposerIntent;
  composerTarget?: ComposerSubmitTarget;
  composerDraft?: ComposerDraftSnapshot;
}
