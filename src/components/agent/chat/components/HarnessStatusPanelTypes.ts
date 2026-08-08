import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import type { ThreadGoal } from "@limecloud/app-server-client";
import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import type { AgentRuntimeToolInventory } from "@/lib/api/agentRuntime/toolInventoryTypes";
import type { CanonicalChildThreadSummary } from "../projection/canonicalChildThreadSummary";
import type {
  ActionRequired,
  AgentThreadItem,
  AgentThreadTurn,
  ConfirmResponse,
  Message,
} from "../types";
import type { HarnessSessionState } from "../utils/harnessState";
import type { HarnessEnvironmentSummary } from "./HarnessActivityTypes";
import type { HarnessFilePreviewResult } from "./useHarnessPreviewDialog";
import type { HarnessFileChangeReviewSummary } from "./useHarnessFileReviewState";
import type { HarnessSectionKey } from "./HarnessStatusSectionFrame";
import type {
  ExecutionPolicyFocusContext,
  ProviderSettingsFocusContext,
} from "@/types/page";

export type HarnessStatusPanelLayout = "default" | "sidebar" | "dialog";

export interface HarnessSummaryCard {
  sectionKey: HarnessSectionKey;
  title: string;
  value: string;
  hint: string;
  icon: LucideIcon;
}

export interface HarnessLeadContentContext {
  fileChangeReviewSummary: HarnessFileChangeReviewSummary;
}

export type HarnessLeadContent =
  | ReactNode
  | ((context: HarnessLeadContentContext) => ReactNode);

export interface HarnessStatusPanelProps {
  harnessState: HarnessSessionState;
  environment: HarnessEnvironmentSummary;
  layout?: HarnessStatusPanelLayout;
  onLoadFilePreview?: (path: string) => Promise<HarnessFilePreviewResult>;
  onOpenFile?: (fileName: string, content: string) => void;
  onRevealPath?: (path: string) => Promise<void>;
  onOpenPath?: (path: string) => Promise<void>;
  onOpenFileCheckpoints?: () => void;
  canonicalChildren?: CanonicalChildThreadSummary[];
  onOpenSubagentSession?: (sessionId: string) => void;
  toolInventory?: AgentRuntimeToolInventory | null;
  toolInventoryLoading?: boolean;
  toolInventoryError?: string | null;
  onRefreshToolInventory?: () => void;
  title?: string;
  description?: string;
  toggleLabel?: string;
  leadContent?: HarnessLeadContent;
  threadRead?: AgentRuntimeThreadReadModel | null;
  threadGoal?: ThreadGoal | null;
  threadGoalError?: unknown;
  threadGoalLoading?: boolean;
  turns?: AgentThreadTurn[];
  threadItems?: AgentThreadItem[];
  currentTurnId?: string | null;
  pendingActions?: ActionRequired[];
  submittedActionsInFlight?: ActionRequired[];
  onRespondToAction?: (response: ConfirmResponse) => void | Promise<void>;
  canInterrupt?: boolean;
  onInterruptCurrentTurn?: () => void | Promise<void>;
  onReplayPendingRequest?: (requestId: string) => boolean | Promise<boolean>;
  onOpenMemoryWorkbench?: () => void;
  onManageProviders?: (context?: ProviderSettingsFocusContext) => void;
  onOpenExecutionPolicySettings?: (
    context?: ExecutionPolicyFocusContext,
  ) => void;
  messages?: Message[];
  diagnosticRuntimeContext?: {
    sessionId?: string | null;
    workspaceId?: string | null;
    workingDir?: string | null;
    providerType?: string | null;
    model?: string | null;
    executionStrategy?: string | null;
    activeTheme?: string | null;
  } | null;
}
