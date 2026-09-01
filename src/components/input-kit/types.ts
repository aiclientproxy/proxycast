import type { ModelReasoningEffortLevel } from "@/lib/types/modelRegistry";

export interface ComposerAttachment {
  data: string;
  mediaType: string;
  sourceUri?: string;
  sourcePath?: string;
  previewUrl?: string;
  metadata?: Record<string, unknown>;
  index?: number;
}

export interface ComposerPathReference {
  id: string;
  path: string;
  name: string;
  isDir: boolean;
  size?: number | null;
  mimeType?: string | null;
  source?: "file_manager" | "system_drop";
}

export interface ComposerMentionBinding {
  id: string;
  name: string;
  path: string;
  from?: number;
  to?: number;
}

export interface ComposerPendingPaste {
  placeholder: string;
  text: string;
}

export interface ComposerDocument {
  text: string;
  selectionStart: number;
  selectionEnd: number;
  textElements: readonly unknown[];
  mentionBindings: readonly ComposerMentionBinding[];
  attachments: readonly ComposerAttachment[];
  pathReferences: readonly ComposerPathReference[];
  pendingPastes: readonly ComposerPendingPaste[];
  inputCapabilityRoute?: unknown;
}

export type ComposerIntent =
  | "start"
  | "queue"
  | "steer"
  | "interrupt"
  | "command"
  | "edit"
  | "clear";

export type ComposerHistoryDirection = "previous" | "next";

export type ComposerSubmitTarget =
  | "start"
  | "queue"
  | "steer"
  | "interrupt"
  | "command";

export type ComposerDraftSnapshot = ComposerDocument;

export interface ComposerControllerSnapshot {
  document: ComposerDocument;
  revision: number;
  lastIntent: ComposerIntent | null;
}

export type ComposerSubmitResult =
  | {
      kind: "accepted";
      intent: ComposerIntent;
      target: ComposerSubmitTarget;
      draft: ComposerDraftSnapshot;
      revision: number;
    }
  | {
      kind: "empty";
      intent: ComposerIntent;
      target: ComposerSubmitTarget;
      draft: ComposerDraftSnapshot;
      revision: number;
    };

export interface ComposerPasteOptions {
  platform?: "macos" | "windows" | "linux" | "unknown";
  bracketed?: boolean;
}

export interface ComposerState {
  text: string;
  isSending: boolean;
  disabled?: boolean;
  attachments?: ComposerAttachment[];
  document?: ComposerDocument;
}

export interface ModelSelectionState {
  providerType: string;
  model: string;
  reasoningEffort?: ModelReasoningEffortLevel | "";
  providersLoading?: boolean;
  modelsLoading?: boolean;
}

export interface ComposerActions {
  setText: (value: string) => void;
  send: (options?: { textOverride?: string }) => void;
  stop?: () => void;
  setProviderType?: (providerType: string) => void;
  setModel?: (model: string) => void;
  setReasoningEffort?: (value: ModelReasoningEffortLevel | "") => void;
}
