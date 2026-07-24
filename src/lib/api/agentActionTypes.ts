export type ApprovalDecision =
  | "allow_once"
  | "allow_for_session"
  | "decline"
  | "cancel";

export interface ActionRequiredScope {
  sessionId?: string;
  threadId?: string;
  turnId?: string;
}

export interface ActionRequestGovernanceMeta {
  strategy: "single_turn_single_question";
  source: "runtime_action_required";
  originalQuestionCount?: number;
  originalFieldCount?: number;
  originalSectionCount?: number;
  retainedQuestionIndex?: number;
  retainedFieldKey?: string;
  retainedSectionIndex?: number;
  deferredQuestionCount?: number;
  deferredFieldCount?: number;
}

export interface Question {
  question: string;
  header?: string;
  options?: QuestionOption[];
  multiSelect?: boolean;
}

export interface QuestionOption {
  label: string;
  description?: string;
}

export interface ActionRequired {
  requestId: string;
  actionType: "tool_confirmation" | "ask_user" | "elicitation";
  toolName?: string;
  arguments?: Record<string, unknown>;
  prompt?: string;
  questions?: Question[];
  requestedSchema?: any;
  scope?: ActionRequiredScope;
  eventName?: string;
  sourceMessageId?: string;
  status?: "pending" | "queued" | "submitted";
  isFallback?: boolean;
  submittedResponse?: string;
  submittedUserData?: unknown;
  detail?: string;
  availableDecisions?: ApprovalDecision[];
  governance?: ActionRequestGovernanceMeta;
}

export interface ConfirmResponse {
  requestId: string;
  confirmed?: boolean;
  decision?: ApprovalDecision;
  response?: string;
  actionType?: ActionRequired["actionType"];
  userData?: unknown;
}
