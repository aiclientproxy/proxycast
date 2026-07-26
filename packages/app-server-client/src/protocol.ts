// @generated types re-export — 从 Rust JSON Schema 自动生成的类型定义
// 新代码优先从这里导入类型；手写类型逐步迁移后将删除
import {
  GENERATED_APP_SERVER_METHODS,
  GENERATED_APP_SERVER_REQUEST_SERIALIZATION_SCOPES,
  METHOD_AGENT_SESSION_EVENT,
  METHOD_TURN_INTERRUPT,
  METHOD_TURN_START,
  METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED,
} from "./generated/protocol-types.js";
import type {
  AgentSessionMediaReadResponse as GeneratedAgentSessionMediaReadResponse,
  ConversationImportJobPhase as GeneratedConversationImportJobPhase,
  ConversationImportJobStatus as GeneratedConversationImportJobStatus,
  ConversationImportSourceClient as GeneratedConversationImportSourceClient,
  ConversationImportSourceStatus as GeneratedConversationImportSourceStatus,
  ConversationImportThreadStatus as GeneratedConversationImportThreadStatus,
  GeneratedAppServerRequestSerializationScope,
  TurnInterruptParams as GeneratedTurnInterruptParams,
  TurnInterruptResponse as GeneratedTurnInterruptResponse,
  TurnStartParams as GeneratedTurnStartParams,
  TurnStartResponse as GeneratedTurnStartResponse,
  WorkspaceRightSurfacePendingChangedParams as GeneratedWorkspaceRightSurfacePendingChangedParams,
} from "./generated/protocol-types.js";
export * from "./generated/protocol-types.js";
export type {
  ExecutionProcessDrainOutputParams,
  ExecutionProcessDrainOutputResponse,
  ExecutionProcessEmptyResponse,
  ExecutionProcessIdParams,
  ExecutionProcessStartParams,
  ExecutionProcessStartResponse,
  ExecutionProcessStatusResponse,
  ExecutionProcessWriteStdinParams,
} from "./generated/protocol-types.js";

export const JSONRPC_VERSION = "2.0";
export const PROTOCOL_VERSION = "appserver.v0";
export const SERVER_NAME = "app-server";
export const METHOD_CANCEL_REQUEST = "$/cancelRequest";
export const METHOD_VOICE_TRANSCRIPTION_POLISH_TEXT =
  "voiceTranscription/polishText";

export const CONVERSATION_IMPORT_SOURCE_CLIENTS = [
  "codex",
] as const satisfies readonly GeneratedConversationImportSourceClient[];
export const CONVERSATION_IMPORT_SOURCE_STATUSES = [
  "ready",
  "missing",
] as const satisfies readonly GeneratedConversationImportSourceStatus[];
export const CONVERSATION_IMPORT_THREAD_STATUSES = [
  "not_imported",
  "importing",
  "imported",
  "conflict",
] as const satisfies readonly GeneratedConversationImportThreadStatus[];
export const CONVERSATION_IMPORT_JOB_STATUSES = [
  "queued",
  "running",
  "completed",
  "failed",
] as const satisfies readonly GeneratedConversationImportJobStatus[];
export const CONVERSATION_IMPORT_JOB_PHASES = [
  "queued",
  "reading_source",
  "building_history",
  "persisting_history",
  "finalizing",
  "completed",
  "failed",
] as const satisfies readonly GeneratedConversationImportJobPhase[];

export type AppServerMethodKind = "request" | "notification" | "serverRequest";

export type AppServerMethodSpec = {
  method: string;
  kind: AppServerMethodKind;
};

export const APP_SERVER_METHODS =
  GENERATED_APP_SERVER_METHODS satisfies readonly AppServerMethodSpec[];

export type AppServerRequestSerializationScope =
  GeneratedAppServerRequestSerializationScope;

export type AppServerRequestSerializationScopeSpec = {
  method: string;
  scope: AppServerRequestSerializationScope;
};

export const APP_SERVER_REQUEST_SERIALIZATION_SCOPES =
  GENERATED_APP_SERVER_REQUEST_SERIALIZATION_SCOPES satisfies readonly AppServerRequestSerializationScopeSpec[];

export const ERROR_CODES = {
  parseError: -32700,
  invalidRequest: -32600,
  methodNotFound: -32601,
  invalidParams: -32602,
  runtimeError: -32000,
  notInitialized: -32002,
  alreadyInitialized: -32003,
  sessionNotFound: -32010,
  turnNotActive: -32011,
  sessionAlreadyExists: -32013,
  capabilityDenied: -32020,
  requestCancelled: -32800,
} as const;

export type RequestId = number | string;
export type RpcResult = unknown;
export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

export type JsonRpcRequest = {
  id: RequestId;
  method: string;
  params?: unknown;
};

export type JsonRpcNotification = {
  method: string;
  params?: unknown;
};

export type JsonRpcCancelRequestParams = {
  id: RequestId;
};

export type JsonRpcResponse<T = RpcResult> = {
  id: RequestId;
  result: T;
};

export type JsonRpcError = {
  code: number;
  message: string;
  data?: unknown;
};

export type JsonRpcErrorResponse = {
  id: RequestId;
  error: JsonRpcError;
};

export type JsonRpcMessage<T = RpcResult> =
  | JsonRpcRequest
  | JsonRpcNotification
  | JsonRpcResponse<T>
  | JsonRpcErrorResponse;

export type ClientInfo = {
  name: string;
  title?: string;
  version?: string;
};

export type ClientCapabilities = {
  eventMethods?: string[];
  experimental?: boolean;
  optOutNotificationMethods?: string[];
};

export type InitializeParams = {
  clientInfo: ClientInfo;
  capabilities?: ClientCapabilities;
};

export type InitializeResponse = {
  serverInfo: ServerInfo;
  platform: PlatformInfo;
  capabilities: ServerCapabilities;
};

export type ServerInfo = {
  name: string;
  version: string;
  protocolVersion: string;
};

export type PlatformInfo = {
  family: string;
  os: string;
};

export type ServerCapabilities = {
  agentSession: boolean;
  capabilityDiscovery: boolean;
  artifact: boolean;
  evidence: boolean;
  workspace: boolean;
};

export type CapabilityListParams = {
  appId?: string;
  workspaceId?: string;
  sessionId?: string;
  cursor?: string;
  limit?: number;
};

export type CapabilityDescriptor = {
  id: string;
  title: string;
  description?: string;
  methods: string[];
};

export type CapabilityListResponse = {
  capabilities: CapabilityDescriptor[];
  runtimeCapabilityManifest?: RuntimeCapabilityManifest;
  nextCursor?: string;
};

export type RuntimeCapabilityManifest = {
  schemaVersion: string;
  runtimeId: string;
  providerId?: string;
  sessionId?: string;
  generatedAt: string;
  capabilities: RuntimeCapabilityEntry[];
};

export type RuntimeCapabilityEntry = {
  id: string;
  status: string;
  scope: string;
  title: string;
  detail?: string;
  version?: string;
  metadata?: Record<string, unknown>;
};

export type ArtifactReadParams = {
  sessionId: string;
  turnId?: string;
  artifactRef?: string;
  includeContent?: boolean;
  cursor?: string;
  limit?: number;
};

export type ArtifactContentStatus =
  | "notRequested"
  | "available"
  | "unavailable";

export type ArtifactSummary = {
  artifactRef: string;
  eventId: string;
  sequence: number;
  turnId?: string;
  artifactId?: string;
  path?: string;
  title?: string;
  kind?: string;
  status?: string;
  content?: string;
  contentStatus: ArtifactContentStatus;
  metadata?: unknown;
};

export type ArtifactReadResponse = {
  artifacts: ArtifactSummary[];
  nextCursor?: string;
};

export type FileSystemListDirectoryParams = {
  path: string;
};

export type FileSystemReadFilePreviewParams = {
  path: string;
  maxSize?: number;
};

export type FileSystemCreateFileParams = {
  path: string;
};

export type FileSystemCreateDirectoryParams = {
  path: string;
};

export type FileSystemRenameFileParams = {
  oldPath: string;
  newPath: string;
};

export type FileSystemDeleteFileParams = {
  path: string;
  recursive?: boolean;
};

export type FileSystemMutationResponse = Record<string, never>;

export type FileSystemDirectoryListing = {
  path: string;
  parentPath: string | null;
  entries: FileSystemFileEntry[];
  error: string | null;
};

export type FileSystemFileEntry = {
  name: string;
  path: string;
  isDir: boolean;
  size: number;
  modifiedAt: number;
  fileType?: string;
  isHidden: boolean;
  modeStr?: string;
  mode?: number;
  mimeType?: string;
  isSymlink: boolean;
  iconDataUrl?: string;
};

export type FileSystemFilePreview = {
  path: string;
  content: string | null;
  isBinary: boolean;
  size: number;
  error: string | null;
};

export type ProjectGitStatusParams = {
  rootPath: string;
};

export type ProjectGitDiffParams = {
  rootPath: string;
  contextLines?: number;
  base?: ProjectGitDiffBase;
  commitSha?: string;
};

export type ProjectGitDiffBase =
  | "unstaged"
  | "staged"
  | "commit"
  | "branch"
  | "previousConversation";

export type ProjectGitCommitListParams = {
  rootPath: string;
  limit?: number;
};

export type ProjectGitBranchCheckoutParams = {
  rootPath: string;
  branch: string;
};

export type ProjectGitBranchCreateParams = {
  rootPath: string;
  branch: string;
};

export type ProjectGitWorktreeCreateParams = {
  rootPath: string;
  name?: string;
  baseBranch?: string;
};

export type ProjectGitStatusResponse = {
  rootPath: string;
  repositoryRoot?: string;
  hasGitRepository: boolean;
  currentBranch?: string;
  branches: string[];
  uncommittedFileCount: number;
};

export type ProjectGitBranchCheckoutResponse = ProjectGitStatusResponse;
export type ProjectGitBranchCreateResponse = ProjectGitStatusResponse;

export type ProjectGitDiffResponse = {
  rootPath: string;
  repositoryRoot?: string;
  hasGitRepository: boolean;
  currentRef?: string | null;
  comparisonBaseRef?: string | null;
  patch: string;
  uncommittedFileCount: number;
};

export type ProjectGitCommitListResponse = {
  rootPath: string;
  repositoryRoot?: string;
  hasGitRepository: boolean;
  commits: ProjectGitCommit[];
};

export type ProjectGitCommit = {
  sha: string;
  shortSha: string;
  subject: string;
  authorName: string;
  authorEmail: string;
  committedAt: string;
};

export type ProjectGitWorktreeCreateResponse = {
  worktreePath: string;
  branch: string;
  status: ProjectGitStatusResponse;
};

export type EvidenceExportParams = {
  sessionId: string;
  turnId?: string;
  includeEvents?: boolean;
  includeArtifacts?: boolean;
  includeEvidencePack?: boolean;
};

export type EvidenceExportResponse = {
  session: AgentSession;
  turns: AgentTurn[];
  events: AgentEvent[];
  artifacts: ArtifactSummary[];
  exportedAt: string;
  evidencePack?: EvidencePackSummary;
};

export type EvidencePackSummary = {
  packRelativeRoot: string;
  packAbsoluteRoot?: string;
  exportedAt: string;
  threadStatus: string;
  latestTurnStatus?: string;
  turnCount: number;
  itemCount: number;
  pendingRequestCount: number;
  queuedTurnCount: number;
  recentArtifactCount: number;
  knownGaps: string[];
  observabilitySummary?: unknown;
  completionAuditSummary?: unknown;
  artifacts: EvidencePackArtifact[];
};

export type EvidencePackArtifact = {
  kind: string;
  title: string;
  relativePath: string;
  absolutePath?: string;
  bytes: number;
};

export type AgentSessionHandoffBundleExportParams = {
  sessionId: string;
  locale?: string;
};

export type AgentSessionHandoffBundleExportResponse = {
  sessionId: string;
  threadId: string;
  workspaceId?: string;
  workspaceRoot: string;
  bundleRelativeRoot: string;
  bundleAbsoluteRoot: string;
  exportedAt: string;
  threadStatus: string;
  latestTurnStatus?: string;
  pendingRequestCount: number;
  queuedTurnCount: number;
  activeSubagentCount: number;
  todoTotal: number;
  todoPending: number;
  todoInProgress: number;
  todoCompleted: number;
  artifacts: AgentSessionHandoffArtifact[];
};

export type AgentSessionHandoffArtifact = {
  kind: string;
  title: string;
  relativePath: string;
  absolutePath: string;
  bytes: number;
};

export type AgentSessionReplayCaseExportParams = {
  sessionId: string;
  locale?: string;
};

export type AgentSessionReplayCaseExportResponse = {
  sessionId: string;
  threadId: string;
  workspaceId?: string;
  workspaceRoot: string;
  replayRelativeRoot: string;
  replayAbsoluteRoot: string;
  handoffBundleRelativeRoot: string;
  evidencePackRelativeRoot: string;
  exportedAt: string;
  threadStatus: string;
  latestTurnStatus?: string;
  pendingRequestCount: number;
  queuedTurnCount: number;
  linkedHandoffArtifactCount: number;
  linkedEvidenceArtifactCount: number;
  recentArtifactCount: number;
  artifacts: AgentSessionHandoffArtifact[];
};

export type AgentSessionAnalysisHandoffExportParams = {
  sessionId: string;
  locale?: string;
};

export type AgentSessionAnalysisHandoffExportResponse = {
  sessionId: string;
  threadId: string;
  workspaceId?: string;
  workspaceRoot: string;
  sanitizedWorkspaceRoot: string;
  analysisRelativeRoot: string;
  analysisAbsoluteRoot: string;
  handoffBundleRelativeRoot: string;
  evidencePackRelativeRoot: string;
  replayCaseRelativeRoot: string;
  exportedAt: string;
  threadStatus: string;
  latestTurnStatus?: string;
  pendingRequestCount: number;
  queuedTurnCount: number;
  title: string;
  copyPrompt: string;
  artifacts: AgentSessionHandoffArtifact[];
};

export type AgentSessionReviewDecisionTemplateExportParams = {
  sessionId: string;
  locale?: string;
};

export type AgentSessionReviewDecisionSaveParams = {
  sessionId: string;
  decisionStatus: string;
  decisionSummary?: string;
  chosenFixStrategy?: string;
  riskLevel: string;
  riskTags?: string[];
  humanReviewer?: string;
  followupActions?: string[];
  regressionRequirements?: string[];
  notes?: string;
  locale?: string;
};

export type AgentSessionReviewDecision = {
  decisionStatus: string;
  decisionSummary?: string;
  chosenFixStrategy?: string;
  riskLevel: string;
  riskTags: string[];
  humanReviewer?: string;
  followupActions: string[];
  regressionRequirements: string[];
  notes?: string;
};

export type AgentSessionReviewDecisionTemplateExportResponse = {
  sessionId: string;
  threadId: string;
  workspaceId?: string;
  workspaceRoot: string;
  reviewRelativeRoot: string;
  reviewAbsoluteRoot: string;
  analysisRelativeRoot: string;
  analysisAbsoluteRoot: string;
  handoffBundleRelativeRoot: string;
  evidencePackRelativeRoot: string;
  replayCaseRelativeRoot: string;
  exportedAt: string;
  threadStatus: string;
  latestTurnStatus?: string;
  pendingRequestCount: number;
  queuedTurnCount: number;
  title: string;
  defaultDecisionStatus: string;
  decision: AgentSessionReviewDecision;
  decisionStatusOptions: string[];
  riskLevelOptions: string[];
  reviewChecklist: string[];
  analysisArtifacts: AgentSessionHandoffArtifact[];
  artifacts: AgentSessionHandoffArtifact[];
};

export type BusinessObjectRef = {
  kind: string;
  id: string;
  title?: string;
  uri?: string;
  metadata?: unknown;
};

export type AgentSessionStartParams = {
  sessionId?: string;
  threadId?: string;
  appId: string;
  workspaceId?: string;
  businessObjectRef?: BusinessObjectRef;
  locale?: string;
};

export type AgentSessionReadParams = {
  sessionId: string;
  historyLimit?: number;
  historyOffset?: number;
  historyBeforeMessageId?: number;
};

export type AgentSessionListParams = {
  includeArchived?: boolean;
  archivedOnly?: boolean;
  cwd?: string | string[];
  workspaceId?: string;
  limit?: number;
};

export type AgentInput = {
  text: string;
  attachments?: AgentAttachment[];
};

export type AgentAttachment = {
  kind: string;
  uri?: string;
  metadata?: unknown;
};

export type RuntimeToolCallStrategy = "native" | "tool_shim";

export type RuntimeSearchMode = "disabled" | "auto" | "required";

export type RuntimeProviderConfig = {
  providerId?: string;
  providerName?: string;
  modelName?: string;
  apiKey?: string;
  baseUrl?: string;
  toolCallStrategy?: RuntimeToolCallStrategy;
  toolshimModel?: string;
  modelCapabilities?: unknown;
  supportsWebsockets?: boolean;
};

export type RuntimeRequest = {
  providerConfig?: RuntimeProviderConfig;
  providerPreference?: string;
  modelPreference?: string;
  reasoningEffort?: string;
  thinkingEnabled?: boolean;
  approvalPolicy?: string;
  sandboxPolicy?: string;
  workspaceId?: string;
  workingDir?: string;
  workspaceRoot?: string;
  projectRoot?: string;
  webSearch?: boolean;
  searchMode?: RuntimeSearchMode;
  executionStrategy?: string;
  autoContinue?: boolean;
  systemPrompt?: string;
  metadata?: unknown;
};

export type RuntimeOptions = {
  capabilityId?: string;
  stream?: boolean;
  eventName?: string;
  queuedTurnId?: string;
  runtimeRequest?: RuntimeRequest;
  expectedOutput?: unknown;
  structuredOutput?: StructuredOutputContract;
  outputSchema?: unknown;
};

export type StructuredOutputContract = {
  type?: string;
  schemaRef?: string;
  schema?: unknown;
  maxValidationRetries?: number;
  failureSubtype?: string;
  materializer?: unknown;
  metadata?: unknown;
};

/** v2 `turn/start` 是当前 runtime 主链唯一的 turn-start payload。 */
export type AgentSessionTurnStartParams = GeneratedTurnStartParams;

export type AgentSessionTurnStartRequest = JsonRpcRequest & {
  method: typeof METHOD_TURN_START;
  params: AgentSessionTurnStartParams;
};

/** @deprecated 使用 TurnInterruptParams。 */
export type AgentSessionTurnCancelParams = GeneratedTurnInterruptParams;

export type AgentSessionTurnInterruptRequest = JsonRpcRequest & {
  method: typeof METHOD_TURN_INTERRUPT;
  params: GeneratedTurnInterruptParams;
};

export type AgentSessionActionType =
  | "tool_confirmation"
  | "ask_user"
  | "elicitation";

export type AgentSessionApprovalDecision =
  | "allow_once"
  | "allow_for_session"
  | "decline"
  | "cancel";

export type AgentSessionActionScope = {
  sessionId?: string;
  threadId?: string;
  turnId?: string;
};

export type AgentSessionActionRespondParams = {
  sessionId: string;
  requestId: string;
  actionType: AgentSessionActionType;
  decision?: AgentSessionApprovalDecision;
  confirmed?: boolean;
  response?: string;
  userData?: unknown;
  metadata?: unknown;
  eventName?: string;
  actionScope?: AgentSessionActionScope;
};

export type AgentSessionStatus =
  | "idle"
  | "running"
  | "waitingAction"
  | "completed"
  | "failed"
  | "canceled";

export type AgentSession = {
  sessionId: string;
  threadId: string;
  appId: string;
  workspaceId?: string;
  businessObjectRef?: BusinessObjectRef;
  status: AgentSessionStatus;
  createdAt: string;
  updatedAt: string;
};

export type AgentTurnStatus =
  | "accepted"
  | "queued"
  | "running"
  | "waitingAction"
  | "completed"
  | "failed"
  | "canceled";

export type AgentTurn = {
  turnId: string;
  sessionId: string;
  threadId: string;
  status: AgentTurnStatus;
  startedAt?: string;
  completedAt?: string;
};

export type AgentEvent = {
  eventId: string;
  sequence: number;
  sessionId: string;
  threadId?: string;
  turnId?: string;
  type: string;
  timestamp: string;
  payload: unknown;
};

export type AgentSessionStartResponse = {
  session: AgentSession;
};

export type AgentSessionOverview = {
  sessionId: string;
  threadId?: string;
  title?: string;
  businessObjectRefMetadata?: unknown;
  model: string;
  createdAt: string;
  updatedAt: string;
  archivedAt?: string;
  workspaceId?: string;
  workingDir?: string;
  executionStrategy?: string;
  messagesCount: number;
  threadStatus?: string;
  latestTurnStatus?: string;
  activeTurnId?: string;
  queuedTurnCount?: number;
};

export type AgentSessionListResponse = {
  sessions: AgentSessionOverview[];
};

export type AgentSessionFileCheckpointListParams = {
  sessionId: string;
};

export type AgentSessionFileCheckpointGetParams = {
  sessionId: string;
  checkpointId: string;
};

export type AgentSessionFileCheckpointDiffParams = {
  sessionId: string;
  checkpointId: string;
};

export type AgentSessionFileCheckpointRestoreParams = {
  sessionId: string;
  checkpointId: string;
  confirmRestore: boolean;
  createBackup?: boolean;
};

export type AgentSessionFileCheckpointSummary = {
  checkpointId: string;
  turnId: string;
  path: string;
  source: string;
  updatedAt: string;
  versionNo?: number;
  versionId?: string;
  requestId?: string;
  title?: string;
  kind?: string;
  status?: string;
  previewText?: string;
  snapshotPath?: string;
  validationIssueCount: number;
};

export type AgentSessionFileCheckpointThreadSummary = {
  count: number;
  latestCheckpoint?: AgentSessionFileCheckpointSummary;
};

export type AgentSessionFileCheckpointListResponse = {
  sessionId: string;
  threadId: string;
  checkpointCount: number;
  checkpoints: AgentSessionFileCheckpointSummary[];
};

export type AgentSessionFileCheckpointDetail = {
  sessionId: string;
  threadId: string;
  checkpoint: AgentSessionFileCheckpointSummary;
  livePath: string;
  snapshotPath: string;
  checkpointDocument?: unknown;
  liveDocument?: unknown;
  versionHistory: unknown[];
  validationIssues: string[];
  metadata?: unknown;
  content?: string;
};

export type AgentSessionFileCheckpointDiffResponse = {
  sessionId: string;
  threadId: string;
  checkpoint: AgentSessionFileCheckpointSummary;
  currentVersionId?: string;
  previousVersionId?: string;
  diff?: unknown;
};

export type AgentSessionFileCheckpointRestoreResponse = {
  sessionId: string;
  threadId: string;
  checkpoint: AgentSessionFileCheckpointSummary;
  livePath: string;
  snapshotPath: string;
  backupPath?: string | null;
  restoredAt: string;
};

export type SessionFileIdParams = {
  sessionId: string;
  fileName: string;
};

export type SessionFileGetOrCreateParams = {
  sessionId: string;
};

export type SessionFileUpdateMetaParams = {
  sessionId: string;
  title?: string;
  theme?: string;
  creationMode?: string;
};

export type SessionFileSaveParams = {
  sessionId: string;
  fileName: string;
  content: string;
  metadata?: Record<string, unknown> | unknown;
};

export type SessionFileMeta = {
  sessionId: string;
  title?: string;
  theme?: string;
  creationMode?: string;
  createdAt: number;
  updatedAt: number;
  fileCount: number;
  totalSize: number;
};

export type SessionFileEntry = {
  name: string;
  fileType: string;
  metadata?: Record<string, unknown> | unknown;
  size: number;
  createdAt: number;
  updatedAt: number;
};

export type SessionFileMetaResponse = {
  meta: SessionFileMeta;
};

export type SessionFileEntryResponse = {
  file: SessionFileEntry;
};

export type SessionFileReadResponse = {
  content: string;
};

export type SessionFileResolvePathResponse = {
  path: string;
};

export type SessionFileListResponse = {
  files: SessionFileEntry[];
};

export type SessionFileMutationResponse = Record<string, never>;

export type WorkspaceReadParams = {
  id: string;
};

export type WorkspaceUpdateParams = {
  id: string;
  name?: string;
  rootPath?: string;
  settings?: unknown;
  icon?: string;
  color?: string;
  isFavorite?: boolean;
  isArchived?: boolean;
  tags?: string[];
  defaultPersonaId?: string;
};

export type WorkspaceDeleteParams = {
  id: string;
  deleteDirectory?: boolean;
};

export type WorkspacePathReadParams = {
  rootPath: string;
};

export type WorkspaceEnsureProjectParams = {
  name: string;
  rootPath: string;
  workspaceType?: string;
};

export type WorkspaceProjectPathResolveParams = {
  name: string;
  parentRootPath?: string;
};

export type WorkspaceEnsureParams = {
  id: string;
};

export type WorkspaceListResponse = {
  workspaces: unknown[];
};

export type WorkspaceReadResponse = {
  workspace?: unknown;
};

export type WorkspaceUpdateResponse = {
  workspace: unknown;
};

export type WorkspaceDeleteResponse = {
  deleted: boolean;
};

export type WorkspaceEnsureProjectResponse = {
  workspace: unknown;
  created: boolean;
  rootCreated: boolean;
};

export type WorkspaceProjectsRootReadResponse = {
  rootPath: string;
};

export type WorkspaceProjectPathResolveResponse = {
  rootPath: string;
};

export type WorkspaceEnsureReadyResponse = {
  result: unknown;
};

export type SkillReadParams = {
  skillId: string;
};

export type SkillScope = "project" | "user" | "app" | "other";

export type SkillSource = "project" | "user" | "app" | "other";

export type SkillAuthority = "workspace" | "user" | "application" | "external";

export type SkillInterface = {
  displayName: string;
  executionMode: string;
  provider?: string;
  model?: string;
  argumentHint?: string;
};

export type SkillToolDependency = {
  type: string;
  value: string;
  required: boolean;
};

export type SkillDependencies = {
  tools: SkillToolDependency[];
};

export type SkillPolicy = {
  allowImplicitInvocation: boolean;
  whenToUse?: string;
};

export type SkillLocator = {
  directory: string;
  skillFilePath: string;
};

export type SkillSummary = {
  skillId: string;
  name: string;
  description: string;
  scope: SkillScope;
  source: SkillSource;
  authority: SkillAuthority;
  enabled: boolean;
  interface: SkillInterface;
  dependencies: SkillDependencies;
  policy: SkillPolicy;
  capabilities: string[];
  locator: SkillLocator;
};

export type SkillWorkflowStep = {
  id: string;
  name: string;
  dependencies: string[];
};

export type SkillDetail = {
  metadata: SkillSummary;
  markdownContent: string;
  workflowSteps: SkillWorkflowStep[];
};

export type SkillListResponse = {
  skills: SkillSummary[];
};

export type SkillReadResponse = {
  skill: SkillDetail;
};

export type SkillManagementListResponse = {
  skills: unknown[];
};

export type SkillManagementListParams = {
  app: string;
  refreshRemote?: boolean;
  scope?: "all" | "local" | "user";
};

export type SkillManagementInstallParams = {
  app: string;
  directory: string;
};

export type SkillManagementUninstallParams = {
  app: string;
  directory: string;
};

export type SkillRepositoryEntry = {
  owner: string;
  name: string;
  branch: string;
  enabled: boolean;
};

export type SkillRepositorySaveParams = {
  repo: SkillRepositoryEntry;
};

export type SkillRepositoryDeleteParams = {
  owner: string;
  name: string;
};

export type SkillLocalInspectParams = {
  app: string;
  directory: string;
};

export type SkillScaffoldCreateParams = {
  app: string;
  request: unknown;
};

export type SkillLocalImportParams = {
  app: string;
  sourcePath: string;
};

export type SkillRemoteInspectParams = {
  owner: string;
  name: string;
  branch: string;
  directory: string;
};

export type SkillManagementWriteResponse = {
  success: boolean;
};

export type SkillRepositoryListResponse = {
  repos: SkillRepositoryEntry[];
};

export type SkillInstalledDirectoriesListResponse = {
  directories: string[];
};

export type SkillLocalInspectResponse = {
  inspection: unknown;
};

export type SkillScaffoldCreateResponse = {
  inspection: unknown;
};

export type SkillLocalImportResponse = {
  directory: string;
};

export type SkillRemoteInspectResponse = {
  inspection: unknown;
};

export type SkillLocalDetailInspectParams = {
  app: string;
  directory: string;
};

export type SkillLocalRenameParams = {
  app: string;
  directory: string;
  newDirectory: string;
};

export type SkillPackageLocalReplaceParams = {
  app: string;
  directory: string;
  sourcePath: string;
};

export type SkillPackageLocalInspectParams = {
  app: string;
  sourcePath: string;
};

export type SkillPackageLocalInstallParams = {
  app: string;
  sourcePath: string;
  skillName?: string;
};

export type SkillPackageExportParams = {
  app: string;
  directory: string;
  targetPath: string;
};

export type SkillMarketplaceBundleFile = {
  path: string;
  content: string;
  encoding?: string;
  sha256?: string;
};

export type SkillMarketplaceInstallParams = {
  app: string;
  manifestVersion: string;
  name: string;
  aliases?: string[];
  version: string;
  contentHash?: string;
  fileCount?: number;
  files: SkillMarketplaceBundleFile[];
};

export type SkillDownloadInstallParams = {
  app: string;
  skillName: string;
  downloadUrl: string;
};

export type SkillPackageLocalInspectResponse = {
  directory: string;
  inspection: unknown;
  files: unknown[];
};

export type SkillLocalDetailInspectResponse = SkillPackageLocalInspectResponse;

export type SkillLocalRenameResponse = {
  directory: string;
};

export type SkillPackageLocalInstallResponse = {
  directory: string;
  inspection: unknown;
};

export type SkillPackageLocalReplaceResponse = SkillPackageLocalInstallResponse;

export type SkillMarketplaceInstallResponse = SkillPackageLocalInstallResponse;

export type SkillDownloadInstallResponse = SkillPackageLocalInstallResponse;

export type SkillPackageExportResponse = {
  directory: string;
  outputPath: string;
  fileCount: number;
  bytesWritten: number;
};

export type WorkspaceSkillBindingsListParams = {
  workspaceRoot: string;
  caller?: string;
  workbench?: boolean;
  browserAssist?: boolean;
};

export type WorkspaceSkillBindingsListResponse = {
  bindings: unknown;
};

export type LogEntry = {
  timestamp: string;
  level: string;
  message: string;
};

export type LogListResponse = {
  entries: LogEntry[];
};

export type LogPersistedTailParams = {
  lines?: number;
};

export type LogPersistedTailResponse = {
  entries: LogEntry[];
};

export type LogClearResponse = {
  cleared: boolean;
};

export type LogArtifactEntry = {
  fileName: string;
  path: string;
  sizeBytes: number;
  modifiedAt?: string;
  compressed: boolean;
};

export type LogStorageDiagnosticsResponse = {
  logDirectory?: string;
  currentLogPath?: string;
  currentLogExists: boolean;
  currentLogSizeBytes?: number;
  inMemoryLogCount: number;
  relatedLogFiles: LogArtifactEntry[];
  rawResponseFiles: LogArtifactEntry[];
};

export type SupportBundleTraceExportSelection = {
  sessionId: string;
  traceId: string;
};

export type SupportBundleExportParams = {
  includeTraceExport?: SupportBundleTraceExportSelection;
};

export type SupportBundleExportResponse = {
  bundlePath: string;
  outputDirectory: string;
  generatedAt: string;
  platform: string;
  includedSections: string[];
  omittedSections: string[];
};

export type DiagnosticsMetricConfig = {
  enabled: boolean;
  ttlSecs: number;
  maxEntries?: number;
  maxBodyBytes?: number;
  cacheableStatusCodes: number[];
  waitTimeoutMs?: number;
  headerName?: string;
};

export type DiagnosticsTelemetrySummary = {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  timeoutRequests: number;
  successRate: number;
  avgLatencyMs: number;
  minLatencyMs?: number | null;
  maxLatencyMs?: number | null;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalTokens: number;
};

export type DiagnosticsCapabilityRoutingMetricsSnapshot = {
  filterEvalTotal: number;
  filterExcludedTotal: number;
  filterExcludedToolsTotal: number;
  filterExcludedVisionTotal: number;
  filterExcludedContextTotal: number;
  providerFallbackTotal: number;
  modelFallbackTotal: number;
  allCandidatesExcludedTotal: number;
};

export type DiagnosticsResponseCacheDiagnostics = {
  config: DiagnosticsMetricConfig;
  stats: unknown;
  hitRatePercent: number;
};

export type DiagnosticsRequestDedupDiagnostics = {
  config: DiagnosticsMetricConfig;
  stats: unknown;
  replayRatePercent: number;
};

export type DiagnosticsIdempotencyDiagnostics = {
  config: DiagnosticsMetricConfig;
  stats: unknown;
  replayRatePercent: number;
};

export type ServerDiagnosticsResponse = {
  generatedAt: string;
  running: boolean;
  host: string;
  port: number;
  telemetrySummary: DiagnosticsTelemetrySummary;
  capabilityRouting: DiagnosticsCapabilityRoutingMetricsSnapshot;
  responseCache: DiagnosticsResponseCacheDiagnostics;
  requestDedup: DiagnosticsRequestDedupDiagnostics;
  idempotency: DiagnosticsIdempotencyDiagnostics;
};

export type WindowsStartupCheck = {
  key: string;
  status: "ok" | "warning" | "error" | string;
  message: string;
  detail?: string;
};

export type WindowsStartupDiagnosticsResponse = {
  platform: string;
  appDataDir?: string;
  legacyLimeDir?: string;
  dbPath?: string;
  webview2Version?: string;
  currentExe?: string;
  currentDir?: string;
  resourceDir?: string;
  homeDir?: string;
  shellEnv?: string;
  comspecEnv?: string;
  resolvedTerminalShell?: string;
  installationKindGuess?: string;
  checks: WindowsStartupCheck[];
  hasBlockingIssues: boolean;
  hasWarnings: boolean;
  summaryMessage?: string;
};

export type GatewayChannelStatusParams = {
  channel: string;
};

export type GatewayChannelStartParams = {
  channel: string;
  accountId?: string;
  pollTimeoutSecs?: number;
};

export type GatewayChannelStopParams = {
  channel: string;
  accountId?: string;
};

export type GatewayChannelStatusResponse = {
  channel: string;
  status: unknown;
};

export type ChannelProbeParams = {
  accountId?: string;
};

export type ChannelProbeResponse = {
  accountId?: string;
  ok: boolean;
  message: string;
  [key: string]: unknown;
};

export type WechatLoginStartParams = {
  baseUrl?: string;
  botType?: string;
  sessionKey?: string;
};

export type WechatLoginStartResponse = {
  sessionKey: string;
  qrcodeUrl: string;
  message: string;
};

export type WechatLoginWaitParams = {
  sessionKey: string;
  baseUrl?: string;
  botType?: string;
  timeoutMs?: number;
  accountName?: string;
};

export type WechatLoginWaitResponse = {
  connected: boolean;
  message: string;
  botToken?: string;
  accountId?: string;
  userId?: string;
  baseUrl?: string;
};

export type WechatConfiguredAccount = {
  account_id: string;
  enabled: boolean;
  name?: string;
  base_url?: string;
  cdn_base_url?: string;
  has_token: boolean;
  scanner_user_id?: string;
};

export type WechatChannelAccountListResponse = {
  accounts: WechatConfiguredAccount[];
};

export type WechatChannelAccountRemoveParams = {
  accountId: string;
  purgeData?: boolean;
};

export type WechatChannelAccountRemoveResponse = Record<string, never>;

export type WechatRuntimeModelSetParams = {
  providerId: string;
  modelId: string;
};

export type WechatRuntimeModelSetResponse = {
  runtimeModel: string;
};

export type GatewayTunnelCreateParams = {
  tunnelName?: string;
  dnsName?: string;
  persist?: boolean;
};

export type GatewayTunnelCreateResult = {
  ok: boolean;
  tunnelName: string;
  tunnelId?: string;
  credentialsFile?: string;
  dnsName?: string;
  publicBaseUrl?: string;
  message: string;
};

export type GatewayTunnelStatusResponse = {
  running: boolean;
  provider: string;
  mode: string;
  binary: string;
  localUrl: string;
  publicBaseUrl?: string;
  pid?: number;
  startedAt?: string;
  lastError?: string;
  lastExit?: string;
  commandPreview?: string;
  connectorActive?: boolean;
  connectorMessage?: string;
};

export type GatewayTunnelProbeResponse = {
  ok: boolean;
  provider: string;
  mode: string;
  binary: string;
  version?: string;
  configReady: boolean;
  message: string;
};

export type GatewayTunnelCloudflaredDetectResponse = {
  installed: boolean;
  binary: string;
  version?: string;
  platform: string;
  packageManager?: string;
  installSupported: boolean;
  installCommand?: string;
  requiresPrivilege: boolean;
  message: string;
};

export type GatewayTunnelCloudflaredInstallParams = {
  confirm?: boolean;
};

export type GatewayTunnelCloudflaredInstallResponse = {
  ok: boolean;
  attempted: boolean;
  platform: string;
  packageManager?: string;
  command?: string;
  exitCode?: number;
  installed: boolean;
  version?: string;
  stdout: string;
  stderr: string;
  message: string;
};

export type GatewayTunnelCreateResponse = {
  result: GatewayTunnelCreateResult;
  status: GatewayTunnelStatusResponse;
};

export type GatewayTunnelSyncWebhookUrlParams = {
  channel: string;
  accountId?: string;
  webhookPath?: string;
  persist?: boolean;
};

export type GatewayTunnelSyncWebhookUrlResponse = {
  channel: string;
  accountId?: string;
  webhookPath: string;
  publicBaseUrl: string;
  webhookUrl: string;
  persisted: boolean;
};

export type ImageStoryboardSlotInput = {
  prompt: string;
  slotId?: string;
  label?: string;
  shotType?: string;
};

export type MediaTaskArtifactImageCreateParams = {
  projectRootPath: string;
  prompt: string;
  title?: string;
  titleGenerationResult?: unknown;
  personaContext?: unknown;
  presentation?: unknown;
  tasteContext?: unknown;
  mode?: string;
  rawText?: string;
  layoutHint?: string;
  size?: string;
  aspectRatio?: string;
  count?: number;
  usage?: string;
  style?: string;
  providerId?: string;
  model?: string;
  executorMode?: string;
  outerModel?: string;
  sessionId?: string;
  threadId?: string;
  turnId?: string;
  projectId?: string;
  contentId?: string;
  entrySource?: string;
  modalityContractKey?: string;
  modality?: string;
  requiredCapabilities?: string[];
  routingSlot?: string;
  runtimeContract?: unknown;
  requestedTarget?: string;
  slotId?: string;
  anchorHint?: string;
  anchorSectionTitle?: string;
  anchorText?: string;
  targetOutputId?: string;
  targetOutputRefId?: string;
  referenceImages?: string[];
  storyboardSlots?: ImageStoryboardSlotInput[];
};

export type MediaTaskArtifactAudioCreateParams = {
  projectRootPath: string;
  sourceText: string;
  title?: string;
  rawText?: string;
  voice?: string;
  voiceStyle?: string;
  targetLanguage?: string;
  mimeType?: string;
  audioPath?: string;
  durationMs?: number;
  providerId?: string;
  model?: string;
  sessionId?: string;
  threadId?: string;
  turnId?: string;
  projectId?: string;
  contentId?: string;
  entrySource?: string;
  modalityContractKey?: string;
  modality?: string;
  requiredCapabilities?: string[];
  routingSlot?: string;
  runtimeContract?: unknown;
  requestedTarget?: string;
  outputPath?: string;
};

export type MediaTaskArtifactVideoCreateParams = {
  projectRootPath: string;
  prompt: string;
  title?: string;
  rawText?: string;
  aspectRatio?: string;
  resolution?: string;
  duration?: number;
  imageUrl?: string;
  endImageUrl?: string;
  seed?: number;
  generateAudio?: boolean;
  cameraFixed?: boolean;
  providerId?: string;
  model?: string;
  sessionId?: string;
  threadId?: string;
  turnId?: string;
  projectId?: string;
  contentId?: string;
  entrySource?: string;
  modalityContractKey?: string;
  modality?: string;
  requiredCapabilities?: string[];
  routingSlot?: string;
  runtimeContract?: unknown;
  requestedTarget?: string;
  outputPath?: string;
};

export type MediaTaskArtifactAudioCompleteParams = {
  projectRootPath: string;
  taskRef: string;
  audioPath: string;
  mimeType?: string;
  durationMs?: number;
  providerId?: string;
  model?: string;
};

export type MediaTaskArtifactLookupParams = {
  projectRootPath: string;
  taskRef: string;
};

export type MediaTaskArtifactListParams = {
  projectRootPath: string;
  status?: string;
  taskFamily?: string;
  taskType?: string;
  modalityContractKey?: string;
  routingOutcome?: string;
  limit?: number;
};

export type MediaTaskArtifactListFilters = {
  status?: string;
  task_family?: string;
  task_type?: string;
  modality_contract_key?: string;
  routing_outcome?: string;
  limit?: number;
};

export type MediaTaskArtifactResponse = {
  success: boolean;
  task_id: string;
  task_type: string;
  task_family: string;
  status: string;
  normalized_status: string;
  current_attempt_id?: string;
  path: string;
  absolute_path: string;
  artifact_path: string;
  absolute_artifact_path: string;
  reused_existing: boolean;
  idempotency_key?: string;
  record: unknown;
};

export type MediaTaskArtifactListResponse = {
  success: boolean;
  workspace_root: string;
  artifact_root: string;
  filters: MediaTaskArtifactListFilters;
  total: number;
  modality_runtime_contracts: unknown;
  tasks: MediaTaskArtifactResponse[];
};

export type GalleryMaterialMetadata = {
  materialId: string;
  imageCategory?: string;
  width?: number;
  height?: number;
  thumbnail?: string;
  colors: string[];
  iconStyle?: string;
  iconCategory?: string;
  colorSchemeJson?: string;
  mood?: string;
  layoutCategory?: string;
  elementCount?: number;
  preview?: string;
  fabricJson?: string;
  createdAt: number;
  updatedAt: number;
};

export type GalleryMaterial = {
  id: string;
  projectId: string;
  name: string;
  type: string;
  filePath?: string;
  fileSize?: number;
  mimeType?: string;
  content?: string;
  tags: string[];
  description?: string;
  createdAt: number;
  metadata?: GalleryMaterialMetadata;
};

export type GalleryMaterialMetadataCreateParams = {
  materialId: string;
  imageCategory?: string;
  width?: number;
  height?: number;
  thumbnail?: string;
  colors?: string[];
  iconStyle?: string;
  iconCategory?: string;
  colorSchemeJson?: string;
  mood?: string;
  layoutCategory?: string;
  elementCount?: number;
  preview?: string;
  fabricJson?: string;
};

export type GalleryMaterialLookupParams = {
  materialId: string;
};

export type GalleryMaterialMetadataUpdateParams = {
  materialId: string;
  metadata: GalleryMaterialMetadataCreateParams;
};

export type GalleryMaterialFilterParams = {
  projectId: string;
  category?: string | null;
  mood?: string | null;
};

export type GalleryMaterialResponse = {
  material?: GalleryMaterial | null;
};

export type GalleryMaterialMetadataResponse = {
  metadata?: GalleryMaterialMetadata | null;
};

export type GalleryMaterialListResponse = {
  materials: GalleryMaterial[];
};

export type GalleryMaterialDeleteResponse = Record<string, never>;

export type ProjectMaterial = {
  id: string;
  projectId: string;
  name: string;
  type: string;
  filePath?: string;
  fileSize?: number;
  mimeType?: string;
  content?: string;
  tags: string[];
  description?: string;
  createdAt: number;
};

export type ProjectMaterialFilter = {
  type?: string;
  tags?: string[];
  searchQuery?: string;
};

export type ProjectMaterialListParams = {
  projectId: string;
  filter?: ProjectMaterialFilter | null;
};

export type ProjectMaterialLookupParams = {
  id: string;
};

export type ProjectMaterialUploadParams = {
  projectId: string;
  name: string;
  type: string;
  filePath?: string;
  content?: string;
  tags?: string[];
  description?: string;
};

export type ProjectMaterialImportFromUrlParams = {
  projectId: string;
  name: string;
  type: string;
  url: string;
  tags?: string[];
  description?: string;
};

export type ProjectMaterialUpdate = {
  name?: string;
  tags?: string[];
  description?: string;
};

export type ProjectMaterialUpdateParams = {
  id: string;
  update: ProjectMaterialUpdate;
};

export type ProjectMaterialListResponse = {
  materials: ProjectMaterial[];
};

export type ProjectMaterialResponse = {
  material?: ProjectMaterial | null;
};

export type ProjectMaterialCountResponse = {
  count: number;
};

export type ProjectMaterialContentResponse = {
  content: string;
};

export type ProjectMaterialDeleteResponse = Record<string, never>;

export type VoiceAsrProviderType =
  | "whisper_local"
  | "sense_voice_local"
  | "xunfei"
  | "baidu"
  | "openai";

export type VoiceAsrWhisperModelSize = "tiny" | "base" | "small" | "medium";

export type VoiceAsrWhisperLocalConfig = {
  model: VoiceAsrWhisperModelSize;
  model_path?: string;
};

export type VoiceAsrSenseVoiceLocalConfig = {
  model_id: string;
  model_dir?: string;
  use_itn: boolean;
  num_threads: number;
  vad_model_id?: string;
};

export type VoiceAsrXunfeiConfig = {
  app_id: string;
  api_key: string;
  api_secret: string;
};

export type VoiceAsrBaiduConfig = {
  api_key: string;
  secret_key: string;
};

export type VoiceAsrOpenAiConfig = {
  api_key: string;
  base_url?: string;
  proxy_url?: string;
};

export type VoiceAsrCredential = {
  id: string;
  provider: VoiceAsrProviderType;
  name?: string;
  is_default: boolean;
  disabled: boolean;
  language: string;
  whisper_config?: VoiceAsrWhisperLocalConfig;
  sensevoice_config?: VoiceAsrSenseVoiceLocalConfig;
  xunfei_config?: VoiceAsrXunfeiConfig;
  baidu_config?: VoiceAsrBaiduConfig;
  openai_config?: VoiceAsrOpenAiConfig;
};

export type VoiceAsrCredentialCreateParams = Omit<VoiceAsrCredential, "id">;

export type VoiceAsrCredentialUpdateParams = {
  credential: VoiceAsrCredential;
};

export type VoiceAsrCredentialIdParams = {
  id: string;
};

export type VoiceAsrCredentialListResponse = {
  credentials: VoiceAsrCredential[];
};

export type VoiceAsrCredentialWriteResponse = {
  credential: VoiceAsrCredential;
};

export type VoiceAsrCredentialMutationResponse = Record<string, never>;

export type VoiceAsrCredentialTestResponse = {
  success: boolean;
  message: string;
};

export type VoiceInstruction = {
  id: string;
  name: string;
  description?: string;
  prompt: string;
  shortcut?: string;
  is_preset: boolean;
  icon?: string;
};

export type VoiceInstructionSaveParams = {
  instruction: VoiceInstruction;
};

export type VoiceInstructionIdParams = {
  id: string;
};

export type VoiceInstructionListResponse = {
  instructions: VoiceInstruction[];
};

export type VoiceInstructionMutationResponse = Record<string, never>;

export type VoiceModelDefaultSetParams = {
  model_id: string;
  install_dir: string;
};

export type VoiceModelDefaultSetResponse = {
  credential: VoiceAsrCredential;
};

export type VoiceModelTestTranscribeFileParams = {
  model_id: string;
  install_dir: string;
  file_path: string;
};

export type VoiceModelTestTranscribeFileResponse = {
  text: string;
  duration_secs: number;
  sample_rate: number;
  language?: string;
};

export type WorkspaceRegisteredSkillsListParams = {
  workspaceRoot: string;
};

export type WorkspaceRegisteredSkillsListResponse = {
  skills: unknown[];
};

export type PluginLocalPackageInspectParams = {
  appDir: string;
};

export type PluginLocalPackageInspectResponse = {
  sourceKind: "local_folder" | string;
  sourceUri: string;
  appDir: string;
  manifestSource: "plugin_json" | string;
  pluginManifest: unknown;
  manifest: unknown;
  manifestHash: string;
  packageHash: string;
  inspectedAt: string;
};

export type PluginCloudReleaseDescriptor = {
  sourceUri: string;
  appId: string;
  version: string;
  releaseId?: string;
  tenantId?: string;
  tenantEnablementRef?: string;
  channel?: string;
  packageUrl: string;
  packageHash: string;
  manifestHash: string;
  signatureRef?: string;
  loadedAt: string;
};

export type PluginFetchCloudPackageParams = {
  descriptor: PluginCloudReleaseDescriptor;
};

export type PluginPackageIdentity = {
  sourceKind: string;
  sourceUri: string;
  appId: string;
  appVersion: string;
  packageHash: string;
  manifestHash: string;
  loadedAt: string;
  releaseId?: string;
  tenantId?: string;
  tenantEnablementRef?: string;
  channel?: string;
  signatureRef?: string;
};

export type PluginPackageCacheEntry = {
  appId: string;
  identity: PluginPackageIdentity;
  manifestSnapshot: unknown;
  packageHash: string;
  manifestHash: string;
  cachePath: string;
  cachedAt: string;
};

export type PluginInstalledSaveParams = {
  state: unknown;
};

export type PluginInstalledDisabledSetParams = {
  appId: string;
  disabled: boolean;
  updatedAt?: string;
};

export type PluginInstalledListResponse = {
  states: unknown[];
  issues: unknown[];
};

export type PluginUninstallRehearsalParams = {
  appId: string;
  mode: "keep-data" | "delete-data" | string;
};

export type PluginUninstallRehearsalTarget = {
  kind: string;
  value: string;
  safeToDelete: boolean;
  action: "delete" | "retain" | "blocked" | string;
  reason: string;
};

export type PluginUninstallRehearsalResponse = {
  appId: string;
  packageHash?: string;
  mode: "keep-data" | "delete-data" | string;
  generatedAt: string;
  deletedTargetCount: number;
  retainedTargetCount: number;
  targets: PluginUninstallRehearsalTarget[];
  warnings: string[];
};

export type PluginUninstallParams = {
  appId: string;
  mode: "keep-data" | "delete-data" | string;
  confirmationPhrase?: string;
};

export type PluginDeleteDataTargetEvidence = {
  kind: string;
  value: string;
  action: string;
  reason: string;
  status: string;
  blockerCodes: string[];
  error?: string | null;
};

export type PluginDeleteDataExecutionEvidence = {
  status: string;
  generatedAt: string;
  dataRoot: string;
  removedTargets: PluginDeleteDataTargetEvidence[];
  missingTargets: PluginDeleteDataTargetEvidence[];
  retainedTargets: PluginDeleteDataTargetEvidence[];
  blockedTargets: PluginDeleteDataTargetEvidence[];
  failedTarget?: PluginDeleteDataTargetEvidence | null;
  blockerCodes: string[];
  postDeleteResidualAudit?: {
    status: string;
    checkedAt: string;
    checkedTargetCount: number;
    remainingTargetCount: number;
    remainingTargets: PluginDeleteDataTargetEvidence[];
    failedTarget?: PluginDeleteDataTargetEvidence | null;
  };
};

export type PluginUninstallResponse = {
  status: string;
  rehearsal: PluginUninstallRehearsalResponse;
  list: PluginInstalledListResponse;
  removedTargetCount: number;
  missingTargetCount: number;
  blockerCodes: string[];
  deleteEvidence?: PluginDeleteDataExecutionEvidence | null;
};

export type PluginShellPrepareParams = {
  descriptor: unknown;
};

export type PluginShellPackageMount = {
  kind: string;
  path: string;
  readOnly: boolean;
  packageHash: string;
  manifestHash: string;
};

export type PluginShellPrepareResponse = {
  appId?: string;
  status: string;
  installMode?: string;
  shellKind?: string;
  descriptorVersion?: number;
  devShell: boolean;
  blockerCodes: string[];
  message?: string;
  packageMount?: PluginShellPackageMount;
  entryKey?: string;
  windowTitle?: string;
  preparedAt: string;
};

export type PluginUiRuntimeStartParams = {
  appId: string;
  entryKey?: string;
};

export type PluginUiRuntimeStatusParams = {
  appId: string;
};

export type PluginUiRuntimeStopParams = {
  appId: string;
};

export type PluginTaskRuntimeContract = {
  enabled: boolean;
  packageRootPath?: string | null;
  workerEntrypoint?: string | null;
  taskKinds?: string[];
  outputArtifactKind?: string | null;
  contractPath?: string | null;
  sampleRequestPath?: string | null;
  blockers?: string[];
  followUps?: string[];
  directProviderAccess: boolean;
  directFilesystemAccess: boolean;
};

export type PluginUiRuntimeStatusResponse = {
  appId: string;
  status: "starting" | "running" | "stopped" | "failed" | string;
  baseUrl?: string;
  entryUrl?: string;
  port?: number;
  pid?: number;
  message?: string;
  entryKey?: string;
  route?: string;
  taskRuntime?: PluginTaskRuntimeContract | null;
};

export type KnowledgeListPacksParams = {
  workingDir: string;
  includeArchived?: boolean;
};

export type KnowledgeListPacksResponse = {
  workingDir: string;
  rootPath: string;
  packs: unknown[];
};

export type KnowledgeReadPackParams = {
  workingDir: string;
  name: string;
};

export type KnowledgeReadPackResponse = {
  pack: unknown;
};

export type KnowledgeImportSourceParams = {
  workingDir: string;
  packName: string;
  description?: string;
  packType?: string;
  language?: string;
  sourceFileName?: string;
  sourceText?: string;
};

export type KnowledgeImportSourceResponse = {
  pack: unknown;
  source: unknown;
};

export type KnowledgeCompilePackParams = {
  workingDir: string;
  name: string;
  builderRuntime?: unknown;
};

export type KnowledgeCompilePackResponse = {
  pack: unknown;
  selectedSourceCount: number;
  compiledView: unknown;
  run: unknown;
  warnings: string[];
};

export type KnowledgeSetDefaultPackParams = {
  workingDir: string;
  name: string;
};

export type KnowledgeSetDefaultPackResponse = {
  defaultPackName: string;
  defaultMarkerPath: string;
};

export type KnowledgeUpdatePackStatusParams = {
  workingDir: string;
  name: string;
  status: string;
};

export type KnowledgeUpdatePackStatusResponse = {
  pack: unknown;
  previousStatus: string;
  clearedDefault: boolean;
};

export type KnowledgeResolveContextPackParams = {
  name: string;
  activation?: string;
};

export type KnowledgeResolveContextParams = {
  workingDir: string;
  name: string;
  packs?: KnowledgeResolveContextPackParams[];
  task?: string;
  maxChars?: number;
  activation?: string;
  writeRun?: boolean;
  runReason?: string;
};

export type KnowledgeContextResolutionResponse = {
  packName: string;
  status: string;
  grounding?: string;
  selectedViews: unknown[];
  selectedFiles: string[];
  sourceAnchors: string[];
  warnings: unknown[];
  missing: string[];
  tokenEstimate: number;
  fencedContext: string;
  runId?: string;
  runPath?: string;
};

export type KnowledgeValidateContextRunParams = {
  workingDir: string;
  name: string;
  runPath: string;
};

export type KnowledgeValidateContextRunResponse = {
  valid: boolean;
  runId?: string;
  status?: string;
  errors: string[];
  warnings: string[];
};

export type AutomationJobListResponse = {
  jobs: unknown[];
};

export type AutomationSchedulerConfigReadResponse = {
  config: unknown;
};

export type AutomationSchedulerConfigUpdateParams = {
  config: unknown;
};

export type AutomationSchedulerConfigUpdateResponse = {
  config: unknown;
};

export type AutomationSchedulerStatusResponse = {
  status: unknown;
};

export type AutomationJobIdParams = {
  id: string;
};

export type AutomationJobReadResponse = {
  job?: unknown;
};

export type AutomationJobCreateParams = {
  request: unknown;
};

export type AutomationJobWriteResponse = {
  job: unknown;
};

export type AutomationJobUpdateParams = {
  id: string;
  request: unknown;
};

export type AutomationJobDeleteResponse = {
  deleted: boolean;
};

export type AutomationJobRunNowResponse = {
  result: unknown;
};

export type AutomationJobHealthParams = {
  query?: unknown;
};

export type AutomationJobHealthResponse = {
  health: unknown;
};

export type AutomationJobRunHistoryParams = {
  id: string;
  limit?: number;
};

export type AutomationJobRunHistoryResponse = {
  runs: unknown[];
};

export type AutomationScheduleParams = {
  schedule: unknown;
};

export type AutomationSchedulePreviewResponse = {
  nextRunAt?: string;
};

export type AutomationScheduleValidateResponse = {
  valid: boolean;
  error?: string;
};

export type McpServerListResponse = {
  servers: unknown[];
};

export type McpServerStatusListResponse = {
  servers: unknown[];
};

export type McpServerCreateParams = {
  server: unknown;
};

export type McpServerUpdateParams = {
  server: unknown;
};

export type McpServerDeleteParams = {
  id: string;
};

export type McpServerEnabledSetParams = {
  id: string;
  appType: string;
  enabled: boolean;
};

export type McpServerImportFromAppParams = {
  appType: string;
};

export type McpServerImportFromAppResponse = {
  importedCount: number;
  servers: unknown[];
};

export type McpServerOauthLoginParams = {
  name: string;
  scopes?: string[];
  timeoutSecs?: number;
};

export type McpServerOauthLoginResponse = {
  authorizationUrl: string;
  state: string;
};

export type McpServerStartParams = {
  name: string;
};

export type McpServerStopParams = {
  name: string;
};

export type McpServerLifecycleResponse = Record<string, never>;

export type McpToolListResponse = {
  tools: unknown[];
};

export type McpToolListForContextParams = {
  caller?: string;
  includeDeferred?: boolean;
};

export type McpToolSearchParams = {
  query: string;
  caller?: string;
  limit?: number;
};

export type McpToolCallParams = {
  toolName: string;
  arguments: unknown;
};

export type McpToolCallWithCallerParams = McpToolCallParams & {
  caller?: string;
};

export type McpContent =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mime_type: string }
  | { type: "resource"; uri: string; text?: string; blob?: string };

export type McpToolCallResponse = {
  content: McpContent[];
  structuredContent?: unknown;
  is_error: boolean;
};

export type McpPromptListResponse = {
  prompts: unknown[];
};

export type McpPromptGetParams = {
  server: string;
  name: string;
  arguments?: Record<string, unknown>;
};

export type McpPromptMessage = {
  role: string;
  content: McpContent;
};

export type McpPromptGetResponse = {
  description?: string;
  messages: McpPromptMessage[];
};

export type McpResourceListResponse = {
  resources: unknown[];
  resourceTemplates?: unknown[];
};

export type McpResourceReadParams = {
  server: string;
  uri: string;
};

export type McpResourceSubscribeParams = {
  server: string;
  uri: string;
};

export type McpResourceUnsubscribeParams = {
  server: string;
  uri: string;
};

export type McpResourceSubscriptionResponse = Record<string, never>;

export type McpResourceReadResponse = {
  uri: string;
  mime_type?: string;
  text?: string;
  blob?: string;
};

export type ProjectMemoryReadParams = {
  projectId: string;
};

export type ProjectMemoryReadResponse = {
  memory: unknown;
};

export type UsageStatsRangeParams = {
  timeRange: string;
};

export type UsageStatsSummary = {
  totalConversations: number;
  totalMessages: number;
  totalTokens: number;
  totalTimeMinutes: number;
  monthlyConversations: number;
  monthlyMessages: number;
  monthlyTokens: number;
  todayConversations: number;
  todayMessages: number;
  todayTokens: number;
};

export type UsageStatsReadResponse = {
  stats: UsageStatsSummary;
};

export type UsageStatsModelUsage = {
  model: string;
  conversations: number;
  tokens: number;
  percentage: number;
};

export type UsageStatsModelRankingListResponse = {
  ranking: UsageStatsModelUsage[];
};

export type UsageStatsDailyUsage = {
  date: string;
  conversations: number;
  tokens: number;
};

export type UsageStatsDailyTrendsListResponse = {
  trends: UsageStatsDailyUsage[];
};

export type AgentSessionReadResponse = {
  session: AgentSession;
  turns: AgentTurn[];
  detail?: unknown;
};

export type AgentSessionTurnStartResponse = GeneratedTurnStartResponse;

/** @deprecated 使用 TurnInterruptResponse。 */
export type AgentSessionTurnCancelResponse = GeneratedTurnInterruptResponse;
export type AgentSessionActionRespondResponse = Record<string, never>;

export type AgentSessionEventParams = {
  event: AgentEvent;
};

export type AgentSessionEventNotification = JsonRpcNotification & {
  method: typeof METHOD_AGENT_SESSION_EVENT;
  params: AgentSessionEventParams;
};

export const AGENT_SESSION_MEDIA_READ_CHUNK_EVENT_TYPE = "media.read.chunk";
export const AGENT_SESSION_MEDIA_READ_COMPLETED_EVENT_TYPE =
  "media.read.completed";

export type AgentSessionMediaReadChunk = Omit<
  GeneratedAgentSessionMediaReadResponse,
  "sha256"
> & {
  sha256?: string | null;
};

export type AgentSessionMediaReadChunkEventPayload = {
  streamId: string;
  chunkIndex: number;
  done: false;
  chunk: AgentSessionMediaReadChunk;
};

export type AgentSessionMediaReadCompletedEventPayload = {
  streamId: string;
  chunkCount: number;
  done: true;
  media: Omit<GeneratedAgentSessionMediaReadResponse, "contentBase64">;
};

export type AgentSessionMediaReadEventNotification =
  AgentSessionEventNotification & {
    params: AgentSessionEventParams & {
      event: AgentEvent & {
        type:
          | typeof AGENT_SESSION_MEDIA_READ_CHUNK_EVENT_TYPE
          | typeof AGENT_SESSION_MEDIA_READ_COMPLETED_EVENT_TYPE;
        payload:
          | AgentSessionMediaReadChunkEventPayload
          | AgentSessionMediaReadCompletedEventPayload;
      };
    };
  };

export type WorkspaceRightSurfacePendingChangedParams =
  GeneratedWorkspaceRightSurfacePendingChangedParams;

export type WorkspaceRightSurfacePendingChangedNotification =
  JsonRpcNotification & {
    method: typeof METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED;
    params: WorkspaceRightSurfacePendingChangedParams;
  };

export type ConnectDeepLinkResolveParams = {
  url: string;
};

export type ConnectPayload = {
  relay: string;
  key: string;
  name?: string;
  refCode?: string;
};

export type ConnectDeepLinkResolveResponse = {
  payload: ConnectPayload;
  relayInfo?: unknown;
  isVerified: boolean;
};

export type ConnectOpenDeepLinkResolveParams = {
  url: string;
};

export type OpenDeepLinkPayload = {
  kind: string;
  slug: string;
  source?: string;
  version?: string;
  action?: string;
};

export type ConnectOpenDeepLinkResolveResponse = {
  payload: OpenDeepLinkPayload;
};

export type ConnectRelayApiKeySaveParams = {
  relayId: string;
  apiKey: string;
  name?: string;
};

export type ConnectRelayApiKeySaveResponse = {
  providerId: string;
  keyId: string;
  providerName: string;
  isNewProvider: boolean;
};

export type ConnectCallbackStatus = "success" | "cancelled" | "error";

export type ConnectCallbackSendParams = {
  relayId: string;
  apiKey: string;
  status: ConnectCallbackStatus;
  refCode?: string;
  errorCode?: string;
  errorMessage?: string;
};

export type ConnectCallbackSendResponse = {
  delivered: boolean;
};

export type ConversationImportSourceClient = "codex";

export type ConversationImportSourceStatus = "ready" | "missing";

export type ConversationImportThreadStatus =
  | "not_imported"
  | "importing"
  | "imported"
  | "conflict";

export type ConversationImportJobStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed";

export type ConversationImportJobPhase =
  | "queued"
  | "reading_source"
  | "building_history"
  | "persisting_history"
  | "finalizing"
  | "completed"
  | "failed";

export type ConversationImportSourceScanParams = {
  sourceClient?: ConversationImportSourceClient;
  sourceRoot?: string;
  projectPath?: string;
  query?: string;
  includeArchived?: boolean;
  limit?: number;
  cursor?: string;
};

export type ConversationImportThreadPreviewParams = {
  sourceClient?: ConversationImportSourceClient;
  sourceRoot?: string;
  sourceThreadId?: string;
  sourcePath?: string;
  limit?: number;
};

export type ConversationImportThreadCommitParams = {
  sourceClient?: ConversationImportSourceClient;
  sourceRoot?: string;
  sourceThreadId?: string;
  sourcePath?: string;
  workspaceId?: string;
  appId?: string;
  confirmed: boolean;
  replaceExisting?: boolean;
};

export type ConversationImportJobReadParams = {
  jobId: string;
};

export type ConversationImportSourceSummary = {
  sourceClient: ConversationImportSourceClient;
  status: ConversationImportSourceStatus;
  sourceRoot?: string;
  readable: boolean;
  threadCount: number;
  sourceHomeExists: boolean;
  stateDbReadable: boolean;
  rolloutFileCount: number;
  indexedAt?: string;
  statePath?: string;
  message?: string;
};

export type ImportedThreadSummary = {
  sourceClient: ConversationImportSourceClient;
  sourceThreadId: string;
  title?: string;
  createdAt?: string;
  updatedAt?: string;
  cwd?: string;
  source?: string;
  modelProvider?: string;
  archived: boolean;
  sourcePath?: string;
  importJobId?: string;
  importStatus: ConversationImportThreadStatus;
  metadata?: unknown;
};

export type ConversationImportSourceScanResponse = {
  source: ConversationImportSourceSummary;
  threads: ImportedThreadSummary[];
  nextCursor?: string;
};

export type ConversationImportSourceProvenance = {
  sourceClient: ConversationImportSourceClient;
  sourceThreadId?: string;
  sourcePath?: string;
  sourceEventType?: string;
  sourceEventSeq?: number;
  sourcePayloadType?: string;
  sourceCallId?: string;
  sourceRole?: string;
  sourceChannel?: string;
};

export type ConversationImportFidelitySummary = {
  messages: number;
  reasoning: number;
  tools: number;
  commands: number;
  patches: number;
  approvals: number;
  mcp: number;
  webSearch: number;
  attachments: number;
  unsupported: number;
  provenanceOnly: number;
  budgetDropped: number;
};

export type ConversationImportPreviewMessage = {
  role: string;
  text: string;
  attachments: AgentAttachment[];
  truncated: boolean;
  omittedBytes: number;
  timestamp?: string;
  sourceType?: string;
  provenance?: ConversationImportSourceProvenance;
};

export type ConversationImportPreviewEvent = {
  kind: string;
  timestamp?: string;
  label?: string;
  provenance?: ConversationImportSourceProvenance;
};

export type ConversationImportPreviewDryRun = {
  willCreateSession: boolean;
  willAppendToExistingSession: boolean;
  willImportMessages: number;
  willImportTurns: number;
  willImportTimelineItems: number;
  willImportAttachments: number;
  unsupportedItems: number;
};

export type ConversationImportPreviewSummary = {
  lineCount: number;
  messageCount: number;
  rolloutEventItems: number;
  unsupportedCount: number;
  dryRun: ConversationImportPreviewDryRun;
  fidelity: ConversationImportFidelitySummary;
  truncated: boolean;
  warnings: string[];
};

export type ConversationImportThreadPreviewResponse = {
  source: ConversationImportSourceSummary;
  thread: ImportedThreadSummary;
  summary: ConversationImportPreviewSummary;
  messages: ConversationImportPreviewMessage[];
  events: ConversationImportPreviewEvent[];
};

export type ConversationImportThreadCommitResponse = {
  session: AgentSession;
  thread: ImportedThreadSummary;
  summary: ConversationImportPreviewSummary;
  importedMessages: number;
  importedTurns: number;
  canContinue: boolean;
  warnings: string[];
};

export type ConversationImportJobProgress = {
  phase: ConversationImportJobPhase;
  completedItems: number;
  totalItems: number;
  completedTurns: number;
  totalTurns: number;
};

export type ConversationImportJob = {
  jobId: string;
  sourceClient: ConversationImportSourceClient;
  sourceThreadId?: string;
  status: ConversationImportJobStatus;
  progress: ConversationImportJobProgress;
  result?: ConversationImportThreadCommitResponse;
  error?: string;
  createdAt: string;
  updatedAt: string;
};

export type ConversationImportThreadCommitStartResponse = {
  job: ConversationImportJob;
};

export type ConversationImportJobReadResponse = {
  job: ConversationImportJob;
};

export type ProtocolSchemaGroup = "jsonrpc" | "v0" | "v2";

export type AppServerProtocolSchemaManifest = {
  protocolVersion: string;
  methods: AppServerMethodSpec[];
  requestSerializationScopes: AppServerRequestSerializationScopeSpec[];
  jsonRpc: {
    version: string;
    sendsJsonRpcVersionField: boolean;
    envelopes: string[];
  };
  schemas: Record<ProtocolSchemaGroup, string[]>;
};

export type ProtocolSchemaFile = {
  group: ProtocolSchemaGroup;
  typeName: string;
  path: string;
};

export function request(
  id: RequestId,
  method: string,
  params?: unknown,
): JsonRpcRequest {
  return compactParams({ id, method, params });
}

export function notification(
  method: string,
  params?: unknown,
): JsonRpcNotification {
  return compactParams({ method, params });
}

export function response<T = RpcResult>(
  id: RequestId,
  result: T,
): JsonRpcResponse<T> {
  return { id, result };
}

export function errorResponse(
  id: RequestId,
  error: JsonRpcError,
): JsonRpcErrorResponse {
  return { id, error };
}

export function cancelRequest(id: RequestId): JsonRpcNotification {
  return notification(METHOD_CANCEL_REQUEST, { id });
}

export function isAppServerRequestMethod(method: string): boolean {
  return APP_SERVER_METHODS.some(
    (spec) => spec.kind === "request" && spec.method === method,
  );
}

export function isAppServerNotificationMethod(method: string): boolean {
  return APP_SERVER_METHODS.some(
    (spec) => spec.kind === "notification" && spec.method === method,
  );
}

export function isAppServerServerRequestMethod(method: string): boolean {
  return APP_SERVER_METHODS.some(
    (spec) => spec.kind === "serverRequest" && spec.method === method,
  );
}

export function getAppServerRequestSerializationScope(
  method: string,
): AppServerRequestSerializationScope | undefined {
  return APP_SERVER_REQUEST_SERIALIZATION_SCOPES.find(
    (spec) => spec.method === method,
  )?.scope;
}

export function encodeMessage(message: JsonRpcMessage): string {
  return `${JSON.stringify(message)}\n`;
}

export function decodeMessage(line: string): JsonRpcMessage {
  const trimmed = line.trim();
  if (trimmed.length === 0) {
    throw new Error("empty JSON-RPC line");
  }
  return JSON.parse(trimmed) as JsonRpcMessage;
}

export function decodeMessages(lines: string[]): JsonRpcMessage[] {
  return lines.map(decodeMessage);
}

export function isJsonRpcNotification(
  message: JsonRpcMessage,
): message is JsonRpcNotification {
  return "method" in message && !("id" in message);
}

export function isJsonRpcRequest(
  message: JsonRpcMessage,
): message is JsonRpcRequest {
  return "method" in message && "id" in message;
}

export function isAgentSessionTurnStartRequest(
  message: JsonRpcMessage,
): message is AgentSessionTurnStartRequest {
  return Boolean(agentSessionTurnStartRequest(message));
}

export function agentSessionTurnStartRequest(
  message: JsonRpcMessage,
): AgentSessionTurnStartRequest | undefined {
  if (!isJsonRpcRequest(message) || message.method !== METHOD_TURN_START) {
    return undefined;
  }
  const params = message.params as
    | Partial<AgentSessionTurnStartParams>
    | undefined;
  if (
    !params ||
    typeof params.threadId !== "string" ||
    !Array.isArray(params.input)
  ) {
    return undefined;
  }
  return message as AgentSessionTurnStartRequest;
}

export function isAgentSessionEventNotification(
  message: JsonRpcMessage,
): message is AgentSessionEventNotification {
  return Boolean(agentSessionEventNotification(message));
}

export function agentSessionEventNotification(
  message: JsonRpcMessage,
): AgentSessionEventNotification | undefined {
  if (
    !isJsonRpcNotification(message) ||
    message.method !== METHOD_AGENT_SESSION_EVENT
  ) {
    return undefined;
  }
  const params = message.params as Partial<AgentSessionEventParams> | undefined;
  if (
    !params ||
    !params.event ||
    !isAgentSessionRawSideChannelType(params.event.type)
  ) {
    return undefined;
  }
  return message as AgentSessionEventNotification;
}

function isAgentSessionRawSideChannelType(type: string): boolean {
  return (
    type.startsWith("action.") ||
    type.startsWith("approval.") ||
    type.startsWith("provider.") ||
    type.startsWith("image_task.") ||
    type.startsWith("image_task_") ||
    type.startsWith("media.") ||
    type.startsWith("runtime.")
  );
}

export function agentSessionMediaReadEventNotification(
  message: JsonRpcMessage,
): AgentSessionMediaReadEventNotification | undefined {
  const notification = agentSessionEventNotification(message);
  if (!notification) {
    return undefined;
  }
  const eventType = notification.params.event.type;
  if (
    eventType !== AGENT_SESSION_MEDIA_READ_CHUNK_EVENT_TYPE &&
    eventType !== AGENT_SESSION_MEDIA_READ_COMPLETED_EVENT_TYPE
  ) {
    return undefined;
  }
  const payload = notification.params.event.payload as
    | Partial<AgentSessionMediaReadChunkEventPayload>
    | Partial<AgentSessionMediaReadCompletedEventPayload>
    | undefined;
  if (!payload || typeof payload.streamId !== "string") {
    return undefined;
  }
  if (
    eventType === AGENT_SESSION_MEDIA_READ_CHUNK_EVENT_TYPE &&
    payload.done === false &&
    typeof (payload as Partial<AgentSessionMediaReadChunkEventPayload>)
      .chunk === "object"
  ) {
    return notification as AgentSessionMediaReadEventNotification;
  }
  if (
    eventType === AGENT_SESSION_MEDIA_READ_COMPLETED_EVENT_TYPE &&
    payload.done === true &&
    typeof (payload as Partial<AgentSessionMediaReadCompletedEventPayload>)
      .media === "object"
  ) {
    return notification as AgentSessionMediaReadEventNotification;
  }
  return undefined;
}

export function isWorkspaceRightSurfacePendingChangedNotification(
  message: JsonRpcMessage,
): message is WorkspaceRightSurfacePendingChangedNotification {
  return Boolean(workspaceRightSurfacePendingChangedNotification(message));
}

export function workspaceRightSurfacePendingChangedNotification(
  message: JsonRpcMessage,
): WorkspaceRightSurfacePendingChangedNotification | undefined {
  if (
    !isJsonRpcNotification(message) ||
    message.method !== METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED
  ) {
    return undefined;
  }
  const params = message.params as
    | Partial<WorkspaceRightSurfacePendingChangedParams>
    | undefined;
  if (!params || typeof params.changeType !== "string") {
    return undefined;
  }
  return message as WorkspaceRightSurfacePendingChangedNotification;
}

export function isJsonRpcResponse<T = RpcResult>(
  message: JsonRpcMessage,
): message is JsonRpcResponse<T> {
  return "id" in message && "result" in message;
}

export function isJsonRpcErrorResponse(
  message: JsonRpcMessage,
): message is JsonRpcErrorResponse {
  return "id" in message && "error" in message;
}

function compactParams<T extends { params?: unknown }>(value: T): T {
  if (value.params === undefined) {
    const { params: _params, ...rest } = value;
    return rest as T;
  }
  return value;
}
