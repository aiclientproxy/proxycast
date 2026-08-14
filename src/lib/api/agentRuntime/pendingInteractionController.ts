import {
  METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
  METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
  METHOD_ITEM_PERMISSIONS_REQUEST_APPROVAL,
  METHOD_ITEM_TOOL_REQUEST_USER_INPUT,
  METHOD_MCP_SERVER_ELICITATION_REQUEST,
  type CommandExecutionApprovalDecision,
  type CommandExecutionRequestApprovalParams,
  type CommandExecutionRequestApprovalResponse,
  type FileChangeRequestApprovalParams,
  type FileChangeRequestApprovalResponse,
  type McpServerElicitationRequestParams,
  type McpServerElicitationRequestResponse,
  type PermissionsRequestApprovalParams,
  type PermissionsRequestApprovalResponse,
  type RequestPermissionProfile,
  type ToolRequestUserInputParams,
  type ToolRequestUserInputResponse,
} from "@limecloud/app-server-client";
import type {
  ActionRequired,
  ApprovalDecision,
  ConfirmResponse,
  Question,
} from "../agentActionTypes";
import { normalizeActionArguments } from "../agentActionArguments";
import {
  getDefaultAppServerServerRequestDispatcher,
  type AppServerServerRequestDispatcher,
} from "../appServerServerRequest";
import {
  type McpElicitationFormContent,
  type McpElicitationFormIssue,
  validateMcpElicitationFormContent,
} from "../mcpServerElicitation";
import type { PendingInteractionProjection } from "./conversationProjection";

type PendingInteractionDispatcher = Pick<
  AppServerServerRequestDispatcher,
  "register"
>;

type PendingInteractionBase = Omit<
  PendingInteractionProjection,
  "kind" | "payload"
>;

export interface PendingApprovalInteraction extends PendingInteractionBase {
  kind: "approval";
  payload: {
    request: ActionRequired;
  };
}

export interface PendingUserInputInteraction extends PendingInteractionBase {
  kind: "request_user_input";
  payload: {
    request: ActionRequired;
  };
}

export interface PendingMcpElicitationInteraction extends PendingInteractionBase {
  kind: "mcp_elicitation";
  payload: {
    message: string;
    requestedSchema: Record<string, unknown>;
    serverName: string;
  };
}

export interface PendingPermissionsApprovalInteraction extends PendingInteractionBase {
  kind: "permissions_approval";
  payload: {
    cwd: string;
    environmentId?: null | string;
    permissions: RequestPermissionProfile;
    reason?: null | string;
  };
}

export type TypedPendingInteraction =
  | PendingApprovalInteraction
  | PendingUserInputInteraction
  | PendingMcpElicitationInteraction
  | PendingPermissionsApprovalInteraction;

export type PendingInteractionResponse =
  | {
      interactionId: string;
      kind: "approval";
      response: ConfirmResponse;
    }
  | {
      confirmed: boolean;
      interactionId: string;
      kind: "request_user_input";
      response?: string;
      userData?: unknown;
    }
  | {
      action: "cancel" | "decline";
      interactionId: string;
      kind: "mcp_elicitation";
    }
  | {
      action: "accept";
      content: McpElicitationFormContent;
      interactionId: string;
      kind: "mcp_elicitation";
    }
  | {
      decision: "decline" | "grant_session" | "grant_turn";
      interactionId: string;
      kind: "permissions_approval";
    };

export type PendingInteractionResponseResult =
  | { accepted: true }
  | { accepted: false; issues?: McpElicitationFormIssue[] };

type PendingWireResponse =
  | CommandExecutionRequestApprovalResponse
  | FileChangeRequestApprovalResponse
  | PermissionsRequestApprovalResponse
  | ToolRequestUserInputResponse
  | McpServerElicitationRequestResponse;

interface PendingInteractionEntry {
  cancelResponse: PendingWireResponse;
  cleanup: () => void;
  projection: TypedPendingInteraction;
  resolve: (response: PendingWireResponse) => void;
  userInputParams?: ToolRequestUserInputParams;
}

/**
 * 五种 App Server reverse request 的唯一前端 pending owner。
 *
 * JSON-RPC action token 由 dispatcher 的请求闭包持有；公开 projection 只包含
 * 领域 identity，避免 Renderer UI 把 transport id 当成可持久化业务 id。
 */
export class PendingInteractionController {
  readonly #dispatcher: PendingInteractionDispatcher;
  readonly #listeners = new Set<() => void>();
  readonly #pending = new Map<string, PendingInteractionEntry>();
  #attachCount = 0;
  #nextMcpOrdinal = 1;
  #snapshot: readonly TypedPendingInteraction[] = [];
  #unregister: Array<() => void> = [];

  constructor(
    dispatcher: PendingInteractionDispatcher = getDefaultAppServerServerRequestDispatcher(),
  ) {
    this.#dispatcher = dispatcher;
  }

  subscribe = (listener: () => void): (() => void) => {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  };

  getSnapshot = (): readonly TypedPendingInteraction[] => this.#snapshot;

  attach(): () => void {
    this.#attachCount += 1;
    if (this.#unregister.length === 0) {
      try {
        this.#registerHandlers();
      } catch (error) {
        this.#attachCount = Math.max(0, this.#attachCount - 1);
        for (const unregister of this.#unregister.splice(0)) {
          unregister();
        }
        throw error;
      }
    }
    let detached = false;
    return () => {
      if (detached) {
        return;
      }
      detached = true;
      this.#attachCount = Math.max(0, this.#attachCount - 1);
      if (this.#attachCount === 0) {
        this.detach();
      }
    };
  }

  respond(
    response: PendingInteractionResponse,
  ): PendingInteractionResponseResult {
    const pending = this.#pending.get(response.interactionId);
    if (
      !pending ||
      pending.projection.kind !== response.kind ||
      pending.projection.status !== "pending"
    ) {
      return { accepted: false };
    }

    switch (response.kind) {
      case "approval": {
        const decision =
          response.response.decision ??
          (response.response.confirmed ? "allow_once" : "decline");
        this.#settle(response.interactionId, {
          decision: toWireDecision(decision),
        });
        return { accepted: true };
      }
      case "request_user_input": {
        const params = pending.userInputParams;
        if (!params) {
          return { accepted: false };
        }
        this.#settle(
          response.interactionId,
          response.confirmed
            ? responseFromUserData(
                params,
                response.userData ?? response.response,
              )
            : { answers: {} },
        );
        return { accepted: true };
      }
      case "mcp_elicitation": {
        if (pending.projection.kind !== "mcp_elicitation") {
          return { accepted: false };
        }
        if (response.action === "accept") {
          const schema = pending.projection.payload.requestedSchema;
          const issues = validateMcpElicitationFormContent(
            schema,
            response.content,
          );
          if (issues.length > 0) {
            return { accepted: false, issues };
          }
          this.#settle(response.interactionId, {
            action: "accept",
            content: response.content,
          });
          return { accepted: true };
        }
        this.#settle(response.interactionId, { action: response.action });
        return { accepted: true };
      }
      case "permissions_approval": {
        if (pending.projection.kind !== "permissions_approval") {
          return { accepted: false };
        }
        this.#settle(
          response.interactionId,
          permissionResponseForDecision(
            response.decision,
            pending.projection.payload.permissions,
          ),
        );
        return { accepted: true };
      }
    }
  }

  detach(): void {
    for (const unregister of this.#unregister.splice(0)) {
      unregister();
    }
    for (const [interactionId, pending] of this.#pending) {
      pending.cleanup();
      pending.resolve(pending.cancelResponse);
      this.#pending.delete(interactionId);
    }
    this.#attachCount = 0;
    this.#publish();
  }

  #registerHandlers(): void {
    this.#unregister.push(
      this.#dispatcher.register<
        CommandExecutionRequestApprovalParams,
        CommandExecutionRequestApprovalResponse
      >(
        METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
        (params, _request, signal) => {
          const request = commandActionFromRequest(params);
          return this.#waitForResponse(
            approvalProjection(request, params.threadId, params.turnId),
            signal,
            { decision: "cancel" },
          );
        },
      ),
    );
    this.#unregister.push(
      this.#dispatcher.register<
        FileChangeRequestApprovalParams,
        FileChangeRequestApprovalResponse
      >(
        METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
        (params, _request, signal) => {
          const request = fileActionFromRequest(params);
          return this.#waitForResponse(
            approvalProjection(request, params.threadId, params.turnId),
            signal,
            { decision: "cancel" },
          );
        },
      ),
    );
    this.#unregister.push(
      this.#dispatcher.register<
        PermissionsRequestApprovalParams,
        PermissionsRequestApprovalResponse
      >(METHOD_ITEM_PERMISSIONS_REQUEST_APPROVAL, (params, _request, signal) =>
        this.#waitForResponse(
          permissionsApprovalProjection(params),
          signal,
          declinedPermissionsResponse(),
        ),
      ),
    );
    this.#unregister.push(
      this.#dispatcher.register<
        ToolRequestUserInputParams,
        ToolRequestUserInputResponse
      >(METHOD_ITEM_TOOL_REQUEST_USER_INPUT, (params, _request, signal) => {
        const request = userInputActionFromRequest(params);
        return this.#waitForResponse(
          userInputProjection(request, params),
          signal,
          { answers: {} },
          params,
        );
      }),
    );
    this.#unregister.push(
      this.#dispatcher.register<
        McpServerElicitationRequestParams,
        McpServerElicitationRequestResponse
      >(METHOD_MCP_SERVER_ELICITATION_REQUEST, (input, _request, signal) => {
        const params = normalizeMcpRequest(input);
        const interactionId = semanticInteractionId(
          "mcp_elicitation",
          params.threadId,
          params.turnId ?? "thread",
          String(this.#nextMcpOrdinal),
        );
        this.#nextMcpOrdinal += 1;
        return this.#waitForResponse(
          {
            id: interactionId,
            thread_id: params.threadId,
            ...(params.turnId ? { turn_id: params.turnId } : {}),
            kind: "mcp_elicitation",
            status: "pending",
            payload: {
              message: params.message,
              requestedSchema: params.requestedSchema,
              serverName: params.serverName,
            },
          },
          signal,
          { action: "cancel" },
        );
      }),
    );
  }

  #waitForResponse<TResponse extends PendingWireResponse>(
    projection: TypedPendingInteraction,
    signal: AbortSignal,
    cancelResponse: TResponse,
    userInputParams?: ToolRequestUserInputParams,
  ): Promise<TResponse> {
    if (this.#pending.has(projection.id)) {
      throw new Error(`Duplicate pending interaction: ${projection.id}`);
    }
    return new Promise<TResponse>((resolve) => {
      let settled = false;
      const onAbort = () => {
        if (settled) {
          return;
        }
        settled = true;
        this.#remove(projection.id);
        resolve(cancelResponse);
      };
      const cleanup = () => signal.removeEventListener("abort", onAbort);
      this.#pending.set(projection.id, {
        cancelResponse,
        cleanup,
        projection,
        resolve: (response) => {
          if (settled) {
            return;
          }
          settled = true;
          cleanup();
          resolve(response as TResponse);
        },
        ...(userInputParams ? { userInputParams } : {}),
      });
      signal.addEventListener("abort", onAbort, { once: true });
      this.#publish();
      if (signal.aborted) {
        onAbort();
      }
    });
  }

  #settle(interactionId: string, response: PendingWireResponse): void {
    const pending = this.#pending.get(interactionId);
    if (!pending) {
      return;
    }
    this.#pending.delete(interactionId);
    pending.cleanup();
    this.#publish();
    pending.resolve(response);
  }

  #remove(interactionId: string): void {
    const pending = this.#pending.get(interactionId);
    if (!pending) {
      return;
    }
    pending.cleanup();
    this.#pending.delete(interactionId);
    this.#publish();
  }

  #publish(): void {
    this.#snapshot = [...this.#pending.values()].map(
      ({ projection }) => projection,
    );
    for (const listener of this.#listeners) {
      listener();
    }
  }
}

let defaultPendingInteractionController: PendingInteractionController | null =
  null;

export function getDefaultPendingInteractionController(): PendingInteractionController {
  if (!defaultPendingInteractionController) {
    defaultPendingInteractionController = new PendingInteractionController();
  }
  return defaultPendingInteractionController;
}

export function resetDefaultPendingInteractionControllerForTests(): void {
  defaultPendingInteractionController?.detach();
  defaultPendingInteractionController = null;
}

export function actionFromPendingInteraction(
  interaction: TypedPendingInteraction,
): ActionRequired | null {
  return interaction.kind === "approval" ||
    interaction.kind === "request_user_input"
    ? interaction.payload.request
    : null;
}

export function findPendingActionInteraction(
  interactions: readonly TypedPendingInteraction[],
  actionType: ActionRequired["actionType"],
  requestId: string,
): PendingApprovalInteraction | PendingUserInputInteraction | null {
  const expectedKind =
    actionType === "tool_confirmation"
      ? "approval"
      : actionType === "ask_user"
        ? "request_user_input"
        : null;
  if (!expectedKind) {
    return null;
  }
  const interaction = interactions.find(
    (candidate) =>
      candidate.kind === expectedKind &&
      candidate.payload.request.requestId === requestId,
  );
  return interaction?.kind === "approval" ||
    interaction?.kind === "request_user_input"
    ? interaction
    : null;
}

function approvalProjection(
  request: ActionRequired,
  threadId: string,
  turnId: string,
): PendingApprovalInteraction {
  return {
    id: semanticInteractionId("approval", threadId, turnId, request.requestId),
    thread_id: threadId,
    turn_id: turnId,
    kind: "approval",
    status: "pending",
    payload: { request },
  };
}

function userInputProjection(
  request: ActionRequired,
  params: ToolRequestUserInputParams,
): PendingUserInputInteraction {
  return {
    id: semanticInteractionId(
      "request_user_input",
      params.threadId,
      params.turnId,
      params.itemId,
    ),
    thread_id: params.threadId,
    turn_id: params.turnId,
    item_id: params.itemId,
    kind: "request_user_input",
    status: "pending",
    payload: { request },
  };
}

function permissionsApprovalProjection(
  params: PermissionsRequestApprovalParams,
): PendingPermissionsApprovalInteraction {
  return {
    id: semanticInteractionId(
      "permissions_approval",
      params.threadId,
      params.turnId,
      params.itemId,
    ),
    thread_id: params.threadId,
    turn_id: params.turnId,
    item_id: params.itemId,
    kind: "permissions_approval",
    status: "pending",
    payload: {
      cwd: params.cwd,
      permissions: params.permissions,
      ...(params.environmentId === undefined
        ? {}
        : { environmentId: params.environmentId }),
      ...(params.reason === undefined ? {} : { reason: params.reason }),
    },
  };
}

function semanticInteractionId(
  kind: TypedPendingInteraction["kind"],
  ...parts: string[]
): string {
  return [kind, ...parts.map((part) => encodeURIComponent(part))].join(":");
}

function commandActionFromRequest(
  params: CommandExecutionRequestApprovalParams,
): ActionRequired {
  const requestId = params.approvalId || params.itemId;
  const command = params.command?.trim();
  const argumentsValue =
    command || params.networkApprovalContext
      ? normalizeActionArguments({
          ...(command ? { command } : {}),
          ...(params.networkApprovalContext
            ? { networkApprovalContext: params.networkApprovalContext }
            : {}),
        })
      : undefined;
  return {
    requestId,
    actionType: "tool_confirmation",
    toolName:
      command || params.networkApprovalContext ? "exec_command" : undefined,
    arguments: argumentsValue,
    prompt: params.reason || command || undefined,
    scope: { threadId: params.threadId, turnId: params.turnId },
    availableDecisions: (
      params.availableDecisions ?? [
        "accept",
        "acceptForSession",
        "decline",
        "cancel",
      ]
    )
      .map(fromWireDecision)
      .filter((decision): decision is ApprovalDecision => decision !== null),
    status: "pending",
  };
}

function fileActionFromRequest(
  params: FileChangeRequestApprovalParams,
): ActionRequired {
  return {
    requestId: params.itemId,
    actionType: "tool_confirmation",
    toolName: "apply_patch",
    prompt: params.reason || undefined,
    scope: { threadId: params.threadId, turnId: params.turnId },
    availableDecisions: [
      "allow_once",
      "allow_for_session",
      "decline",
      "cancel",
    ],
    status: "pending",
  };
}

function userInputActionFromRequest(
  params: ToolRequestUserInputParams,
): ActionRequired {
  return {
    requestId: params.itemId,
    actionType: "ask_user",
    prompt: params.questions[0]?.question,
    questions: params.questions.map<Question>((question) => ({
      header: question.header,
      question: question.question,
      options: question.options?.map((option) => ({
        label: option.label,
        description: option.description,
      })),
    })),
    scope: { threadId: params.threadId, turnId: params.turnId },
    status: "pending",
  };
}

function normalizeMcpRequest(
  input: McpServerElicitationRequestParams,
): McpServerElicitationRequestParams {
  const threadId = requiredText(input.threadId, "threadId");
  const turnId = optionalText(input.turnId, "turnId");
  const serverName = requiredText(input.serverName, "serverName");
  const message = requiredText(input.message, "message");
  if (input.mode !== "form") {
    throw new Error("MCP server elicitation requires form mode");
  }
  validateMcpElicitationFormContent(input.requestedSchema, {});
  return {
    ...(input._meta === undefined ? {} : { _meta: input._meta }),
    message,
    mode: "form",
    requestedSchema: input.requestedSchema,
    serverName,
    threadId,
    turnId,
  };
}

function requiredText(value: unknown, field: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`Pending interaction requires canonical ${field}`);
  }
  return value.trim();
}

function optionalText(value: unknown, field: string): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  return requiredText(value, field);
}

function fromWireDecision(
  decision: CommandExecutionApprovalDecision,
): ApprovalDecision | null {
  switch (decision) {
    case "accept":
      return "allow_once";
    case "acceptForSession":
      return "allow_for_session";
    case "decline":
      return "decline";
    case "cancel":
      return "cancel";
    default:
      return null;
  }
}

function toWireDecision(
  decision: ApprovalDecision,
): CommandExecutionApprovalDecision {
  switch (decision) {
    case "allow_once":
      return "accept";
    case "allow_for_session":
      return "acceptForSession";
    case "decline":
      return "decline";
    case "cancel":
      return "cancel";
  }
}

function permissionResponseForDecision(
  decision: unknown,
  permissions: RequestPermissionProfile,
): PermissionsRequestApprovalResponse {
  switch (decision) {
    case "grant_turn":
      return { permissions, scope: "turn" };
    case "grant_session":
      return { permissions, scope: "session" };
    case "decline":
    default:
      return declinedPermissionsResponse();
  }
}

function declinedPermissionsResponse(): PermissionsRequestApprovalResponse {
  return { permissions: {}, scope: "turn" };
}

function responseFromUserData(
  params: ToolRequestUserInputParams,
  userData: unknown,
): ToolRequestUserInputResponse {
  const answers: ToolRequestUserInputResponse["answers"] = {};
  for (const question of params.questions) {
    const value = answerValue(
      userData,
      question.id,
      question.header,
      question.question,
    );
    const normalized = normalizeAnswers(value);
    if (normalized.length > 0) {
      answers[question.id] = { answers: normalized };
    }
  }
  return { answers };
}

function answerValue(
  userData: unknown,
  id: string,
  header: string,
  question: string,
): unknown {
  if (
    typeof userData !== "object" ||
    userData === null ||
    Array.isArray(userData)
  ) {
    return userData;
  }
  const record = userData as Record<string, unknown>;
  const nested =
    typeof record.answers === "object" && record.answers !== null
      ? (record.answers as Record<string, unknown>)
      : undefined;
  return (
    record[id] ??
    record[header] ??
    record[question] ??
    nested?.[id] ??
    nested?.[header] ??
    nested?.[question]
  );
}

function normalizeAnswers(value: unknown): string[] {
  const values = Array.isArray(value) ? value : [value];
  return values
    .flatMap((entry) =>
      typeof entry === "string" && entry.includes(",")
        ? entry.split(",")
        : [entry],
    )
    .map((entry) =>
      typeof entry === "string"
        ? entry.trim()
        : typeof entry === "number" || typeof entry === "boolean"
          ? String(entry)
          : "",
    )
    .filter(Boolean);
}
