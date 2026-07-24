import {
  METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
  METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
  type CommandExecutionApprovalDecision,
  type CommandExecutionRequestApprovalParams,
  type CommandExecutionRequestApprovalResponse,
  type FileChangeRequestApprovalParams,
  type FileChangeRequestApprovalResponse,
} from "@limecloud/app-server-client";
import type {
  ActionRequired,
  ApprovalDecision,
  ConfirmResponse,
} from "./agentActionTypes";
import {
  getDefaultAppServerServerRequestDispatcher,
  type AppServerServerRequestDispatcher,
} from "./appServerServerRequest";

export { METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL };
export { METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL };
export type CommandApprovalRequest = CommandExecutionRequestApprovalParams;
export type CommandApprovalResponse = CommandExecutionRequestApprovalResponse;
export type FileApprovalRequest = FileChangeRequestApprovalParams;
export type FileApprovalResponse = FileChangeRequestApprovalResponse;

type WireApprovalDecision = CommandExecutionApprovalDecision;

interface PendingApproval {
  action: ActionRequired;
  cleanup: () => void;
  resolve: (decision: WireApprovalDecision) => void;
}

type ApprovalDispatcher = Pick<AppServerServerRequestDispatcher, "register">;

export class AgentApprovalServerRequestController {
  readonly #dispatcher: ApprovalDispatcher;
  readonly #listeners = new Set<() => void>();
  readonly #pending = new Map<string, PendingApproval>();
  #snapshot: readonly ActionRequired[] = [];
  #unregister: Array<() => void> = [];
  #attachCount = 0;

  constructor(
    dispatcher: ApprovalDispatcher = getDefaultAppServerServerRequestDispatcher(),
  ) {
    this.#dispatcher = dispatcher;
  }

  subscribe = (listener: () => void): (() => void) => {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  };

  getSnapshot = (): readonly ActionRequired[] => this.#snapshot;

  attach(): () => void {
    this.#attachCount += 1;
    if (this.#unregister.length === 0) {
      this.#unregister.push(
        this.#dispatcher.register<
          CommandApprovalRequest,
          CommandApprovalResponse
        >(
          METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
          (params, _request, signal) =>
            this.#waitForResponse(
              commandActionFromRequest(params),
              signal,
            ).then((decision) => ({ decision })),
        ),
      );
      this.#unregister.push(
        this.#dispatcher.register<FileApprovalRequest, FileApprovalResponse>(
          METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
          (params, _request, signal) =>
            this.#waitForResponse(fileActionFromRequest(params), signal).then(
              (decision) => ({ decision }),
            ),
        ),
      );
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

  respond(response: ConfirmResponse): boolean {
    if (!this.#pending.has(response.requestId)) {
      return false;
    }
    const decision =
      response.decision ?? (response.confirmed ? "allow_once" : "decline");
    return this.#settle(response.requestId, toWireDecision(decision));
  }

  detach(): void {
    for (const unregister of this.#unregister.splice(0)) {
      unregister();
    }
    for (const [key, pending] of this.#pending) {
      pending.cleanup();
      pending.resolve("cancel");
      this.#pending.delete(key);
    }
    if (this.#snapshot.length > 0) {
      this.#snapshot = [];
      this.#publish();
    }
  }

  #waitForResponse(
    action: ActionRequired,
    signal: AbortSignal,
  ): Promise<WireApprovalDecision> {
    return new Promise((resolve) => {
      let settled = false;
      const key = action.requestId;
      const onAbort = () => {
        if (settled) {
          return;
        }
        settled = true;
        this.#remove(key);
        resolve("cancel");
      };
      const cleanup = () => signal.removeEventListener("abort", onAbort);
      signal.addEventListener("abort", onAbort, { once: true });
      this.#pending.set(key, {
        action,
        cleanup,
        resolve: (response) => {
          if (settled) {
            return;
          }
          settled = true;
          cleanup();
          resolve(response);
        },
      });
      this.#publish();
    });
  }

  #settle(key: string, decision: WireApprovalDecision): boolean {
    const pending = this.#pending.get(key);
    if (!pending) {
      return false;
    }
    pending.cleanup();
    pending.resolve(decision);
    this.#remove(key);
    return true;
  }

  #remove(key: string): void {
    if (!this.#pending.delete(key)) {
      return;
    }
    this.#publish();
  }

  #publish(): void {
    this.#snapshot = [...this.#pending.values()].map(({ action }) => action);
    for (const listener of this.#listeners) {
      listener();
    }
  }
}

let defaultController: AgentApprovalServerRequestController | null = null;

export function getDefaultAgentApprovalServerRequestController(): AgentApprovalServerRequestController {
  if (!defaultController) {
    defaultController = new AgentApprovalServerRequestController();
  }
  return defaultController;
}

export function resetDefaultAgentApprovalServerRequestControllerForTests(): void {
  defaultController?.detach();
  defaultController = null;
}

function commandActionFromRequest(
  params: CommandApprovalRequest,
): ActionRequired {
  const requestId = params.approvalId || params.itemId;
  const command = params.command?.trim();
  return {
    requestId,
    actionType: "tool_confirmation",
    toolName: command ? "exec_command" : undefined,
    arguments: command ? { command } : undefined,
    prompt: params.reason || command || undefined,
    scope: {
      threadId: params.threadId,
      turnId: params.turnId,
    },
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

function fileActionFromRequest(params: FileApprovalRequest): ActionRequired {
  return {
    requestId: params.itemId,
    actionType: "tool_confirmation",
    toolName: "apply_patch",
    prompt: params.reason || undefined,
    scope: {
      threadId: params.threadId,
      turnId: params.turnId,
    },
    availableDecisions: [
      "allow_once",
      "allow_for_session",
      "decline",
      "cancel",
    ],
    status: "pending",
  };
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
