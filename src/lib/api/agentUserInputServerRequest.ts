import {
  METHOD_ITEM_TOOL_REQUEST_USER_INPUT,
  type ToolRequestUserInputParams,
  type ToolRequestUserInputResponse,
} from "@limecloud/app-server-client";
import type {
  ActionRequired,
  ConfirmResponse,
  Question,
} from "./agentActionTypes";
import {
  getDefaultAppServerServerRequestDispatcher,
  type AppServerServerRequestDispatcher,
} from "./appServerServerRequest";

export { METHOD_ITEM_TOOL_REQUEST_USER_INPUT };
export type UserInputRequest = ToolRequestUserInputParams;
export type UserInputResponse = ToolRequestUserInputResponse;

interface PendingUserInput {
  action: ActionRequired;
  params: UserInputRequest;
  cleanup: () => void;
  resolve: (response: UserInputResponse) => void;
}

type UserInputDispatcher = Pick<AppServerServerRequestDispatcher, "register">;

export class AgentUserInputServerRequestController {
  readonly #dispatcher: UserInputDispatcher;
  readonly #listeners = new Set<() => void>();
  readonly #pending = new Map<string, PendingUserInput>();
  #snapshot: readonly ActionRequired[] = [];
  #unregister: (() => void) | null = null;
  #attachCount = 0;

  constructor(
    dispatcher: UserInputDispatcher = getDefaultAppServerServerRequestDispatcher(),
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
    if (!this.#unregister) {
      this.#unregister = this.#dispatcher.register<
        UserInputRequest,
        UserInputResponse
      >(METHOD_ITEM_TOOL_REQUEST_USER_INPUT, (params, _request, signal) =>
        this.#waitForResponse(params, signal),
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
    const pending = this.#pending.get(response.requestId);
    if (!pending) {
      return false;
    }
    return this.#settle(
      response.requestId,
      response.confirmed === false
        ? { answers: {} }
        : responseFromUserData(
            pending.params,
            response.userData ?? response.response,
          ),
    );
  }

  detach(): void {
    this.#unregister?.();
    this.#unregister = null;
    for (const [key, pending] of this.#pending) {
      pending.cleanup();
      pending.resolve({ answers: {} });
      this.#pending.delete(key);
    }
    if (this.#snapshot.length > 0) {
      this.#snapshot = [];
      this.#publish();
    }
  }

  #waitForResponse(
    params: UserInputRequest,
    signal: AbortSignal,
  ): Promise<UserInputResponse> {
    const action = actionFromRequest(params);
    return new Promise((resolve) => {
      let settled = false;
      const key = action.requestId;
      const onAbort = () => {
        if (settled) {
          return;
        }
        settled = true;
        this.#remove(key);
        resolve({ answers: {} });
      };
      const cleanup = () => signal.removeEventListener("abort", onAbort);
      signal.addEventListener("abort", onAbort, { once: true });
      this.#pending.set(key, {
        action,
        params,
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

  #settle(key: string, response: UserInputResponse): boolean {
    const pending = this.#pending.get(key);
    if (!pending) {
      return false;
    }
    pending.cleanup();
    pending.resolve(response);
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

let defaultController: AgentUserInputServerRequestController | null = null;

export function getDefaultAgentUserInputServerRequestController(): AgentUserInputServerRequestController {
  if (!defaultController) {
    defaultController = new AgentUserInputServerRequestController();
  }
  return defaultController;
}

export function resetDefaultAgentUserInputServerRequestControllerForTests(): void {
  defaultController?.detach();
  defaultController = null;
}

function actionFromRequest(params: UserInputRequest): ActionRequired {
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
    scope: {
      threadId: params.threadId,
      turnId: params.turnId,
    },
    status: "pending",
  };
}

function responseFromUserData(
  params: UserInputRequest,
  userData: unknown,
): UserInputResponse {
  const answers: UserInputResponse["answers"] = {};
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
