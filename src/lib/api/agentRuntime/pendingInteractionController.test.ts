import { describe, expect, it, vi } from "vitest";
import {
  METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
  METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
  METHOD_ITEM_TOOL_REQUEST_USER_INPUT,
  METHOD_MCP_SERVER_ELICITATION_REQUEST,
  type CommandExecutionRequestApprovalParams,
  type McpServerElicitationRequestParams,
  type ToolRequestUserInputParams,
} from "@limecloud/app-server-client";
import type { AppServerServerRequestHandler } from "../appServerServerRequest";
import {
  PendingInteractionController,
  type PendingInteractionResponse,
} from "./pendingInteractionController";

type RegisteredHandler = AppServerServerRequestHandler<unknown, unknown>;

function createHarness() {
  const handlers = new Map<string, RegisteredHandler>();
  const dispatcher = {
    register: vi.fn((method: string, handler: RegisteredHandler) => {
      handlers.set(method, handler);
      return () => handlers.delete(method);
    }),
  };
  const controller = new PendingInteractionController(dispatcher as never);
  const detach = controller.attach();

  function dispatch<TParams>(
    method: string,
    params: TParams,
    actionToken: string,
    signal = new AbortController().signal,
  ): Promise<unknown> {
    const handler = handlers.get(method);
    if (!handler) {
      throw new Error(`handler missing: ${method}`);
    }
    return Promise.resolve(
      handler(params, { id: actionToken, method, params }, signal),
    );
  }

  return { controller, detach, dispatch, dispatcher };
}

function commandParams(): CommandExecutionRequestApprovalParams {
  return {
    approvalId: "approval-1",
    availableDecisions: ["accept", "decline"],
    command: "npm test",
    itemId: "item-command-1",
    reason: "允许执行测试？",
    startedAtMs: 1,
    threadId: "thread-1",
    turnId: "turn-1",
  };
}

function userInputParams(): ToolRequestUserInputParams {
  return {
    autoResolutionMs: null,
    itemId: "item-input-1",
    questions: [
      {
        header: "模式",
        id: "mode",
        isOther: false,
        isSecret: false,
        options: [
          { label: "自动", description: "直接继续" },
          { label: "确认", description: "再次确认" },
        ],
        question: "请选择执行模式",
      },
    ],
    threadId: "thread-1",
    turnId: "turn-1",
  };
}

function mcpParams(): McpServerElicitationRequestParams {
  return {
    message: "请确认发布参数",
    mode: "form",
    requestedSchema: {
      type: "object",
      properties: {
        environment: {
          type: "string",
          enum: ["staging", "production"],
        },
        retries: { type: "integer", minimum: 0, maximum: 3 },
      },
      required: ["environment", "retries"],
    },
    serverName: "release-tools",
    threadId: "thread-1",
    turnId: "turn-1",
  };
}

describe("PendingInteractionController", () => {
  it("只用一个 owner 注册四种 current server request", () => {
    const harness = createHarness();

    expect(
      harness.dispatcher.register.mock.calls.map(([method]) => method),
    ).toEqual([
      METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
      METHOD_ITEM_FILE_CHANGE_REQUEST_APPROVAL,
      METHOD_ITEM_TOOL_REQUEST_USER_INPUT,
      METHOD_MCP_SERVER_ELICITATION_REQUEST,
    ]);

    harness.detach();
  });

  it("approval projection 使用 semantic id 且不暴露 Electron action token", async () => {
    const harness = createHarness();
    const response = harness.dispatch(
      METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
      commandParams(),
      "electron-action:approval-secret-token",
    );
    const [interaction] = harness.controller.getSnapshot();

    expect(interaction).toMatchObject({
      id: "approval:thread-1:turn-1:approval-1",
      thread_id: "thread-1",
      turn_id: "turn-1",
      kind: "approval",
      status: "pending",
      payload: {
        request: {
          requestId: "approval-1",
          actionType: "tool_confirmation",
        },
      },
    });
    expect(JSON.stringify(interaction)).not.toContain("electron-action:");

    const submission: PendingInteractionResponse = {
      interactionId: interaction.id,
      kind: "approval",
      response: {
        requestId: "approval-1",
        actionType: "tool_confirmation",
        decision: "allow_once",
      },
    };
    expect(harness.controller.respond(submission)).toEqual({ accepted: true });
    expect(harness.controller.respond(submission)).toEqual({ accepted: false });
    await expect(response).resolves.toEqual({ decision: "accept" });
    expect(harness.controller.getSnapshot()).toEqual([]);
    harness.detach();
  });

  it("requestUserInput 通过同一 projection 按 question id 结构化回包", async () => {
    const harness = createHarness();
    const response = harness.dispatch(
      METHOD_ITEM_TOOL_REQUEST_USER_INPUT,
      userInputParams(),
      "electron-action:user-input-token",
    );
    const [interaction] = harness.controller.getSnapshot();

    expect(interaction).toMatchObject({
      id: "request_user_input:thread-1:turn-1:item-input-1",
      kind: "request_user_input",
      item_id: "item-input-1",
      payload: {
        request: {
          requestId: "item-input-1",
          actionType: "ask_user",
          prompt: "请选择执行模式",
        },
      },
    });
    expect(
      harness.controller.respond({
        interactionId: interaction.id,
        kind: "request_user_input",
        confirmed: true,
        userData: { 模式: "确认" },
      }),
    ).toEqual({ accepted: true });
    await expect(response).resolves.toEqual({
      answers: { mode: { answers: ["确认"] } },
    });
    harness.detach();
  });

  it("MCP 表单校验失败保持 pending，成功后 action token 只能消费一次", async () => {
    const harness = createHarness();
    const response = harness.dispatch(
      METHOD_MCP_SERVER_ELICITATION_REQUEST,
      mcpParams(),
      "electron-action:mcp-token",
    );
    const [interaction] = harness.controller.getSnapshot();

    expect(interaction).toMatchObject({
      id: "mcp_elicitation:thread-1:turn-1:1",
      kind: "mcp_elicitation",
      payload: {
        message: "请确认发布参数",
        serverName: "release-tools",
      },
    });
    expect(
      harness.controller.respond({
        interactionId: interaction.id,
        kind: "mcp_elicitation",
        action: "accept",
        content: { environment: "invalid", retries: 1.5 },
      }),
    ).toEqual({
      accepted: false,
      issues: [
        { code: "invalid_enum", field: "environment" },
        { code: "invalid_integer", field: "retries" },
      ],
    });
    expect(harness.controller.getSnapshot()).toHaveLength(1);

    const submission: PendingInteractionResponse = {
      interactionId: interaction.id,
      kind: "mcp_elicitation",
      action: "accept",
      content: { environment: "production", retries: 2 },
    };
    expect(harness.controller.respond(submission)).toEqual({ accepted: true });
    expect(harness.controller.respond(submission)).toEqual({ accepted: false });
    await expect(response).resolves.toEqual({
      action: "accept",
      content: { environment: "production", retries: 2 },
    });
    harness.detach();
  });

  it("远端 resolved abort 会 fail closed 并清理当前 interaction", async () => {
    const harness = createHarness();
    const abort = new AbortController();
    const response = harness.dispatch(
      METHOD_MCP_SERVER_ELICITATION_REQUEST,
      mcpParams(),
      "electron-action:remote-resolved",
      abort.signal,
    );
    expect(harness.controller.getSnapshot()).toHaveLength(1);

    abort.abort();

    await expect(response).resolves.toEqual({ action: "cancel" });
    expect(harness.controller.getSnapshot()).toEqual([]);
    harness.detach();
  });
});
