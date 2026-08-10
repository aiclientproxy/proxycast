import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./agentStreamUserInputSendPreparation", () => ({
  prepareAgentStreamUserInputSend: vi.fn(),
}));

vi.mock("./agentStreamPreparedSendDispatch", () => ({
  dispatchPreparedAgentStreamSend: vi.fn(),
}));

import { dispatchPreparedAgentStreamSend } from "./agentStreamPreparedSendDispatch";
import { prepareAgentStreamUserInputSend } from "./agentStreamUserInputSendPreparation";
import { sendAgentStreamMessage } from "./agentStreamSend";

describe("sendAgentStreamMessage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const createEnv = (overrides?: Record<string, unknown>) =>
    ({
      sessionIdRef: { current: "session-existing" },
      ensureSession: vi.fn(async () => "session-existing"),
      waitForSessionProviderSelectionSync: vi.fn(async () => undefined),
      ...overrides,
    }) as never;

  it("应串起 prepare 与 dispatch", async () => {
    const env = createEnv();
    const preparedSend = { prepared: true };
    vi.mocked(prepareAgentStreamUserInputSend).mockReturnValue(
      preparedSend as never,
    );

    await sendAgentStreamMessage({
      content: "继续执行",
      images: [],
      webSearch: true,
      thinking: true,
      executionStrategyOverride: "react",
      modelOverride: "gpt-5.4",
      systemPrompt: "system",
      env,
    });

    expect(prepareAgentStreamUserInputSend).toHaveBeenCalledWith(
      expect.objectContaining({
        content: "继续执行",
        webSearch: true,
        thinking: true,
        systemPrompt: "system",
        env,
      }),
    );
    expect(dispatchPreparedAgentStreamSend).toHaveBeenCalledWith({
      preparedSend,
      env,
    });
  });

  it("空态首发应先创建会话再准备乐观消息", async () => {
    const order: string[] = [];
    const sessionIdRef = { current: null as string | null };
    const ensureSession = vi.fn(async () => {
      order.push("ensureSession");
      sessionIdRef.current = "session-created";
      return "session-created";
    });
    const env = createEnv({
      sessionIdRef,
      ensureSession,
    });
    const preparedSend = { prepared: true };
    vi.mocked(prepareAgentStreamUserInputSend).mockImplementation(() => {
      order.push("prepare");
      return preparedSend as never;
    });

    await sendAgentStreamMessage({
      content: "第一条消息",
      images: [],
      env,
    });

    expect(order).toEqual(["ensureSession", "prepare"]);
    expect(ensureSession).toHaveBeenCalledWith({
      targetSessionId: undefined,
      skipSessionRestore: false,
      skipSessionStartHooks: false,
    });
    expect(prepareAgentStreamUserInputSend).toHaveBeenCalledWith(
      expect.objectContaining({ env }),
    );
    expect(dispatchPreparedAgentStreamSend).toHaveBeenCalledWith({
      preparedSend,
      env,
    });
  });

  it("目标会话首发应先绑定 session 再准备乐观消息", async () => {
    const order: string[] = [];
    const sessionIdRef = { current: null as string | null };
    const ensureSession = vi.fn(
      async (options?: { targetSessionId?: string }) => {
        order.push("ensureSession");
        sessionIdRef.current = options?.targetSessionId ?? null;
        return sessionIdRef.current;
      },
    );
    const env = createEnv({
      sessionIdRef,
      ensureSession,
    });
    const preparedSend = { prepared: true };
    vi.mocked(prepareAgentStreamUserInputSend).mockImplementation(() => {
      order.push(`prepare:${sessionIdRef.current ?? "none"}`);
      return preparedSend as never;
    });

    await sendAgentStreamMessage({
      content: "你好",
      images: [],
      options: {
        targetSessionId: "session-materialized",
        skipSessionRestore: true,
        skipSessionStartHooks: true,
      },
      env,
    });

    expect(order).toEqual(["ensureSession", "prepare:session-materialized"]);
    expect(ensureSession).toHaveBeenCalledWith({
      targetSessionId: "session-materialized",
      skipSessionRestore: true,
      skipSessionStartHooks: true,
    });
    expect(dispatchPreparedAgentStreamSend).toHaveBeenCalledWith({
      preparedSend,
      env,
    });
  });

  it("应等待目标会话模型选择同步后再准备发送", async () => {
    const order: string[] = [];
    let releaseSelectionSync: (() => void) | undefined;
    const selectionSync = new Promise<void>((resolve) => {
      releaseSelectionSync = resolve;
    });
    const waitForSessionProviderSelectionSync = vi.fn(
      async (sessionId: string) => {
        order.push(`wait:${sessionId}`);
        await selectionSync;
      },
    );
    const env = createEnv({ waitForSessionProviderSelectionSync });
    const preparedSend = { prepared: true };
    vi.mocked(prepareAgentStreamUserInputSend).mockImplementation(() => {
      order.push("prepare");
      return preparedSend as never;
    });

    const send = sendAgentStreamMessage({
      content: "使用刚选择的模型",
      images: [],
      env,
    });
    await Promise.resolve();

    expect(order).toEqual(["wait:session-existing"]);
    expect(prepareAgentStreamUserInputSend).not.toHaveBeenCalled();

    releaseSelectionSync?.();
    await send;

    expect(order).toEqual(["wait:session-existing", "prepare"]);
    expect(dispatchPreparedAgentStreamSend).toHaveBeenCalledWith({
      preparedSend,
      env,
    });
  });
});
