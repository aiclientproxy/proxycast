import { describe, expect, it } from "vitest";
import type { AppServerJsonRpcNotification } from "@/lib/api/appServer";
import type { Message } from "../types";
import {
  readScopedAssistantTurnUsage,
  rememberBoundedAssistantTurnUsage,
  shouldReplayRecentAssistantTurnUsage,
} from "./useAgentSessionTokenUsage";

function tokenUsageNotification(
  threadId: string,
  turnId: string,
): AppServerJsonRpcNotification {
  return {
    jsonrpc: "2.0",
    method: "thread/tokenUsage/updated",
    params: {
      threadId,
      turnId,
      tokenUsage: {
        total: {
          totalTokens: 31_000,
          inputTokens: 31_000,
          cachedInputTokens: 0,
          cacheWriteInputTokens: 1_200,
          outputTokens: 0,
          reasoningOutputTokens: 0,
        },
        last: {
          totalTokens: 31_000,
          inputTokens: 31_000,
          cachedInputTokens: 0,
          cacheWriteInputTokens: 1_200,
          outputTokens: 0,
          reasoningOutputTokens: 0,
        },
        modelContextWindow: null,
      },
    },
  };
}

function assistant(overrides: Partial<Message> = {}): Message {
  return {
    id: "assistant",
    role: "assistant",
    content: "",
    timestamp: new Date(0),
    ...overrides,
  };
}

describe("useAgentSessionTokenUsage", () => {
  it("应读取当前 thread 的 token usage", () => {
    expect(
      readScopedAssistantTurnUsage({
        notification: tokenUsageNotification("thread-1", "turn-1"),
        currentThreadId: "thread-1",
        messages: [],
        threadTurns: [],
      }),
    ).toEqual({
      runtimeTurnId: "turn-1",
      usage: {
        input_tokens: 31_000,
        output_tokens: 0,
        cached_input_tokens: 0,
        cache_creation_input_tokens: 1_200,
      },
    });
  });

  it("不应读取其它 thread 的 token usage", () => {
    expect(
      readScopedAssistantTurnUsage({
        notification: tokenUsageNotification("thread-other", "turn-1"),
        currentThreadId: "thread-1",
        messages: [],
        threadTurns: [],
      }),
    ).toBeNull();
  });

  it("thread read 尚未 hydrate 时应按已知 turn 限定 session", () => {
    expect(
      readScopedAssistantTurnUsage({
        notification: tokenUsageNotification("thread-1", "turn-1"),
        currentThreadId: null,
        messages: [assistant({ runtimeTurnId: "turn-1" })],
        threadTurns: [],
      })?.runtimeTurnId,
    ).toBe("turn-1");
  });

  it("只有缺少 usage 的图片消息才请求 recent replay", () => {
    const imageMessage = assistant({
      runtimeTurnId: "turn-1",
      imageWorkbenchPreview: {
        taskId: "task-1",
        prompt: "青柠插画",
        mode: "generate",
        status: "complete",
      },
    });
    expect(
      shouldReplayRecentAssistantTurnUsage(
        [imageMessage],
        [{ id: "turn-1", status: "running" }],
      ),
    ).toBe(false);
    expect(
      shouldReplayRecentAssistantTurnUsage(
        [imageMessage],
        [{ id: "turn-1", status: "completed" }],
      ),
    ).toBe(true);
    expect(
      shouldReplayRecentAssistantTurnUsage(
        [{ ...imageMessage, usage: { input_tokens: 1, output_tokens: 0 } }],
        [{ id: "turn-1", status: "completed" }],
      ),
    ).toBe(false);
  });

  it("turn usage 缓存应保持有界并刷新最近项", () => {
    const usageByTurnId = new Map([
      ["turn-1", { input_tokens: 1, output_tokens: 0 }],
      ["turn-2", { input_tokens: 2, output_tokens: 0 }],
    ]);
    rememberBoundedAssistantTurnUsage(
      usageByTurnId,
      "turn-3",
      { input_tokens: 3, output_tokens: 0 },
      2,
    );
    expect([...usageByTurnId.keys()]).toEqual(["turn-2", "turn-3"]);
  });
});
