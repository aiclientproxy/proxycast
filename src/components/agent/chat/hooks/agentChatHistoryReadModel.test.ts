import { describe, expect, it } from "vitest";
import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";
import { hydrateSessionDetailMessagesFromThreadReadToolCalls } from "./agentChatHistoryReadModel";

describe("agentChatHistoryReadModel", () => {
  it("应从 thread_read.turns 恢复工具轨迹消息的 usage", () => {
    const detail = {
      messages: [],
      turns: [],
      items: [],
      thread_read: {
        thread_id: "thread-image",
        turns: [
          {
            turnId: "turn-image",
            status: "completed",
            usage: {
              inputTokens: 31_000,
              outputTokens: 119,
              cachedInputTokens: 0,
              cache_write_input_tokens: 1_200,
            },
          },
        ],
        tool_calls: [
          {
            tool_call_id: "tool-image",
            turn_id: "turn-image",
            tool_name: "lime_create_image_generation_task",
            status: "completed",
            started_at: "2026-07-03T10:00:00.000Z",
            finished_at: "2026-07-03T10:00:01.000Z",
            success: true,
            output: "",
          },
        ],
      },
    } as unknown as AgentSessionDetail;

    const messages = hydrateSessionDetailMessagesFromThreadReadToolCalls(
      detail,
      "session-image",
    );

    expect(messages).toHaveLength(1);
    expect(messages[0]?.runtimeTurnId).toBe("turn-image");
    expect(messages[0]?.usage).toEqual({
      input_tokens: 31_000,
      output_tokens: 119,
      cached_input_tokens: 0,
      cache_creation_input_tokens: 1_200,
    });
  });

  it("应让 thread_read tool_call 继承同源 thread item provenance", () => {
    const detail = {
      messages: [],
      turns: [
        {
          id: "turn-history-replay",
          status: "running",
          started_at: "2026-07-09T10:00:00.000Z",
          updated_at: "2026-07-09T10:00:01.000Z",
        },
      ],
      items: [],
      thread_read: {
        active_turn_id: "turn-history-replay",
        thread_items: [
          {
            id: "history-replay-visual-mcp-read-file",
            type: "tool_call",
            thread_id: "thread-history-replay",
            turn_id: "turn-history-replay",
            sequence: 4,
            status: "in_progress",
            tool_name: "mcp__filesystem__read_file",
            arguments: { path: "README.md" },
            started_at: "2026-07-09T10:00:01.000Z",
            updated_at: "2026-07-09T10:00:01.000Z",
            metadata: {
              mcp: {
                server: "filesystem",
                tool: "read_file",
              },
            },
          },
        ],
        tool_calls: [
          {
            tool_call_id: "history-replay-visual-mcp-read-file",
            turn_id: "turn-history-replay",
            tool_name: "mcp__filesystem__read_file",
            status: "running",
            arguments: { path: "README.md" },
            timestamp: "2026-07-09T10:00:01.000Z",
          },
        ],
      },
    } as unknown as AgentSessionDetail;

    const messages = hydrateSessionDetailMessagesFromThreadReadToolCalls(
      detail,
      "history-replay",
    );
    const toolPart = messages[0]?.contentParts?.find(
      (part) => part.type === "tool_use",
    );

    expect(toolPart).toMatchObject({
      type: "tool_use",
      metadata: {
        source: "agent_thread_item",
        threadItemId: "history-replay-visual-mcp-read-file",
        turnId: "turn-history-replay",
        sequence: 4,
      },
      toolCall: {
        id: "history-replay-visual-mcp-read-file",
        metadata: {
          source: "agent_thread_item",
          threadItemId: "history-replay-visual-mcp-read-file",
          turnId: "turn-history-replay",
          sequence: 4,
        },
      },
    });
  });
});
