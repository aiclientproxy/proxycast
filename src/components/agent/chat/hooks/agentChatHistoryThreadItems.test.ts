import { describe, expect, it } from "vitest";

import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";

import { hydrateSessionDetailMessages } from "./agentChatHistory";
import { mergeThreadItemReasoningIntoMessages } from "./agentChatHistoryReasoning";
import {
  collectDetailThreadItems,
  hydrateSessionDetailMessagesFromThreadItems,
} from "./agentChatHistoryThreadItems";
import { orderStreamingContentPartsForDisplay } from "../components/streamingContentPartOrder";

describe("agentChatHistoryThreadItems", () => {
  it("只有 hosted image item 的历史也应恢复 assistant 图片消息", () => {
    const messages = hydrateSessionDetailMessagesFromThreadItems(
      {
        id: "hosted-image-history",
        created_at: 1,
        updated_at: 2,
        messages: [],
        turns: [
          {
            id: "turn-hosted-image",
            thread_id: "thread-hosted-image",
            status: "completed",
          },
        ],
        items: [
          {
            id: "image-generation-hosted",
            type: "image_generation",
            thread_id: "thread-hosted-image",
            turn_id: "turn-hosted-image",
            sequence: 1,
            status: "completed",
            generation_status: "completed",
            revised_prompt: "a green circle",
            result: "aW1hZ2U=",
            saved_path: "/tmp/green-circle.png",
            started_at: "2026-07-27T10:00:00.000Z",
            updated_at: "2026-07-27T10:00:01.000Z",
            completed_at: "2026-07-27T10:00:01.000Z",
          },
        ],
      } as unknown as AgentSessionDetail,
      "hosted-image-history",
    );

    expect(messages).toEqual([
      expect.objectContaining({
        role: "assistant",
        content: "",
        runtimeTurnId: "turn-hosted-image",
        contentParts: [
          expect.objectContaining({
            type: "media_reference",
            reference: expect.objectContaining({
              uri: "data:image/png;base64,aW1hZ2U=",
              caption: "a green circle",
              sourcePath: "/tmp/green-circle.png",
            }),
          }),
        ],
      }),
    ]);
  });

  it("canonical declined FileChange 历史应保留精确 batch 与拒绝状态", () => {
    const messages = hydrateSessionDetailMessagesFromThreadItems(
      {
        id: "file-change-declined-session",
        created_at: 1,
        updated_at: 2,
        messages: [],
        turns: [
          {
            id: "turn-file-change-declined",
            thread_id: "thread-file-change-declined",
            status: "completed",
          },
        ],
        items: [
          {
            id: "file-change-declined",
            type: "patch",
            thread_id: "thread-file-change-declined",
            turn_id: "turn-file-change-declined",
            sequence: 1,
            status: "failed",
            file_status: "declined",
            text: "declined batch",
            paths: ["src/added.ts", "src/source.ts"],
            changes: [
              {
                path: "src/added.ts",
                kind: { type: "add" },
                diff: "+export const added = true;",
              },
              {
                path: "src/source.ts",
                kind: { type: "update", move_path: "src/destination.ts" },
                diff: "-source\n+destination",
              },
            ],
            started_at: "2026-07-21T00:00:00.000Z",
            updated_at: "2026-07-21T00:00:01.000Z",
            completed_at: "2026-07-21T00:00:01.000Z",
          },
        ],
      } as unknown as AgentSessionDetail,
      "file-change-declined-session",
    );

    expect(messages).toHaveLength(1);
    expect(messages[0]?.contentParts).toEqual([
      expect.objectContaining({
        type: "file_changes_batch",
        aggregate: expect.objectContaining({
          fileCount: 2,
          files: [
            expect.objectContaining({
              path: "src/added.ts",
              fileStatus: "declined",
            }),
            expect.objectContaining({
              path: "src/source.ts",
              movePath: "src/destination.ts",
              fileStatus: "declined",
            }),
          ],
        }),
      }),
    ]);
  });

  it("typed skill/mention 历史 part 应在 GUI projection 中保持可见", () => {
    const detail = {
      id: "typed-reference-parts-session",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          id: "user-reference-parts",
          role: "user",
          runtime_turn_id: "turn-1",
          timestamp: 1784188800,
          content: [
            { type: "skill", name: "review", path: "/skills/review" },
            { type: "mention", name: "README", path: "README.md" },
          ],
        },
      ],
      turns: [],
      items: [],
    } as unknown as AgentSessionDetail;

    expect(
      hydrateSessionDetailMessages(detail, "typed-reference-parts-session"),
    ).toEqual([
      expect.objectContaining({
        id: "typed-reference-parts-session-0",
        role: "user",
        content: "$review (/skills/review)\n@README (README.md)",
        contentParts: [
          {
            type: "text",
            text: "$review (/skills/review)\n@README (README.md)",
          },
        ],
      }),
    ]);
  });

  it("canonical items 应覆盖 legacy messages 缺失的同 Turn 多段 final 与助手消息", () => {
    const detail = {
      id: "canonical-message-owner-session",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "user",
          runtime_turn_id: "turn-1",
          timestamp: 1784188800,
          content: [{ type: "text", text: "第一轮" }],
        },
        {
          role: "assistant",
          runtime_turn_id: "turn-1",
          timestamp: 1784188801,
          content: [{ type: "output_text", text: "第一段最终答复" }],
        },
        {
          role: "user",
          runtime_turn_id: "turn-2",
          timestamp: 1784188802,
          content: [{ type: "text", text: "第二轮" }],
        },
      ],
      turns: [
        {
          id: "turn-1",
          thread_id: "thread-1",
          prompt_text: "第一轮",
          status: "completed",
          started_at: "2026-07-16T08:00:00.000Z",
          created_at: "2026-07-16T08:00:00.000Z",
          updated_at: "2026-07-16T08:00:01.000Z",
        },
        {
          id: "turn-2",
          thread_id: "thread-1",
          prompt_text: "第二轮",
          status: "canceled",
          started_at: "2026-07-16T08:00:02.000Z",
          created_at: "2026-07-16T08:00:02.000Z",
          updated_at: "2026-07-16T08:00:03.000Z",
        },
      ],
      items: [
        {
          id: "user-1",
          type: "user_message",
          thread_id: "thread-1",
          turn_id: "turn-1",
          sequence: 1,
          status: "completed",
          content: "第一轮",
          started_at: "2026-07-16T08:00:00.000Z",
          updated_at: "2026-07-16T08:00:00.000Z",
        },
        {
          id: "final-1a",
          type: "agent_message",
          thread_id: "thread-1",
          turn_id: "turn-1",
          sequence: 2,
          phase: "final_answer",
          status: "completed",
          text: "第一段最终答复",
          started_at: "2026-07-16T08:00:01.000Z",
          updated_at: "2026-07-16T08:00:01.000Z",
        },
        {
          id: "final-1b",
          type: "agent_message",
          thread_id: "thread-1",
          turn_id: "turn-1",
          sequence: 3,
          phase: "final_answer",
          status: "completed",
          text: "第二段最终答复",
          started_at: "2026-07-16T08:00:01.500Z",
          updated_at: "2026-07-16T08:00:01.500Z",
        },
        {
          id: "user-2",
          type: "user_message",
          thread_id: "thread-1",
          turn_id: "turn-2",
          sequence: 4,
          status: "completed",
          content: "第二轮",
          started_at: "2026-07-16T08:00:02.000Z",
          updated_at: "2026-07-16T08:00:02.000Z",
        },
        {
          id: "final-2",
          type: "agent_message",
          thread_id: "thread-1",
          turn_id: "turn-2",
          sequence: 5,
          phase: "final_answer",
          status: "completed",
          text: "取消前已经生成的最终答复",
          started_at: "2026-07-16T08:00:03.000Z",
          updated_at: "2026-07-16T08:00:03.000Z",
        },
      ],
    } as unknown as AgentSessionDetail;

    const messages = hydrateSessionDetailMessages(
      detail,
      "canonical-message-owner-session",
    );
    const assistantMessages = messages.filter(
      (message) => message.role === "assistant",
    );

    expect(messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
      "user",
      "assistant",
    ]);
    expect(
      assistantMessages[0]?.contentParts
        ?.filter((part) => part.type === "text")
        .map((part) => ({
          text: part.text,
          threadItemId: part.metadata?.threadItemId,
          phase: part.metadata?.phase,
        })),
    ).toEqual([
      {
        text: "第一段最终答复",
        threadItemId: "final-1a",
        phase: "final_answer",
      },
      {
        text: "第二段最终答复",
        threadItemId: "final-1b",
        phase: "final_answer",
      },
    ]);
    expect(assistantMessages[1]).toMatchObject({
      runtimeTurnId: "turn-2",
      content: "取消前已经生成的最终答复",
      contentParts: [
        expect.objectContaining({
          type: "text",
          text: "取消前已经生成的最终答复",
          metadata: expect.objectContaining({
            threadItemId: "final-2",
            phase: "final_answer",
          }),
        }),
      ],
    });
  });

  it("同一 Turn 的前序无 phase AgentMessage 应保留为过程文本", () => {
    const detail = {
      id: "legacy-unphased-agent-messages",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "turn-1",
          thread_id: "thread-1",
          prompt_text: "继续",
          status: "completed",
          started_at: "2026-07-16T08:00:00.000Z",
          created_at: "2026-07-16T08:00:00.000Z",
          updated_at: "2026-07-16T08:00:03.000Z",
        },
      ],
      items: [
        {
          id: "user-1",
          type: "user_message",
          thread_id: "thread-1",
          turn_id: "turn-1",
          sequence: 1,
          status: "completed",
          content: "继续",
          started_at: "2026-07-16T08:00:00.000Z",
          updated_at: "2026-07-16T08:00:00.000Z",
        },
        {
          id: "agent-progress",
          type: "agent_message",
          thread_id: "thread-1",
          turn_id: "turn-1",
          sequence: 2,
          status: "completed",
          text: "我先核对当前实现。",
          started_at: "2026-07-16T08:00:01.000Z",
          updated_at: "2026-07-16T08:00:01.000Z",
        },
        {
          id: "agent-final",
          type: "agent_message",
          thread_id: "thread-1",
          turn_id: "turn-1",
          sequence: 3,
          status: "completed",
          text: "最终答复。",
          started_at: "2026-07-16T08:00:02.000Z",
          updated_at: "2026-07-16T08:00:03.000Z",
        },
      ],
    } as unknown as AgentSessionDetail;

    const messages = hydrateSessionDetailMessages(
      detail,
      "legacy-unphased-agent-messages",
    );
    const assistant = messages.find((message) => message.role === "assistant");

    expect(assistant?.content).toBe("最终答复。");
    expect(
      assistant?.contentParts
        ?.filter((part) => part.type === "text")
        .map((part) => part.text),
    ).toEqual(["我先核对当前实现。", "最终答复。"]);
  });

  it("reasoning 的完成 sequence 晚于回答时仍按 canonical ordinal 展示在回答前", () => {
    const detail = {
      id: "canonical-history-order-session",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "canonical-history-order-turn",
          thread_id: "canonical-history-order-thread",
          prompt_text: "请先思考再回答。",
          status: "completed",
          started_at: "2026-07-14T10:00:00.000Z",
          created_at: "2026-07-14T10:00:00.000Z",
          updated_at: "2026-07-14T10:00:05.000Z",
        },
      ],
      items: [],
      thread_read: {
        thread_id: "canonical-history-order-thread",
        status: "completed",
        thread_items: [
          {
            id: "canonical-history-order-user",
            type: "user_message",
            thread_id: "canonical-history-order-thread",
            turn_id: "canonical-history-order-turn",
            ordinal: 1,
            sequence: 1,
            status: "completed",
            content: "请先思考再回答。",
            started_at: "2026-07-14T10:00:00.000Z",
            updated_at: "2026-07-14T10:00:00.000Z",
            completed_at: "2026-07-14T10:00:00.000Z",
          },
          {
            id: "canonical-history-order-reasoning",
            type: "reasoning",
            thread_id: "canonical-history-order-thread",
            turn_id: "canonical-history-order-turn",
            ordinal: 6,
            sequence: 320,
            status: "completed",
            text: "先核对 canonical Item 的首次出现顺序。",
            started_at: "2026-07-14T10:00:01.000Z",
            updated_at: "2026-07-14T10:00:05.000Z",
            completed_at: "2026-07-14T10:00:05.000Z",
          },
          {
            id: "canonical-history-order-answer",
            type: "agent_message",
            thread_id: "canonical-history-order-thread",
            turn_id: "canonical-history-order-turn",
            ordinal: 314,
            sequence: 316,
            status: "in_progress",
            text: "最终回答。",
            started_at: "2026-07-14T10:00:04.000Z",
            updated_at: "2026-07-14T10:00:04.000Z",
          },
        ],
        tool_calls: [],
      },
    } as unknown as AgentSessionDetail;

    const messages = hydrateSessionDetailMessages(
      detail,
      "canonical-history-order-session",
    );
    const assistantMessage = messages.find(
      (message) => message.role === "assistant",
    );

    expect(messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
    ]);
    expect(assistantMessage?.contentParts?.map((part) => part.type)).toEqual([
      "thinking",
      "text",
    ]);
    expect(
      orderStreamingContentPartsForDisplay(assistantMessage?.contentParts)?.map(
        (part) => part.type,
      ),
    ).toEqual(["thinking", "text"]);
    expect(assistantMessage?.contentParts).toEqual([
      expect.objectContaining({
        type: "thinking",
        metadata: expect.objectContaining({ sequence: 6 }),
      }),
      expect.objectContaining({
        type: "text",
        metadata: expect.objectContaining({ sequence: 314 }),
      }),
    ]);
    expect(assistantMessage?.content).toBe("最终回答。");
  });

  it("历史 read model 的 thread_items 应作为 reasoning/tool owner hydrate，且不回退成重复 legacy process", () => {
    const detail = {
      id: "history-replay-visual-session",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "user",
          timestamp: 1783550000,
          content: [
            {
              type: "text",
              text: "请结合本地截图和远程参考图，先说明思路再调用 MCP。",
            },
          ],
        },
        {
          role: "assistant",
          timestamp: 1783550004,
          content: [
            {
              type: "output_text",
              text: "我会先保留 reasoning，再等待 MCP 返回后继续。",
            },
          ],
        },
      ],
      turns: [
        {
          id: "history-replay-visual-turn",
          thread_id: "history-replay-visual-session",
          prompt_text: "请结合本地截图和远程参考图，先说明思路再调用 MCP。",
          status: "running",
          started_at: "2026-07-09T10:00:00.000Z",
          created_at: "2026-07-09T10:00:00.000Z",
          updated_at: "2026-07-09T10:00:04.000Z",
        },
      ],
      items: [],
      thread_read: {
        thread_id: "history-replay-visual-session",
        status: "running",
        active_turn_id: "history-replay-visual-turn",
        thread_items: [
          {
            id: "history-replay-visual-reasoning",
            type: "reasoning",
            thread_id: "history-replay-visual-session",
            turn_id: "history-replay-visual-turn",
            sequence: 3,
            status: "completed",
            text: "先确认本地图片和远程参考图都应作为结构化输入恢复。",
            started_at: "2026-07-09T10:00:02.000Z",
            updated_at: "2026-07-09T10:00:02.000Z",
            completed_at: "2026-07-09T10:00:02.000Z",
            metadata: {
              source: "codex_history_replay_visual",
              source_event_id: "history-replay-visual-reasoning",
            },
          },
          {
            id: "history-replay-visual-mcp-read-file",
            type: "tool_call",
            thread_id: "history-replay-visual-session",
            turn_id: "history-replay-visual-turn",
            sequence: 4,
            status: "in_progress",
            tool_name: "mcp__filesystem__read_file",
            arguments: {
              path: "/workspace/README.md",
            },
            started_at: "2026-07-09T10:00:03.000Z",
            updated_at: "2026-07-09T10:00:03.000Z",
            metadata: {
              owner: "history_replay_visual",
              source_event_id: "history-replay-visual-mcp-read-file",
            },
          },
        ],
        tool_calls: [
          {
            tool_call_id: "history-replay-visual-mcp-read-file",
            turn_id: "history-replay-visual-turn",
            tool_name: "mcp__filesystem__read_file",
            status: "running",
            started_at: "2026-07-09T10:00:03.000Z",
            arguments: {
              path: "/workspace/README.md",
            },
          },
        ],
      },
    } as unknown as AgentSessionDetail;

    const messages = mergeThreadItemReasoningIntoMessages(
      hydrateSessionDetailMessages(detail, "history-replay-visual-session"),
      collectDetailThreadItems(detail),
    );
    const assistantMessage = messages.find(
      (message) => message.role === "assistant",
    );
    const thinkingParts =
      assistantMessage?.contentParts?.filter(
        (part) => part.type === "thinking",
      ) ?? [];
    const toolParts =
      assistantMessage?.contentParts?.filter(
        (part) => part.type === "tool_use",
      ) ?? [];

    expect(thinkingParts).toHaveLength(1);
    expect(thinkingParts[0]).toMatchObject({
      type: "thinking",
      text: "先确认本地图片和远程参考图都应作为结构化输入恢复。",
      metadata: {
        source: "thread_item_reasoning",
        threadItemId: "history-replay-visual-reasoning",
        turnId: "history-replay-visual-turn",
        sequence: 3,
      },
    });
    expect(toolParts).toHaveLength(1);
    expect(toolParts[0]).toMatchObject({
      type: "tool_use",
      metadata: {
        source: "agent_thread_item",
        threadItemId: "history-replay-visual-mcp-read-file",
        turnId: "history-replay-visual-turn",
        sequence: 4,
      },
      toolCall: {
        id: "history-replay-visual-mcp-read-file",
        name: "mcp__filesystem__read_file",
        status: "running",
        metadata: {
          source: "agent_thread_item",
          threadItemId: "history-replay-visual-mcp-read-file",
          turnId: "history-replay-visual-turn",
          sequence: 4,
        },
      },
    });
    expect(assistantMessage?.toolCalls).toHaveLength(1);
  });
});
