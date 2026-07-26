import { describe, expect, it } from "vitest";
import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";
import type { AgentThreadTurn, Message } from "../types";
import { buildHydratedAgentSessionSnapshot } from "./agentSessionState";

function createMessage(overrides: Partial<Message> = {}): Message {
  return {
    id: overrides.id ?? "message-1",
    role: overrides.role ?? "assistant",
    content: overrides.content ?? "默认内容",
    timestamp: overrides.timestamp ?? new Date("2026-03-29T00:00:00.000Z"),
    ...overrides,
  };
}

function createTurn(overrides: Partial<AgentThreadTurn> = {}): AgentThreadTurn {
  return {
    id: overrides.id ?? "turn-1",
    thread_id: overrides.thread_id ?? "thread-1",
    status: overrides.status ?? "completed",
    prompt_text: overrides.prompt_text ?? "默认 turn",
    started_at: overrides.started_at ?? "2026-03-29T00:00:00.000Z",
    completed_at: overrides.completed_at ?? "2026-03-29T00:00:02.000Z",
    created_at: overrides.created_at ?? "2026-03-29T00:00:00.000Z",
    updated_at: overrides.updated_at ?? "2026-03-29T00:00:02.000Z",
    ...overrides,
  };
}

describe("agentSessionState local snapshot hydrate", () => {
  it("首次按 restore 候选 hydrate 时也应合并本地缓存而不是整段覆盖", () => {
    const now = new Date("2026-04-08T10:00:02.000Z");
    const currentMessages = [
      createMessage({
        id: "local-user",
        role: "user",
        content: "把文章保存到项目里",
        timestamp: new Date("2026-04-08T10:00:00.000Z"),
      }),
      createMessage({
        id: "local-assistant",
        role: "assistant",
        content: "内容已保存到项目目录。",
        timestamp: now,
        contentParts: [
          {
            type: "tool_use",
            toolCall: {
              id: "tool-site-restore",
              name: "site_run_adapter",
              arguments: '{"url":"https://x.com/example/article/3"}',
              status: "completed",
              startTime: new Date("2026-04-08T10:00:01.000Z"),
              endTime: now,
              result: {
                success: true,
                output: "saved: articles/google-cloud-tech-3.md",
              },
            },
          },
        ],
        toolCalls: [
          {
            id: "tool-site-restore",
            name: "site_run_adapter",
            arguments: '{"url":"https://x.com/example/article/3"}',
            status: "completed",
            startTime: new Date("2026-04-08T10:00:01.000Z"),
            endTime: now,
            result: {
              success: true,
              output: "saved: articles/google-cloud-tech-3.md",
            },
          },
        ],
      }),
    ];
    const detail = {
      id: "topic-restore",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [
        {
          role: "user",
          timestamp: 1710000001,
          content: [{ type: "text", text: "把文章保存到项目里" }],
        },
      ],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-restore",
      detail,
      currentSessionId: null,
      currentMessages,
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      syncSessionId: true,
    });

    expect(result.snapshot.sessionId).toBe("topic-restore");
    expect(result.snapshot.messages).toHaveLength(2);
    expect(result.snapshot.messages[1]?.toolCalls?.[0]?.id).toBe(
      "tool-site-restore",
    );
  });

  it("切回其他历史会话时也应优先合并目标会话自己的本地快照", () => {
    const now = new Date("2026-04-08T10:00:02.000Z");
    const detail = {
      id: "topic-history-target",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [
        {
          role: "user",
          timestamp: 1710000001,
          content: [{ type: "text", text: "把文章保存到项目里" }],
        },
      ],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-history-target",
      detail,
      currentSessionId: "topic-other",
      currentMessages: [
        createMessage({
          id: "other-session-message",
          role: "assistant",
          content: "这是另一个会话，不应参与合并",
        }),
      ],
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      localSnapshotOverride: {
        sessionId: "topic-history-target",
        messages: [
          createMessage({
            id: "local-user-target",
            role: "user",
            content: "把文章保存到项目里",
            timestamp: new Date("2026-04-08T10:00:00.000Z"),
          }),
          createMessage({
            id: "local-assistant-target",
            role: "assistant",
            content: "内容已保存到项目目录。",
            timestamp: now,
            toolCalls: [
              {
                id: "tool-site-target",
                name: "site_run_adapter",
                arguments: '{"url":"https://x.com/example/article/4"}',
                status: "completed",
                startTime: new Date("2026-04-08T10:00:01.000Z"),
                endTime: now,
                result: {
                  success: true,
                  output: "saved: articles/google-cloud-tech-4.md",
                },
              },
            ],
            contentParts: [
              {
                type: "tool_use",
                toolCall: {
                  id: "tool-site-target",
                  name: "site_run_adapter",
                  arguments: '{"url":"https://x.com/example/article/4"}',
                  status: "completed",
                  startTime: new Date("2026-04-08T10:00:01.000Z"),
                  endTime: now,
                  result: {
                    success: true,
                    output: "saved: articles/google-cloud-tech-4.md",
                  },
                },
              },
            ],
          }),
        ],
        threadTurns: [],
        threadItems: [],
      },
      syncSessionId: true,
    });

    expect(result.snapshot.messages).toHaveLength(2);
    expect(result.snapshot.messages[1]?.content).toBe("内容已保存到项目目录。");
    expect(result.snapshot.messages[1]?.thinkingContent).toBeUndefined();
    expect(result.snapshot.messages[1]?.toolCalls?.[0]?.id).toBe(
      "tool-site-target",
    );
    expect(result.snapshot.messages[1]?.contentParts).toEqual([
      {
        type: "text",
        text: "内容已保存到项目目录。",
      },
    ]);
    expect(result.snapshot.messages[1]?.content).not.toContain(
      "这是另一个会话",
    );
  });

  it("本地快照首轮用户指令与远端不一致时应丢弃快照，避免历史会话串线", () => {
    const detail = {
      id: "topic-image-lemonade",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [],
      turns: [
        createTurn({
          id: "turn-lemonade",
          thread_id: "topic-image-lemonade",
          prompt_text: "@配图 生成一张极简线稿风的柠檬水杯配图，1:1",
          started_at: "2026-05-14T05:20:00.000Z",
          completed_at: "2026-05-14T05:20:01.000Z",
          created_at: "2026-05-14T05:20:00.000Z",
          updated_at: "2026-05-14T05:20:01.000Z",
        }),
      ],
      items: [],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-image-lemonade",
      detail,
      currentSessionId: "topic-image-lemonade",
      currentMessages: [
        createMessage({
          id: "topic-image-lemonade-0",
          role: "user",
          content: "@配图 生成三张章节配图",
          timestamp: new Date("2026-05-14T05:33:22.000Z"),
        }),
        createMessage({
          id: "topic-image-lemonade-1",
          role: "assistant",
          content: "我按三个章节分别生成，方便你逐张查看。",
          timestamp: new Date("2026-05-14T05:33:23.000Z"),
          thinkingContent: "拆成三个章节画面，保持同一视觉风格。",
          contentParts: [
            {
              type: "thinking",
              text: "拆成三个章节画面，保持同一视觉风格。",
            },
            {
              type: "text",
              text: "我按三个章节分别生成，方便你逐张查看。",
            },
          ],
        }),
      ],
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      syncSessionId: true,
    });

    expect(result.snapshot.sessionId).toBe("topic-image-lemonade");
    expect(result.snapshot.messages).toHaveLength(1);
    expect(result.snapshot.messages[0]?.content).toBe(
      "@配图 生成一张极简线稿风的柠檬水杯配图，1:1",
    );
    expect(result.snapshot.messages[0]?.content).not.toContain("章节配图");
  });

  it("切回直执 Skill 历史会话时不应被远端纯正文刷新掉本地思考", () => {
    const now = new Date("2026-04-08T10:00:02.000Z");
    const detail = {
      id: "topic-skill-history-target",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [
        {
          role: "user",
          timestamp: 1710000001,
          content: [
            {
              type: "text",
              text: "请整理产品知识库",
            },
          ],
        },
        {
          role: "assistant",
          timestamp: 1710000002,
          content: [
            {
              type: "text",
              text: "最终 Skill 回复",
            },
          ],
        },
      ],
      turns: [],
      items: [],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-skill-history-target",
      detail,
      currentSessionId: "topic-other",
      currentMessages: [
        createMessage({
          id: "other-session-message",
          role: "assistant",
          content: "这是另一个会话，不应参与合并",
        }),
      ],
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      localSnapshotOverride: {
        sessionId: "topic-skill-history-target",
        messages: [
          createMessage({
            id: "local-skill-user",
            role: "user",
            content: "请整理产品知识库",
            timestamp: new Date("2026-04-08T10:00:00.000Z"),
          }),
          createMessage({
            id: "local-skill-assistant",
            role: "assistant",
            content: "最终 Skill 回复",
            timestamp: now,
            runtimeTurnId: "skill-exec-local-skill-assistant",
            thinkingContent:
              "正在执行 Skill: brand-product-knowledge-builder...",
            contentParts: [
              {
                type: "thinking",
                text: "正在执行 Skill: brand-product-knowledge-builder...",
              },
              {
                type: "text",
                text: "最终 Skill 回复",
              },
            ],
          }),
        ],
        threadTurns: [],
        threadItems: [],
      },
      syncSessionId: true,
    });

    expect(result.snapshot.messages).toHaveLength(2);
    expect(result.snapshot.messages[1]?.content).toBe("最终 Skill 回复");
    expect(result.snapshot.messages[1]?.runtimeTurnId).toBe(
      "skill-exec-local-skill-assistant",
    );
    expect(result.snapshot.messages[1]?.thinkingContent).toBe(
      "正在执行 Skill: brand-product-knowledge-builder...",
    );
    expect(result.snapshot.messages[1]?.contentParts).toEqual([
      {
        type: "thinking",
        text: "正在执行 Skill: brand-product-knowledge-builder...",
      },
      {
        type: "text",
        text: "最终 Skill 回复",
      },
    ]);
  });

  it("切回服务型 Skill 历史会话时不应被远端纯正文刷新掉本地思考", () => {
    const now = new Date("2026-04-08T10:00:03.000Z");
    const detail = {
      id: "topic-service-skill-history-target",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [
        {
          role: "user",
          timestamp: 1710000001,
          content: [
            {
              type: "text",
              text: "请整理产品知识库",
            },
          ],
        },
        {
          role: "assistant",
          timestamp: 1710000002,
          content: [
            {
              type: "text",
              text: "服务型 Skill 最终回复",
            },
          ],
        },
      ],
      turns: [],
      items: [],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-service-skill-history-target",
      detail,
      currentSessionId: "topic-other",
      currentMessages: [],
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      localSnapshotOverride: {
        sessionId: "topic-service-skill-history-target",
        messages: [
          createMessage({
            id: "local-service-skill-user",
            role: "user",
            content: "请整理产品知识库",
            timestamp: new Date("2026-04-08T10:00:00.000Z"),
          }),
          createMessage({
            id: "local-service-skill-assistant",
            role: "assistant",
            content: "服务型 Skill 最终回复",
            timestamp: now,
            runtimeTurnId: "turn-service-skill-runtime",
            inlineProcessRetention: "skill",
            thinkingContent: "先读取服务 Skill，再分析产品资料边界。",
            contentParts: [
              {
                type: "thinking",
                text: "先读取服务 Skill，再分析产品资料边界。",
              },
              {
                type: "text",
                text: "服务型 Skill 最终回复",
              },
            ],
          }),
        ],
        threadTurns: [],
        threadItems: [],
      },
      syncSessionId: true,
    });

    expect(result.snapshot.messages).toHaveLength(2);
    expect(result.snapshot.messages[1]?.content).toBe("服务型 Skill 最终回复");
    expect(result.snapshot.messages[1]?.runtimeTurnId).toBe(
      "turn-service-skill-runtime",
    );
    expect(result.snapshot.messages[1]?.inlineProcessRetention).toBe("skill");
    expect(result.snapshot.messages[1]?.thinkingContent).toBe(
      "先读取服务 Skill，再分析产品资料边界。",
    );
    expect(result.snapshot.messages[1]?.contentParts).toEqual([
      {
        type: "thinking",
        text: "先读取服务 Skill，再分析产品资料边界。",
      },
      {
        type: "text",
        text: "服务型 Skill 最终回复",
      },
    ]);
  });

  it("新建会话 ID 尚未同步时也不应让远端纯正文刷新掉本地 Skill 过程", () => {
    const now = new Date("2026-04-08T10:00:04.000Z");
    const detail = {
      id: "topic-detached-skill-history-target",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [
        {
          role: "user",
          timestamp: 1710000001,
          content: [
            {
              type: "text",
              text: "请整理产品知识库",
            },
          ],
        },
        {
          role: "assistant",
          timestamp: 1710000002,
          content: [
            {
              type: "text",
              text: "最终 Skill 回复",
            },
          ],
        },
      ],
      turns: [],
      items: [],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-detached-skill-history-target",
      detail,
      currentSessionId: null,
      currentMessages: [
        createMessage({
          id: "local-detached-skill-user",
          role: "user",
          content: "请整理产品知识库",
          timestamp: new Date("2026-04-08T10:00:00.000Z"),
        }),
        createMessage({
          id: "local-detached-skill-assistant",
          role: "assistant",
          content: "最终 Skill 回复",
          timestamp: now,
          runtimeTurnId: "skill-exec-local-detached-skill-assistant",
          thinkingContent: "正在执行 Skill: brand-product-knowledge-builder...",
          contentParts: [
            {
              type: "thinking",
              text: "正在执行 Skill: brand-product-knowledge-builder...",
            },
            {
              type: "text",
              text: "最终 Skill 回复",
            },
          ],
        }),
      ],
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
    });

    expect(result.snapshot.messages[1]).toMatchObject({
      id: "local-detached-skill-assistant",
      content: "最终 Skill 回复",
      runtimeTurnId: "skill-exec-local-detached-skill-assistant",
      thinkingContent: "正在执行 Skill: brand-product-knowledge-builder...",
    });
    expect(result.snapshot.messages[1]?.contentParts).toEqual([
      {
        type: "thinking",
        text: "正在执行 Skill: brand-product-knowledge-builder...",
      },
      {
        type: "text",
        text: "最终 Skill 回复",
      },
    ]);
  });
});
