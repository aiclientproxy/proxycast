import { describe, expect, it } from "vitest";
import {
  mockStreamingRenderer,
  mockAgentThreadTimeline,
  render,
} from "./MessageList.testHarness";
import type { Message } from "./MessageList.testHarness";

describe("MessageList reasoning persistence", () => {
  it("当前完成回合缺少持久化 reasoning 时应只保留 final 正文", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-user-thinking-fallback",
        role: "user",
        content: "先分析再回答",
        timestamp: now,
      },
      {
        id: "msg-assistant-thinking-fallback",
        role: "assistant",
        content: "最终说明",
        timestamp: now,
        thinkingContent: "先分析意图。",
        contentParts: [
          {
            type: "thinking",
            text: "先分析意图。",
          },
          {
            type: "text",
            text: "最终说明",
          },
        ],
      },
    ];

    render(messages, {
      currentTurnId: "turn-thinking-fallback",
      turns: [
        {
          id: "turn-thinking-fallback",
          thread_id: "thread-1",
          prompt_text: "先分析再回答",
          status: "completed",
          started_at: "2026-03-28T12:00:00Z",
          completed_at: "2026-03-28T12:00:02Z",
          created_at: "2026-03-28T12:00:00Z",
          updated_at: "2026-03-28T12:00:02Z",
        },
      ],
      threadItems: [],
    });

    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({
        thinkingContent: undefined,
        contentParts: [{ type: "text", text: "最终说明" }],
      }),
    );
  });

  it("当前尾部 assistant 已完成但 reasoning 尚未持久化时也应只保留 final 正文", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-user-tail-thinking-fallback",
        role: "user",
        content: "帮我分析一下",
        timestamp: now,
      },
      {
        id: "msg-assistant-tail-thinking-fallback",
        role: "assistant",
        content: "这是最终回答。",
        timestamp: now,
        thinkingContent: "先列提纲，再组织答案。",
        contentParts: [
          {
            type: "thinking",
            text: "先列提纲，再组织答案。",
          },
          {
            type: "text",
            text: "这是最终回答。",
          },
        ],
      },
    ];

    render(messages, {
      currentTurnId: null,
      turns: [
        {
          id: "turn-tail-thinking-fallback",
          thread_id: "thread-1",
          prompt_text: "帮我分析一下",
          status: "completed",
          started_at: "2026-03-28T12:00:00Z",
          completed_at: "2026-03-28T12:00:02Z",
          created_at: "2026-03-28T12:00:00Z",
          updated_at: "2026-03-28T12:00:02Z",
        },
      ],
      threadItems: [],
    });

    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({
        thinkingContent: undefined,
        contentParts: [{ type: "text", text: "这是最终回答。" }],
      }),
    );
  });

  it("当前尾部 assistant 完成后 turn timeline 暂缺时应继续显示本地思考内容", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-user-tail-runtime-thinking-fallback",
        role: "user",
        content: "先思考再回答",
        timestamp: now,
      },
      {
        id: "msg-assistant-tail-runtime-thinking-fallback",
        role: "assistant",
        content: "这是最终回答。",
        timestamp: now,
        runtimeTurnId: "turn-tail-runtime-thinking-fallback",
        thinkingContent: "先梳理约束。",
        contentParts: [
          {
            type: "thinking",
            text: "先梳理约束。",
          },
          {
            type: "text",
            text: "这是最终回答。",
          },
        ],
      },
    ];

    render(messages, {
      currentTurnId: null,
      turns: [],
      threadItems: [],
    });

    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({
        thinkingContent: "先梳理约束。",
        contentParts: [
          { type: "thinking", text: "先梳理约束。" },
          { type: "text", text: "这是最终回答。" },
        ],
      }),
    );
  });

  it("已完成的直执 Skill 消息即使不在尾部，也不应丢失本地思考内容", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-user-skill-inline-process",
        role: "user",
        content: "整理产品资料",
        timestamp: now,
      },
      {
        id: "msg-assistant-skill-inline-process",
        role: "assistant",
        content: "产品知识库说明。",
        timestamp: now,
        runtimeTurnId: "skill-exec-msg-assistant-skill-inline-process",
        thinkingContent: "先读取 Skill，再分析产品资料边界。",
        contentParts: [
          {
            type: "thinking",
            text: "先读取 Skill，再分析产品资料边界。",
          },
          {
            type: "text",
            text: "产品知识库说明。",
          },
        ],
      },
      {
        id: "msg-user-after-skill-inline-process",
        role: "user",
        content: "继续",
        timestamp: now,
      },
      {
        id: "msg-assistant-after-skill-inline-process",
        role: "assistant",
        content: "继续回答。",
        timestamp: now,
      },
    ];

    render(messages, {
      currentTurnId: null,
      turns: [],
      threadItems: [],
    });

    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({
        content: "产品知识库说明。",
        thinkingContent: "先读取 Skill，再分析产品资料边界。",
        contentParts: [
          {
            type: "thinking",
            text: "先读取 Skill，再分析产品资料边界。",
          },
          {
            type: "text",
            text: "产品知识库说明。",
          },
        ],
      }),
    );
  });

  it("已完成的服务型 Skill 消息即使不在尾部，也不应丢失本地思考内容", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-user-service-skill-inline-process",
        role: "user",
        content: "整理产品资料",
        timestamp: now,
      },
      {
        id: "msg-assistant-service-skill-inline-process",
        role: "assistant",
        content: "产品知识库说明。",
        timestamp: now,
        runtimeTurnId: "turn-service-skill-inline-process",
        inlineProcessRetention: "skill",
        thinkingContent: "先读取服务型 Skill，再分析产品资料边界。",
        contentParts: [
          {
            type: "thinking",
            text: "先读取服务型 Skill，再分析产品资料边界。",
          },
          {
            type: "text",
            text: "产品知识库说明。",
          },
        ],
      },
      {
        id: "msg-user-after-service-skill-inline-process",
        role: "user",
        content: "继续",
        timestamp: now,
      },
      {
        id: "msg-assistant-after-service-skill-inline-process",
        role: "assistant",
        content: "继续回答。",
        timestamp: now,
      },
    ];

    render(messages, {
      currentTurnId: null,
      turns: [],
      threadItems: [],
    });

    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({
        content: "产品知识库说明。",
        thinkingContent: "先读取服务型 Skill，再分析产品资料边界。",
        contentParts: [
          {
            type: "thinking",
            text: "先读取服务型 Skill，再分析产品资料边界。",
          },
          {
            type: "text",
            text: "产品知识库说明。",
          },
        ],
      }),
    );
  });

  it("已完成的直执 Skill 消息已有 turn timeline 时应折叠本地思考内容", () => {
    const now = new Date();
    const turnId = "skill-exec-analysis-retained";
    const messages: Message[] = [
      {
        id: "msg-user-skill-retained-with-timeline",
        role: "user",
        content: "@analysis 帮我分析一下今天的国际形势",
        timestamp: now,
      },
      {
        id: "msg-assistant-skill-retained-with-timeline",
        role: "assistant",
        content: "国际形势分析结果。",
        timestamp: now,
        runtimeTurnId: turnId,
        inlineProcessRetention: "skill",
        thinkingContent: "先读取 analysis Skill，再拆解地区变量。",
        contentParts: [
          {
            type: "thinking",
            text: "先读取 analysis Skill，再拆解地区变量。",
          },
          {
            type: "text",
            text: "国际形势分析结果。",
          },
        ],
      },
    ];

    const container = render(messages, {
      currentTurnId: turnId,
      turns: [
        {
          id: turnId,
          thread_id: "thread-1",
          prompt_text: "@analysis 帮我分析一下今天的国际形势",
          status: "completed",
          started_at: "2026-05-13T12:00:00Z",
          completed_at: "2026-05-13T12:00:02Z",
          created_at: "2026-05-13T12:00:00Z",
          updated_at: "2026-05-13T12:00:02Z",
        },
      ],
      threadItems: [
        {
          id: "reasoning-skill-retained-with-timeline",
          thread_id: "thread-1",
          turn_id: turnId,
          sequence: 1,
          status: "completed",
          started_at: "2026-05-13T12:00:00Z",
          completed_at: "2026-05-13T12:00:01Z",
          updated_at: "2026-05-13T12:00:01Z",
          type: "reasoning",
          text: "先读取 analysis Skill，再拆解地区变量。",
        },
      ],
    });

    expect(
      container.querySelector(
        '[data-testid="message-list-historical-timeline-preview:leading"]',
      ),
    ).not.toBeNull();
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({
        content: "国际形势分析结果。",
        thinkingContent: undefined,
        contentParts: [{ type: "text", text: "国际形势分析结果。" }],
      }),
    );
  });

  it("当前尾部 assistant 完成后 turn 记录暂缺时应按 runtimeTurnId 保留 compact 过程摘要", () => {
    const now = new Date("2026-05-12T10:00:02.000Z");
    const messages: Message[] = [
      {
        id: "msg-user-orphan-runtime-timeline",
        role: "user",
        content: "保留过程",
        timestamp: new Date("2026-05-12T10:00:00.000Z"),
      },
      {
        id: "msg-assistant-orphan-runtime-timeline",
        role: "assistant",
        content: "最终回答。",
        timestamp: now,
        runtimeTurnId: "turn-orphan-runtime-timeline",
      },
    ];

    const container = render(messages, {
      currentTurnId: null,
      turns: [],
      threadItems: [
        {
          id: "reasoning-orphan-runtime-timeline",
          thread_id: "thread-1",
          turn_id: "turn-orphan-runtime-timeline",
          sequence: 1,
          status: "completed",
          started_at: "2026-05-12T10:00:01.000Z",
          completed_at: "2026-05-12T10:00:02.000Z",
          updated_at: "2026-05-12T10:00:02.000Z",
          type: "reasoning",
          text: "先确认过程是否还在。",
        },
      ],
    });

    expect(
      container.querySelector(
        '[data-testid="message-list-historical-timeline-preview:leading"]',
      ),
    ).not.toBeNull();
    expect(
      container
        .querySelector('[data-testid="message-turn-group"]')
        ?.getAttribute("data-runtime-turn-id"),
    ).toBe("turn-orphan-runtime-timeline");
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
    expect(container.textContent).not.toContain("先确认过程是否还在");
  });

  it("消息内已有思考顺序时终态应由 compact 过程摘要承载", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-user-thinking-persisted",
        role: "user",
        content: "先分析再回答",
        timestamp: now,
      },
      {
        id: "msg-assistant-thinking-persisted",
        role: "assistant",
        content: "最终说明",
        timestamp: now,
        thinkingContent: "先分析意图。",
        contentParts: [
          {
            type: "thinking",
            text: "先分析意图。",
          },
          {
            type: "text",
            text: "最终说明",
          },
        ],
      },
    ];

    const container = render(messages, {
      currentTurnId: "turn-thinking-persisted",
      turns: [
        {
          id: "turn-thinking-persisted",
          thread_id: "thread-1",
          prompt_text: "先分析再回答",
          status: "completed",
          started_at: "2026-03-28T12:00:00Z",
          completed_at: "2026-03-28T12:00:02Z",
          created_at: "2026-03-28T12:00:00Z",
          updated_at: "2026-03-28T12:00:02Z",
        },
      ],
      threadItems: [
        {
          id: "reasoning-thinking-persisted",
          thread_id: "thread-1",
          turn_id: "turn-thinking-persisted",
          sequence: 1,
          status: "completed",
          started_at: "2026-03-28T12:00:00Z",
          completed_at: "2026-03-28T12:00:01Z",
          updated_at: "2026-03-28T12:00:01Z",
          type: "reasoning",
          text: "先分析意图。",
          summary: ["先分析意图。"],
        },
      ],
    });

    expect(
      container.querySelector(
        '[data-testid="message-list-historical-timeline-preview:leading"]',
      ),
    ).not.toBeNull();
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
    expect(
      container.querySelector(
        '[data-testid="historical-reasoning-summary:leading"]',
      )?.textContent,
    ).toContain("先分析意图。");
    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({
        thinkingContent: undefined,
        contentParts: [{ type: "text", text: "最终说明" }],
      }),
    );
  });

  it("完成态 timeline 多段 reasoning 应按 sequence 收拢为 compact 摘要", () => {
    const now = new Date("2026-05-30T09:10:00.000Z");
    const messages: Message[] = [
      {
        id: "msg-user-timeline-interleaved-reasoning",
        role: "user",
        content: "帮我分析一下这个文件夹",
        timestamp: now,
      },
      {
        id: "msg-assistant-timeline-interleaved-reasoning",
        role: "assistant",
        content: "我先围绕你给出的路径做只读侦查。\n\n已确认该目录存在。",
        timestamp: new Date("2026-05-30T09:10:05.000Z"),
      },
    ];

    const container = render(messages, {
      currentTurnId: "turn-timeline-interleaved-reasoning",
      turns: [
        {
          id: "turn-timeline-interleaved-reasoning",
          thread_id: "thread-timeline-interleaved-reasoning",
          prompt_text: "帮我分析一下这个文件夹",
          status: "completed",
          started_at: "2026-05-30T09:10:00.000Z",
          completed_at: "2026-05-30T09:10:05.000Z",
          created_at: "2026-05-30T09:10:00.000Z",
          updated_at: "2026-05-30T09:10:05.000Z",
        },
      ],
      threadItems: [
        {
          id: "reasoning-timeline-interleaved-1",
          thread_id: "thread-timeline-interleaved-reasoning",
          turn_id: "turn-timeline-interleaved-reasoning",
          sequence: 1,
          status: "completed",
          started_at: "2026-05-30T09:10:00.500Z",
          completed_at: "2026-05-30T09:10:01.000Z",
          updated_at: "2026-05-30T09:10:01.000Z",
          type: "reasoning",
          text: "Inspecting folder for details",
          summary: ["Inspecting folder for details"],
        },
        {
          id: "agent-timeline-interleaved-1",
          thread_id: "thread-timeline-interleaved-reasoning",
          turn_id: "turn-timeline-interleaved-reasoning",
          sequence: 2,
          status: "completed",
          started_at: "2026-05-30T09:10:01.000Z",
          completed_at: "2026-05-30T09:10:01.500Z",
          updated_at: "2026-05-30T09:10:01.500Z",
          type: "agent_message",
          text: "我先围绕你给出的路径做只读侦查。",
        },
        {
          id: "tool-timeline-interleaved-1",
          thread_id: "thread-timeline-interleaved-reasoning",
          turn_id: "turn-timeline-interleaved-reasoning",
          sequence: 3,
          status: "completed",
          started_at: "2026-05-30T09:10:01.500Z",
          completed_at: "2026-05-30T09:10:02.000Z",
          updated_at: "2026-05-30T09:10:02.000Z",
          type: "command_execution",
          command: "ls /Users/coso/yansu-agent",
          cwd: "/Users/coso",
          aggregated_output: "activity models sherpa bin",
          exit_code: 0,
        },
        {
          id: "reasoning-timeline-interleaved-2",
          thread_id: "thread-timeline-interleaved-reasoning",
          turn_id: "turn-timeline-interleaved-reasoning",
          sequence: 4,
          status: "completed",
          started_at: "2026-05-30T09:10:02.500Z",
          completed_at: "2026-05-30T09:10:03.000Z",
          updated_at: "2026-05-30T09:10:03.000Z",
          type: "reasoning",
          text: "Analyzing file sizes",
          summary: ["Analyzing file sizes"],
        },
        {
          id: "agent-timeline-interleaved-2",
          thread_id: "thread-timeline-interleaved-reasoning",
          turn_id: "turn-timeline-interleaved-reasoning",
          sequence: 5,
          status: "completed",
          started_at: "2026-05-30T09:10:04.000Z",
          completed_at: "2026-05-30T09:10:05.000Z",
          updated_at: "2026-05-30T09:10:05.000Z",
          type: "agent_message",
          text: "已确认该目录存在。",
        },
      ],
    });

    expect(
      container.querySelector(
        '[data-testid="message-list-historical-timeline-preview:leading"]',
      ),
    ).not.toBeNull();
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({
        thinkingContent: undefined,
        contentParts: expect.arrayContaining([
          expect.objectContaining({
            type: "text",
            text: "已确认该目录存在。",
          }),
        ]),
      }),
    );
    const interleavedCall = mockStreamingRenderer.mock.calls.at(-1)?.[0] as {
      contentParts?: Array<{ type: string; text?: string }>;
    };
    expect(
      interleavedCall.contentParts?.some(
        (part) => part.type === "tool_use" || part.type === "thinking",
      ),
    ).toBe(false);
    const reasoningSummary = container.querySelector(
      '[data-testid="historical-reasoning-summary:leading"]',
    );
    expect(reasoningSummary?.textContent).toContain(
      "Inspecting folder for details",
    );
    expect(reasoningSummary?.textContent).toContain("Analyzing file sizes");
  });

  it("已完成短答也应把持久化 reasoning 收拢到 compact 执行摘要", () => {
    const now = new Date("2026-05-09T06:02:56.361Z");
    const messages: Message[] = [
      {
        id: "msg-user-fast-plain-answer",
        role: "user",
        content: "只回答一个字：好",
        timestamp: new Date("2026-05-09T06:02:54.927Z"),
      },
      {
        id: "msg-assistant-fast-plain-answer",
        role: "assistant",
        content: "好",
        timestamp: now,
      },
    ];

    const container = render(messages, {
      currentTurnId: "turn-fast-plain-answer",
      turns: [
        {
          id: "turn-fast-plain-answer",
          thread_id: "thread-fast-plain-answer",
          prompt_text: "只回答一个字：好",
          status: "completed",
          started_at: "2026-05-09T06:02:54.278Z",
          completed_at: "2026-05-09T06:02:56.366Z",
          created_at: "2026-05-09T06:02:54.278Z",
          updated_at: "2026-05-09T06:02:56.366Z",
        },
      ],
      threadItems: [
        {
          id: "turn-summary-fast-plain-answer",
          thread_id: "thread-fast-plain-answer",
          turn_id: "turn-fast-plain-answer",
          sequence: 1,
          status: "completed",
          started_at: "2026-05-09T06:02:54.281Z",
          completed_at: "2026-05-09T06:02:56.365Z",
          updated_at: "2026-05-09T06:02:56.365Z",
          type: "turn_summary",
          text: "直接回答优先\n当前请求无需默认升级为搜索或任务，先直接给出结果，必要时再调用工具。",
          metadata: {
            sourceType: "runtime_status",
            surface: "runtime_status",
            visibility: "diagnostics",
            persistence: "transient",
          },
        },
        {
          id: "user-fast-plain-answer",
          thread_id: "thread-fast-plain-answer",
          turn_id: "turn-fast-plain-answer",
          sequence: 2,
          status: "completed",
          started_at: "2026-05-09T06:02:54.278Z",
          completed_at: "2026-05-09T06:02:54.927Z",
          updated_at: "2026-05-09T06:02:54.927Z",
          type: "user_message",
          content: "只回答一个字：好",
        },
        {
          id: "reasoning-fast-plain-answer",
          thread_id: "thread-fast-plain-answer",
          turn_id: "turn-fast-plain-answer",
          sequence: 3,
          status: "completed",
          started_at: "2026-05-09T06:02:55.716Z",
          completed_at: "2026-05-09T06:02:56.361Z",
          updated_at: "2026-05-09T06:02:56.361Z",
          type: "reasoning",
          text: "我们被要求只回答一个字：好。直接回复即可。",
          summary: ["我们被要求只回答一个字：好。直接回复即可。"],
        },
        {
          id: "assistant-fast-plain-answer",
          thread_id: "thread-fast-plain-answer",
          turn_id: "turn-fast-plain-answer",
          sequence: 4,
          status: "completed",
          started_at: "2026-05-09T06:02:56.289Z",
          completed_at: "2026-05-09T06:02:56.361Z",
          updated_at: "2026-05-09T06:02:56.361Z",
          type: "agent_message",
          text: "好",
        },
      ],
    });

    expect(
      container.querySelector(
        '[data-testid="message-list-historical-timeline-preview:leading"]',
      ),
    ).not.toBeNull();
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
    expect(container.textContent).toContain("好");
    expect(container.textContent).toContain("Processed for 2s");
    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({
        content: "好",
        thinkingContent: undefined,
        contentParts: [expect.objectContaining({ type: "text", text: "好" })],
      }),
    );
  });

  it("历史对话恢复时也应保留已持久化 reasoning 的 compact 摘要", () => {
    const now = new Date("2026-05-09T06:02:56.361Z");
    const messages: Message[] = [
      {
        id: "msg-user-history-reasoning",
        role: "user",
        content: "只回答一个字：好",
        timestamp: new Date("2026-05-09T06:02:54.927Z"),
      },
      {
        id: "msg-assistant-history-reasoning",
        role: "assistant",
        content: "好",
        timestamp: now,
      },
    ];

    const container = render(messages, {
      isRestoringSession: true,
      turns: [
        {
          id: "turn-history-reasoning",
          thread_id: "thread-history-reasoning",
          prompt_text: "只回答一个字：好",
          status: "completed",
          started_at: "2026-05-09T06:02:54.278Z",
          completed_at: "2026-05-09T06:02:56.366Z",
          created_at: "2026-05-09T06:02:54.278Z",
          updated_at: "2026-05-09T06:02:56.366Z",
        },
      ],
      threadItems: [
        {
          id: "reasoning-history-answer",
          thread_id: "thread-history-reasoning",
          turn_id: "turn-history-reasoning",
          sequence: 1,
          status: "completed",
          started_at: "2026-05-09T06:02:55.716Z",
          completed_at: "2026-05-09T06:02:56.361Z",
          updated_at: "2026-05-09T06:02:56.361Z",
          type: "reasoning",
          text: "我们被要求只回答一个字：好。直接回复即可。",
          summary: ["我们被要求只回答一个字：好。直接回复即可。"],
        },
        {
          id: "assistant-history-answer",
          thread_id: "thread-history-reasoning",
          turn_id: "turn-history-reasoning",
          sequence: 2,
          status: "completed",
          started_at: "2026-05-09T06:02:56.289Z",
          completed_at: "2026-05-09T06:02:56.361Z",
          updated_at: "2026-05-09T06:02:56.361Z",
          type: "agent_message",
          text: "好",
        },
      ],
    });

    expect(
      container.querySelector(
        '[data-testid="message-list-historical-timeline-preview:leading"]',
      ),
    ).not.toBeNull();
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({
        content: "好",
        thinkingContent: undefined,
      }),
    );
  });
});
