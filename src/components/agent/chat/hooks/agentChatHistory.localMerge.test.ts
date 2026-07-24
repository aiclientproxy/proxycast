import { describe, expect, it } from "vitest";

import {
  extractThinkingContentFromParts,
  mergeAdjacentAssistantMessages,
  mergeHydratedMessagesWithLocalState,
} from "./agentChatHistory";

const INTERNAL_RUNTIME_ERROR_MESSAGE =
  "运行时返回内部错误，已保留详情用于排查。请稍后重试，或检查服务商与工具连接状态。";

describe("agentChatHistory local merge", () => {
  it("应从内容片段中提取合并后的 thinkingContent", () => {
    expect(
      extractThinkingContentFromParts([
        { type: "text", text: "正文" },
        { type: "thinking", text: "先想" },
        { type: "thinking", text: "后写" },
      ]),
    ).toBe("先想后写");
  });

  it("刷新会话详情时应保留本地用户消息里的图片", () => {
    const localMessages = [
      {
        id: "local-user-1",
        role: "user" as const,
        content: "请参考这张图继续分析",
        images: [
          {
            mediaType: "image/png",
            data: "local-image-base64",
          },
        ],
        timestamp: new Date("2026-03-19T00:00:00.000Z"),
      },
      {
        id: "local-assistant-1",
        role: "assistant" as const,
        content: "已收到，正在分析。",
        timestamp: new Date("2026-03-19T00:00:01.000Z"),
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-1",
        role: "user" as const,
        content: "请参考这张图继续分析",
        timestamp: new Date("2026-03-19T00:00:02.000Z"),
      },
      {
        id: "history-assistant-1",
        role: "assistant" as const,
        content: "已收到，正在分析。",
        timestamp: new Date("2026-03-19T00:00:03.000Z"),
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    expect(mergedMessages[0]?.images).toEqual(localMessages[0]?.images);
    expect(mergedMessages[0]?.id).toBe("local-user-1");
    expect(mergedMessages[1]?.id).toBe("local-assistant-1");
    expect(mergedMessages[1]?.images).toBeUndefined();
  });

  it("刷新会话详情合并本地状态后应继续按 runtime turn 归位", () => {
    const localMessages = [
      {
        id: "local-user-turn-1",
        role: "user" as const,
        content: "第一轮问题",
        timestamp: new Date("2026-06-18T08:00:00.000Z"),
        runtimeTurnId: "turn-merge-1",
      },
      {
        id: "local-user-turn-2",
        role: "user" as const,
        content: "第二轮问题",
        images: [
          {
            mediaType: "image/png",
            data: "image-turn-2",
            sourcePath: "/tmp/turn-2.png",
          },
        ],
        timestamp: new Date("2026-06-18T08:00:02.000Z"),
        runtimeTurnId: "turn-merge-2",
      },
      {
        id: "local-assistant-turn-1",
        role: "assistant" as const,
        content: "第一轮本地过程",
        timestamp: new Date("2026-06-18T08:00:01.000Z"),
        runtimeTurnId: "turn-merge-1",
        thinkingContent: "第一轮思考",
        contentParts: [
          {
            type: "thinking" as const,
            text: "第一轮思考",
          },
        ],
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-turn-1",
        role: "user" as const,
        content: "第一轮问题",
        timestamp: new Date("2026-06-18T08:00:00.500Z"),
        runtimeTurnId: "turn-merge-1",
      },
      {
        id: "history-user-turn-2",
        role: "user" as const,
        content: "第二轮问题",
        timestamp: new Date("2026-06-18T08:00:02.500Z"),
        runtimeTurnId: "turn-merge-2",
      },
      {
        id: "history-assistant-turn-1",
        role: "assistant" as const,
        content: "第一轮回答",
        timestamp: new Date("2026-06-18T08:00:01.000Z"),
        runtimeTurnId: "turn-merge-1",
        usage: {
          input_tokens: 10,
          output_tokens: 20,
        },
        contentParts: [
          {
            type: "text" as const,
            text: "第一轮回答",
          },
        ],
      },
      {
        id: "history-assistant-turn-2",
        role: "assistant" as const,
        content: "第二轮回答",
        timestamp: new Date("2026-06-18T08:00:03.000Z"),
        runtimeTurnId: "turn-merge-2",
        toolCalls: [
          {
            id: "tool-turn-2",
            name: "read_file",
            arguments: '{"path":"/tmp/turn-2.png"}',
            status: "completed" as const,
            startTime: new Date("2026-06-18T08:00:02.600Z"),
            endTime: new Date("2026-06-18T08:00:02.900Z"),
            result: {
              success: true,
              output: "已读取图片",
            },
          },
        ],
        contentParts: [
          {
            type: "tool_use" as const,
            toolCall: {
              id: "tool-turn-2",
              name: "read_file",
              arguments: '{"path":"/tmp/turn-2.png"}',
              status: "completed" as const,
              startTime: new Date("2026-06-18T08:00:02.600Z"),
              endTime: new Date("2026-06-18T08:00:02.900Z"),
              result: {
                success: true,
                output: "已读取图片",
              },
            },
          },
          {
            type: "text" as const,
            text: "第二轮回答",
          },
        ],
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    expect(mergedMessages.map((message) => message.content)).toEqual([
      "第一轮问题",
      "第一轮本地过程",
      "第二轮问题",
      "第二轮回答",
    ]);
    expect(mergedMessages[1]).toMatchObject({
      role: "assistant",
      runtimeTurnId: "turn-merge-1",
      usage: {
        input_tokens: 10,
        output_tokens: 20,
      },
      thinkingContent: "第一轮思考",
    });
    expect(mergedMessages[2]?.images?.[0]?.sourcePath).toBe("/tmp/turn-2.png");
    expect(mergedMessages[3]?.toolCalls?.[0]?.name).toBe("read_file");
  });

  it("相同 prompt 的增量终态 hydration 不应跨 canonical turn 覆盖上一轮", () => {
    const localMessages = [
      {
        id: "local-user-first-greeting",
        role: "user" as const,
        content: "你好",
        timestamp: new Date("2026-07-23T06:49:00.000Z"),
        runtimeTurnId: "turn-first-greeting",
      },
      {
        id: "local-assistant-first-greeting",
        role: "assistant" as const,
        content: "你好，第一轮已经完成。",
        timestamp: new Date("2026-07-23T06:49:01.000Z"),
        runtimeTurnId: "turn-first-greeting",
      },
      {
        id: "local-user-second-greeting",
        role: "user" as const,
        content: "你好",
        timestamp: new Date("2026-07-23T06:50:00.000Z"),
        runtimeTurnId: "pending-turn:second-greeting",
      },
      {
        id: "local-assistant-second-greeting",
        role: "assistant" as const,
        content: "生成中",
        timestamp: new Date("2026-07-23T06:50:00.100Z"),
        runtimeTurnId: "pending-turn:second-greeting",
        isThinking: true,
        runtimeStatus: {
          phase: "generating" as const,
          title: "正在输出",
        },
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-second-greeting",
        role: "user" as const,
        content: "你好",
        timestamp: new Date("2026-07-23T06:50:00.010Z"),
        runtimeTurnId: "turn-second-greeting",
      },
      {
        id: "history-assistant-second-greeting",
        role: "assistant" as const,
        content: "你好，第二轮也已经完成。",
        timestamp: new Date("2026-07-23T06:50:01.000Z"),
        runtimeTurnId: "turn-second-greeting",
        isThinking: false,
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
      { preferHydratedAssistantOutput: true },
    );

    expect(
      mergedMessages.map(({ id, content, runtimeTurnId }) => ({
        id,
        content,
        runtimeTurnId,
      })),
    ).toEqual([
      {
        id: "local-user-first-greeting",
        content: "你好",
        runtimeTurnId: "turn-first-greeting",
      },
      {
        id: "local-assistant-first-greeting",
        content: "你好，第一轮已经完成。",
        runtimeTurnId: "turn-first-greeting",
      },
      {
        id: "local-user-second-greeting",
        content: "你好",
        runtimeTurnId: "turn-second-greeting",
      },
      {
        id: "local-assistant-second-greeting",
        content: "你好，第二轮也已经完成。",
        runtimeTurnId: "turn-second-greeting",
      },
    ]);
    expect(mergedMessages[3]?.isThinking).toBe(false);
    expect(mergedMessages[3]?.runtimeStatus).toBeUndefined();
  });

  it("刷新会话详情时不应用完成标记覆盖同 turn 本地正文", () => {
    const turnId = "turn-news-fixture";
    const localMessages = [
      {
        id: "local-user-news",
        role: "user" as const,
        content: "整理今天的国际新闻",
        timestamp: new Date("2026-06-29T04:26:20.000Z"),
        runtimeTurnId: turnId,
      },
      {
        id: "local-assistant-news",
        role: "assistant" as const,
        content:
          "1. 多国外交议题持续升温，地区安全与经贸协商仍是焦点。\n2. 全球市场继续关注能源、供应链和主要央行政策变化。\n3. 国际组织呼吁在气候、粮食与人道援助议题上保持协调。\n",
        timestamp: new Date("2026-06-29T04:26:20.540Z"),
        runtimeTurnId: turnId,
        contentParts: [
          {
            type: "text" as const,
            text: "1. 多国外交议题持续升温，地区安全与经贸协商仍是焦点。\n2. 全球市场继续关注能源、供应链和主要央行政策变化。\n3. 国际组织呼吁在气候、粮食与人道援助议题上保持协调。\n",
          },
        ],
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-news",
        role: "user" as const,
        content: "整理今天的国际新闻",
        timestamp: new Date("2026-06-29T04:26:20.100Z"),
        runtimeTurnId: turnId,
      },
      {
        id: "history-assistant-news",
        role: "assistant" as const,
        content: "CLAW_NEWS_FIXTURE_DONE",
        timestamp: new Date("2026-06-29T04:26:20.700Z"),
        runtimeTurnId: turnId,
        contentParts: [
          {
            type: "text" as const,
            text: "CLAW_NEWS_FIXTURE_DONE",
          },
        ],
        usage: {
          input_tokens: 120,
          output_tokens: 24,
        },
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    expect(mergedMessages[1]?.content).toContain("全球市场继续关注能源");
    expect(mergedMessages[1]?.content).toContain("国际组织呼吁");
    expect(mergedMessages[1]?.content).not.toContain("CLAW_NEWS_FIXTURE_DONE");
    expect(mergedMessages[1]?.usage).toEqual({
      input_tokens: 120,
      output_tokens: 24,
    });
  });

  it("刷新会话详情时应保留同 turn 本地 commentary 过程文本", () => {
    const turnId = "turn-web-tools-commentary-merge";
    const localMessages = [
      {
        id: "local-user-web-tools-commentary",
        role: "user" as const,
        content: "验证网页搜索渲染",
        timestamp: new Date("2026-06-24T10:00:00.000Z"),
        runtimeTurnId: turnId,
      },
      {
        id: "local-assistant-web-tools-commentary",
        role: "assistant" as const,
        content: "网页搜索渲染结论：最终正文。",
        timestamp: new Date("2026-06-24T10:00:05.000Z"),
        runtimeTurnId: turnId,
        contentParts: [
          {
            type: "text" as const,
            text: "我先联网核实目标页面来源。",
            metadata: {
              source: "agent_text_delta",
              itemId: "agent-message-commentary",
              phase: "commentary",
              sequence: 2,
              turnId,
            },
          },
          {
            type: "tool_use" as const,
            toolCall: {
              id: "web-search",
              name: "WebSearch",
              arguments: '{"query":"Lime WebSearch rendering"}',
              status: "completed" as const,
              startTime: new Date("2026-06-24T10:00:01.000Z"),
            },
            metadata: {
              sequence: 3,
            },
          },
          {
            type: "tool_use" as const,
            toolCall: {
              id: "web-fetch",
              name: "WebFetch",
              arguments:
                '{"url":"https://example.com/lime-websearch-rendering"}',
              status: "completed" as const,
              startTime: new Date("2026-06-24T10:00:03.000Z"),
            },
            metadata: {
              sequence: 7,
            },
          },
          {
            type: "text" as const,
            text: "网页搜索渲染结论：最终正文。",
            metadata: {
              source: "agent_text_delta",
              itemId: "agent-message-final",
              phase: "final_answer",
              sequence: 10,
              turnId,
            },
          },
        ],
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-web-tools-commentary",
        role: "user" as const,
        content: "验证网页搜索渲染",
        timestamp: new Date("2026-06-24T10:00:00.000Z"),
        runtimeTurnId: turnId,
      },
      {
        id: "history-assistant-web-tools-commentary",
        role: "assistant" as const,
        content: "网页搜索渲染结论：最终正文。",
        timestamp: new Date("2026-06-24T10:00:05.000Z"),
        runtimeTurnId: turnId,
        contentParts: [
          {
            type: "tool_use" as const,
            toolCall: {
              id: "web-search",
              name: "WebSearch",
              arguments: '{"query":"Lime WebSearch rendering"}',
              status: "completed" as const,
              startTime: new Date("2026-06-24T10:00:01.000Z"),
              result: {
                success: true,
                output: "source",
              },
            },
            metadata: {
              sequence: 3,
            },
          },
          {
            type: "thinking" as const,
            text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
            metadata: {
              source: "thread_item_reasoning",
              threadItemId: "reasoning-web-tools",
              sequence: 5,
            },
          },
          {
            type: "tool_use" as const,
            toolCall: {
              id: "web-fetch",
              name: "WebFetch",
              arguments:
                '{"url":"https://example.com/lime-websearch-rendering"}',
              status: "completed" as const,
              startTime: new Date("2026-06-24T10:00:03.000Z"),
              result: {
                success: true,
                output: "page",
              },
            },
            metadata: {
              sequence: 7,
            },
          },
          {
            type: "text" as const,
            text: "网页搜索渲染结论：最终正文。",
            metadata: {
              source: "agent_thread_item",
              threadItemId: "agent-message-final",
              phase: "final_answer",
              sequence: 10,
              turnId,
            },
          },
        ],
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    const assistant = mergedMessages[1];
    expect(assistant?.id).toBe("local-assistant-web-tools-commentary");
    expect(assistant?.content).toBe("网页搜索渲染结论：最终正文。");
    expect(assistant?.contentParts?.map((part) => part.type)).toEqual([
      "text",
      "tool_use",
      "thinking",
      "tool_use",
      "text",
    ]);
    expect(assistant?.contentParts?.[0]).toMatchObject({
      type: "text",
      text: "我先联网核实目标页面来源。",
      metadata: {
        itemId: "agent-message-commentary",
        phase: "commentary",
        sequence: 2,
        turnId,
      },
    });
    expect(assistant?.thinkingContent).toBe(
      "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
    );
  });

  it("刷新会话详情时不应把已完成输出替换成后端历史投影", () => {
    const localMessages = [
      {
        id: "local-user-output",
        role: "user" as const,
        content: "请识别这张截图",
        timestamp: new Date("2026-05-12T10:00:00.000Z"),
      },
      {
        id: "local-assistant-output",
        role: "assistant" as const,
        content: "这张截图展示了 Lime 聊天区把图片恢复成了可见缩略图。",
        contentParts: [
          {
            type: "text" as const,
            text: "这张截图展示了 Lime 聊天区把图片恢复成了可见缩略图。",
          },
        ],
        timestamp: new Date("2026-05-12T10:00:02.000Z"),
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-output",
        role: "user" as const,
        content: "请识别这张截图",
        timestamp: new Date("2026-05-12T10:00:03.000Z"),
      },
      {
        id: "history-assistant-output",
        role: "assistant" as const,
        content:
          "历史消息内容过大，首屏已省略完整内容；需要时可加载完整历史查看。",
        contentParts: [
          {
            type: "text" as const,
            text: "历史消息内容过大，首屏已省略完整内容；需要时可加载完整历史查看。",
          },
        ],
        timestamp: new Date("2026-05-12T10:00:04.000Z"),
        usage: {
          input_tokens: 1200,
          output_tokens: 80,
        },
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    expect(mergedMessages).toHaveLength(2);
    expect(mergedMessages[0]?.id).toBe("local-user-output");
    expect(mergedMessages[1]?.id).toBe("local-assistant-output");
    expect(mergedMessages[1]?.content).toBe(
      "这张截图展示了 Lime 聊天区把图片恢复成了可见缩略图。",
    );
    expect(mergedMessages[1]?.contentParts).toEqual([
      {
        type: "text",
        text: "这张截图展示了 Lime 聊天区把图片恢复成了可见缩略图。",
      },
    ]);
    expect(mergedMessages[1]?.usage).toEqual({
      input_tokens: 1200,
      output_tokens: 80,
    });
  });

  it("远端 failed runtimeStatus 不应覆盖本地已输出正文", () => {
    const localMessages = [
      {
        id: "local-user-news",
        role: "user" as const,
        content: "整理今天的国际新闻",
        timestamp: new Date("2026-06-07T09:30:00.000Z"),
      },
      {
        id: "local-assistant-news",
        role: "assistant" as const,
        content: "我会先检索多组来源并交叉核对。",
        contentParts: [
          {
            type: "text" as const,
            text: "我会先检索多组来源并交叉核对。",
          },
        ],
        toolCalls: [
          {
            id: "tool-web-search-running",
            name: "WebSearch",
            arguments: '{"query":"2026年6月7日 国际新闻"}',
            status: "running" as const,
            startTime: new Date("2026-06-07T09:30:02.000Z"),
          },
        ],
        timestamp: new Date("2026-06-07T09:30:01.000Z"),
        isThinking: true,
        runtimeStatus: {
          phase: "routing" as const,
          title: "正在输出",
          detail: "正在等待模型继续输出。",
        },
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-news",
        role: "user" as const,
        content: "整理今天的国际新闻",
        timestamp: new Date("2026-06-07T09:30:00.000Z"),
      },
      {
        id: "history-assistant-news-failed",
        role: "assistant" as const,
        content: `执行失败：${INTERNAL_RUNTIME_ERROR_MESSAGE}`,
        contentParts: [
          {
            type: "text" as const,
            text: `执行失败：${INTERNAL_RUNTIME_ERROR_MESSAGE}`,
          },
        ],
        timestamp: new Date("2026-06-07T09:30:12.000Z"),
        isThinking: false,
        runtimeTurnId: "turn-news-failed",
        runtimeStatus: {
          phase: "failed" as const,
          title: "当前处理失败",
          detail: INTERNAL_RUNTIME_ERROR_MESSAGE,
        },
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    expect(mergedMessages).toHaveLength(2);
    expect(mergedMessages[1]).toMatchObject({
      id: "local-assistant-news",
      content: "我会先检索多组来源并交叉核对。",
      isThinking: false,
      runtimeTurnId: "turn-news-failed",
      runtimeStatus: {
        phase: "failed",
        title: "当前处理失败",
      },
    });
    expect(mergedMessages[1]?.contentParts).toEqual([
      {
        type: "text",
        text: "我会先检索多组来源并交叉核对。",
      },
    ]);
    expect(mergedMessages[1]?.content).not.toContain(
      INTERNAL_RUNTIME_ERROR_MESSAGE,
    );
    expect(mergedMessages[1]?.content).not.toContain(
      "token-plan-cn.xiaomimimo.com",
    );
    expect(mergedMessages[1]?.runtimeStatus?.detail).toBe(
      INTERNAL_RUNTIME_ERROR_MESSAGE,
    );
    expect(mergedMessages[1]?.toolCalls?.[0]).toMatchObject({
      id: "tool-web-search-running",
      status: "failed",
      result: {
        success: false,
        error: INTERNAL_RUNTIME_ERROR_MESSAGE,
      },
    });
  });

  it("刷新会话详情时应保留本地用户消息的输入能力标签", () => {
    const localMessages = [
      {
        id: "local-user-skill",
        role: "user" as const,
        content: "请说明 Minimal API 的测试策略",
        timestamp: new Date("2026-05-13T03:20:00.000Z"),
        inputCapabilityRoute: {
          kind: "installed_skill" as const,
          skillKey: "aspnet-core",
          skillName: "aspnet-core",
        },
      },
      {
        id: "local-assistant-skill",
        role: "assistant" as const,
        content: "",
        timestamp: new Date("2026-05-13T03:20:01.000Z"),
        runtimeStatus: {
          phase: "preparing" as const,
          title: "正在准备回复",
          detail: "正在准备回复",
        },
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-skill",
        role: "user" as const,
        content: "请说明 Minimal API 的测试策略",
        timestamp: new Date("2026-05-13T03:20:02.000Z"),
      },
      {
        id: "history-assistant-skill",
        role: "assistant" as const,
        content: "可以从单元测试和集成测试两层设计。",
        timestamp: new Date("2026-05-13T03:20:03.000Z"),
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    expect(mergedMessages[0]).toMatchObject({
      id: "local-user-skill",
      inputCapabilityRoute: {
        kind: "installed_skill",
        skillKey: "aspnet-core",
        skillName: "aspnet-core",
      },
    });
  });

  it("远端只返回助手正文时仍应保留本地用户输入与 Skill 思考过程", () => {
    const localMessages = [
      {
        id: "local-analysis-user",
        role: "user" as const,
        content: "@analysis 帮我分析一下今天的国际形势",
        timestamp: new Date("2026-05-13T17:51:40.000Z"),
      },
      {
        id: "local-analysis-assistant",
        role: "assistant" as const,
        content: "# 分析结果\n\n## 结论\n国际形势分析结果。",
        timestamp: new Date("2026-05-13T17:51:42.000Z"),
        runtimeTurnId: "skill-exec-local-analysis-assistant",
        inlineProcessRetention: "skill" as const,
        thinkingContent: "先识别 analysis Skill，再组织结论。",
        contentParts: [
          {
            type: "thinking" as const,
            text: "先识别 analysis Skill，再组织结论。",
          },
          {
            type: "text" as const,
            text: "# 分析结果\n\n## 结论\n国际形势分析结果。",
          },
        ],
      },
    ];
    const hydratedMessages = [
      {
        id: "history-analysis-assistant",
        role: "assistant" as const,
        content: "# 分析结果\n\n## 结论\n国际形势分析结果。",
        contentParts: [
          {
            type: "text" as const,
            text: "# 分析结果\n\n## 结论\n国际形势分析结果。",
          },
        ],
        timestamp: new Date("2026-05-13T17:51:45.000Z"),
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    expect(mergedMessages).toHaveLength(2);
    expect(mergedMessages[0]).toMatchObject({
      id: "local-analysis-user",
      role: "user",
      content: "@analysis 帮我分析一下今天的国际形势",
    });
    expect(mergedMessages[1]).toMatchObject({
      id: "local-analysis-assistant",
      role: "assistant",
      runtimeTurnId: "skill-exec-local-analysis-assistant",
      inlineProcessRetention: "skill",
      thinkingContent: "先识别 analysis Skill，再组织结论。",
    });
    expect(mergedMessages[1]?.contentParts).toEqual([
      {
        type: "thinking",
        text: "先识别 analysis Skill，再组织结论。",
      },
      {
        type: "text",
        text: "# 分析结果\n\n## 结论\n国际形势分析结果。",
      },
    ]);
  });

  it("远端详情返回更新的 assistant 正文时应替换本地快照", () => {
    const localMessages = [
      {
        id: "local-user-topic",
        role: "user" as const,
        content: "继续完善上一个方案",
        timestamp: new Date("2026-04-24T10:00:00.000Z"),
      },
      {
        id: "local-assistant-topic",
        role: "assistant" as const,
        content: "这是本地快照里的最近结果。",
        contentParts: [
          {
            type: "text" as const,
            text: "这是本地快照里的最近结果。",
          },
        ],
        timestamp: new Date("2026-04-24T10:00:02.000Z"),
      },
    ];
    const hydratedMessages = [
      {
        id: "remote-user-topic",
        role: "user" as const,
        content: "继续完善上一个方案",
        timestamp: new Date("2026-04-24T10:00:01.000Z"),
      },
      {
        id: "remote-assistant-topic",
        role: "assistant" as const,
        content: "这是远端补全后的最终结果。",
        contentParts: [
          {
            type: "text" as const,
            text: "这是远端补全后的最终结果。",
          },
        ],
        timestamp: new Date("2026-04-24T10:00:05.000Z"),
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    expect(mergedMessages[0]?.id).toBe("local-user-topic");
    expect(mergedMessages[1]?.id).toBe("local-assistant-topic");
    expect(mergedMessages[1]?.content).toBe("这是远端补全后的最终结果。");
    expect(mergedMessages[1]?.contentParts).toEqual([
      {
        type: "text",
        text: "这是远端补全后的最终结果。",
      },
    ]);
  });

  it("刷新会话详情时远端短前缀不应截断当前 turn 的本地完整输出", () => {
    const localMessages = [
      {
        id: "local-user-expert-panel",
        role: "user" as const,
        content:
          "请继续以「代码文学专家」身份，使用刚添加的技能再做一次最小代码审查。",
        timestamp: new Date("2026-06-21T18:42:38.000Z"),
        runtimeTurnId: "turn-expert-panel",
      },
      {
        id: "local-assistant-expert-panel",
        role: "assistant" as const,
        content:
          "我识别到右侧专家面板更新后的 skillRefs，并继续通过 skill_search 选择单个 Skill。\n专家面板 Skills runtime 证据已完成：新增技能引用已进入本轮上下文，执行仍经过 skill_search、SKILL.md 按需读取、gate 和 Skill 调用。",
        contentParts: [
          {
            type: "tool_use" as const,
            toolCall: {
              id: "skill-search-expert-panel",
              name: "skill_search",
              arguments: '{"query":"capability report"}',
              status: "completed" as const,
              startTime: new Date("2026-06-21T18:42:38.500Z"),
              endTime: new Date("2026-06-21T18:42:39.000Z"),
              result: {
                success: true,
                output: "capability-report",
              },
            },
          },
          {
            type: "text" as const,
            text: "我识别到右侧专家面板更新后的 skillRefs，并继续通过 skill_search 选择单个 Skill。\n专家面板 Skills runtime 证据已完成：新增技能引用已进入本轮上下文，执行仍经过 skill_search、SKILL.md 按需读取、gate 和 Skill 调用。",
          },
        ],
        timestamp: new Date("2026-06-21T18:42:39.000Z"),
        runtimeTurnId: "turn-expert-panel",
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-expert-panel",
        role: "user" as const,
        content:
          "请继续以「代码文学专家」身份，使用刚添加的技能再做一次最小代码审查。",
        timestamp: new Date("2026-06-21T18:42:38.300Z"),
        runtimeTurnId: "turn-expert-panel",
      },
      {
        id: "history-assistant-expert-panel",
        role: "assistant" as const,
        content:
          "我识别到右侧专家面板更新后的 skillRefs，并继续通过 skill_search 选择单个 Skill。",
        contentParts: [
          {
            type: "text" as const,
            text: "我识别到右侧专家面板更新后的 skillRefs，并继续通过 skill_search 选择单个 Skill。",
          },
        ],
        timestamp: new Date("2026-06-21T18:42:45.000Z"),
        runtimeTurnId: "turn-expert-panel",
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    expect(mergedMessages[1]?.id).toBe("local-assistant-expert-panel");
    expect(mergedMessages[1]?.content).toContain(
      "专家面板 Skills runtime 证据已完成",
    );
    expect(mergedMessages[1]?.contentParts?.at(-1)).toMatchObject({
      type: "text",
      text: expect.stringContaining("新增技能引用已进入本轮上下文"),
    });
  });

  it("刷新会话详情时不应把上一轮专家过程合并到下一轮 assistant", () => {
    const localMessages = [
      {
        id: "local-user-expert-first",
        role: "user" as const,
        content: "请以「代码文学专家」身份，使用绑定技能完成一次最小代码审查。",
        timestamp: new Date("2026-06-21T18:41:38.000Z"),
        runtimeTurnId: "turn-expert-first",
      },
      {
        id: "local-assistant-expert-first",
        role: "assistant" as const,
        content:
          "专家 Skills runtime 证据已完成：专家声明 skillRefs 只作为候选提示。",
        contentParts: [
          {
            type: "text" as const,
            text: "我识别到专家绑定的 skillRefs，但仍先通过 skill_search 选择。",
            metadata: {
              phase: "commentary",
              turnId: "turn-expert-first",
              itemId: "agent-message-commentary-turn-expert-first",
            },
          },
          {
            type: "text" as const,
            text: "专家 Skills runtime 证据已完成：专家声明 skillRefs 只作为候选提示。",
            metadata: {
              phase: "final_answer",
              turnId: "turn-expert-first",
              itemId: "agent-message-final-turn-expert-first",
            },
          },
        ],
        timestamp: new Date("2026-06-21T18:41:39.000Z"),
        runtimeTurnId: "turn-expert-first",
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-expert-panel",
        role: "user" as const,
        content:
          "请继续以「代码文学专家」身份，使用刚添加的技能再做一次最小代码审查。",
        timestamp: new Date("2026-06-21T18:42:38.300Z"),
        runtimeTurnId: "turn-expert-panel",
      },
      {
        id: "history-assistant-expert-panel",
        role: "assistant" as const,
        content:
          "专家面板新增 Skill 后的下一轮 runtime 证据已完成：右侧面板调整 skillRefs 后，下一轮请求继续通过 skill_search、SKILL.md 按需读取、gate 和 Skill 调用。",
        contentParts: [
          {
            type: "text" as const,
            text: "专家面板新增 Skill 后的下一轮 runtime 证据已完成：右侧面板调整 skillRefs 后，下一轮请求继续通过 skill_search、SKILL.md 按需读取、gate 和 Skill 调用。",
            metadata: {
              phase: "final_answer",
              turnId: "turn-expert-panel",
              itemId: "agent-message-final-turn-expert-panel",
            },
          },
        ],
        timestamp: new Date("2026-06-21T18:42:45.000Z"),
        runtimeTurnId: "turn-expert-panel",
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    expect(mergedMessages[1]?.id).toBe("history-assistant-expert-panel");
    expect(mergedMessages[1]?.runtimeTurnId).toBe("turn-expert-panel");
    expect(mergedMessages[1]?.content).not.toContain(
      "专家 Skills runtime 证据已完成",
    );
    expect(mergedMessages[1]?.contentParts).toEqual(
      hydratedMessages[1]?.contentParts,
    );
  });

  it("相邻 assistant 历史过程不应跨 runtime turn 合并", () => {
    const mergedMessages = mergeAdjacentAssistantMessages([
      {
        id: "assistant-expert-first",
        role: "assistant",
        content: "专家 Skills runtime 证据已完成：第一轮。",
        contentParts: [
          {
            type: "text",
            text: "专家 Skills runtime 证据已完成：第一轮。",
            metadata: {
              phase: "final_answer",
              turnId: "turn-expert-first",
            },
          },
        ],
        timestamp: new Date("2026-06-21T18:41:39.000Z"),
        runtimeTurnId: "turn-expert-first",
      },
      {
        id: "assistant-expert-panel",
        role: "assistant",
        content: "专家面板新增 Skill 后的下一轮 runtime 证据已完成。",
        contentParts: [
          {
            type: "text",
            text: "专家面板新增 Skill 后的下一轮 runtime 证据已完成。",
            metadata: {
              phase: "final_answer",
              turnId: "turn-expert-panel",
            },
          },
        ],
        timestamp: new Date("2026-06-21T18:42:45.000Z"),
        runtimeTurnId: "turn-expert-panel",
      },
    ]);

    expect(mergedMessages).toHaveLength(2);
    expect(mergedMessages[1]?.content).not.toContain(
      "专家 Skills runtime 证据已完成",
    );
    expect(mergedMessages[1]?.runtimeTurnId).toBe("turn-expert-panel");
  });

  it("刷新会话详情时远端 WebTools final 不应覆盖本地已流式输出的导语和搜索过程", () => {
    const localMessages = [
      {
        id: "local-user-web-tools",
        role: "user" as const,
        content: "检查一下 WebSearch 渲染是否和 Codex 一致",
        timestamp: new Date("2026-06-22T00:30:00.000Z"),
        runtimeTurnId: "turn-web-tools-rendering",
      },
      {
        id: "local-assistant-web-tools",
        role: "assistant" as const,
        content:
          "我先联网核实目标页面来源。\n\n网页搜索渲染结论：搜索、抓取和最终 Markdown 应按事件顺序展示。",
        contentParts: [
          {
            type: "text" as const,
            text: "我先联网核实目标页面来源。",
          },
          {
            type: "tool_use" as const,
            toolCall: {
              id: "web-search-rendering",
              name: "WebSearch",
              arguments: '{"query":"Lime WebSearch rendering"}',
              status: "completed" as const,
              startTime: new Date("2026-06-22T00:30:01.000Z"),
              endTime: new Date("2026-06-22T00:30:02.000Z"),
              result: {
                success: true,
                output: "Lime WebSearch Rendering Source",
              },
            },
          },
          {
            type: "tool_use" as const,
            toolCall: {
              id: "web-fetch-rendering",
              name: "WebFetch",
              arguments:
                '{"url":"https://example.com/lime-web-search-rendering"}',
              status: "completed" as const,
              startTime: new Date("2026-06-22T00:30:02.000Z"),
              endTime: new Date("2026-06-22T00:30:03.000Z"),
              result: {
                success: true,
                output: "Fetched page snapshot",
              },
            },
          },
          {
            type: "text" as const,
            text: "\n\n网页搜索渲染结论：搜索、抓取和最终 Markdown 应按事件顺序展示。",
          },
        ],
        timestamp: new Date("2026-06-22T00:30:03.000Z"),
        runtimeTurnId: "turn-web-tools-rendering",
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-web-tools",
        role: "user" as const,
        content: "检查一下 WebSearch 渲染是否和 Codex 一致",
        timestamp: new Date("2026-06-22T00:30:00.500Z"),
        runtimeTurnId: "turn-web-tools-rendering",
      },
      {
        id: "history-assistant-web-tools",
        role: "assistant" as const,
        content:
          "网页搜索渲染结论：搜索、抓取和最终 Markdown 应按事件顺序展示。",
        contentParts: [
          {
            type: "tool_use" as const,
            toolCall: {
              id: "web-search-rendering",
              name: "WebSearch",
              arguments: '{"query":"Lime WebSearch rendering"}',
              status: "completed" as const,
              startTime: new Date("2026-06-22T00:30:01.000Z"),
              endTime: new Date("2026-06-22T00:30:02.000Z"),
              result: {
                success: true,
                output: "Lime WebSearch Rendering Source",
              },
            },
          },
          {
            type: "thinking" as const,
            text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
            metadata: {
              source: "thread_item_reasoning",
              threadItemId: "web-reasoning-rendering",
            },
          },
          {
            type: "tool_use" as const,
            toolCall: {
              id: "web-fetch-rendering",
              name: "WebFetch",
              arguments:
                '{"url":"https://example.com/lime-web-search-rendering"}',
              status: "completed" as const,
              startTime: new Date("2026-06-22T00:30:02.000Z"),
              endTime: new Date("2026-06-22T00:30:03.000Z"),
              result: {
                success: true,
                output: "Fetched page snapshot",
              },
            },
          },
          {
            type: "text" as const,
            text: "网页搜索渲染结论：搜索、抓取和最终 Markdown 应按事件顺序展示。",
          },
        ],
        timestamp: new Date("2026-06-22T00:30:06.000Z"),
        runtimeTurnId: "turn-web-tools-rendering",
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    expect(mergedMessages[1]?.id).toBe("local-assistant-web-tools");
    expect(mergedMessages[1]?.content).toContain("我先联网核实目标页面来源。");
    expect(mergedMessages[1]?.contentParts?.map((part) => part.type)).toEqual([
      "text",
      "tool_use",
      "thinking",
      "tool_use",
      "text",
    ]);
    expect(mergedMessages[1]?.contentParts?.[0]).toMatchObject({
      type: "text",
      text: "我先联网核实目标页面来源。",
    });
    expect(mergedMessages[1]?.contentParts?.[1]).toMatchObject({
      type: "tool_use",
      toolCall: {
        id: "web-search-rendering",
        name: "WebSearch",
        status: "completed",
      },
    });
  });

  it("刷新会话详情时远端 WebTools final 不应覆盖本地已流式输出的导语文本", () => {
    const localMessages = [
      {
        id: "local-user-web-tools-text",
        role: "user" as const,
        content: "验证网页搜索渲染",
        timestamp: new Date("2026-06-22T01:00:00.000Z"),
        runtimeTurnId: "turn-web-tools-text",
      },
      {
        id: "local-assistant-web-tools-text",
        role: "assistant" as const,
        content:
          "我先联网核实目标页面来源。\n\n网页搜索渲染结论：搜索来源已展开，读取页面已归入同一过程，最终正文继续输出。",
        contentParts: [
          {
            type: "text" as const,
            text: "我先联网核实目标页面来源。",
          },
          {
            type: "text" as const,
            text: "\n\n网页搜索渲染结论：搜索来源已展开，读取页面已归入同一过程，最终正文继续输出。",
          },
        ],
        timestamp: new Date("2026-06-22T01:00:02.000Z"),
        runtimeTurnId: "turn-web-tools-text",
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-web-tools-text",
        role: "user" as const,
        content: "验证网页搜索渲染",
        timestamp: new Date("2026-06-22T01:00:00.500Z"),
        runtimeTurnId: "turn-web-tools-text",
      },
      {
        id: "history-assistant-web-tools-text",
        role: "assistant" as const,
        content:
          "网页搜索渲染结论：搜索来源已展开，读取页面已归入同一过程，最终正文继续输出。",
        contentParts: [
          {
            type: "text" as const,
            text: "网页搜索渲染结论：搜索来源已展开，读取页面已归入同一过程，最终正文继续输出。",
          },
        ],
        timestamp: new Date("2026-06-22T01:00:06.000Z"),
        runtimeTurnId: "turn-web-tools-text",
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    expect(mergedMessages[1]?.id).toBe("local-assistant-web-tools-text");
    expect(mergedMessages[1]?.content).toContain("我先联网核实目标页面来源。");
    expect(mergedMessages[1]?.content.match(/网页搜索渲染结论/g)).toHaveLength(
      1,
    );
    expect(mergedMessages[1]?.contentParts?.[0]).toMatchObject({
      type: "text",
      text: "我先联网核实目标页面来源。",
    });
  });

  it("后端暂未返回历史时应保留本地消息，避免刷新后界面空白", () => {
    const localMessages = [
      {
        id: "local-user-1",
        role: "user" as const,
        content: "继续刚才的任务",
        timestamp: new Date("2026-03-19T00:00:00.000Z"),
      },
      {
        id: "local-assistant-1",
        role: "assistant" as const,
        content: "这是刚刚停止后的对话内容",
        timestamp: new Date("2026-03-19T00:00:01.000Z"),
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      [],
    );

    expect(mergedMessages).toEqual(localMessages);
  });

  it("同会话刷新详情时应保留本地 assistant 的 token usage", () => {
    const localMessages = [
      {
        id: "local-user-1",
        role: "user" as const,
        content: "请整理成可继续编辑的文稿",
        timestamp: new Date("2026-04-07T00:00:00.000Z"),
      },
      {
        id: "local-assistant-1",
        role: "assistant" as const,
        content: "已同步到工作区，可继续在画布里阅读、编辑和定位到对应区块。",
        timestamp: new Date("2026-04-07T00:00:01.000Z"),
        usage: {
          input_tokens: 20480,
          output_tokens: 10240,
          cached_input_tokens: 8192,
          cache_creation_input_tokens: 2048,
        },
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-1",
        role: "user" as const,
        content: "请整理成可继续编辑的文稿",
        timestamp: new Date("2026-04-07T00:00:02.000Z"),
      },
      {
        id: "history-assistant-1",
        role: "assistant" as const,
        content: "已同步到工作区，可继续在画布里阅读、编辑和定位到对应区块。",
        timestamp: new Date("2026-04-07T00:00:03.000Z"),
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    expect(mergedMessages[1]?.usage).toEqual({
      input_tokens: 20480,
      output_tokens: 10240,
      cached_input_tokens: 8192,
      cache_creation_input_tokens: 2048,
    });
  });

  it("terminal 详情刷新偏向远端正文时仍应保留本地已停止标记", () => {
    const localMessages = [
      {
        id: "local-user-cancelled",
        role: "user" as const,
        content: "整理今天的国际新闻",
        timestamp: new Date("2026-07-10T00:00:00.000Z"),
      },
      {
        id: "local-assistant-cancelled",
        role: "assistant" as const,
        content: "以下是今日国际新闻简要整理：\n\n(已停止)",
        contentParts: [
          {
            type: "text" as const,
            text: "以下是今日国际新闻简要整理：",
          },
          {
            type: "text" as const,
            text: "(已停止)",
          },
        ],
        timestamp: new Date("2026-07-10T00:00:01.000Z"),
        runtimeTurnId: "turn-cancelled",
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-cancelled",
        role: "user" as const,
        content: "整理今天的国际新闻",
        timestamp: new Date("2026-07-10T00:00:00.500Z"),
      },
      {
        id: "history-assistant-cancelled",
        role: "assistant" as const,
        content: "以下是今日国际新闻简要整理：",
        contentParts: [
          {
            type: "text" as const,
            text: "以下是今日国际新闻简要整理：",
          },
        ],
        timestamp: new Date("2026-07-10T00:00:01.500Z"),
        runtimeTurnId: "turn-cancelled",
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
      { preferHydratedAssistantOutput: true },
    );

    expect(mergedMessages[1]?.content).toBe(
      "以下是今日国际新闻简要整理：\n\n(已停止)",
    );
    expect(
      mergedMessages[1]?.contentParts?.some(
        (part) => part.type === "text" && part.text === "(已停止)",
      ),
    ).toBe(true);
  });

  it("terminal 详情缺少用户 item 时也应按 runtimeTurnId 保留本地已停止标记", () => {
    const localMessages = [
      {
        id: "local-user-cancelled",
        role: "user" as const,
        content: "整理今天的国际新闻",
        timestamp: new Date("2026-07-10T00:00:00.000Z"),
        runtimeTurnId: "turn-cancelled-without-user-item",
      },
      {
        id: "local-assistant-cancelled",
        role: "assistant" as const,
        content: "(已停止)",
        contentParts: [
          {
            type: "text" as const,
            text: "(已停止)",
          },
        ],
        timestamp: new Date("2026-07-10T00:00:01.000Z"),
        runtimeTurnId: "turn-cancelled-without-user-item",
      },
    ];
    const hydratedMessages = [
      {
        id: "history-assistant-cancelled",
        role: "assistant" as const,
        content: "以下是今日国际新闻简要整理：",
        contentParts: [
          {
            type: "text" as const,
            text: "以下是今日国际新闻简要整理：",
          },
        ],
        timestamp: new Date("2026-07-10T00:00:01.500Z"),
        runtimeTurnId: "turn-cancelled-without-user-item",
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
      { preferHydratedAssistantOutput: true },
    );

    expect(mergedMessages).toHaveLength(2);
    expect(mergedMessages[0]?.id).toBe("local-user-cancelled");
    expect(mergedMessages[1]?.id).toBe("local-assistant-cancelled");
    expect(mergedMessages[1]?.content).toBe(
      "以下是今日国际新闻简要整理：\n\n(已停止)",
    );
    expect(
      mergedMessages[1]?.contentParts?.some(
        (part) => part.type === "text" && part.text === "(已停止)",
      ),
    ).toBe(true);
  });

  it("terminal 详情只有真实 turn assistant 时应迁移 pending-turn 已停止标记", () => {
    const localMessages = [
      {
        id: "local-user-cancelled-pending",
        role: "user" as const,
        content: "整理今天的国际新闻",
        timestamp: new Date("2026-07-10T00:00:00.000Z"),
        runtimeTurnId: "pending-turn:cancelled",
      },
      {
        id: "local-assistant-cancelled-pending",
        role: "assistant" as const,
        content: "(已停止)",
        contentParts: [
          {
            type: "text" as const,
            text: "(已停止)",
          },
        ],
        timestamp: new Date("2026-07-10T00:00:01.000Z"),
        runtimeTurnId: "pending-turn:cancelled",
      },
    ];
    const hydratedMessages = [
      {
        id: "history-assistant-cancelled-real-turn",
        role: "assistant" as const,
        content: "以下是今日国际新闻简要整理：",
        contentParts: [
          {
            type: "text" as const,
            text: "以下是今日国际新闻简要整理：",
          },
        ],
        timestamp: new Date("2026-07-10T00:00:01.500Z"),
        runtimeTurnId: "turn-cancelled-real",
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
      { preferHydratedAssistantOutput: true },
    );

    expect(mergedMessages).toHaveLength(2);
    expect(mergedMessages[0]?.id).toBe("local-user-cancelled-pending");
    expect(mergedMessages[1]?.id).toBe("local-assistant-cancelled-pending");
    expect(mergedMessages[1]?.runtimeTurnId).toBe("turn-cancelled-real");
    expect(mergedMessages[1]?.content).toBe(
      "以下是今日国际新闻简要整理：\n\n(已停止)",
    );
    expect(
      mergedMessages[1]?.contentParts?.some(
        (part) => part.type === "text" && part.text === "(已停止)",
      ),
    ).toBe(true);
  });
});
