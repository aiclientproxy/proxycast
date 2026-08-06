import { describe, expect, it } from "vitest";

import {
  applyGeneratedAutoTitleToTopics,
  buildAutoTitleConversationText,
  hasUserTextMessage,
  isAutoTitlePlaceholder,
  isPreviewDerivedTitle,
  sanitizeGeneratedAutoTitle,
  shouldGenerateAutoTitle,
} from "./agentChatAutoTitleViewModel";
import type { Topic } from "./agentChatShared";

describe("agentChatAutoTitleViewModel", () => {
  it("应识别自动标题占位标题", () => {
    expect(isAutoTitlePlaceholder(null)).toBe(true);
    expect(isAutoTitlePlaceholder(undefined)).toBe(true);
    expect(isAutoTitlePlaceholder("")).toBe(true);
    expect(isAutoTitlePlaceholder("未命名")).toBe(true);
    expect(isAutoTitlePlaceholder("未命名对话")).toBe(true);
    expect(isAutoTitlePlaceholder(" 新任务 ")).toBe(true);
    expect(isAutoTitlePlaceholder("新话题")).toBe(true);
    expect(isAutoTitlePlaceholder("新对话")).toBe(true);
    expect(isAutoTitlePlaceholder("支付页错误定位")).toBe(false);
  });

  it("应识别由 assistant 预览派生出的临时标题", () => {
    expect(
      isPreviewDerivedTitle("我会先检查支付页", [
        {
          role: "assistant",
          content: "我会先检查支付页请求链路，并定位 500 错误。",
        },
      ]),
    ).toBe(true);

    expect(
      isPreviewDerivedTitle("我会先检查支付页请求链路，并定位", [
        {
          role: "assistant",
          content: "我会先检查支付页请求链路，并定位 500 错误。",
        },
      ]),
    ).toBe(true);

    expect(
      isPreviewDerivedTitle("支付页错误定位", [
        {
          role: "assistant",
          content: "我会先检查支付页请求链路。",
        },
      ]),
    ).toBe(false);
  });

  it("缺少有效 assistant 文本时不应把标题判为预览派生", () => {
    expect(isPreviewDerivedTitle("任意标题", [])).toBe(false);
    expect(
      isPreviewDerivedTitle("任意标题", [
        {
          role: "user",
          content: "请帮我定位问题",
        },
        {
          role: "assistant",
          content: "   ",
        },
      ]),
    ).toBe(false);
    expect(isPreviewDerivedTitle("   ", [])).toBe(false);
  });

  it("应根据占位或预览派生标题判断是否生成自动标题", () => {
    expect(
      shouldGenerateAutoTitle({
        activeSessionTitle: "新对话",
        messages: [],
      }),
    ).toBe(true);

    expect(
      shouldGenerateAutoTitle({
        activeSessionTitle: "我会先检查支付页",
        messages: [
          {
            role: "assistant",
            content: "我会先检查支付页请求链路。",
          },
        ],
      }),
    ).toBe(true);

    expect(
      shouldGenerateAutoTitle({
        activeSessionTitle: "支付页错误定位",
        messages: [
          {
            role: "assistant",
            content: "我会先检查支付页请求链路。",
          },
        ],
      }),
    ).toBe(false);
  });

  it("应识别是否存在可用于生成标题的用户文本", () => {
    expect(hasUserTextMessage([])).toBe(false);
    expect(
      hasUserTextMessage([
        {
          role: "assistant",
          content: "我会先分析。",
        },
      ]),
    ).toBe(false);
    expect(
      hasUserTextMessage([
        {
          role: "user",
          content: "   ",
        },
      ]),
    ).toBe(false);
    expect(
      hasUserTextMessage([
        {
          role: "user",
          content: "帮我定位支付页 500",
        },
      ]),
    ).toBe(true);
  });

  it("应构造发送给自动标题生成器的对话文本并裁剪到末尾 1000 字符", () => {
    const longUserText = "x".repeat(1200);
    const conversationText = buildAutoTitleConversationText([
      {
        role: "user",
        content: "   ",
      },
      {
        role: "user",
        content: longUserText,
      },
      {
        role: "assistant",
        content: "我会继续处理。",
      },
    ]);

    expect(conversationText).toHaveLength(1000);
    expect(conversationText).toContain("assistant：我会继续处理。");
    expect(conversationText).not.toContain("user：   ");
  });

  it("应把生成标题回填到目标 topic 且保留其他 topic 引用", () => {
    const targetTopic = createTopic("session-1", "新任务");
    const otherTopic = createTopic("session-2", "已有标题");
    const topics = [targetTopic, otherTopic];

    const result = applyGeneratedAutoTitleToTopics(
      topics,
      "session-1",
      "  支付页错误定位  ",
    );

    expect(result).not.toBe(topics);
    expect(result[0]).toEqual({
      ...targetTopic,
      title: "支付页错误定位",
    });
    expect(result[1]).toBe(otherTopic);
  });

  it("图片任务自动标题入库前保留生成器原文", () => {
    expect(
      sanitizeGeneratedAutoTitle(
        "配图：用 Agnes Generate一张深圳夏day午后的城市照片，真实摄影Style",
        "user：@配图 用 Agnes 生成一张深圳夏天午后的城市照片，真实摄影风格",
      ),
    ).toBe("配图：用 Agnes Generate一张深圳夏day午后的城市照片，真实摄影Style");

    const targetTopic = createTopic("session-1", "新任务");
    const result = applyGeneratedAutoTitleToTopics(
      [targetTopic],
      "session-1",
      "配图：用 Agnes Generate一张深圳夏day午后的城市照片，真实摄影Style",
      "user：@配图 用 Agnes 生成一张深圳夏天午后的城市照片，真实摄影风格",
    );

    expect(result[0]?.title).toBe(
      "配图：用 Agnes Generate一张深圳夏day午后的城市照片，真实摄影Style",
    );
  });

  it("正常英文自动标题不应被图片标题清洗误伤", () => {
    expect(
      sanitizeGeneratedAutoTitle("Generate landing page style guide"),
    ).toBe("Generate landing page style guide");
  });

  it("生成标题为空、未命中或标题未变化时应复用原 topics", () => {
    const topics = [
      createTopic("session-1", "支付页错误定位"),
      createTopic("session-2", "已有标题"),
    ];

    expect(applyGeneratedAutoTitleToTopics(topics, "session-1", "   ")).toBe(
      topics,
    );
    expect(
      applyGeneratedAutoTitleToTopics(topics, "missing-session", "新标题"),
    ).toBe(topics);
    expect(
      applyGeneratedAutoTitleToTopics(topics, "session-1", "支付页错误定位"),
    ).toBe(topics);
  });
});

function createTopic(id: string, title: string): Topic {
  return {
    id,
    title,
    createdAt: new Date("2026-01-01T00:00:00.000Z"),
    updatedAt: new Date("2026-01-01T00:00:00.000Z"),
    messagesCount: 0,
    executionStrategy: "react",
    status: "draft",
    lastPreview: "",
    hasUnread: false,
    tag: null,
    sourceSessionId: id,
  };
}
