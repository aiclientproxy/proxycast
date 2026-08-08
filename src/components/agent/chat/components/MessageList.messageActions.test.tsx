import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  mockStreamingRenderer,
  render,
  createRoot,
  MessageList,
  mountedRoots,
} from "./MessageList.testHarness";
import type { Message } from "./MessageList.testHarness";
import { CONVERSATION_CONTENT_MAX_WIDTH } from "../styles/conversationLayoutTokens";

describe("MessageList message actions", () => {
  it("应按回合分组展示同一轮用户与后续助手回复", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-user-1",
        role: "user",
        content: "先打开公众号后台",
        timestamp: new Date(now.getTime()),
      },
      {
        id: "msg-assistant-1",
        role: "assistant",
        content: "已打开登录页。",
        timestamp: new Date(now.getTime() + 1000),
      },
      {
        id: "msg-assistant-2",
        role: "assistant",
        content: "等待你完成扫码。",
        timestamp: new Date(now.getTime() + 2000),
      },
      {
        id: "msg-user-2",
        role: "user",
        content: "我已扫码，继续发布",
        timestamp: new Date(now.getTime() + 3000),
      },
      {
        id: "msg-assistant-3",
        role: "assistant",
        content: "已继续执行发布流程。",
        timestamp: new Date(now.getTime() + 4000),
      },
    ];

    const container = render(messages);
    const groups = Array.from(
      container.querySelectorAll('[data-testid="message-turn-group"]'),
    );

    expect(groups).toHaveLength(2);
    expect(groups[0]?.textContent).toContain("先打开公众号后台");
    expect(groups[0]?.textContent).toContain("已打开登录页。");
    expect(groups[0]?.textContent).toContain("等待你完成扫码。");
    expect(groups[1]?.textContent).toContain("我已扫码，继续发布");
    expect(groups[1]?.textContent).toContain("已继续执行发布流程。");
    expect(
      container.querySelector('[data-testid="message-turn-group:1:header"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="message-turn-group:2:divider"]'),
    ).toBeNull();
  });

  it("用户消息不再渲染引用按钮，避免和 hover footer 冲突", () => {
    const onQuoteMessage = vi.fn();
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-user-quote",
        role: "user",
        content: "请引用这一段内容",
        timestamp: now,
      },
    ];

    const container = render(messages, { onQuoteMessage });
    const quoteButton = container.querySelector(
      'button[aria-label="Quote message"]',
    );

    expect(quoteButton).toBeNull();
    expect(onQuoteMessage).not.toHaveBeenCalled();
    expect(container.querySelector('button[aria-label="编辑消息"]')).toBeNull();
  });

  it("助手正文应将区块级引用/复制能力透传给 StreamingRenderer", () => {
    const onQuoteMessage = vi.fn();
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-block-actions",
        role: "assistant",
        content: "这是需要块级操作的输出",
        timestamp: now,
      },
    ];

    render(messages, { onQuoteMessage });

    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({
        showContentBlockActions: true,
        onQuoteContent: expect.any(Function),
      }),
    );
  });

  it("助手结果应支持保存为技能草稿", () => {
    const onSaveMessageAsSkill = vi.fn();
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-save-skill",
        role: "assistant",
        content:
          "这是一段足够长的结果说明，用来验证助手消息上会出现保存为技能的入口。",
        timestamp: now,
      },
    ];

    const container = render(messages, { onSaveMessageAsSkill });
    const saveButton = container.querySelector(
      'button[aria-label="Save as Skill"]',
    );

    expect(saveButton).not.toBeNull();

    act(() => {
      saveButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onSaveMessageAsSkill).toHaveBeenCalledWith({
      messageId: "msg-assistant-save-skill",
      content:
        "这是一段足够长的结果说明，用来验证助手消息上会出现保存为技能的入口。",
    });
  });

  it("助手结果不应再暴露旧灵感库保存入口", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-save-memory",
        role: "assistant",
        content:
          "这是一段足够长的结果说明，用来验证助手消息不会再出现旧灵感库保存入口。",
        timestamp: now,
      },
    ];

    const container = render(messages);
    const saveButton = container.querySelector(
      'button[aria-label="Save to inspiration"]',
    );

    expect(saveButton).toBeNull();
  });

  it("助手结果应支持保存到项目资料", () => {
    const onSaveMessageAsKnowledge = vi.fn();
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-save-knowledge",
        role: "assistant",
        content:
          "这是一段足够长的项目事实说明，用来验证助手消息上会出现保存到项目资料的入口。",
        timestamp: now,
      },
    ];

    const container = render(messages, { onSaveMessageAsKnowledge });
    const saveButton = container.querySelector(
      'button[aria-label="Save to project knowledge"]',
    );
    const messageActions = container.querySelector(
      '[data-testid="message-actions"]',
    );

    expect(saveButton).not.toBeNull();
    expect(messageActions?.className).toContain("message-actions-persistent");
    expect(saveButton?.getAttribute("title")).toBe("Save to project knowledge");
    expect(saveButton?.textContent).not.toContain("Save to project knowledge");

    act(() => {
      saveButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onSaveMessageAsKnowledge).toHaveBeenCalledWith({
      messageId: "msg-assistant-save-knowledge",
      content:
        "这是一段足够长的项目事实说明，用来验证助手消息上会出现保存到项目资料的入口。",
    });
  });

  it("助手结果带产物时应优先把产物正文保存到项目资料", () => {
    const onSaveMessageAsKnowledge = vi.fn();
    const now = new Date();
    const artifactContent =
      "# 谢晶营销文案包 v1.0\n\n这是一份已经写入项目目录的 Markdown 产物，应该作为项目资料来源。";
    const messages: Message[] = [
      {
        id: "msg-assistant-save-artifact-knowledge",
        role: "assistant",
        content:
          "文件已生成，下面是摘要。这里不应该覆盖真正的 Markdown 产物正文。",
        timestamp: now,
        artifacts: [
          {
            id: "artifact-knowledge-output",
            type: "document",
            title: "谢晶_营销文案包_KnowledgeV2_E2E.md",
            content: artifactContent,
            status: "complete",
            meta: {
              filename: "谢晶_营销文案包_KnowledgeV2_E2E.md",
            },
            position: { start: 0, end: artifactContent.length },
            createdAt: now.getTime(),
            updatedAt: now.getTime(),
          },
        ],
      },
    ];

    const container = render(messages, { onSaveMessageAsKnowledge });
    expect(container.textContent).toContain("Document artifact");
    expect(container.textContent).toContain("Can save to project knowledge");
    const saveButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("Save this document"),
    );

    expect(saveButton).not.toBeNull();

    act(() => {
      saveButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onSaveMessageAsKnowledge).toHaveBeenCalledWith({
      messageId: "msg-assistant-save-artifact-knowledge",
      content: artifactContent,
      sourceName: "谢晶_营销文案包_KnowledgeV2_E2E.md",
      description: "谢晶_营销文案包_KnowledgeV2_E2E.md",
    });
  });

  it("聊天主列与助手消息气泡应和输入区使用同一阅读宽度", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-wide-reading",
        role: "assistant",
        content: "这里是一段较长的结构化输出，用于验证桌面阅读宽度。",
        timestamp: now,
      },
    ];

    const container = render(messages);
    const messageColumn = container.querySelector(
      '[data-testid="message-list-column"]',
    );
    const assistantBubble = container.querySelector('[aria-label="Lime"]');

    expect((messageColumn as HTMLElement | null)?.style.maxWidth).toBe(
      CONVERSATION_CONTENT_MAX_WIDTH,
    );
    expect(assistantBubble).not.toBeNull();
    expect(
      window.getComputedStyle(assistantBubble as Element).maxWidth,
    ).toContain(CONVERSATION_CONTENT_MAX_WIDTH);
  });

  it("助手消息不应再渲染旧的继续处理标签或品牌头像", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-user-seed",
        role: "user",
        content: "继续",
        timestamp: new Date(now.getTime()),
      },
      {
        id: "msg-assistant-first",
        role: "assistant",
        content: "第一条回复",
        timestamp: new Date(now.getTime() + 1000),
      },
      {
        id: "msg-assistant-second",
        role: "assistant",
        content: "第二条回复",
        timestamp: new Date(now.getTime() + 2000),
      },
    ];

    const container = render(messages);

    expect(container.textContent).not.toContain("阶段 00");
    expect(container.textContent).not.toContain("继续处理");
    expect(container.querySelector('img[alt="Lime"]')).toBeNull();
  });

  it("用户图片消息不应渲染内部图片占位文本", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-user-image",
        role: "user",
        content: "[Image #1]",
        images: [
          {
            mediaType: "image/png",
            data: "aGVsbG8=",
          },
        ],
        timestamp: now,
      },
    ];

    const container = render(messages);

    expect(
      container.querySelector('[data-testid="markdown-renderer"]'),
    ).toBeNull();
    const image = container.querySelector(
      '[data-testid="message-image-attachment-0"]',
    );
    expect(image).toBeTruthy();
    expect(container.textContent).not.toContain("[Image #1]");
  });

  it("点击用户图片附件应进入统一消息预览入口", () => {
    const now = new Date();
    const onOpenMessagePreview = vi.fn();
    const messages: Message[] = [
      {
        id: "msg-user-image-open",
        role: "user",
        content: "",
        images: [
          {
            mediaType: "image/png",
            data: "aGVsbG8=",
            sourceUri: "data:image/png;base64,aGVsbG8=",
            previewUrl: "data:image/png;base64,aGVsbG8=",
          },
        ],
        timestamp: now,
      },
    ];

    const container = render(messages, { onOpenMessagePreview });
    const openButton = container.querySelector(
      '[data-testid="message-image-attachment-open-0"]',
    ) as HTMLButtonElement | null;

    act(() => {
      openButton?.click();
    });

    expect(onOpenMessagePreview).toHaveBeenCalledWith(
      {
        kind: "message_attachment",
        attachment: expect.objectContaining({
          mediaType: "image/png",
          sourceUri: "data:image/png;base64,aGVsbG8=",
          previewUrl: "data:image/png;base64,aGVsbG8=",
        }),
        index: 0,
      },
      expect.objectContaining({ id: "msg-user-image-open" }),
    );
  });

  it("助手内部图片标签应在主消息里隐藏", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-image",
        role: "assistant",
        content: "[Image #1]",
        timestamp: now,
      },
    ];

    const container = render(messages);

    expect(
      container.querySelector('[data-testid="streaming-renderer"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain("[Image #1]");
  });

  it("助手消息包含 artifacts 时应渲染产物卡片并响应点击", () => {
    const now = new Date();
    const onArtifactClick = vi.fn();
    const messages: Message[] = [
      {
        id: "msg-assistant-artifact",
        role: "assistant",
        content: "已生成文档",
        timestamp: now,
        artifacts: [
          {
            id: "artifact-demo",
            type: "document",
            title: "demo.md",
            content: "# Demo",
            status: "complete",
            meta: {
              filePath: "docs/demo.md",
              filename: "demo.md",
            },
            position: { start: 0, end: 0 },
            createdAt: now.getTime(),
            updatedAt: now.getTime(),
          },
        ],
      },
    ];

    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);

    act(() => {
      root.render(
        <MessageList messages={messages} onArtifactClick={onArtifactClick} />,
      );
    });

    mountedRoots.push({ container, root });

    const artifactShell = container.querySelector(
      '[data-testid="message-artifact-card"]',
    );
    expect(artifactShell?.className).toContain("border-slate-200");
    expect(artifactShell?.className).not.toContain("bg-sky-50");
    const artifactCard = container.querySelector("button");
    expect(artifactCard?.textContent).toContain("demo.md");
    expect(artifactCard?.textContent).toContain("docs/demo.md");
    expect(container.textContent).toContain("Document artifact");

    act(() => {
      artifactCard?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onArtifactClick).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "artifact-demo",
        title: "demo.md",
      }),
    );
  });

  it("内容工厂文章产物应在过程之后以受控预览展示", () => {
    const now = new Date();
    const onArtifactClick = vi.fn();
    const onSaveMessageAsKnowledge = vi.fn();
    const fullArticle =
      "# 公众号文章草稿\n\n这是第一段正文，应该在独立产物框内完整显示。\n\n这是第二段正文，点击框头后进入右侧文章编辑器。";
    const messages: Message[] = [
      {
        id: "msg-assistant-article-workspace",
        role: "assistant",
        content: "",
        timestamp: now,
        artifacts: [
          {
            id: "preview-article-workspace-article",
            type: "document",
            title: "公众号文章草稿",
            content: fullArticle,
            status: "complete",
            meta: {
              openedFrom: "right_surface_article_workspace",
              filePath: "公众号文章草稿.md",
              filename: "公众号文章草稿.md",
              articleWorkspaceCardPreview: {
                layout: "document",
                summary: null,
                counts: {
                  artifacts: 1,
                  imageSlots: 2,
                  outlineSections: 5,
                  researchRounds: 3,
                },
              },
              articleWorkspace: {
                appId: "content-factory-app",
                sessionId: "session-main",
                objectKind: "articleDraft",
                objectId: "article-1",
                surfaceKind: "document",
              },
              workspacePatch: {
                workerEvidence: [
                  {
                    subagents: ["content-researcher", "article-writer"],
                    skillRefs: ["article-research", "article-writing"],
                    researchRounds: [
                      { id: "research-1", title: "资料检索" },
                      { id: "research-2", title: "资料交叉验证" },
                      { id: "research-3", title: "发布检查" },
                    ],
                    outline: [
                      { id: "section-1", title: "开场" },
                      { id: "section-2", title: "基础" },
                      { id: "section-3", title: "实践" },
                      { id: "section-4", title: "进阶" },
                      { id: "section-5", title: "复盘" },
                    ],
                    writingPlan: [
                      {
                        id: "plan-research",
                        title: "资料检索",
                        owner: "content-researcher",
                        skillRef: "article-research",
                      },
                      {
                        id: "plan-draft",
                        title: "正文写作",
                        owner: "article-writer",
                        skillRef: "article-writing",
                      },
                    ],
                  },
                ],
              },
            },
            position: { start: 0, end: fullArticle.length },
            createdAt: now.getTime(),
            updatedAt: now.getTime(),
          },
        ],
      },
    ];

    const container = render(messages, {
      onArtifactClick,
      onSaveMessageAsKnowledge,
    });
    const artifactFrame = container.querySelector(
      '[data-testid="article-artifact-frame"]',
    );
    const messageActions = container.querySelector(
      '[data-testid="message-actions"]',
    );
    const openButton = artifactFrame?.querySelector(
      "button",
    ) as HTMLButtonElement | null;

    expect(artifactFrame).not.toBeNull();
    expect(messageActions).not.toBeNull();
    expect(
      Boolean(
        artifactFrame!.compareDocumentPosition(messageActions!) &
          Node.DOCUMENT_POSITION_FOLLOWING,
      ),
    ).toBe(true);
    expect(container.textContent).toContain("公众号文章草稿");
    expect(container.textContent).toContain("Document created:");
    expect(container.textContent).toContain("Open document");
    expect(container.textContent).not.toContain("articleArtifacts");
    expect(container.textContent).toContain(
      "这是第一段正文，应该在独立产物框内完整显示。",
    );
    expect(container.textContent).toContain(
      "这是第二段正文，点击框头后进入右侧文章编辑器。",
    );

    act(() => {
      openButton?.click();
    });

    expect(onArtifactClick).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "preview-article-workspace-article",
        meta: expect.objectContaining({
          openedFrom: "right_surface_article_workspace",
          articleWorkspace: expect.objectContaining({
            objectKind: "articleDraft",
          }),
        }),
      }),
    );
  });
});
