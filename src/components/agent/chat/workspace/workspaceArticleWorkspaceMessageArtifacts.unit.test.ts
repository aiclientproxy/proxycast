import { describe, expect, it, vi } from "vitest";
import type { Message } from "../types";
import type { WorkspaceArticleWorkspace } from "./workspaceArticleWorkspaceModel";
import { buildWorkspaceArticleWorkspaceFromThreadRead } from "./workspaceArticleWorkspaceModel";
import {
  attachWorkspaceArticleWorkspacePreviewArtifactToMessages,
  buildWorkspaceArticleWorkspaceFromMessageArtifacts,
  hasWorkspaceArticleWorkspaceMessageArtifactSignals,
} from "./workspaceArticleWorkspaceMessageArtifacts";
import { isWorkspaceArticleWorkspacePreviewArtifact } from "./workspaceArticleWorkspacePreviewArtifact";

vi.mock("@/lib/api/fileSystem", () => ({
  convertLocalFileSrc: (path: string) => `asset://${path}`,
  resolveLocalFilePreviewUrl: (path?: string | null) =>
    path?.startsWith("/") ? `asset://${path}` : null,
}));

const articleWorkspace: WorkspaceArticleWorkspace = {
  schemaVersion: "article-workspace.v1",
  appId: "content-factory-app",
  sessionId: "session-main",
  workspaceId: "workspace-main",
  source: "threadRead",
  objectCount: 1,
  actionHistory: [],
  selectedObjectRef: {
    appId: "content-factory-app",
    kind: "articleDraft",
    id: "article-1",
    sessionId: "session-main",
  },
  objects: [
    {
      ref: {
        appId: "content-factory-app",
        kind: "articleDraft",
        id: "article-1",
        sessionId: "session-main",
        version: "2",
        artifactIds: ["artifact-article-1"],
        sourceTurnId: "turn-article-1",
        sourceTaskId: "task-article-1",
      },
      title: "公众号文章草稿",
      status: "ready",
      source: {
        taskKind: "content.article.generate",
        taskId: "task-article-1",
        documentText: "# 公众号文章草稿\n\n这是正文。",
        finalMarkdown: "# 公众号文章草稿\n\n这是正文。",
      },
    },
  ],
};

function createMessage(overrides: Partial<Message> = {}): Message {
  return {
    id: overrides.id ?? "assistant-1",
    role: overrides.role ?? "assistant",
    content: overrides.content ?? "已整理文章草稿。",
    timestamp: overrides.timestamp ?? new Date("2026-06-28T10:00:00.000Z"),
    ...overrides,
  };
}

describe("workspaceArticleWorkspaceMessageArtifacts", () => {
  it("普通聊天 artifact 不应触发 Article Editor 消息扫描重路径", () => {
    const messages = [
      createMessage({
        id: "assistant-news",
        content: "今天国际新闻已整理。",
        artifacts: [
          {
            id: "artifact-news-summary",
            type: "document",
            title: "国际新闻摘要",
            content: JSON.stringify({
              title: "国际新闻摘要",
              sections: ["中东", "欧洲", "拉美"],
            }),
            status: "complete",
            meta: {
              artifactKind: "report",
            },
            position: { start: 0, end: 0 },
            createdAt: 100,
            updatedAt: 100,
          },
        ],
      }),
    ] satisfies Message[];

    expect(hasWorkspaceArticleWorkspaceMessageArtifactSignals(messages)).toBe(
      false,
    );
    expect(
      buildWorkspaceArticleWorkspaceFromMessageArtifacts(messages),
    ).toBeNull();
  });

  it("消息 artifact 只使用 workspacePatch", () => {
    const messages = [
      createMessage({
        artifacts: [
          {
            id: "artifact-workspace-patch",
            type: "document",
            title: "Article Workspace Patch",
            content: "",
            status: "complete",
            meta: {
              workspacePatch: articleWorkspace,
            },
            position: { start: 0, end: 0 },
            createdAt: 100,
            updatedAt: 100,
          },
        ],
      }),
    ] satisfies Message[];

    const restored =
      buildWorkspaceArticleWorkspaceFromMessageArtifacts(messages);

    expect(restored?.objects[0]?.title).toBe("公众号文章草稿");
  });

  it("应把 Article Editor preview artifact 挂到最后一条助手消息", () => {
    const messages = [
      createMessage({ id: "user-1", role: "user", content: "写文章" }),
      createMessage({ id: "assistant-1" }),
    ];

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages,
        articleWorkspace,
        now: 100,
      });

    expect(nextMessages).toHaveLength(2);
    expect(nextMessages[0]).toBe(messages[0]);
    expect(nextMessages[1]).not.toBe(messages[1]);
    expect(nextMessages[1]?.artifacts).toHaveLength(1);
    expect(nextMessages[1]?.artifacts?.[0]).toMatchObject({
      title: "公众号文章草稿",
      content: expect.stringContaining("这是正文"),
      status: "complete",
      meta: expect.objectContaining({
        openedFrom: "right_surface_article_workspace",
        artifactKind: "report",
        articleWorkspace: expect.objectContaining({
          appId: "content-factory-app",
          objectKind: "articleDraft",
          objectId: "article-1",
          surfaceKind: "document",
        }),
      }),
    });
    expect(
      isWorkspaceArticleWorkspacePreviewArtifact(
        nextMessages[1]!.artifacts![0]!,
      ),
    ).toBe(true);
  });

  it("没有 Article Editor 时不改动消息引用", () => {
    const messages = [createMessage()];

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages,
        articleWorkspace: null,
        now: 100,
      });

    expect(nextMessages).toBe(messages);
  });

  it("重复投影同一个 Article Editor 时不重复插入小卡", () => {
    const firstMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages: [createMessage()],
        articleWorkspace,
        now: 100,
      });
    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages: firstMessages,
        articleWorkspace,
        now: 200,
      });

    expect(nextMessages[0]?.artifacts).toHaveLength(1);
    expect(nextMessages[0]?.artifacts?.[0]?.updatedAt).toBe(200);
  });

  it("同一篇文章完成 inline 配图回填后，应移除其它消息里的旧 pending 小卡", () => {
    const pendingMarkdown = [
      "# 公众号文章草稿",
      "",
      "核心观点段落",
      "",
      "![广州夏天午后街景原位配图](pending-image-task://content-factory-inline-image-task?status=running)",
    ].join("\n");
    const completedMarkdown = [
      "# 公众号文章草稿",
      "",
      "核心观点段落",
      "",
      "![广州夏天午后街景原位配图](https://example.com/lime-fixture-guangzhou-inline.png)",
    ].join("\n");
    const articleWorkspaceWithPendingImage: WorkspaceArticleWorkspace = {
      ...articleWorkspace,
      objects: articleWorkspace.objects.map((object) => ({
        ...object,
        source: {
          ...(object.source ?? {}),
          documentText: pendingMarkdown,
          finalMarkdown: pendingMarkdown,
        },
      })),
    };
    const articleWorkspaceWithCompletedImage: WorkspaceArticleWorkspace = {
      ...articleWorkspace,
      objects: articleWorkspace.objects.map((object) => ({
        ...object,
        source: {
          ...(object.source ?? {}),
          documentText: completedMarkdown,
          finalMarkdown: completedMarkdown,
        },
      })),
    };

    const firstMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages: [
          createMessage({ id: "user-1", role: "user", content: "写文章" }),
          createMessage({
            id: "assistant-1",
            content: "文章草稿已生成。",
          }),
        ],
        articleWorkspace: articleWorkspaceWithPendingImage,
        now: 100,
      });
    expect(firstMessages[1]?.artifacts?.[0]?.content).toContain(
      "pending-image-task://",
    );

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages: [
          ...firstMessages,
          createMessage({
            id: "assistant-2",
            content: "图片任务已完成。",
            timestamp: new Date("2026-06-28T10:01:00.000Z"),
          }),
        ],
        articleWorkspace: articleWorkspaceWithCompletedImage,
        now: 200,
      });
    const allArtifacts = nextMessages.flatMap(
      (message) => message.artifacts ?? [],
    );
    const articlePreviewArtifacts = allArtifacts.filter((artifact) =>
      isWorkspaceArticleWorkspacePreviewArtifact(artifact),
    );

    expect(JSON.stringify(allArtifacts)).not.toContain(
      "pending-image-task://",
    );
    expect(articlePreviewArtifacts).toHaveLength(1);
    expect(articlePreviewArtifacts[0]?.content).toContain(
      "https://example.com/lime-fixture-guangzhou-inline.png",
    );
    expect(nextMessages.find((message) => message.id === "assistant-1"))
      .toMatchObject({
        content: "文章草稿已生成。",
        artifacts: undefined,
      });
    expect(
      nextMessages.find((message) => message.id === "assistant-2")?.artifacts,
    ).toHaveLength(1);
  });

  it("右侧当前选中非文章对象时，聊天小框仍应展示文章正文产物", () => {
    const workspaceWithStoryboardSelected: WorkspaceArticleWorkspace = {
      ...articleWorkspace,
      selectedObjectRef: {
        appId: "content-factory-app",
        kind: "videoStoryboard",
        id: "storyboard-1",
        sessionId: "session-main",
        artifactIds: ["artifact-storyboard-1"],
      },
      objects: [
        ...articleWorkspace.objects,
        {
          ref: {
            appId: "content-factory-app",
            kind: "videoStoryboard",
            id: "storyboard-1",
            sessionId: "session-main",
            artifactIds: ["artifact-storyboard-1"],
          },
          title: "视频分镜",
          status: "ready",
          source: {
            taskKind: "content.video.storyboard.generate",
            scenes: [{ id: "shot-1", title: "开场" }],
          },
        },
      ],
      objectCount: 2,
    };

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages: [createMessage()],
        articleWorkspace: workspaceWithStoryboardSelected,
        now: 100,
      });

    expect(nextMessages[0]?.artifacts?.[0]).toMatchObject({
      title: "公众号文章草稿",
      content: expect.stringContaining("这是正文"),
      meta: expect.objectContaining({
        articleWorkspace: expect.objectContaining({
          objectKind: "articleDraft",
          objectId: "article-1",
          surfaceKind: "document",
        }),
      }),
    });
  });

  it("多个文章草稿并存时，聊天小框应展示多轮检索后的最终稿", () => {
    const finalArticleWorkspace: WorkspaceArticleWorkspace = {
      ...articleWorkspace,
      selectedObjectRef: {
        appId: "content-factory-app",
        kind: "articleDraft",
        id: "article-1",
        sessionId: "session-main",
      },
      primaryObjectRef: {
        appId: "content-factory-app",
        kind: "articleDraft",
        id: "article-1",
        sessionId: "session-main",
      },
      objects: [
        ...articleWorkspace.objects,
        {
          ref: {
            appId: "content-factory-app",
            kind: "articleDraft",
            id: "article-final",
            sessionId: "session-main",
            version: "v3",
            artifactIds: ["artifact-article-final"],
            sourceTaskId: "content-factory-worker-task",
          },
          title: "多轮检索后的公众号文章草稿",
          status: "needs_review",
          summary: "已完成 3 轮资料检索、3 个配图位和完整正文。",
          previewArtifactId: "artifact-article-final",
          source: {
            taskKind: "content.article.generate",
            taskId: "content-factory-worker-task",
            documentText: [
              "# 多轮检索后的公众号文章草稿",
              "",
              "## 三轮资料检索",
              "",
              "- 第一轮：确认用户目标。",
              "- 第二轮：整理场景痛点。",
              "- 第三轮：收敛结构和发布检查。",
              "",
              "## 正文草稿",
              "",
              "这是经过多轮检索后写出的完整正文。",
            ].join("\n"),
            finalMarkdown: [
              "# 多轮检索后的公众号文章草稿",
              "",
              "## 三轮资料检索",
              "",
              "- 第一轮：确认用户目标。",
              "- 第二轮：整理场景痛点。",
              "- 第三轮：收敛结构和发布检查。",
              "",
              "## 正文草稿",
              "",
              "这是经过多轮检索后写出的完整正文。",
            ].join("\n"),
            researchRounds: [
              { id: "research-1", title: "确认用户目标" },
              { id: "research-2", title: "整理场景痛点" },
              { id: "research-3", title: "收敛结构" },
            ],
            outline: [
              { id: "intro", title: "开场", points: [], evidenceIds: [] },
              { id: "research", title: "检索", points: [], evidenceIds: [] },
              { id: "draft", title: "正文", points: [], evidenceIds: [] },
            ],
            citations: [
              { id: "citation-1", title: "用户反馈" },
              { id: "citation-2", title: "Writing 路线图" },
            ],
            imageSlots: [
              { id: "hero", title: "首图" },
              { id: "workflow", title: "流程图" },
              { id: "canvas", title: "画布图" },
            ],
            writingPlan: [
              { id: "plan-1", title: "资料检索" },
              { id: "plan-2", title: "正文写作" },
            ],
          },
        },
      ],
      objectCount: 2,
    };

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages: [createMessage()],
        articleWorkspace: finalArticleWorkspace,
        now: 100,
      });

    expect(nextMessages[0]?.artifacts?.[0]).toMatchObject({
      title: "多轮检索后的公众号文章草稿",
      content: expect.stringContaining("## 三轮资料检索"),
      meta: expect.objectContaining({
        articleWorkspace: expect.objectContaining({
          objectKind: "articleDraft",
          objectId: "article-final",
          artifactIds: ["artifact-article-final"],
        }),
        articleWorkspaceCardPreview: expect.objectContaining({
          counts: expect.objectContaining({
            researchRounds: 3,
            imageSlots: 3,
          }),
        }),
      }),
    });
    expect(nextMessages[0]?.artifacts?.[0]?.content).toContain("## 正文草稿");
    expect(nextMessages[0]?.artifacts?.[0]?.content).not.toContain(
      "这是正文。",
    );
  });

  it("只有过程稿时不应追加文章产物小卡", () => {
    const processOnlyWorkspace: WorkspaceArticleWorkspace = {
      ...articleWorkspace,
      objects: [
        {
          ...articleWorkspace.objects[0]!,
          status: "generating",
          source: {
            taskKind: "content.article.generate",
            processMarkdown: "## 资料检索\n\n这里只能作为对话过程展示。",
          },
        },
      ],
    };
    const messages = [
      createMessage({ id: "user-1", role: "user", content: "@写文章" }),
      createMessage({
        id: "assistant-streaming",
        content: "正在检索资料",
        isThinking: true,
      }),
    ];

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages,
        articleWorkspace: processOnlyWorkspace,
        now: 100,
        status: "streaming",
      });

    expect(nextMessages).toBe(messages);
  });

  it("正在流式输出的助手消息不应被小卡替换，但应追加独立文章产物框", () => {
    const messages = [
      createMessage({ id: "user-1", role: "user", content: "继续写" }),
      createMessage({
        id: "assistant-streaming",
        content: "正在整理",
        isThinking: true,
        runtimeStatus: {
          phase: "preparing",
          title: "准备执行",
          detail: "正在连接运行时",
        },
      }),
    ];

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages,
        articleWorkspace,
        now: 100,
      });

    expect(nextMessages).toHaveLength(3);
    expect(nextMessages[1]).toBe(messages[1]);
    expect(nextMessages[1]?.artifacts).toBeUndefined();
    expect(nextMessages[2]).toMatchObject({
      id: expect.stringMatching(/^article-workspace-preview:/),
      role: "assistant",
      content: "",
    });
    expect(nextMessages[2]?.artifacts?.[0]).toMatchObject({
      title: "公众号文章草稿",
      meta: expect.objectContaining({
        openedFrom: "right_surface_article_workspace",
      }),
    });
  });

  it("发送中已有文章 workspace 时，小框应保持流式状态并展示当前正文", () => {
    const messages = [
      createMessage({ id: "user-1", role: "user", content: "@写文章" }),
      createMessage({
        id: "assistant-streaming",
        content: "正在写文章",
        isThinking: true,
        runtimeStatus: {
          phase: "synthesizing",
          title: "正在生成",
          detail: "正在输出文章正文",
        },
      }),
    ];

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages,
        articleWorkspace,
        now: 100,
        status: "streaming",
      });

    expect(nextMessages).toHaveLength(3);
    expect(nextMessages[2]?.artifacts?.[0]).toMatchObject({
      title: "公众号文章草稿",
      content: expect.stringContaining("这是正文"),
      status: "streaming",
    });
  });

  it("没有助手消息时应补一条只包含小卡的助手消息", () => {
    const messages = [
      createMessage({ id: "user-1", role: "user", content: "写文章" }),
    ];

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages,
        articleWorkspace,
        now: 100,
      });

    expect(nextMessages).toHaveLength(2);
    expect(nextMessages[1]).toMatchObject({
      id: expect.stringMatching(/^article-workspace-preview:/),
      role: "assistant",
      content: "",
      timestamp: new Date(100),
    });
    expect(nextMessages[1]?.artifacts).toHaveLength(1);
  });

  it("历史恢复只有用户消息时，也应从 thread read workspace 补完整文章小框", () => {
    const restoredArticleWorkspace =
      buildWorkspaceArticleWorkspaceFromThreadRead({
        thread_id: "thread-main",
        article_workspace: {
          schemaVersion: "article-workspace.v1",
          appId: "content-factory-app",
          sessionId: "session-main",
          selectedObjectRef: {
            appId: "content-factory-app",
            kind: "articleDraft",
            id: "article-restored",
            sessionId: "session-main",
          },
          objects: [
            {
              ref: {
                appId: "content-factory-app",
                kind: "articleDraft",
                id: "article-restored",
                sessionId: "session-main",
                artifactIds: ["artifact-restored"],
              },
              title: "恢复后的公众号文章",
              status: "needs_review",
              source: {
                taskKind: "content.article.generate",
                documentText:
                  "# 恢复后的公众号文章\n\n这是从历史 read model 恢复的完整正文。",
                finalMarkdown:
                  "# 恢复后的公众号文章\n\n这是从历史 read model 恢复的完整正文。",
              },
            },
          ],
        },
      });

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages: [
          createMessage({
            id: "user-1",
            role: "user",
            content: "@写文章 写一篇公众号文章",
          }),
        ],
        articleWorkspace: restoredArticleWorkspace,
        now: 100,
      });

    expect(nextMessages).toHaveLength(2);
    expect(nextMessages[1]?.artifacts?.[0]).toMatchObject({
      title: "恢复后的公众号文章",
      content: expect.stringContaining("历史 read model 恢复的完整正文"),
      status: "complete",
      meta: expect.objectContaining({
        workspacePatch: expect.objectContaining({
          objects: expect.arrayContaining([
            expect.objectContaining({
              title: "恢复后的公众号文章",
              source: expect.objectContaining({
                documentText: expect.stringContaining("完整正文"),
                finalMarkdown: expect.stringContaining("完整正文"),
              }),
            }),
          ]),
        }),
      }),
    });
    expect(nextMessages[1]?.artifacts?.[0]?.meta).not.toHaveProperty(
      "contentFactoryWorkspacePatch",
    );
  });

  it("Article Editor 无可选对象时应 fail closed", () => {
    const invalidProfile: WorkspaceArticleWorkspace = {
      ...articleWorkspace,
      objectCount: 0,
      objects: [],
    };
    const messages = [createMessage()];

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages,
        articleWorkspace: invalidProfile,
        now: 100,
      });

    expect(nextMessages).toBe(messages);
  });

  it("应把 worker read model 中的文章 Markdown 投影成聊天小框", () => {
    const articleWorkspace = buildWorkspaceArticleWorkspaceFromThreadRead({
      thread_id: "thread-main",
      article_workspace: {
        schemaVersion: "article-workspace.v1",
        appId: "content-factory-app",
        sessionId: "session-main",
        selectedObjectRef: {
          appId: "content-factory-app",
          kind: "articleDraft",
          id: "article-1",
          sessionId: "session-main",
        },
        objects: [
          {
            ref: {
              appId: "content-factory-app",
              kind: "articleDraft",
              id: "article-1",
              sessionId: "session-main",
              artifactIds: ["artifact-article-1"],
            },
            title: "公众号文章草稿",
            status: "needs_review",
            source: {
              taskKind: "content.article.generate",
              documentText:
                "# 公众号文章草稿\n\n点击小框后展开右侧 Article Editor，再继续处理配图。",
              finalMarkdown:
                "# 公众号文章草稿\n\n点击小框后展开右侧 Article Editor，再继续处理配图。",
            },
          },
        ],
      },
    });
    expect(articleWorkspace).not.toBeNull();

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages: [
          createMessage({
            id: "user-1",
            role: "user",
            content: "@写文章 写一篇公众号文章",
          }),
          createMessage({ id: "assistant-1", content: "文章草稿已生成。" }),
        ],
        articleWorkspace: articleWorkspace,
        now: 100,
      });

    expect(nextMessages[1]?.artifacts?.[0]).toMatchObject({
      title: "公众号文章草稿",
      content: expect.stringContaining("点击小框后展开右侧 Article Editor"),
      meta: expect.objectContaining({
        openedFrom: "right_surface_article_workspace",
        articleWorkspace: expect.objectContaining({
          objectKind: "articleDraft",
          artifactIds: ["artifact-article-1"],
        }),
      }),
    });
  });

  it("应从消息 artifact 的内容工厂 workspace patch 生成 Article Editor 小框", () => {
    const messages = [
      createMessage({
        id: "user-1",
        role: "user",
        content: "@写文章 写一篇公众号文章",
      }),
      createMessage({
        id: "assistant-1",
        content: "文章草稿已生成。",
        artifacts: [
          {
            id: "artifact-workspace-patch",
            type: "document",
            title: "Content Factory workspace patch",
            content: JSON.stringify({
              schemaVersion: "article-workspace.v1",
              appId: "content-factory-app",
              sessionId: "session-main",
              selectedObjectRef: {
                appId: "content-factory-app",
                kind: "articleDraft",
                id: "article-1",
                sessionId: "session-main",
              },
              objects: [
                {
                  ref: {
                    appId: "content-factory-app",
                    kind: "articleDraft",
                    id: "article-1",
                    sessionId: "session-main",
                    artifactIds: ["artifact-article-1"],
                  },
                  title: "公众号文章草稿",
                  status: "needs_review",
                  summary: "首版文章草稿已生成。",
                  source: {
                    taskKind: "content.article.generate",
                    documentText: "# 公众号文章草稿\n\n这是小框展开后的正文。",
                    finalMarkdown: "# 公众号文章草稿\n\n这是小框展开后的正文。",
                  },
                },
              ],
            }),
            status: "complete",
            meta: {
              kind: "workspace_patch",
            },
            position: { start: 0, end: 0 },
            createdAt: 100,
            updatedAt: 100,
          },
        ],
      }),
    ] satisfies Message[];

    expect(hasWorkspaceArticleWorkspaceMessageArtifactSignals(messages)).toBe(
      true,
    );
    const articleWorkspace =
      buildWorkspaceArticleWorkspaceFromMessageArtifacts(messages);
    expect(articleWorkspace).toMatchObject({
      appId: "content-factory-app",
      sessionId: "session-main",
      selectedObjectRef: {
        kind: "articleDraft",
        id: "article-1",
      },
    });

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages,
        articleWorkspace: articleWorkspace,
        now: 200,
      });

    expect(nextMessages[1]?.artifacts).toHaveLength(1);
    expect(nextMessages[1]?.artifacts?.[0]).toMatchObject({
      title: "公众号文章草稿",
      content: expect.stringContaining("这是小框展开后的正文"),
      meta: expect.objectContaining({
        openedFrom: "right_surface_article_workspace",
        articleWorkspace: expect.objectContaining({
          objectKind: "articleDraft",
          objectId: "article-1",
        }),
      }),
    });
  });

  it("应替换只有 artifactDocument articleWorkspace metadata 的旧文章小框", () => {
    const messages = [
      createMessage({
        id: "assistant-1",
        content: "文章草稿已生成。",
        artifacts: [
          {
            id: "legacy-article-preview",
            type: "document",
            title: "公众号文章草稿",
            content: "# 公众号文章草稿\n\n旧正文。",
            status: "complete",
            meta: {
              artifactDocument: {
                artifactId: "artifact-document:content-factory-app:article-1",
                kind: "article",
                title: "公众号文章草稿",
                metadata: {
                  articleWorkspace: {
                    appId: "content-factory-app",
                    sessionId: "session-main",
                    workspaceId: "workspace-main",
                    objectKind: "articleDraft",
                    objectId: "article-1",
                    artifactIds: ["artifact-article-1"],
                  },
                },
              },
            },
            position: { start: 0, end: 0 },
            createdAt: 100,
            updatedAt: 100,
          },
        ],
      }),
    ] satisfies Message[];

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages,
        articleWorkspace,
        now: 200,
      });

    expect(nextMessages).toHaveLength(1);
    expect(nextMessages[0]?.artifacts).toHaveLength(1);
    expect(nextMessages[0]?.artifacts?.[0]?.id).not.toBe(
      "legacy-article-preview",
    );
    expect(nextMessages[0]?.artifacts?.[0]).toMatchObject({
      content: expect.stringContaining("这是正文。"),
      meta: expect.objectContaining({
        openedFrom: "right_surface_article_workspace",
        articleWorkspace: expect.objectContaining({
          objectKind: "articleDraft",
          objectId: "article-1",
        }),
      }),
    });
  });

  it("应替换 App Server history 投影的旧文章小框", () => {
    const messages = [
      createMessage({
        id: "assistant-1",
        content: "文章草稿已生成。",
        artifacts: [
          {
            id: "app-server-article-preview",
            type: "document",
            title: "公众号文章草稿",
            content: "# 公众号文章草稿\n\n旧正文。",
            status: "complete",
            meta: {
              openedFrom: "app_server_article_workspace",
              workspacePatch: {
                schemaVersion: "article-workspace.v1",
                appId: "content-factory-app",
                sessionId: "session-main",
                workspaceId: "workspace-main",
                objects: [
                  {
                    ref: {
                      appId: "content-factory-app",
                      kind: "articleDraft",
                      id: "article-1",
                      sessionId: "session-main",
                      artifactIds: ["artifact-article-1"],
                    },
                    title: "公众号文章草稿",
                    status: "ready",
                    source: {
                      documentText: "# 公众号文章草稿\n\n旧正文。",
                      finalMarkdown: "# 公众号文章草稿\n\n旧正文。",
                    },
                  },
                ],
              },
            },
            position: { start: 0, end: 0 },
            createdAt: 100,
            updatedAt: 100,
          },
        ],
      }),
    ] satisfies Message[];

    const nextMessages =
      attachWorkspaceArticleWorkspacePreviewArtifactToMessages({
        messages,
        articleWorkspace,
        now: 200,
      });

    expect(nextMessages).toHaveLength(1);
    expect(nextMessages[0]?.artifacts).toHaveLength(1);
    expect(nextMessages[0]?.artifacts?.[0]?.id).not.toBe(
      "app-server-article-preview",
    );
    expect(nextMessages[0]?.artifacts?.[0]?.content).toContain("这是正文。");
  });
});
