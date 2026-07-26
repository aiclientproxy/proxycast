import { describe, expect, it } from "vitest";
import type { WorkspaceArticleWorkspace } from "./workspaceArticleWorkspaceModel";
import {
  applyWorkspaceArticleEditedDraft,
  buildWorkspaceArticleEditedDraftFromChange,
  buildWorkspaceArticleEditedDraftKey,
  buildWorkspaceArticleEditedDraftArtifactWriteParams,
  markdownContainsWorkspaceArticleInlineImageTask,
  shouldRejectWorkspaceArticleEditedDraftChange,
} from "./workspaceArticleWorkspaceEditedDraft";

const articleWorkspace: WorkspaceArticleWorkspace = {
  schemaVersion: "article-workspace.v1",
  appId: "content-factory-app",
  sessionId: "session-main",
  workspaceId: "workspace-main",
  source: "threadRead",
  objectCount: 2,
  actionHistory: [],
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
        documentText: "# 旧正文\n\n这是旧正文。",
        finalMarkdown: "# 旧正文\n\n这是旧正文。",
        researchRounds: [{ id: "research-1", title: "资料检索" }],
      },
    },
    {
      ref: {
        appId: "content-factory-app",
        kind: "imageGenerationSet",
        id: "images-1",
        sessionId: "session-main",
      },
      title: "配图组",
      status: "ready",
      source: {
        images: [{ id: "image-1", title: "主图" }],
      },
    },
  ],
};

describe("workspaceArticleWorkspaceEditedDraft", () => {
  it("应把 Article Editor 本地编辑正文覆盖回对应 articleDraft", () => {
    const articleObject = articleWorkspace.objects[0]!;
    const editedDraft = buildWorkspaceArticleEditedDraftFromChange(
      {
        articleWorkspace,
        object: articleObject,
        markdown: "# 新正文\n\n这是用户编辑后的正文。",
      },
      () => new Date("2026-06-29T10:00:00.000Z"),
    );

    const nextWorkspace = applyWorkspaceArticleEditedDraft(
      articleWorkspace,
      editedDraft,
    );

    expect(editedDraft).toEqual({
      objectKey: buildWorkspaceArticleEditedDraftKey(articleObject),
      markdown: "# 新正文\n\n这是用户编辑后的正文。",
      updatedAt: "2026-06-29T10:00:00.000Z",
    });
    expect(nextWorkspace).not.toBe(articleWorkspace);
    expect(nextWorkspace?.updatedAt).toBe("2026-06-29T10:00:00.000Z");
    expect(nextWorkspace?.objects[0]).toMatchObject({
      source: expect.objectContaining({
        documentText: "# 新正文\n\n这是用户编辑后的正文。",
        finalMarkdown: "# 新正文\n\n这是用户编辑后的正文。",
        updatedAt: "2026-06-29T10:00:00.000Z",
        researchRounds: [{ id: "research-1", title: "资料检索" }],
      }),
    });
    expect(nextWorkspace?.objects[1]).toBe(articleWorkspace.objects[1]);
  });

  it("应把 Article Editor 编辑正文投影为 artifact/write request", () => {
    const articleObject = articleWorkspace.objects[0]!;
    const change = {
      articleWorkspace,
      object: articleObject,
      markdown: "# 新正文\n\n这是用户编辑后的正文。",
    };
    const editedDraft = buildWorkspaceArticleEditedDraftFromChange(
      change,
      () => new Date("2026-06-29T10:00:00.000Z"),
    );

    expect(
      buildWorkspaceArticleEditedDraftArtifactWriteParams(
        change,
        editedDraft,
        " thread-main ",
      ),
    ).toMatchObject({
      threadId: "thread-main",
      artifact: {
        artifactRef: "artifact-article-1",
        artifactDocumentId: "article-1",
        title: "公众号文章草稿",
        kind: "article_workspace_draft",
        status: "draft",
        content: "# 新正文\n\n这是用户编辑后的正文。",
        metadata: {
          editedDraft: {
            objectKey: buildWorkspaceArticleEditedDraftKey(articleObject),
            objectRef: {
              appId: "content-factory-app",
              kind: "articleDraft",
              id: "article-1",
              sessionId: "session-main",
              artifactIds: ["artifact-article-1"],
            },
            markdown: "# 新正文\n\n这是用户编辑后的正文。",
            updatedAt: "2026-06-29T10:00:00.000Z",
          },
          workspacePatch: {
            objects: [
              expect.objectContaining({
                source: expect.objectContaining({
                  documentText: "# 新正文\n\n这是用户编辑后的正文。",
                }),
              }),
              expect.any(Object),
            ],
          },
        },
      },
    });
  });

  it("非 articleDraft 对象不应生成编辑写回", () => {
    const imageObject = articleWorkspace.objects[1]!;

    expect(
      buildWorkspaceArticleEditedDraftFromChange({
        articleWorkspace,
        object: imageObject,
        markdown: "# 不应写回",
      }),
    ).toBeNull();
  });

  it("空正文不应覆盖现有草稿", () => {
    const articleObject = articleWorkspace.objects[0]!;

    expect(
      buildWorkspaceArticleEditedDraftFromChange({
        articleWorkspace,
        object: articleObject,
        markdown: "   ",
      }),
    ).toBeNull();
    expect(applyWorkspaceArticleEditedDraft(articleWorkspace, null)).toBe(
      articleWorkspace,
    );
  });

  it("已有 inline 配图占位时不应接受会丢失 slot 关系的编辑器写回", () => {
    const currentDraft = {
      objectKey: "content-factory-app:session-main:articleDraft:article-1",
      markdown: [
        "# 标题",
        "",
        "![正文配图](pending-image-task://task-inline?status=running&prompt=%E9%85%8D%E5%9B%BE)",
        "<!-- lime:image-task-slot:article-image-slot-1 -->",
      ].join("\n"),
      updatedAt: "2026-06-29T10:00:00.000Z",
    };
    const nextDraft = {
      ...currentDraft,
      markdown: "# 标题\n\n正文被 Tiptap 序列化后丢失了 slot。",
      updatedAt: "2026-06-29T10:01:00.000Z",
    };

    expect(
      markdownContainsWorkspaceArticleInlineImageTask(currentDraft.markdown),
    ).toBe(true);
    expect(
      shouldRejectWorkspaceArticleEditedDraftChange({
        currentDraft,
        nextDraft,
      }),
    ).toBe(true);
  });

  it("已有 inline 配图占位时应接受已解析图片 URL 的回填写回", () => {
    const currentDraft = {
      objectKey: "content-factory-app:session-main:articleDraft:article-1",
      markdown: [
        "# 标题",
        "",
        "![正文配图](pending-image-task://task-inline?status=running&prompt=%E9%85%8D%E5%9B%BE)",
        "<!-- lime:image-task-slot:article-image-slot-1 -->",
      ].join("\n"),
      updatedAt: "2026-06-29T10:00:00.000Z",
    };
    const nextDraft = {
      ...currentDraft,
      markdown: "# 标题\n\n![正文配图](https://example.com/article-image.png)",
      updatedAt: "2026-06-29T10:01:00.000Z",
    };

    expect(
      shouldRejectWorkspaceArticleEditedDraftChange({
        currentDraft,
        nextDraft,
      }),
    ).toBe(false);
  });

  it("仍允许带着 inline 配图占位继续编辑正文", () => {
    const currentDraft = {
      objectKey: "content-factory-app:session-main:articleDraft:article-1",
      markdown: "# 标题\n\n<!-- lime:image-task-slot:article-image-slot-1 -->",
      updatedAt: "2026-06-29T10:00:00.000Z",
    };
    const nextDraft = {
      ...currentDraft,
      markdown:
        "# 标题\n\n新增正文。\n\n<!-- lime:image-task-slot:article-image-slot-1 -->",
      updatedAt: "2026-06-29T10:01:00.000Z",
    };

    expect(
      shouldRejectWorkspaceArticleEditedDraftChange({
        currentDraft,
        nextDraft,
      }),
    ).toBe(false);
  });

  it("内存草稿为空时也应根据当前 object markdown 拦截 slot 降级", () => {
    const nextDraft = {
      objectKey: "content-factory-app:session-main:articleDraft:article-1",
      markdown: "# 标题\n\n正文被序列化后没有占位。",
      updatedAt: "2026-06-29T10:01:00.000Z",
    };

    expect(
      shouldRejectWorkspaceArticleEditedDraftChange({
        currentDraft: null,
        currentMarkdown:
          "# 标题\n\n<!-- lime:image-task-slot:article-image-slot-1 -->",
        nextDraft,
      }),
    ).toBe(true);
  });
});
