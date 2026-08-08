import { describe, expect, it } from "vitest";
import type { Artifact } from "@/lib/artifact/types";
import { resolveArticleArtifactFrameModel } from "./articleArtifactProjection";

function createWorkspacePatchArtifact(
  overrides: Partial<Artifact> = {},
): Artifact {
  const articleWorkspace = {
    schemaVersion: "article-workspace.v1",
    appId: "content-factory-app",
    sessionId: "session-main",
    workspaceId: "workspace-main",
    selectedObjectRef: {
      appId: "content-factory-app",
      kind: "articleDraft",
      id: "article-2",
      sessionId: "session-main",
    },
    objects: [
      {
        ref: {
          appId: "content-factory-app",
          kind: "articleDraft",
          id: "article-1",
          sessionId: "session-main",
        },
        title: "首稿",
        status: "ready",
        summary: "第一版",
        source: {
          processMarkdown: "## 过程稿\n\n第一版检索与结构草案。",
          documentText: "# 首稿\n\n第一版文章内容。",
        },
      },
      {
        ref: {
          appId: "content-factory-app",
          kind: "articleDraft",
          id: "article-2",
          sessionId: "session-main",
        },
        title: "二稿",
        status: "ready",
        summary: "第二版",
        source: {
          processMarkdown: "## 过程稿\n\n第二版写作过程。",
          documentText: "# 二稿\n\n第二版文章内容。",
        },
      },
    ],
  };

  const content = JSON.stringify(articleWorkspace);
  return {
    id: overrides.id ?? "artifact-workspace-patch",
    type: overrides.type ?? "document",
    title: overrides.title ?? "内容工厂工作区补丁",
    content: overrides.content ?? content,
    status: overrides.status ?? "complete",
    meta: {
      openedFrom: "right_surface_article_workspace",
      workspacePatch: articleWorkspace,
      ...overrides.meta,
    },
    position: overrides.position ?? { start: 0, end: content.length },
    createdAt: overrides.createdAt ?? 1,
    updatedAt: overrides.updatedAt ?? 1,
    error: overrides.error,
  } as Artifact;
}

describe("articleArtifactProjection", () => {
  it("应按对象 key 选中当前草稿而不是退回第一个对象", () => {
    const model = resolveArticleArtifactFrameModel(
      createWorkspacePatchArtifact(),
    );

    expect(model).not.toBeNull();
    expect(model).toMatchObject({
      title: "内容工厂工作区补丁",
      markdown: expect.stringContaining("第二版文章内容"),
      summary: "第二版",
    });
  });

  it("不得从旧 contentFactoryWorkspacePatch 字段恢复文章", () => {
    const legacyWorkspacePatch = {
      schemaVersion: "article-workspace.v1",
      appId: "content-factory-app",
      sessionId: "session-main",
      selectedObjectRef: {
        appId: "content-factory-app",
        kind: "articleDraft",
        id: "legacy-article",
        sessionId: "session-main",
      },
      objects: [
        {
          ref: {
            appId: "content-factory-app",
            kind: "articleDraft",
            id: "legacy-article",
            sessionId: "session-main",
          },
          title: "旧历史稿",
          status: "ready",
          summary: "旧字段",
          source: {
            documentText: "# 旧历史稿\n\n来自旧字段。",
          },
        },
      ],
    };
    const model = resolveArticleArtifactFrameModel(
      createWorkspacePatchArtifact({
        meta: {
          workspacePatch: null,
          contentFactoryWorkspacePatch: legacyWorkspacePatch,
        },
      }),
    );

    expect(model).toBeNull();
  });

  it("仅有过程稿时不应把 processMarkdown 当成正式文章展示", () => {
    const model = resolveArticleArtifactFrameModel(
      createWorkspacePatchArtifact({
        meta: {
          workspacePatch: {
            schemaVersion: "article-workspace.v1",
            appId: "content-factory-app",
            sessionId: "session-main",
            selectedObjectRef: {
              appId: "content-factory-app",
              kind: "articleDraft",
              id: "article-process-only",
              sessionId: "session-main",
            },
            objects: [
              {
                ref: {
                  appId: "content-factory-app",
                  kind: "articleDraft",
                  id: "article-process-only",
                  sessionId: "session-main",
                },
                title: "过程稿",
                status: "generating",
                source: {
                  processMarkdown: "## 过程稿\n\n这里只能出现在对话过程里。",
                },
              },
            ],
          },
        },
      }),
    );

    expect(model).toBeNull();
  });
});
