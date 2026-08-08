import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Artifact } from "@/lib/artifact/types";
import { changeLimeLocale } from "@/i18n/createI18n";
import { clearArtifactFrameRegistry } from "./artifactFrameRegistry";
import { registerArtifactFrameRenderer } from "./artifactFrameRegistry";
import { MessageArtifactCards } from "./MessageArtifactCards";

vi.mock("./MarkdownRenderer", () => ({
  MarkdownRenderer: ({
    content,
    isStreaming,
    renderMode,
  }: {
    content: string;
    isStreaming?: boolean;
    renderMode?: string;
  }) => (
    <div
      data-testid="markdown-renderer"
      data-is-streaming={isStreaming ? "yes" : "no"}
      data-render-mode={renderMode || "standard"}
    >
      {content}
    </div>
  ),
}));

const mountedRoots: Array<{ container: HTMLDivElement; root: Root }> = [];

function createArtifact(overrides: Partial<Artifact> = {}): Artifact {
  const content = overrides.content ?? "# 公众号文章草稿\n\n正文内容";
  return {
    id: overrides.id ?? "artifact-article-1",
    type: overrides.type ?? "document",
    title: overrides.title ?? "公众号文章草稿",
    content,
    status: overrides.status ?? "complete",
    meta: overrides.meta ?? {},
    position: overrides.position ?? { start: 0, end: content.length },
    createdAt: overrides.createdAt ?? 1,
    updatedAt: overrides.updatedAt ?? 1,
    error: overrides.error,
  };
}

async function renderCards(
  props: React.ComponentProps<typeof MessageArtifactCards>,
) {
  await changeLimeLocale("zh-CN");
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  await act(async () => {
    root.render(<MessageArtifactCards {...props} />);
    await Promise.resolve();
  });
  mountedRoots.push({ container, root });
  return container;
}

afterEach(() => {
  for (const { container, root } of mountedRoots.splice(0)) {
    act(() => {
      root.unmount();
    });
    container.remove();
  }
  clearArtifactFrameRegistry();
});

describe("MessageArtifactCards", () => {
  it("文章产物应在独立 ArtifactFrame 内展示完整正文并可打开右侧编辑器", async () => {
    const onArtifactClick = vi.fn();
    const fullArticle = [
      "# 公众号文章草稿",
      "",
      "这是第一段正文，应该在独立产物框里展示为完整正文。",
      "",
      "## 三轮资料检索",
      "",
      "- 第一轮：确认主题和读者。",
      "- 第二轮：整理场景痛点。",
      "- 第三轮：收敛结构和发布检查。",
      "",
      "## 正文草稿",
      "",
      "这是第二段正文，点击框头后应打开右侧 Article Editor。",
      "",
      "## 长文尾部",
      "",
      "这段尾部内容也应该保留在文章产物框内，点击后可进入右侧编辑器继续编辑。",
    ].join("\n");
    const artifact = createArtifact({
      content: fullArticle,
      meta: {
        appServerArtifactRef: "artifact-article-canonical",
        openedFrom: "right_surface_article_workspace",
        articleWorkspace: {
          objectKind: "articleDraft",
        },
        workspacePatch: {
          workerEvidence: [
            {
              subagents: [
                "content-researcher",
                "content-strategist",
                "article-writer",
              ],
              skillRefs: [
                "article-research",
                "article-strategy",
                "article-writing",
                "article-editing",
                "article-image-plan",
              ],
              researchRounds: [
                { id: "research-1", title: "主题和用户目标检索" },
                { id: "research-2", title: "场景痛点检索" },
                { id: "research-3", title: "发布检查检索" },
              ],
              titleCandidates: [
                { id: "title-1", title: "公众号文章草稿" },
                { id: "title-2", title: "公众号文章备选" },
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
                  id: "plan-strategy",
                  title: "选题策划",
                  owner: "content-strategist",
                  skillRef: "article-strategy",
                },
                {
                  id: "plan-draft",
                  title: "正文写作",
                  owner: "article-writer",
                  skillRef: "article-writing",
                },
                {
                  id: "plan-review",
                  title: "审稿校对",
                  owner: "copy-editor",
                  skillRef: "article-editing",
                },
                {
                  id: "plan-image",
                  title: "配图规划",
                  owner: "image-planner",
                  skillRef: "article-image-plan",
                },
              ],
            },
          ],
        },
        articleWorkspaceCardPreview: {
          counts: {
            researchRounds: 3,
            outlineSections: 5,
            imageSlots: 2,
          },
        },
      },
    });

    const container = await renderCards({
      artifacts: [artifact],
      messageId: "msg-article",
      onArtifactClick,
    });

    const bodyNode = container.querySelector(
      '[data-testid="article-artifact-frame-body"]',
    );
    expect(bodyNode).not.toBeNull();

    expect(
      container.querySelector('[data-testid="article-artifact-frame"]'),
    ).not.toBeNull();
    expect(
      container
        .querySelector('[data-testid="article-artifact-frame"]')
        ?.getAttribute("data-artifact-ref"),
    ).toBe("artifact-article-canonical");
    expect(
      container.querySelector('[data-testid="article-artifact-frame-body"]'),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="article-artifact-frame-markdown"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain("已创建文档：");
    expect(container.textContent).toContain("打开文档");
    expect(container.textContent).toContain("正文草稿");
    expect(container.textContent).not.toContain("articleArtifacts");
    expect(container.textContent).toContain(
      "这是第一段正文，应该在独立产物框里展示为完整正文。",
    );
    expect(container.textContent).toContain(
      "这段尾部内容也应该保留在文章产物框内，点击后可进入右侧编辑器继续编辑。",
    );
    expect(container.textContent).not.toContain("过程摘要");

    const openButton = container.querySelector<HTMLButtonElement>(
      '[data-testid="article-artifact-frame"] button',
    );
    act(() => {
      openButton?.click();
    });

    expect(onArtifactClick).toHaveBeenCalledWith(artifact);
  });

  it("流式文章应直接渲染为 articleArtifacts 文章框", async () => {
    const streamingArticle = createArtifact({
      id: "artifact-article-streaming",
      title: "公众号文章草稿",
      content: "# 公众号文章草稿\n\n正文正在流式生成。",
      status: "streaming",
      meta: {
        openedFrom: "right_surface_article_workspace",
        artifactKind: "articleDraft",
        articleWorkspace: {
          objectKind: "articleDraft",
        },
      },
    });

    const container = await renderCards({
      artifacts: [streamingArticle],
      messageId: "msg-article-streaming",
      onArtifactClick: vi.fn(),
    });

    expect(
      container.querySelector('[data-testid="article-artifact-frame"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="message-artifact-card"]'),
    ).toBeNull();
    expect(container.textContent).toContain("正在创建文档：");
    expect(container.textContent).toContain("流式输出中");
    expect(container.textContent).toContain("正文正在流式生成。");
  });

  it("内容工厂 workspace patch 原始 JSON 不应作为独立文件卡重复显示", async () => {
    const rawWorkspacePatch = createArtifact({
      id: "artifact-workspace-patch",
      title: "workspace-patch.json",
      content: JSON.stringify({
        appId: "content-factory-app",
        objects: [],
      }),
      meta: {
        filePath: ".lime/artifacts/content-factory/workspace-patch.json",
        kind: "workspace_patch",
      },
    });
    const articlePreview = createArtifact({
      id: "article-workspace-preview",
      title: "公众号文章草稿",
      content: "# 公众号文章草稿\n\n正文内容",
      meta: {
        openedFrom: "right_surface_article_workspace",
        articleWorkspace: {
          objectKind: "articleDraft",
        },
      },
    });

    const container = await renderCards({
      artifacts: [rawWorkspacePatch, articlePreview],
      messageId: "msg-article",
      onArtifactClick: vi.fn(),
    });

    expect(
      container.querySelectorAll('[data-testid="article-artifact-frame"]'),
    ).toHaveLength(1);
    expect(
      container.querySelectorAll('[data-testid="message-artifact-card"]'),
    ).toHaveLength(0);
    expect(container.textContent).toContain("公众号文章草稿");
    expect(container.textContent).not.toContain("workspace-patch.json");
    expect(container.textContent).not.toContain('"appId"');
  });

  it("workspace patch 不应被误渲染成第二个文章框", async () => {
    const rawWorkspacePatch = createArtifact({
      id: "artifact-workspace-patch-legacy-path",
      title: "内容工厂工作区补丁",
      content: JSON.stringify({
        appId: "content-factory-app",
        objects: [
          {
            ref: {
              appId: "content-factory-app",
              kind: "articleDraft",
              id: "article-1",
              sessionId: "session-1",
            },
            title: "错误的原始补丁框",
            source: {
              markdown: "# 错误的原始补丁框\n\n不应显示。",
            },
          },
        ],
      }),
      meta: {
        filePath: ".lime/artifacts/workspace-patch.json",
        kind: "workspace_patch",
        workspacePatch: {
          appId: "content-factory-app",
          objects: [],
        },
      },
    });
    const articlePreview = createArtifact({
      id: "article-workspace-preview",
      title: "公众号文章草稿",
      content: "# 公众号文章草稿\n\n正文内容",
      meta: {
        openedFrom: "right_surface_article_workspace",
        articleWorkspace: {
          objectKind: "articleDraft",
        },
      },
    });

    const container = await renderCards({
      artifacts: [rawWorkspacePatch, articlePreview],
      messageId: "msg-article",
      onArtifactClick: vi.fn(),
    });

    expect(
      container.querySelectorAll('[data-testid="article-artifact-frame"]'),
    ).toHaveLength(1);
    expect(container.textContent).toContain("公众号文章草稿");
    expect(container.textContent).not.toContain("内容工厂工作区补丁");
    expect(container.textContent).not.toContain("错误的原始补丁框");
  });

  it("普通文档产物应继续使用轻量文件卡", async () => {
    const artifact = createArtifact({
      id: "artifact-doc-1",
      title: "demo.md",
      content: "# Demo",
      meta: {
        filePath: "docs/demo.md",
      },
    });

    const container = await renderCards({
      artifacts: [artifact],
      messageId: "msg-doc",
    });

    expect(
      container.querySelector('[data-testid="artifact-frame"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="message-artifact-card"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("demo.md");
    expect(container.textContent).toContain("docs/demo.md");
  });

  it("新增的自定义 artifact frame renderer 应能接入消息列表", async () => {
    clearArtifactFrameRegistry();
    registerArtifactFrameRenderer({
      id: "chartArtifacts",
      priority: 20,
      supports: (artifact) => artifact.meta.kind === "chartArtifacts",
      component: ({ artifact }) => (
        <section data-testid="chart-artifact-frame">{artifact.title}</section>
      ),
    });

    const artifact = createArtifact({
      id: "artifact-chart-1",
      title: "图表产物",
      meta: {
        kind: "chartArtifacts",
      },
    });

    const container = await renderCards({
      artifacts: [artifact],
      messageId: "msg-chart",
    });

    expect(
      container.querySelector('[data-testid="chart-artifact-frame"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="message-artifact-card"]'),
    ).toBeNull();
    expect(container.textContent).toContain("图表产物");
  });
});
