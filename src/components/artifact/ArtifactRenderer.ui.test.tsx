import { act, type ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import { ArtifactRenderer } from "./ArtifactRenderer";
import { ArtifactDocumentRenderer } from "./renderers/ArtifactDocumentRenderer";
import {
  areLightweightRenderersRegistered,
  registerLightweightRenderers,
} from "./renderers";
import type { ArtifactDocumentV1 } from "@/lib/artifact-document";
import type { Artifact } from "@/lib/artifact/types";

interface MountedRenderer {
  container: HTMLDivElement;
  root: Root;
}

const mountedRenderers: MountedRenderer[] = [];

function createArtifact(overrides: Partial<Artifact> = {}): Artifact {
  const content = overrides.content ?? "";
  return {
    id: overrides.id ?? "artifact-1",
    type: overrides.type ?? "document",
    title: overrides.title ?? "demo.md",
    content,
    status: overrides.status ?? "pending",
    meta: {
      filePath: overrides.meta?.filePath ?? "workspace/demo.md",
      filename: overrides.meta?.filename ?? "demo.md",
      ...overrides.meta,
    },
    position: overrides.position ?? { start: 0, end: content.length },
    createdAt: overrides.createdAt ?? 1,
    updatedAt: overrides.updatedAt ?? 1,
    error: overrides.error,
  };
}

function renderArtifact(
  artifact: Artifact,
  props: Partial<ComponentProps<typeof ArtifactRenderer>> = {},
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(
      <ArtifactRenderer artifact={artifact} tone="light" {...props} />,
    );
  });

  mountedRenderers.push({ container, root });
  return container;
}

function renderArtifactDocument(artifactDocument: ArtifactDocumentV1) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(
      <ArtifactDocumentRenderer document={artifactDocument} tone="light" />,
    );
  });

  mountedRenderers.push({ container, root });
  return container;
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  if (!areLightweightRenderersRegistered()) {
    registerLightweightRenderers();
  }

  await changeLimeLocale("zh-CN");
});

afterEach(async () => {
  while (mountedRenderers.length > 0) {
    const mounted = mountedRenderers.pop();
    if (!mounted) {
      break;
    }
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }

  await changeLimeLocale("zh-CN");
});

describe("ArtifactRenderer 空内容态", () => {
  it("流式写入但暂无内容时应展示类型化骨架", () => {
    const container = renderArtifact(
      createArtifact({
        type: "code",
        title: "index.ts",
        status: "streaming",
        meta: {
          filePath: "workspace/index.ts",
          writePhase: "streaming",
          language: "typescript",
        },
      }),
    );

    const surface = container.querySelector(
      '[data-testid="artifact-empty-surface"]',
    );

    expect(surface).not.toBeNull();
    expect(surface?.getAttribute("data-empty-mode")).toBe("writing");
    expect(container.textContent).toContain("正在写入");
    expect(container.textContent).toContain("workspace/index.ts");
  });

  it("应通过 workspace namespace 渲染英文空内容态和未知类型回退", async () => {
    await changeLimeLocale("en-US");

    const writingContainer = renderArtifact(
      createArtifact({
        type: "code",
        title: "index.ts",
        status: "streaming",
        meta: {
          filePath: "workspace/index.ts",
          writePhase: "streaming",
          language: "typescript",
        },
      }),
      { isStreaming: true },
    );

    expect(writingContainer.textContent).toContain("Writing");
    expect(writingContainer.textContent).toContain(
      "Content is still being written.",
    );
    expect(writingContainer.textContent).toContain("Generating...");
    expect(writingContainer.textContent).not.toContain("正在写入");

    const fallbackContainer = renderArtifact(
      createArtifact({
        type: "custom" as Artifact["type"],
        title: "unknown.artifact",
        status: "complete",
        content: "raw payload",
      }),
    );

    expect(fallbackContainer.textContent).toContain("Unknown type");
    expect(fallbackContainer.textContent).toContain(
      'Type "custom" does not have a renderer, so raw content is shown.',
    );
    expect(fallbackContainer.textContent).toContain("raw payload");
    expect(fallbackContainer.textContent).not.toContain("未知类型");
  });

  it("canvas:design 应先展示紧凑预览，不在工作台卡片内嵌完整编辑器", async () => {
    const container = renderArtifact(
      createArtifact({
        type: "canvas:design",
        title: "design.json",
        status: "complete",
        content: JSON.stringify({
          id: "design-artifact-ui",
          title: "Artifact 图层海报",
          canvas: { width: 1080, height: 1440 },
          layers: [
            {
              id: "headline",
              name: "标题层",
              type: "text",
              text: "可编辑标题",
              x: 120,
              y: 120,
              width: 840,
              height: 120,
              zIndex: 4,
            },
          ],
          assets: [],
          editHistory: [],
          createdAt: "2026-05-05T00:00:00.000Z",
          updatedAt: "2026-05-05T00:00:00.000Z",
        }),
        meta: {
          filePath: "workspace/design.json",
          filename: "design.json",
        },
      }),
    );

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).toContain("图层设计 Canvas");
    expect(container.textContent).toContain("LayeredDesignDocument");
    expect(container.textContent).toContain("Artifact 图层海报");
    expect(container.textContent).toContain("标题层");
    expect(container.textContent).not.toContain("生成全部图片层");
    expect(container.textContent).not.toContain("导出设计工程");
  });

  it("canvas:design 完整编辑器应继承主应用项目上下文，支持工程目录保存和图层生成", async () => {
    const container = renderArtifact(
      createArtifact({
        type: "canvas:design",
        title: "design.json",
        status: "complete",
        content: JSON.stringify({
          id: "design-artifact-context",
          title: "主应用工程上下文",
          canvas: { width: 1080, height: 1440 },
          layers: [
            {
              id: "hero",
              name: "主视觉",
              type: "image",
              assetId: "asset-hero",
              x: 0,
              y: 0,
              width: 1080,
              height: 1440,
              zIndex: 1,
              alphaMode: "none",
            },
          ],
          assets: [
            {
              id: "asset-hero",
              kind: "subject",
              src: "",
              width: 1080,
              height: 1440,
              hasAlpha: false,
              createdAt: "2026-05-08T00:00:00.000Z",
            },
          ],
          editHistory: [],
          createdAt: "2026-05-08T00:00:00.000Z",
          updatedAt: "2026-05-08T00:00:00.000Z",
        }),
        meta: {
          filePath: ".lime/layered-designs/demo.layered-design/design.json",
          filename: "design.json",
        },
      }),
      {
        canvasFactoryProps: {
          projectRootPath: "/workspace/project",
          projectId: "project-1",
          contentId: "content-1",
          imageGenerationProviderId: "fal",
          imageGenerationModelId: "fal-ai/nano-banana-pro",
          imageGenerationSelectionReady: true,
          imageGenerationSelectionWarning: null,
        },
      },
    );

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).not.toContain("生成全部图片层");
    expect(container.textContent).not.toContain("打开最近工程");

    const editButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.getAttribute("title") === "在完整编辑器中打开",
    );
    expect(editButton).toBeTruthy();

    act(() => {
      editButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    await act(async () => {
      await Promise.resolve();
    });

    const fullEditor = document.body.querySelector(
      '[data-testid="canvas-full-editor"]',
    );
    expect(fullEditor).not.toBeNull();

    const generateButton = Array.from(
      fullEditor?.querySelectorAll("button") ?? [],
    ).find((button) => button.textContent?.includes("生成全部图片层"));
    const restoreButton = Array.from(
      fullEditor?.querySelectorAll("button") ?? [],
    ).find((button) => button.textContent?.includes("打开最近工程"));

    expect(generateButton).toBeTruthy();
    expect(generateButton?.getAttribute("title")).toBe(
      "为所有待生成图片层创建图片任务",
    );
    expect(generateButton?.hasAttribute("disabled")).toBe(false);
    expect(restoreButton?.getAttribute("title")).toBe(
      "从项目工程目录打开最近保存的图层设计",
    );
    expect(restoreButton?.hasAttribute("disabled")).toBe(false);
  });

  it("canvas:design 的编辑按钮应打开不受预览卡片宽度限制的完整编辑器", async () => {
    const container = renderArtifact(
      createArtifact({
        type: "canvas:design",
        title: "design.json",
        status: "complete",
        content: JSON.stringify({
          id: "design-artifact-full-editor",
          title: "完整编辑器图层设计",
          canvas: { width: 1080, height: 1440 },
          layers: [
            {
              id: "headline",
              name: "标题层",
              type: "text",
              text: "完整编辑器",
              x: 120,
              y: 120,
              width: 840,
              height: 120,
              zIndex: 4,
            },
          ],
          assets: [],
          editHistory: [],
          createdAt: "2026-05-08T00:00:00.000Z",
          updatedAt: "2026-05-08T00:00:00.000Z",
        }),
        meta: {
          filePath: ".lime/layered-designs/full.layered-design/design.json",
          filename: "design.json",
        },
      }),
      {
        canvasFactoryProps: {
          projectRootPath: "/workspace/project",
          projectId: "project-1",
          contentId: "content-1",
        },
      },
    );

    await act(async () => {
      await Promise.resolve();
    });

    const editButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.getAttribute("title") === "在完整编辑器中打开",
    );
    expect(editButton).toBeTruthy();

    act(() => {
      editButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    await act(async () => {
      await Promise.resolve();
    });

    const fullEditor = document.body.querySelector(
      '[data-testid="canvas-full-editor"]',
    );
    expect(fullEditor).not.toBeNull();
    expect(fullEditor?.parentElement).toBe(document.body);
    expect(fullEditor?.getAttribute("role")).toBe("dialog");
    expect(fullEditor?.getAttribute("aria-modal")).toBe("true");
    expect(document.body.style.overflow).toBe("hidden");
    expect(fullEditor?.textContent).toContain("完整编辑器");
    expect(fullEditor?.textContent).toContain("导出设计工程");
    expect(fullEditor?.textContent).toContain("打开最近工程");

    act(() => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      document.body.querySelector('[data-testid="canvas-full-editor"]'),
    ).toBeNull();
    expect(document.body.style.overflow).toBe("");
  });

  it("失败且没有内容时应展示错误解释态", () => {
    const container = renderArtifact(
      createArtifact({
        status: "error",
        error: "保存失败",
        meta: {
          filePath: "workspace/broken.md",
          writePhase: "failed",
        },
      }),
    );

    const surface = container.querySelector(
      '[data-testid="artifact-empty-surface"]',
    );

    expect(surface?.getAttribute("data-empty-mode")).toBe("failed");
    expect(container.textContent).toContain("写入未完成");
    expect(container.textContent).toContain("保存失败");
  });

  it("命中 ArtifactDocument 协议时应渲染结构化阅读面", async () => {
    const container = renderArtifact(
      createArtifact({
        content: JSON.stringify({
          schemaVersion: "artifact_document.v1",
          artifactId: "artifact-doc-1",
          kind: "analysis",
          title: "结构化报告标题",
          status: "ready",
          language: "zh-CN",
          summary: "摘要信息",
          blocks: [
            {
              id: "hero-1",
              type: "hero_summary",
              summary: "顶部摘要",
            },
            {
              id: "body-1",
              type: "rich_text",
              contentFormat: "markdown",
              content: "正文段落",
            },
          ],
          sources: [],
          metadata: {
            theme: "general",
          },
        }),
        status: "complete",
        meta: {
          filePath: "workspace/report.json",
          filename: "report.json",
          language: "json",
        },
      }),
    );

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="artifact-document-renderer"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("结构化报告标题");
    expect(container.textContent).toContain("正文段落");
    expect(container.textContent).toContain("主题 通用");
    expect(
      container.querySelector(
        '[data-testid="artifact-document-renderer"] article',
      )?.className,
    ).toContain("max-w-[1100px]");
  });

  it("content 为空但 metadata 携带 artifactDocument 时也应渲染结构化阅读面", async () => {
    const container = renderArtifact(
      createArtifact({
        content: "",
        status: "complete",
        meta: {
          filePath: ".lime/artifacts/thread-1/report.artifact.json",
          filename: "report.artifact.json",
          artifactSchema: "artifact_document.v1",
          artifactDocument: {
            schemaVersion: "artifact_document.v1",
            artifactId: "artifact-doc-2",
            kind: "report",
            title: "落盘结构化周报",
            status: "ready",
            language: "zh-CN",
            summary: "来自 metadata 的摘要",
            blocks: [
              {
                id: "hero-1",
                type: "hero_summary",
                summary: "来自 metadata 的摘要",
              },
              {
                id: "body-1",
                type: "rich_text",
                contentFormat: "markdown",
                content: "落盘后的正文内容",
              },
            ],
            sources: [],
            metadata: {
              theme: "general",
            },
          },
          language: "json",
        },
      }),
    );

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="artifact-document-renderer"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("落盘结构化周报");
    expect(container.textContent).toContain("落盘后的正文内容");
    expect(container.textContent).toContain("主题 通用");
  });

  it("source-backed 图片 preview artifact 应用媒体视图渲染", () => {
    const container = renderArtifact(
      createArtifact({
        type: "document",
        title: "hero.png",
        content: "asset:///tmp/lime/images/hero.png",
        status: "complete",
        meta: {
          previewArtifact: true,
          isSourceBacked: true,
          contentKind: "image",
          renderMode: "media",
          previewUrl: "asset:///tmp/lime/images/hero.png",
          filePath: "/tmp/lime/images/hero.png",
          filename: "hero.png",
        },
      }),
    );

    const image = container.querySelector<HTMLImageElement>(
      '[data-testid="preview-artifact-image"]',
    );

    expect(image).not.toBeNull();
    expect(image?.getAttribute("src")).toBe(
      "asset:///tmp/lime/images/hero.png",
    );
    expect(image?.getAttribute("alt")).toBe("hero.png");
    expect(container.textContent).not.toContain("<figure");
    expect(
      container.querySelector('[data-testid="artifact-empty-surface"]'),
    ).toBeNull();
  });

  it.each([
    {
      contentKind: "image",
      filename: "broken-image.bin",
      mimeType: "image/x-unsupported",
      testId: "preview-artifact-image",
    },
    {
      contentKind: "audio",
      filename: "broken-audio.bin",
      mimeType: "audio/x-unsupported",
      testId: "preview-artifact-audio",
    },
  ])(
    "$contentKind preview 解码失败时应切到可见的 unsupported 兜底面",
    ({ contentKind, filename, mimeType, testId }) => {
      const container = renderArtifact(
        createArtifact({
          type: "document",
          title: filename,
          content: "blob:unsupported-media",
          status: "complete",
          meta: {
            previewArtifact: true,
            isSourceBacked: true,
            contentKind,
            renderMode: "media",
            previewUrl: "blob:unsupported-media",
            filePath: `workspace/${filename}`,
            filename,
            mimeType,
          },
        }),
      );

      const media = container.querySelector(`[data-testid="${testId}"]`);
      expect(media).not.toBeNull();

      act(() => {
        media?.dispatchEvent(new Event("error"));
      });

      const fallback = container.querySelector(
        '[data-testid="preview-artifact-fallback-surface"]',
      );
      expect(fallback).not.toBeNull();
      expect(fallback?.getAttribute("data-preview-render-mode")).toBe(
        "unsupported",
      );
      expect(container.textContent).toContain("暂不支持内嵌预览");
      expect(container.textContent).toContain(filename);
      expect(container.textContent).toContain(`类型：${mimeType}`);
      expect(container.querySelector(`[data-testid="${testId}"]`)).toBeNull();
    },
  );

  it("system_open preview artifact 应渲染统一兜底面而不是空内容文档", () => {
    const container = renderArtifact(
      createArtifact({
        type: "document",
        title: "deck.pdf",
        content: "该文档暂不支持在工作台内嵌预览。\n\n/tmp/lime/deck.pdf",
        status: "complete",
        meta: {
          previewArtifact: true,
          isSourceBacked: true,
          contentKind: "document",
          renderMode: "system_open",
          sourcePath: "/tmp/lime/deck.pdf",
          filePath: "/tmp/lime/deck.pdf",
          filename: "deck.pdf",
          mimeType: "application/pdf",
          capabilities: {
            preview: false,
            systemOpen: true,
            reveal: true,
          },
        },
      }),
    );

    const surface = container.querySelector(
      '[data-testid="preview-artifact-fallback-surface"]',
    );

    expect(surface).not.toBeNull();
    expect(surface?.getAttribute("data-preview-render-mode")).toBe(
      "system_open",
    );
    expect(container.textContent).toContain("请用系统应用打开");
    expect(container.textContent).toContain("deck.pdf");
    expect(container.textContent).toContain("/tmp/lime/deck.pdf");
    expect(container.textContent).toContain("类型：application/pdf");
    expect(
      container.querySelector('[data-testid="artifact-empty-surface"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="artifact-document-renderer"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain("写入已结束");
  });

  it("unsupported preview artifact 应渲染不可内嵌说明并保留来源错误", () => {
    const container = renderArtifact(
      createArtifact({
        type: "document",
        title: "unknown.bin",
        content: "无法预览该文件。\n\n缺少安全渲染器",
        status: "error",
        error: "缺少安全渲染器",
        meta: {
          previewArtifact: true,
          isSourceBacked: true,
          contentKind: "unsupported",
          renderMode: "unsupported",
          sourcePath: "/tmp/lime/unknown.bin",
          filePath: "/tmp/lime/unknown.bin",
          filename: "unknown.bin",
          previewError: "缺少安全渲染器",
          capabilities: {
            preview: false,
            systemOpen: false,
            reveal: true,
          },
        },
      }),
    );

    const surface = container.querySelector(
      '[data-testid="preview-artifact-fallback-surface"]',
    );

    expect(surface).not.toBeNull();
    expect(surface?.getAttribute("data-preview-render-mode")).toBe(
      "unsupported",
    );
    expect(container.textContent).toContain("暂不支持内嵌预览");
    expect(container.textContent).toContain("unknown.bin");
    expect(container.textContent).toContain("/tmp/lime/unknown.bin");
    expect(container.textContent).toContain("缺少安全渲染器");
    expect(
      container.querySelector('[data-testid="artifact-empty-surface"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain("写入未完成");
  });

  it("URL preview artifact 应渲染来源摘要面而不是普通文档", () => {
    const container = renderArtifact(
      createArtifact({
        type: "document",
        title: "在线报告",
        content: "导入时保留的网页摘要。",
        status: "complete",
        meta: {
          previewArtifact: true,
          isSourceBacked: true,
          source: "url",
          sourceRef: "https://example.com/report",
          sourcePath: "https://example.com/report",
          filePath: "https://example.com/report",
          filename: "report",
          contentKind: "markdown",
          renderMode: "inline",
        },
      }),
    );

    const surface = container.querySelector(
      '[data-testid="preview-source-summary-renderer"]',
    );

    expect(surface).not.toBeNull();
    expect(surface?.getAttribute("data-preview-source")).toBe("url");
    expect(surface?.getAttribute("data-preview-content-kind")).toBe(
      "markdown",
    );
    expect(container.textContent).toContain("网页来源");
    expect(container.textContent).toContain("在线报告");
    expect(container.textContent).toContain("https://example.com/report");
    expect(container.textContent).toContain("导入时保留的网页摘要。");
    expect(
      container.querySelector('[data-testid="artifact-empty-surface"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="artifact-document-renderer"]'),
    ).toBeNull();
  });

  it("database_record preview artifact 应渲染记录摘要面", () => {
    const container = renderArtifact(
      createArtifact({
        type: "document",
        title: "素材记录",
        content: "这条素材来自导入会话的工具输出。",
        status: "complete",
        meta: {
          previewArtifact: true,
          isSourceBacked: true,
          source: "database_record",
          sourceRef: "material:123",
          sourcePath: "material:123",
          recordId: "123",
          contentKind: "text",
          renderMode: "inline",
        },
      }),
    );

    const surface = container.querySelector(
      '[data-testid="preview-source-summary-renderer"]',
    );

    expect(surface).not.toBeNull();
    expect(surface?.getAttribute("data-preview-source")).toBe(
      "database_record",
    );
    expect(container.textContent).toContain("数据记录");
    expect(container.textContent).toContain("素材记录");
    expect(container.textContent).toContain("material:123");
    expect(container.textContent).toContain("123");
    expect(container.textContent).toContain("这条素材来自导入会话的工具输出。");
  });

  it("app_shell preview artifact 应渲染应用入口摘要面", () => {
    const container = renderArtifact(
      createArtifact({
        type: "document",
        title: "研究工作台",
        content: "",
        status: "complete",
        meta: {
          previewArtifact: true,
          isSourceBacked: true,
          source: "app",
          sourceRef: "plugin:research",
          sourcePath: "plugin:research",
          appId: "research",
          contentKind: "app_shell",
          renderMode: "inline",
        },
      }),
    );

    const surface = container.querySelector(
      '[data-testid="preview-source-summary-renderer"]',
    );

    expect(surface).not.toBeNull();
    expect(surface?.getAttribute("data-preview-source")).toBe("app");
    expect(surface?.getAttribute("data-preview-content-kind")).toBe(
      "app_shell",
    );
    expect(container.textContent).toContain("应用入口");
    expect(container.textContent).toContain("研究工作台");
    expect(container.textContent).toContain("plugin:research");
    expect(container.textContent).toContain("当前来源没有可展示的摘要内容。");
    expect(
      container.querySelector('[data-testid="artifact-empty-surface"]'),
    ).toBeNull();
  });

  it("结构化阅读面应渲染摘要卡、统计卡与来源附录", async () => {
    const container = renderArtifact(
      createArtifact({
        content: JSON.stringify({
          schemaVersion: "artifact_document.v1",
          artifactId: "artifact-doc-3",
          kind: "comparison",
          title: "Hermes Engine 评估",
          status: "ready",
          language: "zh-CN",
          summary: "对比不同引擎的交付稳定性与编排成本。",
          blocks: [
            {
              id: "hero-1",
              type: "hero_summary",
              eyebrow: "执行摘要",
              title: "优先选择稳定交付链路",
              summary: "先收口主链，再决定是否引入更激进的并发策略。",
              highlights: ["稳定性优先", "治理成本更低"],
            },
            {
              id: "section-1",
              type: "section_header",
              title: "核心判断",
              description: "按交付可预测性与维护成本排序。",
            },
            {
              id: "body-1",
              type: "rich_text",
              contentFormat: "markdown",
              content: "正文段落",
            },
          ],
          sources: [
            {
              id: "source-1",
              type: "web",
              label: "OpenAI Blog",
              snippet: "作为模型能力背景参考",
              locator: {
                url: "https://openai.com",
              },
            },
          ],
          metadata: {
            theme: "general",
            audience: "产品与工程负责人",
            intent: "技术选型",
          },
        }),
        status: "complete",
        meta: {
          filePath: "workspace/hermes-report.json",
          filename: "hermes-report.json",
          language: "json",
        },
      }),
    );

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).toContain("执行摘要");
    expect(container.textContent).toContain("要点 01");
    expect(container.textContent).toContain("结构块");
    expect(container.textContent).toContain("亮点");
    expect(container.textContent).toContain("来源附录");
    expect(container.textContent).toContain("OpenAI Blog");
    expect(container.textContent).toContain("主题 通用");
    expect(container.textContent).not.toContain("主题 general");
    expect(container.querySelector("#artifact-block-hero-1")).not.toBeNull();
  });

  it("结构化阅读面应复用 CodeRenderer 并在图片缺失时显示占位图", async () => {
    const container = renderArtifactDocument({
      schemaVersion: "artifact_document.v1",
      artifactId: "artifact-doc-4",
      kind: "analysis",
      title: "协议映射回归",
      status: "ready",
      language: "zh-CN",
      blocks: [
        {
          id: "code-1",
          type: "code_block",
          title: "示例代码",
          language: "typescript",
          code: "const answer = 42;",
        },
        {
          id: "image-1",
          type: "image",
          title: "架构图",
          url: "",
          caption: "等待上传正式图片",
        },
      ],
      sources: [],
      metadata: {
        theme: "general",
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.querySelector(".code-renderer")).not.toBeNull();
    expect(container.textContent).toContain("示例代码");
    expect(container.textContent).toContain("typescript");
    expect(container.textContent).toContain("复制");
    expect(container.textContent).toContain("图片占位图");
    expect(container.textContent).toContain("等待上传正式图片");
  });

  it("结构块缺少结构数据时应回退为 rich_text，并删除 citation_list 空块", async () => {
    const container = renderArtifactDocument({
      schemaVersion: "artifact_document.v1",
      artifactId: "artifact-doc-5",
      kind: "report",
      title: "回退行为回归",
      status: "ready",
      language: "zh-CN",
      blocks: [
        {
          id: "table-fallback-1",
          type: "table",
          title: "对比表",
          columns: [],
          rows: [],
          content: "备用表格文本",
        },
        {
          id: "citation-drop-1",
          type: "citation_list",
          title: "不应出现的参考来源",
          items: [],
        },
      ],
      sources: [],
      metadata: {
        theme: "general",
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).toContain("备用表格文本");
    expect(container.textContent).not.toContain("不应出现的参考来源");
  });
});
