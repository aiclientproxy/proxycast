import { describe, expect, it, vi } from "vitest";
import { act } from "react";
import type { CanvasWorkbenchDefaultPreview } from "./CanvasWorkbenchLayout";
import {
  clickByAriaLabel,
  clickNewWorkbenchTool,
  clickPreviewMode,
  clickWorkbenchTab,
  createTaskFile,
  expectNewWorkbenchToolInMenu,
  expectWorkbenchTabNotInNewMenu,
  flushEffects,
  mockListProjectGitCommits,
  mount,
  mockDestroyEmbeddedBrowserView,
  mockListenEmbeddedBrowserViewState,
  mockListenEmbeddedBrowserViewLoadFailed,
  mockReadProjectGitDiff,
  mockMountEmbeddedBrowserView,
  mockIsEmbeddedBrowserHostAvailable,
  mockNavigateEmbeddedBrowserView,
  mockOpenExternalUrlWithSystemBrowser,
  mockReloadEmbeddedBrowserView,
  mockSetEmbeddedBrowserViewBounds,
  mockExecCommand,
  mockToast,
  mountHarness,
} from "./CanvasWorkbenchLayout.testFixtures";

function updateInputValue(element: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype,
    "value",
  )?.set;
  setter?.call(element, value);
  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
}

function createPointerLikeEvent(
  type: string,
  init: ConstructorParameters<typeof MouseEvent>[1] & {
    pointerId?: number;
  } = {},
) {
  const event = new MouseEvent(type, init);
  Object.defineProperty(event, "pointerId", {
    configurable: true,
    value: init.pointerId ?? 1,
  });
  return event;
}

describe("CanvasWorkbenchLayout coding mode", () => {
  it("普通文件预览请求应切到文件内容 tab 而不是停在审查", async () => {
    const onPreviewOpenRequestHandled = vi.fn();
    const defaultPreview = {
      selectionKey:
        "default-preview:outputs/international-news-analysis-2026-06-16.md",
      title: "international-news-analysis-2026-06-16.md",
      content: "# 今日国际新闻分析\n\n正文内容",
      filePath: "outputs/international-news-analysis-2026-06-16.md",
      absolutePath:
        "/workspace/outputs/international-news-analysis-2026-06-16.md",
    } satisfies CanvasWorkbenchDefaultPreview;
    const baseProps = {
      artifacts: [],
      canvasState: null,
      taskFiles: [],
      workspaceRoot: "/workspace",
      workspaceUnavailable: false,
      defaultPreview,
      loadFilePreview: vi.fn(async (path: string) => ({
        path,
        content: "",
        isBinary: false,
        size: 0,
        error: null,
      })),
      onOpenPath: vi.fn(async () => undefined),
      onRevealPath: vi.fn(async () => undefined),
      workbenchMode: "coding" as const,
      changeView: {
        checkpointCount: 1,
        latestCheckpointPath: ".lime/artifacts/thread-1/index.v2.md",
        items: [],
      },
    };
    const harness = mountHarness(baseProps);

    await flushEffects();

    expect(
      harness.container.querySelector(
        '[data-testid="canvas-workbench-panel-changes"]',
      ),
    ).not.toBeNull();

    harness.rerender({
      ...baseProps,
      previewOpenRequest: {
        requestKey: 1,
        filePath: "outputs/international-news-analysis-2026-06-16.md",
      },
      onPreviewOpenRequestHandled,
    });
    await flushEffects();

    expect(onPreviewOpenRequestHandled).toHaveBeenCalledWith(1);
    expect(
      harness.container.querySelector(
        '[data-testid="canvas-workbench-preview-mode-panel"][data-preview-mode="markdown"]',
      ),
    ).not.toBeNull();
    expect(
      harness.container.querySelector(
        '[data-testid="canvas-workbench-markdown-preview"]',
      )?.textContent,
    ).toContain("今日国际新闻分析");
    expect(
      harness.container.querySelector(
        '[data-testid="canvas-workbench-panel-changes"]',
      ),
    ).toBeNull();
    const fileTab = harness.container.querySelector<HTMLButtonElement>(
      '[aria-label="切换画布标签-international-news-analysis-2026-06-16.md"]',
    );
    const reviewTab = harness.container.querySelector<HTMLButtonElement>(
      '[aria-label="切换画布标签-审查"]',
    );
    expect(fileTab?.getAttribute("aria-selected")).toBe("true");
    expect(reviewTab).not.toBeNull();
    expect(reviewTab?.getAttribute("aria-selected")).toBe("false");
    clickPreviewMode(harness.container, "Code");
    await flushEffects();
    expect(
      harness.container.querySelector(
        '[data-testid="canvas-workbench-code-preview"]',
      ),
    ).not.toBeNull();
    expect(fileTab?.getAttribute("aria-selected")).toBe("true");
    clickWorkbenchTab(harness.container, "审查");
    await flushEffects();
    expect(
      harness.container.querySelector(
        '[data-testid="canvas-workbench-panel-changes"]',
      ),
    ).not.toBeNull();
  });

  it("coding 模式无运行时变更时也应默认暴露审查入口和预览模式切换", async () => {
    const container = mount({
      artifacts: [],
      canvasState: null,
      taskFiles: [
        createTaskFile(
          "task-current",
          "scratch.md",
          "# 当前画布草稿\n\n等待生成。",
          30,
        ),
      ],
      selectedFileId: "task-current",
      workspaceRoot: "/workspace",
      workspaceUnavailable: false,
      defaultPreview: {
        selectionKey: "task:task-current",
        title: "当前画布草稿",
        content: "# 当前画布草稿\n\n等待生成。",
        filePath: "scratch.md",
        absolutePath: "/workspace/scratch.md",
      } satisfies CanvasWorkbenchDefaultPreview,
      loadFilePreview: vi.fn(async (path: string) => ({
        path,
        content: "",
        isBinary: false,
        size: 0,
        error: null,
      })),
      onOpenPath: vi.fn(async () => undefined),
      onRevealPath: vi.fn(async () => undefined),
      workbenchMode: "coding",
    });

    await flushEffects();

    expect(
      container.querySelector('button[aria-label="切换画布标签-审查"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="canvas-workbench-panel-changes"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("上轮对话");
    expect(container.textContent).toContain("+0");
    expect(container.textContent).toContain("-0");
    expect(container.textContent).toContain("还没有可对比的文件变更。");
    expect(container.textContent).not.toContain("提交或推送");
    expect(container.textContent).not.toContain("创建拉取请求");
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-file-list"]',
      ),
    ).toBeNull();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-files-resizer"]',
      ),
    ).toBeNull();
    expect(
      container.querySelector(
        '[aria-label="调整文件列表宽度"][role="separator"]',
      ),
    ).toBeNull();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-direct-tabs"] [aria-label="切换画布标签-审查"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-direct-tabs"] [aria-label="切换画布标签-scratch.md"]',
      ),
    ).not.toBeNull();
    expectNewWorkbenchToolInMenu(container, "终端");
    expectNewWorkbenchToolInMenu(container, "浏览器");
    expectNewWorkbenchToolInMenu(container, "文件");
    expectWorkbenchTabNotInNewMenu(container, "Markdown");
    expectWorkbenchTabNotInNewMenu(container, "HTML");
    expectWorkbenchTabNotInNewMenu(container, "Code");
    expectWorkbenchTabNotInNewMenu(container, "审查");
    mockReadProjectGitDiff.mockResolvedValueOnce({
      rootPath: "/workspace",
      repositoryRoot: "/workspace",
      hasGitRepository: true,
      patch:
        "diff --git a/src/App.tsx b/src/App.tsx\n--- a/src/App.tsx\n+++ b/src/App.tsx\n@@ -1 +1 @@\n-old\n+new",
      uncommittedFileCount: 1,
    });
    clickByAriaLabel(container, "选择审查基准");
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-base-menu"]',
      ),
    ).not.toBeNull();
    act(() => {
      (
        container.querySelector(
          '[data-testid="canvas-workbench-changes-base-option-unstaged"]',
        ) as HTMLButtonElement
      ).click();
    });
    await flushEffects();
    expect(mockReadProjectGitDiff).toHaveBeenCalledWith(
      "/workspace",
      3,
      "unstaged",
      undefined,
    );
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-base-menu"]',
      ),
    ).toBeNull();
    expect(
      container.querySelector('button[aria-label="选择审查基准"]')?.textContent,
    ).toContain("未暂存");
    expect(container.textContent).toContain("App.tsx");
    expect(container.textContent).toContain("+1");
    expect(container.textContent).toContain("-1");
    clickByAriaLabel(container, "选择审查基准");
    await flushEffects();
    act(() => {
      (
        container.querySelector(
          '[data-testid="canvas-workbench-changes-base-option-previousConversation"]',
        ) as HTMLButtonElement
      ).click();
    });
    await flushEffects();
    expect(
      container.querySelector('button[aria-label="选择审查基准"]')?.textContent,
    ).toContain("上轮对话");
    expect(container.textContent).toContain("还没有可对比的文件变更。");
    clickByAriaLabel(container, "更多审查操作");
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-more-menu"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-more-menu"]',
      )?.className,
    ).toContain("z-[80]");
    expect(container.textContent).toContain("刷新");
    expect(container.textContent).toContain("启用自动执行");
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-file-list"]',
      ),
    ).toBeNull();
    clickByAriaLabel(container, "更多审查操作");
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-more-menu"]',
      ),
    ).toBeNull();
    clickByAriaLabel(container, "显示文件");
    await flushEffects();
    expect(
      container.querySelector('[data-testid="canvas-workbench-panel-changes"]'),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-file-list"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-files-resizer"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[aria-label="调整文件列表宽度"][role="separator"]',
      ),
    ).not.toBeNull();
    clickByAriaLabel(container, "隐藏文件");
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-file-list"]',
      ),
    ).toBeNull();
    clickByAriaLabel(container, "显示文件");
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-file-list"]',
      ),
    ).not.toBeNull();

    clickByAriaLabel(container, "打开工作台切换菜单");
    await flushEffects();
    const tabMenu = container.querySelector(
      '[data-testid="canvas-workbench-tab-menu"]',
    );
    expect(tabMenu?.className).toContain("left-0");
    expect(tabMenu?.className).toContain("top-[calc(100%+5px)]");
    expect(
      tabMenu?.querySelector('[aria-label="切换画布标签-Markdown"]'),
    ).toBeNull();
    clickByAriaLabel(container, "打开工作台切换菜单");
    await flushEffects();
  });

  it("coding 审查入口在 admission 成功后不应永久停留在正在启动", async () => {
    const onStartReview = vi.fn().mockResolvedValue(undefined);
    const container = mount({
      artifacts: [],
      canvasState: null,
      taskFiles: [],
      workspaceRoot: "/workspace",
      workspaceUnavailable: false,
      defaultPreview: null,
      loadFilePreview: vi.fn(async (path: string) => ({
        path,
        content: "",
        isBinary: false,
        size: 0,
        error: null,
      })),
      onOpenPath: vi.fn(async () => undefined),
      onRevealPath: vi.fn(async () => undefined),
      workbenchMode: "coding",
      changeView: {
        items: [],
        reviewThreadId: "thread-1",
        onStartReview,
      },
    });
    await flushEffects();
    const button = container.querySelector(
      '[data-testid="code-review-summary-start-review"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      button?.click();
      await Promise.resolve();
    });
    await flushEffects();

    expect(onStartReview).toHaveBeenCalledTimes(1);
    expect(button?.disabled).toBe(false);
    expect(
      container.querySelector('[data-testid="code-review-summary-review-status"]'),
    ).toBeNull();
  });

  it("coding 模式应固定为预览优先标签，并把文件标签收进文件区", async () => {
    const loadFilePreview = vi.fn(async (path: string) => ({
      path,
      content:
        path === "/workspace/README.md"
          ? "README 内容"
          : "<!doctype html><html><body><h1>更新后的页面</h1></body></html>",
      isBinary: false,
      size: 128,
      error: null,
    }));

    const container = mount({
      artifacts: [],
      canvasState: null,
      taskFiles: [
        createTaskFile(
          "task-current",
          "index.html",
          "<!doctype html><html><body><h1>页面预览</h1></body></html>",
          30,
        ),
      ],
      selectedFileId: "task-current",
      workspaceRoot: "/workspace",
      workspaceUnavailable: false,
      defaultPreview: {
        selectionKey: "task:task-current",
        title: "index.html",
        content: "<!doctype html><html><body><h1>页面预览</h1></body></html>",
        filePath: "index.html",
        absolutePath: "/workspace/index.html",
        previousContent:
          "<!doctype html><html><body><h1>上一版</h1></body></html>",
      } satisfies CanvasWorkbenchDefaultPreview,
      loadFilePreview,
      onOpenPath: vi.fn(async () => undefined),
      onRevealPath: vi.fn(async () => undefined),
      workbenchMode: "coding",
      outputView: {
        tabBadge: "1",
        leadContent: (
          <div data-testid="output-lead-probe">失败输出修复入口</div>
        ),
        renderPanel: () => <div data-testid="output-view">输出摘要</div>,
      },
      logView: {
        tabBadge: "运行中",
        renderPanel: () => <div data-testid="log-view">运行日志</div>,
      },
      topRightTools: <div data-testid="task-center-toolbar-probe">toolbar</div>,
    });

    await flushEffects();

    expect(
      container.querySelector('[data-testid="canvas-workbench-panel-changes"]'),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-top-right-tools"] [data-testid="task-center-toolbar-probe"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-header-actions"]',
      ),
    ).toBeNull();
    expect(container.textContent).toContain("上轮对话");
    expect(
      container.querySelector('[data-testid="canvas-workbench-panel-changes"]')
        ?.className,
    ).not.toContain("rounded-[14px]");
    expect(container.textContent).toContain("index.html");
    expect(
      container.querySelector('button[aria-label="切换画布标签-HTML"]'),
    ).toBeNull();
    expectNewWorkbenchToolInMenu(container, "终端");
    expectNewWorkbenchToolInMenu(container, "浏览器");
    expectNewWorkbenchToolInMenu(container, "文件");
    expectWorkbenchTabNotInNewMenu(container, "Code");
    expectWorkbenchTabNotInNewMenu(container, "审查");
    expectWorkbenchTabNotInNewMenu(container, "输出");
    expectWorkbenchTabNotInNewMenu(container, "日志");
    expect(
      container.querySelector('button[aria-label="切换画布标签-index.html"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('button[aria-label="切换画布标签-结果"]'),
    ).toBeNull();

    clickWorkbenchTab(container, "审查");
    await flushEffects();
    expect(
      container.querySelector('[data-testid="canvas-workbench-panel-changes"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("上轮对话");

    clickNewWorkbenchTool(container, "文件");
    await flushEffects();
    const projectFilesPanel = container.querySelector(
      '[data-testid="canvas-workbench-panel-project-files"]',
    ) as HTMLElement | null;
    const projectFilesResizer = container.querySelector(
      '[data-testid="canvas-workbench-project-files-resizer"]',
    ) as HTMLElement | null;
    expect(projectFilesPanel).not.toBeNull();
    expect(projectFilesPanel?.style.gridTemplateColumns).toContain("34%");
    expect(projectFilesResizer).not.toBeNull();
    expect(projectFilesResizer?.getAttribute("role")).toBe("separator");
    expect(projectFilesResizer?.getAttribute("aria-label")).toBe(
      "调整项目文件宽度",
    );
    Object.defineProperty(projectFilesPanel, "getBoundingClientRect", {
      configurable: true,
      value: () => ({
        bottom: 500,
        height: 420,
        left: 0,
        right: 1000,
        top: 80,
        width: 1000,
        x: 0,
        y: 80,
        toJSON: () => ({}),
      }),
    });
    act(() => {
      projectFilesResizer?.dispatchEvent(
        createPointerLikeEvent("pointerdown", {
          bubbles: true,
          clientX: 600,
          pointerId: 1,
        }),
      );
      window.dispatchEvent(
        createPointerLikeEvent("pointermove", {
          bubbles: true,
          clientX: 520,
          pointerId: 1,
        }),
      );
      window.dispatchEvent(
        createPointerLikeEvent("pointerup", {
          bubbles: true,
          pointerId: 1,
        }),
      );
    });
    await flushEffects();
    expect(projectFilesPanel?.style.gridTemplateColumns).toContain("48%");
    act(() => {
      projectFilesResizer?.dispatchEvent(
        new KeyboardEvent("keydown", {
          bubbles: true,
          key: "ArrowRight",
        }),
      );
    });
    await flushEffects();
    expect(projectFilesPanel?.style.gridTemplateColumns).toContain("46%");
    clickByAriaLabel(container, "选择工作区文件-README.md");
    await flushEffects();

    expect(loadFilePreview).toHaveBeenCalledWith("/workspace/README.md");
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-markdown-preview"]',
      ),
    ).not.toBeNull();
    clickPreviewMode(container, "Code");
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-panel-project-files"] [data-testid="canvas-workbench-code-preview"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector('button[aria-label="切换画布标签-README.md"]'),
    ).not.toBeNull();

    clickNewWorkbenchTool(container, "终端");
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-panel-terminal"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-shell-terminal"]',
      ),
    ).not.toBeNull();
    expect(mockExecCommand).toHaveBeenCalledWith(
      expect.objectContaining({
        command: ["/bin/sh", "-i"],
        cwd: "/workspace",
        tty: true,
        size: { cols: 120, rows: 14 },
      }),
    );

    clickNewWorkbenchTool(container, "浏览器");
    await flushEffects();
    expect(
      container.querySelector('[data-testid="canvas-workbench-panel-browser"]'),
    ).not.toBeNull();
    expect(mockMountEmbeddedBrowserView).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "https://www.google.com/",
      }),
    );
    const browserViewport = container.querySelector(
      '[data-testid="canvas-workbench-browser-viewport"]',
    ) as HTMLElement | null;
    expect(browserViewport).not.toBeNull();
    Object.defineProperty(browserViewport, "getBoundingClientRect", {
      configurable: true,
      value: () => ({
        bottom: 380,
        height: 360,
        left: 10,
        right: 650,
        top: 20,
        width: 640,
        x: 10,
        y: 20,
        toJSON: () => ({}),
      }),
    });
    mockSetEmbeddedBrowserViewBounds.mockClear();
    clickByAriaLabel(container, "打开工作台切换菜单");
    await flushEffects();
    expect(
      container.querySelector('[data-testid="canvas-workbench-tab-menu"]')
        ?.className,
    ).toContain("z-[80]");
    expect(
      container
        .querySelector('[data-testid="canvas-workbench-header-row"]')
        ?.closest("header")?.className,
    ).toContain("z-[90]");
    expect(
      container.querySelector('[data-testid="canvas-workbench-layout"]')
        ?.className,
    ).toContain("z-0");
    expect(mockSetEmbeddedBrowserViewBounds).toHaveBeenLastCalledWith(
      expect.objectContaining({
        bounds: {
          x: 10,
          y: 20,
          width: 640,
          height: 360,
        },
        visible: false,
      }),
    );
    clickByAriaLabel(container, "打开工作台切换菜单");
    await flushEffects();
    expect(
      container.querySelector('[data-testid="canvas-workbench-tab-menu"]'),
    ).toBeNull();
    expect(mockSetEmbeddedBrowserViewBounds).toHaveBeenLastCalledWith(
      expect.objectContaining({
        bounds: {
          x: 10,
          y: 20,
          width: 640,
          height: 360,
        },
        visible: true,
      }),
    );
    const browserAddress = container.querySelector(
      '[aria-label="输入网址或搜索"]',
    ) as HTMLInputElement;
    act(() => {
      updateInputValue(browserAddress, "example.com");
      browserAddress.form?.dispatchEvent(
        new Event("submit", { bubbles: true, cancelable: true }),
      );
    });
    await flushEffects();
    expect(mockNavigateEmbeddedBrowserView).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "https://example.com/",
      }),
    );
    expect(mockMountEmbeddedBrowserView).toHaveBeenCalledTimes(1);
    expect(mockDestroyEmbeddedBrowserView).not.toHaveBeenCalled();
    const stateHandler =
      mockListenEmbeddedBrowserViewState.mock.calls.at(-1)?.[0];
    const loadFailedHandler =
      mockListenEmbeddedBrowserViewLoadFailed.mock.calls.at(-1)?.[0];
    const mountedBrowserViewId =
      mockMountEmbeddedBrowserView.mock.calls.at(-1)?.[0]?.viewId;
    expect(stateHandler).toBeTypeOf("function");
    expect(loadFailedHandler).toBeTypeOf("function");
    expect(mountedBrowserViewId).toBeTypeOf("string");
    act(() => {
      stateHandler?.({
        viewId: mountedBrowserViewId,
        url: "https://example.com/",
        title: "Example Domain",
        canGoBack: false,
        canGoForward: false,
        isLoading: false,
      });
    });
    await flushEffects();
    expect(
      container.querySelector('[aria-label="切换画布标签-Example Domain"]'),
    ).not.toBeNull();
    act(() => {
      loadFailedHandler?.({
        viewId: mountedBrowserViewId,
        url: "https://example.com/",
        title: "Example",
        canGoBack: false,
        canGoForward: false,
        isLoading: false,
        errorCode: -105,
        errorDescription: "NAME_NOT_RESOLVED",
      });
    });
    await vi.waitFor(() => {
      expect(
        container.querySelector(
          '[data-testid="canvas-workbench-browser-error"]',
        )?.textContent,
      ).toContain("NAME_NOT_RESOLVED");
    });
    clickByAriaLabel(container, "在系统浏览器打开");
    await flushEffects();
    expect(mockOpenExternalUrlWithSystemBrowser).toHaveBeenCalled();
    clickByAriaLabel(container, "关闭工作台标签-Example Domain");
    await flushEffects();
    expect(
      container.querySelector('[data-testid="canvas-workbench-panel-browser"]'),
    ).toBeNull();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-panel-terminal"]',
      ),
    ).not.toBeNull();

    clickNewWorkbenchTool(container, "文件");
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-panel-project-files"]',
      ),
    ).not.toBeNull();
    clickByAriaLabel(container, "选择工作区文件-README.md");
    await flushEffects();
    expect(loadFilePreview).toHaveBeenCalledWith("/workspace/README.md");
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-panel-project-files"] [data-testid="canvas-workbench-markdown-preview"]',
      ),
    ).not.toBeNull();
  });

  it("coding 模式选择 Git 基准时应展示非 Git 目录诊断", async () => {
    mockReadProjectGitDiff.mockResolvedValueOnce({
      rootPath: "/workspace/non-git",
      hasGitRepository: false,
      patch: "",
      uncommittedFileCount: 0,
    });

    const container = mount({
      artifacts: [],
      canvasState: null,
      taskFiles: [
        createTaskFile("task-current", "scratch.md", "# 当前草稿", 30),
      ],
      selectedFileId: "task-current",
      workspaceRoot: "/workspace/non-git",
      workspaceUnavailable: false,
      defaultPreview: {
        selectionKey: "task:task-current",
        title: "当前草稿",
        content: "# 当前草稿",
        filePath: "scratch.md",
        absolutePath: "/workspace/non-git/scratch.md",
      } satisfies CanvasWorkbenchDefaultPreview,
      loadFilePreview: vi.fn(async (path: string) => ({
        path,
        content: "",
        isBinary: false,
        size: 0,
        error: null,
      })),
      onOpenPath: vi.fn(async () => undefined),
      onRevealPath: vi.fn(async () => undefined),
      workbenchMode: "coding",
    });

    await flushEffects();

    expect(container.textContent).toContain("还没有可对比的文件变更。");
    clickByAriaLabel(container, "选择审查基准");
    await flushEffects();
    act(() => {
      (
        container.querySelector(
          '[data-testid="canvas-workbench-changes-base-option-unstaged"]',
        ) as HTMLButtonElement
      ).click();
    });
    await flushEffects();

    expect(mockReadProjectGitDiff).toHaveBeenCalledWith(
      "/workspace/non-git",
      3,
      "unstaged",
      undefined,
    );
    expect(container.textContent).toContain(
      "当前目录不是 Git 仓库，无法读取文件变更。",
    );
    expect(container.textContent).not.toContain("还没有可对比的文件变更。");
  });

  it("coding 模式应响应对话侧浏览器打开请求并激活右侧浏览器标签", async () => {
    const handleBrowserRequestHandled = vi.fn();
    const harnessProps = {
      artifacts: [],
      canvasState: null,
      taskFiles: [],
      selectedFileId: undefined,
      workspaceRoot: "/workspace",
      workspaceUnavailable: false,
      defaultPreview: null,
      loadFilePreview: vi.fn(async (path: string) => ({
        path,
        content: "",
        isBinary: false,
        size: 0,
        error: null,
      })),
      onOpenPath: vi.fn(async () => undefined),
      onRevealPath: vi.fn(async () => undefined),
      workbenchMode: "coding" as const,
      browserOpenRequest: {
        requestKey: 1,
        url: "https://example.com/docs",
      },
      onBrowserOpenRequestHandled: handleBrowserRequestHandled,
    };
    const container = mount(harnessProps);

    await flushEffects();

    expect(
      container.querySelector('[data-testid="canvas-workbench-panel-browser"]'),
    ).not.toBeNull();
    expect(mockMountEmbeddedBrowserView).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "https://example.com/docs",
      }),
    );
    expect(handleBrowserRequestHandled).toHaveBeenCalledWith(1);
  });

  it("浏览器标签在非桌面宿主中应 fail closed 且不调用内嵌浏览器命令", async () => {
    mockIsEmbeddedBrowserHostAvailable.mockReturnValue(false);
    const container = mount({
      artifacts: [],
      canvasState: null,
      taskFiles: [],
      selectedFileId: undefined,
      workspaceRoot: "/workspace",
      workspaceUnavailable: false,
      defaultPreview: null,
      loadFilePreview: vi.fn(async (path: string) => ({
        path,
        content: "",
        isBinary: false,
        size: 0,
        error: null,
      })),
      onOpenPath: vi.fn(async () => undefined),
      onRevealPath: vi.fn(async () => undefined),
      workbenchMode: "coding",
    });

    clickNewWorkbenchTool(container, "浏览器");
    await flushEffects();

    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-browser-host-unavailable"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain("需要桌面宿主");
    expect(mockMountEmbeddedBrowserView).not.toHaveBeenCalled();

    const browserAddress = container.querySelector(
      '[aria-label="输入网址或搜索"]',
    ) as HTMLInputElement;
    act(() => {
      updateInputValue(browserAddress, "example.com");
      browserAddress.form?.dispatchEvent(
        new Event("submit", { bubbles: true, cancelable: true }),
      );
    });
    await flushEffects();

    expect(mockNavigateEmbeddedBrowserView).not.toHaveBeenCalled();
    const refreshButton = container.querySelector(
      '[aria-label="刷新浏览器标签"]',
    ) as HTMLButtonElement | null;
    expect(refreshButton?.disabled).toBe(true);
    expect(mockReloadEmbeddedBrowserView).not.toHaveBeenCalled();
  });

  it("coding 模式的变更标签应展示本轮多文件变更队列", async () => {
    const openChangedFile = vi.fn(async () => undefined);
    const loadFilePreview = vi.fn(async (path: string) => ({
      path,
      content:
        path === "/workspace/src/App.tsx"
          ? "export function App() {\n  return <main>Ready full file</main>;\n}"
          : path === "/workspace/index.html"
            ? "<!doctype html><html><body><h1>更新后的页面</h1></body></html>"
            : "",
      isBinary: false,
      size: 128,
      error: null,
    }));
    const container = mount({
      artifacts: [],
      canvasState: null,
      taskFiles: [
        createTaskFile(
          "task-current",
          "index.html",
          "<!doctype html><html><body><h1>更新后的页面</h1></body></html>",
          30,
        ),
      ],
      selectedFileId: "task-current",
      workspaceRoot: "/workspace",
      workspaceUnavailable: false,
      defaultPreview: {
        selectionKey: "task:task-current",
        title: "index.html",
        content:
          "<!doctype html><html><body><h1>更新后的页面</h1></body></html>",
        filePath: "index.html",
        absolutePath: "/workspace/index.html",
        previousContent:
          "<!doctype html><html><body><h1>上一版</h1></body></html>",
      } satisfies CanvasWorkbenchDefaultPreview,
      loadFilePreview,
      onOpenPath: vi.fn(async () => undefined),
      onRevealPath: vi.fn(async () => undefined),
      workbenchMode: "coding",
      changeView: {
        checkpointCount: 2,
        latestCheckpointPath: ".lime/artifacts/thread-1/index.v2.html",
        turnDiff:
          "diff --git a/src/turn-diff.ts b/src/turn-diff.ts\n--- a/src/turn-diff.ts\n+++ b/src/turn-diff.ts\n@@\n+export const exactTurnDiff = true;\n",
        onOpenFile: openChangedFile,
        items: [
          {
            id: "change-index",
            path: "index.html",
            absolutePath: "/workspace/index.html",
            displayName: "index.html",
            source: "runtime",
            status: "completed",
            changeKind: "modified",
            preview: "<h1>更新后的页面</h1>",
            previousContent:
              "<!doctype html><html><body><h1>上一版</h1></body></html>",
            currentContent:
              "<!doctype html><html><body><h1>更新后的页面</h1></body></html>",
          },
          {
            id: "change-app",
            path: "src/App.tsx",
            absolutePath: "/workspace/src/App.tsx",
            displayName: "App.tsx",
            source: "runtime",
            status: "in_progress",
            changeKind: "added",
            preview: "export function App() {}",
            previousContent: null,
            currentContent:
              "export function App() {\n  return <main>Ready</main>;\n}",
          },
        ],
      },
    });

    await flushEffects();

    const changesTab = container.querySelector(
      'button[aria-label="切换画布标签-审查"]',
    );
    expect(changesTab?.textContent).toContain("2");

    clickWorkbenchTab(container, "审查");
    await flushEffects();

    expect(
      container.querySelector('[data-testid="canvas-workbench-panel-changes"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("审查");
    expect(container.textContent).toContain("上轮对话");
    expect(container.textContent).not.toContain("提交或推送");
    expect(container.textContent).not.toContain("创建拉取请求");
    expect(container.textContent).toContain("index.html");
    expect(container.textContent).toContain("快照 2");
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-change-status-mark"][data-change-kind="modified"]',
      )?.textContent,
    ).toBe("M");
    expect(container.textContent).toContain("+1");
    expect(container.textContent).toContain("-1");
    expect(container.textContent).toContain("更新后的页面");
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-file-list"]',
      ),
    ).toBeNull();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-file-filter"]',
      ),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="canvas-workbench-inline-diff"]'),
    ).not.toBeNull();
    clickByAriaLabel(container, "选择审查基准");
    await flushEffects();
    const baseMenu = container.querySelector(
      '[data-testid="canvas-workbench-changes-base-menu"]',
    );
    expect(baseMenu).not.toBeNull();
    expect(baseMenu?.textContent).toContain("未暂存");
    expect(baseMenu?.textContent).toContain("已暂存");
    expect(baseMenu?.textContent).toContain("提交");
    expect(baseMenu?.textContent).toContain("分支");
    expect(baseMenu?.textContent).toContain("上轮对话");
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-base-commit-submenu"]',
      )?.textContent,
    ).toContain("分支上暂无提交记录。");
    act(() => {
      (
        container.querySelector(
          '[data-testid="canvas-workbench-changes-base-option-commit"]',
        ) as HTMLButtonElement
      ).click();
    });
    await flushEffects();
    expect(mockListProjectGitCommits).toHaveBeenCalledWith("/workspace", 30);
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-base-commit-submenu"]',
      )?.textContent,
    ).toContain("整理右侧审查面板");
    act(() => {
      (
        container.querySelector(
          '[data-testid="canvas-workbench-changes-base-commit-option-abc1234"]',
        ) as HTMLButtonElement
      ).click();
    });
    await flushEffects();
    expect(mockReadProjectGitDiff).toHaveBeenCalledWith(
      "/workspace",
      3,
      "commit",
      "abc1234567890",
    );
    expect(
      container.querySelector('button[aria-label="选择审查基准"]')?.textContent,
    ).toContain("提交");
    clickByAriaLabel(container, "选择审查基准");
    await flushEffects();
    expect(
      container
        .querySelector(
          '[data-testid="canvas-workbench-changes-base-option-commit"]',
        )
        ?.getAttribute("aria-checked"),
    ).toBe("true");
    act(() => {
      (
        container.querySelector(
          '[data-testid="canvas-workbench-changes-base-option-unstaged"]',
        ) as HTMLButtonElement
      ).click();
    });
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-base-menu"]',
      ),
    ).toBeNull();
    expect(
      container.querySelector('button[aria-label="选择审查基准"]')?.textContent,
    ).toContain("未暂存");
    expect(mockReadProjectGitDiff).toHaveBeenCalledWith(
      "/workspace",
      3,
      "unstaged",
      undefined,
    );
    clickByAriaLabel(container, "选择审查基准");
    await flushEffects();
    act(() => {
      (
        container.querySelector(
          '[data-testid="canvas-workbench-changes-base-option-branch"]',
        ) as HTMLButtonElement
      ).click();
    });
    await flushEffects();
    expect(mockReadProjectGitDiff).toHaveBeenCalledWith(
      "/workspace",
      3,
      "branch",
      undefined,
    );
    expect(
      container.querySelector('button[aria-label="选择审查基准"]')?.textContent,
    ).toContain("分支");
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-branch-compare"]',
      )?.textContent,
    ).toContain("main -> origin/main");
    expect(
      container.querySelectorAll(
        '[data-testid="canvas-workbench-change-diff-file"]',
      ).length,
    ).toBeGreaterThanOrEqual(1);
    expect(
      container.querySelectorAll(
        '[data-testid="canvas-workbench-split-diff-before"]',
      ).length,
    ).toBe(0);
    clickByAriaLabel(container, "切换到拆分差异视图");
    await flushEffects();
    expect(
      container.querySelectorAll(
        '[data-testid="canvas-workbench-split-diff-after"]',
      ).length,
    ).toBeGreaterThan(0);
    clickByAriaLabel(container, "更多审查操作");
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-more-menu"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-file-list"]',
      ),
    ).toBeNull();
    expect(container.textContent).toContain("刷新");
    expect(container.textContent).toContain("启用自动执行");
    expect(container.textContent).toContain("折叠全部差异");
    expect(container.textContent).toContain("不加载完整文件");
    expect(container.textContent).toContain("启用富文本预览");
    expect(container.textContent).toContain("启用文字差异");
    expect(container.textContent).toContain("隐藏空白字符");
    expect(container.textContent).toContain("复制 git apply 命令");
    expect(container.textContent).not.toContain("关闭自动换行");
    expect(container.textContent).not.toContain("关闭文字差异");
    const menu = container.querySelector(
      '[data-testid="canvas-workbench-changes-more-menu"]',
    );
    expect(menu?.className).toContain("z-[80]");
    expect(menu?.querySelector(".lucide-check")).toBeNull();
    const refreshButton = Array.from(
      container.querySelectorAll(
        '[data-testid="canvas-workbench-changes-more-menu"] button',
      ),
    ).find((button) => button.textContent?.includes("刷新"));
    act(() => {
      (refreshButton as HTMLButtonElement).click();
    });
    await flushEffects();
    expect(mockReadProjectGitDiff).toHaveBeenCalledWith(
      "/workspace",
      3,
      "unstaged",
      undefined,
    );
    expect(mockToast.success).toHaveBeenCalledWith("已刷新变更");
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-more-menu"]',
      ),
    ).toBeNull();

    clickByAriaLabel(container, "更多审查操作");
    await flushEffects();
    const copyGitApplyButton = Array.from(
      container.querySelectorAll(
        '[data-testid="canvas-workbench-changes-more-menu"] button',
      ),
    ).find((button) => button.textContent?.includes("复制 git apply 命令"));
    act(() => {
      (copyGitApplyButton as HTMLButtonElement).click();
    });
    await flushEffects();
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      expect.stringContaining("git apply <<'PATCH'"),
    );
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      expect.stringContaining("diff --git a/src/App.tsx b/src/App.tsx"),
    );
    expect(mockToast.success).toHaveBeenCalledWith("已复制 git apply 命令");

    clickByAriaLabel(container, "选择审查基准");
    await flushEffects();
    act(() => {
      (
        container.querySelector(
          '[data-testid="canvas-workbench-changes-base-option-previousConversation"]',
        ) as HTMLButtonElement
      ).click();
    });
    await flushEffects();
    expect(
      container.querySelector('button[aria-label="选择审查基准"]')?.textContent,
    ).toContain("上轮对话");

    clickByAriaLabel(container, "更多审查操作");
    await flushEffects();
    const previousConversationCopyButton = Array.from(
      container.querySelectorAll(
        '[data-testid="canvas-workbench-changes-more-menu"] button',
      ),
    ).find((button) => button.textContent?.includes("复制 git apply 命令"));
    act(() => {
      (previousConversationCopyButton as HTMLButtonElement).click();
    });
    await flushEffects();
    expect(navigator.clipboard.writeText).toHaveBeenLastCalledWith(
      expect.stringContaining("exactTurnDiff = true"),
    );
    expect(navigator.clipboard.writeText).toHaveBeenLastCalledWith(
      expect.not.stringContaining("diff --git a/index.html b/index.html"),
    );

    clickByAriaLabel(container, "更多审查操作");
    await flushEffects();
    const loadFullFileButton = Array.from(
      container.querySelectorAll(
        '[data-testid="canvas-workbench-changes-more-menu"] button',
      ),
    ).find((button) => button.textContent?.includes("不加载完整文件"));
    act(() => {
      (loadFullFileButton as HTMLButtonElement).click();
    });
    await flushEffects();
    expect(loadFilePreview).toHaveBeenCalledWith("/workspace/index.html");

    clickByAriaLabel(container, "更多审查操作");
    await flushEffects();
    const whitespaceToggle = container.querySelector(
      '[data-testid="canvas-workbench-changes-whitespace-toggle"]',
    ) as HTMLButtonElement;
    act(() => {
      whitespaceToggle.click();
    });
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-more-menu"]',
      ),
    ).toBeNull();
    clickByAriaLabel(container, "更多审查操作");
    await flushEffects();
    expect(container.textContent).toContain("隐藏空白字符");
    const hideWhitespaceToggle = container.querySelector(
      '[data-testid="canvas-workbench-changes-whitespace-toggle"]',
    ) as HTMLButtonElement;
    act(() => {
      hideWhitespaceToggle.click();
    });
    await flushEffects();

    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-file-list"]',
      ),
    ).toBeNull();
    clickByAriaLabel(container, "显示文件");
    await flushEffects();
    expect(
      container
        .querySelector('[data-testid="canvas-workbench-changes-file-filter"]')
        ?.getAttribute("placeholder"),
    ).toBe("筛选文件...");
    expect(container.textContent).toContain("src");
    expect(container.textContent).toContain("App.tsx");

    const changeItems = container.querySelectorAll(
      '[data-testid="canvas-workbench-change-item"]',
    );
    expect(changeItems).toHaveLength(2);
    const fileFilter = container.querySelector(
      '[data-testid="canvas-workbench-changes-file-filter"]',
    ) as HTMLInputElement;
    act(() => {
      updateInputValue(fileFilter, "App");
    });
    await flushEffects();
    expect(
      container.querySelectorAll(
        '[data-testid="canvas-workbench-change-item"]',
      ),
    ).toHaveLength(1);
    expect(container.textContent).toContain("App.tsx");
    act(() => {
      updateInputValue(fileFilter, "");
    });
    await flushEffects();
    expect(
      container.querySelectorAll(
        '[data-testid="canvas-workbench-change-item"]',
      ),
    ).toHaveLength(2);
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-file-list"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-checkpoints"]',
      ),
    ).not.toBeNull();
    const fileList = container.querySelector(
      '[data-testid="canvas-workbench-changes-file-list"]',
    );
    expect(fileList?.textContent).toContain("index.html");
    expect(fileList?.textContent).toContain("App.tsx");
    expect(fileList?.textContent).toContain("<h1>更新后的页面</h1>");
    expect(fileList?.textContent).toContain("export function App() {}");
    expect(
      fileList?.querySelector(
        '[data-testid="canvas-workbench-change-status-mark"][data-change-kind="modified"]',
      )?.textContent,
    ).toBe("M");
    expect(
      fileList?.querySelector(
        '[data-testid="canvas-workbench-change-status-mark"][data-change-kind="added"]',
      )?.textContent,
    ).toBe("A");
    expect(fileList?.textContent).not.toContain("修改");
    expect(fileList?.textContent).not.toContain("新增");
    expect(fileList?.textContent).not.toContain("+3");
    expect(fileList?.textContent).not.toContain("-1");
    const srcFolder = fileList?.querySelector(
      '[data-testid="canvas-workbench-change-folder"][data-change-folder-id="folder:src"]',
    ) as HTMLButtonElement | null;
    expect(srcFolder?.textContent).toContain("A");
    act(() => {
      srcFolder?.click();
    });
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-change-item"][data-change-id="change-app"]',
      ),
    ).toBeNull();
    const collapsedSrcFolder = fileList?.querySelector(
      '[data-testid="canvas-workbench-change-folder"][data-change-folder-id="folder:src"]',
    ) as HTMLButtonElement | null;
    act(() => {
      collapsedSrcFolder?.click();
    });
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-change-item"][data-change-id="change-app"]',
      ),
    ).not.toBeNull();
    clickByAriaLabel(container, "隐藏文件");
    await flushEffects();
    expect(
      container.querySelector('[data-testid="canvas-workbench-panel-changes"]'),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-file-list"]',
      ),
    ).toBeNull();
    clickByAriaLabel(container, "显示文件");
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-file-list"]',
      ),
    ).not.toBeNull();
    clickByAriaLabel(container, "隐藏文件");
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-file-list"]',
      ),
    ).toBeNull();
    expect(container.textContent).toContain("更新后的页面");
    expect(container.textContent).toContain("index.html");

    clickByAriaLabel(container, "显示文件");
    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-changes-file-list"]',
      ),
    ).not.toBeNull();

    const restoredChangeItems = container.querySelectorAll(
      '[data-testid="canvas-workbench-change-item"]',
    );
    expect(restoredChangeItems).toHaveLength(2);
    const appChangeItem = container.querySelector(
      '[data-testid="canvas-workbench-change-item"][data-change-id="change-app"]',
    );
    expect(appChangeItem).not.toBeNull();
    act(() => {
      (appChangeItem as HTMLButtonElement).click();
    });
    await flushEffects();
    expect(openChangedFile).toHaveBeenCalledWith("/workspace/src/App.tsx");
    expect(container.textContent).toContain("App.tsx");
    expect(container.textContent).toContain("src/App.tsx");
    expect(container.textContent).toContain("Ready");

    clickByAriaLabel(container, "更多审查操作");
    await flushEffects();
    const loadSelectedFullFileButton = Array.from(
      container.querySelectorAll(
        '[data-testid="canvas-workbench-changes-more-menu"] button',
      ),
    ).find((button) => button.textContent?.includes("不加载完整文件"));
    act(() => {
      (loadSelectedFullFileButton as HTMLButtonElement).click();
    });
    await flushEffects();
    expect(loadFilePreview).toHaveBeenCalledWith("/workspace/src/App.tsx");
    expect(container.textContent).toContain("Ready full file");
  });

  it("coding 模式应优先展示 runtime 变更证据而不是当前默认预览", async () => {
    const container = mount({
      artifacts: [],
      canvasState: null,
      taskFiles: [
        createTaskFile(
          "task-current",
          ".lime/qc/code-artifact-workbench-electron-fixture/src/greeting.ts",
          "export function greeting() { return 'Hello Lime Workbench'; }",
          30,
        ),
      ],
      selectedFileId: "task-current",
      workspaceRoot: "/workspace",
      workspaceUnavailable: false,
      defaultPreview: {
        selectionKey: "task:task-current",
        title: "greeting.ts",
        content:
          "export function greeting() { return 'Hello Lime Workbench'; }",
        filePath:
          ".lime/qc/code-artifact-workbench-electron-fixture/src/greeting.ts",
        absolutePath:
          "/workspace/.lime/qc/code-artifact-workbench-electron-fixture/src/greeting.ts",
        previousContent: "export function greeting() { return 'Hello Lime'; }",
      } satisfies CanvasWorkbenchDefaultPreview,
      loadFilePreview: vi.fn(async (path: string) => ({
        path,
        content: "",
        isBinary: false,
        size: 0,
        error: null,
      })),
      onOpenPath: vi.fn(async () => undefined),
      onRevealPath: vi.fn(async () => undefined),
      workbenchMode: "coding",
      changeView: {
        checkpointCount: 1,
        latestCheckpointPath: "snapshot",
        items: [
          {
            id: "change-greeting",
            path: ".lime/qc/code-artifact-workbench-electron-fixture/src/greeting.ts",
            displayName: "greeting.ts",
            source: "runtime",
            status: "completed",
            changeKind: "modified",
            preview:
              "export function greeting() { return 'Hello Lime Workbench'; }",
            previousContent: null,
            currentContent:
              "export function greeting() { return 'Hello Lime Workbench'; }",
            checkpointPath: "snapshot",
            checkpointLabel: "snapshot",
          },
          {
            id: "change-coding-target",
            path: ".lime/qc/code-artifact-workbench-electron-fixture/src/coding-target.ts",
            displayName: "coding-target.ts",
            source: "runtime",
            status: "completed",
            changeKind: "modified",
            preview: "export const codingWorkbenchSmoke = true;",
            previousContent: null,
            currentContent: "export const codingWorkbenchSmoke = true;",
            checkpointPath: "snapshot",
            checkpointLabel: "snapshot",
          },
        ],
      },
    });

    await flushEffects();
    clickWorkbenchTab(container, "审查");
    await flushEffects();

    const changesPanel = container.querySelector(
      '[data-testid="canvas-workbench-panel-changes"]',
    );
    expect(changesPanel?.textContent).toContain("coding-target.ts");
    expect(changesPanel?.textContent).toContain(
      "export const codingWorkbenchSmoke = true;",
    );
    expect(changesPanel?.textContent).not.toContain(
      "export function greeting() { return 'Hello Lime Workbench'; }",
    );
  });

  it("coding 模式应以短标识渲染 Git diff 文件块", async () => {
    mockReadProjectGitDiff.mockResolvedValueOnce({
      rootPath: "/workspace",
      repositoryRoot: "/workspace",
      hasGitRepository: true,
      currentRef: "main",
      comparisonBaseRef: "origin/main",
      patch: [
        "diff --git a/src/App.tsx b/src/App.tsx",
        "--- a/src/App.tsx",
        "+++ b/src/App.tsx",
        "@@ -1 +1 @@",
        "-old",
        "+new",
        "diff --git a/docs/new.md b/docs/new.md",
        "new file mode 100644",
        "--- /dev/null",
        "+++ b/docs/new.md",
        "@@ -0,0 +1 @@",
        "+hello",
      ].join("\n"),
      uncommittedFileCount: 2,
    });

    const container = mount({
      artifacts: [],
      canvasState: null,
      taskFiles: [
        createTaskFile("task-current", "scratch.md", "# 当前画布草稿", 30),
      ],
      selectedFileId: "task-current",
      workspaceRoot: "/workspace",
      workspaceUnavailable: false,
      defaultPreview: {
        selectionKey: "task:task-current",
        title: "当前画布草稿",
        content: "# 当前画布草稿",
        filePath: "scratch.md",
        absolutePath: "/workspace/scratch.md",
      } satisfies CanvasWorkbenchDefaultPreview,
      loadFilePreview: vi.fn(async (path: string) => ({
        path,
        content: "",
        isBinary: false,
        size: 0,
        error: null,
      })),
      onOpenPath: vi.fn(async () => undefined),
      onRevealPath: vi.fn(async () => undefined),
      workbenchMode: "coding",
    });

    await flushEffects();
    clickByAriaLabel(container, "选择审查基准");
    await flushEffects();
    act(() => {
      (
        container.querySelector(
          '[data-testid="canvas-workbench-changes-base-option-unstaged"]',
        ) as HTMLButtonElement
      ).click();
    });
    await flushEffects();

    const diffFiles = container.querySelectorAll(
      '[data-testid="canvas-workbench-change-diff-file"]',
    );
    expect(diffFiles).toHaveLength(2);
    const appDiffFile = Array.from(diffFiles).find((element) =>
      element.textContent?.includes("src/App.tsx"),
    );
    const docsDiffFile = Array.from(diffFiles).find((element) =>
      element.textContent?.includes("docs/new.md"),
    );
    expect(appDiffFile?.textContent).toContain("M");
    expect(docsDiffFile?.textContent).toContain("A");
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-change-status-mark"][data-change-kind="modified"]',
      )?.textContent,
    ).toBe("M");
    expect(
      container.querySelector(
        '[data-testid="canvas-workbench-change-status-mark"][data-change-kind="added"]',
      )?.textContent,
    ).toBe("A");
  });
});
