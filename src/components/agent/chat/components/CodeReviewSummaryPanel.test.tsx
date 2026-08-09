import { act, type ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { changeLimeLocale } from "@/i18n/createI18n";
import { CodeReviewSummaryPanel } from "./CodeReviewSummaryPanel";

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

function createEmptyHarnessState(): ComponentProps<
  typeof CodeReviewSummaryPanel
>["harnessState"] {
  return {
    runtimeStatus: null,
    pendingApprovals: [],
    latestContextTrace: [],
    plan: {
      phase: "idle",
      items: [],
    },
    activity: {
      planning: 0,
      filesystem: 0,
      execution: 0,
      web: 0,
      skills: 0,
      delegation: 0,
    },
    delegatedTasks: [],
    outputSignals: [],
    activeFileWrites: [],
    recentFileEvents: [],
    hasSignals: false,
  };
}

function renderPanel(
  overrides: Partial<ComponentProps<typeof CodeReviewSummaryPanel>> = {},
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const onOpenSection = vi.fn<
    ComponentProps<typeof CodeReviewSummaryPanel>["onOpenSection"]
  >();
  const onOpenFileCheckpoints = vi.fn();
  const onSubmitCodeFixPrompt = vi.fn<
    NonNullable<
      ComponentProps<typeof CodeReviewSummaryPanel>["onSubmitCodeFixPrompt"]
    >
  >();

  act(() => {
    root.render(
      <CodeReviewSummaryPanel
        harnessState={{
          ...createEmptyHarnessState(),
          outputSignals: [
            {
              id: "signal-test",
              toolCallId: "tool-test",
              toolName: "bash",
              title: "回归测试结果",
              summary: "vitest passed",
              preview: "1 test passed",
              exitCode: 0,
            },
          ],
          recentFileEvents: [
            {
              id: "event-code",
              toolCallId: "tool-write",
              path: "/tmp/workspace/src/ImageCard.tsx",
              displayName: "ImageCard.tsx",
              kind: "code",
              action: "write",
              sourceToolName: "write_file",
              clickable: true,
            },
          ],
          hasSignals: true,
        }}
        fileCheckpointSummary={{
          count: 1,
          latest_checkpoint: {
            checkpoint_id: "checkpoint-1",
            turn_id: "turn-1",
            path: "src/ImageCard.tsx",
            source: "tool_result",
            updated_at: "2026-05-27T01:00:00.000Z",
            validation_issue_count: 0,
          },
        }}
        onOpenSection={onOpenSection}
        onOpenFileCheckpoints={onOpenFileCheckpoints}
        onSubmitCodeFixPrompt={onSubmitCodeFixPrompt}
        {...overrides}
      />,
    );
  });

  mountedRoots.push({ root, container });
  return {
    container,
    onOpenSection,
    onOpenFileCheckpoints,
    onSubmitCodeFixPrompt,
  };
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  await changeLimeLocale("zh-CN");
});

afterEach(async () => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) {
      break;
    }
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  await changeLimeLocale("en-US");
});

describe("CodeReviewSummaryPanel", () => {
  it("应聚合文件变更、测试输出和快照入口", () => {
    const { container, onOpenSection, onOpenFileCheckpoints } = renderPanel();

    const panel = container.querySelector(
      '[data-testid="code-review-summary-panel"]',
    ) as HTMLElement | null;
    const primaryAction = container.querySelector(
      '[data-testid="code-review-summary-primary-action"]',
    ) as HTMLButtonElement | null;
    const checkpointAction = container.querySelector(
      '[data-testid="code-review-summary-checkpoints"]',
    ) as HTMLButtonElement | null;

    expect(panel?.textContent).toContain("代码审阅摘要");
    expect(panel?.textContent).toContain("待审阅变更");
    expect(panel?.textContent).toContain("文件变更 1");
    expect(panel?.textContent).toContain("测试输出 1");
    expect(panel?.textContent).toContain("快照 1");
    expect(panel?.textContent).toContain("ImageCard.tsx");

    act(() => {
      primaryAction?.click();
    });
    expect(onOpenSection).toHaveBeenCalledWith("file_review");

    act(() => {
      checkpointAction?.click();
    });
    expect(onOpenFileCheckpoints).toHaveBeenCalledTimes(1);
  });

  it("英文界面应使用 agent namespace 文案", async () => {
    await changeLimeLocale("en-US");

    const { container } = renderPanel();

    const panel = container.querySelector(
      '[data-testid="code-review-summary-panel"]',
    ) as HTMLElement | null;

    expect(panel?.textContent).toContain("Code review summary");
    expect(panel?.textContent).toContain("File changes 1");
    expect(panel?.textContent).toContain("Test output 1");
    expect(panel?.textContent).toContain("Snapshots 1");
    expect(panel?.textContent).toContain("Changes need review");
    expect(panel?.textContent).toContain("Review result");
  });

  it("失败输出应在输出入口提示需要处理", async () => {
    const { container, onOpenSection, onSubmitCodeFixPrompt } = renderPanel({
      harnessState: {
        ...createEmptyHarnessState(),
        outputSignals: [
          {
            id: "signal-failed",
            toolCallId: "tool-test",
            toolName: "bash",
            title: "测试失败",
            summary: "vitest failed",
            preview: "1 test failed",
            exitCode: 1,
          },
        ],
        recentFileEvents: [
          {
            id: "event-code",
            toolCallId: "tool-write",
            path: "/tmp/workspace/src/ImageCard.tsx",
            displayName: "ImageCard.tsx",
            kind: "code",
            action: "edit",
            sourceToolName: "edit_file",
            clickable: true,
          },
        ],
        hasSignals: true,
      },
    });

    const outputs = container.querySelector(
      '[data-testid="code-review-summary-outputs"]',
    ) as HTMLElement | null;
    const status = container.querySelector(
      '[data-testid="code-review-summary-status"]',
    ) as HTMLElement | null;
    const outputPreview = container.querySelector(
      '[data-testid="code-review-summary-output-preview"]',
    ) as HTMLElement | null;
    const fixAction = container.querySelector(
      '[data-testid="code-review-summary-fix-action"]',
    ) as HTMLButtonElement | null;
    const focus = container.querySelector(
      '[data-testid="code-review-summary-focus"]',
    ) as HTMLElement | null;
    const focusOutput = container.querySelector(
      '[data-testid="code-review-summary-focus-output"]',
    ) as HTMLButtonElement | null;
    const focusFiles = container.querySelector(
      '[data-testid="code-review-summary-focus-files"]',
    ) as HTMLButtonElement | null;
    const pair = container.querySelector(
      '[data-testid="code-review-summary-pair"]',
    ) as HTMLElement | null;
    const pairOutput = container.querySelector(
      '[data-testid="code-review-summary-pair-output"]',
    ) as HTMLButtonElement | null;
    const pairFile = container.querySelector(
      '[data-testid="code-review-summary-pair-file"]',
    ) as HTMLButtonElement | null;

    expect(outputs?.textContent).toContain("测试输出 1");
    expect(outputs?.textContent).toContain("测试失败");
    expect(outputs?.dataset.tone).toBe("danger");
    expect(status?.textContent).toContain("先处理失败输出");
    expect(focus?.dataset.tone).toBe("danger");
    expect(focus?.textContent).toContain("当前审阅焦点");
    expect(focus?.textContent).toContain("先看失败输出");
    expect(focus?.textContent).toContain("测试失败");
    expect(focus?.textContent).toContain("ImageCard.tsx");
    expect(pair?.textContent).toContain("失败输出");
    expect(pair?.textContent).toContain("相关文件");
    expect(pair?.textContent).toContain("ImageCard.tsx");
    expect(pair?.textContent).toContain("共 1 个文件变更");
    expect(pair?.textContent).not.toContain("失败输出提到了这个文件");
    expect(container.textContent).toContain("1 条输出需要处理");
    expect(outputPreview?.textContent).toContain("失败片段");
    expect(outputPreview?.textContent).toContain("1 test failed");
    expect(fixAction?.textContent).toContain("继续修复");

    act(() => {
      focusOutput?.click();
    });
    act(() => {
      focusFiles?.click();
    });
    act(() => {
      pairOutput?.click();
    });
    act(() => {
      pairFile?.click();
    });

    await act(async () => {
      fixAction?.click();
    });

    expect(onOpenSection).toHaveBeenCalledWith("outputs");
    expect(onOpenSection).toHaveBeenCalledWith("file_review");
    expect(onOpenSection).toHaveBeenCalledTimes(4);
    expect(onSubmitCodeFixPrompt).toHaveBeenCalledTimes(1);
    expect(onSubmitCodeFixPrompt.mock.calls[0]?.[0]).toContain("测试失败");
    expect(onSubmitCodeFixPrompt.mock.calls[0]?.[0]).toContain(
      "1 test failed",
    );
  });

  it("英文界面继续修复应生成英文 prompt", async () => {
    await changeLimeLocale("en-US");

    const { container, onSubmitCodeFixPrompt } = renderPanel({
      harnessState: {
        ...createEmptyHarnessState(),
        outputSignals: [
          {
            id: "signal-failed",
            toolCallId: "tool-test",
            toolName: "bash",
            title: "Test failed",
            summary: "vitest failed",
            preview: "1 test failed",
            exitCode: 1,
          },
        ],
        recentFileEvents: [
          {
            id: "event-code",
            toolCallId: "tool-write",
            path: "/tmp/workspace/src/ImageCard.tsx",
            displayName: "ImageCard.tsx",
            kind: "code",
            action: "edit",
            sourceToolName: "edit_file",
            clickable: true,
          },
        ],
        hasSignals: true,
      },
    });

    const fixAction = container.querySelector(
      '[data-testid="code-review-summary-fix-action"]',
    ) as HTMLButtonElement | null;
    const focus = container.querySelector(
      '[data-testid="code-review-summary-focus"]',
    ) as HTMLElement | null;

    expect(fixAction?.textContent).toContain("Continue fixing");
    expect(focus?.textContent).toContain("Current review focus");
    expect(focus?.textContent).toContain("Check failed output");

    await act(async () => {
      fixAction?.click();
    });

    const submittedPrompt = onSubmitCodeFixPrompt.mock.calls[0]?.[0] || "";
    expect(submittedPrompt).toContain(
      "Continue fixing the failed output from this coding run.",
    );
    expect(submittedPrompt).toContain("- Failed tool: bash");
    expect(submittedPrompt).toContain("Failure preview:");
    expect(submittedPrompt).not.toContain("请继续修复");
  });

  it("没有失败输出时不展示继续修复入口", () => {
    const { container } = renderPanel();

    expect(
      container.querySelector('[data-testid="code-review-summary-fix-action"]'),
    ).toBeNull();
  });

  it("存在失败输出时主按钮应优先打开输出区块", () => {
    const { container, onOpenSection } = renderPanel({
      harnessState: {
        ...createEmptyHarnessState(),
        outputSignals: [
          {
            id: "signal-failed",
            toolCallId: "tool-test",
            toolName: "bash",
            title: "回归测试失败",
            summary: "vitest failed",
            preview: "1 test failed",
            exitCode: 1,
          },
        ],
        recentFileEvents: [
          {
            id: "event-code",
            toolCallId: "tool-write",
            path: "/tmp/workspace/src/ImageCard.tsx",
            displayName: "ImageCard.tsx",
            kind: "code",
            action: "edit",
            sourceToolName: "edit_file",
            clickable: true,
          },
        ],
        hasSignals: true,
      },
      fileCheckpointSummary: null,
    });

    const primaryAction = container.querySelector(
      '[data-testid="code-review-summary-primary-action"]',
    ) as HTMLButtonElement | null;

    expect(primaryAction?.textContent).toContain("查看失败输出");

    act(() => {
      primaryAction?.click();
    });

    expect(onOpenSection).toHaveBeenCalledWith("outputs");
  });

  it("多条输出混合时应优先展示失败输出详情", () => {
    const { container } = renderPanel({
      harnessState: {
        ...createEmptyHarnessState(),
        outputSignals: [
          {
            id: "signal-pass",
            toolCallId: "tool-pass",
            toolName: "bash",
            title: "类型检查通过",
            summary: "typecheck passed",
            preview: "tsc --noEmit",
            exitCode: 0,
          },
          {
            id: "signal-failed",
            toolCallId: "tool-failed",
            toolName: "bash",
            title: "回归测试失败",
            summary: "vitest failed",
            preview: "ImageCard.test.tsx failed",
            exitCode: 1,
          },
        ],
        recentFileEvents: [
          {
            id: "event-code",
            toolCallId: "tool-write",
            path: "/tmp/workspace/src/ImageCard.tsx",
            displayName: "ImageCard.tsx",
            kind: "code",
            action: "edit",
            sourceToolName: "edit_file",
            clickable: true,
          },
        ],
        hasSignals: true,
      },
      fileCheckpointSummary: null,
    });

    const outputs = container.querySelector(
      '[data-testid="code-review-summary-outputs"]',
    ) as HTMLElement | null;

    expect(outputs?.textContent).toContain("回归测试失败");
    expect(outputs?.dataset.tone).toBe("danger");
    expect(outputs?.textContent).not.toContain("类型检查通过");
  });

  it("失败输出提到的文件应优先进入审阅焦点和修复 prompt", async () => {
    const { container, onSubmitCodeFixPrompt } = renderPanel({
      harnessState: {
        ...createEmptyHarnessState(),
        outputSignals: [
          {
            id: "signal-failed",
            toolCallId: "tool-failed",
            toolName: "bash",
            title: "回归测试失败",
            summary: "ImageCard.test.tsx assertion failed",
            preview: "src/ImageCard.test.tsx: expected image card to render",
            exitCode: 1,
          },
        ],
        recentFileEvents: [
          {
            id: "event-code-1",
            toolCallId: "tool-write-1",
            path: "/tmp/workspace/src/App.tsx",
            displayName: "App.tsx",
            kind: "code",
            action: "edit",
            sourceToolName: "edit_file",
            clickable: true,
          },
          {
            id: "event-code-2",
            toolCallId: "tool-write-2",
            path: "/tmp/workspace/src/ImageCard.test.tsx",
            displayName: "ImageCard.test.tsx",
            kind: "code",
            action: "edit",
            sourceToolName: "edit_file",
            clickable: true,
          },
          {
            id: "event-code-3",
            toolCallId: "tool-write-3",
            path: "/tmp/workspace/src/ImageCard.tsx",
            displayName: "ImageCard.tsx",
            kind: "code",
            action: "edit",
            sourceToolName: "edit_file",
            clickable: true,
          },
        ],
        hasSignals: true,
      },
      fileCheckpointSummary: null,
    });

    const files = container.querySelector(
      '[data-testid="code-review-summary-files"]',
    ) as HTMLElement | null;
    const focus = container.querySelector(
      '[data-testid="code-review-summary-focus"]',
    ) as HTMLElement | null;
    const pairFile = container.querySelector(
      '[data-testid="code-review-summary-pair-file"]',
    ) as HTMLElement | null;
    const fixAction = container.querySelector(
      '[data-testid="code-review-summary-fix-action"]',
    ) as HTMLButtonElement | null;

    expect(focus?.textContent).toContain("ImageCard.test.tsx");
    expect(pairFile?.textContent).toContain("ImageCard.test.tsx");
    expect(pairFile?.textContent).toContain("失败输出提到了这个文件");
    expect(files?.textContent?.indexOf("ImageCard.test.tsx")).toBeLessThan(
      files?.textContent?.indexOf("App.tsx") ?? Number.POSITIVE_INFINITY,
    );

    await act(async () => {
      fixAction?.click();
    });

    const submittedPrompt = onSubmitCodeFixPrompt.mock.calls[0]?.[0] || "";
    expect(submittedPrompt.indexOf("ImageCard.test.tsx")).toBeLessThan(
      submittedPrompt.indexOf("App.tsx"),
    );
  });

  it("文件变更超过三条时应提示剩余数量", () => {
    const { container } = renderPanel({
      harnessState: {
        ...createEmptyHarnessState(),
        recentFileEvents: [
          {
            id: "event-code-1",
            toolCallId: "tool-write-1",
            path: "/tmp/workspace/src/App.tsx",
            displayName: "App.tsx",
            kind: "code",
            action: "write",
            sourceToolName: "write_file",
            clickable: true,
          },
          {
            id: "event-code-2",
            toolCallId: "tool-write-2",
            path: "/tmp/workspace/src/App.test.tsx",
            displayName: "App.test.tsx",
            kind: "code",
            action: "edit",
            sourceToolName: "edit_file",
            clickable: true,
          },
          {
            id: "event-code-3",
            toolCallId: "tool-write-3",
            path: "/tmp/workspace/src/styles.css",
            displayName: "styles.css",
            kind: "code",
            action: "edit",
            sourceToolName: "edit_file",
            clickable: true,
          },
          {
            id: "event-code-4",
            toolCallId: "tool-write-4",
            path: "/tmp/workspace/src/routes.ts",
            displayName: "routes.ts",
            kind: "code",
            action: "write",
            sourceToolName: "write_file",
            clickable: true,
          },
        ],
        hasSignals: true,
      },
      fileCheckpointSummary: null,
    });

    const files = container.querySelector(
      '[data-testid="code-review-summary-files"]',
    ) as HTMLElement | null;

    expect(files?.textContent).toContain("App.tsx");
    expect(files?.textContent).toContain("App.test.tsx");
    expect(files?.textContent).toContain("styles.css");
    expect(files?.textContent).not.toContain("routes.ts");
    expect(files?.textContent).toContain("另有 1 个文件");
  });

  it("应展示详细文件审阅区的应用和回退状态", () => {
    const { container } = renderPanel({
      fileChangeReviewSummary: {
        total: 3,
        pending: 0,
        applied: 2,
        rejected: 1,
      },
    });

    const panel = container.querySelector(
      '[data-testid="code-review-summary-panel"]',
    ) as HTMLElement | null;
    const status = container.querySelector(
      '[data-testid="code-review-summary-status"]',
    ) as HTMLElement | null;
    const files = container.querySelector(
      '[data-testid="code-review-summary-files"]',
    ) as HTMLElement | null;
    const reviewState = container.querySelector(
      '[data-testid="code-review-summary-review-state"]',
    ) as HTMLElement | null;

    expect(panel?.dataset.status).toBe("snapshots");
    expect(status?.textContent).toContain("已标记回退");
    expect(files?.dataset.confirmed).toBe("3");
    expect(files?.textContent).toContain("待处理 0 · 已应用 2 · 回退 1");
    expect(reviewState?.textContent).toContain(
      "待处理 0 个，已应用 2 个，已标记回退 1 个。",
    );
  });

  it("全部文件已应用时应展示变更已确认", () => {
    const { container } = renderPanel({
      fileChangeReviewSummary: {
        total: 1,
        pending: 0,
        applied: 1,
        rejected: 0,
      },
    });

    const panel = container.querySelector(
      '[data-testid="code-review-summary-panel"]',
    ) as HTMLElement | null;
    const status = container.querySelector(
      '[data-testid="code-review-summary-status"]',
    ) as HTMLElement | null;

    expect(panel?.dataset.status).toBe("outputs");
    expect(status?.textContent).toContain("变更已确认");
  });

  it("只有输出时主按钮应打开输出区块", () => {
    const { container, onOpenSection, onOpenFileCheckpoints } = renderPanel({
      harnessState: {
        ...createEmptyHarnessState(),
        outputSignals: [
          {
            id: "signal-only",
            toolCallId: "tool-test",
            toolName: "bash",
            title: "类型检查",
            summary: "typecheck passed",
            preview: "tsc --noEmit",
            exitCode: 0,
          },
        ],
        hasSignals: true,
      },
      fileCheckpointSummary: null,
    });

    const primaryAction = container.querySelector(
      '[data-testid="code-review-summary-primary-action"]',
    ) as HTMLButtonElement | null;
    const panel = container.querySelector(
      '[data-testid="code-review-summary-panel"]',
    ) as HTMLElement | null;
    const outputs = container.querySelector(
      '[data-testid="code-review-summary-outputs"]',
    ) as HTMLElement | null;

    act(() => {
      primaryAction?.click();
    });

    expect(primaryAction?.textContent).toContain("查看输出");
    expect(panel?.dataset.status).toBe("outputs");
    expect(outputs?.dataset.tone).toBe("success");
    expect(onOpenSection).toHaveBeenCalledWith("outputs");
    expect(onOpenFileCheckpoints).not.toHaveBeenCalled();
  });

  it("没有快照时只展示空态入口", () => {
    const { container } = renderPanel({
      harnessState: {
        ...createEmptyHarnessState(),
        outputSignals: [
          {
            id: "signal-pass",
            toolCallId: "tool-test",
            toolName: "bash",
            title: "测试通过",
            summary: "vitest passed",
            preview: "1 test passed",
            exitCode: 0,
          },
        ],
        hasSignals: true,
      },
      fileCheckpointSummary: null,
    });

    const checkpoints = container.querySelector(
      '[data-testid="code-review-summary-checkpoints"]',
    ) as HTMLButtonElement | null;

    expect(checkpoints?.disabled).toBe(true);
    expect(checkpoints?.textContent).toContain("暂无可回滚快照");
  });

  it("只有快照时主按钮应直接打开快照入口", () => {
    const { container, onOpenSection, onOpenFileCheckpoints } = renderPanel({
      harnessState: createEmptyHarnessState(),
      fileCheckpointSummary: {
        count: 1,
        latest_checkpoint: {
          checkpoint_id: "checkpoint-only",
          turn_id: "turn-only",
          path: "src/App.tsx",
          source: "tool_result",
          updated_at: "2026-05-27T02:00:00.000Z",
          validation_issue_count: 0,
        },
      },
    });

    const primaryAction = container.querySelector(
      '[data-testid="code-review-summary-primary-action"]',
    ) as HTMLButtonElement | null;
    const files = container.querySelector(
      '[data-testid="code-review-summary-files"]',
    ) as HTMLButtonElement | null;
    const outputs = container.querySelector(
      '[data-testid="code-review-summary-outputs"]',
    ) as HTMLButtonElement | null;
    const panel = container.querySelector(
      '[data-testid="code-review-summary-panel"]',
    ) as HTMLElement | null;

    act(() => {
      primaryAction?.click();
    });

    expect(primaryAction?.textContent).toContain("查看快照");
    expect(panel?.dataset.status).toBe("snapshots");
    expect(files?.disabled).toBe(true);
    expect(outputs?.disabled).toBe(true);
    expect(onOpenSection).not.toHaveBeenCalled();
    expect(onOpenFileCheckpoints).toHaveBeenCalledTimes(1);
  });

  it("没有审阅事实时不渲染", () => {
    const { container } = renderPanel({
      harnessState: createEmptyHarnessState(),
      fileCheckpointSummary: null,
    });

    expect(
      container.querySelector('[data-testid="code-review-summary-panel"]'),
    ).toBeNull();
  });

  it("点击开始审查后应立即进入进行中状态", async () => {
    const onStartReview = vi.fn(() => new Promise(() => {}));
    const { container } = renderPanel({
      reviewThreadId: "thread-1",
      onStartReview,
    });

    const button = container.querySelector(
      '[data-testid="code-review-summary-start-review"]',
    ) as HTMLButtonElement | null;
    expect(button?.disabled).toBe(false);

    act(() => {
      button?.click();
    });

    expect(onStartReview).toHaveBeenCalledTimes(1);
    expect(
      container.querySelector('[data-testid="code-review-summary-review-status"]')
        ?.textContent,
    ).toContain("正在启动");
  });

  it("review/start admission 成功后不应永久停留在正在启动", async () => {
    const onStartReview = vi.fn().mockResolvedValue(undefined);
    const { container } = renderPanel({
      reviewThreadId: "thread-1",
      onStartReview,
    });
    const button = container.querySelector(
      '[data-testid="code-review-summary-start-review"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      button?.click();
      await Promise.resolve();
    });

    expect(onStartReview).toHaveBeenCalledTimes(1);
    expect(button?.disabled).toBe(false);
    expect(
      container.querySelector('[data-testid="code-review-summary-review-status"]'),
    ).toBeNull();
  });

  it("应从 canonical exited boundary 展示完成状态和审查结果", () => {
    const { container } = renderPanel({
      reviewThreadId: "thread-1",
      onStartReview: vi.fn().mockResolvedValue(undefined),
      threadItems: [
        {
          id: "review-boundary-1",
          thread_id: "thread-1",
          turn_id: "turn-review",
          sequence: 4,
          status: "completed",
          started_at: "2026-06-06T00:00:00.000Z",
          updated_at: "2026-06-06T00:00:01.000Z",
          type: "review_boundary",
          boundary: "exited",
          review: "发现 2 个建议",
        },
      ],
    });

    expect(
      container.querySelector('[data-testid="code-review-summary-review-status"]')
        ?.textContent,
    ).toContain("审查已完成");
    expect(
      container.querySelector('[data-testid="code-review-summary-review-result"]')
        ?.textContent,
    ).toContain("发现 2 个建议");
  });

  it("缺少 thread 或存在活动回合时应 fail closed", () => {
    const missingThread = renderPanel({
      onStartReview: vi.fn().mockResolvedValue(undefined),
    });
    const missingButton = missingThread.container.querySelector(
      '[data-testid="code-review-summary-start-review"]',
    ) as HTMLButtonElement | null;
    expect(missingButton?.disabled).toBe(true);
    expect(
      missingThread.container.querySelector(
        '[data-testid="code-review-summary-review-control"]',
      )?.textContent,
    ).toContain("请先开始一次对话");

    const activeTurn = renderPanel({
      reviewThreadId: "thread-1",
      currentTurnId: "turn-active",
      canInterrupt: true,
      onStartReview: vi.fn().mockResolvedValue(undefined),
    });
    const activeButton = activeTurn.container.querySelector(
      '[data-testid="code-review-summary-start-review"]',
    ) as HTMLButtonElement | null;
    expect(activeButton?.disabled).toBe(true);
    expect(
      activeTurn.container.querySelector(
        '[data-testid="code-review-summary-review-control"]',
      )?.textContent,
    ).toContain("当前任务仍在运行");
  });

  it("启动失败应展示通用错误并允许重试", async () => {
    const onStartReview = vi
      .fn()
      .mockRejectedValueOnce(new Error("internal review error"))
      .mockResolvedValueOnce(undefined);
    const { container } = renderPanel({
      reviewThreadId: "thread-1",
      onStartReview,
    });
    const button = container.querySelector(
      '[data-testid="code-review-summary-start-review"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      button?.click();
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="code-review-summary-review-error"]')
        ?.textContent,
    ).toContain("审查未能启动");
    expect(button?.disabled).toBe(false);

    await act(async () => {
      button?.click();
      await Promise.resolve();
    });
    expect(onStartReview).toHaveBeenCalledTimes(2);
  });
});
