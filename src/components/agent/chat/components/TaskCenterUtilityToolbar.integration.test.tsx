import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import { TaskCenterUtilityToolbar } from "./TaskCenterUtilityToolbar";
import { TaskCenterShellPanel } from "./TaskCenterShellPanel";

const {
  mockOpenProjectPathWithTool,
  mockOpenExternalUrlWithSystemBrowser,
  mockCheckoutProjectGitBranch,
  mockCreateProjectGitBranch,
  mockCreateProjectGitWorktree,
  mockReadProjectGitDiff,
  mockReadProjectGitStatus,
  mockTerminateCommandExec,
  mockSubscribeCommandExecOutput,
  mockResizeCommandExec,
  mockExecCommand,
  mockWriteCommandExec,
  mockFitAddonFit,
  mockXtermDisposeInput,
  mockXtermOnDataHandlers,
  mockXtermLoadAddon,
  mockXtermTerminalOptions,
  mockXtermWrite,
  mockXtermWriteln,
} = vi.hoisted(() => ({
  mockOpenProjectPathWithTool: vi.fn(),
  mockOpenExternalUrlWithSystemBrowser: vi.fn(),
  mockCheckoutProjectGitBranch: vi.fn(),
  mockCreateProjectGitBranch: vi.fn(),
  mockCreateProjectGitWorktree: vi.fn(),
  mockReadProjectGitDiff: vi.fn(),
  mockReadProjectGitStatus: vi.fn(),
  mockTerminateCommandExec: vi.fn(),
  mockSubscribeCommandExecOutput: vi.fn(),
  mockResizeCommandExec: vi.fn(),
  mockExecCommand: vi.fn(),
  mockWriteCommandExec: vi.fn(),
  mockFitAddonFit: vi.fn(),
  mockXtermDisposeInput: vi.fn(),
  mockXtermOnDataHandlers: [] as Array<(data: string) => void>,
  mockXtermLoadAddon: vi.fn(),
  mockXtermTerminalOptions: [] as Array<Record<string, unknown>>,
  mockXtermWrite: vi.fn(),
  mockXtermWriteln: vi.fn(),
}));

vi.mock("@/lib/api/fileSystem", () => ({
  openProjectPathWithTool: mockOpenProjectPathWithTool,
}));

vi.mock("@/lib/api/externalUrl", () => ({
  openExternalUrlWithSystemBrowser: mockOpenExternalUrlWithSystemBrowser,
}));

vi.mock("@/lib/api/projectGit", () => ({
  checkoutProjectGitBranch: mockCheckoutProjectGitBranch,
  createProjectGitBranch: mockCreateProjectGitBranch,
  createProjectGitWorktree: mockCreateProjectGitWorktree,
  readProjectGitDiff: mockReadProjectGitDiff,
  readProjectGitStatus: mockReadProjectGitStatus,
}));

vi.mock("@/lib/api/commandExec", () => ({
  terminateCommandExec: mockTerminateCommandExec,
  subscribeCommandExecOutput: mockSubscribeCommandExecOutput,
  resizeCommandExec: mockResizeCommandExec,
  execCommand: mockExecCommand,
  writeCommandExec: mockWriteCommandExec,
}));

vi.mock("@xterm/xterm/css/xterm.css", () => ({}));

vi.mock("@xterm/xterm", () => ({
  Terminal: vi.fn().mockImplementation((options: Record<string, unknown>) => {
    mockXtermTerminalOptions.push(options);
    return {
      cols: 120,
      rows: 14,
      dispose: vi.fn(),
      focus: vi.fn(),
      loadAddon: mockXtermLoadAddon,
      onData: vi.fn((handler: (data: string) => void) => {
        mockXtermOnDataHandlers.push(handler);
        return { dispose: mockXtermDisposeInput };
      }),
      open: vi.fn(),
      write: mockXtermWrite,
      writeln: mockXtermWriteln,
    };
  }),
}));

vi.mock("@xterm/addon-fit", () => ({
  FitAddon: vi.fn().mockImplementation(() => ({
    fit: mockFitAddonFit,
  })),
}));

vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
  },
}));

vi.mock("@/i18n/createI18n", () => ({
  changeLimeLocale: vi.fn(async () => "zh-CN"),
}));

vi.mock("react-i18next", () => {
  const t = (key: string, options?: Record<string, unknown>) => {
    const template =
      typeof options?.defaultValue === "string" ? options.defaultValue : key;

    return template.replace(/{{\s*([^}]+?)\s*}}/g, (_, name: string) =>
      String(options?.[name.trim()] ?? ""),
    );
  };
  return {
    initReactI18next: {
      type: "3rdParty",
      init: () => undefined,
    },
    useTranslation: () => ({ t }),
  };
});

vi.mock("@/components/ui/button", () => ({
  Button: React.forwardRef<
    HTMLButtonElement,
    React.ButtonHTMLAttributes<HTMLButtonElement> & {
      variant?: string;
      size?: string;
    }
  >(
    (
      {
        children,
        onClick,
        disabled,
        type,
        variant: _variant,
        size: _size,
        ...rest
      },
      ref,
    ) => (
      <button
        ref={ref}
        type={type ?? "button"}
        onClick={onClick}
        disabled={disabled}
        {...rest}
      >
        {children}
      </button>
    ),
  ),
}));

const PopoverTestContext = React.createContext<{
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
} | null>(null);

vi.mock("@/components/ui/popover", () => ({
  Popover: ({
    children,
    open,
    onOpenChange,
  }: {
    children: React.ReactNode;
    open?: boolean;
    onOpenChange?: (open: boolean) => void;
  }) => (
    <PopoverTestContext.Provider value={{ open, onOpenChange }}>
      {children}
    </PopoverTestContext.Provider>
  ),
  PopoverContent: ({
    children,
    align: _align,
    sideOffset: _sideOffset,
    ...props
  }: React.HTMLAttributes<HTMLDivElement> & {
    align?: string;
    sideOffset?: number;
  }) => <div {...props}>{children}</div>,
  PopoverTrigger: ({ children }: { children: React.ReactNode }) => {
    const context = React.useContext(PopoverTestContext);
    if (!React.isValidElement(children)) {
      return <>{children}</>;
    }
    const child = children as React.ReactElement<{
      onClick?: React.MouseEventHandler<HTMLElement>;
    }>;
    return React.cloneElement(child, {
      onClick: (event: React.MouseEvent<HTMLElement>) => {
        child.props.onClick?.(event);
        context?.onOpenChange?.(!context.open);
      },
    });
  },
}));

interface MountedHarness {
  container: HTMLDivElement;
  root: Root;
}

const mountedRoots: MountedHarness[] = [];

beforeEach(async () => {
  vi.useRealTimers();
  await changeLimeLocale("zh-CN");
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  if (!globalThis.PointerEvent) {
    Object.defineProperty(globalThis, "PointerEvent", {
      configurable: true,
      value: MouseEvent,
    });
  }
  HTMLElement.prototype.setPointerCapture ??= vi.fn();
  HTMLElement.prototype.releasePointerCapture ??= vi.fn();
  mockOpenProjectPathWithTool.mockResolvedValue(undefined);
  mockOpenExternalUrlWithSystemBrowser.mockResolvedValue(undefined);
  mockCheckoutProjectGitBranch.mockResolvedValue({
    rootPath: "/tmp/project",
    hasGitRepository: true,
    currentBranch: "main",
    branches: ["main", "feature/task-center"],
    uncommittedFileCount: 3,
  });
  mockCreateProjectGitBranch.mockResolvedValue({
    rootPath: "/tmp/project",
    hasGitRepository: true,
    currentBranch: "new-branch",
    branches: ["new-branch", "feature/task-center"],
    uncommittedFileCount: 3,
  });
  mockReadProjectGitDiff.mockResolvedValue({
    rootPath: "/tmp/project",
    hasGitRepository: true,
    patch: "+added\n-removed\n",
    uncommittedFileCount: 3,
    currentRef: "feature/task-center",
    comparisonBaseRef: null,
  });
  mockReadProjectGitStatus.mockResolvedValue({
    rootPath: "/tmp/project",
    hasGitRepository: true,
    currentBranch: "feature/task-center",
    branches: [
      "feature/task-center",
      "main",
      "dev-electron",
      "dark-sol-drifts-02h23",
      "hotfix/v1.16.0-republish",
      "release/v1.12.1",
    ],
    uncommittedFileCount: 3,
  });
  mockTerminateCommandExec.mockResolvedValue({});
  mockSubscribeCommandExecOutput.mockReturnValue(vi.fn());
  mockResizeCommandExec.mockResolvedValue({});
  mockExecCommand.mockReturnValue(new Promise(() => undefined));
  mockWriteCommandExec.mockResolvedValue({});
  mockXtermOnDataHandlers.length = 0;
  mockXtermTerminalOptions.length = 0;
});

afterEach(async () => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) break;
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.clearAllMocks();
  await changeLimeLocale("en-US");
});

function mount(node: React.ReactNode) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(node);
  });

  mountedRoots.push({ container, root });
  return container;
}

function renderToolbar(
  props?: Partial<React.ComponentProps<typeof TaskCenterUtilityToolbar>>,
) {
  return mount(
    <TaskCenterUtilityToolbar
      projectRootPath="/tmp/project"
      onProjectChange={vi.fn()}
      showCanvasToggle
      isCanvasOpen={false}
      onToggleCanvas={vi.fn()}
      showHarnessToggle
      harnessPanelVisible={false}
      onToggleHarnessPanel={vi.fn()}
      harnessPendingCount={0}
      harnessAttentionLevel="idle"
      harnessToggleLabel="Harness"
      shellPanelOpen={false}
      onToggleShellPanel={vi.fn()}
      {...props}
    />,
  );
}

describe("TaskCenterUtilityToolbar", () => {
  it("顶部工具栏应允许工具组自适应换行，避免窄宽度下挤压内容", () => {
    const container = renderToolbar({
      isCanvasOpen: true,
      harnessPendingCount: 3,
    });

    const toolbar = container.querySelector(
      '[data-testid="task-center-utility-toolbar"]',
    );
    const panelGroup = container.querySelector(
      '[data-testid="task-center-tool-group-panels"]',
    );
    const workbenchToggle = container.querySelector(
      '[data-testid="task-center-workbench-toggle"]',
    );

    expect(toolbar?.className).toContain("flex-wrap");
    expect(toolbar?.className).toContain("gap-y-1");
    expect(toolbar?.className).not.toContain("flex-nowrap");
    expect(toolbar?.className).not.toContain("whitespace-nowrap");
    expect(panelGroup?.className).toContain("flex-wrap");
    expect(panelGroup?.className).not.toContain("overflow-hidden");
    expect(workbenchToggle?.className).toContain("shrink-0");
    expect(workbenchToggle?.textContent?.trim()).toBe("");
  });

  it("专家信息按钮应在顶部工具列中以图标态展开和收起右栏", () => {
    const onToggleExpertInfoPanel = vi.fn();
    const container = renderToolbar({
      showExpertInfoToggle: true,
      expertInfoPanelVisible: false,
      onToggleExpertInfoPanel,
    });

    const toggle = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-expert-info-toggle"]',
    );
    expect(toggle).not.toBeNull();
    expect(toggle?.textContent?.trim()).toBe("");
    expect(toggle?.getAttribute("aria-expanded")).toBe("false");
    expect(toggle?.getAttribute("aria-label")).toBe("打开专家信息");
    expect(toggle?.className).not.toContain("lime-chrome-tab-active-surface");

    act(() => {
      toggle?.click();
    });

    expect(onToggleExpertInfoPanel).toHaveBeenCalledTimes(1);

    const visibleContainer = renderToolbar({
      showExpertInfoToggle: true,
      expertInfoPanelVisible: true,
      onToggleExpertInfoPanel,
    });
    const visibleToggle = visibleContainer.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-expert-info-toggle"]',
    );

    expect(visibleToggle?.getAttribute("aria-expanded")).toBe("true");
    expect(visibleToggle?.getAttribute("aria-label")).toBe("关闭专家信息");
    expect(visibleToggle?.className).toContain(
      "lime-chrome-tab-active-surface",
    );
  });

  it("右侧 surface projection 应优先驱动专家和工作台按钮状态", () => {
    const container = renderToolbar({
      showExpertInfoToggle: true,
      expertInfoPanelVisible: false,
      isCanvasOpen: false,
      rightSurfaceLaunchers: [
        {
          kind: "workbench",
          active: false,
          disabled: true,
          pendingCount: 2,
          collapseTarget: "topToolbar",
        },
        {
          kind: "expertInfo",
          active: true,
          disabled: false,
          pendingCount: 3,
          collapseTarget: "topToolbar",
        },
      ],
    });

    const expertToggle = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-expert-info-toggle"]',
    );
    const workbenchToggle = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-workbench-toggle"]',
    );

    expect(expertToggle?.getAttribute("aria-expanded")).toBe("true");
    expect(expertToggle?.getAttribute("aria-label")).toBe("关闭专家信息");
    expect(expertToggle?.className).toContain("lime-chrome-tab-active-surface");
    expect(expertToggle?.textContent).toContain("3");

    expect(workbenchToggle?.disabled).toBe(true);
    expect(workbenchToggle?.textContent).toContain("2");
  });

  it("右侧 surface projection 应能驱动 Harness pending badge", () => {
    const container = renderToolbar({
      harnessPendingCount: 0,
      rightSurfaceLaunchers: [
        {
          kind: "harness",
          active: false,
          disabled: false,
          pendingCount: 2,
          collapseTarget: "topToolbar",
        },
      ],
    });

    const harnessToggle = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-harness-toggle"]',
    );

    expect(harnessToggle?.textContent).toContain("2");
  });

  it("右侧 surface projection 应能驱动 Harness 展开态和禁用态", () => {
    const activeContainer = renderToolbar({
      harnessPanelVisible: false,
      rightSurfaceLaunchers: [
        {
          kind: "harness",
          active: true,
          disabled: false,
          pendingCount: 0,
          collapseTarget: "topToolbar",
        },
      ],
    });

    const activeToggle = activeContainer.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-harness-toggle"]',
    );

    expect(activeToggle?.getAttribute("aria-expanded")).toBe("true");
    expect(activeToggle?.getAttribute("aria-label")).toBe("关闭Harness");
    expect(activeToggle?.className).toContain("lime-chrome-tab-active-surface");

    const disabledContainer = renderToolbar({
      harnessPanelVisible: false,
      rightSurfaceLaunchers: [
        {
          kind: "harness",
          active: false,
          disabled: true,
          pendingCount: 4,
          collapseTarget: "topToolbar",
        },
      ],
    });

    const disabledToggle = disabledContainer.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-harness-toggle"]',
    );

    expect(disabledToggle?.disabled).toBe(true);
    expect(disabledToggle?.getAttribute("aria-expanded")).toBe("false");
    expect(disabledToggle?.textContent).toContain("4");
  });

  it("右侧 surface projection 应能驱动 Trace 入口展开态、badge 和点击回调", () => {
    const onToggleTracePanel = vi.fn();
    const container = renderToolbar({
      onToggleTracePanel,
      rightSurfaceLaunchers: [
        {
          kind: "trace",
          active: true,
          disabled: false,
          pendingCount: 2,
          collapseTarget: "topToolbar",
        },
      ],
    });

    const toggle = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-trace-toggle"]',
    );

    expect(toggle).not.toBeNull();
    expect(toggle?.getAttribute("aria-expanded")).toBe("true");
    expect(toggle?.getAttribute("aria-label")).toBe("关闭 Trace");
    expect(toggle?.className).toContain("lime-chrome-tab-active-surface");
    expect(toggle?.textContent).toContain("2");

    act(() => {
      toggle?.click();
    });

    expect(onToggleTracePanel).toHaveBeenCalledTimes(1);
  });

  it("旧字段不应再控制 Trace 入口", () => {
    const onToggleTracePanel = vi.fn();
    const container = renderToolbar({
      showTraceToggle: true,
      tracePanelVisible: false,
      onToggleTracePanel,
    } as Partial<React.ComponentProps<typeof TaskCenterUtilityToolbar>> & {
      showTraceToggle: boolean;
      tracePanelVisible: boolean;
      onToggleTracePanel: () => void;
    });

    const toggle = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-trace-toggle"]',
    );

    expect(toggle).toBeNull();
    expect(onToggleTracePanel).not.toHaveBeenCalled();
  });

  it("右侧 surface projection 应能驱动文件入口展开态、badge 和点击回调", () => {
    const onToggleFilesPanel = vi.fn();
    const container = renderToolbar({
      onToggleFilesPanel,
      rightSurfaceLaunchers: [
        {
          kind: "files",
          active: true,
          disabled: false,
          pendingCount: 1,
          collapseTarget: "topToolbar",
        },
      ],
    });

    const filesToggle = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-files-toggle"]',
    );

    expect(filesToggle).not.toBeNull();
    expect(filesToggle?.getAttribute("aria-expanded")).toBe("true");
    expect(filesToggle?.getAttribute("aria-label")).toBe("打开文件");
    expect(filesToggle?.className).toContain("lime-chrome-tab-active-surface");
    expect(filesToggle?.textContent).toContain("1");

    act(() => {
      filesToggle?.click();
    });

    expect(onToggleFilesPanel).toHaveBeenCalledTimes(1);
  });

  it("右侧 surface projection 应能驱动浏览器入口展开态、badge 和点击回调", () => {
    const onToggleBrowserPanel = vi.fn();
    const container = renderToolbar({
      onToggleBrowserPanel,
      rightSurfaceLaunchers: [
        {
          kind: "browser",
          active: true,
          disabled: false,
          pendingCount: 1,
          collapseTarget: "topToolbar",
        },
      ],
    });

    const browserToggle = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-browser-toggle"]',
    );

    expect(browserToggle).not.toBeNull();
    expect(browserToggle?.getAttribute("aria-expanded")).toBe("true");
    expect(browserToggle?.getAttribute("aria-label")).toBe("关闭浏览器");
    expect(browserToggle?.getAttribute("title")).toBe("浏览器");
    expect(browserToggle?.className).toContain(
      "lime-chrome-tab-active-surface",
    );
    expect(browserToggle?.textContent).toContain("1");

    act(() => {
      browserToggle?.click();
    });

    expect(onToggleBrowserPanel).toHaveBeenCalledTimes(1);
  });

  it("打开位置应展示处理位置菜单，并通过文件壳网关打开本地目录", async () => {
    const container = renderToolbar();
    const trigger = container.querySelector(
      '[data-testid="task-center-app-switcher-trigger"]',
    ) as HTMLButtonElement | null;

    expect(trigger?.textContent).toContain("打开位置");

    await act(async () => {
      trigger?.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    const popover = document.body.querySelector(
      '[data-testid="task-center-app-switcher-popover"]',
    );
    expect(popover?.textContent).toContain("继续使用");
    expect(popover?.textContent).toContain("在本地处理");
    expect(popover?.textContent).toContain("关联 Codex web");
    expect(popover?.textContent).toContain("发送至云端");
    expect(popover?.textContent).toContain("工作树");
    expect(popover?.className).toContain("w-[min(216px,calc(100vw-1rem))]");
    const codexWebButton = popover?.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-location-codex-web"]',
    );
    expect(codexWebButton?.disabled).toBe(false);
    expect(
      popover?.querySelector<HTMLButtonElement>(
        '[data-testid="task-center-location-cloud"]',
      )?.disabled,
    ).toBe(true);
    expect(
      popover?.querySelector<HTMLButtonElement>(
        '[data-testid="task-center-location-worktree"]',
      )?.disabled,
    ).toBe(false);
    const localButton = popover?.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-location-local"]',
    );

    await act(async () => {
      codexWebButton?.click();
      await Promise.resolve();
    });
    expect(mockOpenExternalUrlWithSystemBrowser).toHaveBeenCalledWith(
      "https://chatgpt.com/codex",
    );

    await act(async () => {
      localButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(mockOpenProjectPathWithTool).toHaveBeenCalledWith(
      "/tmp/project",
      "finder",
    );
  });

  it("环境信息应读取真实 Git 状态并展示分支与未提交文件数", async () => {
    const container = renderToolbar();
    const trigger = container.querySelector(
      '[data-testid="task-center-environment-trigger"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      trigger?.click();
      await new Promise((resolve) => setTimeout(resolve, 20));
    });

    const popover = document.body.querySelector(
      '[data-testid="task-center-environment-popover"]',
    );
    expect(mockReadProjectGitStatus).toHaveBeenCalledWith("/tmp/project");
    expect(popover?.textContent).toContain("feature/task-center");
    expect(popover?.textContent).toContain("+1 -1");
    expect(popover?.textContent).toContain("提交或推送");
    expect(popover?.textContent).toContain("比较分支");
    expect(popover?.className).toContain("w-[min(300px,calc(100vw-1rem))]");
    expect(
      popover?.querySelector('[data-testid="task-center-environment-add"]'),
    ).not.toBeNull();
    expect(
      popover?.querySelector('[data-testid="task-center-environment-local"]'),
    ).not.toBeNull();
    expect(
      popover?.querySelector('[data-testid="task-center-environment-branch"]'),
    ).not.toBeNull();
    expect(
      popover?.querySelectorAll(
        '[data-testid="task-center-environment-compare"] svg',
      ),
    ).toHaveLength(2);
  });

  it("任务详情存在时环境信息仍应保持紧凑", async () => {
    const onOpenOutput = vi.fn();
    const container = renderToolbar({
      taskRail: {
        workflowSteps: [
          { id: "read", title: "读取任务区结构", status: "completed" },
          { id: "build", title: "接入顶部任务轨道", status: "active" },
          { id: "verify", title: "验证顶部浮层", status: "pending" },
          { id: "ship", title: "整理交付结果", status: "pending" },
        ],
        messages: [
          {
            id: "assistant-task",
            role: "assistant",
            content: "",
            timestamp: new Date("2026-06-16T10:00:00.000Z"),
            toolCalls: [
              {
                id: "tool-rg",
                name: "rg",
                arguments: JSON.stringify({
                  query: "TaskCenterUtilityToolbar",
                }),
                status: "completed",
                result: {
                  success: true,
                  output: "找到顶部工具栏",
                },
                startTime: new Date("2026-06-16T10:00:01.000Z"),
              },
            ],
            artifacts: [
              {
                id: "artifact-plan",
                type: "document",
                title: "agent-workspace-task-rail.md",
                content: "task rail",
                status: "complete",
                createdAt: new Date("2026-06-16T10:00:02.000Z").getTime(),
                updatedAt: new Date("2026-06-16T10:00:02.000Z").getTime(),
                position: { start: 0, end: 9 },
                meta: {
                  filePath: "internal/roadmap/agent-workspace/task-rail.md",
                },
              },
            ],
          },
        ],
        providerType: "cloud",
        model: "reasoner-pro",
        accessMode: "current",
        reasoningEffort: "medium",
        workspaceRootPath: "/tmp/project",
        threadGoal: {
          createdAt: 1,
          objective: "完成任务轨道",
          status: "active",
          threadId: "thread-1",
          timeUsedSeconds: 0,
          tokensUsed: 0,
          updatedAt: 1,
        },
        threadRead: {
          thread_id: "thread-1",
          active_turn_id: "turn-1",
          profile_status: "running",
          context_summary: {
            sources: [
              "AG-UI spec",
              "https://example.com/report",
              "docs/context.md",
            ],
          },
          evidence_summary: {
            evidence_refs: ["evidence/task-rail.json"],
          },
          change_summary: {
            changed_file_count: 2,
            changed_files: ["src/App.tsx", "src/index.ts"],
            patch_count: 2,
            running_patch_count: 1,
          },
        } as any,
        canonicalChildren: [
          {
            name: "实现",
            parentThreadId: "thread-parent",
            sessionId: "subagent-1",
            status: "running",
            threadId: "thread-subagent-1",
            updatedAtMs: 2,
          },
          {
            name: "验证",
            parentThreadId: "thread-parent",
            sessionId: "subagent-2",
            status: "completed",
            threadId: "thread-subagent-2",
            updatedAtMs: 2,
          },
          {
            name: "收尾",
            parentThreadId: "thread-parent",
            sessionId: "subagent-3",
            status: "completed",
            threadId: "thread-subagent-3",
            updatedAtMs: 2,
          },
        ],
        onOpenOutput,
      },
    });
    const trigger = container.querySelector(
      '[data-testid="task-center-environment-trigger"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      trigger?.click();
      await new Promise((resolve) => setTimeout(resolve, 20));
    });

    const popover = document.body.querySelector(
      '[data-testid="task-center-environment-popover"]',
    );
    expect(popover?.textContent).toContain("环境信息");
    expect(popover?.className).toContain("w-[min(300px,calc(100vw-1rem))]");
    expect(popover?.className).not.toContain("30rem");
    expect(
      document.body.querySelector('[data-testid="task-center-task-rail"]'),
    ).toBeNull();
  });

  it("环境信息的分支菜单应提供搜索、当前分支状态和创建入口", async () => {
    const container = renderToolbar();
    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="task-center-environment-trigger"]',
        )
        ?.click();
      await new Promise((resolve) => setTimeout(resolve, 20));
    });

    const menu = document.body.querySelector(
      '[data-testid="task-center-environment-branch-menu"]',
    );
    expect(menu?.textContent).toContain("分支");
    expect(menu?.textContent).toContain("feature/task-center");
    expect(menu?.textContent).toContain("未提交：3 个文件");
    expect(menu?.textContent).toContain("main");
    expect(menu?.textContent).toContain("创建并检出新分支...");
    expect(
      menu?.querySelector<HTMLInputElement>('input[placeholder="搜索分支"]'),
    ).not.toBeNull();

    const mainBranchButton = Array.from(
      menu?.querySelectorAll<HTMLButtonElement>("button") ?? [],
    ).find((button) => button.textContent?.includes("main"));
    await act(async () => {
      mainBranchButton?.click();
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
    expect(mockCheckoutProjectGitBranch).toHaveBeenCalledWith(
      "/tmp/project",
      "main",
    );
  });

  it("默认不打开环境信息时不应读取项目 Git 状态", () => {
    renderToolbar({
      taskRail: {
        workflowSteps: [
          { id: "read", title: "读取任务区结构", status: "completed" },
        ],
        messages: [],
        threadRead: {
          thread_id: "thread-heavy",
          active_turn_id: "turn-heavy",
          status: "completed",
        },
        canonicalChildren: [
          {
            name: "实现",
            parentThreadId: "thread-parent",
            sessionId: "subagent-heavy",
            status: "completed",
            threadId: "thread-subagent-heavy",
            updatedAtMs: 2,
          },
        ],
      },
    });

    expect(mockReadProjectGitStatus).not.toHaveBeenCalled();
    expect(
      document.body.querySelector('[data-testid="task-center-task-rail"]'),
    ).toBeNull();
  });

  it("导入 provenance 不应创建 imported-only 完整运行记录入口", async () => {
    const container = renderToolbar({
      taskRail: {
        sessionId: "session-imported",
        workflowSteps: [],
        messages: [],
        context: {
          sourceCount: 1,
          sourceLabels: ["restored-history"],
        },
        threadItems: [
          {
            id: "imported-command",
            type: "command_execution",
            thread_id: "thread-1",
            turn_id: "turn-1",
            sequence: 1,
            status: "completed",
            command: "npm test",
            cwd: "/workspace/imported-history",
            started_at: "2026-06-16T10:00:00.000Z",
            completed_at: "2026-06-16T10:00:01.000Z",
            updated_at: "2026-06-16T10:00:01.000Z",
            metadata: {
              imported: true,
              source_client: "codex",
              sourcePath: "/Users/example/.codex/sessions/thread.jsonl",
            },
          },
        ],
      },
    });
    const trigger = container.querySelector(
      '[data-testid="task-center-environment-trigger"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      trigger?.click();
      await new Promise((resolve) => setTimeout(resolve, 20));
    });

    expect(
      document.body.querySelector(
        '[data-testid="imported-runtime-detail-toggle"]',
      ),
    ).toBeNull();
    expect(
      document.body.querySelector(
        '[data-testid="imported-runtime-detail-panel"]',
      ),
    ).toBeNull();
    expect(document.body.textContent).not.toContain("查看完整记录");
    expect(document.body.textContent).not.toContain(
      "/Users/example/.codex/sessions/thread.jsonl",
    );
  });

  it("普通来源会话不应显示完整运行记录入口", async () => {
    const container = renderToolbar({
      taskRail: {
        sessionId: "session-normal",
        workflowSteps: [],
        messages: [],
        context: {
          sourceCount: 1,
          sourceLabels: ["docs.example.com"],
        },
        threadItems: [
          {
            id: "web-source",
            type: "web_search",
            thread_id: "thread-1",
            turn_id: "turn-1",
            sequence: 1,
            status: "completed",
            query: "workspace docs",
            started_at: "2026-06-16T10:00:00.000Z",
            completed_at: "2026-06-16T10:00:01.000Z",
            updated_at: "2026-06-16T10:00:01.000Z",
          },
        ],
      },
    });
    const trigger = container.querySelector(
      '[data-testid="task-center-environment-trigger"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      trigger?.click();
      await new Promise((resolve) => setTimeout(resolve, 20));
    });

    expect(
      document.body.querySelector(
        '[data-testid="imported-runtime-detail-panel"]',
      ),
    ).toBeNull();
  });

  it("Shell、工作台与聊天按钮应分别接入真实能力并保持当前态", async () => {
    const onToggleCanvas = vi.fn();
    const onToggleShellPanel = vi.fn();
    const container = renderToolbar({
      isCanvasOpen: false,
      onToggleCanvas,
      onToggleShellPanel,
    });
    const shellButton = container.querySelector(
      '[data-testid="task-center-shell-toggle"]',
    ) as HTMLButtonElement | null;
    const workbenchButton = container.querySelector(
      '[data-testid="task-center-workbench-toggle"]',
    ) as HTMLButtonElement | null;
    const chatButton = container.querySelector(
      '[data-testid="task-center-chat-toggle"]',
    ) as HTMLButtonElement | null;
    const toolGroups = container.querySelectorAll(
      '[data-testid^="task-center-tool-group-"]',
    );

    expect(shellButton?.disabled).toBe(false);
    expect(workbenchButton?.disabled).toBe(false);
    expect(chatButton).toBeNull();
    expect(toolGroups).toHaveLength(3);
    expect(
      container.querySelector('[data-testid="task-center-tool-group-app"]'),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="task-center-tool-group-environment"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="task-center-tool-group-panels"]'),
    ).not.toBeNull();

    await act(async () => {
      shellButton?.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(onToggleShellPanel).toHaveBeenCalledTimes(1);

    act(() => {
      workbenchButton?.click();
    });

    expect(onToggleCanvas).toHaveBeenCalledTimes(1);
  });

  it("Task Center 应在可切换 Harness 时保留 5 个工具按钮", () => {
    const onToggleHarnessPanel = vi.fn();
    const container = renderToolbar({
      showHarnessToggle: false,
      onToggleHarnessPanel,
    });
    const harnessButton = container.querySelector(
      '[data-testid="task-center-harness-toggle"]',
    ) as HTMLButtonElement | null;
    const panelGroup = container.querySelector(
      '[data-testid="task-center-tool-group-panels"]',
    );

    expect(harnessButton).not.toBeNull();
    expect(panelGroup?.querySelectorAll("button")).toHaveLength(3);

    act(() => {
      harnessButton?.click();
    });

    expect(onToggleHarnessPanel).toHaveBeenCalledTimes(1);
  });

  it("没有项目目录时 Shell 入口应 fail-closed", () => {
    const container = renderToolbar({ projectRootPath: null });
    const shellButton = container.querySelector(
      '[data-testid="task-center-shell-toggle"]',
    ) as HTMLButtonElement | null;

    expect(shellButton?.disabled).toBe(true);
  });
});

describe("TaskCenterShellPanel", () => {
  it("应固定渲染底部 xterm Shell 面板并启动 command/exec PTY", async () => {
    const onClose = vi.fn();
    const onHeightChange = vi.fn();
    const onToggleMaximize = vi.fn();
    const container = mount(
      <TaskCenterShellPanel
        heightPx={236}
        maximized={false}
        projectRootPath="/tmp/project"
        onClose={onClose}
        onHeightChange={onHeightChange}
        onToggleMaximize={onToggleMaximize}
      />,
    );
    const panel = container.querySelector(
      '[data-testid="task-center-bottom-shell-panel"]',
    );

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(panel).not.toBeNull();
    expect((panel as HTMLElement | null)?.style.height).toBe("236px");
    expect(
      container.querySelector('[data-testid="task-center-shell-run"]'),
    ).toBeNull();
    expect(mockExecCommand).toHaveBeenCalledWith({
      command: ["/bin/sh", "-i"],
      processId: expect.any(String),
      tty: true,
      streamStdin: true,
      streamStdoutStderr: true,
      disableOutputCap: true,
      disableTimeout: true,
      cwd: "/tmp/project",
      size: { cols: 120, rows: 14 },
    });
    expect(mockSubscribeCommandExecOutput).toHaveBeenCalledTimes(1);
    expect(mockSubscribeCommandExecOutput).toHaveBeenCalledWith(
      expect.any(String),
      expect.any(Function),
    );
    expect(container.textContent).toContain("project");
    expect(mockXtermLoadAddon).toHaveBeenCalledTimes(1);
    expect(mockFitAddonFit).toHaveBeenCalled();
    expect(mockXtermTerminalOptions[0]).toMatchObject({
      theme: expect.objectContaining({
        background: "#ffffff",
        foreground: "#1f2937",
        blue: "#0969da",
        brightBlue: "#1d4ed8",
        green: "#16a34a",
        brightGreen: "#22c55e",
        yellow: "#ca8a04",
        magenta: "#c026d3",
        scrollbarSliderBackground: "#cbd5e1",
      }),
    });
    expect(mockXtermWriteln).not.toHaveBeenCalledWith(
      "Shell 已就绪，可以输入命令",
    );
    expect(
      container.querySelector('[data-testid="task-center-shell-terminal"]')
        ?.className,
    ).toContain("[&_.xterm]:!bg-white");

    act(() => {
      (
        container.querySelector(
          '[data-testid="task-center-shell-maximize"]',
        ) as HTMLButtonElement | null
      )?.click();
    });

    expect(onToggleMaximize).toHaveBeenCalledTimes(1);

    act(() => {
      (
        container.querySelector(
          '[data-testid="task-center-shell-close"]',
        ) as HTMLButtonElement | null
      )?.click();
    });

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("快捷动作应通过 command/exec/write 写入真实 PTY，而不是前端伪造输出", async () => {
    vi.stubGlobal(
      "prompt",
      vi.fn(
        () => "src/components/agent/chat/components/TaskCenterShellPanel.tsx",
      ),
    );
    mount(
      <TaskCenterShellPanel
        heightPx={236}
        maximized={false}
        projectRootPath="/tmp/project"
        onClose={vi.fn()}
        onHeightChange={vi.fn()}
        onToggleMaximize={vi.fn()}
      />,
    );

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    const listFilesButton = document.body.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-shell-list-files"]',
    );
    const viewFileButton = document.body.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-shell-view-file"]',
    );
    const gitStatusButton = document.body.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-shell-git-status"]',
    );
    const clearButton = document.body.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-shell-clear"]',
    );

    await act(async () => {
      listFilesButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    const processId = mockExecCommand.mock.calls[0]?.[0]?.processId;
    expect(processId).toEqual(expect.any(String));
    expect(mockWriteCommandExec).toHaveBeenLastCalledWith({
      processId,
      deltaBase64: expect.any(String),
    });
    expect(atob(mockWriteCommandExec.mock.lastCall?.[0].deltaBase64)).toContain(
      "ls -la",
    );

    await act(async () => {
      viewFileButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(globalThis.prompt).toHaveBeenCalled();
    expect(mockWriteCommandExec).toHaveBeenLastCalledWith({
      processId,
      deltaBase64: expect.any(String),
    });
    expect(atob(mockWriteCommandExec.mock.lastCall?.[0].deltaBase64)).toContain(
      "TaskCenterShellPanel.tsx",
    );

    await act(async () => {
      gitStatusButton?.click();
      clearButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockWriteCommandExec).toHaveBeenCalledWith({
      processId,
      deltaBase64: btoa("git -c color.status=always status --short --branch\r"),
    });
    expect(mockWriteCommandExec).toHaveBeenCalledWith({
      processId,
      deltaBase64: btoa("clear\r"),
    });
    expect(mockXtermWriteln).not.toHaveBeenCalledWith(
      expect.stringContaining("TaskCenterShellPanel.tsx"),
    );
    vi.unstubAllGlobals();
  });

  it("点击新增 Shell 会话应创建独立 command/exec process 并保留原 tab", async () => {
    const container = mount(
      <TaskCenterShellPanel
        heightPx={236}
        maximized={false}
        projectRootPath="/tmp/project"
        onClose={vi.fn()}
        onHeightChange={vi.fn()}
        onToggleMaximize={vi.fn()}
      />,
    );

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(mockExecCommand).toHaveBeenCalledTimes(1);
    expect(
      container.querySelectorAll('[data-testid="task-center-shell-tab"]'),
    ).toHaveLength(1);

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="task-center-shell-new-session"]',
        )
        ?.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(mockExecCommand).toHaveBeenCalledTimes(2);
    expect(
      container.querySelectorAll('[data-testid="task-center-shell-tab"]'),
    ).toHaveLength(2);
    const firstProcessId = mockExecCommand.mock.calls[0]?.[0]?.processId;
    const secondProcessId = mockExecCommand.mock.calls[1]?.[0]?.processId;
    expect(firstProcessId).toEqual(expect.any(String));
    expect(secondProcessId).toEqual(expect.any(String));
    expect(firstProcessId).not.toBe(secondProcessId);

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="task-center-shell-tab-button-shell-tab-1"]',
        )
        ?.click();
      await Promise.resolve();
    });

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="task-center-shell-clear"]',
        )
        ?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockWriteCommandExec).toHaveBeenLastCalledWith({
      processId: firstProcessId,
      deltaBase64: btoa("clear\r"),
    });
    expect(mockTerminateCommandExec).not.toHaveBeenCalledWith({
      processId: firstProcessId,
    });
  });

  it("command/exec 启动失败时应停留在失败态，不重建第二条旧 session 链", async () => {
    mockExecCommand.mockRejectedValueOnce(
      new Error("command/exec unavailable"),
    );
    mount(
      <TaskCenterShellPanel
        heightPx={236}
        maximized={false}
        projectRootPath="/tmp/project"
        onClose={vi.fn()}
        onHeightChange={vi.fn()}
        onToggleMaximize={vi.fn()}
      />,
    );

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(mockExecCommand).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).toContain("command/exec unavailable");
  });

  it("应消费 command/exec/outputDelta 并在卸载时清理通知订阅和进程", async () => {
    let outputHandler:
      | ((event: {
          processId: string;
          stream: "stdout";
          deltaBase64: string;
          capReached: boolean;
        }) => void)
      | null = null;
    const unlisten = vi.fn();
    mockSubscribeCommandExecOutput.mockImplementationOnce(
      (processId, handler) => {
        outputHandler = handler;
        handler({
          processId,
          stream: "stdout",
          deltaBase64: btoa("early prompt"),
          capReached: false,
        });
        return unlisten;
      },
    );

    const container = mount(
      <TaskCenterShellPanel
        heightPx={236}
        maximized={false}
        projectRootPath="/tmp/project"
        onClose={vi.fn()}
        onHeightChange={vi.fn()}
        onToggleMaximize={vi.fn()}
      />,
    );

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(outputHandler).not.toBeNull();
    expect(mockXtermWrite).toHaveBeenCalledWith("early prompt");

    act(() => {
      mountedRoots.pop()?.root.unmount();
    });
    container.remove();

    expect(unlisten).toHaveBeenCalledTimes(1);
    expect(mockXtermDisposeInput).toHaveBeenCalledTimes(1);
    expect(mockTerminateCommandExec).toHaveBeenCalledWith({
      processId: expect.any(String),
    });
  });

  it("应支持拖拽调整 Shell 高度并重新适配终端", async () => {
    const onHeightChange = vi.fn();
    const container = mount(
      <TaskCenterShellPanel
        heightPx={236}
        maximized={false}
        projectRootPath="/tmp/project"
        onClose={vi.fn()}
        onHeightChange={onHeightChange}
        onToggleMaximize={vi.fn()}
      />,
    );

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    const resizeHandle = container.querySelector(
      '[data-testid="task-center-shell-resize-handle"]',
    ) as HTMLButtonElement | null;

    act(() => {
      resizeHandle?.dispatchEvent(
        new PointerEvent("pointerdown", {
          bubbles: true,
          clientY: 500,
          pointerId: 1,
        }),
      );
      resizeHandle?.dispatchEvent(
        new PointerEvent("pointermove", {
          bubbles: true,
          clientY: 420,
          pointerId: 1,
        }),
      );
      resizeHandle?.dispatchEvent(
        new PointerEvent("pointerup", {
          bubbles: true,
          clientY: 420,
          pointerId: 1,
        }),
      );
    });

    expect(onHeightChange).toHaveBeenCalledWith(316);
    expect(mockFitAddonFit).toHaveBeenCalled();
  });

  it("应串行写入快速输入片段，避免 PTY 收到乱序字符", async () => {
    let resolveFirstWrite: () => void = () => {
      throw new Error("first write promise was not created");
    };
    mockWriteCommandExec
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            resolveFirstWrite = resolve;
          }),
      )
      .mockResolvedValueOnce(undefined);

    mount(
      <TaskCenterShellPanel
        heightPx={236}
        maximized={false}
        projectRootPath="/tmp/project"
        onClose={vi.fn()}
        onHeightChange={vi.fn()}
        onToggleMaximize={vi.fn()}
      />,
    );

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    const onData = mockXtermOnDataHandlers.at(-1);
    expect(onData).toBeTypeOf("function");
    if (!onData) {
      throw new Error("xterm onData handler was not registered");
    }

    act(() => {
      onData("first\r");
      onData("second\r");
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(mockWriteCommandExec).toHaveBeenCalledTimes(1);
    const processId = mockExecCommand.mock.calls[0]?.[0]?.processId;
    expect(mockWriteCommandExec).toHaveBeenNthCalledWith(1, {
      processId,
      deltaBase64: btoa("first\r"),
    });

    resolveFirstWrite();

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockWriteCommandExec).toHaveBeenCalledTimes(2);
    expect(mockWriteCommandExec).toHaveBeenNthCalledWith(2, {
      processId,
      deltaBase64: btoa("second\r"),
    });
  });

  it("应合并快速输入片段并在 Enter 时立即写入", async () => {
    vi.useFakeTimers();
    mount(
      <TaskCenterShellPanel
        heightPx={236}
        maximized={false}
        projectRootPath="/tmp/project"
        onClose={vi.fn()}
        onHeightChange={vi.fn()}
        onToggleMaximize={vi.fn()}
      />,
    );

    await act(async () => {
      await vi.runAllTimersAsync();
      await Promise.resolve();
    });

    const onData = mockXtermOnDataHandlers.at(-1);
    expect(onData).toBeTypeOf("function");

    act(() => {
      onData?.("pri");
      onData?.("ntf");
    });

    expect(mockWriteCommandExec).not.toHaveBeenCalled();

    act(() => {
      onData?.("\r");
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockWriteCommandExec).toHaveBeenCalledTimes(1);
    const processId = mockExecCommand.mock.calls[0]?.[0]?.processId;
    expect(mockWriteCommandExec).toHaveBeenCalledWith({
      processId,
      deltaBase64: btoa("printf\r"),
    });
    vi.useRealTimers();
  });
});
