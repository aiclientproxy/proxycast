import React, { useState } from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatNavbar } from "./ChatNavbar";

const {
  mockProjectSelector,
  mockRequestChatHostOpenPath,
  mockEnsureProjectWorkspace,
} = vi.hoisted(() => ({
  mockProjectSelector: vi.fn(),
  mockRequestChatHostOpenPath: vi.fn(),
  mockEnsureProjectWorkspace: vi.fn(),
}));

vi.mock("@/components/projects/ProjectSelector", () => ({
  ProjectSelector: (props: Record<string, unknown>) => {
    mockProjectSelector(props);
    return <div data-testid="project-selector" />;
  },
}));

vi.mock("@/lib/api/project", () => ({
  ensureProjectWorkspace: mockEnsureProjectWorkspace,
  extractErrorMessage: (error: unknown) =>
    error instanceof Error ? error.message : String(error),
  getProject: vi.fn(async (projectId: string) => ({
    id: projectId,
    name: projectId,
    rootPath: null,
  })),
  getWorkspaceProjectsRoot: vi.fn(async () => "/tmp/projects"),
  resolveProjectRootPath: vi.fn(
    async (name: string, root: string) => `${root}/${name}`,
  ),
}));

vi.mock("@/lib/api/projectGit", () => ({
  checkoutProjectGitBranch: vi.fn(),
  createProjectGitBranch: vi.fn(),
  createProjectGitWorktree: vi.fn(),
  readProjectGitStatus: vi.fn(async () => ({
    hasGitRepository: false,
    currentBranch: null,
    branches: [],
    uncommittedFileCount: 0,
  })),
}));

vi.mock("@/components/agent/chat/host/chatHostCapabilities", () => ({
  requestChatHostOpenPath: mockRequestChatHostOpenPath,
}));

vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    i18n: {
      language: "zh-CN",
    },
    t: (key: string, options?: Record<string, unknown>) => {
      const translations: Record<string, string> = {
        "agentChat.inputbar.core.projectContext.selectFolderDialogTitle":
          "选择项目文件夹",
      };
      const template =
        translations[key] ??
        (typeof options?.defaultValue === "string"
          ? options.defaultValue
          : key);

      return template.replace(/{{\s*([^}]+?)\s*}}/g, (_, name: string) =>
        String(options?.[name.trim()] ?? ""),
      );
    },
  }),
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    onClick,
    disabled,
    ...rest
  }: {
    children: React.ReactNode;
    onClick?: () => void;
    disabled?: boolean;
    [key: string]: unknown;
  }) => (
    <button type="button" onClick={onClick} disabled={disabled} {...rest}>
      {children}
    </button>
  ),
}));

interface MountedHarness {
  container: HTMLDivElement;
  root: Root;
}

const mountedRoots: MountedHarness[] = [];

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  mockRequestChatHostOpenPath.mockReset();
  mockEnsureProjectWorkspace.mockReset();
  mockEnsureProjectWorkspace.mockImplementation(
    async (input: { name: string; rootPath?: string | null }) => ({
      id: input.rootPath || input.name,
      name: input.name,
      rootPath: input.rootPath ?? null,
    }),
  );
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) break;
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.clearAllMocks();
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

function renderChatNavbar(
  props?: Partial<React.ComponentProps<typeof ChatNavbar>>,
) {
  const defaultProps: React.ComponentProps<typeof ChatNavbar> = {
    isRunning: false,
    onToggleFullscreen: vi.fn(),
  };

  return mount(<ChatNavbar {...defaultProps} {...props} />);
}

describe("ChatNavbar", () => {
  it("返回按钮应指向新建任务输入页", () => {
    const onBackHome = vi.fn();
    const container = renderChatNavbar({
      onBackHome,
    });

    const button = container.querySelector(
      'button[aria-label="返回新建任务"]',
    ) as HTMLButtonElement | null;

    expect(button).not.toBeNull();

    act(() => {
      button?.click();
    });

    expect(onBackHome).toHaveBeenCalledTimes(1);
  });

  it("有 Harness 信号时应渲染顶栏切换按钮", () => {
    const onToggleHarnessPanel = vi.fn();
    const container = renderChatNavbar({
      showHarnessToggle: true,
      harnessPanelVisible: false,
      harnessPendingCount: 2,
      onToggleHarnessPanel,
    });

    const button = container.querySelector(
      'button[aria-label="打开Harness"]',
    ) as HTMLButtonElement | null;

    expect(button).not.toBeNull();
    expect(button?.textContent).toContain("Harness");
    expect(button?.textContent).toContain("2");
    expect(
      container.querySelector('[data-testid="chat-navbar-trailing-tools"]')
        ?.className,
    ).toContain("whitespace-nowrap");

    act(() => {
      button?.click();
    });

    expect(onToggleHarnessPanel).toHaveBeenCalledTimes(1);
  });

  it("工作区紧凑顶栏应保留执行入口但隐藏项目选择器", () => {
    const container = renderChatNavbar({
      chrome: "workspace-compact",
      showHarnessToggle: true,
    });

    expect(container.querySelector('[aria-label="切换历史"]')).toBeNull();
    expect(
      container.querySelector('[aria-label="打开Harness"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="project-selector"]'),
    ).toBeNull();
  });

  it("任务中心折叠顶栏应隐藏左侧重导航，但保留右侧项目与工具入口", () => {
    const container = renderChatNavbar({
      collapseChrome: true,
      onBackHome: vi.fn(),
      onBackToResources: vi.fn(),
      onBackToProjectManagement: vi.fn(),
      onToggleSettings: vi.fn(),
      showCanvasToggle: true,
      projectId: "project-1",
      workspaceType: "general",
      showContextCompactionAction: true,
    });

    expect(
      container.querySelector('button[aria-label="返回新建任务"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain("返回资源");
    expect(container.textContent).not.toContain("项目管理");
    expect(container.querySelector('[aria-label="切换历史"]')).toBeNull();
    expect(container.querySelector('[aria-label="展开画布"]')).toBeNull();
    expect(
      container.querySelector('[data-testid="project-selector"]'),
    ).not.toBeNull();
    expect(container.querySelector('[aria-label="打开设置"]')).not.toBeNull();
    expect(container.querySelector('[aria-label="压缩上下文"]')).not.toBeNull();
  });

  it("任务中心顶栏应渲染第一层 workspace tab bar", async () => {
    const onBackToProjectManagement = vi.fn();
    const container = renderChatNavbar({
      contextVariant: "task-center",
      projectId: "project-1",
      openedProjects: [{ id: "project-1", name: "project-1" }],
      workspaceType: "general",
      onBackToProjectManagement,
      onToggleSettings: vi.fn(),
    });

    expect(
      container.querySelector('[data-testid="task-center-workspace-bar"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="inputbar-project-context-bar"]'),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="inputbar-project-context-project-trigger"]',
      )?.textContent,
    ).toContain("project-1");
    expect(mockProjectSelector).not.toHaveBeenCalled();
    expect(
      document.body.querySelector(
        '[data-testid="inputbar-project-context-menu"]',
      ),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="inputbar-project-context-mode"]'),
    ).toBeNull();
    expect(
      container.querySelector(
        '[data-testid="inputbar-project-context-branch"]',
      ),
    ).toBeNull();
    const menuTrigger = container.querySelector(
      'button[aria-label="展开工作区菜单"]',
    ) as HTMLButtonElement | null;
    expect(menuTrigger).not.toBeNull();
    const settingsButton = container.querySelector(
      '[aria-label="打开设置"]',
    ) as HTMLButtonElement | null;
    expect(settingsButton).toBeNull();
    expect(container.querySelector('[aria-label="切换历史"]')).toBeNull();

    await act(async () => {
      menuTrigger?.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(onBackToProjectManagement).not.toHaveBeenCalled();
    expect(
      document.body.querySelector(
        '[data-testid="inputbar-project-context-menu"]',
      ),
    ).not.toBeNull();
    expect(
      document.body.querySelector(
        '[data-testid="inputbar-project-context-add-project"]',
      ),
    ).not.toBeNull();
  });

  it("任务中心项目菜单应能通过现有文件夹进入项目", async () => {
    const onProjectChange = vi.fn();
    mockRequestChatHostOpenPath.mockResolvedValueOnce(
      "/Users/test/existing-project",
    );
    mockEnsureProjectWorkspace.mockResolvedValueOnce({
      id: "project-existing",
      name: "existing-project",
      rootPath: "/Users/test/existing-project",
    });

    const container = renderChatNavbar({
      contextVariant: "task-center",
      projectId: null,
      openedProjects: [],
      workspaceType: "general",
      onProjectChange,
    });

    const menuTrigger = container.querySelector(
      'button[aria-label="展开工作区菜单"]',
    ) as HTMLButtonElement | null;

    await act(async () => {
      menuTrigger?.click();
      await Promise.resolve();
    });

    const addProjectButton = document.body.querySelector(
      '[data-testid="inputbar-project-context-add-project"]',
    ) as HTMLButtonElement | null;
    expect(addProjectButton).not.toBeNull();

    act(() => {
      addProjectButton?.focus();
      addProjectButton?.dispatchEvent(
        new MouseEvent("mouseenter", { bubbles: true }),
      );
    });

    const useExistingButton = document.body.querySelector(
      '[data-testid="inputbar-project-context-use-existing"]',
    ) as HTMLButtonElement | null;
    expect(useExistingButton).not.toBeNull();

    await act(async () => {
      useExistingButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockRequestChatHostOpenPath).toHaveBeenCalledWith({
      directory: true,
      multiple: false,
      title: "选择项目文件夹",
    });
    expect(mockEnsureProjectWorkspace).toHaveBeenCalledWith({
      name: "existing-project",
      rootPath: "/Users/test/existing-project",
      workspaceType: "general",
    });
    expect(onProjectChange).toHaveBeenCalledWith("project-existing");
  });

  it("任务中心顶栏应按已打开项目渲染多个项目 tab 并联动切换", () => {
    const onProjectChange = vi.fn();
    const container = renderChatNavbar({
      contextVariant: "task-center",
      projectId: "project-2",
      openedProjects: [
        { id: "project-1", name: "默认项目" },
        { id: "project-2", name: "lime" },
        { id: "project-3", name: "content-factory-app" },
        { id: "project-2", name: "lime duplicate" },
      ],
      workspaceType: "general",
      onProjectChange,
    });

    const projectShells = Array.from(
      container.querySelectorAll("div[data-project-id]"),
    ) as HTMLElement[];
    const openedProjectTabs = Array.from(
      container.querySelectorAll(
        '[data-testid="task-center-opened-project-tab"]',
      ),
    ) as HTMLButtonElement[];

    expect(projectShells.map((shell) => shell.dataset.projectId)).toEqual([
      "project-1",
      "project-2",
      "project-3",
    ]);
    expect(openedProjectTabs).toHaveLength(2);
    expect(openedProjectTabs.map((tab) => tab.textContent)).toEqual([
      "默认项目",
      "content-factory-app",
    ]);

    act(() => {
      openedProjectTabs[0]?.click();
    });

    expect(onProjectChange).toHaveBeenCalledWith("project-1");
  });

  it("任务中心工作区 tab 应显示目录名并支持关闭", () => {
    const onProjectChange = vi.fn();
    const onCloseProject = vi.fn();
    const container = renderChatNavbar({
      contextVariant: "task-center",
      projectId: "project-1",
      openedProjects: [
        {
          id: "project-1",
          name: "默认工作区",
          rootPath: "/Users/coso/Documents/other/conversations",
        },
        {
          id: "project-2",
          name: "工作区二",
          rootPath:
            "/Users/coso/Documents/other/conversations/conv-1777047467972",
        },
      ],
      workspaceType: "general",
      onProjectChange,
      onCloseProject,
    });

    const openedProjectTab = container.querySelector(
      '[data-testid="task-center-opened-project-tab"]',
    ) as HTMLButtonElement | null;
    const inactiveCloseButton = container.querySelector(
      '[data-testid="task-center-opened-project-close-project-2"]',
    ) as HTMLButtonElement | null;
    const activeCloseButton = container.querySelector(
      '[data-testid="task-center-opened-project-close-project-1"]',
    ) as HTMLButtonElement | null;

    expect(openedProjectTab?.textContent).toContain("conv-1777047467972");
    expect(openedProjectTab?.textContent).not.toContain(
      "/Users/coso/Documents",
    );
    expect(openedProjectTab?.getAttribute("title")).toContain(
      "/Users/coso/Documents/other/conversations/conv-1777047467972",
    );
    expect(inactiveCloseButton?.getAttribute("aria-label")).toBe(
      "关闭conv-1777047467972",
    );
    expect(activeCloseButton?.getAttribute("aria-label")).toBe(
      "关闭conversations",
    );

    act(() => {
      inactiveCloseButton?.click();
    });
    act(() => {
      activeCloseButton?.click();
    });

    expect(onCloseProject).toHaveBeenNthCalledWith(1, "project-2");
    expect(onCloseProject).toHaveBeenNthCalledWith(2, "project-1");
    expect(onProjectChange).not.toHaveBeenCalled();
  });

  it("任务中心第一层应保持紧凑比例并让第二层内容覆盖连接弧面", () => {
    const container = renderChatNavbar({
      contextVariant: "task-center",
      projectId: "project-1",
      openedProjects: [{ id: "project-1", name: "project-1" }],
      workspaceType: "general",
    });

    const workspaceBar = container.querySelector(
      '[data-testid="task-center-workspace-bar"]',
    ) as HTMLElement | null;
    const workspaceShell = container.querySelector(
      '[data-testid="task-center-workspace-shell"]',
    ) as HTMLElement | null;

    expect(workspaceBar?.style.zIndex).toBe("8");
    expect(workspaceShell?.className).toContain("h-9");
    expect(workspaceShell?.className).toContain("min-w-[148px]");
    expect(workspaceShell?.className).toContain("max-w-[224px]");
    expect(workspaceShell?.className).toContain(
      "bg-[color:var(--lime-chrome-tab-active-surface)]",
    );
    expect(workspaceShell?.querySelectorAll("span[aria-hidden]")).toHaveLength(
      2,
    );
  });

  it("任务中心工作区提示已下线，不应渲染气泡或拦截加号", async () => {
    const container = renderChatNavbar({
      contextVariant: "task-center",
      projectId: "project-1",
      openedProjects: [{ id: "project-1", name: "project-1" }],
      workspaceType: "general",
    });

    const hint = container.querySelector(
      '[data-testid="task-center-workspace-hint"]',
    ) as HTMLElement | null;
    const menuTrigger = container.querySelector(
      'button[aria-label="展开工作区菜单"]',
    ) as HTMLButtonElement | null;

    expect(hint).toBeNull();

    await act(async () => {
      menuTrigger?.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(
      container.querySelector('[data-testid="task-center-workspace-hint"]'),
    ).toBeNull();
  });

  it("任务中心顶栏不应重复渲染 Harness 状态入口，避免覆盖工具栏", () => {
    const container = renderChatNavbar({
      contextVariant: "task-center",
      showHarnessToggle: true,
      harnessPendingCount: 3,
    });

    const button = container.querySelector(
      'button[aria-label="打开Harness"]',
    ) as HTMLButtonElement | null;

    expect(button).toBeNull();
  });

  it("点击顶栏按钮后应切换 Harness 面板显隐", () => {
    function HarnessToggleHarness() {
      const [visible, setVisible] = useState(false);

      return (
        <>
          <ChatNavbar
            isRunning={false}
            onToggleFullscreen={() => {}}
            showHarnessToggle
            harnessPanelVisible={visible}
            onToggleHarnessPanel={() => setVisible((current) => !current)}
          />
          {visible ? (
            <div data-testid="harness-panel">Harness Panel</div>
          ) : null}
        </>
      );
    }

    const container = mount(<HarnessToggleHarness />);
    const expandButton = container.querySelector(
      'button[aria-label="打开Harness"]',
    ) as HTMLButtonElement | null;

    expect(container.querySelector('[data-testid="harness-panel"]')).toBeNull();

    act(() => {
      expandButton?.click();
    });

    expect(
      container.querySelector('[data-testid="harness-panel"]'),
    ).not.toBeNull();

    const collapseButton = container.querySelector(
      'button[aria-label="关闭Harness"]',
    ) as HTMLButtonElement | null;

    act(() => {
      collapseButton?.click();
    });

    expect(container.querySelector('[data-testid="harness-panel"]')).toBeNull();
  });

  it("Harness 告警态应使用强调样式", () => {
    const container = renderChatNavbar({
      showHarnessToggle: true,
      harnessAttentionLevel: "warning",
      harnessToggleLabel: "执行提醒",
    });

    const button = container.querySelector(
      'button[aria-label="打开执行提醒"]',
    ) as HTMLButtonElement | null;

    expect(button).not.toBeNull();
    expect(button?.className).toContain(
      "border-[color:var(--lime-warning-border)]",
    );
    expect(button?.className).toContain("text-[color:var(--lime-warning)]");
    expect(button?.className).not.toContain("amber-300");
  });

  it("压缩上下文运行中时应禁用顶栏操作", () => {
    const container = renderChatNavbar({
      showContextCompactionAction: true,
      contextCompactionRunning: true,
    });

    const button = container.querySelector(
      'button[aria-label="正在压缩上下文"]',
    ) as HTMLButtonElement | null;

    expect(button).not.toBeNull();
    expect(button?.disabled).toBe(true);
    expect(button?.textContent).toContain("压缩中...");
  });

  it("通用对话项目选择器应启用管理能力", () => {
    renderChatNavbar({
      workspaceType: "general",
      projectId: "project-1",
    });

    expect(mockProjectSelector).toHaveBeenCalled();
    const lastCall = mockProjectSelector.mock.calls.at(-1)?.[0] as
      | Record<string, unknown>
      | undefined;
    expect(lastCall?.enableManagement).toBe(true);
    expect(lastCall?.density).toBe("compact");
    expect(lastCall?.chrome).toBe("embedded");
  });

  it("应支持从右上角工具组打开设置", () => {
    const onToggleSettings = vi.fn();
    const container = renderChatNavbar({
      onToggleSettings,
    });

    const button = container.querySelector(
      'button[aria-label="打开设置"]',
    ) as HTMLButtonElement | null;

    expect(button).not.toBeNull();

    act(() => {
      button?.click();
    });

    expect(onToggleSettings).toHaveBeenCalledTimes(1);
  });

  it("应支持在顶栏展开和折叠画布", () => {
    const onToggleCanvas = vi.fn();
    const container = renderChatNavbar({
      showCanvasToggle: true,
      isCanvasOpen: false,
      onToggleCanvas,
    });

    const expandButton = container.querySelector(
      'button[aria-label="展开画布"]',
    ) as HTMLButtonElement | null;

    expect(expandButton).not.toBeNull();

    act(() => {
      expandButton?.click();
    });

    expect(onToggleCanvas).toHaveBeenCalledTimes(1);
  });
});
