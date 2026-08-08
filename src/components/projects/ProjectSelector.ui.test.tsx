import React from "react";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanupMountedProjectSelectors,
  createProject,
  createUseProjectsResult,
  findButton,
  flushAsync,
  renderProjectSelector,
} from "./ProjectSelector.uiTestFixtures";

const {
  mockUseProjects,
  mockToastError,
  mockToastSuccess,
  mockGetProject,
  mockGetDefaultProject,
  mockGetProjectByRootPath,
  mockRevealPathInFinder,
  mockOpenDialog,
} = vi.hoisted(() => ({
  mockUseProjects: vi.fn(),
  mockToastError: vi.fn(),
  mockToastSuccess: vi.fn(),
  mockGetProject: vi.fn(),
  mockGetDefaultProject: vi.fn(),
  mockGetProjectByRootPath: vi.fn(),
  mockRevealPathInFinder: vi.fn(),
  mockOpenDialog: vi.fn(),
}));

vi.mock("sonner", () => ({
  toast: {
    success: mockToastSuccess,
    error: mockToastError,
  },
}));

vi.mock("react-i18next", async () => {
  const { translateProjectSelectorTestKey } = await import(
    "./ProjectSelector.i18nTestStub"
  );
  return {
    useTranslation: () => ({
      t: translateProjectSelectorTestKey,
    }),
  };
});

vi.mock("@/hooks/useProjects", () => ({
  useProjects: mockUseProjects,
}));

vi.mock("@/lib/api/fileSystem", () => ({
  revealPathInFinder: mockRevealPathInFinder,
}));

vi.mock("@/lib/desktop-host/dialog", () => ({
  open: mockOpenDialog,
}));

vi.mock("@/lib/api/project", async () => {
  const actual =
    await vi.importActual<typeof import("@/lib/api/project")>(
      "@/lib/api/project",
    );

  return {
    ...actual,
    getProject: mockGetProject,
    getDefaultProject: mockGetDefaultProject,
    getProjectByRootPath: mockGetProjectByRootPath,
  };
});

vi.mock("@/components/projects/CreateProjectDialog", () => ({
  CreateProjectDialog: ({
    open,
    defaultType,
    allowedTypes,
  }: {
    open: boolean;
    defaultType?: string;
    allowedTypes?: string[];
  }) =>
    open ? (
      <div
        data-testid="create-project-dialog"
        data-default-type={defaultType}
        data-allowed-types={(allowedTypes || []).join(",")}
      />
    ) : null,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({ children, onClick, disabled, type = "button", ...rest }: {
    children: React.ReactNode;
    onClick?: () => void;
    disabled?: boolean;
    type?: "button" | "submit" | "reset";
    [key: string]: unknown;
  }) => <button type={type} onClick={onClick} disabled={disabled} {...rest}>{children}</button>,
}));

vi.mock("@/components/ui/input", () => ({
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => (
    <input {...props} />
  ),
}));

vi.mock("@/components/ui/popover", () => ({
  Popover: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  PopoverContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  PopoverTrigger: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: (props: React.HTMLAttributes<HTMLDivElement>) => (
    <div {...props} />
  ),
}));

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) =>
    open ? <div>{children}</div> : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogFooter: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  vi.clearAllMocks();
  mockGetProject.mockResolvedValue(null);
  mockGetDefaultProject.mockResolvedValue(null);
  mockGetProjectByRootPath.mockResolvedValue(null);
  mockOpenDialog.mockResolvedValue(null);
});

afterEach(() => {
  cleanupMountedProjectSelectors();
  vi.clearAllMocks();
});

describe("ProjectSelector 组件", () => {
  it("默认项目在管理模式下不可重命名或移除", () => {
    const defaultProject = createProject({
      id: "default",
      name: "默认项目",
      isDefault: true,
      workspaceType: "general",
    });
    const generalProject = createProject({
      id: "general-1",
      name: "通用项目",
      workspaceType: "general",
    });

    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [defaultProject, generalProject],
        generalProjects: [defaultProject, generalProject],
        defaultProject,
      }),
    );

    const container = renderProjectSelector({
      value: defaultProject.id,
      workspaceType: "general",
      enableManagement: true,
    });

    expect(findButton(container, "重命名")?.disabled).toBe(true);
    expect(findButton(container, "移除")?.disabled).toBe(true);
    expect(container.textContent).toContain("默认项目不可重命名或移除");
  });

  it("移除当前项目后应回退到默认项目", async () => {
    const onChange = vi.fn();
    const remove = vi.fn(async () => true);
    const defaultProject = createProject({
      id: "default",
      name: "默认项目",
      isDefault: true,
      workspaceType: "general",
    });
    const generalProject = createProject({
      id: "general-1",
      name: "通用项目",
      workspaceType: "general",
    });

    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [defaultProject, generalProject],
        generalProjects: [defaultProject, generalProject],
        defaultProject,
        remove,
      }),
    );

    const container = renderProjectSelector({
      value: generalProject.id,
      workspaceType: "general",
      enableManagement: true,
      onChange,
    });

    act(() => {
      findButton(container, "移除")?.click();
    });
    await flushAsync();

    await act(async () => {
      findButton(container, "移除项目")?.click();
      await Promise.resolve();
    });
    await flushAsync();

    expect(remove).toHaveBeenCalledWith(generalProject.id);
    expect(onChange).toHaveBeenCalledWith(defaultProject.id);
    expect(mockToastSuccess).toHaveBeenCalledWith("项目已移除，本地目录未删除");
  });

  it("未启用管理模式时不显示管理操作区", () => {
    const defaultProject = createProject({
      id: "default",
      name: "默认项目",
      isDefault: true,
      workspaceType: "general",
    });

    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [defaultProject],
        generalProjects: [defaultProject],
        defaultProject,
      }),
    );

    const container = renderProjectSelector({
      value: defaultProject.id,
      workspaceType: "general",
      enableManagement: false,
    });

    expect(findButton(container, "新建项目")).toBeNull();
    expect(findButton(container, "重命名")).toBeNull();
    expect(findButton(container, "移除")).toBeNull();
  });

  it("workspace-tab 模式应复刻轻量工作区列表", () => {
    const defaultProject = createProject({
      id: "default",
      name: "默认工作区",
      isDefault: true,
      workspaceType: "general",
    });

    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [defaultProject],
        generalProjects: [defaultProject],
        defaultProject,
      }),
    );

    const container = renderProjectSelector({
      value: defaultProject.id,
      workspaceType: "general",
      enableManagement: true,
      chrome: "workspace-tab",
      placeholder: "选择工作区",
    });

    expect(container.textContent).toContain("默认工作区");
    expect(
      container.querySelector(".lucide-folder")?.getAttribute("class"),
    ).toContain("h-4");
    expect(container.textContent).not.toContain("工作区管理");
    expect(container.textContent).not.toContain("搜索工作区");
    expect(
      container.querySelector('[data-testid="workspace-selector-scroll"]')
        ?.className,
    ).toContain("max-h-[308px]");
    expect(findButton(container, "打开现有文件夹")).not.toBeNull();
    expect(findButton(container, "新建工作区")).not.toBeNull();
    expect(findButton(container, "新建项目")).toBeNull();
  });

  it("workspace-tab 列表非默认工作区行尾应提供移除入口", async () => {
    const remove = vi.fn(async () => true);
    const defaultProject = createProject({
      id: "default",
      name: "默认工作区",
      isDefault: true,
      workspaceType: "general",
    });
    const normalProject = createProject({
      id: "normal",
      name: "普通工作区",
      workspaceType: "general",
    });

    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [defaultProject, normalProject],
        generalProjects: [defaultProject, normalProject],
        defaultProject,
        remove,
      }),
    );

    const container = renderProjectSelector({
      value: defaultProject.id,
      workspaceType: "general",
      enableManagement: true,
      chrome: "workspace-tab",
    });

    const removeButtons = Array.from(
      container.querySelectorAll('button[aria-label="移除工作区"]'),
    ) as HTMLButtonElement[];
    expect(removeButtons).toHaveLength(1);
    expect(removeButtons[0]?.className).toContain("opacity-0");
    expect(removeButtons[0]?.className).toContain("group-hover:opacity-100");

    act(() => {
      removeButtons[0]?.click();
    });
    await flushAsync();

    expect(container.textContent).toContain("移除工作区");
    expect(container.textContent).toContain("普通工作区");
  });

  it("workspace-tab 当前工作区行尾应显示对勾", () => {
    const defaultProject = createProject({
      id: "default",
      name: "默认工作区",
      isDefault: true,
      workspaceType: "general",
    });
    const normalProject = createProject({
      id: "normal",
      name: "普通工作区",
      workspaceType: "general",
    });

    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [defaultProject, normalProject],
        generalProjects: [defaultProject, normalProject],
        defaultProject,
      }),
    );

    const container = renderProjectSelector({
      value: normalProject.id,
      workspaceType: "general",
      enableManagement: true,
      chrome: "workspace-tab",
    });

    const defaultRow = container.querySelector(
      '[data-testid="workspace-selector-row-default"]',
    );
    const normalRow = container.querySelector(
      '[data-testid="workspace-selector-row-normal"]',
    );
    const rows = Array.from(
      container.querySelectorAll('[data-testid^="workspace-selector-row-"]'),
    );

    expect(rows[0]).toBe(normalRow);
    expect(defaultRow?.getAttribute("data-selected")).toBe("false");
    expect(defaultRow?.querySelector(".lucide-check")).toBeNull();
    expect(normalRow?.getAttribute("data-selected")).toBe("true");
    expect(normalRow?.querySelector(".lucide-check")).not.toBeNull();
  });

  it("workspace-tab 未显式选择时应把默认工作区显示为当前选中行", () => {
    const defaultProject = createProject({
      id: "default",
      name: "默认工作区",
      isDefault: true,
      workspaceType: "general",
    });
    const normalProject = createProject({
      id: "normal",
      name: "普通工作区",
      workspaceType: "general",
    });

    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [normalProject, defaultProject],
        generalProjects: [normalProject, defaultProject],
        defaultProject,
      }),
    );

    const container = renderProjectSelector({
      value: null,
      workspaceType: "general",
      enableManagement: true,
      chrome: "workspace-tab",
    });

    const rows = Array.from(
      container.querySelectorAll('[data-testid^="workspace-selector-row-"]'),
    );
    const defaultRow = container.querySelector(
      '[data-testid="workspace-selector-row-default"]',
    );
    const normalRow = container.querySelector(
      '[data-testid="workspace-selector-row-normal"]',
    );

    expect(rows[0]).toBe(defaultRow);
    expect(defaultRow?.getAttribute("data-selected")).toBe("true");
    expect(defaultRow?.querySelector(".lucide-check")).not.toBeNull();
    expect(defaultRow?.className).toContain("bg-slate-100");
    expect(normalRow?.getAttribute("data-selected")).toBe("false");
  });

  it("workspace-tab 当前工作区不在已加载列表时仍应置顶并显示选中态", () => {
    const defaultProject = createProject({
      id: "default",
      name: "默认工作区",
      rootPath: "/Users/coso/.newmax/workspace",
      isDefault: true,
      workspaceType: "general",
    });
    const normalProject = createProject({
      id: "normal",
      name: "普通工作区",
      workspaceType: "general",
    });

    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [normalProject],
        generalProjects: [normalProject],
        defaultProject,
      }),
    );

    const container = renderProjectSelector({
      value: defaultProject.id,
      workspaceType: "general",
      enableManagement: true,
      chrome: "workspace-tab",
    });

    const rows = Array.from(
      container.querySelectorAll('[data-testid^="workspace-selector-row-"]'),
    );
    const defaultRow = container.querySelector(
      '[data-testid="workspace-selector-row-default"]',
    );
    const normalRow = container.querySelector(
      '[data-testid="workspace-selector-row-normal"]',
    );

    expect(rows[0]).toBe(defaultRow);
    expect(defaultRow?.textContent).toContain("默认工作区");
    expect(defaultRow?.getAttribute("data-selected")).toBe("true");
    expect(defaultRow?.querySelector(".lucide-check")).not.toBeNull();
    expect(normalRow?.getAttribute("data-selected")).toBe("false");
  });

  it("workspace-tab 打开现有文件夹时应复用已存在工作区", async () => {
    const onChange = vi.fn();
    const defaultProject = createProject({
      id: "default",
      name: "默认工作区",
      isDefault: true,
      workspaceType: "general",
    });
    const existingProject = createProject({
      id: "existing",
      name: "已有工作区",
      rootPath: "/tmp/existing",
      workspaceType: "general",
    });

    mockOpenDialog.mockResolvedValue("/tmp/existing");
    mockGetProjectByRootPath.mockResolvedValue(existingProject);
    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [defaultProject],
        generalProjects: [defaultProject],
        defaultProject,
      }),
    );

    const container = renderProjectSelector({
      value: defaultProject.id,
      onChange,
      workspaceType: "general",
      enableManagement: true,
      chrome: "workspace-tab",
    });

    await act(async () => {
      findButton(container, "打开现有文件夹")?.click();
      await Promise.resolve();
    });
    await flushAsync();

    expect(mockOpenDialog).toHaveBeenCalledWith({
      directory: true,
      multiple: false,
    });
    expect(mockGetProjectByRootPath).toHaveBeenCalledWith("/tmp/existing");
    expect(onChange).toHaveBeenCalledWith("existing");
    expect(mockToastSuccess).not.toHaveBeenCalled();
  });

  it("workspace-tab 打开新目录时应创建工作区记录", async () => {
    const onChange = vi.fn();
    const create = vi.fn(async () =>
      createProject({
        id: "new-workspace",
        name: "new-workspace",
        rootPath: "/tmp/new-workspace",
        workspaceType: "general",
      }),
    );
    const defaultProject = createProject({
      id: "default",
      name: "默认工作区",
      isDefault: true,
      workspaceType: "general",
    });

    mockOpenDialog.mockResolvedValue("/tmp/new-workspace");
    mockGetProjectByRootPath.mockResolvedValue(null);
    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [defaultProject],
        generalProjects: [defaultProject],
        defaultProject,
        create,
      }),
    );

    const container = renderProjectSelector({
      value: defaultProject.id,
      onChange,
      workspaceType: "general",
      enableManagement: true,
      chrome: "workspace-tab",
    });

    await act(async () => {
      findButton(container, "打开现有文件夹")?.click();
      await Promise.resolve();
    });
    await flushAsync();

    expect(create).toHaveBeenCalledWith({
      name: "new-workspace",
      rootPath: "/tmp/new-workspace",
      workspaceType: "general",
    });
    expect(onChange).toHaveBeenCalledWith("new-workspace");
    expect(mockToastSuccess).not.toHaveBeenCalled();
  });

  it("受控打开且延后加载时应刷新完整工作区列表", async () => {
    const refresh = vi.fn(async () => undefined);
    const defaultProject = createProject({
      id: "default",
      name: "默认工作区",
      isDefault: true,
      workspaceType: "general",
    });

    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [],
        generalProjects: [],
        defaultProject,
        refresh,
      }),
    );

    renderProjectSelector({
      value: defaultProject.id,
      workspaceType: "general",
      enableManagement: true,
      chrome: "workspace-tab",
      open: true,
      onOpenChange: vi.fn(),
      deferProjectListLoad: true,
    });

    await flushAsync();

    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it("显示本地位置时只调用系统文件管理器，不额外弹成功提示", async () => {
    const defaultProject = createProject({
      id: "default",
      name: "默认项目",
      isDefault: true,
      workspaceType: "general",
      rootPath: "/tmp/default-project",
    });

    mockRevealPathInFinder.mockResolvedValue(undefined);
    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [defaultProject],
        generalProjects: [defaultProject],
        defaultProject,
      }),
    );

    const container = renderProjectSelector({
      value: defaultProject.id,
      workspaceType: "general",
      enableManagement: true,
    });

    await act(async () => {
      findButton(container, "显示")?.click();
      await Promise.resolve();
    });
    await flushAsync();

    expect(mockRevealPathInFinder).toHaveBeenCalledWith("/tmp/default-project");
    expect(mockToastSuccess).not.toHaveBeenCalled();
  });

  it("查看内容应进入当前工作区内容视图", () => {
    const onOpenProjectContents = vi.fn();
    const generalProject = createProject({
      id: "general-1",
      name: "通用项目",
      workspaceType: "general",
    });

    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [generalProject],
        generalProjects: [generalProject],
        defaultProject: null,
      }),
    );

    const container = renderProjectSelector({
      value: generalProject.id,
      workspaceType: "general",
      enableManagement: true,
      onOpenProjectContents,
    });

    act(() => {
      findButton(container, "查看内容")?.click();
    });

    expect(onOpenProjectContents).toHaveBeenCalledWith(generalProject.id);
  });

  it("延后列表加载时应先用项目摘要渲染当前项目", async () => {
    const selectedProject = createProject({
      id: "general-1",
      name: "当前项目",
      workspaceType: "general",
    });

    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [],
        generalProjects: [],
        defaultProject: null,
      }),
    );
    mockGetProject.mockResolvedValue(selectedProject);

    const container = renderProjectSelector({
      value: selectedProject.id,
      workspaceType: "general",
      deferProjectListLoad: true,
    });

    await flushAsync();
    await flushAsync();

    expect(mockUseProjects).toHaveBeenCalledWith(
      expect.objectContaining({
        autoLoad: false,
      }),
    );
    expect(mockGetProject).toHaveBeenCalledWith(selectedProject.id);
    expect(container.textContent).toContain(selectedProject.name);
  });

  it("被动展示模式延后列表加载时应推迟项目摘要请求", async () => {
    vi.useFakeTimers();
    try {
      const selectedProject = createProject({
        id: "general-1",
        name: "当前项目",
        workspaceType: "general",
      });

      mockUseProjects.mockReturnValue(
        createUseProjectsResult({
          projects: [],
          generalProjects: [],
          defaultProject: null,
        }),
      );
      mockGetProject.mockResolvedValue(selectedProject);

      renderProjectSelector({
        value: selectedProject.id,
        workspaceType: "general",
        deferProjectListLoad: true,
        passiveTrigger: true,
      });

      await flushAsync();
      expect(mockGetProject).not.toHaveBeenCalled();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(12_000);
      });
      await flushAsync();

      expect(mockGetProject).toHaveBeenCalledWith(selectedProject.id);
    } finally {
      vi.useRealTimers();
    }
  });

  it("延后列表加载且未指定项目时应回填默认项目", async () => {
    const onChange = vi.fn();
    const defaultProject = createProject({
      id: "default",
      name: "默认项目",
      isDefault: true,
      workspaceType: "general",
    });

    mockUseProjects.mockReturnValue(
      createUseProjectsResult({
        projects: [],
        generalProjects: [],
        defaultProject: null,
      }),
    );
    mockGetDefaultProject.mockResolvedValue(defaultProject);

    renderProjectSelector({
      value: null,
      workspaceType: "general",
      deferProjectListLoad: true,
      onChange,
    });

    await flushAsync();
    await flushAsync();

    expect(mockGetDefaultProject).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith(defaultProject.id);
  });
});
