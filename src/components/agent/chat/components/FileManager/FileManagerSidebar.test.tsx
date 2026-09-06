import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FileManagerSidebar } from "./FileManagerSidebar";
import {
  getFileIconDataUrl,
  getFileManagerLocations,
  listDirectory,
  type DirectoryListing,
} from "@/lib/api/fileBrowser";
import {
  openPathWithDefaultApp,
  revealPathInFinder,
} from "@/lib/api/fileSystem";
import { startFileSystemWatch } from "@/lib/api/fileSystemWatch";

const APP_ICON_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=";

const toastMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
}));

const skillPackageInstallActionPattern = /安装/;

vi.mock("sonner", () => ({
  toast: toastMock,
}));

vi.mock("react-i18next", async () => {
  const { agentZhCNResource } = await import("@/i18n/agentResources");
  const agentZhCN = agentZhCNResource as Record<string, string>;

  return {
    useTranslation: () => ({
      i18n: {
        language: "zh-CN",
      },
      t: (key: string, options?: Record<string, unknown>) => {
        const template = agentZhCN[key] ?? key;
        return template.replace(/{{\s*([^}]+?)\s*}}/g, (_, name: string) =>
          String(options?.[name.trim()] ?? ""),
        );
      },
    }),
  };
});

vi.mock("@/lib/api/fileBrowser", () => ({
  getFileIconDataUrl: vi.fn(),
  getFileManagerLocations: vi.fn(),
  listDirectory: vi.fn(),
}));

vi.mock("@/lib/api/fileSystem", () => ({
  openPathWithDefaultApp: vi.fn(),
  revealPathInFinder: vi.fn(),
}));

vi.mock("@/lib/api/fileSystemWatch", () => ({
  startFileSystemWatch: vi.fn(),
}));

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

function createListing(path: string): DirectoryListing {
  if (path === "/Applications") {
    return {
      path,
      parentPath: null,
      entries: [
        {
          name: "Lime.app",
          path: "/Applications/Lime.app",
          isDir: true,
          size: 0,
          modifiedAt: Date.now(),
        },
      ],
      error: null,
    };
  }
  if (path === "/workspace/project") {
    return {
      path,
      parentPath: null,
      entries: [
        {
          name: "source.md",
          path: "/workspace/project/source.md",
          isDir: false,
          size: 256,
          modifiedAt: Date.now(),
          mimeType: "text/markdown",
        },
      ],
      error: null,
    };
  }
  return {
    path,
    parentPath: path === "/Users/demo" ? null : "/Users/demo",
    entries: [
      {
        name: ".config",
        path: "/Users/demo/.config",
        isDir: true,
        size: 0,
        modifiedAt: Date.now(),
      },
      {
        name: ".env",
        path: "/Users/demo/.env",
        isDir: false,
        size: 64,
        modifiedAt: Date.now(),
        mimeType: "text/plain",
      },
      {
        name: "Downloads",
        path: "/Users/demo/Downloads",
        isDir: true,
        size: 0,
        modifiedAt: Date.now(),
      },
      {
        name: "brief.txt",
        path: "/Users/demo/brief.txt",
        isDir: false,
        size: 128,
        modifiedAt: Date.now(),
        mimeType: "text/plain",
      },
      {
        name: "contract.pdf",
        path: "/Users/demo/contract.pdf",
        isDir: false,
        size: 2048,
        modifiedAt: Date.now(),
        mimeType: "application/pdf",
      },
      {
        name: "article-typesetting-master.skill",
        path: "/Users/demo/article-typesetting-master.skill",
        isDir: false,
        size: 4096,
        modifiedAt: Date.now(),
        mimeType: "application/zip",
      },
      {
        name: "article-typesetting-addon.skills",
        path: "/Users/demo/article-typesetting-addon.skills",
        isDir: false,
        size: 4096,
        modifiedAt: Date.now(),
        mimeType: "application/zip",
      },
    ],
    error: null,
  };
}

async function renderFileManagerSidebar(props?: {
  onClose?: () => void;
  onAddPathReferences?: React.ComponentProps<
    typeof FileManagerSidebar
  >["onAddPathReferences"];
  onImportAsKnowledge?: React.ComponentProps<
    typeof FileManagerSidebar
  >["onImportAsKnowledge"];
  onOpenFileInWorkspace?: React.ComponentProps<
    typeof FileManagerSidebar
  >["onOpenFileInWorkspace"];
  onInstallSkillPackage?: React.ComponentProps<
    typeof FileManagerSidebar
  >["onInstallSkillPackage"];
  initialDirectory?: React.ComponentProps<
    typeof FileManagerSidebar
  >["initialDirectory"];
}) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const onClose = props?.onClose ?? vi.fn();
  const onAddPathReferences = props?.onAddPathReferences ?? vi.fn();
  const onImportAsKnowledge = props?.onImportAsKnowledge;
  const onOpenFileInWorkspace = props?.onOpenFileInWorkspace;
  const onInstallSkillPackage = props?.onInstallSkillPackage;

  await act(async () => {
    root.render(
      <FileManagerSidebar
        onClose={onClose}
        onAddPathReferences={onAddPathReferences}
        onImportAsKnowledge={onImportAsKnowledge}
        onOpenFileInWorkspace={onOpenFileInWorkspace}
        onInstallSkillPackage={onInstallSkillPackage}
        initialDirectory={props?.initialDirectory}
      />,
    );
    await Promise.resolve();
    await Promise.resolve();
  });

  mountedRoots.push({ root, container });
  return { container, onClose, onAddPathReferences };
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  vi.mocked(getFileManagerLocations).mockResolvedValue([
    {
      id: "home",
      label: "个人",
      path: "/Users/demo",
      kind: "home",
    },
    {
      id: "downloads",
      label: "下载",
      path: "/Users/demo/Downloads",
      kind: "downloads",
    },
    {
      id: "applications",
      label: "应用程序",
      path: "/Applications",
      kind: "applications",
    },
  ]);
  vi.mocked(listDirectory).mockImplementation(async (path: string) =>
    createListing(path),
  );
  vi.mocked(getFileIconDataUrl).mockImplementation(async (path: string) =>
    path === "/Users/demo/Downloads" || path === "/Applications/Lime.app"
      ? APP_ICON_DATA_URL
      : null,
  );
  vi.mocked(openPathWithDefaultApp).mockResolvedValue(undefined);
  vi.mocked(revealPathInFinder).mockResolvedValue(undefined);
  vi.mocked(startFileSystemWatch).mockResolvedValue(
    vi.fn<() => Promise<void>>().mockResolvedValue(undefined),
  );
  window.localStorage.clear();
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

describe("FileManagerSidebar", () => {
  it("目录变化通知应刷新当前目录并在卸载时停止 watcher", async () => {
    let onChanged:
      | Parameters<typeof startFileSystemWatch>[1]
      | undefined;
    const stop = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
    vi.mocked(startFileSystemWatch).mockImplementation(
      async (_path, handler) => {
        onChanged = handler;
        return stop;
      },
    );

    const { container } = await renderFileManagerSidebar();
    expect(startFileSystemWatch).toHaveBeenCalledWith(
      "/Users/demo",
      expect.any(Function),
    );
    const listingCallsBeforeChange = vi.mocked(listDirectory).mock.calls.length;

    await act(async () => {
      onChanged?.({
        changedPaths: ["/Users/demo/brief.txt"],
        watchId: "file-manager-1",
      });
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(vi.mocked(listDirectory).mock.calls.length).toBeGreaterThan(
      listingCallsBeforeChange,
    );
    expect(container.textContent).toContain("brief.txt");

    await act(async () => {
      const mountedIndex = mountedRoots.findIndex(
        (mounted) => mounted.container === container,
      );
      const mounted = mountedRoots[mountedIndex];
      mounted?.root.unmount();
      if (mountedIndex >= 0) {
        mountedRoots.splice(mountedIndex, 1);
      }
      await Promise.resolve();
    });
    expect(stop).toHaveBeenCalledTimes(1);
  });

  it("默认隐藏点开头目录，并可通过眼睛按钮切换显示", async () => {
    const { container } = await renderFileManagerSidebar();
    const toggle = container.querySelector(
      '[data-testid="file-manager-toggle-hidden-directories"]',
    ) as HTMLButtonElement | null;

    expect(toggle?.getAttribute("aria-label")).toBe("显示隐藏目录");
    expect(toggle?.getAttribute("aria-pressed")).toBe("false");
    expect(container.textContent).not.toContain(".config");
    expect(container.textContent).toContain(".env");
    expect(container.textContent).toContain("Downloads");
    expect(container.textContent).toContain("brief.txt");

    await act(async () => {
      toggle?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(toggle?.getAttribute("aria-label")).toBe("隐藏隐藏目录");
    expect(toggle?.getAttribute("aria-pressed")).toBe("true");
    expect(container.textContent).toContain(".config");

    await act(async () => {
      toggle?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(toggle?.getAttribute("aria-pressed")).toBe("false");
    expect(container.textContent).not.toContain(".config");
  });

  it("图标读取很慢时也应先完成目录加载", async () => {
    vi.mocked(getFileIconDataUrl).mockImplementation(
      () => new Promise<string | null>(() => undefined),
    );

    const { container } = await renderFileManagerSidebar();

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(listDirectory).toHaveBeenCalledWith("/Users/demo");
    expect(container.textContent).toContain("Downloads");
    expect(container.textContent).not.toContain("加载中");
    expect(
      container.querySelector('[data-testid="file-manager-entry-native-icon"]'),
    ).toBeNull();
    expect(getFileIconDataUrl).toHaveBeenCalled();
  });

  it("应加载系统位置与目录条目，并保留安全右键菜单", async () => {
    const onAddPathReferences = vi.fn();
    const { container } = await renderFileManagerSidebar({
      onAddPathReferences,
    });

    expect(getFileManagerLocations).toHaveBeenCalled();
    expect(listDirectory).toHaveBeenCalledWith("/Users/demo");
    expect(container.textContent).toContain("个人");
    expect(container.textContent).toContain("位置");
    expect(container.textContent).toContain("名称");
    expect(container.textContent).toContain("修改日期");
    expect(container.textContent).toContain("大小");
    expect(container.textContent).toContain("Downloads");
    expect(container.textContent).not.toContain("今天");
    expect(container.textContent).not.toContain("本周");

    const downloadsEntry = Array.from(
      container.querySelectorAll('[data-testid="file-manager-entry"]'),
    ).find((entry) => entry.textContent?.includes("Downloads"));
    expect(downloadsEntry).toBeTruthy();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(
      (
        downloadsEntry?.querySelector(
          '[data-testid="file-manager-entry-native-icon"]',
        ) as HTMLImageElement | null
      )?.getAttribute("src"),
    ).toBe(APP_ICON_DATA_URL);

    await act(async () => {
      downloadsEntry?.dispatchEvent(
        new MouseEvent("contextmenu", {
          bubbles: true,
          cancelable: true,
          clientX: 24,
          clientY: 32,
        }),
      );
      await Promise.resolve();
    });

    const menu = document.querySelector(
      '[data-testid="file-manager-context-menu"]',
    );
    expect(menu?.textContent).toContain("加入对话");
    expect(menu?.textContent).toContain("显示位置");
    expect(menu?.textContent).not.toContain("删除");
    expect(menu?.textContent).not.toContain("重命名");

    const addAction = Array.from(menu?.querySelectorAll("button") ?? []).find(
      (button) => button.textContent?.includes("加入对话"),
    );
    expect(addAction).toBeTruthy();

    await act(async () => {
      addAction?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onAddPathReferences).toHaveBeenCalledWith([
      expect.objectContaining({
        path: "/Users/demo/Downloads",
        name: "Downloads",
        isDir: true,
        source: "file_manager",
      }),
    ]);
  });

  it("提供项目目录时应优先打开当前项目", async () => {
    const { container } = await renderFileManagerSidebar({
      initialDirectory: "/workspace/project",
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(listDirectory).toHaveBeenCalledWith("/workspace/project");
    expect(container.textContent).toContain("当前项目");
    expect(container.textContent).toContain("source.md");
  });

  it("顶部位置说明不应直接暴露本机完整路径", async () => {
    const { container } = await renderFileManagerSidebar();

    expect(container.textContent).toContain("个人");
    expect(container.textContent).toContain("本地位置");
    expect(container.textContent).not.toContain("/Users/demo");

    const locationHint = Array.from(container.querySelectorAll("[title]")).find(
      (element) => element.getAttribute("title") === "当前文件夹",
    );
    expect(locationHint?.getAttribute("title")).toBe("当前文件夹");
  });

  it("应支持关闭侧栏", async () => {
    const onClose = vi.fn();
    const { container } = await renderFileManagerSidebar({ onClose });

    const closeButton = container.querySelector(
      'button[aria-label="关闭文件管理器"]',
    ) as HTMLButtonElement | null;
    expect(closeButton).toBeTruthy();

    await act(async () => {
      closeButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("右键文本文件应可直接设为资料", async () => {
    const onImportAsKnowledge = vi.fn();
    const { container } = await renderFileManagerSidebar({
      onImportAsKnowledge,
    });

    const fileEntry = Array.from(
      container.querySelectorAll('[data-testid="file-manager-entry"]'),
    ).find((entry) => entry.textContent?.includes("brief.txt"));

    await act(async () => {
      fileEntry?.dispatchEvent(
        new MouseEvent("contextmenu", {
          bubbles: true,
          cancelable: true,
          clientX: 48,
          clientY: 56,
        }),
      );
      await Promise.resolve();
    });

    const menu = document.querySelector(
      '[data-testid="file-manager-context-menu"]',
    );
    expect(menu?.textContent).toContain("设为资料");

    const importAction = Array.from(
      menu?.querySelectorAll("button") ?? [],
    ).find((button) => button.textContent?.includes("设为资料"));
    expect(importAction).toBeTruthy();

    await act(async () => {
      importAction?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onImportAsKnowledge).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/Users/demo/brief.txt",
        name: "brief.txt",
        isDir: false,
        mimeType: "text/plain",
        source: "file_manager",
      }),
    );
  });

  it("普通点击文本文件应先加入对话，避免直接打开系统应用", async () => {
    const onAddPathReferences = vi.fn();
    const onImportAsKnowledge = vi.fn();
    const { container } = await renderFileManagerSidebar({
      onAddPathReferences,
      onImportAsKnowledge,
    });

    const fileEntry = Array.from(
      container.querySelectorAll('[data-testid="file-manager-entry"]'),
    ).find((entry) => entry.textContent?.includes("brief.txt"));
    expect(fileEntry?.textContent).not.toContain("加入对话");
    expect(fileEntry?.textContent).not.toContain("设为资料");
    expect(fileEntry?.querySelectorAll("button")).toHaveLength(0);

    await act(async () => {
      fileEntry?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onAddPathReferences).toHaveBeenCalledWith([
      expect.objectContaining({
        path: "/Users/demo/brief.txt",
        name: "brief.txt",
        isDir: false,
        source: "file_manager",
      }),
    ]);
    expect(openPathWithDefaultApp).not.toHaveBeenCalled();
  });

  it("普通点击 .skill 安装包应打开安装流程而不是加入对话", async () => {
    const onAddPathReferences = vi.fn();
    const onInstallSkillPackage = vi.fn();
    const { container } = await renderFileManagerSidebar({
      onAddPathReferences,
      onInstallSkillPackage,
    });

    const skillEntry = Array.from(
      container.querySelectorAll('[data-testid="file-manager-entry"]'),
    ).find((entry) =>
      entry.textContent?.includes("article-typesetting-master.skill"),
    );
    expect(skillEntry?.textContent).not.toMatch(
      skillPackageInstallActionPattern,
    );
    expect(skillEntry?.querySelectorAll("button")).toHaveLength(0);

    await act(async () => {
      skillEntry?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onInstallSkillPackage).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/Users/demo/article-typesetting-master.skill",
        name: "article-typesetting-master.skill",
        isDir: false,
      }),
    );
    expect(onAddPathReferences).not.toHaveBeenCalled();
  });

  it("右键 .skill 安装包应展示简短安装入口", async () => {
    const onInstallSkillPackage = vi.fn();
    const { container } = await renderFileManagerSidebar({
      onInstallSkillPackage,
    });

    const skillEntry = Array.from(
      container.querySelectorAll('[data-testid="file-manager-entry"]'),
    ).find((entry) =>
      entry.textContent?.includes("article-typesetting-master.skill"),
    );

    await act(async () => {
      skillEntry?.dispatchEvent(
        new MouseEvent("contextmenu", {
          bubbles: true,
          cancelable: true,
          clientX: 48,
          clientY: 56,
        }),
      );
      await Promise.resolve();
    });

    const menu = document.querySelector(
      '[data-testid="file-manager-context-menu"]',
    );
    expect(menu?.textContent).toMatch(skillPackageInstallActionPattern);

    const installAction = Array.from(
      menu?.querySelectorAll("button") ?? [],
    ).find((button) =>
      skillPackageInstallActionPattern.test(button.textContent ?? ""),
    );
    expect(installAction).toBeTruthy();

    await act(async () => {
      installAction?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onInstallSkillPackage).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/Users/demo/article-typesetting-master.skill",
      }),
    );
  });

  it("普通点击 .skills 安装包应打开安装流程而不是加入对话", async () => {
    const onAddPathReferences = vi.fn();
    const onInstallSkillPackage = vi.fn();
    const { container } = await renderFileManagerSidebar({
      onAddPathReferences,
      onInstallSkillPackage,
    });

    const skillEntry = Array.from(
      container.querySelectorAll('[data-testid="file-manager-entry"]'),
    ).find((entry) =>
      entry.textContent?.includes("article-typesetting-addon.skills"),
    );
    expect(skillEntry?.textContent).not.toMatch(
      skillPackageInstallActionPattern,
    );
    expect(skillEntry?.querySelectorAll("button")).toHaveLength(0);

    await act(async () => {
      skillEntry?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onInstallSkillPackage).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/Users/demo/article-typesetting-addon.skills",
        name: "article-typesetting-addon.skills",
        isDir: false,
      }),
    );
    expect(onAddPathReferences).not.toHaveBeenCalled();
  });

  it("右键文件应打开工作台预览，不替代普通加入对话动作", async () => {
    const onAddPathReferences = vi.fn();
    const onOpenFileInWorkspace = vi.fn();
    const { container } = await renderFileManagerSidebar({
      onAddPathReferences,
      onOpenFileInWorkspace,
    });

    const fileEntry = Array.from(
      container.querySelectorAll('[data-testid="file-manager-entry"]'),
    ).find((entry) => entry.textContent?.includes("brief.txt"));

    await act(async () => {
      fileEntry?.dispatchEvent(
        new MouseEvent("contextmenu", {
          bubbles: true,
          cancelable: true,
          clientX: 48,
          clientY: 56,
        }),
      );
      await Promise.resolve();
    });

    const menu = document.querySelector(
      '[data-testid="file-manager-context-menu"]',
    );
    const previewAction = Array.from(
      menu?.querySelectorAll("button") ?? [],
    ).find((button) => button.textContent?.includes("预览"));
    expect(previewAction).toBeTruthy();

    await act(async () => {
      previewAction?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onOpenFileInWorkspace).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/Users/demo/brief.txt",
        name: "brief.txt",
        isDir: false,
      }),
    );
    expect(onAddPathReferences).not.toHaveBeenCalled();
    expect(openPathWithDefaultApp).not.toHaveBeenCalled();
  });

  it("右键文件里的设为资料动作应直接进入资料整理", async () => {
    const onImportAsKnowledge = vi.fn();
    const { container } = await renderFileManagerSidebar({
      onImportAsKnowledge,
    });

    const fileEntry = Array.from(
      container.querySelectorAll('[data-testid="file-manager-entry"]'),
    ).find((entry) => entry.textContent?.includes("brief.txt"));

    await act(async () => {
      fileEntry?.dispatchEvent(
        new MouseEvent("contextmenu", {
          bubbles: true,
          cancelable: true,
          clientX: 48,
          clientY: 56,
        }),
      );
      await Promise.resolve();
    });

    const menu = document.querySelector(
      '[data-testid="file-manager-context-menu"]',
    );
    const importAction = Array.from(
      menu?.querySelectorAll("button") ?? [],
    ).find((button) => button.textContent?.includes("设为资料"));
    expect(importAction).toBeTruthy();

    await act(async () => {
      importAction?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onImportAsKnowledge).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/Users/demo/brief.txt",
        name: "brief.txt",
        isDir: false,
        mimeType: "text/plain",
        source: "file_manager",
      }),
    );
  });

  it("右键非文本文件不应误导为可直接整理资料", async () => {
    const onImportAsKnowledge = vi.fn();
    const { container } = await renderFileManagerSidebar({
      onImportAsKnowledge,
    });

    const fileEntry = Array.from(
      container.querySelectorAll('[data-testid="file-manager-entry"]'),
    ).find((entry) => entry.textContent?.includes("contract.pdf"));

    await act(async () => {
      fileEntry?.dispatchEvent(
        new MouseEvent("contextmenu", {
          bubbles: true,
          cancelable: true,
          clientX: 48,
          clientY: 56,
        }),
      );
      await Promise.resolve();
    });

    const menu = document.querySelector(
      '[data-testid="file-manager-context-menu"]',
    );
    expect(menu?.textContent).toContain("设为资料");
    expect(menu?.textContent).not.toContain("设为项目资料");

    const importAction = Array.from(
      menu?.querySelectorAll("button") ?? [],
    ).find((button) => button.textContent?.includes("设为资料"));
    expect(importAction).toBeTruthy();
    expect((importAction as HTMLButtonElement | undefined)?.disabled).toBe(
      true,
    );
    expect(importAction?.getAttribute("title")).toContain("转成 Markdown");

    await act(async () => {
      importAction?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onImportAsKnowledge).not.toHaveBeenCalled();
  });

  it("应用程序位置应渲染原生应用图标，侧栏使用桌面文件管理器式位置列表", async () => {
    const { container } = await renderFileManagerSidebar();
    const sidebar = container.querySelector(
      '[data-testid="file-manager-sidebar"]',
    ) as HTMLElement | null;
    const rail = container.querySelector(
      '[data-testid="file-manager-location-rail"]',
    ) as HTMLDivElement | null;
    const applicationsButton = Array.from(
      container.querySelectorAll("button"),
    ).find((button) => button.getAttribute("aria-label") === "应用程序");

    expect(sidebar?.className).toContain("w-[34vw]");
    expect(sidebar?.className).toContain("min-w-[460px]");
    expect(sidebar?.className).toContain("max-w-[720px]");
    expect(rail?.className).toContain("w-[148px]");
    expect(rail?.textContent).toContain("个人");
    expect(rail?.textContent).toContain("下载");
    expect(applicationsButton).toBeTruthy();

    await act(async () => {
      applicationsButton?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    const appEntry = container.querySelector(
      '[data-testid="file-manager-entry"][data-file-path="/Applications/Lime.app"]',
    ) as HTMLButtonElement | null;

    expect(appEntry).toBeTruthy();
    expect(appEntry?.dataset.entryKind).toBe("application");
    expect(
      appEntry
        ?.querySelector('[data-testid="file-manager-entry-icon"]')
        ?.getAttribute("data-icon-kind"),
    ).toBe("application");
    expect(
      appEntry
        ?.querySelector('[data-testid="file-manager-entry-icon"]')
        ?.getAttribute("data-icon-source"),
    ).toBe("native");
    expect(
      (
        appEntry?.querySelector(
          '[data-testid="file-manager-entry-native-icon"]',
        ) as HTMLImageElement | null
      )?.getAttribute("src"),
    ).toBe(APP_ICON_DATA_URL);
  });
});
