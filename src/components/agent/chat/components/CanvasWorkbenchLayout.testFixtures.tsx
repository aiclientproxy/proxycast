import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, vi } from "vitest";
import type { Artifact } from "@/lib/artifact/types";
import type { TaskFile } from "./TaskFiles";
import { CanvasWorkbenchLayout } from "./CanvasWorkbenchLayout";

type MockResizeObserverCallback = (
  entries: Array<{
    target: Element;
    contentRect: {
      width: number;
      height: number;
    };
  }>,
  observer: unknown,
) => void;

const hoisted = vi.hoisted(() => ({
  mockListDirectory: vi.fn(),
  mockListProjectGitCommits: vi.fn(),
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
  mockToast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
  },
  resizeObserverState: {
    width: 1280,
    observers: [] as Array<{
      callback: MockResizeObserverCallback;
      target: Element | null;
    }>,
  },
}));

export const mockListDirectory = hoisted.mockListDirectory;
export const mockListProjectGitCommits = hoisted.mockListProjectGitCommits;
export const mockReadProjectGitDiff = hoisted.mockReadProjectGitDiff;
export const mockReadProjectGitStatus = hoisted.mockReadProjectGitStatus;
export const mockTerminateCommandExec = hoisted.mockTerminateCommandExec;
export const mockSubscribeCommandExecOutput =
  hoisted.mockSubscribeCommandExecOutput;
export const mockResizeCommandExec = hoisted.mockResizeCommandExec;
export const mockExecCommand = hoisted.mockExecCommand;
export const mockWriteCommandExec = hoisted.mockWriteCommandExec;
export const mockFitAddonFit = hoisted.mockFitAddonFit;
export const mockXtermOnDataHandlers = hoisted.mockXtermOnDataHandlers;
export const mockXtermLoadAddon = hoisted.mockXtermLoadAddon;
export const mockXtermTerminalOptions = hoisted.mockXtermTerminalOptions;
export const mockToast = hoisted.mockToast;
const resizeObserverState = hoisted.resizeObserverState;

vi.mock("sonner", () => ({
  toast: hoisted.mockToast,
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    i18n: {
      language: "zh-CN",
    },
    t: (key: string, options?: Record<string, unknown>) => {
      const translations: Record<string, string> = {
        "agentChat.canvasWorkbench.close": "关闭画布工作台",
        "agentChat.canvasWorkbench.kind.artifact": "产物",
        "agentChat.canvasWorkbench.kind.currentDraft": "当前文稿",
        "agentChat.canvasWorkbench.kind.currentVersion": "当前",
        "agentChat.canvasWorkbench.kind.defaultDraft": "主稿",
        "agentChat.canvasWorkbench.kind.taskDocument": "文档",
        "agentChat.canvasWorkbench.kind.taskFile": "任务文件",
        "agentChat.canvasWorkbench.kind.version": "版本",
        "agentChat.canvasWorkbench.kind.versionTitle": `文稿版本 ${String(
          options?.count ?? 0,
        )}`,
        "agentChat.canvasWorkbench.kind.workspaceFile": "文件",
        "agentChat.canvasWorkbench.tabs.files": "文件",
        "agentChat.canvasWorkbench.tabs.generated": "生成",
        "agentChat.canvasWorkbench.tabs.tasks": "任务",
        "agentChat.canvasWorkbench.tabs.sessionMain": "结果",
        "agentChat.canvasWorkbench.tabs.newTabAria": "打开工作台切换菜单",
        "agentChat.canvasWorkbench.tabs.switchAria": `切换画布标签-${String(
          options?.label ?? "",
        )}`,
        "agentChat.canvasWorkbench.tabs.newToolAria": `新建工作台标签-${String(
          options?.label ?? "",
        )}`,
        "agentChat.canvasWorkbench.tabs.closeAria": `关闭工作台标签-${String(
          options?.label ?? "",
        )}`,
        "agentChat.canvasWorkbench.newTabs.terminal": "终端",
        "agentChat.canvasWorkbench.newTabs.files": "文件",
        "agentChat.canvasWorkbench.newTabs.terminalTab": "终端",
        "agentChat.canvasWorkbench.newTabs.filesTab": "打开文件",
        "agentChat.canvasWorkbench.projectFiles.treeTitle": "项目文件",
        "agentChat.canvasWorkbench.projectFiles.empty":
          "当前没有绑定可浏览的项目目录。",
        "agentChat.canvasWorkbench.projectFiles.unavailable":
          "当前项目目录不可用，暂时无法浏览文件。",
        "agentChat.canvasWorkbench.projectFiles.openTitle": "打开文件",
        "agentChat.canvasWorkbench.projectFiles.openHint":
          "从右侧项目目录树中选择文件进行预览。",
        "agentChat.canvasWorkbench.projectFiles.resizeTree": "调整项目文件宽度",
        "agentChat.canvasWorkbench.title.fallback": "工作台",
        "agentChat.canvasWorkbench.tools.columns": "切换列布局",
        "agentChat.canvasWorkbench.tools.editor": "编辑器",
        "agentChat.canvasWorkbench.tools.more": "更多工具",
        "agentChat.canvasWorkbench.tools.outline": "大纲",
        "agentChat.canvasWorkbench.tools.search": "搜索文件",
        "agentChat.canvasWorkbench.window.fullscreen": "进入全屏",
        "agentChat.canvasWorkbench.window.minimize": "最小化",
        "agentChat.canvasWorkbench.window.sidebar": "显示/隐藏侧边栏 ⌥⌘B",
        "agentChat.canvasWorkbench.workspaceFile.binaryUnsupported":
          "该文件为二进制内容，暂不支持画布文本预览。",
        "agentChat.canvasWorkbench.workspaceFile.readFailed": "读取文件失败",
        "agentChat.canvasWorkbench.clipboard.unsupported":
          "当前环境不支持剪贴板写入",
        "agentChat.canvasWorkbench.clipboard.copied": "已复制路径",
        "agentChat.canvasWorkbench.clipboard.copyFailed": "复制路径失败",
        "agentChat.canvasWorkbench.workspace.loadFailed": `读取目录失败：${String(
          options?.message ?? "",
        )}`,
        "agentChat.canvasWorkbench.workspace.expandDirectoryAria": `展开目录-${String(
          options?.name ?? "",
        )}`,
        "agentChat.canvasWorkbench.workspace.collapseDirectoryAria": `折叠目录-${String(
          options?.name ?? "",
        )}`,
        "agentChat.canvasWorkbench.workspace.selectFileAria": `选择工作区文件-${String(
          options?.name ?? "",
        )}`,
        "agentChat.canvasWorkbench.workspace.unavailable":
          "当前工作区路径不可用，暂时无法浏览项目文件。",
        "agentChat.canvasWorkbench.workspace.empty":
          "当前会话没有绑定可浏览的工作区目录。",
        "agentChat.canvasWorkbench.workspace.resultDir": "结果目录",
        "agentChat.canvasWorkbench.workspace.projectDir": "项目目录",
        "agentChat.canvasWorkbench.workspace.loading": "正在加载目录...",
        "agentChat.canvasWorkbench.workspace.emptyDirectory": "暂无目录内容。",
        "agentChat.canvasWorkbench.team.empty": "当前没有可展示的生成结果。",
        "agentChat.fileManager.column.name": "名称",
        "agentChat.fileManager.column.modified": "修改日期",
        "agentChat.fileManager.column.size": "大小",
        "agentChat.canvasWorkbench.actions.copyPath": "复制路径",
        "agentChat.canvasWorkbench.actions.revealPath": "显示位置",
        "agentChat.canvasWorkbench.actions.openPath": "打开",
        "agentChat.canvasWorkbench.actions.download": "下载",
        "agentChat.canvasWorkbench.coding.tabs.files": "文件",
        "agentChat.canvasWorkbench.coding.tabs.changes": "审查",
        "agentChat.canvasWorkbench.coding.tabs.markdown": "Markdown",
        "agentChat.canvasWorkbench.coding.tabs.html": "HTML",
        "agentChat.canvasWorkbench.coding.tabs.code": "Code",
        "agentChat.canvasWorkbench.coding.tabs.outputs": "输出",
        "agentChat.canvasWorkbench.coding.tabs.logs": "日志",
        "agentChat.canvasWorkbench.coding.preview.htmlBadge": "HTML",
        "agentChat.canvasWorkbench.coding.preview.mode.empty":
          "还没有可预览的文件内容。",
        "agentChat.canvasWorkbench.coding.preview.mode.loading":
          "正在读取文件预览...",
        "agentChat.canvasWorkbench.coding.preview.mode.htmlTitle": "HTML 预览",
        "agentChat.canvasWorkbench.coding.preview.mode.markdownAria":
          "切换到 Markdown 预览",
        "agentChat.canvasWorkbench.coding.preview.mode.htmlAria":
          "切换到 HTML 预览",
        "agentChat.canvasWorkbench.coding.preview.mode.codeAria":
          "切换到 Code 预览",
        "agentChat.canvasWorkbench.coding.changes.empty":
          "还没有可对比的文件变更。",
        "agentChat.canvasWorkbench.coding.changes.afterLabel": "当前",
        "agentChat.canvasWorkbench.coding.changes.baseConversation": "上轮对话",
        "agentChat.canvasWorkbench.coding.changes.base.branch": "分支",
        "agentChat.canvasWorkbench.coding.changes.base.commit": "提交",
        "agentChat.canvasWorkbench.coding.changes.base.emptyCommits":
          "分支上暂无提交记录。",
        "agentChat.canvasWorkbench.coding.changes.base.loadingCommits":
          "正在读取提交...",
        "agentChat.canvasWorkbench.coding.changes.base.previousConversation":
          "上轮对话",
        "agentChat.canvasWorkbench.coding.changes.base.selectorAria":
          "选择审查基准",
        "agentChat.canvasWorkbench.coding.changes.base.staged": "已暂存",
        "agentChat.canvasWorkbench.coding.changes.base.untitledCommit":
          "未命名提交",
        "agentChat.canvasWorkbench.coding.changes.base.unstaged": "未暂存",
        "agentChat.canvasWorkbench.coding.changes.beforeLabel": "上一版",
        "agentChat.canvasWorkbench.coding.changes.filterEmpty":
          "没有匹配的文件。",
        "agentChat.canvasWorkbench.coding.changes.filterFiles": "筛选文件...",
        "agentChat.canvasWorkbench.coding.changes.hideFiles": "隐藏文件",
        "agentChat.canvasWorkbench.coding.changes.noBaseline":
          "当前文件还没有上一版本可对比。",
        "agentChat.canvasWorkbench.coding.changes.title": "当前文件变更",
        "agentChat.canvasWorkbench.coding.changes.badge": `变更 ${String(
          options?.count ?? 0,
        )}`,
        "agentChat.canvasWorkbench.coding.changes.queueTitle": "本轮文件变更",
        "agentChat.canvasWorkbench.coding.changes.queueSummary": `${String(
          options?.count ?? 0,
        )} 个文件，${String(options?.pending ?? 0)} 个仍在写入`,
        "agentChat.canvasWorkbench.coding.changes.checkpointBadge": `快照 ${String(
          options?.count ?? 0,
        )}`,
        "agentChat.canvasWorkbench.coding.changes.showFiles": "显示文件",
        "agentChat.canvasWorkbench.coding.changes.source": `来源：${String(
          options?.source ?? "",
        )}`,
        "agentChat.canvasWorkbench.coding.changes.detailTitle": "当前文件",
        "agentChat.canvasWorkbench.coding.changes.latestCheckpoint": `最近快照：${String(
          options?.path ?? "",
        )}`,
        "agentChat.canvasWorkbench.coding.changes.noDiffBadge": "暂无 diff",
        "agentChat.canvasWorkbench.coding.changes.noDiff":
          "当前变更只有文件摘要，暂时没有上一版 diff。",
        "agentChat.canvasWorkbench.coding.changes.status.completed": "已写入",
        "agentChat.canvasWorkbench.coding.changes.notGitRepository":
          "当前目录不是 Git 仓库，无法读取文件变更。",
        "agentChat.canvasWorkbench.coding.changes.status.inProgress": "写入中",
        "agentChat.canvasWorkbench.coding.changes.status.failed": "失败",
        "agentChat.canvasWorkbench.coding.changes.more": "更多审查操作",
        "agentChat.canvasWorkbench.coding.changes.resizeFilesPanel":
          "调整文件列表宽度",
        "agentChat.canvasWorkbench.coding.changes.refresh": "刷新",
        "agentChat.canvasWorkbench.coding.changes.reviewTitle": "审查",
        "agentChat.canvasWorkbench.coding.changes.menu.hideWhitespace":
          "隐藏空白字符",
        "agentChat.canvasWorkbench.coding.changes.menu.showWhitespace":
          "显示空白字符",
        "agentChat.canvasWorkbench.coding.changes.menu.enableWrap":
          "启用自动换行",
        "agentChat.canvasWorkbench.coding.changes.menu.disableWrap":
          "关闭自动换行",
        "agentChat.canvasWorkbench.coding.changes.menu.loadFullFile":
          "加载完整文件",
        "agentChat.canvasWorkbench.coding.changes.menu.unloadFullFile":
          "不加载完整文件",
        "agentChat.canvasWorkbench.coding.changes.menu.enableRichPreview":
          "启用富文本预览",
        "agentChat.canvasWorkbench.coding.changes.menu.disableRichPreview":
          "关闭富文本预览",
        "agentChat.canvasWorkbench.coding.changes.menu.enableTextDiff":
          "启用文字差异",
        "agentChat.canvasWorkbench.coding.changes.menu.disableTextDiff":
          "关闭文字差异",
        "agentChat.canvasWorkbench.coding.changes.menu.copyGitApply":
          "复制 git apply 命令",
        "agentChat.canvasWorkbench.coding.changes.menu.enableAutoExecute":
          "启用自动执行",
        "agentChat.canvasWorkbench.coding.changes.menu.collapseContext":
          "折叠全部差异",
        "agentChat.canvasWorkbench.coding.changes.menu.expandContext":
          "展开未变更上下文",
        "agentChat.canvasWorkbench.coding.changes.switchToInlineDiff":
          "切换到行内差异视图",
        "agentChat.canvasWorkbench.coding.changes.switchToSplitDiff":
          "切换到拆分差异视图",
        "agentChat.canvasWorkbench.coding.changes.omittedLines": `已隐藏 ${String(
          options?.count ?? 0,
        )} 行未变更内容`,
        "agentChat.canvasWorkbench.coding.changes.kind.added": "新增",
        "agentChat.canvasWorkbench.coding.changes.kind.modified": "修改",
        "agentChat.canvasWorkbench.coding.changes.kind.deleted": "删除",
        "agentChat.canvasWorkbench.coding.changes.kind.renamed": "重命名",
        "agentChat.canvasWorkbench.coding.changes.kind.copied": "复制",
        "agentChat.canvasWorkbench.coding.changes.kind.unknown": "变更",
        "agentChat.canvasWorkbench.coding.changes.kindShort.added": "A",
        "agentChat.canvasWorkbench.coding.changes.kindShort.modified": "M",
        "agentChat.canvasWorkbench.coding.changes.kindShort.deleted": "D",
        "agentChat.canvasWorkbench.coding.changes.kindShort.renamed": "R",
        "agentChat.canvasWorkbench.coding.changes.kindShort.copied": "C",
        "agentChat.canvasWorkbench.coding.changes.kindShort.unknown": "?",
        "agentChat.canvasWorkbench.coding.changes.toast.fullFileLoadFailed":
          "加载完整文件失败",
        "agentChat.canvasWorkbench.coding.changes.toast.gitApplyCopied":
          "已复制 git apply 命令",
        "agentChat.canvasWorkbench.coding.changes.toast.gitApplyCopyFailed":
          "复制 git apply 命令失败",
        "agentChat.canvasWorkbench.coding.changes.toast.missingWorkspaceRoot":
          "缺少工作区路径",
        "agentChat.canvasWorkbench.coding.changes.toast.noPatch":
          "没有可复制的 patch",
        "agentChat.canvasWorkbench.coding.changes.toast.refreshed":
          "已刷新变更",
        "agentChat.canvasWorkbench.coding.changes.toast.refreshFailed":
          "刷新变更失败",
        "agentChat.canvasWorkbench.coding.changes.submit": "提交",
        "agentChat.canvasWorkbench.coding.outputs.empty":
          "本轮还没有可展示的输出。",
        "agentChat.canvasWorkbench.coding.logs.empty":
          "本轮还没有可展示的日志。",
      };
      return translations[key] ?? key;
    },
  }),
}));

vi.mock("@/lib/api/fileBrowser", () => ({
  listDirectory: hoisted.mockListDirectory,
}));

vi.mock("@/lib/api/projectGit", () => ({
  listProjectGitCommits: hoisted.mockListProjectGitCommits,
  readProjectGitDiff: hoisted.mockReadProjectGitDiff,
  readProjectGitStatus: hoisted.mockReadProjectGitStatus,
}));

vi.mock("@/lib/api/commandExec", () => ({
  terminateCommandExec: hoisted.mockTerminateCommandExec,
  subscribeCommandExecOutput: hoisted.mockSubscribeCommandExecOutput,
  resizeCommandExec: hoisted.mockResizeCommandExec,
  execCommand: hoisted.mockExecCommand,
  writeCommandExec: hoisted.mockWriteCommandExec,
}));

vi.mock("@xterm/xterm/css/xterm.css", () => ({}));

vi.mock("@xterm/xterm", () => ({
  Terminal: vi.fn().mockImplementation((options: Record<string, unknown>) => {
    hoisted.mockXtermTerminalOptions.push(options);
    return {
      cols: 120,
      rows: 14,
      dispose: vi.fn(),
      focus: vi.fn(),
      loadAddon: hoisted.mockXtermLoadAddon,
      onData: vi.fn((handler: (data: string) => void) => {
        hoisted.mockXtermOnDataHandlers.push(handler);
        return { dispose: hoisted.mockXtermDisposeInput };
      }),
      open: vi.fn(),
      write: vi.fn(),
      writeln: vi.fn(),
    };
  }),
}));

vi.mock("@xterm/addon-fit", () => ({
  FitAddon: vi.fn().mockImplementation(() => ({
    fit: hoisted.mockFitAddonFit,
  })),
}));

export interface MountedHarness {
  container: HTMLDivElement;
  root: Root;
  rerender: (props: CanvasWorkbenchLayoutProps) => void;
}

export type CanvasWorkbenchLayoutProps = React.ComponentProps<
  typeof CanvasWorkbenchLayout
>;

const mountedRoots: MountedHarness[] = [];

export function createArtifact(
  id: string,
  filePath: string,
  content: string,
  updatedAt: number,
): Artifact {
  return {
    id,
    type: "document",
    title: filePath.split("/").pop() || filePath,
    content,
    status: "complete",
    meta: {
      filePath,
      filename: filePath.split("/").pop() || filePath,
      previewText: content,
    },
    position: { start: 0, end: content.length },
    createdAt: updatedAt - 100,
    updatedAt,
  };
}

export function createTaskFile(
  id: string,
  name: string,
  content: string,
  updatedAt: number,
): TaskFile {
  return {
    id,
    name,
    type: "document",
    content,
    version: 1,
    createdAt: updatedAt - 100,
    updatedAt,
  };
}

export function mountHarness(
  props: CanvasWorkbenchLayoutProps,
): MountedHarness {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let currentProps = props;

  act(() => {
    root.render(<CanvasWorkbenchLayout {...currentProps} />);
  });

  const harness: MountedHarness = {
    container,
    root,
    rerender: (nextProps) => {
      currentProps = nextProps;
      act(() => {
        root.render(<CanvasWorkbenchLayout {...currentProps} />);
      });
    },
  };

  mountedRoots.push(harness);
  return harness;
}

export function mount(props: CanvasWorkbenchLayoutProps): HTMLDivElement {
  return mountHarness(props).container;
}

export async function flushEffects(times = 6) {
  for (let i = 0; i < times; i += 1) {
    await act(async () => {
      await Promise.resolve();
    });
  }
}

export async function resizeWorkbench(width: number, height = 720) {
  resizeObserverState.width = width;
  await act(async () => {
    resizeObserverState.observers.forEach((observer) => {
      if (!observer.target) {
        return;
      }
      observer.callback(
        [
          {
            target: observer.target,
            contentRect: {
              width,
              height,
            },
          },
        ],
        {},
      );
    });
    await Promise.resolve();
  });
}

export function clickByAriaLabel(container: HTMLElement, ariaLabel: string) {
  const element = container.querySelector(
    `[aria-label="${ariaLabel}"]`,
  ) as HTMLElement | null;
  if (!element) {
    throw new Error(`未找到元素: ${ariaLabel}`);
  }

  act(() => {
    element.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
}

export function clickWorkbenchTab(container: HTMLElement, label: string) {
  const ariaLabel = `切换画布标签-${label}`;
  const directElement = container.querySelector(
    `[aria-label="${ariaLabel}"]`,
  ) as HTMLElement | null;
  if (
    directElement &&
    !directElement.closest('[data-testid="canvas-workbench-tab-menu"]')
  ) {
    act(() => {
      directElement.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    return;
  }

  clickByAriaLabel(container, "打开工作台切换菜单");
  const menuElement = container.querySelector(
    `[data-testid="canvas-workbench-tab-menu"] [aria-label="${ariaLabel}"]`,
  ) as HTMLElement | null;
  if (!menuElement) {
    throw new Error(`未找到工作台标签: ${label}`);
  }

  act(() => {
    menuElement.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
}

export function clickPreviewMode(container: HTMLElement, label: string) {
  const ariaLabel =
    label === "Markdown"
      ? "切换到 Markdown 预览"
      : label === "HTML"
        ? "切换到 HTML 预览"
        : "切换到 Code 预览";
  clickByAriaLabel(container, ariaLabel);
}

export function expectWorkbenchTabInMenu(
  container: HTMLElement,
  label: string,
) {
  clickByAriaLabel(container, "打开工作台切换菜单");
  const menuElement = container.querySelector(
    `[data-testid="canvas-workbench-tab-menu"] [aria-label="切换画布标签-${label}"]`,
  );
  if (!menuElement) {
    throw new Error(`未找到工作台菜单标签: ${label}`);
  }
  clickByAriaLabel(container, "打开工作台切换菜单");
}

export function clickNewWorkbenchTool(container: HTMLElement, label: string) {
  clickByAriaLabel(container, "打开工作台切换菜单");
  const menuElement = container.querySelector(
    `[data-testid="canvas-workbench-tab-menu"] [aria-label="新建工作台标签-${label}"]`,
  ) as HTMLElement | null;
  if (!menuElement) {
    throw new Error(`未找到新增工作台工具: ${label}`);
  }

  act(() => {
    menuElement.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
}

export function expectNewWorkbenchToolInMenu(
  container: HTMLElement,
  label: string,
) {
  clickByAriaLabel(container, "打开工作台切换菜单");
  const menuElement = container.querySelector(
    `[data-testid="canvas-workbench-tab-menu"] [aria-label="新建工作台标签-${label}"]`,
  );
  if (!menuElement) {
    throw new Error(`未找到新增工作台工具: ${label}`);
  }
  clickByAriaLabel(container, "打开工作台切换菜单");
}

export function expectWorkbenchTabNotInNewMenu(
  container: HTMLElement,
  label: string,
) {
  clickByAriaLabel(container, "打开工作台切换菜单");
  const menuElement = container.querySelector(
    `[data-testid="canvas-workbench-tab-menu"] [aria-label="切换画布标签-${label}"]`,
  );
  if (menuElement) {
    throw new Error(`新增菜单不应包含旧工作台标签: ${label}`);
  }
  clickByAriaLabel(container, "打开工作台切换菜单");
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  vi.clearAllMocks();
  resizeObserverState.width = 1280;
  resizeObserverState.observers = [];
  mockReadProjectGitDiff.mockResolvedValue({
    rootPath: "/workspace",
    repositoryRoot: "/workspace",
    hasGitRepository: true,
    currentRef: "main",
    comparisonBaseRef: "origin/main",
    patch:
      "diff --git a/src/App.tsx b/src/App.tsx\n--- a/src/App.tsx\n+++ b/src/App.tsx\n@@ -0,0 +1 @@\n+export function App() {}",
    uncommittedFileCount: 1,
  });
  mockReadProjectGitStatus.mockResolvedValue({
    rootPath: "/workspace",
    repositoryRoot: "/workspace",
    hasGitRepository: true,
    currentBranch: "main",
    branches: ["main", "origin/main"],
    uncommittedFileCount: 1,
  });
  mockListProjectGitCommits.mockResolvedValue({
    rootPath: "/workspace",
    repositoryRoot: "/workspace",
    hasGitRepository: true,
    commits: [
      {
        sha: "abc1234567890",
        shortSha: "abc1234",
        subject: "整理右侧审查面板",
        authorName: "Test User",
        authorEmail: "test@example.com",
        committedAt: "2026-06-14T10:00:00Z",
      },
    ],
  });
  mockTerminateCommandExec.mockResolvedValue({});
  mockSubscribeCommandExecOutput.mockReturnValue(vi.fn());
  mockResizeCommandExec.mockResolvedValue({});
  mockExecCommand.mockResolvedValue({ exitCode: 0, stdout: "", stderr: "" });
  mockWriteCommandExec.mockResolvedValue({});
  mockXtermOnDataHandlers.length = 0;
  mockXtermTerminalOptions.length = 0;

  vi.stubGlobal(
    "ResizeObserver",
    class ResizeObserver {
      private callback: MockResizeObserverCallback;
      private target: Element | null = null;

      constructor(callback: MockResizeObserverCallback) {
        this.callback = callback;
      }

      observe = (target: Element) => {
        this.target = target;
        resizeObserverState.observers.push({
          callback: this.callback,
          target,
        });
        this.callback(
          [
            {
              target,
              contentRect: {
                width: resizeObserverState.width,
                height: 720,
              },
            },
          ],
          this,
        );
      };

      unobserve = () => {};

      disconnect = () => {
        resizeObserverState.observers = resizeObserverState.observers.filter(
          (observer) =>
            observer.callback !== this.callback ||
            observer.target !== this.target,
        );
      };
    },
  );

  mockListDirectory.mockImplementation(async (path: string) => {
    if (path === "/workspace") {
      return {
        path,
        parentPath: null,
        error: null,
        entries: [
          {
            name: ".lime",
            path: "/workspace/.lime",
            isDir: true,
            size: 0,
            modifiedAt: 100,
          },
          {
            name: "exports",
            path: "/workspace/exports",
            isDir: true,
            size: 0,
            modifiedAt: 100,
          },
          {
            name: "output",
            path: "/workspace/output",
            isDir: true,
            size: 0,
            modifiedAt: 100,
          },
          {
            name: ".DS_Store",
            path: "/workspace/.DS_Store",
            isDir: false,
            size: 10,
            modifiedAt: 100,
          },
          {
            name: "output_image.jpg",
            path: "/workspace/output_image.jpg",
            isDir: false,
            size: 512,
            modifiedAt: 100,
          },
          {
            name: "README.md",
            path: "/workspace/README.md",
            isDir: false,
            size: 128,
            modifiedAt: 100,
          },
          {
            name: "src",
            path: "/workspace/src",
            isDir: true,
            size: 0,
            modifiedAt: 100,
          },
        ],
      };
    }

    if (path === "/workspace/src") {
      return {
        path,
        parentPath: "/workspace",
        error: null,
        entries: [
          {
            name: "binary.dat",
            path: "/workspace/src/binary.dat",
            isDir: false,
            size: 2048,
            modifiedAt: 100,
          },
        ],
      };
    }

    if (path === "/workspace/exports/x-article-export/latest") {
      return {
        path,
        parentPath: "/workspace/exports/x-article-export",
        error: null,
        entries: [
          {
            name: "manifest.json",
            path: "/workspace/exports/x-article-export/latest/manifest.json",
            isDir: false,
            size: 1024,
            modifiedAt: 100,
          },
          {
            name: "images",
            path: "/workspace/exports/x-article-export/latest/images",
            isDir: true,
            size: 0,
            modifiedAt: 100,
          },
          {
            name: "skills",
            path: "/workspace/exports/x-article-export/latest/skills",
            isDir: true,
            size: 0,
            modifiedAt: 100,
          },
          {
            name: "Agents.md",
            path: "/workspace/exports/x-article-export/latest/Agents.md",
            isDir: false,
            size: 256,
            modifiedAt: 100,
          },
          {
            name: "index.md",
            path: "/workspace/exports/x-article-export/latest/index.md",
            isDir: false,
            size: 2048,
            modifiedAt: 100,
          },
        ],
      };
    }

    return {
      path,
      parentPath: "/workspace",
      error: null,
      entries: [],
    };
  });

  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: {
      writeText: vi.fn(async () => undefined),
    },
  });

  Object.defineProperty(globalThis.URL, "createObjectURL", {
    configurable: true,
    value: vi.fn(() => "blob:canvas-workbench"),
  });
  Object.defineProperty(globalThis.URL, "revokeObjectURL", {
    configurable: true,
    value: vi.fn(),
  });

  HTMLAnchorElement.prototype.click = vi.fn();
});

afterEach(() => {
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
});
