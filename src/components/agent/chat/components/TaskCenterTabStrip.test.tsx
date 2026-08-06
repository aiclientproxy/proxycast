import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import {
  TaskCenterTabStrip,
  type TaskCenterTabItem,
} from "./TaskCenterTabStrip";
import { conversationProjectionStore } from "../projection/conversationProjectionStore";

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  await changeLimeLocale("zh-CN");
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
  conversationProjectionStore.clearAgentUiProjectionEvents();
  vi.clearAllMocks();
});

function renderTabStrip(
  props?: Partial<React.ComponentProps<typeof TaskCenterTabStrip>>,
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  const items: TaskCenterTabItem[] = [
    {
      id: "topic-a",
      title: "任务 A",
      status: "running",
      updatedAt: new Date("2026-04-24T10:00:00.000Z"),
      isActive: true,
      hasUnread: true,
    },
    {
      id: "topic-b",
      title: "任务 B",
      status: "done",
      updatedAt: new Date("2026-04-24T09:00:00.000Z"),
      isActive: false,
      hasUnread: false,
    },
  ];

  const defaultProps: React.ComponentProps<typeof TaskCenterTabStrip> = {
    items,
    onSelectTask: vi.fn(),
    onCloseTask: vi.fn(),
    onCreateTask: vi.fn(),
  };

  act(() => {
    root.render(<TaskCenterTabStrip {...defaultProps} {...props} />);
  });

  mountedRoots.push({ root, container });
  return { container, props: { ...defaultProps, ...props } };
}

describe("TaskCenterTabStrip", () => {
  it("应渲染第二层会话 tabs 和加号入口", () => {
    const { container } = renderTabStrip();

    const strip = container.querySelector(
      '[data-testid="task-center-tab-strip"]',
    ) as HTMLElement | null;
    expect(strip).toBeTruthy();
    expect(strip?.getAttribute("role")).toBe("tablist");
    expect(strip?.className).toContain("z-10");
    expect(strip?.className).toContain("min-h-[42px]");
    expect(strip?.className).toContain(
      "bg-[color:var(--lime-chrome-tab-active-surface)]",
    );
    expect(
      strip?.style.getPropertyValue("--task-center-tab-strip-background"),
    ).toContain("--lime-chrome-stage-blend");
    expect(
      strip?.style.getPropertyValue("--task-center-tab-strip-seam"),
    ).toContain("--lime-chrome-stage-seam");
    expect(strip?.className).not.toContain("bg-[#fbfdfb]");
    expect(strip?.className).not.toContain("ml-[");
    expect(container.textContent).toContain("任务 A");
    expect(container.textContent).toContain("任务 B");
    expect(
      container.querySelector('[data-testid="task-center-tab-create-button"]'),
    ).toBeTruthy();

    const activeTab = container.querySelector(
      '[data-testid="task-center-tab-topic-a"]',
    ) as HTMLElement | null;
    const activeTabButton = activeTab?.querySelector("button[role='tab']");
    const inactiveTabButton = container.querySelector(
      '[data-testid="task-center-tab-topic-b"] button[role="tab"]',
    );
    expect(activeTab?.getAttribute("data-active")).toBe("true");
    expect(activeTabButton?.getAttribute("aria-selected")).toBe("true");
    expect(activeTabButton?.getAttribute("aria-current")).toBeNull();
    expect(inactiveTabButton?.getAttribute("aria-selected")).toBe("false");
    expect(activeTab?.className).toContain(
      "bg-[color:var(--lime-chrome-tab-hover)]",
    );
    expect(activeTab?.className).toContain(
      "border-[color:var(--lime-chrome-divider)]",
    );
    expect(
      container.querySelector('[data-testid="task-center-tab-unread-topic-a"]'),
    ).toBeTruthy();
    expect(
      container.querySelector(
        '[data-testid="task-center-tab-loading-topic-a"]',
      ),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="task-center-tab-close-topic-b"]')
        ?.className,
    ).toContain("group-hover:opacity-100");
  });

  it("点击标签应触发任务切换", () => {
    const onSelectTask = vi.fn();
    const { container } = renderTabStrip({ onSelectTask });

    act(() => {
      (
        container.querySelector(
          '[data-testid="task-center-tab-topic-b"] button[title]',
        ) as HTMLButtonElement | null
      )?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onSelectTask).toHaveBeenCalledWith("topic-b");
  });

  it("排队标签应显示排队状态且不使用运行中旋转图标", () => {
    const { container } = renderTabStrip({
      items: [
        {
          id: "topic-queued",
          title: "排队任务",
          status: "queued",
          updatedAt: new Date("2026-04-24T10:00:00.000Z"),
          isActive: true,
          hasUnread: false,
        },
      ],
    });

    const tab = container.querySelector(
      '[data-testid="task-center-tab-topic-queued"] button[title]',
    ) as HTMLButtonElement | null;

    expect(tab?.getAttribute("title")).toContain("排队中");
    expect(
      container.querySelector(
        '[data-testid="task-center-tab-loading-topic-queued"]',
      ),
    ).toBeNull();
  });

  it("关闭标签时不应触发切换", () => {
    const onSelectTask = vi.fn();
    const onCloseTask = vi.fn();
    const { container } = renderTabStrip({ onSelectTask, onCloseTask });

    act(() => {
      (
        container.querySelector(
          '[data-testid="task-center-tab-close-topic-b"]',
        ) as HTMLButtonElement | null
      )?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onCloseTask).toHaveBeenCalledWith("topic-b");
    expect(onSelectTask).not.toHaveBeenCalled();
  });

  it("重命名标签时不应触发切换", () => {
    const onSelectTask = vi.fn();
    const onRenameTask = vi.fn();
    const { container } = renderTabStrip({ onSelectTask, onRenameTask });

    act(() => {
      (
        container.querySelector(
          '[data-testid="task-center-tab-rename-topic-b"]',
        ) as HTMLButtonElement | null
      )?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onRenameTask).toHaveBeenCalledWith("topic-b");
    expect(onSelectTask).not.toHaveBeenCalled();
  });

  it("不可关闭标签不应渲染关闭按钮", () => {
    const onRenameTask = vi.fn();
    const { container } = renderTabStrip({
      onRenameTask,
      items: [
        {
          id: "new-task-home",
          title: "新对话",
          status: "draft",
          updatedAt: new Date("2026-04-24T10:00:00.000Z"),
          isActive: true,
          hasUnread: false,
          renamable: false,
          closable: false,
        },
      ],
    });

    expect(
      container.querySelector('[data-testid="task-center-tab-new-task-home"]'),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="task-center-tab-close-new-task-home"]',
      ),
    ).toBeNull();
    expect(
      container.querySelector(
        '[data-testid="task-center-tab-rename-new-task-home"]',
      ),
    ).toBeNull();
  });

  it("点击加号应触发新建对话回调", () => {
    const onCreateTask = vi.fn();
    const { container } = renderTabStrip({ onCreateTask });

    act(() => {
      (
        container.querySelector(
          '[data-testid="task-center-tab-create-button"]',
        ) as HTMLButtonElement | null
      )?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onCreateTask).toHaveBeenCalledTimes(1);
  });

  it("路径型会话标题应显示文件名并在 tooltip 保留完整路径", () => {
    const conversationPath =
      "/Users/coso/Documents/other/conversations/conv-1777047467972";
    const { container } = renderTabStrip({
      items: [
        {
          id: "topic-path",
          title: conversationPath,
          status: "done",
          updatedAt: new Date("2026-04-24T10:00:00.000Z"),
          isActive: true,
          hasUnread: false,
        },
      ],
    });

    const tab = container.querySelector(
      '[data-testid="task-center-tab-topic-path"] button[title]',
    ) as HTMLButtonElement | null;
    const closeButton = container.querySelector(
      '[data-testid="task-center-tab-close-topic-path"]',
    ) as HTMLButtonElement | null;

    expect(tab?.textContent).toContain("conv-1777047467972");
    expect(tab?.textContent).not.toContain("/Users/coso/Documents");
    expect(tab?.getAttribute("title")).toContain("conv-1777047467972");
    expect(tab?.getAttribute("title")).toContain(conversationPath);
    expect(closeButton?.getAttribute("aria-label")).toBe(
      "关闭 conv-1777047467972",
    );
  });

  it("没有打开会话时也应保留第二层 tabs 壳和加号入口", () => {
    const { container } = renderTabStrip({ items: [] });

    const strip = container.querySelector(
      '[data-testid="task-center-tab-strip"]',
    ) as HTMLElement | null;
    const createButton = container.querySelector(
      '[data-testid="task-center-tab-create-button"]',
    ) as HTMLButtonElement | null;

    expect(strip).not.toBeNull();
    expect(createButton).not.toBeNull();
    expect(container.textContent).not.toContain("新对话");
  });

  it("右侧工具区应只保留工作台入口，不再渲染旧历史按钮", () => {
    const onWorkbenchToggle = vi.fn();
    const { container } = renderTabStrip({
      showWorkbenchToggle: true,
      workbenchVisible: true,
      onWorkbenchToggle,
    });

    act(() => {
      (
        container.querySelector(
          '[data-testid="task-center-tab-workbench"]',
        ) as HTMLButtonElement | null
      )?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onWorkbenchToggle).toHaveBeenCalledTimes(1);
    expect(container.textContent).toContain("工作台");
    expect(
      container.querySelector('[data-testid="task-center-tab-history"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="task-center-tab-toolbar"]'),
    ).toBeTruthy();
  });

  it("不应在会话标签暴露 Agent UI 投影计数", () => {
    conversationProjectionStore.recordAgentUiProjectionEvents([
      {
        type: "task.changed",
        sourceType: "queue_added",
        sequence: 1,
        timestamp: "2026-04-24T10:01:00.000Z",
        sessionId: "topic-a",
        threadId: "topic-a",
        taskId: "topic-a",
        owner: "task",
        scope: "task",
        phase: "submitted",
        surface: "session_tabs",
        persistence: "ui_local",
        control: "steer",
        payload: { taskEvent: "queue_added" },
      },
      {
        type: "action.required",
        sourceType: "action_required",
        sequence: 2,
        timestamp: "2026-04-24T10:02:00.000Z",
        sessionId: "other-topic",
        threadId: "other-topic",
        actionId: "action-other",
        owner: "action",
        scope: "action_request",
        phase: "waiting",
        surface: "hitl",
        persistence: "snapshot",
        control: "approve",
      },
    ]);

    const { container } = renderTabStrip();

    const projectionBadge = container.querySelector(
      '[data-testid="task-center-tab-agentui-topic-a"]',
    ) as HTMLElement | null;
    expect(projectionBadge).toBeNull();
    expect(container.textContent).not.toContain("AgentUI");
    expect(
      container.querySelector(
        '[data-testid="task-center-tab-agentui-topic-b"]',
      ),
    ).toBeNull();
  });
});
