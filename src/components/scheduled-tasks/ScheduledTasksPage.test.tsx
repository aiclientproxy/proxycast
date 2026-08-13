import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import type {
  ScheduledTask,
  ScheduledTaskRunSummary,
  ScheduledTaskSummary,
} from "@/lib/api/scheduledTasks";
import type { Page, PageParams } from "@/types/page";
import { ScheduledTasksPage } from "./ScheduledTasksPage";

const {
  mockList,
  mockRead,
  mockCreate,
  mockUpdate,
  mockRemove,
  mockSetEnabled,
  mockStartRun,
  mockListRuns,
  mockPreviewSchedule,
  mockToastError,
  mockToastSuccess,
} = vi.hoisted(() => ({
  mockList: vi.fn(),
  mockRead: vi.fn(),
  mockCreate: vi.fn(),
  mockUpdate: vi.fn(),
  mockRemove: vi.fn(),
  mockSetEnabled: vi.fn(),
  mockStartRun: vi.fn(),
  mockListRuns: vi.fn(),
  mockPreviewSchedule: vi.fn(),
  mockToastError: vi.fn(),
  mockToastSuccess: vi.fn(),
}));

vi.mock("@/lib/api/scheduledTasks", async () => {
  const actual = await vi.importActual<
    typeof import("@/lib/api/scheduledTasks")
  >("@/lib/api/scheduledTasks");

  return {
    ...actual,
    scheduledTasksApi: {
      list: mockList,
      read: mockRead,
      create: mockCreate,
      update: mockUpdate,
      remove: mockRemove,
      setEnabled: mockSetEnabled,
      startRun: mockStartRun,
      listRuns: mockListRuns,
      previewSchedule: mockPreviewSchedule,
    },
  };
});

vi.mock("sonner", () => ({
  toast: {
    error: mockToastError,
    success: mockToastSuccess,
  },
}));

interface MountedPage {
  container: HTMLDivElement;
  root: Root;
}

const mountedPages: MountedPage[] = [];

function sampleTask(overrides: Partial<ScheduledTask> = {}): ScheduledTask {
  return {
    id: "task-daily",
    title: "每日项目简报",
    prompt: "整理项目进展、阻塞项和下一步行动。",
    enabled: true,
    schedule: {
      type: "weekdays",
      time: "08:30",
      timezone: "Asia/Shanghai",
    },
    execution: {
      threadMode: "new_thread",
      projectId: "project-alpha",
      cwd: "/tmp/project-alpha",
    },
    notificationPolicy: "failures",
    overlapPolicy: "skip_if_running",
    nextRunAt: "2026-08-14T00:30:00Z",
    createdAt: "2026-08-13T00:00:00Z",
    updatedAt: "2026-08-13T01:00:00Z",
    ...overrides,
  };
}

function sampleSummary(task: ScheduledTask): ScheduledTaskSummary {
  return {
    id: task.id,
    title: task.title,
    enabled: task.enabled,
    attention: Boolean(task.lastRunSummary?.error),
    schedule: task.schedule,
    nextRunAt: task.nextRunAt,
    lastRun: task.lastRunSummary,
  };
}

function sampleRun(
  overrides: Partial<ScheduledTaskRunSummary> = {},
): ScheduledTaskRunSummary {
  return {
    id: "run-1",
    taskId: "task-daily",
    status: "completed",
    scheduledFor: "2026-08-13T00:30:00Z",
    startedAt: "2026-08-13T00:30:02Z",
    finishedAt: "2026-08-13T00:30:12Z",
    sessionId: "session-scheduled-run",
    threadId: "thread-scheduled-run",
    turnId: "turn-scheduled-run",
    ...overrides,
  };
}

function renderPage(options?: {
  onNavigate?: (page: Page, params?: PageParams) => void;
}): HTMLDivElement {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(<ScheduledTasksPage onNavigate={options?.onNavigate} />);
  });
  mountedPages.push({ container, root });
  return container;
}

async function flushEffects(times = 6): Promise<void> {
  for (let index = 0; index < times; index += 1) {
    await act(async () => {
      await Promise.resolve();
    });
  }
}

function findButton(container: HTMLElement, label: string): HTMLButtonElement {
  const buttons = Array.from(container.querySelectorAll("button"));
  const button =
    buttons.find((candidate) => candidate.textContent?.trim() === label) ??
    buttons.find((candidate) => candidate.textContent?.includes(label));
  expect(button).toBeTruthy();
  return button as HTMLButtonElement;
}

async function click(element: HTMLElement): Promise<void> {
  await act(async () => {
    element.click();
    await Promise.resolve();
  });
  await flushEffects();
}

describe("ScheduledTasksPage", () => {
  beforeEach(async () => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    await changeLimeLocale("zh-CN");
    vi.clearAllMocks();

    const task = sampleTask();
    mockList.mockResolvedValue({
      items: [sampleSummary(task)],
      nextCursor: null,
    });
    mockRead.mockResolvedValue(task);
    mockListRuns.mockResolvedValue([sampleRun()]);
    mockSetEnabled.mockResolvedValue(sampleTask({ enabled: false }));
    mockRemove.mockResolvedValue(true);
    mockStartRun.mockResolvedValue(sampleRun({ status: "running" }));
    mockPreviewSchedule.mockResolvedValue({ nextRunAt: [], warnings: [] });
  });

  afterEach(() => {
    for (const mounted of mountedPages.splice(0)) {
      act(() => mounted.root.unmount());
      mounted.container.remove();
    }
  });

  it("空态创建菜单可进入真实 Agent 预填或手动编辑器", async () => {
    mockList.mockResolvedValue({ items: [], nextCursor: null });
    const onNavigate = vi.fn<(page: Page, params?: PageParams) => void>();
    const container = renderPage({ onNavigate });
    await flushEffects();

    expect(container.textContent).toContain("还没有已安排任务");
    expect(container.textContent).toContain("选择任务或创建新任务");

    await click(findButton(container, "创建任务"));
    expect(container.textContent).toContain("使用 Lime 创建");
    expect(container.textContent).toContain("手动设置");

    const createWithLime = container.querySelector<HTMLElement>(
      '[role="menuitem"]',
    );
    expect(createWithLime?.textContent).toContain("使用 Lime 创建");
    await click(createWithLime as HTMLElement);
    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        agentEntry: "claw",
        autoRunInitialPromptOnMount: false,
        initialSessionName: "创建已安排任务",
      }),
    );

    await click(findButton(container, "创建任务"));
    const manualItem = Array.from(
      container.querySelectorAll<HTMLElement>('[role="menuitem"]'),
    ).find((item) => item.textContent?.includes("手动设置"));
    expect(manualItem).toBeTruthy();
    await click(manualItem as HTMLElement);
    expect(container.textContent).toContain("创建已安排任务");
    expect(container.querySelector('input[placeholder*="每日项目进展"]')).toBeTruthy();
  });

  it("选择任务后加载详情并通过 typed gateway 暂停", async () => {
    const container = renderPage();
    await flushEffects();

    await click(findButton(container, "每日项目简报"));
    expect(mockRead).toHaveBeenCalledWith("task-daily");
    expect(mockListRuns).toHaveBeenCalledWith("task-daily");
    expect(container.textContent).toContain("整理项目进展、阻塞项和下一步行动。");
    expect(container.textContent).toContain("运行记录");

    await click(findButton(container, "暂停"));
    expect(mockSetEnabled).toHaveBeenCalledWith("task-daily", false);
    expect(container.textContent).toContain("恢复");
  });

  it("仅用运行返回的 sessionId 恢复 Agent 对话", async () => {
    const onNavigate = vi.fn<(page: Page, params?: PageParams) => void>();
    const container = renderPage({ onNavigate });
    await flushEffects();
    await click(findButton(container, "每日项目简报"));

    const openRun = container.querySelector<HTMLButtonElement>(
      'button[aria-label="打开运行对话"]',
    );
    expect(openRun).toBeTruthy();
    await click(openRun as HTMLButtonElement);

    expect(onNavigate).toHaveBeenCalledWith("agent", {
      agentEntry: "claw",
      projectId: "project-alpha",
      initialSessionId: "session-scheduled-run",
      initialSessionName: "每日项目简报",
    });
  });
});
