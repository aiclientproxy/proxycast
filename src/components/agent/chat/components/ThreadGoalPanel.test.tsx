import { act, type ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { changeLimeLocale } from "@/i18n/createI18n";
import type { ThreadGoal } from "@limecloud/app-server-client";
import { ThreadGoalPanel } from "./ThreadGoalPanel";

const {
  clearThreadGoalMock,
  setThreadGoalMock,
  setThreadGoalStatusMock,
  toastMock,
} = vi.hoisted(() => ({
  clearThreadGoalMock: vi.fn(),
  setThreadGoalMock: vi.fn(),
  setThreadGoalStatusMock: vi.fn(),
  toastMock: { success: vi.fn(), error: vi.fn() },
}));

vi.mock("sonner", () => ({ toast: toastMock }));
vi.mock("@/lib/api/agentRuntime/threadGoalClient", () => ({
  clearThreadGoal: clearThreadGoalMock,
  setThreadGoal: setThreadGoalMock,
  setThreadGoalStatus: setThreadGoalStatusMock,
}));

interface MountedPanel {
  container: HTMLDivElement;
  root: Root;
}

const mountedPanels: MountedPanel[] = [];

function createGoal(status: ThreadGoal["status"] = "active"): ThreadGoal {
  return {
    createdAt: 10,
    objective: "完成 Codex ThreadGoal GUI 接入",
    status,
    threadId: "thread-1",
    timeUsedSeconds: 90,
    tokenBudget: 100_000,
    tokensUsed: 2_000,
    updatedAt: 20,
  };
}

function renderPanel(
  props: Partial<ComponentProps<typeof ThreadGoalPanel>> = {},
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(
      <ThreadGoalPanel
        threadId="thread-1"
        threadGoal={createGoal()}
        {...props}
      />,
    );
  });
  mountedPanels.push({ container, root });
  return container;
}

function setTextareaValue(textarea: HTMLTextAreaElement | null, value: string) {
  expect(textarea).toBeTruthy();
  const setter = Object.getOwnPropertyDescriptor(
    HTMLTextAreaElement.prototype,
    "value",
  )?.set;
  act(() => {
    setter?.call(textarea, value);
    textarea?.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

async function clickButton(button: HTMLButtonElement | null) {
  expect(button).toBeTruthy();
  await act(async () => {
    button?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await Promise.resolve();
  });
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  await changeLimeLocale("zh-CN");
  clearThreadGoalMock.mockReset();
  setThreadGoalMock.mockReset();
  setThreadGoalStatusMock.mockReset();
  toastMock.success.mockReset();
  toastMock.error.mockReset();
});

afterEach(() => {
  for (const mounted of mountedPanels.splice(0)) {
    act(() => mounted.root.unmount());
    mounted.container.remove();
  }
  document.body
    .querySelectorAll('[data-testid="modal-overlay"]')
    .forEach((node) => node.remove());
});

describe("ThreadGoalPanel", () => {
  it("只展示 thread identity 匹配的 canonical Goal", () => {
    const empty = renderPanel({ threadGoal: null });
    expect(
      empty.querySelector('[data-testid="thread-goal-empty"]'),
    ).not.toBeNull();

    const wrongThread = renderPanel({
      threadGoal: { ...createGoal(), threadId: "thread-other" },
    });
    expect(
      wrongThread.querySelector('[data-testid="thread-goal-panel"]'),
    ).toBeNull();
  });

  it("空态通过 thread/goal/set 创建 active Goal", async () => {
    setThreadGoalMock.mockResolvedValue(createGoal());
    const container = renderPanel({ threadGoal: null });

    setTextareaValue(
      container.querySelector('[data-testid="thread-goal-create-input"]'),
      "完成 canonical Goal 详情",
    );
    await clickButton(
      container.querySelector('[data-testid="thread-goal-create"]'),
    );

    expect(setThreadGoalMock).toHaveBeenCalledWith({
      threadId: "thread-1",
      objective: "完成 canonical Goal 详情",
      status: "active",
    });
    expect(
      container.querySelector('[data-testid="thread-goal-panel"]'),
    ).not.toBeNull();
  });

  it("detail 显式展示 loading 与 load error，inline 保持紧凑不占位", () => {
    const loading = renderPanel({
      threadGoal: null,
      threadGoalLoading: true,
    });
    expect(
      loading.querySelector('[data-testid="thread-goal-loading"]'),
    ).not.toBeNull();

    const failed = renderPanel({
      threadGoal: null,
      threadGoalError: new Error("goal load failed"),
    });
    expect(failed.textContent).toContain("goal load failed");
    expect(
      failed.querySelector('[data-testid="thread-goal-load-error"]'),
    ).not.toBeNull();

    const inline = renderPanel({
      threadGoal: null,
      threadGoalLoading: true,
      variant: "inline",
    });
    expect(inline.childElementCount).toBe(0);
  });

  it("detail 与 inline 形态消费同一 canonical Goal", () => {
    const detail = renderPanel();
    expect(detail.textContent).toContain("完成 Codex ThreadGoal GUI 接入");
    expect(
      detail.querySelector('[data-testid="thread-goal-panel"]'),
    ).not.toBeNull();

    const inline = renderPanel({ variant: "inline" });
    expect(inline.textContent).toContain("完成 Codex ThreadGoal GUI 接入");
    expect(
      inline.querySelector('[data-testid="thread-goal-inline-panel"]'),
    ).not.toBeNull();
    expect(
      detail.querySelector('[data-testid="thread-goal-tokens"]')?.textContent,
    ).toContain("Tokens /");
    expect(
      detail.querySelector('[data-testid="thread-goal-wall-time"]')
        ?.textContent,
    ).toContain("1 分 30 秒");
    expect(
      detail.querySelector('[data-testid="thread-goal-updated-at"]'),
    ).not.toBeNull();
  });

  it("通过 thread/goal/set 编辑 complete Goal 并重新激活", async () => {
    const completed = createGoal("complete");
    const updated = {
      ...completed,
      objective: "完成 canonical Goal GUI 接入",
      status: "active" as const,
    };
    setThreadGoalMock.mockResolvedValue(updated);
    const container = renderPanel({ threadGoal: completed });

    await clickButton(
      container.querySelector('[data-testid="thread-goal-edit"]'),
    );
    setTextareaValue(
      document.body.querySelector<HTMLTextAreaElement>(
        '[data-testid="thread-goal-objective-input"]',
      ),
      "完成 canonical Goal GUI 接入",
    );
    await clickButton(
      document.body.querySelector('[data-testid="thread-goal-save"]'),
    );

    expect(setThreadGoalMock).toHaveBeenCalledWith({
      threadId: "thread-1",
      objective: "完成 canonical Goal GUI 接入",
      status: "active",
    });
    expect(container.textContent).toContain("canonical Goal GUI");
  });

  it("暂停和清除只调用 canonical ThreadGoal methods", async () => {
    setThreadGoalStatusMock.mockResolvedValue(createGoal("paused"));
    clearThreadGoalMock.mockResolvedValue(true);
    const onGoalChanged = vi.fn();
    const container = renderPanel({ onGoalChanged });

    await clickButton(
      container.querySelector('[data-testid="thread-goal-pause"]'),
    );
    expect(setThreadGoalStatusMock).toHaveBeenCalledWith("thread-1", "paused");

    await clickButton(
      container.querySelector('[data-testid="thread-goal-clear"]'),
    );
    expect(clearThreadGoalMock).toHaveBeenCalledWith("thread-1");
    expect(onGoalChanged).toHaveBeenLastCalledWith(null);
  });

  it("complete 只调用 canonical status mutation，运行中禁用终态动作", async () => {
    setThreadGoalStatusMock.mockResolvedValue(createGoal("complete"));
    const container = renderPanel();

    await clickButton(
      container.querySelector('[data-testid="thread-goal-complete"]'),
    );
    expect(setThreadGoalStatusMock).toHaveBeenCalledWith(
      "thread-1",
      "complete",
    );

    const busy = renderPanel({ runtimeBusy: true });
    expect(
      busy.querySelector<HTMLButtonElement>(
        '[data-testid="thread-goal-complete"]',
      )?.disabled,
    ).toBe(true);
  });

  it("mutation 失败时保留 canonical Goal 并展示错误", async () => {
    setThreadGoalStatusMock.mockRejectedValue(new Error("goal update failed"));
    const container = renderPanel();

    await clickButton(
      container.querySelector('[data-testid="thread-goal-pause"]'),
    );

    expect(container.textContent).toContain("goal update failed");
    expect(toastMock.error).toHaveBeenCalledWith("goal update failed");
    expect(container.textContent).toContain("完成 Codex ThreadGoal GUI 接入");
  });

  it.each(["paused", "blocked", "usageLimited"] as const)(
    "%s Goal 支持恢复为 active",
    async (status) => {
      setThreadGoalStatusMock.mockResolvedValue(createGoal("active"));
      const container = renderPanel({ threadGoal: createGoal(status) });

      await clickButton(
        container.querySelector('[data-testid="thread-goal-resume"]'),
      );
      expect(setThreadGoalStatusMock).toHaveBeenCalledWith(
        "thread-1",
        "active",
      );
    },
  );
});
