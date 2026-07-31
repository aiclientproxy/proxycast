import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TaskCenterUtilityToolbar } from "./TaskCenterUtilityToolbar";

(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;

const { mockOpenProjectPathWithTool, mockReadProjectGitStatus } = vi.hoisted(
  () => ({
    mockOpenProjectPathWithTool: vi.fn(),
    mockReadProjectGitStatus: vi.fn(),
  }),
);

vi.mock("@/lib/api/fileSystem", () => ({
  openProjectPathWithTool: mockOpenProjectPathWithTool,
}));

vi.mock("@/lib/api/projectGit", () => ({
  readProjectGitStatus: mockReadProjectGitStatus,
}));

vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
  },
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
  }) => {
    const context = React.useContext(PopoverTestContext);
    return context?.open ? <div {...props}>{children}</div> : null;
  },
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

const mountedRoots: Array<{ container: HTMLDivElement; root: Root }> = [];

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

function renderToolbar(
  props?: Partial<React.ComponentProps<typeof TaskCenterUtilityToolbar>>,
) {
  mockReadProjectGitStatus.mockResolvedValue({
    hasGitRepository: false,
    currentBranch: null,
    uncommittedFileCount: 0,
  });

  return mount(buildToolbar(props));
}

function buildToolbar(
  props?: Partial<React.ComponentProps<typeof TaskCenterUtilityToolbar>>,
) {
  return (
    <TaskCenterUtilityToolbar
      projectRootPath="/tmp/project"
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
    />
  );
}

function rerenderToolbar(
  props?: Partial<React.ComponentProps<typeof TaskCenterUtilityToolbar>>,
) {
  const mounted = mountedRoots.at(-1);
  if (!mounted) {
    throw new Error("toolbar is not mounted");
  }
  act(() => {
    mounted.root.render(buildToolbar(props));
  });
}

async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
  });
}

describe("TaskCenterUtilityToolbar plan rail reveal", () => {
  it("有 revisioned plan item 时应自动揭示计划轨", async () => {
    const container = renderToolbar({
      taskRail: {
        workflowSteps: [],
        messages: [],
        threadItems: [
          {
            id: "plan-restore",
            type: "plan",
            thread_id: "thread-1",
            turn_id: "turn-1",
            sequence: 1,
            status: "in_progress",
            text: "- [x] 读取任务区域\n- [ ] 恢复运行计划",
            metadata: {
              revisionId: "proposed_plan:task-rail-2",
            },
            started_at: "2026-06-16T10:00:02.000Z",
            updated_at: "2026-06-16T10:00:03.000Z",
          },
        ],
      },
    });

    await flushEffects();

    const popover = container.querySelector(
      '[data-testid="task-center-environment-popover"]',
    );
    const planSection = container.querySelector(
      '[data-testid="task-center-run-control-plan"]',
    );
    const planRevision = container.querySelector(
      '[data-testid="task-center-run-control-plan-revision"]',
    );
    const planItems = Array.from(
      container.querySelectorAll(
        '[data-testid="task-center-run-control-plan-item"]',
      ),
    );

    expect(popover).not.toBeNull();
    expect(planSection?.textContent).toContain("读取任务区域");
    expect(planSection?.textContent).toContain("恢复运行计划");
    expect(planRevision?.getAttribute("data-plan-revision-id")).toBe(
      "proposed_plan:task-rail-2",
    );
    expect(planItems.map((item) => item.getAttribute("data-status"))).toEqual([
      "completed",
      "running",
    ]);
  });

  it("实时 todo checklist 到达时应自动揭示计划轨", async () => {
    const container = renderToolbar({
      taskRail: {
        workflowSteps: [],
        messages: [],
      },
    });

    await flushEffects();
    expect(
      container.querySelector(
        '[data-testid="task-center-environment-popover"]',
      ),
    ).toBeNull();

    rerenderToolbar({
      taskRail: {
        workflowSteps: [],
        messages: [],
        todoItems: [
          { content: "读取 current owner", status: "completed" },
          { content: "验证实时计划", status: "in_progress" },
          { content: "记录 Gate B", status: "pending" },
        ],
      },
    });
    await flushEffects();

    const planSection = container.querySelector(
      '[data-testid="task-center-run-control-plan"]',
    );
    const planItems = Array.from(
      container.querySelectorAll(
        '[data-testid="task-center-run-control-plan-item"]',
      ),
    );
    expect(planSection?.textContent).toContain("读取 current owner");
    expect(planSection?.textContent).toContain("验证实时计划");
    expect(planSection?.textContent).toContain("记录 Gate B");
    expect(planItems.map((item) => item.getAttribute("data-status"))).toEqual([
      "completed",
      "running",
      "pending",
    ]);
  });

  it("没有计划项时不应自动打开环境弹窗", async () => {
    const container = renderToolbar({
      taskRail: {
        workflowSteps: [],
        messages: [],
      },
    });

    await flushEffects();

    expect(
      container.querySelector(
        '[data-testid="task-center-environment-popover"]',
      ),
    ).toBeNull();
  });
});
