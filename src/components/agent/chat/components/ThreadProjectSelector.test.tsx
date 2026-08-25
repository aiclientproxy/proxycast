import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ThreadProjectSelector } from "./ThreadProjectSelector";

const { assign, createAndAssign, refresh, mockDirectory } = vi.hoisted(() => ({
  assign: vi.fn(),
  createAndAssign: vi.fn(),
  refresh: vi.fn(),
  mockDirectory: vi.fn(),
}));

vi.mock("../hooks/useThreadProjectDirectory", () => ({
  useThreadProjectDirectory: mockDirectory,
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue: string, values?: { project?: string }) =>
      defaultValue.replace("{{project}}", values?.project ?? ""),
  }),
}));

vi.mock("@/components/ui/popover", () => ({
  Popover: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  PopoverTrigger: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
  PopoverContent: ({
    children,
    align: _align,
    side: _side,
    sideOffset: _sideOffset,
    ...props
  }: React.HTMLAttributes<HTMLDivElement> & {
    align?: string;
    side?: string;
    sideOffset?: number;
  }) => <div {...props}>{children}</div>,
}));

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  assign.mockResolvedValue(undefined);
  createAndAssign.mockResolvedValue(undefined);
  refresh.mockResolvedValue(undefined);
  mockDirectory.mockReturnValue({
    assign,
    createAndAssign,
    error: null,
    loading: false,
    mutating: false,
    projectId: "project-1",
    projects: [
      {
        id: "project-1",
        name: "Lime Runtime",
        roots: [{ path: "/workspace/lime" }],
        metadata: {},
        position: 0,
        createdAt: 1,
        updatedAt: 1,
      },
      {
        id: "project-2",
        name: "Desktop Shell",
        roots: [{ path: "/workspace/desktop" }],
        metadata: {},
        position: 1,
        createdAt: 1,
        updatedAt: 1,
      },
    ],
    refresh,
  });
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("ThreadProjectSelector", () => {
  it("应显示当前 Thread 归属并允许切换或清空", async () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    try {
      await act(async () => {
        root.render(<ThreadProjectSelector threadId="thread-1" />);
      });

      const trigger = container.querySelector(
        '[data-testid="thread-project-selector"]',
      );
      expect(trigger?.textContent).toContain("Lime Runtime");
      expect(trigger?.getAttribute("data-thread-project-id")).toBe("project-1");

      await act(async () => {
        (
          container.querySelector(
            'button[data-project-id="project-2"]',
          ) as HTMLButtonElement
        ).click();
        await Promise.resolve();
      });
      expect(assign).toHaveBeenCalledWith("project-2");

      await act(async () => {
        (
          container.querySelector(
            'button[data-project-id=""]',
          ) as HTMLButtonElement
        ).click();
        await Promise.resolve();
      });
      expect(assign).toHaveBeenCalledWith(null);
    } finally {
      await act(async () => root.unmount());
    }
  });

  it("目录为空时应提供把当前工作区加入 Project 的下一步", async () => {
    mockDirectory.mockReturnValue({
      ...mockDirectory(),
      projectId: null,
      projects: [],
    });
    const container = document.createElement("div");
    const root = createRoot(container);
    try {
      await act(async () => {
        root.render(
          <ThreadProjectSelector
            threadId="thread-1"
            workspaceName="Lime"
            workspaceRootPath="/workspace/lime"
          />,
        );
      });
      expect(container.textContent).toContain("项目目录为空");

      await act(async () => {
        (
          container.querySelector(
            '[data-testid="thread-project-create-from-workspace"]',
          ) as HTMLButtonElement
        ).click();
        await Promise.resolve();
      });
      expect(createAndAssign).toHaveBeenCalledWith({
        name: "Lime",
        rootPath: "/workspace/lime",
      });
    } finally {
      await act(async () => root.unmount());
    }
  });

  it("没有 canonical Thread 时不应渲染入口", async () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    try {
      await act(async () => {
        root.render(<ThreadProjectSelector threadId={null} />);
      });
      expect(container.innerHTML).toBe("");
    } finally {
      await act(async () => root.unmount());
    }
  });
});
