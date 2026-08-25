import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TaskCenterEnvironmentPanel } from "./TaskCenterEnvironmentPanel";

vi.mock("@/components/ui/popover", () => ({
  Popover: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  PopoverContent: ({
    children,
    align: _align,
    sideOffset: _sideOffset,
    ...props
  }: React.HTMLAttributes<HTMLDivElement> & {
    align?: string;
    sideOffset?: number;
  }) => <div {...props}>{children}</div>,
  PopoverTrigger: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
}));

describe("TaskCenterEnvironmentPanel", () => {
  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("展示当前 Environment 的断线状态和 current protocol identity", async () => {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    const translate = (
      _key: string,
      options?: Record<string, unknown>,
    ): string => String(options?.defaultValue ?? "");

    try {
      await act(async () => {
        root.render(
          <TaskCenterEnvironmentPanel
            normalizedProjectRootPath="/workspace"
            status={null}
            environmentStatusLabel="非 Git 项目"
            lifecycleStatuses={[
              { environmentId: "local", status: "connected" },
              { environmentId: "remote-build", status: "disconnected" },
            ]}
            branchLabel="无分支"
            changeCount={0}
            translate={translate}
          />,
        );
      });

      const remote = container.querySelector(
        '[data-testid="task-center-environment-runtime"]',
      );
      expect(remote?.textContent).toContain("remote-build");
      expect(remote?.textContent).toContain("连接已断开");
      expect(remote?.getAttribute("data-environment-status")).toBe(
        "disconnected",
      );
      expect(remote?.getAttribute("data-protocol-method")).toBe(
        "thread/environment/disconnected",
      );
      expect(
        container.querySelector('[data-testid="task-center-environment-local"]')
          ?.textContent,
      ).toContain("已连接");
    } finally {
      await act(async () => root.unmount());
      container.remove();
    }
  });
});
