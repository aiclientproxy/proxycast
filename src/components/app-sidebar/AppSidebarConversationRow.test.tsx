import React, { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentSessionInfo } from "@/lib/api/agentRuntime/sessionTypes";
import { AppSidebarConversationRow } from "./AppSidebarConversationRow";

function renderRow(
  props: Partial<React.ComponentProps<typeof AppSidebarConversationRow>> = {},
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root: Root = createRoot(container);
  const session: AgentSessionInfo = {
    id: "session-1",
    name: "运行中会话",
    created_at: 1780847000,
    updated_at: 1780847600,
    messages_count: 2,
  };

  act(() => {
    root.render(
      <AppSidebarConversationRow
        session={session}
        title="运行中会话"
        meta="刚刚"
        active={false}
        actionDisabled={false}
        moreActionsLabel="更多操作"
        openActionMenuLabel="打开操作菜单"
        onNavigate={vi.fn()}
        onOpenMenu={vi.fn()}
        {...props}
      />,
    );
  });

  return { container, root };
}

describe("AppSidebarConversationRow", () => {
  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("运行中会话应显示状态图标和可访问标签", () => {
    const { container, root } = renderRow({
      runtimeStatus: "running",
      runtimeStatusLabel: "正在输出",
    });

    const status = container.querySelector(
      '[data-testid="app-sidebar-conversation-runtime-status"]',
    );
    expect(status?.getAttribute("aria-label")).toBe("正在输出");
    expect(status?.getAttribute("data-status")).toBe("running");

    act(() => root.unmount());
  });

  it("终态会话不传 runtimeStatus 时应保留普通会话入口", () => {
    const { container, root } = renderRow();

    expect(
      container.querySelector(
        '[data-testid="app-sidebar-conversation-runtime-status"]',
      ),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="app-sidebar-conversation-open"]'),
    ).not.toBeNull();
    expect(
      container
        .querySelector('[data-testid="app-sidebar-conversation-open"]')
        ?.getAttribute("data-session-id"),
    ).toBe("session-1");

    act(() => root.unmount());
  });
});
