import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import type { TypedPendingInteraction } from "@/lib/api/agentRuntime/pendingInteractionController";
import { PendingInteractionLayer } from "./PendingInteractionLayer";
import { selectActivePendingInteraction } from "./pendingInteractionSelection";

const mounted: Array<{ container: HTMLDivElement; root: Root }> = [];

beforeEach(async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  await changeLimeLocale("zh-CN");
});

afterEach(() => {
  for (const entry of mounted.splice(0)) {
    act(() => entry.root.unmount());
    entry.container.remove();
  }
  vi.unstubAllGlobals();
});

function mcpInteraction(
  id = "mcp_elicitation:thread-1:turn-1:1",
): TypedPendingInteraction {
  return {
    id,
    thread_id: "thread-1",
    turn_id: "turn-1",
    kind: "mcp_elicitation",
    status: "pending",
    payload: {
      message: "请确认发布参数",
      requestedSchema: {
        type: "object",
        properties: {
          environment: {
            type: "string",
            enum: ["staging", "production"],
          },
        },
        required: ["environment"],
      },
      serverName: "release-tools",
    },
  };
}

function approvalInteraction(): TypedPendingInteraction {
  return {
    id: "approval:thread-1:turn-1:approval-1",
    thread_id: "thread-1",
    turn_id: "turn-1",
    kind: "approval",
    status: "pending",
    payload: {
      request: {
        requestId: "approval-1",
        actionType: "tool_confirmation",
        prompt: "允许执行测试？",
        availableDecisions: ["allow_once", "decline"],
        status: "pending",
      },
    },
  };
}

function renderLayer(
  interactions: readonly TypedPendingInteraction[],
  onRespond = vi.fn(() => ({ accepted: true }) as const),
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(
      <PendingInteractionLayer
        interactions={interactions}
        threadId="thread-1"
        onRespond={onRespond}
      />,
    );
  });
  mounted.push({ container, root });
  return { container, onRespond };
}

describe("PendingInteractionLayer", () => {
  it("只选择当前 thread 的第一条 pending interaction", () => {
    const other = { ...mcpInteraction("other"), thread_id: "thread-2" };
    const resolved = { ...approvalInteraction(), status: "resolved" as const };
    const active = mcpInteraction();

    expect(
      selectActivePendingInteraction([other, resolved, active], "thread-1"),
    ).toBe(active);
  });

  it("MCP 使用 Composer 内纯表单且不创建根部 Dialog", () => {
    const { container } = renderLayer([mcpInteraction()]);

    expect(
      container.querySelector('[data-testid="pending-interaction-layer"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="mcp-server-elicitation-form"]'),
    ).not.toBeNull();
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
    expect(container.textContent).toContain("请确认发布参数");
  });

  it("MCP 结构化值通过 semantic interaction id 提交", async () => {
    const onRespond = vi.fn(() => ({ accepted: true }) as const);
    const { container } = renderLayer([mcpInteraction()], onRespond);
    const select = container.querySelector("select") as HTMLSelectElement;
    const submit = [...container.querySelectorAll("button")].find((button) =>
      button.textContent?.includes("提交"),
    );

    act(() => {
      const setter = Object.getOwnPropertyDescriptor(
        HTMLSelectElement.prototype,
        "value",
      )?.set;
      setter?.call(select, "production");
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await act(async () => {
      submit?.click();
      await Promise.resolve();
    });

    expect(onRespond).toHaveBeenCalledWith({
      action: "accept",
      content: { environment: "production" },
      interactionId: "mcp_elicitation:thread-1:turn-1:1",
      kind: "mcp_elicitation",
    });
    expect((submit as HTMLButtonElement).disabled).toBe(true);
  });

  it("approval 也由同一 Layer 投影到 Composer replacement", async () => {
    const onRespond = vi.fn(() => ({ accepted: true }) as const);
    const { container } = renderLayer([approvalInteraction()], onRespond);
    const allow = container.querySelector(
      'button[data-decision="allow_once"]',
    ) as HTMLButtonElement;

    await act(async () => {
      allow.click();
      await Promise.resolve();
    });

    expect(onRespond).toHaveBeenCalledWith({
      interactionId: "approval:thread-1:turn-1:approval-1",
      kind: "approval",
      response: expect.objectContaining({
        requestId: "approval-1",
        decision: "allow_once",
      }),
    });
  });
});
