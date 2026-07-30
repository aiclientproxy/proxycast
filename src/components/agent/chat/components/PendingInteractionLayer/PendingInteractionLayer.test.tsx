import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale, getLimeI18n } from "@/i18n/createI18n";
import { SUPPORTED_LOCALES } from "@/i18n/locales";
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

function permissionsInteraction(): TypedPendingInteraction {
  return {
    id: "permissions_approval:thread-1:turn-1:item-permissions-1",
    thread_id: "thread-1",
    turn_id: "turn-1",
    item_id: "item-permissions-1",
    kind: "permissions_approval",
    status: "pending",
    payload: {
      cwd: "/workspace/lime",
      environmentId: "local-dev",
      reason: "生成构建产物并读取项目配置",
      permissions: {
        network: { enabled: true },
        fileSystem: {
          read: ["/workspace/lime/config"],
          write: ["/workspace/lime/generated"],
          globScanMaxDepth: 2,
          entries: [
            {
              access: "write",
              path: {
                type: "special",
                value: { kind: "project_roots", subpath: "generated" },
              },
            },
            {
              access: "deny",
              path: { type: "glob_pattern", pattern: "**/*.env" },
            },
          ],
        },
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
  it("五种支持语言都提供完整的 permission approval 文案", async () => {
    const i18n = getLimeI18n();
    for (const locale of SUPPORTED_LOCALES) {
      await changeLimeLocale(locale);
      for (const key of [
        "agentChat.permissionsApproval.title",
        "agentChat.permissionsApproval.cwd",
        "agentChat.permissionsApproval.environment",
        "agentChat.permissionsApproval.network",
        "agentChat.permissionsApproval.fileSystem",
        "agentChat.permissionsApproval.action.grantTurn",
        "agentChat.permissionsApproval.action.grantSession",
        "agentChat.permissionsApproval.action.decline",
      ]) {
        expect(i18n.t(key, { ns: "agent" })).not.toBe(key);
      }
    }
  });

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

  it("permissions approval 在同一 Layer 展示 cwd、reason、environment 与完整权限 diff", () => {
    const { container } = renderLayer([permissionsInteraction()]);

    expect(
      container.querySelector('[data-testid="permissions-approval-panel"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="permissions-approval-reason"]')
        ?.textContent,
    ).toContain("生成构建产物并读取项目配置");
    expect(
      container.querySelector('[data-testid="permissions-approval-cwd"]')
        ?.textContent,
    ).toContain("/workspace/lime");
    expect(
      container.querySelector(
        '[data-testid="permissions-approval-environment"]',
      )?.textContent,
    ).toContain("local-dev");
    expect(
      container.querySelector('[data-testid="permissions-approval-network"]')
        ?.textContent,
    ).toContain("允许网络访问");

    const fileSystemDiff = container.querySelector(
      '[data-testid="permissions-approval-file-system"]',
    )?.textContent;
    expect(fileSystemDiff).toContain("/workspace/lime/config");
    expect(fileSystemDiff).toContain("/workspace/lime/generated");
    expect(fileSystemDiff).toContain("Glob 扫描深度: 2");
    expect(fileSystemDiff).toContain("$PROJECT_ROOTS/generated");
    expect(fileSystemDiff).toContain("**/*.env");
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
  });

  it.each([
    ["grant_turn", "本回合允许"],
    ["grant_session", "本会话允许"],
    ["decline", "拒绝"],
  ] as const)(
    "permissions approval 将 %s 精确提交一次",
    async (decision, label) => {
      const onRespond = vi.fn(() => ({ accepted: true }) as const);
      const { container } = renderLayer([permissionsInteraction()], onRespond);
      const button = container.querySelector(
        `button[data-permission-decision="${decision}"]`,
      ) as HTMLButtonElement;

      expect(button.textContent).toContain(label);
      await act(async () => {
        button.click();
        button.click();
        await Promise.resolve();
      });

      expect(onRespond).toHaveBeenCalledTimes(1);
      expect(onRespond).toHaveBeenCalledWith({
        decision,
        interactionId:
          "permissions_approval:thread-1:turn-1:item-permissions-1",
        kind: "permissions_approval",
      });
      expect(
        [...container.querySelectorAll("button")].every(
          (candidate) => candidate.disabled,
        ),
      ).toBe(true);
    },
  );
});
