import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale, getLimeI18n } from "@/i18n/createI18n";
import { SUPPORTED_LOCALES } from "@/i18n/locales";
import {
  validateMcpElicitationFormContent,
  type PendingMcpServerElicitation,
} from "@/lib/api/mcpServerElicitation";
import {
  McpServerElicitationForm,
  type McpServerElicitationFormSubmission,
} from "./McpServerElicitationForm";

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

function request(
  requestedSchema: Record<string, unknown> = {
    type: "object",
    properties: {
      environment: {
        type: "string",
        title: "环境",
        enum: ["staging", "production"],
      },
      retries: {
        type: "integer",
        title: "重试次数",
        minimum: 0,
        maximum: 3,
      },
      confirmed: { type: "boolean", title: "确认发布" },
    },
    required: ["environment", "retries", "confirmed"],
  },
): PendingMcpServerElicitation {
  return {
    key: "mcp_elicitation:thread-1:turn-1:1",
    params: {
      message: "请确认发布参数",
      mode: "form",
      requestedSchema,
      serverName: "release-tools",
      threadId: "thread-1",
      turnId: "turn-1",
    },
  };
}

function renderForm(input = request()) {
  const submissions: McpServerElicitationFormSubmission[] = [];
  const onSubmit = vi.fn((submission: McpServerElicitationFormSubmission) => {
    if (submission.action === "accept") {
      const issues = validateMcpElicitationFormContent(
        input.params.requestedSchema,
        submission.content,
      );
      if (issues.length > 0) {
        return { accepted: false as const, issues };
      }
    }
    submissions.push(submission);
    return { accepted: true as const };
  });
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(
      <McpServerElicitationForm request={input} onSubmit={onSubmit} />,
    );
  });
  mounted.push({ container, root });
  return { container, onSubmit, submissions };
}

function button(container: HTMLElement, text: string): HTMLButtonElement {
  const match = [...container.querySelectorAll("button")].find((candidate) =>
    candidate.textContent?.includes(text),
  );
  if (!match) throw new Error(`button missing: ${text}`);
  return match;
}

function change(element: HTMLInputElement | HTMLSelectElement, value: string) {
  act(() => {
    const setter = Object.getOwnPropertyDescriptor(
      Object.getPrototypeOf(element),
      "value",
    )?.set;
    setter?.call(element, value);
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

describe("McpServerElicitationForm", () => {
  it("五种支持语言都提供稳定的表单文案", async () => {
    const i18n = getLimeI18n();
    for (const locale of SUPPORTED_LOCALES) {
      await changeLimeLocale(locale);
      for (const key of [
        "agentChat.mcpElicitation.title",
        "agentChat.mcpElicitation.action.accept",
        "agentChat.mcpElicitation.action.decline",
        "agentChat.mcpElicitation.action.cancel",
        "agentChat.mcpElicitation.validation.missing_required",
      ]) {
        expect(i18n.t(key, { ns: "agent" })).not.toBe(key);
      }
    }
  });

  it("按 schema key 提交 enum、integer 与 boolean，且提交后保持禁用", async () => {
    const { container, onSubmit, submissions } = renderForm();
    const select = container.querySelector("select") as HTMLSelectElement;
    const number = container.querySelector(
      'input[type="number"]',
    ) as HTMLInputElement;
    const checkbox = container.querySelector(
      'input[type="checkbox"]',
    ) as HTMLInputElement;
    change(select, "production");
    change(number, "2");
    act(() => checkbox.click());

    const submit = button(container, "提交");
    await act(async () => {
      submit.click();
      await Promise.resolve();
    });

    expect(submissions).toEqual([
      {
        action: "accept",
        content: { environment: "production", retries: 2, confirmed: true },
      },
    ]);
    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(submit.disabled).toBe(true);
  });

  it("required 与 integer 校验失败时保持可编辑", async () => {
    const { container, submissions } = renderForm();
    change(
      container.querySelector('input[type="number"]') as HTMLInputElement,
      "1.5",
    );
    const submit = button(container, "提交");

    await act(async () => {
      submit.click();
      await Promise.resolve();
    });

    expect(container.textContent).toContain("此项为必填项");
    expect(container.textContent).toContain("请输入整数");
    expect(submissions).toEqual([]);
    expect(submit.disabled).toBe(false);
  });

  it("optional boolean 经用户操作后可以明确提交 false", async () => {
    const { container, submissions } = renderForm(
      request({
        type: "object",
        properties: { optional: { type: "boolean" } },
      }),
    );
    const checkbox = container.querySelector(
      'input[type="checkbox"]',
    ) as HTMLInputElement;
    act(() => {
      checkbox.click();
      checkbox.click();
    });
    await act(async () => {
      button(container, "提交").click();
      await Promise.resolve();
    });

    expect(submissions).toEqual([
      { action: "accept", content: { optional: false } },
    ]);
  });

  it("datetime-local 转为 RFC3339，拒绝和取消保持独立 action", async () => {
    const dateHarness = renderForm(
      request({
        type: "object",
        properties: {
          scheduledAt: { type: "string", format: "date-time" },
        },
        required: ["scheduledAt"],
      }),
    );
    change(
      dateHarness.container.querySelector(
        'input[type="datetime-local"]',
      ) as HTMLInputElement,
      "2026-07-13T11:00:00",
    );
    await act(async () => {
      button(dateHarness.container, "提交").click();
      await Promise.resolve();
    });
    const accepted = dateHarness.submissions[0] as Extract<
      McpServerElicitationFormSubmission,
      { action: "accept" }
    >;
    expect(accepted.content.scheduledAt).toMatch(
      /^2026-07-13T\d{2}:00:00\.000Z$/,
    );

    const declineHarness = renderForm();
    await act(async () => {
      button(declineHarness.container, "拒绝").click();
      await Promise.resolve();
    });
    expect(declineHarness.submissions).toEqual([{ action: "decline" }]);

    const cancelHarness = renderForm();
    await act(async () => {
      button(cancelHarness.container, "取消").click();
      await Promise.resolve();
    });
    expect(cancelHarness.submissions).toEqual([{ action: "cancel" }]);
  });
});
