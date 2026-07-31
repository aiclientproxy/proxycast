import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  AssistantFirstTokenRuntimeStatus,
  AssistantStreamingInlineIndicator,
  MessageRuntimeStatusPill,
} from "./MessageListRuntimeStatus";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      if (key === "agentChat.messageList.firstTokenStatus.context.title") {
        return "正在准备回复";
      }
      if (key === "agentChat.messageList.firstTokenStatus.context.detail") {
        return "正在整理会话信息和可用资料。";
      }
      if (key === "agentChat.messageList.streamingInline.queued") {
        return "等待输出";
      }
      if (key === "agentChat.messageList.streamingInline.running") {
        return "正在输出";
      }
      if (key === "agentChat.messageList.streamingInline.synthesizing") {
        return "正在整理最终答复";
      }
      return key;
    },
  }),
}));

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) {
      continue;
    }
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.clearAllMocks();
});

function renderRuntimeStatusPill() {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(
      <MessageRuntimeStatusPill
        status={{
          phase: "routing",
          title: "处理中",
          detail:
            "若回复加入团队将带来巨大技术突破，请补充目标、边界、风险和最终交付口径。",
          checkpoints: ["确认当前任务目标", "等待下一条进展"],
        }}
      />,
    );
  });

  mountedRoots.push({ root, container });
  return container;
}

function renderFirstTokenRuntimeStatus() {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(
      <AssistantFirstTokenRuntimeStatus
        status={{
          phase: "context",
          title: "旧标题不应直接使用",
          detail: "旧详情不应直接使用",
        }}
      />,
    );
  });

  mountedRoots.push({ root, container });
  return container;
}

function renderStreamingInlineIndicator(
  runtime: React.ComponentProps<
    typeof AssistantStreamingInlineIndicator
  >["runtime"],
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(<AssistantStreamingInlineIndicator runtime={runtime} />);
  });

  mountedRoots.push({ root, container });
  return container;
}

describe("MessageRuntimeStatusPill", () => {
  it("运行状态说明不应生成原生 title 浮窗", () => {
    const container = renderRuntimeStatusPill();
    const pill = container.querySelector(
      '[data-testid="message-runtime-status-pill"]',
    );

    expect(pill).not.toBeNull();
    expect(pill?.getAttribute("title")).toBeNull();
    expect(pill?.getAttribute("aria-label")).toContain("处理中");
    expect(pill?.getAttribute("aria-label")).toContain("等待下一条进展");
  });
});

describe("AssistantFirstTokenRuntimeStatus", () => {
  it("首字前等待态应保持轻量单行展示，并把说明放入 aria-label", () => {
    const container = renderFirstTokenRuntimeStatus();
    const indicator = container.querySelector(
      '[data-testid="assistant-first-token-runtime-status"]',
    );

    expect(indicator).not.toBeNull();
    expect(indicator?.getAttribute("data-status")).toBe("context");
    expect(indicator?.textContent).toContain("正在准备回复");
    expect(indicator?.textContent).not.toContain(
      "正在整理会话信息和可用资料。",
    );
    expect(indicator?.getAttribute("aria-label")).toContain("正在准备回复");
    expect(indicator?.getAttribute("aria-label")).toContain(
      "正在整理会话信息和可用资料。",
    );
    expect(indicator?.getAttribute("title")).toBeNull();
  });
});

describe("AssistantStreamingInlineIndicator", () => {
  it("运行中且工具批次已结束时，应提示正在整理最终答复", () => {
    const container = renderStreamingInlineIndicator({
      status: "running",
      latestRuntimePhase: null,
      detail: null,
      batchDescriptor: {
        kind: "web_search",
        title: "已搜索网页 3 次，读取网页 2 次",
        supportingLines: [],
        countLabel: "搜 3 / 读 2",
        rawDetailLabel: "展开查看搜索与读取来源",
        hasRunning: false,
      },
      pendingRequestCount: 0,
      subtaskStats: null,
      startedAt: null,
      completedAt: null,
    });

    const indicator = container.querySelector(
      '[data-testid="assistant-streaming-inline-indicator"]',
    );

    expect(indicator?.textContent).toContain("正在整理最终答复");
  });

  it("运行中且仍有工具批次在执行时，应保留正在输出提示", () => {
    const container = renderStreamingInlineIndicator({
      status: "running",
      latestRuntimePhase: null,
      detail: null,
      batchDescriptor: {
        kind: "web_search",
        title: "正在搜索网页 3 次，读取网页 2 次",
        supportingLines: [],
        countLabel: "搜 3 / 读 2",
        rawDetailLabel: "展开查看搜索与读取进度",
        hasRunning: true,
      },
      pendingRequestCount: 0,
      subtaskStats: null,
      startedAt: null,
      completedAt: null,
    });

    const indicator = container.querySelector(
      '[data-testid="assistant-streaming-inline-indicator"]',
    );

    expect(indicator?.textContent).toContain("正在输出");
  });
});
